"""Stage A — the polite import, as an Apache Beam pipeline.

    Create(items)
      -> filter against the `.done` markers (side input, taken ONCE at start)
      -> key by (host, lane)
      -> GroupByKey                 <-- this is what bounds concurrency
      -> ParDo(LaneWorker)          <-- one lane = one polite sequential stream
      -> WriteToText(report.jsonl)

Beam is the ORCHESTRATOR, not the accelerator. GroupByKey guarantees that
every item sharing a key reaches ONE worker as ONE iterable, so the number of
keys per host is the maximum number of simultaneous connections that host
will ever see from us — whatever the runner decides about machines and
threads. The same guarantee holds in Flume; only the worker count and the
sink change.

Four statuses, and there is no fifth (DESIGN §2):

    written   the shard was written, read back, checksummed, and marked
    present   a `.done` marker already said so; nothing was fetched
    queued    not done — in `retry_queue.jsonl`, for the next run
    absent    the archive said 404/410 twice, ≥ 6 h apart, evidence on disk

Run it:

    python -m beam_import.pipeline --tiers 0 --output /data/import \\
        --state-dir /var/tmp/beam_import --report-dir out/tier0 \\
        --runner DirectRunner --direct_running_mode multi_processing \\
        --direct_num_workers 8

Credentials are read from the ENVIRONMENT inside `DoFn.setup()`. They are
never pipeline options and never argv: options are logged and shown in job UIs.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import traceback
from typing import Any, Dict, Iterable, List, Optional, Tuple

import apache_beam as beam
from apache_beam.options.pipeline_options import PipelineOptions, SetupOptions

from . import fetchers, manifest, registry, sinks, transforms
from .hosts import (BlockedError, CircuitOpen, LaneState, NotFound,
                    PermanentError, TransientError, lane_states)

STATUSES = ("written", "present", "queued", "absent")


def _utcnow() -> str:
    return sinks.utcnow()


def progress_path(state_dir: str, host: str, lane: int) -> str:
    """Where a lane appends its results as it goes — one file per lane, so
    two workers never write to the same file and nothing has to be locked."""
    return os.path.join(state_dir, "progress", f"{host}-{lane}.jsonl")


def append_progress(path: str, record: Dict[str, Any]) -> None:
    """Append one record, flush, fsync.

    ml/CLAUDE.md §5.25: a computation longer than half an hour writes its
    result incrementally. Beam only writes report.jsonl when the whole
    pipeline finishes, so a run killed at hour eight would otherwise have
    nothing to show for itself.
    """
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(record, sort_keys=True) + "\n")
        fh.flush()
        os.fsync(fh.fileno())


def _result(item: Dict[str, Any], status: str, **kw: Any) -> Dict[str, Any]:
    assert status in STATUSES, status
    rec = {
        "item_id": item["item_id"], "source": item["source"],
        "host": item["host"], "lane": item["lane"],
        "tier": int(item.get("tier", 0)), "status": status,
        "bytes": 0, "sha256": None, "shard": None, "n_records": 0,
        "missing_dates": [], "seconds": 0.0, "reason": None, "at": None,
    }
    rec.update(kw)
    return rec


def _safe(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", name)


# --------------------------------------------------------------------------
# the worker
# --------------------------------------------------------------------------
class LaneWorker(beam.DoFn):
    """Processes ONE lane: all the items of one (host, lane) key, in order.

    Everything transient is caught and turned into a result record plus a
    queue entry. Only a programming error is allowed to propagate — Beam
    RETRIES a failed bundle (Dataflow four times), and a retried bundle would
    re-download everything the lane had already done, which is the opposite
    of polite.
    """

    def __init__(self, cfg: Dict[str, Any]) -> None:
        super().__init__()
        self.cfg = cfg              # plain dict: picklable, no clients in it

    def setup(self) -> None:
        """Clients are created HERE, once per worker, never in __init__."""
        self.reg = registry.load(self.cfg["registry"])
        self.lanes = lane_states(self.reg)

    # -- the durable, write-then-yield wrapper -----------------------------
    def process(self, element):
        (host, lane_no), _items = element
        lane = self.lanes[host]
        path = progress_path(self.cfg["state_dir"], host, lane_no)
        for rec in self._run_lane(element):
            rec["at"] = _utcnow()
            rec["backoffs_so_far"] = int(lane.counters["backoffs"])
            rec["trips_so_far"] = int(lane.counters["trips"])
            append_progress(path, rec)
            yield rec

    # -- the lane itself ---------------------------------------------------
    def _run_lane(self, element):
        (host, lane_no), items = element
        lane = self.lanes[host]
        state_dir = self.cfg["state_dir"]
        output = self.cfg["output"]

        def note(msg: str) -> None:
            print(f"[{host}/{lane_no}] {msg}", flush=True)

        import time
        for item in items:                       # lazy: one item at a time
            if lane.open:
                sinks.enqueue(state_dir, [item], "circuit breaker")
                yield _result(item, "queued",
                              reason="lane stopped by the circuit breaker")
                continue

            t0 = time.time()
            workdir = os.path.join(state_dir, "work", _safe(item["source"]),
                                   _safe(item["item_id"]))
            try:
                # 1. --dry-run shows the plan and touches nothing.
                if self.cfg["dry_run"]:
                    yield _result(item, "present",
                                  reason="dry run: would fetch "
                                         + (item["urls"][0] if item["urls"]
                                            else str(item.get("dataset_id"))))
                    continue

                # 2. skip before fetch — the marker is checked per item,
                #    because a second run or a retried bundle must not
                #    re-download what is already written.
                marker = sinks.is_done(output, item["item_id"])
                if marker:
                    yield _result(item, "present", bytes=marker.get("bytes", 0),
                                  sha256=marker.get("sha256"),
                                  shard=marker.get("shard"),
                                  n_records=marker.get("n_records", 0),
                                  missing_dates=marker.get("missing_dates", []))
                    lane.record_success()
                    continue

                got = fetchers.fetch_item(item, lane, workdir, note=note)
                payloads, shas, dates = transforms.to_examples(
                    item, got["paths"], got.get("prov", {}), note=note)
                if not payloads:
                    raise TransientError(
                        f"{item['item_id']}: fetched {len(got['paths'])} "
                        "file(s) and produced no records")

                marker = sinks.write_verify_mark(
                    output, item["item_id"], item["source"], payloads, shas,
                    fill=item.get("fill_token"),
                    fill_parent=item.get("fill_parent"),
                    missing_dates=got.get("missing_dates"),
                    extra={"dates": [dates[0], dates[-1]],
                           "wire_bytes": got.get("bytes", 0)},
                    note=note)
                lane.record_success()

                # Every day the archive did not serve goes back on the queue
                # as its own item. Stage A loses nothing; only the
                # two-sighting rule ever calls a day absent.
                missing = got.get("missing_dates") or []
                if missing and not item.get("day"):
                    sinks.enqueue(state_dir,
                                  [manifest.day_item(item, d) for d in missing],
                                  "missing day, re-queued for a later run")
                    note(f"{item['item_id']}: {len(missing)} missing day(s) "
                         "queued as day-level items")

                yield _result(item, "written", bytes=marker["bytes"],
                              sha256=marker["sha256"], shard=marker["shard"],
                              n_records=marker["n_records"],
                              missing_dates=missing,
                              seconds=round(time.time() - t0, 1))

            except NotFound as exc:
                is_absent, doc = sinks.record_not_found(
                    state_dir, item["item_id"], exc.evidence())
                if is_absent:
                    yield _result(item, "absent",
                                  reason=f"404/410 on two runs "
                                         f"{doc.get('gap_hours')} h apart",
                                  seconds=round(time.time() - t0, 1))
                    lane.record_success()
                else:
                    sinks.enqueue(state_dir, [item],
                                  "404 once — needs a second sighting")
                    yield _result(item, "queued",
                                  reason="404/410 seen once; absent needs a "
                                         "second sighting ≥ 6 h later")
                    lane.record_success()
            except BlockedError as exc:
                sinks.enqueue(state_dir, [item], "blocked: no credentials")
                yield _result(item, "queued", reason=str(exc)[:500])
            except CircuitOpen as exc:
                sinks.enqueue(state_dir, [item], "circuit breaker")
                yield _result(item, "queued", reason=str(exc)[:200])
            except (TransientError, PermanentError) as exc:
                tripped = lane.record_failure()
                sinks.enqueue(state_dir, [item], "ladder exhausted")
                yield _result(item, "queued", reason=str(exc)[:500],
                              seconds=round(time.time() - t0, 1))
                if tripped:
                    note("CIRCUIT BREAKER TRIPPED — the rest of this lane is "
                         "queued for the next run.")
            except Exception as exc:                          # noqa: BLE001
                # Anything unclassified is queued rather than allowed to fail
                # the bundle, and reported with its traceback so it can be
                # classified properly next time.
                lane.record_failure()
                sinks.enqueue(state_dir, [item], "unclassified error")
                yield _result(item, "queued",
                              reason=f"UNCLASSIFIED {type(exc).__name__}: "
                                     f"{exc}\n{traceback.format_exc()[-800:]}")
            finally:
                _cleanup(workdir)

        yield {"item_id": "_lane", "source": "_lane", "host": host,
               "lane": lane_no, "tier": -1, "status": "counters",
               "bytes": int(lane.counters["bytes"]), "sha256": None,
               "shard": None, "n_records": 0, "missing_dates": [],
               "seconds": 0.0, "reason": None, "at": None,
               "counters": dict(lane.counters), "breaker_open": lane.open}


def _cleanup(workdir: str) -> None:
    """Delete the raw download. It only runs after write-verify-mark, so a
    deleted file has already been proven to exist, verified, at --output."""
    if not os.path.isdir(workdir):
        return
    import shutil
    shutil.rmtree(workdir, ignore_errors=True)


class _Split(beam.DoFn):
    """Items whose `.done` marker already exists go straight to `present`."""

    def process(self, item, done_index):
        if item["item_id"] in set(done_index):
            yield beam.pvalue.TaggedOutput(
                "present", _result(item, "present", at=_utcnow()))
        else:
            yield item


# --------------------------------------------------------------------------
# the pipeline
# --------------------------------------------------------------------------
def run(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(
        description="Stage A of the polite parallel import (E-073)")
    ap.add_argument("--registry", default=None)
    ap.add_argument("--tiers", default="0")
    ap.add_argument("--only", default=None)
    ap.add_argument("--output", required=True,
                    help="where the shards go: a directory or gs://bucket/path")
    ap.add_argument("--from-queue", default=None,
                    help="use a retry_queue.jsonl as the manifest")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--report-dir", default="out")
    ap.add_argument("--state-dir", default=os.path.join(
        os.environ.get("TMPDIR", "/var/tmp"), "beam_import"))
    ap.add_argument("--include-disabled", action="store_true")
    ap.add_argument("--offline", action="store_true",
                    help="no network in the manifest step (skips the RG scrape)")
    args, beam_argv = ap.parse_known_args(argv)

    reg = registry.load(args.registry)
    if args.from_queue:
        items = manifest.from_queue(args.from_queue)
        rotated = sinks.rotate_queue(args.state_dir)
        print(f"from queue: {len(items)} item(s)"
              + (f"; rotated the old queue to {rotated}" if rotated else ""))
    else:
        tiers = [int(t) for t in str(args.tiers).split(",") if t.strip() != ""]
        items = manifest.build(reg, tiers, only=args.only,
                               include_disabled=args.include_disabled,
                               offline=args.offline)
    if not items:
        print("nothing to do: the manifest is empty", file=sys.stderr)
        return 0

    os.makedirs(args.report_dir, exist_ok=True)
    os.makedirs(args.state_dir, exist_ok=True)

    # The `.done` markers, listed ONCE, as a side input.
    done = {} if args.dry_run else sinks.list_done(args.output)
    print(f"manifest: {len(items)} item(s); {len(done)} already written "
          f"under {args.output}")

    cfg = {"registry": reg.path, "state_dir": args.state_dir,
           "dry_run": bool(args.dry_run), "output": args.output}

    options = PipelineOptions(beam_argv)
    options.view_as(SetupOptions).save_main_session = True
    out_prefix = os.path.join(args.report_dir, "report.jsonl")

    with beam.Pipeline(options=options) as p:
        done_index = beam.pvalue.AsList(
            p | "DoneIndex" >> beam.Create(sorted(done) or [""]))
        split = (p | "Items" >> beam.Create(items)
                 | "SkipDone" >> beam.ParDo(_Split(), done_index)
                 .with_outputs("present", main="todo"))
        worked = (split.todo
                  | "Key" >> beam.Map(lambda it: ((it["host"], it["lane"]), it))
                  | "Group" >> beam.GroupByKey()
                  | "Lane" >> beam.ParDo(LaneWorker(cfg)))
        _ = ((split.present, worked)
             | "Flatten" >> beam.Flatten()
             | "ToJson" >> beam.Map(json.dumps, sort_keys=True)
             | "Write" >> beam.io.WriteToText(out_prefix,
                                              shard_name_template="",
                                              num_shards=1))

    from . import report as report_mod
    report_mod.write_summary(out_prefix,
                             os.path.join(args.report_dir, "summary.md"), reg,
                             state_dir=args.state_dir)
    print(f"report:  {out_prefix}")
    print(f"summary: {os.path.join(args.report_dir, 'summary.md')}")

    queued = len(sinks.read_queue(sinks.queue_uri(args.state_dir)))
    print(f"queue:   {queued} item(s) in "
          f"{sinks.queue_uri(args.state_dir)}")

    # Exit 4 when a host's breaker tripped twice in this run: hard rule 7 says
    # a human decides what happens next, not the loop.
    records = report_mod.collect(out_prefix, args.state_dir)
    trips = report_mod.trips_per_host(records)
    bad = {h: n for h, n in trips.items() if n >= 2}
    if bad:
        print(f"BREAKER TRIPPED TWICE on {bad} — stopping. "
              "Read summary.md and report to Chris (README §9).",
              file=sys.stderr)
        return 4
    return 0 if queued == 0 else 3


if __name__ == "__main__":
    # The __main__ guard is not decoration: without it the DirectRunner's
    # multi_processing mode re-imports this module in every child process and
    # hangs, because each child starts its own pipeline.
    raise SystemExit(run())

"""The Apache Beam pipeline.

    Create(items)
      -> filter against the Hub listing (side input, taken ONCE at start)
      -> key by (host, lane)
      -> GroupByKey                 <-- this is what bounds concurrency
      -> ParDo(LaneWorker)          <-- one lane = one polite sequential stream
      -> WriteToText(report.jsonl)

Beam is the ORCHESTRATOR here, not the accelerator. GroupByKey guarantees that
every item sharing a key reaches ONE worker as ONE iterable, so the number of
keys per host is the maximum number of simultaneous connections that host will
ever see from us — whatever the runner decides about machines and threads.

Run it:

    python -m beam_import.pipeline --tiers 0 --report-dir out/tier0 \
        --state-dir /var/tmp/import \
        --runner DirectRunner --direct_running_mode multi_processing \
        --direct_num_workers 8

Credentials are read from the ENVIRONMENT inside DoFn.setup(). They are never
pipeline options and never argv: options are logged and shown in the job UI.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
import traceback
from typing import Any, Dict, Iterable, List, Optional, Tuple

import apache_beam as beam
from apache_beam.options.pipeline_options import PipelineOptions

from . import fetchers, manifest, publish, registry, transforms
from .hosts import (AbsentError, BlockedError, CircuitOpen, LaneState,
                    PermanentError, TransientError, lane_states)

RESULT_KEYS = ("item_id", "source", "host", "lane", "status", "bytes",
               "sha256", "hub_path", "seconds", "attempts", "error", "at")


def _utcnow() -> str:
    import datetime as dt
    return (dt.datetime.now(dt.timezone.utc)
            .replace(microsecond=0).isoformat())


def progress_path(state_dir: str, host: str, lane: int) -> str:
    """Where a lane appends its results as it goes.

    ONE FILE PER LANE, so two workers never write to the same file and there
    is nothing to lock. `report.py --live` reads the whole directory.
    """
    return os.path.join(state_dir, "progress", f"{host}-{lane}.jsonl")


def append_progress(path: str, record: Dict[str, Any]) -> None:
    """Append one record, then flush and fsync.

    ml/CLAUDE.md §5.25: a computation longer than half an hour writes its
    result incrementally, not at the end. Beam only writes report.jsonl when
    the whole pipeline finishes, so a Tier-0 run that is killed at hour eight
    would otherwise have nothing to show for it. Open-append-close per record
    is cheap here: a record is produced once per ITEM, and an item takes
    seconds to minutes.
    """
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(record, sort_keys=True) + "\n")
        fh.flush()
        os.fsync(fh.fileno())


def _result(item: Dict[str, Any], status: str, **kw: Any) -> Dict[str, Any]:
    rec = {
        "item_id": item["item_id"], "source": item["source"],
        "host": item["host"], "lane": item["lane"], "tier": item["tier"],
        "status": status, "bytes": 0, "sha256": None,
        "hub_path": item["hub_path"], "seconds": 0.0, "attempts": 0,
        "error": None, "note": None, "at": None,
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

    Everything transient is caught and turned into a result record. Only a
    programming error is allowed to propagate — Beam RETRIES a failed bundle
    (Dataflow four times), and a retried bundle would re-download everything
    the lane had already done, which is the opposite of polite.
    """

    def __init__(self, cfg: Dict[str, Any]) -> None:
        super().__init__()
        self.cfg = cfg              # plain dict: picklable, no clients in it

    def setup(self) -> None:
        """Clients are created HERE, once per worker, never in __init__."""
        self.reg = registry.load(self.cfg["registry"])
        self.lanes = lane_states(self.reg)
        interval = publish.commit_min_interval_s(
            self.reg.total_lanes(), self.reg.commit_budget_per_hour())
        # A dry run touches neither an upstream server nor the Hub, so it also
        # needs no credentials — do not build a client it will not use.
        self.pub = None if self.cfg["dry_run"] else publish.make_publisher(
            self.cfg["hub"], self.reg.hub["repo_id"],
            self.reg.hub.get("repo_type", "dataset"),
            commit_interval_s=interval)

    def process(self, element: Tuple[Tuple[str, int], Iterable[Dict[str, Any]]]):
        """Run one lane, writing every record to disk BEFORE yielding it.

        The write-then-yield order is the whole point: if the process dies, or
        the runner is killed, or the consumer of this generator raises, every
        record produced so far is already durable in the lane's progress file.
        """
        (host, lane_no), _items = element
        lane = self.lanes[host]
        path = progress_path(self.cfg["state_dir"], host, lane_no)
        for rec in self._run_lane(element):
            rec["at"] = _utcnow()
            if rec["status"] != "counters":
                # a running snapshot, so `report --live` can show backoffs and
                # trips mid-run rather than only after the lane finishes
                rec["backoffs_so_far"] = int(lane.counters["backoffs"])
                rec["trips_so_far"] = int(lane.counters["trips"])
            append_progress(path, rec)
            yield rec

    def _run_lane(self, element):
        (host, lane_no), items = element
        lane = self.lanes[host]
        state_dir = self.cfg["state_dir"]
        log: List[str] = []

        def note(msg: str) -> None:
            log.append(msg)
            print(f"[{host}/{lane_no}] {msg}", flush=True)

        batch: List[Tuple[str, str]] = []       # (local path, hub path)
        batch_items: List[Dict[str, Any]] = []
        batch_started = 0.0

        def flush():
            """Commit whatever is pending and yield one record per item."""
            nonlocal batch, batch_items, batch_started
            if not batch:
                return []
            out = []
            try:
                shas = self.pub.publish_verified(
                    batch, os.path.join(state_dir, "_scratch"),
                    f"beam_import: {batch_items[0]['source']} "
                    f"({len(batch)} file(s))", note=note)
                for it in batch_items:
                    out.append(_result(it, "published",
                                       sha256=shas.get(it["hub_path"]),
                                       bytes=it.get("_bytes", 0),
                                       seconds=round(time.time() - batch_started, 1),
                                       attempts=it.get("_attempts", 1)))
                lane.record_success()
            except (TransientError, PermanentError) as exc:
                for it in batch_items:
                    out.append(_result(it, "failed", error=str(exc)[:500]))
                lane.record_failure()
            finally:
                for local, _hub in batch:
                    if os.path.exists(local):
                        os.remove(local)
                batch, batch_items, batch_started = [], [], 0.0
            return out

        for item in items:                       # lazy: one item at a time
            if lane.open:
                yield _result(item, "deferred",
                              error="lane stopped by the circuit breaker")
                continue

            t0 = time.time()
            workdir = os.path.join(state_dir, _safe(item["source"]),
                                   _safe(item["item_id"]))
            try:
                # 1. --dry-run shows the plan and touches nothing at all.
                if self.cfg["dry_run"]:
                    yield _result(item, "planned",
                                  note=(item["urls"][0] if item["urls"]
                                        else item.get("dataset_id")))
                    continue

                # 2. skip before fetch — the Hub is HEADed again per item,
                #    because a second run or a retried bundle must not
                #    re-download what is already mirrored.
                if self.pub.exists(item["hub_path"]):
                    yield _result(item, "present")
                    lane.record_success()
                    continue

                # 3. a verify-only source is never fetched.
                if item["mode"] == "verify":
                    yield _result(item, "missing",
                                  error="verify-only source: not on the Hub")
                    continue

                got = fetchers.fetch_item(item, lane, workdir, note=note)
                outs = transforms.apply_transform(
                    item, got["paths"], workdir, note=note)
                item["_bytes"] = int(got.get("bytes", 0))
                item["_attempts"] = 1
                if not batch:
                    batch_started = time.time()
                for local in outs:
                    batch.append((local, item["hub_path"]))
                batch_items.append(item)
                if len(batch) >= max(1, int(item["batch_files"])):
                    for rec in flush():
                        yield rec

            except AbsentError as exc:
                # The archive honestly has no such file. Not our failure, and
                # it must NOT count towards the breaker.
                yield _result(item, "absent", error=str(exc)[:500],
                              seconds=round(time.time() - t0, 1))
                lane.record_success()
            except BlockedError as exc:
                yield _result(item, "blocked", error=str(exc)[:500])
            except CircuitOpen as exc:
                yield _result(item, "deferred", error=str(exc)[:200])
            except (TransientError, PermanentError) as exc:
                tripped = lane.record_failure()
                yield _result(item, "failed", error=str(exc)[:500],
                              seconds=round(time.time() - t0, 1),
                              attempts=lane.attempts_allowed())
                if tripped:
                    note("CIRCUIT BREAKER TRIPPED — the rest of this lane is "
                         "deferred. Wait at least an hour, then re-run.")
            except Exception as exc:                          # noqa: BLE001
                # Anything we did not classify is treated as transient rather
                # than allowed to fail the bundle. It is reported with its
                # traceback so it can be classified properly next time.
                lane.record_failure()
                yield _result(item, "failed",
                              error=f"UNCLASSIFIED {type(exc).__name__}: "
                                    f"{exc}\n{traceback.format_exc()[-800:]}")
            finally:
                _cleanup(workdir, keep=bool(batch))

        for rec in flush():
            yield rec

        yield {"item_id": "_lane", "source": "_lane", "host": host,
               "lane": lane_no, "tier": -1, "status": "counters",
               "bytes": int(lane.counters["bytes"]), "sha256": None,
               "hub_path": None, "seconds": 0.0, "attempts": 0,
               "error": None, "note": None,
               "counters": dict(lane.counters),
               "breaker_open": lane.open,
               "commits": getattr(self.pub, "commits", 0)}


def _cleanup(workdir: str, keep: bool) -> None:
    """Delete the per-item scratch, unless a file in it is awaiting commit."""
    if keep or not os.path.isdir(workdir):
        return
    import shutil
    shutil.rmtree(workdir, ignore_errors=True)


# --------------------------------------------------------------------------
# the pipeline
# --------------------------------------------------------------------------
def run(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(
        description="the polite parallel import (E-073)")
    ap.add_argument("--registry", default=None)
    ap.add_argument("--tiers", default="0")
    ap.add_argument("--only", default=None)
    ap.add_argument("--dry-run", action="store_true",
                    help="show the plan; fetch nothing")
    ap.add_argument("--report-dir", default="out")
    ap.add_argument("--state-dir", default=os.path.join(
        os.environ.get("TMPDIR", "/var/tmp"), "beam_import"))
    ap.add_argument("--hub", default="hub",
                    help="'hub', 'hub:<repo_id>' or 'local:<dir>' for tests")
    ap.add_argument("--include-disabled", action="store_true")
    ap.add_argument("--offline", action="store_true",
                    help="no network in the manifest step (skips the RG scrape)")
    ap.add_argument("--no-preflight", action="store_true",
                    help="skip the Hub round-trip check (not recommended)")
    args, beam_argv = ap.parse_known_args(argv)

    reg = registry.load(args.registry)
    tiers = [int(t) for t in str(args.tiers).split(",") if t.strip() != ""]
    items = manifest.build(reg, tiers, only=args.only,
                           include_disabled=args.include_disabled,
                           offline=args.offline)
    if not items:
        print("nothing to do: the manifest is empty", file=sys.stderr)
        return 1

    os.makedirs(args.report_dir, exist_ok=True)
    os.makedirs(args.state_dir, exist_ok=True)

    interval = publish.commit_min_interval_s(reg.total_lanes(),
                                             reg.commit_budget_per_hour())
    # A dry run needs no Hub client and therefore no credentials.
    pub = None if args.dry_run else publish.make_publisher(
        args.hub, reg.hub["repo_id"], reg.hub.get("repo_type", "dataset"),
        commit_interval_s=interval)

    # The Hub round trip, BEFORE a single upstream byte is fetched.
    if pub is not None and not args.no_preflight:
        sha = publish.roundtrip_preflight(pub, os.path.join(args.state_dir,
                                                            "_scratch"))
        print(f"preflight: Hub round trip OK (sha256 {sha[:12]})")

    # The Hub listing, taken ONCE, as a side input.
    listing = [] if pub is None else pub.list_paths()
    print(f"manifest: {len(items)} item(s); Hub already holds "
          f"{len(listing)} file(s)")

    cfg = {"registry": reg.path, "state_dir": args.state_dir,
           "dry_run": bool(args.dry_run), "hub": args.hub}

    options = PipelineOptions(beam_argv)
    from apache_beam.options.pipeline_options import SetupOptions
    options.view_as(SetupOptions).save_main_session = True
    out_prefix = os.path.join(args.report_dir, "report.jsonl")

    with beam.Pipeline(options=options) as p:
        hub_index = beam.pvalue.AsList(
            p | "HubListing" >> beam.Create(listing or [""]))

        all_items = p | "Items" >> beam.Create(items)
        split = (all_items
                 | "SkipPresent" >> beam.ParDo(_Split(), hub_index)
                 .with_outputs("present", main="todo"))

        worked = (split.todo
                  | "Key" >> beam.Map(lambda it: ((it["host"], it["lane"]), it))
                  | "Group" >> beam.GroupByKey()
                  | "Lane" >> beam.ParDo(LaneWorker(cfg)))

        _ = ((split.present, worked)
             | "Flatten" >> beam.Flatten()
             | "ToJson" >> beam.Map(json.dumps, sort_keys=True)
             | "Write" >> beam.io.WriteToText(out_prefix, shard_name_template="",
                                              num_shards=1))

    print(f"report: {out_prefix}")
    from . import report as report_mod
    report_mod.write_summary(out_prefix,
                             os.path.join(args.report_dir, "summary.md"), reg,
                             state_dir=args.state_dir)
    print(f"summary: {os.path.join(args.report_dir, 'summary.md')}")
    return 0


class _Split(beam.DoFn):
    """Items the Hub already holds go straight to a `present` record."""

    def process(self, item, hub_index):
        if item["hub_path"] in set(hub_index):
            yield beam.pvalue.TaggedOutput(
                "present", _result(item, "present", at=_utcnow()))
        else:
            yield item


if __name__ == "__main__":
    # The __main__ guard is not decoration: without it the DirectRunner's
    # multi_processing mode re-imports this module in every child process and
    # hangs (each child starts its own pipeline).
    raise SystemExit(run())

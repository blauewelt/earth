"""Nothing is dropped, and nothing transient escapes a DoFn.

Two rules meet here. Beam RETRIES a failed bundle — Dataflow four times, the
DirectRunner fails the job — so an uncaught 504 would re-run the whole lane
and re-download everything it had already fetched. And DESIGN §2 allows no
state in which work disappears: an item is `written`, `present`, `queued` or
`absent`, and `queued` means it is in `retry_queue.jsonl` waiting for the next
run. There is no `failed`.
"""
from __future__ import annotations

import datetime as dt
import json
import os

import pytest
import yaml

from beam_import import pipeline, report, sinks
from beam_import.hosts import LaneState, NotFound, TransientError


def _registry(tmp_path, n_files=3, fetcher="transient_test", url=None):
    doc = {
        "version": 1,
        "output": {"num_shards_per_group": 2},
        "hosts": {"h": {"max_lanes": 1, "min_gap_s": 0,
                        "backoff_ladder_s": [0, 0, 0, 0],
                        "serves": "nothing", "evidence": "test"}},
        "sources": [{
            "name": "flaky", "tier": 0, "host": "h", "mode": "fetch",
            "fetcher": fetcher, "chunk": "file",
            "files": [f"f{i}" for i in range(n_files)],
            "url": url or "file:///nowhere/{file}", "filename": "{file}",
            "transform": "opaque", "grid": "opaque",
            "bytes_wire": 1, "bytes_stored": 1,
        }],
    }
    p = tmp_path / "flaky.yaml"
    p.write_text(yaml.safe_dump(doc), encoding="utf-8")
    return str(p)


def _run(reg_path, tmp_path):
    rep, state = tmp_path / "report", tmp_path / "state"
    rc = pipeline.run(["--registry", reg_path, "--tiers", "0",
                       "--output", str(tmp_path / "out"),
                       "--report-dir", str(rep), "--state-dir", str(state),
                       "--offline", "--runner", "DirectRunner"])
    return rc, report.collect(str(rep / "report.jsonl"), str(state)), state


def test_a_transient_fetcher_queues_and_the_pipeline_finishes(tmp_path):
    reg = _registry(tmp_path, n_files=3)
    rc, records, state = _run(reg, tmp_path)
    assert rc == 3                                # "queue not empty", not a crash
    items = [r for r in records if r["status"] != "counters"]
    assert {r["status"] for r in items} == {"queued"}
    queued = {q["item_id"] for q in
              sinks.read_queue(sinks.queue_uri(str(state)))}
    assert queued == {"flaky/f0", "flaky/f1", "flaky/f2"}


def test_a_404_is_queued_once_and_absent_only_the_second_time(tmp_path):
    """The two-sighting rule, as two runs — the second one back-dated the way
    a previous day's evidence file would be."""
    reg = _registry(tmp_path, n_files=1, fetcher="http")
    rc, records, state = _run(reg, tmp_path)
    items = [r for r in records if r["status"] != "counters"]
    assert [r["status"] for r in items] == ["queued"]
    assert "second sighting" in items[0]["reason"]
    assert sinks.list_absent(str(state)) == []

    # back-date the first sighting: this is what "a run six hours ago" looks
    # like on disk.
    path = sinks.evidence_path(str(state), "flaky/f0")
    doc = json.loads(open(path, encoding="utf-8").read())
    doc["sightings"][0]["at"] = (
        dt.datetime.now(dt.timezone.utc) - dt.timedelta(hours=7)
    ).replace(microsecond=0).isoformat()
    open(path, "w", encoding="utf-8").write(json.dumps(doc))

    rc2, records2, state = _run(reg, tmp_path)
    items2 = [r for r in records2 if r["status"] != "counters"]
    assert [r["status"] for r in items2] == ["absent"]
    assert [a["item_id"] for a in sinks.list_absent(str(state))] == ["flaky/f0"]
    assert rc2 == 3       # the item is still in the queue from the FIRST run


def test_the_breaker_queues_the_rest_rather_than_dropping_it(tmp_path):
    reg = _registry(tmp_path, n_files=9)
    rc, records, state = _run(reg, tmp_path)
    items = [r for r in records if r["status"] != "counters"]
    assert len(items) == 9
    assert {r["status"] for r in items} == {"queued"}
    queued = {q["item_id"] for q in
              sinks.read_queue(sinks.queue_uri(str(state)))}
    assert len(queued) == 9, "every item of the stopped lane reached the queue"
    counters = [r for r in records if r["status"] == "counters"]
    assert counters and counters[0]["counters"]["trips"] == 1


def test_a_programming_error_is_queued_with_its_traceback(tmp_path):
    from beam_import import fetchers

    def boom(item, lane, workdir, note=None):
        raise KeyError("a bug, not a server")

    old = fetchers.FETCHERS["transient_test"]
    fetchers.FETCHERS["transient_test"] = boom
    try:
        reg = _registry(tmp_path, n_files=1)
        rc, records, _state = _run(reg, tmp_path)
    finally:
        fetchers.FETCHERS["transient_test"] = old
    items = [r for r in records if r["status"] != "counters"]
    assert items[0]["status"] == "queued"
    assert items[0]["reason"].startswith("UNCLASSIFIED KeyError")


def test_download_checks_the_size(tmp_path):
    """A short transfer must be caught by the size check, not by NetCDF."""
    from beam_import import fetchers

    src = tmp_path / "src.bin"
    src.write_bytes(b"a" * 100)
    lane = LaneState("h", {"max_lanes": 1, "min_gap_s": 0,
                           "backoff_ladder_s": [0]},
                     sleep_fn=lambda s: None)
    real_head = fetchers._head

    def lying_head(url):
        h = real_head(url)
        h["length"] = 999999
        return h

    fetchers._head = lying_head
    try:
        with pytest.raises(TransientError, match="short transfer"):
            fetchers.download_one([f"file://{src}"],
                                  str(tmp_path / "dest.bin"), lane)
    finally:
        fetchers._head = real_head
    assert not os.path.exists(str(tmp_path / "dest.bin"))
    assert not os.path.exists(str(tmp_path / "dest.bin.part"))


def test_a_record_is_on_disk_after_the_first_item_even_if_we_die(tmp_path):
    """ml/CLAUDE.md §5.25 — progress is an artefact, not a log line.

    One item's record is taken out of LaneWorker.process and the generator is
    then thrown away, which is what a killed process looks like from the
    lane's point of view. The first record must already be durable.
    """
    from beam_import.pipeline import LaneWorker, progress_path

    state = tmp_path / "state"
    cfg = {"registry": _registry(tmp_path, n_files=2),
           "state_dir": str(state), "dry_run": False,
           "output": str(tmp_path / "out")}
    worker = LaneWorker(cfg)
    worker.setup()
    items = [{"item_id": f"flaky/f{i}", "source": "flaky", "tier": 0,
              "host": "h", "lane": 0, "mode": "fetch",
              "fetcher": "transient_test", "transform": "opaque",
              "filename": f"f{i}", "urls": [], "bytes_wire": 1,
              "bytes_stored": 1, "unverified_url": False}
             for i in range(2)]

    gen = worker.process((("h", 0), iter(items)))
    first = next(gen)
    gen.close()

    path = progress_path(str(state), "h", 0)
    assert os.path.exists(path)
    lines = [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]
    assert len(lines) == 1
    assert lines[0]["item_id"] == first["item_id"] == "flaky/f0"
    assert lines[0]["status"] == "queued"
    assert lines[0]["at"] and lines[0]["backoffs_so_far"] >= 1


def test_live_and_merged_summaries(tmp_path):
    from beam_import.pipeline import append_progress, progress_path
    state = tmp_path / "state"
    p = progress_path(str(state), "h", 0)
    append_progress(p, {"item_id": "s/1", "source": "s", "host": "h",
                        "lane": 0, "status": "queued", "bytes": 0,
                        "seconds": 1.0, "at": "2026-09-04T10:00:00+00:00"})
    append_progress(p, {"item_id": "s/2", "source": "s", "host": "h",
                        "lane": 0, "status": "written", "bytes": 10,
                        "seconds": 2.0, "at": "2026-09-04T10:01:00+00:00"})
    final = tmp_path / "report.jsonl"
    final.write_text(json.dumps(
        {"item_id": "s/1", "source": "s", "host": "h", "lane": 0,
         "status": "written", "bytes": 5, "seconds": 3.0,
         "at": "2026-09-04T10:02:00+00:00"}) + "\n", encoding="utf-8")

    s = report.summarise(report.collect(str(final), str(state)))
    assert s["n_items"] == 2                      # s/1 deduped
    assert s["by_status"] == {"written": 2}       # the FINAL record wins
    text = report.live(str(state))
    assert "queue" in text and "absent" in text

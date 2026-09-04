"""Nothing transient may escape a DoFn.

Beam RETRIES a failed bundle — Dataflow four times, and the DirectRunner
simply fails the job. A 504 that escaped would therefore make the runner
re-run the whole lane, i.e. re-download everything the lane had already
fetched, i.e. hammer the host we were being careful with. So: a fetcher that
raises must produce a RECORD, and the pipeline must finish.
"""
from __future__ import annotations

import json
import os

import pytest
import yaml

from beam_import import pipeline
from beam_import.hosts import LaneState, PermanentError, TransientError


def _registry_of_one_flaky_source(tmp_path, n_files=3, fetcher="transient_test"):
    doc = {
        "version": 1,
        "hub": {"repo_id": "test/local", "repo_type": "dataset",
                "commit_budget_per_hour": 0},
        "hosts": {"h": {"max_lanes": 1, "min_gap_s": 0,
                        "backoff_ladder_s": [0, 0, 0, 0],
                        "serves": "nothing", "evidence": "test"}},
        "sources": [{
            "name": "flaky", "tier": 0, "host": "h", "mode": "fetch",
            "fetcher": fetcher, "chunk": "file",
            "files": [f"f{i}" for i in range(n_files)],
            "url": "file:///nowhere/{file}", "filename": "{file}",
            "hub_prefix": "flaky", "transform": "passthrough",
            "batch_files": 1, "bytes_wire": 1, "bytes_stored": 1,
        }],
    }
    p = tmp_path / "flaky.yaml"
    p.write_text(yaml.safe_dump(doc), encoding="utf-8")
    return str(p)


def _run(reg_path, tmp_path):
    out = tmp_path / "out"
    rc = pipeline.run(["--registry", reg_path, "--tiers", "0",
                       "--hub", f"local:{tmp_path / 'hub'}",
                       "--report-dir", str(out),
                       "--state-dir", str(tmp_path / "state"),
                       "--offline", "--runner", "DirectRunner"])
    records = [json.loads(l) for l in
               open(out / "report.jsonl", encoding="utf-8") if l.strip()]
    return rc, records


def test_a_transient_fetcher_yields_failed_and_the_pipeline_finishes(tmp_path):
    reg = _registry_of_one_flaky_source(tmp_path, n_files=3)
    rc, records = _run(reg, tmp_path)
    assert rc == 0                                # the JOB succeeded
    items = [r for r in records if r["status"] != "counters"]
    assert len(items) == 3
    assert {r["status"] for r in items} == {"failed"}
    assert all("simulated transient failure" in r["error"] for r in items)


def test_a_missing_file_url_is_reported_not_raised(tmp_path):
    reg = _registry_of_one_flaky_source(tmp_path, n_files=2, fetcher="http")
    rc, records = _run(reg, tmp_path)
    assert rc == 0
    items = [r for r in records if r["status"] != "counters"]
    assert {r["status"] for r in items} == {"failed"}
    assert all("404" in r["error"] for r in items)


def test_the_breaker_defers_rather_than_failing_the_rest(tmp_path):
    reg = _registry_of_one_flaky_source(tmp_path, n_files=9)
    rc, records = _run(reg, tmp_path)
    assert rc == 0
    items = [r for r in records if r["status"] != "counters"]
    assert sum(1 for r in items if r["status"] == "failed") == 5
    assert sum(1 for r in items if r["status"] == "deferred") == 4
    counters = [r for r in records if r["status"] == "counters"]
    assert counters and counters[0]["counters"]["trips"] == 1


def test_a_programming_error_is_not_swallowed_silently(tmp_path):
    """An unclassified exception still becomes a record — but one that says
    UNCLASSIFIED and carries a traceback, so it can be classified next time."""
    from beam_import import fetchers

    def boom(item, lane, workdir, note=None):
        raise KeyError("a bug, not a server")

    old = fetchers.FETCHERS["transient_test"]
    fetchers.FETCHERS["transient_test"] = boom
    try:
        reg = _registry_of_one_flaky_source(tmp_path, n_files=1)
        rc, records = _run(reg, tmp_path)
    finally:
        fetchers.FETCHERS["transient_test"] = old
    assert rc == 0
    items = [r for r in records if r["status"] != "counters"]
    assert items[0]["status"] == "failed"
    assert items[0]["error"].startswith("UNCLASSIFIED KeyError")


def test_download_one_checks_the_size(tmp_path):
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
        h["length"] = 999999            # claim more than the file holds
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


# --------------------------------------------------------------------------
# live progress: a record must be durable BEFORE it is handed to the runner
# --------------------------------------------------------------------------
def test_a_record_is_on_disk_after_the_first_item_even_if_we_die(tmp_path):
    """ml/CLAUDE.md §5.25 — progress is an artefact, not a log line.

    Beam only writes report.jsonl when the whole pipeline finishes, so a
    lane appends every record to its own progress file as it goes. This test
    consumes ONE item's record out of LaneWorker.process and then throws the
    generator away, which is what a killed process looks like from the
    lane's point of view. The first record must already be on disk.
    """
    from beam_import.pipeline import LaneWorker, progress_path

    state = tmp_path / "state"
    cfg = {"registry": _registry_of_one_flaky_source(tmp_path, n_files=2),
           "state_dir": str(state), "dry_run": False,
           "hub": f"local:{tmp_path / 'hub'}"}
    worker = LaneWorker(cfg)
    worker.setup()

    items = [{"item_id": f"flaky/f{i}", "source": "flaky", "tier": 0,
              "host": "h", "lane": 0, "mode": "fetch",
              "fetcher": "transient_test", "transform": "passthrough",
              "filename": f"f{i}", "hub_path": f"sources/flaky/f{i}",
              "urls": [], "batch_files": 1, "bytes_wire": 1,
              "bytes_stored": 1, "unverified_url": False}
             for i in range(2)]

    gen = worker.process((("h", 0), iter(items)))
    first = next(gen)                       # one item only, then walk away
    gen.close()

    path = progress_path(str(state), "h", 0)
    assert os.path.exists(path), "no progress file after the first item"
    lines = [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]
    assert len(lines) == 1
    assert lines[0]["item_id"] == first["item_id"] == "flaky/f0"
    assert lines[0]["status"] == "failed"
    assert lines[0]["at"]                    # every record is timestamped
    assert lines[0]["backoffs_so_far"] >= 1  # the running snapshot is there


def test_live_summary_reads_the_progress_files_mid_run(tmp_path):
    from beam_import import report
    from beam_import.pipeline import LaneWorker

    state = tmp_path / "state"
    cfg = {"registry": _registry_of_one_flaky_source(tmp_path, n_files=2),
           "state_dir": str(state), "dry_run": False,
           "hub": f"local:{tmp_path / 'hub'}"}
    worker = LaneWorker(cfg)
    worker.setup()
    items = [{"item_id": "flaky/f0", "source": "flaky", "tier": 0,
              "host": "h", "lane": 0, "mode": "fetch",
              "fetcher": "transient_test", "transform": "passthrough",
              "filename": "f0", "hub_path": "sources/flaky/f0", "urls": [],
              "batch_files": 1, "bytes_wire": 1, "bytes_stored": 1,
              "unverified_url": False}]
    gen = worker.process((("h", 0), iter(items)))
    next(gen)
    gen.close()

    text = report.live(str(state))
    assert "flaky" in text and "failed" in text
    assert "newest record" in text
    recs = report.collect(None, str(state))
    assert len(recs) == 1


def test_summary_merges_progress_with_the_final_report(tmp_path):
    """The final summary is the union, deduped by item, last record wins."""
    from beam_import import report
    from beam_import.pipeline import append_progress, progress_path

    state = tmp_path / "state"
    p = progress_path(str(state), "h", 0)
    append_progress(p, {"item_id": "s/1", "source": "s", "host": "h",
                        "lane": 0, "status": "failed", "bytes": 0,
                        "seconds": 1.0, "at": "2026-09-04T10:00:00+00:00"})
    append_progress(p, {"item_id": "s/2", "source": "s", "host": "h",
                        "lane": 0, "status": "published", "bytes": 10,
                        "seconds": 2.0, "at": "2026-09-04T10:01:00+00:00"})

    final = tmp_path / "report.jsonl"
    final.write_text(json.dumps(
        {"item_id": "s/1", "source": "s", "host": "h", "lane": 0,
         "status": "published", "bytes": 5, "seconds": 3.0,
         "at": "2026-09-04T10:02:00+00:00"}) + "\n", encoding="utf-8")

    out = tmp_path / "summary.md"
    report.write_summary(str(final), str(out), state_dir=str(state))
    text = out.read_text(encoding="utf-8")
    s = report.summarise(report.collect(str(final), str(state)))
    assert s["n_items"] == 2                       # not 3: s/1 was deduped
    assert s["by_status"] == {"published": 2}      # the FINAL s/1 record wins
    assert "2026-09-04T10:02:00+00:00" in text

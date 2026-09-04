"""Stage A end to end, offline: file:// fixtures in, TFRecord shards out."""
from __future__ import annotations

import json
import os

import pytest

from beam_import import pipeline, report, sinks, tfrecord, verify_output
from beam_import.example import one_str, parse_example


def _run(test_registry, tmp_path, extra=None):
    out = tmp_path / "out"
    state = tmp_path / "state"
    rep = tmp_path / "report"
    argv = ["--registry", test_registry, "--tiers", "0",
            "--output", str(out), "--state-dir", str(state),
            "--report-dir", str(rep), "--offline",
            "--runner", "DirectRunner"] + list(extra or [])
    rc = pipeline.run(argv)
    records = report.collect(str(rep / "report.jsonl"), str(state))
    return rc, records, out, state, rep


def _by_status(records):
    out = {}
    for r in records:
        out.setdefault(r["status"], []).append(r["item_id"])
    return out


def test_end_to_end(test_registry, tmp_path):
    rc, records, out, state, rep = _run(test_registry, tmp_path)
    # exit 3 = "the queue is not empty", which is true: the flaky lane.
    assert rc == 3
    st = _by_status(records)

    assert set(st["written"]) >= {
        "tiny/hello.dat", "ocean/ocean_200307.nc", "oisst/2003",
        "ncep/uflx/2003", "ncep/skt/2003", "ncep/land.sfc.gauss.nc",
        "rg/RG_ArgoClim_200307.nc"}
    assert "failed" not in st, "there is no `failed` status in this design"
    assert len(st["queued"]) == 7            # the whole flaky lane

    markers = sinks.list_done(str(out))
    assert all(len(m["sha256"]) == 64 for m in markers.values())
    shard = markers["ocean/ocean_200307.nc"]["shard"]
    payloads = tfrecord.read_records(shard)
    assert len(payloads) == 5                # five days, one record each
    dates = [one_str(parse_example(p), "date") for p in payloads]
    assert dates == sorted(dates), "records are in date order (DESIGN §4)"

    assert (rep / "summary.md").exists()
    text = (rep / "summary.md").read_text(encoding="utf-8")
    assert "Queue:" in text and "Absent:" in text


def test_a_missing_day_is_kept_and_requeued(test_registry, tmp_path):
    """The OISST fixture asks for five days and four exist. Stage A must
    write the four, record the fifth, and put the fifth back on the queue."""
    _rc, _records, out, state, _rep = _run(test_registry, tmp_path)
    marker = sinks.is_done(str(out), "oisst/2003")
    assert marker["n_records"] == 4
    assert marker["missing_dates"] == ["2003-07-17"]
    queued = {q["item_id"] for q in
              sinks.read_queue(sinks.queue_uri(str(state)))}
    assert "oisst/2003/2003-07-17" in queued


def test_rerun_is_idempotent(test_registry, tmp_path):
    _run(test_registry, tmp_path)
    _rc, records, _out, _state, _rep = _run(test_registry, tmp_path)
    st = _by_status(records)
    assert len(st.get("present", [])) >= 7
    # the seven already-written items are not fetched again
    assert not set(st.get("written", [])) & {"oisst/2003", "ncep/skt/2003"}


def test_dry_run_touches_nothing(test_registry, tmp_path):
    rc, records, out, _state, _rep = _run(test_registry, tmp_path,
                                          ["--dry-run"])
    assert rc == 0
    assert tfrecord.list_uris(str(out)) == []
    assert all(r["status"] == "present" for r in records
               if r["status"] != "counters")


def test_from_queue_uses_the_queue_as_the_manifest(test_registry, tmp_path):
    _rc, _records, out, state, rep = _run(test_registry, tmp_path)
    qpath = sinks.queue_uri(str(state))
    before = len(sinks.read_queue(qpath))
    assert before == 8                        # 7 flaky + 1 missing day

    rc = pipeline.run(["--registry", test_registry,
                       "--output", str(out), "--state-dir", str(state),
                       "--report-dir", str(rep), "--offline",
                       "--from-queue", qpath, "--runner", "DirectRunner"])
    assert rc == 3                             # still not empty: flaky is flaky
    rotated = [f for f in os.listdir(state) if f.startswith("retry_queue.")]
    assert "retry_queue.1.jsonl" in rotated, "the old queue was rotated aside"
    assert len(sinks.read_queue(qpath)) == 8, "nothing was lost on the re-run"


def test_verify_output(test_registry, tmp_path):
    _rc, _records, out, state, rep = _run(test_registry, tmp_path)
    rc = verify_output.main(["--registry", test_registry, "--tiers", "0",
                             "--output", str(out), "--state-dir", str(state),
                             "--offline", "--deep",
                             "--json-out", str(rep / "verify.json")])
    assert rc == 3                             # the flaky items are missing
    doc = json.loads((rep / "verify.json").read_text(encoding="utf-8"))
    assert doc["short"] == [] and doc["deep_bad"] == []
    assert sorted(doc["missing"]) == [f"flaky/{c}" for c in "abcdefg"]

    rc_ok = verify_output.main(["--registry", test_registry, "--tiers", "0",
                                "--only", "ocean", "--output", str(out),
                                "--state-dir", str(state), "--offline"])
    assert rc_ok == 0

"""The whole pipeline, offline: file:// fixtures in, a fake Hub out."""
from __future__ import annotations

import json
import os

import pytest

from beam_import import pipeline, publish, verify_hub


def _run(test_registry, tmp_path, extra=None):
    hub = tmp_path / "hub"
    out = tmp_path / "out"
    state = tmp_path / "state"
    argv = ["--registry", test_registry, "--tiers", "0",
            "--hub", f"local:{hub}", "--report-dir", str(out),
            "--state-dir", str(state), "--offline",
            "--runner", "DirectRunner"] + list(extra or [])
    rc = pipeline.run(argv)
    records = [json.loads(l) for l in
               open(out / "report.jsonl", encoding="utf-8") if l.strip()]
    return rc, records, hub, out


def _by_status(records):
    out = {}
    for r in records:
        out.setdefault(r["status"], []).append(r["item_id"])
    return out


def test_end_to_end(test_registry, tmp_path):
    rc, records, hub, out = _run(test_registry, tmp_path)
    assert rc == 0
    st = _by_status(records)

    # the three good sources landed, restore-verified
    assert sorted(st["published"]) == ["ncep/uflx.sfc.gauss/2001",
                                       "oisst/2001", "tiny/hello.dat"]
    for r in records:
        if r["status"] == "published":
            assert r["sha256"] and len(r["sha256"]) == 64

    # the flaky source tripped its breaker: five failed, the rest deferred
    assert len(st["failed"]) == 5
    assert len(st["deferred"]) == 2
    assert all(i.startswith("flaky/") for i in st["failed"] + st["deferred"])

    # the files are really on the fake Hub
    on_hub = set(publish.LocalPublisher(str(hub)).list_paths())
    assert "sources/tiny/hello.dat" in on_hub
    assert "sources/oisst/oisst_daily_2001.nc" in on_hub
    assert "sources/ncep/uflx.sfc.gauss.2001.nc" in on_hub

    # nothing is left behind on local disk except the report
    assert (out / "summary.md").exists()
    assert "breaker trip" in (out / "summary.md").read_text(encoding="utf-8")


def test_rerun_is_idempotent(test_registry, tmp_path):
    _run(test_registry, tmp_path)
    rc, records, _hub, _out = _run(test_registry, tmp_path)
    assert rc == 0
    st = _by_status(records)
    assert len(st.get("present", [])) == 3        # nothing re-fetched
    assert "published" not in st


def test_dry_run_touches_nothing(test_registry, tmp_path):
    rc, records, hub, _out = _run(test_registry, tmp_path, ["--dry-run"])
    assert rc == 0
    st = _by_status(records)
    assert len(st["planned"]) == 10
    # no preflight file, no uploads — a dry run puts nothing anywhere
    assert publish.LocalPublisher(str(hub)).list_paths() == []


def test_verify_hub_reports_missing(test_registry, tmp_path):
    _rc, _records, hub, out = _run(test_registry, tmp_path)
    rc = verify_hub.main(["--registry", test_registry, "--tiers", "0",
                          "--hub", f"local:{hub}",
                          "--report", str(out / "report.jsonl"),
                          "--out-dir", str(out), "--offline"])
    # flaky/* never landed, so the tier is NOT complete — and it says so.
    assert rc == 3

    rc2 = verify_hub.main(["--registry", test_registry, "--tiers", "0",
                           "--only", "tiny", "--hub", f"local:{hub}",
                           "--report", str(out / "report.jsonl"),
                           "--out-dir", str(out), "--offline", "--publish"])
    assert rc2 == 0
    on_hub = publish.LocalPublisher(str(hub)).list_paths()
    assert "sources/MANIFEST_tier0.json" in on_hub
    doc = json.loads((hub / "sources" / "MANIFEST_tier0.json")
                     .read_text(encoding="utf-8"))
    assert doc["n_files"] == 1
    assert doc["files"][0]["sha256"]


def test_preflight_round_trip(tmp_path):
    pub = publish.LocalPublisher(str(tmp_path / "hub"))
    sha = publish.roundtrip_preflight(pub, str(tmp_path / "scratch"))
    assert len(sha) == 64
    assert pub.exists("sources/_preflight/roundtrip.txt")


def test_commit_interval_arithmetic():
    # 60 commits an hour shared over 27 lanes is one commit per lane per 27 min
    assert publish.commit_min_interval_s(27, 60) == pytest.approx(1620.0)
    assert publish.commit_min_interval_s(27, 0) == 0.0

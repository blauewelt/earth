#!/usr/bin/env python3
"""Registry spans say where the DATA ends, not where the build window ends
(E-085 decision D7, Chris 2026-10-06: "Registry end date: should show where
the data ends").

Pins ml/registry_spans.py on synthetic stores — a tier-P store whose rows end
years before its requested window, a tier-G store whose last bin is part
filled, a group whose first index row holds no frame, empty stores of both
tiers — plus the two builders' wiring and ml/registry_guard.py's --spans /
--allow-group modes. No network.

  python3 tests/test_family1_registry_spans.py
"""
import json
import os
import sys
import tempfile

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ML = os.path.join(os.path.dirname(HERE), "ml")
sys.path.insert(0, ML)

import registry_spans as spans                                  # noqa: E402
import registry_guard as guard                                  # noqa: E402
from family1 import sharded as sh                               # noqa: E402

DAY = 86400


def _sec(iso):
    import datetime as dt
    t = dt.datetime.fromisoformat(iso.replace("Z", "+00:00"))
    if t.tzinfo is None:
        t = t.replace(tzinfo=dt.timezone.utc)
    return int((t - spans.EPOCH).total_seconds())


def _tier_p_dir(times, col="time_s", dtype=np.int32):
    d = tempfile.mkdtemp(prefix="spanP")
    np.save(os.path.join(d, f"{col}.npy"), np.asarray(times, dtype=dtype))
    return d


# ================================================================ tier P ===
def test_tier_p_data_ending_before_its_window():
    t = sorted([_sec("1972-07-24T03:10:00Z"), _sec("1990-01-01T00:00:00Z"),
                _sec("2023-09-10T22:59:59Z")])
    d = _tier_p_dir(t, dtype=np.int64)
    meta = {"N": 3, "schema_version": 3, "bin_first": t[0] // 432000,
            "bin_last": t[-1] // 432000,
            "date_range": ["1972-01-01", "2026-12-31"]}
    out = spans.span_fields(meta, "P", spans.local_source(d))
    assert out["record_span"] == ["1972-07-24", "2023-09-10"], out
    assert out["record_span_instants"] == ["1972-07-24T03:10:00Z",
                                           "2023-09-10T22:59:59Z"], out
    assert out["record_span_basis"] == "time_column"
    assert out["requested_window"] == ["1972-01-01", "2026-12-31"]
    assert out["date_range"] == out["record_span"]
    assert out["record_span"][1] < out["requested_window"][1]


def test_tier_p_schema1_days_column():
    t = np.array([100.25, 15000.5], dtype=np.float32)   # days since 1982
    d = _tier_p_dir(t, col="time_days", dtype=np.float32)
    meta = {"N": 2, "schema_version": 1, "bin_first": 20, "bin_last": 3000,
            "date_range": ["1982-01-01", "2026-12-31"]}
    out = spans.span_fields(meta, "P", spans.local_source(d))
    assert out["record_span"] == ["1982-04-11", "2023-01-26"], out
    assert out["record_span_instants"][0] == "1982-04-11T06:00:00Z"


def test_tier_p_unreadable_column_falls_back_to_the_stores_bins():
    d = tempfile.mkdtemp(prefix="spanPnone")          # no time column at all
    meta = {"N": 5, "schema_version": 2, "bin_first": 0, "bin_last": 1,
            "date_range": ["1982-01-01", "2026-12-31"]}
    out = spans.span_fields(meta, "P", spans.local_source(d))
    assert out["record_span_basis"] == "bins"
    assert out["record_span"] == ["1982-01-01", "1982-01-10"], out
    out2 = spans.span_fields(meta, "P", None)
    assert out2 == out


def test_tier_p_empty_store():
    meta = {"N": 0, "schema_version": 2, "bin_first": None, "bin_last": None,
            "date_range": ["2020-01-01", "2020-12-31"]}
    out = spans.span_fields(meta, "P", None)
    assert out["record_span"] is None and out["date_range"] is None
    assert out["record_span_basis"] == "empty"
    assert out["requested_window"] == ["2020-01-01", "2020-12-31"]
    # an empty column with N missing from store.json: also empty
    d = _tier_p_dir([])
    out = spans.span_fields({"schema_version": 2,
                             "date_range": ["2020-01-01", "2020-12-31"]},
                            "P", spans.local_source(d))
    assert out["record_span_basis"] == "empty"


# ================================================================ tier G ===
def _era5_like(last_frame_present=7):
    """One group, bins 0..2 of 20 six-hourly frames; bin 2 filled only up to
    `last_frame_present`; the window was asked through bin 4."""
    F, fs = 20, 21600
    missing = [{"group": "x", "bin": 2, "frame": f, "reason": "after_record",
                "day": ""} for f in range(last_frame_present + 1, F)]
    missing += [{"group": "x", "bin": b, "frame": f, "reason": "after_record",
                 "day": ""} for b in (3, 4) for f in range(F)]
    return {"frames_per_bin": F, "frame_seconds": fs,
            "date_range": ["1982-01-01", "1982-01-25"],
            "groups": {"x": {"bin_first": 0, "bin_last": 2, "bins": 3,
                             "frames_per_bin": F, "frame_seconds": fs,
                             "frames_present": 40 + last_frame_present + 1,
                             "shard_index": "x/shard_index.npy"}},
            "missing_frames": missing}


def test_tier_g_last_bin_part_filled():
    out = spans.span_fields(_era5_like(7), "G", None)
    # bin 2 starts 1982-01-11; frame 7 starts 1.75 days later, 18 UTC
    assert out["record_span_instants"] == ["1982-01-01T00:00:00Z",
                                           "1982-01-12T18:00:00Z"], out
    assert out["record_span"] == ["1982-01-01", "1982-01-12"], out
    assert out["record_span_basis"] == "frames"
    assert out["requested_window"] == ["1982-01-01", "1982-01-25"]


def _write_index(dirpath, rows, C=1):
    dt_ = sh.shard_index_dtype(C)
    arr = np.zeros(len(rows), dtype=dt_)
    for i, (b, mask) in enumerate(rows):
        arr[i]["bin"] = b
        arr[i]["frame_mask"] = mask
        arr[i]["frames_present"] = bin(mask).count("1")
    os.makedirs(os.path.join(dirpath, "g"), exist_ok=True)
    np.save(os.path.join(dirpath, "g", "shard_index.npy"), arr,
            allow_pickle=False)


def test_tier_g_first_row_with_no_frame_reads_the_shard_index():
    # F = 1; index rows 10 (empty), 12 (present), 15 (present), 16 (empty)
    d = tempfile.mkdtemp(prefix="spanG")
    _write_index(d, [(10, 0), (12, 1), (15, 1), (16, 0)])
    meta = {"frames_per_bin": 1, "frame_seconds": 432000,
            "date_range": ["1982-01-01", "1983-01-01"],
            "groups": {"g": {"bin_first": 10, "bin_last": 16, "bins": 4,
                             "frames_per_bin": 1, "frame_seconds": 432000,
                             "frames_present": 2,
                             "shard_index": "g/shard_index.npy"}},
            "missing_frames": [{"group": "g", "bin": b, "frame": 0,
                                "reason": "no_frame_in_bin", "day": ""}
                               for b in (10, 11, 13, 14, 16)]}
    out = spans.span_fields(meta, "G", spans.local_source(d))
    assert out["record_span_basis"] == "frames+shard_index", out
    assert out["record_span_instants"] == ["1982-03-02T00:00:00Z",
                                           "1982-03-17T00:00:00Z"], out
    assert out["record_span"] == ["1982-03-02", "1982-03-21"], out
    # without a source it can only say it to the bin, and says so
    out2 = spans.span_fields(meta, "G", None)
    assert out2["record_span_basis"] == "frame_bins"
    assert out2["record_span"] == ["1982-02-20", "1982-03-26"], out2


def test_tier_g_multi_group_takes_the_union():
    m = _era5_like(7)
    m["groups"]["y"] = {"bin_first": 1, "bin_last": 3, "bins": 3,
                        "frames_per_bin": 20, "frame_seconds": 21600,
                        "frames_present": 60}
    m["missing_frames"] = [e for e in m["missing_frames"]
                           if not (e["bin"] == 3)] + \
        [{"group": "x", "bin": 3, "frame": f, "reason": "after_record",
          "day": ""} for f in range(20)]
    out = spans.span_fields(m, "G", None)
    assert out["record_span_instants"] == ["1982-01-01T00:00:00Z",
                                           "1982-01-20T18:00:00Z"], out


def test_tier_g_empty_store():
    meta = {"frames_per_bin": 5, "frame_seconds": 86400,
            "date_range": ["2020-01-01", "2020-12-31"],
            "groups": {"g": {"bin_first": None, "bin_last": None, "bins": 0,
                             "frames_present": 0}},
            "missing_frames": []}
    out = spans.span_fields(meta, "G", None)
    assert out["record_span"] is None and out["date_range"] is None
    assert out["record_span_basis"] == "empty"
    out = spans.span_fields({"date_range": ["2020-01-01", "2020-12-31"]},
                            "G", None)
    assert out["record_span_basis"] == "empty"


# ============================================================== builders ===
def test_family1_entry_states_the_data_span_and_keeps_the_window():
    import build_family1_registry as r
    work = tempfile.mkdtemp(prefix="spanW")
    sd = os.path.join(work, "glodap")
    os.makedirs(sd)
    t = [_sec("1972-07-24T00:00:00Z"), _sec("2023-09-10T12:00:00Z")]
    np.save(os.path.join(sd, "time_s.npy"), np.asarray(t, np.int32))
    json.dump({"N": 2, "schema_version": 2, "bin_first": t[0] // 432000,
               "bin_last": t[1] // 432000, "n_bins": 1,
               "date_range": ["1972-01-01", "2026-12-31"], "sha256": {}},
              open(os.path.join(sd, "store.json"), "w"))
    g = r.entry("glodap", r.REGISTRY["glodap"](), r.PUBLIC_REPO,
                use_hub=False, work=work)
    assert g["built"]
    assert g["record_span"] == ["1972-07-24", "2023-09-10"], g["record_span"]
    assert g["date_range"] == g["record_span"]
    assert g["requested_window"] == ["1972-01-01", "2026-12-31"]
    reg = r.build_one("12", use_hub=False)
    assert "span_rule" in reg and "requested_window" in reg["span_rule"]


def test_family10_store_entry_states_the_data_span():
    import build_family10_registry as r10
    d = _tier_p_dir([_sec("1979-02-15T00:00:00Z"),
                     _sec("2024-06-30T06:00:00Z")])
    meta = {"N": 2, "schema_version": 2, "bin_first": -600, "bin_last": 3100,
            "date_range": ["1979-01-01", "2024-12-31"], "sha256": {},
            "channels": [{"name": "sst", "unit": "degC"}]}
    e = r10._store_entry("gdp", meta, "o/r", "tensors/x/gdp", local=d,
                         use_hub=False)
    assert e["date_range"] == ["1979-02-15", "2024-06-30"], e["date_range"]
    assert e["record_span"] == e["date_range"]
    assert e["requested_window"] == ["1979-01-01", "2024-12-31"]


# ================================================================= guard ===
def _reg(groups, **top):
    return {"generated_utc": "t", "builder_git_sha": "s", "groups": groups,
            **top}


def _run_guard(old, new, *flags):
    d = tempfile.mkdtemp(prefix="guard")
    a, b = os.path.join(d, "a.json"), os.path.join(d, "b.json")
    json.dump(old, open(a, "w"))
    json.dump(new, open(b, "w"))
    return guard.main([a, b, *flags])


def test_guard_spans_mode():
    old = _reg([{"name": "s", "date_range": ["2000-01-01", "2026-12-31"],
                 "record_span": ["2000-01-01", "2026-12-31"], "N": 5,
                 "files": [{"name": "a", "sha256": "1"}]}])
    new = _reg([{"name": "s", "date_range": ["2000-03-01", "2023-01-01"],
                 "record_span": ["2000-03-01", "2023-01-01"],
                 "record_span_instants": ["x", "y"],
                 "record_span_basis": "time_column",
                 "requested_window": ["2000-01-01", "2026-12-31"], "N": 5,
                 "files": [{"name": "a", "sha256": "1"}]}],
               span_rule="rule")
    assert _run_guard(old, new) == 1                  # strict: refuses
    assert _run_guard(old, new, "--spans") == 0
    lost = json.loads(json.dumps(new))
    lost["groups"][0]["requested_window"] = ["2000-03-01", "2023-01-01"]
    assert _run_guard(old, lost, "--spans") == 1      # window not kept
    other = json.loads(json.dumps(new))
    other["groups"][0]["files"][0]["sha256"] = "2"
    assert _run_guard(old, other, "--spans") == 1     # a hash changed


def test_guard_allow_group_mode():
    old = _reg([{"name": "era5_q", "files": [{"name": "a", "sha256": "1"}]},
                {"name": "era5_t", "files": [{"name": "a", "sha256": "1"}]}])
    new = json.loads(json.dumps(old))
    new["groups"][0]["files"][0]["sha256"] = "2"
    assert _run_guard(old, new) == 1
    assert _run_guard(old, new, "--allow-group", "era5_q") == 0
    new["groups"][1]["files"][0]["sha256"] = "3"
    assert _run_guard(old, new, "--allow-group", "era5_q") == 1


if __name__ == "__main__":
    n = 0
    for k, f in sorted(globals().items()):
        if k.startswith("test_") and callable(f):
            f()
            n += 1
            print(f"ok  {k}")
    print(f"{n} passed")

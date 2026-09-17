#!/usr/bin/env python3
"""The `tide` and `tide_private` adapters' smoke (family 1.0.tf, E-082).

    python3 -m pytest -q tests/test_family1_tide.py

One synthetic GESLA (`_gesla.make_smoke_sources`) in the UHSLC ERDDAP's
answers — the distinct() record CSV with its units line, the info CSV, and
one NetCDF-3 `.nc` per record written by `_nc3.write` — read by both
adapters. The two stores must hold disjoint records whose union is every
record with data, and the private one must be refused the public repository.
"""
import argparse
import json
import os
import sys

import numpy as np
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "ml"))

import build_family10_stores as b10                             # noqa: E402
import build_family1_stores as b1                               # noqa: E402
from family1 import adapters as fam                             # noqa: E402
from family1.adapters import _gesla, _nc3, tide, tide_private   # noqa: E402

# window 1899-12-30 .. 1900-01-02 = 96 hours.
# public: halifax, st_john's, the_battery hourly = 288, minus eight flagged
#         rows at halifax = 280; the empty-1 record has no row in it, and
#         alpena (a Great Lakes gauge at ~176.8 m) has 96 rows, every one
#         outside the +-50 m bound by design.
# private: bergen ten-minute 576 + capetown 96 + venezia 96 = 768.
PUBLIC_ROWS, PRIVATE_ROWS = 280, 768
PUBLIC_JAN, PRIVATE_JAN = 3 * 48, 288 + 48 + 48


def ns(tmp, src, store, **over):
    ad = tide.TideAdapter()
    d = dict(store=store, work=str(tmp / "w"), source_dir=src,
             start=ad.smoke_window[0], end=ad.smoke_window[1], stage="all",
             force=False, attempts=1, qc_keep=2,
             check_chunk_rows=b10.CHECK_CHUNK_ROWS, assemble="auto",
             parts_from_hub=False, push_parts=False,
             allow_missing_years=False)
    d.update(over)
    return argparse.Namespace(**d)


def test_registered_on_two_tracks():
    assert fam.REGISTRY["tide"] is tide.TideAdapter
    assert fam.REGISTRY["tide_private"] is tide_private.TidePrivateAdapter
    assert tide.TideAdapter.distribution == "public"
    assert tide_private.TidePrivateAdapter.distribution == "private"
    assert tide_private.TidePrivateAdapter.licence["redistribution"] == "no"
    assert b1.layout_for(tide_private.TidePrivateAdapter()).repo_id == \
        "chfrank/earth-tensors-private"
    assert b1.layout_for(tide.TideAdapter()).repo_id == \
        "chfrank/earth-tensors"
    with pytest.raises(SystemExit):
        b10.check_publish_target(
            tide_private.TidePrivateAdapter(),
            b1.layout_for(tide_private.TidePrivateAdapter()),
            "chfrank/earth-tensors")
    for c in (tide.TideAdapter, tide_private.TidePrivateAdapter):
        assert c().C == 1 and c.time_dtype == "int64" and not c.per_year
        assert c.first_year == 1800


def test_the_sort_rule_reads_contributor_and_country():
    assert _gesla.track_of("CMEMS", "FRA") == "private"
    assert _gesla.track_of("CV", "ITA") == "private"
    assert _gesla.track_of("UZ", "ESP") == "private"
    assert _gesla.track_of("UHSLC_FD", "ZAF") == "private"
    assert _gesla.track_of("UHSLC_FD", "zaf ") == "private"
    for a, c in (("NOAA", "USA"), ("UHSLC_RQ", "FRA"), ("REFMAR", "FRA"),
                 ("CMEMS_X", "NOR"), ("", "")):
        assert _gesla.track_of(a, c) == "public"


@pytest.mark.parametrize("store, rows, jan", [
    ("tide", PUBLIC_ROWS, PUBLIC_JAN),
    ("tide_private", PRIVATE_ROWS, PRIVATE_JAN)])
def test_smoke_all_stages_and_the_probe(tmp_path, store, rows, jan):
    res = b1.run_smoke(store, root=str(tmp_path / "s"), keep=True)
    assert len(res["truth"]) == rows
    p = res["probe"]
    assert p["rows"] == jan and p["schema_version"] == 3
    assert p["stored_bytes_per_row"] == 33
    assert sum(p["out_of_bounds_stored"].values()) == 0
    assert p["bytes_fetched"] > 0
    work = res["work"]
    m = json.load(open(os.path.join(work, store, store, "store.json")))
    assert m["N"] == rows and m["distribution"] == (
        "private" if store == "tide_private" else "public")
    plan = json.load(open(os.path.join(work, store, "plan.json")))
    assert plan["sort"]["private"]["by_contributor"] == {
        "CMEMS": 1, "CV": 1, "UHSLC_RQ": 1}
    assert plan["sort"]["public"]["by_contributor"] == {"MEDS": 2, "NOAA": 3}
    assert plan["first_record"]["rows_in_window"] > 0
    k = m["counts"]
    pj = json.load(open(os.path.join(work, store, store, "platforms.json")))
    if store == "tide":
        assert m["per_year"] == {"1899": 136, "1900": 144}
        assert k["records_track"] == 5 and k["records_with_rows"] == 3
        assert k["records_empty_in_window"] == 1
        assert k["rows_read"] == 4 * 120 + 1 + 120
        assert k["rows_outside_window"] == 4 * 24 + 120
        for name in ("flag1_interpolated", "flag1_missing", "flag1_unknown",
                     "flag2_do_not_use", "value_missing",
                     "rows_duplicate_time"):
            assert k[name] == 1, name
        assert k["qc_dropped"] == 2
        # one garbage value at halifax, and all 96 of the lake gauge's rows
        assert k["out_of_bounds"] == {"sea_level": 1 + 96}
        # the probe month (January) has the lake's 48 rows; the garbage
        # value sits on 1899-12-30
        assert p["out_of_bounds"] == {"sea_level": 48}
        assert k["records_all_out_of_bounds"] == 1
        assert k["records_all_out_of_bounds_ids"] == [
            "alpena_mi-9075065-usa-noaa"]
        assert k["qc"] == {"0": 1, "1": 279}
        plat = np.load(os.path.join(work, store, store, "platform.npy"))
        assert b10.platform_hash("alpena_mi-9075065-usa-noaa") not in \
            set(plat.tolist())
        assert {v["contributor"] for v in pj.values()} == {"MEDS", "NOAA"}
        assert "st_john's-1-can-meds" in {v["id"] for v in pj.values()}
    else:
        assert m["per_year"] == {"1899": 384, "1900": 384}
        assert k["records_track"] == 3 and k["records_with_rows"] == 3
        assert k["qc"] == {"1": 768}
        assert {v["contributor"] for v in pj.values()} == {
            "CMEMS", "CV", "UHSLC_RQ"}
        assert {v["country"] for v in pj.values()} == {"NOR", "ITA", "ZAF"}
    assert all(v["datum"] is None for v in pj.values())
    assert all(v["track"] == ("private" if store == "tide_private"
                              else "public") for v in pj.values())


def test_the_two_stores_are_disjoint_and_cover_the_archive(tmp_path):
    src = str(tmp_path / "src")
    _gesla.make_smoke_sources(src, b10.parse_date("1899-12-30"),
                              b10.parse_date("1900-01-02"), "public")
    plats = {}
    for store, cls in (("tide", tide.TideAdapter),
                       ("tide_private", tide_private.TidePrivateAdapter)):
        ad = cls()
        ctx = b10.Ctx(ns(tmp_path / store, src, store), adapter=ad,
                      layout=b1.layout_for(ad))
        rows = [r for y, r, c in ad.fetch_stream(ctx) if y is not None]
        plats[store] = set(np.concatenate([r["platform"] for r in rows])
                           .tolist())
    assert not plats["tide"] & plats["tide_private"]
    with_data = {b10.platform_hash(r[0]) for r in _gesla.SMOKE_RECORDS
                 if r[-1] not in ("none", "lake")}
    assert plats["tide"] | plats["tide_private"] == with_data


def test_the_nc_reader_and_the_row_rules():
    t70 = np.array([0.0, 3600.0, 3600.0, 7200.0, 10800.0])
    sl = np.array([1.0, 2.0, 3.0, _gesla.FILL, 5.0])
    f1 = np.array([1, 1, 1, 1, 0], np.int16)
    f2 = np.array([1, 1, 1, 1, 1], np.int16)
    c = {}
    t, v, q = _gesla.rows_from(t70, sl, f1.astype(np.int64),
                               f2.astype(np.int64), -10**12, 10**12, 2, c)
    assert (t + _gesla.cm.EPOCH_S_1970).tolist() == [0, 3600, 10800]
    assert v.tolist() == [1.0, 3.0, 5.0] and q.tolist() == [1, 1, 0]
    assert c["value_missing"] == 1 and c["rows_duplicate_time"] == 1


def test_a_changed_answer_is_refused(tmp_path):
    p = str(tmp_path / "x.nc")
    _nc3.write(p, [("row", 2)],
               [("time", ("row",), np.array([0.0, 1.0]),
                 {"units": "seconds since 1970-01-01T00:00:00Z"}),
                ("sea_level", ("row",), np.array([1.0, 2.0]),
                 {"units": "mm"}),
                ("flag1", ("row",), np.array([1, 1], np.int16), {}),
                ("flag2", ("row",), np.array([1, 1], np.int16), {})])
    with pytest.raises(_gesla.FormatError, match="units"):
        _gesla.read_nc(open(p, "rb").read())
    with pytest.raises(_gesla.FormatError, match="EMPTY"):
        _gesla.parse_records(",".join(_gesla.RECORD_FIELDS) + "\n,,,,,,\n")
    with pytest.raises(_gesla.FormatError, match="header"):
        _gesla.parse_records("a,b\n")


def test_a_missing_record_is_an_absence_not_an_empty_window(tmp_path):
    src = str(tmp_path / "src")
    _gesla.make_smoke_sources(src, b10.parse_date("1899-12-30"),
                              b10.parse_date("1900-01-02"), "public")
    os.remove(os.path.join(src, "gesla", "data",
                           _gesla.local_name("the_battery-8518750-usa-noaa")))
    ad = tide.TideAdapter()
    ctx = b10.Ctx(ns(tmp_path, src, "tide"), adapter=ad,
                  layout=b1.layout_for(ad))
    out = list(ad.fetch_stream(ctx))
    assert len(ctx.absent) == 1
    assert "the_battery" in ctx.absent[0]["why"]
    c = out[-1][2]
    assert c["records_empty_in_window"] == 1
    assert c["records_retried"] == 1 and c["records_failed"] == 1
    assert c["records_failed_ids"] == ["the_battery-8518750-usa-noaa"]


def test_a_record_that_fails_once_is_retried_after_the_pass(tmp_path,
                                                            monkeypatch):
    src = str(tmp_path / "src")
    truth = _gesla.make_smoke_sources(src, b10.parse_date("1899-12-30"),
                                      b10.parse_date("1900-01-02"), "public")
    ad = tide.TideAdapter()
    ctx = b10.Ctx(ns(tmp_path, src, "tide"), adapter=ad,
                  layout=b1.layout_for(ad))
    real = ad._fetch
    calls = []

    def flaky(ctx_, rid, lo, hi):
        calls.append(rid)
        if rid.startswith("halifax") and calls.count(rid) == 1:
            raise IOError("Connection reset by peer")
        return real(ctx_, rid, lo, hi)
    monkeypatch.setattr(ad, "_fetch", flaky)
    out = list(ad.fetch_stream(ctx))
    assert ctx.absent == []
    c = out[-1][2]
    assert c["records_retried"] == 1 and "records_failed" not in c
    assert sum(len(r["bin"]) for y, r, _ in out if y is not None) == \
        len(truth)
    # the retried record comes LAST in the stream
    assert calls[-1].startswith("halifax")


def test_the_data_url_quotes_the_record_and_the_window():
    u = _gesla.data_url("st_john's-1-can-meds", 1167609600, 1170287999)
    assert "record_id=%22st_john%27s-1-can-meds%22" in u
    assert "time%3E=2019-01-01T00%3A00%3A00Z" in u
    assert "time%3C=2019-01-31T23%3A59%3A59Z" in u
    assert u.startswith(_gesla.TABLEDAP + ".nc?time%2Csea_level%2Cflag1")


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))

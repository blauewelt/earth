#!/usr/bin/env python3
"""The `bgcargo` adapter's smoke (family 1.gf, E-082) — no network.

    python3 -m pytest -q tests/test_family1_bgcargo.py

The synthetic archive (`bgcargo.make_smoke_sources`) is written in the real
GDAC layout: the gzip synthetic-profile index (its eight comment lines and
header) and `dac/<dac>/<wmo>/<wmo>_Sprof.nc` in the NETCDF4_CLASSIC Sprof
layout (STATION_PARAMETERS aligned with PARAMETER_DATA_MODE, raw and
ADJUSTED arrays with their QC). Expected counts are exact.
"""
import gzip
import json
import os
import sys

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "ml"))

import build_family10_stores as b10                             # noqa: E402
import build_family1_stores as b1                               # noqa: E402
import family10_store as f10                                    # noqa: E402
from family1 import adapters as fam                             # noqa: E402
from family1.adapters import bgcargo                            # noqa: E402

# three floats x four window days x two profiles, plus one descending twin,
# less a JULD_QC-4 profile and a position-fill profile (both 2022-01-01)
TRUTH_ROWS = 3 * 4 * 2 + 1 - 2
PROBE_ROWS = 3 * 2 * 2 + 1 - 2


def test_registered():
    assert fam.REGISTRY["bgcargo"] is bgcargo.BGCArgoAdapter
    ad = bgcargo.BGCArgoAdapter()
    assert ad.C == 96 and ad.family == "1gf" and ad.time_dtype == "int32"
    assert ad.per_year and ad.first_year == 2002
    assert ad.channel_names[0] == "DOXY_10"
    assert ad.channel_names[-1] == "DOWNWELLING_PAR_1900"


def test_smoke_all_stages_and_the_probe(tmp_path):
    res = b1.run_smoke("bgcargo", root=str(tmp_path / "s"), keep=True)
    assert len(res["truth"]) == TRUTH_ROWS
    p = res["probe"]
    assert p["rows"] == PROBE_ROWS
    assert p["rows_per_day"][:3] == [5, 6, 0]
    assert p["distinct_platforms"] == 3
    c = p["counts"]
    assert c["floats_listed"] == 4 and c["floats_no_bgc_param"] == 1
    assert c["floats_read"] == 3
    # each file holds 5 written days x 2 profiles; float 1 has 2 extra
    assert c["profiles_in_files"] == 3 * 10 + 2
    assert c["profiles_other_year"] == 3 * 3 * 2          # 2021-12-29..31
    assert c["profiles_duplicate"] == 1
    assert c["profiles_bad_time_qc"] == 1
    assert c["profiles_no_position"] == 1
    assert c["profiles_no_level"] == 0
    assert c["profiles_kept"] == PROBE_ROWS
    # the adjusted array is read for A/D, the raw one for R — never mixed
    assert c["source_used"] == {
        "DOXY adjusted": 4, "NITRATE adjusted": 4,
        "PH_IN_SITU_TOTAL raw": 4, "DOXY raw": 4, "CHLA adjusted": 7,
        "DOWNWELLING_PAR raw": 5}
    # BBP700's raw QC 0 is dropped whole; one DOXY profile's QC 3 too
    assert c["samples_bad_qc"]["BBP700"] == 5 * 171
    assert c["samples_bad_qc"]["DOXY"] == 171 + 10
    assert c["out_of_bounds"] == {f"NITRATE_{q}": 1
                                  for q in (10, 30, 50, 100, 150, 200)}
    assert sum(p["out_of_bounds_stored"].values()) == 0
    # bit 0 DOXY, bit 1 CHLA, bit 3 NITRATE came from ADJUSTED
    assert p["qc"] == {"9": 4, "2": 7}
    assert p["nan_fraction"]["BBP700_10"] == 1.0
    assert p["bytes_fetched"] > 0 and p["schema_version"] == 2


def test_the_stages_and_the_store(tmp_path):
    import argparse
    ad = bgcargo.BGCArgoAdapter()
    src = str(tmp_path / "src")
    lo, hi = (b10.parse_date(x) for x in ad.smoke_window)
    truth = ad.smoke_sources(src, lo, hi)
    a = argparse.Namespace(
        store="bgcargo", work=str(tmp_path / "w"), source_dir=src,
        start=ad.smoke_window[0], end=ad.smoke_window[1], stage="all",
        force=False, attempts=1, qc_keep=2,
        check_chunk_rows=b10.CHECK_CHUNK_ROWS, assemble="auto",
        parts_from_hub=False, push_parts=False, allow_missing_years=False)
    ctx = b10.Ctx(a, adapter=ad, layout=b1.layout_for(ad))
    b10.run_stages(ctx, ["index", "fetch", "assemble"], stage_fn=b1.STAGE_FN,
                   deps=b1.DEPS)
    b10.check_smoke(ctx, truth)
    out = b1.stage_check(ctx)
    assert out["N"] == TRUTH_ROWS
    plan = json.load(open(os.path.join(ctx.root, "plan.json")))
    assert plan["index"]["floats"] == 4
    assert plan["index"]["rows_no_date"] == 1
    assert plan["per_year"]["2021"]["floats"] == 3
    assert plan["per_year"]["2022"]["floats_no_bgc_param"] == 1
    assert plan["first_record"]["format"] == "NETCDF4_CLASSIC"
    m = json.load(open(os.path.join(ctx.store, "store.json")))
    assert m["per_year"] == {"2021": 12, "2022": 11}
    st = f10.Store(ctx.store)
    assert st.N == TRUTH_ROWS


def test_the_index_is_refused_when_empty_or_reshaped():
    def gz(text):
        return gzip.compress(text.encode())
    with pytest.raises(bgcargo.FormatError, match="not a whole gzip"):
        bgcargo.parse_index(b"nope")
    with pytest.raises(bgcargo.FormatError, match="header"):
        bgcargo.parse_index(gz("# c\nfile,date\n"))
    with pytest.raises(bgcargo.FormatError, match="empty listing"):
        bgcargo.parse_index(gz("# c\n" + bgcargo.INDEX_HEADER + "\n"))
    with pytest.raises(bgcargo.FormatError, match="fields"):
        bgcargo.parse_index(gz(bgcargo.INDEX_HEADER + "\na,b\n"))
    idx, meta = bgcargo.parse_index(gz(
        bgcargo.INDEX_HEADER + "\n"
        "aoml/1/profiles/SD1_001.nc,20220501000000,1,1,A,846,AO,"
        "PRES TEMP PSAL DOXY,DDDD,20260101000000\n"
        "aoml/1/profiles/SD1_002.nc,20220611000000,1,1,A,846,AO,"
        "PRES TEMP PSAL,DDD,20260101000000\n"))
    e = idx[2022][("aoml", "1")]
    assert e["profiles"] == 2 and e["months"] == {5: 1, 6: 1}
    assert e["bgc_months"] == {5}
    assert meta["floats"] == 1 and meta["rows_no_date"] == 0


def test_a_missing_sprof_is_an_absence(tmp_path):
    import argparse
    ad = bgcargo.BGCArgoAdapter()
    src = str(tmp_path / "src")
    lo, hi = (b10.parse_date(x) for x in ad.smoke_window)
    ad.smoke_sources(src, lo, hi)
    os.remove(os.path.join(src, "argo", "dac", "csiro", "5905000",
                           "5905000_Sprof.nc"))
    a = argparse.Namespace(
        store="bgcargo", work=str(tmp_path / "w"), source_dir=src,
        start=ad.smoke_window[0], end=ad.smoke_window[1], force=False,
        attempts=1, qc_keep=2, check_chunk_rows=b10.CHECK_CHUNK_ROWS)
    ctx = b10.Ctx(a, adapter=ad, layout=b1.layout_for(ad))
    list(ad.fetch_year(ctx, 2022))
    assert len(ctx.absent) == 1 and "5905000" in ctx.absent[0]["why"]
    assert ctx.absent[0]["unit"] == "2022"


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))

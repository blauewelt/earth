#!/usr/bin/env python3
"""The `xco2` adapter's smoke (family 1.gf, E-082 wave 4) — no network.

    python3 -m pytest -q tests/test_family1_xco2.py

The synthetic archive is written in the GES DISC layout
(`<ShortName>.<version>/<YYYY>/oco[23]_LtCO2_<YYMMDD>_<build>s.nc4`), one Lite
file a day a sensor, with root columns and a `Sounding/` GROUP — because the
real Lite files have both and the adapter's whole format story is that it
RESOLVES each quantity against the file rather than assuming a name table.
Those tests are the important ones here: a required quantity that matches
nothing must take the refusal with the file's own inventory printed.
"""
import argparse
import datetime as dt
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
import family10_store as f10                                    # noqa: E402
from family1 import adapters as fam                             # noqa: E402
from family1.adapters import xco2                               # noqa: E402

# four days x two sensors x (30 soundings - 3 bad quality - 1 without xco2
# - 1 without a position)
FILES = 4 * 2
TRUTH_ROWS = FILES * (xco2.SMOKE_PER_DAY - 5)


def test_registered_and_declared():
    assert fam.REGISTRY["xco2"] is xco2.XCO2Adapter
    ad = xco2.XCO2Adapter()
    assert ad.C == 3 and ad.family == "1gf" and ad.distribution == "public"
    assert ad.channel_names == ["xco2", "xco2_uncertainty", "surface"]
    assert ad.credentials == ("EARTHDATA_USERNAME", "EARTHDATA_PASSWORD")
    # a 1.6 km sounding is BELOW the -4 label, and that is the rule
    assert abs(ad.log2_fp - np.log2(1.6 / 27.83)) < 1e-9 and ad.log2_fp < -4
    assert ad.first_year == 2009           # exception E3 reaches back to GOSAT
    assert {s[0] for s in xco2.SENSORS} == {"OCO-2", "OCO-3", "GOSAT-ACOS"}
    # both OCO-2 collections are listed, newer last so it wins a shared day
    v = [s[3] for s in xco2.SENSORS if s[0] == "OCO-2"]
    assert v == ["11.2r", "11.3r"]


def test_the_layout_is_resolved_and_a_miss_prints_the_inventory():
    inv = {"xco2": {}, "xco2_uncertainty": {}, "latitude": {},
           "longitude": {}, "time": {}, "xco2_quality_flag": {},
           "Sounding/land_water_indicator": {}, "warn_level": {}}
    got = xco2.resolve(inv)
    assert got["surface"] == "Sounding/land_water_indicator"
    assert got["quality"] == "xco2_quality_flag"
    # an OPTIONAL miss is fine
    del inv["Sounding/land_water_indicator"]
    assert "surface" not in xco2.resolve(inv)
    # a REQUIRED miss refuses, and names what the file does hold
    del inv["xco2_uncertainty"]
    with pytest.raises(xco2.FormatError) as e:
        xco2.resolve(inv)
    assert "xco2_uncertainty" in str(e.value) and "latitude" in str(e.value)


def test_the_granule_name_date():
    assert xco2.NAME.search("oco2_LtCO2_190601_B11210Ar_240910224028s.nc4") \
        .group(1) == "190601"
    assert xco2.NAME.search("acos_LtCO2_090418_B11000Ar_x.nc4").group(1) == \
        "090418"
    assert xco2.NAME.search("README.txt") is None


def test_smoke_all_stages_and_the_probe(tmp_path):
    res = b1.run_smoke("xco2", root=str(tmp_path / "s"), keep=True)
    assert len(res["truth"]) == TRUTH_ROWS
    p = res["probe"]
    assert p["rows"] == TRUTH_ROWS
    assert p["distinct_platforms"] == 2
    assert p["counts_scope"] == "month"
    c = p["counts"]
    assert c["files_wanted"] == FILES and c["files_read"] == FILES
    assert c["soundings_in_file"] == FILES * xco2.SMOKE_PER_DAY
    assert c["soundings_bad_quality"] == FILES * 3
    assert c["soundings_no_xco2"] == FILES
    assert c["soundings_no_position"] == FILES
    assert c["soundings_kept"] == TRUTH_ROWS
    assert c["soundings_per_sensor"] == {"OCO-2": TRUTH_ROWS // 2,
                                         "OCO-3": TRUTH_ROWS // 2}
    # one out-of-bounds xco2 per sensor on SMOKE_OOB_DAY, NaN and counted
    assert c["out_of_bounds"] == {"xco2": 2}
    # the RESOLUTION IS RECORDED, as a tally rather than one entry a file
    assert c["resolution"]["surface=Sounding/land_water_indicator"] == FILES
    assert c["quality_flag_meanings"] == {"good bad": FILES}


def test_the_store_the_plan_and_the_inventory(tmp_path):
    ad = xco2.XCO2Adapter()
    src = str(tmp_path / "src")
    lo = b10.parse_date(ad.smoke_window[0])
    hi = b10.parse_date(ad.smoke_window[1])
    truth = ad.smoke_sources(src, lo, hi)
    a = argparse.Namespace(
        store="xco2", work=str(tmp_path / "w"), source_dir=src,
        start=str(lo), end=str(hi), stage="all", force=False, attempts=1,
        qc_keep=2, check_chunk_rows=b10.CHECK_CHUNK_ROWS, assemble="auto",
        parts_from_hub=False, push_parts=False, allow_missing_years=False)
    ctx = b10.Ctx(a, adapter=ad, layout=b1.layout_for(ad))
    b10.run_stages(ctx, ["index", "fetch", "assemble"], stage_fn=b1.STAGE_FN,
                   deps=b1.DEPS)
    b10.check_smoke(ctx, truth)
    out = b1.stage_check(ctx)
    assert out["N"] == TRUTH_ROWS
    plan = json.load(open(os.path.join(ctx.root, "plan.json")))
    assert plan["files"] == FILES
    assert plan["files_per_sensor"] == {"OCO-2": 4, "OCO-3": 4}
    # THE INVENTORY: what makes one runner round trip enough for a format
    # nothing can read anonymously
    inv = plan["first_file"]["inventory"]
    assert "Sounding/land_water_indicator" in inv
    assert inv["xco2"]["units"] == "ppm"
    assert plan["first_file"]["resolution"]["time"] == "time"
    assert plan["footprint_per_platform"]["OCO-3"] == \
        plan["footprint_per_platform"]["OCO-2"]
    m = json.load(open(os.path.join(ctx.store, "store.json")))
    assert m["sensors"] == ["OCO-2", "OCO-3"]
    assert m["footprint_per_platform"]["OCO-2"] < -4
    # the out-of-bounds xco2 is NaN, never clipped to 500
    st = f10.Store(ctx.store)
    v = np.asarray(st["values"], np.float32)
    assert int(np.isnan(v[:, 0]).sum()) == 2
    assert float(np.nanmax(v[:, 0])) <= 500.0
    # `surface` really is 0/1 and `qc`'s water bit agrees with it
    water = v[:, 2] >= 0.5
    qc = np.asarray(st["qc"], np.uint8)
    assert np.array_equal(water, (qc & xco2.QC_WATER_BIT) > 0)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))

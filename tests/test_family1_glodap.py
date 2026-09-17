#!/usr/bin/env python3
"""The `glodap` adapter's smoke (family 1.gf, E-082) — no network.

    python3 -m pytest -q tests/test_family1_glodap.py

The synthetic archive (`glodap.make_smoke_sources`) is written in the real
layout: the NCEI OCADS accession listing (with an older version's CSV, the
.mat, the .nc and a per-basin CSV as decoys) and a merged master CSV with the
real 119-column GLODAPv3 header, ordered by cruise. Expected counts are exact.
"""
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
from family1.adapters import glodap                             # noqa: E402

# three cruises x four window days x three bottles, less one bottle without
# depth (Jan 1) and one with no flag-2 value (Jan 2)
TRUTH_ROWS = 3 * 4 * 3 - 2
PROBE_ROWS = 3 * 2 * 3 - 2


def test_registered():
    assert fam.REGISTRY["glodap"] is glodap.GLODAPAdapter
    ad = glodap.GLODAPAdapter()
    assert ad.C == 13 and ad.family == "1gf" and ad.time_dtype == "int32"
    assert not ad.per_year and ad.first_year == 1972
    assert ad.channel_names == ["depth", "temperature", "salinity", "oxygen",
                                "nitrate", "phosphate", "silicate", "dic",
                                "ta", "ph", "cfc11", "cfc12", "sf6"]
    assert len(glodap._header().split(",")) == 119


def test_smoke_all_stages_and_the_probe(tmp_path):
    res = b1.run_smoke("glodap", root=str(tmp_path / "s"), keep=True)
    assert len(res["truth"]) == TRUTH_ROWS
    p = res["probe"]
    assert p["rows"] == PROBE_ROWS
    assert p["rows_per_day"][:3] == [8, 8, 0]
    assert p["distinct_platforms"] == 3
    c = p["counts"]
    # 3 cruises x 5 written days x 3 bottles, plus the short line
    assert c["lines"] == 45 + 1
    assert c["lines_bad_length"] == 1
    assert c["bottles_no_depth"] == 1
    assert c["bottles_no_value"] == 1
    assert c["time_of_day_missing"] == 1
    assert c["value_flag0"] == {"oxygen": 1}
    assert c["out_of_bounds"] == {"phosphate": 1}
    assert c["bottles_outside_window"] == 3 * 3 * 3     # Dec 29-31
    assert p["qc"] == {"0": PROBE_ROWS - 2, "1": 1, "2": 1}
    assert c["file"] == "GLODAPv3_Merged_Master_File.csv"
    assert p["counts_scope"] == "archive"
    assert p["bytes_fetched"] > 0 and p["schema_version"] == 2


def test_the_stages_and_the_store(tmp_path):
    import argparse
    ad = glodap.GLODAPAdapter()
    src = str(tmp_path / "src")
    lo, hi = (b10.parse_date(x) for x in ad.smoke_window)
    truth = ad.smoke_sources(src, lo, hi)
    a = argparse.Namespace(
        store="glodap", work=str(tmp_path / "w"), source_dir=src,
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
    assert plan["version"] == "3" and plan["columns"] == 119
    assert plan["first_record"]["values"][0] == 5.0
    m = json.load(open(os.path.join(ctx.store, "store.json")))
    assert m["per_year"] == {"1999": 18, "2000": 16}
    # the detection-limit nitrate is kept, the flag-0 oxygen is not
    st = f10.Store(ctx.store)
    v = np.asarray(st["values"], np.float32)
    assert (np.round(v[:, 4], 2) == -0.04).sum() == 1


def test_the_newest_version_is_chosen_and_an_empty_listing_refused():
    html = ('<a href="GLODAPv2.2023_Merged_Master_File.csv">x</a>'
            '<a href="GLODAPv3_Merged_Master_File.csv">x</a>'
            '<a href="GLODAPv2.2022_Merged_Master_File.csv">x</a>')
    assert glodap.parse_listing(html) == \
        ("GLODAPv3_Merged_Master_File.csv", "3")
    with pytest.raises(glodap.FormatError, match="empty listing"):
        glodap.parse_listing('<a href="GLODAPv3_Merged_Master_File.mat">x</a>')
    with pytest.raises(glodap.FormatError, match="lacks"):
        glodap.Parser("expocode,year,month")


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))

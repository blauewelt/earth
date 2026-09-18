#!/usr/bin/env python3
"""The `swot` adapter's smoke (family 1.gf, E-082 wave 4) — no network.

    python3 -m pytest -q tests/test_family1_swot.py

`swot` is a MEASUREMENT, not yet a build: the storage decision (tier-P rows
against a per-pass sharded tile group) has not been made, so every stage past
`probe` refuses unless `SWOT_ALLOW_BUILD=1`, and the probe computes both
options from what it measured. These tests pin the refusal, the granule-name
parser, the quality grading and the probe's `storage_options` arithmetic
against a synthetic archive in the product's own layout.
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
from family1 import adapters as fam                             # noqa: E402
from family1.adapters import swot                               # noqa: E402

# Three passes x 12 lines x 8 pixels = 288 pixels. Dropped: the two nadir-gap
# pixels of every line (3 x 12 x 2 = 72) and the whole line graded `bad`
# (3 x 1 x 8 = 24, of which 2 are already in the gap) -> 288 - 72 - 18 = 198.
TRUTH_ROWS = 198


def test_registered_and_declared():
    assert fam.REGISTRY["swot"] is swot.SWOTAdapter
    ad = swot.SWOTAdapter()
    assert ad.C == 3 and ad.family == "1gf" and ad.distribution == "public"
    assert ad.channel_names == ["ssha_karin", "sig0_karin",
                               "ssh_karin_uncert"]
    assert ad.credentials == ("EARTHDATA_USERNAME", "EARTHDATA_PASSWORD")
    # 2 km pixels: the ledger's (-3.80, -4)
    assert abs(ad.log2_fp - (-3.798)) < 1e-3 and ad.log2_dt == -4.0
    assert "PROBE ONLY" in ad.notes


def test_a_build_refuses_before_its_first_byte(monkeypatch):
    """ml/CLAUDE.md §0.3 — the guard is at dispatch, where the inputs are all
    it has cost."""
    monkeypatch.delenv("SWOT_ALLOW_BUILD", raising=False)
    ad = swot.SWOTAdapter()
    a = argparse.Namespace(stage="all", source_dir="")
    ctx = argparse.Namespace(a=a, source_dir="")
    with pytest.raises(SystemExit, match="MEASUREMENT until the storage"):
        ad.fetch_preflight(ctx)
    # a probe is allowed, and so is SWOT_ALLOW_BUILD=1
    assert ad.fetch_preflight(
        argparse.Namespace(a=argparse.Namespace(stage="probe"),
                           source_dir="")) is None
    monkeypatch.setenv("SWOT_ALLOW_BUILD", "1")
    assert swot.SWOTAdapter().fetch_preflight(ctx) is None


def test_the_granule_name_parser():
    def e(cycle, pss, extra=""):
        n = (f"SWOT_L2_LR_SSH_Expert_{cycle}_{pss}_20240125T001931_"
             f"20240125T011059_PGD0_02_swot{extra}")
        return {"title": n, "granule_size": "32.4",
                "time_start": "2024-01-25T00:19:17.011Z",
                "time_end": "2024-01-25T01:11:13.000Z",
                "links": [{"href": "https://x/podaac-swot-ops-cumulus-public/"
                                   f"{n}.nc"},
                          {"href": "https://x/podaac-swot-ops-cumulus-"
                                   f"protected/{n}.nc"}]}
    gran, counts = swot.parse_granules([e("010", "001"), e("010", "002"),
                                        {"title": "README", "links": []}])
    assert sorted(gran) == [("010", "001"), ("010", "002")]
    assert counts["granules_other_name"] == 1
    assert "protected" in gran[("010", "001")]["url"]
    assert gran[("010", "001")]["bytes"] == 32400000
    with pytest.raises(swot.FormatError, match="two granules"):
        swot.parse_granules([e("010", "001"), e("010", "001")])
    # a granule with no protected .nc link is a refusal, not a skip
    bad = e("010", "003")
    bad["links"] = [{"href": "https://x/podaac-swot-ops-cumulus-public/a.nc"}]
    with pytest.raises(swot.FormatError, match="no protected"):
        swot.parse_granules([bad])


def test_the_quality_vocabulary_comes_from_the_file():
    class V:
        flag_meanings = ("suspect_large_ssh_delta degraded_media_delay "
                         "bad_outside_of_range good_measurement")
    assert swot.qc_grade(V()) == ["good", "suspect", "degraded", "bad"]

    class Bad:
        flag_meanings = "some_other_condition"
    with pytest.raises(swot.FormatError, match="refusing to guess"):
        swot.qc_grade(Bad())


def test_smoke_all_stages_and_the_probe(tmp_path):
    res = b1.run_smoke("swot", root=str(tmp_path / "s"), keep=True)
    assert len(res["truth"]) == TRUTH_ROWS
    p = res["probe"]
    assert p["rows"] == TRUTH_ROWS
    assert p["distinct_platforms"] == 1
    assert p["counts_scope"] == "month"
    c = p["counts"]
    assert c["passes_read"] == len(swot.SMOKE_PASSES)
    assert c["pixels_in_pass"] == \
        len(swot.SMOKE_PASSES) * swot.SMOKE_LINES * swot.SMOKE_PIXELS
    assert c["pixels_valid"] == TRUTH_ROWS
    assert c["pass_geometry"] == \
        {f"{swot.SMOKE_LINES}x{swot.SMOKE_PIXELS}": len(swot.SMOKE_PASSES)}
    # the whole line graded `bad` is dropped, the suspect line is kept
    assert c["qual_grade"]["bad"] == len(swot.SMOKE_PASSES) * \
        swot.SMOKE_PIXELS
    assert c["qual_grade"]["suspect"] == len(swot.SMOKE_PASSES) * \
        swot.SMOKE_PIXELS
    assert p["qc"]["0"] > 0 and p["qc"]["1"] == \
        len(swot.SMOKE_PASSES) * (swot.SMOKE_PIXELS - 2)
    assert c["out_of_bounds"] == {"ssha_karin": 1}
    # THE DECISION MATERIAL: both options, computed from the measurement
    so = c["storage_options"]
    assert isinstance(so, list) and len(so) == 1
    so = so[0]
    assert so["measured"]["passes_read"] == len(swot.SMOKE_PASSES)
    assert so["tier_p"]["bytes_per_row"] == 33
    assert so["tier_g_per_pass_group"]["groups"] == 584
    assert abs(so["ratio_p_over_g"] - 33 / 7.2) < 1e-3
    assert so["tier_p"]["bytes_per_year"] > \
        so["tier_g_per_pass_group"]["bytes_per_year"]


def test_storage_options_is_arithmetic_not_a_guess():
    counts = {"passes_read": 4, "pixels_in_pass": 4_000_000,
              "pixels_valid": 1_000_000}
    so = swot.storage_options(counts, 3)
    assert so["measured"]["pixels_per_pass"] == 1_000_000.0
    assert so["measured"]["valid_pixels_per_pass"] == 250_000.0
    assert so["measured"]["valid_fraction"] == 0.25
    # the REPORTED passes_per_year is rounded to one decimal; the byte
    # arithmetic uses the exact value, so the test recomputes the exact one
    py = 584 * 365.25 / 20.86
    assert abs(so["extrapolation_basis"]["passes_per_year"] - py) < 0.1
    assert so["tier_p"]["bytes_per_year"] == round(250_000 * py * 33)
    assert so["tier_g_per_pass_group"]["bytes_per_year"] == \
        round(250_000 * py * 7.2)
    assert swot.storage_options({"passes_read": 0}, 3) is None


def test_the_index_plan(tmp_path):
    ad = swot.SWOTAdapter()
    src = str(tmp_path / "src")
    lo = b10.parse_date(ad.smoke_window[0])
    hi = b10.parse_date(ad.smoke_window[1])
    ad.smoke_sources(src, lo, hi)
    a = argparse.Namespace(
        store="swot", work=str(tmp_path / "w"), source_dir=src,
        start=str(lo), end=str(hi), stage="index", force=False, attempts=1,
        qc_keep=2, check_chunk_rows=b10.CHECK_CHUNK_ROWS, assemble="auto",
        parts_from_hub=False, push_parts=False, allow_missing_years=False)
    ctx = b10.Ctx(a, adapter=ad, layout=b1.layout_for(ad))
    b10.run_stages(ctx, ["index"], stage_fn=b1.STAGE_FN, deps=b1.DEPS)
    plan = json.load(open(os.path.join(ctx.root, "plan.json")))
    assert plan["granules"] == len(swot.SMOKE_PASSES)
    assert plan["collection"] == swot.COLLECTION
    assert plan["probe_only"] is True
    assert plan["cycle_requested"] == swot.SMOKE_CYCLE


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))

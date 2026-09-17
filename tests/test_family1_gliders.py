#!/usr/bin/env python3
"""The `gliders` adapter's smoke (family 1.0.tf, E-082) — no network.

    python3 -m pytest -q tests/test_family1_gliders.py

The synthetic DAC (`gliders.make_smoke_sources`) is the IOOS Glider DAC's
real layout: ERDDAP's allDatasets.csv and per-dataset info CSV, the THREDDS
catalog tree (users -> deployments -> one `<id>.nc3.nc`), and NetCDF-3
aggregates in the (trajectory=1, profile, obs) layout written by
`_nc3.write`. Counts are exact.
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
import build_family8_argo as f8                                 # noqa: E402
import family10_store as f10                                    # noqa: E402
from family1 import adapters as fam                             # noqa: E402
from family1.adapters import _nc3, gliders                      # noqa: E402

# window 2023-06-01 .. 06-04: 32 profiles (every 3 h) per deployment, three
# deployments read (alpha delayed, bravo real-time, charlie real-time), minus
# alpha's profile without a position, its duplicated time and its profile
# with no usable sample: 96 - 3 = 93. The probe month holds all of them.
TRUTH_ROWS = 93


def ns(tmp, src, **over):
    ad = gliders.GlidersAdapter()
    d = dict(store="gliders", work=str(tmp / "w"), source_dir=src,
             start=ad.smoke_window[0], end=ad.smoke_window[1], stage="all",
             force=False, attempts=1, qc_keep=2,
             check_chunk_rows=b10.CHECK_CHUNK_ROWS, assemble="auto",
             parts_from_hub=False, push_parts=False,
             allow_missing_years=False)
    d.update(over)
    return argparse.Namespace(**d)


def test_registered_with_family8s_levels():
    ad = gliders.GlidersAdapter()
    assert fam.REGISTRY["gliders"] is gliders.GlidersAdapter
    assert ad.C == 64 and ad.time_dtype == "int32" and not ad.per_year
    assert gliders.LEVELS is f8.LEVELS and len(gliders.LEVELS) == 16
    assert ad.channel_names[:2] == ["T_10", "T_30"]
    assert ad.channel_names[16] == "S_10" and ad.channel_names[-1] == \
        "CHL_1900"
    assert ad.log2_fp == -4.0


def test_smoke_all_stages_and_the_probe(tmp_path):
    res = b1.run_smoke("gliders", root=str(tmp_path / "s"), keep=True)
    assert len(res["truth"]) == TRUTH_ROWS
    p = res["probe"]
    assert p["rows"] == TRUTH_ROWS and p["distinct_platforms"] == 3
    # all three specials are on June 1 (06, 12/15 and 21 UTC)
    assert p["rows_per_day"][:4] == [24 - 3, 24, 24, 24]
    assert sum(p["out_of_bounds_stored"].values()) == 0
    c = p["counts"]
    assert c["datasets_listed"] == 6 and c["deployments_in_window"] == 3
    assert c["delayed_used"] == 1 and c["realtime_used"] == 2
    assert c["realtime_used_delayed_missing"] == 1
    assert c["deployments_read"] == 3
    assert c["profiles_in_file"] == 3 * 40
    assert c["profiles_outside_window"] == 3 * 8
    assert c["profiles_no_time_or_position"] == 1
    assert c["profiles_duplicate_time"] == 1
    assert c["profiles_no_level"] == 1
    assert c["qc"] == {"1": 29, "0": 64}
    assert c["units_unrecognized"] == {"CHL 'micrograms m^-3'": 1}
    assert c["variables"] == {"o2:dissolved_oxygen": 2,
                              "o2:oxygen_concentration": 1,
                              "chl:chlorophyll_a": 2}
    assert c["qartod_removed"]["temperature"] > 0
    assert c["samples_default_fill"] == 1
    assert c["layouts"] == {"trajectory_1": 2, "profile_1": 1}
    assert p["platforms"]["probe_platforms_without_entry"] == 0
    m = json.load(open(os.path.join(res["work"], "gliders", "gliders",
                                    "store.json")))
    assert m["N"] == TRUTH_ROWS and m["platforms"]["in_store"] == 3
    pj = json.load(open(os.path.join(res["work"], "gliders", "gliders",
                                     "platforms.json")))
    alpha = [v for v in pj.values() if v["id"] == "alpha-20230530T0000"][0]
    assert sorted(alpha["datasets"]) == ["alpha-20230530T0000",
                                         "alpha-20230530T0000-delayed"]
    plan = json.load(open(os.path.join(res["work"], "gliders", "plan.json")))
    assert plan["thredds_dirs_not_in_erddap"] == 1
    assert plan["first_record"]["dataset"] == "alpha-20230530T0000-delayed"
    # charlie's chlorophyll unit was refused: its CHL columns are all NaN
    st = f10.Store(os.path.join(res["work"], "gliders", "gliders"))
    v = np.asarray(st["values"], np.float32)
    ch = st["platform"] == b10.platform_hash("charlie-20230515T0000")
    assert ch.sum() == 32 and np.isnan(v[ch][:, 48:]).all()
    assert np.isfinite(v[ch][:, :16]).any()


def test_oxygen_per_volume_is_converted_with_the_samples_density(tmp_path):
    src = str(tmp_path / "src")
    ad = gliders.GlidersAdapter()
    lo, hi = (b10.parse_date(x) for x in ad.smoke_window)
    truth = ad.smoke_sources(src, lo, hi)
    bravo = [r for r in truth if r["platform"] ==
             b10.platform_hash("bravo-20230601T0000")]
    assert len(bravo) == 32
    # 230 - p/10 umol/L over 1025 + p/200 kg/m3: at 10 dbar ≈ 223.4 umol/kg
    # (the 10 dbar level may be a first sample up to 20 dbar deep: ±2 umol)
    o2_10 = [r["v"][32] for r in bravo if np.isfinite(r["v"][32])]
    assert o2_10 and all(abs(x - 229 * 1000 / 1025.05) < 2.5 for x in o2_10)


def test_variable_choice_and_units():
    info = gliders.parse_info(gliders._info([
        ("pressure", {"units": "dbar"}),
        ("temperature", {"units": "Celsius"}),
        ("salinity", {"units": "1"}),
        ("sci_oxy4_oxygen", {"units": "umol kg-1",
                             "standard_name": gliders.O2_MASS}),
        ("dissolved_oxygen", {"units": "umol kg-1",
                              "standard_name": gliders.O2_MASS}),
        ("chlorophyll", {"units": "mg m-3",
                         "standard_name": gliders.CHL_SN[1]}),
    ]))
    hv = {"pressure", "temperature", "salinity", "sci_oxy4_oxygen",
          "dissolved_oxygen", "chlorophyll"}
    c = {}
    ch, why = gliders.choose(info, hv, c)
    assert why is None and ch["o2"] == "dissolved_oxygen"
    assert ch["o2_kind"] == "o2_mass" and ch["chl"] == "chlorophyll"
    ch, why = gliders.choose(info, hv - {"dissolved_oxygen"}, c)
    assert ch["o2"] == "sci_oxy4_oxygen"
    info["pressure"]["units"] = "m"
    ch, why = gliders.choose(info, hv, c)
    assert ch is None and "pressure" in why
    info["pressure"]["units"] = "dbar"
    ch, why = gliders.choose(info, hv - {"salinity"}, c)
    assert ch is None and "salinity" in why
    info["dissolved_oxygen"]["units"] = ""
    info["sci_oxy4_oxygen"]["units"] = ""
    c = {}
    ch, why = gliders.choose(info, hv, c)
    assert "o2" not in ch and c["units_unrecognized"] == {"O2 ''": 1}


def test_the_dac_list_and_catalog_parsers():
    with pytest.raises(gliders.FormatError, match="ZERO"):
        gliders.parse_all(",".join(gliders.ALL_FIELDS) + "\n,,,\n"
                          "allDatasets,x,,,NaN,NaN,NaN,NaN,Other\n")
    with pytest.raises(gliders.FormatError, match="header"):
        gliders.parse_all("datasetID\nx\n")
    real = ('<dataset name="angus-20200319T0000.nc3.nc" ID="deployments/'
            'secoora/angus-20200319T0000/angus-20200319T0000.nc3.nc" urlPath='
            '"deployments/secoora/angus-20200319T0000/angus-20200319T0000.nc3'
            '.nc">\n      <dataSize units="Mbytes">104.2</dataSize>\n')
    got = gliders.DATASET_FILE.findall(real)
    assert got == [("angus-20200319T0000.nc3.nc",
                    "deployments/secoora/angus-20200319T0000/"
                    "angus-20200319T0000.nc3.nc", "Mbytes", "104.2")]


def test_read_part_reads_only_the_asked_profiles(tmp_path):
    p = str(tmp_path / "a.nc")
    x = np.arange(1 * 5 * 3, dtype=np.float32).reshape(1, 5, 3)
    _nc3.write(p, [("trajectory", 1), ("profile", 5), ("obs", 3)],
               [("v", ("trajectory", "profile", "obs"), x, {})])
    seen = []
    r = _nc3.NC3(_nc3.file_reader(p, counter=seen.append))
    header = sum(seen)
    part = r.read_part("v", 1, 2, 4)
    assert np.array_equal(part, x[0, 2:4])
    assert sum(seen) - header == 2 * 3 * 4


def test_a_truncated_aggregate_is_refused(tmp_path):
    src = str(tmp_path / "src")
    ad = gliders.GlidersAdapter()
    lo, hi = (b10.parse_date(x) for x in ad.smoke_window)
    ad.smoke_sources(src, lo, hi)
    f = os.path.join(src, "gliders", "thredds", "secoora",
                     "bravo-20230601T0000", "bravo-20230601T0000.nc3.nc")
    raw = open(f, "rb").read()
    open(f, "wb").write(raw[:len(raw) // 3])
    ctx = b10.Ctx(ns(tmp_path, src), adapter=ad, layout=b1.layout_for(ad))
    with pytest.raises(gliders.FormatError, match="bravo"):
        list(ad.fetch_stream(ctx))


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))

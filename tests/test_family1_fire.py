#!/usr/bin/env python3
"""The `fire` adapter's smoke (family 1.0.tf, FIRMS active fire) — no network.

    python3 -m pytest -q tests/test_family1_fire.py

The synthetic archive (`fire.make_smoke_sources`) is the real shape: a
`data_availability` CSV with a `data_id,min_date,max_date` header, and one CSV
per five-day window in the area API's own column order — the MODIS one with
`brightness`, `bright_t31` and an integer confidence, the VIIRS one with
`bright_ti4`, `bright_ti5` and a letter class. The windows are
`FireAdapter.windows()`' own answer, so the archive and the fetch cannot
disagree about where the Standard-Processing / Near-Real-Time boundary falls:
MODIS_SP ends 2023-08-20 and MODIS_NRT starts 2023-08-21, which is exactly the
seam the adapter has to get right.
"""
import argparse
import datetime as dt
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
from family1.adapters import fire as fr                         # noqa: E402

ENV = ("FIRE_INSTRUMENTS", "FIRMS_MAP_KEY")


def clear_env(monkeypatch):
    for k in ENV:
        monkeypatch.delenv(k, raising=False)


@pytest.fixture(scope="module")
def smoke(tmp_path_factory):
    root = str(tmp_path_factory.mktemp("fire_smoke"))
    keep = {k: os.environ.get(k) for k in ENV}
    try:
        return b1.run_smoke("fire", root=root, keep=True)
    finally:
        for k, v in keep.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


# ================================================================ facts ====
def test_registered_as_a_tier_p_store(monkeypatch):
    clear_env(monkeypatch)
    assert fam.REGISTRY["fire"] is fr.FireAdapter
    ad = fr.FireAdapter()
    assert not b1.is_grid(ad)
    assert ad.family == "1tf" and ad.distribution == "public"
    assert ad.credentials == ("FIRMS_MAP_KEY",)
    assert ad.channel_names == ["brightness", "frp", "confidence",
                                "daynight"]
    assert ad.C == 4 and ad.time_dtype == "int32"
    assert ad.platform_meta is True
    assert ad.first_year == 2000
    # the TRUE support of a 1 km pixel, below E-078's -4 point label
    assert abs(ad.log2_fp - (-4.798)) < 1e-3
    assert ad.log2_dt == -4.0
    assert sorted(ad.instruments) == ["modis", "noaa20", "noaa21", "snpp"]


def test_the_platform_table_covers_every_spelling_and_hashes_the_canonical():
    """One satellite is ONE platform however the archive spells it."""
    ad = fr.FireAdapter()
    table = ad.platforms(None)
    assert len(table) == len(fr.PLATFORM_LIST) == 5
    for (name, inst, code, chain, spellings) in fr.PLATFORM_LIST:
        h = b10.platform_hash(f"{name} {inst}")
        assert h in table
        e = table[h]
        assert e["satellite"] == name and e["instrument"] == inst
        assert e["qc_code"] == code
        assert sorted(e["spellings_accepted"]) == sorted(spellings)
        for s in spellings:
            canon, c2, ch2 = fr.PLATFORMS[(s.upper(), inst.upper())]
            assert canon == name and c2 == code and ch2 == chain
    # the two footprints, both measured and both below -4
    fp = {e["satellite"]: e["log2_fp"] for e in table.values()}
    assert abs(fp["Terra"] - (-4.798)) < 1e-3
    assert abs(fp["NOAA-20"] - (-6.214)) < 1e-3


def test_a_modis_window_and_a_viirs_window_are_parsed_by_column_name():
    day = dt.date(2023, 8, 1)
    raw = fr.window_csv("MODIS_SP", day, 1).encode()
    cols, counts = fr.parse_csv(raw, {day}, "modis", False, {})
    assert cols["t"].size == 2
    # MODIS: the confidence PERCENTAGE is in the channel, qc has no class
    assert np.isfinite(cols["values"][:, 2]).all()
    assert ((cols["qc"] >> 3) & 0b11 == 0).all()
    assert sorted(int(q & 0b111) for q in cols["qc"]) == [1, 2]
    raw = fr.window_csv("VIIRS_SNPP_SP", day, 1).encode()
    cols, counts = fr.parse_csv(raw, {day}, "snpp", True, {})
    assert cols["t"].size == 3
    # VIIRS: the channel is EMPTY and the class is in qc
    assert np.isnan(cols["values"][:, 2]).all()
    assert sorted(int((q >> 3) & 0b11) for q in cols["qc"]) == [1, 2, 3]
    assert ((cols["qc"] & 0b111) == 3).all()
    assert ((cols["qc"] >> 5) & 1 == 1).all()          # the NRT bit
    # day/night is a flag, and the acquisition time really is in the second
    assert set(np.unique(cols["values"][:, 3])) <= {0.0, 1.0}


def test_a_row_outside_the_requested_days_is_dropped_and_counted():
    day = dt.date(2023, 8, 1)
    raw = fr.window_csv("MODIS_SP", day, 5).encode()
    cols, counts = fr.parse_csv(raw, {day}, "modis", False, {})
    assert cols["t"].size == 2                          # only day 1 kept
    assert counts["rows_outside_requested_days"] == 8


def test_an_unknown_satellite_and_an_unknown_class_are_refusals():
    day = dt.date(2023, 8, 1)
    good = fr.window_csv("MODIS_SP", day, 1)
    bad = good.replace("Terra", "Sentinel-9")
    with pytest.raises(fr.FormatError) as e:
        fr.parse_csv(bad.encode(), {day}, "modis", False, {})
    assert "platform table" in str(e.value)
    v = fr.window_csv("VIIRS_SNPP_SP", day, 1).replace(",n,", ",medium,")
    with pytest.raises(fr.FormatError) as e:
        fr.parse_csv(v.encode(), {day}, "snpp", False, {})
    assert "l / n / h" in str(e.value)
    # a VIIRS row answered by a MODIS source would be stored twice
    with pytest.raises(fr.FormatError) as e:
        fr.parse_csv(fr.window_csv("VIIRS_SNPP_SP", day, 1).encode(),
                     {day}, "modis", False, {})
    assert "twice" in str(e.value)
    # a body that is not a FIRMS CSV at all
    with pytest.raises(fr.FormatError):
        fr.parse_csv(b"<html>rate limited</html>\n", {day}, "modis", False,
                     {})
    with pytest.raises(fr.FormatError):
        fr.parse_csv(b"", {day}, "modis", False, {})


def test_a_gap_between_sp_and_nrt_is_a_refusal(tmp_path, monkeypatch):
    """SP and NRT are one record; a hole between them is a listing problem."""
    clear_env(monkeypatch)
    root = str(tmp_path)
    base = os.path.join(root, "fire")
    os.makedirs(base, exist_ok=True)
    with open(os.path.join(base, "data_availability.csv"), "w") as fh:
        fh.write("data_id,min_date,max_date\n"
                 "MODIS_SP,2000-11-01,2025-01-31\n"
                 "MODIS_NRT,2025-02-05,2025-06-06\n")
    monkeypatch.setenv("FIRE_INSTRUMENTS", "modis")
    ad = fr.FireAdapter()
    ctx = fr._SmokeCtx(root)
    with pytest.raises(SystemExit) as e:
        ad.coverage(ctx)
    assert "no source" in str(e.value)


def test_availability_is_read_and_never_assumed(tmp_path, monkeypatch):
    clear_env(monkeypatch)
    root = str(tmp_path)
    base = os.path.join(root, "fire")
    os.makedirs(base, exist_ok=True)
    with open(os.path.join(base, "data_availability.csv"), "w") as fh:
        fh.write("nonsense\n1,2\n")
    monkeypatch.setenv("FIRE_INSTRUMENTS", "modis")
    ad = fr.FireAdapter()
    with pytest.raises(SystemExit) as e:
        ad.availability(fr._SmokeCtx(root))
    assert "data_id" in str(e.value)


def test_instruments_env_refuses_a_name_that_is_not_a_source_family(
        monkeypatch):
    clear_env(monkeypatch)
    monkeypatch.setenv("FIRE_INSTRUMENTS", "terra")
    with pytest.raises(SystemExit) as e:
        fr.FireAdapter()
    assert "the instruments are" in str(e.value)


# ============================================================== the smoke ==
def test_smoke_stores_every_detection_once(smoke):
    truth = smoke["truth"]
    # MODIS gives two detections a day and VIIRS three, over 31 days, with
    # every day served by exactly one source
    assert len(truth) == 31 * 5
    # no duplicate (time, platform, lat) triple: the SP/NRT seam is not
    # fetched twice and MODIS_SP is not asked for once per satellite
    keys = {(r["t"], r["platform"], round(r["lat"], 5)) for r in truth}
    assert len(keys) == len(truth)


def test_the_probe_measured_the_month_and_its_platforms(smoke):
    p = smoke["probe"]
    assert p["rows"] == 31 * 5
    assert p["counts_scope"] == "month"
    assert p["bytes_fetched"] > 0
    assert p["platforms"]["probe_platforms_without_entry"] == 0
    assert p["distinct_platforms"] == 3          # Terra, Aqua, Suomi-NPP
    c = p["counts"]
    assert c["windows"] > 0
    assert sum(c["rows_by_spelling"].values()) == 31 * 5
    assert "Terra/MODIS -> Terra" in c["rows_by_spelling"]
    assert "N/VIIRS -> Suomi-NPP" in c["rows_by_spelling"]


def test_both_sources_of_the_seam_are_in_the_store(smoke):
    """MODIS_SP ends 08-20 and MODIS_NRT starts 08-21; the NRT bit in qc is
    what lets a consumer hold the tail to a different standard."""
    import family10_store as f10
    st = f10.open_store(os.path.join(smoke["work"], "fire", "fire"))
    qc = np.asarray(st["qc"])
    nrt = (qc >> 5) & 1
    assert nrt.any() and (~nrt.astype(bool)).any()
    codes = set(int(x) for x in np.unique(qc & 0b111))
    assert codes == {1, 2, 3}

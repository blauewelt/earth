#!/usr/bin/env python3
"""The `gedi` adapter's smoke (family 1.0.tf, E-082 wave 6) — no network.

    python3 -m pytest -q tests/test_family1_gedi.py

GEDI (the Global Ecosystem Dynamics Investigation) is the laser altimeter that
flew on the International Space Station; this store joins three of its
products — L2A relative heights, L2B cover and plant-area index, L4A biomass
— shot for shot, one row per quality shot. Phase A builds ONE YEAR (2022) and
the hosted probe's numbers decide the rest, so these tests pin the size gate
that enforces that, the granule-key join the three products share, the
quality and join branches, and the probe's own arithmetic against a synthetic
archive written in the products' real group and dataset layout.

THE EXPECTED COUNTS, derived from `gedi.make_smoke_sources` rather than
observed from it: four granule stems (two in 2021-12, two in 2022-01) x two
beam groups x six shots = 48 shots. One shot per beam-granule is not a
quality shot and is dropped (8), leaving 40 rows.
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
from family1.adapters import gedi                               # noqa: E402

STEMS = len(gedi.SMOKE_STEMS)                 # 4
BEAMS = len(gedi.SMOKE_BEAMS)                 # 2
SHOTS = gedi.SMOKE_SHOTS                      # 6
SHOTS_TOTAL = STEMS * BEAMS * SHOTS           # 48
DROPPED = STEMS * BEAMS                       # 8 — one non-quality shot each
TRUTH_ROWS = SHOTS_TOTAL - DROPPED            # 40
MONTH_STEMS = 2                               # stems inside the probe month
PROBE_ROWS = MONTH_STEMS * BEAMS * (SHOTS - 1)   # 20


def test_registered_and_declared():
    assert fam.REGISTRY["gedi"] is gedi.GEDIAdapter
    ad = gedi.GEDIAdapter()
    assert ad.C == 11 and ad.family == "1tf" and ad.distribution == "public"
    # C = 11 is load-bearing: the ledger's row is 27 + 2C = 49 bytes
    assert b10.row_bytes(ad.C, "int32") == 49
    assert ad.channel_names == [
        "elev_lowestmode", "rh25", "rh50", "rh75", "rh90", "rh98",
        "cover", "pai", "agbd", "agbd_se", "sensitivity"]
    assert ad.credentials == ("EARTHDATA_USERNAME", "EARTHDATA_PASSWORD")
    # a 25 m footprint and an instantaneous shot: the ledger's (-10.1, -12)
    assert abs(ad.log2_fp - (-10.1205)) < 1e-3
    assert ad.log2_dt == -12.0
    assert ad.platform_meta is True and ad.first_year == 2019
    assert "ONE YEAR (2022)" in ad.notes
    # every channel's bounds fit float16, which is what `values` is stored as
    lo, hi = ad.bounds()
    assert float(np.max(np.abs(hi))) < 65504 and float(np.min(lo)) > -65504


def test_the_size_gate_refuses_before_its_first_byte(monkeypatch, tmp_path):
    """ml/CLAUDE.md §0.3 — a precondition that depends only on the inputs is
    free at dispatch and expensive at hour three. One month of the three
    products is 2.58 TB native, so a multi-year build is refused here."""
    monkeypatch.delenv("GEDI_ALLOW_BUILD", raising=False)
    monkeypatch.delenv("EARTHDATA_USERNAME", raising=False)
    monkeypatch.delenv("EARTHDATA_PASSWORD", raising=False)
    monkeypatch.setenv("NETRC", str(tmp_path / "absent"))

    def ctx(stage, lo, hi, src=""):
        return argparse.Namespace(
            a=argparse.Namespace(stage=stage, source_dir=src),
            source_dir=src, d_lo=b10.parse_date(lo), d_hi=b10.parse_date(hi))

    with pytest.raises(SystemExit, match="crosses a calendar-year boundary"):
        gedi.GEDIAdapter().fetch_preflight(
            ctx("all", "2022-01-01", "2023-12-31"))
    # ONE year is what phase A builds, so it passes the size gate and stops on
    # the SECOND guard instead: there is no way to authenticate to Earthdata
    with pytest.raises(SystemExit, match="no netrc naming"):
        gedi.GEDIAdapter().fetch_preflight(
            ctx("all", "2022-01-01", "2022-12-31"))
    # a probe is past the size gate whatever its window, and the smoke's
    # synthetic archive is past both
    assert gedi.GEDIAdapter().fetch_preflight(
        ctx("all", "2019-01-01", "2025-12-31", src=str(tmp_path))) is None
    monkeypatch.setenv("GEDI_ALLOW_BUILD", "1")
    nr = tmp_path / "netrc"
    nr.write_text("machine urs.earthdata.nasa.gov login u password p\n")
    monkeypatch.setenv("NETRC", str(nr))
    assert gedi.GEDIAdapter().fetch_preflight(
        ctx("all", "2019-01-01", "2025-12-31")) is None


def test_the_granule_key_joins_the_three_products():
    """Measured 2026-09-20: on 2022-06-01 all four GEDI products list 27
    granules and this key matches one-to-one across them."""
    def e(level, stamp, orbit, gran, track, rel, prefix=""):
        n = f"{prefix}GEDI0{level}_{stamp}_O{orbit}_{gran}_T{track}_{rel}"
        return {"title": n, "granule_size": "1062.2",
                "time_start": "2022-06-01T13:32:02.000Z",
                "time_end": "2022-06-01T15:04:53.000Z",
                "links": [{"href": f"https://x/lp-prod-public/{n}.h5"},
                          {"href": f"https://x/lp-prod-protected/{n}.h5"}]}
    a = gedi.parse_entries(
        [e("2_A", "2022152133202", "19644", "01", "00181", "02_004_02_V003")],
        "GEDI02_A")
    b = gedi.parse_entries(
        [e("2_B", "2022152133202", "19644", "01", "00181", "02_004_01_V003")],
        "GEDI02_B")
    c = gedi.parse_entries(
        [e("4_A", "2022152133202", "19644", "01", "00181", "02_004_01_V003")],
        "GEDI L4A v3")
    d = gedi.parse_entries(
        [e("4_A", "2022152133202", "19644", "01", "00181", "02_003_01_V002",
           prefix="GEDI_L4A_AGB_Density_V2_1.")], "GEDI L4A v2.1")
    key = ("2022152133202", "19644", "01", "00181")
    assert set(a) == set(b) == set(c) == set(d) == {key}
    assert "protected" in a[key]["url"] and a[key]["bytes"] == 1062200000
    # two granules for one key is a refusal, not a silent overwrite
    with pytest.raises(gedi.FormatError, match="two granules"):
        gedi.parse_entries(
            [e("2_A", "2022152133202", "19644", "01", "00181",
               "02_004_02_V003"),
             e("2_A", "2022152133202", "19644", "01", "00181",
               "02_004_01_V003")], "GEDI02_A")
    # a listing name this parser does not understand is a refusal
    with pytest.raises(gedi.FormatError, match="carry no"):
        gedi.parse_entries([{"title": "README", "links": []}], "GEDI02_A")
    # a granule with no protected .h5 link is a refusal
    bad = e("2_A", "2022152133202", "19644", "02", "00181", "02_004_02_V003")
    bad["links"] = [{"href": "https://x/lp-prod-public/a.h5"}]
    with pytest.raises(gedi.FormatError, match="no protected"):
        gedi.parse_entries([bad], "GEDI02_A")


def test_the_stem_stamp_dates_a_granule():
    """The acquisition field is YYYYDDDHHMMSS: DOY 152 of 2022 is 2022-06-01."""
    k = ("2022152133202", "19644", "01", "00181")
    want = b10.seconds_since_epoch(dt.date(2022, 6, 1)) + 13 * 3600 + \
        32 * 60 + 2
    assert gedi.stem_seconds(k) == want
    lo = b10.seconds_since_epoch(dt.date(2022, 6, 1))
    hi = lo + 30 * 86400
    assert gedi.in_window({"t0": None, "t1": None}, k, lo, hi)
    assert not gedi.in_window({"t0": None, "t1": None}, k, lo + 40 * 86400,
                              hi + 40 * 86400)


def test_the_shot_number_join():
    spine = np.array([10, 11, 12, 13], np.uint64)
    other = np.array([13, 11, 99], np.uint64)
    idx = gedi.join_by_shot(spine, other)
    assert idx.tolist() == [-1, 1, -1, 0]
    # an EMPTY partner is every shot missing, not a crash
    assert gedi.join_by_shot(spine, np.array([], np.uint64)).tolist() == \
        [-1, -1, -1, -1]


def test_a_renamed_quality_flag_is_resolved_not_guessed(tmp_path):
    """Version 3 renamed `quality_flag` to `l2a_quality_flag_rel3` (measured
    from CMR's variable listing, 2026-09-20). Every quantity is resolved
    against a candidate list and a file matching none of them is a refusal
    carrying the group's own datasets."""
    h5py = pytest.importorskip("h5py")
    p = tmp_path / "g.h5"
    with h5py.File(p, "w") as f:
        g = f.create_group("BEAM0000")
        g.create_dataset("quality_flag", data=np.ones(3, np.uint8))
        g.create_dataset("sensitivity", data=np.ones(3, np.float32))
    with h5py.File(p, "r") as f:
        g = f["BEAM0000"]
        assert gedi.resolve_name(g, gedi.L2A_NEEDED["quality"], "quality",
                                 "L2A/BEAM0000") == "quality_flag"
        with pytest.raises(gedi.FormatError, match="sensitivity"):
            gedi.resolve_name(g, ("nothing_like_this",), "x",
                              "L2A/BEAM0000")


class _Response:
    """Enough of a `requests` response for `_h5range` to read."""

    def __init__(self, status, content, headers, url):
        self.status_code = status
        self.content = content
        self.headers = headers
        self.url = url
        self.history = ()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


class _FileSession:
    """A `requests`-shaped session that serves byte ranges from a local file.

    It is the one part of the fetch design that cannot be exercised by the
    smoke's `--source-dir` path, because that path opens the granule from
    disk: this stands in for LP DAAC and answers HTTP 206 the way the
    archive must, so the HDF5-over-byte-ranges reader is tested rather than
    hoped for.
    """

    def __init__(self, path):
        self.path = path
        self.size = os.path.getsize(path)
        self.requests = 0

    def get(self, url, headers=None, stream=False, timeout=None,
            allow_redirects=True):
        self.requests += 1
        rng = (headers or {}).get("Range", "")
        lo, hi = (int(x) for x in rng.split("=", 1)[1].split("-"))
        hi = min(hi, self.size - 1)
        with open(self.path, "rb") as fh:
            fh.seek(lo)
            body = fh.read(hi - lo + 1)
        return _Response(206, body,
                         {"Content-Range": f"bytes {lo}-{hi}/{self.size}"},
                         url + "?signed")


def test_hdf5_over_byte_ranges_reads_only_what_is_asked_for(tmp_path):
    """The whole fetch design in one test: 2.58 TB a month is unfetchable, so
    the adapter reads the file's index and then the eleven datasets it
    wants. An archive answering 200 instead of 206 is a REFUSAL, because
    that is the whole granule arriving."""
    h5py = pytest.importorskip("h5py")
    from family1.adapters import _h5range as hr
    p = tmp_path / "granule.h5"
    want = np.arange(1000, dtype=np.float32)
    with h5py.File(p, "w") as f:
        g = f.create_group("BEAM0000")
        g.create_dataset("elev_lowestmode", data=want)
        # the waveform bulk this store never reads: 8 MB of the granule
        g.create_dataset("rx_waveform",
                         data=np.zeros(2_000_000, np.float32))
    size = os.path.getsize(p)
    sess = _FileSession(str(p))
    op = hr.open_range(sess, "https://lp/granule.h5", block=1 << 16,
                       cache_blocks=64)
    got = np.asarray(op.h5["BEAM0000"]["elev_lowestmode"][:], np.float32)
    hr.finish_range(op)
    st = op.stats()
    op.close()
    assert np.array_equal(got, want)
    assert st["granule_bytes"] == size
    # THE POINT: a fraction of the granule, not the granule
    assert st["bytes_read"] < size * 0.25, st
    assert 0 < st["read_fraction"] < 0.25
    assert st["http_requests"] > 1

    class _Whole(_FileSession):
        def get(self, url, headers=None, **kw):
            with open(self.path, "rb") as fh:
                return _Response(200, fh.read(), {}, url)

    with pytest.raises(hr.RangeError, match="not serving byte ranges"):
        hr.open_range(_Whole(str(p)), "https://lp/granule.h5")


def test_smoke_all_stages_and_the_probe(tmp_path):
    res = b1.run_smoke("gedi", root=str(tmp_path / "s"), keep=True)
    assert len(res["truth"]) == TRUTH_ROWS
    p = res["probe"]
    assert p["rows"] == PROBE_ROWS
    assert p["counts_scope"] == "month"
    assert p["distinct_platforms"] == BEAMS
    assert p["bytes_fetched"] > 0
    c = p["counts"]
    # the probe month holds two of the four stems
    assert c["granules_wanted"] == MONTH_STEMS
    assert c["shots_in_granule"] == MONTH_STEMS * BEAMS * SHOTS
    assert c["shots_quality_flag_zero"] == MONTH_STEMS * BEAMS
    assert c["rows_kept"] == PROBE_ROWS
    # THE JOIN, branch by branch. One stem in the probe month has no L4A
    # granule at all (every shot of it), the other is missing ONE shot from
    # L4A and carries one shot the L4A quality flag rejects.
    assert c["granules_without_l4a"] == 1
    assert c["shots_without_l4a_granule"] == BEAMS * SHOTS
    assert c["shots_without_l4a"] == BEAMS
    assert c["l4a_quality_zero"] == BEAMS
    assert c["shots_without_l2b"] == 0
    assert c["l2b_quality_zero"] == MONTH_STEMS * BEAMS
    # out of bounds -> NaN and COUNTED, never clipped (contract rule 3), and
    # nothing out of bounds reached the writer
    assert c["out_of_bounds"] == {"elev_lowestmode": MONTH_STEMS * BEAMS}
    assert p["out_of_bounds"]["elev_lowestmode"] == MONTH_STEMS * BEAMS
    assert sum(p["out_of_bounds_stored"].values()) == 0
    # qc: a quality shot in a degraded period is 1, everything else 0
    assert p["qc"] == {"0": PROBE_ROWS - MONTH_STEMS * BEAMS,
                       "1": MONTH_STEMS * BEAMS}
    assert c["degrade_flag"] == {"0": PROBE_ROWS - MONTH_STEMS * BEAMS,
                                 "31": MONTH_STEMS * BEAMS}
    # NaN per channel: biomass is absent for every shot of the L4A-less stem,
    # for the one shot missing from the other, and for the quality-zero shot
    nan_agbd = round((BEAMS * (SHOTS - 1) + BEAMS + BEAMS) / PROBE_ROWS, 6)
    assert p["nan_fraction"]["agbd"] == nan_agbd
    assert p["nan_fraction"]["agbd_se"] == nan_agbd
    assert p["nan_fraction"]["cover"] == round(
        (MONTH_STEMS * BEAMS) / PROBE_ROWS, 6)
    assert p["nan_fraction"]["rh98"] == 0.0
    # the dataset names actually used are recorded, as a LIST of one dict
    rd = c["resolved_datasets"]
    assert isinstance(rd, list) and rd[0]["L2A"]["quality"] == \
        "l2a_quality_flag_rel3"
    assert rd[0]["L4A"]["quality"] == "l4a_quality_flag_rel3"


def test_the_built_store_per_year_and_platforms(tmp_path):
    res = b1.run_smoke("gedi", root=str(tmp_path / "s"), keep=True,
                       probe=False)
    root = os.path.join(res["work"], "gedi", "gedi")
    st = json.load(open(os.path.join(root, "store.json")))
    assert st["N"] == TRUTH_ROWS and st["C"] == 11
    # TWO calendar years, because the smoke window spans a year boundary —
    # which is exactly what the size gate refuses on the real archive
    assert st["per_year"] == {"2021": TRUTH_ROWS // 2, "2022": TRUTH_ROWS // 2}
    assert st["family"] == "family1_tf" and st["family_version"] == "1.0.tf"
    assert st["tier"] == "P"
    c = st["counts"]
    assert c["shots_in_granule"] == SHOTS_TOTAL
    assert c["shots_quality_flag_zero"] == DROPPED
    assert c["rows_kept"] == TRUTH_ROWS
    # one stem of the four has no L4A file; three stems miss one shot each
    assert c["granules_without_l4a"] == 1
    assert c["shots_without_l4a"] == (STEMS - 1) * BEAMS
    # THE PLATFORM TABLE: eight beams offered, the two the store holds kept,
    # and the beam type read from the group's own attribute
    plats = json.load(open(os.path.join(root, "platforms.json")))
    assert len(plats) == BEAMS
    assert st["platforms"]["source_entries"] == len(gedi.BEAMS) == 8
    assert st["platforms"]["in_store"] == BEAMS
    assert st["platforms"]["in_store_without_entry"] == 0
    by_id = {v["id"]: v for v in plats.values()}
    assert set(by_id) == set(gedi.SMOKE_BEAMS)
    assert by_id["BEAM0000"]["beam_type"] == "coverage"
    assert by_id["BEAM0101"]["beam_type"] == "power"
    assert "description" in by_id["BEAM0000"]["beam_type_source"]
    assert str(b10.platform_hash("BEAM0000")) in plats


def test_storage_and_fetch_is_arithmetic_not_a_guess():
    counts = {"rows_kept": 1_000_000, "fetch_seconds": 100.0}
    per = [{"l2a": {"granule_bytes": 1_000_000_000, "bytes_read": 50_000_000},
            "l2b": {"granule_bytes": 600_000_000, "bytes_read": 30_000_000},
            "l4a": {"granule_bytes": 160_000_000, "bytes_read": 8_000_000}}]
    so = gedi.storage_and_fetch(counts, per, 11)
    m = so["measured"]
    assert m["granules_read"] == 1 and m["rows_per_granule"] == 1_000_000.0
    assert m["native_bytes_of_those_granules"] == 1_760_000_000
    assert m["bytes_read"] == 88_000_000
    assert m["read_fraction"] == 0.05
    assert m["bytes_per_second"] == 880_000.0
    pj = so["projected_one_year_2022"]
    assert pj["granules"] == 1428 * 12
    assert pj["rows"] == 1_000_000 * 1428 * 12
    assert pj["store_bytes"] == 1_000_000 * 1428 * 12 * 49
    # 1,516,813.1 + 886,429.2 + 238,860.2 MB, measured on CMR 2026-09-20
    assert pj["bytes_to_fetch"] == round(0.05 * 2_642_102_500_000 * 12)
    assert so["bytes_per_row_stored"] == 49
    assert gedi.storage_and_fetch({"rows_kept": 0}, per, 11) is None


def test_the_index_plan(tmp_path):
    ad = gedi.GEDIAdapter()
    src = str(tmp_path / "src")
    lo = b10.parse_date(ad.smoke_window[0])
    hi = b10.parse_date(ad.smoke_window[1])
    ad.smoke_sources(src, lo, hi)
    a = argparse.Namespace(
        store="gedi", work=str(tmp_path / "w"), source_dir=src,
        start=str(lo), end=str(hi), stage="index", force=False, attempts=1,
        qc_keep=2, check_chunk_rows=b10.CHECK_CHUNK_ROWS, assemble="auto",
        parts_from_hub=False, push_parts=False, allow_missing_years=False)
    ctx = b10.Ctx(a, adapter=ad, layout=b1.layout_for(ad))
    b10.run_stages(ctx, ["index"], stage_fn=b1.STAGE_FN, deps=b1.DEPS)
    plan = json.load(open(os.path.join(ctx.root, "plan.json")))
    assert plan["granules"]["l2a"] == STEMS
    assert plan["granules"]["l4a"] == STEMS - 1
    assert plan["granules_joined"] == STEMS
    assert plan["l4a_version"] == "3"
    assert plan["fetch_mode"] == "range"
    assert plan["one_year_only"] is True
    assert plan["collections"]["l2a"]["concept"] == "C3974616071-LPCLOUD"
    assert plan["collections"]["l4a"]["concept"] == "C4212593885-ORNL_CLOUD"
    assert plan["beams_expected"] == list(gedi.BEAMS)


def test_a_capped_probe_spreads_over_the_window():
    """Six granules from the head of a month is six quarter-orbits of one
    morning; six spread over it is a measurement of the month."""
    assert gedi.spread(list(range(1428)), 6) == [0, 238, 476, 714, 952, 1190]
    assert gedi.spread(list(range(4)), 6) == [0, 1, 2, 3]
    assert gedi.spread([], 6) == []
    assert gedi.spread(list(range(10)), 0) == list(range(10))


def test_the_l4a_version_knob(monkeypatch):
    monkeypatch.setenv("GEDI_L4A_VERSION", "2.1")
    ad = gedi.GEDIAdapter()
    assert ad.l4a["concept"] == "C2237824918-ORNL_CLOUD"
    assert ad.l4a["record"] == ("2019-04-17", "2025-07-09")
    monkeypatch.setenv("GEDI_L4A_VERSION", "9")
    with pytest.raises(SystemExit, match="GEDI_L4A_VERSION"):
        gedi.GEDIAdapter()


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))

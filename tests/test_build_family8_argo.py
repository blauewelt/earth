#!/usr/bin/env python3
"""What must hold before the family-8 Argo observation store is trusted.

`ml/plans/E076_family8_nearest_observations.md` §2, §2.6 and §3. Family 8 is
family 7's dense groups (unchanged) plus a store of RAW Argo profiles and a
k-nearest search over it: instead of asking "is there a value at this cell and
this five-day bin" — which a sparse observing system answers "no" at 92 % of
bins — the cone asks "what are the k nearest measurements, and how far away
are they", so a channel is never empty, the measurement is just far away.

CPU-only, NO NETWORK. Every source is synthetic: the daily files are written
by `build_family8_argo.write_prof_nc`, which is the SAME writer `--smoke` uses
and carries the real GDAC variable names, dimensions and fill values, so the
extractor is tested against the shape of the file it meets on the box. The
end-to-end check is `build_family8_argo.run_smoke()`, the same path
`python3 ml/build_family8_argo.py --smoke` runs.

    python3 -m pytest tests/test_build_family8_argo.py -q
"""
import ast
import datetime as dt
import json
import os
import shutil
import subprocess
import sys
import tempfile

import numpy as np
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
ML = os.path.join(HERE, "..", "ml")
sys.path.insert(0, ML)

import build_family3 as f3                                      # noqa: E402
import build_family8_argo as b8                                 # noqa: E402
import family8_store as fs                                      # noqa: E402
import aggregate_cadence as ac                                  # noqa: E402


# --------------------------------------------------------------- fixtures --
@pytest.fixture(scope="module")
def smoke():
    """One synthetic end-to-end build; several assertions below read it."""
    root = tempfile.mkdtemp(prefix="f8test_")
    work, truth = b8.run_smoke(root=root, keep=True)
    yield dict(root=root, work=work, truth=truth,
               src=os.path.join(root, "src"),
               store=fs.ArgoStore(os.path.join(work, b8.STEM)))
    shutil.rmtree(root, ignore_errors=True)


def linear_profile(t0=18.0, dtdp=-0.006, s0=34.8, dsdp=0.0005,
                   pres=None):
    """A LINEAR T(p), S(p) sampled every 20 dbar — exact under interpolation."""
    p = np.arange(5.0, 2001.0, 20.0) if pres is None else np.asarray(pres, float)
    return p, t0 + dtdp * p, s0 + dsdp * p


def juld_of(day, frac=0.25):
    """`day` (a date) as JULD — days since 1950-01-01, plus a time of day."""
    return (day - b8.JULD_EPOCH).days + frac


def one_day_file(path, profs):
    return b8.write_prof_nc(path, profs)


def extract(path):
    return b8.extract_day(path, "atlantic_ocean", "2015-01-03")


def write_store(dirpath, rows):
    """A store written straight from arrays — the reader's own layout.

    Deliberately NOT built through the builder: the k-nearest tests need
    observations at EXACT offsets and times, and a store assembled from
    synthetic netCDF would only let them be approximately where the test wants
    them.
    """
    os.makedirs(dirpath, exist_ok=True)
    n = len(rows["bin"])
    for k, v in rows.items():
        np.save(os.path.join(dirpath, k + ".npy"), v)
    off = np.searchsorted(np.asarray(rows["bin"], np.int64),
                          np.arange(b8.N_BINS + 1), side="left").astype(np.int64)
    np.save(os.path.join(dirpath, "bin_offsets.npy"), off)
    with open(os.path.join(dirpath, "store.json"), "w") as fh:
        json.dump({"N": int(n), "levels": [float(x) for x in b8.LEVELS],
                   "recipe": b8.RECIPE, "n_bins": b8.N_BINS}, fh)
    return fs.ArgoStore(dirpath)


# ------------------------------------------------------------------- 1 -----
def test_1_the_axis_and_the_ground_scale_are_shared_definitions():
    """One bin rule, one 111.32 km/degree, one level list — never a second copy.

    Family 8's store has to land in the SAME five-day bins as families 4-7 or
    a nearest-observation token would point at a different pentad than the
    anchor it is attached to. The three constants that could drift are pinned
    here against the modules that own them: `aggregate_cadence` (the bin),
    `ml/temporal.py` (the ground scale, read with `ast` so the test does not
    have to import torch) and `build_family3.LEVELS` (the pressures).
    """
    assert b8.PENTAD_DAYS == 5 and b8.N_BINS == 3142
    assert b8.START == ac.EPOCH == dt.date(1982, 1, 1)
    assert ac.bin_index(dt.date(1982, 1, 1), 5) == 0
    assert ac.bin_index(dt.date(2024, 12, 31), 5) == 3141
    # the builder bins by JULD, the axis bins by date: the same number
    for day in (dt.date(1982, 1, 1), dt.date(2004, 1, 1), dt.date(2015, 1, 3),
                dt.date(2024, 12, 31)):
        t = juld_of(day, 0.9) - b8.JULD_OFFSET
        assert int(np.floor(t / 5)) == ac.bin_index(day, 5), day
    assert list(b8.LEVELS) == list(f3.LEVELS) and len(b8.LEVELS) == 16

    src = open(os.path.join(ML, "temporal.py")).read()
    tree = ast.parse(src)
    km = [n.value.value for n in ast.walk(tree)
          if isinstance(n, ast.Assign) and len(n.targets) == 1
          and isinstance(n.targets[0], ast.Name)
          and n.targets[0].id == "KM_PER_DEG"
          and isinstance(n.value, ast.Constant)]
    assert km and km[0] == fs.KM_PER_DEG, (km, fs.KM_PER_DEG)
    assert abs(fs.KM_PER_DEG * 0.25 - 27.83) < 0.005      # ml/cone.py's cell


def test_1b_civil_days_matches_datetime():
    """The vectorised date arithmetic the index stats run on, against stdlib."""
    days = [dt.date(1997, 7, 29), dt.date(2000, 2, 29), dt.date(2004, 1, 1),
            dt.date(2015, 1, 3), dt.date(2024, 12, 31), dt.date(1982, 1, 1)]
    y = [d.year for d in days]
    m = [d.month for d in days]
    dd = [d.day for d in days]
    want = [(d - dt.date(1970, 1, 1)).days for d in days]
    assert list(b8.civil_days(y, m, dd)) == want
    ints = np.array([int(f"{d:%Y%m%d}") for d in days], np.int32)
    assert list(b8.yyyymmdd_to_bin(ints)) == [ac.bin_index(d, 5) for d in days]


# ------------------------------------------------------------------- 2 -----
def test_2_a_linear_profile_interpolates_EXACTLY(tmp_path):
    """Every one of the 16 levels comes back at its analytic value.

    Not "close": a linear function through two samples IS the interpolant, so
    the only tolerance a correct implementation needs is float64 rounding.
    Modelling the arithmetic rather than picking a loose tolerance is what
    makes a real interpolation bug fail this test instead of passing it.
    """
    p, t, s = linear_profile()
    got_t = b8.interp_levels(p, t)
    got_s = b8.interp_levels(p, s)
    want_t = 18.0 - 0.006 * b8.LEVELS_ARR
    want_s = 34.8 + 0.0005 * b8.LEVELS_ARR
    assert np.all(np.isfinite(got_t)) and np.all(np.isfinite(got_s))
    assert np.max(np.abs(got_t - want_t)) < 1e-12
    assert np.max(np.abs(got_s - want_s)) < 1e-12
    # an EXACT sample at a level is its own bracket (gap 0)
    p2 = np.array([10.0, 30.0, 50.0])
    assert b8.interp_levels(p2, np.array([1.0, 2.0, 3.0]))[0] == 1.0
    assert b8.interp_levels(p2, np.array([1.0, 2.0, 3.0]))[1] == 2.0


def test_3_the_level_tolerance_rule(tmp_path):
    """A level with no bracketing samples inside tol(level) is NaN, not guessed.

    tol is 25 dbar at and below 200, 50 at and below 500, 100 above (E-076
    §3). The profile below straddles level 300 with a 200 dbar hole and level
    1500 with a 90 dbar one: the first must be NaN and the second must fill.
    """
    assert list(b8.level_tol([10, 200, 300, 500, 700, 1900])) == \
        [25.0, 25.0, 50.0, 50.0, 100.0, 100.0]
    p = np.array([5.0, 15.0, 195.0, 205.0, 205.0 + 200.0, 1455.0, 1545.0])
    v = np.arange(len(p), dtype=float)
    out = b8.interp_levels(p, v)
    lv = list(b8.LEVELS)
    assert np.isnan(out[lv.index(300.0)]), "a 200 dbar hole filled level 300"
    assert np.isfinite(out[lv.index(1500.0)]), "a 90 dbar gap must fill at 1500"
    assert np.isfinite(out[lv.index(200.0)]), "195/205 brackets 200"
    assert np.isnan(out[lv.index(700.0)]), "nothing brackets 700"
    # the shallowest-level rule: the 10 dbar level takes the shallowest good
    # sample when that sample is within 20 dbar, and NEVER extrapolates.
    lone = b8.interp_levels(np.array([14.0, 900.0]), np.array([7.0, 9.0]))
    assert lone[0] == 7.0
    far = b8.interp_levels(np.array([40.0, 900.0]), np.array([7.0, 9.0]))
    assert np.isnan(far[0]), "a 40 dbar shallowest sample must not fill 10 dbar"


def test_4_qc_rejection_and_T_S_independence(tmp_path):
    """A flag-4 sample never contributes; a POSITION_QC 4 profile is dropped.

    Four profiles in one file: good, POSITION_QC 4, JULD_QC 4, and one whose
    TEMPERATURE samples at 285 and 305 dbar are flagged 4. The last must keep
    its salinity at 300 dbar and lose only its temperature there — T and S are
    independent (E-076 §3), and the 265/325 bracket that remains is 60 dbar
    wide against a tolerance of 50.
    """
    p, t, s = linear_profile()
    day = dt.date(2015, 1, 3)
    q = np.full(len(p), b"1", "S1")
    q[p == 285.0] = b"4"
    q[p == 305.0] = b"4"
    profs = [
        dict(wmo=3901000, cycle=1, juld=juld_of(day), lat=10.0, lon=-30.0,
             pres=p, temp=t, psal=s),
        dict(wmo=3901001, cycle=1, juld=juld_of(day), lat=11.0, lon=-30.0,
             pres=p, temp=t, psal=s, position_qc=b"4"),
        dict(wmo=3901002, cycle=1, juld=juld_of(day), lat=12.0, lon=-30.0,
             pres=p, temp=t, psal=s, juld_qc=b"4"),
        dict(wmo=3901003, cycle=1, juld=juld_of(day), lat=13.0, lon=-30.0,
             pres=p, temp=t, psal=s, temp_qc=q),
    ]
    r = extract(one_day_file(str(tmp_path / "20150103_prof.nc"), profs))
    c = r["counts"]
    assert c["n_prof"] == 4 and c["kept"] == 2
    assert c["drop_position_qc"] == 1 and c["drop_juld_qc"] == 1
    assert list(r["wmo"]) == [3901000, 3901003]
    lv = list(b8.LEVELS)
    good = np.asarray(r["temp"][0], np.float32)
    flagged = np.asarray(r["temp"][1], np.float32)
    assert np.all(np.isfinite(good)), "the clean profile lost a level"
    assert np.isnan(flagged[lv.index(300.0)]), \
        "a flag-4 sample contributed to level 300"
    assert np.isfinite(np.asarray(r["psal"][1], np.float32)[lv.index(300.0)]), \
        "bad temperature took good salinity with it"
    assert np.isfinite(flagged[lv.index(200.0)]) and \
        np.isfinite(flagged[lv.index(400.0)])


def test_5_adjusted_when_D_or_A_raw_when_R(tmp_path):
    """DATA_MODE decides which triple is read, and nothing else does.

    Each profile's ADJUSTED twin is the raw one plus 100, which is not a
    temperature — so a builder that read the wrong variable cannot produce a
    plausible number, it produces 118 degC.
    """
    p, t, s = linear_profile()
    day = dt.date(2015, 1, 3)
    profs = []
    for i, mode in enumerate((b"R", b"A", b"D")):
        profs.append(dict(wmo=3902000 + i, cycle=1, juld=juld_of(day),
                          lat=float(i), lon=-20.0, mode=mode,
                          pres=p, temp=t, psal=s,
                          temp_adjusted=t + 100.0, psal_adjusted=s + 100.0,
                          pres_adjusted=p))
    r = extract(one_day_file(str(tmp_path / "20150103_prof.nc"), profs))
    assert r["counts"]["kept"] == 3
    want = 18.0 - 0.006 * b8.LEVELS_ARR
    t_r = np.asarray(r["temp"][0], np.float32)
    t_a = np.asarray(r["temp"][1], np.float32)
    t_d = np.asarray(r["temp"][2], np.float32)
    assert list(r["mode"]) == [b"R", b"A", b"D"]
    assert np.max(np.abs(t_r - want)) < 0.02, "R must read the RAW variables"
    assert np.max(np.abs(t_a - (want + 100.0))) < 0.2, "A must read ADJUSTED"
    assert np.max(np.abs(t_d - (want + 100.0))) < 0.2, "D must read ADJUSTED"


def test_6_bin_assignment_against_a_hand_computed_date(tmp_path):
    """2015-01-03 is 12,055 days after 1982-01-01, so it is bin 2411.

    Counted by hand rather than by the code under test: 1982-01-01 to
    2015-01-01 is 33 years holding 8 leap days (1984, 88, 92, 96, 2000, 04,
    08, 12), i.e. 33*365 + 8 = 12,053 days; 2015-01-03 is two more, 12,055;
    12,055 // 5 = 2,411.
    """
    assert 33 * 365 + 8 + 2 == 12055
    assert 12055 // 5 == 2411
    p, t, s = linear_profile()
    days = [dt.date(2015, 1, 3), dt.date(2015, 1, 7), dt.date(2015, 1, 8)]
    profs = [dict(wmo=3903000 + i, cycle=1, juld=juld_of(d, 0.5),
                  lat=1.0 + i, lon=5.0, pres=p, temp=t, psal=s)
             for i, d in enumerate(days)]
    r = extract(one_day_file(str(tmp_path / "20150103_prof.nc"), profs))
    assert list(r["bin"]) == [2411, 2411, 2412]
    assert list(r["bin"]) == [ac.bin_index(d, 5) for d in days]
    assert abs(float(r["time_days"][0]) - 12055.5) < 1e-3
    # and the bin's start day is the one the axis says it is
    assert ac.bin_start(2411, 5) == dt.date(2015, 1, 3)


def test_7_dedupe_and_the_thin_profile_rule(tmp_path):
    """< 2 filled T levels AND < 2 filled S levels is dropped; keys collide once."""
    day = dt.date(2015, 1, 3)
    p, t, s = linear_profile()
    thin_p = np.array([1400.0])                     # brackets nothing
    profs = [
        dict(wmo=3904000, cycle=7, juld=juld_of(day), lat=1.0, lon=2.0,
             pres=thin_p, temp=np.array([4.0]), psal=np.array([34.0])),
        dict(wmo=3904001, cycle=7, juld=juld_of(day), lat=1.0, lon=2.0,
             pres=p, temp=t, psal=s),
    ]
    r = extract(one_day_file(str(tmp_path / "20150103_prof.nc"), profs))
    assert r["counts"]["kept"] == 1 and r["counts"]["drop_too_few_levels"] == 1
    k = b8.dedupe_key(np.array([3904000, 3904000, 3904000]),
                      np.array([7, 7, 8]),
                      np.array([b"A", b"D", b"A"]))
    assert k[0] != k[1] and k[0] != k[2] and len(set(int(x) for x in k)) == 3


# ------------------------------------------------------------------- 8 -----
def test_8_store_is_sorted_and_the_csr_offsets_are_right(smoke):
    """Sorted by (bin, then time), and `bin_offsets` is that order's index."""
    st = smoke["store"]
    b = np.asarray(st["bin"])
    t = np.asarray(st["time_days"])
    off = np.asarray(st.bin_offsets)
    assert st.N == len(smoke["truth"]), (st.N, len(smoke["truth"]))
    assert np.all(np.diff(b) >= 0), "not sorted by bin"
    same = b[:-1] == b[1:]
    assert np.all(t[:-1][same] <= t[1:][same]), "not sorted by time within a bin"
    assert len(off) == b8.N_BINS + 1
    assert off[0] == 0 and off[-1] == st.N
    assert np.all(np.diff(off) >= 0)
    for bb in np.unique(b):
        lo, hi = int(off[bb]), int(off[bb + 1])
        assert hi > lo and np.all(b[lo:hi] == bb)
        assert st.count(bb, bb) == hi - lo
        assert np.array_equal(st.bins(bb, bb)["row"], np.arange(lo, hi))
    # a bin with no observations is an EMPTY slice, never a wrong one
    empty = int(np.max(b)) + 1
    assert st.count(empty, empty) == 0
    assert len(st.bins(empty, empty)["bin"]) == 0


def test_9_store_values_match_the_generated_truth(smoke):
    """Every kept profile is the one the generator wrote, level by level."""
    st, truth = smoke["store"], smoke["truth"]
    key = b8.dedupe_key(np.asarray(st["wmo"]), np.asarray(st["cycle"]),
                        np.full(st.N, b"A"))
    assert len(set(int(k) for k in key)) == st.N, "a duplicate survived"
    n_flag = 0
    for i, k in enumerate(key):
        tr = truth[int(k)]
        assert int(st["bin"][i]) == tr["bin"]
        assert abs(float(st["lat"][i]) - tr["lat"]) < 1e-3
        for col in ("temp", "psal"):
            got = np.asarray(st[col][i], np.float32)
            want = tr[col]
            assert np.array_equal(np.isfinite(got), np.isfinite(want))
            m = np.isfinite(want)
            assert np.allclose(got[m], want[m], atol=2e-2)
        n_flag += int(np.isnan(np.asarray(st["temp"][i], np.float32)).any())
    assert n_flag == 24, ("the 12 days x 2 basins of flag-4 profiles should "
                          f"each have lost level 300, got {n_flag}")


def test_10_store_json_records_the_policy_and_the_provenance(smoke):
    """A store that cannot say how it was made is not evidence of anything."""
    with open(os.path.join(smoke["work"], b8.STEM, "store.json")) as fh:
        m = json.load(fh)
    assert m["N"] == smoke["store"].N
    assert m["levels"] == [float(x) for x in b8.LEVELS]
    assert "POSITION_QC" in m["qc_policy"] and "ADJUSTED" in m["qc_policy"]
    assert m["duplicates_dropped"] == 1
    assert m["days_missing"] > 0            # the indian_ocean files never exist
    assert m["recipe"] == b8.RECIPE and m["epoch"] == "1982-01-01"
    assert m["footprint"]["log2_fp"] == -4.0 and m["footprint"]["log2_dt"] == -4.0
    assert sum(m["filled_levels_hist_t"]) > 0
    assert set(m["sha256"]) == {f"{k}.npy" for k in fs.COLUMNS} | \
        {"bin_offsets.npy"}
    for name, digest in m["sha256"].items():
        assert b8.sha256(os.path.join(smoke["work"], b8.STEM, name)) == digest
    stats = json.load(open(os.path.join(smoke["work"], "index_stats.json")))
    assert stats["n_profiles"] == 360        # 2 basins x 12 days x 15
    assert stats["nearest_neighbour_km"]["one_pentad_mean"] > 0
    assert "catalogue" in stats["kth_neighbour_km"]


# ------------------------------------------------------------------ 11 -----
def _knn_store(tmp_path):
    """Six observations at EXACT offsets from (0, 0), around anchor bin 100.

    anchor_time = 5 * (100 + 1) = 505 days, so dt is measured back from the
    END of the anchor's pentad (E-076 §2.1 requires dt >= 0 for an
    observation inside the anchor's own bin).

      row 0  E  bin  90  t 452  dx    10 km   dt 53 d  -> outside T_max = 30
      row 1  C  bin  98  t 492  dx    50 km   dt 13 d  -> d2 0.19028
      row 2  B  bin  99  t 498  dy   200 km   dt  7 d  -> d2 0.09444
      row 3  A  bin 100  t 502  dx   100 km   dt  3 d  -> d2 0.02000
      row 4  F  bin 100  t 503  dx 2,000 km   dt  2 d  -> outside R_max = 1000
      row 5  D  bin 101  t 507  dx     1 km   dt -2 d  -> THE FUTURE
    """
    km = fs.KM_PER_DEG
    rows = dict(
        bin=np.array([90, 98, 99, 100, 100, 101], np.int16),
        time_days=np.array([452.0, 492.0, 498.0, 502.0, 503.0, 507.0], np.float32),
        lat=np.array([0.0, 0.0, 200.0 / km, 0.0, 0.0, 0.0], np.float32),
        lon=np.array([10.0 / km, 50.0 / km, 0.0, 100.0 / km, 2000.0 / km,
                      1.0 / km], np.float32),
        temp=np.tile(np.arange(16, dtype=np.float16), (6, 1)),
        psal=np.full((6, 16), 35.0, np.float16),
        wmo=np.arange(6, dtype=np.int32) + 100,
        cycle=np.arange(6, dtype=np.int16),
        nlev=np.full(6, 50, np.int16),
        maxpres=np.full(6, 1900.0, np.float16),
        mode=np.full(6, b"D", "|S1"),
    )
    return write_store(str(tmp_path / "knn"), rows)


def test_11_knearest_returns_the_k_nearest_in_order(tmp_path):
    st = _knn_store(tmp_path)
    tok = st.knearest(0.0, 0.0, 100, k=5, R_max_km=1000.0, T_max_days=30.0)
    assert tok["n_R"] == 3, tok["n_R"]
    assert tok["n_found"] == 3
    assert list(tok["valid"]) == [True, True, True, False, False]
    assert list(tok["row"][:3]) == [3, 2, 1], "A, B, C is the d^2 order"
    assert np.allclose(tok["d2"][:3], [0.02, 0.0944444, 0.1902778], atol=1e-6)
    assert np.allclose(tok["dx_km"][:3], [100.0, 0.0, 50.0], atol=0.2)
    assert np.allclose(tok["dy_km"][:3], [0.0, 200.0, 0.0], atol=0.2)
    assert np.allclose(tok["dt_days"][:3], [3.0, 7.0, 13.0], atol=1e-4)
    assert np.all(tok["dt_days"][:3] >= 0.0), "E-076 §2.1: dt >= 0, always"
    assert np.allclose(tok["dist_km"][:3], [100.0, 200.0, 50.0], atol=0.2)
    # k = 2 truncates the ANSWER and never the density feature
    two = st.knearest(0.0, 0.0, 100, k=2, R_max_km=1000.0, T_max_days=30.0)
    assert list(two["row"]) == [3, 2] and two["n_R"] == 3


def test_12_a_future_observation_is_never_returned(tmp_path):
    """Row 5 sits 1 km away — the closest thing in the store — and is the future.

    One-sided in time is a HARD CONSTRAINT OF THE SEARCH (E-076 §2.2): the CSR
    slice stops at the anchor's own bin, so no radius and no metric can reach
    a later observation. This is the assertion that stands between family 8
    and a forecaster that has seen the answer.
    """
    st = _knn_store(tmp_path)
    for R in (1000.0, 1e6):
        tok = st.knearest(0.0, 0.0, 100, k=6, R_max_km=R, T_max_days=30.0)
        assert 5 not in list(tok["row"]), f"the future leaked at R_max={R}"
        assert np.all(np.asarray(st["bin"])[tok["row"][tok["valid"]]] <= 100)
    # ... and it IS reachable one pentad later, so the store is not just empty
    later = st.knearest(0.0, 0.0, 101, k=6, R_max_km=1000.0, T_max_days=30.0)
    assert list(later["row"])[0] == 5 and later["n_R"] == 4


def test_13_R_max_and_T_max_bound_the_set_with_miss_tokens(tmp_path):
    """Beyond the bounds the miss token survives — NaN, and `valid` False."""
    st = _knn_store(tmp_path)
    tight = st.knearest(0.0, 0.0, 100, k=5, R_max_km=1000.0, T_max_days=5.0)
    assert tight["n_R"] == 1 and list(tight["row"][:1]) == [3]
    assert not tight["valid"][1:].any()
    assert np.all(np.isnan(tight["temp"][1:])), "a miss token carries a value"
    assert np.all(np.isnan(tight["dx_km"][1:]))
    assert np.all(np.isnan(tight["dt_days"][1:]))
    assert list(tight["row"][1:]) == [-1] * 4
    # a radius that admits F as well moves n_R, and only n_R
    wide = st.knearest(0.0, 0.0, 100, k=5, R_max_km=3000.0, T_max_days=30.0)
    assert wide["n_R"] == 4 and list(wide["row"][:4]) == [3, 2, 1, 4]
    # nothing at all: every slot is a miss and n_R is 0, which is the honest
    # answer for the interior before 2004 (E-076 §2.3)
    none = st.knearest(0.0, 0.0, 20, k=5, R_max_km=1000.0, T_max_days=30.0)
    assert none["n_R"] == 0 and none["n_found"] == 0
    assert not none["valid"].any() and np.all(np.isnan(none["temp"]))
    assert none["temp"].shape == (5, 16)
    far = st.knearest(80.0, 0.0, 100, k=5, R_max_km=1000.0, T_max_days=30.0)
    assert far["n_R"] == 0, "an anchor 8,900 km away found something"


def test_14_the_footprint_fields_are_constants_for_argo(tmp_path):
    """E-076 §2.6: a point profile taken in minutes, both clamped to -4."""
    st = _knn_store(tmp_path)
    tok = st.knearest(0.0, 0.0, 100, k=5, R_max_km=1000.0, T_max_days=30.0)
    assert np.all(tok["log2_fp"] == -4.0) and np.all(tok["log2_dt"] == -4.0)
    assert tok["log2_fp"].shape == (5,), "a per-token FIELD, not a scalar"
    assert fs.LOG2_FP_ARGO == -4.0 and fs.LOG2_DT_ARGO == -4.0


def test_15_the_metric_scales_are_separable(tmp_path):
    """L and T are the metric's, R_max and T_max the search's — E-076 §2.1/§2.2.

    With L shrunk, distance dominates and C (50 km, 13 days old) overtakes A
    (100 km, 3 days old); with T shrunk, recency dominates and A stays first.
    The bounded SET is identical in both, which is why `n_R` may not move.
    """
    st = _knn_store(tmp_path)
    a = st.knearest(0.0, 0.0, 100, k=3, R_max_km=1000.0, T_max_days=30.0,
                    L_km=60.0, T_days=1000.0)
    b = st.knearest(0.0, 0.0, 100, k=3, R_max_km=1000.0, T_max_days=30.0,
                    L_km=1000.0, T_days=1.0)
    assert list(a["row"]) == [1, 3, 2], list(a["row"])
    assert list(b["row"]) == [3, 2, 1], list(b["row"])
    assert a["n_R"] == b["n_R"] == 3


# ------------------------------------------------------------------ 16 -----
def test_16_a_marker_is_absent_when_extraction_raises_mid_year(tmp_path):
    """`.done` may only under-claim (ml/CLAUDE.md §5.21).

    A corrupt daily file in the middle of a year must take the YEAR down —
    never silently — and must leave neither `profiles/<year>.done` nor
    `profiles.done` behind, so the resumed job replays the year it lost.
    """
    import argparse
    src = tmp_path / "src"
    work = tmp_path / "work"
    d_lo, d_hi = dt.date(2015, 1, 1), dt.date(2015, 1, 4)
    p, t, s = linear_profile()
    for basin in b8.BASINS:
        for k in range((d_hi - d_lo).days + 1):
            day = d_lo + dt.timedelta(days=k)
            path = src / "geo" / basin / "2015" / "01" / f"{day:%Y%m%d}_prof.nc"
            os.makedirs(path.parent, exist_ok=True)
            b8.write_prof_nc(str(path), [dict(
                wmo=3905000 + k, cycle=1, juld=juld_of(day), lat=1.0 + k,
                lon=2.0, pres=p, temp=t, psal=s)])
    # the third atlantic day is not a netCDF file at all
    bad = src / "geo" / "atlantic_ocean" / "2015" / "01" / "20150103_prof.nc"
    bad.write_bytes(b"this is not an HDF5 file" * 100)
    with open(src / b8.INDEX_NAME, "wb") as fh:
        import gzip
        fh.write(gzip.compress(b"file,date,latitude,longitude,ocean,"
                               b"profiler_type,institution,date_update\n"
                               b"a/1/profiles/D1_001.nc,20150101120000,1.0,"
                               b"2.0,A,846,AO,20260101000000\n"))
    a = argparse.Namespace(work=str(work), source_dir=str(src),
                           start=str(d_lo), end=str(d_hi), force=False,
                           stage="all", smoke=False, jobs=1, attempts=1,
                           index_anchors=10, index_pentads=1)
    ctx = b8.Ctx(a)
    b8.stage_index(ctx)
    with pytest.raises(Exception):
        b8.stage_profiles(ctx)
    assert not b8.marked(str(work), "profiles/2015"), \
        "the year marker survived a failed extraction"
    assert not b8.marked(str(work), "profiles"), "the stage marker survived"
    assert not os.path.exists(os.path.join(str(work), b8.STEM, "store.json"))
    assert b8.marked(str(work), "index"), "the index stage was undone too"


def test_16b_every_basin_streams_to_its_own_path(tmp_path):
    """All three basins name their daily file `YYYYMMDD_prof.nc`.

    A scratch destination built from the basename alone gives the three of
    them ONE shared file, and the pipeline downloads up to eight at a time —
    so a basin's profiles would be parsed out of another basin's bytes, or the
    file would be deleted from under a live parse. Caught on the first real
    download, 2026-09-07, and pinned here: the destination mirrors the
    archive's full relative path.
    """
    import argparse
    a = argparse.Namespace(work=str(tmp_path / "w"), source_dir="",
                           start="2015-01-03", end="2015-01-03", force=False,
                           stage="all", smoke=False, jobs=1, attempts=1,
                           index_anchors=10, index_pentads=1)
    ctx = b8.Ctx(a)
    day = dt.date(2015, 1, 3)
    dests = set()
    for basin in b8.BASINS:
        urls, rel = b8.day_urls(basin, day)
        assert rel.startswith(f"geo/{basin}/2015/01/") and rel.endswith(
            "20150103_prof.nc")
        assert urls[0].startswith(b8.GDAC) and urls[1].startswith(b8.GDAC_MIRROR)
        dests.add(os.path.join(ctx.scratch, rel))
    assert len(dests) == len(b8.BASINS), dests
    assert len({os.path.basename(d) for d in dests}) == 1, \
        "the basenames DO collide — which is the whole point of this test"


def test_17_stage_order_is_enforced(tmp_path):
    """profiles needs index, publish needs profiles — checked before spending."""
    import argparse
    a = argparse.Namespace(work=str(tmp_path / "w"), source_dir="",
                           start="2015-01-01", end="2015-01-02", force=False,
                           stage="all", smoke=False, jobs=1, attempts=1,
                           index_anchors=10, index_pentads=1)
    ctx = b8.Ctx(a)
    with pytest.raises(SystemExit) as e:
        b8.run_stages(ctx, ["profiles"])
    assert "index" in str(e.value)
    with pytest.raises(SystemExit) as e:
        b8.run_stages(ctx, ["publish"])
    assert "profiles" in str(e.value)


def test_18_a_missing_day_is_a_day_not_a_failure(smoke):
    """No `indian_ocean` file exists in the smoke tree, and the build finished.

    A 404 from every mirror is a legitimately empty day (E-076 §3): it is
    counted and the build continues. Anything else must NOT be — which is what
    test 16 asserts from the other side.
    """
    with open(os.path.join(smoke["work"], b8.STEM, "store.json")) as fh:
        m = json.load(fh)
    assert m["days_missing"] == 12, m["days_missing"]      # 12 indian days
    assert all(x.startswith("indian_ocean/") for x in m["days_missing_list"])
    for y, row in m["per_year"].items():
        assert row["files"] > 0 and row["kept"] > 0, y


def test_19_smoke_cli_runs(tmp_path):
    """`python3 ml/build_family8_argo.py --smoke` is the command the box runs."""
    r = subprocess.run([sys.executable, os.path.join(ML, "build_family8_argo.py"),
                        "--smoke", "--smoke-dir", str(tmp_path / "s")],
                       capture_output=True, text=True, timeout=600)
    assert r.returncode == 0, r.stdout[-3000:] + r.stderr[-3000:]
    assert "smoke     OK" in r.stdout, r.stdout[-2000:]
    assert os.path.exists(tmp_path / "s" / "work" / b8.STEM / "store.json")
    assert not os.path.exists(tmp_path / "s" / "work" / "publish.done")


def test_20_the_index_parser_reads_the_real_format(tmp_path):
    """The GDAC index: `#` comments, a header line, YYYYMMDDHHMMSS dates.

    Transcribed from the live file's first lines (fetched 2026-09-07), plus
    the two malformed shapes a 3-million-line text file always has.
    """
    import gzip
    text = (
        "# Title : Profile directory file of the Argo Global Data Assembly Center\n"
        "# Date of update : 20260907162415\n"
        "file,date,latitude,longitude,ocean,profiler_type,institution,date_update\n"
        "aoml/13857/profiles/D13857_001.nc,19970729200300,0.267,-16.032,A,845,AO,20260220143529\n"
        "aoml/13857/profiles/D13857_002.nc,19970809192112,0.072,-17.659,A,845,AO,20260220143530\n"
        "coriolis/6901823/profiles/D6901823_023.nc,20150103235900,40.817,190.700,A,846,IF,20260301000000\n"
        "held/but/unplaced.nc,20150103235900,,,,846,AO,20260101000000\n"
        "bad/line/short.nc\n")
    p = tmp_path / b8.INDEX_NAME
    p.write_bytes(gzip.compress(text.encode()))
    idx = b8.parse_index(str(p))
    # a record with an empty position is the archive's, not the parser's:
    # 1.06 % of the live index looks like that (measured 2026-09-07)
    assert len(idx["date"]) == 3
    assert idx["lines_no_position"] == 1 and idx["lines_malformed"] == 1
    assert list(idx["date"]) == [19970729, 19970809, 20150103]
    assert idx["date_update_max"] == 20260301000000
    assert abs(float(idx["lon"][2]) - (-169.3)) < 1e-3, "lon must fold to [-180,180)"
    assert list(b8.yyyymmdd_to_bin(idx["date"]))[2] == 2411
    stats = b8.index_stats(idx, n_anchor=5, n_pentads=2)
    assert stats["n_profiles"] == 3
    assert stats["profiles_per_year"] == {1997: 2, 2015: 1}

#!/usr/bin/env python3
"""Family 5 (daily) through the REAL builder, pinned against family 4.

E-038 §3 specified family 5 as "`build_family4.py` with `PENTAD_DAYS = 1` and
its own `RECIPE_REV`, not a copy", so the builder is the shared one behind
`--days`. That sharing is only safe if the daily path is pinned where its
semantics genuinely differ, and the pentad path is pinned unchanged — both
tensors are built here from the SAME fixtures, by the same in-process calls
`tests/test_e034_family4.py` uses.

What is asserted:

  1. **Identity of the axis and the artefact**: daily bins, one per calendar
     day, recipe `f5r1`, and the SIDECAR layout — X lands beside the npz as a
     bare `.npy` that `tensor_io.load_tensor` returns as a read-only memmap.
     (165.6 GB will not fit any box's RAM through np.savez_compressed; the
     layout is the difference between family 5 existing and not.)
  2. **The base channels carry the day itself** — de-normalised, the daily
     tensor's cur_speed/ssh at day d match hypot/identity of the source's day
     d, to float16 round-trip tolerance.
  3. **rg stays one live bin per MONTH** — the whole reason the Chinchilla rg
     term does not scale with cadence (E-038 §2b). On a Jan+Feb axis that is
     exactly 2 live days (the 15ths), everything else missing.
  4. **THE SIGMA IDENTITY, the one designed-in cross-cadence invariant**: the
     daily tensor's tau_x_std at a pentad's midpoint equals the pentad
     tensor's tau_x_std for that bin. Family 4's within-pentad sigma IS the
     centred 5-day window sigma at the midpoint (same five days), so family 5
     samples the same physical quantity daily instead of every fifth day. If
     this drifts, the capacity x cadence matrix compares two different
     channels and calls it cadence.
  5. **The daily mean channel is the day's stress**, not a 5-day mean — the
     smoothing lives in sigma's window, never in the mean.
  6. **truth_daily.npz attaches on the daily axis** with the `rapid` alias the
     trainer reads.

E-041 adds recipe r2 (the appended `sst` channel) at both cadences, and two
more checks, again about a relationship rather than a value:

  7. **r1 is BIT-IDENTICAL inside r2 on the daily path too** — the sidecar
     layout means family 5's X is written and read back through a different
     code path from family 4's npz, so the appending is pinned here as well.
  8. **THE CADENCE IDENTITY FOR SST**: the pentad tensor's sst equals the
     MEAN OF THE FIVE daily tensors' sst for the same bin, and a single day
     does not equal the pentad value. Family 4 aggregates SST where family 5
     samples it, so if this drifts the two cadences are carrying different
     fields under one channel name — the same failure the wind-sigma
     identity (check 4) exists to catch, from the other direction.

    python3 tests/test_e034_family5.py
"""
import datetime as dt
import os
import sys
import tempfile

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "ml"))

import build_family4 as f4                                   # noqa: E402
import build_family3 as f3                                   # noqa: E402
from tensor_io import load_tensor                            # noqa: E402
from test_e034_family4 import (EPOCH, LATS, LONS, H, W,      # noqa: E402
                               START, END, write_fake_base025, write_rg,
                               write_sst, write_wind)

DBINS = list(range((START - EPOCH).days, (END - EPOCH).days + 1))
PBINS = list(range((START - EPOCH).days // 5, (END - EPOCH).days // 5 + 1))


def write_daily_dir(path, rng):
    """A miniature `aggregate_cadence --cadence daily` output."""
    os.makedirs(path, exist_ok=True)
    vals = {}
    for v in ("uo", "vo", "mlotst", "zos"):
        a = rng.normal(size=(len(DBINS), H, W)).astype(np.float32)
        if v == "mlotst":
            a = np.abs(a) * 200 + 20
        a[:, :2, :2] = np.nan                    # fixed land
        np.save(os.path.join(path, f"daily_mean_{v}.npy"), a)
        vals[v] = a
    np.savez(os.path.join(path, "index.npz"),
             bin_index=np.array(DBINS, np.int64),
             has_data=np.ones(len(DBINS), bool),
             epoch=np.array(str(EPOCH)), cadence_days=np.array(1),
             stat=np.array("mean"), min_days=np.array(1),
             bin_deg=np.array(0.25), grid_align=np.array("point"),
             lat=LATS, lon=LONS, vars=np.array(["uo", "vo", "mlotst", "zos"]))
    return vals


def write_pentad_from_daily(path, daily_vals):
    """The pentad aggregation OF THE SAME daily fields, so the two tensors
    describe one world and the sigma identity is checkable."""
    os.makedirs(path, exist_ok=True)
    day_of = {b: i for i, b in enumerate(DBINS)}
    for v, a in daily_vals.items():
        out = np.full((len(PBINS), H, W), np.nan, np.float32)
        for j, pb in enumerate(PBINS):
            rows = [day_of[d] for d in range(pb * 5, pb * 5 + 5)
                    if d in day_of]
            if rows:
                with np.errstate(invalid="ignore"):
                    out[j] = np.nanmean(a[rows], axis=0)
        np.save(os.path.join(path, f"pentad_mean_{v}.npy"), out)
    np.savez(os.path.join(path, "index.npz"),
             bin_index=np.array(PBINS, np.int64),
             has_data=np.ones(len(PBINS), bool),
             epoch=np.array(str(EPOCH)), cadence_days=np.array(5),
             stat=np.array("mean"), min_days=np.array(3),
             bin_deg=np.array(0.25), grid_align=np.array("point"),
             lat=LATS, lon=LONS, vars=np.array(["uo", "vo", "mlotst", "zos"]))


def write_truth_daily(cache):
    inside = (dt.date(2004, 1, 20) - EPOCH).days
    outside = (dt.date(2010, 6, 1) - EPOCH).days
    np.savez(os.path.join(cache, "truth_daily.npz"),
             epoch=np.array(str(EPOCH)), pentad_days=np.array(1),
             truth_rapid=np.array([[inside, 17.2], [outside, 16.0]]))
    return inside


def run_build(cache, src_dir, out, days, extra=()):
    f4.CACHE = cache
    f3.CACHE = cache
    f4.TRUTH_PENTAD = os.path.join(cache, "truth_pentad.npz")
    argv = sys.argv
    sys.argv = ["build", "--days", str(days), "--pentad-dir", src_dir,
                "--out", out, "--memmap", os.path.join(cache, f"b{days}.npy"),
                "--start", str(START), "--end", str(END), *extra]
    try:
        f4.main()
    finally:
        sys.argv = argv


def denorm(d):
    X = np.asarray(d["X"], np.float32)
    return X * d["norm"][:, 1] + d["norm"][:, 0]


def main():
    rng = np.random.default_rng(20260818)
    tmp = tempfile.mkdtemp(prefix="e034f5_")
    cache = os.path.join(tmp, "cache")
    os.makedirs(cache)

    daily_vals = write_daily_dir(os.path.join(tmp, "daily"), rng)
    write_pentad_from_daily(os.path.join(tmp, "pentad"), daily_vals)
    write_fake_base025(cache, LATS, LONS)
    write_rg(cache, rng)
    # The daily path's centred +-2-day sigma window reaches back into
    # December 2003, so that year is part of the fixture. Without it the
    # builder falls through to f3.fetch and a unit test downloads 120 MB
    # from PSL — which is also how it fails when that transfer truncates.
    write_wind(cache, rng, years=(2003, 2004))
    truth_row_bin = write_truth_daily(cache)
    # family 4's truth too, so ITS build stays on its usual path
    np.savez(os.path.join(cache, "truth_pentad.npz"),
             epoch=np.array(str(EPOCH)), pentad_days=np.array(5),
             truth_rapid=np.array([[PBINS[2], 17.0]]))

    out5 = os.path.join(tmp, "family5.npz")
    out4 = os.path.join(tmp, "family4.npz")
    run_build(cache, os.path.join(tmp, "daily"), out5, days=1)
    run_build(cache, os.path.join(tmp, "pentad"), out4, days=5)

    d5, d4 = load_tensor(out5), np.load(out4)

    # ---- 1: axis, recipe, layout ------------------------------------------
    assert str(d5["recipe"]) == "f5r1" and str(d5["cadence"]) == "daily"
    assert int(d5["pentad_days"]) == 1
    assert d5["bin_index"].tolist() == DBINS, "the daily axis is not the days"
    assert os.path.exists(out5[:-4] + "_X.npy"), "no sidecar — family 5 " \
        "through savez_compressed cannot be opened at full size"
    assert isinstance(d5["X"], np.memmap), "X did not come back memmapped"
    assert str(d4["recipe"]) == "f4r1", "the pentad path moved"
    print(f"  1. daily axis {len(DBINS)} bins, recipe f5r1, sidecar layout, "
          f"X memmapped; family 4 still f4r1")

    # ---- 2: base = the day itself -----------------------------------------
    raw5 = denorm(d5)
    day = 7
    want = np.hypot(daily_vals["uo"][day], daily_vals["vo"][day])
    got = raw5[day, :, :, 0]
    ok = np.isfinite(want) & np.isfinite(got)
    assert ok.sum() > 50 and np.allclose(got[ok], want[ok], atol=0.02), \
        "cur_speed at day 7 is not the day's own field"
    print(f"  2. base channels carry the day itself ({int(ok.sum())} cells "
          f"checked at day 7, f16 tolerance)")

    # ---- 3: rg = one live DAY per month -----------------------------------
    n_rg = int(d5["n_rg_live"])
    assert n_rg == 2, f"expected 2 live rg days (Jan 15, Feb 15), got {n_rg}"
    c = f4.C_BASE                                # first rg channel
    live = [i for i in range(len(DBINS))
            if np.isfinite(raw5[i, :, :, c]).any()]
    want_live = [(dt.date(2004, 1, 15) - EPOCH).days - DBINS[0],
                 (dt.date(2004, 2, 15) - EPOCH).days - DBINS[0]]
    assert live == want_live, f"live rg days {live} != the 15ths {want_live}"
    print("  3. rg: exactly the two 15ths are live — the rg term of the "
          "Chinchilla inventory does not scale with cadence")

    # ---- 4 & 5: the sigma identity, and the mean is the day ---------------
    raw4 = denorm(d4)
    cs = f4.C_BASE + f4.C_RG + 2                 # tau_x_std
    cm = f4.C_BASE + f4.C_RG                     # tau_x
    day_of = {b: i for i, b in enumerate(DBINS)}
    pent_of = {b: i for i, b in enumerate(d4["bin_index"].tolist())}
    checked = 0
    for pb in PBINS[1:-1]:                       # midpoint windows fully inside
        mid = pb * 5 + 2
        if mid not in day_of or pb not in pent_of:
            continue
        s5 = raw5[day_of[mid], :, :, cs]
        s4 = raw4[pent_of[pb], :, :, cs]
        ok = np.isfinite(s5) & np.isfinite(s4)
        if not ok.any():
            continue
        assert np.allclose(s5[ok], s4[ok], atol=0.02), (
            f"sigma identity broken at pentad {pb}: the daily centred 5-day "
            f"sigma at the midpoint must equal the within-pentad sigma")
        m5 = raw5[day_of[mid], :, :, cm]
        m4 = raw4[pent_of[pb], :, :, cm]
        okm = np.isfinite(m5) & np.isfinite(m4)
        # the pentad mean and the single day DIFFER (unless wind was constant)
        assert not np.allclose(m5[okm], m4[okm], atol=0.02), (
            "the daily tau_x equals the pentad mean — the mean channel is "
            "being smoothed, but the smoothing belongs to sigma's window only")
        checked += 1
    assert checked >= 5, f"only {checked} midpoints checked"
    print(f"  4. sigma identity holds at {checked} pentad midpoints — same "
          f"channel meaning at both cadences, by construction")
    print("  5. the daily mean channel is the DAY's stress, not a 5-day mean")

    # ---- 6: truth on the daily axis ---------------------------------------
    assert "rapid" in d5.files and "truth_rapid" in d5.files
    tr = np.asarray(d5["truth_rapid"])
    assert len(tr) == 1 and int(tr[0, 0]) == truth_row_bin - DBINS[0], (
        f"truth landed on row {tr[:, 0]}, expected "
        f"{truth_row_bin - DBINS[0]} (the out-of-axis label must be dropped)")
    print("  6. truth_daily attaches on the daily axis, out-of-axis labels "
          "dropped, `rapid` alias present")

    # =================== E-041: recipe r2 at both cadences ================
    sst_dir = os.path.join(tmp, "sst_na025")
    write_sst(sst_dir, PBINS, 5)
    r2 = ("--rev", "r2", "--sst-dir", sst_dir)
    out5b = os.path.join(tmp, "family5_r2.npz")
    out4b = os.path.join(tmp, "family4_r2.npz")
    run_build(cache, os.path.join(tmp, "daily"), out5b, days=1, extra=r2)
    run_build(cache, os.path.join(tmp, "pentad"), out4b, days=5, extra=r2)
    e5, e4 = load_tensor(out5b), np.load(out4b)

    # ---- 7: r1 is bit-identical inside r2, through the sidecar path -------
    assert str(e5["recipe"]) == "f5r2" and str(e4["recipe"]) == "f4r2"
    assert [str(c) for c in e5["chan"]] == list(f3.CHANS) + ["sst"]
    assert e5["X"].shape[3] == 40 and isinstance(e5["X"], np.memmap), \
        "the r2 daily tensor lost the sidecar layout"
    x1, x2 = np.asarray(d5["X"]), np.asarray(e5["X"])
    assert x1.dtype == x2.dtype == np.float16
    assert np.array_equal(x1, x2[..., :f4.NC], equal_nan=True), \
        "appending sst changed one of the 39 channels on the daily path"
    assert np.array_equal(np.asarray(d5["norm"]),
                          np.asarray(e5["norm"])[:f4.NC])
    print(f"  7. daily r2: {e5['X'].shape[3]} channels with `sst` last, "
          f"recipe f5r2, still memmapped — and channels 0..{f4.NC - 1} are "
          f"BIT-IDENTICAL to the r1 daily tensor")

    # ---- 8: the cadence identity for sst ---------------------------------
    r5 = denorm(e5)
    r4 = denorm(e4)
    c = f4.C_SST
    day_of = {b: i for i, b in enumerate(DBINS)}
    pent_of = {b: i for i, b in enumerate(e4["bin_index"].tolist())}
    checked = 0
    for pb in PBINS:
        rows = [day_of[dd] for dd in range(pb * 5, pb * 5 + 5) if dd in day_of]
        if len(rows) < 5 or pb not in pent_of:
            continue
        want = np.nanmean(np.stack([r5[i, :, :, c] for i in rows]), axis=0)
        got = r4[pent_of[pb], :, :, c]
        ok = np.isfinite(want) & np.isfinite(got)
        assert ok.sum() > 20, f"pentad {pb}: only {int(ok.sum())} cells"
        assert np.allclose(got[ok], want[ok], atol=0.05), (
            f"pentad {pb}: the pentad sst is not the mean of the five daily "
            f"sst values (max err {np.max(np.abs(got[ok] - want[ok])):.3f}) — "
            f"the two cadences carry different fields under one name")
        mid = r5[day_of[pb * 5 + 2], :, :, c]
        assert not np.allclose(mid[ok], got[ok], atol=0.05), (
            "one day equals the pentad mean — the fixture cannot tell a "
            "sample from a mean, so the check above cannot fail")
        checked += 1
    assert checked >= 5, f"only {checked} pentads checked"
    print(f"  8. sst cadence identity at {checked} pentads: the pentad value "
          f"is the mean of the five daily values, and one day is not")

    print("\ntests/test_e034_family5.py: all 8 checks passed")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""E-034: pin the streaming pentad aggregator against the one it replaced.

`ml/aggregate_cadence.py` was rewritten on 2026-08-16 because the original
could not finish the job it existed for: it held one accumulator per (bin,
variable) for the whole run, which is 227 GB of RAM and a 45.4 GB npz over
the 2,339 pentad bins of the 1993-2024 GLORYS12 archive. The rewrite streams
— a bin is flushed the moment the date crosses out of it — and that is a
change of BOOKKEEPING, not of arithmetic.

This file is what makes that last sentence checkable rather than claimed.
The pre-streaming algorithm is reimplemented inline as `reference()`, so the
reference cannot drift away from what it is pinning the way a recorded golden
file would. The assertions are:

  1. **Bit-identical, not close.** With no `--bin-deg` every value, every NaN
     and every dropped bin matches the reference exactly. ml/CLAUDE.md §4.9:
     prefer an exact identity to a threshold — a tolerance here would hide
     precisely the accumulation-order bug the rewrite could plausibly
     introduce.
  2. **The streaming claim is measured, not described.** Peak open bins must
     be 1. If a future edit reverts to per-file flushing this still produces
     correct numbers and quietly restores the memory growth, so the number
     is asserted rather than the outcome.
  3. **The straddle is the case that matters, so the fixture is built around
     it.** Pentad bins are counted from 1982-01-01 and know nothing about
     months, so the first bin of any month generally begins in the previous
     one. A bin holding fewer than `--min-days` days must be written MISSING
     rather than as a thinner mean, and a fixture whose months happened to
     align would test none of this.
  4. **Row i is bin_index[i].** The output is a memmap addressed by row, so
     an off-by-one between the row and the bin it claims to be is silent and
     fatal. Asserted with a month deliberately absent from the middle of the
     archive, which is the case that would expose it.
  5. **0.25 degree binning is the mean of the source cells, per day.**
     Checked against a hand-computed cell, because E-034 §2's "binned to 0.25
     degrees" is a recipe decision and a wrong weighting there would be
     invisible in every downstream number.

    python3 tests/test_e034_aggregate.py
"""
import datetime as dt
import os
import subprocess
import sys
import tempfile

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "..", "ml"))

EPOCH = dt.date(1982, 1, 1)
VARS = ("uo", "vo", "mlotst", "zos")
SCRIPT = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                      "..", "ml", "aggregate_cadence.py")

# 1/12 degree, like GLORYS12, and small enough to run in a second.
NLAT, NLON = 13, 13
LAT = np.arange(NLAT) / 12.0                 # 0 .. 1.0
LON = -1.0 + np.arange(NLON) / 12.0          # -1.0 .. 0.0
# `hours since 1950-01-01` is the CMEMS convention, so the fixture exercises
# the /24 branch rather than the easy one.
TIME_UNITS = "hours since 1950-01-01"
T0 = dt.date(1950, 1, 1)


def write_month(path, year, month, ndays, seed):
    """One synthetic daily chunk in the fetcher's shape and naming."""
    import netCDF4 as nc
    rng = np.random.default_rng(seed)
    d = nc.Dataset(path, "w")
    d.createDimension("time", ndays)
    d.createDimension("depth", 1)
    d.createDimension("latitude", NLAT)
    d.createDimension("longitude", NLON)
    t = d.createVariable("time", "f8", ("time",))
    t.units = TIME_UNITS
    d.createVariable("latitude", "f4", ("latitude",))[:] = LAT
    d.createVariable("longitude", "f4", ("longitude",))[:] = LON
    t[:] = [((dt.date(year, month, i + 1) - T0).days) * 24.0
            for i in range(ndays)]
    # A land mask that is FIXED in space (like a coastline) plus a few cells
    # that come and go per day (like an ice edge) — both kinds of missing
    # exist in the real product and they exercise different branches of the
    # per-cell day count.
    land = rng.random((NLAT, NLON)) < 0.25
    for v in VARS:
        var = d.createVariable(v, "f4", ("time", "depth", "latitude",
                                         "longitude"), fill_value=np.nan)
        a = rng.normal(size=(ndays, 1, NLAT, NLON)).astype(np.float32)
        a[:, 0][:, land] = np.nan
        flicker = rng.random((ndays, NLAT, NLON)) < 0.05
        a[:, 0][flicker] = np.nan
        var[:] = a
    d.close()


def reference(files, days=5, min_days=3):
    """The pre-streaming algorithm, verbatim in structure: every bin resident."""
    import netCDF4 as nc
    acc, cnt, shape = {}, {}, None
    for f in sorted(files):
        d = nc.Dataset(f)
        t = d.variables["time"]
        base = t.units.split("since", 1)[1].strip().split()[0]
        y, m, dd = (int(x) for x in base.replace("/", "-").split("-"))
        t0 = dt.date(y, m, dd)
        per = 24.0 if t.units.strip().lower().startswith("hours") else 1.0
        for i in range(len(t)):
            date = t0 + dt.timedelta(days=float(t[i]) / per)
            b = (date - EPOCH).days // days
            for v in VARS:
                var = d.variables[v]
                sl = var[i, 0] if var.ndim == 4 else var[i]
                arr = np.ma.filled(np.ma.masked_invalid(sl.astype(np.float32)),
                                   np.nan)
                shape = arr.shape
                key = (b, v)
                if key not in acc:
                    acc[key] = np.zeros(arr.shape, np.float64)
                    cnt[key] = np.zeros(arr.shape, np.int32)
                ok = np.isfinite(arr)
                acc[key][ok] += arr[ok]
                cnt[key] += ok
        d.close()
    bins = sorted({b for b, _ in acc})
    out = {}
    for v in VARS:
        stack = np.full((len(bins),) + shape, np.nan, np.float32)
        for j, b in enumerate(bins):
            n = cnt[(b, v)]
            with np.errstate(invalid="ignore"):
                stack[j] = np.where(n >= min_days,
                                    acc[(b, v)] / np.maximum(n, 1), np.nan)
        out[v] = stack
    return np.array(bins, np.int64), out


def run(src, out, *extra):
    p = subprocess.run([sys.executable, "-u", SCRIPT, "--in", src,
                        "--cadence", "pentad", "--out", out, *extra],
                       capture_output=True, text=True)
    if p.returncode:
        print(p.stdout)
        print(p.stderr, file=sys.stderr)
        raise SystemExit(f"aggregate_cadence.py exited {p.returncode}")
    return p.stdout


def load(out, stat="mean"):
    idx = np.load(os.path.join(out, "index.npz"))
    arrs = {v: np.asarray(np.load(os.path.join(out, f"pentad_{stat}_{v}.npy"),
                                  mmap_mode="r")) for v in VARS}
    return idx, arrs


def main():
    tmp = tempfile.mkdtemp(prefix="e034_")
    src = os.path.join(tmp, "daily")
    os.makedirs(src)
    # Three consecutive months. 1993-01-01 sits in bin 803, which STARTED on
    # 1992-12-29 — so the first bin carries 2 days and must be dropped, and
    # every month boundary after it straddles too. February is short, which
    # moves the straddle to a different offset each time.
    for (y, m, n, seed) in [(1993, 1, 31, 1), (1993, 2, 28, 2), (1993, 3, 31, 3)]:
        write_month(os.path.join(src, f"glorys_{y}{m:02d}.nc"), y, m, n, seed)

    # ---- 1 & 2: bit-identical to the reference, and streaming -------------
    out = os.path.join(tmp, "native")
    log = run(src, out)
    idx, got = load(out)
    ref_bins, ref = reference(sorted(
        os.path.join(src, f) for f in os.listdir(src) if f.endswith(".nc")))

    assert idx["bin_index"].tolist() == ref_bins.tolist(), (
        f"bin axis differs: {idx['bin_index'][:5]} vs {ref_bins[:5]}")
    for v in VARS:
        a, b = ref[v], got[v]
        assert a.shape == b.shape, f"{v}: shape {a.shape} vs {b.shape}"
        assert np.array_equal(np.isnan(a), np.isnan(b)), f"{v}: NaN mask differs"
        m = ~np.isnan(a)
        assert np.array_equal(a[m], b[m]), (
            f"{v}: values differ, max|diff| = {np.max(np.abs(a[m]-b[m]))}")
    print(f"  1. bit-identical to the pre-streaming implementation "
          f"({len(ref_bins)} bins x {ref['uo'].shape[1:]}, {len(VARS)} vars)")

    peak = [l for l in log.splitlines() if l.startswith("peak open bins")]
    assert peak, "the run did not report its peak open bins"
    n_open = int(peak[0].split()[3])
    assert n_open == 1, (
        f"peak open bins = {n_open}, expected 1 — a bin is being held past "
        f"the date that closes it, which is the memory growth this rewrite "
        f"removed")
    print(f"  2. streaming: peak open bins = {n_open} over 3 months")

    # ---- 3: the straddle bin is MISSING, not a thinner mean ---------------
    first = idx["bin_index"][0]
    start = EPOCH + dt.timedelta(days=int(first) * 5)
    assert start < dt.date(1993, 1, 1), (
        f"fixture does not straddle: first bin starts {start}")
    ndays_in = sum(1 for k in range(5)
                   if (start + dt.timedelta(days=k)) >= dt.date(1993, 1, 1))
    assert ndays_in < 3, "fixture's first bin is not short — nothing to drop"
    for v in VARS:
        assert np.isnan(got[v][0]).all(), (
            f"{v}: first bin has {ndays_in} day(s) < min_days but is not "
            f"all-NaN — a short bin was written as if it were a pentad mean")
    print(f"  3. straddle bin {first} ({start}, {ndays_in} day(s)) dropped "
          f"whole, per min_days=3")

    # ---- 4: row i is bin_index[i], with a month missing -------------------
    gap = os.path.join(tmp, "gap")
    os.makedirs(gap)
    for f in ("glorys_199301.nc", "glorys_199303.nc"):
        os.link(os.path.join(src, f), os.path.join(gap, f))
    gout = os.path.join(tmp, "gapout")
    run(gap, gout)
    gidx, ggot = load(gout)
    bins = gidx["bin_index"]
    has = gidx["has_data"]
    assert (np.diff(bins) == 1).all(), "bin_index is not contiguous"
    assert not has.all(), "a month was removed but every bin claims data"
    feb = [(EPOCH + dt.timedelta(days=int(b) * 5)) for b in bins[~has]]
    assert all(d.month == 2 for d in feb), (
        f"the bins without data are not February's: {feb[:3]}")
    for v in VARS:
        assert np.isnan(ggot[v][~has]).all(), (
            f"{v}: a bin no chunk covered carries values")
    # and the rows that DO have data still agree with the reference over the
    # same two chunks — i.e. the gap did not shift anything.
    rb, rr = reference([os.path.join(gap, f) for f in os.listdir(gap)])
    for v in VARS:
        for j, b in enumerate(rb):
            row = int(np.where(bins == b)[0][0])
            a, c = rr[v][j], ggot[v][row]
            assert np.array_equal(np.isnan(a), np.isnan(c)) and \
                np.array_equal(a[~np.isnan(a)], c[~np.isnan(c)]), \
                f"{v}: bin {b} landed on the wrong row"
    print(f"  4. row/bin alignment holds with {int((~has).sum())} bins "
          f"uncovered by any chunk")

    # ---- 5: 0.25 degree binning is the per-day mean of its source cells ---
    bout = os.path.join(tmp, "binned")
    run(src, bout, "--bin-deg", "0.25")
    bidx, bgot = load(bout)
    assert float(bidx["bin_deg"]) == 0.25
    nlat, nlon = len(bidx["lat"]), len(bidx["lon"])
    assert bgot["uo"].shape[1:] == (nlat, nlon), "grid/array disagree"
    # hand-compute one target cell: every source cell whose centre falls in
    # it, over every day of the bin, equally weighted.
    import aggregate_cadence as ag
    plan = ag.bin_plan(LAT.astype(np.float64), LON.astype(np.float64), 0.25)
    cell = (1, 2)
    flat = plan["flat"].reshape(NLAT, NLON)
    sel = flat == (cell[0] * plan["nlon"] + cell[1])
    assert sel.any(), "test cell has no source cells"
    import netCDF4 as nc
    target_bin = int(bidx["bin_index"][3])            # a full interior bin
    lo = EPOCH + dt.timedelta(days=target_bin * 5)
    per_day, pooled_sum, pooled_n = [], 0.0, 0
    for f in sorted(os.listdir(src)):
        d = nc.Dataset(os.path.join(src, f))
        t = d.variables["time"]
        for i in range(len(t)):
            date = T0 + dt.timedelta(days=float(t[i]) / 24.0)
            if not (lo <= date < lo + dt.timedelta(days=5)):
                continue
            a = np.ma.filled(np.ma.masked_invalid(
                d.variables["uo"][i, 0].astype(np.float32)), np.nan)
            vals = a[sel][np.isfinite(a[sel])]
            if vals.size:
                per_day.append(float(vals.mean()))
                pooled_sum += float(vals.sum())
                pooled_n += vals.size
        d.close()
    # THE ORDER IS THE DECISION. `aggregate_cadence.py` bins each DAILY slice
    # onto the 0.25 degree grid and only then averages over days, so a cell's
    # pentad value is the mean of its DAILY cell-means and `cnt` counts DAYS
    # — which is what keeps --min-days a statement about temporal coverage.
    # Pooling every source-cell-day instead (the `pooled` figure below) is
    # the plausible alternative, and it weights a day by how much of the cell
    # happened to be ice-free that day. The two differ here by ~4e-3, i.e.
    # far above float noise, so this assertion pins the choice rather than
    # merely exercising it.
    want = float(np.mean(per_day))
    pooled = pooled_sum / pooled_n
    got_cell = float(bgot["uo"][3][cell])
    assert abs(want - got_cell) < 2e-6, (
        f"0.25 deg cell {cell} of bin {target_bin}: got {got_cell!r}, "
        f"mean-of-daily-cell-means {want!r} over {len(per_day)} days")
    assert abs(pooled - got_cell) > 1e-4, (
        f"the two weightings agree to {abs(pooled-got_cell):g} on this "
        f"fixture, so this check cannot tell them apart — pick a cell whose "
        f"valid-source-cell count varies across the bin")
    print(f"  5. 0.25 deg cell = mean of {len(per_day)} daily cell-means "
          f"({got_cell:+.6f}), not the {pooled_n}-value pool "
          f"({pooled:+.6f}); grid {nlat}x{nlon}")

    print("\ntests/test_e034_aggregate.py: all 5 checks passed")


if __name__ == "__main__":
    main()

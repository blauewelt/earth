#!/usr/bin/env python3
"""Pin the DAILY cadence before spending an hour of Actions and 25 GB on it.

`aggregate_cadence.py --cadence daily` is documented as a "passthrough", and
passthrough is exactly the kind of claim that is never tested because it looks
like it cannot be wrong. It can: `days=1` walks the same binning, flushing and
`min_days` machinery as pentad, and every one of those has a branch that reads
differently when a bin holds one day instead of five.

  · `min_days` defaults to `3 if days == 5 else 1`. At 3 a daily run would
    drop EVERY bin — 11,688 planned bins, zero written, and the failure looks
    like an empty archive rather than a wrong constant.
  · the flush-when-closed logic keys on the bin index; at days=1 a bin closes
    on the very next slice, so `max_open` must stay at 1.
  · `bin_index` must be days-since-epoch, not months and not pentads, or
    `build_family5.py` would place 32 years of fields on the wrong axis while
    every shape still checked out.

What is asserted:

  1. **The values are the daily fields themselves.** A mean over one day is
     that day, so the output must be bit-identical to the input slice wherever
     the input is finite — not merely close. This is what "passthrough" means
     and it is checkable exactly.
  2. **`bin_index[i]` is (date - 1982-01-01).days** for every written row, and
     the planned range covers the whole span with no gaps.
  3. **`min_days` resolves to 1**, so no day is dropped: `has_data` is all
     True across the planned range and `dropped` is zero.
  4. **Missing data stays missing.** A land cell (NaN on every day) must come
     back NaN, not 0.0 — the aggregator's count-based mean is what guarantees
     that, and at days=1 the count is 0 or 1 with nothing in between.
  5. **Spatial binning to 0.25° point-aligned works at daily too**, on the
     same axes family 3 uses, because `build_family5.py` will refuse anything
     else exactly as `build_family4.py` does.

    python3 tests/test_e034_daily_cadence.py
"""
import datetime as dt
import os
import subprocess
import sys
import tempfile

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
SCRIPT = os.path.join(HERE, "..", "ml", "aggregate_cadence.py")
sys.path.insert(0, os.path.join(HERE, "..", "ml"))

from test_e034_aggregate import VARS, write_month, NLAT, NLON  # noqa: E402

EPOCH = dt.date(1982, 1, 1)


def run(src, out, *extra):
    p = subprocess.run([sys.executable, "-u", SCRIPT, "--in", src,
                        "--cadence", "daily", "--out", out, *extra],
                       capture_output=True, text=True)
    if p.returncode:
        print(p.stdout)
        print(p.stderr, file=sys.stderr)
        raise SystemExit(f"aggregate_cadence.py exited {p.returncode}")
    return p.stdout


def main():
    import netCDF4 as nc

    tmp = tempfile.mkdtemp(prefix="e034d_")
    src = os.path.join(tmp, "daily")
    os.makedirs(src)
    # Two adjacent months, the second one short, so the planned range spans a
    # month boundary — the case where a cadence bug would show as an off-by-one
    # rather than as a crash.
    write_month(os.path.join(src, "glorys_199301.nc"), 1993, 1, 31, seed=1)
    write_month(os.path.join(src, "glorys_199302.nc"), 1993, 2, 28, seed=2)

    out = os.path.join(tmp, "out")
    log = run(src, out)
    assert "min_days=1" in log, f"min_days did not resolve to 1:\n{log}"
    print("  1. --cadence daily resolves min_days=1 (at 3 every bin would drop)")

    idx = np.load(os.path.join(out, "index.npz"))
    bins = idx["bin_index"]
    has = idx["has_data"]
    assert int(idx["cadence_days"]) == 1, "cadence_days is not 1"
    n = 31 + 28
    assert len(bins) == n, f"{len(bins)} planned bins, expected {n}"
    want0 = (dt.date(1993, 1, 1) - EPOCH).days
    assert bins[0] == want0, f"first bin {bins[0]}, expected {want0}"
    assert (np.diff(bins) == 1).all(), "the daily axis is not contiguous"
    assert has.all(), f"{int((~has).sum())} planned day(s) had no chunk"
    print(f"  2. {n} contiguous daily bins, bin_index[0] = "
          f"(1993-01-01 - {EPOCH}).days = {want0}, has_data all True")

    # ---- 1 & 4: the values ARE the daily fields, NaN included --------------
    arrs = {v: np.asarray(np.load(os.path.join(out, f"daily_mean_{v}.npy"),
                                  mmap_mode="r")) for v in VARS}
    d = nc.Dataset(os.path.join(src, "glorys_199301.nc"))
    checked = nan_cells = 0
    for day in (0, 7, 30):
        for v in VARS:
            raw = np.ma.filled(np.ma.masked_invalid(
                d.variables[v][day, 0].astype(np.float32)), np.nan)
            got = arrs[v][day]
            ok = np.isfinite(raw)
            assert np.array_equal(got[ok], raw[ok]), (
                f"{v} day {day}: a one-day mean is not the day itself")
            assert np.isnan(got[~ok]).all(), (
                f"{v} day {day}: unobserved cells came back as numbers")
            checked += int(ok.sum())
            nan_cells += int((~ok).sum())
    d.close()
    print(f"  3. {checked:,} finite cells are bit-identical to the source day; "
          f"{nan_cells:,} unobserved cells stayed NaN")

    # ---- 5: 0.25 deg point-aligned binning at daily cadence ----------------
    out2 = os.path.join(tmp, "out025")
    run(src, out2, "--bin-deg", "0.25", "--grid-align", "point")
    idx2 = np.load(os.path.join(out2, "index.npz"))
    lat, lon = idx2["lat"], idx2["lon"]
    # The fixture's native grid is 0..1.0 lat and -1.0..0 lon at 1/12 deg, so
    # point alignment must land on the multiples of 0.25 INSIDE that box —
    # the same rule that reproduces family 3's 281x481 axes on the real one.
    assert np.allclose(lat, [0.0, 0.25, 0.5, 0.75, 1.0]), lat
    assert np.allclose(lon, [-1.0, -0.75, -0.5, -0.25, 0.0]), lon
    a2 = np.load(os.path.join(out2, "daily_mean_uo.npy"), mmap_mode="r")
    assert a2.shape == (n, len(lat), len(lon)), a2.shape
    assert np.isfinite(np.asarray(a2)).any(), "the binned output is all NaN"
    print(f"  4. --bin-deg 0.25 --grid-align point gives {len(lat)}x{len(lon)} "
          f"on the multiples of 0.25, shape {a2.shape}")

    # ---- the streaming property, at the cadence that stresses it least -----
    assert "peak open bins 1" in log, (
        "a daily bin closes on the next slice; more than one open bin means "
        f"the flush logic is holding state it does not need:\n{log}")
    print("  5. peak open bins = 1 — memory does not grow with the run")

    print("\ntests/test_e034_daily_cadence.py: all 5 checks passed")


if __name__ == "__main__":
    main()

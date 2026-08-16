#!/usr/bin/env python3
"""E-034: reduce DAILY fields to a coarser cadence. Pentad is derived, never fetched.

This file is the other half of the decision recorded in
`ml/fetch_glorys_daily.py`: **fetch daily once, derive every coarser cadence
by aggregation.** A 5-day mean is a pure reduction of the daily series, so a
separate pentad fetcher would be a second implementation of one rule — and
two implementations of one rule is the defect that drew a cosine on the
status page for a run that used expdecay, and that put a hand-traced corridor
in the frontend beside the evaluator's real one. There is one downloaded
byte-stream and one aggregation rule, here.

BIN DEFINITION, shared with `ml/build_truth_pentad.py` by construction rather
than by coincidence: fixed 5-day bins counted from 1982-01-01, index =
floor(days_since_epoch / 5). The state axis and the label axis therefore land
in the same bins, which is what makes a pentad label the target of a pentad
state without any re-alignment step.

WHAT IS AGGREGATED, AND HOW — the choice is per quantity, not global:

  uo, vo    -> mean of the VECTOR components, then speed = hypot(mean_u,
               mean_v) downstream. Averaging the SPEED instead would measure
               something different (a mean of magnitudes is not the magnitude
               of the mean) and would inflate quiet bins where the current
               reverses. The tensor's `cur_speed` channel is built from the
               binned components for exactly this reason.
  mlotst    -> mean. Deep-mixing events are what matter and a 5-day mean of a
               daily MLD keeps them; a max would track single-day storms and
               a min would erase the engine room.
  zos       -> mean.
  tau_std   -> NOT aggregable from a daily mean. A within-pentad standard
               deviation must be computed from the dailies directly, which is
               why `--stats std` exists: it emits the σ over each bin rather
               than the mean of anything.

Bins with fewer than `--min-days` contributing days are written as MISSING,
not as a thinner mean. A 5-day bin holding one day is not a pentad average,
and letting it pass would put a noisier sample beside 3,000 clean ones with
nothing marking it (ml/CLAUDE.md §5.22: never write a number you would have
to caveat later).

Run:
  python3 ml/aggregate_cadence.py --in ml/cache/glorys_daily --cadence pentad \\
      --out ml/cache/glorys_pentad
  python3 ml/aggregate_cadence.py --in ... --cadence daily --out ...   # passthrough
"""
import argparse
import datetime as dt
import glob
import os
import sys

import numpy as np

EPOCH = dt.date(1982, 1, 1)
CADENCE_DAYS = {"daily": 1, "pentad": 5}
MEAN_VARS = ("uo", "vo", "mlotst", "zos")


def bin_index(d, days):
    return (d - EPOCH).days // days


def bin_start(i, days):
    return EPOCH + dt.timedelta(days=int(i) * days)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="src", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--cadence", default="pentad", choices=sorted(CADENCE_DAYS))
    ap.add_argument("--stats", default="mean", choices=("mean", "std"))
    ap.add_argument("--min-days", type=int, default=0,
                    help="bins with fewer contributing days are dropped; "
                         "default 0 = cadence-appropriate (pentad: 3)")
    ap.add_argument("--vars", default=",".join(MEAN_VARS))
    a = ap.parse_args()

    import netCDF4 as nc

    days = CADENCE_DAYS[a.cadence]
    min_days = a.min_days or (3 if days == 5 else 1)
    want = [v for v in a.vars.split(",") if v]
    files = sorted(glob.glob(os.path.join(a.src, "*.nc")))
    if not files:
        sys.exit(f"no .nc files in {a.src}")
    os.makedirs(a.out, exist_ok=True)
    print(f"{len(files)} daily file(s) -> {a.cadence} bins "
          f"({days} d from {EPOCH}), stat={a.stats}, min_days={min_days}")

    # accumulate per bin across files; a bin can straddle a month boundary,
    # which is precisely why the accumulator is keyed by BIN and not by file.
    acc, acc2, cnt, shape = {}, {}, {}, None
    for f in files:
        d = nc.Dataset(f)
        t = d.variables["time"]
        units = t.units
        base = units.split("since", 1)[1].strip().split()[0]
        y, m, dd = (int(x) for x in base.replace("/", "-").split("-"))
        t0 = dt.date(y, m, dd)
        per = 24.0 if units.strip().lower().startswith("hours") else 1.0
        for i in range(len(t)):
            date = t0 + dt.timedelta(days=float(t[i]) / per)
            b = bin_index(date, days)
            for v in want:
                var = d.variables[v]
                sl = var[i, 0] if var.ndim == 4 else var[i]
                arr = np.ma.filled(np.ma.masked_invalid(sl.astype(np.float32)),
                                   np.nan)
                shape = arr.shape
                key = (b, v)
                if key not in acc:
                    acc[key] = np.zeros(arr.shape, np.float64)
                    acc2[key] = np.zeros(arr.shape, np.float64)
                    cnt[key] = np.zeros(arr.shape, np.int32)
                ok = np.isfinite(arr)
                acc[key][ok] += arr[ok]
                acc2[key][ok] += arr[ok] ** 2
                cnt[key] += ok
        d.close()
        print(f"  read {os.path.basename(f)}", flush=True)

    bins = sorted({b for b, _ in acc})
    kept = dropped = 0
    out_path = os.path.join(a.out, f"{a.cadence}_{a.stats}.npz")
    store = {"bin_index": np.array(bins, np.int64),
             "epoch": np.array(str(EPOCH)), "cadence_days": np.array(days),
             "stat": np.array(a.stats)}
    for v in want:
        stack = np.full((len(bins),) + shape, np.nan, np.float32)
        for j, b in enumerate(bins):
            key = (b, v)
            if key not in acc:
                continue
            n = cnt[key]
            good = n >= min_days
            if a.stats == "mean":
                with np.errstate(invalid="ignore"):
                    val = np.where(good, acc[key] / np.maximum(n, 1), np.nan)
            else:
                with np.errstate(invalid="ignore"):
                    mu = acc[key] / np.maximum(n, 1)
                    var_ = acc2[key] / np.maximum(n, 1) - mu ** 2
                    val = np.where(good, np.sqrt(np.maximum(var_, 0)), np.nan)
            stack[j] = val
        store[v] = stack
    for b in bins:
        n_any = max(int(cnt[(b, v)].max()) for v in want if (b, v) in cnt)
        if n_any >= min_days:
            kept += 1
        else:
            dropped += 1
    np.savez_compressed(out_path, **store)
    print(f"\n{len(bins)} bins · {kept} kept · {dropped} below min_days "
          f"({bin_start(bins[0], days)} .. {bin_start(bins[-1], days)})")
    print(f"wrote {out_path} ({os.path.getsize(out_path)/1e6:,.0f} MB)")


if __name__ == "__main__":
    main()

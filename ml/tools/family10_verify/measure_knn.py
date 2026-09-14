#!/usr/bin/env python3
"""Measure the k-th-nearest-neighbour distance and age in a family-10 tier-P
observation store — the measurement `docs/FAMILY10_DATA_HANDOVER.md` §6 asks
for and that the 2026-09-14 verification ran.

A family-10 tier-P store is a table of point observations (drifters, moorings,
ship tracks) sorted by five-day bin; `knearest` answers "what are the k nearest
observations to this place and pentad, and how far away are they". This script
asks that question at many anchors and reports the distribution, so the search
radii a consumer fixes are measured rather than guessed.

Two draws, following `ml/build_family8_argo.py::index_stats`:

  catalogue — anchors are the store's OWN rows, stratified equally across the
              years present so no single dense year dominates. This is "how far
              is the k-th neighbour at a place the system actually samples".
  globe     — anchors are uniform on the globe (lat U(-60, 60), lon U(-180,
              180), a random live bin). Includes land and empty ocean, where
              the honest answer is that nothing is measured.

The search runs with GENEROUS bounds (R_max = 5000 km, T_max = 60 d) so the
k-th neighbour is nearly always found, but with the metric scales L_km/T_days
PINNED to the suggested (R, T) — so the ranking is the one a consumer would
actually use, and dist_km[k-1] is exactly "the R_max that would have retained
the k-th token". n_R is reported separately at the suggested bounds.

Usage:
    python3 measure_knn.py --store gdp --dir /path/to/tensors/family10/gdp
    python3 measure_knn.py --store gtmba --dir ... --anchors 500 --json out.json

`--dir` also accepts a Hub prefix, e.g.
`chfrank/earth-tensors:tensors/family10/gdp`.
"""
import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "..", ".."))
from family10_store import Store            # noqa: E402

# The (k, R_max_km, T_max_days) each store's handover §6 suggests.
SUGGESTED = {"gdp": (8, 300.0, 10.0),
             "gtmba": (4, 1500.0, 15.0),
             "socat": (8, 500.0, 30.0),
             "slatrack": (16, 200.0, 10.0)}
R_GENEROUS, T_GENEROUS = 5000.0, 60.0


def pct(a):
    """median, p90, p99 of a list, NaN when it is empty."""
    a = np.asarray(a, float)
    if a.size == 0:
        return (float("nan"),) * 3
    return (float(np.median(a)), float(np.percentile(a, 90)),
            float(np.percentile(a, 99)))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--store", required=True, choices=sorted(SUGGESTED))
    ap.add_argument("--dir", required=True,
                    help="store directory, or a Hub prefix repo:path")
    ap.add_argument("--anchors", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=20260914)
    ap.add_argument("--json", default=None, help="write the result here too")
    a = ap.parse_args()

    k, r_sug, t_sug = SUGGESTED[a.store]
    rng = np.random.default_rng(a.seed)
    st = Store.open(a.dir)
    b = np.asarray(st["bin"])
    lat = np.asarray(st["lat"])
    lon = np.asarray(st["lon"])
    t = np.asarray(st["time_days"])

    # Catalogue anchors: rows inside the tensor epoch (bin >= 0), one equal
    # share per calendar year present.
    pos = np.nonzero(b >= 0)[0]
    yr = ((np.datetime64("1982-01-01") + (t[pos] * 86400).astype("timedelta64[s]"))
          .astype("datetime64[Y]").astype(int) + 1970)
    years = np.unique(yr)
    per = int(np.ceil(a.anchors / len(years)))
    picks = []
    for y in years:
        idx = pos[yr == y]
        picks.append(rng.choice(idx, size=min(per, len(idx)), replace=False))
    cat = np.concatenate(picks)
    rng.shuffle(cat)
    cat = cat[:a.anchors]

    # Globe anchors need a live bin inside the epoch to be answerable at all.
    cnt = np.diff(st.bin_offsets)
    binvals = np.arange(st.bin_first, st.bin_first + st.n_bins)
    live = binvals[(cnt > 0) & (binvals >= 0)]

    out = {"store": a.store, "k": k, "suggested_R_km": r_sug,
           "suggested_T_days": t_sug, "anchors": a.anchors, "seed": a.seed,
           "years": [int(years.min()), int(years.max())]}
    for draw in ("catalogue", "globe"):
        if draw == "catalogue":
            alat, alon, abin = lat[cat], lon[cat], b[cat].astype(int)
        else:
            n = a.anchors
            alat = rng.uniform(-60, 60, n).astype(np.float32)
            alon = rng.uniform(-180, 180, n).astype(np.float32)
            abin = rng.choice(live, size=n)
        dk, dtk, n_r_sug, ms = [], [], [], []
        short = 0
        for i in range(len(alat)):
            t0 = time.perf_counter()
            tok = st.knearest(float(alat[i]), float(alon[i]), bin=int(abin[i]),
                              k=k, R_max_km=R_GENEROUS, T_max_days=T_GENEROUS,
                              L_km=r_sug, T_days=t_sug)
            ms.append((time.perf_counter() - t0) * 1e3)
            if tok["n_found"] >= k:
                dk.append(float(tok["dist_km"][k - 1]))
                dtk.append(float(tok["dt_days"][k - 1]))
            else:
                short += 1
            tk = st.knearest(float(alat[i]), float(alon[i]), bin=int(abin[i]),
                             k=k, R_max_km=r_sug, T_max_days=t_sug)
            n_r_sug.append(tk["n_R"])
        n_r_sug = np.asarray(n_r_sug)
        out[draw] = {
            "n": int(len(alat)),
            "dist_kth_km": pct(dk),          # median, p90, p99
            "age_kth_days": pct(dtk),
            "frac_fewer_than_k_at_generous_bounds": short / len(alat),
            "n_R_at_suggested": {
                "median": float(np.median(n_r_sug)),
                "p90": float(np.percentile(n_r_sug, 90)),
                "mean": float(n_r_sug.mean()),
                "frac_zero": float((n_r_sug == 0).mean()),
                "frac_lt_k": float((n_r_sug < k).mean())},
            "ms_median": float(np.median(ms)),
            "ms_p90": float(np.percentile(ms, 90))}
        print(a.store, draw, json.dumps(out[draw]), flush=True)

    if a.json:
        with open(a.json, "w") as fh:
            json.dump(out, fh, indent=1)
        print("wrote", a.json)


if __name__ == "__main__":
    main()

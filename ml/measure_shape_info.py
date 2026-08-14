#!/usr/bin/env python3
"""E-024 scoping: WHICH array of input pixels carries the most information?

Chris, 2026-08-14, after the 222 km ring cut one-step error 3.9%: *"Can you
experiment with more input pixels, adding some further away? or is there a
larger array of pixels that would be particularly useful?"*

Same instrument as measure_ring_info.py (incremental held-out variance a
neighbour set explains on top of the centre's own 3-month history, pooled
ridge), with two changes:

1. SHAPES, not just radii: single rings at several radii and point counts,
   plus multi-scale combinations, all under the SAME centre sample so every
   comparison is paired.
2. ONE sequential read. The radius sweep thrashed at 445 km because
   fancy-indexing a 5.6 GB memmap by scattered columns degrades to random
   disk reads once the working set leaves the page cache (this machine has
   8 GB of RAM). Here every shape's pixels are collected FIRST, the union is
   read in contiguous time-chunks in one pass, and every gather after that
   is RAM. The 445+ km tail the first sweep never reached is measured here.

  python3 ml/measure_shape_info.py --z ml/cache/Z_run62.npy \
      --npz-small ml/cache/f3_small.npz --out ml/runs/shape_info.json
"""
import argparse
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from temporal import ring_offsets                       # noqa: E402
from measure_ring_info import ridge_fit_eval            # noqa: E402

# name -> [(r_km, n_points), ...]. The E-023 ring is re-measured under this
# sample as the reference; 8@445 and 8@890 are the tail the first sweep never
# reached; 4@222+4@445 keeps E-023's INPUT WIDTH while splitting it across two
# scales — if that wins, more information fits in the same model.
SHAPES = {
    "ring8@222":            [(222, 8)],
    "ring8@445":            [(445, 8)],
    "ring8@890":            [(890, 8)],
    "ring16@222":           [(222, 16)],
    "ring16@445":           [(445, 16)],
    "ring4@222+4@445":      [(222, 4), (445, 4)],
    "ring8@222+8@445":      [(222, 8), (445, 8)],
    "ring8@111+8@334":      [(111, 8), (334, 8)],
    "ring8@222+8@890":      [(222, 8), (890, 8)],
    "ring8@111+8@222+8@445": [(111, 8), (222, 8), (445, 8)],
}


def shape_nbr(lin, ys, xs, centres, shape, lats, dlat):
    """[n_centres, S] pixel indices, -1 = missing. Rings are circles on the
    ground (temporal.ring_offsets); successive rings are rotated half a sector
    against each other so a two-ring shape never stacks two points on one
    bearing."""
    H, W = lin.shape
    cols = []
    for ri, (r_km, n_pts) in enumerate(shape):
        block = np.full((len(centres), n_pts), -1, np.int64)
        for i, p in enumerate(centres):
            y, x = ys[p], xs[p]
            offs = ring_offsets(float(lats[y]), r_km, n_pts, dlat)
            if ri % 2 == 1:      # rotate alternate rings by half a sector
                offs = ring_offsets(float(lats[y]), r_km, 2 * n_pts, dlat)[1::2]
            for k, (dy, dx) in enumerate(offs):
                yy, xx = y + dy, x + dx
                if 0 <= yy < H and 0 <= xx < W:
                    block[i, k] = lin[yy, xx]
        cols.append(block)
    return np.concatenate(cols, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--z", required=True)
    ap.add_argument("--npz-small", required=True)
    ap.add_argument("--ocean-mask", default=os.path.join(HERE, "cache",
                                                         "ocean_mask.npy"))
    ap.add_argument("--centres", type=int, default=120)
    ap.add_argument("--lags", type=int, default=3)
    ap.add_argument("--holdout-years", default="2009,2017,2023")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=os.path.join(HERE, "runs",
                                                  "shape_info.json"))
    a = ap.parse_args()

    d = np.load(a.npz_small, allow_pickle=False)
    months = [str(m) for m in d["months"]]
    lats = d["lats"]
    moy = np.array([int(m[5:7]) - 1 for m in months])
    yr = np.array([int(m[:4]) for m in months])
    hold = set(int(y) for y in a.holdout_years.split(","))
    train_t = np.array([y not in hold for y in yr])
    dlat = float(np.round(np.diff(lats).mean(), 6))

    ocean = np.load(a.ocean_mask)
    ys, xs = np.where(ocean)
    P = len(ys)
    lin = np.full(ocean.shape, -1, np.int64)
    lin[ys, xs] = np.arange(P)
    Zm = np.load(a.z, mmap_mode="r")
    T, _, dz = Zm.shape

    rng = np.random.default_rng(a.seed)
    margin = 36            # 890 km at 70N is ~33 zonal cells
    ok = ((ys > margin) & (ys < ocean.shape[0] - margin)
          & (xs > margin) & (xs < ocean.shape[1] - margin))
    centres = rng.choice(np.where(ok)[0], min(a.centres, int(ok.sum())),
                         replace=False)

    nbrs = {name: shape_nbr(lin, ys, xs, centres, shape, lats, dlat)
            for name, shape in SHAPES.items()}
    all_px = np.unique(np.concatenate(
        [centres] + [nb[nb >= 0].ravel() for nb in nbrs.values()]))
    pos = np.full(P, -1, np.int64)
    pos[all_px] = np.arange(len(all_px))
    print(f"{len(centres)} centres · {len(all_px):,} unique pixels across "
          f"{len(SHAPES)} shapes — one sequential pass over the cache",
          flush=True)
    Zall = np.empty((T, len(all_px), dz), np.float16)
    for t0 in range(0, T, 16):
        t1 = min(t0 + 16, T)
        Zall[t0:t1] = np.asarray(Zm[t0:t1])[:, all_px]
    print("read complete", flush=True)
    Zall = Zall.astype(np.float32)
    for m in range(12):
        tr = train_t & (moy == m)
        Zall[moy == m] -= Zall[tr].mean(0)

    ci = pos[centres]
    L = a.lags
    idx = np.arange(L - 1, T - 1)
    base_cols, targ, split = [], [], []
    for i in range(len(centres)):
        c = Zall[:, ci[i]]
        base_cols.append(np.concatenate([c[idx - j] for j in range(L)], 1))
        targ.append(c[idx + 1])
        split.append(train_t[idx + 1])
    Xb = np.concatenate(base_cols)
    Y = np.concatenate(targ)
    tr = np.concatenate(split)
    mse_b, _ = ridge_fit_eval(Xb[tr], Y[tr], Xb[~tr], Y[~tr])
    print(f"baseline (own {L}-month history): held-out MSE {mse_b:.5f}",
          flush=True)

    results = {"centres": len(centres), "lags": L, "d_z": int(dz),
               "holdout_years": sorted(hold),
               "mse_baseline": round(mse_b, 6), "shapes": {}}
    for name, nb in nbrs.items():
        S = nb.shape[1]
        ring_cols = []
        for i in range(len(centres)):
            blocks = []
            for k in range(S):
                blocks.append(Zall[:, pos[nb[i, k]]][idx] if nb[i, k] >= 0
                              else np.zeros((len(idx), dz), np.float32))
            ring_cols.append(np.concatenate(blocks, 1))
        Xr = np.concatenate(ring_cols)
        X = np.concatenate([Xb, Xr], 1)
        mse_r, lam = ridge_fit_eval(X[tr], Y[tr], X[~tr], Y[~tr])
        gain = 1 - mse_r / mse_b
        results["shapes"][name] = {
            "n_neighbours": int(S),
            "ocean_frac": round(float((nb >= 0).mean()), 3),
            "mse": round(mse_r, 6), "gain": round(float(gain), 5), "lam": lam}
        print(f"  {name:<24} S={S:>2}  ocean {(nb >= 0).mean():.2f}  "
              f"MSE {mse_r:.5f}  gain {gain:+.4f}", flush=True)
        os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
        with open(a.out, "w") as f:
            json.dump(results, f, indent=1)
        del Xr, X, ring_cols
    print("wrote", a.out)


if __name__ == "__main__":
    main()

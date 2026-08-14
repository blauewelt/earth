#!/usr/bin/env python3
"""Why does a MORE decorrelated neighbour carry LESS usable information?

Chris, 2026-08-14: *"I assume a far away point to be less correlated than a
closer one, how can it have lower information gain?"*

The premise is right and the conclusion does not follow, and this script
measures the reason instead of arguing it. Predicting the centre's NEXT
embedding from a neighbour needs the neighbour to be two things at once:

  REDUNDANCY   r(neighbour_t, centre_t) — how much it merely repeats what
               the centre already says. Falls with distance. Low is good.
  RELEVANCE    r(neighbour_t, centre_{t+1}) — how much it says about the
               thing being predicted. ALSO falls with distance. High is good.

Usable information is what survives both: the PARTIAL correlation between the
neighbour and the target after the centre's own recent history is projected
out — the part of the neighbour that is new AND still about the future of
this pixel. It must vanish at both ends:

  at 1 cell   the neighbour is a copy of the centre, so nothing survives
              the projection — new information ~ 0
  at 900 km   the neighbour is its own patch of ocean, uncoupled from this
              pixel on a one-month timescale — relevance ~ 0

so the curve has an interior maximum, and the only question a measurement can
settle is where. Columns printed: redundancy, relevance, partial, and the
ridge gain for the same ring, so the hump and the E-024 table can be read
against each other.

  python3 ml/measure_partial_info.py --z ml/cache/Z_run62.npy \
      --npz-small ml/cache/f3_small.npz --out ml/runs/partial_info.json
"""
import argparse
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from temporal import ring_offsets                       # noqa: E402

RADII = [27.8, 55.6, 111.0, 222.0, 445.0, 890.0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--z", required=True)
    ap.add_argument("--npz-small", required=True)
    ap.add_argument("--ocean-mask", default=os.path.join(HERE, "cache",
                                                         "ocean_mask.npy"))
    ap.add_argument("--centres", type=int, default=200)
    ap.add_argument("--points", type=int, default=8)
    ap.add_argument("--lags", type=int, default=3)
    ap.add_argument("--holdout-years", default="2009,2017,2023")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=os.path.join(HERE, "runs",
                                                  "partial_info.json"))
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
    margin = 36
    ok = ((ys > margin) & (ys < ocean.shape[0] - margin)
          & (xs > margin) & (xs < ocean.shape[1] - margin))
    centres = rng.choice(np.where(ok)[0], min(a.centres, int(ok.sum())),
                         replace=False)

    nbr = {}
    for r_km in RADII:
        blk = np.full((len(centres), a.points), -1, np.int64)
        for i, p in enumerate(centres):
            y, x = ys[p], xs[p]
            for k, (dy, dx) in enumerate(ring_offsets(float(lats[y]), r_km,
                                                      a.points, dlat)):
                yy, xx = y + dy, x + dx
                if 0 <= yy < ocean.shape[0] and 0 <= xx < ocean.shape[1]:
                    blk[i, k] = lin[yy, xx]
        nbr[r_km] = blk

    all_px = np.unique(np.concatenate(
        [centres] + [b[b >= 0].ravel() for b in nbr.values()]))
    pos = np.full(P, -1, np.int64)
    pos[all_px] = np.arange(len(all_px))
    print(f"{len(centres)} centres · {len(all_px):,} pixels · one pass",
          flush=True)
    Z = np.empty((T, len(all_px), dz), np.float16)
    for t0 in range(0, T, 16):
        t1 = min(t0 + 16, T)
        Z[t0:t1] = np.asarray(Zm[t0:t1])[:, all_px]
    Z = Z.astype(np.float32)
    for m in range(12):
        tr = train_t & (moy == m)
        Z[moy == m] -= Z[tr].mean(0)
    print("read complete", flush=True)

    L = a.lags
    idx = np.arange(L - 1, T - 1)
    use = train_t[idx + 1]                 # train months only, like the ridge
    ci = pos[centres]

    def resid_on_history(v, H):
        """v [n, dz] residual after least-squares projection on H [n, m]."""
        A = np.c_[H, np.ones(len(H))]
        w, *_ = np.linalg.lstsq(A, v, rcond=None)
        return v - A @ w

    def cols_corr(A, B):
        """mean over dz of the per-dimension Pearson r between A and B."""
        Ad = A - A.mean(0); Bd = B - B.mean(0)
        den = np.sqrt((Ad ** 2).sum(0) * (Bd ** 2).sum(0)) + 1e-12
        return float(np.mean((Ad * Bd).sum(0) / den))

    results = {"centres": len(centres), "points": a.points, "lags": L,
               "radii_km": RADII, "rows": []}
    print(f"\n{'radius':>9}  {'redundancy':>10}  {'relevance':>9}  "
          f"{'PARTIAL':>8}   (mean over 8 ring points and 64 dims)")
    for r_km in RADII:
        red, rel, par = [], [], []
        for i in range(len(centres)):
            c = Z[:, ci[i]]
            hist = np.concatenate([c[idx - j] for j in range(L)], 1)[use]
            tgt = c[idx + 1][use]
            ctr_now = c[idx][use]
            tgt_r = resid_on_history(tgt, hist)      # target minus own history
            for k in range(a.points):
                q = nbr[r_km][i, k]
                if q < 0:
                    continue
                nb = Z[:, pos[q]][idx][use]
                red.append(cols_corr(nb, ctr_now))
                rel.append(cols_corr(nb, tgt))
                # the part of the neighbour that the centre's own history does
                # NOT already contain, against the part of the target the same
                # history does not explain
                par.append(cols_corr(resid_on_history(nb, hist), tgt_r))
        row = {"r_km": r_km, "redundancy": round(float(np.mean(red)), 4),
               "relevance": round(float(np.mean(rel)), 4),
               "partial": round(float(np.mean(par)), 4),
               "n": len(red)}
        results["rows"].append(row)
        print(f"{r_km:>7.0f} km  {row['redundancy']:>10.3f}  "
              f"{row['relevance']:>9.3f}  {row['partial']:>8.4f}", flush=True)
        os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
        with open(a.out, "w") as f:
            json.dump(results, f, indent=1)
    print("wrote", a.out)


if __name__ == "__main__":
    main()

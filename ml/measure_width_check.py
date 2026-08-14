#!/usr/bin/env python3
"""Does "a wider neighbour set is worse" survive more data, or was it my
estimator running out of samples?

E-024 reported that 16- and 24-point shapes lose to the 8-point ring at
222 km, and concluded that a larger array of input pixels is not worth
training. Re-running the 8-point rings with twice the centres then showed
every gain rising — 445 km went +0.0021 → +0.0105 and 890 km went NEGATIVE
→ +0.0022 — which is the signature of a per-column estimation cost, not of
absent information. So the width comparison has to be redone at a sample
size where that cost is small, or E-024's recommendation is an artefact.

Two changes from measure_shape_info.py:

* the ridge accumulates X'X and X'y in CHUNKS over centres instead of
  materialising the design matrix, so memory is O(cols²) rather than
  O(samples × cols). The 250-centre run of the earlier script was
  OOM-killed at the first 16-point shape on this 8 GB machine.
* it measures one thing — width at fixed radius, and width split across
  radii — so the answer is not buried in a ten-row table.

  python3 ml/measure_width_check.py --z ml/cache/Z_run62.npy \
      --npz-small ml/cache/f3_small.npz --centres 400
"""
import argparse
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from temporal import ring_offsets                       # noqa: E402
from measure_shape_info import shape_nbr                # noqa: E402

SHAPES = {
    "none (baseline)":       [],
    "ring8@222":             [(222, 8)],
    "ring16@222":            [(222, 16)],
    "ring8@222+8@445":       [(222, 8), (445, 8)],
    "ring8@111+8@222+8@445": [(111, 8), (222, 8), (445, 8)],
}
LAMS = (1e-1, 1, 10, 100, 1000)


def solve_ridge(G, B, lam, n_cols):
    reg = lam * np.eye(n_cols + 1)
    reg[-1, -1] = 0
    return np.linalg.solve(G + reg, B)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--z", required=True)
    ap.add_argument("--npz-small", required=True)
    ap.add_argument("--ocean-mask", default=os.path.join(HERE, "cache",
                                                         "ocean_mask.npy"))
    ap.add_argument("--centres", type=int, default=400)
    ap.add_argument("--lags", type=int, default=3)
    ap.add_argument("--holdout-years", default="2009,2017,2023")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=os.path.join(HERE, "runs",
                                                  "width_check.json"))
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
    nbrs = {n: (shape_nbr(lin, ys, xs, centres, sh, lats, dlat)
                if sh else np.zeros((len(centres), 0), np.int64))
            for n, sh in SHAPES.items()}
    all_px = np.unique(np.concatenate(
        [centres] + [nb[nb >= 0].ravel() for nb in nbrs.values()
                     if nb.size]))
    pos = np.full(P, -1, np.int64)
    pos[all_px] = np.arange(len(all_px))
    print(f"{len(centres)} centres · {len(all_px):,} pixels · one pass",
          flush=True)
    Z = np.empty((T, len(all_px), dz), np.float32)
    for t0 in range(0, T, 16):
        t1 = min(t0 + 16, T)
        Z[t0:t1] = np.asarray(Zm[t0:t1])[:, all_px].astype(np.float32)
    for m in range(12):
        tr = train_t & (moy == m)
        Z[moy == m] -= Z[tr].mean(0)
    print("read complete", flush=True)

    L = a.lags
    idx = np.arange(L - 1, T - 1)
    is_tr = train_t[idx + 1]
    ci = pos[centres]
    n_fit = int(0.8 * len(centres))          # split by CENTRE, not by row:
    # a val slice taken from the same pixels would share their history

    results = {"centres": len(centres), "lags": L, "shapes": {}}
    base_mse = None
    for name, nb in nbrs.items():
        S = nb.shape[1]
        ncol = L * dz + S * dz
        G = {k: np.zeros((ncol + 1, ncol + 1)) for k in ("fit", "all")}
        B = {k: np.zeros((ncol + 1, dz)) for k in ("fit", "all")}
        Gv, Bv, Yv = [], [], []
        te_rows = []
        for i in range(len(centres)):
            c = Z[:, ci[i]]
            blocks = [c[idx - j] for j in range(L)]
            for k in range(S):
                q = nb[i, k]
                blocks.append(Z[:, pos[q]][idx] if q >= 0
                              else np.zeros((len(idx), dz), np.float32))
            X = np.concatenate(blocks, 1)
            Y = c[idx + 1]
            A = np.c_[X, np.ones(len(X))]
            atr, ate = A[is_tr], A[~is_tr]
            ytr, yte = Y[is_tr], Y[~is_tr]
            G["all"] += atr.T @ atr
            B["all"] += atr.T @ ytr
            if i < n_fit:
                G["fit"] += atr.T @ atr
                B["fit"] += atr.T @ ytr
            else:
                Gv.append(atr); Yv.append(ytr)
            te_rows.append((ate, yte))
        Av, Yvv = np.concatenate(Gv), np.concatenate(Yv)
        best, best_lam = np.inf, LAMS[0]
        for lam in LAMS:
            W = solve_ridge(G["fit"], B["fit"], lam, ncol)
            m_ = float(((Av @ W - Yvv) ** 2).mean())
            if m_ < best:
                best, best_lam = m_, lam
        W = solve_ridge(G["all"], B["all"], best_lam, ncol)
        se = n = 0.0
        for ate, yte in te_rows:
            se += float(((ate @ W - yte) ** 2).sum()); n += yte.size
        mse = se / n
        if base_mse is None:
            base_mse = mse
        gain = 1 - mse / base_mse
        results["shapes"][name] = {"n_neighbours": int(S), "mse": round(mse, 6),
                                   "gain": round(float(gain), 5),
                                   "lam": best_lam}
        print(f"  {name:<24} S={S:>2}  MSE {mse:.5f}  gain {gain:+.4f}",
              flush=True)
        os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
        with open(a.out, "w") as f:
            json.dump(results, f, indent=1)
    print("wrote", a.out)


if __name__ == "__main__":
    main()

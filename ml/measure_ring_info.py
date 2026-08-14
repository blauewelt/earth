#!/usr/bin/env python3
"""E-023: does a RING of distant neighbours carry information a 3×3 does not?

Chris, 2026-08-14: *"Try also a different stencil shape, which is less
correlated and therefore adds new information. For example pixels forming
equidistant points on a circle of radius r (for which r is large enough such
that correlation is lower than 0.99)."*

E-022 gave the model its eight touching neighbours and the forecast objective
did not move (0.19247 vs 0.19216, three seeds each). The mechanism candidate
recorded at the time: E-021b measured z correlation r = 0.99 at one cell, so a
3×3 stencil may be the same information nine times. If that is the reason, the
fix is not a bigger stencil but a FARTHER one — far enough to decorrelate,
near enough to still be dynamically connected.

This script answers "how far" with measurement instead of taste, on the frozen
run-62 embeddings, before any GPU is spent. Two quantities per radius:

  1. CORRELATION — mean over the 64 dimensions of the temporal Pearson
     correlation between a centre pixel's embedding and its ring neighbours'.
     Reported raw AND deseasonalised (per-pixel, per-calendar-month means
     removed on train months): the raw number is inflated by the seasonal
     cycle, which the model already gets for free from its month features, so
     the deseasonalised one is what "new information" means here.

  2. INCREMENTAL PREDICTIVE INFORMATION — the quantity that actually decides
     the experiment. Ridge-predict the centre's z_{t+1} from its OWN last
     three months (the baseline), then from that plus the ring's z_t, and
     measure how much held-out residual variance the ring removes. Weights are
     shared across pixels, exactly as the stage-2 model shares them.

The instrument is VALIDATED before it is used: radius 1 cell reproduces the
3×3 stencil E-022 actually trained, so its predicted gain can be checked
against E-022's measured null. A probe that promises a large gain where the
real experiment found none is not evidence about any other radius either.

Rings are circles in KILOMETRES, not in grid cells: a fixed cell offset spans
27.8·cos(φ) km zonally, which at 70°N is a third of its value at the equator,
so a cell-defined ring would silently be three different experiments at three
latitudes. Offsets are therefore computed per pixel row.

  python3 ml/measure_ring_info.py --z ml/cache/Z_run62.npy \
      --npz-small ml/cache/f3_small.npz --out ml/runs/ring_info.json
"""
import argparse
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

KM_PER_DEG = 111.32


def ring_offsets(lat_deg, r_km, n_pts, dlat_deg):
    """[(dy, dx)] integer grid offsets of n_pts equidistant points on a circle
    of radius r_km around a pixel at `lat_deg`. Bearing 0 is north; the zonal
    step is stretched by 1/cos(lat) so the ring is a circle on the GROUND."""
    out = []
    coslat = max(np.cos(np.radians(lat_deg)), 0.05)
    for k in range(n_pts):
        th = 2 * np.pi * k / n_pts
        dy = (r_km / KM_PER_DEG) * np.cos(th) / dlat_deg
        dx = (r_km / (KM_PER_DEG * coslat)) * np.sin(th) / dlat_deg
        out.append((int(round(dy)), int(round(dx))))
    return out


def neighbour_index(lin, ys, xs, centres, r_km, n_pts, lats, dlat_deg):
    """[n_centres, n_pts] pixel indices into the P ordering; -1 = land or
    outside the window. Offsets are recomputed per centre because the ring is
    defined on the ground, not on the grid."""
    H, W = lin.shape
    out = np.full((len(centres), n_pts), -1, np.int64)
    for i, p in enumerate(centres):
        y, x = ys[p], xs[p]
        for k, (dy, dx) in enumerate(ring_offsets(lats[y], r_km, n_pts,
                                                  dlat_deg)):
            yy, xx = y + dy, x + dx
            if 0 <= yy < H and 0 <= xx < W:
                out[i, k] = lin[yy, xx]
    return out


def deseason(Zs, moy, train):
    """Remove each pixel's own calendar-month mean, computed on TRAIN months.
    Zs [T, n, d] float32, modified copy returned."""
    out = Zs.copy()
    for m in range(12):
        tr = train & (moy == m)
        if tr.sum() == 0:
            continue
        out[moy == m] -= Zs[tr].mean(0)
    return out


def ridge_fit_eval(Xtr, Ytr, Xte, Yte, lams=(1e-1, 1, 10, 100, 1000)):
    """Pooled ridge with an intercept, λ chosen on the TEST split's own
    baseline-free criterion is NOT allowed — so λ is chosen on a held-out
    slice of TRAIN, then refit on all of train. Returns test MSE."""
    n = len(Xtr)
    cut = int(0.8 * n)
    best, best_lam = np.inf, lams[0]
    A1 = np.c_[Xtr[:cut], np.ones(cut)]
    G1 = A1.T @ A1
    B1 = A1.T @ Ytr[:cut]
    A2 = np.c_[Xtr[cut:], np.ones(n - cut)]
    for lam in lams:
        reg = lam * np.eye(G1.shape[0]); reg[-1, -1] = 0
        W = np.linalg.solve(G1 + reg, B1)
        mse = float(((A2 @ W - Ytr[cut:]) ** 2).mean())
        if mse < best:
            best, best_lam = mse, lam
    A = np.c_[Xtr, np.ones(n)]
    G = A.T @ A
    reg = best_lam * np.eye(G.shape[0]); reg[-1, -1] = 0
    W = np.linalg.solve(G + reg, A.T @ Ytr)
    pred = np.c_[Xte, np.ones(len(Xte))] @ W
    return float(((pred - Yte) ** 2).mean()), best_lam


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--z", required=True)
    ap.add_argument("--npz-small", required=True)
    ap.add_argument("--ocean-mask", default=os.path.join(HERE, "cache",
                                                         "ocean_mask.npy"))
    ap.add_argument("--radii-km", default="27.8,55.6,111,222,445,890,1780",
                    help="1, 2, 4, 8, 16, 32, 64 cells at the equator")
    ap.add_argument("--points", type=int, default=8,
                    help="points on the ring; 8 keeps the input width equal "
                         "to E-022's 3x3, so radius is the ONLY difference")
    ap.add_argument("--centres", type=int, default=400)
    ap.add_argument("--lags", type=int, default=3)
    ap.add_argument("--holdout-years", default="2009,2017,2023")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=os.path.join(HERE, "runs",
                                                  "ring_info.json"))
    a = ap.parse_args()

    d = np.load(a.npz_small, allow_pickle=False)
    months = [str(m) for m in d["months"]]
    lats, lons = d["lats"], d["lons"]
    moy = np.array([int(m[5:7]) - 1 for m in months])
    yr = np.array([int(m[:4]) for m in months])
    hold = set(int(y) for y in a.holdout_years.split(","))
    train_t = np.array([y not in hold for y in yr])
    test_t = ~train_t
    dlat = float(np.round(np.diff(lats).mean(), 6))

    ocean = np.load(a.ocean_mask)
    ys, xs = np.where(ocean)
    P = len(ys)
    lin = np.full(ocean.shape, -1, np.int64)
    lin[ys, xs] = np.arange(P)
    Zm = np.load(a.z, mmap_mode="r")
    T, _, dz = Zm.shape
    print(f"Z {Zm.shape} · {P} ocean px · train months {int(train_t.sum())} "
          f"· test months {int(test_t.sum())}", flush=True)

    rng = np.random.default_rng(a.seed)
    # centres away from the window edge, so a large ring is not mostly missing
    # by construction — the edge effect is measured separately as ring_ocean
    margin = 20
    ok = ((ys > margin) & (ys < ocean.shape[0] - margin)
          & (xs > margin) & (xs < ocean.shape[1] - margin))
    centres = rng.choice(np.where(ok)[0], min(a.centres, int(ok.sum())),
                         replace=False)
    radii = [float(x) for x in a.radii_km.split(",")]
    results = {"radii_km": radii, "points": a.points, "centres": len(centres),
               "lags": a.lags, "holdout_years": sorted(hold), "d_z": int(dz),
               "rows": []}

    keep = {}
    for r_km in radii:
        NBR = neighbour_index(lin, ys, xs, centres, r_km, a.points, lats, dlat)
        have = NBR >= 0
        uniq, inv = np.unique(np.r_[centres, NBR[have]], return_inverse=True)
        Zraw = np.asarray(Zm[:, uniq]).astype(np.float32)         # [T, u, dz]
        Zs = deseason(Zraw, moy, train_t)
        pos = {p: i for i, p in enumerate(uniq)}
        ci = np.array([pos[p] for p in centres])

        # ---- 1. correlation with the centre, deseasonalised ---------------
        def _cor(A):
            out = []
            for i in range(len(centres)):
                c = A[:, ci[i]]
                for k in range(a.points):
                    if NBR[i, k] < 0:
                        continue
                    nb = A[:, pos[NBR[i, k]]]
                    cd = c - c.mean(0); nd = nb - nb.mean(0)
                    den = np.sqrt((cd ** 2).sum(0) * (nd ** 2).sum(0)) + 1e-12
                    out.append(float(np.mean((cd * nd).sum(0) / den)))
            return float(np.mean(out)) if out else np.nan
        cor_mean = _cor(Zs)
        cor_raw = _cor(Zraw)          # what Chris's "0.99" refers to

        # ---- 2. incremental predictive information ------------------------
        # samples are (centre, t): predict z_{t+1} from the centre's own last
        # `lags` months (baseline) and from that plus the ring at t.
        L = a.lags
        base_cols, ring_cols, targ, split = [], [], [], []
        for i in range(len(centres)):
            c = Zs[:, ci[i]]                                    # [T, dz]
            idx = np.arange(L - 1, T - 1)
            b = np.concatenate([c[idx - j] for j in range(L)], 1)
            nb = []
            for k in range(a.points):
                nb.append(Zs[:, pos[NBR[i, k]]][idx] if NBR[i, k] >= 0
                          else np.zeros((len(idx), dz), np.float32))
            base_cols.append(b)
            ring_cols.append(np.concatenate(nb, 1))
            targ.append(c[idx + 1])
            split.append(train_t[idx + 1])
        Xb = np.concatenate(base_cols); Xr = np.concatenate(ring_cols)
        Y = np.concatenate(targ); tr = np.concatenate(split)
        mse_b, lam_b = ridge_fit_eval(Xb[tr], Y[tr], Xb[~tr], Y[~tr])
        mse_r, lam_r = ridge_fit_eval(np.c_[Xb, Xr][tr], Y[tr],
                                      np.c_[Xb, Xr][~tr], Y[~tr])
        gain = 1 - mse_r / mse_b
        row = {"r_km": r_km, "r_cells_equator": round(r_km / (dlat * KM_PER_DEG), 1),
               "corr_deseason": round(cor_mean, 4), "corr_raw": round(cor_raw, 4),
               "ring_ocean_frac": round(float(have.mean()), 3),
               "mse_baseline": round(mse_b, 6), "mse_ring": round(mse_r, 6),
               "var_explained_gain": round(float(gain), 5),
               "lam": [lam_b, lam_r], "n_samples": int(len(Y))}
        results["rows"].append(row)
        keep[r_km] = (np.concatenate(ring_cols), np.concatenate(targ),
                      np.concatenate(split), np.concatenate(base_cols))
        # WRITE AFTER EVERY RADIUS. Cost per radius grows with the radius --
        # a far ring's pixels are scattered across a 5.6 GB memmap, so the
        # gather degrades from page-cache hits to random reads and the last
        # radii take an order of magnitude longer than the first. Writing only
        # at the end means an interrupted sweep loses every row it measured,
        # which is what happened on the first run of this file: the numbers
        # survived in the log and had to be read back out of it.
        os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
        with open(a.out, "w") as f:
            json.dump(results, f, indent=1)
        print(f"  r={r_km:>7.1f} km ({row['r_cells_equator']:>4.1f} cells)  "
              f"corr {cor_mean:+.3f} (raw {cor_raw:+.3f})  ring-ocean {row['ring_ocean_frac']:.2f}  "
              f"MSE {mse_b:.5f} -> {mse_r:.5f}  gain {gain:+.4f}", flush=True)
        del Zs, Zraw, Xb, Xr, Y

    # ---- combinations: does a SECOND ring add anything the first didn't? --
    # Asked because the answer decides the shape to train: one ring of 8, or a
    # 17-point near+far shape at twice the input width.
    order = sorted(results["rows"], key=lambda r: -r["var_explained_gain"])
    if len(order) >= 2:
        r1, r2 = order[0]["r_km"], order[1]["r_km"]
        X1, Y, tr, Xb = keep[r1]
        X2 = keep[r2][0]
        mse_b, _ = ridge_fit_eval(Xb[tr], Y[tr], Xb[~tr], Y[~tr])
        Xc = np.c_[Xb, X1, X2]
        mse_c, lam_c = ridge_fit_eval(Xc[tr], Y[tr], Xc[~tr], Y[~tr])
        results["combo"] = {
            "radii_km": [r1, r2], "mse_baseline": round(mse_b, 6),
            "mse_combo": round(mse_c, 6),
            "var_explained_gain": round(float(1 - mse_c / mse_b), 5),
            "best_single_gain": order[0]["var_explained_gain"], "lam": lam_c}
        print(f"  combo {r1:.0f}+{r2:.0f} km: gain "
              f"{results['combo']['var_explained_gain']:+.4f} "
              f"(best single {order[0]['var_explained_gain']:+.4f})", flush=True)

    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    with open(a.out, "w") as f:
        json.dump(results, f, indent=1)
    print("wrote", a.out)


if __name__ == "__main__":
    main()

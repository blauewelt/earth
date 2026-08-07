#!/usr/bin/env python3
"""Sequence probe with the seasonality-proof protocol.

Three defences against the calendar (the embedding receives month-of-year as
an input token, so any seasonal signal in a metric is free points):

  1. The RAPID target is DESEASONALISED: its own monthly climatology,
     computed from train years only, is subtracted before probing.
  2. A seasonal-only floor is reported: a ridge from (sin, cos) month alone.
     On the raw target it shows how much of the correlation was calendar;
     on the deseasonalised target it should sit near zero by construction.
  3. Lambda is selected on a train-internal validation tail; held-out years
     are touched exactly once per configuration.

Usage:  python3 ml/probe_sequence.py --run pilot12_anom --anomaly
        python3 ml/probe_sequence.py --run pilot12            (state space)
"""
import argparse
import json
import os
import sys
import warnings

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model import PixelMAE

HERE = os.path.dirname(os.path.abspath(__file__))


def ridge_r(F, y, tr, te):
    """Standardise, pick lambda on a train tail, score Pearson r on te."""
    F = (F - F[tr].mean(0)) / (F[tr].std(0) + 1e-9)
    order = np.argsort(np.where(tr)[0])
    tr_idx = np.where(tr)[0]
    fit, val = tr_idx[: int(0.8 * len(tr_idx))], tr_idx[int(0.8 * len(tr_idx)):]

    def solve(idx, lam):
        A = np.c_[F[idx], np.ones(len(idx))]
        reg = lam * np.eye(A.shape[1]); reg[-1, -1] = 0
        return np.linalg.solve(A.T @ A + reg, A.T @ y[idx])

    best, best_r = 1.0, -np.inf
    for lam in (1e-2, 1e-1, 1, 10, 100, 1000):
        w = solve(fit, lam)
        p = np.c_[F[val], np.ones(len(val))] @ w
        r = np.corrcoef(p, y[val])[0, 1]
        if np.isfinite(r) and r > best_r:
            best_r, best = r, lam
    w = solve(tr_idx, best)
    p = np.c_[F, np.ones(len(F))] @ w
    return round(float(np.corrcoef(p[te], y[te])[0, 1]), 3), best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="pilot12_anom")
    ap.add_argument("--anomaly", action="store_true",
                    help="apply the same anomaly transform the run trained with")
    a = ap.parse_args()

    ck = torch.load(os.path.join(HERE, "runs", a.run, "pixelmae.pt"),
                    map_location="cpu", weights_only=False)
    d = np.load(os.path.join(HERE, "cache", "na_pixels.npz"))
    X = d["X"].copy()
    months = [str(m) for m in d["months"]]
    lats, lons = d["lats"], d["lons"]
    T, H, W, C = X.shape
    hold_years = set(ck["args"]["holdout_years"].split(","))
    t_hold = np.array([m[:4] in hold_years for m in months])
    lo, hi = (float(v) for v in ck["args"]["holdout_lon"].split(","))
    x_hold = (lons >= lo) & (lons < hi)
    moy = np.array([int(m[5:7]) - 1 for m in months])

    if a.anomaly:   # identical transform to train.py --anomaly
        dynamic = [c for c in range(C)
                   if np.nanstd(np.nanmean(X[..., c], axis=(1, 2))) > 1e-6]
        clim = np.full((12, H, W, C), np.nan, dtype=np.float32)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            for m in range(12):
                clim[m] = np.nanmean(X[(moy == m) & ~t_hold], axis=0)
        for c in dynamic:
            X[..., c] = X[..., c] - clim[moy][..., c]
            v = X[..., c][np.isfinite(X[..., c]) & ~t_hold[:, None, None]
                          & ~x_hold[None, None, :]]
            X[..., c] = (X[..., c] - v.mean()) / (v.std() + 1e-6)

    model = PixelMAE(n_chan=C, d_z=ck["d_z"])
    model.load_state_dict(ck["model"])
    model.eval()

    ctx_all = np.stack([np.sin(2 * np.pi * moy / 12), np.cos(2 * np.pi * moy / 12)], 1)
    from temporal import RAPID_LON
    sec_y = int(np.argmin(np.abs(lats - 26.5)))
    sec_x = np.where(np.isfinite(d["X"][0, sec_y, :, 0])
                     & (lons >= RAPID_LON[0]) & (lons <= RAPID_LON[1]))[0]
    Xt = torch.from_numpy(np.nan_to_num(X, nan=0.0))
    OBS = torch.from_numpy(np.isfinite(X))

    print(f"embedding the 26.5N section ({a.run}, anomaly={a.anomaly}) …")
    emb = np.zeros((T, ck["d_z"]), dtype=np.float32)
    with torch.no_grad():
        for t in range(T):
            n = len(sec_x)
            ctx = np.concatenate([np.tile(ctx_all[t], (n, 1)),
                                  (np.full(n, lats[sec_y]) / 90)[:, None],
                                  (lons[sec_x] / 180)[:, None]], 1)
            z = model.encode(Xt[t, sec_y, sec_x], OBS[t, sec_y, sec_x],
                             torch.zeros(n, C, dtype=torch.bool),
                             torch.as_tensor(ctx, dtype=torch.float32))
            emb[t] = z.mean(0).numpy()

    rapid = d["rapid"]
    ridx = rapid[:, 0].astype(int)
    rv_raw = rapid[:, 1].copy()
    # deseasonalise the TARGET with train-year monthly means
    rmoy = moy[ridx]
    tr_all = ~t_hold[ridx]
    rclim = np.array([rv_raw[tr_all & (rmoy == m)].mean() for m in range(12)])
    rv_des = rv_raw - rclim[rmoy]

    out = {"run": a.run, "anomaly_space": bool(a.anomaly), "sweep": []}
    # seasonal-only floor (month features, no data)
    sf = np.stack([np.sin(2 * np.pi * rmoy / 12), np.cos(2 * np.pi * rmoy / 12)], 1)
    te_all = t_hold[ridx]
    out["seasonal_floor_raw"], _ = ridge_r(sf, rv_raw, tr_all, te_all)
    out["seasonal_floor_deseas"], _ = ridge_r(sf, rv_des, tr_all, te_all)

    for K in (1, 3, 6, 12, 24):
        ok = ridx >= K - 1
        ri = ridx[ok]
        F = np.stack([np.concatenate([emb[t - k] for k in range(K)]) for t in ri])
        tr, te = ~t_hold[ri], t_hold[ri]
        r_raw, lam1 = ridge_r(F, rv_raw[ok], tr, te)
        r_des, lam2 = ridge_r(F, rv_des[ok], tr, te)
        out["sweep"].append({"K": K, "r_raw": r_raw, "r_deseasonalised": r_des,
                             "n_test": int(te.sum())})
        print(f"  K={K:>2}  raw r={r_raw:+.3f}   deseasonalised r={r_des:+.3f}")

    print(f"seasonal-only floor: raw {out['seasonal_floor_raw']:+.3f} · "
          f"deseasonalised {out['seasonal_floor_deseas']:+.3f}")
    path = os.path.join(HERE, "runs", a.run, "probe_sequence.json")
    json.dump(out, open(path, "w"), indent=2)
    print("wrote", path)


if __name__ == "__main__":
    main()

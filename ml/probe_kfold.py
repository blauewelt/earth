#!/usr/bin/env python3
"""Year-blocked k-fold RAPID probe — the power upgrade METRICS.md promised.

Doubling the held-out years would only buy sqrt(2): SE(r) 0.29 -> 0.20,
still coarse, and it starves codec training. This instead makes EVERY
RAPID month a test month exactly once: fold by calendar year (blocked, so
autocorrelation cannot leak across the fit/test line), fit the ridge on
the other years (lambda on an inner tail), predict the held year, and
score ONE r over all ~240 assembled out-of-fold predictions. n_eff ~ 100
after autocorrelation -> SE ~ 0.10, i.e. three times the resolution, for
pure compute.

The honest caveat, stated where the numbers are made: the codec itself
was trained (self-supervised, RAPID never an input) with only 3 held-out
years, so embeddings of the other years have seen those months' FIELDS.
The transport was never seen by anything, but treat absolute k-fold r as
mildly optimistic; codec-to-codec COMPARISONS are fair. A block bootstrap
(resampling whole years) puts a CI on each r.

Usage: python3 ml/probe_kfold.py --runs dz8 dz16 actions dz64
"""
import argparse
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model import PixelMAE
from trainprobe import anomaly_transform
from temporal import embed_everything

HERE = os.path.dirname(os.path.abspath(__file__))


def kfold_r(F, y, years, lams=(1e-2, 1e-1, 1, 10, 100, 1000), boot=2000, seed=0):
    """Grouped-by-year k-fold ridge; returns (r, lo95, hi95, n)."""
    F = np.asarray(F, float)
    y = np.asarray(y, float)
    pred = np.full(len(y), np.nan)
    for yr in np.unique(years):
        te = years == yr
        tr = ~te
        idx = np.where(tr)[0]
        fit, val = idx[: int(0.8 * len(idx))], idx[int(0.8 * len(idx)):]
        mu, sd = F[tr].mean(0), F[tr].std(0) + 1e-9
        Fz = (F - mu) / sd

        def solve(sel, lam):
            A = np.c_[Fz[sel], np.ones(len(sel))]
            reg = lam * np.eye(A.shape[1]); reg[-1, -1] = 0
            return np.linalg.solve(A.T @ A + reg, A.T @ y[sel])

        best, best_r = lams[0], -np.inf
        for lam in lams:
            w = solve(fit, lam)
            p = np.c_[Fz[val], np.ones(len(val))] @ w
            r = np.corrcoef(p, y[val])[0, 1]
            if np.isfinite(r) and r > best_r:
                best_r, best = r, lam
        w = solve(idx, best)
        pred[te] = np.c_[Fz[te], np.ones(int(te.sum()))] @ w
    ok = np.isfinite(pred)
    r = float(np.corrcoef(pred[ok], y[ok])[0, 1])
    # block bootstrap over whole years
    rng = np.random.default_rng(seed)
    yrs = np.unique(years)
    rs = []
    for _ in range(boot):
        pick = rng.choice(yrs, len(yrs), replace=True)
        sel = np.concatenate([np.where(years == p)[0] for p in pick])
        if np.isfinite(pred[sel]).sum() > 24 and np.std(y[sel]) > 0:
            rs.append(np.corrcoef(pred[sel], y[sel])[0, 1])
    lo, hi = np.percentile(rs, [2.5, 97.5])
    return r, float(lo), float(hi), int(ok.sum())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", default=["actions"])
    a = ap.parse_args()
    d = np.load(os.path.join(HERE, "cache", "na_pixels.npz"))
    months = [str(m) for m in d["months"]]
    moy = np.array([int(m[5:7]) - 1 for m in months])
    yr = np.array([int(m[:4]) for m in months])
    lats, lons = d["lats"], d["lons"]
    rapid = d["rapid"]
    ridx = rapid[:, 0].astype(int)
    rv = rapid[:, 1].copy()
    ryr = yr[ridx]
    rmoy = moy[ridx]
    # deseasonalise with the OVERALL monthly climatology (every month is test
    # in some fold; per-fold climatologies differ by <0.1 Sv and would leak
    # nothing either way — chosen for simplicity and stated here).
    rclim = np.array([rv[rmoy == m].mean() for m in range(12)])
    rv_des = rv - rclim[rmoy]
    ctx_all = np.stack([np.sin(2 * np.pi * moy / 12), np.cos(2 * np.pi * moy / 12)], 1)
    sec_y = int(np.argmin(np.abs(lats - 26.5)))
    ocean = np.isfinite(d["X"][..., 0]).any(axis=0)
    ys, xs = np.where(ocean)
    sec_sel = np.where(ys == sec_y)[0]

    out = {}
    for run in a.runs:
        ck = torch.load(os.path.join(HERE, "runs", run, "pixelmae.pt"),
                        map_location="cpu", weights_only=False)
        X = d["X"].copy()
        t_hold = np.array([m[:4] in set(ck["args"]["holdout_years"].split(","))
                           for m in months])
        lo_, hi_ = (float(v) for v in ck["args"]["holdout_lon"].split(","))
        x_hold = (lons >= lo_) & (lons < hi_)
        Xa, _ = anomaly_transform(X, moy, t_hold, x_hold)
        codec = PixelMAE(n_chan=X.shape[-1], d_z=ck["d_z"])
        codec.load_state_dict(ck["model"])
        codec.eval()
        Xt = torch.from_numpy(np.nan_to_num(Xa, nan=0.0))
        OBS = torch.from_numpy(np.isfinite(Xa))
        Z, _ = embed_everything(codec, Xt, OBS, ctx_all, lats, lons,
                                ys[sec_sel], xs[sec_sel], codec.d_z)
        F = Z.mean(1)[ridx]
        r, lo95, hi95, n = kfold_r(F, rv_des, ryr)
        out[run] = {"r_kfold_deseas": round(r, 3),
                    "ci95": [round(lo95, 3), round(hi95, 3)], "n": n}
        print(f"{run:<10} d_z={ck['d_z']:<3} k-fold r_deseas {r:+.3f} "
              f"[{lo95:+.3f}, {hi95:+.3f}]  (n={n} months, year-blocked)")

    path = os.path.join(HERE, "runs", "probe_kfold.json")
    json.dump(out, open(path, "w"), indent=2)
    print("wrote", path)


if __name__ == "__main__":
    main()

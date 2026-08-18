#!/usr/bin/env python3
"""The 2009-10 case study: does a codec's out-of-fold prediction see the
most dramatic event in the RAPID record — the winter 2009-10 collapse
(~-30%, substantially wind-driven via the extreme negative-NAO Ekman
forcing)? Born as a break-time curiosity (2026-08-06) that turned out to
be a diagnostic: the 12-channel codec got the sign right from January but
missed the November onset entirely (predicted -1.1 Sv against an observed
-6.9 dip-window mean) — the wind was not in the tensor. This script is
the rematch harness.

Uses the year-blocked k-fold protocol (probe_kfold), so every printed
number is out-of-fold. Requires ml/cache/na_pixels.npz to match the
checkpoint's channel count.

Usage: python3 ml/dip_check.py --run wind14
"""
import argparse
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model import PixelMAE, LazyPixels, codec_from_ckpt
from trainprobe import anomaly_transform
from temporal import embed_everything, rapid_section

HERE = os.path.dirname(os.path.abspath(__file__))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--window", nargs=2, default=["2009-09", "2010-06"])
    ap.add_argument("--data", default=os.path.join(HERE, "cache", "na_pixels.npz"))
    a = ap.parse_args()
    d = np.load(a.data)
    months = [str(m) for m in d["months"]]
    moy = np.array([int(m[5:7]) - 1 for m in months])
    yr = np.array([int(m[:4]) for m in months])
    lats, lons = d["lats"], d["lons"]
    rapid = d["rapid"]
    ridx = rapid[:, 0].astype(int)
    rv = rapid[:, 1].copy()
    ryr = yr[ridx]
    rmoy = moy[ridx]
    rclim = np.array([rv[rmoy == m].mean() for m in range(12)])
    rvd = rv - rclim[rmoy]

    ck = torch.load(os.path.join(HERE, "runs", a.run, "pixelmae.pt"),
                    map_location="cpu", weights_only=False)
    X = d["X"]          # NpzFile decompresses fresh; .copy() doubled it
    if X.shape[-1] != len(ck["chan"]):
        sys.exit(f"tensor C={X.shape[-1]} != checkpoint C={len(ck['chan'])} — rebuild the tensor")
    t_hold = np.array([m[:4] in set(ck["args"]["holdout_years"].split(",")) for m in months])
    lo_, hi_ = (float(v) for v in ck["args"]["holdout_lon"].split(","))
    x_hold = (lons >= lo_) & (lons < hi_)
    Xa, _ = anomaly_transform(X, moy, t_hold, x_hold)
    codec = codec_from_ckpt(ck, X.shape[-1])
    codec.load_state_dict(ck["model"])
    codec.eval()
    # THE ONE LINE THAT WAS MISSING. embed_everything runs on whatever device
    # the MODEL is on (its own docstring says so), and this file never moved
    # it — so the dip check embedded the whole section on CPU while a 4090 sat
    # idle beside it. probe_kfold.py, probe_head.py and probe_sequence.py were
    # all given this line earlier; dip_check.py was missed, and it is the last
    # thing in the probe ladder, which is why runs kept ending with hours of
    # gpu_util=0 and cpu_util=60% that nobody could account for.
    codec.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    # ocean comes from the anomaly array, not a fresh d["X"] read: the
    # transform preserves NaN, and re-indexing the npz decompresses the whole
    # 2.4 GB tensor again while this one is still alive.
    ocean = np.isfinite(Xa[..., 0]).any(axis=0)
    ys, xs = np.where(ocean)
    sec_y, sec_sel = rapid_section(lats, lons, ys, xs)   # protocol v3 clip
    ctx = np.stack([np.sin(2 * np.pi * moy / 12), np.cos(2 * np.pi * moy / 12)], 1)
    # Derived PER BATCH, not materialised: `isfinite(Xa)` is a full-size bool
    # (16.6 GB at pentad, 83 GB at daily) allocated immediately after the
    # ~31-minute anomaly transform — the allocation that OOM-killed this script
    # in run #388. `torch.from_numpy(Xa)` itself was already zero-copy; the
    # mask was not. Same numpy functions, same elements, evaluated after the
    # index (LazyPixels in ml/model.py), so every printed number is unchanged.
    #
    # THE IN-PLACE `np.nan_to_num(Xa, copy=False)` IS DELETED ON PURPOSE.
    # LazyPixels(Xa) fills each indexed batch, so pre-filling Xa would leave
    # LazyPixels(Xa, obs=True) reading isfinite() over an array with no NaNs
    # left: the observation mask would silently be all-True and land would
    # enter the encoder as observed zeros. (`ocean` above is computed from the
    # NaNs and already runs before this point — it must stay there.)
    Xt = LazyPixels(Xa)
    OBS = LazyPixels(Xa, obs=True)
    Z, _ = embed_everything(codec, Xt, OBS, ctx, lats, lons,
                            ys[sec_sel], xs[sec_sel], ck["d_z"])
    F = Z.mean(1)[ridx]

    pred = np.full(len(rvd), np.nan)
    for y0 in np.unique(ryr):
        te = ryr == y0
        tr = ~te
        idx = np.where(tr)[0]
        fit, val = idx[: int(0.8 * len(idx))], idx[int(0.8 * len(idx)):]
        mu, sd = F[tr].mean(0), F[tr].std(0) + 1e-9
        Fz = (F - mu) / sd

        def solve(sel, lam):
            A = np.c_[Fz[sel], np.ones(len(sel))]
            reg = lam * np.eye(A.shape[1]); reg[-1, -1] = 0
            return np.linalg.solve(A.T @ A + reg, A.T @ rvd[sel])

        best, br = 1e-2, -np.inf
        for lam in (1e-2, 1e-1, 1, 10, 100, 1000):
            w = solve(fit, lam)
            p = np.c_[Fz[val], np.ones(len(val))] @ w
            r = np.corrcoef(p, rvd[val])[0, 1]
            if np.isfinite(r) and r > br:
                br, best = r, lam
        w = solve(idx, best)
        pred[te] = np.c_[Fz[te], np.ones(int(te.sum()))] @ w

    w0, w1 = a.window
    sel = [i for i, mi in enumerate(ridx) if w0 <= months[mi] <= w1]
    print(f"{a.run} (C={X.shape[-1]}, d_z={ck['d_z']}) — out-of-fold, deseasonalised (Sv)")
    print("month      observed  predicted")
    for i in sel:
        print(f"{months[ridx[i]]}   {rvd[i]:+7.2f}   {pred[i]:+7.2f}")
    dip = [i for i in sel if "2009-11" <= months[ridx[i]] <= "2010-03"]
    if dip:
        om = np.mean([rvd[i] for i in dip])
        pm = np.mean([pred[i] for i in dip])
        print(f"\ndip window (2009-11..2010-03): observed {om:+.2f}, predicted {pm:+.2f} "
              f"({100 * pm / om:.0f}% of the event captured)")
    ok = np.isfinite(pred)
    r_all = float(np.corrcoef(pred[ok], rvd[ok])[0, 1])
    sign = float(np.mean(np.sign(pred[ok]) == np.sign(rvd[ok])) * 100)
    print(f"full-record out-of-fold r: {r_all:+.3f} · "
          f"sign agreement {sign:.0f}%")
    # Written, not just printed: the dip share is a headline number in the
    # report, and a number that lives only in a CI log is a number that gets
    # re-derived by hand (and mistyped) every time the table is rebuilt.
    out = {"run": a.run, "window": list(a.window), "r_out_of_fold": round(r_all, 3),
           "sign_agreement_pct": round(sign, 1)}
    if dip:
        out["dip_observed_sv"] = round(float(om), 2)
        out["dip_predicted_sv"] = round(float(pm), 2)
        out["dip_captured_pct"] = round(float(100 * pm / om), 1)
    path = os.path.join(HERE, "runs", a.run, "dip_check.json")
    json.dump(out, open(path, "w"), indent=2)
    print("wrote", path)


if __name__ == "__main__":
    main()

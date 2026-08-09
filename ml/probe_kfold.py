#!/usr/bin/env python3
"""Year-blocked k-fold transport probe — RAPID plus every truth series.

Since 2026-08-07 this is MULTI-TARGET (lever 1): each series in TARGETS
(RAPID 26.5N, Florida Current cable 27N, MOVE 16N, OSNAP subpolar, SAMBA
34.5S — the latter fetched by fetch_truth.py) is probed from its own zonal
section's embeddings, deseasonalised, year-blocked, block-bootstrapped.
Sections outside the tensor window are skipped with a note.


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
import gc
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model import PixelMAE, codec_from_ckpt
from trainprobe import anomaly_transform
from temporal import embed_everything, section_of

HERE = os.path.dirname(os.path.abspath(__file__))

# Every basin transport array gets its own zonal section (lat, lon range) —
# the embeddings pooled along it are what the ridge reads. RAPID/FC share
# the 26.5N Atlantic section (the cable is at 27N inside it; the two arrays
# measure sibling quantities across the same boundary). Sections outside
# the tensor window (e.g. SAMBA on the NA pilot) are skipped with a note.
TARGETS = {
    "rapid": {"lat": 26.5, "lon": (-80.0, -13.0), "key": "rapid"},
    "fc":    {"lat": 26.5, "lon": (-80.0, -13.0), "key": "truth_fc"},
    "move":  {"lat": 16.5, "lon": (-61.0, -49.0), "key": "truth_move"},
    "osnap": {"lat": 58.0, "lon": (-45.0, -5.0),  "key": "truth_osnap"},
    "samba": {"lat": -34.5, "lon": (-52.0, 18.0), "key": "truth_samba"},
}



def _mlp_fold(Ftr, ytr, Fte, seed, hidden=32, wd=1e-3, steps=3000):
    """One fold's nonlinear probe: 64->hidden->1 MLP, weight decay, early
    stopping on a train-tail — the SAME inner split discipline as the ridge.
    Deliberately tiny: with ~220 train months the question is whether ANY
    nonlinearity helps, not how much capacity fits."""
    import torch
    import torch.nn as nn
    torch.manual_seed(seed)
    g = torch.Generator().manual_seed(seed)
    n = len(Ftr)
    fit = slice(0, int(0.8 * n))
    val = slice(int(0.8 * n), n)
    net = nn.Sequential(nn.Linear(Ftr.shape[1], hidden), nn.GELU(),
                        nn.Linear(hidden, 1))
    opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=wd)
    Xf = torch.as_tensor(Ftr[fit], dtype=torch.float32)
    yf = torch.as_tensor(ytr[fit], dtype=torch.float32)[:, None]
    Xv = torch.as_tensor(Ftr[val], dtype=torch.float32)
    yv = torch.as_tensor(ytr[val], dtype=torch.float32)[:, None]
    best, best_state, patience = np.inf, None, 0
    for step in range(steps):
        k = torch.randint(0, len(Xf), (min(64, len(Xf)),), generator=g)
        loss = (net(Xf[k]) - yf[k]).pow(2).mean()
        opt.zero_grad(); loss.backward(); opt.step()
        if step % 50 == 0:
            with torch.no_grad():
                v = (net(Xv) - yv).pow(2).mean().item()
            if v < best - 1e-6:
                best, patience = v, 0
                best_state = {k2: v2.clone() for k2, v2 in net.state_dict().items()}
            else:
                patience += 1
                if patience >= 12:            # 600 steps with no val improvement
                    break
    if best_state:
        net.load_state_dict(best_state)
    with torch.no_grad():
        return net(torch.as_tensor(Fte, dtype=torch.float32)).numpy().ravel()


def kfold_r(F, y, years, lams=(1e-2, 1e-1, 1, 10, 100, 1000), boot=2000, seed=0,
            probe="ridge"):
    """Grouped-by-year k-fold; probe='ridge' (default, the defensible
    instrument) or 'mlp' (tiny nonlinear read-out, 3 seeds averaged per
    fold). The MLP exists to test ONE hypothesis: that skill present in the
    embedding is not LINEARLY accessible — e.g. the 25-channel codec's
    RAPID drop. If mlp ~= ridge, the information is genuinely absent or
    entangled beyond small-sample reach; if mlp >> ridge, the probe was the
    limit, not the representation. Same year-blocked folds, same inner-tail
    discipline, so the two numbers differ only in the read-out family."""
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
        if probe == "mlp":
            pred[te] = np.mean([_mlp_fold(Fz[idx], y[idx], Fz[te], sd)
                                for sd in (0, 1, 2)], axis=0)
        else:
            w = solve(idx, best)
            pred[te] = np.c_[Fz[te], np.ones(int(te.sum()))] @ w
    ok = np.isfinite(pred)
    r = float(np.corrcoef(pred[ok], y[ok])[0, 1])
    rmse = float(np.sqrt(np.mean((pred[ok] - y[ok]) ** 2)))
    sigma = float(np.std(y[ok]))
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
    return r, float(lo), float(hi), int(ok.sum()), rmse, sigma, pred


def lowpass_r(tidx, pred, truth, k=18, min_valid=12):
    """Centered k-month running mean of both series on the month axis, then
    Pearson r — the filtering the classical AMOC-reconstruction literature
    reports (Frajka-Williams 2015, Sanchez-Franks 2021 use 18 months).
    Windows with < min_valid finite months are dropped."""
    lo, hi = int(tidx.min()), int(tidx.max())
    n = hi - lo + 1
    p = np.full(n, np.nan)
    t = np.full(n, np.nan)
    p[tidx - lo] = pred
    t[tidx - lo] = truth
    half = k // 2
    ps, ts = [], []
    for i in range(half, n - half):
        wp, wt = p[i - half: i + half], t[i - half: i + half]
        okw = np.isfinite(wp) & np.isfinite(wt)
        if okw.sum() >= min_valid:
            ps.append(wp[okw].mean())
            ts.append(wt[okw].mean())
    if len(ps) < 24:
        return None
    return float(np.corrcoef(ps, ts)[0, 1])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", default=["actions"])
    # The tensor was hardcoded to cache/na_pixels.npz, so probing a codec
    # trained on a different channel set meant COPYING an 866 MB file over
    # the active one and remembering to put it back — a dance that has
    # already produced one silent mismatch. Name the tensor instead; the
    # channel count is checked against the checkpoint below.
    ap.add_argument("--data", default=os.path.join(HERE, "cache", "na_pixels.npz"))
    ap.add_argument("--probe", choices=["ridge", "mlp"], default="ridge",
                    help="read-out family; see kfold_r docstring")
    a = ap.parse_args()
    d = np.load(a.data)
    months = [str(m) for m in d["months"]]
    moy = np.array([int(m[5:7]) - 1 for m in months])
    yr = np.array([int(m[:4]) for m in months])
    lats, lons = d["lats"], d["lons"]
    month_of_ym = {int(m[:4]) * 100 + int(m[5:7]): i for i, m in enumerate(months)}

    def target_series(spec):
        """-> (tidx month-indices, deseasonalised values) or None."""
        if spec["key"] == "rapid":
            arr = d["rapid"]
            tidx = arr[:, 0].astype(int)
            vals = arr[:, 1].copy()
        else:
            if spec["key"] not in d:
                return None
            arr = d[spec["key"]]
            keep = [(month_of_ym[int(ym)], v) for ym, v in arr if int(ym) in month_of_ym]
            if len(keep) < 48:
                return None
            tidx = np.array([k[0] for k in keep], dtype=int)
            vals = np.array([k[1] for k in keep], dtype=float)
        tmoy = moy[tidx]
        # deseasonalise with the OVERALL monthly climatology (every month is
        # test in some fold; per-fold climatologies differ by <0.1 Sv and
        # would leak nothing either way — chosen for simplicity, stated here).
        clim = np.array([vals[tmoy == m].mean() for m in range(12)])
        return tidx, vals - clim[tmoy]

    ctx_all = np.stack([np.sin(2 * np.pi * moy / 12), np.cos(2 * np.pi * moy / 12)], 1)
    # HOIST the raw tensor's two uses out of the loops and free it. Indexing
    # d["X"] decompresses the WHOLE array every time (2.4 GB at C=24), and it
    # was being indexed once for `ocean` plus once per target per run for the
    # wind baseline — five re-materialisations per run, next to the anomaly
    # copy and the torch tensors. That is what OOM-killed the probes.
    Xraw = d["X"]
    ocean = np.isfinite(Xraw[..., 0]).any(axis=0)
    ys, xs = np.where(ocean)
    chan_names = [str(c) for c in d["chan"]]
    wsel = [i for i, c in enumerate(chan_names) if c in ("tau_x", "tau_y")]
    wind_sec = {}
    for tname, spec in TARGETS.items():
        sec_y_, sec_sel_ = section_of(lats, lons, ys, xs, spec["lat"], *spec["lon"])
        if wsel and len(sec_sel_) >= 5 and abs(float(lats[sec_y_]) - spec["lat"]) <= 1.0:
            wind_sec[tname] = np.nan_to_num(
                np.nanmean(Xraw[:, ys[sec_sel_], xs[sec_sel_]][..., wsel], axis=1),
                nan=0.0)
    del Xraw
    gc.collect()

    out = {}
    for run in a.runs:
        ck = torch.load(os.path.join(HERE, "runs", run, "pixelmae.pt"),
                        map_location="cpu", weights_only=False)
        if len(ck["chan"]) != d["X"].shape[-1]:
            sys.exit(f"{run}: codec has {len(ck['chan'])} channels but "
                     f"{os.path.basename(a.data)} has {d['X'].shape[-1]} — "
                     f"pass --data with the matching tensor.")
        X = d["X"]          # NpzFile decompresses fresh; .copy() doubled it
        t_hold = np.array([m[:4] in set(ck["args"]["holdout_years"].split(","))
                           for m in months])
        lo_, hi_ = (float(v) for v in ck["args"]["holdout_lon"].split(","))
        x_hold = (lons >= lo_) & (lons < hi_)
        Xa, _ = anomaly_transform(X, moy, t_hold, x_hold)
        codec = codec_from_ckpt(ck, Xa.shape[-1])
        codec.load_state_dict(ck["model"])
        codec.eval()
        # See the note in probe_head.py: embed_everything follows the model's
        # device, so moving the codec is the entire GPU fix here.
        codec.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        OBS = torch.from_numpy(np.isfinite(Xa))      # mask BEFORE filling
        np.nan_to_num(Xa, nan=0.0, copy=False)       # in place: no second copy
        Xt = torch.from_numpy(Xa)
        del X               # Xa IS Xt's buffer now — never free it here
        gc.collect()
        out[run] = {}
        for tname, spec in TARGETS.items():
            ser = target_series(spec)
            if ser is None:
                continue
            sec_y, sec_sel = section_of(lats, lons, ys, xs,
                                        spec["lat"], *spec["lon"])
            # argmin() clamps to the window edge — a SAMBA request on the NA
            # window would silently probe the equator without this check.
            if abs(float(lats[sec_y]) - spec["lat"]) > 1.0 or len(sec_sel) < 5:
                print(f"{run:<10} {tname}: section outside window, skipped")
                continue
            tidx, v_des = ser
            Z, _ = embed_everything(codec, Xt, OBS, ctx_all, lats, lons,
                                    ys[sec_sel], xs[sec_sel], codec.d_z)
            F = Z.mean(1)[tidx]
            r, lo95, hi95, n, rmse, sigma, pred = kfold_r(F, v_des, yr[tidx], probe=a.probe)
            okp = np.isfinite(pred)
            r_lp = lowpass_r(tidx[okp], pred[okp], v_des[okp])
            # the baseline this literature demands: wind stress alone (Ekman
            # carries much of the subtropical monthly signal — Solodoch 2023).
            # Same protocol, same section, raw tau channels instead of the
            # embedding; if the embedding does not beat this, it has learned
            # only the wind.
            wind_block = None
            if tname in wind_sec:
                W = wind_sec[tname][tidx]
                rw, low, hiw, _, rmw, _, _ = kfold_r(W, v_des, yr[tidx])
                wind_block = {"r": round(rw, 3), "ci95": [round(low, 3), round(hiw, 3)],
                              "rmse_sv": round(rmw, 2)}
            out[run][tname] = {"r_kfold_deseas": round(r, 3),
                               "ci95": [round(lo95, 3), round(hi95, 3)], "n": n,
                               "rmse_sv": round(rmse, 2),
                               "sigma_sv": round(sigma, 2),
                               "r_lowpass18": None if r_lp is None else round(r_lp, 3),
                               "wind_only_baseline": wind_block}
            lp_s = "n/a" if r_lp is None else f"{r_lp:+.3f}"
            w_s = "n/a" if wind_block is None else f"{wind_block['r']:+.3f}"
            print(f"{run:<10} d_z={ck['d_z']:<3} {tname:<6} k-fold r {r:+.3f} "
                  f"[{lo95:+.3f}, {hi95:+.3f}]  (n={n}) · RMSE {rmse:.2f} Sv "
                  f"(sigma {sigma:.2f}) · 18mo-lowpass r {lp_s} · wind-only {w_s}")

    # MERGE, never overwrite. This file is keyed by run and shared by every
    # invocation, so `json.dump(out)` silently deleted every previously
    # measured run: patch24_40k's k-fold survived about ten minutes before
    # the next run erased it, and global15sst's published +0.582 went the
    # same way. A per-run copy is written too, so the shared file is a
    # convenience rather than the only home for a number.
    suffix = "" if a.probe == "ridge" else f"_{a.probe}"
    path = os.path.join(HERE, "runs", f"probe_kfold{suffix}.json")
    merged = {}
    if os.path.exists(path):
        try:
            merged = json.load(open(path))
        except json.JSONDecodeError:
            pass
    merged.update(out)
    json.dump(merged, open(path, "w"), indent=2)
    for run, block in out.items():
        rp = os.path.join(HERE, "runs", run, f"probe_kfold{suffix}.json")
        if os.path.isdir(os.path.dirname(rp)):
            json.dump({run: block}, open(rp, "w"), indent=2)
    print("wrote", path, f"({len(merged)} runs)")


if __name__ == "__main__":
    main()

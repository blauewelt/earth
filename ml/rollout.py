#!/usr/bin/env python3
"""Autoregressive rollout — how far into the future does the model see?

Every metric before this one scored a single step (t+1). This harness
answers the user's question directly: predict a month, feed the
prediction back into the context, predict the next — month after month —
and measure where skill dies.

Protocol (stated precisely, because horizon claims rot fastest):
  · For each held-out year Y and each start month s from Dec(Y-1) to
    Nov(Y), initialise the context with TRUE embeddings up to s (an
    observed initial condition, not leakage), then roll autoregressively.
    Score every (s, h) whose target month s+h lies inside Y — so h=1 has
    12 starts per year and h=12 has one, giving 3 clean year-long
    trajectories plus staggered partials from 3 holdout years.
  · Channel-space skill at horizon h, on decoded predictions, observed
    cells only, in anomaly space, against TWO baselines:
      - persistence: the anomaly at s stays frozen (hard at short range)
      - climatology: the anomaly is zero (the no-skill floor)
    "How far can we predict" = the horizon where model MSE reaches the
    climatology floor. skill = 1 - MSE_model/MSE_baseline (positive =
    better than the baseline).
  · AMOC by horizon: a ridge probe fit on TRUE train-month section
    embeddings, applied to ROLLED section embeddings — deseasonalised r
    in horizon bands 1-3 / 4-6 / 7-12 (single-horizon n is tiny: 36/21/3
    points per horizon at the tail — bands are the honest resolution).

Usage:  python3 ml/rollout.py --run wind14 [--horizon 12] [--pixels 600]
Writes runs/<run>/rollout.json and prints the curve.
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
from temporal import TemporalTransformer, embed_everything, rapid_section

HERE = os.path.dirname(os.path.abspath(__file__))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--horizon", type=int, default=12)
    ap.add_argument("--pixels", type=int, default=600)
    ap.add_argument("--crop-window", choices=["na"], default=None,
                    help="crop the tensor to the pilot window first — for "
                         "evaluating an NA-trained model on a global tensor "
                         "in its own training distribution (anomaly space is "
                         "invariant to the build-time z-score, so the crop "
                         "reproduces the pilot tensor exactly)")
    a = ap.parse_args()
    d = dict(np.load(os.path.join(HERE, "cache", "na_pixels.npz")))
    if a.crop_window == "na":
        la, lo_ = d["lats"], d["lons"]
        ysel = (la >= 0.0) & (la <= 70.0)
        xsel = (lo_ >= -100.0) & (lo_ <= 20.0)
        d["X"] = d["X"][:, ysel][:, :, xsel]
        d["lats"], d["lons"] = la[ysel], lo_[xsel]
        print(f"cropped to NA pilot window: {d['X'].shape}")
    months = [str(m) for m in d["months"]]
    T = len(months)
    moy = np.array([int(m[5:7]) - 1 for m in months])
    lats, lons = d["lats"], d["lons"]

    ck = torch.load(os.path.join(HERE, "runs", a.run, "pixelmae.pt"),
                    map_location="cpu", weights_only=False)
    tk = torch.load(os.path.join(HERE, "runs", a.run, "temporal.pt"),
                    map_location="cpu", weights_only=False)
    K = tk["args"]["K"]
    X = d["X"].copy()
    if X.shape[-1] != len(ck["chan"]):
        sys.exit(f"tensor C={X.shape[-1]} != checkpoint C={len(ck['chan'])}")
    hold_years = sorted(ck["args"]["holdout_years"].split(","))
    t_hold = np.array([m[:4] in set(hold_years) for m in months])
    lo, hi = (float(v) for v in ck["args"]["holdout_lon"].split(","))
    x_hold = (lons >= lo) & (lons < hi)
    Xa, dynamic = anomaly_transform(X, moy, t_hold, x_hold)
    codec = PixelMAE(n_chan=X.shape[-1], d_z=ck["d_z"])
    codec.load_state_dict(ck["model"])
    codec.eval()
    model = TemporalTransformer(d_z=ck["d_z"], d_model=tk["args"]["d_model"],
                                n_heads=4, n_layers=tk["args"]["layers"], k_max=K)
    model.load_state_dict(tk["model"])
    model.eval()

    ocean = np.isfinite(d["X"][..., 0]).any(axis=0)
    ys, xs = np.where(ocean)
    _, sec_sel = rapid_section(lats, lons, ys, xs)
    rng = np.random.default_rng(0)
    keep = rng.choice(len(ys), min(a.pixels, len(ys)), replace=False)
    keep = np.union1d(keep, sec_sel)
    kys, kxs = ys[keep], xs[keep]
    sec_pos = np.where(np.isin(keep, sec_sel))[0]
    P = len(kys)

    ctx_all = np.stack([np.sin(2 * np.pi * moy / 12),
                        np.cos(2 * np.pi * moy / 12)], 1).astype(np.float32)
    Xt = torch.from_numpy(np.nan_to_num(Xa, nan=0.0))
    OBS = torch.from_numpy(np.isfinite(Xa))
    Z, coords = embed_everything(codec, Xt, OBS, ctx_all, lats, lons,
                                 kys, kxs, ck["d_z"])
    Zt = torch.from_numpy(Z)
    Mt = torch.as_tensor(ctx_all)

    # static identity — EXACTLY as stage-2 training builds it
    with torch.no_grad():
        stat_obs = OBS[0].clone()
        for c in dynamic:
            stat_obs[..., c] = False
        ctx0 = np.concatenate([np.zeros((P, 2), np.float32), coords], 1)
        Zstat = codec.encode(Xt[0, kys, kxs], stat_obs[kys, kxs],
                             torch.zeros(P, X.shape[-1], dtype=torch.bool),
                             torch.as_tensor(ctx0)).numpy()
    static_ctx = torch.as_tensor(np.concatenate([Zstat, coords], 1))

    # ---- RAPID probe fit on TRUE train-month section embeddings -----------
    rapid = d["rapid"]
    ridx = rapid[:, 0].astype(int)
    rv = rapid[:, 1].copy()
    rmoy = moy[ridx]
    tr_all = ~t_hold[ridx]
    rclim = np.array([rv[tr_all & (rmoy == m)].mean() for m in range(12)])
    rv_des = rv - rclim[rmoy]
    Fsec_true = Z[:, sec_pos].mean(1)
    # same protocol as ridge_r (standardise on train, lambda on a train
    # tail), but we need the WEIGHTS to apply to rolled embeddings later.
    mu_p = Fsec_true[ridx][tr_all].mean(0)
    sd_p = Fsec_true[ridx][tr_all].std(0) + 1e-9
    Fz = (Fsec_true[ridx] - mu_p) / sd_p
    tr_idx = np.where(tr_all)[0]
    fit_i, val_i = tr_idx[: int(0.8 * len(tr_idx))], tr_idx[int(0.8 * len(tr_idx)):]

    def _solve(idx, lam):
        A = np.c_[Fz[idx], np.ones(len(idx))]
        reg = lam * np.eye(A.shape[1]); reg[-1, -1] = 0
        return np.linalg.solve(A.T @ A + reg, A.T @ rv_des[idx])

    best_lam, best_r = 1.0, -np.inf
    for lam in (1e-2, 1e-1, 1, 10, 100, 1000):
        w = _solve(fit_i, lam)
        p = np.c_[Fz[val_i], np.ones(len(val_i))] @ w
        r = np.corrcoef(p, rv_des[val_i])[0, 1]
        if np.isfinite(r) and r > best_r:
            best_r, best_lam = r, lam
    w_probe = _solve(tr_idx, best_lam)
    ym_to_r = {int(months[mi][:4]) * 100 + int(months[mi][5:7]): i
               for i, mi in enumerate(ridx)}

    # ---- the rollouts ------------------------------------------------------
    month_index = {m: i for i, m in enumerate(months)}
    qc = torch.arange(X.shape[-1])[None, :].expand(P, -1)
    off0 = torch.zeros(P, X.shape[-1], 3, dtype=torch.long)
    H = a.horizon
    sums = {k: np.zeros(H + 1) for k in ("mse_m", "mse_p", "mse_c", "n")}
    probe_pts = {h: [] for h in range(1, H + 1)}

    with torch.no_grad():
        for Y in hold_years:
            for s_off in range(12):
                start_m = (f"{int(Y) - 1}-12" if s_off == 0
                           else f"{Y}-{s_off:02d}")
                if start_m not in month_index:
                    continue
                s = month_index[start_m]
                if s - K + 1 < 0 or s + 1 >= T:
                    continue
                seq = Zt[s - K + 1: s + 1].transpose(0, 1).clone()   # [P,K,dz]
                for h in range(1, H + 1):
                    t_tgt = s + h
                    if t_tgt >= T or months[t_tgt][:4] != Y:
                        break
                    mseq = torch.stack(
                        [Mt[s - K + h + j] for j in range(K)], 0
                    )[None].expand(P, -1, -1)
                    pred, _ = model(seq, mseq, static_ctx)
                    zhat = pred[:, -1]                               # [P,dz]
                    seq = torch.cat([seq[:, 1:], zhat[:, None]], 1)
                    # channel-space scoring at the target month
                    xhat = codec.query(zhat, qc, off0)
                    v_true = Xt[t_tgt, kys, kxs]
                    o = OBS[t_tgt, kys, kxs]
                    v_pers = Xt[s, kys, kxs]
                    op = o & OBS[s, kys, kxs]
                    if op.sum() > 0:
                        sums["mse_m"][h] += float(((xhat - v_true).pow(2) * op).sum())
                        sums["mse_p"][h] += float(((v_pers - v_true).pow(2) * op).sum())
                        sums["mse_c"][h] += float((v_true.pow(2) * op).sum())
                        sums["n"][h] += float(op.sum())
                    # AMOC probe on the rolled section embedding
                    ym = int(months[t_tgt][:4]) * 100 + int(months[t_tgt][5:7])
                    if ym in ym_to_r:
                        f = (zhat[sec_pos].mean(0).numpy() - mu_p) / sd_p
                        pr = float(np.dot(np.r_[f, 1.0], w_probe))
                        probe_pts[h].append((pr, rv_des[ym_to_r[ym]]))

    out = {"run": a.run, "K": K, "horizon": H,
           "protocol": "staggered starts into holdout years, true-context init",
           "chan_skill": []}
    print(f"{a.run}: autoregressive rollout, {len(hold_years)} holdout years")
    print("  h   n_cells   skill vs clim   skill vs persistence")
    for h in range(1, H + 1):
        if sums["n"][h] == 0:
            continue
        mm = sums["mse_m"][h] / sums["n"][h]
        mp = sums["mse_p"][h] / sums["n"][h]
        mc = sums["mse_c"][h] / sums["n"][h]
        s_c, s_p = 1 - mm / mc, 1 - mm / mp
        out["chan_skill"].append({"h": h, "n": int(sums["n"][h]),
                                  "vs_clim": round(s_c, 3),
                                  "vs_persistence": round(s_p, 3)})
        print(f"  {h:2d}  {int(sums['n'][h]):8d}   {s_c:+13.3f}   {s_p:+13.3f}")
    # one scalar for model selection: mean skill-vs-climatology over the
    # horizon sweep ("horizon AUC") — rewards models that stay useful deep
    # into the rollout, not just at t+1.
    if out["chan_skill"]:
        out["horizon_auc"] = round(
            float(np.mean([c["vs_clim"] for c in out["chan_skill"]])), 3)
        print(f"  horizon AUC (mean vs-clim, h=1..{H}): {out['horizon_auc']:+.3f}")
    out["amoc_bands"] = {}
    for name, hs in (("h1-3", (1, 2, 3)), ("h4-6", (4, 5, 6)),
                     ("h7-12", tuple(range(7, H + 1)))):
        pts = [p for h in hs for p in probe_pts.get(h, [])]
        if len(pts) >= 8:
            pr, tv = np.array(pts).T
            r = float(np.corrcoef(pr, tv)[0, 1])
            out["amoc_bands"][name] = {"r": round(r, 3), "n": len(pts)}
            print(f"  AMOC {name}: r {r:+.3f} (n={len(pts)})")
    path = os.path.join(HERE, "runs", a.run, "rollout.json")
    json.dump(out, open(path, "w"), indent=1)
    print("wrote", path)


if __name__ == "__main__":
    main()

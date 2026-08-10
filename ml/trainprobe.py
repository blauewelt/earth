#!/usr/bin/env python3
"""Predictive metrics on FROZEN embeddings — computable during training.

The question a reconstruction loss cannot answer is the one the project
exists for: do the embeddings, as they are RIGHT NOW, support prediction?
This module answers it cheaply enough to run every N training steps:

  · linear_probe — the 26.5°N section's single-month embeddings, ridge to
    the deseasonalised RAPID transport (protocol v2: train-years target
    climatology, lambda on a train tail, scored on held-out years only).
  · mini_temporal — freeze the codec, train a SMALL stage-2 transformer
    (subsampled pixels, short schedule, fixed seed) on the embeddings, and
    score it on held-out months: z-space t+1 vs persistence, decoded
    channel-space t+1 vs persistence, and the RAPID probe from the
    section-pooled hidden state. This is the user-requested metric: "freeze
    the existing embedding, train a transformer to use them to predict,
    compute the metrics with this."

Deterministic (seeded) so curves across checkpoints are comparable. The
mini transformer is deliberately small and short — it is a MEASUREMENT of
the embedding, not a model we keep; the full stage 2 lives in temporal.py.

CLI (backfill an existing run, writes runs/<run>/trainprobe.json):
    python3 ml/trainprobe.py --run pilot4_anom
"""
import argparse
import json
import os
import sys
import time
import warnings

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model import PixelMAE, codec_from_ckpt
from probe_sequence import ridge_r
from temporal import TemporalTransformer, embed_everything, rapid_section

HERE = os.path.dirname(os.path.abspath(__file__))


def anomaly_transform(X, moy, t_hold, x_hold):
    """The one anomaly transform (train.py --anomaly), in one place: dynamic
    channels become departures from their own train-years monthly
    climatology, then z-scored on train data. Returns (X, dynamic)."""
    T, H, W, C = X.shape
    dynamic = [c for c in range(C)
               if np.nanstd(np.nanmean(X[..., c], axis=(1, 2))) > 1e-6]
    clim = np.full((12, H, W, C), np.nan, dtype=np.float32)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for m in range(12):
            clim[m] = np.nanmean(X[(moy == m) & ~t_hold], axis=0)
    for c in dynamic:
        # clim[moy, :, :, c] is [T,H,W]; the equivalent-looking
        # clim[moy][..., c] materialises the whole [T,H,W,C] fancy index and
        # throws all but one channel away — 2.4 GB per dynamic channel at
        # C=24, which OOM-killed every probe on a 7 GB box (2026-08-08).
        X[..., c] = X[..., c] - clim[moy, :, :, c]
        v = X[..., c][np.isfinite(X[..., c]) & ~t_hold[:, None, None]
                      & ~x_hold[None, None, :]]
        X[..., c] = (X[..., c] - v.mean()) / (v.std() + 1e-6)
    return X, dynamic


def probe_now(codec, X, OBS, d, moy, t_hold, x_hold, dynamic,
              n_pixels=600, K=12, tsteps=400, tbatch=128, seed=0, obs_in=None,
              mask_chan=None, light=False):
    """All metrics for the codec AS IT IS NOW. X must already be in the
    space the codec was trained in (anomaly). Returns a flat dict.

    obs_in: optional observation mask for the ENCODER ONLY (ablations —
    channels marked unobserved enter as the codec's native missing tokens).
    Scoring targets always use the true OBS, so 'predict the field from
    less input' is measured against the same reality.

    light=True computes ONLY the linear section probe and returns. That is
    the cheap part by an order of magnitude: it embeds ~67 section pixels,
    where the full probe embeds 600 more and then trains a mini temporal
    transformer for 400 steps (measured on the 10M codec: 300 s full,
    ~30 s light). Its purpose is cadence — a headline number every couple
    of thousand steps instead of every ten thousand, so a run that is not
    going to clear the wind baseline can be seen failing early rather than
    at the end. Both modes emit `linear_r_deseas`, so downstream readers
    (metrics.jsonl, the status page) need no special case."""
    was_training = codec.training
    codec.eval()
    t0 = time.time()
    rng = np.random.default_rng(seed)
    if obs_in is None:
        obs_in = OBS
    lats, lons = d["lats"], d["lons"]
    T, H, W, C = X.shape
    ctx_all = np.stack([np.sin(2 * np.pi * moy / 12),
                        np.cos(2 * np.pi * moy / 12)], 1)
    ocean = np.isfinite(d["X"][..., 0]).any(axis=0)
    ys, xs = np.where(ocean)
    sec_y, sec_sel = rapid_section(lats, lons, ys, xs)   # protocol v3 clip

    # ---- RAPID target, deseasonalised once -------------------------------
    rapid = d["rapid"]
    ridx = rapid[:, 0].astype(int)
    rv_raw = rapid[:, 1].copy()
    rmoy = moy[ridx]
    tr_all = ~t_hold[ridx]
    rclim = np.array([rv_raw[tr_all & (rmoy == m)].mean() for m in range(12)])
    rv_des = rv_raw - rclim[rmoy]
    te_all = t_hold[ridx]

    out = {}

    # ---- 1 · linear section probe (K=1) ----------------------------------
    Zsec, _ = embed_everything(codec, X, obs_in, ctx_all, lats, lons,
                               ys[sec_sel], xs[sec_sel], codec.d_z,
                               mask_chan=mask_chan)
    Fsec = Zsec.mean(1)                                   # [T, d_z]
    out["linear_r_deseas"], _ = ridge_r(Fsec[ridx], rv_des, tr_all, te_all)
    out["linear_r_raw"], _ = ridge_r(Fsec[ridx], rv_raw, tr_all, te_all)

    if light:
        out["light"] = True
        out["probe_seconds"] = round(time.time() - t0, 1)
        if was_training:
            codec.train()
        return out

    # ---- 2 · mini temporal transformer on frozen embeddings ---------------
    keep = rng.choice(len(ys), min(n_pixels, len(ys)), replace=False)
    keep = np.union1d(keep, sec_sel)
    kys, kxs = ys[keep], xs[keep]
    Z, coords = embed_everything(codec, X, obs_in, ctx_all, lats, lons,
                                 kys, kxs, codec.d_z, mask_chan=mask_chan)
    P = len(kys)
    Zt = torch.from_numpy(Z)
    Mt = torch.as_tensor(ctx_all, dtype=torch.float32)
    static_ctx = torch.as_tensor(
        np.concatenate([np.zeros((P, codec.d_z), np.float32), coords], 1))

    torch.manual_seed(seed)
    mini = TemporalTransformer(d_z=codec.d_z, d_model=64, n_heads=4,
                               n_layers=2, k_max=K)
    opt = torch.optim.AdamW(mini.parameters(), lr=2e-3, weight_decay=1e-4)

    ok_t = np.array([t + 1 < T and not t_hold[t + 1] and t + 1 >= K
                     for t in range(T)])
    ok_p = ~x_hold[kxs]
    pt, pp = np.where(ok_t[:, None] & ok_p[None, :])
    pt = torch.as_tensor(pt, dtype=torch.long)
    pp = torch.as_tensor(pp, dtype=torch.long)
    g = torch.Generator().manual_seed(seed)
    for s in range(tsteps):
        k = torch.randint(0, len(pt), (tbatch,), generator=g)
        t, p = pt[k], pp[k]
        base = t - K + 1
        zseq = torch.stack([Zt[base + j, p] for j in range(K)], 1)
        mseq = torch.stack([Mt[base + j] for j in range(K)], 1)
        ztgt = torch.stack([Zt[base + j + 1, p] for j in range(K)], 1)
        pred, _ = mini(zseq, mseq, static_ctx[p])
        loss = (pred - ztgt).pow(2).mean()
        opt.zero_grad(); loss.backward(); opt.step()
    mini.eval()

    with torch.no_grad():
        # held-out months, z and channel space
        ev_t = np.array([t + 1 < T and t_hold[t + 1] and t + 1 >= K
                         for t in range(T)])
        et, ep = np.where(ev_t[:, None] & np.ones(P, bool)[None, :])
        sel = np.random.default_rng(seed).choice(
            len(et), min(8000, len(et)), replace=False)
        et = torch.as_tensor(et[sel], dtype=torch.long)
        ep = torch.as_tensor(ep[sel], dtype=torch.long)
        base = et - K + 1
        zseq = torch.stack([Zt[base + j, ep] for j in range(K)], 1)
        mseq = torch.stack([Mt[base + j] for j in range(K)], 1)
        pred, _ = mini(zseq, mseq, static_ctx[ep])
        zhat, ztrue, zlast = pred[:, -1], Zt[et + 1, ep], Zt[et, ep]
        out["z_mse_model"] = float((zhat - ztrue).pow(2).mean())
        out["z_mse_persistence"] = float((zlast - ztrue).pow(2).mean())

        Xt, OBSt = X, OBS          # already tensors (zeros at missing + mask)
        qc = torch.arange(C)[None, :].expand(len(et), -1)
        # codec.query runs the decoder, which lives on the CODEC's device —
        # cuda during an in-training probe since the GPU-probe change, while
        # zhat and the index tensors here are all CPU (Zt comes back from
        # embed_everything as numpy, the mini transformer is CPU). Bridge the
        # one call: inputs to the codec's device, result back to CPU where the
        # raw-field tensors Xt/OBSt live. This seam is what killed #56-#59 at
        # their first full probe (step 10k): a cuda channel-embedding weight
        # indexed by a cpu chan_idx is the index_select device mismatch.
        qdev = next(codec.parameters()).device
        xhat = codec.query(
            zhat.to(qdev), qc.to(qdev),
            torch.zeros(len(et), C, 3, dtype=torch.long, device=qdev)).cpu()
        kys_t = torch.as_tensor(kys, dtype=torch.long)
        kxs_t = torch.as_tensor(kxs, dtype=torch.long)
        v1 = Xt[et + 1, kys_t[ep], kxs_t[ep]]
        o1 = OBSt[et + 1, kys_t[ep], kxs_t[ep]]
        v0 = Xt[et, kys_t[ep], kxs_t[ep]]
        o0 = OBSt[et, kys_t[ep], kxs_t[ep]]
        dyn = torch.zeros(C, dtype=torch.bool); dyn[dynamic] = True
        both = o0 & o1 & dyn[None, :]
        out["chan_mse_model"] = float(((xhat - v1).pow(2) * both).sum() / both.sum())
        out["chan_mse_persistence"] = float(((v0 - v1).pow(2) * both).sum() / both.sum())

        # RAPID probe from the mini transformer's section hidden state
        sec_in_keep = torch.as_tensor(
            np.where(np.isin(keep, sec_sel))[0], dtype=torch.long)
        F = np.zeros((T, 64), dtype=np.float32)
        for t in range(K - 1, T):
            base = t - K + 1
            zseq = torch.stack([Zt[base + j, sec_in_keep] for j in range(K)], 1)
            mseq = torch.stack([Mt[base + j].expand(len(sec_in_keep), -1)
                                for j in range(K)], 1)
            _, hid = mini(zseq, mseq, static_ctx[sec_in_keep])
            F[t] = hid[:, -1].mean(0).numpy()
        ok = ridx >= K - 1
        ri = ridx[ok]
        tr, te = ~t_hold[ri], t_hold[ri]
        out["temporal_r_deseas"], _ = ridge_r(F[ri], rv_des[ok], tr, te)
        out["temporal_r_raw"], _ = ridge_r(F[ri], rv_raw[ok], tr, te)

    out["chan_vs_persistence_pct"] = round(
        100 * (1 - out["chan_mse_model"] / out["chan_mse_persistence"]), 1)
    out["z_vs_persistence_pct"] = round(
        100 * (1 - out["z_mse_model"] / out["z_mse_persistence"]), 1)
    out["probe_seconds"] = round(time.time() - t0, 1)
    if was_training:
        codec.train()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--data", default=os.path.join(HERE, "cache", "na_pixels.npz"))
    ap.add_argument("--n-pixels", type=int, default=600)
    ap.add_argument("--tsteps", type=int, default=400)
    a = ap.parse_args()

    run_dir = os.path.join(HERE, "runs", a.run)
    ck = torch.load(os.path.join(run_dir, "pixelmae.pt"),
                    map_location="cpu", weights_only=False)
    d = np.load(a.data)
    X = d["X"].copy()
    months = [str(m) for m in d["months"]]
    moy = np.array([int(m[5:7]) - 1 for m in months])
    hold_years = set(ck["args"]["holdout_years"].split(","))
    t_hold = np.array([m[:4] in hold_years for m in months])
    lo, hi = (float(v) for v in ck["args"]["holdout_lon"].split(","))
    x_hold = (d["lons"] >= lo) & (d["lons"] < hi)

    if not ck["args"].get("anomaly"):
        sys.exit("trainprobe measures anomaly-space codecs only "
                 "(state space is disqualified from ranking).")
    X, dynamic = anomaly_transform(X, moy, t_hold, x_hold)

    codec = codec_from_ckpt(ck, X.shape[-1])
    codec.load_state_dict(ck["model"])
    # Standalone path only. Called from train.py the codec is already on the
    # GPU, which is why this was never noticed here; run as a script it would
    # embed on CPU, the same omission dip_check.py and rollout.py had.
    codec.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))

    out = probe_now(codec, torch.from_numpy(np.nan_to_num(X, nan=0.0)),
                    torch.from_numpy(np.isfinite(X)), d, moy, t_hold, x_hold,
                    dynamic, n_pixels=a.n_pixels, tsteps=a.tsteps)
    out["run"] = a.run
    print(json.dumps(out, indent=2))
    json.dump(out, open(os.path.join(run_dir, "trainprobe.json"), "w"), indent=2)
    print(f"wrote {run_dir}/trainprobe.json")


if __name__ == "__main__":
    main()

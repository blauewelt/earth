#!/usr/bin/env python3
"""Stage 2: a temporal transformer over frozen codec embeddings.

The K-sweep (probe_sequence.py, protocol v2) established the precondition:
ANOMALY-space embeddings gain probe skill as history is concatenated
(state-space embeddings lost it — they were seasonally redundant). A linear
read-out of stacked embeddings is the crudest possible sequence model; this
file is the honest next rung — a small causal transformer over each pixel's
embedding sequence z_{t-K+1..t}, with two jobs:

  1. DYNAMICS: predict z_{t+1} (the next month's anomaly embedding).
     Channel-space score: decode ẑ_{t+1} through the FROZEN codec decoder
     at offset 0 and compare against the true next-month channels, vs the
     persistence forecast x_{t+1} := x_t. Same blocked holdout as training.
  2. STATE: the transformer's last hidden state at the 26.5°N section,
     pooled along the section, replaces the concatenated-z features in the
     RAPID probe — same seasonality-proof protocol (deseasonalised target,
     train-years climatology, seasonal-only floor, lambda on a train tail).

Both stages stay in ANOMALY space; the codec is never fine-tuned (two-stage
by construction, so codec improvements and dynamics improvements stay
attributable). Splits are inherited from the codec checkpoint — the same
held-out years and the same mid-Atlantic longitude block, never random.

Usage:  python3 ml/temporal.py --run pilot4_anom --steps 4000
"""
import argparse
import json
import os
import sys
import time
import warnings

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model import PixelMAE
from probe_sequence import ridge_r

HERE = os.path.dirname(os.path.abspath(__file__))


class TemporalTransformer(nn.Module):
    """Causal transformer over one pixel's embedding sequence.

    Inputs are codec embeddings z_t (d_z) plus a per-pixel static context
    (lat, lon, and the codec embedding of the pixel's STATIC channels alone —
    the climatological identity of the place), added to every step. The
    month-of-year enters as sin/cos per STEP: dynamics may be phase-dependent
    (winter mixing vs summer stratification) even when the state is an
    anomaly. Output head predicts z_{t+1} from the hidden state at t.
    """

    def __init__(self, d_z=32, d_model=96, n_heads=4, n_layers=3, k_max=36):
        super().__init__()
        self.inp = nn.Linear(d_z + 2, d_model)     # z_t + (sin m, cos m)
        self.static = nn.Linear(d_z + 2, d_model)  # static-z + (lat, lon)
        self.pos = nn.Embedding(k_max, d_model)
        layer = nn.TransformerEncoderLayer(
            d_model, n_heads, dim_feedforward=4 * d_model,
            batch_first=True, norm_first=True, dropout=0.0)
        self.encoder = nn.TransformerEncoder(layer, n_layers)
        self.head = nn.Linear(d_model, d_z)
        self.d_model = d_model

    def forward(self, z_seq, month_seq, static_ctx):
        """z_seq [B,K,d_z] · month_seq [B,K,2] · static_ctx [B,d_z+2]
        → pred [B,K,d_z] (ẑ at t+1 for every step), h [B,K,d_model]."""
        B, K, _ = z_seq.shape
        h = (self.inp(torch.cat([z_seq, month_seq], -1))
             + self.static(static_ctx).unsqueeze(1)
             + self.pos.weight[None, :K])
        causal = nn.Transformer.generate_square_subsequent_mask(K, device=z_seq.device)
        h = self.encoder(h, mask=causal, is_causal=True)
        return self.head(h), h


def embed_everything(model, X, OBS, ctx_all, lats, lons, ys, xs, d_z,
                     cache_path=None, batch=8192, mask_chan=None):
    """Frozen codec embeddings for every (t, pixel in ys/xs): [T, P, d_z].
    Cached on disk — the embedding pass is the expensive part of stage 2
    (T×P encoder forwards), and every probe variant reuses it."""
    T, H, W, C = X.shape
    P = len(ys)
    coords = np.stack([lats[ys] / 90, lons[xs] / 180], 1).astype(np.float32)
    if cache_path and os.path.exists(cache_path):
        out = np.load(cache_path)
        if out.shape == (T, P, d_z):
            print(f"  (cached: {cache_path})")
            return out, coords
    out = np.zeros((T, P, d_z), dtype=np.float32)
    with torch.no_grad():
        for t in range(T):
            for i in range(0, P, batch):
                sl = slice(i, min(i + batch, P))
                n = sl.stop - sl.start
                ctx = np.concatenate([np.tile(ctx_all[t], (n, 1)), coords[sl]], 1)
                mk = torch.zeros(n, C, dtype=torch.bool)
                if mask_chan is not None:
                    mk[:, mask_chan] = True
                v = X[t, ys[sl], xs[sl]] * (~mk)
                z = model.encode(v, OBS[t, ys[sl], xs[sl]], mk,
                                 torch.as_tensor(ctx, dtype=torch.float32))
                out[t, sl] = z.numpy()
    if cache_path:
        np.save(cache_path, out)
    return out, coords


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="pilot4_anom")
    ap.add_argument("--K", type=int, default=24, help="context length (months)")
    ap.add_argument("--steps", type=int, default=4000)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--d-model", type=int, default=96)
    ap.add_argument("--layers", type=int, default=3)
    ap.add_argument("--seed", type=int, default=0,
                    help="torch/numpy seed (sweeps need more than one)")
    ap.add_argument("--tag", default="",
                    help="suffix for output files: temporal_<tag>.json/.pt")
    ap.add_argument("--max-pixels", type=int, default=0,
                    help="subsample ocean pixels (code-path smoke only; "
                         "the 26.5N section is always kept)")
    a = ap.parse_args()

    torch.manual_seed(a.seed)
    np.random.seed(a.seed)
    run_dir = os.path.join(HERE, "runs", a.run)
    ck = torch.load(os.path.join(run_dir, "pixelmae.pt"),
                    map_location="cpu", weights_only=False)
    if not ck["args"].get("anomaly"):
        sys.exit("stage 2 requires an anomaly-space codec (train.py --anomaly): "
                 "state-space embeddings failed the K-sweep precondition.")
    d = np.load(os.path.join(HERE, "cache", "na_pixels.npz"))
    X = d["X"].copy()
    months = [str(m) for m in d["months"]]
    lats, lons, chan = d["lats"], d["lons"], [str(c) for c in d["chan"]]
    T, H, W, C = X.shape
    moy = np.array([int(m[5:7]) - 1 for m in months])
    hold_years = set(ck["args"]["holdout_years"].split(","))
    t_hold = np.array([m[:4] in hold_years for m in months])
    lo, hi = (float(v) for v in ck["args"]["holdout_lon"].split(","))
    x_hold = (lons >= lo) & (lons < hi)

    # identical anomaly transform to train.py --anomaly (train-years clim)
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

    codec = PixelMAE(n_chan=C, d_z=ck["d_z"])
    codec.load_state_dict(ck["model"])
    codec.eval()

    ocean = np.isfinite(d["X"][..., 0]).any(axis=0)
    ctx_all = np.stack([np.sin(2 * np.pi * moy / 12), np.cos(2 * np.pi * moy / 12)], 1)
    Xt = torch.from_numpy(np.nan_to_num(X, nan=0.0))
    OBS = torch.from_numpy(np.isfinite(X))

    ys, xs = np.where(ocean)
    sec_y = int(np.argmin(np.abs(lats - 26.5)))
    if a.max_pixels and a.max_pixels < len(ys):
        rng = np.random.default_rng(0)
        keep = rng.choice(len(ys), a.max_pixels, replace=False)
        keep = np.union1d(keep, np.where(ys == sec_y)[0])   # probe needs the section
        ys, xs = ys[keep], xs[keep]

    print("embedding every (month, ocean pixel) through the frozen codec …")
    t0 = time.time()
    cache = (os.path.join(HERE, "cache", f"Z_{a.run}.npy")
             if not a.max_pixels else None)
    Z, coords = embed_everything(codec, Xt, OBS, ctx_all, lats, lons, ys, xs,
                                 ck["d_z"], cache_path=cache)
    P = len(ys)
    print(f"  Z [T={T} P={P} d_z={ck['d_z']}]  ({time.time() - t0:.0f}s)")

    # static identity of each pixel: codec embedding of static channels only
    with torch.no_grad():
        stat_obs = OBS[0].clone()
        for c in dynamic:
            stat_obs[..., c] = False
        zs = []
        for i in range(0, P, 8192):
            sl = slice(i, min(i + 8192, P))
            n = sl.stop - sl.start
            ctx = np.concatenate([np.zeros((n, 2), np.float32), coords[sl]], 1)
            zs.append(codec.encode(Xt[0, ys[sl], xs[sl]], stat_obs[ys[sl], xs[sl]],
                                   torch.zeros(n, C, dtype=torch.bool),
                                   torch.as_tensor(ctx)).numpy())
        Zstat = np.concatenate(zs, 0)
    static_ctx = torch.as_tensor(np.concatenate([Zstat, coords], 1))

    # ---- train pool: windows [t-K+1 .. t] whose TARGET month t+1 is a train
    # month and whose pixel is outside the longitude holdout. Windows may LOOK
    # at held-out months (persistence can too); they may never be SCORED on
    # them in training.
    Zt = torch.from_numpy(Z)
    Mt = torch.as_tensor(ctx_all, dtype=torch.float32)
    K = a.K
    ok_t = np.array([t + 1 < T and not t_hold[t + 1] and t + 1 >= K
                     for t in range(T)])
    ok_p = ~x_hold[xs]
    pool_t, pool_p = np.where(ok_t[:, None] & ok_p[None, :])
    pool_t = torch.as_tensor(pool_t, dtype=torch.long)
    pool_p = torch.as_tensor(pool_p, dtype=torch.long)
    print(f"train windows: {len(pool_t):,}")

    model = TemporalTransformer(d_z=ck["d_z"], d_model=a.d_model,
                                n_layers=a.layers, k_max=K)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, a.steps)

    def batch_windows(idx_t, idx_p, n):
        k = torch.randint(0, len(idx_t), (n,))
        t, p = idx_t[k], idx_p[k]
        base = t - K + 1
        zseq = torch.stack([Zt[base + j, p] for j in range(K)], 1)
        mseq = torch.stack([Mt[base + j] for j in range(K)], 1)
        ztgt = torch.stack([Zt[base + j + 1, p] for j in range(K)], 1)
        return zseq, mseq, static_ctx[p], ztgt

    print("training the temporal stage …")
    t0 = time.time()
    for s in range(1, a.steps + 1):
        zseq, mseq, sctx, ztgt = batch_windows(pool_t, pool_p, a.batch)
        pred, _ = model(zseq, mseq, sctx)
        loss = (pred - ztgt).pow(2).mean()
        opt.zero_grad(); loss.backward(); opt.step(); sched.step()
        if s % max(1, a.steps // 10) == 0:
            print(f"  step {s:>6}/{a.steps}  z-mse {loss.item():.4f}"
                  f"  ({time.time() - t0:.0f}s)", flush=True)

    model.eval()
    results = {"run": a.run, "K": K, "d_model": a.d_model, "layers": a.layers,
               "steps": a.steps}

    # ---- eval 1: z-space t+1 on held-out target months --------------------
    with torch.no_grad():
        ev_t = np.array([t + 1 < T and t_hold[t + 1] and t + 1 >= K
                         for t in range(T)])
        et, ep = np.where(ev_t[:, None] & np.ones(P, bool)[None, :])
        sel = np.random.default_rng(a.seed).choice(len(et), min(20000, len(et)), replace=False)
        et = torch.as_tensor(et[sel], dtype=torch.long)
        ep = torch.as_tensor(ep[sel], dtype=torch.long)
        base = et - K + 1
        zseq = torch.stack([Zt[base + j, ep] for j in range(K)], 1)
        mseq = torch.stack([Mt[base + j] for j in range(K)], 1)
        pred, hid = model(zseq, mseq, static_ctx[ep])
        zhat = pred[:, -1]                       # ẑ_{t+1}
        ztrue = Zt[et + 1, ep]
        zlast = Zt[et, ep]                        # persistence in z
        results["z_t+1"] = {
            "mse_model": float((zhat - ztrue).pow(2).mean()),
            "mse_persistence": float((zlast - ztrue).pow(2).mean()),
        }
        results["z_t+1"]["beats_persistence"] = (
            results["z_t+1"]["mse_model"] < results["z_t+1"]["mse_persistence"])

        # ---- eval 2: decode ẑ through the frozen codec → channel space ----
        qc = torch.arange(C)[None, :].expand(len(et), -1)
        off0 = torch.zeros(len(et), C, 3, dtype=torch.long)
        xhat = codec.query(zhat, qc, off0)
        ys_t = torch.as_tensor(ys, dtype=torch.long)
        xs_t = torch.as_tensor(xs, dtype=torch.long)
        v1 = Xt[et + 1, ys_t[ep], xs_t[ep]]
        o1 = OBS[et + 1, ys_t[ep], xs_t[ep]]
        v0 = Xt[et, ys_t[ep], xs_t[ep]]
        o0 = OBS[et, ys_t[ep], xs_t[ep]]
        both = o0 & o1
        dyn = torch.zeros(C, dtype=torch.bool); dyn[dynamic] = True
        both = both & dyn[None, :]
        mse_m = float(((xhat - v1).pow(2) * both).sum() / both.sum())
        mse_p = float(((v0 - v1).pow(2) * both).sum() / both.sum())
        results["chan_t+1"] = {"mse_model": mse_m, "mse_persistence": mse_p,
                               "beats_persistence": mse_m < mse_p,
                               "channels": [chan[c] for c in dynamic]}

    # ---- eval 3: RAPID probe from temporal hidden state -------------------
    # protocol v2: deseasonalised target (train-years clim), seasonal floor,
    # lambda on a train tail — identical scoring path to probe_sequence.py.
    rapid = d["rapid"]
    sec_pix = torch.as_tensor(np.where(ys == sec_y)[0], dtype=torch.long)
    with torch.no_grad():
        F = np.zeros((T, a.d_model), dtype=np.float32)
        for t in range(K - 1, T):
            base = t - K + 1
            zseq = torch.stack([Zt[base + j, sec_pix] for j in range(K)], 1)
            mseq = torch.stack([Mt[base + j].expand(len(sec_pix), -1)
                                for j in range(K)], 1)
            _, hid = model(zseq, mseq, static_ctx[sec_pix])
            F[t] = hid[:, -1].mean(0).numpy()    # pool along the section
    ridx = rapid[:, 0].astype(int)
    rv_raw = rapid[:, 1].copy()
    rmoy = moy[ridx]
    tr_all = ~t_hold[ridx]
    rclim = np.array([rv_raw[tr_all & (rmoy == m)].mean() for m in range(12)])
    rv_des = rv_raw - rclim[rmoy]
    ok = ridx >= K - 1
    ri = ridx[ok]
    tr, te = ~t_hold[ri], t_hold[ri]
    r_raw, _ = ridge_r(F[ri], rv_raw[ok], tr, te)
    r_des, _ = ridge_r(F[ri], rv_des[ok], tr, te)
    results["rapid_probe"] = {"r_raw": r_raw, "r_deseasonalised": r_des,
                              "n_test": int(te.sum()), "features": "hidden(-1) mean over section"}

    print(json.dumps(results, indent=2))
    suffix = f"_{a.tag}" if a.tag else ""
    results["seed"] = a.seed
    torch.save({"model": model.state_dict(), "args": vars(a)},
               os.path.join(run_dir, f"temporal{suffix}.pt"))
    json.dump(results, open(os.path.join(run_dir, f"temporal{suffix}.json"), "w"), indent=2)
    print(f"saved {run_dir}/temporal{suffix}.pt")


if __name__ == "__main__":
    main()

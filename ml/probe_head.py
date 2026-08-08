#!/usr/bin/env python3
"""The top rung of the probe ladder: a supervised attention head over the
UNPOOLED section.

Every other probe in this project — ridge, MLP, the stage-2 hidden-state
read-out — mean-pools the section's ~67 pixel embeddings into one vector
before looking at them. Geostrophic transport is the east-minus-west
density difference ACROSS the section; mean-pooling destroys precisely the
cross-pixel structure the physics lives in. This probe keeps every
(pixel, month) token and lets one cross-attention query learn what to pool.

Ladder semantics (each rung isolates one capability):
  ridge  — what is LINEARLY accessible in the pooled embedding
  mlp    — plus pointwise nonlinearity            (probe_kfold --probe mlp)
  head   — plus spatial structure across the section        (this file)
If head >> mlp, the embedding carries section-structure information the
pooled probes cannot reach. If head ~= mlp, pooling loses nothing and the
representation itself is the limit.

The codec stays FROZEN — this is still a probe, not fine-tuning; gradients
stop at the cached embeddings. Same year-blocked folds, same inner-tail
early stopping, 3 seeds averaged per fold; with n~240 and ~25k parameters
the head is regularized hard (weight decay 1e-2, dropout on tokens).

Usage:
  python3 ml/probe_head.py --run global14 --data ml/cache/na_pixels_c14_global.npz
  python3 ml/probe_head.py --run pixel25_40k --data ml/cache/na_pixels_c25_global.npz --K 3
"""
import argparse
import json
import os
import sys

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model import PixelMAE
from trainprobe import anomaly_transform
from temporal import embed_everything, rapid_section

HERE = os.path.dirname(os.path.abspath(__file__))


class SectionHead(nn.Module):
    """One learned query cross-attends over (pixel x month) tokens.

    Deliberately the SMALLEST architecture that can express 'compare the
    two ends of the section': tokens get a linear lift plus a longitude
    encoding (so 'east' and 'west' are distinguishable after attention),
    one single-head cross-attention pools them, a two-layer MLP reads the
    pooled vector out. ~25k parameters at d=64."""

    def __init__(self, d_z, d=64, K=1):
        super().__init__()
        self.lift = nn.Linear(d_z + 2, d)      # z + (lon_frac, month_idx/K)
        self.q = nn.Parameter(torch.randn(1, 1, d) / d ** 0.5)
        self.att = nn.MultiheadAttention(d, num_heads=1, batch_first=True)
        self.out = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, 32),
                                 nn.GELU(), nn.Linear(32, 1))
        self.drop = nn.Dropout(0.1)

    def forward(self, tok):                     # tok [B, P*K, d_z+2]
        h = self.drop(self.lift(tok))
        pooled, _ = self.att(self.q.expand(len(tok), -1, -1), h, h)
        return self.out(pooled[:, 0]).squeeze(-1)


def fold_fit(Xtr, ytr, Xte, d_z, seed, steps=4000):
    torch.manual_seed(seed)
    g = torch.Generator().manual_seed(seed)
    net = SectionHead(d_z)
    opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-2)
    n = len(Xtr)
    fit = slice(0, int(0.8 * n))
    val = slice(int(0.8 * n), n)
    Xf, yf = Xtr[fit], torch.as_tensor(ytr[fit], dtype=torch.float32)
    Xv, yv = Xtr[val], torch.as_tensor(ytr[val], dtype=torch.float32)
    best, best_state, patience = np.inf, None, 0
    for s in range(steps):
        k = torch.randint(0, len(Xf), (min(32, len(Xf)),), generator=g)
        loss = (net(Xf[k]) - yf[k]).pow(2).mean()
        opt.zero_grad(); loss.backward(); opt.step()
        if s % 50 == 0:
            net.eval()
            with torch.no_grad():
                v = (net(Xv) - yv).pow(2).mean().item()
            net.train()
            if v < best - 1e-6:
                best, patience = v, 0
                best_state = {k2: v2.clone() for k2, v2 in net.state_dict().items()}
            else:
                patience += 1
                if patience >= 12:
                    break
    if best_state:
        net.load_state_dict(best_state)
    net.eval()
    with torch.no_grad():
        return net(Xte).numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--data", default=os.path.join(HERE, "cache", "na_pixels.npz"))
    ap.add_argument("--K", type=int, default=1,
                    help="months of context per sample (1 = instantaneous)")
    a = ap.parse_args()

    ck = torch.load(os.path.join(HERE, "runs", a.run, "pixelmae.pt"),
                    map_location="cpu", weights_only=False)
    d = np.load(a.data)
    if len(ck["chan"]) != d["X"].shape[-1]:
        sys.exit(f"{a.run}: codec has {len(ck['chan'])} channels but the tensor "
                 f"has {d['X'].shape[-1]} — pass --data with the matching tensor.")
    X = d["X"]
    months = [str(m) for m in d["months"]]
    moy = np.array([int(m[5:7]) - 1 for m in months])
    yr = np.array([int(m[:4]) for m in months])
    lats, lons = d["lats"], d["lons"]
    t_hold = np.array([m[:4] in set(ck["args"]["holdout_years"].split(","))
                       for m in months])
    lo, hi = (float(v) for v in ck["args"]["holdout_lon"].split(","))
    x_hold = (lons >= lo) & (lons < hi)
    Xa, _ = anomaly_transform(X, moy, t_hold, x_hold)
    del X

    codec = PixelMAE(n_chan=Xa.shape[-1], d_z=ck["d_z"],
                     patch=ck["args"].get("patch", 1))
    codec.load_state_dict(ck["model"])
    codec.eval()
    OBS = torch.from_numpy(np.isfinite(Xa))
    np.nan_to_num(Xa, nan=0.0, copy=False)
    Xt = torch.from_numpy(Xa)

    ctx_all = np.stack([np.sin(2 * np.pi * moy / 12),
                        np.cos(2 * np.pi * moy / 12)], 1)
    ocean = OBS[..., 0].any(axis=0).numpy()
    ys, xs = np.where(ocean)
    sec_y, sec_sel = rapid_section(lats, lons, ys, xs)
    Z, _ = embed_everything(codec, Xt, OBS, ctx_all, lats, lons,
                            ys[sec_sel], xs[sec_sel], codec.d_z)
    P = Z.shape[1]
    lon_frac = ((lons[xs[sec_sel]] - lons[xs[sec_sel]].min())
                / max(1e-6, np.ptp(lons[xs[sec_sel]]))).astype(np.float32)

    rapid = d["rapid"]
    ridx = rapid[:, 0].astype(int)
    vals = rapid[:, 1].copy()
    rmoy = moy[ridx]
    clim = np.array([vals[rmoy == m].mean() for m in range(12)])
    v_des = vals - clim[rmoy]
    ok = ridx >= a.K - 1
    ridx, v_des = ridx[ok], v_des[ok]

    # tokens [n, P*K, d_z+2]: embedding + (lon position, month offset)
    toks = np.zeros((len(ridx), P * a.K, codec.d_z + 2), dtype=np.float32)
    for i, t in enumerate(ridx):
        for j in range(a.K):
            z = Z[t - j]                              # [P, d_z]
            block = slice(j * P, (j + 1) * P)
            toks[i, block, :codec.d_z] = z
            toks[i, block, codec.d_z] = lon_frac
            toks[i, block, codec.d_z + 1] = j / max(1, a.K - 1) if a.K > 1 else 0.0
    T_ = torch.as_tensor(toks)

    pred = np.full(len(v_des), np.nan)
    years = yr[ridx]
    for yy_ in np.unique(years):
        te = years == yy_
        tr = ~te
        # standardize target on train only
        mu, sd = v_des[tr].mean(), v_des[tr].std() + 1e-9
        p = np.mean([fold_fit(T_[tr], (v_des[tr] - mu) / sd, T_[te],
                              codec.d_z, sd_)
                     for sd_ in (0, 1, 2)], axis=0)
        pred[te] = p * sd + mu
    okp = np.isfinite(pred)
    r = float(np.corrcoef(pred[okp], v_des[okp])[0, 1])
    rmse = float(np.sqrt(np.mean((pred[okp] - v_des[okp]) ** 2)))

    # block bootstrap over years, same as the ridge
    rng = np.random.default_rng(0)
    uy = np.unique(years)
    rs = []
    for _ in range(2000):
        pick = rng.choice(uy, len(uy), replace=True)
        sel = np.concatenate([np.where(years == p_)[0] for p_ in pick])
        if np.std(v_des[sel]) > 0:
            rs.append(np.corrcoef(pred[sel], v_des[sel])[0, 1])
    lo95, hi95 = np.percentile(rs, [2.5, 97.5])

    out = {"run": a.run, "probe": "attention-head", "K": a.K,
           "r_kfold_deseas": round(r, 3),
           "ci95": [round(float(lo95), 3), round(float(hi95), 3)],
           "rmse_sv": round(rmse, 2), "n": int(okp.sum()),
           "note": "unpooled section: one query attends over "
                   f"{P} pixels x {a.K} months"}
    print(f"{a.run} head-probe (K={a.K}): rapid k-fold r {r:+.3f} "
          f"[{lo95:+.3f}, {hi95:+.3f}] · RMSE {rmse:.2f} Sv")
    path = os.path.join(HERE, "runs", a.run, "probe_head.json")
    json.dump(out, open(path, "w"), indent=2)
    print("wrote", path)


if __name__ == "__main__":
    main()

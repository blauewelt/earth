"""E-019b1: decoder-only retrain against FROZEN run-62 embeddings.

Chris: "Yes. Please run it. You can even experiment with various decoder
sizes until you found the optimum."

The question this answers (EXPERIMENTS.md E-019b): is the deep-T variance
the E-019a audit found missing (6.9%, 3x the upper ocean) absent from z,
or merely unexpressed by the production ~1.3M decoder? The encoder is
untouched — z comes from the published embed cache — so any recovery here
proves the information was in z all along, and any residual after enough
decoder capacity is a true property of the embedding.

Architecture note, stated honestly: the production decoder is a per-channel
query MLP (z + chan_emb + off_emb -> scalar). For a CPU-affordable sweep
this trains a MULTI-OUTPUT head instead (z -> hidden^L -> C values, offset
0 only): one forward covers all 39 channels, which is ~39x cheaper per
pixel-month and answers the same information question — any decoder that
reads ONLY z is a valid witness. It is not a drop-in for rollout's query()
path; if a size wins decisively, a query-shaped twin can be trained then.

Protocol guards:
  · Trains ONLY on train months x non-holdout longitudes (the codec's own
    blocked splits, read from the checkpoint args), so the audit's held-out
    splits stay clean.
  · The Z cache is VERIFIED against a local f32 re-encode of sample section
    pixel-months before anything trains (ordering or preprocessing mismatch
    = refuse). f16 quantization bounds the expected delta.
  · Scored by the exact E-019a section audit (same splits, same metrics),
    so numbers are directly comparable to the production decoder's row.

Usage:
  python3 ml/recon_decoder.py --x ml/cache/family3_X.npy \
      --npz-small ml/cache/f3_small.npz --z ml/cache/Z.npy \
      --ckpt ml/cache/f3_anchor41M__pixelmae.pt \
      --hidden 1536 --layers 3 --steps 4000 --batch 4096 \
      --out ml/runs/recon_decoder/L.json
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

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from model import codec_from_ckpt                              # noqa: E402
from recon_eval import (stream_stats, build_slab, score,       # noqa: E402
                        RAPID_LAT, RAPID_LON)
from temporal import section_of, embed_everything              # noqa: E402


class MultiDec(nn.Module):
    """z -> hidden^layers -> C. GELU, same family as the production MLP."""
    def __init__(self, d_z, C, hidden, layers):
        super().__init__()
        seq = [nn.Linear(d_z, hidden), nn.GELU()]
        for _ in range(layers - 1):
            seq += [nn.Linear(hidden, hidden), nn.GELU()]
        seq += [nn.Linear(hidden, C)]
        self.net = nn.Sequential(*seq)

    def forward(self, z):
        return self.net(z)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--x", required=True)
    ap.add_argument("--npz-small", required=True,
                    help="npz with months/lats/lons/chan (small members only)")
    ap.add_argument("--z", required=True, help="assembled Z cache .npy (f16)")
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--hidden", type=int, required=True)
    ap.add_argument("--layers", type=int, required=True)
    ap.add_argument("--steps", type=int, default=4000)
    ap.add_argument("--batch", type=int, default=4096)
    ap.add_argument("--pairs", type=int, default=6_000_000)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    ap.add_argument("--cache-dir", default=os.path.join(HERE, "cache"))
    ap.add_argument("--skip-verify", action="store_true")
    a = ap.parse_args()
    torch.manual_seed(a.seed)
    rng = np.random.default_rng(a.seed)

    ck = torch.load(a.ckpt, map_location="cpu", weights_only=False)
    d = np.load(a.npz_small, allow_pickle=False)
    months = [str(m) for m in d["months"]]
    lats, lons = d["lats"], d["lons"]
    chan = [str(c) for c in d["chan"]]
    Xm = np.load(a.x, mmap_mode="r")
    T, H, W, C = Xm.shape
    moy = np.array([int(m[5:7]) - 1 for m in months])
    hold_years = set(ck["args"]["holdout_years"].split(","))
    t_hold = np.array([m[:4] in hold_years for m in months])
    lo, hi = (float(v) for v in ck["args"]["holdout_lon"].split(","))
    x_hold = (lons >= lo) & (lons < hi)

    # ---- ocean mask & pixel ordering (must match temporal.py exactly) -----
    om_path = os.path.join(a.cache_dir, "ocean_mask.npy")
    if os.path.exists(om_path):
        ocean = np.load(om_path)
    else:
        ocean = np.zeros((H, W), bool)
        for t0 in range(0, T, 16):
            xb = np.asarray(Xm[t0:t0 + 16, :, :, 0])
            ocean |= np.isfinite(xb).any(axis=0)
        np.save(om_path, ocean)
    ys, xs = np.where(ocean)
    P = len(ys)

    Zm = np.load(a.z, mmap_mode="r")
    assert Zm.shape == (T, P, ck["d_z"]), \
        f"Z cache shape {Zm.shape} != expected {(T, P, ck['d_z'])} — the " \
        f"pixel ordering or tensor differs; refusing"
    assert Zm.dtype == np.float16, Zm.dtype

    # ---- standardization stats (streaming; verified impl from E-019a) ----
    st_path = os.path.join(a.cache_dir, "std_stats.npz")
    if os.path.exists(st_path):
        s = np.load(st_path)
        clim, dyn = s["clim"], list(s["dyn"])
        mean_c, std_c = s["mean_c"], s["std_c"]
    else:
        clim, dyn, mean_c, std_c = stream_stats(Xm, moy, t_hold, x_hold)
        np.savez(st_path, clim=clim, dyn=np.array(dyn),
                 mean_c=mean_c, std_c=std_c)
    print(f"stats ready: {len(dyn)}/{C} dynamic", flush=True)

    # ---- section frame (identical to E-019a) ------------------------------
    sec_y = int(np.argmin(np.abs(lats - RAPID_LAT)))
    rows = [sec_y - 1, sec_y, sec_y + 1]
    slab, obs = build_slab(Xm, rows, moy, clim, dyn, mean_c, std_c)
    ocean_row = obs[:, 1, :, 0].any(axis=0)
    xs_all = np.where(ocean_row)[0]
    _, sel = section_of(lats[rows], lons, np.ones(len(xs_all), dtype=int),
                        xs_all, RAPID_LAT, *RAPID_LON)
    xs_sec = xs_all[sel]
    # indices of section pixels inside the Z cache's P ordering
    lin = np.zeros((H, W), np.int64)
    lin[ys, xs] = np.arange(P)
    sec_pidx = lin[sec_y, xs_sec]
    assert (ys[sec_pidx] == sec_y).all() and (xs[sec_pidx] == xs_sec).all()
    truth_sec = np.nan_to_num(slab[:, 1][:, xs_sec], nan=0.0)   # [T,S,C]
    obs_sec = obs[:, 1][:, xs_sec]
    Z_sec = np.asarray(Zm[:, sec_pidx]).astype(np.float32)      # [T,S,dz]
    print(f"section: {len(xs_sec)} px", flush=True)

    # ---- verify the Z cache against a local f32 re-encode -----------------
    if not a.skip_verify:
        codec = codec_from_ckpt(ck, C)
        codec.load_state_dict(ck["model"])
        codec.eval()
        ctx_all = np.stack([np.sin(2 * np.pi * moy / 12),
                            np.cos(2 * np.pi * moy / 12)], 1)
        k = rng.choice(len(xs_sec), 8, replace=False)
        Zl, _ = embed_everything(
            codec, torch.from_numpy(np.nan_to_num(slab, nan=0.0)),
            torch.from_numpy(obs), ctx_all, lats[rows], lons,
            np.ones(len(k), dtype=int), xs_sec[k], ck["d_z"],
            cache_path=None, batch=64)
        for j, tt in enumerate([0, T // 2, T - 1]):
            dmax = float(np.abs(Zl[tt] - Z_sec[tt][k]).max())
            zscale = float(np.abs(Zl[tt]).max())
            assert dmax < max(0.02, 0.005 * zscale), \
                f"Z cache mismatch at t={tt}: max|Δ|={dmax} (scale {zscale})"
        print(f"Z cache verified vs local re-encode (f16 tolerance) ✓",
              flush=True)
        del codec

    # ---- training pairs: train months × non-holdout-lon ocean pixels ------
    # Cached to disk keyed by (pairs, seed): every size in the sweep then
    # trains on IDENTICAL pairs, and the 10.9 GB gather runs once.
    pc = os.path.join(a.cache_dir, f"pairs_{a.pairs}_{a.seed}.npz")
    if os.path.exists(pc):
        pcd = np.load(pc)
        ZT, XT = pcd["ZT"], pcd["XT"]
        N = len(ZT)
        print(f"pairs: {N:,} (cache hit {os.path.basename(pc)})", flush=True)
    else:
        keep_p = ~x_hold[xs]                   # per-P pixel eligibility
        train_t = np.where(~t_hold)[0]
        per_t = a.pairs // len(train_t)
        zs_l, tr_l = [], []
        t0 = time.time()
        elig = np.where(keep_p)[0]
        for i, t in enumerate(train_t):
            sel_p = rng.choice(elig, min(per_t, len(elig)), replace=False)
            sel_p.sort()
            zs_l.append(np.asarray(Zm[t, sel_p]))              # f16 [n,dz]
            xb = np.asarray(Xm[t, ys[sel_p], xs[sel_p]]).astype(np.float32)
            fin = np.isfinite(xb)
            for c in dyn:
                xb[:, c] = ((xb[:, c] - clim[moy[t], ys[sel_p], xs[sel_p], c]
                             - mean_c[c]) / (std_c[c] + 1e-6))
            xb[~fin] = np.nan
            tr_l.append(xb.astype(np.float16))
            if i % 100 == 0:
                print(f"  gather {i}/{len(train_t)} ({time.time()-t0:.0f}s)",
                      flush=True)
        ZT = np.concatenate(zs_l); del zs_l
        XT = np.concatenate(tr_l); del tr_l
        N = len(ZT)
        np.savez(pc, ZT=ZT, XT=XT)
        print(f"pairs: {N:,} ({time.time()-t0:.0f}s; cached)", flush=True)

    # 2% validation split, capped so the periodic val forward stays cheap on
    # CPU (from the same train pool; the section holdout splits are never
    # trained on and never used for early stopping)
    n_val = min(N // 50, 32768)
    np.random.seed(a.seed)
    perm = rng.permutation(N)
    vi, ti = perm[:n_val], perm[n_val:]
    Zv = torch.from_numpy(ZT[vi].astype(np.float32))
    Xv = torch.from_numpy(XT[vi].astype(np.float32))
    Mv = torch.isfinite(Xv)
    Xv = torch.nan_to_num(Xv, nan=0.0)

    dec = MultiDec(ck["d_z"], C, a.hidden, a.layers)
    n_par = sum(p.numel() for p in dec.parameters())
    print(f"decoder {a.hidden}x{a.layers}: {n_par:,} params", flush=True)
    opt = torch.optim.AdamW(dec.parameters(), lr=a.lr, weight_decay=1e-4)
    import math
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda e: 0.5 * (1 + math.cos(math.pi * min(e, a.steps) / a.steps)))

    best_val, best_state, t0 = float("inf"), None, time.time()
    for s in range(1, a.steps + 1):
        idx = ti[np.random.randint(0, len(ti), a.batch)]
        z = torch.from_numpy(ZT[idx].astype(np.float32))
        x = torch.from_numpy(XT[idx].astype(np.float32))
        m = torch.isfinite(x)
        x = torch.nan_to_num(x, nan=0.0)
        pred = dec(z)
        loss = (((pred - x) ** 2) * m).sum() / m.sum().clamp(min=1)
        opt.zero_grad(); loss.backward(); opt.step(); sched.step()
        if s % max(1, a.steps // 40) == 0 or s == a.steps:
            with torch.no_grad():
                pv = dec(Zv)
                vl = float(((((pv - Xv) ** 2) * Mv).sum()
                            / Mv.sum().clamp(min=1)))
            tag = ""
            if vl < best_val:
                best_val, tag = vl, "  *best"
                best_state = {k: v.clone() for k, v in dec.state_dict().items()}
            print(f"  step {s:>5}/{a.steps}  train {float(loss):.4f}  "
                  f"val {vl:.4f}  ({time.time()-t0:.0f}s){tag}", flush=True)
    dec.load_state_dict(best_state)

    # ---- score with the exact E-019a section audit ------------------------
    with torch.no_grad():
        pred_sec = np.stack([dec(torch.from_numpy(Z_sec[t])).numpy()
                             for t in range(T)])
    px_hold = x_hold[xs_sec]
    sel_train_t = np.where(~t_hold)[0]
    sel_hold_t = np.where(t_hold)[0]
    sel_train_x = np.where(~px_hold)[0]
    sel_hold_x = np.where(px_hold)[0]
    res = {
        "arch": f"multi-out {a.hidden}x{a.layers}", "params": n_par,
        "steps": a.steps, "batch": a.batch, "pairs": N,
        "best_val_mse": round(best_val, 5),
        "chan": chan,
        "splits": {
            "train": score(truth_sec, pred_sec, obs_sec,
                           sel_train_t, sel_train_x, "train"),
            "heldout_months": score(truth_sec, pred_sec, obs_sec,
                                    sel_hold_t, sel_train_x, "hold-t"),
            "heldout_lons": score(truth_sec, pred_sec, obs_sec,
                                  np.arange(T), sel_hold_x, "hold-x"),
        },
    }
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with open(a.out, "w") as f:
        json.dump(res, f, indent=1)
    tr = res["splits"]["train"]
    rs = [v["r"] for v in tr.values() if np.isfinite(v["r"])]
    deep = [i for i, nm in enumerate(chan)
            if nm.startswith(("rg_t", "rg_s")) and nm[4:].isdigit()
            and int(nm[4:]) >= 900]
    dr = [tr[c]["r"] for c in deep if c in tr]
    print(f"\n== {a.hidden}x{a.layers} ({n_par/1e6:.1f}M) ==  mean r "
          f"{np.mean(rs):.4f} · deep-channel mean r {np.mean(dr):.4f}")
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()

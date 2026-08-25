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

E-049. This is the DECODER CEILING rung (plan §4c) and therefore the rung the
falsifier is evaluated at, so the pre-audit adaptation reaches it too. Almost
all of it is INHERITED rather than duplicated: `stream_stats` (the float16
accumulator and the uint16 climatology counter), `check_stats` (the refusal),
`open_x` (the npz input path), `argo_split_block` (the per-(bin, pixel) Argo
split) and `bottleneck_spec` all live in `ml/recon_eval.py` and are imported
from there — ONE copy, because two copies of a silent-overflow fix is one copy
that will be missed. What is local to this file:

  · the cached-stats path now carries the tensor's identity and the guard's
    version (see `stats_cache_key`). The old caches were named `std_stats.npz`
    and `ocean_mask.npy` FLAT, so pointing this script at a second tensor
    silently reused the first one's climatology — harmless while there was one
    tensor, wrong the moment there are two, and it would have re-admitted the
    exact statistics the fix above refuses.
  · the training pool is expressed in TRAIN BINS, not train months. The logic
    was already index-based and generalises unchanged at pentad cadence; the
    naming did not, and `per_t = pairs // len(train_t)` silently becomes 0 for
    a long axis and a small `--pairs`, which is now a refusal.
  · the deep-channel summary no longer prints `nan` off an empty selection.

The ANOMALY TRANSFORM IS NOT TOUCHED. It is a 12-bin climatology keyed on
`int(m[5:7]) - 1` from `%Y-%m` labels, which is exactly what `ml/train.py:496`
does for a per-bin pentad codec — the pentad label is the bin's START MONTH
(`ml/build_family4.py:897`), so ~6 bins share each key and the climatology is
a month-of-year climatology at both cadences. `tests/test_e049_recon_audit.py`
check 6 pins the agreement against `trainprobe.anomaly_transform` itself.

Usage:
  python3 ml/recon_decoder.py --x ml/cache/family3_X.npy \
      --npz-small ml/cache/f3_small.npz --z ml/cache/Z.npy \
      --ckpt ml/cache/f3_anchor41M__pixelmae.pt \
      --hidden 1536 --layers 3 --steps 4000 --batch 4096 \
      --out ml/runs/recon_decoder/L.json
"""
import argparse
import hashlib
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
                        check_stats, open_x, argo_split_block,
                        bottleneck_spec, bottleneck_line,
                        print_argo_summary,
                        RAPID_LAT, RAPID_LON)
from temporal import section_of, embed_everything              # noqa: E402

# Bumped whenever the meaning of a cached statistic changes. E-049 is the
# first bump: caches written before the float16 accumulator fix hold a
# climatology divided by a wrapped uint8 count and a `dyn` list missing every
# overflowed channel, and nothing in the file would say so.
STATS_CACHE_VERSION = "e049"


def stats_cache_key(x_path, shape, dtype):
    """A cache name that cannot collide across tensors or across revisions.

    `std_stats.npz` and `ocean_mask.npy` were flat names in `--cache-dir`.
    That was correct while family 3 was the only tensor this script had ever
    been pointed at, and it becomes a silent wrong-answer generator the moment
    a pentad tensor lands in the same cache directory: the shapes differ, so
    `clim` would fail loudly — but a family4 r1 (39ch) and r2 (40ch) pair, or
    two revisions of the same tensor, can share a shape and not a climatology.
    """
    h = hashlib.sha256(
        f"{os.path.basename(x_path)}|{tuple(shape)}|{np.dtype(dtype)}"
        f"|{STATS_CACHE_VERSION}".encode()).hexdigest()[:10]
    return h


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
    ap.add_argument("--x", required=True,
                    help="extracted X.npy (memmapped) OR the tensor .npz "
                         "(sidecar / uncompressed member) — see "
                         "recon_eval.open_x")
    ap.add_argument("--npz-small",
                    help="npz with months/lats/lons/chan (small members only); "
                         "defaults to --x when that is itself an npz")
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
    Xm, meta = open_x(a.x)
    if a.npz_small:
        d = np.load(a.npz_small, allow_pickle=False)
    elif meta is not None:
        d = meta
    else:
        ap.error("--npz-small is required when --x is a bare .npy")
    months = [str(m) for m in d["months"]]
    lats, lons = d["lats"], d["lons"]
    chan = [str(c) for c in d["chan"]]
    T, H, W, C = Xm.shape
    bspec = bottleneck_spec(ck)
    print(f"X [T={T} H={H} W={W} C={C}] {Xm.dtype}", flush=True)
    print(bottleneck_line(bspec), flush=True)
    moy = np.array([int(m[5:7]) - 1 for m in months])
    hold_years = set(ck["args"]["holdout_years"].split(","))
    t_hold = np.array([m[:4] in hold_years for m in months])
    lo, hi = (float(v) for v in ck["args"]["holdout_lon"].split(","))
    x_hold = (lons >= lo) & (lons < hi)

    # ---- ocean mask & pixel ordering (must match temporal.py exactly) -----
    tag = stats_cache_key(a.x, Xm.shape, Xm.dtype)
    om_path = os.path.join(a.cache_dir, f"ocean_mask_{tag}.npy")
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
    st_path = os.path.join(a.cache_dir, f"std_stats_{tag}.npz")
    if os.path.exists(st_path):
        s = np.load(st_path)
        clim, dyn = s["clim"], [int(c) for c in s["dyn"]]
        mean_c, std_c = s["mean_c"], s["std_c"]
        obs_n, bad_mean_n = s["obs_n"], s["bad_mean_n"]
    else:
        clim, dyn, mean_c, std_c = stream_stats(Xm, moy, t_hold, x_hold,
                                                chan=chan)
        # The observed count per channel, so the cached path can run the same
        # refusal the streaming path ran. Cheap: one pass of the finite mask
        # per chunk, and the alternative is a cache that skips the guard.
        obs_n = np.zeros(C, np.int64)
        for t0 in range(0, T, 16):
            obs_n += np.isfinite(np.asarray(Xm[t0:t0 + 16])).sum(axis=(0, 1, 2))
        # zero BY CONSTRUCTION: stream_stats RAISES before returning if any
        # channel had observations and a non-finite spatial mean, so a cache
        # that exists at all is a cache whose statistics passed the guard.
        bad_mean_n = np.zeros(C, np.int64)
        np.savez(st_path, clim=clim, dyn=np.array(dyn, dtype=np.int64),
                 mean_c=mean_c, std_c=std_c, obs_n=obs_n,
                 bad_mean_n=bad_mean_n)
    # A cache is an artefact like any other, and §0.1 says verify the artefact.
    check_stats(chan, dyn, mean_c, std_c, obs_n, bad_mean_n,
                where=f"stream_stats (cache {os.path.basename(st_path)})")
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
        if torch.cuda.is_available():
            codec = codec.cuda()   # embed_everything follows the model's device
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

    # ---- training pairs: train BINS × non-holdout-lon ocean pixels ---------
    # Cached to disk keyed by (tensor, pairs, seed): every size in the sweep
    # then trains on IDENTICAL pairs, and the 10.9 GB gather runs once.
    #
    # "BINS", not "months". The selection was always index-based — `train_t`
    # is `np.where(~t_hold)[0]` over whatever the time axis is — and it
    # generalises to pentad cadence unchanged: `t_hold` comes from the label's
    # YEAR (`m[:4] in hold_years`), which is the same blocked-year holdout at
    # both cadences, and every downstream use indexes `Zm[t]` / `Xm[t]` by
    # position. Only the arithmetic below has a cadence-dependent failure mode:
    # at family 3 `len(train_t)` is ~430 and `--pairs 6,000,000` gives 13,900
    # pixels per bin; at pentad it is ~2,700 and gives 2,222; on a longer axis
    # with a small `--pairs` it reaches 0 and the gather silently produces an
    # EMPTY pair set, which then trains a decoder on nothing.
    pc = os.path.join(a.cache_dir, f"pairs_{tag}_{a.pairs}_{a.seed}.npz")
    if os.path.exists(pc):
        pcd = np.load(pc)
        ZT, XT = pcd["ZT"], pcd["XT"]
        N = len(ZT)
        print(f"pairs: {N:,} (cache hit {os.path.basename(pc)})", flush=True)
    else:
        keep_p = ~x_hold[xs]                   # per-P pixel eligibility
        train_t = np.where(~t_hold)[0]
        if len(train_t) == 0:
            raise SystemExit(
                f"no train bins: every one of {T} timesteps is in the "
                f"holdout years {sorted(hold_years)}. Refusing.")
        per_t = a.pairs // len(train_t)
        if per_t < 1:
            raise SystemExit(
                f"--pairs {a.pairs:,} over {len(train_t):,} train bins is "
                f"{a.pairs / len(train_t):.3f} pixels per bin, which floors to "
                f"0 and would gather an EMPTY training set. Raise --pairs to "
                f"at least {len(train_t):,} (one pixel per bin); the pentad "
                f"axis has ~6x family 3's bin count for the same years.")
        print(f"pairs: {len(train_t):,} train bins x {per_t:,} pixels",
              flush=True)
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

    # CUDA when present (the GPU boxes; ~1 min instead of ~65 in the sandbox,
    # measured 2026-08-13 when three sandbox attempts died to ~2h container
    # restarts mid-train). The pair arrays stay in host memory — only the
    # per-batch slice crosses, same policy as embed_everything.
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dec = MultiDec(ck["d_z"], C, a.hidden, a.layers).to(dev)
    n_par = sum(p.numel() for p in dec.parameters())
    print(f"decoder {a.hidden}x{a.layers}: {n_par:,} params (on {dev})",
          flush=True)
    Zv, Xv, Mv = Zv.to(dev), Xv.to(dev), Mv.to(dev)
    opt = torch.optim.AdamW(dec.parameters(), lr=a.lr, weight_decay=1e-4)
    import math
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda e: 0.5 * (1 + math.cos(math.pi * min(e, a.steps) / a.steps)))

    best_val, best_state, t0 = float("inf"), None, time.time()
    for s in range(1, a.steps + 1):
        idx = ti[np.random.randint(0, len(ti), a.batch)]
        z = torch.from_numpy(ZT[idx].astype(np.float32)).to(dev)
        x = torch.from_numpy(XT[idx].astype(np.float32)).to(dev)
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
                # keep the snapshot on CPU so the saved .pt loads anywhere
                best_state = {k: v.detach().cpu().clone()
                              for k, v in dec.state_dict().items()}
            print(f"  step {s:>5}/{a.steps}  train {float(loss):.4f}  "
                  f"val {vl:.4f}  ({time.time()-t0:.0f}s){tag}", flush=True)
    dec.load_state_dict(best_state)
    # persist the trained decoder next to its JSON — the sweep's first pass
    # threw the weights away, which blocked the decoded-fields transport
    # probe until a retrain
    torch.save({"hidden": a.hidden, "layers": a.layers, "d_z": ck["d_z"],
                "C": C, "model": best_state, "best_val_mse": best_val},
               a.out.replace(".json", ".pt"))

    # ---- score with the exact E-019a section audit ------------------------
    with torch.no_grad():
        pred_sec = np.stack(
            [dec(torch.from_numpy(Z_sec[t]).to(dev)).cpu().numpy()
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
        "bottleneck": bspec, "x_dtype": str(Xm.dtype),
        "splits": {
            "train": score(truth_sec, pred_sec, obs_sec,
                           sel_train_t, sel_train_x, "train"),
            "heldout_months": score(truth_sec, pred_sec, obs_sec,
                                    sel_hold_t, sel_train_x, "hold-t"),
            "heldout_lons": score(truth_sec, pred_sec, obs_sec,
                                  np.arange(T), sel_hold_x, "hold-x"),
        },
    }
    # The E-049 falsifier's axis, ADDITIVE (`splits` above is unchanged). This
    # is the DECODER-CEILING rung, so this block is the one the verdict is
    # read from.
    res["argo_split"] = argo_split_block(
        truth_sec, pred_sec, obs_sec, chan,
        {"train": (sel_train_t, sel_train_x),
         "heldout_months": (sel_hold_t, sel_train_x),
         "heldout_lons": (np.arange(T), sel_hold_x)})
    os.makedirs(os.path.dirname(os.path.abspath(a.out)) or ".", exist_ok=True)
    with open(a.out, "w") as f:
        json.dump(res, f, indent=1)
    tr = res["splits"]["train"]
    rs = [v["r"] for v in tr.values() if np.isfinite(v["r"])]
    # The deep-channel filter, CHECKED against family-4-r2 rather than assumed
    # (§0.1). `ml/build_family4.py` imports `build_family3.CHANS` rather than
    # restating it, so both tensors spell these `rg_t900`..`rg_t1900` /
    # `rg_s900`..`rg_s1900`; `nm[4:]` is the pressure in dbar because `rg_t`
    # and `rg_s` are both four characters. Twelve channels match at r1 and at
    # r2 (r2 appends `sst`, which is not one). What DID need fixing is the
    # summary line: `np.mean([])` is nan with a RuntimeWarning, so a tensor
    # whose deep channels were all too sparse to score printed `nan` as if it
    # were a measurement (§5.22).
    deep = [i for i, nm in enumerate(chan)
            if nm.startswith(("rg_t", "rg_s")) and nm[4:].isdigit()
            and int(nm[4:]) >= 900]
    dr = [tr[c]["r"] for c in deep if c in tr]
    mr = f"{np.mean(rs):.4f}" if rs else "n/a (no scoreable channel)"
    md = (f"{np.mean(dr):.4f} over {len(dr)}/{len(deep)}" if dr else
          f"n/a ({len(deep)} deep channels, none scoreable)")
    print(f"\n== {a.hidden}x{a.layers} ({n_par/1e6:.1f}M) ==  {bottleneck_line(bspec)}")
    print(f"mean r {mr} · deep-channel mean r {md}")
    print_argo_summary(res["argo_split"], chan)
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()

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

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model import PixelMAE, LazyPixels, codec_from_ckpt

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
    ap.add_argument("--data", default=os.path.join(HERE, "cache", "na_pixels.npz"),
                    help="tensor npz (family-3 runs pass family3_na025.npz)")
    a = ap.parse_args()

    ck = torch.load(os.path.join(HERE, "runs", a.run, "pixelmae.pt"),
                    map_location="cpu", weights_only=False)
    # load_tensor == np.load for every single-file npz (families 2/3/4); for
    # family 5's sidecar layout it MEMORY-MAPS X, because 165.6 GB cannot be
    # decompressed on any box we can rent (ml/tensor_io.py). The old
    # `np.load(...)["X"].copy()` could not open family 5 at all, and on a
    # single-file npz the .copy() held the decompressed tensor TWICE — the
    # residual #390 recorded when this script was OOM-killed on the 33 GB
    # pentad tensor even after the LazyPixels work below.
    from tensor_io import load_tensor
    d = load_tensor(a.data)
    X = d["X"]
    months = [str(m) for m in d["months"]]
    lats, lons = d["lats"], d["lons"]
    T, H, W, C = X.shape
    hold_years = set(ck["args"]["holdout_years"].split(","))
    t_hold = np.array([m[:4] in hold_years for m in months])
    lo, hi = (float(v) for v in ck["args"]["holdout_lon"].split(","))
    x_hold = (lons >= lo) & (lons < hi)
    moy = np.array([int(m[5:7]) - 1 for m in months])

    if a.anomaly:
        # THE anomaly transform, and there is exactly one of it. What stood
        # here was a hand-inlined FOURTH copy (train.py had the second until
        # 2026-08-17, temporal.py the third), frozen at the pre-2026-08-17
        # shape, and it carried three defects the canonical one does not:
        #
        #   1. `v.std()` with no `dtype=np.float64`. numpy upcasts the
        #      accumulator for np.mean on float16 but NOT for np.std/np.var.
        #      The z-score sums ~204M squared residuals; in float16 that
        #      passes 65504, returns inf, and (X - mu) / (inf + 1e-6) is
        #      EXACTLY 0.0 — every dynamic channel silently zero, with every
        #      downstream correlation still finite and plausible. Families 4
        #      and 5 are float16. Family 3 is float32 and never reached the
        #      limit, which is the only reason this copy never returned a
        #      wrong number.
        #   2. ~249 full-extent strided traversals of X, ~41 TB of physical
        #      read at family 5's 165.6 GB on a 64 GB box (run #389: seven
        #      hours in the equivalent code, GPU at 0%). Canonical: 6.0.
        #   3. `clim[moy][..., c]` — the form trainprobe's own comment warns
        #      against. It materialises the whole [T,H,W,C] fancy index and
        #      throws all but one channel away, once per dynamic channel
        #      (2.4 GB per channel at pentad). temporal.py's copy used the
        #      cheap `clim[moy, :, :, c]`; this one never got that fix.
        #
        # The import is LAZY because trainprobe imports this module.
        from trainprobe import anomaly_transform
        if isinstance(X, np.memmap) and not X.flags.writeable:
            # Sidecar tensor (family 5): anomaly_transform writes into X in
            # place, and the canonical map must never take those writes — it
            # would leave an anomaly-space tensor where a state-space one is
            # documented, and the next reader would z-score it again with
            # nothing to say so. A scratch copy is disk, not RAM (tensor_io
            # docstring). Only on THIS branch: the state-space path never
            # writes X, so it needs no second 166 GB. X must also STAY ALIVE
            # past here — LazyPixels below uses it as its buffer.
            from tensor_io import writable_copy
            scratch = a.data[:-4] + "_seqprobe_scratch.npy"
            X = writable_copy(X, scratch, verbose=False)
            import atexit
            atexit.register(lambda q=scratch: os.path.exists(q) and os.remove(q))
        X, _dynamic = anomaly_transform(X, moy, t_hold, x_hold)

    model = codec_from_ckpt(ck, C)
    model.load_state_dict(ck["model"])
    model.eval()
    # This one builds its own encoder inputs rather than going through
    # embed_everything, so the inputs move with the model and z comes back.
    _dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(_dev)

    ctx_all = np.stack([np.sin(2 * np.pi * moy / 12), np.cos(2 * np.pi * moy / 12)], 1)
    # Derived PER BATCH, not materialised. Eagerly these two were a full float
    # copy PLUS a full bool on top of X — 49.7 GB at pentad, 248 GB at daily —
    # allocated right after the anomaly transform, which is where run #388
    # OOM-killed this script. LazyPixels (ml/model.py) applies the same numpy
    # functions to the same elements after the index instead of before, so the
    # embedding is bit-identical; `X` must stay alive, since it is the buffer.
    #
    # X KEEPS ITS NaNs FROM HERE ON, deliberately. LazyPixels(X) does the
    # nan_to_num per batch, so filling X in place would leave
    # LazyPixels(X, obs=True) reading isfinite() over an all-finite array — the
    # observation mask would silently become all-True and every land cell would
    # enter the encoder as an observed 0.0. The section mask below (`ocean_row`)
    # reads OBS and depends on this too.
    Xt = LazyPixels(X)
    OBS = LazyPixels(X, obs=True)

    # The section is the pixels observed at ANY time, not the ones observed in
    # the FIRST month. Those are different sets the moment a channel starts
    # later than the tensor does, and family-3's channel 0 is cur_speed
    # (GLORYS, 1993-01) against a tensor that starts 1982-01 — so the old test
    # `isfinite(d["X"][0, sec_y, :, 0])` selected ZERO pixels, `z.mean(0)` over
    # an empty section returned NaN for every month, and every correlation in
    # this file came out NaN. The seasonal floor stayed finite because it never
    # touches the embedding, which is exactly why the failure read as a probe
    # bug rather than a masking bug. This is the all-NaN probe_sequence.json
    # reported by #101 and again by #116.
    #
    # Use the mask temporal.py and probe_head.py use, so the sequence probe
    # scores the SAME section the rest of the ladder does and the rungs stay
    # comparable.
    from temporal import RAPID_LON
    sec_y = int(np.argmin(np.abs(lats - 26.5)))
    ocean_row = OBS[:, sec_y, :, 0].any(axis=0).numpy()
    sec_x = np.where(ocean_row
                     & (lons >= RAPID_LON[0]) & (lons <= RAPID_LON[1]))[0]
    if len(sec_x) == 0:
        sys.exit(f"empty 26.5N section at lat {lats[sec_y]:.2f} — refusing to "
                 f"embed nothing (this is what produced all-NaN output before)")
    print(f"section: {len(sec_x)} pixels at {lats[sec_y]:.2f}N")

    print(f"embedding the 26.5N section ({a.run}, anomaly={a.anomaly}) …")
    emb = np.zeros((T, ck["d_z"]), dtype=np.float32)
    with torch.no_grad():
        for t in range(T):
            n = len(sec_x)
            ctx = np.concatenate([np.tile(ctx_all[t], (n, 1)),
                                  (np.full(n, lats[sec_y]) / 90)[:, None],
                                  (lons[sec_x] / 180)[:, None]], 1)
            if getattr(model, "patch", 1) > 1:
                from model import gather_px
                tt = torch.full((n,), t, dtype=torch.long)
                v, o = gather_px(Xt, OBS, tt,
                                 torch.full((n,), sec_y, dtype=torch.long),
                                 torch.as_tensor(sec_x), model.patch)
                z = model.encode(v.to(_dev), o.to(_dev),
                                 torch.zeros(n, C, dtype=torch.bool, device=_dev),
                                 torch.as_tensor(ctx, dtype=torch.float32).to(_dev))
            else:
                z = model.encode(Xt[t, sec_y, sec_x].to(_dev),
                                 OBS[t, sec_y, sec_x].to(_dev),
                                 torch.zeros(n, C, dtype=torch.bool, device=_dev),
                                 torch.as_tensor(ctx, dtype=torch.float32).to(_dev))
            emb[t] = z.mean(0).cpu().numpy()

    # Refuse rather than write NaN into a results file. A probe that reports
    # NaN looks like a broken probe; a probe that stops names the real fault.
    if not np.isfinite(emb).all():
        sys.exit(f"{int((~np.isfinite(emb)).sum())} non-finite values in the "
                 f"section embedding — every downstream correlation would be "
                 f"NaN, so nothing is written")

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

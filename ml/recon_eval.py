"""Copy-reconstruction audit (E-019a): how close is encode→decode to identity?

Chris, 2026-08-12: "We take some input. We encode it. We decode it, and then
we look how far away it is, and it should be almost identical." This script
measures exactly that, for the embedding the whole programme consumes: each
section pixel-month is encoded at FULL VISIBILITY (mask = none — the same
call temporal.py's embed_everything makes when it builds Z for stage 2 and
for the 0.631 transport probe), then the decoder is queried for all C
channels at offset (0,0,0) and compared to the true standardized values.

Why this is the right first measurement for the decoder programme:
  · The training loss reconstructs MASKED channels (weight 1.0) and only
    incidentally the visible ones (weight 0.1) — so nobody has ever measured
    how faithful the full-visibility round trip is, and loss_rec (~0.09 at
    the end of run-62) is a masked-dominated Huber mix, not this number.
  · If the round trip already reads r ≈ 0.99 everywhere, the 0.631 probe
    ceiling is NOT a bottleneck-lossiness story and decoder work will not
    move it (that is this audit's falsifier). If specific channels lose
    real variance, those channels name what the bottleneck discards, and
    d_z / decoder capacity / loss weighting become the levers.

Protocol notes:
  · The encode path IS temporal.embed_everything — not a reimplementation.
    A 3-row latitude slab around the section keeps gather_px's behaviour
    bit-identical for interior rows (dy = ±1 stays in bounds, lon wraps on
    the full W); the slab is standardized by a STREAMING replica of
    temporal.py's anomaly transform (the in-RAM original needs > 11 GB),
    and the replica is verified against the exact in-RAM recipe on two
    full channels before anything is scored (--skip-verify to bypass).
  · Scores are split three ways, matching the codec's own blocked holdout:
    train (train months, non-holdout lons) · heldout months (2009/2017/2023
    by default, from the checkpoint args) · heldout lon block (-45,-25).
    A large train-vs-holdout fidelity gap = the decoder memorises rather
    than compresses; similar numbers = the round trip generalises.
  · Standardized units make the numbers legible: RMSE² ≈ fraction of the
    channel's variance the round trip loses; r is the same story scale-free.
  · The pooled view (section-mean of recon vs section-mean of truth, per
    month) is reported separately: the transport probe reads pooled
    embeddings, and pooling can cancel per-pixel error.

Usage (sandbox or box; CPU is fine for the 265-pixel section):
  python3 ml/recon_eval.py --x ml/cache/family3_X.npy \
      --npz ml/cache/family3_na025.npz \
      --ckpt ml/cache/f3_anchor41M__pixelmae.pt \
      --out ml/runs/recon_audit/recon_eval.json
"""
import argparse
import json
import os
import sys
import time
import warnings

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from model import codec_from_ckpt                     # noqa: E402
from temporal import embed_everything, section_of     # noqa: E402

RAPID_LAT, RAPID_LON = 26.5, (-80.0, -13.0)


# ---------------------------------------------------------------------------
# Streaming replica of temporal.py's anomaly transform (lines ~745-762).
# The original loads X whole (10.9 GB for family3) and standardizes in RAM;
# this computes the same clim / mean / std with chunked passes over a
# memmap, then materialises only a [T, 3, W, C] latitude slab.
# ---------------------------------------------------------------------------

def stream_stats(Xm, moy, t_hold, x_hold, chunk=8):
    """(clim [12,H,W,C], dynamic [c...], mean_c, std_c) — chunked over T."""
    T, H, W, C = Xm.shape
    s = np.zeros((12, H, W, C), np.float32)
    n = np.zeros((12, H, W, C), np.uint8)          # ≤ 43 train months per moy
    sp_sum = np.zeros((T, C), np.float64)          # spatial nansum per month
    sp_cnt = np.zeros((T, C), np.int64)
    for t0 in range(0, T, chunk):
        t1 = min(t0 + chunk, T)
        xb = np.asarray(Xm[t0:t1])                 # [c,H,W,C]
        fin = np.isfinite(xb)
        xb0 = np.where(fin, xb, 0.0)
        sp_sum[t0:t1] = xb0.sum(axis=(1, 2))
        sp_cnt[t0:t1] = fin.sum(axis=(1, 2))
        for i, t in enumerate(range(t0, t1)):
            if t_hold[t]:
                continue                           # clim from train years only
            m = moy[t]
            s[m] += xb0[i]
            n[m] += fin[i]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        clim = s / n                               # nan where never observed
        sp_mean = np.where(sp_cnt > 0, sp_sum / sp_cnt, np.nan)
    # dynamic test, identical to temporal.py: nanstd of the monthly spatial
    # nanmean series (all months) > 1e-6
    dyn = [c for c in range(C)
           if np.nanstd(sp_mean[:, c]) > 1e-6]
    # pass 2: per-channel mean/std of the anomaly over train months and
    # non-holdout longitudes (temporal.py's `v`)
    acc = np.zeros((C, 3), np.float64)             # sum, sumsq, count
    keep_x = ~x_hold                               # [W]
    for t0 in range(0, T, chunk):
        t1 = min(t0 + chunk, T)
        sel = [t for t in range(t0, t1) if not t_hold[t]]
        if not sel:
            continue
        xb = np.asarray(Xm[sel])                   # [s,H,W,C]
        for c in dyn:
            d = xb[..., c] - clim[moy[sel], :, :, c]
            d = d[:, :, keep_x]
            f = np.isfinite(d)
            acc[c, 0] += d[f].sum(dtype=np.float64)
            acc[c, 1] += (d[f].astype(np.float64) ** 2).sum()
            acc[c, 2] += f.sum()
    mean_c = np.zeros(C, np.float32)
    std_c = np.ones(C, np.float32)
    for c in dyn:
        cnt = acc[c, 2]
        mu = acc[c, 0] / cnt
        var = acc[c, 1] / cnt - mu * mu
        mean_c[c], std_c[c] = mu, np.sqrt(max(var, 0.0))
    return clim, dyn, mean_c, std_c


def build_slab(Xm, rows, moy, clim, dyn, mean_c, std_c):
    """Standardized anomaly slab [T, len(rows), W, C] + its finite mask."""
    T, H, W, C = Xm.shape
    slab = np.asarray(Xm[:, rows]).astype(np.float32)      # [T,3,W,C]
    obs = np.isfinite(slab)
    # index the slab rows FIRST: clim[moy] would materialise [T,H,W,C]
    # (10.9 GB) — the exact fancy-index trap temporal.py documents.
    clim_slab = clim[:, rows]                              # [12,3,W,C]
    for c in dyn:
        slab[..., c] = ((slab[..., c] - clim_slab[moy][..., c]
                         - mean_c[c]) / (std_c[c] + 1e-6))
    slab[~obs] = np.nan
    return slab, obs


def verify_streaming(Xm, moy, t_hold, x_hold, clim, dyn, mean_c, std_c,
                     rows, slab, channels):
    """Replay temporal.py's exact in-RAM recipe on full single channels and
    assert the slab matches. A preprocessing mismatch here would silently
    invalidate every downstream number — this is the audit auditing itself."""
    for c in channels:
        xc = np.asarray(Xm[..., c]).astype(np.float32)     # [T,H,W] ~279 MB
        cl = np.full((12,) + xc.shape[1:], np.nan, np.float32)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            for m in range(12):
                cl[m] = np.nanmean(xc[(moy == m) & ~t_hold], axis=0)
        if c in dyn:
            xc = xc - cl[moy]
            v = xc[np.isfinite(xc) & ~t_hold[:, None, None]
                   & ~x_hold[None, None, :]]
            xc = (xc - v.mean()) / (v.std() + 1e-6)
        ref = xc[:, rows]
        got = slab[..., c]
        f = np.isfinite(ref)
        assert (np.isfinite(got) == f).all(), f"chan {c}: finite mask differs"
        err = np.abs(ref[f] - got[f]).max() if f.any() else 0.0
        assert err < 5e-4, f"chan {c}: streaming vs in-RAM recipe |Δ|={err}"
        print(f"  verify chan {c}: max |Δ| = {err:.2e} ✓", flush=True)


# ---------------------------------------------------------------------------


def score(truth, pred, obs, sel_t, sel_x, label):
    """Per-channel r / RMSE over (t in sel_t) × (px in sel_x), obs only."""
    C = truth.shape[-1]
    out = {}
    tt = truth[np.ix_(sel_t, sel_x)]     # [t,p,C]
    pp = pred[np.ix_(sel_t, sel_x)]
    oo = obs[np.ix_(sel_t, sel_x)]
    for c in range(C):
        m = oo[..., c]
        if m.sum() < 30:
            continue
        a, b = tt[..., c][m], pp[..., c][m]
        r = float(np.corrcoef(a, b)[0, 1]) if a.std() > 1e-9 else float("nan")
        out[c] = {"r": round(r, 4),
                  "rmse": round(float(np.sqrt(((a - b) ** 2).mean())), 4),
                  "n": int(m.sum())}
    return out


def score_pooled(truth, pred, obs, sel_t, sel_x):
    """Section-mean recon vs section-mean truth, per channel, over months."""
    C = truth.shape[-1]
    out = {}
    for c in range(C):
        tt = truth[:, sel_x, c]
        pp = pred[:, sel_x, c]
        oo = obs[:, sel_x, c]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            tm = np.where(oo, tt, np.nan)
            pm = np.where(oo, pp, np.nan)
            a = np.nanmean(tm[sel_t], axis=1)
            b = np.nanmean(pm[sel_t], axis=1)
        f = np.isfinite(a) & np.isfinite(b)
        if f.sum() < 24 or a[f].std() < 1e-9:
            continue
        out[c] = {"r": round(float(np.corrcoef(a[f], b[f])[0, 1]), 4),
                  "rmse": round(float(np.sqrt(((a[f] - b[f]) ** 2).mean())), 4),
                  "n": int(f.sum())}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--x", required=True, help="extracted X.npy (memmapped)")
    ap.add_argument("--npz", required=True, help="tensor npz for small members")
    ap.add_argument("--ckpt", required=True, help="codec checkpoint")
    ap.add_argument("--out", required=True)
    ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--skip-verify", action="store_true")
    a = ap.parse_args()

    ck = torch.load(a.ckpt, map_location="cpu", weights_only=False)
    d = np.load(a.npz)
    months = [str(m) for m in d["months"]]
    lats, lons = d["lats"], d["lons"]
    chan = [str(c) for c in d["chan"]]
    Xm = np.load(a.x, mmap_mode="r")
    T, H, W, C = Xm.shape
    assert len(chan) == C

    moy = np.array([int(m[5:7]) - 1 for m in months])
    hold_years = set(ck["args"]["holdout_years"].split(","))
    t_hold = np.array([m[:4] in hold_years for m in months])
    lo, hi = (float(v) for v in ck["args"]["holdout_lon"].split(","))
    x_hold = (lons >= lo) & (lons < hi)

    print(f"tensor [T={T} H={H} W={W} C={C}] · holdout years "
          f"{sorted(hold_years)} · holdout lons [{lo},{hi})", flush=True)

    t0 = time.time()
    clim, dyn, mean_c, std_c = stream_stats(Xm, moy, t_hold, x_hold)
    print(f"streaming stats: {len(dyn)}/{C} dynamic channels "
          f"({time.time()-t0:.0f}s)", flush=True)

    sec_y = int(np.argmin(np.abs(lats - RAPID_LAT)))
    rows = [sec_y - 1, sec_y, sec_y + 1]
    slab, obs = build_slab(Xm, rows, moy, clim, dyn, mean_c, std_c)

    if not a.skip_verify:
        # one dynamic channel (0 = cur_speed) and one deep RG channel
        deep = next((c for c, nm in enumerate(chan) if nm.startswith("rg_t")
                     and c in dyn), dyn[-1])
        verify_streaming(Xm, moy, t_hold, x_hold, clim, dyn, mean_c, std_c,
                         rows, slab, [0, deep])

    # section pixels exactly as the probe ladder selects them: ocean = channel
    # 0 observed at ANY month (temporal.py's mask), then section_of over the
    # slab's 3-row coordinate frame (its center row is nearest 26.5N, so
    # section_of resolves sec_y == 1 by the same argmin it runs in production)
    ocean_row = obs[:, 1, :, 0].any(axis=0)        # slab row 1 = sec_y
    xs_all = np.where(ocean_row)[0]
    _, sel = section_of(lats[rows], lons, np.ones(len(xs_all), dtype=int),
                        xs_all, RAPID_LAT, *RAPID_LON)
    xs_sec = xs_all[sel]
    print(f"section: {len(xs_sec)} pixels at lat {lats[sec_y]:.2f}", flush=True)

    # ---- encode through the production path -------------------------------
    codec = codec_from_ckpt(ck, C)
    codec.load_state_dict(ck["model"])
    codec.eval()
    ctx_all = np.stack([np.sin(2 * np.pi * moy / 12),
                        np.cos(2 * np.pi * moy / 12)], 1)
    Xt = torch.from_numpy(np.nan_to_num(slab, nan=0.0))
    OBS = torch.from_numpy(obs)
    t0 = time.time()
    Z, _ = embed_everything(codec, Xt, OBS, ctx_all, lats[rows], lons,
                            np.full(len(xs_sec), 1), xs_sec, ck["d_z"],
                            cache_path=None, batch=a.batch)
    print(f"encoded [T={T} P={len(xs_sec)}] in {time.time()-t0:.0f}s", flush=True)

    # ---- decode every channel at offset 0 ---------------------------------
    P = len(xs_sec)
    pred = np.zeros((T, P, C), np.float32)
    qc = torch.arange(C)[None, :]
    off0 = torch.zeros(1, C, 3, dtype=torch.long)
    with torch.no_grad():
        for t in range(T):
            zb = torch.from_numpy(Z[t]).float()
            pred[t] = codec.query(zb, qc.expand(P, -1),
                                  off0.expand(P, -1, -1)).numpy()

    truth = slab[:, 1][:, xs_sec]                  # [T,P,C] center row
    obs_c = obs[:, 1][:, xs_sec]
    truth0 = np.nan_to_num(truth, nan=0.0)

    px_hold = x_hold[xs_sec]
    sel_train_t = np.where(~t_hold)[0]
    sel_hold_t = np.where(t_hold)[0]
    sel_train_x = np.where(~px_hold)[0]
    sel_hold_x = np.where(px_hold)[0]

    res = {
        "doc": "copy-reconstruction audit: full-visibility encode -> decode "
               "all channels at offset 0, vs standardized truth; std units "
               "(rmse^2 ~ fraction of channel variance lost)",
        "ckpt": os.path.basename(a.ckpt),
        "d_z": int(ck["d_z"]), "C": C, "T": T, "P": P,
        "chan": chan, "dynamic": dyn,
        "splits": {
            "train": score(truth0, pred, obs_c, sel_train_t, sel_train_x, "train"),
            "heldout_months": score(truth0, pred, obs_c, sel_hold_t,
                                    sel_train_x, "hold-t"),
            "heldout_lons": score(truth0, pred, obs_c, np.arange(T),
                                  sel_hold_x, "hold-x"),
        },
        "pooled": {
            "train": score_pooled(truth0, pred, obs_c, sel_train_t,
                                  np.arange(P)),
            "heldout_months": score_pooled(truth0, pred, obs_c, sel_hold_t,
                                           np.arange(P)),
        },
    }
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with open(a.out, "w") as f:
        json.dump(res, f, indent=1)

    # readable summary
    tr = res["splits"]["train"]
    hm = res["splits"]["heldout_months"]
    hx = res["splits"]["heldout_lons"]
    rs = [v["r"] for v in tr.values() if np.isfinite(v["r"])]
    print(f"\n== copy reconstruction, {os.path.basename(a.ckpt)} "
          f"(d_z={ck['d_z']}) ==")
    print(f"mean r over {len(rs)} scoreable channels (train): "
          f"{np.mean(rs):.3f}  · worst 5:")
    for c, v in sorted(tr.items(), key=lambda kv: kv[1]["r"])[:5]:
        print(f"  {chan[c]:<12} r={v['r']:.3f} rmse={v['rmse']:.3f}")
    print(f"{'chan':<12} {'train r':>8} {'holdT r':>8} {'holdX r':>8} "
          f"{'train rmse':>10}")
    for c in range(C):
        if c not in tr:
            continue
        print(f"{chan[c]:<12} {tr[c]['r']:>8.3f} "
              f"{hm.get(c, {}).get('r', float('nan')):>8.3f} "
              f"{hx.get(c, {}).get('r', float('nan')):>8.3f} "
              f"{tr[c]['rmse']:>10.3f}")
    print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()

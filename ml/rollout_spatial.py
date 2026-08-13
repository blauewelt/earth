#!/usr/bin/env python3
"""E-022 evaluator: FULL-WINDOW autoregressive rollout for stencil heads.

Why this exists (plan ml/plans/E022_spatial_coupling.md §6): a stencil head's
step-t+1 at pixel p reads p's neighbours at t, so a correct T-step roll of any
region needs that region plus a T×reach halo. The clean solution is to roll
ALL window ocean pixels (84,405 at family3) — which is also EXACT for the
stencil-1 baseline, since the per-pixel model factorises over pixels. Chris's
spec: "the eval needs to involve rolling forward all pixels that contribute
to the AMOC current (from the Gulf of Mexico to northern Europe, and back)".

GPU-native from day one: ml/rollout.py moved the codec back to CPU for its
head loop and burned ~$1.9 of rented 4090 across #211/#217 doing transformer
forwards on cores. Here the head, the codec, the rolling window state
([P, K, d_z] ≈ 519 MB) and the stencil gather all live on the device.

Protocol = rollout.py's, verbatim where they overlap (staggered starts into
holdout years, true-context init, MSSS vs climatology / persistence / damped
persistence on observed cells, truefit AMOC ridge) — because the VALIDATION
GATE below demands the two agree before any spatial head is scored. Scoring
runs over three nested pixel scopes per horizon:

  gate     — rollout.py's exact subset (default_rng(0).choice(P, 600) ∪ the
             RAPID section), where #217's numbers live. The gate compares
             THIS scope against #217; a mismatch means the evaluator is
             wrong and the script refuses to score anything past it.
  corridor — the AMOC corridor, data-derived, not hand-drawn: ocean pixels
             whose train-month mean cur_speed (channel 0) is ≥ the 75th
             percentile over window ocean, dilated 2 cells, ∪ the RAPID
             section. The headline scope (§3.6).
  window   — all window ocean pixels.

Plus, per head: AMOC truefit r in horizon bands h1-3/h4-6/h7-12; a long
hindcast (context ends --long-start, default 2004-12, rolled 240 months
across the RAPID record, median trajectory only — E-021b owns ensembles);
and a future roll from the record's end.

Usage (box, GPU):
  python3 ml/rollout_spatial.py --x ml/cache/family3_X.npy \
      --npz-small ml/cache/f3_dec_small.npz --z ml/cache/Z_....npy \
      --ckpt ml/cache/f3_anchor41M__pixelmae.pt \
      --heads ml/runs/heads/e017_u1_s0.pt ml/runs/heads/e022s9_u1_s0.pt ... \
      --out ml/runs/actions/rollout_spatial.json
"""
import argparse
import json
import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from model import codec_from_ckpt, gather_px                    # noqa: E402
from recon_eval import stream_stats, build_slab                 # noqa: E402
from temporal import (TemporalTransformer, build_stencil,       # noqa: E402
                      embed_everything, rapid_section)
from project_amoc import fit_ridge                              # noqa: E402

# The gate reference: #217 (ml-metrics probes-217.json), head u1_s0 —
# e017_u1_s0 rolled by ml/rollout.py over its 600∪section subset. NOTE the
# metric's name: the 0.643–0.645 "AUC" band quoted everywhere for E-017 is
# rollout.py's `horizon_auc` = mean MSSS vs CLIMATOLOGY over h=1..12 (the
# plan's §1 says "AUC(msss_damped)" — checked against the archive 2026-08-13,
# it is msss_clim; the damped mean is 0.619). Gate on what #217 actually is.
GATE_HEAD = "e017_u1_s0"
GATE_REF = {"auc": 0.643,
            "bands": {"h1-3": 0.470, "h4-6": 0.375, "h7-12": 0.492}}
GATE_TOL = 0.0101          # plan §6.5: ±0.01 (float-boundary slack only)

BANDS = (("h1-3", (1, 2, 3)), ("h4-6", (4, 5, 6)),
         ("h7-12", tuple(range(7, 13))))


def dilate8(m, iters):
    """Binary dilation, 3×3 square structuring element. NOT np.roll — roll
    wraps, and this window's longitudes don't."""
    for _ in range(iters):
        p = np.zeros((m.shape[0] + 2, m.shape[1] + 2), bool)
        p[1:-1, 1:-1] = m
        m = (p[:-2, :-2] | p[:-2, 1:-1] | p[:-2, 2:]
             | p[1:-1, :-2] | p[1:-1, 1:-1] | p[1:-1, 2:]
             | p[2:, :-2] | p[2:, 1:-1] | p[2:, 2:])
    return m


def corridor_pixels(Xm, ocean, ys, xs, t_hold, sec_sel, pctl, dilate):
    """Bool [P]: the AMOC corridor (plan §3.6). Mean of RAW cur_speed
    (channel 0) over TRAIN months, observed samples only; threshold at the
    `pctl` percentile of that mean over window ocean; dilate; ∪ section."""
    s = np.zeros(ocean.shape, np.float64)
    n = np.zeros(ocean.shape, np.int64)
    tr = np.where(~t_hold)[0]
    for i0 in range(0, len(tr), 16):
        xb = np.asarray(Xm[tr[i0:i0 + 16], :, :, 0])
        f = np.isfinite(xb)
        s += np.where(f, xb, 0.0).sum(0)
        n += f.sum(0)
    with np.errstate(invalid="ignore"):
        mean_sp = np.where(n > 0, s / np.maximum(n, 1), np.nan)
    vals = mean_sp[ys, xs]
    thr = float(np.percentile(vals[np.isfinite(vals)], pctl))
    core = np.zeros(ocean.shape, bool)
    core[ys, xs] = np.where(np.isfinite(vals), vals, -np.inf) >= thr
    mask2d = dilate8(core, dilate) & ocean
    cp = mask2d[ys, xs]
    cp[sec_sel] = True
    return cp, thr


def export_mask(path, lats, lons, ocean, ys, xs, corridor, gate_mask,
                sec_sel, sec_y, corridor_def, months, x_name):
    """Write the eval's OWN pixel sets as a baked categorical grid the globe
    app can draw (root CLAUDE.md §2 `classGrid` + `packed` format, row 0 =
    south). Chris, 2026-08-13: *"add a layer to the globe visualiser to see
    which pixels will all be rolled forward in the amoc eval"*.

    It is written HERE, by the evaluator, from the same `corridor_pixels`
    call the scoring uses, so the picture cannot drift from the experiment —
    a hand-drawn corridor in the frontend would be a second definition, and
    the second definition is always the one that goes stale.

    Classes are NESTED (section ⊂ corridor ⊂ rolled); each cell shows its
    most specific one:
      1 rolled   — a window ocean pixel the roll advances every step
      2 corridor — also scored as the headline AMOC corridor
      3 section  — also on the RAPID 26.5°N transport section
    Land and out-of-window cells are empty ("."), which is the honest
    answer: the model has no state there at all."""
    H, W = ocean.shape
    code = np.zeros((H, W), np.uint8)
    code[ys, xs] = 1
    code[ys[corridor], xs[corridor]] = 2
    code[ys[sec_sel], xs[sec_sel]] = 3
    # NO row flip here, unlike scripts/refresh_data.py's drivers bake: the app's
    # grid format wants row 0 = SOUTH and this tensor is already south-first
    # (lats[0] = 0.0 N, lats[-1] = 70.0 N). Flipping "to be safe" is what the
    # first version did, and it put the Gulf Stream at the latitude of the
    # Norwegian Sea — a map that still looks like a plausible ocean, which is
    # exactly why the assertion below exists rather than an eyeball check.
    assert lats[0] < lats[-1], \
        f"lats run north-first ({lats[0]} → {lats[-1]}); this writer assumes " \
        f"south-first rows and would emit a vertically mirrored grid"
    dlat = float(np.round(np.diff(lats).mean(), 6))
    dlon = float(np.round(np.diff(lons).mean(), 6))
    payload = {
        "id": "amoc-eval",
        "title": "AMOC eval: the pixels the model rolls forward",
        "units": "role",
        "source": ("earth / E-022 · ml/rollout_spatial.py over the "
                   f"{x_name} tensor"),
        "citation": (
            "Every pixel this experiment advances one month at a time. The "
            "stage-2 head predicts each pixel's next embedding; the E-022 "
            "evaluator rolls ALL window ocean pixels (a stencil head reads "
            "its neighbours, so a region roll would need a growing halo), "
            "then scores an AMOC corridor derived from the data itself: mean "
            "current speed over training months at or above the "
            f"{corridor_def['pctl']:g}th percentile, dilated "
            f"{corridor_def['dilate_cells']} cells, unioned with the RAPID "
            "26.5°N section. Not hand-drawn, and written by the scoring code."),
        "doc": ("https://github.com/blauewelt/earth/blob/main/ml/plans/"
                "E022_spatial_coupling.md"),
        "classes": [
            {"code": 1, "label": "Rolled forward", "rgb": [72, 116, 168]},
            {"code": 2, "label": "Scored: AMOC corridor", "rgb": [232, 152, 48]},
            {"code": 3, "label": "RAPID 26.5°N section", "rgb": [235, 74, 96]},
        ],
        # cell CENTRES are the tensor's lats/lons, so the bounds are half a
        # cell outside them (sampleGrid floors from west/south)
        "west": round(float(lons[0]) - dlon / 2, 6),
        "east": round(float(lons[-1]) + dlon / 2, 6),
        "south": round(float(lats[0]) - dlat / 2, 6),
        "north": round(float(lats[-1]) + dlat / 2, 6),
        "dlon": dlon, "dlat": dlat, "nx": W, "ny": H,
        "period": f"{months[0]}–{months[-1]}",
        "counts": {"rolled": int(len(ys)), "corridor": int(corridor.sum()),
                   "section": int(len(sec_sel)),
                   "gate_subset": int(gate_mask.sum())},
        "corridor_def": corridor_def,
        "section_row": {"lat": round(float(lats[sec_y]), 4),
                        "n_px": int(len(sec_sel))},
        "packed": "".join("." if v == 0 else str(v) for v in code.ravel()),
    }
    # Read the file back the way the BROWSER will (app.js sampleGrid: floor
    # from west/south), and demand the RAPID section land on RAPID's latitude.
    # An exact expected value, checked at the point of writing — the app's own
    # geometry, not this function's, so an off-by-one or a mirrored row cannot
    # leave here (ml/CLAUDE.md §4.9).
    def _probe(lat, lon):
        ix = int(np.floor((lon - payload["west"]) / payload["dlon"]))
        iy = int(np.floor((lat - payload["south"]) / payload["dlat"]))
        return payload["packed"][iy * payload["nx"] + ix]
    sec_lat = float(lats[sec_y])
    for lon in (-70.0, -40.0):
        got = _probe(sec_lat, lon)
        assert got == "3", (
            f"the {sec_lat}°N section reads '{got}' at {lon}°E when sampled "
            f"the way the app samples it — the grid is mis-oriented, refusing "
            f"to write a map that would look fine and be wrong")
    assert _probe(sec_lat + 5 * payload["dlat"], -40.0) != "3", \
        "the section smears across rows — off-by-one in the row index"

    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        json.dump(payload, f, separators=(",", ":"))
    print(f"wrote {path} ({os.path.getsize(path):,} bytes) — "
          f"{payload['counts']['rolled']:,} rolled · "
          f"{payload['counts']['corridor']:,} corridor · "
          f"{payload['counts']['section']} section", flush=True)


class StdMonths:
    """Standardized-anomaly month fields at the ocean pixels, cached.
    Exactly build_slab's per-month recipe (dyn channels de-climatologised
    and z-scored, static channels RAW — matching rollout.py's
    anomaly_transform space); returns (values nan→0 [P,C] f32, obs [P,C])."""

    def __init__(self, Xm, ys, xs, moy, clim, dyn, mean_c, std_c):
        self.Xm, self.ys, self.xs, self.moy = Xm, ys, xs, moy
        self.clim, self.dyn = clim, dyn
        self.mean_c, self.std_c = mean_c, std_c
        self.cache = {}

    def get(self, t):
        if t not in self.cache:
            x = np.asarray(self.Xm[t]).astype(np.float32)       # [H,W,C]
            obs = np.isfinite(x)
            for c in self.dyn:
                x[..., c] = ((x[..., c] - self.clim[self.moy[t], :, :, c]
                              - self.mean_c[c]) / (self.std_c[c] + 1e-6))
            x = np.where(obs, x, 0.0)
            self.cache[t] = (x[self.ys, self.xs], obs[self.ys, self.xs])
        return self.cache[t]


def ar1_train(std_m, T, t_hold, P, C):
    """Damped-persistence AR1 coefficient per (pixel, channel) — rollout.py's
    construction: lag-1 pairs whose FIRST month is a train month, both months
    observed, clip [0, 0.999], zero under 24 pairs."""
    acc = {k: np.zeros((P, C), np.float64)
           for k in ("n", "sx", "sy", "sxx", "syy", "sxy")}
    prev = None
    for t in range(T):
        v, o = std_m.get(t)
        if prev is not None and not t_hold[t - 1]:
            pv, po = prev
            m = po & o
            x0 = np.where(m, pv, 0.0)
            x1 = np.where(m, v, 0.0)
            acc["n"] += m
            acc["sx"] += x0
            acc["sy"] += x1
            acc["sxx"] += x0 * x0
            acc["syy"] += x1 * x1
            acc["sxy"] += x0 * x1
        prev = (v, o)
        if t >= 2:                       # the cache only needs a 2-month tail
            std_m.cache.pop(t - 2, None)
    n = acc["n"]
    with np.errstate(invalid="ignore", divide="ignore"):
        cov = acc["sxy"] - acc["sx"] * acc["sy"] / np.maximum(n, 1)
        v0 = acc["sxx"] - acc["sx"] ** 2 / np.maximum(n, 1)
        v1 = acc["syy"] - acc["sy"] ** 2 / np.maximum(n, 1)
        rr = cov / (np.sqrt(np.maximum(v0 * v1, 0.0)) + 1e-9)
    return np.where(n >= 24, np.clip(rr, 0, 0.999), 0.0).astype(np.float32)


def month_feats(moys, dev):
    m = np.asarray(moys)
    return torch.from_numpy(np.stack(
        [np.sin(2 * np.pi * m / 12),
         np.cos(2 * np.pi * m / 12)], 1).astype(np.float32)).to(dev)


def roll_step(model, Zwin, NBR_t, static_ctx, mfeat, chunk, amp=False):
    """One autoregressive step over ALL pixels. Zwin [P, K, d_z] on the
    device; NBR_t None (stencil 1) or [P, S]; mfeat [K, 2]. → ẑ [P, d_z]
    float32. The gather mirrors temporal.gather_stencil's layout exactly
    (slot-major over d_z, centre slot 0, missing → zeros); the zero-weight-
    equivalence test pins that layout at the model boundary. `amp` runs the
    forward under fp16 autocast — a SPEED knob whose honesty is enforced by
    the #217 gate, which scores through the identical path."""
    P = Zwin.shape[0]
    outs = []
    for i in range(0, P, chunk):
        sl = slice(i, min(i + chunk, P))
        if NBR_t is None:
            zin = Zwin[sl]
        else:
            nbr = NBR_t[sl]                                   # [n, S]
            miss = nbr < 0
            zj = Zwin[nbr.clamp(min=0)]                       # [n, S, K, dz]
            zj[miss] = 0.0
            zin = zj.permute(0, 2, 1, 3).reshape(
                zj.shape[0], Zwin.shape[1], -1)               # [n, K, S*dz]
        with torch.autocast(device_type="cuda", dtype=torch.float16,
                            enabled=amp):
            pred, _ = model(zin,
                            mfeat[None].expand(zin.shape[0], -1, -1),
                            static_ctx[sl])
        outs.append(pred[:, -1].float())
    return torch.cat(outs, 0)


def decode_all(codec, zhat, C, chunk, amp=False):
    """codec.query at every (pixel, channel), offset 0 — [P, C] numpy."""
    outs = []
    for i in range(0, zhat.shape[0], chunk):
        z = zhat[i:i + chunk]
        n = z.shape[0]
        qc = torch.arange(C, device=z.device)[None].expand(n, -1)
        off0 = torch.zeros(n, C, 3, dtype=torch.long, device=z.device)
        with torch.autocast(device_type="cuda", dtype=torch.float16,
                            enabled=amp):
            xq = codec.query(z, qc, off0)
        outs.append(xq.float().cpu().numpy())
    return np.concatenate(outs, 0)


def new_sums(H):
    return {k: np.zeros(H + 1) for k in
            ("mse_m", "mse_p", "mse_c", "mse_d", "n",
             "sxy", "sxx", "syy", "sx", "sy")}


def accumulate(su, h, xhat, v_true, v_pers, v_damp, op):
    opf = op.astype(np.float64)
    n = opf.sum()
    if n == 0:
        return
    su["mse_m"][h] += (((xhat - v_true) ** 2) * opf).sum()
    su["mse_p"][h] += (((v_pers - v_true) ** 2) * opf).sum()
    su["mse_d"][h] += (((v_damp - v_true) ** 2) * opf).sum()
    su["mse_c"][h] += ((v_true ** 2) * opf).sum()
    su["n"][h] += n
    xm = xhat * opf
    ym = v_true * opf
    su["sxy"][h] += (xm * ym).sum()
    su["sxx"][h] += (xm * xm).sum()
    su["syy"][h] += (ym * ym).sum()
    su["sx"][h] += xm.sum()
    su["sy"][h] += ym.sum()


def skill_block(su, H):
    """rollout.py's chan_skill rows + horizon_auc (mean MSSS-vs-climatology
    — the #217 'AUC'), plus the damped mean for completeness."""
    rows = []
    for h in range(1, H + 1):
        if su["n"][h] == 0:
            continue
        n_ = su["n"][h]
        mm, mp = su["mse_m"][h] / n_, su["mse_p"][h] / n_
        md, mc = su["mse_d"][h] / n_, su["mse_c"][h] / n_
        vx = su["sxx"][h] / n_ - (su["sx"][h] / n_) ** 2
        vy = su["syy"][h] / n_ - (su["sy"][h] / n_) ** 2
        cov = su["sxy"][h] / n_ - su["sx"][h] * su["sy"][h] / n_ ** 2
        acc = cov / (np.sqrt(vx * vy) + 1e-12)
        rows.append({"h": h, "n": int(n_),
                     "msss_clim": round(1 - mm / mc, 3),
                     "msss_pers": round(1 - mm / mp, 3),
                     "msss_damped": round(1 - mm / md, 3),
                     "acc": round(float(acc), 3),
                     "amp_ratio": round(float(np.sqrt(vx / (vy + 1e-12))), 3)})
    out = {"chan_skill": rows}
    if rows:
        out["horizon_auc"] = round(
            float(np.mean([r["msss_clim"] for r in rows])), 3)
        out["auc_damped"] = round(
            float(np.mean([r["msss_damped"] for r in rows])), 3)
    return out


def smooth(vals, k=18, min_valid=12):
    """Centred running mean — the house 18-month lowpass (plot_projection)."""
    v = np.asarray(vals, float)
    out = np.full(len(v), np.nan)
    for i in range(len(v)):
        w = v[max(0, i - k // 2): min(len(v), i + (k - k // 2))]
        w = w[np.isfinite(w)]
        if len(w) >= min_valid:
            out[i] = w.mean()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--x", required=True)
    ap.add_argument("--npz-small", required=True)
    ap.add_argument("--z", help="Z cache .npy (f16 [T,P,dz])")
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--heads", nargs="+")
    ap.add_argument("--out", help="rollout_spatial.json (not needed with "
                                  "--export-mask-only)")
    ap.add_argument("--export-mask",
                    help="also write the eval's pixel sets as a baked "
                         "categorical grid for the globe app "
                         "(data/amoc_eval_mask.json)")
    ap.add_argument("--export-mask-only", action="store_true",
                    help="write that grid and stop — needs no Z, no heads "
                         "and no GPU, because the masks depend only on the "
                         "tensor and the corridor recipe")
    ap.add_argument("--horizon", type=int, default=12)
    ap.add_argument("--chunk", type=int, default=8192)
    ap.add_argument("--pixels-gate", type=int, default=600)
    ap.add_argument("--corridor-pctl", type=float, default=75.0)
    ap.add_argument("--corridor-dilate", type=int, default=2)
    ap.add_argument("--long-start", default="2004-12",
                    help="context end for the long hindcast; '' skips it")
    ap.add_argument("--long-months", type=int, default=240)
    ap.add_argument("--future-months", type=int, default=240,
                    help="0 skips the future roll")
    ap.add_argument("--no-gate", action="store_true",
                    help="score without the e017_u1_s0 gate — smoke/toy ONLY")
    ap.add_argument("--amp", action="store_true",
                    help="fp16 autocast for the roll/decode forwards — the "
                         "gate decides whether the numbers survive it")
    ap.add_argument("--cache-dir", default=os.path.join(HERE, "cache"))
    a = ap.parse_args()
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if dev.type == "cuda":
        # TF32 matmuls: ~2x on Ada at ~10-bit mantissa with fp32 accumulate.
        # The 9-head R4 eval is ~2M forward tokens per roll step x ~700 steps
        # per head — at strict fp32 that is ~11 h of 4090; with TF32 it fits
        # a job_timeout. Numerically far gentler than bf16, and the #217 gate
        # (±0.01) is the check that it changed nothing that matters.
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    if a.export_mask_only and not a.export_mask:
        sys.exit("--export-mask-only needs --export-mask <path>")
    if not a.export_mask_only:
        missing = [f"--{k}" for k in ("z", "heads", "out")
                   if not getattr(a, k)]
        if missing:
            sys.exit(f"missing {', '.join(missing)} (required unless "
                     f"--export-mask-only)")
    # gate discipline up front, where it has cost nothing (ml/CLAUDE.md §0.3)
    gate_paths = [h for h in (a.heads or [])
                  if GATE_HEAD in os.path.basename(h)]
    if not gate_paths and not a.no_gate and not a.export_mask_only:
        sys.exit(f"no {GATE_HEAD} head among --heads and --no-gate not set: "
                 f"the validation gate (plan §6.5) is what makes any spatial "
                 f"number here believable — add the gate head or pass "
                 f"--no-gate (smoke only)")
    heads = gate_paths + [h for h in (a.heads or [])
                          if h not in gate_paths]

    d = np.load(a.npz_small, allow_pickle=False)
    months = [str(m) for m in d["months"]]
    lats, lons = d["lats"], d["lons"]
    T = len(months)
    moy = np.array([int(m[5:7]) - 1 for m in months])
    month_index = {m: i for i, m in enumerate(months)}
    ck = torch.load(a.ckpt, map_location="cpu", weights_only=False)
    C = len(ck["chan"])
    d_z = ck["d_z"]
    hold_years = sorted(ck["args"]["holdout_years"].split(","))
    t_hold = np.array([m[:4] in set(hold_years) for m in months])
    lo, hi = (float(v) for v in ck["args"]["holdout_lon"].split(","))
    x_hold = (lons >= lo) & (lons < hi)

    Xm = np.load(a.x, mmap_mode="r")
    Hg, Wg = Xm.shape[1], Xm.shape[2]
    om_path = os.path.join(a.cache_dir, "ocean_mask.npy")
    ocean = np.load(om_path) if os.path.exists(om_path) else None
    if ocean is None:
        ocean = np.zeros((Hg, Wg), bool)
        for t0 in range(0, T, 16):
            ocean |= np.isfinite(np.asarray(Xm[t0:t0 + 16, :, :, 0])).any(0)
        os.makedirs(a.cache_dir, exist_ok=True)
        np.save(om_path, ocean)
    ys, xs = np.where(ocean)
    P = len(ys)
    coords = np.stack([lats[ys] / 90, lons[xs] / 180], 1).astype(np.float32)
    sec_y, sec_sel = rapid_section(lats, lons, ys, xs)
    S_sec = len(sec_sel)
    print(f"window: {P} ocean px · section {S_sec} px at row {sec_y}",
          flush=True)

    Zm = None
    if not a.export_mask_only:
        Zm = np.load(a.z, mmap_mode="r")
        assert Zm.shape == (T, P, d_z), \
            f"Z {Zm.shape} != {(T, P, d_z)} — ordering mismatch, refusing"

    # ---- anomaly stats + per-month standardized fields --------------------
    st_path = os.path.join(a.cache_dir, "std_stats.npz")
    if os.path.exists(st_path):
        s_ = np.load(st_path)
        clim, dyn = s_["clim"], list(s_["dyn"])
        mean_c, std_c = s_["mean_c"], s_["std_c"]
    else:
        clim, dyn, mean_c, std_c = stream_stats(Xm, moy, t_hold, x_hold)
        np.savez(st_path, clim=clim, dyn=np.array(dyn),
                 mean_c=mean_c, std_c=std_c)
    std_m = StdMonths(Xm, ys, xs, moy, clim, dyn, mean_c, std_c)

    r1 = None
    if not a.export_mask_only:     # the mask needs no baseline arithmetic
        print("AR1 damped-persistence pass over the record ...", flush=True)
        r1 = ar1_train(std_m, T, t_hold, P, C)                 # [P, C]

    # ---- the three scopes -------------------------------------------------
    corridor, cor_thr = corridor_pixels(Xm, ocean, ys, xs, t_hold, sec_sel,
                                        a.corridor_pctl, a.corridor_dilate)
    rng = np.random.default_rng(0)               # rollout.py's exact subset
    keep = np.union1d(rng.choice(P, min(a.pixels_gate, P), replace=False),
                      sec_sel)
    gate_mask = np.zeros(P, bool)
    gate_mask[keep] = True
    scopes = (("gate", gate_mask), ("corridor", corridor),
              ("window", np.ones(P, bool)))
    corridor_def = {"pctl": a.corridor_pctl, "threshold": round(cor_thr, 4),
                    "dilate_cells": a.corridor_dilate,
                    "structuring": "3x3 square",
                    "n_px": int(corridor.sum()), "of": P,
                    "union_section": True}
    print(f"scopes: gate {int(gate_mask.sum())} px · corridor "
          f"{int(corridor.sum())} px (cur_speed ≥ p{a.corridor_pctl:g} "
          f"= {cor_thr:.3f}, dilate {a.corridor_dilate}) · window {P} px",
          flush=True)

    if a.export_mask:
        export_mask(a.export_mask, lats, lons, ocean, ys, xs, corridor,
                    gate_mask, sec_sel, sec_y, corridor_def, months,
                    os.path.basename(a.x))
        if a.export_mask_only:
            return

    # ---- codec + static identity for ALL pixels ---------------------------
    codec = codec_from_ckpt(ck, C)
    codec.load_state_dict(ck["model"])
    codec.eval().to(dev)
    x0 = np.asarray(Xm[0]).astype(np.float32)
    obs0 = np.isfinite(x0)
    for c in dyn:
        x0[..., c] = ((x0[..., c] - clim[moy[0], :, :, c]
                       - mean_c[c]) / (std_c[c] + 1e-6))
    Xt0 = torch.from_numpy(np.where(obs0, x0, 0.0)[None])      # [1,H,W,C]
    stat_obs = torch.from_numpy(obs0).clone()
    for c in dyn:
        stat_obs[..., c] = False
    zs = []
    with torch.no_grad():
        for i in range(0, P, 8192):
            sl = slice(i, min(i + 8192, P))
            n = sl.stop - sl.start
            ctx = np.concatenate([np.zeros((n, 2), np.float32),
                                  coords[sl]], 1)
            if getattr(codec, "patch", 1) > 1:
                vv, oo = gather_px(Xt0, stat_obs[None],
                                   torch.zeros(n, dtype=torch.long),
                                   torch.as_tensor(ys[sl]),
                                   torch.as_tensor(xs[sl]), codec.patch)
            else:
                vv = Xt0[0, ys[sl], xs[sl]]
                oo = stat_obs[ys[sl], xs[sl]]
            zs.append(codec.encode(
                vv.to(dev), oo.to(dev),
                torch.zeros(n, C, dtype=torch.bool, device=dev),
                torch.as_tensor(ctx).to(dev)).cpu().numpy())
    Zstat = np.concatenate(zs, 0)
    print(f"static identity encoded for {P} px", flush=True)

    # ---- verify the Z cache against a live re-encode (project_amoc's
    # guard: a silent ordering mismatch would roll beautiful nonsense) ------
    Zsec = np.asarray(Zm[:, sec_sel]).astype(np.float32)       # [T,S,dz]
    rows3 = [sec_y - 1, sec_y, sec_y + 1]
    slab, obs_sl = build_slab(Xm, rows3, moy, clim, dyn, mean_c, std_c)
    slab_t = torch.from_numpy(np.nan_to_num(slab, nan=0.0))
    obs_t = torch.from_numpy(obs_sl)
    ctx_all = np.stack([np.sin(2 * np.pi * moy / 12),
                        np.cos(2 * np.pi * moy / 12)], 1)
    rngv = np.random.default_rng(1)
    kv = rngv.choice(S_sec, min(8, S_sec), replace=False)
    sxs = xs[sec_sel]
    Zl, _ = embed_everything(codec, slab_t, obs_t, ctx_all, lats[rows3], lons,
                             np.ones(len(kv), dtype=int), sxs[kv], d_z,
                             cache_path=None, batch=64)
    for tt in (0, T // 2, T - 1):
        dmax = float(np.abs(Zl[tt] - Zsec[tt][kv]).max())
        zscale = float(np.abs(Zl[tt]).max())
        assert dmax < max(0.02, 0.005 * zscale), \
            f"Z mismatch at t={tt}: {dmax} vs scale {zscale}"
    print("Z cache verified vs live re-encode ✓", flush=True)

    # ---- transport read-out (truefit protocol, rollout.py verbatim) ------
    rapid = d["rapid"]
    ridx = rapid[:, 0].astype(int)
    rv = rapid[:, 1].copy()
    rmoy = moy[ridx]
    tr_all = ~t_hold[ridx]
    rclim = np.array([rv[tr_all & (rmoy == m)].mean() for m in range(12)])
    rv_des = rv - rclim[rmoy]
    Fsec_true = Zsec.mean(1)
    mu_p, sd_p, w_probe, probe_val_r = fit_ridge(
        Fsec_true[ridx][tr_all], rv_des[tr_all])
    ym_to_r = {int(months[mi][:4]) * 100 + int(months[mi][5:7]): i
               for i, mi in enumerate(ridx)}
    print(f"probe fit: {int(tr_all.sum())} train months, "
          f"val-tail r {probe_val_r:+.3f}", flush=True)
    sec_t = torch.as_tensor(np.asarray(sec_sel), device=dev)

    def read_sv(zhat):
        fr = zhat[sec_t].mean(0).cpu().numpy()
        return float(np.dot(np.r_[(fr - mu_p) / sd_p, 1.0], w_probe))

    def zwin_from_true(s_end, K):
        arr = np.asarray(Zm[s_end - K + 1: s_end + 1])         # [K,P,dz] f16
        return (torch.from_numpy(np.ascontiguousarray(
            arr.transpose(1, 0, 2))).to(dev).float())          # [P,K,dz]

    # ---- per-stencil geometry, built once, shared across heads ------------
    nbr_cache, sctx_cache = {}, {}

    def geometry(stencil):
        if stencil not in nbr_cache:
            if stencil == 1:
                nbr_cache[1] = None
                sctx_cache[1] = torch.from_numpy(
                    np.concatenate([Zstat, coords], 1)).to(dev)
            else:
                NBR = build_stencil(Hg, Wg, ys, xs, stencil)
                nbr_cache[stencil] = torch.as_tensor(NBR, device=dev)
                sctx_cache[stencil] = torch.from_numpy(np.concatenate(
                    [Zstat, coords,
                     (NBR >= 0).astype(np.float32)], 1)).to(dev)
        return nbr_cache[stencil], sctx_cache[stencil]

    results = {"data": os.path.basename(a.x), "horizon": a.horizon,
               "hold_years": hold_years, "corridor_def": corridor_def,
               "gate_ref": dict(GATE_REF, head=GATE_HEAD, tol=GATE_TOL),
               "probe": {"val_tail_r": round(float(probe_val_r), 3)},
               "gate": {"pass": None, "skipped": True},   # overwritten below
               "heads": {}}
    K_seen = None

    for hp in heads:
        tk = torch.load(hp, map_location="cpu", weights_only=False)
        ta = tk["args"]
        K = ta["K"]
        if K_seen is None:
            K_seen = K
            results["K"] = K
        elif K != K_seen:
            sys.exit(f"{hp} has K={K} != {K_seen} — windows not comparable")
        stencil = ta.get("stencil", 1)
        unroll = ta.get("unroll", 1)
        label = (f"s{stencil}" + (f"u{unroll}" if unroll != 1 else "")
                 + f"_s{ta.get('seed', 0)}")
        if label in results["heads"]:
            label += "_" + os.path.basename(hp).replace(".pt", "")
        k_tbl = tk["model"]["pos.weight"].shape[0]  # the file, not a convention
        dir_ = tuple(int(x) for x in
                     str(ta.get("direct") or "").split(",") if x.strip())
        model = TemporalTransformer(d_z=d_z, d_model=ta["d_model"],
                                    n_heads=4, n_layers=ta["layers"],
                                    k_max=k_tbl, direct=dir_, stencil=stencil)
        model.load_state_dict(tk["model"])
        model.eval().to(dev)
        NBR_t, static_ctx = geometry(stencil)
        print(f"head {label}: {os.path.basename(hp)} "
              f"(d_model={ta['d_model']}, layers={ta['layers']}, K={K}, "
              f"stencil={stencil})", flush=True)

        Hh = a.horizon
        sums = {name: new_sums(Hh) for name, _ in scopes}
        probe_pts = {h: [] for h in range(1, Hh + 1)}
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
                    Zwin = zwin_from_true(s, K)
                    cur = list(moy[s - K + 1: s + 1])
                    v_pers, obs_s = std_m.get(s)
                    for h in range(1, Hh + 1):
                        t_tgt = s + h
                        if t_tgt >= T or months[t_tgt][:4] != Y:
                            break
                        # month features are the CURRENT window's months —
                        # rollout.py's mseq at step h spans s-K+h .. s+h-1;
                        # advance AFTER the forward, like project_amoc.py
                        zhat = roll_step(model, Zwin, NBR_t, static_ctx,
                                         month_feats(cur, dev), a.chunk,
                                         a.amp)
                        Zwin = torch.cat([Zwin[:, 1:], zhat[:, None]], 1)
                        cur = cur[1:] + [(cur[-1] + 1) % 12]
                        xhat = decode_all(codec, zhat, C, a.chunk, a.amp)
                        v_true, obs_tt = std_m.get(t_tgt)
                        op = obs_tt & obs_s
                        v_damp = v_pers * r1 ** h
                        for name, m_ in scopes:
                            accumulate(sums[name], h, xhat[m_], v_true[m_],
                                       v_pers[m_], v_damp[m_], op[m_])
                        ym = (int(months[t_tgt][:4]) * 100
                              + int(months[t_tgt][5:7]))
                        if ym in ym_to_r:
                            probe_pts[h].append(
                                (s, read_sv(zhat), rv_des[ym_to_r[ym]]))
                    print(f"  {label} start {start_m}: rolled", flush=True)

        entry = {"meta": {"file": os.path.basename(hp), "stencil": stencil,
                          "seed": ta.get("seed", 0), "unroll": unroll}}
        for name, _ in scopes:
            entry[name] = skill_block(sums[name], Hh)
        entry["amoc_bands"] = {}
        for bn, hs in BANDS:
            pts = [(h, s_, pr, y) for h in hs if h <= Hh
                   for (s_, pr, y) in probe_pts.get(h, [])]
            if len(pts) >= 8:
                pr = np.array([p[2] for p in pts])
                tv = np.array([p[3] for p in pts])
                entry["amoc_bands"][bn] = {
                    "r": round(float(np.corrcoef(pr, tv)[0, 1]), 3),
                    "n": len(pts)}
        for name, _ in scopes:
            if entry[name].get("chan_skill"):
                print(f"  {label} {name}: AUC(clim) "
                      f"{entry[name]['horizon_auc']:+.3f} · damped "
                      f"{entry[name]['auc_damped']:+.3f} · amp h{Hh} "
                      f"{entry[name]['chan_skill'][-1]['amp_ratio']:.3f}",
                      flush=True)
        print(f"  {label} amoc: " + " ".join(
            f"{bn} {v['r']:+.3f}(n={v['n']})"
            for bn, v in entry["amoc_bands"].items()), flush=True)

        # ---- VALIDATION GATE (fatal, before any spatial head is scored) --
        if hp in gate_paths and hp == gate_paths[0]:
            got = {"auc": entry["gate"].get("horizon_auc"),
                   "bands": {bn: entry["amoc_bands"].get(bn, {}).get("r")
                             for bn, _ in BANDS}}
            fails = []
            if got["auc"] is None or abs(got["auc"] - GATE_REF["auc"]) > GATE_TOL:
                fails.append(f"AUC {got['auc']} vs {GATE_REF['auc']}")
            for bn, ref in GATE_REF["bands"].items():
                gv = got["bands"].get(bn)
                if gv is None or abs(gv - ref) > GATE_TOL:
                    fails.append(f"{bn} {gv} vs {ref}")
            results["gate"] = {"head": label, "got": got,
                               "pass": not fails, "fails": fails}
            if fails:
                json.dump(results, open(a.out + ".gatefail", "w"), indent=1)
                sys.exit("VALIDATION GATE FAILED — the evaluator disagrees "
                         "with #217 on the stencil-1 baseline, so no spatial "
                         "number it produces can be trusted (plan §6.5). "
                         "Mismatches: " + "; ".join(fails)
                         + f" — partial results in {a.out}.gatefail")
            print(f"VALIDATION GATE PASSED: {got}", flush=True)

        # ---- long hindcast + future roll (median trajectory only) --------
        def long_roll(s_end, n_months):
            Zwin = zwin_from_true(s_end, K)
            cur = list(moy[s_end - K + 1: s_end + 1])
            yy, mm = int(months[s_end][:4]), int(months[s_end][5:7])
            sv, roll_ym = [], []
            with torch.no_grad():
                for _ in range(n_months):
                    mm += 1
                    if mm > 12:
                        mm, yy = 1, yy + 1
                    zhat = roll_step(model, Zwin, NBR_t, static_ctx,
                                     month_feats(cur, dev), a.chunk, a.amp)
                    Zwin = torch.cat([Zwin[:, 1:], zhat[:, None]], 1)
                    cur = cur[1:] + [(cur[-1] + 1) % 12]
                    sv.append(read_sv(zhat))
                    roll_ym.append(f"{yy:04d}-{mm:02d}")
            return np.array(sv), roll_ym

        if a.long_start and a.long_start in month_index \
                and month_index[a.long_start] - K + 1 >= 0:
            sv, roll_ym = long_roll(month_index[a.long_start], a.long_months)
            truth = np.full(len(sv), np.nan)
            trained = np.zeros(len(sv), bool)
            for i, ym in enumerate(roll_ym):
                key = int(ym[:4]) * 100 + int(ym[5:7])
                if key in ym_to_r:
                    truth[i] = rv_des[ym_to_r[key]]
                    trained[i] = not t_hold[ridx[ym_to_r[key]]]

            def _r(m_):
                if m_.sum() < 8:
                    return None, int(m_.sum())
                return (round(float(np.corrcoef(sv[m_], truth[m_])[0, 1]), 3),
                        int(m_.sum()))
            fin = np.isfinite(truth)
            r_tr, n_tr = _r(fin & trained)
            r_ho, n_ho = _r(fin & ~trained)
            sv_lp, tr_lp = smooth(sv), smooth(np.where(fin, truth, np.nan))
            both = np.isfinite(sv_lp) & np.isfinite(tr_lp)
            r_lp = amp_lp = None
            if both.sum() >= 24:
                r_lp = round(float(np.corrcoef(sv_lp[both],
                                               tr_lp[both])[0, 1]), 3)
                amp_lp = round(float(sv_lp[both].std()
                                     / (tr_lp[both].std() + 1e-9)), 3)
            entry["long"] = {"context_end": a.long_start, "roll_ym": roll_ym,
                             "sv_des": [round(v, 3) for v in sv.tolist()],
                             "r_trained": r_tr, "n_trained": n_tr,
                             "r_heldout": r_ho, "n_heldout": n_ho,
                             "r_lp18": r_lp, "amp_lp18": amp_lp}
            print(f"  {label} long({a.long_start}+{a.long_months}m): "
                  f"r_trained {r_tr} (n={n_tr}) · r_heldout {r_ho} "
                  f"(n={n_ho}) · lp18 r {r_lp} amp {amp_lp}", flush=True)
        if a.future_months > 0:
            sv, roll_ym = long_roll(T - 1, a.future_months)
            entry["future"] = {"context_end": months[-1], "roll_ym": roll_ym,
                               "sv_des": [round(v, 3) for v in sv.tolist()]}
        results["heads"][label] = entry
        model.to("cpu")
        if dev.type == "cuda":
            torch.cuda.empty_cache()

    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    with open(a.out, "w") as f:
        json.dump(results, f, indent=1)
    print(f"wrote {a.out} ({os.path.getsize(a.out):,} bytes)", flush=True)


if __name__ == "__main__":
    main()

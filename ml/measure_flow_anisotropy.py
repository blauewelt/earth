#!/usr/bin/env python3
"""How anisotropic is the flow the stencil is trying to catch?

Chris, 2026-08-14, proposing an ELLIPTIC spiral: *"the north pole is less
important than having a receptive window across 4k km east / west."* Before
encoding that as a shape, measure it — CLAUDE.md forbids hand-picked
thresholds, and an aspect ratio is a threshold wearing a geometry's clothes.

The measurement is GEOSTROPHIC: u = -(g/f) dSSH/dy, v = (g/f) dSSH/dx, from
the family3 tensor's own SSH channel — the flow field exactly as the model
can see it, on the grid it sees it at. Two facts make this sound here:

  * the tensor is z-scored PER CHANNEL with one global mean/std (verified in
    build_family3.py), so every SSH gradient is scaled by the same constant
    and the |u|/|v| RATIO is exact even though the absolute speeds are in
    z-units times an unknown factor;
  * f = 2*Omega*sin(phi) varies with latitude, so rows below 10 N are
    excluded (f -> 0 blows up the geostrophic estimate; the corridor lives
    at 26.5 N+ anyway).

Two flow decompositions, because they advect differently:

  MEAN   time-mean SSH -> the standing gyre/jet system. This is what carries
         water coherently over MULTIPLE months, i.e. the thing a multi-month
         roll needs to reach along.
  MONTHLY per-month fields -> total flow including eddies. Eddies are round-
         ish, so this bounds the anisotropy from below.

    python3 ml/measure_flow_anisotropy.py
"""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
X_PATH = os.path.join(HERE, "cache", "family3_X.npy")
MASK = os.path.join(HERE, "cache", "ocean_mask.npy")
EVAL = os.path.join(os.path.dirname(HERE), "data", "amoc_eval_mask.json")

CH_SSH = 2                    # CHANS = ["cur_speed", "log_mld", "ssh", ...]
KM_PER_DEG = 111.32
DLAT = 0.25
OMEGA = 7.2921e-5


def corridor_mask():
    g = json.load(open(EVAL))
    vals = np.array([-1 if c == "." else int(c) for c in g["packed"]],
                    np.int8).reshape(g["ny"], g["nx"])
    return vals >= 2


def uv_from_ssh(eta, lats):
    """Geostrophic u, v (z-units * const) from one SSH field [H, W]."""
    H, W = eta.shape
    f = 2 * OMEGA * np.sin(np.radians(lats))[:, None]
    dy_km = DLAT * KM_PER_DEG
    dx_km = DLAT * KM_PER_DEG * np.cos(np.radians(lats))[:, None]
    deta_dy = np.full_like(eta, np.nan)
    deta_dx = np.full_like(eta, np.nan)
    deta_dy[1:-1] = (eta[2:] - eta[:-2]) / (2 * dy_km)
    deta_dx[:, 1:-1] = (eta[:, 2:] - eta[:, :-2]) / (2 * dx_km)
    u = -deta_dy / f            # zonal
    v = deta_dx / f             # meridional
    return u, v


def stats(u, v, sel):
    ok = sel & np.isfinite(u) & np.isfinite(v)
    au, av = np.abs(u[ok]), np.abs(v[ok])
    return (float(np.mean(au)), float(np.mean(av)),
            float(np.sqrt(np.mean(u[ok] ** 2))),
            float(np.sqrt(np.mean(v[ok] ** 2))), int(ok.sum()))


def main():
    X = np.load(X_PATH, mmap_mode="r")
    T, H, W, C = X.shape
    ocean = np.load(MASK)
    corr = corridor_mask()
    assert corr.shape == ocean.shape
    lats = 0.125 + np.arange(H) * DLAT          # south-first, 0..70 N
    far_enough = lats >= 10.0
    sel_w = ocean & far_enough[:, None]
    sel_c = corr & far_enough[:, None]

    # MEAN flow: average SSH over every month, then differentiate once.
    eta_sum = np.zeros((H, W)); eta_cnt = np.zeros((H, W))
    step = max(T // 172, 1)
    months = range(0, T, 1)
    for t in months:
        e = np.array(X[t, :, :, CH_SSH], dtype=np.float64)
        fin = np.isfinite(e)
        eta_sum[fin] += e[fin]; eta_cnt += fin
    eta_mean = np.where(eta_cnt > 0, eta_sum / np.maximum(eta_cnt, 1), np.nan)
    u, v = uv_from_ssh(eta_mean, lats)
    print(f"tensor {T} months · {int(sel_w.sum())} window px >=10N · "
          f"{int(sel_c.sum())} corridor px >=10N\n")
    print(f"{'population':<10} {'flow':<8} {'mean|u|':>9} {'mean|v|':>9} "
          f"{'|u|/|v|':>8} {'rms u':>9} {'rms v':>9} {'rms ratio':>9}")
    for name, sel in (("window", sel_w), ("corridor", sel_c)):
        mu_, mv_, ru, rv, n = stats(u, v, sel)
        print(f"{name:<10} {'MEAN':<8} {mu_:>9.4f} {mv_:>9.4f} "
              f"{mu_ / mv_:>8.2f} {ru:>9.4f} {rv:>9.4f} {ru / rv:>9.2f}")

    # MONTHLY flow: differentiate each month, accumulate — sampled every
    # 4th month (129 fields), enough for a ratio and 4x cheaper.
    acc = np.zeros(8); nacc = np.zeros(2, dtype=np.int64)
    for t in range(0, T, 4):
        e = np.array(X[t, :, :, CH_SSH], dtype=np.float64)
        if not np.isfinite(e).any():
            continue          # SSH starts 1993; the tensor 1982. A channel
            # that starts later than the tensor leaves whole early months
            # empty, and a mean over an empty section is NaN, not an error
            # (docs/ML_BASICS.md §3) — one poisoned month NaNs the whole
            # accumulator, which is exactly what happened on first run.
        u, v = uv_from_ssh(e, lats)
        for i, sel in enumerate((sel_w, sel_c)):
            mu_, mv_, ru, rv, n = stats(u, v, sel)
            acc[4 * i:4 * i + 4] += (mu_, mv_, ru ** 2, rv ** 2)
            nacc[i] += 1
    for i, name in enumerate(("window", "corridor")):
        mu_, mv_ = acc[4 * i] / nacc[i], acc[4 * i + 1] / nacc[i]
        ru = np.sqrt(acc[4 * i + 2] / nacc[i])
        rv = np.sqrt(acc[4 * i + 3] / nacc[i])
        print(f"{name:<10} {'MONTHLY':<8} {mu_:>9.4f} {mv_:>9.4f} "
              f"{mu_ / mv_:>8.2f} {ru:>9.4f} {rv:>9.4f} {ru / rv:>9.2f}")
    print("\nunits are z-score-gradient units (one unknown constant, "
          "identical for u and v\nby the per-channel global z-score) — only "
          "the RATIOS are physical.")


if __name__ == "__main__":
    main()

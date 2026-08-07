#!/usr/bin/env python3
"""Assemble the North-Atlantic per-pixel training tensor from the repo's data.

This is Phase-1 scaffolding for the Earth-State Embeddings pilot (see
ml/README.md and the private proposal): one row of channels per (lat, lon,
month) pixel, on the 1-degree GLORYS grid the app already bakes, so the whole
pipeline — masking, bottleneck, neighbour prediction, blocked splits, the
RAPID probe — runs end-to-end TODAY on committed data. The channel list is
deliberately thin (what the repo carries); ml/README.md lists the archives the
Colab run adds (RG-Argo T/S at depth, EN4, ERA5 winds, altimetry SLA, OISST
monthly means) and each lands as just another channel here.

Output: ml/cache/na_pixels.npz
  X       float32 [T, H, W, C]   z-scored channels, NaN where unobserved
  months  [T] "YYYY-MM"          1993-01 .. latest baked
  lats    [H], lons [W]          cell centres
  chan    [C] channel names
  norm    [C, 2]                 (mean, std) used, so embeddings de-normalise
  rapid   [Tr, 2]                (month-index into `months`, transport Sv) —
                                 the held-out physical probe, NEVER a channel

Run:  python3 ml/build_dataset.py
"""
import argparse
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data")
OUT = os.path.join(ROOT, "ml", "cache")

# Windows. "na" is the pilot (proposal §9): the subpolar/subtropical North
# Atlantic, where coverage is richest and the AMOC story is decisive.
# "global" (the default since 2026-08-07) is the entire baked GLORYS grid —
# every ocean the archives cover, ~7× the pilot's cells. The fetchers
# (fetch_rg_channels, fetch_wind_channels) inherit whatever grid this file
# writes, so the window is chosen HERE and only here. 1° cells either way.
WINDOWS = {
    "na": (-100.0, 20.0, 0.0, 70.0),
    "global": (-180.0, 180.0, -80.0, 90.0),
}
WEST, EAST, SOUTH, NORTH = WINDOWS["global"]


def load_monthly(fname, ydir):
    """A monthly 1° grid in the app's split format → dict month → np [ny, nx]."""
    g = json.load(open(os.path.join(DATA, fname)))
    months = {}

    def put(k, flat):
        a = np.array([np.nan if v is None else float(v) for v in flat], dtype=np.float32)
        months[k] = a.reshape(g["ny"], g["nx"])          # row 0 = southmost

    for k, flat in g.get("months", {}).items():
        put(k, flat)
    ydirp = os.path.join(DATA, ydir)
    for f in sorted(os.listdir(ydirp)):
        y = json.load(open(os.path.join(ydirp, f)))
        for k, flat in y["months"].items():
            if k not in months:
                put(k, flat)
    return g, months


def load_static_grid(fname):
    g = json.load(open(os.path.join(DATA, fname)))
    a = np.array([np.nan if v is None else float(v) for v in g["values"]], dtype=np.float32)
    return g, a.reshape(g["ny"], g["nx"])


def window_indices(g):
    """Column/row slices of the pilot window on grid g (row 0 = south)."""
    x0 = int(round((WEST - g["west"]) / g["dlon"]))
    x1 = int(round((EAST - g["west"]) / g["dlon"]))
    y0 = int(round((SOUTH - g["south"]) / g["dlat"]))
    y1 = int(round((NORTH - g["south"]) / g["dlat"]))
    return x0, x1, y0, y1


def resample_to(g_src, arr, lats, lons):
    """Nearest-cell read of a coarser static grid at our 1° cell centres."""
    iy = np.clip(((lats - g_src["south"]) / g_src["dlat"]).astype(int), 0, g_src["ny"] - 1)
    ix = np.clip(((lons - g_src["west"]) / g_src["dlon"]).astype(int) % g_src["nx"], 0, g_src["nx"] - 1)
    return arr[np.ix_(iy, ix)]


def main():
    global WEST, EAST, SOUTH, NORTH
    ap = argparse.ArgumentParser()
    ap.add_argument("--window", choices=sorted(WINDOWS), default="global",
                    help="spatial window (default: global; 'na' reproduces the pilot)")
    a = ap.parse_args()
    WEST, EAST, SOUTH, NORTH = WINDOWS[a.window]
    print(f"window '{a.window}': lon {WEST}..{EAST}, lat {SOUTH}..{NORTH}")
    os.makedirs(OUT, exist_ok=True)

    print("loading GLORYS surface-current speed …")
    gc, cur = load_monthly("currents.json", "currents_y")
    print("loading GLORYS mixed-layer depth …")
    gm, mld = load_monthly("mld.json", "mld_y")
    months = sorted(set(cur) & set(mld))
    print(f"  {len(months)} common months {months[0]} .. {months[-1]}")

    x0, x1, y0, y1 = window_indices(gc)
    lats = gc["south"] + (np.arange(y0, y1) + 0.5) * gc["dlat"]
    lons = gc["west"] + (np.arange(x0, x1) + 0.5) * gc["dlon"]
    H, W = len(lats), len(lons)

    print("loading static climatologies (OISST SST, GPCP precip) …")
    go, oisst = load_static_grid("oisst.json")
    gp, gpcp = load_static_grid("gpcp.json")
    sst_clim = resample_to(go, oisst, lats, lons)
    pr_clim = resample_to(gp, gpcp, lats, lons)

    # Channels. Transforms tame the heavy tails BEFORE z-scoring: MLD spans
    # 10..2000 m log-normally; current speed is half-normal-ish.
    T = len(months)
    chans = ["cur_speed", "log_mld", "sst_clim", "precip_clim"]
    X = np.full((T, H, W, len(chans)), np.nan, dtype=np.float32)
    for t, mth in enumerate(months):
        X[t, :, :, 0] = cur[mth][y0:y1, x0:x1]
        X[t, :, :, 1] = np.log10(np.clip(mld[mth][y0:y1, x0:x1], 1.0, None))
    X[:, :, :, 2] = sst_clim[None, :, :]
    X[:, :, :, 3] = pr_clim[None, :, :]

    # Ocean mask: a pixel is ocean if the monthly fields ever report there.
    ocean = np.isfinite(X[:, :, :, 0]).any(axis=0)
    X[:, ~ocean, :] = np.nan
    print(f"  window {H}x{W}, ocean cells: {int(ocean.sum())}/{H * W}")

    # z-score per channel over observed values only; keep the constants.
    norm = np.zeros((len(chans), 2), dtype=np.float32)
    for c in range(len(chans)):
        v = X[..., c][np.isfinite(X[..., c])]
        mu, sd = float(v.mean()), float(v.std() + 1e-6)
        norm[c] = (mu, sd)
        X[..., c] = (X[..., c] - mu) / sd

    # RAPID 26.5N overturning — the physical probe, held OUT of the channels.
    rapid = []
    try:
        r = json.load(open(os.path.join(DATA, "rapid_moc.json")))
        # month-mean the series onto our month axis
        idx = {m: i for i, m in enumerate(months)}
        acc = {}
        for tstr, v in zip(r["t"], r["moc"]):
            k = str(tstr)[:7]
            if k in idx and v is not None:
                acc.setdefault(k, []).append(float(v))
        rapid = [(idx[k], float(np.mean(vs))) for k, vs in sorted(acc.items())]
        print(f"  RAPID probe: {len(rapid)} monthly means")
    except Exception as e:
        print("  RAPID probe unavailable:", e)

    np.savez_compressed(
        os.path.join(OUT, "na_pixels.npz"),
        X=X, months=np.array(months), lats=lats.astype(np.float32),
        lons=lons.astype(np.float32), chan=np.array(chans), norm=norm,
        rapid=np.array(rapid, dtype=np.float32).reshape(-1, 2),
        window=np.array(a.window),
    )
    mb = os.path.getsize(os.path.join(OUT, "na_pixels.npz")) / 1e6
    print(f"wrote ml/cache/na_pixels.npz  [T={T} H={H} W={W} C={len(chans)}]  {mb:.1f} MB")


if __name__ == "__main__":
    sys.exit(main())

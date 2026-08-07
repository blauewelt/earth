#!/usr/bin/env python3
"""Lever 2 — more long-record dynamic channels: monthly SST (OISST v2.1).

The tensor's SST so far is a STATIC climatology (sst_clim); the actual
monthly sea surface temperature is the longest no-credential dynamic field
available (1981-09 → present, global 1°) and the repo already bakes it for
the app's SST layer (data/oisst_monthly.json + data/oisst_y/YYYY.json) —
so this fetcher reads the repo's own baked files, no download at all.

Appends channel `sst_monthly` (z-scored °C) to ml/cache/na_pixels.npz,
re-runnably (drops an existing sst_monthly first), preserving every other
key (window, truth series, …). Coverage is total on ice-free ocean for
every tensor month — and when the time axis later extends before 1993,
SST + wind stress already reach 1982/1948.

Run AFTER build_dataset/fetch_rg/fetch_wind:  python3 ml/fetch_sst_channel.py
Optional: --npz <path> (testing against a copy).
"""
import argparse
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
DATA = os.path.join(ROOT, "data")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", default=os.path.join(HERE, "cache", "na_pixels.npz"))
    a = ap.parse_args()
    d = dict(np.load(a.npz))
    X = d["X"]
    months = [str(m) for m in d["months"]]
    lats, lons = d["lats"], d["lons"]
    chans = [str(c) for c in d["chan"]]
    if "sst_monthly" in chans:                       # re-run: rebuild from scratch
        keep = [i for i, c in enumerate(chans) if c != "sst_monthly"]
        X = X[..., keep]
        chans = [chans[i] for i in keep]
        d["norm"] = d["norm"][keep]
    T, H, W, C = X.shape
    print(f"tensor: T={T} H={H} W={W} C={C}, appending sst_monthly …")

    g = json.load(open(os.path.join(DATA, "oisst_monthly.json")))
    ny, nx = g["ny"], g["nx"]                        # 180x360, west -180 south -90
    iy = np.clip(((lats - g["south"]) / 1.0).astype(int), 0, ny - 1)
    ix = np.clip(((lons - g["west"]) / 1.0).astype(int), 0, nx - 1)

    add = np.full((T, H, W, 1), np.nan, dtype=np.float32)
    ydir = os.path.join(DATA, "oisst_y")
    cache = {}
    for t, m in enumerate(months):
        y = m[:4]
        if y not in cache:
            yf = os.path.join(ydir, f"{y}.json")
            cache = {y: json.load(open(yf))["months"] if os.path.exists(yf) else {}}
        flat = cache[y].get(m)
        if flat is None:
            continue
        a2 = np.array([np.nan if v is None else float(v) for v in flat],
                      dtype=np.float32).reshape(ny, nx)
        add[t, :, :, 0] = a2[np.ix_(iy, ix)]

    # restrict to the tensor's own ocean mask; z-score on observed values
    ocean = np.isfinite(X[..., 0]).any(axis=0)
    add[:, ~ocean, :] = np.nan
    v = add[np.isfinite(add)]
    mu, sd = float(v.mean()), float(v.std() + 1e-6)
    add = (add - mu) / sd
    cov = np.isfinite(add).mean()
    print(f"  sst_monthly coverage {cov:5.1%}  mu {mu:6.2f}°C  sd {sd:5.2f}")

    d["X"] = np.concatenate([X, add], axis=-1).astype(np.float32)
    d["chan"] = np.array(chans + ["sst_monthly"])
    d["norm"] = np.concatenate([d["norm"], np.array([[mu, sd]], np.float32)], 0)
    np.savez_compressed(a.npz, **d)
    print(f"rewrote {a.npz}: C={d['X'].shape[-1]} channels "
          f"({os.path.getsize(a.npz) / 1e6:.0f} MB)")


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
"""Read a PUBLISHED family-1.2 ERA5 store back from the Hub and compare it
with an independent read of the source archive (E-085).

PLAIN ENGLISH. After a store is assembled and published, this is the check
that the bytes a reader will download say what ERA5 says. It opens the
store's group through `family1/sharded.py :: ShardedGroup` on the Hub's
`resolve/main` URL — the reader every consumer uses, two range reads per
tile — and reads whole frames. For each instant it reads the same field
straight from Google's ARCO-ERA5 zarr with xarray + zarr, code that shares
nothing with the adapter:

  before the 2022 seam   the 1-degree archive, reindexed (south-first rows,
                         columns from -180) — the store holds it unchanged
  from the seam          the 0.25-degree archive, regridded HERE by a dense
                         matrix form of the conservative box mean (weights
                         built from sin(latitude) and longitude overlaps
                         with numpy outer operations, applied with `@`),
                         independent of the adapter's sparse fixed-order sums

and requires every value to lie within half a float16 step of the
reference (the only error the store is allowed), reporting the largest
difference per level. It also checks store.json, tile_grid.json and
shard_index.npy against the record (bins, frames, the declared levels).

  python3 ml/family1/era5_hub_check.py --store era5_t \\
      --when 2015-01-15T12:00 --when 2023-07-15T06:00 --out <json>
"""
import argparse
import datetime as dt
import json
import os
import sys
import time
import urllib.request

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

from family1 import sharded as sh                               # noqa: E402
from family1.adapters import REGISTRY                           # noqa: E402
from family1.adapters import _era5 as e5                        # noqa: E402

HUB = ("https://huggingface.co/datasets/chfrank/earth-tensors/resolve/main/"
       "tensors/family1_2")


def hub_json(url):
    return json.loads(urllib.request.urlopen(url, timeout=120).read())


def half_step(ref):
    mag = np.abs(ref)
    e = np.floor(np.log2(np.maximum(mag, 2.0 ** -24)))
    return 2.0 ** (np.maximum(e, -14.0) - 10.0) / 2


def dense_conservative(src_lat_asc, src_lon, tgt_lat, tgt_lon):
    """Dense [n_tgt, n_src] weight matrices, written independently of the
    adapter: numpy outer min/max, no per-cell loops."""
    def lat_edges(p):
        m = (p[:-1] + p[1:]) / 2
        return np.concatenate([[-90.0], m, [90.0]])
    se, te = np.radians(lat_edges(src_lat_asc)), np.radians(lat_edges(tgt_lat))
    hi = np.minimum(te[1:, None], se[None, 1:])
    lo = np.maximum(te[:-1, None], se[None, :-1])
    WL = np.where(hi > lo, np.sin(hi) - np.sin(lo), 0.0)
    WL /= WL.sum(1, keepdims=True)

    def lon_edges(p):
        d_next = np.diff(np.concatenate([p, [p[0] + 360.0]]))
        d_prev = np.concatenate([[d_next[-1]], d_next[:-1]])
        return p - d_prev / 2, p + d_next / 2
    slo, shi = lon_edges(src_lon)
    tlo, thi = lon_edges(tgt_lon)
    WO = np.zeros((len(tgt_lon), len(src_lon)))
    for k in (-360.0, 0.0, 360.0):
        WO += np.clip(np.minimum(thi[:, None], shi[None, :] + k)
                      - np.maximum(tlo[:, None], slo[None, :] + k), 0, None)
    WO /= WO.sum(1, keepdims=True)
    return WL, WO


def reference(ad, when, seam):
    import xarray as xr
    lat_t, lon_t = e5.target_axes(1.0)
    if when < seam:
        ds = xr.open_zarr(f"{e5.GCS}/{e5.ZARR_A}", consolidated=True,
                          chunks=None)
        da = ds[ad.src_name].sel(time=np.datetime64(when),
                                 level=list(e5.LEVELS_HPA))
        da = da.assign_coords(longitude=((da.longitude + 180.0) % 360.0)
                              - 180.0).sortby("longitude").sortby("latitude")
        v = da.transpose("latitude", "longitude", "level").values
        return v.astype(np.float64) * ad.scale, f"{e5.ZARR_A} (reindexed)"
    ds = xr.open_zarr(f"{e5.GCS}/{e5.ZARR_B}", consolidated=True, chunks=None)
    da = ds[ad.src_name].sel(time=np.datetime64(when),
                             level=list(e5.LEVELS_HPA)).sortby("latitude")
    da = da.transpose("level", "latitude", "longitude")
    src = da.values.astype(np.float64)
    WL, WO = dense_conservative(da.latitude.values.astype(np.float64),
                                da.longitude.values.astype(np.float64),
                                lat_t, lon_t)
    out = np.stack([WL @ src[k] @ WO.T for k in range(src.shape[0])], -1)
    return out * ad.scale, f"{e5.ZARR_B} (dense matrix regrid, this script)"


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--store", required=True)
    ap.add_argument("--when", action="append", required=True)
    ap.add_argument("--base", default=HUB)
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    ad = REGISTRY[a.store]()
    t0 = time.time()
    sm = hub_json(f"{a.base}/{a.store}/store.json")
    gdir = f"{a.base}/{a.store}/{a.store}"
    g = sh.ShardedGroup(gdir)
    sp, idx = g.spec, g.shard_index
    gm = sm["groups"][a.store]
    out = {"store": a.store, "store_json": {
        k: sm.get(k) for k in ("family", "family_version", "distribution",
                               "date_range", "frames_missing_by_reason",
                               "builder_git_sha", "built_at")},
        "group": {k: gm.get(k) for k in ("bins", "bin_first", "bin_last",
                                         "frames_present", "frames_missing",
                                         "tiles_stored", "bytes")},
        "files_in_sha256_block": len(sm.get("sha256") or {}),
        "tile_grid": {k: sp.get(k) for k in ("H", "W", "C", "tile",
                                             "frames_per_bin",
                                             "frame_seconds", "dtype",
                                             "levels_hpa")},
        "shard_index_rows": int(len(idx)),
        "shard_index_frames": int(idx["frames_present"].sum())}
    checks = {
        "levels_hpa": sp.get("levels_hpa") == list(e5.LEVELS_HPA),
        "bins_equal_index_rows": int(gm["bins"]) == int(len(idx)),
        "frames_equal_index": int(gm["frames_present"]) ==
        int(idx["frames_present"].sum()),
        "files_two_per_bin_plus_two": len(sm.get("sha256") or {}) ==
        2 * int(len(idx)) + 2}
    seam = dt.datetime(2022, 1, 1)
    out["frames"] = []
    for w in a.when:
        when = dt.datetime.fromisoformat(w)
        s = int((when - e5.EPOCH).total_seconds())
        b, rem = divmod(s, sh.BIN_SECONDS)
        f = rem // e5.FRAME_SECONDS
        r0 = g.src.ranges
        tr = time.time()
        got = g.read_frame(b, f)
        t_read = time.time() - tr
        ref, how = reference(ad, when, seam)
        d = np.abs(got.astype(np.float64) - ref)
        ok = d <= half_step(ref) * (1 + 1e-6) + 1e-12
        out["frames"].append({
            "instant": str(when), "bin": b, "frame": f,
            "source_of_store": "before seam" if when < seam else "after seam",
            "reference": how,
            "hub_range_reads": g.src.ranges - r0,
            "hub_read_seconds": round(t_read, 2),
            "max_abs_diff": {nm: float(d[:, :, k].max())
                             for k, nm in enumerate(ad.channel_names)},
            "max_abs_diff_all": float(d.max()),
            "values": int(d.size),
            "within_half_float16_step": bool(ok.all()),
            "n_outside": int((~ok).sum())})
        checks[f"frame {when} within half a float16 step"] = bool(ok.all())
    out["checks"] = checks
    out["ok"] = all(checks.values())
    out["seconds"] = round(time.time() - t0, 1)
    js = json.dumps(out, indent=1)
    print(js)
    if a.out:
        with open(a.out, "w") as fh:
            fh.write(js + "\n")
    return 0 if out["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())

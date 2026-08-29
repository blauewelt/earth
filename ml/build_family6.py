#!/usr/bin/env python3
"""Family-6 tensor: CMIP6 piControl on the family-3 grid — an UNLABELLED
pretraining corpus.

WHAT THIS IS. Families 3/4/5 are the observed North Atlantic: real dates, real
labels (RAPID, the Florida cable, OSNAP, MOVE, SAMBA), ~43 years of it. That is
a small corpus for a state model, and E-033's scale program is bounded by it.
This family buys *state* without buying *labels*: ~800 model-years of CMIP6
piControl ocean output from two eORCA025 models, regridded onto family 3's exact
0.25-degree North Atlantic axes and stored in family 3's exact 40-channel layout.

THERE ARE NO LABELS HERE AND THERE CANNOT BE. piControl is a free-running
pre-industrial control: no volcanoes, no CO2 trend, no assimilation, and NO
CORRESPONDENCE TO ANY CALENDAR YEAR. Model-year 137 of HadGEM3's control is not
1987, it is not any year, and the RAPID array did not measure it. So this script
attaches no truth series, writes no `rapid`/`fc`/`osnap`/`move`/`samba` keys,
and any downstream reader that expects them must refuse this tensor rather than
fill them with zeros. It is for SELF-SUPERVISED pretraining only — the mask/
missing-token objective the encoder already trains on — and every supervised
number stays measured on families 3/4/5. A checkpoint pretrained here and
fine-tuned there must say so in its EXPERIMENTS.md entry, because a
distribution-shift claim that is not written down is a claim nobody can retract.

MODELS. Both are NEMO eORCA025 tripolar, nominal 25 km — the SAME nominal
resolution as the target grid, so the regrid is a resampling and never an
upsample. Nothing here invents sub-grid structure:

  HadGEM3-GC31-MM  piControl r1i1p1f1   6000 months (500 model-years)  [.,1205,1440]
  CNRM-CM6-1-HR    piControl r1i1p1f2   3600 months (300 model-years)  [.,1050,1442]

VARIABLES: `mlotst`, `zos`, `tos`. ALL THREE ARE SCALARS, and that is a
deliberate scope limit rather than an oversight — see the comment at
`VARIABLES` below. We do not fetch `uo`/`vo`.

CHANNELS. Family 3's 40-channel r2 layout, imported from `build_family3` so
there is one definition, with exactly three channels live:

    index  1  log_mld   = log10(mlotst)      (mlotst is in m, as family 4's)
    index  2  ssh       = zos                (m)
    index 39  sst       = tos                (degC)

The other 37 are MISSING TOKENS, over the whole corpus. That is a large hole and
it is stated rather than buried: `rg_*` (32 of them) is already absent in 83.6%
of family-4's pentad bins, so permanent absence is inside the regime the
architecture was designed for — the `missing` token is distinct from `mask` by
construction and that distinction was measured to matter. But `cur_speed`
(index 0) is a REAL LIMITATION, not a designed one: family 3 derives it from
GLORYS `uo`/`vo`, and we are not fetching currents, so an encoder pretrained
here has never seen the surface-velocity channel it will meet at fine-tune
time. `tau_x`/`tau_y`/`tau_x_std`/`tau_y_std` are missing for the same reason —
the wind stress lives in the models' Amon/Omon flux variables, not here.

REGRIDDING is the part that must be right, so it is done once and checked
loudly. Both sources are CURVILINEAR tripolar grids: the datasets carry 2-D
`latitude`/`longitude` (HadGEM3) or `lat`/`lon` (CNRM) arrays over (j,i), and
north of ~60N the grid's poles sit over Canada and Siberia, so nothing about
(j,i) is zonal. The mapping is therefore built as a FIXED INDEX ARRAY, once,
and reused for every time slice:

  * source and target cell centres go to 3-D UNIT VECTORS on the sphere. This
    is what makes the dateline and the poles non-special: nearest-neighbour in
    raw lat/lon degrees would be wrong at the seam and anisotropic everywhere.
  * `scipy.spatial.cKDTree` over the source vectors, restricted first to a
    generous North Atlantic bounding box (otherwise it is 1.7M points and both
    slow and memory-hungry), one nearest neighbour per target cell.
  * the MAXIMUM great-circle distance between a target cell and the source cell
    it was given is measured and REPORTED in km. Both grids are ~25 km, so a
    max much beyond ~40 km means the mapping is wrong — a flipped axis, a
    degrees/radians slip, a bounding box that clipped a corner — and the script
    REFUSES rather than writing 100 GB of plausible-looking nonsense.

Nearest-neighbour and not bilinear, deliberately: a conservative or bilinear
remap on a tripolar source needs the source cell corners and a land mask to
avoid bleeding land into coastal ocean, and at equal nominal resolution it buys
almost nothing. Nearest-neighbour transports the model's own land mask exactly.

STORAGE. The family-5 sidecar layout (`ml/tensor_io.py`): a small `.npz` index
beside a bare memmappable `.npy`, because `np.load` on a compressed npz member
decompresses the whole array into RAM.

    family6_na025_cmip6.npz      lats, lons, bins, chan, per-source provenance
    family6_na025_cmip6_X.npy    float16 [T, 281, 481, 40]

At the full 9600 months that is 103.8 GB, which does NOT fit this sandbox
(~20 GB free). `--probe` is the sandbox mode and the only one that should ever
be run here.

MEASURED, on 2026-08-29, from this sandbox through the agent proxy, so nobody
re-derives them: the catalogue is 80.8 MB and came back in 0.9 s; a zarr read
of 36 monthly slices of HadGEM3 `tos` ran at 79.5 MB/s decoded, i.e. 0.087 s
per month per variable, which puts the full 28,800 slice-reads at roughly 0.7 h
of streaming on one thread. The build is therefore bounded by disk, not by the
network — it wants ~110 GB free, and `preflight_write` refuses below that.

Run:
  python3 ml/build_family6.py --self-test          # regrid math, no network
  python3 ml/build_family6.py --probe              # catalogue + map + 1 slice + QC
  python3 ml/build_family6.py --dry-run            # byte arithmetic only
  python3 ml/build_family6.py                      # the full build (a BOX job)
  python3 ml/build_family6.py --max-times 24       # smoke the write path
"""
import argparse
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import build_family3 as f3                                    # noqa: E402
from build_family4 import CHANS_R2, C_SST, check_grid         # noqa: E402

CACHE = os.path.join(HERE, "cache")
OUT_NPZ = os.path.join(CACHE, "family6_na025_cmip6.npz")
CATALOG_URL = "https://storage.googleapis.com/cmip6/pangeo-cmip6.csv"
CATALOG = os.path.join(CACHE, "pangeo-cmip6.csv")
RECIPE_REV = "f6r1"

# The target axes. Family 3's, restated here ONLY so that --probe can run with
# no data cache at all; `check_grid` is what actually decides, against
# `base025_na.npz`, exactly as build_family4.py does it — and warns rather than
# fails when that reference is absent, because a sandbox legitimately lacks it.
LAT0, LAT1, LON0, LON1, DEG = 0.0, 70.0, -100.0, 20.0, 0.25
NLAT, NLON = 281, 481

# The two sources. `zstore` is NOT hardcoded: it is resolved from the published
# catalogue by (source_id, experiment_id, table_id, variable_id, member_id), so
# a version bump on the Pangeo side is picked up instead of 404-ing.
MODELS = [
    dict(source_id="HadGEM3-GC31-MM", member_id="r1i1p1f1", years=500),
    dict(source_id="CNRM-CM6-1-HR", member_id="r1i1p1f2", years=300),
]
EXPERIMENT, TABLE = "piControl", "Omon"

# SCALARS ONLY, and this is the reason, stated where the list is:
# eORCA025 is a tripolar C-grid. `uo`/`vo` are stored on the U- and V-points of
# that grid, in GRID-RELATIVE (i,j) directions, which north of ~60N are rotated
# by a large and spatially varying angle away from east/north. Regridding them
# as if they were east/north components would produce a velocity field that is
# smooth, plausible, correctly masked, and WRONG — a silent-error class, and
# the exact shape of error this repo has been burned by before. Doing it
# properly needs the grid's rotation angles (or the vertices_* corner arrays)
# and a C-grid-to-T-point interpolation first. Scalars at T-points need none of
# that: `mlotst`, `zos` and `tos` are all cell-centred and direction-free, so
# nearest-neighbour on the sphere is exactly right for them. The cost is that
# `cur_speed` is a missing channel; that cost is named in the docstring.
VARIABLES = ["mlotst", "zos", "tos"]
VAR_CHANNEL = {"mlotst": 1, "zos": 2, "tos": C_SST}      # 1, 2, 39

# Generous North Atlantic box for the KD-tree. Generous because a target cell
# at the box edge must still have its true nearest source cell INSIDE the box —
# a tight box would silently hand edge cells a neighbour 200 km away, and the
# max-distance guard would then be the only thing that noticed.
BBOX = dict(south=LAT0 - 8.0, north=min(LAT1 + 8.0, 89.9),
            west=LON0 - 10.0, east=LON1 + 10.0)

R_EARTH_KM = 6371.0088
MAX_NN_KM = 40.0        # both grids are ~25 km; see the docstring


# ------------------------------------------------------------------ grid ----
def target_axes():
    """Family 3's axes, as float64 for the geometry."""
    lats = np.round(np.arange(NLAT) * DEG + LAT0, 6)
    lons = np.round(np.arange(NLON) * DEG + LON0, 6)
    assert lats[-1] == LAT1 and lons[-1] == LON1, (lats[-1], lons[-1])
    return lats, lons


def unit_vectors(lat, lon):
    """(lat, lon) in degrees -> unit vectors on the sphere, shape (..., 3).

    The whole reason the mapping is done in 3-D: on the unit sphere the
    Euclidean (chord) metric is a monotonic function of great-circle distance,
    so a KD-tree nearest neighbour IS the nearest neighbour on the globe — at
    the dateline, at the poles, and at every longitude in between. Degrees are
    neither: 1 deg of longitude is 111 km at the equator and 3.9 km at 88N.
    """
    la = np.radians(np.asarray(lat, np.float64))
    lo = np.radians(np.asarray(lon, np.float64))
    cl = np.cos(la)
    return np.stack([cl * np.cos(lo), cl * np.sin(lo), np.sin(la)], axis=-1)


def chord_to_km(d):
    """Chord length on the unit sphere -> great-circle distance in km."""
    return 2.0 * R_EARTH_KM * np.arcsin(np.clip(np.asarray(d) / 2.0, 0.0, 1.0))


def wrap_lon(lon):
    """Longitudes onto [-180, 180). CMIP6 stores them 0..360 as often as not."""
    return (np.asarray(lon, np.float64) + 180.0) % 360.0 - 180.0


def build_regrid_map(src_lat2d, src_lon2d, lats, lons, max_km=MAX_NN_KM,
                     bbox=None, verbose=True):
    """Nearest-neighbour index map from a curvilinear source onto (lats, lons).

    Returns (flat_src_index, dist_km) where `flat_src_index` has shape
    (len(lats) * len(lons),) and indexes the RAVELLED source (j,i) grid, so a
    regrid is `field.ravel()[flat_src_index].reshape(nlat, nlon)` and costs one
    gather per time slice.

    Built ONCE. The guard on max distance is a precondition that depends only
    on the two grids (ml/CLAUDE.md section 0.3): it fires here, where the
    inputs are all it has cost, and not after a 100 GB download.
    """
    from scipy.spatial import cKDTree
    bbox = bbox or BBOX
    src_lat2d = np.asarray(src_lat2d, np.float64)
    src_lon2d = wrap_lon(src_lon2d)
    keep = ((src_lat2d >= bbox["south"]) & (src_lat2d <= bbox["north"])
            & (src_lon2d >= bbox["west"]) & (src_lon2d <= bbox["east"]))
    # Tripolar grids repeat rows/columns at the fold and the wrap seam; NaNs in
    # the coordinate arrays would poison the tree, so they are excluded too.
    keep &= np.isfinite(src_lat2d) & np.isfinite(src_lon2d)
    sub = np.flatnonzero(keep.ravel())
    if sub.size == 0:
        sys.exit("REGRID: the North Atlantic bounding box selected no source "
                 "cells at all — the coordinate arrays are not what this "
                 "expects (wrong variable, or lon in a convention wrap_lon "
                 "does not handle).")
    if verbose:
        print(f"  source cells in NA box: {sub.size:,} of "
              f"{src_lat2d.size:,} ({100 * sub.size / src_lat2d.size:.1f}%)")
    src_vec = unit_vectors(src_lat2d.ravel()[sub], src_lon2d.ravel()[sub])
    tree = cKDTree(src_vec)
    glat, glon = np.meshgrid(lats, lons, indexing="ij")
    tgt_vec = unit_vectors(glat.ravel(), glon.ravel())
    d, i = tree.query(tgt_vec, k=1, workers=-1)
    dist_km = chord_to_km(d)
    flat = sub[i]
    if verbose:
        print(f"  nearest-neighbour distance: mean {dist_km.mean():.2f} km  "
              f"p99 {np.percentile(dist_km, 99):.2f} km  "
              f"max {dist_km.max():.2f} km")
        print(f"  distinct source cells used: "
              f"{len(np.unique(flat)):,} for {flat.size:,} target cells")
    if dist_km.max() > max_km:
        worst = int(np.argmax(dist_km))
        sys.exit(
            f"REGRID REFUSED: max nearest-neighbour distance "
            f"{dist_km.max():.1f} km > {max_km} km.\n"
            f"  worst target cell: lat {glat.ravel()[worst]:.2f} "
            f"lon {glon.ravel()[worst]:.2f}\n"
            f"Both grids are nominally 25 km, so this is not a resolution "
            f"mismatch — it is a broken mapping (flipped axis, degrees where "
            f"radians belong, a bounding box that clipped a corner). Fix it "
            f"before anything is downloaded.")
    return flat, dist_km


# ------------------------------------------------------------- catalogue ----
def fetch_catalogue(path=CATALOG, url=CATALOG_URL, refresh=False):
    """The published Pangeo CMIP6 catalogue (80 MB), cached on disk."""
    if refresh or not os.path.exists(path) or os.path.getsize(path) < 1 << 20:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        print(f"catalogue  fetching {url} …", flush=True)
        t = time.time()
        f3.fetch(url, path)
        print(f"catalogue  {os.path.getsize(path) / 1e6:.1f} MB in "
              f"{time.time() - t:.1f}s -> {path}")
    return path


def resolve_stores(path=CATALOG, models=MODELS, variables=VARIABLES):
    """(source_id, variable_id) -> zstore URL, resolved from the catalogue.

    Anonymous HTTPS: the catalogue's `gs://cmip6/...` is the same object as
    `https://storage.googleapis.com/cmip6/...`, and the bucket is world
    readable, so no credentials are involved anywhere in this file.
    """
    import pandas as pd
    df = pd.read_csv(path)
    out = {}
    for m in models:
        for v in variables:
            sel = df[(df.source_id == m["source_id"])
                     & (df.experiment_id == EXPERIMENT)
                     & (df.table_id == TABLE)
                     & (df.variable_id == v)
                     & (df.member_id == m["member_id"])]
            if sel.empty:
                sys.exit(f"CATALOGUE: no {EXPERIMENT}/{TABLE}/{v} for "
                         f"{m['source_id']} {m['member_id']}. The store moved "
                         f"or the member label changed; do not guess a path.")
            # More than one version can be listed; take the newest, and say so.
            sel = sel.sort_values("version")
            row = sel.iloc[-1]
            if len(sel) > 1:
                print(f"  ::warning:: {m['source_id']}/{v}: {len(sel)} "
                      f"versions listed, using v{row.version}")
            out[(m["source_id"], v)] = dict(
                zstore=row.zstore,
                url=row.zstore.replace("gs://", "https://storage.googleapis.com/"),
                version=str(row.version), member_id=row.member_id,
                grid_label=row.grid_label)
    return out


def open_store(url):
    """Open one zarr store read-only over anonymous HTTPS.

    `decode_times=False`: piControl has no real calendar, decoding it into
    cftime objects would only invite somebody to treat model-year 137 as 1987,
    and the raw offsets plus the units/calendar attributes are what the index
    should carry anyway.
    """
    import fsspec
    import xarray as xr
    return xr.open_zarr(fsspec.get_mapper(url), consolidated=True,
                        decode_times=False, mask_and_scale=True)


def coord_names(ds):
    """The 2-D (j,i) latitude/longitude arrays, whatever they are called.

    HadGEM3-GC31-MM calls them `latitude`/`longitude`; CNRM-CM6-1-HR calls them
    `lat`/`lon`. Checked at runtime rather than assumed, because getting this
    wrong silently is the whole failure mode this file is built against.
    """
    lat = lon = None
    for a, b in (("latitude", "longitude"), ("lat", "lon"),
                 ("nav_lat", "nav_lon")):
        if a in ds.coords and b in ds.coords and ds[a].ndim == 2:
            lat, lon = a, b
            break
    if lat is None:
        cand = {c: ds[c].dims for c in ds.coords if ds[c].ndim == 2}
        sys.exit(f"no 2-D lat/lon coordinate pair found. 2-D coords present: "
                 f"{cand}. This is a curvilinear grid; without its coordinate "
                 f"arrays there is nothing to regrid from.")
    return lat, lon


# ------------------------------------------------------------------- QC -----
def load_grep(path, var="thetao_oras", month_of_year=None):
    """The GREP/ORAS5 reference field on this exact grid, for the QC panel.

    Returns (field[281,481], lats, lons). `month_of_year` selects that month's
    climatology (mean over all its years); None gives the all-time mean. A
    month-matched comparison is the honest one: a free-running control run and
    a reanalysis share no year, but they do share a seasonal cycle, and
    comparing a model January against a 32-year mean would charge the model for
    the seasonal offset.
    """
    import xarray as xr
    ds = xr.open_dataset(path)
    da = ds[var]
    if "depth" in da.dims:
        da = da.isel(depth=0)
    if month_of_year is not None:
        sel = da["time.month"] == month_of_year
        da = da.isel(time=np.flatnonzero(sel.values))
    fld = da.mean("time", skipna=True).values.astype(np.float64)
    lats = ds["latitude"].values.astype(np.float64)
    lons = ds["longitude"].values.astype(np.float64)
    ds.close()
    return fld, lats, lons


def compare_fields(a, b):
    """Mask counts, overlap correlation, and mask disagreement between two
    fields on the same grid. NaN is 'land or otherwise unobserved' in both."""
    fa, fb = np.isfinite(a), np.isfinite(b)
    both = fa & fb
    n = int(both.sum())
    if n < 100:
        return dict(n_a=int(fa.sum()), n_b=int(fb.sum()), n_both=n, corr=np.nan,
                    rmse=np.nan, bias=np.nan, only_a=int((fa & ~fb).sum()),
                    only_b=int((fb & ~fa).sum()))
    x, y = a[both].astype(np.float64), b[both].astype(np.float64)
    corr = float(np.corrcoef(x, y)[0, 1])
    return dict(n_a=int(fa.sum()), n_b=int(fb.sum()), n_both=n, corr=corr,
                rmse=float(np.sqrt(np.mean((x - y) ** 2))),
                bias=float(np.mean(x - y)),
                only_a=int((fa & ~fb).sum()), only_b=int((fb & ~fa).sum()))


def qc_png(path, cmip, grep, lats, lons, title_a, title_b):
    """Three panels: regridded CMIP6, GREP, and the mask difference.

    `origin="lower"` on all three, because these axes run south-up: row 0 is
    the equator. Getting that wrong flips the Atlantic and makes a broken
    regrid look merely unfamiliar.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    ext = [lons[0], lons[-1], lats[0], lats[-1]]
    fin_a, fin_b = np.isfinite(cmip), np.isfinite(grep)
    diff = np.where(fin_a & ~fin_b, 1.0, np.where(fin_b & ~fin_a, -1.0, 0.0))
    lo = float(np.nanpercentile(np.concatenate(
        [cmip[fin_a].ravel(), grep[fin_b].ravel()]), 1))
    hi = float(np.nanpercentile(np.concatenate(
        [cmip[fin_a].ravel(), grep[fin_b].ravel()]), 99))
    fig, ax = plt.subplots(1, 3, figsize=(16.5, 5.0), constrained_layout=True)
    for a, fld, ttl, kw in (
            (ax[0], cmip, title_a, dict(cmap="RdYlBu_r", vmin=lo, vmax=hi)),
            (ax[1], grep, title_b, dict(cmap="RdYlBu_r", vmin=lo, vmax=hi)),
            (ax[2], diff, "mask difference", dict(cmap="bwr", vmin=-1, vmax=1))):
        im = a.imshow(fld, origin="lower", extent=ext, aspect="auto", **kw)
        a.set_title(ttl, fontsize=10)
        a.set_xlabel("lon"); a.set_ylabel("lat")
        fig.colorbar(im, ax=a, shrink=0.85)
    ax[2].set_title("mask difference  (+1 CMIP6-only ocean, -1 GREP-only)",
                    fontsize=9)
    fig.savefig(path, dpi=110)
    plt.close(fig)
    return path


# -------------------------------------------------------------- preflight ---
def preflight_write(x_path, shape, dtype=np.float16, keep=False):
    """PROVE THE OUTPUT CAN BE WRITTEN AND READ BACK BEFORE FETCHING ANYTHING.

    `ml/fetch_glorys_daily.py:hf_preflight` uploads a probe, downloads it back
    and compares sha256 before a single byte of GLORYS is fetched, because a
    backup that has never been restored is not a backup (ml/CLAUDE.md 0.2).
    The same discipline applies to a 104 GB local write: allocate the full
    sidecar, stamp a known pattern into the first and last rows, close it,
    reopen it, and compare. If the disk is short, the filesystem caps file
    size, or the path is wrong, this dies now instead of after two days of
    downloading.
    """
    need = int(np.prod(shape)) * np.dtype(dtype).itemsize
    st = os.statvfs(os.path.dirname(x_path) or ".")
    free = st.f_bavail * st.f_frsize
    print(f"preflight  need {need / 1e9:.1f} GB, free {free / 1e9:.1f} GB")
    if free < need * 1.02:
        sys.exit(f"PREFLIGHT: {free / 1e9:.1f} GB free but the tensor needs "
                 f"{need / 1e9:.1f} GB. Refusing to start a download that "
                 f"cannot land.")
    X = np.lib.format.open_memmap(x_path, mode="w+", dtype=dtype, shape=shape)
    probe = np.arange(shape[-1], dtype=dtype)
    X[0, 0, 0, :] = probe
    X[-1, -1, -1, :] = probe
    X.flush()
    del X
    back = np.load(x_path, mmap_mode="r")
    ok = (np.array_equal(np.asarray(back[0, 0, 0, :]), probe)
          and np.array_equal(np.asarray(back[-1, -1, -1, :]), probe)
          and back.shape == tuple(shape))
    del back
    if not ok:
        sys.exit("PREFLIGHT FAILED: the sidecar did not read back what was "
                 "written. Refusing to fetch.")
    print(f"preflight  OK: {os.path.basename(x_path)} allocated "
          f"{os.path.getsize(x_path) / 1e9:.1f} GB, probe rows round-tripped")
    if not keep:
        os.remove(x_path)


def preflight_hf(repo, scratch):
    """Optional Hub mirror, using the SAME round-trip proof as the GLORYS
    fetcher — upload, download back, compare sha256 — before anything is
    fetched. No credential ever reaches a command line: the token comes from
    $HF_TOKEN or ~/.hf_token, as `hf_connect` already does it."""
    from fetch_glorys_daily import hf_connect, hf_preflight
    api, repo_id, tok = hf_connect(repo)
    hf_preflight(api, repo_id, tok, scratch)
    return api, repo_id


# ------------------------------------------------------------- self-test ----
def self_test():
    """The regrid math, against a source grid whose answer is known. No
    network, no data cache — this is what makes the geometry testable at all."""
    lats, lons = target_axes()
    print(f"axes       {len(lats)}x{len(lons)}  lat {lats[0]}..{lats[-1]}  "
          f"lon {lons[0]}..{lons[-1]}")
    assert len(lats) == NLAT and len(lons) == NLON

    # 1. A source that IS the target, shifted by nothing: every target cell must
    #    map to itself at 0 km.
    sl, so = np.meshgrid(lats, lons, indexing="ij")
    flat, d = build_regrid_map(sl, so, lats, lons, verbose=False)
    assert np.array_equal(flat, np.arange(NLAT * NLON)), "identity map broken"
    assert d.max() < 1e-6, d.max()
    print(f"self-test  identity map exact, max {d.max():.2e} km")

    # 2. A source in 0..360 longitude convention must give the same answer:
    #    this is the dateline/seam behaviour the unit-vector detour buys.
    flat2, d2 = build_regrid_map(sl, so % 360.0, lats, lons, verbose=False)
    assert np.array_equal(flat2, flat), "0..360 convention changed the map"
    print("self-test  lon convention 0..360 gives the identical map")

    # 3. A half-cell-shifted source: every distance must be a real half-cell,
    #    never zero, and never more than a cell.
    flat3, d3 = build_regrid_map(sl + 0.125, so, lats, lons, verbose=False)
    exp = 0.125 * np.pi / 180.0 * R_EARTH_KM
    assert abs(d3.max() - exp) < 0.5, (d3.max(), exp)
    assert d3.min() > exp - 0.5, d3.min()
    print(f"self-test  half-cell shift measured {d3.mean():.3f} km "
          f"(expected {exp:.3f})")

    # 4. The refusal actually refuses: a source 2 degrees away must not pass.
    try:
        build_regrid_map(sl + 2.0, so, lats, lons, verbose=False)
    except SystemExit as e:
        assert "REGRID REFUSED" in str(e), str(e)
        print("self-test  a 2-degree offset is refused, as it must be")
    else:
        raise AssertionError("a 2-degree offset was NOT refused")

    # 5. chord_to_km against a closed form: antipodes are half the circumference.
    assert abs(chord_to_km(2.0) - np.pi * R_EARTH_KM) < 1e-6
    assert abs(chord_to_km(0.0)) < 1e-12
    print("self-test  chord_to_km exact at 0 and at the antipode")
    # the sparse contract round-trips, and refuses a mismatched index
    xs = np.arange(2 * 3 * 4 * 3, dtype=np.float32).reshape(2, 3, 4, 3)
    full = expand_to_full(xs, [1, 2, 39], 40)
    assert full.shape == (2, 3, 4, 40)
    for k, ci in enumerate([1, 2, 39]):
        assert np.array_equal(full[..., ci], xs[..., k])
    assert np.isnan(full[..., 0]).all() and np.isnan(full[..., 3]).all()
    assert np.isnan(full[..., [c for c in range(40)
                              if c not in (1, 2, 39)]]).all()
    for bad in ((xs, [1, 2], 40), (xs, [1, 2, 40], 40), (xs, [-1, 2, 39], 40)):
        try:
            expand_to_full(*bad)
        except ValueError:
            pass
        else:
            raise AssertionError(f"expand_to_full accepted {bad[1]}")
    print("  sparse expand: round-trip exact, 37 channels NaN, bad index refused")
    print("\nself-test PASSED")


# ----------------------------------------------------------------- probe ----
def expand_to_full(x, channel_index, n_full):
    """Scatter a SPARSE sidecar back into the full 40-channel layout.

    The inverse of the sparse write: everything the corpus never observed
    comes back as NaN, which is what `isfinite` downstream already treats as a
    missing token. Cheap enough to do per batch at read time, which is the
    point — the absence of 37 channels is a fact about the corpus, not 96 GB
    of data.

        X[..., channel_index[k]] = x[..., k]
    """
    x = np.asarray(x)
    ci = np.asarray(channel_index, dtype=int)
    if x.shape[-1] != ci.size:
        raise ValueError(f"sidecar has {x.shape[-1]} channels but "
                         f"channel_index names {ci.size}")
    if ci.size and (ci.min() < 0 or ci.max() >= n_full):
        raise ValueError(f"channel_index out of range for n_full={n_full}")
    out = np.full(x.shape[:-1] + (int(n_full),), np.nan, np.float32)
    out[..., ci] = x
    return out


def probe(a):
    """Catalogue -> stores -> regrid map -> ONE time chunk -> QC. Writes no
    tensor. This is the mode that runs in the sandbox."""
    lats, lons = target_axes()
    print(f"axes       {len(lats)}x{len(lons)}  lat {lats[0]}..{lats[-1]}  "
          f"lon {lons[0]}..{lons[-1]}  (south-up)")
    check_grid(lats, lons)

    fetch_catalogue(a.catalog, refresh=a.refresh_catalog)
    stores = resolve_stores(a.catalog)
    print("\nstores")
    meta = {}
    for m in MODELS:
        for v in VARIABLES:
            s = stores[(m["source_id"], v)]
            ds = open_store(s["url"])
            nt = int(ds.sizes["time"])
            da = ds[v]
            meta[(m["source_id"], v)] = dict(nt=nt, shape=tuple(da.shape),
                                             units=da.attrs.get("units"))
            print(f"  {m['source_id']:16s} {v:7s} T={nt:5d} "
                  f"shape={tuple(da.shape)} units={da.attrs.get('units')!r} "
                  f"chunks={da.encoding.get('chunks')}")
            print(f"      {s['zstore']}")
            ds.close()

    mdl = a.model or MODELS[0]["source_id"]
    print(f"\nregrid map  {mdl} (built once, reused for every time slice)")
    ds = open_store(stores[(mdl, "tos")]["url"])
    cy, cx = coord_names(ds)
    print(f"  2-D coordinates: {cy!r}/{cx!r} over {ds[cy].dims}")
    src_lat = ds[cy].values
    src_lon = ds[cx].values
    t0 = time.time()
    flat, dist = build_regrid_map(src_lat, src_lon, lats, lons,
                                  max_km=a.max_nn_km)
    print(f"  built in {time.time() - t0:.1f}s")

    ti = a.time_index
    print(f"\nslice      {mdl} tos time index {ti} "
          f"(one chunk: {ds['tos'].encoding.get('chunks')})")
    t0 = time.time()
    raw = ds["tos"].isel(time=ti).values.astype(np.float64)
    print(f"  fetched {raw.nbytes / 1e6:.1f} MB (decoded slice) in "
          f"{time.time() - t0:.1f}s")
    tval = float(np.asarray(ds["time"].values)[ti])
    tunits = ds["time"].attrs.get("units")
    tcal = ds["time"].attrs.get("calendar")
    ds.close()
    moy = ti % 12 + 1          # both stores start at January of model-year 1
    fld = raw.ravel()[flat].reshape(NLAT, NLON)
    fin = np.isfinite(fld)
    print(f"  regridded  {fld.shape}  ocean cells {int(fin.sum()):,} / "
          f"{fld.size:,}  min {np.nanmin(fld):.2f} max {np.nanmax(fld):.2f} "
          f"mean {np.nanmean(fld):.2f} degC")
    print(f"  source time {tval} [{tunits}] calendar {tcal} -> "
          f"month-of-year {moy} (index {ti})")

    if not os.path.exists(a.grep):
        print(f"  ::warning:: {a.grep} absent — the decisive comparison "
              f"against a real ocean is SKIPPED, and the geometry is "
              f"UNCHECKED beyond its own self-consistency.")
        return 0
    print(f"\ncompare    {a.grep} thetao_oras, month {moy} climatology")
    grep, glats, glons = load_grep(a.grep, month_of_year=moy)
    if not (np.allclose(glats, lats) and np.allclose(glons, lons)):
        sys.exit("GREP file is not on this grid — the comparison would be "
                 "meaningless. lat/lon do not match the target axes.")
    st = compare_fields(fld, grep)
    print(f"  ocean cells   CMIP6 {st['n_a']:,}   GREP {st['n_b']:,}   "
          f"both {st['n_both']:,}")
    print(f"  mask disagree CMIP6-only {st['only_a']:,}  GREP-only "
          f"{st['only_b']:,}  "
          f"({100 * (st['only_a'] + st['only_b']) / fld.size:.2f}% of cells)")
    print(f"  spatial corr  {st['corr']:.4f}   rmse {st['rmse']:.2f} degC   "
          f"bias {st['bias']:+.2f} degC")
    print(f"  GREP min/max  {np.nanmin(grep):.2f} / {np.nanmax(grep):.2f} degC")

    if a.png:
        os.makedirs(os.path.dirname(a.png) or ".", exist_ok=True)
        qc_png(a.png, fld, grep,
               lats, lons,
               f"{mdl} piControl tos, t={ti} (regridded)",
               f"GREP thetao_oras, month {moy} climatology")
        print(f"  wrote {a.png}")

    ok = st["corr"] >= a.min_corr and \
        (st["only_a"] + st["only_b"]) / fld.size <= a.max_mask_frac
    print(f"\nPROBE {'PASSED' if ok else 'FAILED'}  "
          f"(corr {st['corr']:.4f} vs >= {a.min_corr}, mask disagreement "
          f"{100 * (st['only_a'] + st['only_b']) / fld.size:.2f}% vs "
          f"<= {100 * a.max_mask_frac:.1f}%)")
    if not ok:
        print("The geometry is wrong. Do not build. Check, in this order: the "
              "2-D coordinate arrays actually read, the lon convention, the "
              "row order (south-up), and the bounding box.")
    return 0 if ok else 1


# ----------------------------------------------------------------- build ----
def build(a):
    lats, lons = target_axes()
    print(f"axes       {len(lats)}x{len(lons)}  lat {lats[0]}..{lats[-1]}  "
          f"lon {lons[0]}..{lons[-1]}  (south-up)")
    check_grid(lats, lons)
    chans = list(CHANS_R2)
    nchan = len(chans)

    fetch_catalogue(a.catalog, refresh=a.refresh_catalog)
    stores = resolve_stores(a.catalog)

    # Plan the axis BEFORE fetching: rows, per-source spans, bytes.
    plan, T = [], 0
    for m in MODELS:
        ds = open_store(stores[(m["source_id"], "tos")]["url"])
        nt = int(ds.sizes["time"])
        tvals = np.asarray(ds["time"].values)
        tunits = str(ds["time"].attrs.get("units"))
        tcal = str(ds["time"].attrs.get("calendar"))
        ds.close()
        if a.max_times:
            nt = min(nt, a.max_times)
        plan.append(dict(model=m["source_id"], member=m["member_id"], nt=nt,
                         row0=T, tvals=tvals[:nt], tunits=tunits, cal=tcal))
        T += nt
    for p in plan:
        print(f"  {p['model']:16s} {p['member']} rows {p['row0']}.."
              f"{p['row0'] + p['nt'] - 1}  ({p['nt']} months = "
              f"{p['nt'] / 12:.1f} model-years, calendar {p['cal']})")

    H, W = NLAT, NLON
    # STORAGE. Three of the forty channels are live, so a dense [T,H,W,40]
    # sidecar is 92.5% NaN — 103.8 GB of which 96 GB is the literal absence of
    # Argo and NCEP over a model run that never had them. The daily tensor's
    # 165.6 GB is already a millstone this programme carries (E-038: it fits on
    # no box we can rent, and `temporal.py` cannot open it at all), and this
    # corpus is meant to GROW — more models, and currents once the OPeNDAP
    # route is built. So the default layout is SPARSE: only the live channels
    # are stored, with `channel_index` recording where each belongs in the
    # 40-channel layout and `n_channels_full` recording the layout itself. A
    # reader reconstitutes it with `expand_to_full()` below, which is a
    # three-line scatter into a NaN array. --layout dense is kept for anyone
    # who wants the naive artefact.
    live_idx = sorted(VAR_CHANNEL.values())
    slot = {ci: k for k, ci in enumerate(live_idx)}
    sparse = a.layout == "sparse"
    nchan_out = len(live_idx) if sparse else nchan
    for name, bpe in (("float32", 4), ("float16", 2)):
        print(f"dense      [{T}, {H}, {W}, {nchan}] {name}: "
              f"{T * H * W * nchan * bpe / 1e9:.1f} GB")
    print(f"sparse     [{T}, {H}, {W}, {len(live_idx)}] float16: "
          f"{T * H * W * len(live_idx) * 2 / 1e9:.1f} GB"
          f"   <- {'CHOSEN' if sparse else 'available via --layout sparse'}")
    live = len(VAR_CHANNEL)
    print(f"live       {live} of {nchan} channels "
          f"({', '.join(f'{v}->{chans[c]}' for v, c in VAR_CHANNEL.items())})"
          f"; the other {nchan - live} are missing tokens over the whole corpus")
    if a.dry_run:
        st = os.statvfs(CACHE if os.path.isdir(CACHE) else HERE)
        print(f"disk       {st.f_bavail * st.f_frsize / 1e9:.1f} GB free")
        print("\n--dry-run: nothing built.")
        return 0

    os.makedirs(CACHE, exist_ok=True)
    x_path = a.out[:-4] + "_X.npy"
    if a.hf_repo:
        preflight_hf(a.hf_repo, CACHE)
    preflight_write(x_path, (T, H, W, nchan_out))

    X = np.lib.format.open_memmap(x_path, mode="w+", dtype=np.float16,
                                  shape=(T, H, W, nchan_out))
    # Missing is the default state of every cell, exactly as the family-4
    # builder does it: the fill_* passes write only what they observed, and
    # `isfinite` downstream is what counts a live value.
    for i in range(0, T, 64):
        X[i:i + 64] = np.float16(np.nan)

    prov, rows_written = [], 0
    for p, m in zip(plan, MODELS):
        mdl = p["model"]
        print(f"\n{mdl}")
        ds0 = open_store(stores[(mdl, "tos")]["url"])
        cy, cx = coord_names(ds0)
        flat, dist = build_regrid_map(ds0[cy].values, ds0[cx].values,
                                      lats, lons, max_km=a.max_nn_km)
        ds0.close()
        prov.append(dict(
            source_id=mdl, member_id=p["member"], experiment_id=EXPERIMENT,
            table_id=TABLE, row0=p["row0"], nrows=p["nt"],
            calendar=p["cal"], time_units=p["tunits"],
            max_nn_km=float(dist.max()), mean_nn_km=float(dist.mean()),
            stores={v: stores[(mdl, v)]["zstore"] for v in VARIABLES},
            versions={v: stores[(mdl, v)]["version"] for v in VARIABLES}))
        for v in VARIABLES:
            ci = VAR_CHANNEL[v]
            ds = open_store(stores[(mdl, v)]["url"])
            da = ds[v]
            enc = da.encoding.get("chunks")
            step = int(enc[0]) if enc else 12
            t0 = time.time()
            for s in range(0, p["nt"], step):
                e = min(s + step, p["nt"])
                blk = da.isel(time=slice(s, e)).values.astype(np.float32)
                for k in range(e - s):
                    f = blk[k].ravel()[flat].reshape(H, W)
                    if v == "mlotst":
                        # log10, as family 4's log_mld: measured against family
                        # 3 there, and the same transform must be applied here
                        # or the channel would mean two different things in the
                        # pretraining and fine-tuning corpora.
                        with np.errstate(invalid="ignore", divide="ignore"):
                            f = np.where(f > 0, np.log10(np.maximum(f, 1e-6)),
                                         np.nan)
                    X[p["row0"] + s + k, :, :,
                      slot[ci] if sparse else ci] = f.astype(np.float16)
                if s % (step * 20) == 0:
                    print(f"  {v}: {e}/{p['nt']} "
                          f"({(time.time() - t0) / max(e, 1) * 1000:.0f} ms/step)",
                          flush=True)
            ds.close()
            print(f"  {v} -> channel {ci} ({chans[ci]}): {p['nt']} rows in "
                  f"{time.time() - t0:.0f}s", flush=True)
        rows_written += p["nt"]

    X.flush()
    del X

    # Time labels. There is NO real calendar here, so no `months` of the family
    # 3/4/5 kind is written: `src_time` is the model's own raw offset and
    # `month_of_year` is the only calendar fact that survives — both control
    # runs start at January of their model-year 1 and are strictly monthly.
    src = np.concatenate([np.full(p["nt"], i, np.int16)
                          for i, p in enumerate(plan)])
    src_time = np.concatenate([np.asarray(p["tvals"], np.float64) for p in plan])
    moy = np.concatenate([np.arange(p["nt"], dtype=np.int16) % 12 + 1
                          for p in plan])
    yr = np.concatenate([np.arange(p["nt"], dtype=np.int32) // 12 for p in plan])
    meta = dict(
        lats=lats.astype(np.float32), lons=lons.astype(np.float32),
        chan=np.array(chans), bins=np.arange(T, dtype=np.int64),
        src=src, src_time=src_time, month_of_year=moy, model_year=yr,
        sources=np.array([p["source_id"] for p in prov]),
        provenance=np.array(json.dumps(prov, indent=1)),
        window=np.array("na025"), cadence=np.array("monthly"),
        recipe=np.array(RECIPE_REV),
        live_channels=np.array([VAR_CHANNEL[v] for v in VARIABLES]),
        # SPARSE CONTRACT. channel_index[k] is where sidecar channel k belongs
        # in the n_channels_full layout; every other full-layout channel is
        # missing for the whole corpus. Dense builds set channel_index to the
        # identity so one reader handles both.
        layout=np.array(a.layout),
        channel_index=np.array(live_idx if sparse else list(range(nchan)),
                               dtype=np.int32),
        n_channels_full=np.array(nchan, dtype=np.int32),
        labelled=np.array(False),
        note=np.array("CMIP6 piControl. UNLABELLED pretraining corpus: free-"
                      "running control runs have no real calendar and no "
                      "truth series. 3 of 40 channels are live."))
    from tensor_io import save_tensor
    save_tensor(a.out, np.load(x_path, mmap_mode="r"), **meta)
    print(f"\nwrote {a.out} + {os.path.basename(x_path)}  "
          f"[T={T} H={H} W={W} C={nchan_out}] float16 {a.layout}  "
          f"{os.path.getsize(x_path) / 1e9:.2f} GB  recipe={RECIPE_REV}")
    print(f"rows {rows_written}; NO truth series attached, by design.")

    if a.hf_repo:
        # PUBLISH WHEN THE ARTEFACT EXISTS (ml/CLAUDE.md 5.20), and verify the
        # restore — an upload returning 200 is not evidence the bytes come
        # back (0.2). Without this the corpus lives only on one rented disk,
        # which is how #503's dump trajectories were lost.
        import shutil
        from huggingface_hub import hf_hub_download
        from fetch_glorys_daily import hf_connect, sha256
        api, repo, tok = hf_connect(a.hf_repo)
        vf = os.path.join(CACHE, "vf")
        for local in (a.out, x_path):
            fname = "cmip6/" + os.path.basename(local)
            src = sha256(local)
            gb = os.path.getsize(local) / 1e9
            print(f"  uploading {fname} ({gb:.2f} GB) …", flush=True)
            api.upload_file(path_or_fileobj=local, path_in_repo=fname,
                            repo_id=repo, repo_type="dataset",
                            commit_message=f"family6 CMIP6 pretraining corpus "
                                           f"({RECIPE_REV})")
            back = hf_hub_download(repo, fname, repo_type="dataset",
                                   token=tok, cache_dir=vf)
            ok = sha256(back) == src
            shutil.rmtree(vf, ignore_errors=True)
            if not ok:
                raise RuntimeError(f"{fname} restored with a DIFFERENT sha256 "
                                   f"— the backup is not trustworthy")
            print(f"  {fname}: uploaded + restore-verified ({src[:12]})",
                  flush=True)
    else:
        print("::warning:: no --hf-repo — the corpus exists only on this "
              "disk. A rented box is not an archive.")
    return 0


def main():
    ap = argparse.ArgumentParser(
        description="CMIP6 piControl -> family-3 grid, unlabelled pretraining")
    ap.add_argument("--probe", action="store_true",
                    help="catalogue lookup + regrid map + ONE time chunk + QC, "
                         "writing no tensor. The sandbox mode.")
    ap.add_argument("--self-test", action="store_true",
                    help="exercise the regrid geometry against known answers; "
                         "no network, no data cache")
    ap.add_argument("--dry-run", action="store_true",
                    help="axis and byte arithmetic only")
    ap.add_argument("--layout", choices=("sparse", "dense"), default="sparse",
                    help="sparse (default): store ONLY the live channels, with "
                         "channel_index recording where each belongs in the "
                         "40-channel layout — 7.8 GB at the full corpus. "
                         "dense: the naive [T,H,W,40], 103.8 GB of which 92.5%% "
                         "is the literal absence of Argo and NCEP over a model "
                         "run that never had them.")
    ap.add_argument("--out", default=OUT_NPZ)
    ap.add_argument("--catalog", default=CATALOG)
    ap.add_argument("--refresh-catalog", action="store_true")
    ap.add_argument("--model", default=None,
                    help="probe only: which model to build the map from "
                         f"(default {MODELS[0]['source_id']})")
    ap.add_argument("--time-index", type=int, default=0,
                    help="probe only: which time step to pull")
    ap.add_argument("--max-times", type=int, default=0,
                    help="build only the first N months of EACH model — for "
                         "exercising the write path without 104 GB")
    ap.add_argument("--max-nn-km", type=float, default=MAX_NN_KM,
                    help="refuse a regrid map whose worst target cell is "
                         "farther than this from its source cell")
    ap.add_argument("--grep", default="/home/claude/hindcast/"
                                      "grep_oras_na_surface_monthly.nc",
                    help="reanalysis reference on this same grid, for the "
                         "land-mask and large-scale-pattern check")
    ap.add_argument("--png", default="/home/claude/hindcast/"
                                     "cmip6_regrid_check.png")
    ap.add_argument("--min-corr", type=float, default=0.90)
    ap.add_argument("--max-mask-frac", type=float, default=0.06)
    ap.add_argument("--hf-repo", default="",
                    help="optional Hub mirror; the token comes from $HF_TOKEN "
                         "or ~/.hf_token and never from a command line. The "
                         "upload/download/sha round trip is proved BEFORE any "
                         "CMIP6 byte is fetched.")
    a = ap.parse_args()
    if a.self_test:
        self_test()
        return 0
    if a.probe:
        return probe(a)
    return build(a)


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
"""Writes `data/family10_fixture/` — the family-10 and derived stores the Data
tab's reader is tested on (E-084, the multi-registry extension).

    python3 tests/make_family10_fixture.py          # rewrites the fixture

PLAIN ENGLISH. Family 10 is the Data tab's second family: the four groups of
the global tensor ("family 7.2" — one bin-major .npy per group, every value
z-scored, the (mean, sd) in the site's data/family7_index.json) and the
tier-P point stores of family 10.1 / 10.2 plus family 8's Argo store. The
"derived maps" are the fishing-effort month-major grid and the model
climatology. This script builds a tiny copy of each layout with the REAL
writers wherever the repo has one, so the reader is tested on bytes the
writers produced:

  tensors/fx7/fx7_X_fxg.npy    ml/tensor_io.save_tensor (the family-7 builder's
  tensors/fx7/fx7_X_fxrg.npy   own sidecar writer): [T, H, W, C] float16,
                               z-scored, on a 10° grid (19 x 36, south first);
                               fxg six five-day bins across a month and a year
                               boundary, fxrg three MONTHLY frames filed under
                               the pentad holding each month's 15th, with six
                               levelled channels (rg_t10 .. rg_s50)
  tensors/fxargo/              ml/family8_store.write_synthetic_store: the
                               SCHEMA-1 layout — time_days float32, temp and
                               psal as separate [N, 3] blocks, wmo, no qc
  tensors/fxneg/               the family-10 column assembler (the same path
                               tests/make_family1_fixture.py's point store
                               uses): schema 2 with rows from 1980, i.e.
                               NEGATIVE five-day bins
  fishing/fishing_grid.npy     np.save of [M, H, W, 2] float32 month-major, the
                               shape ml/build_family10_stores.py publishes
  clim/fxg/clim.npy            np.save of [12, C, H, W] float32 z-units, the
                               shape ml/export_family7_clim.py publishes

plus `family10.json` (the live registry's shape), `family7_index.json`,
`fishing_index.json`, `clim_index.json` (the site's indexes, URLs relative to
the index) and `expected.json` — selections and their truth computed here with
numpy, independently of the JavaScript. Deterministic; < 1.5 MB.
"""
import datetime as dt
import json
import os
import shutil
import sys
import tempfile

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "ml"))
sys.path.insert(0, HERE)

import family8_store as f8                                      # noqa: E402
import make_family1_fixture as m1                               # noqa: E402
import tensor_io                                                # noqa: E402

OUT = os.path.join(ROOT, "data", "family10_fixture")
EPOCH = dt.datetime(1982, 1, 1)
EPOCH_UNIX = 378691200
BIN_S = 432000
STEP = 10.0
NY, NX = 19, 36                       # lat -90..90, lon -180..170, south first
LAT = -90.0 + np.arange(NY) * STEP
LON = -180.0 + np.arange(NX) * STEP

G_BINS = list(range(2044, 2050))      # 2009-12-25 .. 2010-01-24
G_CH = [("sst", "°C", "Sea-surface temperature", (15.0, 8.0)),
        ("t2m", "°C", "Air temperature at 2 m", (10.0, 12.0)),
        ("ssh", "m", "Sea surface height", (-0.2, 0.7))]
RG_CH = [(f"rg_{v}{p}", u, f"Ocean {n} at {p} dbar", mu)
         for v, u, n, mu in (("t", "°C", "temperature", (12.0, 6.0)),
                             ("s", "PSU", "salinity", (34.8, 0.6)))
         for p in (10, 30, 50)]
RG_MONTHS = [(2010, 1), (2010, 2), (2010, 3)]


def bin_of_date(y, m, d):
    return ((dt.date(y, m, d) - EPOCH.date()).days) // 5


def g_phys(b):
    """The physical field of bin b, [H, W, C], NaN over the 'land' patch."""
    rng = np.random.default_rng(b)
    LA, LO = np.meshgrid(LAT, LON, indexing="ij")
    a = np.empty((NY, NX, 3))
    a[..., 0] = 28 - 0.3 * np.abs(LA) + rng.normal(0, 0.5, LA.shape)
    a[..., 1] = 25 - 0.4 * np.abs(LA) + rng.normal(0, 1.0, LA.shape) + 0.2 * (b - 2044)
    a[..., 2] = 0.3 * np.sin(np.radians(LO)) + rng.normal(0, 0.05, LA.shape)
    land = (LO >= 10) & (LO <= 40) & (LA >= -30) & (LA <= 30)
    a[land, 0] = np.nan                                  # sst only over sea
    a[land & (LA > 0), 2] = np.nan
    return a


def rg_phys(i):
    rng = np.random.default_rng(700 + i)
    LA = np.meshgrid(LAT, LON, indexing="ij")[0]
    a = np.empty((NY, NX, 6))
    for k, p in enumerate((10, 30, 50)):
        a[..., k] = 20 - 0.2 * np.abs(LA) - 0.05 * p + rng.normal(0, 0.3, LA.shape) + i
        a[..., 3 + k] = 34.5 + 0.01 * p + rng.normal(0, 0.1, LA.shape)
    a[np.abs(LA) > 70] = np.nan
    return a


def zscore(a, norms):
    z = np.empty(a.shape, np.float16)
    for k, (mu, sd) in enumerate(norms):
        z[..., k] = ((a[..., k] - mu) / sd).astype(np.float16)
    return z


def grid_json():
    return {"lat0": -90.0, "lon0": -180.0, "nx": NX, "ny": NY,
            "south_first": True, "step": STEP, "wrap": True}


# ------------------------------------------------------------ point stores --
ARGO_LEVELS = [10.0, 30.0, 50.0]


def argo_rows():
    rng = np.random.default_rng(81)
    n = 400
    d0 = (dt.date(2009, 12, 20) - EPOCH.date()).days
    d1 = (dt.date(2010, 2, 20) - EPOCH.date()).days
    days = rng.uniform(d0, d1, n)
    lat = rng.uniform(-60, 60, n)
    lon = rng.uniform(-180, 180, n)
    temp = np.stack([20 - 0.2 * np.abs(lat) - 0.05 * p + rng.normal(0, 0.5, n)
                     for p in ARGO_LEVELS], 1)
    psal = np.stack([34.5 + 0.01 * p + rng.normal(0, 0.1, n) for p in ARGO_LEVELS], 1)
    temp[rng.random(temp.shape) < 0.1] = np.nan
    return np.floor(days / 5).astype(np.int64), days, lat, lon, temp, psal


class FxNeg(m1.b10.SourceAdapter):
    """Drifter-like rows from 1980 — negative bins — through the real assembler."""
    store = "fxneg"
    title = "fixture: drifter-like reports from 1980 (negative bins)"
    family = "1gf"
    distribution = "public"
    licence = m1.lic()
    channels = (("u", "m s-1", -3.0, 3.0), ("sst", "degC", -2.5, 40.0))
    per_year = True
    first_year = 1980
    sources = ("synthetic",)
    smoke_window = ("1980-01-01", "1980-04-30")

    def index(self, ctx):
        return {"dataset": "fixture"}

    def fetch_year(self, ctx, year):
        rng = np.random.default_rng(1980)
        n = 600
        lo = (dt.datetime(1980, 1, 1) - EPOCH).total_seconds()
        hi = (dt.datetime(1980, 5, 1) - EPOCH).total_seconds() - 1
        t = rng.integers(int(lo), int(hi) + 1, n).astype(np.int64)
        lat = rng.uniform(-50, 50, n)
        lon = rng.uniform(-180, 180, n)
        v = np.stack([rng.normal(0, 0.4, n), rng.uniform(5, 30, n)], 1)
        platform = rng.integers(1, 30, n).astype(np.int64)
        qc = np.ones(n, np.int64)
        y0 = (dt.datetime(int(year), 1, 1) - EPOCH).total_seconds()
        y1 = (dt.datetime(int(year) + 1, 1, 1) - EPOCH).total_seconds()
        k = (t >= max(y0, ctx.t_lo)) & (t < y1) & (t <= ctx.t_hi)
        rows = self.pack(t[k], lat[k], lon[k], v[k], platform[k], qc[k])
        yield str(year), rows, {"rows": int(k.sum())}


def neg_rows(store_dir):
    t = np.load(os.path.join(store_dir, "time_s.npy")).astype(np.int64)
    lat = np.load(os.path.join(store_dir, "lat.npy"))
    lon = np.load(os.path.join(store_dir, "lon.npy"))
    v = np.load(os.path.join(store_dir, "values.npy"))
    return t, lat, lon, v


# ---------------------------------------------------------------- truth ---
def box_idx(box):
    rows = np.where((LAT >= box["s"]) & (LAT <= box["n"]))[0]
    if box["w"] <= box["e"]:
        cols = np.where((LON >= box["w"]) & (LON <= box["e"]))[0]
    else:
        cols = np.concatenate([np.where(LON >= box["w"])[0], np.where(LON <= box["e"])[0]])
    return rows, cols


def as_list(a):
    return [None if not np.isfinite(x) else float(x) for x in np.asarray(a, np.float64).ravel()]


def main():
    tmp = tempfile.mkdtemp(prefix="f10fixture_")
    try:
        if os.path.isdir(OUT):
            shutil.rmtree(OUT)
        os.makedirs(OUT)
        # --- the two tensor groups, through tensor_io's own writer
        g_norm = [mu for _, _, _, mu in G_CH]
        rg_norm = [mu for _, _, _, mu in RG_CH]
        Xg = np.stack([zscore(g_phys(b), g_norm) for b in G_BINS])
        Xrg = np.stack([zscore(rg_phys(i), rg_norm) for i in range(len(RG_MONTHS))])
        tdir = os.path.join(OUT, "tensors", "fx7")
        os.makedirs(tdir)
        paths = tensor_io.save_tensor(os.path.join(tdir, "fx7.npz"), {"fxg": Xg, "fxrg": Xrg},
                                      note=np.array("fixture"))
        os.remove(os.path.join(tdir, "fx7.npz"))         # the meta npz is not read
        rg_bins = [bin_of_date(y, m, 15) for y, m in RG_MONTHS]

        def grp_entry(name, X, chans, live, extra):
            fn = os.path.basename(paths[name])
            hdr = np.lib.format.read_magic(open(paths[name], "rb"))
            with open(paths[name], "rb") as fh:
                np.lib.format.read_magic(fh)
                np.lib.format.read_array_header_1_0(fh)
                hl = fh.tell()
            e = {"name": name, "tier": "G", "C": X.shape[3], "bin_first": extra["bin_first"],
                 "n_bins": X.shape[0], "cadence": extra["cadence"], "dtype": "<f2",
                 "channels": [{"name": n, "unit": u, "label": lab} for n, u, lab, _ in chans],
                 "files": [{"name": fn, "bytes": os.path.getsize(paths[name])}],
                 "grid": grid_json(), "header_len": hl, "live_bins_only": live,
                 "prefix": "tensors/fx7", "path": "tensors/fx7", "repo": "fixture",
                 "shape": list(X.shape), "slab_bytes": int(np.prod(X.shape[1:]) * 2),
                 "normalisation": "z-scored at build time"}
            del hdr
            return e

        groups = [grp_entry("fxg", Xg, G_CH, False, {"bin_first": G_BINS[0], "cadence": "pentad"}),
                  grp_entry("fxrg", Xrg, RG_CH, True, {"bin_first": 0, "cadence": "monthly"})]
        f7 = {"groups": {
            "fxg": {"chans": [c[0] for c in G_CH], "norm": [list(m) for m in g_norm],
                    "units": {c[0]: c[1] for c in G_CH}, "labels": {c[0]: c[2] for c in G_CH},
                    "dtype": "<f2", "grid": grid_json(), "n_bins": len(G_BINS), "bin_first": G_BINS[0]},
            "fxrg": {"chans": [c[0] for c in RG_CH], "norm": [list(m) for m in rg_norm],
                     "units": {c[0]: c[1] for c in RG_CH}, "labels": {c[0]: c[2] for c in RG_CH},
                     "dtype": "<f2", "grid": grid_json(), "n_bins": len(RG_MONTHS),
                     "bin_index": rg_bins, "live_only": True}}}
        # --- the schema-1 Argo-layout store, through family8_store's writer
        ab, ad, alat, alon, atemp, apsal = argo_rows()
        adir = os.path.join(OUT, "tensors", "fxargo")
        f8.write_synthetic_store(adir, ab, ad, alat, alon, atemp, apsal, levels=ARGO_LEVELS,
                                 n_bins=2060, note="fixture")
        sj = json.load(open(os.path.join(adir, "store.json")))
        sj["schema_version"] = 1
        sj["bin_first"] = 0
        with open(os.path.join(adir, "store.json"), "w") as fh:
            json.dump(sj, fh, indent=1)
        groups.append({"name": "fxargo", "tier": "P", "C": 6, "N": int(len(ab)), "bin_first": 0,
                       "bin_last": None, "n_bins": 2060, "schema_version": 1,
                       "cadence": "irregular (per observation)",
                       "channels": [{"name": f"{b}_{int(p)}", "unit": u} for b, u in (("temp", "degC"), ("psal", "PSU"))
                                    for p in ARGO_LEVELS],
                       "date_range": ["2009-12-20", "2010-02-20"],
                       "prefix": "tensors/fxargo", "path": "tensors/fxargo", "repo": "fixture"})
        # --- negative bins, through the family-10 assembler
        nd = m1.build_points(FxNeg, tmp + "/n")
        ndir = os.path.join(OUT, "tensors", "fxneg")
        shutil.copytree(nd, ndir)
        nj = json.load(open(os.path.join(ndir, "store.json")))
        groups.append({"name": "fxneg", "tier": "P", "C": 2, "N": int(nj["N"]),
                       "bin_first": int(nj["bin_first"]), "bin_last": int(nj["bin_last"]),
                       "n_bins": int(nj["n_bins"]), "schema_version": int(nj.get("schema_version", 2)),
                       "cadence": "6-hourly",
                       "channels": [{"name": n, "unit": u} for n, u, _, _ in FxNeg.channels],
                       "date_range": ["1980-01-01", "1980-04-30"],
                       "prefix": "tensors/fxneg", "path": "tensors/fxneg", "repo": "fixture"})
        reg = {"family": "family10", "family_version": "10.2", "repo": "fixture",
               "epoch": "1982-01-01", "pentad_days": 5, "generated_utc": "fixture", "groups": groups}
        json.dump(reg, open(os.path.join(OUT, "family10.json"), "w"), indent=1)
        json.dump(f7, open(os.path.join(OUT, "family7_index.json"), "w"), indent=1)
        # --- derived: a month-major fishing-like grid and a climatology
        months = ["2010-01", "2010-02", "2010-03"]
        rng = np.random.default_rng(10)
        F = np.zeros((3, NY, NX, 2), np.float32)
        F[..., 1] = rng.gamma(1.0, 50.0, F.shape[:3]).astype(np.float32)
        F[..., 0] = (F[..., 1] * rng.uniform(0, 1, F.shape[:3])).astype(np.float32)
        F[:, :, :5, :] = 0.0                          # zero is a value
        os.makedirs(os.path.join(OUT, "fishing"))
        fp = os.path.join(OUT, "fishing", "fishing_grid.npy")
        np.save(fp, F)
        fx = {"chans": ["fishing_hours", "hours"], "dtype": "<f4", "header_len": 128,
              "shape": list(F.shape), "slab_bytes": int(np.prod(F.shape[1:]) * 4), "months": months,
              "grid": grid_json(), "units": {"fishing_hours": "vessel-hours per month", "hours": "vessel-hours per month"},
              "labels": {"fishing_hours": "Apparent fishing hours", "hours": "Hours broadcasting on AIS"},
              "url": "fishing/fishing_grid.npy", "fixture": True}
        json.dump(fx, open(os.path.join(OUT, "fishing_index.json"), "w"), indent=1)
        CL = np.stack([np.stack([((g_phys(2044 + (m % 6))[..., k] - g_norm[k][0]) / g_norm[k][1])
                                 for k in range(3)]) for m in range(12)]).astype(np.float32)
        os.makedirs(os.path.join(OUT, "clim", "fxg"))
        cp = os.path.join(OUT, "clim", "fxg", "clim.npy")
        np.save(cp, CL)
        cx = {"default_version": "all", "layout": "[month, channel, lat, lon] float32 z-units",
              "files": {"all": {"fxg": {"clim_npy": {"url": "clim/fxg/clim.npy", "dtype": "<f4", "header_len": 128,
                                                     "plane_bytes": NY * NX * 4}}}},
              "groups": {"fxg": {"chans": [c[0] for c in G_CH], "grid": grid_json(),
                                 "labels": {c[0]: c[2] for c in G_CH}, "units": {c[0]: c[1] for c in G_CH},
                                 "norm": [list(m) for m in g_norm]}}}
        json.dump(cx, open(os.path.join(OUT, "clim_index.json"), "w"), indent=1)
        # header lengths are what the indexes say
        for p in (paths["fxg"], paths["fxrg"], fp, cp):
            with open(p, "rb") as fh:
                np.lib.format.read_magic(fh)
                np.lib.format.read_array_header_1_0(fh)
                assert fh.tell() == 128, (p, fh.tell())
        json.dump(make_expected(paths, rg_bins, (ab, ad, alat, alon, atemp, apsal), ndir, F, CL),
                  open(os.path.join(OUT, "expected.json"), "w"), separators=(",", ":"))
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    total = sum(os.path.getsize(os.path.join(dp, f)) for dp, _, fs in os.walk(OUT) for f in fs)
    print(f"wrote {OUT}: {total:,} bytes")
    if total > 1_500_000:
        sys.exit("the fixture is over 1.5 MB")


def make_expected(paths, rg_bins, argo, ndir, F, CL):
    cases = []
    # what is ON DISK, read back with numpy and de-z-scored in float64
    Xg = np.load(paths["fxg"]).astype(np.float64)
    Xrg = np.load(paths["fxrg"]).astype(np.float64)
    g_norm = [mu for _, _, _, mu in G_CH]
    rg_norm = [mu for _, _, _, mu in RG_CH]
    box = {"w": -40, "s": -30, "e": 40, "n": 30}
    rows, cols = box_idx(box)
    # 1. binmajor native: the January 2010 bins, channel sst and ssh
    sel = {"family": "10", "store": "fxg", "channels": ["sst", "ssh"], "yearStart": 2010, "yearEnd": 2010,
           "months": [1], "hours": None, "bbox": box, "step": "native", "res": "native"}
    keep = [i for i, b in enumerate(G_BINS) if (EPOCH + dt.timedelta(seconds=b * BIN_S)).year == 2010]
    ci = [0, 2]
    want = np.stack([np.stack([Xg[i][np.ix_(rows, cols)][..., c] * g_norm[c][1] + g_norm[c][0] for c in ci])
                     for i in keep])
    cases.append({"name": "binmajor native", "sel": sel, "shape": list(want.shape),
                  "time": [EPOCH_UNIX + G_BINS[i] * BIN_S for i in keep],
                  "lat": LAT[rows].tolist(), "lon": LON[cols].tolist(), "data": as_list(want)})
    # 2. binmajor monthly mean: t2m over January 2010, nanmean and finite count
    sel2 = dict(sel, channels=["t2m"], step="month")
    st = np.stack([Xg[i][np.ix_(rows, cols)][..., 1] * g_norm[1][1] + g_norm[1][0] for i in keep])
    with np.errstate(invalid="ignore"):
        mean = np.nanmean(st, 0)
    cases.append({"name": "binmajor month mean", "sel": sel2, "data": as_list(mean),
                  "count": np.isfinite(st).sum(0).ravel().tolist()})
    # 3. the levelled monthly group: rg_t10 and rg_s50 in February 2010
    sel3 = {"family": "10", "store": "fxrg", "channels": ["rg_t10", "rg_s50"], "yearStart": 2010, "yearEnd": 2010,
            "months": [2], "hours": None, "bbox": box, "step": "native", "res": "native"}
    want3 = np.stack([Xrg[1][np.ix_(rows, cols)][..., c] * rg_norm[c][1] + rg_norm[c][0] for c in (0, 5)])
    cases.append({"name": "levelled monthly", "sel": sel3, "data": as_list(want3[None]),
                  "time": [int((dt.datetime(2010, 2, 1) - dt.datetime(1970, 1, 1)).total_seconds())],
                  "rg_bins": rg_bins})
    # 4. schema-1 Argo rows: January 2010 in a box; times = rint(days * 86400)
    ab, ad, alat, alon, atemp, apsal = argo
    order = np.lexsort((ad.astype(np.float32).astype(np.float64), ab))
    td = ad[order].astype(np.float32).astype(np.float64)
    la, lo = alat[order].astype(np.float32), alon[order].astype(np.float32)
    tp = atemp[order].astype(np.float16).astype(np.float64)
    sec = np.rint(td * 86400)
    t0 = (dt.datetime(2010, 1, 1) - EPOCH).total_seconds()
    t1 = (dt.datetime(2010, 2, 1) - EPOCH).total_seconds()
    k = (sec >= t0) & (sec < t1) & (la >= -30) & (la <= 30) & (lo >= -90) & (lo <= 90)
    sel4 = {"family": "10", "store": "fxargo", "channels": ["temp_10", "temp_50"], "yearStart": 2010, "yearEnd": 2010,
            "months": [1], "hours": None, "bbox": {"w": -90, "s": -30, "e": 90, "n": 30}, "step": "native", "res": "native"}
    cases.append({"name": "schema1 rows", "sel": sel4, "time": (EPOCH_UNIX + sec[k]).tolist(),
                  "values": as_list(tp[k][:, [0, 2]])})
    # 5. negative bins: March 1980, the whole globe
    t, lat, lon, v = neg_rows(ndir)
    m0 = (dt.datetime(1980, 3, 1) - EPOCH).total_seconds()
    m1_ = (dt.datetime(1980, 4, 1) - EPOCH).total_seconds()
    kk = (t >= m0) & (t < m1_)
    sel5 = {"family": "10", "store": "fxneg", "channels": ["sst"], "yearStart": 1980, "yearEnd": 1980,
            "months": [3], "hours": None, "bbox": None, "step": "native", "res": "native"}
    cases.append({"name": "negative bins", "sel": sel5, "time": (EPOCH_UNIX + t[kk]).tolist(),
                  "values": as_list(v[kk][:, 1].astype(np.float64)),
                  "bins": [int(np.floor(m0 / BIN_S)), int(np.floor((m1_ - 1) / BIN_S))]})
    # 6. the month-major grid: February 2010, raw float32
    sel6 = {"family": "derived", "store": "fishing_grid", "channels": ["fishing_hours", "hours"], "yearStart": 2010,
            "yearEnd": 2010, "months": [2], "hours": None, "bbox": box, "step": "native", "res": "native"}
    cases.append({"name": "monthmajor", "sel": sel6,
                  "data": as_list(np.stack([F[1][np.ix_(rows, cols)][..., c] for c in (0, 1)])[None])})
    # 7. the climatology: January and July, ssh, de-z-scored; years ignored
    sel7 = {"family": "derived", "store": "clim_fxg", "channels": ["ssh"], "yearStart": 1999, "yearEnd": 1999,
            "months": [1, 7], "hours": None, "bbox": box, "step": "native", "res": "native"}
    want7 = np.stack([CL[m][2][np.ix_(rows, cols)].astype(np.float64) * g_norm[2][1] + g_norm[2][0] for m in (0, 6)])
    cases.append({"name": "clim", "sel": sel7, "data": as_list(want7[:, None])})
    return {"cases": cases}


if __name__ == "__main__":
    main()

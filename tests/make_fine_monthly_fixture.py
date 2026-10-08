#!/usr/bin/env python3
"""Two tiny FINE-grid-shaped sharded stores for E-089 (`ml/export_fine_monthly.py`).

Built with the REAL writer and framework (`ml/family1/sharded.py` through
`tests/make_family1_fixture.py::build_grid`, the `index | fetch | assemble |
check` stages `ml/build_family1_stores.py` runs), so the exporter reads the
layout the Hub stores have. The pooled grid of the fixture is 15° instead of
0.25° (`DEG`) so the stores stay a few kilobytes, but the geometry plays every
case the real grids do:

  fxfine   oc4k-shaped: float16, 2 channels, daily, NORTH-first 5° grid
           (36 x 72) in 16 x 16 tiles, 2009-12-25 .. 2010-02-10 (a year
           boundary), one day absent upstream (2010-01-14), random clouds on
           channel 0 and a land block covering the whole north-west tile
           (so that tile is EMPTY in every month and must not be stored);
           3 x 3 native pixels per 15° cell — EXACT blocks, like 6 x 6 at 4 km.
  fxtb     irtb-shaped: uint8 "K - 160", three-hourly (40 frames per bin),
           a SOUTH-first tropical band cut from a 120/18° grid whose first
           row's centre is exactly -30° (on a pooled edge), 50 columns of
           7.2° (one column centre exactly ON a pooled edge too), so the
           pooled blocks alternate 3/2 rows and 2/3 columns — centre-binned
           like the 2 km SST and the real IR grid; 2010-02-20 .. 2010-04-02,
           a full 248-frame March (the uint8 native count's real ceiling: the
           western half is seen in every frame, the eastern half misses 3 %)
           and one absent frame in February. Pooled counts reach 9 x 248 = 2,232,
           past uint8.

`build(dir)` -> {store_key: local store dir}; `BOXES`, `DEG` for the export;
`write_expected(stores, out)` writes numpy's answers (independent of the
exporter: frames read through sharded.py, cells assigned from the grid's own
affine rule) for the website reader's tests. Deterministic.
"""
import datetime as dt
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import make_family1_fixture as m1                               # noqa: E402
from make_family1_fixture import sh                             # noqa: E402

DEG = 15.0
LIC = {"name": "fixture licence", "attribution": "fixture: synthetic data",
       "redistribution": "attribution", "derived_works": "free",
       "redistribution_confirmed": True}


def bin_of(day):
    return (day - dt.date(1982, 1, 1)).days // 5


# ================================================================ fxfine ===
FH, FW, FRES = 36, 72, 5.0
F_GRID = m1.geo(FH, FW, -180.0, 90.0, FRES, -FRES,
                "row 0 is the NORTHERNMOST row (fixture, like oc4k)")
F_LO, F_HI = dt.date(2009, 12, 25), dt.date(2010, 2, 10)
F_ABSENT = dt.date(2010, 1, 14)


def f_centres():
    return (90.0 - (np.arange(FH) + 0.5) * FRES,
            -180.0 + (np.arange(FW) + 0.5) * FRES)


def f_land():
    lat, lon = f_centres()
    LAT, LON = np.meshgrid(lat, lon, indexing="ij")
    m = np.zeros((FH, FW), bool)
    m[:16, :16] = True                         # the whole north-west tile
    m |= (np.abs(LON - 20) < 15) & (np.abs(LAT) < 30)
    return m


def f_frame(b, f):
    day = dt.date(1982, 1, 1) + dt.timedelta(days=5 * b + f)
    if not (F_LO <= day <= F_HI):
        return "outside", day
    if day == F_ABSENT:
        return None, day
    lat, lon = f_centres()
    LAT, LON = np.meshgrid(lat, lon, indexing="ij")
    rng = np.random.default_rng(10_000 + 31 * b + f)
    a = np.empty((FH, FW, 2), np.float32)
    a[..., 0] = (14 + 13 * np.cos(np.radians(LAT)) + 0.01 * LON
                 + rng.normal(0, 0.9, LAT.shape))
    a[..., 1] = -1.2 + 0.4 * np.sin(np.radians(LON)) + rng.normal(0, 0.3,
                                                                  LAT.shape)
    a[..., 0][rng.random(LAT.shape) < 0.2] = np.nan
    a[f_land()] = np.nan
    return a, day


class FxFine(sh.GridAdapter):
    store = "fxfine"
    title = "fixture: daily 2-channel ocean-colour-shaped field, north-first"
    family = "1gf"
    distribution = "public"
    licence = LIC
    channels = (("sst", "degC", -5.0, 40.0),
                ("log_chl", "log10(mg m-3) — fixture", -4.0, 2.5))
    log2_fp = -2.6
    log2_dt = -2.32
    first_year = 2009
    frames_per_bin = 5
    frame_seconds = 86400
    dtype = "float16"
    tile = 16
    zstd_level = 3
    grid = F_GRID
    sources = ("synthetic",)
    smoke_window = (str(F_LO), str(F_HI))

    def index(self, ctx):
        return {"dataset": "fixture"}

    def fetch_frames(self, ctx, wanted):
        for g, b, f in wanted:
            a, day = f_frame(b, f)
            if isinstance(a, str):
                why = "before_record" if day < F_LO else "after_record"
                yield g, b, f, None, {"frame_missing": why}
            elif a is None:
                yield g, b, f, None, {"frame_missing": "absent_upstream"}
            else:
                yield g, b, f, a, {"files_read": 1}


# ================================================================== fxtb ===
T_DY = 120.0 / 18
T_ROWS = (4, 13)
TH, TW, T_DX = T_ROWS[1] - T_ROWS[0], 50, 360.0 / 50
T_GRID = dict(m1.geo(TH, TW, -180.0, -60.0 + T_ROWS[0] * T_DY, T_DX, T_DY,
                     "row 0 is the SOUTHERNMOST row (fixture, like irtb)"),
              band_rows=list(T_ROWS))
T_LO = dt.datetime(2010, 2, 20)
T_HI = dt.datetime(2010, 4, 2, 21)
T_ABSENT = dt.datetime(2010, 2, 25, 3)
T_CH = (("tb", "K - 160 (uint8; add 160 for kelvin)", 0.0, 254.0),)


def t_centres():
    return (T_GRID["y0"] + (np.arange(TH) + 0.5) * T_DY,
            -180.0 + (np.arange(TW) + 0.5) * T_DX)


def t_frame(b, f):
    t = dt.datetime(1982, 1, 1) + dt.timedelta(seconds=b * 432000 + f * 10800)
    if not (T_LO <= t <= T_HI):
        return "outside", t
    if t == T_ABSENT:
        return None, t
    lat, lon = t_centres()
    LAT, LON = np.meshgrid(lat, lon, indexing="ij")
    rng = np.random.default_rng(20_000 + 41 * b + f)
    a = (130 + 25 * np.cos(np.radians(LAT) * 3) + 20 * np.sin(
        np.radians(LON) + f * np.pi / 4) + rng.normal(0, 12, LAT.shape))
    a = np.clip(np.rint(a), 0, 254).astype(np.float32)
    a[(rng.random(LAT.shape) < 0.03) & (LON > 0)] = np.nan   # nobody saw it
    return a[..., None], t


class FxTb(sh.GridAdapter):
    store = "fxtb"
    title = "fixture: uint8 K-160 three-hourly tropical band (irtb-shaped)"
    family = "1gf"
    distribution = "public"
    licence = LIC
    channels = T_CH
    log2_fp = -2.8
    log2_dt = -7.9
    first_year = 2010
    frames_per_bin = 40
    frame_seconds = 10800
    dtype = "uint8"
    tile = 8
    zstd_level = 3
    grid = T_GRID
    sources = ("synthetic",)
    smoke_window = ("2010-02-20", "2010-04-02")

    def index(self, ctx):
        return {"dataset": "fixture"}

    def fetch_frames(self, ctx, wanted):
        for g, b, f in wanted:
            a, t = t_frame(b, f)
            if isinstance(a, str):
                why = "before_record" if t < T_LO else "after_record"
                yield g, b, f, None, {"frame_missing": why}
            elif a is None:
                yield g, b, f, None, {"frame_missing": "absent_upstream"}
            else:
                yield g, b, f, a, {"files_read": 1}


STORES = (("fixture/fxfine", FxFine), ("fixture/fxtb", FxTb))
BOXES = {
    "fixture/fxfine": [(2010, (1, 2), (30.0, 60.0, -60.0, -30.0)),
                       (2009, (12,), (-45.0, -15.0, 0.0, 45.0))],
    "fixture/fxtb": [(2010, (2, 3, 4), (-30.0, 0.0, -45.0, 0.0)),
                     (2010, (3,), (0.0, 30.0, 90.0, 135.0))],
}


def build(tmp):
    """{store_key: local store dir}."""
    return {k: m1.build_grid(cls, os.path.join(tmp, cls.store))
            for k, cls in STORES}


# ============================================================ expected ===
def _frames(base, key):
    """{(y, m): [n, H, W, C] float32 physical} via sharded.py, + spec."""
    g = key.split("/")[-1]
    grp = sh.ShardedGroup(os.path.join(base, g))
    spec = grp.spec
    off = 160.0 if spec["dtype"] == "uint8" else 0.0
    by = {}
    for r in grp.shard_index:
        b, mask = int(r["bin"]), int(r["frame_mask"])
        for f in range(spec["frames_per_bin"]):
            if mask >> f & 1:
                t = sh.frame_datetime(b, f, spec["frame_seconds"])
                by.setdefault((t.year, t.month), []).append(
                    grp.read_frame(b, f) + np.float32(off))
    return {k: np.stack(v) for k, v in by.items()}, spec


def _cells(spec):
    """Cell (J, I) of every native row / column, from the affine rule."""
    gr = spec["grid"]
    lat = gr["y0"] + (np.arange(spec["H"]) + 0.5) * gr["dy"]
    lon = gr["x0"] + (np.arange(spec["W"]) + 0.5) * gr["dx"]

    def c(v, o):
        t = (v - o) / DEG
        r = np.round(t)
        return np.floor(np.where(np.abs(t - r) < 1e-6, r, t)).astype(int)
    return c(lat, -90.0), c(lon, -180.0)


def _stats(x):
    x = x.astype(np.float64)
    n = np.isfinite(x).sum(axis=0)
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return n, np.nanmean(x, axis=0), np.nanstd(x, axis=0)


def write_expected(stores, out):
    """expected.json: numpy over native frames for selections a reader is
    asked — per store: every (year, month) and the union of all months, at a
    few NATIVE pixels and a few POOLED cells, plus whole-grid summaries."""
    res = dict(_source="tests/make_fine_monthly_fixture.py — numpy over the "
                       "native frames, independent of the exporter",
               deg=DEG, stores={})
    for key, base in stores.items():
        fr, spec = _frames(base, key)
        J, I = _cells(spec)
        j0, i0 = int(J.min()), int(I.min())
        rng = np.random.default_rng(5)
        H, W, C = spec["H"], spec["W"], spec["C"]
        pix = [(int(r), int(c)) for r, c in zip(rng.integers(0, H, 8),
                                                rng.integers(0, W, 8))]
        Hp, Wp = int(J.max() - j0 + 1), int(I.max() - i0 + 1)
        cel = [(int(a), int(b)) for a, b in zip(rng.integers(0, Hp, 6),
                                                rng.integers(0, Wp, 6))]
        sel = {f"{y}-{m:02d}": [(y, m)] for y, m in sorted(fr)}
        sel["all"] = sorted(fr)
        out_s = dict(native_pixels=pix, pooled_cells=cel,
                     pooled_note="pooled row 0 is the SOUTHERNMOST cell row",
                     selections={})
        for name, keys in sel.items():
            x = np.concatenate([fr[k] for k in keys])
            n, mu, sd = _stats(x)
            # pooled: every native value whose centre is in the cell
            pn = np.zeros((Hp, Wp, C), np.int64)
            ps = np.zeros((Hp, Wp, C))
            pq = np.zeros((Hp, Wp, C))
            for r in range(H):
                for c in range(W):
                    v = x[:, r, c].astype(np.float64)
                    ok = np.isfinite(v)
                    pn[J[r] - j0, I[c] - i0] += ok.sum(axis=0)
                    ps[J[r] - j0, I[c] - i0] += np.where(ok, v, 0).sum(axis=0)
            with np.errstate(invalid="ignore", divide="ignore"):
                pm = ps / pn
            for r in range(H):
                for c in range(W):
                    v = x[:, r, c].astype(np.float64)
                    d = np.where(np.isfinite(v), v - pm[J[r] - j0, I[c] - i0],
                                 0.0)
                    pq[J[r] - j0, I[c] - i0] += (d * d).sum(axis=0)
            with np.errstate(invalid="ignore", divide="ignore"):
                psd = np.sqrt(pq / pn)

            def rec(nn, mm, ss):
                return [dict(count=int(nn[k]),
                             mean=(None if nn[k] == 0 else float(mm[k])),
                             std=(None if nn[k] == 0 else float(ss[k])))
                        for k in range(C)]
            out_s["selections"][name] = dict(
                months=[list(k) for k in keys], frames=int(len(x)),
                native=[rec(n[r, c], mu[r, c], sd[r, c]) for r, c in pix],
                pooled=[rec(pn[a, b], pm[a, b], psd[a, b]) for a, b in cel],
                native_total_count=int(n.sum()),
                pooled_total_count=int(pn.sum()),
                pooled_max_count=int(pn.max()))
        res["stores"][key] = out_s
    with open(os.path.join(out, "expected.json"), "w") as fh:
        json.dump(res, fh, indent=1, sort_keys=True)
        fh.write("\n")
    return res


if __name__ == "__main__":                                  # pragma: no cover
    import tempfile
    print(build(tempfile.mkdtemp()))

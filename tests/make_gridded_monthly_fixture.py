#!/usr/bin/env python3
"""Two tiny sharded tier-G stores for E-088's monthly sums (`ml/export_gridded_monthly.py`).

Built with the REAL writer and framework (`ml/family1/sharded.py` through
`tests/make_family1_fixture.py::build_grid`, the `index | fetch | assemble |
check` stages `ml/build_family1_stores.py` runs), so the exporter reads exactly
the layout the Hub stores have:

  fxdaily  float16, 2 channels on a 10° POINT-registered south-first grid
           (19 x 36, like family 7.2d), 8 x 8 tiles, one frame per day,
           2009-12-20 .. 2011-01-10 — across two year boundaries, with a land
           mask (always NaN), random cloud gaps on channel 0, one day absent
           upstream (2010-03-14) and values spanning several binades so the
           float32 sums really do round;
  fx6h     float16, 3 "pressure levels" on the same grid, six-hourly instants
           (20 frames per five-day bin), 2010-01-28 .. 2010-03-03 06 UTC —
           month boundaries inside a bin, a partial first and last month, a
           full 112-frame February, no gaps (a reanalysis).

`build(dir)` returns {store_key: local store directory}. Deterministic.
"""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import make_family1_fixture as m1                               # noqa: E402
from make_family1_fixture import sh                             # noqa: E402

H, W, RES = 19, 36, 10.0
GRID = dict(m1.geo(H, W, -180.0, -90.0, RES, RES, "south_first"),
            align="point — row r is latitude -90 + 10 r, column c is "
                  "longitude -180 + 10 c (fixture)")
LIC = {"name": "fixture licence", "attribution": "fixture: synthetic data",
       "redistribution": "attribution", "derived_works": "free",
       "redistribution_confirmed": True}


def centres():
    return -90.0 + np.arange(H) * RES, -180.0 + np.arange(W) * RES


def land():
    lat, lon = centres()
    LAT, LON = np.meshgrid(lat, lon, indexing="ij")
    return (np.abs(LON - 20) < 25) & (np.abs(LAT) < 50)


def bin_of(iso):
    import datetime as dt
    return (dt.date.fromisoformat(iso) - dt.date(1982, 1, 1)).days // 5


def daily_frame(b, f):
    import datetime as dt
    day = dt.date(1982, 1, 1) + dt.timedelta(days=5 * b + f)
    if not (dt.date(2009, 12, 20) <= day <= dt.date(2011, 1, 10)):
        return "outside"
    if day == dt.date(2010, 3, 14):
        return None
    lat, lon = centres()
    LAT, LON = np.meshgrid(lat, lon, indexing="ij")
    doy = day.timetuple().tm_yday
    rng = np.random.default_rng(1000 * b + f)
    a = np.empty((H, W, 2), np.float32)
    a[..., 0] = (15 + 12 * np.cos(np.radians(LAT))
                 + 3 * np.sin(2 * np.pi * doy / 365.25)
                 + rng.normal(0, 0.8, LAT.shape))
    a[..., 1] = np.clip(0.002 * (np.abs(LAT) - 40) + rng.normal(0, 0.01,
                                                               LAT.shape),
                        0, 1)
    a[..., 0][rng.random(LAT.shape) < 0.2] = np.nan     # clouds
    a[land()] = np.nan
    return a


class FxDaily(sh.GridAdapter):
    store = "fxdaily"
    title = "fixture: daily 2-channel ocean field (family 7.2d-shaped)"
    family = "72d"
    distribution = "public"
    licence = LIC
    channels = (("sst", "degC", -5.0, 40.0), ("sea_ice", "0-1", 0.0, 1.0))
    log2_fp = 0.0
    log2_dt = -2.32
    first_year = 2009
    frames_per_bin = 5
    frame_seconds = 86400
    dtype = "float16"
    tile = 8
    zstd_level = 3
    grid = GRID
    sources = ("synthetic",)
    smoke_window = ("2009-12-20", "2011-01-10")

    def index(self, ctx):
        return {"dataset": "fixture"}

    def fetch_frames(self, ctx, wanted):
        lo, hi = bin_of("2009-12-20"), bin_of("2011-01-10")
        for g, b, f in wanted:
            if b < lo or b > hi:
                why = "before_record" if b < lo else "after_record"
                yield g, b, f, None, {"frame_missing": why}
                continue
            a = daily_frame(b, f)
            if isinstance(a, str):
                import datetime as dt
                day = dt.date(1982, 1, 1) + dt.timedelta(days=5 * b + f)
                why = "before_record" if day < dt.date(2009, 12, 20) \
                    else "after_record"
                yield g, b, f, None, {"frame_missing": why}
            elif a is None:
                yield g, b, f, None, {"frame_missing": "absent_upstream"}
            else:
                yield g, b, f, a, {"files_read": 1}


LEVELS = (100, 500, 850)
T6_LO = (bin_of("2010-01-28"), 0)


def six_frame(b, f):
    import datetime as dt
    t = dt.datetime(1982, 1, 1) + dt.timedelta(seconds=b * 432000 + f * 21600)
    if not (dt.datetime(2010, 1, 28) <= t <= dt.datetime(2010, 3, 3, 6)):
        return None
    lat, lon = centres()
    LAT, LON = np.meshgrid(lat, lon, indexing="ij")
    rng = np.random.default_rng(7 * b + f)
    a = np.empty((H, W, len(LEVELS)), np.float32)
    for k, p in enumerate(LEVELS):
        a[..., k] = (210 + 0.08 * p + 25 * np.cos(np.radians(LAT))
                     + 2 * np.sin(np.radians(LON) + f * np.pi / 2)
                     + rng.normal(0, 0.5, LAT.shape))
    return a


class Fx6h(sh.GridAdapter):
    store = "fx6h"
    title = "fixture: ERA5-shaped temperature on 3 levels, six-hourly"
    family = "12"
    distribution = "public"
    licence = LIC
    channels = tuple((f"t_{p}", "K", 150.0, 350.0) for p in LEVELS)
    log2_fp = 2.0
    log2_dt = -4.32
    first_year = 2010
    frames_per_bin = 20
    frame_seconds = 21600
    dtype = "float16"
    tile = 8
    zstd_level = 3
    grid = GRID
    sources = ("synthetic",)
    smoke_window = ("2010-01-28", "2010-03-03")

    def index(self, ctx):
        return {"dataset": "fixture"}

    def fetch_frames(self, ctx, wanted):
        lo, hi = bin_of("2010-01-28"), bin_of("2010-03-03")
        for g, b, f in wanted:
            a = six_frame(b, f)
            if a is None:
                why = "before_record" if b < lo or (
                    b == lo and f < 0) else "after_record"
                import datetime as dt
                t = dt.datetime(1982, 1, 1) + dt.timedelta(
                    seconds=b * 432000 + f * 21600)
                why = "before_record" if t < dt.datetime(2010, 1, 28) \
                    else "after_record"
                yield g, b, f, None, {"frame_missing": why}
            else:
                yield g, b, f, a, {"files_read": 1}


def build(tmp):
    """{store_key: local store dir} — both stores built in `tmp`."""
    out = {}
    for key, cls in (("fixture/fxdaily", FxDaily), ("fixture/fx6h", Fx6h)):
        out[key] = m1.build_grid(cls, os.path.join(tmp, cls.store))
    return out


if __name__ == "__main__":                                  # pragma: no cover
    import tempfile
    d = tempfile.mkdtemp()
    print(build(d))

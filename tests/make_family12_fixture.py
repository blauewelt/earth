#!/usr/bin/env python3
"""Writes `data/family12_fixture/` — family 1.2's shape for `tests/f1data.test.mjs` (E-084).

    python3 tests/make_family12_fixture.py         # rewrites the fixture

PLAIN ENGLISH. Family 1.2 is family 1.gf (inherited by reference — the same
bytes, so the Data tab lists those stores once, under 1.gf) plus ECMWF's ERA5
reanalysis of the atmosphere on 13 pressure levels: temperature, humidity
and wind, six-hourly instants on a 1° grid, levels folded into channels
named `t_500`. This script builds ONE small store in that shape with the
real tier-G writer (`ml/family1/sharded.py`, through the same `index | fetch
| assemble | check` stages `ml/build_family1_stores.py` runs), extending
`tests/make_family1_fixture.py` and reusing its builders and its numpy
reference:

  fxera   float16, 3 "pressure levels" (t_100, t_500, t_850, levels_hpa in
          the registry), a 10° global grid POINT-registered south-first
          like ERA5's (row 0 = -90°, col 0 = -180°; 19 x 36) cut into 8 x 8
          tiles; 20 six-hourly frames per five-day bin; bins 2044 and 2045
          (2009-12-25 .. 2010-01-03), the second only HALF filled — its last
          10 frames (after 2010-01-01 06 UTC) are after the record, the way ERA5's last bin ends on
          2026-06-30 inside a bin that runs to 07-03 — so the reader's
          record end must come from the shard index, not from the bins.

and `family12.json` in the live registry's shape: an `inherits_block`
naming 1.gf's `fxgrid` (which is ALSO listed in `groups` here, to prove the
reader does not list an inherited store twice), the built `fxera` with its
licence attribution and `levels_hpa`, a not-built `fxera_q` (the tab names
it as coming, never as selectable) and a not-built `fxwait` whose licence
still waits on the producer (named nowhere). `expected.json` holds
selections and their truth computed with numpy, independently of the
JavaScript. Deterministic; well under 0.5 MB.
"""
import json
import os
import shutil
import sys
import tempfile

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import make_family1_fixture as m1                               # noqa: E402
from make_family1_fixture import sh                             # noqa: E402

OUT = os.path.join(m1.ROOT, "data", "family12_fixture")
E_BINS = (2044, 2045)
E_FILLED = {2044: 20, 2045: 10}       # frames present per bin (the rest after the record)
E_LEVELS = (100, 500, 850)
E_H, E_W, E_RES = 19, 36, 10.0
E_GRID = m1.geo(E_H, E_W, -185.0, -95.0, E_RES, E_RES,
                "row 0 is the SOUTHERNMOST row (-90°), point-registered")
E_CH = tuple((f"t_{p}", "K", 150.0, 350.0) for p in E_LEVELS)
LICENCE = {"name": "CC BY 4.0 (fixture, ERA5-shaped)",
           "attribution": "fixture: Contains modified Copernicus Climate Change "
                          "Service information [2009-2010].",
           "redistribution": "attribution", "derived_works": "free",
           "redistribution_confirmed": True}


def e_centres():
    lat = -90.0 + np.arange(E_H) * E_RES
    lon = -180.0 + np.arange(E_W) * E_RES
    return lat, lon


def e_frame(b, f):
    """The float32 field handed to the writer, or None (after the record)."""
    if b not in E_FILLED or f >= E_FILLED[b]:
        return None
    lat, lon = e_centres()
    LAT, LON = np.meshgrid(lat, lon, indexing="ij")
    rng = np.random.default_rng(31 * b + f)
    a = np.empty((E_H, E_W, len(E_LEVELS)), np.float32)
    for k, p in enumerate(E_LEVELS):
        # colder aloft, warmer toward the equator, a diurnal wobble by frame
        a[..., k] = (210 + 0.08 * p + 25 * np.cos(np.radians(LAT))
                     + 2 * np.sin(np.radians(LON) + f * np.pi / 2)
                     + rng.normal(0, 0.5, LAT.shape))
    return a                          # a reanalysis has no gaps


class FxEra(sh.GridAdapter):
    store = "fxera"
    title = "fixture: ERA5-shaped temperature on 3 pressure levels, six-hourly"
    family = "12"
    distribution = "public"
    licence = LICENCE
    channels = E_CH
    log2_fp = 2.0
    log2_dt = -4.32
    first_year = 2009
    frames_per_bin = 20
    frame_seconds = 21600
    dtype = "float16"
    tile = 8
    zstd_level = 3
    grid = E_GRID
    sources = ("synthetic",)
    smoke_window = ("2009-12-25", "2010-01-03")

    def index(self, ctx):
        return {"dataset": "fixture"}

    def fetch_frames(self, ctx, wanted):
        for g, b, f in wanted:
            if b not in E_BINS:
                why = "before_record" if b < E_BINS[0] else "after_record"
                yield g, b, f, None, {"frame_missing": why}
                continue
            a = e_frame(b, f)
            if a is None:
                yield g, b, f, None, {"frame_missing": "after_record"}
            else:
                yield g, b, f, a, {"files_read": 1}


_SEL_FRAMES = m1.sel_frames


def sel_frames_days(frames, fs, F, bins, sel):
    """make_family1_fixture.sel_frames plus the reader's day-of-month range."""
    keep = _SEL_FRAMES(frames, fs, F, bins, sel)
    d = sel.get("days")
    if not d:
        return keep
    return [x for x in keep if d[0] <= m1.utc(x[2]).day <= d[1]]


def main():
    tmp = tempfile.mkdtemp(prefix="f12fixture_")
    try:
        built = m1.build_grid(FxEra, tmp + "/e")
        if os.path.isdir(OUT):
            shutil.rmtree(OUT)
        os.makedirs(OUT)
        shutil.copytree(built, os.path.join(OUT, "fxera"))
        e = m1.registry_entry(os.path.join(OUT, "fxera"), "fxera", FxEra, "G")
        e.update({"path": "tensors/family1_2/fxera", "family_version": "1.2",
                  "cadence": "6-hourly", "licence": LICENCE, "exception": "E4",
                  "levels_hpa": list(E_LEVELS),
                  "notes": "FAMILY 1.2, EXCEPTION E4: a model-filled reanalysis, "
                           "not an observation (fixture)"})
        # the registry's own record runs to the END OF THE BIN (and the live
        # one's date_range further still); the truth is in the shard index
        e["record_span"] = ["2009-12-25", "2010-12-31"]
        groups = [
            # inherited by reference: must NOT appear twice in the tab
            {"name": "fxgrid", "tier": "G", "built": True, "distribution": "public",
             "channels": [], "path": "tensors/family1_gf/fxgrid"},
            e,
            {"name": "fxera_q", "tier": "G", "built": False, "distribution": "public",
             "title": "fixture: ERA5-shaped humidity — family 1.2, exception E4",
             "channels": [], "path": "tensors/family1_2/fxera_q", "licence": LICENCE},
            {"name": "fxwait", "tier": "G", "built": False, "distribution": "public",
             "title": "fixture: a store whose licence is pending", "channels": [],
             "path": "tensors/family1_2/fxwait",
             "licence": {"name": "pending", "pending": "asked the producer",
                         "redistribution_confirmed": False}},
        ]
        reg = {"family": "family1_2", "family_version": "1.2",
               "hf_root": "tensors/family1_2", "repo": "fixture",
               "inherits": "family1gf",
               "inherits_block": {"family_version": "1.gf", "groups": ["fxgrid"],
                                  "registry": "tensors/family1_gf/family1gf.json"},
               "epoch": "1982-01-01", "pentad_days": 5,
               "generated_utc": "fixture", "groups": groups,
               "built": ["fxera"], "not_built": ["fxera_q", "fxwait"]}
        with open(os.path.join(OUT, "family12.json"), "w") as fh:
            json.dump(reg, fh, indent=1)
        with open(os.path.join(OUT, "expected.json"), "w") as fh:
            json.dump(make_expected(), fh, separators=(",", ":"))
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    total = sum(os.path.getsize(os.path.join(dp, f))
                for dp, _, fs in os.walk(OUT) for f in fs)
    print(f"wrote {OUT}: {total:,} bytes")
    if total > 500_000:
        sys.exit("the fixture is over 0.5 MB")


def make_expected():
    fr = {(b, f): m1.f16(e_frame(b, f)) for b in E_BINS for f in range(20)}
    frames = lambda b, f: fr[(b, f)]                            # noqa: E731
    # the stored bytes are the given values through float16 — read back
    # with the Python reader so the truth below is the store's own
    grp = sh.ShardedGroup(os.path.join(OUT, "fxera", "fxera"))
    for (b, f), a in fr.items():
        got = grp.read_frame(b, f)
        if a is None:
            assert got is None, (b, f)
        else:
            assert np.array_equal(got, a), (b, f)
    lat_c, lon_c = e_centres()
    names = [c[0] for c in E_CH]
    base = {"family": "1.2", "store": "fxera", "channels": ["t_500"],
            "yearStart": 2009, "yearEnd": 2009, "months": [12], "days": None,
            "hours": None, "bbox": {"w": -55, "s": 25, "e": -25, "n": 55},
            "step": "native", "res": "native"}
    cases = []

    def ecase(name, **over):
        sel = dict(base, **over)
        ci = [names.index(n) for n in sel["channels"]]
        # grid_reference looks sel_frames up at call time: give it the days
        m1.sel_frames = sel_frames_days
        try:
            r = m1.grid_reference(frames, 21600, 20, E_BINS, lat_c, lon_c, sel, ci)
        finally:
            m1.sel_frames = _SEL_FRAMES
        cases.append({"name": name, "sel": sel, "expect": r})

    # one level, one day, hours 12 up to (not including) 18: the 12 UTC frame
    ecase("level 500 one day hours 12-18", days=[28, 28], hours=[12, 18])
    # two levels, hours across midnight: 18 and 00 UTC on two days
    ecase("levels 100+850 hours 18-6", channels=["t_100", "t_850"],
          days=[26, 27], hours=[18, 6])
    # a monthly mean of December: 7 days x 4 instants = 28 frames per cell
    ecase("level 500 december mean", step="month")
    # across the year end into the half-filled last bin
    ecase("level 850 across the record end", channels=["t_850"], yearEnd=2010,
          months=[12, 1], step="all")
    assert cases[0]["expect"]["frames"] == 1
    assert cases[1]["expect"]["frames"] == 4
    assert cases[2]["expect"]["frames"] == 28
    assert cases[3]["expect"]["frames"] == 28 + 2
    return {"note": "written by tests/make_family12_fixture.py; values are the "
                    "store's own (float16 through float32)",
            "epoch_unix": m1.EPOCH_UNIX, "span": ["2009-12-25", "2010-01-01"],
            "coming": ["fxera_q"], "cases": cases}


if __name__ == "__main__":
    main()

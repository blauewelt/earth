#!/usr/bin/env python3
"""E-093 fixture: a synthetic bin-major five-day store (two groups, z-scored
float16, NaN where missing) in family 10's registry shape, the pentad-of-year
sums written from it by the REAL exporter (ml/export_gridded_poy.py), the
index built by the real index builder, and expected.json computed with numpy
straight from the native bins (independent of the exporter).

    python3 tests/make_gridded_poy_fixture.py      # → data/gridded_poy/fixture/

Groups:
  fxp — 2 channels, bins 0 … 364 (1982-01-01 … 1986-12-31: five years, 1984
        a leap year, so slot 72 of 1984 holds two bins); bin 40 entirely NaN
  fxq — 1 channel, bins 100 … 364 (from 1983-05-16: a partial first year)
Grid: 20° (10 × 18), south first, lat0 −90, lon0 −180.
"""
import datetime as dt
import json
import os
import shutil
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "ml"))
import export_gridded_poy as P                                    # noqa: E402

OUT = os.path.join(ROOT, "data", "gridded_poy", "fixture")
STEM = "fxp"
GRID = {"lat0": -90.0, "lon0": -180.0, "nx": 18, "ny": 10, "south_first": True, "step": 20.0, "wrap": True}
GROUPS = {
    "fxp": dict(chans=["sst", "ssh"], norm=[[15.0, 8.0], [-0.2, 0.7]],
                units={"sst": "°C", "ssh": "m"},
                labels={"sst": "Sea-surface temperature", "ssh": "Sea surface height"},
                bin_first=0, n_bins=365),
    "fxq": dict(chans=["log_chl"], norm=[[-0.78, 0.46]], units={"log_chl": "log₁₀ mg/m³"},
                labels={"log_chl": "log10 chlorophyll-a (mg/m³)"}, bin_first=100, n_bins=265),
}


def write_tensor(path, g, rng):
    H, W, C = GRID["ny"], GRID["nx"], len(g["chans"])
    T = g["n_bins"]
    lat = np.arange(H) * 20.0 - 90
    lon = np.arange(W) * 20.0 - 180
    t = (np.arange(T) + g["bin_first"])[:, None, None, None]
    base = (np.cos(np.radians(lat))[None, :, None, None] * 1.2
            + 0.3 * np.sin(np.radians(lon))[None, None, :, None]
            + 0.8 * np.sin(2 * np.pi * t / 73.05)
            + 0.15 * np.arange(C)[None, None, None, :])
    z = base + 0.4 * rng.standard_normal((T, H, W, C))
    land = (np.abs(lat)[:, None] < 25) & (np.abs(lon)[None, :] < 30)           # a "continent"
    z[:, land, :] = np.nan
    z[:, 0, :, :] = np.where(rng.random((T, W, C)) < 0.3, np.nan, z[:, 0, :, :])   # a ragged pole
    if g["bin_first"] <= 40 < g["bin_first"] + T:
        z[40 - g["bin_first"]] = np.nan                                        # a missing bin
    a = z.astype("<f2")
    np.save(path, a)
    return a


def main():
    if os.path.exists(OUT):
        shutil.rmtree(OUT)
    os.makedirs(os.path.join(OUT, "tensors", STEM))
    rng = np.random.default_rng(93)
    norms = {"_source": "tests/make_gridded_poy_fixture.py", "stem": STEM, "base": "./tensors/fxp/",
             "groups": {}}
    reg = {"family": "family10", "family_version": "10.2", "repo": "fixture", "epoch": "1982-01-01",
           "pentad_days": 5, "generated_utc": "fixture", "groups": []}
    native = {}
    for name, g in GROUPS.items():
        fn = f"{STEM}_X_{name}.npy"
        p = os.path.join(OUT, "tensors", STEM, fn)
        native[name] = write_tensor(p, g, rng)
        with open(p, "rb") as fh:
            hl = P.parse_npy_header(fh.read(4096))[0]
        H, W, C = GRID["ny"], GRID["nx"], len(g["chans"])
        ent = dict(chans=g["chans"], norm=g["norm"], units=g["units"], labels=g["labels"], dtype="<f2",
                   grid=GRID, n_bins=g["n_bins"], bin_first=g["bin_first"], header_len=hl,
                   bytes=os.path.getsize(p), file=fn, slab_bytes=H * W * C * 2,
                   shape=[g["n_bins"], H, W, C], sha256=P.sha256_file(p))
        norms["groups"][name] = ent
        reg["groups"].append(dict(
            name=name, tier="G", C=C, bin_first=g["bin_first"], n_bins=g["n_bins"], cadence="pentad",
            dtype="<f2", channels=[dict(name=c, unit=g["units"][c], label=g["labels"][c]) for c in g["chans"]],
            files=[dict(name=fn, bytes=ent["bytes"])], grid=GRID, header_len=hl, live_bins_only=False,
            prefix=f"tensors/{STEM}", path=f"tensors/{STEM}", repo="fixture",
            shape=ent["shape"], slab_bytes=ent["slab_bytes"], normalisation="z-scored at build time"))
    P.write_json(os.path.join(OUT, "family7_index.json"), norms)
    P.write_json(os.path.join(OUT, "family10.json"), reg)

    # the real exporter, every chunk of every group, from the local files
    poy = os.path.join(OUT, "tensors", STEM, "poy", P.VERSION)
    for name in GROUPS:
        for k in range(P.NCHUNK):
            P.export_chunk(name, k, os.path.join(poy, name), index=os.path.join(OUT, "family7_index.json"),
                           base=os.path.join(OUT, "tensors", STEM), threads=1)
    ix = P.build_index(list(GROUPS), local=poy, base="./", stem=STEM)
    ix["generated_utc"] = "fixture"
    for blk in ix["stores"].values():
        blk["source_sha256"] = blk["source_sha256"]
    P.write_json(os.path.join(OUT, "gridded_poy_index.json"), ix)
    # manifests carry wall-clock fields; keep the fixture reproducible
    for name in GROUPS:
        for k in range(P.NCHUNK):
            mp = os.path.join(poy, name, f"manifest_c{k:02d}.json")
            m = json.load(open(mp))
            m.update(generated_utc="fixture", git_sha="fixture", seconds=0)
            P.write_json(mp, m)

    # expected: numpy over the native bins, per slot, full grid
    def expect(name, ya, yb, ex, months):
        g = GROUPS[name]
        a = native[name]
        nm = np.array(g["norm"], np.float64)
        slots = [s for s in range(P.NSLOT) if (dt.date(2000, 1, 1) + dt.timedelta(days=5 * s)).month in months]
        data, count = [], []
        for s in slots:
            acc = np.zeros(a.shape[1:], np.float64)
            n = np.zeros(a.shape[1:], np.int64)
            for i in range(a.shape[0]):
                y, ss = P.slot_of_bin(g["bin_first"] + i)
                if ss != s or y < ya or y > yb or y in ex:
                    continue
                v = a[i].astype(np.float64) * nm[:, 1] + nm[:, 0]
                f = np.isfinite(v)
                acc += np.where(f, v, 0)
                n += f
            m = np.where(n > 0, acc / np.maximum(n, 1), np.nan)
            data.append(np.moveaxis(m, -1, 0))            # [C, H, W]
            count.append(np.moveaxis(n, -1, 0))
        D = np.stack(data)
        return dict(slots=slots, data=[None if not np.isfinite(x) else float(x) for x in D.ravel()],
                    count=[int(x) for x in np.stack(count).ravel()], shape=list(D.shape))

    cases = [
        dict(name="fxp spring, 1984 left out", group="fxp", ya=1982, yb=1986, ex=[1984], months=[2, 3]),
        dict(name="fxp december incl. the two-bin slot", group="fxp", ya=1983, yb=1985, ex=[], months=[12]),
        dict(name="fxq from its partial first year", group="fxq", ya=1982, yb=1986, ex=[], months=[5, 6]),
    ]
    for c in cases:
        c.update(expect(c["group"], c["ya"], c["yb"], c["ex"], c["months"]))
    P.write_json(os.path.join(OUT, "expected.json"), dict(
        _source="tests/make_gridded_poy_fixture.py — numpy over the native bins", grid=GRID, cases=cases))
    tot = sum(os.path.getsize(os.path.join(dp, f)) for dp, _, fs in os.walk(OUT) for f in fs)
    print(f"wrote {OUT} ({tot / 1e6:.2f} MB)")


if __name__ == "__main__":
    main()

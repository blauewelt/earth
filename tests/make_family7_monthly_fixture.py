#!/usr/bin/env python3
"""tests/make_family7_monthly_fixture.py — a small multi-year fixture of E-086's
per-year monthly sums and counts, for the Data tab's "Monthly normals" stores.

    python3 tests/make_family7_monthly_fixture.py      # writes data/family7_monthly/fixture_multi/

The published fixture (data/family7_monthly/fixture/, written by
ml/export_family7_monthly.py over the smoke tensor) holds ONE year, which can
test neither a period, nor an excluded year, nor a by-year stack. This one
holds the same schema — sum.npy float32 in z-units and count.npy uint8, both
[month, channel, year, lat, lon], C order, an index in the shape of
data/family7_monthly_index.json — over two small regional grids:

  fx025  0.25°, 21 × 29 cells from 30° N / 80° W, 2 channels (sst, ssh),
         years 2000–2004, five-day rows (count 0..7);
  fxrg   1°, 11 × 21 cells, 6 levelled channels (rg_t10/30/50, rg_s10/30/50),
         years 2004–2006, monthly rows (count 0 or 1).

Counts are zero in places (the stored sum is then 0.0, as the exporter
writes), so "NaN where Σcount = 0" is exercised. `expected.json` holds the
answers numpy gives for a handful of selections — Σsum/Σcount over the chosen
years, un-z-scored, natively and pooled onto 1° cells as Σsum/Σcount over the
block — computed here independently of src/f1data.js.
"""
import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(os.path.dirname(HERE), "data", "family7_monthly", "fixture_multi")

GROUPS = {
    "fx025": dict(chans=["sst", "ssh"], units={"sst": "°C", "ssh": "m"},
                  labels={"sst": "Sea-surface temperature", "ssh": "Sea-surface height"},
                  norm=[[18.5, 6.25], [-0.1, 0.5]], lat0=30.0, lon0=-80.0, ny=21, nx=29, step=0.25,
                  years=list(range(2000, 2005)), row_kind="pentad", max_count=7),
    "fxrg": dict(chans=["rg_t10", "rg_t30", "rg_t50", "rg_s10", "rg_s30", "rg_s50"],
                 units={c: ("°C" if "_t" in c else "PSU") for c in ["rg_t10", "rg_t30", "rg_t50", "rg_s10", "rg_s30", "rg_s50"]},
                 labels={c: ("Ocean temperature at " if "_t" in c else "Ocean salinity at ") + c.split("_")[1][1:] + " dbar"
                         for c in ["rg_t10", "rg_t30", "rg_t50", "rg_s10", "rg_s30", "rg_s50"]},
                 norm=[[17.0, 9.0], [16.5, 8.8], [15.9, 8.4], [34.9, 0.6], [34.95, 0.55], [35.0, 0.5]],
                 lat0=30.0, lon0=-80.0, ny=11, nx=21, step=1.0,
                 years=[2004, 2005, 2006], row_kind="monthly", max_count=1),
}


def build(name, g, rng):
    C, Y, H, W = len(g["chans"]), len(g["years"]), g["ny"], g["nx"]
    shape = (12, C, Y, H, W)
    count = rng.integers(0, g["max_count"] + 1, size=shape).astype(np.uint8)
    # a whole land corner with no sample in any year, and one year-month
    # with no rows at all (a gap in the record)
    count[:, :, :, :3, :4] = 0
    count[1, :, Y - 1] = 0
    z = rng.normal(0.0, 1.0, size=shape) * np.maximum(count, 1)
    total = np.where(count > 0, z, 0.0).astype(np.float32)
    d = os.path.join(OUT, name)
    os.makedirs(d, exist_ok=True)
    files = {}
    for fn, arr in (("sum.npy", total), ("count.npy", count)):
        p = os.path.join(d, fn)
        np.save(p, arr)
        with open(p, "rb") as f:
            raw = f.read()
        hl = raw.index(b"\n", 10) + 1                      # the header's own length
        files[fn] = dict(url=f"{name}/{fn}", header_len=hl, itemsize=arr.dtype.itemsize,
                         plane_bytes=H * W * arr.dtype.itemsize, shape=list(shape),
                         dtype=arr.dtype.str, bytes=len(raw), fortran_order=False,
                         axes=["month", "channel", "year", "lat", "lon"])
    return total, count, files


def cells(g, box):
    lat = g["lat0"] + np.arange(g["ny"]) * g["step"]
    lon = g["lon0"] + np.arange(g["nx"]) * g["step"]
    r = np.where((lat >= box["s"]) & (lat <= box["n"]))[0]
    c = np.where((lon >= box["w"]) & (lon <= box["e"]))[0]
    return lat, lon, r, c


def compose(g, total, count, sel):
    """numpy's answer: [T, C, Ho, Wo] physical means and counts."""
    years = [y for y in range(sel["yearStart"], sel["yearEnd"] + 1)
             if y in g["years"] and y not in sel.get("excludeYears", [])]
    yi = [g["years"].index(y) for y in years]
    ci = [g["chans"].index(c) for c in sel["channels"]]
    lat, lon, r, c = cells(g, sel["bbox"])
    months = sorted(sel["months"])
    steps = [(None, m) for m in months] if sel["step"] == "normal" else [(y, m) for y in years for m in months]
    res = sel["res"]
    if res == "native":
        olat, olon = lat[r], lon[c]
        rk, ck = np.arange(len(r)), np.arange(len(c))
    else:
        kr, kc = np.floor(lat[r] / res).astype(int), np.floor(lon[c] / res).astype(int)
        ur, uc = np.unique(kr), np.unique(kc)
        olat, olon = (ur + 0.5) * res, (uc + 0.5) * res
        rk, ck = np.searchsorted(ur, kr), np.searchsorted(uc, kc)
    data = np.full((len(steps), len(ci), len(olat), len(olon)), np.nan)
    cnt = np.zeros(data.shape, np.int64)
    for t, (y, m) in enumerate(steps):
        ys = yi if y is None else [g["years"].index(y)]
        for k, ch in enumerate(ci):
            S = total[m - 1, ch][ys][:, r][:, :, c].astype(np.float64).sum(0)
            N = count[m - 1, ch][ys][:, r][:, :, c].astype(np.int64).sum(0)
            SS = np.zeros((len(olat), len(olon))); NN = np.zeros(SS.shape, np.int64)
            np.add.at(SS, (rk[:, None], ck[None, :]), S)
            np.add.at(NN, (rk[:, None], ck[None, :]), N)
            mu, sd = g["norm"][ch]
            with np.errstate(invalid="ignore", divide="ignore"):
                data[t, k] = np.where(NN > 0, SS / np.where(NN > 0, NN, 1) * sd + mu, np.nan)
            cnt[t, k] = NN
    return years, olat, olon, data, cnt


def main():
    rng = np.random.default_rng(20261006)
    index = {
        "_source": "tests/make_family7_monthly_fixture.py — do not hand-edit",
        "fixture": True, "stem": "fixture_multi",
        "axes": ["month", "channel", "year", "lat", "lon"],
        "combine": "mean_z = Σ sum / Σ count over the chosen years, NaN where Σ count = 0; physical = mean_z * norm[c][1] + norm[c][0]",
        "month_rule": "a bin belongs to the calendar month and year its five-day window OPENS in",
        "groups": {},
    }
    arrays = {}
    for name, g in GROUPS.items():
        total, count, files = build(name, g, rng)
        arrays[name] = (total, count)
        index["groups"][name] = dict(
            chans=g["chans"], units=g["units"], labels=g["labels"], norm=g["norm"],
            grid=dict(lat0=g["lat0"], lon0=g["lon0"], ny=g["ny"], nx=g["nx"], step=g["step"], south_first=True, wrap=False),
            n_years=len(g["years"]), year_first=g["years"][0], year_last=g["years"][-1], years=g["years"],
            max_count=g["max_count"], row_kind=g["row_kind"], static_chans=[],
            sum=files["sum.npy"], count=files["count.npy"])
    with open(os.path.join(OUT, "family7_monthly_index.json"), "w") as f:
        json.dump(index, f, indent=1, ensure_ascii=False)
    box = dict(w=-79.0, s=31.0, e=-74.5, n=34.0)
    cases = [
        ("normal_excl", "fx025", dict(channels=["sst", "ssh"], yearStart=2000, yearEnd=2004, excludeYears=[2002],
                                      months=[2, 7], bbox=box, step="normal", res="native")),
        ("normal_1deg", "fx025", dict(channels=["sst"], yearStart=2000, yearEnd=2004, excludeYears=[2002],
                                      months=[2], bbox=box, step="normal", res=1)),
        ("byyear", "fx025", dict(channels=["sst"], yearStart=2001, yearEnd=2003, excludeYears=[],
                                 months=[2], bbox=box, step="by-year", res="native")),
        ("byyear_1deg", "fx025", dict(channels=["ssh"], yearStart=2000, yearEnd=2004, excludeYears=[2001],
                                      months=[1, 2], bbox=box, step="by-year", res=1)),
        ("rg_level", "fxrg", dict(channels=["rg_t30", "rg_s50"], yearStart=2004, yearEnd=2006, excludeYears=[],
                                  months=[1], bbox=dict(w=-80, s=30, e=-60, n=40), step="normal", res="native")),
    ]
    out = []
    for cname, gname, sel in cases:
        total, count = arrays[gname]
        years, olat, olon, data, cnt = compose(GROUPS[gname], total, count, sel)
        out.append(dict(name=cname, group=gname, sel=dict(sel, family="derived", store="normals_" + gname),
                        years=years, lat=olat.tolist(), lon=olon.tolist(), shape=list(data.shape),
                        data=[None if not np.isfinite(v) else float(v) for v in data.ravel()],
                        count=cnt.ravel().tolist()))
    with open(os.path.join(OUT, "expected.json"), "w") as f:
        json.dump(dict(cases=out), f)
    print("wrote", OUT, {k: os.path.getsize(os.path.join(OUT, k, "sum.npy")) for k in GROUPS})


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Writes `data/poy_layer_fixture/` — the pentad-of-year climatology (E-093)
over two of family 7's REAL group names, for the "Pentad climatology" globe
layer's tests in tests/app.spec.js.

    python3 tests/make_poy_layer_fixture.py        # rewrites the fixture

WHY A SECOND FIXTURE. `data/gridded_poy/fixture/` is E-093's own, written by
the real exporter over synthetic groups (`fxp`, `fxq`) that the family-7 index
does not describe. The globe layer takes each channel's label, unit, ramp and
legend range from `data/family7_index.json`, so its tests need the group names
the real index will carry (`g025`, `g100`) — that path is what this fixture
exercises; the exporter's fixture exercises the fallback to the store's own
`labels` / `units` / `norm`.

PLAIN ENGLISH. Family 7's five-day bins run continuously from 1982-01-01 (bin b
covers days [5b, 5b+5)). A bin belongs to slot min(72, floor((day_of_year(bin
start) − 1) / 5)). Per group and slot: `sum_pSS.npy` float32 [C, Y, H, W] in
PHYSICAL units and `count_pSS.npy` uint8, CUMULATIVE over the years. This
script builds that for `g025` on a 15° grid (13 × 24; sst, ssh, with a NaN
"land" block in sst) and `g100` on a 30° grid (7 × 12; t2m, skt), for
2010–2013, with the REAL bin calendar (2012's leap day slides the bins one day
against the calendar), and writes the index in E-093's shape (base "./",
south_first, labels/units dicts, slot_table) plus `expected.json`: cases
composed here with numpy in float64 from the files' own float32 planes.
Deterministic; about 1.2 MB.
"""
import datetime as dt
import json
import os
import shutil

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "..", "data", "poy_layer_fixture")
EPOCH = dt.date(1982, 1, 1)
YEARS = [2010, 2011, 2012, 2013]
NSLOT = 73

GROUPS = {
    "g025": {"step": 15.0, "chans": ["sst", "ssh"], "units": ["°C", "m"],
             "labels": ["Sea-surface temperature", "Sea surface height"],
             "base": [15.0, 0.0], "amp": [6.0, 0.3], "land": True},
    "g100": {"step": 30.0, "chans": ["t2m", "skt"], "units": ["°C", "°C"],
             "labels": ["Air temperature at 2 m", "Skin temperature"],
             "base": [10.0, 12.0], "amp": [12.0, 14.0], "land": False},
}


def slot_of(d):
    return min(NSLOT - 1, (d.timetuple().tm_yday - 1) // 5)


def bins_of_year(y):
    """The opening dates of every five-day bin whose first day is in year y."""
    k = -(-(dt.date(y, 1, 1) - EPOCH).days // 5)
    out = []
    while True:
        d = EPOCH + dt.timedelta(days=5 * k)
        if d.year != y:
            return out
        out.append(d)
        k += 1


def field(g, c, d, lat, lon):
    G = GROUPS[g]
    phase = 2 * np.pi * (d.timetuple().tm_yday - 1) / 365.25
    v = (G["base"][c] + G["amp"][c] * np.cos(phase) * np.cos(np.deg2rad(lat))
         + 0.25 * (d.year - 2010) + 0.01 * ((d - EPOCH).days % 3)
         + 0.05 * np.sin(np.deg2rad(lon)) * (c + 1))
    if G["land"] and c == 0:
        v = np.where((lon >= 0) & (lon <= 60) & (lat >= -15) & (lat <= 45), np.nan, v)
    return v


def header_len(path):
    h = open(path, "rb").read(10)
    return 10 + int.from_bytes(h[8:10], "little")


def main():
    if os.path.isdir(OUT):
        shutil.rmtree(OUT)
    os.makedirs(OUT)
    Y = len(YEARS)
    index = {
        "_source": "tests/make_poy_layer_fixture.py — the globe layer's test fixture, not the published index",
        "base": "./",
        "combine": "mean of a slot over years a…b = (S[b] − S[a−1]) / (N[b] − N[a−1])",
        "generated_utc": "fixture",
        "layout": "sum_pSS.npy [channel, year, lat, lon] float32 PHYSICAL, count_pSS.npy uint8, cumulative over the years",
        "n_slots": NSLOT,
        "plan": "https://blauewelt.github.io/earth/docs.html?f=ml/plans/E093_pentad_climatology.md",
        "slot_rule": "slot = min(72, floor((day_of_year(bin start, UTC) − 1) / 5)); bin b starts on 1982-01-01 + 5·b days",
        "slot_table": {f"p{s:02d}": {"slot": s, "start_doy": 5 * s + 1, "days": 5 if s < 72 else "5 or 6"}
                       for s in range(NSLOT)},
        "stores": {},
    }
    planes = {}
    for g, G in GROUPS.items():
        step = G["step"]
        H, W = int(180 / step) + 1, int(360 / step)
        lat = (-90 + step * np.arange(H))[:, None] * np.ones((1, W))
        lon = np.ones((H, 1)) * (-180 + step * np.arange(W))[None, :]
        C = len(G["chans"])
        S = np.zeros((NSLOT, C, Y, H, W), np.float64)
        N = np.zeros((NSLOT, C, Y, H, W), np.int64)
        for yi, y in enumerate(YEARS):
            for d in bins_of_year(y):
                s = slot_of(d)
                for c in range(C):
                    v = field(g, c, d, lat, lon)
                    ok = np.isfinite(v)
                    S[s, c, yi] += np.where(ok, v, 0.0)
                    N[s, c, yi] += ok
        Sc = np.cumsum(S, axis=2).astype("<f4")
        Nc = np.cumsum(N, axis=2).astype("u1")
        prefix = f"tensors/family7_global025_pentad_l2/poy/v1/{g}"
        gdir = os.path.join(OUT, prefix)
        os.makedirs(gdir)
        slots = {}
        for s in range(NSLOT):
            k = f"p{s:02d}"
            fs, fc = os.path.join(gdir, f"sum_{k}.npy"), os.path.join(gdir, f"count_{k}.npy")
            np.save(fs, np.ascontiguousarray(Sc[s]))
            np.save(fc, np.ascontiguousarray(Nc[s]))
            slots[k] = {"sum_bytes": os.path.getsize(fs), "count_bytes": os.path.getsize(fc),
                        "sum_header": header_len(fs), "count_header": header_len(fc), "start_doy": 5 * s + 1}
        planes[g] = (Sc, Nc)
        index["stores"][f"family7_global025_pentad_l2/{g}"] = {
            "group": g, "stem": "family7_global025_pentad_l2", "prefix": prefix,
            "H": H, "W": W, "lat0": -90.0, "dlat": step, "lon0": -180.0, "dlon": step, "south_first": True,
            "chans": G["chans"], "units": dict(zip(G["chans"], G["units"])),
            "labels": dict(zip(G["chans"], G["labels"])),
            "norm": [[G["base"][c], G["amp"][c] / 2] for c in range(C)], "zscored_source": True,
            "n_years": Y, "year_first": YEARS[0], "year_last": YEARS[-1], "years": YEARS,
            "record_first": "2010-01-01", "record_last": "2013-12-31",
            "slots": slots, "complete": True,
        }
    with open(os.path.join(OUT, "gridded_poy_index.json"), "w") as f:
        json.dump(index, f, indent=1, ensure_ascii=False)

    def mean(g, s, c, iy, ix, a, b, ex=()):
        Sc, Nc = planes[g]
        y0 = YEARS[0]
        sign = {}
        def add(y, v):
            if y >= y0:
                sign[y] = sign.get(y, 0) + v
        add(b, 1); add(a - 1, -1)
        for e in ex:
            add(e, -1); add(e - 1, 1)
        Ssum = sum(v * float(Sc[s, c, y - y0, iy, ix]) for y, v in sign.items())
        Nsum = sum(v * int(Nc[s, c, y - y0, iy, ix]) for y, v in sign.items())
        return (Ssum / Nsum if Nsum > 0 else None), Nsum

    cases = []
    def case(name, g, chan, date, a, b, ex, iy, ix):
        G = GROUPS[g]
        d = dt.date.fromisoformat(date)
        b0 = EPOCH + dt.timedelta(days=5 * ((d - EPOCH).days // 5))
        s = slot_of(b0)
        m, n = mean(g, s, G["chans"].index(chan), iy, ix, a, b, ex)
        cases.append({"name": name, "group": g, "chan": chan, "date": date, "bin_start": b0.isoformat(),
                      "slot": s, "years": [a, b], "exclude": list(ex), "iy": iy, "ix": ix,
                      "lat": -90 + G["step"] * iy, "lon": -180 + G["step"] * ix, "mean": m, "count": n})
    # 2011-03-03 is in the bin opening 2011-02-28 (day 59, slot 11); 2013-03-01 is in the bin
    # opening 2013-02-27 (day 58, slot 11 again) — same slot, another year, another calendar day
    case("whole record", "g100", "t2m", "2011-03-03", 2010, 2013, [], 4, 7)
    case("same slot, another year", "g100", "t2m", "2013-03-01", 2010, 2013, [], 4, 7)
    case("a shorter span", "g100", "t2m", "2011-03-03", 2011, 2012, [], 4, 7)
    case("a year left out", "g100", "t2m", "2011-03-03", 2010, 2013, [2012], 4, 7)
    case("g025 sst whole record", "g025", "sst", "2011-07-20", 2010, 2013, [], 8, 4)
    case("g025 sst over land", "g025", "sst", "2011-07-20", 2010, 2013, [], 8, 13)
    case("last slot", "g100", "t2m", "2012-12-31", 2010, 2013, [], 4, 7)
    with open(os.path.join(OUT, "expected.json"), "w") as f:
        json.dump({"cases": cases}, f, indent=1)
    tot = sum(os.path.getsize(os.path.join(r, x)) for r, _, fs in os.walk(OUT) for x in fs)
    print(f"wrote {OUT}: {tot / 1e6:.2f} MB")
    for c in cases:
        print(c["name"], c["date"], "bin", c["bin_start"], "slot", c["slot"], c["mean"], c["count"])


if __name__ == "__main__":
    main()

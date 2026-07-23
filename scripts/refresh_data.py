#!/usr/bin/env python3
"""Refresh bundled data snapshots for the earth globe.

Produces (relative to repo root):
  data/climatetrace.json  - top facility-level emitters (Climate TRACE, CC BY 4.0)
  data/argo.json          - latest Argo float positions (Argo GDAC via Ifremer ERDDAP)
  data/rapid_moc.json     - RAPID 26.5N overturning transport time series (rapid.ac.uk)

Run from the repo root:  python3 scripts/refresh_data.py
Requires: netCDF4 (pip install netCDF4)
"""
import json
import os
import sys
import time
import urllib.parse
import urllib.request
from datetime import datetime, timedelta, timezone

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
DATA = os.path.join(ROOT, "data")
UA = {"User-Agent": "earth-globe/1.0 (github.com/blauewelt/earth)"}


def fetch_json(url):
    req = urllib.request.Request(url, headers=UA)
    with urllib.request.urlopen(req, timeout=120) as r:
        return json.load(r)


def climatetrace(n=1000, year=2024):
    """Top-N facility-level emitters, all sectors, sorted by CO2e (API default)."""
    print(f"Climate TRACE: fetching top {n} assets for {year} ...")
    assets, offset = [], 0
    while len(assets) < n:
        limit = min(250, n - len(assets))
        url = f"https://api.climatetrace.org/v6/assets?limit={limit}&offset={offset}&year={year}"
        batch = fetch_json(url).get("assets", [])
        if not batch:
            break
        assets.extend(batch)
        offset += limit
        time.sleep(0.5)
    out = []
    for a in assets:
        c = (a.get("Centroid") or {}).get("Geometry")
        em = [e for e in a.get("EmissionsSummary", []) if e.get("Gas") == "co2e_100yr"]
        q = em[0].get("EmissionsQuantity") if em else None
        if not c or q is None:
            continue
        out.append([
            round(c[0], 4), round(c[1], 4),
            round(q / 1e6, 3),                      # Mt CO2e / yr
            a.get("Name", "")[:80],
            a.get("Country", ""),
            a.get("Sector", ""),
        ])
    payload = {
        "source": "Climate TRACE (climatetrace.org), CC BY 4.0",
        "year": year,
        "fields": ["lon", "lat", "mt_co2e", "name", "country", "sector"],
        "snapshot": datetime.now(timezone.utc).strftime("%Y-%m-%d"),
        "assets": out,
    }
    with open(os.path.join(DATA, "climatetrace.json"), "w") as f:
        json.dump(payload, f, separators=(",", ":"))
    print(f"  wrote {len(out)} assets")


def argo(days=10):
    """Latest position of every float reporting in the last `days` days."""
    print(f"Argo: fetching positions from last {days} days ...")
    since = (datetime.now(timezone.utc) - timedelta(days=days)).strftime("%Y-%m-%dT00:00:00Z")
    q = ("platform_number%2Clatitude%2Clongitude%2Ctime"
         f"&time%3E={urllib.parse.quote(since)}&distinct()")
    url = f"https://erddap.ifremer.fr/erddap/tabledap/ArgoFloats.json?{q}"
    rows = fetch_json(url)["table"]["rows"]
    latest = {}
    for pn, lat, lon, t in rows:
        if lat is None or lon is None:
            continue
        if pn not in latest or t > latest[pn][2]:
            latest[pn] = (round(lon, 3), round(lat, 3), t)
    out = [[v[0], v[1], k, v[2][:10]] for k, v in latest.items()]
    payload = {
        "source": "Argo GDAC via Ifremer ERDDAP (doi:10.17882/42182)",
        "fields": ["lon", "lat", "float_id", "date"],
        "snapshot": datetime.now(timezone.utc).strftime("%Y-%m-%d"),
        "floats": out,
    }
    with open(os.path.join(DATA, "argo.json"), "w") as f:
        json.dump(payload, f, separators=(",", ":"))
    print(f"  wrote {len(out)} float positions")


def rapid(step=20):
    """RAPID 26.5N transports, downsampled from 12-hourly to `step`-sample means (10 days at step=20)."""
    print("RAPID: fetching moc_transports.nc ...")
    import netCDF4
    import numpy as np
    url = "https://rapid.ac.uk/sites/default/files/rapid_data/moc_transports.nc"
    tmp = "/tmp/moc_transports.nc"
    req = urllib.request.Request(url, headers=UA)
    with urllib.request.urlopen(req, timeout=300) as r, open(tmp, "wb") as f:
        f.write(r.read())
    ds = netCDF4.Dataset(tmp)
    t = ds.variables["time"]
    dates = netCDF4.num2date(t[:], t.units)

    def series(name):
        v = np.ma.masked_invalid(ds.variables[name][:].astype(float))
        n = (len(v) // step) * step
        blk = v[:n].reshape(-1, step)
        m = blk.mean(axis=1)
        return [None if x is np.ma.masked else round(float(x), 2) for x in m]

    n = (len(dates) // step) * step
    dts = [dates[i].strftime("%Y-%m-%d") for i in range(step // 2, n, step)]
    payload = {
        "source": "RAPID-MOCHA-WBTS array, rapid.ac.uk (NOC/U. Miami/NOAA)",
        "citation": "Moat et al.; doi:10.5285/48d0bf43-0598-ceb2-e063-7086abc062f1",
        "units": "Sv",
        "resolution_days": step / 2,
        "snapshot": datetime.now(timezone.utc).strftime("%Y-%m-%d"),
        "t": dts,
        "moc": series("moc_mar_hc10"),
        "gulf_stream": series("t_gs10"),
        "ekman": series("t_ek10"),
        "upper_mid_ocean": series("t_umo10"),
    }
    with open(os.path.join(DATA, "rapid_moc.json"), "w") as f:
        json.dump(payload, f, separators=(",", ":"))
    print(f"  wrote {len(dts)} samples ({dts[0]} .. {dts[-1]})")


def sealevel():
    """Global sea-level budget: observed GMSL vs its components (steric, glaciers,
    Greenland, Antarctica, terrestrial water storage), 1900-2018, from
    Frederikse et al. 2020 (Nature); plus the satellite-altimetry total from
    NOAA STAR for the modern era. Illustrates budget closure: total ≈ sum of parts."""
    import io
    import openpyxl
    print("Sea level: Frederikse et al. 2020 global budget ...")
    raw = urllib.request.urlopen(urllib.request.Request(
        "https://zenodo.org/records/3862995/files/global_basin_timeseries.xlsx", headers=UA), timeout=120).read()
    wb = openpyxl.load_workbook(io.BytesIO(raw), read_only=True)
    ws = wb["Global"]
    col = {  # column index → output key (means only)
        2: "observed", 5: "sum", 8: "steric", 11: "glaciers",
        14: "greenland", 17: "antarctica", 20: "tws",
    }
    years, series = [], {k: [] for k in col.values()}
    for row in ws.iter_rows(min_row=2, values_only=True):
        if row[0] is None:
            continue
        years.append(int(float(row[0])))
        for i, k in col.items():
            v = row[i]
            series[k].append(round(float(v), 1) if v is not None else None)

    print("Sea level: NOAA STAR satellite altimetry ...")
    txt = urllib.request.urlopen(urllib.request.Request(
        "https://www.star.nesdis.noaa.gov/socd/lsa/SeaLevelRise/slr/slr_sla_gbl_keep_all_66.csv",
        headers=UA), timeout=120).read().decode()
    alt_t, alt_v = [], []
    for line in txt.splitlines():
        if line.startswith("#") or line.startswith("year") or not line.strip():
            continue
        parts = line.split(",")
        t = float(parts[0])
        vals = [float(p) for p in parts[1:] if p.strip()]
        if vals:
            alt_t.append(round(t, 3))
            alt_v.append(round(vals[-1], 1))  # latest available mission
    # rebase altimetry so its 2005 value ~ observed-2005 (both mm, arbitrary datum)
    base_alt = next((v for t, v in zip(alt_t, alt_v) if t >= 2005), alt_v[0])
    base_obs = series["observed"][years.index(2005)] if 2005 in years else 0
    alt_v = [round(v - base_alt + base_obs, 1) for v in alt_v]

    payload = {
        "source": "Frederikse et al. 2020, Nature (doi:10.1038/s41586-020-2591-3); "
                  "satellite altimetry: NOAA/NESDIS Laboratory for Satellite Altimetry",
        "units": "mm (relative to 2002-2018 mean baseline of the source)",
        "snapshot": datetime.now(timezone.utc).strftime("%Y-%m-%d"),
        "years": years,
        "components": series,           # observed, steric, glaciers, greenland, antarctica, tws
        "altimetry": {"t": alt_t, "v": alt_v},
    }
    with open(os.path.join(DATA, "sealevel.json"), "w") as f:
        json.dump(payload, f, separators=(",", ":"))
    print(f"  wrote {len(years)} yr of budget + {len(alt_t)} altimetry samples")


if __name__ == "__main__":
    which = sys.argv[1:] or ["climatetrace", "argo", "rapid", "sealevel"]
    for w in which:
        {"climatetrace": climatetrace, "argo": argo, "rapid": rapid, "sealevel": sealevel}[w]()
    print("done")

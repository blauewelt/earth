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


if __name__ == "__main__":
    which = sys.argv[1:] or ["climatetrace", "argo", "rapid"]
    for w in which:
        {"climatetrace": climatetrace, "argo": argo, "rapid": rapid}[w]()
    print("done")

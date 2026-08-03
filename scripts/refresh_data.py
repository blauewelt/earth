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


def climatetrace(n=1000, years=range(2021, 2026)):
    """Top-N facility-level emitters per YEAR (all sectors, sorted by CO2e).
    Climate TRACE is an annual inventory — the app reads whichever year the
    date selector points at, so we bake every available year (2021-2025 as of
    v6). Each year is an independent top-N, so a facility can appear in some
    years and not others."""
    def one_year(year):
        print(f"Climate TRACE {year}: fetching top {n} ...")
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
                round(q / 1e6, 3),                  # Mt CO2e / yr
                a.get("Name", "")[:80],
                a.get("Country", ""),
                a.get("Sector", ""),
            ])
        return out

    years = [y for y in years]
    by_year = {}
    for y in years:
        rows = one_year(y)
        if rows:
            by_year[str(y)] = rows
    avail = sorted(int(y) for y in by_year)
    payload = {
        "source": "Climate TRACE (climatetrace.org), CC BY 4.0",
        "fields": ["lon", "lat", "mt_co2e", "name", "country", "sector"],
        "snapshot": datetime.now(timezone.utc).strftime("%Y-%m-%d"),
        "years": avail,                             # e.g. [2021,2022,2023,2024,2025]
        "assets_by_year": by_year,
    }
    with open(os.path.join(DATA, "climatetrace.json"), "w") as f:
        json.dump(payload, f, separators=(",", ":"))
    print(f"  wrote {sum(len(v) for v in by_year.values())} assets across "
          f"years {avail[0]}-{avail[-1]}")


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


RGI_REGIONS = [
    "01_alaska", "02_western_canada_usa", "03_arctic_canada_north", "04_arctic_canada_south",
    "05_greenland_periphery", "06_iceland", "07_svalbard_jan_mayen", "08_scandinavia",
    "09_russian_arctic", "10_north_asia", "11_central_europe", "12_caucasus_middle_east",
    "13_central_asia", "14_south_asia_west", "15_south_asia_east", "16_low_latitudes",
    "17_southern_andes", "18_new_zealand", "19_subantarctic_antarctic_islands",
]
RGI_NAMES = {  # o1region → short label for tooltips
    "01": "Alaska", "02": "W Canada & US", "03": "Arctic Canada N", "04": "Arctic Canada S",
    "05": "Greenland periphery", "06": "Iceland", "07": "Svalbard", "08": "Scandinavia",
    "09": "Russian Arctic", "10": "North Asia", "11": "Central Europe", "12": "Caucasus & M.East",
    "13": "Central Asia", "14": "South Asia West", "15": "South Asia East", "16": "Low latitudes",
    "17": "Southern Andes", "18": "New Zealand", "19": "Subantarctic & Antarctic",
}


def glaciers():
    """Every glacier in RGI v7 (G product, ~274k) as centroid + area, joined with
    per-glacier elevation-change rate (dhdt, m/yr, 2000-2020) from Hugonnet et al.
    2021 — so each glacier can be coloured by how fast it is actually thinning."""
    import csv, io, tarfile
    import pandas as pd
    base = "https://cluster.klima.uni-bremen.de/~fmaussion/misc/rgi7_data/l4_rgi7b0_tar/"
    regions = [r.replace("_", "_", 1) for r in RGI_REGIONS]

    print("RGI7: Hugonnet 2021 per-glacier dhdt ...")
    hug = pd.read_parquet(io.BytesIO(urllib.request.urlopen(urllib.request.Request(
        "https://cluster.klima.uni-bremen.de/~oggm/geodetic_ref_mb/"
        "hugonnet_2021_ds_rgi70_pergla_rates_10_20.parquet", headers=UA), timeout=300).read()))
    hug = hug[hug["period"] == "2000-01-01_2020-01-01"]
    dhdt_by_id = hug["dhdt"].to_dict()   # rgiid -> m/yr

    lon, lat, area, dhdt = [], [], [], []
    for r in RGI_REGIONS:
        url = f"{base}RGI2000-v7.0-G-{r}.tar.gz"
        print(f"RGI7 G: {r} ...")
        raw = urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=300).read()
        with tarfile.open(fileobj=io.BytesIO(raw), mode="r:gz") as tf:
            member = next(m for m in tf.getmembers() if m.name.endswith("-attributes.csv"))
            for row in csv.DictReader(io.TextIOWrapper(tf.extractfile(member), encoding="utf-8")):
                try:
                    lon.append(round(float(row["cenlon"]), 3))
                    lat.append(round(float(row["cenlat"]), 3))
                    area.append(round(float(row["area_km2"]), 3))
                    d = dhdt_by_id.get(row["rgi_id"])
                    dhdt.append(round(float(d), 3) if d is not None and d == d else None)
                except (ValueError, KeyError):
                    continue
    matched = sum(1 for d in dhdt if d is not None)
    payload = {
        "source": "Randolph Glacier Inventory v7.0 (rgidata.org, CC BY 4.0); "
                  "elevation-change rate: Hugonnet et al. 2021, Nature (doi:10.1038/s41586-021-03436-z)",
        "note": "One point per glacier at its centroid, sized by area. dhdt = surface "
                "elevation change rate 2000-2020 (m/yr); negative = thinning/melting.",
        "region_names": RGI_NAMES,
        "count": len(lon),
        "total_area_km2": round(sum(area)),
        "dhdt_matched": matched,
        "snapshot": datetime.now(timezone.utc).strftime("%Y-%m-%d"),
        "lon": lon, "lat": lat, "area": area, "dhdt": dhdt,
    }
    with open(os.path.join(DATA, "glaciers.json"), "w") as f:
        json.dump(payload, f, separators=(",", ":"))
    print(f"  wrote {len(lon)} glaciers ({matched} with dhdt), total {payload['total_area_km2']:,} km2")

def gistemp():
    """GISTEMP v4 global temperature anomaly (NASA GISS): land+ocean and land-only
    (met-station) annual means, 1880-present. Land warms faster than the global mean."""
    import csv, io
    def series(url):
        txt = urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=120).read().decode()
        yrs, vals = [], []
        for row in csv.reader(io.StringIO(txt)):
            if not row or not row[0].isdigit():
                continue
            jd = row[13]  # J-D = annual mean column
            if jd in ("", "***", "*****"):
                continue
            yrs.append(int(row[0]))
            vals.append(round(float(jd), 2))
        return yrs, vals
    print("GISTEMP: land+ocean and land-only ...")
    ly, lo = series("https://data.giss.nasa.gov/gistemp/tabledata_v4/GLB.Ts+dSST.csv")
    ky, land = series("https://data.giss.nasa.gov/gistemp/tabledata_v4/GLB.Ts.csv")
    # align on common years
    landmap = dict(zip(ky, land))
    payload = {
        "source": "NASA GISS Surface Temperature Analysis (GISTEMP v4)",
        "citation": "GISTEMP Team 2026; Lenssen et al. 2019, doi:10.1029/2018JD029522",
        "baseline": "anomaly vs 1951-1980 mean (°C)",
        "units": "°C",
        "snapshot": datetime.now(timezone.utc).strftime("%Y-%m-%d"),
        "years": ly,
        "land_ocean": lo,
        "land_only": [landmap.get(y) for y in ly],
    }
    with open(os.path.join(DATA, "gistemp.json"), "w") as f:
        json.dump(payload, f, separators=(",", ":"))
    print(f"  wrote {len(ly)} yr ({ly[0]}-{ly[-1]}); latest land+ocean {lo[-1]}, land {landmap.get(ly[-1])}")


# --------------------------------------------------------------- gridded fields
# GPCP, E-OBS, OISST and MeteoSwiss have no global tile service, so we bake a
# static regular lon/lat grid the browser paints with GridProvider. One helper
# resamples any source (regular or curvilinear) onto a target grid by nearest
# scatter-binning, so every dataset flows through the same code path.

def _download(url, path, note=""):
    if os.path.exists(path):
        print(f"  cached {os.path.basename(path)}{note}")
        return path
    print(f"  downloading {url} ...")
    req = urllib.request.Request(url, headers=UA)
    with urllib.request.urlopen(req, timeout=600) as r, open(path, "wb") as f:
        while True:
            chunk = r.read(1 << 20)
            if not chunk:
                break
            f.write(chunk)
    print(f"    {os.path.getsize(path) / 1e6:.0f} MB{note}")
    return path


def _bin_to_grid(lon, lat, val, west, south, east, north, nx, ny):
    """Nearest scatter-bin source points (any shape) onto a regular grid.
    Returns a flat row-major list (row 0 = southmost), None for empty cells."""
    import numpy as np
    lon = np.asarray(lon, float).ravel()
    lat = np.asarray(lat, float).ravel()
    val = np.asarray(val, float).ravel()
    lon = ((lon + 180.0) % 360.0) - 180.0          # wrap to [-180,180)
    m = np.isfinite(val) & np.isfinite(lon) & np.isfinite(lat)
    lon, lat, val = lon[m], lat[m], val[m]
    dlon = (east - west) / nx
    dlat = (north - south) / ny
    ix = np.floor((lon - west) / dlon).astype(int)
    iy = np.floor((lat - south) / dlat).astype(int)
    keep = (ix >= 0) & (ix < nx) & (iy >= 0) & (iy < ny)
    ix, iy, val = ix[keep], iy[keep], val[keep]
    flat = iy * nx + ix
    ssum = np.bincount(flat, weights=val, minlength=nx * ny)
    scnt = np.bincount(flat, minlength=nx * ny)
    out = np.where(scnt > 0, ssum / np.maximum(scnt, 1), np.nan)
    return out, scnt, dlon, dlat


def _write_grid(id, path, lon, lat, val, bounds, nx, ny, *, units, title,
                source, citation, ramp, vmin, vmax, decimals=0, doc=""):
    import numpy as np
    west, south, east, north = bounds
    out, scnt, dlon, dlat = _bin_to_grid(lon, lat, val, west, south, east, north, nx, ny)
    vals = [None if not np.isfinite(v) else round(float(v), decimals) for v in out]
    if decimals == 0:
        vals = [None if v is None else int(v) for v in vals]
    filled = int((scnt > 0).sum())
    payload = {
        "id": id, "title": title, "units": units, "source": source,
        "citation": citation, "doc": doc, "ramp": ramp,
        "vmin": vmin, "vmax": vmax,
        "west": west, "south": south, "east": east, "north": north,
        "dlon": round(dlon, 6), "dlat": round(dlat, 6), "nx": nx, "ny": ny,
        "snapshot": datetime.now(timezone.utc).strftime("%Y-%m-%d"),
        "values": vals,
    }
    with open(os.path.join(DATA, path), "w") as f:
        json.dump(payload, f, separators=(",", ":"))
    vfin = [v for v in vals if v is not None]
    print(f"  wrote {path}: {nx}x{ny}, {filled} filled cells, "
          f"value range {min(vfin):.0f}..{max(vfin):.0f} {units}")


def gpcp():
    """GPCP v2.3 global monthly precipitation (NOAA PSL), 2.5 deg. We average the
    full record into a mean-annual climatology (mm/year). Global, coarse, complete."""
    import numpy as np
    print("GPCP v2.3: global precipitation climatology ...")
    nc = _download("https://downloads.psl.noaa.gov/Datasets/gpcp/precip.mon.mean.nc",
                   "/tmp/nc/gpcp.nc")
    import netCDF4
    d = netCDF4.Dataset(nc)
    lat = d.variables["lat"][:]
    lon = d.variables["lon"][:]
    p = d.variables["precip"]
    clim = np.ma.filled(p[:].mean(axis=0), np.nan) * 365.25   # mm/day -> mm/year
    lon2, lat2 = np.meshgrid(lon, lat)
    _write_grid("gpcp", "gpcp.json", lon2, lat2, clim,
                (-180, -90, 180, 90), 144, 72,
                units="mm/yr", title="Precipitation climatology (GPCP v2.3)",
                source="NOAA GPCP v2.3 monthly (PSL)",
                citation="Adler et al. 2018, doi:10.3390/atmos9040138",
                doc="https://psl.noaa.gov/data/gridded/data.gpcp.html",
                ramp="precip", vmin=0, vmax=3000)


def eobs():
    """E-OBS v31 daily precipitation (rr) ensemble mean, 0.25 deg, Europe. We read
    the record in time-chunks to bound memory, average to mean-annual mm/year."""
    import numpy as np
    print("E-OBS v31: European precipitation climatology ...")
    nc = _download(
        "https://knmi-ecad-assets-prd.s3.amazonaws.com/ensembles/data/"
        "Grid_0.25deg_reg_ensemble/rr_ens_mean_0.25deg_reg_v31.0e.nc",
        "/tmp/nc/eobs_rr.nc", note=" (E-OBS is Europe-only)")
    import netCDF4
    d = netCDF4.Dataset(nc)
    lat = d.variables["latitude"][:]
    lon = d.variables["longitude"][:]
    rr = d.variables["rr"]
    nt = rr.shape[0]
    ssum = np.zeros(rr.shape[1:], np.float64)
    scnt = np.zeros(rr.shape[1:], np.float64)
    step = 730
    for t0 in range(0, nt, step):
        block = rr[t0:t0 + step]                       # (chunk, ny, nx) masked mm/day
        arr = np.ma.filled(block.astype(np.float64), np.nan)
        ssum += np.nansum(arr, axis=0)
        scnt += np.sum(np.isfinite(arr), axis=0)
    mean_daily = np.where(scnt > 0, ssum / np.maximum(scnt, 1), np.nan)
    clim = mean_daily * 365.25                          # mm/day -> mm/year
    lon2, lat2 = np.meshgrid(lon, lat)
    west, east = float(lon.min()), float(lon.max())
    south, north = float(lat.min()), float(lat.max())
    nx, ny = len(lon), len(lat)
    _write_grid("eobs", "eobs.json", lon2, lat2, clim,
                (west, south, east, north), nx, ny,
                units="mm/yr", title="Precipitation climatology (E-OBS v31, Europe)",
                source="E-OBS v31 0.25 deg ensemble mean (ECA&D / Copernicus)",
                citation="Cornes et al. 2018, doi:10.1029/2017JD028200",
                doc="https://surfobs.climate.copernicus.eu/dataaccess/access_eobs.php",
                ramp="precip", vmin=0, vmax=2500)


def oisst():
    """NOAA OISST v2.1 high-res sea-surface temperature (PSL), 1991-2020 monthly
    long-term-mean climatology. Global 0.25 deg source, coarsened to 1 deg."""
    import numpy as np
    print("OISST v2.1: mean SST climatology ...")
    nc = _download(
        "https://downloads.psl.noaa.gov/Datasets/noaa.oisst.v2.highres/sst.mon.ltm.1991-2020.nc",
        "/tmp/nc/oisst_ltm.nc")
    import netCDF4
    d = netCDF4.Dataset(nc)
    lat = d.variables["lat"][:]
    lon = d.variables["lon"][:]
    sst = d.variables["sst"]
    clim = np.ma.filled(sst[:].mean(axis=0), np.nan)   # annual mean of 12 monthly LTMs, deg C
    lon2, lat2 = np.meshgrid(lon, lat)
    _write_grid("oisst", "oisst.json", lon2, lat2, clim,
                (-180, -90, 180, 90), 360, 180,
                units="°C", title="Sea surface temperature climatology (OISST v2.1)",
                source="NOAA OISST v2.1 1991-2020 LTM (PSL)",
                citation="Huang et al. 2021, doi:10.1175/JCLI-D-20-0166.1",
                doc="https://psl.noaa.gov/data/gridded/data.noaa.oisst.v2.highres.html",
                ramp="sst", vmin=-2, vmax=32, decimals=1)


def meteoswiss():
    """MeteoSwiss OGD gridded climate normals: mean yearly precipitation 1991-2020
    (RnormY9120) over Switzerland. Curvilinear source ships lon/lat, so no reproj."""
    import numpy as np
    print("MeteoSwiss OGD: Swiss precipitation normal 1991-2020 ...")
    nc = _download(
        "https://data.geo.admin.ch/ch.meteoschweiz.ogd-climate-normals-grid/ch/"
        "ogd-climate-normals-grid.rnormy9120_ch01r.swiss.lv95_19910101000000_19910101000000.nc",
        "/tmp/nc/ch_precip.nc")
    import netCDF4
    d = netCDF4.Dataset(nc)
    lon = d.variables["lon"][:]
    lat = d.variables["lat"][:]
    rr = np.ma.filled(d.variables["RnormY9120"][0].astype(float), np.nan)   # mm/year
    west, east = float(np.nanmin(lon)), float(np.nanmax(lon))
    south, north = float(np.nanmin(lat)), float(np.nanmax(lat))
    nx = int(round((east - west) / 0.02))
    ny = int(round((north - south) / 0.02))
    _write_grid("meteoswiss", "meteoswiss.json", lon, lat, rr,
                (west, south, east, north), nx, ny,
                units="mm/yr", title="Precipitation normal (MeteoSwiss, 1991-2020)",
                source="MeteoSwiss OGD climate normals grid (ch01r, CC BY 4.0)",
                citation="MeteoSwiss OGD; RnormY9120 1991-2020",
                doc="https://opendatadocs.meteoswiss.ch/",
                ramp="precip", vmin=0, vmax=2500)


def species():
    """GBIF biodiversity picker: live occurrence counts per broad taxonomic group
    (kingdoms, major animal/plant classes, humans) plus curated climate-indicator
    species. The 'all recorded life' total splits into eight kingdoms; a residual
    is identified only to 'life' (no kingdom)."""
    def cnt(k):
        u = f"https://api.gbif.org/v1/occurrence/search?limit=0&taxonKey={k}"
        return fetch_json(u)["count"]
    total = fetch_json("https://api.gbif.org/v1/occurrence/search?limit=0")["count"]
    print(f"GBIF: total occurrences {total:,}")
    groups = {
        "Kingdoms (all life splits into these)": [
            (1, "Animals (Animalia)"), (6, "Plants (Plantae)"), (5, "Fungi"),
            (3, "Bacteria"), (4, "Algae &amp; protists (Chromista)"),
            (7, "Protozoa"), (2, "Archaea"), (8, "Viruses")],
        "Major animal groups": [
            (212, "Birds (Aves)"), (216, "Insects (Insecta)"), (359, "Mammals (Mammalia)"),
            (131, "Amphibians (Amphibia)"), (11592253, "Reptiles: lizards &amp; snakes (Squamata)"),
            (121, "Sharks &amp; rays (Elasmobranchii)"), (367, "Arachnids (Arachnida)"),
            (225, "Snails &amp; slugs (Gastropoda)")],
        "Major plant groups": [
            (220, "Flowering plants — dicots (Magnoliopsida)"),
            (196, "Monocots: grasses, orchids (Liliopsida)")],
        "Us": [(2436436, "Humans (Homo sapiens)")],
    }
    categories, kingdom_sum = [], 0
    for label, items in groups.items():
        out = []
        for k, name in items:
            c = cnt(k)
            out.append({"key": k, "name": name, "records": c})
            if label.startswith("Kingdoms"):
                kingdom_sum += c
        categories.append({"label": label, "items": out})
    indicators = [
        {"key": 2480876, "common": "Little egret", "records": cnt(2480876),
         "note": "Wetland wading bird expanding poleward as winters warm — a visible marker of range shift."},
        {"key": 2475443, "common": "European bee-eater", "records": cnt(2475443),
         "note": "Warmth-loving bird now breeding far poleward of its former Mediterranean range."},
        {"key": 1898544, "common": "Comma butterfly", "records": cnt(1898544),
         "note": "One of the fastest range-expanding butterflies as the climate warms."},
        {"key": 1340503, "common": "Buff-tailed bumblebee", "records": cnt(1340503),
         "note": "Pollinator whose range and phenology are shifting poleward with temperature."},
        {"key": 2374149, "common": "Atlantic mackerel", "records": cnt(2374149),
         "note": "Fish stock shifting poleward with ocean warming, straining fishery treaties."},
        {"key": 2481661, "common": "Emperor penguin", "records": cnt(2481661),
         "note": "Sea-ice-dependent breeder; a climate-vulnerability icon of Antarctica."},
        {"key": 7673664, "common": "Staghorn coral", "records": cnt(7673664),
         "note": "Reef-building coral acutely sensitive to marine heatwaves and bleaching."},
        {"key": 5219303, "common": "Arctic fox", "records": cnt(5219303),
         "note": "Cold-adapted mammal squeezed poleward by the advancing red fox."},
    ]
    payload = {
        "source": "GBIF.org occurrence counts (live snapshot). Map tiles: GBIF occurrence density.",
        "snapshot": datetime.now(timezone.utc).strftime("%Y-%m-%d"),
        "total": total, "unplaced": total - kingdom_sum,
        "note": ("GBIF holds ~%.1f billion dated, located records of where life has been observed. "
                 "Every record rolls up into one of eight kingdoms; ~%.1f M are identified only to "
                 "'life' (no kingdom). Coverage is wildly uneven — birds alone are the majority of "
                 "animal records, a birdwatching effect, not because birds outnumber insects. Humans "
                 "are recorded too (Homo sapiens), but GBIF restricts human occurrences for privacy, "
                 "so despite 8 billion of us only tens of thousands of records exist."
                 % (total / 1e9, (total - kingdom_sum) / 1e6)),
        "categories": categories,
        "species": indicators,
    }
    with open(os.path.join(DATA, "species.json"), "w") as f:
        json.dump(payload, f, separators=(",", ":"))
    print(f"  wrote species.json: {sum(len(c['items']) for c in categories)} groups + "
          f"{len(indicators)} species; unplaced {total - kingdom_sum:,}")


def eei():
    """Earth's Energy Imbalance panel data, from the NOAA/NCEI global ocean
    heat content series (open, no account): yearly OHC anomalies for 0-700 m
    (1955-) and 0-2000 m (2005-), plus derived heating RATES as centred 5-yr
    OLS slopes converted to W per m^2 of the WHOLE Earth surface. The ocean
    takes ~90% of the planetary imbalance, so rate/0.9 approximates total EEI
    (von Schuckmann et al.); both numbers ship, labelled as what they are.
    Unit chain: 1e22 J/yr / (5.101e14 m^2 * 3.1557e7 s) = 0.6213 W/m^2."""
    base = ("https://www.ncei.noaa.gov/data/oceans/woa/DATA_ANALYSIS/"
            "3M_HEAT_CONTENT/DATA/basin/yearly")
    W_PER = 1e22 / (5.101e14 * 3.1557e7)

    def series(fname):
        req = urllib.request.Request(f"{base}/{fname}", headers=UA)
        with urllib.request.urlopen(req, timeout=120) as r:
            txt = r.read().decode()
        years, vals, errs = [], [], []
        for ln in txt.splitlines()[1:]:
            p = ln.split()
            if len(p) >= 3:
                years.append(int(float(p[0])))
                vals.append(float(p[1]))
                errs.append(float(p[2]))
        return years, vals, errs

    def slope(xs, ys):                       # OLS, units of ys per year
        n = len(xs)
        mx, my = sum(xs) / n, sum(ys) / n
        den = sum((x - mx) ** 2 for x in xs)
        return sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / den

    def rolling_rate(years, vals, w=5):      # centred w-yr slope -> W/m^2 Earth
        out = []
        h = w // 2
        for i in range(len(years)):
            a, b = max(0, i - h), min(len(years), i + h + 1)
            out.append(round(slope(years[a:b], vals[a:b]) * W_PER, 3)
                       if b - a >= 3 else None)
        return out

    y7, v7, e7 = series("h22-w0-700m.dat")
    y2, v2, e2 = series("h22-w0-2000m.dat")
    r10 = slope(y2[-10:], v2[-10:]) * W_PER               # headline: last decade

    # ENSO state per year from NOAA CPC's ONI. Convention: a calendar year is
    # labelled by its DJF value (the season ENSO peaks in — DJF 1998 = the
    # 97/98 El Nino, marking 1998, the year the world felt it). >=+0.5 El
    # Nino, <=-0.5 La Nina. During El Nino the ocean SHEDS heat to the
    # atmosphere, so OHC growth dips; La Nina banks heat — the rate chart's
    # wiggles line up with these bands.
    req = urllib.request.Request(
        "https://www.cpc.ncep.noaa.gov/data/indices/oni.ascii.txt", headers=UA)
    with urllib.request.urlopen(req, timeout=60) as r:
        oni_txt = r.read().decode()
    oni = {}
    for ln in oni_txt.splitlines():
        p = ln.split()
        if len(p) == 4 and p[0] == "DJF":
            yr = int(p[1])
            if yr >= y7[0]:
                oni[yr] = float(p[3])
    # Major stratospheric eruptions with a global climate signal. Aerosols
    # dim sunlight and cool for ~2 years; Hunga Tonga 2022 is the oddball —
    # mostly water vapour, a slight WARMING agent.
    volcanoes = [{"y": 1963, "n": "Agung"}, {"y": 1982, "n": "El Chichón"},
                 {"y": 1991, "n": "Pinatubo"}, {"y": 2022, "n": "Hunga Tonga"}]

    # Effective radiative forcing, annual, AR6 methodology extended yearly by
    # the Indicators of Global Climate Change project (Forster et al.) - the
    # "push" to plot against the measured imbalance. natural = solar +
    # volcanic; the volcanic convention is relative to the long-term mean
    # stratospheric load, so quiet years read slightly positive and big
    # eruptions dive to -2..-3 W/m^2.
    erf_url = ("https://raw.githubusercontent.com/ClimateIndicator/"
               "forcing-timeseries/main/output/ERF_best_aggregates_1750-2024.csv")
    req = urllib.request.Request(erf_url, headers=UA)
    with urllib.request.urlopen(req, timeout=120) as r:
        lines = r.read().decode().splitlines()
    hdr = lines[0].split(",")
    ia, iso, ivo = hdr.index("anthro"), hdr.index("solar"), hdr.index("volcanic")
    erf_years, erf_anthro, erf_natural = [], [], []
    for ln in lines[1:]:
        p = ln.split(",")
        yr = int(float(p[0]))
        if yr < y7[0]:
            continue                                     # chart starts with the OHC record
        erf_years.append(yr)
        erf_anthro.append(round(float(p[ia]), 2))
        erf_natural.append(round(float(p[iso]) + float(p[ivo]), 2))

    payload = {
        "source": "NOAA NCEI Global Ocean Heat Content (Levitus et al.), yearly, world",
        "doc": "https://www.ncei.noaa.gov/products/ocean-heat-salt-sea-level",
        "units": "1e22 J anomaly; rates in W/m^2 of total Earth surface",
        "snapshot": datetime.now(timezone.utc).strftime("%Y-%m-%d"),
        "y700": y7, "ohc700": v7, "err700": e7,
        "y2000": y2, "ohc2000": v2, "err2000": e2,
        "rate700": rolling_rate(y7, v7),
        "rate2000": rolling_rate(y2, v2),
        "rate10": round(r10, 3),                          # ocean 0-2000, last 10 yr
        "eei10": round(r10 / 0.9, 3),                     # implied total (ocean ~90%)
        "zj_since": y2[0],
        "zj_gained": round((v2[-1] - v2[0]) * 10, 1),     # 1e22 J -> ZJ
        "oni": oni,                                       # year -> DJF ONI (degC)
        "volcanoes": volcanoes,
        "erf_years": erf_years,                           # annual ERF (AR6/IGCC)
        "erf_anthro": erf_anthro,                         # total human forcing, W/m^2
        "erf_natural": erf_natural,                       # solar + volcanic, W/m^2
        "erf_src": "Forster et al. / ClimateIndicator ERF_best_aggregates (AR6 method)",
    }
    with open(os.path.join(DATA, "eei.json"), "w") as f:
        json.dump(payload, f, separators=(",", ":"))
    print(f"  wrote eei.json: 0-700m {y7[0]}-{y7[-1]}, 0-2000m {y2[0]}-{y2[-1]}, "
          f"last-10yr ocean rate {r10:.2f} W/m^2 -> EEI ~{r10/0.9:.2f} W/m^2, "
          f"+{payload['zj_gained']} ZJ since {y2[0]}")


RG_BASE = "https://sio-argo.ucsd.edu/pub/www-argo/RG"

# Depth levels the app's ocean column keeps (dbar). Chosen to resolve the
# mixed layer and thermocline tightly and thin out toward the abyss — the
# ocean's action is top-heavy.
COLUMN_LEVELS = [2.5, 10, 20, 30, 50, 75, 100, 150, 200, 300,
                 400, 500, 700, 900, 1200, 1500, 1975]


def _rg_latest_month():
    """Find the newest RG monthly-extension file by scraping the index page."""
    import re
    req = urllib.request.Request("https://sio-argo.ucsd.edu/RG_Climatology.html", headers=UA)
    with urllib.request.urlopen(req, timeout=60) as r:
        html = r.read().decode("utf-8", "replace")
    months = sorted(set(re.findall(r"RG_ArgoClim_(\d{6})_2019\.nc\.gz", html)))
    return months[-1]


def _gunzip(path):
    import gzip
    import shutil
    out = path[:-3]
    if not os.path.exists(out):
        with gzip.open(path, "rb") as fin, open(out, "wb") as fout:
            shutil.copyfileobj(fin, fout)
    return out


def argo_column():
    """Ocean column state from the Roemmich-Gilson Argo climatology (Scripps,
    open, no account): the latest month's ABSOLUTE T/S profile plus the same
    calendar month's 2004-2018 normal, on a 2 deg grid at COLUMN_LEVELS. Both
    lines come from ONE product, so their difference is a clean seasonally-
    correct anomaly (no cross-baseline mixing). Also bakes a 1 deg map of the
    temperature anomaly at ~300 dbar (data/argo_t300.json) - the subsurface
    marine-heatwave layer that no surface product can show."""
    import numpy as np
    import netCDF4 as ncdf

    ym = _rg_latest_month()                        # e.g. "202606"
    month_label = f"{ym[:4]}-{ym[4:]}"
    print(f"argo_column: latest RG month {month_label}")
    tf = _gunzip(_download(f"{RG_BASE}/RG_ArgoClim_Temperature_2019.nc.gz",
                           "/tmp/nc/RG_T.nc.gz", " (RG temperature mean+anomalies)"))
    sf = _gunzip(_download(f"{RG_BASE}/RG_ArgoClim_Salinity_2019.nc.gz",
                           "/tmp/nc/RG_S.nc.gz", " (RG salinity mean+anomalies)"))
    ef = _gunzip(_download(f"{RG_BASE}/RG_ArgoClim_{ym}_2019.nc.gz",
                           f"/tmp/nc/RG_{ym}.nc.gz", " (latest monthly anomaly)"))

    dT, dS, dE = ncdf.Dataset(tf), ncdf.Dataset(sf), ncdf.Dataset(ef)
    press = np.array(dT.variables["PRESSURE"][:])
    lats = np.array(dT.variables["LATITUDE"][:])
    lons = np.array(dT.variables["LONGITUDE"][:])            # 20.5..379.5
    lidx = [int(np.argmin(np.abs(press - p))) for p in COLUMN_LEVELS]
    moy = int(ym[4:]) - 1                                     # 0-based month-of-year

    def field(dsMean, meanVar, dsExt, extVar):
        # Keep netCDF4's land/ice masks intact: np.array() on a masked array
        # silently bakes the fill values in — and since fill − fill = 0, land
        # would render as plausible-looking "zero anomaly" ocean. np.ma.filled
        # with NaN is the only safe conversion here.
        mean = np.ma.filled(dsMean.variables[meanVar][lidx], np.nan)      # (L,145,360)
        anom_var = dsMean.variables[meanVar.replace("MEAN", "ANOMALY")]
        # month-of-year normal anomaly over the 15 years of 2004-2018
        sel = [t for t in range(anom_var.shape[0]) if t % 12 == moy]
        norm_anom = np.ma.filled(np.ma.mean(anom_var[sel][:, lidx], axis=0), np.nan)
        ext = np.ma.filled(dsExt.variables[extVar][0, lidx], np.nan)
        return mean + ext, mean + norm_anom                   # (now, norm)

    t_now, t_norm = field(dT, "ARGO_TEMPERATURE_MEAN", dE, "ARGO_TEMPERATURE_ANOMALY")
    s_now, s_norm = field(dS, "ARGO_SALINITY_MEAN", dE, "ARGO_SALINITY_ANOMALY")

    # -- 2 deg column file: bin each level onto a regular grid
    LON, LAT = np.meshgrid(lons, lats)                        # (145,360)
    bounds, nx, ny = (-180, -65, 180, 79), 180, 72
    def pack(cube):
        out = []
        for k in range(len(lidx)):
            g, _, dlon, dlat = _bin_to_grid(LON, LAT, cube[k], *bounds, nx, ny)
            out.append([None if not np.isfinite(v) else int(round(v * 100)) for v in g])
        return out, dlon, dlat
    (tn, dlon, dlat), (tm, _, _) = pack(t_now), pack(t_norm)
    (sn, _, _), (sm, _, _) = pack(s_now), pack(s_norm)
    payload = {
        "month": month_label,
        "baseline": f"2004–2018 mean for the same calendar month (Argo era)",
        "source": "Roemmich-Gilson Argo Climatology (Scripps): monthly extension + 2004-2018 fields",
        "doc": "https://sio-argo.ucsd.edu/RG_Climatology.html",
        "citation": "Roemmich & Gilson 2009 (Prog. Oceanogr.), updated; scale: values x100 (T degC, S PSS)",
        "levels": COLUMN_LEVELS,
        "west": bounds[0], "south": bounds[1], "east": bounds[2], "north": bounds[3],
        "dlon": dlon, "dlat": dlat, "nx": nx, "ny": ny,
        "snapshot": datetime.now(timezone.utc).strftime("%Y-%m-%d"),
        "t_now": tn, "t_norm": tm, "s_now": sn, "s_norm": sm,
    }
    with open(os.path.join(DATA, "ocean_column.json"), "w") as f:
        json.dump(payload, f, separators=(",", ":"))
    filled = sum(1 for v in tn[0] if v is not None)
    print(f"  wrote ocean_column.json: {nx}x{ny}x{len(lidx)} levels, "
          f"{filled} surface cells, month {month_label}")

    # -- 1 deg subsurface anomaly map at ~300 dbar
    k300 = COLUMN_LEVELS.index(300)
    _write_grid(
        "argo-t300", "argo_t300.json", LON, LAT,
        t_now[k300] - t_norm[k300],
        (-180, -65, 180, 80), 360, 145,
        units="°C", ramp="anom", vmin=-2, vmax=2, decimals=2,
        title=f"Subsurface temperature anomaly (300 m, {month_label})",
        source="Roemmich-Gilson Argo Climatology (Scripps)",
        citation=f"T at ~300 dbar, {month_label} minus 2004-2018 same-month mean",
        doc="https://sio-argo.ucsd.edu/RG_Climatology.html")


def glorys(start_year=1993):
    """Full-archive GLORYS ocean bake - REQUIRES a free Copernicus Marine
    account (data.marine.copernicus.eu/register). Credentials come from a
    prior `copernicusmarine login` or the COPERNICUSMARINE_SERVICE_USERNAME /
    COPERNICUSMARINE_SERVICE_PASSWORD environment variables; they are never
    stored in this repo.

    Two phases, both the SAME model (GLORYS12), chosen to keep downloads sane:
      1. 1993 -> GREP end (2024-12 as of writing): the GLORYS member
         (uo_glor/vo_glor/mlotst_glor) of the 1/4-deg ensemble reanalysis
         GLOBAL_MULTIYEAR_PHY_ENS_001_031, fetched one YEAR per request -
         16x less data than 1/12-deg, and we bin to 1-deg anyway.
      2. after GREP: 1/12-deg GLORYS12 my/myint monthly means, one month per
         request (also feeds ocean_surface.json for the pixel card).

    Output is an INDEX + per-year files, so the app lazy-loads history:
      data/currents.json / data/mld.json - metadata, monthsAvailable (all
        baked YYYY-MM stamps), months (latest year inline), latest,
        values (= latest month, back-compat)
      data/currents_y/YYYY.json / data/mld_y/YYYY.json - {"year", "months"}
      data/ocean_surface.json - u,v,zos,mld packed, latest month only
    Resume-friendly: years already complete in data/*_y/ are not re-fetched;
    per-request NetCDFs are deleted right after baking (disk stays flat)."""
    import numpy as np
    try:
        import copernicusmarine as cm
    except ImportError:
        sys.exit("glorys: pip install copernicusmarine")
    import netCDF4 as ncdf

    bounds = (-180, -80, 180, 90)
    nx, ny = 360, 170
    now = datetime.now(timezone.utc)
    for sub in ("currents_y", "mld_y"):
        os.makedirs(os.path.join(DATA, sub), exist_ok=True)

    # ---- resume: collect months already baked into per-year files
    months_speed, months_mld = {}, {}
    for sub, target in (("currents_y", months_speed), ("mld_y", months_mld)):
        d = os.path.join(DATA, sub)
        for fn in sorted(os.listdir(d)):
            if fn.endswith(".json"):
                with open(os.path.join(d, fn)) as f:
                    target.update(json.load(f)["months"])

    def bake_slices(path, uvar, vvar, mldvar, want_zos=False):
        """Bin every time slice in a NetCDF into the month dicts.
        Returns (stamps, latest_full_fields_or_None)."""
        d = ncdf.Dataset(path)
        tvar = d.variables["time"]
        lat = np.array(d.variables["latitude"][:])
        lon = np.array(d.variables["longitude"][:])
        LON, LAT = np.meshgrid(lon, lat)
        stamps, keep = [], None
        for i in range(len(tvar)):
            stamp = str(ncdf.num2date(tvar[i], tvar.units))[:7]
            def v2(name):
                a = d.variables[name]
                # np.ma.filled, never np.array: netCDF int16 fills otherwise bake
                # in as ±32767×scale and hypot() makes 46,340 m/s "currents"
                return np.ma.filled((a[i, 0] if a.ndim == 4 else a[i]).astype(float), np.nan)
            u, v, mld = v2(uvar), v2(vvar), v2(mldvar)
            speed = np.hypot(u, v)
            gs, _, _, _ = _bin_to_grid(LON, LAT, speed, *bounds, nx, ny)
            gm2, _, _, _ = _bin_to_grid(LON, LAT, mld, *bounds, nx, ny)
            months_speed[stamp] = [None if not np.isfinite(x) else round(float(x), 2) for x in gs]
            months_mld[stamp] = [None if not np.isfinite(x) else int(round(x)) for x in gm2]
            stamps.append(stamp)
            if want_zos:
                keep = (LON, LAT, u, v, v2("zos"), mld)
            print(f"  baked {stamp} ({sum(1 for x in months_speed[stamp] if x is not None)} ocean cells)")
        d.close()
        return stamps, keep

    # ---- phase 1: GREP 1/4-deg GLORYS member, one year per request
    grep_ds = "cmems_mod_glo_phy-all_my_0.25deg_P1M-m"
    grep_last = None
    for y in range(start_year, now.year + 1):
        have = [f"{y}-{m:02d}" for m in range(1, 13) if f"{y}-{m:02d}" in months_speed]
        if len(have) == 12:
            grep_last = max(grep_last or "", have[-1])
            continue                                   # year already baked
        out = f"/tmp/nc/grep_{y}.nc"
        try:
            if not os.path.exists(out):
                print(f"  GREP {y} ...")
                cm.subset(dataset_id=grep_ds,
                          variables=["uo_glor", "vo_glor", "mlotst_glor"],
                          start_datetime=f"{y}-01-01", end_datetime=f"{y}-12-31",
                          minimum_depth=0, maximum_depth=1,
                          output_filename=out)
            stamps, _ = bake_slices(out, "uo_glor", "vo_glor", "mlotst_glor")
            grep_last = max(grep_last or "", max(stamps))
            os.remove(out)
        except Exception as e:
            if os.path.exists(out):
                os.remove(out)
            print(f"    GREP ends before {y}: {type(e).__name__}: {str(e)[:100]}")
            break
    if not grep_last and not months_speed:
        sys.exit("glorys: GREP fetch failed and nothing is baked - check login")

    # ---- phase 2: 1/12-deg GLORYS12 my/myint for months after GREP
    candidates = ["cmems_mod_glo_phy_myint_0.083deg_P1M-m",
                  "cmems_mod_glo_phy_my_0.083deg_P1M-m"]
    latest_fields = None
    y, m = (int(grep_last[:4]), int(grep_last[5:7])) if grep_last else (start_year, 0)
    while True:
        m += 1
        if m > 12:
            m, y = 1, y + 1
        # the archive lags ~2 months behind real time; don't ask past now-1
        # (a failed fetch at the live edge also ends the loop below)
        if y * 12 + m > now.year * 12 + now.month - 1:
            break
        stamp = f"{y}-{m:02d}"
        out = f"/tmp/nc/glorys_{stamp}.nc"
        if stamp in months_speed and not os.path.exists(out):
            continue                                   # already baked (resume)
        got = os.path.exists(out)
        if not got:
            for ds in candidates:
                try:
                    print(f"  trying {ds} @ {stamp}-01 ...")
                    cm.subset(dataset_id=ds,
                              variables=["uo", "vo", "zos", "mlotst"],
                              start_datetime=f"{stamp}-01", end_datetime=f"{stamp}-01",
                              minimum_depth=0, maximum_depth=1,
                              output_filename=out)
                    got = True
                    break
                except Exception as e:
                    if os.path.exists(out):
                        os.remove(out)
                    print(f"    no: {type(e).__name__}: {str(e)[:100]}")
        if not got:
            break                                      # reached the archive's live edge
        _, keep = bake_slices(out, "uo", "vo", "mlotst", want_zos=True)
        if keep:
            latest_fields = keep
        os.remove(out)

    if not months_speed:
        sys.exit("glorys: nothing baked")
    latest = max(months_speed)

    # ---- write per-year files + index
    def write_all(id, index_path, year_dir, months, *, units, ramp, vmin, vmax, title, citation):
        west, south, east, north = bounds
        stamps = sorted(months)
        years = sorted({s[:4] for s in stamps})
        for yr in years:
            with open(os.path.join(DATA, year_dir, f"{yr}.json"), "w") as f:
                json.dump({"year": int(yr),
                           "months": {s: months[s] for s in stamps if s[:4] == yr}},
                          f, separators=(",", ":"))
        latest_year = latest[:4]
        payload = {
            "id": id, "title": title, "units": units,
            "source": "Copernicus Marine GLORYS12 (1/12-deg + 1/4-deg GREP member)",
            "citation": citation,
            "doc": "https://data.marine.copernicus.eu/product/GLOBAL_MULTIYEAR_PHY_001_030/description",
            "ramp": ramp, "vmin": vmin, "vmax": vmax,
            "west": west, "south": south, "east": east, "north": north,
            "dlon": 1.0, "dlat": 1.0, "nx": nx, "ny": ny,
            "snapshot": now.strftime("%Y-%m-%d"),
            "latest": latest,
            "monthsAvailable": stamps,
            "yearDir": f"data/{year_dir}",
            "months": {s: months[s] for s in stamps if s[:4] == latest_year},
            "values": months[latest],
        }
        with open(os.path.join(DATA, index_path), "w") as f:
            json.dump(payload, f, separators=(",", ":"))
        print(f"  wrote {index_path}: {len(stamps)} months, {stamps[0]} -> {latest}, "
              f"{len(years)} year files in data/{year_dir}/")

    write_all("currents", "currents.json", "currents_y", months_speed,
              units="m/s", ramp="precip", vmin=0, vmax=1.5,
              title="Surface current speed (GLORYS monthly)",
              citation="monthly mean |u,v| at surface; 1993->2024 from the GLORYS "
                       "member of the 1/4-deg ensemble reanalysis, then 1/12-deg GLORYS12")
    write_all("mld", "mld.json", "mld_y", months_mld,
              units="m", ramp="precip", vmin=0, vmax=500,
              title="Mixed-layer depth (GLORYS monthly)",
              citation="monthly mean mlotst; 1993->2024 from the GLORYS member of "
                       "the 1/4-deg ensemble reanalysis, then 1/12-deg GLORYS12")

    # packed u/v (cm/s ints) for the pixel card's current arrow - latest month.
    # On a resume run where the latest month was already baked (no fresh 1/12-deg
    # NetCDF processed), the existing ocean_surface.json is already current: keep it.
    if latest_fields:
        LON, LAT, u, v, zos, mld = latest_fields
        gu, _, dlon, dlat = _bin_to_grid(LON, LAT, u, *bounds, nx, ny)
        gv, _, _, _ = _bin_to_grid(LON, LAT, v, *bounds, nx, ny)
        gz, _, _, _ = _bin_to_grid(LON, LAT, zos, *bounds, nx, ny)
        gm, _, _, _ = _bin_to_grid(LON, LAT, mld, *bounds, nx, ny)
        ints = lambda g, s: [None if not np.isfinite(x) else int(round(x * s)) for x in g]
        payload = {"month": latest, "west": -180, "south": -80, "east": 180, "north": 90,
                   "dlon": dlon, "dlat": dlat, "nx": nx, "ny": ny,
                   "source": "Copernicus Marine GLORYS12/interim monthly mean",
                   "scale": "u,v cm/s; zos cm; mld m",
                   "u": ints(gu, 100), "v": ints(gv, 100), "zos": ints(gz, 100), "mld": ints(gm, 1)}
        with open(os.path.join(DATA, "ocean_surface.json"), "w") as f:
            json.dump(payload, f, separators=(",", ":"))
        print(f"  wrote ocean_surface.json ({latest})")
    else:
        print("  ocean_surface.json unchanged (latest month already baked)")


def gfs(days=10):
    """GFS 10-day forecast bake from the NOMADS grib filter - NO account, NO
    key (NOAA public service; be polite, ~50 small subset requests).
    Finds the newest COMPLETE cycle (f240 published), then bakes:
      data/gfs_temp.json   - 2 m temperature, one frame per day (f000..f240),
                             day-keyed like the GLORYS grids but keyLen 10
      data/gfs_precip.json - 24-h precipitation totals summed from the 6-h
                             APCP buckets grouped by UTC day (only full days;
                             <0.5 mm/day bakes as null = transparent)
    The date selector picks the forecast day in the app; "init" records the
    model run so the layer is honest about its age."""
    import numpy as np
    try:
        import pygrib
    except ImportError:
        sys.exit("gfs: pip install pygrib")
    import urllib.request
    import urllib.parse
    import tempfile
    import time as _time

    base = "https://nomads.ncep.noaa.gov/cgi-bin/filter_gfs_0p25.pl"
    now = datetime.now(timezone.utc)

    def fetch(d, cyc, f, var, lev):
        q = {"dir": f"/gfs.{d}/{cyc}/atmos", "file": f"gfs.t{cyc}z.pgrb2.0p25.{f}",
             f"var_{var}": "on", f"lev_{lev}": "on"}
        url = base + "?" + urllib.parse.urlencode(q)
        for attempt in range(3):
            try:
                data = urllib.request.urlopen(url, timeout=180).read()
                return data if data[:4] == b"GRIB" else None
            except Exception as e:
                if "404" in str(e):
                    return None
                _time.sleep(5)
        return None

    def msgs(data):
        with tempfile.NamedTemporaryFile(suffix=".grib2", delete=False) as t:
            t.write(data)
            p = t.name
        try:
            return list(pygrib.open(p)), p
        finally:
            pass

    # newest cycle whose final lead exists = a complete forecast
    last_f = f"f{days * 24:03d}"
    cycle, probe = None, None
    for back_h in range(4, 54, 6):                     # ~4 h publication lag
        t0 = now - timedelta(hours=back_h)
        d, cyc = t0.strftime("%Y%m%d"), f"{(t0.hour // 6) * 6:02d}"
        data = fetch(d, cyc, last_f, "TMP", "2_m_above_ground")
        if data:
            cycle, probe = (d, cyc), data
            break
    if not cycle:
        sys.exit("gfs: no complete cycle found on NOMADS")
    d, cyc = cycle
    init = f"{d[:4]}-{d[4:6]}-{d[6:]}T{cyc}Z"
    print(f"  cycle {init} (complete to {last_f})")

    bounds = (-180, -90, 180, 90)
    nx, ny = 360, 180
    grids = {}                                         # cache the 1-deg binning coords

    def bin1(m):
        vals = np.ma.filled(m.values.astype(float), np.nan)
        if "LL" not in grids:
            lats, lons = m.latlons()
            grids["LL"] = (((lons + 180.0) % 360.0) - 180.0, lats)
        LON, LAT = grids["LL"]
        out, _, _, _ = _bin_to_grid(LON, LAT, vals, *bounds, nx, ny)
        return out

    # ---- 2 m temperature: one frame per day, same hour each day
    temp_frames = {}
    for k in range(0, days + 1):
        f = f"f{24 * k:03d}"
        data = probe if f == last_f else fetch(d, cyc, f, "TMP", "2_m_above_ground")
        if not data:
            print(f"  {f}: missing, skipped")
            continue
        ms, p = msgs(data)
        m = ms[0]
        stamp = m.validDate.strftime("%Y-%m-%d")
        g = bin1(m) - 273.15                           # K -> deg C
        temp_frames[stamp] = [None if not np.isfinite(x) else int(round(x)) for x in g]
        os.remove(p)
        print(f"  temp {f} -> {stamp}")
        _time.sleep(1)

    # ---- precipitation: 6-h buckets summed per UTC day (full days only)
    day_sum, day_cnt = {}, {}
    for lead in range(6, days * 24 + 1, 6):
        data = fetch(d, cyc, f"f{lead:03d}", "APCP", "surface")
        if not data:
            print(f"  f{lead:03d}: missing, skipped")
            continue
        ms, p = msgs(data)
        for m in ms:
            if m.endStep - m.startStep != 6:
                continue                               # skip 0-N running totals
            start = m.validDate - timedelta(hours=6)   # validDate = window end
            key = start.strftime("%Y-%m-%d")
            g = bin1(m)
            if key not in day_sum:
                day_sum[key] = np.zeros(nx * ny)
                day_cnt[key] = 0
            day_sum[key] += np.where(np.isfinite(g), g, 0.0)
            day_cnt[key] += 1
        os.remove(p)
        _time.sleep(1)
    precip_frames = {}
    for key in sorted(day_sum):
        if day_cnt[key] != 4:
            print(f"  precip {key}: partial ({day_cnt[key]}/4 buckets), dropped")
            continue
        # round first, THEN threshold: Python rounds 0.5 to 0 (banker's
        # rounding), which would bake visible-but-zero cells
        precip_frames[key] = [x if x >= 1 else None
                              for x in (int(round(v)) for v in day_sum[key])]
        print(f"  precip {key}: ok")

    def write(path, id, frames, *, units, ramp, vmin, vmax, title, citation):
        stamps = sorted(frames)
        payload = {
            "id": id, "title": title, "units": units,
            "source": "NOAA GFS 0.25-deg via NOMADS grib filter", "citation": citation,
            "doc": "https://www.emc.ncep.noaa.gov/emc/pages/numerical_forecast_systems/gfs.php",
            "ramp": ramp, "vmin": vmin, "vmax": vmax,
            "west": -180, "south": -90, "east": 180, "north": 90,
            "dlon": 1.0, "dlat": 1.0, "nx": nx, "ny": ny,
            "snapshot": now.strftime("%Y-%m-%d"),
            "init": init, "keyLen": 10,
            "latest": stamps[-1],
            "monthsAvailable": stamps,
            "months": {s: frames[s] for s in stamps},
            "values": frames[stamps[0]],
        }
        with open(os.path.join(DATA, path), "w") as f:
            json.dump(payload, f, separators=(",", ":"))
        print(f"  wrote {path}: {len(stamps)} days, {stamps[0]} -> {stamps[-1]}")

    if not temp_frames or not precip_frames:
        sys.exit("gfs: incomplete bake")
    write("gfs_temp.json", "gfs-temp", temp_frames,
          units="degC", ramp="sst", vmin=-30, vmax=40,
          title="Temperature forecast (GFS, 2 m)",
          citation=f"2 m temperature, GFS run {init}, daily frames to +{days} days")
    write("gfs_precip.json", "gfs-precip", precip_frames,
          units="mm/day", ramp="precip", vmin=0, vmax=50,
          title="Precipitation forecast (GFS, daily)",
          citation=f"24-h precipitation totals, GFS run {init}; <0.5 mm/day transparent")


DRIVER_CLASSES = [
    # code, label, hex — WRI/Google DeepMind's own palette, so the globe matches
    # every figure published about this dataset.
    (1, "Permanent agriculture", "#E39D29"),
    (2, "Hard commodities", "#E58074"),
    (3, "Shifting cultivation", "#E9D700"),
    (4, "Logging", "#51A44E"),
    (5, "Wildfire", "#895128"),
    (6, "Settlements & infrastructure", "#A354A0"),
    (7, "Other natural disturbances", "#3A209A"),
]


def drivers(deg=0.25):
    """WRI/Google DeepMind Global Drivers of Forest Loss, 1 km -> a categorical
    0.25-deg grid (data/drivers.json).

    The satellite alert layers (OPERA DIST) answer WHERE forest was lost; this
    answers WHY. It is a CLASSIFICATION, so the bake is a per-block MODE, never
    a mean: averaging "wildfire" and "logging" would produce "shifting
    cultivation", which is nonsense. Cells with no loss at all bake as null and
    render transparent, so the layer paints only the deforestation frontiers.

    0.25 deg (25x25 native cells per block) is a deliberate stop. The source is
    already a 1 km modal attribution, its published use is regional, and the
    driver story IS regional -- the arc of deforestation is agriculture, boreal
    Canada and Siberia are fire, the US southeast is logging. Per-clearing
    detail is the OPERA layer's job, at 30 m.

    CC BY 4.0. ~300 MB download, ~30 s of binning; needs rasterio."""
    import numpy as np
    try:
        import rasterio
    except ImportError:
        sys.exit("drivers: pip install rasterio")

    url = ("https://lcl.wridata.org/drivers_of_loss/1_km/raw/"
           "drivers_forest_loss_1km_2001_2025_v1_3.tif")
    tif = "/tmp/nc/drivers_forest_loss_1km_v1_3.tif"
    _download(url, tif, "drivers of forest loss (v1.3, 2001-2025)")

    src = rasterio.open(tif)
    # The product is a plate-carree uint8 raster, 0.01 deg, 255 = no loss/nodata,
    # band 1 = the class; bands 2-8 are per-class probabilities we don't need.
    # It stops at 84N/56S (beyond the treeline / open ocean), and the grid format
    # carries its own bounds, so keep them rather than padding to the poles.
    west, south = src.bounds.left, src.bounds.bottom
    east, north = src.bounds.right, src.bounds.top
    f = int(round(deg / abs(src.transform.a)))
    if src.width % f or src.height % f:
        sys.exit(f"drivers: {deg} deg does not divide the {src.width}x{src.height} raster")
    a = src.read(1)                                  # north-up, ~500 M cells
    ny, nx = src.height // f, src.width // f
    blocks = a.reshape(ny, f, nx, f).transpose(0, 2, 1, 3).reshape(ny, nx, f * f)
    # Count each class per block and take the winner. argmax breaks ties toward
    # the lower code; ties are rare and the alternative (null) would punch holes
    # in otherwise solid frontiers.
    counts = np.stack([(blocks == c).sum(axis=2) for c, _, _ in DRIVER_CLASSES], axis=2)
    tot = counts.sum(axis=2)
    best = np.where(tot > 0, counts.argmax(axis=2) + 1, 0).astype(np.uint8)
    best = best[::-1]                                # grid format is row 0 = south

    # Share of the mapped loss each driver dominates - quoted in the layer's
    # hover card, and a sanity check on the bake.
    share = {lab: int((best == c).sum()) for c, lab, _ in DRIVER_CLASSES}
    filled = sum(share.values())

    payload = {
        "id": "drivers", "title": "Drivers of forest loss (WRI/DeepMind, 1 km -> 0.25 deg)",
        "units": "driver", "source": "WRI / Google DeepMind, Global Drivers of Forest Loss v1.3",
        "citation": ("Sims, M.J. et al. 2025. Global drivers of forest loss at 1 km "
                     "resolution. Environmental Research Letters 20(7): 074027. "
                     "doi:10.1088/1748-9326/add606 (CC BY 4.0). Dominant driver of "
                     "2001-2025 tree-cover loss; 1 km classes binned to 0.25 deg by mode."),
        "doc": "https://datasets.wri.org/datasets/dominant-drivers-of-tree-cover-loss-at-1km",
        "classes": [{"code": c, "label": lab, "rgb": [int(h[1:3], 16), int(h[3:5], 16), int(h[5:7], 16)]}
                    for c, lab, h in DRIVER_CLASSES],
        "west": west, "south": south, "east": east, "north": north,
        "dlon": deg, "dlat": deg, "nx": nx, "ny": ny,
        "period": "2001-2025",
        "snapshot": datetime.now(timezone.utc).strftime("%Y-%m-%d"),
        "shares": share,
        # Packed rather than a `values` array: one char per cell, "." for empty.
        # A JSON array of 800k single digits and nulls is mostly punctuation
        # (3.5 MB vs 0.8 MB), and the pixel inspector fetches this on a click.
        # The client expands it once on arrival (`unpackGrid`).
        "packed": "".join("." if v == 0 else str(v) for v in best.ravel()),
    }
    with open(os.path.join(DATA, "drivers.json"), "w") as fh:
        json.dump(payload, fh, separators=(",", ":"))
    os.remove(tif)
    print(f"  wrote drivers.json: {nx}x{ny}, {filled} filled cells "
          f"({100 * filled / (nx * ny):.1f}%)")
    for c, lab, _ in DRIVER_CLASSES:
        print(f"    {lab:<30} {share[lab]:>7} cells  {100 * share[lab] / filled:5.1f}%")


def cities():
    """Natural Earth populated places -> data/cities.json: the map's reference
    points.

    A globe of pure data is beautiful and unnavigable. An SST anomaly off a
    coastline you cannot name tells you nothing about WHERE the ocean is warm,
    and the whole app is built on being able to ask "what is happening HERE".

    NASA GIBS serves reference overlays and we use its Reference_Features_15m
    for borders and coastlines -- but its Reference_LABELS layer returns blank
    PNGs (Worldview draws those names from a vector source Cesium would need an
    MVT decoder to read), so the names are baked here instead. That is the
    app's normal posture anyway: no new browser-facing host, one static file.

    Natural Earth is PUBLIC DOMAIN and, more usefully, already decluttered by
    cartographers: `min_zoom` is the web-map zoom at which each place should
    first appear, so the ladder from "eleven world cities on the whole globe"
    to "every town in the valley" is theirs, not a threshold I invented. The
    client turns it into a per-label camera distance."""
    url = ("https://raw.githubusercontent.com/nvkelso/natural-earth-vector/"
           "master/geojson/ne_10m_populated_places_simple.geojson")
    src = "/tmp/nc/ne_10m_populated_places_simple.geojson"
    _download(url, src, "Natural Earth populated places (10m)")

    with open(src) as fh:
        feats = json.load(fh)["features"]

    out = []
    for f in feats:
        p = f["properties"]
        lon, lat = f["geometry"]["coordinates"][:2]
        out.append({
            "n": p["name"],
            # ~11 m precision: this is a label anchor, not a survey mark, and
            # the digits past it cost more than the whole `cap` flag.
            "o": round(float(lon), 4),
            "a": round(float(lat), 4),
            "z": float(p["min_zoom"]),          # cartographers' declutter ladder
            # Natural Earth writes -99 for "not known", its nodata sentinel
            # across the whole vector suite. Passed through it would sort a town
            # below every other place and read as a negative population in any
            # client that shows it; unknown is 0 here.
            "p": max(0, int(p["pop_max"] or 0)),
            "c": p["adm0name"] or "",
            "cap": 1 if p["adm0cap"] == 1 else 0,
        })
    # Most-important first, so a client that ever wants to cut the tail can
    # simply truncate, and so the biggest cities win any tie in draw order.
    out.sort(key=lambda c: (c["z"], -c["p"]))

    payload = {
        "id": "cities",
        "title": "Populated places (Natural Earth 10m)",
        "source": "Natural Earth, ne_10m_populated_places_simple",
        "citation": ("Natural Earth (naturalearthdata.com), 1:10m populated places. "
                     "Public domain. `z` is Natural Earth's min_zoom: the web-map "
                     "zoom level at which the place should first be labelled."),
        "doc": "https://www.naturalearthdata.com/downloads/10m-cultural-vectors/10m-populated-places/",
        "snapshot": datetime.now(timezone.utc).strftime("%Y-%m-%d"),
        "count": len(out),
        "places": out,
    }
    with open(os.path.join(DATA, "cities.json"), "w") as fh:
        json.dump(payload, fh, separators=(",", ":"), ensure_ascii=False)
    os.remove(src)
    caps = sum(c["cap"] for c in out)
    globe = sum(1 for c in out if c["z"] <= 3)
    print(f"  wrote cities.json: {len(out)} places, {caps} national capitals, "
          f"{globe} visible at globe zoom")


def gazetteer(tier="cities5000"):
    """GeoNames populated places -> data/gazetteer.json: the searchable tail.

    Natural Earth (see cities() above) is a CARTOGRAPHIC SELECTION, not a
    gazetteer. It carries 7,342 places worldwide and 24 in all of Portugal --
    Covilha (24,828) is in, Peniche (15,662) is not -- because its job is to
    produce a legible map at every zoom, not to know every town. That is
    exactly right for labels and exactly wrong for a search box: a user looking
    at the water off Peniche and typing "Peniche" must not be told it does not
    exist.

    So this is a second, deeper file with a different job. GeoNames cities5000
    (every populated place over 5,000 people, CC BY 4.0) minus everything
    Natural Earth already has, ~54 k places. It is lazy-loaded -- nobody pays
    for it until they either open the search box or zoom past where Natural
    Earth runs out.

    THE RUNG PROBLEM. Every place on the globe needs a `z`: the rung of the
    declutter ladder at which it starts being drawn (the client turns it into a
    camera distance, PLACE_FAR0 / 2^z). Natural Earth ships its own `min_zoom`
    and we use it verbatim -- but GeoNames has no such field, and CLAUDE.md
    forbids hand-picking one, for good reason.

    The way out is that a zoom ladder has an arithmetic. One rung down halves
    the camera height, so it quarters the visible area, so it can carry ~4x as
    many places at the SAME on-screen density. Natural Earth's own ladder obeys
    this: its cumulative counts at z<=3,4,5,6,7 are 58, 238, 570, 2502, 6924 --
    a geometric mean of ~3.3x per rung. So we measure that factor from Natural
    Earth's file rather than assuming 4, and continue ITS curve: sort the tail
    by population, and give the place at rank i the rung at which a ladder
    growing by G per rung would have reached 7342 + i places:

        z(i) = z_NE_max + log_G((N_NE + i + 1) / N_NE)

    which puts the tail at z 9.00 -> 10.78 and Peniche at 10.19, i.e. visible
    from about 70 km up. Nothing here is a number I chose: the anchor is where
    Natural Earth stops, the slope is Natural Earth's own measured density
    growth, and the ordering is population.

    Deduplication is against cities.json and is deliberately two-sided: within
    0.08 deg (~9 km) regardless of name, because the same town is placed
    slightly differently by the two projects, AND same name within 0.5 deg,
    because for big cities the two point locations can be 20 km apart (Dubai)
    and a doubled label is worse than a missing one."""
    import gzip
    import unicodedata
    import zipfile

    def fold(s):
        """Diacritic-insensitive key. The client folds the same way, so typing
        'Zurich' finds 'Zurich' and 'Peniche' finds 'Peniche' whatever the
        keyboard did."""
        s = unicodedata.normalize("NFD", s)
        return "".join(c for c in s if unicodedata.category(c) != "Mn").lower()

    zpath = f"/tmp/nc/{tier}.zip"
    _download(f"https://download.geonames.org/export/dump/{tier}.zip", zpath,
              f" GeoNames {tier}")
    with zipfile.ZipFile(zpath) as zf:
        rows = [ln.split("\t") for ln in
                zf.read(f"{tier}.txt").decode("utf-8").splitlines() if ln]

    with open(os.path.join(DATA, "cities.json")) as fh:
        ne = json.load(fh)["places"]

    # Two indexes over Natural Earth, one per dedupe rule (see docstring).
    cell, byname = {}, {}
    for c in ne:
        cell.setdefault((round(c["a"] * 10), round(c["o"] * 10)), []).append(c)
        byname.setdefault(fold(c["n"]), []).append(c)

    def already_mapped(name, lat, lon):
        ky, kx = round(lat * 10), round(lon * 10)
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                for c in cell.get((ky + dy, kx + dx), ()):
                    if abs(c["a"] - lat) < 0.08 and abs(c["o"] - lon) < 0.08:
                        return True
        for c in byname.get(fold(name), ()):
            if abs(c["a"] - lat) < 0.5 and abs(c["o"] - lon) < 0.6:
                return True
        return False

    tail = []
    for r in rows:
        # GeoNames columns: 1 name, 4 lat, 5 lon, 8 country code, 14 population
        lat, lon = float(r[4]), float(r[5])
        if already_mapped(r[1], lat, lon):
            continue
        tail.append((int(r[14] or 0), r[1], round(lon, 4), round(lat, 4), r[8]))
    # Rank is the whole ladder: population descending, name as a stable
    # tiebreak so a re-bake of the same dump produces byte-identical rungs.
    tail.sort(key=lambda t: (-t[0], t[1]))

    cum = [sum(1 for p in ne if p["z"] <= k) for k in range(3, 8)]
    import math
    growth = math.exp(sum(math.log(cum[i + 1] / cum[i])
                          for i in range(len(cum) - 1)) / (len(cum) - 1))
    n_ne = len(ne)
    z_ne = max(p["z"] for p in ne)
    places = [{
        "n": name,
        "o": lon, "a": lat,
        "z": round(z_ne + math.log((n_ne + i + 1) / n_ne) / math.log(growth), 2),
        "p": pop,
        "c": cc,           # ISO-3166 alpha-2; names in `countries` below
    } for i, (pop, name, lon, lat, cc) in enumerate(tail)]

    # Country NAMES are a lookup rather than a field: "Portugal" repeated
    # 500 times is the single largest compressible thing in the file, and the
    # search list needs it spelled out ("Peniche, Portugal", not ", PT").
    cinfo = "/tmp/nc/countryInfo.txt"
    _download("https://download.geonames.org/export/dump/countryInfo.txt",
              cinfo, " GeoNames country names")
    names = {}
    with open(cinfo) as fh:
        for ln in fh:
            if ln.startswith("#"):
                continue
            f = ln.split("\t")
            if len(f) > 4:
                names[f[0]] = f[4]
    used = {p["c"] for p in places}

    payload = {
        "id": "gazetteer",
        "title": f"Populated places over 5,000 ({tier}, GeoNames)",
        "source": f"GeoNames {tier}, minus everything already in cities.json",
        "citation": ("GeoNames geographical database (geonames.org), "
                     "CC BY 4.0. Deduplicated against Natural Earth. `z` "
                     "continues Natural Earth's declutter ladder: the rung at "
                     "which a ladder growing by `growth` places per rung would "
                     "have reached this place's global population rank."),
        "doc": "https://www.geonames.org/export/",
        "license": "CC BY 4.0",
        "snapshot": datetime.now(timezone.utc).strftime("%Y-%m-%d"),
        "growth": round(growth, 3),
        "zFrom": z_ne,
        "count": len(places),
        "countries": {k: v for k, v in names.items() if k in used},
        "places": places,
    }
    out = os.path.join(DATA, "gazetteer.json")
    with open(out, "w") as fh:
        json.dump(payload, fh, separators=(",", ":"), ensure_ascii=False)
    raw = os.path.getsize(out)
    with open(out, "rb") as fh:
        gz = len(gzip.compress(fh.read()))
    print(f"  wrote gazetteer.json: {len(places)} places "
          f"(from {len(rows)}, {len(rows) - len(places)} already in cities.json), "
          f"rungs {places[0]['z']}-{places[-1]['z']} continuing Natural Earth's "
          f"{growth:.2f}x/rung, {raw / 1e6:.1f} MB raw / {gz / 1e6:.2f} MB gzipped")


if __name__ == "__main__":
    os.makedirs("/tmp/nc", exist_ok=True)
    default = ["climatetrace", "argo", "rapid", "sealevel", "glaciers", "gistemp"]
    which = sys.argv[1:] or default
    fns = {"climatetrace": climatetrace, "argo": argo, "rapid": rapid,
           "sealevel": sealevel, "glaciers": glaciers, "gistemp": gistemp,
           "gpcp": gpcp, "eobs": eobs, "oisst": oisst, "meteoswiss": meteoswiss,
           "species": species, "argo_column": argo_column, "glorys": glorys, "eei": eei,
           "gfs": gfs, "drivers": drivers, "cities": cities,
           "gazetteer": gazetteer}
    for w in which:
        fns[w]()
    print("done")

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


def glorys():
    """GLORYS/CMEMS ocean bake - REQUIRES a free Copernicus Marine account
    (data.marine.copernicus.eu/register). Credentials come from a prior
    `copernicusmarine login` or the COPERNICUSMARINE_SERVICE_USERNAME /
    COPERNICUSMARINE_SERVICE_PASSWORD environment variables; they are never
    stored in this repo. Downloads ~1-2 GB to /tmp/nc, writes small JSONs:
      data/currents.json   - surface current speed, 1 deg   (globe layer)
      data/mld.json        - mixed-layer depth, 1 deg       (globe layer)
      data/ocean_surface.json - u,v,zos,mld packed for the pixel card
    Dataset ids are tried in order (interim first for recency)."""
    import numpy as np
    try:
        import copernicusmarine as cm
    except ImportError:
        sys.exit("glorys: pip install copernicusmarine")

    candidates = ["cmems_mod_glo_phy_myint_0.083deg_P1M-m",
                  "cmems_mod_glo_phy_my_0.083deg_P1M-m"]
    now = datetime.now(timezone.utc)
    out = "/tmp/nc/glorys_2d.nc"
    got = None
    if not os.path.exists(out):
        for ds in candidates:
            for back in range(2, 9):                       # try recent months
                y = now.year
                m = now.month - back
                while m < 1:
                    m += 12
                    y -= 1
                stamp = f"{y}-{m:02d}-01"
                try:
                    print(f"  trying {ds} @ {stamp} ...")
                    if os.path.exists(out):
                        os.remove(out)
                    cm.subset(dataset_id=ds,
                              variables=["uo", "vo", "zos", "mlotst"],
                              start_datetime=stamp, end_datetime=stamp,
                              minimum_depth=0, maximum_depth=1,
                              output_filename=out)
                    got = (ds, stamp)
                    break
                except Exception as e:
                    print(f"    no: {type(e).__name__}: {str(e)[:100]}")
            if got:
                break
        if not got:
            sys.exit("glorys: no dataset/month combination worked - check login and ids")
        print(f"  got {got[0]} @ {got[1]}")
    import netCDF4 as ncdf
    d = ncdf.Dataset(out)
    # month stamp from the file itself, so a cached re-run stays labelled
    tvar = d.variables["time"]
    stamp = str(ncdf.num2date(tvar[0], tvar.units))[:7]
    lat = np.array(d.variables["latitude"][:])
    lon = np.array(d.variables["longitude"][:])
    LON, LAT = np.meshgrid(lon, lat)
    def v2(name):
        a = d.variables[name]
        # np.ma.filled, never np.array: netCDF int16 fills otherwise bake in
        # as ±32767×scale and hypot() turns them into 46,340 m/s "currents"
        return np.ma.filled((a[0, 0] if a.ndim == 4 else a[0]).astype(float), np.nan)
    u, v, zos, mld = v2("uo"), v2("vo"), v2("zos"), v2("mlotst")
    speed = np.hypot(u, v)
    _write_grid("currents", "currents.json", LON, LAT, speed,
                (-180, -80, 180, 90), 360, 170,
                units="m/s", ramp="precip", vmin=0, vmax=1.5, decimals=2,
                title=f"Surface current speed (GLORYS, {stamp})",
                source="Copernicus Marine GLORYS12 / interim", citation="monthly mean |u,v| at surface",
                doc="https://data.marine.copernicus.eu/product/GLOBAL_MULTIYEAR_PHY_001_030/description")
    _write_grid("mld", "mld.json", LON, LAT, mld,
                (-180, -80, 180, 90), 360, 170,
                units="m", ramp="precip", vmin=0, vmax=500, decimals=0,
                title=f"Mixed-layer depth (GLORYS, {stamp})",
                source="Copernicus Marine GLORYS12 / interim", citation="monthly mean mlotst",
                doc="https://data.marine.copernicus.eu/product/GLOBAL_MULTIYEAR_PHY_001_030/description")
    # packed u/v (cm/s ints) for the pixel card's current arrow
    gu, _, dlon, dlat = _bin_to_grid(LON, LAT, u, -180, -80, 180, 90, 360, 170)
    gv, _, _, _ = _bin_to_grid(LON, LAT, v, -180, -80, 180, 90, 360, 170)
    gz, _, _, _ = _bin_to_grid(LON, LAT, zos, -180, -80, 180, 90, 360, 170)
    gm, _, _, _ = _bin_to_grid(LON, LAT, mld, -180, -80, 180, 90, 360, 170)
    ints = lambda g, s: [None if not np.isfinite(x) else int(round(x * s)) for x in g]
    payload = {"month": stamp, "west": -180, "south": -80, "east": 180, "north": 90,
               "dlon": dlon, "dlat": dlat, "nx": 360, "ny": 170,
               "source": "Copernicus Marine GLORYS12/interim monthly mean",
               "scale": "u,v cm/s; zos cm; mld m",
               "u": ints(gu, 100), "v": ints(gv, 100), "zos": ints(gz, 100), "mld": ints(gm, 1)}
    with open(os.path.join(DATA, "ocean_surface.json"), "w") as f:
        json.dump(payload, f, separators=(",", ":"))
    print(f"  wrote ocean_surface.json ({stamp})")


if __name__ == "__main__":
    os.makedirs("/tmp/nc", exist_ok=True)
    default = ["climatetrace", "argo", "rapid", "sealevel", "glaciers", "gistemp"]
    which = sys.argv[1:] or default
    fns = {"climatetrace": climatetrace, "argo": argo, "rapid": rapid,
           "sealevel": sealevel, "glaciers": glaciers, "gistemp": gistemp,
           "gpcp": gpcp, "eobs": eobs, "oisst": oisst, "meteoswiss": meteoswiss,
           "species": species, "argo_column": argo_column, "glorys": glorys, "eei": eei}
    for w in which:
        fns[w]()
    print("done")

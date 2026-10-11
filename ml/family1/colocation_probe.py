#!/usr/bin/env python3
"""E-092 §5 step 2 — the co-location probe: is there ANY floating-debris signal
in Sentinel-2 under The Ocean Cleanup's 2021 measurements in the Great Pacific
Garbage Patch?

THE QUESTION. Sentinel-2 imaged the open North Pacific gyre exactly once, a
special campaign of 2021-07-02 → 2021-09-19 (plan §2b). The Ocean Cleanup flew
22 aerial surveys and towed manta nets inside that area and those weeks
(Lebreton et al. 2024). For each in-situ point with a Sentinel-2 acquisition
within ±1 UTC day covering it, this script compares the floating debris index
(FDI) in a 2 km disc around the point (the TARGET) with the same acquisition's
open water 10–30 km away in the same tile (the BACKGROUND). Expected: nothing
above noise — the densest patch holds ≲ 13 objects > 50 cm per km², of the
order of 0.001 % of the surface, against a 20 m pixel of 400 m². A clear signal
would be a finding; none closes the `s2fdi_gpgp` store (plan §4) with a
measured reason.

SOURCES (all keyless, all read from the network, all cached under --cache):
  * in situ: figshare article 25674120 (CC BY 4.0), file
    `Lebreton2024_concentration_data.xlsx`, resolved BY NAME through the
    figshare API and checked against its published md5; the column meanings
    are asserted against the item's own description by the family-1.3
    adapter's checker (`mplastic.lebreton2024_check_description`), and the
    sheets are read by the adapter's own reader (`lebreton2024_read`).
    Sheets "Aerial survey" (CD7 = objects > 500 mm per km², the aircraft's
    count as The Ocean Cleanup reprocessed it in 2024) and "Manta trawl"
    (CD1..CD6 per km² — the 2018-midpoint, wind-mixing-corrected values, see
    the adapter's measurement `lebreton2024_manta_is_midpoint`).
  * satellite catalogue: Earth Search STAC v1, collections sentinel-2-l1c and
    sentinel-2-l2a (the same granule, processing baseline 05.00).
  * L1C bands: JPEG 2000 from the `sentinel-s2-l1c` bucket over anonymous
    HTTPS (the bucket is documented as requester-pays; from this sandbox's
    proxy GET and Range both answer 200/206 — re-check elsewhere).
  * L2A scene classification (SCL, 20 m) from the `sentinel-cogs` COGs, read
    by windowed range reads, used ONLY as the cloud / water mask.

METHOD, per co-located (point, acquisition) pair:
  * grid: the tile's 20 m grid. B08 (and B04, which FAI and NDVI need) are
    10 m and are reduced to 20 m — by ACOLITE's own resampling for the
    corrected reflectances, by a 2×2 mean for TOA; B06 and B11 are native 20 m.
  * reflectance, two versions, both reported:
      - "acolite": ACOLITE's dark spectrum fitting (DSF) on L1C,
        `dsf_aot_estimate=fixed` (one aerosol optical thickness per scene
        crop), `resolved_geometry=False` (scene-mean sun/view geometry),
        `ancillary_data=False` (default ozone / pressure), no glint
        correction. ρ_s = surface reflectance. THE PRIMARY RESULT.
      - "toa": top-of-atmosphere reflectance from L1C itself,
        (DN + RADIO_ADD_OFFSET) / QUANTIFICATION_VALUE, both read from the
        product metadata (baseline 05.00: offset −1000, value 10000).
        Labelled "TOA, not atmospherically corrected" wherever it appears.
  * indices (plan §4): FDI = R8 − [R6 + (R11 − R6)·(λ8 − λ4)/(λ11 − λ4)·10],
    factor 1.7722 (S2A) / 1.7770 (S2B); FAI = R8 − [R4 + (R11 − R4)·
    (λ8 − λ4)/(λ11 − λ4)]; NDVI = (R8 − R4)/(R8 + R4).
  * mask: SCL == 6 (water) — the primary; a looser variant keeps every pixel
    not flagged no-data, saturated, cloud shadow, cloud or cirrus
    (SCL ∉ {0, 1, 3, 8, 9, 10}) in case Sen2Cor files bright debris under
    "not vegetated" or "unclassified".
  * target: pixel centres ≤ 2 km from the point; background: 10–30 km, same
    tile, same acquisition. Statistics: n, median, 99th, 99.9th percentile;
    the fraction of target pixels above the BACKGROUND's 99.9th percentile
    (≈ 0.1 % if nothing is there); and a within-scene null — the same
    exceedance for 200 random 2 km discs centred 12–28 km away in the same
    scene, so the target is ranked against discs that share its clouds,
    glint and noise rather than against a nominal 0.1 %.
  * across pairs: a one-sample Wilcoxon signed-rank test of (target
    exceedance − the pair's null median) against zero, the number of pairs
    whose target ranks above its null's 95th percentile, and the Spearman
    correlation of the exceedance with the aerial count of objects > 50 cm
    per km² (and, for the net tows, with pieces per km²), each with its n.
  * the 10 most extreme target pixels pooled over all pairs (FDI excess over
    the pair's background median, in units of the background's p99.9 −
    median), each described: cluster size, NDVI, visible brightness (ships
    are bright in red and NIR alike), distance to the nearest cloud, and how
    often the background holds a pixel as extreme.

OUTPUT: `ml/family1/probes/colocation_gpgp_2021.json` (every per-pair number,
the method, the correction used, the versions) and
`ml/family1/probes/colocation_gpgp_2021.png` (the best pair: true colour, FDI
with the disc and annulus, a zoom on the disc, histograms).

Run:  python3 -I ml/family1/colocation_probe.py [--cache DIR] [--max-pairs 30]
"""
import argparse
import datetime as dt
import hashlib
import json
import math
import os
import platform
import re
import subprocess
import sys
import time

import numpy as np
import requests

HERE = os.path.dirname(os.path.abspath(__file__))
ML = os.path.dirname(HERE)
REPO = os.path.dirname(ML)
sys.path.insert(0, ML)
from family1.adapters import mplastic as mp  # noqa: E402

STAC = "https://earth-search.aws.element84.com/v1"
L1C_HTTP = "https://sentinel-s2-l1c.s3.eu-central-1.amazonaws.com/"
ACOLITE_GIT = "https://github.com/acolite/acolite.git"
DEFAULT_CACHE = ("/tmp/claude-0/-home-claude-earth/"
                 "e3369861-dd35-55bf-addc-816d2b37b228/scratchpad/coloc")
OUT_JSON = os.path.join(HERE, "probes", "colocation_gpgp_2021.json")
OUT_PNG = os.path.join(HERE, "probes", "colocation_gpgp_2021.png")

R_TARGET = 2000.0          # m, the target disc
R_IN, R_OUT = 10000.0, 30000.0   # m, the background annulus
NULL_DISCS = 200           # random discs for the within-scene null
NULL_R = (12000.0, 28000.0)  # where the null discs' centres may sit
MIN_TARGET_WATER_PX = 5000  # SCL-water px the disc must hold (of ~31,400):
#                             at p99.9 that is ≥ 5 expected exceedances
MIN_BG_CLEAR_PX = 100_000  # background water pixels (of ~6.3 M)
DAY_WINDOW = 1             # ± UTC days
PX = 20.0                  # analysis grid, m
SCL_WATER = 6
SCL_BAD_LOOSE = (0, 1, 3, 8, 9, 10)
SCL_CLOUDISH = (3, 8, 9, 10)
CLOUD_BUFFER_M = 300.0     # the buffered-mask variant drops water this close
#                            to a Sen2Cor cloud / shadow / cirrus pixel
SCL_NAMES = {0: "no data", 1: "saturated/defective", 2: "dark area",
             3: "cloud shadow", 4: "vegetation", 5: "not vegetated",
             6: "water", 7: "unclassified", 8: "cloud medium probability",
             9: "cloud high probability", 10: "thin cirrus", 11: "snow/ice"}
# central wavelengths, nm (ESA S2 spectral response functions, as in the plan)
WAVE = {"sentinel-2a": {"B04": 664.6, "B06": 740.5, "B08": 832.8,
                        "B11": 1613.7},
        "sentinel-2b": {"B04": 665.0, "B06": 739.1, "B08": 833.0,
                        "B11": 1610.4}}
FDI_FACTOR = {"sentinel-2a": 1.7722, "sentinel-2b": 1.7770}
ACO_BANDS = {"B02": "492", "B03": "560", "B04": "665", "B06": "740",
             "B08": "833", "B11": "1614"}
# the ACOLITE wavelength label differs by a nanometre between S2A and S2B
ACO_ALT = {"492": ("492",), "560": ("560", "559"), "665": ("665",),
           "740": ("740", "739"), "833": ("833",), "1614": ("1614", "1610")}

BYTES = {"downloaded": 0}
SESSION = requests.Session()


# ================================================================ helpers ==
def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def get_json(url, cache_path, method="GET", body=None):
    if os.path.exists(cache_path):
        with open(cache_path) as f:
            return json.load(f)
    for attempt in range(5):
        try:
            if method == "POST":
                r = SESSION.post(url, json=body, timeout=120)
            else:
                r = SESSION.get(url, timeout=120)
            r.raise_for_status()
            break
        except requests.RequestException as e:
            if attempt == 4:
                raise
            log("retry", url, e)
            time.sleep(3 * (attempt + 1))
    BYTES["downloaded"] += len(r.content)
    d = r.json()
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    tmp = cache_path + ".part"
    with open(tmp, "w") as f:
        json.dump(d, f)
    os.replace(tmp, cache_path)
    return d


def download(url, dst, expect_size=None):
    """Stream `url` to `dst` (atomically). Refuses a truncated file."""
    if os.path.exists(dst) and (expect_size is None or
                                os.path.getsize(dst) == expect_size):
        return dst
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    for attempt in range(5):
        try:
            with SESSION.get(url, stream=True, timeout=300) as r:
                r.raise_for_status()
                n = 0
                with open(dst + ".part", "wb") as f:
                    for c in r.iter_content(1 << 20):
                        f.write(c)
                        n += len(c)
                cl = r.headers.get("Content-Length")
                if cl is not None and int(cl) != n:
                    raise IOError(f"{url}: {n} bytes of {cl}")
            break
        except (requests.RequestException, IOError) as e:
            if attempt == 4:
                raise
            log("retry", url, e)
            time.sleep(5 * (attempt + 1))
    BYTES["downloaded"] += n
    os.replace(dst + ".part", dst)
    return dst


def h8(*parts):
    return hashlib.sha1(json.dumps(parts, sort_keys=True).encode()).hexdigest()[:10]


def pct(a, q):
    return float(np.percentile(a, q)) if a.size else None


def fnum(x, nd=6):
    if x is None:
        return None
    x = float(x)
    if not math.isfinite(x):
        return None
    return float(f"{x:.{nd}g}")


# ================================================================ in situ ==
def insitu(cache):
    d = os.path.join(cache, "insitu")
    art = get_json(mp.FIG2024_ARTICLE, os.path.join(d, "article.json"))
    mp.lebreton2024_check_description(art)
    files = get_json(mp.FIG2024_FILES, os.path.join(d, "files.json"))
    f = [x for x in files if x["name"] == mp.FIG2024_NAME]
    if len(f) != 1:
        raise SystemExit(f"figshare {mp.FIG2024} has no single "
                         f"{mp.FIG2024_NAME}")
    f = f[0]
    path = download(f["download_url"], os.path.join(d, mp.FIG2024_NAME),
                    f.get("size"))
    md5 = hashlib.md5(open(path, "rb").read()).hexdigest()
    if f.get("computed_md5") and md5 != f["computed_md5"]:
        raise SystemExit(f"{path}: md5 {md5} != figshare's {f['computed_md5']}")
    book = mp.lebreton2024_read(path)
    pts = []
    for sheet, kind in (("Aerial survey", "aerial"), ("Manta trawl", "manta")):
        for i, r in enumerate(book[sheet]):
            if int(r["Year"]) != 2021:
                continue
            p = dict(id=f"{kind}:{sheet} row {i + 2}", kind=kind,
                     expedition=mp.l2024_expedition(r["Expedition"]),
                     lat=float(r["Lat"]), lon=float(r["Lon"]),
                     date=dt.date(int(r["Year"]), int(r["Month"]),
                                  int(r["Day"])).isoformat())
            if kind == "aerial":
                p["objects_gt50cm_per_km2"] = mp.lb.num(r.get("CD7"))
                p["mass_gt50cm_g_per_km2_modelled"] = mp.lb.num(r.get("WD7"))
            else:
                cds = [mp.lb.num(r.get(f"CD{k}")) for k in range(1, 7)]
                p["pieces_per_km2_wc"] = (None if any(c is None for c in cds)
                                          else sum(cds))
                p["pieces_5_50cm_per_km2_wc"] = (
                    None if cds[4] is None or cds[5] is None
                    else cds[4] + cds[5])
            pts.append(p)
    meta = dict(article=mp.FIG2024_ARTICLE, file=mp.FIG2024_NAME,
                download_url=f["download_url"], md5=md5,
                licence=(art.get("license") or {}).get("name"),
                doi=art.get("doi"))
    return pts, meta


# ============================================================ catalogue ==
def stac_items(cache, collection, bbox, d0, d1):
    body = {"collections": [collection], "bbox": [round(v, 4) for v in bbox],
            "datetime": f"{d0}T00:00:00Z/{d1}T23:59:59Z", "limit": 200}
    p = os.path.join(cache, "stac", f"{collection}_{h8(body)}.json")
    r = get_json(f"{STAC}/search", p, "POST", body)
    if r.get("numberMatched", 0) > len(r["features"]):
        raise SystemExit(f"STAC paging not implemented ({r['numberMatched']})")
    return r["features"]


def stac_item(cache, collection, item_id):
    p = os.path.join(cache, "stac", f"{collection}_{item_id}.json")
    return get_json(f"{STAC}/collections/{collection}/items/{item_id}", p)


_TF = {}


def to_utm(epsg, lon, lat):
    from pyproj import Transformer
    if epsg not in _TF:
        _TF[epsg] = (Transformer.from_crs(4326, epsg, always_xy=True),
                     Transformer.from_crs(epsg, 4326, always_xy=True))
    return _TF[epsg][0].transform(lon, lat)


def to_ll(epsg, x, y):
    to_utm(epsg, 0, 0)
    return _TF[epsg][1].transform(x, y)


def scene_geom(l2a):
    a = l2a["assets"]["scl"]
    t = a["proj:transform"]
    if t[0] != PX or t[4] != -PX:
        raise SystemExit(f"{l2a['id']}: SCL is not on a 20 m grid ({t})")
    return dict(epsg=int(l2a["properties"]["proj:epsg"]), x0=t[2], y0=t[5],
                shape=tuple(a["proj:shape"]))


def window_for(g, x, y, pad=R_OUT + 2 * PX):
    """Row/col window (20 m grid) covering ±pad around (x, y), clipped."""
    c0 = int(math.floor((x - pad - g["x0"]) / PX))
    c1 = int(math.ceil((x + pad - g["x0"]) / PX))
    r0 = int(math.floor((g["y0"] - (y + pad)) / PX))
    r1 = int(math.ceil((g["y0"] - (y - pad)) / PX))
    H, W = g["shape"]
    return max(r0, 0), min(r1, H), max(c0, 0), min(c1, W)


def dist_grid(g, win, x, y):
    r0, r1, c0, c1 = win
    xs = g["x0"] + (np.arange(c0, c1) + 0.5) * PX
    ys = g["y0"] - (np.arange(r0, r1) + 0.5) * PX
    return np.hypot(xs[None, :] - x, ys[:, None] - y).astype(np.float32)


def read_scl(cache, l2a, win):
    r0, r1, c0, c1 = win
    p = os.path.join(cache, "scl", f"{l2a['id']}_{r0}_{r1}_{c0}_{c1}.npy")
    if os.path.exists(p):
        return np.load(p)
    import rasterio
    from rasterio.windows import Window
    with rasterio.Env(GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR",
                      CPL_VSIL_CURL_ALLOWED_EXTENSIONS=".tif",
                      GDAL_HTTP_MAX_RETRY="5", GDAL_HTTP_RETRY_DELAY="3"):
        with rasterio.open(l2a["assets"]["scl"]["href"]) as ds:
            a = ds.read(1, window=Window(c0, r0, c1 - c0, r1 - r0))
    # windowed COG reads: count the area read, not the file (approximate)
    BYTES["downloaded"] += a.size // 4
    os.makedirs(os.path.dirname(p), exist_ok=True)
    np.save(p + ".tmp.npy", a)
    os.replace(p + ".tmp.npy", p)
    return a


def datatake_key(pr):
    """One acquisition, whatever its processing: the two copies the
    catalogue holds of each 2021 granule (sequence 0 = baseline 03.01,
    sequence 1 = the 2023 reprocessing at 05.00) share it."""
    return re.sub(r"_N\d\d\.\d\d$", "", pr["s2:datatake_id"])


def enumerate_pairs(cache, pts):
    """Every (point, L1C granule) within ±1 UTC day whose footprint comes
    within the target disc of the point (a bbox query; the SCL decides how
    much of the disc is actually imaged). Processing baseline 05.00 only —
    the same acquisitions exist at 03.01 too and would double-count."""
    cands = []
    pad = 0.05
    for p in pts:
        d = dt.date.fromisoformat(p["date"])
        d0 = (d - dt.timedelta(days=DAY_WINDOW)).isoformat()
        d1 = (d + dt.timedelta(days=DAY_WINDOW)).isoformat()
        for it in stac_items(cache, "sentinel-2-l1c",
                             [p["lon"] - pad, p["lat"] - pad,
                              p["lon"] + pad, p["lat"] + pad], d0, d1):
            pr = it["properties"]
            if pr["s2:processing_baseline"] != "05.00":
                continue
            acq = dt.datetime.fromisoformat(pr["datetime"].replace("Z", "+00:00"))
            cands.append(dict(point=p["id"], l1c=it["id"],
                              datatake=datatake_key(pr),
                              platform=pr["platform"],
                              datetime=pr["datetime"],
                              offset_days=(acq.date() - d).days,
                              baseline=pr["s2:processing_baseline"],
                              cloud_cover_tile=pr.get("eo:cloud_cover")))
    return cands


def near_misses(cache, pts):
    """For every point: the distance (km) from the point to the nearest
    baseline-05.00 L1C data footprint acquired on the survey date −1, 0, +1
    (UTC). 0 = inside a footprint; None = no acquisition anywhere near."""
    import shapely
    from shapely.geometry import Point, shape
    from pyproj import Transformer
    tr = Transformer.from_crs(4326, 32606, always_xy=True)

    def stx(geom):
        return shapely.transform(
            geom, lambda xy: np.column_stack(tr.transform(xy[:, 0], xy[:, 1])))
    day = {}
    out = {}
    for p in pts:
        d = dt.date.fromisoformat(p["date"])
        rec = {}
        for off in range(-DAY_WINDOW, DAY_WINDOW + 1):
            dd = (d + dt.timedelta(days=off)).isoformat()
            if dd not in day:
                feats = stac_items(cache, "sentinel-2-l1c",
                                   [-150.0, 33.0, -141.0, 37.0], dd, dd)
                day[dd] = [stx(shape(f["geometry"])) for f in feats
                           if f["properties"]["s2:processing_baseline"] == "05.00"]
            q = stx(Point(p["lon"], p["lat"]))
            ds = [g.distance(q) / 1e3 for g in day[dd]]
            rec[f"{off:+d}"] = round(min(ds), 1) if ds else None
        out[p["id"]] = dict(date=p["date"], km_to_nearest_footprint=rec)
    return out


def clear_stats(scl, dist):
    tgt = dist <= R_TARGET
    bg = (dist >= R_IN) & (dist <= R_OUT)
    n_disc = int(round(math.pi * R_TARGET ** 2 / PX ** 2))
    out = {}
    for name, m in (("target", tgt), ("background", bg)):
        s = scl[m]
        hist = np.bincount(s, minlength=12)
        valid = int((s != 0).sum())
        out[name] = dict(
            pixels_in_tile=int(m.sum()), pixels_with_data=valid,
            water_px=int(hist[SCL_WATER]),
            clear_fraction=(hist[SCL_WATER] / valid) if valid else 0.0,
            scl_histogram={SCL_NAMES[k]: int(v) for k, v in enumerate(hist)
                           if v})
    out["target"]["disc_pixels_nominal"] = n_disc
    out["target"]["water_fraction_of_disc"] = out["target"]["water_px"] / n_disc
    return out


# ================================================================== L1C ==
CATALOGUE_QUIRKS = []


def l1c_paths(it):
    """The tile directory in the L1C bucket, from the item's own assets.
    MEASURED 2026-10-11: some Earth Search sentinel-2-l1c items point their
    band assets at the sentinel-s2-l2a bucket (same tiles/... path). The L1C
    bucket holds the tile under the identical path, and `fetch_safe` refuses
    unless that tile's own tileInfo names the item's L1C product."""
    href = it["assets"]["nir"]["href"]
    m = re.match(r"s3://sentinel-s2-(l1c|l2a)/(tiles/.+/)B08\.jp2$", href)
    if not m:
        raise SystemExit(f"{it['id']}: unexpected L1C href {href}")
    if m.group(1) != "l1c":
        CATALOGUE_QUIRKS.append(dict(item=it["id"], href=href,
                                     used=L1C_HTTP + m.group(2)))
    return m.group(2)


def fetch_safe(cache, it):
    """Rebuild a minimal L1C SAFE from the bucket: scene metadata, tile
    metadata, 13 bands and the detector footprints ACOLITE reads."""
    tile = l1c_paths(it)
    tinfo = get_json(L1C_HTTP + tile + "tileInfo.json",
                     os.path.join(cache, "l1c_meta", it["id"] + "_tileInfo.json"))
    prod = tinfo["productPath"].rstrip("/") + "/"
    name = tinfo["productName"]
    if name != it["properties"]["s2:product_uri"].replace(".SAFE", ""):
        raise SystemExit(f"{it['id']}: bucket product {name} != STAC "
                         f"{it['properties']['s2:product_uri']}")
    safe = os.path.join(cache, "safe", name + ".SAFE")
    mtd = os.path.join(safe, "MTD_MSIL1C.xml")
    download(L1C_HTTP + prod + "metadata.xml", mtd)
    pmd = open(mtd).read()
    files = re.findall(r"<IMAGE_FILE>([^<]+)</IMAGE_FILE>", pmd)
    gdir = files[0].split("/")[1]
    gpath = os.path.join(safe, "GRANULE", gdir)
    download(L1C_HTTP + tile + "metadata.xml", os.path.join(gpath, "MTD_TL.xml"))
    bands = {}
    for f in files:
        b = f.split("_")[-1]
        if b == "TCI":
            continue
        bands[b] = download(L1C_HTTP + tile + b + ".jp2",
                            os.path.join(safe, f + ".jp2"))
        download(L1C_HTTP + tile + f"qi/DETFOO_{b}.jp2",
                 os.path.join(gpath, "QI_DATA", f"MSK_DETFOO_{b}.jp2"))
    base = re.search(r"<PROCESSING_BASELINE>([^<]+)<", pmd).group(1)
    quant = float(re.search(r"<QUANTIFICATION_VALUE[^>]*>([^<]+)<", pmd).group(1))
    offs = {int(k): float(v) for k, v in
            re.findall(r'<RADIO_ADD_OFFSET band_id="(\d+)">([^<]+)<', pmd)}
    if float(base) >= 4.0 and not offs:
        raise SystemExit(f"{name}: baseline {base} but no RADIO_ADD_OFFSET")
    return dict(safe=safe, name=name, bands=bands, baseline=base,
                quantification=quant, offsets=offs, granule=gdir)


# band_id order in the L1C metadata
BAND_ID = {"B01": 0, "B02": 1, "B03": 2, "B04": 3, "B05": 4, "B06": 5,
           "B07": 6, "B08": 7, "B8A": 8, "B09": 9, "B10": 10, "B11": 11,
           "B12": 12}


def read_toa(safe, band, win20, res):
    """TOA reflectance of `band` over the 20 m window, at native `res`;
    DN 0 (no data) → NaN. Also returns the count of pixels at DN ≤ 1 above
    the offset floor (i.e. TOA ≤ 0)."""
    import rasterio
    from rasterio.windows import Window
    r0, r1, c0, c1 = win20
    k = 20 // res
    with rasterio.open(safe["bands"][band]) as ds:
        a = ds.read(1, window=Window(c0 * k, r0 * k, (c1 - c0) * k,
                                     (r1 - r0) * k)).astype(np.float32)
    nod = a == 0
    off = safe["offsets"].get(BAND_ID[band], 0.0)
    rho = (a + off) / safe["quantification"]
    rho[nod] = np.nan
    pinned = int(((a > 0) & (a + off <= 0)).sum())
    return rho, pinned


def mean2x2(a):
    H, W = a.shape
    return a[:H // 2 * 2, :W // 2 * 2].reshape(H // 2, 2, W // 2, 2).mean(axis=(1, 3))


def acolite_dir(cache):
    d = os.path.join(cache, "acolite_src", "acolite")
    if not os.path.isdir(d):
        subprocess.run(["git", "clone", "--depth", "1", ACOLITE_GIT, d],
                       check=True)
    sha = subprocess.run(["git", "-C", d, "rev-parse", "HEAD"],
                         capture_output=True, text=True, check=True).stdout.strip()
    return d, sha


def run_acolite(cache, safe, it, g, win):
    """ACOLITE DSF over the lat/lon box enclosing the 20 m window `win`.
    Returns the L2R path and the [x_off, y_off, w, h] sub-window ACOLITE
    reports on the same 20 m tile grid."""
    r0, r1, c0, c1 = win
    xs = [g["x0"] + c0 * PX, g["x0"] + c1 * PX]
    ys = [g["y0"] - r0 * PX, g["y0"] - r1 * PX]
    lon, lat = zip(*[to_ll(g["epsg"], x, y) for x in xs for y in ys])
    lim = [round(min(lat) - 0.01, 4), round(min(lon) - 0.01, 4),
           round(max(lat) + 0.01, 4), round(max(lon) + 0.01, 4)]
    out = os.path.join(cache, "acolite", f"{it['id']}_{h8(lim)}")
    done = [f for f in (os.listdir(out) if os.path.isdir(out) else [])
            if f.endswith("_L2R.nc")]
    if not done:
        os.makedirs(out, exist_ok=True)
        aco, _ = acolite_dir(cache)
        st = os.path.join(out, "settings.txt")
        with open(st, "w") as f:
            f.write("\n".join([
                f"inputfile={safe['safe']}", f"output={out}",
                "limit=" + ",".join(str(v) for v in lim),
                "s2_target_res=20",
                "atmospheric_correction_method=dark_spectrum",
                "dsf_aot_estimate=fixed", "resolved_geometry=False",
                "ancillary_data=False", "l2w_parameters=None",
                "rgb_rhot=False", "rgb_rhos=False", "map_l2w=False",
                "l1r_export_geotiff=False", "l2r_export_geotiff=False",
                "l2r_include_bands=" + ",".join(
                    f"rhos_{w}" for w in ("492", "560", "559", "665", "740",
                                          "739", "833", "1614", "1610")),
                ""]))
        with open(os.path.join(out, "acolite.log"), "w") as lf:
            rc = subprocess.run([sys.executable, "launch_acolite.py", "--cli",
                                 f"--settings={st}"], cwd=aco, stdout=lf,
                                stderr=subprocess.STDOUT).returncode
        done = [f for f in os.listdir(out) if f.endswith("_L2R.nc")]
        if rc != 0 or not done:
            raise SystemExit(f"ACOLITE failed for {it['id']} (rc {rc}); "
                             f"see {out}/acolite.log")
        for f in os.listdir(out):   # the L1R file is an intermediate
            if f.endswith("_L1R.nc"):
                os.remove(os.path.join(out, f))
    return os.path.join(out, done[0]), lim


def read_acolite(path, win):
    import netCDF4
    d = netCDF4.Dataset(path)
    sub = [int(v) for v in d.getncattr("sub")]   # x_off, y_off, w, h
    r0, r1, c0, c1 = win
    out = {}
    names = set(d.variables)
    for b, w in ACO_BANDS.items():
        v = next((f"rhos_{a}" for a in ACO_ALT[w] if f"rhos_{a}" in names),
                 None)
        if v is None:
            raise SystemExit(f"{path}: no rhos for {b}")
        a = d[v][:].filled(np.nan).astype(np.float32)
        full = np.full((r1 - r0, c1 - c0), np.nan, np.float32)
        # ACOLITE's array covers rows sub[1]..sub[1]+h, cols sub[0]..sub[0]+w
        rr0, cc0 = sub[1], sub[0]
        rr1, cc1 = rr0 + a.shape[0], cc0 + a.shape[1]
        ir0, ir1 = max(r0, rr0), min(r1, rr1)
        ic0, ic1 = max(c0, cc0), min(c1, cc1)
        if ir1 > ir0 and ic1 > ic0:
            full[ir0 - r0:ir1 - r0, ic0 - c0:ic1 - c0] = \
                a[ir0 - rr0:ir1 - rr0, ic0 - cc0:ic1 - cc0]
        out[b] = full
    attrs = {k: d.getncattr(k) for k in ("ac_aot_550", "ac_model", "sza",
                                          "vza", "raa", "acolite_version")
             if k in d.ncattrs()}
    rhot833 = None
    if "rhot_833" in names:
        a = d["rhot_833"][:].filled(np.nan).astype(np.float32)
        rhot833 = (a, sub)
    d.close()
    return out, {k: (v.item() if hasattr(v, "item") else str(v))
                 for k, v in attrs.items()}, rhot833


# =============================================================== indices ==
def indices(R, plat):
    w = WAVE[plat]
    lin = (w["B08"] - w["B04"]) / (w["B11"] - w["B04"])
    fac = FDI_FACTOR[plat]
    if abs(lin * 10 - fac) > 2e-4:
        raise SystemExit(f"FDI factor {fac} does not match the wavelengths "
                         f"({lin * 10:.4f})")
    with np.errstate(invalid="ignore", divide="ignore"):
        fdi = R["B08"] - (R["B06"] + (R["B11"] - R["B06"]) * fac)
        fai = R["B08"] - (R["B04"] + (R["B11"] - R["B04"]) * lin)
        ndvi = (R["B08"] - R["B04"]) / (R["B08"] + R["B04"])
    return dict(FDI=fdi, FAI=fai, NDVI=ndvi)


def dist_stats(a):
    return dict(n=int(a.size), median=fnum(pct(a, 50)), p99=fnum(pct(a, 99)),
                p999=fnum(pct(a, 99.9)), max=fnum(a.max() if a.size else None))


def null_discs(rng, g, win, x, y, dist, ok, vals, thr):
    """Exceedance of `thr` for random 2 km discs centred 12–28 km away."""
    r0, r1, c0, c1 = win
    H, W = r1 - r0, c1 - c0
    xs = g["x0"] + (np.arange(c0, c1) + 0.5) * PX
    ys = g["y0"] - (np.arange(r0, r1) + 0.5) * PX
    rr = int(math.ceil(R_TARGET / PX))
    out = []
    tries = 0
    while len(out) < NULL_DISCS and tries < NULL_DISCS * 20:
        tries += 1
        rad = math.sqrt(rng.uniform(NULL_R[0] ** 2, NULL_R[1] ** 2))
        th = rng.uniform(0, 2 * math.pi)
        cx, cy = x + rad * math.cos(th), y + rad * math.sin(th)
        ci = int((cx - g["x0"]) / PX) - c0
        ri = int((g["y0"] - cy) / PX) - r0
        if not (rr <= ri < H - rr and rr <= ci < W - rr):
            continue
        sl = (slice(ri - rr, ri + rr + 1), slice(ci - rr, ci + rr + 1))
        dd = np.hypot(xs[None, sl[1]] - cx, ys[sl[0], None] - cy) <= R_TARGET
        m = dd & ok[sl]
        if m.sum() < MIN_TARGET_WATER_PX:
            continue
        vv = vals[sl][m]
        out.append((float((vv > thr).mean()), float(np.median(vv))))
    return np.array(out).reshape(-1, 2)


_CDT = {}


def cloud_distance(scl):
    """Distance (m) from each pixel to the nearest Sen2Cor cloud, cloud
    shadow or cirrus pixel (cached per array)."""
    from scipy import ndimage
    k = id(scl)
    if k not in _CDT or _CDT[k][0] is not scl:
        cl = np.isin(scl, SCL_CLOUDISH)
        d = (ndimage.distance_transform_edt(~cl) * PX if cl.any()
             else np.full(scl.shape, np.inf, np.float32))
        _CDT.clear()
        _CDT[k] = (scl, d.astype(np.float32))
    return _CDT[k][1]


def components(mask):
    from scipy import ndimage
    lab, n = ndimage.label(mask, structure=np.ones((3, 3)))
    return lab, n


# ================================================================ main ==
def analyse_pair(rng, pair, pt, g, win, dist, scl, R, plat, label):
    """All statistics for one pair and one reflectance version."""
    ix = indices(R, plat)
    finite = np.all([np.isfinite(R[b]) for b in ("B04", "B06", "B08", "B11")],
                    axis=0)
    masks = {"scl_water": (scl == SCL_WATER) & finite,
             "scl_not_cloud": (~np.isin(scl, SCL_BAD_LOOSE)) & finite,
             "scl_water_cloud_buffer": (scl == SCL_WATER) & finite
             & (cloud_distance(scl) > CLOUD_BUFFER_M)}
    tgt = dist <= R_TARGET
    bg = (dist >= R_IN) & (dist <= R_OUT)
    res = {}
    for mname, ok in masks.items():
        mr = {}
        for iname, v in ix.items():
            t = v[tgt & ok]
            b = v[bg & ok]
            st = dict(target=dist_stats(t), background=dist_stats(b))
            if t.size and b.size:
                thr = np.percentile(b, 99.9)
                st["target_exceed_bg_p999"] = fnum(float((t > thr).mean()))
                st["target_exceed_bg_p99"] = fnum(float(
                    (t > np.percentile(b, 99)).mean()))
                if iname == "FDI":
                    nl2 = null_discs(rng, g, win, pair["x"], pair["y"], dist,
                                     ok, v, thr)
                    nl, nmed = nl2[:, 0], nl2[:, 1]
                    tmed = float(np.median(t))
                    st["null_median_shift"] = dict(
                        note="the target disc's median FDI ranked among the "
                             "null discs' medians (a bulk shift, not a tail)",
                        null_median_of_medians=fnum(pct(nmed, 50)),
                        null_p5=fnum(pct(nmed, 5)), null_p95=fnum(pct(nmed, 95)),
                        target_rank=fnum(float((nmed < tmed).mean()))
                        if nmed.size else None)
                    st["null"] = dict(
                        n_discs=int(nl.size), median=fnum(pct(nl, 50)),
                        p95=fnum(pct(nl, 95)), max=fnum(nl.max() if nl.size else None),
                        target_rank=fnum(float((nl < st["target_exceed_bg_p999"]).mean()
                                               + 0.5 * (nl == st["target_exceed_bg_p999"]).mean()))
                        if nl.size else None)
            mr[iname] = st
        res[mname] = mr
    return ix, masks, res


def pair_top(pd, n=10):
    """The n most extreme target pixels of ONE pair (primary correction, SCL
    water), each described. Ranked by the FDI excess over the background
    median in units of (background p99.9 − median), so pairs pool."""
    ix, ok, dist, thr, R = pd["ix"], pd["ok"], pd["dist"], pd["thr"], pd["R"]
    fdi = ix["FDI"]
    bgm = (dist >= R_IN) & (dist <= R_OUT) & ok
    tg = (dist <= R_TARGET) & ok
    med = float(np.median(fdi[bgm]))
    scale = thr - med
    bgz = np.sort((fdi[bgm] - med) / scale)
    idx = np.argwhere(tg)
    zz = (fdi[tg] - med) / scale
    order = np.argsort(zz)[::-1][:n]
    cdt = cloud_distance(pd["scl"])
    lab, _ = components((fdi > thr) & ok)
    red_p999 = float(np.percentile(R["B04"][bgm], 99.9))
    nir_p999 = float(np.percentile(R["B08"][bgm], 99.9))
    out = []
    for o in order:
        r, c = (int(v) for v in idx[o])
        zv = float(zz[o])
        csize = int((lab == lab[r, c]).sum()) if lab[r, c] else 0
        cdist = float(cdt[r, c]) if np.isfinite(cdt[r, c]) else None
        red_bright = bool(R["B04"][r, c] > red_p999)
        nir_bright = bool(R["B08"][r, c] > nir_p999)
        nd = float(ix["NDVI"][r, c])
        # open-ocean red ρs is ~0.005–0.02; 0.05 is already whitecap/cloud
        bright = float(R["B04"][r, c]) > 0.05
        parts = []
        parts.append("bright" if bright else "dim")
        parts.append("spectrally flat red≈NIR (|NDVI| < 0.15)" if abs(nd) < 0.15
                     else f"NIR > red (NDVI {nd:.2f})")
        parts.append("single pixel" if csize <= 1 else
                     f"{csize}-pixel cluster above the background p99.9")
        if cdist is not None:
            parts.append(f"{cdist:.0f} m from a Sen2Cor cloud/shadow/cirrus "
                         "pixel")
        if cdist is not None and cdist <= CLOUD_BUFFER_M and abs(nd) < 0.15:
            verdict = "cloud fringe / cloudlet Sen2Cor called water"
        elif cdist is not None and cdist <= CLOUD_BUFFER_M:
            verdict = "next to a cloud (adjacency)"
        elif bright and abs(nd) < 0.15 and csize <= 3:
            verdict = "compact bright flat target — ship or cloudlet"
        else:
            verdict = "unexplained"
        look = "; ".join(parts) + " → " + verdict
        x = pd["g"]["x0"] + (pd["win"][2] + c + 0.5) * PX
        y = pd["g"]["y0"] - (pd["win"][0] + r + 0.5) * PX
        lon, lat = to_ll(pd["g"]["epsg"], x, y)
        out.append(dict(
            pair=pd["pair_id"], lat=round(lat, 5), lon=round(lon, 5),
            dist_from_point_m=round(float(dist[r, c])),
            fdi=fnum(fdi[r, c]), fdi_excess_units=fnum(zv, 4),
            fdi_bg_median=fnum(med), fdi_bg_p999=fnum(thr),
            fai=fnum(ix["FAI"][r, c]), ndvi=fnum(ix["NDVI"][r, c], 4),
            rhos={b: fnum(R[b][r, c], 4) for b in ("B04", "B06", "B08", "B11")},
            toa_10m_max_of_2x2={b: fnum(v[r, c], 4)
                                for b, v in pd["toa10max"].items()},
            toa_20m={b: fnum(pd["Rtoa"][b][r, c], 4)
                     for b in ("B04", "B06", "B08", "B11")},
            red_above_bg_p999=red_bright, nir_above_bg_p999=nir_bright,
            cluster_px_above_bg_p999=csize, dist_to_cloud_m=cdist,
            scl=SCL_NAMES[int(pd["scl"][r, c])],
            bg_pixels_as_extreme=int(bgz.size - np.searchsorted(bgz, zv)),
            bg_pixels=int(bgz.size),
            target_pixels=int(tg.sum()),
            looks_like=look))
    return out


def max2x2(a):
    H, W = a.shape
    return a[:H // 2 * 2, :W // 2 * 2].reshape(H // 2, 2, W // 2, 2).max(axis=(1, 3))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--cache", default=DEFAULT_CACHE)
    ap.add_argument("--max-pairs", type=int, default=30)
    ap.add_argument("--max-gb", type=float, default=15.0)
    ap.add_argument("--seed", type=int, default=20211)
    ap.add_argument("--enumerate-only", action="store_true",
                    help="list the pairs and their clear fractions, then stop")
    args = ap.parse_args()
    cache = args.cache
    os.makedirs(cache, exist_ok=True)
    t_start = time.time()
    rng = np.random.default_rng(args.seed)

    pts, insitu_meta = insitu(cache)
    log(f"{len(pts)} in-situ points in 2021 "
        f"({sum(p['kind'] == 'aerial' for p in pts)} aerial)")
    P = {p["id"]: p for p in pts}

    # ---- 1. every candidate (point, L1C item) within ±1 UTC day
    raw = enumerate_pairs(cache, pts)
    log(f"{len(raw)} (point, tile-acquisition) candidates")
    misses = near_misses(cache, pts)
    # ---- 2. geometry + SCL clear fractions per candidate
    by_pt_dt = {}
    for c in raw:
        l2a_id = c["l1c"].replace("_L1C", "_L2A")
        try:
            l2a = stac_item(cache, "sentinel-2-l2a", l2a_id)
        except requests.HTTPError:
            c["skip"] = "no matching L2A item (no SCL mask)"
            continue
        if datatake_key(l2a["properties"]) != c["datatake"]:
            c["skip"] = "L2A datatake differs"
            continue
        g = scene_geom(l2a)
        p = P[c["point"]]
        x, y = to_utm(g["epsg"], p["lon"], p["lat"])
        win = window_for(g, x, y)
        if win[1] <= win[0] or win[3] <= win[2]:
            c["skip"] = "point outside the tile grid"
            continue
        scl = read_scl(cache, l2a, win)
        dist = dist_grid(g, win, x, y)
        cs = clear_stats(scl, dist)
        c.update(l2a=l2a_id, epsg=g["epsg"], x=x, y=y, win=list(win),
                 clear=cs, l2a_baseline=l2a["properties"]["s2:processing_baseline"])
        key = (c["point"], c["datatake"])
        prev = by_pt_dt.get(key)
        score = (cs["target"]["water_px"], cs["background"]["water_px"])
        if prev is None or score > prev[0]:
            by_pt_dt[key] = (score, c)
    pairs = [v[1] for v in by_pt_dt.values()]
    log(f"{len(pairs)} (point, acquisition) pairs")

    def eligible(c):
        return (c["clear"]["target"]["water_px"] >= MIN_TARGET_WATER_PX
                and c["clear"]["background"]["water_px"] >= MIN_BG_CLEAR_PX)
    elig = sorted([c for c in pairs if eligible(c)],
                  key=lambda c: (abs(c["offset_days"]),
                                 -c["clear"]["target"]["water_fraction_of_disc"],
                                 -c["clear"]["background"]["water_px"]))
    sel = elig[:args.max_pairs]
    if args.enumerate_only:
        for c in sorted(pairs, key=lambda c: (c["point"], c["datetime"])):
            t, b = c["clear"]["target"], c["clear"]["background"]
            print(f"{c['point']:32s} {c['l1c']:26s} {c['offset_days']:+d} "
                  f"disc water {t['water_fraction_of_disc']:.2f} "
                  f"clear {t['clear_fraction']:.2f}  bg water {b['water_px']:>8d} "
                  f"clear {b['clear_fraction']:.2f} {'SEL' if c in sel else ''}")
    log(f"{len(elig)} eligible (≥ {MIN_TARGET_WATER_PX} disc water px, "
        f"≥ {MIN_BG_CLEAR_PX} background water px); taking {len(sel)}")

    if args.enumerate_only:
        return
    # ---- 3. per scene: L1C, ACOLITE over the union of the pairs' windows
    aco_dir, aco_sha = acolite_dir(cache)
    best_c = sorted(sel, key=lambda c: (
        abs(c["offset_days"]), P[c["point"]]["kind"] != "aerial",
        -c["clear"]["target"]["water_px"],
        -c["clear"]["background"]["water_px"]))[0] if sel else None
    best = None
    tops = []
    scenes = {}
    for c in sel:
        scenes.setdefault(c["l1c"], []).append(c)
    results = []
    pinned_tot = {}
    xcheck = []
    for sid, cs_ in scenes.items():
        if BYTES["downloaded"] > args.max_gb * 1e9:
            log("download budget reached — stopping before", sid)
            for c in cs_:
                c["skip"] = "download budget"
            continue
        it = stac_item(cache, "sentinel-2-l1c", sid)
        l2a = stac_item(cache, "sentinel-2-l2a", cs_[0]["l2a"])
        g = scene_geom(l2a)
        log(f"scene {sid}: {len(cs_)} pair(s); fetching L1C")
        safe = fetch_safe(cache, it)
        uni = [min(c["win"][0] for c in cs_), max(c["win"][1] for c in cs_),
               min(c["win"][2] for c in cs_), max(c["win"][3] for c in cs_)]
        log(f"  ACOLITE over rows {uni[0]}:{uni[1]} cols {uni[2]}:{uni[3]}")
        l2r, lim = run_acolite(cache, safe, it, g, uni)
        plat = it["properties"]["platform"]
        for c in cs_:
            win = tuple(c["win"])
            p = P[c["point"]]
            dist = dist_grid(g, win, c["x"], c["y"])
            scl = read_scl(cache, l2a, win)
            Raco, aco_attrs, rhot833 = read_acolite(l2r, win)
            # TOA at 20 m from L1C itself
            Rtoa = {}
            pinned = {}
            toa10max = {}
            for b in ("B02", "B03", "B04", "B08"):
                a10, pn = read_toa(safe, b, win, 10)
                pinned[b] = pn
                Rtoa[b] = mean2x2(a10)
                if b in ("B04", "B08"):
                    toa10max[b] = max2x2(a10)[:scl.shape[0], :scl.shape[1]]
                del a10
            for b in ("B06", "B11"):
                Rtoa[b], pinned[b] = read_toa(safe, b, win, 20)
            for b in Rtoa:
                Rtoa[b] = Rtoa[b][:scl.shape[0], :scl.shape[1]]
            for b, v in pinned.items():
                pinned_tot[b] = pinned_tot.get(b, 0) + v
            # cross-check our TOA B08 2×2 mean against ACOLITE's own rhot_833
            if rhot833 is not None:
                a, sub = rhot833
                r0, r1, c0, c1 = win
                rr0, cc0 = sub[1], sub[0]
                ir0, ir1 = max(r0, rr0), min(r1, rr0 + a.shape[0])
                ic0, ic1 = max(c0, cc0), min(c1, cc0 + a.shape[1])
                ours = Rtoa["B08"][ir0 - r0:ir1 - r0, ic0 - c0:ic1 - c0]
                theirs = a[ir0 - rr0:ir1 - rr0, ic0 - cc0:ic1 - cc0]
                m = np.isfinite(ours) & np.isfinite(theirs)
                if m.any():
                    dlt = np.abs(ours[m] - theirs[m])
                    xcheck.append(dict(pair=c["point"], scene=sid,
                                       median_abs_diff=fnum(np.median(dlt)),
                                       p99_abs_diff=fnum(np.percentile(dlt, 99)),
                                       median_value=fnum(np.median(theirs[m]))))
            pid = f"{c['point']} @ {sid}"
            rec = dict(pair_id=pid, point=p, scene=sid, l2a=c["l2a"],
                       platform=plat, datetime=c["datetime"],
                       offset_days=c["offset_days"], baseline=safe["baseline"],
                       radio_add_offset=safe["offsets"].get(BAND_ID["B08"]),
                       quantification=safe["quantification"],
                       clear=c["clear"], window_rows_cols=c["win"],
                       acolite_run=dict(limit=lim, **aco_attrs),
                       pinned_toa_le_0={b: int(v) for b, v in pinned.items()})
            for label, R in (("acolite", Raco), ("toa", Rtoa)):
                ix, masks, st = analyse_pair(rng, c, p, g, win, dist, scl, R,
                                             plat, label)
                aneg = {}
                tg = dist <= R_TARGET
                bg = (dist >= R_IN) & (dist <= R_OUT)
                ok = masks["scl_water"]
                for b in ("B04", "B06", "B08", "B11"):
                    aneg[b] = dict(
                        target=fnum(float((R[b][tg & ok] <= 0).mean())
                                    if (tg & ok).any() else None),
                        background=fnum(float((R[b][bg & ok] <= 0).mean())
                                        if (bg & ok).any() else None))
                rec[label] = dict(stats=st, fraction_le_0=aneg)
                if label == "acolite":
                    thr = float(np.percentile(ix["FDI"][bg & ok], 99.9))
                    pd = dict(pair_id=pid, ix=ix, ok=ok, dist=dist, scl=scl,
                              R=R, toa10max=toa10max, thr=thr, g=g, win=win,
                              rec=rec, Rtoa=Rtoa)
                    tops.extend(pair_top(pd))
                    if c is best_c:
                        best = pd
                    del pd
                del ix, masks
            results.append(rec)
            e = rec["acolite"]["stats"]["scl_water"]["FDI"]
            log(f"  {c['point']}: offset {c['offset_days']:+d} d, target water "
                f"{c['clear']['target']['water_px']} px, exceed "
                f"{e.get('target_exceed_bg_p999')} (null median "
                f"{e['null']['median']}, rank {e['null']['target_rank']})")

    # ---- 4. across pairs
    from scipy import stats as sst

    def cross(label, mask):
        ex, nmed, rank, cd7, mk, sc, mrank = [], [], [], [], [], [], []
        for r in results:
            e = r[label]["stats"][mask]["FDI"]
            if e.get("target_exceed_bg_p999") is None:
                continue
            ex.append(e["target_exceed_bg_p999"])
            nmed.append(e["null"]["median"])
            rank.append(e["null"]["target_rank"])
            mrank.append(e["null_median_shift"]["target_rank"])
            sc.append(r["scene"])
            p = r["point"]
            if p["kind"] == "aerial":
                cd7.append((e["target_exceed_bg_p999"], p["objects_gt50cm_per_km2"],
                            e["target"]["p999"] - e["background"]["p999"]))
            elif p.get("pieces_per_km2_wc") is not None:
                mk.append((e["target_exceed_bg_p999"], p["pieces_per_km2_wc"]))
        ex, nmed, rank = map(np.array, (ex, nmed, rank))
        out = dict(n_pairs=int(ex.size), n_scenes=len(set(sc)))
        if ex.size == 0:
            return out
        diff = ex - nmed
        out.update(
            exceedance=dict(median=fnum(np.median(ex)), mean=fnum(ex.mean()),
                            min=fnum(ex.min()), max=fnum(ex.max())),
            null_median=dict(median=fnum(np.median(nmed)), min=fnum(nmed.min()),
                             max=fnum(nmed.max())),
            pairs_above_null_p95=int(sum(rk > 0.95 for rk in rank)),
            expected_above_null_p95=fnum(0.05 * ex.size, 3),
            target_rank_in_null=dict(median=fnum(np.median(rank)),
                                     mean=fnum(rank.mean())),
            bulk_median_rank_in_null=dict(
                ranks=[fnum(v, 3) for v in mrank],
                above_p95=int(sum(v > 0.95 for v in mrank)),
                below_p5=int(sum(v < 0.05 for v in mrank))))
        if np.any(diff != 0):
            w = sst.wilcoxon(diff, alternative="greater")
            out["wilcoxon_exceed_minus_null_median"] = dict(
                statistic=fnum(w.statistic), p_one_sided_greater=fnum(w.pvalue, 3),
                n=int((diff != 0).sum()))
        w2 = sst.wilcoxon(ex - 0.001, alternative="greater") if np.any(ex != 0.001) else None
        if w2 is not None:
            out["wilcoxon_exceed_minus_0.001"] = dict(
                statistic=fnum(w2.statistic), p_one_sided_greater=fnum(w2.pvalue, 3))
        # one value per scene (pairs in a scene share clouds and noise)
        per_scene = {}
        for s, d_ in zip(sc, diff):
            per_scene.setdefault(s, []).append(d_)
        sd = np.array([np.mean(v) for v in per_scene.values()])
        if np.any(sd != 0) and sd.size >= 3:
            w3 = sst.wilcoxon(sd, alternative="greater")
            out["wilcoxon_per_scene_mean"] = dict(
                n_scenes=int(sd.size), statistic=fnum(w3.statistic),
                p_one_sided_greater=fnum(w3.pvalue, 3))
        if len(cd7) >= 3:
            a = np.array([[u[0], u[1]] for u in cd7], float)
            rho = sst.spearmanr(a[:, 0], a[:, 1])
            b = np.array([[u[2], u[1]] for u in cd7], float)
            rho2 = sst.spearmanr(b[:, 0], b[:, 1])
            out["spearman_vs_aerial_objects_gt50cm"] = dict(
                n=int(a.shape[0]), rho=fnum(rho.statistic, 3),
                p_two_sided=fnum(rho.pvalue, 3),
                note="exceedance fraction vs CD7 (objects > 50 cm per km²)")
            out["spearman_p999_excess_vs_aerial"] = dict(
                n=int(b.shape[0]), rho=fnum(rho2.statistic, 3),
                p_two_sided=fnum(rho2.pvalue, 3),
                note="target p99.9 − background p99.9 vs CD7")
        if len(mk) >= 3:
            a = np.array(mk, float)
            rho = sst.spearmanr(a[:, 0], a[:, 1])
            out["spearman_vs_manta_pieces_per_km2"] = dict(
                n=int(a.shape[0]), rho=fnum(rho.statistic, 3),
                p_two_sided=fnum(rho.pvalue, 3))
        return out

    across = {lab: {m: cross(lab, m) for m in ("scl_water", "scl_not_cloud",
                                               "scl_water_cloud_buffer")}
              for lab in ("acolite", "toa")}
    tops = sorted(tops, key=lambda t: -t["fdi_excess_units"])[:10]

    # ---- 5. figure for the best pair
    if best is not None:
        figure(best, OUT_PNG)

    # ---- 6. record
    def ver(mod):
        try:
            return __import__(mod).__version__
        except Exception:
            return None
    from osgeo import gdal
    import rasterio
    git_sha = subprocess.run(["git", "-C", REPO, "rev-parse", "HEAD"],
                             capture_output=True, text=True).stdout.strip()
    doc = dict(
        experiment="E-092 §5 step 2 — the co-location probe",
        question=("Is there any floating-debris signal in Sentinel-2 under The "
                  "Ocean Cleanup's 2021 measurements in the Great Pacific "
                  "Garbage Patch?"),
        at=dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        script="ml/family1/colocation_probe.py", repo_git_sha=git_sha,
        versions=dict(python=platform.python_version(), numpy=np.__version__,
                      scipy=ver("scipy"), rasterio=rasterio.__version__,
                      gdal_rasterio=rasterio.__gdal_version__,
                      gdal_osgeo=gdal.__version__, pyproj=ver("pyproj"),
                      netCDF4=ver("netCDF4"),
                      acolite=dict(repo=ACOLITE_GIT, commit=aco_sha)),
        sources=dict(
            insitu=insitu_meta, stac=STAC, l1c=L1C_HTTP,
            l1c_access="anonymous HTTPS GET/Range from this sandbox (200/206); "
                       "the bucket is documented as requester-pays",
            scl="sentinel-cogs L2A COG, windowed range reads",
            licence_satellite="Contains modified Copernicus Sentinel data 2021"),
        method=dict(
            grid="the tile's 20 m grid; B08 and B04 reduced 10 m → 20 m "
                 "(ACOLITE's resampling for ρ_s; 2×2 mean for TOA); B06, B11 "
                 "native 20 m",
            corrections=dict(
                acolite="PRIMARY. ACOLITE dark spectrum fitting on L1C: "
                        "dsf_aot_estimate=fixed, resolved_geometry=False, "
                        "ancillary_data=False, no glint correction, "
                        "s2_target_res=20; ρ_s",
                toa="TOA, not atmospherically corrected: (DN + "
                    "RADIO_ADD_OFFSET)/QUANTIFICATION_VALUE from the L1C "
                    "product metadata"),
            fdi="R8 − [R6 + (R11 − R6)·(λ8−λ4)/(λ11−λ4)·10]; 1.7722 S2A, "
                "1.7770 S2B (asserted against the wavelengths)",
            fai="R8 − [R4 + (R11 − R4)·(λ8−λ4)/(λ11−λ4)]",
            ndvi="(R8 − R4)/(R8 + R4)",
            masks=dict(scl_water="SCL == 6 (primary)",
                       scl_not_cloud="SCL ∉ {0,1,3,8,9,10}",
                       scl_water_cloud_buffer=f"SCL == 6 and > {CLOUD_BUFFER_M:.0f}"
                       " m from any SCL 3/8/9/10 pixel"),
            target_m=R_TARGET, background_m=[R_IN, R_OUT],
            null=f"{NULL_DISCS} random 2 km discs centred {NULL_R[0]:.0f}–"
                 f"{NULL_R[1]:.0f} m from the point, same scene, same mask, "
                 f"≥ {MIN_TARGET_WATER_PX} usable px; exceedance of the same "
                 "background p99.9; target_rank = fraction of null discs "
                 "below the target (ties half)",
            day_window=f"±{DAY_WINDOW} UTC day (acquisition UTC date − "
                       "survey date as published)",
            selection=f"≥ {MIN_TARGET_WATER_PX} SCL-water px in the disc, ≥ "
                      f"{MIN_BG_CLEAR_PX} background water px; ordered by "
                      "|offset|, then disc water fraction; at most "
                      f"{args.max_pairs}; one tile per (point, datatake) — "
                      "the tile with the most water in disc and annulus",
            caveat_independence="pairs in one scene share its clouds, aerosol "
                                "and noise; several survey points lie within "
                                "a few km of each other, so their discs "
                                "overlap — per-scene statistics are given"),
        counts=dict(insitu_points=len(pts),
                    insitu_aerial=sum(p["kind"] == "aerial" for p in pts),
                    insitu_manta=sum(p["kind"] == "manta" for p in pts),
                    candidates_tile_acquisitions=len(raw),
                    pairs_point_acquisition=len(pairs),
                    pairs_by_offset={str(k): sum(c["offset_days"] == k for c in pairs)
                                     for k in (-1, 0, 1)},
                    eligible=len(elig),
                    eligible_by_offset={str(k): sum(c["offset_days"] == k for c in elig)
                                        for k in (-1, 0, 1)},
                    analysed=len(results), scenes=len(scenes),
                    points_without_any_acquisition=sorted(
                        set(P) - {c["point"] for c in raw}),
                    toa_pinned_le_0_total=pinned_tot),
        all_pairs_clear=[dict(point=c["point"], scene=c["l1c"],
                              offset_days=c["offset_days"],
                              target_water_fraction=fnum(
                                  c["clear"]["target"]["water_fraction_of_disc"], 4),
                              target_clear_fraction=fnum(
                                  c["clear"]["target"]["clear_fraction"], 4),
                              background_clear_fraction=fnum(
                                  c["clear"]["background"]["clear_fraction"], 4),
                              background_water_px=c["clear"]["background"]["water_px"],
                              eligible=eligible(c), selected=c in sel)
                         for c in sorted(pairs, key=lambda c: (c["point"], c["datetime"]))],
        km_to_nearest_footprint_by_day_offset=misses,
        catalogue_quirks=CATALOGUE_QUIRKS,
        toa_vs_acolite_rhot833=xcheck,
        pairs=results, across_pairs=across, top_target_pixels=tops,
        best_pair=best["pair_id"] if best else None,
        figure="ml/family1/probes/colocation_gpgp_2021.png" if best else None,
        bytes_downloaded_this_run=BYTES["downloaded"],
        bytes_in_cache=dict(
            (k, sum(os.path.getsize(os.path.join(dp, f))
                    for dp, _, fs in os.walk(os.path.join(cache, k)) for f in fs))
            for k in ("safe", "scl", "stac", "insitu", "l1c_meta")),
        bytes_acolite_luts=sum(
            os.path.getsize(os.path.join(dp, f)) for dp, _, fs in
            os.walk(os.path.join(aco_dir, "data", "LUT")) for f in fs),
        wall_s=round(time.time() - t_start))
    os.makedirs(os.path.dirname(OUT_JSON), exist_ok=True)
    tmp = OUT_JSON + ".part"
    with open(tmp, "w") as f:
        json.dump(doc, f, indent=1, ensure_ascii=False)
    os.replace(tmp, OUT_JSON)
    log("wrote", OUT_JSON)
    log(json.dumps(across["acolite"]["scl_water"], indent=1))


# ================================================================ figure ==
def figure(pd, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle
    rec = pd["rec"]
    p = rec["point"]
    T = pd["Rtoa"]
    fdi = pd["ix"]["FDI"]
    ok = pd["ok"]
    dist = pd["dist"]
    H, W = fdi.shape
    # the point, in 20 m pixel coordinates of the window (the distance
    # grid's minimum is the pixel holding it)
    rr, cc = np.unravel_index(np.argmin(dist), dist.shape)
    px, py = cc + 0.5, rr + 0.5
    ext = [-px * PX / 1e3, (W - px) * PX / 1e3, -(H - py) * PX / 1e3, py * PX / 1e3]
    bg = (dist >= R_IN) & (dist <= R_OUT) & ok
    tg = (dist <= R_TARGET) & ok
    thr = pd["thr"]
    lo = np.percentile(fdi[bg], 0.1)

    fig = plt.figure(figsize=(6.4, 21), dpi=120, layout="constrained")
    gs = fig.add_gridspec(4, 1, height_ratios=[1, 1, 1, 0.62])
    off = rec["offset_days"]
    title = (f"{p['kind']} survey {p['date']} at {p['lat']:.3f}N "
             f"{-p['lon']:.3f}W"
             + (f" — {p['objects_gt50cm_per_km2']:.1f} objects > 50 cm/km²"
                if p["kind"] == "aerial" else "")
             + f"\nSentinel-2 {rec['platform'][-2:].upper()} "
               f"{rec['datetime'][:16]}Z (survey date {off:+d} d), tile "
               f"{rec['scene'].split('_')[1]}")
    fig.suptitle(title, fontsize=10.5)

    ax = fig.add_subplot(gs[0])
    rgb = np.dstack([T["B04"], T["B03"], T["B02"]])
    rgb = np.clip((rgb - 0.02) / 0.18, 0, 1) ** 0.6
    rgb = np.nan_to_num(rgb)
    ax.imshow(rgb, extent=ext, interpolation="nearest")
    ax.set_title("True colour (L1C TOA, B4/B3/B2, 2×2 mean to 20 m)", fontsize=10)
    for rad, ls in ((R_TARGET, "-"), (R_IN, "--"), (R_OUT, "--")):
        ax.add_patch(Circle((0, 0), rad / 1e3, fill=False, ec="yellow",
                            ls=ls, lw=1.2))
    ax.set_xlabel("km east of the survey point", fontsize=9)
    ax.set_ylabel("km north", fontsize=9)

    ax = fig.add_subplot(gs[1])
    show = np.where(ok, fdi, np.nan)
    cmap = plt.get_cmap("magma").copy()
    cmap.set_bad("#9aa7b4")
    im = ax.imshow(show, extent=ext, cmap=cmap, vmin=lo, vmax=thr,
                   interpolation="nearest")
    for rad, ls in ((R_TARGET, "-"), (R_IN, "--"), (R_OUT, "--")):
        ax.add_patch(Circle((0, 0), rad / 1e3, fill=False, ec="cyan", ls=ls,
                            lw=1.2))
    ax.set_title("FDI (ACOLITE ρs, 20 m); grey = not SCL water; "
                 "colour top = background p99.9", fontsize=9.5)
    fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02)
    ax.set_xlabel("km east", fontsize=9)

    ax = fig.add_subplot(gs[2])
    k = int(R_TARGET * 1.6 / PX)
    rs, cs = slice(max(rr - k, 0), rr + k), slice(max(cc - k, 0), cc + k)
    sub = show[rs, cs]
    e2 = [(cs.start - px) * PX / 1e3, (cs.stop - px) * PX / 1e3,
          -(rs.stop - py) * PX / 1e3, -(rs.start - py) * PX / 1e3]
    im = ax.imshow(sub, extent=e2, cmap=cmap, vmin=lo, vmax=thr,
                   interpolation="nearest")
    ex = (fdi > thr) & ok
    yy, xx = np.nonzero(ex[rs, cs])
    ax.scatter((xx + cs.start + 0.5 - px) * PX / 1e3,
               -(yy + rs.start + 0.5 - py) * PX / 1e3, s=18, facecolors="none",
               edgecolors="lime", lw=0.9, label="> background p99.9")
    ax.add_patch(Circle((0, 0), R_TARGET / 1e3, fill=False, ec="cyan", lw=1.4))
    ax.legend(loc="lower right", fontsize=8)
    e = rec["acolite"]["stats"]["scl_water"]["FDI"]
    ax.set_title(f"Zoom: the 2 km target disc — {e['target']['n']} water px, "
                 f"{100 * e['target_exceed_bg_p999']:.2f} % above background "
                 f"p99.9", fontsize=9.5)
    fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02)
    ax.set_xlabel("km east", fontsize=9)

    ax = fig.add_subplot(gs[3])
    a_bg, a_tg = fdi[bg], fdi[tg]
    bins = np.linspace(np.percentile(a_bg, 0.01), max(np.percentile(a_bg, 99.99),
                                                       a_tg.max()), 120)
    ax.hist(a_bg, bins=bins, density=True, histtype="step", lw=1.6,
            color="#4477aa", label=f"background 10–30 km (n={a_bg.size:,})")
    ax.hist(a_tg, bins=bins, density=True, histtype="step", lw=1.6,
            color="#cc3311", label=f"target ≤ 2 km (n={a_tg.size:,})")
    ax.axvline(thr, color="k", ls=":", lw=1, label="background p99.9")
    ax.set_yscale("log")
    ax.set_xlabel("FDI (ACOLITE ρs)", fontsize=9)
    ax.set_ylabel("density (log)", fontsize=9)
    nl = e["null"]
    ax.set_title(f"Target vs background — exceedance {100 * e['target_exceed_bg_p999']:.2f} % "
                 f"(null discs: median {100 * nl['median']:.2f} %, p95 "
                 f"{100 * nl['p95']:.2f} %)", fontsize=9.5)
    ax.legend(fontsize=8)
    for a in fig.axes:
        a.tick_params(labelsize=8)
    fig.get_layout_engine().set(h_pad=0.12)
    fig.text(0.01, -0.004, "Contains modified Copernicus Sentinel data 2021 · "
             "in situ: The Ocean Cleanup, Lebreton et al. 2024 (CC BY 4.0) · "
             "E-092 co-location probe", fontsize=7, color="#555")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()

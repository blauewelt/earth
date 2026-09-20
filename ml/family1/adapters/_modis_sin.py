"""The MODIS SINUSOIDAL TILE machinery `pheno500` and `lai500` share
(family 1.0.tf, E-082 wave 6, the biosphere wave). Underscore-prefixed, so the
registry skips it (`_common.py`, `_modis_cmg.py`, `_nc3.py` are the
precedents).

PLAIN ENGLISH. Two of the biosphere wave's land fields are MODIS products on
the same 500-metre grid, from the same archive, found the same way: the leaf
calendar (MCD12Q2, one map a year) and leaf area with the light the canopy
absorbs (MOD15A2H and its Aqua and VIIRS twins, one map every eight days).
Unlike the 0.05-degree Climate Modeling Grid fields `_modis_cmg.py` serves,
these are not one global file: the planet is cut into 10-degree TILES on a
SINUSOIDAL projection, and each tile is its own file and its own store group.
What differs between the two products is which scientific datasets they read
out of a file and what the numbers mean; everything else — the grid, finding a
tile's granule, logging in, downloading it, checking the projection, filing
the frame under its five-day bin — is here, once.

THE GRID, FROM THE PRODUCER'S OWN PAGE (modis-land.gsfc.nasa.gov/GCTP.html and
MODLAND_grid.html, read 2026-09-20). The MODLAND Sinusoidal grid is the
sinusoidal (Sanson-Flamsteed) projection on a SPHERE of radius
6,371,007.181 m, central meridian 0, false easting and northing 0:

    x = R * lon * cos(lat)          lon, lat in radians
    y = R * lat

The world is 36 tiles across and 18 down, numbered from (0, 0) in the UPPER
LEFT and running right (h) and down (v), so the tile in the bottom right is
(35, 17) and 460 of the 648 are "non-fill" — they touch land or sea rather
than the empty corners outside the projected globe. One tile is
2 pi R / 36 = 1,111,950.5197 m square; a 500-metre product is 2,400 x 2,400
pixels in it, so a pixel is 463.3127 m. Row 0 of a tile is its NORTHERNMOST
row and column 0 its westernmost, which is the orientation the HDF-EOS grid
itself is stored in (`GridOrigin = HDFE_CENTER`, upper-left corner first).

THE GRID IS DECLARED AND THEN CHECKED IN EVERY FILE. Every HDF-EOS granule
carries a `StructMetadata.0` attribute naming its own `UpperLeftPointMtrs`,
`LowerRightMtrs`, `XDim`, `YDim` and projection parameters; `check_struct`
compares them with the tile the file's NAME claims and REFUSES on a
difference, so the constants above are a hypothesis every granule falsifies
rather than a transcription. The corners were independently checked against
NASA's own catalogue: the polygon CMR publishes for `h18v04` reads
(-0.0087 E, 50.0070 N) .. (15.5724 E, 49.9990 N) .. (13.0379 E, 39.8144 N),
and the formulas above give 0 E / 50 N, 15.554 E / 50 N and 13.054 E / 40 N —
the same corners to within the padding CMR adds to a granule's bounding ring.

NO TILE-CORNER TABLE IS STORED, and that is deliberate. `sharded.make_spec`
can write the lat/lon of every tile boundary when the adapter supplies an
inverse projection, and for a sinusoidal tile that inverse is not always
DEFINED: the rectangle of a tile near the grid's edge has corners with
|x| > R * pi * cos(lat), i.e. points off the projected globe, where the
inverse asks for a longitude beyond 180 degrees. Rather than store a clamped
number that looks like a coordinate, the grid dict carries the projection,
the affine rule and the inverse formula in words, exactly as the plain
geographic stores do.

FINDING THE FILES: CMR, ONE YEAR AT A TIME. A granule's URL carries the
PRODUCTION timestamp (`MOD15A2H.A2020185.h18v04.061.2020340145645.hdf`),
which is not derivable from the date, so a URL pattern cannot be guessed —
the contract's rule 4 in its strongest form. NASA's Common Metadata
Repository lists every granule with its https link and its declared size,
needs NO account, and pages 2,000 at a time behind the `CMR-Search-After`
header. Unlike the CMG products, these collections are far too large to list
whole — MOD15A2H publishes 13,192 granules in 2020 alone and about 340,000
over its record — so the listing is asked for ONE PRODUCT YEAR at a time with
a `readable_granule_name` pattern and cached, and an adapter lists only the
years its window touches. MEASURED 2026-09-20 from this sandbox, keyless:

  MCD12Q2  v061  LPCLOUD   315 granules a year, 2001..2025, 315 tiles
  MOD15A2H v061  LPCLOUD   13,192 granules in 2020: 46 eight-day composites
                           of 274..290 tiles, union 290; 46.0 GB a year by
                           CMR's own declared sizes
  MYD15A2H v061  LPCLOUD   the same calendar from Aqua, 2002-07-04 ->
  VNP15A2H v002  LPCLOUD   the same calendar from VIIRS, 2012-01-01 ->, and
                           its granules are HDF5 (.h5), not HDF4 (.hdf)

A NOTE ON DUPLICATES, and it is not hypothetical. MCD12Q2's 2025 listing holds
**626** granules for **315** tiles: 311 of the tiles were reprocessed in
September 2026 and the earlier production was not retired. The listing
therefore keys on (date, tile) and keeps the HIGHEST production timestamp,
counts the rest as `granules_superseded`, and records the fact in the index —
it never silently picks the first.

DOWNLOADING THEM: EARTHDATA LOGIN THROUGH A .netrc, IPv4 ONLY — the same
mechanism `_modis_cmg.py` documents, and its functions are imported rather
than copied.

READING THEM: pyhdf for the HDF4 (HDF-EOS2) products and h5py for the VIIRS
HDF5 one. rasterio's GDAL has NO HDF4 driver (measured in this sandbox: 155
drivers, HDF5 and netCDF among them, no HDF4), so `pyhdf` — whose manylinux
wheel bundles the HDF4 library — is a separate dependency, already in
`family1-build.yml`'s install step.
"""
import datetime as dt
import json
import math
import os
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request

import numpy as np

import build_family10_stores as f10b
from family1 import sharded as sh
from family1.adapters import _modis_cmg as mc

# --------------------------------------------------------------- the grid --
R_SPHERE = 6371007.181                 # MODLAND sphere radius, metres
H_TILES, V_TILES = 36, 18
TILE_M = 2.0 * math.pi * R_SPHERE / H_TILES        # 1,111,950.5197 m
X0_GLOBAL = -math.pi * R_SPHERE                    # -20,015,109.354 m (west)
Y0_GLOBAL = math.pi * R_SPHERE / 2.0               # +10,007,554.677 m (north)
PX_500 = 2400                          # pixels a side in a 500 m tile
PROJ4 = ("+proj=sinu +lon_0=0 +x_0=0 +y_0=0 "
         f"+R={R_SPHERE} +units=m +no_defs")
CMR = "https://cmr.earthdata.nasa.gov/search/granules.json"
PAGE = 2000
KM_EQ = 111.31949                      # km per degree of longitude, equator
BIN_KM = 27.83                         # the family's reference footprint

TILE = re.compile(r"^h(\d{2})v(\d{2})$")
GRANULE = re.compile(r"^(?P<prod>[A-Za-z0-9]+)\.A(?P<year>\d{4})(?P<doy>\d{3})"
                     r"\.h(?P<h>\d{2})v(?P<v>\d{2})\.(?P<ver>\d+)"
                     r"\.(?P<made>\d+)")


class FormatError(ValueError):
    """A listing or a granule that does not look like the archive measured."""


def tile_name(h, v):
    return f"h{int(h):02d}v{int(v):02d}"


def tile_hv(name):
    """`h18v04` -> (18, 4). Refuses anything else."""
    m = TILE.match(str(name))
    if not m:
        raise FormatError(f"{name!r} is not a MODIS sinusoidal tile name")
    h, v = int(m.group(1)), int(m.group(2))
    if not (0 <= h < H_TILES and 0 <= v < V_TILES):
        raise FormatError(f"{name!r}: h must be 0..{H_TILES - 1} and v "
                          f"0..{V_TILES - 1}")
    return h, v


def sin_forward(lat, lon):
    """(lat, lon) degrees -> (x, y) metres on the MODLAND sinusoidal grid."""
    phi = np.radians(np.asarray(lat, np.float64))
    lam = np.radians(np.asarray(lon, np.float64))
    return R_SPHERE * lam * np.cos(phi), R_SPHERE * phi


def sin_inverse(x, y):
    """(x, y) metres -> (lat, lon) degrees. NaN where the point is off the
    projected globe (|x| > pi R cos(lat)), which the rectangle of an edge
    tile really does reach."""
    x = np.asarray(x, np.float64)
    y = np.asarray(y, np.float64)
    with np.errstate(invalid="ignore", divide="ignore"):
        phi = y / R_SPHERE
        lam = x / (R_SPHERE * np.cos(phi))
        lat = np.degrees(phi)
        lon = np.degrees(lam)
        bad = (np.abs(lat) > 90.0) | (np.abs(lon) > 180.0) | ~np.isfinite(lon)
        lat = np.where(np.abs(lat) > 90.0, np.nan, lat)
        lon = np.where(bad, np.nan, lon)
    return lat, lon


def tile_bounds_m(h, v):
    """(x west, y south, x east, y north) of tile (h, v), metres."""
    x0 = X0_GLOBAL + int(h) * TILE_M
    y1 = Y0_GLOBAL - int(v) * TILE_M
    return x0, y1 - TILE_M, x0 + TILE_M, y1


def sin_grid(h, v, px=PX_500):
    """One 500 m sinusoidal tile's grid, JSON-able for `tile_grid.json`.

    `x0`/`y0` are the tile's WEST and NORTH edges and `dy` is NEGATIVE, so
    row 0 is the northernmost row — the HDF-EOS grid's own orientation.
    """
    px = int(px)
    dx = TILE_M / px
    x0, ys, x1, y0 = tile_bounds_m(h, v)
    lat_n, _ = sin_inverse(0.0, y0)
    lat_s, _ = sin_inverse(0.0, ys)
    return {
        "H": px, "W": px, "pixel_m": round(dx, 6),
        "pixel_km": round(dx / 1000.0, 6),
        "crs": "MODLAND Sinusoidal (no EPSG code; GCTP_SNSOID on a sphere)",
        "proj4": PROJ4,
        "projection": {"grid_mapping_name": "sinusoidal",
                       "longitude_of_projection_origin": 0.0,
                       "earth_radius": R_SPHERE,
                       "false_easting": 0.0, "false_northing": 0.0,
                       "reference": ("modis-land.gsfc.nasa.gov/GCTP.html, "
                                     "read 2026-09-20: MODLAND Sinusoidal "
                                     "Grid, sphere 6371007.181 m, central "
                                     "meridian 0, FE = FN = 0")},
        "x0": x0, "dx": dx, "y0": y0, "dy": -dx,
        "coordinates": ("x = x0 + (col + 0.5) * dx metres and y = y0 + "
                        "(row + 0.5) * dy with dy NEGATIVE; then lat = "
                        "degrees(y / R) and lon = degrees(x / (R * cos(lat))) "
                        f"with R = {R_SPHERE} m. x0/y0 are the tile's WEST "
                        "and NORTH edges"),
        "row_order": ("row 0 is the NORTHERNMOST row and col 0 the "
                      "westernmost — the HDF-EOS grid's own orientation "
                      "(GridOrigin = HDFE_CENTER, upper-left corner first), "
                      "so nothing is flipped"),
        "tile": tile_name(h, v), "h": int(h), "v": int(v),
        "tiles_across": H_TILES, "tiles_down": V_TILES,
        "tile_metres": TILE_M,
        "extent_m": [x0, ys, x1, y0],
        "extent_m_note": "[west, south, east, north] in projected metres",
        "latitude_band": [round(float(lat_s), 6), round(float(lat_n), 6)],
        "latitude_band_note": ("the tile's south and north edges in degrees; "
                               "its LONGITUDE span widens away from the "
                               "equator because the projection is sinusoidal"),
        "no_tile_corners": ("no lat/lon corner table is stored: the "
                            "rectangle of a tile near the grid's edge has "
                            "corners off the projected globe, where the "
                            "inverse above has no longitude, and a clamped "
                            "number that looks like a coordinate would be "
                            "worse than the formula"),
    }


def log2_fp_for(px=PX_500):
    """The footprint exponent of a `px`-wide tile, log2(km / 27.83)."""
    return float(np.log2(TILE_M / int(px) / 1000.0 / BIN_KM))


# =================================================================== CMR ===
def granule_key(title):
    """`MOD15A2H.A2020185.h18v04.061.2020340145645[.hdf]` ->
    (date, (h, v), production int).

    Raises `FormatError` on anything that is not a MODIS tile granule id: a
    title the parser cannot read is a refusal, never a skipped tile.
    """
    m = GRANULE.match(str(title or ""))
    if not m:
        raise FormatError(f"{title!r}: not a MODIS tile granule id "
                          f"(<PROD>.AYYYYDDD.hHHvVV.<ver>.<production>)")
    y, doy = int(m.group("year")), int(m.group("doy"))
    if not 1 <= doy <= 366:
        raise FormatError(f"{title!r}: day of year {doy}")
    try:
        d = dt.date(y, 1, 1) + dt.timedelta(days=doy - 1)
    except ValueError:
        raise FormatError(f"{title!r}: no such date") from None
    if d.year != y:
        raise FormatError(f"{title!r}: day {doy} is not in {y}")
    hv = (int(m.group("h")), int(m.group("v")))
    if not (0 <= hv[0] < H_TILES and 0 <= hv[1] < V_TILES):
        raise FormatError(f"{title!r}: tile {hv} is outside the "
                          f"{H_TILES} x {V_TILES} grid")
    return d, hv, int(m.group("made"))


def data_link(entry, suffix):
    """The one https `-protected/` data link of a CMR granule entry."""
    for ln in entry.get("links") or ():
        href = str(ln.get("href") or "")
        rel = str(ln.get("rel") or "")
        if (rel.endswith("/data#") and href.startswith("https://")
                and href.endswith(suffix) and "protected" in href):
            return href
    return None


def parse_cmr(entries, suffix):
    """CMR entries -> ({(date, (h, v)): rec}, counts). REFUSES a granule with
    no data link; keeps the HIGHEST production timestamp per (date, tile)."""
    out, counts = {}, {}
    for e in entries:
        title = e.get("title") or e.get("producer_granule_id") or ""
        d, hv, made = granule_key(title)
        url = data_link(e, suffix)
        if not url:
            raise FormatError(
                f"{title}: no https '-protected/' {suffix} data link in the "
                f"CMR entry — the archive moved and the adapter must be "
                f"re-read before anything is built")
        try:
            size = float(e.get("granule_size") or 0.0) * 1024 * 1024
        except (TypeError, ValueError):
            size = 0.0
        rec = {"title": str(title), "url": url, "bytes_cmr": int(size),
               "made": made, "tile": tile_name(*hv),
               "time_start": e.get("time_start"),
               "time_end": e.get("time_end"), "id": e.get("id")}
        old = out.get((d, hv))
        if old is None:
            out[(d, hv)] = rec
            continue
        counts["granules_superseded"] = \
            counts.get("granules_superseded", 0) + 1
        counts.setdefault("granules_superseded_sample", [])
        if len(counts["granules_superseded_sample"]) < 12:
            counts["granules_superseded_sample"].append(str(title))
        if made > old["made"]:
            out[(d, hv)] = rec
    return out, counts


def cmr_page(params, after=None, attempts=4):
    """One CMR page -> (entries, next CMR-Search-After, hits, bytes read)."""
    url = CMR + "?" + urllib.parse.urlencode(params, doseq=True)
    headers = dict(mc.UA)
    if after:
        headers["CMR-Search-After"] = after
    err = None
    for i in range(max(1, attempts)):
        try:
            req = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(req, timeout=mc.TIMEOUT) as r:
                raw = r.read()
                nxt = r.headers.get("CMR-Search-After")
                hits = int(r.headers.get("CMR-Hits", -1))
            f10b.count_bytes(len(raw))
            feed = json.loads(raw).get("feed") or {}
            return list(feed.get("entry") or ()), nxt, hits, len(raw)
        except (IOError, OSError, ValueError) as e:            # noqa: PERF203
            err = e
            if i < attempts - 1:
                time.sleep(3.0 * (2 ** i))
    raise IOError(f"{CMR} ({params.get('short_name')}): "
                  f"{type(err).__name__}: {err}")


def cmr_year(short_name, version, year, suffix, attempts=4, page=PAGE,
             tiles=None):
    """Every granule of ONE PRODUCT YEAR -> ({(date, (h, v)): rec}, counts).

    The pattern is `<SHORT>.A<YYYY>*`, i.e. the product's own A-date, NOT a
    `temporal` window: a composite that straddles New Year belongs to the
    year its A-date names, and a temporal filter would drag in the
    neighbouring year's last composite as well (measured: the 2020 temporal
    window returns 13,466 MOD15A2H granules, 274 of which are 2019's last
    composite).

    `tiles`, when a build is restricted to a few of them, narrows the pattern
    to `<SHORT>.A<YYYY>*.<tile>.*` and asks once per tile. That is what makes
    a PROBE of this store affordable: one MOD15A2H year is 13,192 granules and
    70 MB of JSON, and one tile of it is 46 granules and 240 kB.
    """
    want = list(tiles or [])
    patterns = ([f"{short_name}.A{int(year):04d}*.{t}.*" for t in want]
                if want else [f"{short_name}.A{int(year):04d}*"])
    out, counts = {}, {}
    pages, nbytes, hits = 0, 0, 0
    t0 = time.time()
    for pat in patterns:
        params = {"short_name": short_name, "version": version,
                  "page_size": int(page), "sort_key": "start_date",
                  "readable_granule_name": pat,
                  "options[readable_granule_name][pattern]": "true"}
        after = None
        while True:
            entries, after, h, nb = cmr_page(params, after, attempts)
            if h >= 0:
                hits += h
            pages += 1
            nbytes += nb
            got, c = parse_cmr(entries, suffix)
            f10b._merge_counts(counts, c)
            for k, rec in got.items():
                old = out.get(k)
                if old is None:
                    out[k] = rec
                else:
                    counts["granules_superseded"] = \
                        counts.get("granules_superseded", 0) + 1
                    if rec["made"] > old["made"]:
                        out[k] = rec
            if not after or not entries:
                break
            if pages > 400:
                raise FormatError(f"{short_name} {year}: CMR paged past "
                                  f"{pages} pages — refusing to loop")
    counts["cmr_pages"] = pages
    counts["cmr_bytes"] = nbytes
    counts["cmr_hits"] = hits
    counts["cmr_queries"] = len(patterns)
    counts["cmr_seconds"] = round(time.time() - t0, 1)
    counts["granules_listed"] = len(out)
    return out, counts


def check_attrs(attrs, declared, where="?"):
    """An attribute the file CARRIES and that differs is a REFUSAL; one it
    does not carry is COUNTED, not fatal.

    ml/CLAUDE.md §5.17: only a DEFINITE answer may be fatal. A product whose
    scale factor changed is a definite answer and is refused before a single
    rescaled number reaches the store; a product that documents its fill
    value instead of writing it as an attribute is not an answer at all, and
    a store must not be un-buildable because of it. The absences are counted
    by name (`attributes_absent`) and the index records every attribute the
    file really carries, so a human can see which is which.
    """
    absent = {}
    for name, kv in (declared or {}).items():
        have = (attrs or {}).get(name) or {}
        for k, v in kv.items():
            if k not in have:
                absent[f"{name}.{k}"] = 1
                continue
            got = have[k]
            same = (list(got) == list(v) if isinstance(v, (list, tuple))
                    else abs(float(got) - float(v)) <= 1e-12)
            if not same:
                raise FormatError(
                    f"{where}: dataset {name!r} has {k} = {got!r}, the "
                    f"adapter declares {v!r} — a product whose scaling or "
                    f"fill value changed is refused, not rescaled")
    return absent


# ================================================================ HDF-EOS ==
STRUCT_KEY = "StructMetadata.0"
_NUM = r"[-+0-9.eE]+"
UL = re.compile(r"UpperLeftPointMtrs=\((" + _NUM + r"),(" + _NUM + r")\)")
LR = re.compile(r"LowerRightMtrs=\((" + _NUM + r"),(" + _NUM + r")\)")
XD = re.compile(r"XDim=(\d+)")
YD = re.compile(r"YDim=(\d+)")
PROJ = re.compile(r"Projection=(\w+)")


def parse_struct(text):
    """`StructMetadata.0` -> {upper_left, lower_right, XDim, YDim, projection}.

    Only the five things the grid check needs are read; the rest of the PVL is
    left alone.
    """
    t = str(text or "")
    out = {}
    m = UL.search(t)
    if m:
        out["upper_left"] = [float(m.group(1)), float(m.group(2))]
    m = LR.search(t)
    if m:
        out["lower_right"] = [float(m.group(1)), float(m.group(2))]
    m = XD.search(t)
    if m:
        out["XDim"] = int(m.group(1))
    m = YD.search(t)
    if m:
        out["YDim"] = int(m.group(1))
    m = PROJ.search(t)
    if m:
        out["projection"] = m.group(1)
    return out


def check_struct(st, h, v, px, where="?"):
    """The granule's own corners against the tile its NAME claims. Raises.

    Tolerance is half a pixel: HDF-EOS writes the corners as decimal metres
    and the sphere radius has three decimals, so the last digit moves.
    """
    x0, ys, x1, y0 = tile_bounds_m(h, v)
    tol = TILE_M / int(px) / 2.0
    if st.get("projection") not in (None, "GCTP_SNSOID"):
        raise FormatError(f"{where}: StructMetadata says projection "
                          f"{st.get('projection')!r}, expected GCTP_SNSOID")
    for key, want in (("upper_left", (x0, y0)), ("lower_right", (x1, ys))):
        got = st.get(key)
        if got is None:
            raise FormatError(f"{where}: StructMetadata has no {key}")
        if abs(got[0] - want[0]) > tol or abs(got[1] - want[1]) > tol:
            raise FormatError(
                f"{where}: StructMetadata {key} = {got}, the tile name "
                f"{tile_name(h, v)} says {list(want)} (tolerance {tol:.3f} m, "
                f"half a pixel) — a granule whose grid moved is refused, not "
                f"stored")
    for key in ("XDim", "YDim"):
        got = st.get(key)
        if got is not None and int(got) != int(px):
            raise FormatError(f"{where}: StructMetadata {key} = {got}, the "
                              f"declared grid is {px}")
    return True


def read_hdf4(path, names, px, want=None, tile=None, cycle=0):
    """Read `names` out of one HDF-EOS2 (HDF4) granule -> (arrays, attrs, st).

    Every dataset must be (px, px) — or (n, px, px) / (px, px, n) when the
    product carries several vegetation CYCLES, in which case `cycle` selects
    one. `want` is `{name: {attr: value}}` of attributes the caller DECLARES;
    an attribute the file carries with a different value is a `FormatError` —
    a rescaled product is refused, not silently converted.
    """
    SD, _ = mc._pyhdf()
    try:
        d = SD(path)
    except Exception as ex:                                     # noqa: BLE001
        raise IOError(f"{os.path.basename(path)}: not a readable HDF4 file "
                      f"({ex}) — a truncated download") from None
    out, attrs = {}, {}
    try:
        st = parse_struct((d.attributes() or {}).get(STRUCT_KEY))
        if tile is not None:
            check_struct(st, *tile_hv(tile), px=px,
                         where=os.path.basename(path))
        have = d.datasets()
        for n in names:
            if n not in have:
                raise FormatError(
                    f"{os.path.basename(path)}: no dataset {n!r} (the file "
                    f"has {sorted(have)[:14]}"
                    f"{' …' if len(have) > 14 else ''})")
            s = d.select(n)
            try:
                a = np.asarray(s[:])
                at = s.attributes()
                attrs[n] = {k: at.get(k) for k in
                            ("_FillValue", "scale_factor", "add_offset",
                             "valid_range", "units", "long_name") if k in at}
                for k, v in ((want or {}).get(n) or {}).items():
                    if k not in at:
                        raise FormatError(
                            f"{os.path.basename(path)}: dataset {n!r} carries "
                            f"no {k!r}; the adapter declares {v!r}")
                    got = at[k]
                    same = (list(got) == list(v)
                            if isinstance(v, (list, tuple))
                            else abs(float(got) - float(v)) <= 1e-12)
                    if not same:
                        raise FormatError(
                            f"{os.path.basename(path)}: dataset {n!r} has "
                            f"{k} = {got!r}, the adapter declares {v!r} — a "
                            f"product whose scaling changed is refused, not "
                            f"rescaled")
                out[n] = take_cycle(a, n, px, cycle,
                                    os.path.basename(path))
            finally:
                s.endaccess()
    finally:
        d.end()
    return out, attrs, st


def take_cycle(a, name, px, cycle, where="?"):
    """A dataset of rank 2 or 3 -> the (px, px) plane for `cycle`.

    MCD12Q2 publishes TWO vegetation cycles a year and HDF-EOS can carry that
    third dimension on either end, so both orders are accepted and anything
    else is a refusal that PRINTS the real shape.
    """
    px = int(px)
    a = np.asarray(a)
    if a.shape == (px, px):
        return a
    if a.ndim == 3 and a.shape[1:] == (px, px):
        if cycle >= a.shape[0]:
            raise FormatError(f"{where}: dataset {name!r} has {a.shape[0]} "
                              f"cycle(s), the adapter asked for {cycle}")
        return a[int(cycle)]
    if a.ndim == 3 and a.shape[:2] == (px, px):
        if cycle >= a.shape[2]:
            raise FormatError(f"{where}: dataset {name!r} has {a.shape[2]} "
                              f"cycle(s), the adapter asked for {cycle}")
        return a[:, :, int(cycle)]
    raise FormatError(f"{where}: dataset {name!r} has shape {a.shape}; the "
                      f"declared grid is ({px}, {px}) and the only extra "
                      f"dimension this reader accepts is the vegetation cycle")


def read_hdf5(path, names, px, want=None, tile=None, cycle=0):
    """The same for an HDF5 (HDF-EOS5) granule — VIIRS' VNP15A2H v002."""
    try:
        import h5py
    except ImportError:                                     # pragma: no cover
        sys.exit("the VIIRS continuation group needs the `h5py` package "
                 "(pip install h5py)")
    try:
        f = h5py.File(path, "r")
    except Exception as ex:                                     # noqa: BLE001
        raise IOError(f"{os.path.basename(path)}: not a readable HDF5 file "
                      f"({ex}) — a truncated download") from None
    out, attrs = {}, {}
    try:
        st = parse_struct(_h5_struct(f))
        if tile is not None:
            check_struct(st, *tile_hv(tile), px=px,
                         where=os.path.basename(path))
        found = {}

        def visit(name, obj):
            if hasattr(obj, "shape"):
                found.setdefault(name.rsplit("/", 1)[-1], name)
        f.visititems(visit)
        for n in names:
            if n not in found:
                raise FormatError(
                    f"{os.path.basename(path)}: no dataset {n!r} (the file "
                    f"has {sorted(found)[:14]}"
                    f"{' …' if len(found) > 14 else ''})")
            ds = f[found[n]]
            at = {k: _h5_attr(v) for k, v in ds.attrs.items()}
            attrs[n] = {k: at.get(k) for k in
                        ("_FillValue", "scale_factor", "add_offset",
                         "valid_range", "units", "long_name") if k in at}
            for k, v in ((want or {}).get(n) or {}).items():
                if k not in at:
                    raise FormatError(
                        f"{os.path.basename(path)}: dataset {n!r} carries no "
                        f"{k!r}; the adapter declares {v!r}")
                got = at[k]
                same = (list(np.atleast_1d(got)) == list(v)
                        if isinstance(v, (list, tuple))
                        else abs(float(np.ravel(got)[0]) - float(v)) <= 1e-12)
                if not same:
                    raise FormatError(
                        f"{os.path.basename(path)}: dataset {n!r} has "
                        f"{k} = {got!r}, the adapter declares {v!r}")
            out[n] = take_cycle(np.asarray(ds[:]), n, px, cycle,
                                os.path.basename(path))
    finally:
        f.close()
    return out, attrs, st


def _h5_struct(f):
    for key in ("HDFEOS INFORMATION/StructMetadata.0",
                "/HDFEOS INFORMATION/StructMetadata.0"):
        try:
            v = f[key][()]
        except (KeyError, TypeError):
            continue
        return v.decode("utf-8", "replace") if isinstance(v, bytes) else str(v)
    return ""


def _h5_attr(v):
    if isinstance(v, bytes):
        return v.decode("utf-8", "replace")
    if isinstance(v, np.ndarray):
        return v.tolist() if v.size > 1 else v.ravel()[0].item()
    return v


READERS = {"hdf4": read_hdf4, "hdf5": read_hdf5}
SUFFIX = {"hdf4": ".hdf", "hdf5": ".h5"}


def dump_datasets(path, fmt, limit=40):
    """Every dataset's name, shape, dtype and attributes — no shape enforced.

    What a refusal prints, so one hosted round trip is enough to learn what
    the file really carries rather than one per wrong guess.
    """
    if fmt == "hdf5":
        import h5py
        out = {}
        with h5py.File(path, "r") as f:
            def visit(name, obj):
                if hasattr(obj, "shape") and len(out) < limit:
                    out[name] = {"shape": list(obj.shape),
                                 "dtype": str(obj.dtype),
                                 "attributes": {k: _h5_attr(v)
                                                for k, v in obj.attrs.items()}}
            f.visititems(visit)
        return out
    SD, _ = mc._pyhdf()
    d = SD(path)
    out = {}
    try:
        for n in sorted(d.datasets())[:limit]:
            s = d.select(n)
            try:
                _, rank, dims, nt, _ = s.info()
                out[n] = {"rank": rank, "dims": list(dims), "number_type": nt,
                          "attributes": {k: (list(v) if isinstance(
                              v, (list, tuple)) else v)
                              for k, v in s.attributes().items()}}
            finally:
                s.endaccess()
    finally:
        d.end()
    return out


# ============================================================ the adapter ==
class SinTileAdapter(sh.GridAdapter):
    """A MODIS 500 m sinusoidal-tile field: one granule per tile per period.

    A subclass declares `collections` ({group prefix: (short_name, version,
    file format)}), `tiles` (the measured tuple of tile names the product
    publishes), `channels`, `dtype`, the `datasets` it reads and the
    attributes it expects (`datasets_want`), and implements
    `frame_from(arrs, attrs, key)` -> a float32 `[px, px, C]` array with NaN
    for not-measured. It also implements `periods(ctx, group, year)` ->
    {(bin, frame): period key} and `period_days(key)` -> (first, last) so the
    layout's `frame_table` can be written; everything else — the grid, the
    CMR listing, the login, the download, the grid check, the frame filing,
    the smoke's local-archive mode — is inherited.

    GROUPS are `<sensor>_<tile>` when the store has more than one sensor and
    `<tile>` when it has one, so a reader asks for the instrument it wants.
    """

    family = "1tf"
    distribution = "public"
    time_dtype = "int32"
    credentials = ("EARTHDATA_USERNAME", "EARTHDATA_PASSWORD")
    per_year = True
    tile = sh.TILE
    zstd_level = sh.DEFAULT_LEVEL
    px = PX_500
    log2_fp = log2_fp_for(PX_500)

    collections = {}             # {group prefix: (short_name, version, fmt)}
    groups_default = ()
    groups_env = ""
    tiles_env = ""
    px_env = ""                  # shrinks the tile for `--smoke`
    tiles = ()                   # the measured tile tuple
    datasets = ()
    datasets_want = {}
    cycle = 0                    # which vegetation cycle, where there are two
    WORKERS = 3
    TILE_QUERY_MAX = 12          # above this, list the whole product year

    def __init__(self):
        self.px = int(os.environ.get(self.px_env) or 0) or PX_500
        self.groups = self._pick("sensor", self.groups_env,
                                 sorted(self.collections),
                                 list(self.groups_default))
        want = [t.strip() for t in
                (os.environ.get(self.tiles_env) or "").split(",") if t.strip()]
        for t in want:
            tile_hv(t)
        self.tile_names = want or list(self.tiles)
        self._listing = {}
        self._session = None
        if tuple(self.groups) != tuple(self.groups_default):
            self.notes = (f"{self.notes}\nSENSORS: {self.groups_env} selected "
                          f"{list(self.groups)} instead of the default "
                          f"{list(self.groups_default)}.")
        if want:
            self.notes = (
                f"{self.notes}\nRESTRICTED BUILD: {self.tiles_env} limited "
                f"this store to {len(want)} of the product's "
                f"{len(self.tiles)} tiles ({', '.join(want[:8])}"
                f"{' ...' if len(want) > 8 else ''}). It is NOT the whole "
                f"product.")
        if self.px != PX_500:
            self.notes = (f"{self.notes}\nSMOKE GRID: {self.px_env} set each "
                          f"tile to {self.px} x {self.px} instead of "
                          f"{PX_500} x {PX_500}. This is a synthetic store.")

    def _pick(self, what, env, avail, default):
        raw = (os.environ.get(env) or "").strip()
        if not raw:
            return list(default or avail[:1])
        picked = [g.strip() for g in raw.split(",") if g.strip()]
        bad = [g for g in picked if g not in avail]
        if bad or not picked:
            sys.exit(f"REFUSING {self.store}: {env}={raw!r} names "
                     f"{bad or 'nothing'}; the {what}s this adapter can build "
                     f"are {avail}")
        return picked

    # ------------------------------------------------------------ the grid --
    def group_name(self, sensor, tile):
        return tile if len(self.collections) == 1 else f"{sensor}_{tile}"

    def group_parts(self, group):
        g = str(group)
        if len(self.collections) == 1:
            return self.groups[0], g
        sensor, _, tile = g.partition("_")
        if sensor not in self.collections:
            raise FormatError(f"{group!r}: {sensor!r} is not one of "
                              f"{sorted(self.collections)}")
        return sensor, tile

    def group_grids(self):
        return {self.group_name(s, t): sin_grid(*tile_hv(t), px=self.px)
                for s in self.groups for t in self.tile_names}

    # ------------------------------------------------------------- listing --
    def collection(self, sensor):
        return self.collections[sensor]

    def local_listing_path(self, ctx, sensor, year):
        return os.path.join(ctx.source_dir, self.store,
                            f"cmr_{sensor}_{int(year):04d}.json")

    def local_granule_path(self, ctx, sensor, rec):
        name = os.path.basename(str(rec["url"]).split("?")[0])
        return os.path.join(ctx.source_dir, self.store, "granules", sensor,
                            name)

    def listing(self, ctx, sensor, year):
        """{(date, (h, v)): rec}, counts — from CMR, or the smoke's copy."""
        key = (sensor, int(year))
        if key in self._listing:
            return self._listing[key]
        sn, ver, fmt = self.collection(sensor)
        if ctx.source_dir:
            p = self.local_listing_path(ctx, sensor, year)
            if not os.path.exists(p):
                self._listing[key] = ({}, {"granules_listed": 0,
                                           "year_not_listed": 1})
                return self._listing[key]
            with open(p, "rb") as fh:
                raw = fh.read()
            ctx.count_bytes(len(raw))
            entries = (json.loads(raw).get("feed") or {}).get("entry") or []
            try:
                files, counts = parse_cmr(entries, SUFFIX[fmt])
            except FormatError as e:
                sys.exit(f"REFUSING {self.store}: {p}: {e}")
            counts["granules_listed"] = len(files)
            counts["cmr_pages"] = 0
        else:
            # A RESTRICTED build asks CMR per tile: one MOD15A2H year is
            # 13,192 granules and 70 MB of JSON, one tile of it 46 granules
            # and 240 kB, and a probe measures a handful of tiles.
            narrow = (list(self.tile_names)
                      if 0 < len(self.tile_names) <= self.TILE_QUERY_MAX
                      else None)
            try:
                files, counts = cmr_year(sn, ver, year, SUFFIX[fmt],
                                         attempts=ctx.a.attempts,
                                         tiles=narrow)
            except FormatError as e:
                sys.exit(f"REFUSING {self.store}: the CMR listing of {sn} "
                         f"v{ver} for {year}: {e}")
        self._listing[key] = (files, counts)
        return self._listing[key]

    def listing_years(self, ctx):
        """The product years the window's bins can need.

        A period filed under a bin can have an A-date in the NEIGHBOURING
        calendar year (an eight-day composite that straddles New Year, a
        phenology year whose July bin opens in June), so one year either side
        is listed as well and the extra listing is cached.
        """
        ys = sorted(ctx.years)
        if not ys:
            return []
        return list(range(min(ys) - 1, max(ys) + 2))

    def tiles_live(self, ctx, sensor, years):
        out = set()
        for y in years:
            files, _ = self.listing(ctx, sensor, y)
            out |= {hv for (_d, hv) in files}
        return {tile_name(*hv) for hv in out}

    # ------------------------------------------------------------- session --
    def session(self):
        if self._session is None:
            self._session = mc.earthdata_session()
        return self._session

    def fetch_preflight(self, ctx):
        """The netrc must be there BEFORE the first granule (§0.3)."""
        if ctx.source_dir:
            return None
        if not mc.netrc_has_urs():
            print(f"::warning::{self.store}: no netrc entry for "
                  f"{mc.URS_HOST} was found; the download will fall back to "
                  f"whatever credentials the environment provides and may be "
                  f"refused. family1-build.yml writes that file on hosted "
                  f"runners.")
        return None

    def granule_file(self, ctx, sensor, rec):
        """The granule on local disk -> (path, tmpdir or why)."""
        import tempfile
        if ctx.source_dir:
            p = self.local_granule_path(ctx, sensor, rec)
            if not os.path.exists(p):
                return None, f"{rec['title']}: listed and absent"
            ctx.count_bytes(os.path.getsize(p))
            return p, None
        tmpdir = tempfile.mkdtemp(prefix=f"{self.store}_", dir=ctx.scratch)
        name = os.path.basename(rec["url"].split("?")[0])
        p = os.path.join(tmpdir, name)
        try:
            # AT LEAST SIX ATTEMPTS, whatever --attempts says: a lane is hours
            # long and Earthdata Login stalls for a minute at a time from
            # runner IPs (#151), so losing a tile to one refused handshake
            # costs far more than fifteen minutes of backoff.
            n, why = mc.download(self.session(), rec["url"], p,
                                 attempts=max(int(ctx.a.attempts), 6))
        except Exception:
            import shutil
            shutil.rmtree(tmpdir, ignore_errors=True)
            raise
        if n is None:
            import shutil
            shutil.rmtree(tmpdir, ignore_errors=True)
            return None, f"{rec['url']}: listed, and {why}"
        return p, tmpdir

    # ---------------------------------------------------------- one granule --
    def read_granule(self, ctx, sensor, tile, path, rec=None):
        """One granule -> (float32 [px, px, C], counts, out_of_bounds).

        `rec` is the CMR record the path came from; `frame_from` needs it
        because what a raw number MEANS can depend on the granule's own A-date
        (a phenology date is days since 1970 and becomes a day number in the
        product year), and reading that off shared state would be wrong the
        moment two frames are in flight at once.
        """
        fmt = self.collection(sensor)[2]
        with mc.HDF_LOCK:
            arrs, attrs, st = READERS[fmt](
                path, list(self.datasets_of(sensor)), self.px,
                self.datasets_want_of(sensor), tile=tile, cycle=self.cycle)
        a, counts = self.frame_from(arrs, attrs, rec)
        del arrs
        oob = mc.mask_bounds_low_memory(self, a)
        counts = dict(counts or {})
        counts["files_read"] = 1
        counts["valid_pixels"] = int(np.isfinite(a).sum())
        return a, counts, oob

    def frame_from(self, arrs, attrs, rec=None):
        """{dataset: array} -> float32 [px, px, C], NaN for not-measured."""
        raise NotImplementedError

    def datasets_of(self, sensor):
        """The datasets to read from THIS sensor's granule.

        A continuation instrument can name the same quantity differently —
        VIIRS' VNP15A2H calls MODIS' `Lai_500m` simply `Lai` (measured on LP
        DAAC's own layer tables, 2026-09-20) — so the name list is per sensor
        and the default is the one list.
        """
        return self.datasets

    def datasets_want_of(self, sensor):
        return self.datasets_want

    def read_header(self, ctx, sensor, tile, rec):
        """`index`'s one real granule: opened, grid and attributes CHECKED."""
        import shutil
        tmpdir = None
        fmt = self.collection(sensor)[2]
        try:
            p, why = self.granule_file(ctx, sensor, rec)
            if p is None:
                sys.exit(f"REFUSING {self.store}: the granule "
                         f"{rec['title']} the listing names could not be "
                         f"read: {why}")
            tmpdir = why if not ctx.source_dir else None
            try:
                with mc.HDF_LOCK:
                    arrs, attrs, st = READERS[fmt](
                        p, list(self.datasets_of(sensor)), self.px,
                        self.datasets_want_of(sensor), tile=tile,
                        cycle=self.cycle)
            except (FormatError, IOError) as e:
                # ONE ROUND TRIP IS ENOUGH TO LEARN EVERYTHING. A declared
                # attribute that does not match is a refusal, and the refusal
                # PRINTS what the file really says — otherwise every wrong
                # guess costs another hosted run to find out the next one.
                try:
                    with mc.HDF_LOCK:
                        table = dump_datasets(p, fmt)
                    print("::notice::what the file actually carries:\n"
                          + json.dumps(table, indent=1, default=str))
                except Exception as e2:                         # noqa: BLE001
                    print(f"::warning::could not dump the file's own dataset "
                          f"table either: {type(e2).__name__}: {e2}")
                sys.exit(f"REFUSING {self.store}: {rec['title']} does not "
                         f"match the declared grid or attributes: {e}")
            out = {"title": rec["title"], "url": rec["url"],
                   "bytes": os.path.getsize(p), "tile": tile,
                   "struct_metadata": st,
                   "datasets_read": {}}
            for n, a in arrs.items():
                at = attrs.get(n) or {}
                fill = at.get("_FillValue")
                good = a if fill is None else \
                    a[a != np.asarray(fill, a.dtype)]
                out["datasets_read"][n] = {
                    "shape": list(a.shape), "dtype": str(a.dtype),
                    "attributes": {k: (list(v) if isinstance(v, (list, tuple))
                                       else v) for k, v in at.items()},
                    "fill_pixels": (None if fill is None
                                    else int(a.size - good.size)),
                    "raw_min": (int(good.min()) if good.size else None),
                    "raw_max": (int(good.max()) if good.size else None)}
            return out
        finally:
            if tmpdir:
                shutil.rmtree(tmpdir, ignore_errors=True)

    def index_probe_tiles(self, files):
        """Which listed (date, tile) the index stage tries to open, in order.

        The NEWEST date first: the newest granule is the one whose format a
        future build will meet. Two more are the fallbacks, because Earthdata
        Login refuses a runner's handshake for minutes at a time (#164, #167)
        and a store must not be un-indexable because one granule's login
        happened to be the one that stalled.
        """
        keys = sorted(files)
        return [keys[-1], keys[0], keys[len(keys) // 2]]

    def read_first_file(self, ctx, sensor, files):
        """One real granule, read — and a NETWORK failure is not a verdict.

        A file that IS read and does not match the declared grid or attributes
        is fatal: that is a definite answer about the product. A login that
        does not answer is not an answer at all (ml/CLAUDE.md §5.17), so the
        three candidates are tried in turn and, if every one fails on the
        NETWORK, the index records why and the stage goes on — the fetch meets
        the same login within the minute, where a failure is definite.
        """
        tried = []
        for key in self.index_probe_tiles(files):
            rec = files[key]
            try:
                meta = self.read_header(ctx, sensor, rec["tile"], rec)
            except (IOError, OSError) as e:
                tried.append({"granule": rec["title"],
                              "error": f"{type(e).__name__}: {str(e)[:300]}"})
                print(f"::warning::{self.store}: the index could not open "
                      f"{rec['title']} ({type(e).__name__}) — trying another "
                      f"granule; a login that does not answer is not a "
                      f"verdict about the product")
                continue
            if tried:
                meta["granules_that_would_not_open"] = tried
            return meta
        print(f"::warning::{self.store}: NO granule could be opened at index "
              f"time ({tried}) — the fetch stage meets the same login, where "
              f"a failure is definite")
        return {"read": False, "granules_that_would_not_open": tried,
                "note": ("a network failure at index time is not a verdict "
                         "about the product (ml/CLAUDE.md §5.17); the grid "
                         "and every declared attribute are re-checked in "
                         "EVERY granule the fetch reads")}

    # ------------------------------------------------------- the contract --
    def periods(self, ctx, sensor, year):
        """{(bin, frame): key} for the product's periods with an A-date in
        `year`. `key` is whatever `period_days` and `granule_date` understand.
        """
        raise NotImplementedError

    def granule_date(self, key):
        """The A-date whose granules carry the period `key`."""
        raise NotImplementedError

    def period_days(self, key):
        """(first day, last day) the period `key` covers."""
        raise NotImplementedError

    def slot_map(self, ctx, sensor):
        """{(bin, frame): key} over every year the window can need.

        REFUSES a collision, the way `chirps05` does: two periods in one slot
        would mean one of them was dropped or overwritten, and the layout has
        nowhere to put the second.
        """
        out = {}
        for y in self.listing_years(ctx):
            for k, key in self.periods(ctx, sensor, y).items():
                if k in out and out[k] != key:
                    sys.exit(f"REFUSING {self.store}: two periods land in "
                             f"slot {k}: {out[k]} and {key}. "
                             f"frames_per_bin = {self.frames_per_bin} cannot "
                             f"carry this product's calendar — the check "
                             f"`chirps05` wrote for the same reason.")
                out[k] = key
        return out

    def record(self, ctx, sensor):
        """(first, last) A-date the listing holds over the window's years."""
        lo = hi = None
        for y in self.listing_years(ctx):
            files, _ = self.listing(ctx, sensor, y)
            for (d, _hv) in files:
                lo = d if lo is None else min(lo, d)
                hi = d if hi is None else max(hi, d)
        return lo, hi

    def key_of_date(self, d):
        """The period key whose granules carry the A-date `d`."""
        raise NotImplementedError

    def record_span(self, ctx, sensor):
        """(first day, last day) the listed PERIODS cover.

        Not the A-dates: a product year's A-date is its 1 January and its
        period runs to 31 December, and an eight-day composite's A-date is its
        first day and its period runs seven days further. A slot inside the
        record must not read as `after_record` just because it is past the
        last A-date.
        """
        lo, hi = self.record(ctx, sensor)
        if lo is None:
            return None, None
        return (self.period_days(self.key_of_date(lo))[0],
                self.period_days(self.key_of_date(hi))[1])

    def classify(self, ctx, sensor, tile, b, f):
        """-> (record or None, reason or None) for one (group, bin, frame)."""
        key = self.slot_map(ctx, sensor).get((int(b), int(f)))
        lo, hi = self.record(ctx, sensor)
        d_lo, d_hi = self.record_span(ctx, sensor)
        t = sh.frame_start_seconds(b, f, self.frame_seconds)
        if key is None:
            if lo is None:
                return None, "before_record"
            if t < f10b.seconds_since_epoch(d_lo):
                return None, "before_record"
            if t > f10b.seconds_since_epoch(d_hi) + 86399:
                return None, "after_record"
            return None, self.empty_slot_reason
        d = self.granule_date(key)
        if lo is None or d < lo:
            return None, "before_record"
        if d > hi:
            return None, "after_record"
        files, _ = self.listing(ctx, sensor, d.year)
        rec = files.get((d, tile_hv(tile)))
        if rec is None:
            return None, "absent_upstream"
        return rec, None

    # A bin the product was NEVER going to have a frame for, because its
    # cadence (a year, eight days) is coarser than the five-day axis. The
    # layout SKIPS such a bin instead of writing a shard and an index that
    # say "nothing here" (build_family1_stores, "WHY A BIN CAN BE SKIPPED
    # ALTOGETHER"); it is deliberately NOT `absent_upstream`, which is a
    # frame the product SHOULD have and does not.
    empty_slot_reason = sh.FRAME_NO_FRAME_IN_BIN

    def fetch_frames(self, ctx, wanted):
        from family1.adapters import _common as cm
        jobs = []
        for (g, b, f) in wanted:
            sensor, tile = self.group_parts(g)
            rec, why = self.classify(ctx, sensor, tile, b, f)
            jobs.append((g, b, f, sensor, tile, rec, why))
        todo = [j for j in jobs if j[5] is not None]

        def work(job):
            g, b, f, sensor, tile, rec, why = job
            try:
                p, tmpdir = self.granule_file(ctx, sensor, rec)
                return job, p, tmpdir, None
            except Exception as e:                              # noqa: BLE001
                return job, None, None, e

        workers = 1 if ctx.source_dir else self.WORKERS
        it = cm.ordered_map(work, todo, workers, lookahead=workers + 1)
        t0 = time.time()
        for (g, b, f, sensor, tile, rec, why) in jobs:
            if rec is None:
                yield g, b, f, None, {"frame_missing": why}
                continue
            job, p, tmpdir, err = next(it)
            if job[:3] != (g, b, f):
                sys.exit(f"{self.store}: the download pool answered "
                         f"{job[:3]} where {(g, b, f)} was asked for")
            if err is not None:
                raise err
            try:
                if p is None:
                    ctx.note_absent(f"{g} {rec['title']}", tmpdir)
                    continue
                try:
                    arr, counts, oob = self.read_granule(ctx, sensor, tile, p,
                                                         rec)
                except FormatError as e:
                    sys.exit(f"REFUSING {self.store}: {rec['title']}: {e}")
                except (IOError, OSError) as e:
                    ctx.note_absent(f"{g} {rec['title']}",
                                    f"{type(e).__name__}: {e}")
                    continue
                counts["bytes_files"] = os.path.getsize(p)
            finally:
                if tmpdir:
                    import shutil
                    shutil.rmtree(tmpdir, ignore_errors=True)
            if oob:
                counts["out_of_bounds"] = oob
            counts["fetch_seconds"] = round(time.time() - t0, 2)
            t0 = time.time()
            yield g, b, f, arr, counts

    # --------------------------------------------------------------- smoke --
    def smoke_reinit(self):
        self.__init__()


# ============================================================ smoke help ===
def cmr_json(records):
    """A CMR granules.json response in the service's own shape.

    `records` is [(title, url, size MB)]. The smoke writes this so the REAL
    parser (`parse_cmr`) is what the test exercises.
    """
    entries = []
    for i, (title, url, mb) in enumerate(records):
        d, _hv, _made = granule_key(title)
        entries.append({
            "producer_granule_id": title, "title": title,
            "id": f"G{2000 + i}-TEST",
            "granule_size": f"{float(mb)}",
            "time_start": f"{d}T00:00:00.000Z",
            "time_end": f"{d}T23:59:59.000Z",
            "day_night_flag": "DAY",
            "links": [
                {"rel": "http://esipfed.org/ns/fedsearch/1.1/data#",
                 "href": url, "hreflang": "en-US"},
                {"rel": "http://esipfed.org/ns/fedsearch/1.1/s3#",
                 "href": "s3://bucket/" + os.path.basename(url)},
                {"rel": "http://esipfed.org/ns/fedsearch/1.1/browse#",
                 "href": "https://example.invalid/browse.jpg"},
            ]})
    return {"feed": {"updated": "2026-09-20T00:00:00.000Z",
                     "id": CMR, "title": "ECHO granule metadata",
                     "entry": entries}}


def struct_metadata(h, v, px):
    """A `StructMetadata.0` string in HDF-EOS' own PVL shape, for the smoke."""
    x0, ys, x1, y0 = tile_bounds_m(h, v)
    return ("GROUP=SwathStructure\nEND_GROUP=SwathStructure\n"
            "GROUP=GridStructure\n\tGROUP=GRID_1\n"
            "\t\tGridName=\"MOD_Grid_SMOKE\"\n"
            f"\t\tXDim={int(px)}\n\t\tYDim={int(px)}\n"
            f"\t\tUpperLeftPointMtrs=({x0:.6f},{y0:.6f})\n"
            f"\t\tLowerRightMtrs=({x1:.6f},{ys:.6f})\n"
            "\t\tProjection=GCTP_SNSOID\n"
            f"\t\tProjParams=({R_SPHERE},0,0,0,0,0,0,0,0,0,0,0,0)\n"
            "\t\tSphereCode=-1\n\t\tGridOrigin=HDFE_GD_UL\n"
            "\tEND_GROUP=GRID_1\nEND_GROUP=GridStructure\nEND\n")


def write_smoke_hdf4(path, arrays, h, v, px, attrs=None):
    """One synthetic HDF-EOS2 granule: the datasets, their attributes and a
    real `StructMetadata.0` for the tile."""
    SD, SDC = mc._pyhdf()
    if os.path.exists(path):
        os.remove(path)
    f = SD(path, SDC.WRITE | SDC.CREATE)
    try:
        setattr(f, STRUCT_KEY, struct_metadata(h, v, px))
        for name, a in arrays.items():
            t = {np.dtype("uint16"): SDC.UINT16, np.dtype("int16"): SDC.INT16,
                 np.dtype("uint8"): SDC.UINT8}[a.dtype]
            dims = tuple(int(x) for x in a.shape)
            s = f.create(name, t, dims)
            s[:] = a
            s.endaccess()
    finally:
        f.end()
    if attrs:
        mc.set_sds_attrs(path, attrs)


def write_smoke_archive(root, store, sensor, year, records, arrays, px,
                        url_prefix, attrs=None, absent=()):
    """The synthetic archive in its real layout: `cmr_<sensor>_<year>.json`
    and `granules/<sensor>/*.hdf`.

    `records` is [(title, (date, (h, v)))] and `arrays` {title: {ds: array}}.
    A title in `absent` is LISTED and its file is not written (a download that
    404s).
    """
    base = os.path.join(root, store)
    gdir = os.path.join(base, "granules", sensor)
    os.makedirs(gdir, exist_ok=True)
    recs = []
    for (title, (h, v)) in records:
        name = title + ".hdf"
        recs.append((title, url_prefix + name, 1.0))
        if title in absent:
            continue
        write_smoke_hdf4(os.path.join(gdir, name), arrays[title], h, v, px,
                         attrs)
    with open(os.path.join(base, f"cmr_{sensor}_{int(year):04d}.json"),
              "w") as fh:
        json.dump(cmr_json(recs), fh)
    return recs

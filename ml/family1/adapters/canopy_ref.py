"""The two FINE canopy-height maps, catalogued: ETH 10 m and Meta/WRI 1 m
(family 1.0.tf, tier T, E-082 wave 6).

PLAIN ENGLISH. How tall the trees are is the single most useful number about a
forest: height times area is roughly carbon, and height is what a fire, a
clearance or a drought changes. Two groups have mapped it for the whole planet
at a resolution far finer than anything else in these families — ETH Zurich at
10 m from Sentinel-2 imagery fitted to GEDI's laser footprints (Lang, Jetz,
Schindler and Wegner 2023), and Meta together with the World Resources
Institute at about 1 m from aerial photography (Tolan et al. 2024). Together
they are tens of terabytes of pixels.

THIS STORE HOLDS NO PIXELS. It is the LIST of those tiles — where each one is,
how large it is, which product it belongs to, and the URL a reader fetches it
from. `canopy30` (GLAD's 30 m map) is the one canopy-height map these families
copy; these two are references, for the tile codec of E-078c to read in place.

SOURCES, EVERY NUMBER BELOW MEASURED FROM THIS SANDBOX ON 2026-09-20.

  ETH Global Canopy Height 10 m 2020 (`ETH-10m`, qc `ETH-v1`)
    The product's own TILE INDEX is a public page —
    https://langnico.github.io/globalcanopyheight/assets/tile_index.html
    (HTTP 200, 2,608,985 bytes) — and it embeds the authoritative tile list as
    a GeoJSON FeatureCollection inside its Leaflet map: **2,651 features**, all
    exactly 3 x 3 degrees, bbox -180..180 and -60..84, 48 distinct latitudes
    and 120 distinct longitudes, each carrying `tile_name`
    (`ETH_GlobalCanopyHeight_10m_2020_N00E006_Map.tif`) and `location`
    (`3deg_cogs/<tile_name>`). That page is the listing this adapter reads; it
    is not a scrape of a directory but the producer's own index of the
    product. THE DOWNLOAD HOST answered 429 to a listing attempt on
    2026-09-20 and answered normally twenty seconds later: a HEAD on
    `https://libdrive.ethz.ch/index.php/s/cO8or7iOe5dT2Rt/download
     ?path=%2F3deg_cogs&files=ETH_GlobalCanopyHeight_10m_2020_N00E006_Map.tif`
    returned **HTTP 200, Content-Length 10,969,970**, with the matching
    `Content-Disposition` filename. So the tile URLs are verified, not guessed,
    and the adapter keeps a delay between requests to that host.
    Each tile has a sibling `_Map_SD.tif` — the per-pixel standard deviation of
    the height estimate — and both are in the sidecar.

  Meta/WRI canopy height, two versions, on the anonymous AWS Open Data bucket
  `dataforgood-fb-data` (`META-1m`, qc `META-v1` and `META-v2`)
    v1  `forests/v1/alsgedi_global_v6_float/` — **56,145** level-9 web-mercator
        quadkey COGs under `chm/`, confirmed by a full paged ListObjectsV2
        (56,147 keys including the two prefix markers) AND by the bucket's own
        `tiles.geojson` (15,167,629 bytes, 56,145 features, one property
        `tile`). A HEAD on `chm/001311332.tif` returned HTTP 200,
        Content-Length 21,171,908, `Accept-Ranges: bytes`.
    v2  `forests/v2/global/dinov3_global_chm_v2_ml3/` — **213,109** level-10
        quadkey COGs, from the bucket's own transfer manifest
        `forests/v2/aws_google_chmv2_global_transfer.tsv` (a `TsvHttpData-1.0`
        list, 22,802,679 bytes, 213,110 lines, every one a URL under that one
        prefix). A HEAD on `chm/0013113321.tif` returned HTTP 200,
        Content-Length 1,977,131. The bucket's own README for the derived
        10 m product states the source is "level-10 quadkey tiles at ~1.19 m"
        and counts the same 213,109.

  THE YEAR IS THE PRODUCT'S REFERENCE EPOCH, AND THE REAL PER-TILE DATES ARE
  AN ASSET. ETH's map is for the year 2020 (the producer's own page). The two
  Meta products are COMPOSITES of aerial photography over a decade: sampling
  their per-tile `metadata/<quadkey>.geojson` sidecars measured 2009-09-13 to
  2020-06-05 over 10 random v1 tiles (159 source polygons) and 2010-06-27 to
  2020-07-31 over 30 random v2 tiles (177 polygons). One acquisition date per
  ROW would therefore be a fiction, and one GET per tile for the real ones
  would be 270,000 requests for a store whose point is that it is cheap. So a
  row's `time_s` is the collection's declared reference epoch — the last
  second of its reference year — the sidecar names each tile's own
  `metadata/<quadkey>.geojson` so a consumer can read the true dates, and
  `index` SAMPLES those sidecars every build and refuses if what they say
  falls outside the declared span.

WHAT A ROW IS.
  time_s    the last second of the collection's reference year (above).
  lat, lon  the footprint centre on the sphere.
  platform  platform_hash of the globally unique tile identifier
            (`ETH_GlobalCanopyHeight_10m_2020_N00E006`, `META_CHM_v1_023013213`,
            `META_CHM_v2_0013113321`).
  values    cloud   NaN — an annual composite publishes no scene cloud fraction
            valid   NaN — none is published
            angle   NaN — none is published
            log2_area  log2 of the tile's area on the sphere in km2
            sensor  1 = ETH-10m, 2 = META-1m
  qc        the producer's product version: 1 = ETH-v1, 2 = META-v1,
            3 = META-v2.

LICENCE. Both products are CC BY 4.0 and both are public. The rows are our own
metadata (CC0); the pixels they point at stay with their producers.
"""
import json
import math
import re

from family1.adapters import _stac as st

# ------------------------------------------------------------------- ETH ---
ETH_TILE_INDEX = ("https://langnico.github.io/globalcanopyheight/assets/"
                  "tile_index.html")
ETH_PAGE = "https://langnico.github.io/globalcanopyheight/"
ETH_DOI = "10.3929/ethz-b-000609802"
ETH_SHARE = "https://libdrive.ethz.ch/index.php/s/cO8or7iOe5dT2Rt/download"
ETH_DIR = "3deg_cogs"
ETH_YEAR = 2020
ETH_TILE_DEG = 3.0
ETH_TILES_EXPECTED = 2651

# ------------------------------------------------------------------ Meta ---
FB_BUCKET = "https://dataforgood-fb-data.s3.amazonaws.com"
META_V1_PREFIX = "forests/v1/alsgedi_global_v6_float"
META_V2_PREFIX = "forests/v2/global/dinov3_global_chm_v2_ml3"
META_V2_MANIFEST = f"{FB_BUCKET}/forests/v2/aws_google_chmv2_global_transfer.tsv"
META_V1_TILES = f"{FB_BUCKET}/{META_V1_PREFIX}/tiles.geojson"
META_YEAR = 2020
META_V1_EXPECTED = 56145
META_V2_EXPECTED = 213109
# The acquisition spans MEASURED from the products' own per-tile sidecars on
# 2026-09-20. `index` samples them again every build and refuses if a sampled
# date falls outside — the reference epoch below is a claim about the product,
# and a claim this store makes is one it re-checks.
META_ACQ_SPAN = {"META-v1": ("2009-01-01", "2020-12-31"),
                 "META-v2": ("2009-01-01", "2020-12-31")}
META_SAMPLE = 6

SENSOR_TABLE = {1: "ETH-10m", 2: "META-1m"}
QC_TABLE = {1: "ETH-v1", 2: "META-v1", 3: "META-v2"}

# (name, sensor text, qc text, reference year, native metres)
COLLECTIONS = (
    ("eth10", "ETH-10m", "ETH-v1", ETH_YEAR, 10.0),
    ("meta1_v1", "META-1m", "META-v1", META_YEAR, 1.0),
    ("meta1_v2", "META-1m", "META-v2", META_YEAR, 1.19),
)
RECORD_FIRST = (min(c[3] for c in COLLECTIONS), 1)
RECORD_LAST = (max(c[3] for c in COLLECTIONS), 12)


# ============================================================== quadkeys ====
def quadkey_bbox(qk):
    """A web-mercator quadkey -> (west, south, east, north) in degrees.

    A quadkey IS the tile's extent: each character picks one of the four
    children, so the footprint is arithmetic and no listing is needed for it.
    """
    q = str(qk)
    z = len(q)
    if not z or any(ch not in "0123" for ch in q):
        raise st.Refusal(f"{qk!r} is not a web-mercator quadkey")
    x = y = 0
    for ch in q:
        x <<= 1
        y <<= 1
        d = int(ch)
        if d & 1:
            x |= 1
        if d & 2:
            y |= 1
    n = float(1 << z)
    west = x / n * 360.0 - 180.0
    east = (x + 1) / n * 360.0 - 180.0
    north = math.degrees(math.atan(math.sinh(math.pi * (1 - 2 * y / n))))
    south = math.degrees(math.atan(math.sinh(math.pi * (1 - 2 * (y + 1) / n))))
    return west, south, east, north


def bbox_ring(west, south, east, north):
    """A rectangle as a closed (lat, lon) pair of lists for the sphere maths."""
    lats = [south, south, north, north, south]
    lons = [west, east, east, west, west]
    return lats, lons


def eth_tile_bbox(name):
    """`..._2020_N00E006_Map.tif` -> (west, south, east, north).

    The name carries the tile's SOUTH-WEST corner, which is how every tile in
    the product's own index reads; the index's bboxes are checked against this
    at listing time, so a name and a geometry that disagree is a refusal.
    """
    m = re.search(r"_([NS])(\d{2})([EW])(\d{3})_", str(name))
    if not m:
        raise st.Refusal(f"{name!r} carries no NNSSEEE corner token")
    lat = int(m.group(2)) * (1 if m.group(1) == "N" else -1)
    lon = int(m.group(4)) * (1 if m.group(3) == "E" else -1)
    return float(lon), float(lat), float(lon) + ETH_TILE_DEG, \
        float(lat) + ETH_TILE_DEG


def eth_tile_id(name):
    """The row's identifier: the tile name without `_Map.tif`."""
    return re.sub(r"_Map\.tif$", "", str(name))


def eth_asset_url(name):
    return f"{ETH_SHARE}?path=%2F{ETH_DIR}&files={name}"


# ============================================== reading the producers' lists =
def eth_feature_collection(text):
    """The GeoJSON the ETH tile-index page hands its Leaflet map.

    The page is a folium export: the features arrive as one
    `<id>_add({...FeatureCollection...})` call. Slicing that argument out is
    the whole parse — the JSON inside is the producer's own file, not
    something reconstructed from the HTML.
    """
    i = text.find('_add({"bbox"')
    if i < 0:
        i = text.find("_add({")
    if i < 0:
        raise st.Refusal(
            f"{ETH_TILE_INDEX}: no GeoJSON payload in the page "
            f"({len(text)} bytes). The product's tile index has changed "
            f"shape; a tile list invented here instead would be a guess.")
    i = text.index("({", i) + 1
    key = '"type": "FeatureCollection"}'
    j = text.find(key, i)
    if j < 0:
        raise st.Refusal(f"{ETH_TILE_INDEX}: the GeoJSON payload does not "
                         f"close with a FeatureCollection")
    try:
        return json.loads(text[i:j + len(key)])
    except ValueError as e:
        raise st.Refusal(f"{ETH_TILE_INDEX}: the payload is not JSON "
                         f"({e})") from None


def manifest_quadkeys(text, prefix):
    """A `TsvHttpData-1.0` manifest -> the quadkeys under `prefix`.

    Google's transfer-manifest format: a header line, then one URL a line
    (optionally followed by a size and an MD5, tab separated). A URL that
    leaves the declared prefix is a REFUSAL, not a skipped line — a manifest
    pointing somewhere else is a moved product, not a stray row.
    """
    lines = str(text).splitlines()
    if not lines or not lines[0].startswith("TsvHttpData"):
        raise st.Refusal(f"{META_V2_MANIFEST}: the first line is "
                         f"{lines[0][:60]!r}, not a TsvHttpData header")
    want = f"{FB_BUCKET}/{prefix}/chm/"
    out, seen = [], set()
    for ln in lines[1:]:
        if not ln.strip():
            continue
        url = ln.split("\t")[0].strip()
        if not url.startswith(want) or not url.endswith(".tif"):
            raise st.Refusal(
                f"{META_V2_MANIFEST}: the manifest lists {url[:110]!r}, "
                f"outside the declared prefix {want!r} — the product has "
                f"moved, and listing it anyway would catalogue the wrong "
                f"object")
        qk = url[len(want):-len(".tif")]
        if qk in seen:
            continue
        seen.add(qk)
        out.append(qk)
    if not out:
        raise st.Refusal(f"{META_V2_MANIFEST}: no tile at all in "
                         f"{len(lines)} line(s) — a broken manifest, not an "
                         f"empty product (the 2026-09-14 rule)")
    return out


# =============================================================== the store ==
class CanopyRefCatalogue(st.CatalogueAdapter):
    store = "canopy_ref"
    title = ("Fine canopy-height tile catalogue: ETH Zurich 10 m 2020 and "
             "Meta/WRI ~1 m (v1 and v2), one row per tile")
    licence = {
        "name": ("catalogue rows: our own metadata, CC0. The maps they point "
                 "at are CC BY 4.0 — ETH Zurich's Global Canopy Height 10 m "
                 "2020 and Meta/WRI's canopy height model"),
        "redistribution": "yes", "derived_works": "free",
        "attribution": (
            "Lang, N., W. Jetz, K. Schindler and J. D. Wegner (2023). A "
            "high-resolution canopy height model of the Earth. Nature Ecology "
            f"& Evolution 7, 1778-1789. Data: https://doi.org/{ETH_DOI} "
            "(CC BY 4.0). | Tolan, J. et al. (2024). Very high resolution "
            "canopy height maps from RGB imagery using self-supervised "
            "vision transformer and convolutional decoder trained on aerial "
            "lidar. Remote Sensing of Environment 300, 113888. Data: Meta / "
            "World Resources Institute, s3://dataforgood-fb-data/forests/ "
            "(CC BY 4.0). | Tile list from each producer's own published "
            "index.")}
    first_year = min(c[3] for c in COLLECTIONS)
    # a 3-degree ETH tile is ~334 km at the equator and a level-9 quadkey
    # ~78 km; the store's nominal footprint is the smaller of the two, since
    # the finer product is the one a reader comes here for
    footprint_km = 78.0
    scene_seconds = 365.0 * 86400.0        # an annual map, not a scene
    SENSOR_TABLE = SENSOR_TABLE
    QC_TABLE = QC_TABLE
    credentials = ()
    WORKERS = 2                 # two producers, two hosts; never more
    qc_policy = (
        "qc is the PRODUCT VERSION: 1 = ETH-v1 (ETH Zurich Global Canopy "
        "Height 10 m 2020), 2 = META-v1 (Meta/WRI alsgedi_global_v6_float, "
        "level-9 quadkeys), 3 = META-v2 (dinov3_global_chm_v2_ml3, level-10 "
        "quadkeys). The two Meta versions are separate products over the same "
        "ground, so a place appears once per version -- that is a "
        "reprocessing, not a duplicate, and a consumer selects on qc. "
        "`cloud`, `valid` and `angle` are NaN for every row: an annual "
        "composite publishes no scene cloud fraction, no valid fraction and "
        "no view angle, and a number invented here would be worse than the "
        "NaN. `time_s` is the COLLECTION'S REFERENCE EPOCH (the last second "
        "of its reference year), never a per-tile acquisition time: the Meta "
        "products composite aerial photography measured here as spanning "
        "2009-09-13 to 2020-07-31, so one date a row would be a fiction. The "
        "true per-tile dates live in each tile's own "
        "`metadata/<quadkey>.geojson`, which the sidecar names, and `index` "
        "samples them every build and refuses if they leave the declared "
        "span. A value outside a channel's bounds is NaN and counted "
        "(`out_of_bounds`).")
    sources = (ETH_TILE_INDEX, ETH_PAGE, META_V1_TILES, META_V2_MANIFEST,
               f"{FB_BUCKET}/{META_V1_PREFIX}/chm/<quadkey>.tif",
               f"{FB_BUCKET}/{META_V2_PREFIX}/chm/<quadkey>.tif")
    verified = (
        "2026-09-20 from the sandbox, anonymously. ETH: the tile-index page "
        "answered HTTP 200 with 2,608,985 bytes and its embedded "
        "FeatureCollection holds 2,651 features, every one exactly 3 x 3 "
        "degrees, over bbox [-180, -60, 180, 84] (48 latitudes x 120 "
        "longitudes), each with `tile_name` and `location` under "
        "`3deg_cogs/`; a HEAD on the first tile's libdrive URL returned 200 "
        "with Content-Length 10,969,970 and the matching Content-Disposition "
        "(an earlier attempt that day got 429, so the adapter spaces its "
        "requests to that host). Meta: an anonymous ListObjectsV2 on "
        "dataforgood-fb-data walked `forests/v1/alsgedi_global_v6_float/chm/` "
        "in 57 pages to 56,147 keys, the bucket's own tiles.geojson "
        "(15,167,629 bytes) holds 56,145 features with one property `tile`, "
        "and a HEAD on chm/001311332.tif returned 200 with Content-Length "
        "21,171,908 and Accept-Ranges; the v2 transfer manifest "
        "(22,802,679 bytes, TsvHttpData-1.0) lists 213,109 level-10 quadkeys, "
        "all under forests/v2/global/dinov3_global_chm_v2_ml3/chm/, and a "
        "HEAD on chm/0013113321.tif returned 200 with Content-Length "
        "1,977,131. The acquisition spans were sampled from the products' own "
        "per-tile metadata sidecars: 2009-09-13..2020-06-05 over 10 random v1 "
        "tiles (159 source polygons) and 2010-06-27..2020-07-31 over 30 "
        "random v2 tiles (177 polygons).")
    notes = (
        "No pixels are copied; this is the tile LIST of two canopy-height "
        "maps that are tens of terabytes together. Every footprint is the "
        "producer's own: ETH's from the GeoJSON its published tile-index page "
        "carries, Meta v1's from the bucket's own tiles.geojson, Meta v2's "
        "from the quadkey itself (a quadkey IS an extent, so no listing is "
        "needed for the geometry) with the tile set taken from the bucket's "
        "own transfer manifest. `canopy30` is the one canopy map these "
        "families copy the pixels of; these two are references for the tile "
        "codec to read in place.")
    smoke_window = ("2020-12-01", "2020-12-31")
    smoke_probe_month = "2020-12"

    def record_first(self):
        return RECORD_FIRST

    def record_last(self):
        return RECORD_LAST

    # -- the plan ----------------------------------------------------------
    def plan(self, ctx, year, month=None):
        """One window per collection whose reference year is `year`.

        There is no time axis inside a collection — every tile of an annual
        map carries the same epoch — so a "window" here is a collection, and
        the year (and the month, for the probe) selects which ones are listed
        at all.
        """
        out = []
        for (name, sensor, qc, ry, native) in COLLECTIONS:
            if int(ry) != int(year):
                continue
            if month is not None and int(month) != 12:
                continue                   # the epoch is 31 December
            out.append((name, sensor, qc, ry, native))
        return out

    # -- the listing -------------------------------------------------------
    def scenes(self, ctx, window):
        name, sensor, qc, ry, native = window
        fet = st.Fetcher(ctx, self.store)
        counts = {f"collection_{name}": 1}
        t = st.seconds_of(f"{ry:04d}-12-31T23:59:59Z")
        scode = self.sensor_code(sensor, counts)
        qcode = self.qc_code(qc, counts)
        if name == "eth10":
            rows = self._eth(fet, counts)
        elif name == "meta1_v1":
            rows = self._meta_v1(fet, counts)
        else:
            rows = self._meta_v2(fet, counts)
        out = []
        for (tile_id, lats, lons, base, assets) in rows:
            clat, clon, area = st.sphere_centre_area(lats, lons)
            out.append(st.Scene(
                t, clat, clon,
                [float("nan"), float("nan"), float("nan"),
                 st.area_channel(area, counts), float(scode)],
                tile_id, name, base, assets, qcode))
        counts[f"tiles_{name}"] = len(out)
        return out, counts

    # -- ETH ---------------------------------------------------------------
    def _eth(self, fet, counts):
        text = st._text(fet, ETH_TILE_INDEX)
        fc = eth_feature_collection(text)
        feats = fc.get("features") or []
        if not feats:
            raise st.Refusal(f"{ETH_TILE_INDEX}: the tile index lists ZERO "
                             f"tiles — a broken page, not an empty product")
        counts["eth_features"] = len(feats)
        out = []
        for f in feats:
            p = f.get("properties") or {}
            nm = p.get("tile_name")
            if not nm:
                counts["eth_features_without_name"] = \
                    counts.get("eth_features_without_name", 0) + 1
                continue
            w, s, e, n = eth_tile_bbox(nm)
            bb = f.get("bbox")
            if bb and max(abs(float(bb[0]) - w), abs(float(bb[1]) - s),
                          abs(float(bb[2]) - e), abs(float(bb[3]) - n)) > 1e-6:
                raise st.Refusal(
                    f"{nm}: the index's bbox {list(bb)} and the corner in the "
                    f"tile's own name {[w, s, e, n]} disagree — the naming "
                    f"convention has changed and a footprint derived from "
                    f"either alone would be wrong")
            lats, lons = bbox_ring(w, s, e, n)
            sd = re.sub(r"_Map\.tif$", "_Map_SD.tif", nm)
            out.append((eth_tile_id(nm), lats, lons, ETH_SHARE,
                        f"{eth_asset_url(nm)} {eth_asset_url(sd)}"))
        return out

    # -- Meta --------------------------------------------------------------
    def _meta_v1(self, fet, counts):
        d, _h = fet.get(META_V1_TILES)
        feats = (d or {}).get("features") or []
        if not feats:
            raise st.Refusal(f"{META_V1_TILES}: ZERO features — a broken "
                             f"listing, not an empty product")
        counts["meta_v1_features"] = len(feats)
        base = f"{FB_BUCKET}/{META_V1_PREFIX}"
        out = []
        for f in feats:
            qk = str((f.get("properties") or {}).get("tile") or "")
            if not qk:
                counts["meta_v1_features_without_tile"] = \
                    counts.get("meta_v1_features_without_tile", 0) + 1
                continue
            # THE QUADKEY IS THE FALLBACK, and it is exact: a quadkey names a
            # web-mercator tile, so a feature whose geometry is missing loses
            # nothing. A geometry that is PRESENT and unreadable is a refusal,
            # because then the two disagree about what the tile is.
            geom = f.get("geometry")
            if geom:
                lats, lons = st.ring_from_geojson(geom)[0]
            else:
                lats, lons = bbox_ring(*quadkey_bbox(qk))
            out.append((f"META_CHM_v1_{qk}", lats, lons, base,
                        f"{base}/chm/{qk}.tif {base}/msk/{qk}.tif "
                        f"{base}/metadata/{qk}.geojson"))
        return out

    def _meta_v2(self, fet, counts):
        text = st._text(fet, META_V2_MANIFEST)
        qks = manifest_quadkeys(text, META_V2_PREFIX)
        counts["meta_v2_manifest_tiles"] = len(qks)
        base = f"{FB_BUCKET}/{META_V2_PREFIX}"
        out = []
        for qk in qks:
            lats, lons = bbox_ring(*quadkey_bbox(qk))
            out.append((f"META_CHM_v2_{qk}", lats, lons, base,
                        f"{base}/chm/{qk}.tif "
                        f"{base}/metadata/{qk}.geojson"))
        return out

    # -- the index ---------------------------------------------------------
    def index(self, ctx):
        fet = st.Fetcher(ctx, self.store)
        out = {"dataset": "fine canopy-height tile catalogue (ETH 10 m 2020; "
                          "Meta/WRI ~1 m v1 and v2)",
               "url": ETH_TILE_INDEX, "collections": {},
               "years": [int(v) for v in ctx.years],
               "reference_epoch_rule": (
                   "a row's time_s is the last second of its collection's "
                   "reference year; the per-tile acquisition dates live in "
                   "each tile's own metadata sidecar, which the asset table "
                   "names"),
               "acq_span_declared": dict(META_ACQ_SPAN)}
        total = 0
        for (name, sensor, qc, ry, native) in COLLECTIONS:
            c = {}
            if name == "eth10":
                rows = self._eth(fet, c)
                expect = ETH_TILES_EXPECTED
            elif name == "meta1_v1":
                rows = self._meta_v1(fet, c)
                expect = META_V1_EXPECTED
            else:
                rows = self._meta_v2(fet, c)
                expect = META_V2_EXPECTED
            total += len(rows)
            first = rows[0] if rows else None
            out["collections"][name] = {
                "sensor": sensor, "qc": qc, "reference_year": ry,
                "native_metres": native,
                "tiles_listed": len(rows),
                "tiles_measured_2026_09_20": expect,
                "tiles_differ_from_measured": len(rows) - expect,
                "first_tile": (None if first is None else
                               {"id": first[0], "assets": first[4]}),
                "counts": c}
        if not total:
            raise st.Refusal(f"{self.store}: ZERO tiles across all three "
                             f"collections — a broken listing, not a "
                             f"measurement (the 2026-09-14 rule)")
        out["tiles_total"] = total
        out["acq_sample"] = self._sample_acq(fet, out)
        out["sensor_table"] = {str(k): v for k, v in SENSOR_TABLE.items()}
        out["qc_table"] = {str(k): v for k, v in QC_TABLE.items()}
        return out

    def _sample_acq(self, fet, plan):
        """Re-measure the Meta acquisition span from the products' sidecars.

        The reference epoch is a CLAIM about the product, so it is re-checked
        every build rather than trusted from the day it was measured. A date
        outside the declared span is a refusal: the product has been rebuilt
        from new photography and the epoch has to be revisited.
        """
        out = {}
        for key, prefix in (("META-v1", META_V1_PREFIX),
                            ("META-v2", META_V2_PREFIX)):
            name = "meta1_v1" if key == "META-v1" else "meta1_v2"
            first = (plan["collections"].get(name) or {}).get("first_tile")
            if not first:
                continue
            qk = str(first["id"]).rsplit("_", 1)[-1]
            url = f"{FB_BUCKET}/{prefix}/metadata/{qk}.geojson"
            try:
                d, _h = fet.get(url)
            except st.Refusal:
                out[key] = {"url": url, "read": False}
                continue
            dates = sorted(str((f.get("properties") or {}).get("acq_date"))
                           for f in (d or {}).get("features") or []
                           if (f.get("properties") or {}).get("acq_date"))
            out[key] = {"url": url, "read": True, "polygons": len(dates),
                        "first": dates[0] if dates else None,
                        "last": dates[-1] if dates else None,
                        "declared": list(META_ACQ_SPAN[key])}
            lo, hi = META_ACQ_SPAN[key]
            if dates and (dates[0] < lo or dates[-1] > hi):
                raise st.Refusal(
                    f"{url}: acquisition dates {dates[0]}..{dates[-1]} fall "
                    f"outside the declared span {lo}..{hi} for {key}. The "
                    f"product has been rebuilt from newer photography, so the "
                    f"reference epoch this store files its rows under has to "
                    f"be revisited rather than quietly kept.")
        return out

    def smoke_sources(self, root, d_lo, d_hi, seed=20260920):
        from family1.adapters import _cat_smoke as cs
        return canopy_smoke(cs, self, root, d_lo, d_hi)


# =================================================================== smoke ==
# Three ETH tiles (one across the antimeridian), three Meta v1 quadkeys (one
# whose geometry the listing omits, so the quadkey fallback runs) and four
# Meta v2 quadkeys from a manifest with a blank line in it. Ten rows.
SMOKE_ETH = ("ETH_GlobalCanopyHeight_10m_2020_N00E006_Map.tif",
             "ETH_GlobalCanopyHeight_10m_2020_S60W030_Map.tif",
             "ETH_GlobalCanopyHeight_10m_2020_N48E177_Map.tif")
SMOKE_V1 = ("023013213", "130122211", "310111000")
SMOKE_V1_NO_GEOM = "310111000"
SMOKE_V2 = ("0013113321", "0013113323", "0013131013", "3330000000")
SMOKE_ACQ = {"META-v1": ("2011-07-18", "2019-08-11"),
             "META-v2": ("2018-08-11", "2019-08-09")}


def canopy_smoke(cs, adapter, root, d_lo, d_hi):
    """A canned ETH page, a canned tiles.geojson and a canned TSV manifest."""
    def eth_page():
        feats = []
        for nm in SMOKE_ETH:
            w, s, e, n = eth_tile_bbox(nm)
            feats.append({"bbox": [w, s, e, n],
                          "geometry": {"type": "Polygon", "coordinates": [
                              [[w, n], [e, n], [e, s], [w, s], [w, n]]]},
                          "properties": {"tile_name": nm,
                                         "location": f"{ETH_DIR}/{nm}"},
                          "type": "Feature"})
        fc = {"bbox": [-180.0, -60.0, 180.0, 84.0], "features": feats,
              "type": "FeatureCollection"}
        return ("<html><script>geo_json_x_add(" +
                json.dumps(fc) + ");</script></html>")

    def v1_geojson():
        feats = []
        for qk in SMOKE_V1:
            w, s, e, n = quadkey_bbox(qk)
            geom = None if qk == SMOKE_V1_NO_GEOM else {
                "type": "Polygon",
                "coordinates": [[[w, n], [e, n], [e, s], [w, s], [w, n]]]}
            feats.append({"type": "Feature", "properties": {"tile": qk},
                          "geometry": geom})
        return {"type": "FeatureCollection", "features": feats}

    def v2_manifest():
        lines = ["TsvHttpData-1.0"]
        for qk in SMOKE_V2:
            lines.append(f"{FB_BUCKET}/{META_V2_PREFIX}/chm/{qk}.tif")
        lines.insert(2, "")                 # a blank line the parser skips
        return "\n".join(lines) + "\n"

    def acq(key, qk):
        lo, hi = SMOKE_ACQ[key]
        return {"type": "FeatureCollection", "features": [
            {"type": "Feature", "properties": {"acq_date": lo},
             "geometry": None},
            {"type": "Feature", "properties": {"acq_date": hi},
             "geometry": None}]}

    def serve(method, url, body, headers):
        if url == META_V1_TILES:
            return v1_geojson(), {}
        m = re.match(re.escape(FB_BUCKET) + r"/(.+)/metadata/(\d+)\.geojson$",
                     url)
        if m:
            key = "META-v1" if META_V1_PREFIX in url else "META-v2"
            return acq(key, m.group(2)), {}
        raise AssertionError(f"the smoke's fake producer got {url!r}")

    def text_serve(url):
        if url == ETH_TILE_INDEX:
            return eth_page()
        if url == META_V2_MANIFEST:
            return v2_manifest()
        raise AssertionError(f"the smoke's fake producer got text {url!r}")

    return cs.record(adapter, d_lo, d_hi, serve, root, text_serve=text_serve)


ADAPTER = CanopyRefCatalogue

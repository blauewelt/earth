"""ESA's BIOMASS mission, catalogued: the first P-band radar in orbit
(family 1.0.tf, tier T, E-082 wave 6).

PLAIN ENGLISH. Every radar that has flown before BIOMASS bounces off the
outside of a forest. BIOMASS uses P-band — a 70 cm wavelength, five times
longer than the L-band of PALSAR and twelve times longer than Sentinel-1's
C-band — and at that wavelength the radar goes THROUGH the canopy and returns
from the trunks and the large branches, which is where almost all of a
forest's carbon is. It is the only instrument that measures woody biomass
directly rather than inferring it from height or greenness, and it is why ESA
flew it: the size of the land carbon sink is the largest single unknown in the
carbon budget. It launched on 2025-04-29 and is the fifth Earth Explorer.

THIS STORE HOLDS NO PIXELS: one row per published product, with when, where,
how large, which processing level and the URL a reader fetches it from.

**ARE THE PRODUCTS PUBLICLY LISTED? YES — THROUGH FedEO, MEASURED TODAY.**
Four routes were tried on 2026-09-20 from this sandbox and only one answered;
all four are recorded here because the three that failed are what a future
session would otherwise re-try:

  https://eocat.esa.int/eo-catalogue/collections.json
      **connection reset by peer** (curl 35), at 45 s and again at 120 s; the
      bare host `https://eocat.esa.int/` reset identically. The proxy
      established its tunnel first, so this is the origin closing the
      connection, not a local failure.
  https://biomass-pdgs.eo.esa.int/
      **HTTP 502 from the proxy on the CONNECT** — the host did not accept a
      connection at all.
  https://catalogue.dataspace.copernicus.eu/odata/v1/Products
      answers 200, and `$filter=Collection/Name eq 'BIOMASS'` answers
      **HTTP 400 `{"detail":{"message":"Invalid value: BIOMASS"}}`** while
      `startswith(Name,'BIO_')` answers 200 with an EMPTY value list. BIOMASS
      is not a Copernicus Data Space collection; its `/stac/collections`
      listing (200) names only the CCM, CLMS and Sentinel families.
  https://fedeo.ceos.org/collections?platform=Biomass
      **HTTP 200**, 27 collections, among them `BiomassLevel0`,
      `BiomassLevel1a`, `BiomassLevel1b`, `BiomassLevel1c`, `BiomassLevel2a`,
      `BiomassLevel2b`, each with an `...IOC` twin for the in-orbit
      commissioning phase, plus auxiliary and Cal/Val sets. FedEO is the CEOS
      federated EO catalogue and it is ESA's own public front door for this
      mission. `GET /collections/<id>/items` answers OGC API Features
      GeoJSON with `numberMatched`, `numberReturned`, `datetime=` filtering
      and `marker=` cursor paging, and each feature carries a Polygon
      `geometry`, a `properties.date` interval, `productInformation.size`,
      `creationDate`, `status` and `enclosure` links into ESA's MAAP store
      (`https://catalog.maap.eo.esa.int/data/biomass-pdgs-01/...`).

  ITEM COUNTS MEASURED 2026-09-20 (`items?limit=1`, `numberMatched`):
      BiomassLevel1a    417,521     BiomassLevel1aIOC   283,956
      BiomassLevel1b    313,329     BiomassLevel1bIOC   141,981
      BiomassLevel1c    126,785     BiomassLevel1cIOC     8,550
      BiomassLevel2a     25,792     BiomassLevel2aIOC       747
      BiomassLevel2b          0     BiomassLevel2bIOC         0
      (BiomassLevel0 holds 97,872 and is NOT catalogued: level 0 is
      instrument telemetry, not an observation anybody reads.)
  Paging was exercised: `limit=2` inside one day returned two ids, its `next`
  link returned two DIFFERENT ids and a different `next`, so the cursor
  advances rather than repeating.
  Like CMR, FedEO's `datetime=` matches an OVERLAP of the product's own
  interval, so a day's window returns products that START earlier; `_rows`
  drops those by their own start, exactly as `cat_hls` does.

WHAT A ROW IS.
  time_s    the START of the product's own `date` interval. An L1 slice spans
            a few seconds; an L2A product is a STACK and its interval spans
            weeks (measured: 2026-04-24T04:11:57Z to 2026-05-12T04:12:24Z),
            and the row carries the start of it.
  lat, lon  the footprint centre on the sphere, from the item's own polygon.
  platform  platform_hash of the product identifier
            (`BIO_S1_SCS__1M_20250702T221133_...`).
  values    cloud   NaN — radar does not measure cloud, and a number here
                    would be a category error rather than a missing value
            valid   NaN — none is published
            angle   NaN — no scene incidence angle is published
            log2_area  log2 of the footprint's area on the sphere in km2
            sensor  1 — BIOMASS / P-SAR; one instrument, one code
  qc        the PRODUCT: 1 L1A, 2 L1B, 3 L1C, 4 L2A, 5 L2B during the nominal
            mission; 11..15 the same levels during in-orbit commissioning.
            The two phases are separate collections over the same instrument,
            so a place can appear under both — that is a mission phase, not a
            duplicate, and a consumer selects on qc.

LICENCE. ESA's Earth Explorer products are open under the ESA Earth
Observation data policy, free of charge with attribution. The rows are our own
metadata (CC0); the pixels stay at ESA.
"""
import datetime as dt
import re
import urllib.parse

from family1.adapters import _stac as st

FEDEO = "https://fedeo.ceos.org"
FEDEO_COLLECTIONS = f"{FEDEO}/collections"
GEOJSON = "application/geo+json"
MAAP_DATA = "https://catalog.maap.eo.esa.int/data/biomass-pdgs-01"

LAUNCH = dt.date(2025, 4, 29)
RECORD_FIRST = (LAUNCH.year, LAUNCH.month)

# (FedEO collection id, qc text, measured numberMatched on 2026-09-20)
COLLECTIONS = (
    ("BiomassLevel1a", "L1A", 417521),
    ("BiomassLevel1aIOC", "L1A-IOC", 283956),
    ("BiomassLevel1b", "L1B", 313329),
    ("BiomassLevel1bIOC", "L1B-IOC", 141981),
    ("BiomassLevel1c", "L1C", 126785),
    ("BiomassLevel1cIOC", "L1C-IOC", 8550),
    ("BiomassLevel2a", "L2A", 25792),
    ("BiomassLevel2aIOC", "L2A-IOC", 747),
    # measured at 0 today and catalogued anyway: L2B is the mission's
    # above-ground-biomass product, the one this family actually wants, and a
    # collection that starts publishing must appear without a code change
    ("BiomassLevel2b", "L2B", 0),
    ("BiomassLevel2bIOC", "L2B-IOC", 0),
)
SENSOR_TABLE = {1: "BIOMASS/P-SAR"}
QC_TABLE = {1: "L1A", 2: "L1B", 3: "L1C", 4: "L2A", 5: "L2B",
            11: "L1A-IOC", 12: "L1B-IOC", 13: "L1C-IOC", 14: "L2A-IOC",
            15: "L2B-IOC"}

# THE FOUR ENDPOINTS AND WHAT EACH ANSWERED, 2026-09-20. Kept in the code and
# written into plan.json because three of them cost a session's time to
# re-discover, and a store that fails tomorrow needs to know which of them was
# ever the working one.
ACCESS_PROBES = (
    {"endpoint": "https://eocat.esa.int/eo-catalogue/collections.json",
     "answer": "connection reset by peer (curl 35) at 45 s and at 120 s",
     "usable": False},
    {"endpoint": "https://biomass-pdgs.eo.esa.int/",
     "answer": "HTTP 502 from the proxy on CONNECT — the host refused the "
               "connection",
     "usable": False},
    {"endpoint": "https://catalogue.dataspace.copernicus.eu/odata/v1/Products",
     "answer": "200, but `Collection/Name eq 'BIOMASS'` -> HTTP 400 "
               "\"Invalid value: BIOMASS\" and `startswith(Name,'BIO_')` -> "
               "200 with an empty value list; BIOMASS is not a CDSE "
               "collection",
     "usable": False},
    {"endpoint": f"{FEDEO_COLLECTIONS}?platform=Biomass",
     "answer": "200, 27 collections including BiomassLevel1a/1b/1c/2a/2b and "
               "their IOC twins; items answer OGC API Features GeoJSON with "
               "numberMatched, datetime= and marker= paging",
     "usable": True},
)

# MEASURED from a BiomassLevel1aIOC item's own bbox on 2026-09-20:
# 128.695..130.337 E at about 66.6 S is 72.6 km across, and 67.053..66.131 S
# is 102.6 km — a geometric mean of 86 km. The probe replaces it.
FOOTPRINT_KM = 86.0
# MEASURED: BIO_S1_SCS__1M_20250702T221133_20250702T221146 spans 13.6 s and
# BIO_S1_DGM__1S_20251121T011607_20251121T011628 spans 20.7 s.
SLICE_SECONDS = 20.0
PAGE = 500
MAX_PAGES = 40000


def qc_code_of(text):
    for k, v in QC_TABLE.items():
        if v == text:
            return k
    return st.QC_OTHER


def _next_href(d):
    for ln in (d or {}).get("links") or []:
        if str(ln.get("rel")) == "next" and ln.get("href"):
            return str(ln["href"])
    return None


def with_geojson(url):
    """FedEO's `next` link drops `httpAccept`, and without it the service
    answers Atom. Adding it back is what keeps the cursor on one format."""
    if "httpAccept=" in url:
        return url
    sep = "&" if "?" in url else "?"
    return url + sep + "httpAccept=" + urllib.parse.quote(GEOJSON, safe="")


def fedeo_items(fet, coll, params, counts, what):
    """Every item of one collection in one window, through `marker` paging.

    Refuses three ways, all of which have bitten another catalogue in this
    family: a page that returns nothing while the service says more are
    matched, a cursor that does not advance, and a page count past the point
    where a loop is the only explanation.
    """
    url = with_geojson(f"{FEDEO_COLLECTIONS}/{coll}/items?"
                       + urllib.parse.urlencode(params))
    got, matched, pages, seen_urls = [], None, 0, set()
    while url:
        if url in seen_urls:
            raise st.Refusal(f"{what}: FedEO's cursor returned to a page it "
                             f"had already served after {pages} page(s) — "
                             f"paging is not advancing")
        seen_urls.add(url)
        d, _h = fet.get(url)
        d = d or {}
        if matched is None:
            matched = d.get("numberMatched")
        feats = d.get("features") or []
        pages += 1
        got.extend(feats)
        if not feats:
            break
        if pages >= MAX_PAGES:
            raise st.Refusal(f"{what}: paged past {pages} pages — refusing "
                             f"to loop")
        nxt = _next_href(d)
        url = with_geojson(nxt) if nxt else None
    counts["fedeo_pages"] = counts.get("fedeo_pages", 0) + pages
    if matched is not None and len(got) < int(matched):
        # A SHORT LISTING IS A REFUSAL, not a small window: the producer's own
        # count is the only thing that can tell "nothing there" from "the
        # pager stopped early" (ml/CLAUDE.md, the 2026-09-14 rule).
        raise st.Refusal(
            f"{what}: FedEO says {matched} product(s) match and served "
            f"{len(got)} over {pages} page(s) — a truncated listing, and a "
            f"catalogue that silently kept the prefix would be missing "
            f"scenes nobody could see were missing")
    if matched is not None:
        counts["fedeo_matched"] = counts.get("fedeo_matched", 0) + int(matched)
    return got


class BiomassCatalogue(st.CatalogueAdapter):
    store = "cat_biomass_esa"
    title = ("ESA BIOMASS P-band radar product catalogue (L1A/L1B/L1C/L2A/L2B, "
             "nominal and in-orbit commissioning), one row per product")
    licence = {
        "name": ("catalogue rows: our own metadata, CC0. The products they "
                 "point at are ESA BIOMASS Earth Explorer data under the ESA "
                 "Earth Observation data policy — free of charge, with "
                 "attribution"),
        "redistribution": "attribution",
        "redistribution_confirmed": False,
        "derived_works": "free",
        "attribution": ("Contains modified ESA BIOMASS data (2025-2026), "
                        "processed by ESA. Product list from FedEO, the CEOS "
                        "federated Earth observation catalogue, "
                        "https://fedeo.ceos.org."),
        "terms": ("the ESA Earth Observation data policy makes Earth Explorer "
                  "products free of charge with attribution; the exact "
                  "redistribution clause for BIOMASS was NOT read from a "
                  "terms document here, so `redistribution_confirmed` is "
                  "False and a public publish needs "
                  "--allow-unconfirmed-licence until it is. Only the "
                  "catalogue ROWS are ours to publish either way.")}
    first_year = LAUNCH.year
    footprint_km = FOOTPRINT_KM
    scene_seconds = SLICE_SECONDS
    SENSOR_TABLE = SENSOR_TABLE
    QC_TABLE = QC_TABLE
    credentials = ()
    window_step = None            # one window per collection per month
    WORKERS = 4
    page_size = PAGE
    qc_policy = (
        "qc is the PRODUCT: 1 L1A, 2 L1B, 3 L1C, 4 L2A, 5 L2B in the nominal "
        "mission and 11..15 the same levels during in-orbit commissioning "
        "(the IOC collections). The two phases are separate FedEO collections "
        "over the same instrument, so one place can appear under both -- that "
        "is a mission phase, not a duplicate, and a consumer selects on qc. "
        "A level the table does not list is 255 and counted by name in "
        "`qc_unlisted`. `cloud` is NaN for every row and that is a statement "
        "rather than a gap: a radar does not measure cloud. `valid` and "
        "`angle` are NaN because ESA publishes neither per product. A value "
        "outside a channel's bounds becomes NaN and is counted "
        "(`out_of_bounds`), never clipped. Level 0 is telemetry and is not "
        "catalogued at all.")
    sources = (f"{FEDEO_COLLECTIONS}?platform=Biomass",
               f"{FEDEO_COLLECTIONS}/<collection>/items",
               MAAP_DATA + "/<collection>/<yyyy>/<mm>/<dd>/<product>")
    verified = (
        "2026-09-20 from the sandbox, anonymously. FOUR routes were tried: "
        "eocat.esa.int answered `connection reset by peer` at 45 s and again "
        "at 120 s; biomass-pdgs.eo.esa.int answered HTTP 502 on the proxy's "
        "CONNECT; the Copernicus Data Space OData service answered 200 but "
        "`Collection/Name eq 'BIOMASS'` returned HTTP 400 \"Invalid value: "
        "BIOMASS\" and `startswith(Name,'BIO_')` returned an empty list, so "
        "BIOMASS is not a CDSE collection; and FedEO "
        "(https://fedeo.ceos.org/collections?platform=Biomass) answered 200 "
        "with 27 collections. Item counts by `numberMatched`: 417,521 L1A, "
        "283,956 L1A-IOC, 313,329 L1B, 141,981 L1B-IOC, 126,785 L1C, 8,550 "
        "L1C-IOC, 25,792 L2A, 747 L2A-IOC, 0 L2B and 0 L2B-IOC (and 97,872 "
        "level 0, not catalogued). One item read in full: "
        "BIO_S1_SCS__1M_20250702T221133_20250702T221146_... with a Polygon "
        "geometry, bbox [128.69518, -67.05316, 130.337, -66.131226], a "
        "`date` interval, `productInformation.size` and five `enclosure` "
        "links under catalog.maap.eo.esa.int/data/biomass-pdgs-01/. Paging "
        "was exercised at limit=2: the `next` link served two different ids "
        "and a different cursor.")
    notes = (
        "BIOMASS launched 2025-04-29 and its products ARE publicly listed, "
        "through FedEO rather than through eocat.esa.int or the BIOMASS PDGS "
        "(both of which refused connections from here) or the Copernicus "
        "Data Space (which does not carry the mission). The four endpoints "
        "and their answers are in `ACCESS_PROBES` and in plan.json, so the "
        "three dead ones are not re-tried by hand. L2B -- the "
        "above-ground-biomass product this family actually wants -- matched "
        "ZERO items today and is catalogued anyway, so that the day it starts "
        "publishing needs no code change. FedEO's `datetime=` matches an "
        "OVERLAP, so a window returns products that started earlier and the "
        "framework drops them by their own start, exactly as cat_hls does.")
    smoke_window = ("2026-05-01", "2026-05-02")
    smoke_probe_month = "2026-05"

    def record_first(self):
        return RECORD_FIRST

    # -- the plan ----------------------------------------------------------
    def plan(self, ctx, year, month=None):
        out = []
        for (coll, qc, _n) in COLLECTIONS:
            for (a, b) in self.windows(ctx, year, month=month, step=None):
                out.append((coll, qc, a, b))
        return out

    # -- the listing -------------------------------------------------------
    def scenes(self, ctx, window):
        coll, qc, t0, t1 = window
        fet = st.Fetcher(ctx, self.store)
        counts = {f"window_{coll}": 1}
        params = [("limit", self.page_for(ctx, self.page_size)),
                  ("datetime", f"{st.utc_z(t0)}/{st.utc_z(t1)}")]
        feats = fedeo_items(fet, coll, params, counts,
                            f"{self.store} {coll} {t0:%Y-%m}")
        qcode = self.qc_code(qc, counts)
        scode = self.sensor_code("BIOMASS/P-SAR", counts)
        out = []
        for f in feats:
            sc = self.product(f, coll, qcode, scode, counts)
            if sc is not None:
                out.append(sc)
        counts[f"products_{coll}"] = len(out)
        return out, counts

    def product(self, f, coll, qcode, scode, counts):
        """One FedEO feature -> a `Scene`, or None with a named count."""
        pid = str(f.get("id") or (f.get("properties") or {}).get("title") or "")
        if not pid:
            counts["items_without_id"] = counts.get("items_without_id", 0) + 1
            return None
        props = f.get("properties") or {}
        when = props.get("date") or props.get("datetime")
        if not when:
            counts["items_without_date"] = \
                counts.get("items_without_date", 0) + 1
            return None
        # `date` is an INTERVAL, "<start>/<end>"; the row carries the start.
        start = str(when).split("/")[0]
        geom = f.get("geometry")
        if not geom:
            counts["items_without_geometry"] = \
                counts.get("items_without_geometry", 0) + 1
            return None
        clat, clon, area = st.centre_area(st.ring_from_geojson(geom))
        hrefs = [str(ln.get("href")) for ln in (f.get("links") or [])
                 if str(ln.get("rel")) == "enclosure" and ln.get("href")]
        if not hrefs:
            counts["items_without_enclosure"] = \
                counts.get("items_without_enclosure", 0) + 1
            return None
        # The `zipper` href is a SERVICE that re-packs the product, and it
        # lives in a different directory from the files themselves — the same
        # trap `cat_viirs` hit with OPeNDAP. A catalogue row points at the
        # files.
        base, tmpl = st.asset_template(hrefs, pid, f"{self.store} {pid}",
                                       drop_hosts=(f"{MAAP_DATA}/zipper",
                                                   "https://catalog.maap.eo."
                                                   "esa.int/data/zipper",))
        size = ((props.get("productInformation") or {}).get("size"))
        if size:
            counts["product_bytes"] = counts.get("product_bytes", 0) + int(size)
        return st.Scene(
            st.seconds_of(start), clat, clon,
            [float("nan"), float("nan"), float("nan"),
             st.area_channel(area, counts), float(scode)],
            pid, coll, base, tmpl, qcode)

    # -- the index ---------------------------------------------------------
    def index(self, ctx):
        fet = st.Fetcher(ctx, self.store)
        out = {"dataset": "ESA BIOMASS (P-band radar) product catalogue via "
                          "FedEO, the CEOS federated EO catalogue",
               "url": f"{FEDEO_COLLECTIONS}?platform=Biomass",
               "launch": str(LAUNCH),
               "years": [int(v) for v in ctx.years],
               "access_probes": [dict(p) for p in ACCESS_PROBES],
               "collections": {}}
        total = 0
        first = None
        for (coll, qc, measured) in COLLECTIONS:
            url = with_geojson(f"{FEDEO_COLLECTIONS}/{coll}/items?"
                               + urllib.parse.urlencode([("limit", 1)]))
            d, _h = fet.get(url)
            d = d or {}
            n = d.get("numberMatched")
            n = int(n) if n is not None else None
            total += n or 0
            feats = d.get("features") or []
            out["collections"][coll] = {
                "qc": qc, "qc_code": qc_code_of(qc),
                "products": n,
                "products_measured_2026_09_20": measured,
                "products_differ_from_measured": (None if n is None
                                                  else n - measured)}
            if first is None and feats:
                c0 = {}
                sc = self.product(feats[0], coll, self.qc_code(qc, c0),
                                  self.sensor_code("BIOMASS/P-SAR", c0), c0)
                if sc is not None:
                    first = {"collection": coll, "id": sc.stac_id,
                             "time_s": sc.t, "lat": sc.lat, "lon": sc.lon,
                             "log2_area": sc.values[3], "qc": sc.qc,
                             "asset_dir": sc.base_url,
                             "asset_template": sc.asset_set, "counts": c0}
        if not total:
            raise st.Refusal(
                f"{self.store}: FedEO counts ZERO products across all "
                f"{len(COLLECTIONS)} BIOMASS collections — a moved catalogue, "
                f"not an empty mission (the 2026-09-14 rule). The four "
                f"endpoints tried on 2026-09-20 and their answers are in this "
                f"adapter's ACCESS_PROBES.")
        out["products_total"] = total
        out["first_record"] = first
        out["sensor_table"] = {str(k): v for k, v in SENSOR_TABLE.items()}
        out["qc_table"] = {str(k): v for k, v in QC_TABLE.items()}
        return out

    def smoke_sources(self, root, d_lo, d_hi, seed=20260920):
        from family1.adapters import _cat_smoke as cs
        return biomass_smoke(cs, self, root, d_lo, d_hi)


# =================================================================== smoke ==
# Eight products over two days and four collections: two that must be paged
# (the smoke's page size is 3), one whose interval STARTS before the window
# (FedEO matches on overlap, so it is served and must be dropped by its own
# start), one across the antimeridian, one with no enclosure link at all, and
# one whose enclosure set includes the `zipper` service that must be dropped.
SMOKE_PRODUCTS = (
    # (collection, id, start, end, lat, lon, dlat, dlon, quirk)
    ("BiomassLevel1a", "BIO_S1_SCS__1M_20260501T010203_20260501T010223_T_"
     "G01_M03_C01_T001_F001_02_AAAAAA",
     "2026-05-01T01:02:03Z", "2026-05-01T01:02:23Z", -66.6, 129.5, 0.5, 0.8,
     None),
    ("BiomassLevel1a", "BIO_S1_SCS__1M_20260501T030405_20260501T030425_T_"
     "G01_M03_C01_T002_F002_02_BBBBBB",
     "2026-05-01T03:04:05Z", "2026-05-01T03:04:25Z", 17.4, 96.1, 0.7, 0.4,
     None),
    ("BiomassLevel1a", "BIO_S1_SCS__1M_20260501T050607_20260501T050627_T_"
     "G01_M03_C01_T003_F003_02_CCCCCC",
     "2026-05-01T05:06:07Z", "2026-05-01T05:06:27Z", 4.0, 179.3, 0.5, 1.0,
     None),
    ("BiomassLevel1a", "BIO_S1_SCS__1M_20260501T070809_20260501T070829_T_"
     "G01_M03_C01_T004_F004_02_DDDDDD",
     "2026-05-01T07:08:09Z", "2026-05-01T07:08:29Z", -12.0, -60.0, 0.5, 0.5,
     "zipper"),
    ("BiomassLevel1b", "BIO_S1_DGM__1S_20260501T090000_20260501T090020_T_"
     "G01_M03_C01_T005_F005_02_EEEEEE",
     "2026-05-01T09:00:00Z", "2026-05-01T09:00:20Z", 52.0, 13.0, 0.4, 0.6,
     None),
    ("BiomassLevel1b", "BIO_S1_DGM__1S_20260430T235000_20260501T000010_T_"
     "G01_M03_C01_T006_F006_02_FFFFFF",
     "2026-04-30T23:50:00Z", "2026-05-01T00:00:10Z", 1.0, 32.0, 0.4, 0.6,
     None),                                   # starts BEFORE the window
    ("BiomassLevel2a", "BIO_FP_GN__L2A_20260501T110000_20260512T110030_T_"
     "G01_M03_C___T018_F026_02_GGGGGG",
     "2026-05-01T11:00:00Z", "2026-05-12T11:00:30Z", -3.0, 23.0, 0.9, 0.9,
     None),                                   # an L2A stack: weeks, not seconds
    ("BiomassLevel2a", "BIO_FP_FH__L2A_20260501T130000_20260512T130030_T_"
     "G01_M03_C___T004_F191_02_HHHHHH",
     "2026-05-01T13:00:00Z", "2026-05-12T13:00:30Z", -6.0, 25.0, 0.9, 0.9,
     "no_enclosure"),
)


def biomass_smoke(cs, adapter, root, d_lo, d_hi):
    by_coll = {}
    for (coll, pid, t0, t1, lat, lon, dlat, dlon, quirk) in SMOKE_PRODUCTS:
        files = [f"{MAAP_DATA}/{coll}/2026/05/01/{pid}/{pid}.{ext}"
                 for ext in ("h5", "xml")]
        if quirk == "zipper":
            files.append(f"{MAAP_DATA}/zipper/{coll}/2026/05/01/{pid}.zip")
        links = [{"rel": "self", "href": f"{FEDEO}/x/{pid}"}]
        if quirk != "no_enclosure":
            links += [{"rel": "enclosure", "href": h} for h in files]
        by_coll.setdefault(coll, []).append({
            "type": "Feature", "id": pid,
            "properties": {"date": f"{t0}/{t1}", "title": pid,
                           "status": "ARCHIVED",
                           "productInformation": {"size": 37053}},
            "geometry": {"type": "Polygon",
                         "coordinates": cs.ring(lat, lon, dlat, dlon)},
            "bbox": [lon - dlon, lat - dlat, lon + dlon, lat + dlat],
            "links": links})

    def page(coll, limit, start):
        feats = by_coll.get(coll, [])
        got = feats[start:start + limit]
        nxt = start + limit
        links = []
        if nxt < len(feats):
            links.append({"rel": "next", "href":
                          f"{FEDEO_COLLECTIONS}/{coll}/items?limit={limit}"
                          f"&marker={nxt}"})
        return {"type": "FeatureCollection", "features": got,
                "numberMatched": len(feats), "numberReturned": len(got),
                "links": links}

    def serve(method, url, body, headers):
        m = re.match(re.escape(FEDEO_COLLECTIONS) + r"/([A-Za-z0-9]+)/items",
                     url)
        if not m:
            raise AssertionError(f"the smoke's fake FedEO got {url!r}")
        q = urllib.parse.parse_qs(urllib.parse.urlparse(url).query)
        limit = int((q.get("limit") or ["10"])[0])
        start = int((q.get("marker") or ["0"])[0])
        return page(m.group(1), limit, start), {}

    return cs.record(adapter, d_lo, d_hi, serve, root)


ADAPTER = BiomassCatalogue

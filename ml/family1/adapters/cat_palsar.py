"""JAXA's L-band radar of the land, catalogued: PALSAR-2 ScanSAR and the
25 m global mosaics (family 1.0.tf, tier T, E-082 wave 6).

PLAIN ENGLISH. L-band radar is the one instrument that sees the WOODY part of
a forest. Its 24 cm wavelength passes through leaves and bounces off trunks
and branches, so where an optical sensor measures greenness, this measures
structure — and it does it through cloud and at night, which is why it is the
only global forest sensor that works over the wet tropics all year. Japan has
flown it since 1992 (JERS-1), then ALOS/PALSAR (2006-2011) and ALOS-2/PALSAR-2
(2014 onwards), and publishes two things this store lists:

  * the **25 m global mosaics** — one cloud-free radar picture of all the
    land, once a year, in HH and HV polarisation, cut into 1-degree tiles;
  * the **ScanSAR Level 2.2 scenes** — individual 350 km-wide passes,
    ortho-rectified and terrain-corrected to the CEOS Analysis Ready Data
    standard for Normalised Radar Backscatter.

THE PIXELS ARE NEVER COPIED. This is the list: when, where, how large, which
instrument, which product version, and the URL. `cat_nisar` is its L-band
sibling from 2025 onwards and `cat_s1` the C-band one; between them a model
can ask "is there a radar look at this cell this week" for the whole record.

WHAT WAS MEASURED, 2026-09-20, FROM THIS SANDBOX.

  ScanSAR L2.2 IS ANONYMOUS, ON AWS OPEN DATA, AND ALREADY IN STAC.
  The product page (https://www.eorc.jaxa.jp/ALOS/en/dataset/palsar2_l22_e.htm,
  HTTP 200) names its platforms — G-Portal, Earth Engine, Tellus, AWS and
  ASF — and the AWS Open Data registry entry
  (https://registry.opendata.aws/jaxa-alos-palsar2-scansar/, HTTP 200) gives
  the bucket: `s3://jaxaalos2`, us-west-2, "No AWS account required". An
  anonymous ListObjectsV2 on
  `https://jaxaalos2.s3.us-west-2.amazonaws.com/` answered 200 and listed the
  prefixes `palsar2/`, `palsar2-scansar/`, `palsar2_inventory/`, `STAC_Sample/`
  and `test-folder/`; `palsar2/L2.2/` holds `2025/`, `Africa/`, `Asia/` and
  `For_STAC/`, and `palsar2/L2.2/2025/` is the whole ARD release — at least
  40,000 objects in the first 40 pages of 1,000, of which 5,716 are `.json`.
  Each `.json` is a real **STAC 1.0.0 Item**: fetched anonymously (HTTP 200,
  4,412 bytes) it carries `id`
  (`ALOS2010632250-140804_WBDR2.2GUD`), a Polygon `geometry`, a `bbox`,
  `properties.datetime` (`2014-08-04T00:43:05.059500Z`),
  `sar:polarizations` `["HH", "HV"]`, `sar:frequency_band`,
  `sar:instrument_mode`, `sat:orbit_state`, `proj:epsg` and six `assets`
  (HH_SLP, HV_SLP, MSK, LIN, summary, kml). The scene's DATE is also in its
  own name (`-140804_`), which is what lets a month's lane filter the listing
  before fetching a single item.

  THE 25 m MOSAICS NEED AN ACCOUNT, AND THAT WAS MEASURED TOO. The EORC
  dataset page (https://www.eorc.jaxa.jp/ALOS/en/dataset/fnf_e.htm, HTTP 200)
  carries the version and year list in its own update log — Ver. 2.6.0 for
  PALSAR 2007-2010 (30 April 2026), for PALSAR-2 2015-2019 (21 April 2026),
  2020-2024 (9 April 2026) and 2025 (11 March 2026) — and names the 1-degree
  TILE convention directly, listing the 33 tiles replaced in the 2020 mosaic
  (`N70E165` … `N72E179`). Its DATA directory,
  `https://www.eorc.jaxa.jp/ALOS/en/palsar_fnf/data/index.htm`, answers
  **HTTP 401 Unauthorized** without an account. G-Portal's catalogue service
  IS anonymous — `https://gportal.jaxa.jp/csw/csw?service=CSW&version=3.0.0&
  request=GetCapabilities` answered 200 with an OGC CSW 3.0 capabilities
  document, `.../csw/OpenSearchDescription.xml` answered 200 and documents
  every query parameter (`datasetId`, `sat`, `sen`, `pslv`, `startTime`,
  `bbox`, `count` up to 3000, `startIndex`, `outputFormat=application/json`,
  `tileHNo`, `tileVNo`), and a `datasetId`-filtered `GetRecords` returns a
  GeoJSON FeatureCollection with `properties.numberOfRecordsMatched` — but a
  query WITHOUT a dataset id (`sat=ALOS-2`) did not answer inside 70 seconds,
  so the mosaic's own dataset id was not resolved from here.

  SO THE MOSAIC COLLECTION LISTS NOTHING AND SAYS WHY, COUNTED. It declares
  the documented tile grid and the measured year list in `plan.json`, reads
  `JAXA_USER` and `JAXA_PASSWORD` from the environment, and refuses by name:
  `mosaic_refused_no_credential` without them, and
  `mosaic_refused_no_dataset_id` with them but without
  `JAXA_MOSAIC_DATASET_ID`. Writing a parser for a response nobody here has
  seen is what ADAPTER_CONTRACT rule 4 forbids; one run on a runner with the
  account settles the dataset id and the feature's own property names, and
  `plan.json` records exactly that as what would unblock it.

WHAT A ROW IS.
  time_s    the scene's acquisition start (`properties.datetime`).
  lat, lon  the footprint centre on the sphere, from the item's own polygon.
  platform  platform_hash of the STAC id.
  values    cloud   NaN — radar does not measure cloud, and a number here
                    would be a category error rather than a missing value
            valid   NaN — none is published
            angle   NaN — the local incidence angle is published as a RASTER
                    (`LIN`), not as a scene value, and a number read out of a
                    raster is not metadata this store may invent
            log2_area  log2 of the footprint's area on the sphere in km2
            sensor  1 = ALOS-2/PALSAR-2 ScanSAR, 2 = ALOS/PALSAR mosaic,
                    3 = ALOS-2/PALSAR-2 mosaic
  qc        the product: 1 = ScanSAR L2.2 (CEOS-ARD NRB), 2 = the 25 m global
            mosaic v2.6.

LICENCE. JAXA's terms (https://earth.jaxa.jp/policy/en.html; the ScanSAR page
states "Anyone can use this data free of charge, subject to the terms of use
for each platform"). The rows are our own metadata and are public; the
PIXELS carry JAXA's research terms, so the store declares
`derived_works: "restricted"` and the attribution is (C) JAXA.
"""
import datetime as dt
import os
import re
import urllib.parse

from family1.adapters import _stac as st

# ------------------------------------------------------------- ScanSAR -----
S3_BUCKET = "jaxaalos2"
S3_HOST = f"https://{S3_BUCKET}.s3.us-west-2.amazonaws.com"
SCANSAR_PREFIXES = ("palsar2/L2.2/2025/",)
SCANSAR_FIRST = (2014, 8)          # the ALOS-2 ScanSAR record's own start
AWS_REGISTRY = "https://registry.opendata.aws/jaxa-alos-palsar2-scansar/"
SCANSAR_PAGE = ("https://www.eorc.jaxa.jp/ALOS/en/dataset/palsar2_l22_e.htm")

# ------------------------------------------------------------- mosaics -----
MOSAIC_PAGE = "https://www.eorc.jaxa.jp/ALOS/en/dataset/fnf_e.htm"
MOSAIC_DATA = "https://www.eorc.jaxa.jp/ALOS/en/palsar_fnf/data/index.htm"
GPORTAL_CSW = "https://gportal.jaxa.jp/csw/csw"
GPORTAL_OSD = "https://gportal.jaxa.jp/csw/OpenSearchDescription.xml"
MOSAIC_VERSION = "2.6.0"
# MEASURED from the update log on MOSAIC_PAGE, 2026-09-20. ALOS/PALSAR flew
# 2006-2011 and ALOS-2/PALSAR-2 from 2014, which is why the list has a hole.
MOSAIC_YEARS_PALSAR = (2007, 2008, 2009, 2010)
MOSAIC_YEARS_PALSAR2 = tuple(range(2015, 2026))
MOSAIC_TILE_DEG = 1.0
MOSAIC_TILE_RE = r"^[NS]\d{2}[EW]\d{3}$"
MOSAIC_ENV = ("JAXA_USER", "JAXA_PASSWORD")
MOSAIC_DATASET_ENV = "JAXA_MOSAIC_DATASET_ID"

SENSOR_TABLE = {1: "ALOS-2/PALSAR-2 ScanSAR",
                2: "ALOS/PALSAR mosaic",
                3: "ALOS-2/PALSAR-2 mosaic"}
QC_TABLE = {1: "L2.2", 2: f"MOS-{MOSAIC_VERSION}"}

# a ScanSAR swath is 350 km wide
SCANSAR_SWATH_KM = 350.0
_DATE_IN_NAME = re.compile(r"-(\d{2})(\d{2})(\d{2})[_-]")


def scene_date(key):
    """The acquisition date in a ScanSAR object's own name, or None.

    `ALOS2010632250-140804_WBDR2.2GUD.json` -> 2014-08-04. The item's own
    `properties.datetime` is what the ROW carries; this is only the filter
    that decides which items a month's lane fetches at all, so a name whose
    token does not parse is simply not filtered on (the item is fetched and
    its real time is used).
    """
    m = _DATE_IN_NAME.search(os.path.basename(str(key)))
    if not m:
        return None
    y, mo, d = (int(m.group(1)), int(m.group(2)), int(m.group(3)))
    # ALOS-2 launched in 2014, so a two-digit year is 20xx throughout
    try:
        return dt.date(2000 + y, mo, d)
    except ValueError:
        return None


def s3_list(fet, prefix, max_pages=4000):
    """Every key under `prefix`, through anonymous ListObjectsV2 paging.

    S3 answers XML, so this is the one listing in the catalogue machinery
    that is not JSON; it goes through `_stac._text` so the smoke's canned
    archive drives the real paging code.
    """
    keys, token, pages = [], None, 0
    while True:
        q = [("list-type", "2"), ("prefix", prefix), ("max-keys", "1000")]
        if token:
            q.append(("continuation-token", token))
        url = S3_HOST + "/?" + urllib.parse.urlencode(q)
        body = st._text(fet, url)
        got = re.findall(r"<Key>(.*?)</Key>", body)
        if not got and pages == 0:
            raise st.Refusal(
                f"{url}: the bucket listed NO key under {prefix!r} — a moved "
                f"prefix, not an empty product (the 2026-09-14 rule)")
        keys.extend(got)
        pages += 1
        if "<IsTruncated>true</IsTruncated>" not in body:
            break
        m = re.search(r"<NextContinuationToken>(.*?)</NextContinuationToken>",
                      body)
        if not m:
            raise st.Refusal(f"{url}: the listing says it is truncated and "
                             f"gives no continuation token")
        token = m.group(1)
        if pages >= max_pages:
            raise st.Refusal(f"{prefix}: paged past {pages} pages — refusing "
                             f"to loop")
    return keys, pages


class PalsarCatalogue(st.CatalogueAdapter):
    store = "cat_palsar"
    title = ("JAXA L-band radar catalogue: ALOS-2/PALSAR-2 ScanSAR L2.2 "
             "(CEOS-ARD) scenes and the 25 m global mosaics v2.6, one row "
             "per scene or tile")
    licence = {
        "name": ("catalogue rows: our own metadata, CC0. The imagery they "
                 "point at is JAXA's, under JAXA's terms of use "
                 "(https://earth.jaxa.jp/policy/en.html): free of charge, "
                 "with attribution, and redistribution of the PIXELS is not "
                 "granted here"),
        "redistribution": "attribution",
        "redistribution_confirmed": True,
        "derived_works": "restricted",
        "attribution": ("(C) JAXA. ALOS-2 PALSAR-2 ScanSAR CARD4L (L2.2) and "
                        "the Global 25 m PALSAR-2/PALSAR Mosaic, Japan "
                        "Aerospace Exploration Agency, Earth Observation "
                        "Research Centre. Scene list from the JAXA AWS Open "
                        "Data bucket s3://jaxaalos2."),
        "terms": ("the ScanSAR product page states 'Anyone can use this data "
                  "free of charge, subject to the terms of use for each "
                  "platform'; the AWS Open Data registry entry states 'Data "
                  "is available for free under the terms of use'. The rows "
                  "this store holds are its own metadata and are public; the "
                  "pixels are not redistributed, which is what "
                  "derived_works: restricted records.")}
    first_year = SCANSAR_FIRST[0]
    footprint_km = SCANSAR_SWATH_KM
    scene_seconds = 25.0           # measured: a 2014-08-04 item spans 20.7 s
    SENSOR_TABLE = SENSOR_TABLE
    QC_TABLE = QC_TABLE
    credentials = ()               # the ScanSAR half needs none; see notes
    window_step = None             # one window per month; the bucket is not
    WORKERS = 6                    # time-partitioned, so the listing is shared
    qc_policy = (
        "qc is the PRODUCT: 1 = ScanSAR Level 2.2, the CEOS Analysis Ready "
        "Data Normalised Radar Backscatter form (ortho-rectified and "
        "radiometrically terrain-corrected, gamma-nought); 2 = the Global "
        f"25 m Mosaic v{MOSAIC_VERSION}. `cloud` is NaN for every row and "
        "that is a statement, not a gap: radar does not measure cloud and a "
        "number there would be a category error. `valid` is NaN (none is "
        "published) and `angle` is NaN because the local incidence angle is "
        "published as a RASTER layer (`LIN`) rather than as a scene value -- "
        "a number read out of a raster is not metadata this store may "
        "invent. A value outside a channel's bounds becomes NaN and is "
        "counted (`out_of_bounds`), never clipped. The mosaic collection "
        "lists nothing without a JAXA G-Portal account and says so by name "
        "in `counts` rather than returning an empty window.")
    sources = (f"{S3_HOST}/?list-type=2&prefix=palsar2/L2.2/",
               AWS_REGISTRY, SCANSAR_PAGE, MOSAIC_PAGE, MOSAIC_DATA,
               GPORTAL_CSW, GPORTAL_OSD)
    verified = (
        "2026-09-20 from the sandbox, anonymously. The AWS Open Data registry "
        "entry for PALSAR-2 ScanSAR CARD4L (L2.2) answered 200 and names "
        "s3://jaxaalos2 in us-west-2 with 'No AWS account required'; an "
        "anonymous ListObjectsV2 on that bucket answered 200 and listed "
        "palsar2/, palsar2-scansar/, palsar2_inventory/ and STAC_Sample/, "
        "with palsar2/L2.2/ holding 2025/, Africa/, Asia/ and For_STAC/; "
        "palsar2/L2.2/2025/ held at least 40,000 objects over 40 pages of "
        "1,000, 5,716 of them .json; and "
        "ALOS2010632250-140804_WBDR2.2GUD.json fetched anonymously (200, "
        "4,412 bytes) is a STAC 1.0.0 Item with a Polygon geometry, bbox "
        "[-179.416304, 65.688483, 173.974349, 69.305394], datetime "
        "2014-08-04T00:43:05.059500Z, sar:polarizations ['HH','HV'], "
        "proj:epsg 32660 and six assets. On the mosaic side: the EORC "
        "dataset page answered 200 and its update log gives Ver. 2.6.0 for "
        "PALSAR 2007-2010 and PALSAR-2 2015-2025 and lists 1-degree tile "
        "names (N70E165 .. N72E179); its data directory answered HTTP 401; "
        "G-Portal's CSW GetCapabilities and OpenSearchDescription both "
        "answered 200 (the latter documenting datasetId, sat, sen, pslv, "
        "startTime, bbox, count<=3000, startIndex, outputFormat and "
        "tileHNo/tileVNo) and a datasetId-filtered GetRecords returned a "
        "GeoJSON FeatureCollection, but a query without a dataset id did not "
        "answer inside 70 s, so the mosaic's dataset id is unresolved.")
    notes = (
        "Two collections, and only one of them lists today. `scansar` is "
        "ANONYMOUS: the ARD release sits on AWS Open Data as one STAC Item "
        "per scene, and the scene's date is in its own object name, so a "
        "month's lane filters the shared bucket listing before fetching a "
        "single item. `mosaic` needs a JAXA G-Portal account: the EORC data "
        "directory answers 401 and G-Portal's (anonymous, documented) CSW "
        f"needs the mosaic's dataset id, which is read from "
        f"{MOSAIC_DATASET_ENV}. Without {MOSAIC_ENV[0]}/{MOSAIC_ENV[1]} the "
        "collection refuses by name and counts the refusal; with them and no "
        "dataset id it refuses by a different name. A catalogue needs no "
        "pixel download -- only the per-tile existence -- and for the mosaic "
        "that existence is exactly what the 401 withholds.")
    smoke_window = ("2014-08-04", "2014-08-05")
    smoke_probe_month = "2014-08"

    def __init__(self):
        super().__init__()
        self._keys = None

    def record_first(self):
        return SCANSAR_FIRST

    # -- the plan ----------------------------------------------------------
    def plan(self, ctx, year, month=None):
        """One window per (collection, month) — the bucket is not partitioned
        by time, so the listing is shared and the month is a FILTER."""
        out = [("scansar", w) for w in self.windows(ctx, year, month=month,
                                                    step=None)]
        # the mosaic is annual: one window a year, and only for a year the
        # product actually publishes
        if month in (None, 12) and int(year) in self.mosaic_years():
            out.append(("mosaic", (dt.datetime(int(year), 12, 1),
                                   dt.datetime(int(year) + 1, 1, 1))))
        return out

    @staticmethod
    def mosaic_years():
        return tuple(MOSAIC_YEARS_PALSAR) + tuple(MOSAIC_YEARS_PALSAR2)

    # -- the listing -------------------------------------------------------
    def keys(self, ctx, fet):
        """The bucket's own key list, fetched once for the whole run."""
        if self._keys is None:
            out, pages = [], 0
            for p in SCANSAR_PREFIXES:
                k, n = s3_list(fet, p)
                out.extend(k)
                pages += n
            self._keys = ([k for k in out if k.endswith(".json")], pages,
                          len(out))
        return self._keys

    def scenes(self, ctx, window):
        coll, (t0, t1) = window
        if coll == "mosaic":
            return self.mosaic_scenes(ctx, t0, t1)
        fet = st.Fetcher(ctx, self.store)
        items, pages, n_all = self.keys(ctx, fet)
        counts = {"s3_pages": pages, "s3_keys": n_all,
                  "s3_items_listed": len(items)}
        lo, hi = t0.date(), (t1 - dt.timedelta(seconds=1)).date()
        want = []
        for k in items:
            d = scene_date(k)
            if d is None:
                counts["s3_items_without_date_token"] = \
                    counts.get("s3_items_without_date_token", 0) + 1
                want.append(k)             # fetched, and its real time used
            elif lo <= d <= hi:
                want.append(k)
        counts["s3_items_in_window"] = len(want)
        out = []
        for k in sorted(want):
            d, _h = fet.get(f"{S3_HOST}/{k}")
            sc = self.item(d, k, counts)
            if sc is not None:
                out.append(sc)
        return out, counts

    def item(self, d, key, counts):
        """One STAC Item -> a `Scene`, or None with a named count."""
        d = d or {}
        sid = str(d.get("id") or "")
        if not sid:
            counts["items_without_id"] = counts.get("items_without_id", 0) + 1
            return None
        props = d.get("properties") or {}
        when = props.get("datetime") or props.get("start_datetime")
        if not when:
            counts["items_without_datetime"] = \
                counts.get("items_without_datetime", 0) + 1
            return None
        rings = st.ring_from_geojson(d.get("geometry"))
        clat, clon, area = st.centre_area(rings)
        hrefs = [(a or {}).get("href") for a in
                 (d.get("assets") or {}).values()]
        base, tmpl = st.asset_template([h for h in hrefs if h], sid,
                                       f"{self.store} {sid}")
        sensor = self.sensor_code("ALOS-2/PALSAR-2 ScanSAR", counts)
        qc = self.qc_code("L2.2", counts)
        pol = props.get("sar:polarizations")
        if pol:
            k = "pol_" + "".join(sorted(str(p) for p in pol))
            counts[k] = counts.get(k, 0) + 1
        return st.Scene(
            st.seconds_of(when), clat, clon,
            [float("nan"), float("nan"), float("nan"),
             st.area_channel(area, counts), float(sensor)],
            sid, "scansar", base, tmpl, qc)

    # -- the mosaic, which needs the account -------------------------------
    def mosaic_credentials(self):
        return {k: bool(os.environ.get(k)) for k in MOSAIC_ENV}

    def mosaic_scenes(self, ctx, t0, t1):
        """The 25 m mosaic tiles — or a NAMED refusal, never an empty window.

        An empty listing read as "no tiles this year" is exactly what the
        2026-09-14 rule forbids, so the two ways this collection cannot list
        are counted apart and neither is silent.
        """
        counts = {"mosaic_windows": 1, "mosaic_year": t0.year}
        have = self.mosaic_credentials()
        missing = [k for k, v in have.items() if not v]
        if missing:
            counts["mosaic_refused_no_credential"] = 1
            counts["mosaic_missing_env"] = {k: 1 for k in missing}
            print(f"::warning::{self.store}: the 25 m mosaic tiles for "
                  f"{t0.year} were NOT listed — {' and '.join(missing)} "
                  f"is unset. The EORC data directory ({MOSAIC_DATA}) "
                  f"answers HTTP 401 without a JAXA G-Portal account "
                  f"(free registration). The ScanSAR collection is "
                  f"unaffected and needs no account.")
            return [], counts
        ds = os.environ.get(MOSAIC_DATASET_ENV)
        if not ds:
            counts["mosaic_refused_no_dataset_id"] = 1
            print(f"::warning::{self.store}: {MOSAIC_ENV[0]} and "
                  f"{MOSAIC_ENV[1]} are set but {MOSAIC_DATASET_ENV} is not. "
                  f"G-Portal's catalogue ({GPORTAL_CSW}) is anonymous and its "
                  f"parameters are documented at {GPORTAL_OSD}, but the "
                  f"mosaic's own dataset id was not resolvable without an "
                  f"account and this adapter does not guess one "
                  f"(ADAPTER_CONTRACT rule 4). Set {MOSAIC_DATASET_ENV} from "
                  f"the G-Portal product page and re-run; the run will then "
                  f"record the feature inventory the parser needs.")
            return [], counts
        counts["mosaic_refused_listing_unmeasured"] = 1
        print(f"::warning::{self.store}: {MOSAIC_DATASET_ENV}={ds} — the "
              f"listing itself has never been OBSERVED from here (every "
              f"anonymous attempt either timed out or answered 401), so no "
              f"parser for its feature properties exists yet. Run "
              f"`--stage index` on a runner with the account: it records "
              f"G-Portal's own response shape in plan.json, and the parser "
              f"is written against that rather than against a guess.")
        return [], counts

    # -- the index ---------------------------------------------------------
    def index(self, ctx):
        fet = st.Fetcher(ctx, self.store)
        items, pages, n_all = self.keys(ctx, fet)
        if not items:
            raise st.Refusal(
                f"{self.store}: the bucket listed {n_all} object(s) under "
                f"{list(SCANSAR_PREFIXES)} and NOT ONE STAC item — a moved "
                f"layout, not an empty archive (the 2026-09-14 rule)")
        first = None
        d, _h = fet.get(f"{S3_HOST}/{sorted(items)[0]}")
        c0 = {}
        sc = self.item(d, sorted(items)[0], c0)
        if sc is not None:
            first = {"id": sc.stac_id, "time_s": sc.t, "lat": sc.lat,
                     "lon": sc.lon, "log2_area": sc.values[3], "qc": sc.qc,
                     "asset_dir": sc.base_url, "asset_template": sc.asset_set,
                     "counts": c0}
        dates = [scene_date(k) for k in items]
        have = [d_ for d_ in dates if d_]
        out = {
            "dataset": "JAXA ALOS-2 PALSAR-2 ScanSAR L2.2 (CEOS-ARD NRB) and "
                       f"the Global 25 m PALSAR-2/PALSAR Mosaic v{MOSAIC_VERSION}",
            "url": f"{S3_HOST}/?list-type=2&prefix={SCANSAR_PREFIXES[0]}",
            "years": [int(v) for v in ctx.years],
            "collections": {
                "scansar": {
                    "bucket": S3_BUCKET, "region": "us-west-2",
                    "prefixes": list(SCANSAR_PREFIXES),
                    "anonymous": True,
                    "pages": pages, "objects": n_all,
                    "stac_items": len(items),
                    "items_without_date_token": len(dates) - len(have),
                    "first_date_in_names": str(min(have)) if have else None,
                    "last_date_in_names": str(max(have)) if have else None,
                    "registry": AWS_REGISTRY,
                    "first_record": first},
                "mosaic": {
                    # THE DESIGN ROW, so the registry is complete even though
                    # the collection lists nothing: what the product is, which
                    # years it covers, how its tiles are named, and exactly
                    # what would unblock the listing.
                    "version": MOSAIC_VERSION,
                    "years": list(self.mosaic_years()),
                    "years_source": (f"the update log on {MOSAIC_PAGE}, read "
                                     f"2026-09-20"),
                    "tile_degrees": MOSAIC_TILE_DEG,
                    "tile_name_pattern": MOSAIC_TILE_RE,
                    "tile_name_source": (
                        f"{MOSAIC_PAGE} lists the 33 tiles replaced in the "
                        f"2020 mosaic (N70E165 .. N72E179)"),
                    "polarisations": ["HH", "HV"],
                    "anonymous": False,
                    "data_directory": MOSAIC_DATA,
                    "data_directory_status": 401,
                    "catalogue_service": GPORTAL_CSW,
                    "catalogue_service_open_search": GPORTAL_OSD,
                    "credentials_env": list(MOSAIC_ENV),
                    "dataset_id_env": MOSAIC_DATASET_ENV,
                    "credentials_present": self.mosaic_credentials(),
                    "unblocked_by": (
                        "a free JAXA G-Portal account in JAXA_USER / "
                        "JAXA_PASSWORD, plus the mosaic's own dataset id in "
                        "JAXA_MOSAIC_DATASET_ID; one `--stage index` run with "
                        "those records G-Portal's response shape and the "
                        "listing parser is written against it")},
            },
            "sensor_table": {str(k): v for k, v in SENSOR_TABLE.items()},
            "qc_table": {str(k): v for k, v in QC_TABLE.items()},
        }
        return out

    def smoke_sources(self, root, d_lo, d_hi, seed=20260920):
        from family1.adapters import _cat_smoke as cs
        return palsar_smoke(cs, self, root, d_lo, d_hi)


# =================================================================== smoke ==
# Six scenes over two days: two on each of the two days, one whose object name
# carries no date token (so it is fetched and its own datetime decides), and
# one on a day outside the window (so the window filter is exercised). The
# listing is TRUNCATED after three keys, so the continuation-token path runs.
SMOKE_SCENES = (
    # (id, date, hh:mm:ss, lat, lon, dlat, dlon)
    ("ALOS2010632250-140804_WBDR2.2GUD", "2014-08-04", "00:43:05", 67.5, 150.0,
     1.5, 2.0),
    ("ALOS2010632300-140804_WBDR2.2GUD", "2014-08-04", "02:11:00", -12.0,
     -60.0, 1.5, 2.0),
    ("ALOS2010700100-140805_WBDR2.2GUD", "2014-08-05", "11:02:33", 4.0, 179.2,
     1.0, 2.0),                                  # across the antimeridian
    ("ALOS2010700200-140805_WBDR2.2GUD", "2014-08-05", "13:44:00", 35.0, 138.0,
     1.0, 1.0),
    ("ALOS2010999999_WBDR2.2GUD", "2014-08-05", "20:00:00", -33.0, 18.0,
     1.0, 1.0),                                  # no date token in the name
    ("ALOS2011000000-140901_WBDR2.2GUD", "2014-09-01", "01:00:00", 10.0, 10.0,
     1.0, 1.0),                                  # outside the window
)
SMOKE_ASSETS = ("HH_SLP.tif", "HV_SLP.tif", "MSK.tif", "LIN.tif")
SMOKE_PAGE = 3


def smoke_item(sid, date, hhmmss, lat, lon, dlat, dlon):
    from family1.adapters import _cat_smoke as cs
    ring = cs.ring(lat, lon, dlat, dlon)
    base = f"{S3_HOST}/{SCANSAR_PREFIXES[0]}"
    return {
        "type": "Feature", "stac_version": "1.0.0", "id": sid,
        "properties": {"datetime": f"{date}T{hhmmss}.000000Z",
                       "sar:polarizations": ["HH", "HV"],
                       "sar:frequency_band": "L",
                       "sar:instrument_mode": "WD1",
                       "proj:epsg": 32660},
        "geometry": {"type": "Polygon", "coordinates": ring},
        "bbox": [lon - dlon, lat - dlat, lon + dlon, lat + dlat],
        "assets": {a.split(".")[0]: {"href": f"{base}{sid}_{a}"}
                   for a in SMOKE_ASSETS},
        "links": [],
    }


def palsar_smoke(cs, adapter, root, d_lo, d_hi):
    items = {s[0]: smoke_item(*s) for s in SMOKE_SCENES}
    keys = [f"{SCANSAR_PREFIXES[0]}{sid}.json" for sid in items]
    # the tif objects are listed too, so the `.json` filter is exercised
    for sid in items:
        keys.append(f"{SCANSAR_PREFIXES[0]}{sid}_HH_SLP.tif")

    def listing(page_keys, token, more):
        body = ["<?xml version=\"1.0\" encoding=\"UTF-8\"?>",
                "<ListBucketResult><Name>jaxaalos2</Name>",
                f"<IsTruncated>{'true' if more else 'false'}</IsTruncated>"]
        if more:
            body.append(f"<NextContinuationToken>{token}"
                        f"</NextContinuationToken>")
        for k in page_keys:
            body.append(f"<Contents><Key>{k}</Key><Size>1</Size></Contents>")
        body.append("</ListBucketResult>")
        return "".join(body)

    def text_serve(url):
        q = urllib.parse.parse_qs(urllib.parse.urlparse(url).query)
        if "list-type" not in q:
            raise AssertionError(f"the smoke's fake bucket got {url!r}")
        start = int((q.get("continuation-token") or ["0"])[0])
        page = keys[start:start + SMOKE_PAGE]
        nxt = start + SMOKE_PAGE
        return listing(page, str(nxt), nxt < len(keys))

    def serve(method, url, body, headers):
        m = re.match(re.escape(S3_HOST) + r"/(.+)\.json$", url)
        if m:
            sid = os.path.basename(m.group(1))
            if sid in items:
                return items[sid], {}
        raise AssertionError(f"the smoke's fake bucket got {url!r}")

    return cs.record(adapter, d_lo, d_hi, serve, root, text_serve=text_serve)


ADAPTER = PalsarCatalogue

"""Sentinel-2 L2A — every 110 km tile of Europe's 10 m optical camera.

PLAIN ENGLISH. Two (now three) Sentinel-2 satellites photograph the whole land
surface every five days at 10 m, and ESA publishes the surface-reflectance
form as one archive per 110 km military-grid tile per overpass. That is about
three petabytes a year and 15,800 tiles a day. This store is the LIST: one row
per tile-overpass with the second it was taken, the footprint's centre, the
cloud fraction, the footprint's area, which satellite took it, and the
processing baseline. The pixels stay in the Copernicus Data Space.

SOURCE, VERIFIED 2026-09-18 FROM THIS SANDBOX (anonymous, no CDSE account
needed to SEARCH):
  https://catalogue.dataspace.copernicus.eu/odata/v1/Products
      ?$filter=Collection/Name eq 'SENTINEL-2' and <productType eq 'S2MSI2A'>
              and ContentDate/Start ge <from> and ContentDate/Start lt <to>
      &$expand=Attributes&$orderby=ContentDate/Start asc&$top=1000&$skip=<n>

**THE CDSE STAC ENDPOINT DOES NOT SERVE SENTINEL-2, AND THAT WAS MEASURED.**
`https://catalogue.dataspace.copernicus.eu/stac/collections` answers 200 with
**ten** collections — five Copernicus Contributing Missions groups and five
CLMS burnt-area products — and not one Sentinel collection among them. So the
OData endpoint is not a fallback here, it is the only catalogue CDSE offers
for this data, and the adapter uses it rather than a STAC API that would 404
every query.

TWO SERVER LIMITS DECIDE THE WHOLE DESIGN, both measured:
  * `$top` above 1000 answers **HTTP 422**; 1000 it is.
  * `$skip` above 10,000 answers **HTTP 422** with the body
    `{"detail":[{"type":"less_than_equal","loc":["query","$skip"],
    "msg":"Input should be less than or equal to 10000"}]}`.
So at most 11,000 products are reachable inside ONE filter, and a Sentinel-2
DAY holds 15,844 (2026-09-01, measured). A day therefore cannot be paged at
all. The adapter lists DAY windows and `_stac.odata_windows` halves any window
whose own `@odata.count` exceeds what paging can reach — a 2026 day becomes
two 12-hour windows of about 7,900 — so nothing is ever truncated and the
request count is 40 % of what hour windows cost. Requests, not round trips,
are the binding constraint here: CDSE sits behind a WAF that answers
`{"error":"WAF","message":"Rate limit exceeded"}` and counts something wider
than one client, so `_stac.RATE_LIMITS` holds this host to 120 a minute per
lane and no more than four lanes list it at once. Every window's rows must equal that window's
`@odata.count` or the fetch refuses.

S2A IS NOT RETIRED — A LEDGER CORRECTION. The note's table says
"S2B + S2C (S2A retired 2026-03)". Measured on 2026-09-18 by
`platformSerialIdentifier` over `ContentDate/Start` (the ACQUISITION time, not
the publication time):

  S2A L2A, 2026-02   24,411      S2A L2A, 2026-04   86,193
  S2A L2A, 2026-09   48,637      S2B 2026-09  111,581   S2C 2026-09  109,328

S2A was still acquiring in September 2026, and its April 2026 month is three
and a half times its February one. The store therefore carries THREE
satellites with three sensor codes and no end date for any of them; the index
records the per-satellite counts of the run's own first month so the claim is
re-measured on every build rather than trusted.

MEASURED ARCHIVE SIZE, 2026-09-18 (`@odata.count` per year, productType
`S2MSI2A`): 2015 178,073 · 2016 1,310,563 · 2017 2,249,634 · 2018 3,798,249 ·
2019 3,942,872 · 2020 4,025,098 · 2021 4,072,057 · 2022 4,171,675 ·
2023 4,257,526 · 2024 4,377,414 · 2025 5,067,823 · 2026 3,781,021 (to
2026-09-18) — **41,232,005 products**, the largest of the eight catalogues.
`S2MSI2Ap` (the L2A pilot product type) returns 0 and is not catalogued.

WHAT A ROW IS.
  time_s    `ContentDate/Start` — the sensing instant of the tile, int32
            seconds since 1982-01-01.
  lat, lon  the spherical mean of `GeoFootprint`'s own ring. A quarter of the
            tiles sit on a UTM zone edge and many cross the antimeridian, so
            the centre is computed on the sphere and has no seam.
  platform  platform_hash(the product Name, e.g.
            `S2B_MSIL2A_20240601T000139_N0510_R016_T08XMR_20240601T001801.SAFE`).
  values    cloud   the `cloudCover` attribute, per cent
            valid   NaN — CDSE publishes no nodata or valid-pixel percentage
                    in the product's attributes
            angle   NaN — no mean sun zenith is published either
            log2_area
                    log2 of the footprint's area on the sphere in km²
            sensor  1 S2A, 2 S2B, 3 S2C, 4 S2D (not yet launched)
  qc        the PROCESSING BASELINE, `processorVersion`: measured 05.00 for
            2015-2021 (the reprocessed archive), 05.10 for 2023-2024, 05.11
            for 2025 and 05.12 for 2026. The older 02.x-04.x baselines are in
            the table for the products the reprocessing has not reached.

ASSETS. A Sentinel product is ONE `.SAFE` archive, fetched whole from
`odata/v1/Products(<Id>)/$value`, so `assets.parquet`'s `base_url` is that
product's own endpoint and its template is the single `$value`. The `S3Path`
CDSE also publishes (`/eodata/Sentinel-2/MSI/L2A/<Y>/<m>/<d>/<Name>`) is a
pure function of the product name and the note in store.json says so; storing
it per row would be 60 bytes of the same arithmetic 41 million times.
"""
import datetime as dt

import numpy as np

import build_family10_stores as f10b
from family1.adapters import _stac as st

SENSOR_TABLE = {1: "S2A", 2: "S2B", 3: "S2C", 4: "S2D"}
QC_TABLE = {0: "05.00", 1: "05.10", 2: "05.11", 3: "05.12", 4: "05.13",
            10: "04.00", 11: "03.01", 12: "02.14", 13: "02.12", 14: "02.11",
            15: "02.10", 16: "02.09", 17: "02.08", 18: "02.07", 19: "02.06",
            20: "02.05", 21: "02.04", 22: "02.01", 30: "99.99"}
PRODUCTS = (("l2a", "S2MSI2A"),)
RECORD_FIRST = (2015, 7)
S3_PREFIX = "/eodata/Sentinel-2/MSI/L2A/<YYYY>/<MM>/<DD>/<Name>"


class S2Catalogue(st.OdataCatalogue):
    store = "cat_s2"
    title = ("Sentinel-2 L2A tile catalogue (10/20/60 m surface reflectance, "
             "S2A + S2B + S2C, CDSE), one row per tile-overpass")
    licence = {
        "name": ("catalogue rows: our own metadata, CC0. The imagery they "
                 "point at is Copernicus Sentinel-2 L2A — free and open "
                 "under the Copernicus data policy (Regulation (EU) "
                 "No 1159/2013), redistribution allowed with attribution"),
        "redistribution": "yes", "derived_works": "free",
        "attribution": ("Contains modified Copernicus Sentinel-2 data; "
                        "product list from the Copernicus Data Space "
                        "Ecosystem, "
                        "https://catalogue.dataspace.copernicus.eu")}
    first_year = 2015
    footprint_km = 109.8
    scene_seconds = 1.0
    collection_name = "SENTINEL-2"
    PRODUCTS = PRODUCTS
    SENSOR_TABLE = SENSOR_TABLE
    QC_TABLE = QC_TABLE
    # DAY WINDOWS, SPLIT WHEN THEY DO NOT FIT. A 2026 day holds 15,844
    # products and CDSE can page 11,000, so `_stac.odata_windows` halves it
    # into two 12-hour windows of about 7,900 — which costs ONE count for the
    # day, two for the halves and sixteen pages, against the twenty-four
    # counts and twenty-four pages twenty-four hour-windows need. Against a
    # host whose WAF is the binding constraint, that is 40 % of the requests
    # for the same rows, and an early year (a 2016 day is 3,600 products) is
    # one count and four pages.
    window_step = "day"
    qc_policy = (
        "qc is the ESA PROCESSING BASELINE (`processorVersion`): 0 = 05.00, "
        "1 = 05.10, 2 = 05.11, 3 = 05.12, 4 = 05.13, and 10..30 for the "
        "02.x-04.x baselines and the 99.99 test value. A baseline the table "
        "does not list is 255 and counted by name in `qc_unlisted`, so a new "
        "baseline appears in the build report instead of changing what a row "
        "means. `cloud` is the `cloudCover` attribute; `valid` and `angle` "
        "are NaN for every row because CDSE publishes neither a valid-pixel "
        "percentage nor a mean sun zenith in the product attributes. A value "
        "outside a channel's bounds is NaN and counted (`out_of_bounds`), "
        "never clipped.")
    sources = (st.CDSE_ODATA,
               "https://catalogue.dataspace.copernicus.eu/stac/collections",
               st.CDSE_DOWNLOAD)
    verified = (
        "2026-09-18 from the sandbox, anonymously: the CDSE STAC serves TEN "
        "collections and no Sentinel one, so OData is the catalogue; $top>1000 "
        "and $skip>10000 both answer HTTP 422 (the $skip body names the "
        "limit); $orderby=ContentDate/Start asc works; 12,683 products on "
        "2024-06-01 and 993 in the hour 10:00-11:00; 15,844 on 2026-09-01; "
        "41,232,005 products over 2015-2026 by year; productType S2MSI2Ap is "
        "empty; and S2A is STILL ACQUIRING (48,637 products in 2026-09) "
        "against the ledger's 'retired 2026-03'")
    notes = (
        "The pixels are never copied. THREE satellites: the ledger's 'S2A "
        "retired 2026-03' is wrong and the index re-measures it every build. "
        "Hour windows, because a day cannot be paged inside CDSE's $skip cap "
        "of 10,000; a window whose own count exceeds the reach is halved "
        f"until it fits. The S3 path of a product is {S3_PREFIX}.")
    smoke_window = ("2024-06-01", "2024-06-02")
    smoke_probe_month = "2024-06"

    def record_first(self):
        return RECORD_FIRST

    def sensor_text(self, name, aa):
        s = str(aa.get("platformSerialIdentifier") or "").strip().upper()
        if s in ("A", "B", "C", "D"):
            return "S2" + s
        # the attribute is missing: the product NAME starts with the satellite
        return str(name)[:3].upper()

    def qc_text(self, name, prod, aa):
        return str(aa.get("processorVersion") or "").strip()

    def values(self, name, prod, aa, area, sensor, counts):
        cc = aa.get("cloudCover")
        if cc is None:
            counts["attr_missing_cloudCover"] = \
                counts.get("attr_missing_cloudCover", 0) + 1
        return [st.clamp_pct(cc), float("nan"), float("nan"),
                st.area_channel(area, counts), float(sensor)]

    # -- the index ---------------------------------------------------------
    def index(self, ctx):
        fet = st.Fetcher(ctx, self.store)
        y, m = ctx.d_lo.year, ctx.d_lo.month
        if (y, m) < RECORD_FIRST:
            y, m = RECORD_FIRST
        a = dt.datetime(y, m, 1)
        b = dt.datetime(y + (m == 12), m % 12 + 1, 1)
        total = st.odata_count(fet, st.CDSE_ODATA,
                               self.filt(a, b, PRODUCTS[0][1]))
        per_sat = {}
        for code, sat in sorted(SENSOR_TABLE.items()):
            f = (self.filt(a, b, PRODUCTS[0][1]) +
                 f" and startswith(Name,'{sat}_MSIL2A')")
            per_sat[sat] = st.odata_count(fet, st.CDSE_ODATA, f)
        if not total:
            raise st.Refusal(
                f"{self.store}: CDSE counts ZERO S2MSI2A products in "
                f"{y:04d}-{m:02d} — a broken filter, not a measurement "
                f"(the 2026-09-14 rule).")
        named = sum(per_sat.values())
        if named != total:
            print(f"  ::warning::the per-satellite counts sum to {named} and "
                  f"the month holds {total} — a satellite whose name prefix "
                  f"this adapter does not know is in the archive")
        # ONE REAL PRODUCT, read exactly the way a row is read — the field
        # selection, the footprint, the attribute names and the asset layout
        # are all checked here, where the inputs are all it has cost
        # (ml/CLAUDE.md §0.3).
        one = None
        d, _ = fet.get(st.odata_url(
            st.CDSE_ODATA, filter=self.filt(a, b, PRODUCTS[0][1]), top=1,
            expand=self.expand, orderby="ContentDate/Start asc"))
        for p in (d.get("value") or []):
            c2 = {}
            sc = self.product(p, PRODUCTS[0], c2, "index")
            one = {"name": sc.stac_id, "time_s": sc.t, "lat": sc.lat,
                   "lon": sc.lon,
                   "values": dict(zip(self.channel_names,
                                      [None if not np.isfinite(x) else x
                                       for x in sc.values])),
                   "qc": sc.qc, "base_url": sc.base_url,
                   "asset_template": sc.asset_set,
                   "attributes": st.odata_attrs(p), "counts": c2}
            break
        return {
            "dataset": "Sentinel-2 L2A (CDSE OData)",
            "url": st.CDSE_ODATA,
            "collection": self.collection_name,
            "product_types": [p[1] for p in PRODUCTS],
            "stac_endpoint_serves_sentinel": False,
            "stac_endpoint_note": (
                "https://catalogue.dataspace.copernicus.eu/stac lists ten "
                "collections, all Contributing Missions or CLMS burnt area, "
                "and no Sentinel collection (measured 2026-09-18) — OData is "
                "the only CDSE catalogue for this data"),
            "top_max": st.ODATA_TOP, "skip_max": st.ODATA_SKIP_MAX,
            "paging_reach": st.ODATA_REACH,
            "window_step": self.window_step,
            "years": [int(v) for v in ctx.years],
            "first_month": f"{y:04d}-{m:02d}",
            "products_first_month": total,
            "products_first_month_by_satellite": per_sat,
            "s2a_still_acquiring": bool(per_sat.get("S2A", 0) > 0),
            "s3_path_pattern": S3_PREFIX,
            "sensor_table": {str(k): v for k, v in SENSOR_TABLE.items()},
            "qc_table": {str(k): v for k, v in QC_TABLE.items()},
            "first_record": one,
        }

    def smoke_sources(self, root, d_lo, d_hi, seed=20260918):
        from family1.adapters import _cat_smoke as cs
        return cs.odata_sources(self, root, d_lo, d_hi, cs.S2_PRODUCTS)


ADAPTER = S2Catalogue

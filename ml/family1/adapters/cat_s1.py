"""Sentinel-1 IW — Europe's C-band radar, ground-range and single-look complex.

PLAIN ENGLISH. Sentinel-1 is a radar, so it sees through cloud and works at
night; its everyday mode (Interferometric Wide swath, IW) images a 250 km
strip at 10 m with two polarisations. ESA publishes each strip twice: as GRDH
(ground-range detected, high resolution — the picture) and as SLC (single-look
complex — the same strip with the radar PHASE kept, which is what
interferometry and dual-pol covariance need). Both are catalogued here, with
their own sensor codes, because they are two different products of one
overpass and a model that wants the covariance wants the SLC.

SOURCE, VERIFIED 2026-09-18 FROM THIS SANDBOX (anonymous, no CDSE account
needed to SEARCH):
  https://catalogue.dataspace.copernicus.eu/odata/v1/Products
      ?$filter=Collection/Name eq 'SENTINEL-1' and <productType eq ...>
              and ContentDate/Start ge <from> and ContentDate/Start lt <to>
      &$expand=Attributes&$orderby=ContentDate/Start asc&$top=1000
The CDSE STAC endpoint serves no Sentinel collection at all (see `cat_s2`'s
docstring — ten collections, none of them Sentinel), so OData is the
catalogue rather than a fallback.

**EVERY GRDH SCENE IS IN THE ARCHIVE TWICE, AND `contains(Name,'IW_GRDH')`
DOUBLE-COUNTS IT.** Measured 2026-09-18 by `productType`:

  day           IW_GRDH_1S   IW_GRDH_1S-COG   IW_SLC__1S
  2014-11-01           193              197          132
  2019-06-01         1,110            1,110          919
  2024-06-01           529              529          529
  2026-09-01           922              922          922

`IW_GRDH_1S-COG` is the SAME scene republished as a cloud-optimised GeoTIFF —
`contains(Name,'IW_GRDH')` matched both and returned 1,058 for 2024-06-01,
which is exactly 2 x 529 and is the number a careless adapter would have
stored. So this store filters on `productType` EXACTLY, catalogues the
original `IW_GRDH_1S`, and the index measures the `-COG` twin's count beside
it so the mirror is on the record rather than invisible. Note 2014-11-01,
where the two counts are 193 and 197: the mirror is not a perfect 1:1
historically, which is the second reason not to let one stand for the other.

S1A IS RETIRED AND THE ARCHIVE SAYS SO. Measured by
`startswith(Name,'S1A_IW_GRDH')` over the ACQUISITION time: 28,608 products
in 2026-06 and **0** in 2026-09, which matches the ledger's "S1A retired
2026-06-30". S1C and S1D are both flying (15,139 and 17,485 GRDH products in
2026-09). All four satellites keep their sensor codes; a retired one simply
stops producing rows.

MEASURED ARCHIVE SIZE, 2026-09-18 (`@odata.count` per year):
  IW_GRDH_1S  3,184,957   IW_SLC__1S  3,018,395   -> 6,203,352 rows
  (2014 15,576 / 13,095 · 2017 290,435 / 290,058 · 2020 397,333 / 336,519 ·
   2024 199,409 / 199,408 · 2026 268,742 / 274,895 to 2026-09-18)

WHAT A ROW IS.
  time_s    `ContentDate/Start` — the start of the slice's acquisition.
  lat, lon  the spherical mean of `GeoFootprint`'s ring.
  platform  platform_hash(the product Name, e.g.
            `S1A_IW_GRDH_1SDV_20240601T001631_..._E1AD_COG.SAFE`).
  values    cloud   NaN — a radar does not see cloud, and NaN is the honest
                    entry for a channel this instrument cannot measure
            valid   NaN — none is published
            angle   NaN — CDSE's product attributes carry no incidence
                    angle for an IW product (the per-pixel incidence is in
                    the annotation inside the .SAFE, not in the catalogue),
                    so the channel the note calls "radar incidence" is a gap
                    here rather than a guess
            log2_area
                    log2 of the footprint's area on the sphere in km²
            sensor  11 S1A GRDH, 12 S1A SLC, 21 S1B GRDH, 22 S1B SLC,
                    31 S1C GRDH, 32 S1C SLC, 41 S1D GRDH, 42 S1D SLC —
                    satellite x product, so one code answers both questions
  qc        the IPF PROCESSING BASELINE (`processorVersion`, e.g. 003.71),
            with the polarisation pair folded in: the note's "C-band
            polarimetry: dual, never quad" is a property of the product and
            `polarisationChannels` is how the archive states it, so
            `QC_TABLE` keys on `<version>/<channels>`.
"""
import datetime as dt

import numpy as np

import build_family10_stores as f10b
from family1.adapters import _stac as st

PRODUCTS = (("grdh", "IW_GRDH_1S"), ("slc", "IW_SLC__1S"))
COG_TYPE = "IW_GRDH_1S-COG"
SATS = ("S1A", "S1B", "S1C", "S1D")
#: satellite x product; 10 x satellite index + 1 GRDH / 2 SLC
SENSOR_TABLE = {}
for _i, _s in enumerate(SATS, start=1):
    SENSOR_TABLE[_i * 10 + 1] = f"{_s}/IW_GRDH_1S"
    SENSOR_TABLE[_i * 10 + 2] = f"{_s}/IW_SLC__1S"
POLS = ("VV&VH", "HH&HV", "VV", "HH", "VV&VH&HH&HV")
IPFS = ("002.36", "002.52", "002.53", "002.60", "002.62", "002.70",
        "002.71", "002.72", "002.82", "002.84", "002.90", "002.91",
        "003.10", "003.20", "003.30", "003.31", "003.40", "003.52",
        "003.61", "003.71", "003.80", "003.90", "004.00", "004.10")
QC_TABLE = {}
for _j, _v in enumerate(IPFS):
    for _k, _p in enumerate(POLS):
        QC_TABLE[_j * 5 + _k] = f"{_v}/{_p}"
RECORD_FIRST = (2014, 10)
S3_PREFIX = "/eodata/Sentinel-1/SAR/<productType>/<YYYY>/<MM>/<DD>/<Name>"


class S1Catalogue(st.OdataCatalogue):
    store = "cat_s1"
    title = ("Sentinel-1 IW catalogue — GRDH (ground-range detected) and SLC "
             "(single-look complex), dual-pol 10 m C-band radar, CDSE; one "
             "row per slice-product")
    licence = {
        "name": ("catalogue rows: our own metadata, CC0. The imagery they "
                 "point at is Copernicus Sentinel-1 — free and open under "
                 "the Copernicus data policy (Regulation (EU) No 1159/2013), "
                 "redistribution allowed with attribution"),
        "redistribution": "yes", "derived_works": "free",
        "attribution": ("Contains modified Copernicus Sentinel-1 data; "
                        "product list from the Copernicus Data Space "
                        "Ecosystem, "
                        "https://catalogue.dataspace.copernicus.eu")}
    first_year = 2014
    # An IW slice is ~250 km across track and ~170 km along it; the nominal
    # width is the square root of that area and the probe's median `area`
    # checks it.
    footprint_km = 206.0
    scene_seconds = 30.0
    collection_name = "SENTINEL-1"
    PRODUCTS = PRODUCTS
    SENSOR_TABLE = SENSOR_TABLE
    QC_TABLE = QC_TABLE
    window_step = "day"          # 922 GRDH products a day: one page of 1000
    qc_policy = (
        "qc is the Sentinel-1 IPF processing baseline together with the "
        "polarisation pair the product carries — `<processorVersion>/"
        "<polarisationChannels>`, e.g. `003.71/VV&VH` — because the note's "
        "'dual, never quad' is a property of the product and "
        "`polarisationChannels` is how the archive states it. The table "
        "covers 24 IPF versions x 5 polarisation combinations; anything else "
        "is 255 and counted by name in `qc_unlisted`. `cloud`, `valid` and "
        "`angle` are NaN for every row: a radar does not measure cloud, no "
        "valid fraction is published, and CDSE's product attributes carry no "
        "incidence angle (it lives in the annotation inside the .SAFE). A "
        "value outside a channel's bounds is NaN and counted "
        "(`out_of_bounds`).")
    sources = (st.CDSE_ODATA,
               "https://catalogue.dataspace.copernicus.eu/stac/collections",
               st.CDSE_DOWNLOAD)
    verified = (
        "2026-09-18 from the sandbox, anonymously: productType IW_GRDH_1S and "
        "IW_GRDH_1S-COG hold the SAME scenes (529/529 on 2024-06-01, 922/922 "
        "on 2026-09-01, 1,110/1,110 on 2019-06-01, 193/197 on 2014-11-01), so "
        "contains(Name,'IW_GRDH') double-counts and the filter is on "
        "productType exactly; IW_SLC__1S 529/922/919/132 on the same days; "
        "S1A has 28,608 GRDH products in 2026-06 and 0 in 2026-09 (retired, "
        "as the ledger says) while S1C and S1D have 15,139 and 17,485 in "
        "2026-09; 3,184,957 GRDH and 3,018,395 SLC products over 2014-2026 "
        "by year")
    notes = (
        "The pixels are never copied. The GRDH mirror `IW_GRDH_1S-COG` is the "
        "same scene and is NOT a second row; the index measures its count. "
        "The SLC is catalogued beside the GRDH with its own sensor code, for "
        "the dual-pol covariance. S1A stopped acquiring 2026-06-30 and its "
        "sensor codes simply stop appearing. The incidence angle the note's "
        "third channel asks for is not in CDSE's catalogue metadata and is "
        f"NaN. The S3 path of a product is {S3_PREFIX}.")
    smoke_window = ("2024-06-01", "2024-06-02")
    smoke_probe_month = "2024-06"

    def record_first(self):
        return RECORD_FIRST

    def sensor_text(self, name, aa):
        s = str(aa.get("platformSerialIdentifier") or "").strip().upper()
        sat = ("S1" + s) if s in ("A", "B", "C", "D") else str(name)[:3].upper()
        pt = str(aa.get("productType") or "").strip()
        return f"{sat}/{pt}"

    def qc_text(self, name, prod, aa):
        return (f"{str(aa.get('processorVersion') or '').strip()}/"
                f"{str(aa.get('polarisationChannels') or '').strip()}")

    def values(self, name, prod, aa, area, sensor, counts):
        return [float("nan"), float("nan"), float("nan"),
                st.area_channel(area, counts), float(sensor)]

    # -- the index ---------------------------------------------------------
    def index(self, ctx):
        fet = st.Fetcher(ctx, self.store)
        y, m = ctx.d_lo.year, ctx.d_lo.month
        if (y, m) < RECORD_FIRST:
            y, m = RECORD_FIRST
        a = dt.datetime(y, m, 1)
        b = dt.datetime(y + (m == 12), m % 12 + 1, 1)
        per_type = {}
        for key, pt in PRODUCTS:
            per_type[pt] = st.odata_count(fet, st.CDSE_ODATA,
                                          self.filt(a, b, pt))
        # THE MIRROR, MEASURED. `IW_GRDH_1S-COG` is the same scene republished;
        # counting it here is what keeps "we catalogue each scene once" a
        # measurement rather than a claim.
        cog = st.odata_count(fet, st.CDSE_ODATA, self.filt(a, b, COG_TYPE))
        per_sat = {}
        for sat in SATS:
            for key, pt in PRODUCTS:
                f = (self.filt(a, b, pt) +
                     f" and startswith(Name,'{sat}_{pt[:8]}')")
                per_sat[f"{sat}/{pt}"] = st.odata_count(fet, st.CDSE_ODATA, f)
        total = sum(per_type.values())
        if not total:
            raise st.Refusal(
                f"{self.store}: CDSE counts ZERO IW GRDH or SLC products in "
                f"{y:04d}-{m:02d} — a broken filter, not a measurement "
                f"(the 2026-09-14 rule).")
        one = None
        for key, pt in PRODUCTS:
            if not per_type[pt]:
                continue
            d, _ = fet.get(st.odata_url(
                st.CDSE_ODATA, filter=self.filt(a, b, pt), top=1,
                expand=self.expand, orderby="ContentDate/Start asc"))
            for p in (d.get("value") or []):
                c2 = {}
                sc = self.product(p, (key, pt), c2, "index")
                one = {"name": sc.stac_id, "product_type": pt,
                       "time_s": sc.t, "lat": sc.lat, "lon": sc.lon,
                       "log2_area": sc.values[3], "sensor": sc.values[4],
                       "qc": sc.qc, "base_url": sc.base_url,
                       "asset_template": sc.asset_set,
                       "attributes": st.odata_attrs(p), "counts": c2}
                break
            if one:
                break
        return {
            "dataset": "Sentinel-1 IW GRDH + SLC (CDSE OData)",
            "url": st.CDSE_ODATA,
            "collection": self.collection_name,
            "product_types": [p[1] for p in PRODUCTS],
            "cog_mirror_product_type": COG_TYPE,
            "cog_mirror_first_month": cog,
            "cog_mirror_note": (
                "the -COG product type republishes the same GRDH scenes; "
                "catalogued ONCE, as IW_GRDH_1S. contains(Name,'IW_GRDH') "
                "would have matched both and doubled the store."),
            "s1a_retired": bool(sum(v for k, v in per_sat.items()
                                    if k.startswith("S1A")) == 0),
            "top_max": st.ODATA_TOP, "skip_max": st.ODATA_SKIP_MAX,
            "paging_reach": st.ODATA_REACH,
            "window_step": self.window_step,
            "years": [int(v) for v in ctx.years],
            "first_month": f"{y:04d}-{m:02d}",
            "products_first_month": per_type,
            "products_first_month_by_satellite": per_sat,
            "s3_path_pattern": S3_PREFIX,
            "sensor_table": {str(k): v for k, v in SENSOR_TABLE.items()},
            "qc_table": {str(k): v for k, v in QC_TABLE.items()},
            "first_record": one,
        }

    def smoke_sources(self, root, d_lo, d_hi, seed=20260918):
        from family1.adapters import _cat_smoke as cs
        return cs.odata_sources(self, root, d_lo, d_hi, cs.S1_PRODUCTS)


ADAPTER = S1Catalogue

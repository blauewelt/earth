"""NISAR L2 GCOV — the first free quad-polarimetric L-band radar, catalogued.

PLAIN ENGLISH. NISAR is the NASA–ISRO radar satellite launched in 2025. Its
L-band radar penetrates vegetation and its geocoded covariance product (GCOV)
keeps the full polarimetric information — which is what makes it the only free
quad-pol L-band record there is. This store is the LIST of those frames: when,
where the footprint's centre is, how large it is, which polarimetric mode it
was acquired in and which processing tier produced it. The pixels, 0.7 to 1
petabyte a year, stay at the Alaska Satellite Facility.

SOURCE, VERIFIED 2026-09-18 FROM THIS SANDBOX (anonymous):
  https://api.daac.asf.alaska.edu/services/search/param
      ?dataset=NISAR&processingLevel=GCOV&start=<from>&end=<to>
      &output=jsonlite&maxResults=250
  and the same query with `output=count`, which answers a BARE INTEGER (not
  JSON) and is the producer's own count this store checks every window
  against.

THE PARAMETERS, MEASURED RATHER THAN TRANSCRIBED.
  * `dataset=NISAR&processingLevel=GCOV` -> **138,762** granules.
    `platform=NISAR&processingLevel=GCOV` -> 138,101, i.e. 661 fewer; the
    `dataset` form is the wider and is the one used.
  * `collectionName=NISAR_L2_GCOV_PROVISIONAL_V1` as a FILTER returns **0**,
    although every result carries exactly that string in its
    `collectionName` field. So the collection cannot be filtered on and is
    read off each granule instead — a parameter that looks like it works and
    silently returns nothing is the shape of failure this whole contract is
    written against.
  * `maxResults` above 250 FAILS, and ASF offers no cursor, so the only way
    to page is to narrow the window. The adapter lists HOUR windows (about 50
    granules in a 2026 hour) and `_stac.asf_windows` halves any window whose
    own `output=count` exceeds 250, so nothing is ever truncated.
  * a day with no acquisitions answers `0` and is COUNTED
    (`windows_empty`), not refused: 2026-08-01 to 08-08 really holds none,
    while 08-15 to 09-01 holds 23,318 of the month's 29,992.

THE RECORD STARTS 2025-10, NOT 2026-06 — A LEDGER CORRECTION. The note dates
this store "2026-06-17 →" with "L2 GCOV provisional (2026-07-20 →)". Measured
by month: 2025-09 **0**, 2025-10 **1,719**, 2025-11 7,697, 2025-12 9,537,
2026-01 6,638, 2026-06 20,465, 2026-07 37,262. The early months are the
**NISAR_L2_GCOV_BETA_V1** collection (PGE R05.00.8); the provisional
collection (R05.02.3) takes over later. Both are catalogued, and `qc` says
which tier a row came from, so a consumer that wants only the provisional
product selects on it. `first_year` is 2025.

WHAT A ROW IS.
  time_s    `startTime` — the start of the frame's acquisition.
  lat, lon  the spherical mean of the granule's own `wkt` polygon.
  platform  platform_hash(`granuleName`, e.g.
            `NISAR_L2_PR_GCOV_028_099_D_078_0005_NADV_A_20260820T015433_...`).
  values    cloud   NaN — a radar does not see cloud
            valid   NaN — none is published
            angle   NaN — `offNadirAngle` and `pointingAngle` are BOTH null
                    on every GCOV granule measured (131 of 131 in one
                    two-hour window), so the "radar incidence" the note's
                    third channel asks for is a gap here rather than a guess
            log2_area
                    log2 of the footprint's area on the sphere in km²
            sensor  the POLARIMETRIC MODE, from the granule name's own
                    four-character field: two characters per frequency band,
                    each one of SH / SV / DH / DV / QQ / CL / CR / NA (single,
                    dual, quad, compact-left, compact-right, not acquired).
                    Measured in the archive: NADV, DHDH, SHNA, NASV, SVSH,
                    DVDV, SHSH. All 64 combinations have codes so a mode that
                    has not flown yet does not become 255 the day it does.
  qc        the processing TIER and the product class: 0 BETA/PR, 1 BETA/UR,
            2 PROVISIONAL/PR, 3 PROVISIONAL/UR, 4 VALIDATED/PR,
            5 VALIDATED/UR. `PR` and `UR` are the granule name's own third
            field — UR is an Urgent Response product, which the archive
            serves beside the routine one. The exact PGE version
            (`R05.00.8`, `R05.02.3`, …) is not a uint8 vocabulary; it is
            counted per build under `pge_versions` and recorded in the index.

ASSETS. Only the `downloadUrl` — the granule's own `.h5` — goes into
`assets.parquet`. ASF also publishes browse PNGs under a `BROWSE/` prefix and
`nisar.additionalUrls` under a `NISAR-JPL-PRIVATE-DATA/` prefix; those are
different directories and, in the private case, not something a consumer can
fetch, so the store points at the one file that is the data. The `asset_set`
template is therefore the single `.h5` name.
"""
import datetime as dt
import itertools

import numpy as np

import build_family10_stores as f10b
from family1.adapters import _stac as st

ASF = "https://api.daac.asf.alaska.edu/services/search/param"
DATASET = "NISAR"
LEVEL = "GCOV"
POL_CODES = ("SH", "SV", "DH", "DV", "QQ", "CL", "CR", "NA")
#: every two-band polarimetric mode NISAR can fly, 1..64
SENSOR_TABLE = {i: a + b for i, (a, b) in
                enumerate(itertools.product(POL_CODES, POL_CODES), start=1)}
TIERS = ("BETA", "PROVISIONAL", "VALIDATED")
CLASSES = ("PR", "UR")
QC_TABLE = {}
for _i, _t in enumerate(TIERS):
    for _j, _c in enumerate(CLASSES):
        QC_TABLE[_i * 2 + _j] = f"{_t}/{_c}"
RECORD_FIRST = (2025, 10)
PRIVATE_MARK = "NISAR-JPL-PRIVATE-DATA"


def tier_of(collection):
    """`NISAR_L2_GCOV_PROVISIONAL_V1` -> `PROVISIONAL`."""
    s = str(collection or "").upper()
    for t in TIERS:
        if t in s:
            return t
    return s or "?"


def name_fields(granule):
    """The granule name's own fields, or a refusal.

    `NISAR_L2_PR_GCOV_028_099_D_078_0005_NADV_A_20260820T015433_...` — 18
    underscore-separated fields, measured on 310 granules across six windows
    from 2025-11 to 2026-09. Field 2 is the product class (PR / UR) and field
    9 the polarimetric mode.
    """
    p = str(granule).split("_")
    if len(p) < 10:
        raise st.Refusal(f"a NISAR granule name with {len(p)} fields, not the "
                         f"18 this adapter was written against: {granule!r}")
    return p


class NisarCatalogue(st.CatalogueAdapter):
    store = "cat_nisar"
    title = ("NISAR L2 GCOV frame catalogue (L-band geocoded polarimetric "
             "covariance, ASF), one row per granule")
    licence = {
        "name": ("catalogue rows: our own metadata, CC0. The imagery they "
                 "point at is NASA/ISRO NISAR L2 GCOV (ASF DAAC) — NASA open "
                 "data, free to use and redistribute with attribution; the "
                 "product is PROVISIONAL or BETA and its calibration may "
                 "change"),
        "redistribution": "yes", "derived_works": "free",
        "attribution": ("NISAR L2 GCOV, NASA/ISRO, distributed by the Alaska "
                        "Satellite Facility DAAC; granule list from "
                        "https://api.daac.asf.alaska.edu")}
    first_year = 2025
    # A GCOV frame's own median area is what the probe measures; the nominal
    # width below is the square root of a 250 x 130 km frame.
    footprint_km = 180.0
    scene_seconds = 40.0
    SENSOR_TABLE = SENSOR_TABLE
    QC_TABLE = QC_TABLE
    window_step = "hour"
    qc_policy = (
        "qc is the processing TIER and product class: 0 BETA/PR, 1 BETA/UR, "
        "2 PROVISIONAL/PR, 3 PROVISIONAL/UR, 4 VALIDATED/PR, 5 VALIDATED/UR. "
        "The tier comes from each granule's own `collectionName` (the "
        "`collectionName` QUERY PARAMETER returns 0 results and cannot be "
        "used) and the class from the granule name's third field. The exact "
        "PGE version is counted per build under `pge_versions` rather than "
        "encoded, because it is not a uint8 vocabulary. `cloud`, `valid` and "
        "`angle` are NaN for every row: a radar does not measure cloud, no "
        "valid fraction is published, and both `offNadirAngle` and "
        "`pointingAngle` are null on every GCOV granule measured. A value "
        "outside a channel's bounds is NaN and counted (`out_of_bounds`).")
    sources = (ASF + "?dataset=NISAR&processingLevel=GCOV",
               ASF + "?dataset=NISAR&processingLevel=GCOV&output=count",
               "https://asf.alaska.edu/datasets/daac/nisar/")
    verified = (
        "2026-09-18 from the sandbox, anonymously: dataset=NISAR&"
        "processingLevel=GCOV counts 138,762 granules (platform=NISAR counts "
        "138,101, i.e. 661 fewer, so the dataset form is used); "
        "collectionName as a FILTER returns 0 while every result carries it "
        "as a FIELD; maxResults=250 works and 1000 fails; output=count "
        "answers a bare integer and is additive across windows (6,675 + "
        "23,318 = 29,993 against the month's 29,992); the record begins "
        "2025-10 with 1,719 granules (2025-09 holds none) and runs 7,697 / "
        "9,537 / 6,638 / 20,465 / 37,262 in 2025-11, 2025-12, 2026-01, "
        "2026-06 and 2026-07; two collections (BETA_V1 with PGE R05.00.8, "
        "PROVISIONAL_V1 with R05.02.3) and seven polarimetric modes (NADV, "
        "DHDH, SHNA, NASV, SVSH, DVDV, SHSH) over 310 sampled granules; "
        "offNadirAngle and pointingAngle are null on all of them")
    notes = (
        "The pixels are never copied. The record starts 2025-10 (the BETA "
        "collection), not the ledger's 2026-06; both tiers are catalogued and "
        "qc says which. The sensor code is the POLARIMETRIC MODE, which for "
        "this instrument is the thing a token builder must condition on. "
        "Urgent Response (UR) granules are catalogued beside the routine (PR) "
        "ones. Only the granule's own .h5 is in assets.parquet: the browse "
        "images and the JPL-private sidecars live under other prefixes.")
    smoke_window = ("2026-08-20", "2026-08-21")
    smoke_probe_month = "2026-08"

    def record_first(self):
        return RECORD_FIRST

    # -- the plan ----------------------------------------------------------
    def plan(self, ctx, year, month=None):
        return self.windows(ctx, year, month, step=self.window_step)

    def asf_params(self, t0, t1):
        return [("dataset", DATASET), ("processingLevel", LEVEL),
                ("start", st.utc_z(t0)), ("end", st.utc_z(t1))]

    # -- one window --------------------------------------------------------
    def scenes(self, ctx, window):
        t0, t1 = window
        counts = {}
        fet = st.Fetcher(ctx, self.store)
        what = f"{self.store} {t0:%FT%H:%M}"
        out = []
        for lo, hi, n in st.asf_windows(fet, ASF, self.asf_params, t0, t1,
                                        counts, what):
            if n == 0:
                counts["windows_empty"] = counts.get("windows_empty", 0) + 1
                continue
            for g in st.asf_window(fet, ASF, self.asf_params(lo, hi), counts,
                                   what):
                sc = self.granule(g, counts, what)
                if sc is not None:
                    out.append(sc)
        counts["requests"] = counts.get("requests", 0) + fet.requests
        counts["request_retries"] = counts.get("request_retries", 0) + \
            fet.retries
        counts["scenes"] = counts.get("scenes", 0) + len(out)
        return out, counts

    def granule(self, g, counts, what):
        gid = g.get("granuleName")
        if not gid:
            raise st.Refusal(f"{what}: an ASF result with no granuleName")
        if not g.get("startTime"):
            raise st.Refusal(f"{what}: granule {gid} has no startTime")
        t = st.seconds_of(g["startTime"])
        wkt = g.get("wkt")
        if not wkt:
            raise st.Refusal(f"{what}: granule {gid} has no wkt footprint")
        lat, lon, area = st.centre_area(st.ring_from_wkt(wkt))
        url = g.get("downloadUrl")
        if not url or PRIVATE_MARK in str(url):
            raise st.Refusal(f"{what}: granule {gid} has no public "
                             f"downloadUrl (got {url!r})")
        base, tmpl = st.asset_template([url], gid, f"{what} {gid}")
        p = name_fields(gid)
        sens = self.sensor_code(p[9], counts)
        tier = tier_of(g.get("collectionName"))
        qc = self.qc_code(f"{tier}/{p[2]}", counts)
        pv = str(g.get("pgeVersion") or "?")
        d = counts.setdefault("pge_versions", {})
        d[pv] = d.get(pv, 0) + 1
        return st.Scene(t, lat, lon,
                        [float("nan"), float("nan"), float("nan"),
                         st.area_channel(area, counts), float(sens)],
                        gid, str(g.get("collectionName") or LEVEL), base,
                        tmpl, qc)

    # -- the index ---------------------------------------------------------
    def index(self, ctx):
        fet = st.Fetcher(ctx, self.store)
        y, m = ctx.d_lo.year, ctx.d_lo.month
        if (y, m) < RECORD_FIRST:
            y, m = RECORD_FIRST
        a = dt.datetime(y, m, 1)
        b = dt.datetime(y + (m == 12), m % 12 + 1, 1)
        total = st.asf_count(fet, ASF, [("dataset", DATASET),
                                        ("processingLevel", LEVEL)])
        month = st.asf_count(fet, ASF, self.asf_params(a, b))
        by_platform = st.asf_count(fet, ASF, [("platform", DATASET),
                                              ("processingLevel", LEVEL)])
        if not total:
            raise st.Refusal(
                f"{self.store}: ASF counts ZERO GCOV granules at all — "
                f"the dataset or processingLevel parameter has changed "
                f"(the 2026-09-14 rule).")
        if not month:
            raise st.Refusal(
                f"{self.store}: ASF counts ZERO GCOV granules in "
                f"{y:04d}-{m:02d} while the whole archive holds {total}. A "
                f"build window with no data in it is a mis-set --start, not "
                f"a measurement.")
        one = None
        tiers, modes, pges = {}, {}, {}
        got = st.asf_window(fet, ASF, self.asf_params(
            a, min(b, a + dt.timedelta(hours=1))), {}, "index")
        for g in got:
            tiers[tier_of(g.get("collectionName"))] = \
                tiers.get(tier_of(g.get("collectionName")), 0) + 1
            p = name_fields(g["granuleName"])
            modes[p[9]] = modes.get(p[9], 0) + 1
            pv = str(g.get("pgeVersion") or "?")
            pges[pv] = pges.get(pv, 0) + 1
            if one is None:
                c2 = {}
                sc = self.granule(g, c2, "index")
                one = {"granule": sc.stac_id, "collection": sc.collection,
                       "time_s": sc.t, "lat": sc.lat, "lon": sc.lon,
                       "log2_area": sc.values[3], "sensor": sc.values[4],
                       "qc": sc.qc, "base_url": sc.base_url,
                       "asset_template": sc.asset_set,
                       "off_nadir_angle": g.get("offNadirAngle"),
                       "pointing_angle": g.get("pointingAngle"),
                       "polarization_field": g.get("polarization"),
                       "counts": c2}
        return {
            "dataset": "NISAR L2 GCOV (ASF DAAC) via services/search/param",
            "url": ASF,
            "query": {"dataset": DATASET, "processingLevel": LEVEL},
            "granules_total": total,
            "granules_total_by_platform_param": by_platform,
            "platform_vs_dataset_note": (
                "dataset=NISAR counts more granules than platform=NISAR "
                f"({total} vs {by_platform}); the dataset form is used. "
                "collectionName as a FILTER returns 0 and is read off each "
                "granule instead."),
            "max_results": st.ASF_MAX,
            "window_step": self.window_step,
            "years": [int(v) for v in ctx.years],
            "first_month": f"{y:04d}-{m:02d}",
            "granules_first_month": month,
            "tiers_first_hour": tiers,
            "pol_modes_first_hour": modes,
            "pge_versions_first_hour": pges,
            "sensor_table": {str(k): v for k, v in SENSOR_TABLE.items()},
            "qc_table": {str(k): v for k, v in QC_TABLE.items()},
            "first_record": one,
        }

    def smoke_sources(self, root, d_lo, d_hi, seed=20260918):
        from family1.adapters import _cat_smoke as cs
        return cs.asf_sources(self, root, d_lo, d_hi, cs.NISAR_GRANULES)


ADAPTER = NisarCatalogue

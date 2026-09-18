"""ECOSTRESS L2T LSTE — 70 m thermal tiles from the Space Station.

PLAIN ENGLISH. ECOSTRESS is a thermal camera bolted to the International
Space Station. Because the station's orbit is not sun-synchronous it sees the
same place at a different hour each time, which is the point: it measures how
hot the land gets at every time of day. The tiled Level-2 product is one 70 m
land-surface-temperature tile per overpass per grid square. This store is the
LIST of those tiles — when, where, how large, and which processing version.
The pixels, about 10 TB a year, stay at the LP DAAC.

SOURCE, VERIFIED 2026-09-18 FROM THIS SANDBOX (anonymous search):
  https://cmr.earthdata.nasa.gov/search/granules.umm_json
      ?short_name=ECO_L2T_LSTE&version=<002|003>&temporal=<from>,<to>
  provider LPCLOUD, collections ECO_L2T_LSTE v002 and v003

TWO VERSIONS, AND THEY ARE NOT THE SAME SCENES. CMR holds
**15,616,308** v002 granules and **3,908,292** v003 — 19,524,600 together,
against the note's estimate of "1.37 M granules in 2024" a year. Both
collections declare 2018-07-09 as their start. They do NOT currently overlap
in time: 2024-06-01 has 3,140 v002 granules and 0 v003, while 2026-09-01 has
787 v002 and 411 v003 — so v003 is a newer processing that has reached the
recent record and is working backwards. Both are catalogued, and the
PROCESSING VERSION IS `qc` (2 or 3), which is exactly what `qc` is for in a
tier-T catalogue: a scene reprocessed under a new baseline is a different
product of the same overpass, and a consumer that wants one takes the rows
with that qc. The index measures the per-version counts of the run's own
first month so the overlap is on the record rather than assumed.

WHAT A ROW IS.
  time_s    the granule's `BeginningDateTime`, int32 seconds since 1982.
  lat, lon  the centre of the granule's `BoundingRectangles` footprint on the
            sphere. ECOSTRESS publishes a RECTANGLE, not a polygon (measured:
            `SpatialExtent.HorizontalSpatialDomain.Geometry.BoundingRectangles`
            with no `GPolygons`), so the centre is the rectangle's and the
            area is the rectangle's area on the sphere — which for a tiled
            product is the tile, so the two agree.
  platform  platform_hash(the granule UR, e.g.
            `ECOv003_L2T_LSTE_46506_024_02FML_20260917T205540_01`).
  values    cloud   NaN — the producer ships a `cloud.tif` MASK rather than a
                    scene cloud fraction, and a number read out of a raster
                    is not metadata this store may invent
            valid   NaN — none is published
            angle   NaN — no mean view or solar angle is published
            log2_area
                    log2 of the footprint's area on the sphere in km²
            sensor  1 — ISS / ECOSTRESS (PHyTIR); one instrument, one code
  qc        2 = v002, 3 = v003 (the processing version).

MEASURED, 2026-09-18: 3,140 granules on 2024-06-01 (v002) and 787 + 411 on
2026-09-01; a granule is 19,080 bytes in `umm_json` against 22,403 in the
`.json` feed — ECOSTRESS publishes 121 related URLs, which is why this is the
heaviest of the eight catalogues to LIST and why it is listed in year lanes.

"No cadence; catalogued only" is the note's own description of this store, and
three of its five channels are NaN for that reason. What it is FOR is the row:
a model asking "is there a 70 m thermal image of this cell this week, and at
what hour" gets its answer here without anybody moving a terabyte.
"""
import datetime as dt

import numpy as np

import build_family10_stores as f10b
from family1.adapters import _stac as st

SENSOR_TABLE = {1: "ISS/ECOSTRESS"}
QC_TABLE = {2: "002", 3: "003"}
COLLECTIONS = (("v002", "ECO_L2T_LSTE", "002"),
               ("v003", "ECO_L2T_LSTE", "003"))
RECORD_FIRST = (2018, 7)


class EcostressCatalogue(st.CmrCatalogue):
    store = "cat_ecostress"
    title = ("ECOSTRESS L2T LSTE tile catalogue (70 m land-surface "
             "temperature from the ISS, v002 and v003), one row per granule")
    licence = {
        "name": ("catalogue rows: our own metadata, CC0. The imagery they "
                 "point at is NASA ECOSTRESS L2T LSTE (LP DAAC) — NASA open "
                 "data, free to use and redistribute with attribution"),
        "redistribution": "yes", "derived_works": "free",
        "attribution": ("ECOSTRESS L2T LSTE, NASA LP DAAC; granule list from "
                        "NASA's Common Metadata Repository, "
                        "https://cmr.earthdata.nasa.gov")}
    first_year = 2018
    footprint_km = 109.8           # the L2T tile
    scene_seconds = 1.0
    COLLECTIONS = COLLECTIONS
    SENSOR_TABLE = SENSOR_TABLE
    QC_TABLE = QC_TABLE
    window_step = "day"
    qc_policy = (
        "qc is the ECOSTRESS processing version: 2 = v002, 3 = v003. The two "
        "versions are separate CMR collections over the same mission, so one "
        "overpass can appear twice with different qc — that is a "
        "reprocessing, not a duplicate, and a consumer selects on qc. A "
        "version the table does not list is 255 and counted by name in "
        "`qc_unlisted`. `cloud`, `valid` and `angle` are NaN for every row: "
        "the producer ships a cloud RASTER rather than a scene cloud "
        "fraction, and publishes no valid fraction and no mean angle. A value "
        "outside a channel's bounds is NaN and counted (`out_of_bounds`).")
    sources = (st.CMR_GRANULES,
               "https://cmr.earthdata.nasa.gov/search/collections.json"
               "?short_name=ECO_L2T_LSTE",
               "https://lpdaac.usgs.gov/products/eco_l2t_lstev002/")
    verified = (
        "2026-09-18 from the sandbox, anonymously: both collections exist "
        "(ECO_L2T_LSTE v002 and v003, LPCLOUD, both declaring 2018-07-09); "
        "CMR-Hits 15,616,308 for v002 and 3,908,292 for v003; 3,140 v002 and "
        "0 v003 granules on 2024-06-01, 787 v002 and 411 v003 on 2026-09-01; "
        "the UMM-G granule carries BoundingRectangles (never GPolygons), no "
        "cloud, valid or angle attribute, and 121 related URLs — 19,080 bytes "
        "in umm_json against 22,403 in the .json feed")
    notes = (
        "The pixels are never copied. BOTH processing versions are "
        "catalogued and qc says which; they do not overlap in time today "
        "(v003 is working backwards from the present) and the index measures "
        "the run's own first month. The footprint is a rectangle, not a "
        "polygon. No cadence: the ISS orbit drifts through the day, which is "
        "what the store is for.")
    smoke_window = ("2024-06-01", "2024-06-02")
    smoke_probe_month = "2024-06"

    def record_first(self):
        return RECORD_FIRST

    def qc_text(self, gid, coll, umm, aa):
        return str(coll[2])

    def values(self, gid, coll, umm, aa, area, sensor, counts):
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
        out = {"dataset": "ECOSTRESS L2T LSTE (LP DAAC) via CMR granule "
                          "search",
               "url": st.CMR_GRANULES, "page_size": self.page_size,
               "years": [int(v) for v in ctx.years], "collections": {},
               "first_month": f"{y:04d}-{m:02d}"}
        total = 0
        first = None
        for coll in COLLECTIONS:
            c, _ = fet.get(
                "https://cmr.earthdata.nasa.gov/search/collections.json"
                f"?short_name={coll[1]}&version={coll[2]}&page_size=5")
            ents = (c.get("feed") or {}).get("entry") or []
            if not ents:
                raise st.Refusal(f"CMR lists no collection {coll[1]} "
                                 f"v{coll[2]} — the store's source has moved")
            n_month = st.cmr_hits(fet, st.CMR_GRANULES,
                                  self.cmr_params(a, b, coll))
            total += n_month
            out["collections"][coll[0]] = {
                "short_name": coll[1], "version": coll[2],
                "dataset_id": ents[0].get("dataset_id"),
                "time_start": ents[0].get("time_start"),
                "concept_id": ents[0].get("id"),
                "granules_total": st.cmr_hits(
                    fet, st.CMR_GRANULES,
                    [("short_name", coll[1]), ("version", coll[2])]),
                "granules_first_month": n_month}
            if first is None and n_month:
                d, _ = fet.get(st.cmr_url(st.CMR_GRANULES,
                                          self.cmr_params(a, b, coll), 1))
                items = d.get("items") or []
                if items:
                    c2 = {}
                    sc = self.granule(items[0], coll, c2, "index")
                    first = {"collection": coll[0], "id": sc.stac_id,
                             "time_s": sc.t, "lat": sc.lat, "lon": sc.lon,
                             "log2_area": sc.values[3], "qc": sc.qc,
                             "asset_dir": sc.base_url,
                             "asset_template": sc.asset_set, "counts": c2}
        if not total:
            raise st.Refusal(
                f"{self.store}: CMR counts ZERO ECOSTRESS tiles in "
                f"{y:04d}-{m:02d} across both versions — a broken query, not "
                f"a measurement (the 2026-09-14 rule).")
        out["granules_first_month_total"] = total
        out["first_record"] = first
        out["sensor_table"] = {str(k): v for k, v in SENSOR_TABLE.items()}
        out["qc_table"] = {str(k): v for k, v in QC_TABLE.items()}
        return out

    def smoke_sources(self, root, d_lo, d_hi, seed=20260918):
        from family1.adapters import _cat_smoke as cs
        return cs.cmr_sources(self, root, d_lo, d_hi, cs.ECO_GRANULES)


ADAPTER = EcostressCatalogue

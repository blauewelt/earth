"""Every Landsat Collection 2 Level-2 scene since 1982, one row per scene.

PLAIN ENGLISH. The USGS has photographed the whole land surface every 16 days
since 1982 with five successive cameras, and every one of those pictures is
free. This store is the LIST of them: one row per scene with the second it was
taken, where the footprint's centre is, how cloudy it was, how high the sun
stood, how large the footprint is and which camera on which satellite took it.
The pixels — 0.3 to 0.5 petabytes a year at today's rate — stay at the USGS,
and `assets.parquet` says exactly which files a consumer would fetch.

SOURCE, VERIFIED 2026-09-18 FROM THIS SANDBOX (no account, US government):
  https://landsatlook.usgs.gov/stac-server                    (the STAC API)
  https://landsatlook.usgs.gov/stac-server/search             (POST)
  collection `landsat-c2l2-sr`, temporal extent 1982-08-22 -> open
The server is a `stac-server` with CURSOR paging: a `next` link carries a
POST body with a `next` token of `"<datetime>,<id>,<collection>"`. Measured
limits: `limit=100` answers in 0.5-1.2 s; `limit=250` and above answer
**HTTP 502** — the gateway, not the query — so 100 is the page size. Eight
day-windows in flight measured **859 scenes/s at 6.7 MB/s**, against 123/s
with one, and every window's walk equalled the server's own `numberMatched`
exactly (22,629 over sixteen days of June 2024).

WHICH COLLECTION, AND THE CROSS-CHECK. `landsat-c2l2-sr` is the Level-2
surface-reflectance product and holds every sensor: TM (Landsat 4, 5), ETM+
(7), OLI/TIRS (8, 9). The sibling `landsat-c2l2-st` (surface temperature) is
the SAME scenes minus the ones with no usable thermal band, and Earth Search's
`landsat-c2-l2` on AWS is one item per scene carrying both. Monthly counts
measured 2026-09-18:

  month     c2l2-sr   c2l2-st   earth-search landsat-c2-l2
  1982-09        13        13        13
  1985-06     5,357     5,349     5,357
  1990-06    12,422    12,117    12,422
  1995-06    13,827    13,449    13,830
  1999-08    22,182    21,413    22,227
  2005-06    20,153    19,898    20,153
  2013-06    26,715    26,215    26,715
  2020-06    34,700    33,538    34,700
  2024-06    42,339    39,961    42,390
  2026-06    42,523    40,098    42,561
  2026-08    40,965    38,577    40,963

So the two independent catalogues of the same archive agree to within 0.12 %
(Earth Search is 0 to 51 scenes ahead in the recent months and 2 behind in
2026-08). The index records both numbers for the store's first year, and the
probe records both for its month; a disagreement larger than 1 % is reported,
because it would mean one of the two catalogues is missing scenes.

WHAT A ROW IS.
  time_s    `properties.datetime`, the acquisition instant, int32 seconds
            since 1982-01-01. The record starts 1982-08-22, so int32 is
            enough and the store is schema 2.
  lat, lon  the spherical mean of the scene's own footprint polygon
            (`geometry`), not the bbox centre: a Landsat scene is a
            parallelogram rotated by the orbit, and near the poles its bbox
            centre can sit 40 km from the footprint's.
  platform  platform_hash(the STAC item id, e.g.
            `LC09_L2SP_095022_20240630_20240702_02_T2_SR`).
  values    cloud   `eo:cloud_cover`, per cent (the whole scene, not
                    `landsat:cloud_cover_land`)
            valid   NaN — the USGS publishes no valid-pixel fraction in the
                    STAC item, and a fraction this store invented would be
                    worse than a gap
            angle   the SUN ZENITH, 90 - `view:sun_elevation`
            log2_area
                    log2 of the footprint polygon's own area on the
                    sphere in km² — log2 because `values` is float16 (see
                    `_stac.CAT_CHANNELS`); the area is 2**log2_area
            sensor  4/5/7/8/9 — the satellite, which fixes the camera
  qc        the processing level and tier: L2SP tier 1 / tier 2 / real-time,
            L2SR tier 1 / tier 2 (a scene with no usable thermal band is
            L2SR). `QC_TABLE` is the whole vocabulary and store.json carries
            it.

ASSETS. Every asset of a scene lives in ONE directory and is named
`<scene>_<suffix>`, verified on 360 scenes across six eras (1985, 1999, 2003,
2013, 2024, 2026) and all three sensor families: 360 of 360 matched, with
exactly TWO suffix sets — sixteen files for TM/ETM+ (`SR_B1`..`SR_B7`,
`SR_ATMOS_OPACITY`, `SR_CLOUD_QA`, the QA bands and the four metadata files)
and sixteen for OLI/TIRS (`SR_B1`..`SR_B7` with a different band mapping plus
`SR_QA_AEROSOL`). The `index` asset is the exception: it points at
`stac-browser`, not `data`, and is a web page rather than a file, so it is not
in the table. A scene whose asset set matches NEITHER of the two is a refusal
naming what differs both ways, because an `assets.parquet` row pointing at a
file that is not there would be worse than no row; `ASSET_SETS` is that
verification table and store.json carries it.

MEMORY. One day-window per worker (1,400 scenes of parsed JSON, about 11 MB),
eight workers, and the year's packed rows in batches of 500,000.
"""
import datetime as dt
import math
import sys

import numpy as np

import build_family10_stores as f10b
from family1.adapters import _stac as st

BASE = "https://landsatlook.usgs.gov/stac-server"
SEARCH = BASE + "/search"
COLLECTION = "landsat-c2l2-sr"
ES_SEARCH = "https://earth-search.aws.element84.com/v1/search"
ES_COLLECTION = "landsat-c2-l2"
PAGE = 100                     # measured: limit=250 answers HTTP 502
RECORD_FIRST = (1982, 8)

# WHAT THE WALK ASKS FOR. `fields` must carry BOTH halves, and this is a trap
# worth the paragraph: an `exclude`-only request that names a DOTTED path
# (`assets.*.file:checksum`) makes this stac-server drop every `properties`
# key but `datetime` — measured 2026-09-18, 27 properties down to 1, with no
# error and no warning. A store built from that request would have had a
# platform of None, a sun elevation of None and a processing tier of None on
# every row: all five channels NaN or 255, and nothing to say so. So the
# request `include`s the six properties by name AND `exclude`s the asset
# sub-keys, which is also the smallest feature this server will serve:
# 9,114 bytes against 24,524 unfiltered (measured on the same item).
INCLUDE = ["id", "bbox", "geometry",
           "properties.datetime", "properties.eo:cloud_cover",
           "properties.platform", "properties.view:sun_elevation",
           "properties.landsat:correction",
           "properties.landsat:collection_category", "assets"]
EXCLUDE = ["assets.*.file:checksum", "assets.*.alternate", "assets.*.eo:bands",
           "assets.*.description", "assets.*.title", "assets.*.roles",
           "assets.*.type", "assets.*.classification:bitfields",
           "assets.*.gsd", "assets.*.proj:shape", "assets.*.proj:transform",
           "links", "stac_extensions", "description"]
FIELDS = {"include": INCLUDE, "exclude": EXCLUDE}
#: every property a row needs; the index REFUSES if the server drops one
NEED_PROPS = ("datetime", "eo:cloud_cover", "platform", "view:sun_elevation",
              "landsat:correction", "landsat:collection_category")

SENSOR_TABLE = {4: "LANDSAT_4", 5: "LANDSAT_5", 7: "LANDSAT_7",
                8: "LANDSAT_8", 9: "LANDSAT_9"}
QC_TABLE = {0: "L2SP/T1", 1: "L2SP/T2", 2: "L2SP/RT",
            3: "L2SR/T1", 4: "L2SR/T2", 5: "L2SR/RT"}
CORRECTIONS = ("L2SP", "L2SR")
TIERS = ("T1", "T2", "RT")
DATA_HOST = "https://landsatlook.usgs.gov/data/"
BROWSER_HOST = "https://landsatlook.usgs.gov/stac-browser/"


def qc_text(correction, tier):
    return f"{correction}/{tier}"


def scene_id(item_id):
    """The STAC item id minus the product suffix — the scene's own name."""
    return item_id[:-3] if item_id.endswith("_SR") else item_id


class LandsatCatalogue(st.CatalogueAdapter):
    store = "cat_landsat"
    title = ("Landsat Collection 2 Level-2 scene catalogue, all sensors "
             "(TM, ETM+, OLI/TIRS), 1982 onward — one row per scene")
    licence = {
        "name": ("catalogue rows: our own metadata, CC0. The imagery they "
                 "point at is a US government work (USGS Landsat "
                 "Collection 2), public domain, no restriction"),
        "redistribution": "yes", "derived_works": "free",
        "attribution": ("Landsat Collection 2 Level-2 data courtesy of the "
                        "U.S. Geological Survey; scene list from "
                        "https://landsatlook.usgs.gov/stac-server")}
    first_year = 1982
    footprint_km = 185.0       # a WRS-2 scene is 185 km across track
    scene_seconds = 24.0       # the ~24 s a scene takes to image
    SENSOR_TABLE = SENSOR_TABLE
    QC_TABLE = QC_TABLE
    ASSET_SETS = {
        # verified on 360 scenes, 2026-09-18; `{id}` is the STAC item id's
        # scene name (the item id without its `_SR` suffix)
        "tm": ["{id}_ANG.txt", "{id}_MTL.json", "{id}_MTL.txt",
               "{id}_MTL.xml", "{id}_SR_ATMOS_OPACITY.TIF",
               "{id}_SR_B1.TIF", "{id}_SR_B2.TIF", "{id}_SR_B3.TIF",
               "{id}_SR_B4.TIF", "{id}_SR_B5.TIF", "{id}_SR_B7.TIF",
               "{id}_SR_CLOUD_QA.TIF", "{id}_QA_PIXEL.TIF",
               "{id}_QA_RADSAT.TIF", "{id}_thumb_large.jpeg",
               "{id}_thumb_small.jpeg"],
        "oli-tirs": ["{id}_ANG.txt", "{id}_MTL.json", "{id}_MTL.txt",
                     "{id}_MTL.xml", "{id}_SR_B1.TIF", "{id}_SR_B2.TIF",
                     "{id}_SR_B3.TIF", "{id}_SR_B4.TIF", "{id}_SR_B5.TIF",
                     "{id}_SR_B6.TIF", "{id}_SR_B7.TIF",
                     "{id}_SR_QA_AEROSOL.TIF", "{id}_QA_PIXEL.TIF",
                     "{id}_QA_RADSAT.TIF", "{id}_thumb_large.jpeg",
                     "{id}_thumb_small.jpeg"],
    }
    qc_policy = (
        "qc is the USGS processing level and collection tier: 0 L2SP/T1, "
        "1 L2SP/T2, 2 L2SP/RT, 3 L2SR/T1, 4 L2SR/T2, 5 L2SR/RT (L2SR is a "
        "scene with no usable thermal band; RT is real-time, reprocessed "
        "later into T1 or T2). Anything else is 255 and counted by name in "
        "`qc_unlisted`. `valid` is NaN for every row: the STAC item carries "
        "no valid-pixel fraction. `cloud` is `eo:cloud_cover` and is NaN "
        "where the producer reports -1 (not computed); a value outside "
        "[0, 100] is NaN and counted (`out_of_bounds`). `angle` is the sun "
        "ZENITH, 90 - view:sun_elevation, NaN where the elevation is "
        "missing or negative (a night-time or terminator scene).")
    sources = (SEARCH, BASE + f"/collections/{COLLECTION}", ES_SEARCH)
    verified = (
        "2026-09-18 from the sandbox: the collection list (18 collections, "
        "landsat-c2l2-sr from 1982-08-22, open-ended); numberMatched for "
        "eleven months from 1982-09 to 2026-08 against landsat-c2l2-st and "
        "against Earth Search's landsat-c2-l2 (they agree to 0.12 %); "
        "limit=100 works and limit>=250 answers HTTP 502; cursor paging over "
        "sixteen day-windows of June 2024 walked 22,629 items and the server "
        "counted 22,629; the asset URL layout on 360 items across six eras "
        "and three sensor families (360 of 360 in one directory named after "
        "the scene, two distinct suffix sets)")
    notes = (
        "The pixels are never copied. One row per Level-2 scene; the sibling "
        "surface-temperature collection is the same scenes and is not a "
        "second row. The `index` asset points at the STAC browser rather "
        "than at data and is not in assets.parquet. Earth Search's "
        "landsat-c2-l2 is the independent count, recorded in the index and "
        "in the probe.")
    smoke_window = ("2024-06-01", "2024-06-03")
    smoke_probe_month = "2024-06"

    # -- the plan ----------------------------------------------------------
    def plan(self, ctx, year, month=None):
        """One window per DAY. A 2026 day holds 1,362 scenes: 14 pages."""
        return self.windows(ctx, year, month, step="day")

    def record_first(self):
        return RECORD_FIRST

    # -- one window --------------------------------------------------------
    def scenes(self, ctx, window):
        t0, t1 = window
        counts = {}
        fet = st.Fetcher(ctx, self.store)
        body = {"collections": [COLLECTION],
                "datetime": f"{st.utc_z(t0)}/"
                            f"{st.utc_z(t1 - dt.timedelta(seconds=1))}",
                "fields": FIELDS}
        out = []
        what = f"{self.store} {t0:%F}"
        for f in st.stac_search(fet, SEARCH, body, self.page_for(ctx, PAGE),
                                counts, what):
            sc = self._scene(f, counts, what)
            if sc is not None:
                out.append(sc)
        counts["requests"] = counts.get("requests", 0) + fet.requests
        counts["request_retries"] = counts.get("request_retries", 0) + \
            fet.retries
        counts["scenes"] = counts.get("scenes", 0) + len(out)
        return out, counts

    def _scene(self, f, counts, what):
        p = f.get("properties") or {}
        iid = f.get("id")
        if not iid:
            raise st.Refusal(f"{what}: a STAC item with no id")
        if not p.get("datetime"):
            raise st.Refusal(f"{what}: item {iid} carries no "
                             f"properties.datetime")
        t = st.seconds_of(p["datetime"])
        geom = f.get("geometry")
        if geom:
            lat, lon, area = st.centre_area(st.ring_from_geojson(geom))
        elif f.get("bbox"):
            w, s, e, n = f["bbox"][:4]
            lat, lon, area = st.centre_area([([s, s, n, n], [w, e, e, w])])
            counts["footprint_from_bbox"] = \
                counts.get("footprint_from_bbox", 0) + 1
        else:
            raise st.Refusal(f"{what}: item {iid} has neither geometry nor "
                             f"bbox")
        elev = p.get("view:sun_elevation")
        zen = float("nan")
        if elev is not None:
            try:
                e_ = float(elev)
                zen = 90.0 - e_ if e_ > 0 else float("nan")
            except (TypeError, ValueError):
                zen = float("nan")
        if not np.isfinite(zen):
            counts["sun_elevation_missing"] = \
                counts.get("sun_elevation_missing", 0) + 1
        plat = str(p.get("platform") or "")
        sens = self.sensor_code(plat, counts)
        corr = str(p.get("landsat:correction") or "")
        tier = str(p.get("landsat:collection_category") or "")
        qc = self.qc_code(qc_text(corr, tier), counts)
        sid = scene_id(iid)
        base, aset = self._assets(f, sid, p, counts, what)
        return st.Scene(t, lat, lon,
                        [st.clamp_pct(p.get("eo:cloud_cover")),
                         float("nan"), zen, st.area_channel(area, counts),
                         float(sens)],
                        iid, COLLECTION, base, aset, qc)

    def _assets(self, f, sid, p, counts, what):
        """The scene's asset directory and which suffix set it carries.

        The producer's OWN hrefs are read (nothing is derived from the scene
        name): every data asset must sit in one directory whose last element
        is the scene name, and the set of file names must be one of the two
        `ASSET_SETS`. A scene that satisfies neither is a refusal — an
        `assets.parquet` row that pointed at a file that is not there would be
        worse than no row.
        """
        assets = f.get("assets") or {}
        if not assets:
            raise st.Refusal(f"{what}: item {f.get('id')} carries no assets")
        dirs, names = set(), set()
        for k, a in assets.items():
            href = (a or {}).get("href") or ""
            if not href or href.startswith(BROWSER_HOST):
                continue                      # the `index` web page
            d, _, nm = href.rpartition("/")
            dirs.add(d)
            names.add(nm)
        if len(dirs) != 1:
            raise st.Refusal(f"{what}: item {f.get('id')} spreads its assets "
                             f"over {len(dirs)} directories: "
                             f"{sorted(dirs)[:3]}")
        base = dirs.pop()
        if not base.endswith("/" + sid):
            raise st.Refusal(f"{what}: item {f.get('id')}'s asset directory "
                             f"{base!r} does not end in the scene name "
                             f"{sid!r}")
        want = {k: {n.replace("{id}", sid) for n in v}
                for k, v in self.ASSET_SETS.items()}
        for k, v in want.items():
            if names == v:
                # the TEMPLATE, not the set's name: assets.parquet holds the
                # same kind of string for all eight catalogues, and parquet's
                # dictionary encoding collapses the two distinct values of
                # this column to two dictionary entries for the whole archive
                return base, " ".join(sorted(self.ASSET_SETS[k]))
        # not one of the two known sets: name what differs, both ways
        best = min(want, key=lambda k: len(names ^ want[k]))
        raise st.Refusal(
            f"{what}: item {f.get('id')} carries an asset set this adapter "
            f"does not know. Closest is {best!r}; extra "
            f"{sorted(names - want[best])[:6]}, missing "
            f"{sorted(want[best] - names)[:6]}. Add the set to "
            f"cat_landsat.ASSET_SETS after checking it on real scenes.")

    # -- the index ---------------------------------------------------------
    def index(self, ctx):
        fet = st.Fetcher(ctx, self.store)
        coll, _ = fet.get(f"{BASE}/collections/{COLLECTION}")
        interval = (((coll.get("extent") or {}).get("temporal") or {})
                    .get("interval") or [[None, None]])[0]
        if not interval or not interval[0]:
            raise st.Refusal(f"{COLLECTION} declares no temporal extent")
        start = st.parse_iso(interval[0])
        if (start.year, start.month) != RECORD_FIRST:
            print(f"  ::warning::{COLLECTION} now starts {start:%F}, and this "
                  f"adapter's RECORD_FIRST is {RECORD_FIRST}")
        # THE MONTH THE RUN ACTUALLY STARTS IN, clipped up to the record's
        # own first month — never January by default, so a lane over 2019
        # cross-checks 2019-01 and a two-day smoke cross-checks its own month.
        y, m = ctx.d_lo.year, ctx.d_lo.month
        if (y, m) < RECORD_FIRST:
            y, m = RECORD_FIRST
        mine, theirs, first = self._cross_check(ctx, fet, y, m)
        return {
            "dataset": "Landsat Collection 2 Level-2 (USGS STAC)",
            "url": SEARCH,
            "collection": COLLECTION,
            "collection_extent": interval,
            "page_size": PAGE,
            "page_size_note": "limit>=250 answers HTTP 502 (measured "
                              "2026-09-18); 100 is the page size",
            "years": [int(v) for v in ctx.years],
            "cross_check_month": f"{y:04d}-{m:02d}",
            "cross_check": {
                "usgs_landsatlook_numberMatched": mine,
                "earth_search_landsat_c2_l2_numberMatched": theirs,
                "difference": (None if None in (mine, theirs)
                               else int(theirs) - int(mine))},
            "sensor_table": {str(k): v for k, v in SENSOR_TABLE.items()},
            "qc_table": {str(k): v for k, v in QC_TABLE.items()},
            "first_record": first,
        }

    def _cross_check(self, ctx, fet, y, m):
        """The month's count from BOTH catalogues, and one real item."""
        a = dt.datetime(y, m, 1)
        b = dt.datetime(y + (m == 12), m % 12 + 1, 1)
        rng = f"{st.utc_z(a)}/{st.utc_z(b - dt.timedelta(seconds=1))}"
        d, _ = fet.post(SEARCH, {"collections": [COLLECTION],
                                 "datetime": rng, "limit": 1,
                                 "fields": FIELDS})
        mine = d.get("numberMatched")
        first = None
        if d.get("features"):
            f = d["features"][0]
            p = f.get("properties") or {}
            # THE FIELD SELECTION, CHECKED ON A REAL ITEM BEFORE THE FETCH.
            # An `exclude`-only request silently strips every property but
            # `datetime` on this server, so the one thing that turns this
            # store into a column of NaN is asserted where the inputs are all
            # it has cost (ml/CLAUDE.md §0.3, §5.16).
            miss = [k for k in NEED_PROPS if k not in p]
            if miss:
                raise st.Refusal(
                    f"{SEARCH} answered item {f.get('id')} without "
                    f"{miss} — the `fields` selection is dropping properties "
                    f"this store's channels come from. Every row would carry "
                    f"NaN or 255 there. Fix cat_landsat.FIELDS.")
            sid = scene_id(f["id"])
            base, aset = self._assets(f, sid, p, {}, "index")
            first = {"id": f["id"], "datetime": p.get("datetime"),
                     "platform": p.get("platform"),
                     "instruments": p.get("instruments"),
                     "correction": p.get("landsat:correction"),
                     "tier": p.get("landsat:collection_category"),
                     "cloud": p.get("eo:cloud_cover"),
                     "sun_elevation": p.get("view:sun_elevation"),
                     "asset_dir": base, "asset_set": aset}
        theirs = None
        if not fet.source_dir:
            try:
                e, _ = fet.post(ES_SEARCH, {"collections": [ES_COLLECTION],
                                            "datetime": rng, "limit": 1})
                theirs = e.get("numberMatched")
            except (st.Refusal, IOError) as ex:
                print(f"  ::warning::the Earth Search cross-check did not "
                      f"answer ({ex}); the USGS count stands alone")
        if None not in (mine, theirs) and mine and \
                abs(int(theirs) - int(mine)) > 0.01 * int(mine):
            print(f"  ::warning::the two catalogues of the same archive "
                  f"disagree by more than 1 %: landsatlook {mine}, "
                  f"Earth Search {theirs} for {y:04d}-{m:02d}")
        return mine, theirs, first

    # -- the smoke ---------------------------------------------------------
    def smoke_sources(self, root, d_lo, d_hi, seed=20260918):
        from family1.adapters import _cat_smoke as cs
        return cs.landsat_sources(self, root, d_lo, d_hi, seed)


ADAPTER = LandsatCatalogue

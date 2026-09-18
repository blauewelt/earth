"""VIIRS I-band L1B — the 375 m imaging swaths, one row per six-minute granule.

PLAIN ENGLISH. VIIRS is the imaging radiometer on the American polar weather
satellites. Its five "imagery" bands see 375 m pixels across a 3,000 km swath,
and the archive cuts the orbit into six-minute granules — 242 a day per
satellite, all day and all night, since 2012. This store is the LIST of those
granules: when, where the swath's centre is, how large it is, and which
satellite flew it. The radiances themselves are about 15 TB a year per
satellite and stay at LAADS.

SOURCE, VERIFIED 2026-09-18 FROM THIS SANDBOX (anonymous search):
  https://cmr.earthdata.nasa.gov/search/granules.umm_json
      ?short_name=VNP02IMG&temporal=<from>,<to>&page_size=2000
  LAADS collections VNP02IMG v2, VJ102IMG v2.1, VJ202IMG v2.1

THREE SATELLITES, NOT TWO — A LEDGER CORRECTION. The note's table names
VNP02IMG and VJ102IMG and dates the record "2012 / 2018 →". A collection
search on 2026-09-18 finds a THIRD operational I-band L1B collection of the
same product family: **VJ202IMG v2.1, from JPSS-2 / NOAA-21, 2023-02-10 →**,
with the same 242 granules a day. It is catalogued here, with its own sensor
code, because the rule is that nothing is hard-coded that the archive can
tell you (ADAPTER_CONTRACT rule 4) and a store that listed two of the three
satellites would be silently short by a third of the recent record. The NRT
(near-real-time) twins under the LANCEMODIS provider are NOT catalogued: they
are the same overpasses served faster and less calibrated, so they would be
duplicate rows.

MEASURED, 2026-09-18 (`CMR-Hits`):
  VNP02IMG total 1,267,013 granules, 2012-01-19T00:00 → 2026-09-17T11:48
  242 granules on 2013-06-01, on 2020-06-01 and on 2024-06-10
  monthly counts 2023-2026 run 6,476 .. 7,441 — a full month is 7,440
  (31 × 240) and the shortfalls are REAL instrument gaps: 2024-06 is 6,476
  and 2024-06-01 alone holds none at all, which is why an empty day-window
  is counted (`windows_empty`) and is not a refusal here, while an empty
  MONTH still is.
Payload: **2,480 bytes a granule** in `umm_json` against 14,995 in the
`.json` feed — VIIRS granules carry four related URLs, so the UMM form is six
times leaner and is what this store walks.

WHAT A ROW IS.
  time_s    the granule's `BeginningDateTime`, int32 seconds since 1982.
  lat, lon  the spherical mean of the granule's own `GPolygons` boundary —
            a six-minute swath is a long curved quadrilateral and its
            bounding-box centre would be far off it.
  platform  platform_hash(the granule UR, e.g. `LAADS:8280371664`). NOTE
            that LAADS's granule UR is an OPAQUE ARCHIVE NUMBER, not the file
            name; the file name (`VNP02IMG.A2024161.0000.002.…nc`) is in
            `assets.parquet`'s template, which is where a consumer reads it.
  values    cloud   NaN — an L1B radiance granule carries no cloud fraction
            valid   NaN — none is published either
            angle   NaN — no mean solar or view angle is published for an
                    I-band granule (the per-pixel angles are in the
                    geolocation product, VNP03IMG, and are not metadata)
            log2_area
                    log2 of the boundary's area on the sphere in km² —
                    a six-minute granule is about 7 x 10**6 km², which
                    OVERFLOWS float16 in km² and is why the channel is log2
            sensor  1 Suomi-NPP, 2 NOAA-20, 3 NOAA-21
  qc        the collection version: 0 = v2 (Suomi-NPP), 1 = v2.1 (both JPSS
            satellites).
THREE OF FIVE CHANNELS ARE NaN FOR EVERY ROW, and that is the honest answer
rather than a number this store would have had to invent (ADAPTER_CONTRACT
rule 3, ml/CLAUDE.md §5.22). What the store is FOR is the other two plus the
row's existence: when a 375 m image of this place exists, and which
instrument took it.
"""
import datetime as dt

import numpy as np

import build_family10_stores as f10b
from family1.adapters import _stac as st

SENSOR_TABLE = {1: "Suomi-NPP/VIIRS", 2: "NOAA-20/VIIRS", 3: "NOAA-21/VIIRS"}
QC_TABLE = {0: "v2", 1: "v2.1"}
COLLECTIONS = (("npp", "VNP02IMG", "2"),
               ("j01", "VJ102IMG", "2.1"),
               ("j02", "VJ202IMG", "2.1"))
RECORD_FIRST = (2012, 1)


class ViirsCatalogue(st.CmrCatalogue):
    store = "cat_viirs"
    title = ("VIIRS I-band L1B swath catalogue (VNP02IMG / VJ102IMG / "
             "VJ202IMG, 375 m, LAADS), one row per six-minute granule")
    licence = {
        "name": ("catalogue rows: our own metadata, CC0. The radiances they "
                 "point at are NASA VIIRS L1B (LAADS DAAC) — NASA open data, "
                 "free to use and redistribute with attribution"),
        "redistribution": "yes", "derived_works": "free",
        "attribution": ("VIIRS L1B, NASA LAADS DAAC; granule list from NASA's "
                        "Common Metadata Repository, "
                        "https://cmr.earthdata.nasa.gov")}
    first_year = 2012
    # A six-minute I-band granule is ~3,060 km across track and ~2,400 km
    # along it; the nominal footprint width is the square root of that area,
    # and the probe's median `area` is what checks it.
    footprint_km = 2710.0
    scene_seconds = 360.0
    COLLECTIONS = COLLECTIONS
    SENSOR_TABLE = SENSOR_TABLE
    QC_TABLE = QC_TABLE
    window_step = "day"
    qc_policy = (
        "qc is the collection version: 0 = v2 (VNP02IMG, Suomi-NPP), "
        "1 = v2.1 (VJ102IMG and VJ202IMG). A version the table does not list "
        "is 255 and counted by name in `qc_unlisted`. `cloud`, `valid` and "
        "`angle` are NaN for EVERY row: an I-band L1B granule's metadata "
        "publishes no cloud fraction, no valid fraction and no mean angle, "
        "and this store does not invent one. `area` comes from the granule's "
        "own footprint polygon; a value outside a channel's bounds is NaN and "
        "counted (`out_of_bounds`).")
    sources = (st.CMR_GRANULES,
               "https://cmr.earthdata.nasa.gov/search/collections.json"
               "?short_name=VNP02IMG",
               "https://ladsweb.modaps.eosdis.nasa.gov/")
    verified = (
        "2026-09-18 from the sandbox, anonymously: the three I-band L1B "
        "collections (VNP02IMG v2 LAADS from 2012-01-19, VJ102IMG v2.1 from "
        "2018-01-05, VJ202IMG v2.1 from 2023-02-10) and the six NRT twins "
        "that are deliberately excluded; VNP02IMG's 1,267,013 granules, its "
        "first (2012-01-19T00:00) and its newest (2026-09-17T11:48); 242 "
        "granules a day on three sampled days and monthly counts of "
        "6,476-7,441 across 2023-2026; a granule is 2,480 bytes in umm_json "
        "against 14,995 in the .json feed")
    notes = (
        "The pixels are never copied. THREE satellites, not the note's two: "
        "VJ202IMG (NOAA-21, 2023-02 onward) is in the archive and is "
        "catalogued, with its own sensor code. The near-real-time twins "
        "(provider LANCEMODIS) are the same overpasses and are excluded. The "
        "granule UR is LAADS's opaque archive number; the file name is in "
        "assets.parquet.")
    smoke_window = ("2024-06-10", "2024-06-11")
    smoke_probe_month = "2024-06"

    def record_first(self):
        return RECORD_FIRST

    def qc_text(self, gid, coll, umm, aa):
        return f"v{coll[2]}"

    def asset_id(self, gid, umm):
        """LAADS's UR is an archive number, so no name carries it.

        Returning a string the file names cannot contain leaves the template
        as the literal file names, which is right: there is nothing to
        abbreviate, and a template that pretended otherwise would produce
        urls that do not exist.
        """
        return "\x00"

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
        out = {"dataset": "VIIRS I-band L1B (LAADS) via CMR granule search",
               "url": st.CMR_GRANULES, "page_size": self.page_size,
               "years": [int(v) for v in ctx.years], "collections": {},
               "first_month": f"{y:04d}-{m:02d}"}
        total_month = 0
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
            total_month += n_month
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
                             "log2_area": sc.values[3], "sensor": sc.values[4],
                             "qc": sc.qc, "asset_dir": sc.base_url,
                             "asset_template": sc.asset_set, "counts": c2}
        if not total_month:
            raise st.Refusal(
                f"{self.store}: CMR counts ZERO I-band granules in "
                f"{y:04d}-{m:02d} across all three collections. A month of a "
                f"live archive with nothing in it is a broken query, not a "
                f"measurement (the 2026-09-14 rule).")
        out["granules_first_month_total"] = total_month
        out["first_record"] = first
        out["sensor_table"] = {str(k): v for k, v in SENSOR_TABLE.items()}
        out["qc_table"] = {str(k): v for k, v in QC_TABLE.items()}
        return out

    def smoke_sources(self, root, d_lo, d_hi, seed=20260918):
        from family1.adapters import _cat_smoke as cs
        return cs.cmr_sources(self, root, d_lo, d_hi, cs.VIIRS_GRANULES)


ADAPTER = ViirsCatalogue

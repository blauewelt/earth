"""HLS v2.0 — NASA's harmonised Landsat and Sentinel-2 tiles, one row per tile.

PLAIN ENGLISH. NASA takes the Sentinel-2 and Landsat pictures of the land,
puts them on one 30 m grid with one set of bands and one atmospheric
correction, and publishes the result as HLS ("Harmonized Landsat
Sentinel-2"). It is the same sky as `cat_landsat` and `cat_s2` see, sampled
onto a grid a model can compare across sensors — which is exactly why it is
worth cataloguing beside them. S30 comes from Sentinel-2's MSI, L30 from
Landsat's OLI; each granule is one 110 km military-grid (MGRS) tile of one
overpass. The pixels — about 0.7 petabytes a year — stay at the LP DAAC.

SOURCE, VERIFIED 2026-09-18 FROM THIS SANDBOX (anonymous, no Earthdata login
needed to SEARCH; a login is needed to DOWNLOAD the pixels, which this store
never does):
  https://cmr.earthdata.nasa.gov/search/granules.umm_json
      ?short_name=HLSS30&version=2.0&temporal=<from>,<to>&page_size=2000
  collections HLSS30 v2.0 and HLSL30 v2.0, provider LPCLOUD

WHY `umm_json` AND NOT THE STAC ENDPOINT. `https://cmr.earthdata.nasa.gov/
stac/LPCLOUD/search` works anonymously, and it was measured and rejected:
a POST with `limit` above 1 answers **HTTP 502** (measured on both
collections), and each item is **14.1 kB** because it expands all forty
assets into a STAC `assets` block. The native CMR granule search pages
properly (`page_size` 2000, `CMR-Search-After` cursor, `CMR-Hits` header as
the producer's own count) and its UMM-G form is the ONLY one that carries the
three numbers this store's channels need — `CLOUD_COVERAGE`,
`SPATIAL_COVERAGE` and `MEAN_SUN_ZENITH_ANGLE` are UMM `AdditionalAttributes`
and appear in neither the `.json` feed nor the STAC item. Measured rates:
2,000 granules in 1.7 s (17.5 MB/s) at 14.4 kB a granule.

MEASURED COUNTS, 2026-09-18 (`CMR-Hits`):
  HLSS30 2024-06   226,982 granules   (the note's "2.71 M in 2025" ≈ 12 × this)
  HLSL30 2024-06     ~34,000 in the first day-window sample; the index
                     measures the month
Record: HLSL30 from 2013-04 (Landsat 8's first year), HLSS30 from 2015-11.

WHAT A ROW IS.
  time_s    `TemporalExtent.RangeDateTime.BeginningDateTime` — the sensing
            instant, int32 seconds since 1982-01-01.
  lat, lon  the spherical mean of the granule's own `GPolygons` boundary.
  platform  platform_hash(the granule UR, e.g.
            `HLS.S30.T01WEV.2024153T000609.v2.0`).
  values    cloud   `CLOUD_COVERAGE`, per cent
            valid   `SPATIAL_COVERAGE`, per cent of the tile that carries
                    data — the one catalogue of the eight whose producer
                    publishes it
            angle   `MEAN_SUN_ZENITH_ANGLE`, degrees
            log2_area
                    log2 of the boundary's area on the sphere in km²
                    (float16 cannot hold a large footprint in km²)
            sensor  1..5 — WHICH satellite fed the harmonisation:
                    Sentinel-2A/2B/2C for S30, Landsat-8/9 for L30, from the
                    granule's own `Platforms` block rather than from the
                    collection, so the store distinguishes them
  qc        the HLS product version (`v2.0` -> 0), from the granule UR's own
            suffix.

ASSETS. A granule's forty-odd files sit in one directory named after the
granule and are named `<granule>.<BAND>.tif`; the URLs come from the
granule's own `RelatedUrls` (`GET DATA`), never from a pattern, and
`assets.parquet` stores the directory plus the file-name template. The `s3://`
duplicates and the browse images are not data URLs and are not in the table.

MEMORY. One day-window of one collection per worker; the walker is a
generator, so the peak is one page (2,000 granules, ~29 MB) plus that
window's `Scene` objects.
"""
import datetime as dt

import numpy as np

import build_family10_stores as f10b
from family1.adapters import _stac as st

SENSOR_TABLE = {1: "Sentinel-2A/Sentinel-2 MSI",
                2: "Sentinel-2B/Sentinel-2 MSI",
                3: "Sentinel-2C/Sentinel-2 MSI",
                4: "LANDSAT-8/OLI",
                5: "LANDSAT-9/OLI"}
QC_TABLE = {0: "v2.0", 1: "v2.1"}
COLLECTIONS = (("S30", "HLSS30", "2.0"), ("L30", "HLSL30", "2.0"))
RECORD_FIRST = (2013, 4)


class HlsCatalogue(st.CmrCatalogue):
    store = "cat_hls"
    title = ("HLS v2.0 scene catalogue — harmonised Sentinel-2 (S30) and "
             "Landsat (L30) 30 m tiles, one row per granule")
    licence = {
        "name": ("catalogue rows: our own metadata, CC0. The imagery they "
                 "point at is NASA HLS v2.0 (LP DAAC) — NASA open data, free "
                 "to use and redistribute with attribution"),
        "redistribution": "yes", "derived_works": "free",
        "attribution": ("HLS v2.0, NASA LP DAAC; granule list from NASA's "
                        "Common Metadata Repository, "
                        "https://cmr.earthdata.nasa.gov")}
    first_year = 2013
    footprint_km = 109.8           # one MGRS tile
    scene_seconds = 1.0            # a tile carries one sensing instant
    COLLECTIONS = COLLECTIONS
    SENSOR_TABLE = SENSOR_TABLE
    QC_TABLE = QC_TABLE
    window_step = "day"
    # 500, NOT the 2,000 CMR allows: a page of 2,000 HLS granules is 28 MB and
    # CMR cut two of them mid-chunk in one run (IncompleteRead at 23.6 MB and
    # 38.0 MB). Seven megabytes a page goes through.
    page_size = 500
    qc_policy = (
        "qc is the HLS product version taken from the granule's own "
        "identifier suffix: 0 = v2.0, 1 = v2.1 (not yet published). A version "
        "the table does not list is 255 and counted by name in "
        "`qc_unlisted`. `cloud` is CLOUD_COVERAGE, `valid` is "
        "SPATIAL_COVERAGE and `angle` is MEAN_SUN_ZENITH_ANGLE, each NaN when "
        "the granule does not carry it and counted "
        "(`attr_missing_<name>`); a value outside its channel's bounds is NaN "
        "and counted (`out_of_bounds`), never clipped.")
    sources = (st.CMR_GRANULES,
               "https://cmr.earthdata.nasa.gov/search/collections.json"
               "?short_name=HLSS30",
               "https://cmr.earthdata.nasa.gov/stac/LPCLOUD")
    verified = (
        "2026-09-18 from the sandbox, anonymously: both collections exist "
        "(HLSS30 v2.0 from 2015-11, HLSL30 v2.0 from 2013-04); CMR-Hits for "
        "HLSS30 2024-06 is 226,982 and for 2024-06-01 alone 7,698; "
        "page_size=2000 with the CMR-Search-After cursor walks a window and "
        "the walk equals CMR-Hits; the UMM-G granule carries CLOUD_COVERAGE, "
        "SPATIAL_COVERAGE, MEAN_SUN_ZENITH_ANGLE, GPolygons and Platforms, "
        "and the CMR-STAC endpoint answers HTTP 502 for any limit above 1")
    notes = (
        "The pixels are never copied and no Earthdata login is used: the "
        "SEARCH is anonymous. The sensor code names the satellite the "
        "harmonisation was fed by, read from each granule's Platforms block. "
        "`valid` (SPATIAL_COVERAGE) is published by this producer and by no "
        "other of the eight catalogues.")
    smoke_window = ("2024-06-01", "2024-06-02")
    smoke_probe_month = "2024-06"

    def record_first(self):
        return RECORD_FIRST

    # -- one granule -------------------------------------------------------
    def asset_id(self, gid, umm):
        return gid

    def qc_text(self, gid, coll, umm, aa):
        """`v2.0` — the granule identifier's own version suffix."""
        parts = str(gid).split(".")
        if len(parts) >= 2 and parts[-2].startswith("v"):
            return f"{parts[-2]}.{parts[-1]}"
        return str(coll[2] and f"v{coll[2]}" or "")

    def values(self, gid, coll, umm, aa, area, sensor, counts):
        cloud = self._attr(aa, "CLOUD_COVERAGE", counts)
        valid = self._attr(aa, "SPATIAL_COVERAGE", counts)
        zen = self._attr(aa, "MEAN_SUN_ZENITH_ANGLE", counts)
        return [st.clamp_pct(cloud), st.clamp_pct(valid),
                (float(zen) if zen is not None else float("nan")),
                st.area_channel(area, counts), float(sensor)]

    @staticmethod
    def _attr(aa, name, counts):
        v = aa.get(name)
        if v is None or v == "":
            k = f"attr_missing_{name}"
            counts[k] = counts.get(k, 0) + 1
            return None
        if isinstance(v, list):
            v = v[0] if v else None
        try:
            return float(v)
        except (TypeError, ValueError):
            k = f"attr_not_numeric_{name}"
            counts[k] = counts.get(k, 0) + 1
            return None

    # -- the index ---------------------------------------------------------
    def index(self, ctx):
        fet = st.Fetcher(ctx, self.store)
        out = {"dataset": "HLS v2.0 (NASA LP DAAC) via CMR granule search",
               "url": st.CMR_GRANULES, "page_size": self.page_size,
               "years": [int(y) for y in ctx.years], "collections": {}}
        # THE MONTH THE RUN ACTUALLY STARTS IN, not January: a lane over
        # 2019 must check 2019-01, and the smoke's two-day window must check
        # its own month or the index would query a month the canned archive
        # has never heard of.
        y, m = ctx.d_lo.year, ctx.d_lo.month
        if (y, m) < RECORD_FIRST:
            y, m = RECORD_FIRST
        a = dt.datetime(y, m, 1)
        b = dt.datetime(y + (m == 12), m % 12 + 1, 1)
        for coll in COLLECTIONS:
            info = {}
            c, _ = fet.get(
                "https://cmr.earthdata.nasa.gov/search/collections.json"
                f"?short_name={coll[1]}&version={coll[2]}&page_size=5")
            ents = (c.get("feed") or {}).get("entry") or []
            if not ents:
                raise st.Refusal(f"CMR lists no collection {coll[1]} "
                                 f"v{coll[2]} — the store's source has moved")
            info["dataset_id"] = ents[0].get("dataset_id")
            info["time_start"] = ents[0].get("time_start")
            info["time_end"] = ents[0].get("time_end")
            info["concept_id"] = ents[0].get("id")
            info["granules_total"] = st.cmr_hits(
                fet, st.CMR_GRANULES,
                [("short_name", coll[1]), ("version", coll[2])])
            info["granules_first_month"] = st.cmr_hits(
                fet, st.CMR_GRANULES, self.cmr_params(a, b, coll))
            info["first_month"] = f"{y:04d}-{m:02d}"
            out["collections"][coll[0]] = info
        # ONE REAL GRANULE, read the way a row is read — the field selection
        # and the asset layout are checked here, where the inputs are all it
        # has cost (ml/CLAUDE.md §0.3).
        first = None
        for coll in COLLECTIONS:
            d, _ = fet.get(st.cmr_url(st.CMR_GRANULES,
                                      self.cmr_params(a, b, coll), 1))
            items = d.get("items") or []
            if not items:
                continue
            c2 = {}
            sc = self.granule(items[0], coll, c2, "index")
            first = {"collection": coll[0], "id": sc.stac_id,
                     "time_s": sc.t, "lat": sc.lat, "lon": sc.lon,
                     "values": dict(zip(self.channel_names,
                                        [None if not np.isfinite(x) else x
                                         for x in sc.values])),
                     "qc": sc.qc, "asset_dir": sc.base_url,
                     "asset_template": sc.asset_set, "counts": c2}
            break
        if first is None:
            raise st.Refusal(f"{self.store}: no granule at all in "
                             f"{y:04d}-{m:02d} for either collection — an "
                             f"empty first month is a broken query")
        out["first_record"] = first
        out["sensor_table"] = {str(k): v for k, v in SENSOR_TABLE.items()}
        out["qc_table"] = {str(k): v for k, v in QC_TABLE.items()}
        return out

    # -- the smoke ---------------------------------------------------------
    def smoke_sources(self, root, d_lo, d_hi, seed=20260918):
        from family1.adapters import _cat_smoke as cs
        return cs.cmr_sources(self, root, d_lo, d_hi, cs.HLS_GRANULES)


ADAPTER = HlsCatalogue

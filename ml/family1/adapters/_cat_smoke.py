"""Synthetic producers for the eight scene-catalogue stores' smokes.

PLAIN ENGLISH. A catalogue adapter's "archive" is a web API, not a directory
of files, so its smoke cannot write a few files and read them back. What it
writes instead is a directory of CANNED RESPONSES — one JSON file per request
the adapter makes, named after the request (`_stac.canned_key`) — and
`--source-dir` makes the adapter read those instead of the network. The real
paging code, the real refusals, the real count checks and the real row packing
all run; only the socket is replaced.

THE FILES ARE RECORDED, NOT WRITTEN BY HAND. `record()` runs the adapter once
against an in-process fake producer (`serve(method, url, body)`), remembering
every request and its answer, and writes them out. So:

  * every request the adapter makes has a file, by construction — a hand-built
    fixture drifts the first time a URL changes, and the failure looks like a
    broken adapter;
  * the TRUTH `check_smoke` compares the store against is what the adapter's
    own `scenes()` produced, so a smoke that passes proves the store holds
    exactly the rows the adapter emitted, in the order it emitted them;
  * a request the adapter makes that the fake producer does not know about
    raises there and then, which is how a paging bug shows up in a test
    instead of on a runner.

The synthetic producers below are HOSTILE on purpose. Each one carries, in the
shapes its real producer actually uses: a window with no scenes at all, a page
boundary that must be walked (so `limit`/`$top`/`page_size` paging is
exercised for real), a scene whose cloud cover is absent, one whose value is
out of bounds (so `mask_bounds` counts it and the store holds NaN), a sensor
and a processing baseline the code table does not list (so the 255 paths are
exercised), a footprint that crosses the antimeridian, and a scene listed twice
(so the de-duplication is exercised).
"""
import argparse
import datetime as dt
import json
import math
import os
import re
import shutil
import urllib.parse

import numpy as np

import build_family10_stores as f10b
from family1.adapters import _stac as st


# =================================================================== record ==
def record(adapter, d_lo, d_hi, serve, root, text_serve=None):
    """Run `adapter` against `serve`, write the canned files, return the truth.

    `serve(method, url, body) -> (obj, headers)` is the fake producer.
    `text_serve(url) -> str` answers the one non-JSON endpoint (ASF's
    `output=count`); a store that has none passes nothing.
    """
    import build_family1_stores as b1
    seen = {}

    def fake_do(self, method, url, body, headers):
        obj, hdr = serve(method, url, body, headers)
        raw = json.dumps(obj, separators=(",", ":")).encode("utf-8")
        self.ctx.count_bytes(len(raw))
        self.requests += 1
        seen[st.canned_key(method, url, body, headers)] = \
            _with_headers(obj, hdr)
        return obj, dict(hdr or {})

    def fake_text(fet, url):
        s = str(text_serve(url))
        fet.ctx.count_bytes(len(s.encode("utf-8")))
        fet.requests += 1
        seen[st.canned_key("GET", url, None)] = {"__text__": s}
        return s

    work = os.path.join(root, "_record_work")
    shutil.rmtree(work, ignore_errors=True)
    os.makedirs(work, exist_ok=True)
    lay = b1.layout_for(adapter)

    def make_ctx(lo, hi):
        ns = argparse.Namespace(
            # `source_dir` IS SET during the recording too, even though the
            # fake producer bypasses the canned reader: it is what
            # `CatalogueAdapter.page_for` and `workers` branch on, so a
            # recording made with it unset would page differently from the
            # replay and every second page would be a missing file.
            store=adapter.store, work=work, source_dir=str(root),
            start=str(lo),
            end=str(hi), stage="all", force=False, attempts=1, qc_keep=2,
            smoke=True, parts_from_hub=False, push_parts=False,
            allow_missing_years=False, assemble="auto",
            grid_dtype=f10b.GRID_DTYPE_DEFAULT,
            max_hours=f10b.FISHING_HOURS_CEILING, socat_url="")
        return f10b.Ctx(ns, adapter=adapter, layout=lay)

    ctx = make_ctx(d_lo, d_hi)
    real_do, real_text = st.Fetcher._do, st._text
    st.Fetcher._do = fake_do
    if text_serve is not None:
        st._text = fake_text
    try:
        adapter.index(ctx)
        truth = []
        for y in ctx.years:
            for w in adapter.plan(ctx, y):
                scenes, _c = adapter.scenes(ctx, w)
                for sc in scenes:
                    if not (ctx.t_lo <= sc.t <= ctx.t_hi):
                        continue
                    truth.append(sc)
        # THE PROBE'S OWN WINDOWS TOO. `--smoke` runs `--stage probe
        # --probe-month <smoke_probe_month>` on this same canned archive, and
        # `probe_namespace` widens the window to the WHOLE month — so the
        # month's other days must have canned (empty) answers or the probe
        # refuses on a missing file. Recording them here is also the only
        # place the empty-window path gets exercised at all.
        pm = getattr(adapter, "smoke_probe_month", None)
        if pm:
            py, pmo = (int(x) for x in pm.split("-"))
            import calendar
            nd = calendar.monthrange(py, pmo)[1]
            pctx = make_ctx(dt.date(py, pmo, 1), dt.date(py, pmo, nd))
            adapter.index(pctx)
            for w in adapter.plan(pctx, py, month=pmo):
                adapter.scenes(pctx, w)
    finally:
        st.Fetcher._do = real_do
        st._text = real_text
    shutil.rmtree(work, ignore_errors=True)
    out = os.path.join(root, adapter.store)
    os.makedirs(out, exist_ok=True)
    for k, obj in seen.items():
        with open(os.path.join(out, k + ".json"), "w") as fh:
            json.dump(obj, fh, separators=(",", ":"))
    return _truth_rows(adapter, truth)


def _with_headers(obj, hdr):
    if not hdr or not isinstance(obj, dict):
        return obj
    d = dict(obj)
    d["__headers__"] = dict(hdr)
    return d


def _truth_rows(adapter, scenes):
    """`Scene`s, in yield order, as `check_smoke`'s truth dicts.

    The out-of-bounds masking the fetch applies is applied here too — the
    truth is what the STORE must hold, and contract rule 3 says an
    out-of-bounds value is stored as NaN and counted.
    """
    out = []
    seen = set()
    for sc in scenes:
        if sc.platform in seen:
            continue                    # the adapter de-duplicates; so does this
        seen.add(sc.platform)
        v = np.array(sc.values, np.float64).reshape(1, adapter.C)
        adapter.mask_bounds(v)
        out.append({"t": int(sc.t), "lat": float(sc.lat), "lon": float(sc.lon),
                    "platform": int(sc.platform), "v": [float(x) for x in v[0]],
                    "qc": int(sc.qc)})
    return out


# ================================================================= geometry ==
def ring(lat, lon, dlat=1.0, dlon=1.0):
    """A closed rectangle ring as GeoJSON coordinates (lon, lat)."""
    return [[[lon - dlon, lat - dlat], [lon + dlon, lat - dlat],
             [lon + dlon, lat + dlat], [lon - dlon, lat + dlat],
             [lon - dlon, lat - dlat]]]


def wkt(lat, lon, dlat=1.0, dlon=1.0):
    pts = [(lon - dlon, lat - dlat), (lon + dlon, lat - dlat),
           (lon + dlon, lat + dlat), (lon - dlon, lat + dlat),
           (lon - dlon, lat - dlat)]
    return "POLYGON ((" + ",".join(f"{a} {b}" for a, b in pts) + "))"


def cmr_polygon(lat, lon, dlat=1.0, dlon=1.0):
    """CMR's "lat lon lat lon ..." ring, counter-clockwise and closed."""
    pts = [(lat - dlat, lon - dlon), (lat - dlat, lon + dlon),
           (lat + dlat, lon + dlon), (lat + dlat, lon - dlon),
           (lat - dlat, lon - dlon)]
    return [" ".join(f"{a} {b}" for a, b in pts)]


# ============================================================== cat_landsat ==
LS_SCENES = [
    # (day offset, hh:mm:ss, platform, correction, tier, cloud, sun_elev,
    #  path, row, lat, lon, set)
    (0, "00:11:22", "LANDSAT_8", "L2SP", "T1", 27.34, 32.03, "092", "079",
     -27.4, 148.6, "oli-tirs"),
    (0, "02:33:44", "LANDSAT_5", "L2SP", "T2", 0.0, 61.5, "026", "044",
     29.7, -96.1, "tm"),
    (0, "04:55:00", "LANDSAT_7", "L2SR", "T1", None, 14.2, "018", "122",
     -66.0, 179.4, "tm"),          # no cloud value, and across the dateline
    (0, "06:00:00", "LANDSAT_9", "L2SP", "RT", 100.000001, 0.0, "095", "022",
     54.5, 165.6, "oli-tirs"),     # sun elevation 0 -> NaN zenith; cloud 100
    (1, "01:02:03", "LANDSAT_4", "L2SP", "T1", 140.0, 45.0, "200", "030",
     10.0, 20.0, "tm"),            # cloud out of bounds -> NaN, counted
    (1, "03:04:05", "LANDSAT_3", "L2SP", "T1", 12.0, 30.0, "201", "031",
     11.0, 21.0, "tm"),            # a platform the table does not list -> 255
    (1, "05:06:07", "LANDSAT_9", "L2XX", "T9", 5.0, 50.0, "095", "023",
     55.0, 166.0, "oli-tirs"),     # a baseline the table does not list -> 255
]
LS_DIR = ("https://landsatlook.usgs.gov/data/collection02/level-2/standard/"
          "{sensor}/{year}/{path}/{row}/{scene}")
LS_SENSOR_DIR = {"LANDSAT_4": "tm", "LANDSAT_5": "tm", "LANDSAT_7": "etm",
                 "LANDSAT_8": "oli-tirs", "LANDSAT_9": "oli-tirs",
                 "LANDSAT_3": "mss"}
LS_PREFIX = {"LANDSAT_3": "LM03", "LANDSAT_4": "LT04", "LANDSAT_5": "LT05",
             "LANDSAT_7": "LE07", "LANDSAT_8": "LC08", "LANDSAT_9": "LC09"}


def landsat_sources(adapter, root, d_lo, d_hi, seed=20260918):
    """A two-day landsatlook, paged at THREE items so paging is real."""
    from family1.adapters import cat_landsat as cl
    page = 3
    items = {}                       # "YYYY-MM-DD" -> [feature]
    for off, hms, plat, corr, tier, cloud, elev, p, r, lat, lon, aset in \
            LS_SCENES:
        day = d_lo + dt.timedelta(days=off)
        scene = (f"{LS_PREFIX[plat]}_{corr}_{p}{r}_"
                 f"{day:%Y%m%d}_{day:%Y%m%d}_02_{tier}")
        base = LS_DIR.format(sensor=LS_SENSOR_DIR[plat], year=day.year,
                             path=p, row=r, scene=scene)
        assets = {f"a{i}": {"href": f"{base}/{n.replace('{id}', scene)}"}
                  for i, n in enumerate(adapter.ASSET_SETS[aset])}
        assets["index"] = {"href": ("https://landsatlook.usgs.gov/"
                                    f"stac-browser/x/{scene}")}
        props = {"datetime": f"{day:%Y-%m-%d}T{hms}.000000Z",
                 "platform": plat, "landsat:correction": corr,
                 "landsat:collection_category": tier,
                 "view:sun_elevation": elev}
        if cloud is not None:
            props["eo:cloud_cover"] = cloud
        items.setdefault(f"{day:%Y-%m-%d}", []).append({
            "type": "Feature", "id": scene + "_SR",
            "collection": cl.COLLECTION,
            "bbox": [lon - 1, lat - 1, lon + 1, lat + 1],
            "geometry": {"type": "Polygon", "coordinates": ring(lat, lon)},
            "properties": props, "assets": assets})
    # THE SAME SCENE LISTED TWICE, in the second day's window: the store keeps
    # one row and counts `rows_duplicate_scene`.
    d1 = f"{d_lo + dt.timedelta(days=1):%Y-%m-%d}"
    items[d1] = items[d1] + [items[d1][0]]

    def serve(method, url, body, headers=None):
        if method == "GET" and url.endswith(f"/collections/{cl.COLLECTION}"):
            return {"id": cl.COLLECTION, "extent": {"temporal": {"interval": [
                ["1982-08-22T00:00:00.000Z", None]]}}}, {}
        if method != "POST":
            raise AssertionError(f"smoke: unexpected {method} {url}")
        if url == cl.ES_SEARCH:
            # the Earth Search cross-check: answer with a count that agrees
            lo = str(body["datetime"]).split("/")[0][:10]
            n = sum(len(v) for k, v in items.items() if k[:7] == lo[:7])
            return {"type": "FeatureCollection", "numberMatched": n,
                    "numberReturned": 0, "features": [], "links": []}, {}
        if url != cl.SEARCH:
            raise AssertionError(f"smoke: unexpected POST {url}")
        lo, hi = str(body["datetime"]).split("/")
        pool = []
        for k in sorted(items):
            if lo[:10] <= k <= hi[:10]:
                pool.extend(items[k])
        lim = int(body.get("limit") or page)
        start = int(body.get("next") or 0)
        chunk = pool[start:start + lim]
        links = []
        if start + lim < len(pool):
            links.append({"rel": "next", "method": "POST", "href": cl.SEARCH,
                          "body": {"next": start + lim}})
        return {"type": "FeatureCollection", "numberMatched": len(pool),
                "numberReturned": len(chunk),
                "features": [_project(f, body.get("fields")) for f in chunk],
                "links": links}, {}

    return record(adapter, d_lo, d_hi, serve, root)


def _project(feat, fields):
    """Apply a STAC `fields` include/exclude the way stac-server does.

    Only the two shapes the adapters use are implemented: a top-level or
    `properties.<k>` include list, and an `assets.*.<k>` exclude. The point is
    that the smoke sees the same pruned feature the server sends, so an
    adapter that reads a field it did not ask for fails in the test.
    """
    if not fields:
        return feat
    inc = list(fields.get("include") or [])
    exc = list(fields.get("exclude") or [])
    out = {"type": feat.get("type"), "id": feat.get("id"),
           "collection": feat.get("collection")}
    if inc:
        props = {}
        for k in inc:
            if k.startswith("properties."):
                kk = k.split(".", 1)[1]
                if kk in (feat.get("properties") or {}):
                    props[kk] = feat["properties"][kk]
            elif k in feat:
                out[k] = feat[k]
        out["properties"] = props
    else:
        out.update({k: v for k, v in feat.items()})
    drop = {e.split(".")[-1] for e in exc if e.startswith("assets.*.")}
    if "assets" in out and drop:
        out["assets"] = {k: {kk: vv for kk, vv in (a or {}).items()
                             if kk not in drop}
                         for k, a in out["assets"].items()}
    for e in exc:
        if "." not in e and e in out:
            out.pop(e)
    return out


# ================================================== a CMR producer, canned ==
def _umm(g):
    """One synthetic UMM-G granule from a smoke descriptor."""
    urls = [{"URL": f"{g['dir']}/{n}", "Type": "GET DATA"} for n in g["files"]]
    urls += [{"URL": f"s3://bucket/{g['ur']}/{g['files'][0]}",
              "Type": "GET DATA VIA DIRECT ACCESS"},
             {"URL": f"https://browse.example/{g['ur']}.jpg",
              "Type": "GET RELATED VISUALIZATION"}]
    umm = {"GranuleUR": g["ur"],
           "TemporalExtent": {"RangeDateTime": {
               "BeginningDateTime": g["t"], "EndingDateTime": g["t"]}},
           "AdditionalAttributes": [{"Name": k, "Values": [str(v)]}
                                    for k, v in (g.get("attrs") or {}).items()],
           "RelatedUrls": urls,
           "Platforms": [{"ShortName": g["platform"],
                          "Instruments": [{"ShortName": g["instrument"]}]}]}
    if g.get("box"):
        s, w, n, e = g["box"]
        umm["SpatialExtent"] = {"HorizontalSpatialDomain": {"Geometry": {
            "BoundingRectangles": [{"SouthBoundingCoordinate": s,
                                    "WestBoundingCoordinate": w,
                                    "NorthBoundingCoordinate": n,
                                    "EastBoundingCoordinate": e}]}}}
    else:
        lat, lon = g["lat"], g["lon"]
        pts = [(lat - 1, lon - 1), (lat - 1, lon + 1), (lat + 1, lon + 1),
               (lat + 1, lon - 1), (lat - 1, lon - 1)]
        umm["SpatialExtent"] = {"HorizontalSpatialDomain": {"Geometry": {
            "GPolygons": [{"Boundary": {"Points": [
                {"Latitude": a, "Longitude": b} for a, b in pts]}}]}}}
    return {"meta": {"concept-id": "G1-X"}, "umm": umm}


def cmr_sources(adapter, root, d_lo, d_hi, make, page=3):
    """Canned CMR responses for a catalogue, paged at THREE granules.

    `make(adapter, d_lo, d_hi)` returns `{(short_name, version,
    "YYYY-MM-DD"): [descriptor, ...]}`. The fake CMR answers `collections.json` for the
    index, `granules.umm_json` with a `CMR-Hits` header and a
    `CMR-Search-After` cursor, and — crucially — an EMPTY page with a real
    `CMR-Hits` of 0 for every window the store plans and the descriptors do
    not fill, which is the path a real archive gap takes.
    """
    pool = make(adapter, d_lo, d_hi)

    def serve(method, url, body, headers=None):
        u = urllib.parse.urlparse(url)
        q = urllib.parse.parse_qs(u.query)
        if u.path.endswith("/collections.json"):
            sn = (q.get("short_name") or [""])[0]
            vr = (q.get("version") or [""])[0]
            n = sum(len(v) for k, v in pool.items()
                    if k[0] == sn and k[1] == vr)
            return {"feed": {"entry": [{
                "id": f"C1-{sn}", "dataset_id": f"{sn} (smoke)",
                "short_name": sn, "version_id": (q.get("version") or [""])[0],
                "time_start": f"{d_lo}T00:00:00.000Z", "time_end": None,
                "granules": n}]}}, {}
        if not u.path.endswith("granules.umm_json"):
            raise AssertionError(f"smoke: unexpected CMR path {u.path}")
        sn = (q.get("short_name") or [""])[0]
        vr = (q.get("version") or [""])[0]
        tmp = (q.get("temporal") or [","])[0].split(",")
        lo, hi = tmp[0][:10], tmp[1][:10] if len(tmp) > 1 else tmp[0][:10]
        # THE VERSION IS PART OF THE KEY. ECOSTRESS serves two processing
        # versions under ONE short_name, so a fake producer that filtered on
        # the name alone would answer both queries with both versions' rows
        # and the store would hold every granule twice.
        items = []
        for k in sorted(pool):
            if k[0] == sn and k[1] == vr and lo <= k[2] <= hi:
                items.extend(pool[k])
        # THE FAKE CMR HONOURS page_size EXACTLY, the way the real one does.
        # Capping it at a smaller number here would make every page look
        # short and the short-page rule would refuse the whole window —
        # `CatalogueAdapter.page_for` is what makes the page small instead.
        ps = int((q.get("page_size") or [page])[0])
        # The cursor is the index of the next granule, and it travels in a
        # HEADER — which is why `canned_key` hashes the headers too.
        start = int((headers or {}).get("CMR-Search-After") or 0)
        chunk = items[start:start + ps]
        hdr = {"CMR-Hits": str(len(items))}
        nxt = start + len(chunk)
        if nxt < len(items):
            hdr["CMR-Search-After"] = str(nxt)
        return {"hits": len(items), "took": 1,
                "items": [_umm(g) for g in chunk]}, hdr

    return record(adapter, d_lo, d_hi, serve, root)


# ================================================================== cat_hls ==
HLS_BASE = ("https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/"
            "{coll}.020/{ur}")
HLS_S30_BANDS = ("B01", "B02", "B03", "B04", "B08", "B8A", "B11", "B12",
                 "Fmask", "SZA", "SAA", "VZA", "VAA")
HLS_L30_BANDS = ("B01", "B02", "B03", "B04", "B05", "B06", "B07", "B09",
                 "B10", "B11", "Fmask", "SZA", "SAA", "VZA", "VAA")


def HLS_GRANULES(adapter, d_lo, d_hi):
    """Seven granules over two days, hostile in seven different ways."""
    rows = [
        # day, hh:mm:ss, coll, tile, platform, cloud, valid, zenith, lat, lon
        (0, "00:06:19", "S30", "T01WEV", "Sentinel-2B", "18", "45",
         "49.39715493", 71.1, -175.5),
        (0, "10:11:12", "S30", "T33UUP", "Sentinel-2A", "0", "100",
         "31.5", 48.2, 16.4),
        (0, "00:00:13", "L30", "T56MKV", "LANDSAT-9", "100", "1",
         "38.95231598", -8.5, 152.9),
        (0, "12:00:00", "L30", "T30UXB", "LANDSAT-8", None, "77",
         "40.0", 51.0, -1.0),           # no CLOUD_COVERAGE at all -> NaN
        (1, "01:02:03", "S30", "T60XWQ", "Sentinel-2C", "7", "60",
         "88.0", 72.0, 179.6),          # across the antimeridian
        (1, "03:04:05", "S30", "T99ZZZ", "Sentinel-2X", "140", "50",
         "95.0", 10.0, 20.0),           # unlisted platform; cloud and zenith
                                        # both out of bounds -> NaN, counted
        (1, "05:06:07", "L30", "T31TCJ", "LANDSAT-9", "12", "abc",
         "30.0", 43.6, 1.4),            # SPATIAL_COVERAGE not a number
    ]
    out = {}
    for off, hms, ck, tile, plat, cloud, valid, zen, lat, lon in rows:
        day = d_lo + dt.timedelta(days=off)
        doy = day.timetuple().tm_yday
        short = "HLSS30" if ck == "S30" else "HLSL30"
        ur = (f"HLS.{ck}.{tile}.{day:%Y}{doy:03d}T"
              f"{hms.replace(':', '')}.v2.0")
        base = HLS_BASE.format(coll=short, ur=ur)
        bands = HLS_S30_BANDS if ck == "S30" else HLS_L30_BANDS
        attrs = {"MGRS_TILE_ID": tile[1:], "SPATIAL_COVERAGE": valid,
                 "MEAN_SUN_ZENITH_ANGLE": zen}
        if cloud is not None:
            attrs["CLOUD_COVERAGE"] = cloud
        out.setdefault((short, "2.0", f"{day:%Y-%m-%d}"), []).append({
            "ur": ur, "t": f"{day:%Y-%m-%d}T{hms}.072Z", "dir": base,
            "files": [f"{ur}.{b}.tif" for b in bands],
            "platform": plat,
            "instrument": "Sentinel-2 MSI" if ck == "S30" else "OLI",
            "lat": lat, "lon": lon, "attrs": attrs})
    return out


# ================================================================ cat_viirs ==
VIIRS_DIR = ("https://ladsweb.modaps.eosdis.nasa.gov/archive/allData/"
             "{ver}/{short}/{year}/{doy:03d}")


def VIIRS_GRANULES(adapter, d_lo, d_hi):
    """Six granules over two days: three satellites, one unlisted platform."""
    rows = [
        (0, "00:00:00", "npp", "Suomi-NPP", 12.0, 30.0),
        (0, "00:06:00", "npp", "Suomi-NPP", 40.0, -100.0),
        (0, "00:12:00", "j01", "NOAA-20", -70.0, 179.0),   # antimeridian
        (0, "00:18:00", "j02", "NOAA-21", 65.0, 12.0),
        (1, "01:00:00", "npp", "Suomi-NPP", 0.0, 0.0),
        (1, "01:06:00", "npp", "NOAA-99", 5.0, 5.0),       # unlisted platform
    ]
    by = {c[0]: c for c in adapter.COLLECTIONS}
    out = {}
    for off, hms, ck, plat, lat, lon in rows:
        day = d_lo + dt.timedelta(days=off)
        doy = day.timetuple().tm_yday
        short, ver = by[ck][1], by[ck][2]
        # LAADS's granule UR is an opaque archive number, and the FILE name is
        # a different string — exactly the shape the adapter must handle.
        num = 8280000000 + off * 1000 + int(hms.replace(":", ""))
        fn = (f"{short}.A{day:%Y}{doy:03d}.{hms[:2]}{hms[3:5]}."
              f"{ver.replace('.', '')}.{day:%Y%j%H%M%S}.nc")
        base = VIIRS_DIR.format(ver=ver.replace(".", ""), short=short,
                                year=day.year, doy=doy)
        out.setdefault((short, ver, f"{day:%Y-%m-%d}"), []).append({
            "ur": f"LAADS:{num}", "t": f"{day:%Y-%m-%d}T{hms}.000Z",
            "dir": base, "files": [fn], "platform": plat,
            "instrument": "VIIRS", "lat": lat, "lon": lon, "attrs": {}})
    return out


# ============================================================ cat_ecostress ==
ECO_DIR = ("https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/"
           "ECO_L2T_LSTE.{ver}/{ur}")
ECO_FILES = ("_LST.tif", "_LST_err.tif", "_EmisWB.tif", "_QC.tif",
             "_cloud.tif", "_water.tif", "_height.tif", "_view_zenith.tif",
             ".json", ".h5")


def ECO_GRANULES(adapter, d_lo, d_hi):
    """Five granules over two days, both versions, one crossing the seam."""
    rows = [
        (0, "01:01:12", "002", "38UMB", 50.46, 43.56, 51.45, 45.14),
        (0, "13:22:00", "002", "11SPA", 33.00, -118.0, 34.00, -117.0),
        (0, "23:59:34", "003", "02FML", -20.00, 178.5, -19.00, 179.5),
        (1, "05:00:00", "002", "31TCJ", 43.00, 1.00, 44.00, 2.00),
        (1, "06:00:00", "003", "60XWQ", 71.00, 179.0, 72.00, -179.0),
    ]
    out = {}
    for off, hms, ver, tile, s, w, n, e in rows:
        day = d_lo + dt.timedelta(days=off)
        orb = 46000 + off * 7 + int(hms[:2])
        ur = (f"ECOv{ver}_L2T_LSTE_{orb}_004_{tile}_"
              f"{day:%Y%m%d}T{hms.replace(':', '')}_01")
        base = ECO_DIR.format(ver=ver, ur=ur)
        out.setdefault(("ECO_L2T_LSTE", ver, f"{day:%Y-%m-%d}"),
                       []).append({
            "ur": ur, "t": f"{day:%Y-%m-%d}T{hms}.388Z", "dir": base,
            "files": [ur + x for x in ECO_FILES],
            "platform": "ISS", "instrument": "ECOSTRESS",
            "box": [s, w, n, e], "attrs": {}, "version": ver})
    return out


# ================================================ a CDSE producer, canned ==
_RE_PT = re.compile(r"'productType' and a/OData\.CSC\.StringAttribute/Value "
                    r"eq '([^']+)'")
_RE_TL = re.compile(r"'timeliness' and t/OData\.CSC\.StringAttribute/Value "
                    r"eq '([^']+)'")
_RE_GE = re.compile(r"ContentDate/Start ge (\S+?)\.000Z")
_RE_LT = re.compile(r"ContentDate/Start lt (\S+?)\.000Z")
_RE_SW = re.compile(r"startswith\(Name,'([^']+)'\)")


def odata_sources(adapter, root, d_lo, d_hi, make):
    """Canned CDSE OData responses, served by a fake that honours the filter.

    `make(adapter, d_lo, d_hi)` returns a flat list of product dicts, each
    with `Name`, `Id`, `ContentDate`, `GeoFootprint` and `Attributes`. The
    fake parses the four filter clauses the adapters actually build —
    `productType`, the `ContentDate/Start` half-open range, an optional
    `timeliness` and an optional `startswith(Name, ...)` — and answers
    `$top` / `$skip` / `$count` / `$orderby` the way the real endpoint does,
    INCLUDING the 422 when `$skip` passes 10,000. `$top` is capped for the
    smoke through `CatalogueAdapter.page_for`, not here, so a short page is a
    real short page.
    """
    prods = make(adapter, d_lo, d_hi)

    def matches(p, filt):
        aa = {a["Name"]: a["Value"] for a in (p.get("Attributes") or [])}
        m = _RE_PT.search(filt)
        if m and str(aa.get("productType")) != m.group(1):
            return False
        m = _RE_TL.search(filt)
        if m and str(aa.get("timeliness")) != m.group(1):
            return False
        m = _RE_SW.search(filt)
        if m and not str(p["Name"]).startswith(m.group(1)):
            return False
        t = str(p["ContentDate"]["Start"])[:19]
        m = _RE_GE.search(filt)
        if m and t < m.group(1):
            return False
        m = _RE_LT.search(filt)
        if m and t >= m.group(1):
            return False
        return True

    def serve(method, url, body, headers=None):
        if method != "GET":
            raise AssertionError(f"smoke: unexpected {method} {url}")
        q = urllib.parse.parse_qs(urllib.parse.urlparse(url).query)
        filt = (q.get("$filter") or [""])[0]
        top = int((q.get("$top") or [1000])[0])
        skip = int((q.get("$skip") or [0])[0])
        if skip > st.ODATA_SKIP_MAX:
            raise AssertionError(
                f"smoke: the adapter asked for $skip={skip}, which the real "
                f"CDSE refuses with HTTP 422 (<= {st.ODATA_SKIP_MAX})")
        hit = sorted((p for p in prods if matches(p, filt)),
                     key=lambda p: (p["ContentDate"]["Start"], p["Name"]))
        want = (q.get("$expand") or [""])[0] == "Attributes"
        out = []
        for p in hit[skip:skip + top]:
            d = {k: v for k, v in p.items() if k != "Attributes"}
            if want:
                d["Attributes"] = p["Attributes"]
            out.append(d)
        r = {"@odata.context": "$metadata#Products", "value": out}
        if (q.get("$count") or [""])[0] == "true":
            r["@odata.count"] = len(hit)
        return r, {}

    return record(adapter, d_lo, d_hi, serve, root)


def _prod(name, start, lat, lon, attrs, pid=None, dlat=1.0, dlon=1.0):
    return {"Id": pid or ("%032x" % (abs(hash(name)) % (1 << 128))),
            "Name": name,
            "ContentDate": {"Start": start, "End": start},
            "GeoFootprint": {"type": "Polygon",
                             "coordinates": ring(lat, lon, dlat, dlon)},
            "Attributes": [{"Name": k, "Value": v}
                           for k, v in attrs.items()]}


# =================================================================== cat_s2 ==
def S2_PRODUCTS(adapter, d_lo, d_hi):
    """Eight tiles over two days, hostile in six ways."""
    rows = [
        # day, hh:mm:ss, serial, baseline, cloud, tile, lat, lon
        (0, "00:01:39", "B", "05.10", 75.949842, "T08XMR", 81.4, -137.5),
        (0, "00:01:40", "A", "05.10", 0.0, "T33UUP", 48.2, 16.4),
        (0, "10:00:00", "C", "05.12", 100.000001, "T60XWQ", 72.0, 179.5),
        (0, "10:00:01", "A", "02.14", 12.5, "T31TCJ", 43.6, 1.4),
        (0, "10:00:02", "B", "05.10", None, "T18TWL", 41.0, -74.0),
        (1, "01:02:03", "A", "05.10", 140.0, "T55HDV", -33.9, 151.2),
        (1, "01:02:04", "X", "07.77", 5.0, "T22MGB", -3.1, -60.0),
        (1, "23:59:59", "C", "05.12", 33.3, "T01KAB", -17.7, -178.5),
    ]
    out = []
    for off, hms, ser, base, cloud, tile, lat, lon in rows:
        day = d_lo + dt.timedelta(days=off)
        stamp = f"{day:%Y%m%d}T{hms.replace(':', '')}"
        name = (f"S2{ser}_MSIL2A_{stamp}_N{base.replace('.', '')}_R016_"
                f"{tile}_{stamp}.SAFE")
        attrs = {"productType": "S2MSI2A", "platformSerialIdentifier": ser,
                 "processorVersion": base, "tileId": tile[1:],
                 "platformShortName": "SENTINEL-2",
                 "instrumentShortName": "MSI"}
        if cloud is not None:
            attrs["cloudCover"] = cloud
        out.append(_prod(name, f"{day:%Y-%m-%d}T{hms}.024000Z", lat, lon,
                         attrs))
    return out


# =================================================================== cat_s1 ==
def S1_PRODUCTS(adapter, d_lo, d_hi):
    """GRDH, SLC, and the -COG mirror that must NOT become a second row."""
    rows = [
        (0, "00:16:31", "A", "IW_GRDH_1S", "003.71", "VV&VH", -33.0, 18.0),
        (0, "04:44:40", "A", "IW_SLC__1S", "003.71", "VV&VH", 51.0, 4.0),
        (0, "12:00:00", "C", "IW_GRDH_1S", "004.00", "HH&HV", 78.0, 15.0),
        (1, "01:00:00", "D", "IW_GRDH_1S", "004.00", "VV&VH", -70.0, 179.2),
        (1, "02:00:00", "B", "IW_SLC__1S", "002.90", "VV", 35.0, -120.0),
        (1, "03:00:00", "A", "IW_GRDH_1S", "009.99", "QQ", 10.0, 10.0),
    ]
    out = []
    for off, hms, ser, pt, ipf, pol, lat, lon in rows:
        day = d_lo + dt.timedelta(days=off)
        stamp = f"{day:%Y%m%d}T{hms.replace(':', '')}"
        tag = pt.replace("_1S", "").replace("IW_", "")
        name = (f"S1{ser}_IW_{tag}_1SDV_{stamp}_{stamp}_054122_0694D3_"
                f"E1AD.SAFE")
        attrs = {"productType": pt, "platformSerialIdentifier": ser,
                 "processorVersion": ipf, "polarisationChannels": pol,
                 "operationalMode": "IW", "platformShortName": "SENTINEL-1",
                 "instrumentShortName": "SAR"}
        out.append(_prod(name, f"{day:%Y-%m-%d}T{hms}.820372Z", lat, lon,
                         attrs, dlat=1.5, dlon=2.0))
        if pt == "IW_GRDH_1S":
            # THE MIRROR. Same scene, productType `-COG`: it must never be
            # matched by the store's filter, and the smoke would count six
            # rows instead of four if it were.
            cog = dict(attrs)
            cog["productType"] = "IW_GRDH_1S-COG"
            out.append(_prod(name[:-5] + "_COG.SAFE",
                             f"{day:%Y-%m-%d}T{hms}.820372Z", lat, lon, cog,
                             dlat=1.5, dlon=2.0))
    return out


# ================================================================= cat_olci ==
def OLCI_PRODUCTS(adapter, d_lo, d_hi):
    """Five overpasses: two published twice (NR and NT), one NR only."""
    rows = [
        # day, hh:mm:ss, serial, baseline, timeliness pair, lat, lon
        (0, "04:02:57", "B", "004", ("NT", "NR"), 40.0, -9.5),   # coastal
        (0, "10:00:00", "A", "004", ("NT",), -25.0, -140.0),     # open ocean
        (0, "20:00:00", "A", "004", ("NR",), 55.0, 5.0),         # NR only
        (1, "01:00:00", "B", "003", ("NT", "NR"), 70.0, 179.0),  # seam
        (1, "02:00:00", "A", "009", ("ST",), 0.0, 0.0),          # unlisted qc
    ]
    out = []
    for off, hms, ser, base, tls, lat, lon in rows:
        day = d_lo + dt.timedelta(days=off)
        a = f"{day:%Y%m%d}T{hms.replace(':', '')}"
        b = f"{day:%Y%m%d}T{hms.replace(':', '')}"
        for k, tl in enumerate(tls):
            name = (f"S3{ser}_OL_2_WFR____{a}_{b}_"
                    f"{day:%Y%m%d}T10313{k}_0180_093_318_1800_MAR_O_{tl}_"
                    f"{base}.SEN3")
            attrs = {"productType": "OL_2_WFR___",
                     "platformSerialIdentifier": ser,
                     "baselineCollection": base, "timeliness": tl,
                     "platformShortName": "SENTINEL-3",
                     "instrumentShortName": "OLCI"}
            out.append(_prod(name, f"{day:%Y-%m-%d}T{hms}.991510Z", lat, lon,
                             attrs, dlat=5.0, dlon=6.0))
    return out


# ================================================= an ASF producer, canned ==
def asf_sources(adapter, root, d_lo, d_hi, make):
    """Canned ASF responses: `output=count` (plain text) and `output=jsonlite`.

    The fake honours `start`/`end` and refuses a `maxResults` above 250, the
    way the real endpoint does. `output=count` answers a bare integer through
    the `text_serve` hook, because it is the one endpoint of the eight that is
    not JSON.
    """
    gran = make(adapter, d_lo, d_hi)

    def hit(url):
        q = urllib.parse.parse_qs(urllib.parse.urlparse(url).query)
        lo = (q.get("start") or ["0000"])[0].replace("Z", "")
        hi = (q.get("end") or ["9999"])[0].replace("Z", "")
        mr = int((q.get("maxResults") or [250])[0])
        if mr > st.ASF_MAX:
            raise AssertionError(
                f"smoke: the adapter asked for maxResults={mr}, which the "
                f"real ASF refuses (<= {st.ASF_MAX})")
        if (q.get("dataset") or [""])[0] not in ("NISAR", ""):
            return []
        return [g for g in gran
                if lo <= g["startTime"].replace("Z", "") < hi]

    def text_serve(url):
        q = urllib.parse.parse_qs(urllib.parse.urlparse(url).query)
        if (q.get("output") or [""])[0] != "count":
            raise AssertionError(f"smoke: unexpected text request {url}")
        if "start" not in q:
            return str(len(gran))
        return str(len(hit(url)))

    def serve(method, url, body, headers=None):
        if method != "GET":
            raise AssertionError(f"smoke: unexpected {method} {url}")
        return {"results": hit(url)}, {}

    return record(adapter, d_lo, d_hi, serve, root, text_serve=text_serve)


NISAR_URL = ("https://nisar.asf.earthdatacloud.nasa.gov/NISAR/{coll}/{g}/"
             "{g}.h5")


def NISAR_GRANULES(adapter, d_lo, d_hi):
    """Six frames in two hours of one day, plus an empty hour after them."""
    rows = [
        # hh:mm:ss, class, polmode, collection tier, pge, lat, lon
        ("00:54:33", "PR", "NADV", "PROVISIONAL", "R05.02.3", 22.1, -110.4),
        ("00:55:10", "PR", "DHDH", "PROVISIONAL", "R05.02.3", -33.0, 18.0),
        ("00:56:00", "UR", "SHNA", "PROVISIONAL", "R05.02.3", 70.0, 179.3),
        ("01:10:00", "PR", "QQQQ", "BETA", "R05.00.8", 10.0, 10.0),
        ("01:11:00", "PR", "ZZZZ", "BETA", "R05.00.8", -5.0, -60.0),
        ("01:12:00", "PR", "DVDV", "EXPERIMENTAL", "R09.99.9", 0.0, 0.0),
    ]
    out = []
    day = d_lo
    for hms, cls, pol, tier, pge, lat, lon in rows:
        stamp = f"{day:%Y%m%d}T{hms.replace(':', '')}"
        g = (f"NISAR_L2_{cls}_GCOV_028_099_D_078_0005_{pol}_A_{stamp}_"
             f"{stamp}_P05023_N_P_J_001")
        coll = f"NISAR_L2_GCOV_{tier}_V1"
        out.append({
            "granuleName": g, "fileName": g + ".h5",
            "collectionName": coll, "pgeVersion": pge,
            "startTime": f"{day:%Y-%m-%d}T{hms}Z",
            "stopTime": f"{day:%Y-%m-%d}T{hms}Z",
            "processingLevel": "GCOV", "dataset": "NISAR",
            "instrument": "L-SAR", "polarization": None,
            "offNadirAngle": None, "pointingAngle": None,
            "downloadUrl": NISAR_URL.format(coll=coll, g=g),
            "browse": [f"https://nisar.asf.earthdatacloud.nasa.gov/BROWSE/"
                       f"{coll}/{g}/{g}_LATLON.png"],
            "nisar": {"additionalUrls": [
                f"https://nisar.asf.earthdatacloud.nasa.gov/"
                f"NISAR-JPL-PRIVATE-DATA/{coll}/{g}/{g}.context.json"]},
            "wkt": wkt(lat, lon, 1.2, 1.4),
        })
    return out

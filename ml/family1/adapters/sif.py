"""Solar-induced chlorophyll fluorescence soundings — photosynthesis seen
from orbit (family 1.0.tf, E-082 wave 6).

PLAIN ENGLISH. A leaf that is photosynthesising re-emits a faint red glow,
about one part in a hundred of the light it reflects. That glow is called
solar-induced chlorophyll fluorescence, SIF for short, and it is the only
thing a satellite can see that is emitted BY the photosynthesis itself rather
than inferred from how green the ground looks. TROPOMI — the Tropospheric
Monitoring Instrument, the spectrometer on the European Sentinel-5P satellite
— measures it every day over the whole land surface. This store keeps one row
per sounding: where, when, how much fluorescence, how uncertain, how cloudy,
and the two viewing angles that set how much canopy the instrument was
looking through.

THE SOURCE WAS CHOSEN BY MEASUREMENT, NOT BY PREFERENCE (all of it verified
from this sandbox on 2026-09-20; both candidates are anonymous over HTTPS):

  (a) ESA's S5P-PAL service, collection `L2B_SIF___` — CHOSEN.
      `https://data-portal.s5p-pal.com/api/s5p-l2` is a SpatioTemporal Asset
      Catalog (STAC) API. The collection's own temporal extent begins
      2018-04-30 and it holds **one netCDF file per day**: 402,064,007 bytes
      for 2023-07-01, 377,766,360 for 2026-08-15, 290,953,821 for its first
      day. The download URL redirects to an Open Telekom Cloud object store
      that answers an anonymous ranged GET with HTTP 206 and an exact
      Content-Length, so a day streams and a truncated day raises.
      A hosted lane therefore fetches ~0.4 GB a day and ~12 GB a month, one
      file at a time, and never holds more than one day on disk.

  (b) Caltech's ungridded 740 nm record (Koehler et al. 2018), CaltechDATA
      record `8hm1f-w5492`, licence "cc-zero" — DOCUMENTED AND NOT USED.
      An HTTPS path does exist (`https://data.caltech.edu/api/records/
      8hm1f-w5492/files/<name>/content`, a 302 to a signed OSN S3 URL that
      answers 206), but the archive is FOUR yearly gzipped tarballs and
      nothing finer: TROPO_SIF_ARCHIVE_2018-2019 63,300,122,916 B,
      2020 40,844,292,172 B, 2021 43,106,165,569 B, 2022 34,395,789,435 B —
      181,646,370,092 B in all, the note's 181.6 GB exactly. A `.tar.gz` is a
      single gzip stream with no member index, so one day cannot be reached
      without decompressing every byte before it: the smallest unit the
      archive can serve is 34 GB. The record also STOPS AT 2022, while
      S5P-PAL's L2B runs to the present. Per-day files are therefore NOT
      reachable, and the multi-gigabyte-per-day rule in the brief settles it.

  Not used either: S5P-PAL's per-orbit `L2__SIF___`, which is the same
  retrieval before the daily aggregation — 14-16 files and 3.2-3.7 GB a day
  (measured on 2018-04-30, 2023-07-01 and 2026-08-15), i.e. nine times the
  bytes for the same soundings.

THE RECORD'S FIRST DAY, MEASURED AND NOT AS THE NOTE HAS IT. The collection
extent says 2018-04-30 and the notes repeat it, but the FIRST L2B ITEM starts
2018-05-01T00:00:52 (the items from 2018-04-29 to 2018-05-05 are exactly
five, beginning on 05-01). 2018-04-30 is the first day of the per-orbit
`L2__SIF___` collection, which has 16 granules that day. The daily product
therefore begins one day later than the ledger row says.

WHAT ONE FILE HOLDS, read off the real 2023-07-01 granule with netCDF4:
NETCDF4, `n_elem` = 3,322,606 soundings, groups `/PRODUCT`,
`/PRODUCT/SUPPORT_DATA/{DETAILED_RESULTS,GEOLOCATIONS,INPUT_DATA}` and
`/METADATA/ALGORITHM_SETTINGS`. The variables this adapter reads:
  `/PRODUCT/time`          one value, "seconds since 2010-01-01 00:00:00",
                           the day's reference instant
  `/PRODUCT/delta_time`    int32 per sounding, "milliseconds since
                           <the day>T00:00:00 UTC" — the epoch is read from
                           the variable's own `units`, never assumed, because
                           a file spans midnight (2023-07-01T00:24 ..
                           2023-07-02T00:05, delta_time 0.92 h .. 23.58 h)
  `SIF_743` `SIF_Corr_743` `SIF_ERROR_743`   the 743-758 nm fitting window
  `SIF_735` `SIF_Corr_735` `SIF_ERROR_735`   the wider 735-758 nm window
  `latitude` `longitude`, and in GEOLOCATIONS `solar_zenith_angle`,
  `viewing_zenith_angle`, `latitude_bounds`/`longitude_bounds` (four corners
  per sounding), and in INPUT_DATA `cloud_fraction_L2` and `LC_MASK`.

THE 743-758 nm WINDOW IS THE ONE STORED, and that is a measurement too. Both
windows retrieve fluorescence at the same 740 nm reference wavelength (the
file's own `SIF reference wavelength (nm)` setting) with different fitting
intervals; the 743 window is the product's baseline (three polynomial degrees
and four state vectors against the 735 window's seven). On 2023-07-01 the 743
triplet is complete for all 3,322,606 soundings while `SIF_735` carries the
netCDF4 default float fill 9.969209968386869e+36 on 9,504 of them (0.29 %)
AND DECLARES NO `_FillValue` ATTRIBUTE — so a reader that trusts the
attributes gets 1e37 as a value. This adapter treats any magnitude at or
above `FILL_FLOOR` (1e30) as missing in EVERY channel and counts it
(`fill_valued`), which is what that measurement is for.

WHAT A ROW IS.
  time_s    the day's reference time plus the sounding's own `delta_time`,
            both through `f10b._cf_time_to_seconds` on their OWN `units`
            strings, rounded to the second. int32 seconds since 1982-01-01
            (schema 2); `first_year` is 2014 because OCO-2 is a declared
            platform, and 2014 is inside int32's range.
  lat, lon  `latitude`, `longitude`, longitude wrapped to [-180, 180).
  values    C = 6, in the note's order: sif_740 (`SIF_743`), sif_740_daily
            (`SIF_Corr_743`, the daylength-corrected value), sif_740_error
            (`SIF_ERROR_743`, the 1-sigma retrieval error), cloud_fraction
            (`cloud_fraction_L2`), solar_zenith, viewing_zenith.
  platform  platform_hash("TROPOMI"), or "OCO-2" / "OCO-3" when those are
            built; `platform_meta = True` and `platforms()` names each
            instrument's footprint and retrieval wavelength.
  qc        DERIVED, and the derivation is here because the product ships no
            per-sounding quality flag at all — its quality control is applied
            UPSTREAM and recorded only as thresholds in
            `/METADATA/ALGORITHM_SETTINGS`: `Cloud fraction threshold` 0.8,
            `SZA threshold` 70.0, `VZA threshold` 60.0, `Quality level
            threshold` 80. Those thresholds ARE the data's own limits on the
            day measured: cloud fraction reaches exactly 0.8000, solar zenith
            exactly 70.0000 and viewing zenith exactly 60.0000, and no
            sounding exceeds them. So `qc` grades the cloud fraction in
            quarters of the file's OWN threshold T — 0 for <= T/4, 1 for
            <= T/2, 2 for <= T, 3 for a sounding past T or past either angle
            threshold (counted by which one) — and a file whose
            ALGORITHM_SETTINGS does not carry all three numbers is a REFUSAL,
            not a guess. The quartering is this store's choice and is stated;
            the numbers it quarters are the producer's.

THE FOOTPRINT IS A CONSTANT, AND THE FILE COULD HAVE GIVEN A PER-ROW ONE.
`latitude_bounds`/`longitude_bounds` carry four real corners per sounding,
and on 2023-07-01 the square root of the corner polygon's area is 4.50 km at
the 1st percentile, 5.10 km at the median and 8.29 km at the 99th, i.e.
log2(side / 27.83 km) from -2.63 to -1.75 with a median of -2.45 — against
the nominal 7 x 3.5 km's geometric mean of -2.4912, which is what `log2_fp`
carries. The tier-P layout stores ONE (log2_fp, log2_dt) pair for a whole
store — `fp.npy` is filled from `ad.log2_fp` by both assemblers and
`check_store` ASSERTS the columns are constant — so a per-row footprint is
not representable today. The corner geometry is not thrown away: the probe
measures its distribution (`footprint_km`, `log2_fp_measured`) so the main
session can see what a per-row footprint would buy before anyone changes the
layout.

THE PROBE, 2023-07, MEASURED 2026-09-20 FROM THIS SANDBOX (`--stage probe
--probe-month 2023-07`, ml/family1/probes/sif_2023-07.json): all 31 days,
12,313,108,712 bytes streamed and parsed in 578 s (9.6 min, 21.3 MB/s), one
day on disk at a time; **101,722,256 rows** in the month from 101,753,881
soundings (the last day's file spills 31,625 soundings into 1 August), 3.28 M
a day, **121.0 source bytes a row** against 39 stored. NOT ONE NaN and NOT
ONE value out of bounds in any of the six channels; three soundings in the
whole month carry the undeclared fill. qc 0 (cloud fraction <= 0.2) 64.9 M,
qc 1 (<= 0.4) 16.4 M, qc 2 (<= 0.8) 20.4 M, qc 3 none — the producer's own
filter really does hold. The corner-derived footprint's median is 5.048 ..
5.121 km across the 31 days, log2 -2.463 .. -2.442.

THE STORE IS BIGGER THAN THE LEDGER SAYS, AND THE CLOUD CUT IS WHY. At
3.28 M rows a day the record costs **46.8 GB a year** and **245.5 GB from
2018-05-01 to the end of the probe's window** (1,918 days) with no cloud cut
at all — about 392 GB to today. The ledger's "60-160 GB, the cloud cut
decides" is the figure for a CUT AND A SHORTER RECORD: keeping only
cloud fraction <= 0.2 keeps 63.9 % of the soundings and 156.8 GB of that
1,918-day span, and <= 0.1 keeps 51.5 % and 126.3 GB. The probe writes all
five cuts into `store_projection`, so the decision is priced rather than
argued.

OCO-2 AND OCO-3 ARE WIRED AND GATED. The ledger names them as the second and
third platforms of this store. Their SIF Lite files live at GES DISC, which
is UNAPPROVED on this account (the same block that holds `xco2` and `irtb`),
so the default build fetches TROPOMI only, counts the skip
(`platforms_skipped_no_ges_disc`) and says so in `notes`, and setting
`SIF_OCO=1` turns the code path on. With the flag set the adapter checks for
Earthdata credentials at `fetch_preflight` — where the inputs are all it has
cost — and refuses there rather than at hour three.

TWO ENVIRONMENT KNOBS, both read at CONSTRUCTION time so the fresh adapter
`stage_probe` builds for itself sees them, exactly as `swot` reads
`SWOT_MAX_PASSES`. `SIF_MAX_DAYS` caps the days a single fetch reads (0 =
every day of the window) — a month is ~12 GB, which a probe should not always
spend — and `SIF_OCO` is the platform gate above. A capped run writes the cap
into `notes`, which reaches store.json, so a restricted store can never look
like a whole one.
"""
import calendar
import datetime as dt
import json
import os
import sys
import time
import urllib.parse

import numpy as np

import build_family10_stores as f10b
from family1.adapters import _common as cm

STAC = "https://data-portal.s5p-pal.com/api/s5p-l2"
COLLECTION = "L2B_SIF___"
ORBIT_COLLECTION = "L2__SIF___"          # the per-orbit product, not read
CALTECH = ("https://data.caltech.edu/api/records/8hm1f-w5492/files/"
           "TROPO_SIF_ARCHIVE_<year>.tar.gz/content")

# The netCDF4 default float fill, which this product writes WITHOUT declaring
# a `_FillValue` (measured on `SIF_735`, 9,504 of 3,322,606 soundings on
# 2023-07-01). Any magnitude at or above this is missing, in every channel.
FILL_FLOOR = 1e30

# The 743-758 nm fitting window; see the docstring for why it and not 735.
WINDOW = "743"
CHANNELS = (("sif_740", "mW m-2 sr-1 nm-1", -20.0, 20.0),
            ("sif_740_daily", "mW m-2 sr-1 nm-1", -20.0, 20.0),
            ("sif_740_error", "mW m-2 sr-1 nm-1", 0.0, 20.0),
            ("cloud_fraction", "1", 0.0, 1.0),
            ("solar_zenith", "degree", 0.0, 90.0),
            ("viewing_zenith", "degree", 0.0, 90.0))

PRODUCT = "/PRODUCT"
GEO = "/PRODUCT/SUPPORT_DATA/GEOLOCATIONS"
INPUT = "/PRODUCT/SUPPORT_DATA/INPUT_DATA"
SETTINGS = "/METADATA/ALGORITHM_SETTINGS"
# the three thresholds `qc` is graded against, by the attribute name the file
# itself uses for each
THRESHOLDS = {"cloud": "Cloud fraction threshold",
              "sza": "SZA threshold",
              "vza": "VZA threshold"}

# The instruments this store can hold. `footprint_km` is (along-track x
# across-track) at nadir as each mission's own documentation states it; the
# store's log2_fp is TROPOMI's geometric mean over the family's 27.83 km
# reference cell.
PLATFORMS = {
    "TROPOMI": {"mission": "Sentinel-5P", "instrument": "TROPOMI",
                "footprint_km": [7.0, 3.5],
                "wavelength_nm": 740.0,
                "fitting_window_nm": [743.0, 758.0],
                "record_from": "2018-05-01",
                "source": "ESA S5P-PAL L2B_SIF___ (anonymous)"},
    "OCO-2": {"mission": "OCO-2", "instrument": "OCO-2 grating spectrometer",
              "footprint_km": [2.25, 1.3],
              "wavelength_nm": 740.0,
              "fitting_window_nm": [757.0, 771.0],
              "record_from": "2014-09-06",
              "source": "NASA GES DISC OCO2_L2_Lite_SIF v11.2r (Earthdata)"},
    "OCO-3": {"mission": "OCO-3 (International Space Station)",
              "instrument": "OCO-3 grating spectrometer",
              "footprint_km": [2.25, 1.3],
              "wavelength_nm": 740.0,
              "fitting_window_nm": [757.0, 771.0],
              "record_from": "2019-08-06",
              "source": "NASA GES DISC OCO3_L2_Lite_SIF v11r (Earthdata)"},
}
FIRST_YEAR = 2014                       # OCO-2; int32 seconds reach it easily
# The first day the DAILY L2B product has an item for, measured on the STAC
# 2026-09-20: the collection's declared extent and the ledger both say
# 2018-04-30, which is the first day of the PER-ORBIT product.
RECORD_FIRST_DAY = dt.date(2018, 5, 1)
NOMINAL_KM = float(np.sqrt(PLATFORMS["TROPOMI"]["footprint_km"][0]
                           * PLATFORMS["TROPOMI"]["footprint_km"][1]))
REFERENCE_KM = 27.83                    # the family's cell, E-078 §2
EARTH_R_KM = 6371.0088


class FormatError(ValueError):
    """A STAC answer or a granule that is not the product measured above."""


# ================================================================== STAC ====
def stac_items(d_lo, d_hi, collection=COLLECTION, attempts=4, count=None,
               limit=500):
    """Every item of `collection` overlapping [d_lo, d_hi] (dates).

    ANONYMOUS. The API filters on the item's own start/end interval, and an
    L2B item runs from one day's first orbit to the next day's first, so the
    window is widened by a day at each end and the caller matches items to
    days by their `start_datetime`. An empty answer is returned as an empty
    list and the CALLER decides whether that is a refusal.
    """
    lo = (d_lo - dt.timedelta(days=1)).isoformat() + "T00:00:00Z"
    hi = (d_hi + dt.timedelta(days=1)).isoformat() + "T23:59:59Z"
    q = {"datetime": f"{lo}/{hi}", "limit": int(limit)}
    url = (f"{STAC}/collections/{urllib.parse.quote(collection)}/items?"
           + urllib.parse.urlencode(q))
    raw, why = cm.get_bytes(url, attempts=attempts)
    if raw is None:
        raise FormatError(f"the S5P-PAL STAC answered {why} for {url}")
    if count is not None:
        count(len(raw))
    try:
        js = json.loads(raw)
    except ValueError as e:
        raise FormatError(f"{url}: not JSON ({e})") from None
    if "features" not in js:
        raise FormatError(f"{url}: a STAC item page with no `features` "
                          f"(keys {sorted(js)[:8]})")
    ctxt = js.get("context") or {}
    if ctxt.get("matched") is not None and \
            int(ctxt["matched"]) > len(js["features"]):
        raise FormatError(
            f"{url}: the service matched {ctxt['matched']} item(s) and "
            f"returned {len(js['features'])} — raise `limit` rather than "
            f"quietly building from a page")
    return js["features"]


def parse_items(features):
    """STAC features -> ({date: entry}, counts). Refuses a reshaped item."""
    out, counts = {}, {"items": len(features)}
    for f in features:
        p = f.get("properties") or {}
        start = p.get("start_datetime") or p.get("datetime")
        if not start:
            raise FormatError(f"{f.get('id')!r}: no start_datetime")
        a = (f.get("assets") or {}).get("product")
        if not a or not a.get("href"):
            raise FormatError(f"{f.get('id')!r}: no `product` asset href "
                              f"(assets {sorted(f.get('assets') or {})})")
        day = dt.date.fromisoformat(str(start)[:10])
        if day in out:
            raise FormatError(
                f"two L2B items start on {day}: {out[day]['id']} and "
                f"{f.get('id')} — the product is one file a day and this "
                f"adapter would silently read only one of them")
        out[day] = {"id": f.get("id"), "url": a["href"],
                    "bytes": int(a.get("file:size") or 0),
                    "name": a.get("file:local_path") or (str(f.get("id"))
                                                         + ".nc"),
                    "start": str(start),
                    "end": str(p.get("end_datetime") or ""),
                    "processor": p.get("s5p:processor_version")}
    return out, counts


# ============================================================== one granule ==
def thresholds_of(ds):
    """The producer's own quality thresholds, out of ALGORITHM_SETTINGS.

    A file that does not carry all three is a REFUSAL rather than a guess:
    `qc` is graded against these numbers and nothing else, so inventing one
    would put a fabricated grade in the store (ADAPTER_CONTRACT rule 4).
    """
    try:
        g = ds[SETTINGS]
    except (IndexError, KeyError):
        raise FormatError(f"no `{SETTINGS}` group; the file's groups are "
                          f"{sorted(ds.groups)}") from None
    out = {}
    for key, attr in THRESHOLDS.items():
        if attr not in g.ncattrs():
            raise FormatError(
                f"`{SETTINGS}` does not carry {attr!r} — `qc` is graded "
                f"against the producer's own thresholds and this adapter "
                f"refuses to invent one. The group holds "
                f"{sorted(g.ncattrs())}")
        try:
            out[key] = float(str(getattr(g, attr)).strip())
        except ValueError:
            raise FormatError(f"{attr} = {getattr(g, attr)!r} is not a "
                              f"number") from None
        if not (out[key] > 0):
            raise FormatError(f"{attr} = {out[key]} is not positive")
    return out


def nc_check(ds):
    """An open L2B granule against the layout measured above. Raises."""
    if "n_elem" not in ds.dimensions:
        raise FormatError(f"no `n_elem` dimension; dimensions are "
                          f"{sorted(ds.dimensions)}")
    need = {PRODUCT: ["time", "delta_time", "latitude", "longitude",
                      f"SIF_{WINDOW}", f"SIF_Corr_{WINDOW}",
                      f"SIF_ERROR_{WINDOW}"],
            GEO: ["solar_zenith_angle", "viewing_zenith_angle",
                  "latitude_bounds", "longitude_bounds"],
            INPUT: ["cloud_fraction_L2"]}
    for grp, names in need.items():
        try:
            g = ds[grp]
        except (IndexError, KeyError):
            raise FormatError(f"no `{grp}` group") from None
        missing = [n for n in names if n not in g.variables]
        if missing:
            raise FormatError(f"{grp}: variables {missing} are absent; it "
                              f"holds {sorted(g.variables)}")
    for nm in ("time", "delta_time"):
        u = str(getattr(ds[PRODUCT].variables[nm], "units", ""))
        if " since " not in u.lower():
            raise FormatError(f"{nm} units {u!r} carry no epoch")
    n = int(len(ds.dimensions["n_elem"]))
    return {"n_elem": n,
            "time_units": str(ds[PRODUCT].variables["time"].units),
            "delta_time_units":
                str(ds[PRODUCT].variables["delta_time"].units),
            "format": ds.data_model,
            "title": str(getattr(ds, "title", "")),
            "processor_version": str(getattr(ds, "processor_version", "")),
            "has_corners": int(len(ds.dimensions.get("ncorner", [])) or 0)}


def _var(g, name):
    """One variable as float64 with the declared fill AND the undeclared
    netCDF4 default fill turned into NaN."""
    v = g.variables[name]
    a = np.asarray(np.ma.filled(v[:], np.nan), np.float64)
    fill = getattr(v, "_FillValue", None)
    if fill is not None:
        a = np.where(a == np.float64(fill), np.nan, a)
    a = np.where(np.abs(a) >= FILL_FLOOR, np.nan, a)
    a[~np.isfinite(a)] = np.nan
    return a


def footprint_km(lat, lon, lat_b, lon_b):
    """sqrt(area) in km for every sounding, from its four corners.

    The corners are projected onto a local tangent plane at the sounding's own
    centre — which is also what wraps a footprint that straddles the dateline
    back into one piece, the difference between a 5 km pixel and a 376 km one
    (measured: without the wrap the maximum reads 376.067 km).
    """
    dlon = lon_b - lon[:, None]
    dlon = (dlon + 180.0) % 360.0 - 180.0
    x = np.deg2rad(dlon) * np.cos(np.deg2rad(lat[:, None])) * EARTH_R_KM
    y = np.deg2rad(lat_b - lat[:, None]) * EARTH_R_KM
    area = 0.5 * np.abs(np.sum(x * np.roll(y, -1, axis=1)
                               - np.roll(x, -1, axis=1) * y, axis=1))
    return np.sqrt(area)


def read_day(path, counts):
    """One L2B granule -> (t, lat, lon, values[n, 6], qc, meta), or None.

    Raises `FormatError` on a file that is not the product; an unreadable file
    is the caller's `note_absent`.
    """
    import netCDF4
    ds = netCDF4.Dataset(path)
    try:
        meta = nc_check(ds)
        thr = thresholds_of(ds)
        P, G, I = ds[PRODUCT], ds[GEO], ds[INPUT]
        t0 = f10b._cf_time_to_seconds(
            np.asarray(P.variables["time"][:], np.float64)[:1],
            meta["time_units"])[0]
        dtm = np.asarray(P.variables["delta_time"][:], np.float64)
        # the delta_time epoch is the file's OWN day, not `time`'s epoch
        t = f10b._cf_time_to_seconds(dtm, meta["delta_time_units"])
        lat = _var(P, "latitude")
        lon = _var(P, "longitude")
        vals = np.stack([_var(P, f"SIF_{WINDOW}"),
                         _var(P, f"SIF_Corr_{WINDOW}"),
                         _var(P, f"SIF_ERROR_{WINDOW}"),
                         _var(I, "cloud_fraction_L2"),
                         _var(G, "solar_zenith_angle"),
                         _var(G, "viewing_zenith_angle")], axis=-1)
        lat_b = _var(G, "latitude_bounds")
        lon_b = _var(G, "longitude_bounds")
    finally:
        ds.close()
    meta["thresholds"] = thr
    meta["reference_time_s"] = float(t0)
    n = int(lat.size)
    counts["soundings_in_file"] = counts.get("soundings_in_file", 0) + n
    fillish = int((~np.isfinite(vals)).sum())
    if fillish:
        counts["fill_valued"] = counts.get("fill_valued", 0) + fillish
    lon = np.where(lon >= 180.0, lon - 360.0, lon)
    ok = (np.isfinite(t) & np.isfinite(lat) & np.isfinite(lon)
          & np.isfinite(vals[..., 0]))
    counts["soundings_no_sif"] = counts.get("soundings_no_sif", 0) + \
        int((~ok).sum())
    # THE GRADE, against the file's own three thresholds (see the docstring).
    cf, sza, vza = vals[..., 3], vals[..., 4], vals[..., 5]
    T = thr["cloud"]
    qc = np.where(cf <= T / 4.0, 0, np.where(cf <= T / 2.0, 1,
                  np.where(cf <= T, 2, 3))).astype(np.int64)
    qc[~np.isfinite(cf)] = 2               # no cloud fraction: not assessed
    over_c = np.isfinite(cf) & (cf > T)
    over_s = np.isfinite(sza) & (sza > thr["sza"])
    over_v = np.isfinite(vza) & (vza > thr["vza"])
    for key, m in (("cloud", over_c), ("solar_zenith", over_s),
                   ("viewing_zenith", over_v)):
        if m.any():
            d = counts.setdefault("beyond_producer_threshold", {})
            d[key] = d.get(key, 0) + int(m.sum())
    qc[over_s | over_v] = 3
    fp = footprint_km(lat, lon, lat_b, lon_b)
    meta["footprint_km"] = _percentiles(fp[ok])
    meta["log2_fp_measured"] = (
        None if not ok.any() else
        round(float(np.log2(np.median(fp[ok]) / REFERENCE_KM)), 4))
    meta["cloud_fraction_cut"] = _cloud_cut(cf[ok], T)
    meta["valid_fraction"] = round(float(ok.mean()), 6) if n else None
    hist = counts.setdefault("qc_grade", {})
    for u, k in zip(*np.unique(qc[ok], return_counts=True)):
        hist[str(int(u))] = hist.get(str(int(u)), 0) + int(k)
    counts["rows_kept"] = counts.get("rows_kept", 0) + int(ok.sum())
    if not ok.any():
        return None, meta
    return (t[ok], lat[ok], lon[ok], vals[ok],
            qc[ok].astype(np.uint8)), meta


def _percentiles(a):
    if not a.size:
        return None
    q = np.percentile(a, [1, 25, 50, 75, 99])
    return {"min": round(float(a.min()), 3), "p1": round(float(q[0]), 3),
            "p25": round(float(q[1]), 3), "median": round(float(q[2]), 3),
            "p75": round(float(q[3]), 3), "p99": round(float(q[4]), 3),
            "max": round(float(a.max()), 3)}


def _cloud_cut(cf, threshold):
    """What a cloud cut would keep — the note's 60-160 GB hinges on it."""
    cf = cf[np.isfinite(cf)]
    if not cf.size:
        return None
    return {f"le_{c:g}": round(float((cf <= c).mean()), 6)
            for c in (0.1, 0.2, 0.3, 0.5, threshold)}


# ================================================================ adapter ===
class SIFAdapter(f10b.SourceAdapter):
    store = "sif"
    title = ("Solar-induced chlorophyll fluorescence at 740 nm: TROPOMI "
             "(Sentinel-5P) daily L2B soundings from ESA's S5P-PAL service, "
             "one row per sounding; OCO-2 and OCO-3 SIF Lite gated on "
             "SIF_OCO=1 until GES DISC is approved")
    family = "1tf"
    distribution = "public"
    licence = {"name": "Copernicus open data (Sentinel Data Legal Notice)",
               "redistribution": "attribution",
               "redistribution_confirmed": True,
               "derived_works": "free",
               "attribution": ("Contains modified Copernicus Sentinel-5P data "
                               "processed by the S5P-PAL service (TROPOSIF "
                               "L2B, processor 01.00.01); the retrieval is "
                               "Koehler, P., C. Frankenberg et al. (2018), "
                               "Global retrievals of solar-induced "
                               "chlorophyll fluorescence with TROPOMI: first "
                               "results and intersensor comparison to OCO-2, "
                               "Geophys. Res. Lett. 45, 10456-10463"),
               "terms": ("the collection's own STAC licence link is "
                         "https://sentinels.copernicus.eu/documents/247904/"
                         "690755/Sentinel_Data_Legal_Notice, which grants "
                         "free access and redistribution with attribution. "
                         "Caltech's equivalent ungridded record is CC0 "
                         "('cc-zero' in its CaltechDATA metadata) and is the "
                         "documented alternative source")}
    time_dtype = "int32"
    platform_meta = True
    credentials = ()                      # TROPOMI is anonymous; see SIF_OCO
    channels = CHANNELS
    # 7 x 3.5 km -> geometric mean 4.9497 km over the family's 27.83 km cell
    log2_fp = float(np.log2(NOMINAL_KM / REFERENCE_KM))          # -2.4912
    log2_dt = -4.0                        # an instantaneous sounding
    per_year = True
    first_year = FIRST_YEAR
    fetch_month_scope = "month"
    qc_policy = (
        "The product ships NO per-sounding quality flag: its quality control "
        "is applied upstream and survives only as thresholds in the file's "
        "own /METADATA/ALGORITHM_SETTINGS (cloud fraction 0.8, solar zenith "
        "70 deg, viewing zenith 60 deg on the granule measured, and those are "
        "exactly the data's own limits there). qc therefore grades the cloud "
        "fraction in quarters of the file's OWN threshold T: 0 for <= T/4, 1 "
        "for <= T/2, 2 for <= T, 3 for a sounding past T or past either angle "
        "threshold, counted by which one (`beyond_producer_threshold`); a "
        "sounding with no cloud fraction is qc 2, not assessed. A file whose "
        "ALGORITHM_SETTINGS lacks any of the three numbers is a REFUSAL, not "
        "a guess. A sounding with no fluorescence, no time or no position is "
        "dropped and counted; any magnitude >= 1e30 is the netCDF4 default "
        "fill this product writes WITHOUT declaring it and becomes NaN for "
        "that channel, counted as `fill_valued`. A value outside its "
        "channel's bounds becomes NaN for that channel alone and is counted, "
        "never clipped.")
    sources = (f"{STAC}/collections/{COLLECTION}/items (anonymous STAC)",
               "https://data-portal.s5p-pal.com/download/s5p-l2/<uuid> "
               "(302 to an anonymous object store, HTTP 206, exact "
               "Content-Length)",
               CALTECH + " (the CC0 alternative; four yearly tarballs, "
                         "not used)")
    verified = (
        "2026-09-20 from this sandbox, anonymously. The S5P-PAL STAC API "
        "(https://data-portal.s5p-pal.com/api/s5p-l2) and its L2B_SIF___ "
        "collection: temporal extent 2018-04-30 -> open, ONE item a day, "
        "402,064,007 B for 2023-07-01, 377,766,360 B for 2026-08-15, and the "
        "first five items starting 2018-05-01 .. 2018-05-05 (so the DAILY "
        "product begins 2018-05-01, a day after the extent and the ledger "
        "say). The per-orbit L2__SIF___ collection for the same three days: "
        "16 / 14 / 14 granules and 3.22 / 3.69 / 3.69 GB a day. The "
        "2023-07-01 granule downloaded whole (402,064,007 B) and opened with "
        "netCDF4: 3,322,606 soundings, the variable and attribute inventory "
        "in the module docstring, SIF_735 carrying 9,504 undeclared fill "
        "values against SIF_743's none, cloud fraction / solar zenith / "
        "viewing zenith reaching exactly the 0.8 / 70.0 / 60.0 of "
        "ALGORITHM_SETTINGS, and a corner-derived footprint whose median is "
        "5.098 km (log2 -2.4488 against the nominal -2.4912). Caltech's "
        "CaltechDATA record 8hm1f-w5492 ('cc-zero'): four yearly tarballs, "
        "181,646,370,092 B in all, 2018-2022 only, served over HTTPS through "
        "a signed redirect that honours Range (206) but is a single gzip "
        "stream with no member index, so one day cannot be reached without "
        "the whole 34-63 GB year.")
    notes = ""
    smoke_window = ("2023-07-01", "2023-07-03")
    smoke_probe_month = "2023-07"

    def __init__(self):
        try:
            self.max_days = int(os.environ.get("SIF_MAX_DAYS") or 0)
        except ValueError:
            sys.exit("SIF_MAX_DAYS must be an integer (0 = every day)")
        self.oco = (os.environ.get("SIF_OCO") or "") == "1"
        self._items = None
        self.notes = (
            "TROPOMI only. The ledger's OCO-2 and OCO-3 platforms are wired "
            "and GATED on SIF_OCO=1, because GES DISC (the NASA data centre "
            "that serves OCO SIF Lite) is unapproved on this account — the "
            "same block that holds `xco2` and `irtb`; the skip is counted as "
            "`platforms_skipped_no_ges_disc`. The daily L2B product begins "
            "2018-05-01, one day after the collection's declared extent and "
            "the ledger's 2018-04-30, which is the per-orbit product's first "
            "day (measured). The 743-758 nm fitting window is stored and the "
            "735-758 nm one is not; see the module docstring. The stored "
            "footprint is the instrument's nominal 7 x 3.5 km geometric mean "
            "because the tier-P layout carries ONE (log2_fp, log2_dt) pair "
            "per store, while the granule carries four real corners per "
            "sounding whose distribution the probe records."
            + (f" RESTRICTED FETCH: SIF_MAX_DAYS={self.max_days} — this run "
               f"reads at most {self.max_days} day(s) of its window and is "
               f"NOT the whole product." if self.max_days else "")
            + (" SIF_OCO=1: the OCO platforms are enabled." if self.oco
               else ""))

    # ------------------------------------------------------------- listing --
    def items(self, ctx):
        """{date: entry} for the whole requested window, listed once."""
        if self._items is not None:
            return self._items
        if ctx.source_dir:
            root = os.path.join(ctx.source_dir, self.store)
            p = os.path.join(root, "items.json")
            if not os.path.exists(p):
                sys.exit(f"REFUSING sif: no {p} — the smoke's synthetic STAC "
                         f"answer is missing")
            with open(p, "rb") as fh:
                raw = fh.read()
            ctx.count_bytes(len(raw))
            feats = json.loads(raw)["features"]
        else:
            try:
                feats = stac_items(ctx.d_lo, ctx.d_hi, count=ctx.count_bytes,
                                   attempts=ctx.a.attempts)
            except FormatError as e:
                sys.exit(f"REFUSING sif: {e}")
            if not feats:
                sys.exit(f"REFUSING sif: the S5P-PAL STAC lists no "
                         f"{COLLECTION} item between {ctx.d_lo} and "
                         f"{ctx.d_hi} — an empty listing is a refusal "
                         f"(ml/CLAUDE.md, the 2026-09-14 rule)")
        try:
            by_day, counts = parse_items(feats)
        except FormatError as e:
            sys.exit(f"REFUSING sif: {e}")
        self._items = (by_day, counts)
        return self._items

    def _days(self, ctx, d_lo, d_hi):
        by_day, _ = self.items(ctx)
        days = [d for d in sorted(by_day) if d_lo <= d <= d_hi]
        if self.max_days:
            days = days[:self.max_days]
        return days

    def _get(self, ctx, day, entry):
        """The day's granule on local disk -> (path, delete_after)."""
        if ctx.source_dir:
            p = os.path.join(ctx.source_dir, self.store, entry["name"])
            if not os.path.exists(p):
                raise f10b._NotFound(p)
            ctx.count_bytes(os.path.getsize(p))
            return p, False
        dest = os.path.join(ctx.scratch, self.store, entry["name"])
        got = f10b.fetch_first([entry["url"]], dest,
                               attempts=max(1, ctx.a.attempts))
        if got is None:
            raise f10b._NotFound(entry["url"])
        return dest, True

    # ---------------------------------------------------------------- rows --
    def _rows(self, ctx, label, d_lo, d_hi, t_lo, t_hi):
        by_day, icounts = self.items(ctx)
        days = self._days(ctx, d_lo, d_hi)
        counts = {"days_listed": len(by_day), "days_wanted": len(days),
                  "max_days": self.max_days}
        f10b._merge_counts(counts, dict(icounts))
        if not self.oco:
            counts["platforms_skipped_no_ges_disc"] = len(PLATFORMS) - 1
        per_day, t0 = [], time.time()
        plat = f10b.platform_hash("TROPOMI")
        for day in days:
            e = by_day[day]
            try:
                path, tmp = self._get(ctx, day, e)
            except f10b._NotFound:
                ctx.note_absent(label, f"{e['name']}: listed in the S5P-PAL "
                                       f"STAC, and 404")
                continue
            except IOError as x:
                ctx.note_absent(label, f"{e['name']}: {type(x).__name__}: {x}")
                continue
            c = {}
            try:
                r, meta = read_day(path, c)
            except FormatError as x:
                sys.exit(f"REFUSING sif: {e['name']}: {x}")
            except (OSError, IOError) as x:
                ctx.note_absent(label, f"{e['name']}: not a readable netCDF "
                                       f"({type(x).__name__}: {x}) — a "
                                       f"truncated download")
                continue
            finally:
                if tmp and os.path.exists(path):
                    os.remove(path)
            f10b._merge_counts(counts, c)
            meta["day"] = str(day)
            meta["bytes"] = e["bytes"]
            per_day.append(meta)
            if r is None:
                continue
            t, lat, lon, vals, qc = r
            oob = self.mask_bounds(vals)
            if oob:
                f10b._merge_counts(counts, {"out_of_bounds": oob})
            inside = (t >= t_lo) & (t <= t_hi)
            counts["rows_outside_window"] = counts.get(
                "rows_outside_window", 0) + int((~inside).sum())
            if not inside.all():
                t, lat, lon, vals, qc = (x[inside] for x in
                                         (t, lat, lon, vals, qc))
            if not t.size:
                continue
            counts["rows_in_window"] = counts.get("rows_in_window", 0) + \
                int(t.size)
            p = np.full(t.size, plat, np.int64)
            yield label, self.pack(t, lat, lon, vals, p, qc), None
        counts["fetch_seconds"] = round(time.time() - t0, 1)
        counts["per_day"] = per_day[:40]
        proj = store_projection(counts, per_day, self.C, d_hi)
        # A LIST OF ONE, not a dict: `f10b._merge_counts` merges a dict ONE
        # level deep and ADDS its values, so a nested dict carrying a string
        # raises there (the shape `swot` uses for `storage_options`).
        counts["store_projection"] = [proj] if proj else []
        counts["note_estimate"] = [NOTE_ESTIMATE]
        yield label, None, counts

    # ------------------------------------------------------------ contract --
    def fetch_preflight(self, ctx):
        """The OCO gate, where the inputs are all it has cost (§0.3).

        TROPOMI needs no credential at all. The OCO half needs Earthdata
        Login AND a GES DISC approval this account does not have, so a run
        that asks for it without credentials is refused before its first
        byte rather than after a listing.
        """
        if not self.oco or ctx.source_dir:
            return None
        if not cm.earthdata_ready():
            sys.exit(
                "REFUSING sif: SIF_OCO=1 asks for the OCO-2 and OCO-3 SIF "
                "Lite platforms, which live at GES DISC behind Earthdata "
                "Login, and this process can authenticate in NEITHER way: no "
                "EARTHDATA_USERNAME / EARTHDATA_PASSWORD in the environment "
                "and no netrc naming urs.earthdata.nasa.gov. Note that "
                "credentials are not enough on their own — GES DISC is "
                "unapproved on this account (the block that also holds "
                "`xco2` and `irtb`) and needs one click in Chris's Earthdata "
                "profile. Unset SIF_OCO to build TROPOMI alone. Nothing has "
                "been fetched.")
        return None

    def index(self, ctx):
        by_day, counts = self.items(ctx)
        days = sorted(by_day)
        nb = sum(e["bytes"] for e in by_day.values())
        out = {"dataset": f"S5P-PAL {COLLECTION} (TROPOSIF L2B)",
               "url": f"{STAC}/collections/{COLLECTION}",
               "collection": COLLECTION,
               "orbit_collection_not_read": ORBIT_COLLECTION,
               "alternative_source_not_read": CALTECH,
               "days_listed": len(days),
               "first_day": str(days[0]) if days else None,
               "last_day": str(days[-1]) if days else None,
               "bytes": int(nb),
               "bytes_per_day_mean": (round(nb / len(days), 1) if days
                                      else None),
               "max_days": self.max_days,
               "platforms_enabled": (sorted(PLATFORMS) if self.oco
                                     else ["TROPOMI"]),
               "oco_enabled": self.oco,
               "other": counts,
               "window": [str(ctx.d_lo), str(ctx.d_hi)]}
        # ONE REAL GRANULE, opened and checked against the declared layout.
        if days:
            day = days[0]
            e = by_day[day]
            try:
                path, tmp = self._get(ctx, day, e)
            except (f10b._NotFound, IOError) as x:
                sys.exit(f"REFUSING sif: {e['name']} is listed by the STAC "
                         f"and cannot be read ({x})")
            try:
                c = {}
                _r, meta = read_day(path, c)
            except FormatError as x:
                sys.exit(f"REFUSING sif: {e['name']}: {x}")
            finally:
                if tmp and os.path.exists(path):
                    os.remove(path)
            out["first_file"] = {"day": str(day), "id": e["id"],
                                 "bytes": e["bytes"], **meta}
        return out

    def fetch_year(self, ctx, year):
        d_lo = max(dt.date(int(year), 1, 1), ctx.d_lo)
        d_hi = min(dt.date(int(year), 12, 31), ctx.d_hi)
        lo = max(f10b.seconds_since_epoch(dt.date(int(year), 1, 1)), ctx.t_lo)
        hi = min(f10b.seconds_since_epoch(dt.date(int(year), 12, 31)) + 86399,
                 ctx.t_hi)
        yield from self._rows(ctx, str(year), d_lo, d_hi, lo, hi)

    def fetch_month(self, ctx, year, month):
        ndays = calendar.monthrange(int(year), int(month))[1]
        d_lo = max(dt.date(int(year), int(month), 1), ctx.d_lo)
        d_hi = min(dt.date(int(year), int(month), ndays), ctx.d_hi)
        lo, hi = f10b.month_bounds_s(year, month)
        yield from self._rows(ctx, f"{year}-{int(month):02d}", d_lo, d_hi,
                              max(lo, ctx.t_lo), min(hi, ctx.t_hi))

    def platforms(self, ctx):
        """Every instrument this store can hold, named with its footprint."""
        out = {}
        for name, meta in PLATFORMS.items():
            d = dict(meta)
            d["id"] = name
            fp = float(np.sqrt(meta["footprint_km"][0]
                               * meta["footprint_km"][1]))
            d["footprint_geometric_mean_km"] = round(fp, 4)
            d["log2_fp"] = round(float(np.log2(fp / REFERENCE_KM)), 4)
            d["enabled"] = (name == "TROPOMI") or self.oco
            out[int(f10b.platform_hash(name))] = d
        return out

    # --------------------------------------------------------------- smoke --
    def smoke_sources(self, root, d_lo, d_hi, seed=20260920):
        truth = make_smoke_sources(root, d_lo, d_hi, seed)
        self.__init__()
        return truth


def store_projection(counts, per_day, C, last_day):
    """The whole record, from the days the probe read. All inputs measured.

    The ledger's own estimate hangs on a CLOUD CUT ("60-160 GB for TROPOMI;
    the cloud cut decides and the probe measures it"), so the projection is
    given at every cut the per-day report carries as well as at none. The
    record's first day is the one the S5P-PAL STAC's own first L2B item
    begins on, measured; the last is the probe window's, so the record figure
    is a LOWER bound and says so.
    """
    n = len(per_day)
    rows = int(counts.get("rows_kept") or 0)
    if not n or not rows:
        return None
    row = f10b.row_bytes(C, "int32")
    per = rows / float(n)
    days = (last_day - RECORD_FIRST_DAY).days + 1
    cuts = {}
    for key in (per_day[0].get("cloud_fraction_cut") or {}):
        share = sum(p["cloud_fraction_cut"][key] for p in per_day) / n
        cuts[key] = {"share_kept": round(share, 6),
                     "record_bytes": int(round(per * days * row * share))}
    return {
        "days_measured": n,
        "rows_measured": rows,
        "rows_per_day": round(per, 1),
        "stored_bytes_per_row": row,
        "bytes_per_year": int(round(per * 365.25 * row)),
        "rows_per_year": int(round(per * 365.25)),
        "record_first_day": str(RECORD_FIRST_DAY),
        "record_days_to_probe_window": int(days),
        "record_bytes_no_cloud_cut": int(round(per * days * row)),
        "record_bytes_by_cloud_cut": cuts,
        "basis": ("the probe's own days, averaged and carried over the "
                  "record from the S5P-PAL STAC's first L2B item to the end "
                  "of the probe's window — a LOWER bound on the record, "
                  "because the product is still being produced. The "
                  "soundings-a-day count is a July figure and the boreal "
                  "summer is the densest part of the year"),
    }


NOTE_ESTIMATE = {
    "bytes": 160e9,
    "what": ("family1tf.tex ledger: '60-160 GB for TROPOMI; the cloud cut "
             "decides and the probe measures it', over an estimated 5e8 "
             "clear soundings a year"),
}


# ================================================================== smoke ==
SMOKE_DAYS = 3
SMOKE_N = 24                      # soundings a day in the synthetic granule
SMOKE_THRESHOLDS = {"Cloud fraction threshold": "0.8",
                    "SZA threshold": "70.0",
                    "VZA threshold": "60.0",
                    "Quality level threshold": "80",
                    "SIF reference wavelength (nm)": "740.0"}


def write_granule(path, day, n=SMOKE_N, seed=0, thresholds=None):
    """One synthetic L2B granule in the product's real group layout.

    Hostile on purpose: an undeclared netCDF4 fill in SIF_743 (the trap the
    real SIF_735 sets), a sounding with no position, an out-of-bounds
    fluorescence, one sounding in each cloud quartile plus one past the
    producer's own cloud threshold and one past the solar-zenith threshold,
    and a footprint that straddles the dateline.
    """
    import netCDF4
    rng = np.random.default_rng(seed)
    ds = netCDF4.Dataset(path, "w", format="NETCDF4")
    ds.createDimension("time", 1)
    ds.createDimension("n_elem", n)
    ds.createDimension("num_bd_rfl", 7)
    ds.createDimension("ncorner", 4)
    ds.title = "TROPOSIF_L2B"
    ds.processor_version = "01.00.01"
    meta = ds.createGroup("METADATA")
    alg = meta.createGroup("ALGORITHM_SETTINGS")
    for k, v in (thresholds or SMOKE_THRESHOLDS).items():
        setattr(alg, k, v)
    P = ds.createGroup("PRODUCT")
    sup = P.createGroup("SUPPORT_DATA")
    G = sup.createGroup("GEOLOCATIONS")
    I = sup.createGroup("INPUT_DATA")
    v = P.createVariable("time", "f4", ("time",))
    v[:] = np.float32((day - dt.date(2010, 1, 1)).days * 86400.0)
    v.units = "seconds since 2010-01-01 00:00:00"
    # 00:20:00 + 1200 s a sounding, so a day's soundings sit in three bins
    dtm = 1200.0 * 1000.0 * np.arange(n) + 1200000.0
    v = P.createVariable("delta_time", "i4", ("n_elem",))
    v[:] = dtm.astype(np.int32)
    v.units = f"milliseconds since {day.isoformat()}T00:00:00 UTC"
    lat = np.round(np.linspace(-40.0, 60.0, n), 4)
    lon = np.round(np.linspace(-170.0, 179.5, n), 4)
    lon[-1] = 179.99                 # its footprint straddles the dateline
    sif = np.round(rng.uniform(-0.5, 3.0, n), 4)
    corr = np.round(sif * 0.4, 4)
    err = np.round(rng.uniform(0.19, 0.45, n), 4)
    cf = np.linspace(0.0, 0.79, n)
    sza = np.round(rng.uniform(10.0, 60.0, n), 4)
    vza = np.round(rng.uniform(1.0, 50.0, n), 4)
    # one sounding per cloud quartile, and one past the producer's own cut
    cf[0], cf[1], cf[2], cf[3] = 0.05, 0.3, 0.7, 0.9
    sza[4] = 75.0                            # past the SZA threshold
    sif[5] = 40.0                                        # out of bounds
    # the netCDF4 default float fill, written WITHOUT a `_FillValue` — the
    # trap the real `SIF_735` sets on 9,504 soundings a day
    sif[6] = 9.969209968386869e+36
    lat[7] = 9.969209968386869e+36                       # no position
    for name, a, unit in (("SIF_743", sif, "mW/m2/sr/nm"),
                          ("SIF_Corr_743", corr, "mW/m2/sr/nm"),
                          ("SIF_ERROR_743", err, "mW/m2/sr/nm"),
                          ("SIF_735", sif * 1.02, "mW/m2/sr/nm"),
                          ("SIF_Corr_735", corr * 1.02, "mW/m2/sr/nm"),
                          ("SIF_ERROR_735", err * 1.02, "mW/m2/sr/nm"),
                          ("latitude", lat, "degrees_north"),
                          ("longitude", lon, "degrees_east")):
        vv = P.createVariable(name, "f4", ("n_elem",))
        vv[:] = np.asarray(a, np.float32)
        vv.units = unit
    for name, a in (("solar_zenith_angle", sza),
                    ("viewing_zenith_angle", vza)):
        vv = G.createVariable(name, "f4", ("n_elem",))
        vv[:] = np.asarray(a, np.float32)
    # four corners, ~5 km across, the last sounding straddling the dateline
    dlat = np.tile(np.array([-0.0225, -0.0225, 0.0225, 0.0225]), (n, 1))
    dlon = np.tile(np.array([-0.0225, 0.0225, 0.0225, -0.0225]), (n, 1))
    lb = G.createVariable("latitude_bounds", "f4", ("n_elem", "ncorner"))
    lb[:] = np.asarray(lat[:, None] + dlat, np.float32)
    lb.units = "degrees_north"
    ob = G.createVariable("longitude_bounds", "f4", ("n_elem", "ncorner"))
    wrapped = (lon[:, None] + dlon + 180.0) % 360.0 - 180.0
    ob[:] = np.asarray(wrapped, np.float32)
    ob.units = "degrees_east"
    vv = I.createVariable("cloud_fraction_L2", "f4", ("n_elem",))
    vv[:] = np.asarray(cf, np.float32)
    vv = I.createVariable("LC_MASK", "u1", ("n_elem",))
    vv[:] = np.asarray(rng.integers(1, 17, n), np.uint8)
    ds.close()


def make_smoke_sources(root, d_lo, d_hi, seed=20260920):
    """`<root>/sif/items.json` (the STAC answer) plus one granule a day.

    Returns the truth rows in the order the adapter yields them: day
    ascending, then the granule's own sounding order.
    """
    os.environ.pop("SIF_MAX_DAYS", None)
    os.environ.pop("SIF_OCO", None)
    base = os.path.join(root, "sif")
    os.makedirs(base, exist_ok=True)
    plat = int(f10b.platform_hash("TROPOMI"))
    feats, truth = [], []
    day = d_lo
    while day <= d_hi:
        name = (f"S5P_PAL__L2B_SIF____{day.strftime('%Y%m%dT%H%M%S')}_"
                f"{(day + dt.timedelta(days=1)).strftime('%Y%m%dT%H%M%S')}_"
                f"20260920T000000.nc")
        p = os.path.join(base, name)
        write_granule(p, day, seed=seed + day.toordinal())
        feats.append({
            "id": name[:-3], "type": "Feature",
            "properties": {"start_datetime": day.isoformat()
                           + "T00:20:00+00:00",
                           "end_datetime": (day + dt.timedelta(days=1))
                           .isoformat() + "T00:05:00+00:00",
                           "product_type": COLLECTION,
                           "s5p:processor_version": 10001},
            "assets": {"product": {"href": "https://smoke/" + name,
                                   "file:local_path": name,
                                   "file:size": os.path.getsize(p)}}})
        truth += _granule_truth(p, day, plat)
        day += dt.timedelta(days=1)
    with open(os.path.join(base, "items.json"), "w") as fh:
        json.dump({"type": "FeatureCollection", "features": feats,
                   "context": {"limit": 500, "matched": len(feats),
                               "returned": len(feats)}}, fh)
    return truth


def _granule_truth(path, day, plat):
    """What the adapter must keep from one synthetic granule."""
    import netCDF4
    ds = netCDF4.Dataset(path)
    P, G, I = ds[PRODUCT], ds[GEO], ds[INPUT]
    units = str(P.variables["delta_time"].units)
    t = f10b._cf_time_to_seconds(
        np.asarray(P.variables["delta_time"][:], np.float64), units)
    lat = _var(P, "latitude")
    lon = _var(P, "longitude")
    cols = [_var(P, f"SIF_{WINDOW}"), _var(P, f"SIF_Corr_{WINDOW}"),
            _var(P, f"SIF_ERROR_{WINDOW}"), _var(I, "cloud_fraction_L2"),
            _var(G, "solar_zenith_angle"), _var(G, "viewing_zenith_angle")]
    thr = thresholds_of(ds)
    ds.close()
    lon = np.where(lon >= 180.0, lon - 360.0, lon)
    out = []
    T = thr["cloud"]
    for i in range(lat.size):
        if not (np.isfinite(t[i]) and np.isfinite(lat[i])
                and np.isfinite(lon[i]) and np.isfinite(cols[0][i])):
            continue
        v = []
        for (nm, _u, lo, hi), a in zip(CHANNELS, cols):
            x = a[i]
            v.append(np.nan if (not np.isfinite(x) or not (lo <= x <= hi))
                     else float(x))
        cf, sza, vza = cols[3][i], cols[4][i], cols[5][i]
        if not np.isfinite(cf):
            q = 2
        elif cf <= T / 4.0:
            q = 0
        elif cf <= T / 2.0:
            q = 1
        elif cf <= T:
            q = 2
        else:
            q = 3
        if (np.isfinite(sza) and sza > thr["sza"]) or \
                (np.isfinite(vza) and vza > thr["vza"]):
            q = 3
        out.append({"t": int(round(float(t[i]))), "lat": float(lat[i]),
                    "lon": float(lon[i]), "platform": plat, "v": v, "qc": q})
    return out


ADAPTER = SIFAdapter

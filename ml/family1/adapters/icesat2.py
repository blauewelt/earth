"""ICESat-2 ATL08 — terrain and canopy height every 100 metres along six
laser tracks, one row per land segment (family 1.0.tf, E-082 wave 6).

PLAIN ENGLISH. ICESat-2 carries ATLAS, a photon-counting laser that splits
into six beams and counts individual returned photons rather than timing a
pulse. ATL08 is the land-and-vegetation product built from those photons: for
each 100-metre stretch of ground under each beam it reports where the ground
is, how tall the vegetation on it is, how rough the canopy is, and how many
photons went into each answer. Where GEDI stops at 51.6 degrees of latitude
because it flew on the Space Station, ICESat-2 reaches 88 degrees — so this
store is what carries canopy and terrain structure across the boreal forest,
Alaska, Siberia and the ice sheets.

THE ROW is one 100-metre land segment: six channels, the segment's own
position and time, and the ground track it came from as the platform.

SOURCE, VERIFIED 2026-09-20 FROM THIS SANDBOX (NASA's Common Metadata
Repository — CMR — is anonymous; the granules are Earthdata-Login protected,
so the bytes are a hosted runner's job):

  ATL08 release 007, C3565574177-NSIDC_CPRD, 2018-10-14 -> ongoing.
  CMR lists exactly ONE ATL08 collection, so release 007 is the whole
  product as far as the archive is concerned.
  Each granule carries a DIRECT protected HTTPS url:
    https://data.nsidc.earthdatacloud.nasa.gov/nsidc-cumulus-prod-protected/
      ATLAS/ATL08/007/<YYYY>/<MM>/<DD>/ATL08_<YYYYMMDDhhmmss>_<RGTccrr>_007_
      <rev>.h5
  and the same object as `s3://nsidc-cumulus-prod-protected/ATLAS/ATL08/...`,
  which is readable only from inside NASA's own AWS region and is therefore
  not the route a GitHub runner takes.

ONE MONTH MEASURED IN FULL (2022-06, every granule CMR lists, sizes summed
from the listing's own `granule_size`): **4,641 granules, 385,160 MB =
376.1 GB, mean 83.0 MB a granule, smallest 16 MB, largest 344 MB.** At twelve
such months a year of ATL08 is about 4.5 TB native.

THE FETCH, DECIDED FROM THE MEASUREMENT. 376 GB a month does not fit a hosted
lane's disk, but granules are read and discarded one at a time, so whole-file
downloading is a question of TIME rather than of space: at 40 MB/s a month is
2.6 hours and a year is 31, i.e. twelve monthly lanes inside the six-hour
limit with no margin for a slow host. The six channels are a few tens of
bytes a segment out of a granule dominated by the per-photon
`signal_photons` group, which this store never reads — so the same HDF5
byte-range subset `gedi` uses applies here and should cost a small fraction
of that. Which fraction is the PROBE's measurement, not an assumption:
`storage_and_fetch` reports bytes read against the granules' own sizes.

  OPeNDAP: NOT associated with this collection. Measured on CMR 2026-09-20,
  ATL08 007's service associations are the Harmony Trajectory Subsetter
  (S2836723123), NSIDC's own Harmony trajectory subsetter
  (S3213793274-NSIDC_CPRD), an HTTPS file-system service and an S3 service.
  Each granule's CMR links DO carry an
  `opendap.earthdata.nasa.gov/collections/C3565574177-NSIDC_CPRD/granules/
  <name>.h5` url, but it is published under the metadata relation, it is not
  backed by a service association, and an anonymous GET of its `.dmr` answers
  HTTP 401 after two redirects to Earthdata Login — so nothing about it can
  be confirmed from here. The Harmony subsetter remains the fallback if the
  probe says the range route is too slow; it is an asynchronous job service
  with its own queue and is deliberately not built before it is needed.

THE SIZE GATE. Phase A builds ONE YEAR, 2022, and the probe's numbers decide
the rest (`ml/plans/E082_biosphere_wave.md` §2.1). Every stage past `probe`
refuses a window that crosses a calendar-year boundary unless
`ICESAT2_ALLOW_BUILD=1` — the same shape as `swot`'s and `gedi`'s refusal, and
for the same reason (ml/CLAUDE.md §0.3).

THE CHANNELS (C = 6, so a row is 27 + 2C = 39 bytes, the ledger's 39 B).
  h_te_best_fit      metres, the best-fit terrain elevation at the segment's
                     mid-point.
  h_canopy           metres, the 98th-percentile relative canopy height.
  canopy_openness    metres, the standard deviation of the canopy photon
                     heights — how rough or open the canopy is.
  n_ca_photons       how many photons were classified as canopy.
  n_te_photons       how many were classified as terrain.
  segment_landcover  the Copernicus 100 m land-cover class of the segment,
                     as a code (its table is in the counts and in `notes`).

WHERE THE NAMES COME FROM. CMR publishes NO variable metadata for this
collection (0 variable associations, measured 2026-09-20 — unlike the three
GEDI collections, which publish thousands), so the group and dataset names
are read from the product's own data dictionary, fetched anonymously the same
day: `nsidc.org/sites/default/files/documents/technical-reference/
icesat2_atl08_data_dict_v007.pdf`. It gives the layout as

  /gtx/land_segments/            delta_time, latitude, longitude,
                                 segment_landcover, segment_watermask,
                                 msw_flag, terrain_flg, night_flag, ...
  /gtx/land_segments/canopy/     h_canopy, canopy_openness, n_ca_photons,
                                 h_canopy_uncertainty
  /gtx/land_segments/terrain/    h_te_best_fit, n_te_photons
  /orbit_info/                   sc_orient

with the six ground tracks numbered "from the left to the right in the
direction of spacecraft travel as: 1L, 1R in the left-most pair of beams; 2L,
2R for the center pair; and 3L, 3R for the right-most pair". Every dataset is
still RESOLVED against a candidate list at read time and a file that matches
none of them is a refusal carrying the group's own contents — a dictionary is
a document, and the file is the measurement.

LAND ONLY, AND WHAT IS DROPPED. ATL08 runs its land algorithm over whatever
the beam crossed, so a granule also holds segments over inland water, over
the open sea and over sea ice. This store keeps LAND, and the filter is the
product's own two fields rather than a coastline of ours:

  `segment_watermask` == 1  -> dropped (`segments_inland_water`). The
      dictionary: "Water mask (i.e. flag) indicating inland water as
      referenced from the Global Raster Water Mask (ANC33) at 250 m".
      0 = no_water, 1 = water, 255 = Undetermined; 255 is KEPT and counted.
  `segment_landcover` in the classes the variable's own flag table names as
      water or sea -> dropped, counted by class
      (`segments_dropped_by_landcover`). In release 007 those are 80
      (Permanent_water_bodies) and 200 (Open_sea). The codes are read from
      the dataset's `flag_values`/`flag_meanings` attributes when the file
      carries them and from the data dictionary's table when it does not,
      and which of the two was used is recorded
      (`landcover_table_source`).
  Class 70 (Snow_and_ice) is KEPT, deliberately: an ice sheet is land, its
      terrain height is a real measurement this model wants, and dropping it
      would silently remove Greenland and Antarctica. The whole
      `segment_landcover` histogram is in the counts, so that choice can be
      revisited from the probe rather than from an argument.
  0 (No_data) and 255 (Undetermined) are not classes: they become NaN in the
      land-cover channel and are counted (`landcover_no_data`), and the
      segment is kept, because a segment whose land cover is unknown is still
      a terrain and canopy measurement.

WHAT A ROW IS.
  time_s    `land_segments/delta_time`, seconds since 2018-01-01 (the
            dataset's own `units` attribute when it names an epoch, and the
            documented epoch counted as an assumption otherwise), plus the
            exact integer seconds from 1982-01-01 to 2018-01-01.
  lat, lon  `land_segments/latitude`, `land_segments/longitude` — the
            centre-most signal photon of the segment.
  platform  `platform_hash(<ground track>)`, one of gt1l..gt3r;
            `platforms.json` carries the beam's strength and spot number READ
            FROM THE TRACK GROUP'S OWN attributes, and says so when the file
            does not carry them rather than deriving them from the
            spacecraft orientation, which is a rule and not a measurement.
  qc        the WORST of three of the product's own conditions, so a larger
            number is never better: 2 when `terrain_flg` is non-zero (the
            terrain fit deviated past the product's own check), 1 when
            `msw_flag` is non-zero (the multiple-scattering warning: cloud,
            aerosol or blowing snow) or when `h_canopy_uncertainty` is
            missing, 0 otherwise. Each condition is counted separately, and
            the flag histograms are in the counts.
  values    NaN where the product's fill sits, counted; NaN and counted where
            a value falls outside its channel's bounds, never clipped.

THE BOUNDS ARE FLOAT16 BOUNDS TOO. `values` is stored as float16, whose
largest finite number is 65,504, so a photon count above that could not be
stored at all — it would become an infinity that reads as "never measured".
The two photon-count channels are therefore bounded at 60,000: a segment
past it becomes NaN and is COUNTED, which is visible, rather than silently
infinite, which is not.
"""
import datetime as dt
import json
import os
import re
import sys
import time
import urllib.parse
import urllib.request

import numpy as np

import build_family10_stores as f10b
from family1.adapters import _common as cm
from family1.adapters import _h5range as hr

CMR = "https://cmr.earthdata.nasa.gov/search/granules.json"
COLLECTION = "C3565574177-NSIDC_CPRD"
SHORT_NAME = "ATL08"
VERSION = "007"
# ATL08_<YYYYMMDDhhmmss>_<RGT><cycle><region>_<release>_<revision>
NAME = re.compile(r"ATL08_(\d{14})_(\d{4})(\d{2})(\d{2})_(\d{3})_(\d{2})")

# The six ground tracks, from the product's own data dictionary (section
# "Group: /gtx"), read 2026-09-20.
TRACKS = ("gt1l", "gt1r", "gt2l", "gt2r", "gt3l", "gt3r")

LAND = "land_segments"
CANOPY = "land_segments/canopy"
TERRAIN = "land_segments/terrain"

CHANNELS = (
    ("h_te_best_fit", "m", -1000.0, 9000.0),
    ("h_canopy", "m", 0.0, 200.0),
    ("canopy_openness", "m", 0.0, 200.0),
    # 60,000 is a FLOAT16 bound, not a physical one — see the docstring's
    # last paragraph. A count past it is NaN and counted, never an infinity.
    ("n_ca_photons", "1", 0.0, 60000.0),
    ("n_te_photons", "1", 0.0, 60000.0),
    ("segment_landcover", "class", 1.0, 254.0),
)

# Every quantity as a CANDIDATE LIST, the product's current name first. The
# path is relative to the ground-track group.
NEEDED = {
    "delta_time": (f"{LAND}/delta_time",),
    "lat": (f"{LAND}/latitude",),
    "lon": (f"{LAND}/longitude",),
    "h_te_best_fit": (f"{TERRAIN}/h_te_best_fit",),
    "h_canopy": (f"{CANOPY}/h_canopy",),
    "canopy_openness": (f"{CANOPY}/canopy_openness",),
    "n_ca_photons": (f"{CANOPY}/n_ca_photons",),
    "n_te_photons": (f"{TERRAIN}/n_te_photons",),
    "segment_landcover": (f"{LAND}/segment_landcover",),
    "segment_watermask": (f"{LAND}/segment_watermask",),
    "h_canopy_uncertainty": (f"{CANOPY}/h_canopy_uncertainty",),
    "msw_flag": (f"{LAND}/msw_flag",),
    "terrain_flg": (f"{LAND}/terrain_flg", f"{LAND}/terrain_flag"),
}
CHANNEL_KEYS = ("h_te_best_fit", "h_canopy", "canopy_openness",
                "n_ca_photons", "n_te_photons", "segment_landcover")

# ATLAS's own no-data constants, from the data dictionary's FillValue column:
# INVALID_R4B for floats and INVALID_I4B for the photon counts.
INVALID_R4B = float(np.finfo(np.float32).max)     # 3.4028234663852886e38
INVALID_I4B = 2147483647
# `segment_landcover`, release 007, as the data dictionary's own flag table
# gives it. Used ONLY when the file does not carry the table itself.
LANDCOVER_TABLE = {
    0: "No_data", 20: "Shrubs", 30: "Herbaceous", 40: "Cultivated", 50: "Urban",
    60: "Bare_sparse_vegetation", 70: "Snow_and_ice",
    80: "Permanent_water_bodies", 90: "Herbaceous_wetland",
    100: "Moss_and_lichen", 111: "Closed_forest_evergreen_needle_leaf",
    112: "Closed_forest_evergreen_broad_leaf",
    113: "Closed_forest_deciduous_needle_leaf",
    114: "Closed_forest_deciduous_broad_leaf", 115: "Closed_forest_mixed",
    116: "Closed_forest_unknown", 121: "Open_forest_evergreen_needle_leaf",
    122: "Open_forest_evergreen_broad_leaf",
    123: "Open_forest_deciduous_needle_leaf",
    124: "Open_forest_deciduous_broad_leaf", 125: "Open_forest_mixed",
    126: "Open_forest_unknown", 200: "Open_sea", 255: "Undetermined",
}
NO_DATA_WORDS = re.compile(r"no[_ ]?data|undetermined", re.I)
WATER_WORDS = re.compile(r"water|sea", re.I)
NOT_WATER_WORDS = re.compile(r"wetl", re.I)   # "wetland" is land, not water

EPOCH_2018 = int((dt.date(2018, 1, 1) - dt.date(1982, 1, 1)).days) * 86400
DELTA_TIME_UNITS = "seconds since 2018-01-01"
FIRST_YEAR = 2018
# 2022-06, measured on CMR 2026-09-20
GRANULES_MONTH = 4641
BYTES_MONTH = 385_160 * 1_000_000


class FormatError(ValueError):
    """A granule listing or an HDF5 file that is not the product."""


# ================================================================ the index ==
def cmr_granules(params, count=None, attempts=4, page=500, cap=200000):
    """Every granule matching `params`, paged with `CMR-Search-After`.

    ANONYMOUS — CMR needs no login (the BYTES do). The page size is 500 rather
    than 2000 because an ATL08 entry carries thirty-odd browse links and a
    2000-entry page was measured cutting off mid-body (`IncompleteRead`) on
    2026-09-20. An empty answer comes back as an empty list and the CALLER
    decides whether that is a refusal (ml/CLAUDE.md, the 2026-09-14 rule).
    """
    out, after, err = [], None, None
    p = dict(params)
    p["page_size"] = page
    url = f"{CMR}?{urllib.parse.urlencode(p)}"
    while True:
        js = None
        for i in range(max(1, attempts)):
            try:
                req = urllib.request.Request(
                    url, headers={**f10b.UA,
                                  **({"CMR-Search-After": after} if after
                                     else {})})
                with urllib.request.urlopen(
                        req, timeout=f10b.SOCKET_TIMEOUT) as r:
                    raw = r.read()
                    nxt = r.headers.get("CMR-Search-After")
                if count is not None:
                    count(len(raw))
                f10b.count_bytes(len(raw))
                js, after = json.loads(raw), nxt
                break
            except (IOError, *cm.RETRY_ERRORS) as e:            # noqa: B014
                err = e
                if i < attempts - 1:
                    time.sleep(3.0 * (2 ** i))
        if js is None:
            raise FormatError(f"CMR: {type(err).__name__}: {err} for {url}")
        ents = js.get("feed", {}).get("entry", [])
        out += ents
        if not ents or not after or len(out) >= cap:
            return out


def parse_entries(entries):
    """CMR entries -> {granule key: {name, url, bytes, t0, t1}}.

    The key is (acquisition stamp, reference ground track, cycle, region),
    which is the granule's own identity; two granules claiming it, or a name
    this parser does not understand, is a refusal.
    """
    out, other = {}, []
    for g in entries:
        title = str(g.get("title", ""))
        m = NAME.search(title)
        if not m:
            other.append(title)
            continue
        key = m.group(1, 2, 3, 4)
        if key in out:
            raise FormatError(f"two granules for {key}: {out[key]['name']} "
                              f"and {title}")
        url = None
        for ln in g.get("links", []):
            h = str(ln.get("href", ""))
            if h.startswith("https") and "protected" in h and h.endswith(".h5"):
                url = h
                break
        if url is None:
            raise FormatError(
                f"{title}: no protected https .h5 link "
                f"({[l.get('href') for l in g.get('links', [])][:3]})")
        out[key] = {"name": title, "url": url,
                    "bytes": int(float(g.get("granule_size") or 0) * 1e6),
                    "t0": g.get("time_start"), "t1": g.get("time_end"),
                    "release": m.group(5), "revision": m.group(6)}
    if other:
        raise FormatError(
            f"{len(other)} listed granule name(s) are not ATL08 granules, "
            f"e.g. {sorted(other)[:3]} — refusing to read a listing this "
            f"parser does not understand")
    return out


# ================================================================== reading ==
def resolve_path(group, candidates, what, where):
    """The first candidate path present under `group`, or a refusal."""
    for c in candidates:
        if c in group:
            return c
    have = []
    try:
        for k in group:
            have.append(k)
            sub = group[k]
            if hasattr(sub, "keys"):
                have += [f"{k}/{kk}" for kk in sub]
    except (TypeError, AttributeError):                         # noqa: BLE001
        pass
    raise FormatError(
        f"{where}: none of {list(candidates)} is a dataset of this group, so "
        f"'{what}' cannot be read. The group holds {sorted(have)[:40]}")


def _to_float(ds, a, counts, key):
    """Raw values -> float64 with ATLAS's fill turned into NaN, counted."""
    v = np.asarray(a, np.float64)
    fills = []
    for attr in ("_FillValue", "fill_value", "missing_value"):
        val = ds.attrs.get(attr) if hasattr(ds, "attrs") else None
        if val is None:
            continue
        for x in np.atleast_1d(np.asarray(val)).ravel():
            try:
                fills.append(float(x))
            except (TypeError, ValueError):
                pass
    n = 0
    for f in set(fills):
        hit = v == f
        n += int(hit.sum())
        v[hit] = np.nan
    # the product's documented constants, applied as well: several ATL08
    # datasets carry the sentinel without declaring it
    with np.errstate(invalid="ignore"):
        hit = (np.abs(v) >= INVALID_R4B) | (v == INVALID_I4B)
    n += int(hit.sum())
    v[hit] = np.nan
    v[~np.isfinite(v)] = np.nan
    if n:
        d = counts.setdefault("fill_values", {})
        d[key] = d.get(key, 0) + n
    return v


def delta_time_seconds(ds, counts):
    """`delta_time` -> seconds since 1982-01-01, float64.

    ATL08 writes "seconds since 2018-01-01" in the dataset's own `units`; when
    it does not, the documented ATLAS Standard Data Product epoch is used and
    the assumption is COUNTED rather than passed off as a reading.
    """
    u = ds.attrs.get("units")
    if isinstance(u, bytes):
        u = u.decode("utf-8", "replace")
    a = np.asarray(ds[:], np.float64)
    if u and " since " in str(u).lower():
        return f10b._cf_time_to_seconds(a, str(u))
    counts["delta_time_epoch_assumed"] = \
        counts.get("delta_time_epoch_assumed", 0) + 1
    return a + EPOCH_2018


def landcover_classes(ds, counts):
    """(water codes, no-data codes, where the table came from).

    Preferred: the dataset's OWN `flag_values` / `flag_meanings` attributes,
    so a release that renumbers the classes is followed rather than
    contradicted. Fallback: the release-007 data dictionary's table, and the
    fallback is recorded in the counts.
    """
    vals = ds.attrs.get("flag_values") if hasattr(ds, "attrs") else None
    means = ds.attrs.get("flag_meanings") if hasattr(ds, "attrs") else None
    if isinstance(means, bytes):
        means = means.decode("utf-8", "replace")
    table, source = None, "the ATL08 release-007 data dictionary"
    if vals is not None and means:
        words = str(means).split()
        codes = [int(x) for x in np.atleast_1d(np.asarray(vals)).ravel()]
        if len(words) == len(codes):
            table = dict(zip(codes, words))
            source = "the segment_landcover dataset's own flag table"
    if table is None:
        table = dict(LANDCOVER_TABLE)
    water = {c for c, w in table.items()
             if WATER_WORDS.search(w) and not NOT_WATER_WORDS.search(w)}
    nodata = {c for c, w in table.items() if NO_DATA_WORDS.search(w)}
    counts["landcover_table_source"] = source
    return water, nodata, table


def read_track(h5, track, counts):
    """One ground track -> the columns the store needs, unfiltered."""
    g = h5.get(track)
    if g is None:
        return None
    where = f"{track}"
    name = {k: resolve_path(g, v, k, where) for k, v in NEEDED.items()}
    out = {"resolved": {k: name[k] for k in sorted(name)}}
    out["t"] = delta_time_seconds(g[name["delta_time"]], counts)
    out["lat"] = np.asarray(g[name["lat"]][:], np.float64)
    out["lon"] = np.asarray(g[name["lon"]][:], np.float64)
    for k in CHANNEL_KEYS:
        ds = g[name[k]]
        out[k] = _to_float(ds, ds[:], counts, k)
    lcds = g[name["segment_landcover"]]
    out["landcover_raw"] = np.asarray(lcds[:], np.int64)
    out["water"], out["nodata"], out["lc_table"] = \
        landcover_classes(lcds, counts)
    out["watermask"] = np.asarray(g[name["segment_watermask"]][:], np.int64)
    out["msw"] = np.asarray(g[name["msw_flag"]][:], np.int64)
    out["terrain_flg"] = np.asarray(g[name["terrain_flg"]][:], np.int64)
    ds = g[name["h_canopy_uncertainty"]]
    out["h_canopy_uncertainty"] = _to_float(ds, ds[:], counts,
                                            "h_canopy_uncertainty")
    return out


def track_meta(h5, track):
    """The ground track's own attributes — beam strength and spot number."""
    g = h5.get(track)
    if g is None:
        return {}
    out = {}
    for a in ("atlas_beam_type", "atlas_spot_number", "groundtrack_id",
              "sc_orientation"):
        v = g.attrs.get(a)
        if isinstance(v, bytes):
            v = v.decode("utf-8", "replace")
        if v is not None:
            out[a] = str(np.atleast_1d(np.asarray(v)).ravel()[0]) \
                if not isinstance(v, str) else v
    orb = h5.get("orbit_info")
    if orb is not None and "sc_orient" in orb:
        u = np.unique(np.asarray(orb["sc_orient"][:], np.int64))
        out["sc_orient"] = ",".join(str(int(x)) for x in u)
    return out


# ================================================================== adapter ==
class ICESat2Adapter(f10b.SourceAdapter):
    store = "icesat2"
    title = ("ICESat-2 ATL08 release 007: terrain height, canopy height, "
             "canopy openness and photon counts per 100 m LAND segment, six "
             "ground tracks to 88 degrees")
    family = "1tf"
    distribution = "public"
    licence = {"name": "NASA open data (EOSDIS, no restriction on use)",
               "redistribution": "yes",
               "redistribution_confirmed": True,
               "derived_works": "free",
               "attribution": ("Neuenschwander, A. et al. ATLAS/ICESat-2 L3A "
                               "Land and Vegetation Height, Version 7. NSIDC "
                               "DAAC, doi:10.5067/ATLAS/ATL08.007")}
    time_dtype = "int32"
    credentials = ("EARTHDATA_USERNAME", "EARTHDATA_PASSWORD")
    channels = CHANNELS
    # 100 m segments: log2(0.1 / 27.83) = -8.12, the ledger's -8.1
    log2_fp = float(np.log2(0.1 / 27.83))
    # an instantaneous pass over the segment: the clamp
    log2_dt = -12.0
    per_year = True
    first_year = FIRST_YEAR
    platform_meta = True
    fetch_month_scope = "month"
    qc_policy = (
        "LAND ONLY. A segment whose `segment_watermask` is 1 (inland water, "
        "from the product's own 250 m Global Raster Water Mask) is dropped "
        "and counted; so is one whose `segment_landcover` class is named "
        "water or sea by the variable's own flag table — 80 "
        "(Permanent_water_bodies) and 200 (Open_sea) in release 007 — with "
        "the drop counted per class and the whole land-cover histogram "
        "recorded. Class 70 (Snow_and_ice) is KEPT: an ice sheet is land and "
        "its terrain height is a measurement. 0 (No_data) and 255 "
        "(Undetermined) become NaN in the land-cover channel and the segment "
        "is kept. `qc` is the WORST of three of the product's own "
        "conditions, so larger is never better: 2 when `terrain_flg` is "
        "non-zero, 1 when `msw_flag` (the multiple-scattering warning: "
        "cloud, aerosol or blowing snow) is non-zero or "
        "`h_canopy_uncertainty` is missing, 0 otherwise — each condition "
        "counted separately. A segment with no finite time or position is "
        "dropped and counted. ATLAS's fill constants become NaN and are "
        "counted per channel. A value outside its channel's bounds becomes "
        "NaN for that channel alone and is counted, never clipped; the two "
        "photon-count channels are bounded at 60,000 because `values` is "
        "stored as float16 and a larger number could only be stored as an "
        "infinity, which would read as 'never measured'.")
    sources = (f"{CMR}?collection_concept_id={COLLECTION} (anonymous)",
               "https://data.nsidc.earthdatacloud.nasa.gov/"
               "nsidc-cumulus-prod-protected/ATLAS/ATL08/007/<YYYY>/<MM>/"
               "<DD>/ATL08_<YYYYMMDDhhmmss>_<RGTccrr>_007_<rev>.h5",
               "https://nsidc.org/sites/default/files/documents/"
               "technical-reference/icesat2_atl08_data_dict_v007.pdf")
    verified = (
        "2026-09-20 from this sandbox, anonymously against NASA CMR and "
        "NSIDC (the granules are Earthdata-protected, so the bytes are a "
        "hosted runner's job). CMR lists exactly ONE ATL08 collection: "
        "release 007, C3565574177-NSIDC_CPRD, 2018-10-14 -> ongoing. 2022-06 "
        "IN FULL: 4,641 granules, 385,160 MB (376.1 GB), mean 83.0 MB, "
        "smallest 16 MB, largest 344 MB, every one with a direct protected "
        "https .h5 url on data.nsidc.earthdatacloud.nasa.gov and the same "
        "object on s3://nsidc-cumulus-prod-protected. Service associations: "
        "the Harmony Trajectory Subsetter, NSIDC's own Harmony trajectory "
        "subsetter, an HTTPS file-system service and an S3 service — NO "
        "OPeNDAP service, and the opendap.earthdata.nasa.gov url in each "
        "granule's links answers HTTP 401 anonymously. CMR publishes no "
        "variable metadata for this collection (0 associations), so the "
        "group and dataset names come from the release-007 data dictionary "
        "PDF fetched the same day, which gives /gtx/land_segments with "
        "canopy/ and terrain/ subgroups and the six tracks gt1l..gt3r. NOT "
        "MEASURED HERE, and what the runner probe is for: whether NSIDC "
        "serves HTTP 206 for a range read, what fraction of a granule the "
        "six datasets cost, segments a granule, how many are dropped as "
        "water, the flag histograms, and therefore rows and bytes a year.")
    notes = ""
    smoke_window = ("2021-12-01", "2022-01-31")
    smoke_probe_month = "2021-12"

    def __init__(self):
        # READ AT CONSTRUCTION TIME, so the fresh adapter `stage_probe` builds
        # for itself sees the same knobs. `family1-build.yml`'s `adapter_env`
        # input is how a dispatch sets them.
        self._session = None
        self._listing = {}
        self._tracks = {}
        self._resolved = {}
        self._lc_table = {}
        self.fetch_mode = (os.environ.get("ICESAT2_FETCH") or "range").strip()
        if self.fetch_mode not in ("range", "whole"):
            sys.exit("ICESAT2_FETCH must be 'range' (HDF5 byte-range subset) "
                     "or 'whole' (download the granule)")
        try:
            self.max_granules = int(
                os.environ.get("ICESAT2_MAX_GRANULES") or 12)
        except ValueError:
            sys.exit("ICESAT2_MAX_GRANULES must be an integer")
        self.allow_build = (os.environ.get("ICESAT2_ALLOW_BUILD") or "") == "1"
        self.notes = (
            f"ATL08 release {VERSION} ({COLLECTION}); fetch mode "
            f"{self.fetch_mode!r}"
            + (f"; at most {self.max_granules} granule(s) per call"
               if self.max_granules else "; every granule in the window")
            + ". LAND SEGMENTS ONLY — inland-water and open-sea segments are "
              "dropped and counted; snow and ice are kept. PHASE A IS ONE "
              "YEAR (2022): a window crossing a calendar-year boundary is "
              "refused past `probe` unless ICESAT2_ALLOW_BUILD=1, because a "
              "month is 376 GB native and the record's cost is the probe's "
              "to measure. A capped call is not a whole month and says so "
              "here.")

    # -------------------------------------------------------------- listing --
    def listing(self, ctx, t_lo, t_hi):
        key = (int(t_lo), int(t_hi))
        if key in self._listing:
            return self._listing[key]
        if ctx.source_dir:
            root = os.path.join(ctx.source_dir, self.store)
            names = sorted(os.listdir(root)) if os.path.isdir(root) else []
            got = {}
            for n in names:
                if not n.endswith(".h5"):
                    continue
                m = NAME.search(n)
                if not m:
                    continue
                p = os.path.join(root, n)
                got[m.group(1, 2, 3, 4)] = {
                    "name": n[:-3], "url": p, "bytes": os.path.getsize(p),
                    "t0": None, "t1": None, "release": m.group(5),
                    "revision": m.group(6)}
            if not got:
                sys.exit(f"REFUSING icesat2: no ATL08 .h5 under {root} — the "
                         f"smoke's synthetic archive is missing")
            ctx.count_bytes(sum(e["bytes"] for e in got.values()))
        else:
            temporal = f"{_iso(t_lo)},{_iso(t_hi)}"
            try:
                ents = cmr_granules(
                    {"collection_concept_id": COLLECTION,
                     "temporal": temporal, "sort_key": "start_date"},
                    count=ctx.count_bytes, attempts=ctx.a.attempts)
            except FormatError as e:
                sys.exit(f"REFUSING icesat2: {e}")
            if not ents:
                sys.exit(
                    f"REFUSING icesat2: CMR lists no {SHORT_NAME} granule for "
                    f"{temporal} — an empty listing is a refusal, not an "
                    f"empty month (ml/CLAUDE.md, the 2026-09-14 rule)")
            try:
                got = parse_entries(ents)
            except FormatError as e:
                sys.exit(f"REFUSING icesat2: {e}")
        self._listing[key] = got
        return got

    # ---------------------------------------------------------------- bytes --
    def _open(self, ctx, entry):
        if ctx.source_dir:
            ctx.count_bytes(os.path.getsize(entry["url"]))
            return hr.open_local(entry["url"])
        if self._session is None:
            self._session = cm.earthdata_session()
        if self.fetch_mode == "whole":
            dest = os.path.join(ctx.scratch, self.store,
                                entry["name"] + ".h5")
            return hr.open_whole(self._session, entry["url"], dest,
                                 attempts=max(1, ctx.a.attempts))
        return hr.open_range(self._session, entry["url"],
                             attempts=max(1, ctx.a.attempts))

    # ----------------------------------------------------------------- rows --
    def _rows(self, ctx, label, t_lo, t_hi):
        got = self.listing(ctx, t_lo, t_hi)
        keys = [k for k in sorted(got) if in_window(got[k], k, t_lo, t_hi)]
        counts = {"granules_listed": len(got), "fetch_mode": self.fetch_mode,
                  "max_granules_cap": self.max_granules,
                  "release": VERSION}
        if self.max_granules and len(keys) > self.max_granules:
            counts["granules_skipped_by_cap"] = len(keys) - self.max_granules
            keys = spread(keys, self.max_granules)
        counts["granules_wanted"] = len(keys)
        per_granule = []
        t0 = time.time()
        for k in keys:
            e = got[k]
            try:
                op = self._open(ctx, e)
            except f10b._NotFound as why:
                ctx.note_absent(f"{label} granule {e['name']}",
                                f"listed in CMR, and {why}")
                continue
            except hr.RangeError as why:
                sys.exit(f"REFUSING icesat2: {why}")
            try:
                for out in self._granule_rows(op, counts, t_lo, t_hi):
                    yield label, out, None
            except FormatError as why:
                sys.exit(f"REFUSING icesat2: {e['name']}: {why}")
            finally:
                hr.finish_range(op)
                per_granule.append({"granule": e["name"], **op.stats()})
                op.close()
        counts["fetch_seconds"] = round(time.time() - t0, 1)
        counts["per_granule"] = per_granule[:20]
        # A LIST OF ONE, not a dict: `f10b._merge_counts` ADDS the values of a
        # dict one level deep, so a dict of dataset NAMES would be
        # concatenated into nonsense the second time counts were merged.
        counts["resolved_datasets"] = [dict(self._resolved)] \
            if self._resolved else []
        counts["landcover_table"] = [dict(self._lc_table)] \
            if self._lc_table else []
        fb = storage_and_fetch(counts, per_granule, self.C)
        counts["storage_and_fetch"] = [fb] if fb else []
        yield label, None, counts

    def _granule_rows(self, op, counts, t_lo, t_hi):
        """One granule, ground track by ground track."""
        h5 = op.h5
        tracks = [t for t in sorted(h5) if t.startswith("gt")]
        if not tracks:
            raise FormatError(f"no gtx group; the file holds "
                              f"{sorted(h5)[:20]}")
        counts["tracks_seen"] = counts.get("tracks_seen", 0) + len(tracks)
        for track in tracks:
            s = read_track(h5, track, counts)
            if s is None:
                continue
            self._resolved.setdefault(track, s["resolved"])
            meta = track_meta(h5, track)
            if meta or track not in self._tracks:
                self._tracks[track] = meta
            n = len(s["t"])
            if n == 0:
                continue
            counts["segments_in_granule"] = \
                counts.get("segments_in_granule", 0) + n
            lc = s["landcover_raw"]
            hist = counts.setdefault("segment_landcover", {})
            for u, c in zip(*np.unique(lc, return_counts=True)):
                hist[str(int(u))] = hist.get(str(int(u)), 0) + int(c)
            self._lc_table.update({str(k): v for k, v in s["lc_table"].items()})
            # ---- LAND ONLY -------------------------------------------------
            wet = s["watermask"] == 1
            counts["segments_inland_water"] = \
                counts.get("segments_inland_water", 0) + int(wet.sum())
            water = np.isin(lc, sorted(s["water"]))
            drop_lc = counts.setdefault("segments_dropped_by_landcover", {})
            for code in sorted(s["water"]):
                k = int(((lc == code) & ~wet).sum())
                if k:
                    nm = s["lc_table"].get(code, str(code))
                    drop_lc[f"{code}_{nm}"] = drop_lc.get(f"{code}_{nm}", 0) + k
            land = ~(wet | water)
            # ---- the land-cover CHANNEL: no-data is NaN, the segment stays --
            nodata = np.isin(lc, sorted(s["nodata"]))
            counts["landcover_no_data"] = \
                counts.get("landcover_no_data", 0) + int((nodata & land).sum())
            lcv = s["segment_landcover"].copy()
            lcv[nodata] = np.nan
            # ---- the row ---------------------------------------------------
            good = land & np.isfinite(s["t"]) & np.isfinite(s["lat"]) \
                & np.isfinite(s["lon"])
            counts["segments_no_time_or_position"] = \
                counts.get("segments_no_time_or_position", 0) + \
                int((land & ~good).sum())
            inside = good & (s["t"] >= t_lo) & (s["t"] <= t_hi)
            counts["segments_outside_window"] = \
                counts.get("segments_outside_window", 0) + \
                int((good & ~inside).sum())
            if not inside.any():
                continue
            values = np.full((int(inside.sum()), self.C), np.nan, np.float64)
            for i, key in enumerate(CHANNEL_KEYS):
                col = lcv if key == "segment_landcover" else s[key]
                values[:, i] = col[inside]
            # ---- qc: the WORST of three of the product's own conditions ----
            terr = s["terrain_flg"][inside] != 0
            msw = s["msw"][inside] != 0
            unc = ~np.isfinite(s["h_canopy_uncertainty"][inside])
            counts["qc_terrain_flag_nonzero"] = \
                counts.get("qc_terrain_flag_nonzero", 0) + int(terr.sum())
            counts["qc_msw_flag_nonzero"] = \
                counts.get("qc_msw_flag_nonzero", 0) + int(msw.sum())
            counts["qc_canopy_uncertainty_missing"] = \
                counts.get("qc_canopy_uncertainty_missing", 0) + int(unc.sum())
            qc = np.maximum(terr * 2, (msw | unc) * 1).astype(np.uint8)
            hist = counts.setdefault("msw_flag", {})
            for u, c in zip(*np.unique(s["msw"][inside], return_counts=True)):
                hist[str(int(u))] = hist.get(str(int(u)), 0) + int(c)
            oob = self.mask_bounds(values)
            if oob:
                f10b._merge_counts(counts, {"out_of_bounds": oob})
            counts["rows_kept"] = counts.get("rows_kept", 0) + \
                int(inside.sum())
            plat = np.full(int(inside.sum()), f10b.platform_hash(track),
                           np.int64)
            yield self.pack(s["t"][inside], s["lat"][inside],
                            s["lon"][inside], values, plat, qc)

    # ------------------------------------------------------------- contract --
    def fetch_preflight(self, ctx):
        """Both guards where the inputs are all they have cost (§0.3).

        THE SIZE GATE: one month of ATL08 is 376 GB native and phase A is one
        year (2022). THE CREDENTIAL GUARD: the granules are
        Earthdata-protected, and a fetch with no way to authenticate spends
        the whole listing to discover a page of 401s (family1-build run #152).
        """
        if ctx.source_dir:
            return None
        stage = getattr(ctx.a, "stage", "")
        if stage != "probe" and not self.allow_build:
            if ctx.d_lo.year != ctx.d_hi.year:
                sys.exit(
                    f"REFUSING to build icesat2 for {ctx.d_lo} .. "
                    f"{ctx.d_hi}: that crosses a calendar-year boundary and "
                    f"phase A is ONE YEAR (2022), with the rest decided on "
                    f"the probe's measured rows and bytes "
                    f"(ml/plans/E082_biosphere_wave.md §2.1). One month of "
                    f"ATL08 release 007 is 4,641 granules and 376 GB native, "
                    f"measured on CMR 2026-09-20. Probe it with `--stage "
                    f"probe --probe-month 2022-06` (the report lands as "
                    f"ml/family1/probes/icesat2_2022-06.json and its "
                    f"`storage_and_fetch` block is the decision material); "
                    f"set ICESAT2_ALLOW_BUILD=1 once those numbers are "
                    f"recorded. Nothing has been fetched.")
        if not cm.earthdata_ready():
            sys.exit(
                "REFUSING icesat2: the granules are Earthdata-protected and "
                "this process can authenticate to Earthdata Login in NEITHER "
                "way: no EARTHDATA_USERNAME / EARTHDATA_PASSWORD in the "
                "environment and no netrc naming urs.earthdata.nasa.gov. "
                "family1-build.yml gives both on a HOSTED runner from the "
                "repository secrets (ml/CLAUDE.md §6); a box gets neither. "
                "Nothing has been fetched.")
        return None

    def index(self, ctx):
        got = self.listing(ctx, ctx.t_lo, ctx.t_hi)
        keys = sorted(got)
        per_day, rel = {}, {}
        for k in keys:
            e = got[k]
            if e["t0"]:
                per_day[e["t0"][:10]] = per_day.get(e["t0"][:10], 0) + 1
            rel[e["release"]] = rel.get(e["release"], 0) + 1
        if rel and set(rel) != {VERSION}:
            sys.exit(
                f"REFUSING icesat2: the listing mixes ATL08 releases {rel} — "
                f"this adapter reads release {VERSION} and a mixed window "
                f"would join two processings into one store")
        nb = sum(e["bytes"] for e in got.values())
        return {
            "dataset": f"ICESat-2 ATL08 release {VERSION} ({SHORT_NAME})",
            "url": f"{CMR}?collection_concept_id={COLLECTION}",
            "collection": COLLECTION,
            "version": VERSION,
            "releases_seen": rel,
            "fetch_mode": self.fetch_mode,
            "max_granules": self.max_granules,
            "window": [str(ctx.d_lo), str(ctx.d_hi)],
            "granules": len(got),
            "bytes": int(nb),
            "bytes_per_granule_mean": (round(nb / len(got), 1) if got
                                       else None),
            "granules_per_day": per_day,
            "tracks_expected": list(TRACKS),
            "channels": self.channel_names,
            "land_only": True,
            "landcover_table": LANDCOVER_TABLE,
            "record": [got[keys[0]]["t0"], got[keys[-1]]["t1"]]
                      if keys else None,
            "one_year_only": not self.allow_build,
        }

    def fetch_year(self, ctx, year):
        lo = max(f10b.seconds_since_epoch(dt.date(int(year), 1, 1)), ctx.t_lo)
        hi = min(f10b.seconds_since_epoch(dt.date(int(year), 12, 31)) + 86399,
                 ctx.t_hi)
        yield from self._rows(ctx, str(year), lo, hi)

    def fetch_month(self, ctx, year, month):
        lo, hi = f10b.month_bounds_s(year, month)
        yield from self._rows(ctx, f"{year}-{int(month):02d}",
                              max(lo, ctx.t_lo), min(hi, ctx.t_hi))

    # ------------------------------------------------------------ platforms --
    def platforms(self, ctx):
        """The six ground tracks. The BEAM STRENGTH comes from the file."""
        out = {}
        for t in sorted(set(TRACKS) | set(self._tracks)):
            seen = self._tracks.get(t) or {}
            strength = seen.get("atlas_beam_type")
            out[int(f10b.platform_hash(t))] = {
                "id": t,
                "pair": t[2],
                "side": t[3],
                "beam_strength": strength,
                "atlas_spot_number": seen.get("atlas_spot_number"),
                "sc_orient": seen.get("sc_orient"),
                "beam_strength_source": (
                    "the ground track group's own `atlas_beam_type` attribute"
                    if strength else
                    "not available: no granule read in this run carried an "
                    "`atlas_beam_type` attribute on this track. Which of a "
                    "pair is strong depends on the spacecraft orientation "
                    "(`/orbit_info/sc_orient`), and deriving it from that is "
                    "a RULE rather than a reading, so it is left unsaid"),
                "instrument": "ATLAS on ICESat-2",
            }
        return out

    # ---------------------------------------------------------------- smoke --
    def smoke_sources(self, root, d_lo, d_hi, seed=20260920):
        truth = make_smoke_sources(root, d_lo, d_hi, seed)
        self.__init__()
        return truth


def _iso(t_s):
    base = dt.datetime(1982, 1, 1) + dt.timedelta(seconds=int(t_s))
    return base.strftime("%Y-%m-%dT%H:%M:%SZ")


def stamp_seconds(key):
    """The granule key's own YYYYMMDDhhmmss stamp -> seconds since 1982."""
    s = key[0]
    d = dt.date(int(s[:4]), int(s[4:6]), int(s[6:8]))
    return f10b.seconds_since_epoch(d) + int(s[8:10]) * 3600 + \
        int(s[10:12]) * 60 + int(s[12:14])


GRANULE_MAX_SECONDS = 2 * 3600     # far longer than one ATL08 granule


def spread(keys, n):
    """`n` of `keys`, EVENLY SPACED, keeping their order.

    A capped probe that read the first twelve granules of a month would
    measure twelve passes of one morning; spreading them over the window
    costs the same and measures the month.
    """
    keys = list(keys)
    if n <= 0 or len(keys) <= n:
        return keys
    step = len(keys) / float(n)
    return [keys[min(len(keys) - 1, int(i * step))] for i in range(n)]


def in_window(entry, key, t_lo, t_hi):
    """Does this granule overlap [t_lo, t_hi] seconds?

    The listing's own `time_start`/`time_end` when CMR gave them; otherwise
    the name's acquisition stamp, widened by more than any granule's length.
    """
    if entry.get("t0") and entry.get("t1"):
        def s(stamp):
            d = dt.datetime.strptime(stamp[:19], "%Y-%m-%dT%H:%M:%S")
            return f10b.seconds_since_epoch(d.date()) + d.hour * 3600 + \
                d.minute * 60 + d.second
        return s(entry["t0"]) <= t_hi and s(entry["t1"]) >= t_lo
    t = stamp_seconds(key)
    return (t_lo - GRANULE_MAX_SECONDS) <= t <= t_hi


def storage_and_fetch(counts, per_granule, C):
    """What the probe measured, turned into the numbers the decision needs."""
    n = len(per_granule)
    rows = int(counts.get("rows_kept") or 0)
    if not n or not rows:
        return None
    native = sum(int(m["granule_bytes"]) for m in per_granule)
    read = sum(int(m["bytes_read"]) for m in per_granule)
    row_bytes = 27 + 2 * C
    secs = counts.get("fetch_seconds") or 0
    rate = (read / secs) if secs else None
    return {
        "measured": {
            "granules_read": n,
            "rows_kept": rows,
            "rows_per_granule": round(rows / n, 1),
            "native_bytes_of_those_granules": native,
            "bytes_read": read,
            "read_fraction": (round(read / native, 6) if native else None),
            "fetch_seconds": counts.get("fetch_seconds"),
            "bytes_per_second": (round(rate, 1) if rate else None),
        },
        "extrapolation_basis": {
            "granules_per_month_2022_06": GRANULES_MONTH,
            "native_bytes_per_month_2022_06": BYTES_MONTH,
            "note": ("4,641 granules and 385,160 MB in 2022-06, measured "
                     "from CMR's own granule sizes on 2026-09-20"),
        },
        "projected_one_year_2022": {
            "granules": GRANULES_MONTH * 12,
            "rows": round(rows / n * GRANULES_MONTH * 12),
            "store_bytes": round(rows / n * GRANULES_MONTH * 12 * row_bytes),
            "bytes_to_fetch": (round(read / native * BYTES_MONTH * 12)
                               if native else None),
            "fetch_hours": (
                round(read / native * BYTES_MONTH * 12 / rate / 3600, 2)
                if native and rate else None),
        },
        "bytes_per_row_stored": row_bytes,
    }


# ==================================================================== smoke ==
# A synthetic archive in ATL08's own group layout: `/gtx/land_segments` with
# `canopy/` and `terrain/` subgroups and `/orbit_info/sc_orient`, the exact
# dataset names the adapter resolves. Four granules over two months, two
# ground tracks each, eight segments a track — one inland-water segment, one
# open-sea segment and one permanent-water segment to drop, one canopy height
# out of bounds, and one segment for each of the three `qc` conditions.
SMOKE_TRACKS = ("gt1l", "gt2r")
SMOKE_SEGMENTS = 8
SMOKE_GRANULES = (
    # (YYYYMMDDhhmmss, reference ground track, cycle, region)
    ("20211203060000", "1001", "13", "01"),
    ("20211204060000", "1002", "13", "02"),
    ("20220105060000", "1101", "14", "01"),
    ("20220106060000", "1102", "14", "02"),
)
SMOKE_BEAM_TYPE = {"gt1l": "weak", "gt2r": "strong"}
SMOKE_SPOT = {"gt1l": "2", "gt2r": "3"}
# the codes the smoke writes, one per segment index
SMOKE_LANDCOVER = [111, 200, 20, 30, 40, 60, 80, 112]
DROPPED_SEGMENTS = 3          # inland water, open sea, permanent water
KEPT_SEGMENTS = SMOKE_SEGMENTS - DROPPED_SEGMENTS


def _granule_name(g):
    return f"ATL08_{g[0]}_{g[1]}{g[2]}{g[3]}_{VERSION}_01"


def make_smoke_sources(root, d_lo, d_hi, seed=20260920):
    """`<root>/icesat2/<granule>.h5`, and the truth rows in yield order."""
    import h5py
    rng = np.random.default_rng(seed)
    base = os.path.join(root, "icesat2")
    os.makedirs(base, exist_ok=True)
    truth = []
    for gi, g in enumerate(SMOKE_GRANULES):
        stamp = g[0]
        day = dt.date(int(stamp[:4]), int(stamp[4:6]), int(stamp[6:8]))
        t0 = (dt.datetime(day.year, day.month, day.day, 6)
              - dt.datetime(2018, 1, 1)).total_seconds()
        path = os.path.join(base, _granule_name(g) + ".h5")
        with h5py.File(path, "w") as f:
            oi = f.create_group("orbit_info")
            oi.create_dataset("sc_orient", data=np.ones(1, np.int8))
            for ti, track in enumerate(SMOKE_TRACKS):
                n = SMOKE_SEGMENTS
                gt = f.create_group(track)
                gt.attrs["atlas_beam_type"] = SMOKE_BEAM_TYPE[track]
                gt.attrs["atlas_spot_number"] = SMOKE_SPOT[track]
                ls = gt.create_group(LAND)
                cn = gt.create_group(CANOPY)
                tr = gt.create_group(TERRAIN)
                delta = t0 + 600.0 * (ti + 1) + np.arange(n) * 11.0
                lat = np.linspace(-60.0, 80.0, n) + ti
                lon = np.linspace(-120.0, 120.0, n) + gi
                h_te = np.round(rng.uniform(5.0, 900.0, n), 3)
                h_can = np.round(rng.uniform(1.0, 40.0, n), 3)
                openness = np.round(rng.uniform(0.2, 8.0, n), 3)
                n_ca = (rng.integers(5, 900, n)).astype(np.int32)
                n_te = (rng.integers(50, 4000, n)).astype(np.int32)
                lc = np.array(SMOKE_LANDCOVER, np.uint8)
                water = np.zeros(n, np.uint8)
                msw = np.zeros(n, np.int8)
                terr = np.zeros(n, np.uint8)
                unc = np.round(rng.uniform(0.1, 3.0, n), 3)
                # segment 0: inland water -> dropped
                water[0] = 1
                # segment 1: open sea (landcover 200) -> dropped
                # segment 6: permanent water bodies (80) -> dropped
                # segment 2: the canopy height is out of bounds -> NaN, counted
                h_can[2] = 9999.0
                # segment 3: the terrain flag is set -> qc 2
                terr[3] = 1
                # segment 4: the multiple-scattering warning -> qc 1
                msw[4] = 3
                # segment 5: no canopy uncertainty -> qc 1
                unc[5] = INVALID_R4B
                d = ls.create_dataset("delta_time", data=delta)
                d.attrs["units"] = DELTA_TIME_UNITS
                ls.create_dataset("latitude", data=lat.astype(np.float32))
                ls.create_dataset("longitude", data=lon.astype(np.float32))
                lcd = ls.create_dataset("segment_landcover", data=lc)
                lcd.attrs["flag_values"] = np.array(
                    sorted(LANDCOVER_TABLE), np.uint8)
                lcd.attrs["flag_meanings"] = " ".join(
                    LANDCOVER_TABLE[k] for k in sorted(LANDCOVER_TABLE))
                ls.create_dataset("segment_watermask", data=water)
                ls.create_dataset("msw_flag", data=msw)
                ls.create_dataset("terrain_flg", data=terr)
                cd = cn.create_dataset("h_canopy", data=h_can.astype(
                    np.float32))
                cd.attrs["_FillValue"] = np.float32(INVALID_R4B)
                cn.create_dataset("canopy_openness",
                                  data=openness.astype(np.float32))
                cn.create_dataset("n_ca_photons", data=n_ca)
                ud = cn.create_dataset("h_canopy_uncertainty",
                                       data=unc.astype(np.float32))
                ud.attrs["_FillValue"] = np.float32(INVALID_R4B)
                tr.create_dataset("h_te_best_fit",
                                  data=h_te.astype(np.float32))
                tr.create_dataset("n_te_photons", data=n_te)
                for i in range(n):
                    if water[i] == 1 or int(lc[i]) in (80, 200):
                        continue
                    v = [float(h_te[i]), float(h_can[i]), float(openness[i]),
                         float(n_ca[i]), float(n_te[i]), float(lc[i])]
                    for c, (_nm, _u, lo, hi) in enumerate(CHANNELS):
                        if np.isfinite(v[c]) and not (lo <= v[c] <= hi):
                            v[c] = np.nan
                    truth.append({"t": int(round(delta[i])) + EPOCH_2018,
                                  "lat": float(np.float32(lat[i])),
                                  "lon": float(np.float32(lon[i])),
                                  "platform": f10b.platform_hash(track),
                                  "v": v})
    return truth


ADAPTER = ICESat2Adapter

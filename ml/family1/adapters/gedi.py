"""GEDI laser footprints — canopy height, cover and biomass from the Space
Station, one row per quality shot (family 1.0.tf, E-082 wave 6).

PLAIN ENGLISH. GEDI (the Global Ecosystem Dynamics Investigation) is a laser
altimeter that flew on the ISS (the International Space Station). It fires
eight parallel beams at the ground and times the returning pulse, so for each
25-metre circle of forest it gets the whole vertical profile of what the light
bounced off: the ground at the bottom, the leaves above it, the canopy top.
Optical and radar satellites see a forest from above and guess its height;
this measures it. Three products are joined here, shot for shot:

  L2A  the height profile — ground elevation and the RELATIVE HEIGHTS RH25 to
       RH98, the heights below which 25 % to 98 % of the returned laser energy
       lies (RH98 is, in practice, the canopy top).
  L2B  canopy cover (the fraction of ground hidden by leaves) and plant-area
       index (square metres of leaf and stem above a square metre of ground).
  L4A  above-ground biomass density in megagrams of dry matter per hectare,
       and the standard error of that prediction.

THE ROW is one quality shot: eleven channels, its own 25-metre footprint, an
instant in time, and the beam it came from as the platform.

SOURCES, VERIFIED 2026-09-20 FROM THIS SANDBOX (NASA's Common Metadata
Repository — CMR — is anonymous; the granules themselves are Earthdata-Login
protected, so the bytes are a hosted runner's job):

  GEDI02_A version 003   C3974616071-LPCLOUD    2019-04-04 -> ongoing
  GEDI02_B version 003   C3974616135-LPCLOUD    2019-04-04 -> ongoing
  GEDI L4A version 3     C4212593885-ORNL_CLOUD 2019-04-04 -> ongoing
  GEDI L4A version 2.1   C2237824918-ORNL_CLOUD 2019-04-17 -> 2025-07-09

  L2A/L2B bytes   https://data.lpdaac.earthdatacloud.nasa.gov/
                    lp-prod-protected/GEDI02_A.003/<name>/<name>.h5
  L4A v3 bytes    https://data.ornldaac.earthdata.nasa.gov/
                    ornl-cumulus-prod-protected/gedi/
                    GEDI_L4A_AGB_Density_V3/data/<name>.h5
  L4A v2.1 bytes  https://data.ornldaac.earthdata.nasa.gov/protected/gedi/
                    GEDI_L4A_AGB_Density_V2_1/data/<name>.h5

ONE MONTH MEASURED IN FULL (2022-06, every granule CMR lists, sizes summed
from the listing's own `granule_size`):

  | product        | granules | bytes       | mean a granule |
  |----------------|---------:|------------:|---------------:|
  | L2A v003       |    1,428 |    1.48 TB  |      1,062 MB  |
  | L2B v003       |    1,428 |    0.87 TB  |        621 MB  |
  | L4A v3         |    1,428 |    0.23 TB  |        167 MB  |
  | L4A v2.1       |    1,427 |    0.27 TB  |        190 MB  |
  | the three read |    1,428 |  **2.58 TB**|      1,850 MB  |

WHICH L4A VERSION, and why the default is 3 rather than the note's 2.1. The
ledger says "L4A v2.1"; CMR now lists a version 3 whose record starts
2019-04-04, the same day as L2A/L2B version 3, and whose own user guide
(daac.ornl.gov/GEDI/guides/GEDI_L4A_AGB_Density_V3.html, read 2026-09-20)
states that it "uses GEDI02_A Version 3 as input". Version 2.1 was produced
from GEDI02_A version 2, so joining version-3 shots to it would cross a
reprocessing. `GEDI_L4A_VERSION=2.1` selects the older one and the choice is
recorded in the store.

THE GRANULE IDS LINE UP, MEASURED. For 2022-06-01 all four products list
exactly 27 granules, and the key
`(acquisition YYYYDDDHHMMSS, orbit, sub-orbit granule, reference track)` —
read out of the file name — matches one-to-one across all four, with no
granule in one product missing from another. The rest of each name (product
level, PPDS release, production version, V-number) differs by product and is
NOT part of the key:

  L2A      GEDI02_A_2022152133202_O19644_01_T00181_02_004_02_V003
  L2B      GEDI02_B_2022152133202_O19644_01_T00181_02_004_01_V003
  L4A v3   GEDI04_A_2022152133202_O19644_01_T00181_02_004_01_V003
  L4A v2.1 GEDI_L4A_AGB_Density_V2_1.GEDI04_A_2022152133202_O19644_01_
           T00181_02_003_01_V002.h5

The index re-measures that alignment on the window's first day and writes it
into plan.json (`granule_alignment`), so a reprocessing that breaks it is a
refusal rather than a silent hole.

THE FETCH, DECIDED FROM THE MEASUREMENT. 2.58 TB a month is 31 TB for 2022:
a hosted lane has about 86 GB of free disk and six hours, so downloading whole
granules is not a build, it is an impossibility — and it would also be waste,
because the eleven quantities read here are a small part of eight hundred
datasets dominated by waveform groups this store never looks at. Two subset
routes exist and exactly one of them needs no extra service:

  (a) OPeNDAP variable subsetting — NOT AVAILABLE for these collections.
      Measured on CMR 2026-09-20: the only service ASSOCIATED with GEDI02_A
      v003, GEDI02_B v003 and GEDI L4A v3 is the Harmony Trajectory Subsetter
      (S2836723123); none of the three is associated with an OPeNDAP service.
      (GEDI L4A v2.1 alone is, together with the Harmony OPeNDAP SubSetter.)
      An anonymous GET of
      `opendap.earthdata.nasa.gov/collections/<c>/granules/<g>.dmr` answers
      HTTP 401 after two redirects to Earthdata Login for every one of them,
      so nothing further can be confirmed without credentials this sandbox
      does not hold.
  (b) HTTP RANGE READS OF THE HDF5 ITSELF — the route taken. HDF5 is a little
      file system with an index, `h5py` accepts any seekable file-like object,
      and `family1/adapters/_h5range.py` supplies one whose reads are
      `Range:` requests behind a 1 MiB block cache. The bytes fetched are then
      bounded by the datasets asked for rather than by the granule, and the
      probe reports the ratio it actually achieved (`read_fraction` per
      granule, `bytes_per_row` overall). `GEDI_FETCH=whole` switches to
      downloading and deleting each granule, which is the honest fallback if
      a host turns out not to serve 206.

  The Harmony Trajectory Subsetter stays the third option and is deliberately
  NOT built yet: it is an asynchronous job service with its own queue and its
  own failure modes, and it should be bought only if the probe says the range
  route is too slow. The probe's `storage_and_fetch` counts give the numbers
  that decision needs.

THE SIZE GATE. Phase A builds ONE YEAR, 2022, and the probe's numbers decide
the rest (`ml/plans/E082_biosphere_wave.md` §2.1). So every stage past `probe`
refuses a window that crosses a calendar-year boundary unless
`GEDI_ALLOW_BUILD=1` is set — the same shape as `swot`'s refusal, and for the
same reason: a precondition that depends only on the inputs is free at
dispatch and expensive at hour three (ml/CLAUDE.md §0.3).

THE CHANNELS (C = 11, so a row is 27 + 2C = 49 bytes, the ledger's 49 B).
  elev_lowestmode  L2A, metres above the reference ellipsoid: the ground.
  rh25 rh50 rh75 rh90 rh98   L2A `rh` columns 25, 50, 75, 90, 98 of 101.
  cover            L2B, fraction of ground covered by canopy.
  pai              L2B, plant-area index.
  agbd agbd_se     L4A, biomass and its standard error, Mg/ha.
  sensitivity      L2A, the maximum canopy cover the waveform's
                   signal-to-noise could have seen through.
The ledger's eleventh channel is "quality". A FLAG IS NOT A FLOAT16 CHANNEL —
the same argument `swot` makes about its 32-bit quality mask — so the quality
and degrade flags go into `qc`, and `sensitivity`, the per-shot number GEDI's
own quality filtering is built on, takes the channel. C stays 11 and the
ledger's byte arithmetic is unchanged; the swap is stated here rather than
buried.

WHAT A ROW IS.
  time_s    L2A `delta_time`, seconds since 2018-01-01 (the dataset's own
            `units` attribute is used when it carries an epoch; otherwise the
            product's documented epoch is used and the fallback is counted as
            `delta_time_epoch_assumed`). Plus the exact integer seconds from
            1982-01-01 to 2018-01-01.
  lat, lon  `lat_lowestmode`, `lon_lowestmode` — the ground return's position.
  platform  `platform_hash(<beam group name>)`, one of the eight beams;
            `platforms.json` carries the beam's number and its type
            (coverage or full power) READ FROM THE GROUP'S OWN `description`
            attribute, never from a table typed here.
  qc        0 a quality shot with `degrade_flag` 0; 1 a quality shot taken
            during a degraded pointing or trajectory period. A shot whose L2A
            quality flag is 0 is NOT a quality shot and is dropped and counted
            — "quality shots" is what the ledger's row count means.
  values    NaN wherever a product did not cover the shot: a shot with no L2B
            or no L4A partner keeps its L2A channels and carries NaN biomass,
            and every such case is counted by name.

THE QUALITY FLAG'S NAME CHANGED IN VERSION 3, measured. CMR's variable
listing for GEDI02_A v003 (3,336 variables, read anonymously 2026-09-20) has
NO `/BEAMXXXX/quality_flag`: version 3 renamed it `l2a_quality_flag_rel3`
(with `l2a_quality_flag_rel2` kept beside it), and `quality_flag` survives
only under `rx_assess/`. L2B likewise carries `l2b_quality_flag_rel3`, and
L4A version 3 `l4a_quality_flag_rel3` where version 2.1 had `l4_quality_flag`.
So every quantity is RESOLVED against a candidate list and a file that matches
none of them is a refusal carrying the group's own dataset list — the rule
wave 4 wrote for `xco2` and `irtb` (BUILD_LOG.md, "no anonymous route to a
NASA file's variable list"), which here had an anonymous route and used it.
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

# The three collections, each id read from CMR on 2026-09-20 rather than
# transcribed. `short` and `version` are what a human would search with; the
# concept id is what is actually queried, because it cannot be ambiguous.
COLL_L2A = {"concept": "C3974616071-LPCLOUD", "short": "GEDI02_A",
            "version": "003", "record": ("2019-04-04", None)}
COLL_L2B = {"concept": "C3974616135-LPCLOUD", "short": "GEDI02_B",
            "version": "003", "record": ("2019-04-04", None)}
COLL_L4A = {
    "3": {"concept": "C4212593885-ORNL_CLOUD",
          "short": "GEDI_L4A_AGB_Density_V3_2508", "version": "3",
          "record": ("2019-04-04", None)},
    "2.1": {"concept": "C2237824918-ORNL_CLOUD",
            "short": "GEDI_L4A_AGB_Density_V2_1_2056", "version": "2.1",
            "record": ("2019-04-17", "2025-07-09")},
}

# The key that joins the three products. Everything after the track number
# (PPDS release, production version, V-number) differs by product and is
# deliberately outside the group.
STEM = re.compile(r"GEDI0[24]_[AB]_(\d{13})_O(\d+)_(\d+)_T(\d+)_")

# The eight beam groups, read from CMR's variable listing for GEDI02_A v003
# on 2026-09-20 (all three products carry the same eight).
BEAMS = ("BEAM0000", "BEAM0001", "BEAM0010", "BEAM0011",
         "BEAM0101", "BEAM0110", "BEAM1000", "BEAM1011")

RH_PERCENTILES = (25, 50, 75, 90, 98)
RH_COLUMNS = 101

CHANNELS = (
    # the ground, from the product's own valid range (-1000, 25000)
    ("elev_lowestmode", "m", -1000.0, 25000.0),
    # the relative heights, from `rh`'s own valid range (-213, 213)
    ("rh25", "m", -213.0, 213.0),
    ("rh50", "m", -213.0, 213.0),
    ("rh75", "m", -213.0, 213.0),
    ("rh90", "m", -213.0, 213.0),
    ("rh98", "m", -213.0, 213.0),
    ("cover", "1", 0.0, 1.0),
    # PLANT-AREA INDEX AND BIOMASS ARE DELIBERATELY WIDE. The L2B and L4A
    # dictionaries were not reachable anonymously (the LP DAAC L2B page is
    # rendered client-side and the numbered document urls 404), so these two
    # bounds are not the producer's own. A bound that is too TIGHT turns good
    # data into counted NaN; one that is too wide loses nothing but the
    # sentinel, which the fill handling has already removed. The probe's
    # `out_of_bounds` per channel is what narrows them from a distribution.
    ("pai", "m2/m2", 0.0, 50.0),
    ("agbd", "Mg/ha", 0.0, 10000.0),
    ("agbd_se", "Mg/ha", 0.0, 10000.0),
    ("sensitivity", "1", 0.0, 1.0),
)

# Every quantity as a CANDIDATE LIST, most recent name first. A granule that
# matches none of the candidates is a refusal carrying the group's own
# datasets, never a guess (see the module docstring's last paragraph).
L2A_NEEDED = {
    "delta_time": ("delta_time",),
    "lat": ("lat_lowestmode",),
    "lon": ("lon_lowestmode",),
    "elev": ("elev_lowestmode",),
    "rh": ("rh",),
    "shot": ("shot_number",),
    "quality": ("l2a_quality_flag_rel3", "quality_flag",
                "l2a_quality_flag_rel2"),
    "degrade": ("degrade_flag",),
    "sensitivity": ("sensitivity",),
}
L2B_NEEDED = {
    "shot": ("shot_number",),
    "cover": ("cover",),
    "pai": ("pai",),
    "quality": ("l2b_quality_flag_rel3", "l2b_quality_flag",
                "l2b_quality_flag_rel2"),
}
L4A_NEEDED = {
    "shot": ("shot_number",),
    "agbd": ("agbd",),
    "agbd_se": ("agbd_se",),
    "quality": ("l4a_quality_flag_rel3", "l4_quality_flag"),
}

# GEDI's documented no-data sentinel for its float datasets, used only where
# the dataset carries no fill attribute of its own.
GEDI_FILL = -9999.0
# 1982-01-01 -> 2018-01-01, exact integer seconds.
EPOCH_2018 = int((dt.date(2018, 1, 1) - dt.date(1982, 1, 1)).days) * 86400
DELTA_TIME_UNITS = "seconds since 2018-01-01 00:00:00"
FIRST_YEAR = 2019
WORKERS = 2


class FormatError(ValueError):
    """A granule listing or an HDF5 file that is not the product."""


# ================================================================ the index ==
def cmr_granules(params, count=None, attempts=4, page=2000, cap=200000):
    """Every granule matching `params`, paged with `CMR-Search-After`.

    ANONYMOUS — CMR needs no login (the BYTES do). An empty answer comes back
    as an empty list and the CALLER decides whether that is a refusal
    (ml/CLAUDE.md, the 2026-09-14 rule).
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


def parse_entries(entries, product):
    """CMR entries -> {stem key: {name, url, bytes, t0, t1}}.

    The stem key is `(acquisition stamp, orbit, sub-orbit granule, track)`,
    which is what makes the three products joinable (module docstring). A name
    that does not carry one, or two granules claiming the same key, is a
    refusal.
    """
    out, other = {}, []
    for g in entries:
        title = str(g.get("title", ""))
        m = STEM.search(title)
        if not m:
            other.append(title)
            continue
        key = m.groups()
        if key in out:
            raise FormatError(
                f"{product}: two granules for {key}: {out[key]['name']} and "
                f"{title}")
        url = None
        for ln in g.get("links", []):
            h = str(ln.get("href", ""))
            if h.startswith("https") and "protected" in h and h.endswith(".h5"):
                url = h
                break
        if url is None:
            raise FormatError(
                f"{product}: {title} has no protected https .h5 link "
                f"({[l.get('href') for l in g.get('links', [])][:3]})")
        out[key] = {"name": title, "url": url,
                    "bytes": int(float(g.get("granule_size") or 0) * 1e6),
                    "t0": g.get("time_start"), "t1": g.get("time_end")}
    if other:
        raise FormatError(
            f"{product}: {len(other)} listed granule name(s) carry no "
            f"GEDI stem, e.g. {sorted(other)[:3]} — refusing to join a "
            f"listing this parser does not understand")
    return out


# ================================================================== reading ==
def resolve_name(group, candidates, what, where):
    """The first candidate present in `group`, or a refusal naming what is."""
    for c in candidates:
        if c in group:
            return c
    raise FormatError(
        f"{where}: none of {list(candidates)} is a dataset of this group, so "
        f"'{what}' cannot be read. The group holds "
        f"{sorted(k for k in group)[:40]}")


def _as_float(ds, a):
    """Raw values -> float64 with the product's no-data turned into NaN.

    The dataset's own fill attribute is preferred; GEDI's documented -9999 is
    the fallback and is applied as well, because several L2A/L2B datasets
    carry the sentinel without declaring it.
    """
    v = np.asarray(a, np.float64)
    fills = []
    for key in ("_FillValue", "fill_value", "missing_value"):
        val = ds.attrs.get(key) if hasattr(ds, "attrs") else None
        if val is None:
            continue
        for x in np.atleast_1d(np.asarray(val)).ravel():
            try:
                fills.append(float(x))
            except (TypeError, ValueError):
                pass
    fills.append(GEDI_FILL)
    n = 0
    for f in set(fills):
        hit = v == f
        n += int(hit.sum())
        v[hit] = np.nan
    v[~np.isfinite(v)] = np.nan
    return v, n


def delta_time_seconds(ds, counts):
    """`delta_time` -> seconds since 1982-01-01, float64.

    The dataset's own `units` attribute is used when it names an epoch; GEDI
    writes "seconds since 2018-01-01" there. When the attribute is absent the
    documented epoch is used and the assumption is COUNTED, so a file that
    stopped carrying it cannot pass unnoticed.
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


def beam_description(h5, beam):
    """The beam group's own `description` attribute -> "coverage" | "power"."""
    g = h5.get(beam)
    if g is None:
        return None, None
    d = g.attrs.get("description")
    if isinstance(d, bytes):
        d = d.decode("utf-8", "replace")
    if d is None:
        return None, None
    text = str(d)
    low = text.lower()
    kind = ("coverage" if "coverage" in low else
            "power" if "power" in low else None)
    return kind, text


def read_l2a_beam(h5, beam, counts):
    """One L2A beam -> the spine: times, positions, the eight L2A columns."""
    g = h5[beam]
    where = f"L2A/{beam}"
    name = {k: resolve_name(g, v, k, where) for k, v in L2A_NEEDED.items()}
    rh = g[name["rh"]]
    if rh.ndim != 2 or rh.shape[1] != RH_COLUMNS:
        raise FormatError(
            f"{where}: `{name['rh']}` has shape {rh.shape}, not (shots, "
            f"{RH_COLUMNS}) — the relative-height percentiles cannot be "
            f"addressed by column")
    t = delta_time_seconds(g[name["delta_time"]], counts)
    lat = np.asarray(g[name["lat"]][:], np.float64)
    lon = np.asarray(g[name["lon"]][:], np.float64)
    shot = np.asarray(g[name["shot"]][:], np.uint64)
    qual = np.asarray(g[name["quality"]][:], np.int64)
    degr = np.asarray(g[name["degrade"]][:], np.int64)
    elev, nf = _as_float(g[name["elev"]], g[name["elev"]][:])
    sens, nf2 = _as_float(g[name["sensitivity"]], g[name["sensitivity"]][:])
    fills = nf + nf2
    cols = [elev]
    for p in RH_PERCENTILES:
        # ONE COLUMN AT A TIME, so the HDF5 library reads the chunks that
        # intersect that column rather than the whole 101-wide array. Whether
        # the chunking lets it is the probe's measurement, not an assumption.
        v, k = _as_float(rh, rh[:, p])
        fills += k
        cols.append(v)
    counts["fill_values_l2a"] = counts.get("fill_values_l2a", 0) + int(fills)
    counts["shots_in_granule"] = counts.get("shots_in_granule", 0) + len(shot)
    return {"t": t, "lat": lat, "lon": lon, "shot": shot, "qual": qual,
            "degr": degr, "cols": cols, "sens": sens,
            "resolved": {k: name[k] for k in sorted(name)}}


def read_partner_beam(h5, beam, needed, label, counts):
    """One L2B or L4A beam -> (shot numbers, {quantity: values}, quality)."""
    g = h5.get(beam)
    if g is None:
        return None
    where = f"{label}/{beam}"
    name = {k: resolve_name(g, v, k, where) for k, v in needed.items()}
    shot = np.asarray(g[name["shot"]][:], np.uint64)
    qual = np.asarray(g[name["quality"]][:], np.int64)
    vals, fills = {}, 0
    for k in needed:
        if k in ("shot", "quality"):
            continue
        v, n = _as_float(g[name[k]], g[name[k]][:])
        vals[k] = v
        fills += n
    counts[f"fill_values_{label.lower()}"] = \
        counts.get(f"fill_values_{label.lower()}", 0) + int(fills)
    return shot, vals, qual, {k: name[k] for k in sorted(name)}


def join_by_shot(spine_shot, other_shot):
    """Index of each spine shot in `other_shot`, or -1. Sorted search.

    `shot_number` is GEDI's own cross-product key (its format is
    OOOOOBBRRGNNNNNNNN — orbit, beam, reserved, sub-orbit granule, shot), so a
    shot present in two products carries the identical integer in both.
    """
    order = np.argsort(other_shot, kind="stable")
    s = other_shot[order]
    pos = np.searchsorted(s, spine_shot)
    pos = np.clip(pos, 0, max(len(s) - 1, 0))
    if len(s) == 0:
        return np.full(len(spine_shot), -1, np.int64)
    hit = s[pos] == spine_shot
    return np.where(hit, order[pos], -1).astype(np.int64)


# ================================================================== adapter ==
class GEDIAdapter(f10b.SourceAdapter):
    store = "gedi"
    title = ("GEDI laser footprints: L2A relative heights joined to L2B "
             "cover and plant-area index and to L4A biomass by shot number, "
             "one row per quality shot")
    family = "1tf"
    distribution = "public"
    licence = {"name": "NASA open data (EOSDIS, no restriction on use)",
               "redistribution": "yes",
               "redistribution_confirmed": True,
               "derived_works": "free",
               "attribution": (
                   "Dubayah, R. et al. GEDI L2A Elevation and Height Metrics "
                   "and L2B Canopy Cover and Vertical Profile Metrics, "
                   "Version 3, NASA LP DAAC; Dubayah, R. et al. GEDI L4A "
                   "Footprint Level Aboveground Biomass Density, ORNL DAAC.")}
    time_dtype = "int32"
    credentials = ("EARTHDATA_USERNAME", "EARTHDATA_PASSWORD")
    channels = CHANNELS
    # a 25 m footprint: log2(0.025 / 27.83) = -10.12, the ledger's -10.1
    log2_fp = float(np.log2(0.025 / 27.83))
    # an instantaneous shot: the clamp
    log2_dt = -12.0
    per_year = True
    first_year = FIRST_YEAR
    platform_meta = True
    fetch_month_scope = "month"
    qc_policy = (
        "A row is a QUALITY SHOT. The L2A quality flag (`l2a_quality_flag_"
        "rel3` in version 3, `quality_flag` in version 2 — resolved against a "
        "candidate list and refused if none matches) must be 1; a shot with 0 "
        "is dropped and counted (`shots_quality_flag_zero`), which is what the "
        "ledger's 4-10 x 10^9 'quality rows' means. `qc` then carries 0 for a "
        "shot with `degrade_flag` 0 and 1 for one taken during a degraded "
        "pointing or trajectory period, with the whole degrade histogram "
        "recorded. A shot with no finite time or position is dropped and "
        "counted. The L2B and L4A channels are NaN where that product's own "
        "quality flag is 0 (`l2b_quality_zero`, `l4a_quality_zero`) and where "
        "the shot has no partner at all (`shots_without_l2b`, "
        "`shots_without_l4a`, `granules_without_l4a`) — an absence is never a "
        "zero. A value outside its channel's bounds becomes NaN for that "
        "channel alone and is counted, never clipped. The eleventh channel is "
        "`sensitivity`, not the quality flag: float16 cannot carry a flag, so "
        "the flag is in `qc` and the number is in the channel.")
    sources = (
        f"{CMR}?collection_concept_id={COLL_L2A['concept']} (anonymous)",
        f"{CMR}?collection_concept_id={COLL_L2B['concept']} (anonymous)",
        f"{CMR}?collection_concept_id={COLL_L4A['3']['concept']} (anonymous)",
        "https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/"
        "GEDI02_A.003/<granule>/<granule>.h5",
        "https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/"
        "GEDI02_B.003/<granule>/<granule>.h5",
        "https://data.ornldaac.earthdata.nasa.gov/ornl-cumulus-prod-protected/"
        "gedi/GEDI_L4A_AGB_Density_V3/data/<granule>.h5")
    verified = (
        "2026-09-20 from this sandbox, anonymously against NASA CMR (the "
        "granules are Earthdata-protected, so the bytes are a hosted "
        "runner's job). Collections: GEDI02_A 003 C3974616071-LPCLOUD and "
        "GEDI02_B 003 C3974616135-LPCLOUD, both 2019-04-04 -> ongoing; GEDI "
        "L4A version 3 C4212593885-ORNL_CLOUD, 2019-04-04 -> ongoing; GEDI "
        "L4A version 2.1 C2237824918-ORNL_CLOUD, 2019-04-17 -> 2025-07-09. "
        "2022-06 IN FULL: 1,428 granules each of L2A (1,516,813 MB), L2B "
        "(886,429 MB) and L4A v3 (238,860 MB), 1,427 of L4A v2.1 (271,839 "
        "MB) — 2.58 TB for the three products read. 2022-06-01: all four "
        "products list 27 granules and the (stamp, orbit, sub-orbit granule, "
        "track) key matches one-to-one across all four. Dataset names from "
        "CMR's own variable listing (3,336 variables for L2A v003, 1,672 for "
        "L2B, 1,056 for L4A v3): eight beam groups BEAM0000..BEAM1011, and "
        "version 3 has NO `/BEAMXXXX/quality_flag` — it is "
        "`l2a_quality_flag_rel3`. Service associations: the Harmony "
        "Trajectory Subsetter for all three, and OPeNDAP for L4A v2.1 ONLY. "
        "NOT MEASURED HERE, and what the runner probe is for: whether these "
        "hosts serve HTTP 206 for a range read, what fraction of a granule "
        "the eleven datasets cost, shots a granule, the quality and degrade "
        "histograms, the join rates, and therefore rows and bytes a year.")
    notes = ""
    smoke_window = ("2021-12-01", "2022-01-31")
    smoke_probe_month = "2021-12"

    def __init__(self):
        # READ AT CONSTRUCTION TIME, so the fresh adapter `stage_probe` builds
        # for itself sees the same knobs (the mechanism `lossyear` and `swot`
        # use). `family1-build.yml`'s `adapter_env` input is how a dispatch
        # sets them.
        self._session = None
        self._listing = {}
        self._beams = {}
        self._resolved = {}
        self.l4a_version = (os.environ.get("GEDI_L4A_VERSION") or "3").strip()
        if self.l4a_version not in COLL_L4A:
            sys.exit(f"GEDI_L4A_VERSION must be one of {sorted(COLL_L4A)}")
        self.l4a = COLL_L4A[self.l4a_version]
        self.fetch_mode = (os.environ.get("GEDI_FETCH") or "range").strip()
        if self.fetch_mode not in ("range", "whole"):
            sys.exit("GEDI_FETCH must be 'range' (HDF5 byte-range subset) or "
                     "'whole' (download the granule)")
        # THE CAP IS THE PROBE'S, NEVER THE BUILD'S. `GEDI_MAX_GRANULES` bounds
        # how many granules ONE PROBE reads (spread over the month), so a
        # probe costs minutes; a build reads every granule in its window. On
        # 2026-09-20 the first ICESat-2 build lanes ran with the probe's
        # default cap, read twelve granules of 4,884, and MARKED THE YEAR DONE
        # with a fraction of it in 38 seconds -- a short store that reported
        # success (ml/CLAUDE.md 0.2). So `fetch_year` never applies a cap, and
        # a cap set explicitly for a build is a refusal unless
        # GEDI_ALLOW_CAPPED_BUILD=1 names the intention.
        raw_cap = (os.environ.get("GEDI_MAX_GRANULES") or "").strip()
        try:
            self.max_granules = int(raw_cap) if raw_cap else 6
        except ValueError:
            sys.exit("GEDI_MAX_GRANULES must be an integer")
        self.cap_was_set = bool(raw_cap)
        self.allow_capped_build = (
            os.environ.get("GEDI_ALLOW_CAPPED_BUILD") or "") == "1"
        self.allow_build = (os.environ.get("GEDI_ALLOW_BUILD") or "") == "1"
        self.notes = (
            f"L4A version {self.l4a_version} ({self.l4a['concept']}); fetch "
            f"mode {self.fetch_mode!r}"
            + (f"; a probe reads at most {self.max_granules} granule(s), "
               f"spread over its month; a build reads every granule in its "
               f"window" if self.max_granules else "; every granule in the "
               f"window")
            + ". PHASE A IS ONE YEAR (2022): a window crossing a calendar-year "
              "boundary is refused past `probe` unless GEDI_ALLOW_BUILD=1, "
              "because one month of the three products is 2.58 TB native and "
              "the record's cost is the probe's to measure. A capped call is "
              "not a whole month and says so here.")

    # -------------------------------------------------------------- listing --
    def listing(self, ctx, t_lo, t_hi):
        """{stem key: {"l2a": e, "l2b": e, "l4a": e or None}} for a window."""
        key = (int(t_lo), int(t_hi))
        if key in self._listing:
            return self._listing[key]
        if ctx.source_dir:
            got = self._listing_local(ctx)
        else:
            temporal = f"{_iso(t_lo)},{_iso(t_hi)}"
            got = {}
            for tag, coll in (("l2a", COLL_L2A), ("l2b", COLL_L2B),
                              ("l4a", self.l4a)):
                try:
                    ents = cmr_granules(
                        {"collection_concept_id": coll["concept"],
                         "temporal": temporal, "sort_key": "start_date"},
                        count=ctx.count_bytes, attempts=ctx.a.attempts)
                except FormatError as e:
                    sys.exit(f"REFUSING gedi: {e}")
                if not ents:
                    sys.exit(
                        f"REFUSING gedi: CMR lists no {coll['short']} granule "
                        f"for {temporal} — an empty listing is a refusal, not "
                        f"an empty month (ml/CLAUDE.md, the 2026-09-14 rule)")
                try:
                    got[tag] = parse_entries(ents, coll["short"])
                except FormatError as e:
                    sys.exit(f"REFUSING gedi: {e}")
        merged = {}
        for k, e in got["l2a"].items():
            merged[k] = {"l2a": e, "l2b": got["l2b"].get(k),
                         "l4a": got["l4a"].get(k)}
        self._listing[key] = (merged, got)
        return self._listing[key]

    def _listing_local(self, ctx):
        """The smoke's synthetic archive, read as if it were a CMR listing."""
        root = os.path.join(ctx.source_dir, self.store)
        names = sorted(os.listdir(root)) if os.path.isdir(root) else []
        got = {"l2a": {}, "l2b": {}, "l4a": {}}
        for n in names:
            if not n.endswith(".h5"):
                continue
            p = os.path.join(root, n)
            tag = ("l2a" if n.startswith("GEDI02_A") else
                   "l2b" if n.startswith("GEDI02_B") else
                   "l4a" if "GEDI04_A" in n else None)
            if tag is None:
                continue
            m = STEM.search(n)
            if not m:
                raise FormatError(f"{n}: no GEDI stem")
            got[tag][m.groups()] = {
                "name": n[:-3], "url": p, "bytes": os.path.getsize(p),
                "t0": None, "t1": None}
        if not got["l2a"]:
            sys.exit(f"REFUSING gedi: no GEDI02_A .h5 under {root} — the "
                     f"smoke's synthetic archive is missing")
        ctx.count_bytes(sum(e["bytes"] for g in got.values()
                            for e in g.values()))
        return got

    # ---------------------------------------------------------------- bytes --
    def _open(self, ctx, entry, tag):
        if ctx.source_dir:
            # what the network would have carried for this granule, so the
            # probe's bytes-per-row is exercised by the smoke too
            ctx.count_bytes(os.path.getsize(entry["url"]))
            return hr.open_local(entry["url"])
        if self._session is None:
            self._session = cm.earthdata_session()
        if self.fetch_mode == "whole":
            dest = os.path.join(ctx.scratch, self.store,
                                f"{tag}_{entry['name']}.h5")
            return hr.open_whole(self._session, entry["url"], dest,
                                 attempts=max(1, ctx.a.attempts))
        return hr.open_range(self._session, entry["url"],
                             attempts=max(1, ctx.a.attempts))

    def _close(self, op):
        if op is None:
            return None
        hr.finish_range(op)
        st = op.stats()
        op.close()
        return st

    # ----------------------------------------------------------------- rows --
    def _rows(self, ctx, label, t_lo, t_hi, cap=0):
        merged, got = self.listing(ctx, t_lo, t_hi)
        keys = [k for k in sorted(merged)
                if in_window(merged[k]["l2a"], k, t_lo, t_hi)]
        counts = {"granules_listed_l2a": len(got["l2a"]),
                  "granules_listed_l2b": len(got["l2b"]),
                  "granules_listed_l4a": len(got["l4a"]),
                  "l4a_version": self.l4a_version,
                  "fetch_mode": self.fetch_mode,
                  "max_granules_cap": cap}
        if cap and len(keys) > cap:
            counts["granules_skipped_by_cap"] = len(keys) - cap
            keys = spread(keys, cap)
        counts["granules_wanted"] = len(keys)
        per_granule = []
        t0 = time.time()
        for k in keys:
            e = merged[k]
            unit = f"{label} granule {e['l2a']['name']}"
            if e["l2b"] is None:
                ctx.note_absent(unit, "no GEDI02_B granule for this stem")
                counts["granules_without_l2b"] = \
                    counts.get("granules_without_l2b", 0) + 1
            if e["l4a"] is None:
                counts["granules_without_l4a"] = \
                    counts.get("granules_without_l4a", 0) + 1
            ops = {}
            try:
                for tag in ("l2a", "l2b", "l4a"):
                    if e[tag] is not None:
                        ops[tag] = self._open(ctx, e[tag], tag)
            except f10b._NotFound as why:
                for op in ops.values():
                    op.close()
                ctx.note_absent(unit, f"listed in CMR, and {why}")
                continue
            except hr.RangeError as why:
                for op in ops.values():
                    op.close()
                sys.exit(f"REFUSING gedi: {why}")
            try:
                for out in self._granule_rows(ctx, e, ops, counts, t_lo, t_hi):
                    yield label, out, None
            except FormatError as why:
                sys.exit(f"REFUSING gedi: {e['l2a']['name']}: {why}")
            finally:
                meta = {"granule": e["l2a"]["name"]}
                for tag, op in ops.items():
                    st = self._close(op)
                    if st:
                        meta[tag] = st
                per_granule.append(meta)
        counts["fetch_seconds"] = round(time.time() - t0, 1)
        counts["per_granule"] = per_granule[:20]
        # A LIST OF ONE, not a dict: `f10b._merge_counts` adds the VALUES of a
        # dict one level deep, so a dict whose values are dataset NAMES would
        # be concatenated into nonsense the second time a year's counts were
        # merged. A list concatenates, which is what a ledger wants.
        counts["resolved_datasets"] = [dict(self._resolved)] \
            if self._resolved else []
        fb = storage_and_fetch(counts, per_granule, self.C)
        counts["storage_and_fetch"] = [fb] if fb else []
        yield label, None, counts

    def _granule_rows(self, ctx, entry, ops, counts, t_lo, t_hi):
        """One granule, beam by beam — a batch per beam, so memory is bounded
        by one beam's shots rather than by a whole 1 GB granule."""
        h2a = ops["l2a"].h5
        beams = [b for b in sorted(h2a) if b.startswith("BEAM")]
        if not beams:
            raise FormatError(
                f"{entry['l2a']['name']}: no BEAM group; the file holds "
                f"{sorted(h2a)[:20]}")
        counts["beams_seen"] = counts.get("beams_seen", 0) + len(beams)
        for beam in beams:
            kind, text = beam_description(h2a, beam)
            if beam not in self._beams or kind:
                self._beams[beam] = {"beam_type": kind, "description": text}
            s = read_l2a_beam(h2a, beam, counts)
            self._resolved.setdefault("L2A", s["resolved"])
            n = len(s["shot"])
            if n == 0:
                continue
            values = np.full((n, self.C), np.nan, np.float64)
            for i, col in enumerate(s["cols"]):
                values[:, i] = col
            values[:, self.C - 1] = s["sens"]
            # L2B: cover and pai
            self._merge_partner(ops.get("l2b"), beam, L2B_NEEDED, "L2B",
                                s["shot"], values, {"cover": 6, "pai": 7},
                                counts)
            # L4A: biomass and its standard error
            self._merge_partner(ops.get("l4a"), beam, L4A_NEEDED, "L4A",
                                s["shot"], values,
                                {"agbd": 8, "agbd_se": 9}, counts)
            keep = s["qual"] == 1
            counts["shots_quality_flag_zero"] = \
                counts.get("shots_quality_flag_zero", 0) + int((~keep).sum())
            good = (keep & np.isfinite(s["t"]) & np.isfinite(s["lat"])
                    & np.isfinite(s["lon"]))
            counts["shots_no_time_or_position"] = \
                counts.get("shots_no_time_or_position", 0) + \
                int((keep & ~good).sum())
            inside = good & (s["t"] >= t_lo) & (s["t"] <= t_hi)
            counts["shots_outside_window"] = \
                counts.get("shots_outside_window", 0) + \
                int((good & ~inside).sum())
            if not inside.any():
                continue
            t = s["t"][inside]
            v = values[inside]
            degr = s["degr"][inside]
            hist = counts.setdefault("degrade_flag", {})
            for u, c in zip(*np.unique(degr, return_counts=True)):
                hist[str(int(u))] = hist.get(str(int(u)), 0) + int(c)
            qc = (degr != 0).astype(np.uint8)
            oob = self.mask_bounds(v)
            if oob:
                f10b._merge_counts(counts, {"out_of_bounds": oob})
            counts["rows_kept"] = counts.get("rows_kept", 0) + int(t.size)
            plat = np.full(t.size, f10b.platform_hash(beam), np.int64)
            yield self.pack(t, s["lat"][inside], s["lon"][inside], v, plat, qc)

    def _merge_partner(self, op, beam, needed, label, shot, values, where,
                       counts):
        """Join one partner product's columns onto the L2A spine, by shot."""
        low = label.lower()
        if op is None:
            counts[f"shots_without_{low}_granule"] = \
                counts.get(f"shots_without_{low}_granule", 0) + len(shot)
            return
        got = read_partner_beam(op.h5, beam, needed, label, counts)
        if got is None:
            counts[f"beams_without_{low}"] = \
                counts.get(f"beams_without_{low}", 0) + 1
            counts[f"shots_without_{low}"] = \
                counts.get(f"shots_without_{low}", 0) + len(shot)
            return
        o_shot, o_vals, o_qual, resolved = got
        self._resolved.setdefault(label, resolved)
        idx = join_by_shot(shot, o_shot)
        miss = idx < 0
        counts[f"shots_without_{low}"] = \
            counts.get(f"shots_without_{low}", 0) + int(miss.sum())
        take = np.where(miss, 0, idx)
        ok = (~miss) & (o_qual[take] == 1)
        counts[f"{low}_quality_zero"] = counts.get(f"{low}_quality_zero", 0) + \
            int(((~miss) & (o_qual[take] != 1)).sum())
        for k, col in where.items():
            v = o_vals[k][take]
            values[:, col] = np.where(ok, v, np.nan)

    # ------------------------------------------------------------- contract --
    def fetch_preflight(self, ctx):
        """Both guards where the inputs are all they have cost (§0.3).

        THE SIZE GATE. One month of the three products is 2.58 TB native and
        phase A is one year (2022). A window that crosses a calendar-year
        boundary is refused past `probe` unless GEDI_ALLOW_BUILD=1.

        THE CREDENTIAL GUARD. The granules are Earthdata-protected; a fetch
        with no way to authenticate spends the whole listing to discover a
        page of 401s (family1-build run #152, on `swot`).
        """
        if ctx.source_dir:
            # the smoke's synthetic archive: building THAT is how the code is
            # exercised and is not a build of the store
            return None
        if getattr(ctx.a, "parts_from_hub", False):
            # a box assembling parked lanes reads no source: no size gate to
            # apply and no Earthdata credential to demand (swot's preflight
            # says the same; irtb's whole-record assembly #724 died on this
            # guard, 2026-09-24)
            return None
        stage = getattr(ctx.a, "stage", "")
        if stage != "probe" and not self.allow_build:
            if ctx.d_lo.year != ctx.d_hi.year:
                sys.exit(
                    f"REFUSING to build gedi for {ctx.d_lo} .. {ctx.d_hi}: "
                    f"that crosses a calendar-year boundary and phase A is "
                    f"ONE YEAR (2022), with the rest decided on the probe's "
                    f"measured rows and bytes (ml/plans/E082_biosphere_wave.md "
                    f"§2.1). One month of GEDI L2A + L2B + L4A is 2.58 TB "
                    f"native, measured on CMR 2026-09-20. Probe it with "
                    f"`--stage probe --probe-month 2022-06` (the report "
                    f"lands as ml/family1/probes/gedi_2022-06.json and its "
                    f"`storage_and_fetch` block is the decision material); "
                    f"set GEDI_ALLOW_BUILD=1 once those numbers are "
                    f"recorded. Nothing has been fetched.")
        if not cm.earthdata_ready():
            sys.exit(
                "REFUSING gedi: the granules are Earthdata-protected and this "
                "process can authenticate to Earthdata Login in NEITHER way: "
                "no EARTHDATA_USERNAME / EARTHDATA_PASSWORD in the "
                "environment and no netrc naming urs.earthdata.nasa.gov. "
                "family1-build.yml gives both on a HOSTED runner from the "
                "repository secrets (ml/CLAUDE.md §6); a box gets neither. "
                "Nothing has been fetched.")
        return None

    def index(self, ctx):
        """List the window and RE-MEASURE the three products' alignment.

        Contract rule 4: a url pattern is a hypothesis the index verifies. The
        hypothesis here is that the three products share one granule key, and
        the index checks it on the window's first day rather than trusting the
        sentence in the docstring.
        """
        merged, got = self.listing(ctx, ctx.t_lo, ctx.t_hi)
        keys = sorted(merged)
        day = {}
        if keys and not ctx.source_dir:
            d0 = ctx.d_lo
            lo = f10b.seconds_since_epoch(d0)
            hi = lo + 86399
            dmerged, dgot = self.listing(ctx, lo, hi)
            day = {"date": str(d0),
                   "granules": {k: len(v) for k, v in dgot.items()},
                   "l2a_without_l2b": sum(1 for v in dmerged.values()
                                          if v["l2b"] is None),
                   "l2a_without_l4a": sum(1 for v in dmerged.values()
                                          if v["l4a"] is None),
                   "l2b_without_l2a": len(set(dgot["l2b"]) - set(dgot["l2a"])),
                   "l4a_without_l2a": len(set(dgot["l4a"]) - set(dgot["l2a"]))}
            if day["l2a_without_l2b"] or day["l2b_without_l2a"]:
                sys.exit(
                    f"REFUSING gedi: on {d0} the L2A and L2B listings do not "
                    f"line up ({day}) — the shot-number join is written on "
                    f"the measurement that they do (2026-09-20, 27 granules "
                    f"each, one-to-one). A reprocessing that breaks it is a "
                    f"refusal, not a hole.")
        per_day = {}
        for k in keys:
            t0 = merged[k]["l2a"]["t0"]
            if t0:
                per_day[t0[:10]] = per_day.get(t0[:10], 0) + 1
        nb = {tag: sum(e["bytes"] for e in got[tag].values()) for tag in got}
        return {
            "dataset": ("GEDI L2A + L2B version 3 (LP DAAC) joined to GEDI "
                        f"L4A version {self.l4a_version} (ORNL DAAC) by shot "
                        f"number"),
            "url": f"{CMR}?collection_concept_id={COLL_L2A['concept']}",
            "collections": {"l2a": COLL_L2A, "l2b": COLL_L2B,
                            "l4a": self.l4a},
            "l4a_version": self.l4a_version,
            "fetch_mode": self.fetch_mode,
            "max_granules": self.max_granules,
            "window": [str(ctx.d_lo), str(ctx.d_hi)],
            "granules": {tag: len(got[tag]) for tag in got},
            "granules_joined": len(keys),
            "bytes": nb,
            "bytes_total": int(sum(nb.values())),
            "bytes_per_granule_mean": {
                tag: (round(nb[tag] / len(got[tag]), 1) if got[tag] else None)
                for tag in got},
            "granules_per_day": per_day,
            "granule_alignment": day,
            "beams_expected": list(BEAMS),
            "channels": self.channel_names,
            "record": [merged[keys[0]]["l2a"]["t0"],
                       merged[keys[-1]]["l2a"]["t1"]] if keys else None,
            "one_year_only": not self.allow_build,
        }

    def fetch_year(self, ctx, year):
        lo = max(f10b.seconds_since_epoch(dt.date(int(year), 1, 1)), ctx.t_lo)
        hi = min(f10b.seconds_since_epoch(dt.date(int(year), 12, 31)) + 86399,
                 ctx.t_hi)
        if self.cap_was_set and self.max_granules and not self.allow_capped_build:
            sys.exit(f"{self.store}: GEDI_MAX_GRANULES={self.max_granules} is "
                     f"set for a BUILD. A capped build marks the year done "
                     f"with a fraction of its granules and reports success; "
                     f"the cap is the probe's. Unset it, or say "
                     f"GEDI_ALLOW_CAPPED_BUILD=1 if a partial year is what "
                     f"is wanted (it is recorded in store.json).")
        cap = self.max_granules if self.allow_capped_build else 0
        yield from self._rows(ctx, str(year), lo, hi, cap=cap)

    def fetch_month(self, ctx, year, month):
        lo, hi = f10b.month_bounds_s(year, month)
        yield from self._rows(ctx, f"{year}-{int(month):02d}",
                              max(lo, ctx.t_lo), min(hi, ctx.t_hi),
                              cap=self.max_granules)

    # ------------------------------------------------------------ platforms --
    def platforms(self, ctx):
        """The eight beams. The TYPE comes from the group's own attribute."""
        out = {}
        for b in sorted(set(BEAMS) | set(self._beams)):
            seen = self._beams.get(b) or {}
            out[int(f10b.platform_hash(b))] = {
                "id": b,
                "beam_bits": b[4:],
                "beam_type": seen.get("beam_type"),
                "description": seen.get("description"),
                "beam_type_source": (
                    "the BEAMXXXX group's own `description` attribute"
                    if seen.get("beam_type") else
                    "not read yet — no granule holding this beam has been "
                    "opened in this run"),
                "instrument": "GEDI (Global Ecosystem Dynamics Investigation)",
            }
        return out

    # ---------------------------------------------------------------- smoke --
    def smoke_sources(self, root, d_lo, d_hi, seed=20260920):
        truth = make_smoke_sources(root, d_lo, d_hi, seed)
        self.__init__()
        return truth


def _iso(t_s):
    """Seconds since 1982-01-01 -> the ISO stamp CMR's `temporal` wants."""
    base = dt.datetime(1982, 1, 1) + dt.timedelta(seconds=int(t_s))
    return base.strftime("%Y-%m-%dT%H:%M:%SZ")


def stem_seconds(key):
    """The granule key's own YYYYDDDHHMMSS stamp -> seconds since 1982-01-01.

    That field is the granule's ACQUISITION time and is part of the file name
    the three products share, so it dates a granule with no listing metadata
    at all — which is what the smoke's local archive has.
    """
    s = key[0]
    y, doy = int(s[:4]), int(s[4:7])
    hh, mm, ss = int(s[7:9]), int(s[9:11]), int(s[11:13])
    d = dt.date(y, 1, 1) + dt.timedelta(days=doy - 1)
    return f10b.seconds_since_epoch(d) + hh * 3600 + mm * 60 + ss


GRANULE_MAX_SECONDS = 6 * 3600     # far longer than a GEDI sub-orbit granule


def spread(keys, n):
    """`n` of `keys`, EVENLY SPACED, keeping their order.

    A capped probe that read the first six granules of a month would measure
    six quarter-orbits of one morning — one set of latitudes, one set of land
    covers, one host's mood. Spreading them over the window is the same cost
    and a representative one, and it exercises the tail of the listing as
    well as its head.
    """
    keys = list(keys)
    if n <= 0 or len(keys) <= n:
        return keys
    step = len(keys) / float(n)
    return [keys[min(len(keys) - 1, int(i * step))] for i in range(n)]


def in_window(entry, key, t_lo, t_hi):
    """Does this granule overlap [t_lo, t_hi] seconds?

    The listing's own `time_start`/`time_end` when CMR gave them; otherwise
    the name's acquisition stamp, widened by more than any granule's length so
    a granule that STARTS before the window but reaches into it is kept.
    """
    if entry.get("t0") and entry.get("t1"):
        def s(stamp):
            d = dt.datetime.strptime(stamp[:19], "%Y-%m-%dT%H:%M:%S")
            return f10b.seconds_since_epoch(d.date()) + d.hour * 3600 + \
                d.minute * 60 + d.second
        return s(entry["t0"]) <= t_hi and s(entry["t1"]) >= t_lo
    t = stem_seconds(key)
    return (t_lo - GRANULE_MAX_SECONDS) <= t <= t_hi


def storage_and_fetch(counts, per_granule, C):
    """What the probe measured, turned into the numbers the decision needs.

    Everything here comes from the fetch itself — granules read, bytes read
    against the granules' own sizes, rows kept — and from the store's own byte
    rule (a tier-P row is 27 + 2C bytes). The extrapolation basis is the
    MEASURED 2022-06 listing, not a projection.
    """
    n = len(per_granule)
    rows = int(counts.get("rows_kept") or 0)
    if not n or not rows:
        return None
    native = read = 0
    for m in per_granule:
        for tag in ("l2a", "l2b", "l4a"):
            st = m.get(tag)
            if st:
                native += int(st["granule_bytes"])
                read += int(st["bytes_read"])
    row_bytes = 27 + 2 * C
    # 2022-06, measured on CMR 2026-09-20
    granules_month, bytes_month = 1428, 2_642_102_500_000
    return {
        "measured": {
            "granules_read": n,
            "rows_kept": rows,
            "rows_per_granule": round(rows / n, 1),
            "native_bytes_of_those_granules": native,
            "bytes_read": read,
            "read_fraction": (round(read / native, 6) if native else None),
            "fetch_seconds": counts.get("fetch_seconds"),
            "bytes_per_second": (
                round(read / counts["fetch_seconds"], 1)
                if counts.get("fetch_seconds") else None),
        },
        "extrapolation_basis": {
            "granules_per_month_2022_06": granules_month,
            "native_bytes_per_month_2022_06": bytes_month,
            "note": ("1,428 granules of each of L2A, L2B and L4A v3 in "
                     "2022-06, 2.64e12 bytes in all, measured from CMR's own "
                     "granule sizes on 2026-09-20"),
        },
        "projected_one_year_2022": {
            "granules": granules_month * 12,
            "rows": round(rows / n * granules_month * 12),
            "store_bytes": round(rows / n * granules_month * 12 * row_bytes),
            "bytes_to_fetch": (round(read / native * bytes_month * 12)
                               if native else None),
            "fetch_hours": (
                round(read / native * bytes_month * 12
                      / (read / counts["fetch_seconds"]) / 3600, 2)
                if native and counts.get("fetch_seconds") and read else None),
        },
        "bytes_per_row_stored": row_bytes,
    }


# ==================================================================== smoke ==
# A synthetic archive in the products' own group and dataset layout: three
# .h5 files per granule stem, eight beam groups each, the exact dataset names
# the adapter resolves. Two months, so the per-year split and the probe's
# single month are both exercised; two granule stems have no L4A partner at
# all and one beam drops shots from L2B and L4A, so every join branch the
# counts name is taken by the truth as well as by the code.
SMOKE_BEAMS = ("BEAM0000", "BEAM0101")
SMOKE_SHOTS = 6
SMOKE_STEMS = (
    # (stamp, orbit, sub-orbit granule, track, month offset, has L4A).
    # The stamp is YYYYDDDHHMMSS, the product's own acquisition field, and it
    # agrees with the day the shots are written on: DOY 337 of 2021 is
    # 2021-12-03, DOY 005 of 2022 is 2022-01-05.
    ("2021337000000", "17000", "01", "00101", 0, True),
    ("2021338000000", "17100", "02", "00102", 0, False),
    ("2022005000000", "17900", "01", "00201", 1, True),
    ("2022006000000", "17950", "03", "00202", 1, True),
)
# BEAM0000 is a coverage beam and BEAM0101 a full-power one in the real
# product; the adapter never assumes that — it reads the attribute this
# writer sets, which is the point of writing it.
SMOKE_BEAM_DESC = {"BEAM0000": "Coverage beam", "BEAM0101": "Full power beam"}


def _name(kind, stem):
    stamp, orbit, gran, track = stem[:4]
    lvl = {"l2a": "GEDI02_A", "l2b": "GEDI02_B", "l4a": "GEDI04_A"}[kind]
    rel = {"l2a": "02_004_02", "l2b": "02_004_01", "l4a": "02_004_01"}[kind]
    return f"{lvl}_{stamp}_O{orbit}_{gran}_T{track}_{rel}_V003"


def make_smoke_sources(root, d_lo, d_hi, seed=20260920):
    """`<root>/gedi/<granule>.h5` for the three products, and the truth rows.

    The truth is in the order the adapter yields: granule stem ascending, then
    beam name, then the file's own shot order. Every synthetic shot has its
    own second, so the store's (bin, time_s) sort cannot reorder them.
    """
    import h5py
    rng = np.random.default_rng(seed)
    base = os.path.join(root, "gedi")
    os.makedirs(base, exist_ok=True)
    day0 = d_lo
    truth = []
    for si, stem in enumerate(SMOKE_STEMS):
        month_off = stem[4]
        has_l4a = stem[5]
        # month 0 lands in `d_lo`'s month, month 1 in the month after
        day = (dt.date(day0.year + (day0.month + month_off - 1) // 12,
                       (day0.month + month_off - 1) % 12 + 1, 3 + si))
        dtime0 = (dt.datetime(day.year, day.month, day.day)
                  - dt.datetime(2018, 1, 1)).total_seconds()
        files = {k: h5py.File(os.path.join(base, _name(k, stem) + ".h5"), "w")
                 for k in (("l2a", "l2b", "l4a") if has_l4a
                           else ("l2a", "l2b"))}
        for k, f in files.items():
            f.attrs["short_name"] = {"l2a": "GEDI_L2A", "l2b": "GEDI_L2B",
                                     "l4a": "GEDI_L4A"}[k]
        for bi, beam in enumerate(SMOKE_BEAMS):
            n = SMOKE_SHOTS
            shot = np.arange(1, n + 1, dtype=np.uint64) + \
                np.uint64(1000 * (si + 1) + 100 * (bi + 1))
            delta = dtime0 + 3600.0 * (bi + 1) + np.arange(n) * 7.0
            lat = np.linspace(-20.0, 20.0, n) + bi
            lon = np.linspace(-30.0, 30.0, n) + si
            elev = np.round(rng.uniform(10.0, 400.0, n), 3)
            rh = np.zeros((n, RH_COLUMNS), np.float32)
            for p in range(RH_COLUMNS):
                rh[:, p] = np.round(0.1 * p + 0.5 * np.arange(n), 3)
            sens = np.round(rng.uniform(0.9, 0.99, n), 4)
            qual = np.ones(n, np.uint8)
            degr = np.zeros(n, np.uint8)
            # shot 0: not a quality shot -> dropped and counted
            qual[0] = 0
            # shot 1: a quality shot taken during a degraded period -> qc 1
            degr[1] = 31
            # shot 2: the ground elevation is out of bounds -> NaN, counted
            elev[2] = 99999.0
            # shot 3: no L4A partner in a granule that HAS an L4A file
            #         -> NaN biomass, counted
            # shot 4: L2B quality 0 -> NaN cover and pai, counted
            g = files["l2a"].create_group(beam)
            g.attrs["description"] = SMOKE_BEAM_DESC[beam]
            d = g.create_dataset("delta_time", data=delta.astype(np.float64))
            d.attrs["units"] = DELTA_TIME_UNITS
            g.create_dataset("lat_lowestmode", data=lat.astype(np.float64))
            g.create_dataset("lon_lowestmode", data=lon.astype(np.float64))
            g.create_dataset("elev_lowestmode", data=elev.astype(np.float32))
            g.create_dataset("rh", data=rh)
            g.create_dataset("shot_number", data=shot)
            g.create_dataset("l2a_quality_flag_rel3", data=qual)
            g.create_dataset("degrade_flag", data=degr)
            g.create_dataset("sensitivity", data=sens.astype(np.float32))
            # L2B: every shot, one of them at quality 0
            cover = np.round(rng.uniform(0.1, 0.9, n), 4)
            pai = np.round(rng.uniform(0.2, 4.0, n), 4)
            bq = np.ones(n, np.uint8)
            bq[4] = 0
            gb = files["l2b"].create_group(beam)
            gb.attrs["description"] = SMOKE_BEAM_DESC[beam]
            gb.create_dataset("shot_number", data=shot)
            gb.create_dataset("cover", data=cover.astype(np.float32))
            gb.create_dataset("pai", data=pai.astype(np.float32))
            gb.create_dataset("l2b_quality_flag_rel3", data=bq)
            # L4A: shot 3 is ABSENT from the file, shot 5 is quality 0
            agbd = np.round(rng.uniform(5.0, 400.0, n), 3)
            agbd_se = np.round(rng.uniform(1.0, 40.0, n), 3)
            aq = np.ones(n, np.uint8)
            aq[5] = 0
            if has_l4a:
                sel = np.array([i for i in range(n) if i != 3])
                ga = files["l4a"].create_group(beam)
                ga.attrs["description"] = SMOKE_BEAM_DESC[beam]
                ga.create_dataset("shot_number", data=shot[sel])
                ga.create_dataset("agbd", data=agbd[sel].astype(np.float32))
                ga.create_dataset("agbd_se",
                                  data=agbd_se[sel].astype(np.float32))
                ga.create_dataset("l4a_quality_flag_rel3", data=aq[sel])
            for i in range(n):
                if qual[i] != 1:
                    continue
                v = [float(elev[i])]
                for p in RH_PERCENTILES:
                    v.append(float(rh[i, p]))
                v.append(float(cover[i]) if bq[i] == 1 else np.nan)
                v.append(float(pai[i]) if bq[i] == 1 else np.nan)
                have4 = has_l4a and i != 3 and aq[i] == 1
                v.append(float(agbd[i]) if have4 else np.nan)
                v.append(float(agbd_se[i]) if have4 else np.nan)
                v.append(float(sens[i]))
                for c, (nm, _u, lo, hi) in zip(range(len(v)), CHANNELS):
                    if np.isfinite(v[c]) and not (lo <= v[c] <= hi):
                        v[c] = np.nan
                truth.append({"t": int(round(delta[i])) + EPOCH_2018,
                              "lat": float(lat[i]), "lon": float(lon[i]),
                              "platform": f10b.platform_hash(beam),
                              "v": v})
        for f in files.values():
            f.close()
    return truth


ADAPTER = GEDIAdapter

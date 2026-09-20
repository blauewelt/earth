"""MCD64A1 v6.1 — the day each 500 m pixel burned, monthly, worldwide
(family 1.0.tf, tier G, E-082 wave 6).

PLAIN ENGLISH. A fire leaves a scar the ground keeps for weeks: the surface
darkens, then greens back. MCD64A1 (the MODIS — Moderate Resolution Imaging
Spectroradiometer — burned-area product) watches every 500 m pixel of the land
surface from the Terra and Aqua satellites and decides, once a month, WHETHER
that pixel burned and ON WHICH DAY. That is a different measurement from the
`fire` store, which holds the active-fire detections — the moments a satellite
saw something hot. A flame is seen only if the satellite happens to pass while
it is burning and the sky is clear; a scar is still there next week. So `fire`
is the event and `burned500` is the CONSEQUENCE, and the two disagree a lot:
most burned area was never seen alight.

WHY THIS STORE IS THE REASON THE FRAMEWORK GREW A THIRD SKIP REASON. Every
tier-G store files its frames on the family's five-day bin axis. This product's
cadence is a MONTH — coarser than the axis — so a month's map is filed under
the one bin that holds the 15th of its month, and the other five bins in six
hold no frame of their own. Until E-082 wave 6 the fetch wrote a shard and an
index for every one of those empty bins: about 310 monthly frames spread over
about 1,890 bins, i.e. five sixths of the files in the store saying "nothing
here". `sharded.FRAME_NO_FRAME_IN_BIN` is the reason that skips them,
and it is deliberately NOT `absent_upstream`: a month the product should have
and does not is a hole a reader must be told about, while a bin the product was
never going to have a frame for is arithmetic.

SOURCE, VERIFIED 2026-09-20 FROM THIS SANDBOX (anonymous CMR search; the
DOWNLOAD needs an Earthdata Login, which the hosted runners have):
  https://cmr.earthdata.nasa.gov/search/granules.umm_json
      ?collection_concept_id=C2565786756-LPCLOUD&temporal=<from>,<to>
  collection MCD64A1 v061, provider LPCLOUD, `EndsAtPresentFlag`, record
  beginning 2000-11-01, temporal resolution 1 Month.
  2019-08 lists **268 granules** (`CMR-Hits: 268`), one per MODIS sinusoidal
  land tile, 0.151 MB each — about 40 MB for a whole month of the planet. A
  granule's `TemporalExtent` is the calendar month
  (2019-08-01T00:00:00Z .. 2019-08-31T23:59:59Z), its `GranuleUR` is
  `MCD64A1.A2019213.h01v11.061.2021309104831`, and its data URL is
  `https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/MCD64A1.061/
  <UR>/<UR>.hdf`. The 268 tiles fall in rows v02..v14.

THE FILES ARE HDF4 (HDF-EOS2) AND ARE READ WITH `pyhdf`. rasterio's GDAL has
no HDF4 driver — measured in this sandbox by the wave-4 MODIS CMG stores, and
why `pyhdf` (whose manylinux wheel bundles the HDF4 library) is in the
family1-build workflow's install step. This module reuses `_modis_cmg`'s
readers rather than opening a second copy of that dependency.

THE SDS NAMES ARE RESOLVED AGAINST THE FILE, NEVER TRANSCRIBED (contract rule
4). No anonymous route to a NASA file's variable list exists — an LP DAAC
granule answers an unauthenticated GET with 302 -> 401 and CMR publishes no
UMM-Var for this collection — so each quantity is resolved from a candidate
list against the granule's own inventory, a required quantity that matches
nothing is a REFUSAL carrying the file's whole SDS list, and `plan.json` keeps
one real file's complete inventory (name, rank, dims, number type, attributes)
so one runner round trip settles the format whichever way it goes.

WHAT IS STORED: C = 2, one frame per calendar month per group, float16.

  burn_doy   day of year   the day the pixel burned, 1..366; **0 means the
                           pixel WAS mapped and did not burn**, which is a
                           measurement and not a gap — the same arrangement
                           `lossyear` uses for its loss year. NaN where the
                           product could not map the pixel (its "unmapped due
                           to insufficient data" class) or where it is water.
                           The two are counted separately
                           (`pixels_unmapped`, `pixels_water`) so an
                           unretrievable region and an ocean never merge.
  qa         bit flags     the product's own per-pixel QA layer, stored as the
                           number it is (0..255) rather than interpreted. A
                           consumer that wants one bit takes it; this store
                           does not decide for it.

THE GROUPS ARE BLOCKS OF A MODIS TILE ROW, AND THE WIDTH IS ARITHMETIC. One
group per tile would be 268 groups x ~310 months x 2 files = 166,160 files,
past the Hub's 100,000-per-repository limit even with every empty bin skipped
(measured in BUILD_LOG, wave 4). One group per whole tile row is 13 groups and
8,060 files, but its frame is 2,400 x 86,400 pixels, which the probe's float64
bounds check turns into 3.3 GB — more than a hosted runner should be asked
for. `TILES_PER_GROUP` = 9 sits between them: **46 groups**, 28,520 files at
the end of the record, and a 2,400 x 21,600 frame whose float64 form is
0.83 GB — half of what `lossyear` already accepts. The three columns of that
arithmetic are in `GROUP_ARITHMETIC` so a future change is made against the
numbers rather than against a preference.

THE GRID IS THE MODIS SINUSOIDAL PROJECTION, `x = R lon cos(lat)`,
`y = R lat` on a sphere of R = 6,371,007.181 m, so no projection library is
needed. Row 0 of every group is its NORTHERNMOST row
(`row_order = "north_first"`), because that is the order the HDF file itself
stores. NO TILE-CORNER TABLE IS WRITTEN: the inverse is undefined where
|x| > R pi cos(lat), and a nine-tile-wide block at 70 N has corners exactly
there, so the store carries the formula in words rather than a clamped number
that would look like a coordinate.

DUPLICATION, NAMED RATHER THAN HIDDEN. `_modis_sin.py` (the biosphere wave's
other sinusoidal stores, `pheno500` and `lai500`) carries the same grid
constants — they agree to the digit: R 6,371,007.181 m, a tile of
1,111,950.5197 m, a 500 m pixel of 463.3127 m — plus a `check_struct` that
falsifies them against every granule's own `StructMetadata.0`, and a CMR
walker. This module does not use it because a group here is a BLOCK of nine
tiles rather than one tile, which is a different adapter shape, and because
both modules landed in the same wave. They should be consolidated once both
have, the way `_modis_cmg.py`'s duplicated Earthdata trio is already on that
list; until then this is the smaller of the two and the constants are the same
constants.

SIZE. The note's ledger row is "~10 GB" for the whole record. Burned area is
sparse — a few per cent of the land surface a year — so almost every tile is
`burn_doy = 0` and compresses hard; the probe measures the real number and
`ml/family1/probes/burned500_2019-08.json` is what replaces the estimate.
"""
import datetime as dt
import json
import os
import re
import sys
import time

import numpy as np

import build_family10_stores as f10b
from family1 import sharded as sh
from family1.adapters import _common as cm
from family1.adapters import _modis_cmg as mc

CMR_GRANULES = "https://cmr.earthdata.nasa.gov/search/granules.umm_json"
COLLECTION = "C2565786756-LPCLOUD"
SHORT_NAME = "MCD64A1"
VERSION = "061"
DOI = "10.5067/MODIS/MCD64A1.061"

# ---- the MODIS sinusoidal grid, from the product's own definition ----------
R_SPHERE = 6371007.181                 # metres, the MODIS sphere
TILE_M = 1111950.5196666666            # metres, one 10-degree tile at 0 N
TILE_PX = 2400                         # 500 m pixels across one tile
PIX_M = TILE_M / TILE_PX               # 463.31271652777775 m
X_MIN = -20015109.354                  # the grid's western edge
Y_MAX = 10007554.677                   # the grid's northern edge

# ---- the record ------------------------------------------------------------
RECORD_FIRST = (2000, 11)              # CMR: BeginningDateTime 2000-11-01
# The product runs to the present (`EndsAtPresentFlag`), so the last month is
# MEASURED at index time and never written here.

# ---- how a month lands on the five-day axis --------------------------------
ANCHOR_DAY = 15
FRAMES_PER_BIN = 1

# ---- the groups ------------------------------------------------------------
TILES_PER_GROUP = 9
GROUP_ARITHMETIC = {
    "measured": "CMR, 2026-09-20: 268 MCD64A1 v061 tiles in 2019-08, rows "
                "v02..v14; about 310 monthly frames in the record so far",
    "files_rule": "groups x months x 2 (a shard and its index)",
    "options": [
        {"tiles_per_group": 1, "groups": 268, "files": 166160,
         "frame_px": 5760000, "probe_float64_bytes": 92160000,
         "why_not": "past the Hub's 100,000-file-per-repository limit"},
        {"tiles_per_group": 9, "groups": 46, "files": 28520,
         "frame_px": 51840000, "probe_float64_bytes": 829440000,
         "why_not": None},
        {"tiles_per_group": 36, "groups": 13, "files": 8060,
         "frame_px": 207360000, "probe_float64_bytes": 3317760000,
         "why_not": "the probe's float64 bounds check is 3.3 GB, more than a "
                    "hosted runner should be asked for"},
    ],
}

# ---- the SDS names, RESOLVED against the file ------------------------------
# Each entry is (channel, [candidate SDS names, most likely first], required).
# Contract rule 4: the names are a HYPOTHESIS the granule verifies. A required
# quantity that matches nothing is a refusal carrying the file's own list.
SDS_CANDIDATES = (
    ("burn_doy", ("Burn Date", "burn_date", "BurnDate", "Burndate"), True),
    ("qa", ("QA", "Burn Date QA", "qa", "First Day"), True),
)

# The product's own classes for the burn-date layer, from the MCD64A1 v6.1
# user guide: 0 unburned, 1..366 the burn day of year, -1 unmapped for lack of
# data, -2 water. They are DECLARED here and CHECKED against the granule: a
# value that is none of them is counted under `burn_date_unknown_class` and
# becomes NaN, and a file whose `valid_range` contradicts the declaration is a
# refusal rather than a silent reinterpretation.
BURN_UNBURNED = 0
BURN_UNMAPPED = -1
BURN_WATER = -2
BURN_DOY_MAX = 366

SMOKE_PX_ENV = "BURNED500_SMOKE_PX"
SMOKE_TILES_ENV = "BURNED500_TILES"
SMOKE_PX = 48                          # 48 = 256/5 rounded down; two tiles of
                                       # it still exercise a tile boundary


class FormatError(ValueError):
    """A granule that is not what the adapter declares."""


# =============================================================== the grid ===
def tile_of(name):
    """`h01v11` -> (1, 11); anything else is a refusal."""
    m = re.fullmatch(r"h(\d{2})v(\d{2})", str(name))
    if not m:
        raise FormatError(f"{name!r} is not a MODIS tile name (h<NN>v<NN>)")
    h, v = int(m.group(1)), int(m.group(2))
    if not (0 <= h <= 35 and 0 <= v <= 17):
        raise FormatError(f"{name}: outside the 36 x 18 MODIS tile grid")
    return h, v


def group_name(v, h0):
    """The group a tile row `v` and a block starting at column `h0` make."""
    return f"v{int(v):02d}h{int(h0):02d}"


def group_parts(group):
    """`v11h00` -> (11, 0)."""
    m = re.fullmatch(r"v(\d{2})h(\d{2})", str(group))
    if not m:
        raise FormatError(f"{group!r} is not a burned500 group name")
    return int(m.group(1)), int(m.group(2))


def group_of_tile(h, v, per_group=TILES_PER_GROUP):
    return group_name(v, (int(h) // int(per_group)) * int(per_group))


def grid_of(group, px=TILE_PX, per_group=TILES_PER_GROUP):
    """The group's grid declaration — H, W and the affine rule, no arrays."""
    v, h0 = group_parts(group)
    step = TILE_M / float(px)
    return {
        "H": int(px), "W": int(px) * int(per_group),
        "x0": X_MIN + h0 * TILE_M, "dx": step,
        "y0": Y_MAX - v * TILE_M, "dy": -step,
        "row_order": "north_first",
        "crs": ("MODIS sinusoidal on a sphere of radius 6371007.181 m: "
                "x = R * lon_rad * cos(lat_rad), y = R * lat_rad "
                "(PROJ: +proj=sinu +R=6371007.181 +units=m)"),
        "cell_rule": ("the centre of cell (row, col) is at "
                      "(x0 + (col + 0.5) * dx, y0 + (row + 0.5) * dy)"),
        "inverse": ("lat = degrees(y / R); lon = degrees(x / (R * cos(lat))). "
                    "It is UNDEFINED where |x| > R * pi * cos(lat) — a point "
                    "off the projected globe, which the corners of a "
                    "rectangular tile block near the grid's edge really are — "
                    "so no tile-corner table is stored: a clamped number "
                    "would look like a coordinate and be none"),
        "modis_tiles": [f"h{h0 + k:02d}v{v:02d}" for k in range(per_group)],
        "tile_px": int(px),
    }


def sinu_to_latlon(x, y):
    """The MODIS sinusoidal inverse — two lines, no projection library.

    NaN where the point is OFF THE PROJECTED GLOBE (|x| > R pi cos(lat)),
    which is not an edge case here: a group is nine tiles wide, so at 70 N its
    rectangle reaches far past the meridian the projection has room for.
    Clamping to +/-180 there would write a number that looks like a longitude
    and is not one, which is why this adapter is NOT wired as `grid_latlon`
    and the store keeps the formula in words instead.
    """
    lat = np.degrees(np.asarray(y, np.float64) / R_SPHERE)
    c = np.cos(np.radians(lat))
    with np.errstate(divide="ignore", invalid="ignore"):
        lon = np.degrees(np.asarray(x, np.float64) / (R_SPHERE * c))
    return lat, np.where(np.abs(lon) <= 180.0, lon, np.nan)


# ============================================================ month <-> bin ==
def month_bin(y, m):
    """The five-day bin that holds the 15th of the month `y`-`m`."""
    t = f10b.seconds_since_epoch(dt.date(int(y), int(m), ANCHOR_DAY))
    return int(t) // sh.BIN_SECONDS


def bin_month(b):
    """The month whose anchor day falls in bin `b`, or None.

    A bin is five consecutive days and the shortest month is 28, so at most
    one month's 15th can fall inside one bin — the map is a function.
    """
    d0 = sh.frame_day(int(b), 0, sh.BIN_SECONDS)
    for k in range(sh.BIN_SECONDS // 86400):
        d = d0 + dt.timedelta(days=k)
        if d.day == ANCHOR_DAY:
            return d.year, d.month
    return None


def month_span(y, m):
    """(first second, last second) of the calendar month, since the epoch."""
    return f10b.month_bounds_s(int(y), int(m))


def classify_bin(b, first=RECORD_FIRST, last=None):
    """-> (year, month) for a bin that carries a frame, else the REASON.

    ONE definition, used by the adapter and by the smoke's truth, so the two
    cannot drift. The record bounds are compared by MONTH where the bin
    anchors one and by the bin's own DAYS where it does not — so the bins of
    the record's last month that fall after its 15th read `no_frame_in_bin`
    (they are inside the record and simply anchor nothing) rather than
    `after_record`, and a bin in a later month reads `after_record`.
    """
    b = int(b)
    ym = bin_month(b)
    if ym is not None:
        if ym < tuple(first):
            return "before_record"
        if last is not None and ym > tuple(last):
            return "after_record"
        return ym
    d0 = sh.frame_day(b, 0, sh.BIN_SECONDS)
    d1 = d0 + dt.timedelta(days=sh.BIN_SECONDS // 86400 - 1)
    if (d1.year, d1.month) < tuple(first):
        return "before_record"
    if last is not None and (d0.year, d0.month) > tuple(last):
        return "after_record"
    return sh.FRAME_NO_FRAME_IN_BIN


# ============================================================== the listing ==
def cmr_month(year, month, attempts=4):
    """Every granule of one calendar month -> ({tile: rec}, counts).

    The listing is the product's own: which tiles exist in a month is a
    measurement, never a constant. `rec` carries the granule id, the URL and
    the size CMR reports.
    """
    lo = dt.datetime(int(year), int(month), 1)
    hi = (dt.datetime(int(year) + (int(month) == 12),
                      int(month) % 12 + 1, 1) - dt.timedelta(seconds=1))
    url = (f"{CMR_GRANULES}?collection_concept_id={COLLECTION}"
           f"&temporal={lo:%Y-%m-%dT%H:%M:%SZ},{hi:%Y-%m-%dT%H:%M:%SZ}"
           f"&page_size=2000")
    raw, why = cm.get_bytes(url, attempts=attempts)
    if raw is None:
        raise IOError(f"CMR ({SHORT_NAME} {year}-{month:02d}): {why}")
    return parse_cmr(raw, year, month)


def parse_cmr(raw, year, month):
    """A `granules.umm_json` body -> ({tile: rec}, counts). Refuses junk."""
    try:
        d = json.loads(raw)
    except ValueError as e:
        raise FormatError(f"CMR {year}-{month:02d}: not JSON ({e})") from None
    items = d.get("items")
    if items is None:
        raise FormatError(f"CMR {year}-{month:02d}: no `items` key "
                          f"({sorted(d)[:8]})")
    out, counts = {}, {"cmr_bytes": len(raw), "granules_listed": len(items)}
    for it in items:
        umm = it.get("umm") or {}
        gid = umm.get("GranuleUR") or ""
        m = re.search(r"\.h(\d{2})v(\d{2})\.", gid)
        if not m:
            counts["granules_without_tile"] = \
                counts.get("granules_without_tile", 0) + 1
            continue
        tile = f"h{m.group(1)}v{m.group(2)}"
        url = None
        for r in umm.get("RelatedUrls") or []:
            u = str(r.get("URL") or "")
            if r.get("Type") == "GET DATA" and u.startswith("http") \
                    and u.endswith(".hdf"):
                url = u
                break
        if url is None:
            counts["granules_without_url"] = \
                counts.get("granules_without_url", 0) + 1
            continue
        made = gid.rsplit(".", 1)[-1]
        old = out.get(tile)
        if old is not None:
            # LP DAAC republishes a tile under a later production stamp; the
            # newest wins and the older is counted rather than dropped in
            # silence.
            counts["granules_superseded"] = \
                counts.get("granules_superseded", 0) + 1
            if made <= old["made"]:
                continue
        out[tile] = {"id": gid, "url": url, "made": made,
                     "bytes": _size_bytes(umm)}
    if not out:
        raise FormatError(
            f"CMR {year}-{month:02d}: {len(items)} item(s) and NO tile — a "
            f"broken query or a changed granule naming, not a measurement "
            f"(the 2026-09-14 rule)")
    return out, counts


def _size_bytes(umm):
    for a in ((umm.get("DataGranule") or {})
              .get("ArchiveAndDistributionInformation") or []):
        try:
            mb = float(a.get("Size"))
        except (TypeError, ValueError):
            continue
        if str(a.get("SizeUnit") or "MB").upper() == "MB":
            return int(mb * 1e6)
    return None


# ================================================================== the SDS ==
def resolve_sds(have):
    """{channel: SDS name} from the file's OWN inventory, or a refusal."""
    out = {}
    for ch, cands, required in SDS_CANDIDATES:
        for c in cands:
            if c in have:
                out[ch] = c
                break
        else:
            if required:
                raise FormatError(
                    f"{SHORT_NAME}: no SDS for the {ch!r} channel — tried "
                    f"{list(cands)}; the granule carries "
                    f"{sorted(have)}. The names are a hypothesis this "
                    f"resolution verifies (ADAPTER_CONTRACT rule 4); add the "
                    f"real name to SDS_CANDIDATES rather than renaming a "
                    f"channel.")
    return out


def read_tile(path, px=TILE_PX):
    """One granule -> (burn_doy, qa) as float16 [px, px], plus counts."""
    names = mc.sds_names(path)
    res = resolve_sds(set(names))
    arrays, attrs = mc.read_sds(path, [res["burn_doy"], res["qa"]],
                                w=int(px), h=int(px))
    burn = np.asarray(arrays[res["burn_doy"]])
    qa = np.asarray(arrays[res["qa"]])
    counts = {"granules_read": 1}

    vr = (attrs.get(res["burn_doy"]) or {}).get("valid_range")
    if vr is not None:
        try:
            lo, hi = float(vr[0]), float(vr[1])
        except (TypeError, ValueError, IndexError):
            lo = hi = None
        if lo is not None and (lo > BURN_WATER or hi < BURN_DOY_MAX):
            raise FormatError(
                f"{os.path.basename(path)}: the burn-date SDS declares "
                f"valid_range {list(vr)}, which cannot hold the classes this "
                f"adapter stores ({BURN_WATER} water .. {BURN_DOY_MAX} day of "
                f"year). A product whose value scheme changed is refused, "
                f"never reinterpreted.")

    b = burn.astype(np.int32)
    out = np.full((int(px), int(px), 2), np.nan, np.float16)
    burned = (b >= 1) & (b <= BURN_DOY_MAX)
    unburned = b == BURN_UNBURNED
    unmapped = b == BURN_UNMAPPED
    water = b == BURN_WATER
    known = burned | unburned
    odd = ~(known | unmapped | water)
    n_odd = int(odd.sum())
    if n_odd:
        counts["burn_date_unknown_class"] = n_odd
    out[:, :, 0] = np.where(known, b.astype(np.float16), np.float16("nan"))
    # qa follows the burn layer's mask: a QA byte for a pixel the product did
    # not map describes nothing.
    out[:, :, 1] = np.where(known, np.asarray(qa, np.float64)
                            .astype(np.float16), np.float16("nan"))
    counts["pixels_burned"] = int(burned.sum())
    counts["pixels_unburned"] = int(unburned.sum())
    counts["pixels_unmapped"] = int(unmapped.sum())
    counts["pixels_water"] = int(water.sum())
    return out, counts, {"sds": res, "attributes": attrs}


# ============================================================= the adapter ===
class Burned500Adapter(sh.GridAdapter):
    store = "burned500"
    title = ("MODIS MCD64A1 v6.1 burned-area date, 500 m, monthly, MODIS "
             "sinusoidal tiles in 46 nine-tile row blocks (annual-style "
             "target on the five-day axis)")
    family = "1tf"
    distribution = "public"
    licence = {
        "name": "NASA open data (no restriction on use or redistribution)",
        "redistribution": "attribution",
        "redistribution_confirmed": True,
        "derived_works": "free",
        "attribution": ("Giglio, L., C. Justice, L. Boschetti and D. Roy "
                        "(2021). MODIS/Terra+Aqua Burned Area Monthly L3 "
                        "Global 500m SIN Grid V061. NASA EOSDIS Land "
                        "Processes Distributed Active Archive Centre. "
                        f"https://doi.org/{DOI}"),
        "terms": ("NASA Earth science data are open: 'free and open ... "
                  "without restriction'. An Earthdata Login account is an "
                  "access control on the DOWNLOAD, not a licence condition "
                  "on the data."),
    }
    time_dtype = "int32"
    credentials = ("EARTHDATA_USERNAME", "EARTHDATA_PASSWORD")
    channels = (("burn_doy",
                 "day of year (0 = mapped and did not burn)",
                 0.0, float(BURN_DOY_MAX)),
                ("qa", "bit flags", 0.0, 255.0))
    # 463.31 m / 27.83 km = 1/60.07 of the family's footprint unit
    log2_fp = float(np.log2(PIX_M / 1000.0 / 27.83))             # -5.908
    log2_dt = float(np.log2(365.25 / 12.0 / 5.0))                # +2.606
    per_year = True
    first_year = RECORD_FIRST[0]
    frames_per_bin = FRAMES_PER_BIN
    frame_seconds = sh.BIN_SECONDS
    dtype = "float16"
    tile = sh.TILE
    zstd_level = 15
    note_estimate = {"bytes": 10e9,
                     "what": "family1tf.tex ledger: 'the date each 500 m "
                             "pixel burned', MCD64A1 v6.1, ~10 GB"}
    qc_policy = (
        "There is no per-row qc in a tier-G store; the product's own QA layer "
        "is a CHANNEL instead, stored as the number it is (0..255) rather "
        "than interpreted, so a consumer that wants one bit takes it. The "
        "burn-date layer's classes are the product's own: 0 the pixel was "
        "MAPPED AND DID NOT BURN (a measurement, stored as 0), 1..366 the "
        "day of year it burned, -1 unmapped for lack of data and -2 water -- "
        "both NaN and counted separately (`pixels_unmapped`, `pixels_water`) "
        "so an unretrievable region and an ocean never merge. A value that is "
        "none of those is NaN and counted (`burn_date_unknown_class`), and a "
        "granule whose `valid_range` cannot hold the declared classes is a "
        "REFUSAL rather than a reinterpretation. The QA byte follows the burn "
        "layer's mask: a QA value for a pixel the product did not map "
        "describes nothing. A value outside a channel's bounds becomes NaN "
        "and is counted (`out_of_bounds`), never clipped.")
    sources = (f"{CMR_GRANULES}?collection_concept_id={COLLECTION}",
               "https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/"
               "MCD64A1.061/<granule>/<granule>.hdf",
               f"https://doi.org/{DOI}")
    verified = (
        "2026-09-20 from the sandbox, anonymously: CMR lists exactly one "
        "MCD64A1 collection, C2565786756-LPCLOUD v061, whose TemporalExtent "
        "begins 2000-11-01 with EndsAtPresentFlag and a TemporalResolution of "
        "1 Month; the 2019-08 granule search answered CMR-Hits 268 and "
        "returned 268 items, one per MODIS sinusoidal land tile, in rows "
        "v02..v14, each 0.151 MB, each with a GET DATA URL under "
        "data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/MCD64A1.061/ "
        "and a TemporalExtent of the whole calendar month "
        "(2019-08-01T00:00:00Z .. 2019-08-31T23:59:59Z); a granule's "
        "AdditionalAttributes carry HORIZONTALTILENUMBER and "
        "VERTICALTILENUMBER. The DOWNLOAD was not attempted from here: an LP "
        "DAAC granule answers an unauthenticated GET with 302 -> 401, which "
        "is why the SDS names are resolved against the granule on a runner "
        "and plan.json keeps one real file's whole inventory.")
    notes = (
        "A MONTHLY product on a five-day axis. Each month's map is filed "
        f"under the bin holding the {ANCHOR_DAY}th of its month, and the "
        "other five bins in six carry no frame at all -- "
        "`sharded.FRAME_NO_FRAME_IN_BIN`, which is what stops the "
        "fetch writing a shard and an index for each of them (about 1,580 "
        "empty bins against about 310 real frames). That reason is NOT "
        "`absent_upstream`: a month the product should have and does not is "
        "still a hole and is still counted as one. Groups are blocks of "
        f"{TILES_PER_GROUP} tiles in one MODIS row -- 46 of them, 28,520 "
        "files at the end of the record -- because one group per tile is "
        "166,160 files, past the Hub's 100,000-per-repository limit, and one "
        "group per whole row gives the probe a 3.3 GB float64 array; see "
        "GROUP_ARITHMETIC. Row 0 of a group is its NORTHERNMOST row, the "
        "order the HDF file stores. The files are HDF4 and need `pyhdf`; "
        "rasterio's GDAL has no HDF4 driver.")
    smoke_window = ("2019-08-11", "2019-09-20")
    smoke_probe_month = "2019-08"

    WORKERS = 4

    def __init__(self):
        self.px = int(os.environ.get(SMOKE_PX_ENV) or TILE_PX)
        self.per_group = TILES_PER_GROUP
        want = [t.strip() for t in
                (os.environ.get(SMOKE_TILES_ENV) or "").split(",")
                if t.strip()]
        for t in want:
            tile_of(t)                        # refuses a bad name
        self.tiles = tuple(want) or TILES
        self._months = {}
        self._record_last = None
        self._session = None
        self._inventory = None
        if want:
            self.notes = (
                f"{self.notes}\nRESTRICTED BUILD: {SMOKE_TILES_ENV} limited "
                f"this store to {len(want)} of the product's {len(TILES)} "
                f"tiles ({', '.join(want[:8])}"
                f"{' ...' if len(want) > 8 else ''}). It is NOT the whole "
                f"product.")
        if self.px != TILE_PX:
            self.notes = (f"{self.notes}\nSMOKE GRID: {SMOKE_PX_ENV} set each "
                          f"MODIS tile to {self.px} x {self.px} instead of "
                          f"{TILE_PX} x {TILE_PX}. This is a synthetic store.")

    # ------------------------------------------------------------ the grid --
    def groups(self):
        out = []
        for t in self.tiles:
            h, v = tile_of(t)
            g = group_of_tile(h, v, self.per_group)
            if g not in out:
                out.append(g)
        return sorted(out)

    def group_grids(self):
        return {g: grid_of(g, self.px, self.per_group) for g in self.groups()}

    # NO `grid_latlon`, deliberately. `sharded.make_spec` writes a
    # `tile_corners` table when an adapter supplies an inverse projection, and
    # for a nine-tile-wide sinusoidal block that table would be mostly
    # fabricated: the rectangle's corners at 70 N sit far outside the
    # longitudes the projection has room for, where the inverse is undefined.
    # The formula is in the grid dict instead, in words, which is the same
    # decision `_modis_sin.py` made for the other sinusoidal stores.

    def specs(self):
        out = super().specs()
        for spec in out.values():
            spec["monthly_target"] = {
                "rule": (f"a monthly map is filed under the five-day bin that "
                         f"holds the {ANCHOR_DAY}th of its month; every other "
                         f"bin carries no frame and is skipped "
                         f"(no_frame_in_bin)"),
                "anchor_day": ANCHOR_DAY,
                "frames_per_bin": FRAMES_PER_BIN,
                "record_first_month": "%04d-%02d" % RECORD_FIRST,
                "burn_doy_values": {
                    "0": "mapped land, did not burn in this month",
                    "1..366": "the day of year the pixel burned",
                    "NaN": "not mapped (insufficient data) or water",
                },
            }
        return out

    def record_frames(self, ctx, group):
        """How many monthly frames the record holds — for the extrapolation."""
        last = self._record_last or self.record_last(ctx)
        if last is None:
            return None
        y0, m0 = RECORD_FIRST
        y1, m1 = last
        return (y1 - y0) * 12 + (m1 - m0) + 1

    # ------------------------------------------------------------- listing --
    def record_last(self, ctx):
        """The newest month the product publishes — MEASURED, never typed."""
        if self._record_last is not None:
            return self._record_last
        if ctx.source_dir:
            p = os.path.join(ctx.source_dir, self.store, "record.json")
            if not os.path.exists(p):
                sys.exit(f"REFUSING {self.store}: no {p} — the smoke's "
                         f"synthetic record bounds are missing")
            with open(p, "rb") as fh:
                raw = fh.read()
            ctx.count_bytes(len(raw))
            d = json.loads(raw)
            self._record_last = (int(d["last"][0]), int(d["last"][1]))
            return self._record_last
        url = (f"{CMR_GRANULES}?collection_concept_id={COLLECTION}"
               f"&sort_key%5B%5D=-start_date&page_size=1")
        raw, why = cm.get_bytes(url, attempts=ctx.a.attempts)
        if raw is None:
            raise IOError(f"CMR ({SHORT_NAME}, newest granule): {why}")
        ctx.count_bytes(len(raw))
        items = (json.loads(raw).get("items") or [])
        if not items:
            raise FormatError(
                f"{self.store}: CMR returned NO newest granule for "
                f"{COLLECTION} — a broken query, not an empty archive")
        beg = (((items[0]["umm"].get("TemporalExtent") or {})
                .get("RangeDateTime") or {}).get("BeginningDateTime") or "")
        m = re.match(r"(\d{4})-(\d{2})", str(beg))
        if not m:
            raise FormatError(f"{self.store}: the newest granule's "
                              f"BeginningDateTime is {beg!r}")
        self._record_last = (int(m.group(1)), int(m.group(2)))
        return self._record_last

    def month_listing(self, ctx, y, m):
        """{tile: rec} for one month, listed once and cached."""
        key = (int(y), int(m))
        if key in self._months:
            return self._months[key], {}
        if ctx.source_dir:
            p = os.path.join(ctx.source_dir, self.store,
                             f"cmr_{y:04d}-{m:02d}.json")
            if not os.path.exists(p):
                sys.exit(f"REFUSING {self.store}: no {p} — the smoke's "
                         f"synthetic listing for {y:04d}-{m:02d} is missing")
            with open(p, "rb") as fh:
                raw = fh.read()
            ctx.count_bytes(len(raw))
            got, counts = parse_cmr(raw, y, m)
        else:
            got, counts = cmr_month(y, m, attempts=ctx.a.attempts)
            ctx.count_bytes(counts.get("cmr_bytes", 0))
        self._months[key] = got
        return got, counts

    def index(self, ctx):
        last = self.record_last(ctx)
        y, m = ctx.d_lo.year, ctx.d_lo.month
        if (y, m) < RECORD_FIRST:
            y, m = RECORD_FIRST
        if (y, m) > last:
            y, m = last
        listing, counts = self.month_listing(ctx, y, m)
        live = set(listing)
        known = set(self.tiles)
        out = {
            "dataset": f"{SHORT_NAME} v{VERSION} monthly burned-area date, "
                       f"500 m, MODIS sinusoidal (HDF-EOS2)",
            "url": f"{CMR_GRANULES}?collection_concept_id={COLLECTION}",
            "collection": COLLECTION, "short_name": SHORT_NAME,
            "version": VERSION, "doi": DOI,
            "record_first_month": "%04d-%02d" % RECORD_FIRST,
            "record_last_month": "%04d-%02d" % last,
            "record_months": self.record_frames(ctx, None),
            "first_month_listed": f"{y:04d}-{m:02d}",
            "tiles_listed": sorted(live),
            "tiles_declared": len(known),
            "tiles_listed_not_declared": sorted(live - known),
            "tiles_declared_not_listed": sorted(known - live),
            "groups": self.groups(),
            "tiles_per_group": self.per_group,
            "group_arithmetic": GROUP_ARITHMETIC,
            "anchor_day": ANCHOR_DAY,
            "frames_per_bin": FRAMES_PER_BIN,
            "frame_seconds": self.frame_seconds,
            "bin_of_first_month": month_bin(*RECORD_FIRST),
            "bin_of_last_month": month_bin(*last),
            "counts": counts,
            "sds_candidates": {ch: list(c) for ch, c, _r in SDS_CANDIDATES},
        }
        if not live:
            raise FormatError(
                f"{self.store}: CMR lists ZERO tiles for {y:04d}-{m:02d} — a "
                f"broken query or a moved collection, not a measurement")
        if self._inventory is not None:
            out["granule_inventory"] = self._inventory
        return out

    # ----------------------------------------------------------- the fetch --
    def session(self):
        if self._session is None:
            self._session = mc.earthdata_session()
        return self._session

    def fetch_preflight(self, ctx):
        if ctx.source_dir:
            return None
        if not mc.netrc_has_urs():
            print(f"::warning::{self.store}: no netrc entry for "
                  f"{mc.URS_HOST} was found; the download will fall back to "
                  f"whatever credentials the environment provides and may be "
                  f"refused. family1-build.yml writes that file on hosted "
                  f"runners.")
        return None

    def classify(self, ctx, b):
        """-> (year, month) for a bin that carries a frame, else the reason."""
        return classify_bin(b, RECORD_FIRST,
                            self._record_last or self.record_last(ctx))

    def granule_path(self, ctx, rec, tmpdir):
        """The granule on disk -> (path, None) or (None, why)."""
        if ctx.source_dir:
            p = os.path.join(ctx.source_dir, self.store, "granules",
                             os.path.basename(rec["url"]))
            if not os.path.exists(p):
                return None, f"{p} does not exist"
            ctx.count_bytes(os.path.getsize(p))
            return p, None
        p = os.path.join(tmpdir, os.path.basename(rec["url"]))
        n, why = mc.download(self.session(), rec["url"], p,
                             attempts=ctx.a.attempts)
        if n is None:
            return None, why
        ctx.count_bytes(int(n))
        return p, None

    def fetch_frames(self, ctx, wanted):
        import shutil
        import tempfile
        px, pg = self.px, self.per_group
        # Split the ask into the bins that carry nothing (no I/O at all) and
        # the real ones, grouped by MONTH so one listing answers a month.
        absent, real = [], {}
        for (g, b, f) in wanted:
            why = self.classify(ctx, b)
            if isinstance(why, str):
                absent.append((g, b, f, why))
            else:
                real.setdefault(why, []).append((g, b, f))
        for (g, b, f, why) in absent:
            yield g, b, f, None, {"frame_missing": why}

        t0 = time.time()
        for (y, m) in sorted(real):
            try:
                listing, lc = self.month_listing(ctx, y, m)
            except (IOError, FormatError) as e:
                ctx.note_absent(f"{y:04d}-{m:02d}", str(e))
                continue
            counts_month = dict(lc)
            tmpdir = None if ctx.source_dir else tempfile.mkdtemp(
                prefix="mcd64_", dir=ctx.scratch)
            try:
                for (g, b, f) in sorted(real[(y, m)]):
                    v, h0 = group_parts(g)
                    names = [f"h{h0 + k:02d}v{v:02d}" for k in range(pg)]
                    have = [t for t in names
                            if t in listing and t in set(self.tiles)]
                    if not have:
                        # The whole block is outside the product's land mask
                        # in this month — a real absence of the frame, and it
                        # is named as one rather than stored as empty pixels.
                        yield g, b, f, None, {
                            "frame_missing": "absent_upstream",
                            "frames_absent_no_tile_in_block": 1}
                        continue
                    arr = np.full((px, px * pg, 2), np.nan, np.float16)
                    c = dict(counts_month)
                    counts_month = {}          # the listing counts, once
                    lost = None
                    for t in have:
                        k = tile_of(t)[0] - h0
                        p, why = self.granule_path(ctx, listing[t], tmpdir)
                        if p is None:
                            lost = f"{listing[t]['url']}: {why}"
                            break
                        try:
                            a, tc, inv = read_tile(p, px)
                        except (FormatError, IOError) as e:
                            lost = f"{listing[t]['url']}: {e}"
                            break
                        finally:
                            if not ctx.source_dir and os.path.exists(p):
                                os.remove(p)
                        if self._inventory is None:
                            self._inventory = {
                                "granule": listing[t]["id"],
                                "sds_resolved": inv["sds"],
                                "attributes": {
                                    k2: {kk: (list(vv) if isinstance(
                                        vv, (list, tuple)) else vv)
                                        for kk, vv in (v2 or {}).items()}
                                    for k2, v2 in
                                    (inv["attributes"] or {}).items()}}
                        arr[:, k * px:(k + 1) * px, :] = a
                        f10b._merge_counts(c, tc)
                    if lost is not None:
                        ctx.note_absent(f"{g} {y:04d}-{m:02d}", lost)
                        continue
                    oob = mc.mask_bounds_low_memory(self, arr)
                    if oob:
                        c["out_of_bounds"] = oob
                    c["tiles_in_frame"] = len(have)
                    c["months_read"] = 1 if "granules_read" in c else 0
                    yield g, b, f, arr, c
                    ctx.prog.item(f"{self.store} {y:04d}-{m:02d} {g}", None,
                                  {"elapsed_s": round(time.time() - t0, 1)})
            finally:
                if tmpdir:
                    shutil.rmtree(tmpdir, ignore_errors=True)

    # --------------------------------------------------------------- smoke --
    def smoke_sources(self, root, d_lo, d_hi):
        truth = make_smoke_sources(root, d_lo, d_hi)
        self.__init__()
        return truth


# =================================================================== tiles ==
# THE PRODUCT'S TILE SET, MEASURED FROM CMR ON 2026-09-20 (the 268 granules of
# 2019-08). It is a constant so that every lane of a build declares the same
# groups — `stage_assemble_grid` refuses parts written for a different grid
# declaration, so a group set derived from whatever one lane happened to list
# would make the lanes unassemblable. `index` compares it with the live
# listing and REPORTS both differences rather than trusting either.
TILES = (
    "h00v08", "h00v09", "h00v10", "h01v07", "h01v08", "h01v09",
    "h01v10", "h01v11", "h02v06", "h02v08", "h02v09", "h02v10",
    "h02v11", "h03v06", "h03v07", "h03v09", "h03v10", "h03v11",
    "h04v09", "h04v10", "h04v11", "h05v10", "h05v11", "h05v13",
    "h06v03", "h06v11", "h07v03", "h07v05", "h07v06", "h07v07",
    "h08v03", "h08v04", "h08v05", "h08v06", "h08v07", "h08v08",
    "h08v09", "h08v11", "h09v02", "h09v03", "h09v04", "h09v05",
    "h09v06", "h09v07", "h09v08", "h09v09", "h10v02", "h10v03",
    "h10v04", "h10v05", "h10v06", "h10v07", "h10v08", "h10v09",
    "h10v10", "h10v11", "h11v02", "h11v03", "h11v04", "h11v05",
    "h11v06", "h11v07", "h11v08", "h11v09", "h11v10", "h11v11",
    "h11v12", "h12v02", "h12v03", "h12v04", "h12v05", "h12v07",
    "h12v08", "h12v09", "h12v10", "h12v11", "h12v12", "h12v13",
    "h13v02", "h13v03", "h13v04", "h13v08", "h13v09", "h13v10",
    "h13v11", "h13v12", "h13v13", "h13v14", "h14v02", "h14v03",
    "h14v04", "h14v09", "h14v10", "h14v11", "h14v14", "h15v02",
    "h15v03", "h15v05", "h15v07", "h15v11", "h16v02", "h16v05",
    "h16v06", "h16v07", "h16v08", "h16v09", "h16v12", "h17v02",
    "h17v03", "h17v04", "h17v05", "h17v06", "h17v07", "h17v08",
    "h17v10", "h17v12", "h17v13", "h18v02", "h18v03", "h18v04",
    "h18v05", "h18v06", "h18v07", "h18v08", "h18v09", "h19v02",
    "h19v03", "h19v04", "h19v05", "h19v06", "h19v07", "h19v08",
    "h19v09", "h19v10", "h19v11", "h19v12", "h20v02", "h20v03",
    "h20v04", "h20v05", "h20v06", "h20v07", "h20v08", "h20v09",
    "h20v10", "h20v11", "h20v12", "h20v13", "h21v02", "h21v03",
    "h21v04", "h21v05", "h21v06", "h21v07", "h21v08", "h21v09",
    "h21v10", "h21v11", "h21v13", "h22v02", "h22v03", "h22v04",
    "h22v05", "h22v06", "h22v07", "h22v08", "h22v09", "h22v10",
    "h22v11", "h22v13", "h23v02", "h23v03", "h23v04", "h23v05",
    "h23v06", "h23v07", "h23v08", "h23v09", "h23v10", "h23v11",
    "h24v02", "h24v03", "h24v04", "h24v05", "h24v06", "h24v07",
    "h24v12", "h25v02", "h25v03", "h25v04", "h25v05", "h25v06",
    "h25v07", "h25v08", "h25v09", "h26v02", "h26v03", "h26v04",
    "h26v05", "h26v06", "h26v07", "h26v08", "h27v03", "h27v04",
    "h27v05", "h27v06", "h27v07", "h27v08", "h27v09", "h27v10",
    "h27v11", "h27v12", "h28v03", "h28v04", "h28v05", "h28v06",
    "h28v07", "h28v08", "h28v09", "h28v10", "h28v11", "h28v12",
    "h28v13", "h29v03", "h29v05", "h29v06", "h29v07", "h29v08",
    "h29v09", "h29v10", "h29v11", "h29v12", "h29v13", "h30v05",
    "h30v06", "h30v07", "h30v08", "h30v09", "h30v10", "h30v11",
    "h30v12", "h30v13", "h31v06", "h31v07", "h31v08", "h31v09",
    "h31v10", "h31v11", "h31v12", "h31v13", "h32v07", "h32v08",
    "h32v09", "h32v10", "h32v11", "h32v12", "h33v07", "h33v08",
    "h33v09", "h33v10", "h33v11", "h34v07", "h34v08", "h34v09",
    "h34v10", "h35v08", "h35v09", "h35v10",
)


# =================================================================== smoke ==
SMOKE_TILES = ("h19v08", "h20v08")          # one group, two adjacent tiles
SMOKE_ABSENT_TILE = "h21v08"                # listed nowhere: the empty half
SMOKE_RECORD_LAST = (2019, 8)


def smoke_fields(tile, px, month):
    """One synthetic granule's two SDS, in the product's own dtypes."""
    h, v = tile_of(tile)
    burn = np.zeros((px, px), np.int16)          # mapped land, did not burn
    qa = np.ones((px, px), np.uint8)
    # a burn scar whose day of year is deterministic per tile and month
    doy = 200 + h + 2 * month
    burn[2:6, 3:9] = doy
    qa[2:6, 3:9] = 3
    # the two non-measurement classes, one block each
    burn[px - 4:, :2] = BURN_UNMAPPED
    burn[:2, px - 4:] = BURN_WATER
    # and one value in NO declared class at all, so the counter has something
    burn[0, 0] = 999
    return {"Burn Date": burn, "QA": qa}


def make_smoke_sources(root, d_lo, d_hi):
    """The archive in its real shape: a CMR body per month plus real HDF4.

    Returns the truth: {(group, bin, frame): (float16 [px, px*pg, 2] or None,
    reason or None)} for every frame of every bin overlapping the window.
    """
    os.environ[SMOKE_PX_ENV] = str(SMOKE_PX)
    os.environ[SMOKE_TILES_ENV] = ",".join(SMOKE_TILES)
    px = SMOKE_PX
    base = os.path.join(root, "burned500")
    gdir = os.path.join(base, "granules")
    os.makedirs(gdir, exist_ok=True)
    with open(os.path.join(base, "record.json"), "w") as fh:
        json.dump({"first": list(RECORD_FIRST), "last": list(SMOKE_RECORD_LAST)},
                  fh)

    t_lo = f10b.seconds_since_epoch(d_lo)
    t_hi = f10b.seconds_since_epoch(d_hi) + 86399
    bins = sh.bins_overlapping(t_lo, t_hi)
    months = sorted({ym for ym in
                     (classify_bin(b, RECORD_FIRST, SMOKE_RECORD_LAST)
                      for b in bins) if not isinstance(ym, str)})
    fields = {}
    for (y, m) in months:
        items = []
        for t in SMOKE_TILES + (SMOKE_ABSENT_TILE,):
            doy = (dt.date(y, m, 1) - dt.date(y, 1, 1)).days + 1
            gid = (f"{SHORT_NAME}.A{y:04d}{doy:03d}.{t}.{VERSION}."
                   f"{2021000000000 + doy}")
            name = gid + ".hdf"
            url = ("https://data.lpdaac.earthdatacloud.nasa.gov/"
                   f"lp-prod-protected/{SHORT_NAME}.{VERSION}/{gid}/{name}")
            items.append({"umm": {
                "GranuleUR": gid,
                "TemporalExtent": {"RangeDateTime": {
                    "BeginningDateTime": f"{y:04d}-{m:02d}-01T00:00:00.000Z"}},
                "DataGranule": {"ArchiveAndDistributionInformation": [
                    {"Name": "Not provided", "Size": 0.151104,
                     "SizeUnit": "MB"}]},
                "RelatedUrls": [{"Type": "GET DATA", "URL": url},
                                {"Type": "GET DATA VIA DIRECT ACCESS",
                                 "URL": "s3://lp-prod-protected/" + name}]}})
            if t == SMOKE_ABSENT_TILE:
                continue                      # listed for realism, not a group
            arrays = smoke_fields(t, px, m)
            fields[(y, m, t)] = arrays
            _write_hdf(os.path.join(gdir, name), arrays)
        with open(os.path.join(base, f"cmr_{y:04d}-{m:02d}.json"), "w") as fh:
            json.dump({"hits": len(items), "items": items}, fh)

    ad = Burned500Adapter()
    pg = ad.per_group
    truth = {}
    for g in ad.groups():
        v, h0 = group_parts(g)
        for b in bins:
            # THE SAME CLASSIFIER THE ADAPTER USES, so the truth cannot drift
            # from the store it is meant to check.
            ym = classify_bin(b, RECORD_FIRST, SMOKE_RECORD_LAST)
            if isinstance(ym, str):
                truth[(g, b, 0)] = (None, ym)
                continue
            y, m = ym
            arr = np.full((px, px * pg, 2), np.nan, np.float16)
            n_have = 0
            for k in range(pg):
                t = f"h{h0 + k:02d}v{v:02d}"
                if (y, m, t) not in fields or t not in SMOKE_TILES:
                    continue
                n_have += 1
                burn = fields[(y, m, t)]["Burn Date"].astype(np.int32)
                qa = fields[(y, m, t)]["QA"]
                known = ((burn >= 1) & (burn <= BURN_DOY_MAX)) | (burn == 0)
                blk = np.full((px, px, 2), np.nan, np.float16)
                blk[:, :, 0] = np.where(known, burn.astype(np.float16),
                                        np.float16("nan"))
                blk[:, :, 1] = np.where(known, qa.astype(np.float16),
                                        np.float16("nan"))
                arr[:, k * px:(k + 1) * px, :] = blk
            if not n_have:
                truth[(g, b, 0)] = (None, "absent_upstream")
                continue
            mc.mask_bounds_low_memory(ad, arr)
            truth[(g, b, 0)] = (arr, None)
    return truth


def _write_hdf(path, arrays):
    SD, SDC = mc._pyhdf()
    if os.path.exists(path):
        os.remove(path)
    f = SD(path, SDC.WRITE | SDC.CREATE)
    try:
        for name, a in arrays.items():
            t = {np.dtype("int16"): SDC.INT16,
                 np.dtype("uint8"): SDC.UINT8}[a.dtype]
            s = f.create(name, t, tuple(int(x) for x in a.shape))
            try:
                s[:] = a
            finally:
                s.endaccess()
    finally:
        f.end()


ADAPTER = Burned500Adapter

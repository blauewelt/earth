"""Satellite column CO2 — OCO-2, OCO-3 and (exception E3) GOSAT via ACOS,
one row per sounding (family 1.gf, E-082 wave 4).

PLAIN ENGLISH. A spectrometer looking down at sunlight reflected off the
ground can measure how much carbon dioxide the whole air column above that
spot holds — XCO2, the column-averaged dry-air mole fraction, in parts per
million. NASA's two Orbiting Carbon Observatories have done that since 2014
and 2019, and NASA's ACOS processor does the same for Japan's GOSAT back to
2009. This is the ATMOSPHERE'S side of the air-sea carbon flux: the ocean
carbon sink is a difference between the water and the air, and family 10.1's
`socat` store holds only the water's side.

WHAT THE LEDGER ASKS FOR (`family1gf.tex`, the `xco2` row and exception E3):
OCO-2 L2 Lite XCO2 v11.2r (+ v11.3r from 2024) and OCO-3 v11r, tier P, one
row per sounding, C = 3, with GOSAT ACOS admitted as a THIRD SENSOR CODE
rather than a second store, "with `platform` distinguishing the instrument
and `fp` carrying its own footprint".

SOURCE, VERIFIED 2026-09-18 FROM THIS SANDBOX (CMR is anonymous; the granules
are Earthdata-protected behind GES DISC, so the BYTES are a hosted runner's
job — `credentials = ("EARTHDATA_USERNAME", "EARTHDATA_PASSWORD")`):

  sensor  collection (CMR)               record          measured
  OCO-2   OCO2_L2_Lite_FP 11.2r          2014-06-01 ->   ~56-63 MB a day
          C2912085112-GES_DISC
  OCO-2   OCO2_L2_Lite_FP 11.3r          2014-06-01 ->   the newer processing
          C4203038017-GES_DISC
  OCO-3   OCO3_L2_Lite_FP 11r            2019-08-06 ->   ~47-82 MB a day
          C2910086168-GES_DISC
  GOSAT   ACOS_L2_Lite_FP 11.0 (E3)      2009-04-18 ->   ~6.9 GB in total
          C4106152863-GES_DISC

One file a day per sensor, e.g.
`https://data.gesdisc.earthdata.nasa.gov/data/OCO2_DATA/
OCO2_L2_Lite_FP.11.2r/2019/oco2_LtCO2_190601_B11210Ar_240910224028s.nc4`.
The per-day URL is never constructed: it is read out of the granule's own CMR
entry, so a re-processed build number (`B11210Ar` -> something else) is
followed rather than guessed. An anonymous GET of one answers
`401 HTTP Basic: Access denied.`, and so does the `.nc4.xml` sidecar — which
is why the file's INTERNAL layout is resolved on the runner rather than
transcribed here (below).

WHICH PROCESSING VERSION, AND WHY BOTH OCO-2 COLLECTIONS ARE LISTED. 11.3r is
the newer OCO-2 stream and 11.2r the one the note names; both declare the same
2014-06-01 start. `index` lists BOTH, reports how many days each holds, and
prefers the NEWER version for any day both serve, recording per day which
collection it took (`version_per_day`). Nothing is blended: a day comes whole
from one file, and `qc` records nothing about the version — `plan.json` and
store.json do.

THE FILE'S INTERNAL LAYOUT IS RESOLVED, NOT ASSUMED. `OCO2_L2_Lite` is a
netCDF4 file with root variables and several GROUPS (`Sounding/`,
`Retrieval/`, `Meteorology/`, `Preprocessors/`), and there is no anonymous way
to read its variable list — so instead of writing a name table here and
hoping, `resolve(ds)` matches each quantity against a CANDIDATE LIST (below),
takes the first name the file actually has, and REFUSES with the file's WHOLE
inventory printed when a required quantity matches nothing. `index`
additionally writes that inventory — every variable's path, dtype, shape,
units and fill value — into plan.json, so one runner round trip settles the
format whichever way it goes. The resolution that was used is recorded in
plan.json and in store.json, so a store can never be read without knowing
which variable each channel came from.

WHAT A ROW IS. One GOOD sounding.
  time_s    the file's own time axis through `f10b._cf_time_to_seconds` on its
            `units` attribute (the Lite files' root `time` is seconds since
            1970-01-01); the `date` array (Y, M, D, h, m, s, ms) is the
            fallback and is read the same way for both.
  lat, lon  `latitude`, `longitude`, longitude wrapped into [-180, 180).
  values    C = 3: `xco2` (ppm), `xco2_uncertainty` (ppm) and `surface`
            (0 land, 1 water) — the ledger's three channels.
  platform  `platform_hash("OCO-2" | "OCO-3" | "GOSAT-ACOS")`, so the three
            sensors are one store with a sensor code, as exception E3 asks,
            and a consumer can hold out an instrument.
  qc        the sounding's own quality bits: bit 0 the retrieval's
            `xco2_quality_flag` (always 0 here — see below), bit 1 set when
            the sounding is over water, bits 2-4 the `warn_level` scaled into
            three bits when the file has one. The grade vocabulary comes from
            the file, never from a table here.

ONLY GOOD SOUNDINGS ARE KEPT, and that is why the brief's third channel is not
the quality flag. The brief asks for `C = 3 (xco2 ppm, uncertainty, quality
flag)`; the Lite files carry every sounding with `xco2_quality_flag` 0 (good)
or 1 (bad), and a bad-quality XCO2 is a FAILED RETRIEVAL rather than a
measurement of CO2. Keeping only the good ones is also what makes the note's
size arithmetic true — it counts "≈ 400 M GOOD soundings (OCO-2) + OCO-3" at
≈ 18 GB, where every sounding would be several times that. With the bad ones
gone the flag channel would be a column of zeros, so the third channel is the
ledger's `surface` (land or water, which decides whether a sounding sees the
ocean's side of the flux at all) and the flag lives in `qc`. Dropped soundings
are counted by reason (`soundings_bad_quality`, `soundings_no_xco2`,
`soundings_no_position`), never silently.

THE FOOTPRINT. OCO-2's and OCO-3's soundings are under 3 km^2, i.e. about
1.6 km across, so log2_fp = log2(1.6 / 27.83) = -4.12 — BELOW the -4 label,
which `family1gf.tex` §4 says is exactly right: "a pixel with a real support
carries its true value even below -4". GOSAT's footprint is 10.5 km across
(-1.41), so the E3 sensor does not share OCO's number and the adapter writes
the per-sensor value into store.json's `footprint_per_platform` while the
store-level `log2_fp` stays OCO's. log2_dt is the -4 label: a sounding is
instantaneous.

SIZE. The note projects ≈ 400 M good OCO-2 soundings plus OCO-3, ≈ 18 GB at
27 + 2C = 33 bytes a row. 231 + 53 GB of OCO-2 and OCO-3 source plus 117 GB of
GOSAT is fetched to produce it, so the store is about 4 % of what is read and
the LANES are sized by download time, not by disk. THE PROBE REPLACES THOSE
NUMBERS.
"""
import datetime as dt
import json
import os
import re
import sys
import time
import urllib.parse

import numpy as np

import build_family10_stores as f10b
from family1.adapters import _common as cm

CMR = "https://cmr.earthdata.nasa.gov/search/granules.json"

# (sensor code, CMR collection concept id, short name, version, the footprint
# across in km). The ORDER is the preference order when two collections serve
# the same sensor-day: later wins, so 11.3r beats 11.2r.
SENSORS = (
    ("OCO-2", "C2912085112-GES_DISC", "OCO2_L2_Lite_FP", "11.2r", 1.6),
    ("OCO-2", "C4203038017-GES_DISC", "OCO2_L2_Lite_FP", "11.3r", 1.6),
    ("OCO-3", "C2910086168-GES_DISC", "OCO3_L2_Lite_FP", "11r", 1.6),
    ("GOSAT-ACOS", "C4106152863-GES_DISC", "ACOS_L2_Lite_FP", "11.0", 10.5),
)
# `oco2_LtCO2_190601_B11210Ar_240910224028s.nc4` -> the YYMMDD in the middle
NAME = re.compile(r"(?:oco2|oco3|acos)_LtCO2_(\d{6})_", re.I)
CHANNELS = (("xco2", "ppm", 300.0, 500.0),
            ("xco2_uncertainty", "ppm", 0.0, 20.0),
            ("surface", "0 land / 1 water", 0.0, 1.0))
FIRST_YEAR = 2009
WORKERS = 3
YIELD_ROWS = 1_000_000

# THE CANDIDATE LISTS. Each quantity is resolved against the file by taking
# the first path it actually has; a required one that matches nothing is a
# REFUSAL with the file's whole inventory printed. Paths are netCDF4 group
# paths ("Sounding/land_water_indicator") and are searched in this order.
WANT = {
    "xco2": (("xco2",), True),
    "xco2_uncertainty": (("xco2_uncertainty", "xco2_uncert"), True),
    "latitude": (("latitude", "vertex_latitude"), True),
    "longitude": (("longitude", "vertex_longitude"), True),
    "time": (("time", "date"), True),
    "quality": (("xco2_quality_flag", "Retrieval/xco2_quality_flag"), True),
    "surface": (("Sounding/land_water_indicator",
                 "Retrieval/land_water_indicator",
                 "Sounding/land_fraction"), False),
    "warn": (("warn_level", "Retrieval/warn_level"), False),
}
GOOD_QUALITY = 0
QC_WATER_BIT = 1


class FormatError(ValueError):
    """A granule listing or a Lite file that does not look like the product."""


# ================================================================= listing =
def cmr_days(cid, attempts=4, count=None, temporal=None):
    """{date: entry} for one collection — every granule, paged.

    ANONYMOUS. Paged by `cm.cmr_entries` on the `CMR-Search-After` cursor
    and checked against `CMR-Hits`; a cut listing raises `cm.CMRTruncated`
    (the 2026-09-23 irtb failure: a partial page during a slow hour of CMR
    read as the end of the listing).
    """
    q = {"collection_concept_id": cid, "sort_key": "start_date",
         "temporal": temporal or "1900-01-01T00:00:00Z,2100-01-01T00:00:00Z"}
    out = {}
    for g in cm.cmr_entries(CMR, q, attempts=attempts, count=count,
                            what=f"{cid} {q['temporal']}"):
        title = g.get("title", "")
        m = NAME.search(title)
        if not m:
            continue
        stamp = m.group(1)
        y = 2000 + int(stamp[:2])
        try:
            d = dt.date(y, int(stamp[2:4]), int(stamp[4:6]))
        except ValueError:
            raise FormatError(f"{title}: {stamp} is not a YYMMDD "
                              f"date") from None
        url = None
        for ln in g.get("links", []):
            h = str(ln.get("href", ""))
            if h.startswith("https") and h.endswith((".nc4", ".nc")) \
                    and "opendap" not in h:
                url = h
                break
        if url is None:
            raise FormatError(f"{title}: no https .nc4 link in its CMR "
                              f"entry")
        out[d] = {"name": title, "url": url,
                  "bytes": int(float(g.get("granule_size") or 0) * 1e6)}
    return out


# ================================================================ one file =
def inventory(ds, prefix=""):
    """Every variable in the file and its groups: path -> a describing dict.

    This is what makes ONE runner round trip enough: whatever the Lite file's
    layout turns out to be, plan.json carries it.
    """
    out = {}
    for name, v in ds.variables.items():
        p = f"{prefix}{name}"
        out[p] = {"dtype": str(v.dtype), "shape": [int(x) for x in v.shape],
                  "dims": list(v.dimensions),
                  "units": str(getattr(v, "units", "")),
                  "fill": (float(getattr(v, "_FillValue"))
                           if hasattr(v, "_FillValue") else None),
                  "long_name": str(getattr(v, "long_name", ""))[:120],
                  "flag_values": str(getattr(v, "flag_values", ""))[:80],
                  "flag_meanings": str(getattr(v, "flag_meanings", ""))[:160]}
    for name, g in ds.groups.items():
        out.update(inventory(g, f"{prefix}{name}/"))
    return out


def resolve(inv):
    """{quantity: the path this file actually has}. Raises on a required miss.

    ADAPTER_CONTRACT rule 4: the name table is a HYPOTHESIS the file
    falsifies. A required quantity that matches nothing takes the whole
    inventory down with it, so the refusal names what the file DOES hold.
    """
    got, missing = {}, []
    for q, (cands, required) in WANT.items():
        for c in cands:
            if c in inv:
                got[q] = c
                break
        else:
            if required:
                missing.append((q, cands))
    if missing:
        raise FormatError(
            "the Lite file holds none of the candidate names for "
            + "; ".join(f"{q} ({', '.join(c)})" for q, c in missing)
            + f". The file's own variables are: "
            + ", ".join(sorted(inv)[:80])
            + (" ..." if len(inv) > 80 else ""))
    return got


def _var(ds, path):
    node = ds
    parts = path.split("/")
    for p in parts[:-1]:
        node = node.groups[p]
    return node.variables[parts[-1]]


def _col(ds, path):
    v = _var(ds, path)
    a = np.asarray(np.ma.filled(v[:], np.nan), np.float64)
    fill = getattr(v, "_FillValue", None)
    if fill is not None:
        a = np.where(a == np.float64(fill), np.nan, a)
    a[~np.isfinite(a)] = np.nan
    return a


def _time_seconds(ds, path):
    """The Lite file's time axis -> seconds since 1982-01-01, float64.

    Two shapes, both handled: a 1-D axis with CF `units` (the root `time`,
    "seconds since 1970-01-01"), and the `date` array (n, 7) of
    (Y, M, D, h, m, s, ms) which some versions carry instead.
    """
    v = _var(ds, path)
    a = np.asarray(np.ma.filled(v[:], np.nan), np.float64)
    if a.ndim == 2 and a.shape[1] >= 6:
        y, mo, d = a[:, 0], a[:, 1], a[:, 2]
        days = cm.days_from_civil(y.astype(np.int64), mo.astype(np.int64),
                                  d.astype(np.int64)).astype(np.float64)
        return days * 86400.0 + a[:, 3] * 3600.0 + a[:, 4] * 60.0 + a[:, 5]
    u = str(getattr(v, "units", ""))
    if " since " not in u.lower():
        raise FormatError(f"{path} is 1-D with units {u!r}, which carry no "
                          f"epoch, and is not the (n, 7) `date` array either")
    return f10b._cf_time_to_seconds(a, u)


def read_day(path, sensor, t_lo, t_hi, counts):
    """One Lite file -> (t, lat, lon, values[n, 3], platform, qc) or None."""
    import netCDF4
    ds = netCDF4.Dataset(path)
    try:
        inv = inventory(ds)
        names = resolve(inv)
        n_all = int(np.asarray(_var(ds, names["xco2"]).shape).prod())
        t = _time_seconds(ds, names["time"])
        lat = _col(ds, names["latitude"])
        lon = _col(ds, names["longitude"])
        x = _col(ds, names["xco2"])
        unc = _col(ds, names["xco2_uncertainty"])
        qual = _col(ds, names["quality"])
        water = (_col(ds, names["surface"]) if "surface" in names
                 else np.full(n_all, np.nan))
        warn = (_col(ds, names["warn"]) if "warn" in names
                else np.zeros(n_all))
        qmeanings = str(getattr(_var(ds, names["quality"]), "flag_meanings",
                                ""))
    finally:
        ds.close()
    for a in (lat, lon, x, unc):
        if a.ndim != 1:
            raise FormatError(f"a resolved column has shape {a.shape}; the "
                              f"Lite columns are one-dimensional")
    counts["soundings_in_file"] = counts.get("soundings_in_file", 0) + n_all
    # A DICT OF COUNTS, not a list: `f10b._merge_counts` concatenates lists,
    # so one entry a file would put hundreds of identical resolutions in a
    # year's ledger. As "quantity=path" -> n it merges into a tally, and a
    # file that resolved DIFFERENTLY shows up as a second key rather than
    # being lost in the noise.
    res = counts.setdefault("resolution", {})
    for q, pth in names.items():
        res[f"{q}={pth}"] = res.get(f"{q}={pth}", 0) + 1
    if qmeanings:
        qm = counts.setdefault("quality_flag_meanings", {})
        qm[qmeanings[:160]] = qm.get(qmeanings[:160], 0) + 1
    bad_q = ~(qual == GOOD_QUALITY)
    no_x = ~np.isfinite(x)
    no_p = ~np.isfinite(lat) | ~np.isfinite(lon) | ~np.isfinite(t)
    keep = ~bad_q & ~no_x & ~no_p
    with np.errstate(invalid="ignore"):
        keep &= (t >= t_lo) & (t <= t_hi)
    for k, v in (("soundings_bad_quality", bad_q.sum()),
                 ("soundings_no_xco2", (~bad_q & no_x).sum()),
                 ("soundings_no_position", (~bad_q & ~no_x & no_p).sum())):
        if v:
            counts[k] = counts.get(k, 0) + int(v)
    if not keep.any():
        return None
    # `land_water_indicator` is 0 land / 1 water in the OCO products;
    # `land_fraction` is a PERCENTAGE, so it is turned into the same 0/1 by
    # its own name rather than by its values.
    w = water[keep]
    if "surface" in names and names["surface"].endswith("land_fraction"):
        w = np.where(np.isfinite(w), (w < 50.0).astype(np.float64), np.nan)
    vals = np.stack([x[keep], unc[keep], w], axis=1)
    qc = np.zeros(int(keep.sum()), np.uint8)
    qc |= np.where(np.isfinite(w) & (w >= 0.5), QC_WATER_BIT, 0) \
        .astype(np.uint8)
    wl = warn[keep]
    qc |= (np.clip(np.nan_to_num(wl, nan=0.0), 0, 21).astype(np.uint8)
           // 3 << 2)
    plat = np.full(int(keep.sum()), f10b.platform_hash(sensor), np.int64)
    counts["soundings_kept"] = counts.get("soundings_kept", 0) + len(plat)
    per = counts.setdefault("soundings_per_sensor", {})
    per[sensor] = per.get(sensor, 0) + len(plat)
    return t[keep], lat[keep], lon[keep], vals, plat, qc


# ================================================================ adapter ==
class XCO2Adapter(f10b.SourceAdapter):
    store = "xco2"
    title = ("Satellite column CO2: OCO-2 and OCO-3 L2 Lite XCO2, with GOSAT "
             "via ACOS as a third sensor code (exception E3) — one row per "
             "good sounding")
    family = "1gf"
    distribution = "public"
    licence = {"name": "NASA open data (CC0-equivalent)",
               "redistribution": "yes",
               "redistribution_confirmed": True,
               "derived_works": "free",
               "attribution": ("OCO-2/OCO-3 Science Team, Vivienne Payne, "
                               "Abhishek Chatterjee (2024), OCO-2 / OCO-3 "
                               "Level 2 bias-corrected XCO2 and other select "
                               "fields from the full-physics retrieval "
                               "aggregated as daily files, Goddard Earth "
                               "Sciences Data and Information Services Center "
                               "(GES DISC); ACOS Science Team for the GOSAT "
                               "retrieval")}
    time_dtype = "int32"
    credentials = ("EARTHDATA_USERNAME", "EARTHDATA_PASSWORD")
    channels = CHANNELS
    # <3 km^2, i.e. ~1.6 km across: log2(1.6 / 27.83) = -4.121. Below the -4
    # LABEL on purpose (family1gf.tex §4: a real support is written as it is).
    log2_fp = float(np.log2(1.6 / 27.83))
    log2_dt = -4.0
    per_year = True
    first_year = FIRST_YEAR
    fetch_month_scope = "month"
    qc_policy = (
        "ONLY GOOD SOUNDINGS ARE STORED: a sounding whose "
        "`xco2_quality_flag` is not 0 is a FAILED RETRIEVAL rather than a "
        "measurement of CO2, and it is dropped and counted "
        "(`soundings_bad_quality`), which is also what makes the note's "
        "'400 M GOOD soundings, 18 GB' arithmetic true. With the bad ones "
        "gone a quality channel would be a column of zeros, so the brief's "
        "third channel is the ledger's `surface` (0 land / 1 water) and the "
        "flag lives in `qc`: bit 0 the quality flag, bit 1 set over water, "
        "bits 2-4 the `warn_level` in three bits where the file has one. A "
        "sounding with no xco2, no position or no time is dropped and "
        "counted. A value outside its channel's bounds becomes NaN for that "
        "channel alone and is counted, never clipped. The sensor is the "
        "`platform` (OCO-2, OCO-3 or GOSAT-ACOS), so exception E3's third "
        "instrument is a code rather than a second store, and a consumer can "
        "hold out an instrument.")
    sources = tuple(f"{CMR}?collection_concept_id={cid} ({sn} {v}, anonymous)"
                    for (_s, cid, sn, v, _fp) in SENSORS)
    verified = (
        "2026-09-18 from this sandbox (CMR only — a GES DISC .nc4 and even "
        "its .nc4.xml sidecar answer `401 HTTP Basic: Access denied.` "
        "anonymously, so the BYTES are a hosted runner's job): the four "
        "collections and their records — OCO2_L2_Lite_FP 11.2r "
        "(C2912085112-GES_DISC, 2014-06-01 ->, ~56-63 MB a day), 11.3r "
        "(C4203038017-GES_DISC, 2014-06-01 ->), OCO3_L2_Lite_FP 11r "
        "(C2910086168-GES_DISC, 2019-08-06 ->, ~47-82 MB a day) and "
        "ACOS_L2_Lite_FP 11.0 (C4106152863-GES_DISC, 2009-04-18 ->, "
        "exception E3) — and each granule's own https .nc4 link read from "
        "its CMR entry. NOT MEASURED HERE, and what the runner probe "
        "settles: the Lite file's INTERNAL variable layout. There is no "
        "anonymous route to it, so the adapter resolves each quantity "
        "against a candidate list, refuses with the file's WHOLE inventory "
        "when a required one matches nothing, and `index` writes that "
        "inventory into plan.json — one round trip settles it either way, "
        "and the resolution that was used is recorded in the store.")
    notes = (
        "Exception E3: GOSAT through NASA's ACOS retrieval is a third SENSOR "
        "CODE in this store, not a second store, so `platform` distinguishes "
        "OCO-2 / OCO-3 / GOSAT-ACOS. Their footprints differ — 1.6 km across "
        "for the OCOs (log2_fp -4.12) and 10.5 km for GOSAT (-1.41) — so "
        "store.json carries `footprint_per_platform` beside the store-level "
        "log2_fp, which is the OCOs'. Two OCO-2 collections are listed: "
        "11.3r is the newer processing and wins any day both serve, and "
        "plan.json records per day which version was taken.")
    smoke_window = ("2019-06-01", "2019-06-04")
    smoke_probe_month = "2019-06"

    def __init__(self):
        only = (os.environ.get("XCO2_SENSORS") or "").strip()
        self.sensors = tuple(
            s for s in SENSORS
            if not only or s[0] in {x.strip() for x in only.split(",")})
        if not self.sensors:
            sys.exit(f"XCO2_SENSORS={only!r} names none of "
                     f"{sorted({s[0] for s in SENSORS})}")
        if only:
            self.notes = (f"{self.notes}\nXCO2_SENSORS restricted this build "
                          f"to {only!r}. This is not the whole store.")
        self._days = None
        self._session = None

    # ------------------------------------------------------------- listing --
    def days(self, ctx):
        """{date: entry} with the sensor and version resolved per day."""
        if self._days is not None:
            return self._days
        out, per_coll = {}, {}
        if ctx.source_dir:
            root = os.path.join(ctx.source_dir, self.store)
            for sensor, _cid, sn, ver, _fp in self.sensors:
                d0 = os.path.join(root, f"{sn}.{ver}")
                if not os.path.isdir(d0):
                    continue
                got = {}
                for dirpath, _dirs, files in os.walk(d0):
                    for n in sorted(files):
                        m = NAME.search(n)
                        if not m or not n.endswith((".nc4", ".nc")):
                            continue
                        st = m.group(1)
                        d = dt.date(2000 + int(st[:2]), int(st[2:4]),
                                    int(st[4:6]))
                        p = os.path.join(dirpath, n)
                        got[d] = {"name": n.rsplit(".", 1)[0], "url": p,
                                  "bytes": os.path.getsize(p)}
                        ctx.count_bytes(0)
                per_coll[f"{sn}.{ver}"] = len(got)
                for d, e in got.items():
                    out[(sensor, d)] = {**e, "sensor": sensor,
                                        "version": f"{sn}.{ver}"}
        else:
            for sensor, cid, sn, ver, _fp in self.sensors:
                try:
                    got = cmr_days(cid, attempts=ctx.a.attempts,
                                   count=ctx.count_bytes)
                except (FormatError, cm.CMRTruncated) as e:
                    sys.exit(f"REFUSING xco2: {e}")
                if not got:
                    sys.exit(f"REFUSING xco2: CMR lists no granule of "
                             f"{sn} {ver} ({cid}) — an empty listing is a "
                             f"refusal (ml/CLAUDE.md, the 2026-09-14 rule)")
                per_coll[f"{sn}.{ver}"] = len(got)
                for d, e in got.items():
                    out[(sensor, d)] = {**e, "sensor": sensor,
                                        "version": f"{sn}.{ver}"}
        if not out:
            sys.exit("REFUSING xco2: no granule at all for "
                     f"{[s[0] for s in self.sensors]}")
        self._days = (out, per_coll)
        return self._days

    def _get(self, ctx, key, entry):
        if ctx.source_dir:
            p = entry["url"]
            if not os.path.exists(p):
                raise f10b._NotFound(p)
            ctx.count_bytes(os.path.getsize(p))
            return p, False
        # EARTHDATA LOGIN, not a plain GET: a GES DISC .nc4 answers an
        # unauthenticated request with `401 HTTP Basic: Access denied.` after
        # a redirect to urs.earthdata.nasa.gov. `cm.earthdata_download`
        # carries the .netrc through the redirect chain, to the login host
        # and nowhere else, and forces IPv4.
        dest = os.path.join(ctx.scratch, self.store,
                            f"{entry['sensor']}_{entry['name']}.nc4")
        if self._session is None:
            self._session = cm.earthdata_session()
        n, why = cm.earthdata_download(self._session, entry["url"], dest,
                                       attempts=max(1, ctx.a.attempts))
        if n is None:
            raise f10b._NotFound(f"{entry['url']}: {why}")
        return dest, True

    # ---------------------------------------------------------------- rows --
    def _rows(self, ctx, label, d_lo, d_hi, t_lo, t_hi):
        days, _ = self.days(ctx)
        keys = sorted((k for k in days if d_lo <= k[1] <= d_hi),
                      key=lambda k: (k[1], k[0]))
        counts = {"files_wanted": len(keys)}
        t0 = time.time()

        def get(k):
            try:
                return k, self._get(ctx, k, days[k]), None
            except f10b._NotFound:
                return k, None, "listed in CMR, and 404"
            except IOError as e:
                return k, None, f"{type(e).__name__}: {e}"

        workers = 1 if ctx.source_dir else WORKERS
        for k, got, err in cm.ordered_map(get, keys, workers,
                                          lookahead=2 * workers):
            if got is None:
                ctx.note_absent(f"{label} {k[0]} {k[1]}",
                                f"{days[k]['name']}: {err}")
                continue
            path, tmp = got
            c = {}
            try:
                r = read_day(path, k[0], t_lo, t_hi, c)
            except FormatError as e:
                sys.exit(f"REFUSING xco2: {days[k]['name']}: {e}")
            except (OSError, IOError) as e:
                ctx.note_absent(f"{label} {k[0]} {k[1]}",
                                f"{days[k]['name']}: not a readable netCDF "
                                f"({type(e).__name__}: {e})")
                continue
            finally:
                if tmp and os.path.exists(path):
                    os.remove(path)
            counts["files_read"] = counts.get("files_read", 0) + 1
            f10b._merge_counts(counts, c)
            if r is None:
                continue
            t, lat, lon, vals, plat, qc = r
            oob = self.mask_bounds(vals)
            if oob:
                f10b._merge_counts(counts, {"out_of_bounds": oob})
            cols = self.pack(t, lat, lon, vals, plat, qc)
            for lo in range(0, len(t), YIELD_ROWS):
                yield (f"{label} {k[0]} {k[1]}",
                       {kk: v[lo:lo + YIELD_ROWS] for kk, v in cols.items()},
                       None)
        counts["fetch_seconds"] = round(time.time() - t0, 1)
        yield label, None, counts

    # ------------------------------------------------------------ contract --
    def index(self, ctx):
        days, per_coll = self.days(ctx)
        per_year, per_sensor, version_per_day = {}, {}, {}
        nb = 0
        for (sensor, d), e in days.items():
            per_year.setdefault(str(d.year), {})
            per_year[str(d.year)][sensor] = \
                per_year[str(d.year)].get(sensor, 0) + 1
            per_sensor[sensor] = per_sensor.get(sensor, 0) + 1
            version_per_day.setdefault(str(d), {})[sensor] = e["version"]
            nb += e["bytes"]
        dates = sorted({d for (_s, d) in days})
        out = {"dataset": ("OCO-2 / OCO-3 L2 Lite XCO2 with GOSAT via ACOS "
                           "(exception E3)"),
               "url": CMR, "version": "OCO-2 11.3r preferred over 11.2r",
               "collections": [{"sensor": s, "concept_id": cid,
                                "short_name": sn, "version": v,
                                "footprint_km": fp,
                                "granules": per_coll.get(f"{sn}.{v}")}
                               for (s, cid, sn, v, fp) in self.sensors],
               "files": len(days),
               "files_per_sensor": per_sensor,
               "files_per_year": per_year,
               "bytes": int(nb),
               "record": [str(dates[0]), str(dates[-1])],
               "version_per_day_sample": dict(
                   sorted(version_per_day.items())[:5]),
               "footprint_per_platform": {
                   s: round(float(np.log2(fp / 27.83)), 4)
                   for (s, _c, _sn, _v, fp) in self.sensors}}
        # ONE REAL FILE, downloaded and INVENTORIED: the whole point of the
        # index here is that the Lite layout has never been read anonymously.
        want = [k for k in sorted(days, key=lambda k: (k[1], k[0]))
                if ctx.d_lo <= k[1] <= ctx.d_hi] or \
            sorted(days, key=lambda k: (k[1], k[0]))[:1]
        k = want[0]
        try:
            path, tmp = self._get(ctx, k, days[k])
        except (IOError, f10b._NotFound) as e:
            sys.exit(f"REFUSING xco2: the first file {days[k]['name']} could "
                     f"not be read: {e}")
        try:
            import netCDF4
            ds = netCDF4.Dataset(path)
            try:
                inv = inventory(ds)
                dims = {n: int(len(d)) for n, d in ds.dimensions.items()}
                fmt = ds.data_model
            finally:
                ds.close()
        finally:
            if tmp and os.path.exists(path):
                os.remove(path)
        out["first_file"] = {"name": days[k]["name"], "sensor": k[0],
                             "date": str(k[1]), "format": fmt,
                             "dimensions": dims,
                             "variables": len(inv),
                             "inventory": inv}
        try:
            out["first_file"]["resolution"] = resolve(inv)
        except FormatError as e:
            sys.exit(f"REFUSING xco2: {days[k]['name']}: {e}")
        return out

    def extra_meta(self, ctx, dest, N, values):
        return {"footprint_per_platform": {
            s: round(float(np.log2(fp / 27.83)), 4)
            for (s, _c, _sn, _v, fp) in self.sensors},
            # DEDUPLICATED: OCO-2 appears twice in `SENSORS` because two
            # collections serve it, and a store.json that listed the sensor
            # twice would read as three instruments.
            "sensors": list(dict.fromkeys(s[0] for s in self.sensors))}

    def fetch_year(self, ctx, year):
        d_lo = max(dt.date(int(year), 1, 1), ctx.d_lo)
        d_hi = min(dt.date(int(year), 12, 31), ctx.d_hi)
        yield from self._rows(ctx, str(year), d_lo, d_hi, ctx.t_lo, ctx.t_hi)

    def fetch_month(self, ctx, year, month):
        lo, hi = f10b.month_bounds_s(year, month)
        last = (dt.date(int(year) + (int(month) == 12), int(month) % 12 + 1, 1)
                - dt.timedelta(days=1))
        yield from self._rows(ctx, f"{year}-{int(month):02d}",
                              max(dt.date(int(year), int(month), 1), ctx.d_lo),
                              min(last, ctx.d_hi),
                              max(lo, ctx.t_lo), min(hi, ctx.t_hi))

    # --------------------------------------------------------------- smoke --
    def fetch_preflight(self, ctx):
        """The netrc must exist BEFORE the first granule (ml/CLAUDE.md §0.3).

        Every Lite file is Earthdata-protected, so a fetch without a netrc
        spends the whole CMR listing to discover a page of 401s — which is
        exactly what the first swot probe did (family1-build run #152).
        """
        if not ctx.source_dir and not cm.earthdata_ready():
            sys.exit(
                "REFUSING xco2: "
                "the Lite files are Earthdata-protected and this "
                "process can authenticate to Earthdata Login in NEITHER way: "
                "no EARTHDATA_USERNAME / EARTHDATA_PASSWORD in the "
                "environment and no netrc naming urs.earthdata.nasa.gov. "
                "family1-build.yml gives both on a HOSTED runner from the "
                "repository secrets (ml/CLAUDE.md §6); a box gets neither. "
                "Nothing has been fetched.")
        return None

    def smoke_sources(self, root, d_lo, d_hi, seed=20260918):
        truth = make_smoke_sources(root, d_lo, d_hi, seed)
        self.__init__()
        return truth


# ================================================================== smoke ==
SMOKE_PER_DAY = 30
SMOKE_SENSORS = ("OCO-2", "OCO-3")
SMOKE_OOB_DAY = dt.date(2019, 6, 2)
FILL = -999999.0


def write_lite(path, d, sensor, seed):
    """A Lite-layout netCDF4: root columns plus a `Sounding/` group."""
    import netCDF4
    rng = np.random.default_rng(seed + d.toordinal() + len(sensor))
    n = SMOKE_PER_DAY
    ds = netCDF4.Dataset(path, "w", format="NETCDF4")
    ds.createDimension("sounding_id", n)
    t0 = (d - dt.date(1970, 1, 1)).days * 86400.0
    t = t0 + np.sort(rng.choice(86400, n, replace=False)).astype(np.float64)
    v = ds.createVariable("time", "f8", ("sounding_id",))
    v[:] = t
    v.units = "seconds since 1970-01-01"
    lat = np.round(rng.uniform(-60.0, 60.0, n), 4)
    lon = np.round(rng.uniform(-180.0, 179.99, n), 4)
    x = np.round(rng.uniform(395.0, 415.0, n), 4)
    unc = np.round(rng.uniform(0.3, 1.5, n), 4)
    q = np.zeros(n, np.int8)
    q[:3] = 1                                  # three bad-quality soundings
    x[4] = FILL                                # one with no xco2
    lat[5] = FILL                              # one with no position
    if d == SMOKE_OOB_DAY:
        x[6] = 900.0                           # out of bounds -> NaN, counted
    for name, a, unit in (("latitude", lat, "degrees_north"),
                          ("longitude", lon, "degrees_east"),
                          ("xco2", x, "ppm"),
                          ("xco2_uncertainty", unc, "ppm")):
        vv = ds.createVariable(name, "f4", ("sounding_id",),
                               fill_value=np.float32(FILL))
        vv[:] = a
        vv.units = unit
    vq = ds.createVariable("xco2_quality_flag", "i1", ("sounding_id",))
    vq[:] = q
    vq.flag_values = np.array([0, 1], np.int8)
    vq.flag_meanings = "good bad"
    vw = ds.createVariable("warn_level", "i1", ("sounding_id",))
    vw[:] = (np.arange(n) % 21).astype(np.int8)
    g = ds.createGroup("Sounding")
    vl = g.createVariable("land_water_indicator", "i1", ("sounding_id",))
    vl[:] = (np.arange(n) % 2).astype(np.int8)
    vl.flag_meanings = "land water"
    ds.close()
    return {"time": t, "lat": lat, "lon": lon, "xco2": x, "unc": unc,
            "q": q, "water": np.arange(n) % 2,
            "warn": np.arange(n) % 21}


def make_smoke_sources(root, d_lo, d_hi, seed=20260918):
    """`<root>/xco2/<ShortName>.<version>/<YYYY>/<name>.nc4`, the GES DISC
    layout, one file a day a sensor, plus the truth rows.

    The truth is ordered the way the store breaks ties among equal
    (bin, time_s): the adapter's own yield order, which is (date, sensor).
    """
    os.environ["XCO2_SENSORS"] = ",".join(SMOKE_SENSORS)
    base = os.path.join(root, "xco2")
    coll = {"OCO-2": "OCO2_L2_Lite_FP.11.2r", "OCO-3": "OCO3_L2_Lite_FP.11r"}
    pre = {"OCO-2": "oco2", "OCO-3": "oco3"}
    truth = []
    d = d_lo
    while d <= d_hi:
        for sensor in SMOKE_SENSORS:
            dirn = os.path.join(base, coll[sensor], f"{d.year:04d}")
            os.makedirs(dirn, exist_ok=True)
            name = (f"{pre[sensor]}_LtCO2_{d.strftime('%y%m%d')}_B11210Ar_"
                    f"240910224028s.nc4")
            cols = write_lite(os.path.join(dirn, name), d, sensor, seed)
            off = (dt.date(1970, 1, 1) - dt.date(1982, 1, 1)).days * 86400
            for i in range(SMOKE_PER_DAY):
                if cols["q"][i] != 0:
                    continue
                if cols["xco2"][i] == FILL or cols["lat"][i] == FILL:
                    continue
                vv = []
                raw = [cols["xco2"][i], cols["unc"][i],
                       float(cols["water"][i])]
                for (nm, _u, lo, hi), xx in zip(CHANNELS, raw):
                    vv.append(np.nan if not (lo <= xx <= hi) else float(xx))
                truth.append({"t": int(round(cols["time"][i] + off)),
                              "lat": float(cols["lat"][i]),
                              "lon": float(cols["lon"][i]),
                              "platform": f10b.platform_hash(sensor),
                              "v": vv})
        d += dt.timedelta(days=1)
    return truth


ADAPTER = XCO2Adapter

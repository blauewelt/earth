"""ESA CCI Sea State v4 along-track significant wave height — how high the
waves were, one 1 Hz sample at a time, along every radar altimeter's ground
track since 1991 (family 1.gf, E-082 wave 4).

PLAIN ENGLISH. A radar altimeter measures the height of the sea surface
underneath it every second; the SHAPE of the returned pulse also says how
rough that surface is, which is the significant wave height (the average
height of the highest third of the waves). Ifremer's ESA Climate Change
Initiative Sea State processor merges every altimeter that has ever flown —
Topex-Poseidon, ERS-1/2, Envisat, Jason-1/2/3, Cryosat-2, Saral/AltiKa,
Sentinel-3A/B, Sentinel-6A — into ONE file per day of 1 Hz along-track
samples. Waves are how the wind stirs the top of the ocean, so this is the
mixing term the sea-surface-temperature goal needs, and it is a read-out the
model can be scored on.

SOURCE, VERIFIED 2026-09-18 FROM THIS SANDBOX — AND IT NEEDS NO ACCOUNT.
The note (`family1gf.tex`, the `swh` ledger row) has this product reached
through Copernicus Marine "(account, as `slatrack`)". It is also on
Copernicus Marine's OWN native object store, which answers an ANONYMOUS
ListObjectsV2 and an anonymous GET:

  https://s3.waw3-1.cloudferro.com/mdl-native-05/native/
      WAVE_GLO_PHY_SWH_L3_MY_014_005/
      cci_obs-wave_glo_phy-swh_my_l3_PT1S-i_202607/
      <YYYY>/<MM>/ESACCI-SEASTATE-L3-SWH-MULTI_1D-<YYYYMMDD>-fv01.nc

The prefix and the `native` asset URL are not guessed: they are read out of
the product's PUBLIC STAC metadata
(`https://stac.marine.copernicus.eu/metadata/WAVE_GLO_PHY_SWH_L3_MY_014_005/
product.stac.json` -> the one dataset item -> `assets.native.href`), so a
re-versioned dataset directory (`..._202607`) is followed rather than
transcribed. `credentials = ()`: the adapter declares none, and a build
therefore runs on a hosted runner OR a box OR this sandbox. The
`copernicusmarine` toolbox is not imported at all — the brief's suggestion to
reuse `ml/build_family10_stores.py`'s `slatrack` toolbox pattern was
measured against this route and the plain object store is simpler, faster and
keyless. (If CloudFerro ever closes the bucket, the toolbox route is the
fallback and `COPERNICUSMARINE_SERVICE_USERNAME` / `_PASSWORD` would go into
`credentials`.)

THE LISTING, MEASURED IN FULL 2026-09-18 (12 paginated ListObjectsV2 calls):
**11,832 daily files, 116,351,837,853 bytes (116.35 GB)**,
`...-19910803-fv01.nc` .. `...-20231231-fv01.nc`. The record spans 11,839
days, so SEVEN days are absent upstream, in two runs:
**1992-07-20/21/22 and 2001-01-19/20/21/22**. Those are recorded as
`missing_days` in plan.json and as `days_absent_upstream` in the fetch's
counts; they are NOT a refusal (the archive simply has no file), while a
LISTED day that will not download or will not parse is `ctx.note_absent` and
stops the year's marker. Bytes a year run 448 MB (1991, from August) and
2.35-3.0 GB in the single-altimeter 1990s to 5.5-6.3 GB from 2016 once
Sentinel-3 and Sentinel-6 joined.

ONE FILE, READ AND MEASURED (ESACCI-SEASTATE-L3-SWH-MULTI_1D-20150101-fv01.nc,
9,802,660 B): NETCDF4, ONE dimension `time` = 139,842, `featureType`
"trajectory", and one variable per quantity along it —

  time               float64, "seconds since 1981-01-01", proleptic_gregorian
  lat, lon           float64 degrees, lon valid_range [-180, 180]
  swh                float64 m, the retracker's own estimate, no correction
  swh_adjusted       float64 m, bias corrected
  swh_denoised       float64 m, bias corrected AND denoised (EMD, Quilfen et
                     al.) — the file's own `key_variables`
  swh_uncertainty    float64 m, speckle-noise and sampling uncertainty
  bathymetry         float64 m (GEBCO_2024), auxiliary
  distance_to_coast  float64 m (GSFC), auxiliary
  satellite          uint8 with `flag_values` 0..11 and `flag_meanings`
                     "cryosat-2 jason-1 jason-2 jason-3 saral sentinel-3_a
                     envisat topex-poseidon ers-..." — the mission per sample
  cycle              uint16, relative_pass uint16 — the orbit bookkeeping
  every float variable has `_FillValue` 1e+20

That day's three platforms are cryosat-2, jason-2 and saral, at stated
spatial resolutions of 6 km, 5.8 km and 7 km.

WHAT A ROW IS. One 1 Hz sample.
  time_s    the file's `time` axis through `f10b._cf_time_to_seconds` (its
            own `units` attribute, never a hard-coded epoch), rounded to the
            second in `_pack`. int32 reaches 2050, the record ends 2023.
  lat, lon  `lat`, `lon`, longitude wrapped into [-180, 180).
  values    C = 3, the ledger's three: `swh`, `swh_denoised`,
            `swh_uncertainty`, in metres.
  platform  `platform_hash` of the MISSION NAME the file's own
            `flag_meanings` gives for that sample's `satellite` code — so a
            consumer holds out a mission the way it holds out a year, and the
            name comes from the file rather than from a table here.
  qc        1 for every kept row. THE PRODUCT PUBLISHES NO PER-SAMPLE QUALITY
            FLAG: there is no `*_qc` variable in the file at all (measured).
            The brief's `C = 2 (SWH m, quality)` cannot be built, because the
            second channel does not exist; the ledger's three channels can
            be, and `swh_uncertainty` is the quantity closest to a quality
            number. Saying so here is the whole of the honesty: a `qc` column
            invented from `distance_to_coast` or `bathymetry` would be a
            grade this product never gave.

A ROW IS DROPPED, and counted, when BOTH `swh` and `swh_denoised` are fill or
non-finite (`rows_no_swh`) or its time or position is fill
(`rows_no_time`, `rows_no_position`). A value outside its channel's bounds
becomes NaN for that channel alone and is counted (`out_of_bounds`), never
clipped — bounds are [0, 25] m for `swh`, [-1, 25] m for `swh_denoised`
(the STAC's own minimum is -0.081 m: denoising can undershoot zero) and
[0, 5] m for `swh_uncertainty`. The tallest significant wave height ever
measured is about 20 m (Sentinel-6/North Atlantic buoy records), so 25 is a
bound a measurement should not reach and a corrupt file will.

THE FOOTPRINT. The three missions in the checked file state 5.8-7 km
footprints, so log2_fp = log2(7 / 27.83) = -1.99, the ledger's -2.0; log2_dt
is the -4 LABEL for an instantaneous sample (`family1gf.tex` §4).

SIZE, PROJECTED FROM THE MEASUREMENT. 9,802,660 source bytes gave 139,842
samples on 2015-01-01 — 70.1 netCDF bytes a sample. Over the whole 116.35 GB
listing that is about **1.66e9 rows**, and a schema-2 row costs 27 + 2C = 33
bytes, so about **55 GB stored**. The note estimates ~2e9 rows and ~70 GB,
so the measurement is ~17 % under it. THE PROBE REPLACES BOTH NUMBERS.

LANES AND THE BOX. One lane per year: the biggest year is 6.3 GB of source
and about 2.9 GB of parts, well inside a hosted runner's six hours and 86 GB.
The ASSEMBLY is not: 55 GB of parts plus a 55 GB store is 110 GB, so swh is
parked for the box exactly as ghcnd and icoads were (`BUILD_LOG.md`).
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

PRODUCT = "WAVE_GLO_PHY_SWH_L3_MY_014_005"
STAC = (f"https://stac.marine.copernicus.eu/metadata/{PRODUCT}/"
        f"product.stac.json")
# The one dataset item's `native` asset, as read from the STAC on 2026-09-18.
# `index` re-reads the STAC and REFUSES if the live href no longer starts
# with this bucket/prefix — the path is a hypothesis, not a fact.
S3_HOST = "https://s3.waw3-1.cloudferro.com"
BUCKET = "mdl-native-05"
DATASET = "cci_obs-wave_glo_phy-swh_my_l3_PT1S-i_202607"
PREFIX = f"native/{PRODUCT}/{DATASET}/"
FILE = re.compile(r"ESACCI-SEASTATE-L3-SWH-MULTI_1D-(\d{8})-fv(\d+)\.nc$")
FILL = 1e20
FIRST_YEAR = 1991
# The file's own dimension and variables, checked on every file read.
TIME_DIM = "time"
NEEDED = ("time", "lat", "lon", "swh", "swh_denoised", "swh_uncertainty",
          "satellite")
CHANNELS = (("swh", "m", 0.0, 25.0),
            ("swh_denoised", "m", -1.0, 25.0),
            ("swh_uncertainty", "m", 0.0, 5.0))
WORKERS = 4
# Downloads are ~10 MB each and a year is ~365 of them; a batch is parsed and
# deleted before the next is asked for, so at most `WORKERS * 2` sit on disk.
YIELD_ROWS = 1_000_000


class FormatError(ValueError):
    """A listing or a netCDF that does not look like the archive measured."""


# ================================================================ listing ==
def s3_list(prefix, attempts=4, count=None):
    """Every (key, size) under `prefix`, following ListObjectsV2 pages.

    ANONYMOUS. The bucket answers `list-type=2` with no credential (measured
    2026-09-18). A page that comes back with no key at all is a REFUSAL, not
    an empty archive (ml/CLAUDE.md, the 2026-09-14 rule) — the caller decides,
    because a per-year prefix legitimately has none before the record starts.
    """
    out, token = [], None
    while True:
        q = {"list-type": "2", "prefix": prefix, "max-keys": "1000"}
        if token:
            q["continuation-token"] = token
        raw, why = cm.get_bytes(f"{S3_HOST}/{BUCKET}?{urllib.parse.urlencode(q)}",
                                attempts=attempts)
        if raw is None:
            raise FormatError(f"the listing of {prefix} answered {why}")
        x = raw.decode("utf-8", "replace")
        if count is not None:
            count(len(raw))
        for k, s in re.findall(r"<Key>([^<]*)</Key>\s*<LastModified>[^<]*"
                               r"</LastModified>\s*<ETag>[^<]*</ETag>\s*"
                               r"<Size>(\d+)</Size>", x):
            out.append((k, int(s)))
        m = re.search(r"<NextContinuationToken>([^<]*)</NextContinuationToken>",
                      x)
        token = m.group(1) if ("<IsTruncated>true" in x and m) else None
        if not token:
            return out


def parse_listing(keys):
    """[(key, size)] -> ({date: (key, size)}, counts). REFUSES a duplicate."""
    out, other = {}, []
    for k, s in keys:
        m = FILE.search(k)
        if not m:
            other.append(k)
            continue
        stamp = m.group(1)
        try:
            d = dt.date(int(stamp[:4]), int(stamp[4:6]), int(stamp[6:8]))
        except ValueError:
            raise FormatError(f"{k}: {stamp} is not a date") from None
        if d in out:
            raise FormatError(f"{k}: two files for {d} "
                              f"({out[d][0]} is the other)")
        out[d] = (k, s)
    counts = {}
    if other:
        counts["files_other"] = len(other)
        counts["files_other_names"] = sorted(other)[:20]
    return out, counts


def gaps(days):
    """Every calendar day between the first and last file that has none."""
    if not days:
        return []
    lo, hi = min(days), max(days)
    out, d = [], lo
    while d <= hi:
        if d not in days:
            out.append(d)
        d += dt.timedelta(days=1)
    return out


# ============================================================== one file ===
def satellite_names(var):
    """`flag_values` / `flag_meanings` of the `satellite` variable -> {code:
    name}, from the FILE. The names are never written here."""
    meanings = str(getattr(var, "flag_meanings", "")).split()
    raw = getattr(var, "flag_values", None)
    if raw is None:
        values = list(range(len(meanings)))
    else:
        values = [int(v) for v in np.asarray(raw).ravel()]
    if not meanings or len(meanings) != len(values):
        raise FormatError(
            f"the `satellite` variable declares {len(values)} flag_values and "
            f"{len(meanings)} flag_meanings — refusing to guess which mission "
            f"a code is")
    return {int(v): m for v, m in zip(values, meanings)}


def nc_check(ds):
    """An open netCDF against the layout measured on the real file. Raises."""
    if TIME_DIM not in ds.dimensions:
        raise FormatError(f"no `{TIME_DIM}` dimension; dimensions are "
                          f"{sorted(ds.dimensions)}")
    if len(ds.dimensions) != 1:
        raise FormatError(f"{len(ds.dimensions)} dimension(s) "
                          f"{sorted(ds.dimensions)} — the trajectory files "
                          f"have exactly one")
    missing = [v for v in NEEDED if v not in ds.variables]
    if missing:
        raise FormatError(f"variables {missing} are absent; the file has "
                          f"{sorted(ds.variables)}")
    for v in NEEDED:
        if ds.variables[v].dimensions != (TIME_DIM,):
            raise FormatError(f"{v} has dimensions "
                              f"{ds.variables[v].dimensions}, not "
                              f"('{TIME_DIM}',)")
    u = getattr(ds.variables["time"], "units", "")
    if " since " not in str(u).lower():
        raise FormatError(f"time units {u!r} carry no epoch")
    return {"n_time": int(len(ds.dimensions[TIME_DIM])),
            "time_units": str(u),
            "variables": sorted(ds.variables),
            "platforms": _attr_list(ds, "platform"),
            "instruments": _attr_list(ds, "instrument"),
            "spatial_resolution": _attr_list(ds, "spatial_resolution"),
            "key_variables": str(getattr(ds, "key_variables", "")),
            "product_version": str(getattr(ds, "product_version", "")),
            "licence_attr": str(getattr(ds, "license", "")),
            "format": ds.data_model}


def _attr_list(ds, name):
    """A global attribute the producer writes as a python-ish list literal."""
    v = getattr(ds, name, None)
    if v is None:
        return None
    if isinstance(v, (list, tuple, np.ndarray)):
        return [str(x) for x in np.asarray(v).ravel()]
    s = str(v)
    try:
        j = json.loads(s.replace("'", '"'))
        return [str(x) for x in j] if isinstance(j, list) else [s]
    except Exception:                                           # noqa: BLE001
        return [s]


def _col(ds, name):
    a = np.asarray(ds.variables[name][:], np.float64)
    fill = getattr(ds.variables[name], "_FillValue", None)
    if fill is not None:
        a = np.where(a == np.float64(fill), np.nan, a)
    a[~np.isfinite(a)] = np.nan
    return a


def read_day(path, t_lo, t_hi, counts):
    """One daily file -> (t, lat, lon, values[n, 3], platform, qc) or None."""
    import netCDF4
    ds = netCDF4.Dataset(path)
    try:
        meta = nc_check(ds)
        n = meta["n_time"]
        counts["samples_in_file"] = counts.get("samples_in_file", 0) + n
        if n == 0:
            return None
        t = f10b._cf_time_to_seconds(_col(ds, "time"), meta["time_units"])
        lat, lon = _col(ds, "lat"), _col(ds, "lon")
        vals = np.stack([_col(ds, c[0]) for c in CHANNELS], axis=1)
        sat = np.asarray(ds.variables["satellite"][:]).astype(np.int64)
        names = satellite_names(ds.variables["satellite"])
    finally:
        ds.close()
    bad_t = ~np.isfinite(t)
    bad_p = ~np.isfinite(lat) | ~np.isfinite(lon)
    no_swh = ~np.isfinite(vals[:, 0]) & ~np.isfinite(vals[:, 1])
    keep = ~bad_t & ~bad_p & ~no_swh
    if t_lo is not None:
        with np.errstate(invalid="ignore"):
            keep &= (t >= t_lo) & (t <= t_hi)
    for k, v in (("rows_no_time", bad_t.sum()),
                 ("rows_no_position", (~bad_t & bad_p).sum()),
                 ("rows_no_swh", (~bad_t & ~bad_p & no_swh).sum())):
        if v:
            counts[k] = counts.get(k, 0) + int(v)
    unknown = sorted({int(c) for c in np.unique(sat[keep])} - set(names))
    if unknown:
        raise FormatError(f"`satellite` code(s) {unknown} are not in the "
                          f"file's own flag_meanings {sorted(names.values())}")
    per = counts.setdefault("samples_per_mission", {})
    for c in np.unique(sat[keep]):
        per[names[int(c)]] = per.get(names[int(c)], 0) + \
            int((sat[keep] == c).sum())
    if not keep.any():
        return None
    plat = np.array([f10b.platform_hash(names[int(c)]) for c in sat[keep]],
                    np.int64)
    return (t[keep], lat[keep], lon[keep], vals[keep], plat,
            np.ones(int(keep.sum()), np.uint8))


# =============================================================== adapter ===
class SWHAdapter(f10b.SourceAdapter):
    store = "swh"
    title = ("ESA CCI Sea State v4 along-track significant wave height, all "
             "altimeter missions 1991-2023, one row per 1 Hz sample")
    family = "1gf"
    distribution = "public"
    licence = {
        "name": "ESA CCI Data Policy (free and open) / Copernicus Marine",
        "redistribution": "attribution",
        "redistribution_confirmed": True,
        "derived_works": "free",
        "attribution": ("these data were obtained from the ESA CCI Sea State "
                        "project (Ifremer/CERSAT, ESA), distributed by the "
                        "Copernicus Marine Service as "
                        "WAVE_GLO_PHY_SWH_L3_MY_014_005. Dodet, G. et al. "
                        "(2020), The Sea State CCI dataset v1, Earth Syst. "
                        "Sci. Data 12, 1929-1951."),
        "terms": ("the daily files' own `license` global attribute, read on "
                  "ESACCI-SEASTATE-L3-SWH-MULTI_1D-20150101-fv01.nc "
                  "2026-09-18: 'ESA CCI Data Policy - free and open access', "
                  "with `acknowledgment` giving the sentence above; served "
                  "under the Copernicus Marine Service licence, which permits "
                  "redistribution with attribution (the licence the family "
                  "10.1 `slatrack` store is published under)"),
    }
    time_dtype = "int32"
    credentials = ()
    platform_meta = False
    channels = CHANNELS
    log2_fp = float(np.log2(7.0 / 27.83))            # -1.99
    log2_dt = -4.0                                   # an instantaneous sample
    per_year = True
    first_year = FIRST_YEAR
    fetch_month_scope = "month"
    qc_policy = (
        "THE PRODUCT PUBLISHES NO PER-SAMPLE QUALITY FLAG — the daily files "
        "carry no `*_qc` variable at all (measured 2026-09-18 on "
        "ESACCI-SEASTATE-L3-SWH-MULTI_1D-20150101-fv01.nc, whose variables "
        "are time, lat, lon, swh, swh_adjusted, swh_denoised, "
        "swh_uncertainty, bathymetry, distance_to_coast, satellite, cycle, "
        "relative_pass). The L2P-to-L3 processor edits upstream, so every "
        "kept row carries qc 1 and the store says so rather than inventing a "
        "grade; `swh_uncertainty` is the per-sample number closest to one and "
        "is a CHANNEL. A row is kept when its time and position are present "
        "and at least one of `swh` / `swh_denoised` is; a value outside its "
        "channel's bounds becomes NaN for that channel alone and is counted, "
        "never clipped. The mission is the `platform` (from the file's own "
        "flag_meanings), so a consumer can hold out an altimeter the way it "
        "holds out a year.")
    sources = (f"{S3_HOST}/{BUCKET}/{PREFIX}<YYYY>/<MM>/"
               f"ESACCI-SEASTATE-L3-SWH-MULTI_1D-<YYYYMMDD>-fv01.nc",
               STAC + " (public metadata, no login)")
    verified = (
        "2026-09-18 from this sandbox, with NO credentials: the product's "
        "public STAC (one dataset item, "
        "cci_obs-wave_glo_phy-swh_my_l3_PT1S-i_202607, time extent "
        "1991-08-03T21:22:46Z .. 2023-12-31T23:59:59Z, variables swh, "
        "swh_adjusted, swh_denoised, swh_uncertainty, bathymetry, "
        "distance_to_coast) and its `native` asset href on "
        "s3.waw3-1.cloudferro.com; the WHOLE anonymous ListObjectsV2 listing "
        "of that prefix (12 pages, 11,832 daily files, 116,351,837,853 bytes, "
        "19910803 .. 20231231, exactly 7 calendar days absent: 1992-07-20/21/"
        "22 and 2001-01-19/20/21/22, no day with two files); and one real "
        "file downloaded and opened with netCDF4 "
        "(ESACCI-SEASTATE-L3-SWH-MULTI_1D-20150101-fv01.nc, 9,802,660 B, "
        "NETCDF4, one dimension time = 139,842, featureType trajectory, "
        "float64 columns with _FillValue 1e+20, time in 'seconds since "
        "1981-01-01' proleptic_gregorian, satellite uint8 with flag_values "
        "0..11 and flag_meanings 'cryosat-2 jason-1 jason-2 jason-3 saral "
        "sentinel-3_a envisat topex-poseidon ers-...', that day's platforms "
        "cryosat-2 / jason-2 / saral at 6 / 5.8 / 7 km, license attribute "
        "'ESA CCI Data Policy - free and open access')")
    notes = (
        "NO ACCOUNT IS NEEDED. The ledger reaches this product through "
        "Copernicus Marine with an account; the same files sit on Copernicus "
        "Marine's own native object store, which answers an anonymous listing "
        "and an anonymous GET, so `credentials = ()` and the "
        "`copernicusmarine` toolbox is never imported. C = 3, the ledger's "
        "channels (swh, swh_denoised, swh_uncertainty): the brief's second "
        "channel 'quality' does not exist in this product, and inventing one "
        "would be a grade the producer never gave. Seven days of the record "
        "are absent upstream (1992-07-20..22, 2001-01-19..22) and are counted "
        "by name, never a silent gap.")
    smoke_window = ("1991-08-03", "1991-08-12")
    smoke_probe_month = "1991-08"

    def __init__(self):
        self._listing = None
        self._native = None

    # ------------------------------------------------------------- listing --
    def native_prefix(self, ctx):
        """The dataset's own `native` asset prefix, from the PUBLIC STAC.

        A re-versioned dataset directory is FOLLOWED, not transcribed; a href
        that leaves this bucket is a refusal, because the adapter's whole
        access story is that this one bucket is anonymous.
        """
        if self._native is not None:
            return self._native
        if ctx.source_dir:
            self._native = PREFIX
            return self._native
        raw, why = cm.get_bytes(STAC, attempts=ctx.a.attempts)
        if raw is None:
            sys.exit(f"REFUSING swh: the STAC {STAC} answered {why}")
        js = json.loads(raw)
        items = [ln for ln in js.get("links", [])
                 if ln.get("rel") == "item"
                 and str(ln.get("href", "")).endswith("/dataset.stac.json")]
        if len(items) != 1:
            sys.exit(f"REFUSING swh: {STAC} lists {len(items)} dataset "
                     f"item(s); the product had exactly one on 2026-09-18 and "
                     f"the adapter reads the one")
        base = STAC.rsplit("/", 1)[0] + "/"
        draw, why = cm.get_bytes(base + items[0]["href"],
                                 attempts=ctx.a.attempts)
        if draw is None:
            sys.exit(f"REFUSING swh: the dataset STAC answered {why}")
        dj = json.loads(draw)
        href = str(((dj.get("assets") or {}).get("native") or {})
                   .get("href", ""))
        want = f"{S3_HOST}/{BUCKET}/native/{PRODUCT}/"
        if not href.startswith(want):
            sys.exit(f"REFUSING swh: the STAC's native asset is {href!r}, "
                     f"which does not start with {want!r} — the anonymous "
                     f"object store this adapter reads has moved, so the "
                     f"access route has to be re-measured before a build")
        self._native = href[len(f"{S3_HOST}/{BUCKET}/"):].rstrip("/") + "/"
        if self._native != PREFIX:
            self.notes = (f"{self.notes}\nThe STAC's native prefix is "
                          f"{self._native!r}, not the {PREFIX!r} measured on "
                          f"2026-09-18 — the dataset directory was "
                          f"re-versioned and the build followed it.")
        return self._native

    def listing(self, ctx):
        """{date: (key, bytes)} for the whole record, and its counts."""
        if self._listing is not None:
            return self._listing
        if ctx.source_dir:
            root = os.path.join(ctx.source_dir, self.store)
            keys = []
            for dirpath, _dirs, files in os.walk(root):
                for n in files:
                    p = os.path.join(dirpath, n)
                    keys.append((os.path.relpath(p, root),
                                 os.path.getsize(p)))
            if not keys:
                sys.exit(f"REFUSING swh: no files under {root} — the smoke's "
                         f"synthetic archive is missing")
            ctx.count_bytes(sum(s for _k, s in keys))
        else:
            pref = self.native_prefix(ctx)
            try:
                keys = s3_list(pref, attempts=ctx.a.attempts,
                               count=ctx.count_bytes)
            except FormatError as e:
                sys.exit(f"REFUSING swh: {e} — a listing that comes back "
                         f"empty is a refusal (ml/CLAUDE.md, the 2026-09-14 "
                         f"rule)")
        try:
            days, counts = parse_listing(keys)
        except FormatError as e:
            sys.exit(f"REFUSING swh: the listing: {e}")
        if not days:
            sys.exit(f"REFUSING swh: the listing holds no "
                     f"ESACCI-SEASTATE-L3-SWH-MULTI_1D file at all "
                     f"({counts}) — an empty listing is a refusal")
        self._listing = (days, counts)
        return self._listing

    def _url(self, ctx, key):
        if ctx.source_dir:
            return os.path.join(ctx.source_dir, self.store, key)
        return f"{S3_HOST}/{BUCKET}/{key}"

    def _get(self, ctx, key, dest):
        """-> path or raises. A short body is a refusal (`http_to_file`)."""
        if ctx.source_dir:
            p = self._url(ctx, key)
            if not os.path.exists(p):
                raise f10b._NotFound(p)
            ctx.count_bytes(os.path.getsize(p))
            return p, False
        url = self._url(ctx, key)
        err = None
        for i in range(max(1, ctx.a.attempts)):
            try:
                f10b.http_to_file(url, dest)
                return dest, True
            except f10b._NotFound:
                raise
            except (IOError, *cm.RETRY_ERRORS) as e:
                err = e
                if i < ctx.a.attempts - 1:
                    time.sleep(3.0 * (2 ** i))
        raise IOError(f"{url}: {type(err).__name__}: {err}")

    # -------------------------------------------------------------- rows ----
    def _days_in(self, lo, hi, days):
        return [d for d in sorted(days) if lo <= d <= hi]

    def _rows(self, ctx, label, d_lo, d_hi, t_lo, t_hi):
        days, _ = self.listing(ctx)
        want = self._days_in(d_lo, d_hi, days)
        counts = {"days_listed": len(want),
                  "days_absent_upstream": len(
                      [d for d in gaps(days) if d_lo <= d <= d_hi])}
        scratch = os.path.join(ctx.scratch, self.store)
        os.makedirs(scratch, exist_ok=True)
        t0 = time.time()

        def get(d):
            key = days[d][0]
            dest = os.path.join(scratch, os.path.basename(key))
            try:
                return d, self._get(ctx, key, dest), None
            except f10b._NotFound:
                return d, None, "listed, and 404"
            except IOError as e:
                return d, None, f"{type(e).__name__}: {e}"

        workers = 1 if ctx.source_dir else WORKERS
        for d, got, err in cm.ordered_map(get, want, workers,
                                          lookahead=2 * workers):
            if got is None:
                ctx.note_absent(f"{label} {d}", f"{days[d][0]}: {err}")
                continue
            path, tmp = got
            c = {}
            try:
                r = read_day(path, t_lo, t_hi, c)
            except FormatError as e:
                sys.exit(f"REFUSING swh: {days[d][0]}: {e}")
            except (OSError, IOError) as e:
                ctx.note_absent(f"{label} {d}",
                                f"{days[d][0]}: not a readable netCDF "
                                f"({type(e).__name__}: {e})")
                continue
            finally:
                if tmp and os.path.exists(path):
                    os.remove(path)
            counts["days_read"] = counts.get("days_read", 0) + 1
            f10b._merge_counts(counts, c)
            if r is None:
                continue
            t, lat, lon, vals, plat, qc = r
            oob = self.mask_bounds(vals)
            if oob:
                f10b._merge_counts(counts, {"out_of_bounds": oob})
            counts["rows_kept"] = counts.get("rows_kept", 0) + len(t)
            cols = self.pack(t, lat, lon, vals, plat, qc)
            for lo in range(0, len(t), YIELD_ROWS):
                yield (f"{label} {d}",
                       {k: v[lo:lo + YIELD_ROWS] for k, v in cols.items()},
                       None)
        counts["fetch_seconds"] = round(time.time() - t0, 1)
        yield label, None, counts

    # ---------------------------------------------------------- contract ----
    def index(self, ctx):
        days, counts = self.listing(ctx)
        lo, hi = min(days), max(days)
        per_year, bytes_year = {}, {}
        for d, (_k, s) in days.items():
            per_year[str(d.year)] = per_year.get(str(d.year), 0) + 1
            bytes_year[str(d.year)] = bytes_year.get(str(d.year), 0) + s
        miss = gaps(days)
        out = {"dataset": ("ESA CCI Sea State v4 L3 along-track significant "
                           "wave height, daily multi-mission files "
                           f"({PRODUCT})"),
               "url": f"{S3_HOST}/{BUCKET}/{self.native_prefix(ctx)}",
               "stac": STAC,
               "version": DATASET,
               "files": len(days),
               "files_per_year": per_year,
               "bytes": int(sum(s for _k, s in days.values())),
               "bytes_per_year": bytes_year,
               "record": [str(lo), str(hi)],
               "record_days": (hi - lo).days + 1,
               "missing_days": [str(d) for d in miss],
               "missing_days_n": len(miss),
               "other_files": counts,
               "credentials": "none — the object store answers anonymously"}
        # ONE REAL FILE, downloaded and opened, checked against the declared
        # layout (ADAPTER_CONTRACT rule 4: the layout is a hypothesis).
        first = self._days_in(ctx.d_lo, ctx.d_hi, days) or [lo]
        key = days[first[0]][0]
        scratch = os.path.join(ctx.scratch, self.store)
        os.makedirs(scratch, exist_ok=True)
        dest = os.path.join(scratch, "index_" + os.path.basename(key))
        try:
            path, tmp = self._get(ctx, key, dest)
        except (IOError, f10b._NotFound) as e:
            sys.exit(f"REFUSING swh: the first file {key} could not be read: "
                     f"{e}")
        try:
            import netCDF4
            ds = netCDF4.Dataset(path)
            try:
                meta = nc_check(ds)
                meta["missions"] = sorted(
                    satellite_names(ds.variables["satellite"]).values())
            except FormatError as e:
                sys.exit(f"REFUSING swh: the first file {key} does not match "
                         f"the declared layout: {e}")
            finally:
                ds.close()
        finally:
            if tmp and os.path.exists(path):
                os.remove(path)
        meta["name"] = os.path.basename(key)
        meta["bytes"] = days[first[0]][1]
        out["first_file"] = meta
        return out

    def fetch_year(self, ctx, year):
        d_lo = max(dt.date(int(year), 1, 1), ctx.d_lo)
        d_hi = min(dt.date(int(year), 12, 31), ctx.d_hi)
        yield from self._rows(ctx, str(year), d_lo, d_hi, ctx.t_lo, ctx.t_hi)

    def fetch_month(self, ctx, year, month):
        lo, hi = f10b.month_bounds_s(year, month)
        lo, hi = max(lo, ctx.t_lo), min(hi, ctx.t_hi)
        last = (dt.date(int(year) + (int(month) == 12), int(month) % 12 + 1, 1)
                - dt.timedelta(days=1))
        d_lo = max(dt.date(int(year), int(month), 1), ctx.d_lo)
        d_hi = min(last, ctx.d_hi)
        yield from self._rows(ctx, f"{year}-{int(month):02d}", d_lo, d_hi,
                              lo, hi)

    # --------------------------------------------------------------- smoke --
    def smoke_sources(self, root, d_lo, d_hi, seed=20260918):
        truth = make_smoke_sources(root, d_lo, d_hi, seed)
        self.__init__()
        return truth


# ================================================================== smoke ==
SMOKE_PER_DAY = 40
SMOKE_MISSIONS = ("ers-1", "topex-poseidon")
# One day the archive does not hold at all: a real upstream gap, counted in
# `days_absent_upstream` and NOT an absence (the two runs of missing days in
# the real record are exactly this). A LISTED day that will not download is
# `ctx.note_absent`, and the test exercises that by removing a written file.
SMOKE_GAP = dt.date(1991, 8, 7)
SMOKE_OOB_DAY = dt.date(1991, 8, 5)


def _smoke_day(d, seed):
    """Deterministic samples for one day -> dict of columns (float64)."""
    rng = np.random.default_rng(seed + d.toordinal())
    n = SMOKE_PER_DAY
    epoch = dt.date(1981, 1, 1)
    t0 = (d - epoch).days * 86400.0
    # DISTINCT seconds, sorted: `check_smoke` orders the truth by (bin,
    # time_s) and breaks a tie by the adapter's own yield order, so a day
    # with two samples in the same second would make the comparison depend
    # on which of the two the sort happened to put first.
    t = t0 + np.sort(rng.choice(86400, n, replace=False)).astype(np.float64)
    lat = np.round(rng.uniform(-66.0, 66.0, n), 4)
    lon = np.round(rng.uniform(-180.0, 179.99, n), 4)
    swh = np.round(rng.uniform(0.2, 9.0, n), 4)
    den = np.round(swh + rng.normal(0.0, 0.05, n), 4)
    unc = np.round(rng.uniform(0.02, 0.4, n), 4)
    sat = (np.arange(n) % len(SMOKE_MISSIONS)).astype(np.uint8)
    # one sample with no time, one with no position, one with no SWH at all
    t[0] = FILL
    lat[1] = FILL
    swh[2] = FILL
    den[2] = FILL
    if d == SMOKE_OOB_DAY:
        swh[3] = 99.0                      # out of bounds -> NaN, counted
    return {"time": t, "lat": lat, "lon": lon, "swh": swh,
            "swh_denoised": den, "swh_uncertainty": unc, "satellite": sat}


def write_day(path, cols):
    import netCDF4
    ds = netCDF4.Dataset(path, "w", format="NETCDF4")
    ds.createDimension(TIME_DIM, len(cols["time"]))
    ds.featureType = "trajectory"
    ds.license = "ESA CCI Data Policy - free and open access"
    ds.key_variables = "swh_denoised"
    ds.product_version = "4.0"
    ds.platform = str(list(SMOKE_MISSIONS))
    ds.instrument = str(["RA", "Poseidon"])
    ds.spatial_resolution = str(["7 km", "6 km"])
    for name in ("time", "lat", "lon", "swh", "swh_denoised",
                 "swh_uncertainty"):
        v = ds.createVariable(name, "f8", (TIME_DIM,), fill_value=FILL)
        v[:] = cols[name]
    ds.variables["time"].units = "seconds since 1981-01-01"
    ds.variables["time"].calendar = "proleptic_gregorian"
    for name, unit in (("lat", "degrees_north"), ("lon", "degrees_east")):
        ds.variables[name].units = unit
    for name in ("swh", "swh_denoised", "swh_uncertainty"):
        ds.variables[name].units = "m"
    s = ds.createVariable("satellite", "u1", (TIME_DIM,))
    s[:] = cols["satellite"]
    s.flag_values = np.arange(len(SMOKE_MISSIONS), dtype=np.uint8)
    s.flag_meanings = " ".join(SMOKE_MISSIONS)
    ds.close()


def make_smoke_sources(root, d_lo, d_hi, seed=20260918):
    """The archive in its real layout — `<prefix>/<YYYY>/<MM>/<name>.nc` under
    `<root>/swh/` — and the truth rows `f10b.check_smoke` compares against.

    SMOKE_GAP is not written and not listed (a real upstream hole);
    SMOKE_ABSENT is listed by the walker only if written, so it is instead
    exercised by the `skip`-style test that removes it after the fact. The
    truth is ordered the way the store breaks ties: day ascending, then the
    file's own sample order.
    """
    base = os.path.join(root, "swh", PREFIX.rstrip("/"))
    truth = []
    d = d_lo
    while d <= d_hi:
        if d == SMOKE_GAP:
            d += dt.timedelta(days=1)
            continue
        cols = _smoke_day(d, seed)
        dirn = os.path.join(base, f"{d.year:04d}", f"{d.month:02d}")
        os.makedirs(dirn, exist_ok=True)
        name = (f"ESACCI-SEASTATE-L3-SWH-MULTI_1D-{d.strftime('%Y%m%d')}"
                f"-fv01.nc")
        write_day(os.path.join(dirn, name), cols)
        epoch_off = (dt.date(1981, 1, 1) - dt.date(1982, 1, 1)).days * 86400
        for i in range(len(cols["time"])):
            t = cols["time"][i]
            lat, lon = cols["lat"][i], cols["lon"][i]
            v = [cols["swh"][i], cols["swh_denoised"][i],
                 cols["swh_uncertainty"][i]]
            if t == FILL or lat == FILL or lon == FILL:
                continue
            if v[0] == FILL and v[1] == FILL:
                continue
            vv = []
            for (nm, _u, lo, hi), x in zip(CHANNELS, v):
                vv.append(np.nan if (x == FILL or not (lo <= x <= hi))
                          else float(x))
            truth.append({"t": int(round(t + epoch_off)),
                          "lat": float(lat), "lon": float(lon),
                          "platform": f10b.platform_hash(
                              SMOKE_MISSIONS[int(cols["satellite"][i])]),
                          "v": vv})
        d += dt.timedelta(days=1)
    return truth


ADAPTER = SWHAdapter

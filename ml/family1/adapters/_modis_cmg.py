"""The MODIS Climate Modeling Grid machinery `lst05`, `snow05` and `refl05`
share (family 1.0.tf, E-082 wave 4). Underscore-prefixed, so the registry
skips it (`_common.py`, `_gesla.py`, `_nc3.py` are the precedents).

PLAIN ENGLISH. Three of family 1.0.tf's daily fields are MODIS products on
the same grid, from the same archives, in the same file format, found the same
way: land-surface temperature (MOD11C1), snow cover (MOD10C1) and surface
reflectance (MOD09CMG). Each is ONE HDF-EOS2 file per calendar day on the
0.05-degree Climate Modeling Grid, behind NASA's Earthdata Login. What differs
between them is which scientific datasets they read out of the file and what
the numbers mean; everything else — finding the day's granule, logging in,
downloading it, checking the grid, filing the frame under its five-day bin —
is here, once.

FINDING THE FILES: CMR, NOT A URL PATTERN. The granule's URL carries the
PRODUCTION timestamp (`MOD11C1.A2015182.061.2021358223019.hdf`), which is not
derivable from the date, so a URL pattern cannot be guessed — the contract's
rule 4, in its strongest form. NASA's Common Metadata Repository lists every
granule of a collection with its https link and its byte size, needs NO
account, and pages 2,000 at a time behind the `CMR-Search-After` header.
MEASURED 2026-09-18 from this sandbox, keyless:

  MOD11C1  v061  LPCLOUD      9,587 granules, 2000-02-27 -> , ~40 MB each
  MOD10C1  v61   NSIDC_CPRD   9,634 granules, 2000-02-24 -> , ~4 MB each
  MOD09CMG v061  LPCLOUD      9,622 granules, 2000-02-24 -> , ~540 MB each
  MYD11C1  v061  LPCLOUD      8,815 granules, 2002-07-04 ->  (Aqua)

The whole MOD10C1 listing came back in 12.7 s and 39 MB of JSON.

DOWNLOADING THEM: EARTHDATA LOGIN THROUGH A .netrc, IPv4 ONLY. Every data
link is a `-protected/` object that redirects to
`urs.earthdata.nasa.gov/oauth/authorize`; `requests` re-applies netrc
credentials per redirect hop (`Session.rebuild_auth`) and strips the
Authorization header on any cross-host hop, so with a `.netrc` naming
`urs.earthdata.nasa.gov` ALONE the password reaches Earthdata Login and
nothing else. That netrc is written by `family1-build.yml` on hosted runners
only, mode 600, and deleted at the end of the job (ml/CLAUDE.md 6). Name
resolution is restricted to A records for the whole process
(`family1.earthdata_check.force_ipv4`): GitHub's runners resolve AAAA and have
no IPv6 route, which in run #3 read as `[Errno 101] Network is unreachable`
and looked exactly like an archive refusing the account (BUILD_LOG, "The
Earthdata account is GOOD").

READING THEM: pyhdf. These are HDF4 (HDF-EOS2) files and rasterio's GDAL has
NO HDF4 driver (measured in this sandbox: 155 drivers, HDF5 and netCDF among
them, no HDF4). `pyhdf` 0.11.7 publishes a manylinux wheel that bundles the
HDF4 library, so `pip install pyhdf` is all the runner needs; it is in
`family1-build.yml`'s install step beside rasterio.

THE GRID, DECLARED AND THEN CHECKED IN EVERY FILE: 7,200 x 3,600 at 0.05
degrees, EPSG:4326, row 0 the NORTHERNMOST (90 N), col 0 the westernmost
(180 W) — MODIS' CMG, whose HDF-EOS grid corners are (-180, 90) and
(180, -90). Every SDS this machinery reads is checked for that shape, and for
the `scale_factor` / `_FillValue` / `valid_range` the adapter declares; a file
whose attributes differ is a REFUSAL, not a silently rescaled frame.

THE FRAMES: F = 5 daily frames per five-day bin, `frame_seconds` = 86,400, so
frame f of bin b IS the calendar day b*5 + f after 1982-01-01 and no
frame_table is needed (this is the plain case chirps05's exception E4 is the
exception to). A day inside the record with no granule is `absent_upstream`;
before the first and after the last granule is `before_record` /
`after_record`; a granule the listing has and the server will not serve is
`ctx.note_absent`, never a silent gap.

A NOTE ON DUPLICATES. A CMR collection can list two granules for one day (a
reprocessing that did not retire its predecessor). The listing keeps the one
with the HIGHEST production timestamp, counts the rest as
`granules_superseded`, and records the fact in the index — it never silently
picks the first.
"""
import datetime as dt
import json
import os
import re
import sys
import threading
import time

import numpy as np

import build_family10_stores as f10b
from family1 import sharded as sh

CMR = "https://cmr.earthdata.nasa.gov/search/granules.json"
URS_HOST = "urs.earthdata.nasa.gov"
W, H = 7200, 3600
DX = 0.05
X0, Y0 = -180.0, 90.0           # the grid's WEST and NORTH edges
PAGE = 2000                     # CMR's maximum page_size
KM_EQ = 111.31949               # km per degree of longitude at the equator
BIN_KM = 27.83                  # the family's reference footprint (0.25 deg)
UA = {"User-Agent": "earth-science-pipeline/1.0 (research; github "
                    "blauewelt/earth)"}
TIMEOUT = 300
# A SEPARATE, SHORT CONNECT TIMEOUT. Measured on run #151 (the first refl05
# probe): `urs.earthdata.nasa.gov` refused to answer the TCP handshake and
# `requests` sat on it for the whole 300-second timeout, four times over, so
# eight minutes of the job went on one granule's login and the run died
# without measuring anything. A stalled CONNECT is exactly the failure a
# retry fixes, so it is given 30 seconds and the retry ladder does the
# waiting; the READ timeout stays long, because a 540 MB granule legitimately
# takes minutes.
CONNECT_TIMEOUT = 30

# pyhdf's HDF4 library is not thread-safe across datasets in every build, so
# every open/read/close holds this lock (chirps05 holds TIF_LOCK, seaice_asi
# NC_LOCK, for the same reason).
HDF_LOCK = threading.Lock()

GRANULE = re.compile(r"^(?P<prod>[A-Za-z0-9]+)\.A(?P<year>\d{4})(?P<doy>\d{3})"
                     r"\.(?P<ver>\d+)\.(?P<made>\d+)")


class FormatError(ValueError):
    """A listing or an HDF file that does not look like the archive measured."""


# ================================================================== grid ===
def cmg_grid(w=W, h=H):
    """The MODIS Climate Modeling Grid, JSON-able for `tile_grid.json`."""
    dx = 360.0 / int(w)
    dy = 180.0 / int(h)
    return {
        "H": int(h), "W": int(w), "pixel_deg": dx,
        "pixel_km_equator": round(dx * KM_EQ, 4),
        "crs": "EPSG:4326",
        "proj4": "+proj=longlat +datum=WGS84 +no_defs",
        "projection": {"grid_mapping_name": "latitude_longitude",
                       "semi_major_axis": 6378137.0,
                       "inverse_flattening": 298.257223563},
        "x0": X0, "dx": dx, "y0": Y0, "dy": -dy,
        "coordinates": ("lon = x0 + (col + 0.5) * dx, lat = y0 + (row + 0.5) "
                        "* dy with dy NEGATIVE; x0/y0 are the grid's WEST and "
                        "NORTH edges, which are MODIS' own CMG corners "
                        "(-180, 90) and (180, -90)"),
        "row_order": ("row 0 is the NORTHERNMOST row, 90 N — the order the "
                      "HDF-EOS SDS itself is stored in (YDim ascending "
                      "southward), so the store keeps the source's own "
                      "orientation and nothing is flipped"),
        "extent": [X0, -Y0, X0 + 360.0, Y0],
        "extent_note": "[west, south, east, north] in degrees; global",
        "no_tile_corners": ("a plain geographic grid: the affine rule above "
                            "IS the lat/lon of every pixel and every tile "
                            "corner, so no corner table is stored"),
    }


def log2_fp_for(w=W):
    """The footprint exponent of a `w`-column global grid, log2(km / 27.83)."""
    return float(np.log2(360.0 / int(w) * KM_EQ / BIN_KM))


# =================================================================== CMR ===
def granule_key(title):
    """`MOD11C1.A2015182.061.2021358223019[.hdf]` -> (date, production int).

    Raises `FormatError` on anything that is not a MODIS granule id: a title
    the parser cannot read is a refusal, never a skipped day.
    """
    m = GRANULE.match(str(title or ""))
    if not m:
        raise FormatError(f"{title!r}: not a MODIS granule id "
                          f"(<PROD>.AYYYYDDD.<ver>.<production>)")
    y, doy = int(m.group("year")), int(m.group("doy"))
    if not 1 <= doy <= 366:
        raise FormatError(f"{title!r}: day of year {doy}")
    try:
        d = dt.date(y, 1, 1) + dt.timedelta(days=doy - 1)
    except ValueError:
        raise FormatError(f"{title!r}: no such date") from None
    if d.year != y:
        raise FormatError(f"{title!r}: day {doy} is not in {y}")
    return d, int(m.group("made"))


def data_link(entry):
    """The one https `-protected/` .hdf link of a CMR granule entry."""
    best = None
    for ln in entry.get("links") or ():
        href = str(ln.get("href") or "")
        rel = str(ln.get("rel") or "")
        if not rel.endswith("/data#") or not href.startswith("https://"):
            continue
        if not href.endswith(".hdf"):
            continue
        if "protected" not in href:
            continue
        if best is None:
            best = href
    return best


def parse_cmr(entries):
    """CMR entries -> ({date: {...}}, counts). REFUSES a granule with no link.

    Two granules for one day keep the HIGHEST production timestamp and the
    other is counted `granules_superseded`; nothing is dropped silently.
    """
    out, counts = {}, {}
    for e in entries:
        title = e.get("title") or e.get("producer_granule_id") or ""
        d, made = granule_key(title)
        url = data_link(e)
        if not url:
            raise FormatError(
                f"{title}: no https '-protected/' .hdf data link in the CMR "
                f"entry — the archive moved and the adapter must be re-read "
                f"before anything is built")
        try:
            size = float(e.get("granule_size") or 0.0) * 1024 * 1024
        except (TypeError, ValueError):
            size = 0.0
        rec = {"title": str(title), "url": url, "bytes_cmr": int(size),
               "made": made,
               "time_start": e.get("time_start"), "id": e.get("id")}
        old = out.get(d)
        if old is None:
            out[d] = rec
            continue
        counts["granules_superseded"] = counts.get("granules_superseded", 0) + 1
        counts.setdefault("granules_superseded_days", [])
        if len(counts["granules_superseded_days"]) < 20:
            counts["granules_superseded_days"].append(str(d))
        if made > old["made"]:
            out[d] = rec
    return out, counts


def cmr_page(short_name, version, after=None, page=PAGE, attempts=4):
    """One CMR page -> (entries, next CMR-Search-After, hits, bytes read)."""
    import urllib.error
    import urllib.parse
    import urllib.request
    q = {"short_name": short_name, "version": version,
         "page_size": int(page), "sort_key": "start_date"}
    url = CMR + "?" + urllib.parse.urlencode(q)
    headers = dict(UA)
    if after:
        headers["CMR-Search-After"] = after
    err = None
    for i in range(max(1, attempts)):
        try:
            req = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(req, timeout=TIMEOUT) as r:
                raw = r.read()
                nxt = r.headers.get("CMR-Search-After")
                hits = int(r.headers.get("CMR-Hits", -1))
            f10b.count_bytes(len(raw))
            feed = json.loads(raw).get("feed") or {}
            return list(feed.get("entry") or ()), nxt, hits, len(raw)
        except (IOError, OSError, ValueError) as e:      # noqa: PERF203
            err = e
            if i < attempts - 1:
                time.sleep(3.0 * (2 ** i))
    raise IOError(f"{CMR} ({short_name} {version}): "
                  f"{type(err).__name__}: {err}")


def cmr_listing(short_name, version, attempts=4, page=PAGE):
    """Every granule of a collection -> ({date: rec}, counts). Paged."""
    out, counts = {}, {}
    after, pages, nbytes, hits = None, 0, 0, -1
    t0 = time.time()
    while True:
        entries, after, h, nb = cmr_page(short_name, version, after, page,
                                         attempts)
        if h >= 0:
            hits = h
        pages += 1
        nbytes += nb
        got, c = parse_cmr(entries)
        f10b._merge_counts(counts, c)
        for d, rec in got.items():
            old = out.get(d)
            if old is None or rec["made"] > old["made"]:
                if old is not None:
                    counts["granules_superseded"] = \
                        counts.get("granules_superseded", 0) + 1
                out[d] = rec
            elif old is not None:
                counts["granules_superseded"] = \
                    counts.get("granules_superseded", 0) + 1
        if not after or not entries:
            break
        if pages > 200:
            raise FormatError(f"{short_name}: CMR paged past {pages} pages — "
                              f"refusing to loop")
    counts["cmr_pages"] = pages
    counts["cmr_bytes"] = nbytes
    counts["cmr_hits"] = hits
    counts["cmr_seconds"] = round(time.time() - t0, 1)
    counts["granules_listed"] = len(out)
    return out, counts


# ============================================================= Earthdata ===
_IPV4 = {"done": False}


def force_ipv4_once():
    """Restrict this process to A records (see the module docstring)."""
    if _IPV4["done"]:
        return
    from family1 import earthdata_check as edc
    edc.force_ipv4(True)
    _IPV4["done"] = True


def earthdata_session():
    """A `requests` session that logs in to Earthdata through the .netrc.

    `trust_env = True` is the whole mechanism: `requests` looks the request's
    host up in `~/.netrc` on EVERY redirect hop and strips any Authorization
    header on a cross-host hop, so a netrc naming `urs.earthdata.nasa.gov`
    alone sends the password there and nowhere else. The credentials are never
    read, printed or stored by this code.
    """
    import requests
    force_ipv4_once()
    s = requests.Session()
    s.trust_env = True
    s.headers.update(UA)
    return s


def netrc_has_urs(home=None):
    """Does a netrc this process can see name Earthdata Login?"""
    p = os.environ.get("NETRC") or os.path.join(
        home or os.path.expanduser("~"), ".netrc")
    try:
        with open(p, "r") as fh:
            return URS_HOST in fh.read()
    except OSError:
        return False


def download(session, url, path, attempts=4, sleep=3.0):
    """GET `url` -> `path`, size-verified. (bytes, None) or (None, why).

    A definite 404 comes back as `(None, "notfound")` — a legitimate absence
    the caller counts. Anything else retries and then RAISES: a short or
    failed transfer is a refusal, never a smaller frame (the 2026-09-14 rule).
    """
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    part = f"{path}.part{os.getpid()}"
    err = None
    for i in range(max(1, attempts)):
        got = 0
        want = None
        try:
            with session.get(url, stream=True,
                             timeout=(CONNECT_TIMEOUT, TIMEOUT),
                             allow_redirects=True) as r:
                if r.status_code == 404:
                    return None, "notfound"
                if r.status_code in (401, 403):
                    raise IOError(
                        f"{url}: HTTP {r.status_code} after "
                        f"{len(r.history)} redirect(s), final host "
                        f"{_host(r.url)} — Earthdata Login refused this "
                        f"account for this archive. Run the workflow with "
                        f"check_credentials=true and read the approval URL "
                        f"it prints.")
                if r.status_code != 200:
                    raise IOError(f"{url}: HTTP {r.status_code}")
                cl = r.headers.get("Content-Length")
                want = int(cl) if cl is not None else None
                with open(part, "wb") as fh:
                    for chunk in r.iter_content(1 << 20):
                        if chunk:
                            fh.write(chunk)
                            got += len(chunk)
            if want is not None and got != want:
                raise IOError(f"{url}: {got:,} of {want:,} bytes — truncated")
            if got == 0:
                raise IOError(f"{url}: an empty body — a download that comes "
                              f"back empty is a refusal")
            f10b.count_bytes(got)
            os.replace(part, path)
            return got, None
        except Exception as e:                                  # noqa: BLE001
            err = e
            if os.path.exists(part):
                os.remove(part)
            if "Earthdata Login refused" in str(e):
                raise
            if i < attempts - 1:
                time.sleep(sleep * (2 ** i))
    raise IOError(f"{url}: {type(err).__name__}: {err}")


def _host(u):
    import urllib.parse
    return urllib.parse.urlparse(str(u or "")).hostname or "?"


# ================================================================== HDF4 ===
def _pyhdf():
    try:
        from pyhdf.SD import SD, SDC
    except ImportError:                                     # pragma: no cover
        sys.exit("the MODIS CMG adapters need the `pyhdf` package (pip "
                 "install pyhdf — the manylinux wheel bundles HDF4); it is in "
                 "the family1-build workflow's install step. rasterio's GDAL "
                 "has no HDF4 driver, which is why this is a separate "
                 "dependency.")
    return SD, SDC


def sds_names(path):
    """Every SDS name in an HDF4 file, sorted. Raises IOError on a bad file."""
    SD, _ = _pyhdf()
    try:
        d = SD(path)
    except Exception as ex:                                     # noqa: BLE001
        raise IOError(f"{os.path.basename(path)}: not a readable HDF4 file "
                      f"({ex}) — a truncated download") from None
    try:
        return sorted(d.datasets())
    finally:
        d.end()


def read_sds(path, names, w=W, h=H, want=None):
    """Read `names` out of one HDF4 file -> {name: raw array}, CHECKED.

    Every SDS must be 2-D `(h, w)`. `want` is `{name: {attr: value}}` of
    attributes the caller DECLARES; an attribute the file carries with a
    different value is a `FormatError` — a rescaled product is refused, not
    silently converted. Returns `(arrays, attributes)`.
    """
    SD, _ = _pyhdf()
    try:
        d = SD(path)
    except Exception as ex:                                     # noqa: BLE001
        raise IOError(f"{os.path.basename(path)}: not a readable HDF4 file "
                      f"({ex}) — a truncated download") from None
    out, attrs = {}, {}
    try:
        have = d.datasets()
        for n in names:
            if n not in have:
                raise FormatError(
                    f"{os.path.basename(path)}: no SDS {n!r} (the file has "
                    f"{sorted(have)[:12]}{' …' if len(have) > 12 else ''})")
            s = d.select(n)
            try:
                _, rank, dims, _, _ = s.info()
                if rank != 2 or tuple(dims) != (int(h), int(w)):
                    raise FormatError(
                        f"{os.path.basename(path)}: SDS {n!r} is rank {rank} "
                        f"{tuple(dims)}, the declared grid is "
                        f"{(int(h), int(w))}")
                a = s.attributes()
                attrs[n] = {k: a.get(k) for k in
                            ("_FillValue", "scale_factor", "valid_range",
                             "units", "long_name") if k in a}
                for k, v in ((want or {}).get(n) or {}).items():
                    if k not in a:
                        raise FormatError(
                            f"{os.path.basename(path)}: SDS {n!r} carries no "
                            f"{k!r}; the adapter declares {v!r}")
                    got = a[k]
                    same = (list(got) == list(v)
                            if isinstance(v, (list, tuple))
                            else abs(float(got) - float(v)) <= 1e-12)
                    if not same:
                        raise FormatError(
                            f"{os.path.basename(path)}: SDS {n!r} has "
                            f"{k} = {got!r}, the adapter declares {v!r} — a "
                            f"product whose scaling changed is refused, not "
                            f"rescaled")
                out[n] = np.asarray(s[:])
            finally:
                s.endaccess()
    finally:
        d.end()
    return out, attrs


def dump_all_sds(path, limit=48):
    """Every SDS's name, shape, dtype and attributes — no shape enforced.

    What a refusal prints, so one hosted round trip is enough to learn what
    the file really carries rather than one per wrong guess.
    """
    SD, _ = _pyhdf()
    d = SD(path)
    out = {}
    try:
        for n in sorted(d.datasets())[:limit]:
            s = d.select(n)
            try:
                _, rank, dims, nt, _ = s.info()
                out[n] = {"rank": rank, "dims": list(dims), "number_type": nt,
                          "attributes": {k: (list(v) if isinstance(
                              v, (list, tuple)) else v)
                              for k, v in s.attributes().items()}}
            finally:
                s.endaccess()
    finally:
        d.end()
    return out


def hdf_meta(path, names, w=W, h=H, want=None):
    """What the index stage records about ONE real granule."""
    with HDF_LOCK:
        arrs, attrs = read_sds(path, names, w, h, want)
        every = sds_names(path)
    out = {"sds_in_file": len(every), "sds_names_sample": every[:24],
           "sds_read": {}}
    for n, a in arrs.items():
        fill = (attrs[n] or {}).get("_FillValue")
        good = a if fill is None else a[a != np.asarray(fill, a.dtype)]
        out["sds_read"][n] = {
            "shape": list(a.shape), "dtype": str(a.dtype),
            "attributes": {k: (list(v) if isinstance(v, (list, tuple)) else v)
                           for k, v in (attrs[n] or {}).items()},
            "fill_pixels": (None if fill is None
                            else int((a == np.asarray(fill, a.dtype)).sum())),
            "valid_fraction": (None if fill is None
                               else round(float(good.size / a.size), 6)),
            "raw_min": (int(good.min()) if good.size else None),
            "raw_max": (int(good.max()) if good.size else None)}
    return out


def mask_bounds_low_memory(ad, arr):
    """`ad.mask_bounds` channel by channel, on the [H, W, C] array in place.

    The framework's own `mask_bounds` takes an (N, C) float64 view; for a
    25.9-million-pixel frame with nine channels that is a 1.9 GB temporary
    per comparison. This loops the channels and compares against PYTHON
    floats, so no temporary is larger than one channel's bool mask, and it
    returns the identical `{channel: n}` dict.
    """
    lo, hi = ad.bounds()
    out = {}
    for i, name in enumerate(ad.channel_names):
        v = arr[:, :, i]
        with np.errstate(invalid="ignore"):
            bad = np.isfinite(v) & ((v < float(lo[i])) | (v > float(hi[i])))
        n = int(bad.sum())
        if n:
            v[bad] = np.nan
            out[name] = n
    return out


# ============================================================ the adapter ==
class CmgAdapter(sh.GridAdapter):
    """A daily MODIS CMG field: one HDF-EOS2 granule a day on the 0.05 grid.

    A subclass declares `short_name`, `version`, `channels`, `dtype`, the
    `sds` it reads (in channel order where that is one-to-one) and the
    attributes it expects (`sds_want`), and implements `frame_from(arrs)` ->
    a float32 `[H, W, C]` array with NaN for not-measured. Everything else —
    the CMR listing, the login, the download, the grid check, the frame
    filing, the smoke's local-archive mode — is inherited.
    """

    family = "1tf"
    distribution = "public"
    time_dtype = "int32"
    credentials = ("EARTHDATA_USERNAME", "EARTHDATA_PASSWORD")
    per_year = True
    frames_per_bin = 5
    frame_seconds = 86400
    tile = sh.TILE
    zstd_level = sh.DEFAULT_LEVEL
    log2_fp = log2_fp_for(W)                      # -2.3219
    log2_dt = float(np.log2(1.0 / 5.0))           # a DAY inside a five-day bin

    short_name = ""
    version = ""
    sds = ()
    sds_want = {}
    smoke_grid_env = ""
    groups_env = ""              # e.g. "LST05_GROUPS=terra,aqua"
    WORKERS = 3

    # {group: (short_name, version)}. A store's continuation instruments (Aqua
    # for lst05, VIIRS for all three) are SEPARATE GROUPS on the identical
    # grid, so the layout carries them side by side and a reader asks for the
    # one it wants. `groups_available` is every group the adapter knows how to
    # build; `groups` (from `groups_env`, default the first) is what THIS
    # instance builds, because a group doubles the store.
    groups_available = {}
    groups_default = ()

    def __init__(self):
        g = (os.environ.get(self.smoke_grid_env) or "").split(",")
        self.w, self.h = (int(g[0]), int(g[1])) if len(g) == 2 else (W, H)
        self.groups = self._pick_groups()
        self._listing = {}
        self._session = None
        if (self.w, self.h) != (W, H):
            self.notes = (f"{self.notes}\nSMOKE GRID: {self.smoke_grid_env} "
                          f"set the grid to {self.w} x {self.h} instead of "
                          f"{W} x {H}. This is a synthetic store.")
        if tuple(self.groups) != tuple(self.groups_default):
            self.notes = (f"{self.notes}\nGROUPS: {self.groups_env} selected "
                          f"{list(self.groups)} instead of the default "
                          f"{list(self.groups_default)}.")

    def _pick_groups(self):
        avail = self.groups_available or {self.store: (self.short_name,
                                                       self.version)}
        want = (os.environ.get(self.groups_env) or "").strip()
        if not want:
            return list(self.groups_default or [sorted(avail)[0]])
        picked = [g.strip() for g in want.split(",") if g.strip()]
        bad = [g for g in picked if g not in avail]
        if bad or not picked:
            sys.exit(f"REFUSING {self.store}: {self.groups_env}={want!r} names "
                     f"{bad or 'nothing'}; the groups this adapter can build "
                     f"are {sorted(avail)}")
        return picked

    def collection(self, group):
        avail = self.groups_available or {self.store: (self.short_name,
                                                       self.version)}
        return avail[group]

    # ------------------------------------------------------------ the grid --
    @property
    def grid(self):
        return cmg_grid(self.w, self.h)

    @property
    def grids(self):
        return {g: cmg_grid(self.w, self.h) for g in self.groups}

    # ------------------------------------------------------------- listing --
    def local_listing_path(self, ctx, group):
        return os.path.join(ctx.source_dir, self.store, f"cmr_{group}.json")

    def local_granule_path(self, ctx, group, rec):
        return os.path.join(ctx.source_dir, self.store, "granules", group,
                            rec["title"] if rec["title"].endswith(".hdf")
                            else rec["title"] + ".hdf")

    def listing(self, ctx, group=None):
        """{date: granule record}, counts — from CMR, or the smoke's copy."""
        group = group or self.groups[0]
        if group in self._listing:
            return self._listing[group]
        sn, ver = self.collection(group)
        if ctx.source_dir:
            p = self.local_listing_path(ctx, group)
            if not os.path.exists(p):
                sys.exit(f"REFUSING {self.store}: no {p} — the smoke's "
                         f"synthetic CMR listing is missing")
            with open(p, "rb") as fh:
                raw = fh.read()
            ctx.count_bytes(len(raw))
            entries = (json.loads(raw).get("feed") or {}).get("entry") or []
            try:
                files, counts = parse_cmr(entries)
            except FormatError as e:
                sys.exit(f"REFUSING {self.store}: {p}: {e}")
            counts["granules_listed"] = len(files)
            counts["cmr_pages"] = 0
        else:
            try:
                files, counts = cmr_listing(sn, ver,
                                            attempts=ctx.a.attempts)
            except FormatError as e:
                sys.exit(f"REFUSING {self.store}: the CMR listing of "
                         f"{sn} v{ver}: {e}")
        if not files:
            sys.exit(f"REFUSING {self.store}: CMR lists no granule of "
                     f"{sn} v{ver} at all — an empty listing is a refusal "
                     f"(ml/CLAUDE.md, the 2026-09-14 rule)")
        self._listing[group] = (files, counts)
        return self._listing[group]

    def record(self, ctx, group=None):
        files, _ = self.listing(ctx, group)
        return min(files), max(files)

    def record_frames(self, ctx, group):
        files, _ = self.listing(ctx, group)
        return len(files)

    def session(self):
        if self._session is None:
            self._session = earthdata_session()
        return self._session

    def fetch_preflight(self, ctx):
        """The netrc must be there BEFORE the first granule (§0.3)."""
        if ctx.source_dir:
            return None
        if not netrc_has_urs():
            print(f"::warning::{self.store}: no netrc entry for {URS_HOST} "
                  f"was found; the download will fall back to whatever "
                  f"credentials the environment provides and may be refused. "
                  f"family1-build.yml writes that file on hosted runners.")
        return None

    # --------------------------------------------------------------- index --
    def index(self, ctx):
        out = {"dataset": f"{self.short_name} v{self.version} daily "
                          f"0.05-degree CMG (HDF-EOS2)",
               "url": CMR + f"?short_name={self.short_name}"
                            f"&version={self.version}",
               "groups": {}, "files": 0, "files_per_year": {},
               "frames_per_bin": self.frames_per_bin,
               "frame_seconds": self.frame_seconds,
               "grid": self.grid}
        lo_all = hi_all = None
        for group in self.groups:
            sn, ver = self.collection(group)
            files, counts = self.listing(ctx, group)
            lo, hi = min(files), max(files)
            lo_all = lo if lo_all is None else min(lo_all, lo)
            hi_all = hi if hi_all is None else max(hi_all, hi)
            per_year, gaps = {}, []
            d = lo
            while d <= hi:
                if d in files:
                    per_year[str(d.year)] = per_year.get(str(d.year), 0) + 1
                else:
                    gaps.append(str(d))
                d += dt.timedelta(days=1)
            g = {"short_name": sn, "version": ver,
                 "files": len(files),
                 "files_per_year": per_year,
                 "record": [str(lo), str(hi)],
                 "days_in_record": (hi - lo).days + 1,
                 "days_absent_upstream": len(gaps),
                 "days_absent_upstream_sample": gaps[:40],
                 "bytes_cmr": int(sum(r["bytes_cmr"]
                                      for r in files.values())),
                 "bytes_note": ("CMR's own `granule_size` in MB, summed; the "
                                "probe measures exact bytes"),
                 "listing": counts,
                 "bin_first": f10b.seconds_since_epoch(lo) // sh.BIN_SECONDS,
                 "bin_last": f10b.seconds_since_epoch(hi) // sh.BIN_SECONDS}
            # ONE REAL GRANULE PER GROUP, downloaded and read against the
            # declared grid and the declared SDS attributes
            g["first_file"] = self.read_first_file(ctx, group, files)
            out["groups"][group] = g
            out["files"] += len(files)
            for k, v in per_year.items():
                out["files_per_year"][k] = out["files_per_year"].get(k, 0) + v
        out["record"] = [str(lo_all), str(hi_all)]
        out["record_days"] = out["record"]
        return out

    def index_probe_days(self, files):
        """Which listed days the index stage tries to open, in order.

        The NEWEST first: the newest granule is the one whose format a future
        build will meet. The oldest and the middle one are the fallbacks,
        because Earthdata Login refuses a runner's handshake for minutes at a
        time (#164, #167) and a store must not be un-indexable because one
        granule's login happened to be the one that stalled.
        """
        days = sorted(files)
        return [days[-1], days[0], days[len(days) // 2]]

    def read_first_file(self, ctx, group, files):
        """One real granule, read — and a NETWORK failure is not a verdict.

        A file that IS read and does not match the declared grid or attributes
        is fatal (`read_header` exits): that is a definite answer about the
        product. A login that does not answer is not an answer at all
        (ml/CLAUDE.md §5.17), and it stopped two runs dead at the index stage
        while the same account was downloading granules successfully in
        another job. So the three candidate days are tried in turn, and if
        every one of them fails on the NETWORK the index records why and the
        stage goes on — the fetch will meet the same login within the minute,
        where a failure is definite and is a refusal.
        """
        tried = []
        for day in self.index_probe_days(files):
            try:
                meta = self.read_header(ctx, group, files[day])
            except (IOError, OSError) as e:
                tried.append({"date": str(day),
                              "error": f"{type(e).__name__}: "
                                       f"{str(e)[:300]}"})
                print(f"::warning::{self.store}: the index could not open "
                      f"{files[day]['title']} ({type(e).__name__}) — trying "
                      f"another day; a login that does not answer is not a "
                      f"verdict about the product")
                continue
            meta["date"] = str(day)
            if tried:
                meta["days_that_would_not_open"] = tried
            return meta
        print(f"::warning::{self.store}: NO granule could be opened at index "
              f"time ({tried}) — the fetch stage meets the same login, where "
              f"a failure is definite")
        return {"read": False, "days_that_would_not_open": tried,
                "note": ("a network failure at index time is not a verdict "
                         "about the product (ml/CLAUDE.md §5.17); the grid "
                         "and every declared SDS attribute are re-checked in "
                         "EVERY granule the fetch reads")}

    def read_header(self, ctx, group, rec):
        import shutil
        tmpdir = None
        try:
            p, why = self.granule_file(ctx, group, rec)
            if p is None:
                sys.exit(f"REFUSING {self.store}: the granule "
                         f"{rec['title']} the listing names could not be "
                         f"read: {why}")
            tmpdir = why if not ctx.source_dir else None
            try:
                meta = hdf_meta(p, list(self.sds), self.w, self.h,
                                self.sds_want)
            except (FormatError, IOError) as e:
                # ONE ROUND TRIP IS ENOUGH TO LEARN EVERYTHING. A declared
                # attribute that does not match is a refusal, and the refusal
                # prints what the file really says — otherwise every wrong
                # guess costs another hosted run to find out the next one.
                try:
                    with HDF_LOCK:
                        table = dump_all_sds(p)
                    print("::notice::what the file actually carries:\n"
                          + json.dumps(table, indent=1, default=str))
                except Exception as e2:                         # noqa: BLE001
                    print(f"::warning::could not dump the file's own SDS "
                          f"table either: {type(e2).__name__}: {e2}")
                sys.exit(f"REFUSING {self.store}: {rec['title']} does not "
                         f"match the declared grid or attributes: {e}")
            meta["title"] = rec["title"]
            meta["url"] = rec["url"]
            meta["bytes"] = os.path.getsize(p)
            return meta
        finally:
            if tmpdir:
                shutil.rmtree(tmpdir, ignore_errors=True)

    # ----------------------------------------------------------- one file ---
    def granule_file(self, ctx, group, rec):
        """The granule on local disk -> (path, tmpdir or why)."""
        import tempfile
        if ctx.source_dir:
            p = self.local_granule_path(ctx, group, rec)
            if not os.path.exists(p):
                return None, f"{rec['title']}: listed and absent"
            ctx.count_bytes(os.path.getsize(p))
            return p, None
        tmpdir = tempfile.mkdtemp(prefix=f"{self.store}_", dir=ctx.scratch)
        name = os.path.basename(rec["url"].split("?")[0])
        p = os.path.join(tmpdir, name)
        try:
            # AT LEAST SIX ATTEMPTS, whatever --attempts says. A lane is
            # hours long and Earthdata Login stalls for a minute at a time
            # from runner IPs (#151); losing the year to one refused
            # handshake costs far more than fifteen minutes of backoff.
            n, why = download(self.session(), rec["url"], p,
                              attempts=max(int(ctx.a.attempts), 6))
        except Exception:
            import shutil
            shutil.rmtree(tmpdir, ignore_errors=True)
            raise
        if n is None:
            import shutil
            shutil.rmtree(tmpdir, ignore_errors=True)
            return None, f"{rec['url']}: listed, and {why}"
        return p, tmpdir

    # ----------------------------------------------------- the frame -------
    def frame_from(self, arrs, attrs):
        """{sds: raw array} -> float32 [h, w, C] with NaN for not-measured."""
        raise NotImplementedError

    def read_frame(self, ctx, path):
        """One granule -> (float32 [h, w, C], counts, out_of_bounds)."""
        with HDF_LOCK:
            arrs, attrs = read_sds(path, list(self.sds), self.w, self.h,
                                   self.sds_want)
        a, counts = self.frame_from(arrs, attrs)
        del arrs
        oob = mask_bounds_low_memory(self, a)
        counts = dict(counts or {})
        counts["files_read"] = 1
        counts["valid_pixels"] = int(np.isfinite(a).sum())
        return a, counts, oob

    # ------------------------------------------------------- the contract --
    def classify(self, ctx, group, b, f):
        """-> (date or None, reason or None) for one daily slot."""
        files, _ = self.listing(ctx, group)
        lo, hi = min(files), max(files)
        day = sh.frame_day(b, f, self.frame_seconds)
        if day < lo:
            return None, "before_record"
        if day > hi:
            return None, "after_record"
        if day not in files:
            return None, "absent_upstream"
        return day, None

    def fetch_frames(self, ctx, wanted):
        from family1.adapters import _common as cm
        jobs = []
        for (g, b, f) in wanted:
            day, why = self.classify(ctx, g, b, f)
            jobs.append((g, b, f, day, why))
        todo = [j for j in jobs if j[4] is None]

        def work(job):
            g, b, f, day, why = job
            try:
                files, _ = self.listing(ctx, g)
                p, tmpdir = self.granule_file(ctx, g, files[day])
                return job, p, tmpdir, None
            except Exception as e:                              # noqa: BLE001
                return job, None, None, e

        workers = 1 if ctx.source_dir else self.WORKERS
        it = cm.ordered_map(work, todo, workers, lookahead=workers + 1)
        t0 = time.time()
        for (g, b, f, day, why) in jobs:
            if why is not None:
                yield g, b, f, None, {"frame_missing": why}
                continue
            job, p, tmpdir, err = next(it)
            if job[:3] != (g, b, f):
                sys.exit(f"{self.store}: the download pool answered "
                         f"{job[:3]} where {(g, b, f)} was asked for")
            if err is not None:
                raise err
            files, _ = self.listing(ctx, g)
            try:
                if p is None:
                    ctx.note_absent(f"{g} {day}", tmpdir)
                    continue
                try:
                    arr, counts, oob = self.read_frame(ctx, p)
                except FormatError as e:
                    sys.exit(f"REFUSING {self.store}: {files[day]['title']}: "
                             f"{e}")
                except (IOError, OSError) as e:
                    ctx.note_absent(f"{g} {day}", f"{type(e).__name__}: {e}")
                    continue
                counts["bytes_files"] = os.path.getsize(p)
            finally:
                if tmpdir:
                    import shutil
                    shutil.rmtree(tmpdir, ignore_errors=True)
            if oob:
                counts["out_of_bounds"] = oob
            counts["fetch_seconds"] = round(time.time() - t0, 2)
            t0 = time.time()
            yield g, b, f, arr, counts

    # --------------------------------------------------------------- smoke --
    def smoke_reinit(self):
        self.__init__()


# ============================================================ smoke help ===
def cmr_json(records):
    """A CMR granules.json response, in the service's own shape.

    `records` is [(title, url, size MB)]. The smoke writes this so the REAL
    parser (`parse_cmr`) is what the test exercises.
    """
    entries = []
    for i, (title, url, mb) in enumerate(records):
        d, _made = granule_key(title)
        entries.append({
            "producer_granule_id": title, "title": title,
            "id": f"G{1000 + i}-TEST",
            "granule_size": f"{float(mb)}",
            "time_start": f"{d}T00:00:00.000Z",
            "time_end": f"{d}T23:59:59.000Z",
            "day_night_flag": "BOTH",
            "links": [
                {"rel": "http://esipfed.org/ns/fedsearch/1.1/data#",
                 "href": url, "hreflang": "en-US"},
                {"rel": "http://esipfed.org/ns/fedsearch/1.1/s3#",
                 "href": "s3://bucket/" + os.path.basename(url)},
                {"rel": "http://esipfed.org/ns/fedsearch/1.1/browse#",
                 "href": "https://example.invalid/browse.jpg"},
            ]})
    return {"feed": {"updated": "2026-09-18T00:00:00.000Z",
                     "id": CMR, "title": "ECHO granule metadata",
                     "entry": entries}}


def write_smoke_archive(root, store, group, days, arrays, w, h, url_prefix,
                        short_name, version, attrs=None, absent=(), skip=()):
    """The synthetic archive in its real layout: `cmr_<group>.json` and
    `granules/<group>/*.hdf`.

    `arrays` is {date: {sds: raw array}}. A date in `absent` is LISTED and its
    file is not written (a download that 404s); a date in `skip` is neither
    listed nor written (a day the archive does not have).
    """
    SD, SDC = _pyhdf()
    base = os.path.join(root, store)
    gdir = os.path.join(base, "granules", group)
    os.makedirs(gdir, exist_ok=True)
    recs = []
    for d in sorted(days):
        if d in skip:
            continue
        doy = (d - dt.date(d.year, 1, 1)).days + 1
        title = (f"{short_name}.A{d.year:04d}{doy:03d}.{version}."
                 f"{2026000000000 + doy}")
        name = title + ".hdf"
        recs.append((title, url_prefix + name, 1.0))
        if d in absent:
            continue
        p = os.path.join(gdir, name)
        if os.path.exists(p):
            os.remove(p)
        f = SD(p, SDC.WRITE | SDC.CREATE)
        for sname, a in arrays[d].items():
            t = {np.dtype("uint16"): SDC.UINT16, np.dtype("int16"): SDC.INT16,
                 np.dtype("uint8"): SDC.UINT8}[a.dtype]
            s = f.create(sname, t, (int(h), int(w)))
            s[:] = a
            s.endaccess()
        f.end()
        if attrs:
            set_sds_attrs(p, attrs)
    with open(os.path.join(base, f"cmr_{group}.json"), "w") as fh:
        json.dump(cmr_json(recs), fh)
    return recs


def set_sds_attrs(path, attrs):
    """Give an already-written HDF4 file its SDS attributes."""
    SD, SDC = _pyhdf()
    d = SD(path, SDC.WRITE)
    try:
        for n, kv in attrs.items():
            s = d.select(n)
            try:
                for k, v in kv.items():
                    # `_FillValue` is a PREDEFINED HDF attribute: a plain
                    # setattr is silently ignored and `setfillvalue` is what
                    # writes it (measured with pyhdf 0.11.7).
                    if k == "_FillValue":
                        s.setfillvalue(v)
                    else:
                        setattr(s, k, v)
            finally:
                s.endaccess()
    finally:
        d.end()

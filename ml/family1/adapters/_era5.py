"""ERA5 on pressure levels — family 1.2's atmosphere (E-085; sharded tier G).

PLAIN ENGLISH. ERA5 is ECMWF's reanalysis: a weather model run over the whole
observational record, nudged every twelve hours towards every observation the
centre holds, so its output is a complete, physically consistent picture of
the atmosphere at every hour since 1940. Family 1.2 is family 1.gf (the global
fine OBSERVATION stores, inherited unchanged) plus three channels a model of
the climate cannot get from observations at a global scale: the air
TEMPERATURE, the SPECIFIC HUMIDITY (grams of water vapour per kilogram of
air) and the WIND (its eastward `u` and northward `v` components), each on the
13 standard pressure levels from 50 hPa (about 20 km up) to 1000 hPa (the
surface). One store per variable, the 13 levels folded into the channel axis:

    era5_t   t_50 .. t_1000    K
    era5_q   q_50 .. q_1000    g/kg (= 1000 x the archive's kg/kg; see below)
    era5_u   u_50 .. u_1000    m/s
    era5_v   v_50 .. v_1000    m/s

Every frame is ONE INSTANT of the analysis at 00, 06, 12 or 18 UTC — six
hours apart, 20 frames per five-day bin — on a 1-degree grid, point-aligned
like family 7's `g100`: row 0 is latitude -90, row 180 is +90, column 0 is
longitude -180, column 359 is +179. The value at a point is the area-weighted
(CONSERVATIVE) mean of ERA5's 0.25-degree field over the 1-degree box centred
on it, clipped at the poles. That rule is stated in `tile_grid.json`.

THIS IS AN EXCEPTION TO 1.gf's RULE, NUMBERED E4. Family 1.gf admits
observations at <= 10 km and <= 5 days. ERA5 is a model-filled reanalysis at
about 31 km native resolution (spectral TL639), stored here as 1-degree box
means; it is admitted by decision (Chris, 2026-10-05) because the upper air is
otherwise unobserved at global scale, and `family12.json` names the exception.

THE SOURCE — Google's analysis-ready copy of ERA5 on public Cloud Storage
(`gcp-public-data-arco-era5`, read anonymously over plain HTTPS; no gcloud, no
credentials). MEASURED 2026-10-05 from the sandbox, by reading each zarr's own
consolidated metadata and axes rather than its documentation:

  A  `ar/1959-2022-1h-360x181_equiangular_with_poles_conservative.zarr`
     HOURLY, the 1-degree with-poles grid (latitude -90..90 ascending, 181;
     longitude 0..359, 360), 37 pressure levels, conservatively regridded from
     0.25 degrees by the archive's producer. float32 `(time, level,
     longitude, latitude)` chunked [8 hours, 37, 360, 181], blosc-lz4 with
     byte shuffle. time = hours since 1959-01-01, 552,264 steps, contiguous,
     1959-01-01T00 .. 2021-12-31T23.
  B  `ar/full_37-1h-0p25deg-chunk-1.zarr-v3`
     HOURLY, 0.25 degrees (latitude 90..-90 DESCENDING, 721; longitude
     0..359.75, 1440), the same 37 levels, float32 `(time, level, latitude,
     longitude)` chunked [1 hour, 37, 721, 1440], blosc-lz4 + shuffle. Its
     time axis is pre-allocated 1900-01-01 .. 2050-12-31; the ROOT ATTRIBUTES
     say how far it is real: `valid_time_start` 1940-01-01,
     `valid_time_stop` 2026-06-30 (final ERA5) and `valid_time_stop_era5t`
     2026-09-29 (the preliminary ERA5T, which ECMWF may still revise).

  Frames BEFORE the seam come from A, unchanged (a reindex: rows already
  south-first, columns rolled so -180 is first). Frames FROM the seam come
  from B, regridded HERE to the same grid by the same rule. THE SEAM is the
  first hour A does not hold — read from A's own time axis, 2022-01-01T00 —
  and frames after B's `valid_time_stop` (its last FINAL hour) are
  `after_record`: ERA5T is not admitted.

  THE TWO RULES AGREE, MEASURED. At 2015-01-01T00 and 2021-12-31T18, this
  module's regrid of B against A over every cell and level: temperature
  within 1.4e-3 K, humidity within 1.3e-4 g/kg, wind within 3.0e-3 m/s —
  float32-accumulation noise in the producer's regrid — and 99.0-99.99 % of
  values identical once rounded to float16 (`ml/family1/era5_check.py`,
  recorded in the probe reports).

READING ONLY THE BYTES THE STORE NEEDS. A chunk of A holds 8 hours x 37
levels; the store wants 1 or 2 of those hours and 13 of those levels — 6 % of
the chunk. Blosc compresses a chunk as independent BLOCKS (512 KiB of the
uncompressed C-order buffer each, measured) and its header lists where every
block starts, so this module reads the 4 KiB header, works out which blocks
hold the wanted (hour, level) slabs, fetches only those with ranged GETs
(adjacent blocks coalesced into one request), and decodes them by
re-wrapping them as a small, valid blosc chunk handed to `numcodecs` — the
only decoder used, so the decoding is c-blosc's own. Measured on real chunks:
bit-identical to decoding the whole chunk. A short or failed read, a 200
where a 206 was asked for, a header that disagrees with the zarr's
declaration, or a block table that is not increasing is a REFUSAL (the
frame's input is noted absent and its year is not marked), never a skip.

ERA5's OWN SMALL NEGATIVE HUMIDITIES ARE KEPT. The analysis carries a few
slightly negative values in the tropical upper troposphere (January 2015:
11 of 10.1 million, the lowest -0.0042 g/kg, at 100-200 hPa). They are
ERA5's numbers, not corruption, so `q`'s lower bound is -0.01 g/kg rather
than 0; a consumer that needs q >= 0 clips them itself.

SPECIFIC HUMIDITY IS STORED IN g/kg. The archive's kg/kg runs from ~1e-6 in
the stratosphere to ~0.02 at the surface, and float16's smallest NORMAL
number is 6.1e-5: in kg/kg every stratospheric value would be a subnormal
with one or two significant bits. Multiplying by 1000 (exact in float64 for
a float32 input) moves the whole range into float16's normal range — 1e-3 to
25 g/kg — where the relative error is at most 2^-11 (0.049 %). The probe
measures the error per level; anything that would still be subnormal is
counted (`f16_subnormal_values`).

PRESSURE LEVELS UNDER THE GROUND are ERA5's own extrapolation (1000 hPa over
Tibet, 850 hPa over the Andes): the archive fills them, they are stored as
given, and a consumer that cares masks them with surface pressure, which this
store does not hold. Valid fraction is therefore 1.0 unless an input is NaN.

THE REGRID (used for B only; A is the producer's own): separable
conservative weights from the archives' own coordinate axes — latitude cell
edges at the midpoints between points, clipped to +/-90, overlap measured in
sin(latitude) (area); longitude edges at the midpoints, periodic, overlap in
degrees — each target row normalised to sum to 1. Applied as explicit
weighted sums over the (at most a handful of) overlapping source rows and
columns, in a fixed order, with NO matrix library: the result is the same
bytes on every CPU (ADAPTER_CONTRACT, "Reproducibility"). A NaN in any
contributing source cell makes the target cell NaN (counted), never a
partial mean.
"""
import datetime as dt
import json
import math
import os
import re
import struct
import sys
import threading
import time

import numpy as np

import build_family10_stores as f10b
from family1 import sharded as sh
from family1.adapters import _common as cm

GCS = "https://storage.googleapis.com/gcp-public-data-arco-era5/ar"
ZARR_A = "1959-2022-1h-360x181_equiangular_with_poles_conservative.zarr"
ZARR_B = "full_37-1h-0p25deg-chunk-1.zarr-v3"
#: under `--source-dir`, the archive is laid out as `<dir>/arco-era5/ar/<zarr>`
SOURCE_SUBDIR = os.path.join("arco-era5", "ar")
LEVELS_HPA = (50, 100, 150, 200, 250, 300, 400, 500, 600, 700, 850, 925, 1000)
FRAME_SECONDS = 21600                       # 00, 06, 12, 18 UTC
FRAMES_PER_BIN = sh.BIN_SECONDS // FRAME_SECONDS   # 20
FIRST_YEAR = 1982
EPOCH = dt.datetime(1982, 1, 1)
assert EPOCH.date() == f10b.START
HEADER_PROBE = 4096          # one read holds the blosc header + block table
WORKERS = 16                 # measured: 16 parallel GETs ~100 MB/s here
F16_TINY = float(np.finfo(np.float16).tiny)  # 6.1035e-05, smallest normal

#: variable code -> (archive name, stored unit, scale to the stored unit,
#: (lo, hi) bounds IN THE STORED UNIT, plain-English name)
VARIABLES = {
    "t": ("temperature", "K", 1.0, (150.0, 350.0), "air temperature"),
    # the lower bound admits ERA5's OWN small negative humidities — measured
    # in January 2015: 11 values, the lowest -0.0042 g/kg, all at 100-200 hPa
    # in the tropics (a known numerical artefact of the analysis). They are
    # kept as given; anything below -0.01 g/kg is treated as corrupt
    "q": ("specific_humidity", "g/kg (1000 x the archive's kg/kg)", 1000.0,
          (-0.01, 40.0), "specific humidity"),
    "u": ("u_component_of_wind", "m/s (eastward)", 1.0, (-200.0, 200.0),
          "eastward wind"),
    "v": ("v_component_of_wind", "m/s (northward)", 1.0, (-200.0, 200.0),
          "northward wind"),
}

#: stored bytes per frame, MEASURED by the January-2015 probes (E-085 §4,
#: ml/family1/probes/era5_<v>_2015-01.json) — what a fetch lane's parts cost
#: on disk before it pushes, used by `fetch_preflight` to size the lane
PROBE_BYTES_PER_FRAME = {"t": 831656, "q": 1445900, "u": 1525392,
                         "v": 1558604}
#: headroom on that projection: frame sizes vary by season and level of
#: activity, and the push restores each part to scratch to hash it
DISK_MARGIN = 1.15
DISK_SPARE_BYTES = 2_000_000_000

LICENCE = {
    "name": "CC BY 4.0 (ERA5, Copernicus Climate Change Service)",
    "redistribution": "attribution",
    "redistribution_confirmed": True,
    "derived_works": "free",
    "attribution": (
        "Contains modified Copernicus Climate Change Service information "
        "[1982-2026]. Neither the European Commission nor ECMWF is "
        "responsible for any use that may be made of the Copernicus "
        "information or data it contains. Hersbach, H. et al. (2020), The "
        "ERA5 global reanalysis, Q. J. R. Meteorol. Soc. 146, 1999-2049, "
        "doi:10.1002/qj.3803; ERA5 hourly data on pressure levels, "
        "doi:10.24381/cds.bd0915c6. Read from Google's ARCO-ERA5 copy "
        "(Carver & Merose 2023, gs://gcp-public-data-arco-era5), whose "
        "1-degree conservative regrid is WeatherBench 2's (Rasp et al. "
        "2024)."),
}


class FormatError(ValueError):
    """An archive that does not match what this module declares it to be."""


# ============================================================ the grids ====
def target_axes(deg=1.0):
    """The store's grid: latitude -90..90 and longitude -180..180-deg."""
    n_lat = int(round(180.0 / deg)) + 1
    n_lon = int(round(360.0 / deg))
    lat = -90.0 + deg * np.arange(n_lat, dtype=np.float64)
    lon = -180.0 + deg * np.arange(n_lon, dtype=np.float64)
    return lat, lon


def grid_dict(deg=1.0):
    lat, lon = target_axes(deg)
    return {
        "H": int(len(lat)), "W": int(len(lon)),
        "crs": "EPSG:4326",
        "proj4": "+proj=longlat +datum=WGS84 +no_defs",
        "x0": -180.0 - deg / 2, "dx": float(deg),
        "y0": -90.0 - deg / 2, "dy": float(deg),
        "coordinates": (
            f"POINT-aligned like family 7's g100: lon = -180 + col * {deg:g}, "
            f"lat = -90 + row * {deg:g} (x0/y0 are the half-cell-shifted "
            f"WEST and SOUTH edges, so lon = x0 + (col + 0.5) * dx as for "
            f"every other tier-G group)"),
        "row_order": ("row 0 is latitude -90 (the SOUTH POLE) and the last "
                      "row is +90; column 0 is longitude -180"),
        "cell_rule": (
            f"each value is the CONSERVATIVE (area-weighted) mean of ERA5's "
            f"0.25-degree field over [lat - {deg / 2:g}, lat + {deg / 2:g}] "
            f"clipped to [-90, 90] x [lon - {deg / 2:g}, lon + {deg / 2:g}]; "
            f"the two pole rows are half-cells"),
        "extent": [-180.0 - deg / 2, -90.0, 180.0 - deg / 2, 90.0],
        "extent_note": "[west, south, east, north] in degrees",
        "no_tile_corners": ("a plain geographic grid: the rule above IS the "
                            "lat/lon of every pixel and tile corner"),
    }


def _lat_edges(points_asc):
    """Cell edges for ascending latitude points: midpoints, clipped to
    +/-90."""
    p = [float(x) for x in points_asc]
    mid = [(p[i] + p[i + 1]) / 2.0 for i in range(len(p) - 1)]
    return [-90.0] + mid + [90.0]


def _lon_edges(points):
    """(lo, hi) per longitude point: midpoints to the neighbours, periodic."""
    p = [float(x) for x in points]
    n = len(p)
    lo, hi = [], []
    for i in range(n):
        prev = p[i - 1] - (360.0 if i == 0 else 0.0)
        nxt = p[(i + 1) % n] + (360.0 if i == n - 1 else 0.0)
        lo.append((prev + p[i]) / 2.0)
        hi.append((p[i] + nxt) / 2.0)
    return lo, hi


def _sparse(rows):
    """[[(src, w), ...], ...] -> (idx [n, K] int64, w [n, K] float64),
    normalised per target; a pad repeats the first source with weight 0, so
    NaN propagation is exactly "any contributing source cell is NaN"."""
    K = max(len(r) for r in rows)
    idx = np.zeros((len(rows), K), np.int64)
    w = np.zeros((len(rows), K), np.float64)
    for i, r in enumerate(rows):
        if not r:
            raise FormatError(f"target cell {i} overlaps no source cell")
        tot = math.fsum(x for _s, x in r)
        for k, (s, x) in enumerate(r):
            idx[i, k], w[i, k] = s, x / tot
        for k in range(len(r), K):
            idx[i, k], w[i, k] = r[0][0], 0.0
    return idx, w


def lat_weights(src_asc, tgt_asc):
    """Conservative latitude weights, overlap measured in sin(latitude)."""
    se = _lat_edges(src_asc)
    te = _lat_edges(tgt_asc)
    rows = []
    for i in range(len(tgt_asc)):
        a, b = te[i], te[i + 1]
        r = []
        for j in range(len(src_asc)):
            lo, hi = max(a, se[j]), min(b, se[j + 1])
            if hi > lo:
                r.append((j, math.sin(math.radians(hi))
                          - math.sin(math.radians(lo))))
        rows.append(r)
    return _sparse(rows)


def lon_weights(src, tgt):
    """Conservative longitude weights, overlap in degrees, periodic."""
    slo, shi = _lon_edges(src)
    tlo, thi = _lon_edges(tgt)
    rows = []
    for i in range(len(tgt)):
        r = []
        for j in range(len(src)):
            ov = 0.0
            for k in (-360.0, 0.0, 360.0):
                lo, hi = max(tlo[i], slo[j] + k), min(thi[i], shi[j] + k)
                if hi > lo:
                    ov += hi - lo
            if ov > 0:
                r.append((j, ov))
        rows.append(r)
    return _sparse(rows)


def apply_weights(field, wlat, wlon):
    """[n_src_lat, n_src_lon] (lat ascending) -> [n_tgt_lat, n_tgt_lon].

    Explicit weighted sums in a fixed order — elementwise float64 ufuncs
    only, no BLAS — so the bytes do not depend on the machine.
    """
    f = np.asarray(field, np.float64)
    li, lw = wlat
    acc = lw[:, 0:1] * f[li[:, 0], :]
    for k in range(1, li.shape[1]):
        acc = acc + lw[:, k:k + 1] * f[li[:, k], :]
    oi, ow = wlon
    out = ow[None, :, 0] * acc[:, oi[:, 0]]
    for k in range(1, oi.shape[1]):
        out = out + ow[None, :, k] * acc[:, oi[:, k]]
    return out


# ======================================================== the archive I/O ===
class _NotFound(IOError):
    pass


class Archive:
    """Whole-object and ranged reads of the bucket, or of a local copy.

    HTTP is plain `requests` with one keep-alive session per thread. A range
    read MUST come back 206 with exactly the bytes asked for (or, for the
    header probe only, end exactly at the object's end); anything else
    raises. Every byte is counted once, under a lock, into the framework's
    counter (`f10b.count_bytes`, which the probe reads as bytes fetched).
    """

    def __init__(self, base, local, attempts=4):
        self.base = str(base).rstrip("/")
        self.local = bool(local)
        self.attempts = max(1, int(attempts))
        self._tl = threading.local()
        self._lock = threading.Lock()
        self.bytes = 0
        self.requests = 0

    def _count(self, n):
        with self._lock:
            self.bytes += int(n)
            self.requests += 1
            f10b.count_bytes(int(n))

    def _session(self):
        s = getattr(self._tl, "s", None)
        if s is None:
            import requests
            s = self._tl.s = requests.Session()
            s.headers.update(f10b.UA)
        return s

    def path(self, rel):
        return os.path.join(self.base, *rel.split("/")) if self.local \
            else f"{self.base}/{rel}"

    def get(self, rel):
        p = self.path(rel)
        if self.local:
            if not os.path.exists(p):
                raise _NotFound(p)
            with open(p, "rb") as fh:
                b = fh.read()
            self._count(len(b))
            return b
        err = None
        for i in range(self.attempts):
            try:
                r = self._session().get(p, timeout=f10b.SOCKET_TIMEOUT)
                if r.status_code == 404:
                    raise _NotFound(p)
                if r.status_code != 200:
                    raise IOError(f"{p}: HTTP {r.status_code}")
                b = r.content
                want = r.headers.get("x-goog-stored-content-length") or \
                    r.headers.get("Content-Length")
                if want is not None and r.headers.get(
                        "Content-Encoding") in (None, "", "identity") \
                        and int(want) != len(b):
                    raise IOError(f"{p}: {len(b)} of {want} bytes")
                self._count(len(b))
                return b
            except _NotFound:
                raise
            except Exception as e:                          # noqa: BLE001
                err = e
                if i < self.attempts - 1:
                    time.sleep(2.0 * (2 ** i))
        raise IOError(f"{p}: {type(err).__name__}: {err}")

    def read(self, rel, off, n, allow_eof=False):
        """Exactly n bytes at off -> (bytes, object_size)."""
        p = self.path(rel)
        if n <= 0:
            raise ValueError("an empty range")
        if self.local:
            if not os.path.exists(p):
                raise _NotFound(p)
            size = os.path.getsize(p)
            with open(p, "rb") as fh:
                fh.seek(off)
                b = fh.read(n)
            if len(b) != n and not (allow_eof and off + len(b) == size):
                raise IOError(f"{p}: {len(b)} of {n} bytes at {off}")
            self._count(len(b))
            return b, size
        err = None
        for i in range(self.attempts):
            try:
                r = self._session().get(
                    p, headers={"Range": f"bytes={off}-{off + n - 1}"},
                    timeout=f10b.SOCKET_TIMEOUT)
                if r.status_code == 404:
                    raise _NotFound(p)
                if r.status_code != 206:
                    raise IOError(f"{p}: asked for a byte range and got HTTP "
                                  f"{r.status_code} — refusing to read the "
                                  f"whole object")
                b = r.content
                m = re.match(r"bytes (\d+)-(\d+)/(\d+)",
                             r.headers.get("Content-Range", ""))
                if not m or int(m.group(1)) != off:
                    raise IOError(f"{p}: Content-Range "
                                  f"{r.headers.get('Content-Range')!r} for "
                                  f"bytes={off}-{off + n - 1}")
                size = int(m.group(3))
                if len(b) != n and not (allow_eof
                                        and off + len(b) == size
                                        and int(m.group(2)) == size - 1):
                    raise IOError(f"{p}: range {off}+{n} returned {len(b)} "
                                  f"bytes")
                self._count(len(b))
                return b, size
            except _NotFound:
                raise
            except Exception as e:                          # noqa: BLE001
                err = e
                if i < self.attempts - 1:
                    time.sleep(2.0 * (2 ** i))
        raise IOError(f"{p} bytes {off}+{n}: {type(err).__name__}: {err}")


# =========================================================== blosc blocks ===
BLOSC_MEMCPYED = 0x2


def parse_header(head, want_nbytes, object_size, where):
    """A blosc-1 chunk header + block table -> dict. Raises FormatError."""
    if len(head) < 16:
        raise FormatError(f"{where}: {len(head)} header bytes")
    ver, verlz, flags, ts = head[0], head[1], head[2], head[3]
    nbytes, bsize, cbytes = struct.unpack("<iii", head[4:16])
    if ts != 4:
        raise FormatError(f"{where}: blosc typesize {ts}, float32 is 4")
    if nbytes != want_nbytes:
        raise FormatError(f"{where}: blosc says {nbytes} uncompressed bytes, "
                          f"the zarr's chunk shape is {want_nbytes}")
    if cbytes != object_size:
        raise FormatError(f"{where}: blosc says {cbytes} compressed bytes, "
                          f"the object is {object_size}")
    out = {"version": ver, "versionlz": verlz, "flags": flags,
           "typesize": ts, "nbytes": nbytes, "blocksize": bsize,
           "cbytes": cbytes, "memcpyed": bool(flags & BLOSC_MEMCPYED)}
    if out["memcpyed"]:
        out["nblocks"] = 0
        return out
    if bsize <= 0:
        raise FormatError(f"{where}: blosc blocksize {bsize}")
    nb = -(-nbytes // bsize)
    need = 16 + 4 * nb
    if len(head) < need:
        out["need"] = need
        return out
    bst = np.frombuffer(head[16:need], "<i4").astype(np.int64)
    # A BLOCK'S BYTES END WHERE THE NEXT-HIGHER START BEGINS. c-blosc writes
    # blocks in the order its threads finish them, so the table is NOT in
    # index order in general (measured on numcodecs' own multi-threaded
    # output); sorted, the starts must tile [end of table, cbytes) exactly.
    order = np.argsort(bst, kind="stable")
    srt = bst[order]
    if srt[0] < need or srt[-1] >= cbytes or (np.diff(srt) <= 0).any():
        raise FormatError(f"{where}: the blosc block table does not tile "
                          f"the chunk (starts {srt[:3].tolist()} .. "
                          f"{srt[-1]}, table ends at {need}, chunk is "
                          f"{cbytes} bytes)")
    ends = np.empty_like(bst)
    ends[order] = np.concatenate([srt[1:], [cbytes]])
    out.update(nblocks=nb, bstarts=bst, bends=ends, head=bytes(head[:16]))
    return out


def subchunk(hdr, blocks, k0, k1):
    """Blocks k0..k1 of a chunk, re-wrapped as a valid blosc chunk."""
    bs, nbytes = hdr["blocksize"], hdr["nbytes"]
    sizes = [min(bs, nbytes - k * bs) for k in range(k0, k1 + 1)]
    nb = len(sizes)
    pos = 16 + 4 * nb
    starts = []
    for b in blocks:
        starts.append(pos)
        pos += len(b)
    return (hdr["head"][:4] + struct.pack("<iii", sum(sizes), bs, pos)
            + struct.pack(f"<{nb}i", *starts) + b"".join(blocks))


def blosc_decode(buf):
    try:
        from numcodecs import blosc
    except ImportError:                                      # pragma: no cover
        sys.exit("the ERA5 adapters need `numcodecs` (pip install numcodecs) "
                 "— it is in the family1-build workflow's install step")
    return blosc.decompress(buf)


# =============================================================== the zarr ===
def _tally(t, n):
    if t is not None:
        t["bytes"] += int(n)
        t["requests"] += 1


def _hours_units(units):
    m = re.match(r"hours since (\d{4})-(\d{2})-(\d{2})(?:[ T](\d{2}):(\d{2})"
                 r"(?::(\d{2}))?)?", str(units).strip())
    if not m:
        raise FormatError(f"time units {units!r} are not 'hours since ...'")
    y, mo, d = int(m.group(1)), int(m.group(2)), int(m.group(3))
    hh = int(m.group(4) or 0)
    mm = int(m.group(5) or 0)
    return dt.datetime(y, mo, d, hh, mm)


class Zarr:
    """One consolidated zarr-v2 group of the archive, read from its metadata."""

    def __init__(self, archive, name):
        self.ar = archive
        self.name = name
        try:
            raw = archive.get(f"{name}/.zmetadata")
        except _NotFound:
            raise FormatError(f"{archive.path(name)}: no .zmetadata — not a "
                              f"consolidated zarr") from None
        md = json.loads(raw)
        if md.get("zarr_consolidated_format") != 1:
            raise FormatError(f"{name}: zarr_consolidated_format "
                              f"{md.get('zarr_consolidated_format')}")
        self.md = md["metadata"]
        self.attrs = self.md.get(".zattrs", {}) or {}
        self._axes = {}
        self._hdr = {}
        self._lock = threading.Lock()

    def array(self, var):
        za = self.md.get(f"{var}/.zarray")
        if za is None:
            raise FormatError(f"{self.name}: no array {var!r}; it holds "
                              + ", ".join(sorted(k[:-8] for k in self.md
                                                 if k.endswith("/.zarray"))))
        at = self.md.get(f"{var}/.zattrs", {}) or {}
        return {"shape": [int(x) for x in za["shape"]],
                "chunks": [int(x) for x in za["chunks"]],
                "dtype": za["dtype"], "order": za.get("order", "C"),
                "compressor": za.get("compressor"),
                "filters": za.get("filters"),
                "fill_value": za.get("fill_value"),
                "dims": list(at.get("_ARRAY_DIMENSIONS") or []),
                "units": at.get("units"), "attrs": at}

    def axis(self, name):
        if name in self._axes:
            return self._axes[name]
        a = self.array(name)
        if len(a["shape"]) != 1 or a["filters"]:
            raise FormatError(f"{self.name}/{name}: not a plain 1-d axis")
        comp = a["compressor"]
        n, ch = a["shape"][0], a["chunks"][0]
        parts = []
        for i in range(-(-n // ch)):
            raw = self.ar.get(f"{self.name}/{name}/{i}")
            if comp is None:
                dec = raw
            elif comp.get("id") == "blosc":
                dec = blosc_decode(raw)
            else:
                raise FormatError(f"{self.name}/{name}: compressor {comp}")
            parts.append(np.frombuffer(dec, np.dtype(a["dtype"])))
        v = np.concatenate(parts)[:n]
        if len(v) != n:
            raise FormatError(f"{self.name}/{name}: {len(v)} of {n} values")
        self._axes[name] = v
        return v

    def check_var(self, var):
        """The declared layout this module reads, or a refusal."""
        a = self.array(var)
        if a["dtype"] != "<f4" or a["order"] != "C" or a["filters"]:
            raise FormatError(f"{self.name}/{var}: {a['dtype']} order "
                              f"{a['order']} filters {a['filters']} — this "
                              f"reader expects plain little-endian float32 "
                              f"in C order")
        c = a["compressor"] or {}
        if c.get("id") != "blosc":
            raise FormatError(f"{self.name}/{var}: compressor {c} — the block "
                              f"reader is for blosc")
        if a["dims"][:2] != ["time", "level"] or \
                sorted(a["dims"][2:]) != ["latitude", "longitude"]:
            raise FormatError(f"{self.name}/{var}: dims {a['dims']}")
        if a["chunks"][1:] != a["shape"][1:]:
            raise FormatError(f"{self.name}/{var}: chunks {a['chunks']} split "
                              f"a level or the globe (shape {a['shape']}); "
                              f"the block reader needs one spatial chunk")
        return a

    def times(self):
        """(datetime of index 0, the hourly axis as int64 hours) — refuses an
        axis that is not contiguous hourly."""
        a = self.array("time")
        t0 = _hours_units(a["units"])
        t = self.axis("time").astype(np.int64)
        if len(t) > 1 and not (np.diff(t) == 1).all():
            raise FormatError(f"{self.name}: the time axis is not contiguous "
                              f"hourly")
        return t0, t

    def header(self, rel, a, tally=None):
        """The chunk's parsed blosc header, cached; `tally` (a dict) gets
        this call's own bytes and requests — the archive's counter is shared
        by every thread and cannot say which frame a byte was for."""
        with self._lock:
            h = self._hdr.get(rel)
        if h is not None:
            return h
        want = int(np.prod(a["chunks"])) * 4
        head, size = self.ar.read(rel, 0, HEADER_PROBE, allow_eof=True)
        _tally(tally, len(head))
        h = parse_header(head, want, size, rel)
        if "need" in h:
            head, size = self.ar.read(rel, 0, h["need"])
            _tally(tally, len(head))
            h = parse_header(head, want, size, rel)
        with self._lock:
            if len(self._hdr) > 64:
                self._hdr.clear()
            self._hdr[rel] = h
        return h

    def read_levels(self, var, ti, lev_idx, a=None):
        """Slabs [len(lev_idx), d2, d3] float32 of time index ti — only the
        blosc blocks that hold them are fetched. -> (array, stats)."""
        a = a or self.check_var(var)
        ct = a["chunks"][0]
        ci, tin = divmod(int(ti), ct)
        key = ".".join([str(ci)] + ["0"] * (len(a["shape"]) - 1))
        rel = f"{self.name}/{var}/{key}"
        L = a["chunks"][1]
        d2, d3 = a["chunks"][2], a["chunks"][3]
        plane = d2 * d3
        tally = {"bytes": 0, "requests": 0}
        h = self.header(rel, a, tally)
        out = np.empty((len(lev_idx), d2, d3), np.float32)
        items = [((tin * L) + int(li)) * plane for li in lev_idx]
        if h["memcpyed"]:
            for k, it in enumerate(items):
                raw, _ = self.ar.read(rel, 16 + it * 4, plane * 4)
                _tally(tally, len(raw))
                out[k] = np.frombuffer(raw, "<f4").reshape(d2, d3)
            return out, {"blocks": 0, **tally}
        bs = h["blocksize"]
        need = set()
        spans = []
        for it in items:
            k0, k1 = it * 4 // bs, ((it + plane) * 4 - 1) // bs
            spans.append((k0, k1))
            need.update(range(k0, k1 + 1))
        # FETCH: the needed blocks in BYTE order, byte-adjacent ones
        # coalesced into one ranged GET each
        byorder = sorted(need, key=lambda k: int(h["bstarts"][k]))
        ranges, cur = [], None
        for k in byorder:
            lo_, hi_ = int(h["bstarts"][k]), int(h["bends"][k])
            if cur and lo_ == cur[1]:
                cur[1] = hi_
                cur[2].append(k)
            else:
                cur = [lo_, hi_, [k]]
                ranges.append(cur)
        got = {}
        for lo_, hi_, ks in ranges:
            raw, _ = self.ar.read(rel, lo_, hi_ - lo_)
            _tally(tally, len(raw))
            for k in ks:
                got[k] = raw[int(h["bstarts"][k]) - lo_:
                             int(h["bends"][k]) - lo_]
        # DECODE: runs of consecutive block INDICES, each re-wrapped as a
        # small valid blosc chunk
        runs, cur = [], None
        for k in sorted(need):
            if cur and k == cur[1] + 1:
                cur[1] = k
            else:
                cur = [k, k]
                runs.append(cur)
        decoded = {}
        for k0, k1 in runs:
            blocks = [got[k] for k in range(k0, k1 + 1)]
            dec = blosc_decode(subchunk(h, blocks, k0, k1))
            want = sum(min(bs, h["nbytes"] - k * bs)
                       for k in range(k0, k1 + 1))
            if len(dec) != want:
                raise FormatError(f"{rel}: blocks {k0}..{k1} decoded to "
                                  f"{len(dec)} bytes, expected {want}")
            decoded[k0] = (k1, np.frombuffer(dec, "<f4"))
        for k, it in enumerate(items):
            for k0, (k1, arr) in decoded.items():
                if k0 * bs // 4 <= it and it + plane <= \
                        (k0 * bs + len(arr) * 4) // 4:
                    o = it - k0 * bs // 4
                    out[k] = arr[o:o + plane].reshape(d2, d3)
                    break
            else:                                         # pragma: no cover
                raise FormatError(f"{rel}: slab {it} not in a decoded run")
        return out, {"blocks": len(need), "runs": len(runs),
                     "ranges": len(ranges), **tally}


# ============================================================ the sources ===
class Sources:
    """Both zarrs, their measured axes, the seam and the valid end."""

    def __init__(self, base, local, deg=1.0, attempts=4):
        self.ar = Archive(base, local, attempts)
        self.deg = float(deg)
        self.A = Zarr(self.ar, ZARR_A)
        self.B = Zarr(self.ar, ZARR_B)
        self.tlat, self.tlon = target_axes(self.deg)
        # --- A: the producer's 1-degree conservative grid, as the store's
        a_lat = self.A.axis("latitude").astype(np.float64)
        a_lon = self.A.axis("longitude").astype(np.float64)
        if a_lat.shape != self.tlat.shape or \
                np.abs(a_lat - self.tlat).max() > 1e-6:
            raise FormatError(f"{ZARR_A}: latitude {a_lat[:3]}..{a_lat[-3:]} "
                              f"is not the store's {self.tlat[:3]}.."
                              f"{self.tlat[-3:]}")
        self.a_col = []
        for x in self.tlon:
            d = np.abs(((a_lon - x) + 180.0) % 360.0 - 180.0)
            j = int(np.argmin(d))
            if d[j] > 1e-6:
                raise FormatError(f"{ZARR_A}: no longitude {x}")
            self.a_col.append(j)
        self.a_col = np.array(self.a_col, np.int64)
        if len(set(self.a_col.tolist())) != len(self.tlon):
            raise FormatError(f"{ZARR_A}: longitudes do not map one to one")
        self.a_t0, self.a_t = self.A.times()
        # --- B: 0.25 degrees, regridded here
        b_lat = self.B.axis("latitude").astype(np.float64)
        b_lon = self.B.axis("longitude").astype(np.float64)
        self.b_flip = bool(b_lat[0] > b_lat[-1])
        b_lat_asc = b_lat[::-1] if self.b_flip else b_lat
        if not (np.diff(b_lat_asc) > 0).all():
            raise FormatError(f"{ZARR_B}: latitude is not monotone")
        self.wlat = lat_weights(b_lat_asc, self.tlat)
        self.wlon = lon_weights(b_lon, self.tlon)
        self.b_t0, self.b_t = self.B.times()
        self.b_attrs = dict(self.B.attrs)
        stop = self.b_attrs.get("valid_time_stop")
        if not stop:
            raise FormatError(f"{ZARR_B}: no `valid_time_stop` in the root "
                              f"attributes — the archive does not say how "
                              f"far its pre-allocated time axis is real")
        d = dt.date.fromisoformat(str(stop)[:10])
        self.b_last = dt.datetime(d.year, d.month, d.day, 23)
        start = self.b_attrs.get("valid_time_start")
        self.b_first = (dt.datetime.fromisoformat(str(start)[:10])
                        if start else self.b_t0 + dt.timedelta(
                            hours=int(self.b_t[0])))
        # --- the levels, by VALUE, in the store's order
        self.a_lev = self._levels(self.A)
        self.b_lev = self._levels(self.B)
        # --- the seam: the first hour A does not hold
        self.a_first = self.a_t0 + dt.timedelta(hours=int(self.a_t[0]))
        self.a_last = self.a_t0 + dt.timedelta(hours=int(self.a_t[-1]))
        self.seam = self.a_last + dt.timedelta(hours=1)
        if self.b_first > self.seam:
            raise FormatError(f"the 0.25-degree archive starts "
                              f"{self.b_first}, after the seam {self.seam} — "
                              f"a gap between the two sources")

    def _levels(self, z):
        lv = [int(x) for x in z.axis("level")]
        miss = [p for p in LEVELS_HPA if p not in lv]
        if miss:
            raise FormatError(f"{z.name}: levels {miss} hPa are not in its "
                              f"level axis {lv}")
        return [lv.index(p) for p in LEVELS_HPA]

    def where(self, when):
        """(source, time index) or (None, reason) for one instant."""
        if when < self.a_first:
            return None, "before_record"
        if when < self.seam:
            h = int((when - self.a_t0).total_seconds() // 3600)
            i = h - int(self.a_t[0])
            if not (0 <= i < len(self.a_t)) or int(self.a_t[i]) != h:
                raise FormatError(f"{ZARR_A}: {when} not at index {i}")
            return "A", i
        if when > self.b_last:
            return None, "after_record"
        h = int((when - self.b_t0).total_seconds() // 3600)
        i = h - int(self.b_t[0])
        if not (0 <= i < len(self.b_t)) or int(self.b_t[i]) != h:
            raise FormatError(f"{ZARR_B}: {when} not at index {i}")
        return "B", i

    def frame(self, var, src, ti):
        """[H, W, 13] float64 in the ARCHIVE's unit, plus read stats."""
        if src == "A":
            a = self.A.check_var(var)
            slabs, st = self.A.read_levels(var, ti, self.a_lev, a)
            dims = a["dims"][2:]
            out = np.empty((len(self.tlat), len(self.tlon), len(LEVELS_HPA)),
                           np.float64)
            for k in range(len(LEVELS_HPA)):
                s = slabs[k]
                if dims == ["longitude", "latitude"]:
                    s = s.T
                out[:, :, k] = s[:, self.a_col]
            return out, st
        a = self.B.check_var(var)
        slabs, st = self.B.read_levels(var, ti, self.b_lev, a)
        dims = a["dims"][2:]
        out = np.empty((len(self.tlat), len(self.tlon), len(LEVELS_HPA)),
                       np.float64)
        for k in range(len(LEVELS_HPA)):
            s = slabs[k]
            if dims == ["longitude", "latitude"]:
                s = s.T
            if self.b_flip:
                s = s[::-1]
            out[:, :, k] = apply_weights(s, self.wlat, self.wlon)
        return out, st


# ============================================================ the adapter ===
def channels_for(var):
    """((name, unit, lo, hi), ...) — one channel per level, `t_50` .. ."""
    _name, unit, _scale, (lo, hi), _plain = VARIABLES[var]
    return tuple((f"{var}_{p}", unit, lo, hi) for p in LEVELS_HPA)


def frame_instant(b, f):
    return EPOCH + dt.timedelta(
        seconds=sh.frame_start_seconds(b, f, FRAME_SECONDS))


class ERA5Base(sh.GridAdapter):
    """One ERA5 pressure-level variable as a sharded tier-G store."""

    var = None                       # "t" | "q" | "u" | "v"
    family = "12"
    distribution = "public"
    licence = LICENCE
    time_dtype = "int32"
    credentials = ()
    frames_per_bin = FRAMES_PER_BIN
    frame_seconds = FRAME_SECONDS
    dtype = "float16"
    # 64 x 64 one-degree tiles: 64 degrees on a side, the same footprint in
    # degrees as a 256-pixel tile of the 0.25-degree phase B, so a cone a few
    # thousand kilometres wide reads 1 to 4 tiles instead of the whole globe
    tile = 64
    zstd_level = sh.DEFAULT_LEVEL
    per_year = True
    first_year = FIRST_YEAR
    # the cell is a 1-degree box mean: log2(111.32 km / 27.83 km) = 2.0; an
    # instantaneous analysis has no averaging time of its own, so the time
    # field is the frame spacing, six hours: log2(0.25 day / 5 days) = -4.32
    log2_fp = float(np.log2(111.31949 / 27.83))
    log2_dt = float(np.log2(0.25 / 5.0))
    levels_hpa = LEVELS_HPA
    channel_axis = ("pressure level in hPa, ascending: channel k is level "
                    "levels_hpa[k], from 50 hPa (~20 km) down to 1000 hPa "
                    "(the surface)")
    exception = "E4"
    sources = (f"{GCS}/{ZARR_A} (hourly, 1-degree conservative; frames "
               f"before the seam)",
               f"{GCS}/{ZARR_B} (hourly, 0.25-degree; frames from the seam, "
               f"regridded here)")
    smoke_window = ("2021-12-25", "2022-01-09")
    smoke_probe_month = "2022-01"

    def __init__(self):
        name, unit, scale, (lo, hi), plain = VARIABLES[self.var]
        self.src_name, self.unit, self.scale = name, unit, scale
        self.channels = channels_for(self.var)
        self.deg = float(os.environ.get("ERA5_SMOKE_DEG") or 1.0)
        try:
            self.workers = int(os.environ.get("ERA5_WORKERS") or WORKERS)
        except ValueError:
            sys.exit("ERA5_WORKERS must be an integer")
        self._src = None
        self.title = (f"ERA5 {plain} on 13 pressure levels (50-1000 hPa), "
                      f"six-hourly instants on a 1-degree grid "
                      f"(conservative box means) — family 1.2, exception E4")
        self.qc_policy = (
            f"ERA5 publishes no per-value quality flag: a reanalysis value is "
            f"the model's analysis. Values are stored in {unit}"
            + (" (the archive's kg/kg x 1000, so float16 keeps them in its "
               "normal range)" if self.var == "q" else "")
            + f". A value outside [{lo:g}, {hi:g}] becomes NaN and is COUNTED "
              f"(`out_of_bounds`), never clipped. Pressure levels below the "
              f"ground hold ERA5's own extrapolation and are kept. A NaN in "
              f"any source cell a target cell averages makes that cell NaN "
              f"(`source_nan_pixels`).")
        self.notes = (
            "FAMILY 1.2, EXCEPTION E4: a ~31 km model-filled reanalysis, not "
            "an observation at <= 10 km — admitted by decision (2026-10-05). "
            "SOURCES AND SEAM: frames before the first hour the 1-degree "
            f"archive does not hold come from {ZARR_A} unchanged (the "
            "producer's conservative regrid, reindexed); frames from that "
            f"hour on come from {ZARR_B}, regridded here by the same rule "
            "(tile_grid.json `source`). Frames after the 0.25-degree "
            "archive's `valid_time_stop` (final ERA5) are after_record: "
            "ERA5T is not admitted. The six-hourly frame is the INSTANT at "
            "its start (00/06/12/18 UTC), not a six-hour mean.")
        if self.deg != 1.0:
            self.notes += (f"\nSMOKE GRID: ERA5_SMOKE_DEG={self.deg:g} — a "
                           f"synthetic store.")

    @property
    def grid(self):
        g = grid_dict(self.deg)
        if self.deg != 1.0:
            g["smoke"] = True
        return g

    def specs(self):
        out = super().specs()
        sp = out[self.store]
        sp["levels_hpa"] = list(LEVELS_HPA)
        sp["channel_axis"] = self.channel_axis
        sp["channel_encoding"] = (
            f"float16 in {self.unit}" + (
                "; kg/kg = value / 1000" if self.var == "q" else ""))
        sp["frame_rule"] = (
            "frame f of bin b is the ERA5 analysis INSTANT at b * 432000 + "
            "f * 21600 seconds since 1982-01-01T00:00Z — 00, 06, 12 or 18 "
            "UTC — not a mean over the six hours")
        sp["source"] = {
            "before_seam": f"{GCS}/{ZARR_A}: the producer's 1-degree "
                           f"conservative regrid, reindexed (rows "
                           f"south-first already; columns rolled to start "
                           f"at -180)",
            "from_seam": f"{GCS}/{ZARR_B}: 0.25-degree, regridded here "
                         f"with the cell_rule above (ml/family1/adapters/"
                         f"_era5.py :: lat_weights, lon_weights, "
                         f"apply_weights)",
            "seam": "the first hour the 1-degree archive does not hold, read "
                    "from its time axis at index time (plan.json `seam`)",
            "end": "the 0.25-degree archive's root attribute "
                   "`valid_time_stop`; ERA5T after it is not admitted"}
        sp["exception"] = self.exception
        return out

    # ------------------------------------------------------------- sources --
    def sources_for(self, ctx):
        if self._src is None:
            if ctx.source_dir:
                base, local = os.path.join(ctx.source_dir, SOURCE_SUBDIR), True
            else:
                base, local = GCS, False
            try:
                self._src = Sources(base, local, self.deg,
                                    attempts=max(1, ctx.a.attempts))
            except (FormatError, IOError) as e:
                sys.exit(f"REFUSING {self.store}: {e}")
        return self._src

    def record_frames(self, ctx, group):
        s = self.sources_for(ctx)
        last = s.b_last.replace(hour=18)
        return int((last - EPOCH).total_seconds() // FRAME_SECONDS) + 1

    def fetch_preflight(self, ctx):
        try:
            import numcodecs                                    # noqa: F401
        except ImportError:
            sys.exit(f"REFUSING {self.store}: `numcodecs` is not installed "
                     f"(pip install numcodecs) — it decodes the archive's "
                     f"blosc chunks. Nothing has been fetched.")
        self.sources_for(ctx)
        self.disk_preflight(ctx)

    def disk_preflight(self, ctx):
        """SIZE THE LANE FROM ITS OWN ALLOCATION (ml/CLAUDE.md §5.18).

        A lane keeps every year's parts on disk until `--push-parts` parks
        them, so its peak is the window's frames x the probe's measured
        stored bytes per frame. Checked here, while the inputs are all the
        lane has cost, rather than as an ENOSPC at hour two. Skipped for a
        synthetic source (the smoke's grid is not the probe's)."""
        import shutil
        bins = sum(len(v) for v in (getattr(ctx, "grid_bins", None)
                                    or {}).values())
        frames = bins * FRAMES_PER_BIN
        per = PROBE_BYTES_PER_FRAME[self.var]
        need = int(frames * per * DISK_MARGIN) + DISK_SPARE_BYTES
        os.makedirs(ctx.parts, exist_ok=True)
        free = shutil.disk_usage(ctx.parts).free
        print(f"  disk: {free / 1e9:.1f} GB free at {ctx.parts}; this "
              f"window's {frames} frames project to "
              f"{frames * per / 1e9:.1f} GB of parts "
              f"({per / 1e6:.3f} MB a frame, the probe's), "
              f"{need / 1e9:.1f} GB with margin", flush=True)
        if ctx.source_dir:
            return
        if free < need:
            sys.exit(f"REFUSING {self.store}: the lane needs ~{need / 1e9:.1f} "
                     f"GB for its parts and the disk has {free / 1e9:.1f} GB "
                     f"free. Split the window into shorter year ranges. "
                     f"Nothing has been fetched.")

    def index(self, ctx):
        s = self.sources_for(ctx)
        aA, aB = s.A.check_var(self.src_name), s.B.check_var(self.src_name)
        lo = EPOCH + dt.timedelta(seconds=int(ctx.t_lo))
        hi = EPOCH + dt.timedelta(seconds=int(ctx.t_hi))
        out = {
            "dataset": ("ERA5 hourly on pressure levels (ECMWF), Google's "
                        "ARCO-ERA5 copy"),
            "url": GCS, "version": f"{ZARR_A} + {ZARR_B}",
            "variable": self.src_name, "levels_hpa": list(LEVELS_HPA),
            "A": {"zarr": ZARR_A, "shape": aA["shape"],
                  "chunks": aA["chunks"], "dims": aA["dims"],
                  "compressor": aA["compressor"], "units": aA["units"],
                  "time": [str(s.a_first), str(s.a_last)],
                  "time_steps": int(len(s.a_t)),
                  "level_index": s.a_lev},
            "B": {"zarr": ZARR_B, "shape": aB["shape"],
                  "chunks": aB["chunks"], "dims": aB["dims"],
                  "compressor": aB["compressor"], "units": aB["units"],
                  "time_axis": [str(s.b_t0 + dt.timedelta(
                      hours=int(s.b_t[0]))), str(s.b_t0 + dt.timedelta(
                          hours=int(s.b_t[-1])))],
                  "root_attrs": s.b_attrs, "valid_last_hour": str(s.b_last),
                  "level_index": s.b_lev},
            "seam": str(s.seam),
            "window": [str(lo), str(hi)],
            "grid": self.grid,
        }
        # ONE REAL FRAME, read and decoded — unless the parts come from the
        # Hub, where the source is never read
        if getattr(ctx.a, "parts_from_hub", False):
            out["first_frame"] = {"not_read": "parts from the Hub"}
            return out
        first = None
        b0 = int(ctx.t_lo) // sh.BIN_SECONDS
        for f in range(FRAMES_PER_BIN * 2):
            bb, ff = b0 + f // FRAMES_PER_BIN, f % FRAMES_PER_BIN
            w = frame_instant(bb, ff)
            if w < lo:
                continue
            src, ti = s.where(w)
            if src is not None:
                first = (w, src, ti)
                break
        if first is None:
            out["first_frame"] = {"none": "no frame of the window is inside "
                                          "the record"}
            return out
        w, src, ti = first
        try:
            arr, st = s.frame(self.src_name, src, ti)
        except (FormatError, IOError) as e:
            sys.exit(f"REFUSING {self.store}: the first frame {w} could not "
                     f"be read: {e}")
        cosw = np.cos(np.radians(s.tlat))[:, None]
        out["first_frame"] = {
            "instant": str(w), "source": src, "time_index": int(ti),
            "read": st,
            "area_mean_per_level": {
                str(p): round(float((arr[:, :, k] * cosw).sum()
                                    / (cosw.sum() * arr.shape[1])
                                    * self.scale), 6)
                for k, p in enumerate(LEVELS_HPA)}}
        return out

    def fetch_frames(self, ctx, wanted):
        s = self.sources_for(ctx)
        jobs = []
        for (g, b, f) in wanted:
            w = frame_instant(b, f)
            try:
                src, ti = s.where(w)
            except FormatError as e:
                sys.exit(f"REFUSING {self.store}: {e}")
            jobs.append((g, b, f, w, src, ti))
        todo = [j for j in jobs if j[4] is not None]

        def work(job):
            g, b, f, w, src, ti = job
            t0 = time.time()
            try:
                arr, st = s.frame(self.src_name, src, ti)
                return job, arr, st, None, time.time() - t0
            except _NotFound as e:
                return job, None, None, f"chunk absent: {e}", 0.0
            except (FormatError, IOError, ValueError) as e:
                return job, None, None, f"{type(e).__name__}: {e}", 0.0

        workers = 1 if ctx.source_dir else max(1, self.workers)
        it = cm.ordered_map(work, todo, workers, lookahead=2 * workers)
        names = self.channel_names
        for job in jobs:
            g, b, f, w, src, ti = job
            if src is None:
                yield g, b, f, None, {"frame_missing": ti}
                continue
            j2, arr, st, err, secs = next(it)
            if j2[:3] != (g, b, f):
                sys.exit(f"{self.store}: the read pool answered {j2[:3]} "
                         f"where {(g, b, f)} was asked for")
            if arr is None:
                ctx.note_absent(f"{w:%Y-%m-%dT%H}Z",
                                f"{self.src_name} from "
                                f"{ZARR_A if src == 'A' else ZARR_B} index "
                                f"{ti}: {err}")
                continue
            v = arr.reshape(-1, len(names)) * self.scale
            nan_in = int(np.isnan(v).sum())
            oob = self.mask_bounds(v)
            counts = {"frames_from": {
                          ("arco_1deg_conservative" if src == "A"
                           else "arco_025_regridded_here"): 1},
                      "source_bytes": {src: int(st.get("bytes", 0))},
                      "range_requests": int(st.get("requests", 0)),
                      "blosc_blocks_read": int(st.get("blocks", 0)),
                      "max_frame_read_seconds": round(secs, 3)}
            if nan_in:
                counts["source_nan_pixels"] = nan_in
            if oob:
                counts["out_of_bounds"] = oob
            # what float16 does to every value, measured on the frame itself
            q16 = v.astype(np.float16).astype(np.float64)
            with np.errstate(invalid="ignore"):
                err_abs = np.abs(q16 - v)
                mag = np.abs(v)
            for k, nm in enumerate(names):
                col, e = v[:, k], err_abs[:, k]
                ok = np.isfinite(col)
                if not ok.any():
                    continue
                counts[f"max_value_{nm}"] = float(col[ok].max())
                counts[f"max_negated_min_{nm}"] = float(-col[ok].min())
                counts[f"max_f16err_abs_{nm}"] = float(e[ok].max())
                norm = ok & (mag[:, k] >= F16_TINY)
                if norm.any():
                    counts[f"max_f16err_rel_{nm}"] = float(
                        (e[norm] / mag[norm, k]).max())
                sub = int((ok & (mag[:, k] < F16_TINY)
                           & (mag[:, k] > 0)).sum())
                if sub:
                    counts.setdefault("f16_subnormal_values", {})[nm] = sub
            yield g, b, f, v.reshape(arr.shape), counts

    # ---------------------------------------------------------------- smoke --
    def smoke_sources(self, root, d_lo, d_hi):
        truth = make_smoke_sources(root, d_lo, d_hi, self)
        self.__init__()
        return truth


# ================================================================= smoke ===
SMOKE_DEG = 10.0
SMOKE_LEVELS = (1, 10, 50, 100, 150, 200, 250, 300, 400, 500, 600, 700, 850,
                925, 975, 1000)
# The 1-degree part ends 2021-12-29T23, so the smoke's seam falls INSIDE bin
# 2921 (2021-12-27 .. 31) and one bin holds frames from both sources. The
# real seam, 2022-01-01T00, is exactly the start of bin 2922 (14,610 days
# after the epoch), which would test nothing.
SMOKE_A = (dt.datetime(2021, 12, 20), dt.datetime(2021, 12, 29, 23))
SMOKE_B_AXIS = (dt.datetime(2021, 12, 27), dt.datetime(2022, 1, 9, 23))
SMOKE_B_VALID = (dt.date(2021, 12, 27), dt.date(2022, 1, 5))
#: one value past its bounds, at one instant and level, in the A part
SMOKE_OOB = (dt.datetime(2021, 12, 28, 12), 500)
SMOKE_BLOCKSIZE = 4096       # many blosc blocks per chunk, as in the archive


def smoke_value(var, level, lat, lon, hours):
    """A deterministic field in the ARCHIVE's unit (q in kg/kg)."""
    la, lo = np.radians(lat), np.radians(lon)
    if var == "t":
        return 200.0 + 0.08 * level + 25.0 * np.cos(la) \
            + 3.0 * np.sin(lo + hours / 7.0)
    if var == "q":
        return (2e-6 + 1.8e-5 * level) * (0.4 + 0.6 * np.cos(la) ** 2) \
            * (1.0 + 0.2 * np.sin(2 * lo + hours / 5.0))
    if var == "u":
        return 30.0 * np.cos(la) * (level / 1000.0) ** 0.3 \
            + 5.0 * np.sin(lo - hours / 9.0)
    return 8.0 * np.sin(2 * la) * np.cos(lo + hours / 11.0) + level / 500.0


def _write_zarr(root, name, arrays, attrs=None, keep_chunk=None):
    """A consolidated zarr-v2 group, blosc-lz4 + shuffle, as the archive.

    `keep_chunk(var, idx)` false -> that chunk file is not written (a zarr
    reads an absent chunk as its fill value; the reader here REFUSES one)."""
    from numcodecs import Blosc
    codec = Blosc(cname="lz4", clevel=5, shuffle=Blosc.SHUFFLE,
                  blocksize=SMOKE_BLOCKSIZE)
    md = {".zgroup": {"zarr_format": 2}, ".zattrs": attrs or {}}
    base = os.path.join(root, name)
    for var, spec in arrays.items():
        data, chunks, dims, units = (spec["data"], spec["chunks"],
                                     spec["dims"], spec.get("units"))
        d = os.path.join(base, var)
        os.makedirs(d, exist_ok=True)
        comp = codec.get_config()
        md[f"{var}/.zarray"] = {
            "chunks": list(chunks), "compressor": comp, "dtype":
                np.dtype(data.dtype).str, "fill_value": None, "filters":
                None, "order": "C", "shape": list(data.shape),
                "zarr_format": 2}
        at = {"_ARRAY_DIMENSIONS": dims}
        if units:
            at["units"] = units
        md[f"{var}/.zattrs"] = at
        grid = [range(-(-n // c)) for n, c in zip(data.shape, chunks)]
        import itertools
        for idx in itertools.product(*grid):
            if keep_chunk is not None and not keep_chunk(var, idx):
                continue
            sl = tuple(slice(i * c, min((i + 1) * c, n))
                       for i, c, n in zip(idx, chunks, data.shape))
            blk = np.zeros(chunks, data.dtype)
            part = data[sl]
            blk[tuple(slice(0, s) for s in part.shape)] = part
            with open(os.path.join(d, ".".join(map(str, idx))), "wb") as fh:
                fh.write(codec.encode(np.ascontiguousarray(blk)))
    os.makedirs(base, exist_ok=True)
    with open(os.path.join(base, ".zmetadata"), "w") as fh:
        json.dump({"metadata": md, "zarr_consolidated_format": 1}, fh)


def write_smoke_archive(root, deg=SMOKE_DEG):
    """Both zarrs, all four variables, under <root>/arco-era5/ar/."""
    base = os.path.join(root, SOURCE_SUBDIR)
    lv = np.array(SMOKE_LEVELS, np.int64)
    # A: the coarse 'with poles' grid, (time, level, longitude, latitude)
    a_lat = -90.0 + deg * np.arange(int(round(180 / deg)) + 1)
    a_lon = deg * np.arange(int(round(360 / deg)))
    a_h0 = int((SMOKE_A[0] - dt.datetime(1959, 1, 1)).total_seconds() // 3600)
    nA = int((SMOKE_A[1] - SMOKE_A[0]).total_seconds() // 3600) + 1
    a_hours = a_h0 + np.arange(nA, dtype=np.int64)
    # B: 4x finer, latitude DESCENDING, (time, level, latitude, longitude)
    fd = deg / 4.0
    b_lat = 90.0 - fd * np.arange(int(round(180 / fd)) + 1)
    b_lon = fd * np.arange(int(round(360 / fd)))
    b_h0 = int((SMOKE_B_AXIS[0] - dt.datetime(1900, 1, 1)).total_seconds()
               // 3600)
    nB = int((SMOKE_B_AXIS[1] - SMOKE_B_AXIS[0]).total_seconds() // 3600) + 1
    b_hours = b_h0 + np.arange(nB, dtype=np.int64)
    arrays_a = {"time": {"data": a_hours, "chunks": [len(a_hours)],
                         "dims": ["time"],
                         "units": "hours since 1959-01-01"},
                "level": {"data": lv, "chunks": [len(lv)], "dims": ["level"]},
                "latitude": {"data": a_lat, "chunks": [len(a_lat)],
                             "dims": ["latitude"]},
                "longitude": {"data": a_lon, "chunks": [len(a_lon)],
                              "dims": ["longitude"]}}
    arrays_b = {"time": {"data": b_hours, "chunks": [len(b_hours)],
                         "dims": ["time"],
                         "units": "hours since 1900-01-01 00:00:00"},
                "level": {"data": lv, "chunks": [len(lv)], "dims": ["level"]},
                "latitude": {"data": b_lat.astype(np.float32),
                             "chunks": [len(b_lat)], "dims": ["latitude"]},
                "longitude": {"data": b_lon.astype(np.float32),
                              "chunks": [len(b_lon)], "dims": ["longitude"]}}
    # ONE clock for both parts — hours since SMOKE_A[0] — so they describe
    # the same atmosphere at the same instant
    hA = (a_hours - a_h0).astype(np.float64)
    hB = (b_hours - b_h0).astype(np.float64) + (
        SMOKE_B_AXIS[0] - SMOKE_A[0]).total_seconds() / 3600.0
    # the two archives describe the same atmosphere: the field is a function
    # of the instant, so both sources see the same hours since 2021-12-20
    for code, (name, *_r) in VARIABLES.items():
        LA, LO = np.meshgrid(a_lat, a_lon, indexing="xy")      # [lon, lat]
        A = smoke_value(code, lv[None, :, None, None].astype(np.float64),
                        LA[None, None], LO[None, None],
                        hA[:, None, None, None]).astype(np.float32)
        if code == "t":
            i = int((SMOKE_OOB[0] - SMOKE_A[0]).total_seconds() // 3600)
            A[i, list(lv).index(SMOKE_OOB[1]), 3, 5] = 1000.0
        arrays_a[name] = {"data": A, "chunks": [8, len(lv), len(a_lon),
                                                len(a_lat)],
                          "dims": ["time", "level", "longitude", "latitude"],
                          "units": "K"}
        LA, LO = np.meshgrid(b_lat, b_lon, indexing="ij")      # [lat, lon]
        B = smoke_value(code, lv[None, :, None, None].astype(np.float64),
                        LA[None, None], LO[None, None],
                        hB[:, None, None, None]).astype(np.float32)
        arrays_b[name] = {"data": B, "chunks": [1, len(lv), len(b_lat),
                                                len(b_lon)],
                          "dims": ["time", "level", "latitude", "longitude"],
                          "units": "K"}
    _write_zarr(base, ZARR_A, arrays_a)
    # the 0.25-degree part holds only the four analysis hours a day (the
    # reader never asks for another; a zarr may omit chunks), which keeps
    # the synthetic archive a few tens of megabytes
    _write_zarr(base, ZARR_B, arrays_b, keep_chunk=lambda var, idx: (
        len(idx) == 1 or (b_h0 + idx[0]) % 6 == 0), attrs={
        "valid_time_start": str(SMOKE_B_VALID[0]),
        "valid_time_stop": str(SMOKE_B_VALID[1]),
        "valid_time_stop_era5t": str(SMOKE_B_AXIS[1].date())})
    return base


def make_smoke_sources(root, d_lo, d_hi, ad):
    """The synthetic archive, and the truth {(group, bin, frame): (stored
    float16 [H, W, 13] or None, reason or None)} for adapter `ad`'s
    variable. The truth reads the archive back through this module's own
    `Sources` (the smoke checks the binning, the seam, the shards and the
    read-back; `tests/test_family12_era5.py` checks the reader and the
    regrid against independent arithmetic)."""
    os.environ["ERA5_SMOKE_DEG"] = f"{SMOKE_DEG:g}"
    base = write_smoke_archive(root, SMOKE_DEG)
    s = Sources(base, True, SMOKE_DEG)
    probe = type(ad)()
    lo, hi = probe.bounds()
    t_lo = f10b.seconds_since_epoch(d_lo)
    t_hi = f10b.seconds_since_epoch(d_hi) + 86400 - 1
    truth = {}
    for b in sh.bins_overlapping(t_lo, t_hi):
        for f in range(FRAMES_PER_BIN):
            w = frame_instant(b, f)
            src, ti = s.where(w)
            if src is None:
                truth[(ad.store, b, f)] = (None, ti)
                continue
            arr, _st = s.frame(probe.src_name, src, ti)
            v = arr.reshape(-1, probe.C) * probe.scale
            with np.errstate(invalid="ignore"):
                v[np.isfinite(v) & ((v < lo) | (v > hi))] = np.nan
            truth[(ad.store, b, f)] = (
                v.reshape(arr.shape).astype(np.float16), None)
    return truth

"""The SHARDED tier-G layout — gridded fields finer than 0.25° (E-082 wave 2).

PLAIN ENGLISH. Family 7's gridded groups are one array per group, ordered by
five-day bin, so one bin is one contiguous byte range. That stops working
below 0.25°: a daily 0.05° frame is 52 MB per channel, a forecast anchor needs
a few hundred kilometres of it, and most of the pixels of a daily satellite
field are cloud, night or land. This module stores such a field as
compressed TILES instead, so a reader fetches only the few tiles around the
place it predicts, and an unobserved pixel costs almost nothing.

THE LAYOUT (family 1.gf note, "Storage"; one GROUP = one grid):

  <group>/tile_grid.json          what a tile IS: H, W, C, the tile size, the
                                  tile counts and the pad, dtype and missing
                                  value, F and frame_seconds, the channels,
                                  the grid's projection and the coordinate
                                  GENERATOR (an affine rule, not arrays), the
                                  lat/lon of every tile corner, the index
                                  file's shape and header length, the codec.
  <group>/shard_index.npy         one row per bin (a structured array — no
                                  pickle): bin, year, the shard and index
                                  paths, the shard's byte size, frames
                                  present / missing, a frame bitmask, tiles
                                  stored, valid pixels and valid fraction per
                                  channel.
  <group>/<yyyy>/bin_<NNNN>.zst   ONE SHARD PER (group, bin): the compressed
                                  tiles of the bin's F frames, concatenated,
                                  frames ascending, tiles row-major.
                                  <yyyy> is the calendar year of the bin's
                                  FIRST day (the Hub allows 10 k files per
                                  folder; a year folder holds 73 bins).
  <group>/<yyyy>/bin_<NNNN>.idx.npy
                                  the shard's own index, int64 little-endian
                                  [F, n_tiles_y, n_tiles_x, 2] of (offset,
                                  length) into the shard. Its header length is
                                  fixed for a group and recorded in
                                  tile_grid.json, so entry (f, ty, tx) is the
                                  16 bytes at header + 16 * ((f * nty + ty) *
                                  ntx + tx) — one range read.

  (offset, length) semantics — the whole of "a missing day is never a silent
  gap" lives here:
      offset >= 0, length > 0   a stored tile
      offset >= 0, length == 0  the frame exists and this tile has no valid
                                pixel (open ocean for a land field, the
                                tropics for sea ice); it is never stored
      offset == -1, length == 0 THE FRAME IS NOT IN THE SHARD. Why is counted
                                by name in store.json (a day absent upstream,
                                a day before or after the record), and the
                                shard_index row's frame_mask says the same.

A TILE is T x T x C (T = 256) in C order [row, col, channel], little-endian,
in the group's dtype, zstd-compressed on its own (one frame of one zstd
stream per tile, so any tile decompresses without its neighbours). Edge tiles
are PADDED to T x T with the missing value; the pad is recorded
(`pad_rows`, `pad_cols`, and per-tile `row_extents` / `col_extents`) and
`check_sharded` asserts every pad pixel is missing. Missing values:

  float16   NaN
  uint8     255 (`U8_MISSING`). A uint8 group stores 0..254; a float input is
            rounded half-to-even (`np.rint`), NaN becomes 255, and anything
            outside 0..254 after rounding is REFUSED — the adapter masks its
            bounds first (contract rule 3). The reader returns float32 with
            NaN unless asked for the raw bytes.

A READ is two range reads per tile (`read_tile`): the index entry, then the
tile — on a local path or an HTTP URL (the Hub's `resolve/main/...`),
through `family1.adapters._common.range_reader`, which refuses a 200 where a
206 was asked for (the host ignored the Range and is sending the file).

STORED BYTES ≈ pixels × frames × valid fraction × 2C × 1.2 in the note's
arithmetic; `tests/test_family1_sharded.py` measures the real number on a
synthetic 30 %-valid field and the probe measures it on the real source.

Nothing here touches the network except the reader's HTTP branch, and nothing
here knows about a particular source: `GridAdapter` (bottom) is the adapter
interface `ml/build_family1_stores.py` drives.
"""
import datetime as dt
import io
import json
import os
import random
import sys
import urllib.parse
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_ML = os.path.dirname(_HERE)
if _ML not in sys.path:
    sys.path.insert(0, _ML)

import build_family10_stores as f10b                            # noqa: E402
from build_family7 import atomic_json, read_json, sha256        # noqa: E402

FORMAT = "family1-sharded/1"
TILE = 256
BIN_SECONDS = f10b.PENTAD_SECONDS                 # 432,000
EPOCH = f10b.START                                # 1982-01-01
DTYPES = {"float16": np.dtype("<f2"), "uint8": np.dtype("u1")}
U8_MISSING = 255
INDEX_DTYPE = np.dtype("<i8")
FRAME_ABSENT = -1
DEFAULT_LEVEL = 15        # measured on a real ASI frame: 0.15 s, 5 % under level 9
MAX_FRAMES_BITMASK = 64

# HASHING AND CHECKING A WHOLE STORE. `check_store` hashes every file on a
# thread pool: hashlib releases the GIL for any update larger than 2 KiB, so
# HASH_WORKERS threads reading HASH_BUF-sized chunks hash in parallel up to
# what the disk delivers. `progress(label, done, total)` is called every
# PROGRESS_EVERY_FILES files hashed and every PROGRESS_EVERY_BINS bins
# checked, so a live log shows a long check moving (ml/CLAUDE.md §5.25).
HASH_WORKERS = 8
HASH_BUF = 1 << 22                                   # 4 MiB per read
PROGRESS_EVERY_FILES = 200
PROGRESS_EVERY_BINS = 100

# THE REASONS A MISSING FRAME CAN GIVE, and the three of them that let a whole
# bin be skipped rather than stored as an empty shard. They live here, in the
# layout module, because they are a property of the layout and because an
# adapter can import this module without the circular import that naming them
# in `build_family1_stores` would cost (that module imports the adapters).
# `build_family1_stores` re-exports them under its own older names.
FRAME_ABSENT_UPSTREAM = "absent_upstream"
FRAME_OUTSIDE_RECORD = ("before_record", "after_record")
FRAME_NO_FRAME_IN_BIN = "no_frame_in_bin"
FRAME_SKIP_REASONS = FRAME_OUTSIDE_RECORD + (FRAME_NO_FRAME_IN_BIN,)


class ShardError(ValueError):
    """A shard, an index or a tile that does not match its declaration."""


def zstd():
    try:
        import zstandard
    except ImportError:                                     # pragma: no cover
        sys.exit("the sharded tier-G layout needs the `zstandard` package "
                 "(pip install zstandard) — it is in the family1-build "
                 "workflow's install step")
    return zstandard


# ================================================================= time ====
def bin_start_date(b):
    return EPOCH + dt.timedelta(days=int(b) * f10b.PENTAD_DAYS)


def bin_year(b):
    return bin_start_date(b).year


def bins_overlapping(t_lo, t_hi):
    """Every bin with a second inside [t_lo, t_hi] (seconds since 1982)."""
    return list(range(int(t_lo) // BIN_SECONDS, int(t_hi) // BIN_SECONDS + 1))


def frame_start_seconds(b, f, frame_seconds):
    return int(b) * BIN_SECONDS + int(f) * int(frame_seconds)


def frame_datetime(b, f, frame_seconds):
    """The UTC instant frame f of bin b starts at."""
    return dt.datetime(EPOCH.year, EPOCH.month, EPOCH.day) + dt.timedelta(
        seconds=frame_start_seconds(b, f, frame_seconds))


def frame_day(b, f, frame_seconds):
    """The calendar day frame f of bin b starts on (a date)."""
    return frame_datetime(b, f, frame_seconds).date()


def shard_relpath(b):
    return f"{bin_year(b):04d}/bin_{int(b):04d}.zst"


def index_relpath(b):
    return f"{bin_year(b):04d}/bin_{int(b):04d}.idx.npy"


# ================================================================ tiling ===
def tile_layout(H, W, T=TILE):
    nty, ntx = -(-int(H) // T), -(-int(W) // T)
    return {"n_tiles_y": nty, "n_tiles_x": ntx,
            "pad_rows": nty * T - int(H), "pad_cols": ntx * T - int(W),
            "row_extents": [[i * T, min((i + 1) * T, int(H))]
                            for i in range(nty)],
            "col_extents": [[j * T, min((j + 1) * T, int(W))]
                            for j in range(ntx)]}


def index_header_bytes(shape):
    """The .npy (v1.0) header length for an int64 array of `shape`."""
    buf = io.BytesIO()
    np.lib.format.write_array_header_1_0(
        buf, {"descr": np.lib.format.dtype_to_descr(INDEX_DTYPE),
              "fortran_order": False, "shape": tuple(int(x) for x in shape)})
    return len(buf.getvalue())


def shard_index_dtype(C):
    return np.dtype([
        ("bin", "<i4"), ("year", "<i2"),
        ("frames_present", "<i2"), ("frames_missing", "<i2"),
        ("frame_mask", "<u8"), ("tiles_stored", "<i4"), ("nbytes", "<i8"),
        ("valid_pixels", "<i8", (int(C),)),
        ("valid_fraction", "<f4", (int(C),)),
        ("shard", "<U40"), ("index", "<U40")])


def make_spec(group, grid, channels, F, frame_seconds, dtype, tile=TILE,
              level=DEFAULT_LEVEL, latlon=None, footprint=None):
    """tile_grid.json's content for one group.

    `grid` is JSON-able: at least H, W, and the projection/coordinate
    description (x0, dx, y0, dy, row_order, crs, …). `latlon(x, y)` — the
    adapter's inverse projection — gives the lat/lon of every tile corner;
    it is called here and not stored.
    """
    H, W = int(grid["H"]), int(grid["W"])
    C = len(channels)
    F = int(F)
    if dtype not in DTYPES:
        raise ShardError(f"{group}: dtype {dtype!r} is not one of "
                         f"{sorted(DTYPES)}")
    if F < 1 or F > MAX_FRAMES_BITMASK:
        raise ShardError(f"{group}: frames_per_bin {F} — the shard index's "
                         f"frame_mask holds at most {MAX_FRAMES_BITMASK}; "
                         f"widen it before a half-hourly group is built")
    if F * int(frame_seconds) != BIN_SECONDS:
        raise ShardError(f"{group}: {F} frames x {frame_seconds} s != one "
                         f"five-day bin ({BIN_SECONDS} s)")
    lay = tile_layout(H, W, tile)
    ishape = [F, lay["n_tiles_y"], lay["n_tiles_x"], 2]
    spec = {
        "format": FORMAT, "group": group,
        "H": H, "W": W, "C": C, "tile": int(tile),
        "tile_shape": [int(tile), int(tile), C],
        "tile_order": "C order [row, col, channel], little-endian",
        **lay,
        "dtype": dtype,
        "missing": ("NaN" if dtype == "float16" else U8_MISSING),
        "pad_value": ("NaN" if dtype == "float16" else U8_MISSING),
        "frames_per_bin": F, "frame_seconds": int(frame_seconds),
        "bin_seconds": BIN_SECONDS, "epoch": str(EPOCH),
        "frame_rule": ("frame f of bin b covers [b * bin_seconds + f * "
                       "frame_seconds, + frame_seconds) seconds since the "
                       "epoch"),
        "channels": [{"name": n, "unit": u, "min": lo, "max": hi}
                     for (n, u, lo, hi) in channels],
        "grid": grid,
        "shard_path": "<yyyy>/bin_<NNNN>.zst (yyyy = year of the bin's "
                      "first day)",
        "index_path": "<yyyy>/bin_<NNNN>.idx.npy",
        "index_shape": ishape, "index_dtype": INDEX_DTYPE.str,
        "index_header_bytes": index_header_bytes(ishape),
        "index_semantics": ("(offset, length): offset >= 0 and length > 0 a "
                            "stored tile; offset >= 0 and length 0 a present "
                            "frame's tile with no valid pixel (not stored); "
                            "offset -1 and length 0 the frame is not in the "
                            "shard"),
        "compression": {"codec": "zstd", "level": int(level),
                        "unit": "one zstd frame per tile"},
    }
    if footprint is not None:
        spec["footprint"] = footprint
    if latlon is not None:
        g = grid
        xs = [g["x0"] + e[0] * g["dx"] for e in lay["col_extents"]] + \
            [g["x0"] + lay["col_extents"][-1][1] * g["dx"]]
        ys = [g["y0"] + e[0] * g["dy"] for e in lay["row_extents"]] + \
            [g["y0"] + lay["row_extents"][-1][1] * g["dy"]]
        X, Y = np.meshgrid(np.array(xs, np.float64), np.array(ys, np.float64))
        lat, lon = latlon(X, Y)
        spec["tile_corners"] = {
            "x": [float(v) for v in xs], "y": [float(v) for v in ys],
            "lat": np.round(lat, 6).tolist(), "lon": np.round(lon, 6).tolist(),
            "note": "corner [i][j] is at (x[j], y[i]) — the grid-edge "
                    "positions of tile boundaries; the last is the real grid "
                    "edge, not the pad's"}
    return spec


# ============================================================== encoding ===
def encode_frame(arr, spec):
    """[H, W, C] (or [H, W] when C == 1) -> the stored dtype, validated."""
    H, W, C = spec["H"], spec["W"], spec["C"]
    a = np.asarray(arr)
    if a.ndim == 2 and C == 1:
        a = a[:, :, None]
    if a.shape != (H, W, C):
        raise ShardError(f"{spec['group']}: a frame of shape {a.shape}, the "
                         f"group is {(H, W, C)}")
    dt_ = DTYPES[spec["dtype"]]
    if spec["dtype"] == "float16":
        if a.dtype.kind != "f":
            raise ShardError(f"{spec['group']}: float16 group given "
                             f"{a.dtype}")
        if np.isinf(a).any():
            raise ShardError(f"{spec['group']}: an infinite value — mask it "
                             f"as NaN (and count it) before the writer")
        with np.errstate(over="ignore", invalid="ignore"):
            out = a.astype(dt_)
        if np.isinf(out).any():
            raise ShardError(f"{spec['group']}: a value past float16's range")
        return out
    if a.dtype == np.uint8:
        return a
    if a.dtype.kind != "f":
        raise ShardError(f"{spec['group']}: uint8 group given {a.dtype}")
    r = np.rint(a)
    nan = np.isnan(r)
    bad = ~nan & ((r < 0) | (r > U8_MISSING - 1))
    if bad.any():
        raise ShardError(f"{spec['group']}: {int(bad.sum())} value(s) outside "
                         f"0..{U8_MISSING - 1} — mask bounds before writing")
    out = np.where(nan, U8_MISSING, r).astype(np.uint8)
    return out


def valid_mask(stored, dtype):
    if dtype == "float16":
        return ~np.isnan(stored)
    return stored != U8_MISSING


def decode(stored, dtype):
    """Stored tile/frame -> float32 with NaN for missing."""
    if dtype == "float16":
        return stored.astype(np.float32)
    out = stored.astype(np.float32)
    out[stored == U8_MISSING] = np.nan
    return out


def _pad_tile(stored, spec, ty, tx):
    T, C = spec["tile"], spec["C"]
    r0, r1 = spec["row_extents"][ty]
    c0, c1 = spec["col_extents"][tx]
    blk = stored[r0:r1, c0:c1]
    if blk.shape[:2] == (T, T):
        return np.ascontiguousarray(blk)
    fill = np.nan if spec["dtype"] == "float16" else U8_MISSING
    t = np.full((T, T, C), fill, DTYPES[spec["dtype"]])
    t[:r1 - r0, :c1 - c0] = blk
    return t


# ================================================================ writer ===
class ShardWriter:
    """Writes one bin's shard + index for one group."""

    def __init__(self, spec):
        self.spec = spec
        self.cctx = zstd().ZstdCompressor(
            level=int(spec["compression"]["level"]))

    def write_bin(self, b, frames, shard_path, index_path):
        """`frames` is a length-F list of arrays or None (absent).

        Returns the shard_index row as a dict. The shard and the index are
        written to temporary siblings and renamed, index LAST, so a present
        index always describes a complete shard.
        """
        sp = self.spec
        F, nty, ntx = sp["frames_per_bin"], sp["n_tiles_y"], sp["n_tiles_x"]
        C, H, W = sp["C"], sp["H"], sp["W"]
        if len(frames) != F:
            raise ShardError(f"bin {b}: {len(frames)} frames, F = {F}")
        idx = np.zeros((F, nty, ntx, 2), INDEX_DTYPE)
        valid = np.zeros(C, np.int64)
        present = mask = tiles = 0
        os.makedirs(os.path.dirname(shard_path) or ".", exist_ok=True)
        tmp = f"{shard_path}.tmp{os.getpid()}"
        off = 0
        with open(tmp, "wb") as fh:
            for f, arr in enumerate(frames):
                if arr is None:
                    idx[f, :, :, 0] = FRAME_ABSENT
                    idx[f, :, :, 1] = 0
                    continue
                st = encode_frame(arr, sp)
                present += 1
                mask |= 1 << f
                valid += valid_mask(st, sp["dtype"]).reshape(-1, C) \
                    .sum(axis=0)
                for ty in range(nty):
                    for tx in range(ntx):
                        t = _pad_tile(st, sp, ty, tx)
                        idx[f, ty, tx, 0] = off
                        if not valid_mask(t, sp["dtype"]).any():
                            idx[f, ty, tx, 1] = 0
                            continue
                        blob = self.cctx.compress(t.tobytes())
                        fh.write(blob)
                        idx[f, ty, tx, 1] = len(blob)
                        off += len(blob)
                        tiles += 1
        os.replace(tmp, shard_path)
        itmp = f"{index_path}.tmp{os.getpid()}.npy"
        np.save(itmp, idx)
        want = sp["index_header_bytes"]
        got = os.path.getsize(itmp) - idx.nbytes
        if got != want:
            os.remove(itmp)
            raise ShardError(f"bin {b}: index header is {got} bytes, "
                             f"tile_grid.json declares {want}")
        os.replace(itmp, index_path)
        denom = max(present, 1) * H * W
        return {"bin": int(b), "year": bin_year(b),
                "frames_present": present, "frames_missing": F - present,
                "frame_mask": mask, "tiles_stored": tiles, "nbytes": off,
                "valid_pixels": valid.tolist(),
                "valid_fraction": (valid / denom).tolist(),
                "shard": shard_relpath(b), "index": index_relpath(b)}


def entries_to_array(entries, C):
    dt_ = shard_index_dtype(C)
    out = np.zeros(len(entries), dt_)
    for i, e in enumerate(sorted(entries, key=lambda e: e["bin"])):
        for k in dt_.names:
            out[i][k] = e[k]
    return out


def array_to_entries(arr):
    out = []
    for r in arr:
        e = {}
        for k in arr.dtype.names:
            v = r[k]
            e[k] = v.tolist() if hasattr(v, "tolist") else v
        out.append(e)
    return out


def save_shard_index(path, entries, C):
    arr = entries_to_array(entries, C)
    bins = arr["bin"]
    if len(bins) != len(np.unique(bins)):
        raise ShardError(f"{path}: a bin appears twice")
    tmp = f"{path}.tmp{os.getpid()}.npy"
    np.save(tmp, arr, allow_pickle=False)
    os.replace(tmp, path)
    return arr


def load_shard_index(path_or_bytes):
    if isinstance(path_or_bytes, (bytes, bytearray)):
        return np.load(io.BytesIO(path_or_bytes), allow_pickle=False)
    return np.load(path_or_bytes, allow_pickle=False)


# ================================================================ reader ===
def _is_url(s):
    return str(s).startswith(("http://", "https://"))


def _join(base, rel):
    if _is_url(base):
        return base.rstrip("/") + "/" + urllib.parse.quote(rel)
    return os.path.join(base, rel)


class _Source:
    """Whole-file and ranged reads, on a directory or a URL prefix."""

    def __init__(self, base, headers=None, attempts=4):
        self.base = str(base)
        self.headers = dict(headers or {})
        self.attempts = attempts
        self.url = _is_url(self.base)
        self.ranges = 0

    def get(self, rel):
        p = _join(self.base, rel)
        if not self.url:
            with open(p, "rb") as fh:
                return fh.read()
        return f10b.http_bytes(p, headers=self.headers)

    def read_at(self, rel, off, n):
        self.ranges += 1
        p = _join(self.base, rel)
        if not self.url:
            with open(p, "rb") as fh:
                fh.seek(off)
                b = fh.read(n)
            if len(b) != n:
                raise ShardError(f"{p}: {len(b)} of {n} bytes at {off}")
            return b
        from family1.adapters import _common as cm
        b = cm.range_reader(p, attempts=self.attempts,
                            headers=self.headers)(off, n)
        if len(b) != n:
            raise ShardError(f"{p}: {len(b)} of {n} bytes at {off}")
        return b


class ShardedGroup:
    """One group, opened on a local directory or an HTTP prefix."""

    def __init__(self, base, headers=None, attempts=4):
        self.src = _Source(base, headers, attempts)
        self.spec = json.loads(self.src.get("tile_grid.json"))
        if self.spec.get("format") != FORMAT:
            raise ShardError(f"{base}: tile_grid.json format "
                             f"{self.spec.get('format')!r}, not {FORMAT}")
        self._dctx = zstd().ZstdDecompressor()
        self._index = None

    @property
    def shard_index(self):
        if self._index is None:
            self._index = load_shard_index(self.src.get("shard_index.npy"))
        return self._index

    def entry(self, b, f, ty, tx):
        """(offset, length) for one tile — ONE range read."""
        sp = self.spec
        F, nty, ntx = sp["frames_per_bin"], sp["n_tiles_y"], sp["n_tiles_x"]
        if not (0 <= f < F and 0 <= ty < nty and 0 <= tx < ntx):
            raise IndexError(f"(frame {f}, tile {ty},{tx}) outside "
                             f"[{F}, {nty}, {ntx}]")
        flat = (f * nty + ty) * ntx + tx
        raw = self.src.read_at(index_relpath(b),
                               sp["index_header_bytes"] + 16 * flat, 16)
        o, n = np.frombuffer(raw, INDEX_DTYPE)
        return int(o), int(n)

    def tile_bytes(self, b, off, n):
        return self.src.read_at(shard_relpath(b), off, n)

    def decode_tile(self, blob):
        sp = self.spec
        T, C = sp["tile"], sp["C"]
        dt_ = DTYPES[sp["dtype"]]
        want = T * T * C * dt_.itemsize
        raw = self._dctx.decompress(blob, max_output_size=want)
        if len(raw) != want:
            raise ShardError(f"a tile decompressed to {len(raw)} bytes, the "
                             f"declared tile is {want}")
        return np.frombuffer(raw, dt_).reshape(T, T, C)

    def empty_tile(self):
        sp = self.spec
        fill = np.nan if sp["dtype"] == "float16" else U8_MISSING
        return np.full((sp["tile"], sp["tile"], sp["C"]), fill,
                       DTYPES[sp["dtype"]])

    def read_tile(self, b, f, ty, tx, raw=False, crop=False):
        """The tile, or None when the FRAME is not in the shard.

        Two range reads (index entry, tile). float32 with NaN unless `raw`;
        T x T with the pad unless `crop`.
        """
        off, n = self.entry(b, f, ty, tx)
        if off == FRAME_ABSENT:
            return None
        t = self.decode_tile(self.tile_bytes(b, off, n)) if n else \
            self.empty_tile()
        if crop:
            r0, r1 = self.spec["row_extents"][ty]
            c0, c1 = self.spec["col_extents"][tx]
            t = t[:r1 - r0, :c1 - c0]
        return t if raw else decode(t, self.spec["dtype"])

    def read_frame(self, b, f, raw=False):
        """The whole [H, W, C] frame (all tiles), or None if absent."""
        sp = self.spec
        fill = np.nan if sp["dtype"] == "float16" else U8_MISSING
        out = np.full((sp["H"], sp["W"], sp["C"]), fill, DTYPES[sp["dtype"]])
        for ty in range(sp["n_tiles_y"]):
            for tx in range(sp["n_tiles_x"]):
                t = self.read_tile(b, f, ty, tx, raw=True, crop=True)
                if t is None:
                    return None
                r0, r1 = sp["row_extents"][ty]
                c0, c1 = sp["col_extents"][tx]
                out[r0:r1, c0:c1] = t
        return out if raw else decode(out, sp["dtype"])


_OPEN = {}


def open_group(base, headers=None):
    key = (str(base), tuple(sorted((headers or {}).items())))
    g = _OPEN.get(key)
    if g is None:
        g = _OPEN[key] = ShardedGroup(base, headers)
    return g


def read_tile(group_dir, b, frame, ty, tx, raw=False, crop=False,
              headers=None):
    """One tile of one frame of one bin: two range reads (index, tile)."""
    return open_group(group_dir, headers).read_tile(b, frame, ty, tx, raw=raw,
                                                    crop=crop)


# ================================================================= check ===
def check_sharded(group_dir, sample=None, seed=0, bounds=True,
                  progress=None):
    """Every index entry against its shard and its declaration.

    Full (sample=None): every shard's size, the index's shape, dtype and
    header, offsets contiguous in write order, absent frames all (-1, 0),
    EVERY stored tile decompressed to the declared size, its pad all missing,
    its values inside the channel bounds, the per-bin valid-pixel counts and
    frame counts equal to shard_index.npy's. `sample=k` decompresses only k
    random stored tiles (the publish's restore check) and still checks every
    index's structure. Raises ShardError; returns a summary.
    `progress(label, done, total)` is called every PROGRESS_EVERY_BINS bins
    and after the last one.
    """
    g = ShardedGroup(group_dir)
    sp = g.spec
    arr = g.shard_index
    F, nty, ntx = sp["frames_per_bin"], sp["n_tiles_y"], sp["n_tiles_x"]
    C, T = sp["C"], sp["tile"]
    dt_ = DTYPES[sp["dtype"]]
    ishape = (F, nty, ntx, 2)
    if arr.dtype != shard_index_dtype(C):
        raise ShardError(f"{group_dir}: shard_index.npy dtype {arr.dtype}")
    bins = arr["bin"]
    if len(bins) and (np.diff(bins) <= 0).any():
        raise ShardError(f"{group_dir}: shard_index.npy bins are not strictly "
                         f"increasing")
    lo = np.array([c["min"] for c in sp["channels"]], np.float64)
    hi = np.array([c["max"] for c in sp["channels"]], np.float64)
    stored_tiles = []
    tot = {"bins": int(len(arr)), "frames_present": 0, "frames_missing": 0,
           "tiles_stored": 0, "tiles_checked": 0, "bytes": 0}
    label = f"check {os.path.basename(os.path.normpath(group_dir))}"
    for i_row, row in enumerate(arr, 1):
        b = int(row["bin"])
        if row["shard"] != shard_relpath(b) or row["index"] != index_relpath(b):
            raise ShardError(f"bin {b}: paths {row['shard']}, {row['index']}")
        if int(row["year"]) != bin_year(b):
            raise ShardError(f"bin {b}: year {row['year']}")
        sp_path = os.path.join(group_dir, row["shard"])
        size = os.path.getsize(sp_path)
        if size != int(row["nbytes"]):
            raise ShardError(f"bin {b}: shard is {size} bytes, the index "
                             f"row says {int(row['nbytes'])}")
        ip = os.path.join(group_dir, row["index"])
        idx = np.load(ip, allow_pickle=False)
        if idx.shape != ishape or idx.dtype != INDEX_DTYPE:
            raise ShardError(f"bin {b}: index {idx.shape} {idx.dtype}, "
                             f"declared {ishape} {INDEX_DTYPE}")
        if os.path.getsize(ip) - idx.nbytes != sp["index_header_bytes"]:
            raise ShardError(f"bin {b}: index header length differs from "
                             f"tile_grid.json's")
        present = mask = 0
        pos = 0
        n_stored = 0
        for f in range(F):
            o = idx[f, :, :, 0]
            n = idx[f, :, :, 1]
            if (o == FRAME_ABSENT).any():
                if not ((o == FRAME_ABSENT).all() and (n == 0).all()):
                    raise ShardError(f"bin {b} frame {f}: partly absent")
                continue
            present += 1
            mask |= 1 << f
            for ty in range(nty):
                for tx in range(ntx):
                    oo, nn = int(o[ty, tx]), int(n[ty, tx])
                    if oo != pos or nn < 0:
                        raise ShardError(f"bin {b} ({f},{ty},{tx}): offset "
                                         f"{oo}, expected {pos}")
                    if nn:
                        stored_tiles.append((b, f, ty, tx, oo, nn))
                        n_stored += 1
                    pos += nn
        if pos != size:
            raise ShardError(f"bin {b}: index lengths sum to {pos}, the "
                             f"shard is {size} bytes")
        if (present != int(row["frames_present"])
                or F - present != int(row["frames_missing"])
                or mask != int(row["frame_mask"])
                or n_stored != int(row["tiles_stored"])):
            raise ShardError(f"bin {b}: frames/tiles disagree with "
                             f"shard_index.npy ({present}, {mask:b}, "
                             f"{n_stored})")
        tot["frames_present"] += present
        tot["frames_missing"] += F - present
        tot["tiles_stored"] += n_stored
        tot["bytes"] += size
        if sample is None:
            valid = np.zeros(C, np.int64)
            with open(sp_path, "rb") as fh:
                blob = fh.read()
            for (_b, f, ty, tx, oo, nn) in [t for t in stored_tiles
                                            if t[0] == b]:
                t = g.decode_tile(blob[oo:oo + nn])
                valid += _check_tile(t, sp, ty, tx, lo, hi, bounds, b, f)
                tot["tiles_checked"] += 1
            if valid.tolist() != [int(v) for v in row["valid_pixels"]]:
                raise ShardError(f"bin {b}: {valid.tolist()} valid pixels "
                                 f"decoded, shard_index.npy says "
                                 f"{row['valid_pixels'].tolist()}")
            stored_tiles = []
        if progress is not None and (i_row % PROGRESS_EVERY_BINS == 0
                                     or i_row == len(arr)):
            progress(label, i_row, len(arr))
    if sample is not None and stored_tiles:
        rng = random.Random(seed)
        pick = rng.sample(stored_tiles, min(int(sample), len(stored_tiles)))
        for (b, f, ty, tx, oo, nn) in pick:
            with open(os.path.join(group_dir, shard_relpath(b)), "rb") as fh:
                fh.seek(oo)
                t = g.decode_tile(fh.read(nn))
            _check_tile(t, sp, ty, tx, lo, hi, bounds, b, f)
            tot["tiles_checked"] += 1
        tot["sampled"] = [list(p[:4]) for p in pick]
    tot["tile_nbytes"] = T * T * C * dt_.itemsize
    return tot


def _check_tile(t, sp, ty, tx, lo, hi, bounds, b, f):
    r0, r1 = sp["row_extents"][ty]
    c0, c1 = sp["col_extents"][tx]
    ok = valid_mask(t, sp["dtype"])
    pad = np.ones(ok.shape[:2], bool)
    pad[:r1 - r0, :c1 - c0] = False
    if ok[pad].any():
        raise ShardError(f"bin {b} ({f},{ty},{tx}): a pad pixel is not "
                         f"missing")
    if not ok.any():
        raise ShardError(f"bin {b} ({f},{ty},{tx}): a stored tile with no "
                         f"valid pixel")
    if bounds:
        v = decode(t, sp["dtype"]).astype(np.float64)
        with np.errstate(invalid="ignore"):
            out = ok & ((v < lo) | (v > hi))
        if out.any():
            raise ShardError(f"bin {b} ({f},{ty},{tx}): {int(out.sum())} "
                             f"value(s) outside the channel bounds")
    return ok.reshape(-1, sp["C"]).sum(axis=0)


def store_files(store_dir, groups):
    """Every file a sharded store is made of, relative to the store."""
    out = []
    for g in groups:
        gd = os.path.join(store_dir, g)
        out += [f"{g}/tile_grid.json", f"{g}/shard_index.npy"]
        arr = load_shard_index(os.path.join(gd, "shard_index.npy"))
        for r in arr:
            out += [f"{g}/{r['shard']}", f"{g}/{r['index']}"]
    return out


def sha256_block(store_dir, names, workers=HASH_WORKERS, progress=None,
                 expect=None):
    """{name: sha256} of every named file under `store_dir`, in `names` order,
    hashed on `workers` threads (1 or less: serially, in order).

    With `expect` ({name: sha256}) it REFUSES — ShardError — on the first file
    whose hash differs, and cancels every hash not yet started. `progress
    (label, done, total)` is called every PROGRESS_EVERY_FILES files and after
    the last one. The assembler builds store.json's block with this and
    `check_store` verifies it with this, so both read the files the same way.
    """
    total = len(names)
    out = {}

    def one(n):
        return n, sha256(os.path.join(store_dir, n), HASH_BUF)

    def got(i, n, h):
        if expect is not None and h != expect[n]:
            raise ShardError(f"{store_dir}/{n}: sha256 differs from "
                             f"store.json")
        out[n] = h
        if progress is not None and (i % PROGRESS_EVERY_FILES == 0
                                     or i == total):
            progress("sha256", i, total)

    if workers <= 1:
        for i, n in enumerate(names, 1):
            got(i, *one(n))
        return out
    ex = ThreadPoolExecutor(max_workers=int(workers))
    try:
        futs = [ex.submit(one, n) for n in names]
        for i, fu in enumerate(as_completed(futs), 1):
            got(i, *fu.result())
    finally:
        ex.shutdown(wait=True, cancel_futures=True)
    return {n: out[n] for n in names}


def check_store(store_dir, sample=None, seed=0, workers=HASH_WORKERS,
                progress=None):
    """store.json's sha256 block and every group's `check_sharded`.

    Every file is hashed (on `workers` threads) and every index's STRUCTURE
    is checked in full whatever `sample` is; `sample` is handed to
    `check_sharded` unchanged, so None decompresses every stored tile and k
    decompresses k random ones per group. `progress(label, done, total)` —
    see `sha256_block` and `check_sharded`.
    """
    meta = read_json(os.path.join(store_dir, "store.json"), {})
    if meta.get("tier") != "G" or meta.get("layout") != "sharded":
        raise ShardError(f"{store_dir}: store.json is not a sharded tier-G "
                         f"store")
    groups = sorted(meta.get("groups") or {})
    names = store_files(store_dir, groups)
    block = meta.get("sha256") or {}
    if sorted(block) != sorted(names):
        extra = sorted(set(block) - set(names))
        miss = sorted(set(names) - set(block))
        raise ShardError(f"{store_dir}: store.json's sha256 block and the "
                         f"shard indices disagree (only in the block: "
                         f"{extra[:4]}, only in the indices: {miss[:4]})")
    sha256_block(store_dir, names, workers=workers, progress=progress,
                 expect=block)
    out = {"files": len(names), "groups": {}}
    per_year = {}
    for g in groups:
        s = check_sharded(os.path.join(store_dir, g), sample=sample,
                          seed=seed, progress=progress)
        gm = meta["groups"][g]
        for k in ("bins", "frames_present", "frames_missing",
                  "tiles_stored", "bytes"):
            if int(gm.get(k, -1)) != int(s[k]):
                raise ShardError(f"{store_dir}: group {g} {k} is {s[k]}, "
                                 f"store.json says {gm.get(k)}")
        arr = load_shard_index(os.path.join(store_dir, g, "shard_index.npy"))
        for r in arr:
            y = str(int(r["year"]))
            per_year.setdefault(y, {}).setdefault(g, 0)
            per_year[y][g] += int(r["frames_present"])
        out["groups"][g] = s
    if per_year != (meta.get("per_year") or {}):
        raise ShardError(f"{store_dir}: per-year frame counts "
                         f"{per_year} differ from store.json's "
                         f"{meta.get('per_year')}")
    return out


# ========================================================= the adapter ====
class GridAdapter(f10b.SourceAdapter):
    """A tier-G source: frames on fixed grids, not rows.

    Declares `tier = "G"`, `layout = "sharded"`, `frames_per_bin`,
    `frame_seconds`, `dtype`, and either `grid` (one group, named after the
    store) or `grids` ({group: grid}); each grid is JSON-able (H, W, the
    projection text, the affine coordinate generator). Implements
    `fetch_frames(ctx, wanted)` — `wanted` is a list of (group, bin, frame) —
    yielding `(group, bin, frame, array_or_None, counts)`. A None array is a
    frame that is not in the source; `counts["frame_missing"]` names why:

      absent_upstream   the product SHOULD have this frame and does not — a
                        hole, and the reader must be told about it
      before_record     the bin is earlier than the product's first frame
      after_record      the bin is later than its last
      no_frame_in_bin   the product's CADENCE IS COARSER than the five-day
                        bin, so this bin was never going to have a frame of
                        its own (a monthly map filed under the bin holding its
                        month's 15th leaves five bins in six empty). Never a
                        substitute for `absent_upstream`: one says "there is
                        nothing to have", the other "what should be here is
                        missing".

    A bin ALL of whose frames are missing for one of the last three writes no
    shard at all and is counted by name (`build_family1_stores.SKIP_BIN_REASONS`).
    An input the adapter could not READ is `ctx.note_absent(...)`, never a None
    frame — that is a refusal, and the year is not marked.

    `fetch_year(ctx, year)` is the framework's call and is defined here: it
    hands `fetch_frames` every frame of the year's bins (build_family1_stores
    decides which bins those are). An adapter that yields 3-tuples
    `(bin, frame, array)` is accepted for a single-group store.
    """

    tier = "G"
    layout = "sharded"
    frames_per_bin = 5
    frame_seconds = 86400
    dtype = "float16"
    tile = TILE
    zstd_level = DEFAULT_LEVEL
    grid = None
    grids = None
    note_estimate = None          # {"bytes": ..., "what": ...} from the note

    def group_grids(self):
        if self.grids:
            return dict(self.grids)
        if self.grid:
            return {self.store: self.grid}
        raise ShardError(f"{self.store}: a tier-G adapter declares `grid` or "
                         f"`grids`")

    def grid_latlon(self, group, x, y):
        """Inverse projection for tile corners; None if not provided."""
        return None

    def specs(self):
        out = {}
        for g, grid in self.group_grids().items():
            ll = (lambda X, Y, _g=g: self.grid_latlon(_g, X, Y))
            probe = self.grid_latlon(g, np.zeros(1), np.zeros(1))
            out[g] = make_spec(
                g, grid, self.channels, self.frames_per_bin,
                self.frame_seconds, self.dtype, tile=self.tile,
                level=self.zstd_level,
                latlon=(ll if probe is not None else None),
                footprint={"log2_fp": self.log2_fp,
                           "log2_dt": self.log2_dt})
        return out

    def record_frames(self, ctx, group):
        """How many frames the whole record holds — for the probe's
        extrapolation. None if the adapter cannot say."""
        return None

    def group_subset(self):
        """The groups THIS INSTANCE was restricted to, or None for the whole
        product (E-082 wave 7).

        A store whose groups are tiles and whose one year is far more than a
        six-hour hosted lane — `canopy30` is 261 tiles and 11 to 21 hours,
        `lossyear` 280 — is fetched as several TILE-SUBSET LANES, one runner
        each, and the framework names the lane after the subset
        (`build_family1_stores.group_lane`). An adapter that answers with a
        list here is saying "this run is a lane of the product, not the
        product"; the default answers None, so every other tier-G store keeps
        the unnamed lane it has always written.

        It is deliberately NOT inferred from whatever environment variable
        narrowed the groups: a knob that chooses WHICH PRODUCT to build
        (`LST05_GROUPS` picks a satellite, `LAI500_GROUPS` picks Terra or
        Aqua) selects a whole store, not a subset of one, and a store already
        on the Hub must not acquire a lane because of one.
        """
        return None

    def fetch_frames(self, ctx, wanted):
        raise NotImplementedError

    def fetch_year(self, ctx, year):
        wanted = ctx.grid_wanted(year)
        yield from self.fetch_frames(ctx, wanted)

    def fetch_month(self, ctx, year, month):             # pragma: no cover
        raise NotImplementedError("a tier-G probe uses fetch_frames")


def save_json(path, obj):
    atomic_json(path, obj)

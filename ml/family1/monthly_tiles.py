"""MONTHLY TILES — per-(year, month) sum / count / m2 of a sharded tier-G
group, in the group's OWN tiling (E-089, `ml/plans/E089_fine_grid_monthly_sums.md`).

PLAIN ENGLISH. A fine store (`ml/family1/sharded.py`) keeps one file of
compressed 256 x 256 tiles per five-day bin. This module keeps, beside it, one
file per CALENDAR MONTH holding, for every pixel and channel, the SUM of that
month's finite values, how many there were (COUNT) and the sum of squared
deviations about that month's own mean (M2) — the three numbers from which a
mean and a standard deviation over ANY set of months is composed exactly
(Chan et al.). It is `sharded.py`'s layout with a calendar month in place of
a five-day bin and the three statistics in place of the F frames, so a reader
that knows the store's tiles knows these.

THE LAYOUT (one directory per group, e.g. `.../monthly/native/`):

  tile_grid.json                  `make_spec`: the source group's H, W, C,
                                  tile size, tile counts, pad, row / col
                                  extents and grid (copied), the channels in
                                  PHYSICAL units (a uint8 "K - 160" store is
                                  summed in kelvin: `value_offset` 160), the
                                  three PARTS and their dtypes, the index's
                                  shape and header length, the codec.
  <yyyy>/m_<yyyy>-<mm>.zst        ONE FILE PER (year, month) with at least one
                                  frame: the tiles' parts, concatenated in
                                  write order — tiles row-major (ty, then tx),
                                  and within a tile the parts sum, count, m2.
                                  So one tile's three parts are ADJACENT: a
                                  mean is ONE range read (sum + count), a
                                  standard deviation the same range extended
                                  by m2.
  <yyyy>/m_<yyyy>-<mm>.idx.npy    int64 little-endian [n_tiles_y, n_tiles_x,
                                  3, 2] of (offset, length) into the .zst —
                                  entry (ty, tx, p) is the 16 bytes at
                                  index_header_bytes + 16 * ((ty * ntx + tx) *
                                  3 + p).

  (offset, length):
      length > 0     a stored part (one zstd frame, decompresses alone)
      length == 0    the tile holds NO observation this month (count 0 in
                     every pixel and channel): nothing is stored for any of
                     its three parts, and the reader returns sum 0, count 0,
                     m2 0. offset is then where the tile would have started.
  A month with no frame at all has no file; the store's month table
  (stats.json `frames_present`) says 0 for it.

A PART is T x T x C (T = the source tile) in C order [row, col, channel],
little-endian: sum float32, count uint8, m2 float32. Edge tiles are padded to
T x T with ZEROS (count 0 — a pad pixel is "no observation", never data).
sum and m2 are 0.0 wherever count is 0; m2 is 0.0 where count is 1.

Nothing here knows about a particular store; nothing touches the network
except `sharded._Source`'s HTTP branch (which refuses a 200 where a 206 was
asked for).
"""
import io
import json
import os

import numpy as np

from family1 import sharded as sh

FORMAT = "family1-monthly-tiles/1"
PARTS = (("sum", np.dtype("<f4")), ("count", np.dtype("u1")),
         ("m2", np.dtype("<f4")))
PART_NAMES = tuple(p for p, _ in PARTS)
NP = len(PARTS)
INDEX_DTYPE = sh.INDEX_DTYPE
DEFAULT_LEVEL = sh.DEFAULT_LEVEL


class MonthlyTileError(ValueError):
    """A month file, an index or a part that does not match its spec."""


def shard_relpath(y, m):
    """m is 1-based (January = 1)."""
    return f"{int(y):04d}/m_{int(y):04d}-{int(m):02d}.zst"


def index_relpath(y, m):
    return f"{int(y):04d}/m_{int(y):04d}-{int(m):02d}.idx.npy"


def physical(spec_src):
    """(value_offset, [channels in physical units]) for a source tile_grid.

    float16 groups are already physical. A uint8 group whose unit reads
    "K - <n> …" stores kelvin minus n and is summed in KELVIN; any other uint8
    group is refused rather than summed in storage units by accident."""
    chans = []
    off = 0.0
    if spec_src["dtype"] == "uint8":
        offs = set()
        for c in spec_src["channels"]:
            u = str(c["unit"])
            if not u.startswith("K - "):
                raise MonthlyTileError(
                    f"{spec_src['group']}: uint8 channel {c['name']} has unit "
                    f"{u!r} — no physical conversion is known for it")
            offs.add(float(u[4:].split()[0]))
        if len(offs) != 1:
            raise MonthlyTileError(f"{spec_src['group']}: mixed offsets {offs}")
        off = offs.pop()
        for c in spec_src["channels"]:
            chans.append(dict(name=c["name"], unit="K",
                              min=float(c["min"]) + off,
                              max=float(c["max"]) + off,
                              stored_unit=c["unit"]))
    elif spec_src["dtype"] == "float16":
        for c in spec_src["channels"]:
            chans.append(dict(c))
    else:
        raise MonthlyTileError(f"{spec_src['group']}: dtype {spec_src['dtype']}")
    return off, chans


def make_spec(spec_src, level=DEFAULT_LEVEL):
    """tile_grid.json for the monthly files of the group `spec_src` describes
    (the source group's own tile_grid.json)."""
    if spec_src.get("format") != sh.FORMAT:
        raise MonthlyTileError(f"source format {spec_src.get('format')!r}")
    T, C = int(spec_src["tile"]), int(spec_src["C"])
    nty, ntx = int(spec_src["n_tiles_y"]), int(spec_src["n_tiles_x"])
    off, chans = physical(spec_src)
    ishape = [nty, ntx, NP, 2]
    return {
        "format": FORMAT, "source_format": spec_src["format"],
        "group": spec_src["group"],
        "H": int(spec_src["H"]), "W": int(spec_src["W"]), "C": C, "tile": T,
        "n_tiles_y": nty, "n_tiles_x": ntx,
        "pad_rows": spec_src["pad_rows"], "pad_cols": spec_src["pad_cols"],
        "row_extents": spec_src["row_extents"],
        "col_extents": spec_src["col_extents"],
        "grid": spec_src["grid"],
        "channels": chans, "value_offset": off,
        "source_dtype": spec_src["dtype"],
        "source_channels": spec_src["channels"],
        "parts": [{"name": n, "dtype": d.str, "itemsize": d.itemsize,
                   "part_bytes": T * T * C * d.itemsize} for n, d in PARTS],
        "tile_shape": [T, T, C],
        "tile_order": "C order [row, col, channel], little-endian",
        "pad": "edge tiles padded to tile x tile with zeros (count 0)",
        "shard_path": "<yyyy>/m_<yyyy>-<mm>.zst (mm = 01 .. 12)",
        "index_path": "<yyyy>/m_<yyyy>-<mm>.idx.npy",
        "write_order": "tiles row-major (ty, then tx); within a tile the "
                       "parts sum, count, m2 — one tile's parts are adjacent",
        "index_shape": ishape, "index_dtype": INDEX_DTYPE.str,
        "index_header_bytes": sh.index_header_bytes(ishape),
        "index_semantics": ("(offset, length) per (ty, tx, part): length > 0 "
                            "a stored part; length 0 for all three parts — "
                            "the tile has no observation this month (sum 0, "
                            "count 0, m2 0), nothing stored"),
        "zeros": "sum and m2 are 0.0 where count is 0; m2 is 0.0 where "
                 "count is 1",
        "compression": {"codec": "zstd", "level": int(level),
                        "unit": "one zstd frame per (tile, part)"},
    }


def _pad(a, T, ty, tx, spec):
    r0, r1 = spec["row_extents"][ty]
    c0, c1 = spec["col_extents"][tx]
    blk = a[r0:r1, c0:c1]
    if blk.shape[:2] == (T, T):
        return np.ascontiguousarray(blk)
    t = np.zeros((T, T) + a.shape[2:], a.dtype)
    t[:r1 - r0, :c1 - c0] = blk
    return t


def write_month(spec, S, N, M2, shard_path, index_path, cctx=None):
    """Write one month: S float32, N uint8, M2 float32, each [H, W, C].

    Temporary siblings renamed into place, index LAST, so a present index
    always describes a complete file. Returns {tiles_stored, nbytes}."""
    H, W, C, T = spec["H"], spec["W"], spec["C"], spec["tile"]
    nty, ntx = spec["n_tiles_y"], spec["n_tiles_x"]
    for name, a, (pn, dt_) in zip(("sum", "count", "m2"), (S, N, M2), PARTS):
        if a.shape != (H, W, C) or a.dtype != dt_:
            raise MonthlyTileError(f"{name}: {a.shape} {a.dtype}, the spec is "
                                   f"{(H, W, C)} {dt_}")
    if cctx is None:
        cctx = sh.zstd().ZstdCompressor(level=int(spec["compression"]["level"]))
    idx = np.zeros((nty, ntx, NP, 2), INDEX_DTYPE)
    os.makedirs(os.path.dirname(shard_path) or ".", exist_ok=True)
    tmp = f"{shard_path}.tmp{os.getpid()}"
    off = tiles = 0
    with open(tmp, "wb") as fh:
        for ty in range(nty):
            for tx in range(ntx):
                tn = _pad(N, T, ty, tx, spec)
                idx[ty, tx, :, 0] = off
                if not tn.any():
                    idx[ty, tx, :, 1] = 0
                    continue
                for p, a in enumerate((S, tn, M2)):
                    t = tn if p == 1 else _pad(a, T, ty, tx, spec)
                    blob = cctx.compress(t.tobytes())
                    fh.write(blob)
                    idx[ty, tx, p, 0] = off
                    idx[ty, tx, p, 1] = len(blob)
                    off += len(blob)
                tiles += 1
    os.replace(tmp, shard_path)
    itmp = f"{index_path}.tmp{os.getpid()}.npy"
    np.save(itmp, idx)
    got = os.path.getsize(itmp) - idx.nbytes
    if got != spec["index_header_bytes"]:
        os.remove(itmp)
        raise MonthlyTileError(f"index header {got} bytes, the spec declares "
                               f"{spec['index_header_bytes']}")
    os.replace(itmp, index_path)
    return {"tiles_stored": tiles, "nbytes": off}


def check_structure(idx, size, spec, where=""):
    """An index against its file: shape, write order, total length, and the
    all-or-nothing rule per tile. Returns the number of stored tiles."""
    nty, ntx = spec["n_tiles_y"], spec["n_tiles_x"]
    if idx.shape != (nty, ntx, NP, 2) or idx.dtype != INDEX_DTYPE:
        raise MonthlyTileError(f"{where}: index {idx.shape} {idx.dtype}")
    o = idx[..., 0].reshape(-1)
    n = idx[..., 1].reshape(-1)
    if (n < 0).any():
        raise MonthlyTileError(f"{where}: a negative length")
    exp = np.concatenate((np.zeros(1, np.int64), np.cumsum(n[:-1])))
    if not np.array_equal(o, exp):
        k = int(np.argmax(o != exp))
        raise MonthlyTileError(f"{where}: entry {k} offset {int(o[k])}, "
                               f"expected {int(exp[k])}")
    if int(n.sum()) != int(size):
        raise MonthlyTileError(f"{where}: lengths sum to {int(n.sum())}, the "
                               f"file is {size} bytes")
    per = (idx[..., 1] > 0)
    some, all_ = per.any(axis=2), per.all(axis=2)
    if (some != all_).any():
        raise MonthlyTileError(f"{where}: a tile with some parts stored and "
                               f"some not")
    return int(all_.sum())


class MonthlyTiles:
    """The monthly files of one group, on a local directory or an HTTP
    prefix (sharded._Source: ranged reads, a 200 for a range is refused)."""

    def __init__(self, base, headers=None, attempts=4):
        self.src = sh._Source(base, headers, attempts)
        self.spec = json.loads(self.src.get("tile_grid.json"))
        if self.spec.get("format") != FORMAT:
            raise MonthlyTileError(f"{base}: format "
                                   f"{self.spec.get('format')!r}, not {FORMAT}")
        self._dctx = sh.zstd().ZstdDecompressor()
        self._idx = {}

    def index(self, y, m):
        """The whole [nty, ntx, 3, 2] index of (y, m) — one read."""
        k = (int(y), int(m))
        if k not in self._idx:
            raw = self.src.get(index_relpath(y, m))
            a = np.load(io.BytesIO(raw), allow_pickle=False)
            if len(raw) - a.nbytes != self.spec["index_header_bytes"]:
                raise MonthlyTileError(f"{index_relpath(y, m)}: header length")
            self._idx[k] = a
        return self._idx[k]

    def _decode(self, blob, p):
        sp = self.spec
        T, C = sp["tile"], sp["C"]
        dt_ = PARTS[p][1]
        want = T * T * C * dt_.itemsize
        raw = self._dctx.decompress(blob, max_output_size=want)
        if len(raw) != want:
            raise MonthlyTileError(f"a {PART_NAMES[p]} part decompressed to "
                                   f"{len(raw)} bytes, declared {want}")
        return np.frombuffer(raw, dt_).reshape(T, T, C)

    def read_tile(self, y, m, ty, tx, parts=PART_NAMES, crop=True):
        """{part: array} of one tile of (y, m): ONE range read covering the
        requested parts (they are adjacent). A tile with no observation
        returns zeros without reading the file."""
        sp = self.spec
        T, C = sp["tile"], sp["C"]
        ps = sorted(PART_NAMES.index(p) for p in parts)
        e = self.index(y, m)[ty, tx]
        out = {}
        if int(e[0, 1]) == 0:
            for p in ps:
                out[PART_NAMES[p]] = np.zeros((T, T, C), PARTS[p][1])
        else:
            a, b = int(e[ps[0], 0]), int(e[ps[-1], 0] + e[ps[-1], 1])
            buf = self.src.read_at(shard_relpath(y, m), a, b - a)
            for p in ps:
                o, n = int(e[p, 0]) - a, int(e[p, 1])
                out[PART_NAMES[p]] = self._decode(buf[o:o + n], p)
        if crop:
            r0, r1 = sp["row_extents"][ty]
            c0, c1 = sp["col_extents"][tx]
            out = {k: v[:r1 - r0, :c1 - c0] for k, v in out.items()}
        return out

    def read_month(self, y, m, parts=PART_NAMES):
        """{part: [H, W, C]} of the whole month; the file read once."""
        sp = self.spec
        H, W, C = sp["H"], sp["W"], sp["C"]
        idx = self.index(y, m)
        blob = self.src.get(shard_relpath(y, m))
        check_structure(idx, len(blob), sp, shard_relpath(y, m))
        ps = [PART_NAMES.index(p) for p in parts]
        out = {PART_NAMES[p]: np.zeros((H, W, C), PARTS[p][1]) for p in ps}
        for ty in range(sp["n_tiles_y"]):
            r0, r1 = sp["row_extents"][ty]
            for tx in range(sp["n_tiles_x"]):
                c0, c1 = sp["col_extents"][tx]
                for p in ps:
                    o, n = int(idx[ty, tx, p, 0]), int(idx[ty, tx, p, 1])
                    if n:
                        t = self._decode(blob[o:o + n], p)
                        out[PART_NAMES[p]][r0:r1, c0:c1] = t[:r1 - r0,
                                                             :c1 - c0]
        return out

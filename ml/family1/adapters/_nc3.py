"""A minimal NetCDF-3 ("classic" and "64-bit offset") reader and writer.

WHY THIS EXISTS. Two family-1 sources serve NetCDF-3: the UHSLC ERDDAP's
`.nc` answer for GESLA (`tide`, `tide_private`) and the IOOS Glider DAC's
per-deployment `<id>.nc3.nc` aggregates on THREDDS (`gliders`). The hosted
fetch lane installs only numpy and huggingface_hub, and a glider aggregate can
be 1.7 GB of which the store needs a few variables — so this reader takes a
`read_at(offset, n)` callable and pulls ONLY the header and the variables it
is asked for, which lets the glider adapter use HTTP Range requests instead of
downloading the file.

The format is the Unidata "NetCDF Classic Format Specification" (CDF-1 and
CDF-2; CDF-5 is refused). VERIFIED 2026-09-17 against netCDF4 1.7.4 on two
real files from this sandbox: a GESLA ERDDAP answer (`row` dimension, four
variables) and a THREDDS glider aggregate (`angus-20200319T0000.nc3.nc`,
104,213,412 bytes, dimensions trajectory x profile x obs) — every variable the
adapters read came back bit-identical.

The writer exists for the smoke tests: it writes CDF-1 with fixed dimensions
only, which is the layout both sources use.
"""
import struct

import numpy as np

NC_DIMENSION = 0x0A
NC_VARIABLE = 0x0B
NC_ATTRIBUTE = 0x0C
TYPES = {1: np.dtype("i1"), 2: np.dtype("S1"), 3: np.dtype(">i2"),
         4: np.dtype(">i4"), 5: np.dtype(">f4"), 6: np.dtype(">f8")}
CODE_OF = {"i1": 1, "S1": 2, "i2": 3, "i4": 4, "f4": 5, "f8": 6}


class NC3Error(ValueError):
    """Bytes that are not the NetCDF-3 layout this reader was written for."""


def _pad4(n):
    return (4 - n % 4) % 4


class _Cursor:
    """Sequential reads over `read_at`, fetching the header in growing blocks."""

    def __init__(self, read_at, block=1 << 16):
        self.read_at = read_at
        self.buf = b""
        self.pos = 0
        self.block = block
        self.eof = False

    def need(self, n):
        while len(self.buf) < self.pos + n:
            if self.eof:
                raise NC3Error(f"the header ends before byte "
                               f"{self.pos + n:,} (read {len(self.buf):,})")
            more = self.read_at(len(self.buf), max(self.block, n))
            if not more:
                self.eof = True
                continue
            if len(more) < max(self.block, n):
                self.eof = True
            self.buf += more
            self.block *= 2

    def take(self, n):
        self.need(n)
        b = self.buf[self.pos:self.pos + n]
        self.pos += n
        return b

    def u32(self):
        return struct.unpack(">I", self.take(4))[0]

    def u64(self):
        return struct.unpack(">Q", self.take(8))[0]

    def name(self):
        n = self.u32()
        s = self.take(n)
        self.take(_pad4(n))
        return s.decode("utf-8", "replace")


class NC3:
    """The header of a NetCDF-3 file, and on-demand reads of its variables."""

    def __init__(self, read_at):
        self.read_at = read_at
        c = _Cursor(read_at)
        magic = c.take(4)
        if magic[:3] != b"CDF" or magic[3] not in (1, 2):
            raise NC3Error(f"not a NetCDF-3 classic/64-bit-offset file "
                           f"(magic {magic!r})")
        self.version = magic[3]
        self.numrecs = c.u32()
        self.dims = []                        # [(name, length)], 0 = record
        tag, n = c.u32(), c.u32()
        if tag not in (0, NC_DIMENSION):
            raise NC3Error(f"dim_list tag {tag}")
        for _ in range(n):
            self.dims.append((c.name(), c.u32()))
        self.attrs = self._attrs(c)
        self.vars = {}
        tag, n = c.u32(), c.u32()
        if tag not in (0, NC_VARIABLE):
            raise NC3Error(f"var_list tag {tag}")
        for _ in range(n):
            name = c.name()
            nd = c.u32()
            dimids = [c.u32() for _ in range(nd)]
            vatts = self._attrs(c)
            typ = c.u32()
            if typ not in TYPES:
                raise NC3Error(f"variable {name}: type {typ} is not a "
                               f"NetCDF-3 classic type")
            vsize = c.u32()
            begin = c.u64() if self.version == 2 else c.u32()
            self.vars[name] = {"dimids": dimids, "attrs": vatts,
                               "dtype": TYPES[typ], "vsize": vsize,
                               "begin": begin}
        self.header_bytes = c.pos
        rec = [v for v in self.vars.values()
               if v["dimids"] and self.dims[v["dimids"][0]][1] == 0]
        # the record size: the sum of the record variables' vsize, except that
        # a file with ONE record variable stores it unpadded
        if len(rec) == 1:
            v = rec[0]
            self.recsize = v["dtype"].itemsize * int(np.prod(
                [self.dims[d][1] for d in v["dimids"][1:]], dtype=np.int64))
        else:
            self.recsize = sum(v["vsize"] for v in rec)

    @staticmethod
    def _attrs(c):
        out = {}
        tag, n = c.u32(), c.u32()
        if tag not in (0, NC_ATTRIBUTE):
            raise NC3Error(f"att_list tag {tag}")
        for _ in range(n):
            name = c.name()
            typ, nel = c.u32(), c.u32()
            dt = TYPES.get(typ)
            if dt is None:
                raise NC3Error(f"attribute {name}: type {typ}")
            raw = c.take(nel * dt.itemsize)
            c.take(_pad4(nel * dt.itemsize))
            if typ == 2:
                out[name] = raw.decode("utf-8", "replace").rstrip("\x00")
            else:
                a = np.frombuffer(raw, dt).astype(dt.newbyteorder("="))
                out[name] = a[0].item() if a.size == 1 else a
        return out

    def dim_size(self, name):
        for n, k in self.dims:
            if n == name:
                return self.numrecs if k == 0 else k
        raise KeyError(name)

    def shape(self, name):
        v = self.vars[name]
        return tuple(self.numrecs if self.dims[d][1] == 0 else self.dims[d][1]
                     for d in v["dimids"])

    def dimnames(self, name):
        return tuple(self.dims[d][0] for d in self.vars[name]["dimids"])

    def nbytes(self, name):
        v = self.vars[name]
        return int(np.prod(self.shape(name), dtype=np.int64)) \
            * v["dtype"].itemsize

    def read(self, name):
        """The whole variable, native byte order (char -> S1 array)."""
        v = self.vars[name]
        shp = self.shape(name)
        dt = v["dtype"]
        is_rec = v["dimids"] and self.dims[v["dimids"][0]][1] == 0
        if not is_rec:
            n = int(np.prod(shp, dtype=np.int64)) * dt.itemsize
            raw = self.read_at(v["begin"], n) if n else b""
            if len(raw) != n:
                raise NC3Error(f"{name}: {len(raw):,} of {n:,} bytes — the "
                               f"file is shorter than its header says")
            a = np.frombuffer(raw, dt).reshape(shp)
        else:
            per = int(np.prod(shp[1:], dtype=np.int64)) * dt.itemsize
            parts = []
            for r in range(self.numrecs):
                off = v["begin"] + r * self.recsize
                raw = self.read_at(off, per)
                if len(raw) != per:
                    raise NC3Error(f"{name}: record {r} is short")
                parts.append(raw)
            a = np.frombuffer(b"".join(parts), dt).reshape(shp)
        if dt.kind == "S":
            return a
        return a.astype(dt.newbyteorder("="))

    def read_part(self, name, axis, start, stop):
        """Indices [start, stop) of dimension `axis` of a FIXED variable whose
        dimensions before `axis` all have size 1 — the glider aggregate's
        (trajectory=1, profile, obs) — without reading the rest."""
        v = self.vars[name]
        shp = self.shape(name)
        if v["dimids"] and self.dims[v["dimids"][0]][1] == 0:
            raise NC3Error(f"{name} is a record variable")
        if any(k != 1 for k in shp[:axis]):
            raise NC3Error(f"{name}{shp}: the dimensions before axis {axis} "
                           f"are not all 1")
        start, stop = max(0, start), min(shp[axis], stop)
        dt = v["dtype"]
        per = int(np.prod(shp[axis + 1:], dtype=np.int64)) * dt.itemsize
        n = (stop - start) * per
        raw = self.read_at(v["begin"] + start * per, n) if n > 0 else b""
        if len(raw) != n:
            raise NC3Error(f"{name}: {len(raw):,} of {n:,} bytes")
        a = np.frombuffer(raw, dt).reshape(
            (stop - start,) + tuple(shp[axis + 1:]))
        return a if dt.kind == "S" else a.astype(dt.newbyteorder("="))

    def masked(self, name, dtype=np.float64):
        """The variable as `dtype` with _FillValue / missing_value -> NaN."""
        a = self.read(name).astype(dtype)
        at = self.vars[name]["attrs"]
        for k in ("_FillValue", "missing_value"):
            f = at.get(k)
            if f is None:
                continue
            for x in np.atleast_1d(f):
                if np.isnan(x):
                    continue
                a[a == x] = np.nan
        return a


def file_reader(path, counter=None):
    """`read_at` over a local file; `counter(n)` is told every byte read."""
    def read_at(off, n):
        with open(path, "rb") as fh:
            fh.seek(off)
            b = fh.read(n)
        if counter is not None:
            counter(len(b))
        return b
    return read_at


def bytes_reader(data):
    def read_at(off, n):
        return data[off:off + n]
    return read_at


def chars(a):
    """An S1 array (..., strlen) -> str array (...)."""
    a = np.asarray(a)
    if a.ndim == 0:
        return a.tobytes().decode("utf-8", "replace").rstrip("\x00 ")
    flat = a.reshape(-1, a.shape[-1])
    out = [b"".join(r).split(b"\x00", 1)[0].decode("utf-8", "replace").strip()
           for r in flat.tolist()]
    return np.array(out, dtype=object).reshape(a.shape[:-1])


# ================================================================== writer ==
def _enc_name(s):
    b = s.encode("utf-8")
    return struct.pack(">I", len(b)) + b + b"\x00" * _pad4(len(b))


def _enc_attrs(attrs):
    if not attrs:
        return struct.pack(">II", 0, 0)
    out = struct.pack(">II", NC_ATTRIBUTE, len(attrs))
    for k, v in attrs.items():
        out += _enc_name(k)
        if isinstance(v, str):
            b = v.encode("utf-8")
            out += struct.pack(">II", 2, len(b)) + b + b"\x00" * _pad4(len(b))
        else:
            a = np.atleast_1d(np.asarray(v))
            code = CODE_OF[a.dtype.str[1:]]
            raw = a.astype(TYPES[code]).tobytes()
            out += struct.pack(">II", code, a.size) + raw \
                + b"\x00" * _pad4(len(raw))
    return out


def write(path, dims, variables, gattrs=None):
    """CDF-1, fixed dimensions only. `dims` = [(name, size)], `variables` =
    [(name, (dimname, ...), array, attrs)] with array dtype i1/S1/i2/i4/f4/f8."""
    dix = {n: i for i, (n, _) in enumerate(dims)}
    head = b"CDF\x01" + struct.pack(">I", 0)
    head += struct.pack(">II", NC_DIMENSION, len(dims))
    for n, k in dims:
        if k <= 0:
            raise ValueError("the smoke writer writes fixed dimensions only")
        head += _enc_name(n) + struct.pack(">I", k)
    head += _enc_attrs(gattrs or {})
    blobs = []
    for name, dn, arr, attrs in variables:
        a = np.asarray(arr)
        code = CODE_OF[a.dtype.str[1:] if a.dtype.kind != "S" else "S1"]
        raw = a.astype(TYPES[code]).tobytes()
        blobs.append((name, dn, attrs, code, raw))
    # the header length does not depend on the begin values (fixed width)
    def var_list(begins):
        out = struct.pack(">II", NC_VARIABLE, len(blobs))
        for (name, dn, attrs, code, raw), b in zip(blobs, begins):
            out += _enc_name(name) + struct.pack(">I", len(dn))
            out += b"".join(struct.pack(">I", dix[d]) for d in dn)
            out += _enc_attrs(attrs)
            vs = len(raw) + _pad4(len(raw))
            out += struct.pack(">III", code, vs, b)
        return out
    hlen = len(head) + len(var_list([0] * len(blobs)))
    begins, off = [], hlen
    for *_, raw in blobs:
        begins.append(off)
        off += len(raw) + _pad4(len(raw))
    with open(path, "wb") as fh:
        fh.write(head + var_list(begins))
        for *_, raw in blobs:
            fh.write(raw + b"\x00" * _pad4(len(raw)))
    return path

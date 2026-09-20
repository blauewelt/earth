"""Reading a FEW datasets out of a BIG HDF5 file over HTTP byte ranges.

Underscore-prefixed, so `ml/family1/adapters/__init__.py` skips it: this is a
helper, not a store.

PLAIN ENGLISH. HDF5 (Hierarchical Data Format version 5) is the container
NASA's laser-altimeter products come in. One GEDI (Global Ecosystem Dynamics
Investigation) L2A granule is about a gigabyte and holds roughly eight hundred
datasets; the eleven this project reads are a small part of it. An HDF5 file
is not a stream — it is a little file system with an index — so a reader that
can seek does not have to download the parts it never touches. `h5py` accepts
any Python file-like object, and this module supplies one whose `read` is an
HTTP `Range:` request with a block cache in front of it. The result is that
the bytes fetched are bounded by the QUESTION (which datasets, which rows)
rather than by the archive, exactly as the globe app's family-7 layer is
bounded by the frame rather than by the 46 GB `.npy` behind it.

WHY NOT OPeNDAP. Measured on NASA's Common Metadata Repository 2026-09-20,
anonymously: the only subsetting service ASSOCIATED with GEDI02_A version 003,
GEDI02_B version 003 and GEDI L4A version 3 is the Harmony Trajectory
Subsetter (service S2836723123). Neither collection is associated with an
OPeNDAP service, and a plain GET of
`https://opendap.earthdata.nasa.gov/collections/<collection>/granules/<name>.dmr`
answers HTTP 401 after two redirects to Earthdata Login for every one of them,
so nothing about it can be confirmed without credentials this sandbox does not
hold. (GEDI L4A version 2.1 IS associated with OPeNDAP and with the Harmony
OPeNDAP SubSetter; version 3 is not.) ICESat-2 ATL08 release 007 is likewise
associated with Harmony trajectory subsetters and not with OPeNDAP. So the
variable-subsetting route that needs no extra service, no asynchronous job and
no queue is this one: read the file's own index and then the datasets wanted.

WHAT THE ARCHIVE MUST DO FOR THIS TO WORK, and what happens if it does not.
The host must honour `Range:` with HTTP 206. `RangeFile` REFUSES a 200 — that
is the whole file arriving, and consuming it would silently turn a subset read
into a gigabyte download — and a hosted probe is what confirms that LP DAAC
(NASA's land-processes data centre), ORNL DAAC (its terrestrial-ecology one)
and NSIDC (its snow-and-ice one) answer 206 for these granules. Earthdata
Login redirects a protected URL to a signed, short-lived object-store URL;
that redirect is followed ONCE and the resolved URL is reused for every
subsequent range, so one authentication buys the whole file.

FALLBACK. `whole_file` downloads the granule with the ordinary Earthdata
download helper and opens it from disk, which is what a store whose probe says
subsetting is not worth it would use. Both routes hand back a `h5py.File`, so
the reading code above them is identical.
"""
import io
import os
import time

import build_family10_stores as f10b
from family1.adapters import _common as cm

BLOCK = 1 << 20                 # 1 MiB: the unit a cache miss fetches
CACHE_BLOCKS = 96               # ~96 MiB held per open granule, at most
RESOLVE_ATTEMPTS = 4


class RangeError(IOError):
    """The host did not serve a byte range the way a subset read needs."""


def resolve(session, url, attempts=RESOLVE_ATTEMPTS, timeout=120):
    """Follow Earthdata Login once -> (final url, size in bytes).

    A one-byte range request is the cheapest thing that answers both
    questions: `Content-Range: bytes 0-0/<size>` gives the length, and the
    response's own url is the signed object-store url the rest of the reads go
    to. A host that answers 200 to this has ignored the range and is sending
    the whole granule: that is a REFUSAL, not a slow success.
    """
    err = None
    for i in range(max(1, attempts)):
        try:
            with session.get(url, headers={"Range": "bytes=0-0"}, stream=True,
                             timeout=timeout, allow_redirects=True) as r:
                if r.status_code in (401, 403):
                    raise RangeError(
                        f"{url}: HTTP {r.status_code} after {len(r.history)} "
                        f"redirect(s) — Earthdata Login refused this account "
                        f"for this archive. Dispatch family1-build.yml with "
                        f"check_credentials=true and read the approval URL it "
                        f"prints.")
                if r.status_code == 404:
                    raise f10b._NotFound(url)
                if r.status_code != 206:
                    raise RangeError(
                        f"{url}: asked for one byte and got HTTP "
                        f"{r.status_code} — this host is not serving byte "
                        f"ranges, so a variable subset would silently become "
                        f"a whole-granule download. Refusing.")
                cr = str(r.headers.get("Content-Range", ""))
                body = r.content
                final = r.url
            f10b.count_bytes(len(body))
            if "/" not in cr:
                raise RangeError(f"{url}: HTTP 206 with no Content-Range "
                                 f"({cr!r}) — the size is unknowable")
            size = int(cr.rsplit("/", 1)[1])
            return final, size
        except (RangeError, f10b._NotFound):
            raise
        except Exception as e:                                  # noqa: BLE001
            err = e
            if i < attempts - 1:
                time.sleep(3.0 * (2 ** i))
    raise IOError(f"{url}: {type(err).__name__}: {err}")


class RangeFile(io.RawIOBase):
    """A seekable, read-only file over HTTP `Range:` requests.

    Reads are served from a block cache, so `h5py`'s many small index reads
    cost a handful of requests rather than one each. `blocks_fetched` and
    `bytes_read` are what the probe reports: bytes_read against the granule's
    own size IS the subsetting ratio, measured rather than argued.
    """

    def __init__(self, session, url, block=BLOCK, cache_blocks=CACHE_BLOCKS,
                 attempts=RESOLVE_ATTEMPTS):
        self._s = session
        self._url0 = url
        self._url, self.size = resolve(session, url, attempts=attempts)
        self._block = int(block)
        self._max = int(cache_blocks)
        self._cache = {}
        self._order = []
        self._pos = 0
        self._attempts = attempts
        self.blocks_fetched = 0
        self.bytes_fetched = 1         # the resolve's one byte
        self.bytes_read = 0
        self.requests = 1              # the resolve is one

    # -- the file-like surface h5py uses ------------------------------------
    def readable(self):
        return True

    def seekable(self):
        return True

    def writable(self):
        return False

    def tell(self):
        return self._pos

    def seek(self, off, whence=io.SEEK_SET):
        if whence == io.SEEK_SET:
            self._pos = int(off)
        elif whence == io.SEEK_CUR:
            self._pos += int(off)
        else:
            self._pos = self.size + int(off)
        self._pos = max(0, min(self._pos, self.size))
        return self._pos

    def read(self, n=-1):
        if n is None or n < 0:
            n = self.size - self._pos
        n = max(0, min(int(n), self.size - self._pos))
        if n == 0:
            return b""
        out = bytearray()
        pos = self._pos
        while len(out) < n:
            b = pos // self._block
            blk = self._block_bytes(b)
            off = pos - b * self._block
            take = min(len(blk) - off, n - len(out))
            if take <= 0:
                break
            out += blk[off:off + take]
            pos += take
        self._pos = pos
        self.bytes_read += len(out)
        return bytes(out)

    def readinto(self, buf):
        b = self.read(len(buf))
        buf[:len(b)] = b
        return len(b)

    def close(self):
        self._cache.clear()
        self._order.clear()
        super().close()

    # -- the cache ----------------------------------------------------------
    def _block_bytes(self, b):
        hit = self._cache.get(b)
        if hit is not None:
            return hit
        lo = b * self._block
        hi = min(lo + self._block, self.size) - 1
        body = self._get(lo, hi)
        self._cache[b] = body
        self._order.append(b)
        while len(self._order) > self._max:
            self._cache.pop(self._order.pop(0), None)
        self.blocks_fetched += 1
        return body

    def _get(self, lo, hi):
        err = None
        want = hi - lo + 1
        for i in range(max(1, self._attempts)):
            try:
                with self._s.get(self._url,
                                 headers={"Range": f"bytes={lo}-{hi}"},
                                 stream=True, timeout=300,
                                 allow_redirects=True) as r:
                    self.requests += 1
                    if r.status_code == 403 and self._url != self._url0:
                        # a signed url can expire mid-granule; re-authenticate
                        self._url, _sz = resolve(self._s, self._url0,
                                                 attempts=self._attempts)
                        raise IOError("signed url expired; re-resolved")
                    if r.status_code != 206:
                        raise RangeError(
                            f"{self._url0}: range {lo}-{hi} answered HTTP "
                            f"{r.status_code}, not 206")
                    body = r.content
                if len(body) != want:
                    raise IOError(f"{self._url0}: range {lo}-{hi} returned "
                                  f"{len(body)} of {want} bytes")
                f10b.count_bytes(len(body))
                self.bytes_fetched += len(body)
                return body
            except RangeError:
                raise
            except Exception as e:                              # noqa: BLE001
                err = e
                if i < self._attempts - 1:
                    time.sleep(2.0 * (2 ** i))
        raise IOError(f"{self._url0} bytes {lo}-{hi}: "
                      f"{type(err).__name__}: {err}")


class Opened:
    """An open granule plus what reading it cost. Use as a context manager."""

    def __init__(self, h5, mode, bytes_read, size, requests, path=None):
        self.h5 = h5
        self.mode = mode                # "range" | "whole" | "local"
        self.bytes_read = int(bytes_read)
        self.size = int(size)
        self.requests = int(requests)
        self.path = path

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
        return False

    def close(self):
        try:
            self.h5.close()
        finally:
            if self.path and os.path.exists(self.path):
                os.remove(self.path)

    def stats(self):
        return {"mode": self.mode, "bytes_read": self.bytes_read,
                "granule_bytes": self.size, "http_requests": self.requests,
                "read_fraction": (round(self.bytes_read / self.size, 6)
                                  if self.size else None)}


def open_local(path):
    """A granule already on disk — the smoke's synthetic archive."""
    import h5py
    n = os.path.getsize(path)
    return Opened(h5py.File(path, "r"), "local", n, n, 0)


def open_range(session, url, block=BLOCK, cache_blocks=CACHE_BLOCKS,
               attempts=RESOLVE_ATTEMPTS):
    """Open a remote granule and read only what is asked for."""
    import h5py
    rf = RangeFile(session, url, block=block, cache_blocks=cache_blocks,
                   attempts=attempts)
    try:
        h5 = h5py.File(rf, "r")
    except Exception:
        rf.close()
        raise
    op = Opened(h5, "range", 0, rf.size, 0)
    op._rf = rf
    return op


def finish_range(op):
    """Copy the counters off the range file once the granule has been read.

    `bytes_read` is what the SOCKET carried — every block fetched, the last
    one at its real length rather than a whole block — because that is the
    number the storage decision needs. The bytes h5py asked for are fewer and
    would flatter the subset.
    """
    rf = getattr(op, "_rf", None)
    if rf is not None:
        op.bytes_read = rf.bytes_fetched
        op.requests = rf.requests
    return op


def open_whole(session, url, dest, attempts=4):
    """Download the whole granule and open it — the fallback route."""
    import h5py
    n, why = cm.earthdata_download(session, url, dest, attempts=attempts)
    if n is None:
        raise f10b._NotFound(f"{url}: {why}")
    return Opened(h5py.File(dest, "r"), "whole", n, n, 1, path=dest)

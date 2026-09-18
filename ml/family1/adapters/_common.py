"""Helpers shared by the family-1.0.tf point adapters (ndbc, igra, tide,
tide_private, gliders). Underscore-prefixed, so the registry skips it.

Everything here is small on purpose: the civil-calendar arithmetic that lets
a whole year of timestamps become seconds without a Python loop, a fixed-width
integer parser, an HTTP GET that retries the transient failures the sandbox
and the hosted runners both see (connection resets, 5xx) and refuses a short
body, and an ordered thread map with a bounded look-ahead so a pool of
downloads never holds more than a few files at once.
"""
import collections
import os
import re
import concurrent.futures as cf
import time
import urllib.error
import urllib.request

import numpy as np

import build_family10_stores as f10b

EPOCH_DAYS_1970 = 4383          # 1970-01-01 -> 1982-01-01, days
EPOCH_S_1970 = EPOCH_DAYS_1970 * 86400
MLEN = np.array([0, 31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31], np.int64)


def days_from_civil(y, m, d):
    """Proleptic-Gregorian (y, m, d) -> days since 1982-01-01, vectorised.

    Howard Hinnant's `days_from_civil`, shifted to the family-10 epoch. The
    caller validates the date first (`valid_date`); this function assumes it.
    """
    y = np.asarray(y, np.int64) - (np.asarray(m) <= 2)
    m = np.asarray(m, np.int64)
    d = np.asarray(d, np.int64)
    era = np.floor_divide(y, 400)
    yoe = y - era * 400
    mp = (m + 9) % 12
    doy = (153 * mp + 2) // 5 + d - 1
    doe = yoe * 365 + yoe // 4 - yoe // 100 + doy
    return era * 146097 + doe - 719468 - EPOCH_DAYS_1970


def is_leap(y):
    y = np.asarray(y, np.int64)
    return ((y % 4 == 0) & (y % 100 != 0)) | (y % 400 == 0)


def valid_date(y, m, d):
    y = np.asarray(y, np.int64)
    m = np.asarray(m, np.int64)
    d = np.asarray(d, np.int64)
    mc = np.clip(m, 0, 12)
    ml = MLEN[mc] + ((mc == 2) & is_leap(y))
    return (m >= 1) & (m <= 12) & (d >= 1) & (d <= ml)


def fixed_int(a, starts, lo, hi, what="field"):
    """Right-aligned integer fields at [starts+lo, starts+hi) -> int64.

    `a` is the uint8 view of the text. Blanks are spaces; one leading minus is
    allowed. A field that is all blank parses as 0 and is reported in the
    second return value, so a caller can tell 0 from nothing.
    """
    w = hi - lo
    ch = a[starts[:, None] + lo + np.arange(w)].astype(np.int64)
    blank = ch == 32
    neg = ch == 45
    dig = (ch >= 48) & (ch <= 57)
    if not (blank | neg | dig).all():
        bad = int(np.flatnonzero(~(blank | neg | dig).all(axis=1))[0])
        raise ValueError(f"{what}: a non-numeric character in "
                         f"{bytes(a[starts[bad] + lo:starts[bad] + hi])!r}")
    if (neg.sum(axis=1) > 1).any():
        raise ValueError(f"{what}: two minus signs in one field")
    val = np.where(dig, ch - 48, 0)
    # the place value of every digit: count digits to its right
    right = np.cumsum(dig[:, ::-1], axis=1)[:, ::-1] - dig
    out = (val * (10 ** right)).sum(axis=1)
    out = np.where(neg.any(axis=1), -out, out)
    return out, ~dig.any(axis=1)


RETRY_ERRORS = (urllib.error.URLError, ConnectionError, TimeoutError,
                OSError)


def get_bytes(url, attempts=4, sleep=3.0, headers=None, timeout=None):
    """GET -> bytes, retried on transient failures; None on a definite 404.

    `f10b.http_bytes` counts the bytes and tells ERDDAP's "no matching
    results" 404 (`f10b._Empty`) apart from a plain one; both come back as
    None here with the reason, because both mean "nothing at this url" to a
    per-file adapter. Any other failure retries with backoff and then RAISES.
    """
    err = None
    for i in range(max(1, attempts)):
        try:
            return f10b.http_bytes(url, headers=headers,
                                   timeout=timeout or f10b.SOCKET_TIMEOUT), "ok"
        except f10b._Empty:
            return None, "empty"
        except f10b._NotFound:
            return None, "notfound"
        except (IOError, *RETRY_ERRORS) as e:
            err = e
            # a 4xx other than 404 is our own bad request: do not retry it
            if "HTTP 4" in str(e) and "HTTP 429" not in str(e):
                break
            if i < attempts - 1:
                time.sleep(sleep * (2 ** i))
    raise IOError(f"{url}: {type(err).__name__}: {err}")


def range_reader(url, attempts=4, counter=None, headers=None):
    """`read_at(off, n)` over HTTP Range requests.

    The server must honour Range: a 200 where 206 was asked for is REFUSED,
    because that is the whole file arriving. A 206 shorter than asked is
    accepted ONLY when its Content-Range says the file ends there (a header
    read past a small file's end); anything else short RAISES. No listed size
    is trusted for clamping — THREDDS lists sizes rounded to 10 kB (measured:
    "37.24 Mbytes"). `headers` are added to every request (the sharded tier-G
    reader, `family1/sharded.py`, passes none for a public Hub file).
    """
    def read_at(off, n):
        if n <= 0:
            return b""
        end = off + n - 1
        err = None
        for i in range(max(1, attempts)):
            try:
                req = urllib.request.Request(
                    url, headers={**f10b.UA, **(headers or {}),
                                  "Range": f"bytes={off}-{end}"})
                with urllib.request.urlopen(req,
                                            timeout=f10b.SOCKET_TIMEOUT) as r:
                    if r.status != 206:
                        raise ValueError(f"{url}: asked for a byte range and "
                                         f"got HTTP {r.status} — refusing to "
                                         f"read the whole file")
                    cr = r.headers.get("Content-Range", "")
                    b = r.read()
                f10b.count_bytes(len(b))
                if counter is not None:
                    counter(len(b))
                if len(b) != n:
                    m = re.match(r"bytes (\d+)-(\d+)/(\d+)", cr)
                    at_eof = bool(m) and int(m.group(2)) == \
                        int(m.group(3)) - 1 and \
                        int(m.group(2)) - int(m.group(1)) + 1 == len(b)
                    if not at_eof:
                        raise IOError(f"{url}: range {off}-{end} returned "
                                      f"{len(b)} of {n} bytes ({cr!r})")
                return b
            except urllib.error.HTTPError as e:
                if e.code == 416:
                    return b""
                if e.code == 404:
                    raise f10b._NotFound(url) from None
                err = e
            except (IOError, *RETRY_ERRORS) as e:
                err = e
            if i < attempts - 1:
                time.sleep(2.0 * (2 ** i))
        raise IOError(f"{url} bytes {off}-{end}: {type(err).__name__}: {err}")
    return read_at


def ordered_map(fn, items, workers=4, lookahead=None):
    """`map(fn, items)` on a thread pool, results IN ORDER, never more than
    `lookahead` (default 2 x workers) results held at once."""
    items = list(items)
    if workers <= 1:
        for x in items:
            yield fn(x)
        return
    lookahead = lookahead or 2 * workers
    with cf.ThreadPoolExecutor(workers) as ex:
        q = collections.deque()
        it = iter(items)
        for x in it:
            q.append(ex.submit(fn, x))
            if len(q) >= lookahead:
                break
        while q:
            yield q.popleft().result()
            for x in it:
                q.append(ex.submit(fn, x))
                break


def batch_rows(adapter, cols, counts, label, yield_rows=1_000_000):
    """Packed columns -> (label, rows, counts) batches; counts on the first."""
    n = len(cols["bin"])
    if n == 0:
        yield label, None, counts
        return
    for i, lo in enumerate(range(0, n, yield_rows)):
        part = {k: v[lo:lo + yield_rows] for k, v in cols.items()}
        yield f"{label} rows {lo:,}+", part, (counts if i == 0 else None)


def concat_rows(parts, C, time_dtype):
    parts = [p for p in parts if p is not None and len(p["bin"])]
    if not parts:
        return f10b.empty_rows(C, time_dtype)
    return {k: np.concatenate([p[k] for p in parts], axis=0)
            for k in f10b.ROW_KEYS}


def add_counts(into, new):
    return f10b._merge_counts(into, new)


# ========================================================== Earthdata =====
# EARTHDATA LOGIN FOR A DOWNLOAD, THROUGH A .netrc, IPv4 ONLY.
#
# Every NASA archive in family 1 (LP DAAC, GES DISC, PO.DAAC, the SWOT
# archive) answers an unauthenticated GET with a 302 to
# `urs.earthdata.nasa.gov/oauth/authorize`, and `urllib` follows that
# redirect without credentials and lands on `HTTP 401 — HTTP Basic: Access
# denied.` (measured 2026-09-18, family1-build run #152: every SWOT pass).
# `requests` is what works: `Session.rebuild_auth` looks the host up in
# `~/.netrc` on EVERY redirect hop and STRIPS any Authorization header on a
# cross-host hop, so a netrc naming `urs.earthdata.nasa.gov` ALONE sends the
# password to Earthdata Login and to nothing else — not to the archive, not
# to the signed S3 URL the cloud hands back. `family1-build.yml` writes that
# netrc on hosted runners only, mode 600, and deletes it in an always() step.
#
# IPv4 IS FORCED for the same reason `ml/family1/earthdata_check.py` forces
# it: GitHub's runners resolve AAAA records for these hosts and have no IPv6
# route, so a request dies with `[Errno 101] Network is unreachable` before
# any HTTP status exists — which reads exactly like an archive refusing the
# account and is nothing of the kind (BUILD_LOG.md, run #3).
#
# NOTE, 2026-09-18: `family1/adapters/_modis_cmg.py` carries the same trio,
# written in the same wave for the MODIS CMG stores. They should be one copy;
# consolidating them is a follow-up, and until then a change to one belongs
# in both.
URS_HOST = "urs.earthdata.nasa.gov"
_IPV4 = {"done": False}


def force_ipv4_once():
    if _IPV4["done"]:
        return
    from family1 import earthdata_check as edc
    edc.force_ipv4(True)
    _IPV4["done"] = True


def netrc_has_urs(home=None):
    """Does a netrc this process can see name Earthdata Login?"""
    p = os.environ.get("NETRC") or os.path.join(
        home or os.path.expanduser("~"), ".netrc")
    try:
        with open(p, "r") as fh:
            return URS_HOST in fh.read()
    except OSError:
        return False


def earthdata_credentials(env=None):
    """(user, password) from the environment, or (None, None)."""
    env = os.environ if env is None else env
    u = env.get("EARTHDATA_USERNAME") or ""
    p = env.get("EARTHDATA_PASSWORD") or ""
    return (u, p) if (u and p) else (None, None)


def earthdata_ready(env=None):
    """Can this process authenticate to Earthdata at all — env or netrc?"""
    u, _p = earthdata_credentials(env)
    return bool(u) or netrc_has_urs()


def earthdata_session(env=None):
    """A `requests` session that authenticates to Earthdata Login ONLY.

    TWO ROUTES, AND THE EXPLICIT ONE IS PREFERRED BECAUSE IT WAS MEASURED.
    `ml/family1/earthdata_check.session(user, password)` sets
    `trust_env = False` and re-prepares Basic auth on every hop INTO
    urs.earthdata.nasa.gov (and strips it on every other hop), and that is the
    session family1-build run #90 measured getting HTTP 206 out of LP DAAC,
    GES DISC and PO.DAAC with this account (BUILD_LOG: "The Earthdata account
    is GOOD"). The netrc route — `trust_env = True`, letting `requests` look
    the host up per redirect hop — works for LP DAAC and for PO.DAAC's SWOT
    archive (measured: family1-build #150 and #168) and returned
    `HTTP 401 after 2 redirect(s)` from GES DISC's
    data.gesdisc.earthdata.nasa.gov in #170, whose 302 carries the SAME
    Earthdata client_id (e2WVk8Pw6weeLUKZYOxvTQ) as the disc2 host #90
    succeeded against — so the account is not the problem and the session
    construction is. The netrc route is kept as the fallback for a process
    that has a netrc and no environment variables.
    """
    from family1 import earthdata_check as edc
    user, password = earthdata_credentials(env)
    if user:
        force_ipv4_once()
        s = edc.session(user, password)
        s.headers.update(f10b.UA)
        return s
    import requests
    force_ipv4_once()
    s = requests.Session()
    s.trust_env = True                 # .netrc per redirect hop; see above
    s.headers.update(f10b.UA)
    return s


def earthdata_download(session, url, path, attempts=4, sleep=3.0,
                       timeout=300):
    """GET `url` -> `path`, size-verified. (bytes, None) or (None, "notfound").

    A definite 404 is a legitimate absence the caller counts. A 401/403 after
    an Earthdata hop is a DEFINITE refusal about the account and raises at
    once with the check to run; anything else retries and then raises. A short
    or empty body is a refusal, never a smaller record (ml/CLAUDE.md, the
    2026-09-14 rule).
    """
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    part = f"{path}.part{os.getpid()}"
    err = None
    for i in range(max(1, attempts)):
        got, want = 0, None
        try:
            with session.get(url, stream=True, timeout=timeout,
                             allow_redirects=True) as r:
                if r.status_code == 404:
                    return None, "notfound"
                if r.status_code in (401, 403):
                    raise IOError(
                        f"{url}: HTTP {r.status_code} after "
                        f"{len(r.history)} redirect(s) — Earthdata Login "
                        f"refused this account for this archive. Dispatch "
                        f"family1-build.yml with check_credentials=true and "
                        f"read the approval URL it prints.")
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

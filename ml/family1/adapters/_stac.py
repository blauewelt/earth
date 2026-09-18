"""Shared machinery for the SCENE CATALOGUE stores of family 1.0.tf (tier T).

PLAIN ENGLISH. Eight of family 1.0.tf's stores hold no pixels at all: they
list, one row per satellite scene, WHEN the picture was taken, WHERE its
footprint centre is, HOW CLOUDY it was, and WHICH instrument took it — so a
model can ask "which images exist for this cell in this week" without anyone
copying a petabyte. The pixels stay at the producer, and a sidecar table
(`assets.parquet`) maps each row back to the producer's own identifier and
download URLs.

This module is the part every one of those eight adapters shares:

  * `Fetcher`  — a JSON client that counts every byte into `ctx.bytes_fetched`,
    retries a transient failure, REFUSES a 4xx that is our own bad request,
    and — when `ctx.source_dir` is set — reads a CANNED response from disk
    instead, keyed by the request itself (`canned_key`). The smoke tests write
    those canned files, so a smoke exercises the REAL paging code.
  * four listing walkers, one per producer's API, each of which refuses a
    truncated page or a count that does not match the producer's own:
      - `stac_search`   a STAC API `POST /search` with cursor paging
                        (`numberMatched`); USGS landsatlook, Earth Search
      - `odata_window`  CDSE OData `$filter`/`$top`/`$skip` inside one time
                        window (`@odata.count`); `$skip` is capped at 10,000
                        by the server, so a window whose count exceeds what
                        paging can reach is SUBDIVIDED, never truncated
      - `cmr_granules`  CMR's granule search with `search-after` paging and
                        the `CMR-Hits` header as the producer's count
      - `asf_window`    ASF's `services/search/param` with `output=count` as
                        the producer's count and time windows for paging
  * `sphere_centre_area` — a footprint polygon (any of the four shapes the
    four producers publish it in) -> footprint centre and area in km² on the
    sphere.
  * `AssetWriter` / `read_asset_parts` / `write_assets_parquet` — the sidecar.
    A fetch writes its asset rows beside the year's `.npz` parts as
    `assets-NNNNN.npy` (a `.npy` is carried by `family10_parts_hub` and
    ignored by the assembler, so the sidecar travels with a lane's parts to
    the Hub and back for free); the assemble stage concatenates every part
    into one `assets.parquet` through `extra_files`.
  * `CatalogueAdapter` — the tier-P adapter base: the five channels, the
    footprint and time scales, the sensor and baseline code tables, the
    per-year day/month window plan, and the probe's one-month override.

THE FIVE CHANNELS (family1tf.tex, "The image catalogue, tier T"):

    cloud   %      cloud fraction, as the producer reports it
    valid   %      the fraction of the footprint that carries data
    angle   deg    sun ZENITH for an optical scene, radar INCIDENCE for a
                   SAR one (the note's "sun zenith or radar incidence")
    area    km2    the footprint's area on the sphere
    sensor  1      WHICH INSTRUMENT ON WHICH PLATFORM, a code from the
                   `sensor_table` in store.json

A channel a producer does not publish is NaN, never a guess: contract rule 3
and ml/CLAUDE.md §5.22. `cat_olci` adds a sixth, `coastal`.

`qc` IS THE PRODUCER'S PROCESSING BASELINE, mapped to a uint8 through the
adapter's `QC_TABLE` (recorded in store.json as `qc_table`). A baseline the
table does not list becomes `QC_OTHER` (255) and is COUNTED BY NAME in
`counts["qc_unlisted"]`, so a new baseline shows up in the build report
instead of silently becoming a different store. The table is fixed in the
adapter because `qc` must mean the same thing in every lane of a build, and a
table derived from what one lane happened to see would not.

`platform` IS THE HASHED SCENE IDENTIFIER — `f10b.platform_hash(stac_id)`, 60
bits of sha1. Two different identifiers hashing to the same value would make
`assets.parquet` ambiguous, so `AssetWriter` DETECTS that within a part and
refuses; across parts the assemble stage checks it again over the whole table.
`platform_meta` is False for every catalogue: the vocabulary IS the store
(one hash per row), and `assets.parquet` is its table.
"""
import datetime as dt
import hashlib
import io
import json
import math
import os
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request

import numpy as np

import build_family10_stores as f10b

EARTH_R_KM = 6371.0088
BIN_KM = 27.83                 # the family's footprint unit (family1tf.tex)
QC_OTHER = 255
SENSOR_OTHER = 255
ASSET_PART_PREFIX = "assets-"
ASSET_PART_SUFFIX = ".npy"
ASSETS_PARQUET = "assets.parquet"

# The five channels every catalogue store carries, in this order.
#
# THE FOOTPRINT AREA IS STORED AS log2(km2), NOT km2, AND THAT IS FORCED BY
# THE STORE'S OWN DTYPE. A family-10/1 store keeps `values` as **float16**,
# whose largest finite number is 65,504. A Landsat scene is 33,000 km2 and
# fits; a Sentinel-1 slice is 42,500 and just fits; but a Sentinel-3 OLCI
# granule is 1.5e6 km2 and a VIIRS six-minute granule is about 7e6, so both
# round to INFINITY on the way in. That is not a theory: `--smoke` for
# cat_olci and cat_s1 failed with `values.npy holds an infinity` the first
# time the channel was written in km2. log2 of the area cannot overflow, it is
# scale-free in the same way the store's own `fp` column is, and the note's
# "area" is recovered as 2**log2_area. The PRICE is float16's ten-bit
# mantissa: measured round trips are 0.007 % at 12,364 km2 (a Sentinel-2
# tile), 0.014 % at 33,300 (a Landsat scene), 0.06 % at 1.5e6 (an OLCI
# granule) and 0.31 % at 7e6 (a VIIRS granule), with a worst case of about
# 1.2 % at the top of the range — which is the accuracy of "how big is this
# footprint", not of a measurement. A footprint of zero or negative area is
# NaN and counted (`area_not_positive`), never -inf.
CAT_CHANNELS = (("cloud", "%", 0.0, 100.0),
                ("valid", "%", 0.0, 100.0),
                ("angle", "deg", 0.0, 90.0),
                ("log2_area", "log2(km2)", -10.0, 26.0),
                ("sensor", "1", 0.0, 255.0))


def area_channel(area_km2, counts):
    """The `log2_area` channel's value for a footprint area in km²."""
    try:
        a = float(area_km2)
    except (TypeError, ValueError):
        a = float("nan")
    if not (a > 0.0) or not math.isfinite(a):
        counts["area_not_positive"] = counts.get("area_not_positive", 0) + 1
        return float("nan")
    return math.log2(a)


class Refusal(ValueError):
    """A listing that does not add up — the 2026-09-14 rule.

    An empty page where the producer's own count says there are rows, a page
    shorter than the limit before the last one, a window whose rows do not sum
    to the producer's count, a response whose shape is not the one the adapter
    was written against. Every one of these is a refusal, never a skip.
    """


# ============================================================== the client ==
def canned_key(method, url, body=None, headers=None):
    """The file name a `--source-dir` run reads for one request.

    The smoke writes `<source_dir>/<store>/<canned_key(...)>.json`; the
    adapter reads it. Keying on the REQUEST means a smoke exercises the real
    URL building, the real paging and the real refusals — a canned file that
    the adapter never asks for is simply never read, and a request the smoke
    did not anticipate is a loud refusal rather than a pass.

    THE HEADERS ARE PART OF THE KEY, because one producer pages in a header
    and not in the URL: CMR's second page of a window is the same URL with a
    `CMR-Search-After` cursor. Leaving them out made every page of a window
    collapse onto one canned file, so the smoke could not exercise paging at
    all — the bug this parameter exists to prevent.
    """
    canon = json.dumps(body, sort_keys=True, separators=(",", ":")) if body \
        is not None else ""
    hd = json.dumps({str(k): str(v) for k, v in sorted((headers or {}).items())},
                    sort_keys=True, separators=(",", ":"))
    h = hashlib.sha1(f"{method} {url}\n{canon}\n{hd}".encode("utf-8")).hexdigest()
    return h[:24]


class Fetcher:
    """Counted, retried JSON over HTTP — or canned JSON from a directory."""

    RETRY_ON = (500, 502, 503, 504, 429, 408)

    def __init__(self, ctx, store, attempts=None, timeout=180.0):
        self.ctx = ctx
        self.store = store
        self.attempts = int(attempts or getattr(getattr(ctx, "a", None),
                                                "attempts", 3) or 3)
        self.timeout = timeout
        self.source_dir = getattr(ctx, "source_dir", "") or ""
        self.requests = 0
        self.retries = 0
        self.seconds = 0.0

    # -- canned ------------------------------------------------------------
    def _canned(self, method, url, body, headers=None):
        p = os.path.join(self.source_dir, self.store,
                         canned_key(method, url, body, headers) + ".json")
        if not os.path.exists(p):
            raise Refusal(
                f"{self.store}: --source-dir has no canned response for\n"
                f"  {method} {url}\n  body={json.dumps(body)[:400]}\n"
                f"  (expected {p}). The smoke must write every request the "
                f"adapter makes, so an unplanned request is a failure.")
        with open(p, "rb") as fh:
            raw = fh.read()
        self.ctx.count_bytes(len(raw))
        self.requests += 1
        return json.loads(raw.decode("utf-8")), dict(_canned_headers(raw))

    # -- network -----------------------------------------------------------
    def _do(self, method, url, body, headers):
        if self.source_dir:
            return self._canned(method, url, body, headers)
        data = None if body is None else \
            json.dumps(body, separators=(",", ":")).encode("utf-8")
        hdr = dict(f10b.UA)
        if data is not None:
            hdr["Content-Type"] = "application/json"
        hdr.update(headers or {})
        err = None
        for i in range(self.attempts):
            t0 = time.time()
            try:
                req = urllib.request.Request(url, data=data, headers=hdr,
                                             method=method)
                with urllib.request.urlopen(req, timeout=self.timeout) as r:
                    raw = r.read()
                    rh = dict(r.headers)
                self.seconds += time.time() - t0
                self.requests += 1
                f10b.count_bytes(len(raw))
                if not raw:
                    raise Refusal(f"{url}: an empty body with HTTP 200")
                return json.loads(raw.decode("utf-8")), rh
            except urllib.error.HTTPError as e:
                self.seconds += time.time() - t0
                detail = b""
                try:
                    detail = e.read()[:400]
                except Exception:                               # noqa: BLE001
                    pass
                err = f"HTTP {e.code} {detail!r}"
                # OUR OWN BAD REQUEST IS NOT TRANSIENT: a 400/404/422 means the
                # query is wrong and retrying it wastes the window.
                if e.code not in self.RETRY_ON:
                    raise Refusal(f"{url}: {err}") from None
            except (json.JSONDecodeError, UnicodeDecodeError) as e:
                self.seconds += time.time() - t0
                raise Refusal(f"{url}: the body is not JSON ({e})") from None
            except (OSError, urllib.error.URLError) as e:
                self.seconds += time.time() - t0
                err = f"{type(e).__name__}: {e}"
            if i < self.attempts - 1:
                self.retries += 1
                time.sleep(2.0 * (2 ** i))
        raise IOError(f"{url}: {err} (after {self.attempts} attempts)")

    def get(self, url, headers=None):
        return self._do("GET", url, None, headers)

    def post(self, url, body, headers=None):
        return self._do("POST", url, body, headers)


def _canned_headers(raw):
    """A canned file may carry its headers under `__headers__`."""
    try:
        d = json.loads(raw.decode("utf-8"))
    except Exception:                                           # noqa: BLE001
        return {}
    return (d.get("__headers__") or {}) if isinstance(d, dict) else {}


# ============================================== walker 1: a STAC API search ==
def stac_search(fet, url, body, limit, counts, what, max_pages=100000):
    """Every item of one STAC `POST /search`, paged by the `next` link.

    A GENERATOR, page by page: a Landsat day holds 1,400 scenes at 8 kB each
    and a Sentinel-2 day 15,800 at 4 kB, so materialising a window's raw JSON
    would cost hundreds of megabytes per worker for nothing. The caller turns
    each item into a `Scene` (about 200 bytes) and the page is freed.

    The producer's own count is `numberMatched` (STAC API 1.0's `context`
    extension calls it `context.matched`; both are read). Refusals:

      * a page with no `features` key at all — the response is not a STAC
        FeatureCollection;
      * a page SHORTER than `limit` that is not the last page (there is a
        `next` link) — a truncated page;
      * a total that is not the producer's `numberMatched`.

    `numberMatched` is read from the FIRST page and required to be stable:
    a catalogue that changes under a walk would otherwise be reported as a
    mismatch with no way to tell which number was right.
    """
    seen = 0
    matched = None
    page = 0
    nxt = None
    short_at = None
    while True:
        b = dict(body)
        b["limit"] = int(limit)
        if nxt is not None:
            b["next"] = nxt
        d, _ = fet.post(url, b)
        page += 1
        if "features" not in d:
            raise Refusal(f"{what}: {url} answered without a `features` key "
                          f"(keys {sorted(d)[:8]}) — not a STAC search")
        feats = d["features"] or []
        m = d.get("numberMatched")
        if m is None:
            m = (d.get("context") or {}).get("matched")
        if m is not None:
            m = int(m)
            if matched is None:
                matched = m
            elif m != matched:
                raise Refusal(
                    f"{what}: numberMatched changed from {matched} to {m} "
                    f"between page 1 and page {page} — the catalogue moved "
                    f"under the walk; re-run the window")
        link = _next_link(d)
        # A SHORT PAGE IS NOT TRUNCATION ON ITS OWN. Measured against
        # landsatlook 2026-09-18: the LAST page of a window comes back short
        # (66 of 100) and STILL carries a `next` link, so "short and a next
        # link" is the normal end of a window. What would be truncation is a
        # short page FOLLOWED BY more items — the server dropped rows in the
        # middle — and that is what is refused here. The count check at the
        # end is the other half and is the one that cannot be fooled.
        if short_at is not None and feats:
            raise Refusal(f"{what}: page {short_at} returned fewer items than "
                          f"the limit of {limit} and page {page} then "
                          f"returned {len(feats)} more — the server dropped "
                          f"items in the middle of the window")
        if len(feats) < limit:
            short_at = page
        seen += len(feats)
        yield from feats
        if link is None or not feats:
            break
        nxt = link
        if page >= max_pages:
            raise Refusal(f"{what}: more than {max_pages} pages — refusing "
                          f"to walk further")
    if matched is None:
        raise Refusal(f"{what}: the search never reported numberMatched, so "
                      f"there is nothing to check the {seen} item(s) "
                      f"against")
    if seen != matched:
        raise Refusal(f"{what}: walked {seen} item(s) and the producer's "
                      f"numberMatched says {matched}")
    counts["pages"] = counts.get("pages", 0) + page
    counts["producer_count"] = counts.get("producer_count", 0) + matched


def _next_link(d):
    """The `next` cursor of a stac-server response, or None at the end."""
    for l in (d.get("links") or []):
        if l.get("rel") != "next":
            continue
        b = l.get("body") or {}
        if "next" in b:
            return b["next"]
        # a GET-style next link: pull the cursor out of the query string
        q = urllib.parse.parse_qs(
            urllib.parse.urlparse(l.get("href", "")).query)
        for k in ("next", "cursor", "searchAfter"):
            if k in q:
                return q[k][0]
        raise Refusal(f"a `next` link this walker cannot follow: "
                      f"{json.dumps(l)[:300]}")
    return None


# ================================================ walker 2: CDSE OData ======
ODATA_TOP = 1000               # measured 2026-09-18: $top=2000 answers 422
ODATA_SKIP_MAX = 10000         # measured: "$skip should be <= 10000"
ODATA_REACH = ODATA_TOP + ODATA_SKIP_MAX


def odata_url(base, **kw):
    return base + "?" + urllib.parse.quote(
        "&".join("$" + k + "=" + str(v) for k, v in kw.items()),
        safe="=&$/',()*")


def odata_count(fet, base, filt):
    d, _ = fet.get(odata_url(base, filter=filt, top=1, count="true"))
    if "@odata.count" not in d:
        raise Refusal(f"{base}: $count=true answered without @odata.count "
                      f"(keys {sorted(d)[:8]})")
    return int(d["@odata.count"])


def odata_window(fet, base, filt, expand, counts, what, count=None):
    """Every product of one OData `$filter`, checked against `@odata.count`.

    CDSE caps `$skip` at 10,000, so at most 11,000 products are reachable
    inside one filter. A caller hands windows small enough for that; this
    function REFUSES a window it cannot walk whole rather than returning the
    first 11,000 (`odata_windows` is the splitter that avoids it).
    """
    n = odata_count(fet, base, filt) if count is None else int(count)
    if n > ODATA_REACH:
        raise Refusal(f"{what}: the producer counts {n} product(s) and "
                      f"$skip is capped at {ODATA_SKIP_MAX}, so only "
                      f"{ODATA_REACH} are reachable — split the window")
    got = 0
    skip = 0
    pages = 0
    while skip < n:
        kw = {"filter": filt, "top": min(ODATA_TOP, n - skip)}
        if skip:
            kw["skip"] = skip
        if expand:
            kw["expand"] = expand
        kw["orderby"] = "ContentDate/Start asc"
        d, _ = fet.get(odata_url(base, **kw))
        pages += 1
        if "value" not in d:
            raise Refusal(f"{what}: an OData page without a `value` key "
                          f"(keys {sorted(d)[:8]})")
        v = d["value"]
        if not v:
            raise Refusal(f"{what}: page {pages} at $skip={skip} is EMPTY "
                          f"while the producer counts {n} product(s) — an "
                          f"empty page is a broken listing, not the end")
        want = min(ODATA_TOP, n - skip)
        if len(v) < want and skip + len(v) < n:
            raise Refusal(f"{what}: page {pages} returned {len(v)} of {want} "
                          f"product(s) with {n - skip - len(v)} still to come "
                          f"— a truncated page")
        got += len(v)
        skip += len(v)
        yield from v
        if pages > 32:
            raise Refusal(f"{what}: more than 32 OData pages for a window "
                          f"the server says holds {n}")
    if got != n:
        raise Refusal(f"{what}: walked {got} product(s) and "
                      f"@odata.count says {n}")
    counts["pages"] = counts.get("pages", 0) + pages
    counts["producer_count"] = counts.get("producer_count", 0) + n


def odata_windows(fet, base, filt_for, t0, t1, counts, what, depth=0):
    """Split [t0, t1) until every piece is inside OData's paging reach.

    Yields `(lo, hi, count)`. The split is BINARY on time, so a burst of
    products in one hour costs a few extra count queries and nothing else. A
    window of one second that still holds more than 11,000 products cannot be
    paged at all and is a refusal — it has never happened and it would be a
    real failure, not a thing to work around.
    """
    n = odata_count(fet, base, filt_for(t0, t1))
    if n <= ODATA_REACH:
        yield t0, t1, n
        return
    span = (t1 - t0).total_seconds()
    if span <= 1.0:
        raise Refusal(f"{what}: {n} product(s) share the second {t0:%FT%TZ} "
                      f"and OData can page {ODATA_REACH}")
    if depth > 24:
        raise Refusal(f"{what}: window splitting reached depth {depth}")
    mid = t0 + dt.timedelta(seconds=math.floor(span / 2))
    counts["window_splits"] = counts.get("window_splits", 0) + 1
    yield from odata_windows(fet, base, filt_for, t0, mid, counts, what,
                             depth + 1)
    yield from odata_windows(fet, base, filt_for, mid, t1, counts, what,
                             depth + 1)


# ============================================ walker 3: CMR granule search ==
CMR_PAGE = 2000                # CMR's documented maximum page_size


def cmr_granules(fet, base, params, counts, what, page_size=CMR_PAGE,
                 max_pages=100000):
    """Every granule of one CMR query, paged with `search-after`.

    `CMR-Hits` is the producer's own count and is read from the first
    response; `CMR-Search-After` is the cursor. Refusals: a response without
    a `feed.entry` list, a short page before the last one, a total that is
    not `CMR-Hits`.
    """
    got = 0
    hits = None
    after = None
    pages = 0
    short_at = None
    while True:
        url = base + "?" + urllib.parse.urlencode(
            list(params) + [("page_size", int(page_size))], doseq=True)
        hdr = {"CMR-Search-After": after} if after else None
        d, rh = fet.get(url, headers=hdr)
        pages += 1
        # `.umm_json` answers `{"hits", "took", "items"}`; the older `.json`
        # answers `{"feed": {"entry": [...]}}`. Both are read, because the
        # eight catalogues use the leaner of the two per producer (umm_json is
        # 2.5 kB a VIIRS granule against 15 kB, and it is the ONLY form that
        # carries HLS's cloud cover, valid fraction and sun zenith).
        if "items" in d:
            ents = d["items"] or []
        elif isinstance(d.get("feed"), dict) and "entry" in d["feed"]:
            ents = d["feed"]["entry"] or []
        else:
            raise Refusal(f"{what}: a CMR response with neither `items` nor "
                          f"`feed.entry` (keys {sorted(d)[:8]})")
        h = _header(rh, "CMR-Hits")
        if h is not None:
            h = int(h)
            if hits is None:
                hits = h
            elif h != hits:
                raise Refusal(f"{what}: CMR-Hits changed from {hits} to {h} "
                              f"on page {pages} — the catalogue moved under "
                              f"the walk")
        after = _header(rh, "CMR-Search-After")
        # As in `stac_search`: a short LAST page that still carries a cursor
        # is the ordinary end of a window; a short page followed by more
        # granules is the server dropping rows, and that is the refusal. The
        # CMR-Hits check below is the half that cannot be fooled.
        if short_at is not None and ents:
            raise Refusal(f"{what}: page {short_at} returned fewer than the "
                          f"{page_size} asked for and page {pages} then "
                          f"returned {len(ents)} more — CMR dropped granules "
                          f"in the middle of the window")
        if len(ents) < page_size:
            short_at = pages
        got += len(ents)
        yield from ents
        if not ents or not after:
            break
        if pages >= max_pages:
            raise Refusal(f"{what}: more than {max_pages} CMR pages")
    if hits is None:
        raise Refusal(f"{what}: CMR never sent a CMR-Hits header, so the "
                      f"{got} granule(s) have nothing to check against")
    if got != hits:
        raise Refusal(f"{what}: walked {got} granule(s) and CMR-Hits "
                      f"says {hits}")
    counts["pages"] = counts.get("pages", 0) + pages
    counts["producer_count"] = counts.get("producer_count", 0) + hits


def cmr_url(base, params, page_size=1):
    return base + "?" + urllib.parse.urlencode(
        list(params) + [("page_size", int(page_size))], doseq=True)


def cmr_hits(fet, base, params):
    url = base + "?" + urllib.parse.urlencode(list(params) +
                                              [("page_size", 1)], doseq=True)
    _, rh = fet.get(url)
    h = _header(rh, "CMR-Hits")
    if h is None:
        raise Refusal(f"{url}: no CMR-Hits header")
    return int(h)


def _header(headers, name):
    low = name.lower()
    for k, v in (headers or {}).items():
        if str(k).lower() == low:
            return v
    return None


# ================================================ walker 4: ASF param search ==
ASF_MAX = 250                  # measured 2026-09-18: maxResults=1000 fails


def asf_url(base, params):
    return base + "?" + urllib.parse.urlencode(list(params), doseq=True)


def asf_count(fet, base, params):
    """`output=count` returns a bare integer, not JSON — read it as text."""
    url = asf_url(base, list(params) + [("output", "count")])
    raw = _text(fet, url)
    s = raw.strip()
    if not re.fullmatch(r"\d+", s):
        raise Refusal(f"{url}: output=count answered {s[:80]!r}, not an "
                      f"integer")
    return int(s)


def _text(fet, url):
    """A GET whose body is NOT JSON (ASF's `output=count`)."""
    if fet.source_dir:
        d, _ = fet.get(url)
        if not isinstance(d, dict) or "__text__" not in d:
            raise Refusal(f"{url}: the canned response carries no __text__")
        return str(d["__text__"])
    err = None
    for i in range(fet.attempts):
        t0 = time.time()
        try:
            req = urllib.request.Request(url, headers=dict(f10b.UA))
            with urllib.request.urlopen(req, timeout=fet.timeout) as r:
                raw = r.read()
            fet.seconds += time.time() - t0
            fet.requests += 1
            f10b.count_bytes(len(raw))
            return raw.decode("utf-8", "replace")
        except urllib.error.HTTPError as e:
            fet.seconds += time.time() - t0
            if e.code not in Fetcher.RETRY_ON:
                raise Refusal(f"{url}: HTTP {e.code}") from None
            err = e
        except (OSError, urllib.error.URLError) as e:
            fet.seconds += time.time() - t0
            err = e
        if i < fet.attempts - 1:
            fet.retries += 1
            time.sleep(2.0 * (2 ** i))
    raise IOError(f"{url}: {err}")


def asf_window(fet, base, params, counts, what):
    """Every granule of one ASF time window, checked against `output=count`.

    ASF has no cursor: `maxResults` caps a response at 250 and the only way
    to page is to narrow the window. So the caller hands windows and this
    function refuses one that does not fit, and `asf_windows` splits.
    """
    n = asf_count(fet, base, params)
    if n > ASF_MAX:
        raise Refusal(f"{what}: {n} granule(s) and ASF serves at most "
                      f"{ASF_MAX} per request — split the window")
    if n == 0:
        counts["windows_empty"] = counts.get("windows_empty", 0) + 1
        return []
    d, _ = fet.get(asf_url(base, list(params) + [("output", "jsonlite"),
                                                 ("maxResults", ASF_MAX)]))
    res = d.get("results")
    if not isinstance(res, list):
        raise Refusal(f"{what}: an ASF response without a `results` list "
                      f"(keys {sorted(d)[:8]})")
    if len(res) != n:
        raise Refusal(f"{what}: jsonlite returned {len(res)} granule(s) and "
                      f"output=count says {n}")
    counts["pages"] = counts.get("pages", 0) + 2
    counts["producer_count"] = counts.get("producer_count", 0) + n
    return res


def asf_windows(fet, base, params_for, t0, t1, counts, what, depth=0):
    """Split [t0, t1) until every piece holds at most `ASF_MAX` granules."""
    n = asf_count(fet, base, params_for(t0, t1))
    if n <= ASF_MAX:
        yield t0, t1, n
        return
    span = (t1 - t0).total_seconds()
    if span <= 1.0:
        raise Refusal(f"{what}: {n} granule(s) share one second at "
                      f"{t0:%FT%TZ} and ASF serves {ASF_MAX}")
    if depth > 28:
        raise Refusal(f"{what}: window splitting reached depth {depth}")
    mid = t0 + dt.timedelta(seconds=math.floor(span / 2))
    counts["window_splits"] = counts.get("window_splits", 0) + 1
    yield from asf_windows(fet, base, params_for, t0, mid, counts, what,
                           depth + 1)
    yield from asf_windows(fet, base, params_for, mid, t1, counts, what,
                           depth + 1)


# ================================================================ geometry ==
def sphere_centre_area(lats, lons):
    """A footprint's ring -> (centre lat, centre lon in [-180,180), area km²).

    The CENTRE is the spherical mean of the ring's vertices: each vertex is a
    unit 3-vector, the vectors are averaged, and the average is projected back
    to the sphere. Averaging the latitudes and longitudes instead is wrong
    across the antimeridian — and a quarter of Sentinel-2's tiles and most of
    NISAR's polar frames cross it — while the 3-vector mean has no seam at
    all. A degenerate ring (all vertices antipodal in the mean, which cannot
    happen for a real scene) is a refusal.

    The AREA is the spherical polygon's own area,

        A = R² · |Σ (λ_{i+1} − λ_i)·(2 + sin φ_i + sin φ_{i+1})| / 2

    with each longitude difference wrapped into (−π, π], so the formula is
    seam-free too. It is exact on the sphere (no projection), which is what
    the `area` channel claims.
    """
    la = np.asarray(lats, np.float64)
    lo = np.asarray(lons, np.float64)
    if la.size != lo.size or la.size < 3:
        raise Refusal(f"a footprint ring of {la.size} vertex/vertices")
    if not (np.isfinite(la).all() and np.isfinite(lo).all()):
        raise Refusal("a footprint ring with a non-finite vertex")
    if la.size > 3 and abs(la[0] - la[-1]) < 1e-12 and \
            abs(lo[0] - lo[-1]) < 1e-12:
        la, lo = la[:-1], lo[:-1]           # drop the repeated closing vertex
    phi = np.radians(la)
    lam = np.radians(lo)
    x = (np.cos(phi) * np.cos(lam)).sum()
    y = (np.cos(phi) * np.sin(lam)).sum()
    z = np.sin(phi).sum()
    r = math.sqrt(x * x + y * y + z * z)
    if r < 1e-9:
        raise Refusal("a footprint ring whose vertices average to the centre "
                      "of the Earth")
    clat = math.degrees(math.asin(max(-1.0, min(1.0, z / r))))
    clon = math.degrees(math.atan2(y, x))
    p2 = np.roll(phi, -1)
    l2 = np.roll(lam, -1)
    dl = (l2 - lam + math.pi) % (2 * math.pi) - math.pi
    s = float((dl * (2.0 + np.sin(phi) + np.sin(p2))).sum())
    area = abs(s) * EARTH_R_KM * EARTH_R_KM / 2.0
    return clat, float(f10b.f10.wrap_lon(clon)), area


def ring_from_wkt(wkt):
    """`POLYGON ((lon lat, ...))` -> (lats, lons). CDSE and ASF both use it."""
    m = re.search(r"POLYGON\s*\(\(\s*(.*?)\s*\)\)", str(wkt), re.S | re.I)
    if not m:
        raise Refusal(f"not a simple WKT POLYGON: {str(wkt)[:120]!r}")
    lats, lons = [], []
    for pair in m.group(1).split(","):
        p = pair.split()
        if len(p) < 2:
            raise Refusal(f"a WKT vertex {pair!r}")
        lons.append(float(p[0]))
        lats.append(float(p[1]))
    return [(lats, lons)]


def ring_from_geojson(geom):
    """A GeoJSON Polygon or MultiPolygon -> a LIST of `(lats, lons)` rings.

    Every ring reader in this module returns a LIST, even for one ring, so
    `centre_area` never has to guess how deeply its argument is nested. A
    footprint the producer publishes as a MultiPolygon is a scene the
    antimeridian cut in two: the pieces are the same scene, so their areas add
    and the centre is the area-weighted spherical mean of the pieces.
    """
    t = (geom or {}).get("type")
    if t == "Polygon":
        c = geom["coordinates"][0]
        return [([p[1] for p in c], [p[0] for p in c])]
    if t == "MultiPolygon":
        return [([p[1] for p in poly[0]], [p[0] for p in poly[0]])
                for poly in geom["coordinates"]]
    raise Refusal(f"a footprint geometry of type {t!r}")


def centre_area(rings):
    """`sphere_centre_area` over a LIST of `(lats, lons)` rings."""
    if isinstance(rings, dict):
        rings = ring_from_geojson(rings)
    if not rings:
        raise Refusal("a footprint with no ring at all")
    if len(rings) == 1:
        return sphere_centre_area(rings[0][0], rings[0][1])
    parts = [sphere_centre_area(a, b) for a, b in rings]
    tot = sum(p[2] for p in parts)
    if tot <= 0:
        raise Refusal("a multi-part footprint of zero area")
    x = y = z = 0.0
    for clat, clon, a in parts:
        w = a / tot
        ph, lm = math.radians(clat), math.radians(clon)
        x += w * math.cos(ph) * math.cos(lm)
        y += w * math.cos(ph) * math.sin(lm)
        z += w * math.sin(ph)
    r_ = math.sqrt(x * x + y * y + z * z)
    return (math.degrees(math.asin(max(-1.0, min(1.0, z / r_)))),
            float(f10b.f10.wrap_lon(math.degrees(math.atan2(y, x)))), tot)


def ring_from_cmr(entry):
    """CMR's `polygons` / `boxes` / `points` -> rings, whichever it publishes.

    `polygons` is a list of lists of `"lat lon lat lon ..."`; `boxes` is
    `"s w n e"`; `points` is `"lat lon"` (a granule with no footprint, which
    is a refusal — a catalogue row with no place is not a row).
    """
    pol = entry.get("polygons")
    if pol:
        rings = []
        for grp in pol:
            for s in (grp if isinstance(grp, list) else [grp]):
                v = [float(x) for x in str(s).split()]
                if len(v) < 6 or len(v) % 2:
                    raise Refusal(f"a CMR polygon of {len(v)} number(s)")
                rings.append((v[0::2], v[1::2]))
        return rings
    box = entry.get("boxes")
    if box:
        v = [float(x) for x in str(box[0]).split()]
        if len(v) != 4:
            raise Refusal(f"a CMR box of {len(v)} number(s)")
        s, w, n, e = v
        return [([s, s, n, n], [w, e, e, w])]
    raise Refusal(f"granule {entry.get('title')!r} publishes no footprint "
                  f"(keys {sorted(entry)[:12]})")


def log2_fp_for_km(width_km):
    """The note's footprint scale: a 110 km Sentinel-2 tile is +2.0."""
    return float(math.log2(float(width_km) / BIN_KM))


def log2_dt_for_seconds(seconds):
    """The note's time scale: the fraction of a 5-day bin one scene occupies."""
    return float(math.log2((float(seconds) / 86400.0) / 5.0))


# ============================================================ the sidecar ===
ASSET_COLS = ("platform", "stac_id", "collection", "base_url", "asset_set")


class AssetWriter:
    """The asset rows of one year, written beside that year's `.npz` parts.

    WHY A `.npy`. `family10_parts_hub` carries `.npz`, `.zst` and `.npy` out
    of a year directory to the Hub and back, and both assemblers read only the
    `.npz`. So a `.npy` in the year directory is exactly the vehicle a sidecar
    needs: a lane's asset rows travel with its parts, a `--parts-from-hub`
    assembly gets them for free, and no part of the framework has to learn
    about them. The bytes inside are an Arrow IPC stream (zstd-compressed),
    which is what `assets.parquet` is written from.

    The writer flushes every `flush_rows` rows so a year larger than memory
    still costs one buffer, and it refuses a hash collision inside the year.
    """

    def __init__(self, ctx, year, flush_rows=1_000_000):
        self.ctx = ctx
        self.year = int(year)
        self.dir = ctx.year_dir(self.year)
        self.flush_rows = int(flush_rows)
        self.seq = 0
        self.n = 0
        self.rows = {k: [] for k in ASSET_COLS}
        # A SET OF HASHES, not a hash -> identifier map. Sentinel-2's busiest
        # year is 5.07 million products and holding their 65-character
        # identifiers here would cost about a gigabyte for a check that a set
        # of int64 does in 300 MB. What the set cannot do is tell a REPEATED
        # scene from a 60-bit hash COLLISION, so it counts both as
        # `rows_duplicate_scene` and the authoritative check happens once, at
        # assemble time, in `write_assets_parquet`, where numpy can test the
        # whole column at once.
        self._seen = set()

    def add(self, platform, stac_id, collection, base_url, asset_set):
        if platform in self._seen:
            return False                      # the same scene twice: one row
        self._seen.add(platform)
        self.rows["platform"].append(int(platform))
        self.rows["stac_id"].append(str(stac_id))
        self.rows["collection"].append(str(collection))
        self.rows["base_url"].append(str(base_url))
        self.rows["asset_set"].append(str(asset_set))
        self.n += 1
        if len(self.rows["platform"]) >= self.flush_rows:
            self.flush()
        return True

    def flush(self):
        if not self.rows["platform"]:
            return
        os.makedirs(self.dir, exist_ok=True)
        blob = _asset_ipc_bytes(self.rows)
        p = os.path.join(self.dir,
                         f"{ASSET_PART_PREFIX}{self.seq:05d}"
                         f"{ASSET_PART_SUFFIX}")
        tmp = f"{p}.tmp{os.getpid()}.npy"
        np.save(tmp, np.frombuffer(blob, np.uint8))
        os.replace(tmp, p)
        self.seq += 1
        self.rows = {k: [] for k in ASSET_COLS}

    def close(self):
        self.flush()
        return self.n


def _pa():
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq          # noqa: F401
    except ImportError:                                         # pragma: no cover
        sys.exit("the catalogue stores publish assets.parquet and pyarrow is "
                 "not installed — `pip install pyarrow` (it is in "
                 ".github/workflows/family1-build.yml's install step)")
    return pa


def _asset_table(cols):
    pa = _pa()
    return pa.table({
        "platform": pa.array(cols["platform"], pa.int64()),
        "stac_id": pa.array(cols["stac_id"], pa.string()),
        "collection": pa.array(cols["collection"], pa.string()),
        "base_url": pa.array(cols["base_url"], pa.string()),
        "asset_set": pa.array(cols["asset_set"], pa.string()),
    })


def _asset_ipc_bytes(cols):
    pa = _pa()
    t = _asset_table(cols)
    sink = pa.BufferOutputStream()
    opt = pa.ipc.IpcWriteOptions(compression="zstd")
    with pa.ipc.new_stream(sink, t.schema, options=opt) as w:
        w.write_table(t)
    return sink.getvalue().to_pybytes()


def asset_part_names(d):
    if not os.path.isdir(d):
        return []
    return sorted(n for n in os.listdir(d)
                  if n.startswith(ASSET_PART_PREFIX)
                  and n.endswith(ASSET_PART_SUFFIX))


def read_asset_parts(ctx):
    """Every year's asset sidecars, as Arrow tables, in year order."""
    pa = _pa()
    for y in ctx.years:
        d = ctx.year_dir(y)
        for n in asset_part_names(d):
            raw = np.load(os.path.join(d, n))
            with pa.ipc.open_stream(pa.py_buffer(raw.tobytes())) as r:
                yield y, n, r.read_all()


def write_assets_parquet(ctx, dest, expect_rows=None):
    """Concatenate every sidecar into `assets.parquet`. Returns (path, meta).

    The whole table's `platform` column must be UNIQUE — a collision inside
    one year is caught by `AssetWriter`, and this is the check across years —
    and, when `expect_rows` is given (the store's N), the table must have
    exactly that many rows: one asset entry per catalogue row, no more and no
    fewer.
    """
    pa = _pa()
    import pyarrow.parquet as pq
    out = os.path.join(dest, ASSETS_PARQUET)
    tmp = out + f".tmp{os.getpid()}"
    # STREAMED, ONE SIDECAR AT A TIME. Sentinel-2's table is 41.2 million rows
    # whose `stac_id` and `base_url` are unique per row (a CDSE product's url
    # carries its own uuid), so concatenating every sidecar into one Arrow
    # table before writing would need about 8 GB — more than a hosted runner
    # has to spare beside the store it is assembling. Each part becomes one
    # parquet row group instead, and the uniqueness check accumulates the
    # `platform` column alone (8 bytes a row, 330 MB at that size).
    plats, parts, rows = [], [], 0
    w = None
    try:
        for y, n, t in read_asset_parts(ctx):
            if w is None:
                w = pq.ParquetWriter(tmp, t.schema, compression="zstd",
                                     compression_level=15,
                                     use_dictionary=True)
            w.write_table(t)
            plats.append(t.column("platform").to_numpy(zero_copy_only=False))
            rows += t.num_rows
            parts.append(f"{y}/{n}")
    finally:
        if w is not None:
            w.close()
    if w is None:
        raise Refusal(
            f"{ctx.adapter.store}: no {ASSET_PART_PREFIX}*{ASSET_PART_SUFFIX} "
            f"sidecar in any year directory, so the store cannot publish "
            f"{ASSETS_PARQUET}. The fetch writes one per part; a store "
            f"assembled from Hub parts pushed by an older builder has none, "
            f"and must be re-fetched.")
    plat = np.concatenate(plats) if plats else np.zeros(0, np.int64)
    del plats
    u = np.unique(plat)
    if u.size != plat.size:
        vals, cnt = np.unique(plat, return_counts=True)
        dup = vals[cnt > 1][:4].tolist()
        raise Refusal(
            f"{ctx.adapter.store}: {plat.size - u.size} repeated platform "
            f"hash(es) across the year sidecars, e.g. {dup}. Either a scene "
            f"is listed in two YEARS (a window bug) or two identifiers "
            f"collide in the 60-bit hash; assets.parquet must be one row per "
            f"scene either way.")
    if expect_rows is not None and plat.size != int(expect_rows):
        raise Refusal(f"{ctx.adapter.store}: {ASSETS_PARQUET} has "
                      f"{plat.size} row(s) and the store has "
                      f"{int(expect_rows)} — every catalogue row must have "
                      f"exactly one asset entry")
    os.replace(tmp, out)
    return out, {"rows": int(rows), "parts": len(parts),
                 "bytes": os.path.getsize(out),
                 "row_groups": len(parts),
                 "columns": list(ASSET_COLS)}


# ============================================================== the adapter ==
class CatalogueAdapter(f10b.SourceAdapter):
    """The tier-T catalogue base: five channels, a sidecar, and window plans.

    A subclass states the producer (`SENSOR_TABLE`, `QC_TABLE`, the endpoint
    and the collection names), implements

        plan(ctx, year)    -> a list of opaque "window" objects
        scenes(ctx, w, counts) -> yields `Scene` for one window

    and inherits everything else: the channels, `fetch_year`, `fetch_month`,
    the asset sidecar, the store.json extras and the probe's assertions.
    """
    family = "1tf"
    distribution = "public"
    time_dtype = "int32"
    per_year = True
    platform_meta = False
    tier = "P"
    channels = CAT_CHANNELS
    credentials = ()
    fetch_month_scope = "month"

    #: the code table for the `sensor` channel: {code: text}
    SENSOR_TABLE = {}
    #: the code table for `qc`, the producer's processing baseline
    QC_TABLE = {}
    #: {name: [file names, with `{id}` for the scene identifier]}
    ASSET_SETS = {}

    YIELD_ROWS = 500_000
    WORKERS = 8

    scene_seconds = 1.0        # the nominal acquisition length of one scene
    footprint_km = 110.0       # the nominal footprint width

    def __init__(self):
        self._plan_cache = {}

    # -- derived scales ----------------------------------------------------
    @property
    def log2_fp(self):
        return log2_fp_for_km(self.footprint_km)

    @property
    def log2_dt(self):
        return log2_dt_for_seconds(self.scene_seconds)

    # -- code tables -------------------------------------------------------
    def sensor_code(self, text, counts):
        for code, name in self.SENSOR_TABLE.items():
            if name == text:
                return code
        d = counts.setdefault("sensor_unlisted", {})
        d[str(text)] = d.get(str(text), 0) + 1
        return SENSOR_OTHER

    def qc_code(self, text, counts):
        for code, name in self.QC_TABLE.items():
            if name == text:
                return code
        d = counts.setdefault("qc_unlisted", {})
        d[str(text)] = d.get(str(text), 0) + 1
        return QC_OTHER

    # -- the two things a subclass writes ----------------------------------
    def plan(self, ctx, year, month=None):
        """The time windows this year (or month) is listed in, in order."""
        raise NotImplementedError

    def scenes(self, ctx, window):
        """One window -> `(list of Scene, counts)`.

        RUNS IN A WORKER THREAD, one window per thread, so it takes no
        shared state and returns its own counts for the caller to merge.
        The windows are consumed IN ORDER (`cm.ordered_map`), which is what
        makes the store's tie-break among equal `(bin, time_s)` reproducible
        and what the smoke's truth ordering relies on.
        """
        raise NotImplementedError

    def record_first(self):
        """(year, month) the producer's record starts — for the index."""
        return (self.first_year, 1)

    def workers(self, ctx):
        return 1 if getattr(ctx, "source_dir", "") else self.WORKERS

    SMOKE_PAGE = 3

    def page_for(self, ctx, normal):
        """The page size — three against a canned archive, `normal` otherwise.

        A smoke whose page size is the production one never turns a page: the
        whole synthetic window fits in the first response, so the cursor, the
        short-page rule and the count check are all untested. Three items a
        page makes a seven-granule window three pages, which is what the
        smokes actually exercise. The value is a function of the CONTEXT, not
        a mutable attribute, because the probe builds its own adapter instance
        and a mutation would not reach it.
        """
        return self.SMOKE_PAGE if getattr(ctx, "source_dir", "") \
            else int(normal)

    def windows(self, ctx, year, month=None, step="day"):
        """The time windows of `year` (or one month), CLIPPED THREE WAYS.

        A window is listed only if it overlaps all three of: the calendar year
        (or month) asked for, the producer's own record (`record_first()` and
        `record_last()`), and the run's `--start`/`--end` days. The third
        clip is what makes a two-day smoke cost two requests instead of 366,
        and what keeps a probe from asking for a month's worth of windows
        outside the month.
        """
        first = self.record_first()
        last = getattr(self, "record_last", lambda: None)()
        lo = dt.datetime.combine(getattr(ctx, "d_lo", None) or
                                 dt.date(int(year), 1, 1), dt.time())
        hi = dt.datetime.combine(getattr(ctx, "d_hi", None) or
                                 dt.date(int(year), 12, 31), dt.time()) \
            + dt.timedelta(days=1)
        spans = month_range(int(year), month, first=first, last=last)
        step_dt = {"day": dt.timedelta(days=1),
                   "hour": dt.timedelta(hours=1)}.get(step)
        out = []
        for a, b in spans:
            a, b = max(a, lo), min(b, hi)
            if a >= b:
                continue
            if step_dt is None:                 # one window per month
                out.append((a, b))
                continue
            w = a
            while w < b:
                out.append((w, min(w + step_dt, b)))
                w += step_dt
        return out

    # -- the fetch ---------------------------------------------------------
    def _rows(self, ctx, windows, label, t_lo, t_hi, sidecar, counts):
        """Every window's scenes, packed, yielded in batches.

        The windows are FETCHED CONCURRENTLY and CONSUMED IN ORDER. Listing a
        whole archive is latency-bound — one Landsat page of 100 scenes costs
        0.8 s of round trip and 0.01 s of parsing — so eight windows in flight
        is the difference between 123 scenes a second and 859 (measured
        2026-09-18 against landsatlook). Everything after the fetch (the
        collision check, the packing, the sidecar) stays single-threaded and
        in window order.
        """
        t, la, lo, vals, plat, qc = [], [], [], [], [], []
        n_out = 0

        def emit(final=False):
            nonlocal t, la, lo, vals, plat, qc, n_out
            if not t or (not final and len(t) < self.YIELD_ROWS):
                return None
            v = np.array(vals, np.float64).reshape(len(t), self.C)
            oob = self.mask_bounds(v)
            if oob:
                f10b._merge_counts(counts, {"out_of_bounds": oob})
            rows = self.pack(np.array(t, np.int64),
                             np.array(la, np.float32),
                             np.array(lo, np.float32), v,
                             np.array(plat, np.int64),
                             np.array(qc, np.uint8))
            n_out += len(t)
            t, la, lo, vals, plat, qc = [], [], [], [], [], []
            return rows

        from family1.adapters import _common as cm
        fn = (lambda w: self.scenes(ctx, w))
        for scs, c in cm.ordered_map(fn, windows, self.workers(ctx)):
            f10b._merge_counts(counts, c or {})
            for sc in scs:
                if not (t_lo <= sc.t <= t_hi):
                    counts["rows_outside_window"] = \
                        counts.get("rows_outside_window", 0) + 1
                    continue
                if sidecar is not None and not sidecar.add(
                        sc.platform, sc.stac_id, sc.collection, sc.base_url,
                        sc.asset_set):
                    counts["rows_duplicate_scene"] = \
                        counts.get("rows_duplicate_scene", 0) + 1
                    continue
                t.append(sc.t)
                la.append(sc.lat)
                lo.append(sc.lon)
                vals.append(sc.values)
                plat.append(sc.platform)
                qc.append(sc.qc)
            r = emit()
            if r is not None:
                yield f"{label} rows {n_out - len(r['bin']):,}+", r, None
        r = emit(final=True)
        counts["rows_kept"] = counts.get("rows_kept", 0) + n_out
        yield label, r, counts

    def fetch_year(self, ctx, year):
        t0 = time.time()
        counts = {}
        windows = self.plan(ctx, year)
        counts["windows"] = len(windows)
        if not windows:
            ctx.note_absent(year, f"{self.store}: the plan for {year} is "
                                  f"empty, and an empty plan for a year "
                                  f"inside the record is a broken index")
            return
        side = AssetWriter(ctx, year)
        held = None
        for label, rows, c in self._rows(ctx, windows, str(year), ctx.t_lo,
                                         ctx.t_hi, side, counts):
            if c is None:
                yield label, rows, c
            else:
                # HOLD THE BATCH THAT CARRIES THE COUNTS until the sidecar is
                # closed: `PartWriter.add` copies the dict it is handed, so a
                # number added to `counts` after the yield is silently lost.
                held = (label, rows, c)
        side.close()
        if held is not None:
            label, rows, c = held
            c["asset_rows"] = side.n
            c["asset_parts"] = side.seq
            c["fetch_seconds"] = round(time.time() - t0, 1)
            yield label, rows, c

    def fetch_month(self, ctx, year, month):
        """The probe: the month's own windows, nothing else fetched."""
        lo, hi = f10b.month_bounds_s(year, month)
        lo, hi = max(lo, ctx.t_lo), min(hi, ctx.t_hi)
        counts = {"probe_month_only": 1}
        windows = self.plan(ctx, year, month=month)
        counts["windows"] = len(windows)
        # The probe keeps the SAME de-duplication as a build (a scene listed
        # in two windows is one row either way) and writes no sidecar, so its
        # row count is the build's row count for that month.
        yield from self._rows(ctx, windows, f"{year}-{month:02d}", lo, hi,
                              _Dedup(), counts)

    # -- what the store publishes beside its columns -----------------------
    def extra_files(self, ctx, dest):
        # The row count is checked in `extra_meta`, which is the hook the
        # framework hands N to; here the table is only concatenated and
        # written, so a mismatch is reported with both numbers in scope.
        path, meta = write_assets_parquet(ctx, dest)
        self._assets_meta = meta
        print(f"  assets: {meta['rows']:,} scene(s) from {meta['parts']} "
              f"part(s) -> {ASSETS_PARQUET} "
              f"{meta['bytes'] / 1e6:.1f} MB")
        return {ASSETS_PARQUET: path}

    def extra_meta(self, ctx, dest, N, values):
        am = dict(getattr(self, "_assets_meta", {}) or {})
        if am.get("rows") not in (None, int(N)):
            sys.exit(f"{self.store}: assets.parquet holds {am['rows']} "
                     f"scene(s) and the store holds {N} row(s)")
        return {
            "tier_t_catalogue": True,
            "sensor_table": {str(k): v for k, v in
                             sorted(self.SENSOR_TABLE.items())},
            "sensor_other_code": SENSOR_OTHER,
            "qc_table": {str(k): v for k, v in sorted(self.QC_TABLE.items())},
            "qc_other_code": QC_OTHER,
            "asset_sets": {k: list(v) for k, v in
                           sorted(self.ASSET_SETS.items())},
            "assets": am,
            "assets_note": (
                "assets.parquet has one row per catalogue row: `platform` (the "
                "row's hash), `stac_id` (the producer's identifier), "
                "`collection`, `base_url` (the directory the scene's files "
                "live in) and `asset_set` (a key of `asset_sets`, whose value "
                "lists the file names, with `{id}` standing for `stac_id`). A "
                "file's URL is base_url + '/' + name.replace('{id}', "
                "stac_id)."),
            "footprint_km_nominal": self.footprint_km,
            "scene_seconds_nominal": self.scene_seconds,
            "qc_keep_not_applicable": True,
            "qc_keep_note": (
                "store.json's `qc_keep_max` is family 10.1's per-value "
                "quality GRADE and means nothing for a tier-T catalogue: `qc` "
                "here is the producer's PROCESSING BASELINE through "
                "`qc_table`, so no row is ever dropped for its qc and a "
                "higher code is not a worse row."),
        }


# ======================================================= the asset template ==
def asset_template(hrefs, scene_id, what, drop_hosts=()):
    """The producer's own asset URLs -> `(base_url, template)`.

    `hrefs` is whatever the producer published. Every one of them must sit in
    ONE directory (the scene's own), and the template is the file names with
    the scene identifier replaced by `{id}`, sorted and space-joined. Nothing
    is derived from the scene name: the URLs come from the producer and the
    template is a lossless re-encoding of them, so a consumer rebuilds each
    URL as `base_url + "/" + name.replace("{id}", stac_id)`.

    It is one string per row and parquet's dictionary encoding collapses it to
    one dictionary entry per DISTINCT set — two for the whole Landsat archive,
    a handful for each of the others — which is why the template is stored
    literally rather than through a code table that the lanes of one build
    could not agree on.
    """
    dirs, names = set(), set()
    for h in hrefs:
        h = str(h or "")
        if not h or any(h.startswith(p) for p in drop_hosts):
            continue
        d, _, nm = h.rpartition("/")
        if not d or not nm:
            raise Refusal(f"{what}: an asset url with no file name: {h!r}")
        dirs.add(d)
        names.add(nm)
    if not names:
        raise Refusal(f"{what}: no asset url at all")
    if len(dirs) != 1:
        raise Refusal(f"{what}: assets spread over {len(dirs)} directories: "
                      f"{sorted(dirs)[:3]}")
    base = dirs.pop()
    tmpl = " ".join(sorted(n.replace(scene_id, "{id}") for n in names))
    return base, tmpl


# ============================================== a CMR-listed catalogue store ==
CMR_GRANULES = "https://cmr.earthdata.nasa.gov/search/granules.umm_json"
CMR_DATA_TYPES = ("GET DATA", "GET DATA VIA DIRECT ACCESS", "USE SERVICE API")


def cmr_add_attrs(umm):
    out = {}
    for a in (umm.get("AdditionalAttributes") or []):
        v = a.get("Values") or []
        out[a.get("Name")] = v[0] if len(v) == 1 else v
    return out


def cmr_rings(umm):
    """A UMM-G granule's footprint -> a list of `(lats, lons)` rings."""
    h = ((umm.get("SpatialExtent") or {}).get("HorizontalSpatialDomain")
         or {})
    g = h.get("Geometry") or {}
    rings = []
    for p in (g.get("GPolygons") or []):
        pts = ((p.get("Boundary") or {}).get("Points") or [])
        if len(pts) >= 3:
            rings.append(([float(q["Latitude"]) for q in pts],
                          [float(q["Longitude"]) for q in pts]))
    if rings:
        return rings
    for b in (g.get("BoundingRectangles") or []):
        s, n = float(b["SouthBoundingCoordinate"]), \
            float(b["NorthBoundingCoordinate"])
        w, e = float(b["WestBoundingCoordinate"]), \
            float(b["EastBoundingCoordinate"])
        rings.append(([s, s, n, n], [w, e, e, w]))
    if rings:
        return rings
    raise Refusal(f"granule {umm.get('GranuleUR')!r} publishes no footprint "
                  f"(geometry keys {sorted(g)})")


def cmr_platform(umm):
    """`"<platform>/<instrument>"` — what the `sensor` code table keys on."""
    for p in (umm.get("Platforms") or []):
        ins = p.get("Instruments") or []
        i = (ins[0].get("ShortName") if ins else "") or ""
        return f"{p.get('ShortName') or ''}/{i}"
    return ""


class CmrCatalogue(CatalogueAdapter):
    """A catalogue listed from NASA's CMR — HLS, VIIRS, ECOSTRESS.

    A window is `(t0, t1, collection)` where `collection` is one of
    `COLLECTIONS`: `(key, short_name, version, sensor_hint)`. Each window is
    one CMR query whose rows must equal the `CMR-Hits` header.
    """
    #: ((key, short_name, version_or_None), ...)
    COLLECTIONS = ()
    window_step = "day"
    page_size = CMR_PAGE

    def plan(self, ctx, year, month=None):
        out = []
        for t0, t1 in self.windows(ctx, year, month, step=self.window_step):
            for c in self.COLLECTIONS:
                out.append((t0, t1, c))
        return out

    # -- the query ---------------------------------------------------------
    def cmr_params(self, t0, t1, coll):
        key, short, ver = coll[0], coll[1], coll[2]
        p = [("short_name", short),
             ("temporal", f"{utc_z(t0)},{utc_z(t1)}"),
             ("sort_key", "start_date")]
        if ver:
            p.append(("version", ver))
        return p

    def scenes(self, ctx, window):
        t0, t1, coll = window
        counts = {}
        fet = Fetcher(ctx, self.store)
        what = f"{self.store} {coll[0]} {t0:%F}T{t0:%H}"
        out = []
        for it in cmr_granules(fet, CMR_GRANULES,
                               self.cmr_params(t0, t1, coll), counts, what,
                               page_size=self.page_for(ctx, self.page_size)):
            sc = self.granule(it, coll, counts, what)
            if sc is not None:
                out.append(sc)
        counts["requests"] = counts.get("requests", 0) + fet.requests
        counts["request_retries"] = counts.get("request_retries", 0) + \
            fet.retries
        counts[f"scenes_{coll[0]}"] = counts.get(f"scenes_{coll[0]}", 0) + \
            len(out)
        return out, counts

    # -- one granule -------------------------------------------------------
    def granule(self, item, coll, counts, what):
        umm = item.get("umm") if isinstance(item, dict) else None
        if umm is None:
            raise Refusal(f"{what}: a CMR item with no `umm` block "
                          f"(keys {sorted(item or {})[:8]})")
        gid = umm.get("GranuleUR")
        if not gid:
            raise Refusal(f"{what}: a granule with no GranuleUR")
        tr = ((umm.get("TemporalExtent") or {}).get("RangeDateTime") or {})
        start = tr.get("BeginningDateTime") or \
            (umm.get("TemporalExtent") or {}).get("SingleDateTime")
        if not start:
            raise Refusal(f"{what}: granule {gid} carries no start time")
        t = seconds_of(start)
        lat, lon, area = centre_area(cmr_rings(umm))
        aa = cmr_add_attrs(umm)
        hrefs = [u.get("URL") for u in (umm.get("RelatedUrls") or [])
                 if (u.get("Type") in CMR_DATA_TYPES
                     and str(u.get("URL", "")).startswith("http"))]
        base, tmpl = asset_template(hrefs, self.asset_id(gid, umm), f"{what} "
                                    f"{gid}")
        sens = self.sensor_code(cmr_platform(umm), counts)
        qc = self.qc_code(self.qc_text(gid, coll, umm, aa), counts)
        return Scene(t, lat, lon,
                     self.values(gid, coll, umm, aa, area, sens, counts),
                     gid, coll[1], base, tmpl, qc)

    def asset_id(self, gid, umm):
        """The string the asset file names are built on (usually the UR)."""
        return gid

    def qc_text(self, gid, coll, umm, aa):
        raise NotImplementedError

    def values(self, gid, coll, umm, aa, area, sensor, counts):
        raise NotImplementedError


# ============================================= a CDSE-listed catalogue store ==
CDSE_ODATA = "https://catalogue.dataspace.copernicus.eu/odata/v1/Products"
CDSE_DOWNLOAD = ("https://catalogue.dataspace.copernicus.eu/odata/v1/"
                 "Products({id})/$value")


def odata_attrs(prod):
    return {a.get("Name"): a.get("Value")
            for a in (prod.get("Attributes") or [])}


class OdataCatalogue(CatalogueAdapter):
    """A catalogue listed from the Copernicus Data Space OData API.

    A window is `(t0, t1, product)` where `product` is one of `PRODUCTS`:
    `(key, productType)`. `$skip` is capped at 10,000 by the server, so a
    window is SPLIT until it fits (`odata_windows`) rather than truncated —
    a Sentinel-2 day holds 15,800 products and cannot be paged whole.
    """
    #: ((key, productType), ...)
    PRODUCTS = ()
    collection_name = ""
    window_step = "hour"
    expand = "Attributes"

    def filt(self, t0, t1, product_type):
        return (f"Collection/Name eq '{self.collection_name}' and "
                f"Attributes/OData.CSC.StringAttribute/any(a:a/Name eq "
                f"'productType' and a/OData.CSC.StringAttribute/Value eq "
                f"'{product_type}') and "
                f"ContentDate/Start ge {t0:%Y-%m-%dT%H:%M:%S}.000Z and "
                f"ContentDate/Start lt {t1:%Y-%m-%dT%H:%M:%S}.000Z")

    def plan(self, ctx, year, month=None):
        out = []
        for t0, t1 in self.windows(ctx, year, month, step=self.window_step):
            for p in self.PRODUCTS:
                out.append((t0, t1, p))
        return out

    def scenes(self, ctx, window):
        t0, t1, prod = window
        counts = {}
        fet = Fetcher(ctx, self.store)
        what = f"{self.store} {prod[0]} {t0:%FT%H:%M}"
        out = []
        for lo, hi, n in odata_windows(
                fet, CDSE_ODATA, lambda a, b: self.filt(a, b, prod[1]),
                t0, t1, counts, what):
            if n == 0:
                counts["windows_empty"] = counts.get("windows_empty", 0) + 1
                continue
            for p in odata_window(fet, CDSE_ODATA,
                                  self.filt(lo, hi, prod[1]), self.expand,
                                  counts, what, count=n):
                sc = self.product(p, prod, counts, what)
                if sc is not None:
                    out.append(sc)
        counts["requests"] = counts.get("requests", 0) + fet.requests
        counts["request_retries"] = counts.get("request_retries", 0) + \
            fet.retries
        counts[f"scenes_{prod[0]}"] = counts.get(f"scenes_{prod[0]}", 0) + \
            len(out)
        return out, counts

    def product(self, p, prod, counts, what):
        name = p.get("Name")
        if not name:
            raise Refusal(f"{what}: an OData product with no Name")
        cd = p.get("ContentDate") or {}
        if not cd.get("Start"):
            raise Refusal(f"{what}: product {name} has no ContentDate/Start")
        t = seconds_of(cd["Start"])
        geo = p.get("GeoFootprint")
        if not geo:
            raise Refusal(f"{what}: product {name} has no GeoFootprint")
        lat, lon, area = centre_area(ring_from_geojson(geo))
        aa = odata_attrs(p)
        pid = p.get("Id")
        if not pid:
            raise Refusal(f"{what}: product {name} has no Id")
        # A Sentinel product is ONE archive, fetched whole from one url; the
        # `base_url` is the product endpoint and the template is the single
        # `$value` the producer serves it as. The `S3Path` the catalogue also
        # publishes is recorded in store.json's note, not per row: it is the
        # same path with a fixed prefix.
        base = CDSE_DOWNLOAD.format(id=pid).rsplit("/", 1)[0]
        sens = self.sensor_code(self.sensor_text(name, aa), counts)
        qc = self.qc_code(self.qc_text(name, prod, aa), counts)
        return Scene(t, lat, lon,
                     self.values(name, prod, aa, area, sens, counts),
                     name, prod[1], base, "$value", qc)

    def sensor_text(self, name, aa):
        raise NotImplementedError

    def qc_text(self, name, prod, aa):
        raise NotImplementedError

    def values(self, name, prod, aa, area, sensor, counts):
        raise NotImplementedError


class _Dedup:
    """`AssetWriter.add`'s de-duplication alone — the probe writes no sidecar.

    A probe therefore counts the same rows a build would, which is what makes
    the probe's row count comparable with the producer's own count for that
    month.
    """

    def __init__(self):
        self._seen = set()
        self.n = 0
        self.seq = 0

    def add(self, platform, stac_id, collection, base_url, asset_set):
        if platform in self._seen:
            return False
        self._seen.add(platform)
        self.n += 1
        return True


class Scene:
    """One catalogue row, before packing."""
    __slots__ = ("t", "lat", "lon", "values", "platform", "qc", "stac_id",
                 "collection", "base_url", "asset_set")

    def __init__(self, t, lat, lon, values, stac_id, collection, base_url,
                 asset_set, qc):
        self.t = int(t)
        self.lat = float(lat)
        self.lon = float(lon)
        self.values = list(values)
        self.stac_id = str(stac_id)
        self.collection = str(collection)
        self.base_url = str(base_url)
        self.asset_set = str(asset_set)
        self.qc = int(qc)
        self.platform = f10b.platform_hash(self.stac_id)


# ------------------------------------------------------------------ helpers --
ISO = "%Y-%m-%dT%H:%M:%SZ"


def parse_iso(s):
    """A producer's ISO-8601 instant -> an aware-free UTC datetime.

    Every one of the eight producers writes UTC; the shapes differ (a `Z`, a
    `+00:00`, three or six fractional digits, or none). Anything else is a
    refusal rather than a guess.
    """
    s = str(s).strip()
    m = re.fullmatch(r"(\d{4})-(\d{2})-(\d{2})[T ](\d{2}):(\d{2}):(\d{2})"
                     r"(?:\.(\d{1,9}))?(Z|[+-]00:?00)?", s)
    if not m:
        raise Refusal(f"not a UTC ISO-8601 instant: {s[:40]!r}")
    y, mo, d, h, mi, sec = (int(m.group(i)) for i in range(1, 7))
    return dt.datetime(y, mo, d, h, mi, sec)


def seconds_of(s):
    return f10b.seconds_since_epoch(parse_iso(s))


def month_range(year, month=None, first=None, last=None):
    """[(t0, t1), ...] — one (datetime, datetime) per month, half-open.

    `first` and `last` are `(year, month)` bounds of the producer's record;
    a month outside them is not listed at all.
    """
    months = [int(month)] if month else list(range(1, 13))
    out = []
    for m in months:
        if first and (year, m) < tuple(first):
            continue
        if last and (year, m) > tuple(last):
            continue
        a = dt.datetime(year, m, 1)
        b = dt.datetime(year + (m == 12), m % 12 + 1, 1)
        out.append((a, b))
    return out


def day_range(year, month=None, first=None, last=None):
    """[(t0, t1), ...] — one day per element, half-open, inside the record."""
    out = []
    for a, b in month_range(year, month, first, last):
        d = a
        while d < b:
            out.append((d, d + dt.timedelta(days=1)))
            d += dt.timedelta(days=1)
    return out


def utc_z(t):
    return t.strftime(ISO)


def clamp_pct(x):
    """A producer's percentage -> a float in [0, 100], or NaN.

    A cloud cover of −1 (Landsat's "not computed"), of 255 or of an empty
    string is NOT a percentage and becomes NaN rather than a clipped 0 or 100
    (contract rule 3: never clipped). A value in (100, 100.5] is rounded down
    to 100 — three producers publish 100.000001 — and anything further out is
    left alone so `mask_bounds` counts it.
    """
    if x is None or x == "":
        return float("nan")
    try:
        v = float(x)
    except (TypeError, ValueError):
        return float("nan")
    if not math.isfinite(v) or v < 0:
        return float("nan")
    if 100.0 < v <= 100.5:
        return 100.0
    return v

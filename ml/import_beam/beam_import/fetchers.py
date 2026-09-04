"""Getting the bytes for one work item.

Every fetcher has the same shape:

    fetch_item(item, lane, workdir, note) -> {
        "paths": [local files],
        "bytes": total wire bytes,
        "prov":  {path: {url, bytes, sha256, fetched_at}},
        "missing_dates": [ISO days the archive did not serve],
    }

`lane` is the LaneState for this item's host, and the fetchers call
`lane.run_with_backoff(...)` themselves — pacing, the ladder and Retry-After
apply per REQUEST rather than per item, because a year of OISST is 365
requests and each deserves its own ladder.

**Partial is not failure.** A month or a year whose upstream is missing some
days returns the days it HAS, with the missing dates listed. The caller writes
what arrived and puts the missing days back on the retry queue as day-level
items. Only the two-sighting rule in `sinks.record_not_found` ever turns a
missing day into `absent`. A fetch raises only when it got NOTHING.

Failure vocabulary, from hosts.py:
    TransientError  429/5xx/timeout/short transfer  -> climbs the ladder,
                    then the item goes to the retry queue
    NotFound        the server said 404/410 -> evidence, then `queued` or,
                    on the second sighting six hours later, `absent`
    BlockedError    a gated source with no credentials -> `queued` with a
                    reason, and nothing is requested
"""
from __future__ import annotations

import hashlib
import os
import shutil
import urllib.parse
from typing import Any, Callable, Dict, List, Optional

from .hosts import (BlockedError, LaneState, NotFound, PermanentError,
                    TransientError)
from .manifest import USER_AGENT, days_of_month, days_of_year
from .sinks import utcnow

TRANSIENT_STATUS = {408, 425, 429, 500, 502, 503, 504, 522, 524}
NOT_FOUND_STATUS = {404, 410}


def _note(note: Optional[Callable[[str], None]], msg: str) -> None:
    if note:
        note(msg)


def _sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while True:
            b = fh.read(1 << 20)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


# --------------------------------------------------------------------------
# http (and file:// for the offline tests)
# --------------------------------------------------------------------------
def _is_file_url(url: str) -> bool:
    return url.startswith("file://")


def _file_url_path(url: str) -> str:
    return urllib.parse.unquote(urllib.parse.urlparse(url).path)


def _head(url: str) -> Dict[str, Any]:
    if _is_file_url(url):
        p = _file_url_path(url)
        if not os.path.exists(p):
            return {"status": 404, "length": None, "headers": {}}
        return {"status": 200, "length": os.path.getsize(p), "headers": {}}

    import requests
    try:
        r = requests.head(url, timeout=60, allow_redirects=True,
                          headers={"User-Agent": USER_AGENT})
    except Exception as exc:                                  # noqa: BLE001
        raise TransientError(f"HEAD {url}: {exc}") from exc
    if r.status_code in TRANSIENT_STATUS:
        err = TransientError(f"HEAD {url}: HTTP {r.status_code}")
        err.retry_after = _retry_after(r)                     # type: ignore[attr-defined]
        raise err
    length = r.headers.get("Content-Length")
    return {"status": r.status_code,
            "length": int(length) if length and length.isdigit() else None,
            "headers": {k: v for k, v in list(r.headers.items())[:20]}}


def _retry_after(response) -> Optional[float]:
    val = response.headers.get("Retry-After")
    if not val:
        return None
    try:
        return float(val)
    except ValueError:
        return None


def _stream_to(url: str, dest: str, expect: Optional[int]) -> int:
    """Download to `dest`.part, check the size, then rename into place.

    The `.part` name plus the size check is the whole defence against the
    failure that cost a rebuild: a truncated transfer raised no exception and
    surfaced, much later, as `NetCDF: HDF error`.
    """
    part = dest + ".part"
    os.makedirs(os.path.dirname(os.path.abspath(dest)), exist_ok=True)
    if os.path.exists(part):
        os.remove(part)

    if _is_file_url(url):
        shutil.copyfile(_file_url_path(url), part)
    else:
        import requests
        try:
            with requests.get(url, stream=True, timeout=(60, 600),
                              headers={"User-Agent": USER_AGENT}) as r:
                if r.status_code in TRANSIENT_STATUS:
                    err = TransientError(f"GET {url}: HTTP {r.status_code}")
                    err.retry_after = _retry_after(r)          # type: ignore[attr-defined]
                    raise err
                if r.status_code in NOT_FOUND_STATUS:
                    raise NotFound(f"GET {url}: HTTP {r.status_code}",
                                   status=r.status_code, url=url,
                                   headers=dict(list(r.headers.items())[:20]))
                if r.status_code != 200:
                    raise PermanentError(f"GET {url}: HTTP {r.status_code}")
                with open(part, "wb") as fh:
                    for chunk in r.iter_content(chunk_size=1 << 20):
                        if chunk:
                            fh.write(chunk)
        except (TransientError, NotFound, PermanentError):
            raise
        except Exception as exc:                              # noqa: BLE001
            raise TransientError(f"GET {url}: {exc}") from exc

    got = os.path.getsize(part)
    if expect is not None and got != expect:
        os.remove(part)
        raise TransientError(
            f"short transfer: {url} sent {got} bytes, HEAD said {expect}")
    if got == 0:
        os.remove(part)
        raise TransientError(f"empty file from {url}")
    os.replace(part, dest)
    return got


def download_one(urls: List[str], dest: str, lane: LaneState,
                 note: Optional[Callable[[str], None]] = None
                 ) -> Dict[str, Any]:
    """Download the first URL that exists. Returns {bytes, sha256, url}.

    Raises NotFound (carrying the last response's status and headers, which
    become the absent evidence) when every candidate said 404/410.
    """
    def attempt() -> Dict[str, Any]:
        last_nf: Optional[NotFound] = None
        for url in urls:
            head = _head(url)
            if head["status"] in NOT_FOUND_STATUS:
                last_nf = NotFound(f"HEAD {url}: HTTP {head['status']}",
                                   status=head["status"], url=url,
                                   headers=head.get("headers") or {})
                if lane.head_poll_s > 0:
                    head = _poll_head(url, lane, note)
                    if head["status"] != 200:
                        continue
                else:
                    continue
            if head["status"] != 200:
                raise PermanentError(f"HEAD {url}: HTTP {head['status']}")
            lane.pace()                          # the GET is a second request
            n = _stream_to(url, dest, head["length"])
            lane.note_bytes(n)
            return {"bytes": n, "url": url, "sha256": _sha256_file(dest),
                    "fetched_at": utcnow()}
        if last_nf is not None:
            raise last_nf
        raise PermanentError(f"no usable URL among {urls}")

    return lane.run_with_backoff(attempt, on_note=note)


def _poll_head(url: str, lane: LaneState,
               note: Optional[Callable[[str], None]]) -> Dict[str, Any]:
    """PSL only: HEAD every 60 s for up to `head_poll_s` before giving up."""
    import time
    deadline = time.monotonic() + lane.head_poll_s
    while time.monotonic() < deadline:
        _note(note, f"HEAD-polling {url}")
        time.sleep(60)
        lane.pace()
        head = _head(url)
        if head["status"] == 200:
            return head
    return {"status": 404, "length": None, "headers": {}}


def http_fetch(item, lane, workdir, note=None) -> Dict[str, Any]:
    """One item, one file."""
    dest = os.path.join(workdir, item["filename"])
    got = download_one(item["urls"], dest, lane, note=note)
    return {"paths": [dest], "bytes": got["bytes"], "prov": {dest: got},
            "missing_dates": []}


# --------------------------------------------------------------------------
# per-day products: OISST from NCEI, and anything CMEMS
# --------------------------------------------------------------------------
def _day_urls(item: Dict[str, Any], day: str) -> List[str]:
    ymd = day.replace("-", "")
    ym = ymd[:6]
    urls = [u.replace("{ym}", ym).replace("{ymd}", ymd) for u in item["urls"]]
    if item.get("url_preliminary"):
        urls.append(item["url_preliminary"]
                    .replace("{ym}", ym).replace("{ymd}", ymd))
    return urls


def ncei_oisst_days_fetch(item, lane, workdir, note=None) -> Dict[str, Any]:
    """Every day of one year (or the single day of a day-level item).

    A day the archive does not serve is RECORDED, not fatal: the caller
    writes the days that arrived and re-queues the rest. This is the rule
    that replaced the old "a middle gap is fatal" — a gap is now a fact with
    a date attached, and it is chased until the archive answers twice.
    """
    days = _days_of(item)
    daydir = os.path.join(workdir, "days")
    os.makedirs(daydir, exist_ok=True)
    got, missing, prov, total = [], [], {}, 0
    for day in days:
        dest = os.path.join(daydir, f"oisst.{day.replace('-', '')}.nc")
        if os.path.exists(dest) and os.path.getsize(dest) > 0:
            got.append(dest)
            prov[dest] = {"url": _day_urls(item, day)[0],
                          "bytes": os.path.getsize(dest),
                          "sha256": _sha256_file(dest),
                          "fetched_at": utcnow()}
            continue
        try:
            meta = download_one(_day_urls(item, day), dest, lane, note=note)
            total += meta["bytes"]
            prov[dest] = meta
            got.append(dest)
        except (NotFound, TransientError, PermanentError) as exc:
            missing.append(day)
            _note(note, f"{item['item_id']}: {day} not served ({exc})")
    if not got:
        raise TransientError(
            f"{item['item_id']}: no day was served ({len(missing)} missing)")
    return {"paths": sorted(got), "bytes": total, "prov": prov,
            "missing_dates": missing}


def _days_of(item: Dict[str, Any]) -> List[str]:
    """The days one item covers: a single day, a month, or a year."""
    if item.get("day"):
        return [item["day"]]
    if item.get("days"):
        return list(item["days"])
    if item.get("month"):
        return days_of_month(item["month"])
    return days_of_year(int(item["year"]), item.get("start_month"))


def cmems_fetch(item, lane, workdir, note=None) -> Dict[str, Any]:
    """One CMEMS subset request PER DAY.

    One day per request is not a preference: a whole month in one request is
    5.95 GB of RAM and is OOM-killed on a 7 GB machine (DESIGN §2).
    """
    if not (os.environ.get("COPERNICUSMARINE_SERVICE_USERNAME")
            and os.environ.get("COPERNICUSMARINE_SERVICE_PASSWORD")):
        raise BlockedError(
            "CMEMS needs COPERNICUSMARINE_SERVICE_USERNAME and "
            "COPERNICUSMARINE_SERVICE_PASSWORD in the environment. See "
            "CREDENTIALS.md — and quote the password, it contains a '%'.")
    try:
        import copernicusmarine
    except ImportError as exc:                                # pragma: no cover
        raise BlockedError("the `copernicusmarine` package is not installed; "
                           "pip install copernicusmarine") from exc

    bbox = item.get("bbox") or {}
    depth = item.get("depth") or {}
    daydir = os.path.join(workdir, "days")
    os.makedirs(daydir, exist_ok=True)
    got, missing, prov, total = [], [], {}, 0

    for day in _days_of(item):
        out = f"{item['source']}_{day.replace('-', '')}.nc"
        dest = os.path.join(daydir, out)
        if os.path.exists(dest) and os.path.getsize(dest) > 0:
            got.append(dest)
            prov[dest] = _cmems_prov(item, dest)
            continue

        def one_day(day=day, out=out, dest=dest):
            kwargs: Dict[str, Any] = dict(
                dataset_id=item["dataset_id"],
                variables=list(item.get("variables") or []),
                output_directory=daydir, output_filename=out,
                overwrite=True, disable_progress_bar=True)
            if bbox:
                kwargs.update(minimum_longitude=bbox["lon_min"],
                              maximum_longitude=bbox["lon_max"],
                              minimum_latitude=bbox["lat_min"],
                              maximum_latitude=bbox["lat_max"])
            if depth:
                kwargs.update(minimum_depth=depth.get("min"),
                              maximum_depth=depth.get("max"))
            kwargs.update(start_datetime=f"{day}T00:00:00",
                          end_datetime=f"{day}T23:59:59")
            try:
                copernicusmarine.subset(**kwargs)
            except Exception as exc:                          # noqa: BLE001
                text = str(exc).lower()
                if ("credential" in text or "unauthor" in text
                        or "login" in text):
                    raise PermanentError(f"CMEMS auth: {exc}") from exc
                raise TransientError(f"CMEMS subset {day}: {exc}") from exc
            if not os.path.exists(dest) or os.path.getsize(dest) == 0:
                raise TransientError(f"CMEMS subset {day}: no output file")
            return os.path.getsize(dest)

        try:
            n = lane.run_with_backoff(one_day, on_note=note)
            lane.note_bytes(n)
            total += n
            got.append(dest)
            prov[dest] = _cmems_prov(item, dest)
        except (TransientError, NotFound) as exc:
            missing.append(day)
            _note(note, f"{item['item_id']}: {day} not served ({exc})")

    if not got:
        raise TransientError(
            f"{item['item_id']}: CMEMS served no day "
            f"({len(missing)} missing)")
    return {"paths": sorted(got), "bytes": total, "prov": prov,
            "missing_dates": missing}


def _cmems_prov(item, dest) -> Dict[str, Any]:
    return {"url": f"cmems://{item.get('dataset_id', '')}",
            "bytes": os.path.getsize(dest), "sha256": _sha256_file(dest),
            "fetched_at": utcnow()}


def psl_fallback_fetch(item, lane, workdir, note=None) -> Dict[str, Any]:
    """One PSL yearly OISST file (sst.day.mean.YYYY.nc / icec...).

    NOT automatic. Reached only through the `oisst_psl` source, which is
    `enabled: false` and lives on the `psl` HOST, so running it uses PSL's own
    single lane and 20-second gap. Falling back automatically from inside an
    NCEI lane would open a second stream on PSL without PSL's budget agreeing
    to it, which DESIGN §2 forbids.
    """
    return http_fetch(item, lane, workdir, note=note)


def cds_fetch(item, lane, workdir, note=None) -> Dict[str, Any]:
    url, key = os.environ.get("CDSAPI_URL"), os.environ.get("CDSAPI_KEY")
    if not (url and key):
        raise BlockedError(
            "ERA5 is blocked: no CDS (Copernicus Climate Data Store) "
            "credentials. Set CDSAPI_URL and CDSAPI_KEY in the environment. "
            "The free account can only be created by Chris — see "
            "README_FOR_GEMINI.md §7. Nothing was requested; the item stays "
            "in the queue.")
    try:
        import cdsapi
    except ImportError as exc:                                # pragma: no cover
        raise BlockedError("the `cdsapi` package is not installed; "
                           "pip install cdsapi") from exc

    dest = os.path.join(workdir, item["filename"])
    year, mon = item["month"].split("-")

    def attempt() -> int:
        client = cdsapi.Client(url=url, key=key, quiet=True)
        request = {
            "product_type": "reanalysis", "variable": item["var"],
            "year": year, "month": mon,
            "day": [f"{d:02d}" for d in range(1, 32)],
            "daily_statistic": "daily_mean", "time_zone": "utc+00:00",
            "frequency": "1_hourly", "format": "netcdf",
        }
        try:
            client.retrieve(item["dataset_id"], request, dest)
        except Exception as exc:                              # noqa: BLE001
            raise TransientError(
                f"CDS retrieve {item['item_id']}: {exc}") from exc
        if not os.path.exists(dest) or os.path.getsize(dest) == 0:
            raise TransientError(f"CDS retrieve {item['item_id']}: empty")
        return os.path.getsize(dest)

    n = lane.run_with_backoff(attempt, on_note=note)
    lane.note_bytes(n)
    return {"paths": [dest], "bytes": n,
            "prov": {dest: {"url": f"cds://{item['dataset_id']}", "bytes": n,
                            "sha256": _sha256_file(dest),
                            "fetched_at": utcnow()}},
            "missing_dates": []}


def transient_test_fetch(item, lane, workdir, note=None) -> Dict[str, Any]:
    """Always fails, transiently. Used by the tests to trip the breaker and
    prove the remaining items reach the retry queue. Never a real source."""
    def attempt():
        raise TransientError("simulated transient failure (transient_test)")
    return lane.run_with_backoff(attempt, on_note=note)


FETCHERS: Dict[str, Callable[..., Dict[str, Any]]] = {
    "http": http_fetch,
    "cmems": cmems_fetch,
    "cds": cds_fetch,
    "ncei_oisst_days": ncei_oisst_days_fetch,
    "psl_fallback": psl_fallback_fetch,
    "transient_test": transient_test_fetch,
}


def fetch_item(item: Dict[str, Any], lane: LaneState, workdir: str,
               note=None) -> Dict[str, Any]:
    """Dispatch to the fetcher this item's source names.

    There is deliberately no automatic cross-HOST fallback: a fetcher only
    ever uses the LaneState of its own item's host.
    """
    fn = FETCHERS[item["fetcher"]]
    os.makedirs(workdir, exist_ok=True)
    out = fn(item, lane, workdir, note=note)
    out.setdefault("prov", {})
    out.setdefault("missing_dates", [])
    return out

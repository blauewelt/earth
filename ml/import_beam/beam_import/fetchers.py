"""Getting the bytes for one work item.

Every fetcher has the same shape:

    fetch_item(item, lane, workdir, note) -> {"paths": [...], "bytes": N}

`paths` are LOCAL files in `workdir`; transforms.py turns them into the one
file that gets published. `lane` is the LaneState for this item's host — the
fetchers call `lane.run_with_backoff(...)` themselves, so pacing, the backoff
ladder and Retry-After all apply per REQUEST rather than per item (a year of
OISST is 365 requests and each of them deserves its own ladder).

Failure vocabulary, from hosts.py:
    TransientError  429/5xx/timeout/short transfer  -> climbs the ladder
    AbsentError     the archive genuinely has no such file -> `absent`
    PermanentError  a 404 where we were sure         -> `failed`
    BlockedError    a gated source with no credentials -> `blocked`
"""
from __future__ import annotations

import os
import shutil
import urllib.parse
from typing import Any, Callable, Dict, List, Optional

from .hosts import (AbsentError, BlockedError, LaneState, PermanentError,
                    TransientError)
from .manifest import USER_AGENT, days_of_month, days_of_year

# Statuses that mean "try again later" rather than "this does not exist".
TRANSIENT_STATUS = {408, 425, 429, 500, 502, 503, 504, 522, 524}


def _note(note: Optional[Callable[[str], None]], msg: str) -> None:
    if note:
        note(msg)


# --------------------------------------------------------------------------
# http (and file:// for the offline tests)
# --------------------------------------------------------------------------
def _is_file_url(url: str) -> bool:
    return url.startswith("file://")


def _file_url_path(url: str) -> str:
    return urllib.parse.unquote(urllib.parse.urlparse(url).path)


def _head(url: str) -> Dict[str, Any]:
    """HEAD one URL. Returns {'status', 'length'} or raises TransientError."""
    if _is_file_url(url):
        p = _file_url_path(url)
        if not os.path.exists(p):
            return {"status": 404, "length": None}
        return {"status": 200, "length": os.path.getsize(p)}

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
            "length": int(length) if length and length.isdigit() else None}


def _retry_after(response) -> Optional[float]:
    """The server's own Retry-After, in seconds, when it sent one."""
    val = response.headers.get("Retry-After")
    if not val:
        return None
    try:
        return float(val)
    except ValueError:
        return None                # an HTTP-date; our ladder is good enough


def _stream_to(url: str, dest: str, expect: Optional[int]) -> int:
    """Download to `dest`.part, check the size, then rename into place.

    The `.part` name plus the size check is the whole defence against the
    failure that cost us a rebuild: a truncated transfer raised no exception
    and only surfaced, much later, as `NetCDF: HDF error`.
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
                if r.status_code == 404:
                    raise AbsentError(f"GET {url}: HTTP 404")
                if r.status_code != 200:
                    raise PermanentError(f"GET {url}: HTTP {r.status_code}")
                with open(part, "wb") as fh:
                    for chunk in r.iter_content(chunk_size=1 << 20):
                        if chunk:
                            fh.write(chunk)
        except (TransientError, AbsentError, PermanentError):
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
                 absent_ok: bool = False,
                 note: Optional[Callable[[str], None]] = None) -> int:
    """Download the first URL in `urls` that exists, with the lane's ladder.

    Returns the number of bytes written. Raises AbsentError when every
    candidate answered 404 and `absent_ok`; PermanentError when it did not.
    """
    def attempt() -> int:
        last_status = None
        for url in urls:
            head = _head(url)                    # paced by run_with_backoff
            if head["status"] == 404:
                last_status = 404
                # PSL answers 5xx under load and 404 while a file is being
                # replaced; on that host a HEAD-poll is cheaper than giving up.
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
            return n
        if last_status == 404:
            if absent_ok:
                raise AbsentError(f"not published: {urls[0]}")
            raise PermanentError(f"404 on every candidate URL: {urls}")
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
    return {"status": 404, "length": None}


def http_fetch(item: Dict[str, Any], lane: LaneState, workdir: str,
               note=None) -> Dict[str, Any]:
    """One item, one file."""
    dest = os.path.join(workdir, item["filename"])
    n = download_one(item["urls"], dest, lane,
                     absent_ok=bool(item.get("absent_ok")), note=note)
    return {"paths": [dest], "bytes": n}


# --------------------------------------------------------------------------
# OISST: 365 per-day NCEI files -> one year
# --------------------------------------------------------------------------
def ncei_oisst_year_fetch(item: Dict[str, Any], lane: LaneState, workdir: str,
                          note=None) -> Dict[str, Any]:
    """Every day of one year from NCEI, into workdir/days/.

    A day that is missing at the END of the record is legal (the year is not
    finished). A day missing in the MIDDLE is fatal — a year with a hole in it
    is not the file the tensor build expects, and silently shipping one is how
    a gap becomes a permanent, invisible feature of the training data.
    """
    year = int(item["year"])
    days = item.get("days") or days_of_year(year, item.get("start_month"))
    daydir = os.path.join(workdir, "days")
    os.makedirs(daydir, exist_ok=True)

    got: List[str] = []
    missing: List[str] = []
    total = 0
    for day in days:
        ymd = day.replace("-", "")
        ym = ymd[:6]
        urls = [u.replace("{ym}", ym).replace("{ymd}", ymd)
                for u in item["urls"]]
        if item.get("url_preliminary"):
            urls.append(item["url_preliminary"]
                        .replace("{ym}", ym).replace("{ymd}", ymd))
        dest = os.path.join(daydir, f"oisst.{ymd}.nc")
        if os.path.exists(dest) and os.path.getsize(dest) > 0:
            got.append(dest)                     # resumed from a previous run
            continue
        try:
            total += download_one(urls, dest, lane, absent_ok=True, note=note)
            got.append(dest)
        except AbsentError:
            missing.append(day)

    if missing:
        trailing = _trailing(days, missing)
        middle = [d for d in missing if d not in trailing]
        if middle:
            raise PermanentError(
                f"OISST {year}: {len(middle)} day(s) missing in the MIDDLE of "
                f"the year, first {middle[0]} — a middle gap is fatal")
        _note(note, f"OISST {year}: {len(missing)} missing day(s) at the tail")
    if not got:
        raise AbsentError(f"OISST {year}: no days served at all")
    return {"paths": sorted(got), "bytes": total, "missing_days": missing}


def _trailing(days: List[str], missing: List[str]) -> set:
    """The maximal run of missing days at the END of the year."""
    tail = set()
    miss = set(missing)
    for d in reversed(days):
        if d in miss:
            tail.add(d)
        else:
            break
    return tail


def psl_fallback_fetch(item: Dict[str, Any], lane: LaneState, workdir: str,
                       note=None) -> Dict[str, Any]:
    """One PSL yearly OISST file (sst.day.mean.YYYY.nc / icec.day.mean.YYYY.nc).

    THIS IS NOT AUTOMATIC. It is reached only through the `oisst_psl` source,
    which is `enabled: false` in the registry and lives on the `psl` HOST, so
    running it uses PSL's own single lane and its 20 s gap. Falling back
    automatically from inside an NCEI lane would have put a second stream on
    PSL without PSL's budget agreeing to it, and DESIGN §2 is explicit that a
    source may never create a connection budget by accident.

    PSL is why the whole backoff ladder exists: two 477 MB files back to back
    is what made it answer 504 and then go quiet for a quarter of an hour.
    """
    dest = os.path.join(workdir, item["filename"])
    n = download_one(item["urls"], dest, lane, absent_ok=True, note=note)
    return {"paths": [dest], "bytes": n, "via": "psl_fallback"}


# --------------------------------------------------------------------------
# CMEMS (Copernicus Marine)
# --------------------------------------------------------------------------
def cmems_fetch(item: Dict[str, Any], lane: LaneState, workdir: str,
                note=None) -> Dict[str, Any]:
    """One subset request PER DAY, into workdir/days/.

    One day per request is not a preference. A whole month in one request is
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
    days = days_of_month(item["month"]) if item.get("month") else [None]

    paths, total = [], 0
    for day in days:
        out = f"{item['source']}_{(day or 'static').replace('-', '')}.nc"
        dest = os.path.join(daydir, out)
        if os.path.exists(dest) and os.path.getsize(dest) > 0:
            paths.append(dest)
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
            if day:
                kwargs.update(start_datetime=f"{day}T00:00:00",
                              end_datetime=f"{day}T23:59:59")
            try:
                copernicusmarine.subset(**kwargs)
            except Exception as exc:                          # noqa: BLE001
                # The toolbox raises one exception type for everything, so a
                # bad password and a flaky object store look identical. Treat
                # an auth message as permanent and everything else as
                # transient; retrying a wrong password is not politeness.
                text = str(exc).lower()
                if "credential" in text or "unauthor" in text or "login" in text:
                    raise PermanentError(f"CMEMS auth: {exc}") from exc
                raise TransientError(f"CMEMS subset {day}: {exc}") from exc
            if not os.path.exists(dest) or os.path.getsize(dest) == 0:
                raise TransientError(f"CMEMS subset {day}: no output file")
            return os.path.getsize(dest)

        n = lane.run_with_backoff(one_day, on_note=note)
        lane.note_bytes(n)
        total += n
        paths.append(dest)

    return {"paths": sorted(paths), "bytes": total}


# --------------------------------------------------------------------------
# CDS (ERA5) — refuses politely until the account exists
# --------------------------------------------------------------------------
def cds_fetch(item: Dict[str, Any], lane: LaneState, workdir: str,
              note=None) -> Dict[str, Any]:
    url, key = os.environ.get("CDSAPI_URL"), os.environ.get("CDSAPI_KEY")
    if not (url and key):
        raise BlockedError(
            "ERA5 is blocked: no CDS (Copernicus Climate Data Store) "
            "credentials. Set CDSAPI_URL and CDSAPI_KEY in the environment. "
            "The free account can only be created by Chris — see "
            "README_FOR_GEMINI.md §7. Nothing was requested.")
    try:
        import cdsapi
    except ImportError as exc:                                # pragma: no cover
        raise BlockedError("the `cdsapi` package is not installed; "
                           "pip install cdsapi") from exc

    dest = os.path.join(workdir, item["filename"])
    month = item["month"]
    year, mon = month.split("-")

    def attempt() -> int:
        client = cdsapi.Client(url=url, key=key, quiet=True)
        request = {
            "product_type": "reanalysis",
            "variable": item["var"],
            "year": year,
            "month": mon,
            "day": [f"{d:02d}" for d in range(1, 32)],
            "daily_statistic": "daily_mean",
            "time_zone": "utc+00:00",
            "frequency": "1_hourly",
            "format": "netcdf",
        }
        try:
            client.retrieve(item["dataset_id"], request, dest)
        except Exception as exc:                              # noqa: BLE001
            raise TransientError(f"CDS retrieve {item['item_id']}: {exc}") from exc
        if not os.path.exists(dest) or os.path.getsize(dest) == 0:
            raise TransientError(f"CDS retrieve {item['item_id']}: empty")
        return os.path.getsize(dest)

    n = lane.run_with_backoff(attempt, on_note=note)
    lane.note_bytes(n)
    return {"paths": [dest], "bytes": n}


# --------------------------------------------------------------------------
# test-only
# --------------------------------------------------------------------------
def transient_test_fetch(item: Dict[str, Any], lane: LaneState, workdir: str,
                         note=None) -> Dict[str, Any]:
    """Always fails, transiently. Used by the smoke test to trip the breaker
    and prove the pipeline finishes anyway. Never used by a real source."""
    def attempt():
        raise TransientError("simulated transient failure (transient_test)")
    return lane.run_with_backoff(attempt, on_note=note)


FETCHERS: Dict[str, Callable[..., Dict[str, Any]]] = {
    "http": http_fetch,
    "cmems": cmems_fetch,
    "cds": cds_fetch,
    "ncei_oisst_year": ncei_oisst_year_fetch,
    "psl_fallback": psl_fallback_fetch,
    "transient_test": transient_test_fetch,
}


def fetch_item(item: Dict[str, Any], lane: LaneState, workdir: str,
               note=None) -> Dict[str, Any]:
    """Dispatch to the fetcher this item's source names.

    There is deliberately no automatic cross-HOST fallback here: a fetcher
    only ever uses the LaneState of its own item's host.
    """
    fn = FETCHERS[item["fetcher"]]
    os.makedirs(workdir, exist_ok=True)
    return fn(item, lane, workdir, note=note)

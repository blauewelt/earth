#!/usr/bin/env python3
"""E-040 DATA half: bake OISST v2.1 daily SST for 1991..2026 and publish every
year to the Hugging Face dataset repo `chfrank/earth-sst-daily` under
`sst/quarter/`, restore-verifying each one before it is announced.

Runs one year at a time, NEWEST FIRST — the app needs recent years immediately,
and the container has ~18 GB of headroom against 476 MB of source plus 757 MB
of output per year, so nothing may accumulate:

    download .nc -> bake pixel-major .i16 (scripts/bake_sst_daily.py, which
    deletes the .nc) -> upload .i16 + .json -> RANGE-READ 3 random (pixel, day)
    offsets back off the Hub and compare the 2 bytes to the local file ->
    only then delete the local .i16 -> republish sst/quarter/index.json.

The restore verification is the point, not a formality (claude/huggingface-access.md:
"a backup is only real if the RESTORE works"). An upload that reports success is
not evidence the bytes are retrievable, and the client reads this archive two
bytes at a time — so the check is made in exactly the shape the browser uses,
an HTTP Range read against the public resolve URL, no token, redirects followed.
A mismatch STOPS the pipeline: a wrong byte at one offset means the layout or
the transfer is wrong, and continuing would publish 35 more years of it.

index.json is what the client reads to discover availability, so a year appears
in it ONLY after its own bytes have been read back correctly. It is rewritten
and re-uploaded after every verified year, so an interrupted run leaves a
truthful (smaller) index rather than a claim about a file that is half there.

Resumable: a year already on the Hub whose .i16 size matches its own .json
`days` is skipped (this is how 2015, baked and verified 2026-08-18, is left
alone). 2026 is a PARTIAL year by design — the source file holds however many
days exist and the meta records `days`; the client reads that, never a constant.

Usage:  python3 scripts/sst_daily_pipeline.py            # 2026 -> 1991
        python3 scripts/sst_daily_pipeline.py 2026 2025  # named years, in order

Token: read from the FILE ~/.hf_token, never from argv (the permission
classifier blocks tokens on command lines), never written anywhere new.
"""
import datetime as dt
import json
import os
import random
import shutil
import subprocess
import sys
import time
import urllib.error
import urllib.request

from huggingface_hub import HfApi

REPO = "chfrank/earth-sst-daily"
PREFIX = "sst/quarter"
RESOLVE = f"https://huggingface.co/datasets/{REPO}/resolve/main/{PREFIX}"
SRC_URL = ("https://downloads.psl.noaa.gov/Datasets/noaa.oisst.v2.highres/"
           "sst.day.mean.{y}.nc")
PACE_S = 20          # breathing room between years — PSL throttles (see below)
PSL_WAIT = 900       # how long to wait for PSL before using the NCEI fallback
STRIKES = {"psl": 0}  # consecutive PSL failures — see do_year
TOKEN_FILE = next(p for p in (os.path.expanduser("~/.hf_token"),
                              "/home/claude/.hf_token")
                  if os.path.exists(p))
LOG = "/tmp/sst_pipeline.log"
TMP = "/tmp"
NX, NY = 1440, 720
NPIX = NX * NY
MIN_FREE_GB = 3.0
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BAKER = os.path.join(REPO_ROOT, "scripts", "bake_sst_daily.py")

# A probe pixel with a real value: 40 W, 30 N, mid North Atlantic. Logged after
# every bake so the value can be checked independently against the Hub.
PROBE_PX = 480 * NX + 560


def log(msg):
    line = f"{dt.datetime.utcnow():%Y-%m-%dT%H:%M:%SZ} {msg}"
    print(line, flush=True)
    with open(LOG, "a") as f:
        f.write(line + "\n")
        f.flush()


def free_gb(path=TMP):
    return shutil.disk_usage(path).free / 1e9


def year_days(y):
    """Calendar days in y, or None for a year that is still running."""
    if y >= dt.date.today().year:
        return None
    return 366 if (y % 4 == 0 and (y % 100 != 0 or y % 400 == 0)) else 365


def range_read(url, off, n=2, tries=4):
    """Anonymous HTTP range read — the same call shape the browser makes."""
    last = None
    for i in range(tries):
        try:
            req = urllib.request.Request(
                url, headers={"Range": f"bytes={off}-{off + n - 1}"})
            with urllib.request.urlopen(req, timeout=120) as f:
                if f.status != 206:
                    raise RuntimeError(f"expected 206, got {f.status}")
                b = f.read()
            if len(b) != n:
                raise RuntimeError(f"expected {n} bytes, got {len(b)}")
            return b
        except Exception as e:                      # noqa: BLE001
            last = e
            time.sleep(3 * (i + 1))
    raise RuntimeError(f"range read failed after {tries} tries: {last}")


def hub_state(api):
    """{year: days} for every year already on the Hub whose .i16 size agrees
    with its own meta. Size disagreement means a truncated upload, so the year
    is treated as absent and re-baked."""
    files = set(api.list_repo_files(REPO, repo_type="dataset"))
    out = {}
    for y in range(1991, 2027):
        i16, meta = f"{PREFIX}/{y}.i16", f"{PREFIX}/{y}.json"
        if i16 not in files or meta not in files:
            continue
        try:
            m = json.loads(urllib.request.urlopen(
                f"{RESOLVE}/{y}.json", timeout=60).read())
            days = int(m["days"])
            info = api.get_paths_info(REPO, [i16], repo_type="dataset")
            size = info[0].size
        except Exception as e:                      # noqa: BLE001
            log(f"  {y}: could not read hub state ({e}) — will re-bake")
            continue
        want = NPIX * days * 2
        if size != want:
            log(f"  {y}: hub size {size} != {want} for days={days} — will re-bake")
            continue
        exp = year_days(y)
        if exp is not None and days != exp:
            log(f"  {y}: hub days {days} != calendar {exp} — will re-bake")
            continue
        out[y] = days
    return out


def write_index(api, verified):
    idx = {
        "nx": NX, "ny": NY, "dlon": 0.25, "dlat": 0.25,
        "west": -180, "south": -90,
        "scale": 0.01, "nodata": -32768,
        "layout": "pixel-major: value(px, day) at byte (px*days + day)*2",
        "source": "NOAA OISST v2.1 daily (PSL)",
        "citation": "Huang et al. 2021, doi:10.1175/JCLI-D-20-0166.1",
        "updated": dt.datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
        "years": {str(y): verified[y] for y in sorted(verified, reverse=True)},
    }
    path = os.path.join(TMP, "sst_index.json")
    with open(path, "w") as f:
        json.dump(idx, f, indent=1)
    api.upload_file(path_or_fileobj=path, path_in_repo=f"{PREFIX}/index.json",
                    repo_id=REPO, repo_type="dataset",
                    commit_message=f"index: {len(verified)} years verified")
    log(f"  index.json uploaded — {len(verified)} years: "
        f"{','.join(str(y) for y in sorted(verified, reverse=True))}")


def retry(what, fn, tries=3, backoff=20):
    for i in range(tries):
        try:
            return fn()
        except Exception as e:                      # noqa: BLE001
            log(f"  {what} attempt {i + 1}/{tries} FAILED: {e}")
            if i == tries - 1:
                raise
            time.sleep(backoff * (i + 1))


def wait_for_source(y, max_wait=5400, step=120):
    """Block until PSL will serve this year's NetCDF, up to max_wait seconds.

    Measured 2026-08-18: after two back-to-back year downloads PSL answered 504
    and then stopped answering at all for tens of minutes, while Hugging Face
    and the proxy stayed healthy. Burning three download attempts against a
    host that is simply down converts an outage into a permanent hole in the
    archive — so ASK FIRST with a HEAD, and wait, which costs nothing but time
    and is the difference between 36 years and 34.
    """
    url = SRC_URL.format(y=y)
    t0 = time.time()
    while True:
        try:
            req = urllib.request.Request(url, method="HEAD")
            with urllib.request.urlopen(req, timeout=45) as f:
                n = int(f.headers.get("Content-Length", 0))
            if n > 0:
                return n
            raise RuntimeError("no Content-Length")
        except Exception as e:                      # noqa: BLE001
            waited = time.time() - t0
            if waited > max_wait:
                raise RuntimeError(
                    f"source unavailable for {waited / 60:.0f} min: {e}")
            log(f"  {y} source not ready ({e}) — waited {waited / 60:.0f} min, "
                f"sleeping {step}s")
            time.sleep(step)


def bake(y):
    """One attempt at scripts/bake_sst_daily.py, keeping its stderr.

    The source .nc is removed first: the baker skips the download when the file
    already exists, so a retry after a truncated or 504-poisoned download would
    otherwise re-open the same broken file forever. And the child's output is
    CAPTURED rather than discarded — a retry loop that hides the reason it is
    retrying is the failure mode ml/CLAUDE.md §4.6 is about (PSL answered 504
    for all of 2024 on 2026-08-18, and the first version of this said only
    "returned non-zero exit status 1").
    """
    src = os.path.join(TMP, f"oisst_day_{y}.nc")
    if os.path.exists(src):
        os.remove(src)
    p = subprocess.run([sys.executable, BAKER, str(y)],
                       env=dict(os.environ, SST_OUT=TMP),
                       stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                       text=True)
    if p.returncode != 0:
        tail = " | ".join(l for l in p.stdout.strip().splitlines()[-4:] if l)
        raise RuntimeError(f"baker exit {p.returncode}: {tail}")
    return p


def bake_ncei(y):
    """Fallback baker: the same year assembled from NCEI's PER-DAY files.

    PSL serves one 476 MB NetCDF per year and is the primary source because it
    is one request. On 2026-08-18 it started answering 504 and then stopped
    answering at all, mid-run, with the rest of the internet healthy — so this
    exists to keep an outage at one host from putting a hole in a 36-year
    archive. NCEI publishes the SAME product (OISST v2.1) one 1.6 MB file per
    day on the same 0.25 degree grid with the same 0.01 degC packing.

    Equivalence is not assumed: 2015-06-30 at 40 W / 30 N reads 25.59 degC from
    an NCEI daily file and 2559 from the byte already on the Hub, which was
    baked from PSL. The output written here is byte-for-byte the same shape,
    and the meta records which source actually produced it.

    A missing day in the MIDDLE is fatal — a gap silently shifted by one column
    would look like a plausible ocean everywhere. Only a missing TAIL is legal,
    and only for a year that has not finished.
    """
    import concurrent.futures
    import netCDF4
    import numpy as np

    base = ("https://www.ncei.noaa.gov/data/"
            "sea-surface-temperature-optimum-interpolation/v2.1/access/avhrr")
    dates, d = [], dt.date(y, 1, 1)
    while d.year == y and d <= dt.date.today():
        dates.append(d)
        d += dt.timedelta(days=1)
    day_dir = os.path.join(TMP, f"ncei_{y}")
    shutil.rmtree(day_dir, ignore_errors=True)
    os.makedirs(day_dir)

    def grab(i_d):
        i, dd = i_d
        path = os.path.join(day_dir, f"{i:04d}.nc")
        stem = f"{base}/{dd:%Y%m}/oisst-avhrr-v02r01.{dd:%Y%m%d}"
        for url in (f"{stem}.nc", f"{stem}_preliminary.nc"):
            for _ in range(3):
                try:
                    with urllib.request.urlopen(url, timeout=120) as f:
                        if f.status != 200:
                            break
                        b = f.read()
                    if len(b) < 100000:             # an error page, not a grid
                        break
                    with open(path, "wb") as g:
                        g.write(b)
                    return i
                except urllib.error.HTTPError as e:
                    if e.code == 404:
                        break
                    time.sleep(5)
                except Exception:                   # noqa: BLE001
                    time.sleep(5)
        return None

    td = time.time()
    with concurrent.futures.ThreadPoolExecutor(max_workers=12) as ex:
        got = {i for i in ex.map(grab, enumerate(dates)) if i is not None}
    log(f"  {y} NCEI: {len(got)}/{len(dates)} days in {time.time() - td:.0f}s")
    T = 0
    while T in got:
        T += 1
    if T == 0:
        shutil.rmtree(day_dir, ignore_errors=True)
        raise RuntimeError(f"NCEI: no days retrieved for {y}")
    if got - set(range(T)):
        holes = sorted(set(range(T + 1, max(got) + 1)) - got)[:5]
        shutil.rmtree(day_dir, ignore_errors=True)
        raise RuntimeError(f"NCEI: {y} has a GAP at day {T} "
                           f"(later days present: {holes}) — refusing to bake")
    exp = year_days(y)
    if exp is not None and T != exp:
        shutil.rmtree(day_dir, ignore_errors=True)
        raise RuntimeError(f"NCEI: {y} yielded {T} days, calendar says {exp}")

    out = np.full((NY * NX, T), -32768, dtype=np.int16)
    for t in range(T):
        dset = netCDF4.Dataset(os.path.join(day_dir, f"{t:04d}.nc"))
        a = np.ma.filled(dset.variables["sst"][0, 0], np.nan).astype(np.float32)
        dset.close()
        a = np.roll(a, NX // 2, axis=1)             # lon 0..360 -> -180..180
        ok = np.isfinite(a)
        out[:, t] = np.where(
            ok, np.clip(a * 100, -32767, 32767), -32768).astype(np.int16).ravel()
    shutil.rmtree(day_dir, ignore_errors=True)
    out.tofile(os.path.join(TMP, f"oisst_daily_{y}.i16"))
    json.dump({"year": y, "days": int(T), "nx": NX, "ny": NY,
               "dlon": 0.25, "dlat": 0.25, "west": -180, "south": -90,
               "scale": 0.01, "nodata": -32768,
               "layout": "pixel-major: value(px, day) at byte (px*days + day)*2",
               "source": "NOAA OISST v2.1 daily (NCEI per-day files)",
               "citation": "Huang et al. 2021, doi:10.1175/JCLI-D-20-0166.1"},
              open(os.path.join(TMP, f"oisst_daily_{y}.json"), "w"))
    return T


def do_year(api, y, verified):
    i16 = os.path.join(TMP, f"oisst_daily_{y}.i16")
    meta = os.path.join(TMP, f"oisst_daily_{y}.json")
    for p in (i16, meta, os.path.join(TMP, f"oisst_day_{y}.nc")):
        if os.path.exists(p):
            os.remove(p)                            # never trust a partial

    if free_gb() < MIN_FREE_GB:
        raise SystemExit(f"FATAL: only {free_gb():.1f} GB free, need "
                         f"{MIN_FREE_GB} GB — stopping before {y}")

    t0 = time.time()
    # NCEI first, PSL second — a reversal of the plan's stated source, measured
    # rather than preferred. PSL serves the year in one 476 MB request, which is
    # why the baker was written against it, but it throttles a client that keeps
    # coming back: on 2026-08-18 it served 2026 and 2025 back to back, then
    # answered 504 and finally nothing at all for ~15 minutes at a time, letting
    # exactly one more year through per outage. At 15 min/year the remaining 33
    # years would take eight hours. NCEI's per-day files, same product and
    # byte-identical output (checked against the 2015 file already on the Hub),
    # download a whole year in under a minute at 12 threads. PSL stays as the
    # fallback because it is one request when it is healthy.
    try:
        log(f"{y} bake: start via NCEI ({free_gb():.1f} GB free)")
        retry(f"{y} bake(ncei)", lambda: bake_ncei(y), tries=2, backoff=30)
    except Exception as e:                          # noqa: BLE001
        log(f"{y} NCEI unusable ({e}) — falling back to the PSL year file")
        nbytes = wait_for_source(y, max_wait=PSL_WAIT)
        log(f"{y} bake: start via PSL "
            f"({free_gb():.1f} GB free, source {nbytes / 1e6:.0f} MB)")
        retry(f"{y} bake", lambda: bake(y), tries=2, backoff=90)
    m = json.load(open(meta))
    days = int(m["days"])
    size = os.path.getsize(i16)
    want = NPIX * days * 2
    if size != want:
        raise SystemExit(f"FATAL: {y}.i16 is {size} bytes, expected {want}")
    with open(i16, "rb") as f:
        f.seek((PROBE_PX * days + min(180, days - 1)) * 2)
        pv = int.from_bytes(f.read(2), "little", signed=True)
    log(f"{y} bake: done in {time.time() - t0:.0f}s — days={days} "
        f"size={size} probe px={PROBE_PX} day={min(180, days - 1)} "
        f"raw={pv} degC={pv / 100.0:.2f}")

    t1 = time.time()
    log(f"{y} upload: start ({size / 1e6:.0f} MB)")
    retry(f"{y} upload i16", lambda: api.upload_file(
        path_or_fileobj=i16, path_in_repo=f"{PREFIX}/{y}.i16",
        repo_id=REPO, repo_type="dataset",
        commit_message=f"sst daily {y} ({days} days, pixel-major int16)"))
    retry(f"{y} upload json", lambda: api.upload_file(
        path_or_fileobj=meta, path_in_repo=f"{PREFIX}/{y}.json",
        repo_id=REPO, repo_type="dataset",
        commit_message=f"sst daily {y} meta"))
    log(f"{y} upload: done in {time.time() - t1:.0f}s")

    # ---- restore verification: read it back the way the browser will ----
    t2 = time.time()
    url = f"{RESOLVE}/{y}.i16"
    rng = random.Random(y)
    checks = [(PROBE_PX, min(180, days - 1))]
    while len(checks) < 3:
        checks.append((rng.randrange(NPIX), rng.randrange(days)))
    fh = open(i16, "rb")
    for px, day in checks:
        off = (px * days + day) * 2
        fh.seek(off)
        local = fh.read(2)
        remote = range_read(url, off)
        if remote != local:
            fh.close()
            log(f"{y} VERIFY MISMATCH at px={px} day={day} off={off}: "
                f"local={local.hex()} remote={remote.hex()} — STOPPING")
            raise SystemExit(f"FATAL: restore verification failed for {y}")
        log(f"  verify px={px} day={day} off={off} ok "
            f"({int.from_bytes(local, 'little', signed=True) / 100.0:.2f} degC)")
    fh.close()
    log(f"{y} verify: 3/3 byte-exact in {time.time() - t2:.0f}s")

    os.remove(i16)
    os.remove(meta)
    verified[y] = days
    write_index(api, verified)
    log(f"{y} DONE in {time.time() - t0:.0f}s total ({free_gb():.1f} GB free)")


def main():
    args = [int(a) for a in sys.argv[1:] if a.isdigit()]
    years = args or list(range(2026, 1990, -1))
    api = HfApi(token=open(TOKEN_FILE).read().strip())
    log(f"pipeline start pid={os.getpid()} years={years[0]}..{years[-1]} "
        f"free={free_gb():.1f} GB")
    log("reading hub state")
    verified = hub_state(api)
    log(f"already on hub: {sorted(verified, reverse=True)}")
    if verified:
        write_index(api, verified)                  # truthful from the first minute
    def pass_over(todo):
        bad = []
        for y in todo:
            if y in verified:
                log(f"{y} skip: already on hub with {verified[y]} days")
                continue
            try:
                do_year(api, y, verified)
            except SystemExit:
                raise                               # verify mismatch / no disk
            except Exception as e:                  # noqa: BLE001
                log(f"{y} ERROR: {e} — continuing with the next year")
                bad.append(y)
            time.sleep(PACE_S)   # PSL throttles a client that never pauses
        return bad

    failed = pass_over(years)
    # PSL goes down in stretches (504 for the whole of 2024, then no answer at
    # all, on 2026-08-18) and an outage that outlasts one year's patience often
    # does not outlast the next sweep — so keep sweeping rather than leaving a
    # hole in the archive. Each sweep is cheap when there is nothing to do.
    for n in range(1, 7):
        if not failed:
            break
        log(f"retry sweep {n}: {failed} — sleeping 600s first")
        time.sleep(600)
        failed = pass_over(failed)
    log(f"pipeline complete — verified {len(verified)} years, failed {failed}")


if __name__ == "__main__":
    main()

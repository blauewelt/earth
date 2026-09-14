#!/usr/bin/env python3
"""Copy the NOAA PSL inputs of family 7 to the Hugging Face Hub.

WHY THIS EXISTS — one measurement. The family-7.1 build runs on a rented Vast
box in the UK. From that box `downloads.psl.noaa.gov` trickles at **0.17 MB/s**
(measured 2026-09-13: one 477 MB `sst.day.mean.YYYY.nc` took 47 minutes; the
control run of the same file from another box took 33 s), so the `sst` and
`ncep` stages — 43 years x (477 MB + 70 MB) of OISST plus 43 years x 13 stems
x ~37 MB of NCEP gaussian dailies, about 45 GB in all — cannot finish inside
the workflow's 24 h timeout. The same box pulls the Hub at full speed (384
GLORYS chunks at 6 s each). From a US location PSL serves at 40-90 MB/s.

So the files are copied ONCE, from a GitHub-hosted runner that is fast to PSL,
into the dataset repo under `mirrors/psl/Datasets/...` — PSL's own URL path,
so the mapping is mechanical (`build_family7.psl_mirror_path`) and needs no
table. `build_family7.download_verified` then reads the mirror FIRST and falls
back to PSL unchanged, so a file this script has not reached yet costs the
build nothing but the old slow path.

TWO HOSTS NOW, AND THE ORDER IS MEASURED (2026-09-14). `downloads.psl.noaa.gov`
is still the PRIMARY, but at ~10:15Z that day it STOPPED SERVING: 0 bytes in
60 s from three GitHub-hosted runners and from the sandbox, on paths that were
correct (`.../Datasets/ncep.reanalysis/surface_gauss/air.2m.gauss.1998.nc`).
PSL's THREDDS front, `https://psl.noaa.gov/thredds/fileServer/` + the SAME path
after the host, was up the whole time and serves the IDENTICAL bytes: HEAD
declares `Content-Length: 37119095` for that file and a full download hashes
`6ed13870bfc6dff6075fb05731425a5e0958bbaa2c23132d3f6f46c8a18f7a80`, equal to
the Hub mirror copy uploaded earlier from downloads.

THREDDS IS THEREFORE THE FALLBACK, NOT THE PRIMARY, because it TRUNCATES about
half of its transfers: consecutive full GETs of one file returned 12587948
(complete), 11412001, 12587948; of another 12605590, 12605590, 11984793,
10822825 — always HTTP 200, always with the correct `Content-Length`, sometimes
ending in `curl: (92) HTTP/2 stream ... INTERNAL_ERROR` (`--http1.1` truncates
the same way, without the stream error). A truncated file is the one failure
mode that is SILENT: the symptom is `NetCDF: HDF error` at open time, a whole
year of the axis gone. So thredds is usable ONLY under the size check this
script already performs, with up to six attempts per file — the code base
already knows the shape (`build_family7.download_verified` compares against
`remote_size`; `build_family3.fetch` cycles mirrors).

`--source auto` (the default) MEASURES both hosts once per run, with a 20 MiB
range read and a 30 s budget, and takes the first of [downloads, thredds] that
serves at least 2 MB/s; `--source downloads|thredds` skips the probe. The Hub
path (`rel`) is derived from the DOWNLOADS URL either way, so the build's
mapping (`build_family7.psl_mirror_path`) is unchanged whichever host the bytes
came from — the manifest records `source_url` and `source_host` so a file can
be traced back to the one that served it.

THE RULE THIS SCRIPT ENFORCES, from `ml/hf_mirror.py`: **a backup is only real
if the restore works.** Every file is size-verified against PSL's own
Content-Length, uploaded, then DOWNLOADED BACK and sha256-compared against the
local bytes before either copy is deleted. An upload that returns 200 is not
evidence the bytes are retrievable (ml/CLAUDE.md §0.2 — assert the effect, not
the invocation), and a mirrored file that does not round-trip is WORSE than an
absent one, because the builder would trust it: such a file is DELETED from
the Hub and counted as failed. The job continues through the rest of the list
and exits non-zero at the end if anything failed.

ONE COMMIT PER BATCH, NOT PER FILE (2026-09-14). `upload_file` is one Hugging
Face Hub COMMIT, and the Hub allows **256 commits per repository per hour**.
Measured at 08:50Z: after 308 of 646 files the Hub answered `429 … You have
exceeded the rate limit for repository commits (256 per hour)` and 46 further
files failed. So up to `--batch-files` (8) or `--batch-gb` (3.0) of files are
downloaded and hashed, uploaded in ONE `create_commit`, and only then restored
and verified one by one — 646 files become ~81 commits, and a 429 is slept
through by `build_family7.hub_commit` rather than losing the file.

DISK. A hosted runner has ~14 GB free and one OISST year is 477 MB, so the
batch is capped by BYTES as well as by count: at most `--batch-gb` of source
files are on disk at once, plus the one file being restored for the check.

Credentials: `HF_TOKEN` in the environment, never argv (ml/CLAUDE.md §6).

Run:
  python3 ml/mirror_psl.py --what all --dry-run
  python3 ml/mirror_psl.py --what oisst --start 1982 --end 2024
  python3 ml/mirror_psl.py --what ncep --start 2020 --end 2024 --force
  python3 ml/mirror_psl.py --what ncep --start 2020 --end 2024 --source thredds
"""
import argparse
import json
import os
import shutil
import sys
import tempfile
import time
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import build_family3 as f3                                     # noqa: E402
import build_family7 as b7                                     # noqa: E402

# Module-level so the tests can replace them; the guarded `fetch` is
# build_family3's (a PSL host that trickles is aborted and retried rather than
# holding the runner for its whole six hours).
fetch = f3.fetch
remote_size = b7.remote_size
sha256 = b7.sha256

MIRROR_PREFIX = b7.HUB_MIRROR_PREFIX
MANIFEST_PATH = f"{MIRROR_PREFIX}/manifest.json"
DEFAULT_START, DEFAULT_END = 1982, 2024

# The two fronts PSL serves the same files from. The path after the host is
# IDENTICAL on both (measured 2026-09-14, see the module docstring), which is
# the whole reason `psl_host_url` is a rewrite and not a table.
PSL_HOSTS = {"downloads": "https://downloads.psl.noaa.gov/",
             "thredds": "https://psl.noaa.gov/thredds/fileServer/"}
HOST_ORDER = ("downloads", "thredds")
DOWNLOADS_HOST = PSL_HOSTS["downloads"]

PROBE_BYTES = 20 * 1024 * 1024          # 20 MiB is plenty to time a host
PROBE_BUDGET_S = 30.0                   # WALL CLOCK, not urlopen's per-read
MIN_MBPS = 2.0                          # the same floor the workflow refuses at

# Attempts per file. Three is `download_verified`'s number; thredds truncates
# roughly half its transfers, so six gives a file ~1.5% chance of losing all
# of them rather than ~12%.
ATTEMPTS = {"downloads": 3, "thredds": 6}

UA = {"User-Agent": "earth-science-pipeline/1.0 "
                    "(research; github blauewelt/earth)"}


class MirrorError(Exception):
    """One file failed. The list continues; the job exits non-zero."""


# ------------------------------------------------------------- the hosts --
def psl_host_url(url, host):
    """The same file on `host`: only the part BEFORE the path changes.

    `https://downloads.psl.noaa.gov/Datasets/x.nc`, "thredds" ->
    `https://psl.noaa.gov/thredds/fileServer/Datasets/x.nc`. The Hub path is
    derived from the downloads URL and never from this one.
    """
    if host not in PSL_HOSTS:
        raise ValueError(f"unknown PSL host {host!r}; know {sorted(PSL_HOSTS)}")
    if not isinstance(url, str) or not url.startswith(DOWNLOADS_HOST):
        raise ValueError(f"{url} is not a {DOWNLOADS_HOST} URL — the host "
                         f"rewrite is defined on that path only")
    return PSL_HOSTS[host] + url[len(DOWNLOADS_HOST):].lstrip("/")


def probe_host(url, budget_s=PROBE_BUDGET_S, want_bytes=PROBE_BYTES):
    """(MB/s, bytes, seconds) for one 20 MiB range read of a REAL file.

    `urlopen(timeout=)` is per-READ, not for the whole transfer — a host that
    trickles one byte per second would never time out — so the budget is
    enforced here, on the wall clock, by reading in chunks.
    """
    req = urllib.request.Request(
        url, headers=dict(UA, Range=f"bytes=0-{want_bytes - 1}"))
    t0 = time.time()
    got = 0
    try:
        with urllib.request.urlopen(req, timeout=budget_s) as r:
            while got < want_bytes and time.time() - t0 < budget_s:
                chunk = r.read(1 << 20)
                if not chunk:
                    break
                got += len(chunk)
    except Exception:                                          # noqa: BLE001
        pass                       # a host that errors is a host at 0 MB/s
    secs = max(time.time() - t0, 1e-6)
    return got / 1e6 / secs, got, secs


def probe_hosts(url, order=HOST_ORDER, min_mbps=MIN_MBPS):
    """Measure each host ONCE and return (host_or_None, {host: (mbps, b, s)}).

    ml/CLAUDE.md §0.3 — a precondition checked where the inputs are all it has
    cost. One 20 MiB read per host decides the whole run.
    """
    meas, pick = {}, None
    for host in order:
        mbps, got, secs = probe_host(psl_host_url(url, host))
        meas[host] = (mbps, got, secs)
        print(f"  probe {host}: {mbps:.2f} MB/s ({got:,} B in {secs:.1f} s)",
              flush=True)
        if pick is None and mbps >= min_mbps:
            pick = host
    return pick, meas


def probe_summary(meas):
    """The measurements as one line, for the message that refuses the run."""
    return " · ".join(f"{h}: {m[0]:.2f} MB/s ({m[1]:,} B in {m[2]:.1f} s)"
                      for h, m in meas.items())


# ------------------------------------------------------------- the file list --
def wanted(what, start, end):
    """[(source_url, path_in_repo)] in the order they will be mirrored.

    OISST first because it is the big half (43 x 547 MB against NCEP's
    43 x 13 x ~37 MB), so an interrupted run has moved the most bytes.
    """
    urls = []
    if what in ("oisst", "all"):
        for y in range(start, end + 1):
            for kind in ("sst", "icec"):
                urls.append(f"{b7.PSL_OISST}/{kind}.day.mean.{y}.nc")
    if what in ("ncep", "all"):
        # The land/sea mask is one untimed file and `ncep_land_mask` needs it
        # before any year can be regridded, so it leads the NCEP block.
        urls.append(f"{b7.PSL_NCEP}/{b7.NCEP_LAND}.nc")
        for y in range(start, end + 1):
            for stem in b7.NCEP_FILES.values():
                urls.append(f"{b7.PSL_NCEP}/{stem}.{y}.nc")
    out = []
    for u in urls:
        rel = b7.psl_mirror_path(u)
        if rel is None:                    # cannot happen; say so if it does
            raise SystemExit(f"{u} is not a downloads.psl.noaa.gov URL — the "
                             f"mirror layout is derived from that path")
        out.append((u, rel))
    return out


# ------------------------------------------------------------------- the Hub --
def hub_api():
    """(HfApi, repo_id, token). HF_TOKEN from the environment, never argv."""
    from huggingface_hub import HfApi
    tok = os.environ.get("HF_TOKEN") or (
        open("/home/claude/.hf_token").read().strip()
        if os.path.exists("/home/claude/.hf_token") else "")
    if not tok:
        sys.exit("no HF_TOKEN in the environment (never in argv) — see the "
                 "project doc claude/huggingface-access.md")
    api = HfApi(token=tok)
    return api, f"{api.whoami()['name']}/{b7.HF_DATASET}", tok


def hub_listing(api, repo):
    """{path_in_repo: bytes} for everything already under the mirror prefix.

    ONE call for the whole tree, not one per file: at 600+ files a per-file
    existence check is 600 round trips before a single byte moves.
    """
    out = {}
    try:
        # list() INSIDE the try: list_repo_tree is a lazy paginator, so a
        # missing prefix raises on the first iteration, not on the call —
        # the first-ever psl-mirror run (2026-09-14) died on exactly that,
        # with this except never reached.
        tree = list(api.list_repo_tree(repo, path_in_repo=MIRROR_PREFIX,
                                       repo_type="dataset", recursive=True))
    except Exception as e:                                     # noqa: BLE001
        print(f"  (no {MIRROR_PREFIX}/ on {repo} yet: {str(e)[:100]})")
        return out
    for it in tree or ():
        n = getattr(it, "size", None)
        p = getattr(it, "path", None)
        if p is not None and n is not None:
            out[p] = int(n)
    return out


def hf_download(repo, path_in_repo, dest_dir, token=None):
    """Download one repo file into `dest_dir` and return the local path."""
    from huggingface_hub import hf_hub_download
    kw = {"repo_type": "dataset", "local_dir": dest_dir}
    if token:
        kw["token"] = token
    return hf_hub_download(repo, path_in_repo, **kw)


def read_manifest(repo, token):
    """The manifest already on the Hub, as {path: record}; {} when absent."""
    tmp = tempfile.mkdtemp(prefix="pslman_")
    try:
        p = hf_download(repo, MANIFEST_PATH, tmp, token)
        man = json.load(open(p))
        return {r["path"]: r for r in man.get("files", [])}
    except Exception as e:                                     # noqa: BLE001
        print(f"  (no {MANIFEST_PATH} yet: {str(e)[:100]})")
        return {}
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


# ------------------------------------------------------------------ one file --
def stage_one(url, rel, workdir, host="downloads"):
    """Download from `host` onto the disk, size-verified, and hash it.

    Returns `{url, source_url, source_host, rel, name, path, bytes, sha256,
    down_s}`; raises MirrorError when the host could not send the file whole in
    `ATTEMPTS[host]` tries. Nothing has touched the Hub yet — the caller
    batches several of these into one commit.

    THE RETRY IS THE POINT ON THREDDS (2026-09-14): that host truncates about
    half of its transfers with a 200 and a correct `Content-Length`, so the
    size check plus six attempts is what makes it usable at all. `rel` is
    unchanged — it comes from the DOWNLOADS URL, so the build's mapping does
    not care which host served the bytes.
    """
    name = os.path.basename(rel)
    local = os.path.join(workdir, name)
    src = psl_host_url(url, host)
    want = remote_size(src)
    attempts = ATTEMPTS.get(host, 3)
    try:
        for i in range(attempts):
            t0 = time.time()
            fetch(src, local)
            down_s = max(time.time() - t0, 1e-6)
            got = os.path.getsize(local)
            if want is None or got == want:
                break
            print(f"  ::warning:: {name}: {host} sent {got:,} of {want:,} "
                  f"bytes — truncated transfer, refetching "
                  f"({i + 1}/{attempts})", flush=True)
            drop_staged({"path": local})   # fetch() returns early if it exists
        else:
            raise MirrorError(f"{name}: {host} sent {got:,} of {want:,} bytes "
                              f"in {attempts} attempts")
    except Exception:
        drop_staged({"path": local})
        raise
    return {"url": url, "source_url": src, "source_host": host,
            "rel": rel, "name": name, "path": local,
            "bytes": got, "sha256": sha256(local), "down_s": down_s}


def drop_staged(s):
    """Delete one staged file and its `.part` sibling. A runner has ~14 GB."""
    for p in (s["path"], s["path"] + ".part"):
        if p and os.path.exists(p):
            try:
                os.remove(p)
            except OSError:
                pass


def publish_batch(api, repo, staged, workdir, token=None):
    """ONE commit for the whole batch, then DOWNLOAD EACH BACK and compare.

    Returns `(records, failures)`. A file whose restore does not match is
    DELETED from the Hub — a mirrored file that does not round-trip is worse
    than an absent one, because the builder would read it and the truncation
    would surface hours later as `NetCDF: HDF error` — and is reported as a
    failure. The whole batch is one commit because the Hub allows only 256 per
    repository per hour (2026-09-14).
    """
    from huggingface_hub import CommitOperationAdd, CommitOperationDelete
    ops = [CommitOperationAdd(path_in_repo=s["rel"], path_or_fileobj=s["path"])
           for s in staged]
    names = ", ".join(s["name"] for s in staged)
    total = sum(s["bytes"] for s in staged)
    t0 = time.time()
    b7.hub_commit(api, repo, ops,
                  f"PSL mirror: {len(staged)} file(s), {total:,} bytes — "
                  f"{names}"[:900])
    up_s = max(time.time() - t0, 1e-6)
    print(f"  commit: {len(staged)} file(s), {total / 1e6:,.0f} MB in one "
          f"Hub commit, {up_s:.0f}s", flush=True)

    back = os.path.join(workdir, "back")
    records, failures, bad = [], [], []
    for s in staged:
        shutil.rmtree(back, ignore_errors=True)
        os.makedirs(back, exist_ok=True)
        try:
            p = hf_download(repo, s["rel"], back, token)
            restored = sha256(p)
        except Exception as e:                                 # noqa: BLE001
            failures.append((s["rel"], f"{s['name']}: restore FAILED "
                                       f"({type(e).__name__}: {str(e)[:120]})"))
            bad.append(s["rel"])
            continue
        finally:
            shutil.rmtree(back, ignore_errors=True)
        if restored != s["sha256"]:
            failures.append((s["rel"],
                             f"{s['name']}: RESTORE MISMATCH (uploaded "
                             f"{s['sha256'][:16]}, downloaded {restored[:16]}) "
                             f"— deleted from the Hub"))
            bad.append(s["rel"])
            continue
        records.append({"path": s["rel"], "bytes": s["bytes"],
                        "sha256": s["sha256"],
                        "source_url": s.get("source_url", s["url"]),
                        "source_host": s.get("source_host", "downloads"),
                        "mirrored_at": b7.utcnow(),
                        "_mb": s["bytes"] / 1e6,
                        "_down_mbps": s["bytes"] / 1e6 / s["down_s"],
                        "_up_s": up_s})
    if bad:
        b7.hub_commit(api, repo,
                      [CommitOperationDelete(path_in_repo=r) for r in bad],
                      f"PSL mirror: DELETE {len(bad)} file(s) — the round trip "
                      f"did not verify")
    return records, failures


def mirror_one(api, repo, url, rel, workdir, token=None, host="downloads"):
    """One file, staged and published on its own. Raises MirrorError.

    The batch path is what `main` runs; this is the one-file form the tests
    and an operator use, and it goes through exactly the same commit helper.
    """
    s = stage_one(url, rel, workdir, host)
    try:
        records, failures = publish_batch(api, repo, [s], workdir, token)
    finally:
        drop_staged(s)
    if failures:
        raise MirrorError(failures[0][1])
    return records[0]


# ---------------------------------------------------------------------- main --
def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--what", choices=("oisst", "ncep", "all"), default="all")
    ap.add_argument("--start", type=int, default=DEFAULT_START)
    ap.add_argument("--end", type=int, default=DEFAULT_END)
    ap.add_argument("--dry-run", action="store_true",
                    help="list what would be mirrored and stop")
    ap.add_argument("--force", action="store_true",
                    help="re-upload even when the Hub already holds the file "
                         "at the same size")
    ap.add_argument("--batch-files", type=int, default=8,
                    help="how many files go into ONE Hub commit (the Hub "
                         "allows 256 commits per repo per hour)")
    ap.add_argument("--source", choices=("auto", "downloads", "thredds"),
                    default="auto",
                    help="which PSL front to pull from. auto MEASURES both "
                         "once (20 MiB, 30 s each) and takes the first of "
                         "[downloads, thredds] at >= 2 MB/s; naming one skips "
                         "the probe")
    ap.add_argument("--batch-gb", type=float, default=3.0,
                    help="and how many gigabytes — a hosted runner has ~14 GB "
                         "free and one OISST year is 477 MB")
    a = ap.parse_args(argv)
    if a.end < a.start:
        sys.exit(f"--end {a.end} precedes --start {a.start}")

    files = wanted(a.what, a.start, a.end)
    print(f"PSL mirror -> {MIRROR_PREFIX}/ · what={a.what} "
          f"{a.start}..{a.end} · {len(files)} file(s)")
    if a.dry_run and not os.environ.get("HF_TOKEN"):
        for u, rel in files[:10]:
            print(f"  {rel}   <- {u}")
        if len(files) > 10:
            print(f"  … and {len(files) - 10} more")
        print("dry run (no HF_TOKEN): nothing was contacted")
        return 0

    api, repo, tok = hub_api()
    listing = hub_listing(api, repo)
    print(f"  {repo}: {len(listing)} file(s) already under {MIRROR_PREFIX}/")
    manifest = read_manifest(repo, tok)

    # WHICH HOST, MEASURED ONCE, BEFORE ANY FILE MOVES. downloads is the
    # primary and thredds the fallback (module docstring, 2026-09-14); a run
    # where NEITHER serves is refused here rather than after six hours of
    # nothing.
    if a.source == "auto":
        host, meas = probe_hosts(files[0][0])
        if host is None:
            sys.exit(f"no PSL host reaches {MIN_MBPS:.0f} MB/s — "
                     f"{probe_summary(meas)}. downloads.psl.noaa.gov stopped "
                     f"serving at ~10:15Z on 2026-09-14 and the thredds front "
                     f"is the fallback; if both are down there is nothing to "
                     f"mirror from, so re-dispatch later.")
        print(f"  source: {host} ({meas[host][0]:.2f} MB/s), "
              f"{ATTEMPTS.get(host, 3)} attempt(s) per file")
    else:
        host = a.source
        print(f"  source: {host} (--source given, no probe), "
              f"{ATTEMPTS.get(host, 3)} attempt(s) per file")

    todo = []
    skipped = 0
    for u, rel in files:
        if not a.force and rel in listing:
            want = remote_size(psl_host_url(u, host))
            if want is None or listing[rel] == want:
                skipped += 1
                continue
            print(f"  ::warning:: {rel} is {listing[rel]:,} bytes on the Hub "
                  f"and {want:,} at PSL — re-mirroring")
        todo.append((u, rel))
    print(f"  {skipped} already mirrored at the same size · {len(todo)} to do")

    if a.dry_run:
        for u, rel in todo[:20]:
            print(f"  would mirror {rel}")
        if len(todo) > 20:
            print(f"  … and {len(todo) - 20} more")
        return 0

    work = tempfile.mkdtemp(prefix="pslmirror_")
    failed, done = [], 0
    cap_files = max(1, int(a.batch_files))
    cap_bytes = max(1, int(a.batch_gb * 1e9))
    print(f"  batching up to {cap_files} file(s) / {a.batch_gb:.1f} GB per "
          f"Hub commit")
    try:
        staged, nb = [], 0

        def flush():
            """Publish what is staged, record it, and free the disk."""
            nonlocal staged, nb, done
            if not staged:
                return
            try:
                records, failures = publish_batch(api, repo, staged, work, tok)
            except Exception as e:                             # noqa: BLE001
                for s in staged:
                    failed.append((s["rel"], f"commit FAILED: {str(e)[:180]}"))
                    print(f"  {s['name']}  FAILED: commit "
                          f"{str(e)[:140]}", flush=True)
                records, failures = [], []
            for rel_, why in failures:
                failed.append((rel_, why))
                print(f"  {os.path.basename(rel_)}  FAILED: {why[:160]}",
                      flush=True)
            for rec in records:
                done += 1
                print(f"  [{done}/{len(todo)}] {os.path.basename(rec['path'])}"
                      f"  {rec.pop('_mb'):,.0f} MB  "
                      f"{rec.pop('_down_mbps'):.1f} MB/s down  "
                      f"{rec.pop('_up_s'):.0f}s up (batch)", flush=True)
                manifest[rec["path"]] = rec
                if done % 25 == 0:
                    write_manifest(api, repo, manifest)
            for s in staged:
                drop_staged(s)
            staged, nb = [], 0

        for i, (u, rel) in enumerate(todo, 1):
            try:
                s = stage_one(u, rel, work, host)
            except Exception as e:                             # noqa: BLE001
                failed.append((rel, str(e)[:200]))
                print(f"  [{i}/{len(todo)}] {os.path.basename(rel)}  FAILED: "
                      f"{str(e)[:160]}", flush=True)
                continue
            staged.append(s)
            nb += s["bytes"]
            if len(staged) >= cap_files or nb >= cap_bytes:
                flush()
        flush()
    finally:
        shutil.rmtree(work, ignore_errors=True)

    if done:
        write_manifest(api, repo, manifest)
    print(f"\nmirrored {done} file(s), skipped {skipped}, "
          f"{len(failed)} failed")
    for rel, why in failed:
        print(f"  FAILED {rel}: {why}")
    if failed:
        print(f"::error::{len(failed)} file(s) did not mirror; re-dispatch to "
              f"retry them — the ones that succeeded are skipped by size")
        return 1
    return 0


def write_manifest(api, repo, manifest):
    """`mirrors/psl/manifest.json`, merged with whatever was already there."""
    recs = [manifest[k] for k in sorted(manifest)]
    body = {"prefix": MIRROR_PREFIX,
            "updated_at": b7.utcnow(),
            "note": ("NOAA PSL inputs of family 7, mirrored because PSL "
                     "serves the build box at 0.17 MB/s. Every file was "
                     "size-verified against Content-Length, uploaded, "
                     "downloaded back and sha256-compared."),
            "count": len(recs),
            "bytes": int(sum(r.get("bytes", 0) for r in recs)),
            "files": recs}
    tmp = tempfile.mkdtemp(prefix="pslman_")
    try:
        p = os.path.join(tmp, "manifest.json")
        with open(p, "w") as fh:
            json.dump(body, fh, indent=1, sort_keys=True)
        b7.hub_upload_with_backoff(api, repo, p, MANIFEST_PATH,
                                   f"PSL mirror: manifest ({len(recs)} files)")
        print(f"  manifest: {len(recs)} file(s), "
              f"{body['bytes'] / 1e9:.1f} GB")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    sys.exit(main())

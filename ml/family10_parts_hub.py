#!/usr/bin/env python3
"""Family 10's PER-YEAR COLUMN PARTS, parked on the Hub between two machines.

WHY THIS EXISTS. `slatrack` — Copernicus Marine's along-track sea level, 29
altimeter missions, 1993-2024 — is the one family-10 store whose fetch and
whose assembly cannot happen on the same machine:

  * the FETCH needs Copernicus credentials, and `ml/CLAUDE.md` §6 forbids
    those on a rented Vast box (the host has root on the physical machine).
    They may exist only on a GitHub-HOSTED runner, which masks its secrets and
    is destroyed with the job.
  * the ASSEMBLY needs tens of gigabytes of disk. Measured 2026-09-14, 2015
    alone holds ~70 M samples; 1993-2024 is ~1.5-2.5 e9 rows at ~33 bytes a
    row in the store, i.e. 50-80 GB. A hosted runner has ~14 GB of disk and a
    six-hour job cap.

So the two halves are split, and this module is the seam. A hosted lane
fetches ONE YEAR, pushes that year's parts here, and deletes them; a box with
nothing but `HF_TOKEN` pulls every year back and assembles
(`build_family10_stores.py --stage fetch,publish --parts-from-hub`).

THE LAYOUT, and it is the whole contract:

    partials/family10_<minor>/<store>/<year>/00000.npz   the column parts, as
    partials/family10_<minor>/<store>/<year>/00001.npz   `PartWriter` wrote them
    partials/family10_<minor>/<store>/<year>/...
    partials/family10_<minor>/<store>/<year>/counts.json the year's own ledger
    partials/family10_<minor>/<store>/<year>/done.json   THE MARKER — last

LANES (E-082 wave 7), and they are an ADDITION to that layout, not a change to
it. A fetch lane is one GitHub-hosted runner: six hours and about 86 GB. Some
family-1 stores need more than one lane per year — a year of ICESat-2's ATL08
is 35 hours of fetching, so it goes as twelve monthly lanes — and one folder
per year with ONE index, ONE ledger and ONE marker means two such lanes
overwrite each other and the assembler builds a short store with nothing to say
so. So a NAMED lane gets a folder of its own inside the year's:

    partials/<family>/<store>/<year>/<lane>/00000.npz    the lane's own parts
    partials/<family>/<store>/<year>/<lane>/counts.json  the lane's own ledger,
                                                         carrying `lane`, the
                                                         window it covers and
                                                         its group subset
    partials/<family>/<store>/<year>/<lane>/done.json    THE LANE'S MARKER —
                                                         last, and carrying
                                                         `lane`

The lane's NAME says what it covers: `m06` (one month), `q3` (one quarter),
`d0701-0930` (any other window) or `g-<hash>` (a subset of a tier-G store's
groups, the list itself in the lane's ledger). A WHOLE-YEAR LANE WITH NO GROUP
SUBSET IS THE UNNAMED LANE and writes the top-level files above — exactly what
was written before this existed, which is why every store and every part
already on the Hub reads unchanged. `push`/`push_many` take `lane=` and upload
only that lane's folder; `pull` LISTS the year folder and brings back the
top-level files as the unnamed lane plus every sub-folder that carries a
`done.json`, so a year can be pushed by twelve machines and pulled by one.
Merging them into a year is the assembler's job, and its rules (disjoint
windows, disjoint (group, bin) shards, summed ledgers, declared completeness)
are in `ml/build_family1_stores.py`.

THE PREFIX CARRIES THE FAMILY VERSION, and that is not decoration. Family 10.1
stores the time as `time_s` (int32 seconds) where family 10 stored `time_days`
(float32 days), and a v1 part CANNOT be upgraded — float32 days resolve 21-84 s
over this record, so the seconds the archive published were destroyed at the
cast, before the part was written. `partials/family10_1/` is therefore a fresh
prefix rather than a place to mix schemas, and `push` and `pull` BOTH refuse a
part carrying `time_days` by name (`family10_store.check_part_schema`), because
a store assembled from one would claim a precision it does not have. The v1
parts under `partials/family10/` are not inputs to any build and nothing here
reads them.

THE PREFIX FOLLOWS `FAMILY_VERSION`, WHICH IS NOW "10.2" (E-081) — so this
module reads and writes `partials/family10_2/`. The slatrack lanes of 10.1
parked their parts under `partials/family10_1/` and those are NOT visible from
here, which is correct and deliberate: 10.2 adds the `fishing` store and
inherits slatrack's finished store from `tensors/family10_1/slatrack/` rather
than reassembling it. A slatrack build under 10.2 would therefore have to
re-run its fetch lanes; there is no reason to, and the registry's `inherits`
block records which version each store came from.

A PUSH NEVER SHORTENS A YEAR (2026-09-22). If the Hub's `done.json` for the
same (year, lane) vouches for a shard this push does not carry, the push is
refused before any upload (`ledger_would_shrink`) — pace4k's 2025 lane had
replaced a 62-bin 2024 with a one-bin copy. `pull`'s download retries
transient Hub failures (`transient_download_error`).

`done.json` is the year's marker and it obeys §5.21: it is uploaded only after
every part has been uploaded AND DOWNLOADED BACK with a matching sha256. A year
whose `done.json` is absent is a year nobody may use — `pull` reports it
missing and the fetch lane simply refetches it. A marker that can only
under-claim is the difference between "this year is short and nothing says so"
and "this year is refetched".

  push    --store slatrack --year 2015 --work /tmp/f10
  pull    --store slatrack --years 1993-2024 --work /opt/earth-cache/f10
  status  --store slatrack [--years 1993-2024]

NO CREDENTIAL IS EVER PRINTED OR READ HERE. `HF_TOKEN` reaches the Hub through
`build_family7.hub_repo()` and nothing else; the Copernicus credentials are not
touched by this module at all, which is the point of it.
"""
import argparse
import collections
import concurrent.futures as cf
import hashlib
import json
import os
import re
import shutil
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import family10_store as f10                                     # noqa: E402
from build_family7 import (atomic_json, git_sha, hub_add_ops,    # noqa: E402
                           hub_commit, hub_repo, mark, marked, parse_years,
                           read_json, sha256, utcnow)

# ONE CONSTANT, derived from `family10_store.FAMILY_VERSION` like every other
# family-10 path. Nothing here spells a prefix out.
HF_PARTIALS = f10.HF_PARTIALS                      # partials/family10_2
DONE = "done.json"
COUNTS = "counts.json"

# One Hub commit carries this many files. The Hub allows 256 commits per
# repository per hour (build_family7's `hub_commit` header); a slatrack year is
# ~70 parts at FLUSH_ROWS = 1e6, so batching keeps a year to two or three
# commits instead of seventy and leaves the lane's other years room in the
# window.
BATCH = 64

# Parts of one lane download this many at a time (2026-09-23): serially, 148
# x ~35 MB took ~3.5 min on #459 (slow disk) AND #593 (NVMe) — latency-bound.
PULL_WORKERS = 8


# ----------------------------------------------------------------- the Hub --
# Four seams, and they exist so the tests can hand this module a directory
# instead of the network. NOTHING else in here talks to huggingface_hub.
def _hub():
    """(api, repo, token) — `HF_TOKEN` from the environment, never argv."""
    return hub_repo()


def _list_files(api, repo, prefix):
    """Every repo path under `prefix`, as a set."""
    files = api.list_repo_files(repo, repo_type="dataset")
    pre = prefix.rstrip("/") + "/"
    return {p for p in files if p.startswith(pre)}


def _upload(api, repo, pairs, message):
    """[(path_in_repo, local_path)] -> ONE commit, through the 429 backoff."""
    return hub_commit(api, repo, hub_add_ops(pairs), message)


DOWNLOAD_ATTEMPTS = 6
DOWNLOAD_BACKOFF_S = (5, 15, 45, 120, 300)


def _is_404(e):
    """A Hub 'entry not found' of any spelling (huggingface_hub / httpx)."""
    try:
        from huggingface_hub.errors import EntryNotFoundError
        if isinstance(e, EntryNotFoundError):
            return True
    except Exception:                                   # pragma: no cover
        pass
    code = getattr(getattr(e, "response", None), "status_code", None)
    return code == 404 or "404" in str(e)[:80]


def transient_download_error(e):
    """A Hub read that may succeed on retry: a dropped connection, a
    timeout, a TLS handshake that timed out, a 5xx or a 429 — never a 4xx
    that names OUR request (a 401 or a 404 will not change by asking again).

    `hf_hub_download` reports a HEAD that could not reach the Hub as
    `LocalEntryNotFoundError` ("…we cannot find the requested files in the
    local cache"), an OSError; that is a connection failure, not an answer.

    With PULL_WORKERS threads sharing huggingface_hub's one httpx client, a
    worker that hits a connection error closes that client to recover, and a
    sibling that picked it up in the same instant raises a plain
    RuntimeError("Cannot send a request, as the client has been closed.") —
    a dropped connection seen from the other thread, so it retries too.
    """
    if isinstance(e, RuntimeError) and "client has been closed" in str(e):
        return True
    try:
        from huggingface_hub.utils import HfHubHTTPError
    except Exception:                                   # pragma: no cover
        HfHubHTTPError = ()
    if HfHubHTTPError and isinstance(e, HfHubHTTPError):
        code = getattr(getattr(e, "response", None), "status_code", None)
        return code is None or code == 429 or code >= 500
    name = type(e).__name__
    mod = type(e).__module__ or ""
    if mod.startswith(("httpx", "httpcore", "requests", "urllib3")):
        return not name.endswith(("HTTPStatusError", "InvalidURL"))
    return isinstance(e, (ConnectionError, TimeoutError, OSError))


def _download(repo, path_in_repo, token, dest_dir, just_uploaded=False):
    """Stream one repo file into `dest_dir`; returns the local path.

    `just_uploaded=True` (the restore-verify right after a commit) also
    retries a 404: measured 2026-09-23 on family1-build #597 (sst_acspo02
    2017 H2, a 2.9 h lane), the Hub answered 404 for
    `sst_acspo02__bin_2621.idx.npy` seconds after the commit that carried it
    and served the same file minutes later — propagation lag, not absence.
    The lane died with its parts on the Hub and no done.json.

    WITH A RETRY LADDER FOR TRANSIENT NETWORK FAILURES, the one
    `build_family1_stores._download` got in 300905a. lst05's box assembly
    (family1-build #425, 2026-09-21) was pulling thousands of parked parts
    when one GET died on `_ssl.c:989: The handshake operation timed out`
    (`2021/terra__bin_2859.idx.npy`); a pull is a loop of thousands of
    requests, so one of them failing transiently is the expected case, and
    it must cost a few seconds of backoff rather than the job. Six attempts,
    ~8 minutes of backoff at most; a 4xx is raised at once.
    """
    # PLAIN HTTPS, NOT hf_hub_download (2026-09-24). huggingface_hub fetches
    # through its hf_xet chunk store when it can, and on several rented
    # hosts that path crawled behind "peer closed connection without
    # sending complete message body" resumes — the swot restore on
    # Singapore and Maryland, the pheno500 parts pull on California (18 GB
    # in an hour over 11,000 small files) — while a streamed GET on the
    # resolve URL ran at the host's line rate. `hub_stream` is that GET,
    # written straight to the destination file, Range-resumed on a drop.
    os.makedirs(dest_dir, exist_ok=True)
    dest = os.path.join(dest_dir, os.path.basename(path_in_repo))
    tmp = dest + ".part"
    last = None
    for i in range(DOWNLOAD_ATTEMPTS):
        try:
            with open(tmp, "wb") as fh:
                hub_stream(repo, path_in_repo, token, fh.write,
                           just_uploaded=just_uploaded,
                           private=_private_repo(repo))
            os.replace(tmp, dest)
            return dest
        except Exception as e:                          # noqa: BLE001
            fresh_404 = just_uploaded and _is_404(e)
            if (not transient_download_error(e) and not fresh_404) \
                    or i == DOWNLOAD_ATTEMPTS - 1:
                raise
            last = e
            wait = DOWNLOAD_BACKOFF_S[min(i, len(DOWNLOAD_BACKOFF_S) - 1)]
            print(f"::warning::{path_in_repo}: {type(e).__name__}: "
                  f"{str(e)[:160]} — attempt {i + 1}/{DOWNLOAD_ATTEMPTS}, "
                  f"retrying in {wait}s", flush=True)
            time.sleep(wait)
    raise last                                          # pragma: no cover


STREAM_CHUNK = 8 << 20
STREAM_ATTEMPTS = 12
STREAM_TIMEOUT = (30, 120)


class StreamRefused(IOError):
    """A definite answer from the Hub (a plain 4xx, a Range ignored) — not
    retried, unlike a dropped connection or a 5xx."""


def hub_stream(repo, path_in_repo, token, consume, just_uploaded=False,
               attempts=STREAM_ATTEMPTS, private=False):
    """Stream one repo file over PLAIN HTTPS, feeding `consume(bytes)`;
    returns the byte count. Nothing touches the disk.

    WHY NOT hf_hub_download. Since huggingface_hub grew the hf_xet backend a
    file is fetched as deduplicated chunks from the Xet content store, and on
    the rented boxes that path crawled on the swot store's big columns:
    family1-build #734 (Singapore) and #817 (Maryland, 2026-09-24) sat on the
    36.5 GB `lat.npy` for hours at a few MB/s behind "peer closed connection
    without sending complete message body" resumes, while a plain ranged GET
    of the same object from the sandbox streamed at 7–109 MB/s. A restore
    only needs the bytes once, in order, so it streams them: one GET on the
    resolve URL, hashed as it arrives, resumed from the byte reached with a
    Range header when the connection drops.

    A 404 is retried only when `just_uploaded` (Hub propagation lag, see
    `_download`); any other 4xx raises at once. The token is sent only for a
    private repository — `requests` drops Authorization on the cross-host
    redirect to the CDN anyway, and a public object needs none.
    """
    import requests
    url = f"https://huggingface.co/datasets/{repo}/resolve/main/{path_in_repo}"
    got, total, last = 0, None, None
    for i in range(max(1, attempts)):
        hdr = {"User-Agent": "earth-science-pipeline/1.0"}
        if private and token:
            hdr["Authorization"] = f"Bearer {token}"
        if got:
            hdr["Range"] = f"bytes={got}-"
        try:
            with requests.get(url, headers=hdr, stream=True,
                              timeout=STREAM_TIMEOUT,
                              allow_redirects=True) as r:
                if r.status_code == 404 and just_uploaded and i < attempts - 1:
                    raise IOError(f"HTTP 404 (just uploaded, propagating)")
                if r.status_code in (429, 500, 502, 503, 504):
                    raise IOError(f"HTTP {r.status_code}")
                if r.status_code >= 400:
                    raise StreamRefused(f"{url}: HTTP {r.status_code}")
                if got and r.status_code != 206:
                    # the server ignored the Range: the whole file would come
                    # again and be hashed twice — refuse, never guess
                    raise StreamRefused(f"{url}: asked for bytes {got}- and "
                                        f"got HTTP {r.status_code} without a "
                                        f"range; a resume is not possible "
                                        f"here")
                if total is None:
                    cl = r.headers.get("Content-Length")
                    total = int(cl) if cl else None
                for blk in r.iter_content(chunk_size=STREAM_CHUNK):
                    if blk:
                        consume(blk)
                        got += len(blk)
            if total is not None and got < total:
                raise IOError(f"stream ended at {got} of {total} bytes")
            return got
        except StreamRefused:
            raise
        except (IOError, OSError, requests.RequestException) as e:
            last = e
            if i == attempts - 1:
                break
            wait = DOWNLOAD_BACKOFF_S[min(i, len(DOWNLOAD_BACKOFF_S) - 1)]
            print(f"::warning::{path_in_repo}: {type(e).__name__}: "
                  f"{str(e)[:160]} at byte {got} — attempt {i + 1}/"
                  f"{attempts}, resuming in {wait}s", flush=True)
            time.sleep(wait)
    raise IOError(f"{url}: {type(last).__name__}: {last} (after {attempts} "
                  f"attempts, {got} bytes)")


def hub_stream_sha256(repo, path_in_repo, token, just_uploaded=False,
                      private=False, attempts=STREAM_ATTEMPTS):
    """(sha256 hex, bytes) of one repo file, streamed — see `hub_stream`."""
    h = hashlib.sha256()
    n = hub_stream(repo, path_in_repo, token, h.update,
                   just_uploaded=just_uploaded, private=private,
                   attempts=attempts)
    return h.hexdigest(), n


def _private_repo(repo):
    """The token goes only to a private repository (the `-private` twin,
    `build_family10_stores.PRIVATE_SUFFIX`); `requests` drops it on the
    redirect to the CDN anyway."""
    return str(repo).endswith("-private")


def _ordered_map(fn, items, workers, lookahead=None):
    """`map(fn, items)` on a thread pool, results IN ORDER, never more than
    `lookahead` (default 2 x workers) submitted at once — the house pattern
    (`family1/adapters/_common.ordered_map`), kept local so this module does
    not import numpy and the family-10 builder for twenty lines.

    ONE DIFFERENCE, and it is the point: when the consumer stops early — a
    worker raised, or the caller `sys.exit`s on a mismatch — the futures
    still QUEUED are cancelled rather than run. A plain `with` exit waits
    for every submitted future, so a failed pull would first sit through up
    to `lookahead` more downloads, each with its own ~8-minute retry ladder,
    before it reported anything. Futures already running cannot be stopped;
    at most `workers` of them finish in the background.
    """
    items = list(items)
    lookahead = lookahead or 2 * workers
    ex = cf.ThreadPoolExecutor(max(1, workers))
    q = collections.deque()
    try:
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
    finally:
        for f in q:
            f.cancel()
        ex.shutdown(wait=True)


# --------------------------------------------------------------- the paths --
def hub_prefix(store, year=None, partials=None, lane=None):
    """`partials/<family>/<store>[/<year>[/<lane>]]`.

    `partials` defaults to family 10's prefix; family 1 passes its own
    (E-082). `lane` is "" or None for the UNNAMED lane, which is the layout
    every year on the Hub was written with.
    """
    p = f"{partials or HF_PARTIALS}/{store}"
    if year is None:
        return p
    p = f"{p}/{year}"
    return f"{p}/{lane}" if lane else p


def store_root(work, store):
    return os.path.join(os.path.abspath(work), store)


def year_dir(work, store, year, lane=None):
    """The local folder one year's (or one lane's) parts live in."""
    d = os.path.join(store_root(work, store), "parts", str(year))
    return os.path.join(d, lane) if lane else d


def part_key(year, lane=None):
    """The local marker name for a year's parts — `parts/<year>` unnamed,
    `parts/<year>/<lane>` named. Same rule as `Ctx.part_key`."""
    return f"parts/{year}/{lane}" if lane else f"parts/{year}"


# What a year directory's parts are. `.npz` is a tier-P column part; `.zst`
# and `.npy` are a family-1 SHARDED tier-G year (E-082 wave 2: the year's
# shards, their indices and one shard_index.npy per group — see
# ml/family1/sharded.py). A family-10 year directory holds only `.npz`, so its
# list, and therefore its done.json, is exactly what it always was.
PART_SUFFIXES = (".npz", ".zst", ".npy")


def local_part_files(d):
    """The files a year directory contributes, in the order `read_parts` reads
    them: the numbered parts sorted by name, then `counts.json`."""
    if not os.path.isdir(d):
        return []
    npz = sorted(n for n in os.listdir(d) if n.endswith(PART_SUFFIXES))
    out = list(npz)
    if os.path.exists(os.path.join(d, COUNTS)):
        out.append(COUNTS)
    return out


def _entries(d, names):
    return [{"name": n, "bytes": os.path.getsize(os.path.join(d, n)),
             "sha256": sha256(os.path.join(d, n))} for n in names]


def _by_name(done):
    return {e["name"]: e["sha256"] for e in (done or {}).get("files", [])}


# A SHARD is the one kind of part whose NAME says what it covers: the
# (group, bin) of a tier-G year. `.idx.npy` travels with it; `shard_index.npy`
# and `counts.json` are the year's ledger and are rewritten by every push.
SHARD_SUFFIX = ".zst"


def ledger_would_shrink(have, entries):
    """The shards the Hub's `done.json` for this (year, lane) already
    VOUCHES FOR and this push does not carry — [] when the push covers them.

    WHY. `push`/`push_many` used to skip a year only when the Hub's marker
    matched byte for byte and otherwise OVERWROTE the year's ledger, shard
    index and marker. pace4k (2026-09-20): the 2025 lane's window reached
    back into bin 3141 (2024-12-31 .. 2025-01-04), filed it under 2024, and
    pushed a ONE-BIN "2024" after the 2024 lane had pushed 62 bins — so
    `2024/done.json` and `pace4k__shard_index.npy` now describe one bin and
    the other 61 shards sit in the folder indexed by nothing. lst05/2007 and
    pheno500/2017 and /2019 carry the same signature. A push may REPLACE a
    ledger with one that covers at least the same shards (a re-fetch of the
    year); it may never replace it with one that covers fewer.
    """
    if not have:
        return []
    old = {e["name"] for e in have.get("files", [])
           if str(e.get("name", "")).endswith(SHARD_SUFFIX)}
    new = {e["name"] for e in entries if e["name"].endswith(SHARD_SUFFIX)}
    return sorted(old - new)


def shrink_refusal(what, have, lost, n_new):
    return (f"push refuses {what}: the Hub's done.json for it (written "
            f"{have.get('at')}, builder {str(have.get('builder_git_sha'))[:8]}"
            f") vouches for {len(lost)} shard(s) this push does not carry "
            f"({', '.join(lost[:4])}{' …' if len(lost) > 4 else ''}); this "
            f"push has {n_new}. Pushing would REPLACE that year's ledger, "
            f"shard index and marker with a shorter one and orphan those "
            f"shards — the pace4k/2024 collision (a lane whose window reached "
            f"into a bin that STARTS in the previous year). Nothing was "
            f"uploaded. If this lane should not own that year at all, fix its "
            f"window (a lane owns the bins whose first day falls inside it, "
            f"ml/family1/ADAPTER_CONTRACT.md 'Lanes'); if the Hub's ledger is "
            f"itself the damaged one, rebuild it from its shards with "
            f"`build_family1_stores.py --stage repair`.")


def hub_lanes(listing, store, year, partials=None):
    """Which LANES of one year are done on the Hub, in MERGE ORDER.

    Read out of the repository listing rather than asked of the Hub file by
    file: a path `<year>/done.json` is the unnamed lane, a path
    `<year>/<lane>/done.json` is that named lane, and anything deeper is not a
    lane. The unnamed lane comes first, then the named ones in name order —
    the same order `Ctx.lanes_of` uses locally, so the parts arrive in the
    order the assembler will walk them.

    An empty answer means the year has no marker at all, which `pull` reports
    as missing; it never means "the Hub would not tell us" (`read_done` is
    where that distinction is made and refused).
    """
    pre = hub_prefix(store, year, partials) + "/"
    named, top = set(), False
    for p in listing or ():
        if not p.startswith(pre) or not p.endswith("/" + DONE):
            continue
        rest = p[len(pre):].split("/")
        if len(rest) == 1:
            top = True
        elif len(rest) == 2:
            named.add(rest[0])
    return ([""] if top else []) + sorted(named)


def read_done(api, repo, tok, store, year, scratch, listing=None,
              partials=None, lane=None):
    """The year's (or the lane's) `done.json` off the Hub, or None if it is
    NOT THERE.

    "Not there" means not there: the repo listing does not carry the path, so
    the year was never pushed (or its push did not finish). It does NOT mean
    "the Hub would not give it to us", and the two used to arrive here as the
    same `None` — one bare `except Exception`. A Hub outage, an expired token
    or a rate limit therefore read as "that year was never fetched", which is
    the wrong half of the sentence: `pull` then told the operator to re-run a
    fetch lane that had already done its work, and `pull --allow-missing`
    assembled a store that silently dropped a year whose parts were sitting on
    the Hub the whole time. So a download that fails on a path the listing DOES
    carry raises, naming the year (ml/CLAUDE.md §4.6, and family 7's
    `truth_files` in commit fd3b446).

    With no `listing` in hand there is no way to tell the two apart, so a 404
    is read out of the error text and anything else raises.
    """
    what = f"{store} {year}" + (f" lane {lane}" if lane else "")
    path = f"{hub_prefix(store, year, partials, lane)}/{DONE}"
    if listing is not None and path not in listing:
        return None
    tmp = os.path.join(scratch, f"done_{year}{('_' + lane) if lane else ''}")
    shutil.rmtree(tmp, ignore_errors=True)
    try:
        p = _download(repo, path, tok, tmp)
        got = read_json(p, None)
        if got is None:
            raise IOError(f"{repo}:{path} downloaded but does not parse as "
                          f"JSON — the marker for {what} is corrupt")
        return got
    except Exception as e:                                    # noqa: BLE001
        if listing is None and _looks_absent(e):
            return None
        raise IOError(
            f"cannot read {repo}:{path}, the marker for {what}: "
            f"{type(e).__name__}: {e}. The path IS in the repository listing, "
            f"so this is the Hub refusing to serve it (an outage, a rate "
            f"limit, or an HF_TOKEN without read access) and NOT a year that "
            f"was never fetched — refusing to report it as missing, because "
            f"--allow-missing would then drop a year whose parts are on the "
            f"Hub.") from e
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def _looks_absent(exc):
    """Is this exception the Hub saying 404, rather than saying no?

    `LocalEntryNotFoundError` subclasses `EntryNotFoundError`, and it is what
    `hf_hub_download` raises when its HEAD could not REACH the Hub — a
    connection failure, which must never read as "that year was never
    pushed"."""
    if type(exc).__name__ == "LocalEntryNotFoundError":
        return False
    try:
        from huggingface_hub.utils import EntryNotFoundError
        if isinstance(exc, EntryNotFoundError):
            return True
    except Exception:                                         # noqa: BLE001
        pass
    code = getattr(getattr(exc, "response", None), "status_code", None)
    if code == 404:
        return True
    return type(exc).__name__ in ("EntryNotFoundError", "FileNotFoundError")


# ==================================================================== push ===
def push(store, year, work, scratch=None, partials=None, hub=None,
         private=False, lane=None):
    """Upload one fetched year's parts, verify by restore, THEN mark it done.

    E-082: `partials` (the prefix), `hub` (a callable returning (api, repo,
    token)) and `private` route a family-1 store; left at None/False they
    are family 10's module defaults, and the `_hub` seam the tests replace.

    `lane` (E-082 wave 7) uploads ONE LANE'S folder and nothing else: its own
    parts, its own ledger and its own `done.json`, under
    `<year>/<lane>/`. Left at None it is the unnamed lane and this writes the
    year's top-level files exactly as it always did.

    Refuses a lane the local build has not marked: an unmarked lane is a lane
    whose fetch did not finish, and pushing it would publish an over-claiming
    marker to a machine that cannot tell (§5.21).
    """
    year = int(year)
    root = store_root(work, store)
    d = year_dir(work, store, year, lane)
    what = f"{store} {year}" + (f" lane {lane}" if lane else "")
    scratch = scratch or os.path.join(root, "src", "hub")
    key = part_key(year, lane)
    if not marked(root, key):
        sys.exit(f"push refuses {what}: {root}/{key}.done is "
                 f"missing, so the fetch of that year did not finish. Refetch "
                 f"it; a marker may only under-claim (ml/CLAUDE.md §5.21).")
    names = local_part_files(d)
    if COUNTS not in names:
        sys.exit(f"push refuses {what}: no {COUNTS} in {d}")
    # SCHEMA FIRST, before a byte is uploaded: a v1 part on this prefix would
    # be pulled by a box months later and assembled into a store claiming
    # seconds it does not have (§0.3 — check the precondition where the inputs
    # are all it has cost).
    for n in names:
        if n.endswith(".npz"):
            try:
                f10.check_part_schema(os.path.join(d, n))
            except ValueError as e:
                sys.exit(f"push refuses {what}: {e}")
    led = read_json(os.path.join(d, COUNTS), {})
    entries = _entries(d, names)
    rows = int(led.get("rows", 0))
    n_parts = sum(1 for n in names if n.endswith(".npz"))
    total = sum(e["bytes"] for e in entries)

    api, repo, tok = (hub or _hub)()
    if bool(private) != str(repo).endswith("-private"):
        sys.exit(f"push refuses {what}: private={bool(private)} and "
                 f"the target repository is {repo!r} — a private store's "
                 f"parts go only to a '-private' repository, and a public "
                 f"store's never do (E-082). Nothing was uploaded.")
    api.create_repo(repo, repo_type="dataset", exist_ok=True,
                    private=bool(private))
    prefix = hub_prefix(store, year, partials, lane)

    have = read_done(api, repo, tok, store, year, scratch,
                     partials=partials, lane=lane)
    if have is not None and _by_name(have) == {e["name"]: e["sha256"]
                                               for e in entries}:
        print(f"  {what}: already on the Hub with matching hashes "
              f"({len(entries)} file(s), {rows:,} row(s)) — nothing to do")
        return 0
    lost = ledger_would_shrink(have, entries)
    if lost:
        sys.exit(shrink_refusal(what, have, lost,
                                sum(1 for e in entries
                                    if e["name"].endswith(SHARD_SUFFIX))))

    print(f"  {what}: uploading {len(entries)} file(s), "
          f"{total / 1e6:.1f} MB, {rows:,} row(s) -> {repo}:{prefix}/",
          flush=True)
    for i in range(0, len(names), BATCH):
        chunk = names[i:i + BATCH]
        _upload(api, repo,
                [(f"{prefix}/{n}", os.path.join(d, n)) for n in chunk],
                f"family 10 partials ({what}): {len(chunk)} file(s)")

    # RESTORE-VERIFY, file for file, exactly as `stage_publish` does: an upload
    # that returned 200 is not evidence the bytes are retrievable (§0.2).
    for e in entries:
        tmp = os.path.join(scratch, "verify")
        shutil.rmtree(tmp, ignore_errors=True)
        back = _download(repo, f"{prefix}/{e['name']}", tok, tmp, just_uploaded=True)
        got = sha256(back)
        shutil.rmtree(tmp, ignore_errors=True)
        if got != e["sha256"]:
            sys.exit(f"RESTORE MISMATCH {prefix}/{e['name']}: uploaded "
                     f"{e['sha256']}, downloaded {got} — the push is not "
                     f"trustworthy and no done.json was written")

    # THE MARKER IS LAST.
    done = _done_record(store, year, rows, n_parts, total, entries, lane, led)
    dp = os.path.join(scratch, f"{DONE}.{year}{('.' + lane) if lane else ''}")
    os.makedirs(scratch, exist_ok=True)
    atomic_json(dp, done)
    _upload(api, repo, [(f"{prefix}/{DONE}", dp)],
            f"family 10 partials ({what}): done")
    os.remove(dp)
    print(f"  {what}: pushed and verified — {rows:,} row(s) in "
          f"{n_parts} part(s), {total / 1e6:.1f} MB, done.json written last")
    return 0


def _done_record(store, year, rows, n_parts, total, entries, lane, led):
    """The marker. A NAMED LANE also records its own name and the window (or
    group subset) its ledger says it covers, so the folder on the Hub says
    what it is without anybody having to open a part. The unnamed lane's
    record is byte for byte the one every year on the Hub already carries."""
    done = {"store": store, "year": year, "rows": rows, "n_parts": n_parts,
            "bytes": total, "files": entries,
            "builder_git_sha": git_sha(), "at": utcnow()}
    if lane:
        done["lane"] = lane
        for k in ("lane_window", "lane_groups"):
            if (led or {}).get(k) is not None:
                done[k] = led[k]
    return done


# Across YEARS, one commit carries at most this many files or this many bytes.
# E-082's lanes park hundreds of years (ghcnd 1763->, icoads 1662->) and the
# early ones are a few kB each, so `push`'s two commits a year would spend the
# Hub's 256 commits an hour on a single lane.
MANY_FILES = 400
MANY_BYTES = 4 * 1024 ** 3


def push_many(store, years, work, scratch=None, partials=None, hub=None,
              private=False, max_files=MANY_FILES, max_bytes=MANY_BYTES,
              lane=None):
    """`push` for a whole lane: every year's parts in as few commits as the
    batch allows, every file restore-verified, and THEN every year's
    `done.json` in one final commit (per `max_files`).

    The contract is `push`'s, year for year: an unmarked year is refused
    before any upload, a year already on the Hub with matching hashes is
    skipped, and no `done.json` is written unless every file of every year in
    this call came back with its sha256. A failure anywhere leaves no marker
    for any year of the call — the lane is re-run, and years whose parts did
    land are cheap to re-push. Returns the list of years now marked on the Hub.

    `lane` (E-082 wave 7) is ONE NAME for the whole call, because a lane is
    one fetch of one window: a monthly lane of 2022 pushes `2022/m06/`, and a
    lane whose window crosses New Year pushes `<year>/<lane>/` for each year
    it touched. Left at None every year's top-level folder is written, which
    is what every year on the Hub already holds.
    """
    years = [int(y) for y in years]
    todo = []
    for year in years:
        root = store_root(work, store)
        d = year_dir(work, store, year, lane)
        what = f"{store} {year}" + (f" lane {lane}" if lane else "")
        key = part_key(year, lane)
        if not marked(root, key):
            sys.exit(f"push refuses {what}: {root}/{key}.done "
                     f"is missing, so the fetch of that year did not finish. "
                     f"Refetch it; a marker may only under-claim "
                     f"(ml/CLAUDE.md §5.21).")
        names = local_part_files(d)
        if COUNTS not in names:
            sys.exit(f"push refuses {what}: no {COUNTS} in {d}")
        for n in names:
            if n.endswith(".npz"):
                try:
                    f10.check_part_schema(os.path.join(d, n))
                except ValueError as e:
                    sys.exit(f"push refuses {what}: {e}")
        todo.append((year, d, names, _entries(d, names)))
    if not todo:
        return []
    scratch = scratch or os.path.join(store_root(work, store), "src", "hub")
    api, repo, tok = (hub or _hub)()
    if bool(private) != str(repo).endswith("-private"):
        sys.exit(f"push refuses {store}: private={bool(private)} and the "
                 f"target repository is {repo!r} — a private store's parts go "
                 f"only to a '-private' repository, and a public store's "
                 f"never do (E-082). Nothing was uploaded.")
    api.create_repo(repo, repo_type="dataset", exist_ok=True,
                    private=bool(private))
    listing = _list_files(api, repo, hub_prefix(store, None, partials))
    pending, skipped, refused = [], [], []
    for year, d, names, entries in todo:
        have = read_done(api, repo, tok, store, year, scratch, listing,
                         partials=partials, lane=lane)
        if have is not None and _by_name(have) == {e["name"]: e["sha256"]
                                                   for e in entries}:
            skipped.append(year)
            continue
        lost = ledger_would_shrink(have, entries)
        if lost:
            what = f"{store} {year}" + (f" lane {lane}" if lane else "")
            refused.append(shrink_refusal(
                what, have, lost,
                sum(1 for e in entries if e["name"].endswith(SHARD_SUFFIX))))
            continue
        pending.append((year, d, names, entries))
    # BEFORE ANY BYTE: a refused year's shard files would otherwise be
    # overwritten by this call's parts commit even though its marker is not.
    if refused:
        sys.exit("\n".join(refused))
    if skipped:
        print(f"  {store}: {len(skipped)} year(s) already on the Hub with "
              f"matching hashes — skipped")
    # the parts, batched across years
    files = [(f"{hub_prefix(store, y, partials, lane)}/{e['name']}",
              os.path.join(d, e["name"]), e["bytes"], y)
             for y, d, _n, ents in pending for e in ents]
    total = sum(f[2] for f in files)
    print(f"  {store}: uploading {len(files)} file(s) of {len(pending)} "
          f"year(s), {total / 1e6:.1f} MB -> {repo}:"
          f"{hub_prefix(store, None, partials)}/", flush=True)
    batch, size, k = [], 0, 0

    def flush():
        nonlocal batch, size, k
        if not batch:
            return
        k += 1
        ys = sorted({f[3] for f in batch})
        _upload(api, repo, [(f[0], f[1]) for f in batch],
                f"family 1 partials ({store} {ys[0]}-{ys[-1]}): "
                f"{len(batch)} file(s), batch {k}")
        batch, size = [], 0
    for f in files:
        if batch and (len(batch) >= max_files or size + f[2] > max_bytes):
            flush()
        batch.append(f)
        size += f[2]
    flush()
    # restore-verify every file before any marker
    for (rel, _local, _b, _y), sha in zip(
            files, [e["sha256"] for _y, _d, _n, ents in pending
                    for e in ents]):
        tmp = os.path.join(scratch, "verify")
        shutil.rmtree(tmp, ignore_errors=True)
        back = _download(repo, rel, tok, tmp, just_uploaded=True)
        got = sha256(back)
        shutil.rmtree(tmp, ignore_errors=True)
        if got != sha:
            sys.exit(f"RESTORE MISMATCH {rel}: uploaded {sha}, downloaded "
                     f"{got} — the push is not trustworthy and no done.json "
                     f"was written for any year of this call")
    # THE MARKERS ARE LAST
    os.makedirs(scratch, exist_ok=True)
    marks = []
    for year, d, names, entries in pending:
        led = read_json(os.path.join(d, COUNTS), {})
        done = _done_record(
            store, year, int(led.get("rows", 0)),
            sum(1 for n in names if n.endswith(".npz")),
            sum(e["bytes"] for e in entries), entries, lane, led)
        dp = os.path.join(scratch,
                          f"{DONE}.{year}{('.' + lane) if lane else ''}")
        atomic_json(dp, done)
        marks.append((f"{hub_prefix(store, year, partials, lane)}/{DONE}", dp))
    for i in range(0, len(marks), max_files):
        chunk = marks[i:i + max_files]
        _upload(api, repo, chunk,
                f"family 1 partials ({store}): done.json for "
                f"{len(chunk)} year(s)")
    for _rel, dp in marks:
        os.remove(dp)
    print(f"  {store}: {len(pending)} year(s) pushed and verified in {k} "
          f"part commit(s), done.json written last "
          f"({len(skipped)} already present)")
    return sorted(skipped + [p[0] for p in pending])


# ==================================================================== pull ===
def pull(store, years, work, allow_missing=False, scratch=None,
         partials=None, hub=None, private=False):
    """Bring every requested year's parts down and mark each one locally.

    Returns (present, missing). A year already on disk whose marker stands and
    whose hashes match the Hub's `done.json` is skipped without downloading.

    EVERY LANE OF THE YEAR COMES BACK (E-082 wave 7): the repository listing
    says which lanes are done (`hub_lanes`), each is downloaded into its own
    folder and marked on its own, and a year with no lane at all is missing.
    A year that was pushed as one unnamed lane — every year on the Hub before
    this — has exactly one lane, so this is the loop it always was with one
    more level in it. Whether those lanes ADD UP to the year is the
    assembler's question, not this one's (`lanes_preflight`).
    """
    years = [int(y) for y in years]
    root = store_root(work, store)
    os.makedirs(root, exist_ok=True)
    scratch = scratch or os.path.join(root, "src", "hub")
    api, repo, tok = (hub or _hub)()
    listing = _list_files(api, repo, hub_prefix(store, None, partials))
    present, missing, skipped, rows_total, n_lanes = [], [], 0, 0, 0
    for y in years:
        lanes = hub_lanes(listing, store, y, partials)
        if not lanes:
            missing.append(y)
            continue
        for lane in lanes:
            n_lanes += 1
            what = f"{store} {y}" + (f" lane {lane}" if lane else "")
            done = read_done(api, repo, tok, store, y, scratch, listing,
                             partials=partials, lane=lane)
            if done is None:                              # pragma: no cover
                missing.append(y)
                break
            entries = done.get("files") or []
            d = year_dir(work, store, y, lane)
            if marked(root, part_key(y, lane)) and _local_matches(d, entries):
                rows_total += int(done.get("rows", 0))
                skipped += 1
                continue
            # Download into a scratch dir, verify, and only then move the lane
            # into place and write the marker. A half-downloaded lane must
            # never be marked (§5.21).
            suffix = ("_" + lane) if lane else ""
            tmp = os.path.join(scratch, f"pull_{y}{suffix}")
            shutil.rmtree(tmp, ignore_errors=True)
            os.makedirs(tmp, exist_ok=True)
            # THE LANE'S PARTS COME DOWN IN PARALLEL (`PULL_WORKERS`), each
            # worker downloading and hashing one; results arrive in done.json
            # order, and a worker's exception (a 4xx, or a retry ladder that
            # ran out) is re-raised here and fails the pull.
            pre = hub_prefix(store, y, partials, lane)
            dl = os.path.join(tmp, "dl")

            def fetch(e, pre=pre, dl=dl):
                p = _download(repo, f"{pre}/{e['name']}", tok, dl)
                return e["name"], p, sha256(p)

            got = []
            results = _ordered_map(fetch, entries, PULL_WORKERS)
            try:
                for i, (n, p, h) in enumerate(results):
                    e = entries[i]
                    if h != e["sha256"]:
                        sys.exit(f"PULL MISMATCH {what} {e['name']}: "
                                 f"done.json says {e['sha256']}, the Hub "
                                 f"served {h}")
                    got.append((n, p))
            finally:
                results.close()      # cancel what is queued, join the rest
            # THE SCHEMA IS CHECKED ON WHAT ARRIVED, not on what the marker
            # says. `done.json` records names, bytes and sha256 and knows
            # nothing about the column layout inside a part, so a year pushed
            # by an older builder verifies perfectly and is still unusable.
            # Refusing here, before the lane is moved into place and marked,
            # keeps it a lane that is simply not present rather than one the
            # assembler has to discover.
            for n, p in got:
                if n.endswith(".npz"):
                    try:
                        f10.check_part_schema(p)
                    except ValueError as ex:
                        sys.exit(f"pull refuses {what}: {ex}")
            os.makedirs(d, exist_ok=True)
            for n, p in got:
                os.replace(p, os.path.join(d, n))
            shutil.rmtree(tmp, ignore_errors=True)
            mark(root, part_key(y, lane))
            rows_total += int(done.get("rows", 0))
            print(f"  {what}: {len(entries)} file(s), "
                  f"{int(done.get('rows', 0)):,} row(s) -> {d}")
        else:
            present.append(y)
    print(f"  pull: {len(present)} year(s) present in {n_lanes} lane(s) "
          f"({skipped} already local), "
          f"{rows_total:,} row(s), {len(missing)} missing")
    if missing:
        msg = (f"{len(missing)} year(s) have no {DONE} under "
               f"{hub_prefix(store, None, partials)}/: "
               f"{', '.join(str(y) for y in missing)}")
        if not allow_missing:
            sys.exit(f"pull refuses: {msg}. Those years were never fetched (or "
                     f"their push did not finish) — run the fetch lane for "
                     f"them, or pass --allow-missing to assemble a store that "
                     f"is deliberately short.")
        print(f"::warning::{msg} — continuing on --allow-missing")
    return present, missing


def _local_matches(d, entries):
    names = set(local_part_files(d))
    if names != {e["name"] for e in entries}:
        return False
    for e in entries:
        p = os.path.join(d, e["name"])
        if os.path.getsize(p) != e["bytes"] or sha256(p) != e["sha256"]:
            return False
    return True


# ================================================================== status ===
def status(store, years=None, scratch=None, partials=None, hub=None):
    """Which years are DONE on the Hub, with their rows and bytes."""
    api, repo, tok = (hub or _hub)()
    listing = _list_files(api, repo, hub_prefix(store, None, partials))
    pre = hub_prefix(store, None, partials) + "/"
    found = sorted({int(p[len(pre):].split("/")[0])
                    for p in listing
                    if p.endswith("/" + DONE)
                    and p[len(pre):].split("/")[0].isdigit()})
    if years:
        want = {int(y) for y in years}
        found = [y for y in found if y in want]
    scratch = scratch or os.path.join(os.path.abspath("."), ".f10status")
    rows_total = bytes_total = 0
    for y in found:
        # A YEAR IS THE SUM OF ITS LANES (E-082 wave 7). A year pushed as one
        # unnamed lane has exactly one, so this prints what it always printed.
        for lane in hub_lanes(listing, store, y, partials):
            done = read_done(api, repo, tok, store, y, scratch, listing,
                             partials=partials, lane=lane) or {}
            r, b = int(done.get("rows", 0)), int(done.get("bytes", 0))
            rows_total += r
            bytes_total += b
            print(f"  {store} {y}{(' ' + lane) if lane else ''}: rows={r:,} "
                  f"parts={done.get('n_parts')} "
                  f"bytes={b / 1e6:.1f}M at={done.get('at')}")
    shutil.rmtree(scratch, ignore_errors=True)
    # The machine-readable line the fetch workflow greps. Keep the prefix and
    # the space separation: `.github/workflows/family10-slatrack-fetch.yml`
    # parses it to decide which years to skip.
    print(f"done_years: {' '.join(str(y) for y in found)}")
    print(f"  total: {len(found)} year(s), {rows_total:,} row(s), "
          f"{bytes_total / 1e9:.2f} GB")
    if years:
        miss = sorted(set(int(y) for y in years) - set(found))
        print(f"missing_years: {' '.join(str(y) for y in miss)}")
    return found


# ============================================ files over the Hub's size limit ==
# The Hub takes at most 50 GB per file. Measured 2026-09-24 02:50Z on
# family1-build #680: the swot whole-record point store (~9e9 rows) assembled
# fine and its publish was refused — `platform.npy` 72.99 GB, `time_s.npy`
# 73 GB, `values.npy` ~55 GB. So `build_family10_stores.stage_publish` uploads
# a store file larger than `HUB_SPLIT_BYTES` as consecutive byte ranges
# `<name>.part000`, `<name>.part001`, … and records them in store.json:
#
#   "hub_split": {"<name>": {"bytes": <whole file>, "chunk_bytes": <split>,
#                            "parts": [{"name": "<name>.part000",
#                                       "bytes": n, "sha256": h}, …]}}
#
# while `sha256[<name>]` stays the WHOLE file's digest. The helpers below are
# the one definition of that layout: the part names, the byte ranges, the
# validation of an entry, the streamed restore and the reader.
SPLIT_BUF = 1 << 22            # 4 MiB reads while copying and hashing a part
_PART_RE = re.compile(r"\.part(\d{3})$")


class SplitError(ValueError):
    """A split file whose parts do not reassemble into the file store.json
    records. The message names the part."""


def split_part_name(name, i):
    """`values.npy`, 0 -> `values.npy.part000`."""
    return f"{name}.part{int(i):03d}"


def split_ranges(total, chunk):
    """[(offset, length)] of consecutive byte ranges covering `total`, each
    at most `chunk`; only the last may be short."""
    total, chunk = int(total), int(chunk)
    if chunk <= 0:
        raise ValueError(f"split chunk must be positive, got {chunk}")
    return [(o, min(chunk, total - o)) for o in range(0, total, chunk)]


def write_split_part(src, offset, length, dst, buf=SPLIT_BUF):
    """Copy bytes [offset, offset + length) of `src` into `dst` (temp sibling
    + os.replace) and return their sha256."""
    tmp = f"{dst}.tmp{os.getpid()}"
    h = hashlib.sha256()
    left = int(length)
    try:
        with open(src, "rb") as fi, open(tmp, "wb") as fo:
            fi.seek(int(offset))
            while left:
                blk = fi.read(min(buf, left))
                if not blk:
                    raise SplitError(f"{src} ended {left} byte(s) short of the "
                                     f"range {offset}+{length}")
                h.update(blk)
                fo.write(blk)
                left -= len(blk)
        os.replace(tmp, dst)
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)
    return h.hexdigest()


def split_parts(name, entry):
    """Validate ONE `hub_split[name]` entry and return its part list.

    The parts must be named `<name>.part000` onward with no gap, each carry a
    64-hex sha256 and a positive size no larger than `chunk_bytes`, every part
    but the last be exactly `chunk_bytes`, and the sizes sum to `bytes`.
    Anything else is an entry no reader can reassemble, and is refused here
    rather than discovered as a wrong file."""
    entry = entry or {}
    parts = list(entry.get("parts") or [])
    if not parts:
        raise SplitError(f"hub_split[{name!r}] lists no parts")
    total = int(entry.get("bytes", -1))
    chunk = int(entry.get("chunk_bytes", 0) or 0)
    for i, p in enumerate(parts):
        want = split_part_name(name, i)
        if p.get("name") != want:
            raise SplitError(f"hub_split[{name!r}] part {i} is named "
                             f"{p.get('name')!r}, expected {want!r}")
        h = p.get("sha256")
        if not (isinstance(h, str) and len(h) == 64):
            raise SplitError(f"{want}: hub_split carries no sha256")
        b = int(p.get("bytes", -1))
        if b <= 0 or (chunk and b > chunk) or \
                (chunk and i < len(parts) - 1 and b != chunk):
            raise SplitError(f"{want}: {b} bytes against chunk_bytes {chunk} "
                             f"(every part but the last is exactly one chunk)")
    if sum(int(p["bytes"]) for p in parts) != total:
        raise SplitError(f"hub_split[{name!r}]: the parts sum to "
                         f"{sum(int(p['bytes']) for p in parts)} bytes, the "
                         f"entry records {total}")
    return parts


def stream_split(repo, prefix, name, entry, token, scratch, sink=None,
                 just_uploaded=False, private=False):
    """Fetch every part of a split file IN ORDER, one at a time, check each
    part's size and sha256 against `entry`, feed its bytes through ONE
    whole-file sha256 (and into `sink`, a writable binary file, when given)
    and delete it before the next. Returns the concatenation's sha256.

    The parts are STREAMED (`hub_stream`), so the restore of a 73 GB column
    needs no disk at all; `scratch` is kept for the signature. Raises
    SplitError naming the first part that disagrees."""
    parts = split_parts(name, entry)
    whole = hashlib.sha256()
    for p in parts:
        # streamed, never written to `scratch`: each part's bytes go through
        # the part hash, the whole-file hash and `sink` as they arrive
        # (`hub_stream` — the xet download path crawled on these objects)
        h = hashlib.sha256()

        def feed(blk, h=h):
            h.update(blk)
            whole.update(blk)
            if sink is not None:
                sink.write(blk)
        n = hub_stream(repo, f"{prefix}/{p['name']}", token, feed,
                       just_uploaded=just_uploaded, private=private)
        got = h.hexdigest()
        if n != int(p["bytes"]) or got != p["sha256"]:
            raise SplitError(
                f"{p['name']}: downloaded {n} bytes sha256 {got}, "
                f"store.json's hub_split records {p['bytes']} bytes "
                f"sha256 {p['sha256']}")
    return whole.hexdigest()


def hub_split_download(repo, prefix, name, meta, token, dest_dir):
    """Fetch store file `name` under `prefix` into ONE local file
    `dest_dir/name`, whether the Hub holds it whole or as `hub_split` parts.

    `meta` is the store's store.json (a dict). A split file is reassembled by
    streaming its parts in order into a temp sibling — each part checked
    against its own sha256 and deleted as soon as it is appended — and the
    concatenation's sha256 must equal `meta["sha256"][name]` before it is
    renamed into place. A whole file is checked against the same record
    when store.json carries one. Returns the local path; raises SplitError
    (and leaves nothing at `dest_dir/name`) on any disagreement."""
    os.makedirs(dest_dir, exist_ok=True)
    out = os.path.join(dest_dir, name)
    want = (meta.get("sha256") or {}).get(name)
    entry = (meta.get("hub_split") or {}).get(name)
    tmp_dir = os.path.join(dest_dir, f".{name}.fetch")
    partial = f"{out}.partial{os.getpid()}"
    shutil.rmtree(tmp_dir, ignore_errors=True)
    try:
        if entry is None:
            got = _download(repo, f"{prefix}/{name}", token, tmp_dir)
            digest = sha256(got) if want else None
            if want and digest != want:
                raise SplitError(f"{name}: downloaded sha256 {digest}, "
                                 f"store.json records {want}")
            os.replace(got, out)
            return out
        if not want:
            raise SplitError(f"{name} is split on the Hub and store.json "
                             f"carries no sha256 for the whole file — the "
                             f"reassembly cannot be checked")
        with open(partial, "wb") as fo:
            digest = stream_split(repo, prefix, name, entry, token, tmp_dir,
                                  sink=fo)
        if digest != want:
            raise SplitError(f"{name}: its {len(entry['parts'])} part(s) "
                             f"concatenate to sha256 {digest}, store.json "
                             f"records {want} for the whole file")
        os.replace(partial, out)
        return out
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)
        if os.path.exists(partial):
            os.remove(partial)


def _tree(api, repo, prefix):
    """The FILES directly under `prefix`: {name: {"size", "sha256"}}, where
    `sha256` is the LFS object's digest (None for a file the Hub does not
    report one for). One listing of one folder — never the whole repository.
    The seam the tests replace."""
    out = {}
    for it in api.list_repo_tree(repo, path_in_repo=prefix,
                                 repo_type="dataset", recursive=False):
        if type(it).__name__ == "RepoFolder":
            continue
        size = getattr(it, "size", None)
        if size is None:
            continue
        lfs = getattr(it, "lfs", None)
        out[it.path.rsplit("/", 1)[-1]] = {
            "size": int(size),
            "sha256": getattr(lfs, "sha256", None) if lfs else None}
    return out


def _parts_of(tree, name):
    return sorted(k for k in tree if k.startswith(name + ".part")
                  and _PART_RE.search(k[len(name):]))


def split_disagreements(tree, hub_split):
    """The Hub folder listing against `hub_split`: a list of problems, empty
    when every split file is present AS ALL ITS PARTS — sizes equal, and
    sha256 equal wherever the Hub reports one — with no whole copy of it and
    no stray part beside them (either would be a second answer to "what is
    `<name>`" on the Hub)."""
    bad = []
    for name, entry in sorted((hub_split or {}).items()):
        try:
            parts = split_parts(name, entry)
        except SplitError as e:
            bad.append(str(e))
            continue
        for p in parts:
            got = tree.get(p["name"])
            if got is None:
                bad.append(f"{p['name']}: not on the Hub")
            elif got["size"] != int(p["bytes"]):
                bad.append(f"{p['name']}: {got['size']} bytes on the Hub, "
                           f"hub_split records {p['bytes']}")
            elif got.get("sha256") and got["sha256"] != p["sha256"]:
                bad.append(f"{p['name']}: sha256 {got['sha256']} on the Hub, "
                           f"hub_split records {p['sha256']}")
        if name in tree:
            bad.append(f"{name}: a WHOLE copy sits on the Hub beside its parts")
        want = {p["name"] for p in parts}
        stray = [k for k in _parts_of(tree, name) if k not in want]
        if stray:
            bad.append(f"{name}: stray part(s) on the Hub that hub_split does "
                       f"not list: {stray[:6]}")
    return bad


def stale_split_paths(tree, names, hub_split):
    """Names under the prefix a publish of THIS layout must delete: the whole
    copy of a file that is now split, and every `<name>.partNNN` that
    `hub_split` does not list (all of them, for a file no longer split)."""
    out = []
    for n in names:
        entry = (hub_split or {}).get(n)
        want = {p["name"] for p in entry["parts"]} if entry else set()
        if entry and n in tree:
            out.append(n)
        out += [k for k in _parts_of(tree, n) if k not in want]
    return out


# =================================================================== driver ==
def main(argv=None):
    ap = argparse.ArgumentParser(
        description=f"Park a family-{f10.FAMILY_VERSION} store's per-year "
                    f"column parts on the Hub under "
                    f"{HF_PARTIALS}/<store>/<year>/, so the credentialed fetch "
                    f"and the big assembly can happen on different machines. "
                    f"A part carrying the v1 `time_days` column is refused by "
                    f"both push and pull. See the module docstring.")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("push", help="upload ONE fetched year, verify by "
                                    "restore, then write done.json")
    p.add_argument("--store", required=True)
    p.add_argument("--year", required=True, type=int)
    p.add_argument("--work", required=True)
    p.add_argument("--lane", default="",
                   help="the LANE to push (E-082 wave 7): its folder alone "
                        "goes to <year>/<lane>/. Empty is the unnamed lane — "
                        "the whole year at the top of the year folder, which "
                        "is what every year on the Hub holds")

    p = sub.add_parser("pull", help="download every requested year's parts "
                                    "and write the local markers (every LANE "
                                    "of every year)")
    p.add_argument("--store", required=True)
    p.add_argument("--years", required=True,
                   help="1993-2024 / 2003,2007 / 1997-1999,2004")
    p.add_argument("--work", required=True)
    p.add_argument("--allow-missing", action="store_true",
                   help="do not refuse when a year has no done.json — the "
                        "store will be short and store.json will show it")

    p = sub.add_parser("status", help="which years are done on the Hub")
    p.add_argument("--store", required=True)
    p.add_argument("--years", default="")

    a = ap.parse_args(argv)
    if a.cmd == "push":
        return push(a.store, a.year, a.work, lane=a.lane or None)
    if a.cmd == "pull":
        pull(a.store, parse_years(a.years), a.work,
             allow_missing=a.allow_missing)
        return 0
    status(a.store, parse_years(a.years) if a.years else None)
    return 0


if __name__ == "__main__":
    sys.exit(main())

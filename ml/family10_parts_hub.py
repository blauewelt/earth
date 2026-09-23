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
import json
import os
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


def transient_download_error(e):
    """A Hub read that may succeed on retry: a dropped connection, a
    timeout, a TLS handshake that timed out, a 5xx or a 429 — never a 4xx
    that names OUR request (a 401 or a 404 will not change by asking again).

    `hf_hub_download` reports a HEAD that could not reach the Hub as
    `LocalEntryNotFoundError` ("…we cannot find the requested files in the
    local cache"), an OSError; that is a connection failure, not an answer.
    """
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


def _download(repo, path_in_repo, token, dest_dir):
    """Stream one repo file into `dest_dir`; returns the local path.

    WITH A RETRY LADDER FOR TRANSIENT NETWORK FAILURES, the one
    `build_family1_stores._download` got in 300905a. lst05's box assembly
    (family1-build #425, 2026-09-21) was pulling thousands of parked parts
    when one GET died on `_ssl.c:989: The handshake operation timed out`
    (`2021/terra__bin_2859.idx.npy`); a pull is a loop of thousands of
    requests, so one of them failing transiently is the expected case, and
    it must cost a few seconds of backoff rather than the job. Six attempts,
    ~8 minutes of backoff at most; a 4xx is raised at once.
    """
    from huggingface_hub import hf_hub_download
    os.makedirs(dest_dir, exist_ok=True)
    last = None
    for i in range(DOWNLOAD_ATTEMPTS):
        try:
            return hf_hub_download(repo, path_in_repo, repo_type="dataset",
                                   token=token, local_dir=dest_dir)
        except Exception as e:                          # noqa: BLE001
            if not transient_download_error(e) or i == DOWNLOAD_ATTEMPTS - 1:
                raise
            last = e
            wait = DOWNLOAD_BACKOFF_S[min(i, len(DOWNLOAD_BACKOFF_S) - 1)]
            print(f"::warning::{path_in_repo}: {type(e).__name__}: "
                  f"{str(e)[:160]} — attempt {i + 1}/{DOWNLOAD_ATTEMPTS}, "
                  f"retrying in {wait}s", flush=True)
            time.sleep(wait)
    raise last                                          # pragma: no cover


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
        back = _download(repo, f"{prefix}/{e['name']}", tok, tmp)
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
        back = _download(repo, rel, tok, tmp)
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
            got = []
            for e in entries:
                p = _download(
                    repo,
                    f"{hub_prefix(store, y, partials, lane)}/{e['name']}", tok,
                    os.path.join(tmp, "dl"))
                h = sha256(p)
                if h != e["sha256"]:
                    sys.exit(f"PULL MISMATCH {what} {e['name']}: done.json "
                             f"says {e['sha256']}, the Hub served {h}")
                got.append((e["name"], p))
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

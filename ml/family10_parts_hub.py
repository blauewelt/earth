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


def _download(repo, path_in_repo, token, dest_dir):
    """Stream one repo file into `dest_dir`; returns the local path."""
    from huggingface_hub import hf_hub_download
    os.makedirs(dest_dir, exist_ok=True)
    return hf_hub_download(repo, path_in_repo, repo_type="dataset",
                           token=token, local_dir=dest_dir)


# --------------------------------------------------------------- the paths --
def hub_prefix(store, year=None, partials=None):
    """`partials/<family>/<store>[/<year>]`. `partials` defaults to family
    10's prefix; family 1 passes its own (E-082)."""
    p = f"{partials or HF_PARTIALS}/{store}"
    return p if year is None else f"{p}/{year}"


def store_root(work, store):
    return os.path.join(os.path.abspath(work), store)


def year_dir(work, store, year):
    return os.path.join(store_root(work, store), "parts", str(year))


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


def read_done(api, repo, tok, store, year, scratch, listing=None,
              partials=None):
    """The year's `done.json` off the Hub, or None if it is NOT THERE.

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
    path = f"{hub_prefix(store, year, partials)}/{DONE}"
    if listing is not None and path not in listing:
        return None
    tmp = os.path.join(scratch, f"done_{year}")
    shutil.rmtree(tmp, ignore_errors=True)
    try:
        p = _download(repo, path, tok, tmp)
        got = read_json(p, None)
        if got is None:
            raise IOError(f"{repo}:{path} downloaded but does not parse as "
                          f"JSON — the marker for {store} {year} is corrupt")
        return got
    except Exception as e:                                    # noqa: BLE001
        if listing is None and _looks_absent(e):
            return None
        raise IOError(
            f"cannot read {repo}:{path}, the marker for {store} {year}: "
            f"{type(e).__name__}: {e}. The path IS in the repository listing, "
            f"so this is the Hub refusing to serve it (an outage, a rate "
            f"limit, or an HF_TOKEN without read access) and NOT a year that "
            f"was never fetched — refusing to report it as missing, because "
            f"--allow-missing would then drop a year whose parts are on the "
            f"Hub.") from e
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def _looks_absent(exc):
    """Is this exception the Hub saying 404, rather than saying no?"""
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
         private=False):
    """Upload one fetched year's parts, verify by restore, THEN mark it done.

    E-082: `partials` (the prefix), `hub` (a callable returning (api, repo,
    token)) and `private` route a family-1 store; left at None/False they
    are family 10's module defaults, and the `_hub` seam the tests replace.

    Refuses a year the local build has not marked: an unmarked year is a year
    whose fetch did not finish, and pushing it would publish an over-claiming
    marker to a machine that cannot tell (§5.21).
    """
    year = int(year)
    root = store_root(work, store)
    d = year_dir(work, store, year)
    scratch = scratch or os.path.join(root, "src", "hub")
    if not marked(root, f"parts/{year}"):
        sys.exit(f"push refuses {store} {year}: {root}/parts/{year}.done is "
                 f"missing, so the fetch of that year did not finish. Refetch "
                 f"it; a marker may only under-claim (ml/CLAUDE.md §5.21).")
    names = local_part_files(d)
    if COUNTS not in names:
        sys.exit(f"push refuses {store} {year}: no {COUNTS} in {d}")
    # SCHEMA FIRST, before a byte is uploaded: a v1 part on this prefix would
    # be pulled by a box months later and assembled into a store claiming
    # seconds it does not have (§0.3 — check the precondition where the inputs
    # are all it has cost).
    for n in names:
        if n.endswith(".npz"):
            try:
                f10.check_part_schema(os.path.join(d, n))
            except ValueError as e:
                sys.exit(f"push refuses {store} {year}: {e}")
    entries = _entries(d, names)
    rows = int(read_json(os.path.join(d, COUNTS), {}).get("rows", 0))
    n_parts = sum(1 for n in names if n.endswith(".npz"))
    total = sum(e["bytes"] for e in entries)

    api, repo, tok = (hub or _hub)()
    if bool(private) != str(repo).endswith("-private"):
        sys.exit(f"push refuses {store} {year}: private={bool(private)} and "
                 f"the target repository is {repo!r} — a private store's "
                 f"parts go only to a '-private' repository, and a public "
                 f"store's never do (E-082). Nothing was uploaded.")
    api.create_repo(repo, repo_type="dataset", exist_ok=True,
                    private=bool(private))
    prefix = hub_prefix(store, year, partials)

    have = read_done(api, repo, tok, store, year, scratch,
                     partials=partials)
    if have is not None and _by_name(have) == {e["name"]: e["sha256"]
                                               for e in entries}:
        print(f"  {store} {year}: already on the Hub with matching hashes "
              f"({len(entries)} file(s), {rows:,} row(s)) — nothing to do")
        return 0

    print(f"  {store} {year}: uploading {len(entries)} file(s), "
          f"{total / 1e6:.1f} MB, {rows:,} row(s) -> {repo}:{prefix}/",
          flush=True)
    for i in range(0, len(names), BATCH):
        chunk = names[i:i + BATCH]
        _upload(api, repo,
                [(f"{prefix}/{n}", os.path.join(d, n)) for n in chunk],
                f"family 10 partials ({store} {year}): {len(chunk)} file(s)")

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
    done = {"store": store, "year": year, "rows": rows, "n_parts": n_parts,
            "bytes": total, "files": entries,
            "builder_git_sha": git_sha(), "at": utcnow()}
    dp = os.path.join(scratch, f"{DONE}.{year}")
    os.makedirs(scratch, exist_ok=True)
    atomic_json(dp, done)
    _upload(api, repo, [(f"{prefix}/{DONE}", dp)],
            f"family 10 partials ({store} {year}): done")
    os.remove(dp)
    print(f"  {store} {year}: pushed and verified — {rows:,} row(s) in "
          f"{n_parts} part(s), {total / 1e6:.1f} MB, done.json written last")
    return 0


# ==================================================================== pull ===
def pull(store, years, work, allow_missing=False, scratch=None,
         partials=None, hub=None, private=False):
    """Bring every requested year's parts down and mark each one locally.

    Returns (present, missing). A year already on disk whose marker stands and
    whose hashes match the Hub's `done.json` is skipped without downloading.
    """
    years = [int(y) for y in years]
    root = store_root(work, store)
    os.makedirs(root, exist_ok=True)
    scratch = scratch or os.path.join(root, "src", "hub")
    api, repo, tok = (hub or _hub)()
    listing = _list_files(api, repo, hub_prefix(store, None, partials))
    present, missing, skipped, rows_total = [], [], 0, 0
    for y in years:
        done = read_done(api, repo, tok, store, y, scratch, listing,
                         partials=partials)
        if done is None:
            missing.append(y)
            continue
        entries = done.get("files") or []
        d = year_dir(work, store, y)
        if marked(root, f"parts/{y}") and _local_matches(d, entries):
            present.append(y)
            rows_total += int(done.get("rows", 0))
            skipped += 1
            continue
        # Download into a scratch dir, verify, and only then move the year into
        # place and write the marker. A half-downloaded year must never be
        # marked (§5.21).
        tmp = os.path.join(scratch, f"pull_{y}")
        shutil.rmtree(tmp, ignore_errors=True)
        os.makedirs(tmp, exist_ok=True)
        got = []
        for e in entries:
            p = _download(repo,
                          f"{hub_prefix(store, y, partials)}/{e['name']}", tok,
                          os.path.join(tmp, "dl"))
            h = sha256(p)
            if h != e["sha256"]:
                sys.exit(f"PULL MISMATCH {store} {y} {e['name']}: done.json "
                         f"says {e['sha256']}, the Hub served {h}")
            got.append((e["name"], p))
        # THE SCHEMA IS CHECKED ON WHAT ARRIVED, not on what the marker says.
        # `done.json` records names, bytes and sha256 and knows nothing about
        # the column layout inside a part, so a year pushed by an older builder
        # verifies perfectly and is still unusable. Refusing here, before the
        # year is moved into place and marked, keeps it a year that is simply
        # not present rather than one the assembler has to discover.
        for n, p in got:
            if n.endswith(".npz"):
                try:
                    f10.check_part_schema(p)
                except ValueError as ex:
                    sys.exit(f"pull refuses {store} {y}: {ex}")
        os.makedirs(d, exist_ok=True)
        for n, p in got:
            os.replace(p, os.path.join(d, n))
        shutil.rmtree(tmp, ignore_errors=True)
        mark(root, f"parts/{y}")
        present.append(y)
        rows_total += int(done.get("rows", 0))
        print(f"  {store} {y}: {len(entries)} file(s), "
              f"{int(done.get('rows', 0)):,} row(s) -> {d}")
    print(f"  pull: {len(present)} year(s) present ({skipped} already local), "
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
        done = read_done(api, repo, tok, store, y, scratch, listing,
                         partials=partials) or {}
        r, b = int(done.get("rows", 0)), int(done.get("bytes", 0))
        rows_total += r
        bytes_total += b
        print(f"  {store} {y}: rows={r:,} parts={done.get('n_parts')} "
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

    p = sub.add_parser("pull", help="download every requested year's parts "
                                    "and write the local markers")
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
        return push(a.store, a.year, a.work)
    if a.cmd == "pull":
        pull(a.store, parse_years(a.years), a.work,
             allow_missing=a.allow_missing)
        return 0
    status(a.store, parse_years(a.years) if a.years else None)
    return 0


if __name__ == "__main__":
    sys.exit(main())

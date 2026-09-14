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

THE THREDDS MIRROR IS NOT THE ANSWER, and that is why this script exists at
all: `psl.noaa.gov/thredds/fileServer/...` returns TRUNCATED files under load
(measured 2026-09-04), which is the one failure mode a silent one -- the
symptom is `NetCDF: HDF error` at open time, a whole year of the axis gone.

THE RULE THIS SCRIPT ENFORCES, from `ml/hf_mirror.py`: **a backup is only real
if the restore works.** Every file is size-verified against PSL's own
Content-Length, uploaded, then DOWNLOADED BACK and sha256-compared against the
local bytes before either copy is deleted. An upload that returns 200 is not
evidence the bytes are retrievable (ml/CLAUDE.md §0.2 — assert the effect, not
the invocation), and a mirrored file that does not round-trip is WORSE than an
absent one, because the builder would trust it: such a file is DELETED from
the Hub and counted as failed. The job continues through the rest of the list
and exits non-zero at the end if anything failed.

DISK. A hosted runner has ~14 GB free and one OISST year is 477 MB, so exactly
one file is on disk at a time: downloaded, uploaded, downloaded back, deleted.

Credentials: `HF_TOKEN` in the environment, never argv (ml/CLAUDE.md §6).

Run:
  python3 ml/mirror_psl.py --what all --dry-run
  python3 ml/mirror_psl.py --what oisst --start 1982 --end 2024
  python3 ml/mirror_psl.py --what ncep --start 2020 --end 2024 --force
"""
import argparse
import json
import os
import shutil
import sys
import tempfile
import time

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


class MirrorError(Exception):
    """One file failed. The list continues; the job exits non-zero."""


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
def mirror_one(api, repo, url, rel, workdir, token=None):
    """Download from PSL, upload, DOWNLOAD BACK, compare sha256.

    Returns the manifest record. Raises MirrorError on any failure — and when
    the failure is the ROUND TRIP, the Hub copy is deleted first: a mirrored
    file whose restore does not match is worse than an absent one, because the
    builder would read it and the truncation would surface hours later as
    `NetCDF: HDF error`.
    """
    name = os.path.basename(rel)
    local = os.path.join(workdir, name)
    back = os.path.join(workdir, "back")
    try:
        want = remote_size(url)
        t0 = time.time()
        fetch(url, local)
        down_s = max(time.time() - t0, 1e-6)
        got = os.path.getsize(local)
        if want is not None and got != want:
            raise MirrorError(f"{name}: PSL sent {got:,} of {want:,} bytes")
        digest = sha256(local)

        t1 = time.time()
        api.upload_file(path_or_fileobj=local, path_in_repo=rel, repo_id=repo,
                        repo_type="dataset",
                        commit_message=f"PSL mirror: {name} ({got:,} bytes)")
        up_s = max(time.time() - t1, 1e-6)

        shutil.rmtree(back, ignore_errors=True)
        os.makedirs(back, exist_ok=True)
        p = hf_download(repo, rel, back, token)
        restored = sha256(p)
        if restored != digest:
            api.delete_file(path_in_repo=rel, repo_id=repo, repo_type="dataset",
                            commit_message=f"PSL mirror: DELETE {name} — the "
                                           f"round trip did not verify")
            raise MirrorError(f"{name}: RESTORE MISMATCH (uploaded {digest[:16]}, "
                              f"downloaded {restored[:16]}) — deleted from the Hub")
        return {"path": rel, "bytes": got, "sha256": digest,
                "source_url": url, "mirrored_at": b7.utcnow(),
                "_mb": got / 1e6, "_down_mbps": got / 1e6 / down_s,
                "_up_s": up_s}
    finally:
        for p in (local, local + ".part"):
            if os.path.exists(p):
                os.remove(p)
        shutil.rmtree(back, ignore_errors=True)


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

    todo = []
    skipped = 0
    for u, rel in files:
        if not a.force and rel in listing:
            want = remote_size(u)
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
    try:
        for i, (u, rel) in enumerate(todo, 1):
            try:
                rec = mirror_one(api, repo, u, rel, work, tok)
            except Exception as e:                             # noqa: BLE001
                failed.append((rel, str(e)[:200]))
                print(f"  [{i}/{len(todo)}] {os.path.basename(rel)}  FAILED: "
                      f"{str(e)[:160]}", flush=True)
                continue
            done += 1
            print(f"  [{i}/{len(todo)}] {os.path.basename(rel)}  "
                  f"{rec.pop('_mb'):,.0f} MB  {rec.pop('_down_mbps'):.1f} MB/s down  "
                  f"{rec.pop('_up_s'):.0f}s up", flush=True)
            manifest[rel] = rec
            if done % 25 == 0:
                write_manifest(api, repo, manifest)
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
        api.upload_file(path_or_fileobj=p, path_in_repo=MANIFEST_PATH,
                        repo_id=repo, repo_type="dataset",
                        commit_message=f"PSL mirror: manifest "
                                       f"({len(recs)} files)")
        print(f"  manifest: {len(recs)} file(s), "
              f"{body['bytes'] / 1e9:.1f} GB")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    sys.exit(main())

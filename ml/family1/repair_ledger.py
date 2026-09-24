"""REBUILD A TIER-G YEAR'S LEDGER FROM THE SHARDS ON THE HUB (2026-09-22).

PLAIN ENGLISH. A parked tier-G year on the Hub is a folder of shards (one
compressed file per group and five-day bin, with its own small tile index)
plus three files that SAY what the folder holds: one `<group>__shard_index.npy`
per group (one row per bin), `counts.json` (the year's ledger) and
`done.json` (the marker: every file with its size and sha256, written last).
The assembler believes those three and nothing else.

Four years on the Hub carry a ledger that describes a DIFFERENT, much smaller
year than the folder holds, because a neighbouring lane whose window reached
back into a bin that STARTS in the previous year pushed its own one-bin view
of that year over the real one (ADAPTER_CONTRACT.md, "Lanes"; the push now
refuses to do that, `family10_parts_hub.ledger_would_shrink`):

    pace4k   2024   done.json: 1 bin (3141); folder: 62 shards, 6.99 GB
    lst05    2007   done.json: 1 bin, 5 frames; folder: 149 files, 5.66 GB
    pheno500 2017   done.json: 0 frames; folder: 947 files, 3.96 GB
    pheno500 2019   done.json: 0 frames; folder: 947 files, 3.93 GB

The shards themselves are intact — every shard was uploaded, downloaded back
and hashed before any marker was written (§5.21) — so re-fetching them from
the source (hours of Earthdata) is not needed. This module rebuilds the three
ledger files FROM THE SHARDS THAT ARE PRESENT:

  * each shard's own `.idx.npy` gives the frames present and missing, the
    frame mask, the tiles stored and the byte count — and is CHECKED against
    the shard: the tiles must be contiguous in the writer's order and their
    lengths must sum to the shard's size on the Hub, or the repair refuses;
  * the valid pixels per channel (which the shard index also records) come
    from decompressing every stored tile, exactly as `ShardWriter.write_bin`
    counted them — so a rebuilt row is the row the fetch wrote;
  * a missing frame's REASON (`before_record`, `absent_upstream`, …) lives
    only in the lane's ledger. It is kept where the current ledger describes
    that very shard (its done.json lists it with the same sha256); for the
    orphans it is read from the ledger that wrote them, as the repository's
    own HISTORY holds it (`history_ledger`: the revision of the newest commit
    that wrote an orphan), trusted only for a shard whose historical
    shard-index row equals the rebuilt row field for field — and whose valid
    pixels equal the decoded ones, or the repair refuses; anything else is
    recorded as `unrecorded_ledger_rebuilt` — never guessed;
  * `counts.json` carries `rebuilt_from_shards`: when, by which commit, what
    the previous marker said, and which shards were adopted.

Then, as a push does: the shard indices and the ledger are uploaded,
downloaded back and hashed, and `done.json` is written LAST.

`--dry-run` reads the listing, the markers and every shard's `.idx.npy`
(kilobytes each), validates every shard against its index, and prints — and
writes to `<work>/<store>/repair/<year>.plan.json` — the ledger it would
write, with the valid-pixel counts marked as measured at write time. An
anonymous read is enough for a public store.

  python3 ml/build_family1_stores.py --store pace4k --stage repair \\
      --start 2024-01-01 --end 2024-12-31 --dry-run

THE SECOND DAMAGE CLASS: A LEDGER OVERWRITTEN UNDER AN INTACT MARKER
(measured 2026-09-24, pheno500 2016). The year's done.json lists every shard
in the folder with the right hashes, and the shards are fine — but a
neighbouring lane (window 2016-2017) pushed its own one-bin view of 2016's
ledger files three minutes after done.json was written, without rewriting
done.json. All 315 shard indices and counts.json on the Hub now differ from
the sha256 done.json gives them, so a pull refuses ("PULL MISMATCH …") while
the folder's shard list looks consistent. The repair measures every ledger
file done.json lists against the Hub (the listing's LFS digest, or the small
file downloaded and hashed) and, when only those differ, repairs EXACTLY:
it fetches each one as it was at the commit that wrote done.json
(`last_commits`, `download_at`), refuses unless every one matches done.json's
sha256 and size, uploads them, downloads them back and hashes them. done.json
is left as it is — once the files are back its hashes are true again. If
history does not hold that ledger, nothing is uploaded and `--force` rebuilds
it from the shards instead; with orphans, absent shards or unindexed groups
beside it, the rebuild above runs and rewrites the ledger files anyway.

Only the UNNAMED lane (the year folder's own top-level files) is repaired;
every damaged year is one. A year whose marker already lists exactly the
shards present, and whose ledger files have the hashes it gives them, is
reported consistent and left alone, so the repair is idempotent and a window
may span intact years.
"""
import json
import os
import re
import shutil
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_ML = os.path.dirname(_HERE)
if _ML not in sys.path:
    sys.path.insert(0, _ML)

import family10_parts_hub as ph                                 # noqa: E402
from build_family7 import (atomic_json, git_sha, read_json,     # noqa: E402
                           sha256, utcnow)
from family1 import sharded as sh                               # noqa: E402

SHARD_RE = re.compile(r"^(?P<group>.+)__bin_(?P<bin>-?\d+)\.zst$")
IDX_RE = re.compile(r"^(?P<group>.+)__bin_(?P<bin>-?\d+)\.idx\.npy$")
INDEX_RE = re.compile(r"^(?P<group>.+)__shard_index\.npy$")
UNRECORDED = "unrecorded_ledger_rebuilt"


class RepairError(SystemExit):
    """A year the repair will not rebuild, and why."""


# ------------------------------------------------------------ the Hub seam --
def hub_tree(api, repo, prefix):
    """The FILES directly under `prefix`: {name: {"size", "sha256"}}, and the
    names of its sub-folders (named lanes). `sha256` is the LFS object's
    digest where the file is stored in LFS, else None. The one call here that
    the tests replace with a directory."""
    files, dirs = {}, []
    for it in api.list_repo_tree(repo, path_in_repo=prefix,
                                 repo_type="dataset", recursive=False):
        name = it.path.rsplit("/", 1)[-1]
        if type(it).__name__ == "RepoFolder":
            dirs.append(name)
            continue
        lfs = getattr(it, "lfs", None)
        files[name] = {"size": int(getattr(it, "size", 0) or 0),
                       "sha256": getattr(lfs, "sha256", None) if lfs else None}
    return files, sorted(dirs)


WORKERS = 8


def _pmap(fn, items, workers=WORKERS):
    """`map` over a thread pool, results in the order of `items`; the first
    exception (a refusal) propagates."""
    items = list(items)
    if workers <= 1 or len(items) <= 1:
        return [fn(x) for x in items]
    from concurrent.futures import ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=workers) as ex:
        return list(ex.map(fn, items))


def _get(repo, rel, tok, dest, just_uploaded=False):
    """One file off the Hub (the retrying `family10_parts_hub._download`;
    `just_uploaded` also retries the 404 of a commit still propagating)."""
    shutil.rmtree(dest, ignore_errors=True)
    if just_uploaded:
        return ph._download(repo, rel, tok, dest, just_uploaded=True)
    return ph._download(repo, rel, tok, dest)


# ------------------------------------------------------------- the folder --
def read_folder(files, specs, year):
    """The year folder's listing -> ({(group, bin): {"zst", "idx"}},
    {groups with a shard index}, [problems])."""
    shards, indexed, problems = {}, set(), []
    for n in sorted(files):
        if n in (ph.DONE, ph.COUNTS):
            continue
        m = SHARD_RE.match(n)
        kind = "zst"
        if not m:
            m, kind = IDX_RE.match(n), "idx"
        if m:
            g, b = m.group("group"), int(m.group("bin"))
            shards.setdefault((g, b), {})[kind] = n
            continue
        m = INDEX_RE.match(n)
        if m:
            indexed.add(m.group("group"))
            continue
        problems.append(f"{n}: not a shard, a shard's index, a shard index, "
                        f"counts.json or done.json")
    for (g, b), p in sorted(shards.items()):
        if set(p) != {"zst", "idx"}:
            problems.append(f"{g} bin {b}: only its "
                            f"{'shard' if 'zst' in p else '.idx.npy'} is in "
                            f"the folder")
        if sh.bin_year(b) != int(year):
            problems.append(f"{g} bin {b}: starts in {sh.bin_year(b)}, not "
                            f"{year}")
    unknown = sorted(({g for g, _ in shards} | indexed) - set(specs))
    for g in unknown:
        problems.append(f"group {g!r} is not one of this adapter's groups "
                        f"(an adapter_env knob selecting another product?)")
    return shards, indexed, problems


def check_index(spec, b, idx, size):
    """The shard's own index against its declaration and the shard's size.

    Returns (present frame flags, lengths). The writer lays tiles down frame
    by frame, row-major, with no gaps, and writes a PRESENT frame's empty
    tiles at the running offset with length 0 — so every present tile's
    offset must equal the running total, and the total must be the shard's
    size. Anything else means the shard is not what its index says, and a
    ledger must not vouch for it.
    """
    shape = tuple(int(x) for x in spec["index_shape"])
    if idx.dtype != sh.INDEX_DTYPE or idx.shape != shape:
        raise sh.ShardError(f"bin {b}: index {idx.dtype} {idx.shape}, "
                            f"tile_grid declares {sh.INDEX_DTYPE} {shape}")
    off, ln = idx[..., 0], idx[..., 1]
    absent = (off == sh.FRAME_ABSENT).all(axis=(1, 2))
    part = (off == sh.FRAME_ABSENT).any(axis=(1, 2)) & ~absent
    if part.any() or (ln[absent] != 0).any() or (ln < 0).any():
        raise sh.ShardError(f"bin {b}: an index frame is partly absent, or "
                            f"carries a negative length")
    run = 0
    for f in range(shape[0]):
        if absent[f]:
            continue
        o, n = off[f].reshape(-1), ln[f].reshape(-1)
        want = run + np.concatenate([[0], np.cumsum(n)[:-1]])
        if not np.array_equal(o, want):
            raise sh.ShardError(f"bin {b} frame {f}: the tiles are not "
                                f"contiguous in the writer's order")
        run += int(n.sum())
    if run != int(size):
        raise sh.ShardError(f"bin {b}: the index accounts for {run} bytes, "
                            f"the shard on the Hub is {size}")
    return ~absent, ln


def shard_entry(spec, b, idx, size, shard_path=None):
    """The `shard_index.npy` row `ShardWriter.write_bin` wrote for this bin.

    Without `shard_path` (the dry run) `valid_pixels`/`valid_fraction` are
    None: they need every stored tile decompressed."""
    present, ln = check_index(spec, b, idx, size)
    F, C = int(spec["frames_per_bin"]), int(spec["C"])
    n_present = int(present.sum())
    mask = sum(1 << f for f in range(F) if present[f])
    e = {"bin": int(b), "year": sh.bin_year(b),
         "frames_present": n_present, "frames_missing": F - n_present,
         "frame_mask": mask, "tiles_stored": int((ln > 0).sum()),
         "nbytes": int(size), "valid_pixels": None, "valid_fraction": None,
         "shard": sh.shard_relpath(b), "index": sh.index_relpath(b)}
    if shard_path is None:
        return e
    T, dtype = int(spec["tile"]), spec["dtype"]
    dt_ = sh.DTYPES[dtype]
    want = T * T * C * dt_.itemsize
    dctx = sh.zstd().ZstdDecompressor()
    valid = np.zeros(C, np.int64)
    off = idx[..., 0]
    with open(shard_path, "rb") as fh:
        for f, ty, tx in zip(*np.nonzero(ln > 0)):
            fh.seek(int(off[f, ty, tx]))
            raw = dctx.decompress(fh.read(int(ln[f, ty, tx])),
                                  max_output_size=want)
            if len(raw) != want:
                raise sh.ShardError(f"bin {b} ({f},{ty},{tx}): a tile "
                                    f"decompressed to {len(raw)} bytes, the "
                                    f"declared tile is {want}")
            t = np.frombuffer(raw, dt_).reshape(T, T, C)
            valid += sh.valid_mask(t, dtype).reshape(-1, C).sum(axis=0)
    H, W = int(spec["H"]), int(spec["W"])
    e["valid_pixels"] = valid.tolist()
    e["valid_fraction"] = (valid / (max(n_present, 1) * H * W)).tolist()
    return e


def last_commits(api, repo, prefix):
    """{name: last_commit} for the files directly under `prefix` — the
    commit (`.oid`, `.date`) that last wrote each one. One expanded listing;
    the seam the tests replace for the repository's history."""
    out = {}
    for it in api.list_repo_tree(repo, path_in_repo=prefix,
                                 repo_type="dataset", expand=True):
        lc = getattr(it, "last_commit", None)
        if lc is not None:
            out[it.path.rsplit("/", 1)[-1]] = lc
    return out


def download_at(repo, tok, prefix, name, oid, dest):
    """`<prefix>/<name>` AS IT WAS at revision `oid`, into the directory
    `dest` (emptied first); returns the local path. Transient failures are
    retried on `family10_parts_hub`'s ladder; anything else (a 404: the file
    did not exist at that revision) is raised. The other history seam."""
    from huggingface_hub import hf_hub_download
    for i in range(ph.DOWNLOAD_ATTEMPTS):
        try:
            shutil.rmtree(dest, ignore_errors=True)
            return hf_hub_download(repo, f"{prefix}/{name}",
                                   repo_type="dataset", token=tok,
                                   revision=oid, local_dir=dest)
        except Exception as e:                          # noqa: BLE001
            if not ph.transient_download_error(e) or \
                    i == ph.DOWNLOAD_ATTEMPTS - 1:
                raise
            import time
            time.sleep(ph.DOWNLOAD_BACKOFF_S[
                min(i, len(ph.DOWNLOAD_BACKOFF_S) - 1)])


def history_ledger(api, repo, tok, prefix, names, scratch):
    """The ledger that last described `names` (orphaned shards), from the
    repository's OWN HISTORY: the newest commit that wrote any of them is a
    revision at which the lane that pushed them had its `counts.json` and
    shard indices in place (they travel in the same batched push, done.json
    after). Returns (revision, counts, {group: {bin: shard-index row}}) or
    None. Best effort — every row it returns is checked against the rebuild
    before a reason is taken from it, and a failure costs only the reasons.
    """
    try:
        want = set(names)
        latest = None
        for n, lc in last_commits(api, repo, prefix).items():
            if n in want and (latest is None or lc.date > latest.date):
                latest = lc
        if latest is None:
            return None

        def get(n):
            return download_at(repo, tok, prefix, n, latest.oid,
                               os.path.join(scratch, "history", n))
        counts = read_json(get(ph.COUNTS), None)
        gs = sorted((counts or {}).get("groups") or {})
        arrs = _pmap(lambda g: sh.load_shard_index(
            get(f"{g}__shard_index.npy")), gs)
        rows = {g: {int(r["bin"]): r for r in arr}
                for g, arr in zip(gs, arrs)}
        return latest.oid, counts, rows
    except Exception as e:                              # noqa: BLE001
        print(f"  ::notice::the repository history did not give the ledger "
              f"that described the orphaned shards ({type(e).__name__}: "
              f"{str(e)[:160]}) — their missing frames are recorded as "
              f"'{UNRECORDED}'")
        return None


ROW_FIELDS = ("frames_present", "frames_missing", "frame_mask",
              "tiles_stored", "nbytes")


def row_agrees(row, e):
    return all(int(row[k]) == int(e[k]) for k in ROW_FIELDS)


# ------------------------------------------------------------- the ledger --
def old_reasons(counts, done, files):
    """{(group, bin, frame): reason} from the CURRENT ledger, for the shards
    that ledger actually describes: those its done.json lists with the sha256
    the shard has on the Hub now. Only there is its reason about THIS shard."""
    have = {e["name"]: e.get("sha256") for e in (done or {}).get("files", [])}
    out = {}
    for m in ((counts or {}).get("counts") or {}).get("missing_frames") or []:
        g, b = str(m.get("group")), int(m.get("bin"))
        n = f"{g}__{os.path.basename(sh.shard_relpath(b))}"
        live = (files.get(n) or {}).get("sha256")
        if n in have and (live is None or have[n] == live):
            out[(g, b, int(m.get("frame")))] = str(m.get("reason"))
    return out


def build_ledger(year, specs, entries, reasons, frame_seconds, prov):
    """counts.json for the rebuilt year: the shape `fetch_grid_year` writes,
    from the rebuilt shard-index rows."""
    summary, counts, missing = {}, {"frames_present": {}}, []
    wanted = 0
    for g in sorted(entries):
        es = entries[g]
        F = int(specs[g]["frames_per_bin"])
        wanted += F * len(es)
        summary[g] = {
            "bins": len(es),
            "frames_present": sum(e["frames_present"] for e in es),
            "frames_missing": sum(e["frames_missing"] for e in es),
            "tiles_stored": sum(e["tiles_stored"] for e in es),
            "bytes": sum(e["nbytes"] for e in es)}
        if summary[g]["frames_present"]:
            counts["frames_present"][g] = summary[g]["frames_present"]
        for e in es:
            for f in range(F):
                if e["frame_mask"] >> f & 1:
                    continue
                why = reasons.get((g, e["bin"], f), UNRECORDED)
                counts.setdefault("frames_missing", {})
                counts["frames_missing"][why] = \
                    counts["frames_missing"].get(why, 0) + 1
                missing.append({"group": g, "bin": e["bin"], "frame": f,
                                "day": str(sh.frame_day(e["bin"], f,
                                                        frame_seconds)),
                                "reason": why})
    counts["frames_wanted"] = wanted
    if missing:
        counts["missing_frames"] = missing
    rows = sum(v["frames_present"] for v in summary.values())
    return {"year": int(year), "tier": "G", "rows": rows, "parts": 0,
            "groups": summary, "counts": counts,
            "grids": json.loads(json.dumps({g: specs[g] for g in entries})),
            "at": utcnow(), "rebuilt_from_shards": prov}


# ------------------------------------------------- the ledger vs its marker --
def is_ledger_file(name):
    """A shard index or counts.json — what a lane's push rewrites whole."""
    return name == ph.COUNTS or bool(INDEX_RE.match(name))


def stale_ledger_files(repo, tok, prefix, done, files, scratch):
    """The ledger files done.json lists whose copy on the Hub is not the one
    it describes: absent, a different size, or a different sha256. The LFS
    digest from the listing is used where there is one; otherwise the file
    (a few kilobytes) is downloaded and hashed. Sorted names."""
    listed = {e["name"]: e for e in (done or {}).get("files", [])
              if is_ledger_file(e["name"])}

    def check(n):
        e, live = listed[n], files.get(n)
        if live is None:
            return n
        if e.get("bytes") is not None and int(live["size"]) != int(e["bytes"]):
            return n
        got = live.get("sha256")
        if got is None:
            got = sha256(_get(repo, f"{prefix}/{n}", tok,
                              os.path.join(scratch, "current", n)))
        return n if got != e.get("sha256") else None
    return sorted(n for n in _pmap(check, sorted(listed)) if n)


def restore_from_history(api, repo, tok, prefix, what, done, files, stale,
                         rep, scratch, work, store, year, dry_run):
    """THE EXACT REPAIR of a year whose shards and done.json agree and whose
    ledger files were overwritten after done.json was written: put back the
    ledger files AS THEY WERE at the commit that wrote done.json, each
    checked against done.json's own sha256 and size before anything is
    uploaded. done.json is not rewritten — once they are back its hashes are
    true again. Refuses (nothing uploaded) if history does not hold exactly
    the ledger done.json describes; `--force` rebuilds from the shards."""
    want = {e["name"]: e for e in done.get("files", [])}
    lc = last_commits(api, repo, prefix).get(ph.DONE)
    if lc is None:
        raise RepairError(
            f"repair refuses {what}: {len(stale)} ledger file(s) differ from "
            f"done.json, and the repository history does not say which "
            f"commit wrote done.json — nothing to restore from; --force "
            f"rebuilds the ledger from the shards instead")
    oid = lc.oid
    print(f"  {what}: done.json was written by revision {oid[:10]} "
          f"({getattr(lc, 'date', None)}); recovering {len(stale)} ledger "
          f"file(s) as they were there")

    def recover(n):
        try:
            p = download_at(repo, tok, prefix, n, oid,
                            os.path.join(scratch, "restore", n))
        except Exception as e:                          # noqa: BLE001
            return n, None, f"{type(e).__name__}: {str(e)[:120]}"
        got, size = sha256(p), os.path.getsize(p)
        if got != want[n].get("sha256") or size != int(want[n]["bytes"]):
            return n, None, (f"{size} bytes sha256 {got[:12]}…, done.json "
                             f"says {want[n]['bytes']} bytes sha256 "
                             f"{str(want[n].get('sha256'))[:12]}…")
        return n, p, None
    got = _pmap(recover, stale)
    bad = [(n, why) for n, p, why in got if p is None]
    if bad:
        raise RepairError(
            f"repair refuses {what}: the repository history does not hold "
            f"the ledger done.json describes — at revision {oid[:10]} (the "
            f"commit that wrote done.json) {len(bad)} of {len(stale)} stale "
            f"ledger file(s) do not match it:\n  "
            + "\n  ".join(f"{n}: {why}" for n, why in bad[:8])
            + ("\n  …" if len(bad) > 8 else "")
            + "\nnothing was uploaded; --force rebuilds the ledger from the "
              "shards instead")
    local = {n: p for n, p, _ in got}
    if dry_run:
        plan = {"report": rep,
                "restore_from_history": {"revision": oid,
                                         "files": list(stale)}}
        pp = os.path.join(os.path.abspath(work), store, "repair",
                          f"{year}.plan.json")
        atomic_json(pp, plan)
        print(f"  {what}: DRY RUN — would restore {len(stale)} ledger "
              f"file(s) from revision {oid[:10]}, each matching done.json's "
              f"sha256 ({', '.join(stale[:4])}"
              + (", …" if len(stale) > 4 else "")
              + f"); done.json unchanged; plan -> {pp}")
        rep["action"] = "dry run (restore from history)"
        rep["plan"] = pp
        return rep

    # the folder must be what was read: every shard, its index and done.json
    now, _ = hub_tree(api, repo, prefix)
    changed = sorted(n for n in set(now) | set(files)
                     if not is_ledger_file(n)
                     and (now.get(n) or {}).get("size")
                     != (files.get(n) or {}).get("size"))
    if changed:
        raise RepairError(f"repair refuses {what}: the folder changed while "
                          f"it was read ({changed[:4]}) — another job is "
                          f"writing it; nothing was written")
    pairs = [(f"{prefix}/{n}", local[n]) for n in stale]
    for i in range(0, len(pairs), ph.MANY_FILES):
        ph._upload(api, repo, pairs[i:i + ph.MANY_FILES],
                   f"family 1 partials ({what}): {len(stale)} ledger file(s) "
                   f"restored from history {oid[:10]}")

    def verify(n):
        back = _get(repo, f"{prefix}/{n}", tok,
                    os.path.join(scratch, "verify", n), just_uploaded=True)
        h = sha256(back)
        if h != want[n]["sha256"]:
            raise RepairError(f"RESTORE MISMATCH {prefix}/{n}: uploaded "
                              f"{want[n]['sha256']}, the Hub served {h} — "
                              f"done.json was not touched")
    _pmap(verify, stale)
    shutil.rmtree(scratch, ignore_errors=True)
    print(f"  {what}: REPAIRED — {len(stale)} ledger file(s) restored from "
          f"revision {oid[:10]}; done.json unchanged")
    rep["action"] = "restored from history"
    rep["restored"] = {"revision": oid, "files": len(stale)}
    return rep


# ------------------------------------------------------------- the repair --
def repair_year(adapter, layout, year, work, dry_run=True, hub=None,
                force=False):
    """Rebuild `partials/<family>/<store>/<year>/`'s ledger from its shards.

    `hub` is a callable returning (api, repo, token), as `Layout.hub`; the
    dry run falls back to an anonymous read of the layout's repository.
    Returns a report dict; refuses (SystemExit) on anything it cannot vouch
    for, before a byte is written.
    """
    store, year = adapter.store, int(year)
    specs = adapter.specs()
    prefix = ph.hub_prefix(store, year, layout.hf_partials)
    scratch = os.path.join(os.path.abspath(work), store, "repair", str(year))
    os.makedirs(scratch, exist_ok=True)
    if hub is not None:
        api, repo, tok = hub()
    elif dry_run and not layout.token()[1]:
        from huggingface_hub import HfApi
        api, repo, tok = HfApi(), layout.repo_id, None
        print(f"  repair: anonymous read of {repo} (dry run, no token)")
    else:
        api, repo, tok = layout.hub()
    if not dry_run and bool(layout.private) != str(repo).endswith("-private"):
        raise RepairError(f"repair refuses {store} {year}: private="
                          f"{layout.private} and the repository is {repo!r}")
    files, lanes = hub_tree(api, repo, prefix)
    what = f"{store} {year}"
    if not files:
        raise RepairError(f"repair refuses {what}: {repo}:{prefix}/ holds no "
                          f"file at its top level — there is no unnamed lane "
                          f"to repair" + (f" (named lanes: {lanes})"
                                          if lanes else ""))
    shards, indexed, problems = read_folder(files, specs, year)
    if problems:
        raise RepairError(f"repair refuses {what}: the folder is not a "
                          f"tier-G year it can vouch for:\n  "
                          + "\n  ".join(problems[:20]))

    def fetch_json(name):
        if name not in files:
            return None
        return read_json(_get(repo, f"{prefix}/{name}", tok,
                              os.path.join(scratch, "meta")), None)
    done, counts = fetch_json(ph.DONE), fetch_json(ph.COUNTS)
    listed = {e["name"] for e in (done or {}).get("files", [])}
    zst = {p["zst"] for p in shards.values()}
    groups = sorted({g for g, _ in shards} | indexed)
    orphans = sorted(n for n in zst if n not in listed)
    gone = sorted(n for n in listed if n.endswith(ph.SHARD_SUFFIX)
                  and n not in files)
    unindexed = [g for g in groups if g not in indexed]
    rep = {"store": store, "year": year, "repo": repo, "prefix": prefix,
           "dry_run": bool(dry_run), "named_lanes_beside": lanes,
           "shards_in_folder": len(zst),
           "shards_in_done_json": sum(1 for n in listed
                                      if n.endswith(ph.SHARD_SUFFIX)),
           "orphaned_shards": len(orphans), "listed_but_absent": gone,
           "previous_done": {k: (done or {}).get(k) for k in
                             ("rows", "bytes", "at", "builder_git_sha")}}
    # THE LEDGER FILES AGAINST THE MARKER: a neighbour's push over them
    # leaves done.json listing the right shards with hashes the ledger files
    # on the Hub no longer have
    stale = stale_ledger_files(repo, tok, prefix, done, files, scratch)
    rep["stale_ledger_files"] = stale
    print(f"  {what}: {len(zst)} shard(s) in the folder over {len(groups)} "
          f"group(s); done.json lists {rep['shards_in_done_json']} "
          f"(rows {rep['previous_done']['rows']}, at "
          f"{rep['previous_done']['at']}); {len(orphans)} orphaned"
          + (f"; {len(gone)} listed but ABSENT" if gone else "")
          + f"; {len(stale)} ledger file(s) differ from done.json"
          + (f"; named lanes beside it (untouched): {lanes}" if lanes else ""))
    clean = done is not None and not orphans and not gone and not unindexed
    if clean and not stale and not force:
        print(f"  {what}: CONSISTENT — done.json lists exactly the shards in "
              f"the folder and every ledger file has its hash; nothing to "
              f"repair")
        rep["action"] = "none (consistent)"
        return rep
    if clean and not force:
        return restore_from_history(api, repo, tok, prefix, what, done, files,
                                    stale, rep, scratch, work, store, year,
                                    dry_run)

    # EVERY SHARD'S INDEX, checked against the shard's size on the Hub —
    # read WORKERS at a time: pheno500 is 315 groups, so a year is about a
    # thousand small reads, and one at a time that is most of an hour
    def load_old(g):
        n = f"{g}__shard_index.npy"
        if n not in files:
            return g, None
        try:
            arr = sh.load_shard_index(_get(repo, f"{prefix}/{n}", tok,
                                           os.path.join(scratch, "old", g)))
            return g, {int(r["bin"]): r for r in arr}
        except (ValueError, OSError) as e:
            print(f"  ::warning::{what}: the current {n} does not load "
                  f"({e}) — it is replaced")
            return g, None
    old_si = {g: v for g, v in _pmap(load_old, groups) if v is not None}

    def read_shard(item):
        (g, b), p = item
        spec = specs[g]
        ip = _get(repo, f"{prefix}/{p['idx']}", tok,
                  os.path.join(scratch, "idx", p["idx"]))
        idx = np.load(ip, allow_pickle=False)
        isum = (os.path.getsize(ip), sha256(ip))
        size = files[p["zst"]]["size"]
        sp, zsum = None, (size, files[p["zst"]]["sha256"])
        try:
            if not dry_run:
                sp = _get(repo, f"{prefix}/{p['zst']}", tok,
                          os.path.join(scratch, "zst", p["zst"]))
                got = sha256(sp)
                lfs = files[p["zst"]]["sha256"]
                if os.path.getsize(sp) != size or (lfs and got != lfs):
                    raise RepairError(
                        f"repair refuses {what}: {p['zst']} downloaded as "
                        f"{os.path.getsize(sp)} bytes sha256 {got}, the Hub "
                        f"lists {size} bytes sha256 {lfs}")
                zsum = (size, got)
            try:
                e = shard_entry(spec, b, idx, size, sp)
            except sh.ShardError as ex:
                raise RepairError(f"repair refuses {what}: {g}: {ex}")
        finally:
            if sp:
                shutil.rmtree(os.path.join(scratch, "zst", p["zst"]),
                              ignore_errors=True)
        return g, b, p, isum, zsum, e

    entries, sums = {g: [] for g in groups}, {}
    agree = disagree = 0
    for i, (g, b, p, isum, zsum, e) in enumerate(
            _pmap(read_shard, sorted(shards.items())), 1):
        sums[p["idx"]], sums[p["zst"]] = isum, zsum
        entries[g].append(e)
        old = old_si.get(g, {}).get(b)
        if old is not None:
            same = row_agrees(old, e)
            agree += same
            disagree += not same
        if i % 25 == 0:
            print(f"    {i}/{len(shards)} shard(s) read", flush=True)
    rep["current_index_rows_agreeing"] = agree
    rep["current_index_rows_disagreeing"] = disagree

    live = {n: {**v, "sha256": (sums.get(n) or (None, None))[1]
                or v["sha256"]} for n, v in files.items()}
    reasons = old_reasons(counts, done, live)
    # THE ORPHANS' OWN LEDGER, from the repository's history, trusted row by
    # row only where it describes exactly the shard now in the folder
    hist = history_ledger(api, repo, tok, prefix, orphans, scratch) \
        if orphans else None
    if hist:
        rev, hc, hrows = hist
        verified = set()
        for g, es in entries.items():
            for e in es:
                r = hrows.get(g, {}).get(e["bin"])
                if r is None or not row_agrees(r, e):
                    continue
                if e["valid_pixels"] is not None and \
                        [int(v) for v in r["valid_pixels"]] != \
                        [int(v) for v in e["valid_pixels"]]:
                    raise RepairError(
                        f"repair refuses {what}: {g} bin {e['bin']} decodes "
                        f"to {e['valid_pixels']} valid pixels and the ledger "
                        f"that wrote it (revision {rev[:10]}) says "
                        f"{r['valid_pixels'].tolist()} — the shard is not "
                        f"the one that ledger described")
                verified.add((g, e["bin"]))
        taken = 0
        for m in ((hc or {}).get("counts") or {}).get("missing_frames") or []:
            k = (str(m.get("group")), int(m.get("bin")), int(m.get("frame")))
            if k[:2] in verified and k not in reasons:
                reasons[k] = str(m.get("reason"))
                taken += 1
        rep["history"] = {"revision": rev, "rows_verified": len(verified),
                          "reasons_recovered": taken}
        print(f"  {what}: history revision {rev[:10]} — {len(verified)} "
              f"shard-index row(s) agree with the rebuild field for field; "
              f"{taken} missing-frame reason(s) recovered from it")
    prov = {"at": utcnow(), "builder_git_sha": git_sha(),
            "why": ("the year's done.json, counts.json and shard index "
                    "described fewer shards than the folder holds (a "
                    "neighbouring lane's push over it, ADAPTER_CONTRACT.md "
                    "'Lanes'); rebuilt from every shard present, each "
                    "checked against its own index and the Hub's size"),
            "previous_done": rep["previous_done"],
            "previous_shards_listed": rep["shards_in_done_json"],
            "shards_adopted": orphans,
            "history": rep.get("history"),
            "missing_frame_reasons": (
                f"kept from the previous ledger for the shards it described, "
                f"recovered from the repository history for the shards whose "
                f"historical shard-index row matches the rebuild exactly, "
                f"'{UNRECORDED}' for any other")}
    led = build_ledger(year, specs, entries, reasons,
                       adapter.frame_seconds, prov)
    rep["rebuilt"] = {"bins": sum(len(v) for v in entries.values()),
                      "rows": led["rows"], "groups": led["groups"],
                      "frames_missing": led["counts"].get("frames_missing")}
    per = {g: (min(e["bin"] for e in es), max(e["bin"] for e in es))
           for g, es in entries.items() if es}
    for g in groups[:8]:
        s = led["groups"][g]
        print(f"    {g}: {s['bins']} bin(s)"
              + (f" {per[g][0]}..{per[g][1]}" if g in per else "")
              + f", {s['frames_present']} frame(s) present, "
                f"{s['frames_missing']} missing, {s['tiles_stored']} tile(s), "
                f"{s['bytes'] / 1e6:.1f} MB")
    if len(groups) > 8:
        print(f"    … and {len(groups) - 8} more group(s)")
    print(f"  {what}: the rebuilt ledger — {rep['rebuilt']['bins']} bin(s), "
          f"{led['rows']} frame(s) present, missing by reason "
          f"{led['counts'].get('frames_missing') or {}}; {agree} row(s) of "
          f"the current shard index agree with the rebuild"
          + (f", {disagree} DISAGREE" if disagree else ""))

    out = os.path.join(scratch, "out")
    shutil.rmtree(out, ignore_errors=True)
    os.makedirs(out)
    if dry_run:
        plan = {"report": rep, "counts_json": led,
                "shard_index_rows": entries,
                "note": ("DRY RUN: valid_pixels / valid_fraction are None "
                         "here and are measured from every stored tile at "
                         "write time; nothing was written to the Hub")}
        pp = os.path.join(os.path.abspath(work), store, "repair",
                          f"{year}.plan.json")
        atomic_json(pp, plan)
        print(f"  {what}: DRY RUN — would write {len(groups)} shard "
              f"index(es), counts.json and done.json ({len(zst)} shard(s)); "
              f"plan -> {pp}")
        rep["action"] = "dry run"
        rep["plan"] = pp
        return rep

    # WRITE: the shard indices and the ledger, verified, then the marker
    for g in groups:
        sh.save_shard_index(os.path.join(out, f"{g}__shard_index.npy"),
                            entries[g], specs[g]["C"])
    atomic_json(os.path.join(out, ph.COUNTS), led)
    now, _ = hub_tree(api, repo, prefix)
    changed = sorted(n for n in set(now) | set(files)
                     if n not in (ph.DONE, ph.COUNTS)
                     and not INDEX_RE.match(n)
                     and (now.get(n) or {}).get("size")
                     != (files.get(n) or {}).get("size"))
    if changed:
        raise RepairError(f"repair refuses {what}: the folder changed while "
                          f"it was read ({changed[:4]}) — another job is "
                          f"writing it; nothing was written")
    new = sorted(os.listdir(out))
    pairs = [(f"{prefix}/{n}", os.path.join(out, n)) for n in new]
    for i in range(0, len(pairs), ph.MANY_FILES):
        ph._upload(api, repo, pairs[i:i + ph.MANY_FILES],
                   f"family 1 partials ({what}): ledger rebuilt from "
                   f"{len(zst)} shard(s)")
    for rel, local in pairs:
        back = _get(repo, rel, tok, os.path.join(scratch, "verify"))
        if sha256(back) != sha256(local):
            raise RepairError(f"RESTORE MISMATCH {rel}: the rebuilt ledger "
                              f"did not come back as uploaded — no done.json "
                              f"was written")
        sums[os.path.basename(rel)] = (os.path.getsize(local), sha256(local))
    names = sorted(n for n in sums if n != ph.COUNTS) + [ph.COUNTS]
    fl = [{"name": n, "bytes": int(sums[n][0]), "sha256": sums[n][1]}
          for n in names]
    marker = ph._done_record(store, year, led["rows"], 0,
                             sum(e["bytes"] for e in fl), fl, None, led)
    marker["rebuilt_from_shards"] = {"at": prov["at"],
                                     "builder_git_sha": prov["builder_git_sha"],
                                     "shards_adopted": len(orphans)}
    dp = os.path.join(scratch, ph.DONE)
    atomic_json(dp, marker)
    ph._upload(api, repo, [(f"{prefix}/{ph.DONE}", dp)],
               f"family 1 partials ({what}): done.json (ledger rebuilt)")
    shutil.rmtree(scratch, ignore_errors=True)
    print(f"  {what}: REPAIRED — done.json now lists {len(fl)} file(s), "
          f"{led['rows']} frame(s), written last")
    rep["action"] = "repaired"
    return rep

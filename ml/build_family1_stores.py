#!/usr/bin/env python3
"""Families 1.gf, 1.0.tf and 0.9.tf — the tier-P OBSERVATION STORES (E-082).

PLAIN ENGLISH. Family 10 turned four ocean observing systems into one kind of
store: rows of (time, place, values) sorted by five-day bin, which a model
reads as "the k nearest measurements and how far away they are". Family 1
does the same for the fine-granularity observations of land, coast and ocean —
weather stations, tide gauges, buoys, radiosondes, ship reports, profiles —
and this is its builder.

IT IS FAMILY 10'S BUILDER, BY IMPORT. Every stage below calls
`ml/build_family10_stores.py` (`f10b`): `Ctx`, `PartWriter`, `stage_index`,
`stage_fetch`, both assemblers, `_finish_store`, `check_store`,
`stage_publish`, `run_smoke`. What this file adds is only what family 1 has
and family 10 does not:

  * THE REGISTRY. `ml/family1/adapters/` holds one module per store and
    `family1.adapters.REGISTRY` maps store -> adapter class, refusing a
    duplicate name or a licence that forbids redistribution on a public store.
  * THE LAYOUT (`f10b.Layout`): `tensors/family1_tf/<store>`,
    `partials/family1_tf/<store>/<year>`, `ml/cache/family1_tf`, per the
    adapter's `family` (`1gf`, `1tf`, `09tf`).
  * THE TWO TRACKS. `distribution = "private"` publishes to
    `chfrank/earth-tensors-private` with the token from `HF_PRIVATE_TOKEN`
    (else `HF_TOKEN`, else `HF_PRIVATE_READ_TOKEN` for a read), and the
    publish stage REFUSES any repository id that does not end in `-private`
    (`f10b.check_publish_target`). A public adapter publishes to
    `chfrank/earth-tensors` and is refused the private one.
  * CREDENTIALS PREFLIGHT. `adapter.credentials` names environment
    variables; if any is unset the job stops before its first byte.
  * SCHEMA 3. An adapter with `time_dtype = "int64"` (records before 1914)
    writes int64 `time_s`; `ml/family10_store.py` reads it.
  * platforms.json, when `adapter.platform_meta` — written beside the arrays
    and hashed into store.json like a column.
  * FIVE STAGES, `index | fetch | assemble | publish | check` (or `all`),
    because family 1's big stores fetch on hosted lanes and assemble on a box:
      index     list the archive, read one record, write plan.json
      fetch     per-year column parts, each year marked only after its parts
                are on disk (§5.21). `--push-parts` then parks every marked
                year on the Hub under partials/<family>/<store>/<year>/
                (done.json last); `--parts-from-hub` replaces the source with
                that pull on the assembling machine.
      assemble  the store, from the parts (streaming above 50 M rows)
      publish   upload, download every file back, compare sha256
      check     the store's sha256 against store.json, E-079 §4's
                assertions, and — once published — the Hub's manifest and
                store.json against the local ones
  * TIER G, THE SHARDED LAYOUT (E-082 wave 2): an adapter that subclasses
    `family1.sharded.GridAdapter` yields FRAMES on fixed grids instead of
    rows, and the same five stages build `ml/family1/sharded.py`'s layout —
    one zstd shard per (group, five-day bin), 256 x 256 tiles, a per-shard
    (offset, length) index, `tile_grid.json`, `shard_index.npy`. A "year" is
    the bins whose first day falls in it; its parts are its shards and
    indices and travel through `--push-parts` / `--parts-from-hub` exactly
    like tier-P parts; `assemble` links them into place without
    re-compressing; `publish` downloads every file back and decompresses 50
    random tiles; `check` decompresses every tile. `--stage probe` measures
    frames, valid fraction, compressed bytes per tile and per frame, bytes
    per valid pixel and the extrapolated store size.
  * THE LICENCE GATE: a public adapter whose licence says
    `redistribution_confirmed: False` is refused a publish and a parts push
    unless `--allow-unconfirmed-licence` is passed.
  * `--check-credentials` (no store): one authenticated request to LP DAAC,
    GES DISC and PO.DAAC with the Earthdata account, and what each answered
    (`ml/family1/earthdata_check.py`).
  * THE PROBE, `--stage probe --probe-month YYYY-MM`: index, then one
    calendar month through the real adapter, measured —
    `<work>/probe/<store>/<YYYY-MM>.json` with rows, rows per day, distinct
    platforms, NaN fraction and out-of-bounds count per channel, bytes off
    the network, wall seconds and bytes per row. Every size in the design
    notes is an estimate until this file replaces it
    (`estimate_rows_per_year` is left null for the session that reads it).

Run:
  python3 ml/build_family1_stores.py --store ghcnd --smoke
  python3 ml/build_family1_stores.py --store ghcnd --stage probe --probe-month 2024-01
  python3 ml/build_family1_stores.py --store ghcnd --stage index,fetch \\
      --start 2020-01-01 --end 2020-12-31 --push-parts          # a hosted lane
  python3 ml/build_family1_stores.py --store ghcnd --stage all \\
      --parts-from-hub --start 1763-01-01                         # the box
  python3 ml/build_family1_stores.py --store seaice_asi --smoke   # tier G
  python3 ml/build_family1_stores.py --store seaice_asi --stage probe \\
      --probe-month 2020-03
  python3 ml/build_family1_stores.py --check-credentials          # hosted
"""
import argparse
import calendar
import datetime as dt
import json
import os
import platform as _platform
import shutil
import sys
import tempfile
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import build_family10_stores as f10b                            # noqa: E402
import family10_store as f10                                    # noqa: E402
from build_family7 import (END, atomic_json, git_sha, mark,     # noqa: E402
                           marked, marker, read_json, utcnow)
from family1 import sharded as sh                               # noqa: E402
from family1.adapters import FAMILIES, REGISTRY                 # noqa: E402

CACHE = os.path.join(HERE, "cache")
PUBLIC_REPO = "chfrank/earth-tensors"
PRIVATE_REPO = "chfrank/earth-tensors-private"
PUBLIC_TOKEN_ENV = ("HF_TOKEN",)
PRIVATE_TOKEN_ENV = ("HF_PRIVATE_TOKEN", "HF_TOKEN", "HF_PRIVATE_READ_TOKEN")

STAGES = ["index", "fetch", "assemble", "publish", "check"]
DEPS = {"fetch": ["index"], "assemble": ["fetch"],
        "publish": ["assemble"], "check": ["assemble"]}


# ================================================================ routing ==
def layout_for(adapter):
    """The adapter's `family` and `distribution` -> an `f10b.Layout`."""
    fam = getattr(adapter, "family", None)
    if fam not in FAMILIES:
        raise ValueError(f"{adapter.store}: family {fam!r} is not one of "
                         f"{sorted(FAMILIES)}")
    name, version, slug = FAMILIES[fam]
    private = getattr(adapter, "distribution", "public") == "private"
    return f10b.Layout(
        family=name, family_version=version,
        hf_root=f"tensors/{slug}", hf_partials=f"partials/{slug}",
        cache_dirname=slug,
        repo_id=PRIVATE_REPO if private else PUBLIC_REPO,
        private=private,
        token_env=PRIVATE_TOKEN_ENV if private else PUBLIC_TOKEN_ENV,
        builder="ml/build_family1_stores.py",
        label=f"family {version}")


def default_work(adapter):
    return os.path.join(CACHE, layout_for(adapter).cache_dirname)


def credentials_preflight(adapter, env=None):
    """REFUSE before any byte when a declared credential is unset.

    Names only, never values: the message lists which variables are missing
    and nothing else (ml/CLAUDE.md §6).
    """
    env = os.environ if env is None else env
    missing = [k for k in getattr(adapter, "credentials", ()) or ()
               if not env.get(k)]
    if missing:
        sys.exit(f"REFUSING {adapter.store}: its source needs the environment "
                 f"variable(s) {', '.join(missing)}, which are not set. They "
                 f"come from repository secrets on a GitHub-HOSTED runner "
                 f"only (ml/CLAUDE.md §6) — a box assembles from Hub parts "
                 f"with --parts-from-hub and never needs them. Nothing has "
                 f"been fetched.")
    return True


def needs_source(a, stages):
    """Does this invocation read the source archive (so need credentials)?"""
    if getattr(a, "source_dir", ""):
        return False
    touches = {"index", "fetch", "probe"} & set(stages)
    if not touches:
        return False
    if getattr(a, "parts_from_hub", False) and "probe" not in stages:
        return False
    return True


LICENCE_PENDING_NOTE = (
    "licence_pending: published to the PRIVATE repository by "
    "`--distribution private` while the producer's redistribution answer is "
    "pending; the adapter's own distribution is 'public'. Move it to the "
    "public repository only once the licence is confirmed.")


def apply_distribution(ad, want):
    """`--distribution` on ONE adapter instance, before its layout is made.

    '' leaves the adapter as declared. 'private' on a PUBLIC adapter routes it
    to the private repository (parts, store and publish alike) and records
    `licence_pending` in the store's notes — the safe direction, used while a
    producer has not answered (E-082: seaice_asi and Bremen). 'public' on a
    PRIVATE adapter is REFUSED: a licence that forbids redistribution is not
    something a command-line flag can overrule. The class is never touched, so
    the registry and every other instance keep the declared track.
    """
    want = (want or "").strip()
    if not want:
        return ad
    have = getattr(ad, "distribution", "public")
    if want not in ("public", "private"):
        sys.exit(f"--distribution {want!r}: expected 'public' or 'private'")
    if want == have:
        return ad
    if want == "public":
        sys.exit(f"REFUSING --distribution public for {ad.store}: the adapter "
                 f"declares distribution {have!r} (licence "
                 f"{(getattr(ad, 'licence', {}) or {}).get('name')!r}). A "
                 f"private store never becomes public by a flag; nothing has "
                 f"been fetched or uploaded.")
    ad.distribution = "private"
    notes = getattr(ad, "notes", "") or ""
    ad.notes = (notes + "\n" if notes else "") + LICENCE_PENDING_NOTE
    ad.distribution_override = {"declared": have, "used": "private",
                                "note": "licence_pending"}
    print(f"::notice::{ad.store}: --distribution private — declared "
          f"{have!r}, routed to {PRIVATE_REPO} (licence_pending)")
    return ad


def make_ctx(a, adapter_cls):
    ad = apply_distribution(adapter_cls(), getattr(a, "distribution", ""))
    ctx = f10b.Ctx(a, adapter=ad, layout=layout_for(ad))
    if is_grid(ad):
        prepare_grid_ctx(ctx)
    return ctx


def is_grid(ad):
    """A tier-G (sharded) adapter, as opposed to family 10's tier-P rows."""
    return getattr(ad, "tier", "P") == "G"


def licence_gate(ctx, what="publish"):
    """REFUSE a PUBLIC upload of data whose redistribution is unconfirmed.

    An adapter whose licence carries `redistribution_confirmed: False` (the
    producer has been asked and has not answered) may be built and checked,
    and may be published to the PRIVATE repository, but a public publish —
    or a public parts push, which is the same bytes under another prefix —
    needs `--allow-unconfirmed-licence`, said out loud on the command line.
    """
    ad = ctx.adapter
    lic = getattr(ad, "licence", {}) or {}
    if getattr(ad, "distribution", "public") != "public":
        return True
    if lic.get("redistribution_confirmed", True) is not False:
        return True
    if getattr(ctx.a, "allow_unconfirmed_licence", False):
        print(f"::warning::{ad.store}: the licence's redistribution terms are "
              f"UNCONFIRMED ({lic.get('pending', lic.get('name'))}) and "
              f"--allow-unconfirmed-licence was passed — {what} proceeds to "
              f"the public repository")
        return True
    sys.exit(f"REFUSING to {what} {ad.store} publicly: its licence "
             f"({lic.get('name')!r}) has redistribution_confirmed = False — "
             f"{lic.get('pending', 'the producer has not confirmed the terms')}"
             f". Build, check and keep the store; publish it once the terms "
             f"are confirmed (set redistribution_confirmed = True in the "
             f"adapter), or pass --allow-unconfirmed-licence to publish now "
             f"on your own judgement. Nothing has been uploaded.")


# ================================================================= stages ==
def stage_fetch(ctx):
    """Parts only; the store is the `assemble` stage's. Then, optionally,
    park every marked year on the Hub (`--push-parts`)."""
    f10b.stage_fetch(ctx, assemble_store_after=False)
    mark(ctx.root, "fetch")
    if getattr(ctx.a, "push_parts", False):
        push_parts(ctx)


def push_parts(ctx):
    import family10_parts_hub as ph
    licence_gate(ctx, "push parts of")
    kw = f10b.parts_hub_kwargs(ctx)
    kw = {k: v for k, v in kw.items() if k in ("partials", "hub", "private")}
    for y in ctx.years:
        if not marked(ctx.root, f"parts/{y}"):
            sys.exit(f"--push-parts: {ctx.adapter.store} {y} is not marked "
                     f"done locally — refusing to push a partial year")
    # ONE batched push for the lane: a lane of 150 early years costs a few
    # commits instead of 300 against the Hub's 256 an hour (E-082 wave 1).
    pushed = ph.push_many(ctx.adapter.store, ctx.years, ctx.work, **kw)
    if sorted(pushed) != sorted(int(y) for y in ctx.years):
        sys.exit(f"--push-parts: {ctx.adapter.store} pushed {len(pushed)} of "
                 f"{len(ctx.years)} year(s)")
    print(f"  push-parts: {len(pushed)} year(s) on the Hub under "
          f"{ctx.layout.hf_partials}/{ctx.adapter.store}/")
    return pushed


def stage_assemble(ctx):
    meta = f10b.assemble(ctx)
    mark(ctx.root, "assemble")
    ctx.prog.item("store", 1, {"N": meta["N"], "bin_first": meta["bin_first"]})
    return meta


def stage_publish(ctx):
    licence_gate(ctx)
    return f10b.stage_publish(ctx)


def hub_agrees(ctx, sm, check_name):
    """The Hub's store.json and manifest.json against the local records."""
    from huggingface_hub import hf_hub_download
    ad = ctx.adapter
    api, repo, tok = ctx.hub()
    f10b.check_publish_target(ad, ctx.layout, repo)
    prefix = ctx.layout.prefix(ad.store)
    scratch = os.path.join(ctx.scratch, "check_hub")
    got = {}
    for name in ("store.json", "manifest.json"):
        p = hf_hub_download(repo, f"{prefix}/{name}", repo_type="dataset",
                            token=tok, local_dir=scratch)
        got[name] = read_json(p, {})
    if got["store.json"].get("sha256") != sm.get("sha256"):
        sys.exit(f"{check_name} {ad.store}: the Hub's store.json does not "
                 f"carry the local sha256 block — the published store is not "
                 f"the one checked here")
    man = {e["name"]: e["sha256"] for e in got["manifest.json"]
           .get("files", [])}
    bad = [k for k, v in (sm.get("sha256") or {}).items()
           if man.get(k) != v]
    if bad:
        sys.exit(f"{check_name} {ad.store}: the Hub manifest disagrees on "
                 f"{bad[:8]}")
    return {"repo": repo, "prefix": prefix, "files": len(man)}


def stage_check(ctx):
    """The store as it will be (or was) served: sha256, E-079 §4, the Hub.

    Once the store is published, the Hub's `manifest.json` and `store.json`
    are downloaded (two small files) and their sha256 records must equal the
    local ones — the publish's restore check proved every uploaded file
    matched, and this proves the files that were checked are the files the
    Hub lists.
    """
    ad = ctx.adapter
    n = f10.verify_store(ctx.store)
    st = f10b.check_store(ctx.store, ad, chunk_rows=ctx.check_chunk)
    sm = read_json(os.path.join(ctx.store, "store.json"), {})
    want_schema = f10b.SCHEMA_OF_TIME_DTYPE[ctx.time_dtype]
    if st.schema_version != want_schema:
        sys.exit(f"check {ad.store}: the store is schema "
                 f"{st.schema_version}, the adapter's time_dtype "
                 f"{ctx.time_dtype} is schema {want_schema}")
    out = {"store": ad.store, "files_verified": n, "N": st.N,
           "schema_version": st.schema_version, "hub": None}
    if marked(ctx.root, "publish"):
        out["hub"] = hub_agrees(ctx, sm, "check")
    atomic_json(os.path.join(ctx.root, "check.json"),
                {**out, "at": utcnow(), "builder_git_sha": git_sha()})
    mark(ctx.root, "check")
    print(f"  check: {ad.store} — {n} file(s) verified, N={st.N:,}, schema "
          f"{st.schema_version}"
          + (f", Hub {out['hub']['repo']}:{out['hub']['prefix']} agrees"
             if out["hub"] else " (not published yet)"))
    return out


STAGE_FN = {"index": f10b.stage_index, "fetch": stage_fetch,
            "assemble": stage_assemble, "publish": stage_publish,
            "check": stage_check}


# ====================================================== tier G (sharded) ==
# E-082 wave 2. A tier-G adapter (`family1.sharded.GridAdapter`) yields
# FRAMES on fixed grids, not rows, and the store is the sharded layout of
# `ml/family1/sharded.py`. The stages keep their names and their order:
#
#   index     unchanged (`f10b.stage_index` — the adapter lists the archive,
#             reads one real file and writes plan.json).
#   fetch     per YEAR, where a year is the set of BINS whose first day falls
#             in it (a bin straddling New Year belongs to the year it
#             starts in, so every lane writes whole bins and no bin is ever
#             split between two lanes). `--start/--end` choose the bins;
#             every frame of a chosen bin is asked for. The year's parts are
#             its shards, their indices and one shard_index.npy per group,
#             written flat as `<group>__bin_<NNNN>.zst` etc. under
#             parts/<year>/, then counts.json (the ledger, with the grid
#             specs), then the year's marker. `--push-parts` and
#             `--parts-from-hub` move those files exactly as they move tier-P
#             parts (`family10_parts_hub`).
#   assemble  LINKS the year parts into <store>/<group>/<yyyy>/ and
#             concatenates the per-year shard indices — nothing is
#             re-compressed — then writes tile_grid.json, shard_index.npy and
#             store.json (sha256 of every file), and runs the full
#             `sharded.check_store`.
#   publish   the store's files (from store.json, never a directory listing),
#             store.json LAST, then every file downloaded back and hashed,
#             50 random tiles decompressed from the downloaded copy and
#             compared with the local ones, and — for a public repository —
#             a few of them read through the Hub's HTTP range path.
#   check     `sharded.check_store` (every tile decompressed) and, once
#             published, the Hub's store.json/manifest against the local.
GRID_UPLOAD_BATCH = 500
GRID_RESTORE_SAMPLE = 50
GRID_HTTP_SAMPLE = 5
OUTSIDE_RECORD = ("before_record", "after_record")


def prepare_grid_ctx(ctx):
    """Years, bins and wanted frames for a tier-G context."""
    ad = ctx.adapter
    by_year = {}
    for b in sh.bins_overlapping(ctx.t_lo, ctx.t_hi):
        by_year.setdefault(sh.bin_year(b), []).append(b)
    ctx.grid_bins = by_year
    ctx.years = sorted(by_year)
    ctx.grid_specs = ad.specs()
    F = int(ad.frames_per_bin)

    def wanted(year):
        return [(g, b, f) for g in sorted(ctx.grid_specs)
                for b in by_year.get(int(year), []) for f in range(F)]
    ctx.grid_wanted = wanted
    return ctx


def part_name(group, rel):
    """parts/<year>/ is FLAT (family10_parts_hub pushes one directory)."""
    return f"{group}__{os.path.basename(rel)}"


def _grid_item(item, ad):
    if len(item) == 3:
        b, f, arr = item
        return ad.store, int(b), int(f), arr, None
    g, b, f, arr, c = item
    return g, int(b), int(f), arr, c


def fetch_grid_year(ctx, y):
    """One year's bins -> shards under parts/<year>/, then the marker."""
    ad = ctx.adapter
    specs = ctx.grid_specs
    F = int(ad.frames_per_bin)
    d = ctx.year_dir(y)
    os.makedirs(d, exist_ok=True)
    wanted = ctx.grid_wanted(y)
    want = set(wanted)
    writers = {g: sh.ShardWriter(specs[g]) for g in specs}
    buf, seen = {}, set()
    entries = {g: [] for g in specs}
    counts = {"frames_wanted": len(wanted)}
    before = len(ctx.absent)
    t0 = time.time()

    def flush(g, b):
        frames, reasons = buf.pop((g, b))
        if all(fr is None for fr in frames) and \
                all(r in OUTSIDE_RECORD for r in reasons):
            f10b._merge_counts(counts, {"bins_outside_record": 1})
            return
        e = writers[g].write_bin(
            b, frames, os.path.join(d, part_name(g, sh.shard_relpath(b))),
            os.path.join(d, part_name(g, sh.index_relpath(b))))
        entries[g].append(e)

    for i, item in enumerate(ad.fetch_year(ctx, y), 1):
        g, b, f, arr, c = _grid_item(item, ad)
        key = (g, b, f)
        if key not in want:
            sys.exit(f"{ad.store}: fetch_year({y}) yielded {key}, which was "
                     f"not asked for — an adapter bug")
        if key in seen:
            sys.exit(f"{ad.store}: fetch_year({y}) yielded {key} twice")
        seen.add(key)
        c = dict(c or {})
        why = c.pop("frame_missing", None)
        slot = buf.setdefault((g, b), ([None] * F, [None] * F))
        day = str(sh.frame_day(b, f, ad.frame_seconds))
        if arr is None:
            if not why:
                sys.exit(f"{ad.store}: {key} came back as None with no "
                         f"`frame_missing` reason — a missing frame is "
                         f"counted by name, never silently")
            slot[1][f] = why
            c["frames_missing"] = {why: 1}
            c["missing_frames"] = [{"group": g, "bin": b, "frame": f,
                                    "day": day, "reason": why}]
        else:
            slot[0][f] = arr
            c["frames_present"] = {g: 1}
        f10b._merge_counts(counts, c)
        if sum(1 for ff in range(F) if (g, b, ff) in seen) == F:
            flush(g, b)
        if i % 25 == 0:
            ctx.prog.item(f"{ad.store} {y} {g} bin {b}", None,
                          {"frames": len(seen),
                           "elapsed_s": round(time.time() - t0, 1)})
    lost = ctx.absent[before:]
    if lost:
        print(f"  ::warning::{y}: NOT MARKED — {len(lost)} input(s) could not "
              f"be read ({'; '.join(e['unit'] for e in lost[:4])}"
              f"{' …' if len(lost) > 4 else ''}); the retry clears the year "
              f"and fetches it whole")
        return None
    unseen = [k for k in wanted if k not in seen]
    if unseen or buf:
        sys.exit(f"{ad.store}: fetch_year({y}) never answered {len(unseen)} "
                 f"frame(s) (first {unseen[:3]}) and reported no absence — a "
                 f"frame is either yielded (an array or None with a reason) "
                 f"or its input is noted absent")
    summary = {}
    for g in specs:
        sh.save_shard_index(os.path.join(d, f"{g}__shard_index.npy"),
                            entries[g], specs[g]["C"])
        es = entries[g]
        summary[g] = {
            "bins": len(es),
            "frames_present": sum(e["frames_present"] for e in es),
            "frames_missing": sum(e["frames_missing"] for e in es),
            "tiles_stored": sum(e["tiles_stored"] for e in es),
            "bytes": sum(e["nbytes"] for e in es)}
    counts["fetch_seconds"] = round(time.time() - t0, 1)
    rows = sum(v["frames_present"] for v in summary.values())
    atomic_json(os.path.join(d, "counts.json"),
                {"year": y, "tier": "G", "rows": rows, "parts": 0,
                 "groups": summary, "counts": counts,
                 "grids": json.loads(json.dumps(specs)), "at": utcnow()})
    mark(ctx.root, f"parts/{y}")
    print(f"  {y}: " + ", ".join(
        f"{g} {v['bins']} bin(s) {v['frames_present']} frame(s) "
        f"{v['bytes'] / 1e6:.1f} MB" for g, v in summary.items())
        + f" ({time.time() - t0:.1f}s)")
    return summary


def stage_fetch_grid(ctx):
    ad = ctx.adapter
    ad.fetch_preflight(ctx)
    if getattr(ctx.a, "parts_from_hub", False):
        import family10_parts_hub as ph
        ctx.prog.stage_start(f"pull {ad.store} parts", len(ctx.years))
        ph.pull(ad.store, ctx.years, ctx.work,
                allow_missing=bool(getattr(ctx.a, "allow_missing_years",
                                           False)),
                **f10b.parts_hub_kwargs(ctx))
    else:
        ctx.prog.stage_start(f"fetch {ad.store} (tier G)", len(ctx.years))
        for y in ctx.years:
            if marked(ctx.root, f"parts/{y}") and not ctx.a.force:
                print(f"  {y}: already fetched — skipping")
                continue
            shutil.rmtree(ctx.year_dir(y), ignore_errors=True)
            mp = marker(ctx.root, f"parts/{y}")
            if os.path.exists(mp):
                os.remove(mp)
            fetch_grid_year(ctx, y)
    f10b.fetch_absence_check(ctx)
    mark(ctx.root, "fetch")
    if getattr(ctx.a, "push_parts", False):
        push_parts(ctx)


def _link(src, dst):
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    if os.path.exists(dst):
        os.remove(dst)
    try:
        os.link(src, dst)
    except OSError:
        shutil.copyfile(src, dst)


def stage_assemble_grid(ctx):
    ad = ctx.adapter
    lay = ctx.layout
    specs = ctx.grid_specs
    want_specs = json.loads(json.dumps(specs))
    allow = bool(getattr(ctx.a, "allow_missing_years", False))
    ctx.prog.stage_start(f"{ad.store} store (tier G)", 1)
    dest = ctx.store
    shutil.rmtree(dest, ignore_errors=True)
    os.makedirs(dest)
    entries = {g: [] for g in specs}
    counts_all, bad, degraded = {}, [], []
    for y in ctx.years:
        yd = ctx.year_dir(y)
        if not marked(ctx.root, f"parts/{y}"):
            msg = f"{y}: no parts/{y}.done — the year's fetch did not finish"
            (degraded if allow else bad).append(msg)
            continue
        c = read_json(os.path.join(yd, "counts.json"), None)
        if not c or c.get("tier") != "G":
            bad.append(f"{y}: counts.json is missing or not a tier-G ledger")
            continue
        if c.get("grids") != want_specs:
            bad.append(f"{y}: the parts were written for a different grid "
                       f"declaration than this adapter's — a lane and a box "
                       f"running different code")
            continue
        for g in specs:
            ip = os.path.join(yd, f"{g}__shard_index.npy")
            if not os.path.exists(ip):
                bad.append(f"{y}: no {g}__shard_index.npy")
                continue
            es = sh.array_to_entries(sh.load_shard_index(ip))
            if len(es) != int(c["groups"][g]["bins"]):
                bad.append(f"{y} {g}: {len(es)} bin(s) in the index, the "
                           f"ledger says {c['groups'][g]['bins']}")
            for e in es:
                if sh.bin_year(e["bin"]) != y:
                    bad.append(f"{y} {g}: bin {e['bin']} belongs to "
                               f"{sh.bin_year(e['bin'])}")
                for rel in (e["shard"], e["index"]):
                    src = os.path.join(yd, part_name(g, rel))
                    if not os.path.exists(src):
                        bad.append(f"{y} {g}: {part_name(g, rel)} missing")
                        continue
                    _link(src, os.path.join(dest, g, rel))
                entries[g].append(e)
        f10b._merge_counts(counts_all, c.get("counts") or {})
    if bad:
        sys.exit(f"REFUSING to assemble {ad.store}:\n  " + "\n  ".join(bad)
                 + "\nRe-run the fetch stage with the same --work value, or "
                   "pass --allow-missing-years for a deliberately short "
                   "store.")
    groups, per_year = {}, {}
    for g, spec in specs.items():
        gd = os.path.join(dest, g)
        os.makedirs(gd, exist_ok=True)
        atomic_json(os.path.join(gd, "tile_grid.json"), spec)
        arr = sh.save_shard_index(os.path.join(gd, "shard_index.npy"),
                                  entries[g], spec["C"])
        vp = arr["valid_pixels"].sum(axis=0) if len(arr) else \
            np.zeros(spec["C"], np.int64)
        fp = int(arr["frames_present"].sum())
        for r in arr:
            yy = str(int(r["year"]))
            per_year.setdefault(yy, {}).setdefault(g, 0)
            per_year[yy][g] += int(r["frames_present"])
        groups[g] = {
            "prefix": g, "tile_grid": f"{g}/tile_grid.json",
            "shard_index": f"{g}/shard_index.npy",
            "H": spec["H"], "W": spec["W"], "C": spec["C"],
            "tile": spec["tile"], "n_tiles_y": spec["n_tiles_y"],
            "n_tiles_x": spec["n_tiles_x"],
            "frames_per_bin": spec["frames_per_bin"],
            "frame_seconds": spec["frame_seconds"], "dtype": spec["dtype"],
            "crs": spec["grid"].get("crs"),
            "bins": int(len(arr)),
            "bin_first": int(arr["bin"][0]) if len(arr) else None,
            "bin_last": int(arr["bin"][-1]) if len(arr) else None,
            "frames_present": fp,
            "frames_missing": int(arr["frames_missing"].sum()),
            "tiles_stored": int(arr["tiles_stored"].sum()),
            "bytes": int(arr["nbytes"].sum()),
            "valid_fraction": [
                (float(v) / (fp * spec["H"] * spec["W"]) if fp else None)
                for v in vp],
        }
    plan = read_json(os.path.join(ctx.root, "plan.json"), {})
    missing = counts_all.get("missing_frames") or []
    meta = {
        "family": lay.family, "tier": "G", "layout": "sharded",
        "format": sh.FORMAT, "store": ad.store, "title": ad.title,
        "family_version": lay.family_version,
        "family_code": getattr(ad, "family", None),
        "distribution": getattr(ad, "distribution", None),
        "licence": getattr(ad, "licence", None),
        "channels": ad.schema(), "C": ad.C, "dtype": ad.dtype,
        "frames_per_bin": int(ad.frames_per_bin),
        "frame_seconds": int(ad.frame_seconds),
        "epoch": str(f10b.START), "pentad_days": f10b.PENTAD_DAYS,
        "footprint": {"log2_fp": ad.log2_fp, "log2_dt": ad.log2_dt,
                      "note": "E-078 §2: log2(pixel_km / 27.83) and "
                              "log2(frame_days / 5); one token per pixel "
                              "centre"},
        "groups": groups,
        "per_year": per_year,
        "frames_missing_by_reason": counts_all.get("frames_missing") or {},
        "missing_frames": missing,
        "counts": {k: v for k, v in counts_all.items()
                   if k != "missing_frames"},
        "normalisation": "RAW units — not z-scored, not anomalised",
        "qc_policy": ad.qc_policy,
        "date_range": [str(ctx.d_lo), str(ctx.d_hi)],
        "bins_requested": [ctx.b_lo, ctx.b_hi],
        "resume_granularity": ("year — the bins whose FIRST day falls in "
                               "the year; every frame of a chosen bin is "
                               "fetched"),
        "sources": list(ad.sources), "verified": ad.verified,
        "plan": {k: plan.get(k) for k in ("dataset", "url", "version")
                 if k in plan},
        "source_dir": (f"file://{ctx.source_dir}" if ctx.source_dir
                       else None),
        "builder": lay.builder, "builder_git_sha": git_sha(),
        "built_at": utcnow(),
    }
    if ad.notes:
        meta["notes"] = ad.notes
    if getattr(ad, "distribution_override", None):
        meta["distribution_override"] = ad.distribution_override
    if ctx.absent or degraded:
        meta["degraded"] = {"allow_missing_years": allow,
                            "inputs_not_read": ctx.absent,
                            "years_admitted_unmarked": degraded}
    names = sh.store_files(dest, sorted(groups))
    meta["sha256"] = {n: f10b.sha256(os.path.join(dest, n)) for n in names}
    atomic_json(os.path.join(dest, "store.json"), meta)
    st = sh.check_store(dest)
    mark(ctx.root, "assemble")
    ctx.prog.item("store", 1, {"files": st["files"]})
    print(f"  store: {ad.store} — " + ", ".join(
        f"{g} {v['bins']} bin(s), {v['frames_present']} frame(s), "
        f"{v['tiles_stored']} tile(s), {v['bytes'] / 1e6:.1f} MB"
        for g, v in groups.items()) + f" -> {dest}")
    return meta


def _grid_store_names(ctx):
    dest = ctx.store
    sm = read_json(os.path.join(dest, "store.json"), {})
    names = sorted(sm.get("sha256") or {})
    if not names or sm.get("tier") != "G":
        sys.exit(f"cannot publish: {dest}/store.json is not a tier-G store "
                 f"with a sha256 block. Re-run the assemble stage.")
    have = set()
    for dp, _, fs in os.walk(dest):
        for n in fs:
            have.add(os.path.relpath(os.path.join(dp, n), dest)
                     .replace(os.sep, "/"))
    have.discard("store.json")
    extra = sorted(have - set(names))
    miss = sorted(set(names) - have)
    if miss:
        sys.exit(f"cannot publish: {len(miss)} file(s) named in store.json "
                 f"are missing ({miss[:4]})")
    if extra:
        sys.exit(f"cannot publish: {len(extra)} file(s) in {dest} are not in "
                 f"store.json ({extra[:4]}) — a leftover of another build")
    return sm, names


def _pick_tiles(dest, groups, k, seed):
    """k random STORED tiles across the groups: (g, b, f, ty, tx, off, n)."""
    import random
    pool = []
    for g in groups:
        arr = sh.load_shard_index(os.path.join(dest, g, "shard_index.npy"))
        for r in arr:
            if int(r["tiles_stored"]) == 0:
                continue
            idx = np.load(os.path.join(dest, g, r["index"]))
            for f, ty, tx in zip(*np.nonzero(idx[..., 1] > 0)):
                pool.append((g, int(r["bin"]), int(f), int(ty), int(tx),
                             int(idx[f, ty, tx, 0]), int(idx[f, ty, tx, 1])))
    rng = random.Random(seed)
    return rng.sample(pool, min(k, len(pool)))


def _download(repo, rel, tok, dest_dir):
    from huggingface_hub import hf_hub_download
    return hf_hub_download(repo, rel, repo_type="dataset", token=tok,
                           local_dir=dest_dir)


def http_verify(repo, prefix, picks, dest, private):
    """A few picked tiles read through the Hub's HTTP range path and compared
    with the local bytes. Public repositories only (an anonymous read is
    what a consumer does). A 200 where a 206 was asked for, or different
    bytes, is fatal; a network failure is a warning (§5.17)."""
    if private or not picks:
        return {"checked": 0, "skipped": "private" if private else "none"}
    n = 0
    for (g, b, f, ty, tx, off, ln) in picks:
        url = f"https://huggingface.co/datasets/{repo}/resolve/main/{prefix}/{g}"
        try:
            got = sh.read_tile(url, b, f, ty, tx, raw=True)
        except (ValueError, sh.ShardError) as e:
            sys.exit(f"HTTP RESTORE FAILED {g} bin {b} ({f},{ty},{tx}): {e}")
        except (IOError, OSError) as e:
            print(f"::warning::the Hub's HTTP range path could not be read "
                  f"({type(e).__name__}: {e}) — the downloaded copy was "
                  f"verified; this read is not")
            return {"checked": n, "warning": str(e)[:300]}
        want = sh.open_group(os.path.join(dest, g)).read_tile(
            b, f, ty, tx, raw=True)
        if not np.array_equal(got, want, equal_nan=got.dtype.kind == "f"):
            sys.exit(f"HTTP RESTORE MISMATCH {g} bin {b} ({f},{ty},{tx})")
        n += 1
    return {"checked": n}


def stage_publish_grid(ctx):
    ad = ctx.adapter
    lay = ctx.layout
    dest = ctx.store
    licence_gate(ctx)
    sm, names = _grid_store_names(ctx)
    groups = sorted(sm["groups"])
    sh.check_store(dest)
    # the restore check downloads one folder (<group>/<yyyy>) at a time
    folders = {}
    for n in names:
        folders.setdefault(os.path.dirname(n), []).append(n)
    biggest = max(sum(os.path.getsize(os.path.join(dest, n)) for n in v)
                  for v in folders.values())
    free = shutil.disk_usage(ctx.scratch).free
    if free < biggest * f10b.RESTORE_HEADROOM:
        sys.exit(f"REFUSING to publish: the largest folder is "
                 f"{biggest / 1e9:.2f} GB and {ctx.scratch} has "
                 f"{free / 1e9:.2f} GB free. Nothing has been uploaded.")
    api, repo, tok = ctx.hub()
    f10b.check_publish_target(ad, lay, repo)
    api.create_repo(repo, repo_type="dataset", exist_ok=True,
                    private=lay.private)
    prefix = lay.prefix(ad.store)
    ctx.prog.stage_start(f"publish {ad.store}", len(names) + 1)
    batches = [names[i:i + GRID_UPLOAD_BATCH]
               for i in range(0, len(names), GRID_UPLOAD_BATCH)]
    for i, chunk in enumerate(batches, 1):
        f10b.hub_commit(api, repo, f10b.hub_add_ops(
            [(f"{prefix}/{n}", os.path.join(dest, n)) for n in chunk]),
            f"{lay.label} ({ad.store}): {len(chunk)} file(s), batch "
            f"{i}/{len(batches)}")
    # store.json LAST: a consumer that finds it finds every file it names
    f10b.hub_commit(api, repo, f10b.hub_add_ops(
        [(f"{prefix}/store.json", os.path.join(dest, "store.json"))]),
        f"{lay.label} ({ad.store}): store.json")
    seed = int(f10b.sha256(os.path.join(dest, "store.json"))[:8], 16)
    picks = _pick_tiles(dest, groups, GRID_RESTORE_SAMPLE, seed)
    by_folder = {}
    for p in picks:
        by_folder.setdefault(f"{p[0]}/{sh.shard_relpath(p[1])}", []).append(p)
    scratch = os.path.join(ctx.scratch, "verify_grid")
    from family1.adapters import _common as cm
    checked = 0
    tiles_ok = 0
    for folder in sorted(folders) + [""]:
        rels = folders.get(folder, []) if folder else ["store.json"]
        shutil.rmtree(scratch, ignore_errors=True)

        def get(n):
            return n, _download(repo, f"{prefix}/{n}", tok, scratch)
        for n, p in cm.ordered_map(get, rels, 1 if len(rels) < 4 else 8):
            want = (sm["sha256"][n] if n != "store.json"
                    else f10b.sha256(os.path.join(dest, "store.json")))
            got = f10b.sha256(p)
            if got != want:
                sys.exit(f"RESTORE MISMATCH {n}: uploaded {want}, "
                         f"downloaded {got} — the publish is not "
                         f"trustworthy")
            checked += 1
            key = n if n.endswith(".zst") else None
            for (g, b, f, ty, tx, off, ln) in by_folder.get(key, []):
                with open(p, "rb") as fh:
                    fh.seek(off)
                    blob = fh.read(ln)
                grp = sh.open_group(os.path.join(dest, g))
                back = grp.decode_tile(blob)
                here = grp.read_tile(b, f, ty, tx, raw=True)
                if not np.array_equal(back, here,
                                      equal_nan=back.dtype.kind == "f"):
                    sys.exit(f"RESTORE MISMATCH tile {g} bin {b} "
                             f"({f},{ty},{tx})")
                tiles_ok += 1
        ctx.prog.item(folder or "store.json", checked, {"files": checked})
    shutil.rmtree(scratch, ignore_errors=True)
    if tiles_ok != len(picks):
        sys.exit(f"restore: {tiles_ok} of {len(picks)} sampled tiles were "
                 f"checked — the sample and the downloads disagree")
    http = http_verify(repo, prefix, picks[:GRID_HTTP_SAMPLE], dest,
                       lay.private)
    man = {"family": lay.family, "tier": "G", "layout": "sharded",
           "store": ad.store, "repo": repo, "prefix": prefix,
           "groups": sm["groups"], "channels": sm.get("channels"),
           "date_range": sm.get("date_range"),
           "restore": {"files": checked, "tiles_decompressed": tiles_ok,
                       "http_range": http},
           "builder_git_sha": git_sha(), "built_at": utcnow(),
           "files": [{"name": n, "bytes": os.path.getsize(
               os.path.join(dest, n)), "sha256": sm["sha256"][n]}
               for n in names]}
    mp = os.path.join(ctx.root, "manifest.json")
    atomic_json(mp, man)
    f10b.hub_upload_with_backoff(api, repo, mp, f"{prefix}/manifest.json",
                                 f"{lay.label} ({ad.store}): manifest")
    mark(ctx.root, "publish")
    print(f"  publish: {checked} file(s) verified by restore, {tiles_ok} "
          f"tile(s) decompressed, HTTP range {http} -> "
          f"https://huggingface.co/datasets/{repo}/tree/main/{prefix}")
    return man


def stage_check_grid(ctx):
    ad = ctx.adapter
    st = sh.check_store(ctx.store)
    sm = read_json(os.path.join(ctx.store, "store.json"), {})
    out = {"store": ad.store, "tier": "G", "files_verified": st["files"],
           "groups": st["groups"], "hub": None}
    if marked(ctx.root, "publish"):
        out["hub"] = hub_agrees(ctx, sm, "check")
    atomic_json(os.path.join(ctx.root, "check.json"),
                {**out, "at": utcnow(), "builder_git_sha": git_sha()})
    mark(ctx.root, "check")
    print(f"  check: {ad.store} — {st['files']} file(s) verified, "
          + ", ".join(f"{g}: {v['tiles_checked']} tile(s) decompressed"
                      for g, v in st["groups"].items())
          + (f", Hub {out['hub']['repo']} agrees" if out["hub"]
             else " (not published yet)"))
    return out


GRID_STAGE_FN = {"index": f10b.stage_index, "fetch": stage_fetch_grid,
                 "assemble": stage_assemble_grid,
                 "publish": stage_publish_grid, "check": stage_check_grid}


def stage_fns(ad):
    return GRID_STAGE_FN if is_grid(ad) else STAGE_FN


def stage_probe_grid(a, adapter_cls, month):
    """One calendar month of a tier-G source, through the real writer."""
    y, m = parse_month(month)
    ns, root = probe_namespace(a, adapter_cls.store, y, m)
    ad = adapter_cls()
    if not ns.source_dir:
        credentials_preflight(ad)
    ctx = f10b.Ctx(ns, adapter=ad, layout=layout_for(ad))
    prepare_grid_ctx(ctx)
    t0 = time.time()
    ctx.reset_bytes()
    f10b.stage_index(ctx)
    bytes_index = ctx.bytes_fetched
    t_index = time.time() - t0
    ad.fetch_preflight(ctx)
    lo, hi = f10b.month_bounds_s(y, m)
    F, fs = int(ad.frames_per_bin), int(ad.frame_seconds)
    specs = ctx.grid_specs
    wanted = [(g, b, f) for g in sorted(specs)
              for b in sh.bins_overlapping(lo, hi) for f in range(F)
              if lo <= sh.frame_start_seconds(b, f, fs) <= hi]
    slots = sorted({sh.frame_start_seconds(b, f, fs)
                    for (_g, b, f) in wanted})
    scratch = os.path.join(root, "shards")
    shutil.rmtree(scratch, ignore_errors=True)
    os.makedirs(scratch)
    writers = {g: sh.ShardWriter(specs[g]) for g in specs}
    acc = {g: {"frames": 0, "missing": {}, "present_slots": [0] * len(slots),
               "frame_bytes": [], "tile_bytes": [], "tiles_empty": 0,
               "valid": np.zeros(specs[g]["C"], np.int64),
               "valid_per_frame": [], "encode_s": 0.0} for g in specs}
    lo_b, hi_b = ad.bounds()
    counts, oob_stored = {}, 0
    ctx.reset_bytes()
    t1 = time.time()
    for item in ad.fetch_frames(ctx, wanted):
        g, b, f, arr, c = _grid_item(item, ad)
        c = dict(c or {})
        why = c.pop("frame_missing", None)
        f10b._merge_counts(counts, c)
        A = acc[g]
        if arr is None:
            A["missing"][why] = A["missing"].get(why, 0) + 1
            continue
        v = np.asarray(arr, np.float64).reshape(-1, specs[g]["C"])
        with np.errstate(invalid="ignore"):
            oob_stored += int((np.isfinite(v) & ((v < lo_b) | (v > hi_b)))
                              .sum())
        frames = [None] * F
        frames[f] = arr
        te = time.time()
        sp_, ip_ = (os.path.join(scratch, f"{g}_{b}_{f}.zst"),
                    os.path.join(scratch, f"{g}_{b}_{f}.idx.npy"))
        e = writers[g].write_bin(b, frames, sp_, ip_)
        A["encode_s"] += time.time() - te
        idx = np.load(ip_)
        lens = idx[f, :, :, 1].ravel()
        A["tile_bytes"] += [int(x) for x in lens if x > 0]
        A["tiles_empty"] += int((lens == 0).sum())
        A["frame_bytes"].append(int(e["nbytes"]))
        A["valid"] += np.asarray(e["valid_pixels"], np.int64)
        A["valid_per_frame"].append(round(float(e["valid_fraction"][0]), 6))
        A["frames"] += 1
        A["present_slots"][slots.index(sh.frame_start_seconds(b, f, fs))] = 1
        os.remove(sp_)
        os.remove(ip_)
    wall = time.time() - t1
    fetched = ctx.bytes_fetched
    shutil.rmtree(scratch, ignore_errors=True)
    out_groups, est_total, formula_total = {}, 0, 0
    total_frames = sum(A["frames"] for A in acc.values())
    for g, A in acc.items():
        spec = specs[g]
        H, W, C = spec["H"], spec["W"], spec["C"]
        n = A["frames"]
        fb = np.array(A["frame_bytes"] or [0], np.float64)
        tb = np.array(A["tile_bytes"] or [0], np.float64)
        vfrac = (A["valid"] / (n * H * W)) if n else None
        rec = ad.record_frames(ctx, g)
        idx_bytes = spec["index_header_bytes"] + 16 * F * spec["n_tiles_y"] \
            * spec["n_tiles_x"]
        est = formula = None
        if rec and n:
            est = int(fb.mean() * rec + (rec / F) * idx_bytes)
            formula = int(H * W * rec * float(vfrac.mean()) * 2 * C * 1.2)
            est_total += est
            formula_total += formula
        out_groups[g] = {
            "H": H, "W": W, "C": C, "dtype": spec["dtype"],
            "tiles_per_frame": spec["n_tiles_y"] * spec["n_tiles_x"],
            "frames_requested": sum(1 for w in wanted if w[0] == g),
            "frames_fetched": n,
            "frames_missing": A["missing"],
            "frame_present_by_slot": A["present_slots"],
            "valid_fraction": ({nm: round(float(vfrac[i]), 6) for i, nm in
                                enumerate(ad.channel_names)} if n else None),
            "valid_fraction_per_frame": A["valid_per_frame"],
            "bytes_per_frame": {"mean": round(float(fb.mean()), 1),
                                "min": int(fb.min()), "max": int(fb.max())},
            "bytes_per_stored_tile": {
                "mean": round(float(tb.mean()), 1),
                "median": float(np.median(tb)),
                "p90": float(np.percentile(tb, 90)), "max": int(tb.max())},
            "tiles_stored": len(A["tile_bytes"]),
            "tiles_empty": A["tiles_empty"],
            "raw_bytes_per_frame": H * W * C * sh.DTYPES[spec["dtype"]]
            .itemsize,
            # `fb.mean()` is 0 when every tile of the probed frame is
            # missing-valued and so nothing was stored (a lossyear sub-tile
            # that is all ocean, a polar group with no retrieval). That is a
            # legitimate frame, not a broken one, so the ratio is None rather
            # than a ZeroDivisionError that loses the whole probe.
            "compression_ratio": (round(H * W * C * sh.DTYPES[spec["dtype"]]
                                        .itemsize / float(fb.mean()), 2)
                                  if n and fb.mean() else None),
            "bytes_per_valid_pixel": (round(float(fb.sum())
                                            / max(int(A["valid"].sum()), 1),
                                            4) if n else None),
            "encode_seconds_per_frame": (round(A["encode_s"] / n, 3)
                                         if n else None),
            "record_frames": rec,
            "index_bytes_per_bin": idx_bytes,
            "estimate_store_bytes": est,
            "note_formula_bytes": formula,
            "note_formula": "pixels x frames x valid fraction x 2C x 1.2",
        }
    out = {
        "store": ad.store, "family": ad.family, "tier": "G",
        "layout": "sharded",
        "distribution": getattr(ad, "distribution", None),
        "licence_redistribution_confirmed":
            (ad.licence or {}).get("redistribution_confirmed", True),
        "month": f"{y:04d}-{m:02d}",
        "source": "local:" + ns.source_dir if ns.source_dir else
                  (ad.sources[0] if ad.sources else ""),
        "frames_per_bin": F, "frame_seconds": fs,
        "zstd_level": int(ad.zstd_level),
        "frames_fetched": total_frames,
        "groups": out_groups,
        "out_of_bounds": (counts.get("out_of_bounds") or {}),
        "out_of_bounds_stored": oob_stored,
        "inputs_not_read": ctx.absent,
        "counts": counts,
        "bytes_fetched": int(fetched),
        "bytes_index": int(bytes_index),
        "bytes_fetched_per_frame": (round(fetched / total_frames, 1)
                                    if total_frames else None),
        "wall_seconds": round(wall, 2),
        "wall_seconds_index": round(t_index, 2),
        "seconds_per_frame": (round(wall / total_frames, 3)
                              if total_frames else None),
        "estimate_store_bytes": est_total or None,
        "note_formula_bytes": formula_total or None,
        "note_estimate": getattr(ad, "note_estimate", None),
        "runner": os.environ.get("RUNNER_NAME", _platform.node()),
        "builder_git_sha": git_sha(),
        "at": utcnow(),
    }
    os.makedirs(root, exist_ok=True)
    p = os.path.join(root, f"{y:04d}-{m:02d}.json")
    atomic_json(p, out)
    print(json.dumps(out, indent=1))
    print(f"probe -> {p}")
    if total_frames == 0:
        sys.exit(f"probe {ad.store} {month}: ZERO frames — a broken listing "
                 f"or reader, not a measurement (the 2026-09-14 rule)")
    if oob_stored:
        sys.exit(f"probe {ad.store} {month}: {oob_stored} value(s) outside "
                 f"the channel bounds reached the writer (contract rule 3)")
    if ctx.absent:
        sys.exit(f"probe {ad.store} {month}: {len(ctx.absent)} input(s) "
                 f"could not be read — see inputs_not_read")
    return out, p


def check_grid_smoke(ctx, truth):
    """Every frame of the store, read back tile by tile, against the truth."""
    ad = ctx.adapter
    n_frames = n_absent = n_skipped = 0
    by_reason = {}
    for g in ctx.grid_specs:
        grp = sh.ShardedGroup(os.path.join(ctx.store, g))
        bins = set(int(b) for b in grp.shard_index["bin"])
        keys = sorted(k for k in truth if k[0] == g)
        for (_g, b, f) in keys:
            want, why = truth[(g, b, f)]
            if why:
                by_reason[why] = by_reason.get(why, 0) + 1
            if b not in bins:
                assert all(truth[(g, b, ff)][1] in OUTSIDE_RECORD
                           for ff in range(ad.frames_per_bin)), (g, b)
                n_skipped += 1
                continue
            got = grp.read_frame(b, f, raw=True)
            if want is None:
                assert got is None, (g, b, f, why)
                n_absent += 1
                continue
            assert got is not None, (g, b, f)
            assert got.shape == want.shape and got.dtype == want.dtype
            # `equal_nan` for a float group: NaN IS the missing value there
            # (`sharded.DTYPES`), so a plain array_equal can never pass on a
            # float16 store with any missing pixel. `stage_publish_grid` and
            # `http_verify` already compare this way.
            assert np.array_equal(got, want,
                                  equal_nan=got.dtype.kind == "f"), (g, b, f)
            n_frames += 1
    sm = read_json(os.path.join(ctx.store, "store.json"), {})
    stored = dict(sm.get("frames_missing_by_reason") or {})
    # the ledger counts EVERY missing frame the fetch was told about,
    # including those of bins that were not written (wholly outside the
    # record), so it equals the truth's reasons exactly
    assert stored == by_reason, (stored, by_reason)
    return {"frames_equal": n_frames, "frames_absent": n_absent,
            "frames_in_skipped_bins": n_skipped, "by_reason": by_reason}


def run_smoke_grid(store, root=None, keep=False, probe=True):
    cls = REGISTRY[store]
    ad = cls()
    lay = layout_for(ad)
    start, end = ad.smoke_window
    tmp = root or tempfile.mkdtemp(prefix=f"f1smoke_{store}_")
    src = os.path.join(tmp, "src")
    work = os.path.join(tmp, "work")
    os.makedirs(work, exist_ok=True)
    t0 = time.time()
    d_lo, d_hi = f10b.parse_date(start), f10b.parse_date(end)
    truth = ad.smoke_sources(src, d_lo, d_hi)
    print(f"smoke     {store}: sources -> {src} ({len(truth)} truth "
          f"frame(s), {time.time() - t0:.1f}s)")
    a = argparse.Namespace(
        store=store, work=work, source_dir=src, start=start, end=end,
        stage="all", force=False, attempts=1, qc_keep=2,
        check_chunk_rows=f10b.CHECK_CHUNK_ROWS, assemble="auto",
        parts_from_hub=False, push_parts=False, allow_missing_years=False,
        allow_unconfirmed_licence=False, probe_month="", smoke=True)
    ctx = f10b.Ctx(a, adapter=ad, layout=lay)
    prepare_grid_ctx(ctx)
    f10b.run_stages(ctx, ["index", "fetch", "assemble", "check"],
                    stage_fn=GRID_STAGE_FN, deps=DEPS)
    res = check_grid_smoke(ctx, truth)
    print(f"smoke     {store} OK in {time.time() - t0:.1f}s — {res}")
    result = {"work": work, "truth": truth, "ctx": ctx, "check": res}
    if probe:
        month = getattr(ad, "smoke_probe_month", None) or \
            f"{d_lo.year:04d}-{d_lo.month:02d}"
        y, m = parse_month(month)
        lo, hi = f10b.month_bounds_s(y, m)
        pa = argparse.Namespace(**{**vars(a), "work": work})
        out, path = stage_probe_grid(pa, cls, month)
        for g in ctx.grid_specs:
            want = sum(1 for (gg, b, f), (arr, why) in truth.items()
                       if gg == g and arr is not None and lo <=
                       sh.frame_start_seconds(b, f, ad.frame_seconds) <= hi)
            got = out["groups"][g]["frames_fetched"]
            assert got == want, (g, got, want)
        assert out["bytes_fetched"] > 0, "the probe counted no bytes"
        result["probe"] = out
        print(f"smoke     probe {month}: {out['frames_fetched']} frame(s) == "
              f"truth -> {path}")
    if not keep and root is None:
        shutil.rmtree(tmp, ignore_errors=True)
    return result


# ================================================================== probe ==
def parse_month(s):
    try:
        y, m = (int(x) for x in str(s).split("-"))
        dt.date(y, m, 1)
    except Exception:                                           # noqa: BLE001
        sys.exit(f"--probe-month {s!r}: expected YYYY-MM")
    return y, m


def probe_namespace(a, store, y, m):
    ndays = calendar.monthrange(y, m)[1]
    root = os.path.join(os.path.abspath(a.work), "probe", store)
    ns = vars(a).copy()
    ns.update(store=store, work=os.path.join(root, "work"),
              start=f"{y:04d}-{m:02d}-01", end=f"{y:04d}-{m:02d}-{ndays:02d}",
              force=True, parts_from_hub=False, push_parts=False)
    return argparse.Namespace(**ns), root


def stage_probe(a, adapter_cls, month):
    """One calendar month through the REAL adapter, measured (E-082 §1.2)."""
    y, m = parse_month(month)
    ns, root = probe_namespace(a, adapter_cls.store, y, m)
    ad = adapter_cls()
    if not ns.source_dir:
        credentials_preflight(ad)
    ctx = f10b.Ctx(ns, adapter=ad, layout=layout_for(ad))
    t0 = time.time()
    ctx.reset_bytes()
    f10b.stage_index(ctx)
    bytes_index = ctx.bytes_fetched
    t_index = time.time() - t0
    ad.fetch_preflight(ctx)
    ctx.reset_bytes()
    t1 = time.time()
    lo, hi = f10b.month_bounds_s(y, m)
    ndays = calendar.monthrange(y, m)[1]
    C = ad.C
    rows = 0
    per_day = np.zeros(ndays, np.int64)
    nan = np.zeros(C, np.int64)
    oob_stored = np.zeros(C, np.int64)
    plats = set()
    qc = {}
    counts = {}
    outside = 0
    lo_b, hi_b = ad.bounds()
    for _label, r, c in ad.fetch_month(ctx, y, m):
        if c:
            f10b._merge_counts(counts, c)
        if r is None or not len(r["bin"]):
            continue
        t = np.asarray(r["time_s"], np.int64)
        inside = (t >= lo) & (t <= hi)
        outside += int((~inside).sum())
        if not inside.all():
            r = {k: v[inside] for k, v in r.items()}
            t = t[inside]
        n = int(t.size)
        rows += n
        per_day += np.bincount((t - lo) // f10b.SECONDS_PER_DAY,
                               minlength=ndays)[:ndays]
        v = np.asarray(r["values"], np.float32).reshape(n, C)
        fin = np.isfinite(v)
        nan += (~fin).sum(axis=0)
        with np.errstate(invalid="ignore"):
            oob_stored += (fin & ((v < lo_b) | (v > hi_b))).sum(axis=0)
        plats.update(np.unique(np.asarray(r["platform"], np.int64)).tolist())
        u, k = np.unique(np.asarray(r["qc"], np.int64), return_counts=True)
        for a_, b_ in zip(u.tolist(), k.tolist()):
            qc[str(a_)] = qc.get(str(a_), 0) + b_
    wall = time.time() - t1
    fetched = ctx.bytes_fetched
    names = ad.channel_names
    oob_counted = (counts.get("out_of_bounds") or {})
    pm = None
    if getattr(ad, "platform_meta", False):
        table = ad.platforms(ctx) or {}
        have = {int(k) for k in table}
        pm = {"entries": len(table),
              "probe_platforms_without_entry": len(plats - have)}
    out = {
        "store": ad.store, "family": ad.family,
        "distribution": getattr(ad, "distribution", None),
        "month": f"{y:04d}-{m:02d}",
        "source": "local:" + ns.source_dir if ns.source_dir else
                  (ad.sources[0] if ad.sources else ""),
        "rows": rows,
        "rows_per_day": per_day.tolist(),
        "days_with_no_rows": int((per_day == 0).sum()),
        "distinct_platforms": len(plats),
        "channels": names,
        "nan_fraction": {nm: (round(int(nan[i]) / rows, 6) if rows else None)
                         for i, nm in enumerate(names)},
        "out_of_bounds": {nm: int(oob_counted.get(nm, 0)) for nm in names},
        "out_of_bounds_stored": {nm: int(oob_stored[i])
                                 for i, nm in enumerate(names)},
        "rows_outside_month_dropped_by_probe": outside,
        "qc": qc,
        "counts": counts,
        "counts_scope": getattr(ad, "fetch_month_scope", "year"),
        "bytes_fetched": int(fetched),
        "bytes_index": int(bytes_index),
        "wall_seconds": round(wall, 2),
        "wall_seconds_index": round(t_index, 2),
        "bytes_per_row": (round(fetched / rows, 3) if rows else None),
        "stored_bytes_per_row": f10b.row_bytes(C, ctx.time_dtype),
        "time_dtype": ctx.time_dtype,
        "schema_version": ctx.schema_version,
        "platforms": pm,
        "estimate_rows_per_year": None,
        "estimate_store_bytes": None,
        "note_estimate": None,
        "runner": os.environ.get("RUNNER_NAME", _platform.node()),
        "builder_git_sha": git_sha(),
        "at": utcnow(),
    }
    os.makedirs(root, exist_ok=True)
    p = os.path.join(root, f"{y:04d}-{m:02d}.json")
    atomic_json(p, out)
    print(json.dumps(out, indent=1))
    print(f"probe -> {p}")
    # THE PROBE'S OWN ASSERTIONS, after the file is written so a refusal
    # still leaves the measurement behind.
    if rows == 0:
        sys.exit(f"probe {ad.store} {month}: ZERO rows. A month of a live "
                 f"archive with nothing in it is a broken listing or parser, "
                 f"not a measurement (the 2026-09-14 rule).")
    if int(oob_stored.sum()):
        sys.exit(f"probe {ad.store} {month}: {int(oob_stored.sum())} stored "
                 f"value(s) lie outside their channel bounds — the adapter "
                 f"must set them to NaN and count them (contract rule 3)")
    return out, p


# ================================================================== smoke ==
def run_smoke(store, root=None, keep=False, probe=True):
    """The adapter's synthetic archive through index + fetch + assemble, the
    store checked row for row, then (by default) the probe on the same
    archive — its row count must equal the truth's rows in that month."""
    cls = REGISTRY[store]
    ad = cls()
    if is_grid(ad):
        return run_smoke_grid(store, root=root, keep=keep, probe=probe)
    if not hasattr(ad, "smoke_sources"):
        sys.exit(f"{store}: the adapter has no smoke_sources(root, d_lo, d_hi) "
                 f"— its smoke lives in tests/test_family1_{store}.py")
    lay = layout_for(ad)
    result = {}

    def after(ctx, truth, src):
        if not probe:
            return
        month = getattr(ad, "smoke_probe_month", None) or \
            f"{ctx.d_lo.year:04d}-{ctx.d_lo.month:02d}"
        y, m = parse_month(month)
        lo, hi = f10b.month_bounds_s(y, m)
        want = sum(1 for r in truth if lo <= int(r["t"]) <= hi)
        pa = argparse.Namespace(**{**vars(ctx.a), "work": ctx.work,
                                   "source_dir": src})
        out, path = stage_probe(pa, cls, month)
        assert out["rows"] == want, (out["rows"], want)
        assert sum(out["rows_per_day"]) == want
        assert out["bytes_fetched"] > 0, "the probe counted no bytes"
        result["probe"] = out
        print(f"smoke     probe {month}: {out['rows']} row(s) == truth, "
              f"{out['bytes_fetched']} byte(s) counted -> {path}")

    work, truth = f10b.run_smoke(
        store, root=root, keep=keep, adapter=ad, layout=lay,
        make_sources=lambda r, s, lo, hi: ad.smoke_sources(r, lo, hi),
        extra_ns={"parts_from_hub": False, "push_parts": False,
                  "allow_missing_years": False, "assemble": "auto"},
        after=after)
    result.update(work=work, truth=truth)
    return result


# =================================================================== main ==
def build_parser():
    ap = argparse.ArgumentParser(
        description="Build a family-1 (1.gf, 1.0.tf, 0.9.tf) tier-P store. "
                    "See ml/plans/E082_family1_builds.md and "
                    "ml/family1/ADAPTER_CONTRACT.md.")
    ap.add_argument("--store", default="", choices=[""] + sorted(REGISTRY),
                    help="the store, from ml/family1/adapters/ (required "
                         "unless --check-credentials)")
    ap.add_argument("--work", default="",
                    help="the build directory (default ml/cache/family1_<x> "
                         "per the adapter's family). RE-RUN WITH THE SAME "
                         "VALUE TO RESUME.")
    ap.add_argument("--stage", default="all",
                    help="`all` (= index,fetch,assemble,publish,check), a "
                         "comma list of those, or `probe` (with "
                         "--probe-month). Order is fixed.")
    ap.add_argument("--probe-month", default="",
                    help="YYYY-MM: the month `--stage probe` measures")
    ap.add_argument("--start", default="",
                    help="first day (YYYY-MM-DD); default the adapter's "
                         "first year")
    ap.add_argument("--end", default=str(END), help="last day (YYYY-MM-DD)")
    ap.add_argument("--source-dir", default="",
                    help="read every source file from local disk (tests, "
                         "smoke)")
    ap.add_argument("--smoke", action="store_true",
                    help="the adapter's synthetic archive end to end, plus "
                         "the probe, in seconds, with no network")
    ap.add_argument("--smoke-dir", default="",
                    help="with --smoke: keep the temp tree here")
    ap.add_argument("--no-probe", action="store_true",
                    help="with --smoke: skip the probe")
    ap.add_argument("--qc-keep", type=int, default=2,
                    help="the worst QC grade kept (recorded in store.json)")
    ap.add_argument("--assemble", default="auto",
                    choices=("auto", "streaming", "memory"),
                    help="the assembler (auto streams above "
                         f"{f10b.STREAM_ROWS:,} rows)")
    ap.add_argument("--parts-from-hub", action="store_true",
                    help="stage fetch pulls the years' parts from "
                         "partials/<family>/<store>/<year>/ on the adapter's "
                         "repository instead of the source (the keyless box)")
    ap.add_argument("--push-parts", action="store_true",
                    help="after fetch, park every marked year's parts on the "
                         "Hub (the hosted lane), done.json last")
    ap.add_argument("--allow-missing-years", action="store_true",
                    help="build past an input that could not be read; the "
                         "store records every such unit in `degraded`")
    ap.add_argument("--check-chunk-rows", type=int,
                    default=f10b.CHECK_CHUNK_ROWS,
                    help="rows per block in the assertion pass")
    ap.add_argument("--attempts", type=int, default=3,
                    help="download attempts per file")
    ap.add_argument("--force", action="store_true",
                    help="redo a stage or a year whose .done marker exists")
    ap.add_argument("--allow-unconfirmed-licence", action="store_true",
                    help="publish (or push parts of) a PUBLIC store whose "
                         "licence says redistribution_confirmed = False — "
                         "refused without this flag")
    ap.add_argument("--distribution", default="",
                    choices=("", "public", "private"),
                    help="override the adapter's track for this run: "
                         "'private' sends a PUBLIC adapter's parts and store "
                         "to chfrank/earth-tensors-private and notes "
                         "licence_pending in store.json (a producer that has "
                         "not answered); 'public' on a private adapter is "
                         "refused")
    ap.add_argument("--check-credentials", action="store_true",
                    help="no store: one authenticated request to each of LP "
                         "DAAC, GES DISC and PO.DAAC with "
                         "EARTHDATA_USERNAME / EARTHDATA_PASSWORD, and a "
                         "report of what each archive answered (a hosted "
                         "runner only). Exits non-zero only on a definite "
                         "refusal")
    ap.add_argument("--check-credentials-out", default="",
                    help="with --check-credentials: also write the report "
                         "JSON here")
    return ap


def main(argv=None):
    a = build_parser().parse_args(argv)
    if a.check_credentials:
        from family1 import earthdata_check as edc
        return edc.main(out=a.check_credentials_out or None)
    if not a.store:
        sys.exit("--store is required (or pass --check-credentials)")
    cls = REGISTRY[a.store]
    if a.smoke:
        run_smoke(a.store, root=a.smoke_dir or None, keep=bool(a.smoke_dir),
                  probe=not a.no_probe)
        return 0
    if not a.work:
        a.work = default_work(cls)
    if a.stage.strip() == "probe":
        if not a.probe_month:
            sys.exit("--stage probe needs --probe-month YYYY-MM")
        if is_grid(cls):
            stage_probe_grid(a, cls, a.probe_month)
        else:
            stage_probe(a, cls, a.probe_month)
        return 0
    stages = f10b.parse_stages(a.stage, STAGES)
    ad = apply_distribution(cls(), a.distribution)
    lay = layout_for(ad)
    if needs_source(a, stages):
        credentials_preflight(ad)
    ctx = f10b.Ctx(a, adapter=ad, layout=lay)
    if is_grid(ad):
        prepare_grid_ctx(ctx)
    print(f"store     {ad.store} — {ad.title}")
    print(f"family    {lay.family} ({lay.family_version}) · "
          f"{ad.distribution} -> {lay.repo_id}:{lay.prefix(ad.store)}")
    print(f"axis      {ctx.d_lo} .. {ctx.d_hi}  bins {ctx.b_lo}..{ctx.b_hi} "
          f"· time_s {ctx.time_dtype} (schema {ctx.schema_version})")
    print(f"channels  C={ad.C}: {', '.join(ad.channel_names)}")
    if is_grid(ad):
        print(f"tier G    sharded, groups {sorted(ctx.grid_specs)}, "
              f"F={ad.frames_per_bin} x {ad.frame_seconds} s, {ad.dtype}, "
              f"years (by bin start) {ctx.years[0]}..{ctx.years[-1]}")
    print(f"work      {ctx.work}")
    f10b.run_stages(ctx, stages, stage_fn=stage_fns(ad), deps=DEPS)
    return 0


if __name__ == "__main__":
    sys.exit(main())

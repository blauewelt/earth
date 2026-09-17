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
"""
import argparse
import calendar
import datetime as dt
import json
import os
import platform as _platform
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import build_family10_stores as f10b                            # noqa: E402
import family10_store as f10                                    # noqa: E402
from build_family7 import (END, atomic_json, git_sha, mark,     # noqa: E402
                           marked, read_json, utcnow)
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


def make_ctx(a, adapter_cls):
    ad = adapter_cls()
    return f10b.Ctx(a, adapter=ad, layout=layout_for(ad))


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
    kw = f10b.parts_hub_kwargs(ctx)
    kw = {k: v for k, v in kw.items() if k in ("partials", "hub", "private")}
    pushed = []
    for y in ctx.years:
        if not marked(ctx.root, f"parts/{y}"):
            sys.exit(f"--push-parts: {ctx.adapter.store} {y} is not marked "
                     f"done locally — refusing to push a partial year")
        ph.push(ctx.adapter.store, y, ctx.work, **kw)
        pushed.append(y)
    print(f"  push-parts: {len(pushed)} year(s) on the Hub under "
          f"{ctx.layout.hf_partials}/{ctx.adapter.store}/")
    return pushed


def stage_assemble(ctx):
    meta = f10b.assemble(ctx)
    mark(ctx.root, "assemble")
    ctx.prog.item("store", 1, {"N": meta["N"], "bin_first": meta["bin_first"]})
    return meta


def stage_publish(ctx):
    return f10b.stage_publish(ctx)


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
        from huggingface_hub import hf_hub_download
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
            sys.exit(f"check {ad.store}: the Hub's store.json does not carry "
                     f"the local sha256 block — the published store is not "
                     f"the one checked here")
        man = {e["name"]: e["sha256"] for e in got["manifest.json"]
               .get("files", [])}
        bad = [k for k, v in (sm.get("sha256") or {}).items()
               if man.get(k) != v]
        if bad:
            sys.exit(f"check {ad.store}: the Hub manifest disagrees on {bad}")
        out["hub"] = {"repo": repo, "prefix": prefix,
                      "files": len(man)}
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
    ap.add_argument("--store", required=True, choices=sorted(REGISTRY),
                    help="the store, from ml/family1/adapters/")
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
    return ap


def main(argv=None):
    a = build_parser().parse_args(argv)
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
        stage_probe(a, cls, a.probe_month)
        return 0
    stages = f10b.parse_stages(a.stage, STAGES)
    ad = cls()
    lay = layout_for(ad)
    if needs_source(a, stages):
        credentials_preflight(ad)
    ctx = f10b.Ctx(a, adapter=ad, layout=lay)
    print(f"store     {ad.store} — {ad.title}")
    print(f"family    {lay.family} ({lay.family_version}) · "
          f"{ad.distribution} -> {lay.repo_id}:{lay.prefix(ad.store)}")
    print(f"axis      {ctx.d_lo} .. {ctx.d_hi}  bins {ctx.b_lo}..{ctx.b_hi} "
          f"· time_s {ctx.time_dtype} (schema {ctx.schema_version})")
    print(f"channels  C={ad.C}: {', '.join(ad.channel_names)}")
    print(f"work      {ctx.work}")
    f10b.run_stages(ctx, stages, stage_fn=STAGE_FN, deps=DEPS)
    return 0


if __name__ == "__main__":
    sys.exit(main())

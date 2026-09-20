#!/usr/bin/env python3
"""The REGISTRIES of families 1.gf, 1.0.tf and 0.9.tf — one file per family.

E-082 §4 ("the three registries"), modelled on `ml/build_family10_registry.py`
and written for the same reader: **a consumer dispatches on `tier` and needs
no other document.** Writes three files beside family 10's own registry:

    family1gf.json    family 1.gf   — global ocean and atmosphere, <= 10 km
    family1tf.json    family 1.0.tf — land and coast, at the same scale
    family09tf.json   family 0.9.tf — 1.0.tf with the raw satellite imagery
                                      replaced by an embedding; everything
                                      else INHERITED by reference

WHAT IS DIFFERENT FROM FAMILY 10'S REGISTRY, and why.

  **The adapters are the source of truth, not the Hub.** Family 10's five
  stores are all built, so its registry can be assembled out of five
  `store.json` files. Family 1 has forty-odd adapters and about a third of
  them are built; a registry that listed only those would be a list of what
  happened to finish, and the note's whole point is the LEDGER — every store
  the family is designed to hold, with its source, its licence, its
  granularity and its size estimate, so a reader can see what is coming as
  well as what is there. So the walk is over `family1.adapters.REGISTRY` and
  every adapter gets an entry; `built` says whether the Hub has it.

  **Three numbers per store, and they are kept apart.** The NOTE'S ESTIMATE
  (`note_estimate`, what the design row guessed), the PROBE'S MEASUREMENT
  (`ml/family1/probes/<store>_<YYYY-MM>.json`, one real month put through the
  real writer) and the BUILT STORE (`store.json` on the Hub: N, the record
  span, the checksums, the builder's commit). ml/CLAUDE.md §"measure, then
  estimate": the estimate is what the measurement REPLACES, and a registry
  that overwrote one with the other would lose the comparison that says
  whether the family's sizing is any good.

  **A private-track store is listed and its checksums are not.**
  `distribution: "private"` stores live in `chfrank/earth-tensors-private`,
  which answers an anonymous read with 401. Reading them would need a token,
  and a registry that carried a private store's file hashes would put them in
  a public file. So a private store is listed with its design row, its probe
  if it has one, `distribution: "private"` and no `files` block at all — and
  says so in `built_note` rather than being silently marked unbuilt.

  **0.9.tf inherits by REFERENCE.** Family 0.9.tf is family 1.0.tf with the
  raw imagery swapped for Google's AlphaEarth embedding; every other store is
  the same bytes under the same path. So `family09tf.json` carries
  `inherits: "family1tf"` — the sibling registry's name — plus a resolved
  `inherited_groups` list, and `groups` holds only what 0.9.tf builds itself.
  A reader resolves the inheritance by reading the named registry, which is
  the one arrangement in which the two cannot drift.

  **An access failure is not evidence of absence** (ml/CLAUDE.md §0.2). A 404
  from the Hub means the store is not published; a 401 or a 403 means the read
  was REFUSED, and this builder raises rather than writing `built: false`,
  which would be a statement about the archive made from a failure to reach
  it. Family 10's registry learned that the expensive way and the rule is
  copied here deliberately.

`--check` is the mode a workflow runs: it builds the three registries and
exits non-zero if any registered adapter is missing from all of them, if an
adapter appears in more than one, or if a file on disk disagrees with what a
fresh build produces.

Run:
  python3 ml/build_family1_registry.py --no-hub          # offline, writes 3
  python3 ml/build_family1_registry.py                   # reads the Hub
  python3 ml/build_family1_registry.py --check --no-hub  # a workflow's gate
  python3 ml/build_family1_registry.py --publish         # upload all three
"""
import argparse
import glob
import json
import os
import re
import sys
import urllib.error
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import family10_store as f10                                    # noqa: E402
import build_family1_stores as b1                               # noqa: E402
from build_family7 import (atomic_json, git_sha, hub_repo,      # noqa: E402
                           read_json, sha256, utcnow)
from family1 import sharded as sh                               # noqa: E402
from family1.adapters import FAMILIES, REGISTRY                 # noqa: E402

PUBLIC_REPO = b1.PUBLIC_REPO
PRIVATE_REPO = b1.PRIVATE_REPO
HUB_BASE = "https://huggingface.co/datasets/{repo}/resolve/main/{path}"
PROBE_DIR = os.path.join(HERE, "family1", "probes")

# WHERE THE FILES GO. Family 10's registry is written to
# `ml/cache/family10_2/family10.json` and published to
# `tensors/family10_2/family10.json`; the three family-1 registries are
# written BESIDE it — one directory holds all four, so the `siblings` line
# family 10.2 now carries resolves locally as well as on the Hub — and each
# is published under ITS OWN family root.
REGISTRY_DIR = os.path.join(HERE, "cache", f10.CACHE_DIRNAME)

#: family code -> (registry file name, the note that designs it)
REGISTRY_NAME = {"1gf": "family1gf.json",
                 "1tf": "family1tf.json",
                 "09tf": "family09tf.json"}
NOTE = {"1gf": "ml/paper/notes/family1gf.tex",
        "1tf": "ml/paper/notes/family1tf.tex",
        "09tf": "ml/paper/notes/family09tf.tex"}
#: 0.9.tf is 1.0.tf with the imagery replaced; everything else is the same
#: bytes under the same path, so it is inherited by reference.
INHERITS = {"09tf": "1tf"}

PLAN = "ml/plans/E082_family1_builds.md"
DESIGN = "ml/plans/E078_multi_granularity.md"
WAVE6 = "ml/plans/E082_biosphere_wave.md"
CONTRACT = "ml/family1/ADAPTER_CONTRACT.md"

TIERS = {
    "P": "points, profiles and tracks — columns sorted by pentad bin with CSR "
         "offsets; a read is a k-nearest search",
    "G": "gridded dense, finer than 0.25° — compressed TILES per (group, "
         "bin); a read is two range reads per tile "
         "(ml/family1/sharded.py)",
    "T": "a scene or tile CATALOGUE — tier-P rows of one row per scene, plus "
         "`assets.parquet` mapping each row to the producer's own URLs. No "
         "pixels are copied",
}

# log2_dt is log2(support_days / 5), so the table is the same one family 10's
# registry uses, extended for the finer and coarser supports family 1 holds.
CADENCE = {
    -7.91: "half-hourly", -7.9: "half-hourly",
    -6.91: "hourly", -6.9: "hourly",
    -4.32: "6-hourly", -4.3: "6-hourly",
    -2.32: "daily", -2.3: "daily",
    -1.0: "2.5-day", 0.0: "pentad",
    0.68: "8-day", 2.58: "monthly", 2.6: "monthly", 2.61: "monthly",
    6.19: "annual", 6.18: "annual", 6.2: "annual",
}


# ==================================================================== Hub ===
def http_json(url, timeout=60):
    req = urllib.request.Request(
        url, headers={"User-Agent": "earth-science-pipeline/1.0 "
                                    "(research; github blauewelt/earth)"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def hub_json(repo, path, timeout=60):
    """A JSON file from the Hub, or None if it is NOT THERE.

    Anonymous and read-only. A 404 is an absence and answers None; a 401 or a
    403 is a REFUSAL and raises, because a registry that turned "the Hub would
    not talk to me" into "that store is not built" would be making a statement
    about the archive out of a failure to reach it (ml/CLAUDE.md §0.2, and the
    same guard family 10's registry carries).
    """
    try:
        return http_json(HUB_BASE.format(repo=repo, path=path), timeout)
    except urllib.error.HTTPError as e:
        if e.code == 404:
            return None
        if e.code in (401, 403):
            raise IOError(
                f"{repo}:{path} answered HTTP {e.code} — the Hub refused the "
                f"read rather than saying the file is absent. This registry "
                f"would otherwise list that store as NOT BUILT, which is a "
                f"statement about the archive made from a failure to reach "
                f"it. Check the token's read access to {repo} and re-run.") \
                from e
        raise


# ================================================================= probes ===
def probe_path(store, probe_dir=PROBE_DIR):
    """The newest probe report for `store`, or None.

    The naming convention is `<store>_<YYYY>-<MM>.json`, which is what
    `build_family1_stores.stage_probe` writes; the newest month wins where a
    store has been probed twice. `<store>` is matched EXACTLY, so `tide` never
    picks up `tide_private`'s report.
    """
    pat = re.compile(re.escape(str(store)) + r"_(\d{4})-(\d{2})\.json$")
    best, best_key = None, None
    for p in sorted(glob.glob(os.path.join(probe_dir, "*.json"))):
        m = pat.match(os.path.basename(p))
        if not m:
            continue
        key = (int(m.group(1)), int(m.group(2)))
        if best_key is None or key > best_key:
            best, best_key = p, key
    return best


def probe_summary(path):
    """The numbers from a probe report that belong in a registry entry.

    Not the whole report — it runs to tens of kilobytes per store and carries
    per-day row counts and per-channel NaN fractions that a registry reader
    does not want. The FILE is named so anybody who does can read it.
    """
    d = read_json(path, None)
    if d is None:
        return None
    out = {"file": os.path.relpath(path, os.path.dirname(HERE)),
           "month": d.get("month"),
           "runner": d.get("runner"),
           "builder_git_sha": d.get("builder_git_sha"),
           "at": d.get("at"),
           "bytes_fetched": d.get("bytes_fetched"),
           "wall_seconds": d.get("wall_seconds"),
           "estimate_store_bytes": d.get("estimate_store_bytes"),
           "note_formula_bytes": d.get("note_formula_bytes")}
    if d.get("tier") == "G" or "groups" in d:
        out.update({"tier": "G",
                    "frames_fetched": d.get("frames_fetched"),
                    "bytes_fetched_per_frame":
                        d.get("bytes_fetched_per_frame"),
                    "groups": sorted(d.get("groups") or {})})
    else:
        out.update({"tier": d.get("tier", "P"),
                    "rows": d.get("rows"),
                    "counts_scope": d.get("counts_scope"),
                    "distinct_platforms": d.get("distinct_platforms"),
                    "bytes_per_row": d.get("bytes_per_row"),
                    "stored_bytes_per_row": d.get("stored_bytes_per_row"),
                    "estimate_rows_per_year":
                        d.get("estimate_rows_per_year")})
    return {k: v for k, v in out.items() if v is not None}


# ================================================================= entries ==
def cadence_of(ad):
    dtl = round(float(getattr(ad, "log2_dt", -4.0)), 2)
    return CADENCE.get(dtl, "irregular (per observation)")


def tier_of(ad):
    return getattr(ad, "tier", "P")


def layout_of(ad):
    if tier_of(ad) == "G":
        return ("sharded tiles: <group>/<yyyy>/bin_<NNNN>.zst plus its "
                ".idx.npy; a read is two range reads per tile "
                "(ml/family1/sharded.py)")
    base = ("columns sorted by (bin, time_s) with CSR bin offsets; a read is "
            "a k-nearest search")
    if is_catalogue(ad):
        return base + "; `assets.parquet` beside the columns maps each row's "\
                      "platform hash to the producer's own asset URLs"
    return base


def is_catalogue(ad):
    """A tier-T catalogue is a tier-P store whose rows are scene metadata."""
    from family1.adapters import _stac as st
    return isinstance(ad, st.CatalogueAdapter)


def declared_tier(ad):
    """`T` for a catalogue, else the adapter's own tier.

    A catalogue store declares `tier = "P"` because its LAYOUT is tier P — the
    same columns, the same reader — while the note calls it tier T. Both are
    true and the registry says both, because a consumer dispatching on `tier`
    must not be handed a catalogue where it expects observations.
    """
    return "T" if is_catalogue(ad) else tier_of(ad)


def entry(store, ad, repo, use_hub=True, work=None, probe_dir=PROBE_DIR):
    """One store's registry entry: the design row, the probe, the store."""
    lay = b1.layout_for(ad)
    prefix = f"{lay.hf_root}/{store}"
    private = bool(lay.private)
    ch = getattr(ad, "channels", ()) or ()
    out = {
        "name": store,
        "tier": declared_tier(ad),
        "tier_layout": tier_of(ad),
        "catalogue": is_catalogue(ad),
        "title": getattr(ad, "title", ""),
        "family_code": ad.family,
        "family_version": lay.family_version,
        "path": prefix,
        "repo": PRIVATE_REPO if private else PUBLIC_REPO,
        "distribution": "private" if private else "public",
        "layout": layout_of(ad),
        "cadence": cadence_of(ad),
        "channels": [{"name": c[0], "unit": c[1], "min": c[2], "max": c[3]}
                     for c in ch],
        "C": len(ch),
        "footprint": {"log2_fp": float(getattr(ad, "log2_fp", 0.0)),
                      "log2_dt": float(getattr(ad, "log2_dt", 0.0)),
                      "note": "log2 of the measurement's support against the "
                              "family's 27.83 km / 5-day unit (E-078 §2)"},
        "time_dtype": getattr(ad, "time_dtype", "int32"),
        "schema_version": (3 if getattr(ad, "time_dtype", "int32") == "int64"
                           else 2),
        "first_year": getattr(ad, "first_year", None),
        "per_year": bool(getattr(ad, "per_year", True)),
        "platform_meta": bool(getattr(ad, "platform_meta", False)),
        "credentials": list(getattr(ad, "credentials", ()) or ()),
        "licence": dict(getattr(ad, "licence", {}) or {}),
        "qc_policy": getattr(ad, "qc_policy", ""),
        "sources": list(getattr(ad, "sources", ()) or ()),
        "verified": getattr(ad, "verified", ""),
        "notes": getattr(ad, "notes", ""),
        "note_estimate": getattr(ad, "note_estimate", None),
        "adapter": f"ml/family1/adapters/{store}.py",
        "builder": "ml/build_family1_stores.py",
    }
    if tier_of(ad) == "G":
        out["frames_per_bin"] = int(getattr(ad, "frames_per_bin", 1))
        out["frame_seconds"] = int(getattr(ad, "frame_seconds",
                                           sh.BIN_SECONDS))
        out["dtype"] = getattr(ad, "dtype", "float16")
        out["reader"] = ("ml/family1/sharded.py :: ShardedGroup.read_frame — "
                         "local path or the Hub's resolve/main URL")
    else:
        out["reader"] = ("ml/family10_store.py :: Store.open(<dir or "
                         "'<owner>/<repo>:<prefix>'>).knearest(...)")

    pp = probe_path(store, probe_dir)
    out["probe"] = probe_summary(pp) if pp else None

    meta = None
    if work:
        for cand in (os.path.join(work, store, store, "store.json"),
                     os.path.join(work, store, "store.json")):
            if os.path.exists(cand):
                meta = read_json(cand, None)
                break
    if meta is None and use_hub and not private:
        meta = hub_json(PUBLIC_REPO, f"{prefix}/store.json")
    if meta is None:
        out["built"] = False
        out["built_note"] = (
            "no store.json under the private repository is read here: it "
            "answers an anonymous GET with 401, and a public registry must "
            "not carry a private store's file hashes. The design row and the "
            "probe above are what this registry states about it."
            if private else
            f"not published: {PUBLIC_REPO} has no {prefix}/store.json. The "
            f"design row and the probe above are the store's whole record "
            f"until a build lands.")
        return out

    out["built"] = True
    out["N"] = meta.get("N")
    out["bin_first"] = meta.get("bin_first")
    out["bin_last"] = meta.get("bin_last")
    out["n_bins"] = meta.get("n_bins")
    out["date_range"] = meta.get("date_range")
    out["record_span"] = meta.get("date_range")
    out["built_at"] = meta.get("built_at")
    out["builder_git_sha"] = meta.get("builder_git_sha")
    out["store_schema_version"] = meta.get("schema_version")
    out["counts"] = meta.get("counts")
    for k in ("groups", "frames_missing_by_reason", "per_year", "assets",
              "platforms", "degraded", "tier_t_catalogue"):
        if meta.get(k) is not None:
            out[f"store_{k}"] = meta[k]
    out["files"] = [{"name": n, "sha256": h}
                    for n, h in sorted((meta.get("sha256") or {}).items())]
    return out


# =============================================================== registries ==
def stores_of(code):
    """Every adapter that declares family `code`, in name order."""
    return sorted(s for s, cls in REGISTRY.items()
                  if getattr(cls, "family", None) == code)


def build_one(code, repo=PUBLIC_REPO, use_hub=True, work=None,
              probe_dir=PROBE_DIR):
    name, version, slug = FAMILIES[code]
    groups = [entry(s, REGISTRY[s](), repo, use_hub=use_hub, work=work,
                    probe_dir=probe_dir)
              for s in stores_of(code)]
    built = [g["name"] for g in groups if g["built"]]
    reg = {
        "family": name,
        "family_code": code,
        "family_version": version,
        "hf_root": f"tensors/{slug}",
        "registry": f"tensors/{slug}/{REGISTRY_NAME[code]}",
        "schema_version_note": (
            "every store states its OWN `schema_version`: 2 where `time_s` is "
            "int32 seconds since 1982-01-01T00:00:00Z, 3 where a record "
            "before 1914 forces int64. Tier G stores carry frames rather than "
            "rows and have no time column at all — their bin axis is the "
            "shard index."),
        "description": (
            "Family 1 is the FINE observation families: everything at 10 km "
            "or finer and 5 days or finer that the model reads beside the "
            "0.25° global tensor. Nothing is resampled to a common grid at "
            "storage time; every store keeps its own resolution and cadence "
            "and carries the measurement's footprint, so a consumer can tell "
            "a 500 m monthly burned-area map from a half-hourly flux tower at "
            "the same place. Dispatch on `tier`."),
        "plan": PLAN, "design": DESIGN, "wave6": WAVE6, "contract": CONTRACT,
        "note": NOTE[code],
        "epoch": str(sh.EPOCH), "pentad_days": f10.PENTAD_DAYS,
        "bin_rule": ("bin = floor(time_s / 432000 s) for a tier-P store; for "
                     "a tier-G store the shard index's `bin` column is the "
                     "same axis, and frame f of bin b covers "
                     "[b * 432000 + f * frame_seconds, + frame_seconds)"),
        "tiers": dict(TIERS),
        "numbers_note": (
            "three numbers, kept apart on purpose: `note_estimate` is what "
            "the design row GUESSED, `probe` is one real month put through "
            "the real writer, and `N` / `files` / `record_span` come from the "
            "published store. The estimate is what the measurement replaces "
            "(ADAPTER_CONTRACT rule 6); overwriting one with the other would "
            "lose the comparison that says whether the family's sizing is any "
            "good."),
        "repo": PUBLIC_REPO,
        "private_repo": PRIVATE_REPO,
        "n_groups": len(groups),
        "n_built": len(built),
        "built": built,
        "not_built": [g["name"] for g in groups if not g["built"]],
        "groups": groups,
        "builder": "ml/build_family1_registry.py",
        "builder_git_sha": git_sha(),
        "generated_utc": utcnow(),
    }
    if code in INHERITS:
        parent = INHERITS[code]
        pname, pversion, pslug = FAMILIES[parent]
        reg["inherits"] = REGISTRY_NAME[parent].replace(".json", "")
        reg["inherits_block"] = {
            "family_code": parent,
            "family_version": pversion,
            "registry": f"tensors/{pslug}/{REGISTRY_NAME[parent]}",
            "root": f"tensors/{pslug}",
            "groups": stores_of(parent),
            "note": (
                f"family {version} is family {pversion} with the raw "
                f"satellite imagery replaced by Google's AlphaEarth "
                f"embedding. Every other store is the SAME BYTES under the "
                f"same path, so it is listed BY REFERENCE: read the named "
                f"registry for them. `groups` below holds only what "
                f"{version} builds itself — a copy of the parent's rows is "
                f"the thing that would go stale."),
        }
    return reg


def build_all(repo=PUBLIC_REPO, use_hub=True, work=None,
              probe_dir=PROBE_DIR):
    return {code: build_one(code, repo=repo, use_hub=use_hub, work=work,
                            probe_dir=probe_dir)
            for code in sorted(FAMILIES)}


def siblings_block():
    """The `siblings` line family 10.2's registry gains.

    Family 10 and family 1 are different families over the same planet, and
    nothing in either registry said the other existed. This is the line that
    does — additive, one key, resolvable by a reader with no other document.
    """
    return {
        "note": ("the FINE observation families (E-082): everything at 10 km "
                 "or finer and 5 days or finer, read beside this family's "
                 "0.25° tensor. Different families, one planet; each has its "
                 "own registry in the same shape as this one."),
        "registries": [
            {"family": FAMILIES[c][0], "family_version": FAMILIES[c][1],
             "registry": f"tensors/{FAMILIES[c][2]}/{REGISTRY_NAME[c]}",
             "builder": "ml/build_family1_registry.py"}
            for c in ("1gf", "1tf", "09tf")],
    }


# ==================================================================== check ==
def check(regs):
    """Every registered adapter appears in exactly one registry. -> problems."""
    bad = []
    seen = {}
    for code, reg in sorted(regs.items()):
        if reg["family_code"] != code:
            bad.append(f"{code}: the registry declares family_code "
                       f"{reg['family_code']!r}")
        for g in reg["groups"]:
            seen.setdefault(g["name"], []).append(code)
            if g["family_code"] != code:
                bad.append(f"{code}: {g['name']} declares family "
                           f"{g['family_code']!r}")
            if g["tier"] not in TIERS:
                bad.append(f"{code}: {g['name']} has tier {g['tier']!r}")
            if not g["channels"] and g["tier"] != "T":
                bad.append(f"{code}: {g['name']} declares no channel")
            lic = g.get("licence") or {}
            if lic.get("redistribution") == "no" \
                    and g["distribution"] != "private":
                bad.append(f"{code}: {g['name']} may not be redistributed and "
                           f"is on the public track")
    for store in sorted(REGISTRY):
        where = seen.get(store) or []
        if not where:
            bad.append(f"{store}: registered as an adapter and in NO registry "
                       f"— its `family` is "
                       f"{getattr(REGISTRY[store], 'family', None)!r}")
        elif len(where) > 1:
            bad.append(f"{store}: in {len(where)} registries ({where})")
    for store in sorted(seen):
        if store not in REGISTRY:
            bad.append(f"{store}: in a registry and not an adapter")
    return bad


# ================================================================== publish ==
def publish_one(path, code, repo=None):
    """Upload one registry and DOWNLOAD IT BACK to compare sha256."""
    from huggingface_hub import hf_hub_download
    import shutil
    api, resolved, tok = hub_repo()
    repo = repo or resolved
    _n, _v, slug = FAMILIES[code]
    dest = f"tensors/{slug}/{REGISTRY_NAME[code]}"
    src = sha256(path)
    api.create_repo(repo, repo_type="dataset", exist_ok=True, private=False)
    api.upload_file(path_or_fileobj=path, path_in_repo=dest, repo_id=repo,
                    repo_type="dataset",
                    commit_message=f"family {FAMILIES[code][1]}: the registry")
    scratch = path + ".verify"
    shutil.rmtree(scratch, ignore_errors=True)
    back = hf_hub_download(repo, dest, repo_type="dataset", token=tok,
                           local_dir=scratch)
    got = sha256(back)
    shutil.rmtree(scratch, ignore_errors=True)
    if got != src:
        sys.exit(f"RESTORE MISMATCH {dest}: uploaded {src}, downloaded {got} "
                 f"— the publish is not trustworthy")
    print(f"  publish: verified by restore -> https://huggingface.co/"
          f"datasets/{repo}/blob/main/{dest}")
    return got


# ===================================================================== main ==
def main():
    ap = argparse.ArgumentParser(
        description="Write family1gf.json, family1tf.json and "
                    "family09tf.json — the three family-1 registries "
                    "(E-082 §4).")
    ap.add_argument("--out-dir", default=REGISTRY_DIR,
                    help="where to write them (default: beside family 10's "
                         "own registry)")
    ap.add_argument("--work", default="",
                    help="a local build directory; a store.json found here "
                         "is preferred over the Hub copy")
    ap.add_argument("--repo", default=PUBLIC_REPO)
    ap.add_argument("--probes", default=PROBE_DIR,
                    help="the directory holding <store>_<YYYY-MM>.json")
    ap.add_argument("--no-hub", action="store_true",
                    help="never reach the Hub; every store is then listed "
                         "from its adapter and its probe alone")
    ap.add_argument("--check", action="store_true",
                    help="build and verify that every registered adapter is "
                         "in exactly one registry; exit non-zero otherwise")
    ap.add_argument("--publish", action="store_true",
                    help="upload all three and verify each restore")
    a = ap.parse_args()

    regs = build_all(repo=a.repo, use_hub=not a.no_hub,
                     work=(os.path.abspath(a.work) if a.work else None),
                     probe_dir=a.probes)
    os.makedirs(a.out_dir, exist_ok=True)
    paths = {}
    for code in sorted(FAMILIES):
        p = os.path.join(a.out_dir, REGISTRY_NAME[code])
        atomic_json(p, regs[code])
        paths[code] = p
        reg = regs[code]
        tiers = {}
        for g in reg["groups"]:
            tiers[g["tier"]] = tiers.get(g["tier"], 0) + 1
        inh = (f", inherits {reg['inherits']}" if reg.get("inherits") else "")
        print(f"registry  {p}")
        print(f"          family {reg['family_version']}: "
              f"{reg['n_groups']} store(s), {reg['n_built']} built"
              f"{inh}")
        print("          " + ", ".join(f"{n} tier-{t}"
                                       for t, n in sorted(tiers.items())))
        for g in reg["groups"]:
            print(f"          {g['tier']} {g['name']:16s} C={g['C']:<3d} "
                  f"{g['cadence']:<22s} "
                  f"{'built' if g['built'] else '-    '} "
                  f"{'probed' if g.get('probe') else '-'}")

    problems = check(regs)
    if problems:
        print("::warning::the registries do not cover the adapters:")
        for m in problems:
            print(f"  {m}")
    if a.check:
        if problems:
            print(f"CHECK FAILED: {len(problems)} problem(s)")
            return 1
        print(f"CHECK OK: {len(REGISTRY)} adapter(s), each in exactly one of "
              f"{', '.join(REGISTRY_NAME[c] for c in sorted(FAMILIES))}")
    if a.publish:
        for code in sorted(FAMILIES):
            publish_one(paths[code], code, repo=a.repo)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

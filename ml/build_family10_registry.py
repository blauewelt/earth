#!/usr/bin/env python3
"""Family 10's REGISTRY — one file that lists every group of the family.

E-079 §2 ("The registry") and E-078 §3 ("One registry"), made runnable.
Writes `tensors/family10/family10.json`: every group of family 10 with its
`tier`, layout, cadence, channels with units, footprint constants, `bin_first`,
files with sha256, sources and builder commit. **A consumer dispatches on
`tier` and needs no other document.**

WHAT A GROUP IS, and why the two tiers look different on purpose.

  tier G — gridded dense. Family 7.1's four channel groups (`g025` the 0.25°
  ocean, `g100` the 1° atmosphere and land, `rg100` the monthly 1° Argo depth
  column, `oc025` the ocean colour E-077 adds). Each is one bin-major `.npy`
  at its NATIVE grid; a read is a byte-range slab of one bin. They are listed
  BY REFERENCE — the registry carries the manifest's URL and the per-file
  sha256 the manifest publishes, never a copy of the tensor's own metadata,
  because a copy is the thing that goes stale.

  tier P — points, profiles and tracks. Family 8's Argo store and the four
  stores of E-079. Each is a column store sorted by pentad bin with CSR
  offsets; a read is a k-nearest search. Their entries come from their own
  `store.json`, which already carries the schema, the footprint, the QC policy
  and the sha256 of every file.

WHICH FAMILY-7 MANIFEST. E-079 §2 says family 7.1. At the time of writing
E-077's `f7l1` build has been WRITTEN but not DISPATCHED, so
`tensors/family7_global025_pentad_l1/manifest.json` is a 404 on the Hub
(measured 2026-09-13) and `..._l0/manifest.json` is a 200. The builder asks
for l1 FIRST, falls back to l0, and RECORDS WHICH ONE IT USED in
`tier_g.manifest` and `tier_g.fallback_note` — a registry that silently
described the wrong tensor would be worse than one that refused.

Run:
  python3 ml/build_family10_registry.py --out /tmp/family10.json
  python3 ml/build_family10_registry.py --work ml/cache/family10 --publish
  python3 ml/build_family10_registry.py --stores gdp,socat --no-hub   # offline
"""
import argparse
import datetime as dt
import json
import os
import sys
import urllib.error
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import family10_store as f10                                    # noqa: E402
from build_family7 import (atomic_json, git_sha, hub_repo,       # noqa: E402
                           read_json, sha256, utcnow)

FAMILY = "family10"
HF_ROOT = "tensors/family10"
REGISTRY_NAME = "family10.json"
HUB_REPO_DEFAULT = "chfrank/earth-tensors"
HUB_BASE = "https://huggingface.co/datasets/{repo}/resolve/main/{path}"

# The tier-G tensor, newest recipe first. E-077's f7l1 is the one E-079 §2
# names; f7l0 is what is on the Hub until that build is dispatched.
F7_STEMS = ("family7_global025_pentad_l1", "family7_global025_pentad_l0")
F7_INDEX = os.path.join(os.path.dirname(HERE), "data", "family7_index.json")

# The Argo store of family 8, which becomes family 10's first tier-P group
# UNCHANGED (E-078 §7: the design adds a registry and a tier and takes nothing
# away).
F8_PREFIX = "tensors/family8_argo_l0"
F8_NAME = "argo"

# E-079's four new stores, in the plan's own order.
F10_STORES = ("gdp", "gtmba", "socat", "slatrack")

# E-076 §2.6 / E-078 §2, per tier-G group. These are the SUPPORT of the
# measurement, not the grid it is stored on — which is the whole point of the
# field, and why `g100` reads +2.9 rather than +2: NCEP's native T62 Gaussian
# grid is about 1.9 degrees, and 1 degree is only how the bytes are arranged.
TIER_G_FOOTPRINT = {
    "g025": {"log2_fp": 0.0, "log2_dt": 0.0,
             "note": "a 0.25° cell, a five-day mean — the unit both fields are "
                     "defined against, so both are 0 by construction"},
    "g100": {"log2_fp": 2.9, "log2_dt": 0.0,
             "note": "NCEP at its true T62 Gaussian support (~1.9°), not at "
                     "the 1° grid it is stored on: log2(1.9/0.25) = 2.93"},
    "rg100": {"log2_fp": 2.0, "log2_dt": 2.6,
              "note": "the Roemmich–Gilson mapped field: a 1° cell, a MONTHLY "
                      "mean — log2(1/0.25) = 2, log2(30/5) = 2.58"},
    "oc025": {"log2_fp": 0.0, "log2_dt": 0.0,
              "note": "OC-CCI's 4 km pixels box-averaged 6×6 onto the 0.25° "
                      "cell (E-077), so the STORED value's support is the "
                      "cell; `chl_cov` says how much of it was clear"},
}
TIER_G_CADENCE = {"g025": "pentad", "g100": "pentad", "rg100": "monthly",
                  "oc025": "pentad"}

TOKEN_SCHEMA = {
    "reference": "E-078 §2 (ml/plans/E078_multi_granularity.md)",
    "fields": ["value[C]", "mask[C]", "dx_km", "dy_km", "dt_days",
               "log2_fp", "log2_dt", "n_R", "source", "channel"],
    "log2_fp": "log2(footprint_km / 27.83) — spatial support in 0.25°-cell "
               "units; 27.83 km is 0.25° of latitude at 111.32 km/degree",
    "log2_dt": "log2(support_days / 5) — temporal support in pentad units",
    "clamp": list(f10.LOG2_FP_RANGE),
    "note": "the encoder never learns which file a value was read from, only "
            "what it is, where it is, and how much it averaged",
}


def http_json(url, timeout=60):
    req = urllib.request.Request(
        url, headers={"User-Agent": "earth-science-pipeline/1.0 "
                                    "(research; github blauewelt/earth)"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def hub_json(repo, path, timeout=60):
    """A JSON file from the Hub, or None if it is not there.

    Anonymous and read-only: `resolve/main/...` answers a 302 to a CDN and
    urllib follows it, which is the same path the app's own reads take.
    """
    try:
        return http_json(HUB_BASE.format(repo=repo, path=path), timeout)
    except urllib.error.HTTPError as e:
        if e.code == 404:
            return None                    # the store is not published yet
        if e.code in (401, 403):
            # NOT the same answer as a 404, and returning None for both meant
            # a private repo, a revoked token or a rate limit came out of this
            # function as "that store does not exist" — and the registry then
            # PUBLISHED itself with the store in `groups_missing`, which is a
            # promise about the Hub made by something that could not read the
            # Hub. §0.2: an access failure is not evidence of absence.
            raise IOError(
                f"{repo}:{path} answered HTTP {e.code} — the Hub refused the "
                f"read rather than saying the file is absent. This registry "
                f"would otherwise list that store as MISSING, which is a "
                f"statement about the archive made from a failure to reach "
                f"it. Check HF_TOKEN's read access to {repo} (the registry "
                f"reads anonymously, so a repo that has gone private needs "
                f"one) and re-run.") from e
        raise


# ================================================================== tier G ===
def tier_g_groups(repo, use_hub=True, index_path=F7_INDEX):
    """Family 7.1's groups, BY REFERENCE to its manifest.

    Two sources, deliberately: the MANIFEST is authoritative for which files
    exist and what they hash to (it is written by the publish that verified
    them), and `data/family7_index.json` is authoritative for the channel names,
    units, labels and grid geometry (it is written from the tensor's own npz
    headers by `ml/publish_family7_index.py`). Neither is retyped here.
    """
    stem = None
    man = None
    if use_hub:
        for s in F7_STEMS:
            man = hub_json(repo, f"tensors/{s}/manifest.json")
            if man is not None:
                stem = s
                break
    note = None
    if man is None:
        note = ("no family-7 manifest was readable on the Hub; the tier-G "
                "entries below carry channels and geometry from "
                "data/family7_index.json and NO file hashes")
        stem = F7_STEMS[-1]
    elif stem == F7_STEMS[-1]:
        note = (f"E-079 §2 names family 7.1 ({F7_STEMS[0]}), whose build "
                f"(E-077, recipe f7l1) has not been dispatched — "
                f"tensors/{F7_STEMS[0]}/manifest.json is a 404 on the Hub. "
                f"This registry therefore describes {F7_STEMS[-1]} (recipe "
                f"f7l0, three groups, no ocean colour). Re-run this builder "
                f"after the f7l1 build and it will pick up l1 and the fourth "
                f"group `oc025` with no change here.")
    idx = read_json(index_path, {}) or {}
    hashes = {f["name"]: f for f in (man or {}).get("files", [])}
    out = []
    for name, g in (idx.get("groups") or {}).items():
        fname = g.get("file", "")
        if stem and fname:
            fname = fname.replace(F7_STEMS[-1], stem)
        fm = hashes.get(fname, {})
        out.append({
            "name": name, "tier": "G",
            "layout": "one bin-major .npy at the group's NATIVE grid; a read "
                      "is one HTTP range of (bin - bin_first) * ny * nx * C * "
                      "itemsize bytes",
            "cadence": TIER_G_CADENCE.get(name, "pentad"),
            "channels": [{"name": c,
                          "unit": (g.get("units") or {}).get(c, ""),
                          "label": (g.get("labels") or {}).get(c, "")}
                         for c in g.get("chans", [])],
            "C": len(g.get("chans", [])),
            "footprint": TIER_G_FOOTPRINT.get(
                name, {"log2_fp": 0.0, "log2_dt": 0.0,
                       "note": "not in E-076 §2.6's table — assumed 0.25°, "
                               "pentad; CHECK before it is used"}),
            "bin_first": int(g.get("bin_first", 0) or 0),
            "n_bins": g.get("n_bins"),
            "shape": g.get("shape"),
            "dtype": g.get("dtype"),
            "header_len": g.get("header_len"),
            "slab_bytes": g.get("slab_bytes"),
            "grid": g.get("grid"),
            "live_bins_only": bool(g.get("live_only")),
            "normalisation": "z-scored at build time; `norm` in the group's "
                             "own index entry carries (mean, sd) per channel",
            "repo": repo, "prefix": f"tensors/{stem}",
            "files": ([{"name": fname, "bytes": fm.get("bytes"),
                        "sha256": fm.get("sha256")}] if fname else []),
        })
    out.sort(key=lambda e: e["name"])
    return out, {"stem": stem,
                 "manifest": (f"{HUB_BASE.format(repo=repo, path=f'tensors/{stem}/manifest.json')}"
                              if man is not None else None),
                 "manifest_present": man is not None,
                 "recipe": (man or {}).get("recipe"),
                 "builder_git_sha": (man or {}).get("builder_git_sha"),
                 "built_at": (man or {}).get("built_at"),
                 "index": "data/family7_index.json",
                 "sources": (man or {}).get("sources") or idx.get("sources"),
                 "fallback_note": note}


# ================================================================== tier P ===
def _store_entry(name, meta, repo, prefix, local=None):
    """One tier-P group, out of a store.json that already says everything."""
    ch = meta.get("channels")
    if ch and isinstance(ch[0], dict):
        channels = [{"name": c.get("name"), "unit": c.get("unit", "")}
                    for c in ch]
    else:
        # A family-8 store: its two value blocks ARE the channels, named the
        # way `family10_store.Store` names them when it reads that layout, so
        # the registry and the reader cannot disagree.
        lv = meta.get("levels") or []
        channels = [{"name": f"temp_{int(round(p))}", "unit": "degC"}
                    for p in lv] + \
                   [{"name": f"psal_{int(round(p))}", "unit": "PSU"}
                    for p in lv]
    files = [{"name": n, "sha256": h,
              "bytes": (os.path.getsize(os.path.join(local, n))
                        if local and os.path.exists(os.path.join(local, n))
                        else None)}
             for n, h in sorted((meta.get("sha256") or {}).items())]
    return {
        "name": name, "tier": "P",
        "layout": "columns sorted by (bin, time_days) with CSR bin offsets; a "
                  "read is a k-nearest search, one-sided in time, bounded by "
                  "(R_max_km, T_max_days), miss tokens to a fixed k",
        "reader": "ml/family10_store.py :: Store.open(<dir or "
                  "'<owner>/<repo>:<prefix>'>).knearest(...)",
        "cadence": meta.get("cadence") or _cadence_of(meta),
        "channels": channels,
        "C": meta.get("C", len(channels)),
        "N": meta.get("N"),
        "footprint": meta.get("footprint") or {
            "log2_fp": f10.DEFAULT_LOG2_FP, "log2_dt": f10.DEFAULT_LOG2_DT,
            "note": "the store carries no footprint block — E-076 §2.6's "
                    "point-profile constants are assumed"},
        "per_row_footprint": "fp.npy" in (meta.get("sha256") or {}),
        "bin_first": int(meta.get("bin_first", 0) or 0),
        "bin_last": meta.get("bin_last"),
        "n_bins": meta.get("n_bins"),
        "n_live_bins": meta.get("n_live_bins"),
        "date_range": meta.get("date_range"),
        "normalisation": meta.get("normalisation"),
        "qc_policy": meta.get("qc_policy"),
        "qc_keep_max": meta.get("qc_keep_max"),
        "platform_id": meta.get("platform_id"),
        "per_year": meta.get("per_year"),
        "sources": meta.get("sources"),
        "verified": meta.get("verified"),
        "builder": meta.get("builder"),
        "builder_git_sha": meta.get("builder_git_sha"),
        "built_at": meta.get("built_at"),
        "repo": repo, "prefix": prefix,
        "files": files,
    }


def _cadence_of(meta):
    """The store's own sampling interval, named from its footprint.

    log2_dt is log2(support_days / 5), so -4.32 is six-hourly, -2.32 daily, 0
    pentad. A store whose footprint matches none of those gets "irregular (per
    observation)", which is the truth for a ship track and for an Argo float —
    and is a cadence in its own right, not a fallback for a missing one.
    """
    dtl = round(float((meta.get("footprint") or {}).get("log2_dt", -4.0)), 2)
    return {-4.3: "6-hourly", -4.32: "6-hourly",
            -2.3: "daily", -2.32: "daily",
            -1.0: "2.5-day", 0.0: "pentad", 2.58: "monthly",
            2.6: "monthly"}.get(dtl, "irregular (per observation)")


def tier_p_groups(repo, work=None, stores=F10_STORES, use_hub=True,
                  include_argo=True):
    out, missing = [], []
    if include_argo:
        meta = None
        local = None
        if work:
            p = os.path.join(work, "family8_argo_l0", "store.json")
            if os.path.exists(p):
                meta, local = read_json(p, None), os.path.dirname(p)
        if meta is None and use_hub:
            meta = hub_json(repo, f"{F8_PREFIX}/store.json")
        if meta is None:
            missing.append(F8_NAME)
        else:
            out.append(_store_entry(F8_NAME, meta, repo, F8_PREFIX, local))
    for s in stores:
        meta, local = None, None
        if work:
            for cand in (os.path.join(work, s, s, "store.json"),
                         os.path.join(work, s, "store.json")):
                if os.path.exists(cand):
                    meta, local = read_json(cand, None), os.path.dirname(cand)
                    break
        if meta is None and use_hub:
            meta = hub_json(repo, f"{HF_ROOT}/{s}/store.json")
        if meta is None:
            missing.append(s)
            continue
        out.append(_store_entry(s, meta, repo, f"{HF_ROOT}/{s}", local))
    return out, missing


# ================================================================== registry ==
def build_registry(repo=HUB_REPO_DEFAULT, work=None, stores=F10_STORES,
                   use_hub=True, index_path=F7_INDEX, include_argo=True):
    g, g_meta = tier_g_groups(repo, use_hub=use_hub, index_path=index_path)
    p, missing = tier_p_groups(repo, work=work, stores=stores,
                               use_hub=use_hub, include_argo=include_argo)
    groups = g + p
    reg = {
        "family": FAMILY,
        "description": (
            "Family 10 is family 7.1's dense gridded groups (tier G) and the "
            "observation stores (tier P) as ONE family under one token schema: "
            "nothing is resampled to a common grid at storage time, every "
            "group keeps its own resolution and cadence, and the token carries "
            "the measurement's footprint so a consumer can tell a 1.9° "
            "reanalysis average from a point profile at the same distance. "
            "Dispatch on `tier`."),
        "plan": "ml/plans/E079_family10_point_stores.md",
        "design": "ml/plans/E078_multi_granularity.md",
        "handover": "docs/FAMILY10_DATA_HANDOVER.md",
        "epoch": f10.EPOCH, "pentad_days": f10.PENTAD_DAYS,
        "bin_rule": ("bin = floor((date - 1982-01-01) / 5 days); a tier-P "
                     "store may hold NEGATIVE bins (drifters from 1979, SOCAT "
                     "from 1957) and its CSR index runs over its own "
                     "bin_first .. bin_first + n_bins - 1"),
        "token_schema": TOKEN_SCHEMA,
        "tiers": {
            "G": "gridded dense — one bin-major .npy at the native grid; a "
                 "read is a byte-range slab of one bin",
            "P": "points, profiles and tracks — columns sorted by pentad bin "
                 "with CSR offsets; a read is a k-nearest search",
            "T": "tiles — a catalogue plus a local codec's cached tokens, read "
                 "through the tier-P reader. NOT BUILT (E-078 §6 step 3).",
        },
        "tier_g": g_meta,
        "readers": {
            "G": "ml/cone_sampler.py (and src/app.js for the globe layer)",
            "P": "ml/family10_store.py :: Store — which also opens family 8's "
                 "Argo store unchanged, by reading its temp/psal blocks as one "
                 "32-column value matrix",
        },
        "repo": repo,
        "n_groups": len(groups),
        "groups_missing": missing,
        "groups": groups,
        "builder": "ml/build_family10_registry.py",
        "builder_git_sha": git_sha(),
        "generated_utc": utcnow(),
    }
    if missing:
        reg["groups_missing_note"] = (
            f"{missing} could not be read, from the work directory or from the "
            f"Hub. They are NOT in `groups` — a registry that listed a group "
            f"it could not describe would be worse than one that is short. "
            f"Build or publish them and re-run this builder.")
    return reg


def publish(path, repo=None):
    """Upload the registry and DOWNLOAD IT BACK to compare sha256."""
    from huggingface_hub import hf_hub_download
    import shutil
    api, resolved, tok = hub_repo()
    repo = repo or resolved
    src = sha256(path)
    api.create_repo(repo, repo_type="dataset", exist_ok=True, private=False)
    api.upload_file(path_or_fileobj=path,
                    path_in_repo=f"{HF_ROOT}/{REGISTRY_NAME}",
                    repo_id=repo, repo_type="dataset",
                    commit_message="family 10: the registry")
    scratch = path + ".verify"
    shutil.rmtree(scratch, ignore_errors=True)
    back = hf_hub_download(repo, f"{HF_ROOT}/{REGISTRY_NAME}",
                           repo_type="dataset", token=tok, local_dir=scratch)
    got = sha256(back)
    shutil.rmtree(scratch, ignore_errors=True)
    if got != src:
        sys.exit(f"RESTORE MISMATCH {REGISTRY_NAME}: uploaded {src}, "
                 f"downloaded {got} — the publish is not trustworthy")
    print(f"  publish: verified by restore -> https://huggingface.co/datasets/"
          f"{repo}/blob/main/{HF_ROOT}/{REGISTRY_NAME}")
    return got


def main():
    ap = argparse.ArgumentParser(
        description="Write tensors/family10/family10.json — the family-10 "
                    "registry (E-079 §2, E-078 §3).")
    ap.add_argument("--out", default="",
                    help="where to write it (default <work>/family10.json)")
    ap.add_argument("--work", default=os.path.join(HERE, "cache", FAMILY),
                    help="a local build directory; a store.json found here is "
                         "preferred over the Hub copy")
    ap.add_argument("--repo", default=HUB_REPO_DEFAULT)
    ap.add_argument("--stores", default=",".join(F10_STORES),
                    help="comma-separated tier-P stores to include")
    ap.add_argument("--index", default=F7_INDEX,
                    help="data/family7_index.json — the tier-G channel and "
                         "grid metadata")
    ap.add_argument("--no-hub", action="store_true",
                    help="never reach the Hub; use only what is on disk")
    ap.add_argument("--no-argo", action="store_true",
                    help="leave family 8's Argo store out")
    ap.add_argument("--publish", action="store_true",
                    help="upload the registry and verify the restore")
    ap.add_argument("--require-all", action="store_true",
                    help="exit non-zero if any named group could not be read")
    a = ap.parse_args()

    stores = [s.strip() for s in a.stores.split(",") if s.strip()]
    reg = build_registry(repo=a.repo, work=os.path.abspath(a.work),
                         stores=stores, use_hub=not a.no_hub,
                         index_path=a.index, include_argo=not a.no_argo)
    out = a.out or os.path.join(os.path.abspath(a.work), REGISTRY_NAME)
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    atomic_json(out, reg)
    tiers = {}
    for g in reg["groups"]:
        tiers[g["tier"]] = tiers.get(g["tier"], 0) + 1
    print(f"registry  {out}")
    print(f"          {reg['n_groups']} group(s): "
          + ", ".join(f"{n} tier-{t}" for t, n in sorted(tiers.items())))
    for g in reg["groups"]:
        print(f"          {g['tier']} {g['name']:9s} C={g.get('C')} "
              f"bin_first={g.get('bin_first')} "
              f"fp=({g['footprint']['log2_fp']:g}, "
              f"{g['footprint']['log2_dt']:g}) "
              f"{len(g.get('files') or [])} file(s)")
    if reg["tier_g"].get("fallback_note"):
        print(f"::warning::{reg['tier_g']['fallback_note']}")
    if reg["groups_missing"]:
        print(f"::warning::not listed: {reg['groups_missing']} — "
              f"{reg['groups_missing_note']}")
        if a.require_all:
            return 1
    if a.publish:
        publish(out, repo=a.repo)
    return 0


if __name__ == "__main__":
    sys.exit(main())

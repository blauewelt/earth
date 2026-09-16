#!/usr/bin/env python3
"""Family 10's REGISTRY — one file that lists every group of the family.

E-079 §2 ("The registry"), E-078 §3 ("One registry") and E-081 §2, made
runnable. Writes `tensors/family10_2/family10.json`: every group of family
10.2 with its `tier`, layout, cadence, channels with units, footprint
constants, `bin_first`, its own `path`, files with sha256, sources and builder
commit. **A consumer dispatches on `tier` and needs no other document.**

FAMILY 10.2 ADDS ONE GROUP AND MOVES NOTHING. `fishing` (E-081) is built under
`tensors/family10_2/`; the four tier-P stores of 10.1 are listed from
`tensors/family10_1/<store>/` where they already are, and family 7.1's tier-G
groups from their own manifest — so a 10.2 registry is written without a byte
of 10.1 or 7.1 being rebuilt or re-uploaded. `STORE_ROOTS` below is the table
that says which root each store lives under, every group carries its own
`path`, and the top-level `inherits` block names what was taken as it stood.

FAMILY 10.1. The registry carries `family_version` "10.1" and
`schema_version` 2 — the tier-P stores' time column is `time_s`, int32 seconds
since 1982-01-01T00:00:00Z, where family 10's was `time_days`, float32 days
(E-079 §10.1). Tier G is UNCHANGED: the same `f7l2` family-7.1 manifest, listed
by reference. Each tier-P entry additionally states ITS OWN `schema_version`,
read out of its `store.json`, because family 8's Argo store joins the family
unchanged and is still schema 1 — a consumer that needs the seconds has to be
able to see which groups have them.

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

WHICH FAMILY-7 MANIFEST. E-079 §2 says family 7.1, and family 7.1 has now been
built twice: `f7l1` (2026-09-14 15:19Z), whose elevation static was published
empty and whose two logarithmic wind-forcing channels did not reproduce, and
`f7l2` (2026-09-14 19:58Z), the corrected build — same four groups, three of
them inherited byte-for-byte (E-077 §10). The builder asks for the NEWEST
recipe first and walks back — l2, then l1, then the three-group l0 — and
RECORDS WHICH ONE IT USED in `tier_g.manifest`, `tier_g.recipe` and, whenever
that is not the newest, `tier_g.fallback_note`. A registry that silently
described the wrong tensor would be worse than one that refused.

Run:
  python3 ml/build_family10_registry.py --out /tmp/family10.json
  python3 ml/build_family10_registry.py --work ml/cache/family10_2 --publish
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

# ONE CONSTANT, in `ml/family10_store.py`: FAMILY_VERSION = "10.2" derives
# `tensors/family10_2` here and `partials/family10_2` in the parts hub, so the
# registry cannot end up describing one prefix while the builder writes
# another. What it does NOT derive is where an INHERITED store lives — see
# `STORE_ROOTS` below.
FAMILY = f10.FAMILY
FAMILY_VERSION = f10.FAMILY_VERSION
SCHEMA_VERSION = f10.SCHEMA_VERSION
HF_ROOT = f10.HF_ROOT                          # tensors/family10_2
REGISTRY_NAME = f10.REGISTRY_NAME              # family10.json
HUB_REPO_DEFAULT = "chfrank/earth-tensors"
HUB_BASE = "https://huggingface.co/datasets/{repo}/resolve/main/{path}"

# The tier-G tensor, NEWEST RECIPE FIRST — the order is the preference, and
# the fallback note below is written from the position the loop stopped at.
# f7l2 is the corrected family-7.1 build (E-077 §10); f7l1 is the first
# four-group one; f7l0 is the three-group tensor that has no ocean colour.
F7_STEMS = ("family7_global025_pentad_l2", "family7_global025_pentad_l1",
            "family7_global025_pentad_l0")
F7_INDEX = os.path.join(os.path.dirname(HERE), "data", "family7_index.json")

# The Argo store of family 8, which becomes family 10's first tier-P group
# UNCHANGED (E-078 §7: the design adds a registry and a tier and takes nothing
# away).
F8_PREFIX = "tensors/family8_argo_l0"
F8_NAME = "argo"

# E-079's four stores and E-081's fifth, in the plans' own order.
F10_STORES = ("gdp", "gtmba", "socat", "slatrack", "fishing")

# WHICH ROOT EACH TIER-P STORE LIVES UNDER — the whole of what family 10.2
# changes about the registry (E-081 §2). The four stores of E-079 §10.1 stay
# exactly where they were published, under `tensors/family10_1/`, and are
# listed from there BY REFERENCE: not one byte of them is rebuilt, re-hashed
# or re-uploaded, the same way tier G inherits family 7.1's manifest. Only
# `fishing` is a 10.2 build and only it lives under `tensors/family10_2/`.
#
# A store's root is a property of WHERE IT WAS BUILT, so it belongs in a table
# rather than in a format string over `FAMILY_VERSION`: deriving it would move
# the four the moment the family's version changed, and the registry would
# then describe four prefixes that hold nothing.
INHERITS_VERSION = "10.1"
INHERITS_ROOT = "tensors/family10_1"
INHERITED_STORES = ("gdp", "gtmba", "socat", "slatrack")
STORE_ROOTS = {s: INHERITS_ROOT for s in INHERITED_STORES}
STORE_ROOTS["fishing"] = HF_ROOT               # tensors/family10_2


def store_root(name):
    """The Hub root a tier-P store is published under. One lookup, one place."""
    return STORE_ROOTS.get(name, HF_ROOT)


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
    elif stem != F7_STEMS[0]:
        # Something newer than what we are describing exists in the preference
        # list and did not answer. Say which, and say what the reader loses.
        skipped = ", ".join(F7_STEMS[:F7_STEMS.index(stem)])
        note = (f"the newest family-7 tensor this builder knows about "
                f"({F7_STEMS[0]}) has no readable manifest on the Hub — "
                f"tensors/{F7_STEMS[0]}/manifest.json did not answer, and "
                f"neither did: {skipped}. This registry therefore describes "
                f"{stem}. Re-run this builder once the newer build has "
                f"published and it will pick it up with no change here.")
    idx = read_json(index_path, {}) or {}
    hashes = {f["name"]: f for f in (man or {}).get("files", [])}
    out = []
    for name, g in (idx.get("groups") or {}).items():
        # `data/family7_index.json` may describe a DIFFERENT generation of the
        # tensor than the manifest that answered — it is regenerated by hand
        # after a build lands, and the two can be one publish apart. The file
        # names belong to the manifest's generation, so rewrite whichever stem
        # the index happens to carry rather than assuming the oldest one.
        fname = g.get("file", "")
        if stem and fname:
            for s in F7_STEMS:
                if fname.startswith(s):
                    fname = stem + fname[len(s):]
                    break
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
            "path": f"tensors/{stem}",
            "inherited_from": "7.1",
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
    # The store's OWN schema, out of its own store.json. Family 8's Argo store
    # is schema 1 and stays so; E-079 §10.1's four rebuilds are schema 2.
    sv = int(meta.get("schema_version", 1) or 1)
    tcol = f10.TIME_COLUMN[sv]
    return {
        "name": name, "tier": "P",
        # `path` is where this group's files ARE, and it is not derivable from
        # the registry's own version: family 10.2 lists four groups that live
        # under `tensors/family10_1/` and are inherited unchanged (E-081 §2).
        "path": prefix,
        "family_version": meta.get("family_version"),
        "inherited_from": (INHERITS_VERSION
                           if prefix.startswith(INHERITS_ROOT + "/") else None),
        "schema_version": sv,
        "time_column": (
            f"{tcol}.npy — "
            + ("int32 SECONDS since 1982-01-01T00:00:00Z, exact"
               if sv >= 2 else
               "float32 days since 1982-01-01, which resolves 21 s in 1993 "
               "and 84 s in 2024 (schema 1)")),
        "layout": f"columns sorted by (bin, {tcol}) with CSR bin offsets; a "
                  f"read is a k-nearest search, one-sided in time, bounded by "
                  f"(R_max_km, T_max_days), miss tokens to a fixed k",
        "reader": "ml/family10_store.py :: Store.open(<dir or "
                  "'<owner>/<repo>:<prefix>'>).knearest(...) — reads schema 1 "
                  "and schema 2, and reports dt in days under both",
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
        root = store_root(s)
        if work:
            # The 10.1 stores were built under `ml/cache/family10_1` and 10.2
            # builds under `ml/cache/family10_2`, so a local copy of an
            # INHERITED store sits in the sibling directory — checked here so
            # a box that still holds one does not have to re-read the Hub.
            sib = os.path.join(os.path.dirname(os.path.abspath(work)),
                               root.split("/")[-1])
            for cand in (os.path.join(work, s, s, "store.json"),
                         os.path.join(work, s, "store.json"),
                         os.path.join(sib, s, s, "store.json"),
                         os.path.join(sib, s, "store.json")):
                if os.path.exists(cand):
                    meta, local = read_json(cand, None), os.path.dirname(cand)
                    break
        if meta is None and use_hub:
            meta = hub_json(repo, f"{root}/{s}/store.json")
        if meta is None:
            missing.append(s)
            continue
        out.append(_store_entry(s, meta, repo, f"{root}/{s}", local))
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
        "family_version": FAMILY_VERSION,
        "schema_version": SCHEMA_VERSION,
        "schema_version_note": (
            "schema 2 (family 10.1): a tier-P store's time column is "
            "`time_s`, int32 SECONDS since 1982-01-01T00:00:00Z, and "
            "bin = floor(time_s / 432000) is integer arithmetic. Schema 1 "
            "(family 10, family 8) carried `time_days`, float32 days, which "
            "resolves 21 s in 1993 and 84 s in 2024 — against slatrack's 1 Hz "
            "sampling up to 316 consecutive samples shared one timestamp. "
            "Every group states its own `schema_version`; family 8's Argo "
            "store is schema 1 and is not rebuilt. Tier G is unchanged."),
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
        "bin_rule": ("bin = floor(time_s / 432000 s) on a schema-2 store, "
                     "floor((date - 1982-01-01) / 5 days) on a schema-1 one — "
                     "the same bins, derived without a float in the newer "
                     "form. A tier-P store may hold NEGATIVE bins (drifters "
                     "from 1979, SOCAT from 1957) and its CSR index runs over "
                     "its own bin_first .. bin_first + n_bins - 1"),
        "token_schema": TOKEN_SCHEMA,
        "tiers": {
            "G": "gridded dense — one bin-major .npy at the native grid; a "
                 "read is a byte-range slab of one bin",
            "P": "points, profiles and tracks — columns sorted by pentad bin "
                 "with CSR offsets; a read is a k-nearest search",
            "T": "tiles — a catalogue plus a local codec's cached tokens, read "
                 "through the tier-P reader. NOT BUILT (E-078 §6 step 3).",
        },
        # WHAT 10.2 DID NOT REBUILD. Family 10.2 adds `fishing` and changes
        # nothing else: the four tier-P stores of 10.1 and family 7.1's four
        # tier-G groups are listed from the prefixes that already hold them,
        # with the sha256 those builds published. A reader can tell an
        # inherited group from a new one without diffing two registries, and a
        # future 10.3 extends this block rather than moving any bytes.
        "inherits": {
            INHERITS_VERSION: {
                "root": INHERITS_ROOT,
                "groups": [g["name"] for g in p
                           if (g.get("path") or "").startswith(
                               INHERITS_ROOT + "/")],
                "registry": f"{INHERITS_ROOT}/{REGISTRY_NAME}",
                "note": ("built by E-079 §10.1 and unchanged: not one byte "
                         "was rebuilt, re-hashed or re-uploaded for 10.2. "
                         "Each group's own `path` says where it is."),
            },
            "8": {"root": F8_PREFIX, "groups": [F8_NAME] if include_argo
                  else [],
                  "note": "family 8's Argo store, schema 1, joined unchanged"},
            "7.1": {"root": f"tensors/{g_meta.get('stem')}",
                    "groups": [x["name"] for x in g],
                    "note": ("tier G by reference to family 7.1's own "
                             "manifest — the registry never copies a "
                             "tensor's metadata, because a copy is the thing "
                             "that goes stale")},
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
        description=f"Write {HF_ROOT}/{REGISTRY_NAME} — the family-"
                    f"{FAMILY_VERSION} registry (E-079 §2, §10.1, E-078 §3).")
    ap.add_argument("--out", default="",
                    help="where to write it (default <work>/family10.json)")
    ap.add_argument("--work",
                    default=os.path.join(HERE, "cache", f10.CACHE_DIRNAME),
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

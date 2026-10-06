#!/usr/bin/env python3
"""Refuse to publish a REBUILT registry that changes more than it was rebuilt
for (E-085).

PLAIN ENGLISH. Family 10.2's registry was last written on a box that held
some of its stores locally; rebuilt on a hosted runner it reads every store
from the Hub instead. Rebuilding it to add ONE line — family 1.2 in its
`siblings` list — must not silently change anything else, so this compares
the published file with the new one and allows exactly:

  * `siblings` (the line being added), `generated_utc`, `builder_git_sha`;
  * a file's `bytes` going from null to a number while its `sha256` is the
    same (the builder now reads sizes from the store's own manifest when the
    store is not on local disk; it never invents them).

Two opt-in widenings, each for one named reason:

  --spans        decision D7 (2026-10-06): the span fields of every group
                 (`record_span`, `record_span_instants`, `record_span_basis`,
                 `requested_window`, `date_range` — ml/registry_spans.py) and
                 the top-level `span_rule` may change, PROVIDED each group's
                 new `requested_window` equals what its old `date_range` said
                 (the requested window moved, it was not lost).
  --allow-group NAME
                 one store was rebuilt (e.g. era5_q under decision D6): every
                 field of the group whose `name` is NAME may change, and only
                 that group.

  --allow-group rebuilt_<code>
                 EVERY store of the registry whose `family_code` is <code>
                 was rebuilt (E-087 §14: family 7.2d's four stores extended
                 to each producer's last day, one of them built for the first
                 time): every group may change, PROVIDED the set of group
                 names is the same, and so may the top-level build state
                 (`built`, `not_built`, `n_built`) and `description`, which
                 says what the rebuild changed. Every other top-level field
                 is still held to equality. The token fits the workflow's
                 `allow_group` input (^[a-z0-9_]*$), and the workflow passes
                 it through unchanged. The changed groups are printed with
                 what changed in each, for review.

Anything else — a group gone, a sha256 changed, a size changed, a size LOST
— exits 1 with the paths that differ, and nothing is published.

  python3 ml/registry_guard.py <published.json> <rebuilt.json>
          [--spans] [--allow-group NAME ...]
"""
import argparse
import json
import re
import sys

ALLOWED_TOP = {"siblings", "generated_utc", "builder_git_sha"}
REBUILT_TOP = {"built", "not_built", "n_built", "description"}
SPAN_TOP = {"span_rule"}
SPAN_FIELDS = {"record_span", "record_span_instants", "record_span_basis",
               "requested_window", "date_range"}
GROUP_PATH = re.compile(r"^\.groups\[(\d+)\]")


def walk(a, b, path, bad, filled, spans_seen, opts):
    if isinstance(a, dict) and isinstance(b, dict):
        for k in sorted(set(a) | set(b)):
            if not path and (k in ALLOWED_TOP
                             or (opts["spans"] and k in SPAN_TOP)
                             or (opts["rebuilt"] and k in REBUILT_TOP)):
                continue
            if opts["spans"] and k in SPAN_FIELDS \
                    and GROUP_PATH.match(path) \
                    and path == GROUP_PATH.match(path).group(0):
                if a.get(k, "<absent>") != b.get(k, "<absent>"):
                    spans_seen.append((f"{path}.{k}", a.get(k, "<absent>"),
                                       b.get(k, "<absent>")))
                continue
            walk(a.get(k, "<absent>"), b.get(k, "<absent>"), f"{path}.{k}",
                 bad, filled, spans_seen, opts)
    elif isinstance(a, list) and isinstance(b, list) and len(a) == len(b):
        for i, (x, y) in enumerate(zip(a, b)):
            if path == ".groups" and opts["allow_group"] \
                    and isinstance(x, dict) and isinstance(y, dict) \
                    and x.get("name") == y.get("name") \
                    and (x.get("name") in opts["allow_group"]
                         or opts["rebuilt"]):
                if x != y:
                    opts["groups_changed"].append(x.get("name"))
                continue
            walk(x, y, f"{path}[{i}]", bad, filled, spans_seen, opts)
    elif a != b:
        if path.endswith(".bytes") and a is None and isinstance(b, int):
            filled.append(path)
        else:
            bad.append(f"{path}: {json.dumps(a)[:120]} -> {json.dumps(b)[:120]}")


def window_kept(old, new, bad):
    """With --spans: each group's new requested_window must equal the window
    its old date_range stated (when the old one had a date_range)."""
    og = {g.get("name"): g for g in old.get("groups") or []}
    for g in new.get("groups") or []:
        o = og.get(g.get("name"))
        if not o or "date_range" not in o:
            continue
        if o.get("requested_window") is not None:
            continue                       # already migrated earlier
        if g.get("requested_window") != o.get("date_range"):
            bad.append(f"group {g.get('name')}: requested_window "
                       f"{g.get('requested_window')} != the old date_range "
                       f"{o.get('date_range')} — the window would be lost")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("published")
    ap.add_argument("rebuilt")
    ap.add_argument("--spans", action="store_true")
    ap.add_argument("--allow-group", action="append", default=[])
    a = ap.parse_args(argv)
    old, new = json.load(open(a.published)), json.load(open(a.rebuilt))
    bad, filled, spans_seen = [], [], []
    rebuilt = [g[len("rebuilt_"):] for g in a.allow_group
               if g.startswith("rebuilt_")]
    if rebuilt and str(old.get("family_code")) not in rebuilt:
        print(f"REFUSING: --allow-group rebuilt_{rebuilt[0]} names family "
              f"{rebuilt[0]}, and this registry is family "
              f"{old.get('family_code')}")
        return 1
    if rebuilt and sorted(g.get("name") for g in old.get("groups") or []) \
            != sorted(g.get("name") for g in new.get("groups") or []):
        print("REFUSING: the rebuilt registry does not hold the same stores")
        return 1
    opts = {"spans": a.spans, "allow_group": set(a.allow_group),
            "groups_changed": [], "rebuilt": bool(rebuilt)}
    walk(old, new, "", bad, filled, spans_seen, opts)
    if a.spans:
        window_kept(old, new, bad)
    sib_old = [s.get("registry") for s in (old.get("siblings") or {})
               .get("registries", [])]
    sib_new = [s.get("registry") for s in (new.get("siblings") or {})
               .get("registries", [])]
    print(f"siblings: {sib_old} -> {sib_new}")
    print(f"file sizes filled from manifests (null -> bytes): {len(filled)}")
    if a.spans:
        names = {}
        for p, x, y in spans_seen:
            gi = int(GROUP_PATH.match(p).group(1))
            names.setdefault(gi, []).append(p.rsplit(".", 1)[1])
        print(f"span fields changed in {len(names)} group(s):")
        groups = new.get("groups") or []
        for gi in sorted(names):
            g = groups[gi] if gi < len(groups) else {}
            o = (old.get("groups") or [{}] * (gi + 1))[gi] \
                if gi < len(old.get("groups") or []) else {}
            print(f"  {g.get('name')}: record_span "
                  f"{o.get('record_span', o.get('date_range'))} -> "
                  f"{g.get('record_span')} ({g.get('record_span_basis')}); "
                  f"fields {sorted(names[gi])}")
    if opts["groups_changed"]:
        print(f"groups rebuilt and allowed to change: "
              f"{sorted(set(opts['groups_changed']))}")
    if opts["rebuilt"]:
        og = {g.get("name"): g for g in old.get("groups") or []}
        for g in new.get("groups") or []:
            o = og.get(g.get("name"), {})
            ch = sorted(k for k in set(o) | set(g) if o.get(k) != g.get(k))
            print(f"  {g.get('name')}: built {o.get('built')} -> "
                  f"{g.get('built')}; record_span {o.get('record_span')} -> "
                  f"{g.get('record_span')}; requested_window "
                  f"{o.get('requested_window')} -> "
                  f"{g.get('requested_window')}; fields changed {ch}")
        for k in sorted(REBUILT_TOP):
            if old.get(k) != new.get(k) and k != "description":
                print(f"  top-level {k}: {old.get(k)} -> {new.get(k)}")
    if bad:
        print(f"REFUSING: {len(bad)} other difference(s):")
        for m in bad[:40]:
            print(f"  {m}")
        return 1
    print("GUARD OK: only the allowed fields differ"
          + (" (siblings, timestamps, null sizes"
             + (", span fields" if a.spans else "")
             + (f", group(s) {sorted(opts['allow_group'])}"
                if opts["allow_group"] else "") + ")"))
    return 0


if __name__ == "__main__":
    sys.exit(main())

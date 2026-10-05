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

Anything else — a group gone, a sha256 changed, a size changed, a size LOST
— exits 1 with the paths that differ, and nothing is published.

  python3 ml/registry_guard.py <published.json> <rebuilt.json>
"""
import json
import sys

ALLOWED_TOP = {"siblings", "generated_utc", "builder_git_sha"}


def walk(a, b, path, bad, filled):
    if isinstance(a, dict) and isinstance(b, dict):
        for k in sorted(set(a) | set(b)):
            if not path and k in ALLOWED_TOP:
                continue
            walk(a.get(k, "<absent>"), b.get(k, "<absent>"), f"{path}.{k}",
                 bad, filled)
    elif isinstance(a, list) and isinstance(b, list) and len(a) == len(b):
        for i, (x, y) in enumerate(zip(a, b)):
            walk(x, y, f"{path}[{i}]", bad, filled)
    elif a != b:
        if path.endswith(".bytes") and a is None and isinstance(b, int):
            filled.append(path)
        else:
            bad.append(f"{path}: {json.dumps(a)[:120]} -> {json.dumps(b)[:120]}")


def main(old_path, new_path):
    old, new = json.load(open(old_path)), json.load(open(new_path))
    bad, filled = [], []
    walk(old, new, "", bad, filled)
    sib_old = [s.get("registry") for s in (old.get("siblings") or {})
               .get("registries", [])]
    sib_new = [s.get("registry") for s in (new.get("siblings") or {})
               .get("registries", [])]
    print(f"siblings: {sib_old} -> {sib_new}")
    print(f"file sizes filled from manifests (null -> bytes): {len(filled)}")
    if bad:
        print(f"REFUSING: {len(bad)} other difference(s):")
        for m in bad[:40]:
            print(f"  {m}")
        return 1
    print("GUARD OK: only the siblings line, the timestamps and null sizes "
          "differ")
    return 0


if __name__ == "__main__":
    sys.exit(main(*sys.argv[1:3]))

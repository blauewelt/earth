#!/usr/bin/env python3
"""Free PUBLISHED store work directories from a box's persistent cache.

PLAIN ENGLISH. A rented box keeps every store it assembled under
/opt/earth-cache/<family>/<store>. A published, restore-verified store is
re-derivable from the Hub (parts AND store), so it is the one thing on a box
that is safe to delete — and the `publish.done` marker (written only after
every file was downloaded back and hashed against store.json) is the proof,
not the run colour (ml/CLAUDE.md §0.2). This is how one box assembles a big
store after several published ones.

THE PATH RULE. An item is `<family>/<store>`: `<family>` is `family` then
lower-case letters, digits and `_` (family1_gf, family1_2, family7_2d …),
`<store>` is lower-case letters, digits and `_`, exactly one `/`, nothing
else — no `.`, no `..`, no second slash, no absolute path. The shell rule it
replaces (`family1_*/[a-z0-9_]*`) refused every family 7.2d directory (E-087
§14: a 250 GB box filled because ncep, oisst and occci could not be freed)
and, because a shell `*` also matches `/` and `.`, accepted
`family1_x/../..`-shaped items.

  python3 ml/free_cache.py --root /opt/earth-cache family7_2d/oisst025d,family1_gf/oc4k
"""
import argparse
import os
import re
import shutil
import sys

ITEM = re.compile(r"^family[a-z0-9_]+/[a-z0-9_]+$")


def parse(spec):
    """The comma list -> items, refusing any that break the path rule."""
    items, bad = [], []
    for it in (spec or "").split(","):
        it = it.strip()
        if not it:
            continue
        (items if ITEM.match(it) else bad).append(it)
    if bad:
        raise ValueError(f"free_cache: {bad} not <family>/<store> "
                         f"(family[a-z0-9_]+/[a-z0-9_]+, no dots, one slash)")
    return items


def free(root, items, log=print):
    """Delete each published work directory. Refuses (raises) on one with no
    publish.done; a directory that is not there is reported and skipped.
    Returns the directories deleted."""
    root = os.path.realpath(root)
    gone = []
    for it in items:
        d = os.path.join(root, it)
        # belt and braces: the regex already forbids escaping the root
        if os.path.commonpath([root, os.path.realpath(d)]) != root:
            raise ValueError(f"free_cache: {it} resolves outside {root}")
        if not os.path.isdir(d):
            log(f"free_cache: {d} is not there -- nothing to free")
            continue
        if not os.path.isfile(os.path.join(d, "publish.done")):
            raise ValueError(f"free_cache: {d} has no publish.done -- not a "
                             f"published-and-restored store; refusing to "
                             f"delete it")
        chk = os.path.join(d, "check.done")
        log(f"free_cache: {it} published "
            f"{open(os.path.join(d, 'publish.done')).read().strip()} checked "
            f"{open(chk).read().strip() if os.path.isfile(chk) else '(check stage not run)'}")
        shutil.rmtree(d)
        log(f"freed {d}")
        gone.append(d)
    return gone


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--root", default="/opt/earth-cache")
    ap.add_argument("items")
    a = ap.parse_args(argv)
    try:
        free(a.root, parse(a.items))
    except ValueError as e:
        print(f"::error::{e}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())

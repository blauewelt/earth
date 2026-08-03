#!/usr/bin/env python3
"""Stamp every local asset URL with the hash of the file it points at.

The problem this solves is real and was reported from a phone: a deploy went
out, the page was reloaded twice, and the new feature was not there. Nothing
was wrong with the code -- `index.html` asked for `src/app.js`, the browser
already had a copy of `src/app.js`, and GitHub Pages had told it that copy was
fresh. There is no build step in this project to rename files, so the version
has to be written into the URL instead.

Each asset gets `?v=` plus the first 8 hex of its own sha256. That is MEASURED
from the file, never typed in and never a counter someone has to remember to
bump: change a byte of `src/app.js` and its URL changes with it, so the old
cached copy is simply not what the page asks for any more. Leave the file
alone and the URL is unchanged, so caching still works the way it should.

`manifest.json` is stamped first (its icon `src`s carry the icon hashes) and
only then hashed for `index.html`, because its content depends on theirs.

Run before committing a deploy:  python3 scripts/stamp_assets.py
tests/data.spec.js re-derives every hash and fails if any is stale, so a
forgotten run is caught by the suite rather than by a user on a train.
"""

import hashlib
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

ICONS = ["icon-192.png", "icon-512.png", "icon-512-maskable.png"]
# index.html's local assets, in the order they must be hashed: manifest.json
# last, since stamping the icons rewrites it.
HTML_ASSETS = ICONS + ["src/style.css", "src/app.js", "manifest.json"]


def digest(rel):
    with open(os.path.join(ROOT, rel), "rb") as fh:
        return hashlib.sha256(fh.read()).hexdigest()[:8]


def restamp(text, asset, h):
    """Rewrite every `asset?v=xxxxxxxx` occurrence, whole-name matches only.

    The lookbehind stops "icon-512.png" from also matching the tail of
    "icon-512-maskable.png" -- they differ only by a suffix, and a careless
    match would stamp the maskable icon with the plain icon's hash.
    """
    pat = re.compile(r"(?<![\w./-])" + re.escape(asset) + r"\?v=[0-9a-f]{8}")
    return pat.sub(f"{asset}?v={h}", text)


def main():
    check = "--check" in sys.argv
    stale = []

    # 1. icons -> manifest.json
    man_path = os.path.join(ROOT, "manifest.json")
    man = open(man_path, encoding="utf-8").read()
    for icon in ICONS:
        man = restamp(man, icon, digest(icon))
    # First run: the manifest ships unstamped, so add the query the first time.
    for icon in ICONS:
        man = re.sub(r'"src": "' + re.escape(icon) + r'"',
                     f'"src": "{icon}?v={digest(icon)}"', man)
    if man != open(man_path, encoding="utf-8").read():
        stale.append("manifest.json")
        if not check:
            open(man_path, "w", encoding="utf-8").write(man)

    # 2. everything -> index.html, plus the visible build marker
    idx_path = os.path.join(ROOT, "index.html")
    original = open(idx_path, encoding="utf-8").read()
    idx = original
    for asset in HTML_ASSETS:
        idx = restamp(idx, asset, digest(asset))
    idx = re.sub(r'(<code id="build-id">)[0-9a-f]{8}(</code>)',
                 lambda m: m.group(1) + digest("src/app.js") + m.group(2), idx)
    if idx != original:
        stale.append("index.html")
        if not check:
            open(idx_path, "w", encoding="utf-8").write(idx)

    if check:
        if stale:
            print("stale asset stamps in: " + ", ".join(stale))
            return 1
        print("asset stamps up to date")
        return 0
    print("stamped: " + (", ".join(stale) if stale else "nothing to do"))
    for a in HTML_ASSETS:
        print(f"  {a:24s} {digest(a)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

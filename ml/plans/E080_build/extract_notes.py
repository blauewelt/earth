#!/usr/bin/env python3
"""Extract the deck's slide titles and speaker notes from the E-080 spec.

The spec is the single source of truth. Every `## Slide N · Title` section
contributes its title and the text of its `*Notes.*` paragraph, whitespace
collapsed to a single line. The notes land in the PPTX notes pane and in the
notes PDF verbatim — nothing here rewrites them.

Writes <build>/content.json as {"notes": {"1": ...}, "titles": {"1": ...}}.
"""
import json
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
PLANS = os.environ.get("E080_PLANS", os.path.dirname(HERE))
BUILD = os.environ.get("E080_BUILD", os.path.join(HERE, "build"))
SPEC = os.environ.get("E080_SPEC", os.path.join(PLANS, "E080_hourglass_cone.md"))

src = open(SPEC, encoding="utf8").read()
parts = re.split(r"^## Slide (\d+) · (.+)$", src, flags=re.M)

notes, titles = {}, {}
for k in range(1, len(parts), 3):
    n = int(parts[k])
    titles[n] = parts[k + 1].strip()
    m = re.search(r"^\*Notes\.\*\s+(.+?)(?:\n\n|\n---|\Z)", parts[k + 2],
                  flags=re.S | re.M)
    if not m:
        sys.exit(f"slide {n} has no *Notes.* paragraph")
    notes[n] = " ".join(m.group(1).split())

expected = list(range(1, len(notes) + 1))
if sorted(notes) != expected:
    sys.exit(f"slide numbers are not 1..{len(notes)}: {sorted(notes)}")

os.makedirs(BUILD, exist_ok=True)
with open(os.path.join(BUILD, "content.json"), "w", encoding="utf8") as fh:
    json.dump({"notes": {str(k): v for k, v in notes.items()},
               "titles": {str(k): v for k, v in titles.items()}},
              fh, ensure_ascii=False, indent=1)

links = sum(len(re.findall(r"\[[^\]]+\]\([^)]+\)", v)) for v in notes.values())
print(f"content.json: {len(notes)} slides, {links} markdown links in the notes")

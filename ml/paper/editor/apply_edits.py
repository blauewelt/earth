#!/usr/bin/env python3
"""Fold the edits saved on the paper's editing page into paper.tex.

The page stores one document per edited block in its database collection
`edits` (see build_editor.py). Read them with the ArtifactData tool (action
"list", collection "edits"), save the result as JSON, and run

    python3 ml/paper/editor/apply_edits.py edits.json [--dry-run]

Accepted JSON shapes: a list of edit bodies, a list of {"id", "data"} rows, or
{"docs": [...]}. Each edit carries `original` (the block's text when the edit
was made) and `text` (the new text). An edit is applied only if `original`
occurs EXACTLY ONCE in the current paper.tex; otherwise it is reported and
left for a person — an edit made against text that has since changed is never
laid over the new text. Prints what was applied and what was not; exits 1 if
anything was not applied.
"""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
TEX = os.path.join(os.path.dirname(HERE), "paper.tex")


def rows(obj):
    if isinstance(obj, dict):
        obj = obj.get("docs") or obj.get("documents") or obj.get("rows") or []
    for r in obj:
        body = r.get("data", r) if isinstance(r, dict) else None
        if isinstance(body, dict) and "original" in body and "text" in body:
            yield r.get("id", body.get("block", "?")), body


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    dry = "--dry-run" in sys.argv
    tex = open(TEX, encoding="utf-8").read()
    edits = list(rows(json.load(open(args[0], encoding="utf-8"))))
    applied, skipped = [], []
    for bid, e in sorted(edits, key=lambda x: x[1].get("line") or 0):
        n = tex.count(e["original"])
        if n != 1:
            skipped.append((bid, e.get("line"), f"original occurs {n} times"))
            continue
        tex = tex.replace(e["original"], e["text"], 1)
        applied.append((bid, e.get("line"), e.get("where", "")))
    for bid, line, where in applied:
        print(f"applied  {bid}  line {line}  {where}")
    for bid, line, why in skipped:
        print(f"SKIPPED  {bid}  line {line}  {why}")
    if applied and not dry:
        open(TEX, "w", encoding="utf-8").write(tex)
    print(f"{len(applied)} applied, {len(skipped)} skipped"
          + (" (dry run, nothing written)" if dry else ""))
    return 1 if skipped else 0


if __name__ == "__main__":
    sys.exit(main())

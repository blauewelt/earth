#!/usr/bin/env python3
"""Static guards on .github/workflows/ml-train.yml and ml/recipes/.

These are cheap and they protect against failures whose blast radius is the
WHOLE repository, not one run:

Case 1: every `run:` block stays under the 21,000-char expression ceiling.
        On 2026-08-17 a six-line COMMENT added inside the Probes block pushed
        it over, GitHub answered "failed to parse workflow: Exceeded max
        expression length 21000", and for ten minutes EVERY dispatch 422'd —
        including other sessions' unrelated jobs. The largest block runs at
        ~19.6k, so the headroom is about one long comment wide.
Case 2: no ${RECIPE_*} or ${WINDOW} shell variable appears in a YAML
        expression context (key:, env:, if:, with:). Those are evaluated by
        GitHub before a shell exists, so a shell variable there is silently
        literal text. The recipe wiring hit this twice in one pass: once in
        the cache key, and once in the resolver step's own env.
Case 3: every recipe parses, names only real ml-train.yml inputs, and names
        only inputs the workflow actually reads as $RECIPE_<KEY>.
Case 4: the workflow still passes an architecture through to train.py, and
        does NOT reintroduce a literal pilot fallback for it.

    python3 tests/test_workflow_config.py
"""
import glob
import json
import os
import re
import sys

import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
WF = os.path.join(ROOT, ".github", "workflows", "ml-train.yml")

CEILING = 21000
# The observed failure was at ~21k with the block at 19,625 + ~1,400 of
# comment. Warn well before the cliff, because the cliff takes the repo down.
WARN_AT = 20000


def main():
    raw = open(WF).read()
    doc = yaml.safe_load(raw)
    steps = doc["jobs"]["train"]["steps"]
    ok = 0

    # ---- case 1: expression ceiling --------------------------------------
    worst, worst_name = 0, ""
    for i, st in enumerate(steps):
        r = st.get("run")
        if not r:
            continue
        name = st.get("name") or f"step {i}"
        if len(r) > worst:
            worst, worst_name = len(r), name
        if len(r) >= CEILING:
            raise SystemExit(
                f"case 1 FAILED: run block {name!r} is {len(r):,} chars, at or "
                f"over the {CEILING:,} ceiling. GitHub will refuse to parse "
                f"the whole workflow and EVERY dispatch in this repo will "
                f"422 — not just this one. Move the text into a script under "
                f"scripts/ and call it in one line.")
    print(f"case 1 ok — largest run block {worst_name!r} at {worst:,} chars "
          f"({CEILING - worst:,} of headroom)")
    if worst >= WARN_AT:
        print(f"  WARNING: within {CEILING - worst:,} chars of the ceiling. "
              f"The next long comment is the one that breaks dispatch.")
    ok += 1

    # ---- case 2: no shell variables in YAML expression contexts ----------
    offenders = []
    for i, st in enumerate(steps):
        for key, val in st.items():
            if key == "run":
                continue
            blob = yaml.dump(val)
            for pat in ("${RECIPE_", "${WINDOW}"):
                if pat in blob:
                    offenders.append((st.get("name", f"step {i}"), key, pat))
    if offenders:
        for n, k, pat in offenders:
            print(f"  {n!r} -> {k}: contains {pat}")
        raise SystemExit(
            "case 2 FAILED: a shell variable sits in a YAML expression "
            "context. GitHub evaluates those before a shell exists, so the "
            "text is passed through literally and whatever read it gets "
            "garbage — silently.")
    print("case 2 ok — shell variables appear only inside run: blocks")
    ok += 1

    # ---- case 3: recipes are real and wired ------------------------------
    blk = raw[raw.index("  workflow_dispatch:"):raw.index("\npermissions:")]
    valid = set(re.findall(r"^      (\w+):\s*$", blk, re.M))
    if not valid:
        raise SystemExit("case 3 FAILED: parsed zero inputs out of the "
                         "workflow — the input-block markers moved")
    recipes = sorted(glob.glob(os.path.join(ROOT, "ml", "recipes", "*.json")))
    if not recipes:
        raise SystemExit("case 3 FAILED: no recipes found")
    for path in recipes:
        name = os.path.basename(path)[:-5]
        d = json.load(open(path))
        keys = [k for k in d if not k.startswith("_")]
        if not keys:
            raise SystemExit(f"case 3 FAILED: recipe {name} sets nothing")
        for k in keys:
            if k not in valid:
                raise SystemExit(f"case 3 FAILED: recipe {name} sets {k!r}, "
                                 f"which is not an ml-train.yml input")
            if f"RECIPE_{k.upper()}" not in raw:
                raise SystemExit(
                    f"case 3 FAILED: recipe {name} sets {k!r} but the "
                    f"workflow never reads $RECIPE_{k.upper()} — the setting "
                    f"would appear to apply and do nothing")
        if not d.get("_description") or not d.get("_provenance"):
            raise SystemExit(f"case 3 FAILED: recipe {name} is missing "
                             f"_description or _provenance. A recipe records "
                             f"a configuration that was RUN; say which run.")
    print(f"case 3 ok — {len(recipes)} recipes, all wired and documented")
    ok += 1

    # ---- case 4: the architecture still reaches train.py, unfaked --------
    train = [st for st in steps if st.get("name") == "Train"]
    if not train:
        raise SystemExit("case 4 FAILED: no Train step")
    run = train[0]["run"]
    for flag, inp in (("--d-model", "codec_d_model"),
                      ("--n-layers", "codec_layers"),
                      ("--n-heads", "codec_heads"),
                      ("--d-dec", "codec_d_dec")):
        if flag not in run:
            raise SystemExit(f"case 4 FAILED: Train no longer passes {flag}")
        line = [ln for ln in run.split("\n") if flag in ln][0]
        if f"RECIPE_{inp.upper()}" not in line:
            raise SystemExit(f"case 4 FAILED: {flag} does not consult "
                             f"$RECIPE_{inp.upper()} — recipes cannot set it")
        # The pilot fallback is what made an omitted field a different
        # experiment instead of an error. It must not come back.
        if re.search(r"\|\|\s*'\d+'", line):
            raise SystemExit(
                f"case 4 FAILED: {flag} has a literal fallback again "
                f"({line.strip()}). train.py refuses an unset architecture on "
                f"purpose; a fallback here restores the silent-wrong-model "
                f"behaviour that killed #395 and mis-built #387.")
    print("case 4 ok — architecture flows through, with no literal fallback")
    ok += 1

    print(f"\nall {ok}/4 workflow guards hold")


if __name__ == "__main__":
    main()

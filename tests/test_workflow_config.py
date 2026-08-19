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
    # THE INPUT LIST IS FULL AT 25 and a 26th makes the whole file
    # unparseable, so a knob with nowhere to go may be declared a RECIPE-ONLY
    # key in the workflow's own comment block. It is exempt from "must be a
    # dispatch input" and NOT exempt from the $RECIPE_<KEY> consumption check
    # below, which is the half that stops a setting from silently doing
    # nothing. Same regex as scripts/resolve_recipe.sh, same source of truth.
    recipe_only = set(re.findall(r"#\s*recipe-only:\s*(\w+)", raw))
    if len(valid) > 25:
        raise SystemExit(
            f"case 3 FAILED: ml-train.yml declares {len(valid)} "
            f"workflow_dispatch inputs. The cap is 25 and a 26th does not "
            f"fail gracefully — GitHub refuses to parse the file and EVERY "
            f"dispatch in the repo 422s (ml/CLAUDE.md §7). Encode the knob "
            f"into an existing input, or declare it a RECIPE-ONLY key.")
    valid |= recipe_only
    consumers = raw + "".join(
        open(f).read() for f in sorted(glob.glob(os.path.join(ROOT, "scripts", "*.sh"))))
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
                                 f"which is neither an ml-train.yml input nor "
                                 f"a declared RECIPE-ONLY key")
            # Same corpus as scripts/resolve_recipe.sh: the workflow PLUS
            # the scripts it calls, since probes_run.sh now holds most of the
            # ${RECIPE_X} reads.
            if f"RECIPE_{k.upper()}" not in consumers:
                raise SystemExit(
                    f"case 3 FAILED: recipe {name} sets {k!r} but the "
                    f"workflow never reads $RECIPE_{k.upper()} — the setting "
                    f"would appear to apply and do nothing")
            # ...and no reference to it may sit in a YAML expression
            # context, where ${RECIPE_X} is literal text and the recipe would
            # govern half the job while the dispatch governs the other half.
            # `tensor` and `anomaly` both shipped that way for an hour: the
            # recipe switched the data-cache branch while `env: TENSOR:` kept
            # the old path, and the recipe governed the trainer while
            # `if: inputs.anomaly` governed whether the probes ran at all.
            # The one legal form is the deliberate hand-off,
            # `IN_<KEY>: ${{ inputs.<key> }}`, resolved in a shell afterwards.
            for st2 in steps:
                for field, val in st2.items():
                    if field == "run":
                        continue
                    for line in yaml.dump(val).splitlines():
                        if f"inputs.{k}" not in line:
                            continue
                        if re.match(r"\s*IN_" + k.upper() + r"\s*:", line):
                            continue
                        raise SystemExit(
                            f"case 3 FAILED: recipe {name} sets {k!r}, but "
                            f"{st2.get('name', '?')}::{field} reads "
                            f"inputs.{k} in a YAML expression context — "
                            f"evaluated before any shell, so $RECIPE_{k.upper()}"
                            f" cannot reach it. The recipe would apply to part "
                            f"of the job only. Move the read into a run: block "
                            f"(hand off as IN_{k.upper()}) or drop the key.")
            job = doc["jobs"]["train"]
            for field in ("runs-on", "timeout-minutes"):
                if f"inputs.{k}" in str(job.get(field, "")):
                    raise SystemExit(
                        f"case 3 FAILED: recipe {name} sets {k!r}, read at "
                        f"job::{field} — before any step can set an env var.")
        if not d.get("_description") or not d.get("_provenance"):
            raise SystemExit(f"case 3 FAILED: recipe {name} is missing "
                             f"_description or _provenance. A recipe records "
                             f"a configuration that was RUN; say which run.")
    print(f"case 3 ok — {len(recipes)} recipes, all wired and documented; "
          f"{len(valid) - len(recipe_only)} inputs (cap 25) + "
          f"{len(recipe_only)} recipe-only keys {sorted(recipe_only)}")
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

    # ---- case 5: anything that READS $TENSOR must derive and export it ---
    # TENSOR used to arrive from each step's env:, where a recipe could not
    # reach it. It is now derived in the shell — but a shell assignment is
    # invisible to a python heredoc, and "Record provenance" reads
    # os.environ["TENSOR"] with no `$` anywhere, so the first sweep missed it
    # and #403 died with KeyError: 'TENSOR'. Reads take three forms; check
    # for all three, and require `export`.
    READS = re.compile(r'\$\{?TENSOR\b|environ\["TENSOR"\]|environ\.get\("TENSOR"')
    bodies = [(st.get("name", f"step {i}"), st["run"])
              for i, st in enumerate(steps) if isinstance(st.get("run"), str)]
    bodies += [(os.path.basename(f), open(f).read())
               for f in sorted(glob.glob(os.path.join(ROOT, "scripts", "*.sh")))]
    checked = 0
    for name, body in bodies:
        if not READS.search(body):
            continue
        checked += 1
        if "export TENSOR=" in body:
            if "RECIPE_TENSOR" not in body:
                raise SystemExit(
                    f"case 5 FAILED: {name} exports TENSOR without consulting "
                    f"$RECIPE_TENSOR — a recipe naming a different family "
                    f"would apply to the rest of the job and not to this "
                    f"step.")
            continue
        # Not exporting is fine IF something that does export INVOKES it —
        # sroll_run.sh and dectrain_run.sh are called by probes_run.sh and
        # inherit its environment. That inheritance is the whole reason the
        # export matters, so assert the chain rather than the symbol.
        callers = [n for n, b in bodies
                   if name in b and n != name and "export TENSOR=" in b]
        if not callers:
            raise SystemExit(
                f"case 5 FAILED: {name} reads $TENSOR, does not export it, "
                f"and nothing that does export it invokes {name}. It used to "
                f"arrive from the step's env:; it is now derived from "
                f"${{RECIPE_TENSOR:-$IN_TENSOR}} so a recipe can reach it, "
                f"and it must be exported or inherited — otherwise the "
                f"python heredocs cannot see it (#403, KeyError: 'TENSOR').")
    if checked < 3:
        raise SystemExit(f"case 5 FAILED: only {checked} TENSOR readers found; "
                         f"the detector has stopped seeing them")
    print(f"case 5 ok — all {checked} $TENSOR readers derive and export it")
    ok += 1

    print(f"\nall {ok}/5 workflow guards hold")


if __name__ == "__main__":
    main()

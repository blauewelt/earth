#!/usr/bin/env bash
# Resolve a `recipe:<name>` token on the `window` input into dispatch inputs.
#
# WHY — see ml/recipes/README.md. Short version: on #358, an ordinary run, 17
# of 24 inputs had to be overridden, so every dispatch was a 24-field copying
# exercise. #395 and #387 are what copying exercises produce. A recipe lets a
# dispatch name the experiment and inherit the rest.
#
# CONTRACT
#   window: recipe:<name>[,<rest...>]   ->  exports RECIPE_<INPUT> for every key
#                                           in ml/recipes/<name>.json, and
#                                           exports WINDOW with the token
#                                           stripped (so `sroll:`, `stencil:`
#                                           and friends still work after it).
#   window: anything-else               ->  exports WINDOW unchanged, nothing
#                                           else. A no-op, by design: this must
#                                           be safe on every existing dispatch.
#
# Precedence is RECIPE WINS over the workflow's own input defaults, and the
# workflow decides that by writing "${RECIPE_X:-<the input>}" at each use — so
# a recipe cannot silently override something a dispatch stated on purpose,
# because the workflow only falls back to the input when RECIPE_X is unset.
#
# Writes to $GITHUB_ENV when it exists (the Actions path) and echoes the
# assignments either way, so the same script is runnable and readable locally:
#   bash scripts/resolve_recipe.sh 'recipe:f4-40M,stencil:234'
set -euo pipefail
WINDOW="${1-}"
OUT="${GITHUB_ENV:-/dev/null}"

emit() { echo "$1=$2"; echo "$1=$2" >> "$OUT"; }

case "$WINDOW" in
  recipe:*) ;;
  *) emit WINDOW "$WINDOW"; exit 0 ;;
esac

REST="${WINDOW#recipe:}"
NAME="${REST%%,*}"
if [ "$NAME" = "$REST" ]; then TAIL=""; else TAIL="${REST#*,}"; fi
FILE="ml/recipes/${NAME}.json"

# REFUSE on a missing recipe rather than falling through to the defaults. A
# typo'd name that quietly ran the pilot would be this whole change defeated.
[ -f "$FILE" ] || {
  echo "::error::no recipe at ${FILE}. Available: $(ls ml/recipes/*.json 2>/dev/null | xargs -n1 basename 2>/dev/null | sed 's/\.json$//' | tr '\n' ' ')"
  exit 1
}

# Keys must be REAL inputs of ml-train.yml — or keys declared in that file's
# "RECIPE-ONLY KEYS" block, for knobs the 25-input ceiling leaves no room for —
# checked against the workflow file itself rather than a list kept in step
# here: a list would drift, and a recipe key nothing reads is a setting that
# appears to apply and does not.
python3 - "$FILE" "$NAME" <<'PY'
import json, re, sys
path, name = sys.argv[1], sys.argv[2]
wf = open(".github/workflows/ml-train.yml").read()
# The consumer corpus is the workflow PLUS the scripts it calls: bodies that
# outgrew the 21,000-char run-block ceiling (probes_run.sh, dectrain_run.sh,
# sroll_run.sh) now hold most of the ${RECIPE_X} reads, and a check that only
# looked at the workflow would refuse a key purely because its consumer had
# been carved out into a file.
import glob as _glob
consumers = wf + "".join(open(f).read() for f in sorted(_glob.glob("scripts/*.sh")))
blk = wf[wf.index("  workflow_dispatch:"):wf.index("\npermissions:")]
valid = set(re.findall(r'^      (\w+):\s*$', blk, re.M))
# RECIPE-ONLY KEYS. workflow_dispatch takes at most 25 inputs and ml-train.yml
# has exactly 25, so a knob that cannot be squeezed into an existing input has
# nowhere to go as a 26th — a 26th makes the whole file unparseable and 422s
# every dispatch in the repo. A key declared in the workflow's "RECIPE-ONLY
# KEYS" block is exempt from "must be a dispatch input" and NOT exempt from
# the check below, which is the one that actually protects anything: it must
# still be read as $RECIPE_<KEY> somewhere the job runs. The declaration lives
# in ml-train.yml so this list cannot drift from it.
valid |= set(re.findall(r'#\s*recipe-only:\s*(\w+)', wf))
d = json.load(open(path))
keys = [k for k in d if not k.startswith("_")]
bad = [k for k in keys if k not in valid]
if bad:
    raise SystemExit(f"::error::recipe {name} sets unknown input(s) "
                     f"{sorted(bad)} — neither an input of ml-train.yml nor a "
                     f"key declared in its RECIPE-ONLY KEYS block. A key "
                     f"nothing reads is a setting that appears to apply and "
                     f"does not.")
# AND the workflow must actually CONSUME it as ${RECIPE_<KEY>:-...}. This is
# the check that cannot drift: it asks the workflow itself rather than a list
# kept in step with it by hand. `runner` and `job_timeout` are real inputs but
# are read by runs-on/timeout-minutes at JOB START, before any step can set an
# env var — so a recipe naming them would look applied and do nothing, which is
# the precise failure this whole change exists to remove.
unread = [k for k in keys if ("RECIPE_" + k.upper()) not in consumers]
if unread:
    raise SystemExit(f"::error::recipe {name} sets {sorted(unread)}, which "
                     f"ml-train.yml never reads as $RECIPE_<KEY>. Either wire "
                     f"the input up in the workflow or drop it from the "
                     f"recipe — a setting that appears to apply and does not "
                     f"is worse than no setting at all.")

# The OTHER half of "does this setting fully apply?" — that no reference to
# the key sits in a YAML expression context (if:, env:, with:, runs-on:),
# where ${RECIPE_X} is literal text — is enforced by
# tests/test_workflow_config.py rather than here. Deliberately: it is a
# static property of the repo, identical for every run, and checking it needs
# a YAML parser. This step runs BEFORE "Install deps", so the box has no
# PyYAML — run #402 died exactly there, `ModuleNotFoundError: No module named
# 'yaml'`, on a check that could never have told one dispatch from another.
# Development-time rules belong in tests; run-time rules belong here.
lines = [f"RECIPE_{k.upper()}={v}" for k, v in d.items() if not k.startswith("_")]
open("/tmp/recipe.env", "w").write("\n".join(lines) + ("\n" if lines else ""))
print(f"recipe {name}: {d.get('_description', '(no description)')}")
if d.get("_provenance"):
    print(f"  provenance: {d['_provenance']}")
PY

while IFS= read -r line; do
  [ -n "$line" ] || continue
  echo "$line"
  echo "$line" >> "$OUT"
done < /tmp/recipe.env

# The recipe NAME travels with the run, so a reader of metrics.jsonl or the
# status page can tell which configuration produced a curve.
emit RECIPE_NAME "$NAME"
emit WINDOW "$TAIL"

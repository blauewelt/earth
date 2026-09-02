#!/usr/bin/env bash
# The CLASSICAL BASELINE eval — the body of the `lim:` window token.
#
# Spec: lim:K=<k>[,<k>...][,ckpt:<asset|path>][,horizon:N][,starts:N]
#                         [,scope:window|corridor][,maxgb:G]
#   e.g. lim:K=50,100,200
#        lim:K=50,100,200,starts:3,ckpt:run-415.pt
#
# WHAT IT RUNS. `ml/lim_baseline.py`: a Linear Inverse Model fitted on the
# training bins of the tensor's own standardized-anomaly field, rolled from the
# protocol's staggered starts and scored through ml/rollout_spatial.py's OWN
# battery (its StdMonths, ar1_train, corridor_pixels, gate_subset,
# nested_scopes, new_sums, accumulate, skill_block, write_results — imported,
# not reimplemented). The output is a `rollout_spatial.json` in the shape the
# archive already carries, with one `heads` entry per K named `lim_k<K>`, so
# scripts/archive_probes.py publishes it with no change and a paper table can
# put the LIM row beside a head's per-lead numbers.
#
# WHY THE `K=` TOKEN IS SPELLED THAT WAY. The window's token separator is `,`
# and a K LIST is commas too, so `lim:K=50,100,200` would otherwise parse as
# three tokens. The rule is explicit rather than clever: `K=` opens the list
# and every BARE all-digit token after it extends the list, until any other
# `name:value` token closes it. `lim:K=50,100,200,starts:3` therefore reads
# K=50,100,200 and starts=3, and a typo like `lim:K=50,foo` is a refusal
# rather than a silently shorter sweep.
#
# WHAT IT COSTS, AND WHAT IT NEEDS. CPU only. No GPU, no embedding, no Z
# cache, no head weights — the LIM works on the raw pixel field, so the whole
# `sroll:` apparatus of embed-cache keys, chunked release pulls and head
# fetches is absent by construction. What it does need:
#   * the TENSOR (`$TENSOR`, resolved by scripts/probes_run.sh from
#     ${RECIPE_TENSOR:-$IN_TENSOR}) and its X memmap — the `_X.npy` sidecar if
#     the tensor has one (family 5), otherwise one dectrain_extract pass, the
#     same fork sroll_run.sh takes and keyed by the same names so a warm box
#     does not re-extract 10.9 GiB;
#   * a CODEC CHECKPOINT, read ONLY for `chan` (the channel names) and `args`
#     (`holdout_years`, `holdout_lon`). No weights are loaded. On a dispatch
#     carrying `resume: !run-415` the Train phase leaves that checkpoint at
#     ml/runs/actions/pixelmae.pt — the same file the `publishembed` mode
#     reads — and this script picks it up. `ckpt:` overrides, and is the way
#     to run with no Train phase at all.
#
# MEMORY. The training matrix is [T_train, P * n_channels] float32: 8.1 GB at
# the production pentad shape (2,923 training bins x 86,698 window px x 8
# scoreable channels). The boxes have 52-110 GB. `maxgb:` is passed through to
# --max-state-gb, which REFUSES before the read pass rather than OOM-ing after
# it; `scope:corridor` cuts the state to the corridor (2.8 GB) and then
# reports gate/window as unavailable rather than scoring pixels the fitted
# state does not cover.
#
# Lives in a script, not inline, because the Probes `run:` block has a
# 21,000-char dispatch-time expression ceiling (see dectrain_run.sh and
# sroll_run.sh). Fatal on failure, deliberately: this eval IS this mode's job.
set -e
WINDOW="${1:?usage: lim_run.sh 'lim:K=50,100,200[,ckpt:<asset|path>]'}"

SPEC="${WINDOW#lim:}"
KLIST=""
CKPT_SPEC=""
HORIZON_TOK=""
STARTS_TOK=""
SCOPE_TOK=""
MAXGB_TOK=""
IN_K=0
want_int() {                       # want_int <token> <value>
  case "$2" in
    ''|*[!0-9]*) echo "::error::lim token '$1' needs a whole number, got '$2'"
                 exit 1 ;;
  esac
}
for tok in ${SPEC//,/ }; do
  case "$tok" in
    K=*|k=*)   KLIST="${tok#*=}"; IN_K=1
               want_int K "$KLIST" ;;
    ckpt:*)    CKPT_SPEC="${tok#ckpt:}"; IN_K=0 ;;
    horizon:*) HORIZON_TOK="${tok#horizon:}"; IN_K=0
               want_int horizon "$HORIZON_TOK" ;;
    starts:*)  STARTS_TOK="${tok#starts:}";   IN_K=0
               want_int starts "$STARTS_TOK" ;;
    scope:*)   SCOPE_TOK="${tok#scope:}";     IN_K=0
               case "$SCOPE_TOK" in
                 window|corridor) ;;
                 *) echo "::error::lim token 'scope' wants window or"
                    echo "::error::corridor, got '$SCOPE_TOK'"; exit 1 ;;
               esac ;;
    maxgb:*)   MAXGB_TOK="${tok#maxgb:}";     IN_K=0 ;;
    *:*|*=*)   echo "::error::unknown lim token '$tok'"; exit 1 ;;
    "")        ;;
    *)         # A BARE token. Only ever a continuation of the K list, and
               # only while the list is open — otherwise a mistyped token
               # would be swallowed as if it had been asked for.
               if [ "$IN_K" = 1 ]; then
                 want_int K "$tok"; KLIST="${KLIST},${tok}"
               else
                 echo "::error::stray lim token '$tok' — a bare value is only"
                 echo "::error::read as another K, and only directly after"
                 echo "::error::K=<n>. Write 'lim:K=50,100,200'."; exit 1
               fi ;;
  esac
done
[ -n "$KLIST" ] || { echo "::error::no K= in '$WINDOW' — nothing to fit"; exit 1; }
echo "lim: K=$KLIST${CKPT_SPEC:+ ckpt=$CKPT_SPEC}${SCOPE_TOK:+ scope=$SCOPE_TOK}"
df -h / | tail -1
free -g 2>/dev/null | head -2 || true

TENSOR="${TENSOR:-ml/cache/family3_na025.npz}"
[ -s "$TENSOR" ] || { echo "::error::no tensor at $TENSOR"; exit 1; }
STEM="$(basename "$TENSOR" .npz)"
echo "tensor: $TENSOR ($(stat -c%s "$TENSOR") bytes)"

# ---- the codec, for its ARGS only ----------------------------------------
# NOT for its weights: nothing here embeds anything. It is the authority on
# which YEARS are held out and which CHANNELS the tensor carries, and reading
# those from the checkpoint rather than from the dispatch is what stops a LIM
# being fitted on a bin the head it is a baseline for was denied.
if [ -n "$CKPT_SPEC" ]; then
  case "$CKPT_SPEC" in
    */*) CKPT="$CKPT_SPEC" ;;                      # a path on this box
    *)   CKPT="ml/cache/${CKPT_SPEC}"
         [ -s "$CKPT" ] || curl -fsSL -o "$CKPT" \
           "https://github.com/${GITHUB_REPOSITORY}/releases/download/model-checkpoints-v1/${CKPT_SPEC}" ;;
  esac
elif [ -s ml/runs/actions/pixelmae.pt ]; then
  # What a `resume: !run-<n>` dispatch leaves behind: the Train phase reloads
  # the named checkpoint and writes it here (the same file the `publishembed`
  # mode reads for the same reason). Pair `resume` with a `steps` equal to
  # that checkpoint's own step count so nothing trains — E-044 §3.
  CKPT=ml/runs/actions/pixelmae.pt
  echo "codec: taking the checkpoint the Train phase left at $CKPT"
else
  # NO BACKTICKS in a double-quoted echo: bash would run the contents as a
  # command substitution and splice its output into the message (root
  # CLAUDE.md §6 — third occurrence of that family of bug in this repo).
  echo "::error::no codec checkpoint. This mode reads one for its chan names"
  echo "::error::and its args (holdout_years, holdout_lon) — never for its"
  echo "::error::weights — and it will not guess either. Either dispatch"
  echo "::error::with resume: !run-<n> (the Train phase then writes"
  echo "::error::ml/runs/actions/pixelmae.pt), or name it in the window:"
  echo "::error::lim:K=...,ckpt:<release asset or path on this box>."
  exit 1
fi
[ -s "$CKPT" ] || { echo "::error::codec checkpoint $CKPT is missing/empty"; exit 1; }
echo "codec: $CKPT ($(stat -c%s "$CKPT") bytes)"

# ---- the HORIZON, in DAYS, derived from the tensor ------------------------
# Identical reasoning to sroll_run.sh: --horizon is a count of AXIS STEPS and
# this script's job is not to rescale a flag (ml/CLAUDE.md §5.24) but to stop
# it being implicit. The day-matched step count is computed from the same
# TimeAxis the scoring will build and passed explicitly — 12 at monthly, 73 at
# pentad — so the launch log states what will be rolled.
AXIS="$(python - "$TENSOR" <<'PY_AXIS'
import sys
sys.path.insert(0, "ml")
import numpy as np
from rollout_spatial import TimeAxis
ax = TimeAxis(np.load(sys.argv[1], allow_pickle=False))
h = ax.steps_for_months(12)
print(f"{ax.cadence} {h} {ax.span_days(h):g} {ax.step_days:g}")
PY_AXIS
)" || { echo "::error::could not read the tensor's time axis — the horizon"
        echo "::error::below would be a guess about what a step means"; exit 1; }
read -r CADENCE H_DAYMATCH H_SPAN_D STEP_D <<< "$(echo "$AXIS" | tail -1)"
[ -n "$H_DAYMATCH" ] || { echo "::error::empty time axis read: '$AXIS'"; exit 1; }
HORIZON="${HORIZON_TOK:-$H_DAYMATCH}"
echo "axis: $CADENCE · one step = ${STEP_D} d · 12 months = ${H_DAYMATCH} steps = ${H_SPAN_D} d"
if [ -n "$HORIZON_TOK" ] && [ "$HORIZON_TOK" != "$H_DAYMATCH" ]; then
  echo "::warning::horizon:$HORIZON_TOK is NOT the day-matched $H_DAYMATCH steps (${H_SPAN_D} d) — horizon_auc will not be comparable with the archive; only horizon_auc_daymatched will"
fi
echo "horizon: $HORIZON steps"
if [ -n "$STARTS_TOK" ]; then
  echo "starts: $STARTS_TOK per holdout year"
else
  echo "starts: 3 per holdout year (ml/lim_baseline.py's default — the same count E-044 recommends for a pentad roll, so the LIM and the head are scored from the same rows)"
fi

# ---- X: the sidecar if the tensor has one, else one extraction -----------
# sroll_run.sh's fork, verbatim in behaviour and keyed by the same names, so a
# box warm from a head roll does not re-extract 10.9 GiB for the baseline.
SIDECAR="ml/cache/${STEM}_X.npy"
if [ -s "$SIDECAR" ]; then
  XPATH="$SIDECAR"
  echo "X: sidecar $XPATH — no extraction needed"
else
  case "$STEM" in
    family3*) XPATH=ml/cache/family3_X.npy
              XSMALL=ml/cache/f3_dec_small.npz ;;
    *)        XPATH="ml/cache/${STEM}_X.npy"
              XSMALL="ml/cache/${STEM}_dec_small.npz" ;;
  esac
  python -u scripts/dectrain_extract.py --tensor "$TENSOR" \
    --out-x "$XPATH" --out-small "$XSMALL"
fi
[ -s "$XPATH" ] || { echo "::error::no X memmap at $XPATH"; exit 1; }

OUT="ml/runs/actions/rollout_spatial.json"
# BACKGROUNDED behind the publisher loop, exactly as sroll_run.sh does it:
# the fit is a multi-minute sgemm over an 8 GB matrix and the rolls are
# minutes more, and Actions will not serve a running job's log.
# ml/lim_baseline.py appends rollout_spatial's OWN `sroll` progress records to
# metrics.jsonl, so the status page needs no new record family (§0d).
rm -f ml/runs/actions/metrics.jsonl
python -u ml/lim_baseline.py \
  --x "$XPATH" --npz-small "$TENSOR" --ckpt "$CKPT" \
  --K "$KLIST" --horizon "$HORIZON" \
  ${STARTS_TOK:+--starts $STARTS_TOK} \
  ${SCOPE_TOK:+--scope-fit $SCOPE_TOK} \
  ${MAXGB_TOK:+--max-state-gb $MAXGB_TOK} \
  --out "$OUT" &
EVAL_PID=$!
TICK=0
while kill -0 $EVAL_PID 2>/dev/null; do
  sleep 30
  TICK=$((TICK + 1))
  if [ $((TICK % 5)) -eq 0 ]; then
    bash scripts/publish_live_metrics.sh "ml-live-${GITHUB_RUN_NUMBER}" || true
  fi
done
wait $EVAL_PID || { echo "::error::lim_baseline.py failed"; exit 1; }
bash scripts/publish_live_metrics.sh "ml-live-${GITHUB_RUN_NUMBER}" || true

# ASSERT THE EFFECT (ml/CLAUDE.md §0.2), not this step's colour: the file
# exists, parses, reached its FINAL write, scored the horizon this script
# chose, carries the cross-cadence-comparable AUC, and says in its own
# artefact that it is a LIM and that its gate was not taken. `$HORIZON` and
# `$KLIST` are passed as argv rather than read back out of the file they are
# meant to be checking.
python - "$OUT" "$HORIZON" "$KLIST" <<'PYEOF'
import json, math, os, sys
p, want_h, want_k = sys.argv[1], int(sys.argv[2]), sys.argv[3]
d = json.load(open(p))
assert "in_progress" not in d, (
    "rollout_spatial.json still carries in_progress — the fit died mid-run "
    "and left a partial artefact that parses and is not what was asked for "
    "(ml/CLAUDE.md 5.25): " + repr(d.get("in_progress")))
assert d.get("horizon") == want_h, (
    f"scored horizon {d.get('horizon')} where this script asked for {want_h}")
m = d.get("model") or {}
assert m.get("model") == "lim", f"this is not a LIM artefact: {m.get('model')}"
g = d.get("gate") or {}
assert g.get("skipped") is True and g.get("certified") is False, g
assert g.get("reason"), "gate not taken with no recorded reason"
heads = d.get("heads") or {}
assert heads, "no LIM entry at all"
asked = [int(x) for x in want_k.split(",")]
scored = [k for k, e in heads.items() if not e["meta"].get("unscored")]
assert scored, (
    "every K produced an undamped propagator and none was scored — this run "
    "answers nothing")
seen_k = sorted({e["meta"]["K_requested"] for e in heads.values()})
missing = [k for k in asked if k not in seen_k]
assert not missing, (
    f"K values asked for but absent from the artefact: {missing}. A K that "
    f"clamps onto one already scored is reported as skipped in the log; a K "
    f"that vanishes is a bug.")
# NEVER NaN (5.22). The writer refuses too; this is the reader's own check.
def walk(o, path="results"):
    if isinstance(o, dict):
        for k, v in o.items():
            yield from walk(v, f"{path}.{k}")
    elif isinstance(o, list):
        for i, v in enumerate(o):
            yield from walk(v, f"{path}[{i}]")
    elif isinstance(o, float) and not math.isfinite(o):
        yield path, o
bad = list(walk(d))
assert not bad, f"non-finite values in the artefact: {bad[:5]}"
print(f"LIM: {len(heads)} entr{'y' if len(heads) == 1 else 'ies'} "
      f"({len(scored)} scored) · {m['train_bins']:,} training bins x "
      f"{m['state_dim']:,} state dims · channels "
      f"{','.join(m['channels'])} · scope {m['scope_fit']} · tau "
      f"{m['tau_days']} d")
for lab, e in heads.items():
    if e["meta"].get("unscored"):
        print(f"  {lab}: NOT SCORED — {e['meta']['unscored_reason'][:120]}")
        continue
    ei = e["meta"]["eigen"]
    # The corridor is the headline scope and is scored under BOTH --scope-fit
    # modes, so its absence is a bug in either.
    cor = e.get("corridor")
    assert cor and cor.get("chan_skill"), f"{lab}: no corridor skill rows"
    dm = cor.get("horizon_auc_daymatched")
    assert dm is not None, (
        f"{lab}: no corridor horizon_auc_daymatched — the roll reached none "
        f"of the twelve monthly-equivalent leads, so nothing it produced can "
        f"be compared with an archived corridor AUC")
    win = (e.get("window") or {}).get("horizon_auc")
    print(f"  {lab}: corridor AUC {cor['horizon_auc']:+.3f} "
          f"(day-matched {dm:+.3f}; window "
          f"{'n/a' if win is None else format(win, '+.3f')}) · "
          f"|lambda|max {ei['spectral_radius']:.4f} · e-folding "
          f"{ei['leading_efolding_days']} d")
if m.get("scopes_unavailable"):
    print("  UNAVAILABLE scopes: " + ", ".join(m["scopes_unavailable"]))
print(f"rollout_spatial.json OK ({os.path.getsize(p):,} bytes)")
PYEOF
echo "lim: wrote $OUT — archive_probes.py will publish it"

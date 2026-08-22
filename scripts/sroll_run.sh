#!/usr/bin/env bash
# E-022 / E-044 spatial rollout eval — the body of the `sroll:` window token.
#
# Spec: sroll:<tag,tag,...>[,ckpt:<asset|path>][,horizon:N][,starts:N]
#                          [,long:N][,future:N][,longstart:L[+L...]][,dumproll]
#   e.g. sroll:e017_u1_s0,e017_u1_s1,e022s9_u1_s0,e022s13_u1_s0
#        sroll:e044x144zn_u1_s0,ckpt:run-415.pt,starts:3
#
# THE HORIZON, THE STARTS AND THE JOB TIMEOUT (2026-08-20, E-044). Everything
# in this block is a DISPATCH decision; it lives here because a dispatcher
# copying the sroll pattern reads this file and nothing else.
#
#   horizon:N  rolled horizon in AXIS STEPS. OMIT IT and this script computes
#              the DAY-MATCHED value from the tensor itself
#              (`TimeAxis.steps_for_months(12)`) and passes it explicitly —
#              12 at monthly, which is the argparse default and every
#              archived roll, and 73 (365.0 d) at pentad. Until today the
#              script passed no --horizon at all, so a pentad roll silently
#              took the literal 12 = 60 DAYS and would have been reported
#              beside the monthly archive's 365 (ml/recipes/
#              xl144-zn-pentad-nolonhold.json `_description`, "THE HORIZON
#              DECISION"). Pass it only to score something OTHER than the
#              archive's twelve months; rollout_spatial.py then warns, in the
#              artefact's log, that the resulting `horizon_auc` is not
#              comparable with anything published.
#   starts:N   staggered starts per holdout year: every k-th of the year's
#              list, k = len(list)//N, first N. OMITTED = all of them, which
#              is the monthly protocol's 12 and pentad's 73. E-044 RECOMMENDS
#              starts:3 at pentad — stride 24 pentads, phases near
#              1 Jan / 1 May / 1 Sep, the smallest count giving more than one
#              seasonal phase. Not defaulted here on purpose: it is a COST
#              decision, and 73 starts is a 34.6x monthly scored-step count.
#   longstart:L[+L...]  context END(s) for the long hindcast, `+`-separated
#              because `,` already splits tokens: `longstart:2004-12+2014-12`.
#              OMITTED = the roll's own single default (2004-12), which is
#              every archived hindcast. The FIRST label lands in the artefact's
#              `long` block exactly as it always has; the rest land in the NEW
#              `long_multi` block. This is the CALENDAR-vs-CONTEXT phase
#              discriminator (2026-08-22): every head's unforced future roll
#              mode-locks to the calendar, and one context end cannot tell a
#              model that replays the calendar from one whose phase its own
#              state selects. Each extra label costs a full `--long-months`
#              roll — 240 steps at monthly, i.e. ~1 h/head at 15.06 s/step —
#              so five extra ends are ~5 h per head. SIZE THE TIMEOUT FOR IT.
#   long:N     ) OMIT BOTH unless the dispatch deliberately overrides them.
#   future:N   ) rollout_spatial.py's defaults are ALREADY day-correct since
#              e9f3d8d: `--long-months`/`--future-months` default to -1, which
#              resolves to `round(20 * ax.steps_per_year)` — exactly the
#              historical 240 at monthly and 1,461 at pentad, i.e. 20 YEARS
#              of the tensor's own axis at either cadence. Passing 240
#              explicitly would silently become 3.3 years at pentad, so the
#              trustworthy thing here is to pass NOTHING. The tokens exist
#              only so a shortened hindcast is possible and VISIBLE.
#
#   job_timeout. #417's monthly sroll rolled three heads in 700 min. One
#   PENTAD head at horizon:73,starts:3 is 3,363 roll steps against a monthly
#   head's 714 — 4.71x — which at the measured monthly rate of ~15.1 s/step is
#   ~14.1 h. So: job_timeout >= 1000 (minutes) for ONE pentad head, >= 1900
#   for a seed pair. UNVERIFIED: every one of those hours is arithmetic over
#   the MONTHLY per-step rate; nothing has ever rolled the pentad tensor. The
#   first progress record in ml-live-<n> settles it in minutes — read it
#   before letting a 14-hour job run.
#
# Pulls each published stage-2 head and rolls it over ALL window ocean
# pixels with ml/rollout_spatial.py — the only evaluator that can feed a
# stencil head its neighbourhood inputs. On the MONTHLY family-3 tensor the
# e017_u1_s0 head is the VALIDATION GATE (plan §6.5): rollout_spatial refuses
# to score anything unless that head reproduces #217's numbers within ±0.01,
# so every monthly sroll dispatch should include it in the tag list. Writes
# rollout_spatial.json into the run's probe dir, where archive_probes.py
# picks it up into probes-<run>.json on ml-metrics.
#
# NOTHING HERE IS FAMILY-3 ANY MORE (2026-08-19, E-044). Until today this
# script named family 3 four times in literals no input could override — the
# codec asset `f3_anchor41M__pixelmae.pt`, the Z glob
# `Z_*_6c52f0687b_adcbe700fb.npy` (that codec's weight hash and that tensor's
# sha256), a fixed four-chunk fallback download, and
# `--x ml/cache/family3_X.npy --npz-small ml/cache/f3_dec_small.npz` — so a
# `recipe:` naming the pentad tensor would have rolled family 3 with a
# family-3 codec and reported it as a pentad number. Every one of them is now
# DERIVED:
#   * the tensor comes from $TENSOR, which scripts/probes_run.sh already
#     resolves from ${RECIPE_TENSOR:-$IN_TENSOR};
#   * X is the tensor's sidecar `_X.npy` when it has one (family 5) and is
#     otherwise extracted from the npz, ONCE, keyed by the tensor's stem;
#   * `--npz-small` is the TENSOR ITSELF. Its small members are read lazily
#     out of the zip, they are byte-identical to what dectrain_extract.py
#     copies into f3_dec_small.npz, and — the reason for the change —
#     `bin_index` / `pentad_days` / `cadence` / `epoch` are NOT in that
#     script's SMALL list, so the extracted file cannot tell the roll what
#     cadence it is looking at;
#   * the Z cache is keyed by the two hashes the embed cache is actually
#     named for (ml/temporal.py `codec_weight_hash` + `data_fingerprint`),
#     computed here from the checkpoint and the tensor rather than pasted;
#   * the codec is named by a `ckpt:` token, defaulting to the published f3
#     anchor ONLY on the monthly family-3 tensor. On any other tensor there
#     is nothing to derive it from and the script REFUSES, because a wrong
#     codec is the #10/#11 failure: the shape check passes, the Z verify
#     passes against its own re-encode, and the roll is beautiful nonsense.
#
# Lives in a script, not inline, because the Probes `run:` block has a
# 21,000-char dispatch-time expression ceiling (see dectrain_run.sh).
# Fatal on failure, deliberately: this eval IS this mode's job.
set -e
WINDOW="${1:?usage: sroll_run.sh 'sroll:<tag,tag,...>[,ckpt:<asset|path>]'}"

SPEC="${WINDOW#sroll:}"
TAGS=""
CKPT_SPEC=""
HORIZON_TOK=""
STARTS_TOK=""
LONG_TOK=""
FUTURE_TOK=""
DUMP_TOK=""
LS_TOK=""
# A token whose value must be a step count is CHECKED here, where the inputs
# are all it has cost (ml/CLAUDE.md §0.3/§5.16). `horizon:x` would otherwise
# reach argparse as a string, die there, and take the whole dispatch with it
# after the tensor pull and the extraction.
want_int() {                       # want_int <token> <value>
  case "$2" in
    ''|*[!0-9]*) echo "::error::sroll token '$1' needs a whole number of AXIS"
                 echo "::error::STEPS, got '$2'"; exit 1 ;;
  esac
}
# The SHAPE of a date label, checked here where the inputs are all it has cost
# (§0.3); whether the label has a ROW is the axis's question and the roll
# answers it by SKIPPING with a reason. `2004_12` would otherwise reach the
# roll, resolve to None, be skipped, and leave a dispatch that asked for six
# hindcasts quietly producing five.
want_labels() {                    # want_labels <token> <comma list>
  [ -n "$2" ] || { echo "::error::sroll token '$1' is empty"; exit 1; }
  local IFS=,
  for l in $2; do
    case "$l" in
      [0-9][0-9][0-9][0-9]-[0-9][0-9]|[0-9][0-9][0-9][0-9]-[0-9][0-9]-[0-9][0-9]) ;;
      *) echo "::error::sroll token '$1' wants YYYY-MM or YYYY-MM-DD labels"
         echo "::error::joined by '+', got '$l' in '$2'"; exit 1 ;;
    esac
  done
}
for tok in ${SPEC//,/ }; do
  case "$tok" in
    ckpt:*)    CKPT_SPEC="${tok#ckpt:}" ;;
    horizon:*) HORIZON_TOK="${tok#horizon:}"; want_int horizon "$HORIZON_TOK" ;;
    starts:*)  STARTS_TOK="${tok#starts:}";   want_int starts  "$STARTS_TOK" ;;
    # BEFORE `long:*` — the two are distinguishable to the glob (`longstart:`
    # does not start with `long:`), but a reader should not have to prove that,
    # and a future `long*:` would silently swallow this one.
    longstart:*) LS_TOK="${tok#longstart:}"; LS_TOK="${LS_TOK//+/,}"
                 want_labels longstart "$LS_TOK" ;;
    long:*)    LONG_TOK="${tok#long:}";       want_int long    "$LONG_TOK" ;;
    future:*)  FUTURE_TOK="${tok#future:}";   want_int future  "$FUTURE_TOK" ;;
    # A BARE token, and it must be matched BEFORE the `*)` arm below, which
    # reads anything without a colon as a HEAD TAG — an unmatched `dumproll`
    # would be fetched from the release, 404, and (since 2026-08-20) refuse
    # the whole dispatch.
    dumproll)  DUMP_TOK=1 ;;
    *:*)       echo "::error::unknown sroll token '$tok'"; exit 1 ;;
    "")        ;;
    *)         TAGS="$TAGS $tok" ;;
  esac
done
[ -n "$TAGS" ] || { echo "::error::no head tags in '$WINDOW'"; exit 1; }
echo "sroll: tags=$(echo $TAGS | tr ' ' ',')${CKPT_SPEC:+ ckpt=$CKPT_SPEC}"
df -h / | tail -1

TENSOR="${TENSOR:-ml/cache/family3_na025.npz}"
[ -s "$TENSOR" ] || { echo "::error::no tensor at $TENSOR"; exit 1; }
STEM="$(basename "$TENSOR" .npz)"
echo "tensor: $TENSOR ($(stat -c%s "$TENSOR") bytes)"

# ---- the codec -----------------------------------------------------------
if [ -n "$CKPT_SPEC" ]; then
  case "$CKPT_SPEC" in
    */*) CKPT="$CKPT_SPEC" ;;                      # a path on this box
    *)   CKPT="ml/cache/${CKPT_SPEC}"
         [ -s "$CKPT" ] || curl -fsSL -o "$CKPT" \
           "https://github.com/${GITHUB_REPOSITORY}/releases/download/model-checkpoints-v1/${CKPT_SPEC}" ;;
  esac
elif [ "${STEM#family3}" != "$STEM" ]; then
  CKPT=ml/cache/f3_anchor41M__pixelmae.pt
  [ -s "$CKPT" ] || curl -fsSL -o "$CKPT" \
    "https://github.com/${GITHUB_REPOSITORY}/releases/download/model-checkpoints-v1/f3_anchor41M__pixelmae.pt"
else
  echo "::error::tensor '$STEM' is not the monthly family-3 tensor and the"
  echo "::error::dispatch named no codec. The heads were trained on ONE"
  echo "::error::codec's embeddings and there is nothing in the tensor that"
  echo "::error::says which — pass it: sroll:<tags>,ckpt:<release asset or"
  echo "::error::path on this box>. Guessing here is the #10/#11 failure:"
  echo "::error::the shape check, the Z verify and the roll all succeed."
  exit 1
fi
[ -s "$CKPT" ] || { echo "::error::codec checkpoint $CKPT is missing/empty"; exit 1; }
echo "codec: $CKPT ($(stat -c%s "$CKPT") bytes)"

# ---- the HORIZON, in DAYS, derived from the tensor ------------------------
# rollout_spatial.py's --horizon is a count of AXIS STEPS and stays one — it
# is not this script's business to rescale a flag (ml/CLAUDE.md §5.24). It IS
# this script's business not to leave it implicit: an omitted --horizon takes
# argparse's literal 12, which is the archive's twelve months on families 2/3
# and SIXTY DAYS on family 4. So the day-matched step count is computed HERE,
# from the same TimeAxis the roll will build, and passed explicitly. At
# monthly that is the literal 12 and the artefact is bit-identical to every
# archived one; at pentad it is 73 = 365.0 d. Costs one import before any
# extraction or GPU (§0.3) and prints what it will spend the day on.
AXIS="$(python - "$TENSOR" <<'PY_AXIS'
import sys
sys.path.insert(0, "ml")
import numpy as np
from rollout_spatial import TimeAxis
ax = TimeAxis(np.load(sys.argv[1], allow_pickle=False))
h = ax.steps_for_months(12)
print(f"{ax.cadence} {h} {ax.span_days(h):g} {ax.step_days:g} "
      f"{len(ax.bands())}")
PY_AXIS
)" || { echo "::error::could not read the tensor's time axis — the horizon"
        echo "::error::below would be a guess about what a step means"; exit 1; }
read -r CADENCE H_DAYMATCH H_SPAN_D STEP_D N_BANDS <<< "$(echo "$AXIS" | tail -1)"
[ -n "$H_DAYMATCH" ] || { echo "::error::empty time axis read: '$AXIS'"; exit 1; }
HORIZON="${HORIZON_TOK:-$H_DAYMATCH}"
echo "axis: $CADENCE · one step = ${STEP_D} d · 12 months = ${H_DAYMATCH} steps = ${H_SPAN_D} d"
if [ -n "$HORIZON_TOK" ]; then
  echo "horizon: $HORIZON steps (window token horizon:$HORIZON_TOK)"
  [ "$HORIZON_TOK" = "$H_DAYMATCH" ] || \
    echo "::warning::horizon:$HORIZON_TOK is NOT the day-matched $H_DAYMATCH steps (${H_SPAN_D} d) — rollout_spatial.py will say so in the artefact, and horizon_auc will not be comparable with the monthly archive"
else
  echo "horizon: $HORIZON steps = ${H_SPAN_D} d (day-matched to the monthly archive's 12 months; no horizon: token given)"
fi
if [ -n "$STARTS_TOK" ]; then
  echo "starts: $STARTS_TOK per holdout year (window token starts:$STARTS_TOK)"
else
  echo "::warning::no starts: token — EVERY start of every holdout year is scored (12 at monthly, 73 at pentad). At pentad that is ~34.6x the monthly scored-step count; E-044 recommends starts:3"
fi
# `[ ... ] && [ ... ] && echo` would EXIT HERE under `set -e` the moment a
# token was set — a false condition is a non-zero status (ml/CLAUDE.md §7).
if [ -n "$LS_TOK" ]; then
  echo "longstart: $LS_TOK — $(echo "$LS_TOK" | tr ',' '\n' | wc -l) hindcast(s) per head, the first into \`long\` (today's key) and the rest into \`long_multi\`. Each is a full --long-months roll: budget for it."
else
  echo "longstart: not set — one hindcast per head from rollout_spatial.py's own default context end (2004-12), exactly as every archived roll"
fi
if [ -z "$LONG_TOK" ] && [ -z "$FUTURE_TOK" ]; then
  echo "long/future: rollout_spatial.py's own defaults — round(20 * steps_per_year), i.e. 20 years of THIS axis (240 at monthly, 1461 at pentad)"
else
  echo "long/future: OVERRIDDEN by the window — long:${LONG_TOK:-<default>} future:${FUTURE_TOK:-<default>}, in AXIS STEPS"
fi

# ---- X: the sidecar if the tensor has one, else one extraction -----------
SIDECAR="ml/cache/${STEM}_X.npy"
if [ -s "$SIDECAR" ]; then
  XPATH="$SIDECAR"
  echo "X: sidecar $XPATH — no extraction needed"
else
  # family 3 keeps BOTH historical output names. dectrain_extract.py is
  # idempotent on (out-x size, out-small exists), so renaming either of them
  # would make every box with a warm cache re-extract 10.9 GiB.
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

# ---- Z: keyed by (codec weight hash, tensor sha256), computed not pasted --
# Same two functions the trainer and embed_cache_sync.py name the cache with,
# imported rather than reimplemented — two copies of a hash rule are two
# chances to disagree, and the disagreement is silent (ml/temporal.py's own
# docstring for codec_weight_hash).
cat > /tmp/sroll_zkey.py <<'PY_ZKEY'
import sys, torch
sys.path.insert(0, "ml")
from temporal import codec_weight_hash, data_fingerprint
ck = torch.load(sys.argv[1], map_location="cpu", weights_only=False)
print(codec_weight_hash(ck), data_fingerprint(sys.argv[2]))
PY_ZKEY
HASHES="$(python /tmp/sroll_zkey.py "$CKPT" "$TENSOR" | tail -1)"
WHASH="${HASHES%% *}"
DHASH="${HASHES##* }"
[ -n "$WHASH" ] && [ -n "$DHASH" ] && [ "$WHASH" != "$DHASH" ] \
  || { echo "::error::could not key the embed cache (got '$HASHES')"; exit 1; }
echo "embed cache key: codec $WHASH · tensor $DHASH"

ZPATH="$(ls ml/cache/Z_*_${WHASH}_${DHASH}.npy 2>/dev/null | head -1 || true)"
if [ -z "$ZPATH" ]; then
  echo "::warning::no local Z for ${WHASH}/${DHASH} — pulling chunks from embed-cache-v1"
  ZPATH="ml/cache/Z_sroll_${WHASH}_${DHASH}.npy"
  rm -f "$ZPATH"
  N=0
  # publish splits at 1.5 GiB, so the chunk count scales with the tensor (4
  # at monthly family 3, ~11 at pentad). Stop at the first missing suffix
  # rather than at a hard-coded four.
  for a in a b c d; do
    for b in a b c d e f g h i j k l m n o p q r s t u v w x y z; do
      curl -fsSL \
        "https://github.com/${GITHUB_REPOSITORY}/releases/download/embed-cache-v1/Z_${WHASH}_${DHASH}.npy.${a}${b}" \
        >> "$ZPATH" || break 2
      N=$((N + 1))
    done
  done
  [ "$N" -gt 0 ] || { echo "::error::no embed cache published for ${WHASH}/${DHASH} — run the embed pass first"; exit 1; }
  echo "pulled $N chunk(s)"
fi
# ASSERT THE EFFECT: a Z whose file length disagrees with its own .npy header
# maps cleanly and returns real numbers belonging to the wrong rows.
python - "$ZPATH" <<'PYEOF'
import sys
sys.path.insert(0, "ml")
from embed_cache_sync import verify
ok, why = verify(sys.argv[1])
print(f"Z: {sys.argv[1]} — {why}")
if not ok:
    raise SystemExit(f"::error::embed cache rejected: {why}")
PYEOF

mkdir -p ml/runs/heads
HPATHS=""
# Two asset-name conventions live on the release. The original one is
# "<tag>__temporal.pt" — a full training checkpoint (weights + Adam moments
# + args), written by snapshot_head.sh. From 2026-08-16 a head that big no
# longer fits: a 205-217M head with moments is 2.49-2.61 GB and GitHub
# rejects any release asset over ~2 GiB with HTTP 422. Those heads are
# published weights-only by scripts/publish_head_weights.sh under their own
# name, "head-weights-<arm>.pt" (~845-869 MB). rollout_spatial.py reads both
# shapes. So: try the __temporal.pt name first (every pre-wave-8 head, and
# the e017_u1_s0 gate, still lives there), fall back to the tag verbatim.
# A NAMED HEAD THAT DOES NOT ARRIVE IS A REFUSAL, NOT A WARNING. Measured
# 2026-08-20, run #421 (E-043b roll): the dispatch named the gate and
# `head-weights-e043b-xl144-nolonhold-s0`, the head's fetch 404'd, this loop
# warned and skipped it, `[ -n "$HPATHS" ]` passed because the GATE had
# arrived, and the run went on to roll the gate ALONE — a green,
# gate-passing, fully-archived job that answered nothing. It was caught only
# because someone read `"heads": 1` in the live metrics; the artefact's own
# assertions cannot catch it, because every head that IS present is complete
# and correct. The old behaviour was written so one bad tag could not lose a
# multi-head roll, but that trade is wrong at this cost: a head is named
# because the run is FOR it, and re-dispatching a two-minute refusal is
# cheaper than discovering a void six-hour roll at harvest (ml/CLAUDE.md
# §0.2, §1 — check the precondition where the inputs are all it has cost).
# Why the fetch 404'd is worth knowing, because it will happen again: a
# release asset appears in the API with its final size while `state` is still
# `starter`, and a GET in that window returns Azure `BlobNotFound` which
# GitHub's Fastly edge then caches on the asset PATH for one hour, ignoring
# the signed query string. Poll `state` until `uploaded` BEFORE the first GET,
# from anywhere — the POP that serves the box is poisoned by whoever asked
# first.
MISSING=""
for tag in $TAGS; do
  if curl -fsSL -o "ml/runs/heads/${tag}.pt" \
      "https://github.com/${GITHUB_REPOSITORY}/releases/download/model-checkpoints-v1/${tag}__temporal.pt" \
   || curl -fsSL -o "ml/runs/heads/${tag}.pt" \
      "https://github.com/${GITHUB_REPOSITORY}/releases/download/model-checkpoints-v1/${tag}.pt"; then
    HPATHS="$HPATHS ml/runs/heads/${tag}.pt"
    echo "head ${tag}: fetched ($(stat -c%s "ml/runs/heads/${tag}.pt") bytes)"
  else
    rm -f "ml/runs/heads/${tag}.pt"
    MISSING="$MISSING ${tag}"
  fi
done
if [ -n "$MISSING" ]; then
  echo "::error::named head(s) not fetched from the release:${MISSING}"
  echo "::error::tried both <tag>__temporal.pt and <tag>.pt for each. This is"
  echo "::error::a REFUSAL, not a skip: the run was dispatched FOR these heads"
  echo "::error::and rolling the rest would produce a green job that answers a"
  echo "::error::different question than the one asked. Check the asset name"
  echo "::error::(weights-only heads are published as 'head-weights-<arm>.pt',"
  echo "::error::so the window token must carry that whole stem) and check the"
  echo "::error::asset's 'state' is 'uploaded' rather than 'starter'."
  exit 1
fi
[ -n "$HPATHS" ] || { echo "::error::no heads fetched — nothing to roll"; exit 1; }
echo "heads: $(echo $TAGS | wc -w) named, $(echo $HPATHS | wc -w) fetched"

OUT="ml/runs/actions/rollout_spatial.json"
# ---- the roll-forward sequence dump (window token `dumproll`) -------------
# Chris, 2026-08-22: "Save the roll forward sequence for the held out years
# somewhere (so that we can use it as animation in the UI)". OFF unless the
# token says so, because it is BYTES: ~411 MB per pentad trajectory, ~3.7 GB
# for one head at horizon 73 / starts 3, and a roll nobody animates should
# not carry them.
#
# It writes into a SUBDIRECTORY of the probe bundle dir, and the subdirectory
# is load-bearing twice over: `.github/workflows/ml-train.yml`'s "Upload probe
# results" step names `ml/runs/actions/roll_dump/*` explicitly (its `path:` is
# a LIST, not a directory — rollout_spatial.json itself is not in it, so
# "write it in the bundle dir and the artifact carries it" is false here),
# and scripts/archive_probes.py globs `probe_head*.json` in `ml/runs/actions`
# and its parent — a manifest sitting beside those would be one widened glob
# away from a 3.7 GB push to the ml-metrics BRANCH.
DUMPDIR="ml/runs/actions/roll_dump"
if [ -n "$DUMP_TOK" ]; then
  rm -rf "$DUMPDIR"
  echo "dumproll: roll-forward sequences -> $DUMPDIR (uploaded with the probe artifact; NOT archived to ml-metrics)"
  df -h . | tail -1
else
  echo "dumproll: not set — no roll-forward sequences are written (the roll JSON is unchanged either way)"
fi
# BACKGROUNDED behind a publisher loop, exactly as the training step does.
# This eval is hours long and, until 2026-08-14, published nothing while it
# ran: Actions will not serve a running job's log, so "how far along is it?"
# had no answer from outside for the whole run. rollout_spatial.py appends
# progress records (head i/N, phase, steps done, ETA) to metrics.jsonl; this
# loop pushes them to ml-live-<n> every ~2.5 minutes, which the status page
# and any curl can read. Best-effort at the CALLER (`|| true`), because a
# telemetry push must never take down the eval it is reporting on.
rm -f ml/runs/actions/metrics.jsonl
# --horizon is ALWAYS passed (derived above when no token names it); the other
# three appear only when the window carried them, so the roll's own
# cadence-correct defaults are what runs otherwise. Unquoted on purpose: each
# is empty or a checked integer pair.
python -u ml/rollout_spatial.py --x "$XPATH" \
  --npz-small "$TENSOR" --z "$ZPATH" --ckpt "$CKPT" \
  --horizon "$HORIZON" \
  ${STARTS_TOK:+--starts-per-year $STARTS_TOK} \
  ${LS_TOK:+--long-start $LS_TOK} \
  ${LONG_TOK:+--long-months $LONG_TOK} \
  ${FUTURE_TOK:+--future-months $FUTURE_TOK} \
  ${DUMP_TOK:+--dump-roll $DUMPDIR} \
  --heads $HPATHS --out "$OUT" &
EVAL_PID=$!
TICK=0
while kill -0 $EVAL_PID 2>/dev/null; do
  sleep 30
  TICK=$((TICK + 1))
  if [ $((TICK % 5)) -eq 0 ]; then
    bash scripts/publish_live_metrics.sh "ml-live-${GITHUB_RUN_NUMBER}" || true
  fi
done
wait $EVAL_PID || { echo "::error::rollout_spatial.py failed"; exit 1; }
bash scripts/publish_live_metrics.sh "ml-live-${GITHUB_RUN_NUMBER}" || true

# Assert the EFFECT: the file exists, parses, every fetched head carries a
# scored corridor block, the horizon this script chose is the horizon the roll
# actually ran, every head carries the cross-cadence-comparable AUC, and the
# gate is either PASSED or explicitly uncertified with a recorded reason —
# never silently absent. `$HORIZON` is passed as argv[2] rather than read back
# out of the file it is meant to be checking.
python - "$OUT" "$HORIZON" <<'PYEOF'
import json, os, sys
p = sys.argv[1]
want_h = int(sys.argv[2])
d = json.load(open(p))
g = d.get("gate") or {}
cad = d.get("cadence")            # written only when a step is not a month
assert d.get("horizon") == want_h, (
    f"the roll scored horizon {d.get('horizon')} where this script asked for "
    f"{want_h}. A horizon that does not survive the call is exactly the "
    f"silent-60-days failure the horizon: token was added to close.")
if cad is None:
    assert g.get("pass") is True, f"gate did not pass: {g}"
    print("gate: PASSED (monthly axis, #217 reference)")
else:
    # A monthly reference cannot certify this roll and rollout_spatial.py
    # refuses to pretend otherwise. What it MUST do is say so, in the
    # artefact, so the harvest never reads these numbers as certified.
    assert g.get("skipped") is True and g.get("certified") is False, g
    assert g.get("reason"), "gate skipped with no recorded reason"
    print(f"gate: NOT CERTIFIED at {cad['name']} cadence — {g['reason'][:200]}")
    print(f"cadence: {cad['name']} · {cad['step_days']} d/step · "
          f"horizon {cad['horizon_steps']} steps = {cad['horizon_span_days']} d"
          f" · day-matched horizon {cad['horizon_daymatched_steps']} steps"
          f" ({'YES' if cad['horizon_is_daymatched'] else 'NO'})"
          f" · starts/yr {cad['starts_per_year']} of "
          f"{sorted(set(cad['starts_available_per_holdout_year'].values()))}"
          f" · leads {cad['daymatched_leads']}")
    if not cad["horizon_is_daymatched"]:
        print("::warning::this roll is NOT horizon-matched to the monthly "
              "archive; horizon_auc below averages a different set of lead "
              "times and only horizon_auc_daymatched may be compared")
assert d.get("heads"), "no heads scored"
for lab, e in d["heads"].items():
    cs = e.get("corridor", {}).get("chan_skill")
    assert cs, f"{lab}: no corridor skill rows"
    # The comparable number must EXIST. It is a pure function of the rows
    # right there in the file, so its absence means the roll never had the
    # leads — a horizon too short to reach them — and the harvest would
    # silently fall back to `horizon_auc`, which is the lead-sampling
    # artefact this whole change exists to stop being quoted.
    dm = e["corridor"].get("horizon_auc_daymatched")
    assert dm is not None, (
        f"{lab}: no corridor horizon_auc_daymatched — the roll reached none "
        f"of the twelve monthly-equivalent leads, so nothing it produced can "
        f"be compared with an archived corridor AUC")
    print(f"  {lab}: corridor AUC {e['corridor']['horizon_auc']:+.3f} "
          f"(day-matched {dm:+.3f}; window "
          f"{e['window']['horizon_auc']:+.3f})")
print(f"rollout_spatial.json OK ({os.path.getsize(p):,} bytes)")
PYEOF
echo "sroll: wrote $OUT — archive_probes.py will publish it"

# The dump is asserted the same way, and only when it was ASKED FOR: a token
# that silently produced nothing is the failure this repo keeps relearning
# (ml/CLAUDE.md §0.2). Fatal, because `dumproll` is a request for data the
# UI work depends on, and a 14-hour roll that has to be repeated for it is
# far more expensive than a refusal here.
if [ -n "$DUMP_TOK" ]; then
  python - "$DUMPDIR" <<'PYEOF'
import json, os, sys
d = sys.argv[1]
mp = os.path.join(d, "dump_manifest.json")
assert os.path.exists(mp), (
    f"dumproll was set and {mp} does not exist — the roll wrote no "
    f"trajectories at all")
m = json.load(open(mp))
fs = m.get("files") or []
assert fs, "dump_manifest.json lists no files"
tot = 0
for f in fs:
    p = os.path.join(d, f["file"])
    assert os.path.exists(p), f"manifest names {f['file']}, which is not there"
    got = os.path.getsize(p)
    assert got == f["bytes"], (f["file"], got, f["bytes"])
    assert f["n_states"] == f["shape"][0] >= 2, f
    tot += got
assert tot == m["total_bytes"], (tot, m["total_bytes"])
heads = sorted({f["head"] for f in fs})
print(f"dumproll: {len(fs)} trajectories, {tot / 1e9:.2f} GB, heads {heads}, "
      f"codec {m['codec']['weight_hash']} ({m['codec']['file']}), "
      f"{m['n_px']:,} px x d_z {m['d_z']} float16")
for f in fs[:3]:
    print(f"  {f['file']}: {f['n_states']} states {f['dates'][0]}..{f['dates'][1]}")
PYEOF
  du -sh "$DUMPDIR"
fi

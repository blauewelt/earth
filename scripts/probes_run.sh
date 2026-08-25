#!/usr/bin/env bash
# The Probes phase — K-sweep, stage-2 temporal transformer, head probe, and
# the several eval MODES that ride on the `window` input.
#
# WHY THIS IS A FILE AND NOT A `run:` BLOCK. GitHub compiles a run: block as
# ONE expression template with a hard ceiling of 21,000 characters, and it
# does not fail politely: the WHOLE workflow becomes unparseable and every
# dispatch in the repo answers 422 — other sessions' jobs included. That
# happened on 2026-08-17, for ten minutes, because six lines of COMMENT were
# added here. This block then sat at 20,570 of 21,000 — 430 characters, about
# one paragraph, from doing it again. dectrain_run.sh and sroll_run.sh were
# carved out of the same block for the same reason; this is the rest of it.
#
# Moving the body out does not just buy slack, it changes the unit of risk:
# a shell script is syntax-checkable locally (`bash -n`), diffable, and its
# comments cost nothing. The workflow now spends ~40 characters here.
#
# CONTRACT. Everything arrives by environment, because a script cannot read
# ${{ inputs.* }}:
#   WINDOW                 the mode string (recipe: already stripped)
#   TENSOR                 path to the .npz
#   GITHUB_TOKEN           release reads/writes
#   IN_TEMPORAL_STEPS      \ the four dispatch inputs this phase reads; each is
#   IN_TEMPORAL_D_MODEL     ) consulted as ${RECIPE_X:-$IN_X} so a recipe wins
#   IN_TEMPORAL_LAYERS     /  over the dispatch default and nothing else moves
#   IN_HEAD_PROBE          /
#   RECIPE_HEAD_TARGETS    recipe-only (no dispatch input — the list is full
#                          at 25): comma-separated probe_head targets,
#                          default "rapid"
#   GITHUB_RUN_NUMBER      supplied by Actions itself
#
# `set -e` reproduces the default `bash -e {0}` the run: block ran under, so
# the failure semantics are unchanged; the step keeps continue-on-error, so a
# non-zero exit here still does not fail the job.
set -e

# WINDOW is exported by the "Resolve recipe" step through $GITHUB_ENV. This
# step carries `if: always()`, so it ALSO runs when that step failed — and
# then WINDOW is not merely empty, it is unset, and every `case "${WINDOW}"`
# below silently takes the default branch. Empty is legitimate (a bare
# `recipe:<name>` with no tail leaves nothing behind); UNSET is not.
# TENSOR is derived here for the same reason the Train step derives it: the
# step's env: is a YAML expression context, so ${RECIPE_TENSOR} cannot be
# resolved there. A recipe naming family4 must reach THIS phase too, or the
# probes would read a different tensor than the training did.
T="${RECIPE_TENSOR:-$IN_TENSOR}"
[ "$T" = "family2" ] && T=na_pixels
export TENSOR="ml/cache/${T}.npz"   # exported: heredocs read os.environ["TENSOR"]

# ANOMALY gate, moved out of the step's `if:` for the same reason — an `if:`
# reads inputs.anomaly and a recipe cannot reach it, so a recipe that set
# anomaly would have governed the trainer and not whether this phase ran.
if [ "${RECIPE_ANOMALY:-$IN_ANOMALY}" != "true" ]; then
  echo "probes: anomaly is not true — this phase is anomaly-space only, skipping"
  exit 0
fi

if [ -z "${WINDOW+x}" ]; then
  echo "::error::WINDOW is unset — the 'Resolve recipe' step did not run or "\
       "failed. Refusing to guess the mode: an unset window looks exactly "\
       "like the default one, and this phase would do the wrong thing quietly."
  exit 1
fi

# THE SHAPE THE EMBEDDING CACHE MUST HAVE, for every push and pull below.
#
# WHY THE SIDECAR HAS TO CARRY THIS. The publish loop further down ships
# whatever /opt/earth-cache/Z_*.npy holds, and it cannot see how that file was
# made. Run #462 made it with --time-stride: 8.72 GB of one bin in two,
# written under the name every unstrided run pulls, and published from here.
# The tail of the previous 16.7 GB publish stayed on the release beside it, so
# what came back down was a chimera and `verify()` refused it — every run
# since paid a four-hour rebuild. ml/temporal.py now refuses to write a
# strided Z under the shared name at all; this is the second line of defence,
# and it is the one that also catches a Z that arrived by some other route.
#
# T comes from the tensor's OWN .npy header — ~128 bytes, whether X is inside
# the .npz (families 2-4) or beside it as family 5's 165.6 GB sidecar. Never
# `np.load(...)["X"].shape`, which would decompress 33 GB to learn one integer.
#
# The fallback is `auto`, NOT "skip the check": embed_cache_sync reads T from
# --data with the same function when it is handed that word. So a broken
# helper here costs a log line, never the guard.
#
# A FUNCTION, called from the two branches that touch the cache, rather than a
# line at the top: the modes above (disktriage, rolleval, project…) neither
# push nor pull, and paying a torch import — or warning about a tensor a
# triage run never needed — in all of them would be noise in the one place a
# reader goes looking for signal.
embed_expect_t() {
  EXPECT_T="$(python -u ml/embed_cache_sync.py tensor-t --data "$TENSOR" \
              2>/dev/null | tail -1)"
  case "$EXPECT_T" in
    ''|*[!0-9]*)
      echo "::warning::could not read T from ${TENSOR} here — passing"\
           "--expect-t auto, which reads it inside embed_cache_sync"
      EXPECT_T=auto;;
    *)
      echo "embed cache shape gate: the tensor has T=${EXPECT_T} bins; a Z"\
           "whose header disagrees is refused at the push and discarded at"\
           "the pull";;
  esac
}

# ROLLOUT EVALUATION MODE — window carries `rolleval:<tag>,<tag>,…`
# (the 25-input cap again). Fetches published temporal heads from
# the model-checkpoints release (public, curl, no gh/node — the
# boxes have neither) and scores them with ml/rollout.py: the model
# fed its OWN predictions month after month, channel-space skill at
# each horizon against persistence/damped/climatology, and the AMOC
# probe read from ROLLED section embeddings. This is the evaluation
# E-010 lacked: its z-ratio was teacher-forced horizon-1, which is
# the regime an unroll objective de-emphasises by design. Trains
# nothing; `exit 0` skips the training-run ladder below.
case "${WINDOW}" in
  disktriage*)
    # DISK TRIAGE (2026-08-11): the box went 50/50 mid-queue and the
    # failure mode was a treadmill — hygiene freed the published Z
    # (correct: re-pullable), but the disk stayed full from
    # something else, so the pull refused and every run paid an
    # 80-minute in-RAM rebuild it could not persist. This mode
    # answers "what is actually eating the disk" from a normal job,
    # because the box has no SSH path from the session sandbox.
    echo "== df =="; df -h / /opt/earth-cache 2>/dev/null || df -h /
    echo "== du / (level 1) =="
    du -x -BM --max-depth=1 / 2>/dev/null | sort -rn | head -15
    echo "== du /opt (level 2) =="
    du -x -BM --max-depth=2 /opt 2>/dev/null | sort -rn | head -25
    echo "== du workspace =="
    du -x -BM --max-depth=2 "$GITHUB_WORKSPACE/.." 2>/dev/null | sort -rn | head -20
    echo "== du /root =="
    du -x -BM --max-depth=2 /root 2>/dev/null | sort -rn | head -20
    echo "== biggest files =="
    find / -xdev -type f -size +200M -printf "%s\t%p\n" 2>/dev/null | sort -rn | head -25
    case "${WINDOW}" in
      "disktriage:rm "*)
        # Targeted removal, allowlisted roots only — a typo in a
        # window input must not be able to delete the OS.
        P="${WINDOW}"; P="${P#disktriage:rm }"
        for x in $P; do
          case "$x" in
            /opt/earth-cache/*|/root/.cache/*|/tmp/*|/opt/runner/_work/*|/opt/runner/_diag/*)
              du -sh "$x" 2>/dev/null
              rm -rf "$x" && echo "removed $x";;
            *) echo "REFUSED (not an allowlisted root): $x";;
          esac
        done
        echo "== df after =="; df -h /
        ;;
    esac
    exit 0
    ;;
  publishtensor)
    # One-off: push this box's BUILT tensor to the data-cache
    # release, keyed by its sha256 prefix. The sources are durable;
    # the built tensor was not, and every E-010 result is on a
    # build (`adcbe700`) that exists on exactly one rented disk.
    python -u scripts/publish_tensor.py --data "$TENSOR" \
      || echo "::warning::tensor publish failed — it remains single-copy"
    exit 0
    ;;
  publishembed)
    # The publishtensor of the OTHER single-copy artefact. Until now
    # `embed_cache_sync.py push` was reachable ONLY from inside the
    # stage-2 block below, i.e. only on a dispatch that trains — so
    # the one way to publish an existing Z was to spend 1.5-2 h and
    # an ~85 GB RAM peak re-entering temporal.py for a cache that was
    # already sitting on the disk. E-044's 16.24 GiB pentad Z (8.5 h
    # of a 4090, #423, never published) is exactly that case, and
    # while it lives on one rented disk the seed-1 arm and the
    # pentad sroll are pinned to one box (E-044 §4).
    #
    # push() derives the cache's own name from the codec at
    # ml/runs/actions/pixelmae.pt and this tensor (cache_name ->
    # codec_weight_hash + data_fingerprint), verifies the .npy
    # header length and dtype before uploading, no-ops when every
    # chunk is already on the release with the right total size, and
    # refuses when the disk cannot hold one 1.5 GiB part. So the
    # dispatch decides only WHICH codec is on disk — which is what
    # `steps` + `resume` decide, and why they must be the pair that
    # trains nothing (E-044 §3).
    #
    # `|| echo`, at the CALLER, per ml/CLAUDE.md §7: a failed push
    # exits 1 (correctly, since 2026-08-10) and `bash -e` would
    # otherwise take the step down. Which also means THIS STEP GOES
    # GREEN ON A FAILED PUBLISH — read the durable/no-op line below,
    # never the step's colour (§0.2).
    echo "publishembed: pushing this box's embed cache for the codec at"
    echo "  ml/runs/actions/pixelmae.pt against ${TENSOR}."
    echo "  VERIFY THE EFFECT, not this step's colour: the next lines must end"
    echo "  in 'is now durable' (uploaded) or 'already published and complete'"
    echo "  (no-op). Anything else — 'no embed cache on disk', 'refusing',"
    echo "  'chunk ... failed' — means the release did NOT gain the cache."
    ls -l /opt/earth-cache/Z_*.npy 2>/dev/null || echo "  (no /opt/earth-cache/Z_*.npy on this box)"
    embed_expect_t
    python -u ml/embed_cache_sync.py push --run actions --data "$TENSOR" \
      --expect-t "$EXPECT_T" \
      || echo "::warning::embed cache publish failed — the Z remains single-copy on this box's disk"
    exit 0
    ;;
  rolleval:*)
    SPEC="${WINDOW}"; SPEC="${SPEC#rolleval:}"
    mkdir -p ml/runs/heads
    HPATHS=""
    for tag in ${SPEC//,/ }; do
      if curl -fsSL -o "ml/runs/heads/${tag}.pt" \
          "https://github.com/${GITHUB_REPOSITORY}/releases/download/model-checkpoints-v1/${tag}__temporal.pt"; then
        HPATHS="$HPATHS ml/runs/heads/${tag}.pt"
      else
        echo "::warning::head ${tag} not on the release — skipped"
      fi
    done
    if [ -z "$HPATHS" ]; then
      echo "::error::no heads could be fetched — nothing to evaluate"
      exit 1
    fi
    python -u ml/rollout.py --run actions --data "$TENSOR" \
      --horizon 12 --pixels 600 --temporal $HPATHS \
      || echo "::warning::rollout eval failed"
    bash scripts/publish_live_metrics.sh "ml-live-${GITHUB_RUN_NUMBER}" || true
    exit 0
    ;;
  dectrain:*)
    # DECODER RETRAIN MODE (E-019b1) — trains a multi-output
    # decoder against the frozen run-62 Z and publishes weights +
    # audit JSON to model-checkpoints-v1. Body lives in the script
    # because this run: block has a 21,000-char dispatch-time
    # ceiling ("Exceeded max expression length" — the whole
    # workflow 422s, same all-or-nothing as the 26th input).
    # Fatal on failure, deliberately: training IS this mode's job.
    bash scripts/dectrain_run.sh "${WINDOW}"
    exit 0
    ;;
  project:*)
    # E-021 "the 20-year fan" — roll the published heads 240 months
    # past the record (and from a 2004 context, where RAPID truth
    # exists) with residual-scaled ensembles. Writes
    # project_amoc.json for archive_probes.py. Body in the script:
    # this run: block has a 21,000-char dispatch ceiling.
    bash scripts/project_run.sh "${WINDOW}"
    exit 0
    ;;
  headpub*)
    bash scripts/publish_head_weights.sh "${WINDOW}"
    exit 0
    ;;
  sroll:*)
    # E-022 spatial rollout eval — full-window roll of stencil
    # heads against the e017 baseline, gated on reproducing #217.
    # Body in scripts/sroll_run.sh (21,000-char ceiling here).
    bash scripts/sroll_run.sh "${WINDOW}"
    exit 0
    ;;
esac
# GUARDED, like every probe below it. These are independent
# measurements: one failing is a missing number, not a reason to
# delete the others. Under `bash -e` a bare call does exactly that,
# and it is how #131 lost probe_kfold AND dip_check to an unrelated
# upload that ran out of disk. Only the precheck below stays fatal,
# because "this checkpoint cannot do what you asked" is a definite
# answer (ml/CLAUDE.md §5.17).
python -u ml/probe_sequence.py --run actions --anomaly --data "$TENSOR" \
  || echo "::warning::probe_sequence failed — continuing to stage 2"
# A joint: run has ALREADY trained a temporal model inside
# train_joint.py, and "joint:4000" is not an integer — passing it to
# temporal.py --steps aborts the step, and with it every probe after
# it. That is why #94 finished a clean joint run and produced no
# k-fold number at all. Skip stage 2 here on joint runs; the ladder
# below still scores the resulting codec, which is the whole point.
case "${RECIPE_TEMPORAL_STEPS:-$IN_TEMPORAL_STEPS}" in joint:*) SKIP_S2=1;; *) SKIP_S2=0;; esac
if [ "${RECIPE_TEMPORAL_STEPS:-$IN_TEMPORAL_STEPS}" != "0" ] && [ "$SKIP_S2" = "0" ]; then
  # `window` doubles as a stage-2 modifier here, same reuse as on a
  # joint: run — "unroll:4" turns on the autoregressive unroll that
  # matches training to how rollout.py actually scores the model.
  # Another input is not available: workflow_dispatch caps at 25 and
  # this file is exactly at the ceiling.
  UNROLL=1
  case "${WINDOW}" in unroll:*) UNROLL="${WINDOW}"; UNROLL="${UNROLL#unroll:}";; esac
  # SEED, same overloaded input, comma-separated so it composes:
  # "unroll:4,seed:2". No room for a 26th input, and a 26th breaks
  # every dispatch. WHY seeds exist here: ml/CLAUDE.md §3
  # (replicates, not arms) and EXPERIMENTS.md E-009/E-010 — a
  # four-arm sweep at one seed each could only have detected an
  # effect the instrument itself could not hold still.
  SEED=0
  case "${WINDOW}" in
    *seed:*) SEED="${WINDOW}"; SEED="${SEED##*seed:}"; SEED="${SEED%%,*}";;
  esac
  # Strip any trailing ,seed:N from the unroll value, so
  # "unroll:4,seed:2" yields UNROLL=4 rather than "4,seed:2" — which
  # argparse would reject and take the whole probe ladder with it.
  UNROLL="${UNROLL%%,*}"
  # direct:3-6-12 — DIRECT multi-horizon heads (E-014): extra linear
  # readouts predicting z at t+h in ONE forward, the direct-vs-
  # iterated alternative to rolling predictions forward. DASHES in
  # the horizon list because commas separate window fields; turned
  # back into the comma list temporal.py expects. Same anywhere-
  # match + first-comma strip as seed:. Composes with everything:
  # "unroll:1,seed:0,direct:3-6-12,sched:expdecay ..." — and the
  # sched:-goes-LAST convention still holds, since sched: swallows
  # the rest of the string verbatim.
  DIRECT=""
  case "${WINDOW}" in
    *direct:*) DIRECT="${WINDOW}"
               DIRECT="${DIRECT##*direct:}"; DIRECT="${DIRECT%%,*}"
               DIRECT="--direct ${DIRECT//-/,}";;
  esac
  # uprobs:0.5-0.25-0.125-0.125 — SAMPLED unroll depth (E-016):
  # per-step draw of U_t in 1..unroll from these probabilities.
  # Dashes because commas separate window fields; same anywhere-
  # match + first-comma strip as seed:/direct:; sched: stays LAST.
  UPROBS=""
  case "${WINDOW}" in
    *uprobs:*) UPROBS="${WINDOW}"
               UPROBS="${UPROBS##*uprobs:}"; UPROBS="${UPROBS%%,*}"
               UPROBS="--unroll-probs ${UPROBS//-/,}";;
  esac
  # stencil:9 — E-022 spatial coupling: the neighbourhood input.
  # Same anywhere-match + first-comma strip as seed:.
  STENCIL=1
  case "${WINDOW}" in
    *stencil:*) STENCIL="${WINDOW}"; STENCIL="${STENCIL##*stencil:}"; STENCIL="${STENCIL%%,*}";;
  esac
  # ring:222 — E-023: put the stencil's non-centre slots on a circle
  # of that radius in KM instead of the touching-neighbour table.
  RING=0
  case "${WINDOW}" in
    *ring:*) RING="${WINDOW}"; RING="${RING##*ring:}"; RING="${RING%%,*}"
             RING="${RING//-/,}";;
  esac
  echo "stage 2: unroll=$UNROLL seed=$SEED stencil=$STENCIL ring=$RING ${DIRECT:-} ${UPROBS:-}"
  # `window` also carries resume2:<tag> — CONTINUE a stage-2 head
  # rather than train a fresh one. Packed into an existing input
  # because workflow_dispatch caps at 25 and this file is at the
  # ceiling. temporal.py refuses if the tag is absent or already
  # past --steps, so a typo cannot silently become a fresh run.
  R2=""; LR2=""; MODE2=""; SCHED=""
  # sched:<name> — the LR schedule, in the same overloaded input as
  # everything else, because workflow_dispatch caps at 25 and this
  # file is at the ceiling. cosine bakes the total step count into
  # the rate (which is what makes a resumed run read lr=0.0 and what
  # forces every budget to be its own experiment); wsd holds a
  # horizon-free stable phase and then cools down, so a run can be
  # extended AND still converge.
  case "${WINDOW}" in
    *sched:*)
             # ANYWHERE-match, so the schedule composes with the
             # other window fields: "unroll:4,seed:2,sched:expdecay
             # --lr-cooldown-frac 0" carries all three. The prefix
             # match made sched: mutually exclusive with unroll: and
             # seed:, which E-012 was the first to need. CONVENTION:
             # sched: goes LAST — everything after it (spaces and
             # flags included) is handed to temporal.py verbatim.
             SCHED="${WINDOW}"
             SCHED="--lr-schedule ${SCHED##*sched:}"
             echo "stage-2 schedule: $SCHED";; esac
  case "${WINDOW}" in
    resume2:*) T2="${WINDOW}"; T2="${T2#resume2:}"
               T2="${T2%%,*}"  # tag ends at 1st comma (8d5e711)
               # @<lr> suffix = the continuation's PEAK learning rate
               case "$T2" in *@*) LR2="${T2#*@}"; T2="${T2%@*}";; esac
               # Pull it from the release if this box does not hold it.
               if [ ! -f "/opt/earth-cache/ckpt/${T2}.pt" ]; then
                 curl -fsSL --max-time 900 -o "/opt/earth-cache/ckpt/${T2}.pt.part" \
                   "https://github.com/${GITHUB_REPOSITORY}/releases/download/model-checkpoints-v1/${T2}.pt" \
                   && mv "/opt/earth-cache/ckpt/${T2}.pt.part" "/opt/earth-cache/ckpt/${T2}.pt" \
                   && echo "seeded stage-2 head ${T2}.pt from the release" \
                   || echo "::warning::could not seed ${T2}.pt — temporal.py will refuse, loudly"
               fi
               R2="--resume-temporal ${T2}"; MODE2=resume
               [ -n "$LR2" ] && R2="$R2 --lr $LR2"
               echo "stage-2 continuation: $R2";;
    # warm2:<tag>[@lr] — WARM RESTART: the WEIGHTS only, then a
    # fresh cosine for temporal_steps MORE steps. A separate mode
    # from resume2 because they are separate experiments, and
    # because no head published before 2026-08-10 carries optimiser
    # state at all: every one is {args, model}, so a continuation
    # from them is impossible and asking for one earns a refusal.
    warm2:*)   T2="${WINDOW}"; T2="${T2#warm2:}"
               T2="${T2%%,*}"
               case "$T2" in *@*) LR2="${T2#*@}"; T2="${T2%@*}";; esac
               if [ ! -f "/opt/earth-cache/ckpt/${T2}.pt" ]; then
                 curl -fsSL --max-time 900 -o "/opt/earth-cache/ckpt/${T2}.pt.part" \
                   "https://github.com/${GITHUB_REPOSITORY}/releases/download/model-checkpoints-v1/${T2}.pt" \
                   && mv "/opt/earth-cache/ckpt/${T2}.pt.part" "/opt/earth-cache/ckpt/${T2}.pt" \
                   && echo "seeded stage-2 head ${T2}.pt from the release" \
                   || echo "::warning::could not seed ${T2}.pt — temporal.py will refuse, loudly"
               fi
               R2="--init-temporal ${T2}"; MODE2=warm
               [ -n "$LR2" ] && R2="$R2 --lr $LR2"
               echo "stage-2 WARM RESTART: $R2 (temporal_steps is the EXTRA)";; esac
  # CHECK THE CHECKPOINT NOW, NOT AFTER THE EMBEDDING. #119 spent 93
  # minutes rebuilding Z and then exited one second later because
  # this file has no optimiser state — a fact readable the moment it
  # is on disk. The guard was right; its position cost an hour and a
  # half of a rented 4090. A precondition that depends only on the
  # inputs belongs where the inputs are all it has cost you.
  if [ -n "$R2" ]; then
    python -u scripts/precheck_stage2_head.py "$T2" --mode "$MODE2"
  fi
  # PULL THE EMBEDDING BEFORE BUILDING IT. Z is ~95 minutes of a
  # 4090 and is identical for every run that freezes the same codec
  # — #112, #117 and #119 each rebuilt the same tensor. It is now
  # published to the embed-cache-v1 release under the codec's weight
  # hash, so a box that has never seen this codec downloads 5.2 GiB
  # instead of spending an hour and a half. Best-effort: a miss
  # costs GPU time, never correctness, and the script says which
  # branch it took.
  # `|| true`, and A MISS IS THE NORMAL PATH. pull exits 1 when the
  # release has no cache for this codec — which is not a failure, it
  # is the message "build it yourself", printed one line above. Under
  # `bash -e` that honest exit code aborted the entire step: #139
  # spent twenty minutes reaching this line on a box that simply had
  # no local Z, printed "no embed cache published … it will be
  # built", and then died instead of building it.
  #
  # THIRD instance of one pattern in a night: a callee made honest
  # about failure, with the caller left intolerant. The push below
  # was the second. Both call sites are guarded now, and the rule is
  # in ml/CLAUDE.md — best-effort is the CALLER's decision.
  embed_expect_t
  python -u ml/embed_cache_sync.py pull --run actions --data "$TENSOR" \
    --expect-t "$EXPECT_T" \
    || echo "::warning::no embed cache to pull — stage 2 will build it"
  # BACKGROUNDED, with the same 5-minute publisher the Train and
  # joint steps use. Called inline, the publish below only ran
  # AFTER temporal.py returned — 7 hours on a 60,000-step run, so
  # the status page showed nothing for the entire experiment. Same
  # defect as the joint step had, one step further down.
  # --train-lon-hold is a RECIPE-ONLY key (declared in ml-train.yml under
  # "RECIPE-ONLY KEYS"): the 25-input ceiling is full, and a knob that can
  # only be set by naming a recipe is what §1 asks a dispatch to do anyway.
  # "inherit" is today's behaviour to the pixel — the pool follows whatever
  # the frozen codec's own args say — so a dispatch that does not name a
  # recipe carrying this key is bit-identical to one from before it existed.
  # It governs the TRAINING POOL only; the anomaly-transform statistics
  # always follow the codec (see the two-masks comment in ml/temporal.py).
  # WRITTEN AS --train-lon-hold=VALUE, one word. An explicit block such as
  # "-45,-25" in the NEXT argv slot is read by argparse as an option string
  # (it starts with "-" and is not a negative number), and the job dies with
  # "expected one argument" before temporal.py does anything.
  python -u ml/temporal.py --run actions --K 24 \
    --data "$TENSOR" \
    --steps "${RECIPE_TEMPORAL_STEPS:-$IN_TEMPORAL_STEPS}" \
    --unroll "$UNROLL" --seed "$SEED" --stencil "$STENCIL" \
    --ring-km "$RING" \
    $R2 $SCHED $DIRECT $UPROBS \
    "--train-lon-hold=${RECIPE_TRAIN_LON_HOLD:-inherit}" \
    --d-model "${RECIPE_TEMPORAL_D_MODEL:-$IN_TEMPORAL_D_MODEL}" \
    --layers "${RECIPE_TEMPORAL_LAYERS:-$IN_TEMPORAL_LAYERS}" &
  S2_PID=$!
  # THE MARKER IS PER JOB, NOT PER CONTAINER. Its only job is "do not
  # re-upload the same Z every ten minutes inside THIS job". Keyed to a fixed
  # path it quietly meant something much larger — "never push again on this
  # box" — because /tmp persists in the long-lived self-hosted runner
  # container ACROSS JOBS and nothing ever clears it: scripts/disk_hygiene.sh
  # tier 0 removes /opt/runner/_diag/*.log and /opt/runner/_work/_temp/* and
  # touches nothing under /tmp.
  #
  # Measured 2026-08-21, the in-training push had stopped firing altogether.
  # #414 (E-035 xl233 seed-1 roll-forward) ran 14 h of stage 2 on
  # gpu-box-30257785 with this loop demonstrably alive — live metrics
  # published every 5 min, a head snapshot every 30 — and the string "embed
  # cache" appears exactly ONCE in its whole log, at 07:07:29Z, AFTER
  # "saved .../temporal.pt": that is the post-training push at the bottom of
  # this block, never the loop. #396 (E-035 seed-0 roll-forward) ran ~15.5 h
  # on gpu-box-45731106 with the loop alive and its pull succeeding ("embed
  # cache already local and valid") and produced ZERO push lines.
  #
  # The cost is the artefact sitting single-copy for the length of a run:
  # #427's (E-044 pentad stage-2) Z is 17.43 GB and 8.9 h of a 4090 —
  # ~10.14 s/month x 3142 months, from ml-live-423's own progress records —
  # and without this it would not be published until the run's 200,000 steps
  # end. ml/CLAUDE.md §5.20, verbatim: publish a shared artefact when it
  # EXISTS, not when the job ends.
  #
  # ${GITHUB_RUN_ID} is the key, and the fallback is $$ — this shell's pid —
  # rather than the bare path, because the bare path IS the bug: an unset run
  # id would silently restore a container-wide marker and nothing would look
  # wrong. A per-process key can only fail in the harmless direction (one
  # extra push, which embed_cache_sync.py no-ops), and that is the direction
  # a fallback has to fail in.
  EMBED_MARK="/tmp/embed-cache-pushed.${GITHUB_RUN_ID:-$$}"
  # The legacy container-wide marker, left behind on every box that has ever
  # run this step. Nothing reads it any more; remove it so a future reader
  # digging through /tmp is not told a story that stopped being true here.
  rm -f /tmp/embed-cache-pushed
  EMBED_STATE=""   # the last state ANNOUNCED, so a standing condition costs
                   # one line in the log instead of one line every ten minutes
  TICK=0
  while kill -0 $S2_PID 2>/dev/null; do
    sleep 30
    TICK=$((TICK + 1))
    if [ $((TICK % 10)) -eq 0 ]; then
      bash scripts/publish_live_metrics.sh "ml-live-${GITHUB_RUN_NUMBER}" || true
    fi
    # Every ~30 min, push the head OFF THE BOX. temporal.py mirrors
    # it locally every metrics point with optimiser moments, schedule
    # position and RNG — everything a true continuation needs — but
    # that mirror dies with the instance, and a 200,000-step run is a
    # full day of GPU. 7.3 MB an upload turns "the box died, start
    # again" into "resume from the last half hour".
    if [ $((TICK % 60)) -eq 0 ]; then
      bash scripts/snapshot_head.sh "${GITHUB_RUN_NUMBER}" || true
    fi
    # PUBLISH THE EMBEDDING THE MOMENT IT EXISTS, not when training
    # ends. The push used to sit after `wait $S2_PID`, i.e. sixteen
    # to twenty-three hours after the cache was finished — so a
    # second job started in the meantime could not pull it and had
    # to spend the same 95 minutes over again. That is exactly what
    # happened to #120/#121, and it is why "wait for the first run,
    # then start the second" would not have helped: there was
    # nothing to wait FOR until the first run ended.
    # Retry every ~10 min until it actually lands. The marker is
    # written only on a REAL success — an early version marked it done
    # whatever happened, because the script exited 0 on failure — and
    # "real" now means VERIFIED: the full push re-lists the release
    # after uploading and returns 0 only if every chunk is there at its
    # right size. Nothing else stops this loop.
    #
    # AND IT NO LONGER WAITS FOR THE EMBEDDING TO FINISH. Chris,
    # 2026-08-25: "Publishing should happen 'during' the embedding
    # computation... A new job that needs the same embedding can choose
    # to continue the computation (if 32/100 are already complete it
    # will start with chunk 33)." So the branches below are a ladder on
    # what is actually on the disk: `.done` (the pass finished) takes
    # the full push, `.progress` (still running) takes `push --partial`
    # and ships the whole 1.5 GiB chunks that are finished, and neither
    # is a reason to stop looking. This also closes the failure of
    # 2026-08-24, where a single mid-job shot no-op'd against a
    # poisoned key and the artefact then waited for the job to end.
    #
    # AND SAY WHICH BRANCH RAN (ml/CLAUDE.md §4.6/§4.7). The silent skip is
    # what made the bug above cost three runs' logs to find: a log with no
    # "embed cache" line in it is equally consistent with "the loop never
    # reached the push", "there was nothing to push yet" and "another job's
    # marker suppressed it", and only the third was true. Every branch prints
    # now, and $EMBED_STATE keeps a standing condition to one line.
    if [ $((TICK % 20)) -eq 0 ]; then
      if [ -f "$EMBED_MARK" ]; then
        if [ "$EMBED_STATE" != published ]; then
          echo "embed cache push: already published by this job ($EMBED_MARK) — not re-pushing"
          EMBED_STATE=published
        fi
      elif ls /opt/earth-cache/Z_*.npy.done >/dev/null 2>&1; then
        # THE EMBEDDING HAS FINISHED. A full push: it verifies the release
        # after uploading and only then returns 0, and only that writes the
        # marker that stops this loop.
        echo "embed cache push: STARTING in-training at tick ${TICK} (~$((TICK / 2)) min in) — the embedding is COMPLETE (.done) — $(du -h /opt/earth-cache/Z_*.npy 2>/dev/null | tr '\n' ' ')"
        if python -u ml/embed_cache_sync.py push --run actions --data "$TENSOR" \
             --expect-t "$EXPECT_T"; then
          touch "$EMBED_MARK"
          echo "embed cache push: DONE in-training — VERIFIED durable on the release, hours before this job ends"
          EMBED_STATE=published
        else
          echo "::warning::embed cache push: FAILED in-training at tick ${TICK} — no marker written, retrying in ~10 min"
          EMBED_STATE=failed
        fi
      elif ls /opt/earth-cache/Z_*.npy.progress >/dev/null 2>&1; then
        # …AND WHILE IT IS STILL RUNNING, SHIP WHAT IS FINISHED. Chris,
        # 2026-08-25: "Publishing should happen 'during' the embedding
        # computation … A new job that needs the same embedding can choose to
        # continue the computation (if 32/100 are already complete it will
        # start with chunk 33)." The unit is the 1.5 GiB release chunk (~9% of
        # a pentad Z); embed_cache_sync skips the chunks the release already
        # holds at their right size, so a repeat costs one API call, and it
        # writes the manifest AFTER the chunks so the release can only
        # under-claim. It prints every ten minutes on purpose: the silent skip
        # is what made the 2026-08-21 bug cost three runs' logs to find.
        echo "embed cache push: STARTING PARTIAL in-training at tick ${TICK} (~$((TICK / 2)) min in) — the embedding is still running — $(du -h /opt/earth-cache/Z_*.npy* 2>/dev/null | tr '\n' ' ')"
        if python -u ml/embed_cache_sync.py push --partial --run actions \
             --data "$TENSOR" --expect-t "$EXPECT_T"; then
          echo "embed cache push: PARTIAL done at tick ${TICK} — the finished chunks are on the release; no marker, because the cache is not complete yet"
          EMBED_STATE=partial
        else
          echo "::warning::embed cache push: PARTIAL FAILED at tick ${TICK} — retrying in ~10 min"
          EMBED_STATE=failed
        fi
      elif ! ls /opt/earth-cache/Z_*.npy >/dev/null 2>&1; then
        if [ "$EMBED_STATE" != waiting ]; then
          echo "embed cache push: no /opt/earth-cache/Z_*.npy yet at tick ${TICK} (~$((TICK / 2)) min in) — nothing to publish; checking again every ~10 min"
          EMBED_STATE=waiting
        fi
      else
        # A cache with NEITHER marker: pulled by an older build, or written
        # before the markers existed. Try the full push — it refuses an
        # unattested cache by name, which is a line in the log rather than a
        # silence.
        echo "embed cache push: STARTING in-training at tick ${TICK} (~$((TICK / 2)) min in) — $(du -h /opt/earth-cache/Z_*.npy 2>/dev/null | tr '\n' ' ')"
        if python -u ml/embed_cache_sync.py push --run actions --data "$TENSOR" \
             --expect-t "$EXPECT_T"; then
          touch "$EMBED_MARK"
          echo "embed cache push: DONE in-training — the release has it now, hours before this job ends"
          EMBED_STATE=published
        else
          echo "::warning::embed cache push: FAILED in-training at tick ${TICK} — no marker written, retrying in ~10 min"
          EMBED_STATE=failed
        fi
      fi
    fi
  done
  wait $S2_PID
  # PUBLISH IT while the box still has it. The cache outlives this
  # job only if it leaves the box: the hygiene step prunes it under
  # disk pressure, and a destroyed instance takes it with no notice
  # at all. Runs immediately after temporal.py rather than at the
  # end of the step, so a probe-ladder failure below cannot take the
  # 95 minutes with it.
  # `|| echo`, and the omission cost #131 its whole probe ladder.
  # On 2026-08-10 embed_cache_sync was fixed so a failed push exits
  # 1 instead of 0 — correct, and the commit message for it said in
  # so many words that "best-effort is the CALLER's decision
  # (|| true), never a lie told by the callee". The caller half was
  # never written. Under `bash -e` the honest exit code aborted the
  # step, and probe_kfold.py and dip_check.py below it never ran:
  # #131 finished "successfully" with no codec control and no dip
  # capture, because a 5.2 GiB upload could not find room on a box
  # with ~3 GB free.
  #
  # This is ml/CLAUDE.md §5.17 exactly — only a DEFINITE answer may
  # be fatal — and the definite answer here is "the cache did not
  # get published", which costs the NEXT run some GPU and costs this
  # one's results nothing. It must not be able to take the numbers
  # with it.
  #
  # IT STAYS, and it stays UNCONDITIONAL — it is the net under the loop
  # above. If the in-training push failed (no room, no token, a 500 from
  # uploads.github.com) the marker was never written and this is the only
  # thing left that publishes the cache. It costs nothing when the loop
  # already succeeded: embed_cache_sync's "already published and complete"
  # branch no-ops when every chunk name is present in the release AND their
  # sizes sum to the file on disk, and returns 0. After a run of partial
  # pushes it is also the step that turns the last chunks and the manifest
  # complete — the partial path deliberately never writes `complete: true`.
  echo 'embed cache push: STARTING post-training (unconditional, after wait $S2_PID)'
  if python -u ml/embed_cache_sync.py push --run actions --data "$TENSOR" \
       --expect-t "$EXPECT_T"; then
    echo "embed cache push: DONE post-training — the line above says which it was: uploaded here, or already complete from the in-training push"
  else
    echo "::warning::embed cache push: FAILED post-training — the probe ladder continues"
  fi
  # Publish so the stage-2 curve appears on the status page WHILE
  # the ladder finishes, not only after the archive step at the very
  # end. temporal.py writes into the same metrics.jsonl, but nothing
  # in this step pushed it — so the panel existed and stayed empty
  # for the whole probe stage.
  bash scripts/publish_live_metrics.sh "ml-live-${GITHUB_RUN_NUMBER}" || true
fi
# The decisive numbers, computed HERE rather than locally. The
# k-fold is the only probe figure we argue from, and running it
# afterwards on a laptop-sized box means holding a second global
# tensor next to whatever else is running — which OOM-killed it
# on 2026-08-07. The runner already has the matching tensor built
# and the memory to probe it, so the run ships its own verdict.
# Section-only embeddings, so this is minutes, not hours.
python -u ml/probe_kfold.py --runs actions --data "$TENSOR" \
  || echo "::warning::probe_kfold failed — no codec control this run"
python -u ml/dip_check.py --run actions --data "$TENSOR" \
  || echo "::warning::dip_check failed — no dip capture this run"
# The TOP RUNG. Every other probe mean-pools the section before
# reading it, and geostrophic transport is the east-minus-west
# density difference ACROSS the section — the one statistic a mean
# destroys. Runs the head on the embeddings and, immediately after,
# the SAME head on raw 3x3 tokens: the pair is what separates "the
# codec knows this" from "any read-out with spatial structure knows
# this". Matched to a patch codec via --raw-patch.
#
# BOTH SIDES OF A COMPARISON SWITCH TOGETHER (ml/CLAUDE.md §3, 2026-08-21).
# The head is scored against TWO unpooled bars, both fitted here in the same
# loop as the head itself, because a bar that is dispatched separately is a
# bar that goes missing: probe_head has produced output in 3 of 183 archived
# bundles, and every one of those numbers was quoted against probe_kfold's
# POOLED wind baseline. An unpooled codec beside a pooled bar does not measure
# the codec, it measures the read-out.
#   --raw --raw-patch   the 3x3 end-to-end control (is this PRETRAINING or
#                       merely RECEPTIVE FIELD?)
#   --raw --wind-only   the wind bar with the section mean removed and nothing
#                       else changed
# HEAD_TARGETS is a recipe-only key (declared in ml-train.yml) because the
# input list is full at 25. Default "rapid" reproduces every head-probe
# dispatch before today; each extra target is a whole head fit x3, so this is
# a cost knob and it is opt-in.
if [ "${RECIPE_HEAD_PROBE:-$IN_HEAD_PROBE}" = "true" ]; then
  for HT in $(echo "${RECIPE_HEAD_TARGETS:-rapid}" | tr ',' ' '); do
    echo "head probe: target ${HT} (unpooled), with its 3x3 and wind bars"
    python -u ml/probe_head.py --run actions --data "$TENSOR" --K 1 \
      --target "$HT" \
      || echo "::warning::head probe (${HT}) failed — its controls still run"
    python -u ml/probe_head.py --run actions --data "$TENSOR" --K 1 \
      --target "$HT" --raw --raw-patch \
      || echo "::warning::raw-3x3 control (${HT}) failed — the head number for ${HT} has no receptive-field control"
    python -u ml/probe_head.py --run actions --data "$TENSOR" --K 1 \
      --target "$HT" --raw --wind-only \
      || echo "::warning::UNPOOLED WIND BAR (${HT}) failed — the head number for ${HT} has no matched bar, and must NOT be read against probe_kfold's pooled one"
  done
else
  echo "::warning::head_probe is not true — this run produces POOLED numbers only. ml/CLAUDE.md §3 (2026-08-21) says a pooled read-out is a legacy comparable and never a verdict, so nothing this run archives can settle a comparison."
fi

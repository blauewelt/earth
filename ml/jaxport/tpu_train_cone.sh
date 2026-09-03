#!/bin/bash
# The startup script for E-069's CONE-CODEC runs on a Cloud TPU node
# (ml/plans/E069_HANDOVER.md §8.8; JAX port, stage 1). Cloned from
# tpu_train_field.sh — same lifecycle, same self-reap, same EXIT trap, same
# disk guard, same beacon, same publish guard — with the differences that are
# the whole point of this file:
#
#   1. It drives `ml/jaxport/train_cone.py`, not train_field.py, so the
#      artefacts it protects are `ckpt_latest.npz` + `cone_codec.pt` +
#      `metrics.jsonl` + an `in_progress` `results.json` + (at the end)
#      `velocity_probe.json` — and, when SNAPSHOT_ABLATION=1, the L_in=0
#      twin's `metrics_snapshot.jsonl` / `snapshot_codec.pt` /
#      `ckpt_latest_snapshot.npz` beside them.
#   2. THERE IS NO Z, NO CODEC ASSET AND NO PIXELS OBJECT. The cone codec IS
#      stage 1: it reads the raw pentad tensor and writes the embedding
#      everything downstream is built on. Every Z / pixels / codec-staging
#      branch the field launcher carries is deleted here rather than left
#      defaulted, because a knob that does nothing is a knob that will one
#      day be read as if it did (SPOT_LEDGER 08-28).
#   3. It has TWO MODES. `MODE=verify` proves the deployed stack end to end on
#      real chips and real data for a few dollars — the C1–C9 parity gates on
#      CPU, then VERIFY_STEPS steps AT THE REAL GEOMETRY ON THE REAL TENSOR —
#      and it PRINTS `measured: <s/step>`, computed from the trainer's own
#      `wall_s` records, so STEP_EST_S stops being a stand-in before the long
#      run is launched. `MODE=train` is that long run. Verify is the DEFAULT,
#      because the cheap mistake is running the cheap thing twice and the
#      expensive one is a 30 h node that dies at hour 29 on a typo.
#
#   sed -e 's|__BUCKET__|earth-tpu-staging|' \
#       -e 's|__NODE__|e069-cone-s0|' \
#       -e 's|__TPUZONE__|us-west1-c|' \
#       -e 's|^TENSOR_SHA=.*|TENSOR_SHA="<the Phase-A sha256, in full>"|' \
#       ml/jaxport/tpu_train_cone.sh > /tmp/e069.sh
#   python3 scripts/tpu_box.py create e069-cone-s0 --zone us-west1-c --spot \
#       --accelerator-type v5litepod-4 --startup-file /tmp/e069.sh
#
# SED THE PREVIOUS RUN'S OWN FILE FOR A RELAUNCH, NEVER THIS PRISTINE
# TEMPLATE, AND `diff` THE KNOB BLOCKS (SPOT_LEDGER.md 08-28: a template
# rebake reverted every knob and spent 62 min re-embedding a published Z).
#
# A STARTUP SCRIPT INHERITS NO ENVIRONMENT FROM THE MACHINE THAT LAUNCHED IT.
# The `${K:-v}` forms below therefore always take their DEFAULTS on a node;
# to change a knob for a real run, sed it in the same call as __BUCKET__:
#
#       -e 's|^MODE=.*|MODE="train"|' -e 's|^STEPS=.*|STEPS=20000|'
#
# and read the launch log back: the script prints every knob it resolved.
# Every knob below is on its OWN line so that form of sed is safe and so
# `grep -n '^[A-Z_]*=' /tmp/e069.sh` lists all of them — see the knob block.
#
# ──────────────────────────────────────────────────────────────────────────
# THERE IS NO CHEAP STOPPED STATE. DELETE IS THE NORMAL END. Four exits,
# exactly tpu_train_field.sh's: (1) the run finishes (or verify finishes) →
# EXIT trap → ship → self-delete; (2) the run STALLS → progress watchdog, no
# NEW checkpoint object in the bucket for STALL_MIN minutes → reap, which is
# the only monitor that can tell a wedged trainer from a healthy one (both
# hold the chips at 100%); (3) MAX_HOURS, unconditional; (4) in verify mode
# the cap is VERIFY_MAX_HOURS, because "a few dollars" is a promise the node
# has to keep on its own.
#
# To CONTINUE a run, relaunch a node with the SAME __NODE__: the node name is
# the run's identity and step 7 resumes from whatever that bucket prefix holds
# (optimiser moments, the cosine schedule position and BOTH host RNG streams
# included — a true continuation, not a warm start; train_cone.py refuses a
# resume whose --steps differs, because --steps IS the schedule).
#
# ──────────────────────────────────────────────────────────────────────────
# THE OBJECT THIS EXPECTS. One, and it is the only one:
#
#   gs://<bucket>/tensors/family4_na025_pentad_r3_<sha10>.npz
#
# the r3 pentad tensor Phase A builds and publishes (HANDOVER §4). Its name
# is TENSOR_NAME plus the first ten hex of TENSOR_SHA, i.e. exactly the name
# the data-cache-v1 chunks carry (`…npz.aa/ab/ac/ad` — FOUR parts for r3,
# not three: it chunks at 1.5 GiB and r3 is two channels larger than r2's
# three-part file; TENSOR_PARTS names however many the release actually
# holds and defaults to r3's own count), un-chunked.
#
#   object present in the bucket   → stage it (~1 min: same-cloud bytes)
#   object absent, release present → ASSEMBLE it from data-cache-v1 with a
#                                    FULL sha256 verify against TENSOR_SHA,
#                                    then PUBLISH it to the default GCS path
#                                    so the next node stages in a minute
#                                    instead of paying the same ~5 minutes
#                                    (ml/CLAUDE.md §5.26).
#   both absent                    → REFUSE, with the staging command.
#
# TENSOR_SHA IS REQUIRED AND IS REFUSED IF IT IS STILL THE PLACEHOLDER. The
# sha is the tensor's identity: the box effect (ml/CLAUDE.md §7) moved a head
# k-fold by 0.041 between two builds of "the same" tensor, so a run that
# cannot name its bytes is a run that cannot be compared to anything. There
# is no default here and there will never be one — Phase A writes the sha
# into ml/EXPERIMENTS.md and claude/expectations.md, and the bake seds it in.
#
# ──────────────────────────────────────────────────────────────────────────
# THE STALL ARITHMETIC, WHICH IS WHY E-051 BURNED A NODE.
#
# The progress watchdog measures SHIPPED checkpoints, so the quantity that has
# to clear STALL_MIN is not the first checkpoint WRITE, it is the first
# checkpoint SHIP — one more SHIP_EVERY_MIN cycle behind it:
#
#   t_first_ship = SETUP_EST_MIN                 (apt + venv + jax/flax/optax
#                                                 + CPU torch + clone + the
#                                                 anomaly transform and XLA
#                                                 compile)
#                + (staged bytes / STAGE_MBPS)   (the tensor, nothing else)
#                + CKPT_EVERY * STEP_EST_S       (steps to the first ckpt)
#                + SHIP_EVERY_MIN                (worst-case wait for the
#                                                 shipper's next cycle)
#
# The cone's FIRST-RUN numbers, stated so the log can be checked against them:
# SETUP_EST_MIN=12, STAGE_MBPS=200, RELEASE_MBPS=40, the tensor 11.5 GB → ~1
# min from the bucket / ~5 min from the release, CKPT_EVERY=1000, and
# STEP_EST_S **unknown — 0.5 s is a STAND-IN until the verify leg measures
# it** (the host gather bounds it: ml/cone_sampler.py's numpy gather runs on
# the 112-CPU host and took 0.15 s per 256 anchors on a 4-core sandbox).
# So t_first_ship ≈ 12 + 1 + 1000·0.5/60 + 10 ≈ 31 min against STALL_MIN=90 —
# it clears, with room for the estimate to be wrong by a factor of five.
# Step 1 computes it from the RESOLVED config and REFUSES if it does not
# clear; `ALLOW_STALL_RISK=1` overrides, because an estimate is an estimate.
#
# IF THE VERIFY LEG READS > 4 s/step the gather is starving the chip: batch
# anchors by latitude row in the sampler call (ConeSampler already groups by
# row) and prefetch one batch on a thread — the tpu_train_s2.sh pipeline-knob
# pattern (GATHER_WORKERS, PREFETCH). That is a code change, not a knob here.
#
# ──────────────────────────────────────────────────────────────────────────
# THE BUCKET LAYOUT THIS WRITES (runs/<node>/): metrics.jsonl, results.json
# (with in_progress), ckpt_latest.npz, cone_codec.pt, velocity_probe.json (at
# the end), verify_report.json (verify mode), logs/<STAMP>.txt — and, under
# SNAPSHOT_ABLATION=1, metrics_snapshot.jsonl, snapshot_codec.pt and
# ckpt_latest_snapshot.npz. scripts/tpu_status_mirror.py copies
# <node>/metrics.jsonl into ml-live-tpu and status.html's coneChart draws it.
# ──────────────────────────────────────────────────────────────────────────
set -euo pipefail

# --------------------------------------------------------------------------
# knobs — every one of them is echoed in step 0 before anything is spent
# --------------------------------------------------------------------------
BUCKET="__BUCKET__"                 # substituted at launch
NODE_NAME="__NODE__"
TPUZONE="__TPUZONE__"
REPO="blauewelt/earth"
WORK="/opt/earth-cone"
OUT="${WORK}/run"                   # train_cone.py's --out: the checkpoint,
                                    # the .pt, metrics.jsonl, results.json and
                                    # velocity_probe.json all live in HERE.
RUNS_PREFIX="runs/${NODE_NAME}"

# verify | train. See the header. Verify is the default on purpose.
MODE="${MODE:-verify}"

# THE CODE, pinned. Empty is allowed ONLY in verify mode — the point of a
# verify run is to test what is deployed — and is REFUSED in train mode: a
# run whose code moved under it is not comparable to anything.
GIT_SHA="${GIT_SHA:-}"

# knobs — the cone codec
TENSOR_NAME="${TENSOR_NAME:-family4_na025_pentad_r3}"
TENSOR_SHA="${TENSOR_SHA:-<the Phase-A sha256>}"        # REQUIRED, refuse if empty
# FOUR parts for family4_na025_pentad_r3 (Phase A, #535: it chunks at
# 1.5 GiB and r3 is two channels larger than r2's three-part file —
# aa/ab/ac at 1,572,864,000 B each + ad at 584,889,823 B, measured by
# listing the data-cache-v1 release, not assumed). A future TENSOR_NAME
# with a different part count overrides this at launch, same as every
# other knob (see the sed recipe in the header).
TENSOR_PARTS="${TENSOR_PARTS:-aa ab ac ad}"
GCS_TENSOR="${GCS_TENSOR:-tensors/${TENSOR_NAME}_${TENSOR_SHA:0:10}.npz}"
EST_TENSOR_BYTES="${EST_TENSOR_BYTES:-11500000000}"
# ONE ASSIGNMENT PER LINE, deliberately. HANDOVER §8.8 packs these five to a
# line; the bake's own documented step is `sed -e 's|^STEPS=.*|STEPS=…|'` and
# `grep -n '^[A-Z_]*='`, and against a packed line that sed DELETES the four
# knobs sharing it while the grep shows only the first. That is the 08-28
# incident (a rebake reverted every knob and spent 62 min on the wrong
# configuration) with the knife handed over. Same names, same values, one
# knob per line, so every knob is sed-able and every knob is in the grep.
STEPS=20000
BATCH=256
LR=3e-4
SEED=0
D_Z=32
D_MODEL=256
HEADS=8
N_LATENTS=64
LAYERS=6
D_DEC=256
DEC_LAYERS=2
L_IN=6
FUTURE_LAGS="1,2"
AUX_W=0.25
DOT_QUERIES=256
HOLDOUT_YEARS="2008,2009,2016,2017,2021,2022,2023,2024"   # the frozen protocol (PROTOCOL_RESET §3.1): train ≤ 2020; 2009,2017,2023 was the development split
EVAL_EVERY=1000
CKPT_EVERY=1000
VELOCITY_PROBE=1
SNAPSHOT_ABLATION=0
TAG=""
EXTRA_ARGS=""
# no Z, no codec asset, no pixels object: the cone codec is stage 1
#
# TAG is the run's label. train_cone.py has NO --tag flag (it would be
# refused, not ignored — the jaxport convention), and its only label channel
# is the RECIPE_NAME environment variable, which it records as `recipe` in
# the config record. So TAG is exported onto the trainer's own invocation
# line, where it lands in metrics.jsonl and results.json and travels with the
# artefact. Empty is the normal case.

# Verify mode's own budget. The leg is VERIFY_STEPS at the REAL geometry,
# batch and tensor — an s/step measured at a toy shape would be a number
# about a toy — with eval-every == steps so exactly ONE eval happens at the
# end (train_cone.py also evaluates at step 0, and the pair is what the
# s/step measurement is computed from).
VERIFY_STEPS="${VERIFY_STEPS:-300}"
VERIFY_MAX_HOURS="${VERIFY_MAX_HOURS:-2}"

# Lifecycle.
SHIP_EVERY_MIN="${SHIP_EVERY_MIN:-10}"   # upload cadence (train mode)
STALL_MIN="${STALL_MIN:-90}"             # progress watchdog (train mode)
MAX_HOURS="${MAX_HOURS:-30}"             # unconditional cap (train mode)

# Inputs to the stall arithmetic in step 1. All of them are ESTIMATES and the
# echo says so; their provenance is in the header.
SETUP_EST_MIN="${SETUP_EST_MIN:-12}"     # apt+venv+pip+clone+anomaly+compile
STAGE_MBPS="${STAGE_MBPS:-200}"          # GCS <-> TPU VM, MB/s, conservative
RELEASE_MBPS="${RELEASE_MBPS:-40}"       # GitHub release CDN -> TPU VM, MB/s
# UNKNOWN for this model. 0.5 s/step is a STAND-IN, bounded by the host
# gather (0.15 s per 256 anchors on a 4-core sandbox; this host has 112
# cores), and the verify leg replaces it with a measurement.
STEP_EST_S="${STEP_EST_S:-0.5}"
ALLOW_STALL_RISK="${ALLOW_STALL_RISK:-0}"

# The disk guard. Sized from the allocation it guards (ml/CLAUDE.md §5.18):
# the r3 tensor archive ~11.5 GB, and — the term that dominates — the ANOMALY
# SCRATCH COPY of the 42-channel pentad tensor that train_cone.py's load_data
# writes on the node, ~36 GB in float32, plus a rolling checkpoint pair and
# the venv's jax/torch wheels. 120 GB covers all of it; the v5e boot disk
# serves ~90 GB free, so the /dev/shm fallback below is the normal path.
NEED_GB="${NEED_GB:-120}"

STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
LOG=/tmp/train_cone.log
LAST_CKPT_MARK=/tmp/last_ckpt_upload     # touched when a NEW ckpt is SHIPPED
SIGDIR=/tmp/shipsig                      # one file per shipped object
CKPT_NPZ_NAME="ckpt_latest.npz"
CKPT_PT_NAME="cone_codec.pt"
PROBE_NAME="velocity_probe.json"
SNAP_NPZ_NAME="ckpt_latest_snapshot.npz"
SNAP_PT_NAME="snapshot_codec.pt"
SNAP_METRICS_NAME="metrics_snapshot.jsonl"
REPORT="/tmp/verify_report.json"

# --------------------------------------------------------------------------
# 0 · self-reap, log shipping, the shipper and the two watchdogs
# --------------------------------------------------------------------------
exec >>"${LOG}" 2>&1
mkdir -p "${SIGDIR}"

# `warn` exists so that every best-effort step says WHY it is best-effort in
# the same breath as giving up (ml/CLAUDE.md §4.6). There is no bare `|| true`
# in this file.
warn() { echo "WARN: $*"; }

token() {
  curl -sf -H 'Metadata-Flavor: Google' \
    'http://metadata.google.internal/computeMetadata/v1/instance/service-accounts/default/token' \
    | python3 -c 'import json,sys; print(json.load(sys.stdin)["access_token"])'
}

# `-T` STREAMS the file; `--data-binary @file` buffers the whole thing in
# memory, which is fine for a few-KB JSON and is not fine for a multi-GB
# checkpoint (ml/CLAUDE.md §7 — a 1.5 GiB chunk died of exactly this, every
# time, for a day).
gcs_put() {   # gcs_put <local file> <object name> <content-type>
  local T
  T="$(token)" || return 1
  curl -sf -X POST -H "Authorization: Bearer ${T}" \
    -H "Content-Type: ${3}" -T "${1}" \
    "https://storage.googleapis.com/upload/storage/v1/b/${BUCKET}/o?uploadType=media&name=$(
       python3 -c 'import sys,urllib.parse; print(urllib.parse.quote(sys.argv[1], safe=""))' "${2}")" \
    >/dev/null
}

gcs_get() {   # gcs_get <object name> <local file>  — 0 on success, 1 if absent
  local T
  T="$(token)" || return 1
  curl -sf -H "Authorization: Bearer ${T}" -o "${2}" \
    "https://storage.googleapis.com/storage/v1/b/${BUCKET}/o/$(
       python3 -c 'import sys,urllib.parse; print(urllib.parse.quote(sys.argv[1], safe=""))' "${1}")?alt=media"
}

# The object's declared size, on stdout; non-zero and a REASON on stderr when
# the object is absent or unreadable. Deliberately NOT `curl -f`: a 404 body
# says `notFound` and a 403 body says which permission is missing, and that
# line is the one that explains the branch (ml/CLAUDE.md §4.6 — `2>/dev/null`
# on a command whose failure you are branching on hides exactly this).
gcs_size() {  # gcs_size <object name>
  # TRANSPORT failures are retried; only a metadata answer decides. On
  # 2026-08-26 a one-shot "curl: (6) Could not resolve host" ~19 s after
  # boot — with the two checks a second earlier both fine — took the
  # "|| return 1" path, and the caller's refusal branch read that as
  # "object absent" and reaped a healthy node. DNS on a fresh node is not
  # settled the first minute; a 404 is an answer, a resolver hiccup is not.
  local T BODY tries
  for tries in 1 2 3 4; do
    T="$(token)" && \
    BODY="$(curl -sS -H "Authorization: Bearer ${T}" \
      "https://storage.googleapis.com/storage/v1/b/${BUCKET}/o/$(
         python3 -c 'import sys,urllib.parse; print(urllib.parse.quote(sys.argv[1], safe=""))' "${1}")?fields=size")" \
      && break
    if [ "${tries}" = 4 ]; then return 1; fi
    echo "gcs_size: transport failure for ${1} (attempt ${tries}) — retrying in 10 s" >&2
    sleep 10
  done
  printf '%s' "${BODY}" | python3 -c '
import json, sys
try:
    d = json.load(sys.stdin)
except Exception as e:
    print("gcs_size: unparseable metadata response: %s" % e, file=sys.stderr)
    sys.exit(1)
if "size" not in d:
    print("gcs_size: %s" % json.dumps(d.get("error", d))[:300], file=sys.stderr)
    sys.exit(1)
print(d["size"])'
}

enc() { python3 -c 'import sys,urllib.parse; print(urllib.parse.quote(sys.argv[1], safe=""))' "${1}"; }

gcs_delete() {   # gcs_delete <object name>
  local T
  T="$(token)" || return 1
  curl -sf -X DELETE -H "Authorization: Bearer ${T}" \
    "https://storage.googleapis.com/storage/v1/b/${BUCKET}/o/$(enc "${1}")" >/dev/null
}

gcs_copy() {     # gcs_copy <src object> <dst object> — server-side, no bytes on the wire
  local T
  T="$(token)" || return 1
  curl -sf -X POST -H "Authorization: Bearer ${T}" -H 'Content-Length: 0' \
    "https://storage.googleapis.com/storage/v1/b/${BUCKET}/o/$(enc "${1}")/copyTo/b/${BUCKET}/o/$(enc "${2}")" \
    >/dev/null
}

# PUBLISH AN ASSEMBLED ARTEFACT UNDER A NAME THAT MEANS "VERIFIED".
#
# The guard is upload-to-`<name>.uploading` → read the size the BUCKET reports
# → assert it equals the local size → server-side copy onto the real name →
# delete the temp. The reason for the two-step rather than the size check
# alone is that the size check can only be made AFTER the upload finishes: a
# plain upload straight to the final name would make that name exist, and be
# stageable by a concurrent node, for the whole minutes-long window in which
# its bytes are unverified. With the temp name, the real name comes into
# existence only after the assertion, in one server-side operation that moves
# no bytes. On a mismatch the temp is DELETED rather than left as litter that
# a future reader has to decide about (§5.21's flush-then-mark, in its
# object-store form).
#
# BEST-EFFORT BY DESIGN: every failure path here warns and returns 1. The
# artefact is already on this node and already verified; publishing it is an
# optimisation for the NEXT node (§5.26) and must never take this run down.
gcs_publish() {  # gcs_publish <local file> <object name> <label>
  local TMP WANT GOT VIA
  WANT="$(stat -c %s "${1}")"
  TMP="${2}.uploading"
  echo "publishing ${3}: ${1} -> gs://${BUCKET}/${2} (${WANT} B) via ${TMP} …"
  # gcloud/gsutil FIRST when the image has them, for the same reason
  # tpu_box.py's own `stage` is resumable in 64 MiB chunks: a single-request
  # media POST of an 11.5 GB object loses everything to one dropped
  # connection. The curl path stays as the fallback so nothing here REQUIRES
  # the SDK.
  if command -v gcloud >/dev/null 2>&1 && gcloud storage cp "${1}" "gs://${BUCKET}/${TMP}"; then
    VIA="gcloud storage"
  elif command -v gsutil >/dev/null 2>&1 && gsutil -q cp "${1}" "gs://${BUCKET}/${TMP}"; then
    VIA="gsutil"
  elif gcs_put "${1}" "${TMP}" "application/octet-stream"; then
    VIA="curl+JSON API"
  else
    warn "publish of ${3} failed during upload — best effort: this run has the artefact and continues; the next node pays the assembly again"
    gcs_delete "${TMP}" || warn "…and the partial ${TMP} could not be deleted either; it is NOT the artefact name, so nothing will mistake it for one"
    return 1
  fi
  echo "uploaded ${3} to ${TMP} via ${VIA} — verifying before it takes the artefact name"
  GOT="$(gcs_size "${TMP}" || true)"
  if [ "${GOT}" != "${WANT}" ]; then
    warn "publish of ${3} ABORTED: the bucket reports ${GOT:-<unreadable>} B for ${TMP} against ${WANT} B locally — deleting the temp object"
    gcs_delete "${TMP}" || warn "could not delete ${TMP} — it never wore the artefact name, so it is litter and not a trap"
    return 1
  fi
  if ! gcs_copy "${TMP}" "${2}"; then
    warn "publish of ${3} failed at the final copy — ${TMP} holds verified bytes but the artefact name was NOT created"
    gcs_delete "${TMP}" || warn "could not delete ${TMP}"
    return 1
  fi
  gcs_delete "${TMP}" || warn "published ${2} but could not delete ${TMP} — harmless, and a later launch overwrites it"
  echo "measured: PUBLISHED gs://${BUCKET}/${2} — ${GOT} B, verified equal to" \
       "the local ${WANT} B before the name existed. The next node stages this" \
       "in minutes instead of re-assembling it from the releases."
}

upload_log() { gcs_put "${LOG}" "${RUNS_PREFIX}/logs/${STAMP}.txt" "text/plain" \
  || warn "log upload failed this cycle — best effort because losing the log must never take the run with it"; }

self_delete() {
  # RETRIED UNTIL THE NODE IS REALLY GONE. A DELETE issued while the CREATE
  # operation is still settling is silently dropped (measured 2026-08-23 on
  # the smoke node), so this asserts the 404 rather than trusting the DELETE.
  local P T CODE
  P="$(curl -sf -H 'Metadata-Flavor: Google' \
      'http://metadata.google.internal/computeMetadata/v1/project/project-id')"
  for _i in 1 2 3 4 5 6; do
    T="$(token)"
    curl -sf -X DELETE -H "Authorization: Bearer ${T}" \
      "https://tpu.googleapis.com/v2/projects/${P}/locations/${TPUZONE}/nodes/${NODE_NAME}" \
      || warn "DELETE call itself failed on attempt ${_i} — the 404 assertion below is what decides"
    sleep 30
    CODE="$(curl -s -o /dev/null -w '%{http_code}' -H "Authorization: Bearer ${T}" \
      "https://tpu.googleapis.com/v2/projects/${P}/locations/${TPUZONE}/nodes/${NODE_NAME}")"
    if [ "${CODE}" = "404" ]; then echo "self-delete confirmed (404)"; return 0; fi
    echo "node still answers ${CODE} after delete attempt ${_i} — retrying"
  done
}

# rsync-style: upload ONLY if the file changed since the last successful ship.
# This is what makes the progress watchdog measure PROGRESS rather than
# measure the uploader — a wedged trainer keeps the same checkpoint on disk,
# nothing new is shipped, the marker stops moving, and the node is reaped.
# Returns 0 only when something NEW actually landed in the bucket.
ship_file() {  # ship_file <local> <object name> <content-type> <sig tag>
  local SIG SF
  [ -f "${1}" ] || return 1
  SF="${SIGDIR}/${4}"
  SIG="$(stat -c '%Y:%s' "${1}")"
  if [ -f "${SF}" ] && [ "${SIG}" = "$(cat "${SF}")" ]; then
    return 1
  fi
  if gcs_put "${1}" "${2}" "${3}"; then
    echo "${SIG}" > "${SF}"
    echo "shipped ${2} (${SIG}, $(du -h "${1}" | cut -f1))"
    return 0
  fi
  warn "upload of ${2} FAILED this cycle (will retry next cycle)"
  return 1
}

ship_state() {
  # NOTE ON THE `|| true`s BELOW: `ship_file` returns 1 for "nothing changed,
  # nothing to do", which is its NORMAL busiest-case answer, not an error —
  # the error path is inside ship_file and says so with `warn`. Without the
  # `|| true` a quiet cycle would abort the shipper under `set -e`.
  # metrics.jsonl and the in_progress results.json EVERY cycle they change:
  # they are small, they are the black box, and §5.25 says a reader must be
  # able to see the numbers a long job has already computed. metrics.jsonl is
  # also what scripts/tpu_status_mirror.py copies into ml-live-tpu, i.e. what
  # status.html's coneChart draws.
  ship_file "${OUT}/metrics.jsonl" "${RUNS_PREFIX}/metrics.jsonl" \
    "application/x-ndjson" "metrics" || true
  ship_file "${OUT}/results.json" "${RUNS_PREFIX}/results.json" \
    "application/json" "results" || true
  # The H1 artefact. It exists only after the run's last step, so this is a
  # no-op on every cycle but the last one — and it must be shipped by the
  # ordinary path rather than only by the trap, because the trap is the
  # safety net.
  ship_file "${OUT}/${PROBE_NAME}" "${RUNS_PREFIX}/${PROBE_NAME}" \
    "application/json" "probe" || true
  # The CHECKPOINT, and the torch-loadable codec WITH it, so the velocity
  # probe and the eval ladder never have to wait for the run to end
  # (HANDOVER §8.6). Only a NEW ckpt moves the watchdog marker.
  if ship_file "${OUT}/${CKPT_NPZ_NAME}" "${RUNS_PREFIX}/${CKPT_NPZ_NAME}" \
       "application/octet-stream" "ckpt"; then
    ship_file "${OUT}/${CKPT_PT_NAME}" "${RUNS_PREFIX}/${CKPT_PT_NAME}" \
      "application/octet-stream" "ckptpt" || true
    touch "${LAST_CKPT_MARK}"
    echo "progress: NEW checkpoint landed in gs://${BUCKET}/${RUNS_PREFIX}/"
  fi
  # The L_in=0 twin, when it is being trained. Its checkpoint deliberately
  # does NOT move the watchdog marker: the twin runs AFTER the cone arm, and
  # the arm that is billing the chips is the one the watchdog must watch.
  if [ "${SNAPSHOT_ABLATION}" = "1" ]; then
    ship_file "${OUT}/${SNAP_METRICS_NAME}" "${RUNS_PREFIX}/${SNAP_METRICS_NAME}" \
      "application/x-ndjson" "snapmetrics" || true
    ship_file "${OUT}/${SNAP_NPZ_NAME}" "${RUNS_PREFIX}/${SNAP_NPZ_NAME}" \
      "application/octet-stream" "snapckpt" || true
    ship_file "${OUT}/${SNAP_PT_NAME}" "${RUNS_PREFIX}/${SNAP_PT_NAME}" \
      "application/octet-stream" "snapckptpt" || true
  fi
}

ship_report() {
  [ -f "${REPORT}" ] || return 0
  if gcs_put "${REPORT}" "${RUNS_PREFIX}/verify_report.json" "application/json"; then
    echo "shipped gs://${BUCKET}/${RUNS_PREFIX}/verify_report.json"
  else
    warn "verify_report.json upload failed — the log is then the only copy of the verdict"
  fi
  gcs_put "${REPORT}" "${RUNS_PREFIX}/verify_report_${STAMP}.json" "application/json" \
    || warn "stamped verify_report copy failed — best effort: the canonical name above is the one a reader looks for"
}

ship_final() {
  if [ "${MODE}" = "verify" ]; then ship_report; else ship_state; fi
}

# The trap ships EVERYTHING first and deletes the node last. Anything not on
# the bucket does not survive the reap.
trap 'code=$?; echo "exit ${code} — shipping final state, then the node self-deletes"; ship_final; upload_log; self_delete' EXIT

echo "=== tpu_train_cone ${STAMP} · node ${NODE_NAME} · bucket ${BUCKET} · MODE ${MODE} ==="

# BOOT BEACON — the first bucket object must land within ~3 minutes of the
# script starting, unconditionally. It fires HERE, at the banner, because the
# trap is already armed above it: a log in the bucket therefore proves this
# script is genuinely executing and that the machinery which reaps the node
# exists, not merely that the node booted. Without it a node whose startup
# script never ran (the us-west1-c maintenance-event zombies, 2026-08-26) and
# a node quietly mid-staging are the SAME OBSERVATION for half an hour — and
# on 2026-08-26 a possibly-healthy node was reaped at minute 16 on that
# ambiguity while a genuine zombie had earlier been left to bill for 53. One
# early log upload plus a 3-minute shipper turns "no object under
# runs/<node>/ in ~6 min" into a certain zombie verdict (ml/CLAUDE.md §7).
# The 3-minute shipper is RETIRED by `kill "${BEACON_PID}"` when section 8
# arms the 10-minute one, so two uploaders never race on the same object.
upload_log
( while true; do sleep 180; upload_log; done ) &
BEACON_PID=$!
disown
echo "measured: boot beacon shipped and 3-min early log shipper armed (pid ${BEACON_PID})"

# EVERY KNOB, RESOLVED, in one place. A startup script inherits no
# environment, so "I set STEPS=20000 when I launched it" is a claim about the
# launching shell and not about this node; this is the line that settles it.
# READ IT FIRST on every launch (HANDOVER §8.9 step 5): the config is the
# experiment.
echo "resolved knobs: mode ${MODE} · steps ${STEPS} × batch ${BATCH} · lr ${LR} ·" \
     "seed ${SEED} · d_z ${D_Z} · d_model ${D_MODEL} × layers ${LAYERS} ×" \
     "heads ${HEADS} · n_latents ${N_LATENTS} · d_dec ${D_DEC} ×" \
     "dec_layers ${DEC_LAYERS} · L_in ${L_IN} · future_lags ${FUTURE_LAGS} ·" \
     "aux_w ${AUX_W} · dot_queries ${DOT_QUERIES} ·" \
     "holdout_years ${HOLDOUT_YEARS} · eval_every ${EVAL_EVERY} ·" \
     "ckpt_every ${CKPT_EVERY} · velocity_probe ${VELOCITY_PROBE} ·" \
     "snapshot_ablation ${SNAPSHOT_ABLATION} · tag '${TAG}' ·" \
     "extra '${EXTRA_ARGS}'"
echo "resolved data: tensor gs://${BUCKET}/${GCS_TENSOR} · no Z, no codec" \
     "asset, no pixels object (the cone codec IS stage 1) ·" \
     "code ${REPO}@${GIT_SHA:-main HEAD at launch}"
echo "resolved release fallback (used only for an absent bucket object, and" \
     "the assembled file is published back to that path): tensor" \
     "${TENSOR_NAME}_${TENSOR_SHA:0:10}.npz parts '${TENSOR_PARTS}' from" \
     "data-cache-v1, full sha256 pin ${TENSOR_SHA} · declared size estimate" \
     "${EST_TENSOR_BYTES} B · release rate estimate ${RELEASE_MBPS} MB/s"
echo "resolved verify budget: ${VERIFY_STEPS} steps at the REAL geometry on" \
     "the REAL tensor + the C1–C9 parity gates on CPU · cap ${VERIFY_MAX_HOURS} h"
echo "resolved lifecycle: ship every ${SHIP_EVERY_MIN} min · stall watchdog" \
     "${STALL_MIN} min · hard cap ${MAX_HOURS} h · disk guard ${NEED_GB} GB"

# Watchdog B: the unconditional cap. It is the one that fires when the failure
# is in watchdog A. Verify gets a much shorter one — the promise that a verify
# node costs a few dollars has to be kept by the node, not by a watcher.
CAP_HOURS="${MAX_HOURS}"
if [ "${MODE}" = "verify" ]; then CAP_HOURS="${VERIFY_MAX_HOURS}"; fi
( sleep $(( CAP_HOURS * 3600 ))
  echo "HARD CAP: ${CAP_HOURS} h reached — force self-delete"
  ship_final; upload_log; self_delete ) &
disown
echo "measured: hard cap armed at ${CAP_HOURS} h"

# --------------------------------------------------------------------------
# 0b · the refusals that depend on NOTHING but the resolved config
# --------------------------------------------------------------------------
# These are the earliest refusals this script can make and they cost nothing
# but the seconds already spent. They live BELOW the EXIT trap and the beacon
# on purpose: a refusal above the trap would exit a node that then bills until
# somebody notices, which is the one failure mode this whole file exists to
# prevent (SPOT_LEDGER, the §8 orphan).
if [ -z "${TENSOR_SHA}" ] || [ "${TENSOR_SHA}" = "<the Phase-A sha256>" ]; then
  echo "REFUSING: TENSOR_SHA is ${TENSOR_SHA:-<empty>} — the template's" \
       "placeholder, not a hash. The sha256 IS the tensor's identity: two" \
       "builds of 'the same' tensor moved a head k-fold by 0.041 (the box" \
       "effect, ml/CLAUDE.md §7), the bucket object is NAMED by its first ten" \
       "hex, and the release fallback verifies against it in full. A run that" \
       "cannot name its bytes cannot be compared to anything, so there is no" \
       "default here and there will never be one. Phase A records the sha in" \
       "ml/EXPERIMENTS.md and claude/expectations.md; bake it in:"
  echo "  sed -e 's|^TENSOR_SHA=.*|TENSOR_SHA=\"<the 64 hex>\"|' … "
  echo "Nothing has been staged and nothing has been spent."
  exit 1
fi
if [ "${#TENSOR_SHA}" != "64" ]; then
  echo "REFUSING: TENSOR_SHA is ${#TENSOR_SHA} characters, not 64. The" \
       "release fallback compares it against sha256sum output in FULL, and a" \
       "truncated pin would either never match or — worse — be silently" \
       "reformatted into a prefix comparison by whoever fixes the mismatch."
  exit 1
fi
if [ "${MODE}" = "train" ] && [ -z "${GIT_SHA}" ]; then
  echo "REFUSING: MODE=train with an empty GIT_SHA. A verify leg SHOULD run" \
       "main's HEAD — the point is to test what is deployed — but a 20,000-" \
       "step run whose code moved under it is not comparable to anything, and" \
       "this run's whole value is a cross-backend comparison against a torch" \
       "twin (§8.9 step 9). Sed the sha the dispatch names:"
  echo "  -e 's|^GIT_SHA=.*|GIT_SHA=\"\${GIT_SHA:-<40 hex>}\"|'"
  exit 1
fi
if [ "${MODE}" != "verify" ] && [ "${MODE}" != "train" ]; then
  echo "REFUSING: MODE must be 'verify' or 'train', got '${MODE}'."
  exit 1
fi

# --------------------------------------------------------------------------
# 1 · the STALL ARITHMETIC, before anything is spent
# --------------------------------------------------------------------------
# Read the header for the formula. This step also DECIDES THE ROUTE for the
# tensor — bucket or release-assembly — because the two cost wildly different
# amounts of wall clock and the watchdog does not care which one you are
# paying. Deciding it here, from one small metadata GET, means the refusal
# case costs nothing.
echo "--- stall arithmetic ---"
STAGE_BYTES=0          # bytes to move GCS -> node
ASSEMBLE_BYTES=0       # bytes to pull from the releases AND push back
TENSOR_ROUTE="bucket"

SZ="$(gcs_size "${GCS_TENSOR}" || true)"
if [ -n "${SZ}" ]; then
  echo "measured: tensor gs://${BUCKET}/${GCS_TENSOR} exists, ${SZ} B — staging route"
  STAGE_BYTES=$(( STAGE_BYTES + SZ ))
else
  TENSOR_ROUTE="release"
  ASSEMBLE_BYTES=$(( ASSEMBLE_BYTES + EST_TENSOR_BYTES ))
  echo "measured: tensor gs://${BUCKET}/${GCS_TENSOR} is ABSENT — falling back" \
       "to data-cache-v1 (${TENSOR_NAME}_${TENSOR_SHA:0:10}.npz.{${TENSOR_PARTS// /,}})," \
       "then publishing the assembled file back to that path"
fi

FIRST_SHIP_MIN="$(python3 -c '
import sys
(setup, mbps, byts, asm_bytes, rel_mbps,
 ckpt_every, step_s, ship) = (float(x) for x in sys.argv[1:])
dl = byts / (mbps * 1e6) / 60.0
# an assembled artefact is paid for TWICE: pulled from the release CDN, then
# pushed back to the bucket for the next node.
asm = asm_bytes / (rel_mbps * 1e6) / 60.0 + asm_bytes / (mbps * 1e6) / 60.0
train = ckpt_every * step_s / 60.0
print("%.1f %.1f %.1f %.1f" % (setup + dl + asm + train + ship, dl, asm, train))' \
  "${SETUP_EST_MIN}" "${STAGE_MBPS}" "${STAGE_BYTES}" "${ASSEMBLE_BYTES}" \
  "${RELEASE_MBPS}" "${CKPT_EVERY}" "${STEP_EST_S}" "${SHIP_EVERY_MIN}")"
EST_TOTAL="$(echo "${FIRST_SHIP_MIN}" | cut -d' ' -f1)"
EST_DL="$(echo "${FIRST_SHIP_MIN}" | cut -d' ' -f2)"
EST_ASM="$(echo "${FIRST_SHIP_MIN}" | cut -d' ' -f3)"
EST_TRAIN="$(echo "${FIRST_SHIP_MIN}" | cut -d' ' -f4)"
echo "estimated time to the FIRST SHIPPED checkpoint: ${EST_TOTAL} min" \
     "= setup ${SETUP_EST_MIN} + staging ${EST_DL} (${STAGE_BYTES} B at" \
     "${STAGE_MBPS} MB/s) + assembly+publish ${EST_ASM} (${ASSEMBLE_BYTES} B" \
     "at ${RELEASE_MBPS} MB/s down and ${STAGE_MBPS} MB/s back up)" \
     "+ ${CKPT_EVERY} steps ${EST_TRAIN} (at ${STEP_EST_S} s/step, which for" \
     "this model is UNKNOWN — a stand-in bounded by the host gather until the" \
     "verify leg measures it) + shipper cycle ${SHIP_EVERY_MIN}." \
     "Route: tensor ${TENSOR_ROUTE}. Watchdog limit STALL_MIN ${STALL_MIN} min."
if [ "${MODE}" = "train" ]; then
  if python3 -c 'import sys; sys.exit(0 if float(sys.argv[1]) >= float(sys.argv[2]) else 1)' \
       "${EST_TOTAL}" "${STALL_MIN}"; then
    if [ "${ALLOW_STALL_RISK}" = "1" ]; then
      echo "WARN: the estimate (${EST_TOTAL} min) does NOT clear STALL_MIN" \
           "(${STALL_MIN}) — proceeding only because ALLOW_STALL_RISK=1."
    else
      echo "REFUSING: this configuration self-reaps while healthy. The first" \
           "shipped checkpoint is estimated at ${EST_TOTAL} min against a" \
           "STALL_MIN of ${STALL_MIN}. Raise STALL_MIN, lower CKPT_EVERY, or" \
           "set ALLOW_STALL_RISK=1 if you believe STEP_EST_S is pessimistic." \
           "If the assembly term (${EST_ASM} min) is what pushed it over," \
           "the cheaper fix is to stage the tensor once from a box that" \
           "already holds it — every later node then takes the bucket route." \
           "(E-051 lost a node to exactly this arithmetic.)"
      exit 1
    fi
  else
    echo "measured: the arithmetic CLEARS the watchdog (${EST_TOTAL} <" \
         "${STALL_MIN} min)."
  fi
else
  echo "verify mode: the progress watchdog is not armed (there are no" \
       "checkpoints to ship), so this arithmetic is FYI for the train launch" \
       "that follows — recompute it with the s/step this run measures."
fi

# --------------------------------------------------------------------------
# 2 · what this host actually is, and where the data can live
# --------------------------------------------------------------------------
echo "--- host ---"
mkdir -p "${WORK}" "${OUT}" "${WORK}/cache"
echo "measured: $(nproc) CPUs, $(awk '/MemTotal/{printf "%.0f GB RAM", $2/1048576}' /proc/meminfo)"
AVAIL_GB="$(df -BG --output=avail "${WORK}" | tail -1 | tr -dc '0-9')"
echo "measured: ${AVAIL_GB} GB free on ${WORK}"
if [ "${AVAIL_GB}" -lt "${NEED_GB}" ]; then
  # The v5e boot disk serves ~90 GB free — short of the cone allocation —
  # but the host carries 189 GB of RAM (measured, E-051 2026-08-25), so
  # before refusing, try tmpfs: remount /dev/shm large enough and move WORK
  # there. RAM-backed staging is also what removes the memmap-read cost from
  # the sampler's gather, which is the term most likely to starve the chip.
  # (This is tpu_train_s2.sh's own fallback, unchanged — the fix E-051's
  # node 1 bought at ~$0.20.)
  RAM_GB="$(awk '/MemTotal/{printf "%.0f", $2/1048576}' /proc/meminfo)"
  if [ "${RAM_GB}" -ge 160 ]; then
    echo "boot disk short (${AVAIL_GB} GB < ${NEED_GB}) but host has" \
         "${RAM_GB} GB RAM — falling back to tmpfs"
    mount -o remount,size=170G /dev/shm \
      || warn "remount of /dev/shm failed — the size check below decides whether that was fatal"
    WORK=/dev/shm/earth-cone
    OUT="${WORK}/run"
    mkdir -p "${WORK}" "${OUT}" "${WORK}/cache"
    AVAIL_GB="$(df -BG --output=avail "${WORK}" | tail -1 | tr -dc '0-9')"
    echo "measured: ${AVAIL_GB} GB free on ${WORK} (tmpfs)"
  fi
fi
if [ "${AVAIL_GB}" -lt "${NEED_GB}" ]; then
  echo "REFUSING: ${AVAIL_GB} GB free on ${WORK}, need ~${NEED_GB}. The r3" \
       "pentad tensor archive is ~11.5 GB and train_cone.py's load_data" \
       "writes an ANOMALY SCRATCH COPY of the 42-channel tensor beside it," \
       "~36 GB. Create the node with a larger boot disk."
  exit 1
fi

# --------------------------------------------------------------------------
# 3 · dependencies
# --------------------------------------------------------------------------
echo "--- deps ---"
export DEBIAN_FRONTEND=noninteractive
# unattended-upgrades holds the dpkg lock for MINUTES on first boot — measured
# 2026-08-24: it outlived all five 20 s retries, the script exited 1 two
# minutes in, and the self-reap correctly threw away a healthy node. Stop the
# service, then let apt WAIT on the lock instead of racing it; the retry loop
# stays for everything else apt can throw.
systemctl stop unattended-upgrades 2>&1 \
  || warn "unattended-upgrades stop failed — best effort: it may simply not exist on this image; the DPkg::Lock::Timeout below is the real fix"
systemctl kill --kill-who=all unattended-upgrades 2>&1 \
  || warn "unattended-upgrades kill failed — same reason as above"
for i in 1 2 3 4 5; do
  if apt-get -o DPkg::Lock::Timeout=600 update -qq && apt-get -o DPkg::Lock::Timeout=600 install -y -qq python3-venv; then
    echo "measured: python3-venv installed (attempt ${i})"; break
  fi
  if [ "${i}" = 5 ]; then echo "apt failed 5x — giving up"; exit 1; fi
  echo "apt attempt ${i} failed (boot lock?) — retrying in 60 s"; sleep 60
done
python3 -m venv "${WORK}/venv"
PY="${WORK}/venv/bin/python"
"${PY}" -m pip install --upgrade pip
"${PY}" -m pip install 'jax[tpu]' \
    -f https://storage.googleapis.com/jax-releases/libtpu_releases.html
# flax < 0.11 — the node image is Ubuntu 22.04 / Python 3.10 and flax 0.11
# requires 3.11 (HANDOVER §8.8). Pinning it here rather than discovering it
# from a pip resolver message eleven minutes into a $4.80/h node.
"${PY}" -m pip install 'flax<0.11' numpy optax
# CPU TORCH, IN BOTH MODES, and unlike the field launcher it is NOT optional:
# train_cone.py needs it on the healthy path for THREE things —
# `cone_from_torch` builds the JAX module from a torch `ConeMAE` constructed
# under `torch.manual_seed` (§8.4: initialisation is from the torch module,
# ALWAYS, so a same-seed twin starts from identical weights), `load_data`
# imports the shared numpy/torch plumbing rather than copying it, and the
# velocity probe's `kfold_r2` goes through `ml/probe_kfold.py`, which imports
# torch. A cone node without torch fails at step 0, or — worse — after the
# last step, in the probe.
"${PY}" -m pip install --index-url https://download.pytorch.org/whl/cpu torch
echo "measured: jax $("${PY}" -c 'import jax; print(jax.__version__)')" \
     "· flax $("${PY}" -c 'import flax; print(flax.__version__)')" \
     "· optax $("${PY}" -c 'import optax; print(optax.__version__)')" \
     "· torch $("${PY}" -c 'import torch; print(torch.__version__)')"
echo "measured: jax.devices() -> $("${PY}" -c 'import jax; print(jax.devices())')"

# --------------------------------------------------------------------------
# 4 · the code, at a pinned sha
# --------------------------------------------------------------------------
echo "--- clone ---"
if [ -n "${GIT_SHA}" ]; then
  # A --depth 1 clone cannot check out an arbitrary sha, so fetch that one
  # commit by name. This ASSERTS the sha it landed on rather than trusting the
  # fetch: a run whose code is not the code the dispatch names is a run that
  # cannot be compared to anything.
  git init -q "${WORK}/earth"
  git -C "${WORK}/earth" remote add origin "https://github.com/${REPO}.git"
  git -C "${WORK}/earth" fetch -q --depth 1 origin "${GIT_SHA}"
  git -C "${WORK}/earth" checkout -q FETCH_HEAD
  GOT_SHA="$(git -C "${WORK}/earth" rev-parse HEAD)"
  if [ "${GOT_SHA}" != "${GIT_SHA}" ]; then
    echo "REFUSING: asked for ${GIT_SHA}, checked out ${GOT_SHA}."
    exit 1
  fi
else
  # Reachable only in verify mode — step 0b refuses an empty GIT_SHA in train.
  git clone --depth 1 "https://github.com/${REPO}.git" "${WORK}/earth"
fi
echo "measured: code at $(git -C "${WORK}/earth" rev-parse HEAD)" \
     "($(git -C "${WORK}/earth" log -1 --format=%cI)) — pin ${GIT_SHA:-<none: main HEAD, verify mode>}"

# --------------------------------------------------------------------------
# 5 · the data — staged from the bucket, verified, and said out loud
# --------------------------------------------------------------------------
# gcloud/gsutil when the image has them (they do resumable, parallel-sliced
# downloads), the metadata-token JSON API when it does not. Nothing here
# REQUIRES the SDK.
stage_object() {   # stage_object <object name> <local path> <label>
  local OBJ="${1}" DST="${2}" LABEL="${3}" WANT GOT VIA
  WANT="$(gcs_size "${OBJ}" || true)"
  if [ -z "${WANT}" ]; then
    echo "REFUSING: gs://${BUCKET}/${OBJ} vanished between step 1 and now."
    exit 1
  fi
  echo "staging ${LABEL}: gs://${BUCKET}/${OBJ} -> ${DST} (${WANT} B) …"
  if command -v gcloud >/dev/null 2>&1 && gcloud storage cp "gs://${BUCKET}/${OBJ}" "${DST}"; then
    VIA="gcloud storage"
  elif command -v gsutil >/dev/null 2>&1 && gsutil -q cp "gs://${BUCKET}/${OBJ}" "${DST}"; then
    VIA="gsutil"
  elif gcs_get "${OBJ}" "${DST}"; then
    VIA="curl+JSON API"
  else
    echo "REFUSING: could not stage gs://${BUCKET}/${OBJ} by any route" \
         "(gcloud, gsutil, curl). The node's service account needs" \
         "storage.objects.get on this bucket."
    rm -f "${DST}"
    exit 1
  fi
  # ASSERT THE EFFECT, NOT THE INVOCATION (ml/CLAUDE.md §0.2): a cp that
  # returns 0 has told you about a process, not about a file.
  GOT="$(stat -c %s "${DST}")"
  if [ "${GOT}" = "0" ]; then
    echo "REFUSING: ${DST} staged as ZERO bytes via ${VIA}."
    exit 1
  fi
  if [ "${GOT}" != "${WANT}" ]; then
    echo "REFUSING: ${DST} is ${GOT} B but gs://${BUCKET}/${OBJ} declares" \
         "${WANT} B — a truncated stage is the failure that looks like a" \
         "model bug three hours later."
    exit 1
  fi
  # THE FULL sha256, and it is a REFUSAL, not a print. This is the one artefact
  # the whole experiment rests on, hashing 11.5 GB costs ~40 s of a $4.80/h
  # node (~$0.05), and the alternative is discovering at harvest time that the
  # bucket object under this name is not the tensor Phase A published. The
  # field launcher hashes only the first 1 MB because its Z is 16 GiB and has
  # its own structural verifier; the cone has neither excuse.
  GOT="$(sha256sum "${DST}" | cut -d' ' -f1)"
  if [ "${GOT}" != "${TENSOR_SHA}" ]; then
    echo "REFUSING: staged ${LABEL} sha256 ${GOT} != pin ${TENSOR_SHA}." \
         "The object at gs://${BUCKET}/${OBJ} is not the tensor this run" \
         "names. Nothing has been trained on it."
    exit 1
  fi
  echo "measured: ${LABEL} ${WANT} B ($(du -h "${DST}" | cut -f1)) via ${VIA} ·" \
       "sha256 VERIFIED ${TENSOR_SHA:0:10} in full"
}

# --------------------------------------------------------------------------
# 5b · the release fallback, transcribed from tpu_train_field.sh §5b
# --------------------------------------------------------------------------
# Used only when step 1 found the bucket object absent. It ENDS by publishing
# the assembled tensor to the bucket path that was missing, which is the whole
# reason it is worth having on a $4.80/h node: the assembly is an expensive
# intermediate and §5.26 says publish it, so the SECOND node pays one minute
# instead of the same five.
assemble_tensor() {   # assemble_tensor <destination path>
  local TF="${1}" GOT
  echo "--- tensor: assembling from data-cache-v1 ---"
  # The pin is verified in FULL here — not the first 1 MB — because this is
  # the one place the bytes are being created rather than copied, the release
  # publishes them in chunks that can be superseded, and a wrong tensor is
  # invisible until a k-fold moves by 0.041 (ml/CLAUDE.md §7).
  local sfx s
  for sfx in ${TENSOR_PARTS}; do
    echo "fetching ${TENSOR_NAME}_${TENSOR_SHA:0:10}.npz.${sfx} …"
    if ! curl -fsSL --max-time 1800 --retry 3 --retry-delay 10 \
         -o "${TF}.${sfx}.part" \
         "https://github.com/${REPO}/releases/download/data-cache-v1/${TENSOR_NAME}_${TENSOR_SHA:0:10}.npz.${sfx}"; then
      echo "REFUSING: neither route to the tensor is available — it is not at" \
           "gs://${BUCKET}/${GCS_TENSOR} and the release asset" \
           "${TENSOR_NAME}_${TENSOR_SHA:0:10}.npz.${sfx} could not be fetched" \
           "from data-cache-v1. Phase A publishes both; stage it by hand:"
      echo "  python3 scripts/tpu_box.py stage ml/cache/${TENSOR_NAME}.npz \\"
      echo "      gs://${BUCKET}/${GCS_TENSOR}"
      rm -f "${TF}".*.part
      exit 1
    fi
  done
  # shellcheck disable=SC2086
  cat $(for s in ${TENSOR_PARTS}; do printf '%s ' "${TF}.${s}.part"; done) > "${TF}.new"
  rm -f "${TF}".*.part
  GOT="$(sha256sum "${TF}.new" | cut -d' ' -f1)"
  if [ "${GOT}" != "${TENSOR_SHA}" ]; then
    echo "REFUSING: assembled tensor sha ${GOT} != pin ${TENSOR_SHA}."
    rm -f "${TF}.new"; exit 1
  fi
  mv "${TF}.new" "${TF}"
  echo "measured: tensor assembled, $(du -h "${TF}" | cut -f1), sha VERIFIED ${TENSOR_SHA:0:10}"
  # Publish only what verified. The `|| warn` is inside gcs_publish.
  gcs_publish "${TF}" "${GCS_TENSOR}" "tensor" || true
}

echo "--- staging ---"
TF="${WORK}/cache/$(basename "${GCS_TENSOR}")"
if [ "${TENSOR_ROUTE}" = "bucket" ]; then
  stage_object "${GCS_TENSOR}" "${TF}" "tensor"
else
  assemble_tensor "${TF}"
fi
echo "STAGED, and this is what every number below was computed on:"
echo "  tensor : gs://${BUCKET}/${GCS_TENSOR}  ->  ${TF}   [route: ${TENSOR_ROUTE}]"
echo "  sha256 : ${TENSOR_SHA}"

# --------------------------------------------------------------------------
# 6 · the trainer's argv, assembled ONCE
# --------------------------------------------------------------------------
# BOTH legs go through this function. The field launcher writes its 25-flag
# invocation out twice (det and diff) and the two can drift; ml/CLAUDE.md §1
# names copying exercises as the thing that produces #387 and #395, so the
# verify leg here differs from the train leg in exactly the three numbers it
# is allowed to differ in — steps, eval cadence, ckpt cadence — and in nothing
# else.
#
# EVERY KNOB IS PASSED EXPLICITLY, including the ones whose value equals
# train_cone.py's own default: a knob the trainer defaults silently is a knob
# the `resolved knobs` line above lies about, and that line is the one a
# reader trusts (SPOT_LEDGER 08-28 — a silently reverted knob block cost 62
# minutes and a wrong experiment).
run_trainer() {   # run_trainer <steps> <eval_every> <ckpt_every> <out dir>
  local R_STEPS="${1}" R_EVAL="${2}" R_CKPT="${3}" R_OUT="${4}"
  local PROBE_ARG="" SNAP_ARG=""
  if [ "${VELOCITY_PROBE}" = "1" ]; then PROBE_ARG="--velocity-probe"; fi
  if [ "${SNAPSHOT_ABLATION}" = "1" ]; then SNAP_ARG="--snapshot-ablation"; fi
  echo "invoking: train_cone.py --steps ${R_STEPS} --eval-every ${R_EVAL}" \
       "--ckpt-every ${R_CKPT} --out ${R_OUT} · probe '${PROBE_ARG}'" \
       "· snapshot '${SNAP_ARG}' · resume '${RESUME_ARG}' · tag '${TAG}'"
  # shellcheck disable=SC2086
  RECIPE_NAME="${TAG}" "${PY}" -u ml/jaxport/train_cone.py \
    --tensor "${TF}" \
    --holdout-scope window \
    --holdout-years "${HOLDOUT_YEARS}" \
    --steps "${R_STEPS}" \
    --batch "${BATCH}" \
    --lr "${LR}" \
    --seed "${SEED}" \
    --d-model "${D_MODEL}" \
    --n-heads "${HEADS}" \
    --n-latents "${N_LATENTS}" \
    --n-layers "${LAYERS}" \
    --d-z "${D_Z}" \
    --d-dec "${D_DEC}" \
    --dec-layers "${DEC_LAYERS}" \
    --L-in "${L_IN}" \
    --future-lags "${FUTURE_LAGS}" \
    --n-dot-queries "${DOT_QUERIES}" \
    --aux-latent-w "${AUX_W}" \
    --eval-every "${R_EVAL}" \
    --ckpt-every "${R_CKPT}" \
    --out "${R_OUT}" \
    --metrics metrics.jsonl \
    ${PROBE_ARG} \
    ${SNAP_ARG} \
    ${RESUME_ARG} \
    ${EXTRA_ARGS}
}

# The s/step the launch AFTER this one should put in STEP_EST_S. Computed
# from the trainer's OWN `wall_s` records — the held-out eval lines, which
# carry seconds since the first eval — not from the shell's clock around the
# whole leg: the shell's number includes the anomaly transform, the XLA
# compile and the certificate, and would over-state the pace of a 20,000-step
# run by minutes per thousand steps. Both are printed; the metrics one is the
# one the header's arithmetic wants.
measured_s_per_step() {   # measured_s_per_step <metrics.jsonl>
  "${PY}" - "${1}" <<'PYEOF'
import json, sys
pts = []
try:
    with open(sys.argv[1]) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except Exception:
                continue
            if isinstance(r, dict) and isinstance(r.get("step"), int) \
                    and isinstance(r.get("wall_s"), (int, float)):
                pts.append((r["step"], float(r["wall_s"])))
except OSError as e:
    print("measured: s/step UNAVAILABLE — %s" % e)
    raise SystemExit(0)
if len(pts) < 2:
    print("measured: s/step UNAVAILABLE — %d wall_s record(s) in %s; the leg "
          "did not reach its second eval, so there is nothing to divide"
          % (len(pts), sys.argv[1]))
    raise SystemExit(0)
(s0, w0), (s1, w1) = pts[0], pts[-1]
if s1 <= s0:
    print("measured: s/step UNAVAILABLE — the last wall_s record is at step "
          "%d, not after the first (%d)" % (s1, s0))
    raise SystemExit(0)
sps = (w1 - w0) / (s1 - s0)
print("measured: %.4f s/step (from metrics.jsonl wall_s: (%.1f - %.1f) s over "
      "steps %d -> %d, i.e. training plus its evals and checkpoints and NOT "
      "the one-off setup). Put this in STEP_EST_S for the train launch; the "
      "header's stall arithmetic is written against it." % (sps, w1, w0, s0, s1))
PYEOF
}

cd "${WORK}/earth"

# --------------------------------------------------------------------------
# 7 · VERIFY MODE — gates, then a real-data smoke, then a report, then out
# --------------------------------------------------------------------------
RESUME_ARG=""
if [ "${MODE}" = "verify" ]; then
  echo "--- verify: parity gates ---"
  GATE_NAMES=(tests/test_jaxport_cone.py)
  GATE_RC=()
  GATE_WALL=()
  GATE_FAILED=0
  for G in "${GATE_NAMES[@]}"; do
    echo "--- gate ${G} ---"
    T0="$(date +%s)"
    # The rc is CAPTURED, not fatal: a verify run's job is to report on every
    # gate, and aborting at the first failure would ship a report that cannot
    # say whether the rest pass. The exit status of this script carries the
    # verdict, below.
    #
    # JAX_PLATFORMS=cpu: the parity gates certify the PORT — torch and JAX
    # computing the same numbers from the same code — and that comparison is
    # only defined with both frameworks on the same device. Left to itself on
    # this host, JAX takes the TPU and its default-precision (bf16-multiply)
    # matmuls, and a 1e-5 gate reads device numerics as a port failure; that
    # cost two verify verdicts on 2026-08-26. The TPU is exercised where it is
    # the subject: the smoke leg below.
    if JAX_PLATFORMS=cpu "${PY}" -u "${G}"; then RC=0; else RC=$?; fi
    T1="$(date +%s)"
    GATE_RC+=("${RC}")
    GATE_WALL+=("$(( T1 - T0 ))")
    if [ "${RC}" = "0" ]; then
      echo "GATE PASS: ${G} ($(( T1 - T0 )) s)"
    else
      echo "GATE FAIL: ${G} rc ${RC} ($(( T1 - T0 )) s)"
      GATE_FAILED=1
    fi
  done

  echo "--- verify: real-data smoke at the REAL geometry ---"
  # VERIFY_STEPS steps at the REAL d_model/layers/L_in/batch against the REAL
  # staged tensor, with eval-every == steps so exactly one eval happens at the
  # end (train_cone.py evaluates at step 0 as well, and that PAIR is what
  # measured_s_per_step divides). A smoke at a toy shape would measure a toy.
  VERIFY_OUT="${OUT}/verify"
  mkdir -p "${VERIFY_OUT}"
  T0="$(date +%s)"
  if run_trainer "${VERIFY_STEPS}" "${VERIFY_STEPS}" "${VERIFY_STEPS}" "${VERIFY_OUT}"; then
    VER_RC=0
  else
    VER_RC=$?
  fi
  T1="$(date +%s)"
  VER_WALL="$(( T1 - T0 ))"
  echo "verify leg: rc ${VER_RC}, ${VER_WALL} s for ${VERIFY_STEPS} steps" \
       "(all-in, including the anomaly transform, the XLA compile and the" \
       "certificate: $(python3 -c "print('%.4f' % (${VER_WALL} / ${VERIFY_STEPS}))") s/step all-in)"
  # THE LINE THIS LEG EXISTS FOR.
  measured_s_per_step "${VERIFY_OUT}/metrics.jsonl"

  echo "--- verify: report ---"
  # The report is assembled by python because it embeds JSON the trainer
  # wrote; the shell's job is to hand over paths and numbers, not to quote
  # JSON by hand. Everything it prints came from a file on this node.
  CODE_SHA="$(git -C "${WORK}/earth" rev-parse HEAD)"
  export RPT_STAMP="${STAMP}" RPT_NODE="${NODE_NAME}" RPT_BUCKET="${BUCKET}" \
         RPT_TENSOR="${GCS_TENSOR}" RPT_TENSOR_SHA="${TENSOR_SHA}" \
         RPT_SHA="${CODE_SHA}" \
         RPT_GATE_NAMES="${GATE_NAMES[*]}" RPT_GATE_RC="${GATE_RC[*]}" \
         RPT_GATE_WALL="${GATE_WALL[*]}" \
         RPT_RC="${VER_RC}" RPT_WALL="${VER_WALL}" RPT_STEPS="${VERIFY_STEPS}" \
         RPT_DIR="${VERIFY_OUT}" \
         RPT_DMODEL="${D_MODEL}" RPT_LAYERS="${LAYERS}" RPT_HEADS="${HEADS}" \
         RPT_LATENTS="${N_LATENTS}" RPT_DZ="${D_Z}" RPT_LIN="${L_IN}" \
         RPT_BATCH="${BATCH}" RPT_DOTQ="${DOT_QUERIES}" \
         RPT_OUT="${REPORT}"
  "${PY}" - <<'PYEOF'
import json, os

def jload(path, what):
    try:
        with open(path) as f:
            return json.load(f)
    except Exception as e:                      # a missing/partial file is a
        return {"_unreadable": f"{what}: {e}"}  # FINDING, not a crash

def records(path):
    out = []
    try:
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    out.append(json.loads(line))
                except Exception:
                    out.append({"_raw": line[:400]})
    except Exception as e:
        return [{"_unreadable": str(e)}]
    return out

d = os.environ["RPT_DIR"]
steps = int(os.environ["RPT_STEPS"])
wall = float(os.environ["RPT_WALL"])
recs = records(os.path.join(d, "metrics.jsonl"))
walls = [(r["step"], float(r["wall_s"])) for r in recs
         if isinstance(r, dict) and isinstance(r.get("step"), int)
         and isinstance(r.get("wall_s"), (int, float))]
sps = None
if len(walls) >= 2 and walls[-1][0] > walls[0][0]:
    sps = round((walls[-1][1] - walls[0][1]) / (walls[-1][0] - walls[0][0]), 4)
evals = [r for r in recs if isinstance(r, dict) and "held_out_nll" in r]
cfg = next((r["config"] for r in recs
            if isinstance(r, dict) and isinstance(r.get("config"), dict)), None)

names = os.environ["RPT_GATE_NAMES"].split()
rcs = [int(x) for x in os.environ["RPT_GATE_RC"].split()]
gwalls = [float(x) for x in os.environ["RPT_GATE_WALL"].split()]
gates = [{"test": n, "rc": r, "pass": r == 0, "wall_s": w}
         for n, r, w in zip(names, rcs, gwalls)]
rc = int(os.environ["RPT_RC"])
report = {
    "kind": "tpu_train_cone verify report",
    "stamp": os.environ["RPT_STAMP"],
    "node": os.environ["RPT_NODE"],
    "bucket": os.environ["RPT_BUCKET"],
    "code_sha": os.environ["RPT_SHA"],
    "staged": {
        "tensor": f"gs://{os.environ['RPT_BUCKET']}/{os.environ['RPT_TENSOR']}",
        "tensor_sha256": os.environ["RPT_TENSOR_SHA"],
    },
    "geometry": {
        "d_model": int(os.environ["RPT_DMODEL"]),
        "layers": int(os.environ["RPT_LAYERS"]),
        "heads": int(os.environ["RPT_HEADS"]),
        "n_latents": int(os.environ["RPT_LATENTS"]),
        "d_z": int(os.environ["RPT_DZ"]),
        "L_in": int(os.environ["RPT_LIN"]),
        "batch": int(os.environ["RPT_BATCH"]),
        "n_dot_queries": int(os.environ["RPT_DOTQ"]),
    },
    "gates": gates,
    "gates_pass": all(g["pass"] for g in gates),
    "smoke": {
        "rc": rc,
        "pass": rc == 0,
        "steps": steps,
        "wall_s": wall,
        # The number the NEXT launch sets STEP_EST_S from. Two of them, and
        # they mean different things: all-in includes the one-off setup this
        # leg pays once and a 20,000-step run also pays once, so it OVER-states
        # the long run's pace; the metrics one is measured between the
        # trainer's own wall_s records and is the honest per-step figure.
        "s_per_step_all_in": round(wall / steps, 4) if steps else None,
        "s_per_step_from_wall_s": sps,
        "config_record": cfg,
        "held_out_records": evals,
        "results_json": jload(os.path.join(d, "results.json"), "results.json"),
        "velocity_probe": jload(os.path.join(d, "velocity_probe.json"),
                                "velocity_probe.json"),
    },
    "verdict": ("PASS" if all(g["pass"] for g in gates) and rc == 0 else "FAIL"),
    "note": ("The smoke leg is a CODE-PATH assertion on real data at the real "
             "geometry, not a skill measurement: 300 steps measures that the "
             "stack runs and how fast, and nothing about the cone codec's "
             "embedding. If s_per_step_from_wall_s exceeds ~4 s the host "
             "gather is starving the chip — see the launcher header."),
}
with open(os.environ["RPT_OUT"], "w") as f:
    json.dump(report, f, indent=2, sort_keys=True, default=str)
print("verdict:", report["verdict"],
      "· gates", [g["pass"] for g in gates],
      "· s/step (wall_s)", sps,
      "· s/step (all-in)", report["smoke"]["s_per_step_all_in"])
PYEOF
  # Ship the report NOW, not only from the trap: the trap is the safety net,
  # and a safety net that is also the primary path has never been tested.
  ship_report
  if [ "${GATE_FAILED}" = "1" ] || [ "${VER_RC}" != "0" ]; then
    echo "=== tpu_train_cone VERIFY FAILED — see verify_report.json; the EXIT trap ships the log and deletes this node ==="
    exit 1
  fi
  echo "=== tpu_train_cone VERIFY PASSED — set STEP_EST_S from the" \
       "'measured: <s/step>' line above (the report's" \
       "s_per_step_from_wall_s), re-read the header's stall arithmetic, and" \
       "launch MODE=train with a pinned GIT_SHA. The EXIT trap now ships the" \
       "log and deletes this node ==="
  exit 0
fi

# --------------------------------------------------------------------------
# 8 · TRAIN MODE — resume from the bucket
# --------------------------------------------------------------------------
# The node is disposable; the run is not. Whatever gs://BUCKET/runs/NODE/
# holds is the newest state this run reached, because those objects are
# overwritten on every ship and nothing else writes them. A resume here is a
# TRUE continuation — the trainer restores optimiser moments, the cosine
# schedule position and both host RNG streams from ckpt_latest.npz — which is
# why relaunching under the same node name is the documented way to continue.
echo "--- resume ---"
if gcs_get "${RUNS_PREFIX}/${CKPT_NPZ_NAME}" "${OUT}/${CKPT_NPZ_NAME}"; then
  RESUME_ARG="--resume"
  echo "measured: pulled gs://${BUCKET}/${RUNS_PREFIX}/${CKPT_NPZ_NAME}" \
       "($(du -h "${OUT}/${CKPT_NPZ_NAME}" | cut -f1))"
  "${PY}" -c '
import sys, numpy as np
z = np.load(sys.argv[1], allow_pickle=False)
for k in ("_step", "step", "global_step"):
    if k in z.files:
        print("measured: checkpoint is at step", int(np.asarray(z[k]).reshape(-1)[0]))
        break
else:
    print("measured: checkpoint carries no step key; keys are", z.files[:12])' \
    "${OUT}/${CKPT_NPZ_NAME}" \
    || warn "could not read a step out of the checkpoint — a log line, not a gate; the trainer's own resume is what decides"
  if gcs_get "${RUNS_PREFIX}/${CKPT_PT_NAME}" "${OUT}/${CKPT_PT_NAME}"; then
    echo "measured: pulled the torch-loadable codec beside it"
  else
    # `curl -o` leaves a zero-byte file behind on a 404, and a zero-byte
    # artefact is worse than none — it would ship over the bucket copy.
    rm -f "${OUT}/${CKPT_PT_NAME}"
    warn "no ${CKPT_PT_NAME} in the bucket — harmless, the trainer rewrites it at the next checkpoint"
  fi
  if gcs_get "${RUNS_PREFIX}/metrics.jsonl" "${OUT}/metrics.jsonl"; then
    echo "measured: pulled metrics.jsonl ($(wc -l < "${OUT}/metrics.jsonl") lines) — the curve continues"
  else
    rm -f "${OUT}/metrics.jsonl"
    warn "no metrics.jsonl in the bucket yet — this run starts the curve"
  fi
  if gcs_get "${RUNS_PREFIX}/results.json" "${OUT}/results.json"; then
    echo "measured: pulled the previous results.json (in_progress expected)"
  else
    rm -f "${OUT}/results.json"
    warn "no results.json in the bucket yet — the trainer writes the first one at its first eval"
  fi
  if [ "${SNAPSHOT_ABLATION}" = "1" ]; then
    # The twin trains AFTER the cone arm, so on most resumes it has not
    # started and every one of these 404s. That is the ordinary case and the
    # zero-byte leftovers are removed the same way.
    if gcs_get "${RUNS_PREFIX}/${SNAP_NPZ_NAME}" "${OUT}/${SNAP_NPZ_NAME}"; then
      echo "measured: pulled the snapshot twin's checkpoint"
    else
      rm -f "${OUT}/${SNAP_NPZ_NAME}"
      warn "no ${SNAP_NPZ_NAME} in the bucket — the twin had not started; it trains fresh"
    fi
    if gcs_get "${RUNS_PREFIX}/${SNAP_METRICS_NAME}" "${OUT}/${SNAP_METRICS_NAME}"; then
      echo "measured: pulled the snapshot twin's metrics"
    else
      rm -f "${OUT}/${SNAP_METRICS_NAME}"
      warn "no ${SNAP_METRICS_NAME} in the bucket — the twin starts its own curve"
    fi
  fi
else
  rm -f "${OUT}/${CKPT_NPZ_NAME}"
  echo "measured: nothing at gs://${BUCKET}/${RUNS_PREFIX}/${CKPT_NPZ_NAME} —" \
       "this is a FRESH run, not a continuation. Say so in its doc string."
fi

# --------------------------------------------------------------------------
# 9 · the shipper and the progress watchdog
# --------------------------------------------------------------------------
touch "${LAST_CKPT_MARK}"
( while true; do sleep $(( SHIP_EVERY_MIN * 60 )); ship_state; upload_log; done ) &
disown
# RETIRE THE BEACON. From here the 10-minute shipper uploads the same log
# object; two uploaders racing on one name is how a half-written log gets
# served to somebody diagnosing a node.
kill "${BEACON_PID}" 2>/dev/null \
  || warn "the 3-min beacon shipper was already gone — harmless: the 10-min shipper now owns the log object"
echo "measured: shipper armed, every ${SHIP_EVERY_MIN} min to gs://${BUCKET}/${RUNS_PREFIX}/ · 3-min beacon retired"

# Watchdog A: the PROGRESS watchdog. It is the only monitor that can tell a
# wedged trainer from a healthy one — both hold the chips at 100% — and it
# watches the SHIPPED checkpoint, not the local file, so a node that trains
# fine but cannot reach the bucket is also caught.
(
  while true; do
    sleep 300
    AGE=$(( $(date +%s) - $(stat -c %Y "${LAST_CKPT_MARK}") ))
    if [ "${AGE}" -gt $(( STALL_MIN * 60 )) ]; then
      echo "PROGRESS WATCHDOG: no new checkpoint object has landed in" \
           "gs://${BUCKET}/${RUNS_PREFIX}/ for $(( AGE / 60 )) min" \
           "(limit ${STALL_MIN}) — the trainer is stalled and the chips are" \
           "billing. Reaping."
      ship_state
      upload_log
      self_delete
      exit 0
    fi
  done
) &
disown
echo "measured: progress watchdog armed at ${STALL_MIN} min"

# --------------------------------------------------------------------------
# 10 · train
# --------------------------------------------------------------------------
echo "--- train ---"
run_trainer "${STEPS}" "${EVAL_EVERY}" "${CKPT_EVERY}" "${OUT}"

# THE EXIT PATH WHEN THE TRAINER ENDS, and a post-loop final save
# (SPOT_LEDGER 08-27: "E-054a's node idled ~55 min after its trainer exited
# because the self-reap never fired — the launch script sat in its ship loop;
# ~$1.5 of idle spot. Fix owed: an exit path when the trainer ends + a
# post-loop final save."). Two things close it here, and both are needed:
# the trainer is the LAST FOREGROUND COMMAND, so its return falls straight
# off the end of the script into the EXIT trap rather than into a ship loop;
# and this explicit ship_state is the post-loop final save, on the primary
# path rather than only in the trap — velocity_probe.json in particular is
# written after the last step and would otherwise reach the bucket only from
# the safety net.
ship_state
upload_log
echo "=== tpu_train_cone done — final state shipped; the EXIT trap now ships the log and deletes this node ==="

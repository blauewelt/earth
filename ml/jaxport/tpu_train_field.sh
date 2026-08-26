#!/bin/bash
# The startup script for E-052's FIELD-HEAD runs on a Cloud TPU node
# (ml/plans/E052_field_diffusion.md; JAX port tier 3b). Modelled line for line
# on tpu_train_s2.sh — same lifecycle, same self-reap, same EXIT trap, same
# disk guard — with three differences that are the whole point of this file:
#
#   1. It drives `ml/jaxport/train_field.py`, not train_stage2.py, so the
#      artefacts it protects are `ckpt_latest.npz` + `field_latest.pt` +
#      `metrics.jsonl` + an `in_progress` `results.json` (ml/CLAUDE.md §5.25),
#      not `temporal_*_jax.npz`.
#   2. It STAGES FROM THE BUCKET, not from GitHub releases. The tensor and the
#      Z are single GCS objects named by env, pulled with gcloud/gsutil when
#      the image has them and with the metadata-token JSON API when it does
#      not. See "THE OBJECTS THIS EXPECTS" below — this is the part to read
#      before launching.
#   3. It has TWO MODES. `MODE=verify` proves the deployed stack end to end on
#      real chips and real data for a few dollars; `MODE=train` is the long
#      run. Verify is the DEFAULT, because the cheap mistake is running the
#      cheap thing twice and the expensive one is a 30 h node that dies at
#      hour 29 on a typo.
#
#   sed -e 's|__BUCKET__|earth-tpu-staging|' \
#       -e 's|__NODE__|e052-field-verify|' \
#       -e 's|__TPUZONE__|us-west1-c|' \
#       ml/jaxport/tpu_train_field.sh > /tmp/f.sh
#   python3 scripts/tpu_box.py create e052-field-verify \
#       --accelerator-type v5litepod-4 --startup-file /tmp/f.sh
#
# A STARTUP SCRIPT INHERITS NO ENVIRONMENT FROM THE MACHINE THAT LAUNCHED IT.
# The `${K:-144}` forms below therefore always take their DEFAULTS on a node;
# to change a knob for a real run, sed it in the same call as __BUCKET__:
#
#       -e 's|^MODE=.*|MODE="train"|' -e 's|^STEPS=.*|STEPS="200000"|'
#
# and read the launch log back: the script prints every knob it resolved.
#
# ──────────────────────────────────────────────────────────────────────────
# THERE IS NO CHEAP STOPPED STATE. DELETE IS THE NORMAL END. Four exits, the
# first three exactly tpu_train_s2.sh's: (1) the run finishes (or verify
# finishes) → EXIT trap → ship → self-delete; (2) the run STALLS → progress
# watchdog, no NEW checkpoint object in the bucket for STALL_MIN minutes →
# reap, which is the only monitor that can tell a wedged trainer from a
# healthy one (both hold the chips at 100%); (3) MAX_HOURS, unconditional;
# (4) in verify mode the cap is VERIFY_MAX_HOURS, because "a few dollars" is a
# promise the node has to keep on its own.
#
# To CONTINUE a run, relaunch a node with the SAME __NODE__: the node name is
# the run's identity and step 6 resumes from whatever that bucket prefix holds
# (optimiser state and schedule position included — a true continuation, not a
# warm start).
#
# ──────────────────────────────────────────────────────────────────────────
# THE OBJECTS THIS EXPECTS, AND WHAT IS ACTUALLY IN THE BUCKET TODAY.
#
# MEASURED 2026-08-26 ~06:30Z, by listing gs://earth-tpu-staging through the
# storage JSON API rather than by assuming: the bucket holds 39 objects and
# EVERY ONE of them is under `runs/`, `tpu_smoke/`, `tpu_smoke_train/` or
# `tpu_smoke_logs/`. There is no `tensors/` prefix. **No tensor and no Z have
# ever been staged there** — E-051 pulled both from GitHub releases
# (`data-cache-v1` chunks + `embed-cache-v1` chunks, assembled on-node).
#
# So the two default object names below are NOT invented and they are also NOT
# yet real. Their FILE NAMES are the repo's own pins:
#
#   · `family4_na025_pentad_r2_37e146384b.npz` — tpu_train_s2.sh's
#     TENSOR_NAME `family4_na025_pentad_r2` with the first ten hex of its
#     pinned TENSOR_SHA `37e146384b6f622f…`, i.e. the exact name the
#     data-cache-v1 chunks carry (`…npz.aa/ab/ac`), un-chunked.
#   · `Z_8b639abe36_37e146384b.npy` — the published clean pentad Z that
#     E-051's dispatch entry names (`ml/EXPERIMENTS.md` §E-051: "Z PULLED from
#     the published clean Z_8b639abe36_37e146384b"), keyed (codec weight hash,
#     tensor sha) as embed_cache_sync does. 16.24 GiB, (3142, 86698, 32) f16.
#
# and their PREFIX is `tpu_box.py`'s own documented staging convention
# (`gs://<bucket>/tensors/family3_na025.npy`, module docstring / --help).
#
# THE DECISION FLOW WHEN AN OBJECT IS ABSENT, which is the normal case today:
#
#   object present in the bucket   → stage it (minutes: same-cloud bytes)
#   object absent, release present → ASSEMBLE it from the GitHub releases the
#                                    way tpu_train_s2.sh does — full-sha256
#                                    verify for the tensor, HEADER-BOUNDED
#                                    assembly + `embed_cache_sync.verify`
#                                    against the tensor's own time axis for
#                                    the Z — and then PUBLISH the assembled
#                                    artefact to the default GCS path, so the
#                                    next node stages in minutes instead of
#                                    paying the same ~40 minutes again
#                                    (ml/CLAUDE.md §5.26: publish the
#                                    expensive intermediate).
#   both absent                    → REFUSE, with the staging commands below.
#
# The publish is BEST-EFFORT and never fatal: the artefact is already on this
# node and correct, and a node that can train must not be killed by a bucket
# write. It is guarded — see `gcs_publish` — so a half-written object can
# never wear the real name.
#
# The manual route stays available and is still the cheapest when a box
# already holds the files (any Vast box that has run a pentad job):
#
#   python3 scripts/tpu_box.py stage ml/cache/family4_na025_pentad_r2.npz \
#       gs://earth-tpu-staging/tensors/family4_na025_pentad_r2_37e146384b.npz
#   python3 scripts/tpu_box.py stage ml/cache/Z_8b639abe36_37e146384b.npy \
#       gs://earth-tpu-staging/tensors/Z_8b639abe36_37e146384b.npy
#
# Override GCS_TENSOR / GCS_Z if you stage them under different names — the
# script echoes the full gs:// URI of everything it staged, loudly, twice, and
# says of each artefact whether it came from the bucket or from a release.
# `GCS_PIXELS` has NO release fallback (no pixels asset is published under any
# name this repo records), so an absent pixels object still refuses in step 1.
#
# ──────────────────────────────────────────────────────────────────────────
# THE STALL ARITHMETIC, WHICH IS WHY E-051 BURNED A NODE.
#
# `ml/handoffs/2026-08-26-e051-session.md`: "A run whose setup + first
# `--ckpt-every` steps exceed STALL_MIN self-reaps while healthy — check the
# arithmetic at every launch (node 4 cleared it: ~28 min vs 90)." The progress
# watchdog measures SHIPPED checkpoints, so the quantity that has to clear
# STALL_MIN is not the first checkpoint WRITE, it is the first checkpoint
# SHIP — one more SHIP_EVERY_MIN cycle behind it:
#
#   t_first_ship = SETUP_EST_MIN                 (apt + venv + jax/flax/optax
#                                                 + clone + trainer warm-up
#                                                 and XLA compile)
#                + (staged bytes / STAGE_MBPS)   (tensor + Z + pixels)
#                + CKPT_EVERY * STEP_EST_S       (steps to the first ckpt)
#                + SHIP_EVERY_MIN                (worst-case wait for the
#                                                 shipper's next cycle)
#
# Step 1 computes that from the resolved config, echoes it against STALL_MIN,
# and REFUSES if it does not clear — `ALLOW_STALL_RISK=1` overrides, because
# an estimate is an estimate. STEP_EST_S defaults to 0.30 s/step, which is
# E-051's MEASURED all-in 0.275 s/step at K=144 on this same v5litepod-4 with
# a 206.5M stage-2 head — a stand-in from a different model, stated as one.
# **Verify mode measures the field head's own s/step and puts it in
# verify_report.json; set STEP_EST_S from that number for the train launch.**
# ──────────────────────────────────────────────────────────────────────────
set -euo pipefail

# --------------------------------------------------------------------------
# knobs — every one of them is echoed in step 0 before anything is spent
# --------------------------------------------------------------------------
BUCKET="__BUCKET__"                 # substituted at launch
NODE_NAME="__NODE__"
TPUZONE="__TPUZONE__"
REPO="blauewelt/earth"
WORK="/opt/earth-field"
OUT="${WORK}/run"
CKPT_DIR="${OUT}/ckpt"
RUNS_PREFIX="runs/${NODE_NAME}"

# verify | train. See the header. Verify is the default on purpose.
MODE="${MODE:-verify}"

# THE CODE, pinned. Empty = whatever main's HEAD is at launch, which is the
# right default for a verify run (the point is to test what is deployed) and
# the wrong one for a long train (a run whose code moved under it is not
# comparable to anything). Set GIT_SHA for every train launch.
GIT_SHA="${GIT_SHA:-}"

# THE DATA, by GCS object name (bucket-relative, no gs:// prefix). Defaults
# and their provenance: see "THE OBJECTS THIS EXPECTS" above. GCS_PIXELS is
# optional — train_field.py only needs `--pixels` when the tensor npz does not
# carry the (ys, xs) pixel index itself.
GCS_TENSOR="${GCS_TENSOR:-tensors/family4_na025_pentad_r2_37e146384b.npz}"
GCS_Z="${GCS_Z:-tensors/Z_8b639abe36_37e146384b.npy}"
GCS_PIXELS="${GCS_PIXELS:-}"

# THE RELEASE-SIDE IDENTITY OF THE SAME TWO ARTEFACTS — the fallback route
# when the bucket object is absent, transcribed from tpu_train_s2.sh (§4)
# because it is the same pin: a run on different bytes is not comparable to
# anything (the box effect, ml/CLAUDE.md §7 — a differently-built tensor moved
# the head k-fold by 0.041 at a fixed seed), and the embed cache is KEYED by
# (codec weight hash, tensor sha256), so a node that assembles its own tensor
# from a different pin cannot use the published Z at all.
TENSOR_NAME="${TENSOR_NAME:-family4_na025_pentad_r2}"
TENSOR_SHA="${TENSOR_SHA:-37e146384b6f622fefe3c7e18ad9bab0389c9538be79536899fe8729bb2d0826}"
TENSOR_PARTS="${TENSOR_PARTS:-aa ab ac}"
Z_ASSET="${Z_ASSET:-Z_8b639abe36_37e146384b.npy}"

# Sizes used ONLY by step 1's arithmetic, for an artefact that is not in the
# bucket and therefore has no declared size to read. Provenance: the pentad
# tensor archive is "~11 GB" and the published pentad Z "16.24 GiB" in
# tpu_train_s2.sh's own disk-guard comment and in E-051's launch addendum
# ((3142, 86698, 32) float16 = 17,437,229,056 B with its 128-byte header).
EST_TENSOR_BYTES="${EST_TENSOR_BYTES:-11000000000}"
EST_Z_BYTES="${EST_Z_BYTES:-17437229056}"

# The run. Geometry defaults are the E-052 plan's first real arm (pentad,
# 720-day span = K 144 pentads, 4x4 patch tokens over the 0.25 deg window,
# ~5.3k tokens); the holdout years are the fleet's own blocked years, read off
# `ml/recipes/f3-anchor-41M-nolonhold.json`'s _provenance ('2009,2017,2023'),
# never typed from memory.
FIELD_MODE="${FIELD_MODE:-det}"          # the TRAINER's --mode: det | diff
HOLDOUT_YEARS="${HOLDOUT_YEARS:-2009,2017,2023}"
K="${K:-144}"
PATCH="${PATCH:-4}"
D_MODEL="${D_MODEL:-512}"
LAYERS="${LAYERS:-12}"
HEADS="${HEADS:-8}"
D_COND="${D_COND:-256}"
COND_LAYERS="${COND_LAYERS:-4}"
STEPS="${STEPS:-200000}"
BATCH="${BATCH:-8}"
LR="${LR:-3e-4}"
LR_SCHEDULE="${LR_SCHEDULE:-expdecay}"
LR_HALFLIFE="${LR_HALFLIFE:-40000}"
LR_WARMUP="${LR_WARMUP:-2000}"
GRAD_CLIP="${GRAD_CLIP:-128}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0}"
SEED="${SEED:-0}"
INPUT_ZNOISE="${INPUT_ZNOISE:-0.7}"
EVAL_EVERY="${EVAL_EVERY:-2000}"
EVAL_WINDOWS="${EVAL_WINDOWS:-16}"
NFE="${NFE:-18}"
MEMBERS="${MEMBERS:-8}"
CKPT_EVERY="${CKPT_EVERY:-2000}"
COND_CHUNK="${COND_CHUNK:-}"             # empty = the trainer's own default
EXTRA_ARGS="${EXTRA_ARGS:-}"

# Verify mode's own budget. The det leg is VERIFY_STEPS at the REAL K, batch
# and geometry — an s/step measured at a toy shape would be a number about a
# toy — and the diff leg is deliberately fixed at 300 steps plus ONE sampled
# eval, which is what exercises the sampler without paying for it.
VERIFY_STEPS="${VERIFY_STEPS:-300}"
VERIFY_DIFF_STEPS="${VERIFY_DIFF_STEPS:-300}"
VERIFY_MEMBERS="${VERIFY_MEMBERS:-4}"
VERIFY_NFE="${VERIFY_NFE:-12}"
VERIFY_EVAL_WINDOWS="${VERIFY_EVAL_WINDOWS:-8}"
VERIFY_MAX_HOURS="${VERIFY_MAX_HOURS:-3}"

# Lifecycle.
SHIP_EVERY_MIN="${SHIP_EVERY_MIN:-10}"   # upload cadence (train mode)
STALL_MIN="${STALL_MIN:-90}"             # progress watchdog (train mode)
MAX_HOURS="${MAX_HOURS:-30}"             # unconditional cap (train mode)

# Inputs to the stall arithmetic in step 1. All three are ESTIMATES and the
# echo says so; STEP_EST_S's provenance is in the header.
SETUP_EST_MIN="${SETUP_EST_MIN:-12}"     # apt+venv+pip+clone+XLA compile
STAGE_MBPS="${STAGE_MBPS:-200}"          # GCS <-> TPU VM, MB/s, conservative
# GitHub release CDN -> TPU VM, MB/s. E-051's node 3 went from boot 21:07Z to
# training ~21:30Z with ~27 GB of release assets to pull plus apt, pip and a
# clone, which bounds the aggregate at ≳20 MB/s; 40 is a middle estimate and
# it is used for ONE purpose — deciding whether an assembling node would be
# reaped by its own watchdog before it ever checkpoints.
RELEASE_MBPS="${RELEASE_MBPS:-40}"
STEP_EST_S="${STEP_EST_S:-0.30}"
ALLOW_STALL_RISK="${ALLOW_STALL_RISK:-0}"

# The disk guard. Sized from the allocation it guards (ml/CLAUDE.md §5.18),
# and the allocation is the same pentad one tpu_train_s2.sh guards: the
# tensor archive ~11 GB and ~34 GB decompressed, the Z 16.24 GiB, a rolling
# checkpoint pair beside them, and the venv's jax/torch wheels.
NEED_GB="${NEED_GB:-120}"

STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
LOG=/tmp/train_field.log
LAST_CKPT_MARK=/tmp/last_ckpt_upload     # touched when a NEW ckpt is SHIPPED
SIGDIR=/tmp/shipsig                      # one file per shipped object
CKPT_NPZ_NAME="ckpt_latest.npz"
CKPT_PT_NAME="field_latest.pt"
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
  local T BODY
  T="$(token)" || return 1
  BODY="$(curl -sS -H "Authorization: Bearer ${T}" \
    "https://storage.googleapis.com/storage/v1/b/${BUCKET}/o/$(
       python3 -c 'import sys,urllib.parse; print(urllib.parse.quote(sys.argv[1], safe=""))' "${1}")?fields=size")" \
    || return 1
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
# delete the temp. Both mechanisms the brief allows are in there, and the
# reason for the two-step rather than the size check alone is that the size
# check can only be made AFTER the upload finishes: a plain upload straight to
# the final name would make that name exist, and be stageable by a concurrent
# node, for the whole minutes-long window in which its bytes are unverified.
# With the temp name, the real name comes into existence only after the
# assertion, in one server-side operation that moves no bytes. On a mismatch
# the temp is DELETED rather than left as litter that a future reader has to
# decide about (§5.21's flush-then-mark, in its object-store form).
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
  # media POST of a 16 GiB object loses everything to one dropped connection.
  # The curl path stays as the fallback so nothing here REQUIRES the SDK.
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
  # able to see the numbers a long job has already computed.
  ship_file "${OUT}/metrics.jsonl" "${RUNS_PREFIX}/metrics.jsonl" \
    "application/x-ndjson" "metrics" || true
  ship_file "${OUT}/results.json" "${RUNS_PREFIX}/results.json" \
    "application/json" "results" || true
  # The CHECKPOINT, and the torch-loadable head WITH it, so the eval ladder
  # never has to wait for the run to end (JAX_PORT.md §1b). Only a NEW ckpt
  # moves the watchdog marker.
  if ship_file "${CKPT_DIR}/${CKPT_NPZ_NAME}" "${RUNS_PREFIX}/${CKPT_NPZ_NAME}" \
       "application/octet-stream" "ckpt"; then
    ship_file "${CKPT_DIR}/${CKPT_PT_NAME}" "${RUNS_PREFIX}/${CKPT_PT_NAME}" \
      "application/octet-stream" "ckptpt" || true
    touch "${LAST_CKPT_MARK}"
    echo "progress: NEW checkpoint landed in gs://${BUCKET}/${RUNS_PREFIX}/"
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

echo "=== tpu_train_field ${STAMP} · node ${NODE_NAME} · bucket ${BUCKET} · MODE ${MODE} ==="

# EVERY KNOB, RESOLVED, in one place. A startup script inherits no
# environment, so "I set STEPS=200000 when I launched it" is a claim about the
# launching shell and not about this node; this is the line that settles it.
echo "resolved knobs: mode ${MODE} · trainer-mode ${FIELD_MODE} · K ${K} ·" \
     "patch ${PATCH} · d_model ${D_MODEL} × layers ${LAYERS} × heads ${HEADS} ·" \
     "d_cond ${D_COND} cond_layers ${COND_LAYERS} cond_chunk '${COND_CHUNK}' ·" \
     "steps ${STEPS} × batch ${BATCH} · lr ${LR} sched ${LR_SCHEDULE}" \
     "halflife ${LR_HALFLIFE} warmup ${LR_WARMUP} · grad_clip ${GRAD_CLIP} ·" \
     "weight_decay ${WEIGHT_DECAY} · znoise ${INPUT_ZNOISE} · seed ${SEED} ·" \
     "holdout_years ${HOLDOUT_YEARS} · eval_every ${EVAL_EVERY}" \
     "eval_windows ${EVAL_WINDOWS} · nfe ${NFE} members ${MEMBERS} ·" \
     "ckpt_every ${CKPT_EVERY} · extra '${EXTRA_ARGS}'"
echo "resolved data: tensor gs://${BUCKET}/${GCS_TENSOR} · Z gs://${BUCKET}/${GCS_Z} ·" \
     "pixels '${GCS_PIXELS:-<none — the npz carries its own pixel index>}' ·" \
     "code ${REPO}@${GIT_SHA:-main HEAD at launch}"
echo "resolved release fallback (used only for an absent bucket object, and" \
     "the assembled file is published back to that path): tensor" \
     "${TENSOR_NAME}_${TENSOR_SHA:0:10}.npz parts '${TENSOR_PARTS}' from" \
     "data-cache-v1, sha pin ${TENSOR_SHA:0:10} · Z ${Z_ASSET} from" \
     "embed-cache-v1, header-bounded · release rate estimate ${RELEASE_MBPS} MB/s"
echo "resolved verify budget: det ${VERIFY_STEPS} steps + 1 eval · diff" \
     "${VERIFY_DIFF_STEPS} steps + 1 sampled eval at members ${VERIFY_MEMBERS}" \
     "nfe ${VERIFY_NFE} · eval_windows ${VERIFY_EVAL_WINDOWS} · cap ${VERIFY_MAX_HOURS} h"
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
# 1 · the STALL ARITHMETIC, before anything is spent
# --------------------------------------------------------------------------
# Read the header for the formula. This step also DECIDES THE ROUTE for each
# artefact — bucket or release-assembly — because the two cost wildly
# different amounts of wall clock and the watchdog does not care which one you
# are paying. Deciding it here, from three small metadata GETs, means the
# refusal case (neither route available for the pixels file) costs nothing.
echo "--- stall arithmetic ---"
STAGE_BYTES=0          # bytes to move GCS -> node
ASSEMBLE_BYTES=0       # bytes to pull from the releases AND push back
TENSOR_ROUTE="bucket"
Z_ROUTE="bucket"

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

SZ="$(gcs_size "${GCS_Z}" || true)"
if [ -n "${SZ}" ]; then
  echo "measured: Z gs://${BUCKET}/${GCS_Z} exists, ${SZ} B — staging route"
  STAGE_BYTES=$(( STAGE_BYTES + SZ ))
else
  Z_ROUTE="release"
  ASSEMBLE_BYTES=$(( ASSEMBLE_BYTES + EST_Z_BYTES ))
  echo "measured: Z gs://${BUCKET}/${GCS_Z} is ABSENT — falling back to" \
       "embed-cache-v1 (${Z_ASSET}, header-bounded assembly), then publishing" \
       "the assembled file back to that path"
fi

# The pixels file has no release fallback — no pixels asset is published under
# any name this repo records — so an absent one is a dead end, and a dead end
# is worth refusing at minute one rather than at minute forty.
if [ -n "${GCS_PIXELS}" ]; then
  SZ="$(gcs_size "${GCS_PIXELS}" || true)"
  if [ -z "${SZ}" ]; then
    echo "REFUSING: gs://${BUCKET}/${GCS_PIXELS} does not exist (or is" \
         "unreadable by this node's service account) and there is NO release" \
         "fallback for a pixels file. Nothing has been spent yet. Stage it:"
    echo "  python3 scripts/tpu_box.py stage ml/cache/<pixels>.npz \\"
    echo "      gs://${BUCKET}/${GCS_PIXELS}"
    echo "…or unset GCS_PIXELS if the tensor npz carries its own pixel index."
    exit 1
  fi
  echo "measured: pixels gs://${BUCKET}/${GCS_PIXELS} exists, ${SZ} B"
  STAGE_BYTES=$(( STAGE_BYTES + SZ ))
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
     "+ ${CKPT_EVERY} steps ${EST_TRAIN} (at ${STEP_EST_S} s/step, an" \
     "ESTIMATE — E-051's measured all-in pace at K=144 on this hardware, for" \
     "a different model) + shipper cycle ${SHIP_EVERY_MIN}." \
     "Routes: tensor ${TENSOR_ROUTE} · Z ${Z_ROUTE}." \
     "Watchdog limit STALL_MIN ${STALL_MIN} min."
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
           "the cheaper fix is to stage the artefacts once from a box that" \
           "already holds them — every later node then takes the bucket route." \
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
mkdir -p "${WORK}" "${OUT}" "${CKPT_DIR}" "${WORK}/cache"
echo "measured: $(nproc) CPUs, $(awk '/MemTotal/{printf "%.0f GB RAM", $2/1048576}' /proc/meminfo)"
AVAIL_GB="$(df -BG --output=avail "${WORK}" | tail -1 | tr -dc '0-9')"
echo "measured: ${AVAIL_GB} GB free on ${WORK}"
if [ "${AVAIL_GB}" -lt "${NEED_GB}" ]; then
  # The v5e boot disk serves ~90 GB free — short of the pentad allocation —
  # but the host carries 189 GB of RAM (measured, E-051 2026-08-25), so
  # before refusing, try tmpfs: remount /dev/shm large enough and move WORK
  # there. RAM-backed staging is also what removed the memmap-read cost from
  # the gather. Only then refuse. (This is tpu_train_s2.sh's own fallback,
  # unchanged — it is the fix E-051's node 1 bought at ~$0.20.)
  RAM_GB="$(awk '/MemTotal/{printf "%.0f", $2/1048576}' /proc/meminfo)"
  if [ "${RAM_GB}" -ge 160 ]; then
    echo "boot disk short (${AVAIL_GB} GB < ${NEED_GB}) but host has" \
         "${RAM_GB} GB RAM — falling back to tmpfs"
    mount -o remount,size=170G /dev/shm \
      || warn "remount of /dev/shm failed — the size check below decides whether that was fatal"
    WORK=/dev/shm/earth-field
    OUT="${WORK}/run"
    CKPT_DIR="${OUT}/ckpt"
    mkdir -p "${WORK}" "${OUT}" "${CKPT_DIR}" "${WORK}/cache"
    AVAIL_GB="$(df -BG --output=avail "${WORK}" | tail -1 | tr -dc '0-9')"
    echo "measured: ${AVAIL_GB} GB free on ${WORK} (tmpfs)"
  fi
fi
if [ "${AVAIL_GB}" -lt "${NEED_GB}" ]; then
  echo "REFUSING: ${AVAIL_GB} GB free on ${WORK}, need ~${NEED_GB}. The pentad" \
       "tensor archive is ~11 GB, decompressed ~34 GB, and the Z 16.24 GiB." \
       "Create the node with a larger boot disk."
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
"${PY}" -m pip install flax numpy optax
echo "measured: jax $("${PY}" -c 'import jax; print(jax.__version__)')" \
     "· flax $("${PY}" -c 'import flax; print(flax.__version__)')" \
     "· optax $("${PY}" -c 'import optax; print(optax.__version__)')"
echo "measured: jax.devices() -> $("${PY}" -c 'import jax; print(jax.devices())')"

# CPU TORCH, IN VERIFY MODE ONLY, and only because the PARITY GATES need it:
# tests/test_field_diffusion.py and tests/test_probscore.py score the JAX
# field head against `ml/field_model.py` / `ml/probscore.py`, which are torch
# and numpy. A train run never imports torch — the .pt export is written by
# `ml/jaxport/convert.py` — so paying ~2 minutes and ~800 MB for it there
# would be paying for nothing.
if [ "${MODE}" = "verify" ]; then
  "${PY}" -m pip install --index-url https://download.pytorch.org/whl/cpu torch
  echo "measured: torch $("${PY}" -c 'import torch; print(torch.__version__)') (CPU wheel, for the parity gates only)"
fi

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
  git clone --depth 1 "https://github.com/${REPO}.git" "${WORK}/earth"
fi
echo "measured: code at $(git -C "${WORK}/earth" rev-parse HEAD)" \
     "($(git -C "${WORK}/earth" log -1 --format=%cI)) — pin ${GIT_SHA:-<none: main HEAD>}"

# --------------------------------------------------------------------------
# 5 · the data — staged from the bucket, verified, and said out loud
# --------------------------------------------------------------------------
# gcloud/gsutil when the image has them (they do resumable, parallel-sliced
# downloads, which matters for a 16 GiB object), the metadata-token JSON API
# when it does not. tpu_train_s2.sh assumes only curl + python3 on this image
# and so does the fallback here — nothing in this script REQUIRES the SDK.
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
  # A cheap identity check: the sha256 of the first 1 MB. It is NOT a content
  # hash of a 16 GiB file and is not offered as one — hashing the whole Z
  # would cost minutes of a $4.80/h node every launch. What it IS: a value
  # that changes if somebody re-stages a different array under the same name,
  # printed here so two launches can be compared by eye and by grep.
  echo "measured: ${LABEL} ${GOT} B ($(du -h "${DST}" | cut -f1)) via ${VIA} ·" \
       "sha256(first 1MB) $(head -c 1048576 "${DST}" | sha256sum | cut -d' ' -f1)"
}

# --------------------------------------------------------------------------
# 5b · the release fallback, transcribed from tpu_train_s2.sh §4
# --------------------------------------------------------------------------
# Used only when step 1 found the bucket object absent. Both routines END by
# publishing the assembled artefact to the bucket path that was missing, which
# is the whole reason they are worth having on a $4.80/h node: the assembly is
# an expensive intermediate and §5.26 says publish it, so the SECOND node pays
# minutes instead of the same forty.

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
           "from data-cache-v1. Stage it by hand:"
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

assemble_z() {        # assemble_z <destination path> <tensor path>
  local ZF="${1}" TFP="${2}" WANT WANT_BYTES IDX SFX GOTB
  echo "--- Z: assembling from embed-cache-v1 (header-bounded) ---"
  # WHY HEADER-BOUNDED, transcribed from tpu_train_s2.sh's own block comment
  # because the bug it fixes is still live on the release:
  # `ml/embed_cache_sync.py:pull` concatenates chunks aa, ab, … until a fetch
  # 404s. Measured against embed-cache-v1 on 2026-08-24, `Z_8b639abe36_…npy`
  # has TWELVE chunk assets of which `.af` is a short 852,643,840 B in the
  # MIDDLE of the run — the tail of a 6-chunk publish whose header declares
  # (1571, 86698, 32), sitting on top of the orphaned ag..al tail of the
  # 12-chunk (3142, 86698, 32) cache. Concatenating to the first miss yields a
  # file that matches NEITHER publish and is correctly discarded by every
  # verifier. Reading the header FIRST and stopping at the byte count it
  # implies gets the intact file out, and refuses loudly if the bytes still do
  # not add up.
  echo "fetching ${Z_ASSET}.aa …"
  if ! curl -fsSL --max-time 1800 --retry 3 --retry-delay 10 \
       -o "${ZF}.aa.part" \
       "https://github.com/${REPO}/releases/download/embed-cache-v1/${Z_ASSET}.aa"; then
    echo "REFUSING: neither route to the Z is available — it is not at" \
         "gs://${BUCKET}/${GCS_Z} and ${Z_ASSET}.aa could not be fetched from" \
         "embed-cache-v1. Stage it by hand:"
    echo "  python3 scripts/tpu_box.py stage ml/cache/${Z_ASSET} \\"
    echo "      gs://${BUCKET}/${GCS_Z}"
    rm -f "${ZF}.aa.part"
    exit 1
  fi
  WANT="$("${PY}" -c '
import io, sys, numpy as np
with open(sys.argv[1], "rb") as f:
    head = f.read(1 << 16)
b = io.BytesIO(head)
major, minor = np.lib.format.read_magic(b)
reader = {(1, 0): np.lib.format.read_array_header_1_0,
          (2, 0): np.lib.format.read_array_header_2_0}[(major, minor)]
shape, _, dt = reader(b)
print(b.tell() + int(np.prod(shape)) * dt.itemsize, *shape, dt, sep=" ")' "${ZF}.aa.part")"
  WANT_BYTES="$(echo "${WANT}" | cut -d' ' -f1)"
  echo "measured: ${Z_ASSET} header declares ${WANT} -> ${WANT_BYTES} bytes"
  cp "${ZF}.aa.part" "${ZF}.new"
  rm -f "${ZF}.aa.part"
  IDX=1
  while [ "$(stat -c %s "${ZF}.new")" -lt "${WANT_BYTES}" ]; do
    SFX="$(printf "%s%s" \
      "$(printf "\\$(printf '%03o' $((97 + IDX / 26)))")" \
      "$(printf "\\$(printf '%03o' $((97 + IDX % 26)))")")"
    echo "fetching ${Z_ASSET}.${SFX} ($(stat -c %s "${ZF}.new") / ${WANT_BYTES} B) …"
    if ! curl -fsSL --max-time 1800 --retry 3 --retry-delay 10 \
         -o "${ZF}.part" \
         "https://github.com/${REPO}/releases/download/embed-cache-v1/${Z_ASSET}.${SFX}"; then
      echo "REFUSING: ${Z_ASSET}.${SFX} is absent but the header still wants" \
           "$(( WANT_BYTES - $(stat -c %s "${ZF}.new") )) more bytes. The" \
           "published cache is incomplete; stage a complete Z by hand:"
      echo "  python3 scripts/tpu_box.py stage ml/cache/${Z_ASSET} \\"
      echo "      gs://${BUCKET}/${GCS_Z}"
      rm -f "${ZF}.new" "${ZF}.part"; exit 1
    fi
    cat "${ZF}.part" >> "${ZF}.new"
    rm -f "${ZF}.part"
    IDX=$(( IDX + 1 ))
    if [ "${IDX}" -gt 64 ]; then
      echo "REFUSING: more than 64 chunks and still short of the header's" \
           "${WANT_BYTES} B."
      rm -f "${ZF}.new"; exit 1
    fi
  done
  GOTB="$(stat -c %s "${ZF}.new")"
  # TRUNCATE TO THE HEADER, and say so. The last chunk of a superseded larger
  # publish can overshoot; the header is the authority on where the array
  # ends, and a file longer than its own header is exactly the "reassembled
  # out of order" case embed_cache_sync.verify() refuses.
  if [ "${GOTB}" -gt "${WANT_BYTES}" ]; then
    echo "measured: assembled ${GOTB} B against the header's ${WANT_BYTES} B" \
         "— truncating the overshoot from the final chunk"
    truncate -s "${WANT_BYTES}" "${ZF}.new"
  fi
  mv "${ZF}.new" "${ZF}"
  echo "measured: Z assembled to $(du -h "${ZF}" | cut -f1) in ${IDX} chunk(s), header-bounded"
  # VERIFY BEFORE TRUSTING — and against the TENSOR'S OWN AXIS, not only
  # against the Z's own header. The truncate above bounds an assembled Z by
  # what its header claims, which is the right answer to a stale chunk tail
  # and no answer at all to a Z that was STRIDED before it was published
  # (#462): that file is internally consistent and simply covers one bin in
  # two. `tensor_t` reads ~128 bytes off the tensor this run is using.
  (cd "${WORK}/earth" && "${PY}" -c '
import sys
sys.path.insert(0, "ml")
from embed_cache_sync import tensor_t, verify
ok, why = verify(sys.argv[1], tensor_t(sys.argv[2]))
print(("Z VERIFIED: " if ok else "Z REJECTED: ") + why)
sys.exit(0 if ok else 1)' "${ZF}" "${TFP}") || {
    echo "REFUSING: the assembled Z does not verify against the tensor's own" \
         "time axis. Training on an embedding of unknown provenance is worse" \
         "than not training. Stage a known-good Z:"
    echo "  python3 scripts/tpu_box.py stage ml/cache/${Z_ASSET} \\"
    echo "      gs://${BUCKET}/${GCS_Z}"
    exit 1; }
  # Publish only what verified — a rejected Z is never uploaded, so the bucket
  # name can only ever hold a Z that passed embed_cache_sync.verify() on some
  # node. That is a stronger promise than the release itself makes.
  gcs_publish "${ZF}" "${GCS_Z}" "Z" || true
}

echo "--- staging ---"
TF="${WORK}/cache/$(basename "${GCS_TENSOR}")"
ZF="${WORK}/cache/$(basename "${GCS_Z}")"
# ORDER MATTERS: the Z's verification reads the tensor's time axis, so the
# tensor must exist locally first, whichever route each of them took.
if [ "${TENSOR_ROUTE}" = "bucket" ]; then
  stage_object "${GCS_TENSOR}" "${TF}" "tensor"
else
  assemble_tensor "${TF}"
fi
if [ "${Z_ROUTE}" = "bucket" ]; then
  stage_object "${GCS_Z}" "${ZF}" "Z"
else
  assemble_z "${ZF}" "${TF}"
fi
PIXELS_ARG=""
if [ -n "${GCS_PIXELS}" ]; then
  PF="${WORK}/cache/$(basename "${GCS_PIXELS}")"
  stage_object "${GCS_PIXELS}" "${PF}" "pixels"
  PIXELS_ARG="--pixels ${PF}"
fi
echo "STAGED, and this is what every number below was computed on:"
echo "  tensor : gs://${BUCKET}/${GCS_TENSOR}  ->  ${TF}   [route: ${TENSOR_ROUTE}]"
echo "  Z      : gs://${BUCKET}/${GCS_Z}  ->  ${ZF}   [route: ${Z_ROUTE}]"
echo "  pixels : ${GCS_PIXELS:-<none>}"

# --------------------------------------------------------------------------
# 6 · VERIFY MODE — gates, then a real-data smoke, then a report, then out
# --------------------------------------------------------------------------
cd "${WORK}/earth"

if [ "${MODE}" = "verify" ]; then
  echo "--- verify: parity gates ---"
  GATE_NAMES=(tests/test_jaxport_field.py tests/test_field_diffusion.py tests/test_probscore.py)
  GATE_RC=()
  GATE_WALL=()
  GATE_FAILED=0
  for G in "${GATE_NAMES[@]}"; do
    echo "--- gate ${G} ---"
    T0="$(date +%s)"
    # The rc is CAPTURED, not fatal: a verify run's job is to report on all
    # three gates, and aborting at the first failure would ship a report that
    # cannot say whether the other two pass. The exit status of this script
    # carries the verdict (step 6c).
    if "${PY}" -u "${G}"; then RC=0; else RC=$?; fi
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

  echo "--- verify: real-data smoke, det ---"
  # The det leg runs at the REAL K, batch and geometry against the REAL staged
  # Z, with eval-every == steps so exactly ONE eval happens, at the end. That
  # is what makes its s/step a number the train launch can set STEP_EST_S
  # from; a smoke at a toy shape would measure a toy.
  DET_OUT="${OUT}/verify_det"
  mkdir -p "${DET_OUT}"
  T0="$(date +%s)"
  # shellcheck disable=SC2086
  if "${PY}" -u ml/jaxport/train_field.py \
      --z-cache "${ZF}" \
      --data "${TF}" \
      ${PIXELS_ARG} \
      --holdout-years "${HOLDOUT_YEARS}" \
      --mode det \
      --K "${K}" \
      --patch "${PATCH}" \
      --d-model "${D_MODEL}" \
      --layers "${LAYERS}" \
      --heads "${HEADS}" \
      --d-cond "${D_COND}" \
      --cond-layers "${COND_LAYERS}" \
      --steps "${VERIFY_STEPS}" \
      --batch "${BATCH}" \
      --lr "${LR}" \
      --lr-schedule "${LR_SCHEDULE}" \
      --lr-halflife "${LR_HALFLIFE}" \
      --lr-warmup "${LR_WARMUP}" \
      --grad-clip "${GRAD_CLIP}" \
      --weight-decay "${WEIGHT_DECAY}" \
      --seed "${SEED}" \
      --input-znoise "${INPUT_ZNOISE}" \
      --eval-every "${VERIFY_STEPS}" \
      --eval-windows "${VERIFY_EVAL_WINDOWS}" \
      --ckpt-dir "${DET_OUT}/ckpt" \
      --ckpt-every "${VERIFY_STEPS}" \
      --out "${DET_OUT}/results.json" \
      --metrics "${DET_OUT}/metrics.jsonl" \
      ${COND_CHUNK:+--cond-chunk "${COND_CHUNK}"} \
      ${EXTRA_ARGS}; then DET_RC=0; else DET_RC=$?; fi
  T1="$(date +%s)"
  DET_WALL="$(( T1 - T0 ))"
  echo "verify det leg: rc ${DET_RC}, ${DET_WALL} s for ${VERIFY_STEPS} steps"

  echo "--- verify: real-data smoke, diff (sampled eval) ---"
  # The diff leg is the one that exercises the sampler: a short train plus ONE
  # sampled eval at a small member count and a short NFE ladder. It is not a
  # skill measurement and nothing in the report presents it as one.
  DIFF_OUT="${OUT}/verify_diff"
  mkdir -p "${DIFF_OUT}"
  T0="$(date +%s)"
  # shellcheck disable=SC2086
  if "${PY}" -u ml/jaxport/train_field.py \
      --z-cache "${ZF}" \
      --data "${TF}" \
      ${PIXELS_ARG} \
      --holdout-years "${HOLDOUT_YEARS}" \
      --mode diff \
      --K "${K}" \
      --patch "${PATCH}" \
      --d-model "${D_MODEL}" \
      --layers "${LAYERS}" \
      --heads "${HEADS}" \
      --d-cond "${D_COND}" \
      --cond-layers "${COND_LAYERS}" \
      --steps "${VERIFY_DIFF_STEPS}" \
      --batch "${BATCH}" \
      --lr "${LR}" \
      --lr-schedule "${LR_SCHEDULE}" \
      --lr-halflife "${LR_HALFLIFE}" \
      --lr-warmup "${LR_WARMUP}" \
      --grad-clip "${GRAD_CLIP}" \
      --weight-decay "${WEIGHT_DECAY}" \
      --seed "${SEED}" \
      --input-znoise "${INPUT_ZNOISE}" \
      --eval-every "${VERIFY_DIFF_STEPS}" \
      --eval-windows "${VERIFY_EVAL_WINDOWS}" \
      --nfe "${VERIFY_NFE}" \
      --members "${VERIFY_MEMBERS}" \
      --ckpt-dir "${DIFF_OUT}/ckpt" \
      --ckpt-every "${VERIFY_DIFF_STEPS}" \
      --out "${DIFF_OUT}/results.json" \
      --metrics "${DIFF_OUT}/metrics.jsonl" \
      ${COND_CHUNK:+--cond-chunk "${COND_CHUNK}"} \
      ${EXTRA_ARGS}; then DIFF_RC=0; else DIFF_RC=$?; fi
  T1="$(date +%s)"
  DIFF_WALL="$(( T1 - T0 ))"
  echo "verify diff leg: rc ${DIFF_RC}, ${DIFF_WALL} s for ${VERIFY_DIFF_STEPS} steps"

  echo "--- verify: report ---"
  # The report is assembled by python because it embeds JSON the trainer
  # wrote; the shell's job is to hand over paths and numbers, not to quote
  # JSON by hand. Everything it prints came from a file on this node.
  CODE_SHA="$(git -C "${WORK}/earth" rev-parse HEAD)"
  export RPT_STAMP="${STAMP}" RPT_NODE="${NODE_NAME}" RPT_BUCKET="${BUCKET}" \
         RPT_TENSOR="${GCS_TENSOR}" RPT_Z="${GCS_Z}" RPT_PIXELS="${GCS_PIXELS}" \
         RPT_SHA="${CODE_SHA}" \
         RPT_GATE_NAMES="${GATE_NAMES[*]}" RPT_GATE_RC="${GATE_RC[*]}" \
         RPT_GATE_WALL="${GATE_WALL[*]}" \
         RPT_DET_RC="${DET_RC}" RPT_DET_WALL="${DET_WALL}" RPT_DET_STEPS="${VERIFY_STEPS}" \
         RPT_DET_DIR="${DET_OUT}" \
         RPT_DIFF_RC="${DIFF_RC}" RPT_DIFF_WALL="${DIFF_WALL}" RPT_DIFF_STEPS="${VERIFY_DIFF_STEPS}" \
         RPT_DIFF_DIR="${DIFF_OUT}" \
         RPT_MEMBERS="${VERIFY_MEMBERS}" RPT_NFE="${VERIFY_NFE}" \
         RPT_K="${K}" RPT_PATCH="${PATCH}" RPT_DMODEL="${D_MODEL}" \
         RPT_LAYERS="${LAYERS}" RPT_HEADS="${HEADS}" RPT_BATCH="${BATCH}" \
         RPT_OUT="${REPORT}"
  "${PY}" - <<'PYEOF'
import json, os

def jload(path, what):
    try:
        with open(path) as f:
            return json.load(f)
    except Exception as e:                      # a missing/partial file is a
        return {"_unreadable": f"{what}: {e}"}  # FINDING, not a crash

def tail(path, n=8):
    try:
        with open(path) as f:
            lines = [l for l in f.read().splitlines() if l.strip()]
    except Exception as e:
        return [{"_unreadable": str(e)}]
    out = []
    for l in lines[-n:]:
        try:
            out.append(json.loads(l))
        except Exception:
            out.append({"_raw": l[:400]})
    return out

def leg(prefix):
    d = os.environ[f"RPT_{prefix}_DIR"]
    steps = int(os.environ[f"RPT_{prefix}_STEPS"])
    wall = float(os.environ[f"RPT_{prefix}_WALL"])
    rc = int(os.environ[f"RPT_{prefix}_RC"])
    res = jload(os.path.join(d, "results.json"), "results.json")
    recs = tail(os.path.join(d, "metrics.jsonl"))
    # the FINAL eval records, which is what a reader of this report wants:
    # every tail record that carries an eval-ish key, plus the results file.
    evals = [r for r in recs
             if any(k for k in r if "eval" in k or "val" in k or "ratio" in k
                    or "crps" in k)]
    return {
        "rc": rc,
        "pass": rc == 0,
        "steps": steps,
        "wall_s": wall,
        # s/step INCLUDES setup, compile and the single eval — it is the
        # number a launcher should put in STEP_EST_S, which is also the
        # pessimistic direction, so say what it contains.
        "s_per_step_all_in": round(wall / steps, 4) if steps else None,
        "results_json": res,
        "final_eval_records": evals or recs[-3:],
        "metrics_tail": recs,
    }

names = os.environ["RPT_GATE_NAMES"].split()
rcs = [int(x) for x in os.environ["RPT_GATE_RC"].split()]
walls = [float(x) for x in os.environ["RPT_GATE_WALL"].split()]
gates = [{"test": n, "rc": r, "pass": r == 0, "wall_s": w}
         for n, r, w in zip(names, rcs, walls)]

det, diff = leg("DET"), leg("DIFF")
report = {
    "kind": "tpu_train_field verify report",
    "stamp": os.environ["RPT_STAMP"],
    "node": os.environ["RPT_NODE"],
    "bucket": os.environ["RPT_BUCKET"],
    "code_sha": os.environ["RPT_SHA"],
    "staged": {
        "tensor": f"gs://{os.environ['RPT_BUCKET']}/{os.environ['RPT_TENSOR']}",
        "z": f"gs://{os.environ['RPT_BUCKET']}/{os.environ['RPT_Z']}",
        "pixels": os.environ["RPT_PIXELS"] or None,
    },
    "geometry": {
        "K": int(os.environ["RPT_K"]), "patch": int(os.environ["RPT_PATCH"]),
        "d_model": int(os.environ["RPT_DMODEL"]),
        "layers": int(os.environ["RPT_LAYERS"]),
        "heads": int(os.environ["RPT_HEADS"]),
        "batch": int(os.environ["RPT_BATCH"]),
    },
    "gates": gates,
    "gates_pass": all(g["pass"] for g in gates),
    "smoke_det": det,
    "smoke_diff": dict(diff, members=int(os.environ["RPT_MEMBERS"]),
                       nfe=int(os.environ["RPT_NFE"])),
    "verdict": ("PASS" if all(g["pass"] for g in gates)
                and det["pass"] and diff["pass"] else "FAIL"),
    "note": ("Smoke legs are CODE-PATH assertions on real data, not skill "
             "measurements: 300 steps measures that the stack runs and how "
             "fast, and nothing about the field head's forecast. "
             "s_per_step_all_in includes setup, XLA compile and one eval."),
}
with open(os.environ["RPT_OUT"], "w") as f:
    json.dump(report, f, indent=2, sort_keys=True, default=str)
print("verdict:", report["verdict"],
      "· gates", [g["pass"] for g in gates],
      "· det s/step", det["s_per_step_all_in"],
      "· diff s/step", diff["s_per_step_all_in"])
PYEOF
  # Ship the report NOW, not only from the trap: the trap is the safety net,
  # and a safety net that is also the primary path has never been tested.
  ship_report
  if [ "${GATE_FAILED}" = "1" ] || [ "${DET_RC}" != "0" ] || [ "${DIFF_RC}" != "0" ]; then
    echo "=== tpu_train_field VERIFY FAILED — see verify_report.json; the EXIT trap ships the log and deletes this node ==="
    exit 1
  fi
  echo "=== tpu_train_field VERIFY PASSED — set STEP_EST_S from the report's" \
       "s_per_step_all_in and launch MODE=train. The EXIT trap now ships the" \
       "log and deletes this node ==="
  exit 0
fi

if [ "${MODE}" != "train" ]; then
  echo "REFUSING: MODE must be 'verify' or 'train', got '${MODE}'."
  exit 1
fi

# --------------------------------------------------------------------------
# 7 · TRAIN MODE — resume from the bucket
# --------------------------------------------------------------------------
# The node is disposable; the run is not. Whatever gs://BUCKET/runs/NODE/
# holds is the newest state this run reached, because those objects are
# overwritten on every ship and nothing else writes them. A resume here is a
# TRUE continuation — the trainer restores optimiser moments and the schedule
# position from ckpt_latest.npz — which is why relaunching under the same node
# name is the documented way to continue.
echo "--- resume ---"
RESUME_ARG=""
if gcs_get "${RUNS_PREFIX}/${CKPT_NPZ_NAME}" "${CKPT_DIR}/${CKPT_NPZ_NAME}"; then
  RESUME_ARG="--resume"
  echo "measured: pulled gs://${BUCKET}/${RUNS_PREFIX}/${CKPT_NPZ_NAME}" \
       "($(du -h "${CKPT_DIR}/${CKPT_NPZ_NAME}" | cut -f1))"
  "${PY}" -c '
import sys, numpy as np
z = np.load(sys.argv[1], allow_pickle=True)
for k in ("_step", "step", "global_step"):
    if k in z.files:
        print("measured: checkpoint is at step", int(np.asarray(z[k]).reshape(-1)[0]))
        break
else:
    print("measured: checkpoint carries no step key; keys are", z.files[:12])' \
    "${CKPT_DIR}/${CKPT_NPZ_NAME}" \
    || warn "could not read a step out of the checkpoint — a log line, not a gate; the trainer's own resume is what decides"
  if gcs_get "${RUNS_PREFIX}/${CKPT_PT_NAME}" "${CKPT_DIR}/${CKPT_PT_NAME}"; then
    echo "measured: pulled the torch-loadable head beside it"
  else
    # `curl -o` leaves a zero-byte file behind on a 404, and a zero-byte
    # artefact is worse than none — it would ship over the bucket copy.
    rm -f "${CKPT_DIR}/${CKPT_PT_NAME}"
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
else
  rm -f "${CKPT_DIR}/${CKPT_NPZ_NAME}"
  echo "measured: nothing at gs://${BUCKET}/${RUNS_PREFIX}/${CKPT_NPZ_NAME} —" \
       "this is a FRESH run, not a continuation. Say so in its doc string."
fi

# --------------------------------------------------------------------------
# 8 · the shipper and the progress watchdog
# --------------------------------------------------------------------------
touch "${LAST_CKPT_MARK}"
( while true; do sleep $(( SHIP_EVERY_MIN * 60 )); ship_state; upload_log; done ) &
disown
echo "measured: shipper armed, every ${SHIP_EVERY_MIN} min to gs://${BUCKET}/${RUNS_PREFIX}/"

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
# 9 · train
# --------------------------------------------------------------------------
echo "--- train ---"
# shellcheck disable=SC2086
"${PY}" -u ml/jaxport/train_field.py \
  --z-cache "${ZF}" \
  --data "${TF}" \
  ${PIXELS_ARG} \
  --holdout-years "${HOLDOUT_YEARS}" \
  --mode "${FIELD_MODE}" \
  --K "${K}" \
  --patch "${PATCH}" \
  --d-model "${D_MODEL}" \
  --layers "${LAYERS}" \
  --heads "${HEADS}" \
  --d-cond "${D_COND}" \
  --cond-layers "${COND_LAYERS}" \
  --steps "${STEPS}" \
  --batch "${BATCH}" \
  --lr "${LR}" \
  --lr-schedule "${LR_SCHEDULE}" \
  --lr-halflife "${LR_HALFLIFE}" \
  --lr-warmup "${LR_WARMUP}" \
  --grad-clip "${GRAD_CLIP}" \
  --weight-decay "${WEIGHT_DECAY}" \
  --seed "${SEED}" \
  --input-znoise "${INPUT_ZNOISE}" \
  --eval-every "${EVAL_EVERY}" \
  --eval-windows "${EVAL_WINDOWS}" \
  --nfe "${NFE}" \
  --members "${MEMBERS}" \
  --ckpt-dir "${CKPT_DIR}" \
  --ckpt-every "${CKPT_EVERY}" \
  --out "${OUT}/results.json" \
  --metrics "${OUT}/metrics.jsonl" \
  ${COND_CHUNK:+--cond-chunk "${COND_CHUNK}"} \
  ${RESUME_ARG} \
  ${EXTRA_ARGS}

echo "=== tpu_train_field done — the EXIT trap now ships the final state and deletes this node ==="

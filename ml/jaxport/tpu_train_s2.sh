#!/bin/bash
# The startup script for a REAL stage-2 head training run on a Cloud TPU node
# (ml/plans/JAX_PORT.md tier 3b). Modelled on tpu_train.sh, same lifecycle, one
# extra staging problem: stage 2 needs the TENSOR *and* the Z.
#
#   sed -e 's|__BUCKET__|my-staging-bucket|' \
#       -e 's|__NODE__|e045-1-tpu|' \
#       -e 's|__TPUZONE__|us-central1-a|' \
#       ml/jaxport/tpu_train_s2.sh > /tmp/t.sh
#   python3 scripts/tpu_box.py create e045-1-tpu --spot --startup-file /tmp/t.sh
#
# A STARTUP SCRIPT INHERITS NO ENVIRONMENT FROM THE MACHINE THAT LAUNCHED IT.
# The `${K:-24}` forms below therefore always take their DEFAULTS on a node; to
# change a knob for a real run, sed it in the same call as __BUCKET__:
#
#       -e 's|^K=.*|K="144"|' -e 's|^STEPS=.*|STEPS="200000"|'
#
# and read the launch log back: the script prints every knob it resolved.
#
# ──────────────────────────────────────────────────────────────────────────
# THERE IS NO CHEAP STOPPED STATE. DELETE IS THE NORMAL END. Same three exits
# as tpu_train.sh, for the same reasons: (1) the run finishes → EXIT trap →
# ship → self-delete; (2) the run STALLS → progress watchdog, no NEW checkpoint
# object in the bucket for STALL_MIN minutes → reap, which is the only monitor
# that can tell a wedged trainer from a healthy one (both hold the chips at
# 100%); (3) HARD_CAP_HOURS, unconditional.
#
# To CONTINUE a run, relaunch a node with the SAME __NODE__: the node name is
# the run's identity and step 5 resumes from whatever that bucket prefix holds.
#
# ──────────────────────────────────────────────────────────────────────────
# THE Z IS THE PART THAT IS NOT LIKE STAGE 1, AND IT HAS TWO ROUTES.
#
#   Z_ASSET set   → pull the published chunks from the embed-cache-v1 release
#                   by name. This is the cheap path and the one that makes a
#                   TPU arm a TWIN of a torch arm rather than a lookalike: the
#                   embedding is a FUNCTION of the codec and the tensor, and
#                   pulling the published one removes the encoder as a variable
#                   exactly the way JAX_PORT.md §6b's G3 does.
#   Z_ASSET empty → embed on-node from the codec asset. Correct, and expensive:
#                   the pentad window is 272M encoder forwards.
#
# ASSEMBLY IS BOUNDED BY THE .npy HEADER, NOT BY THE FIRST MISSING CHUNK, and
# that is a bug fix rather than a refinement. `ml/embed_cache_sync.py:pull`
# concatenates chunks aa, ab, … until a fetch 404s. Measured against
# embed-cache-v1 on 2026-08-24: `Z_8b639abe36_37e146384b.npy` has TWELVE chunk
# assets, of which `.af` is a short 852,643,840 B in the MIDDLE of the run —
# the tail of a 6-chunk publish whose header declares (1571, 86698, 32) =
# 8,716,963,840 B = exactly chunks aa..af, sitting on top of the orphaned
# ag..al tail of the 12-chunk (3142, 86698, 32) cache that EXPERIMENTS.md
# records #427 pushing. Concatenating to the first miss therefore yields
# 16,713,707,392 B, which matches NEITHER, and `verify()` correctly discards
# it — so that Z currently costs a puller the download AND the 8.5 h rebuild.
# Reading the header first and stopping at the byte count it implies gets the
# 6-chunk file out intact, and REFUSES loudly if the bytes still do not add up.
# ──────────────────────────────────────────────────────────────────────────
set -euo pipefail

# --------------------------------------------------------------------------
# knobs
# --------------------------------------------------------------------------
BUCKET="__BUCKET__"                 # substituted at launch
NODE="__NODE__"
TPUZONE="__TPUZONE__"
REPO="blauewelt/earth"
WORK="/opt/earth-train"
OUT="${WORK}/run"
RUNS_PREFIX="runs/${NODE}"

# THE TENSOR, pinned by sha. A run on different bytes is not comparable to
# anything (the box effect, ml/CLAUDE.md §7 — a differently-built tensor moved
# the head k-fold by 0.041 at a fixed seed), and here it is sharper still: the
# embed cache is keyed by (codec weight hash, TENSOR sha256), so a box that
# builds its own tensor cannot pull the published Z at all.
TENSOR_NAME="${TENSOR_NAME:-family4_na025_pentad_r2}"
TENSOR_SHA="${TENSOR_SHA:-37e146384b6f622fefe3c7e18ad9bab0389c9538be79536899fe8729bb2d0826}"
TENSOR_PARTS="${TENSOR_PARTS:-aa ab ac}"

# THE CODEC, by release asset name on model-checkpoints-v1. Frozen: stage 2
# trains the head and never the codec.
CODEC_ASSET="${CODEC_ASSET:-run-415__pixelmae.pt}"

# THE Z, by asset name on embed-cache-v1. Empty = embed on-node.
Z_ASSET="${Z_ASSET:-}"

# The run. Defaults are ml/temporal.py's own, so an unsubstituted launch is a
# small honest run rather than a wrong big one.
K="${K:-24}"
STEPS="${STEPS:-200000}"
BATCH="${BATCH:-256}"
LR="${LR:-1e-3}"
D_MODEL="${D_MODEL:-1024}"
LAYERS="${LAYERS:-16}"
STENCIL="${STENCIL:-1}"
RING_KM="${RING_KM:-0}"
LR_SCHEDULE="${LR_SCHEDULE:-expdecay}"
LR_HALFLIFE="${LR_HALFLIFE:-40000}"
LR_COOLDOWN_FRAC="${LR_COOLDOWN_FRAC:-0}"
LR_WARMUP="${LR_WARMUP:-2000}"
INPUT_ZNOISE="${INPUT_ZNOISE:-0}"
GRAD_CLIP="${GRAD_CLIP:-0}"
# E-054b · micro-batching for a head whose ACTIVATIONS do not fit the chip.
# N > 1 splits each step's BATCH into N micro-batches of BATCH/N and takes ONE
# AdamW update on their averaged gradient — the same optimisation as a single
# BATCH step, so BATCH stays the number every record and every comparison
# names. 1 (the default) builds no accumulation graph at all. Must divide
# BATCH. The 400M rung (1280x20, K 144) needs this: it asked for 5.09 G with
# 4.03 G free on a v5e-4 chip at BATCH 256 and died at step 1.
GRAD_ACCUM="${GRAD_ACCUM:-1}"
MILESTONE_STEPS="${MILESTONE_STEPS:-}"
TRAIN_LON_HOLD="${TRAIN_LON_HOLD:-inherit}"
# What the YEAR holdout excludes from the window pool. `window` (default)
# drops any window that TOUCHES a held-out bin. `target` keeps the legacy
# pool and masks every per-frame loss term whose TARGET bin is held out.
# `endpoint_contaminated` is the legacy pool and it LEAKS: only a window
# whose SCORED bin t+1 is held out is dropped, so a window ending just after
# a holdout year still teacher-forces that year's transitions. Pass it
# explicitly, and only to reproduce a run archived before c25f6ff.
HOLDOUT_SCOPE="${HOLDOUT_SCOPE:-window}"
SEED="${SEED:-0}"
TAG="${TAG:-}"
CKPT_EVERY="${CKPT_EVERY:-2000}"
EXTRA_ARGS="${EXTRA_ARGS:-}"

# The input pipeline (2026-08-25, E-051's third node): at K=144 the DEFAULT
# host pipeline is ~15 s/step of numpy RNG plus ~2 s of gather+cast against
# ~0.3 s of TPU step — the chips idle 98% of the wall clock. These four move
# the noise on-device, ship fp16, thread the gather and overlap it with the
# device step. Values are unchanged (fp16→fp32 is exact; the noise stream is
# jax.random instead of numpy — a fresh-run fact the config line records).
NOISE_BACKEND="${NOISE_BACKEND:-device}"
GATHER_FP16="${GATHER_FP16:-1}"
GATHER_WORKERS="${GATHER_WORKERS:-8}"
PREFETCH="${PREFETCH:-2}"

# Lifecycle.
SHIP_EVERY_MIN="${SHIP_EVERY_MIN:-10}"   # upload cadence
STALL_MIN="${STALL_MIN:-90}"             # progress watchdog
HARD_CAP_HOURS="${HARD_CAP_HOURS:-30}"   # unconditional cap

STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
LOG=/tmp/train_s2.log
LAST_CKPT_MARK=/tmp/last_ckpt_upload     # touched when a NEW ckpt lands
LAST_CKPT_SIG=/tmp/last_ckpt_sig         # mtime+size of what was last shipped
CKPT_NPZ_NAME="temporal${TAG:+_${TAG}}_jax.npz"
CKPT_PT_NAME="temporal${TAG:+_${TAG}}.pt"

# --------------------------------------------------------------------------
# 0 · self-reap, log shipping, and the two watchdogs
# --------------------------------------------------------------------------
exec >>"${LOG}" 2>&1

token() {
  curl -sf -H 'Metadata-Flavor: Google' \
    'http://metadata.google.internal/computeMetadata/v1/instance/service-accounts/default/token' \
    | python3 -c 'import json,sys; print(json.load(sys.stdin)["access_token"])'
}

# `-T` STREAMS the file; `--data-binary @file` buffers the whole thing in
# memory, which is fine for a few-KB JSON and is not fine for a multi-GB
# optimiser-state checkpoint (ml/CLAUDE.md §7).
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

upload_log() { gcs_put "${LOG}" "${RUNS_PREFIX}/logs/${STAMP}.txt" "text/plain" || true; }

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
      "https://tpu.googleapis.com/v2/projects/${P}/locations/${TPUZONE}/nodes/${NODE}" || true
    sleep 30
    CODE="$(curl -s -o /dev/null -w '%{http_code}' -H "Authorization: Bearer ${T}" \
      "https://tpu.googleapis.com/v2/projects/${P}/locations/${TPUZONE}/nodes/${NODE}")"
    if [ "${CODE}" = "404" ]; then echo "self-delete confirmed (404)"; return 0; fi
    echo "node still answers ${CODE} after delete attempt ${_i} — retrying"
  done
}

ship_state() {
  # metrics.jsonl EVERY cycle — it is small, it is the black box, and a run
  # whose curve is unreadable is a run nobody can act on.
  if [ -f "${OUT}/metrics.jsonl" ]; then
    gcs_put "${OUT}/metrics.jsonl" "${RUNS_PREFIX}/metrics.jsonl" \
      "application/x-ndjson" \
      && echo "shipped metrics.jsonl ($(wc -l < "${OUT}/metrics.jsonl") lines)" \
      || echo "metrics upload FAILED this cycle (will retry next cycle)"
  fi
  # The CHECKPOINT only when it has actually changed. This is what makes the
  # progress watchdog measure PROGRESS rather than measure the uploader: a
  # wedged trainer keeps the same checkpoint on disk, nothing new is shipped,
  # the marker stops moving, and the node is reaped.
  local C="${OUT}/${CKPT_NPZ_NAME}" SIG=""
  [ -f "${C}" ] || return 0
  SIG="$(stat -c '%Y:%s' "${C}")"
  if [ -f "${LAST_CKPT_SIG}" ] && [ "${SIG}" = "$(cat "${LAST_CKPT_SIG}")" ]; then
    echo "checkpoint unchanged since the last ship (${SIG}) — not re-uploading"
    return 0
  fi
  if gcs_put "${C}" "${RUNS_PREFIX}/${CKPT_NPZ_NAME}" "application/octet-stream"; then
    # The torch-loadable head travels WITH it, so the eval ladder never has to
    # wait for the run to end (JAX_PORT.md §1b).
    [ -f "${OUT}/${CKPT_PT_NAME}" ] && gcs_put "${OUT}/${CKPT_PT_NAME}" \
      "${RUNS_PREFIX}/${CKPT_PT_NAME}" "application/octet-stream" || true
    # MILESTONE HEADS TOO. They are the rungs a 200k run must retain
    # (E-031/E-032) and they exist only on a node that is going to be deleted.
    for MS in "${OUT}"/temporal_ms*.pt; do
      [ -f "${MS}" ] || continue
      gcs_put "${MS}" "${RUNS_PREFIX}/$(basename "${MS}")" \
        "application/octet-stream" || true
    done
    echo "${SIG}" > "${LAST_CKPT_SIG}"
    touch "${LAST_CKPT_MARK}"
    echo "shipped NEW checkpoint ${SIG} ($(du -h "${C}" | cut -f1)) to gs://${BUCKET}/${RUNS_PREFIX}/"
  else
    echo "checkpoint upload FAILED this cycle (will retry next cycle)"
  fi
}

trap 'code=$?; echo "exit ${code} — shipping final state, then the node self-deletes"; ship_state; upload_log; self_delete' EXIT

# Watchdog A: the PROGRESS watchdog.
touch "${LAST_CKPT_MARK}"
(
  while true; do
    sleep 300
    AGE=$(( $(date +%s) - $(stat -c %Y "${LAST_CKPT_MARK}") ))
    if [ "${AGE}" -gt $(( STALL_MIN * 60 )) ]; then
      echo "PROGRESS WATCHDOG: no new checkpoint object has landed in" \
           "gs://${BUCKET}/${RUNS_PREFIX}/ for $(( AGE / 60 )) min" \
           "(limit ${STALL_MIN}) — the trainer is stalled and the chips are" \
           "billing. Reaping."
      upload_log
      self_delete
      exit 0
    fi
  done
) &
disown

# Watchdog B: the unconditional cap. It is the one that fires when the failure
# is in the watchdog above.
( sleep $(( HARD_CAP_HOURS * 3600 ))
  echo "HARD CAP: ${HARD_CAP_HOURS} h reached — force self-delete"
  upload_log; self_delete ) &
disown

echo "=== tpu_train_s2 ${STAMP} · node ${NODE} · bucket ${BUCKET} ==="

# BOOT BEACON — the first bucket object must land within ~3 minutes of the
# script starting, unconditionally. Before this existed, the first object on
# the healthy path was whatever section 6's shipper uploaded, and that
# shipper SLEEPS FIRST for SHIP_EVERY_MIN and is armed only behind the
# clone, the 4.3 G tensor pull and the 17 G Z pull: measured on E-054b's
# node, READY 08:10:33Z, first object 08:27:34Z — seventeen minutes of
# silence from an entirely healthy node, which from the outside is the SAME
# OBSERVATION as a node whose startup script never ran (the us-west1-c
# maintenance-event zombies, 2026-08-26). It fires HERE, at the banner,
# because the trap and both watchdogs are already armed above it: a log in
# the bucket therefore proves this script is genuinely executing and that
# the machinery which reaps the node exists, not merely that the node
# booted. One early log upload plus a 3-minute shipper turns "no object
# under runs/<exp>/ in ~6 min" into a certain zombie verdict
# (ml/CLAUDE.md §7). upload_log swallows its own failures, so a beacon
# whose PUT 500s costs a cycle and never the run.
upload_log
( while true; do sleep 180; upload_log; done ) &
BEACON_PID=$!
disown
echo "measured: boot beacon shipped and 3-min early log shipper armed"
echo "lifecycle: ship every ${SHIP_EVERY_MIN} min · stall watchdog ${STALL_MIN} min · hard cap ${HARD_CAP_HOURS} h"
# EVERY KNOB, RESOLVED, in one place. A startup script inherits no environment,
# so "I set K=144 when I launched it" is a claim about the launching shell and
# not about this node; this is the line that settles it.
echo "resolved knobs: K ${K} · steps ${STEPS} · batch ${BATCH} · lr ${LR} ·" \
     "${D_MODEL}x${LAYERS} · stencil ${STENCIL} ring '${RING_KM}' ·" \
     "sched ${LR_SCHEDULE} halflife ${LR_HALFLIFE} cooldown ${LR_COOLDOWN_FRAC}" \
     "warmup ${LR_WARMUP} · znoise ${INPUT_ZNOISE} · grad_clip ${GRAD_CLIP} ·" \
     "grad_accum ${GRAD_ACCUM} (micro $(( BATCH / (GRAD_ACCUM > 0 ? GRAD_ACCUM : 1) ))) ·" \
     "milestones '${MILESTONE_STEPS}' · train_lon_hold ${TRAIN_LON_HOLD} ·" \
     "holdout_scope ${HOLDOUT_SCOPE} ·" \
     "seed ${SEED} · tag '${TAG}' · tensor ${TENSOR_NAME} (${TENSOR_SHA:0:10})" \
     "· codec ${CODEC_ASSET} · Z '${Z_ASSET:-<embed on node>}' ·" \
     "ckpt_every ${CKPT_EVERY} · extra '${EXTRA_ARGS}' ·" \
     "pipeline: noise ${NOISE_BACKEND} · fp16 ${GATHER_FP16} ·" \
     "gather_workers ${GATHER_WORKERS} · prefetch ${PREFETCH}"

# --------------------------------------------------------------------------
# 1 · what this host actually is
# --------------------------------------------------------------------------
echo "--- host ---"
mkdir -p "${WORK}" "${OUT}" "${WORK}/cache"
echo "measured: $(nproc) CPUs, $(awk '/MemTotal/{printf "%.0f GB RAM", $2/1048576}' /proc/meminfo)"
AVAIL_GB="$(df -BG --output=avail "${WORK}" | tail -1 | tr -dc '0-9')"
echo "measured: ${AVAIL_GB} GB free on ${WORK}"
# SIZE THE GUARD FROM THE ALLOCATION IT GUARDS (ml/CLAUDE.md §5.18). Stage 2 is
# heavier than stage 1 because the Z is a first-class artefact: the pentad
# tensor archive is ~11 GB, decompressed ~34 GB, the anomaly scratch copy the
# same again, and the published pentad Z is 16.24 GiB on top.
NEED_GB=120
if [ "${AVAIL_GB}" -lt "${NEED_GB}" ]; then
  # The v5e boot disk serves ~90 GB free — short of the pentad allocation —
  # but the host carries 189 GB of RAM (measured, E-051 2026-08-25), so
  # before refusing, try tmpfs: remount /dev/shm large enough and move WORK
  # there. RAM-backed staging is also what removed the memmap-read cost from
  # the gather. Only then refuse.
  RAM_GB="$(awk '/MemTotal/{printf "%.0f", $2/1048576}' /proc/meminfo)"
  if [ "${RAM_GB}" -ge 160 ]; then
    echo "boot disk short (${AVAIL_GB} GB < ${NEED_GB}) but host has" \
         "${RAM_GB} GB RAM — falling back to tmpfs"
    mount -o remount,size=170G /dev/shm || true
    WORK=/dev/shm/earth-train
    OUT="${WORK}/run"
    mkdir -p "${WORK}" "${OUT}" "${WORK}/cache"
    AVAIL_GB="$(df -BG --output=avail "${WORK}" | tail -1 | tr -dc '0-9')"
    echo "measured: ${AVAIL_GB} GB free on ${WORK} (tmpfs)"
  fi
fi
if [ "${AVAIL_GB}" -lt "${NEED_GB}" ]; then
  echo "REFUSING: ${AVAIL_GB} GB free on ${WORK}, need ~${NEED_GB}. The pentad" \
       "tensor archive is ~11 GB, decompressed ~34 GB, the anomaly scratch copy" \
       "the same again, and the Z cache 16.24 GiB. Create the node with a" \
       "larger boot disk."
  exit 1
fi

# --------------------------------------------------------------------------
# 2 · dependencies
# --------------------------------------------------------------------------
echo "--- deps ---"
export DEBIAN_FRONTEND=noninteractive
# unattended-upgrades holds the dpkg lock for MINUTES on first boot — measured
# 2026-08-24 on the first 60k launch: it outlived all five 20 s retries, the
# script exited 1 two minutes in, and the self-reap correctly threw away a
# healthy node. Stop the service, then let apt WAIT on the lock instead of
# racing it; the retry loop stays for everything else apt can throw.
systemctl stop unattended-upgrades 2>&1 || true
systemctl kill --kill-who=all unattended-upgrades 2>&1 || true
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
# CPU torch. train_stage2.py needs it for THREE things, not one: reading the
# codec .pt, the shared numpy plumbing it imports rather than copies
# (anomaly_transform, ridge_r, build_stencil, season_ctx, lon_holdout_mask),
# and — since the G5c finding — constructing the torch head under
# torch.manual_seed so a same-seed torch run starts from identical weights.
"${PY}" -m pip install --index-url https://download.pytorch.org/whl/cpu torch
echo "measured: jax $("${PY}" -c 'import jax; print(jax.__version__)')" \
     "· flax $("${PY}" -c 'import flax; print(flax.__version__)')" \
     "· optax $("${PY}" -c 'import optax; print(optax.__version__)')" \
     "· torch $("${PY}" -c 'import torch; print(torch.__version__)')"
echo "measured: jax.devices() -> $("${PY}" -c 'import jax; print(jax.devices())')"

# --------------------------------------------------------------------------
# 3 · the code
# --------------------------------------------------------------------------
echo "--- clone ---"
git clone --depth 1 "https://github.com/${REPO}.git" "${WORK}/earth"
echo "measured: cloned $(git -C "${WORK}/earth" rev-parse --short HEAD)" \
     "($(git -C "${WORK}/earth" log -1 --format=%cI))"

# --------------------------------------------------------------------------
# 4 · the data — the tensor, the codec, and (maybe) the Z
# --------------------------------------------------------------------------
echo "--- tensor ---"
TF="${WORK}/cache/${TENSOR_NAME}.npz"
if [ -f "${TF}" ] && [ "$(sha256sum "${TF}" | cut -d' ' -f1)" = "${TENSOR_SHA}" ]; then
  echo "measured: tensor already present, sha matches ${TENSOR_SHA:0:10}"
else
  for sfx in ${TENSOR_PARTS}; do
    echo "fetching ${TENSOR_NAME}_${TENSOR_SHA:0:10}.npz.${sfx} …"
    curl -fsSL --max-time 1800 --retry 3 --retry-delay 10 \
      -o "${TF}.${sfx}.part" \
      "https://github.com/${REPO}/releases/download/data-cache-v1/${TENSOR_NAME}_${TENSOR_SHA:0:10}.npz.${sfx}"
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
  echo "measured: tensor $(du -h "${TF}" | cut -f1), sha verified ${TENSOR_SHA:0:10}"
fi

echo "--- codec ---"
CK="${WORK}/cache/${CODEC_ASSET}"
if [ ! -f "${CK}" ]; then
  curl -fsSL --max-time 1800 --retry 3 --retry-delay 10 -o "${CK}.part" \
    "https://github.com/${REPO}/releases/download/model-checkpoints-v1/${CODEC_ASSET}"
  mv "${CK}.part" "${CK}"
fi
echo "measured: codec $(du -h "${CK}" | cut -f1) · weight hash" \
     "$(cd "${WORK}/earth" && "${PY}" -c '
import sys, torch
sys.path.insert(0, "ml")
from temporal import codec_weight_hash
print(codec_weight_hash(torch.load(sys.argv[1], map_location="cpu", weights_only=False)))' "${CK}")"

Z_ARG=""
if [ -n "${Z_ASSET}" ]; then
  echo "--- Z (published cache) ---"
  ZF="${WORK}/cache/${Z_ASSET}"
  if [ -f "${ZF}" ]; then
    echo "measured: Z already present ($(du -h "${ZF}" | cut -f1))"
  else
    # Pull chunk aa first, read its .npy header, and let the header decide how
    # many bytes the file is. See the note at the top of this script: the
    # published pentad Z has orphaned tail chunks under the same asset name,
    # and assembling to the first missing chunk produces a file that matches
    # neither publish and is correctly discarded by every verifier.
    echo "fetching ${Z_ASSET}.aa …"
    curl -fsSL --max-time 1800 --retry 3 --retry-delay 10 \
      -o "${ZF}.aa.part" \
      "https://github.com/${REPO}/releases/download/embed-cache-v1/${Z_ASSET}.aa"
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
             "published cache is incomplete; embed on-node instead (unset Z_ASSET)."
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
    # TRUNCATE TO THE HEADER, and say so. The last chunk of a superseded
    # larger publish can overshoot; the header is the authority on where the
    # array ends, and a file longer than its own header is exactly the
    # "reassembled out of order" case embed_cache_sync.verify() refuses.
    if [ "${GOTB}" -gt "${WANT_BYTES}" ]; then
      echo "measured: assembled ${GOTB} B against the header's ${WANT_BYTES} B" \
           "— truncating the overshoot from the final chunk"
      truncate -s "${WANT_BYTES}" "${ZF}.new"
    fi
    mv "${ZF}.new" "${ZF}"
    echo "measured: Z assembled to $(du -h "${ZF}" | cut -f1) in $(( IDX ))" \
         "chunk(s), header-bounded"
  fi
  # VERIFY BEFORE TRUSTING, with embed_cache_sync's own function rather than a
  # second copy of the rule.
  # …and against the TENSOR'S OWN AXIS, not only against its own header. The
  # truncate above bounds an assembled Z by what its header claims, which is
  # the right answer to a stale chunk tail and no answer at all to a Z that
  # was strided before it was published (#462): that file is internally
  # consistent and simply covers one bin in two. `tensor_t` reads ~128 bytes
  # off ${TF} — the same tensor this run pinned by sha above.
  (cd "${WORK}/earth" && "${PY}" -c '
import sys
sys.path.insert(0, "ml")
from embed_cache_sync import tensor_t, verify
ok, why = verify(sys.argv[1], tensor_t(sys.argv[2]))
print(("Z VERIFIED: " if ok else "Z REJECTED: ") + why)
sys.exit(0 if ok else 1)' "${ZF}" "${TF}") || {
    echo "REFUSING: the assembled Z does not verify. Unset Z_ASSET to embed" \
         "on-node instead of training on an embedding of unknown provenance."
    exit 1; }
  Z_ARG="--z ${ZF}"
else
  echo "--- Z: none named, the trainer will EMBED ON-NODE from ${CODEC_ASSET} ---"
  echo "note: the pentad window is ~272M encoder forwards. Budget hours, and" \
       "check STALL_MIN (${STALL_MIN} min) is longer than the embed pass, or" \
       "the progress watchdog reaps a node that is working correctly."
fi

# --------------------------------------------------------------------------
# 5 · RESUME from the bucket
# --------------------------------------------------------------------------
# The node is disposable; the run is not. Whatever gs://BUCKET/runs/NODE/ holds
# is the newest state this run reached, because that object is overwritten on
# every ship and nothing else writes it.
echo "--- resume ---"
RESUME_ARG=""
if gcs_get "${RUNS_PREFIX}/${CKPT_NPZ_NAME}" "${OUT}/${CKPT_NPZ_NAME}"; then
  RESUME_ARG="--resume ${OUT}/${CKPT_NPZ_NAME}"
  echo "measured: pulled a checkpoint from gs://${BUCKET}/${RUNS_PREFIX}/${CKPT_NPZ_NAME}" \
       "($(du -h "${OUT}/${CKPT_NPZ_NAME}" | cut -f1)), step" \
       "$("${PY}" -c 'import numpy,sys; print(int(numpy.load(sys.argv[1])["_step"]))' "${OUT}/${CKPT_NPZ_NAME}")"
  gcs_get "${RUNS_PREFIX}/metrics.jsonl" "${OUT}/metrics.jsonl" \
    && echo "measured: pulled metrics.jsonl ($(wc -l < "${OUT}/metrics.jsonl") lines)" \
    || echo "no metrics.jsonl in the bucket yet — this run starts the curve"
else
  rm -f "${OUT}/${CKPT_NPZ_NAME}"
  echo "measured: nothing at gs://${BUCKET}/${RUNS_PREFIX}/${CKPT_NPZ_NAME} —" \
       "this is a FRESH run, not a continuation. Say so in its doc string."
fi

# --------------------------------------------------------------------------
# 6 · the shipper
# --------------------------------------------------------------------------
( while true; do sleep $(( SHIP_EVERY_MIN * 60 )); ship_state; upload_log; done ) &
disown
echo "measured: shipper armed, every ${SHIP_EVERY_MIN} min to gs://${BUCKET}/${RUNS_PREFIX}/"
# RETIRE THE BOOT BEACON: the shipper above writes the SAME log object, and
# two uploaders racing on one object name make the newest log the loser of a
# coin toss. Killing the subshell is the whole retirement — it was disowned,
# not detached from this process tree. `|| true` because a kill that finds
# nothing (the beacon already gone) must not take a healthy run down under
# `set -e`.
kill "${BEACON_PID}" 2>/dev/null || true
echo "measured: boot beacon retired — the shipper owns the log object now"

# --------------------------------------------------------------------------
# 7 · train
# --------------------------------------------------------------------------
echo "--- train ---"
cd "${WORK}/earth"
export CKPT_TAG="${NODE}"
# shellcheck disable=SC2086
"${PY}" -u ml/jaxport/train_stage2.py \
  --data "${TF}" \
  --ckpt "${CK}" \
  ${Z_ARG} \
  --out "${OUT}" \
  --K "${K}" \
  --steps "${STEPS}" \
  --batch "${BATCH}" \
  --lr "${LR}" \
  --d-model "${D_MODEL}" \
  --layers "${LAYERS}" \
  --stencil "${STENCIL}" \
  --ring-km="${RING_KM}" \
  --lr-schedule "${LR_SCHEDULE}" \
  --lr-halflife "${LR_HALFLIFE}" \
  --lr-cooldown-frac "${LR_COOLDOWN_FRAC}" \
  --lr-warmup "${LR_WARMUP}" \
  --input-znoise "${INPUT_ZNOISE}" \
  --noise-backend "${NOISE_BACKEND}" \
  $( [ "${GATHER_FP16:-0}" != "0" ] && echo "--gather-fp16" ) \
  --gather-workers "${GATHER_WORKERS}" \
  --prefetch "${PREFETCH}" \
  --grad-clip "${GRAD_CLIP}" \
  --grad-accum "${GRAD_ACCUM}" \
  ${MILESTONE_STEPS:+--milestone-steps "${MILESTONE_STEPS}"} \
  --train-lon-hold="${TRAIN_LON_HOLD}" \
  --holdout-scope "${HOLDOUT_SCOPE}" \
  --seed "${SEED}" \
  ${TAG:+--tag "${TAG}"} \
  --ckpt-every "${CKPT_EVERY}" \
  ${RESUME_ARG} \
  ${EXTRA_ARGS}

echo "=== tpu_train_s2 done — the EXIT trap now ships the final state and deletes this node ==="

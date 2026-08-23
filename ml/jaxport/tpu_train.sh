#!/bin/bash
# The startup script for a REAL stage-1 codec training run on a Cloud TPU
# node (ml/plans/JAX_PORT.md §1b, tier 3). Launched as a TPU node's metadata
# `startup-script`:
#
#   sed -e 's|__BUCKET__|my-staging-bucket|' \
#       -e 's|__NODE__|codec-a|' \
#       -e 's|__TPUZONE__|us-central1-a|' \
#       ml/jaxport/tpu_train.sh > /tmp/t.sh
#   python3 scripts/tpu_box.py create codec-a --spot --startup-file /tmp/t.sh
#
# A STARTUP SCRIPT INHERITS NO ENVIRONMENT FROM THE MACHINE THAT LAUNCHED IT.
# The `${STEPS:-60000}` forms in the knobs block below therefore always take
# their DEFAULTS on a node; the `${VAR:-…}` spelling is there so the file can
# also be sourced or run locally with overrides while it is being tested. To
# change a knob for a real run, sed it in the same call as __BUCKET__:
#
#       -e 's|^STEPS=.*|STEPS="200000"|' -e 's|^PATCH=.*|PATCH="1"|'
#
# and read the launch log back: the script prints every knob it resolved.
#
# ──────────────────────────────────────────────────────────────────────────
# THERE IS NO CHEAP STOPPED STATE. DELETE IS THE NORMAL END.
#
# A Vast box can be stopped: the disk survives, the meter mostly stops, and a
# later job resumes from /opt/earth-cache. A Cloud TPU node has no equivalent.
# A node that exists bills for its chips whether or not anything is running on
# it, and there is no "stopped, keep the disk" state to park it in. So the
# lifecycle here is deliberately one-way: the node comes up, trains, ships its
# state to the bucket, and DELETES ITSELF. Every checkpoint lives in
# gs://__BUCKET__/runs/__NODE__/, never on the node, because the node is the
# thing that goes away.
#
# That is why this script has THREE independent ways to end, and why none of
# them depends on a human or on a session staying alive:
#
#   1. the run finishes                → EXIT trap → ship → self-delete;
#   2. the run STALLS                  → progress watchdog: if no NEW
#      checkpoint object has landed in the bucket for STALL_MIN minutes, the
#      node is reaped. A trainer that has wedged (a starved host gather, a
#      hung upload, an XLA deadlock) holds the chips at 100% and looks exactly
#      like a healthy one to every other monitor — this is the one that can
#      tell them apart, and it is the reason a fixed timer is not enough;
#   3. anything else                   → HARD_CAP_HOURS, unconditional.
#
# To CONTINUE a run, relaunch a node with the SAME __NODE__ substitution: the
# node name is the run's identity, the bucket prefix is keyed on it, and step 3
# below resumes from whatever checkpoint that prefix holds. A different name
# starts a different run.
#
# Every step echoes WHAT IT MEASURED, not what it attempted (ml/CLAUDE.md
# §0.2 / §4.7). `set -euo pipefail` throughout and no `2>/dev/null` anywhere.
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

# The tensor, pinned by sha exactly as tpu_smoke.sh pins it and for the same
# reason: a run on different bytes is not comparable to anything (the box
# effect, ml/CLAUDE.md §7 — a differently-built tensor moved the head k-fold
# by 0.041 at a fixed seed).
TENSOR_SHA="adcbe700fb6e160b1c84d7acd8dc0333b34d07829db9abd62c721aab278b4848"
TENSOR_PARTS="aa ab"
TENSOR_ASSET_PREFIX="family3_na025_${TENSOR_SHA:0:10}.npz"

# The run. Substitutable at launch the same way the bucket is; the defaults
# are the f3 anchor geometry (ml/recipes/f3-anchor-41M.json).
STEPS="${STEPS:-60000}"
BATCH="${BATCH:-512}"
LR="${LR:-3e-4}"
D_Z="${D_Z:-64}"
PATCH="${PATCH:-3}"
D_MODEL="${D_MODEL:-576}"
N_LAYERS="${N_LAYERS:-10}"
N_HEADS="${N_HEADS:-8}"
D_DEC="${D_DEC:-768}"
HOLDOUT_LON="${HOLDOUT_LON:-0,0}"
TIME_BLOCK="${TIME_BLOCK:-}"
LIGHT_PROBE_EVERY="${LIGHT_PROBE_EVERY:-2000}"
CKPT_EVERY="${CKPT_EVERY:-2000}"
EXTRA_ARGS="${EXTRA_ARGS:-}"

# Lifecycle.
SHIP_EVERY_MIN="${SHIP_EVERY_MIN:-10}"   # upload cadence
STALL_MIN="${STALL_MIN:-90}"             # progress watchdog
HARD_CAP_HOURS="${HARD_CAP_HOURS:-30}"   # unconditional cap

STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
LOG=/tmp/train.log
LAST_CKPT_MARK=/tmp/last_ckpt_upload     # touched when a NEW ckpt lands
LAST_CKPT_SIG=/tmp/last_ckpt_sig         # mtime+size of what was last shipped

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
# memory, which is fine for a few-KB JSON and is not fine for a 2.5 GB
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
  local C="${OUT}/pixelmae_jax.npz" SIG=""
  [ -f "${C}" ] || return 0
  SIG="$(stat -c '%Y:%s' "${C}")"
  if [ -f "${LAST_CKPT_SIG}" ] && [ "${SIG}" = "$(cat "${LAST_CKPT_SIG}")" ]; then
    echo "checkpoint unchanged since the last ship (${SIG}) — not re-uploading"
    return 0
  fi
  if gcs_put "${C}" "${RUNS_PREFIX}/pixelmae_jax.npz" "application/octet-stream"; then
    # The torch-loadable artefact travels WITH it, so the eval ladder never
    # has to wait for the run to end (JAX_PORT.md §1b).
    [ -f "${OUT}/pixelmae.pt" ] && gcs_put "${OUT}/pixelmae.pt" \
      "${RUNS_PREFIX}/pixelmae.pt" "application/octet-stream" || true
    echo "${SIG}" > "${LAST_CKPT_SIG}"
    touch "${LAST_CKPT_MARK}"
    echo "shipped NEW checkpoint ${SIG} ($(du -h "${C}" | cut -f1)) to gs://${BUCKET}/${RUNS_PREFIX}/"
  else
    echo "checkpoint upload FAILED this cycle (will retry next cycle)"
  fi
}

trap 'code=$?; echo "exit ${code} — shipping final state, then the node self-deletes"; ship_state; upload_log; self_delete' EXIT

# Watchdog A: the PROGRESS watchdog. It replaces the smoke script's fixed
# 55-minute timer, which was right for a bounded measurement and is wrong for
# a run whose whole point is to take many hours.
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

# Watchdog B: the unconditional cap. Sized for training, not for a smoke run,
# and substitutable — but never absent: it is the one that fires when the
# failure is in the watchdog above.
( sleep $(( HARD_CAP_HOURS * 3600 ))
  echo "HARD CAP: ${HARD_CAP_HOURS} h reached — force self-delete"
  upload_log; self_delete ) &
disown

echo "=== tpu_train ${STAMP} · node ${NODE} · bucket ${BUCKET} ==="
echo "lifecycle: ship every ${SHIP_EVERY_MIN} min · stall watchdog ${STALL_MIN} min · hard cap ${HARD_CAP_HOURS} h"
# EVERY KNOB, RESOLVED, in one line. A startup script inherits no environment,
# so "I set STEPS=200000 when I launched it" is a claim about the launching
# shell and not about this node; this is the line that settles it.
echo "resolved knobs: steps ${STEPS} · batch ${BATCH} · lr ${LR} · d_z ${D_Z}" \
     "· patch ${PATCH} · ${D_MODEL}x${N_LAYERS} heads ${N_HEADS} · d_dec" \
     "${D_DEC} · holdout_lon ${HOLDOUT_LON} · time_block '${TIME_BLOCK}' ·" \
     "light_probe_every ${LIGHT_PROBE_EVERY} · ckpt_every ${CKPT_EVERY} ·" \
     "extra '${EXTRA_ARGS}'"

# --------------------------------------------------------------------------
# 1 · what this host actually is
# --------------------------------------------------------------------------
echo "--- host ---"
mkdir -p "${WORK}" "${OUT}"
echo "measured: $(nproc) CPUs, $(awk '/MemTotal/{printf "%.0f GB RAM", $2/1048576}' /proc/meminfo)"
AVAIL_GB="$(df -BG --output=avail "${WORK}" | tail -1 | tr -dc '0-9')"
echo "measured: ${AVAIL_GB} GB free on ${WORK}"
if [ "${AVAIL_GB}" -lt 40 ]; then
  echo "REFUSING: ${AVAIL_GB} GB free on ${WORK}. The tensor archive is 3.0 GB," \
       "the decompressed tensor ~10.9 GB, the anomaly scratch copy the same" \
       "again, and the checkpoints on top. Create the node with a larger boot disk."
  exit 1
fi

# --------------------------------------------------------------------------
# 2 · dependencies
# --------------------------------------------------------------------------
echo "--- deps ---"
# The TPU VM image (Ubuntu 22.04, v2-alpha-tpuv5-lite) ships WITHOUT
# python3-venv — measured 2026-08-23, `python3 -m venv` dies in seconds with
# "ensurepip is not available". Retried because first-boot apt can hold locks
# under cloud-init.
export DEBIAN_FRONTEND=noninteractive
for i in 1 2 3 4 5; do
  if apt-get update -qq && apt-get install -y -qq python3-venv; then
    echo "measured: python3-venv installed (attempt ${i})"; break
  fi
  if [ "${i}" = 5 ]; then echo "apt failed 5x — giving up"; exit 1; fi
  echo "apt attempt ${i} failed (boot lock?) — retrying in 20 s"; sleep 20
done
python3 -m venv "${WORK}/venv"
PY="${WORK}/venv/bin/python"
"${PY}" -m pip install --upgrade pip
"${PY}" -m pip install 'jax[tpu]' \
    -f https://storage.googleapis.com/jax-releases/libtpu_releases.html
"${PY}" -m pip install flax numpy optax
# CPU torch. On the smoke run this was only for the checkpoint converter; the
# trainer needs it for a second reason as well — ml/jaxport/train_stage1.py
# IMPORTS the shared numpy plumbing (anomaly_transform, ridge_r,
# rapid_section, lon_holdout_mask, fit_schedule) rather than copying it, and
# those helpers live in modules that import torch at their top.
"${PY}" -m pip install --index-url https://download.pytorch.org/whl/cpu torch
echo "measured: jax $("${PY}" -c 'import jax; print(jax.__version__)')" \
     "· flax $("${PY}" -c 'import flax; print(flax.__version__)')" \
     "· optax $("${PY}" -c 'import optax; print(optax.__version__)')" \
     "· torch $("${PY}" -c 'import torch; print(torch.__version__)')"
echo "measured: jax.devices() -> $("${PY}" -c 'import jax; print(jax.devices())')"
# flax >= 0.11 needs Python >= 3.11, which this image (3.10) cannot install,
# and `nnx.data` exists only there. ml/jaxport/models.py carries a getattr
# shim for exactly this, so a current clone needs nothing here; the line below
# only REPORTS which regime the node is in, so a parity question later has an
# answer instead of a guess.
echo "measured: nnx.data present -> $("${PY}" -c 'from flax import nnx; print(hasattr(nnx, "data"))')"

# --------------------------------------------------------------------------
# 3 · the code
# --------------------------------------------------------------------------
echo "--- clone ---"
git clone --depth 1 "https://github.com/${REPO}.git" "${WORK}/earth"
echo "measured: cloned $(git -C "${WORK}/earth" rev-parse --short HEAD)" \
     "($(git -C "${WORK}/earth" log -1 --format=%cI))"

# --------------------------------------------------------------------------
# 4 · the data
# --------------------------------------------------------------------------
echo "--- data ---"
mkdir -p "${WORK}/cache"
TF="${WORK}/cache/family3_na025.npz"
if [ -f "${TF}" ] && [ "$(sha256sum "${TF}" | cut -d' ' -f1)" = "${TENSOR_SHA}" ]; then
  echo "measured: tensor already present, sha matches ${TENSOR_SHA:0:10}"
else
  for sfx in ${TENSOR_PARTS}; do
    echo "fetching ${TENSOR_ASSET_PREFIX}.${sfx} …"
    curl -fsSL --max-time 1800 --retry 3 --retry-delay 10 \
      -o "${TF}.${sfx}.part" \
      "https://github.com/${REPO}/releases/download/data-cache-v1/${TENSOR_ASSET_PREFIX}.${sfx}"
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

# --------------------------------------------------------------------------
# 5 · RESUME from the bucket
# --------------------------------------------------------------------------
# The node is disposable; the run is not. Whatever gs://BUCKET/runs/NODE/
# holds is the newest state this run reached, because that object is
# overwritten on every ship and nothing else writes it.
echo "--- resume ---"
RESUME_ARG=""
if gcs_get "${RUNS_PREFIX}/pixelmae_jax.npz" "${OUT}/pixelmae_jax.npz"; then
  RESUME_ARG="--resume ${OUT}/pixelmae_jax.npz"
  echo "measured: pulled a checkpoint from gs://${BUCKET}/${RUNS_PREFIX}/pixelmae_jax.npz" \
       "($(du -h "${OUT}/pixelmae_jax.npz" | cut -f1)), step" \
       "$("${PY}" -c 'import numpy,sys; print(int(numpy.load(sys.argv[1])["_step"]))' "${OUT}/pixelmae_jax.npz")"
  # Its own metrics too, so the shipped curve stays ONE curve across nodes
  # rather than restarting in mid-air on every preemption.
  gcs_get "${RUNS_PREFIX}/metrics.jsonl" "${OUT}/metrics.jsonl" \
    && echo "measured: pulled metrics.jsonl ($(wc -l < "${OUT}/metrics.jsonl") lines)" \
    || echo "no metrics.jsonl in the bucket yet — this run starts the curve"
  # A checkpoint WITHOUT its own architecture would be resumed into whatever
  # the flags below build; train_stage1.py adopts the architecture from the
  # checkpoint's args and REFUSES on a contradiction, which is the guard that
  # #395 did not have.
else
  rm -f "${OUT}/pixelmae_jax.npz"
  echo "measured: nothing at gs://${BUCKET}/${RUNS_PREFIX}/pixelmae_jax.npz —" \
       "this is a FRESH run, not a continuation. Say so in its doc string."
fi

# --------------------------------------------------------------------------
# 6 · the shipper
# --------------------------------------------------------------------------
( while true; do sleep $(( SHIP_EVERY_MIN * 60 )); ship_state; upload_log; done ) &
disown
echo "measured: shipper armed, every ${SHIP_EVERY_MIN} min to gs://${BUCKET}/${RUNS_PREFIX}/"

# --------------------------------------------------------------------------
# 7 · train
# --------------------------------------------------------------------------
echo "--- train ---"
cd "${WORK}/earth"
# CKPT_TAG lands in the checkpoint's `tag` field, so a rescued artefact says
# which run wrote it (ml/train.py:save_ckpt has the same field for the same
# reason).
export CKPT_TAG="${NODE}"
# shellcheck disable=SC2086
"${PY}" -u ml/jaxport/train_stage1.py \
  --data "${TF}" \
  --out "${OUT}" \
  --steps "${STEPS}" \
  --batch "${BATCH}" \
  --lr "${LR}" \
  --d-z "${D_Z}" \
  --patch "${PATCH}" \
  --d-model "${D_MODEL}" \
  --n-layers "${N_LAYERS}" \
  --n-heads "${N_HEADS}" \
  --d-dec "${D_DEC}" \
  --anomaly \
  --holdout-lon="${HOLDOUT_LON}" \
  ${TIME_BLOCK:+--time-block "${TIME_BLOCK}"} \
  --light-probe-every "${LIGHT_PROBE_EVERY}" \
  --ckpt-every "${CKPT_EVERY}" \
  ${RESUME_ARG} \
  ${EXTRA_ARGS}

echo "=== tpu_train done — the EXIT trap now ships the final state and deletes this node ==="

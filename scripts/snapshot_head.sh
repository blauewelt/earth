#!/usr/bin/env bash
# Push the stage-2 head's box-local mirror to the release, mid-training.
#
# temporal.py already mirrors the head to /opt/earth-cache/ckpt every metrics
# point, with optimiser moments, schedule position and RNG — everything a true
# continuation needs. But that mirror lives on ONE rented box. It survives a
# cancelled job; it does not survive the instance being destroyed, reclaimed,
# or simply failing, and a 200,000-step run is a full day of GPU to lose.
#
# Chris, 2026-08-10: "make sure to save all data at regular intervals such that
# we can continue experiments if we need to (eg, if a box crashes in the middle
# of the 200k experiment)."
#
# So the publisher loop calls this every ~30 minutes. 7.3 MB per upload, a few
# dozen times a day: negligible against $0.28/h of compute, and it converts
# "the box died, start again" into "resume from the last half hour".
#
# Usage: bash scripts/snapshot_head.sh <run_number>
# Needs GITHUB_TOKEN (the job token's contents:write is enough).
set -uo pipefail                 # NOT -e: this is best-effort by design...
RUN="${1:?usage: snapshot_head.sh <run_number>}"
REPO="${GITHUB_REPOSITORY:-blauewelt/earth}"
TAG="model-checkpoints-v1"
SRC="/opt/earth-cache/ckpt/${CKPT_TAG:+${CKPT_TAG}-}temporal.pt"
ASSET="run-${RUN}-temporal-latest.pt"

# ...but every branch below says WHY it gave up. A best-effort path that
# reports success while doing nothing is the failure mode that cost this
# project four separate incidents (CLAUDE.md 6c rule 6).
if [ ! -f "$SRC" ]; then
  echo "snapshot: no head mirror at $SRC yet — nothing to save"; exit 0
fi
if [ -z "${GITHUB_TOKEN:-}" ]; then
  echo "::warning::snapshot: GITHUB_TOKEN empty — the head is NOT being backed up"
  exit 0
fi

API="https://api.github.com"
AUTH=(-H "Authorization: token ${GITHUB_TOKEN}" -H "Accept: application/vnd.github+json")

REL=$(curl -fsSL "${AUTH[@]}" "${API}/repos/${REPO}/releases/tags/${TAG}") || {
  echo "::warning::snapshot: cannot read release ${TAG}"; exit 0; }
REL_ID=$(printf '%s' "$REL" | sed -n 's/.*"id": *\([0-9]*\).*/\1/p' | head -1)
[ -n "$REL_ID" ] || { echo "::warning::snapshot: no release id"; exit 0; }

# Replace, never accumulate: one asset per run, always the newest state.
OLD=$(printf '%s' "$REL" | tr '}' '\n' | grep -F "\"name\": \"${ASSET}\"" \
      | sed -n 's/.*"id": *\([0-9]*\).*/\1/p' | head -1)
if [ -n "$OLD" ]; then
  curl -fsSL -X DELETE "${AUTH[@]}" "${API}/repos/${REPO}/releases/assets/${OLD}" \
    || echo "::warning::snapshot: could not delete the previous ${ASSET}"
fi

# Copy first: temporal.py rewrites the mirror atomically, but uploading the
# live path could still race a replace mid-read.
CP="/tmp/${ASSET}"
cp "$SRC" "$CP" || { echo "::warning::snapshot: copy failed"; exit 0; }
STEP=$(python -c "
import torch,sys
try: print(torch.load('$CP', map_location='cpu', weights_only=False).get('step','?'))
except Exception as e: print('?')" 2>/dev/null)

if curl -fsSL -X POST "${AUTH[@]}" \
     -H "Content-Type: application/octet-stream" --data-binary "@${CP}" \
     "https://uploads.github.com/repos/${REPO}/releases/${REL_ID}/assets?name=${ASSET}" \
     >/dev/null; then
  echo "snapshot: ${ASSET} saved at step ${STEP} ($(du -h "$CP" | cut -f1)) — "\
"this run is now resumable from another box"
else
  echo "::warning::snapshot: upload of ${ASSET} FAILED — the head is only on this box"
fi
rm -f "$CP"

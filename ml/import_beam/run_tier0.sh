#!/usr/bin/env bash
# The real Tier-0 run, once.
#
#   export EARTH_REPO=/path/to/earth
#   export COPERNICUSMARINE_SERVICE_USERNAME=... PASSWORD=...   # for GLORYS
#   OUTPUT=/data/import bash run_tier0.sh
#
# For the usual case — keep going until nothing is left in the queue — use
# `run_until_complete.sh` instead; this script is one round of it.
#
# Credentials are read from the ENVIRONMENT by the workers. They are never
# passed on a command line and never become pipeline options: options are
# logged and shown in job UIs.
#
# The run is IDEMPOTENT. If it is interrupted — a breaker trip, a lost SSH
# session, a reboot — run exactly this command again. Everything with a
# `.done` marker is skipped.
set -uo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
PY="${PY:-$HERE/beamenv/bin/python}"
[ -x "$PY" ] || PY="$(command -v python3)"

OUTPUT="${OUTPUT:-}"
REPORT_DIR="${REPORT_DIR:-$HERE/out/tier0}"
STATE_DIR="${STATE_DIR:-/var/tmp/beam_import}"
WORKERS="${WORKERS:-8}"
ONLY_ARG=()
[ "${ONLY:-}" != "" ] && ONLY_ARG=(--only "$ONLY")

if [ -z "$OUTPUT" ]; then
  echo "Set OUTPUT to where the shards go, e.g."
  echo "  OUTPUT=/data/import bash run_tier0.sh"
  echo "  OUTPUT=gs://my-bucket/earth-import bash run_tier0.sh"
  exit 1
fi
if [ -z "${EARTH_REPO:-}" ]; then
  export EARTH_REPO="$HERE/../earth"
fi
if [ ! -d "$EARTH_REPO/ml" ]; then
  echo "EARTH_REPO=$EARTH_REPO has no ml/ directory."
  echo "Clone it:  git clone https://github.com/blauewelt/earth.git"
  exit 1
fi

mkdir -p "$REPORT_DIR" "$STATE_DIR"

echo "== registry check"
"$PY" -m beam_import.registry --check

echo
echo "== manifest (tier 0)"
"$PY" -m beam_import.manifest --tiers 0 "${ONLY_ARG[@]}" \
    --json "$REPORT_DIR/manifest_tier0.json" | tail -30

echo
echo "== running Stage A -> $OUTPUT"
echo "   watch it from another terminal with:"
echo "     $PY -m beam_import.report --live $STATE_DIR"
"$PY" -m beam_import.pipeline \
    --tiers 0 "${ONLY_ARG[@]}" \
    --output "$OUTPUT" \
    --report-dir "$REPORT_DIR" \
    --state-dir "$STATE_DIR" \
    --runner DirectRunner \
    --direct_running_mode multi_processing \
    --direct_num_workers "$WORKERS"
RC=$?

echo
echo "== live progress, one last time"
"$PY" -m beam_import.report --live "$STATE_DIR" || true

echo
echo "== verifying the output"
"$PY" -m beam_import.verify_output --tiers 0 "${ONLY_ARG[@]}" \
    --output "$OUTPUT" --state-dir "$STATE_DIR" \
    --json-out "$REPORT_DIR/verify_tier0.json" || true

echo
echo "Read $REPORT_DIR/summary.md."
if [ "$RC" -eq 4 ]; then
  echo "A host's breaker tripped TWICE — stop, and report (README §9)."
elif [ "$RC" -eq 3 ]; then
  echo "The queue is not empty. Nothing was lost: run run_until_complete.sh,"
  echo "or re-run this after at least an hour."
fi
exit "$RC"

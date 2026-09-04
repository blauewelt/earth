#!/usr/bin/env bash
# The real Tier-0 run.
#
#   export HF_TOKEN=...            (or put it in ~/.hf_token, mode 600)
#   export EARTH_REPO=/path/to/earth
#   bash run_tier0.sh
#
# Credentials are read from the ENVIRONMENT by the workers. They are never
# passed on a command line and never become pipeline options: options are
# logged and shown in the job UI.
#
# The run is IDEMPOTENT. If it is interrupted — a breaker trip, a lost SSH
# session, a reboot — run exactly this command again. Everything already on
# the Hub is skipped.
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
PY="${PY:-$HERE/beamenv/bin/python}"
[ -x "$PY" ] || PY="$(command -v python3)"

REPORT_DIR="${REPORT_DIR:-$HERE/out/tier0}"
STATE_DIR="${STATE_DIR:-/var/tmp/beam_import}"
WORKERS="${WORKERS:-8}"
ONLY_ARG=()
[ "${ONLY:-}" != "" ] && ONLY_ARG=(--only "$ONLY")

if [ -z "${EARTH_REPO:-}" ]; then
  export EARTH_REPO="$HERE/../earth"
fi
if [ ! -d "$EARTH_REPO/ml" ]; then
  echo "EARTH_REPO=$EARTH_REPO has no ml/ directory."
  echo "Clone it:  git clone https://github.com/blauewelt/earth.git"
  exit 1
fi
if [ -z "${HF_TOKEN:-}" ] && [ ! -f "$HOME/.hf_token" ]; then
  echo "No HF_TOKEN in the environment and no ~/.hf_token file."
  echo "See CREDENTIALS.md. Nothing was started."
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
echo "== running"
echo "   watch progress from another terminal with:"
echo "     python -m beam_import.report --live $STATE_DIR"
echo "   (report.jsonl is only written when the pipeline finishes; the live"
echo "    view reads the per-lane progress files, which are appended to as"
echo "    each item completes)"
"$PY" -m beam_import.pipeline \
    --tiers 0 \
    "${ONLY_ARG[@]}" \
    --report-dir "$REPORT_DIR" \
    --state-dir "$STATE_DIR" \
    --runner DirectRunner \
    --direct_running_mode multi_processing \
    --direct_num_workers "$WORKERS"

echo
echo "== live progress, one last time"
"$PY" -m beam_import.report --live "$STATE_DIR" || true

echo
echo "== verifying the tier against the Hub"
"$PY" -m beam_import.verify_hub --tiers 0 "${ONLY_ARG[@]}" \
    --report "$REPORT_DIR/report.jsonl" --out-dir "$REPORT_DIR" || true

echo
echo "Done. Read $REPORT_DIR/summary.md."
echo "If anything is 'deferred', wait at least an hour and run this again."

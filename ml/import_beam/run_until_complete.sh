#!/usr/bin/env bash
# Run Stage A until the retry queue is empty.
#
# This is the loop DESIGN §2 calls "run until complete": a throttle changes
# WHEN an item is fetched, never WHETHER, and this script is what turns that
# from a sentence into a promise. It runs the pipeline; if anything is still
# queued it sleeps (1 h, then 2, 4, 8 — the host is being given a night off)
# and runs again with the queue AS the manifest; it stops when the queue is
# empty.
#
#   OUTPUT=/data/import bash run_until_complete.sh --tiers 0
#
# Environment:
#   OUTPUT       where the shards go (required; or pass --output yourself)
#   STATE_DIR    default /var/tmp/beam_import
#   REPORT_DIR   default ./out/tier0
#   WORKERS      DirectRunner worker processes, default 8
#   SLEEP_S      override the sleep ladder, in seconds — the tests set it to 1
#   MAX_ROUNDS   stop after this many rounds anyway, default 12
#
# IT STOPS BY ITSELF, LOUDLY, in exactly one case besides success: a host
# whose circuit breaker tripped twice in one run (hard rule 7). That is a
# decision for a human, not for a loop.
set -uo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
PY="${PY:-$HERE/beamenv/bin/python}"
[ -x "$PY" ] || PY="$(command -v python3)"

STATE_DIR="${STATE_DIR:-/var/tmp/beam_import}"
REPORT_DIR="${REPORT_DIR:-$HERE/out/tier0}"
WORKERS="${WORKERS:-8}"
MAX_ROUNDS="${MAX_ROUNDS:-12}"
QUEUE="$STATE_DIR/retry_queue.jsonl"

OUT_ARG=()
[ -n "${OUTPUT:-}" ] && OUT_ARG=(--output "$OUTPUT")

# The ladder, in seconds. Capped at 8 h and repeated from there.
LADDER=(3600 7200 14400 28800)

run_round () {          # $1 = extra args
  # shellcheck disable=SC2086
  "$PY" -m beam_import.pipeline \
      "${OUT_ARG[@]}" \
      --state-dir "$STATE_DIR" \
      --report-dir "$REPORT_DIR" \
      --runner DirectRunner \
      --direct_running_mode multi_processing \
      --direct_num_workers "$WORKERS" \
      "$@"
}

mkdir -p "$STATE_DIR" "$REPORT_DIR"

echo "== round 1 (the full manifest)"
run_round "$@"
rc=$?
if [ "$rc" -eq 4 ]; then
  echo
  echo "STOPPED: a host's circuit breaker tripped twice in one run."
  echo "That is hard rule 7 — a human decides what happens next, not this"
  echo "script. Read $REPORT_DIR/summary.md, then report to Chris (README §9)."
  exit 4
fi

round=1
while : ; do
  n=$("$PY" - "$QUEUE" <<'PY'
import json, os, sys
path = sys.argv[1]
seen = {}
if os.path.exists(path):
    for line in open(path, encoding="utf-8"):
        line = line.strip()
        if line:
            try:
                seen[json.loads(line)["item_id"]] = 1
            except Exception:
                pass
print(len(seen))
PY
)
  echo "== queue after round $round: $n item(s)"
  [ "$n" -eq 0 ] && break
  if [ "$round" -ge "$MAX_ROUNDS" ]; then
    echo "STOPPED: $MAX_ROUNDS rounds and the queue is still $n item(s)."
    echo "Nothing was lost — the queue is at $QUEUE. Report to Chris."
    exit 5
  fi
  idx=$(( round - 1 ))
  [ "$idx" -ge ${#LADDER[@]} ] && idx=$(( ${#LADDER[@]} - 1 ))
  sleep_s="${SLEEP_S:-${LADDER[$idx]}}"
  echo "   sleeping ${sleep_s}s before round $(( round + 1 )) — the hosts get"
  echo "   a rest; nothing is lost by waiting."
  sleep "$sleep_s"

  round=$(( round + 1 ))
  echo "== round $round (the queue is the manifest)"
  run_round "$@" --from-queue "$QUEUE"
  rc=$?
  if [ "$rc" -eq 4 ]; then
    echo
    echo "STOPPED: a host's circuit breaker tripped twice in one run (rule 7)."
    exit 4
  fi
done

echo
echo "QUEUE EMPTY after $round round(s). Every item is `written`, `present`"
echo "or `absent` with evidence. Now verify:"
echo "  $PY -m beam_import.verify_output --tiers 0 ${OUT_ARG[*]} --state-dir $STATE_DIR"

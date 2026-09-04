#!/usr/bin/env bash
# The offline smoke test. NO NETWORK IS USED — every "upstream" URL is a
# file:// URL pointing at a fixture this script generates, and the output is a
# local directory. Run it before every real run; it takes about a minute.
#
#   bash run_smoke.sh
#
# It exercises, in order:
#   Stage A          — an opaque file, a 0.25° ocean month, an OISST year with
#                      a DELIBERATELY missing day, three NCEP files, one
#                      Roemmich-Gilson month, and a lane that always fails
#   the queue        — the flaky lane's items and the missing day must be
#                      `queued`, never dropped
#   run_until_complete — one extra round, with the sleep overridden to 1 s
#   Stage B          — the pentad assembly, all three channel groups
#   verify_output    — markers, byte counts, the absent list
#
# It must end with:  SMOKE TEST: PASS
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
PY="${PY:-$HERE/beamenv/bin/python}"
[ -x "$PY" ] || PY="$(command -v python3)"
WORK="${WORK:-$(mktemp -d /tmp/beam_import_smoke.XXXXXX)}"
export EARTH_REPO="${EARTH_REPO:-$HERE/../earth}"

echo "== python      $PY"
echo "== earth repo  $EARTH_REPO"
echo "== work dir    $WORK"
cd "$HERE"

echo
echo "== 1/6  building the fixtures"
"$PY" tests/fixtures/make_fixtures.py "$WORK/fixtures"
REG="$WORK/fixtures/sources_test.yaml"
OUT="$WORK/out"
STATE="$WORK/state"

echo
echo "== 2/6  checking the test registry and the manifest"
"$PY" -m beam_import.registry --registry "$REG" --check
"$PY" -m beam_import.manifest --registry "$REG" --tiers 0 --offline | tail -6

echo
echo "== 3/6  Stage A on the DirectRunner, 2 worker processes"
set +e
"$PY" -m beam_import.pipeline \
    --registry "$REG" --tiers 0 \
    --output "$OUT" --state-dir "$STATE" --report-dir "$WORK/report" \
    --offline --runner DirectRunner \
    --direct_running_mode multi_processing --direct_num_workers 2
RC=$?
set -e
echo "   (exit $RC — 3 means 'the queue is not empty', which is expected here)"

echo
echo "== 4/6  run_until_complete, ONE extra round, sleep overridden to 1 s"
OUTPUT="$OUT" STATE_DIR="$STATE" REPORT_DIR="$WORK/report" WORKERS=2 \
  SLEEP_S=1 MAX_ROUNDS=2 PY="$PY" \
  bash run_until_complete.sh --registry "$REG" --tiers 0 --offline || true

echo
echo "== 5/6  Stage B — the pentad assembly"
"$PY" -m beam_import.assemble --registry "$REG" --output "$OUT" \
    --runner DirectRunner | tee "$WORK/coverage.txt"

echo
echo "== 6/6  verify_output"
set +e
"$PY" -m beam_import.verify_output --registry "$REG" --tiers 0 \
    --output "$OUT" --state-dir "$STATE" --offline --deep
set -e

echo
echo "== checking what the run produced"
"$PY" - "$WORK" <<'PY'
import json, os, sys
sys.path.insert(0, os.getcwd())
from beam_import import report, sinks, tfrecord
from beam_import.example import one_int, one_str, parse_example, str_list

work = sys.argv[1]
out, state = os.path.join(work, "out"), os.path.join(work, "state")
# The union of every round's live progress files and the final report, so a
# multi-round run is judged on what it did overall, not on its last round.
recs = report.collect(os.path.join(work, "report", "report.jsonl"), state)
items = [r for r in recs if r["status"] != "counters"]
st = {}
for r in items:
    st.setdefault(r["status"], []).append(r["item_id"])

def need(cond, msg):
    print(("  ok   " if cond else "  FAIL ") + msg)
    if not cond:
        sys.exit(1)

markers = sinks.list_done(out)
need(set(markers) >= {"tiny/hello.dat", "ocean/ocean_200307.nc",
                      "oisst/2003", "ncep/uflx/2003", "ncep/skt/2003",
                      "ncep/land.sfc.gauss.nc", "rg/RG_ArgoClim_200307.nc"},
     "Stage A wrote every good source (opaque, 0.25° ocean, OISST, NCEP, RG)")
need(all(len(m.get("sha256") or "") == 64 for m in markers.values()),
     "every marker carries the sha256 its read-back compared")
need(markers["oisst/2003"]["missing_dates"] == ["2003-07-17"],
     "the OISST year kept its four days and recorded the missing one")
need(markers["oisst/2003"]["n_records"] == 4,
     "the shard holds the four days that WERE served")
need("failed" not in st, "there is no `failed` status anywhere")

# the queue: the flaky lane plus the missing day, and NOTHING dropped
qfile = os.path.join(state, "retry_queue.jsonl")
rotated = sorted(f for f in os.listdir(state) if f.startswith("retry_queue."))
allq = set()
for f in rotated + (["retry_queue.jsonl"] if os.path.exists(qfile) else []):
    for line in open(os.path.join(state, f), encoding="utf-8"):
        if line.strip():
            allq.add(json.loads(line)["item_id"])
need("oisst/2003/2003-07-17" in allq,
     "the missing day was queued as a day-level item")
need(len([q for q in allq if q.startswith("flaky/")]) == 7,
     "all seven flaky items reached the queue (breaker + ladder), none lost")

trips = max([int((r.get("counters") or {}).get("trips", 0))
             for r in recs if r["status"] == "counters"] or [0])
need(trips == 1, "one circuit-breaker trip per round was recorded")
need(os.path.exists(os.path.join(state, "retry_queue.1.jsonl")),
     "run_until_complete rotated round 1's queue and re-ran from it")

# Stage B
cov = json.load(open(os.path.join(out, "pentad", "coverage.json")))
need(all(cov["groups"][g]["bins_present"] == 1
         for g in ("g025", "g100", "rg100")),
     "Stage B produced all three channel groups for the fixture's one bin")
need(cov["groups"]["g025"]["bin_min"] == 1573,
     "the bin is 1573 — floor(day_index/5) from 1982-01-01, imported")

uris = tfrecord.list_uris(os.path.join(out, "pentad", "g025"), ".tfrecord")
payloads = [p for u in uris for p in tfrecord.read_records(u)]
need(len(payloads) == 1, "one g025 record for one bin")
rec = parse_example(payloads[0])
dp = list(rec["days_present"])
need(dp[:5] == [5, 5, 5, 5, 5] and dp[6] == 4,
     f"days_present is per channel and honest: {dp}")
need(str_list(rec, "chan_names")[0] == "cur_speed",
     "the channel order is build_family7's, imported")
print("  --   coverage: " + open(os.path.join(work, "coverage.txt")).read()
      .strip().replace("\n", " | "))
PY

echo
echo "SMOKE TEST: PASS"
echo "(artefacts in $WORK — delete it when you are done)"

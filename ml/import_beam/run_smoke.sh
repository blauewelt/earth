#!/usr/bin/env bash
# The offline smoke test. NO NETWORK IS USED — every "upstream" URL is a
# file:// URL pointing at a fixture this script generates, and the "Hub" is a
# local directory. Run it before every real run; it takes about half a minute.
#
#   bash run_smoke.sh
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
echo "== 1/4  building the fixtures"
"$PY" tests/fixtures/make_fixtures.py "$WORK/fixtures"
REG="$WORK/fixtures/sources_test.yaml"

echo
echo "== 2/4  checking the test registry"
"$PY" -m beam_import.registry --registry "$REG" --check

echo
echo "== 3/4  the manifest"
"$PY" -m beam_import.manifest --registry "$REG" --tiers 0 --offline

echo
echo "== 4/4  the pipeline on the DirectRunner, 2 worker processes"
"$PY" -m beam_import.pipeline \
    --registry "$REG" \
    --tiers 0 \
    --hub "local:$WORK/hub" \
    --report-dir "$WORK/out" \
    --state-dir "$WORK/state" \
    --offline \
    --runner DirectRunner \
    --direct_running_mode multi_processing \
    --direct_num_workers 2

echo
echo "== checking what the run produced"
"$PY" - "$WORK" <<'PY'
import json, os, sys
work = sys.argv[1]
recs = [json.loads(l) for l in
        open(os.path.join(work, "out", "report.jsonl"), encoding="utf-8")
        if l.strip()]
items = [r for r in recs if r["status"] != "counters"]
st = {}
for r in items:
    st.setdefault(r["status"], []).append(r["item_id"])

def need(cond, msg):
    print(("  ok   " if cond else "  FAIL ") + msg)
    if not cond:
        sys.exit(1)

need(sorted(st.get("published", [])) ==
     ["ncep/uflx.sfc.gauss/2001", "oisst/2001", "tiny/hello.dat"],
     "three sources published (file:// http, OISST year-fold, NCEP-like)")
need(all(len(r["sha256"] or "") == 64 for r in items
         if r["status"] == "published"),
     "every published item carries the sha256 the restore-verify compared")
need(len(st.get("failed", [])) == 5,
     "the flaky source failed exactly 5 items (the breaker threshold)")
need(len(st.get("deferred", [])) == 2,
     "the rest of that lane is `deferred`, not `failed`")
trips = sum((r.get("counters") or {}).get("trips", 0) for r in recs
            if r["status"] == "counters")
need(trips == 1, "exactly one circuit-breaker trip was recorded")
hub = os.path.join(work, "hub")
on_hub = sorted(os.path.relpath(os.path.join(d, f), hub)
                for d, _s, fs in os.walk(hub) for f in fs)
need("sources/oisst/oisst_daily_2001.nc" in on_hub,
     "the folded OISST year is on the fake Hub")
need("sources/_preflight/roundtrip.txt" in on_hub,
     "the Hub round-trip preflight ran")
need(os.path.exists(os.path.join(work, "out", "summary.md")),
     "summary.md was written")
prog = os.path.join(work, "state", "progress")
files = sorted(os.listdir(prog)) if os.path.isdir(prog) else []
live = sum(1 for f in files
           for l in open(os.path.join(prog, f), encoding="utf-8") if l.strip())
need(len(files) == 4 and live == len(recs),
     f"live progress: one file per lane ({len(files)}), {live} records "
     "appended during the run")
PY

echo
echo "== the live mid-run view of the same run"
"$PY" -m beam_import.report --live "$WORK/state" | head -20

echo
echo "SMOKE TEST: PASS"
echo "(artefacts in $WORK — delete it when you are done)"

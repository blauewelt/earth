#!/usr/bin/env bash
# Record WHICH PHASE a run is in, and push it where the status page can see it.
#
#   bash scripts/publish_phase.sh ml-live-42 "building the tensor" "…detail…"
#
# Chris, 2026-08-09: "many experiments say they are in the data building
# stage. is that correct?" It was not. The page had exactly one message for
# every run with no curves yet — "building dataset / seeding cache" — so a
# QUEUED job waiting for a free runner, a job running the probe ladder, and a
# job genuinely building a tensor all read identically, and the one that was
# actually building was usually the minority. A run that cannot say what it is
# doing invites exactly this misreading.
#
# Cheap by construction: a handful of small pushes per run, on a branch that
# is deleted when the job ends.
set -e
BRANCH="$1"
PHASE="$2"
DETAIL="${3:-}"
mkdir -p ml/runs/actions
PHASE="$PHASE" DETAIL="$DETAIL" python3 - <<'EOF' > ml/runs/actions/phase.json
import json, os, time
json.dump({"phase": os.environ["PHASE"],
           "detail": os.environ.get("DETAIL", ""),
           "at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())},
          __import__("sys").stdout)
EOF
# Never let a phase push fail a job: this is signage, not science.
bash scripts/publish_live_metrics.sh "$BRANCH" || true

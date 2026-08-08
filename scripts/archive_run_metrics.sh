#!/usr/bin/env bash
# Move a finished run's metrics.jsonl onto the LONG-LIVED `ml-metrics`
# branch, so its curves survive the run.
#
# WHY. The per-run `ml-live-<n>` branch is deleted when the job ends (it is
# a scratch channel for plotting a run WHILE it trains). That meant a
# successful run's loss and probe curves vanished the moment it succeeded —
# exactly the runs whose curves you most want to look at later. Chris
# caught this on the status page: "is it intentional that successful
# experiments have their plots removed?" It was not.
#
# Design: ONE branch, one file per run (`run-<n>.jsonl`), appended by
# fetching the branch and adding a file. The files are a few kB each, so
# the branch stays small for hundreds of runs, and a single well-known
# path keeps the status page's fetch trivial:
#   raw.githubusercontent.com/<repo>/ml-metrics/run-<n>.jsonl
# The per-run scratch branch is still deleted afterwards by the workflow.
#
# Usage (in the workflow, after training, before cleanup):
#   bash scripts/archive_run_metrics.sh <run_number>
set -e
N="$1"
F=ml/runs/actions/metrics.jsonl
[ -n "$N" ] || { echo "usage: archive_run_metrics.sh <run_number>"; exit 2; }
[ -s "$F" ] || { echo "no metrics to archive"; exit 0; }

REPO_URL="https://x-access-token:${GITHUB_TOKEN}@github.com/${GITHUB_REPOSITORY}.git"
DIR=$(mktemp -d)
cp "$F" "$DIR/run-$N.jsonl"
cd "$DIR"
git init -q
git config user.email "ml-live@users.noreply.github.com"
git config user.name "ml-live"
# Fetch the existing archive if there is one; start fresh if not. Both
# paths end with the branch checked out and our new file staged.
if git fetch -q --depth=1 "$REPO_URL" ml-metrics 2>/dev/null; then
  git checkout -q FETCH_HEAD
  git checkout -q -b ml-metrics
else
  git checkout -q -b ml-metrics
  cat > README.md <<'EOF'
# ml-metrics — the training curves of every ml-train run

One file per run: `run-<run_number>.jsonl`, the metrics.jsonl the job
produced (loss points every few hundred steps; probe points at each
eval interval). Read by status.html so a completed run keeps its charts.

This branch is append-only bookkeeping and is never merged into main.
EOF
  git add README.md
fi
cp "$DIR/run-$N.jsonl" . 2>/dev/null || true
git add "run-$N.jsonl"
git commit -q -m "metrics for run #$N" || { echo "nothing new to commit"; exit 0; }
# Not forced: concurrent runs each add their own file, so a rejected push
# means someone else archived first — refetch and retry once.
git push -q "$REPO_URL" HEAD:refs/heads/ml-metrics 2>/dev/null || {
  echo "push raced; retrying once …"
  git fetch -q --depth=1 "$REPO_URL" ml-metrics
  git reset -q --soft FETCH_HEAD
  git commit -q --amend -m "metrics for run #$N"
  git push -q "$REPO_URL" HEAD:refs/heads/ml-metrics
}
echo "archived run #$N metrics to the ml-metrics branch"

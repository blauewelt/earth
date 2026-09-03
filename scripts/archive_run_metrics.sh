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
# THE SECOND FILE, added 2026-09-03 for E-069. A cone run with
# `--snapshot-ablation` trains a second arm in-process — the L_in=0 SNAPSHOT
# TWIN, the control H1 is measured against — and that arm's curve goes to
# ml/runs/actions/metrics_snapshot.jsonl, which nothing archived. So #537
# left the experiment's treatment curve on a permanent branch and its
# control curve on a box that no longer exists: the comparison was archived
# half. It rides here as `run-<n>-snapshot.jsonl`, in the same commit, and
# an ordinary run simply has no such file.
#
# Usage (in the workflow, after training, before cleanup):
#   bash scripts/archive_run_metrics.sh <run_number>
set -e
N="$1"
F=ml/runs/actions/metrics.jsonl
S=ml/runs/actions/metrics_snapshot.jsonl
[ -n "$N" ] || { echo "usage: archive_run_metrics.sh <run_number>"; exit 2; }
[ -s "$F" ] || [ -s "$S" ] || { echo "no metrics to archive"; exit 0; }

REPO_URL="https://x-access-token:${GITHUB_TOKEN}@github.com/${GITHUB_REPOSITORY}.git"
DIR=$(mktemp -d)
[ -s "$F" ] && cp "$F" "$DIR/run-$N.jsonl"
[ -s "$S" ] && cp "$S" "$DIR/run-$N-snapshot.jsonl"
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
# `git checkout` above may have replaced the working tree with the fetched
# branch, so re-place both files (they were staged into $DIR before the
# checkout) and add whichever exist. ASSERT THE EFFECT (ml/CLAUDE.md §0.2):
# `git add` of a missing path is a hard error under `set -e`, so a file that
# was never copied must not be named.
for f in "run-$N.jsonl" "run-$N-snapshot.jsonl"; do
  [ -f "$DIR/$f" ] || continue
  cp "$DIR/$f" . 2>/dev/null || true
  git add "$f"
done
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
echo "archived run #$N metrics to the ml-metrics branch:$(
  [ -f "run-$N.jsonl" ] && echo " run-$N.jsonl ($(wc -l < "run-$N.jsonl") lines)")$(
  [ -f "run-$N-snapshot.jsonl" ] && echo " run-$N-snapshot.jsonl ($(wc -l < "run-$N-snapshot.jsonl") lines, the E-069 snapshot twin)")"

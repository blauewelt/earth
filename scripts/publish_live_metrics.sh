#!/usr/bin/env bash
# Push the in-progress metrics.jsonl to a tiny orphan branch so training
# curves can be plotted WHILE the run is still going (ml/live_curve.py
# fetches it from raw.githubusercontent.com). Called in a loop beside
# train.py by .github/workflows/ml-train.yml; needs contents: write.
set -e
BRANCH="$1"
F=ml/runs/actions/metrics.jsonl
[ -s "$F" ] || exit 0
DIR=$(mktemp -d)
cp "$F" "$DIR/metrics.jsonl"
cd "$DIR"
git init -q -b "$BRANCH" 2>/dev/null || { git init -q && git checkout -q -b "$BRANCH"; }
git config user.email "ml-live@users.noreply.github.com"
git config user.name "ml-live"
git add metrics.jsonl
git commit -q -m "live metrics $(date -u +%Y-%m-%dT%H:%MZ)"
git push -q -f "https://x-access-token:${GITHUB_TOKEN}@github.com/${GITHUB_REPOSITORY}.git" \
  "HEAD:refs/heads/${BRANCH}"
rm -rf "$DIR"

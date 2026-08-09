#!/usr/bin/env bash
# Publish a running job's live state to its orphan `ml-live-<n>` branch.
#
# Two files, both optional:
#   metrics.jsonl — the curves (loss, probes, stage 2)
#   phase.json    — WHICH WORKFLOW STEP the job is in right now
#
# phase.json exists because the status page is credential-free by design: it
# reads public raw branch content and never authenticates. The GitHub API
# would tell it the current step, but the page already spends its whole
# unauthenticated budget (60 req/h) on the runs and releases calls at a 2-min
# refresh, so per-run job lookups would rate-limit it into a banner. Pushing
# the phase into a file the page already knows how to fetch costs the page
# nothing — raw.githubusercontent is not part of that budget.
#
# Both files are published TOGETHER on every call because this force-pushes an
# orphan branch: publishing one alone would delete the other.
set -e
BRANCH="$1"
M=ml/runs/actions/metrics.jsonl
P=ml/runs/actions/phase.json
[ -s "$M" ] || [ -s "$P" ] || exit 0
DIR=$(mktemp -d)
[ -s "$M" ] && cp "$M" "$DIR/metrics.jsonl"
[ -s "$P" ] && cp "$P" "$DIR/phase.json"
cd "$DIR"
git init -q -b "$BRANCH" 2>/dev/null || { git init -q && git checkout -q -b "$BRANCH"; }
git config user.email "ml-live@users.noreply.github.com"
git config user.name "ml-live"
git add -A
git commit -q -m "live state $(date -u +%Y-%m-%dT%H:%MZ)"
git push -q -f "https://x-access-token:${GITHUB_TOKEN}@github.com/${GITHUB_REPOSITORY}.git" \
  "HEAD:refs/heads/${BRANCH}"
rm -rf "$DIR"

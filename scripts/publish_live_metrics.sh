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
# BE LOUD. Every caller invokes this as `... || true`, because a failed
# telemetry push must never kill a training job — which means a silent
# failure here is invisible for the whole run. #100's joint step had no
# GITHUB_TOKEN in its env, so every push authenticated as nobody, failed,
# and vanished into that `|| true`; the status page showed an empty chart
# for 48 minutes and nothing anywhere said why. Say it on stderr instead.
if [ -z "${GITHUB_TOKEN:-}" ]; then
  echo "publish_live_metrics: GITHUB_TOKEN is EMPTY — this step is missing" \
       "'GITHUB_TOKEN: \${{ github.token }}' in its env: block, so nothing" \
       "will ever reach $BRANCH" >&2
  exit 1
fi
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
if git push -q -f "https://x-access-token:${GITHUB_TOKEN}@github.com/${GITHUB_REPOSITORY}.git" \
     "HEAD:refs/heads/${BRANCH}"; then
  echo "published $(wc -l < metrics.jsonl 2>/dev/null || echo 0) metric lines to $BRANCH"
else
  echo "publish_live_metrics: push to $BRANCH FAILED (see git error above)" >&2
  rm -rf "$DIR"; exit 1
fi
rm -rf "$DIR"

#!/usr/bin/env bash
# Publish ARBITRARY small files to an orphan live branch, while a job runs.
#
#   publish_live_files.sh <branch> <file>...
#
# `scripts/publish_live_metrics.sh` does the same thing for a TRAINING run and
# knows the four file names it ships. This is the general form, written for the
# family-8 build (E-076), whose live state is a different set of files —
# `progress.json`, `index_stats.json` and a tail of the build log — and whose
# branch is `ml-live-f8-<run>` rather than `ml-live-<run>`. Those two
# namespaces MUST NOT be shared: `ml-live-<n>` is keyed by ml-train.yml's run
# numbers, and family8-build has its own counter, so a family-8 job would sooner
# or later force-push over a live training branch the status page was reading.
#
# WHY IT EXISTS AT ALL. ml/CLAUDE.md §5.25: progress is an ARTEFACT, not a log
# line — a progress line in a log the box takes with it is not progress
# anybody has. The family-8 profiles stage is a multi-hour, download-bound
# pull; its first run was cancelled after twenty minutes precisely because
# somebody had to guess, from the outside, whether it was making progress. The
# builder already rewrites `progress.json` atomically at every phase boundary,
# so shipping it every three minutes turns that guess into a fetch.
#
# ARGUMENTS ARE PATHS; the branch gets their BASENAMES. A file that is absent
# or empty is skipped, so a caller may name files that only exist later in the
# job (index_stats.json before the index stage finishes) without a guard.
#
# EVERY FILE GOES ON EVERY CALL, because this force-pushes an ORPHAN branch:
# publishing one alone would delete the others.
set -e
BRANCH="$1"
shift || true

# BE LOUD. Every caller invokes this as `... || true`, because a failed
# telemetry push must never kill the build it is reporting on — which means a
# silent failure here is invisible for the whole run. Run #100's joint step had
# no GITHUB_TOKEN in its env, so every push authenticated as nobody, failed,
# and vanished into that `|| true`; the status page showed an empty chart for
# 48 minutes and nothing anywhere said why. Say it on stderr instead.
if [ -z "${GITHUB_TOKEN:-}" ]; then
  echo "publish_live_files: GITHUB_TOKEN is EMPTY — this step is missing" \
       "'GITHUB_TOKEN: \${{ github.token }}' in its env: block, so nothing" \
       "will ever reach $BRANCH" >&2
  exit 1
fi
if [ -z "${BRANCH}" ] || [ "$#" -eq 0 ]; then
  echo "publish_live_files: usage: publish_live_files.sh <branch> <file>..." >&2
  exit 1
fi
if [ -z "${GITHUB_REPOSITORY:-}" ]; then
  echo "publish_live_files: GITHUB_REPOSITORY is EMPTY — there is no repo to" \
       "push $BRANCH to" >&2
  exit 1
fi

DIR=$(mktemp -d)
N=0
for f in "$@"; do
  # `cp` of a file the writer replaces with os.replace() can only ever see a
  # complete version — which is why the builder writes progress.json that way
  # (ml/CLAUDE.md §5.25) and why no locking is needed here.
  if [ -s "$f" ]; then
    cp "$f" "$DIR/$(basename "$f")"
    N=$((N + 1))
  fi
done
if [ "$N" -eq 0 ]; then
  echo "publish_live_files: none of $# named file(s) exist yet — nothing to" \
       "publish to $BRANCH (this is normal in a job's first minutes)" >&2
  rm -rf "$DIR"
  exit 0
fi

cd "$DIR"
git init -q -b "$BRANCH" 2>/dev/null || { git init -q && git checkout -q -b "$BRANCH"; }
git config user.email "ml-live@users.noreply.github.com"
git config user.name "ml-live"
git add -A
git commit -q -m "live state $(date -u +%Y-%m-%dT%H:%MZ)"
if git push -q -f "https://x-access-token:${GITHUB_TOKEN}@github.com/${GITHUB_REPOSITORY}.git" \
     "HEAD:refs/heads/${BRANCH}"; then
  echo "published ${N} file(s) to ${BRANCH}: $(ls | tr '\n' ' ')"
else
  echo "publish_live_files: push to $BRANCH FAILED (see git error above)" >&2
  rm -rf "$DIR"; exit 1
fi
rm -rf "$DIR"

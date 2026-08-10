#!/usr/bin/env bash
# Free disk on a box WITHOUT deleting anything that exists only there.
#
# Chris, 2026-08-10: "it's fine to cleanup backed up checkpoints that won't get
# used by ongoing queued runs... do _never_ cleanup the embeddings unless these
# are backed up (similar to checkpoints)."
#
# The rule this replaces deleted by SIZE, which meant it always reached for the
# embedding cache first — the single most expensive artefact on the box (~95
# minutes of a 4090) and, until today, the only one with nowhere else to live.
# Delete it and the next run rebuilds 5.2 GiB, which puts the box back under
# the threshold and queues up the next deletion. A treadmill driven by the
# cheapest possible heuristic.
#
# The rule now is BACKED UP OR UNTOUCHED, in tiers:
#
#   tier 0  free           runner logs, pip cache, actions _temp, *.partial
#   tier 1  reclaimable    artefacts CONFIRMED present in a release
#   tier 2  never          anything else, including every unpublished Z_*.npy
#
# Tier 1 is confirmed by asking the release, not by assuming: a HEAD against
# the asset URL. Assert the effect, not the invocation — the whole reason the
# checkpoint seed silently did nothing for three runs was a step that believed
# its own intentions.
#
# Usage: bash scripts/disk_hygiene.sh [min_free_gb]
set -uo pipefail
MIN_FREE_GB="${1:-16}"
REPO="${GITHUB_REPOSITORY:-blauewelt/earth}"
CACHE=/opt/earth-cache
free_gb() { df -BG --output=avail / | tail -1 | tr -dc 0-9; }

published() {   # published <release-tag> <asset-name>  -> 0 if it exists
  curl -fsSLI --max-time 60 -o /dev/null \
    "https://github.com/${REPO}/releases/download/$1/$2" 2>/dev/null
}

echo "disk hygiene: $(free_gb) GB free, want ${MIN_FREE_GB} GB"
[ "$(free_gb)" -ge "$MIN_FREE_GB" ] && { echo "  nothing to do"; exit 0; }

# ---- tier 0: pure scratch, never the only copy of anything -------------------
echo "  tier 0 — scratch"
rm -rf /opt/runner/_diag/*.log /opt/runner/_work/_temp/* 2>/dev/null
pip cache purge >/dev/null 2>&1
# A .partial with no .progress marker beside it cannot be resumed, so it is
# scratch. One WITH a marker is a half-built embedding somebody can continue —
# that is tier 2, and deleting it would throw away up to 95 minutes of GPU.
# UPLOAD PARTS ARE PURE SCRATCH — a byte-for-byte copy of a range already in
# the .npy beside them. On 2026-08-10 an ENOSPC while chunking left one behind
# and the box went to 50/50: every later job died in "Set up job", before any
# step, so THIS script could never run to clean it. That is the shape of the
# trap — the cleanup lives inside the thing the mess prevents from starting —
# and it is why embed_cache_sync now refuses to start a chunk it cannot fit.
for f in "$CACHE"/Z_*.up; do
  [ -e "$f" ] || continue
  sz=$(du -h "$f" | cut -f1)
  rm -f "$f" && echo "    freed $(basename "$f") ($sz) — an abandoned upload chunk"
done

for f in "$CACHE"/Z_*.npy.partial; do
  [ -e "$f" ] || continue
  if [ -e "$f.progress" ]; then
    echo "    KEEP $(basename "$f") — resumable ($(grep -o '"months_done": *[0-9]*' "$f.progress" | tr -dc 0-9) months done)"
  else
    rm -f "$f" && echo "    freed $(basename "$f") (no progress marker, not resumable)"
  fi
done
echo "  after tier 0: $(free_gb) GB free"
[ "$(free_gb)" -ge "$MIN_FREE_GB" ] && exit 0

# ---- tier 1: reclaimable ONLY where a release confirms a second copy ---------
echo "  tier 1 — artefacts confirmed present in a release"

# Embedding caches. Named Z_<run>_<hash>.npy locally, published as Z_<hash>.npy
# in chunks; the hash is the codec's identity, so the local name tells us
# exactly which asset to ask for.
for f in "$CACHE"/Z_*.npy; do
  [ -e "$f" ] || continue
  b=$(basename "$f")
  hash=$(printf '%s' "$b" | sed -n 's/.*_\([0-9a-f]\{10\}\)\.npy$/\1/p')
  sz=$(du -h "$f" | cut -f1)
  if [ -z "$hash" ]; then
    echo "    KEEP $b ($sz) — cannot parse a codec hash, so cannot confirm a backup"
    continue
  fi
  if published embed-cache-v1 "Z_${hash}.npy.aa"; then
    rm -f "$f" && echo "    freed $b ($sz) — published as Z_${hash}.npy.*, re-pullable"
  else
    echo "    KEEP $b ($sz) — NOT in embed-cache-v1. This is ~95 min of GPU and"
    echo "         the only copy; it is never deleted to make room."
  fi
done
echo "  after embeddings: $(free_gb) GB free"
[ "$(free_gb)" -ge "$MIN_FREE_GB" ] && exit 0

# Rescued orphan checkpoints: copies of copies, and the rescue step uploads
# them. Only drop the ones the release confirms.
for f in "$CACHE"/ckpt/orphan-*; do
  [ -e "$f" ] || continue
  b=$(basename "$f")
  if published model-checkpoints-v1 "rescued-$b"; then
    rm -f "$f" && echo "    freed ckpt/$b — published as rescued-$b"
  else
    echo "    KEEP ckpt/$b — not confirmed in model-checkpoints-v1"
  fi
done

echo "disk hygiene: finished with $(free_gb) GB free"
if [ "$(free_gb)" -lt "$MIN_FREE_GB" ]; then
  echo "::warning::still below ${MIN_FREE_GB} GB. Everything left is either in"
  echo "use or has no second copy. Publish it, or rent a box with a bigger disk"
  echo "(vast cannot resize one — see ml/CLAUDE.md §7)."
fi
exit 0

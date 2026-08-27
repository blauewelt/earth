#!/usr/bin/env bash
# Push the stage-2 head's box-local mirror to the release, mid-training.
#
# temporal.py already mirrors the head to /opt/earth-cache/ckpt every metrics
# point, with optimiser moments, schedule position and RNG — everything a true
# continuation needs. But that mirror lives on ONE rented box. It survives a
# cancelled job; it does not survive the instance being destroyed, reclaimed,
# or simply failing, and a 200,000-step run is a full day of GPU to lose.
#
# Chris, 2026-08-10: "make sure to save all data at regular intervals such that
# we can continue experiments if we need to (eg, if a box crashes in the middle
# of the 200k experiment)."
#
# So the publisher loop calls this every ~30 minutes. 7.3 MB per upload, a few
# dozen times a day: negligible against $0.28/h of compute, and it converts
# "the box died, start again" into "resume from the last half hour".
#
# THE 2 GiB CAP (measured 2026-08-27). GitHub release assets are hard-capped
# at 2 GiB: uploads.github.com answers 422 `"size must be less than
# 2147483648"` at HEADERS time, so the refusal is INSTANT — no 30-minute
# transfer, no partial asset, just a curl that fails in a second and a warning
# nobody reads. An xl-tier stage-2 head with optimiser state is ~2.3-2.6 GiB,
# which means this script's mirror upload has NEVER ONCE succeeded at that
# tier. The last successful mirror was run-423's 1.076 GiB 88M head. For a
# month the largest models in the programme had exactly one durable copy: the
# rented box's disk (docs/INFRASTRUCTURE.md §2b, "the 2 GiB cliff").
#
# So: anything over CHUNK_BYTES is uploaded SPLIT. The release then carries
#   run-<n>-temporal-latest.pt.partaa, .partab, …   (split -b, -a 2)
#   run-<n>-temporal-latest.manifest                (uploaded LAST)
# and the manifest — size, whole-file sha256, step, the part names in order,
# and a `cat …part* > temporal.pt` hint — is the MARKER that every part is
# durable (ml/CLAUDE.md §5.21, flush THEN mark). No manifest means do not
# trust the parts. A file at or under the cap keeps the old single-asset name
# unchanged, and the two representations are never allowed to coexist.
# (The archived-checkpoint convention for the same problem is the older
# `<tag>__temporal.full.partNN` + `.sha256` pair; this is its live-run
# sibling, kept to the name the publisher loop and its readers already use.)
#
# Usage: bash scripts/snapshot_head.sh <run_number>
# Needs GITHUB_TOKEN (the job token's contents:write is enough).
# SNAPSHOT_CHUNK_BYTES overrides the split threshold (testing only).
set -uo pipefail                 # NOT -e: this is best-effort by design...
RUN="${1:?usage: snapshot_head.sh <run_number>}"
REPO="${GITHUB_REPOSITORY:-blauewelt/earth}"
TAG="model-checkpoints-v1"
SRC="/opt/earth-cache/ckpt/${CKPT_TAG:+${CKPT_TAG}-}temporal.pt"
ASSET="run-${RUN}-temporal-latest.pt"
MANIFEST="run-${RUN}-temporal-latest.manifest"
CHUNK_BYTES="${SNAPSHOT_CHUNK_BYTES:-1900000000}"   # < 2 GiB, with margin

# ...but every branch below says WHY it gave up. A best-effort path that
# reports success while doing nothing is the failure mode that cost this
# project four separate incidents (CLAUDE.md 6c rule 6).
if [ ! -f "$SRC" ]; then
  echo "snapshot: no head mirror at $SRC yet — nothing to save"; exit 0
fi
if [ -z "${GITHUB_TOKEN:-}" ]; then
  echo "::warning::snapshot: GITHUB_TOKEN empty — the head is NOT being backed up"
  exit 0
fi

API="https://api.github.com"
AUTH=(-H "Authorization: token ${GITHUB_TOKEN}" -H "Accept: application/vnd.github+json")

REL=$(curl -fsSL "${AUTH[@]}" "${API}/repos/${REPO}/releases/tags/${TAG}") || {
  echo "::warning::snapshot: cannot read release ${TAG}"; exit 0; }
REL_ID=$(printf '%s' "$REL" | sed -n 's/.*"id": *\([0-9]*\).*/\1/p' | head -1)
[ -n "$REL_ID" ] || { echo "::warning::snapshot: no release id"; exit 0; }

# Copy first: temporal.py rewrites the mirror atomically, but uploading the
# live path could still race a replace mid-read.
CP="/tmp/${ASSET}"
cp "$SRC" "$CP" || { echo "::warning::snapshot: copy failed"; exit 0; }
STEP=$(python -c "
import torch,sys
try: print(torch.load('$CP', map_location='cpu', weights_only=False).get('step','?'))
except Exception as e: print('?')" 2>/dev/null)

# The boxes have bash, curl and python. No jq, no node — every JSON read
# below goes through `python -c`, and nothing may depend on a CLI the boxes
# do not have (that exact mistake made one release-seed step report success
# while downloading nothing for three runs; CLAUDE.md §4.6).
PY=python; command -v python >/dev/null 2>&1 || PY=python3

# ---------------------------------------------------------------- listing --
# NEVER read the `assets` array embedded in the release object. Measured
# 2026-08-27: that array carried 572 entries where the paginated /assets
# endpoint returned 577 — it is TRUNCATED, and the duplicate-name grep that
# used to stand here therefore missed existing assets, leaving the upload to
# 422 on a name that was already taken. Page the real endpoint instead.
# Emits "id name" per line, one page at a time, until a page comes back empty.
list_assets_paged() {
  local page=1 body out
  while [ "$page" -le 500 ]; do
    body=$(curl -fsSL "${AUTH[@]}" \
      "${API}/repos/${REPO}/releases/${REL_ID}/assets?per_page=100&page=${page}") || {
      echo "::warning::snapshot: cannot list release assets (page ${page}) —" \
           "cannot tell what is already on the release; a stale asset may 422 the upload" >&2
      return 1; }
    out=$(printf '%s' "$body" | "$PY" -c 'import json,sys
try: a=json.load(sys.stdin)
except Exception: a=[]
[print(x.get("id",""), x.get("name","")) for x in a]') || {
      echo "::warning::snapshot: could not parse the asset page ${page} JSON" >&2
      return 1; }
    [ -n "$out" ] || break        # an empty page is the end of the list
    printf '%s\n' "$out"
    page=$((page+1))
  done
  return 0
}

# Delete the asset called <name>, if the release has one. Warn and continue on
# failure: a stale asset costs us this upload, not the run.
delete_asset_named() {
  local want="$1" id name found=""
  while read -r id name; do
    if [ "$name" = "$want" ]; then found="$id"; break; fi
  done < <(list_assets_paged)
  [ -n "$found" ] || return 0
  curl -fsSL -X DELETE "${AUTH[@]}" \
    "${API}/repos/${REPO}/releases/assets/${found}" >/dev/null \
    || echo "::warning::snapshot: could not delete the existing ${want}" \
            "(asset ${found}) — the upload that follows may 422 on a taken name"
}

# Delete every part asset whose name is NOT in the newline-separated keep-list
# on stdin. Two jobs, both about the glob in the manifest's reassembly hint:
# sweeping the whole chunked representation (empty keep-list, single-file
# path), and sweeping parts left over from a LONGER previous split. A head
# that shrinks from five parts to three leaves .partad/.partae behind, and
# `cat …part*` would then append two chunks of a stale checkpoint to a good
# one — a corrupt resume with no error anywhere.
delete_surplus_parts() {
  local keep id name
  keep=$(cat)
  while read -r id name; do
    case "$name" in
      "${ASSET}".part*)
        printf '%s\n' "$keep" | grep -qxF "$name" && continue
        curl -fsSL -X DELETE "${AUTH[@]}" \
          "${API}/repos/${REPO}/releases/assets/${id}" >/dev/null \
          || echo "::warning::snapshot: could not delete surplus ${name}" \
                  "(asset ${id}) — a reassembly by glob would now pick up a stale chunk" ;;
    esac
  done < <(list_assets_paged)
}

# `-T` STREAMS the body. `--data-binary "@f"` buffers the whole file in
# memory, and that is what stood here until 2026-08-17: at the 88M tier a
# head with optimiser state was ~1 GB and curl could hold it, but at 205-217M
# it is ~2.6 GB and every upload died with "curl: option --data-binary: out
# of memory". #358 trained all 200,000 steps and published NOTHING for the
# last fifteen hours, warning quietly every thirty minutes while the run
# stayed green.
#
# The exact same defect, in the same words, is already recorded in
# ml/CLAUDE.md and already fixed in the ORPHAN-RESCUE loop twenty lines into
# ml-train.yml -- which streams with `-T` and carries a comment explaining
# why. One of the two call sites was fixed and its sibling was not, which is
# the "read the failing line before fixing the third thing around it" lesson
# arriving from the other direction: the fix was found, applied once, and
# never grepped for.
upload_one() {
  local path="$1" name="$2"
  delete_asset_named "$name"      # replace, never accumulate
  curl -fsSL -X POST "${AUTH[@]}" \
    -H "Content-Type: application/octet-stream" -T "$path" \
    "https://uploads.github.com/repos/${REPO}/releases/${REL_ID}/assets?name=${name}" \
    >/dev/null
}

SIZE=$(wc -c < "$CP" | tr -d ' ')
HUMAN=$(du -h "$CP" | cut -f1)

if [ "$SIZE" -le "$CHUNK_BYTES" ]; then
  # ---------------------------------------------------------------- single --
  # Small enough for one asset: the historic name and shape, unchanged.
  # The manifest goes FIRST, before anything else moves. It is the marker
  # (§5.21) and an over-claiming marker is the dangerous state: a manifest
  # left behind by a chunked era, pointing at parts we are about to delete,
  # would tell a reader to reassemble files that no longer exist.
  delete_asset_named "$MANIFEST"
  if upload_one "$CP" "$ASSET"; then
    printf '' | delete_surplus_parts        # keep no parts at all
    echo "snapshot: ${ASSET} saved at step ${STEP} (${HUMAN}, single asset, 1 part)" \
         "— this run is now resumable from another box"
  else
    echo "::warning::snapshot: upload of ${ASSET} FAILED — the head is only on this box"
  fi
else
  # --------------------------------------------------------------- chunked --
  # Over the 2 GiB cap. Split, upload every part, and only then mark.
  rm -f "/tmp/${ASSET}".part*
  if ! split -b "$CHUNK_BYTES" -a 2 "$CP" "/tmp/${ASSET}.part"; then
    echo "::warning::snapshot: split of ${CP} failed (${SIZE} bytes) —" \
         "cannot chunk a head over the 2 GiB cap, so the head is only on this box"
    rm -f "$CP" "/tmp/${ASSET}".part*
    exit 0
  fi
  PARTS=("/tmp/${ASSET}".part*)   # glob order is lexicographic = split order
  if [ ! -e "${PARTS[0]}" ]; then
    echo "::warning::snapshot: split produced no parts — the head is only on this box"
    rm -f "$CP"; exit 0
  fi

  ALL_OK=1
  NAMES=""
  for p in "${PARTS[@]}"; do
    pn="${p##*/}"
    if upload_one "$p" "$pn"; then
      NAMES="${NAMES}${pn}
"
    else
      echo "::warning::snapshot: part ${pn} FAILED — the head is only on this box"
      ALL_OK=0
      break
    fi
  done

  if [ "$ALL_OK" = 1 ]; then
    # sha256 in python: sha256sum is not on every box image, python is.
    # Streamed, so a 2.6 GiB head does not become a 2.6 GiB allocation.
    SHA=$("$PY" - "$CP" <<'PYEOF' 2>/dev/null
import hashlib, sys
h = hashlib.sha256()
with open(sys.argv[1], 'rb') as f:
    for blk in iter(lambda: f.read(1 << 20), b''):
        h.update(blk)
print(h.hexdigest())
PYEOF
)
    if [ -z "$SHA" ]; then
      SHA="unavailable"
      echo "::warning::snapshot: could not compute sha256 (python hashlib failed) —" \
           "the manifest will name the parts but a reader cannot verify the join"
    fi
    MF="/tmp/${MANIFEST}"
    {
      echo "# ${MANIFEST} — chunked release mirror of run ${RUN}'s stage-2 head"
      echo "run: ${RUN}"
      echo "step: ${STEP}"
      echo "size_bytes: ${SIZE}"
      echo "sha256: ${SHA}"
      echo "chunk_bytes: ${CHUNK_BYTES}"
      echo "parts: ${#PARTS[@]}"
      printf '%s' "$NAMES" | while read -r n; do echo "part: ${n}"; done
      echo "reassemble: cat ${ASSET}.part* > temporal.pt"
    } > "$MF"

    # The single-file representation must not survive alongside the parts:
    # two answers to "where is run-N's head" is how a reader picks the stale
    # one. Same for parts from a longer previous split. Both go BEFORE the
    # manifest, so the marker is still the last thing written (§5.21).
    printf '%s' "$NAMES" | delete_surplus_parts
    delete_asset_named "$ASSET"
    if upload_one "$MF" "$MANIFEST"; then
      echo "snapshot: ${ASSET} saved at step ${STEP} (${HUMAN}, chunked," \
           "${#PARTS[@]} parts + manifest) — this run is now resumable from another box"
    else
      echo "::warning::snapshot: all ${#PARTS[@]} parts uploaded but the ${MANIFEST}" \
           "upload FAILED — the parts are durable but UNMARKED, so no reader will" \
           "trust them; the next snapshot re-uploads and re-marks"
    fi
    rm -f "$MF"
  fi
  rm -f "/tmp/${ASSET}".part*
fi
rm -f "$CP"

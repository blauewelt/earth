#!/usr/bin/env bash
# Publish a stage-2 head as WEIGHTS ONLY, so it fits in a release asset.
#
# WHY THIS EXISTS — measured 2026-08-17, and it is not the bug we thought.
#
# Wave 8 (#357-#364, the 205-217M tier) finished training and published no
# head at all. Two stacked faults, and only fixing the first is what let the
# second hide for a day:
#
#   1. scripts/snapshot_head.sh uploaded with `curl --data-binary "@f"`,
#      which buffers the whole body. At 88M a head with optimiser state is
#      ~1 GB and curl held it; at 205-217M it is ~2.5 GB and every upload
#      died "out of memory". Fixed in 2464ac8 by switching to `-T`.
#
#   2. `-T` streams correctly AND THE UPLOAD STILL FAILS, because a GitHub
#      release asset is capped around 2 GiB. The orphan-rescue loop in
#      ml-train.yml has always used `-T`, and its own logs say:
#
#         rescue upload rescued-orphan-latest-369.pt          -> HTTP 201
#         rescue upload rescued-orphan-temporal-latest-369.pt -> HTTP 422
#         rescue upload rescued-orphan-metrics-latest-369.jsonl -> HTTP 201
#
#      488 MB: 201. 2.49 GB: 422. Streaming was never the whole problem —
#      the file is simply too big to BE an asset.
#
# I asserted on 2026-08-17 that the rescue path "works" and that every
# wave-8 head was therefore recoverable through it. That was inference from
# `-T` being the right flag, not a measurement of the effect, and it was
# wrong — the same "assert the EFFECT, not the invocation" rule the repo
# already carries, failed at the level of my own fix.
#
# THE FIX. An EVALUATION needs weights, not optimiser moments:
# rollout_spatial.py rolls a head, it does not continue training one. So
# strip {args, model} out of the 2.5 GB checkpoint and publish that — about
# 870 MB for a 217M head, comfortably inside the cap. The full checkpoint
# stays on the box for anyone who genuinely wants to resume it.
#
#   window: headpub                      # publish orphan-temporal-latest.pt
#   window: headpub:<tag>                # name the asset explicitly
#   window: headpub:<tag>@temporal       # pin the source to temporal.pt
#   window: headpub:<tag>@orphan         # pin the source to orphan-temporal-latest.pt
#
# WHY @temporal EXISTS — measured 2026-08-17, runs #378/#381/#382. Orphan-first
# is right ONLY on a box whose run DIED mid-training (the orphan snapshot is
# the head). On a box whose run COMPLETED, temporal.pt is the head and
# orphan-temporal-latest.pt can be a STALE leftover from an earlier
# experiment: #381/#382 stripped Aug-15 leftovers (768x12 and 576x8 models)
# and published them under E-037 names; #378 published a stale xl144 s1 head
# as e035b. The printed step/d_model/layers/stencil/seed/znoise line below is
# how a reader catches this — CHECK IT against the arm before trusting the
# asset.
set -e
WINDOW="${1:-headpub}"
TAG="${WINDOW#headpub}"; TAG="${TAG#:}"
PICK="${TAG##*@}"; [ "$PICK" = "$TAG" ] && PICK=""; TAG="${TAG%@*}"
case "$PICK" in
  temporal) SRC=/opt/earth-cache/ckpt/temporal.pt ;;
  orphan)   SRC=/opt/earth-cache/ckpt/orphan-temporal-latest.pt ;;
  "")       SRC=/opt/earth-cache/ckpt/orphan-temporal-latest.pt
            [ -f "$SRC" ] || SRC=/opt/earth-cache/ckpt/temporal.pt ;;
  *) echo "::error::unknown headpub source '@${PICK}' (use @temporal or @orphan)"; exit 1 ;;
esac
[ -f "$SRC" ] || { echo "::error::no head on this box at $SRC"; exit 1; }
ASSET="head-weights-${TAG:-${GITHUB_RUN_NUMBER}}.pt"
OUT="/tmp/${ASSET}"

echo "source: $SRC ($(du -h "$SRC" | cut -f1))"
# STRIP, and print what the file actually holds — the step count and the
# architecture are how a later reader tells which arm this was, and they are
# the fields a dispatch would otherwise have to guess.
python - "$SRC" "$OUT" <<'PY'
import sys, torch
src, out = sys.argv[1], sys.argv[2]
d = torch.load(src, map_location="cpu", weights_only=False)
keep = {k: d[k] for k in ("args", "model") if k in d}
if "model" not in keep:
    raise SystemExit("refusing: no 'model' key in %s (keys: %s)" % (src, sorted(d)))
a = keep.get("args", {})
a = a if isinstance(a, dict) else vars(a)
print("step=%s d_model=%s layers=%s stencil=%s seed=%s znoise=%s" % (
    d.get("step", "?"), a.get("d_model", "?"), a.get("layers", "?"),
    a.get("stencil", "?"), a.get("seed", "?"), a.get("input_znoise", "?")))
print("params=%.1fM" % (sum(v.numel() for v in keep["model"].values()) / 1e6))
torch.save(keep, out)
PY
SZ=$(stat -c %s "$OUT")
echo "weights-only: $ASSET  $(du -h "$OUT" | cut -f1)  ($SZ bytes)"
# The cap is the whole reason this script exists, so refuse rather than
# discover it again in an HTTP code.
if [ "$SZ" -gt 2000000000 ]; then
  echo "::error::${ASSET} is ${SZ} bytes — still over the ~2 GiB release-asset cap. Split it."
  exit 1
fi

API=https://api.github.com
REL_ID=$(curl -fsSL -H "Authorization: Bearer $GITHUB_TOKEN" \
  "${API}/repos/${GITHUB_REPOSITORY}/releases/tags/model-checkpoints-v1" \
  | sed -n 's/.*"id": *\([0-9]*\).*/\1/p' | head -1)
[ -n "$REL_ID" ] || { echo "::error::no release id"; exit 1; }

UP=$(curl -s -o /tmp/up.json -w "%{http_code}" -X POST \
  -H "Authorization: Bearer $GITHUB_TOKEN" \
  -H "Content-Type: application/octet-stream" -T "$OUT" \
  "https://uploads.github.com/repos/${GITHUB_REPOSITORY}/releases/${REL_ID}/assets?name=${ASSET}")
echo "upload ${ASSET} -> HTTP ${UP}"
rm -f "$OUT"
# ASSERT THE EFFECT. A 422 here is what a whole day of "the head is safe"
# was built on; this step must not be able to report success without one.
case "$UP" in
  201) echo "published ${ASSET}";;
  *)   echo "::error::upload refused (HTTP ${UP}) — $(head -c 300 /tmp/up.json)"; exit 1;;
esac

#!/usr/bin/env bash
# DECODER RETRAIN MODE (E-019b1) — the body of the `dectrain:` window token.
#
# Lives here rather than inline in ml-train.yml because the Probes step's
# `run:` block has a 21,000-character ceiling GitHub enforces at DISPATCH
# time ("Exceeded max expression length", measured 2026-08-13): the inline
# version parsed fine, pushed fine, and then 422'd every dispatch of the
# whole workflow — the same all-or-nothing failure shape as the 26th input.
#
# Spec: dectrain:<hidden>x<layers>@<steps>@<pairs>[@seed<k>]
#   e.g. dectrain:1536x3@4000@6000000
#
# Trains a multi-output decoder against the FROZEN run-62 Z cache and
# publishes weights + audit JSON to model-checkpoints-v1. The box already
# holds everything this needs: the sha-verified tensor (seeded upstream)
# and the probes' own embed cache. A 4090 does the optimisation in about a
# minute; the sandbox's ~65-minute CPU train died to container restarts
# three times on 2026-08-13, which is why this mode exists.
#
# Fatal on failure, deliberately (bash -e from the caller): training IS
# this mode's job, and publish_decoder.py's assert-the-effect re-list is
# what says it worked. Needs GITHUB_TOKEN + TENSOR in env.
set -e
WINDOW="${1:?usage: dectrain_run.sh 'dectrain:<h>x<l>@<steps>@<pairs>[@seed<k>]'}"

SPEC="${WINDOW#dectrain:}"
HID="${SPEC%%x*}"; REST="${SPEC#*x}"
LAY="${REST%%@*}"; REST="${REST#*@}"
STEPS="${REST%%@*}"; REST="${REST#*@}"
PAIRS="${REST%%@*}"
DSEED=0
case "$SPEC" in *@seed*) DSEED="${SPEC##*@seed}";; esac
case "$HID$LAY$STEPS$PAIRS$DSEED" in
  ''|*[!0-9]*) echo "::error::bad dectrain spec '$SPEC'"; exit 1;;
esac
echo "dectrain: hidden=$HID layers=$LAY steps=$STEPS pairs=$PAIRS seed=$DSEED"
df -h / | tail -1

# The codec checkpoint PAIRS with the Z hash below: run-62 / f3_anchor41M,
# weight hash 6c52f0687b. A different codec needs a different Z and a
# different hash — change both or neither.
CKPT=ml/cache/f3_anchor41M__pixelmae.pt
[ -s "$CKPT" ] || curl -fsSL -o "$CKPT" \
  "https://github.com/${GITHUB_REPOSITORY}/releases/download/model-checkpoints-v1/f3_anchor41M__pixelmae.pt"

# Raw X memmap + small members, extracted ONCE from the box's sha-verified
# tensor; idempotent, with a disk guard sized from the zip entry (§5.18).
python -u scripts/dectrain_extract.py --tensor "$TENSOR"

# Z cache: the probes' own embed cache if this box has one
# (embed_cache_path convention: Z_<run>_<whash>_<dhash>.npy), else pull
# the published chunks — loudly, it is 5.2 GiB.
ZPATH="$(ls ml/cache/Z_*_6c52f0687b_adcbe700fb.npy 2>/dev/null | head -1 || true)"
if [ -z "$ZPATH" ]; then
  echo "::warning::no local Z cache — pulling 4 chunks from embed-cache-v1 (~5.2 GiB)"
  ZPATH=ml/cache/Z_dectrain_6c52f0687b_adcbe700fb.npy
  rm -f "$ZPATH"
  for c in aa ab ac ad; do
    curl -fsSL "https://github.com/${GITHUB_REPOSITORY}/releases/download/embed-cache-v1/Z_6c52f0687b_adcbe700fb.npy.$c" >> "$ZPATH"
  done
fi
echo "Z: $ZPATH ($(stat -c%s "$ZPATH") bytes)"

OUTTAG="dec${HID}x${LAY}s${DSEED}"
python -u ml/recon_decoder.py --x ml/cache/family3_X.npy \
  --npz-small ml/cache/f3_dec_small.npz --z "$ZPATH" --ckpt "$CKPT" \
  --hidden "$HID" --layers "$LAY" --steps "$STEPS" --batch 4096 \
  --pairs "$PAIRS" --seed "$DSEED" \
  --out "ml/runs/recon_decoder/${OUTTAG}.json"

# Publish weights + audit JSON; deletes-then-posts under deterministic
# names, streams with `curl -T`, asserts the EFFECT before exiting 0.
python -u scripts/publish_decoder.py --tag "$OUTTAG"
echo "dectrain ${OUTTAG}: published to model-checkpoints-v1"

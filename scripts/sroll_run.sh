#!/usr/bin/env bash
# E-022 spatial rollout eval — the body of the `sroll:` window token.
#
# Spec: sroll:<tag,tag,...>
#   e.g. sroll:e017_u1_s0,e017_u1_s1,e017_u1_s2,e022s9_u1_s0,e022s13_u1_s0
#
# Pulls each published stage-2 head and rolls it over ALL window ocean
# pixels with ml/rollout_spatial.py — the only evaluator that can feed a
# stencil head its neighbourhood inputs. The e017_u1_s0 head is the
# VALIDATION GATE (plan §6.5): rollout_spatial refuses to score anything
# unless that head reproduces #217's numbers within ±0.01, so every sroll
# dispatch should include it in the tag list. Writes rollout_spatial.json
# into the run's probe dir, where archive_probes.py picks it up into
# probes-<run>.json on ml-metrics.
#
# Lives in a script, not inline, because the Probes `run:` block has a
# 21,000-char dispatch-time expression ceiling (see dectrain_run.sh).
# Fatal on failure, deliberately: this eval IS this mode's job.
set -e
WINDOW="${1:?usage: sroll_run.sh 'sroll:<tag,tag,...>'}"

TAGS="${WINDOW#sroll:}"
[ -n "$TAGS" ] || { echo "::error::no head tags in '$WINDOW'"; exit 1; }
echo "sroll: tags=$TAGS"
df -h / | tail -1

CKPT=ml/cache/f3_anchor41M__pixelmae.pt
[ -s "$CKPT" ] || curl -fsSL -o "$CKPT" \
  "https://github.com/${GITHUB_REPOSITORY}/releases/download/model-checkpoints-v1/f3_anchor41M__pixelmae.pt"

# X memmap + small members from the box's sha-verified tensor (shared with
# dectrain/project; idempotent, disk guard sized from the zip entry).
python -u scripts/dectrain_extract.py --tensor "$TENSOR"

# Z cache: the probes' own if this box has one, else the published chunks.
ZPATH="$(ls ml/cache/Z_*_6c52f0687b_adcbe700fb.npy 2>/dev/null | head -1 || true)"
if [ -z "$ZPATH" ]; then
  echo "::warning::no local Z cache — pulling 4 chunks from embed-cache-v1 (~5.2 GiB)"
  ZPATH=ml/cache/Z_sroll_6c52f0687b_adcbe700fb.npy
  rm -f "$ZPATH"
  for c in aa ab ac ad; do
    curl -fsSL "https://github.com/${GITHUB_REPOSITORY}/releases/download/embed-cache-v1/Z_6c52f0687b_adcbe700fb.npy.$c" >> "$ZPATH"
  done
fi
echo "Z: $ZPATH ($(stat -c%s "$ZPATH") bytes)"

mkdir -p ml/runs/heads
HPATHS=""
for tag in ${TAGS//,/ }; do
  if curl -fsSL -o "ml/runs/heads/${tag}.pt" \
      "https://github.com/${GITHUB_REPOSITORY}/releases/download/model-checkpoints-v1/${tag}__temporal.pt"; then
    HPATHS="$HPATHS ml/runs/heads/${tag}.pt"
  else
    echo "::warning::head ${tag} not on the release — skipped"
  fi
done
[ -n "$HPATHS" ] || { echo "::error::no heads fetched — nothing to roll"; exit 1; }

OUT="ml/runs/actions/rollout_spatial.json"
python -u ml/rollout_spatial.py --x ml/cache/family3_X.npy \
  --npz-small ml/cache/f3_dec_small.npz --z "$ZPATH" --ckpt "$CKPT" \
  --heads $HPATHS --out "$OUT"

# Assert the EFFECT: the file exists, parses, the gate PASSED, and every
# fetched head carries a scored corridor block.
python - "$OUT" <<'PYEOF'
import json, os, sys
p = sys.argv[1]
d = json.load(open(p))
g = d.get("gate") or {}
assert g.get("pass") is True, f"gate did not pass: {g}"
assert d.get("heads"), "no heads scored"
for lab, e in d["heads"].items():
    cs = e.get("corridor", {}).get("chan_skill")
    assert cs, f"{lab}: no corridor skill rows"
    print(f"  {lab}: corridor AUC {e['corridor']['horizon_auc']:+.3f} "
          f"(window {e['window']['horizon_auc']:+.3f})")
print(f"rollout_spatial.json OK ({os.path.getsize(p):,} bytes)")
PYEOF
echo "sroll: wrote $OUT — archive_probes.py will publish it"

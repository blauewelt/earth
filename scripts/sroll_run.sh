#!/usr/bin/env bash
# E-022 / E-044 spatial rollout eval — the body of the `sroll:` window token.
#
# Spec: sroll:<tag,tag,...>[,ckpt:<asset|path>]
#   e.g. sroll:e017_u1_s0,e017_u1_s1,e022s9_u1_s0,e022s13_u1_s0
#        sroll:e044x144zn_u1_s0,ckpt:run-415.pt
#
# Pulls each published stage-2 head and rolls it over ALL window ocean
# pixels with ml/rollout_spatial.py — the only evaluator that can feed a
# stencil head its neighbourhood inputs. On the MONTHLY family-3 tensor the
# e017_u1_s0 head is the VALIDATION GATE (plan §6.5): rollout_spatial refuses
# to score anything unless that head reproduces #217's numbers within ±0.01,
# so every monthly sroll dispatch should include it in the tag list. Writes
# rollout_spatial.json into the run's probe dir, where archive_probes.py
# picks it up into probes-<run>.json on ml-metrics.
#
# NOTHING HERE IS FAMILY-3 ANY MORE (2026-08-19, E-044). Until today this
# script named family 3 four times in literals no input could override — the
# codec asset `f3_anchor41M__pixelmae.pt`, the Z glob
# `Z_*_6c52f0687b_adcbe700fb.npy` (that codec's weight hash and that tensor's
# sha256), a fixed four-chunk fallback download, and
# `--x ml/cache/family3_X.npy --npz-small ml/cache/f3_dec_small.npz` — so a
# `recipe:` naming the pentad tensor would have rolled family 3 with a
# family-3 codec and reported it as a pentad number. Every one of them is now
# DERIVED:
#   * the tensor comes from $TENSOR, which scripts/probes_run.sh already
#     resolves from ${RECIPE_TENSOR:-$IN_TENSOR};
#   * X is the tensor's sidecar `_X.npy` when it has one (family 5) and is
#     otherwise extracted from the npz, ONCE, keyed by the tensor's stem;
#   * `--npz-small` is the TENSOR ITSELF. Its small members are read lazily
#     out of the zip, they are byte-identical to what dectrain_extract.py
#     copies into f3_dec_small.npz, and — the reason for the change —
#     `bin_index` / `pentad_days` / `cadence` / `epoch` are NOT in that
#     script's SMALL list, so the extracted file cannot tell the roll what
#     cadence it is looking at;
#   * the Z cache is keyed by the two hashes the embed cache is actually
#     named for (ml/temporal.py `codec_weight_hash` + `data_fingerprint`),
#     computed here from the checkpoint and the tensor rather than pasted;
#   * the codec is named by a `ckpt:` token, defaulting to the published f3
#     anchor ONLY on the monthly family-3 tensor. On any other tensor there
#     is nothing to derive it from and the script REFUSES, because a wrong
#     codec is the #10/#11 failure: the shape check passes, the Z verify
#     passes against its own re-encode, and the roll is beautiful nonsense.
#
# Lives in a script, not inline, because the Probes `run:` block has a
# 21,000-char dispatch-time expression ceiling (see dectrain_run.sh).
# Fatal on failure, deliberately: this eval IS this mode's job.
set -e
WINDOW="${1:?usage: sroll_run.sh 'sroll:<tag,tag,...>[,ckpt:<asset|path>]'}"

SPEC="${WINDOW#sroll:}"
TAGS=""
CKPT_SPEC=""
for tok in ${SPEC//,/ }; do
  case "$tok" in
    ckpt:*) CKPT_SPEC="${tok#ckpt:}" ;;
    *:*)    echo "::error::unknown sroll token '$tok'"; exit 1 ;;
    "")     ;;
    *)      TAGS="$TAGS $tok" ;;
  esac
done
[ -n "$TAGS" ] || { echo "::error::no head tags in '$WINDOW'"; exit 1; }
echo "sroll: tags=$(echo $TAGS | tr ' ' ',')${CKPT_SPEC:+ ckpt=$CKPT_SPEC}"
df -h / | tail -1

TENSOR="${TENSOR:-ml/cache/family3_na025.npz}"
[ -s "$TENSOR" ] || { echo "::error::no tensor at $TENSOR"; exit 1; }
STEM="$(basename "$TENSOR" .npz)"
echo "tensor: $TENSOR ($(stat -c%s "$TENSOR") bytes)"

# ---- the codec -----------------------------------------------------------
if [ -n "$CKPT_SPEC" ]; then
  case "$CKPT_SPEC" in
    */*) CKPT="$CKPT_SPEC" ;;                      # a path on this box
    *)   CKPT="ml/cache/${CKPT_SPEC}"
         [ -s "$CKPT" ] || curl -fsSL -o "$CKPT" \
           "https://github.com/${GITHUB_REPOSITORY}/releases/download/model-checkpoints-v1/${CKPT_SPEC}" ;;
  esac
elif [ "${STEM#family3}" != "$STEM" ]; then
  CKPT=ml/cache/f3_anchor41M__pixelmae.pt
  [ -s "$CKPT" ] || curl -fsSL -o "$CKPT" \
    "https://github.com/${GITHUB_REPOSITORY}/releases/download/model-checkpoints-v1/f3_anchor41M__pixelmae.pt"
else
  echo "::error::tensor '$STEM' is not the monthly family-3 tensor and the"
  echo "::error::dispatch named no codec. The heads were trained on ONE"
  echo "::error::codec's embeddings and there is nothing in the tensor that"
  echo "::error::says which — pass it: sroll:<tags>,ckpt:<release asset or"
  echo "::error::path on this box>. Guessing here is the #10/#11 failure:"
  echo "::error::the shape check, the Z verify and the roll all succeed."
  exit 1
fi
[ -s "$CKPT" ] || { echo "::error::codec checkpoint $CKPT is missing/empty"; exit 1; }
echo "codec: $CKPT ($(stat -c%s "$CKPT") bytes)"

# ---- X: the sidecar if the tensor has one, else one extraction -----------
SIDECAR="ml/cache/${STEM}_X.npy"
if [ -s "$SIDECAR" ]; then
  XPATH="$SIDECAR"
  echo "X: sidecar $XPATH — no extraction needed"
else
  # family 3 keeps BOTH historical output names. dectrain_extract.py is
  # idempotent on (out-x size, out-small exists), so renaming either of them
  # would make every box with a warm cache re-extract 10.9 GiB.
  case "$STEM" in
    family3*) XPATH=ml/cache/family3_X.npy
              XSMALL=ml/cache/f3_dec_small.npz ;;
    *)        XPATH="ml/cache/${STEM}_X.npy"
              XSMALL="ml/cache/${STEM}_dec_small.npz" ;;
  esac
  python -u scripts/dectrain_extract.py --tensor "$TENSOR" \
    --out-x "$XPATH" --out-small "$XSMALL"
fi
[ -s "$XPATH" ] || { echo "::error::no X memmap at $XPATH"; exit 1; }

# ---- Z: keyed by (codec weight hash, tensor sha256), computed not pasted --
# Same two functions the trainer and embed_cache_sync.py name the cache with,
# imported rather than reimplemented — two copies of a hash rule are two
# chances to disagree, and the disagreement is silent (ml/temporal.py's own
# docstring for codec_weight_hash).
cat > /tmp/sroll_zkey.py <<'PY_ZKEY'
import sys, torch
sys.path.insert(0, "ml")
from temporal import codec_weight_hash, data_fingerprint
ck = torch.load(sys.argv[1], map_location="cpu", weights_only=False)
print(codec_weight_hash(ck), data_fingerprint(sys.argv[2]))
PY_ZKEY
HASHES="$(python /tmp/sroll_zkey.py "$CKPT" "$TENSOR" | tail -1)"
WHASH="${HASHES%% *}"
DHASH="${HASHES##* }"
[ -n "$WHASH" ] && [ -n "$DHASH" ] && [ "$WHASH" != "$DHASH" ] \
  || { echo "::error::could not key the embed cache (got '$HASHES')"; exit 1; }
echo "embed cache key: codec $WHASH · tensor $DHASH"

ZPATH="$(ls ml/cache/Z_*_${WHASH}_${DHASH}.npy 2>/dev/null | head -1 || true)"
if [ -z "$ZPATH" ]; then
  echo "::warning::no local Z for ${WHASH}/${DHASH} — pulling chunks from embed-cache-v1"
  ZPATH="ml/cache/Z_sroll_${WHASH}_${DHASH}.npy"
  rm -f "$ZPATH"
  N=0
  # publish splits at 1.5 GiB, so the chunk count scales with the tensor (4
  # at monthly family 3, ~11 at pentad). Stop at the first missing suffix
  # rather than at a hard-coded four.
  for a in a b c d; do
    for b in a b c d e f g h i j k l m n o p q r s t u v w x y z; do
      curl -fsSL \
        "https://github.com/${GITHUB_REPOSITORY}/releases/download/embed-cache-v1/Z_${WHASH}_${DHASH}.npy.${a}${b}" \
        >> "$ZPATH" || break 2
      N=$((N + 1))
    done
  done
  [ "$N" -gt 0 ] || { echo "::error::no embed cache published for ${WHASH}/${DHASH} — run the embed pass first"; exit 1; }
  echo "pulled $N chunk(s)"
fi
# ASSERT THE EFFECT: a Z whose file length disagrees with its own .npy header
# maps cleanly and returns real numbers belonging to the wrong rows.
python - "$ZPATH" <<'PYEOF'
import sys
sys.path.insert(0, "ml")
from embed_cache_sync import verify
ok, why = verify(sys.argv[1])
print(f"Z: {sys.argv[1]} — {why}")
if not ok:
    raise SystemExit(f"::error::embed cache rejected: {why}")
PYEOF

mkdir -p ml/runs/heads
HPATHS=""
# Two asset-name conventions live on the release. The original one is
# "<tag>__temporal.pt" — a full training checkpoint (weights + Adam moments
# + args), written by snapshot_head.sh. From 2026-08-16 a head that big no
# longer fits: a 205-217M head with moments is 2.49-2.61 GB and GitHub
# rejects any release asset over ~2 GiB with HTTP 422. Those heads are
# published weights-only by scripts/publish_head_weights.sh under their own
# name, "head-weights-<arm>.pt" (~845-869 MB). rollout_spatial.py reads both
# shapes. So: try the __temporal.pt name first (every pre-wave-8 head, and
# the e017_u1_s0 gate, still lives there), fall back to the tag verbatim.
for tag in $TAGS; do
  if curl -fsSL -o "ml/runs/heads/${tag}.pt" \
      "https://github.com/${GITHUB_REPOSITORY}/releases/download/model-checkpoints-v1/${tag}__temporal.pt" \
   || curl -fsSL -o "ml/runs/heads/${tag}.pt" \
      "https://github.com/${GITHUB_REPOSITORY}/releases/download/model-checkpoints-v1/${tag}.pt"; then
    HPATHS="$HPATHS ml/runs/heads/${tag}.pt"
  else
    echo "::warning::head ${tag} not on the release — skipped"
  fi
done
[ -n "$HPATHS" ] || { echo "::error::no heads fetched — nothing to roll"; exit 1; }

OUT="ml/runs/actions/rollout_spatial.json"
# BACKGROUNDED behind a publisher loop, exactly as the training step does.
# This eval is hours long and, until 2026-08-14, published nothing while it
# ran: Actions will not serve a running job's log, so "how far along is it?"
# had no answer from outside for the whole run. rollout_spatial.py appends
# progress records (head i/N, phase, steps done, ETA) to metrics.jsonl; this
# loop pushes them to ml-live-<n> every ~2.5 minutes, which the status page
# and any curl can read. Best-effort at the CALLER (`|| true`), because a
# telemetry push must never take down the eval it is reporting on.
rm -f ml/runs/actions/metrics.jsonl
python -u ml/rollout_spatial.py --x "$XPATH" \
  --npz-small "$TENSOR" --z "$ZPATH" --ckpt "$CKPT" \
  --heads $HPATHS --out "$OUT" &
EVAL_PID=$!
TICK=0
while kill -0 $EVAL_PID 2>/dev/null; do
  sleep 30
  TICK=$((TICK + 1))
  if [ $((TICK % 5)) -eq 0 ]; then
    bash scripts/publish_live_metrics.sh "ml-live-${GITHUB_RUN_NUMBER}" || true
  fi
done
wait $EVAL_PID || { echo "::error::rollout_spatial.py failed"; exit 1; }
bash scripts/publish_live_metrics.sh "ml-live-${GITHUB_RUN_NUMBER}" || true

# Assert the EFFECT: the file exists, parses, every fetched head carries a
# scored corridor block, and the gate is either PASSED or explicitly
# uncertified with a recorded reason — never silently absent.
python - "$OUT" <<'PYEOF'
import json, os, sys
p = sys.argv[1]
d = json.load(open(p))
g = d.get("gate") or {}
cad = d.get("cadence")            # written only when a step is not a month
if cad is None:
    assert g.get("pass") is True, f"gate did not pass: {g}"
    print("gate: PASSED (monthly axis, #217 reference)")
else:
    # A monthly reference cannot certify this roll and rollout_spatial.py
    # refuses to pretend otherwise. What it MUST do is say so, in the
    # artefact, so the harvest never reads these numbers as certified.
    assert g.get("skipped") is True and g.get("certified") is False, g
    assert g.get("reason"), "gate skipped with no recorded reason"
    print(f"gate: NOT CERTIFIED at {cad['name']} cadence — {g['reason'][:200]}")
    print(f"cadence: {cad['name']} · {cad['step_days']} d/step · "
          f"horizon {cad['horizon_steps']} steps = {cad['horizon_span_days']} d")
assert d.get("heads"), "no heads scored"
for lab, e in d["heads"].items():
    cs = e.get("corridor", {}).get("chan_skill")
    assert cs, f"{lab}: no corridor skill rows"
    print(f"  {lab}: corridor AUC {e['corridor']['horizon_auc']:+.3f} "
          f"(window {e['window']['horizon_auc']:+.3f})")
print(f"rollout_spatial.json OK ({os.path.getsize(p):,} bytes)")
PYEOF
echo "sroll: wrote $OUT — archive_probes.py will publish it"

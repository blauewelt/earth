#!/usr/bin/env bash
# E-021 "the 20-year fan" — the body of the `project:` window token.
#
# Spec: project:<months>@<members>@<tag,tag,...>
#   e.g. project:240@12@e017_u1_s0,e017_u1_s1,e017_u1_s2
#
# Rolls each published stage-2 head 240 months past the record AND from a
# 2004-12 context (where RAPID truth exists, so the fan's width can be
# checked before the future fan is believed), with two ensemble families
# scaled by each head's own measured one-step residual. Writes
# project_amoc.json into the run's probe dir, where archive_probes.py
# picks it up into probes-<run>.json on ml-metrics.
#
# Lives in a script, not inline, because the Probes `run:` block has a
# 21,000-char dispatch-time expression ceiling (see dectrain_run.sh).
# Fatal on failure, deliberately: projecting IS this mode's job.
set -e
WINDOW="${1:?usage: project_run.sh 'project:<months>@<members>@<tags>'}"

SPEC="${WINDOW#project:}"
MONTHS="${SPEC%%@*}"; REST="${SPEC#*@}"
MEMBERS="${REST%%@*}"; TAGS="${REST#*@}"
case "$MONTHS$MEMBERS" in
  ''|*[!0-9]*) echo "::error::bad project spec '$SPEC'"; exit 1;;
esac
[ -n "$TAGS" ] || { echo "::error::no head tags in '$SPEC'"; exit 1; }
echo "project: months=$MONTHS members=$MEMBERS tags=$TAGS"
df -h / | tail -1

CKPT=ml/cache/f3_anchor41M__pixelmae.pt
[ -s "$CKPT" ] || curl -fsSL -o "$CKPT" \
  "https://github.com/${GITHUB_REPOSITORY}/releases/download/model-checkpoints-v1/f3_anchor41M__pixelmae.pt"

# X memmap + small members from the box's sha-verified tensor (shared with
# dectrain; idempotent, disk guard sized from the zip entry).
python -u scripts/dectrain_extract.py --tensor "$TENSOR"

# Z cache: the probes' own if this box has one, else the published chunks.
ZPATH="$(ls ml/cache/Z_*_6c52f0687b_adcbe700fb.npy 2>/dev/null | head -1 || true)"
if [ -z "$ZPATH" ]; then
  echo "::warning::no local Z cache — pulling 4 chunks from embed-cache-v1 (~5.2 GiB)"
  ZPATH=ml/cache/Z_project_6c52f0687b_adcbe700fb.npy
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
[ -n "$HPATHS" ] || { echo "::error::no heads fetched — nothing to project"; exit 1; }

OUT="ml/runs/actions/project_amoc.json"
python -u ml/project_amoc.py --x ml/cache/family3_X.npy \
  --npz-small ml/cache/f3_dec_small.npz --z "$ZPATH" --ckpt "$CKPT" \
  --heads $HPATHS --months "$MONTHS" --members "$MEMBERS" \
  --starts "future,2004-12" --out "$OUT"

# Assert the EFFECT: the file exists, parses, and carries a pooled fan.
python - "$OUT" <<'PYEOF'
import json, os, sys
p = sys.argv[1]
d = json.load(open(p))
assert d.get("pooled"), "no pooled fan in the output"
for start, fams in d["pooled"].items():
    for fam, v in fams.items():
        n = len(v["q"]["p50"])
        print(f"  {start}/{fam}: {v['n_members']} members x {n} months")
        assert n > 0 and v["n_members"] > 0
print(f"project_amoc.json OK ({os.path.getsize(p):,} bytes)")
PYEOF
echo "project: wrote $OUT — archive_probes.py will publish it"

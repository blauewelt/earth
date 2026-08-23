#!/bin/bash
# The startup script for the FIRST TPU smoke run (ml/plans/TPU_ACCESS.md §6,
# ml/plans/JAX_PORT.md §1b). Launched as a Cloud TPU node's
# metadata `startup-script`:
#
#   sed 's|__BUCKET__|my-staging-bucket|' ml/jaxport/tpu_smoke.sh > /tmp/s.sh
#   python3 scripts/tpu_box.py create smoke-1 --spot --startup-file /tmp/s.sh
#
# WHAT THIS MEASURES, AND WHAT IT DELIBERATELY DOES NOT.
#
# TPU_ACCESS.md §6 names one specific thing that could make the whole TPU bet
# disappointing, and it is not the model: the per-batch host gather
# (`LazyPixels`/`gather_px`) runs on the TPU VM's HOST CPU, and if it cannot
# feed the accelerator the chips idle at ~$1.20/chip-hour — $9.60/h for the
# v5e-8. So this run answers exactly one question: **how fast does the host
# gather + encode path go, and how busy is the host CPU while it does?** It
# times `ml/jaxport`'s `embed_everything_jax` over a BOUNDED number of
# batches, times the host gather ALONE over the same batches, samples
# /proc/stat around both, and writes the numbers to the staging bucket.
#
# STAGE-1 TRAINER WORK IS OUT OF SCOPE. No training, no probe, no skill
# number. There is no anomaly transform either, which means **no number this
# script produces is a probe number** — the tensor is raw, the arithmetic is
# the encoder's but the inputs are not what any published result was scored
# on. It is a throughput measurement wearing no other clothes.
#
# ONE BIAS, STATED RATHER THAN HIDDEN. The tensor is decompressed into RAM
# (the v5e-8 host has ample), not memmapped from disk the way a real trainer
# would read it. That makes this an UPPER BOUND on gather throughput. The
# useful direction is one-sided and it is the direction that matters: if the
# gather starves the TPU from RAM, it certainly starves it from a memmap.
#
# Every step echoes WHAT IT MEASURED, not what it attempted (ml/CLAUDE.md
# §0.2 / §4.7). `set -euo pipefail` throughout and no `2>/dev/null` anywhere:
# a step that reports success is not evidence it did anything, and stderr is
# the line that explains the branch.
set -euo pipefail

# --------------------------------------------------------------------------
# knobs. BUCKET is substituted at launch; everything else is a real, checked-in
# release asset name (see the block comment on each).
# --------------------------------------------------------------------------
BUCKET="__BUCKET__"                 # substituted at launch — the §4 staging bucket
REPO="blauewelt/earth"
WORK="/opt/earth-smoke"

# The tensor. These are the EXACT names the fleet seeds from today: the
# `data-cache-v1` release carries the monthly family-3 tensor split in two
# parts named after the first ten hex of its sha256, and
# .github/workflows/ml-train.yml pins that sha so every box holds the same
# bytes (the "box effect" — a differently-built tensor moves the head k-fold
# by 0.041, ml/CLAUDE.md §7). We verify the same pin here for the same reason:
# a smoke run on different bytes is not comparable to anything.
TENSOR_SHA="adcbe700fb6e160b1c84d7acd8dc0333b34d07829db9abd62c721aab278b4848"
TENSOR_PARTS="aa ab"
TENSOR_ASSET_PREFIX="family3_na025_${TENSOR_SHA:0:10}.npz"

# The codec. `f3_anchor41M__pixelmae.pt` is the asset ml-train.yml falls back
# to for every run (the "${TAG}__pixelmae.pt" / "f3_anchor41M__pixelmae.pt"
# pair in the checkpoint-seed step), and it is the 40.7M codec gates G1/G2′/G3
# were all scored on (JAX_PORT.md §5).
CODEC_ASSET="f3_anchor41M__pixelmae.pt"

# The bound. 4 months x (P/8192) batches is a few minutes of work, which is
# all a throughput question needs — and a bound is what keeps a $9.60/h node
# from becoming a full 95-minute embedding nobody asked for.
SMOKE_MONTHS="${SMOKE_MONTHS:-4}"
SMOKE_BATCH="${SMOKE_BATCH:-8192}"
SMOKE_PIXELS="${SMOKE_PIXELS:-32768}"   # 4 batches per month at the default

RESULT="/tmp/tpu_smoke_result.json"
TRAIN_RESULT="/tmp/tpu_smoke_train.json"
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"

# --------------------------------------------------------------------------
# 0 · self-reap. The node deletes ITSELF — on completion, on any failure
# (trap EXIT under `set -e`), and unconditionally at 55 minutes via a
# disowned watchdog. This exists because the first smoke attempt (2026-08-22)
# proved the failure mode is real: the session driving the local reaper died,
# and a READY-but-stuck node billed for 8 hours with nobody watching. A
# watcher on somebody's laptop is a hope; a deletion armed on the node itself
# is a guarantee. Deletion needs no key on the box — the attached service
# account's token is one metadata call away, same as the publish step.
# __NODE__ and __TPUZONE__ are substituted at launch like __BUCKET__.
NODE="__NODE__"
TPUZONE="__TPUZONE__"
# Everything this script prints goes to a local log, and the EXIT trap ships
# that log to the bucket BEFORE the node deletes itself — added after the
# first self-reaping attempt died in seconds and left nothing to read. The
# serial console is not reachable over the v2 API; this log is the only
# black box recorder the node has.
exec >>/tmp/smoke.log 2>&1
upload_log() {
  T="$(curl -sf -H 'Metadata-Flavor: Google' \
      'http://metadata.google.internal/computeMetadata/v1/instance/service-accounts/default/token' \
      | python3 -c 'import json,sys; print(json.load(sys.stdin)["access_token"])')" || return 0
  curl -sf -X POST -H "Authorization: Bearer ${T}" -H 'Content-Type: text/plain' \
    --data-binary @/tmp/smoke.log \
    "https://storage.googleapis.com/upload/storage/v1/b/${BUCKET}/o?uploadType=media&name=tpu_smoke_logs/${STAMP}.txt" \
    || true
}
self_delete() {
  # RETRIED UNTIL THE NODE IS REALLY GONE: a DELETE issued while the CREATE
  # operation is still settling is silently dropped (measured 2026-08-23 —
  # the fast-failing startup's delete raced its own create, the node came up
  # READY afterwards, and only the 55-min watchdog would have caught it). Six
  # tries, 30 s apart, asserting the 404 rather than trusting the DELETE.
  P="$(curl -sf -H 'Metadata-Flavor: Google' \
      'http://metadata.google.internal/computeMetadata/v1/project/project-id')"
  for _i in 1 2 3 4 5 6; do
    T="$(curl -sf -H 'Metadata-Flavor: Google' \
        'http://metadata.google.internal/computeMetadata/v1/instance/service-accounts/default/token' \
        | python3 -c 'import json,sys; print(json.load(sys.stdin)["access_token"])')"
    curl -sf -X DELETE -H "Authorization: Bearer ${T}" \
      "https://tpu.googleapis.com/v2/projects/${P}/locations/${TPUZONE}/nodes/${NODE}" || true
    sleep 30
    CODE="$(curl -s -o /dev/null -w '%{http_code}' -H "Authorization: Bearer ${T}" \
      "https://tpu.googleapis.com/v2/projects/${P}/locations/${TPUZONE}/nodes/${NODE}")"
    if [ "${CODE}" = "404" ]; then echo "self-delete confirmed (404)"; return 0; fi
    echo "node still answers ${CODE} after delete attempt ${_i} — retrying"
  done
}
trap 'code=$?; echo "exit ${code} — shipping log, then node self-deletes"; upload_log; self_delete' EXIT
( sleep 3300; echo "watchdog: 55 min — force self-delete"; self_delete ) &
disown

echo "=== tpu_smoke ${STAMP} · bucket ${BUCKET} · repo ${REPO} ==="

# --------------------------------------------------------------------------
# 0 · what this host actually is
# --------------------------------------------------------------------------
echo "--- host ---"
mkdir -p "${WORK}"
echo "measured: $(nproc) CPUs, $(awk '/MemTotal/{printf "%.0f GB RAM", $2/1048576}' /proc/meminfo)"
AVAIL_GB="$(df -BG --output=avail "${WORK}" | tail -1 | tr -dc '0-9')"
echo "measured: ${AVAIL_GB} GB free on ${WORK}"
if [ "${AVAIL_GB}" -lt 12 ]; then
  echo "REFUSING: ${AVAIL_GB} GB free on ${WORK}; the tensor archive alone is" \
       "3.0 GB and the codec ~0.5 GB. Create the node with a larger boot disk."
  exit 1
fi

# --------------------------------------------------------------------------
# 1 · dependencies
# --------------------------------------------------------------------------
# A venv, not the system interpreter: recent Debian images mark the system
# python externally-managed (PEP 668) and `pip install` there fails with a
# message that reads like a network problem. The venv also means the exact
# wheel set is visible in one directory if anything has to be diagnosed.
echo "--- deps ---"
# The TPU VM image (Ubuntu 22.04, v2-alpha-tpuv5-lite) ships WITHOUT
# python3-venv — measured 2026-08-23: `python3 -m venv` dies in seconds with
# "ensurepip is not available", which killed the first two smoke nodes.
# Retried because first-boot apt can hold locks under cloud-init.
export DEBIAN_FRONTEND=noninteractive
for i in 1 2 3 4 5; do
  if apt-get update -qq && apt-get install -y -qq python3-venv; then
    echo "measured: python3-venv installed (attempt ${i})"; break
  fi
  if [ "${i}" = 5 ]; then echo "apt failed 5x — giving up"; exit 1; fi
  echo "apt attempt ${i} failed (boot lock?) — retrying in 20 s"; sleep 20
done
python3 -m venv "${WORK}/venv"
PY="${WORK}/venv/bin/python"
"${PY}" -m pip install --upgrade pip
# jax[tpu] is what reaches the chips; the -f index is where libtpu lives.
"${PY}" -m pip install 'jax[tpu]' \
    -f https://storage.googleapis.com/jax-releases/libtpu_releases.html
# flax is jaxport's module system (NNX — JAX_PORT.md §6).
"${PY}" -m pip install flax numpy optax
# CPU torch, and ONLY for the checkpoint converter: the published codec is a
# torch `.pt` dict and `jaxport.convert` is the single place that reads one
# (JAX_PORT.md §3.1). The CUDA wheels are ~2.5 GB of nothing on a TPU host, so
# this pins the CPU index explicitly rather than letting the default resolve.
"${PY}" -m pip install --index-url https://download.pytorch.org/whl/cpu torch
echo "measured: jax $("${PY}" -c 'import jax; print(jax.__version__)')" \
     "· flax $("${PY}" -c 'import flax; print(flax.__version__)')" \
     "· torch $("${PY}" -c 'import torch; print(torch.__version__)')" \
     "· numpy $("${PY}" -c 'import numpy; print(numpy.__version__)')"
echo "measured: jax.devices() -> $("${PY}" -c 'import jax; print(jax.devices())')"

# --------------------------------------------------------------------------
# 2 · the code
# --------------------------------------------------------------------------
echo "--- clone ---"
git clone --depth 1 "https://github.com/${REPO}.git" "${WORK}/earth"
echo "measured: cloned $(git -C "${WORK}/earth" rev-parse --short HEAD)" \
     "($(git -C "${WORK}/earth" log -1 --format=%cI))"

# --------------------------------------------------------------------------
# 3 · the data, from the repo's public releases
# --------------------------------------------------------------------------
# curl, not `gh`: the release is public so no auth is needed, and this is the
# path the Vast fleet has seeded from since #47 (ml/CLAUDE.md §7 — every
# `gh release download` on a box returned 127 with its stderr swallowed and
# the step reported success having downloaded nothing).
echo "--- data ---"
mkdir -p "${WORK}/cache"
TF="${WORK}/cache/family3_na025.npz"
if [ -f "${TF}" ] && [ "$(sha256sum "${TF}" | cut -d' ' -f1)" = "${TENSOR_SHA}" ]; then
  echo "measured: tensor already present, sha matches ${TENSOR_SHA:0:10}"
else
  for sfx in ${TENSOR_PARTS}; do
    echo "fetching ${TENSOR_ASSET_PREFIX}.${sfx} …"
    curl -fsSL --max-time 1800 --retry 3 --retry-delay 10 \
      -o "${TF}.${sfx}.part" \
      "https://github.com/${REPO}/releases/download/data-cache-v1/${TENSOR_ASSET_PREFIX}.${sfx}"
  done
  # shellcheck disable=SC2086
  cat $(for s in ${TENSOR_PARTS}; do printf '%s ' "${TF}.${s}.part"; done) > "${TF}.new"
  rm -f "${TF}".*.part
  GOT="$(sha256sum "${TF}.new" | cut -d' ' -f1)"
  if [ "${GOT}" != "${TENSOR_SHA}" ]; then
    echo "REFUSING: assembled tensor sha ${GOT} != pin ${TENSOR_SHA}."
    echo "A smoke run on different bytes is not comparable to the fleet's" \
         "numbers (the box effect, ml/CLAUDE.md §7). Not proceeding."
    rm -f "${TF}.new"
    exit 1
  fi
  mv "${TF}.new" "${TF}"
  echo "measured: tensor $(du -h "${TF}" | cut -f1), sha verified ${TENSOR_SHA:0:10}"
fi

CK="${WORK}/cache/${CODEC_ASSET}"
if [ -f "${CK}" ]; then
  echo "measured: codec already present, $(du -h "${CK}" | cut -f1)"
else
  echo "fetching ${CODEC_ASSET} …"
  curl -fsSL --max-time 1800 --retry 3 --retry-delay 10 -o "${CK}.part" \
    "https://github.com/${REPO}/releases/download/model-checkpoints-v1/${CODEC_ASSET}"
  mv "${CK}.part" "${CK}"
  echo "measured: codec $(du -h "${CK}" | cut -f1)" \
       "sha256 $(sha256sum "${CK}" | cut -c1-10)"
fi

# --------------------------------------------------------------------------
# 4 · the measurement
# --------------------------------------------------------------------------
echo "--- embed timing ---"
BUCKET="${BUCKET}" REPO="${REPO}" WORK="${WORK}" TF="${TF}" CK="${CK}" \
TRAIN_RESULT="${TRAIN_RESULT}" \
STAMP="${STAMP}" RESULT="${RESULT}" \
SMOKE_MONTHS="${SMOKE_MONTHS}" SMOKE_BATCH="${SMOKE_BATCH}" \
SMOKE_PIXELS="${SMOKE_PIXELS}" \
"${PY}" - <<'PY'
import json, math, os, subprocess, sys, time

sys.path.insert(0, os.path.join(os.environ["WORK"], "earth", "ml"))

import numpy as np
import jax
import torch

# COMPAT, before any jaxport import: `nnx.data` exists only in flax >= 0.11,
# which needs Python >= 3.11 — more than this image's 3.10 can install. On
# flax <= 0.10 a plain list auto-registers (the behaviour nnx.data makes
# explicit), so the identity is the correct old-regime spelling. Verified
# 2026-08-23: the full jaxport parity suite (17 checks) passes under exactly
# this identity on old flax, max|delta| <= 7e-7 vs torch. The durable fix is
# the same getattr shim in ml/jaxport/models.py; this line covers a node
# whose clone predates it.
from flax import nnx
if not hasattr(nnx, "data"):
    nnx.data = lambda x: x
    print("compat: flax", __import__("flax").__version__,
          "has no nnx.data — identity shim installed (parity-verified)",
          flush=True)

from model import LazyPixels
from jaxport.convert import codec_from_ckpt_jax
from jaxport.embed import embed_everything_jax, gather_px_np

MONTHS = int(os.environ["SMOKE_MONTHS"])
BATCH = int(os.environ["SMOKE_BATCH"])
PIXELS = int(os.environ["SMOKE_PIXELS"])


def cpu_sample():
    """(busy_jiffies, total_jiffies) from /proc/stat's aggregate line.

    Sampled around the loop rather than by `top -bn1`, which reports an
    instant and would miss a loop that alternates a saturated gather with an
    idle wait — precisely the shape a starving host would have.
    """
    with open("/proc/stat") as fh:
        f = [float(v) for v in fh.readline().split()[1:]]
    idle = f[3] + (f[4] if len(f) > 4 else 0.0)     # idle + iowait
    return sum(f) - idle, sum(f)


def util_between(a, b):
    dbusy, dtot = b[0] - a[0], b[1] - a[1]
    return round(dbusy / dtot, 4) if dtot > 0 else None


def pct(v):
    """Format a fraction, or say it is missing. Never invent a 0.0 for a
    reading that did not happen — a fabricated 0% host utilisation is exactly
    the number that would end this investigation with the wrong answer."""
    return "UNAVAILABLE (no /proc/stat delta)" if v is None else f"{v:.1%}"


t0 = time.time()
z = np.load(os.environ["TF"], allow_pickle=True)
# The TAIL of the tensor, not the head: channels that start later than the
# tensor leave the early months all-NaN (the month-0 empty-mask trap,
# ml/CLAUDE.md §7) — measured here 2026-08-23 as "0 ocean pixels" over
# months[:4]. The last months have every channel that will ever report.
X = z["X"][-MONTHS:]
lats, lons, months = z["lats"], z["lons"], [str(m) for m in z["months"]]
decompress_s = time.time() - t0
print(f"measured: decompressed {MONTHS} months of the tensor in "
      f"{decompress_s:.1f} s -> {X.shape} {X.dtype}, "
      f"{X.nbytes / 1e9:.2f} GB resident", flush=True)

# Ocean pixels over the bounded slice, exactly the predicate probe_kfold and
# score_section_probe use (`isfinite` on channel 0).
ocean = np.isfinite(X[..., 0]).any(0)
ys, xs = np.where(ocean)
n_ocean = len(ys)
if n_ocean == 0:
    raise SystemExit("REFUSING: 0 ocean pixels in the sliced months — "
                     "the slice is wrong, not the ocean")
ys, xs = ys[:PIXELS], xs[:PIXELS]
P = len(ys)
batches = MONTHS * math.ceil(P / BATCH)
print(f"measured: {n_ocean} ocean pixels in the window; timing {P} of them "
      f"x {MONTHS} months = {batches} batches of {BATCH}", flush=True)

moy = np.array([int(m[5:7]) - 1 for m in months[-MONTHS:]])
ctx_all = np.stack([np.sin(2 * np.pi * moy / 12),
                    np.cos(2 * np.pi * moy / 12)], 1)

ck = torch.load(os.environ["CK"], map_location="cpu", weights_only=False)
codec = codec_from_ckpt_jax(ck, X.shape[-1])
patch = ck["args"].get("patch", 1)
print(f"measured: codec {ck.get('tag')} d_z {ck['d_z']} patch {patch} "
      f"d_model {ck['args'].get('d_model')} over {X.shape[-1]} channels",
      flush=True)

# LazyPixels: nan_to_num / isfinite AFTER the per-batch index. X must NOT be
# pre-filled — OBS would be all-True and every land cell would enter the
# encoder as an observed 0.0 with no NaN anywhere to notice.
Xt, OBS = LazyPixels(X), LazyPixels(X, obs=True)

# --- warm-up: one month, so the timed run measures the LOOP and not XLA's
# first compile. Reported separately rather than hidden, because "the first
# batch took 40 s" is itself a fact about running this on a TPU.
t = time.time()
# ONE month of tensor to go with the one month of ctx: the month loop is
# derived from the TENSOR (T = X.shape[0]), so slicing only ctx_all walks
# month 2 into a length-1 context — measured 2026-08-23, IndexError at t=1.
embed_everything_jax(codec, LazyPixels(X[:1]), LazyPixels(X[:1], obs=True),
                     ctx_all[:1], lats, lons,
                     ys[:BATCH], xs[:BATCH], codec.d_z, batch=BATCH)
warm_s = time.time() - t
print(f"measured: warm-up (1 batch, includes XLA compile) {warm_s:.1f} s",
      flush=True)

# --- the timed embedding run ------------------------------------------------
marks = []
c0 = cpu_sample()
t = time.time()
embed_everything_jax(codec, Xt, OBS, ctx_all, lats, lons, ys, xs, codec.d_z,
                     batch=BATCH,
                     progress=lambda d: marks.append((d["month"],
                                                      round(time.time() - t, 3))))
embed_s = time.time() - t
c1 = cpu_sample()
embed_util = util_between(c0, c1)
per_month = [round(b - a, 3) for a, b in zip([0.0] + [m[1] for m in marks],
                                             [m[1] for m in marks])]
steps_per_s = batches / embed_s
print(f"measured: embed {batches} batches in {embed_s:.1f} s = "
      f"{steps_per_s:.2f} batches/s ({steps_per_s * BATCH:,.0f} pixel-encodes/s)",
      flush=True)
print(f"measured: host CPU utilisation across the embed loop {pct(embed_util)} "
      f"of {os.cpu_count()} cores", flush=True)
print(f"measured: per-month wall seconds {per_month}", flush=True)

# --- the host gather ALONE, over the identical batches ----------------------
# This is the number JAX_PORT.md §1b actually asks for. If it is a large
# fraction of the embed time the host is the bottleneck and the fix is a
# pre-gathered shard format — a build-side change, not a model change.
g0 = cpu_sample()
t = time.time()
gathered = 0
for ti in range(MONTHS):
    for i in range(0, P, BATCH):
        sl = slice(i, min(i + BATCH, P))
        n = sl.stop - sl.start
        if patch > 1:
            tt = np.full((n,), ti, dtype=np.int64)
            v, o = gather_px_np(Xt, OBS, tt, ys[sl], xs[sl], patch)
        else:
            v = np.asarray(Xt[ti, ys[sl], xs[sl]])
            o = np.asarray(OBS[ti, ys[sl], xs[sl]])
        gathered += int(v.size)
gather_s = time.time() - t
g1 = cpu_sample()
gather_util = util_between(g0, g1)
gather_frac = gather_s / embed_s if embed_s > 0 else None
print(f"measured: host gather alone {gather_s:.1f} s for the same {batches} "
      f"batches = {pct(gather_frac)} of the embed wall time "
      f"({gathered / gather_s / 1e6:.1f} M values/s)", flush=True)
print(f"measured: host CPU utilisation across the gather loop "
      f"{pct(gather_util)}", flush=True)
# The one line the whole run exists to produce, stated as a reading rather
# than left for a reader to derive from two timings (JAX_PORT.md §1b).
if gather_frac is not None:
    print("measured: the host gather is "
          + ("THE BOTTLENECK — a pre-gathered shard format is the "
             "build-side fix TPU_ACCESS.md §6 names"
             if gather_frac > 0.5 else
             "not the bottleneck at this batch size; the encoder dominates"),
          flush=True)

devices = [str(d) for d in jax.devices()]
print(f"measured: jax.devices() {devices}", flush=True)

res = {
    "stamp": os.environ["STAMP"],
    "what": "TPU v5e host-gather throughput smoke (TPU_ACCESS.md §6, "
            "JAX_PORT.md §1b). Throughput only: the tensor is RAW (no "
            "anomaly transform), so no number here is a probe number.",
    "caveat": "tensor decompressed into RAM, not memmapped — an UPPER BOUND "
              "on gather throughput",
    "repo": os.environ["REPO"],
    "commit": subprocess.run(
        ["git", "-C", os.path.join(os.environ["WORK"], "earth"),
         "rev-parse", "HEAD"],
        check=True, stdout=subprocess.PIPE).stdout.decode().strip(),
    "jax_version": jax.__version__,
    "jax_devices": devices,
    "jax_device_count": jax.device_count(),
    "host": {"cpus": os.cpu_count(),
             "mem_total_kb": int(open("/proc/meminfo").readline().split()[1])},
    "codec": {"tag": ck.get("tag"), "d_z": int(ck["d_z"]), "patch": int(patch),
              "d_model": ck["args"].get("d_model"),
              "n_layers": ck["args"].get("n_layers"),
              "channels": int(X.shape[-1])},
    "tensor": {"shape": [int(v) for v in X.shape], "dtype": str(X.dtype),
               "months_timed": MONTHS, "ocean_pixels_in_window": int(n_ocean),
               "decompress_s": round(decompress_s, 2)},
    "loop": {"batch": BATCH, "pixels": int(P), "batches": int(batches)},
    "timings": {
        "warmup_s": round(warm_s, 3),
        "embed_s": round(embed_s, 3),
        "embed_batches_per_s": round(steps_per_s, 4),
        "embed_pixel_encodes_per_s": round(steps_per_s * BATCH, 1),
        "embed_per_month_s": per_month,
        "gather_only_s": round(gather_s, 3),
        "gather_fraction_of_embed": (None if gather_frac is None
                                     else round(gather_frac, 4)),
        "gather_values_per_s": round(gathered / gather_s, 1),
    },
    "host_util": {"embed_loop": embed_util, "gather_loop": gather_util,
                  "source": "/proc/stat aggregate, busy=(total-idle-iowait)"},
}
with open(os.environ["RESULT"], "w") as fh:
    json.dump(res, fh, indent=2)
print(f"measured: wrote {os.environ['RESULT']} "
      f"({os.path.getsize(os.environ['RESULT'])} bytes)", flush=True)

# ---------------------------------------------------------------------------
# 4b · TRAINING smoke — a small model actually training on the TPU.
#
# A from-scratch PixelMAE at the PILOT geometry (128x4, d_z 32, ~0.9M params
# — the workflow's own default architecture) trained for a bounded number of
# steps on real tensor batches with a masked-channel reconstruction MSE.
# THIS IS A MECHANICAL SMOKE, NOT STAGE-1: the objective is a toy (no
# neighbour losses, no schedule, no anomaly transform), so the only numbers
# it produces are steps/sec, the compile time, and that the loss FALLS.
# Nothing here is comparable to any published number.
#
# Best-effort BUT LOUD (ml/CLAUDE.md §4.6): a failure here must not cost the
# embed result that already published above, so this block never raises —
# it writes either its measurements or its full traceback to TRAIN_RESULT,
# which the bash step publishes either way.
# ---------------------------------------------------------------------------
print("--- train smoke ---", flush=True)
train_res = {"status": "FAILED", "note": "toy masked-recon objective; not stage-1"}
try:
    import traceback
    import optax
    import jax.numpy as jnp
    from flax import nnx
    from jaxport.models import PixelMAE

    STEPS = int(os.environ.get("TRAIN_STEPS", "300"))
    TB = int(os.environ.get("TRAIN_BATCH", "4096"))
    C = int(X.shape[-1])
    rng = np.random.default_rng(0)

    model = PixelMAE(n_chan=C, rngs=nnx.Rngs(0))          # pilot defaults
    graphdef, state = nnx.split(model)
    n_params = sum(int(np.prod(v.shape)) for v in jax.tree_util.tree_leaves(state))
    tx = optax.adam(3e-4)
    opt_state = tx.init(state)
    coords_tr = np.stack([lats[ys] / 90, lons[xs] / 180], 1).astype(np.float32)

    def batch_np():
        t = int(rng.integers(0, X.shape[0]))
        idx = rng.integers(0, P, TB)
        xb = X[t, ys[idx], xs[idx], :].astype(np.float32)
        obs = np.isfinite(xb)
        xb = np.nan_to_num(xb)
        mask = (rng.random((TB, C)) < 0.3) & obs          # hide 30% of observed
        ctx = np.concatenate([np.tile(ctx_all[t], (TB, 1)),
                              coords_tr[idx]], 1).astype(np.float32)
        return (jnp.asarray(xb), jnp.asarray(obs), jnp.asarray(mask),
                jnp.asarray(ctx))

    chan_idx_tr = jnp.tile(jnp.arange(C)[None, :], (TB, 1))
    off0_tr = jnp.zeros((TB, C, 3), jnp.int32)

    def loss_fn(st, xb, obs, mask, ctx):
        m = nnx.merge(graphdef, st)
        z = m.encode(xb, obs, mask, ctx)                  # mask hides tokens
        pred = m.query(z, chan_idx_tr, off0_tr)
        w = (mask & obs).astype(jnp.float32)              # score hidden cells
        return jnp.sum((pred - xb) ** 2 * w) / jnp.maximum(jnp.sum(w), 1.0)

    @jax.jit
    def train_step(st, opt_st, xb, obs, mask, ctx):
        loss, grads = jax.value_and_grad(loss_fn)(st, xb, obs, mask, ctx)
        upd, opt_st = tx.update(grads, opt_st, st)
        return optax.apply_updates(st, upd), opt_st, loss

    losses = []
    t_c = time.time()
    state, opt_state, l0 = train_step(state, opt_state, *batch_np())
    compile_s = time.time() - t_c
    losses.append(float(l0))
    print(f"measured: train step 1 (includes XLA compile) {compile_s:.1f} s, "
          f"loss {float(l0):.4f}", flush=True)
    c0 = cpu_sample()
    t_l = time.time()
    for i in range(1, STEPS):
        state, opt_state, l = train_step(state, opt_state, *batch_np())
        losses.append(float(l))
    train_s = time.time() - t_l
    c1 = cpu_sample()
    sps = (STEPS - 1) / train_s
    train_res = {
        "status": "OK",
        "note": "toy masked-recon objective; NOT stage-1; no published number is comparable",
        "arch": {"d_model": 128, "n_layers": 4, "d_z": 32, "patch": 1,
                 "params": n_params, "channels": C},
        "loop": {"steps": STEPS, "batch": TB, "optimizer": "adam(3e-4)",
                 "mask_rate": 0.3},
        "timings": {"compile_s": round(compile_s, 2),
                    "train_s": round(train_s, 2),
                    "steps_per_s": round(sps, 3),
                    "pixel_examples_per_s": round(sps * TB, 1)},
        "loss": {"first10_mean": round(float(np.mean(losses[:10])), 5),
                 "last10_mean": round(float(np.mean(losses[-10:])), 5),
                 "fell": bool(np.mean(losses[-10:]) < np.mean(losses[:10]))},
        "host_util_train_loop": util_between(c0, c1),
        "jax_devices": [str(d) for d in jax.devices()],
    }
    print(f"measured: {STEPS} steps in {train_s:.1f} s = {sps:.2f} steps/s "
          f"({n_params:,} params, batch {TB})", flush=True)
    print(f"measured: loss first-10 {train_res['loss']['first10_mean']} -> "
          f"last-10 {train_res['loss']['last10_mean']} "
          f"(fell: {train_res['loss']['fell']})", flush=True)
except Exception:
    train_res["traceback"] = traceback.format_exc()
    print("TRAIN SMOKE FAILED — traceback follows (shipped in TRAIN_RESULT)",
          flush=True)
    print(train_res["traceback"], flush=True)
with open(os.environ["TRAIN_RESULT"], "w") as fh:
    json.dump(train_res, fh, indent=2)
print(f"measured: wrote {os.environ['TRAIN_RESULT']}", flush=True)
PY

# --------------------------------------------------------------------------
# 5 · publish the result to the staging bucket
# --------------------------------------------------------------------------
# The metadata server, not a key: TPU_ACCESS.md §5's rule is that the
# service-account JSON never lands on a machine we do not own, and a TPU VM
# already carries its attached service account's token one HTTP call away.
# Nothing secret is written to disk here and nothing outlives the node.
echo "--- publish ---"
OBJ="tpu_smoke/${STAMP}.json"
TOKEN="$(curl -fsSL -H 'Metadata-Flavor: Google' \
  'http://metadata.google.internal/computeMetadata/v1/instance/service-accounts/default/token' \
  | "${PY}" -c 'import json,sys; print(json.load(sys.stdin)["access_token"])')"
# --data-binary buffers the whole file, which ml/CLAUDE.md §7 warns about for
# GB-sized uploads (`-T` streams). Deliberate here: this is a few-KB JSON, and
# a simple `uploadType=media` POST is the whole transfer.
curl -fsSL -X POST \
  -H "Authorization: Bearer ${TOKEN}" \
  -H "Content-Type: application/json" \
  --data-binary "@${RESULT}" \
  "https://storage.googleapis.com/upload/storage/v1/b/${BUCKET}/o?uploadType=media&name=$(
     "${PY}" -c 'import sys,urllib.parse; print(urllib.parse.quote(sys.argv[1], safe=""))' "${OBJ}")"
echo
echo "measured: published gs://${BUCKET}/${OBJ}" \
     "($(stat -c%s "${RESULT}") bytes)"

# The training smoke's result (or its traceback — 4b never raises, so this
# file exists either way and the publish is unconditional).
TOBJ="tpu_smoke_train/${STAMP}.json"
curl -fsSL -X POST \
  -H "Authorization: Bearer ${TOKEN}" \
  -H "Content-Type: application/json" \
  --data-binary "@${TRAIN_RESULT}" \
  "https://storage.googleapis.com/upload/storage/v1/b/${BUCKET}/o?uploadType=media&name=$(
     "${PY}" -c 'import sys,urllib.parse; print(urllib.parse.quote(sys.argv[1], safe=""))' "${TOBJ}")"
echo
echo "measured: published gs://${BUCKET}/${TOBJ}" \
     "($(stat -c%s "${TRAIN_RESULT}") bytes)"

echo "=== tpu_smoke done — results published; the EXIT trap now deletes this node ==="

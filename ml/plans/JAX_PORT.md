# JAX port — effort assessment

**Status: TIER 1 LANDED, TIER 2 EMBEDDING PATH LANDED (2026-08-21).**
§§2–6 are the original assessment. `ml/jaxport/` holds the models, the
converter and `embed_everything_jax`. Gates so far:

- **G1 PASSED** — converted `f3_anchor41M` (40.7M, 576×10 patch 3, d_z 64)
  agrees with torch to max|Δ| 1.3e-5 on `encode`, 6.6e-7 on `query` (gate
  1e-4); toy suite (`tests/test_jaxport_parity.py`, 6 checks) ≤ 1e-6.
- **G2′ PASSED** (the monthly stand-in defined in §5 — G2 itself stays
  OPEN): the rapid ridge over the real `family3_na025` tensor reads
  **r 0.62696** from the JAX embedding and **0.62696** from torch, against
  the archived control **0.627**; the two backends differ by **5.7e-8** in
  r, and their Z arrays by mean|Δ| 1.2e-6 (max 7.8e-3 = one float16
  storage ULP; float32 re-encode 1.0e-5).
- **G3 PASSED** — the `e017_u1_s0` gate head re-rolls at gate AUC **0.643**
  from both backends (recomputed 0.642833 either side, Δ 1.7e-4 vs the
  archive, tol 0.0101), with `auc_damped` 0.619 also reproduced. All twelve
  per-horizon rows — `msss_clim`, `msss_pers`, `msss_damped`, `acc`,
  `amp_ratio` and the sample counts — are **bit-identical between torch and
  jax**; the rolled states differ by mean|Δ| 2.4e-6 (max 1.1e-3 at h=12,
  7e-5 relative), compounding monotonically over the 12 iterated steps as
  it must. Scored on the gate scope's 864 pixels; corridor and window are
  reported as NOT SCORED rather than approximated — see §6b.

Reproduce: `ml/jaxport/score_section_probe.py` (G2′) and
`ml/jaxport/score_gate_roll.py` (G3), both `--backend {jax,torch}`.

## 1 · Why

A framework-independent implementation of the models makes them usable in
JAX-based research stacks (which includes most TPU environments), lets other
groups reproduce and extend the results without adopting our PyTorch
tooling, and gives the programme a second, independent implementation of its
own arithmetic — the strongest kind of protocol check we can build, given
that the probe ladder's reference constants (§5) already exist to score it
against.

The port is a REFERENCE implementation, not a migration. The fleet, the
dispatch workflow, the recipes and every operational number stay on the
PyTorch stack. The JAX tree must be able to *read* the published artefacts
(checkpoints, tensors, embedding caches) — it must never be required for
producing them.

## 1b · Goal update (2026-08-21): TPU training

The port now has an operational target beyond reference: **be able to train
these models on Google Cloud TPUs.** The rented-GPU fleet's economics are
known ($0.30–0.94/h, 15–36 h per 200k-step run); TPU spot capacity is the
one obvious lever this programme has not tried, and JAX is the native way
to reach it. Consequences for the plan:

- **Tier 3 (training parity) moves onto the critical path** — TPU training
  is stage-1/stage-2 trainers in JAX, so tier 3 is no longer "only after
  someone asks". The ordering discipline stands: tier 3 is dispatched only
  after tier 2's gates are green, because a trainer validated against an
  eval stack that itself drifted proves nothing.
- **Tier 4 gets a concrete shape**: a TPU VM registered as a self-hosted
  Actions runner, exactly the Vast pattern (`runner: tpu-<name>` pin, same
  workflow, same artefact releases, same security posture — `ml/CLAUDE.md`
  §6 applies unchanged, including never adding non-dispatch triggers).
  Until then, TPU runs can be driven directly over `gcloud` without any
  workflow changes.
- **Data plane on TPU**: tensors stage to a GCS bucket and download to the
  TPU VM's local disk/RAM at job start (the `data-cache-v1` seed pattern,
  different remote). The per-batch host gather (`LazyPixels`/`gather_px`)
  runs on the TPU VM's host CPU with prefetch/double-buffering; whether it
  feeds a TPU fast enough is a MEASUREMENT for the first smoke run, not an
  assumption — if it starves, the fix is a pre-gathered shard format, which
  is a build-side change, not a model change.
- **Cross-framework discipline (§3.4) applies with force**: TPU-trained
  numbers are a new tier under `ml/CLAUDE.md` §3b — the first result buys
  its own replication, and nothing is pooled with the torch/GPU record.
  The cheap validation is the other direction: a TPU-trained checkpoint,
  converted back, scored through the UNCHANGED torch eval ladder.

What this needs from the operator, none of which blocks tiers 1–2: a GCP
project with TPU quota (spot v5e/v6e-8 is the sensible first shape),
credentials the session can use (a service account key, or the operator
running the provisioning commands), and a GCS bucket for tensor staging.

## 2 · What exists (measured, not guessed)

`ml/` is ~19,900 lines of Python; **~10,100 lines across 19 files import
torch**. But the MODELS themselves are small — four `nn.Module`s, ~450 lines
total:

| module | file | size | what it is |
|---|---|---|---|
| `PixelMAE` | `ml/model.py` | ~120 lines | codec: token-per-channel transformer encoder (`nn.TransformerEncoder`, norm-first, dropout 0) + queryable MLP decoder over (z, channel, Δlat, Δlon, Δt) |
| `TemporalTransformer` | `ml/temporal.py` | ~60 lines | stage 2: causal transformer over per-pixel embedding sequences, stencil-widened input, optional direct multi-horizon heads |
| `SectionHead` | `ml/probe_head.py` | ~40 lines | unpooled cross-attention read-out over section tokens |
| `MultiDec` | `ml/recon_decoder.py` | ~50 lines | reconstruction decoder used by recon_eval |

Everything else that imports torch is *machinery around* those four:
training loops (`train.py` 1,127 · `temporal.py` 2,435 · `train_joint.py`
790), eval/rollout (`rollout_spatial.py` 1,885 · `rollout.py` 646), probes
(`probe_kfold.py` 471 · `probe_head.py` 491 · `probe_sequence.py` 243 ·
`trainprobe.py` 494), and utilities. Of the 54 Python tests, 30 import
torch.

Three structural facts make this port much cheaper than the line count
suggests:

- **The data plane is numpy, not torch.** Tensors are memmapped numpy
  (float16/float32), `LazyPixels`/`gather_px` index numpy and convert at the
  batch boundary, the embedding cache `Z` is a numpy memmap keyed by codec
  weight hash, and `probe_kfold`'s ridge is numpy. All of it is reusable
  as-is from JAX (host-side gather → `jnp.asarray` per batch).
- **No custom kernels.** No `autograd.Function`, no C++/CUDA extensions, no
  hand-written triton. The only exotic dependency is torch's own internal
  triton path under `SectionHead` (the cc-guard saga), which a JAX port
  simply doesn't have.
- **Mixed precision is shallow.** fp16 shows up as (a) float16 *storage*
  widened to the weight dtype at the encoder boundary and (b) an optional
  `--amp` autocast on the roll/decode forwards. No GradScaler anywhere.

## 3 · What is genuinely hard

1. **Checkpoint conversion, not architecture.** ~365 published assets on
   `model-checkpoints-v1` are torch `.pt` dicts whose `args` carry the
   architecture (`codec_from_ckpt` is the single reader — port that contract
   exactly). `nn.TransformerEncoderLayer` packs QKV as one in-proj; Flax/
   Equinox split them — the converter owns that mapping, and a wrong slice
   produces plausible garbage, which on this project is the most dangerous
   failure shape there is.
2. **Parity is tolerance-based, never bit-exact.** The archive gives us
   hard acceptance gates for free (§5), but "the gate reproduces 0.643
   within 0.0101" across frameworks needs the same read-out pipeline
   (masking semantics, month features, stencil order — CENTRE FIRST, the
   checkpoint layout depends on it) transplanted without drift.
3. **Optimizer-state resume.** Post-2026-08-10 snapshots carry `opt`/
   `sched`/`step` (torch Adam state). Mapping that into optax state is
   fiddly and only matters if JAX ever *continues* a torch run — out of
   scope for tiers 1–2, decide-then-build for tier 3.
4. **Cross-framework numbers are a new tier under §3b.** The measured "box
   effect" (0.041 head k-fold at fixed seed from an environment difference)
   is the warning: JAX-trained results must not be pooled with torch
   results, and the first JAX result at any tier buys its own replication.
   The port changes no science until that is paid for.

## 4 · Tiered plan and effort

Effort in focused agent-days (implementation delegated per `ml/CLAUDE.md`
§0b; design and verification stay with the planning session).

| tier | scope | new code | effort | risk |
|---|---|---|---|---|
| **1 — models + converter** | the four modules in Flax NNX (or Equinox — pick once, §6), `.pt` → pytree converter honouring `args`, forward-parity tests vs torch on CPU (toy + one real released codec, atol ~1e-5 fp32) | ~900 lines + tests | **2–3 days** | low |
| **2 — eval stack** | `embed_everything`, `roll_step`/`decode_all`, `probe_head` fold-fit (optax), probe_kfold/sequence/dip_check drivers reusing the numpy plumbing; scored against §5 gates | ~2,500 lines | **4–6 days** | medium — this is where read-out drift can hide |
| **3 — training parity** | stage-1 and stage-2 trainers (masking, neighbour losses, invsqrt/wsd/expdecay in optax, grad clip, collapse guard, light probes, milestone/resume) | ~3,000 lines | **7–10 days** | high — trainer behaviours are load-bearing lore |
| **4 — fleet integration** | recipes/workflow/dispatch for JAX runs | — | **not planned** | the fleet stays torch; revisit only if a JAX training result ever needs renting GPUs |

Total for the recommended scope (tiers 1–2): **roughly one to one-and-a-half
weeks of delegated implementation**, ~3,500 lines of new code in an isolated
`ml/jaxport/` package with zero imports from it into the operational tree. Tier
3 roughly doubles that and should be dispatched only after tier 2's gates
are green.

## 5 · Acceptance gates (pre-registered)

The protocol-determinism archive is the scoring instrument; each gate names
the number the port must reproduce and the tolerance it inherits:

- **G1 (tier 1):** converted codec forward on a fixed batch matches torch
  elementwise, atol 1e-4 fp32, on the pilot codec AND one xl checkpoint.
- **G2 (tier 2):** `probe_kfold` over `f3_anchor41M` on the pentad tensor
  reads rapid r **0.660** (torch reproduces this identically across #390/
  #392/#397/#406); accept within ±0.005. **STILL OPEN**, for a data reason
  rather than a port reason: the pentad tensor is not published on
  `data-cache-v1`, so it cannot be staged into a sandbox. Score it on the
  first box that has the tensor.
- **G2′ (the monthly stand-in that was actually scored, 2026-08-21).** Same
  codec, same instrument, the tensor that IS published:
  `family3_na025_adcbe700fb` (sha `adcbe700fb…`, 516×281×481×39). Its
  archived control is **rapid r 0.627 [0.503, 0.735], n 240, RMSE 2.17 Sv**
  — `probes-140.json`, and identically in all 95 bundles #140→#360 that
  froze `!run-62,run-63` on this tensor (`ml/EXPERIMENTS.md` §"Where the
  control number comes from"). Accept within ±0.005.
  **Substituting a gate needs saying out loud, so: G2′ is weaker than G2 in
  one way and stronger in another.** Weaker — it is monthly, so it never
  exercises the pentad cadence. Stronger — it is scored TWICE through the
  identical numpy read-out, once from a torch Z and once from a JAX Z, so
  it measures the thing G2 can only infer: that swapping the framework
  moves the published instrument by 5.7e-8. Neither number was tuned; the
  torch backend was required to land on 0.627 first, precisely so the JAX
  number would be scored against the right thing.
- **G3 (tier 2):** the `e017_u1_s0` gate head re-rolls at gate AUC
  **0.643** (18 torch reproductions on record); accept within the eval
  wave's own tolerance, 0.0101.
- **G4 (tier 3, if built):** a from-scratch toy training run (the
  `--smoke` tensor) tracks the torch loss curve within seed-level spread.

A gate that fails is a finding about the port, never something to widen.

## 6 · Decisions to make before dispatch

- **Flax NNX vs Equinox** — NNX is the ecosystem default and the safer
  choice for outside users; Equinox is closer in feel to `nn.Module`.
  Recommendation: NNX.
- **Converter direction** — one-way (torch → JAX) is all tiers 1–2 need.
  Two-way is only needed if a JAX-trained model must enter the torch eval
  ladder; defer.
- **Where parity tests run** — CPU-only in CI (JAX CPU wheel is cheap);
  no GPU rental for any of tiers 1–2.

## 7 · Findings from the tier-2 work (2026-08-21)

**A memmapped `anomaly_transform` can be OOM-killed silently on a
RAM-oversubscribed box, leaving a HALF-TRANSFORMED tensor on disk.** Measured
here on a 7 GB box against the 10.88 GB monthly tensor (1.55×
oversubscribed): the transform's pass 3 died with no traceback and no OOM
line, and the file left behind was z-scored for part of its time axis and
raw for the rest. `trainprobe.py:anomaly_transform` is chunked over time
precisely so it can run oversubscribed, but numpy's memmap only msyncs on an
explicit `flush()`, which the transform never calls mid-pass — so on a box
where writeback cannot keep up, the dirty pages accumulate until the kernel
kills the process. The fix used here is caller-side (a write-through proxy
that msyncs every 1 GiB, plus `chunk=8`); `trainprobe.py` was NOT changed,
because changing it would change the arithmetic path of every published
number and that is a decision for its own experiment.

Why it matters beyond this sandbox: the daily family-5 tensor is **2.6×**
oversubscribed on a 64 GB box, i.e. further into this regime than the case
that failed here, and the surviving artefact is the dangerous part. A
half-transformed tensor is not detectably wrong — it is finite, correctly
shaped, and passes every check the pipeline makes — so the failure would be
read as "the probe died, re-run it", and the re-run would embed a tensor
that is anomaly-space in one half and raw in the other. The cheap guard is
the repo's own flush-then-mark discipline (`ml/CLAUDE.md` §5.21): write a
marker beside the tensor AFTER the transform completes, and refuse to embed
a tensor whose marker is missing. That guard does not exist on the
operational path today.

## 6b · What G3 scored, and what it deliberately did not

G3 was run on the **gate scope's 864 pixels**, not on the corridor (29,627)
or the whole window (84,405). That is a compute refusal with an exactness
argument behind it, not a shortcut, and the distinction matters enough to
write down.

One roll step is one forward per rolled pixel and the protocol rolls 234 of
them. Measured on the two CPU cores available here: 7.1 s/step torch and
8.6 s/step jax at 864 pixels — 27.6 and 33.4 minutes. The cost is linear in
pixels, so the full window would be **~45 h and ~54 h**. Nothing about that
is a statement about TPUs or GPUs; it is a statement about scoring a gate on
a laptop-class box.

**The reduction is exact for the gate scope and only for it.** `e017_u1_s0`
is a **stencil-1** head, so `roll_step` has no cross-pixel term at all —
window, static context, decode, the AR1 baseline and `accumulate`'s sums are
each per-pixel — and the gate's sums are therefore bit-identical whether the
other 83,541 pixels rolled beside them or not. Corridor and window are NOT
recoverable that way, so the driver emits a `not_scored` key naming the
archived value and the reason, rather than printing an AUC over
corridor∩gate under the name "corridor". A stencil>1 head is refused
outright under `--scope gate`, for the mirror-image reason: there the
neighbours are part of the arithmetic and dropping them would change the
number silently.

Both backends ran the identical 864-pixel subset, the identical published Z
and the identical cached static-identity embedding, which is what makes the
backend-to-backend comparison like-for-like.

**Why the published Z is the right input here.** G3's job is to isolate the
ROLLOUT. The embedding path already has its own gate (G2′, backends agreeing
to 5.7e-8 in r), so feeding both backends the same torch-produced
`Z_6c52f0687b_adcbe700fb` — canonical for this codec on this tensor, keyed
by the codec's own weight hash — removes the encoder as a variable instead
of re-testing it. It also costs nothing: embedding 84,405 pixels from
scratch is ~43.5M encoder forwards, which on this box is over a hundred
hours.

# JAX port — effort assessment

**Status: ASSESSMENT (2026-08-21). Nothing is ported yet; this document sizes
the work and fixes the ground rules before any of it is delegated.**

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
`ml/jax/` package with zero imports from it into the operational tree. Tier
3 roughly doubles that and should be dispatched only after tier 2's gates
are green.

## 5 · Acceptance gates (pre-registered)

The protocol-determinism archive is the scoring instrument; each gate names
the number the port must reproduce and the tolerance it inherits:

- **G1 (tier 1):** converted codec forward on a fixed batch matches torch
  elementwise, atol 1e-4 fp32, on the pilot codec AND one xl checkpoint.
- **G2 (tier 2):** `probe_kfold` over `f3_anchor41M` on the pentad tensor
  reads rapid r **0.660** (torch reproduces this identically across #390/
  #392/#397/#406); accept within ±0.005.
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

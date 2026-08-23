# E-046 · An FSQ bottleneck for the pentad codec — PLAN, not yet dispatched

**Status: SPEC ONLY.** Nothing here has been trained. Written 2026-08-22 so the
decision below is made deliberately rather than at 2 a.m. with a box running.

## 0 · Summary

Chris's direction: *the head has too much input for its capacity*. The pentad
stage-2 head reads a **continuous** 32-dimensional embedding per pixel per
step — at K=24 and stencil 145 that is 111,360 real numbers entering one
forward — and the arm that trains on it (#427) lands at a one-step ratio of
**0.5056** where the monthly arm reaches 0.0139, with an unforced roll that
mode-locks to the calendar. This plan replaces the codec's continuous
bottleneck with **Finite Scalar Quantization** (FSQ, arxiv 2309.15505,
*"Finite Scalar Quantization: VQ-VAE Made Simple"*), so the embedding the whole
programme consumes becomes a **finite alphabet** instead of an unbounded
vector, and re-trains the pentad codec on it.

Two things make this affordable and separable. It is **one codec retrain**
(~20 h, ~$6, the #415 class) and nothing downstream changes shape:
`ml/temporal.py` consumes `z_q` exactly as it consumes `z` today, because a
quantized embedding is still a float vector of width `d_z`. And the cheap
version of the same hypothesis is **already in the tree and testable tonight
without a retrain** — `ml/temporal.py --input-quant`, which quantizes what the
HEAD reads while the codec stays continuous (E-044c, committed 2026-08-22).
**Run that first.** If restricting the head's input alphabet moves nothing,
the capacity hypothesis is weaker than it looks and this retrain should wait.

## 1 · What FSQ is, and what the paper actually claims

FSQ replaces VQ-VAE's learned codebook with per-dimension rounding to a fixed
set of levels. For a `d`-dimensional bottleneck with level counts
`L = [L_1 … L_d]`:

    z -> bound(z) -> round(.) -> z_q,     codebook size = prod(L_i)

with the gradient passed **straight through** the rounding. There is no
codebook parameter, so there is no commitment loss, no EMA update, no
codebook-restart or dead-code machinery — that removal *is* the paper's
contribution, and the reason it is cheap to try here.

Claims I am relying on, with my confidence marked:

- **FSQ matches VQ-VAE at large codebook sizes** (roughly 2^11 and above) on
  reconstruction and downstream generative metrics, in the paper's MaskGIT and
  UViM settings. *(Confident on the shape of the claim; the exact crossover
  size is TO VERIFY against the paper.)*
- **Codebook usage is essentially 100 %** by construction, where VQ commonly
  leaves a large fraction of entries dead. *(Confident.)*
- **FSQ is worse than VQ at small codebooks** (below ~2^10), because the
  per-dimension grid cannot allocate capacity where the data is. *(Fairly
  confident; TO VERIFY.)* This matters for our level choice below.
- **Recommended level configurations** by target codebook size, from the
  paper's own table: `2^8 -> [8,6,5]`, `2^10 -> [8,5,5,5]`,
  `2^11 -> [8,8,6,5]`, `2^12 -> [7,5,5,5,5]`, `2^14 -> [8,8,8,6,5]`,
  `2^16 -> [8,8,8,5,5,5]`. *(TO VERIFY — these are from memory of Table 1.
  The pattern — a handful of dimensions with 5-8 levels each — is the part I
  am confident in, and the pattern is what the design below rests on.)*

The implementation in `ml/temporal.py:InputQuant` (E-044c) already follows this
map, including the even-`L` offset that makes "L levels" true rather than
`L-1`, and it refuses `L=2` as degenerate. **Reuse it**; do not write a second
quantizer (this repo has a standing rule about second copies of one transform,
and it was written after the anomaly transform had four).

## 2 · What changes in `ml/train.py`

One module and one line of forward, plus the arguments that record them.

1. **A bottleneck module.** `PixelMAE.encode` ends `return self.to_z(h[:, 0])`
   (`ml/model.py:137`). FSQ inserts one step: `z = quant(self.to_z(h[:, 0]))`,
   where `quant` is the E-044c map with the codec's own per-dimension scale.
   The scale cannot be measured from Z here (Z is what we are producing), so
   the bound must be **learned or fixed**: the paper bounds with `tanh` on the
   raw pre-quantization activation, which is what we do — `sigma_d = 1` and let
   the encoder learn the scale into `to_z`. *(This is the one real difference
   from the E-044c head-side knob, where sigma is measured from an existing Z.)*
2. **No new losses.** No commitment term, no codebook loss, no EMA. The
   reconstruction and neighbour losses in `step_loss` (`ml/train.py:562-594`)
   are untouched.
3. **`d_z` becomes the FSQ dimension count**, and this is the substantive
   design decision — see §4.
4. **New args, saved by `vars(a)` as everything else is**: `--fsq-levels`
   (comma list, empty = today's continuous bottleneck = bit-identical) and the
   resulting codebook size logged once at startup. A codec trained without the
   flag must remain bit-identical, and the test for it is the pattern
   `tests/test_e044_grad_clip.py` and `tests/test_e044c_knobs.py` already use.
5. **`ml/model.py:codec_from_ckpt` reads the levels back** from `ck["args"]`,
   so every consumer (`temporal.py`, `probe_kfold.py`, `rollout_spatial.py`,
   `recon_eval.py`) rebuilds the same bottleneck without being told.

## 3 · What does NOT change

- **`ml/temporal.py` is untouched.** It receives `[T, P, d_z]` floats. That the
  values now live on a lattice is invisible to it, which is the whole reason
  this is a codec-side change rather than a programme-wide one.
- **The embed-cache mechanism is unchanged** — and the *key* changes by itself,
  correctly: `codec_weight_hash` is an md5 over the codec's first four weight
  tensors, so a new codec is a new hash, a new cache name, and no possibility
  of colliding with `Z_8b639abe36_37e146384b`. Budget the **8.5 h embed** for
  the first stage-2 run on it (measured, #423).
- **The tensor is unchanged.** `family4_na025_pentad_r2`, sha256
  `37e146384b…`, pulled by the workflow since `ce361b0`.
- **The eval protocol is unchanged**, which is what makes the falsifier below
  a comparison and not a new measurement.

## 4 · The decisions, and what I recommend

**(a) Levels, and therefore the codebook.** The paper's own configurations put
5-8 levels on each of 3-6 dimensions. Two candidates:

| option | levels | dims | codebook | bits/pixel/step |
|---|---|---|---|---|
| **A (recommended)** | `[8,8,8,6,5]` | 5 | 15,360 ≈ 2^13.9 | 13.9 |
| B | `[8,8,8,5,5,5]` | 6 | 64,000 ≈ 2^16 | 16.0 |

Against today's continuous `d_z=32` at float16, which carries up to 512 bits
per pixel per step and is why the capacity argument exists at all. Option A is
the paper's 2^14 configuration and sits comfortably above the ~2^10 region
where FSQ is reported to lose to VQ.

**(b) `d_z`.** Option A means **`d_z` drops from 32 to 5**, and that is not a
side effect — it is the intervention. It also shrinks the head's input by
6.4x (111,360 numbers -> 17,400 at K=24, stencil 145), shrinks Z from
16.24 GiB to **2.54 GiB** (embed, publish and pull all get cheap), and changes
the stage-2 head's first-layer width. Everything downstream reads `d_z` from
the checkpoint already, so nothing needs editing — but every published pentad
number is against `d_z=32` and **the comparison must be stated as
codec-vs-codec, not as a knob turn**.

**(c) Do NOT bundle the r3 Argo-fill tensor revision.** The round-6 audit
found the mechanism behind the corridor NaNs: `rg_*` is a monthly product
written into **one pentad bin per month** (`n_rg_live` 252/3142, the mid-month
stamp, 79.41 % of ocean pixels when present), and the same audit showed the
codec's round trip collapsing on exactly those bins (FVU 15-18 % against
0.4-0.5 % on Argo-free bins) because 40 channels compete for 32 dimensions.
An r3 tensor that interpolates Argo across the month would change **the data**
at the same moment FSQ changes **the representation**, and a single arm cannot
attribute the result. One variable. FSQ first — it is the cheaper of the two
and its falsifier is sharper.

**(d) Cadence.** Pentad, on the existing tensor. The monthly anchor is a
control that already exists; re-running it under FSQ is a second experiment
and only worth it if the pentad one moves.

## 5 · Cost

| item | estimate | basis |
|---|---|---|
| codec retrain (stage 1) | **~20 h, ~$6** | #415's class: 197,428 steps x 512 on the pentad tensor at ~$0.30/h |
| first embed on the new codec | ~8.5 h, ~$2.6 | MEASURED (#423); halves-ish if `d_z=5` shrinks the forward, TO VERIFY |
| one stage-2 head to the 20k decision point | ~2 h, ~$0.6 | #427's 0.371 s/step, stopped at the falsifier |
| **to a first verdict** | **~$9** | |
| full stage-2 (200k) + roll, only if the verdict is positive | +~$11 | #427 20.6 h + a 14.1 h roll |

## 6 · The falsifier

Two numbers, both against controls that already exist, and the arm is stopped
at the first.

1. **Stage-2 one-step ratio at 20,000 steps.** The control is **#427's
   trajectory on the same tensor and the same protocol** (final ratio
   `z_t+1.mse_model / mse_persistence` = **0.5056**; its 20k point is on
   `ml-metrics` `run-427.jsonl` and must be read off it rather than
   remembered). FSQ passes only if it is **clearly below** the control at the
   same step — not within the noise of it. If it is at or above, the
   bottleneck's continuity was not the binding constraint and the retrain is
   answered for $9.
2. **Does the roll stop collapsing?** The unforced future roll of every head
   to date mode-locks to the calendar (gate 12-month, the nolonhold pair
   exactly 36-month with peaks pinned to the same months and phase-identical
   across seeds). The E-044c `longstart:` discriminator (multi-context-end
   hindcasts, committed 2026-08-22) is the instrument: if a FSQ head's peaks
   still land on the same calendar months regardless of context end, the
   quantization changed the input alphabet and not the failure.

**What a positive result would NOT establish**, said in advance: n=1 at a new
codec, a new `d_z` and a new bottleneck at once. §3b's harder clause applies —
the first result at a tier buys its own replication — so a seed-1 arm is
required before any number is written as a level.

## 7 · Order of work

1. **Tonight, no retrain:** `--input-quant 8` on a pentad stage-2 arm (E-044c).
   It is the same hypothesis at $0.6, and it constrains this plan's priors.
2. Implement `--fsq-levels` in `ml/train.py` + `ml/model.py`, default-off,
   with the bit-identity test. ~1 day, no GPU.
3. A **smoke retrain** (2,000 steps) to confirm the bottleneck trains at all
   and that `codec_from_ckpt` round-trips the levels. ~$0.2.
4. The full retrain, then the 20k decision point above.

## 8 · Open questions to settle before step 2

- The exact level table from the paper (§1) — read the paper, do not trust
  this document's memory of Table 1.
- Whether the FSQ bound should be `tanh` on a learned scale or an explicit
  per-dimension normalisation — the paper uses the former; our head-side
  `InputQuant` uses a measured sigma because it quantizes an existing Z.
- Whether `patch=1` (pentad) interacts with a 5-dimensional bottleneck: at
  patch 1 the encoder compresses one pixel's 40 channels into `d_z`, and
  40 -> 5 is a 8x channel-to-dimension ratio where the monthly anchor runs
  39 -> 64. This is the strongest reason to consider option B, or `[8,8,8,6,5]`
  with **two** FSQ groups per pixel (10 dims, 2^27.8) — TO DECIDE, and the
  round-6 recon audit is the measurement that should decide it.

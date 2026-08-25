# E-049 · Road B: one 16-bit token per pixel-bin, judged by a non-linear decoder

Chris, 2026-08-25: *"I think it's best that we continue with road B and do that
very diligently, and test it very well with a decoder (a non-linear decoder).
Please proceed."* — "road B" from the token-design review of the same morning:
the **paper-faithful FSQ token** (arxiv 2309.15505's own regime — few
dimensions, small levels, ONE token per unit from a vocabulary the paper's
downstream models can actually hold), as against "road A" (regrouping the
existing 32×8 lattice's digits into 8 tokens of 4,096, which changes the
spelling and not the representation).

In the same message Chris REFUTED the per-step-error reading of the pentad
collapse — *"The issue here is not additional error at all, but that with
pentad we tried to predict the next frame from much less (6x less) past
context (measured in days)"* — which is exactly what the E-045 factorial
measured (A11: 30-d step at 120-d span → 0.527 ≈ the pentad pair's 0.505;
E-045.2: 10-d step at 720-d span → 0.0804). Consequence for THIS plan: the
5-day unit is **not disqualified** as road B's unit, because the cadence
difficulty was context starvation, not per-step noise. Road B therefore runs
at the per-bin unit — the unit every measured pentad reference number lives
at — and the span question stays where it belongs, in stage-2's K.

## 1 · Hypothesis, and the falsifier

**Hypothesis.** A pixel-bin's live state — typically the 8 fast channels,
~128 bits of stored float16 — is compressible to **one 16-bit token**
(d_z 6, FSQ levels [8,8,8,5,5,5], implicit codebook 2^16 = 64,000) with
enough fidelity that a non-linear decoder reconstructs the fast channels at
a price comparable to what this programme has already accepted for a big
squeeze: the month-block codec's 9–19% FVU on Argo-free cells (E-047 Tier-1,
accepted as "the structural price of a real ≈4:1 squeeze").

**Falsifier, registered before dispatch.** If, at the DECODER CEILING
(decoder-only retrain against the frozen token lattice, §4c), any of the 8
fast channels on Argo-free bins reads FVU at or near 100% — the channel's
own variance, i.e. a constant would do as well — then 16 bits cannot carry a
bin and this configuration is dead. The pre-registered next rung is then
**two tokens per bin** (d_z 12 = [8,8,8,5,5,5]×2, 32 bits), not a bigger
single vocabulary — 2^16 is already the top of the FSQ paper's own table.

**What is deliberately expected and does not count against road B:** the
Argo-carrying bins (8.0% of bins, 40 channels) will reconstruct badly at
d_z 6 — they already collapse at d_z 32 (cur_speed 112%, ssh 86% FVU,
E-044b-roll audit). Road B's unit answer for Argo is a block/window unit or
a second token; this experiment measures the damage rather than hiding it,
and scores its hypothesis on the Argo-free 92%.

## 2 · The arms — attribution by construction

Three cells, two of them new. Every comparison is FVU per channel from the
same audit protocol (§4), so the two taxes separate:

| cell | bottleneck | status | what its FVU means |
|---|---|---|---|
| run-415 | d_z 32 continuous | recorded (0.4–0.6% Argo-free) | the reference |
| **E-049a** `f4r2-40M-dz6` | d_z 6 continuous | to train | **width tax** = a − 415 |
| **E-049b** `f4r2-40M-dz6-fsq65k` | d_z 6, FSQ [8,8,8,5,5,5], fitted ladder, ln-bounded | to train | **quantization tax** = b − a |

Everything else is `f4r2-40M-nolonhold` field for field: pentad r2 tensor,
512×12×4 heads, d_dec 256, patch 1, batch 512, anomaly, holdout_lon "0,0",
head_probe true, 200k steps. One variable per contrast.

Arm b's quantizer is a composite of three settings that only function
together and are declared as ONE variable: the levels, the fitted ladder
(`auto`, refits at steps 50/200/2000/20000), and the **intrinsic bound**
(§3). A no-bound FSQ arm is a contingency, not a member of this wave —
run-455 already measured what an unbounded per-bin FSQ codec does (|v|~3e4,
the 8-level lattice worn as a sign code).

## 3 · What is new in code, and why it is load-bearing

**`--fsq-bound ln` — LayerNorm (no affine) on the pre-quantization
activation.** The E-048 record is unambiguous: "let the encoder learn the
scale in" is an incentive that does not exist (straight-through gives to_z no
gradient toward the bound). e048a collapsed at |v|~87 vs R=2; run-455
saturated at |v|~3e4 (sign code); e048a2's fitted ladder closed the collapse
but NOT the drift (std_med 0.73→20 across 28k steps, each fitted lattice
outgrown within ~10k steps). The experiment log's own design fork (08-25
00:30Z) names the fix: *give the latent an intrinsic bound — normalization
before the lattice — so the fit has a stationary target.* E-049b takes that
fork's first branch, for the road-B line only (E-048's w6s3 stays held on
its own fork). At 16 bits total, a sign-code degeneration would leave 6 bits
— the bound is not a refinement here, it is the difference between running
the experiment and re-measuring a known disease.

Mechanics: zero-mean/unit-RMS over the d_z dims of each vector, no learned
affine, applied before the ladder; recorded in the checkpoint (`fsq_bound`)
and rebuilt by every loader; `codec_from_ckpt`'s unknown-fsq-key refusal
extends to it, so an old loader refuses rather than silently dropping the
bound. The JAX trainer refuses `fsq_bound` until it implements it (the
port's standing contract). The fitted ladder stays on top: the bound makes
the distribution stationary, the fit still chooses form and per-dim radius
on the measured sample.

**Workflow wiring for `fsq_auto_step` (and `fsq_bound`)** as recipe-only
keys — the multi-step refit schedule exists in ml/train.py since a1de68f but
no torch dispatch could set it.

**`ml/fsq_usage.py` — effective bits, measured.** Loads a codec, encodes a
probe sample, and reports the per-dimension digit histogram, its entropy,
and the summed **effective bits per token** beside the nominal 16. This is
the instrument that distinguishes "16 bits, used" from "16 bits, worn as
6" — the number the run-455 lesson says must never again be inferred from
the levels argument. Run at every audit; report beside every FVU.

## 4 · The verdict instrument: a non-linear decoder, three ways

Chris's ask is explicit: test with a **decoder**, non-linear. Three rungs,
all non-linear, in increasing strength:

**(a) The codec's own decoder, full visibility** — the E-019a protocol
(encode with nothing masked, query all 40 channels at offset 0, FVU =
rmse² in standardized units), split train / held-out months / Argo-free vs
Argo bins. This is the same audit that produced every reference number in
§2's table, so the columns compare.

**(b) Per-channel r and the pooled view**, same audit — pooling can cancel
per-pixel error and the transport probes read pooled states; report both,
verdict on (a).

**(c) The decoder ceiling: decoder-only retrain against the frozen tokens**
— the E-019b1 protocol (multi-output head, the 1536×3 class that took the
monthly anchor's deep-T from 6.9% to 0.85%), trained on train-split bins
only, scored by the same audit. The information question — what do the 16
bits CARRY — is answered here, not at (a): the production decoder's
training emphasis is masked-dominated and has understated z before.
**The falsifier of §1 is evaluated at this rung.**

recon_eval.py / recon_decoder.py were built against the monthly family-3
section; the pentad application exists once (the E-044b-roll mechanism
audit of run-415 — which was an ad-hoc round trip, NOT this script, so the
0.4–0.6% reference row must be reproduced under whatever audit scores
E-049, not assumed comparable). Compatibility was audited at this commit
and the adaptation is a NAMED PRE-AUDIT TASK for the ~20 h training
window, because one blocker is silent: `stream_stats` sums float16 with a
float16 accumulator, and at family-4 shape an absolute-valued channel
overflows to inf → nan → is classified NOT DYNAMIC → passes to the codec
un-standardized, producing plausible garbage with no error. Also owed
before the audit: the per-(t,pixel) Argo-bin split (the falsifier's own
axis — today's scorer can only express month × longitude products), a
uint8→uint16 climatology counter (244 of 255 used at pentad cadence), and
an .npz-tensor input path. None of this was patched blind into the monthly
scripts at this commit; it is the first work item after dispatch.

## 5 · Pre-registered readings

- **Road B viable:** E-049b decoder-ceiling fast-channel FVU (Argo-free)
  within or below the month-block's 9–19% band, effective bits ≥ ~14 of 16.
  Licenses the stage-2 rung (a separate, later dispatch: token-input head,
  and only then token-output AR — neither is part of E-049).
- **Quantization is the binding constraint:** b − a large while a − 415
  small → the token, not the width, is what costs; escalate tokens/bin.
- **Width is the binding constraint:** a − 415 large already → 6 dims can't
  hold a bin even continuously; road B's next rung is d_z 12 continuous
  first, quantize second.
- **Sign-code check:** effective bits ≪ 16 with the bound in place would be
  a NEW disease (the bound removes the scale explanation) — stop and
  diagnose before spending on the next rung.
- **n = 1 discipline (ml/CLAUDE.md §3b):** a new codec, a new bottleneck
  and a new bound = a tier with no replicate band. First results are
  directions; the winner buys its seed pair before anything is a level.

## 6 · Cost and schedule

Two codec trainings at the run-415/455 class (200k × 512, ~20 h, ~$6 each on
a 4090-class box; job_timeout sized to the measured class, not the default),
plus a CPU-affordable audit + decoder retrain (E-019b1 ran on CPU). Dispatch
per ml/CLAUDE.md: recipes named, EXPERIMENTS.md entry at dispatch, plans
published, first-minutes verification (LR, and for arm b the bound's own
smoke signal — **`prequant_rms` = 1 and FLAT at every fit**. Not std_med:
LayerNorm fixes each VECTOR's rms, so a dimension carrying its energy in
the mean legitimately shows a small per-dimension std — measured 0.9998 /
0.38 on the CPU toy. A drifting std_med inside an intact rms is a moving
encoder, not a broken bound).

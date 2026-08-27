# E-057 · FGN-mode stage-2: noise-conditioned stencil head trained with fair CRPS

**Status: REGISTERED 2026-08-27 — approved by Chris same day (*"let's
prioritize an experiment with: 1. Noise-conditioned stage-2 head trained
with fair CRPS (the core FGN move)"*). Implementation (E-057.0) is the next
step; NOTHING is dispatched and nothing here touches the in-flight queue
(#489–#491, E-054a, E-052.1/1b).** Design argument:
[the E-052 FGN addendum](E052_FGN_addendum.md); source: FGN,
arXiv:2506.10772. E-057 supersedes E-052.p (same slot in the decomposition,
upgraded from σ-ladder conditioning to the full FGN move). In the E-052
2×2 this is the **{factorized, ε-sampling}** cell.

## The question

Does replacing the stage-2 head's deterministic MSE objective with FGN's
distribution head — a shared noise vector ε ~ N(0,1)^k through conditional
layer-norms, trained with the fair CRPS estimator at N=2 — buy, at NFE=1:

1. an **ensemble-mean rolled skill** at least matching the hand-dosed
   input-noise champion (znoise × xl144/xl233 ≈ 0.7235/0.7240 corridor AUC,
   two-seed pairs) — i.e. does the *learned* perturbation subsume the
   *hand-tuned* one;
2. **calibrated spread** — spread–error ≈ 1, fair CRPS and dip-event Brier
   better than the deterministic znoise head treated as a degenerate
   ensemble;
3. a working **dispersion instrument** for the replay question (§replay's
   un-bought falsifier) as a by-product.

Two technical facts motivate the design and are worth restating. Under a
pure MSE loss a noise-conditioned head learns to *ignore* ε (the
conditional mean is optimal) — so noise conditioning and the CRPS objective
are **one change, not two**. And the monthly znoise dose measurably does not
transfer across cadence (σ=0.7 relative dose → 0.8145 at pentad, A4);
FGN's perturbation is learned per regime by the objective itself, which is
the transfer story E-057.4 eventually tests.

## The arms

**E-057.0 · implementation + tests (torch, no GPU).** In `temporal.py`'s
stencil head: per-sample ε ~ N(0,1)^k (default k=32, per FGN), embedded by
a small MLP into per-layer LayerNorm scale/shift offsets (adaLN-zero
style), **zero-init so the ε-path is the identity at init** — at step 0 the
model IS the deterministic incumbent, bitwise (the E-057 twin of "r_fore
reads exactly 1.000000 at step 1"). Fair CRPS loss in z-space at N=2 (two
forwards per batch, ε¹ ≠ ε², shared everything else); ε drawn from a seeded
per-run stream, never global RNG. Reuse `ml/probscore.py` definitions;
the training loss is a torch mirror of its fair-CRPS estimator and must be
pinned against it numerically in tests. Test suite, exact where possible:
(i) fair CRPS at M=1 == MAE; identical members ⇒ the |x−y| term only;
torch loss == probscore on shared arrays to float precision. (ii) zero-init
identity: ε-conditioned forward == incumbent forward, `torch.equal`.
(iii) seeded ε reproducibility; member m of an M-member call == the M=1
call at the derived seed. (iv) **the shared-coin toy, per-pixel edition**:
x_{t+1} = x_t ± PATTERN with one coin for the whole field — a shared-ε
factorized head must reach |field-mean residual sign| ≈ 1 per member (the
FGN existence claim in our own harness), against a per-pixel-independent-ε
control pinned at the factorized floor. (v) checkpoint/resume bitwise.
(vi) ε-collapse telemetry: per-eval member variance logged; a member
variance sliding to 0 is the failure mode's signature and must be visible
in the run artefacts, not discovered post hoc.

**E-057.1 · the primary pair (2 seeds — head numbers are never quoted from
one).** Monthly cadence, xl144 configuration exactly (1024×16,
sunflower-144, K=24, 200k steps, same schedule), **input-znoise OFF** —
pre-registered hypothesis: the learned perturbation subsumes the input
corruption. Rolled at M=16 members, ε resampled each rolled step within a
member (FGN's convention), member seed fixed per trajectory. Read-outs:
ensemble-mean corridor AUC against the clean pair (0.6781) and the znoise
pair (0.7235); fair CRPS, spread–error, dip-Brier via `probscore.py`
against the znoise arm as degenerate ensemble; per-horizon table (the
znoise signature was −0.006 at h=1 growing to +0.074 at h=12 — the FGN arm
should reproduce the shape with a calibrated spread on top). The frozen
gate must reproduce 0.643 as always; the gate stays deterministic and
untouched.

**E-057.2 · the composition arm (1 seed, after the pair reads).**
ε-CRPS **plus** input-znoise 0.7: does explicit input corruption still add
anything once the perturbation is learned? Registered expectation: no —
and if it does, that is evidence the learned perturbation is not reaching
the input-error scale and k or the conditioning depth is binding.

**E-057.3 · the replay dispersion read (eval-only, ~$0).** Roll M=16
members from the six context ends of §replay (2004-12 … 2024-03).
Genuine dynamics: dispersion grows with lead and peak placement varies
with the initial state. Replay: flat dispersion, calendar-pinned peaks
(November 2027 et seq.) in every member. **Mandatory before any E-057
rolled number is quoted as a forecast** — this is the instrument §replay
declined to buy, and it comes with the head for free.

**E-057.4 · the cadence-transfer arm (LATER; gated on the E-054a K=144
roll verdict).** The same head at pentad K=144. This is the
self-calibration test: the hand dose demonstrably does not transfer across
cadence; a learned one should. Not designed further here — it inherits
whatever the pentad roll verdict says about that substrate.

## Falsifiers (pre-registered)

- **F1 (skill):** ensemble-mean corridor AUC of the E-057.1 pair below the
  znoise pair (0.7235-class) by more than the tier's paired seed band ⇒
  the FGN move fails to replace hand-dosed noise on the incumbent
  scoreboard, and the case for it rests on calibration alone.
- **F2 (distribution):** fair CRPS / dip-Brier not better than the
  deterministic znoise head as a degenerate ensemble, or spread–error far
  from 1, or member variance → 0 (ε ignored / collapse) ⇒ the distribution
  head is not earning anything at k=32; a k sweep (8/32/128, FGN's own
  open hyperparameter) is the registered follow-up, not a rescue of the
  headline claim.
- **F3 (replay):** E-057.3 shows flat dispersion with calendar-pinned
  members ⇒ every E-057 rolled number is reported as replay tracking,
  never as forecast — exactly as the archive's long-r numbers now are.

## Costs and controls

Training cost ≈ 2× a standard xl144 200k arm per seed (N=2 forwards);
rolling cost ≈ M× one deterministic roll per evaluation (NFE=1 — the whole
point). Controls all exist and are two-seed: clean xl144 0.6781, znoise
xl144 0.7235 (Table "twobytwo"). No codec change, no evaluator change
beyond additional probscore keys, no touch to `read_sv`, the gate, or any
archived number. Dispatch waits on box availability after the current
queue; its §0d entry is written at dispatch, as always.

# E-052 · The field head: joint next-field prediction, deterministic then generative

**Status: BUILT AND CPU-TESTED (2026-08-25/26 overnight session) — no GPU arm
dispatched yet.** Chris approved the staged path in the AR-vs-diffusion deck
(2026-08-25: *"Sounds good, let's try this. Take things step by step ...
please continue and thoroughly test the diffusion implementation"*) and this
plan is its experiment form. The deck is the design argument:
[One Step Forward: AR vs Diffusion](https://blauewelt.github.io/earth/ml/figures/ar_vs_diffusion.html).

## The question

Stage 2 today predicts each pixel's next embedding independently: a 145-slot
stencil is concatenated per frame by one fixed `Linear`, attention runs over
time only, and the t+1 field is 84,405 conditionally-independent point
estimates. "Move to diffusion" conflates three separable axes:

| axis | today | candidate |
|---|---|---|
| A · output scope | per-pixel, factorized | joint field, all points at once |
| B · output type | deterministic conditional mean | sample from the conditional distribution |
| C · computation | one-shot map | iterative refinement (NFE) |

E-052 buys them separately, in order, each with its own falsifier — because a
three-axis move in one dispatch would not say which axis paid, whatever it
returned.

## The arms

**E-052.0 · the probabilistic scoreboard (code only, no GPU).**
`ml/probscore.py`: fair-CRPS ensemble estimator (+ Gaussian closed form as its
own test), ensemble-mean MSE/ratio, spread–error ratio with the (M+1)/M
correction, and a dip-event Brier/BSS on threshold events. NEW keys beside the
existing read-outs; `read_sv`, the gate, and every archived number untouched.
Without this, squared-error scoring decides the whole question by itself:
E‖x−x̂‖² for a single sample exceeds the conditional-mean error by exactly the
conditional variance, i.e. a sampling head is penalized most at pentad/daily,
precisely where its advantage would live.

**E-052.1 · the clean axis-A ablation: joint deterministic field head.**
`ml/field_model.py` in `det` mode: ocean-patch tokens (4×4 default over the
0.25° window; ~5.3k tokens), per-token temporal conditioner over K frames,
DiT-style blocks attending over SPACE, one-shot regression of the next-field
RESIDUAL (x̂_{t+1} = x_t + f(·), so zero-init f ⇒ exact persistence at step 0).
Scored on today's scoreboard (one-step ratio vs its own persistence baseline)
against the stencil head at matched cadence, span and (approximately) params.
**Falsifier:** if joint spatial attention ≈ per-pixel stencil-concat at
matched conditions, axis A buys nothing here — the 145-slot concat was never
the bottleneck — and the diffusion case must rest on axis B alone.

**E-052.2 · the generative head on the same backbone (axis B, small C).**
Same tokenizer/conditioner/backbone in `diff` mode: EDM parameterization
(c_skip/c_out/c_in on the residual, σ ~ lognormal, λ(σ) weighting), seeded
deterministic Heun sampler on a Karras σ-ladder, M-member ensembles.
**Read:** ensemble-mean ratio must ≈ E-052.1 (no MSE tax after averaging);
CRPS, dip Brier and spread–error must beat E-052.1 treated as a degenerate
(M=1, zero-spread) ensemble. **Falsifier:** if the ensemble mean matches but
CRPS/Brier do not improve on the deterministic head + trivial noise ensemble,
the distribution head is not earning its NFE.

**E-052.p · the cheapest mechanism probe (zero new architecture, listed for
completeness).** Condition the EXISTING stencil head on a noise-level
embedding and train across a σ-ladder (the +0.057 result trains at ONE level).
Not built tonight; it touches temporal.py and deserves its own small diff.

## What is built, and how it is tested

Three new modules, no existing file changed:

- `ml/probscore.py` — the scoreboard (numpy, NaN-aware).
- `ml/field_model.py` — `OceanTokenizer` (pixel↔patch-token scatter/gather
  with ocean-mask channel; exact round-trip), `TemporalCond` (per-token
  encoder over K frames + season features), `FieldDiT` (learned 2-D pos-emb,
  adaLN-zero global conditioning on σ, per-token conditioning added at the
  embedding; zero-init final layer), EDM wrapper + Heun sampler (seeded
  `torch.Generator`, no global RNG use).
- `ml/train_field.py` — trainer for both modes: `--toy shift|bimodal|gauss`
  synthetic laws for CPU science, `--z-cache Z.npy --data tensor.npz` for the
  real substrate (same [T,P,d_z] embed-cache artefact stage 2 already
  publishes); per-year val split; incremental result JSON written atomically
  with a top-level `in_progress` key (ml/CLAUDE.md §5.25) and removed only at
  a completed end; full checkpoint (model/opt/sched/RNG) resume.

The test suite (`tests/test_probscore.py`, `tests/test_field_diffusion.py`)
holds EXACT identities where possible (§4.9), not thresholds:

1. CRPS of a large Gaussian ensemble ↔ the closed form; fair-estimator
   M-invariance; CRPS at M=1 = MAE exactly.
2. MSE_sample = MSE_mean + Var, measured on synthetic ensembles to float
   precision — the slide-4 identity that motivates E-052.0.
3. Tokenizer round-trip is the identity on ocean pixels; land cells never
   leak into the loss (masked-loss invariance to land values).
4. **Deterministic mode at init IS persistence: ratio 1.000000 at step 0** —
   the E-052.1 twin of "r_fore must read exactly 1.0 at step 1".
5. **Diffusion mode at init: x̂₀ = c_skip(σ)·x_noisy exactly** (zero-init
   final layer through the EDM skip), and the σ→0 limit returns the input.
6. Sampler determinism: same seed ⇒ bitwise-identical samples; different
   seed ⇒ different; M members from one call ≡ M calls with derived seeds.
7. Checkpoint round-trip: save → load → the next training step is
   bit-identical to the uninterrupted run.
8. Overfit smokes: `det` drives a tiny deterministic toy's ratio « 1;
   `diff` on a known Gaussian law reproduces its conditional spread.
9. The result file is atomic (never a half-written JSON) and carries
   `in_progress` until the run completes.

## The toy-science demonstration (run tonight, numbers in EXPERIMENTS.md)

Two synthetic laws, chosen so each axis has a microcosm:

- **`shift`** — x_{t+1} = one-cell eastward roll of x_t (+ small noise): a
  purely SPATIAL law. A per-pixel head with no neighbours cannot beat
  persistence here by construction; the field head must. Axis A's microcosm.
- **`bimodal`** — x_{t+1} = x_t + s·pattern, s = ±1 coin-flip shared by the
  whole field: conditional mean = persistence (the blur), truth is always
  sharp, and the two futures are field-COHERENT. The deterministic head's
  best possible ratio is ~1.0; a working diffusion head samples the two modes
  with |field-mean residual sign| ≈ 1 per member (a factorized sampler would
  give ≈ 0 — this statistic is the joint-law detector), wins CRPS, and its
  ensemble mean recovers ≈ the deterministic optimum. Axis B's microcosm.

## What E-052 does NOT claim or change

- No workflow wiring yet: `ml-train.yml` sits at the 25-input ceiling and a
  new trainer needs its own dispatch step — a reviewed change, with the
  Workflows PAT permission, not an overnight one. Until then the field
  trainer runs only by hand or under a future window-mode key.
- The conditioner inherits the E-045 verdict unchanged: skill lives in the
  720-day span, so the real-data arms must carry K equivalent to A2a/E-045.1
  spans, and the first pentad arm buys its own seed pair (§3b: new tier).
- Nothing here touches the collapse guard, `read_sv`, the gate, or any
  pooled/unpooled read-out; the new metrics are additional keys only.

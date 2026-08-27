# E-052 addendum · FGN (arXiv:2506.10772) read against the AR-vs-diffusion decomposition

**2026-08-27.** Analysis addendum to
[the deck](https://blauewelt.github.io/earth/ml/figures/ar_vs_diffusion.html)
and [the E-052 plan](E052_field_diffusion.md), written after reading FGN —
Alet, Price, El-Kadi et al., *"Skillful joint probabilistic weather
forecasting from marginals"* (GDM, June 2025), the successor to GenCast.
Chris read this analysis and approved its consequence the same day
(*"let's prioritize an experiment with: noise-conditioned stage-2 head
trained with fair CRPS"*) — that experiment is registered as
[E-057](E057_fgn_head.md). Nothing here changes E-052.1/1b, which are in
flight with their falsifier as registered.

## What FGN is

FGN replaces GenCast's diffusion sampler with **one forward pass per
member**: a global noise vector ε ~ N(0,1)^32 is fed into all of the
network's conditional layer-norm layers, and the model is trained with the
**fair CRPS estimator at N=2 samples, on per-location marginals only** —
no joint objective anywhere. It beats GenCast across the board, *including
the joint metrics*: +6.5% average marginal CRPS (up to 18%), +8.7%/+7.5%
average avg-/max-pooled CRPS, better cross-variable CRPS (10m wind speed,
z300−z500), ≈24 h tropical-cyclone track advantage, spread–skill ≈ 1 at all
leads. Epistemic uncertainty is a separate axis: J=4 independently trained
seeds, members drawn equally from each, model seed fixed per trajectory,
ε resampled **each timestep**. AR rollout fine-tuning (≤8 steps) is
"helpful, but not essential". Their mechanism claim: with only 32 shared
degrees of freedom across the whole output field, "the easiest way for the
model to jointly optimize the CRPS of all marginals is to try to model
their inter-dependencies as well."

## The interpretation, on the deck's own axes

FGN is the closest existing thing to the deck's A/B/C decomposition run at
planetary scale by the group that built the diffusion incumbent, and its
answer is: **axis B (sample vs. conditional mean) carried the value; axis C
(iterative refinement / NFE) did not.** "Diffusion vs AR" partly dissolves —
FGN *is* an AR head (autoregressive in time, one-shot per step) that is also
generative and joint.

- **Axis A** — FGN keeps a field-level backbone, so it is consistent with
  the field-head bet but is not an ablation of it. E-052.1's falsifier
  stands untouched.
- **Axis B** — the joint law does not need a joint objective or a sampler:
  marginal fair CRPS + a low-dim shared-noise bottleneck recovered joint
  structure well enough to beat full diffusion on pooled metrics. Slide 4's
  scoreboard argument is FGN's own *training loss*: `ml/probscore.py`'s
  fair-CRPS estimator is literally their objective (E-052.0 built the loss
  without knowing it).
- **Axis C** — the costly axis (NFE × M passes per rolled step) is the one
  FGN discards. Caveat: FGN-vs-GenCast is not a controlled ablation of C
  (~180M/seed vs ~57M total, 6-h vs 12-h step, more compute), so this is a
  strong direction, not a theorem — but their stripped-down ablation "still
  outperforms the prior state-of-the-art almost across the board."

## One amendment to the toy record

The bimodal head-to-head (E-052(b)) describes field-coherent sampling as
"the property no per-pixel head can have at any capacity." That is true for
the incumbent deterministic head and for *independently*-noised per-pixel
samplers (the 0.031 factorized floor), but **not** for a per-pixel head
conditioned on a *shared* ε: every pixel can map the same ε to a consistent
sign, so a shared-ε factorized head can pass the sign-coherence detector in
principle. FGN is the existence proof that joint structure rides on shared
low-dim noise under a marginal loss. Consequence: **axis B is testable
inside the incumbent stencil architecture** — the cell the decomposition
wanted and assumed it couldn't have. That cell is E-057.

## Proposed consequences for the E-052 arms (for the owning session to
adopt at dispatch; recorded here, no existing plan text edited)

1. **E-052.2 gains an FGN-mode arm and a sharper falsifier.** The FieldDiT
   already has the conditioning pathway (adaLN-zero on σ); FGN-mode swaps
   the σ-embedding for ε ~ N(0,1)^32, trains with fair CRPS at N=2
   (training cost ≈ EDM's), and samples at **NFE=1**. The registered
   falsifier "the distribution head must earn its NFE" becomes two-sided:
   **EDM must beat FGN-mode — not just the deterministic head — to justify
   NFE > 1.** If FGN-mode ties EDM on CRPS / dip-Brier / spread–error, we
   get ensembles at deterministic-head rollout cost (a ~29× factor on every
   73-step pentad roll × M members).
2. **E-052.p is superseded by E-057** (same idea, upgraded from σ-ladder
   conditioning to the full FGN move: shared ε + fair CRPS training).
3. The resulting 2×2 is completable and clean: {factorized, field} ×
   {deterministic, ε-sampling}, all at NFE=1 — E-052.1 (field, det),
   E-057 (factorized, ε), E-052.2-FGN (field, ε), incumbent (factorized,
   det) — with EDM as the lone axis-C arm on top.
4. **Gate discipline survives axis B**: a fixed (seed, ε-set) at NFE=1 is
   exactly as reproducible as the deterministic head, which defuses most of
   the deck's case-against item 3 (it genuinely bites only for a
   stochastic-σ multi-step sampler).

## What FGN does not settle here

The conditioner/span problem (FGN conditions on two frames of a *dense*
atmosphere; the E-045 720-day verdict is about an observationally sparse
substrate and transfers unchanged to any head). Replay (sampling reveals it
via non-growing dispersion but does not cure it; spread–skill ≈ 1 is the
shape of a healthy answer, and the dispersion falsifier becomes essentially
free — it stays mandatory before any rolled number is quoted as forecast).
Missingness (joint-from-marginals is proven on dense reanalysis; whether it
survives 83–97% missing-token Argo channels is exactly what the real-data
arms measure). And genuinely multimodal conditionals — the one place
iterative refinement could still earn its NFE, and now the entire remaining
falsifiable content of "diffusion vs AR".

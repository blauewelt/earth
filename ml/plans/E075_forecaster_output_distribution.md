# E-075 · What shape should the forecaster's output be? — continuous flow matching, discrete autoregression, or discrete marginals under shared noise

**Written 2026-09-05**, from Chris's question of the same afternoon: *"There
is the alternative path, so two alternatives: continuous embeddings → MSE loss
→ blockwise flow matching / hierarchical channel quantization → cross-entropy
loss → autoregressive prediction? can you put these two side by side
considering our rather ambitious el-nino or amoc or … prediction goals?"*

**Reviewed 2026-09-05 (Fable) against the record, after being drafted under
Opus 5 the same afternoon: two factual corrections, both marked in place —
the token-versus-continuous forecast comparison in §4 (the draft cited a
retired number and had the sign wrong) and the external-precedent sentence in
§3 (FGN and AIFS-CRPS are one-pass heads, not flow matching). The review also
surfaced an unharvested result, #534, now read into §4.**

**This document is written to be handed to an agent or a person with no prior
context.** Every abbreviation is expanded on first use and there is a glossary
in §9. Nothing here is dispatched. Every number is either measured in this
programme and cited to the log entry that measured it, or taken from a named
paper, or arithmetic shown in place.

---

## 1 · The situation, for a reader starting cold

**What the programme is building.** A forward model of the Earth's surface
state. Encode what the observing system knows about each grid cell into a
compact vector (stage 1, "the codec"); predict that vector forward in time
(stage 2, "the head"); read any quantity of interest out of the predicted
state (transport of the Atlantic overturning circulation at 26.5° N, the El
Niño index, sea ice, surface colour). The read-outs are downstream of one
forecast, which is the whole point: predicting the state predicts everything
at once.

**The data.** Family 7, recipe identifier `f7l0`: the whole globe at 0.25°
(721 latitudes × 1,440 longitudes = 1,038,240 grid cells), five-day time steps
("pentads") from 1982 to the end of 2024 (3,142 of them), 54 channels in three
groups — 7 ocean channels at 0.25°, 15 atmosphere-and-land channels at 1°, and
32 ocean-interior channels at 1° that exist only once a month. About 13.8
billion observed values. Full description, channel by channel, with the
distribution of each:
[the family-7 data handover](https://blauewelt.github.io/earth/docs.html?f=docs/FAMILY7_DATA_HANDOVER.md).

**The defect this document is about.** Forecasts rolled forward from this
programme's heads are systematically *under-dispersed*: they lose amplitude
with lead time and converge toward a smooth average. Measured across the
width ladder, the ratio of forecast amplitude to observed amplitude is
0.54–0.78, and the mean-squared-skill-score against climatology reads −0.439
— but rises to **+0.019 on the identical rolled states** once the amplitude
alone is recalibrated
([E-062 §(h)](https://blauewelt.github.io/earth/docs.html?f=ml/EXPERIMENTS.md#e-062)).
In other words, the loss is arithmetic, not representational: the model knows
roughly the right *pattern* and states it too quietly. That is the classic
signature of a model trained to emit a conditional mean, and it is the single
most expensive recurring problem in the programme.

**Why it matters more for these targets than for weather.** A ten-day weather
forecast is scored at short lead where the conditional distribution is narrow.
El Niño and the Atlantic overturning are scored at six to twenty-four months,
where the conditional distribution is wide and quite possibly two-humped
("an El Niño develops, or it does not"). A model that reports the average of
two futures reports a future that will not happen.

---

## 2 · The right way to frame the choice — three axes, not two paths

This programme already decomposed "autoregression versus diffusion" into three
independent axes, in
[the E-052 addendum](https://blauewelt.github.io/earth/docs.html?f=ml/plans/E052_FGN_addendum.md)
and [the deck it annotates](https://blauewelt.github.io/earth/ml/figures/ar_vs_diffusion.html):

- **Axis A — factorized or field-level backbone.** Does the network predict
  each cell from a local neighbourhood, or does it process the whole field
  jointly?
- **Axis B — sample or conditional mean.** Does the model emit a *draw from a
  distribution*, or a single best guess? This is the axis that fixes the
  amplitude defect of §1.
- **Axis C — iterative refinement.** Does producing one forecast step require
  one pass through the network, or many (twenty to fifty), each sharpening the
  last? Diffusion models and flow matching are on the "many" side. The cost
  measure is **NFE**, "number of function evaluations" — how many times the
  network runs per forecast step.

**The recorded verdict, from DeepMind's own comparison: axis B carried the
value; axis C did not.** Their FGN model ("functional generative networks" —
Alet, Price, El-Kadi et al., *Skillful joint probabilistic weather forecasting
from marginals*, arXiv 2506.10772, June 2025) replaced the diffusion sampler
of its predecessor GenCast with **one forward pass per ensemble member** and
beat it across the board, including on the joint metrics diffusion was
supposed to own: +6.5 % average marginal CRPS (up to 18 %), better pooled and
cross-variable scores, roughly a 24-hour advantage on tropical-cyclone tracks,
and a spread-to-skill ratio near 1 at all lead times. Caveat recorded in the
addendum and repeated here: FGN-versus-GenCast is **not** a controlled
ablation — different parameter counts, different time steps, more compute — so
this is a strong direction, not a theorem.

The addendum's closing sentence is the one that decides how to read Chris's
question: **genuinely multimodal conditionals are "the one place iterative
refinement could still earn its NFE, and now the entire remaining falsifiable
content of 'diffusion vs AR'."**

**The framing to discard.** The question is *not* "mean-squared error versus
cross-entropy". Flow matching's training loss *is* a squared error, but it
regresses a velocity field at a randomly chosen interpolation time, and the
sample it produces is a draw from the learned law — **not** a conditional
mean. Flow matching does not suffer the collapse of §1. Both of Chris's paths
buy axis B. They differ only in **how a multi-humped conditional distribution
is represented**, and that is the entire fork.

---

## 3 · Path A — continuous embeddings, flow matching

**What it is.** Keep the state continuous. Train a network to carry a sample
from pure noise to a sample from the conditional distribution of the next
state, by learning the velocity field of that transport. Producing a forecast
step means numerically integrating that velocity field, which costs twenty to
fifty network evaluations. "Blockwise" means a whole block of the field (or a
block of time steps) is produced in one such integration rather than one cell
at a time.

**Strengths.**
- Unlimited precision: the output is a real number, with no bins.
- Naturally parallel over space — a single integration produces all 1,038,240
  cells, so field size is not a sequence-length problem.
- External precedent for *continuous probabilistic* weather heads is strong,
  but be precise about which axis it supports: GenCast is the direct precedent
  for iterative refinement (a diffusion sampler); FGN and ECMWF's CRPS-trained
  AIFS ensemble are **one-pass** heads, so they are precedent for axis B (a
  continuous *sampled* output) and explicitly *not* for axis C. An earlier
  draft of this document listed all three as "in this family"; only GenCast
  is.
- Represents arbitrary distributions, including multi-humped ones, without
  choosing a vocabulary.

**Weaknesses.**
- **Cost is multiplied by rollout length.** A one-year forecast at five-day
  steps is 73 steps; with 8 ensemble members and 30 evaluations per step that
  is **17,520 network evaluations per forecast**, against 584 for a one-pass
  head. Two years doubles it. This is the dominant practical objection for
  seasonal-to-interannual targets, and it does not apply to a ten-day weather
  forecast, which is why the weather literature tolerates it.
- **The rollout can leave the data manifold.** Nothing constrains an
  intermediate state to be a physically realisable one. This programme has
  measured exactly that failure: an embedding collapse in run #387 that sat
  hidden behind a healthy-looking reconstruction loss for nine hours, and a
  250× expansion that the collapse guard rendered in grey because it was
  written for contraction only.
- **The training loss is a poor cross-model instrument.** Its value depends on
  the noise schedule and the interpolation-time distribution, so two
  configurations' losses are not directly comparable.

---

## 4 · Path B — hierarchical channel quantization, cross-entropy, autoregression

**What it is.** Give every channel its own monotone "warp" fitted to that
channel's own training distribution, then read off the digits of the warped
value in a small base: a coarse digit and one or two fine digits. Predict
those digits as categories with a cross-entropy loss, autoregressively. The
warp scheme and its theory are the subject of its own plan,
[E-074](https://blauewelt.github.io/earth/docs.html?f=ml/plans/E074_hierarchical_channel_quantization.md);
the essentials are that the level placement follows point density `p(x)^α`
(α = 1 gives equal-probability bins, α = 1/3 is the placement that minimises
squared error, α = 0 is uniform), that only the coarse digit's α has to be
chosen because the residual inside a bin is asymptotically uniform, and that
the two unbounded end bins are the one exception.

**Strengths.**
- **Explicit multi-humped distributions at one network pass.** A probability
  per bin *is* a multi-humped density. This is precisely the capability the
  E-052 addendum says was the last thing holding axis C up — and it is
  obtained here without paying axis C's cost.
- **Rollouts cannot leave the data manifold.** Every predicted token
  corresponds to a real bin of a real channel. Contrast the failures in §3.
- **A self-balancing multi-channel loss.** With K equal-probability bins every
  channel's loss is bounded by `log K` nats, so 54 channels of wildly different
  predictability weight themselves, with no hand-tuned per-channel weights and
  no learned uncertainty parameter that can "explain away" a hard target.
- **An interpretable dense metric.** Held-out cross-entropy in nats is
  comparable across every configuration and is computed over ~10⁹ targets, so
  its sampling error is negligible. This matters here more than usual: the
  programme's read-out instruments are noisy — the corridor score reproduces
  across seeds to 0.0020 but the transport-band correlations spread up to
  0.119, and the RAPID probe's three-seed spread is 0.245.

**Weaknesses.**
- **Precision is capped by bin width.** Root-mean-square error cannot go below
  `Δ/√12` for a bin of width Δ. Fine digits buy it back; the floor must be
  computed per channel up front so nobody mistakes it for model error.
- **Quantization has a measured RECONSTRUCTION cost in this programme — and,
  on the one clean comparison that exists, no forecasting cost.** Compressing
  the bottleneck to a 16-bit code cost about 18 % of reconstruction loss
  (0.270 against the continuous parent's 0.229, E-050 / run #485). The
  forecasting side is the opposite of what an earlier draft of this document
  said. The often-quoted 0.539-versus-0.506 (E-056a, run #504) is a
  **retired** number: it was trained under the contaminated endpoint pool, and
  #504 ran at ~5.8× the intended noise dose besides — the log itself marks it
  "not a verdict"; its dose-matched twin #507 read 0.510, a tie. The clean-pool
  pair is E-064: **#528 (the 7.6 M head on the 16-bit token)** reached a
  held-out one-step ratio of **0.564 at its minimum and 0.669 at 20 k**, while
  **#534 (the identical head on the continuous d_z-32 embedding)** reached
  **0.617 and 0.696**, each against its own persistence, dose-matched at 0.151
  relative noise, n = 1 each (`run-528.jsonl`, `run-534.jsonl` on
  `ml-metrics`; #534's curve is complete but its probe ladder never ran — the
  archive holds no `temporal.json` — so this is the one-step read-out only, and
  it was **unharvested in the log until this review, 2026-09-05**). By E-064's
  pre-registered criterion — within 0.02 means the token is a competitive
  substrate — the token is not merely competitive, it is ahead. Two cautions
  travel with that: a ratio against one's own persistence in a 6-dimensional
  quantized space is not the same quantity as one in a 32-dimensional
  continuous space, and one seed each is a direction, never a level (§3b of
  `ml/CLAUDE.md`). These measurements are on *bottleneck* quantization, not
  per-channel input quantization, so they bound the risk rather than settle
  it — but the sign of the bound is favourable, not unfavourable.
- **The sequence-length problem, which is the serious one.** See §5.
- **Essentially no external precedent** for token-autoregression on a global
  geophysical field.

---

## 5 · Path B's real obstacle, and why it damages the argument for Path B

Family 7 is 1,038,240 cells × 54 channels. Per-channel digits are on the order
of **10⁸ tokens per time step**. Strict autoregression — predicting tokens one
after another, each conditioned on all the previous — is not possible at that
length. Path B must therefore do one of three things, and only the third keeps
what made it attractive:

1. **Compress spatially first, then quantize the embedding.** But then the
   digits sit on learned coordinates, not on physical channels, and the
   per-channel-distribution argument — the strongest part of the idea —
   evaporates. This is also the road already measured (§4): a reconstruction
   tax of ~18 %, and — on the one clean pair — a one-step forecast that is
   not worse, so the cost of this option is the lost argument, not a lost
   number.
2. **Decode in parallel with masking** (predict a subset of tokens, condition
   on them, predict more — the MaskGIT / discrete-diffusion pattern). This is
   iterative refinement in discrete clothing: it is axis C again, with axis C's
   cost.
3. **Predict every cell's distribution independently in one pass, and supply
   the spatial coherence some other way.** This is §6, and it is the proposal.

---

## 6 · The proposal — per-cell categorical marginals under a shared low-dimensional noise vector ("discrete FGN")

### 6.1 · The mechanism, in plain terms

For every grid cell and every channel, the network outputs **a probability for
each value-bin** — "this cell has a 12 % chance of landing in bin 40, 31 % in
bin 41, …". All of them are produced in **one single run of the network**,
with no refinement passes and no cell waiting on another cell's answer.

On its own that would treat the cells as independent, which would produce
speckle rather than a coherent warm anomaly. What prevents it: **one small
random draw — 32 numbers, drawn once for the entire planet — is fed into every
layer of the network**, so all the cells' predicted distributions shift
together in a physically consistent way. Draw a different 32 numbers and you
get a different, equally coherent, whole-Earth forecast. That is one ensemble
member.

This is FGN's architecture with one substitution: the final layer emits a
probability per bin instead of a mean and a spread. It keeps the per-channel
digits of Path B and the "sample, do not average" property of Path A, at the
cost of one network pass.

### 6.2 · Why the shared noise is what makes it work

The counter-intuitive part, and FGN's actual finding: **you can train only on
each cell's own distribution and still get the cells to agree with one
another.** The mechanism is the bottleneck. With only 32 shared degrees of
freedom available to perturb a million-cell field, the cheapest way for the
network to improve every cell's individual score is to use those 32 numbers to
encode large-scale, physically shaped patterns of variation. Independent
per-cell randomness cannot do this. In this programme's own toy test the
difference is stark: sign-coherence 0.99 with a shared draw against 0.15 with
independent per-cell noise (E-057.0, verified on a processor with no GPU).

### 6.3 · What has to be decided or built

| piece | state |
|---|---|
| shared-noise plumbing (`--fgn-eps`, injecting one draw into every conditional layer-norm) | **BUILT and CPU-verified** in `ml/temporal.py` from E-057; zero-initialised so it is bitwise identical to the existing head when switched off |
| the fair-CRPS training loss (a proper score at two samples) | **BUILT** — `fair_crps2` in `ml/temporal.py`, mirroring `ml/probscore.py` |
| a categorical output head anywhere in the programme | **NOT BUILT.** There is no `cross_entropy` in `ml/temporal.py`. `ml/plans/E049_roadB_token.md` states token-output prediction "is not part of E-049", and the overview lists it as gated on an audit that never ran |
| the per-channel warps | **NOT BUILT** — E-074, and its first experiment needs no GPU |
| the training loss for a categorical head under shared noise | **AN OPEN DESIGN QUESTION** — see the risk below |

**The open design question, stated plainly so it is not glossed over.** FGN
trains with the fair continuous ranked probability score, which is defined for
real-valued predictions. A categorical head's natural loss is cross-entropy,
which is a *marginal* likelihood and — unlike the fair CRPS estimator at two
samples — does not obviously reward the shared noise for creating coherence.
Two candidate resolutions, both cheap to test on a toy before any real
training: (i) train the categorical head with the discrete ranked probability
score, which is the ordered-category analogue of CRPS and is a proper score
that a shared noise draw can improve; or (ii) keep cross-entropy but evaluate
it on *two* sampled noise draws in the fair-estimator arrangement. **Option
(i) is the principled one and is what I would build first**, because the bins
are ordered — bin 41 is genuinely closer to bin 40 than to bin 3 — and plain
cross-entropy throws that ordering away.

---

## 7 · Which path the read-out goals actually argue for

**Multimodality is more likely to be real at these lead times.** At six to
twenty-four months the conditional distribution of the El Niño index is
plausibly two-humped. That is exactly the cell the E-052 addendum leaves open,
and it argues for an *explicit* distribution over an implicit sampler.

**Rollout length multiplies axis C's cost and nothing else's.** 73 steps a
year, 146 for two. Whatever iterative refinement costs per step, it is paid
that many times per member. This is the strongest practical argument against
flow matching for seasonal-to-interannual targets specifically.

**The read-out cannot settle the architecture question, and this must be said
before anyone tries.** The RAPID array gives roughly 9 effective degrees of
freedom over 240 months; the record holds perhaps 10–15 independent El Niño
events. This programme has measured what that does to a comparison: three
seeds at one fixed configuration span 0.245 on the RAPID probe, while the
objective those same runs optimise reproduces to 0.0017. **No architecture
comparison will be resolved on an Atlantic-overturning or El Niño index.** It
must be settled on a dense field metric, with the index as a downstream check.
Both paths give a dense held-out loss; nats are the more interpretable of the
two.

**And neither is the binding constraint.** E-072's conclusion stands: at
today's data the model size is set by the data, and the width ladder was flat
from 7.6 M to 206.66 M parameters on the rolled field. With 13.8 billion
observed values both paths are over-parameterised. Since compute is available
at scale, **the data ladder matters more than this fork does.** Anyone picking
this up should hold that in view and not let an architecture debate consume the
budget that a bigger, better-designed dataset would use better.

---

## 8 · What AlphaEarth Foundations chose, and what transfers

AlphaEarth Foundations (Google DeepMind, arXiv 2507.22291, July 2025) is the
nearest large system to what E-072 proposes, so its choices are worth reading
against this fork. **The most important thing about it is what it does not
do: it does not forecast.** It summarises the observations over a stated time
window into an embedding you can then use for mapping tasks with very few
labels. Nothing rolls forward, so the amplitude problem of §1 never arises for
it, and it therefore has nothing direct to say about the output-distribution
question this document is about. What it does have is a considered answer to
three questions we also face.

**Heterogeneity: per-source learned encoders, not per-channel value
transforms.** Sources are normalised with global statistics in preprocessing,
and then *"individual source encoders transform inputs to the same latent
space before entering the bulk of the model."* Optical imagery, radar, lidar,
climate fields and geotagged text each get their own small encoder and meet
only after being embedded. This is the mainstream alternative to E-074's
input half: absorb the distributional differences in a learned per-source
network rather than in a fitted per-channel warp. It is cheaper to build and
carries no bins, and it does not give you the two properties E-074's warp
gives — a loss that balances itself across channels, and an interpretable
per-channel likelihood.

Chris's guess that they faced less channel heterogeneity than we do is half
right, and the half that is wrong is the instructive one. They face *more*
source diversity (imagery, radar, text) but *less* need for calibrated
per-channel predictions, because they never have to state a distribution over
a physical value at a long lead. Our 54 channels are fewer and duller, but we
must predict each one back in its own unit with an honest spread.

**Quantization: post-hoc, for storage, not a modelling device.** Embeddings
are 64-dimensional and are shipped as 64 bytes — the 32-bit floats are
quantized to 8 bits after training, *"resulting in a 4x reduction in storage
with negligible impact on performance."* That is a
shipping decision taken after training, not a representation decision taken
before it. Read against our fork: it is mild evidence that quantization is
not needed as a modelling device — but only mild, because they never forecast,
and the case for discrete outputs in §6 is entirely about the forecast.

**Two things they engineered that we should copy regardless of the fork.**
First, an explicit anti-collapse term: embeddings are compared with
batch-rotated versions of themselves by dot product and the absolute value of
that is minimised, so the representation cannot quietly contract onto a point.
This programme has been bitten by exactly that failure twice — run #387 sat at
a healthy-looking reconstruction loss for nine hours after its embedding
collapsed, and a later 250× *expansion* rendered in grey because the guard was
written for contraction only. A uniformity term is cheap and structural where
our probe-correlation guard is a tripwire that fires after the fact. Second, a
bounded embedding geometry: outputs are treated as the mean direction of a von
Mises–Fisher distribution — the natural bell-curve analogue on the surface of a
sphere — and decoding samples from it. Putting the embedding on a sphere makes
runaway scale impossible by construction rather than by threshold.

Their objective is also worth noting as a whole, because it is four terms and
not one: reconstruction of all source variables conditioned on time and sensor
metadata, the batch-uniformity term, a teacher–student consistency term, and a
contrastive term aligning embeddings with geotagged text.

## 9 · The decision procedure — two cheap measurements, not an argument

Do not pick the fork now. Each branch has an unharvested or free measurement in
front of it, and each collapses one branch.

**Step 1 — finish E-057 (costs about one box-day).** It is the direct test of
axis B at one forward pass, on the incumbent architecture, against measured
two-seed controls (clean 0.6781, hand-dosed noise 0.7235). It is built and
verified; its real-data runs died or were superseded and nothing is running
now. If a one-pass generative head lands near the diffusion arm, **axis C is
closed for this programme** and the whole fork shrinks to "Gaussian marginals
or categorical marginals under shared noise" — a small experiment rather than
an architecture bet.
[E-057 plan](https://blauewelt.github.io/earth/docs.html?f=ml/plans/E057_fgn_head.md)

**Step 2 — run E-074a (costs nothing, needs no GPU).** Fit the per-channel
warps on family 7's training years and score them as descriptions of held-out
data. Its conditional-entropy column answers the precondition for Path B:
whether a coarse digit is a predictable object across a five-day step or one
that flips on noise. Full read-outs and four pre-registered falsifiers in the
E-074 plan.

**Step 3, only if both pass — build the categorical head as one arm inside
whatever wave is running.** Same embeddings, same pool, same holdout scope as
runs #528 / #534 (the 7.6 M-parameter head on the 16-bit token and on the
continuous 32-dimensional embedding), changing only the output. Three arms:
continuous distributional head (the E-072 §4.2 default), single categorical
over 4,096 bins, two digits of 64 bins each. Read out the gradient statistics,
the one-step ratio, and — the number that matters — **the amplitude ratio on a
short roll, decoded by SAMPLING the digits, never by taking the most likely
one.** Taking the most likely bin is mean-collapse under a different name and
would reproduce exactly the defect the arm exists to test.

---

## 10 · Glossary

| term | plain English |
|---|---|
| **stage 1 / codec** | the encoder that compresses each location's observations into a short vector |
| **stage 2 / head** | the model that takes that vector and predicts the next one |
| **cell** | one grid point — one of family 7's 1,038,240 quarter-degree points |
| **channel** | one measured field, e.g. sea-surface temperature or eastward current |
| **pentad** | a fixed five-day time bin counted from 1982-01-01 |
| **categorical distribution** | a weighted-die distribution: a finite list of outcomes each with a probability, summing to 1 |
| **marginal** | the distribution for one cell on its own, ignoring its neighbours |
| **joint** | the distribution of all cells together, including how they co-vary |
| **coherence** | neighbours agreeing — a warm anomaly appearing as one blob rather than speckle |
| **forward pass** | one run of the network from input to output |
| **NFE** | "number of function evaluations": how many network runs one forecast step costs. 1 for a one-pass head, 20–50 for a diffusion or flow sampler |
| **shared low-dimensional noise vector**, written **ε** here and **z** in the FGN paper | one small list of random numbers (32) drawn per ensemble member and injected everywhere in the network. Note the collision: this programme also uses `z` for the codec's embedding. Rename before writing code |
| **head (categorical / Gaussian)** | the final layer. A Gaussian head emits a mean and a spread; a categorical head emits one probability per bin |
| **softmax** | the standard function turning raw network scores into probabilities that sum to 1; how a categorical head is built |
| **multimodal** | a distribution with more than one peak — two genuinely different possible outcomes rather than their blurred average |
| **flow matching** | train a network to carry noise to data by learning a velocity field; sample by integrating it, which costs many passes |
| **autoregressive** | predicting the next item conditioned on the previous ones, in sequence |
| **cross-entropy** | the standard training loss for a categorical prediction: the negative log-probability the model assigned to the value that actually occurred |
| **CRPS** | "continuous ranked probability score", a proper score for a predicted *distribution* against a single observed value. "Fair" refers to an estimator that is unbiased when computed from a small number of samples |
| **RPS / discrete RPS** | the ordered-category analogue of CRPS — the proper score to use when the bins have a natural order |
| **proper score** | a score minimised, in expectation, only by stating the true distribution — so it cannot be gamed by reporting a false confidence |
| **conditional mean** | the average of all possible futures. Minimises squared error, and is the reason forecasts go quiet |
| **amplitude ratio** | forecast variability divided by observed variability. 1 is honest, 0.6 is a forecast that has gone quiet |
| **MSSS** | "mean-squared skill score" — how much better than a reference (here, climatology) the forecast is, on squared error |
| **effective degrees of freedom** | how many genuinely independent observations a correlated series is worth. RAPID's 240 months are worth about 9 |
| **quantile / equal-probability bins** | bin edges placed so that every bin holds the same fraction of the data |
| **α (alpha)** | the exponent in point density ∝ `p(x)^α` that selects the bin placement: 1 equal-probability, 1/3 squared-error-optimal, 0 uniform |
| **nats** | units of information, like bits but base *e*. The natural unit for a cross-entropy |
| **on-manifold** | a predicted state that is a physically realisable one, rather than a point the decoder has never seen |
| **holdout scope / frozen protocol** | which years are excluded from training so the score is honest. Here: development years 2009 / 2017 / 2023, and everything after 2020 reserved for the final test |

---

## 11 · Prior art

The pieces are standard; the specific combination in §6 is not one a search
locates. Chronos bins time-series values uniformly, one layer, per value.
UniTok (June 2026) uses single-stage finite scalar quantization and explicitly
criticises Chronos's uniform binning without ever testing quantile bins against
it. Visual autoregressive models do coarse-to-fine, but over spatial scale
rather than value precision. Residual-vector-quantization audio codecs do
value-level coarse-to-fine with a shared vector codebook, and the coarse-to-fine
autoregressive stack over those tokens is the AudioLM / MusicLM structure —
which is what §6 becomes when the vector codebook is replaced by per-channel
scalars and the sequence by one-shot marginals. FGN supplies the shared-noise
mechanism but with a continuous head.

This is a search, not a proof of novelty.

- [Skillful joint probabilistic weather forecasting from marginals — FGN (arXiv 2506.10772)](https://arxiv.org/abs/2506.10772)
- [UniTok · Time Series as Language (arXiv 2606.09861)](https://arxiv.org/html/2606.09861v1)
- [Non-uniform quantizers, Lloyd–Max optimality and high-resolution theory (Stanford EE269)](http://web.stanford.edu/class/ee269/Lecture_nonuniform_quantization.pdf)
- [CARP · coarse-to-fine autoregressive prediction (ICCV 2025)](https://openaccess.thecvf.com/content/ICCV2025/papers/Gong_CARP_Visuomotor_Policy_Learning_via_Coarse-to-Fine_Autoregressive_Prediction_ICCV_2025_paper.pdf)
- [AlphaEarth Foundations (arXiv 2507.22291)](https://arxiv.org/html/2507.22291v1)

# E-074 · Hierarchical channel quantization — per-channel warped digits as the model's value representation

**Written 2026-09-05**, from Chris's question of the same morning: *"I have
this fixed idea, that mapping levels by distribution lumps in the channels
input data distribution … would be helpful"*, then *"a quantiled RVQ?"*,
then the correction that names it: *"I guess it's not RVQ because it's per
channel. So **hierarchical channel quantization** would be a good name"*, and
the purpose: *"(enabling autoregressive and yet precise prediction
capability)"*.

This is a PLAN. Nothing here is dispatched, and the first experiment in it
needs no GPU at all. Every measured number is cited to the log entry that
measured it; every other number is arithmetic shown in place.

Read with:
[E-072 · the Earth foundation model](https://blauewelt.github.io/earth/docs.html?f=ml/plans/E072_earth_foundation_model.md)
(§4.2 settles the objective — β-NLL — and this plan is the alternative to that
section's per-channel-group likelihood zoo, not a competitor to its β),
[E-070 · the family-7 build spec](https://blauewelt.github.io/earth/docs.html?f=ml/plans/E070_family7_build.md),
[the family-7 data handover](https://blauewelt.github.io/earth/docs.html?f=docs/FAMILY7_DATA_HANDOVER.md)
(the channel-by-channel distributions this plan is built on),
[E-048 · the FSQ block sweep](https://blauewelt.github.io/earth/docs.html?f=ml/plans/E048_fsq_block_sweep.md)
(the two level ladders that exist today, and why they are per z-dimension),
[the standing overview](https://blauewelt.github.io/earth/docs.html?f=ml/OVERVIEW.md).

---

## 0 · The one-paragraph answer

Give every input channel its own monotone **warp** fitted to that channel's
own training distribution, then read off the **digits** of the warped value in
a small base: a coarse digit that the forecaster rolls, and one or two fine
digits that restore precision. The warp is chosen from a one-parameter family
— point density proportional to `p(x)^α`, where α = 1 is equiprobable
(quantile) bins, α = 1/3 is the mean-squared-error-optimal placement, and
α = 0 is uniform. **α only has to be chosen once, for the coarse digit**: the
residual inside a bin is asymptotically uniform, so every later digit is
uniform by construction and the whole hierarchy collapses to base-K digits of
one warped scalar. The single exception is the two unbounded end bins, where
the residual is not uniform and the physics we care about lives. What this
buys is not capacity — nested exact quantiles are provably the same object as
one finer quantile layer — it is (i) a loss that balances itself across
heterogeneous channels without a hand-built likelihood per channel type,
(ii) a factorised softmax that makes an autoregressive value head tractable,
and (iii) a sampled fine digit, which is the cheapest attack this programme
has on its most expensive recurring defect, amplitude damping. The open
question, and the reason for the free experiment, is whether α = 1 is right
for the coarse digit: maximum entropy is not the same as maximum
predictability, and for a forecaster we want the second.

---

## 1 · The construction, exactly

### 1.1 · The warp

For channel `c` with training density `p_c`, define the point density of the
levels as `λ_c(x) ∝ p_c(x)^α`. A companding quantizer realises that by mapping

```
u = G_c(x),    G_c(x) = ∫_{-∞}^{x} p_c(t)^α dt  /  ∫ p_c^α        (so u ∈ [0,1])
```

and then quantizing `u` **uniformly**. Three values of α name themselves:

| α | name | what it optimises | classical statement |
|---|---|---|---|
| **1** | equiprobable / quantile — `G_c = F_c`, the CDF | entropy of the code; every bin equally likely | maximum-entropy quantizer |
| **1/3** | Panter–Dite | mean squared error at a fixed number of levels | `D ≈ (1/12N²)(∫ p^{1/3})³` |
| **0** | uniform | nothing about `p`; constant resolution in the unit | Gish–Pierce: uniform + entropy coding is within ~0.25 bit of the rate–distortion bound at high rate |

α = 0 is a serious contender rather than a straw man, because an
autoregressive transformer over the codes **is** an entropy model, which is
exactly the condition Gish–Pierce assumes. The case for α = 1 is therefore
not information-theoretic; it is two optimisation arguments — no rare classes
in a finite-sample softmax, and equal maximum entropy (`log K` nats) per
channel, which balances a multi-channel loss with no per-channel weights. §4
argues that a third consideration may beat both.

### 1.2 · The digits, and why α does not need a schedule

Quantize `u` in base `K`: the coarse digit is `d₁ = ⌊u·K₁⌋`, the next is
`d₂ = ⌊(u·K₁ − d₁)·K₂⌋`, and so on. That is the whole hierarchy — no second
fitted codebook, no per-bin bookkeeping.

**Why later digits need no α of their own.** Inside a layer-1 bin the
conditional density is `p_c` restricted and renormalised. High-resolution
theory's founding approximation is that `p_c` is locally flat across a narrow
bin, so the within-bin residual is approximately **uniform** — and for a
uniform distribution `p^α` is constant for every α, so all ladders coincide.
Uniform sub-division is not a compromise at digit 2; it is optimal by every
criterion at once. Quantization whitens, and one digit is enough to whiten.

**Where that fails, quantitatively.** The relative variation of `p_c` across a
bin is about `Δ_k · |p_c'| / p_c`, and `Δ_k ∝ 1/(K λ_c) = 1/(K p_c^α)`, so

```
relative variation across a bin  ∝  |p_c'| / (K · p_c^{1+α})
```

which **grows as `p_c → 0` and grows faster the larger α is.** So the
whitening argument is strongest at the mode and weakest in the tails, and
weakest of all for the quantile warp — precisely the combination that matters
here, because the top bin of an α = 1 ladder is `[q_{1−1/K}, ∞)`, unbounded,
and the strong-current core that carries Atlantic transport sits inside it.

**Therefore: α is per-BIN, not per-layer.** Digit 2 is uniform in every
interior bin and is placed by the conditional tail (α ≈ 1, or a fitted Pareto)
in the two end bins. One special case, no schedule.

### 1.3 · What this is not

It is not residual vector quantization. RVQ's mechanism is a *shared vector*
codebook over the residual, whose gain comes from correlations **between**
dimensions; per-channel scalar digits have no such correlation to exploit, and
importing the RVQ argument would import a benefit that is not present. It is
also not "layers" in the RVQ sense: with digit 2 uniform these are literally
the digits of one warped scalar. **Coarse digit / fine digits** describes the
mechanism; *hierarchical channel quantization* names the scheme.

Nor is it what `ml/fsq_ladder.py` builds today. That module fits a saturation
radius per **z-dimension** of the bottleneck (candidates 2σ, p90, p99, max, at
two scalings) and then places levels uniformly or exponentially inside it. Its
own docstring rules the present plan out of its scope: *"a per-input-channel
ladder would be a different object entirely (it would have to sit before the
encoder, on the input tokens)."* E-048's exponential ladder was an
approximation to companding; this is the principled version, one level
earlier in the system.

---

## 2 · Why family 7 in particular wants it

The tensor's 54 channels do not share a distributional shape, and the
handover records each one
([FAMILY7_DATA_HANDOVER.md](https://blauewelt.github.io/earth/docs.html?f=docs/FAMILY7_DATA_HANDOVER.md)
§3, all figures measured on the published bytes):

| channel | shape of its distribution | what a single Gaussian on z-scores does with it |
|---|---|---|
| `log_swe` (snow water equivalent) | huge mass at zero, mean 2.07, sd 3.57 | models the zero spike as a tail |
| `log_prate` (precipitation) | zero-inflated, mean 0.87, sd 0.77 | same |
| `sea_ice` | **NaN below 0.15**, then bounded in [0.15, 1] — 155,141 finite cells of 1,038,240 at bin 2411 | puts mass outside the support |
| `cur_speed` | non-negative, heavy right tail, mean 0.158, sd 0.159 | under-weights the tail that carries transport |
| `sp` (surface pressure) | mean 965 hPa, sd 95, left-skewed by plateaus | passable |
| `rg_s10 … rg_s1900` (salinity) | sd falls 1.06 → 0.13 with depth | sixteen channels of very different scale |

E-072 §4.2's answer is a **zoo**: Gaussian for continuous anomalies,
log-Gaussian for log-normal fields, a hurdle model (Bernoulli wet/dry ×
log-Gaussian amount) for zero-inflated precipitation, logit-Gaussian or Beta
for bounded fractions, Student-t where tails are heavy. That is correct and it
is six hand-chosen families with their own failure modes. **Per-channel warped
digits replace all six with one mechanism**: zero-inflation, bounded support,
heavy tails and multimodality are all absorbed by the empirical warp, and the
likelihood is a categorical whose maximum entropy is identical across
channels. The zoo becomes a fit.

---

## 3 · What it is FOR — three claims, each against a named defect

**(a) Loss balance without hand-tuned weights.** E-072 §4.2's reason (i) for
the negative log-likelihood is that channels differ in predictability by
orders of magnitude, so plain squared error on standardised targets is
dominated by the unpredictable ones unless weights are tuned; NLL learns the
weighting as σ. A categorical over K equiprobable bins gets the same balance
by construction — every channel's loss is bounded by `log K` nats — with no σ
head, and therefore none of the explain-away pathology §4.2 has to cure with
β-NLL, a σ floor and a warm start.

**(b) A tractable autoregressive value head — Chris's stated purpose.** 4,096
levels as one token is a 4,096-way softmax; as two digits of 64 it is two
64-way softmaxes, the second conditioned on the first, with a ~32× smaller
output matrix. The coarse digit is low-entropy and is what stage 2 rolls; the
fine digits restore precision for the read-out. **No token-output head has
ever been built in this programme** — `ml/plans/E049_roadB_token.md` states
that token-output autoregression "is not part of E-049", the overview lists it
as a downstream item gated on an audit that never ran, and there is no
`cross_entropy` anywhere in `ml/temporal.py`. Every quantization experiment to
date (E-046, E-048, E-049, E-050, E-056, E-064, E-065) quantised the
**input** substrate and left the output a regression.

**(c) Amplitude damping — the expensive one.** Rolled fields from this
programme are systematically under-dispersed: amplitude ratio 0.54–0.78 across
the width ladder, and an msss against climatology of −0.439 that becomes
**+0.019 under amplitude calibration alone**, on the identical rolled states
([E-062 §(h)](https://blauewelt.github.io/earth/docs.html?f=ml/EXPERIMENTS.md#e-062)).
That is the signature of propagating a conditional mean. A **sampled** fine
digit keeps the variance by construction. This is a third independent attack
on that defect beside E-057's fair CRPS and E-072 §4.2's distributional head,
and the cheapest of the three to test.

A fourth thing it is **not** needed for: gradient spikes. Re-measured
2026-09-05 on the archived curve of **#432 (E-044b seed 1 — the 206.5M pentad
head whose step-6,000 spike is the one on the record)**: of 100 logged windows
exactly one is anomalous (`grad_norm_max` 96,469 and `clip_frac` 0.117 at step
6,000), every other window sits between 2.8 and 32 with `clip_frac` 0.0000,
and the run converged normally to loss 0.99 at 200k. Seed 0 peaked at 452. A
128-clip already handles it at negligible cost; buying a discrete objective to
cure it would be curing a symptom that is already cured.

---

## 4 · The complication: maximum entropy is not maximum predictability

Bin width goes as `1/λ_c(x)`, so a fluctuation of fixed physical size δ crosses
about `δ · λ_c(x) · K` boundaries. At α = 1 the boundaries are densest exactly
at the mode — **the coarse digit flips fastest where the field spends most of
its time.** Marginal entropy is maximised, and the conditional entropy
`H(d₁(t+1) | d₁(t), context)` may be maximised with it. That is the opposite
of what the coarse digit is for in §3(b), where its job is to be the stable,
rollable, multi-day-predictable object.

So the criterion for α₁ is not "how much information does this digit carry"
but **"how much of it survives a pentad"**. The two pull against each other:
α = 0 gives a sluggish digit that is easy to predict and says little; α = 1
gives a maximally informative digit that may be mostly noise near the mode.
The optimum is interior.

**Pre-registered prediction, so it can be wrong: scored on conditional entropy
rather than marginal entropy, α₁ lands nearer 1/3 than 1** — which would be a
tidy outcome, since 1/3 is also the mean-squared-error-optimal placement. If
conditional entropy turns out flat in α, α₁ = 1 stands on the class-balance
argument of §1.1 and is taken.

---

## 5 · Inventory — what exists, what does not

| piece | state |
|---|---|
| per-z-dimension uniform / exponential ladders on the bottleneck | **BUILT** — `ml/fsq_ladder.py`, E-048; parity-pinned across torch/JAX/numpy |
| quantiles used to choose a saturation radius | **BUILT** — `scale_candidates`, same module |
| levels PLACED at quantiles | **not built** |
| any per-INPUT-channel quantizer | **not built** — `--input-quant` (E-044c) quantizes what the stage-2 head reads, i.e. the embedding again |
| hierarchical / multi-digit scalar codes | **not built** |
| a categorical (token-output) head anywhere | **not built** — no `cross_entropy` in `ml/temporal.py` |
| the family-7 channel distributions | **measured and documented** — the handover, §3 |

---

## 6 · The experiments

Ordered by cost. Each states what would refute it, per `ml/CLAUDE.md` §1.

### E-074a · The ladder bake-off, on the data alone — **zero GPU, one afternoon, $0**

*Absolute description:* fit warps on family 7's training years and score them
as descriptions of held-out data, with no model anywhere. `stage` data-only ·
`data` `family7_global025_pentad_l0` · `arch` none · `steps` none.

- **Fit** on training bins only (≤ 2020, with the development holdout years
  2009 / 2017 / 2023 excluded), from a stratified sample of ~200 pentads across
  the span, per channel, over finite values only.
- **Grid:** α ∈ {0, 1/3, 1/2, 1} × K₁ ∈ {16, 64, 256} × depth ∈ {1 digit,
  2 digits with K₂ = 64 and α₂ = 0} × tail exception {off, on}.
- **Read-outs, per channel:**
  1. **Held-out cross-entropy in nats.** Every scheme must be scored on a
     COMMON fine evaluation partition, or a discrete likelihood and a
     continuous density are not comparable and the number means nothing.
     Baselines on the same partition: a Gaussian on standardised anomalies,
     and E-072 §4.2's per-group family for that channel.
  2. **Coarse-digit persistence and conditional entropy** — `P(d₁(t+1) = d₁(t))`
     over one pentad and the empirical `H(d₁(t+1) | d₁(t))`. This is §4's
     instrument and probably the decisive column.
  3. **Reconstruction RMSE in the channel's own unit**, overall AND restricted
     to the top 1 % of `cur_speed` and to the RAPID 26.5° N section cells.
  4. **The quantization floor**, `Δ_k/√12` per bin, computed up front so that
     nobody later mistakes the floor for model error.
  5. Calibration: bin-wise reliability of the fitted marginal.
- **Falsifiers, pre-registered.**
  - If α = 1 does not beat α = 0 on held-out nats by more than the
    channel-to-channel spread, the quantile idea does not survive its own data
    and nothing downstream is bought.
  - If conditional entropy is flat in α (§4 void), α₁ = 1 is taken and the
    remaining questions are all downstream.
  - If α = 1's tail RMSE on `cur_speed` is worse than α = 1/3's by more than
    the α = 1/3 value itself, tails decide and α₁ = 1/3 is taken regardless of
    the nats column.
  - If the two-digit scheme's held-out nats differ from the one-digit scheme
    at matched `K₁·K₂` by more than 1 %, the whitening argument of §1.2 is
    wrong somewhere and the implementation is at fault, not the theory.

### E-074b · Input-side digits, as ONE ARM INSIDE PHASE E — **≈ $1 marginal**

*Absolute description:* the cone codec reading per-channel digit embeddings
instead of standardised continuous values, everything else identical to its
continuous twin. `stage` encoder · `data` family 7 · `arch` ConeMAE 7.05M ·
`resume` none.

Family 7 has no codec baseline of its own yet, so a standalone arm would have
to buy its own control (§3b: the first result at a new tensor buys its own
replication). **Do not buy a separate wave.** Phase E is already going to
train cone-codec seeds on family 7 against a continuous control; add the digit
arm there and it shares that control for the cost of one extra seed.

- **Control:** the same wave's continuous arm, and on family 4 r3 the measured
  E-069b pair (cone 0.630 / 0.659 and 0.579 / 0.587 against the raw-patch bar
  0.665 / 0.701).
- **Read-out:** held-out per-family MSE against `msebar` (anchor, future, dot
  families separately — the decomposition that diagnosed E-069), plus the
  velocity probe.
- **Falsifier:** if per-family held-out MSE is not better than the continuous
  arm's on at least the anchor and future families, input digitisation buys
  nothing and only the output side (E-074c) remains interesting.

### E-074c · The output head — the decisive one for the autoregressive claim — **≈ $1.5–3**

*Absolute description:* the 7.6M stage-2 head, same embeddings, same pool and
holdout scope as **#528 / #534 (E-064a/b — the 7.6M head on the 16-bit token
and on the continuous d_z-32 embedding)**, changing only the OUTPUT.
`params` 7.6M · `stage` stage-2 · `arch` 256×8, K 144, stencil 145 ·
`steps×batch` 20k × 256.

Three arms: (1) continuous β-NLL output, the E-072 §4.2 default; (2) a single
categorical over K = 4,096; (3) two digits, 64 × 64, at the α chosen by
E-074a.

- **Read-outs:** `grad_norm_max` and `clip_frac` (stability); the one-step
  ratio against its own persistence denominator (accuracy); and the
  **amplitude ratio and msss on a short roll** (the defect of §3(c)).
- **Mandatory protocol note:** the amplitude read-out must **sample** the
  digits, not take the argmax. Argmax decoding is mean-collapse under a
  different name and would reproduce exactly the damping the arm exists to
  test.
- **Falsifier:** if the two-digit head's one-step ratio is worse than the
  continuous arm's by more than the tier's measured spread AND its sampled
  amplitude ratio is no closer to 1, output-side digitisation is dead and
  E-074d is not bought.

### E-074d · Precision that decays with lead time — only if E-074c passes

*Absolute description:* the same head rolled two ways — coarse digit predicted
with the fine digits also predicted, versus coarse digit predicted and fine
digits **sampled from their conditional prior**. Read amplitude ratio, msss
and corridor accuracy against lead.

The hypothesis is that at long lead only the coarse digit is forecastable, and
that sampling the rest is both honest and better-calibrated than predicting a
mean for it. **Falsifier:** if sampled-fine is worse than predicted-fine on
amplitude at every lead, the graceful-degradation story is wrong.

---

## 7 · Registered risks

1. **Tail starvation.** α = 1 allocates resolution by frequency; AMOC
   transport is carried by the strong-current tail, and an α = 1 ladder gives
   the Gulf Stream core a single bin. E-074a's read-out 3 exists for this and
   its falsifier is written above.
2. **Digit flapping.** §4 — the coarse digit may be least predictable exactly
   where the data is densest. Read-out 2, and the reason the prediction in §4
   is written down before the measurement.
3. **The unbounded end bins.** Within-bin normalisation is undefined on
   `[q_{1−1/K}, ∞)`. Requires either a clamp (the existing `R = 2σ` saturation
   is the precedent) or an explicit tail model; the choice is a measured arm in
   E-074a ("tail exception on/off"), never a silent default.
4. **A quantization floor is not model error.** Read-out 4 exists so this is
   never confused, and so that a head whose error reaches the floor is
   recognised as finished rather than as failing.
5. **Comparability.** A discrete likelihood and a continuous density are not
   comparable without a common partition. Stated in E-074a's read-out 1 because
   getting it wrong produces a number that looks decisive and is not.
6. **Smoothing away hard examples.** A categorical loss caps the gradient a
   surprising target can produce, which is stabilising and is also a form of
   ignoring it. The one direct test of this class in the record inverted:
   **#438 (E-044c arm A3 — the pentad head with Argo-target windows excluded)**
   cut `grad_norm` from ~450 to 24 and made the forecast *worse*, 0.570 against
   ~0.50. The diagnostic that separates the two is whether held-out loss on
   those same hard targets improves, not whether the gradient norm falls.

---

## 8 · Relationship to E-072 §4.2

They compose, and only one part is a substitution.

- §4.2's **β-NLL, σ floor and MSE warm start** are the cure for the
  heteroscedastic-Gaussian pathology, and remain the right answer for any
  continuous head. Unchanged by this plan.
- §4.2's **per-channel-group likelihood zoo** is what this plan proposes to
  replace with one fitted warp per channel. That substitution is the only
  place the two documents disagree, and E-074a decides it on held-out nats
  before any model is built.
- §4.2's **reporting rule** — mean squared error against the predict-the-mean
  bar per family — is kept verbatim. A categorical head still reports MSE
  against `msebar`; it simply computes it from the distribution's mean.

---

## 9 · Prior art, as far as a search finds it

The pieces are all standard and the combination is not one I can locate.
Chronos bins values uniformly, per value, one layer. UniTok (June 2026) uses
single-stage finite scalar quantization and explicitly criticises Chronos's
uniform binning without ever testing quantiles against it. Visual
autoregressive models do coarse-to-fine, but over spatial scale rather than
value precision. Residual vector quantization audio codecs do value-level
coarse-to-fine, but vector-valued with a shared codebook — and the
coarse-to-fine autoregressive stack over such tokens is the AudioLM / MusicLM
structure, which is what this design becomes when the vector codebook is
replaced by per-channel companded scalars. That substitution is the right one
when the channels are physically heterogeneous and not exchangeable, which is
the case here and is not the case for codec dimensions.

This is a search, not a proof of novelty.

- [UniTok · Time Series as Language (arXiv 2606.09861)](https://arxiv.org/html/2606.09861v1)
- [Non-uniform quantizers, Lloyd–Max optimality and high-resolution theory — Panter–Dite and Gish–Pierce (Stanford EE269)](http://web.stanford.edu/class/ee269/Lecture_nonuniform_quantization.pdf)
- [Residual quantization with implicit neural codebooks (arXiv 2401.14732)](https://arxiv.org/pdf/2401.14732)
- [Quantizing space and time: fusing time series and images for Earth observation (arXiv 2510.23118)](https://arxiv.org/html/2510.23118v4)
- [CARP · coarse-to-fine autoregressive prediction (ICCV 2025)](https://openaccess.thecvf.com/content/ICCV2025/papers/Gong_CARP_Visuomotor_Policy_Learning_via_Coarse-to-Fine_Autoregressive_Prediction_ICCV_2025_paper.pdf)

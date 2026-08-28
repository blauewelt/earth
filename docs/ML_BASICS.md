# ML basics — what this programme is, and what its numbers mean

Chris, 2026-08-10: *"note all meanwhile learnings and design decisions into an
ML_BASICS.md."*

This is the conceptual layer. It answers "what are we building, what does a
result mean, and which decisions are already settled" — so that a new session
does not re-derive them, and does not re-run experiments the log already
answered.

Its siblings:

| file | answers |
|---|---|
| `ml/CLAUDE.md` | how to WORK here — the rules, the traps, the dispatch discipline |
| `ml/EXPERIMENTS.md` | what was RUN and what it returned, newest first |
| `docs/INFRASTRUCTURE.md` | the fleet, its failure taxonomy, its invariants |
| this file | what the system IS, and how to read its numbers |

---

## 1 · The question

Can a self-supervised model of the ocean's state carry enough information to
predict large-scale circulation — specifically AMOC transport at 26.5°N
(RAPID), and its relatives (Florida Current, MOVE, OSNAP, SAMBA)?

The programme is deliberately structured so that "the embedding knows this" and
"a read-out with enough capacity can learn this from anything" are separable
claims. Almost every design decision below exists to keep that separation
honest.
### 1b · What is actually being built: a predictor of everything

Chris, 2026-08-19: *"What we're building is a predictor for everything, so it
doesn't matter whether the embedding will contain more or less information than
the raw pixels. The embedding makes large chunks of data 'attendable' by a
transformer. And we can predict everything from predicting embeddings (not just
AMOC). That's the overall plan."*

The system is a **forward model of the North Atlantic state** — all channels,
all pixels, rolled forward in time. AMOC at 26.5°N is the headline number
because RAPID is the best-instrumented read-out we have to score against, not
because transport is the thing being learned. It is one read-out of a general
forecast.

**The codec's job is ATTENDABILITY, not information.** An embedding is a
compression, so it cannot contain more than the pixels it came from, and no
experiment needs to establish that. What it does is turn the ~84,405 active
ocean pixels × 39 channels of a single time step (§3) into a token sequence a
transformer can attend over and roll forward. The daily family-5 tensor is 165.6 GB of raw
pixels: not attendable at any batch size on any box we rent. The codec is what
makes the state a *sequence*.

**So the representation is scored by what stage 2 predicts FROM it** — rollout
skill, corridor AUC, band correlations (§5b and the roll metrics). A
current-state probe comparison — embedding-vs-raw on today's transport — is a
**read-out control** (§5), and a valuable one: it is what separates "the codec
knows this" from "any read-out with spatial structure knows this", and it is how
the pooling artefact was caught. But parity there says nothing about forecast
substrate quality, because the forecaster never has to answer the question that
probe asks. There is also no raw-pixel forecaster to compare against at this
resolution — that is the tractability problem the codec exists to solve.

**Predicting embeddings predicts everything at once.** Anything readable from a
real embedding is readable from a predicted one — RAPID, Florida Current, MOVE,
OSNAP, SAMBA, and any field nobody has probed yet, because the decoder is
already there. The plan in three steps: **encode → predict forward in embedding
space → read out anything.**

---

## 2 · The architecture, and why it is two stages

**Stage 1 — PixelMAE, a codec over pixel-months.** Each ocean pixel at each
month is encoded from its channels (optionally its 3×3 neighbourhood) plus a
context vector (sin/cos month, lat, lon) into `z ∈ R^64`. Training is masked
reconstruction: hide a fraction of channels, predict them back. The codec never
sees the AMOC target.

**Stage 2 — a causal transformer over each pixel's embedding sequence.** Given
`z_{t-K+1..t}` it predicts `z_{t+1}`. The codec is **frozen**; only the
temporal head trains.

**Why frozen, and why two stages.** If the codec trained against the forecast
objective, "the representation improved" and "the dynamics model improved"
would be one number, and neither could be attributed. Freezing makes stage 2's
gain purely a dynamics result, and makes every stage-2 run over the same codec
directly comparable. It also makes the embedding a reusable artefact — which is
what lets `Z` be cached and shared (§8).

The joint variant (`ml/train_joint.py`) deliberately breaks the freeze and is a
separate line of work, currently blocked on E-006 (§7).

**The 40M anchor** is the reference geometry: patch 3, `d_model` 576, 10
layers, 8 heads, `d_dec` 768, `d_z` 64 → **40.69M** parameters at C=39.

---

## 3 · The data, and a trap that has cost two misdiagnoses

`family3_na025.npz` — the quarter-degree North Atlantic tensor.

| | |
|---|---|
| shape | `X[T=516, H=281, W=481, C=39]`, months 1982-01 → 2024-12 |
| ocean cells | **84,405** (channel 0, any month) |
| channels | `cur_speed`, `log_mld`, `ssh`, `rg_t*` (17 depths), `rg_s*` (17), `tau_x`, `tau_y`, `tau_x_std`, `tau_y_std` |

**The channels do not all start when the tensor does.** Channel 0 is
`cur_speed` (GLORYS, from 1993-01) against a tensor beginning 1982-01. Any mask
taken from month 0 is therefore empty, and code that selects a section that way
gets **zero pixels** — after which `mean(axis=0)` over nothing is NaN, silently,
everywhere downstream. That produced an all-NaN `probe_sequence.json` twice
(#101, #116) and was logged both times as "the sequence probe has a bug of its
own". It was a masking bug.

Select from `OBS[..., 0].any(axis=0)` — observed at *any* time — which is the
mask `temporal.py` and `probe_head.py` already use, so every rung of the ladder
scores the same 265-pixel section.

**Anomaly space.** All work is in anomalies: the per-month climatology is
removed using TRAIN YEARS ONLY and each dynamic channel is standardised. State
space was disqualified early — embeddings there were seasonally redundant and
the K-sweep lost skill as history was added.

---

## 4 · The protocol

Everything below exists because a plausible number is easy and a defensible one
is not.

- **Blocked holdouts, never random.** Held-out YEARS plus a held-out
  mid-Atlantic LONGITUDE block, both inherited from the codec checkpoint so
  stage 1 and stage 2 cannot disagree about what was held out.
- **What the YEAR holdout excludes is a choice, `--holdout-scope`, with
  THREE settings — and the legacy answer was narrower than it read.** Until
  2026-08-28 a stage-2 window was dropped only when its FINAL scored bin —
  t+1, plus each unroll and `--direct` offset — fell in a held-out year. But
  the stage-2 loss is dense over the window: every frame predicts the bin
  after itself, so a window ENDING in the K bins after a held-out year still
  carried that year's bins as context AND as teacher-forced targets. Measured
  on the pentad axis (T = 3,142, K = 144, holdout years 2009/2017/2023, 219
  held-out bins, 86,698 pixels):
  - `endpoint_contaminated` — the legacy pool, 2,779 end-bins and all 400,176
    frame-targets. It LEAKS, and it is kept for one reason: the 98 stage-2
    runs archived before c25f6ff trained under it.
  - `target` — the minimal correct fix and the cheapest: the pool is the
    legacy one bin for bin, and the loss simply drops every per-frame term
    whose TARGET bin is held out. **5.25%** of the frame-targets (21,018 of
    400,176), no end-bin lost. No held-out bin is ever a target; held-out
    bins MAY still be read as context.
  - `window` — **the default.** A window is eligible only if none of the bins
    its forward pass touches (the frames, each frame's target, the scored
    reach) is held out: 2,417 end-bins, **13.03%** of the frame-targets gone.
    The held-out year is invisible to training, context included.
  Each setting prints a runtime certificate — an exact recount by a second
  expression — and the run's `stage2_config` records both `holdout_scope` and
  `holdout_masked_frac`, so an artefact says which objective it trained under.
  Numbers from the three scopes are not interchangeable.
- **Reproducing any stage-2 run archived before c25f6ff requires passing
  `--holdout-scope endpoint_contaminated` explicitly** — it is no longer the
  default, and the default (`window`) trains on a different pool.
- **The target is deseasonalised** with a climatology computed from train years
  only. The embedding receives month-of-year as an input, so any seasonal
  signal left in the target is free points.
- **A seasonal-only floor is always reported** — a ridge from `(sin, cos)`
  month alone. On the raw target it shows how much of a correlation was
  calendar; on the deseasonalised target it should sit near zero by
  construction, and if it does not, something is wrong with the split.
- **Lambda is chosen on a train-internal validation tail.** Held-out years are
  touched exactly once per configuration.
- **The one instrument we argue from is `probe_kfold.py`** — year-blocked
  k-fold with a block bootstrap over years. In-training "light probe" values
  are single-split and must be labelled as such wherever quoted.
- **How many seeds a result needs depends on the metric and the scale, and the
  answer is measured, not assumed.** Seed spread in this programme runs from
  0.002 (rolled corridor AUC at the 205M tier) to 0.245 (the RAPID head k-fold
  on a 1.8M head) — two orders of magnitude, on the same programme, sometimes
  on the same checkpoints. `ml/CLAUDE.md` §3b carries the table of every
  replicate the archive holds and the rule that follows from it: a single seed
  is quotable only for corridor AUC at the xl tier and only for effects
  ≥ 0.025, every probe-scored claim and every untried tier still buys two
  seeds, and a single-seed number inside its tier's band is written as
  "consistent with X", never as a level.

**Why year blocks and not months.** AMOC transport is autocorrelated over
months (`r_lowpass18 = 0.82`). An i.i.d. month bootstrap would treat ~240
months as ~240 independent observations and report an interval several times
too tight. n = 240 months is roughly **68 effective DOF**, and about **9** after
an 18-month low-pass.

---

## 5 · The probe ladder — each rung isolates one capability

| rung | what it adds | file |
|---|---|---|
| pooled ridge | what is LINEARLY accessible in the mean-pooled embedding | `probe_kfold.py` |
| MLP | + pointwise nonlinearity | `probe_kfold.py --probe mlp` |
| attention head | + spatial structure ACROSS the section (no pooling) | `probe_head.py` |
| raw / raw-3×3 | the same head on RAW channels — the end-to-end control | `probe_head.py --raw [--raw-patch]` |

The head rung exists because geostrophic transport is an east-minus-west
density difference across the section, and mean-pooling destroys exactly that.
If head ≈ MLP, pooling loses nothing and the representation is the limit; if
head ≫ MLP, the embedding carries section structure the pooled probes cannot
reach.

The raw control is what separates "the codec knows this" from "any read-out
with spatial structure knows this". Match it to the codec's receptive field:
pair a patch-3 codec against `--raw-patch`, not bare `--raw`.

**Comparing two rungs needs a PAIRED test, not two intervals.** Two probes
scored on the same months and the same year blocks share most of their error,
so their marginal CIs overlap far more than their difference varies.
`scripts/paired_probe.py` resamples YEARS and rescores BOTH probes on the same
resampled years; that interval, not the overlap of two others, is what decides
whether a gap is real. This is why `probe_head.py` dumps its out-of-fold
predictions.

### 5b · EVERY RUNG ABOVE SCORES THE CODEC. Stage 2 has its own instrument.

This is the single most confusable fact in the project, and it cost a
four-arm experiment on 2026-08-10.

Every probe in the table takes the **frozen embeddings** and fits a read-out
from them to a transport series. The temporal transformer is not in any of
them. So **two runs that freeze the same codec return the same number from
`probe_kfold.py`, whatever stage 2 did** — #116 (a 60,000-step head) and #125
(a 200,000-step head on a different schedule) both read RAPID
`0.631 [0.513, 0.732]`, rmse 2.16, to the last digit.

That is not a bug. `probe_kfold` answers "what does this REPRESENTATION
contain", and for that question the head is correctly absent — which is why
it is the right instrument for E-002 (codec steps) and E-003 (codec width),
and why "codec-to-codec comparisons are fair" is the caveat written in its
own docstring.

It is the wrong instrument for every stage-2 question. Those are answered by
`temporal.py`'s own evals, which run on the head:

| key in `temporal.json` | what it measures | sample |
|---|---|---|
| `z_t+1.mse_model / mse_persistence` | forecast skill in embedding space | all train months |
| `chan_t+1` | the same, decoded into channels | all train months |
| `rapid_probe` | RAPID from the head's pooled hidden state, **single split** | 36 months |
| `rapid_probe_kfold` | the same features, year-blocked k-fold + block-bootstrap CI | ~240 months |

`rapid_probe_kfold` was added 2026-08-10 for exactly this reason: until then
every stage-2 comparison in the log rested on 36 test months, an SE of order
0.15, and no interval. #88 (U=1) and #93 (U=4) differ by 0.28 on that
instrument, which is either the most important result in the programme or
noise, and there was no way to tell.

**The rule.** Before quoting a number, ask which of the two things you varied.
Codec → the ladder above. Head, schedule, unroll, budget → `temporal.json`,
and prefer `rapid_probe_kfold` to `rapid_probe`. A stage-2 sweep should still
print the codec k-fold as a **control**: it must be identical across arms, and
if it is not, the arms were not holding the codec fixed and nothing else in
the table is a comparison. `scripts/sweep_table.mjs` checks this.

---

## 6 · Baselines, and what a number means

| quantity | value |
|---|---|
| wind-stress-only ridge, 1° tensors | **0.531** |
| wind-stress-only ridge, quarter-degree | **0.568** |
| RAPID monthly σ | 2.79 Sv |
| n | 240 months (~68 effective DOF) |

A correlation without its baseline is not a result. The wind-only ridge is the
one that matters most: much of AMOC variability at monthly scale IS wind, so a
probe that fails to beat it has demonstrated nothing about ocean state.

For stage 2 the reported figure is a **ratio to persistence** —
`z_mse_model / z_mse_persistence` and its data-space twin — because a raw MSE
in `z` has no interpretable scale. 1.0 means "no better than assuming next
month equals this month".

---

## 7 · Loss design — four retractions and what settled it

Between 2026-08-09 and 2026-08-10 four normalisations of the joint objective
were built and retracted. They are worth understanding as one mistake, not
four.

**The mistake:** the forecast term was scored in `z` — a space the encoder
authors. Reconstruction error is measured against observed channels, which are
fixed and external. Forecast error measured against `z` has no fixed units,
because the encoder can rescale `z` freely. Every denominator tried was an
attempt to referee a quantity with no scale, and the model found the free
direction each time: `z_shrink` reached 1/40 in one run and ×250 in another.

**E-006, the resolution:** decode the forecast back into the data before
scoring it.

    L = MSE(x̂_t^masked, x_t) / var(x)  +  MSE(x̂_{t+1}, x_{t+1}) / var(x)

**The algebra, checked before the code** (`tests/test_e006_algebra.py`), with
`s` the encoder's free output scale:

| loss | `L(s)` | `dL/ds` |
|---|---|---|
| z-space | `s²‖a−b‖²/c` | `2s‖a−b‖²/c` — descent shrinks `z` |
| data space | `‖aw−b‖²/var(x)` | **exactly 0** |

Under `z → s·z`, `decoder → decoder/s` the decoded field is unchanged, so the
loss cannot see `s`. The degeneracy is not closed or policed — **there is no
free direction for it to live in.** Also pinned: `dL/d(persistence) = 0`
(the baseline is not in the objective at all), `∂var(x)/∂θ = 0` for every model
parameter, and `∂L/∂rec = ∂L/∂fore`, which is why a plain sum replaces the
smooth max.

**The principles that generalise** (also in `ml/CLAUDE.md`):

1. Normalise by properties of the DATA, never of the MODEL. A denominator the
   model can move is a term in the objective and will be optimised.
2. Keep diagnostics out of the objective. "Am I still as good as the model I
   started from?" is a thing to LOG.
3. Prefer the formulation that removes a failure mode to the one that guards
   against it. Adding a correction to a correction means the earlier choice was
   wrong.
4. A degeneracy you can NAME must be closed or measured, never ranked as
   improbable. The second cheat (inflating the persistence baseline) was
   written down as "worth noting, not worth blocking" and then arrived faster
   than the one being fixed.

---

## 8 · The embedding cache

`Z[T, P, d_z]` — every ocean pixel at every month through the frozen codec.
**10.4 GiB at float32, 5.2 GiB at float16**, and ~95 minutes of an RTX 4090
(43.5M encoder forwards).

- **It is shared.** Every stage-2 run over the same frozen codec needs the
  identical array. #112, #117, #119, #120 and #121 each rebuilt it.
- **It is keyed by the CODEC'S WEIGHT HASH**, not the run name. A run-keyed
  cache poisoned runs #10/#11: the shape check passed, the embeddings belonged
  to a different codec, and two stage-2 models trained on `z` their own decoder
  did not speak — healthy z-space skill, catastrophic decoded skill.
- **It is float16**, and the precision cost is measured, not assumed: 4.3e-8
  MSE on unit-scale embeddings, ~1e-7 of the z-MSE reported (0.39–0.82), and
  1.8e-7 on the model/persistence ratio because the error is common-mode.
- **It is resumable.** A marker beside the `.partial` records `months_done`,
  written AFTER the data is flushed so it can only under-claim. An
  over-claiming marker would skip months holding zeros — real numbers, wrong
  months, no symptom.
- **It is published** to the `embed-cache-v1` release the moment it exists, in
  1.5 GiB chunks, and verified on pull by length and dtype.

---

## 9 · Learning-rate schedules, and why the horizon matters

`CosineAnnealingLR(T_max=steps)` bakes the total step count into the rate. That
one choice produced two distinct problems:

- **A silent bug.** Reload a schedule while asking for a larger total and it
  believes it has finished: `lr = 0.0`, hours of updating nothing, every status
  reading success.
- **A comparability tax.** A 6,000-step run and a 200,000-step run sit at
  different rates at every shared step, so they are two experiments sharing an
  architecture rather than a prefix and its extension. This is why E-007's
  points must each be described as "its own converged cosine", and why a 200k
  point cannot be a continuation of a 60k one.

`--lr-schedule invsqrt` (Noam) makes `lr(s)` a pure function of `s`: resume
stops being a case, and two budgets become a prefix and its extension.

**The trade, stated because it is one.** Cosine anneals to zero and therefore
CONVERGES at a known point, which is what makes "the 60k result" a settled
number. invsqrt never reaches zero, so results are "at step N". For a question
of the form "does more compute help?" that is the better shape, but switching
deserves its own experiment (one budget, both schedules).

### What the literature says, and what we have NOT verified

Read 2026-08-10. Summarised honestly, including its distance from our setting.

- **Horizon-free schedules work, with a caveat.** [Anytime
  Pretraining](https://arxiv.org/html/2602.03702v1) evaluates constant, 1/√t
  and WSD against a "cosine envelope" (cosine separately tuned per duration)
  and finds they track it — but **weight averaging (EMA) is load-bearing**;
  without it polynomial decay typically misses the optimal rate. 1/√t also
  needs its α tuned, which reintroduces a weak horizon dependence.
- **WSD is the current default answer.** Warmup → stable → short cooldown:
  horizon-free while stable, but the cooldown restores a genuine converged
  endpoint. [River-valley
  intuition](https://arxiv.org/abs/2410.05192). For this programme the
  consequence is concrete: E-007's four budgets could be ONE run with four
  cooldowns branched off the stable phase.
- **Re-warming from a minimum is actively harmful.** [Beyond Cosine
  Decay](https://arxiv.org/html/2503.02844), on continual pre-training,
  reports that re-warming from the minimum causes instability and worsens
  forgetting — which is the exact shape of a warm restart that annealed to
  zero first. This is a third candidate explanation for any underperformance
  of E-008's warm restart, alongside "more compute doesn't help" and "the rate
  was too low".
- **Decaying to zero — SUPPORTED, NOT SETTLED.** [Straight to
  Zero](https://arxiv.org/html/2502.15938v2) finds linear decay-to-zero beats
  cosine, dramatically at over-trained budgets, and [convex
  theory](https://arxiv.org/pdf/2501.18965) agrees the optimal cooldown
  fraction is 1. **Chris pushed back on this and was right to.** It rests on
  a thin sample of one literature, all of it LLM pretraining at 124M–610M
  parameters and Chinchilla-scale token budgets; our stage-2 head is 1.8M
  parameters against 240 months of target. No contradicting work was sought.

  **The planned test**, because deferring to literature is the weaker move
  when the experiment is cheap: same budget, same peak, one variable in the
  tail — WSD cooling to **zero** vs WSD cooling to a **floor** (~10% of peak),
  with #121's cosine-to-zero as a third point. Until that runs, treat
  decay-to-zero as a borrowed prior, not a result of ours.

---

## 10 · Resume semantics: three different things

| | what carries over | what it answers |
|---|---|---|
| **continuation** (`--resume-temporal`) | weights, Adam moments, schedule position, RNG | the same trajectory, uninterrupted |
| **warm restart** (`--init-temporal`) | weights only, fresh cosine | does more compute on these weights help? |
| **from scratch** | nothing | a comparable point on a curve of from-scratch runs |

Loading weights alone and calling it a continuation is the error the guard
exists to prevent — Adam's moments take hundreds of steps to rebuild and the
schedule position is lost, so the trajectory differs. `tests/test_resume_temporal.py`
proves each dropped piece matters, and that a warm restart lands somewhere
other than a continuation.

**Every head published before 2026-08-10 is `{args, model}` only** — measured
on `f3_s2_60k`, `f3_s2_24k` and every rescue mirror. None can be continued. The
snapshots written from 2026-08-10 onward carry `opt`/`sched`/`step` and are the
first that can.

---

## 11 · Settled — do not re-run without a reason

- **State-space embeddings**: seasonally redundant; anomaly space only.
- **Capacity on the quarter-degree tensor (E-003)**: null.
- **Training the codec longer (E-002)**: null.
- **Every joint-loss variant before E-006**: retracted, see §7.
- **A "second seed" for `probe_head`**: the seeds were hardwired `(0,1,2)` and
  averaged, so a rerun reproduces the estimator bit-for-bit rather than
  resampling it. Use `--seed-base`, and prefer the paired test (§5) — it is the
  right instrument, and a second seed never was.
- **E-007's shape**: forecast skill still improving at 60k and decelerating,
  while the RAPID probe plateaued at 24k (0.319 → 0.321). E-008 tests whether
  that plateau is a compute artefact.

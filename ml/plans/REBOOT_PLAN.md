# Rebooting the research programme · a handover for whoever continues it

**Written 2026-08-31.** This document is for a researcher — human or agent —
who is picking up the North Atlantic forecasting programme fresh, with read
access to the public repository and its artefacts, and their own compute.
It says what the programme has established, what it has got wrong, what the
open questions are, and how to continue the research so that the next
numbers can be trusted. It is about the science and its method. Where it
names a file, that is for reading: the definition of a protocol lives in
code, and reading the code is the reliable way to reproduce it.

---

## 1. What the programme is trying to do

Learn a forward model of the North Atlantic ocean state from 43 years of
reanalysis (1982–2024), at 0.25° and five-day cadence, and use it to
forecast. The architecture is two-stage: a self-supervised per-pixel codec
compresses each pixel's 40 physical channels into a 32-dimensional embedding
`z`; a causal transformer over a two-year window of embeddings (K = 144
pentads) plus a 145-point spatial stencil predicts the next embedding; a
"roll" advances every pixel forward step by step from a true context and
scores the decoded fields against truth. The headline read-out has been AMOC
transport at 26.5°N (the RAPID array), because it is the best-instrumented
truth series available — not because transport is the only target. The
stated ambition is a predictor of the whole state, from which any quantity
can be read out.

---

## 2. What is known, with the numbers that carry it

**The information bound.** The stage-2 training pool has exactly
2,417 distinct end-times (43 years of pentads minus the held-out years and
the context window), multiplied by 86,698 ocean pixels that are correlated
views of the same 2,417 temporal patterns. For field-level prediction the
effective sample count is in the thousands; for the AMOC target it is about
nine effective independent starts (RAPID is ~20 years, AMOC anomalies
decorrelate over months to years). No architecture, resolution increase or
synthetic corpus changes these numbers. Keep them in front of you.

**Everything before 2026-08-29 was contaminated, and the wreckage is nearly
total.** The stage-2 loss is dense over the 144-frame window, but the pool
only checked that the *final* target was not held out; windows straddling a
holdout year teacher-forced that year's transitions into the weights
(21,018 of 400,176 scored frame-targets were held-out bins). Four
independent signatures confirmed memorisation: field anomaly correlation
0.985 at 5 days and 0.973 at 365 days on the old head (a forecast decays;
a replay does not); corridor skill 0.838 on trained longitudes against 0.176
on held-out ones; an eight-member ensemble that stays pinned together inside
the training record and fans out past it. Every rolled number from before
the fix is retired as evidence of skill. The fix is a window-scope pool: no
frame the forward pass touches may be held out (−13 % supervision).

**The first clean roll (head E-059, 206.66M parameters, evaluated as run
#516) decays like a forecast, and that is nearly all it does.** On the
corridor scope, three starts per held-out year (2009, 2017, 2023), horizon
73 pentads:

| lead h (×5 d) | msss_clim | msss_pers | msss_damped | acc | amp_ratio |
|---|---|---|---|---|---|
| 1 | +0.365 | +0.194 | −0.011 | +0.606 | 0.692 |
| 2 | +0.109 | +0.252 | −0.131 | +0.412 | 0.673 |
| 3 | −0.172 | +0.178 | −0.354 | +0.265 | 0.761 |
| 6 | −0.168 | +0.252 | −0.244 | +0.204 | 0.672 |
| 12 | −0.320 | +0.201 | −0.342 | +0.107 | 0.690 |
| 18 | −0.322 | +0.211 | −0.333 | +0.132 | 0.725 |
| 36 | −0.354 | +0.337 | −0.357 | +0.153 | 0.778 |
| 54 | −0.479 | +0.227 | −0.479 | +0.009 | 0.720 |
| 73 | −0.719 | −0.101 | −0.719 | −0.031 | 0.822 |
| **mean of 73 leads** | **−0.439** | **+0.204** | **−0.461** | **+0.105** | **0.780** |

Definitions (standardised anomaly space, per pixel and channel, pooled over
the scope): `msss_x = 1 − MSE_model / MSE_x` where `clim` predicts zero
anomaly, `pers` repeats the last observed anomaly, `damped` decays it with a
per-pixel AR(1) coefficient fitted on training years; `acc` is the anomaly
correlation; `amp_ratio` is forecast std / truth std. The quantity the
programme used to call "corridor AUC" is the mean `msss_clim` over leads. It
is not an AUC; negative means the squared error exceeds the anomaly
variance, not "below chance".

Per channel (only 8 of the 40 are scoreable — the 32 Argo `rg_*` channels
have no scored samples at any lead):

| channel | acc h=1 | acc h=6 | acc h=18 | mean msss_clim | mean msss_damped |
|---|---|---|---|---|---|
| `sst` | +0.759 | +0.343 | +0.337 | −0.165 | −0.191 |
| `ssh` | +0.838 | +0.252 | +0.113 | −0.499 | −0.562 |
| `cur_speed` | +0.604 | +0.131 | +0.052 | −0.591 | −0.606 |
| `log_mld` | +0.572 | +0.178 | −0.009 | −0.464 | −0.469 |
| `tau_x` / `tau_y` | +0.490 / +0.235 | +0.097 / +0.039 | +0.022 / −0.017 | −0.383 / −0.426 | −0.386 / −0.428 |
| `tau_x_std` / `tau_y_std` | +0.222 / +0.207 | +0.134 / +0.070 | +0.104 / +0.057 | −0.409 / −0.465 | −0.410 / −0.466 |

Transport (unpooled read-out of the rolled section states against RAPID):
r = +0.107 over 5–90 d, −0.242 over 95–180 d, +0.163 over 185–365 d — all
inside the noise of zero at nine effective starts.

Three readings of this table matter more than any other fact in this
document:

1. **The clean head loses to damped persistence at every one of the 73
   leads.** "Beats persistence" was true only of raw persistence. The
   cheapest classical null already wins, everywhere.
2. **The negative `msss` is largely a calibration failure, not absent
   correlation.** The identity `msss_clim = 1 − (1 + a² − 2a·ACC)`, with
   `a = amp_ratio`, reproduces all 73 corridor leads to a mean absolute error
   of 0.0135. The head emits anomalies at 78 % amplitude while its
   correlation is 10 %. Rescaled to `a = ACC` the same rolled states would
   score about +0.02 — small, but positive. This is a decoding question, not
   a capacity one.
3. **Where correlation survives is specific:** SST and SSH at one step;
   SST alone beyond a few pentads. The subsurface is unscored, the winds are
   essentially noise past one step.

**Capacity was not the axis.** Under the clean pool, heads of 7.6M, 40.4M,
206.7M and 400M parameters all reach their best held-out one-step loss
inside roughly the first 2,000 of 200,000 steps and get worse for the rest
(best levels 0.58–0.62 across a 53× parameter span). At the pre-registered
comparison step (20,000) the 7.6M arm was 0.051 *worse* than 206.7M — a
prediction that it would be within 0.02 failed and is recorded as failed. On
the RAPID probe the ordering reverses: 7.6M holds 0.598–0.611 across ten
probes while 206.7M collapses 0.616 → 0.515. Small models are the right
default for cost and probe stability; they are not proven right for loss.
About 99 % of every training budget so far was spent past the point where
anything generalisable was learned.

**The evaluation stack has been wrong twice and survived months each
time** — once structurally (the pool), once semantically (the "AUC").
Assume it is wrong somewhere else; verifying a metric is ordinary work here.

**What has never been done:** a Linear Inverse Model or a nearest-analogue
baseline on this protocol; any confidence interval on a rolled number; a
roll of an early-stopped checkpoint (every rolled head is an end state,
far past its held-out minimum); a clean-pool head with an ensemble (so the
dispersion test is unanswered for clean heads, not failed); any roll on a
tensor with a longitude hole (so nothing can be said about spatial
generalisation of a clean head); a codec trained under the terminal holdout.

---

## 3. The frozen protocol, and one thing it needs before it can be used

Decided 2026-08-30 and to be kept:

- **Terminal holdout for the final test: train ≤ 2020, test 2021–2024, no
  gap.** The tensor ends 2024-12.
- **Trained-longitude and held-out-longitude scores reported separately;
  never the blend.**
- **Lead-decay is a standing falsifier** on every rolled result: a profile
  that does not decay with lead is a replay, whatever its level.
- **A null ladder beside every number** (§4, step 1).
- **Early stopping at the held-out minimum.**
- **Rolled skill is the verdict; a probe is never a verdict.**
- **Small tier (7.6M) is the default; anything larger needs a written
  argument.**

**The dependency people miss:** the held-out years are a property of the
*codec*. The codec was trained with 2009, 2017 and 2023 held out (from both
its self-supervised training and the anomaly statistics), and the stage-2
pool inherits the codec's holdout years. So every archived number lives on
the *interspersed* split, and the codec is clean for it. For the terminal
split the existing codec is not usable: it reconstructed 2021–2024 during
training, and a stage-2 head cannot be trained on a holdout its codec did
not respect without leaking. **A terminal-holdout programme therefore
begins with a codec trained on ≤ 2020** (same architecture and data as the
current one: 37.976M parameters, 512×12, four heads, decoder width 256,
d_z 32, patch 1, all longitudes; ~19 h on one consumer GPU last time), its
anomaly statistics from ≤ 2020, and a re-embedding of the tensor with it.
Everything in §4 that is evaluation of *existing* heads runs on the
interspersed split and does not wait for this; start the codec early and
develop on the interspersed split meanwhile, keeping 2021–2024 untouched by
any development decision.

---

## 4. How to continue — the programme, in the order it should run

The ordering rests on one principle: **nothing new is trained until the
nulls and the instrument exist.** The programme's history is that
evaluation errors outran modelling gains by a wide margin, and the newest
evaluation fact (damped persistence wins at every lead) says the bar is
already above the model. Steps 1–3 need no accelerator. Step 4 is the
codec. Steps 5 onward spend compute, cheaply.

### Step 1 · Put the classical nulls through the identical battery

Score three predictors with exactly the protocol of the clean roll — same
tensor, same anomaly standardisation from training years, same three scopes
(gate: 600 random pixels ∪ the RAPID section; corridor: the fastest quarter
of the window by current speed, dilated two cells, ∪ the section, 30,158
pixels; window: all pixels), same start rule (3 per held-out year, spread
round the seasonal cycle), same horizon (73), same metric definitions —
and place them in one table beside the clean heads. The evaluator code is
the specification; reuse its functions for the statistics, the scopes, the
start selection and the per-lead arithmetic rather than re-deriving them,
because re-implementation is how a third evaluation error would enter.
Score in the standardised physical-channel space the heads are scored in.

The three nulls, in increasing informativeness:

1. **Damped persistence.** Already computed inside every roll as the
   `damped` denominator; read it out as its own row (its `msss_clim`, `acc`,
   `amp_ratio` per lead) so it stands in the table rather than hiding in a
   ratio. Per pixel and channel, AR(1) decay toward zero anomaly, coefficient
   fitted on training years.
2. **Nearest-analogue retrieval.** Library = training-year states only,
   excluding a ±1-year buffer around each start. State = the standardised
   anomaly field of the 8 scoreable channels on the corridor (or its leading
   EOFs — use the same basis as the LIM so the two are comparable). For each
   start take the k nearest library states (k = 1, 5, 20; Euclidean in the
   reduced space) and forecast lead h as the mean of what followed each
   analogue h pentads later. If retrieval matches the transformer, the
   transformer is an expensive lookup table.
3. **A Linear Inverse Model.** EOFs of the training-year anomaly fields on
   the corridor; truncate to m modes; one-pentad propagator
   `G(1) = C(1)·C(0)⁻¹` in PC space; forecast `x(h) = G(1)^h · x(0)`; project
   back; score. Choose m by cross-validation inside the training years (try
   10, 20, 50, 100 and report all four). Run the standard τ-test (fit at
   τ₀ = 1 and 2 pentads; a valid LIM gives a consistent `L = log G / τ₀`).
   Do it twice: on the raw anomaly fields, and on the frozen codec's
   embeddings `z`. The difference between the two is the first measurement
   of whether the codec discards predictable signal — a question about the
   two-stage architecture nobody has been able to ask.

**Pre-registered expectation:** the cross-validated LIM matches or beats
both clean heads on `acc` at every lead ≥ 2 and on mean `msss_clim` in every
scope. It is falsified if a head beats the LIM at any lead ≤ 6 on SST or SSH
by more than the interval from step 2. Either outcome is a headline. If the
LIM wins, the programme's central result is that 43 years of reanalysis
support a linear predictable component and nothing the transformer has
found beyond it, and the modelling question becomes "what does a learned
model add at leads 1–2, and can any change make it beat the LIM at lead 3".
If the transformer wins at short leads, there is finally a defensible gap
to explain, with the LIM as the standing bar.

### Step 2 · Build the instrument: intervals, a development split, and the smallest detectable effect

Every comparison so far is n = 1 with "inside the noise of zero" argued
rather than measured. Before any development decision:

- **Block-bootstrap confidence intervals** on every per-lead metric and on
  every *difference* between two rows (head vs null, head vs head). The
  roll accumulates per-lead sums; keep them per (held-out year, start) and
  resample those blocks with replacement (2,000 draws; report 5–95 %). For
  the transport bands the block is the year. Report intervals beside every
  number from now on.
- **A rolling-origin blocked cross-validation for development.** Four
  folds of one held-out year each, spaced at least five years apart, all
  ≤ 2020, each under the window-scope exclusion, each rolled from its own
  year. This gives development decisions an n. It requires the pool and the
  start selection to accept an explicit list of held-out years, and —
  because the holdout is a codec property — either a codec per fold or a
  codec that held out all four fold-years at once (the latter is cheaper and
  is the right choice: one codec, four folds).
- **The minimum detectable effect table.** From the bootstrap spread of a
  head-vs-null difference, publish per lead and per scope the smallest
  difference distinguishable from zero at 90 %. Several open questions may
  be formally unanswerable at this sample count; that table is how one
  says so, and it is a result.

**Falsifier for the instrument itself:** if the interval on corridor `acc`
at lead 18 is wider than ±0.15, the three-year interspersed battery cannot
rank heads at month-plus horizons at all, and the programme should move to
the terminal codec with more starts per year (up to 73 are available)
before ranking anything.

### Step 3 · Change the cost structure, so replication is the default

Cap stage-2 training at about 5,000 steps with a checkpoint every 500, and
roll the checkpoint with the minimum held-out one-step loss (ties to the
earlier step; selection by script from the metrics, never by eye). Keep the
learning-rate schedule exactly as before (exponential decay, half-life
40,000 steps, warm-up 2,000, peak 1e-3), so the first 5,000 steps trace the
same trajectory as every archived early curve and remain comparable. Keep
the selected checkpoint *and* the step-5,000 end state, so "does early
stopping matter for the roll" is answered per arm at no extra cost. At the
7.6M tier this makes an arm roughly a 1.5-hour job instead of eight; five
seeds per configuration then cost an afternoon and become the default. A
claim needs the range of one configuration to clear the range of the other.
The 200,000-step budget is retired for stage 2 unless a held-out curve is
still falling at 5,000 — check that, and extend only then.

### Step 4 · The terminal-holdout codec

As §3 describes. One codec, holdout 2021–2024 (or, to serve step 2 as well,
2021–2024 plus the four development fold-years). Record its held-out
reconstruction curve and check whether it, too, plateaus early; if so the
step-3 cap applies to codecs as well. Embed the tensor with it. This is the
long pole; start it first and let steps 1–3 run beside it.

### Step 5 · The cheap evaluation quartet, on the interspersed split

All four compare directly to the clean roll in §2 and to the step-1 nulls;
write the expectation for each down before it runs.

1. **Roll an early-stopped checkpoint.** No early checkpoint of E-059 or
   E-060a survives — only final states — so train the 7.6M configuration
   (256×8, K 144, stencil 145, spiral ring `111-4444-0.71-0.5`, input
   z-noise 0.7, gradient clip 128, batch 256, all longitudes, window scope)
   to 5,000 steps, select per step 3, and roll both the selected and the
   end checkpoint through the same battery. Expectation: the selected
   checkpoint's `acc` at leads 2–18 exceeds the 20k head's by more than the
   interval; falsified if they agree — which would say the end states did
   not lose rolled skill by over-training, and the early-peak story is about
   the one-step loss only. Both answers matter.
2. **Five seeds of that arm.** The first replicate set at pentad cadence for
   rolled skill: the spread on mean `msss_clim`, per-lead `acc` and the
   transport bands. Until it exists no two 7.6M numbers can be called
   different.
3. **Amplitude calibration as a decoding option.** Per lead and channel,
   multiply the rolled anomaly by a factor fitted on *training-year* starts
   only — the correlation-optimal `a* = ACC_train(h)` and the MSE-optimal
   regression slope — and re-score the held-out starts. Report calibrated
   numbers beside uncalibrated ones, for nulls and heads alike (a calibrated
   LIM is the fair comparison for a calibrated head). Expectation: the
   corridor mean goes from −0.439 to within the interval of +0.02; falsified
   otherwise, in which case the identity in §2 is wrong somewhere and that is
   worth knowing.
4. **Drop the 32 `rg_*` channels from the model's input.** They are 80 % of
   the tensor's bytes and of the input dimensions, null at every lead, 1°
   native, monthly, and absent before 2004. Train the same 7.6M arm on the 8
   scoreable channels (masking them at stage-2 time if the code path allows;
   otherwise this waits for an 8-channel codec and joins step 4).
   Expectation: no loss of rolled skill; falsified if `acc` at leads ≤ 6
   drops by more than the seed range.

### Step 6 · Decision gate, and the headline question

When the step-1 table exists with step-2 intervals:

- LIM ≥ transformer at every lead ≥ 2 → the LIM is the programme's reference
  model; further modelling asks only what a learned model adds at leads
  1–2 and whether any change beats the LIM at lead 3.
- Transformer beats the LIM at short leads by more than the interval → the
  re-ranking programme proceeds (cadence → stencil → unroll → z-noise →
  ensemble/FGN → width, at 7.6M, five seeds, short budgets), each rung asked
  "does it beat the LIM by more than the interval, at which leads, on which
  channels", never "does it beat last week's arm".
- The minimum-detectable-effect table says ±0.1 in `acc` at lead ≥ 12 is
  unresolvable on the interspersed battery → move development to the
  terminal codec with more starts per year before ranking anything.

**The headline decision, which is the operator's:** whether AMOC transport
stays the headline or the headline becomes field-level subseasonal-to-
seasonal skill against classical nulls, with AMOC kept as a bounded,
interval-carrying secondary read-out and the predictability/power bound
written up as a result in itself. The recommendation of this handover is to
pivot: at nine effective starts the AMOC question is not resolvable with
this data by any model, the field question has thousands of effective
samples, and the clean roll already localises where the signal is (SST and
SSH, days to weeks). Nothing in steps 1–5 depends on the decision.

### Step 7 · Data — clean before adding, and add only what touches the bound

Only new *temporal* samples touch the 2,417-end-bin constraint; more pixels
of the same 43 years (finer grids) do not, and are declined. In order, each
as a measured before/after on the step-3 arm:

1. Move the 32 `rg_*` channels to a coarse sidecar (frees ~27 GB, more than
   every proposed import combined; the loader upsamples on read).
2. Extend the tensor through 2025–2026 (+4.5 % end-bins; lets a later
   terminal test run to 2026).
3. Extend backward toward 1958 with a coarse (1°) temperature/salinity
   product (EN4 or IAP, ~3 GB), run *as a measurement*: does the held-out
   minimum move, does rolled `acc` move. It is the only candidate aimed at
   the bound.
4. Surface heat and freshwater flux forcing (ERA5; needs an account).
   Missing physics rather than more samples — worth adding, will not move
   the bound.
5. The three-member observationally-constrained reanalysis ensemble as
   low-bias augmentation, again as a measurement.
6. The CMIP6 pre-industrial-control corpus (800 model-years, already built,
   three channels): only ever for pretraining the *codec*, never the
   forecaster — CMIP6 AMOC strength spans ~10–30 Sv against RAPID's ~17 and
   its variability is among the most model-dependent quantities in the
   archive, so a forecaster trained on it would distil the one thing these
   models are least trustworthy about. The mandatory control: embed the
   reanalysis with the same 37 channels masked, or "pretraining helped"
   cannot be told from "the fine-tune adapted to its pretraining channel
   set".

### Step 8 · The paper

Every "corridor AUC" in the current draft is a mean `msss_clim` from a
contaminated head; mark those sections as retired now, before the step-1
table exists, so no reader takes them as current. Add acknowledgements and
data availability for the sources already used (CMEMS/GLORYS/GREP licence
terms, Argo, NCEP R1, OISST, GPCP). Once steps 1–2 read out, the paper's
centre of gravity is likely "the limits of learned subseasonal-to-seasonal
prediction from a 43-year reanalysis: where a transformer beats classical
nulls, where it does not, and what the sample count can resolve".

---

## 5. Working rules that the reboot exists to enforce

- **Write the hypothesis, the control and the falsifier down before the run
  starts.** If you cannot say what result would falsify the expectation, the
  configuration cannot test it. A result that arrives without a
  pre-registered reading is exploratory, and is labelled so.
- **One variable per comparison.** Two things changed means no result.
- **Every number comes from an artefact** — a results file you can re-open —
  never from a log line or a memory of one. Read the artefact's file list
  and its contents; a run's success colour is not an inventory in either
  direction.
- **A single-seed number inside its tier's spread is a consistency, never a
  level.** Say "consistent with X at n = 1 against a spread of ±s", not
  "rolls at X".
- **Quote the null beside the number, and say which null.** Raw persistence
  and damped persistence give opposite verdicts on the same head.
- **Check the same-class control before registering an expectation**, not
  only before calling one violated. An expectation written for one kind of
  run and applied to another is the most productive source of false alarms
  this programme has had.
- **Size a guard from the allocation it guards**, and put preconditions
  where the inputs are all that has been spent.
- **Verify the artefact, not the intention** — open the checkpoint, read its
  own configuration, before building anything on it.
- **Record failed predictions as failed**, in the same place the prediction
  was made. A future reader should not find a softened version anywhere.
- **Report in four sections:** completed work (what it was, what it
  returned against its named control); what changed in the picture and the
  most promising next steps; what is queued or in flight (with what it must
  beat and when it lands); and proposed re-prioritisations with their cost
  and what evidence would reverse them. Every experiment gets a one-sentence
  plain-English statement of the question it asks before any number.

---

## 6. What to keep a record of, and how

Keep one research log, append-only, that a stranger can read six weeks from
now without you. For each experiment, before it runs: an identifier; one
sentence saying what question it asks; the configuration in absolute terms
(parameter count, stage, tensor by name, architecture, steps × batch, what
it seeds from, holdout years and scope, seed); the named control; the
pre-registered expectation and its falsifier; the cost estimate. After it
runs: the numbers with intervals and the null beside them; whether the
falsifier fired; the actual cost; and one line on what it changes. Never
edit the "before" half after the fact — append a correction instead.

Keep one standing summary page, short, re-stamped with a date every time it
changes: what the programme currently believes (one line per settled
result, with its interval and its null), what is in flight, and the ranked
list of next steps with the reason for the ranking. A reader who finds the
stamp old should distrust the in-flight section.

Keep one results table for the null ladder and every clean head, all on
the same protocol: rows are predictors, columns are mean `msss_clim` and
`auc_damped` per scope, mean `acc`, `acc` at leads 1/6/18/73, `amp_ratio`,
SST-only `msss_clim` at leads 1/6/18, each with its interval. This table is
the programme's state; everything else is commentary on it.

Keep a plain list of corrections to the evaluation stack as they are found,
each with the date and what number it changed. There will be more.

---

## 7. Where the public material is

Everything below is readable without a login.

| what | where |
|---|---|
| Repository (the code is the protocol's definition) | <https://github.com/blauewelt/earth> |
| Any repo markdown, phone-readable | `https://blauewelt.github.io/earth/docs.html?f=<path>` |
| The evaluator (scopes, starts, metrics, baselines) | `ml/rollout_spatial.py` |
| The stage-2 trainer and pool (holdout scope, window rule) | `ml/temporal.py`, `ml/jaxport/train_stage2.py` |
| The codec trainer | `ml/train.py`, `ml/jaxport/train_stage1.py` |
| Result bundles, one JSON per run | `https://raw.githubusercontent.com/blauewelt/earth/ml-metrics/probes-<n>.json` — the clean roll is `probes-516.json`; its head block is under `files.rollout_spatial.json.heads` |
| Codecs and stage-2 heads (weights, with their own configuration inside) | GitHub release `model-checkpoints-v1`: `run-415__pixelmae.pt` (the codec), `head-weights-e059-200k-window-s0.pt` (206.66M), `head-weights-e060a-20k-window-s0.pt` (7.60M), `head-weights-e060b-20k-window-s0.pt` (40.39M) |
| The tensor `family4_na025_pentad_r2` (T 3,142 × 281 × 481 × 40; channel order `cur_speed, log_mld, ssh, 16 × rg_t*, 16 × rg_s*, tau_x, tau_y, tau_x_std, tau_y_std, sst`) | release `data-cache-v1` (chunked) and the Hugging Face dataset `chfrank/earth-tensors` (also the truth series, the SST bake and the CMIP6 corpus) |
| The codec's embedding of the tensor, `Z_8b639abe36_37e146384b.npy` ([3142, 86698, 32] float16) | release `embed-cache-v1`, 1.5 GiB chunks plus a manifest |
| The experiment log, the standing overview, the protocol reset | `ml/EXPERIMENTS.md`, `ml/OVERVIEW.md`, `ml/plans/PROTOCOL_RESET.md` |
| Background: what the system is and what its numbers mean | `docs/ML_BASICS.md` |
| The two handovers this document condenses | `ml/handoffs/SESSION_HANDOVER_2026-08-31.md`, `ml/handoffs/REBOOT_HANDOVER_2026-08-31.md` |

---

## 8. Things that will look like results and are not

- A rolled score that does not decay with lead. It is a replay.
- A skill number with no null beside it, or with the wrong null (raw
  persistence flatters; damped persistence is the honest bar).
- A negative `msss` read as "worse than chance". It means the error exceeds
  the anomaly variance — usually miscalibration, as §2 shows.
- A difference between two single-seed arms smaller than the seed spread.
- A trained/held-out longitude comparison on the current tensor. It has no
  longitude hole; the two scopes are identical.
- A probe number as a verdict. The probe is a read-out of the current state;
  the programme's question is what the state does next.
- An improvement measured on the interspersed years and reported as if it
  were the terminal test. The terminal years are opened once.
- A green run with an empty results bundle, or a failed run whose
  deliverable is complete. Read the artefact, not the colour.

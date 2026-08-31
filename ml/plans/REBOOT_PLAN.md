# The reboot plan · what to do next, in what order, and how to know it worked

**Written 2026-08-31 ~09:30Z.** This is the execution plan for rebooting the
research programme after the protocol reset. It is written so that a session
with none of this project's chat history can pick up any work package below
and run it: every package says what it is for, what to build, where the
inputs are, what "done" looks like, and what result would falsify the
expectation behind it.

Read first, in this order: `ml/OVERVIEW.md` (top block), `ml/plans/PROTOCOL_RESET.md`,
`ml/handoffs/SESSION_HANDOVER_2026-08-31.md`, `ml/handoffs/REBOOT_HANDOVER_2026-08-31.md`,
`ml/CLAUDE.md`. Everything renders at `https://blauewelt.github.io/earth/docs.html?f=<path>`.

**Rules that apply to every package here** (they are `ml/CLAUDE.md`'s rules,
restated because they are what the reboot is for):

- Write the `ml/EXPERIMENTS.md` entry — hypothesis, control, falsifier —
  BEFORE the run is dispatched, never after. Use the structured header of
  `ml/CLAUDE.md` §0d. A run number never appears without its summary (§0c).
- One variable per comparison. If two things changed, the number is not a
  result.
- Every number comes from an artefact on `ml-metrics` or in the repo, never
  from a log line or a memory of one.
- A single-seed number inside its tier's spread is a consistency, not a level.
- Assume the evaluation stack is still wrong somewhere. It has been wrong
  twice (the pool; "corridor AUC"). Verifying a metric is ordinary work.
- Update `ml/OVERVIEW.md` (move the stamp) and the project's
  `claude/expectations.md` in the same breath as dispatching or harvesting.

---

## 0. Where the programme stands, in numbers

- **The wall.** The stage-2 pool is `2,417 end-bins × 86,698 pixels`. There
  are 2,417 distinct temporal patterns in 43 years of pentads; every pixel is
  a correlated view of them. For the AMOC target there are **~9 effective
  starts**. No architecture, resolution or CMIP6 corpus changes either
  number.
- **Every clean arm peaks early.** Under `--holdout-scope window`, heads from
  7.6M to 400M reach their best held-out one-step loss inside ~2,000 of
  200,000 steps and worsen for the rest (E-060). ~99 % of every training
  budget to date was spent past the point where anything generalisable was
  learned.
- **The first honest roll (E-062-R0, #516) decays like a forecast** — field
  anomaly correlation `acc` 0.606 at 5 d → −0.031 at 365 d (its contaminated
  twin #510 read 0.985 → 0.973, flat). Corridor `msss_clim` −0.439, a
  calibration failure (mean `amp_ratio` 0.780 at mean `acc` 0.105).
- **New reading, 2026-08-31, from the same artefact:** the clean head beats
  RAW persistence at every lead but the last (mean `msss_pers` +0.204) — and
  **loses to DAMPED persistence at every one of the 73 leads** (corridor
  `msss_damped` −0.011 at h=1, −0.131 at h=2, mean −0.461; `auc_damped`
  negative in all three scopes). The evaluator already computes a per-pixel,
  per-channel AR(1) decay-to-climatology baseline (`ar1_train`,
  `ml/rollout_spatial.py` ~line 2094) and the clean transformer does not
  clear it once. **This is the single most important fact for the reboot:
  the cheapest classical null already wins.** Everything in §2 is designed
  to find out whether anything learned can beat it.
- **Where residual skill lives:** SST (`acc` 0.759 at h=1, the only channel
  with positive `msss_clim` past a few pentads), SSH at 5 days (`acc` 0.838
  at h=1). Only 8 of 40 channels are scored at all — the 32 `rg_*` Argo
  channels are null at every lead and are 80 % of the tensor's bytes.
- **The holdout.** Every archived number uses the INTERSPERSED holdout years
  2009 / 2017 / 2023 (the codec's `--holdout-years` default; the stage-2
  pool inherits `hold_years` from the codec checkpoint's `args`). The
  frozen protocol (Chris, 2026-08-30) is a TERMINAL holdout: train ≤ 2020,
  test 2021–2024, no gap. **No codec trained under the terminal holdout
  exists yet** — see WP3.
- **In flight:** E-062-R0b (#518 died in 3 min on a full disk; re-dispatch
  pending) — the 7.6M head through #516's identical battery. Its result is
  the width axis at two points under a clean pool.

---

## 1. The four ideas behind the ordering

1. **Nulls before training.** Nothing new is trained until the transformer
   has been placed beside damped persistence, nearest-analogue retrieval and
   a Linear Inverse Model on the identical protocol. The programme's history
   is that evaluation errors outran modelling gains by a wide margin, and
   the newest evaluation fact (above) says the bar is already above the head.
2. **An instrument with error bars before a terminal exam.** The terminal
   holdout can be opened once. Development decisions need a blocked
   cross-validation over several held-out years and block-bootstrap
   intervals, or every comparison is n = 1.
3. **Stop paying for step 2,001 onwards.** Cap stage-2 budgets at ~5,000
   steps with checkpoint selection at the held-out minimum. A 7.6M arm then
   costs ~1.5 h instead of ~8 h, and five seeds cost an afternoon. This is
   what makes replication and the whole R1 programme affordable.
4. **Pivot the headline to what the data can decide.** At ~9 effective
   starts, "can this predict AMOC at 26.5°N" is not resolvable. "Where does
   learned skill beat classical nulls, and out to what lead" is — and R0
   already localises it. AMOC stays as a bounded secondary read-out with
   intervals. (This decision is Chris's; §2 does not depend on it.)

---

## 2. Work packages

Each package: **Goal · Why · Build · Inputs · Done when · Falsifier · Cost ·
Record.** Packages marked ★ need no accelerator at all. Dependencies are in
§3.

### WP0 · Finish what is bought ★ (mostly done)

- The E-060 heads are on `model-checkpoints-v1` as
  `head-weights-e060a-20k-window-s0.pt` (7.6M) and
  `head-weights-e060b-20k-window-s0.pt` (40.4M), verified against their own
  `args`. Done 08-31.
- **E-062-R0b**: re-dispatch #518's inputs unchanged once Vast 49242934's
  disk is cleared (`/opt/earth-cache/family4_na025_pentad_r2_scratch.npy`,
  34 GB, a rebuilt-every-run scratch copy; `scripts/disk_hygiene.sh` now
  frees it unconditionally at job start, commit `6b73c22`). Reading is
  pre-registered in `ml/EXPERIMENTS.md#e-062` §(j): shape first (lead-decay
  must pass), then level against #516. Done when `probes-<n>.json` is on
  `ml-metrics` and §(j) carries the table.
- **The 36-month dispersion test** (PROTOCOL_RESET §2d) is UNANSWERED for a
  clean head because E-059/E-060 are deterministic. Do not claim it either
  way; it needs a clean-pool head with an ensemble mechanism (WP6, later).

### WP1 · The null ladder on the exact R0 protocol ★

**Goal.** Score three classical predictors through the SAME battery, starts,
scopes, hold years and metric definitions as #516, so that every
transformer number has a null beside it.

**Why.** Damped persistence already beats the clean head at every lead. If a
linear model on ~10³ parameters matches or beats the transformer at every
lead, that is the programme's central result and it redirects everything
downstream. If the transformer wins at short leads, there is finally a
defensible gap to explain.

**Build.** One new script, `ml/null_ladder.py`, that reuses the roller's own
functions rather than re-deriving them (the evaluator has been wrong twice;
re-implementing it is how a third error enters). Concretely, import from
`ml/rollout_spatial.py`: the anomaly/standardisation machinery
(`stream_stats`, `StdMonths` — statistics from TRAINING years only), the
scopes (`corridor_pixels`, the `gate` subset with `np.random.default_rng(0)`,
`window` = all pixels), the start rule (`starts` block: `per_year` 3, "every
k-th start of the holdout year's list"), and the `chan_skill` row arithmetic
(the function documented at ~line 1196: `msss_clim`, `msss_pers`,
`msss_damped`, `acc`, `amp_ratio` from one set of `[H+1]` sums, plus the
`per_channel` rows). Produce a JSON with the same shape as a head's block in
`rollout_spatial.json` (`corridor`, `gate`, `window`, each with `chan_skill`,
`per_channel`, `horizon_auc`, `auc_damped`), one block per null, so
`scripts/sweep_table.mjs`-style tooling and a human can read nulls and heads
in one table. Score in the same space the heads are scored in
(standardised physical channels, the 8 scoreable ones), at horizon 73,
starts 3 per holdout year, hold years 2009/2017/2023 (the interspersed
protocol — this package compares against #516, so it uses #516's split).

The three nulls:

1. **Damped persistence** — already computed inside every roll as
   `msss_damped`. Read it OUT as its own block (its `msss_clim`, `acc`,
   `amp_ratio` per lead) so it appears as a row in the table, not only as a
   denominator. Definition: per pixel, per channel, AR(1) toward zero
   anomaly with the coefficient fitted on training years (`ar1_train`).
2. **Nearest-analogue retrieval.** Library = all TRAINING-year states
   (exclude the holdout years and ±1 year around each start). State vector =
   the standardised anomaly field of the 8 scoreable channels on the
   corridor pixels (or the leading ~50 EOFs of it — see 3; use the same
   basis for both so the comparison is clean). For each start, find the k
   nearest library states (cosine or Euclidean in the reduced space; report
   k = 1, 5, 20), and the forecast at lead h is the mean of what followed
   each analogue h pentads later. Score exactly like a head.
3. **Linear Inverse Model.** Fit on training years only: EOFs of the
   standardised anomaly field (corridor pixels × 8 channels; also a z-space
   variant on the frozen codec's embeddings — the difference between the
   two tells whether the codec discards predictable signal), truncate to m
   modes, estimate the one-pentad propagator `G(1) = C(1) C(0)⁻¹` in PC
   space, forecast `x(h) = G(1)^h x(0)`, project back, score. Choose m by
   cross-validation INSIDE the training years (try 10 / 20 / 50 / 100);
   report all four, with the CV-chosen one as the headline. Check the
   Nyquist/tau test (fit `G` at τ₀ = 1 and 2 pentads; a proper LIM gives
   consistent `L = log(G)/τ₀`) and report it — it is the standard sanity
   check on the linear assumption.

Also add `--calibrate` support to the table (see WP5) so calibrated and
uncalibrated numbers sit side by side for nulls and heads alike.

**Inputs.** `family4_na025_pentad_r2` (the r2 tensor, `X` sidecar 34 GB),
`Z_8b639abe36_37e146384b.npy` (16.2 GiB, for the z-space LIM), the frozen
codec `run-415__pixelmae.pt`, `probes-516.json` (the head block to place the
nulls beside). All on the releases / `ml-metrics`; a 128 GB-RAM box or a
memmap is enough — this is CPU work.

**Done when.** One table, in `ml/EXPERIMENTS.md` under a new entry
**E-064 · the null ladder**, with rows {damped persistence, retrieval k∈{1,5,20},
LIM m∈{10,20,50,100} (field) and LIM (z-space), #516's head, #R0b's head} and
columns {corridor `horizon_auc`, `auc_damped`, mean `acc`, `acc` at h=1/6/18/73,
`amp_ratio`, SST-only `msss_clim` at h=1/6/18}, plus the per-lead curves as a
figure, plus intervals from WP2 once it exists.

**Falsifier (pre-registered).** Expectation: the CV-chosen LIM matches or
beats both transformer heads on `acc` at every lead ≥ 2 and on `horizon_auc`
in every scope. The expectation is FALSE if a head beats the LIM by more than
the WP2 interval at any lead ≤ 6 on SST or SSH — that would be the first
evidence of learned nonlinear skill in the programme, and it should be the
headline of the next paper draft.

**Cost.** Code: 1–2 sessions. Compute: CPU, hours.

**Record.** E-064 entry; `ml/OVERVIEW.md` "most promising next steps" gets
re-ranked from its result.

### WP2 · The instrument: blocked CV, block-bootstrap intervals, minimum detectable effect ★

**Goal.** Replace "inside noise of zero, by argument" with intervals, and
replace the single-holdout n = 1 with a development protocol that has n.

**Build.**

- **Block bootstrap over the skill sums.** The roller accumulates per-lead
  sums (`n, mse_m, mse_p, mse_d, mse_c, sx, sy, sxx, syy, sxy`) for the
  whole battery. Change it (additively — new keys, never altered ones; the
  byte-identity tests in `tests/test_per_channel_skill.py` and
  `tests/test_roll_monthly_identity.py` must keep passing) to ALSO dump the
  sums per (holdout year, start), so a scorer can resample blocks. Blocks =
  (year, start) pairs; resample with replacement 2,000 times; report the 5–95
  % interval on every `chan_skill` field and on the differences between two
  named blocks (head vs null, head vs head). Store under a new
  `intervals` key. Do it for the transport bands too (block = year).
- **Rolling-origin blocked CV for development.** Add `--holdout-years` as
  an override to the stage-2 pool AND to the roller's start selection, so a
  head can be trained with e.g. {2005, 2011, 2017, 2023} held out and rolled
  from those years, under `window` scope. Define the DEVELOPMENT split as
  four folds of one held-out year each, spaced ≥ 5 years apart, all ≤ 2020,
  so the terminal years are never touched during development. (This needs
  the codec question in WP3 answered first: the codec's `holdout_years`
  currently decides the pool's.)
- **Minimum detectable effect.** From the bootstrap spread of a head-vs-null
  difference, compute and publish, per lead and per scope, the smallest
  difference the battery can distinguish from zero at 90 %. Put the table in
  `docs/ML_BASICS.md` §"Metrics and their statistical power" (the paper's
  §metrics is the same text). Several open questions may be formally
  unresolvable at this sample count; that table is how one says so.

**Done when.** #516's artefact re-scored with intervals; the E-064 table
carries them; the MDE table exists; `tests/` pins that intervals are
additive (old keys byte-identical).

**Falsifier.** If the bootstrap interval on the corridor `acc` at h=18 is
wider than ±0.15, the interspersed three-year battery cannot rank heads at
the month-plus horizon at all, and R1 must be re-planned around the terminal
holdout with more starts per year (up to 73) rather than more arms.

**Cost.** Code: 1–2 sessions. Compute: none beyond re-scoring.

### WP3 · The codec ruling, and the terminal-holdout codec (E-063)

**Goal.** Decide, and then make true, that nothing in the stack has seen the
test years.

**Ruling, stated now so nobody re-derives it.** `run-415__pixelmae.pt` was
trained by `ml/train.py` with its default `--holdout-years 2009,2017,2023`
(verified in `probes-415.json`'s provenance and `train.py:151`), which
excludes those years from the codec's self-supervised training and from the
anomaly statistics. **So for the INTERSPERSED protocol the codec is clean.**
For the TERMINAL protocol (test 2021–2024) it is not: it reconstructed 2021–
2024 during training, and the stage-2 pool inherits `hold_years` from the
codec checkpoint, so a terminal-holdout stage-2 head CANNOT be trained on
this codec at all without an override that would itself be a leak.

**Build.** E-063 — a fresh codec at run-415's exact architecture and data
(`family4_na025_pentad_r2`, 512×12, 4 heads, d_dec 256, d_z 32, patch 1,
`--holdout-lon 0,0`), with `--holdout-years 2021,2022,2023,2024`. The
anomaly/standardisation statistics must come from ≤ 2020 as well (they
follow `t_hold`, so they will). `ml/jaxport/train_stage1.py` exists and takes
`--holdout-years`; the torch path is `ml/train.py`. Budget as run-415 was
budgeted (200k × 512, cosine), because the codec is not where the early-peak
problem lives — but DO record its held-out reconstruction curve and check
whether it, too, plateaus early; if it does, WP4's cap applies to codecs as
well. Publish as `run-<n>__pixelmae.pt`; embed the tensor with it and publish
the Z (`embed-cache-v1`), since every terminal-holdout head will need it.
While it trains, run WP1–WP2 on the interspersed split; nothing there waits
for it.

**Done when.** A published codec whose `args.holdout_years` is
`2021,2022,2023,2024`, its Z on the release, and a one-line E-063 entry with
its held-out reconstruction and its `probe_head` on RAPID (for the record;
not a verdict).

**Falsifier.** None needed — this is infrastructure. But pre-register: its
reconstruction on 2021–24 should be no worse than run-415's on 2009/2017/2023
by more than the seed spread of codecs (unknown; this is the first pair —
budget a second seed if the number will be quoted).

**Cost.** One codec training (~19 h on the class of hardware run-415 used) +
one embed pass.

### WP4 · The cost reform: short budgets, checkpoint selection, seeds ★ (a policy, then a recipe)

**Goal.** Make every stage-2 arm cost ~1.5 h at 7.6M so that five seeds are
the default and the R1 programme runs in days.

**Build.**

- New stage-2 recipe(s) in `ml/recipes/` derived from E-060a's knob block
  (7.6M: 256×8, K 144, stencil 145, ring `spiral:111-4444-0.71-0.5`, batch
  256, lr 1e-3, `expdecay` halflife 40,000, warmup 2,000, znoise 0.7,
  grad-clip 128, `--train-lon-hold none`, `holdout_scope window`) with
  **`STEPS` 5,000 and `CKPT_EVERY` 500**. Keep the LR schedule's halflife at
  40,000 so the first 5,000 steps trace a bit-identical LR trajectory to
  E-059/E-060 — that keeps every archived early curve comparable.
- **Checkpoint selection rule, written down once:** the head that gets
  rolled is the checkpoint with the minimum held-out one-step ratio over
  steps ≥ 500, ties to the earlier step. The trainer must save every
  `CKPT_EVERY` checkpoint (not only the latest) and the selection must be
  done by a script from `metrics.jsonl`, not by eye. Publish the selected
  head as `head-weights-<exp>-best<step>-<scope>-s<seed>.pt` and ALSO the
  step-5,000 end state, so "does early stopping matter for the ROLL" is
  answerable per arm at no extra cost.
- **Seeds.** Five seeds per configuration is the default at the 7.6M tier
  (§3b's mandatory-replicate clause applies: pentad cadence has no measured
  pair for rolled skill). Report mean and range; a claim needs the range of
  one configuration to clear the range of the other.
- The 200k-step budget is retired for stage 2 unless an arm's held-out curve
  is still falling at 5,000 — pre-register that check and extend only then.

**Done when.** Recipe committed and pinned by `tests/test_train_config_guards.py`;
the selection script exists with a test; the policy line is in
`ml/CLAUDE.md` §1 ("size the job against its own timeout" gains "and against
where the held-out minimum actually is").

**Cost.** Code only.

### WP5 · The cheap evaluation quartet, now affordable

All on the INTERSPERSED split, so they compare directly to #516 and to WP1's
nulls; all pre-registered in one E-065 entry.

1. **The step-2,000 roll (Q3).** There is no early checkpoint in the bucket
   for E-059 or E-060a (verified 08-31: each prefix holds only the final
   `.pt`, its `_jax.npz`, `metrics.jsonl` and one log). So: train the 7.6M
   recipe of WP4 to 5,000 steps with `CKPT_EVERY` 500, select per the rule,
   roll BOTH the selected checkpoint and the step-5,000 one through #516's
   battery. Expectation: the selected checkpoint's `acc` at leads 2–18 is
   higher than the 20k head's (#R0b) by more than the WP2 interval; falsified
   if they agree. Either answer matters: agreement says the 20k/200k end
   states did not lose rolled skill by over-training, and the early-peak
   story is about the one-step loss only.
2. **Five seeds of that arm.** The first replicate set at pentad cadence
   for rolled skill. Deliverable: the tier's spread on `horizon_auc`, `acc`
   per lead and the transport bands — the number §3b needs before any 7.6M
   comparison can be called a level.
3. **Amplitude calibration as a decoding option.** Add `--calibrate` to the
   roller: per lead h (and per channel), multiply the rolled anomaly by a
   factor `a*(h)` fitted on TRAINING-year starts only (the `acc`-optimal
   scaling is `a* = ACC_train(h)`; also try the MSE-optimal regression slope
   `cov(x,y)/var(x)` on training years). Write the calibrated rows as new
   fields (`msss_clim_cal`, `amp_ratio_cal`) beside the existing ones, never
   instead. The identity `msss = 1 − (1 + a² − 2a·ACC)` predicts #516's
   corridor mean goes −0.439 → about +0.02; falsified if the calibrated
   number, fitted on training years and applied to the holdout, is not
   within the WP2 interval of that. Apply it to the nulls too — a calibrated
   LIM is the fair comparison for a calibrated head.
4. **Drop the 32 `rg_*` channels from the INPUT** (a modelling experiment,
   separate from the storage change in WP7): train the same 7.6M arm on the
   8 scoreable channels only (codec unchanged for now — mask the channels as
   missing tokens at stage-2 time if the code path allows; otherwise this
   waits for a re-embed with an 8-channel codec and moves behind WP3).
   Expectation: no loss of rolled skill; falsified if `acc` at h ≤ 6 drops by
   more than the seed range.

**Cost.** ~6 short trainings (~1.5 h each) + ~6 rolls (~20 h each on a 4090,
less for a 7.6M head — measure the first one and re-price; consider a
shorter battery, `longm:0,futm:0`, for development rolls since the 36-month
blocks answer nothing for a deterministic head).

### WP6 · The R1 re-ranking, under the new harness — only after WP1–WP5 read out

Cadence → stencil → unroll → znoise → FGN → width, at 7.6M, five seeds,
short budgets, each rung asked "does it beat the CV-chosen LIM by more than
the interval, at which leads, on which channels" — never "does it beat last
week's arm". The FGN/ensemble thread is where the dispersion test (WP0) and
the calibration-versus-sharpness question live; it re-enters here, not
before. Anything at 206M+ needs a specific argument written in its entry.

### WP7 · Data — clean before adding, add only what touches the wall

In this order, each as its own small entry with a measured before/after:

1. **`rg_*` to a 1° sidecar** (storage): frees ~27 GB, more than every
   proposed import combined. Loader upsamples on read. Verify the tensor
   round-trips byte-identically for the 8 kept channels.
2. **2025–2026 continuation** of the pentad tensor (+4.5 % end-bins, nearly
   free; lets the terminal test run to 2026 later).
3. **Backward extension toward 1958** (EN4.2.2 or IAPv4 as a 1° T/S-only
   tensor, ~3 GB). Run AS A MEASUREMENT: train the WP4 arm with and without
   the extra years and ask whether the held-out minimum and the rolled
   `acc` move. It is the only item on the ladder aimed at the 2,417-bin
   constraint; whichever way it answers is worth having.
4. **ERA5 surface heat/freshwater flux** (needs a CDS account): missing
   physics, not more samples. Worth doing; will not move the wall; say so.
5. **GREP 3-member** as low-bias augmentation, again as a measurement.
6. **CMIP6 (family 6): codec pretraining ONLY, never the forecaster**, with
   the mandatory control (embed the reanalysis with the same 37 channels
   masked). Last in the queue. The corpus exists (see the #517 record);
   publishing it needs the `HF_TOKEN` Actions secret, a human step.

Declined, and the reason to quote at anyone who proposes them: 1/12°, MUR
1 km, ARMOR3D native — more pixels of the same 2,417 bins.

### WP8 · The paper ★

- Put an explicit invalidation note at the top of every results section
  that quotes a pre-`c25f6ff` rolled number ("corridor AUC" values are
  `msss_clim` from contaminated heads; retired). Do it now, before WP1 lands,
  so no reader takes them as current.
- Add Acknowledgements and Data Availability (CMEMS/GLORYS/GREP licence
  terms, Argo, NCEP R1, OISST, GPCP; CMIP6's required model/institution table
  if family 6 is ever used in a reported number).
- Once WP1–WP2 read out, the paper's centre of gravity is likely "the limits
  of learned subseasonal-to-seasonal prediction from a 43-year reanalysis:
  where a transformer beats classical nulls, where it does not, and what
  the sample count can resolve". Draft that outline when E-064's table
  exists, not before.

---

## 3. Sequence and dependencies

```
now ──► WP0 (R0b re-dispatch) ──────────────────────────────┐
        WP1 nulls ★ ──┐                                      │
        WP2 harness ★ ─┼─► E-064 table with intervals ──► DECISION GATE 1
        WP8 quarantine ★                                      │
        WP4 recipe/policy ★ ──► WP5 quartet (interspersed) ──┘
        WP3 codec E-063 (long; start early, nothing waits on it)
DECISION GATE 1 ──► headline decision (§4) ──► WP6 R1 + WP7 data, on the
                    terminal-holdout codec once E-063 is published
```

WP1, WP2, WP4, WP8 are pure code and can run in parallel in separate
sessions. WP3 is the long pole and should be started first. WP5 needs WP4's
recipe and is the first accelerator spend after the reboot. WP6/WP7 wait for
Gate 1.

**Decision gate 1 (after E-064 with intervals):**

- If the LIM ≥ transformer at every lead ≥ 2: R1 is re-scoped to "what does
  the transformer add at h = 1–2 on SST/SSH, and does any change let it beat
  the LIM at h ≥ 3". The LIM becomes the reference model of the paper.
- If the transformer beats the LIM at short leads by more than the interval:
  R1 proceeds as listed, with the LIM as the standing bar.
- If the MDE table says the interspersed battery cannot resolve ±0.1 in `acc`
  at h ≥ 12: development moves to the terminal codec with more starts per
  year before any further ranking.

---

## 4. The open headline decision (Chris's)

Recommendation, with the reasoning in one paragraph: pivot the headline from
AMOC transport at 26.5°N to field-level subseasonal-to-seasonal skill
(SST / SSH / MLD) against classical nulls, with AMOC kept as a bounded,
interval-carrying secondary read-out and the predictability/power bound
written up as a result in its own right. The AMOC question has ~9 effective
starts and cannot be resolved by this data with any model; the field
question has thousands, and R0 already localises where the signal is.
Nothing in WP0–WP5 depends on this decision; WP6–WP8 do.

---

## 5. How to report

Every status report uses `ml/CLAUDE.md` §0f's four sections (completed ·
new results and next steps · queued · proposed changes), each experiment
with a plain-English TL;DR before any number, and every link as a markdown
link on its own line pointing at `docs.html?f=…` or the status page, never a
bare Actions URL.

---

## 6. Pitfalls a new session should expect

- `horizon_auc` is `msss_clim`, not an AUC; negative means the error exceeds
  the anomaly variance, not "below chance".
- `msss_pers` is RAW persistence; `msss_damped` is the AR(1) null. Quote
  both, and say which. The clean head beats the first and loses to the
  second.
- The evaluator's `chan_skill` is pooled over channels; `per_channel` has
  the decomposition. Only 8 of 40 channels ever score.
- The stage-2 pool's hold years come from the CODEC checkpoint's `args`; the
  terminal protocol needs a terminal codec (WP3).
- `_trainlon` equals the parent on the pentad r2 tensor (no longitude hole);
  no spatial-generalisation claim can be made from it.
- A green run with an empty bundle, a red run with all its goods, a scratch
  copy that fills a disk: read the artefact list and the log, not the
  colour.
- Write the entry at dispatch. If the entry does not exist, the run is not
  dispatched.

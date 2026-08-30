# The protocol reset · what may count as a result after 2026-08-29

**Status: PROPOSAL, written 2026-08-29 ~17:4xZ.** Chris, this session:
*"No need to compare models on the contaminated data. Let's do a proper
comparison. Also: I feel we need to reboot the programme as until now we just
evaluated how much for the training data the stage 2 can learn by heart. It
feels we need smaller models, and we need to re-evaluate all dimensions as the
previous eval results could not be trusted."*

This document is the standing answer to "what is a proper comparison". §2 is
the evidence, §3 is the protocol to freeze, §4 is the re-ranking programme,
§5 is what is retired. Sections 3 and 4 displace queued work and are
proposals until Chris answers (ml/CLAUDE.md §0f.4); §2 and §5 are readings of
artefacts already on `ml-metrics` and stand on their own.

---

## 1. The one correction to the framing

**"Smaller models" is right for cost and wrong for cause.** The width ladder
already tested it. **AMENDED 2026-08-30**, when the E-060 read-out landed and
falsified its own pre-registered prediction — the claim survives in a narrower
and better-stated form. E-060a at **7,597,856** parameters reaches a best
held-out one-step loss of **0.60951** (step 1,200) against E-059's **0.61049**
(step 2,000) at **206,658,592** — 0.001 apart across a 27× span, and every arm
on the ladder bests in 0.58–0.62 across a 53× span. **Both peak in the first
~2,000 steps and get worse for the next 198,000.** What is NOT true is that
they match at a fixed later step: at 20,000 the small arm is 0.051 worse, and
the ordering is not even monotone (40.4M beats 206.7M there). So width buys how
fast an arm reaches the level and how badly it then degrades, not the level.
Shrinking the head does not remove the failure; it makes the failure cheap to
observe — and the read-out damage runs the other way, with 7.6M holding RAPID
at 0.598–0.611 across ten probes while 206.66M collapses 0.616 → 0.515.

The binding constraint is named in the E-061 plan and it is not capacity:
the pool's `209,549,066` train windows is **2,417 end-bins × 86,698 ocean
pixels**. The temporal diversity of the entire training set is **2,417
windows** — 43 years of ocean, 20 of RAPID. Against that, 7.6M parameters is
already over-parameterised, and 206M is 27× more of the same mistake.

So: adopt small heads as the DEFAULT TIER because they make the re-ranking
affordable (§4), not because size is the disease.

---

## 2. Four independent memorisation signatures, each from an artefact

Every number below is from `ml-metrics`, harvested 2026-08-29.

**READ THIS BEFORE THE NUMBERS. The quantity called "corridor AUC" throughout
this programme is NOT an area under a curve.** It is `horizon_auc` = the mean of
**`msss_clim` = 1 − MSE_model / MSE_climatology** over the leads, and because the
fields are already anomalies against the train-month climatology
(`mse_c += (v_true**2)`, `ml/rollout_spatial.py:1154`), "climatology" is the
forecast *zero anomaly*. So **1.0 is perfect, 0.0 is exactly as good as
predicting no anomaly, and a negative value means the model's squared error
exceeds the anomaly variance.** `ml/rollout_spatial.py:117` has said this since
E-017 ("NOTE the metric's name..."). A value below 0.5 is not "below chance";
a negative value is not "the ranking inverts". The key name is left alone —
183 archived bundles use it — but nothing may reason about it as an AUC again.
This paragraph was added 2026-08-30 because §2c below had done exactly that.

### 2a · The pool bug (known, fixed, retired-with-caveat)

`--holdout-scope endpoint_contaminated` supervised **21,018 per-frame targets
that were held-out bins**. Fixed in c25f6ff; default flipped in 58eb286.
Every stage-2 head archived before c25f6ff — all 98, monthly and pentad —
carries it. Recorded in E-059.

### 2b · Skill that does not decay with lead — #510

`head-weights-e051-398k-xl144zn-pentad-s0` (E-051, 398k steps, pentad),
rolled 2026-08-28→29, 22 h 47 m, `probes-510.json`:

| band | lead | r |
|---|---|---|
| h1–18 | 5–90 d | **0.511** |
| h19–36 | 95–180 d | **0.591** |
| h37–73 | 185–365 d | **0.565** |

Genuine forecast skill decays with lead. A profile that is flat — or that
*rises* from 90 d to 180 d — is a recall curve. Corridor `horizon_auc` 0.888 on
the same roll. #503 said the same thing at 0.944 before it was cancelled.

**AMENDED 2026-08-30, and this is the sharper form of the same signature.**
The `acc` column of `chan_skill` — the anomaly correlation between the rolled
and the true FIELD, per lead — was in the artefact all along and had never been
read. On #510 it is **0.985 at h=1 (5 d) and 0.973 at h=73 (365 d)**, mean
0.946: a near-unity field correlation twelve months out, flat across the whole
year. Nothing in ocean physics forecasts a field anomaly at 365 days with
r = 0.97. The transport bands were a muffled version of this; the field
correlation states it without ambiguity. The clean-pool control is E-062-R0
(#516), which on the identical battery decays **0.606 → −0.031**.

### 2c · No spatial generalisation at all — #513

The FGN seed-0 head (`head-weights-e057fgn_s0`, 279.6M, M=8 ensemble mean),
`probes-513.json`. The 20°-wide longitude block −45…−25 is held out of
training in **both** stages, so this is a system-level read, not a stage-2
read:

| scope | trained lons | held-out lons |
|---|---|---|
| gate | 0.821 | **0.330** |
| corridor | 0.838 | **0.176** |
| window | 0.834 | **0.266** |

And the deterministic gate head `e017_u1_s0` on the same roll: corridor
**0.804 → 0.058**. Under the metric's true definition (above) that reads: on
trained longitudes the head removes ~80 % of the anomaly variance; on the
held-out block it removes ~6 %, i.e. it is **barely distinguishable from
predicting no anomaly at all**. An earlier version of this paragraph called
0.058 "anti-skill, the ranking inverts" — that was an AUC reading of an MSSS
and is withdrawn. The FINDING is unchanged and is arguably cleaner: on ocean it
was never trained on, the system retains essentially nothing.

The published aggregates (corridor 0.648 for FGN, 0.589 for the gate) are
**blends** of these two populations, ~24 % untrained pixels. The blend is
deflationary, so every headline aggregate this programme has quoted is a
lower bound on its own memorised half and tells you nothing about
generalisation. Report the split, never the blend.

### 2d · The ensemble is confident exactly where it has memorised — #513

`long_dispersion` rolls 36 months inside the record (2005-01→2007-12);
`future_dispersion` rolls 36 months past its end (2025-01→2027-12). Same
head, same M=8, no truth in either:

| | first 12 mo | last 12 mo | growth |
|---|---|---|---|
| inside the record · transport spread | 0.1060 | 0.1015 | **flat** |
| inside the record · field variance | 0.0092 | 0.0269 | ×2.9 |
| past the record · transport spread | 0.1974 | 0.4197 | ×2.1 |
| past the record · field variance | 0.0443 | 0.3242 | ×7.3 |

Eight independently-noised trajectories stay pinned together over years the
model has seen and fan out over years it has not. That is the memorisation
signature stated in the ensemble's own units, and it needed no labels.

Separately, and worth fixing on its own: the ensemble is **badly
underdispersed** everywhere — spread/RMSE 0.25–0.28 on the corridor,
0.36–0.44 on trained longitudes, against ~1.0 for a calibrated ensemble. The
FGN machinery works (the noise channel is alive, dispersion grows with lead
in the future roll) but its spread is 2–4× too small to be a usable
uncertainty.

---

## 3. The protocol to freeze

Fix these once, before any arm runs, and do not re-litigate them per
experiment. A number produced outside this protocol is not a result.

**3.1 · Terminal holdout, not interspersed years. DECIDED 2026-08-30
(Chris): train ≤ 2020, test 2021–2024, NO GAP.** Today's `hold_years`
(2009, 2017, 2023) are single years surrounded by training years. Three
things are wrong with that, and none of them is leakage: predicting 2009 from
a memorised 2008 *and* 2010 is **interpolation between two anchors**, where a
terminal block has no future anchor and forecasting is extrapolation; AMOC
variability is **multi-year to decadal**, so a single held-out year's answer
is largely present in its neighbours and only a contiguous block removes a
chunk of the slow state; and a terminal block is **the use case** — a forecast
forward from now.

**The gap this section originally asked for is WITHDRAWN.** It was
over-caution. Under `--holdout-scope window` no training window touches a
held-out bin, so with 2021–2024 held out the last training target is 2020-12
and no training target lies in the test period — there is no target leak for a
gap to close. What a gap would remove is the advantage of *starting* from a
memorised state, and that is not an artefact: operationally a forecast always
starts from a well-observed recent past. The residual worry, that memorised
initial conditions make the first leads easy, is what the persistence baseline
in §3.4 measures — a control, not a reason to discard a year of data.

**Why not 2022–2025:** the tensor ends **2024-12-31** (`build_family4.END`),
so 2025 does not exist and 2022–2025 would be three test years plus a year of
nothing. 2021–2024 is the last four years the record has. Extending the axis
to 2026 is a data-import task (§4-R2 and the data ladder), worth ~+4.5% of
end-bins.

**3.2 · The trained/untrained split is the headline, never the blend.**
Every scope reports `_trainlon` and `_holdlon`. The parent aggregate stays in
the artefact and stays out of the prose.

**3.3 · Skill-vs-lead decay is a falsifier, not a diagnostic.** A run whose
band correlations do not decrease with lead has failed, whatever its AUC.
Register the expected decay before the roll.

**3.4 · Every number carries its null ladder.** Persistence, climatology,
damped persistence, the wind-stress ridge (**0.531** on 1°, **0.568** on
quarter-degree, already measured), and one addition: a **nearest-analogue
retrieval baseline** — for each initial state, return the continuation of its
closest training-set analogue. That is memorisation with no parameters at
all. A head that does not beat retrieval has not learned dynamics, and no
other number about it is interesting.

**3.5 · Early-stop at the held-out minimum.** E-059 and E-060a both bottom
out at ~step 2,000 and get worse for the next 198,000. The programme has been
rolling the memorised end state. Roll the early-stopped checkpoint;
checkpoint at the minimum and publish that.

**3.6 · Rolled skill is the verdict; z-space loss is not.** Stated already in
E-060 §7 and violated by habit. No arm is promoted on a probe number.

---

## 4. The re-ranking programme

**Nothing rolled so far used a clean-pool head.** #503, #508/#510 and #513
all rolled heads trained under `endpoint_contaminated`. The first honest
rolled number in this programme's history does not exist yet. That is the gap
to close first, and it is one dispatch.

**R0 · The first honest roll — RAN as #516; skill battery landed 2026-08-30
21:42Z. VERDICT: the clean-pool head DECAYS with lead (field `acc` 0.606 at 5 d
→ −0.031 at 365 d) where its contaminated twin is FLAT AT 0.97, so the
pre-registered falsifier separates them and the pool was the dominant artefact.
Corridor `horizon_auc` +0.888 → −0.439; the negative value is a CALIBRATION
failure, not anti-skill — mean `acc` 0.105 against mean `amp_ratio` 0.780, and
the identity msss = 1 − (1 + a² − 2a·ACC) reproduces all 73 leads to 0.0135, so
the same rolled states amplitude-calibrated would score +0.019. Full reading:
`ml/EXPERIMENTS.md#e-062` §(a)–(h).** `temporal_e059.pt` (206.66M,
`window` scope, 200k) and `temporal_e060a.pt` (7.6M, `window` scope, 20k)
through the same `sroll` ladder that produced #510 and #513 — two heads, one
job, `longm:36,futm:36`. Cost ≈ one #510 (~22 h, ~$6). It answers two
questions at once: what does a clean-pool head actually forecast, and does
27× less capacity forecast worse. Caveat to register at dispatch: the two
arms differ in training budget (200k vs 20k) as well as width; per §3.5 both
are also past their held-out minimum, so this is a floor, not the arms'
best case.

**R1 · Re-rank the dimensions at the small tier.** Every axis this programme
ranked — width, depth, stencil/ring, unroll, znoise dose, monthly vs pentad
cadence, FGN noise conditioning — was ranked on the contaminated pool and
must be re-ranked. At E-060a's tier (7.6M, ~1 h/arm on v5e-4 spot) the whole
ladder is affordable in a way it never was at 206M. Order by how much each
axis cost to get wrong: **cadence → stencil → unroll → znoise → FGN → width**
(width last, because E-060 has already shown it is not the axis).

**R2 · Attack the data constraint.** E-061 (800 model-years of CMIP6
`piControl`, built by #514) is the only live work aimed at the actual
bottleneck, and it stays. Its registered confound — a 3-of-40-channel corpus
is a distribution shift as well as a transfer — needs its control arm
(embed the reanalysis with the same 37 channels masked) budgeted from the
start, or the result is uninterpretable either way.

---

## 5. What is retired, explicitly

- **Every rolled number from a pre-c25f6ff head**, as evidence about forecast
  skill. They remain in the log as measurements of recall.
- **E-057's F1.** The FGN falsifier was "ensemble-mean corridor AUC vs the
  znoise pair 0.7235 / clean pair 0.6781". All three sides sit on the
  contaminated pool, so the contrast is internally consistent and externally
  meaningless. For the record it did not win — 0.648 against 0.678 and 0.7235
  — but that comparison is withdrawn rather than reported. What survives from
  E-057 is machinery, tested and passing: the noise channel does not collapse
  (member_var 0.078 at 200k), dispersion grows with lead outside the record
  (§2d), and fair-CRPS training at N=2 is implemented and runs. Its
  calibration is bad (§2d) and that is a finding about FGN, not about the pool.
- **The capacity ladder as a skill question.** E-054/E-060 answered it: width
  is not the axis. E-054b's 400M rung inherits the pool caveat and is not
  re-run.
- **Seed 1 of E-057** (#502, dead on an offline host). Under §3b it was owed
  only if FGN won. It did not, and the comparison is withdrawn regardless.
  Not re-run. Vast `48937793` holds its only head copy; nothing depends on it.

---

## 6. Decisions taken, and the one open blocker

All three questions this section originally asked were answered by Chris on
2026-08-30:

1. **Terminal holdout: train ≤ 2020, test 2021–2024, no gap** (§3.1, rewritten
   above with the reasoning).
2. **R0 runs now**, on today's holdout, because the heads already exist —
   dispatched as **#516**, see `ml/EXPERIMENTS.md#e-062`. The retrains under
   the new split follow at the small tier.
3. **Small tier is the default.** 206M+ arms now need a specific argument.
   E-060a's 7.6M rung is the working default.

**The one blocker.** R0 rolls only the 206.66M arm. The 7.6M arm
(`temporal_e060a.pt`) lives in the TPU results bucket and the roll evaluator
reads heads from the `model-checkpoints-v1` release, so the head must be
mirrored across — a sandbox-side step needing the GCS read credential, which
this session was unable to obtain (the repo's own `secret-handoff` workflow
could not be dispatched from here). E-059's head was rollable only because
another session had already published it. Until E-060a is mirrored, the
small-vs-large half of R0 cannot run, and neither can any new TPU training
under the terminal holdout, since node creation needs the same credential.

---

## 7. R0's verdict, and what it changes in this document (2026-08-30)

The first honest rolled number in this programme's history exists. It is
E-062-R0 / #516, and its skill battery is scored and final (the roll is still
running its dispersion phases; `ml/CLAUDE.md` §5.25's `in_progress` caveat
applies to nothing quoted here).

**What it confirms.** §5's retirement of every pre-c25f6ff rolled number is now
a measurement rather than an argument: one variable — the training pool —
moves the corridor from +0.888 to −0.439 and the 365-day field correlation from
0.973 to −0.031. §3.3's lead-decay falsifier does the job it was registered to
do, and does it on `acc`, which is the instrument to quote for it from now on.

**What it changes.**

- **§3.3 gets a named instrument.** "Register the expected decay before the
  roll" now means: register it on `chan_skill[*].acc`, per scope, and report
  the h=1 and h=H values with the mean. `horizon_auc` cannot serve as the
  decay instrument because it confounds correlation with amplitude (§(d) of the
  E-062 entry).
- **§3.4's null ladder gains a member that costs nothing: the
  variance-calibrated version of the arm itself**, `ACC²`. Any arm scoring
  below its own `ACC²` is losing to arithmetic, not to a baseline, and the fix
  is calibration rather than capacity or data.
- **§3.2 is unreadable on the pentad r2 tensor.** It has no longitude hole
  (`holdout_lon "0,0"`), so `_trainlon` equals the parent and `_holdlon` is
  empty by construction. The split stays the headline rule wherever a hole
  exists; on this tensor the honest report is that the split was not measured.
- **A new item joins §4 ahead of R1: roll the step-2,000 checkpoint** (§3.5).
  R0 measured the 200k memorised end state, which §3.5 already says is the
  wrong checkpoint to roll. It is the cheapest untried thing in the programme.
- **§1's correction is reinforced from a new direction.** The gap between
  −0.439 and the +0.019 the same states would score if amplitude-calibrated is
  not a capacity gap and not a data gap. It is a decoding gap, and it is
  available at any tier.

**What it does not change.** The terminal-holdout retrains (§3.1) still stand,
for the reasons given there — extrapolation rather than interpolation, the
multi-year slow state, and matching the use case. R0 does not rescue them and
does not displace them. The GCS blocker in §6 is untouched: the 7.6M arm still
cannot be rolled, so the small-vs-large half of R0 is still owed.

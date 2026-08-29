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
already tested it. E-060a at **7,597,856** parameters reproduces E-059's
plateau at **206,658,592** — the best held-out one-step loss arrives at
**step 2,000** in both, 27× apart in capacity, and everything after is
memorisation. Shrinking the head does not remove the failure; it makes the
failure cheap to observe.

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
*rises* from 90 d to 180 d — is a recall curve. Corridor AUC 0.888 on the
same roll. #503 said the same thing at 0.944 before it was cancelled.

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
**0.804 → 0.058**. Below 0.5 is not "weaker skill", it is anti-skill: the
ranking inverts. On ocean it was never trained on, the system does not
degrade gracefully — it fails.

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

**3.1 · Terminal holdout, not interspersed years.** Today's `hold_years`
(2009, 2017, 2023) are single years surrounded by training years. Under
`window` scope no training window *touches* them — that part is now correct —
but the evaluation still asks the model to predict 2009 given a perfectly
memorised 2008. For a quantity with multi-year memory that is close to
free. **Train on a contiguous prefix, test on a contiguous terminal block,
with a gap of at least one context window (K=144 pentads ≈ 1.97 y) between
them.** It is the harder test and it is the actual use case: forecasting
forward from now.

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

**R0 · The first honest roll (do this first).** `temporal_e059.pt` (206.66M,
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

## 6. Open decisions for Chris

1. **The terminal-holdout split** (§3.1): where to cut. Train ≤2018 / gap
   2019 / test 2020–2024 costs ~5 years of training data and buys the only
   honest forward test. A shorter test block keeps more training years.
2. **Whether R0 runs before or after the protocol change.** R0 as written
   uses today's interspersed holdout, because those two heads already exist
   and rolling them costs one job. Under §3.1 they would both need retraining
   first (~1 h each at the small tier, ~13 h at 206M).
3. **Whether the small tier becomes the default for everything**, with 206M+
   arms requiring a specific argument rather than being the norm.

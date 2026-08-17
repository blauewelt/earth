# E-038 · Retrain the codec on the new cadences — a 2×2 matrix

**Status: PLANNED.** Chris, 2026-08-17: *"let's change the plan to retrain the
codec (40m and 200m?) on the new data (1 day, 5 days). So a 2x2 matrix, 4
codecs need training. Only with the new codecs do we then re-encode the data
into embeddings. (the reason is that we have a new kind of data that is out of
domain for the existing codec)"*

This document records the decision and its arithmetic before anything is
dispatched, so the reasoning cannot be reconstructed from whatever the runs
happened to do.

---

## 1 · The call, and why it is right

**The ordering changes.** The programme's habit has been: freeze the codec,
embed, train stage 2. Every run for six weeks froze the codec via
`resume: "!run-62,run-63"`. Applied to pentad or daily fields that would embed
**out-of-domain data with an in-domain codec**, and every downstream number
would be measuring the codec's extrapolation rather than the cadence's value.

New order:

1. aggregate the cadence (E-034 step 3½ — pentad running 2026-08-17)
2. build the tensor (`build_family4.py`, E-034 step 4)
3. **train four codecs from scratch** — this document
4. **only then** re-encode embeddings, with the matching codec
5. stage 2 / the transport probes

**The out-of-domain claim is not a vibe, and it should be measured rather than
asserted.** Three distinct shifts, in descending order of how badly they break
a monthly-trained codec:

- **The missingness pattern inverts.** This is the big one. RG-Argo is a
  monthly product, and E-034 §4 chose one live timestep per month with a
  learned `missing` token elsewhere. So on the RG channels:

  | cadence | steps carrying RG | steps carrying `missing` |
  |---|---|---|
  | monthly (f3) | ~100% post-2004 | ~0% |
  | **pentad (f4)** | **~17%** (1 in 6) | **~83%** |
  | **daily (f5)** | **~3%** (1 in 30) | **~97%** |

  32 of 39 channels are RG. A codec that has essentially never seen the
  `missing` token on those channels is being asked to spend most of its
  capacity on it. The architecture's whole claim is that *missingness is
  information* and that the `missing` token is distinct from `mask` by
  design — this is exactly the regime where that distinction is load-bearing,
  and exactly where a frozen codec has no representation for it.

- **The variance moves into a different band.** A 5-day mean keeps the storm
  band that a monthly mean averages away; a daily field keeps all of it. The
  fields are not a rescaling of the monthly ones, they are a different
  spectrum. `tau_std` changes meaning outright — family 3's is a within-month
  σ (storm band mixed with the seasonal cycle), family 4's is a within-pentad
  σ, which is a storminess measure.

- **The normalisation differs.** `build_family4.py` computes its own `norm`
  from its own data, as it must. A codec carrying family 3's implied input
  scale is mis-calibrated before the first layer.

**What would falsify the premise** (ml/CLAUDE.md §1): if a freshly trained
pentad codec scores no better than the frozen monthly codec applied to pentad
data, the out-of-domain argument is wrong and the cadence work should go back
to reusing the codec. **That control is cheap and must be run**: embed the
pentad tensor with the existing frozen codec once, and keep it as the baseline
every new codec is reported against. A number without its baseline is not a
result (§3).

---

## 2 · The matrix

**40M and 200M are CODEC PARAMETER COUNTS** — `PixelMAE` in `ml/model.py`,
not tensor sizes, not head widths, not months of data.

|  | **pentad (5-day)** | **daily (1-day)** |
|---|---|---|
| **~40M codec params** | f4-40M | f5-40M |
| **~200M codec params** | f4-200M | f5-200M |

Measured by instantiating the real class at `n_chan=39`, rather than
estimated:

| `d_model` | `n_layers` | `d_dec` | codec params |
|---|---|---|---|
| 512 | 12 | 256 | **37,975,889** ← the ~40M rung |
| 640 | 12 | 320 | 59,286,161 |
| 768 | 12 | 384 | 85,323,217 |
| 896 | 16 | 448 | 154,668,817 |
| 1024 | 16 | 512 | **201,962,577** ← the ~200M rung |

**The 40M rung is the CURRENT codec's capacity, and that is what makes this a
clean experiment.** `f3_anchor41M` (run #62) is **40.7M params** — the frozen
codec everything since has ridden on. So the f4-40M cell holds architecture
and capacity fixed and changes *only the data*. If it beats the frozen 40.7M
codec applied to pentad fields, that difference **is** the domain shift, with
capacity controlled — which is exactly the claim §1 makes and exactly what a
single-arm test could not have isolated. The 200M row is then the independent
scale question, asked on top of a control that already holds.

Two axes, deliberately: capacity × cadence. It answers "does finer cadence
pay?" and "does the answer depend on capacity?" in one design, which matters
because **wave 6A already showed the two interact** — znoise-big55 at 88M beat
xl55 at 205M on the rolled corridor AUC, so "scale pays most" is false on that
axis and a single-capacity cadence test could easily mislead.

### 2b · The Chinchilla anchor, PREDICTED per cadence

Chris, 2026-08-17: *"let's use it to add a good prediction on the necessary
number of params in the codec."* Derived from family 3's inventory as
**measured at build time** (`ml/SCALING.md`, 2026-08-08) rather than
re-guessed — and the derivation self-checks, because the `base` and `wind`
group counts independently reproduce the recorded 84,405 ocean cells.

The scaling is per group, and it is not a single multiplier, which is the
whole point:

- **base** (3 ch) scales with the GLORYS steps: ×6.09 at pentad, ×30.4 at daily.
- **wind** (4 ch) scales with the full axis: same factors.
- **rg** (32 ch) **does not scale at all.** E-034 §4 puts *one live timestep
  per month* at every cadence, so finer cadence adds `missing` tokens on 32 of
  the 39 channels — not observed values. This is the same fact as §1's
  missingness table, seen from the counting side.

| cadence | T | base | rg | wind | total observed | **anchor = /20** |
|---|---|---|---|---|---|---|
| monthly (f3) | 516 | 97.2 M | 551.2 M | 174.2 M | 822.6 M | **41.1 M** |
| **pentad (f4)** | 3,142 | 592.0 M | 551.2 M | 1,060.8 M | **2,204.0 M** | **110.2 M** |
| **daily (f5)** | 15,706 | 2,959.6 M | 551.2 M | 5,302.7 M | **8,813.4 M** | **440.7 M** |

**The method reproduces the known answer**, which is why it is worth trusting
one step further: the monthly row gives 41.1 M and `f3_anchor41M` was built at
**40.7 M**. The anchor is not a new instrument, it is the one that already
sized the codec everything rides on.

**What this says about the rungs:**

- **Pentad: 40M and 200M bracket the 110 M anchor** (0.36× and 1.83×). That is
  a good pair and it should stand — one deliberately under-parameterised, one
  over, with the anchor between them.
- **Daily: both rungs sit BELOW the 441 M anchor.** The matrix does not
  bracket at daily; 200M is 0.45× the anchor. If the daily arm is meant to ask
  the same question the pentad arm asks, its top rung wants to be ~400–450 M,
  not 200 M. Worth deciding deliberately rather than inheriting the pentad
  pair — and worth noting that a 441 M codec is a materially bigger training
  bill than anything the programme has run.

**The caveat, which cuts harder at finer cadence than it ever did at
monthly.** `ml/SCALING.md` already warns to treat the anchor as a ceiling and
not a target, because the RG majority of the count is intrinsically 1°-smooth
and 0.25° cells are spatially correlated. Finer cadence adds a *second and
larger* deflation: **consecutive pentads are far more correlated than
consecutive months, and consecutive days more still.** The naive count says
6× the data; the effective independent-sample count grows by much less. So
110 M and 441 M are ceilings, and looser ones than 41 M was. Treat the pentad
pair as bracketing a soft target, and do not read "441 M" as a
recommendation — read it as the point beyond which more capacity is certainly
not data-sanctioned.

`build_family4.py` prints the real inventory at build time. **Check the
prediction against it** — if the printed total is far from 2.20 G, something
in the recipe differs from this arithmetic and that is worth knowing before
sizing anything.

---

## 3 · The blocker: the daily tensor does not fit anything we own

Measured, not modelled — `[T, 281, 481, 39]` at float16:

| tensor | T | fp16 | fits a 100 GB box? |
|---|---|---|---|
| pentad 1982–2024 | 3,142 | **33.1 GB** | yes |
| daily 2004–2024 (RAPID era) | 7,671 | **80.9 GB** | no (81 + 15 GB image + cache) |
| daily 1993–2024 (GLORYS era) | 11,688 | **123.2 GB** | no |
| daily 1982–2024 (full axis) | 15,706 | **165.6 GB** | no |

The current fleet is 100 GB. E-033 §5 named object storage as the precondition
for the daily rung and it was deferred for exactly this reason.

**The answer is already in Chris's own standing fleet rule** (2026-08-16): any
NEW box must be large-disk, 600 GB+, and the surveyed offers were an RTX 4090
at **$0.336/h with 1520 GB** (Nevada, rel 0.998) and **$0.244/h with 741 GB**
(Brazil). Either holds the full daily tensor plus the image plus an embedding
cache. So the daily arm is **not blocked on new engineering, it is blocked on
renting the right box** — which is a decision, not a project.

**DECIDED, Chris 2026-08-17: "yes, we need a large box for the daily arm."**
Rent it when Phase B is ready to use it, not before — `build_family5.py` does
not exist yet and the daily aggregation has not run, so a box rented now bills
at $0.24–0.34/h to sit idle. The fleet rule (600 GB+) is satisfied by either
surveyed offer; prefer the 1520 GB Nevada machine at $0.336/h if the 165.6 GB
tensor is to coexist with an embedding cache, since the daily cache scales
with T just as the tensor does.

**Recommended staging**, so the cheap half is not held hostage by the
expensive half:

- **Phase A — pentad pair, now.** 33.1 GB fits the existing 100 GB box. Train
  f4-40M and f4-200M as soon as `build_family4.py` has run. This also
  de-risks the whole matrix: if the pentad codecs behave, the daily arm is the
  same code with one constant changed.
- **Phase B — daily pair, on a large-disk box.** Needs the daily aggregation
  (`aggregate_cadence.py --cadence daily`, which is a passthrough and already
  written) and `build_family5.py` — which should be `build_family4.py` with
  `PENTAD_DAYS = 1` and its own `RECIPE_REV`, not a copy.

### The 1982–92 decade: keep it. My earlier framing was wrong.

Chris asked whether the sparsity is already an issue today and whether daily
makes it worse. **It is already the case, and daily does not make it worse —
the fraction is identical at every cadence**, because it is the same calendar
span:

| cadence | axis | 1982–92 steps with no base | share |
|---|---|---|---|
| monthly | 516 | 132 | **25.6%** |
| pentad | 3,142 | 804 | **25.6%** |
| daily | 15,706 | 4,018 | **25.6%** |

Family 3 has carried a quarter of its axis base-missing since it was built,
deliberately (§2 of E-034: wind fills the gap and the cable's decade becomes
usable truth). Calling 26% a *daily* problem was my error — it is the status
quo, restated in more rows.

**And at daily that decade gets BETTER, not worse.** Both channels that live
there are natively daily:

- **NCEP R1 wind is daily.** At monthly those 132 steps carried a monthly
  mean; at daily the 4,018 steps carry genuine day-to-day wind. That is real
  new information, not one number repeated 30 times.
- **The Florida cable is daily since 1982** — measured, the file announces
  itself as `DAILY FLORIDA CURRENT TRANSPORT`. So the decade yields ~4,000
  daily labels against 132 monthly ones: **~30× the truth, in exactly the
  decade with no GLORYS.**

The only cost is bytes, ~42 GB — and Phase B is renting a 741–1520 GB box
anyway, so the constraint that motivated the question does not bind. **Keep
the full 1982–2024 axis at daily.** The alternative would discard the
programme's single largest block of transport labels to save 25% of a disk
that has room.

*(Superseded framing, kept so the reversal is legible: this section previously
suggested starting the daily arm at 1993.)* That decade is real truth for
the cable and it is why family 3's axis starts in 1982. But at daily cadence
it is a large, wholly-base-missing block, and it is worth asking whether the
daily arm should start at 1993 (123.2 GB) and leave 1982–92 to the pentad
tensor, which already covers it at a twelfth of the cost per year.

---

## 4 · Order of work

1. ~~Pentad aggregation~~ — dispatched 2026-08-17 (`pentad-aggregate.yml`).
2. `build_family4.py` on a box — `--max-bins` against the REAL rg/wind caches
   first (they have only met synthetic files in their own schemas), then the
   full 33.1 GB.
3. **Read the Chinchilla inventory the build prints** and confirm or move the
   40M/200M rungs.
4. Baseline control: embed the pentad tensor with the existing frozen codec.
   This is the number every new codec is reported against.
5. Phase A: train f4-40M and f4-200M from scratch. **Verify the dispatch
   actually trains a fresh codec** — every run for six weeks froze it via
   `resume: "!run-62,run-63"`, so check the fresh-codec branch in
   `ml-train.yml` before spending GPU.
6. Phase B: rent a large-disk box, daily aggregation, `build_family5.py`,
   train f5-40M and f5-200M.
7. Re-encode embeddings, each with its matching codec. Never across a pair.
8. Stage 2 and the transport probes, reported against the frozen-codec
   baseline from step 4.

---

## 5 · Costs to record, because an entry showing only the successful run makes
the answer look cheaper than it was (§3)

- GPU hours per codec, per rung, measured rather than planned.
- The large-disk box's rate and hours for Phase B.
- The wall-clock of each tensor build and aggregation.

Nothing in this plan has been dispatched. The pentad aggregation is the only
thing running.

---

*Written 2026-08-17. Storage figures computed from the real channel count and
grid; cadence coverage figures follow from E-034 §4's Argo policy.*

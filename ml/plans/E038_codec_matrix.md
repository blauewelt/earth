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

|  | **pentad (5-day)** | **daily (1-day)** |
|---|---|---|
| **~40M** | f4-40M | f5-40M |
| **~200M** | f4-200M | f5-200M |

Two axes, deliberately: capacity × cadence. It answers "does finer cadence
pay?" and "does the answer depend on capacity?" in one design, which matters
because **wave 6A already showed the two interact** — znoise-big55 at 88M beat
xl55 at 205M on the rolled corridor AUC, so "scale pays most" is false on that
axis and a single-capacity cadence test could easily mislead.

**Are 40M and 200M the right rungs?** They should be *checked against the
re-anchored Chinchilla count, not inherited.* `build_family4.py` prints the
observed-value inventory, and E-034 §4 warns explicitly that it must be
re-read rather than scaled from the monthly one — the Argo policy drops the
observed count per pentad by construction. If the pentad anchor lands near,
say, 90M, then 40M and 200M bracket it, which is a good design. If it lands at
250M, then 40M is not a useful rung and the pair should move. **Read the
number the build prints before fixing the sizes.**

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

**An open question worth deciding rather than defaulting.** At daily cadence
the 1982–92 decade carries *no* base channels at all (GLORYS12 starts 1993),
so 4,018 of 15,706 steps — 26% — would be base-missing, costing ~42 GB to
represent wind and the Florida cable's labels. That decade is real truth for
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

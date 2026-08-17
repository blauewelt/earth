# E-039 PROPOSAL — the forecast archive: predict the globe once, score it forever

**Status:** proposal, nothing started. Written 2026-08-16 from Chris's question.

> "Ideally, we want to be able to predict the whole globe (the embeddings
> first, and then the AMOC or surface temperature or whatever we want). You can
> think of it as an animation that you can play on the globe visualization
> view but starting at a given date. Once we computed the whole globe, we save
> the outcome, and then we can visualize our predictions on the globe (again,
> any layer we want). Does this make sense? Am I being too naive?"

It makes sense, and it is less naive than it sounds — for a reason we only
measured today. But there are four ways to build it that would produce a
beautiful animation that lies, and they are the substance of this plan.

---

## 1. Why "the whole globe" is not ambition, it is the requirement

From `claude/dependency-cone-and-window.md` (measured, real coastline, seeded
from the AMOC corridor at 7.1% of world ocean):

| stencil | 12-month dependency cone, as % of **world ocean** |
|---|---|
| ring-8 @ 222 km | 29.5% |
| sunflower-89 @ 4444 km | **100%, reached by month 3** |

At the reach the leaderboard's champion actually uses, the set of pixels a
December corridor prediction depends on **is the world ocean**. So "predict
the whole globe" and "do the 12-month rollout correctly" are the same
sentence. This proposal does not add scope to the rollout; it names what the
rollout already has to be.

Corollary worth keeping: the cone stops being an optimisation at that reach
(it saves nothing and costs a scatter/gather per step), and starts being one
again below ~1000 km. **Reach and compute are one axis, not two.**

---

## 2. The architectural point: split PREDICT from SCORE

Today `ml/rollout_spatial.py` does both in one pass — it rolls, and it
accumulates MSSS into summary JSON inline. Every new question ("per-region
skill?", "does the Gulf Stream decay faster than the subpolar gyre?", "what
about a different damped-persistence reference?") costs a **GPU re-roll**.

The proposal is one artefact in between:

```
   heads  ──roll──▶  FORECAST ARCHIVE  ──score──▶  AUC, per-channel, per-region,
   (GPU)             (predicted ẑ and/or          maps, the globe animation,
                      decoded fields, on disk)     anything invented later
                                                   (CPU only, seconds)
```

Three things fall out, and they are the actual answer to "how do we build the
future AUC infrastructure":

1. **Metrics become queries.** A new metric needs no GPU and no re-roll. The
   AUC stops being a number a script computed once and becomes a number
   anyone can recompute from the artefact that produced it.
2. **A leaderboard entry becomes reproducible.** Today a row is a summary of a
   run that no longer exists. With an archive, the row's inputs are on disk.
3. **The globe animation is a VIEW of the archive**, not a separate pipeline.
   Same bytes, different renderer.

`--export-mask` and the E-026b audit block (per-channel curves, per-pixel skill
map at `--map-h`) are early, hard-coded instances of exactly this pattern. They
exist because someone wanted a decomposition the summary JSON didn't carry. An
archive generalises them and lets both be deleted.

---

## 3. What it costs — measured against artefacts we already ship

Global 0.25° is ~736,000 ocean pixels (E-033), 39 channels, K=24, d_z=64.

| artefact | size | home |
|---|---|---|
| rolling state `[P,K,d_z]` fp16 | 2.3 GB (resident) | GPU |
| predicted **embeddings**, 12 months, fp16 | 1.1 GB | Hugging Face |
| predicted embeddings, 240-month roll | 23 GB | Hugging Face |
| decoded **all 39 channels**, 12 months, fp16 | 0.7 GB | Hugging Face |
| **display grids for the app**, packed | see below | the repo |

The display number is the surprise, and it is why the app half is cheap.
`data/drivers.json` is 806,400 cells at 0.25°, packed one character per cell,
and it is **0.81 MB**. So:

- 0.25° global, packed: **~1 MB per channel per month** → 12 MB for a
  12-month, one-channel animation.
- 0.5° global, packed: **~0.26 MB** per channel-month → **~3 MB** for the same
  animation. **Recommended for the app**: the published Pages site is already
  ~403 MB against a 1 GB limit, and nobody reads a 0.25° forecast anomaly at
  0.25° on a phone.

So: **the science artefact goes to Hugging Face (the mirror already exists);
the app gets a 3 MB derived file.** They are two products of one roll, and
conflating them is how the repo gets to 1 GB.

---

## 4. The app side is ~80% already built

The GFS 10-day forecast layer is structurally the same thing one axis coarser,
and its machinery is generic:

- `monthlyGrid` + `forecastGrid` layer flags,
- `keyLen` (7 = month-keyed, 10 = day-keyed) — a monthly forecast is `keyLen: 7`,
- `monthsAvailable` / `yearDir` lazy per-year fetch (`ensureGridMonth`),
- `uiMaxDate()` / `syncDateMax()`, which already extend the date axis into the
  future while a forecast layer is on and pull it back when it is off,
- `init` (the model run) quoted in the layer's toast,
- `GridProvider`, `packed`, ramp legends, the probe, the pixel card.

What is genuinely new on the app side is **playback** — a play/pause/scrub
control that steps the date monthly and holds the camera. That is a small
feature, and it should be built against the GFS layer first, where the data
already exists, rather than waiting for the model.

---

## 5. The four ways this produces a beautiful animation that lies

These are the reasons to build it deliberately rather than quickly.

### 5.1 Skill decays with horizon, and an animation hides that (BLOCKING)

A smooth global animation carries the visual grammar of a weather forecast and
therefore its implied confidence. At h=12 the model is far closer to
climatology than to truth, and nothing in a pretty ramp says so.

**Requirement, not a nice-to-have: every forecast layer ships with a companion
per-pixel SKILL layer** — MSSS vs climatology at that horizon, measured on
holdout years — and the pixel card reads it alongside the value. The evaluator
already computes exactly this (`map_msss_clim_window` at `--map-h`); it needs
to be produced at every h, not one.

The honest default is to render low-skill pixels **desaturated**, so the eye
is drawn to where the model actually knows something. A forecast layer with no
skill layer beside it should be refused by the layer config, the way
`classGrid` refuses `deltaRange`.

### 5.2 The codec is a floor, and its blur will read as forecast error

The fields are decoded from a 64-number bottleneck. Even a **perfect**
embedding forecast reproduces the world only to the codec's reconstruction
error, so the animation's floor is not zero.

**Requirement:** ship "truth, encoded and decoded" as a selectable reference
for the same dates. Without it, every viewer — us included — will attribute
codec blur to forecast error, and we will optimise the wrong thing.

### 5.3 Anomaly space is not what the globe shows

The tensor is deseasonalised and standardised per channel. Turning ẑ into
"sea surface temperature in °C" means decode → destandardise → add the
month's climatology, and every one of those steps has a sign and an index that
can be wrong while still producing a plausible-looking map.

**Requirement:** a test that a zero-anomaly forecast decodes **exactly** to the
month's climatology, per channel. That single identity catches all three.

### 5.4 Long autoregressive rolls drift

Feeding predictions back for 240 steps usually ends in collapse to the mean or
a slow blow-up. The current long roll (`--long-months 240`) reports a median
trajectory and no stability diagnostic.

**Requirement:** log the embedding norm and the decoded field's variance per
step; stop and flag when either leaves a measured envelope. A forecast that has
collapsed to climatology should say so rather than animate smoothly.

---

## 6. Two more risks worth stating

- **The global tensor does not exist yet** (E-033), and global is not merely
  bigger: the ACC, the tropical Pacific and the sea-ice edge are different
  dynamical regimes from the North Atlantic. A head trained on family3 will
  not simply transfer, and the first global animation will look worst exactly
  where we have never trained.
- **The longitude wrap** (fixed 2026-08-16, `wraps_longitude`) is a
  precondition, not a detail: a global roll with the old zonal clip would have
  carried a slice of the Pacific as zeros while training and plotting normally.

---

## 7. Staged, so each stage is useful alone

1. **Playback in the app, on GFS.** No model work. Delivers a scrubbable
   forecast animation immediately and de-risks the UI half.
2. **Archive format + writer, on family3.** `rollout_spatial --write-archive`:
   predicted ẑ and decoded channels per (start, horizon), plus the per-pixel
   skill map at every h. Regional, so it is small and fast.
3. **Scorer reads the archive.** Reproduce today's AUC from the artefact and
   pin it against the inline number — the same gate discipline `#217` already
   enforces. Then delete the inline accumulation.
4. **Derived display grids + the skill companion**, shipped to the app at 0.5°.
   First real animation of a *model* on the globe, honest by construction.
5. **Global**, once E-033's tensor exists. By then everything above is tested
   at regional scale and the only new thing is size.

Stages 1–4 need no global tensor and no new hardware.

---

## 8. So: naive?

No — with one correction. The naive version is "compute the globe, save it,
look at it". The version worth building is "**compute the globe, save it
*with its own error bars*, and make every metric a query over that**". The
first is a demo; the second is the AUC infrastructure the question was
actually asking for, and it happens to produce the demo for free.

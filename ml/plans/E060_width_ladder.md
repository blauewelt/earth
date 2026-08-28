# E-060 · The width ladder, measured under a clean pool

**Written 2026-08-28 ~20:3xZ, BEFORE dispatch.** Everything in §4 is
pre-registration.

Chris, after reading E-059's 24k/32k split: *"Sounds good, let's do this."*

---

## 1. Why this exists

E-054 asks whether capacity keeps paying at the pentad frontier, and its
whole premise is a sentence from its own TL;DR: *"E-051's val curve is a
clean power law to 200k with no saturation (0.0812@20k → 0.0414@100k →
0.0330@200k, α≈0.33)."*

That power law is measured on a validation set whose transitions were in
E-051's training data. Under `--holdout-scope window` the same architecture,
seed, codec, Z and validation windows produce **no power law at all**:

| step | E-051 val ratio (contaminated) | E-059 val ratio (clean) |
|---|---|---|
| 2,000 | 0.2247 | **0.6105** |
| 20,000 | 0.0812 | **0.6411** |
| 32,000 | 0.0670 | **0.6409** |

E-059's val is flat from step 2,000 while its train loss falls 3.9468 →
1.0056, a train/val gap of **13.7×** against E-051's steady 1.2–1.3×.

**E-054b is measuring the capacity rung of that contaminated law right now.**
At step 40,000 the 400M head reads ratio **0.05478** against E-051's
**0.06057** at the same step — a 10% "capacity keeps paying" signal, on
contaminated val. That is a measurement of how well 400M parameters memorise
2,417 temporal windows. Chris's ruling: **do not relaunch E-054b**, and spend
a fraction of it here instead.

## 2. The question, stated so it can be answered

Two hypotheses produce the same flat-val/falling-train picture:

- **H1 — over-parameterised.** 206.66M parameters against 2,417 distinct
  temporal windows (~85,500 parameters each; the pool's 209,549,066 is
  exactly 2,417 end-bins × 86,698 pixels). Capacity beyond some much smaller
  width buys only memorisation.
- **H2 — the task is near its ceiling at h=1.** One-step pentad prediction on
  unseen years has ~0.61 of persistence in it and no more, whatever the
  width.

A single width does not separate them. A ladder does, and it is cheap because
**E-059's own val plateaus by step 2,000** — nothing after ~20k changes the
reading.

## 3. Design

Four points. **E-059 supplies the 206.659M point for free** at its own step
20,000; three new arms are dispatched.

| arm | d_model × layers | parameters | vs 206.659M | node | zone |
|---|---|---|---|---|---|
| E-060a | 256 × 8 | **7,597,856** (7.598M) | 0.037× | `e060a-8m` | us-central1-a |
| E-060b | 512 × 12 | **40,388,128** (40.388M) | 0.195× | `e060b-40m` | us-east5-a |
| — | 1024 × 16 | 206,658,592 | 1.000× | `e059-window` | **already running** |
| E-060c | 1280 × 20 | **399,947,552** (399.948M) | 1.935× | `e060c-400m` | europe-west4-a |

A **53× span** in capacity.

**Everything else is E-059, unchanged**: `--holdout-scope window`,
family4_na025_pentad_r2 (`37e146384b`) via the published Z
(`Z_8b639abe36_37e146384b.npy`), frozen run-415 codec, K 144, stencil 145
ring `spiral:111-4444-0.71-0.5`, batch 256, lr 1e-3, expdecay halflife
40,000, warmup 2,000, znoise 0.7, grad-clip 128, `--train-lon-hold none`,
seed 0, `n_heads` 4 (hard-coded in `ml/temporal.py`; the JAX trainer refuses
anything else). Each startup file is E-059's with **only** `NODE`, `D_MODEL`,
`LAYERS`, `STEPS`, `TAG` — and `GRAD_ACCUM` on E-060c alone — changed, and
the knob blocks are `diff`ed against E-059's before launch.

**STEPS = 20,000.** The LR schedule is `LR * 2^(-(t - warmup)/halflife)`,
independent of the step total — E-059's lr at 20,000 reads 7.320e-04 =
1e-3 · 2^(-18000/40000), so each arm's first 20,000 steps trace a
**bit-identical LR trajectory** to E-059's first 20,000. The reading is taken
at the matched step, and the step-20,000 RAPID probe fires in all four.

E-060c carries `GRAD_ACCUM=4` because 1280×20 at K=144 batch 256 does not fit
otherwise — the same decomposition E-054b runs, certified exact to max rel
2.419e-07, so it is a memory decomposition of the same batch-256 step and not
a batch change.

## 4. Pre-registered readings

The comparison is `stage2_val_zmse / 21.44621` at step 20,000, against
**E-059's 0.64106**. Every arm must first reproduce the pool certificate —
2,417 end-bins, 0 of 2,417 touching a held-out bin over 350,465 checks,
209,549,066 train windows, `holdout_scope window`, `val_persistence`
21.44621 — since the pool does not depend on width.

**Predicted, before the arms run:** H1. Specifically, that **7.598M reaches
within 0.02 of 206.659M's 0.641** — i.e. a 27× parameter cut costs almost
nothing on held-out years — and that 399.948M is **no better than 0.63**.

The prediction is made because E-059's val plateaued at step 2,000, when its
train loss was still 3.95: whatever the model had learned that generalises,
it had learned before capacity could plausibly have mattered.

## 5. Falsifiers

- **H1 is wrong** if val improves monotonically with width across the 53×
  span — 7.598M materially worse than 40.388M materially worse than
  206.659M. Then capacity does buy forecast skill under a clean pool, the
  E-054 programme survives its contamination with its conclusion intact, and
  the right next move is the capacity rung re-run at `window`.
- **H2 is confirmed over H1** if all four points land within noise of each
  other, INCLUDING 7.598M. Then width is not the axis at all and the ceiling
  is the task; the answer is more data or a different objective, not a
  different model.
- **Something else is wrong** if any arm's pool certificate differs from
  E-059's. The pool is width-independent; a different number means the arm is
  not the experiment.
- **The ladder is uninformative** if an arm's TRAIN loss at 20,000 does not
  order by capacity (bigger fits better). That ordering is the sanity check
  that the widths are doing what widths do; without it the val comparison
  means nothing.

## 6. What it costs

Three v5litepod-4 SPOT nodes in parallel. E-059's measured pace is 333 s per
2,000 steps at 206.659M, so 20,000 steps is ~0.93 h there; E-054b's 400M with
grad-accum 4 measures ~1,850 s per 2,000, so E-060c is ~5.1 h. Add ~1 h of
per-node setup (tensor fetch, anomaly transform, Z pull). Total ≈ **9 node-
hours ≈ $18 at the day's spot rate** — well under one E-054b relaunch.

## 7. What it does NOT settle

The one-step ratio at h=1 is not the headline. The RAPID probe was
indistinguishable between E-051 and E-059 at step 20,000 (0.612 vs 0.616),
and the decisive test of forecast skill is the rolled corridor AUC, not the
z-space loss. This ladder answers "does width buy one-step generalisation
under a clean pool", which is the question the E-054 programme is currently
spending money on. It does not answer whether any of these heads can
forecast, and no roll should be dispatched off it.

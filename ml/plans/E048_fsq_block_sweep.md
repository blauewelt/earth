# E-048 · The FSQ block-codec hillclimb — 30-day windows, 15-day advance, and where the levels sit

**Status: BUILT AND TESTED END TO END. Not yet dispatched.** Written
2026-08-24 from Chris's direction the same morning. Nothing has been trained
beyond CPU toys; every number below that is not labelled MEASURED is
arithmetic, and says so.

## 1 · Why, and in whose words

Chris, 2026-08-24: *"hillclimb on the codec… One embedding every 30 days (6
pentads as input), advance by 30 days. One embedding every 15 days (6
pentads/30 days as input, OVERLAPPING windows), advance by 15 days — each
embedding has Argo as part of it, two consecutive ones share the same monthly
Argo values. Use FSQ for these experiments. For each channel try to compute a
distribution such that the FSQ levels can be on the scalar ladder (uniform:
level\*c) or on the exponential ladder (c\*\*level)."*

Three things are being climbed at once, deliberately — this is a hillclimb, not
a factorial:

- **A WINDOW instead of a calendar month.** E-047's month blocks are ragged (6
  or 7 bins, k_max 7, 16.6 % of cells padding on this record's edges). A fixed
  6-bin window is uniform, has no padding at all, and still contains exactly
  one Argo stamp — RG-Argo is a monthly product written into ONE pentad bin
  per month (MEASURED: `n_rg_live` 252/3,142 = 8.02 %, the mid-month stamp,
  79.41 % of ocean pixels when present).
- **OVERLAP, as a second arm.** At stride 3 the embedding axis advances 15
  days while the input stays 30, so the head gets twice the temporal
  resolution and consecutive embeddings share half their input — including,
  as Chris says, the same monthly Argo values.
- **FSQ, with the level placement itself under test.** E-046 put the levels on
  an evenly spaced lattice. E-048 adds the exponential ladder and, more
  usefully, an `auto` mode that MEASURES the choice per z-dimension.

## 2 · The two configurations

| | **arm A** `f4r2-70M-fsqblock-w6s6` | **arm B** `f4r2-70M-fsqblock-w6s3` |
|---|---|---|
| `time_block` | `6/6` | `6/3` |
| input per embedding | 6 pentads = 30 days | 6 pentads = 30 days |
| axis advances | 30 days | **15 days** |
| overlap | none | **3 of 6 bins** |
| blocks over the pentad record | ~523 | ~1,046 |
| Argo stamps per embedding | 1 | 1 (SHARED with the neighbour) |
| `k_max` / pad cells | 6 / **none** | 6 / **none** |
| everything else | 768×10, 8 heads (head_dim 96), d_dec 512, d_z 64, patch 1, `fsq_levels` 8, `fsq_ladder` auto, batch 512 | identical |

**MEASURED PARAMS: 71,335,697 (71.336M)** — encoder 70,970,944, decoder
364,753, including the two E-047 embeddings (`time_emb` 6×768, `q_time` 6×16).
From:

```
python3 -c "import sys;sys.path.insert(0,'ml');from model import PixelMAE;\
m=PixelMAE(n_chan=40,d_model=768,n_heads=8,n_layers=10,d_z=64,d_dec=512,\
patch=1,dec_layers=2,k_time=6);print(sum(p.numel() for p in m.parameters()))"
```

Head_dim is **96**, under the 128 ceiling that #387 died above (head_dim 256 at
202M). d_z stays **64** — E-047a's own value, so the pair changes the WINDOW
and the BOTTLENECK against a width the record has already run.

## 3 · The axis, and the four questions it had to settle

`ml/timeblocks.py`. `--time-block W/S` is a width-W window advancing S bins;
`N` is exactly `N/N`, so E-047's fixed-N mode is the non-overlapping case of
ONE rule rather than a second rule that has to agree with it
(`tests/test_e048_overlap_blocks.py` check 1 pins `6/6` against `6` array for
array).

**(a) Count and labels.** `n_blocks = floor((T−W)/S) + 1`; the trailing
remainder is dropped rather than padded, which is fixed-N's own rule. A block
is labelled by its **first bin** — the window's calendar ANCHOR — because that
is what fixed-N already did and because `TimeAxis` parses the label and needs a
real calendar position. **At stride 3 those labels REPEAT** (two windows a
fortnight apart can sit in one calendar month), so a window axis is NOT a
unique monthly key and nothing downstream may read it as one.

**(b) The roll reads the stride off the axis.** `BlockAxis.axis_dict()` hands
`ml/rollout_spatial.py:TimeAxis` a descriptor: the monthly one for month mode
(bit-identical to what the roll built before E-048), and a **BINNED** one for a
window, whose step is S source bins. Everything day-defined then follows with
no further edit, and the test measures all of it: `step_days` 30 / 15, bands cut
at the same DAY edges (`h1-3/h4-6/h7-12` at 30 d — the monthly partition
exactly — and `h1-6/h7-12/h13-24` at 15 d), day-matched leads the same twelve
DURATIONS (1..12 vs 2..24), and each scored cell's lead `h × stride × 5` days.
Before this, the roll took TimeAxis's monthly path for any block codec: right
for month mode, and for a window mode it would either refuse (repeating labels)
or advance a MONTH per step while the state advanced 15 days.

**(c) The month-of-year token keys on the window's anchor; the CODEC's own
context keys on the window's CENTRE.** `BlockAxis.remap_rows` needed no change
— ownership is still a partition, because `block_of_row` returns the LATEST
window containing a row (`min(floor(r/S), B−1)`), whose anchor is never more
than S−1 bins earlier, so every covered row lands on exactly one block and no
truth value is counted twice. What DID need fixing: `ml/temporal.py` embedded
block codecs with the month-quantized `ctx_all` while `ml/train.py` trains them
on `BlockAxis.ctx_phase()` (the continuous fraction-of-year phase of the
block's centre) and `ml/rollout_spatial.py` re-encodes with the same — the two
consumers of one codec disagreed. A rounding error at month blocks; at stride 3
it is structural, because the month token is IDENTICAL across a step the axis
takes. The embed cache name now carries the mode, so a Z written under the old
context is a MISS and not a lie.

**(d) Refusals, not guesses.** `S > W` is refused (it would leave bins in NO
embedding — a gap in simulated time that nothing downstream can see); so are
`S = 0`, a non-numeric mode, a window wider than the record, and `axis_dict()`
on an axis built from month labels alone (a window of labels is not a
duration).

## 4 · THE PERSISTENCE CAVEAT AT STRIDE 3 — read this before reading any ratio

At S < W the previous embedding is built from a window sharing W−S of its W
bins with the target. **"Predict the next embedding" is therefore a strictly
easier question at stride 3 than at stride 6, and persistence is stronger BY
CONSTRUCTION.** This moves the BASELINE, not the metric.

- A z-space one-step ratio from arm B is comparable with **another 15-day roll
  and with nothing else** — not with arm A, not with E-047a's month blocks, not
  with the pentad archive.
- What IS comparable across the two arms: skill at a **fixed horizon in DAYS**
  (h1-6 at 15 d covers the same 90 days as h1-3 at 30 d, by construction of the
  day-defined bands), the reconstruction audit, and the probe.
- The same applies inside training: the codec's neighbour term at `dt = ±1` is
  the adjacent, OVERLAPPING window, so it is partly a reconstruction term. That
  is deliberate — the alternative would make `dt` mean a different duration at
  every stride and break the one rule the block axis has, that `off`'s dt counts
  BLOCKS.

The axis says so at startup, the roll's artefact carries
`blocks.persistence_note` with width/stride/overlap/step_days beside the
numbers, and `ml/train.py`'s `step_loss_block` says it where the term is
computed. A future reader who quotes an arm-B ratio against a 30-day number has
to walk past three warnings to do it.

## 5 · The ladder question

`--fsq-ladder` takes `uniform` | `exp` | `auto`; the arithmetic lives in
`ml/fsq_ladder.py`, which is numpy-only and is IMPORTED by both the torch model
and the JAX mirror rather than copied.

**`uniform`** is E-046's lattice, the default, and BIT-IDENTICAL to `7f8dabb`
(pinned across three geometries × three level specs).

**`exp`** places the same L levels geometrically — `sign(v)·a·c^j` with
`a·c^(n−1) = R = 2σ`, the same saturation radius, assigned by nearest level in
LOG space, with a zero level iff L is odd. It **REPLACES the tanh bound rather
than composing with it**, and that is forced rather than chosen: the de-scale
is linear by design (inverting the tanh would send the outermost levels to
infinity), so the tanh can only move the BOUNDARIES and never the POINTS —
"exponential levels behind a tanh bound" is not expressible in this
parameterisation at all. One statement per ladder, both in z-units, both
bounded identically. Its straight-through gradient is **|z_q|/|v|**, measured:
0.709..1.411 at c = 2, i.e. inside [c^−1/2, c^1/2], decaying as R/|v| under
saturation.

**`auto`** measures the choice PER Z-DIMENSION on a sample of the run's own
pre-quantization activations (one deliberate `encode_pre` over
`--fsq-auto-n` = 4,096 train pixels at `--fsq-auto-step` = 2,000), compares both
ladders' quantization MSE per dimension over `AUTO_BASES` = (1.5, 2, 3, 4),
keeps uniform on a tie, and writes the fitted lattice into the checkpoint. Every
loader rebuilds from that string and **none re-fits** — re-fitting at eval time
would score a different model from the one that trained, and an `auto`
checkpoint with no recorded fit is REFUSED rather than quantized uniformly.

**"For each channel" is per Z-DIMENSION, and the difference is not cosmetic.**
40 channels enter and d_z coordinates leave, mixed by an encoder; the
bottleneck has no per-channel axis to choose on. A per-input-channel ladder
would be a different object, sitting before the encoder on the input tokens —
which is `--input-quant`'s side of the system, not this one.

### 5b · A MEASUREMENT MADE WHILE BUILDING THIS, and what it does to the reading

At **even L the E-046 uniform map is not antisymmetric.** `shift =
atanh(offset/half)` is applied INSIDE the tanh and `offset` is subtracted
OUTSIDE it, so the two cancel exactly at v = 0 and nowhere else, and the whole
lattice sits about half a step high. Measured on 200,000 N(0,1) draws
(`tests/test_e048_fsq_ladders.py` check 5 pins all four numbers):

| ladder | mean z_q | max \|q(−v) + q(v)\| |
|---|---|---|
| uniform, L = 8 | **+0.233** | **0.571** — one full step |
| uniform, L = 6 | **+0.329** | 0.800 |
| uniform, L = 7 (odd, no offset) | −0.003 | 0.000 |
| exp, L = 8 | −0.004 | 0.000 |

A value sitting exactly ON a negative level rounds to the next level UP.

This is **not repaired here**, for three reasons, and the third is why the
paragraph exists. It is `7f8dabb`'s shipped map — the codec the in-flight E-046
arm is training — and `uniform` must stay bit-identical to it. `to_z` is a free
linear map, so the encoder can learn a compensating offset; what the bias costs
is codebook symmetry, not capacity. And **it is part of why `auto` chooses
`exp` on essentially any centred distribution**, so a sweep that reported "the
exponential ladder wins" without saying this would be selling a bias correction
as a tail argument.

**The contrast that separates the two stories is pre-registered here: at ODD L
the uniform map is unbiased and exactly antisymmetric.** If `exp` still wins at
L = 7 by a comparable margin, the win is about COMPANDING; if the margin
collapses at L = 7, it was about the even-L bias, and the cheaper fix is a
symmetric uniform map rather than a geometric ladder. That is a CPU-only
measurement on the fitted checkpoints — it costs nothing and is owed before any
"the exponential ladder is better" sentence is written.

## 6 · Hypothesis and PRE-REGISTERED FALSIFIERS

**Hypothesis.** A finite-alphabet bottleneck over a FIXED 30-day window beats
the continuous calendar-month codec on (i) the 20,000-step stage-2 head and
(ii) the reconstruction audit; and the 15-day advance buys the head resolution
without costing the codec anything the audit can see.

The controls are named, and every number must be READ at harvest rather than
remembered:

1. **The 20k head.** Control: **#457** (E-047-HEAD — #427's exact head config,
   xl144-zn-pentad-nolonhold, stencil 145, znoise 0.7, grad-clip 128, seed 0,
   20k, on the TPU month-block codec's z), read off `ml-metrics`
   `run-457.jsonl`, plus the SELECTION baseline **E-045-A2a's 0.0721** that
   E-047 was itself dispatched against. **Arm A passes only if its 20k ratio is
   CLEARLY below #457's** — the axes are both ~monthly-cadence (30 d vs a
   calendar month), so this comparison is legal. **Arm B's ratio is NOT
   compared to either** (§4); arm B's head is read at a fixed horizon in days
   and against its own persistence.
2. **The reconstruction audit.** Control: E-047a's own accepted round-6 numbers
   (MEASURED 2026-08-24: the Argo-cell collapse CURED — fast channels at the
   Argo cell FVU 7.6 %/17.4 % trained/held-out where per-bin d_z-32 read
   25–112 % — at the price of Argo-FREE cells reading 9–19 % where per-bin was
   under 1 %, with the winds carrying it: held-out winter tau FVU 0.65/0.75).
   E-048 arm A must not make the Argo-free cells worse; d_z is unchanged at 64
   and the window is one cell NARROWER than a month, so the ~4:1 squeeze is if
   anything relieved.
3. **The ladder.** Falsified if `auto` picks uniform on most dimensions (the
   exponential ladder was not what the distribution wanted) OR if the L = 7
   contrast of §5b collapses the margin (the win was the even-L bias, not
   companding).

**What a pass would NOT establish**, stated in advance: n = 1 at a tier with no
measured replicate band — a new codec, a new cadence AND a new bottleneck at
once, and `ml/CLAUDE.md` §3b's harder clause puts every pentad-and-finer arm in
the "first result buys its own replication" class. No number off these recipes
is written as a LEVEL until its seed-1 twin exists.

## 7 · Cost, at measured TPU rates

MEASURED, 2026-08-24: the E-047a 40M month-block codec ran **0.167 s/step at
batch 512** on a v5litepod-4 and finished 60,000 steps in **2.8 h** of training
for **~$22** on-demand (node `e047a-tpu-60k`).

Scaling to this geometry — FLOPs go as layers × d_model² × tokens, and one
window is 6×40+2 = **242** encoder tokens against a month block's 7×40+2 = 282:

    0.833 (10/12 layers) × 2.25 (768²/512²) × 0.858 (242/282) = 1.61×

so **~0.27 s/step, ~4.5 h and ~$35 per arm at 60,000 steps; ~9 h and ~$70 for
the pair** on-demand, less on spot (the TPU script's whole lifecycle is built
for preemption). THIS IS ARITHMETIC, NOT A MEASUREMENT, and the first log
window replaces it.

Memory, same footing: per-sample activation scales as tokens × d_model ×
layers = 0.858 × 1.5 × 0.833 = **1.07×** the 40M month-block's at the same
batch, on the chips that already held that configuration at 512; parameters add
71.3M × 4 B × 3 (weights + two Adam moments) = 856 MB against 456 MB, a further
400 MB of 16 GB. **Batch 512, with 256 as the next rung down** — raise or lower
it deliberately and say so in the doc, do not let it drift.

Downstream, and it is the one place arm B costs more: arm B's Z has **twice the
rows** of arm A's over the same record, so the stage-2 embed pass and every
stencil the head reads are 2×. The measured embed reference is **8.5 h** (#423,
pentad); a window axis is far coarser than pentad, but budget arm B at twice arm
A's.

## 8 · What was built

| file | what changed |
|---|---|
| `ml/timeblocks.py` | `parse_mode`, the `W/S` window mode, `axis_dict()`, `head_season()`, overlap-aware `describe`/`block_of_row` docs |
| `ml/fsq_ladder.py` | **new** — level parsing, both ladders' parameters, the numpy reference quantizer, the per-dimension `auto` fit, the recorded-fit string |
| `ml/model.py` | `InputQuant` gains the ladder (uniform bit-identical), `encode`/`encode_pre` split, `codec_from_ckpt` reads and refuses the ladder args |
| `ml/train.py` | `--fsq-ladder`, `--fsq-exp-base`, `--fsq-auto-n`, `--fsq-auto-step`, `--fsq-ladder-fit`; the fit step; ARCH adoption; metrics |
| `ml/temporal.py` | the codec's embed context is the block axis's own; `head_season`; the mode in the cache name |
| `ml/rollout_spatial.py` | the roll's axis comes from `axis_dict()`; stride/width/overlap/step_days and the persistence note in the artefact |
| `ml/jaxport/{models,convert,train_stage1}.py` | the same bottleneck, the same fit, the same refusals; the jit closure rebuilt when the lattice changes |
| `ml/recipes/f4r2-70M-fsqblock-w6s{6,3}.json` | the two arms |
| `.github/workflows/ml-train.yml` | `fsq_ladder` as a recipe-only key |
| `tests/test_e048_overlap_blocks.py` | 7 checks, including an end-to-end 6/3 train → embed → roll |
| `tests/test_e048_fsq_ladders.py` | 9 checks, including torch/JAX/numpy agreement on both ladders |

**Deliberately left.** The per-bin codec evaluation still skips for any block
codec (E-047's decision, unchanged — its recon rows, its t+1 persistence and
its channel skills are all per-bin quantities). `--fsq-exp-base` is not a recipe
key: `auto` fits the base per dimension, and a deliberate `exp` arm gets the
default 2.0 until an experiment needs otherwise. And the even-L asymmetry of
§5b is recorded, not repaired.

## 9 · Dispatch

**GPU / Actions.** `window: recipe:f4r2-70M-fsqblock-w6s6` (or `-w6s3`),
`resume` EMPTY — a fresh codec both times.

**TPU**, the sweep vehicle. The two arms differ in ONE substitution:

```
# arm A — 6/6, 30-day advance
sed -e 's|__BUCKET__|<bucket>|' -e 's|__NODE__|e048a-w6s6|' \
    -e 's|__TPUZONE__|<zone>|' \
    -e 's|^STEPS=.*|STEPS="60000"|'   -e 's|^BATCH=.*|BATCH="512"|' \
    -e 's|^D_Z=.*|D_Z="64"|'          -e 's|^PATCH=.*|PATCH="1"|' \
    -e 's|^D_MODEL=.*|D_MODEL="768"|' -e 's|^N_LAYERS=.*|N_LAYERS="10"|' \
    -e 's|^N_HEADS=.*|N_HEADS="8"|'   -e 's|^D_DEC=.*|D_DEC="512"|' \
    -e 's|^TIME_BLOCK=.*|TIME_BLOCK="6/6"|' \
    -e 's|^EXTRA_ARGS=.*|EXTRA_ARGS="--fsq-levels 8 --fsq-ladder auto"|' \
    ml/jaxport/tpu_train.sh > /tmp/e048a.sh

# arm B — 6/3, 15-day advance: the SAME line with two substitutions changed
    -e 's|__NODE__|e048b-w6s3|' … -e 's|^TIME_BLOCK=.*|TIME_BLOCK="6/3"|'
```

`tpu_train.sh` still pins the **family-3 monthly** tensor by sha
(`TENSOR_SHA`/`TENSOR_ASSET_PREFIX`) — a window codec needs the **pentad** one
(`family4_na025_pentad_r2`), so those two lines must be substituted as well or
the node will train the right architecture on the wrong bytes. That is a
dispatch-time fact, not a code change, and it is written here because it is
exactly the kind of thing a launch script gets right by accident once and wrong
the second time.

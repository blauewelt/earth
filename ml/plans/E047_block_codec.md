# E-047 · A codec that fuses several pentad bins into one embedding

**Status: BUILT AND SMOKE-TESTED END TO END. Not yet dispatched.** Written
2026-08-22; the open decision of §6 was made the same day (option **b**) and
the wiring landed with it. Nothing has been trained beyond a 12-step CPU toy.

## 1 · Why

Chris, 2026-08-22: a codec that FUSES multiple pentad bins into one embedding,
with proper sub-month time labelling.

The measurement that makes this the obvious next move is from the same day's
copy-reconstruction audit. `rg_*` is a MONTHLY product written into **one
pentad bin per month** — the tensor's own `n_rg_live` is 252 of 3,142 bins
(8.02 %), always the mid-month stamp, covering 79.41 % of ocean pixels when
present. So today's per-bin codec sees, in five bins out of six, a 40-channel
pixel with 35 channels missing; and on the sixth it sees all 40 and its round
trip collapses — FVU **15-18 %** on Argo-carrying bins against **0.4-0.5 %**
on Argo-free ones, with the eight always-present channels falling to r 0.77-0.95
purely because 35 more channels are competing for the same 32 dimensions
(measured: hiding `rg_*` from the encoder on the SAME bins and pixels restores
them to r 0.997-0.998).

A block codec makes that a non-question. One month enters as a **k_max x C grid
of cells**; the Argo anchor and the five pentads around it land in ONE
representation; and "Argo is missing in five bins of six" stops being a special
case and becomes what the codec has always handled — an unobserved cell.

The second half is Chris's labelling requirement. `month_feats`
(`ml/rollout_spatial.py:709`) and `ctx_all` (`ml/temporal.py:1394`) are both
sin/cos of an **integer** month, so on a pentad axis all ~6 bins of a month
share one identical season token: a staircase forcing on a 5-day axis. A block
codec carries the **continuous fraction-of-year phase of the block's centre**
from birth, so nothing downstream has to undo a quantization the encoder baked
in.

## 2 · What a block is

`ml/timeblocks.py` — built and tested (`tests/test_e047_time_blocks.py`, 7
checks). It is axis arithmetic only: no model, no loss, no training, and one
definition shared by the codec, the embed path and the tests, because two
copies of an axis rule are two places for the same off-by-one to live.

| | `month` (primary) | `N` (fixed, N=2 planned second arm) |
|---|---|---|
| grouping | by calendar-month LABEL | consecutive non-overlapping N |
| bins/block | 6 or 7 (ragged), edges partial | exactly N |
| `k_max` | 7 | N |
| padding | short months padded, pad cells flagged | none |
| label | `YYYY-MM` | the block's FIRST bin's label |
| remainder | none — every bin is used | trailing < N bins dropped |

Measured on a 2-year pentad axis: 146 bins → 25 blocks, 23 interior months of
6-7 bins (2 of them 7), both EDGE blocks partial (1 and 5 bins — the record
starts and ends mid-month, which is what the padding exists for), 16.6 % of
cells padding, one whole calendar year summing to 73 bins.

Three properties worth keeping:

- **A pad cell points at a REAL row** (its block's last), never at 0 or −1. It
  is masked wherever it is read, and a gather that indexes a valid row cannot
  produce a silent wrap to the start of the record.
- **`cell_obs` fuses padding and missing data into ONE mask.** A pad cell is
  unobserved whatever the source said.
- **The label is a calendar position**, so in month mode the resulting Z has a
  MONTHLY axis built entirely out of 5-day data — `TimeAxis` reads it as
  monthly, and the roll's horizon, bands and day-matched leads are the
  archive's own with no cadence-matching argument at all.

## 3 · What the encoder does with it

`ml/model.py` — built. `PixelMAE(..., k_time=K)` takes `[B, k_time, C]` values,
obs and mask; each cell's token is `chan_emb[c] + time_emb[j]`, so the encoder
can tell the 3rd pentad of a month from the 5th. Missing and pad cells go
through the existing `miss_tok` path unchanged.

**`k_time=1` is the archived codec, key for key and value for value** —
`time_emb` is created only when `k_time > 1`, and the test asserts an identical
`state_dict` and an identical `z` on the same input. `codec_from_ckpt` reads
`k_time` back out of the args, so every consumer rebuilds the right encoder
without being told.

Sizing, which `BlockAxis.describe()` prints at startup: one month block is
`7 x 40 + 2 = 282` encoder tokens against `42` per bin (6.7x per forward), and
there are `1/6` as many forwards — **~1.12x the encoder work per pass**, plus
whatever the attention's quadratic term adds within a forward (282 vs 42 is
45x the attention FLOPs but on a sequence still far shorter than the MLP cost
dominates at this width — TO MEASURE on the smoke run, not assumed).

## 4 · Hypothesis and falsifiers

**HYPOTHESIS — fusion beats selection.** E-045's A2a arm answered the cadence
question by SELECTING one bin per month (stride 6) and reached a 20k one-step
ratio of **0.0721**. A month-block codec keeps all six bins and fuses them, so
its 20k stage-2 head should beat 0.0721 — and its roll, being monthly-cadence,
is directly day-matched comparable to the monthly archive's **+0.939** corridor
line while being built entirely from 5-day data.

**FALSIFIERS.**

1. **Parity with 0.0721** (say, within the pair noise of it) means selection
   suffices: the extra five bins carry nothing the anchor bin did not, fusion
   is not worth the retrain, and E-045's stride arm is the cheap answer to keep.
2. **Worse than 0.0721** means the architecture is wrong, not the idea — most
   likely the grid dilutes the anchor bin (35 of 40 channels appear in 1 of 7
   cells) or `d_z` 64 is not enough for 7x the input.
3. **The roll still mode-locks to the calendar** (secondary, via the
   `longstart:` multi-context-end discriminator): fusion changed the input and
   not the failure.

**CONTROLS, both already measured:** E-045 A2a's 0.0721 at 20k on the same
tensor and protocol, and the monthly archive's +0.939 corridor `_trainlon` for
the roll.

## 5 · Cost

| item | estimate |
|---|---|
| month-block codec retrain (stage 1, d_z 64) | **~1.15x #415** ≈ 23 h ≈ **$3.5-4** on a normal box |
| embed on the new codec | ~1/6 the forwards of a per-bin pass, but 6.7x per forward ≈ **~1.1x**, so budget the same 8.5 h class |
| one 20k head on its z | **~$1** |
| its roll (monthly cadence, so the ARCHIVE's cost, not the pentad one) | **$1-2** |
| **to a first verdict** | **~$6-8** |

## 6 · THE DECISION — how the decoder queries a cell: **(b), decided**

Point 5 said "decoder reconstructs all cells"; the decoder could not express a
cell query without a choice, because `off = (Δlon, Δlat, Δmonth)` runs through
ONE shared 7-entry table (`q_off`, `max_abs_offset` 3, `ml/model.py:140`).

**DECIDED: option (b) — a new `q_time = nn.Embedding(k_time, 16)`**, the exact
mirror of the encoder's `time_emb`, concatenated into the decoder input. The
rejected alternative (a) reused the `dt` slot centred at `j-3`, which fits the
existing table exactly and costs nothing — and would make ONE index mean two
things, "one bin later inside this block" for a cell query and "one block
later" for the neighbour loss, with nothing in the input telling the decoder
which. §4.1: prefer the formulation that removes a failure mode. The cost is 16
input dimensions on a ~1.3M-parameter decoder; `k_time = 1` checkpoints keep
their shape exactly, asserted in the tests.

**AND `dt` NOW MEANS BLOCKS.** Under blocking the neighbour loss's `dt = ±1` is
the next or previous BLOCK, not bin — because the block axis IS the time axis
downstream, and the neighbour loss should see the world the head will see. A
month codec's neighbour term therefore reaches one month either way, which is
what the archive's monthly codecs have always meant by it.

## 7 · What was built (and the two things deliberately left)

**Done, and tested before any GPU:**

1. **`ml/train.py --time-block`** (`month` | `N`, default `""` = off =
   bit-identical). Blocks are cut AFTER the anomaly transform, never before —
   the climatology and the per-channel z-score are properties of the SOURCE
   axis, and computing them per block would change the statistics a frozen
   encoder is later asked to reproduce. The pool is over (block, pixel); a
   BLOCK is held out iff ANY of its bins is, so a block can never leak a
   holdout bin into training through a shared embedding. `ctx` is the block
   centre's continuous phase. `k_time`, `time_block` and `ctx_mode` ride
   `vars(a)` into the checkpoint.
2. **`ml/model.py`**: `PixelMAE(k_time=K)` reads a `[B, k_time, C]` grid
   (cell token = `chan_emb[c] + time_emb[j]`), and `query(..., tpos)` names a
   cell through `q_time`. A block codec asked for a channel without a cell
   RAISES rather than picking one.
3. **`ml/temporal.py`**: `k_time` comes from the CODEC, never a flag. The
   block axis becomes the axis — labels, `moy`, `t_hold` (any bin held out ⇒
   block held out), the RAPID rows (`BlockAxis.remap_rows`), the pool and the
   persistence baselines. `embed_everything` assembles the same grids at full
   visibility and REFUSES to embed a block codec bin-by-bin. `--time-stride`
   on a block codec is refused: two time surgeries at once.
4. **The cache name carries `blk<k_max>`.** The weight hash already separates
   a block codec from a per-bin one; the name is so a human can triage a full
   disk without loading the file.
5. **Plumbing**: `time_block` is a RECIPE-ONLY key (`RECIPE_TIME_BLOCK`),
   declared in the workflow's recipe-only block and passed by the Train step.
   It is recipe-only for two reasons, not one: the 25-input list is full, AND
   it is a property of a CODEC rather than of a dispatch — `temporal.py` reads
   `k_time` back out of the checkpoint. **This also makes the `sched:`-tail
   question moot for stage 1:** the tail is built by `scripts/probes_run.sh`
   and reaches `ml/temporal.py`, not the Train step, so a stage-1 flag could
   not have travelled that way regardless.
6. **`ml/recipes/f4r2-40M-monthblock.json`** — `f4r2-40M-nolonhold` with
   exactly two changes, `time_block: "month"` and `d_z: 64`.
7. **Tests**: `tests/test_e047_time_blocks.py` (7 checks: grouping, ragged
   edges, pad cells that point at real rows, the obs-mask fusion, block-centre
   phase against `season_feat_of`, fixed-N, the RAPID remap, and `k_time=1`
   identity) and `tests/test_e047_block_smoke.py` (4 checks end to end on a
   CPU toy: the codec trains, the checkpoint carries both new embeddings,
   `temporal.py` embeds one z per month onto a `YYYY-MM` axis and a stage-2
   head trains on it, `--time-block 2` works, the knob off is the per-bin
   codec, and a block codec refuses to be embedded as bins).

**Left, deliberately, and both SKIPPED LOUDLY rather than approximated:**

- **`ml/train.py`'s own end-of-run evaluation** is per-bin throughout (recon
  rows, a `t+1` that means the next bin, per-channel skills). Under blocking
  each of those is a different question. It records `eval_skipped` in
  `eval.json` and says so in the log.
- **`temporal.py`'s eval 2 (`chan_t+1`)** decodes ẑ to channel space, which
  under blocking names a CELL — and which cell stands for the block is an
  evaluation semantic nobody has chosen. It records `chan_t+1.skipped`.
  **eval 1 (`z_t+1`) is unaffected and is the headline** the falsifier in §4
  is written against.

## 8 · Relationship to E-046 (FSQ)

**Orthogonal, and they compose later if both pay.** E-046 quantizes the
ALPHABET a symbol is drawn from (continuous d_z → a finite grid); E-047 changes
WHAT ONE SYMBOL COVERS (one 5-day bin → one month of them). Neither needs the
other, and running them together would leave a result nobody can attribute —
the same reason this plan does not bundle the r3 Argo-fill tensor revision.
If both pay, an FSQ month-block codec is the natural third arm.

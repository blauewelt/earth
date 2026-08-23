# E-047 · A codec that fuses several pentad bins into one embedding

**Status: DESIGN + the axis half BUILT. One architectural decision is OPEN and
blocks the training wiring — §6.** Written 2026-08-22. Nothing has been
trained.

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

## 6 · THE OPEN DECISION — how the decoder queries a cell

**This blocks the training wiring and I am not choosing it unilaterally.**

Point 5 of the design says "masked-reconstruction training as today, masking
CELLS of the k_max x C grid; decoder reconstructs all cells". The encoder side
is specified down to the embedding. The decoder side is not, and the existing
decoder cannot express a cell query without a decision:

    query(z, chan_idx, off) ->  q_chan(chan) ++ q_off(off + max_off)     [ml/model.py:140]
    q_off = nn.Embedding(2 * max_abs_offset + 1, 16)   shared by (dx, dy, dt)

`off` is `(Δlon, Δlat, Δmonth)` and `max_abs_offset` is **3**, so the table
holds exactly 7 entries. Three ways to ask for "channel c of cell j":

**(a) Reuse the `dt` slot, centred.** `dt = j - 3` for `k_max = 7` fits the
existing 7-entry table exactly, adds no parameters, and changes no checkpoint
shape. **Cost:** it COLLIDES with the neighbour loss, which already uses
`dt = ±1` to mean *the next BIN* (`NEI` in `ml/train.py:547`). Under blocking,
one index would mean both "one bin later inside this block" and "one block
later", and the model cannot tell them apart.

**(b) A new query-time embedding**, `q_time = nn.Embedding(k_time, 16)`
concatenated into the decoder input — the exact mirror of the encoder's new
`time_emb`. Clean separation: `off` keeps meaning spatial/inter-block, `q_time`
means within-block position. **Cost:** the decoder's first layer grows by 16
(`d_z + 64 + 3*16` → `+ 16`), so a `k_time > 1` checkpoint has a different
decoder shape from every archived one. At `k_time = 1` nothing changes.

**(c) Drop the neighbour loss's temporal offsets while blocking**, and reuse
`dt` for the within-block position. No new parameters, no collision. **Cost:**
it removes a training signal that exists today, and changes what the neighbour
loss means in the one arm being compared against arms that have it.

**My recommendation is (b)** — it is the only one of the three where a symbol's
meaning does not depend on which loss term is reading it, the cost is 16 input
dimensions on a ~1.3M-parameter decoder, and `k_time = 1` stays bit-identical
so nothing archived moves. But it changes an architecture, so it is Chris's
call, not mine. Say the word and the wiring is a day.

## 7 · What is left after the decision

1. `--time-block` in `ml/train.py`: block assembly, the grid batch, the cell
   mask, the sizing print, `k_time`/`ctx_mode` into the args, and the refusal
   of `--time-block` together with `--time-stride` (one time-surgery at a time).
2. The embed path in `ml/temporal.py`: one z per block, labelled with the
   BLOCK's label; the pool, the persistence baselines, the monitor and the
   RAPID row remap all keyed on the block axis (`BlockAxis.remap_rows`, which
   is the same shape as the `--time-stride` remap that already exists).
3. The CPU toy smoke: tiny tensor → month-block codec trains a few steps →
   embeds → a tiny `temporal.py` head consumes the block-z without error.
4. A recipe file (`ml/recipes/f4r2-40M-monthblock.json`) so the dispatch is one
   token rather than fourteen fields.

## 8 · Relationship to E-046 (FSQ)

**Orthogonal, and they compose later if both pay.** E-046 quantizes the
ALPHABET a symbol is drawn from (continuous d_z → a finite grid); E-047 changes
WHAT ONE SYMBOL COVERS (one 5-day bin → one month of them). Neither needs the
other, and running them together would leave a result nobody can attribute —
the same reason this plan does not bundle the r3 Argo-fill tensor revision.
If both pay, an FSQ month-block codec is the natural third arm.

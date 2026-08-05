# Leaderboard — predictive skill of frozen embeddings

**The ranking metric is prediction, not reconstruction** (`trainprobe.py`,
runnable mid-training via `train.py --eval-every N`): freeze the codec as it
is, train a small fixed-seed temporal transformer on its embeddings (600
subsampled pixels + the full 26.5°N section, K=12, 400 steps), and score on
the blocked holdout — protocol v2 throughout (held-out years + mid-Atlantic
longitude block; deseasonalised RAPID target from train-years climatology;
lambda chosen on a train tail). ~40 s per measurement on 2 CPU cores.

Columns: **chan%** / **z%** = held-out t+1 error reduction vs persistence in
channel / embedding space (higher is better); **r_tmp** = RAPID probe
(deseasonalised) from the mini transformer's section-pooled hidden state;
**r_lin** = linear single-month section probe, deseasonalised. 36 held-out
RAPID months — r differences under ~0.1 are noise at this n.

## Ranked (anomaly-space codecs only)

| run | chans | codec steps | chan% | z% | r_tmp | r_lin | provenance |
|---|---|---|---|---|---|---|---|
| pilot4_anom | 4 | 8000 | **+29.3** | +31.4 | 0.291 | 0.307 | in-sandbox 2026-08-05; `runs/pilot4_anom/trainprobe.json` (backfilled) |
| pilot4_anom_smoke | 4 | 1500 | +25.0 | +31.5 | 0.360 | 0.300 | in-sandbox 2026-08-05; backfilled |
| actions #1 | 12 | 30000 | *pending* | | | | run 31028748779; backfill from `pixelmae-1` artifact via `trainprobe.py` when it lands |

The full (non-mini) stage 2 on pilot4_anom — K=24, d=96×3, 2000 steps
(`temporal.py`) — scores chan +32.4%, r_deseas 0.333: the mini probe tracks
it closely at 1/60th the cost, which is what makes it usable as a
training-time metric.

Reading the two 4-channel rows: 5× more codec training bought +4pt of
channel-space skill but no probe improvement — at 4 channels the probe is
data-limited, not optimisation-limited. The 12-channel run tests exactly
that.

## Provenance only (not rankable)

- **pilot12_anom** (2026-08-05, previous container — checkpoint lost, no
  trainprobe backfill possible): linear K-concat sweep r_deseas 0.38 (K=1)
  → 0.43 (K=3), holding to K=24; codec t+1 0.669 vs 0.753 persistence.
  Numbers from commit `67c4a6b` / `fd1efd7` messages.
- **pilot12 (state space)**: disqualified from ranking (commit `67c4a6b`) —
  embeddings seasonally redundant; K-sweep skill FELL with history. Kept as
  the negative result that motivated anomaly space.

## House rules

- Backfill every finished run: `python3 ml/trainprobe.py --run <name>`
  writes `runs/<name>/trainprobe.json`; copy the row here with provenance
  (where it ran, commit, artifact). Never rank a state-space codec.
- Curves during training land in `runs/<name>/metrics.jsonl`
  (`--eval-every`); the row here is the final measurement.
- The codec is FROZEN in every ranked measurement — the metric isolates
  representation quality. End-to-end fine-tuning (backprop into the
  encoder) is a separate, unranked experiment until it exists with a
  data-space grounding loss; see the discussion in `temporal.py`'s header.

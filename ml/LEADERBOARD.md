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

| run | chans | d_z | codec steps | chan% | z% | r_tmp | r_lin | curve | provenance |
|---|---|---|---|---|---|---|---|---|---|
| dz64 (#4) | 12 | 64 | 30000 | **+30.6** | +28.3 | 0.391 | 0.457 | [png](curves/dz64.png) | Actions run 31096118740 (2026-08-06), backfilled |
| actions #1 | 12 | 32 | 30000 | +30.5 | +27.8 | 0.361 | 0.428 | [png](curves/actions.png) | Actions run 31028748779 (2026-08-05); backfilled |
| dz16 (#3) | 12 | 16 | 30000 | +30.3 | +27.7 | 0.346 | 0.390 | [png](curves/dz16.png) | Actions run 31096115165 (2026-08-06), backfilled |
| pilot4_anom | 4 | 32 | 8000 | +29.3 | +31.4 | 0.291 | 0.307 | [png](curves/pilot4_anom.png) | in-sandbox 2026-08-05; backfilled |
| dz8 (#2) | 12 | 8 | 30000 | +28.6 | +28.3 | 0.223 | 0.416 | [png](curves/dz8.png) | Actions run 31096111610 (2026-08-06), backfilled |
| pilot4_anom_smoke | 4 | 32 | 1500 | +25.0 | +31.5 | 0.360 | 0.300 | [png](curves/pilot4_anom_smoke.png) | in-sandbox 2026-08-05; backfilled |

**The d_z verdict (2026-08-06 sweep, 4 codecs at matched 30k steps):** the
bottleneck saturates between 16 and 32 — chan% is flat from d_z=16 to 64
(30.3/30.5/30.6, inside seed noise) and only d_z=8 pays a real price
(−2 pts; squeezing 12 channels into 8 dims loses skill — a 12-channel
d_z=8 codec ranks BELOW the 4-channel d_z=32 pilot). Probe columns drift
upward with d_z but stay inside the ±0.57 CI (METRICS.md) — and the
year-blocked k-fold probe (3× sharper; METRICS.md) then resolved that
drift into a real monotone gain: k-fold r 0.11/0.15/0.18/0.31 for d_z
8/16/32/64, every CI excluding zero. AMENDED VERDICT: d_z=32 suffices
for field prediction, but the TRANSPORT read-out keeps gaining to
d_z=64 — the probe wants a wider bottleneck than the reconstruction
loss rewards. A d_z=128 run is the natural next dispatch. These three runs are also the
first born fully instrumented: dense loss curves + 4-point probe curves
rendered on the runners themselves.

Reading the top row against the 4-channel rows: adding the RG depth
structure (T/S at 10/200/700/1500 dbar) moved BOTH probe reads up — linear
0.31→0.43, temporal 0.29→0.36 — while channel-space dynamical skill held at
~+30%. Direction is consistent across two independent read-outs, but the
margins are ~1σ at 36 held-out months, so this is *evidence for* the
density-structure hypothesis, not proof; the probe is now target-limited
(more RAPID months don't exist — OSNAP/MOVE/SAMBA as extra probes is the
next lever). Caveat for the record: run #1's own FULL stage-2
(`temporal.json`: K=24, d=96×3, 6000 steps) scored r_deseas only 0.13 on
the same codec where the standardized mini probe scores 0.36 — one more
instance of single-seed probe variance at this n, and the reason the
leaderboard ranks the fixed-seed mini probe, not bespoke stage-2 runs.

Every run renders `runs/<run>/curve.png` automatically at the end of
training (`plot_run.py`: steps vs loss, steps vs held-out t+1 skill, steps
vs probe r; dense when the run logged `--eval-every` curves, sparse —
log-reconstructed loss + one backfilled probe point — for runs that predate
the hook). Sandbox runs publish theirs to `ml/curves/` (committed, since
sandbox containers are ephemeral); Actions runs ship theirs in the
checkpoint artifact.

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

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

## The frontier — GLOBAL window (2026-08-07, run #8)

The first codec trained on the whole ocean: 44,964 cells (7.8× the
pilot), 186.7 M observed values, same 14 channels, d_z=64, 30k steps
(0.8 epochs — fully fresh data). chan%/z% are on the global tensor's own
persistence baseline (not comparable to any NA row).

| run | chans | d_z | chan% | z% | r_tmp | r_lin | k-fold RAPID [CI] | curve | provenance |
|---|---|---|---|---|---|---|---|---|---|
| global14 (#8) | 14 | 64 | +30.6 | +32.2 | 0.54 | 0.495 | **0.602 [0.461, 0.728]** | [png](curves/global14.png) | Actions ml-train #8 (2026-08-07) |
| global14b (#11 codec) | 14 | 64 | +30.9 | — | — | — | 0.556 [0.434, 0.676] | — | replication: identical config, different runner |

The #11-codec replication measures codec-training seed noise on the
defensible metric for the first time: k-fold spread ≈ ±0.05, chan%
within the known ±0.5 band, every multi-target null replicated
(FC 0.081/0.102, MOVE 0.049/0.071, OSNAP 0.100/0.126, SAMBA
−0.028/−0.070), and the wind-only baseline bit-identical (0.531 — it
bypasses the codec, as it must). Caution recorded: the 18-month
low-passed r swung 0.64→0.41 between seeds — few effective DOF; never
argue from it alone.

The transfer verdict: going global cost the Atlantic read-out NOTHING —
k-fold 0.602 vs the NA champion's 0.604 (v3 bridge), dip capture 50%
with the best sign agreement yet (70%). Comparability grades (METRICS.md
/ LITERATURE.md): RMSE 2.23 Sv (σ 2.79), 18-mo low-passed r 0.639,
wind-only ridge baseline 0.531 — the honest decomposition is density
structure ~0.3 + wind ~0.53 → combined 0.60. Rollout ACC 0.62 at 1
month, crossing the 0.5 useful-skill line near 4 months, 0.36 at 12;
damped persistence beaten at every horizon. The generic-embedding hypothesis
holds at fixed capacity. Its own stage-2 (d96×3) scores chan +35.1% on
this tensor. First in-distribution multi-target probe: FC +0.102, MOVE
+0.071, OSNAP +0.126, SAMBA −0.070 — all CIs include zero. The embedding
carries the 26.5°N overturning specifically; the other arrays await more
capacity, deeper read-outs, or longer records (OSNAP's lean is the one
to watch). The preview's SAMBA −0.27 (NA-trained codec on global cells)
is confirmed as a distribution-shift artifact.

## The NA-window frontier — 14-channel tensor (wind stress added 2026-08-06)

chan%/z% below are NOT comparable with the 12-channel table (different
channel set → different persistence baseline; house rule 5). Cross-tensor
judging goes through the k-fold probe and the 2009-10 case study.

| run | chans | d_z | codec steps | chan% | z% | r_tmp | r_lin | k-fold r [CI] | curve | provenance |
|---|---|---|---|---|---|---|---|---|---|---|
| wind14 (#6) | 14 | 64 | 30000 | +35.6 | +33.9 | 0.537 | 0.47 | **0.586 [0.451, 0.720]** | [png](curves/wind14.png) | Actions ml-train #6 (2026-08-07) |

Adding NCEP wind stress (τx, τy) nearly doubled the defensible probe
(12-ch d_z=64: 0.308 [0.13, 0.46]) — the new CI excludes the old point
estimate — and tripled the captured amplitude of the winter 2009-10
collapse (16% → 50% of the observed dip, out-of-fold at matched d_z=64;
`dip_check.py`, FINDINGS.md §4.6).
Its own full stage-2 (d256×5) scores chan +41.6% on this tensor.
Physics note: RAPID transport contains an Ekman component computed from
wind, so this is the embedding acquiring a real ingredient of the
target — the right channels, not a modelling trick (FINDINGS.md §4.6).

## Protocol v3 (2026-08-07): the Atlantic-clipped section

Going global made the old section definition wrong: the grid row nearest
26.5°N would circle the planet through the Pacific. Protocol v3 clips
the probe section to RAPID's Atlantic span (80°W–13°W) in all probe
scripts. Bridge measurement on the champion (same NA tensor, same
codec): wind14 k-fold v2 **0.586** [0.451, 0.720] → v3 **0.604**
[0.474, 0.720] — the clip is benign (if anything it sharpens: the
dropped Gulf-of-Mexico and NW-African shelf cells were never part of
the physical array). NA-window numbers below are v2; global-window runs
are scored under v3 and comparisons across that line carry this note.

## The defensible probe ranking (year-blocked k-fold, all 240 months)

| codec | window | chans | d_z | k-fold r | 95% CI |
|---|---|---|---|---|---|
| **wind14** (v3: 0.604) | NA | 14 | 64 | **0.586** | [0.451, 0.720] |
| **global14** (v3) | global | 14 | 64 | **0.602** | [0.461, 0.728] |
| dz64 | NA | 12 | 64 | 0.308 | [0.13, 0.46] |
| actions #1 | NA | 12 | 32 | 0.182 | [0.05, 0.31] |
| dz128 | NA | 12 | 128 | 0.166 | [0.072, 0.295] |
| dz16 | NA | 12 | 16 | 0.151 | [0.01, 0.28] |
| dz8 | NA | 12 | 8 | 0.111 | [0.01, 0.20] |

Every CI excludes zero. This table is the project's headline, and it
now has shape: r rises with d_z to a **peak at 64, then turns over at
128** (ridge features vs ~220 truth months per fold — FINDINGS.md
§4.3), and one channel family (wind) is worth more than every width
step combined.

## Ranked — 12-channel tensor (anomaly-space codecs only)

| run | chans | d_z | codec steps | chan% | z% | r_tmp | r_lin | curve | provenance |
|---|---|---|---|---|---|---|---|---|---|
| dz64 (#4) | 12 | 64 | 30000 | **+30.6** | +28.3 | 0.391 | 0.457 | [png](curves/dz64.png) | Actions run 31096118740 (2026-08-06), backfilled |
| dz128 (#5) | 12 | 128 | 30000 | +30.2 | +26.3 | 0.338 | 0.528 | [png](curves/dz128.png) | Actions ml-train #5 (2026-08-07), backfilled |
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
drift into a real gain: k-fold r 0.11/0.15/0.18/0.31 for d_z 8/16/32/64,
every CI excluding zero. AMENDED VERDICT (completed 2026-08-07): d_z=32
suffices for field prediction, the TRANSPORT read-out keeps gaining to
d_z=64 — and the d_z=128 dispatch answered the open question by TURNING
OVER (k-fold 0.166 [0.072, 0.295], chan% flat at 30.2). The curve peaks
at 64; that is the working bottleneck until the truth series grows.
dz128 also set the all-time fixed-holdout r_lin record (0.528) while
k-fold read 0.166 — the definitive exhibit for why single-holdout probe
draws are never trusted here. These runs are also the first born fully
instrumented: dense loss curves + 4-point probe curves rendered on the
runners themselves.

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

- **xxlarge stage-2 attempt** (2026-08-06, in-sandbox): d320×6 (7.4 M
  params), the rung above xlarge on the width×depth curve — zero steps
  in 8 h on the sandbox's 2 CPU cores; killed. Needs the training
  workflow to accept stage-2 size inputs so it can run on Actions
  runners. The scaling curve past 4 M params is therefore still
  unmeasured, with the last measured point (xlarge, 37.7%) still rising.
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

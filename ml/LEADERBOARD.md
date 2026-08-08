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

## Master table — every codec run, one table

chan% is comparable only WITHIN one channel set (its persistence baseline
moves when channels change — house rule 5); the cross-run columns are the
year-blocked k-fold RAPID r and RMSE in Sv. Dip = share of the 2009-10
collapse amplitude captured out-of-fold. Wind-stress-only ridge baseline
(no embedding): r 0.531, the line every codec must beat.

| run | window | C | patch | d_z | steps | chan%† | k-fold RAPID r [95% CI] | RMSE Sv | dip | status |
|---|---|---|---|---|---|---|---|---|---|---|
| pilot4_anom | NA | 4 | 1 | 32 | 8k | +29.3 | — | — | — | done |
| dz8 (#2) | NA | 12 | 1 | 8 | 30k | +28.6 | 0.111 [0.01, 0.20] | — | — | done |
| dz16 (#3) | NA | 12 | 1 | 16 | 30k | +30.3 | 0.151 [0.01, 0.28] | — | — | done |
| actions (#1) | NA | 12 | 1 | 32 | 30k | +30.5 | 0.182 [0.05, 0.31] | — | — | done |
| dz64 (#4) | NA | 12 | 1 | 64 | 30k | +30.6 | 0.308 [0.13, 0.46] | — | 16% | done |
| dz128 (#5) | NA | 12 | 1 | 128 | 30k | +30.2 | 0.166 [0.072, 0.295] | — | — | done |
| wind14 (#6) | NA | 14 | 1 | 64 | 30k | +35.6 | 0.604 [0.474, 0.720] | — | 50% | done |
| global14 (#8) | global | 14 | 1 | 64 | 30k | +30.6 | **0.602** [0.461, 0.728] | 2.23 | 50% | done |
| global14b (#11 codec) | global | 14 | 1 | 64 | 30k | +30.9 | 0.556 [0.434, 0.676] | 2.34 | — | replication |
| global15sst (#10) | global | 15 | 1 | 64 | 30k | +30.8 | 0.582 [0.43, 0.71] | 2.27 | 47% | done |
| patch24_40k (#18) | global | 24 | **3** | 64 | 40k | — | 0.543 [0.428, 0.659] | 2.36 | 27% | done |
| pixel25_40k (#17) | global | 25 | 1 | 64 | 40k | — | 0.536 [0.378, 0.683] | 2.35 | **59%** | done |
| pixel24 (#21/#22) | global | 24 | 1 | 64 | 40k/30k | — | … | … | … | queued (controls) |
| patch24_30k (#19r) | global | 24 | 3 | 64 | 30k | — | … | … | … | queued |
| patch25_30k (#20) | global | 25 | 3 | 64 | 30k | — | … | … | … | running |

† within-tensor only. NA k-fold values are protocol v2 except wind14
(v3 bridge); global rows are v3. RMSE backfill for NA runs pending.

**Two channel families, and they are not comparable.** Everything up to
global15sst was built when `fetch_rg_channels.py` sampled 4 Argo pressure
levels; it now samples 8, so every run from #17 onward carries 24 channels
(25 with monthly SST) whether or not anyone asked for it. Compare within a
family, never across — and read C from the checkpoint, never from a run's
name, which is how `patch14_40k` spent an hour mislabelled before being
renamed `patch24_40k`.

**The channel expansion COST monthly RAPID skill (2026-08-08).** Both
24/25-channel codecs land essentially on the wind-only baseline: 0.543 and
0.536 against 0.531, i.e. margins of +0.012 and +0.005, where the
14-channel codecs held +0.07. Two independent runs agreeing makes it a
pattern rather than seed noise, though the CIs do overlap the 14-channel
values. The working explanation: d_z=64 was tuned at C=12-14, and sixteen
temperature and salinity levels now compete for the same bottleneck that a
*linear* probe must then read a single projection of.

**But the same channels HELPED at MOVE, exactly where the physics says they
should.** pixel25_40k scores 0.206 [0.044, 0.346] at 16°N — the first
multi-target result whose CI excludes zero — with an 18-month low-passed r
of 0.623, against 0.111 and 0.379 for global15sst. MOVE measures the deep
western-boundary flow, and what the expansion added was T/S down to 1900
dbar. So this is not "more data hurt": it is one shared 64-dimensional
bottleneck being asked to serve a wind-dominated target and a
density-dominated one at once. pixel25_40k also captures 59% of the
2009-10 dip, the best of any run, so event anatomy improved while the
monthly correlation fell.


## Family 3 — the 0.25° North Atlantic (opened 2026-08-08)

Third channel family: `ml/build_family3.py` → `family3_na025.npz`,
0.25° NA window (84,405 ocean cells, 14.6× the 1° NA), axis 1982-01..
2024-12 (pre-1993 base months are missing tokens; wind covers all 516
months; the Florida cable's 1982-92 decade is truth), C=39 (cur_speed /
log_mld / **ssh** + RG T/S at 16 levels, bilinear from 1° and therefore
1°-smooth + NCEP τ mean and within-month std, both from the daily
files). 822.6 M observed values → Chinchilla anchor 41.1 M params
(SCALING.md). **No number in this block is comparable to families 1/2.**

| run | C | patch | d_z | params | steps | k-fold RAPID r [95% CI] | status |
|---|---|---|---|---|---|---|---|
| f3_pilot_40k (#44) | 39 | 3 | 64 | 0.92M | 40k | … | running |
| f3_anchor41M (#47) | 39 | 3 | 64 | 40.7M | 60k (~1 epoch) | … | queued |

The pair reruns the family-2 capacity question on 4.4× the data: the
pilot control isolates what 0.25° data buys at fixed capacity, the
anchored run is the first codec this project has trained AT its data
anchor. A steps-matched anchored 40k control follows if the 60k result
warrants it. Wind-only ridge baseline for THIS tensor: to be measured
(the family-2 0.531 does not carry over).


## The probe ladder (2026-08-08): pooling was hiding the signal

Three read-outs of the SAME frozen embeddings, same year-blocked folds,
increasing only in what they are allowed to see (probe_kfold --probe,
probe_head.py):

| codec | ridge (pooled) | MLP (pooled) | attention head (unpooled section) |
|---|---|---|---|
| global14 (C=14) | 0.602 | 0.582 | **0.635** [0.53, 0.74] |
| pixel25 (C=25) | 0.536 | 0.514 | **0.617** [0.49, 0.73] |

Two findings, cleanly dissociated. **Pointwise nonlinearity adds
nothing**: the MLP trails the ridge everywhere, so the ridge numbers
measure the representation, not the read-out. **Spatial structure adds a
lot**: one attention query over the ~67 unpooled section pixels (with a
longitude encoding) beats every pooled probe — and the 25-channel codec
recovers from 0.536 to 0.617, statistically indistinguishable from the
14-channel codec. Mechanistically this is the thermal-wind story:
geostrophic transport is the east-minus-west density difference across
the section, and mean-pooling averages exactly that difference away. The
deeper channel set pushed more of the transport signal into cross-pixel
structure, which the pooled probes then billed as a "cost".

Consequences: the "channel expansion cost monthly RAPID skill" reading
above is RETRACTED as a representation claim — it was a property of the
pooled probes. The MOVE gain stands (measured pooled; the unpooled
version can only raise it). The probe ladder is now part of the standard
suite; the ridge remains the comparable-across-time instrument, the head
the capability bound.


## The attribution matrix (2026-08-08, late): what the 0.690 is made of

The end-to-end baseline Chris asked for — the SAME attention head fed raw
section anomalies instead of embeddings — with the receptive field
controlled (probe_head --raw / --raw --raw-patch):

| tokens carry | raw data | codec embedding |
|---|---|---|
| single pixel | 0.613 [0.48, 0.73] | 0.617 [0.49, 0.73] |
| 3x3 neighbourhood | 0.659 [0.55, 0.75] | **0.690** [0.57, 0.78] |

Decomposition of the headline number, in descending order: UNPOOLING the
section (ridge 0.54 -> head 0.61+) is the largest term; the 3x3 RECEPTIVE
FIELD adds ~+0.045 and does so on raw data alone; PRETRAINING adds +0.031
at matched receptive field — positive in direction, not significant (CIs
overlap heavily), and ~+0.004 at single pixel, i.e. nothing.

Stated plainly: for the RAPID monthly probe, a supervised attention head
over raw section data recovers nearly everything, and the codec's measured
contribution is small. The pretraining case now rests on the scoreboards
where raw supervision has no answer: field prediction (stage-2 beats
persistence everywhere; an end-to-end field-forecaster baseline is the
open follow-up), the MOVE transfer (raw-baseline pending there), and the
open question of whether the pretraining margin GROWS with codec capacity
— the 10M runs' head probes will say. Two lessons for the paper: report
the matrix, not the corner of it; and the demand for a simpler baseline
reframed the headline twice in one day.

### The second seed (2026-08-08, afternoon): 0.690 does NOT hold up as a margin

Chris asked for the 0.690 to be replicated on a second codec seed. It was
(run #43, identical config, different runner), and the result reframes the
attribution matrix:

| probe | patch24 seed 1 | patch24 seed 2 | spread |
|---|---|---|---|
| pooled ridge | 0.543 [0.43, 0.66] | 0.531 [0.40, 0.65] | 0.012 |
| attention head | **0.690** [0.57, 0.78] | **0.654** [0.54, 0.75] | 0.036 |

Two things follow, one reassuring and one not:

1. The HEADLINE SURVIVES as a capability claim. Both seeds beat every
   pooled probe and the wind-only baseline by a wide margin; the unpooling
   result is not a fluke. But the honest headline is the two-seed mean
   **0.672**, and 0.690 was the luckier draw.
2. The PRETRAINING MARGIN DOES NOT SURVIVE. Raw-3x3 (codec-independent)
   scores 0.659. Seed 2's embedding head scores 0.654 — BELOW the raw
   baseline. Across both seeds the embedding advantage is
   0.672 - 0.659 = **+0.013**, i.e. nothing. The earlier "+0.031, n.s."
   was one seed's noise read as a direction.

Note also that the head is a NOISIER instrument than the ridge (seed
spread 0.036 vs 0.012) — expected for the more expressive probe, and a
reason every future head claim needs two seeds before it is stated.

For the pixel control the same replication gives 0.515 -> 0.564
(spread 0.049), consistent with the +/-0.05 seed-noise estimate that has
held all programme long.

### Head capacity sweep (2026-08-08): the margin is not a parameter artifact

Chris asked whether the raw baseline was parameter-bottlenecked. It is the
opposite — BOTH columns of the matrix degrade when the head grows
(--head-dim 128 --head-blocks 1, ~325k params vs the standard ~47k):

| head size | raw 3x3 | embedding 3x3 |
|---|---|---|
| ~47k params | 0.659 [0.55, 0.75] | **0.690** [0.57, 0.78] |
| ~325k params | 0.590 [0.44, 0.72] | 0.611 [0.48, 0.72] |

220 labels per fold cannot feed 325k parameters; the standard head is
already past the optimum, and the raw baseline loses MORE than the
embedding column when oversized. Chinchilla logic applied to the head
(labels/20 -> ~10-50k params at 220-fold labels) lands exactly where the
standard head sits. The +0.02-0.03 embedding margin persists at both
sizes; it is small at every capacity, not an artifact of any one.


## The 1M-step headroom run (2026-08-08, run #30): flat from 50k on

Chris's protocol — decay LR to a 10% floor by 50k, then run at constant
LR to 1M steps with probes every 50k, abort when flat. It completed in
2h39m on one 4090 (104 steps/s) and the answer is unambiguous:

    step      50k   200k   400k   600k   800k    1M
    linear r  0.44   0.44   0.47   0.46   0.47   0.46
    chan%     32.1   32.3   32.3   32.3   32.2   32.1

Every probe metric (linear r raw/deseas, temporal r, chan-vs-persistence,
z-vs-persistence) oscillates within seed noise from the FIRST probe to the
last: 20x the compute of the pilot regime buys nothing at 0.92M params.
k-fold on the final checkpoint: RAPID 0.571 [0.42, 0.70] — within noise of
patch24_40k's 0.543. The pilot is CAPACITY-limited, not compute-limited,
which is precisely what the Chinchilla arithmetic said (~13M params for
this tensor) and why the 10M runs exist. Checkpoint: ml/runs/patch24_1M.


## Stage-2 results withdrawn (poisoned embedding cache, 2026-08-07)

The stage-2 numbers for **global14b** and **global15sst** are WITHDRAWN, not
merely stale. Both were trained on an Actions runner whose `ml/cache`
carried run #8's `Z_actions.npy`; the shape check `(T, P, d_z)` matched, so
stage 2 trained happily on a *different codec's* embeddings. The signature
is unmistakable in hindsight — healthy z-space skill next to catastrophic
decoded skill (chan% -33.1 and -60.3), because the z they learned to
predict is not the z their decoder speaks. `temporal.py` now names its
cache `Z_<run>_<weight-hash>.npy`, so a stale cache is a miss and never a
lie, and run #17 confirms the fix at chan% +35.5.

Their `temporal.json` files are renamed `temporal.WITHDRAWN.json` so the
generated table prints "—" rather than a wrong number. They were not
regenerated: both codecs belong to the superseded 14/15-channel family, and
several hours of embedding to restore a number nobody should cite is worse
value than not claiming it. The codec-level rows for both runs are
unaffected — the bug lived entirely in stage 2.


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

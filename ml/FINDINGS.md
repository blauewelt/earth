# Findings — what we understand so far, and how we know it

The one document that holds the whole story: the question, the system,
the metrics, every experiment with its numbers, the methodological
lessons, and what's next. Companion references: [METRICS.md](METRICS.md)
(each metric defined, with measured statistical power),
[SCALING.md](SCALING.md) (data-sufficiency audit),
[LEADERBOARD.md](LEADERBOARD.md) (ranked runs, updated per run). Last
full revision: 2026-08-07.

## 1 · The question

Can a self-supervised embedding of the ocean's observable state predict
how that state evolves — and does it carry the Atlantic overturning
circulation (AMOC), which no input ever measures directly? The AMOC is
the project's north star because it is climate-critical, directly
monitored (RAPID array, 26.5°N, 2004→) yet short-recorded, and
physically composite: geostrophic shear (density structure) + Ekman
transport (wind) + boundary terms.

## 2 · The system

**Data tensor** (`build_dataset.py` + `fetch_*_channels.py`): monthly,
1°, 1993-01→present. Every experiment below ran on the pilot
North-Atlantic window (100°W–20°E, 0–70°N, 5,787 ocean cells); as of
2026-08-07 the default window is **global** (the full baked GLORYS grid,
−80°S–90°N, ~7× the cells — `--window na` reproduces the pilot), with
the RAPID probe section clipped to the array's Atlantic span (protocol
v3; the clip is bridge-measured as benign, LEADERBOARD.md). Channels have grown 4 → 12 → 14: GLORYS surface-current
speed + mixed-layer depth (1993→), OISST/GPCP static climatologies,
RG-Argo T & S at 10/200/700/1500 dbar (2004→), NCEP wind stress τx/τy
(1948→, added 2026-08-06). RAPID monthly transports ride along as the
never-input truth series.

**Stage 1 — codec** (`model.py`, `train.py`): PixelMAE, ~0.9 M params —
a masked autoencoder over ONE pixel's channels with explicit
missing-tokens (absence is information), whose 32-d bottleneck must
reconstruct masked channels AND predict spatial/temporal neighbours.
Trained in **anomaly space** (departures from each pixel's own
train-years monthly climatology) with blocked splits: whole held-out
years (2009, 2017, 2023) plus a held-out mid-Atlantic longitude block —
never random splits.

**Stage 2 — temporal transformer** (`temporal.py`): a causal transformer
over each pixel's frozen embedding sequence (K=24 months), predicting
next month's embedding; scored after decoding back to physical channels.
The codec is frozen by design — gains stay attributable, and the naive
end-to-end variant would let the temporal loss collapse the embedding it
predicts (an end-to-end run needs a data-space grounding loss; unranked
until it exists).

**Probes** (`trainprobe.py`, `probe_sequence.py`, `probe_kfold.py`,
`dip_check.py`): ridge read-outs from the 26.5°N section's embeddings to
RAPID transport, deseasonalised, λ chosen on a train tail — in three
grades of statistical power (fixed holdout → mini-temporal suite →
year-blocked k-fold with block-bootstrap CIs).

## 3 · The metrics, in one paragraph

**chan%** (held-out next-month prediction error reduction vs persistence,
in physical units, in anomaly space — so the seasons can't be claimed as
skill) is the primary, decision-grade metric: ±0.5 pt seed spread,
differences ≥2 pts real. **Probe r** against RAPID is the mission metric
but statistically coarse: RAPID's autocorrelation (0.41) makes 36
held-out months worth ~15 samples (SE 0.29, CI ±0.57); the year-blocked
**k-fold** protocol tests all 240 months once each (SE ~0.10) and is the
only probe read worth arguing over. Full definitions and the reliability
tiers: [METRICS.md](METRICS.md).

## 4 · What the experiments established

### 4.1 Anomaly space, not state space (the founding negative result)

A codec trained on raw state aces reconstruction by memorising the
seasonal cycle, and its embeddings are seasonally redundant — stacking
months made the probe WORSE. Subtracting each pixel's own train-years
monthly climatology fixed both: the first t+1-beats-persistence result
and probe skill that grows with history. State-space runs are
permanently disqualified from ranking.

### 4.2 Dynamics are real, and stage-2 capacity buys them

The stage-2 transformer beats persistence on held-out years by a third,
and a steps-matched sweep (2 seeds per config, same frozen 12-ch
embeddings) showed width×depth — not context length, not more steps —
is the axis:

| stage-2 config | params | steps | chan% (seeds) |
|---|---|---|---|
| small d64×2 K12 | 0.11 M | 2000 | 30.1 (30.7/29.6) |
| mid d96×3 K24 | 0.35 M | 2000 | 30.9 (31.4/30.4) |
| mid, longer | 0.35 M | 4000 | 31.9 (32.4/31.4) |
| large d192×4 | 1.81 M | 4000 | 35.9 (35.5/36.2) |
| xlarge d256×5 | 3.98 M | 4000 | **37.7** (37.6/37.7) |

The curve is still rising at the value-count Chinchilla anchor,
decelerating log-linearly. K∈{6,36} at mid size changed nothing. The
next rung (d320×6, 7.4 M) exceeded the sandbox (zero steps in 8 h on 2
CPU cores) — it needs runner support. Practical footnote: doubling
training steps bought +1 pt where 2.2× parameters bought +1.8.

### 4.3 The bottleneck: 32 dims suffice for the field, the probe wants more

Four 12-channel codecs at matched 30k steps: chan% 28.6 / 30.3 / 30.5 /
30.6 for d_z 8/16/32/64 — the field saturates by 16–32, and only d_z=8
truly pays (a 12-channel d_z=8 codec ranks below the 4-channel pilot).
But the k-fold probe rises with width — r 0.111 → 0.151 → 0.182 →
**0.308** (d_z 8→64), every CI excluding zero — and then **turns over**:
d_z=128 scores 0.166 [0.072, 0.295] (harvested 2026-08-07), with chan%
flat at 30.2. The curve peaks at 64. The likely mechanism is truth-data
scale, not representation quality: 128 features against ~220 training
months per fold is ridge-overfitting territory, and spreading
transport-relevant variance over twice the dims dilutes the section
read-out. Bonus cautionary tale: dz128's *fixed-holdout* linear probe
read 0.528 — the highest single draw ever — while the k-fold said
0.166. The rubber ruler flatters widest exactly where overfitting is
easiest. Verdict: **d_z=64 is the working bottleneck** until the truth
series grows.

### 4.4 Which datasets matter (mask-token ablation, frozen 12-ch codec)

| condition | chan% | r_lin | reading |
|---|---|---|---|
| full | 29.4 | 0.43 | — |
| drop GLORYS (currents+MLD) | 25.3 | **0.12** | biggest unique loss; carries the probe |
| drop RG-temperature | 25.2 | 0.18 | equal unique loss for the field |
| drop RG-salinity | 26.8 | 0.40 | modest; T and S largely redundant here |
| drop statics | 29.4 | 0.43 | **exactly zero** — coordinates already encode them |
| only RG-T | 22.3 | 0.03 | best solo field carrier; no transport alone |
| only GLORYS | 11.1 | 0.18 | weak alone, irreplaceable in company |

The probe is a *conjunction* — no group carries it alone — matching the
physics (shear + Ekman). Statics are dead weight; future channels should
be dynamic.

### 4.5 The probe, measured honestly

The original fixed-holdout probe was a rubber ruler (±0.57 CI): the same
config drew 0.09 and 0.33 on two seeds; 0.4+ draws were flattering. The
year-blocked k-fold (all 240 months tested once; block bootstrap)
delivered the first defensible claims: the embeddings carry real
deseasonalised AMOC signal (all CIs exclude zero), true effect ~0.1–0.3
at 12 channels (peaking at d_z=64, §4.3) and ~0.45–0.72 once the wind
is in (§4.6). "Hold out twice as much" was considered and rejected by
arithmetic: √2 improvement vs k-fold's 3×.

### 4.6 Case study — the winter 2009-10 collapse, and the wind

The most dramatic event in the RAPID record (~30% AMOC drop, strongly
wind-driven via the extreme negative-NAO winter). The 12-channel codec's
out-of-fold prediction got the January–March sign right and the spring
recovery, but missed the November onset entirely and read −1.1 Sv
against an observed −6.9 dip mean: the tensor had no wind. NCEP R1
momentum flux (τx, τy, 1948→) was added 2026-08-06; the 14-channel
codec's rematch (`dip_check.py --run wind14`, all out-of-fold):

| out-of-fold, matched d_z=64 | 12-ch dz64 (no wind) | 14-ch wind14 |
|---|---|---|
| dip-window mean (obs −6.87 Sv) | −1.13 (16%) | **−3.43 (50%)** |
| deepest month read | −3.96 (Mar) | −5.74 (Feb, obs −7.12) |
| November onset | missed (+1.9 vs −5.69) | still missed (+0.5 vs −5.69) |
| full-record out-of-fold r | 0.308 [0.13, 0.46] | **0.586 [0.451, 0.720]** |
| sign agreement | 62% | 68% |

Adding the wind tripled the captured dip amplitude and nearly doubled
the full-record k-fold probe — the new CI excludes the old point
estimate, so this is the largest single defensible gain of the project.
The November onset is still missed: the codec sees each month's wind but
has no submonthly information, and the 2009-11 collapse was triggered
inside the month. Honest attribution: RAPID's transport *contains* an
Ekman component computed from wind stress, so τx/τy hand the embedding a
direct physical ingredient of the target. That is adding the right
physics, not leakage (the probe still has to find it in a 64-d summary
of 14 channels, and the geostrophic majority of the signal still needs
the density structure) — but the gain belongs to the channels, not to
any modelling change.

### 4.7 Scaling reads (details: SCALING.md)

The codec sits at its data anchor (18.1 M observed values ≈ 20×0.9 M
params) and past the useful-repetition zone at 30k steps — feed it
channels, don't grow it. Stage 2's anchor must be counted in VALUES
(transitions × d_z), not transitions — the token-count anchor predicted
large would fail, the experiment said +4 pts, and the correction is
recorded under the wrong prediction. The probe's binding constraint is
truth months, which no model choice fixes.

## 5 · Methodological lessons (paid for, keep them)

1. **Measure the ruler before the measurement.** Autocorrelation-adjusted
   power analysis turned probe chaos into interpretable numbers.
2. **In-distribution ablation or no ablation.** Hiding a channel via the
   missing-token was out-of-distribution for always-present channels and
   fabricated a spectacular artifact (statics "destroying" the probe);
   the mask-token pathway — seen constantly in training — measured truth.
3. **Steps-matched controls before scaling conclusions**, and anchor
   continuous sequences on value-count.
4. **Persistence in anomaly space is the honest baseline** — seasons are
   subtracted from both sides.
5. **chan% comparisons only hold within one channel set** — persistence
   MSE moves when channels change; cross-tensor judging goes through the
   probe and case studies.
6. **Archives lie in their own ways**: EOT20's estuary blow-ups, NCEP's
   directory maze, PSL truncated downloads — verify before rename, cap
   with evidence, never guess what a server serves.
7. **Single-seed probe values are compass needles.** Two seeds minimum;
   k-fold for verdicts.

## 6 · Best current configuration

**14-channel anomaly tensor** (wind in) · PixelMAE codec **d_z=64** —
the measured optimum: field-MSE saturates by 32, the probe peaks at 64
and turns over at 128 — 30k steps on runners · stage-2 d256×5, 4000
steps · judged by chan% (within-tensor), k-fold probe r with CI (across
everything), and the 2009-10 case study. Best defensible numbers as of
this revision: **37.7%** next-month error reduction vs persistence
(12-ch, xlarge stage-2; the 14-ch equivalent scores 41.6% on its own
tensor, not cross-comparable); k-fold probe **0.586 [0.451, 0.720]**
(14-ch, d_z=64); **50%** of the 2009-10 collapse amplitude captured
out-of-fold.

## 7 · Open questions, ranked

1. **More truth — LANDED 2026-08-07 afternoon** (`fetch_truth.py`): four
   arrays now ride in the tensor as never-input truth: Florida Current
   cable (daily 1982→, 359 months inside the axis), MOVE 16°N (271),
   OSNAP subpolar (96), SAMBA 34.5°S (66) — 792 truth months beside
   RAPID's 240, each probed from its own zonal section
   (`probe_kfold.py --runs X`, TARGETS table). First preview (NA-trained
   wind14 read over the global tensor — distribution-shifted, treat as a
   stress test): RAPID 0.607 holds; FC/MOVE/OSNAP null; SAMBA reads
   −0.27 on a never-trained section. The first in-distribution
   multi-target verdict comes from the first global codec.
2. **More channels — monthly SST staged** (`fetch_sst_channel.py`, from
   the repo's own baked OISST, 1981→, zero download): appended as channel
   15 for the next codec dispatch (`sst_channel: true` workflow input).
   ERA5 wind (needs CDS credentials) and DUACS SLA remain the follow-ups.
3. **Stage-2 past 4 M params** on runners (workflow needs stage-2 size
   inputs); multi-horizon (t+3, t+6) and spatial-attention objectives.
4. **End-to-end fine-tuning** with a data-space grounding loss —
   unranked experiment until designed properly.
5. **Wider windows — IN PROGRESS (2026-08-07)**: the channel set having
   stabilised at 14, the pipeline now defaults to the global grid
   (~42k ocean cells, ~7× the pilot's data — which moves the codec's
   Chinchilla anchor from ~1.1 M to ~8 M params, reopening codec growth
   for the first time). First global codec dispatched on runners.

## 8 · Reproduction

Everything runs from the repo: `build_dataset.py` →
`fetch_rg_channels.py` → `fetch_wind_channels.py` (tensor);
`train.py --anomaly --eval-every N` (codec + live metrics);
`temporal.py --tag ... --seed ...` (stage 2); `trainprobe.py --run X`
(standardized backfill); `probe_kfold.py --runs ...` (the serious probe);
`dip_check.py --run X` (case study); `plot_run.py --run X --publish`
(curves); `scaling_audit.py` (the counts). Runner training:
`.github/workflows/ml-train.yml` (inputs: steps, d_z, anomaly,
temporal_steps, eval_every). Sandbox quirks and push mechanics:
CLAUDE.md §1.

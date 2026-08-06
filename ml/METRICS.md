# Metrics glossary — what each number means and how much to trust it

Every number in the leaderboard, the training curves, and the commit
messages, defined once. Powers and standard errors are **measured**
(`scaling_audit.py`, 2026-08-06), not guessed. The one-line summary:
**trust channel-space skill, treat probe r as a coarse compass.**

## The metrics

**rec / nei (training loss).** Huber loss of the codec's masked-channel
reconstruction (rec) and space/time-neighbour prediction (nei), in
z-scored anomaly units. Progress indicator only — never compare across
runs with different channels or normalisation.

**recon/<channel> skill.** On held-out data: `1 − MSE/MSE_channel-mean`
for masked-channel reconstruction, per channel. 1 = perfect, 0 = no
better than predicting the channel's mean. Static channels score ~0.9+
trivially (they never change); dynamic-channel recon in anomaly space is
honest and low (~0.0–0.6).

**chan% — held-out t+1 skill vs persistence, channel space.** THE primary
metric. A stage-2 model predicts next month's embedding; it is decoded
through the frozen codec into physical channel values and scored against
what happened, relative to the persistence forecast (next month = this
month, in anomaly space — the seasons are already subtracted, so
persistence is a strong baseline). `chan% = 100·(1 − MSE_model/MSE_pers)`
over dynamic channels on held-out months. Evaluated on 8–20k pixel-months;
observed seed-to-seed spread ≈ ±0.5 pt. **Reliable at the ~1-point
level** — differences ≥2 pts are real.

**z% — the same, in embedding space.** Same power as chan%, one caveat:
z-geometry is codec-specific, so z% comparisons are only meaningful
*within* one codec, never across codecs.

**r_lin — linear RAPID probe (deseasonalised).** Ridge regression from
the 26.5°N section's single-month embeddings to RAPID transport with its
train-years monthly climatology subtracted; λ chosen on a train tail;
Pearson r on held-out years only. 33 parameters on ~204 train months —
statistically sane to FIT, but see "probe power" below for the TEST.

**r_tmp — temporal-probe RAPID r.** As r_lin, but the features are the
mini temporal transformer's section-pooled hidden state (65-parameter
read-out on 64 dims). Same test-power limit.

**K-concat sweep r.** Linear probe on K stacked months of embeddings —
up to 769 parameters on ~190 train points (p > n at K≥8!). Kept as a
diagnostic of whether history carries information at all; its point
values are the least reliable numbers we produce.

**seasonal floor.** The same ridge from (sin, cos) month alone: what the
calendar predicts by itself (raw target +0.27; deseasonalised −0.17 ≈ 0
by construction). Any probe r must be read relative to this floor.

**displaced volume, tide (globe).** Unrelated to ML — the Tides layer's
km³-above-mean counter, exact arithmetic on the harmonic field.

## Probe power — the number that explains the wobble

RAPID monthly transport has lag-1 autocorrelation **0.41**, so n months
are worth `n·(1−ρ)/(1+ρ)` independent samples:

| test set | n_eff | SE(r) | 95% CI half-width |
|---|---|---|---|
| current: 36 held-out months | ~15 | **0.29** | **±0.57** |
| year-blocked k-fold over all 240 months | ~100 | 0.10 | ±0.20 |

That ±0.57 is why the same config scores 0.33 and 0.09 on two seeds, and
why "0.43 vs 0.31" was evidence, not proof. **Any single probe r in
[−0.25, +0.55] is compatible with a true r of ~0.15–0.30.** The observed
CONSISTENCY of direction across independent read-outs (linear + temporal
both rising with the 12-channel tensor) carries more information than any
point value.

## Reliability tiers

1. **chan%** (and z% within one codec) — decision-grade at ±1 pt.
2. **recon skill per channel** — decision-grade for representation checks.
3. **Direction agreement of r_lin and r_tmp across independent probes** —
   suggestive evidence.
4. **Any single r value** — compass needle, not a measurement.
5. **K-sweep point values** — diagnostics only.

## Protocol upgrades that buy real power (ranked)

1. **Year-blocked k-fold probing — BUILT (`probe_kfold.py`,
   2026-08-06):** all 240 RAPID months become test exactly once (folds
   blocked by calendar year; λ on an inner tail; block-bootstrap CI over
   whole years). First run, linear section probe, deseasonalised:

   | codec | k-fold r | 95% CI |
   |---|---|---|
   | dz8 | 0.111 | [0.01, 0.20] |
   | dz16 | 0.151 | [0.01, 0.28] |
   | d_z=32 (actions #1) | 0.182 | [0.05, 0.31] |
   | dz64 | **0.308** | [0.13, 0.46] |

   Three lessons the coarse instrument could not deliver: every CI
   excludes zero (the embeddings carry REAL deseasonalised AMOC signal —
   the first statistically defensible version of that claim); the true
   effect size is ~0.1–0.3, i.e. the 36-month draws of 0.4+ were
   flattering; and r rises monotonically with d_z even though chan%
   saturates at 32 — the transport read-out wants a wider bottleneck
   than field prediction does. Caveat (in the script header): the codec
   saw non-holdout months' FIELDS during self-supervised training, so
   absolute values are mildly optimistic; comparisons are fair. On
   "hold out twice as much" instead: 6 years would buy only √2
   (SE 0.29→0.20) and starve training — k-fold dominates it.
2. **More truth series**: Florida Current cable (daily, 1982→ — several
   times RAPID's span), OSNAP, MOVE, SAMBA as additional never-seen
   transports. This is the only lever that adds *independent* months.
3. **Multi-seed medians** with seed spread reported (now standard in the
   sweep tooling).
4. **Multi-horizon skill** (t+3, t+6 chan%) — more informative about
   dynamics than t+1 alone, same statistical power class.

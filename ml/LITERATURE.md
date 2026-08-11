# Literature — who has done this, what they measure, and how we compare

Three-track survey (2026-08-07): AMOC reconstruction from observables ·
ML ocean forecasting & skill-metric conventions · self-supervised /
foundation-model representation learning for ocean & climate. Full
citations at the end of each section. Companion docs: [METRICS.md](METRICS.md),
[FINDINGS.md](FINDINGS.md).

## 0 · The one-paragraph verdict

Our exact pattern — masked self-supervised embedding of the sparse
ocean-interior tensor with missingness-as-information, frozen encoder,
ridge probes to transport arrays at multiple latitudes, small temporal
transformer for autoregressive prediction — **does not appear in the
literature**. The pieces exist separately: anomaly-space masked
pretraining (Prithvi WxC, atmosphere), frozen-backbone-plus-cheap-heads
as a quality metric (Lehmann et al. 2026), linear probes on frozen Earth
FM embeddings (Aurora probing, classification only), supervised
NN regressions to MOC transport (Solodoch 2023, Wölker 2025, Michel
2025 — none from frozen SSL features, most never applied to real data).
Three of our habits turn out to be *unoccupied territory*: ridge probes
from frozen embeddings to integrated circulation indices, treating the
historical observation pattern as signal, and probe-based model
selection. And one of our headline numbers is, as far as the survey
found, **the first of its kind: a year-blocked cross-validated monthly
deseasonalized r against real RAPID data from fields that never include
RAPID (0.60)** — every larger published number is either low-pass
filtered, calibrated in-sample, or lives entirely inside a simulation.

## 1 · AMOC reconstruction — the numbers to compare against

| work | inputs | target & filtering | skill | protocol |
|---|---|---|---|---|
| Frajka-Williams 2015 (GRL) | SLA + cable + Ekman | RAPID MOC, 18-mo low-pass | >90% variance (r≈0.95) | in-sample calibration |
| Sanchez-Franks 2021 (OS) | SLA Δη + thermal wind | RAPID, deseas + 18-mo filter | r 0.83 (AMOC), 0.75 (UMO) | in-sample, hindcast 1993→ |
| Worthington 2021 (OS) | boundary densities (4 depths) | RAPID UMO monthly | R² 0.78, SE 1.23 Sv; held-out r 0.75 | trained 2006-17, tested 2017-18 |
| Caesar 2018 (Nature) | SPG SST fingerprint | AMOC index, 20-yr LOWESS | trend attribution only | model-calibrated |
| Solodoch 2023 (JAMES) | OBP+SSH+SST+τ strips | ECCO MOC, monthly deseas | r≈0.98, RMSE≈0.5 Sv | **simulation-only**, chrono 70/30 |
| Meng 2024 (JAMES) | SSH+OBP maps | eddying channel MOC | skill ≈0.44 (MOC) | simulation-only |
| Wölker 2025 (OS) | virtual Argo + τ + cables | VIKING20X 26.5°N, monthly | R²≈0.74 monthly, <1 Sv | simulation-only |
| Michel 2025 (ERL) | annual NA SST | AMOC 20-60°N, annual, 10-yr smooth | figures only | CMIP-trained, leave-model-out |
| **ours (global14/wind14)** | 14-ch embedding, no SSH/OBP | **real RAPID, monthly, deseas** | **k-fold r 0.60 [0.46, 0.73]** | year-blocked k-fold on obs |

Key readings: the eddying-model papers (Meng 0.44, Wölker 0.74) are the
honest difficulty scale — Solodoch's 0.98 lives in laminar 1° ECCO. Our
0.60 on *real* monthly data sits exactly where the eddying simulations
suggest it should, and no published work has our cross-validation
strictness on observations. Also: nobody has published a surface→OSNAP
reconstruction — our null there is a first attempt, not a failed
replication.

**What we must add to be comparable** (in priority order):
1. **RMSE in Sv** beside every probe r (field convention; RAPID σ ≈ 3 Sv
   deseasonalized — quote it so R² is recoverable).
2. **18-month low-passed r** from the same out-of-fold predictions — the
   headline the classical literature reports; our monthly 0.60 will look
   artificially weak next to FW2015's 0.95 unless the filtering
   difference is on the table.
3. **A wind-stress-only ridge baseline** (τ section means → RAPID, no
   embedding): reviewers will ask how much is just Ekman. Solodoch found
   wind alone carries much of the subtropical signal.
4. State the claim precisely: obs-trained, same-month nowcast,
   never-input target, year-blocked k-fold — versus simulation-trained
   (Solodoch/Wei/Meng/Wölker) and in-sample calibrated (FW/S-F).

## 2 · Forecasting — our metrics already have standard names

- Our "skill vs climatology" **is MSSS** (mean squared skill score,
  Goddard et al. 2013, the WMO decadal-verification framework) with the
  climatological (zero-anomaly) reference. Rename and cite.
- Our "skill vs persistence" is persistence-referenced MSSS. The field's
  preferred cheap baseline is **damped persistence** (AR1: anomaly ×
  lag-h autocorrelation) — raw persistence goes negative-MSSS at long
  leads, so beating it at h=12 is weak evidence. Add damped persistence.
- The lingua franca is **ACC per lead** (anomaly correlation), with
  ACC=0.5 the classic "useful skill" line (ENSO/seasonal convention),
  0.6 the stricter synoptic line. Translation bounds: for an unbiased
  forecast MSSS = 2·ACC−1; for optimally-damped MSSS = ACC². Our global
  rollout MSSS 0.38@h1 → ACC ≈ 0.62-0.69; 0.11@h12 → ACC ≈ 0.33-0.56 —
  we plausibly cross ACC 0.5 mid-horizon. **Compute ACC directly** and
  add the **Murphy/Goddard decomposition** (r² − conditional bias²) to
  say whether the decay is decorrelation or amplitude damping (the
  autoregressive-damping question WenHai made a headline issue).
- Detrended AND raw variants (Jacox et al. 2022 detrend 1991-2020: the
  warming trend can masquerade as skill, worst for subsurface).
- Reference points at our cadence: Ham 2019 (Niño3.4 r>0.5 to ~17 mo —
  the easiest index in the ocean), SEAS5 (0.93@2mo → 0.78@11mo for
  January ENSO starts), Taylor & Feng 2022 (global monthly SSTA,
  Unet-LSTM, RMSE 0.48°C@1mo → 0.63@18mo — the closest published
  analogue to our field forecasting), CAS-Canglong, NMME. The daily
  ocean forecasters (XiHe, WenHai, GLONET, NeuralOM: 1-60 days,
  1/4-1/12°) are a different regime — cite, don't compare. Community
  benchmarks: Mercator's OceanBench (Class-4 verification against real
  observations is the direction of travel), OceanForecastBench,
  WeatherBench2 conventions.
- Sobering precedent worth internalizing: the S2S AI Challenge winner
  beat calibrated climatology by RPSS +0.046 — at hard leads, small
  positive skill is the norm and is publishable.

### 2b · Forecasting the TRANSPORT itself — the actual competition (2026-08-11)

Chris reframed the objective as "the world's best predictor model, for
AMOC specifically", so the bar is: who has published forward-looking
skill on the RAPID transport series, at what leads, against what
baseline. The field is thin, and its one famous entry is a cautionary
tale:

- **Matei et al. 2012 (Science)** — MPI-ESM initialized decadal system,
  claimed multiyear predictability of the MONTHLY-MEAN 26.5°N
  transport. The published **Comment** dismantled it: most of the
  claimed skill is the SEASONAL CYCLE plus the wind-driven Ekman term,
  and deseasonalised skill beyond persistence was not demonstrated.
  Fourteen years later this remains the reflexive referee objection to
  any AMOC-prediction claim, and our protocol was already built to
  survive it: deseasonalised target, a wind-only ridge bar, damped
  persistence as the operative baseline.
- **Foukal (state-space analogue, ~2015, unpublished beyond an OSNAP
  research post)** — 18-month lead from 10-dimensional delay embedding
  + 14 nearest neighbours: RMSE 2.46 Sv vs 2.98 for seasonal
  climatology alone, 48.5% variance 2004–2014, without modern
  out-of-sample discipline.
- **Initialized decadal systems since** (DCPP-class) publish ANNUAL-mean
  AMOC skill at years 1–5+ — a different cadence; cite, don't compare.
- The deep-learning AMOC literature is nowcast/reconstruction (§1:
  Michel 2025 from SST; Solodoch 2023 et al. in simulation) or
  tipping-point anticipation in idealized models — none of it forecasts
  the observed monthly transport.

**The niche is open**: monthly-resolution, deseasonalised, strictly
out-of-sample (year-blocked holdout) forecast skill on real RAPID
transport at leads 1–12 months, against damped persistence and a
wind-only bar, has as far as this survey can find NEVER been published.
Our E-011 rollout numbers (probe on rolled states, bands r 0.31/0.11/
0.21; field ACC 0.40 at h=12) are weak in absolute terms but are, to
current knowledge, the only numbers of their kind — and E-013/E-014
(rolled-fit probes, direct heads) attack exactly their weakest link,
the readout at horizon. A "world's best" claim should be phrased as:
best *documented* out-of-sample monthly AMOC forecast, with the
benchmark table above published alongside so the claim is checkable.

## 3 · Foundation models — where we sit

Flagship Earth FMs (ClimaX, Aurora, ORBIT, the 1/12° ocean forecasters)
evaluate by full fine-tuning; frozen-representation evaluation is the
exception: Prithvi WxC (anomaly-space masked pretraining + frozen
backbone + small heads — our closest architectural relative,
atmosphere-only), WV-Net (the classic kNN/linear/MLP/fine-tune probe
ladder, SAR), Presto (per-pixel time-series tokens with structured
missingness masking, land EO), Lehmann et al. 2026 (frozen Aurora +
shallow decoders, arguing frozen-extension should be a standard FM
quality metric — the published statement of our evaluation philosophy).
Gaps the survey could not fill: no SSL foundation model exists for the
gridded ocean interior; no one probes frozen ocean embeddings for
circulation indices; missingness-as-information is essentially untouched
(DINCAE's error-weighted inputs come closest); label-efficiency curves
(skill vs number of truth months) are the standard argument for
embeddings we haven't run yet — and our truth series make it natural.

## 4 · Metric additions queued (comparability program)

1. `probe_kfold.py`: report RMSE (Sv) + target σ; add 18-month
   low-passed r from the same out-of-fold predictions; add the
   wind-stress-only ridge baseline row.
2. `rollout.py`: ACC per horizon (direct, not inferred); damped
   persistence (per-pixel AR1) as third baseline; Murphy decomposition;
   rename skill-vs-clim → MSSS in all outputs and docs.
3. Report both detrended and raw probe/rollout variants.
4. Label-efficiency curve: probe r as a function of truth months used —
   the standard "embeddings beat scratch" argument, natural with FC's
   359 months.

## 5 · Source reports

The three full survey reports (per-paper details, verified numbers,
flagged uncertainties, all URLs) are archived in this repo's history and
summarized above. Principal sources: Frajka-Williams 2015 (GRL 42),
Sanchez-Franks et al. 2021 (OS 17), Worthington et al. 2021 (OS 17),
Caesar et al. 2018 (Nature 556), Solodoch et al. 2023 (JAMES 15), Meng
et al. 2024 (JAMES 16), Wölker et al. 2025 (OS 21), Michel et al. 2025
(ERL 20), DelSole & Nedza 2022, Zhai et al. 2024, Ham et al. 2019
(Nature 573), Taylor & Feng 2022 (Front. Climate), Johnson et al. 2019
(SEAS5, GMD 12), Jacox et al. 2022 (Nature), Goddard et al. 2013 (Clim.
Dyn. 40), Rasp et al. 2024 (WeatherBench2), Vitart et al. 2022 (S2S AI
Challenge, BAMS), XiHe (arXiv:2402.02995), WenHai (Nat. Commun. 2025),
GLONET (JGR-MLC 2025), Samudra (GRL 2025), NeuralOM, AI-GOMS, Aurora
(Nature 2025), Prithvi WxC (arXiv:2409.13598), AtmoRep, ClimaX (ICML
2023), Presto, DINCAE 2.0 (GMD 15), Lehmann et al. (JGR-MLC 2026),
Richards & Balan (Aurora probing), MacMillan & Ouellette (GraphCast
SAEs), WV-Net (AIES 2025), OceanBench (NeurIPS 2025), OceanForecastBench.

# E-076b · the optimal-interpolation ceiling for the Argo ocean interior — results

*Generated 2026-09-08T06:22:41Z from `ml/oi_ceiling.py`, seed 0, 644 s on two CPU cores. Every number below is read out of the `oi_ceiling.json` that run wrote; nothing is transcribed by hand.*

**What this experiment is, in one paragraph.** E-076a trained a small neural encoder — a *cone codec*, the network that reads a thirty-day cone of past measurements around a point on the globe — and asked it to guess an Argo float's temperature and salinity profile from the float profiles nearby. It beat the seasonal average by 2-3 % on temperature above 300 dbar and by nothing at all on salinity or in the deep ocean. Two readings of that were possible and the experiment could not tell them apart: either the interior is simply not predictable from a handful of scattered float profiles, or the model under-used what it was given. **E-076b decides between them with no model at all**, by measuring what the classical method gets. That method is *optimal interpolation* (OI) — the standard way a scattered set of ocean measurements is turned into a map, and the method the gridded Argo product this project uses as its climatology is itself built with. It estimates the departure from the seasonal average at the target as a weighted sum of the neighbours' departures, with weights chosen so that the expected squared error is as small as possible under a covariance that falls off with distance and with age. It costs nothing but CPU.

**The answer is that the interior IS predictable, and by a wide margin.** Optimal interpolation with twenty neighbours reaches a root-mean-square error of **0.686** of the climatology's on temperature and **0.661** on salinity, on the terminal held-out years — a 31 % and 34 % reduction in error, against the 2.5 % and 0.1 % the trained codec managed on the same kind of target. The information is there; E-076a's model did not use it.

## How to read this

- **Anomaly.** Every estimator predicts a *departure from the seasonal average*, not a temperature. The seasonal average is the climatology defined below, evaluated at the profile's own 1-degree cell and its own calendar month.
- **RMSE** is the root-mean-square error of that prediction, converted back to degrees Celsius or practical salinity units, at one pressure level.
- **Skill** is the mean over the sixteen levels of RMSE divided by the climatology's RMSE at the same level. **1.0 is no better than the seasonal average; lower is better.** This is the same summary E-076a reports, so the two are directly comparable.
- **The five estimators.** *climatology* predicts a zero departure — the bar. *nearest* copies the nearest neighbour's departure (E-076a calls this persistence). *inverse-distance* takes a weighted mean of the neighbours' departures with weights one over the squared space-time distance. *OI* is optimal interpolation. *OI+shrink* multiplies the OI estimate by a scalar between zero and one, fitted on training years, which guards against over-confident weights.
- **Everything is fitted on training years only.** The three OI hyper-parameters (correlation length L, decorrelation time T, noise-to-signal ratio r) and the shrinkage are chosen by grid search on 6,000 targets drawn from training years, separately for three depth bands and for temperature versus salinity, and are then applied unchanged to the held-out splits. No held-out number was used to choose anything.

## 1 · The sample, and the climatology every number is measured against

- **Targets** are real Argo profiles drawn uniformly at random from the family-8 observation store (2,678,439 profiles, 2004-2024), with a fixed seed: 6,000 from training years, 6,000 from 2021-2024, and 6,000 from 2008, 2009, 2016, 2017. Training years are **<= 2020 except 2008, 2009, 2016, 2017** — the frozen protocol's `holdout_years`.
- A profile is eligible only if the climatology is defined at its own cell and month for at least **8** temperature levels, and if at least **2** other profiles lie inside the search bounds once the target itself is taken out. Neither condition bites hard: on the terminal split 6,282 shuffled profiles were walked to keep 6,000, 282 of them (4.5%) rejected for having no climatology — those are the high latitudes, outside the Roemmich-Gilson band of -64.5 to 79.5 degrees — and every one of the 6,000 that survived it had at least two other profiles in range.
- **The same targets are used for every k**, so the four searches are compared on one population. Eligibility was decided against the k = 20 / 30 d / 1000 km search; widening the search can only add neighbours, so no target becomes ineligible under it.
- **Neighbours** come from `ml/family8_store.py::knearest`, which is one-sided in time by construction — it reads no observation later than the target's own five-day bin. The search is asked for k + 1 and the target's own row is dropped, leaving the k nearest OTHER profiles.

**The climatology.** The train-years monthly mean of the `rg100` group of the family-7 global tensor — the Roemmich-Gilson gridded Argo column, one 1-degree map per month, 2004-01 to 2024-12 — un-z-scored with the tensor's own `norm_rg100` and averaged per calendar month, per cell, per channel over **156 training rows (13 per month)** out of 252. That is the same quantity `ml/trainprobe.py::anomaly_transform` computes inside the training pipeline, so it is the same bar E-076a used. The file it was built from (`family7_global025_pentad_l0_X_rg100.npy`, 1,050,900,608 bytes) was checked against the Hub dataset's own `manifest.json` by size and sha256 before it was read.

### 1.1 · Does the climatology bar reproduce E-076a's?

Broadly yes, and where it does not, the reason is the target distribution rather than the arithmetic. E-076a drew *grid cells* uniformly over the ocean and scored the nearest profile to each; E-076b draws *profiles* uniformly out of the store, as its specification asks. Argo is denser in the western boundary currents than a uniform draw over the ocean is, and those are exactly the places where the main thermocline is most variable — so the bar here sits above E-076a's between 100 and 500 dbar and agrees with it at the surface and in the deep.

| dbar | E-076b climatology RMSE (°C) | E-076a climatology RMSE (°C) | n here |
|---|---|---|---|
| 10 | 1.091 | 1.086 | 5,961 |
| 30 | 1.265 | 1.232 | 5,968 |
| 50 | 1.416 | 1.328 | 5,965 |
| 100 | 1.437 | 1.306 | 5,933 |
| 200 | 1.192 | 1.061 | 5,867 |
| 300 | 1.022 | 0.831 | 5,825 |
| 500 | 0.796 | 0.637 | 5,714 |
| 900 | 0.384 | 0.330 | 5,621 |
| 1500 | 0.128 | 0.151 | 4,617 |

At 10 dbar the two agree to 0.5 % (1.091 against 1.086), at 900 dbar to 16 % and at 1,500 dbar to 15 %, with the largest disagreement — 23 % — at 300 dbar. E-076a's registered anchors are the spec's stated reference values (1.09 at 10 dbar, 0.83 at 300, 0.33 at 900); the surface and deep ones are reproduced, the thermocline one is not, and the sampling difference above is why. **Every comparison in this document is internal** — each estimator against the climatology measured on its own targets — so that difference does not move any skill number.

### 1.2 · What the observing system looks like at these targets

| search | mean neighbours in range | median distance to the nearest | fraction with one inside 20 km | fraction inside 50 km |
|---|---|---|---|---|
| k = 5 · 30 d · 1,000 km | 141 | 130 km | 10.8% | 19.8% |
| k = 10 · 30 d · 1,000 km | 141 | 130 km | 10.8% | 19.8% |
| k = 20 · 30 d · 1,000 km | 141 | 130 km | 10.8% | 19.8% |
| k = 20 · 60 d · 1,500 km (the widened search) | 553 | 108 km | 13.5% | 25.9% |

The median target has another float profile 130 km away inside the thirty-day window, and about 140 profiles somewhere inside the 1,000 km disc — the array is far denser than the k = 5 the codec was given. About one target in nine has a profile within 20 km, which is usually the same float on its previous or next cycle; §5 measures what the headline looks like with those taken out.

## 2 · Skill, every estimator × search × split

Mean over the sixteen levels of RMSE / climatology-RMSE. **Lower is better; 1.000 is the seasonal average.**

### Temperature

| search | split | nearest | inverse-distance | **OI** | OI+shrink |
|---|---|---|---|---|---|
| k = 5 · 30 d · 1,000 km | terminal | 1.0832 | 0.8773 | **0.7856** | 0.7858 |
| k = 5 · 30 d · 1,000 km | interspersed | 1.1619 | 0.9286 | **0.8252** | 0.8240 |
| k = 10 · 30 d · 1,000 km | terminal | 1.0827 | 0.8398 | **0.7113** | 0.7118 |
| k = 10 · 30 d · 1,000 km | interspersed | 1.1621 | 0.8897 | **0.7689** | 0.7685 |
| k = 20 · 30 d · 1,000 km | terminal | 1.0826 | 0.8301 | **0.6857** | 0.6861 |
| k = 20 · 30 d · 1,000 km | interspersed | 1.1622 | 0.8729 | **0.7391** | 0.7390 |
| k = 20 · 60 d · 1,500 km (the widened search) | terminal | 1.0440 | 0.8092 | **0.6807** | 0.6815 |
| k = 20 · 60 d · 1,500 km (the widened search) | interspersed | 1.1051 | 0.8521 | **0.7289** | 0.7290 |

### Salinity

| search | split | nearest | inverse-distance | **OI** | OI+shrink |
|---|---|---|---|---|---|
| k = 5 · 30 d · 1,000 km | terminal | 1.0473 | 1.0823 | **0.8552** | 0.8551 |
| k = 5 · 30 d · 1,000 km | interspersed | 1.2348 | 0.9715 | **0.8426** | 0.8415 |
| k = 10 · 30 d · 1,000 km | terminal | 1.0468 | 0.9691 | **0.7856** | 0.7867 |
| k = 10 · 30 d · 1,000 km | interspersed | 1.2355 | 0.9180 | **0.7818** | 0.7806 |
| k = 20 · 30 d · 1,000 km | terminal | 1.0468 | 0.9336 | **0.6612** | 0.6660 |
| k = 20 · 30 d · 1,000 km | interspersed | 1.2357 | 0.8972 | **0.7525** | 0.7520 |
| k = 20 · 60 d · 1,500 km (the widened search) | terminal | 1.0176 | 0.9160 | **0.6561** | 0.6621 |
| k = 20 · 60 d · 1,500 km (the widened search) | interspersed | 1.1900 | 0.8716 | **0.7410** | 0.7405 |

Four things read straight off that table.

1. **Optimal interpolation beats the seasonal average everywhere**, on both variables, on both held-out splits, at every k. The margin grows with k: 21 %, 29 %, 31 % on terminal-split temperature at k = 5, 10, 20.
2. **Copying the nearest profile is worse than the seasonal average**, at every k and on both splits (skill 1.02-1.24). This reproduces E-076a's persistence finding exactly — a float 130 km away is a worse guess than the monthly climatology at the target's own cell — and it is the clearest statement of why the estimator, not the data, was the missing piece.
3. **Inverse-distance recovers most of the gain but not all of it** (0.81-0.97 against OI's 0.66-0.86), and on k = 5 salinity it is actually WORSE than the seasonal average (1.082). The gap has two causes and both are instructive. OI accounts for the neighbours being correlated *with each other*, so it does not count three profiles from the same eddy three times, and it has a noise term, so it shrinks towards the climatology exactly as far as the data warrant. Inverse-distance has neither, which also makes it fragile: the largest single neighbour anomaly in the k = 5 terminal set is 19.9 PSU against a 99.9th percentile of 1.46, and a weighted mean with no noise model carries an outlier like that straight into its answer.
4. **Widening the search buys almost nothing.** Going from 30 days and 1,000 km to 60 days and 1,500 km at the same k = 20 moves terminal temperature skill by 0.005 and salinity by 0.005. The ceiling is set by how many neighbours the estimator is allowed to use, not by how far it is allowed to look — which is a statement about the array's density, not about the ocean.

### 2.1 · By depth band

The bands are fitted independently. `10-100` is the seasonal thermocline, `150-500` the permanent thermocline, `700-1900` the deep water below it.

| search | split | variable | 10-100 dbar | 150-500 dbar | 700-1900 dbar |
|---|---|---|---|---|---|
| k = 5 · 30 d · 1,000 km | terminal | temperature | 0.7447 | 0.7482 | 0.8356 |
| k = 5 · 30 d · 1,000 km | terminal | salinity | 0.7635 | 0.8143 | 0.9369 |
| k = 5 · 30 d · 1,000 km | interspersed | temperature | 0.7854 | 0.7985 | 0.8669 |
| k = 5 · 30 d · 1,000 km | interspersed | salinity | 0.7892 | 0.8302 | 0.8819 |
| k = 10 · 30 d · 1,000 km | terminal | temperature | 0.7008 | 0.6843 | 0.7365 |
| k = 10 · 30 d · 1,000 km | terminal | salinity | 0.6958 | 0.7356 | 0.8726 |
| k = 10 · 30 d · 1,000 km | interspersed | temperature | 0.7592 | 0.7472 | 0.7899 |
| k = 10 · 30 d · 1,000 km | interspersed | salinity | 0.7490 | 0.7671 | 0.8111 |
| k = 20 · 30 d · 1,000 km | terminal | temperature | 0.6823 | 0.6577 | 0.7078 |
| k = 20 · 30 d · 1,000 km | terminal | salinity | 0.6606 | 0.6842 | 0.6451 |
| k = 20 · 30 d · 1,000 km | interspersed | temperature | 0.7407 | 0.7190 | 0.7525 |
| k = 20 · 30 d · 1,000 km | interspersed | salinity | 0.7262 | 0.7381 | 0.7779 |
| k = 20 · 60 d · 1,500 km (the widened search) | terminal | temperature | 0.6812 | 0.6521 | 0.7009 |
| k = 20 · 60 d · 1,500 km (the widened search) | terminal | salinity | 0.6537 | 0.6787 | 0.6413 |
| k = 20 · 60 d · 1,500 km (the widened search) | interspersed | temperature | 0.7393 | 0.7069 | 0.7388 |
| k = 20 · 60 d · 1,500 km (the widened search) | interspersed | salinity | 0.7188 | 0.7325 | 0.7597 |

**This is the finding that most directly contradicts E-076a's reading.** E-076a found its whole 2-3 % gain above 300 dbar and nothing below it, and concluded that the surface channels were doing the work. Optimal interpolation's gain is spread evenly over the whole column: on temperature it is if anything LARGER in the permanent thermocline than at the surface (0.658 against 0.682 at k = 20 on the terminal split), and it is **22-35 % in the deep water at k = 20 across both splits and both variables**, where the codec had exactly zero. The deep interior is the part a surface-driven model cannot reach and the part neighbouring profiles predict best — the deep ocean varies slowly and smoothly, so a profile 130 km away is a good witness for it. Note also that at k = 5 the deep band is the WEAKEST (0.836-0.937), and it is the band that gains most from more neighbours: with only five profiles the deep signal is buried in the mesoscale noise the extra neighbours average away.

## 3 · The fitted hyper-parameters

Grid: L ∈ {50, 100, 150, 200, 300, 500} km · T ∈ {10, 20, 30, 60} days · r ∈ {0.1, 0.3, 1.0, 3.0, 10.0} · shrinkage α over 0..1 in 101 steps. Chosen by minimising the mean over the band's levels of the RMSE **on training-year targets**.

| search | variable | band | L (km) | T (days) | r | α |
|---|---|---|---|---|---|---|
| k = 5 · 30 d · 1,000 km | temperature | 10-100 | 150 | 20 | 0.3 | 0.91 |
| k = 5 · 30 d · 1,000 km | temperature | 150-500 | 100 | 60 | 0.3 | 0.97 |
| k = 5 · 30 d · 1,000 km | temperature | 700-1900 | 100 | 60 | 0.3 | 0.90 |
| k = 5 · 30 d · 1,000 km | salinity | 10-100 | 150 | 20 | 0.3 | 0.91 |
| k = 5 · 30 d · 1,000 km | salinity | 150-500 | 100 | 30 | 0.3 | 0.95 |
| k = 5 · 30 d · 1,000 km | salinity | 700-1900 | 50 | 30 | 0.1 | 1.00 |
| k = 10 · 30 d · 1,000 km | temperature | 10-100 | 150 | 20 | 0.3 | 0.92 |
| k = 10 · 30 d · 1,000 km | temperature | 150-500 | 100 | 30 | 0.3 | 0.99 |
| k = 10 · 30 d · 1,000 km | temperature | 700-1900 | 100 | 60 | 0.3 | 0.97 |
| k = 10 · 30 d · 1,000 km | salinity | 10-100 | 150 | 20 | 0.3 | 0.92 |
| k = 10 · 30 d · 1,000 km | salinity | 150-500 | 100 | 30 | 0.3 | 0.96 |
| k = 10 · 30 d · 1,000 km | salinity | 700-1900 | 100 | 60 | 0.3 | 0.93 |
| k = 20 · 30 d · 1,000 km | temperature | 10-100 | 150 | 20 | 0.3 | 0.93 |
| k = 20 · 30 d · 1,000 km | temperature | 150-500 | 100 | 30 | 0.3 | 0.99 |
| k = 20 · 30 d · 1,000 km | temperature | 700-1900 | 100 | 60 | 0.3 | 0.99 |
| k = 20 · 30 d · 1,000 km | salinity | 10-100 | 100 | 20 | 0.3 | 0.97 |
| k = 20 · 30 d · 1,000 km | salinity | 150-500 | 100 | 30 | 0.3 | 0.97 |
| k = 20 · 30 d · 1,000 km | salinity | 700-1900 | 100 | 60 | 0.3 | 0.95 |
| k = 20 · 60 d · 1,500 km (the widened search) | temperature | 10-100 | 150 | 20 | 0.3 | 0.92 |
| k = 20 · 60 d · 1,500 km (the widened search) | temperature | 150-500 | 100 | 30 | 0.3 | 0.98 |
| k = 20 · 60 d · 1,500 km (the widened search) | temperature | 700-1900 | 100 | 60 | 0.3 | 0.97 |
| k = 20 · 60 d · 1,500 km (the widened search) | salinity | 10-100 | 100 | 30 | 0.3 | 0.94 |
| k = 20 · 60 d · 1,500 km (the widened search) | salinity | 150-500 | 100 | 30 | 0.3 | 0.96 |
| k = 20 · 60 d · 1,500 km (the widened search) | salinity | 700-1900 | 100 | 60 | 0.3 | 0.94 |

The fit is remarkably stable across all four searches and both variables. **The correlation length lands on 100 km in seventeen of the twenty-four fits and on 150 km in six** (once, for k = 5 salinity in the deep, on 50 km — the one fit that reaches an edge of the L grid). That is the ocean's mesoscale, and it is the length the Roemmich-Gilson mapping this project uses as its climatology is itself built with; it sits in the interior of the grid, so the grid did not decide it. **The noise-to-signal ratio is 0.3 in twenty-three of the twenty-four fits** — about a quarter of the variance at a point is treated as unrepresentative of its neighbourhood, which is what mesoscale eddies and instrument error together should look like. **The decorrelation time rises with depth**: the seasonal thermocline takes 20 or 30 days, the permanent thermocline and the deep take 30 or 60. Sixty days is the top of the T grid and eight fits choose it, so the true decorrelation time below the seasonal layer may well be longer than this grid can express — the one place the search is bounded by its own grid rather than by the data. **The shrinkage is 0.90-1.00 and buys nothing**: the most it ever improves held-out skill by is 0.0012, and on k = 20 salinity it makes it 0.005-0.006 WORSE — a shrinkage fitted on training years transferring imperfectly, which is the honest small cost of fitting one more constant. The OI weights are already close to unbiased and the guard the shrinkage provides is not needed.

## 4 · Per-level RMSE

`clim` is the climatology bar in the level's own unit, `n` the number of targets scored at that level (a target counts when it measured the level AND at least one neighbour did — one mask for all five estimators, so they are compared on identical targets).

### k = 5 · 30 d · 1,000 km · terminal · temperature (°C)

| dbar | climatology | nearest | inverse-distance | OI | OI+shrink | n |
|---|---|---|---|---|---|---|
| 10 | 1.091 | 0.912 | 0.741 | 0.758 | 0.759 | 5,961 |
| 30 | 1.265 | 1.222 | 0.997 | 0.961 | 0.957 | 5,968 |
| 50 | 1.416 | 1.379 | 1.119 | 1.072 | 1.070 | 5,965 |
| 100 | 1.437 | 1.439 | 1.169 | 1.103 | 1.104 | 5,933 |
| 150 | 1.356 | 1.397 | 1.115 | 1.045 | 1.046 | 5,889 |
| 200 | 1.192 | 1.249 | 0.996 | 0.900 | 0.900 | 5,867 |
| 300 | 1.022 | 1.080 | 0.875 | 0.762 | 0.763 | 5,825 |
| 400 | 0.937 | 0.983 | 0.795 | 0.683 | 0.684 | 5,785 |
| 500 | 0.796 | 0.844 | 0.683 | 0.589 | 0.590 | 5,714 |
| 700 | 0.548 | 0.613 | 0.501 | 0.436 | 0.437 | 5,685 |
| 900 | 0.384 | 0.433 | 0.363 | 0.318 | 0.319 | 5,621 |
| 1100 | 0.243 | 0.297 | 0.242 | 0.206 | 0.205 | 4,992 |
| 1300 | 0.172 | 0.206 | 0.165 | 0.145 | 0.145 | 4,725 |
| 1500 | 0.128 | 0.156 | 0.124 | 0.107 | 0.107 | 4,617 |
| 1700 | 0.103 | 0.125 | 0.101 | 0.088 | 0.088 | 4,499 |
| 1900 | 0.089 | 0.107 | 0.087 | 0.075 | 0.075 | 4,346 |

### k = 5 · 30 d · 1,000 km · terminal · salinity (PSU)

| dbar | climatology | nearest | inverse-distance | OI | OI+shrink | n |
|---|---|---|---|---|---|---|
| 10 | 0.336 | 0.288 | 0.252 | 0.241 | 0.242 | 5,051 |
| 30 | 0.297 | 0.264 | 0.232 | 0.221 | 0.222 | 5,063 |
| 50 | 0.266 | 0.247 | 0.215 | 0.208 | 0.207 | 5,058 |
| 100 | 0.205 | 0.206 | 0.178 | 0.166 | 0.165 | 5,027 |
| 150 | 0.181 | 0.191 | 0.165 | 0.150 | 0.150 | 4,980 |
| 200 | 0.166 | 0.170 | 0.151 | 0.134 | 0.134 | 4,962 |
| 300 | 0.137 | 0.145 | 0.131 | 0.110 | 0.110 | 4,922 |
| 400 | 0.118 | 0.126 | 0.115 | 0.093 | 0.093 | 4,895 |
| 500 | 0.097 | 0.107 | 0.101 | 0.081 | 0.081 | 4,831 |
| 700 | 0.067 | 0.076 | 0.081 | 0.057 | 0.057 | 4,800 |
| 900 | 0.050 | 0.055 | 0.071 | 0.045 | 0.045 | 4,744 |
| 1100 | 0.045 | 0.050 | 0.056 | 0.042 | 0.042 | 4,203 |
| 1300 | 0.041 | 0.046 | 0.053 | 0.039 | 0.039 | 4,001 |
| 1500 | 0.038 | 0.042 | 0.041 | 0.037 | 0.037 | 3,922 |
| 1700 | 0.036 | 0.039 | 0.051 | 0.035 | 0.035 | 3,809 |
| 1900 | 0.034 | 0.037 | 0.055 | 0.033 | 0.033 | 3,675 |

### k = 5 · 30 d · 1,000 km · interspersed · temperature (°C)

| dbar | climatology | nearest | inverse-distance | OI | OI+shrink | n |
|---|---|---|---|---|---|---|
| 10 | 1.025 | 0.962 | 0.785 | 0.783 | 0.782 | 5,861 |
| 30 | 1.198 | 1.239 | 0.996 | 0.972 | 0.968 | 5,897 |
| 50 | 1.380 | 1.417 | 1.143 | 1.100 | 1.097 | 5,945 |
| 100 | 1.463 | 1.509 | 1.186 | 1.126 | 1.126 | 5,928 |
| 150 | 1.317 | 1.427 | 1.114 | 1.063 | 1.062 | 5,887 |
| 200 | 1.117 | 1.260 | 0.980 | 0.905 | 0.904 | 5,767 |
| 300 | 0.912 | 1.023 | 0.812 | 0.717 | 0.718 | 5,819 |
| 400 | 0.831 | 0.919 | 0.737 | 0.659 | 0.659 | 5,369 |
| 500 | 0.732 | 0.840 | 0.685 | 0.583 | 0.583 | 4,933 |
| 700 | 0.545 | 0.668 | 0.530 | 0.434 | 0.436 | 5,499 |
| 900 | 0.363 | 0.468 | 0.368 | 0.296 | 0.297 | 5,381 |
| 1100 | 0.238 | 0.304 | 0.245 | 0.199 | 0.199 | 4,492 |
| 1300 | 0.178 | 0.226 | 0.184 | 0.155 | 0.155 | 4,036 |
| 1500 | 0.135 | 0.177 | 0.140 | 0.121 | 0.121 | 3,704 |
| 1700 | 0.111 | 0.146 | 0.117 | 0.105 | 0.104 | 3,515 |
| 1900 | 0.089 | 0.113 | 0.092 | 0.080 | 0.079 | 3,341 |

### k = 5 · 30 d · 1,000 km · interspersed · salinity (PSU)

| dbar | climatology | nearest | inverse-distance | OI | OI+shrink | n |
|---|---|---|---|---|---|---|
| 10 | 0.266 | 0.375 | 0.259 | 0.212 | 0.210 | 5,620 |
| 30 | 0.243 | 0.234 | 0.189 | 0.181 | 0.181 | 5,659 |
| 50 | 0.230 | 0.320 | 0.213 | 0.183 | 0.182 | 5,668 |
| 100 | 0.190 | 0.205 | 0.164 | 0.155 | 0.155 | 5,653 |
| 150 | 0.159 | 0.179 | 0.142 | 0.131 | 0.131 | 5,627 |
| 200 | 0.137 | 0.158 | 0.124 | 0.112 | 0.112 | 5,531 |
| 300 | 0.109 | 0.127 | 0.102 | 0.090 | 0.090 | 5,584 |
| 400 | 0.094 | 0.111 | 0.090 | 0.080 | 0.080 | 5,163 |
| 500 | 0.078 | 0.097 | 0.078 | 0.065 | 0.065 | 4,743 |
| 700 | 0.055 | 0.069 | 0.055 | 0.046 | 0.046 | 5,285 |
| 900 | 0.037 | 0.047 | 0.037 | 0.031 | 0.031 | 5,174 |
| 1100 | 0.031 | 0.040 | 0.032 | 0.027 | 0.027 | 4,329 |
| 1300 | 0.030 | 0.039 | 0.032 | 0.027 | 0.027 | 3,901 |
| 1500 | 0.025 | 0.033 | 0.026 | 0.022 | 0.022 | 3,574 |
| 1700 | 0.020 | 0.026 | 0.021 | 0.018 | 0.018 | 3,386 |
| 1900 | 0.016 | 0.021 | 0.017 | 0.015 | 0.015 | 3,236 |

### k = 10 · 30 d · 1,000 km · terminal · temperature (°C)

| dbar | climatology | nearest | inverse-distance | OI | OI+shrink | n |
|---|---|---|---|---|---|---|
| 10 | 1.091 | 0.912 | 0.725 | 0.725 | 0.725 | 5,961 |
| 30 | 1.265 | 1.222 | 0.968 | 0.905 | 0.903 | 5,968 |
| 50 | 1.416 | 1.379 | 1.089 | 0.998 | 0.999 | 5,965 |
| 100 | 1.437 | 1.438 | 1.137 | 1.032 | 1.035 | 5,934 |
| 150 | 1.355 | 1.396 | 1.088 | 0.976 | 0.976 | 5,891 |
| 200 | 1.192 | 1.248 | 0.969 | 0.829 | 0.829 | 5,869 |
| 300 | 1.022 | 1.080 | 0.848 | 0.698 | 0.698 | 5,831 |
| 400 | 0.936 | 0.983 | 0.769 | 0.620 | 0.621 | 5,790 |
| 500 | 0.795 | 0.844 | 0.657 | 0.526 | 0.526 | 5,720 |
| 700 | 0.548 | 0.613 | 0.478 | 0.381 | 0.382 | 5,695 |
| 900 | 0.384 | 0.433 | 0.345 | 0.274 | 0.275 | 5,631 |
| 1100 | 0.243 | 0.297 | 0.226 | 0.180 | 0.180 | 5,004 |
| 1300 | 0.172 | 0.206 | 0.155 | 0.128 | 0.128 | 4,734 |
| 1500 | 0.128 | 0.156 | 0.117 | 0.096 | 0.096 | 4,629 |
| 1700 | 0.103 | 0.125 | 0.095 | 0.078 | 0.078 | 4,512 |
| 1900 | 0.089 | 0.107 | 0.082 | 0.067 | 0.067 | 4,362 |

### k = 10 · 30 d · 1,000 km · terminal · salinity (PSU)

| dbar | climatology | nearest | inverse-distance | OI | OI+shrink | n |
|---|---|---|---|---|---|---|
| 10 | 0.336 | 0.288 | 0.245 | 0.220 | 0.221 | 5,052 |
| 30 | 0.296 | 0.264 | 0.223 | 0.199 | 0.200 | 5,064 |
| 50 | 0.266 | 0.247 | 0.206 | 0.188 | 0.189 | 5,059 |
| 100 | 0.205 | 0.206 | 0.169 | 0.154 | 0.153 | 5,029 |
| 150 | 0.181 | 0.191 | 0.157 | 0.139 | 0.139 | 4,983 |
| 200 | 0.166 | 0.170 | 0.143 | 0.123 | 0.124 | 4,965 |
| 300 | 0.137 | 0.145 | 0.122 | 0.099 | 0.100 | 4,929 |
| 400 | 0.118 | 0.126 | 0.106 | 0.083 | 0.083 | 4,901 |
| 500 | 0.097 | 0.107 | 0.091 | 0.071 | 0.071 | 4,838 |
| 700 | 0.067 | 0.076 | 0.071 | 0.051 | 0.051 | 4,814 |
| 900 | 0.050 | 0.055 | 0.060 | 0.041 | 0.041 | 4,758 |
| 1100 | 0.045 | 0.050 | 0.048 | 0.039 | 0.039 | 4,225 |
| 1300 | 0.041 | 0.046 | 0.046 | 0.037 | 0.037 | 4,016 |
| 1500 | 0.038 | 0.042 | 0.038 | 0.034 | 0.034 | 3,940 |
| 1700 | 0.036 | 0.039 | 0.043 | 0.033 | 0.033 | 3,833 |
| 1900 | 0.034 | 0.037 | 0.045 | 0.032 | 0.032 | 3,703 |

### k = 10 · 30 d · 1,000 km · interspersed · temperature (°C)

| dbar | climatology | nearest | inverse-distance | OI | OI+shrink | n |
|---|---|---|---|---|---|---|
| 10 | 1.025 | 0.962 | 0.764 | 0.764 | 0.763 | 5,861 |
| 30 | 1.198 | 1.239 | 0.975 | 0.945 | 0.941 | 5,897 |
| 50 | 1.380 | 1.417 | 1.104 | 1.053 | 1.052 | 5,945 |
| 100 | 1.463 | 1.509 | 1.154 | 1.081 | 1.082 | 5,929 |
| 150 | 1.317 | 1.426 | 1.081 | 1.019 | 1.018 | 5,890 |
| 200 | 1.117 | 1.260 | 0.950 | 0.863 | 0.862 | 5,776 |
| 300 | 0.912 | 1.023 | 0.787 | 0.665 | 0.666 | 5,822 |
| 400 | 0.830 | 0.918 | 0.710 | 0.601 | 0.601 | 5,376 |
| 500 | 0.731 | 0.839 | 0.649 | 0.539 | 0.539 | 4,947 |
| 700 | 0.545 | 0.669 | 0.503 | 0.381 | 0.383 | 5,508 |
| 900 | 0.363 | 0.469 | 0.348 | 0.258 | 0.259 | 5,391 |
| 1100 | 0.238 | 0.303 | 0.232 | 0.178 | 0.179 | 4,517 |
| 1300 | 0.178 | 0.226 | 0.175 | 0.142 | 0.142 | 4,062 |
| 1500 | 0.135 | 0.177 | 0.133 | 0.114 | 0.114 | 3,736 |
| 1700 | 0.111 | 0.146 | 0.110 | 0.099 | 0.098 | 3,554 |
| 1900 | 0.090 | 0.114 | 0.087 | 0.074 | 0.074 | 3,390 |

### k = 10 · 30 d · 1,000 km · interspersed · salinity (PSU)

| dbar | climatology | nearest | inverse-distance | OI | OI+shrink | n |
|---|---|---|---|---|---|---|
| 10 | 0.266 | 0.375 | 0.232 | 0.203 | 0.200 | 5,621 |
| 30 | 0.243 | 0.234 | 0.184 | 0.171 | 0.172 | 5,660 |
| 50 | 0.230 | 0.320 | 0.195 | 0.174 | 0.173 | 5,669 |
| 100 | 0.190 | 0.205 | 0.159 | 0.147 | 0.146 | 5,655 |
| 150 | 0.158 | 0.179 | 0.136 | 0.124 | 0.123 | 5,631 |
| 200 | 0.137 | 0.157 | 0.119 | 0.106 | 0.105 | 5,541 |
| 300 | 0.109 | 0.127 | 0.098 | 0.082 | 0.082 | 5,588 |
| 400 | 0.094 | 0.111 | 0.085 | 0.072 | 0.072 | 5,171 |
| 500 | 0.078 | 0.097 | 0.073 | 0.060 | 0.060 | 4,759 |
| 700 | 0.055 | 0.069 | 0.052 | 0.041 | 0.041 | 5,295 |
| 900 | 0.037 | 0.047 | 0.036 | 0.028 | 0.028 | 5,186 |
| 1100 | 0.031 | 0.040 | 0.031 | 0.025 | 0.025 | 4,358 |
| 1300 | 0.030 | 0.039 | 0.030 | 0.025 | 0.025 | 3,929 |
| 1500 | 0.025 | 0.033 | 0.025 | 0.021 | 0.021 | 3,611 |
| 1700 | 0.020 | 0.026 | 0.020 | 0.017 | 0.017 | 3,432 |
| 1900 | 0.016 | 0.021 | 0.016 | 0.013 | 0.013 | 3,287 |

### k = 20 · 30 d · 1,000 km · terminal · temperature (°C)

| dbar | climatology | nearest | inverse-distance | OI | OI+shrink | n |
|---|---|---|---|---|---|---|
| 10 | 1.091 | 0.912 | 0.724 | 0.703 | 0.704 | 5,961 |
| 30 | 1.265 | 1.222 | 0.963 | 0.879 | 0.878 | 5,968 |
| 50 | 1.416 | 1.379 | 1.083 | 0.975 | 0.976 | 5,965 |
| 100 | 1.437 | 1.438 | 1.131 | 1.008 | 1.011 | 5,934 |
| 150 | 1.355 | 1.396 | 1.083 | 0.949 | 0.949 | 5,891 |
| 200 | 1.192 | 1.248 | 0.963 | 0.797 | 0.797 | 5,869 |
| 300 | 1.022 | 1.080 | 0.842 | 0.669 | 0.670 | 5,834 |
| 400 | 0.936 | 0.983 | 0.766 | 0.593 | 0.593 | 5,794 |
| 500 | 0.795 | 0.843 | 0.657 | 0.502 | 0.502 | 5,724 |
| 700 | 0.548 | 0.612 | 0.473 | 0.365 | 0.365 | 5,700 |
| 900 | 0.384 | 0.433 | 0.340 | 0.261 | 0.261 | 5,638 |
| 1100 | 0.243 | 0.297 | 0.220 | 0.172 | 0.172 | 5,013 |
| 1300 | 0.172 | 0.206 | 0.152 | 0.124 | 0.124 | 4,741 |
| 1500 | 0.128 | 0.156 | 0.114 | 0.091 | 0.091 | 4,635 |
| 1700 | 0.103 | 0.125 | 0.093 | 0.076 | 0.076 | 4,520 |
| 1900 | 0.089 | 0.107 | 0.080 | 0.065 | 0.065 | 4,368 |

### k = 20 · 30 d · 1,000 km · terminal · salinity (PSU)

| dbar | climatology | nearest | inverse-distance | OI | OI+shrink | n |
|---|---|---|---|---|---|---|
| 10 | 0.336 | 0.288 | 0.242 | 0.211 | 0.212 | 5,053 |
| 30 | 0.296 | 0.264 | 0.222 | 0.188 | 0.189 | 5,065 |
| 50 | 0.266 | 0.247 | 0.205 | 0.179 | 0.180 | 5,060 |
| 100 | 0.205 | 0.206 | 0.166 | 0.145 | 0.145 | 5,030 |
| 150 | 0.181 | 0.191 | 0.154 | 0.133 | 0.133 | 4,984 |
| 200 | 0.166 | 0.170 | 0.141 | 0.117 | 0.117 | 4,966 |
| 300 | 0.137 | 0.145 | 0.119 | 0.093 | 0.093 | 4,933 |
| 400 | 0.118 | 0.126 | 0.103 | 0.077 | 0.077 | 4,906 |
| 500 | 0.097 | 0.107 | 0.088 | 0.063 | 0.064 | 4,843 |
| 700 | 0.067 | 0.076 | 0.067 | 0.044 | 0.044 | 4,820 |
| 900 | 0.050 | 0.055 | 0.057 | 0.033 | 0.033 | 4,766 |
| 1100 | 0.045 | 0.050 | 0.046 | 0.030 | 0.030 | 4,235 |
| 1300 | 0.041 | 0.046 | 0.044 | 0.027 | 0.028 | 4,024 |
| 1500 | 0.038 | 0.042 | 0.036 | 0.025 | 0.025 | 3,947 |
| 1700 | 0.036 | 0.039 | 0.042 | 0.021 | 0.022 | 3,842 |
| 1900 | 0.034 | 0.037 | 0.039 | 0.021 | 0.021 | 3,710 |

### k = 20 · 30 d · 1,000 km · interspersed · temperature (°C)

| dbar | climatology | nearest | inverse-distance | OI | OI+shrink | n |
|---|---|---|---|---|---|---|
| 10 | 1.025 | 0.962 | 0.755 | 0.754 | 0.754 | 5,861 |
| 30 | 1.198 | 1.239 | 0.958 | 0.917 | 0.915 | 5,897 |
| 50 | 1.380 | 1.417 | 1.084 | 1.025 | 1.025 | 5,945 |
| 100 | 1.463 | 1.509 | 1.142 | 1.052 | 1.053 | 5,929 |
| 150 | 1.317 | 1.426 | 1.064 | 0.989 | 0.989 | 5,890 |
| 200 | 1.117 | 1.260 | 0.935 | 0.834 | 0.834 | 5,776 |
| 300 | 0.912 | 1.023 | 0.778 | 0.638 | 0.638 | 5,822 |
| 400 | 0.830 | 0.918 | 0.702 | 0.575 | 0.575 | 5,376 |
| 500 | 0.731 | 0.839 | 0.638 | 0.516 | 0.517 | 4,948 |
| 700 | 0.545 | 0.669 | 0.495 | 0.366 | 0.367 | 5,511 |
| 900 | 0.363 | 0.469 | 0.340 | 0.244 | 0.244 | 5,394 |
| 1100 | 0.237 | 0.303 | 0.226 | 0.170 | 0.170 | 4,524 |
| 1300 | 0.178 | 0.226 | 0.170 | 0.135 | 0.135 | 4,064 |
| 1500 | 0.135 | 0.177 | 0.130 | 0.109 | 0.109 | 3,740 |
| 1700 | 0.110 | 0.146 | 0.108 | 0.093 | 0.093 | 3,557 |
| 1900 | 0.090 | 0.114 | 0.085 | 0.071 | 0.071 | 3,392 |

### k = 20 · 30 d · 1,000 km · interspersed · salinity (PSU)

| dbar | climatology | nearest | inverse-distance | OI | OI+shrink | n |
|---|---|---|---|---|---|---|
| 10 | 0.266 | 0.375 | 0.219 | 0.198 | 0.197 | 5,621 |
| 30 | 0.243 | 0.234 | 0.183 | 0.169 | 0.169 | 5,660 |
| 50 | 0.230 | 0.320 | 0.188 | 0.167 | 0.167 | 5,669 |
| 100 | 0.190 | 0.205 | 0.156 | 0.140 | 0.140 | 5,655 |
| 150 | 0.158 | 0.179 | 0.133 | 0.119 | 0.119 | 5,631 |
| 200 | 0.137 | 0.157 | 0.118 | 0.103 | 0.102 | 5,541 |
| 300 | 0.109 | 0.127 | 0.096 | 0.079 | 0.079 | 5,588 |
| 400 | 0.094 | 0.111 | 0.083 | 0.068 | 0.069 | 5,171 |
| 500 | 0.077 | 0.097 | 0.071 | 0.057 | 0.057 | 4,761 |
| 700 | 0.055 | 0.069 | 0.051 | 0.039 | 0.039 | 5,298 |
| 900 | 0.037 | 0.047 | 0.035 | 0.027 | 0.027 | 5,190 |
| 1100 | 0.031 | 0.039 | 0.030 | 0.023 | 0.023 | 4,364 |
| 1300 | 0.030 | 0.039 | 0.030 | 0.024 | 0.024 | 3,933 |
| 1500 | 0.025 | 0.033 | 0.024 | 0.020 | 0.020 | 3,617 |
| 1700 | 0.020 | 0.026 | 0.019 | 0.016 | 0.016 | 3,438 |
| 1900 | 0.016 | 0.021 | 0.015 | 0.013 | 0.013 | 3,291 |

### k = 20 · 60 d · 1,500 km (the widened search) · terminal · temperature (°C)

| dbar | climatology | nearest | inverse-distance | OI | OI+shrink | n |
|---|---|---|---|---|---|---|
| 10 | 1.091 | 0.895 | 0.721 | 0.703 | 0.705 | 5,961 |
| 30 | 1.265 | 1.178 | 0.950 | 0.876 | 0.876 | 5,968 |
| 50 | 1.416 | 1.334 | 1.063 | 0.973 | 0.975 | 5,965 |
| 100 | 1.437 | 1.374 | 1.105 | 1.006 | 1.010 | 5,934 |
| 150 | 1.355 | 1.327 | 1.056 | 0.943 | 0.943 | 5,891 |
| 200 | 1.192 | 1.190 | 0.934 | 0.794 | 0.794 | 5,869 |
| 300 | 1.022 | 1.039 | 0.814 | 0.663 | 0.664 | 5,834 |
| 400 | 0.936 | 0.936 | 0.739 | 0.586 | 0.587 | 5,794 |
| 500 | 0.795 | 0.809 | 0.632 | 0.495 | 0.496 | 5,724 |
| 700 | 0.548 | 0.588 | 0.461 | 0.362 | 0.363 | 5,700 |
| 900 | 0.384 | 0.420 | 0.332 | 0.258 | 0.259 | 5,638 |
| 1100 | 0.243 | 0.289 | 0.214 | 0.168 | 0.169 | 5,013 |
| 1300 | 0.172 | 0.199 | 0.148 | 0.122 | 0.122 | 4,741 |
| 1500 | 0.128 | 0.150 | 0.111 | 0.091 | 0.091 | 4,635 |
| 1700 | 0.103 | 0.122 | 0.091 | 0.075 | 0.075 | 4,520 |
| 1900 | 0.089 | 0.104 | 0.078 | 0.065 | 0.065 | 4,368 |

### k = 20 · 60 d · 1,500 km (the widened search) · terminal · salinity (PSU)

| dbar | climatology | nearest | inverse-distance | OI | OI+shrink | n |
|---|---|---|---|---|---|---|
| 10 | 0.336 | 0.281 | 0.238 | 0.209 | 0.210 | 5,053 |
| 30 | 0.296 | 0.256 | 0.217 | 0.185 | 0.186 | 5,065 |
| 50 | 0.266 | 0.242 | 0.200 | 0.177 | 0.177 | 5,060 |
| 100 | 0.205 | 0.201 | 0.163 | 0.145 | 0.145 | 5,030 |
| 150 | 0.181 | 0.184 | 0.151 | 0.131 | 0.131 | 4,984 |
| 200 | 0.166 | 0.165 | 0.137 | 0.116 | 0.117 | 4,966 |
| 300 | 0.137 | 0.142 | 0.115 | 0.092 | 0.092 | 4,933 |
| 400 | 0.118 | 0.120 | 0.099 | 0.076 | 0.076 | 4,906 |
| 500 | 0.097 | 0.102 | 0.085 | 0.063 | 0.063 | 4,843 |
| 700 | 0.067 | 0.073 | 0.065 | 0.044 | 0.044 | 4,820 |
| 900 | 0.050 | 0.053 | 0.055 | 0.033 | 0.033 | 4,766 |
| 1100 | 0.045 | 0.049 | 0.045 | 0.030 | 0.030 | 4,235 |
| 1300 | 0.041 | 0.045 | 0.043 | 0.027 | 0.028 | 4,024 |
| 1500 | 0.038 | 0.041 | 0.037 | 0.025 | 0.025 | 3,947 |
| 1700 | 0.036 | 0.039 | 0.041 | 0.021 | 0.022 | 3,842 |
| 1900 | 0.034 | 0.036 | 0.040 | 0.021 | 0.021 | 3,710 |

### k = 20 · 60 d · 1,500 km (the widened search) · interspersed · temperature (°C)

| dbar | climatology | nearest | inverse-distance | OI | OI+shrink | n |
|---|---|---|---|---|---|---|
| 10 | 1.025 | 0.944 | 0.754 | 0.758 | 0.757 | 5,861 |
| 30 | 1.198 | 1.210 | 0.946 | 0.918 | 0.916 | 5,897 |
| 50 | 1.380 | 1.377 | 1.063 | 1.021 | 1.021 | 5,945 |
| 100 | 1.463 | 1.452 | 1.116 | 1.042 | 1.044 | 5,929 |
| 150 | 1.317 | 1.380 | 1.044 | 0.970 | 0.969 | 5,890 |
| 200 | 1.117 | 1.205 | 0.911 | 0.814 | 0.814 | 5,776 |
| 300 | 0.912 | 0.974 | 0.754 | 0.629 | 0.630 | 5,822 |
| 400 | 0.830 | 0.871 | 0.681 | 0.567 | 0.568 | 5,376 |
| 500 | 0.731 | 0.788 | 0.617 | 0.509 | 0.510 | 4,948 |
| 700 | 0.545 | 0.634 | 0.480 | 0.358 | 0.360 | 5,510 |
| 900 | 0.363 | 0.441 | 0.329 | 0.241 | 0.241 | 5,393 |
| 1100 | 0.237 | 0.288 | 0.219 | 0.167 | 0.167 | 4,524 |
| 1300 | 0.178 | 0.210 | 0.166 | 0.132 | 0.132 | 4,063 |
| 1500 | 0.135 | 0.163 | 0.127 | 0.106 | 0.106 | 3,739 |
| 1700 | 0.111 | 0.138 | 0.106 | 0.092 | 0.092 | 3,556 |
| 1900 | 0.090 | 0.108 | 0.083 | 0.070 | 0.070 | 3,393 |

### k = 20 · 60 d · 1,500 km (the widened search) · interspersed · salinity (PSU)

| dbar | climatology | nearest | inverse-distance | OI | OI+shrink | n |
|---|---|---|---|---|---|---|
| 10 | 0.266 | 0.367 | 0.211 | 0.198 | 0.197 | 5,621 |
| 30 | 0.243 | 0.231 | 0.178 | 0.168 | 0.169 | 5,660 |
| 50 | 0.230 | 0.316 | 0.181 | 0.164 | 0.164 | 5,669 |
| 100 | 0.190 | 0.198 | 0.152 | 0.138 | 0.138 | 5,655 |
| 150 | 0.158 | 0.172 | 0.130 | 0.118 | 0.118 | 5,631 |
| 200 | 0.137 | 0.151 | 0.115 | 0.102 | 0.102 | 5,541 |
| 300 | 0.109 | 0.122 | 0.093 | 0.079 | 0.079 | 5,588 |
| 400 | 0.094 | 0.107 | 0.081 | 0.068 | 0.068 | 5,171 |
| 500 | 0.077 | 0.093 | 0.069 | 0.057 | 0.057 | 4,761 |
| 700 | 0.055 | 0.066 | 0.049 | 0.038 | 0.038 | 5,297 |
| 900 | 0.037 | 0.045 | 0.034 | 0.026 | 0.026 | 5,189 |
| 1100 | 0.031 | 0.038 | 0.029 | 0.022 | 0.022 | 4,364 |
| 1300 | 0.030 | 0.037 | 0.029 | 0.024 | 0.024 | 3,933 |
| 1500 | 0.025 | 0.031 | 0.024 | 0.020 | 0.020 | 3,617 |
| 1700 | 0.020 | 0.025 | 0.019 | 0.016 | 0.016 | 3,438 |
| 1900 | 0.016 | 0.020 | 0.015 | 0.013 | 0.013 | 3,293 |

## 5 · The sensitivity pass: what if the companion profiles are removed?

The obvious objection to a 31 % gain is that an Argo float cycles every ten days or so, so about one target in nine has a profile from the same float within 20 km of it — and finding your own previous cycle is not interpolation. This section re-scores every estimator, **with the hyper-parameters unchanged and nothing refitted**, on the targets whose nearest neighbour is at least 50 km away.

| search | split | variable | targets | OI skill, all | OI skill, ≥ 50 km |
|---|---|---|---|---|---|
| k = 5 · 30 d · 1,000 km | terminal | temperature | 4,814 of 6,000 | 0.7856 | 0.8333 |
| k = 5 · 30 d · 1,000 km | terminal | salinity | 4,814 of 6,000 | 0.8552 | 0.9042 |
| k = 5 · 30 d · 1,000 km | interspersed | temperature | 4,820 of 6,000 | 0.8252 | 0.8565 |
| k = 5 · 30 d · 1,000 km | interspersed | salinity | 4,820 of 6,000 | 0.8426 | 0.8702 |
| k = 10 · 30 d · 1,000 km | terminal | temperature | 4,814 of 6,000 | 0.7113 | 0.7505 |
| k = 10 · 30 d · 1,000 km | terminal | salinity | 4,814 of 6,000 | 0.7856 | 0.8279 |
| k = 10 · 30 d · 1,000 km | interspersed | temperature | 4,820 of 6,000 | 0.7689 | 0.7936 |
| k = 10 · 30 d · 1,000 km | interspersed | salinity | 4,820 of 6,000 | 0.7818 | 0.8038 |
| k = 20 · 30 d · 1,000 km | terminal | temperature | 4,814 of 6,000 | 0.6857 | 0.7218 |
| k = 20 · 30 d · 1,000 km | terminal | salinity | 4,814 of 6,000 | 0.6612 | 0.6926 |
| k = 20 · 30 d · 1,000 km | interspersed | temperature | 4,820 of 6,000 | 0.7391 | 0.7618 |
| k = 20 · 30 d · 1,000 km | interspersed | salinity | 4,820 of 6,000 | 0.7525 | 0.7701 |
| k = 20 · 60 d · 1,500 km (the widened search) | terminal | temperature | 4,445 of 6,000 | 0.6807 | 0.7231 |
| k = 20 · 60 d · 1,500 km (the widened search) | terminal | salinity | 4,445 of 6,000 | 0.6561 | 0.6890 |
| k = 20 · 60 d · 1,500 km (the widened search) | interspersed | temperature | 4,363 of 6,000 | 0.7289 | 0.7539 |
| k = 20 · 60 d · 1,500 km (the widened search) | interspersed | salinity | 4,363 of 6,000 | 0.7410 | 0.7605 |

**The objection is real and it is not the answer.** Removing the 20 % of targets with a companion inside 50 km costs about 0.03-0.04 of skill — the terminal-split k = 20 temperature number moves from 0.686 to 0.722 and salinity from 0.661 to 0.693. That is a genuine part of the gain and it is now quantified; the remaining **28 % and 31 % reduction in error is earned against neighbours at least 50 km away**, and it is still an order of magnitude more than the codec extracted with the companions included.

## 6 · Beside E-076a's model arms

E-076a's two arms are a 7 M-parameter cone codec trained for 20,000 steps: the **twin** reads the Argo interior as the gridded monthly column it always had, the **family-8 arm** reads the five nearest raw float profiles with their offsets. Both were scored on the same kind of target as this document — the nearest real profile's temperature and salinity, in raw units, against the same climatology — with the same k = 5, thirty-day, 1,000 km search. Its numbers below are transcribed from `ml/EXPERIMENTS.md#e-076a`; ours are the k = 5 row of §2, so the search is matched.

| terminal split, mean skill over 16 levels | temperature | salinity |
|---|---|---|
| E-076a twin, seed 0 (gridded column) | 0.9747 | 0.9992 |
| E-076a family-8 arm, seed 0 (five raw profiles) | 0.9700 | 0.9986 |
| E-076a twin, seed 1 | 0.9793 | 0.9989 |
| **E-076b optimal interpolation, k = 5** | **0.7856** | **0.8552** |
| **E-076b optimal interpolation, k = 20** | **0.6857** | **0.6612** |

On the SAME search the classical estimator takes 21 % off the temperature error where the codec took 2.5-3.0 %, and 14 % off the salinity error where the codec took 0.1 %. That is a factor of seven to eight on temperature and a factor of over a hundred on salinity, and it is nowhere near any seed spread: the twin's own two seeds differ by 0.0046, and the gap here is 0.19.

Per level, temperature, at the matched k = 5 search. **Read the two experiments' RMSE columns against their OWN climatology column and not against each other**: E-076a scores all held-out anchors and E-076b the terminal split, on differently distributed targets (§1.1), so the absolute degrees are not comparable across the vertical rule — only each estimator's distance from the bar beside it is.

| dbar | E-076a twin s0 | E-076a family 8 s0 | E-076a climatology | E-076b climatology | **E-076b OI (k=5)** | **E-076b OI (k=20)** |
|---|---|---|---|---|---|---|
| 10 | 0.981 | 0.964 | 1.086 | 1.091 | **0.758** | **0.703**|
| 30 | 1.152 | 1.134 | 1.232 | 1.265 | **0.961** | **0.879**|
| 50 | 1.271 | 1.255 | 1.328 | 1.416 | **1.072** | **0.975**|
| 100 | 1.248 | 1.234 | 1.306 | 1.437 | **1.103** | **1.008**|
| 200 | 1.048 | 1.044 | 1.061 | 1.192 | **0.900** | **0.797**|
| 300 | 0.825 | 0.819 | 0.831 | 1.022 | **0.762** | **0.669**|
| 500 | 0.632 | 0.634 | 0.637 | 0.796 | **0.589** | **0.502**|
| 900 | 0.327 | 0.329 | 0.330 | 0.384 | **0.318** | **0.261**|
| 1500 | 0.150 | 0.150 | 0.151 | 0.128 | **0.107** | **0.091**|

## 7 · What this answers, and what it does not

**The question E-076a registered was: can ANY method extract interior skill beyond climatology from the k nearest Argo profiles within 1,000 km and 30 days?** It registered the two readings in advance: *"If OI beats climatology by 10 %, the codec under-trained or under-used its tokens and a longer/larger arm is justified; if OI cannot beat climatology either, the k = 5 / 30-day search is too sparse for the interior and family 8's value lies elsewhere."*

**The first branch is taken, with room to spare.** OI beats climatology by **21 %** at the matched k = 5 and by **31 %** at k = 20, on temperature, on the terminal held-out years; by 14 % and 34 % on salinity; and by 28 % and 31 % on temperature when the companion profiles are removed. The registered threshold was 10 %.

Four things follow.

1. **E-076a's null is about the model, not about the data.** Its own closing sentence — *"neither the gridded column nor five raw profiles teach a 7 M cone codec anything about the interior in 20 k steps"* — is correct as written, and the reason is now known: the information was there and the codec did not use it. A 7 M-parameter model at 20,000 steps, with the interior as one of many reconstruction targets, is not a fair test of the representation.
2. **The E-076a ablation still did not discriminate between the two representations, and it still cannot be read as one.** OI is not a proxy for either arm: it is given the raw profiles with their offsets, i.e. the family-8 representation, and there is no OI equivalent of reading the gridded column. What this experiment settles is the ceiling, not the comparison.
3. **k = 5 is the wrong k.** The array has ~140 profiles inside the search bounds at a median target and the skill is still improving at k = 20 (0.79 → 0.71 → 0.69). Any future arm should carry more neighbours, and widening the radius instead of raising k is measured here to be nearly worthless.
4. **A useful yardstick now exists for any interior read-out.** OI's per-level RMSE in §4 is a bar a model must beat to be said to have learnt interpolation at all, and it costs ten minutes of CPU to recompute.

**What it does not answer.** It says nothing about FORECASTING — every number here is an estimate of the present state from present measurements, which is the analysis problem, not the prediction problem. It says nothing about whether a codec that reached this ceiling would forecast better; that is the programme's actual question and needs a model. And it is a single seed of the target draw: the estimators are deterministic given the targets, so the only sampling term is the draw itself, at n = 6,000 per split, but the numbers are RMSE ratios and their sampling error is roughly 1 % relative — small against the effects reported, large against the 0.005 that separates the widened search from the plain one.

## 8 · Reproducing it

```
python3 ml/oi_ceiling.py \
    --store  <dir>/family8_argo_l0 \
    --tensor-npz <dir>/family7_global025_pentad_l0.npz \
    --rg100  <dir>/family7_global025_pentad_l0_X_rg100.npy \
    --out    <dir>/e076b --n 6000 --seed 0
```

The full result — every level, every count, the fitted hyper-parameters and the sensitivity pass — is committed beside this document as `ml/plans/E076b_results.json`, which is the file every table above is generated from.

The store and the two family-7 files are on the public Hugging Face dataset `chfrank/earth-tensors` under `tensors/family8_argo_l0/` and `tensors/family7_global025_pentad_l0/`; no token is needed and every file's size and sha256 are in that prefix's own `manifest.json`. The run needs numpy and nothing else, takes about eleven minutes on two CPU cores and about 1.1 GB of disk for the interior tensor. `tests/test_oi_ceiling.py` checks the estimator against a Gaussian field whose answer is known in closed form — the OI error reaches the value the theory predicts, the grid search recovers the field's true correlation length and decorrelation time to within one grid step, and a missing level is shown to be exactly equivalent to a smaller linear system — and needs no network at all.

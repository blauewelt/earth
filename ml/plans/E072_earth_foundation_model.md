# E-072 · The Earth foundation model — data design first, then the scaling ladder, then the recipe

**Written 2026-09-04**, from Chris's ask of the same afternoon: *"I'd like to
build an earth foundation model (1B – 10B parameters, maybe MoE based), so a
chinchilla scaling ladder and a comprehensive study on how to design the
training data is needed. … Also include how to best train the foundation model
(what kind of architecture, what learning rate regime, what z-noise, which
optimizer, which loss). The question to you is how to best represent all the
worlds data inside a large model. And then start predicting El Nino and AMOC
and surface color and whatever we need."*

This is a PLAN with numbers, not a result. Every number below is either
measured in this programme (cited to its log entry) or a stated assumption
with its arithmetic shown. Nothing here is dispatched.

Read with:
[E-071 · cone v2](https://blauewelt.github.io/earth/docs.html?f=ml/plans/E071_cone_v2.md)
(the sampler, the harmonic anomaly, profile tokens, the design speeds, land
and ice and the shared-channel principle — the DATA half of this plan),
[E-070 · the global tensor](https://blauewelt.github.io/earth/docs.html?f=ml/plans/E070_global_tensor.md),
[the data ladder](https://blauewelt.github.io/earth/docs.html?f=ml/plans/DATA_LADDER.md),
[the scaling audit](https://blauewelt.github.io/earth/docs.html?f=ml/SCALING.md),
[the protocol reset](https://blauewelt.github.io/earth/docs.html?f=ml/plans/PROTOCOL_RESET.md),
[the standing overview](https://blauewelt.github.io/earth/docs.html?f=ml/OVERVIEW.md).

---

## 0 · The one-paragraph answer

At the data this programme has today — one basin, 0.25°, five-day bins, 42
ocean channels — the model size is set by the data and it is small: the
width ladder is flat from 7.6 M to 206.66 M parameters on the rolled field
(E-062-R0, three rungs, 0.105 / 0.104 / 0.103), a 200-mode *linear* model
out-forecasts every learned head from 15 days on (E-066), and every held-out
minimum arrives inside 2,000 steps (E-059/E-060). A 1–10 B model is therefore
not a training decision, it is a **data-design decision**: the globe (×8
cells), the other spheres (×4.5 values, E-071 §6), daily cadence (×5) and the
kilometre-scale rungs of the input ladder (×770) are what raise the
value-count ceiling from ~0.12 B parameters to 1 B, 4.5 B, 23 B and beyond —
and at each rung the compute stays cheap (a 1 B model over the global
pentad tensor is ~30 RTX-4090-hours) until the kilometre rung, where it
finally becomes compute-bound. The recipe (§4) is the one this programme has
already paid to learn — a location-token encoder with a queryable Gaussian
decoder, masked reconstruction plus forecast, AdamW with a short warm-up and
an early held-out minimum, embedding noise as the one regulariser that has
measured positive — with two additions the new data forces: mixture-of-experts
routed by sphere, and a null ladder (persistence, LIM, retrieval) beside every
read-out. The read-outs (§5) are heads on one frozen representation: Niño-3.4
and the warm-water volume, RAPID and the other arrays, ocean colour, sea ice,
land water — and the first three each have a data prerequisite named in
E-071 and CONE_DATA_AND_ENSO.md that must land before the head is worth
training.

---

## 1 · What the programme has measured that a foundation model must respect

| finding | where | what it means for E-072 |
|---|---|---|
| Rolled skill is flat across 7.6 M → 40.4 M → 206.66 M stage-2 heads (corridor acc 0.103 / 0.104 / 0.105); the whole spread in MSSS is amplitude calibration | E-062-R0 (#516, #520, #523) | at today's data, capacity is not the axis; **more parameters buy nothing until the data grows** |
| Every clean head's held-out minimum is reached inside ~2,000 steps and worsens for the next 198,000 | E-059, E-060 | the regime is data-limited: **early stopping and ≤ 4 epochs**, never long schedules |
| A 200-mode linear inverse model beats the learned head at every lead ≥ 15 d and beats damped persistence from 30 d | E-066 (#527) | the **null ladder is mandatory**; a foundation model must be read against the LIM in its own embedding space, not against persistence alone |
| Input noise on the embedding (z-noise 0.7) is the one regulariser worth +0.045 / +0.050 on the rolled field | E-036, E-037 | keep it; it is a denoising objective in disguise (§4.4) |
| Contamination: a dense window loss teacher-forced on held-out bins produced near-unity skill at 365 d — a memorised trajectory | the protocol reset, c25f6ff | **window-scope pool discipline and the terminal holdout are non-negotiable** at any scale; the bigger the model, the more it can memorise |
| The cone codec learns persistence from history (hidden SSH ~0.36 of bar vs 0.75–0.90; +7–8 % at 1–2 pentads) but **not velocity from displacement** at 7 M / 20 k steps, under two objectives, five seeds | E-069, E-069b | a displacement primitive is a modelling question (capacity, objective, architecture), not a data question; do not promise it from scale alone |
| The 32 upsampled Argo channels are 80 % of the bytes and ~0.28 GB of information; null at every lead in the rolled evaluator | E-062, DATA_LADDER §2 | **information per byte decides the data design**, never bytes |
| Chinchilla anchor at values / 20 is a *ceiling* because 0.25° cells are correlated over hundreds of km | SCALING.md, E-070 §1 | quote the ceiling and the measured flatness together |

---

## 2 · The data-design study — how to represent all the world's data

### 2.1 · The unit: a token is a place at a time, carrying every channel there

E-071 §6.4's token — one (location, lag) carrying the full value vector of
every channel with a per-channel observed mask, the sphere code and the
Fourier-encoded coordinates (latitude, longitude, time, depth or height) —
is the representation unit of the foundation model. Three consequences:

- **Channel count is free.** Adding a channel adds a column to the value
  vector, not a token. The codec's cost is in tokens (≈ 300 per anchor,
  E-071 §6.4), the data's cost is in bytes.
- **Native point observations are tokens too.** An Argo profile, a
  radiosonde, a weather station, a drifter, a tide gauge is a token with its
  own coordinates and a value vector that is mostly `miss` — the profile
  token of E-071 §3 already is one. "Dots in, embedding out" (survey deck
  slide 41) is this rule.
- **Imagery is a sub-token.** A 10 m – 1 km patch (rungs 1–3 of the input
  ladder) is embedded by a small patch encoder into a fixed-width vector
  that rides in the value vector of the 0.25° token it falls in — the
  coarse token carries a summary of its fine content, and the fine patches
  are queried only by a decoder asked about that place. This is how the
  kilometre rung enters without a kilometre-scale token grid.

### 2.2 · The ladder of data, in observed values

Assumptions: 0.25° pentads 1982-01 → 2024-12 (3,142 bins); the global grid
721 × 1,440; ~686 k ocean cells and ~352 k land + ice cells; channel sets as
in E-071 §6; a token carries 65 channels. "Ceiling" is values / 20, the
programme's stated Chinchilla anchor, and it IS a ceiling (§1, last row).

| rung of data | observed values | tokens (÷ 65) | ceiling, values / 20 | what it takes |
|---|---|---|---|---|
| North Atlantic ocean pentads, r3 — today | 2.5 B | 0.04 B | 0.12 B params | exists |
| global ocean pentads, family 7 | 20 B | 0.31 B | 1.0 B | E-070 Phases B–F (pull done; the tensor is not built) |
| + the shared 20 and land-only 5 channels of E-071 §6 | 91 B | 1.4 B | 4.5 B | ERA5 (needs the CDS account), MODIS, ASCAT, SMAP, GRACE, OSI SAF, CCI soil moisture: ~170 GB |
| daily cadence, all spheres | 455 B | 7.0 B | 23 B | the daily global tensor is 1.3 TB (E-070 §8); the streaming loader E-033 Phase 3 deferred |
| 1 km daily, all spheres (rung 3 radiometers) | 350,000 B | 5,400 B | 17,000 B | not a tensor: a patch-encoder feed (§2.1); compute-bound for the first time |

Reading it honestly: **the 1–10 B target lives between rows 3 and 4**, and
both rows are data we do not have yet. Row 2 is a 1 B ceiling with the
measured caveat that the North Atlantic's 0.12 B ceiling was flat from
7.6 M. The multiplier that matters most per unit of work is row 3 — the
other spheres — because it multiplies values by 4.5 at ~170 GB, whereas
daily cadence multiplies by 5 at 1.3 TB and, being autocorrelated at one
day, adds fewer *independent* values than it adds bytes.

### 2.3 · What "independent" costs — the correction the ceiling needs

A 0.25° cell is correlated with its neighbours over ~300 km (the
correlation length the cone families use) and over 2–6 months at the
surface (survey deck, family B). In the North Atlantic that is ~150 cells
and ~15 pentads per independent sample, i.e. the 2.5 B raw values are of
order 1–10 M independent ones — which is exactly the regime in which a
7.6 M head equals a 206 M head. The multipliers in §2.2 therefore buy less
than they say: the globe adds *new regimes* (independent by construction —
E-070's "regimes are samples"), the spheres add *new physics* (land does not
advect; the atmosphere decorrelates in days), and those are worth their
face value; daily cadence within a pentad is mostly the same sample five
times and is worth perhaps ×1.5; the kilometre rung adds fine-scale
variance that is genuinely new. The ladder's honest reading: **globe and
spheres first, then resolution, cadence last.**

### 2.4 · The five design rules

1. **Observed over derived** (survey deck slide 39; GENERIC_EMBEDDING_INPUTS
   §2): a derived product (a reanalysis, an L4 gap-fill, an index) is
   admitted only for information it carries that we do not ingest raw, is
   flagged by a source token, and is dropped from the input at ≥ 50 % of
   anchors so the model also works from observations alone. Reanalyses peek
   forward by half their assimilation window and are time-shifted (E-071
   §1's "5 of 42 channels").
2. **Information per byte decides** (E-062's Argo finding). Every candidate
   channel is priced in the data ladder's form: bytes, values, live
   fraction, correlation length, memory — and a channel that is 80 % of the
   bytes for 1 % of the information is stored as a sidecar, never in the
   dense tensor.
3. **A channel is a quantity and an instrument, never a sphere** (E-071
   §6.1), with the correction of 4 Sep: shared only when the measurand and
   the instrument are the same on both sides — ERA5's fields, ASCAT's σ⁰,
   the altimeters; the observed sea-surface and land-surface temperatures
   are separate channels because they measure different things.
4. **Every value carries its time and its sphere.** The harmonic
   climatology (E-071 §2) removes the seasonal cycle continuously; the
   sphere code and the depth/height coordinate are inputs, so one decoder
   can answer for an ocean column, a soil column and a firn column.
5. **The holdout is terminal and spatial** (the frozen protocol): train
   ≤ 2020, test 2021–2024, plus longitude holes — at any size. A
   foundation model that has seen 2021–2024 can never be scored on the 2026
   El Niño.

---

## 3 · The scaling ladder — what to train, in what order, and what each rung must show

Compute is priced at 6·N·T·epochs FLOPs, an RTX 4090 at 165 TFLOP/s bf16
(E-033's measurement) at 40 % utilisation, a v5e-4 TPU node at 4 × 197
TFLOP/s at the same utilisation. Four epochs, the data-constrained limit
(Muennighoff et al. 2023, SCALING.md), except at the top rung.

| rung | N | data | tokens × epochs | 4090-hours | v5e-4 hours | the question it must answer before the next rung is bought |
|---|---|---|---|---|---|---|
| L0 | 7 M (E-069's ConeMAE) | NA pentads | 0.15 B | < 1 | < 1 | done: persistence yes, velocity no |
| L1 | 30–125 M | NA pentads, cone v2 geometry + harmonic anomaly + profile tokens | 0.15 B | < 1 | < 1 | does the geometry change (E-071) move the per-family held-out loss and the rolled skill at leads 1–3 beyond the LIM? |
| L2 | 0.3–1 B | global ocean pentads (family 7) | 1.2 B | ~30 | ~7 | G1–G3 of E-070: does the globe transfer (Kuroshio, ACC) and does the NA number hold at matched steps? |
| L3 | 1–4.5 B, MoE | + all spheres (E-071 §6) | 5.6 B | ~570 | ~120 | does a sphere-routed MoE beat a dense model of the same active parameters; do land and atmosphere improve the ocean's rolled skill (the forcing argument) and vice versa |
| L4 | 4–23 B, MoE | daily, all spheres | 14 B (2 epochs) | ~3,500 | ~740 | the first compute-bound rung; is the daily cadence worth ×5 bytes on the terminal holdout |
| L5 | 10 B+ | + kilometre patches | — | compute-bound | — | only after L4 has a positive answer |

Rules of the ladder: **each rung is bought only on the previous rung's
read-out, under the frozen protocol, with the null ladder beside it**
(§4.6); a rung whose held-out minimum arrives inside 2,000 steps at full
data is over-parameterised and the next rung is not bought; the width
ladder result (flat 7.6 M → 206 M) is re-measured at L2 before L3 exists,
because it is the one measurement that can tell a data-limited rung from a
compute-limited one.

---

## 4 · The training recipe

### 4.1 · Architecture

- **Encoder: a transformer over location tokens (E-071 §6.4), Perceiver-style
  cross-attention into a fixed latent set** — ConeMAE's shape, which is
  already ported to JAX/TPU and certified (E-069's C1–C10 gates). Latents
  64 × 256 at 7 M; the ladder widens latents and depth together, and at L3
  the FFN becomes a **mixture of experts routed by sphere code and channel
  group** (ocean / land / ice / atmosphere): the value vectors differ by
  sphere in statistics and physics, expert specialisation is the natural
  fit, and active parameters stay at the dense rung's cost. Top-2 routing
  with a load-balancing loss; the router sees the sphere code, so routing
  is partly deterministic and the balance loss is small.
- **Decoder: the queryable Gaussian decoder, reading z alone** (E-069's
  closed degeneracy — a [z + latents] memory lets the bottleneck carry
  nothing). Queries are (coordinates, channel) and the answer is a mean and
  a variance, so one decoder serves every read-out and every point
  observation.
- **Stage 2 stays a separate sequence model over embeddings** (the "two
  stencils" split of E-071 §4.5, now in time): a transformer over the
  outer window of embeddings, global rings per lag, K = 144 pentads with a
  log-spaced longer window as the deferred option; it is what rolls
  forward, and it is what the read-outs are scored on. A joint
  encoder–forecaster (E-006's `--loss-mode data`) is the follow-up once the
  frozen-codec version has a level.

### 4.2 · Objective

Masked reconstruction over hidden values and hidden dots (the E-069b
masking plan: a dropped channel is hidden at the present-day patch only,
hidden dots from the recent-lag band and a bearing wedge, the anchor family
scored on hidden channels only) **plus** the future-target family at lags
+1, +2 — a Gaussian negative log-likelihood, so the model is scored on its
stated uncertainty. The per-family predict-the-mean bar (E-069b) is logged
at every eval and is the diagnostic that says which family carries signal.
For the forecaster: the same NLL on the next embedding, with the fair-CRPS
functional-generative head (E-052/E-057, FGN) as the probabilistic option
once the deterministic level exists.

### 4.3 · Optimiser and schedule

AdamW, β₁ 0.9, β₂ 0.95, weight decay 0.1, gradient clip 1.0 on standardised
inputs (the programme's 128 was on un-standardised targets), bf16 with
fp32 master weights, 2,000 warm-up steps, cosine to 10 % of peak. Peak
learning rate transfers across the ladder by the **width ladder, not by
guess**: 3 × 10⁻⁴ at 7 M (E-069), 4 × 10⁻⁴ at the 1280 × 20 heads (E-054);
at L2 run a three-point LR sweep at the smallest width and carry it up
with μP-style scaling (the per-layer parametrisation that keeps the optimal
LR fixed across width) — cheap at these compute costs, and the one thing
that makes a 4.5 B run a one-shot rather than a search. **Early stopping at
the held-out minimum**, with the minimum's step recorded, and ≤ 4 epochs.

### 4.4 · Noise, and why

z-noise 0.7 on the forecaster's input embeddings (E-036/E-037: +0.045 /
+0.050 rolled; dose-matched per lattice in E-064/E-065) stays. Its mechanism
is denoising: the forecaster learns the manifold of plausible embeddings
rather than the identity, which is also why it is the one regulariser that
survived the protocol reset. For the encoder, the masking plan is the
noise. A larger model at L3+ should re-measure the dose — the optimum moved
with the lattice in E-065 and will move with width.

### 4.5 · What a bigger model does NOT fix

The cone codec's H1 (velocity from displacement) failed at 7 M under two
objectives and on a synthetic field; nothing in §2–4 addresses it. If a
displacement primitive matters — for eddy propagation it does — it is a
separate arm: an explicit correlation or optical-flow head over consecutive
pentads, or a relative-position attention bias, tested at L1 on the
synthetic advection field (`tests/test_cone_advection.py`) before any
scale is spent on it.

### 4.6 · The null ladder every rung reports

Climatology · persistence · damped persistence · the LIM at K = 200 in
pixel space and in the rung's embedding space · nearest-analogue retrieval
· the previous rung. MSSS against climatology and against damped
persistence, per lead, per channel, per scope (trained / held-out
longitudes), with block-bootstrap intervals; amplitude calibration
reported separately (E-062: the −0.439 that was +0.019 calibrated).
Replication per ml/CLAUDE.md §3b: the first result at every new tier buys
its pair.

---

## 5 · The read-outs — what to predict, and what each needs first

Every read-out is a head on the frozen representation plus the rolled
forecast — never a separate model — so adding one costs a probe, not a run.

| read-out | the label | the data prerequisite | the null it must beat |
|---|---|---|---|
| **El Niño** — Niño-3.4 / ONI, and the 2026 event's peak and exit | our own OISST bake (Niño-3.4 −0.57 Jan → +2.09 Jul 2026, CONE_DATA_AND_ENSO §4) reconciled with CPC's ONI | the tropical Pacific is in family 7 (E-070); the subsurface (GODAS / ORAS5 or the warm-water volume) and daily winds are the additions ranked in CONE_DATA_AND_ENSO §4.5 — without them the head is an SST-persistence forecast | the IRI/CPC plume; the LIM; the spring-barrier skill curve |
| **AMOC** — RAPID 26.5° N, then MOVE, OSNAP, SAMBA, the Florida cable | the truth series already in the tensor (`truth_rapid`, `truth_fc`) | E-071's buoyancy forcing (ERA5 fluxes) and the 3-D interior (GREP) — the two additions DATA_LADDER ranks first; the terminal holdout for a low-frequency number (E-062's 3-year roll finding) | damped persistence; the LIM; the wind-stress ridge bar |
| **surface colour** — chlorophyll (OC-CCI 4 km 1997→, PACE 2024→) | the OC-CCI / PACE product, as a **target** channel (E-071 §6.1: reflectance is the input, colour the target) | reflectance bands in the shared set; BGC-Argo profile tokens | persistence; the seasonal harmonic; a ridge on reflectance |
| sea ice — concentration, extent, the September minimum | OSI SAF / NSIDC, already the frozen-fraction channel | nothing new: it is in Phase L0 | damped persistence; the LIM |
| land water — GRACE storage, soil moisture, snow | GRACE, CCI SM, MOD10 | Phase L0 / L1 of E-071 §6.5 | persistence; the harmonic |
| the atmosphere — 2 m temperature, precipitation at 5–30 days | ERA5, IMERG | Phase L0 | ECMWF's own open-data forecast (a strong, free null) |

The order of work is the order of the prerequisites: sea ice and land
water are free the day Phase L0 exists; El Niño needs the Pacific subsurface;
AMOC needs the buoyancy forcing and the interior; colour needs the
reflectance bands and the biosphere sphere.

---

## 6 · What to do first, in one list

1. E-071 on the North Atlantic (L1): cone v2 geometry + harmonic anomaly +
   profile tokens, per-family loss and rolled skill against the LIM — the
   cheapest measurement that tells whether the data half of this plan moves
   anything.
2. Family 7's Phases B–F (the global tensor is pulled, not built) and the
   L2 rung with E-070's gates G1–G3, and the width ladder re-measured there.
3. Phase L0 of E-071 §6 (six ERA5 / ice / soil channels, ~40 GB) and the
   first sea-ice and land-water heads — the cheapest new read-outs.
4. The Pacific subsurface and daily winds (CONE_DATA_AND_ENSO §4.5) → the
   El Niño head, scored on the 2026 event's peak and exit against the IRI
   plume — the read-out with a public benchmark.
5. L3: the sphere-routed MoE at 1–4.5 B, on the all-spheres tensor, with
   the μP LR transfer measured at L2.

Nothing above the L1 rung is dispatched by this plan; each rung is bought on
the one below it.

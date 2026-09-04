# Handover, 2026-09-04 — cone v2, the four spheres, the Earth foundation model, and where to fork

**For:** any session picking up one area of this work to go deeper. Written
by the Fable session of 3–4 Sep 2026 at Chris's request (*"write a
comprehensive summary of the work so far, which I want to pass to other
sessions to fork going deeper into certain areas"*). Every claim links to
its artefact; nothing here is a result that is not in
`ml/EXPERIMENTS.md` or a plan that is not in `ml/plans/`.

Read first, in this order: root `CLAUDE.md` §0–0c (links, plain English),
`ml/CLAUDE.md` §0b–0g and §3b (who plans, who implements; run numbers never
alone; replication rule),
[`ml/OVERVIEW.md`](https://blauewelt.github.io/earth/docs.html?f=ml/OVERVIEW.md)
(the two-minute map, with its `Last updated` stamp), and the project docs
`claude/github-access.md` and `claude/huggingface-access.md` for the two
credentials (bootstrapped into `/home/claude/.gh_pat` and
`/home/claude/.hf_token` — never argv).

---

## 1 · Where the programme stands (the science, one paragraph each)

**The reset.** Every rolled number before commit c25f6ff was contaminated
(a dense window loss teacher-forced on held-out bins); the frozen protocol
is train ≤ 2020, test 2021–2024, plus the 2008–09 / 2016–17 development
blocks, longitude holes, a null ladder on every number, rolled skill as
the verdict. [Protocol reset](https://blauewelt.github.io/earth/docs.html?f=ml/plans/PROTOCOL_RESET.md) ·
[Reboot plan](https://blauewelt.github.io/earth/docs.html?f=ml/plans/REBOOT_PLAN.md).

**What is measured cleanly.** The width ladder of stage-2 heads is flat
(7.6 M / 40.4 M / 206.66 M → corridor acc 0.103 / 0.104 / 0.105; the MSSS
spread is amplitude calibration) — E-062-R0. A 200-mode linear inverse model
beats every learned head from 15 days out and is the reference model —
E-066. Held-out minima arrive inside 2,000 steps. The programme is
data-limited. [E-062](https://blauewelt.github.io/earth/docs.html?f=ml/EXPERIMENTS.md#e-062) ·
[E-066](https://blauewelt.github.io/earth/docs.html?f=ml/EXPERIMENTS.md#e-066).

**The cone codec (E-069).** Built, ported to JAX/TPU with parity gates,
run at five seeds under two objectives: **H1 refuted** — the 30-day cone of
dots does not put the anchor's own velocity into the embedding beyond
geostrophy-from-the-SSH-patch (the snapshot twin sits at the raw-patch bar,
the cone at or below it); a CPU synthetic advection test finds no
displacement primitive. What the history buys: persistent fields
reconstructed far better (hidden SSH ~0.36 of bar vs 0.75–0.90) and
one-to-two-pentad forecasts 7–8 % better. Recommendation on record: no
larger cone codec on H1; H2/H3 (stage-2 rollout from cone z vs snapshot z)
is the instrument that matters.
[E-069](https://blauewelt.github.io/earth/docs.html?f=ml/EXPERIMENTS.md#e-069).

**The global tensor (E-070).** Family 7 — 0.25° pentads, 681 × 1,440,
GLORYS12 binned at fetch — Phase A (the 384-chunk pull) is COMPLETE on the
Hub; Phases B–F (the other channels, the build, the codec, the gates) are
not started. [E-070](https://blauewelt.github.io/earth/docs.html?f=ml/plans/E070_global_tensor.md).

**Fleet.** Idle at writing; the Ontario 4090 box is stopped. Check the
status page and `claude/fleet-check-*.md` (hourly) before renting anything.

---

## 2 · What the 3–4 Sep session produced

All in the representations deck
(`ml/figures/geofm_survey/`, 55 slides after this session's additions, [README](https://blauewelt.github.io/earth/docs.html?f=ml/figures/geofm_survey/README.md))
and its companion page, plus three plans and one app feature:

- **Slide 45 — what feeds the cone codec today.** Four upstream products
  (GLORYS12 → 5 channels, OISST → 1, Roemmich–Gilson Argo → 32, NCEP R1
  wind stress → 4, plus the RAPID / Florida-cable labels), the derivation
  graph to the 42 channels, and eight ranked ocean-and-biosphere additions.
  Full inventory with file:line refs in
  [CONE_DATA_AND_ENSO.md](https://blauewelt.github.io/earth/docs.html?f=ml/figures/geofm_survey/CONE_DATA_AND_ENSO.md).
- **Slide 46 — boundaries.** Measured: 22.0 % of anchors have off-grid
  dots, 2.68 % of dots off-grid, 8.17 % on land, 10.85 % carry no value; the
  holdout shadow costs 0.9–1.1 % of bins; 411 of 3,142 pentads straddle a
  calendar month and 106 of them have one day in the month whose
  climatology they get. Decisions: global grid, harmonic climatology.
- **Slide 47 — the speeds.** The cone implemented one ocean speed
  (0.3 m/s); the Argo channels had zero spatial reach. Researched maxima
  (Somali Current 3.6 m/s peak, 2.79 in our own bake; Kelvin 2.8;
  barotropic 200; jet stream 100) → design speed 3.6 × 1.5 = 5.4 m/s for
  the ocean, "global at once" for the atmosphere, log-radial dots.
- **Slide 44 — two stencils, one cone, re-cut.** The E-069 record and the
  v2 geometry: the split between codec and stage 2 moves from distance to
  time.
- **Slide 48 — land, ice and air.** The shared-channel principle: a
  channel is a quantity and an instrument, never a sphere; thirteen
  quantities cross the coastline; Antarctica to −90°; per-location tokens;
  Phase L0. **Correction of 4 Sep (Chris):** observed SST and LST stay
  separate channels; ERA5 skin temperature is the shared one.
- **The El Niño 2026 slide (the last content slide before Sources).** Our OISST bake gives Niño-3.4 +2.09 °C in
  July 2026 (CPC: +1.4 on ERSSTv5 — the gap is unexplained and is itself a
  reason to carry the official index); catalog audit (274 records, ~30 bear
  on ENSO, the `amoc` flag mis-sorts); ranked additions (warm-water volume,
  GODAS/ORAS5, NOAA OLR, the index series, daily winds, TAO dots, DUACS,
  the IRI plume as the benchmark, a `nino.json` bake).
- **[E-071 · cone v2](https://blauewelt.github.io/earth/docs.html?f=ml/plans/E071_cone_v2.md)**
  — Chris's four decisions as a spec: global sampler (§1), day-of-year
  harmonic climatology (§2), Argo profile tokens at the live lags (§3),
  reach from the fastest mechanism × 1.5 for every family (§4, with the
  two-stencils adaptation in §4.5), what E-069 already said (§5), land, ice,
  air and the shared channels (§6). Nothing implemented.
- **[E-072 · the Earth foundation model](https://blauewelt.github.io/earth/docs.html?f=ml/plans/E072_earth_foundation_model.md)**
  — the data-design study (a token is a place at a time carrying every
  channel; the ladder of data in observed values: 2.5 B → 20 B → 91 B →
  455 B → 350,000 B; the independence correction), the scaling ladder L0–L5
  with compute (a 1 B model on the global pentad tensor is ~30 4090-hours;
  the kilometre rung is the first compute-bound one), the recipe
  (Perceiver over location tokens + queryable Gaussian decoder, sphere-routed
  MoE at L3, masked reconstruction + forecast NLL, AdamW β₂ 0.95, warm-up
  2 k, cosine, μP LR transfer measured at L2, early stopping, ≤ 4 epochs,
  z-noise 0.7 as denoising), the null ladder, and the read-outs (El Niño,
  AMOC, colour, sea ice, land water, atmosphere) with each one's data
  prerequisite.
- **The Cones tab's Data mode** on the live app: `ml/export_cone_sample.py`
  calls the real `ConeSampler.sample` and `cone.outer_spiral`, computed the
  trainer's anomaly transform bit-identically as a stream, exported five
  anchors × 24 pentads (2015-01 → 2015-04) to the Hub
  (`chfrank/earth-tensors/cone_samples/`, CORS measured), and the tab
  colours the dots by value, drives the anchor date from the Play tab, and
  reads out every dot with its own date. [How to regenerate](https://blauewelt.github.io/earth/docs.html?f=docs/CONE_DATA_DEMO.md).

---

## 3 · Where to fork — one area per session

Each item names the question, the entry document, and the first concrete
step. Pick one; do not pick two.

**A · Implement cone v2 on the North Atlantic (E-071 §1–4, rung L1).**
Entry: E-071 and `ml/cone.py`, `ml/cone_sampler.py`, `ml/trainprobe.py`.
First step: the harmonic climatology in `anomaly_transform` with its
sawtooth check (E-071 §2 "Verification"), then the sphere-destination dot
placement and longitude wrap in the sampler with the coverage identities
re-asserted, then the profile tokens. Measure against E-069's geometry on
the same tensor, three seeds, per-family loss and rolled skill vs the LIM.
Cheap: < 1 4090-hour per seed.

**B · Family 7 Phases B–F and rung L2 (E-070).** Entry: E-070 §6 phases,
`ml/fetch_glorys_daily.py`, `ml/build_family4.py`. First step: Phase B (the
rg sidecar and NCEP global files) on the runner; then the global build with
the NA sub-block certified equal to family 4's; then a 7 M codec on the
globe with gates G1–G3 and the width ladder re-measured (E-072 §3).

**C · Phase L0 of the four spheres (E-071 §6.5) and the first new heads.**
Entry: E-071 §6, DATA_LADDER §7 (ERA5 needs the free CDS account — the one
blocker). First step: six ERA5 channels + frozen fraction + CCI soil
moisture at 0.25° pentads, ~40 GB, global to −90°; then sea-ice and
land-water heads on the frozen representation (E-072 §5).

**D · El Niño (CONE_DATA_AND_ENSO §4, E-072 §5).** Entry: the companion
page §4 and `scripts/refresh_data.py::oisst_monthly`. First step: a
`nino.json` bake beside `eei.json` reconciled with CPC's ONI (the +2.09 vs
+1.4 gap); then the warm-water volume and GODAS pulls; then an ENSO head on
family 7's z scored on the 2026 peak against the IRI plume.

**E · The velocity question (E-069's H1, E-072 §4.5).** Entry: E-069/E-069b
in the log, `tests/test_cone_advection.py`. First step: the synthetic
advection field at ~50 k GPU steps with un-standardised tracers (ceiling
~0.37), then an explicit displacement head or relative-position bias — a
modelling arm, never a scale arm.

**F · The foundation-model recipe study (E-072 §3–4).** Entry: E-072,
SCALING.md, E-033. First step: at rung L1/L2, the LR sweep and the μP
transfer test, the MoE-by-sphere prototype on the NA tensor with sphere
codes faked from the land mask, and the null ladder wired into the
trainer's metrics so every later rung reports it for free.

**G · The deck and the paper.** Entry: `ml/figures/geofm_survey/README.md`,
`ml/paper/paper.tex` (v8). First step: fold E-071/E-072 into the four-sphere
slides (34–46) and add the three foundation-model slides that follow slide 48; the paper
takes nothing from these until a rung has a level.

---

## 4 · Standing constraints every fork inherits

- Deploy-first and stamp for the app (root CLAUDE.md §1); dispatch rules
  for the fleet (ml/CLAUDE.md §1–2); never force-push; the git proxy refuses
  plain pushes — `node scripts/git_api_push.mjs --branch main --token-file
  /home/claude/.gh_pat`, then fast-forward gh-pages with `force:false`.
- Every run number carries its summary; every document is registered in
  `docs.html`'s `DOCS` (the docs test fails otherwise); every acronym is
  spelled out once.
- Replication per ml/CLAUDE.md §3b: the first result at a new tier buys its
  pair; probe-scored claims need three seeds.
- Budget is Chris's constraint, not a session's (ml/CLAUDE.md §0e): report
  the arithmetic, never park the fleet.
- The Projects tool attaches only to a session started from the Earth
  project; a session started without it cannot reach the credential docs
  (measured 3 Sep — cost an hour).

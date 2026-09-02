# What should a *generic* Earth embedding see? — input ladder, dependency cones, and a cone-native codec

Earth 2 research note · 2 September 2026 · status: **proposal** (design + predictions; nothing here has been trained)

Companion to the survey deck (`geospatial-representation-models.pptx`, slides 34–45) and to the dependency-cone note of 16 August. Dataset facts below were re-verified against the agencies' product pages on 2 September 2026; the spec sheet is in Appendix A. Physical speeds and memories are order-of-magnitude values, the same ones the cone slides use, and are marked as such.

---

## 0. Summary

1. **The rule for inputs.** A generic embedding should ingest every signal that is (a) relevant to the state of the four spheres at a point and (b) *not derivable* from the other inputs. Observations are irreducible and always in. Derived products (reanalyses, gap-filled "L4" grids, model outputs) are allowed when they carry information from sources we do not ingest, but they are **flagged as derived, masked at a higher rate, and time-shifted by their assimilation window** so the model cannot lean on them or peek through them.
2. **The input ladder.** Thirteen rungs from 10 m optical/radar imagery up to 300 km gravimetry and static context (Section 2). Each rung is justified by what it adds that nothing below it contains. Rung 3 (300 m–1 km daily radiometers) is the first that covers the ocean; rung 8 (Argo, moorings, drifters, GLORYS) is the first that sees below the surface.
3. **Cones per input.** Every rung falls into one of five cone families — fast-wide-short (atmosphere), slow-narrow-long (ocean interior, sea level), L-shaped unions (sea-surface temperature and salinity), column-only (land, no horizontal transport, forced from above) and static (Section 3). This decides the *shape* of the context each channel needs, and it is very different across rungs: 10 m imagery wants a small, dense local patch with a long local history; a 25 km wind field wants a 4,000 km, five-day reach and no history at all.
4. **The proposal: a cone-native codec — "dots in, embedding out".** Instead of embedding one pixel-month and letting a stage-2 stencil connect embeddings, the codec itself reads a *set of dots* — (Δx, Δy, depth, lag, channel, value) samples drawn from each channel's cone around an anchor — through Perceiver-style cross-attention, and is trained to predict held-out dots, including dots at future lags (Section 4). Two dots of the same field at different lags contain displacement; displacement over lag is velocity. That is how momentum features (currents, drift, tendencies) become part of the embedding rather than something stage 2 has to reconstruct from lossy codes.
5. **Is it zero-sum?** No — but it is asymmetric (Section 5). A cone codec can always emulate a per-pixel codec (ignore the context), but a per-pixel codec cannot recover what its bottleneck discarded before the dynamics were visible. The real trade is compute and reusability against dynamics and sample efficiency. The recommended design is hierarchical: a cheap **local codec** (per tile-date, cached, snapshot semantics — good for mapping) feeding a **cone codec** (per anchor, dynamic semantics — good for prediction), with a thin stage 2 left for teleconnections, aggregation across anchors, and task heads. The split is made by physics (what the cone decides) versus task (what the label decides), not by fiat.
6. **A decisive experiment already fits the current data** (Section 6, Phase 0): the same 24 months × sunflower-89 tokens the stage-2 pool reads, once as "codec + stage-2 attention" and once as "cone codec + linear probe", scored under the corrected window-scope protocol of the 2 September paper reset (rolled MSSS per lead against climatology and damped persistence, the LIM null beside every number, n ≥ 3 seeds) with a velocity probe as a diagnostic. If the cone codec matches with a linear head what stage 2 needs attention for, the "work in the embedding" side wins on sample efficiency; if the velocity probe only works for the cone codec, the momentum argument is confirmed. Note that nothing in the existing record decides the question: the 3 × 3 codec-vs-raw probe contrast (0.672 vs 0.659) is inside the single-seed probe noise band, and every rolled-skill number from before 2 September was withdrawn as contaminated.

---

## 1. The rule: everything relevant that cannot be derived

**Observed vs derived.** An *observation* is a number a sensor produced (a reflectance, a radar echo, a radar range to the sea surface, a thermometer reading on a float). A *derived* product is a number a program produced from observations plus assumptions — a gap-filled daily map, a reanalysis (a weather or ocean model nudged towards observations), a machine-learned "best estimate". Both are useful; they are not the same kind of input.

**Three tests for a derived product to be admitted:**

- *It carries information we do not otherwise ingest.* ERA5 assimilates radiosondes, aircraft, and satellite radiances that are impractical to feed raw; GLORYS assimilates altimeter tracks, SST and in-situ profiles with a full ocean model in between. Admitted, flagged `derived`.
- *It is a useful target even if a poor input.* Vegetation indices, geostrophic currents from sea-level slope, surface-water class maps: all functions of inputs already on the ladder. They are **not** inputs; they are optional prediction targets (masked-channel losses on them cost nothing and give free supervision).
- *It is not a peek into the future.* Delayed-time L4 grids and reanalyses use observations from *after* the nominal time (the delayed-time DUACS sea-level grids use a centred ±6-week window of altimeter tracks — the NRT stream uses only the past 7 weeks; OSTIA and MUR use a window of recent days; ERA5 uses 12-hour 4D-Var windows; GLORYS a 7-day assimilation cycle). For a codec trained to predict the present or the future from the past, a derived channel at lag ℓ is really information at lag ℓ − (half the window). Rule: shift the *effective time* of derived channels by their window, or use the NRT streams (which only see the past) for training the forecasting part of the objective.

**What we drop as derivable** (and keep only as targets): NDVI/EVI and other band ratios (from rungs 1–3); solar geometry, day length, Coriolis parameter (from position and time); slope, aspect, distance-to-coast (from rung 12); geostrophic currents and OSCAR (from SSH, wind, SST); Dynamic World and similar per-scene classifiers (from Sentinel-2); wave Stokes drift (from the wave spectrum); ERA5-Land's derived fluxes (a land model run on ERA5 forcing with no assimilation — kept only as a weak prior at high mask rate).

**Why the flag matters for masked-channel training.** If the codec learns that "GLORYS u at lag 0" is always present and always right, it copies it, and the embedding learns no flow physics of its own. The remedy is the same one the current Earth 2 codec uses for never-measured tokens: every dot carries a *source token* (observed swath / observed point / L4 gap-filled / reanalysis / model), and derived sources are dropped from the input at a higher rate (say 50 % of anchors) so that the model must be able to produce them from observations alone.

---

## 2. The input ladder — fine-grained first, adding rungs until holistic

Ordering is by native pixel size (finest first); cadence is given because the two together define "granularity". "Adds" is the test each rung must pass: something no lower rung contains. Numbers are the verified 2026 specs (Appendix A). O = observation, D = derived; ~ = long record but retiring/changing in 2026–27 (Section 7).

| Rung | Input (native resolution · cadence) | What it contains, in plain English | O/D | Adds (what nothing below it has) |
|---|---|---|---|---|
| 0 | **Anchor coordinates**: latitude, longitude, time of year, time of day, depth | Where and when we are looking. Not data, but the model needs it to know that "15 °C" means something different in January at 60° N than in July at the equator. | — | The frame of reference; everything else is relative to it. |
| 1 | **10 m optical + radar**: Sentinel-2 L2A (13 bands, 10/20/60 m, 5 days with two satellites, three in orbit until end-2026); Sentinel-1 GRD (VV/VH backscatter, 10 m pixels, 6-day repeat; constellation now S1C + S1D) | Sentinel-2 is a colour-plus-infrared photograph of the sunlit surface, corrected for the atmosphere: crops, forests, bare soil, water, snow, buildings. Sentinel-1 is a radar image, taken day or night through cloud, of how *rough* and how *wet* the surface is: flooded fields, sea-ice edges, soil moisture, ships. | O | Field-scale texture and the land biosphere's fine state; radar sees through cloud and at night. Land and coast only (radar over open ocean only in selected modes). |
| 2 | **30 m optical + thermal**: Landsat 8/9 Collection 2 (surface reflectance + surface temperature, 30 m, 8-day combined; record 1982–) | The same kind of photograph as rung 1 plus a *thermal* image — how warm the ground is — at field scale, and a forty-year archive. | O | Thermal at field scale; the long record for phenology and change. |
| 3 | **300 m–1 km daily radiometers**: Sentinel-3 OLCI (21 bands, 300 m, < 2 days) and SLSTR (500 m/1 km, SST and LST, daily); VIIRS on NOAA-20/21 (375/750 m daily; MODIS ~ retiring late 2026/27) | Daily whole-planet pictures in visible, infrared and thermal. Over the ocean, the colour of the water tells you how much phytoplankton is in it (chlorophyll), the thermal channels give the skin temperature of sea and land, and hot spots are fires. | O | **First rung that covers the ocean surface** and gives daily cadence; ocean colour (the biosphere), fires, snow, daily land temperature. |
| 4 | **Geostationary imagers**: GOES-19/18 ABI, Meteosat-12 FCI, Himawari-9 AHI (16 channels, 0.5–2 km, every 10 min; full disks ≈ ± 60–70° latitude) | A continuous *movie* of clouds, water vapour and surface radiance from fixed points in the sky. Clouds are visible moving, forming and dissipating between frames. | O | Atmospheric *motion* at minute scale — the only direct view of the wind field's effect on clouds and of the diurnal cycle; the fast end of the cone. |
| 5 | **Kilometre-scale specialty sensors (sparse tracks/swaths)**: SWOT KaRIn SSH (2 km swath, 21-day repeat); nadir altimetry (Sentinel-6 MF/6B, Jason-3, Sentinel-3: ~7 km along-track, ~10-day repeat); Sentinel-1 OCN winds (1 km on swaths); ground weather radar (MRMS 1 km/2 min CONUS; OPERA CIRRUS 1 km/5 min Europe); GLM lightning (8 km, continuous); lidar (GEDI 25 m footprints ± 51.6°; ICESat-2 11 m footprints, canopy and sea-ice freeboard); TROPOMI (NO₂, CO, CH₄ at 3.5–7 km daily); OCO-2/3 (CO₂ columns, ~3 km² soundings, sparse) | Sea-surface *height* to a few centimetres along the satellite's track (the ocean's "pressure map" — slopes of the surface drive currents); rain seen from the ground every few minutes; where lightning strikes; the height of trees and of sea ice; the trace gases in the air column. | O | Sea level and hence currents; precipitation at storm scale; 3-D structure of vegetation and ice; greenhouse-gas columns. All are *sparse in space or time* — exactly the case where a point-set (dot) input beats a gridded one. |
| 6 | **5–25 km gap-filled surface fields (daily to hourly)**: OSTIA SST (5 km daily; MUR 1 km), DUACS sea-level L4 (0.125° daily, SLA/ADT + geostrophic u, v), IMERG precipitation (0.1°, 30 min), ASCAT winds (12.5/25 km, ≈ twice daily), AMSR2 sea-ice concentration (10 km / 6.25 km daily), OSI SAF ice drift (62.5 km, 2-day), SMAP soil moisture L3 (36 km daily) and L4 (9 km 3-hourly, model-assimilated), SSS L4 (0.125–0.25°, weekly), Copernicus waves (1/12°, 3-hourly, model) | The "cleaned-up" daily maps: sea temperature with the clouds filled in, sea level everywhere (not just under the satellite track), rain everywhere, wind over the whole ocean, how much of the sea is ice-covered and how fast it drifts, how wet the top few centimetres of soil are. | D (from O of rungs 3, 5) — except ASCAT/SMAP L3/AMSR2, which are swath observations on a grid | Convenience and completeness; the useful *initial state* for the slow media. Keep because they merge sources we do not ingest raw (AVHRR, microwave SST, all altimeters); mask heavily; time-shift by their windows. |
| 7 | **25–31 km hourly atmosphere (the reanalysis rung)**: ERA5 (31 km native, 0.25° grid, hourly, 1940–, ~5-day latency; 2 m temperature, 10 m wind, sea-level pressure, precipitation, surface solar/thermal radiation, cloud cover, boundary-layer height, 2 m dewpoint, plus 37 pressure levels); for real time: ECMWF IFS/AIFS open data (0.25°, 6-hourly) or GFS (0.25°, 6-hourly) | The best physically consistent *estimate* of the whole atmosphere at every hour: temperature, wind, pressure, humidity, radiation, clouds, at the surface and aloft. Made by running a weather model and continuously nudging it towards millions of observations. | D (carries radiosonde, aircraft, radiance information) | The forcing of everything else: heat and momentum fluxes into ocean and land, the vertical structure of the atmosphere. Note: ERA5's SST and sea ice are *inputs* to ERA5, not outputs — never use them as ocean observations. |
| 8 | **Ocean interior**: Argo core (T, S profiles 0–2000 m every 10 days; 4,372 operational floats, ~150 k profiles/yr), Deep Argo (to 4–6 km; 219 floats), BGC-Argo (O₂, NO₃, pH, chlorophyll, backscatter, irradiance; 989 floats), moorings (GTMBA/TAO, OceanSITES), surface drifters (1,317; velocity at 15 m + SST), GLORYS12 reanalysis (1/12° ≈ 8 km, 50 levels, daily; T, S, u, v, SSH, mixed-layer depth, sea ice; 1993–) and the Copernicus 1/12° analysis/forecast for real time | What the ocean is doing *below* the surface: how warm and salty it is at each depth (heat content, density, the mixed layer), how fast and in which direction it flows, where fresh water and heat are stored. The floats are sparse points; the reanalysis is a model's best gridded estimate around them. | O (floats, moorings, drifters) + D (GLORYS) | **First rung below the surface.** Heat capacity, density gradients (thermal wind), the AMOC — the slow memory of the system. Only dots (native profiles) do this justice; gridding Argo throws away its depth resolution. |
| 9 | **Ocean biosphere and carbon**: ocean colour L3 (OC-CCI v6 / GlobColour, 4 km daily, observed, cloud gaps; L4 gap-filled), PACE OCI (hyperspectral 1.2 km, 2-day, 2024–), BGC-Argo (rung 8), SOCAT surface CO₂ (44 M point observations), Copernicus surface carbon (0.25° monthly, ML reconstruction), SeaFlux (1° monthly) | How much plant life is in the surface ocean and of what kind, how much CO₂ the water holds relative to the air (whether the ocean is absorbing or releasing carbon), oxygen and nutrients at depth. | O (colour L3, BGC-Argo, SOCAT) + D (L4, carbon maps) | The ocean carbon sink and its drivers. |
| 10 | **Land biosphere and land water**: LAI/FAPAR (Copernicus 300 m 10-day; MODIS 500 m 8-day), NDVI/EVI (250 m 16-day — target only), SIF (TROPOMI ~7 km daily), LST (1 km daily), evapotranspiration (MOD16 500 m 8-day; GLEAM 0.1° daily — models), soil moisture (rung 6; ASCAT 12.5 km, ~2 h latency), snow cover (MODIS 500 m daily; IMS 1 km NH), fire (VIIRS 375 m, FIRMS < 3 h; burned area 500 m monthly), terrestrial water storage (GRACE-FO, ~300 km monthly), river discharge (GRDC points; GloFAS 0.05° daily model), lake/river heights (Hydroweb altimetry), flux towers (FLUXNET/ICOS/AmeriFlux: half-hourly CO₂, water and heat exchange at ~500 sites) | How much leaf area there is and how actively it photosynthesises (fluorescence is the plants' own glow), how much water evaporates, how much snow and soil water is stored, where it burns, how much water sits in rivers, lakes and aquifers. | O (LAI/FAPAR retrievals, SIF, LST, snow, fire, GRACE, gauges, towers) + D (ET models, GloFAS) | The land carbon and water cycles: the fourth sphere. Almost everything here is *column-only* (it does not move sideways) — see Section 3. |
| 11 | **In-situ atmosphere (points)**: surface stations (ISD > 14,000 active, hourly; GHCN-Daily > 100,000), radiosondes (IGRA, ~800 stations in near-real time, 00/12 UTC), aircraft (AMDAR), ships and buoys (ICOADS, NDBC), tide gauges (GLOSS) | Thermometers, barometers and anemometers on the ground, balloons through the atmosphere, instruments on ships and buoys, sea-level gauges on coasts — the measurements the reanalysis of rung 7 is fitted to. | O | Irreducible ground truth. Mostly redundant with rung 7 *if* rung 7 is present; essential when it is masked out (the model must be able to rebuild the atmospheric state from points). |
| 12 | **Static and slowly varying context**: elevation (Copernicus DEM 30 m), bathymetry (GEBCO 15″ ≈ 450 m), land cover (WorldCover 10 m 2021; CCI 300 m annual), soil properties (SoilGrids 250 m), canopy height (ETH 10 m; Meta/WRI 1 m), glacier outlines (RGI 7), population (WorldPop/GHSL 100 m), night lights (VIIRS DNB 500 m monthly), roads (OSM) | The stage on which everything happens: how high, how deep, what kind of surface, what kind of soil, how tall the trees, where the people are. | O/D (compiled maps) | Boundary conditions: depth sets what currents can do, terrain sets rain and rivers, soil sets what water does. One dot each; v = 0, τ = ∞. |
| 13 | **Atmospheric composition (optional in phase 1)**: MAIAC aerosol optical depth (1 km daily), MERRA-2 aerosols (0.5°, hourly, 3-week latency), TROPOMI/OCO/GOSAT columns (rung 5) | Dust, smoke and pollution in the air and the greenhouse-gas columns above each point. | O + D | Radiative forcing detail and the carbon-cycle link between land, ocean and air. |

**Reading the ladder.** Rungs 1–2 are the *land* fine scale; rung 3 is where the ocean surface first appears; rung 4 is the atmosphere's fast scale; rungs 5–7 are the physical fields; rung 8 is the ocean's memory; rungs 9–10 are the two biospheres; rung 11 is the ground truth; rung 12 the boundary conditions. "Holistic" is reached at rung 12 for the four-sphere goal; rung 13 is the extension towards composition and radiative forcing.

**Per-sphere minimal set** (if one had to start with one rung per sphere): atmosphere — rung 7 (+ 4 later); ocean physics — rungs 6 + 8; ocean biosphere — rung 9 (colour L3 + BGC-Argo); land biosphere — rung 1 + 10. That is roughly the sequencing already agreed (ocean interior → ocean colour → atmosphere as forcing → land).

---

## 3. The dependency cone, applied to the ladder

Recap of the formalism (deck slides 26–31): for a process p with propagation speed v_p and memory τ_p, the information that can reach an anchor from lag ℓ (looking back ℓ from the anchor time, with the codec's own step Δt) lies within Δx ≤ v_p (Δt + ℓ), and only lags with Δt + ℓ ≤ τ_p are worth reading. A channel that is moved by several processes gets the union of their cones. Reach is floored by a correlation length L_corr (a field can be coherent over 100 km without anything moving) and capped at 10,000 km. Fast drivers acting on a slow medium add: L_corr + v_slow (Δt + ℓ).

### 3.1 Five cone families

| Family | Carrier (what moves the information) | v (order of magnitude) | τ (memory at a fixed point) | Reach after 1 day / 1 week / 1 month | Shape | Ladder rungs |
|---|---|---|---|---|---|---|
| **A. Fast, wide, short** | wind, synoptic weather | 10 m/s ≈ 860 km/day | 3–10 days (precipitation: hours–1 day) | 860 km / 6,000 km / capped | Flat, wide disc; almost no history | 4 (geo imagers), 7 (ERA5 wind, pressure, humidity, clouds), 5 (radar rain, lightning), 6 (IMERG, ASCAT) |
| **B. Slow, narrow, long** | ocean currents, eddies; Rossby waves for sea level | currents 0.1–0.2 m/s ≈ 10–15 km/day; Rossby 0.03 m/s ≈ 2.6 km/day (westward); Kelvin/coastal 2.5 m/s along boundaries | eddies/SSH 2–6 months at a point; interior T/S 1–10 yr (thermocline), decades (deep) | 13 km / 90 km / 400 km (currents); 3 / 18 / 80 km (Rossby) | Thin, tall column, tilted upstream (and westward for SSH); one long arm along coasts | 5 (altimetry, SWOT), 6 (DUACS SSH, SSS), 8 (Argo, GLORYS, drifters) |
| **C. L-shaped union** | fast atmospheric forcing on a slow medium | atmosphere A for the forcing arm; ocean B for the memory arm | SST/SSS 3–6 months (mixed-layer thermal inertia; Frankignoul & Hasselmann 1977); sea ice: seasonal + multi-year thickness memory | forcing arm 860 km at lag 0; memory arm 100 km + 13 km/day | Wide at short lags, narrow at long lags — the "L" | 3 (SST, LST over water), 6 (OSTIA, MUR, sea ice conc.), 9 (surface chlorophyll: advected by B, forced by light/wind/mixing from A, grows on 1–4-week timescales) |
| **D. Column-only (land)** | nothing horizontal (soil, plants, snow stay put); forced from above by A; rivers move water downstream at ~1 m/s along the network | 0 horizontally (except river network ≈ 90 km/day along channels; sea-ice drift ≈ 5–10 km/day) | soil moisture 1–3 weeks (surface), months (root zone); vegetation weeks–seasons; snow seasonal; water storage months–years; fire days | L_corr (≈ 1–10 km: terrain, fields) at all lags; plus the atmosphere disc at lags ≤ τ_atm | A narrow column with a long history, wearing a wide flat hat (the atmosphere) at the top | 1, 2 (imagery over land), 10 (land biosphere & water), part of 12 |
| **E. Static** | none | 0 | ∞ | one dot | a point | 12 (DEM, bathymetry, soil, land cover, canopy height, population) |

### 3.2 The depth axis

Vertical velocities in the ocean are tiny (10⁻⁶–10⁻⁵ m/s, i.e. 0.1–1 m/day), but the *mixed layer* (the top 10–100 m, seasonally to 300 m and more at high latitudes in winter) is stirred by wind and cooling within hours to days, so surface and mixed layer belong to one cone. Below the mixed layer the water column is stratified and memory grows with depth: months at 100–300 m, years at 500–1,500 m, decades below 2,000 m. The vertical cone is therefore: *reach in depth ≈ mixed-layer depth + w·(Δt + ℓ)* — essentially "the mixed layer, plus a slow deepening" — while the *history* the model should read grows with depth. Depth dots should be log-spaced (0, 10, 30, 60, 100, 200, 400, 700, 1,000, 1,500, 2,000 m; Deep Argo below), dense near the surface, and each depth band gets its own τ. For the atmosphere the same logic applies in reverse with much faster mixing: the boundary layer (0.5–2 km) couples to the surface within hours; free-troposphere levels carry the synoptic signal (family A) — three or four pressure levels (1000, 850, 500, 250 hPa) are the atmospheric "depth dots".

### 3.3 What this means for sampling

The ladder's rungs want opposite kinds of context:

- **Fine rungs (1–3, 10) want a small, dense local patch and a long local history.** A 10 m field does not move; what predicts its state is its own past seasons and its immediate neighbourhood (terrain, the field next door, the river 2 km away). Reach ≈ L_corr ≈ 1–10 km; history up to a few years at coarse cadence (monthly composites) with the last few weeks at full cadence.
- **Coarse physical rungs (4–8) want a large, sparse far field with short (A) or long but thin (B) history.** A wind dot 800 km upstream from yesterday is worth more than a wind dot 10 km away from last month.
- **The ocean interior (8) wants depth and years, not area.** Rung 8 alone justifies 24–48-month histories.

A single fixed grid × fixed window cannot serve all three; a per-channel dot sampler with the cone as its support can. The slot arithmetic of the cone slides (clamp(round(89·(r/4444)²), 9, 89) slots per lag, 851 vs 12,816 for the worked example) generalises directly: budgets per channel come from the family, and the total per anchor for a 15–20-channel-group ocean anchor lands around 2,000–4,000 dots — the same order as a 224 × 224 image at 16-pixel patches (196 tokens) times a modest temporal stack, and far below the 12,816 of the naive cylinder.

---

## 4. Proposal: a cone-native codec — dots in, embedding out

### 4.1 Anchor

An anchor is a point (x, y, z, t): a location, a time, and optionally a depth — *not* a pixel of a particular product. The embedding describes the state of the four spheres at that point *together with the dynamic context that determines it*. Anchor cadence and spacing are choices of the training sampler (ocean: 5–25 km, pentad or month; land: 100 m–1 km, dekad; interior: the profile positions themselves), not of the architecture.

### 4.2 Dot sampling

For each channel group c (≈ 15–25 groups, one per row family of the ladder), draw N_c dots from the cone C_c(anchor) = {(Δx, Δy, z, ℓ) : |Δx| ≤ reach_c(ℓ), z ∈ Z_c, ℓ ≤ τ_c}:

- **Gridded channels**: stratified log-polar sampling in space (denser near the anchor) × log-spaced lags (denser near the present), so every (radius bin, lag bin) cell holds at least one dot; per-lag slot counts from the family's budget. Directional skew (upstream, westward for SSH, along-coast) enters as a sampling prior, not a hard mask.
- **Point / profile / track channels** (Argo, drifters, stations, altimeter tracks, SWOT swaths, lidar footprints, flux towers): the dots *are the observations* that fall inside the cone. No gridding, no interpolation, no invented values. This is the single largest practical gain of the dot formulation: the sparse rungs (5, 8, 9, 11) enter natively.
- **Fine imagery** (rungs 1–3): not raw pixels — a *local codec* (Section 4.7) turns each tile-date into a few tokens, and those tokens are the dots for the local patch and its history.
- **Static** (rung 12): one dot per layer at the anchor (plus a handful in the local patch for terrain context).

Missing dots are simply absent (a set input has no holes to fill), which retires the "never-measured vs masked" distinction on the input side; it survives on the *target* side (Section 4.5).

### 4.3 Tokens

Each dot becomes one token = value embedding + coordinate encoding + channel embedding + source flag:

- **value**: per-channel linear map of the (standardised) value, or a small MLP for multi-valued dots (a 13-band reflectance, a T/S pair, a wind vector). Quantisation into bins with learned embeddings is an alternative that makes distributional prediction natural.
- **coordinates**: Fourier features of (Δx, Δy) in km on a log scale (the cone spans 1–10,000 km), of z in metres (log), of ℓ in days (log), plus absolute (lat, lon, day-of-year, hour) for the anchor only. Relative coordinates make the encoder translation-equivariant; the log scale makes 10 m and 1,000 km live in the same code.
- **channel**: learned embedding per channel group (with sub-channel for bands/levels).
- **source flag**: observed-swath / observed-point / L4-gap-filled / reanalysis / model / local-codec-token — Section 1.

### 4.4 Encoder

A Perceiver-IO-style encoder: K learned latents (K = 128–256, width 512–768) cross-attend to the dot set (N ≈ 2,000–4,000), followed by 4–8 self-attention blocks over the latents. Cost is O(N·K) per cross-attention, not O(N²): at N = 3,000, K = 256, width 512 one cross-attention is under 1 GFLOP, and the whole encoder a few GFLOPs — cheaper per anchor than a ViT-Base forward on a single 224 × 224 image (≈ 17 GFLOPs). Pool the latents (attention pooling) to the **dynamic embedding** e_dyn ∈ ℝ^D (D = 256, int8-quantised for storage, as AlphaEarth and TESSERA do); keep the latents themselves for the decoder during training.

Why a set encoder and not a 4-D transformer over (x, y, z, t) grids: the input is ragged (a profile here, a swath there, a 10 m tile in the corner) and the cone is not a box; the set formulation handles both without padding.

### 4.5 Objective: masked dots, including the future

The training signal is the existing one — mask and predict — generalised from channels to *sets of dots*. A held-out dot is a query token (coordinates + channel, no value); a light decoder cross-attends from queries to the latents and predicts the value as a distribution (Gaussian head, or quantile/bin head), scored by NLL/CRPS. Masking schemes, mixed per batch:

1. **Channel drop** (the current scheme): remove a whole channel group from the input; predict its dots. Derived sources dropped at a higher rate (Section 1).
2. **Lag-band drop**: remove all dots with ℓ < ℓ₀ (the recent past); predict them from the older past. This *is* forecasting, inside pretraining: the model must move information from upstream-and-earlier to here-and-now.
3. **Future dots**: include query dots at negative lags (ℓ ∈ [−Δt, −τ]) — the future relative to the anchor — with values available in the training archive. This turns the codec into a forecaster without any change of architecture; the embedding of an anchor is then explicitly "the state plus its tendency".
4. **Sector drop**: remove a spatial sector (e.g., everything upstream); predict its dots. Teaches spatial coherence and the direction of transport.
5. **Depth-band drop**: remove all dots below z₀; predict interior from surface (and the reverse: predict surface from interior).
6. **Anchor reconstruction**: always predict the anchor's own channels (so the embedding still *is* the state at the point).

Targets keep the never-measured vs masked distinction: a query at a place-time where nothing was ever observed gets no loss; a query where an observation exists but was hidden gets the loss. Loss weights per channel family follow the AIFS-surface-ocean practice of up-weighting slowly evolving fields so the fast atmosphere does not dominate.

### 4.6 Why velocity becomes representable (and how to make sure it does)

A photograph does not show speed; two photographs a known time apart do. If a field is advected, f(x, t + ℓ) ≈ f(x − vℓ, t), so the value at the anchor is best predicted by the dot at displacement −vℓ at lag ℓ; the cross-correlation between the anchor's neighbourhood at lag 0 and the field at lag ℓ peaks at Δx = vℓ. That is exactly the maximum-cross-correlation method oceanographers use to track features in SST images (Emery et al. 1986), and attention over dots with relative coordinate encodings can implement it: a query attends to the dots whose (Δx, ℓ) pair fits, and the ratio Δx/ℓ that wins *is* a velocity readout. The same holds for tendency (∂f/∂t from two lags) and for divergence/convergence (from dots around the anchor) — the "momentum features" the request asks for.

Three measures to make sure the capacity is used rather than merely available:

- **Supervised flow targets as maskable channels.** Drifter velocities (rung 8), sea-ice drift (rung 6), OSCAR/GLORYS u, v (derived — flagged), ERA5 10 m wind, ASCAT wind, river discharge: all enter as channels, so the masked objective directly asks the embedding to *produce* velocity when the velocity channel is hidden.
- **Multiple lags of the same field in the input by construction** (family B/C budgets always include lags > 0 for the slow media; the cone guarantees the upstream dots exist).
- **An optional advected-sampling prior**: draw a fraction of the dots along candidate trajectories x − v̂ ℓ using a first-guess velocity (geostrophic u, v from SSH; ERA5 wind for the atmosphere). This is PARADIS's semi-Lagrangian gather (arXiv 2601.21151) used as a *sampler* rather than a layer: it makes the right dots more likely to be present without hard-wiring the velocity.

Evaluation of this claim is a *velocity probe* (Section 6): a linear head from the frozen embedding to drifter velocity or GLORYS u, v at the anchor. A snapshot embedding should fail it (R² near the climatological value); the cone codec should not.

### 4.7 The local codec, and the hybrid

Rungs 1–3 are dense and fine; a 3 × 3 km patch of Sentinel-2 at 10 m is 90,000 pixels × 13 bands. They should not be dots. Instead a small **local codec** — the existing per-pixel/3 × 3 MAE codec generalised to a tile (say 64 × 64 px at 10 m, with the STP-style pyramid of AlphaEarth or the FlexiViT tokens of OlmoEarth) — turns each tile-date into a few 64-D tokens that are cached once and reused by every anchor whose cone contains the tile. The local codec keeps *snapshot* semantics (this place, this date), which is what mapping tasks want; the cone codec adds *dynamic* semantics. Both codes are exposed downstream: e_loc (64-D, cheap, static tasks) and e_dyn (256-D, dynamic tasks).

The hierarchy is: raw observations → local codec (per tile-date, cached; masked-pixel objective) → cone codec (per anchor; masked-dot objective over local tokens + coarse dots + point observations) → stage 2 (task; thin). AlphaEarth's STP pyramid does this within one image; here the pyramid runs across sources and time scales.

### 4.8 What stage 2 still does

- **Teleconnections beyond the cone**: ENSO's effect on a mid-latitude anchor arrives through the atmosphere in days (inside cone A) but its *predictability* comes from Pacific ocean state months earlier — 10,000 km and 6 months away, outside any single cone. Stage 2 attention over dynamic embeddings of distant anchors covers this.
- **Aggregation across anchors**: AMOC transport is an integral over a section; a carbon budget is an integral over a basin. Stage 2 pools embeddings along the section or basin.
- **Task heads and labels**: the ladder of probes (ridge → MLP → attention head) stays; with dynamics in the embedding the expectation is that more tasks stop at ridge.

### 4.9 Outputs and reusability

Because the cone only looks backwards (future dots are *targets*, never inputs), an anchor's embedding is final once every source in its cone has arrived. Sources arrive at different latencies (NRT streams in hours to days; ERA5 in ~5 days; delayed-time L4 and reanalyses in weeks to months), so there are two natural products, exactly as ERA5 has ERA5T and ERA5: an **NRT embedding** (observations + NRT streams only) and a **final embedding** (all sources), with the NRT one being the honest one for forecasting evaluation. Both are stored per anchor, quantised, and indexed; nothing needs recomputing when new *future* data arrives.

### 4.10 Compute and storage

Per anchor: local tokens (cached) + ≈ 3,000 dots × cross-attention to 256 latents × 6–8 blocks ≈ 3–5 GFLOPs. A global ocean at 25 km, pentad cadence, is ≈ 6 × 10⁵ anchors × 73 pentads ≈ 4 × 10⁷ anchors/yr ≈ 2 × 10¹⁷ FLOPs — well under an hour at a sustained 100 TFLOP/s. Land at 1 km monthly is 1.5 × 10⁸ × 12 ≈ 2 × 10⁹ anchors/yr ≈ 10¹⁹ FLOPs — about a day at the same rate. The dot *sampler* (random access into many zarr cubes and point archives) is the engineering bottleneck, not the network; a pre-materialised cone index per channel family (which tiles/dates/profiles fall in which anchors' cones) is the piece of infrastructure to build first.

### 4.11 Pitfalls and mitigations

| Pitfall | Why it bites | Mitigation |
|---|---|---|
| Shortcut copying of derived channels (GLORYS u, v; ERA5 wind) | Model learns "copy the reanalysis" and never learns flow from observations | Source flags; derived channels dropped from the input at ≥ 50 % of anchors; velocity probe evaluated with derived channels *masked* |
| Future leakage through L4 / reanalysis windows | delayed-time DUACS uses tracks up to 6 weeks after the nominal day; ERA5 12 h; GLORYS 7 d; OSTIA days | Effective-time shift per source; NRT streams for the forecasting objective; never use delayed-time products at lag 0 |
| Cone mis-specified (too small) | Information physically present but outside the support cannot be learned around | Cones deliberately generous (× 2 on v and τ); the four ablations of the cone slides decide; attention prunes |
| Cone too large | Slots wasted, noise admitted | Slot budgets; ablate down |
| Ragged, heterogeneous input makes batching hard | Dots per anchor vary 10× | Bucket anchors by dot count; pad with masked dots inside a bucket; Perceiver cost is linear in N so padding is cheap |
| Embedding semantics drift from "this place" to "this neighbourhood" | Harder to attribute, harder to compare with per-pixel GeoFMs | Always predict the anchor's own channels (4.5.6); expose e_loc alongside e_dyn |
| Sensor churn 2026–27 (Section 7) | Channels appear/disappear | Channel-drop training makes this a *masked* condition, not a failure; source flags carry the sensor identity |

---

## 5. Is it zero-sum? "Work in the embedding" vs "work in stage 2"

The question: with a fixed budget, do we either (A) keep the embedding fine-grained and local, and make stage 2 do the spatial/temporal connecting, or (B) put the connecting into the embedding — and does one side's gain equal the other's loss?

### 5.1 The two ends

**A — thin codec, thick stage 2.** The codec sees one pixel-month (or a 3 × 3 patch), compresses to 64-D; stage 2 assembles stencils (ring-8 at 222 km, sunflower-89 at 4,444 km, 24 months) of codes and learns the task with attention. This is the current Earth 2 design and the design of every GeoFM in the survey (they embed images; the user connects them).

**B — thick codec, thin stage 2.** The codec sees the cone (Section 4); stage 2 is a linear or shallow head plus the far-field/aggregation part.

### 5.2 Pros and cons

| | A — thin codec, thick stage 2 | B — thick (cone) codec, thin stage 2 |
|---|---|---|
| **Cost per embedding** | Minimal (one pixel-month; 27 tokens for 3 × 3 × 3 months) | 30–100× more input per anchor (≈ 3,000 dots), though still ≈ one ViT-B image forward |
| **Reusability / caching** | Embedding = this place, this time; computed once, never changes | Embedding = this place, this time *and its past context*; also final once its sources have arrived (two versions: NRT and final) |
| **Interpretability** | Clear provenance (this pixel) | Provenance is a set of dots; attention maps over dots recover it, but it is more work |
| **Dynamics (velocity, tendency, transport)** | Not representable from a snapshot; must be reconstructed by stage 2 from *compressed* codes | Representable in the codec (Section 4.6); supervised by flow channels |
| **Information bottleneck** | Codec compresses *before* seeing what stage 2 needs; sub-pixel phase and small gradients that carry velocity may be discarded as noise (data-processing inequality: what the code drops, stage 2 cannot recover) | Compression happens *after* the dynamics are visible; the objective (predict future/upstream dots) tells the codec what to keep |
| **Sparse channels (Argo, tracks, stations)** | A single profile with no context yields a near-empty code; gridding loses depth resolution | Points enter natively as dots and are contextualised at encode time |
| **Cone geometry** | Re-learned per task and per channel by stage 2 (24 × 89 × channels tokens each time) | Fixed once by physics, shared by all tasks; the budget is 7 % of the cylinder |
| **Sample efficiency of probes** | Attention head needs labels to learn the geometry (AMOC: ~20 years of monthly labels ≈ 240 points) | Geometry already in the embedding; ridge/linear probes should suffice more often |
| **Static / mapping tasks** (land cover, LCZ) | Ideal: snapshot semantics | Slightly worse or neutral: context can blur a static class; mitigated by exposing e_loc |
| **Forecasting in pretraining** | Not available (no time axis in the codec) | Native (future dots as targets) |
| **Training complexity** | Simple per-pixel pipeline | Heterogeneous dot sampler; bucketing; leakage control |
| **Risk of shortcut learning** | Low (nothing to copy) | Real (derived channels) — needs source flags and drop rates |
| **Comparability with published GeoFMs** | Direct | Indirect (different unit of embedding) |

### 5.3 Why it is not zero-sum — the asymmetry

Two facts break the symmetry:

1. **B contains A.** A cone codec that ignores all dots but the anchor's own is a per-pixel codec. Whatever A can represent, B can; the reverse is false because of the bottleneck row above. So in *capability* the trade is one-sided; the price is paid in compute and in the cleanliness of the embedding's meaning.
2. **Compute is paid in different places.** A pays little per embedding but re-pays the context assembly in *every* stage-2 model, for every task, every time; B pays once per anchor and amortises across tasks. For a programme whose goal is many downstream questions (AMOC, carbon uptake, warming trajectories, biosphere predictability) from one representation, the amortisation favours B — provided the cone is generic enough that tasks do not each need a different one. That is precisely what the physics-set cone promises: the support depends on the *channel*, not on the task.

There *is* a genuine trade-off, and it is about **where the bottleneck sits relative to the dynamics**, not about "how much work" in total. The principled split:

- **into the embedding**: everything whose relevance is decided by *physics* — the cone (v, τ per channel), depth coupling, the local past. This is the same for every task.
- **into stage 2**: everything whose relevance is decided by the *task* — which section to integrate over, which teleconnection matters, which label.

That split makes the hybrid of Section 4.7 the answer rather than a compromise: a cached snapshot code for mapping, a dynamic code for prediction, and a thin task layer. What the existing Earth 2 record says: nothing decisive yet. The attribution matrix's 3 × 3 codec-vs-raw contrast (0.672 vs 0.659) is a single-seed probe number inside the probe noise band of `ml/CLAUDE.md` §3b (head k-fold spreads 0.036–0.245), so it is consistent with parity; and every rolled-skill number from before 2 September 2026 — including the old "corridor AUC" ablations — was withdrawn with the paper reset, because the stage-2 heads had been trained under the endpoint pool and had seen the held-out years. The v8 paper's first item, a Linear Inverse Model in pixel space against one in the codec's embedding space, is the first clean measurement of the bottleneck row above (if the embedding-space LIM is worse, the codec discards predictable signal), and Phase 0 is designed to follow it.

### 5.4 When A remains the right choice

Purely static targets (land cover, soil, canopy height, LCZ), pixel-level anomaly detection, and any setting where embeddings must be recomputed at very high volume with tiny compute (on-satellite inference, as Prithvi-EO's in-orbit deployment). Keep e_loc for these.

---

## 6. Phased plan and the decisive experiments

**Phase 0 — the A/B on existing data (weeks).** Same tokens as today (Earth 2 ocean pixel-months, 3 × 3 codec, sunflower-89 × 24 months). Train a cone codec over those tokens with the masked-dot objective (channel drop + lag-band drop + anchor reconstruction). Compare at equal total context, under the corrected protocol of paper v8 (2 Sep 2026) and after its null ladder exists (its own ordering: nothing trained before the nulls and the intervals exist), with n ≥ 3 seeds per arm:
- A: current codec + stage-2 attention head; B: cone codec + ridge probe; B′: cone codec + attention head.
- Verdict metric: rolled skill under the window-scope pool — MSSS against climatology and against damped persistence per lead, trained- and held-out-longitude scores reported separately, block-bootstrap intervals over (year, start) blocks, and the LIM null (pixel space and embedding space) beside every number. Lead-decay as the standing falsifier: a profile that does not decay with lead is a replay.
- Diagnostics, never verdicts: a **velocity probe** (GLORYS u, v at the anchor, and drifter velocities where available; derived channels masked at test) and the RAPID transport probe.
- Predictions: B ≥ A on rolled skill with a *linear* head, and B beats the embedding-space LIM where A does not; velocity probe: B ≫ A; static probes (e.g., bathymetry class) equal. Effects inside the tier's replicate band are written as consistencies, never levels.

**Phase 1 — ocean surface + atmosphere as forcing (months).** Add rungs 6 and 7 (OSTIA, DUACS, ASCAT, IMERG, AMSR2 ice; ERA5 / IFS open data) as gridded dots, with the L-shaped and A-family cones; pentad anchors at 25 km. New probes: SST and SSH forecast skill at 1–4 weeks vs persistence and vs the Copernicus 1/12° forecast; sea-ice edge.

**Phase 2 — interior (months).** Argo core/Deep/BGC profiles and moorings as native dots; GLORYS as flagged derived prior with 50 % drop; depth-band drop objective. Probes: AMOC (again), mixed-layer depth, heat content 0–700 m.

**Phase 3 — ocean biosphere (months).** Ocean-colour L3 (cloud gaps stay gaps — they are dots that are absent), PACE, SOCAT, BGC-Argo variables; carbon-flux reconstruction probe against SeaFlux/Copernicus carbon (held out, not input).

**Phase 4 — land (quarters).** Local codec on Sentinel-2/-1 and Landsat tiles (rungs 1–2); rung 10 fields as dots; column-only cones with the atmosphere hat. Probes: LAI/SIF forecast, soil-moisture forecast, fire risk, the standard GeoFM benchmarks on e_loc for comparability.

**Phase 5 — fast atmosphere (later).** Geostationary imagers and ground radar at 10-minute cadence via a second local codec (spatio-temporal tubelets); the fast disc of family A at hourly Δt.

**Ablations that run in every phase:** (i) cone-in vs cone-on-top at equal context; (ii) drop lag bands; (iii) drop far field; (iv) drop depth; (v) derived channels masked at test; (vi) cone × 2 vs cone × ½ on v and τ.

---

## 7. Data-continuity risks 2026–27 (verified 2 Sep 2026)

The verification pass turned up an unusual amount of churn; the channel-drop objective is the architectural answer (a missing sensor is a masked channel), but the *training archive* must be built sensor-agnostic:

- Sentinel-1: S1B failed Dec 2021; S1C launched 5 Dec 2024, S1D 4 Nov 2025 (data open 17 Apr 2026); **S1A terminated 29 Jun 2026** — the constellation is S1C + S1D, 6-day repeat, timeline shifted one day.
- Sentinel-2: three satellites (S2A/B/C) only until **end-2026**, then back to two.
- MODIS: Terra/Aqua begin shutting down **late 2026 / early 2027**; NASA directs users to VIIRS. **Suomi-NPP data delivery ends 1 Nov 2026** → use NOAA-21 (primary) / NOAA-20.
- Passive microwave: DMSP SSMIS retiring ~Sep 2026; NSIDC's NRT SSMIS sea-ice product retired 18 Jun 2026; NOAA AMSR2 NRT products end Sep 2026 in favour of AMSR3 (GOSAT-GW, launched 28 Jun 2025). NRT sea-ice concentration is now effectively a single-satellite (AMSR2, 2012) dependency.
- Altimetry: Sentinel-6B launched 17 Nov 2025, reference-mission hand-over date not yet published; DUACS L4 is now 0.125°.
- Climate-quality records that stop: OC-CCI v6 ends 31 Dec 2024 (watch for v7); ESA CCI soil moisture and snow are annual retrospective releases (no NRT).
- Licences to watch: EN4 (non-commercial), GRDC (research only, no redistribution), OPERA radar (EUMETNET members only), GLEAM (commercial use needs approval).

---

## Appendix A — dataset spec sheet (verified against agency pages, 2 Sep 2026)

Condensed from the four verification passes; "(unv.)" marks numbers the agency page did not state. Full URLs are in the Sources section.

**Imagery and radar.** Sentinel-2 MSI L2A: 13 bands, 4 × 10 m / 6 × 20 m / 3 × 60 m, 5-day two-satellite revisit, 56° S–84° N, L1C since Jul 2015, global L2A since Dec 2018, free/open (CDSE). Sentinel-1 IW GRDH: 20 × 22 m resolution on 10 m pixels, 251.8 km swath, 6-day repeat with two satellites, since Oct 2014; OCN wind on a 1 km grid. Landsat 8/9 C2 L2: 30 m reflectance, 100 m thermal resampled to 30 m, 8-day combined revisit, Landsat 8 since Feb 2013 / 9 since Sep 2021, public domain. Sentinel-3 OLCI: 21 bands, 300 m, < 2 days with two satellites, since 2016; SLSTR: 500 m VIS/SWIR, 1 km TIR, SST better than 0.3 K, daily. MODIS: 250/500/1,000 m, Terra 1999 / Aqua 2002, retiring late 2026–27. VIIRS: 375 m I-bands, 750 m M-bands and day/night band, SNPP 2011 / NOAA-20 2017 / NOAA-21 2022; SNPP data end 1 Nov 2026. Geostationary: GOES ABI 0.5/1/2 km, full disk 10 min (mesoscale 30–60 s); MTG-I1 FCI 1 km VIS/NIR, 2 km IR, 10 min (operational 4 Dec 2024; 2.5-min Europe scan awaits MTG-I2, ~2027); Himawari-9 AHI 0.5/1/2 km, 10 min (2.5 min Japan). Altimetry: Sentinel-6 MF/6B and Jason-3 ~7 km along-track at 1 Hz, ~10-day repeat, ± 66°; Sentinel-3 SRAL 27-day repeat; NRT 3 h / STC 36–48 h / NTC 1–2 months. SWOT KaRIn: 2 × 2 km gridded SSH on two 50 km swaths (20 km nadir gap), 20.86-day repeat, ± 78°, science orbit since Jul 2023. ASCAT Metop-B/C: 12.5 km and 25 km wind products, two 550 km swaths, ~daily global per satellite, OSI SAF within 2 h 45 min, CC BY 4.0. AMSR2: sea-ice concentration 12.5/25 km (NSIDC) and 6.25/3.125 km (U. Bremen), SST 0.25° all-weather, since Jul 2012. SMOS (35 km, 2009–) and SMAP (36 km L3 daily, 9 km L4 3-hourly model-assimilated, ~40 km SSS on 0.25°; 2015–). GPM IMERG V07: 0.1°, 30 min, Early ~4 h / Late ~14 h / Final ~3.5 months, 2000–. MRMS: 1 km, 2 min, CONUS + Alaska/Hawaii, since 2014; OPERA CIRRUS: 1 km, 5 min, 33 EUMETNET countries, since 2024 (licensed to third parties). GEDI: 25 m footprints every 60 m, ± 51.6°, Apr 2019– (13-month gap Mar 2023–Apr 2024). ICESat-2: ~11 m footprints, 6 beams, 91-day repeat, since Oct 2018. CYGNSS: 25 km winds, ± 38°, median revisit 3 h, 2017–.

**Atmosphere.** ERA5: TL639 (31 km) native, 0.25° grid, hourly, 137 model / 37 pressure levels, 1940–, ERA5T ~5 days behind, final 2–3 months; SST/sea ice prescribed from HadISST2/OSI SAF/OSTIA; CC-BY. ERA5-Land: 9 km, hourly, 1950–, land model forced by ERA5 with no assimilation. ECMWF IFS 9 km (137 levels) licensed; Open Data 0.25° IFS + AIFS, 4 cycles/day, CC-BY-4.0 ("HRES" renamed "ENS control" with Cy50r1, Oct 2025). GFS: 13 km model on 0.25° output, 4 runs/day, 127 layers, open (NODD). HRRR: 3 km CONUS, hourly cycles, 2014– archive. MERRA-2: 0.5° × 0.625°, hourly 2-D / 3-hourly 3-D, aerosols assimilated, 1980–, ~3-week latency. ISD: > 35,000 stations (> 14,000 active), hourly, 1901–. GHCN-Daily: > 100,000 stations, daily. WMO GBON: ~ 9,000 surface stations at 200 km, ~ 1,000 upper-air at 500 km, hourly / twice daily since 1 Jan 2023. IGRA 2.2: > 2,800 stations historical, ~ 800 NRT, 00/12 UTC, 1905–. AMDAR: > 700,000 aircraft observations/day (WMO figure). ICOADS: marine reports since 1662, NRT monthly. NDBC: hourly buoy/C-MAN reports. GOES GLM: 8 km nadir, 20 s products, Americas to 52° N. MAIAC MCD19A2: 1 km AOD daily, 2000–. OCO-2: ~3 km² soundings, 16-day repeat, 2014–; OCO-3 on ISS 2019–. TROPOMI: 3.5 × 5.5 km (7 × 5.5 km SWIR for CH₄/CO), daily global, 2017–. GOSAT: ~10 km footprints (unv.), 3-day revisit, 2009–. MOD10A1 snow: 500 m daily, 2000–; IMS: 24/4/1 km daily NH, 1997–.

**Ocean.** OSTIA L4 SST: 0.05°, daily, NRT ~1 day, reprocessed record 1 Oct 1981–; MUR: 0.01°, daily, 2002–; OISST: 0.25°, daily, 1981–. DUACS L4: 0.125° daily SLA/ADT/geostrophic u, v; NRT ~1 day; multi-year record 1993–; SWOT-enhanced L4 only experimental (Mar 2023–Jun 2025). Argo (OceanOPS 2 Sep 2026): 4,372 operational floats (3,229 core), 10-day cycle, 0–2,000 m, ~ 12–13 k profiles/month, real-time ≤ 12–24 h; Deep Argo 219 floats (design 1,228); BGC-Argo 989 floats with ≥ 1 sensor (573 with all six variables; target 1,000). GLORYS12V1: 1/12°, 50 levels to 5,500 m, daily and monthly, 1993–mid-2021 + interim extension (to Jun 2026); Copernicus global analysis/forecast: 1/12°, 50 levels, hourly surface / daily, 10-day forecast, updated daily 08:00 UTC. OSCAR v2: 0.25°, daily, top 30 m, final to Aug 2022, NRT ~2 days. Global Drifter Program: 1,317 operational (design 1,250), hourly/6-hourly, 1979–. OC-CCI v6: 4 km, daily/5-day/8-day/monthly, Sep 1997–Dec 2024; GlobColour: 4 km multi-sensor (300 m OLCI L3), L4 interpolated, L3 observed, 1997–. PACE OCI: hyperspectral 340–895 nm at 5 nm, ~1.2 km, 2-day global, launched 8 Feb 2024. Sea ice: OSI SAF AMSR2 10 km (5 h latency); Bremen ASI 6.25/3.125 km; NSIDC-0051 25 km 1978–2025; OSI SAF drift 62.5 km, 48 h; ICESat-2 ATL10 freeboard; CCI thickness 25 km monthly winter. Waves: Copernicus MFWAM 1/12° 3-hourly + 10-day forecast; altimeter Hs L4 0.5° 6-hourly. SSS: OISSS 0.25° 7-day (2011–); Copernicus multi-obs 0.125° daily/weekly (1993–). Moorings: GTMBA 54 operational (TAO 43), OceanSITES 65, GLOSS tide gauges 149 core. SOCAT v2026: 44 M fCO₂ observations, 1957–2025, CC BY 4.0. Copernicus surface carbon: 0.25° monthly, 1985–, FFNN reconstruction; SeaFlux v2023.02: 1° monthly, 1982–2022. GEBCO_2026: 15 arc-seconds (~450 m), released 23 Apr 2026. Copernicus INSITU TAC: hourly updates, 24–48 h latency; CORA OA 0.5°, 187 levels, monthly, 1960–; EN4: 1°, 42 levels, monthly, 1900–, non-commercial.

**Land / static.** Copernicus DEM GLO-30: 30 m global DSM (TanDEM-X 2011–15), free. NASADEM/SRTM: 30 m, 60° N–56° S. ESA WorldCover: 10 m, 11 classes, 2020 and 2021 maps, CC BY 4.0. Dynamic World: 10 m per Sentinel-2 scene, 2015–. ESA CCI land cover: 300 m annual 1992–, 22 classes. MCD12Q1: 500 m annual 2001–. SoilGrids 2.0: 250 m, six depth layers, CC BY 4.0. MOD13Q1: 250 m 16-day NDVI/EVI; MOD15A2H: 500 m 8-day LAI/FPAR; Copernicus LAI/FAPAR: 300 m, 10-daily, 2014–. TROPOSIF: 3.5 × 5.5 km daily, May 2018–2021 on the official FTP. MOD11A1 LST: 1 km, day + night daily; SLSTR LST: 1 km, 0.9-day revisit with two satellites. SMAP L3: 36 km daily (~1 d latency); L4: 9 km 3-hourly (~2.7 d). ESA CCI soil moisture: 0.25° daily 1978–2024 (annual releases). ASCAT H SAF SSM: 12.5 km, ~2 h latency. MOD16A2GF ET: 500 m 8-day 2000–; GLEAM v4.3: 0.1° daily 1980–2024. Fire: MOD14 1 km / VNP14IMG 375 m, FIRMS < 3 h; MCD64A1 burned area 500 m monthly. Snow: MOD10A1 500 m daily; RGI 7.0: 274,531 glacier outlines; GlobSnow/CCI SWE 25 km / 0.1°. GRACE/GRACE-FO JPL mascons: 3° equal-area on 0.5° grid, monthly, Apr 2002–, ~40–60 d latency. FLUXNET2015: 212 sites (frozen Feb 2020); AmeriFlux 552 sites with data; ICOS > 100 ecosystem stations; half-hourly. GRDC: ~ 9,000 gauges (research only). GloFAS: 0.05° daily, 1979–, LISFLOOD on ERA5. Hydroweb: 124 lakes + 11,336 river points, ≤ 1.5 days after a pass. WorldPop 100 m/1 km annual; GHS-POP 100 m, 5-yearly 1975–2030. OSM roads: ODbL. VIIRS DNB Black Marble: 500 m daily/monthly/yearly, 2012–. Canopy height: Meta/WRI 1 m (2009–2020 composite), ETH 10 m (2020); GEDI L4B biomass 1 km.

---

## Sources

Verification passes on 2 Sep 2026 against: Copernicus Data Space / Sentinel Online (Sentinel-1/2/3/6 mission and product pages, orbital reconfiguration and S1D notices), USGS Landsat Collection 2 pages, NASA MODIS/VIIRS transition notice and LAADS, NOAA NESDIS (GOES-19 operational, SNPP cessation, AMSR2→AMSR3), WMO OSCAR (FCI, ASCAT, SMOS), JMA Himawari, PO.DAAC (SWOT, MUR, OSCAR, SMAP, OISSS, CYGNSS, GRACE mascons), OSI SAF product pages (OSI-104, OSI-405, OSI-408, SSMIS end-of-data story), NSIDC (AMSR2, SSMIS notices, SMAP, ATL07/08/10, MOD10A1, IMS), GPM IMERG, NSSL MRMS, EUMETNET OPERA, GEDI mission status, ICESat-2 specs; CDS ERA5 / ERA5-Land documentation, ECMWF Open Data and Set I pages, NCEI GFS/ISD/GHCN-D/IGRA/ICOADS, NOAA HRRR, GMAO MERRA-2, WMO GOS/GBON, NDBC, GOES GLM, LP DAAC (MCD19A2, MOD11/13/15/16, MCD12Q1, MCD64A1, MOD14, NASADEM, SRTM), OCO-2/3, TROPOMI/S5P, GOSAT; Copernicus Marine product pages (OSTIA NRT/REP, DUACS NRT/MY, GLORYS12, global analysis/forecast, GlobColour L3/L4, waves, SSS, carbon, INSITU TAC, CORA), Argo (argo.ucsd.edu, biogeochemical-argo.org) and the OceanOPS API (operational platform counts, 2 Sep 2026), AVISO (SWOT L4 experimental), NCEI OISST, AOML GDP, ESA CCI (ocean colour, sea ice, soil moisture, snow, land cover), PACE OCI, PMEL GTMBA, OceanSITES, PSMSL/UHSLC/GLOSS, SOCAT, SeaFlux, GEBCO, Met Office EN4; Copernicus DEM, ESA WorldCover, Dynamic World (Earth Engine catalog), ISRIC SoilGrids, Copernicus Global Land LAI/FAPAR, TROPOSIF, H SAF ASCAT SSM, GLEAM, FIRMS, GLIMS/RGI, GlobSnow, FLUXNET/ICOS/AmeriFlux, GRDC, GloFAS (EWDS), Hydroweb.next, WorldPop, GHSL, OpenStreetMap, Black Marble, Meta/WRI and ETH canopy height, ORNL DAAC GEDI L4A/L4B.

Method references: Frankignoul & Hasselmann (1977), *Tellus* 29, 289 — stochastic SST-anomaly model (3–6-month e-folding at mid-latitudes, per Deser, Alexander & Timlin 2003); Pujol et al. (2016), *Ocean Sci.* 12, 1067 — DUACS DT2014 (±6-week centred window; 10–45-day temporal correlation scales); Lellouche et al. (2021), *Front. Earth Sci.* 9, 698876 — GLORYS12 (7-day cycle); Emery, Thomas, Collins, Crawford & Mackas (1986), *J. Geophys. Res.* 91, 12865 — maximum cross-correlation feature tracking of SST imagery; Jaegle et al. 2021, *Perceiver IO* (arXiv 2107.14795); Pereira et al. 2026, *PARADIS — Learning to Advect* (arXiv 2601.21151); Hahner, Zampieri et al. 2026, AIFS surface ocean (arXiv 2604.25559) for slow-field loss scaling; Earth 2 dependency-cone note (16 Aug 2026); `ml/paper/paper.tex` v8 (2 Sep 2026, the reset paper) for the corrected protocol, the retained probe numbers and the withdrawn claims; `ml/CLAUDE.md` §3b for the probe noise band.

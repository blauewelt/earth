# The state of one pixel

*Which of the catalog's 247 sources compose into a holistic assessment of a
single place on Earth — and into a prediction of where that place, and its
neighbours, are heading.*

This is the composition question behind the whole project: not "what data
exists" but "what stacks". The answer below organises the catalog by the
**role** each source plays in a per-pixel state vector. Roughly a quarter of
the catalog composes directly; the rest is truth anchors, context scalars, or
priors — still essential, but they enter differently.

## 1 · The frame

"The situation of a pixel" decomposes into five things:

1. **State now** — the instantaneous physical fields at this location.
2. **Memory** — the pixel's own history: climatology, anomaly, trend.
   A 20 °C SST pixel means nothing until you know its normal is 17 °C.
3. **Forcing** — what pushes the pixel: radiation balance, GHG burden,
   emissions in and around it, human presence.
4. **Flow** — what couples the pixel to its neighbours: wind, currents,
   river routing, ice motion. This is what makes *prediction* possible
   at all; tomorrow's weather here is today's weather upwind.
5. **Future** — explicit forecasts and projections at every lead from
   hours to decades, plus the pixel's own extrapolable trends.

**The common grid.** The natural resolution is **0.25°, daily** (~28 km at
the equator, ~1 million pixels). Not because it's the finest available —
Landsat is 30 m — but because it is where the ensemble converges: ERA5,
the AI forecast models, DUACS altimetry, OISST, CCI soil moisture, GPCP,
seasonal forecasts and NEX-GDDP downscaled projections all live at or near
0.25°. Finer sources (1 km MODIS, 10 m WorldCover) aggregate *down* to it
without loss of meaning; coarser ones (GRACE ~300 km) interpolate onto it
with an honesty flag — the value is real, the resolution is not. A finer
frame multiplies storage without adding sources; a coarser one throws away
the majority of the stack.

## 2 · The backbone: reanalysis

One source family does more work than the next twenty combined:

| Source | Why it is the spine |
|---|---|
| **ERA5 / ARCO-ERA5** (`era5`, `arco-era5`) | ~100 variables × hourly × 0.25° × 1940–present, analysis-ready Zarr on a public bucket. Temperature, wind, humidity, pressure, precipitation, radiation, soil layers, wave state — at every pixel, every hour, for 85 years. This *is* the state vector's first ~100 channels, and the memory axis comes free (85 years of the pixel's own history). |
| **ERA5-Land** (`era5-land`) | The same at 0.1° over land, for surface detail. |
| **GLORYS12** (`glorys`) | The ocean's ERA5: 1/12°, 50 depth levels, full T/S/currents/SSH/ice, 1993–present. An ocean pixel is a column, not a surface — this supplies the column. `oras5`/`en4` extend the record back (1958/1900) at coarser grain. |
| **CAMS reanalysis** (`cams-composition`) | The chemistry ERA5: aerosols, O₃, CO, NO₂, CH₄/CO₂ at ~0.4–0.75°, 2003–present. |

The strongest evidence that this composition suffices is that it is already
being *used* as one: **GraphCast, Pangu, FourCastNet and ECMWF's AIFS**
(`graphcast`, `aifs`) take exactly this — an ERA5-format 0.25° global state —
as their sole input and beat the operational NWP system at 10-day forecasts.
The per-pixel state vector is not a hypothetical construct; it is the input
layer of the current best forecast models. Anything we add beyond ERA5 is
enrichment of a vector already proven predictive.

## 3 · State-now channels by sphere

Satellite products that observe what reanalysis only estimates — each one a
direct field on (or trivially regridded to) the common grid.

**Ocean pixel** — `ostia` or `oisst` (SST, the workhorse), `smap-sss`/
`smos-sss` (surface salinity), `sealevel-cmems` (sea-level anomaly — the
ocean's pressure gauge), `oc-cci`/`nasa-oceancolor` (chlorophyll: the
biology channel), `argo` (the sparse truth column beneath, 0–2000 m),
`gebco` (static depth — the boundary condition).

**Land pixel** — `modis-lst` (skin temperature), `esa-cci-sm` (surface soil
moisture, 1978–present), `grace-tws` (total water storage — the only
observation of *deep* water, at mascon resolution), `imerg`/`chirps`/`gpcp`
(precipitation at three resolution/record trade-offs), `modis-ndvi`
(vegetation vigour), `spei` (drought state integrated over months),
`glofas` (river discharge — the pixel's hydrological output), `worldcover` +
`modis-lc`/`dynamicworld` (what the pixel *is*: forest, crop, city),
`hansen-gfc` (whether that just changed).

**Cryosphere pixel** — `osisaf-sic`/`nsidc-sic-cdr` (sea-ice concentration,
1978–present), `awi-cs2smos` (ice thickness), `ims-snow`/`modis-snow` (snow
extent), `snow-cci` (snow water equivalent), `permafrost-cci` (ground
thermal state), `rgi` + `itslive` (glacier outlines and velocity),
`grace-mascon` (ice-sheet mass), `bedmachine` (static bed — the boundary).

**Composition & energy** — `ceres` (top-of-atmosphere radiation balance:
the pixel's energy budget, the ultimate forcing), `s5p-methane` (TROPOMI
CH₄/NO₂), `oco` (CO₂), `modis-aod` (aerosol).

**Human & impact** — `worldpop`/`ghsl` (who is at the pixel — turns hazard
into risk), `edgar`/`ceds` (gridded emissions at 0.1°: what the pixel puts
into the atmosphere), `climatetrace` (the point sources doing it),
`firms` (fire now), `gfed` (fire history), `ibtracs` (every cyclone track
that ever passed), `hadex3` (extremes indices — the pixel's tail behaviour).

**Biosphere** — `gbif`/`obis` (occurrence density as a biodiversity field),
`fluxnet` (carbon-flux truth at towers).

## 4 · Memory: the pixel's own past

Three kinds of history channel, all computable from sources above:

- **Climatology** — WOA (`woa`) for the ocean column, the app's own grid
  snapshots (GPCP/OISST/E-OBS) for the surface, ERA5 1991–2020 normals for
  everything else. State only means something as an *anomaly against these*.
- **Trend** — the pixel's local derivative: SST warming rate (from
  `oisst`, 44 years), sea-level rise rate (`sealevel-cmems`, 31 years),
  ice-mass loss (`grace-mascon`), glacier thinning (`rgi` + Hugonnet, in
  the app already), NDVI greening/browning, soil-moisture drying. These
  are the cheapest useful "prediction": persistence of trend.
- **Extremes memory** — `hadex3` percentiles, `ibtracs` recurrence,
  `emdat` impact history: how bad it has ever been here.

## 5 · Flow: why neighbours are not optional

A pixel's future is *not a function of that pixel*. Weather information
propagates at the jet stream's ~25 m/s — 2000 km/day, or 80 pixels/day at
0.25°. Any per-pixel prediction that looks only inward is wrong by lunch.
The coupling fields are:

- **ERA5 winds** (u, v at multiple levels) — the atmospheric advection
  operator.
- **GLORYS currents** + `gdp` drifters — the ocean's, ~1000× slower,
  which is why ocean anomalies (and marine heatwaves) persist for months:
  the ocean is the climate system's memory, the atmosphere its messenger.
- **GloFAS routing** — water leaves the pixel along the river network,
  not isotropically.
- **ITS_LIVE velocity** — ice flows; today's thinning at a glacier
  terminus was set in motion upstream years ago.

This is exactly the structure GraphCast learns: a graph neural network
whose edges *are* the neighbour couplings. The composition argument and
the ML architecture agree.

## 6 · The future axis, by lead time

| Lead | Source | What the pixel gets |
|---|---|---|
| 0–15 days | `aifs` (open GRIB, real-time), `graphcast`-class models | Deterministic weather at 0.25° |
| 0–5 days | `geos-cf`, `cams-composition` | Air-quality forecast |
| 0–30 days | `glofas` | Flood forecast on the river network |
| 0–10 days | `cmems` | Ocean physics forecast |
| 1–13 months | `c3s-seasonal`, `nmme` | Probabilistic seasonal anomalies (ENSO-driven skill) |
| 1–10 years | `dcpp` | Decadal hindcast/forecast ensembles (initialised — the AMOC lives here) |
| decades–2100 | `cmip6-gcs`/`cmip7`, downscaled to the common grid by `nex-gddp` (0.25°!) and `cordex`; impacts via `isimip` | Scenario envelopes per pixel |
| next-gen | `destine` | 5 km eddy-resolving digital twin, 2020–2040 |

The remarkable fact: `nex-gddp` puts CMIP6 *on the same 0.25° grid as ERA5*,
so a pixel's column can hold its 1940 reanalysis, today's observation, next
week's AI forecast and its 2100 SSP2-4.5 envelope in one aligned stack.

## 7 · What does NOT compose per-pixel — and how it enters anyway

- **Station networks** (`ghcnd`, `psmsl`, `openaq`, `aeronet`, `wdcgg`…) —
  points, not fields. They enter as **truth anchors**: the calibration and
  validation set for every gridded channel, and the bias-correction target.
- **AMOC arrays** (`rapid`, `osnap`, `move`, `samba`) — section integrals.
  They compose as **global context scalars**: an AMOC-state channel
  broadcast to every North Atlantic pixel (whose local expression — the
  cold blob, `caesar2018` — *is* per-pixel).
- **Modes & indices** (ENSO from `ersst`, NAO from pressure fields) — same
  pattern: a handful of scalars that condition every pixel's statistics.
  Seasonal forecast skill *is* mostly these scalars.
- **Country tables** (`primap`, `owid-co2`, `unfccc`, `ndgain`) — policy
  context; rasterise only crudely via `edgar`'s gridding.
- **Paleo** (`epica`, `pages2k`, `path-proxy`…) — not state but **prior**:
  the distribution of what the system has done before, and the only
  evidence that AMOC collapse is a real mode of the system.

## 8 · Honest gaps

No open source in the catalog gives: wind profiles observed (not
reanalysed) globally (Aeolus ended); the ocean below 2000 m (Deep Argo is
a pilot); soil moisture below ~5 cm except via GRACE's 300 km blur;
sub-daily precipitation with global truth (IMERG is satellite-only over
ocean); biology beyond chlorophyll and occurrences. Every one of these is
a place where the state vector is an estimate, not an observation — the
uncertainty channel should say so.

## 9 · The minimal viable composition

If forced to pick ~25 of the 244 for a working per-pixel prototype:

**State**: `arco-era5`, `glorys`, `ostia`, `sealevel-cmems`, `smap-sss`,
`oc-cci`, `esa-cci-sm`, `grace-tws`, `imerg`, `modis-ndvi`, `modis-lst`,
`osisaf-sic`, `snow-cci`, `ceres`, `cams-composition`.
**Static boundary**: `gebco`, `worldcover`, `ghsl`.
**Memory**: `woa` + normals computed from the state stack.
**Flow**: ERA5 winds + GLORYS currents (already in the state picks).
**Future**: `aifs`, `c3s-seasonal`, `nex-gddp`, `glofas`.
**Anchors**: `argo`, `ghcnd`.

Fifteen state sources ≈ 150–200 channels at 0.25°/daily ≈ a few hundred
numbers that say what a place *is*, what it *was*, what pushes it, what
flows through it, and where it is heading — which is, in one sentence, the
input specification for an Earth state embedding.

---

*See also: [COMBINING_DATASETS.md](COMBINING_DATASETS.md) for which sources
measure the same quantity (redundancy → ensembles and uncertainty), and
[CATALOG.md](CATALOG.md) for every record's access details.*

# E-082 wave 6 · The biosphere — forests, photosynthesis, life — in the three fine families

*Design, 20 September 2026, main session. Chris, 2026-09-19: "We want to make
this a globally useful model that can predict biosphere (forests). Lidar is
very important for those. But maybe many other datasets. Can you make sure all
families have such data?" This note is the design the agents build from; the
three dataset notes (families 1.gf, 1.0.tf, 0.9.tf) get the rows, the paper
gets the read-out, and `ml/family1/adapters/` gets the adapters. Nothing here
is measured until a store's probe replaces its estimate.*

E-082 is the build of the three fine-granularity observation families — the
datasets at ten kilometres or finer and five days or finer that the model reads
beside the global 0.25° tensor. Wave 6 adds the living surface to them.

## 1 · What "predict the biosphere" means for this model

The model predicts what an instrument would read at a place, time and
footprint. For the land biosphere the instruments are:

| what is asked | the truth series | record | why it is the read-out |
|---|---|---|---|
| **canopy structure** — how tall, how dense, how much carbon a canopy holds | GEDI laser footprints (relative-height profile, cover, biomass), ICESat-2 segments beyond 51.6° | 2019 → (GEDI), 2018 → (ICESat-2) | the only direct measurement of the forest in three dimensions; every canopy-height map is fitted to it |
| **photosynthesis** — how much the vegetation is working right now | solar-induced chlorophyll fluorescence (SIF) soundings from TROPOMI and OCO-2/3; gross primary production at eddy-covariance towers | 2018 → (TROPOMI), 2014 → (OCO-2), 1991 → (towers) | fluorescence is emitted by photosynthesis itself; the towers are the long in-situ record, the biosphere's equivalent of the overturning arrays |
| **leaf state and its calendar** — leaf area, green-up and senescence dates | MODIS/VIIRS leaf-area index and FPAR every 8 days; MCD12Q2 phenology metrics per year | 2000 → | the seasonal cycle a forecast at weeks must get right |
| **disturbance** — where forest is cleared or burns | Hansen loss year and the JRC tropical-forest change (built as `lossyear`), forest-clearing alerts, burned area, active fires (`fire`, built) | 1990 / 2000 → | the annual and daily change targets |
| **where life is** — species occurrences | GBIF dated, georeferenced occurrence records | 1600s → | three billion observations of organisms; the biosphere's ICOADS (the ship-report archive), sparse, biased, and the only global record of species |

So the paper's read-outs become five: the Atlantic overturning, the ocean
carbon sink, El Niño, sea-surface temperature, **and the land biosphere** —
scored by (a) the held-out towers' carbon flux against a per-site day-of-year
climatology and damped persistence, and (b) canopy height at held-out GEDI
footprints against the 2020 30 m canopy-height map (the static prior) and the
previous year's footprint value. Fluorescence at held-out soundings is the
dense-field check for land, the way the level-3 SST cell loss is for the
ocean.

**What carries a biosphere signal, for the per-channel cone.** Nothing on land
advects: leaf area, canopy height and species are where they are. Their
drivers are the atmosphere and the soil (temperature, rain, radiation, vapour
deficit, soil moisture — the shared channels of every sphere), so the land
channels' drift is ≈ 0 and their aperture opens over the atmospheric tokens;
fire is the one land channel with a carrier (the wind, at 5–30 km per day).
Their memories are long — leaf area weeks to months, canopy height years to
decades — longer than the two-year cone. A slow channel is therefore read as a
*state*, not a history: the previous annual map (canopy height, biomass,
phenology, loss) enters as an input token at lag ≥ 73 pentads (one year), and
the target is next year's map at the same footprint. That is the hourglass at
the annual time footprint, `log2 f_t = log2(365/5) = +6.2`, not a new
mechanism.

## 2 · The stores

Every size below is an estimate (†) until `--stage probe` writes `probe.json`;
records, tile counts and licences marked *measured* were verified by request
on 2026-09-20 (the notes agent's listing calls — NASA CMR, GLAD, GBIF, the
AWS buckets, CEDA, S5P-PAL). ESA BIOMASS's catalogue refused connections that
day; ETH's 10 m host answered 429.
Tier P rows cost `27 + 2C` bytes (schema 2) or `31 + 2C` (schema 3, records
before 1914). "EDL" = NASA Earthdata Login, whose secrets are in place and
verified against LP DAAC (the land data centre), NSIDC (snow and ice), PO.DAAC
and OB.DAAC (ocean); **GES DISC** (the atmosphere data centre, which serves
OCO-2/3) is unapproved on the account until Chris approves the application in
his Earthdata profile — the same block that holds `xco2` and `irtb`.

### 2.1 Family 1.0.tf — land and coast

| store | what it is (source; channels C) | tier · footprint | record; rows or frames | stored† | licence · track | phase | credential |
|---|---|---|---|---|---|---|---|
| `gedi` | **laser canopy height, cover and biomass from the Space Station.** GEDI (Global Ecosystem Dynamics Investigation) L2A + L2B v003 (LP DAAC) joined to L4A **version 3** (ORNL DAAC; listed from 2019-04-04, built from L2A v003; v2.1 selectable) by shot number: ground elevation, RH25/50/75/90/98 (the heights below which 25–98 % of the returned laser energy lies), canopy cover, plant-area index, above-ground biomass and its error, sensitivity (C = 11; the `_rel3` quality and degrade flags go into `qc`) | P · 25 m footprints every 60 m along 8 tracks, ±51.6°; log2 f_p = −10.1, f_t instantaneous | 2019-04 → 2023-03 and 2024-04 → 2025-07 released; 4–10 × 10⁹ quality shots | 0.2–0.5 TB | NASA open · public | **A** (was B) — phase A builds one year (2022) and measures; the rest follows the probe | EDL (in place). Measured 2026-09-20: one month of the three products is **2.58 TB** (2022-06: 1,428 granules each), OPeNDAP is offered for none of them; the adapter reads only its datasets over HTTP byte ranges (`_h5range.py`; a 200 where 206 was asked is a refusal). Probe #296 in flight |
| `icesat2` | **laser terrain and canopy heights from a polar satellite.** ICESat-2 ATL08 release 007 (NSIDC): terrain height, canopy height (h_canopy, the 98th-percentile relative height), canopy cover, photon counts, segment quality per 100 m segment (C = 6) | P · 100 m segments, 6 beams, ±88°; log2 f_p = −8.1 | 2018-10 →; ≈ 4 × 10⁹ land segments a year; 376 GB a month native (2022-06: 4,641 granules, measured) | ≈ 1.1 TB for the record | NASA open · public | **A** (was C) — one year (2022) in phase A; the rest phase B by size | EDL (NSIDC in place); byte-range reads as `gedi`; probe #297 in flight |
| `canopy30` | **the 2020 canopy-height map at 30 m.** GLAD's forest height 2020 from the Global Land Cover and Land Use Change 2000–2020 set (Potapov et al. 2022; the method is Potapov et al. 2021, Landsat fitted to GEDI); uint8 metres, codes for water/no-data (C = 1); annual sharded grid on the Hansen 10° tile grid cut into 2.5° groups exactly as `lossyear` | G · 30 m, annual (one bin) | 2020 only; **261 tiles** (13–616 MB each, median 77, 35.95 GB in all; a strict subset of `lossyear`'s 280; no water or no-data code — 0 is "no woody vegetation ≥ 3 m", the map is dense) | **probe run here** (tile 50N_000E): 281 MB stored for 304 MB fetched, valid 1.0, 9.8× → **≈ 33 GB**; 11–21 h for 261 tiles in one bin → a box | CC BY (GLAD states free redistribution with citation) · public | **A** | none (GLAD storage, public) |
| `canopy_ref` | **the finer canopy maps, catalogued, pixels at the source.** ETH Global Canopy Height 10 m 2020 (Lang et al. 2023; height and its standard deviation; 3 TB native) and Meta/WRI 1 m canopy height (Tolan et al. 2024; AWS `dataforgood-fb-data`, tens of TB): one row per tile with footprint, sensor code and asset URL, the tier-T catalogue form | T · 10 m and 1 m | 2020 (ETH: 2,651 tiles of 3° × 3°, the producer's own index, public); Meta v1 56,145 and v2 213,109 quadkey tiles, decade composites 2009–2020 measured from their sidecars | catalogue < 1 GB | CC BY 4.0 · public | **A** (catalogue; adapter landed) | none |
| `biomass100` | **yearly above-ground biomass maps, 100 m.** ESA CCI Biomass v7.0: biomass and its standard deviation (C = 2, uint16 Mg/ha), years 2005–2012 and 2015–2024; GEDI L4B 1 km gridded biomass 2019–2023 as a second group (C = 2) | G · 100 m annual; L4B 1 km | annual (CEDA: 2005–2012, 2015–2024; L4B v2.1 2019-04 → 2023-03) | ≈ 30 GB per year at 100 m; L4B < 1 GB | ESA CCI terms read: "any purpose" with acknowledgement; redistribution not named → public with attribution (decision 4) | **A** for 2020 and the L4B group (was C); other years phase B | none (CEDA) |
| `sif` | **photosynthesis seen from orbit.** Solar-induced chlorophyll fluorescence: TROPOMI (Sentinel-5P) ungridded daily soundings (Caltech, Köhler et al. 2018; and ESA TROPOSIF from the S5P-PAL service): SIF at 740 nm and its daily-corrected value, 1-σ, cloud fraction, solar and viewing zenith (C = 6); OCO-2 and OCO-3 SIF Lite (GES DISC) as second and third platforms with the same channels | P · 7 × 3.5 km (TROPOMI), 1.3 × 2.25 km (OCO); log2 f_p = −2.49 / −4.02 (probe median −2.45); instantaneous | daily file **2018-05-01** → (TROPOMI, S5P-PAL `L2B_SIF___`, one 0.3–0.4 GB netCDF a day; 04-30 is the per-orbit product), 2014-09 → (OCO-2), 2019-08 → (OCO-3); **probe run here** (2023-07): 101.7 M rows, 12.3 GB in 577 s, no NaN, no out-of-bounds → 1.2 × 10⁹ soundings a year | **46.8 GB a year uncut**; 245 GB for 2018-05 → 2023-07, ≈ 390 GB to today; 63.9 % kept at cloud ≤ 0.2 (Caltech's tarballs are 34 GB gzip streams with no member index and stop at 2022 — rejected) | Caltech: CC0; ESA: Copernicus open; NASA open · public | **A** (TROPOMI now; OCO after the GES DISC approval) | none for TROPOMI; EDL + GES DISC for OCO |
| `lai500` (E7) | **leaf area and light absorbed, every 8 days.** MOD15A2H v061 (Terra), MYD15A2H (Aqua), VNP15A2H (VIIRS) 8-day leaf-area index, FPAR (fraction of absorbed photosynthetically active radiation), their standard deviations and quality (C = 4); the 8-day composite is an exception E7 to the five-day rule, carried as a `frame_table` like `chirps05`'s pentads | G · 500 m sinusoidal tiles, sharded on the native tile grid (the layout already holds a projected grid: `seaice_asi` is polar stereographic) | 2000-02-18 → (Terra, v061), 2002-07-04 → (Aqua), 2012-01-01 → (VIIRS, v002, HDF5 with unsuffixed dataset names); 46 composites a year, 274–290 tiles per composite (union 290; 13,192 Terra granules in 2020, 46.0 GB); the 8-day composites file into 1,656 distinct bins over 2000–2035 at a minimum midpoint gap of exactly 5.0 days, F = 1 | ≈ 60 GB per sensor-year after compression (probe #294 replaces it) | NASA open · public | **A** builds Terra 2020–2024 and measures; the record phase B | EDL (LP DAAC in place); probe #294 in flight |
| `pheno500` | **the leaf calendar, per year.** MCD12Q2 v061 land-surface phenology: greenup, mid-greenup, peak, senescence, mid-greendown, dormancy dates (day of year, int16), EVI minimum, amplitude, area, quality, first cycle (C = 10) | G · 500 m sinusoidal, annual | 2001–**2025** (v061; 315 tiles per year; 2025 double-lists 311 of them and the highest production is kept; CMR declares a constant 274.756 MB per granule) | ≈ 10 GB per year (probe #293 replaces it) | NASA open · public | **A** | EDL; probe #293 in flight |
| `gbif` | **three billion species observations.** GBIF occurrence snapshot (parquet on the AWS Open Data registry, monthly): records with a date to the day or month and coordinates; time from the event date (noon when no time; f_t = one day, or one month for month-precision records); footprint from the coordinate uncertainty; platform = the hashed taxon key; kingdom code, class code, basis-of-record code, individual count (C = 4); `platforms.json` maps the hash to the taxon key and name; schema 3 | P · point; per-row footprint | 1600s →; **3.67 × 10⁹** records with coordinates, a year and no geospatial issue (GBIF API, 2026-09-20; the 2026-09-01 snapshot is 9,899 parquet objects, 285.3 GB) | **probe run here** (eight parts of the 2026-09-01 snapshot): 2.592 × 10⁹ rows / **101.1 GB public**, 7.17 × 10⁸ rows / **28.0 GB private** (`gbif_nc`), 68 GB of transfer to build either; the two tracks are two adapters from one snapshot | per record: CC0 (11.0 %) and CC BY 4.0 (72.2 %) → public; CC BY-NC 4.0 → private | **A** | none (anonymous S3) |
| `cat_palsar` (E5) | **Japanese L-band radar mosaics, yearly.** JAXA PALSAR/PALSAR-2 25 m global mosaics v2.6, HH and HV; ScanSAR L2.2 scenes — as a catalogue, pixels at the source | T · 25 m, annual | 2007–2010, 2015–2025 | catalogue < 1 GB | JAXA research terms, © JAXA · public catalogue | **A** (was B); the ScanSAR L2.2 half lists anonymously on AWS Open Data (`s3://jaxaalos2`, STAC items already written, 5,716 for 2025) — adapter landed | the 25 m mosaics answer 401: **a JAXA G-Portal account — needed from Chris** for that half only |
| `cat_biomass_esa` | **ESA's P-band radar (the BIOMASS satellite, launched 2025-04-29).** Catalogue of the released L1/L2 products, if public | T | 2025 → | catalogue | ESA terms (redistribution unconfirmed) | **A** — the products ARE publicly listed through FedEO (27 collections; on 2026-09-20 L1A 417,521 · L1B 313,329 · L1C 126,785 · L2A 25,792 · L2B 0); adapter landed, catalogues L2B for the day it publishes | none (FedEO is anonymous; `eocat.esa.int` resets, `biomass-pdgs` 502s, CDSE has no BIOMASS collection) |

`lossyear`, `fire`, `flux`, `burned500`, `alerts`, `chirps05`, `lst05`,
`snow05` and `refl05` already carry the rest of the land biosphere; `flux` is
fixed and built in this wave (§4).

### 2.2 Family 1.gf — the ocean's biosphere

`oc4k` (OC-CCI 4 km daily colour, built), `bgcargo` (floats with oxygen,
nitrate, pH, chlorophyll, built), `glodap` and `wod` (bottle chemistry and
profiles, built) are the ocean biosphere; wave 6 adds one store:

| store | what it is | tier · footprint | record | stored† | licence | phase | credential |
|---|---|---|---|---|---|---|---|
| `pace4k` | **hyperspectral ocean colour and phytoplankton community.** PACE OCI daily 4 km mapped files (OB.DAAC; measured on CMR 2026-09-20): L3M BGC v3.2 (chlorophyll, particulate organic carbon, phytoplankton carbon), L3M AOP v3.2 (apparent visible wavelength) and L4M MOANA v3.2 (Prochlorococcus, Synechococcus, picoeukaryote concentrations) — three files a day, one store (C = 7); the three MOANA channels are a level-4 derived product and are flagged as such in `qc`, admitted because no other source carries the community composition | G · 4 km daily, sharded | 2024-03-05 → (measured on three days: 25–27 MB a frame, valid 0.084–0.091 for the four global channels; MOANA's file is REGIONAL, 70 S–70 N, 85 W–25 E, placed at its measured offset) | **≈ 9.3 GB a year**, ≈ 23 GB so far | NASA open · public | **A** | **none** — OB.DAAC's OPeNDAP serves the bundles anonymously with variable subsetting (46–54 MB a day); probe #295 in flight for memory, not credentials |

Why 4 km bins and not the 1.2 km swath: the level-2 swath at 1.2 km is
≈ 1 TB a year of cells (2.5 × 10⁸ ocean cells a day, a quarter clear), for a
product whose value is the *spectrum and the community*, which the 4 km
daily bins keep; the swath is a phase C row. This is the one daily binned
product admitted beside `oc4k`, because OC-CCI v6 does not yet merge PACE.
Marine GBIF records ride `gbif`.

### 2.3 Family 0.9.tf

Inherits every 1.0.tf store above through `inherits`; nothing new to build.
The AlphaEarth embedding it holds was itself trained against GEDI, so the
canopy read-out is where 0.9.tf's cache and the programme's own tile codec
are compared with everything else fixed (paper §8, comparison (4)).

## 3 · Admission, in the notes' own rule

Raw observations first: `gedi`, `icesat2`, `sif`, `gbif`, `lai500`, `pace4k`
are measurements. `canopy30`, `biomass100`, `pheno500` are derived maps
admitted as **targets**, the way `lossyear` is: each is a fit to a
measurement the family also holds (GEDI, radar, MODIS reflectance), and each
is what a user will ask the model for. `canopy_ref` and the two radar
catalogues are references, not copies. Every store must lower the error on
the land-biosphere read-out (or one of the four ocean read-outs) under
comparison (4), or be dropped — the rule is unchanged.

## 4 · Fixes that ride with the wave

- **`flux`** — done 2026-09-20. 27 AmeriFlux towers ship no BADM file and no `UTC_OFFSET`, and
  `--allow-missing-years` was inert for one-stream stores at
  `build_family10_stores.py:3980` (the branch withheld every year marker
  unconditionally, so a "successful" run left the ledger empty). Fix: derive the offset from longitude
  (`round(lon/15)` hours) for a tower without one, record it in
  `platforms.json` with `utc_offset_source: "longitude"`, count it; make
  `--allow-missing-years` honoured for one-stream stores. Then `stage=all`.
- **`burned500`** — monthly frames on the five-day axis: a third skip reason
  in `OUTSIDE_RECORD` so a bin with no frame of its own writes no shard.
  Done: `sharded.FRAME_NO_FRAME_IN_BIN`, distinct from `absent_upstream` and
  counted by name; the adapter uses blocks of nine sinusoidal tiles (46
  groups) because one group per tile is 166,160 files, past the Hub's
  100,000-per-repository limit.
- **`alerts`** — the DIST-ANN (annual) route first, as the handover proposes;
  DIST-ALERT later with a granule filter.
- **The three registries** — done: `ml/build_family1_registry.py --check`
  (46 adapters; 1.gf 13 stores, 1.0.tf 30, 0.9.tf inherits). `family1gf.json`, `family1tf.json`,
  `family09tf.json` (with `inherits`), written from the adapters' declared
  metadata and the Hub's `store.json` files by a registry builder modelled on
  `ml/build_family10_registry.py`; a store not on the Hub is listed with its
  design row and `built: false`.

## 4b · State on 2026-09-20, evening

All fourteen adapters landed in one wave with smokes (607 tests green over
the family-10 and family-1 suites); three probes ran from the sandbox (`sif`,
`gbif`/`gbif_nc`, `canopy30`) and five are on hosted runners (#293 pheno500,
#294 lai500, #295 pace4k, #296 gedi, #297 icesat2). The measured numbers and
every place the sources contradicted this design are in
`ml/family1/BUILD_LOG.md`, "Notes — E-082 wave 6". Two framework facts came
out of it: tier P has no per-row footprint column (a `row_fp` hook is the
follow-up; `gbif` carries its footprint in `qc`'s upper bits meanwhile), and
a single-bin tier-G store (`canopy30`, `lossyear`) cannot be laned by year,
so it goes to a box until the parts layout admits several lanes writing
disjoint groups into one year.

## 5 · Order of work and who does it

Main session: this design; the paper (§1 read-outs, §2 tiers and a biosphere
table, §3 the land cone, §4 annual targets, §6 the canopy query, §8
comparison (4) and its Results row, related work). Opus agents, launched
together:

1. **Notes** — the rows of §2 into `family1tf.tex` (new subsection "The
   biosphere" after the LiDAR subsection; phases of `gedi`, `icesat2`,
   `biomass100`, `cat_palsar` moved; `canopy30`, `canopy_ref`, `sif`,
   `lai500`, `pheno500`, `gbif`, `cat_biomass_esa` added; exception E7 named
   in the rule section), `family1gf.tex` (`pace4k`), `family09tf.tex` (the
   inheritance sentence and the canopy comparison), and the three
   `_summary.tex` files; PDFs rebuilt.
2. **Adapters, measurement first** — `sif` (TROPOMI), `gbif`, `pace4k`,
   `canopy30`, `pheno500`, `lai500`, `gedi`, `icesat2`, `canopy_ref`,
   `cat_biomass_esa`, each with its synthetic smoke (exact expected counts)
   and its `--stage probe` dispatched on a hosted runner; `cat_palsar` waits
   for the account. No full build is dispatched before its `probe.json` is
   read.
3. **Fixes and registries** — §4.

Verification before anything is called built is E-082's (`check_store` after
a restore, counts equal to the fetch ledger, the probe's numbers in the
registry beside the estimate). Costs: hosted runners $0; the two LiDAR
stores' full records are the first family-1 stores to need a box for
*fetching* as well as assembling, and that is decided on their probes.

## 6 · Decisions for Chris

1. **JAXA account** for `cat_palsar` (free registration at G-Portal; or the
   store waits).
2. **GES DISC approval** in the Earthdata profile — unblocks `xco2`, `irtb`
   and the OCO half of `sif` in one click.
3. **GBIF's non-commercial records**: the private track as designed, or drop
   them (the public CC0/CC BY records are 83.2 % of the dated, georeferenced
   snapshot; measured).
4. **ESA CCI Biomass terms**: the terms file (read 2026-09-20) says the data
   "may be used by any user for any purpose" with acknowledgement and
   citation; redistribution is neither granted nor forbidden in words. The
   pick is the public track with attribution; say if you want the private one.

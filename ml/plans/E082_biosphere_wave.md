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
| `gedi` | **laser canopy height, cover and biomass from the Space Station.** GEDI (Global Ecosystem Dynamics Investigation) L2A + L2B v003 (LP DAAC) joined to L4A v2.1 (ORNL DAAC) by shot number: ground elevation, RH25/50/75/90/98 (the heights below which 25–98 % of the returned laser energy lies), canopy cover, plant-area index, above-ground biomass and its error, quality (C = 11) | P · 25 m footprints every 60 m along 8 tracks, ±51.6°; log2 f_p = −10.1, f_t instantaneous | 2019-04 → 2023-03 and 2024-04 → 2025-07 released; 4–10 × 10⁹ quality shots | 0.2–0.5 TB | NASA open · public | **A** (was B) — phase A builds one year (2022) and measures; the rest follows the probe | EDL (in place). The probe decides whether OPeNDAP variable subsetting or whole-granule HDF5 is the fetch; L2A granules are 106 TB native, so the fetch must be subset or streamed, never kept |
| `icesat2` | **laser terrain and canopy heights from a polar satellite.** ICESat-2 ATL08 release 007 (NSIDC): terrain height, canopy height (h_canopy, the 98th-percentile relative height), canopy cover, photon counts, segment quality per 100 m segment (C = 6) | P · 100 m segments, 6 beams, ±88°; log2 f_p = −8.1 | 2018-10 →; ≈ 4 × 10⁹ land segments a year | ≈ 1.1 TB for the record | NASA open · public | **A** (was C) — one year (2022) in phase A; the rest phase B by size | EDL (NSIDC in place) |
| `canopy30` | **the 2020 canopy-height map at 30 m.** GLAD's forest height 2020 from the Global Land Cover and Land Use Change 2000–2020 set (Potapov et al. 2022; the method is Potapov et al. 2021, Landsat fitted to GEDI); uint8 metres, codes for water/no-data (C = 1); annual sharded grid on the Hansen 10° tile grid cut into 2.5° groups exactly as `lossyear` | G · 30 m, annual (one bin) | 2020 only; **261 tiles** (measured 2026-09-20, 0.33–0.61 GB each, ≈ 120 GB† native) | ≈ 60–100 GB (dense metres) | CC BY (GLAD states free redistribution with citation) · public | **A** | none (GLAD storage, public) |
| `canopy_ref` | **the finer canopy maps, catalogued, pixels at the source.** ETH Global Canopy Height 10 m 2020 (Lang et al. 2023; height and its standard deviation; 3 TB native) and Meta/WRI 1 m canopy height (Tolan et al. 2024; AWS `dataforgood-fb-data`, tens of TB): one row per tile with footprint, sensor code and asset URL, the tier-T catalogue form | T · 10 m and 1 m | 2020 (ETH), 2018–2020 composite (Meta) | catalogue < 1 GB | CC BY 4.0 · public | **A** (catalogue); the tile codec reads the pixels in E-078c | none |
| `biomass100` | **yearly above-ground biomass maps, 100 m.** ESA CCI Biomass v7.0: biomass and its standard deviation (C = 2, uint16 Mg/ha), years 2005–2012 and 2015–2024; GEDI L4B 1 km gridded biomass 2019–2023 as a second group (C = 2) | G · 100 m annual; L4B 1 km | annual | ≈ 30 GB per year at 100 m; L4B < 1 GB | ESA CCI terms (verify) · public if the terms allow, else private | **A** for 2020 and the L4B group (was C); other years phase B | none (CEDA) |
| `sif` | **photosynthesis seen from orbit.** Solar-induced chlorophyll fluorescence: TROPOMI (Sentinel-5P) ungridded daily soundings (Caltech, Köhler et al. 2018; and ESA TROPOSIF from the S5P-PAL service): SIF at 740 nm and its daily-corrected value, 1-σ, cloud fraction, solar and viewing zenith (C = 6); OCO-2 and OCO-3 SIF Lite (GES DISC) as second and third platforms with the same channels | P · 7 × 3.5 km (TROPOMI), 1.3 × 2.25 km (OCO); log2 f_p = −2.5 / −4.0 (geometric-mean support over 27.83 km); instantaneous | 2018-04-30 → (TROPOMI; measured on S5P-PAL), 2014-09 → (OCO-2), 2019-08 → (OCO-3); ≈ 5 × 10⁸ clear soundings a year | 60–160 GB for TROPOMI depending on the cloud cut, which the probe measures (Caltech's four yearly tarballs 2018–2022 are 181.6 GB gzipped, measured) | Caltech: CC0; ESA: Copernicus open; NASA open · public | **A** (TROPOMI now; OCO after the GES DISC approval) | none for TROPOMI; EDL + GES DISC for OCO |
| `lai500` (E7) | **leaf area and light absorbed, every 8 days.** MOD15A2H v061 (Terra), MYD15A2H (Aqua), VNP15A2H (VIIRS) 8-day leaf-area index, FPAR (fraction of absorbed photosynthetically active radiation), their standard deviations and quality (C = 4); the 8-day composite is an exception E7 to the five-day rule, carried as a `frame_table` like `chirps05`'s pentads | G · 500 m sinusoidal tiles, sharded on the native tile grid (the layout already holds a projected grid: `seaice_asi` is polar stereographic) | 2000-02-18 → (Terra, v061), 2002-07-04 → (Aqua), 2012-01-01 → (VIIRS, v002); 46 composites a year, **293 tiles** per composite (measured from CMR: 13,466 Terra granules in 2020) | ≈ 60 GB per sensor-year after compression | NASA open · public | **A** builds Terra 2020–2024 and measures; the record phase B | EDL (LP DAAC in place) |
| `pheno500` | **the leaf calendar, per year.** MCD12Q2 v061 land-surface phenology: greenup, mid-greenup, peak, senescence, mid-greendown, dormancy dates (day of year, int16), EVI minimum, amplitude, area, quality, first cycle (C = 10) | G · 500 m sinusoidal, annual | 2001–**2025** (v061; 315 tiles per year, measured) | ≈ 10 GB per year | NASA open · public | **A** | EDL |
| `gbif` | **three billion species observations.** GBIF occurrence snapshot (parquet on the AWS Open Data registry, monthly): records with a date to the day or month and coordinates; time from the event date (noon when no time; f_t = one day, or one month for month-precision records); footprint from the coordinate uncertainty; platform = the hashed taxon key; kingdom code, class code, basis-of-record code, individual count (C = 4); `platforms.json` maps the hash to the taxon key and name; schema 3 | P · point; per-row footprint | 1600s →; **3.67 × 10⁹** records with coordinates, a year and no geospatial issue (GBIF API, 2026-09-20; the 2026-09-01 snapshot is 9,899 parquet objects, 285.3 GB) | ≈ 130 GB public; the CC BY-NC records (**16.8 %** measured) on the private track | per record: CC0 (11.0 %) and CC BY 4.0 (72.2 %) → public; CC BY-NC 4.0 → private | **A** | none (anonymous S3) |
| `cat_palsar` (E5) | **Japanese L-band radar mosaics, yearly.** JAXA PALSAR/PALSAR-2 25 m global mosaics v2.6, HH and HV; ScanSAR L2.2 scenes — as a catalogue, pixels at the source | T · 25 m, annual | 2007–2010, 2015–2025 | catalogue < 1 GB | JAXA research terms, © JAXA · public catalogue | **A** (was B) | **a JAXA G-Portal account — needed from Chris** |
| `cat_biomass_esa` | **ESA's P-band radar (the BIOMASS satellite, launched 2025-04-29).** Catalogue of the released L1/L2 products, if public | T | 2025 → | catalogue | ESA terms (verify) | **A** if products are public; else a phase C row with the reason | to verify |

`lossyear`, `fire`, `flux`, `burned500`, `alerts`, `chirps05`, `lst05`,
`snow05` and `refl05` already carry the rest of the land biosphere; `flux` is
fixed and built in this wave (§4).

### 2.2 Family 1.gf — the ocean's biosphere

`oc4k` (OC-CCI 4 km daily colour, built), `bgcargo` (floats with oxygen,
nitrate, pH, chlorophyll, built), `glodap` and `wod` (bottle chemistry and
profiles, built) are the ocean biosphere; wave 6 adds one store:

| store | what it is | tier · footprint | record | stored† | licence | phase | credential |
|---|---|---|---|---|---|---|---|
| `pace4k` | **hyperspectral ocean colour and phytoplankton community.** PACE OCI daily 4 km mapped files (OB.DAAC; measured on CMR 2026-09-20): L3M BGC v3.2 (chlorophyll, particulate organic carbon, phytoplankton carbon), L3M AOP v3.2 (apparent visible wavelength) and L4M MOANA v3.2 (Prochlorococcus, Synechococcus, picoeukaryote concentrations) — three files a day, one store (C = 7); the three MOANA channels are a level-4 derived product and are flagged as such in `qc`, admitted because no other source carries the community composition | G · 4 km daily, sharded | 2024-03-05 → | ≈ 50 GB per year | NASA open · public | **A** | EDL (OB.DAAC in place) |

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

- **`flux`** — 27 AmeriFlux towers ship no BADM file and no `UTC_OFFSET`, and
  `--allow-missing-years` is inert for one-stream stores at
  `build_family10_stores.py:3980`. Fix: derive the offset from longitude
  (`round(lon/15)` hours) for a tower without one, record it in
  `platforms.json` with `utc_offset_source: "longitude"`, count it; make
  `--allow-missing-years` honoured for one-stream stores. Then `stage=all`.
- **`burned500`** — monthly frames on the five-day axis: a third skip reason
  in `OUTSIDE_RECORD` so a bin with no frame of its own writes no shard.
- **`alerts`** — the DIST-ANN (annual) route first, as the handover proposes;
  DIST-ALERT later with a granule filter.
- **The three registries** — `family1gf.json`, `family1tf.json`,
  `family09tf.json` (with `inherits`), written from the adapters' declared
  metadata and the Hub's `store.json` files by a registry builder modelled on
  `ml/build_family10_registry.py`; a store not on the Hub is listed with its
  design row and `built: false`.

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

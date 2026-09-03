# What feeds the cone codec today — and what the catalog could predict El Niño 2026 with

Companion page to slides 45, 46, 47 and 50 of the geospatial representation-model survey deck
(`ml/figures/geofm_survey/`). Written 3 Sep 2026.

Four questions, answered from the repository rather than from memory:

1. **What data actually reaches the cone codec** — the encoder of experiment E-069, which
   reads a cone-shaped neighbourhood in space and time around one ocean point and turns it
   into one embedding vector — and what the honest ranked list of missing ocean and
   biosphere inputs is.
2. **What the 274-record open-data catalog (`data/catalog.json`) can and cannot say about the
   El Niño now developing in the Pacific**, and what to add.
3. **What the sampler does at every boundary it meets** — the edge of the window, land, the ends
   of the time axis, the held-out years, and the calendar month — and which of those edges is
   handled and which is not.
4. **Which propagation speeds the cone can and cannot represent**, and what that costs.

Every acronym is spelled out the first time it appears (root `CLAUDE.md` §0c). Numbers with a
`file:line` reference were checked against that file; numbers marked *general knowledge* were
not, and are flagged.

---

## Part 1 · The dataset inventory of the cone codec

### 1.1 · What the tensor is

`family4_na025_pentad_r3.npz` — the input array the cone codec is trained on. "family 4" is
the fourth generation of input tensor in this project; "recipe r3" is its third revision, the
one that added the two ocean-current direction channels the cone needs.

- Shape `[T = 3142, H = 281, W = 481, C = 42]`, float16, 5.30 GB.
- Grid: 0.25°, point-aligned, latitude 0–70° N, longitude 100° W–20° E (the North Atlantic
  window). 84,405 cells are ocean.
- Time axis: fixed 5-day bins ("pentads") from 1982-01-01 to 2024-12-31.
- Built once and pinned by its sha256 fingerprint `fa460837fa…`, by run #535 — the job that
  built and published the r3 tensor (`ml/plans/E069_HANDOVER.md:319-330`).

There are **four upstream data products** and **two label series**. That is the whole input
diet.

### 1.2 · The four products, one block each

**S1 · GLORYS12V1 — Copernicus Marine global ocean reanalysis**
*A reanalysis is a model run repeatedly nudged towards whatever measurements exist, so that it
produces a complete, gap-free history.*

- Producer: Mercator Ocean International / Copernicus Marine. Credentialed download.
- Variables used: `uo`, `vo` (eastward and northward current), `mlotst` (mixed-layer depth),
  `zos` (sea-surface height) — the 0–1 m slice only.
- Native: 1/12°, daily, 50 depth levels. As used: North Atlantic window, binned to 0.25° and
  to pentad means.
- Span used: 1993-01-01 … 2024-12-31.
- Role: **input**, channels 0, 1, 2, 40, 41.
- Refs: `ml/fetch_glorys_daily.py:22-28,96` → `ml/aggregate_cadence.py:19-40` →
  `ml/build_family4.py:43-51,83-96`.

**S2 · NOAA OISST v2.1 — Optimum Interpolation Sea Surface Temperature**
*An analysis, not a reanalysis: satellite and in-situ measurements interpolated onto a grid,
with no ocean model involved.*

- Producer: NOAA NCEI, distributed by NOAA's Physical Sciences Laboratory.
- Variable: `sst`. Native 0.25°, daily, 1981-09 →.
- As used: bilinearly interpolated onto the tensor's own axes, stored as int16 at 0.01 °C,
  then averaged to pentads with NaN awareness.
- Span used: 1982-01-01 … 2024-12-31 — the only channel live on the whole axis.
- Role: **input**, channel 39.
- Refs: `ml/fetch_sst_na.py:35-65,105-110` → `ml/build_family4.py:72-82,562-640`.

**S3 · Roemmich–Gilson Argo climatology (Scripps Institution of Oceanography)**
*Argo is the global array of ~4,000 free-drifting floats that dive to 2,000 m and report
temperature and salinity every ten days. Roemmich–Gilson maps those profiles onto a grid.*

- Variables used: temperature and salinity, mean plus anomaly, at 16 of the product's 58
  pressure levels (10 … 1900 decibars — roughly 10 m to 1900 m depth).
- Native 1°, monthly, 2004 →. As used: bilinear 1° → 0.25°; **one live pentad per month** (the
  bin containing the 15th), the `missing` token in the other five. Forward-filling was
  rejected on purpose: it would tell the model the subsurface was observed on days it was not.
- Span used: 2004-01 … 2024-12.
- Role: **input**, channels 3–34 — 32 of the 42 channels and about 80 % of the bytes.
- Refs: `ml/build_family3.py:64-71,180-188`; `ml/fetch_rg_channels.py:10-15`;
  `ml/build_family4.py:52-62,288-357`.

**S4 · NCEP/NCAR Reanalysis 1 — surface momentum flux**
*The first-generation atmospheric reanalysis. Its surface wind stress is diagnosed by the
model, not measured.*

- Variables: `uflx`, `vflx`, sign-negated so that positive means stress **on** the ocean.
- Native ~1.9° Gaussian grid, daily, 1948 →. **The archive is frozen: updates ended March
  2026.** As used: bilinear to 0.25°, then per pentad both the mean (`tau_x`, `tau_y`) and the
  within-pentad standard deviation (`tau_x_std`, `tau_y_std`), computed from the daily fields
  because a standard deviation cannot be re-derived from a mean.
- Span used: 1982-01 … 2024-12 — live on the whole axis.
- Role: **input**, channels 35–38.
- Refs: `ml/build_family4.py:63-71,361-448`; `ml/fetch_wind_channels.py:11-18`.

**S5 · GREP 0.25° ensemble reanalysis — grid reference only**

In recipe r3 this product supplies nothing but the latitude and longitude vectors, which are
asserted to match. It is the ancestor of the older monthly family-3 tensor.
Refs: `ml/fetch_cmems025.py:7-12,36-38`; `ml/build_family4.py:258-284`.

**S6 · RAPID-MOCHA-WBTS at 26.5° N — a label, never an input**

The moored array that measures the Atlantic Meridional Overturning Circulation (AMOC — the
basin-scale flow that carries warm water north near the surface and cold water south at
depth) as a transport in sverdrups. 12-hourly, 2004-04 →; averaged to pentads, guarded to the
range −15 … 45 Sv. Stored in the tensor file as `truth_rapid` and **never read by
`ml/train_cone.py`**. Refs: `ml/build_truth_pentad.py:56,131-160` →
`ml/build_family4.py:669-690`.

**S7 · Florida Current cable transport at 27° N — a label, never an input**

A submarine telephone cable measures the voltage the Florida Current induces, which converts
to a transport. Daily, 1982 → — the only label spanning the whole time axis. Stored as
`truth_fc`. Ref: `ml/build_truth_pentad.py:57,163-186`.

*OSNAP, MOVE and SAMBA appear in `ml/fetch_truth.py:13-20` but only for the older monthly
tensor families; they are not in r3.*

### 1.3 · The 42 channels

| idx | channel | source | live bins |
|---|---|---|---|
| 0 | `cur_speed` | S1 GLORYS12 | 1993 + |
| 1 | `log_mld` | S1 | 1993 + |
| 2 | `ssh` | S1 | 1993 + |
| 3–18 | `rg_t10` … `rg_t1900` | S3 Argo | 2004 +, 1 bin in 6 |
| 19–34 | `rg_s10` … `rg_s1900` | S3 | 2004 +, 1 bin in 6 |
| 35 | `tau_x` | S4 NCEP R1 | all |
| 36 | `tau_y` | S4 | all |
| 37 | `tau_x_std` | S4 | all |
| 38 | `tau_y_std` | S4 | all |
| 39 | `sst` | S2 OISST | all |
| 40 | `cur_u` | S1 | 1993 + |
| 41 | `cur_v` | S1 | 1993 + |

Derivations worth knowing:

- `cur_speed` = `hypot(cur_u, cur_v)` computed from the **binned** components, so it is not
  independent of channels 40 and 41. Averaging speeds instead would inflate quiet bins where
  the current reverses (`ml/build_family4.py:43-51`).
- `log_mld` = `log10(mlotst)`; base 10 was measured against family 3, not assumed.
- Channel lists: `ml/build_family3.py:65-73`; `ml/build_family4.py:198-204`.
- Token budget per anchor point: 42 patch tokens + 706 dot tokens = **748**
  (`ml/plans/E069_cone_codec.md:96-103`).

### 1.4 · What training derives for itself

- **Anomaly**: each channel minus a per-calendar-month average computed from the **training
  years only**, then z-scored (divided by its own spread) on the training set. No external
  climatology is used (`ml/trainprobe.py:140-143`; `ml/train_cone.py:289-322`).
- **Ocean mask**: cells where channel 0 is finite in any bin → 84,405 cells
  (`ml/train_cone.py:318`).
- **Holdout**: calendar years 2009, 2017 and 2023, under the window-scope pool
  (`ml/train_cone.py:72-118`); the pool self-certificate reported 0 violations in 4,096
  anchors.
- **`missing` vs `mask` tokens** come from the tensor's own NaN pattern; the two are
  deliberately distinct, because "never observed" is information.
- **Velocity probe (hypothesis H1)**: the target is the anchor's own `cur_u`, `cur_v` in
  anomaly space, with the current channels hidden at encode time; ridge regression, year-
  blocked folds (`ml/train_cone.py:640-680`).
- **Snapshot twin**: an identical cone codec trained with `L_in = 0` — no lags at all — as the
  control.

### 1.5 · Pinned for stage 2, not yet consumed

- The run-415 control embedding on recipe r2 (the E-064b control).
- The linear inverse model null from E-066 — a plain linear forecast fitted to the same
  field — over 8 scoreable channels (`ml/lim_baseline.py:34-82`), with climatology,
  persistence and damped-persistence references beside it.
- The AMOC evaluation mask `data/amoc_eval_mask.json`: 84,405 rolled / 29,627 corridor / 265
  section pixels (`ml/rollout_spatial.py:532-553,599-670`).

---

## Part 2 · The derivation graph, as an edge list

`[R]` = stated in this repository, with the reference. `[K]` = general knowledge about the
product, not asserted anywhere in the repo.

### 2.1 · Upstream — instruments into products

- satellite altimetry + satellite and in-situ sea-surface temperature + Argo/XBT/CTD profiles
  → **GLORYS12**, by assimilation on a 7-day cycle — `[R]`
  `ml/figures/geofm_survey/GENERIC_EMBEDDING_INPUTS.md:26,28`; `docs/COMBINING_DATASETS.md:41-44`
- AVHRR radiances + ships, buoys and Argo → **OISST v2.1** — `[R]` `ml/fetch_sst_na.py:23`;
  `ml/plans/E042_sst_channel.md:154`; `docs/COMBINING_DATASETS.md:37-38`
- Argo floats → **Roemmich–Gilson 1° monthly climatology** — `[R]`
  `ml/fetch_rg_channels.py:10-15`; `ml/plans/DATA_LADDER.md:66-67`
- radiosondes, aircraft, ship and satellite observations → **NCEP R1** → model-diagnosed
  `uflx`/`vflx` — `[R]` `ml/fetch_wind_channels.py:11-14`; `ml/plans/DATA_LADDER.md:182-183`;
  `docs/CATALOG.md:39` ("updates ended")
- moorings at 26.5° N + Florida Current cable + an Ekman term → **RAPID overturning
  transport** — `[R]` identity stated in `ml/classical_baseline.py:9-15`
- cable voltage → **Florida Current transport** — `[R]` `ml/fetch_truth.py:10-12`

### 2.2 · In-repo — products into channels

- GLORYS12 daily North Atlantic → 0.25° binning + pentad means
  (`ml/aggregate_cadence.py:13-40`) → Hugging Face `pentad025/` → `cur_speed` (hypotenuse),
  `log_mld`, `ssh`, `cur_u`, `cur_v` — `[R]`
- OISST daily → bilinear onto the tensor axes → `sst_na025/` → pentad mean → `sst` — `[R]`.
  **Sibling, not parent:** `scripts/bake_sst_daily.py` (the globe app's daily SST bake) shares
  only the download — `[R]` `ml/plans/E042_sst_channel.md:145-151`
- `RG_ArgoClim` + monthly updates → 16 pressures → bilinear 1° → 0.25° → only the pentad
  holding the 15th; forward-fill **rejected** — `[R]` `ml/plans/E034_pentad_tensor.md:88-100`
- NCEP R1 `uflx`/`vflx` → sign flip → bilinear → pentad mean **and** within-pentad σ — `[R]`
- GREP `base025_na.npz` → latitude/longitude vectors only (a refusal precondition) — `[R]`
- RAPID + Florida Current → `truth_pentad.npz` → `truth_rapid` / `truth_fc` keys in r3, not
  read by the trainer — `[R]`
- r3 → training-years climatology → anomaly → z-score → ocean mask → holdout + certificate →
  cone sampler dots → ConeMAE → `cone_codec.pt`, `velocity_probe.json` — `[R]`
- r2 (channels 0–39 bit-identical to r3) → run-415 PixelMAE → control embedding; → linear
  inverse model null; → `amoc_eval_mask.json` — `[R]`

### 2.3 · Edges that do NOT exist (and why that matters)

- family 6 (the CMIP6 pre-industrial-control pretraining corpus) → E-069: **no**. E-069 does
  not touch it.
- family 7 (the first global tensor, E-070) → E-069: **no** — the dependency runs the other
  way; E-070 is gated on E-069.
- the globe app's WMO 1991–2020 sea-surface-temperature normal → the anomaly baseline:
  **deliberately severed**, because that normal averages over the held-out years and would
  leak the test period into training — `[R]` `ml/fetch_sst_na.py:27-33`.

### 2.4 · The leakage edge that does exist

Derived products look forward as well as backward: a value dated day *t* was built from
observations on both sides of *t*. GLORYS12's assimilation window is 7 days, so five of the
42 channels (`cur_speed`, `cur_u`, `cur_v`, `log_mld`, `ssh`) carry information from up to
~3.5 days after their nominal time and need an effective-time shift —
`[R]` `ml/figures/geofm_survey/GENERIC_EMBEDDING_INPUTS.md:28,175`.

---

## Part 3 · Missing from "the ocean and its biosphere" — additions, ranked

Ordered by information per byte and by what the model cannot derive from what it already
holds. Items 1, 2 and 4 are the top of `ml/plans/DATA_LADDER.md` §7; the rest follow from the
same document's sections C1–C4 and from the input-ladder proposal.

**1 · Buoyancy forcing — ERA5 surface heat and freshwater fluxes.**
Net shortwave and longwave radiation at the surface (`msnswrf`, `msnlwrf`), sensible and
latent heat (`msshf`, `mslhf`), and evaporation minus precipitation (`mer`, `mtpr`) — plus
wind-stress curl and Ekman pumping, which are finite differences of channels we would then
hold. This is **the only forcing entirely absent today**: the tensor can see the wind pushing
the ocean but not the sky heating, cooling, freshening or salting it, and the subpolar
salinity anomaly is the classic AMOC precursor. It also replaces the frozen NCEP R1 momentum
flux with a 0.25°-native product on the project's own grid. Net storage after retiring the
four NCEP channels: **+5 GB**. Blocked only on a free Climate Data Store account
(`ml/plans/DATA_LADDER.md:181-205`).

**2 · The ocean interior at 0.25° in three dimensions — GREP.**
Temperature, salinity and both velocity components at 8 depth levels, 1993 →, about +20 GB.
Same product, same credentials, same grid and same code path as a download the project already
makes. It buys velocity at depth, which the tensor has never had, eleven extra years, and
**six times as many live subsurface bins** — because it is a real gridded field rather than a
monthly climatology live one pentad in six. The data ladder ranks it #1: the best ratio of new
physics to new risk in the document (`ml/plans/DATA_LADDER.md:76-113,265-270`).

**3 · Observed sea level — DUACS altimetry.**
Sea-level anomaly, absolute dynamic topography and the geostrophic velocities derived from
them, 0.125°, daily, 1993 →. Today the tensor's `ssh` is GLORYS's *fitted* sea surface; DUACS
is the *observation it was fitted to*, and carrying both gives the model an independent
channel and a way to tell fit from data. Delayed-time DUACS uses a centred ±6-week window of
altimeter tracks, so it must be time-shifted or replaced by the near-real-time stream for the
forecasting objective.

**4 · Statics — the geometry the ocean runs in.**
Bathymetry and its gradient, an explicit land/ocean mask (so that "land" and "not observed"
stop sharing a token), the Coriolis parameter *f* and its meridional gradient β, distance to
coast and to the 1000 m isobath, and mean dynamic topography. One static channel is 0.27 MB;
ten cost **~3 MB**. The western boundary, the Mid-Atlantic Ridge and the Greenland–Scotland
sills *are* the overturning's geometry, and none of it is inferable from the channels we hold.
Ranked #2 (`ml/plans/DATA_LADDER.md:224-234`).

**5 · Sea-ice concentration — OSTIA or OSI SAF.**
The northern boundary condition of a window that reaches 70° N. Today ice-covered water enters
the tensor as ordinary water. Free alongside the OSTIA sea-surface temperature download
(`ml/plans/DATA_LADDER.md:167-171`).

**6 · Drifters — the Global Drifter Program.**
About 1,300 surface buoys reporting 15 m velocity and sea-surface temperature. These are the
*observed* velocities against which the H1 velocity probe — "does the embedding know which way
the water is moving?" — should really be scored; scoring it against GLORYS's own currents
tests the codec against the same model that produced its inputs.

**7 · The biosphere — as a target first, an input second.**
Ocean-colour chlorophyll (ESA OC-CCI at 4 km, 1997 → 2024-12; NASA PACE from 2024), BGC-Argo
oxygen, nitrate and chlorophyll profiles as native dots, and SOCAT surface CO₂ observations.
The data ladder explicitly **rejected ocean colour as an input** for the overturning question —
"no mechanistic path at these timescales" (`ml/plans/DATA_LADDER.md:239`) — and that rejection
is exactly what makes the biosphere the right *held-out sphere*: something the embedding must
predict without ever having seen it, which is the cleanest available test of whether the
representation is generic rather than AMOC-shaped.

**8 · Sea-surface salinity, and RAPID's unused files.**
SMOS (2010 →) and SMAP (2015 →) salinity are short next to a 43-year axis — which is why the
ladder skipped them — but they are the freshwater lid the buoyancy forcing acts through, and
worth revisiting once item 1 exists. Free and immediate, meanwhile: the RAPID distribution
already shipped `moc_vertical`, the overturning streamfunction as a function of depth (a
depth-resolved *label* that pairs with depth-resolved *state* from item 2), and `ts_gridded`,
observed boundary temperature and salinity for validating those depth channels rather than
trusting them (`ml/plans/DATA_LADDER.md:216-222`).

**And before any of it:** moving the 32 Argo channels to a 1° sidecar frees **27 GB** — more
than items 1, 3, 4 and the 2025–26 continuation cost put together
(`ml/plans/DATA_LADDER.md` §2).

---

## Part 4 · El Niño 2026

### 4.1 · What El Niño is, in one paragraph

The El Niño–Southern Oscillation (ENSO) is a see-saw in the tropical Pacific. Normally the
trade winds blow east to west, piling warm water up near Indonesia and letting cold water rise
off South America. When the trades weaken, that warm pile sloshes back east, the eastern
Pacific warms, and the world's rainfall patterns move with it. **Niño-3.4** is the agreed
thermometer: the average sea-surface temperature anomaly in the box 5° S–5° N, 170° W–120° W.
The **Oceanic Niño Index (ONI)** is the official version of that number — a three-month
running mean computed on the ERSSTv5 reconstruction against a shifting 30-year base — and an
event is declared when it stays above +0.5 °C for five overlapping seasons.

### 4.2 · The index table, computed from this repository

Computed with Python from `data/oisst_y/<year>.json` (NOAA OISST v2.1 monthly means, block-
meaned to 1°, baked by `oisst_monthly()` at `scripts/refresh_data.py:1847`) against
`data/oisst_clim/<mm>.json` (the 1991–2020 mean for that calendar month, `refresh_data.py:1907`).
Box definitions: Niño-3.4 = 5° S–5° N, 170° W–120° W (500 cells) · Niño-3 = 5° S–5° N,
150° W–90° W (600) · Niño-4 = 5° S–5° N, 160° E–150° W, wrapping the dateline (500) ·
Niño-1+2 = 10° S–0°, 90° W–80° W (100 cells, 99 of them ocean).

**December peaks of the six reference events, °C:**

| month | Niño-3.4 | Niño-3 | Niño-4 | Niño-1+2 |
|---|---|---|---|---|
| 1982-12 | +2.30 | +2.90 | +0.39 | +2.95 |
| 1997-12 | +2.18 | +3.16 | +0.46 | +3.79 |
| 1998-12 | −1.81 | −1.22 | −1.53 | +0.19 |
| 2010-12 | −1.21 | −1.32 | −1.17 | −1.22 |
| 2015-12 | +2.46 | +2.64 | +1.45 | +2.19 |
| 2023-12 | +2.01 | +2.10 | +1.42 | +1.38 |

**The 2026 season so far, °C:**

| month | Niño-3.4 | Niño-3 | Niño-4 | Niño-1+2 |
|---|---|---|---|---|
| 2025-11 | −0.72 | −0.64 | −0.56 | −0.38 |
| 2026-01 | −0.57 | −0.50 | −0.05 | −0.20 |
| 2026-02 | −0.19 | +0.00 | +0.22 | +0.88 |
| 2026-03 | +0.02 | +0.24 | +0.30 | +1.26 |
| 2026-04 | +0.47 | +0.56 | +0.84 | +1.62 |
| 2026-05 | +0.94 | +1.17 | +1.00 | +1.90 |
| 2026-06 | +1.60 | +1.79 | +1.27 | +2.93 |
| **2026-07** | **+2.09** | **+2.39** | **+1.08** | **+3.64** |

Reading it: Niño-3 above Niño-3.4 above Niño-4, with Niño-1+2 leading since February — a
canonical **eastern-Pacific** El Niño in rapid development, crossing zero in March and
reaching the level of the December 2023 peak by July.

**Caveats.** This is a 1°-regridded monthly mean against a fixed 1991–2020 base, not the
official index, which uses a coarser reconstruction (ERSSTv5), a three-month running mean and
a centred 30-year base. Treat these as "Niño-3.4 sea-surface-temperature anomaly", good to
roughly ±0.1–0.2 °C of the official number for the historical months — and see §4.3, where the
July gap is 0.7 °C. Cell values in the bake are rounded to 0.1 °C, which averages out over
hundreds of cells.

### 4.3 · The official status, quoted

From NOAA's Climate Prediction Center ENSO diagnostic discussion, **13 August 2026**:

- **"El Niño Advisory"** is in effect.
- The July value was **"+1.4 °C in Niño-3.4"**.
- **"greater than 90% chance of a very strong event during the Northern Hemisphere fall and
  winter 2026-27"**.
- **"69% chance of a historic event that would exceed the strength of previous El Niño events
  dating back to 1950 (+2.5 °C or more)"** in October–December.

Source: [CPC ENSO diagnostic discussion, Aug 2026](https://www.cpc.ncep.noaa.gov/products/analysis_monitoring/enso_disc_aug2026/ensodisc.shtml)

The 0.7 °C gap between our +2.09 and the official +1.4 is NOT explained here: a different
sea-surface-temperature dataset (ERSSTv5 vs OISST), a different averaging, and possibly the
"relative" Niño-3.4 index (the anomaly minus the tropical-mean warming) that CPC has moved
towards would each account for part of it. Reconciling the two is the first thing a `nino.json`
bake should do, and the gap is the argument for carrying the official index **alongside** our
own bake rather than instead of it.

### 4.4 · Catalog audit — what carries ENSO signal, and its limit

The catalog (`data/catalog.json`, 274 records, compiled 2026-07) was assembled for the
Atlantic overturning question. Its `amoc` boolean **mis-sorts for ENSO** and will mislead if
used as a relevance filter: `rapid`, `osnap`, `move`, `samba`, `florida-current` and every
ice-core record carry `amoc: true` and no ENSO content, while `ersst` — the substrate of the
official index — carries `globe: false` and no ENSO framing at all. A search of all 274
records for `enso|el ni|niño|southern oscillation|walker|thermocline|kelvin wave|TAO|TRITON|MJO`
matched **7 records**, only two of them on ENSO substance. **There is not one ENSO index, one
ENSO forecast product, or one tropical-Pacific-specific diagnostic in the catalog.**

What is genuinely useful, with its limit:

- **`oisst` — NOAA OISST v2.1.** The label itself: all four Niño boxes, and the Niño-3-vs-
  Niño-4 contrast that separates eastern-Pacific from central-Pacific events. Global 0.25°,
  1981-09 →. **In-repo and working**: the 1° monthly archive `data/oisst_y/` (46 files) plus
  the `data/oisst_clim/` normals produced §4.2. Limit: the 1° regrid slightly smooths the
  equatorial front.
- **`glorys` — GLORYS12.** Subsurface heat content, thermocline depth, mixed-layer depth,
  zonal currents (the Equatorial Undercurrent) and sea-surface height — everything ENSO's
  ocean memory lives in. Limits: starts 1993, so it misses 1982–83 entirely; and in this
  repository only the **surface** speed and mixed-layer depth are baked
  (`data/currents_y/`, `data/mld_y/`, 1993–2026), not the depth levels.
- **`gpcp` (1979 →) and `imerg` (2000 →).** The rainfall shift towards the dateline that *is*
  El Niño's atmospheric expression. Limits: `data/gpcp.json` is a **climatology only**, not a
  time series — a re-bake is needed; IMERG is live but misses 1982–83 and 1997–98.
- **`merra2-wind` — MERRA-2 10 m wind.** Trade-wind strength and the Walker circulation.
  Limit: the app's tiles are **monthly means only**, and a westerly wind burst is a 5–15 day
  event that a monthly mean averages away.
- **`nasa-ssh` — MEaSUREs sea-surface height.** Equatorial Kelvin and Rossby waves. Limit: the
  app's tiles **end 2019-01** (`CLAUDE.md` layer matrix) and serve only exact 5-day epoch
  dates — useless for 2026 through tiles; go to PO.DAAC.
- **`oscar` — OSCAR near-surface currents.** Equatorial zonal advection, the observational
  counterpart to GLORYS. Limit: tiles run **2014-10 → 2024-07** only, missing 1997–98,
  2015–16 and all of 2025–26; the PO.DAAC files hold 1992-10 → present.
- **`ceres` — CERES EBAF.** Often used as an outgoing-longwave proxy for convection. Two
  limits, either of them fatal here: the app's tiles **end 2018-10**, and EBAF is *monthly net
  flux*, not the daily outgoing-longwave field the index needs.
- **`gtmba` — TAO/TRITON, PIRATA and RAMA moorings.** The single most ENSO-specific record in
  the catalog: equatorial moored sea-surface temperature, subsurface temperature and salinity
  profiles (hence thermocline depth), zonal currents, winds and fluxes along 137° E–95° W since
  the late 1970s. Limit: catalogued **only for its Atlantic side** — the notes read "PIRATA
  feeds tropical Atlantic AMOC context", with `globe: false` — so the Pacific array is
  completely unexploited here. Real-world limit: array degradation after 2012.
- **Catalogued and unused:** `oras5` (1958 →, 5 members) · `soda` (SODA3, 1980 →) · `ersst`
  (ERSSTv5, 1854 →, the official index's substrate) · `hadisst` (1870 →) · `en4` (1900 →,
  42 levels) · `era5` and `arco-era5` (1940 →, hourly) · `cfsr` (coupled, 1979 →) ·
  `c3s-seasonal` (SEAS5 and seven systems, hindcasts from 1981) · `nmme` (~8 models, leads
  0–11 months, 1982 →).
- **The machine-learning tensors are North-Atlantic-only.** Families 3–6 are hard-coded to
  0–70° N, 100° W–20° E (`ml/build_family6.py:126`) — no tropical Pacific at all. Family 7 —
  the first tensor covering the whole globe at the same 0.25°, 5-day grid, experiment E-070 —
  is mid-fetch, and its own plan names the equatorial waveguide as a target regime. Its
  channel set (sea-surface height, mixed-layer depth, a 16-level temperature and salinity
  column, wind stress and its variability) is **almost exactly a classic ENSO feature set**;
  what it lacks is a convection channel and a label.

### 4.5 · Add, ranked — with links

Each of these is free and small. The URLs are general knowledge and were not verified against
the repository, except the CPC discussion, which was fetched.

**1 · Warm Water Volume and 20 °C-isotherm depth (NOAA PMEL), monthly 1980 →.**
How much water above 20 °C is stacked up across the equatorial Pacific. The single best
~6-month-lead predictor; recharge–discharge theory *is* this number. Its absence is the
biggest gap in the catalog.
[PMEL Warm Water Volume](https://www.pmel.noaa.gov/tao/wwv/data/)

**2 · GODAS, or ORAS5 — the subsurface field.**
NOAA's operational Global Ocean Data Assimilation System: 1/3° × 1° at the equator, 40 levels,
pentad and monthly, 1980 →, keyless OPeNDAP — easier to get than ORAS5 and absent from the
catalog. ORAS5 (already catalogued, 1958 →, 5 members) is the long alternative.
[GODAS (NOAA PSL)](https://psl.noaa.gov/data/gridded/data.godas.html)

**3 · NOAA interpolated outgoing longwave radiation, 2.5°, daily, 1974 →.**
Cold cloud tops mark deep convection; this is *the* Walker-circulation and Madden–Julian
diagnostic, and CERES does not replace it.
[NOAA interpolated OLR](https://psl.noaa.gov/data/gridded/data.interp_OLR.html)

**4 · Index series, as labels and as nulls.**
The Oceanic Niño Index (official, ERSSTv5, 1950 →); weekly Niño-1+2/3/3.4/4; the Southern
Oscillation Index (the atmospheric half of ENSO — Tahiti minus Darwin sea-level pressure) and
its equatorial variant; the Multivariate ENSO Index; the RMM Madden–Julian Oscillation index
(the westerly-wind-burst trigger); and the Pacific Decadal Oscillation and Pacific Meridional
Mode, the latter a genuine boreal-spring precursor.
[Oceanic Niño Index (CPC)](https://origin.cpc.ncep.noaa.gov/products/analysis_monitoring/ensostuff/ONI_v5.php)
[NOAA PSL climate indices](https://psl.noaa.gov/data/climateindices/list/)
[Multivariate ENSO Index v2](https://psl.noaa.gov/enso/mei/)
[RMM Madden–Julian index (Bureau of Meteorology)](https://www.bom.gov.au/climate/mjo/)

**5 · Daily winds.**
ERA5 10 m eastward and northward wind (0.25°, hourly, 1940 →) or CCMP multi-platform ocean
vector winds (6-hourly, 1987 →): westerly wind bursts and wind-stress curl at the cadence at
which they actually happen, instead of the monthly means the app's tiles carry.
[CCMP ocean vector winds (REMSS)](https://www.remss.com/measurements/ccmp/)

**6 · TAO/TRITON moorings as native dots on the Pacific side.**
Sea-surface temperature, subsurface temperature and salinity, currents and winds along
137° E–95° W since 1980. The array is already catalogued, as `gtmba` — this is a labelling and
ingestion gap, not a data gap.
[Global Tropical Moored Buoy Array (PMEL)](https://www.pmel.noaa.gov/gtmba/)

**7 · DUACS sea level directly, daily 0.125°, 1993 →.**
Equatorial Kelvin and Rossby wave propagation, and a proxy for warm water volume — the
highest-signal precursor after the subsurface temperature itself.
[Copernicus Marine sea level (DUACS L4)](https://data.marine.copernicus.eu/product/SEALEVEL_GLO_PHY_L4_NRT_008_046)
And, for the surface currents whose tiles stop in 2024, go to the source:
[OSCAR L4 final v2.0 (PO.DAAC)](https://podaac.jpl.nasa.gov/dataset/OSCAR_L4_OC_FINAL_V2.0)

**8 · Benchmarks — what any ENSO model must beat.**
The IRI/CPC forecast plume (about 25 dynamical and statistical models, monthly since 2002), and
the NMME and SEAS5 hindcast archives, both already catalogued and unused. Beat them, or say
you did not — and plot the skill curve across the spring predictability barrier.
[IRI/CPC ENSO forecast plume](https://iri.columbia.edu/our-expertise/climate/forecasts/enso/current/)

**9 · In the app: a `nino.json` bake.**
Every ingredient is already baked and refreshed, and no ENSO index is computed anywhere in this
repository — §4.2 had to be derived by hand. A `nino()` function beside `oisst_monthly()` in
`scripts/refresh_data.py` is roughly 40 lines, and would sit next to `data/eei.json` and
`data/gistemp.json`; the four Niño boxes then draw on the globe.
[data/catalog.json (source)](https://github.com/blauewelt/earth/blob/main/data/catalog.json)
[docs/CATALOG.md — the catalog in prose](https://blauewelt.github.io/earth/docs.html?f=docs/CATALOG.md)

**10 · The cone codec on family 7, with an ENSO head.**
Sea-surface height, mixed-layer depth, the temperature and salinity column and wind stress are
already the classic feature set; what is missing is a convection channel (outgoing longwave or
precipitation) and the label. That makes an ENSO head the cheapest possible test of the
programme's central claim — that predicting embeddings predicts everything — on a target that
is not the AMOC.

### 4.6 · Three ways to "predict this year's El Niño"

Onset is behind us: the index crossed zero in March 2026. So the question splits into three,
and they need different data and different nulls.

1. **Hindcast the onset.** From the November 2025 state — a −0.72 °C, mildly La-Niña-ish
   surface sitting over a recharged subsurface — could a model have called the March crossing?
   This is a test of whether the subsurface memory was readable, and it needs items 1 and 2
   above; sea-surface temperature alone cannot answer it, which is the whole point.
2. **Forecast the peak and the exit.** How high, in which month (the official outlook says
   October–December), and when does it decay through 2027? Amplitude prediction leans on warm
   water volume and on the wind bursts that keep discharging it, so items 1, 2 and 5.
3. **Forecast the teleconnections.** The rainfall, drought, fire and cyclone shifts that are
   what anyone outside the Pacific actually experiences. This is where the catalog is
   strongest already — IMERG, GPCP, GRACE terrestrial water storage, IBTrACS cyclone tracks —
   and where a learned representation has the most to add over an index.

All three need an honest null beside them, every time: persistence, a linear inverse model,
and the operational forecast plume. A number without its baseline is not a result
(`ml/CLAUDE.md` §3).

---

## Part 5 · Boundaries: how the sampler handles every edge

Four different edges meet the cone codec. Three of them are handled by making the missing thing
explicitly missing. The fourth is not an edge in the data at all — it is one we introduced — and
nothing corrects for it.

Two words are needed first, because the whole section turns on the difference.

- An **invalid** token is a position that *does not exist*: the dot fell outside our rectangle.
  It is deleted from the attention mask, so the transformer never sees it at all.
- A **miss** token is a real position where the world *was not observed* — land under a
  sea-surface sensor, a pentad with no Argo profile. It is a token the model learns from, because
  "nobody measured this" is itself information.

Everything below is one of those two, or neither.

### 5.1 · Space — the window edge (100° W – 20° E, 0–70° N)

**The rule.** A dot that lands outside the rectangle is INVALID, never wrapped. The tensor is a
basin, not a globe: `ml/cone_sampler.py`'s docstring (lines 19–23) says a longitude wrap "would
put the Iberian shelf one cell west of Florida". `ml/model.py::gather_px` does wrap, because it
was written for a global tensor; the cone sampler deliberately does not.

**The mechanism.** The index is clipped to the array bounds, the value that comes back is thrown
away, and `valid = False` becomes the attention mask:
`ok = (tt>=0)&(tt<T)&(yy>=0)&(yy<H)&(xx>=0)&(xx<W)` in `ConeSampler.sample`
(`ml/cone_sampler.py:293`). The lag-0 3×3 patch does the same thing in `_patch`
(`ml/cone_sampler.py:263`).

**The anchors do not avoid the edge.** They are drawn uniformly over ocean cells with no edge
margin (`ml/train_cone.py::draw_anchors:358`), so an edge anchor legitimately sees a truncated,
one-sided cone.

**The measured cost**, over all 84,405 ocean anchors with the real 706-dot stencil:

- **22.0 %** of anchors have at least one off-grid dot.
- **2.68 %** of all dots are off-grid — invalid, and dropped from attention entirely.

**The asymmetry this creates.** An anchor twenty cells from the eastern edge sees no upstream dots
from the east at lags 5–6. Nothing tells the model that those dots are missing because of *our*
rectangle rather than because of the ocean — the two look identical from inside the attention mask.

### 5.2 · Space — land

**The rule.** A dot on land is a real token whose value was never observed: `obs = False` becomes
the codec's **miss** token — the same `miss_tok` that `PixelMAE` (the pixel-level masked
auto-encoder this codec inherits from) uses — and *not* an invalid or masked token. The
distinction is deliberate and it is the one the whole architecture rests on.

**The measured cost.**

- **8.17 %** of all dots land on land: on-grid, unobserved.
- Combined with the edge, **10.85 %** of the 706 dots at a typical anchor carry no value.

Land dots are still spent from the token budget — they occupy their slot in the stencil and are
attended to as tokens — so a coastal anchor pays for its geography twice: once in dots it cannot
read, once in dots that left the rectangle.

### 5.3 · Time — the archive ends, and the holdout blocks

**Rule 1, the ends of the axis.** An anchor whose cone runs off either end of the tensor — it
needs bins *t−6 … t+2* — is INADMISSIBLE, not silently short (`ConeSampler.admissible`,
`ml/cone_sampler.py:314`). Nothing is padded and nothing is trained on half a cone.

**Rule 2, the holdout.** `--holdout-scope window` — the only scope implemented, and
`ml/train_cone.py:116` refuses any other — says that every bin the cone touches, six pentads back
and both future targets, must be a training bin. A held-out year therefore casts a **shadow of
eight pentads** around itself (six back and two forward, the exact span of the cone), and no
training anchor can peek into it by any path. A brute-force `certify()`
(`ml/cone_sampler.py:337`), written deliberately as a separate loop rather than as a reuse of the
admissibility test, counts violations; run **#537** (the E-069 cone-codec build) measured
**0 violations in 4,096 anchors**.

**The measured cost of the shadow.**

- Development split (calendar years 2009, 2017 and 2023 held out): **2,923** training bins →
  **2,891** admissible anchor bins — **32 bins lost, 1.1 %**.
- Frozen protocol (2008–09, 2016–17, 2021–24 — train up to 2020, test the terminal years):
  **2,557 → 2,533** — **24 bins lost, 0.9 %**.

Cheap, and it is what makes the holdout real.

**Held-out EVALUATION anchors are different on purpose.** Their dots MAY come from held-out bins.
Otherwise the held-out set would be a second training pool rather than a measurement.

### 5.4 · Time — the calendar month, the one edge that is NOT masked

This is the honest weak spot, and it is worth saying so plainly. There are two separate month
edges, and neither is masked, measured or interpolated away.

**(i) The anomaly baseline.** Every channel is turned into a departure from a per-calendar-month
climatology built on training years only (`ml/trainprobe.py::anomaly_transform:140`). A pentad's
month label is the month of the bin's FIRST day (`ml/build_family4.py:960`, via `bin_start`). So:

- **411 of the 3,142 bins (13.1 %)** straddle a calendar-month boundary.
- **106** of those have only ONE of their five days inside the month whose climatology is
  subtracted.

The anomaly therefore takes a small step at every month boundary — a sawtooth of order the
month-to-month climatology difference, which in the seasonal thermocline is not negligible.
Nothing interpolates. A day-of-year harmonic climatology would remove it.

**(ii) Argo's monthly cadence.** Roemmich–Gilson is written into the single pentad holding the
15th of each month and is `missing` in the other five (`ml/build_family4.py`); forward-filling was
explicitly REJECTED in `ml/plans/E034_pentad_tensor.md:88-100`, because it would tell the model the
subsurface was observed on days it was not. That is **252 live pentads over 2004–2024, one in
six.** A 35-day inner window (the anchor plus six lags) therefore contains ONE live Argo bin,
occasionally two — and which one is a property of where the anchor sits inside the month, i.e. a
phase the model can see through the season token but that nothing corrects for.

**The seasonal context token itself is fine.** It is the bin's opening day-of-year
(`ml/cone_sampler.py::pentad_doy`), so it crosses month and year boundaries smoothly.

### 5.5 · Reading

Three of the four edges are handled by making the missing thing explicitly missing, which is the
right pattern: an invalid token is dropped from attention, a miss token says the world was not
observed, and an inadmissible anchor is not trained on. The fourth is not an edge in the data at
all but one we introduced by binning the year into twelve boxes. It is the cheapest of the four to
remove, and the only one currently unmeasured.

---

## Part 6 · The speeds the cone does not have

Chris, 3 September 2026: *"the AMOC has some velocity (4000 km/month?) and this is not reflected
anywhere"*. It is not — twice over.

Two words, because the section depends on the difference between them:

- **Advection** is transport by the water itself moving: a warm parcel physically carried north.
- **Wave propagation** is a disturbance travelling *through* the water while the water largely
  stays put, so a signal can cross a basin far faster than any parcel does. A **Kelvin wave** is
  the fast kind that runs along a coastline or the equator, trapped against the boundary.

The **Deep Western Boundary Current (DWBC)** is the slow return limb of the overturning: cold
dense water formed in the north creeping southward along the American continental slope.

**The speed figures below are order-of-magnitude values from standard oceanography plus this
project's own proposal document. They are not fitted, and no individual number is attributed to a
particular paper.**

### 6.1 · The speed ladder, slowest to fastest

- **Deep Western Boundary Current advection ≈ 0.02–0.1 m/s** — the multi-year southward advective
  path.
- **Baroclinic Rossby waves ≈ 0.03 m/s**, westward. The plan already cites Chelton et al. 2011
  for this (`ml/plans/E069_cone_codec.md:124`).
- **Eddies and boundary currents ≈ 0.1–0.3 m/s** — **the only ocean speed the cone implements**
  (family B, `v_ms = 0.3`, `ml/cone.py:97`).
- **Chris's number, 4,000 km/month = 1.52 m/s.**
- **Coastal / boundary Kelvin waves ≈ 2.5 m/s** — a number the project's own proposal already
  carries. `ml/figures/geofm_survey/GENERIC_EMBEDDING_INPUTS.md:72` describes family B as
  "currents 0.1–0.2 m/s … Rossby 0.03 m/s (westward); Kelvin/coastal 2.5 m/s along boundaries",
  with the stencil shape "thin, tall column, tilted upstream (and westward for SSH); **one long
  arm along coasts**". `ml/cone.py` implements the 0.3 m/s disc and no coastal arm.
- **Wind stress 10 m/s** (family A) — implemented, but capped at the 500 km correlation length and
  present only at lags 0–1, so it is not a long-range channel either.

### 6.2 · Half one — the Argo channels have no spatial reach at all

`ml/cone.py::channel_family:122` assigns `rg_t*` and `rg_s*` — the 32 Roemmich–Gilson temperature
and salinity channels — to family B (0.3 m/s). But `ml/cone.py::channel_dots:261` OVERRIDES that
for depth channels and returns the ANCHOR COLUMN ONLY: `[(lag, 0, 0) for lag in 1..L_in]`.

So the family assignment is decorative for them. Their effective reach is **0 km at every lag**.

The stated reason is token economy: Argo is live one pentad in six, so a sunflower would be about
74 tokens of which about 70 are structurally missing.

The consequence is that the subsurface temperature and salinity structure is read only at the
anchor's own column, so **no propagation of a subsurface anomaly toward the anchor is visible to
the codec at any speed**. And the 32 depth channels are still **192 of the 706 dots (27.2 %)** — a
quarter of the token budget spent on one vertical column.

The dot budget in full, for context:

- currents and sea-surface height (family B, 4 channels) — **320 dots, 45.3 %**
- Argo depth (32 channels) — **192 dots, 27.2 %**
- SST and mixed-layer depth (family C, 2 channels) — **162 dots, 22.9 %**
- wind stress (family A, 4 channels) — **32 dots, 4.5 %**

### 6.3 · Half two — the fastest ocean speed in the model is 0.3 m/s

Family B's reach is *v · Δt · (1 + ℓ)*, which at lags 0–6 is:

**129.6 · 259.2 · 388.8 · 518.4 · 648.0 · 777.6 · 907.2 km.**

At Chris's 1.52 m/s a signal covers **657 km per pentad** — 5.07× family B's 129.6 — and
**4,600 km over the six-lag window, against the cone's 907 km**.

Stage 2's outer cone grows at the same 0.3 m/s and reaches its 4,444 km cap only at **lag 34, about
170 days**; 1.52 m/s covers 4,444 km in about **34 days**, and 2.5 m/s in about **21**.

**The failure mode is not symmetric, and this is the point.** A cone that is too WIDE only costs
tokens. A cone that is too NARROW excludes true causes: it is an assumption that the driver cannot
have arrived, and the model has no way to discover otherwise.

`ml/cone.py`'s own docstring (`:84`) says the speeds were "chosen generous so the stencil is not
the thing that loses information". 0.3 m/s is generous for eddy advection and a factor of five to
eight too slow for boundary-wave adjustment.

### 6.4 · Three fixes, ranked — all of them PROPOSALS, none of them measured

1. **A fourth family, "fast boundary adjustment".** v ≈ 2.5 m/s, short memory (days), applied to
   sea-surface height and the mixed-layer and temperature channels, giving a reach of about
   1,080 km at lag 0 growing to about 7,500 km at lag 6, capped at the basin. It costs tokens; the
   cheapest test is whether the velocity probe and the held-out loss move at all. Mechanically it
   is a one-line change to `FAMILIES` plus a token-budget check.
2. **The coastal ARM the proposal already specifies.** Instead of an isotropic disc, extend the
   stencil ALONG the western boundary and the shelf break, which is where the fast waves actually
   travel. This needs the statics from the data ladder — distance to coast, distance to the
   1,000 m isobath — which rung 12 already ranks #2 and which are about 3 MB.
3. **Give the depth channels a real, if sparse, sunflower** at the lags where Argo is live. The
   token objection is about the DEAD bins, and liveness is knowable at build time.

### 6.5 · The other end of the range

The DWBC advective path from the Labrador Sea to 26.5° N is roughly 5,000 km at 0.02 m/s — about
**eight years** — which is outside even stage 2's two-year outer window. So the cone brackets the
eddy field well and misses the mechanism at BOTH ends of the speed range.

### 6.6 · Reading

The cone is currently a one-speed model of a multi-speed ocean. The 0.3 m/s disc is right for the
eddy field it was calibrated on and wrong for the two mechanisms that actually carry an overturning
anomaly between latitudes; and for the Argo channels the question does not even arise, because they
are read at one column. None of the three fixes has been tested; the first is a one-line change to
`FAMILIES` plus a token-budget check.

---

## Where this came from

- [E-069 plan — the cone codec](https://blauewelt.github.io/earth/docs.html?f=ml/plans/E069_cone_codec.md)
- [ml/cone.py — the cone geometry and the channel families (source)](https://github.com/blauewelt/earth/blob/main/ml/cone.py)
- [ml/cone_sampler.py — the sampler, its off-grid rule and the holdout certificate (source)](https://github.com/blauewelt/earth/blob/main/ml/cone_sampler.py)
- [The data ladder — what to import next](https://blauewelt.github.io/earth/docs.html?f=ml/plans/DATA_LADDER.md)
- [Generic-embedding input proposal](https://blauewelt.github.io/earth/docs.html?f=ml/figures/geofm_survey/GENERIC_EMBEDDING_INPUTS.md)
- [The survey deck README](https://blauewelt.github.io/earth/docs.html?f=ml/figures/geofm_survey/README.md)
- [docs/CATALOG.md — the open-data catalog in prose](https://blauewelt.github.io/earth/docs.html?f=docs/CATALOG.md)

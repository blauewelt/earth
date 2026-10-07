# The Data tab — download the project's stores from your browser

The **Data** tab (between *Cones* and *Play*) lets you pick one of the project's
data stores, narrow it to the period, months, hours, region and
resolution you want, see what that will cost before anything is read, preview it
on the globe, and save it as a NetCDF or CSV file. The file is built **in your
browser** from the store's own files: no server computes anything.

The store list is the union of five groups, each read from its own registry:

- **Fine observations (family 1.gf)** — the global set of observation stores
  at 10 km / 5 days or finer, each kept at its own native resolution rather
  than resampled onto a common grid; registry `tensors/family1_gf/family1gf.json`.
- **Atmosphere on pressure levels (family 1.2 — ERA5 reanalysis)** — family
  1.2 is family 1.gf plus ECMWF's ERA5 reanalysis of the atmosphere:
  temperature, humidity and wind at 13 pressure levels, six-hourly, on a 1°
  grid; registry `tensors/family1_2/family12.json`. It *inherits* every 1.gf
  store by reference (the same bytes), so those are listed once, under 1.gf,
  and the 1.2 group shows only what 1.2 adds.
- **Global tensor and point observations (family 10)** — the 0.25° and 1°
  five-day grids the forecaster is trained on, the monthly Argo grid at 16
  depths, and the point stores beside them (Argo profiles, drifters, moored
  buoys, ship CO₂, altimeter tracks, fishing vessels); registry
  `tensors/family10_2/family10.json`.
- **Global tensor, daily (family 7.2d)** — the global tensor's channels (the
  0.25° and 1° grids above) as one frame per DAY instead of a five-day mean,
  one store per source, each running to the last day its producer serves;
  registry `tensors/family7_2d/family72d.json`.
- **Derived maps** — maps the project computes from those: the 0.25° monthly
  fishing-effort map and the **monthly normals** of the global tensor over any
  period you choose (what the forecaster calls "normal", from per-year monthly
  sums and counts). Their indexes are the site's own `data/fishing_index.json`
  and `data/family7_monthly_index.json`.

All the data live on the project's public data store, the Hugging Face dataset
[chfrank/earth-tensors](https://huggingface.co/datasets/chfrank/earth-tensors).
A store a registry announces but has not built yet is **never selectable**: the
tab names it in one quiet "Coming: …" line under the heading, and it appears in
the list by itself the day the registry marks it built — the tab reads the
registries every time it opens, so no change to the page is needed. (A store
whose licence still waits on its producer is not "coming" and is not named.)
If one registry or one store cannot be read, the others still load and the tab
prints a line naming what is unavailable and why. The plan that specified this tab and its reader is
[E-084](https://blauewelt.github.io/earth/docs.html?f=ml/plans/E084_data_tab.md);
the stores themselves are described in the
[family 1.gf design note](https://blauewelt.github.io/earth/ml/paper/notes/family1gf.pdf).

## 1. The stores

The tab reads the list of stores from the registries at run time, so a store
appears in the tab the day it is published; these tables are the set as of
2026-10-07 (35 stores). Three kinds:

- a **map store** (a gridded product): one value per pixel per frame;
- a **point store**: every report at its own position and time — a ship's
  observation, a float's profile, a bottle sample, a satellite sounding;
- a **normals store**: one sum and one count per year, calendar month,
  channel and cell, from which the tab composes the mean over the years you
  choose (§1, *Monthly normals*).

Every store is offered at its native space and time and nothing coarser is
pretended: the 0.25° and 1° choices under *Resolution* appear only where they
are coarser than the store itself.

### Fine observations (family 1.gf)

| store | what it measures | kind | native space | native time | record |
|---|---|---|---|---|---|
| `oc4k` | ocean colour: chlorophyll (as its base-10 logarithm), water clarity (`kd_490`), observation count | map | 4 km (1/24°) | daily | 1997-09 → 2022-12 |
| `pace4k` | chlorophyll, particulate carbon, phytoplankton carbon and four more, from NASA's PACE satellite | map | 4 km | daily | 2024-03 → 2026-09 |
| `sst_acspo02` | sea-surface temperature as satellites saw it, clear sky only, with a quality grade | map | 0.02° (2 km) | daily | 2000-02 → 2026-09 |
| `irtb` | cloud-top brightness temperature from geostationary satellites, 30°S–30°N | map | 4 km | 3-hourly as stored | 1998 → 2026-09 |
| `icoads` | ship and buoy reports: sea and air temperature, pressure, wind, dew point, waves, cloud | points | per report | per report | 1662 → 2026 |
| `wod` | ship casts: temperature, salinity, oxygen and more at 16 pressures | points | per cast | per cast | 1772 → 2026 |
| `glodap` | bottle samples: carbon, alkalinity, pH, oxygen, nutrients | points | per bottle | per bottle | 1972 → 2023 |
| `bgcargo` | biogeochemical floats: oxygen, nitrate, pH, chlorophyll and more at 16 pressures | points | per profile | per profile | 2002 → 2026 |
| `oceansites` | open-ocean moorings | points | per mooring | hourly | 1980 → 2026 |
| `xco2` | column carbon dioxide soundings from three satellites (OCO-2, OCO-3, GOSAT) | points | under 3 km² | per sounding | 2009 → 2026 |
| `swh` | significant wave height along satellite altimeter tracks | points | about 7 km | one per second along track | 1991 → 2023 |
| `swot` | sea-level anomaly on the SWOT satellite's 2 km swaths | points | 2 km | per pass | 2023-07 → 2026-09 |

### Atmosphere on pressure levels (family 1.2 — ERA5 reanalysis)

| store | what it measures | kind | native space | native time | levels | record | size on the data store |
|---|---|---|---|---|---|---|---|
| `era5_t` | air temperature, K | map | 1° (from ERA5's ~31 km) | six-hourly instants, 00/06/12/18 UTC | 13 | 1982-01-01 → 2026-06-30 18 UTC | 53.8 GB |
| `era5_q` | specific humidity, **g/kg** (grams of water vapour per kilogram of air — 1000 × ERA5's kg/kg) | map | 1° | six-hourly instants | 13 | 1982-01-01 → 2026-06-30 18 UTC | 94.7 GB |
| `era5_u` | eastward wind, m/s (positive toward the east) | map | 1° | six-hourly instants | 13 | 1982-01-01 → 2026-06-30 18 UTC | 99.0 GB |
| `era5_v` | northward wind, m/s (positive toward the north) | map | 1° | six-hourly instants | 13 | 1982-01-01 → 2026-06-30 18 UTC | 101.2 GB |

The 13 levels are pressures: 50, 100, 150, 200, 250, 300, 400, 500, 600, 700,
850, 925 and 1000 hPa — from about 20 km up down to the surface. The tab shows
them as a level picker beside the variable and starts on **500 hPa**, the
middle of the troposphere. Each store opens on its most recent month at 500 hPa
in the first preset box: about 11 MB to read for temperature and 20–22 MB for
humidity and the winds (their tiles compress less well).

**Humidity can be slightly negative.** The reanalysis can produce slightly
negative specific humidity (a numerical artefact of the model, not a
measurement). Since the store was republished on 2026-10-06 it keeps those
values as ERA5 gives them — the smallest in the whole record is −0.136 g/kg —
and only a value outside the registry's channel range (−1 to 40 g/kg) would be
stored as missing. The preview's colour scale uses that registry range, so a
slightly negative cell is not singled out.

**This is a reanalysis, not an observation.** ERA5 is ECMWF's best estimate of
the atmosphere: a weather model run forward and pulled toward every available
observation, natively at about 31 km and averaged here (area-weighted) onto a
1° grid. It is global and gap-free *because* the model fills every place and
height nobody measured, so it does not belong with the stores above as a
measurement. Each frame is the analysis at one instant (00, 06, 12 or 18 UTC),
not a six-hour mean. Every file carries the licence (CC BY 4.0, Copernicus
Climate Change Service) and the full attribution in its global attributes, and
the panel prints it under the store. **Reading one level reads all 13**: the
levels sit side by side in every compressed tile, so the estimate counts the
real read and says so.

### Global tensor and point observations (family 10)

| store | what it measures | kind | native space | native time | record |
|---|---|---|---|---|---|
| `g025` | ocean surface state: sea-surface temperature, sea-surface height, surface currents (east, north, speed), mixed-layer depth (as a logarithm), sea-ice fraction | map | 0.25° | five-day means | 1982-01 → 2024-12 |
| `g100` | air–sea fluxes, weather and land: wind stress and its variability, 2 m air temperature, 10 m wind, surface pressure, rain (log), snow (log), soil moisture and temperature, latent and sensible heat flux, skin temperature | map | 1° | five-day means | 1982-01 → 2024-12 |
| `oc025` | chlorophyll (as a logarithm) and how much of the cell was seen | map | 0.25° | five-day means | 1997-09 → 2024-12 |
| `rg100` | ocean temperature and salinity at 16 pressures from 10 to 2000 dbar (the Argo gridded analysis) | map | 1° | monthly | 2004-01 → 2024-12 |
| `argo` | Argo float profiles: temperature and salinity at the same 16 pressures | points | per profile | per profile | 2004 → 2024 |
| `gdp` | surface drifters: current (east, north), sea-surface temperature, whether the drogue was attached | points | per report | 6-hourly | 1979 → 2024 |
| `gtmba` | tropical moored buoys (TAO, PIRATA, RAMA): surface temperature and salinity, air temperature, wind, current, temperature at 12 depths to 500 m | points | per buoy | daily | 1977 → 2024 |
| `socat` | ship measurements of ocean CO₂ (fCO₂) with sea temperature, salinity and pressure | points | per measurement | per measurement | 1957 → 2024 |
| `slatrack` | sea-level anomaly along altimeter tracks, filtered and unfiltered, with the mean dynamic topography | points | about 7 km | one per second along track | 1993 → 2024 |
| `fishing` | fishing effort per vessel and day from ship transponders (AIS, Global Fishing Watch): fishing hours and hours present | points | 0.1° cell per vessel | daily | 2012 → 2024 |

The four grids are the forecaster's own inputs. They are stored as z-scores (a
value minus its long-term mean, divided by its spread) in 16-bit floats; the
reader turns them back into physical units with the mean and spread the site
keeps in `data/family7_index.json`, so a file from the tab carries °C, m, m s⁻¹
and so on, not z-scores. A grid stores all its channels side by side, so the
read always covers every channel of the store whichever you tick — the
estimate says so.

### Global tensor, daily (family 7.2d)

Family 7.2d is the global tensor (the `g025`, `g100` and `oc025` grids above)
with one frame per calendar DAY instead of the mean of the five days in a bin
— the same channels, names, units and derivation, in physical units, stored
like the family-1.gf maps (compressed tiles per five-day bin, five daily
frames each). The Argo depth grid is monthly and has no daily form. Each
record runs to the last day its producer served when the store was built
(2026-10-06), past family 7.2's own end of 2024-12-31; the tab reads every
record from the registry (`record_span`, where the data actually is), never
from this table.

| store | plain-English name | what it measures | native space | native time | record |
|---|---|---|---|---|---|
| `glorys025d` | Ocean currents, sea-surface height and mixed layer (GLORYS reanalysis) | surface current east, north and speed (m/s), sea-surface height (m), mixed-layer depth (log₁₀ m) — Copernicus Marine's GLORYS12 ocean reanalysis | 0.25° | daily means | 1993-01-01 → 2026-08-18 |
| `oisst025d` | Sea-surface temperature and sea ice (NOAA OISST) | sea-surface temperature (°C) and sea-ice concentration (0–1, none below 0.15) — NOAA's daily optimum-interpolation analysis | 0.25° | daily means | 1982-01-01 → 2026-10-04 |
| `ncep100d` | Atmosphere and land (NCEP/NCAR reanalysis) | the 15 surface channels: wind stress and its variability, 2 m air and skin temperature, 10 m wind, pressure, rain and snow (log1p), soil moisture and temperature, latent and sensible heat flux | 1° | daily means | 1982-01-01 → 2026-03-17 |
| `occci025d` | Ocean colour (ESA OC-CCI) | chlorophyll-a (log₁₀ mg m⁻³) and the fraction of 4 km cells seen clear | 0.25° | daily means | 1997-09-04 → 2026-06-30 |

What is worth knowing before using them — the tab prints it under the store
and writes it into every file's `comment`:

- **Two of the four are reanalyses**, not measurements: GLORYS (ocean) and
  NCEP/NCAR (atmosphere) are models constrained by observations.
- **The two wind-stress variability channels (`tau_x_std`, `tau_y_std`) are
  a five-day spread centred on each day** — the standard deviation of the
  six-hourly stress over the five days around it — not a one-day value. The
  stored unit says "centred 5-day sigma"; the tab reads it as N/m² with that
  note.
- **Through 2024-12-31 each store reproduces the five-day tensor**: the mean
  of a bin's days by each channel's rule equals family 7.2 to float16
  rounding, NaN exactly where it is NaN, and every bin was checked. **From
  2025-01-01 there is no five-day tensor to check against**, so each day was
  checked instead against an independent read of its own source file — a
  weaker guarantee (it proves the right bytes went through the same code, not
  agreement with a second product). For GLORYS this matters most: the days
  after 2024 come from the same Copernicus dataset id
  (`cmems_mod_glo_phy_my_0.083deg_P1D-m`, version 202311) as the checked
  years, fetched later, with nothing independent to compare them with.
- **OISST's newest days are preliminary**: 2026-09-21 → 10-04 are NCEI's
  preliminary values, kept as published; a later rebuild replaces them.
- **NCEP's record ends 2026-03-17** because NOAA PSL has not updated its 2026
  files since 2026-03-19.
- **Licences**: GLORYS — Copernicus Marine Service data licence (free, with
  attribution); OISST and NCEP — NOAA public domain; OC-CCI — the ESA CCI data
  policy (free and open, cite). The attribution each producer asks for is
  printed under the store and written as the `license` and `attribution`
  global attributes of every NetCDF, as for the ERA5 stores.
- **A five-day mean the tab computes** (time step *five-day mean*) is the plain
  mean of the finite days. Family 7.2's pentad needs three finite days for
  OISST, and uses its own rule for current speed (the speed of the mean
  current) and mixed-layer depth (the log of the mean depth), so those can
  differ from the tensor where those rules bite.

### Cheap long-period averages (precomputed monthly sums)

Eight map stores also keep, for every year, calendar month, channel and cell,
the **sum** of that month's native maps, **how many** had a value, and the
**sum of squared deviations** from that month's own mean
([E-088](https://blauewelt.github.io/earth/docs.html?f=ml/plans/E088_monthly_sums_gridded.md),
the experiment that computed and verified them once, on a rented machine):

| store | family | grid | from |
|---|---|---|---|
| `oisst025d` | 7.2d (daily) | 0.25° | 1982 |
| `glorys025d` | 7.2d (daily) | 0.25° | 1993 |
| `ncep100d` | 7.2d (daily) | 1° | 1982 |
| `occci025d` | 7.2d (daily) | 0.25° | 1997 (from September) |
| `era5_t`, `era5_q`, `era5_u`, `era5_v` | 1.2 (six-hourly) | 1° | 1982 |

The index is the site's own `data/gridded_monthly_index.json`. A mean over any
set of (year, month) cells is Σ sums ÷ Σ counts; a spread is
(Σ m2 + Σ n·(month's mean − the mean)²) ÷ Σ n, the population form.

**When the tab uses them.** For these stores, the time steps *monthly mean*,
*one mean over the whole selection* and *normal — one mean per calendar month
over the period* are read from the precomputed sums **whenever the selection
is made of whole calendar months**: no day-of-month range and, for the
six-hourly ERA5 stores, no hour filter. Otherwise every native map is read,
exactly as before. The estimate box opens with one sentence saying which, and
why, and it counts the path it names:

- *Read from precomputed monthly sums — 2 requests, 56 kB. Whole calendar
  months with no day-of-month or hour filter: each (year, month) is one
  precomputed plane of sums and counts instead of its native maps. Reading its
  124 native maps instead would be 255 requests, 24.3 MB.* (ERA5 temperature
  at 500 hPa, January 2015, 30–60° N × 60–10° W)
- *Read from the native maps: your selection cuts months (days 1–10), so every
  native map is read.*

The native path is also taken for the native time step, five-day means (they
cross month edges), and — a safety rule — any month whose frames in the store
differ from the frames that were summed (a store updated since its sums were
made reads natively until they are remade; the sentence names the month).
**read every native map instead** (the *check* row) forces the native path, to
compare the two; the file name then ends in `_nativeread`.

**What it buys**, measured in the browser against the live store on
2026-10-07 (from the sandbox, where every request is relayed, so the seconds
are slow in absolute terms; the ratios are the point):

| selection | precomputed sums | every native map |
|---|---|---|
| ERA5 temperature 500 hPa, January 2015, 30–60° N × 60–10° W | 2 requests, 0.06 MB, 6 s | 248 requests, 24.2 MB, 166 s |
| OISST July 2018–2020, 35–40° N × 50–45° W | 6 requests, 0.45 MB, 5 s | 186 requests, 10.1 MB, 128 s |
| OISST July normal 1991–2020, same box | 60 requests, 4.5 MB, 43 s | ≈ 2,070 requests, 102 MB |
| ERA5 temperature 500 hPa, one mean over 1991–2020, 30–60° N × 60–10° W | 24 requests, 114 MB, 26 s | ≈ 89,856 requests, 8.3 GB — over the cap |

**The two paths agree.** Same selection through both: the means within the
float32 bound of the stored sums (Σ ½ulp(sum) ÷ Σ count per cell — in practice
identical to the last bit for one month, 1.5 × 10⁻⁵ K for a 30-year ERA5 mean
against numpy over the same planes), the counts identical. Checked in the
reader's tests on two small native stores and the sums made from them, and
live (`scripts/datatab_browser_check.mjs`, `scripts/f1data_live_check.mjs --e088`).

**Standard deviation** (the *spread* row, off by default): an extra variable
`<channel>_std` beside each mean — the spread of the native maps behind it, in
population form (divided by N, not N − 1). Offered for these stores on either
path. Checked live: OISST July 2019 in a 2° box against numpy's `nanstd` of the
31 native daily maps, max difference 9 × 10⁻⁸ °C.

**Counts and coverage.** `<channel>_count` is always written (finite native
frames per cell), and the file adds `frames_present` and `frames_possible`
per time step and a global attribute such as *930 of 930 possible frames* — so
"28 of 31 days" tells a gap upstream from a cloudy pixel. `read_path` and
`read_path_reason` record which way the file was read.

**A frame belongs to the calendar month of its own start (UTC).** A daily map
is its day; a six-hourly ERA5 map is its instant. This is NOT the rule of the
monthly normals below, where a five-day bin belongs to the month it OPENS in —
so a July normal of `oisst025d` and of `normals_g025` differ slightly at month
edges (a bin opening on 30 June is June to the tensor and five-sixths July
here), besides the tensor's own five-day rules.

ERA5's level picker works on this path as on the other (one level, the sums of
that level's channel); unlike the native tiles, the sums are stored per channel,
so one level reads one level.

### Derived maps

| store | plain-English name | kind | native space | native time | record |
|---|---|---|---|---|---|
| `fishing_grid` | Fishing effort map (AIS) — fishing hours and hours present, summed over each 0.25° cell and month | map | 0.25° | monthly sums | 2012-01 → 2024-12 |
| `normals_g025` | Monthly normals of the global tensor — sea-surface temperature, currents, sea-surface height, mixed layer and sea ice | normals | 0.25° | one sum and count per year and calendar month | 1982 → 2024 |
| `normals_g100` | Monthly normals of the global tensor — air–sea fluxes, weather and land | normals | 1° | per year and calendar month | 1982 → 2024 |
| `normals_oc025` | Monthly normals of the global tensor — ocean colour | normals | 0.25° | per year and calendar month | 1997 → 2024 (from September 1997) |
| `normals_rg100` | Monthly normals of the global tensor — ocean temperature and salinity at 16 depths | normals | 1° | per year and calendar month | 2004 → 2024 |

#### Monthly normals over a period you choose

These replaced, on 2026-10-06, the four fixed "all years" climatology stores
(`clim_g025` …), and with them the three fixed versions of the *Model
climatology* layer: *"No 'paper holdout' or similar anymore, just setting a
period is enough"* (Chris). The model climatology — what the forecaster calls
*normal* — is, per calendar month, channel and cell, the mean of the stored
value over the five-day bins of the chosen years that open in that month. A
mean is a sum divided by a count, and sums and counts add across years, so the
project published once, for every year, month, channel and cell, the **sum**
of that year's bins and **how many** had a value
([E-086](https://blauewelt.github.io/earth/docs.html?f=ml/plans/E086_monthly_by_year.md)).
The tab composes the mean over ANY set of years,

    mean = (Σ of the chosen years' sums) ÷ (Σ of their counts),   NaN where the counts sum to 0,

then turns it into the channel's unit (value = z × sd + mean, with the
tensor's own constants), so a file carries °C, m/s, PSU and so on.

- **Period**: a first and a last year, and **leave out** — whole years to
  drop ("2009, 2017"; "2009-2011" for a range). The paper's split is *1982 to
  2020, leave out 2009, 2017* — one click on the **paper split** chip (below,
  *The paper's climatology*); the period 1982–2024 reproduces the Model
  climatology layer's all-years normal.
- **Months**: the twelve chips; one normal per chosen month.
- **Time step**: *normal* — one mean per calendar month over the period (the
  file is marked a CF climatology: a `climatology_bounds` variable and the
  global attributes `period_start`, `period_end`, `excluded_years`,
  `years_used`); or *by year* — each year's own monthly mean side by side, so
  "the 23 Februaries 1982–2004" is 23 fields.
- **Counts are always written**: `<channel>_count` is the number of five-day
  bins behind each value (for `rg100`, the number of monthly rows — years).
- **Resolution**: native, plus 1° for the 0.25° groups. A 1° cell **pools**
  the sums and the counts of the sixteen 0.25° cells and every year in it
  (Σ sums ÷ Σ counts) — not a mean of the sixteen means, which would give a
  cell with one sample the same weight as a cell with forty.
- **Depths**: `normals_rg100` shows the variables × levels picker (dbar).
- **What a read costs.** A run of years for one month and channel is one
  contiguous range in each file (axes `[month, channel, year, lat, lon]`) —
  the whole globe at 0.25° is 178.6 MB of sums and 44.6 MB of counts for 43
  years. A box's latitude band is contiguous only *within* one year's map.
  The reader works out both ways and picks per selection: **year by year**
  (the box's band of each year's map) when the rows between two years' bands
  cost more than a request (about 0.5 MB), else **one stretch per month and
  channel** — which is every 1° group, whose whole map is 0.26 MB. Measured
  from the sandbox, February SST 1982–2024 over 35–45° N, 60–40° W at 0.25°:
  year by year 86 requests, 12.7 MB; one stretch 28 requests, 131.2 MB. Over
  0–60° N, 80° W–0°: 74.6 MB against 158.3 MB. The same small box at 1°
  (`g100`): 3 requests and 13.5 MB as one stretch, against 86 requests of
  0.85 MB year by year. The estimate line says which way it reads and what
  the other would cost. A stretch can pass over an excluded year's map; its
  values are never added.
- **First look**: the whole record, this calendar month, the Gulf Stream box
  — 12.7 MB for 0.25° SST, under 15 MB for every group.

#### The paper's climatology — the exact recipe

The forecaster's climatology — what it subtracts before training and scores
against — was published in three versions, and the tab reproduces each one:

1. **Store**: *Monthly normals of the global tensor* for the group with your
   channel — `normals_g025` (0.25° ocean surface), `normals_g100` (1° fluxes,
   weather and land), `normals_oc025` (ocean colour) or `normals_rg100` (Argo
   at depth).
2. **One chip** in the *splits* row, which sets the period, the years left
   out, all twelve months and the time step *normal — one mean per calendar
   month over the period*:
   - **paper split** — first year 1982, last year 2020, leave out 2009 and
     2017 (the paper's numbers);
   - **development split** — 1982–2024, leave out 2009, 2017 and 2023;
   - **all years** — 1982–2024, nothing left out (what the Model climatology
     layer paints).
3. **Size it**: a whole-globe twelve-month normal of one 0.25° channel reads
   about 2.3 GB of sums and counts — far over the 600 MB cap — so take **one
   month at a time** (about 190 MB for the whole globe; the chip stays lit) or
   a box. 1° cells shrink the file, not the read, of a 0.25° group. A 1°
   group's whole-globe twelve-month normal is about 150 MB per channel, inside
   the cap.
4. **Download** (NetCDF): a CF climatology whose `period_start`,
   `period_end` and `excluded_years` say which split it is.

**Only these stores reproduce the paper**: they use the model's own rule — a
five-day bin belongs to the month it opens in — and the tensor's own values;
the daily stores above use true calendar months and are close but not
identical.

Verified end to end on 2026-10-07 through the tab's Download button, each file
against the published plane read independently by range and turned into the
channel's unit with the index's mean and spread: sea-surface temperature
(`g025`), February, 35–45° N × 50–40° W, 1,681 cells — paper split max
difference 1.10 × 10⁻⁶ °C, development split 1.09 × 10⁻⁶ °C; 2 m air
temperature (`g100`), July, 40–60° N × 20° W–10° E, 651 cells — 1.53 × 10⁻⁶ °C
and 1.57 × 10⁻⁶ °C. Each is inside two float32 roundings (the published file's
standard score times the spread, and the tab's own float32 value), with NaN in
exactly the same cells.

The published whole-globe files of the three versions (from
`data/family7_clim_index.json`, which has every size and sha256; `clim.nc` is
in physical units, `clim.npy` is `[12, C, H, W]` float32 standard scores —
value = z × sd + mean with the constants in that version's `stats.json`):

- paper split, g025 (0.25° ocean surface (sea-surface temperature, height, currents, mixed layer, sea ice)): [clim/paper/g025/clim.nc](https://huggingface.co/datasets/chfrank/earth-tensors/resolve/main/tensors/family7_global025_pentad_l2/clim/paper/g025/clim.nc) (150 MB, physical units) · [clim.npy](https://huggingface.co/datasets/chfrank/earth-tensors/resolve/main/tensors/family7_global025_pentad_l2/clim/paper/g025/clim.npy) (349 MB, standard scores)
- paper split, g100 (1° air–sea fluxes, weather and land): [clim/paper/g100/clim.nc](https://huggingface.co/datasets/chfrank/earth-tensors/resolve/main/tensors/family7_global025_pentad_l2/clim/paper/g100/clim.nc) (30 MB, physical units) · [clim.npy](https://huggingface.co/datasets/chfrank/earth-tensors/resolve/main/tensors/family7_global025_pentad_l2/clim/paper/g100/clim.npy) (47 MB, standard scores)
- paper split, oc025 (0.25° ocean colour): [clim/paper/oc025/clim.nc](https://huggingface.co/datasets/chfrank/earth-tensors/resolve/main/tensors/family7_global025_pentad_l2/clim/paper/oc025/clim.nc) (45 MB, physical units) · [clim.npy](https://huggingface.co/datasets/chfrank/earth-tensors/resolve/main/tensors/family7_global025_pentad_l2/clim/paper/oc025/clim.npy) (100 MB, standard scores)
- paper split, rg100 (1° Argo temperature and salinity at depth): [clim/paper/rg100/clim.nc](https://huggingface.co/datasets/chfrank/earth-tensors/resolve/main/tensors/family7_global025_pentad_l2/clim/paper/rg100/clim.nc) (32 MB, physical units) · [clim.npy](https://huggingface.co/datasets/chfrank/earth-tensors/resolve/main/tensors/family7_global025_pentad_l2/clim/paper/rg100/clim.npy) (100 MB, standard scores)
- development split, g025 (0.25° ocean surface (sea-surface temperature, height, currents, mixed layer, sea ice)): [clim/dev/g025/clim.nc](https://huggingface.co/datasets/chfrank/earth-tensors/resolve/main/tensors/family7_global025_pentad_l2/clim/dev/g025/clim.nc) (150 MB, physical units) · [clim.npy](https://huggingface.co/datasets/chfrank/earth-tensors/resolve/main/tensors/family7_global025_pentad_l2/clim/dev/g025/clim.npy) (349 MB, standard scores)
- development split, g100 (1° air–sea fluxes, weather and land): [clim/dev/g100/clim.nc](https://huggingface.co/datasets/chfrank/earth-tensors/resolve/main/tensors/family7_global025_pentad_l2/clim/dev/g100/clim.nc) (30 MB, physical units) · [clim.npy](https://huggingface.co/datasets/chfrank/earth-tensors/resolve/main/tensors/family7_global025_pentad_l2/clim/dev/g100/clim.npy) (47 MB, standard scores)
- development split, oc025 (0.25° ocean colour): [clim/dev/oc025/clim.nc](https://huggingface.co/datasets/chfrank/earth-tensors/resolve/main/tensors/family7_global025_pentad_l2/clim/dev/oc025/clim.nc) (45 MB, physical units) · [clim.npy](https://huggingface.co/datasets/chfrank/earth-tensors/resolve/main/tensors/family7_global025_pentad_l2/clim/dev/oc025/clim.npy) (100 MB, standard scores)
- development split, rg100 (1° Argo temperature and salinity at depth): [clim/dev/rg100/clim.nc](https://huggingface.co/datasets/chfrank/earth-tensors/resolve/main/tensors/family7_global025_pentad_l2/clim/dev/rg100/clim.nc) (32 MB, physical units) · [clim.npy](https://huggingface.co/datasets/chfrank/earth-tensors/resolve/main/tensors/family7_global025_pentad_l2/clim/dev/rg100/clim.npy) (100 MB, standard scores)
- all years, g025 (0.25° ocean surface (sea-surface temperature, height, currents, mixed layer, sea ice)): [clim/all/g025/clim.nc](https://huggingface.co/datasets/chfrank/earth-tensors/resolve/main/tensors/family7_global025_pentad_l2/clim/all/g025/clim.nc) (150 MB, physical units) · [clim.npy](https://huggingface.co/datasets/chfrank/earth-tensors/resolve/main/tensors/family7_global025_pentad_l2/clim/all/g025/clim.npy) (349 MB, standard scores)
- all years, g100 (1° air–sea fluxes, weather and land): [clim/all/g100/clim.nc](https://huggingface.co/datasets/chfrank/earth-tensors/resolve/main/tensors/family7_global025_pentad_l2/clim/all/g100/clim.nc) (30 MB, physical units) · [clim.npy](https://huggingface.co/datasets/chfrank/earth-tensors/resolve/main/tensors/family7_global025_pentad_l2/clim/all/g100/clim.npy) (47 MB, standard scores)
- all years, oc025 (0.25° ocean colour): [clim/all/oc025/clim.nc](https://huggingface.co/datasets/chfrank/earth-tensors/resolve/main/tensors/family7_global025_pentad_l2/clim/all/oc025/clim.nc) (45 MB, physical units) · [clim.npy](https://huggingface.co/datasets/chfrank/earth-tensors/resolve/main/tensors/family7_global025_pentad_l2/clim/all/oc025/clim.npy) (100 MB, standard scores)
- all years, rg100 (1° Argo temperature and salinity at depth): [clim/all/rg100/clim.nc](https://huggingface.co/datasets/chfrank/earth-tensors/resolve/main/tensors/family7_global025_pentad_l2/clim/all/rg100/clim.nc) (32 MB, physical units) · [clim.npy](https://huggingface.co/datasets/chfrank/earth-tensors/resolve/main/tensors/family7_global025_pentad_l2/clim/all/rg100/clim.npy) (100 MB, standard scores)

Not offered: the 6.25 km sea-ice store from the University of Bremen stays
private until Bremen answers on redistribution — a browser has no access token
and must not have one.

## 2. The controls, top to bottom

1. **Store and channels.** The store list is grouped by family and shows each
   store's plain-English title with its code name beside it; under it, one
   sentence on what the store is, its native resolution and its record. Tick
   one or more channels. A store measured at depths (the Argo grid and
   profiles, the casts, the floats, the moorings) shows two rows instead of a
   long list: the **variables** (temperature, salinity, …) and the **levels**
   (pressures in dbar for the ocean, hPa for the atmosphere), with *all* /
   *none*; the channels are every variable at every level that is on. ERA5
   opens on one level, 500 hPa.
2. **Years.** A start and an end year, clamped to the store's record. A
   normals store adds **leave out** (whole years to drop from the average),
   the three **splits** chips (*paper split*, *development split*, *all
   years* — a chip is lit while the years and the years left out are its
   split) and has no days. The eight stores with precomputed monthly sums
   have **leave out** too (years left out of every mean).
3. **Months.** Twelve chips with *all* / *none*. "Every February from 1998 to
   2004" is: years 1998 to 2004, only *Feb* on.
   **Days.** A day-of-month range applied inside every chosen month (*1 to 31*
   is the whole month). On a five-day grid a frame belongs to the day it
   starts on; a monthly store has no days. It is the finest period control, and the one that
   makes the densest stores downloadable at all: a month of SWOT swaths is
   about 3 GB to read, a single day about 50–150 MB.

   **What a store opens on.** Unless you have chosen a period yourself, each
   store opens on its own *first look*: the most recent month that has data,
   shortened to its first 10, 5, 2 or 1 days until the estimate is at most
   about 40 MB to read and to build — so the first estimate you see is always
   downloadable. (The stores differ too much for one fixed default: a year of
   bottle samples is 0.3 MB, a year of SWOT 36 GB.) Once you set the years,
   months, days, hours or box yourself, that choice carries over when you
   switch stores.
4. **Hours (UTC).** Shown only where the store has a time of day finer than a
   day — the 3-hourly cloud tops, the six-hourly ERA5 maps and every point
   store. The range runs from
   the first hour up to, not including, the second: *0 to 6* keeps the 00:00
   and 03:00 cloud-top maps. *0 to 24* is the whole day; *22 to 2* wraps
   across midnight.
5. **Box.** West, south, east and north edges in degrees. *Use the current view*
   copies the part of the globe on screen; the six place buttons (the same
   places the Cones tab uses) put a box of ±5° latitude × ±7.5° longitude round
   each. A box whose west edge is east of its east edge crosses the dateline,
   and that is allowed. The box is drawn on the globe while the tab is open. A
   map store **needs** a box; a point store may leave it empty for the whole
   globe. The cloud-top store covers only 30°S–30°N, so its first box is the
   Niño 3.4 place; a box outside a store's coverage is said in words and the
   download stays off.

   **Drawing the box on the globe.** *draw on the globe* arms a mode (the
   button turns yellow and a chip over the globe says what to do next).
   On a computer, click one corner and then the opposite one — a dashed
   rectangle follows the pointer in between — or press, drag and release. On a
   touch screen, tap two corners, or drag with one finger. Only that one
   gesture is taken from the globe: unarmed, dragging rotates it as always,
   and a second finger joining a drawing gesture hands it back for a pinch.
   *Esc* or the chip's *Cancel* stops drawing and leaves the box as it was.
   The corners are rounded to 0.01° and written into the four fields; nothing
   else is snapped (the reader takes the cells whose centres fall inside, as
   it always has). Two longitudes are joined the **shorter way round**, so
   170° E and 170° W make a 20°-wide box across the dateline (W > E);
   *the other way round* swaps W and E for the 340°-wide one. The drawn box
   keeps **eight handles** — the four corners and the middles of the four
   edges — to drag; a handle moves only its own edges. While drawing (and for a
   moment after the last click) a click is a corner, not a question: the pixel
   inspector, the value read-out and the pick cards stay closed. Corners come
   from the globe's surface, not from what is drawn on it, so a place name
   under the pointer cannot swallow a click.
6. **Time step.** *Native* keeps every frame (or every report); *five-day mean*,
   *monthly mean* and *one mean over the whole selection* average in time. The
   eight stores with precomputed monthly sums add *normal — one mean per
   calendar month over the period* (with *monthly mean* as its by-year stack),
   a **spread** row (*standard deviation too*) and a **check** row (*read every
   native map instead*) — §1, *Cheap long-period averages*.
7. **Resolution.** For a map store: *native*, 0.25° or 1°, the coarser two being
   an average over the native pixels in each cell. For a point store: *rows*
   (every report as it is, with its own time) or 0.25° / 1° **cells**. Cells
   are an average over a cell *and* a stretch of time, so they need one of the
   three time means — the page switches a native time step to the five-day
   mean and says so. Every mean skips missing values and comes with a per-cell
   **count** of the observations it averaged.
8. **Estimate.** Updated as you change anything, before any data is read:
   the number of requests, the megabytes to read, what the file will hold, and
   the megabytes of the file. With the native time step it counts maps (or
   rows); with a time mean it says both what is read and what is kept —
   "30 daily maps read → 1 monthly mean" — because the read follows the first
   number and the file the second. A selection with no data in it (a period
   outside the record, a box outside the coverage) is said in words. Over the cap the download button is
   off, the line says which limit is binding and what to shrink, and it links
   the store's folder on the data store so you can fetch the files directly.
9. **Preview on the globe.** Reads one frame (or one five-day bin of points),
   the first one in the selection that has data, and paints it inside the box —
   a picture for a map store, dots for a point store — with a legend. With
   more than one channel ticked, a *showing* picker beside the legend switches
   channel without reading anything again. The colour scale is the channel's
   full range from the registry whenever the frame spans at least half of it,
   so the same colour means the same value from one preview to the next;
   otherwise it is the frame's own range (on the full range, a December frame
   of chlorophyll painted as one flat colour), and the legend says which and
   prints the other. The preview disappears when you change the selection or
   leave the tab.
10. **Download.** *NetCDF* for map stores and binned points; *CSV* for rows,
    and for map selections small enough that a `time,lat,lon,value` text file
    stays something a spreadsheet can open. A progress bar runs while the
    store is read; *Cancel* stops the reads in flight and saves nothing.
11. **Waiting: the progress block.** Every read that takes more than about
    0.4 s — the estimate (inside the estimate box), the first look a store
    opens on, the preview (under its button) and the download (under its row)
    — shows the same block:
    - a **bar**: a fraction where the total is known, running without a
      percentage where it is not;
    - a **line**: "N of M requests · X MB of Y MB · 12 s · about 20 s left";
    - a **why**, for a slow read whose cause the counts show — "many small
      requests (each a round trip, at most six at a time)", "a lot of data
      (MB at MB/s)", or "still reading the store's indexes";
    - **Cancel**, which aborts every request in flight and leaves the panel
      clean (a cancelled estimate says so and offers *estimate again*;
      nothing is downloadable until an estimate stands).

    Where the totals come from, case by case: a **download**'s data phase knows
    its requests and its exact bytes (the normals and the global-tensor grids),
    or an upper bound (a point store's rows, marked ≈); the 4 km tiled grids
    learn their tile count as their indexes arrive, so the megabytes are shown
    against the estimate's figure (≈). Before that, every read (estimate,
    preview or download) first reads the store's indexes to plan, and that
    phase has no total anyone could know: it counts requests and megabytes so
    far. The exception is a tiled grid's estimate, which knows how many
    five-day files' tile indexes it will read (all of them, or a sample of
    them) and counts them off. The **first look** is a search that stops at the
    first month (or the first 10, 5, 2 or 1 days) that fits, so it never has a
    total: it says which candidate it is trying and counts across all of them.
    The time left appears only after 5 s, and only while the rate over the last
    4 s agrees with the rate over the whole read to within a third — an
    unsteady rate gets no guess. The bar's stripes stand still under the
    system's *reduce motion* setting.

## 3. Limits — what the estimate is telling you

- **The cap.** At most 600 MB read or 400 MB of result, whichever binds first.
  One five-day bin of the whole globe from `oc4k` is about 108 MB, so a year of
  the whole globe (about 8 GB) is refused and the estimate links the store's
  files instead; the same year over the North Atlantic is a few hundred
  megabytes.
- **Coarser is not cheaper to read.** A coarser resolution and a time mean are
  computed in your browser *after* the native bytes are read. They shrink the
  file, not the download. That is why the estimate prints both numbers.
- **Point stores read the whole period.** The box is applied in the browser
  after the reports' positions have been read, so a short period is cheap and a
  long one is not, whatever the box. The row count is exact: the store records
  where every five-day bin starts.
- **No silent holes.** Every read asks for an exact byte range and refuses any
  other answer. If a read fails the whole download stops and the error, with
  the address that failed, is printed in the panel. A file you save is never
  missing a part without saying so.

## 4. What the values are

- **`sst_acspo02`** is what the satellites saw: it has gaps wherever there was
  cloud. It is not a gap-filled analysis, and it is not a finer version of the
  0.25° OISST analysis the rest of the project uses.
- **`irtb`** is stored as a whole number of kelvin minus 160 (one byte per
  pixel), 3-hourly; the source is half-hourly. The file carries kelvin.
- **`oc4k`'s chlorophyll** is the base-10 logarithm of the concentration in
  mg m⁻³, not the concentration itself: 0 is 1 mg m⁻³, −1 is 0.1 mg m⁻³.
- **ERA5** (family 1.2) is a reanalysis — a model's analysis, not a
  measurement (§1). Temperature is in kelvin, humidity in **g/kg** (not
  ERA5's kg/kg: 1000 × it, as stored), winds in m/s. Each map is an instant
  at 00, 06, 12 or 18 UTC.
- **The family-10 grids** are the physical values recovered from z-scores
  (see §1), to within the 16-bit storage: about one part in a thousand of the
  channel's spread. Channels named `log_…` are logarithms as the forecaster
  sees them.
- **`rg100`** is monthly: one frame per calendar month, dated to the month's
  first day (the store files each month under the five-day bin that holds its
  15th). The days range does not apply to it.
- **`fishing_grid`** holds a sum of vessel-hours per 0.25° cell and month. A
  coarser cell or a longer time step *averages* those sums (a mean per 0.25°
  cell and month); it does not add them up. Zero means no vessel broadcast
  there, which is not the same as no fishing.
- **The monthly normals** are in the channel's unit (composed in z-units and
  converted once). A *normal* file's time is each month in the first year of
  the period, with `climatology_bounds` spanning the period; a *by-year*
  file's time is the first of each month.
- **Family 7.2d** values are daily, physical, float16 as stored; each frame is
  dated to its day at 00:00 UTC.
- **Means from the precomputed monthly sums** (family 7.2d, ERA5) are the same
  means as averaging every native map, to within float32 rounding of the
  stored sums; a frame counts in the calendar month of its own start. A
  *normal* file's time is each month in the first year of the period, with
  `climatology_bounds`, as for the monthly normals. `<channel>_std`, where
  asked for, is the population standard deviation.
- Every file states its units and carries, in its global attributes, the store
  and the full selection that produced it.

## 5. Where the code is

| part | file |
|---|---|
| the reader: registry, estimate, range reads, means, NetCDF and CSV writers (`window.F1Data`) | `src/f1data.js` |
| the zstd decoder the reader uses for the compressed tiles | `lib/fzstd.js` |
| the tab: controls, box on the globe, estimate line, preview, save | `src/app.js` (the block headed "data tab"), `index.html` (`#panel-data`), `src/style.css` |
| tests of the tab | `tests/app.spec.js` ("Data tab: …") |
| tests of the reader, and the small fixtures they read | `tests/f1data.test.mjs`; `data/family1_fixture/`, `data/family12_fixture/` (an ERA5-shaped store with levels and six-hourly frames), `data/family10_fixture/` and `data/family7_monthly/fixture_multi/` (per-year monthly sums and counts, five years, two small grids), written by `tests/make_family1_fixture.py`, `tests/make_family12_fixture.py`, `tests/make_family10_fixture.py` and `tests/make_family7_monthly_fixture.py`; `data/gridded_monthly/fixture/` (per-year monthly sums, counts and m2 of two tiny stores) with `fixture_native/` (the native stores they were made from, `tests/make_gridded_paths_fixture.py`), so one selection is read both ways |
| a check against the live data store, with independent Python reads | `scripts/f1data_live_check.mjs` (`--e088` for the precomputed sums and the paper's climatology only) |
| the same, in a real browser through the tab's own controls and Download button | `scripts/datatab_browser_check.mjs` |

The reader keeps one handler per layout — the zstd-tiled family-1.gf grids
(also family 1.2's and family 7.2d's), the five-day-binned point stores (both
schemas, including bins before 1982), the bin-major family-10 grids, the
month-major fishing map, the per-year monthly sums and counts of the normals,
and the calendar-month climatology (still readable, no longer listed) — behind
the same interface. The tab talks to the
reader only through the six functions in the plan's
§4 (`loadRegistry`, `estimate`, `run`, `preview`, `toNetCDF`, `toCSV`), so the
reader can change how it reads without the tab noticing.

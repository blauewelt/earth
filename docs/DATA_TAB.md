# The Data tab — download the project's stores from your browser

The **Data** tab (between *Cones* and *Play*) lets you pick one of the project's
data stores, narrow it to the period, months, hours, region and
resolution you want, see what that will cost before anything is read, preview it
on the globe, and save it as a NetCDF or CSV file. The file is built **in your
browser** from the store's own files: no server computes anything.

The store list is the union of three groups, each read from its own registry:

- **Fine observations (family 1.gf)** — the global set of observation stores
  at 10 km / 5 days or finer, each kept at its own native resolution rather
  than resampled onto a common grid; registry `tensors/family1_gf/family1gf.json`.
- **Global tensor and point observations (family 10)** — the 0.25° and 1°
  five-day grids the forecaster is trained on, the monthly Argo grid at 16
  depths, and the point stores beside them (Argo profiles, drifters, moored
  buoys, ship CO₂, altimeter tracks, fishing vessels); registry
  `tensors/family10_2/family10.json`.
- **Derived maps** — maps the project computes from those: the 0.25° monthly
  fishing-effort map and the model climatology (what the forecaster calls
  "normal", one map per calendar month). Their indexes are the site's own
  `data/fishing_index.json` and `data/family7_clim_index.json`.

All the data live on the project's public data store, the Hugging Face dataset
[chfrank/earth-tensors](https://huggingface.co/datasets/chfrank/earth-tensors).
A future **family 1.2** registry (`tensors/family1_2/family12.json`) is one
line in the reader's registry list; until it is published it is simply not
offered. If one registry or one store cannot be read, the others still load and
the tab prints a line naming what is unavailable and why. The plan that specified this tab and its reader is
[E-084](https://blauewelt.github.io/earth/docs.html?f=ml/plans/E084_data_tab.md);
the stores themselves are described in the
[family 1.gf design note](https://blauewelt.github.io/earth/ml/paper/notes/family1gf.pdf).

## 1. The stores

The tab reads the list of stores from the registries at run time, so a store
appears in the tab the day it is published; these tables are the set as of
2026-10-05 (27 stores). Three kinds:

- a **map store** (a gridded product): one value per pixel per frame;
- a **point store**: every report at its own position and time — a ship's
  observation, a float's profile, a bottle sample, a satellite sounding;
- a **calendar store** (the climatology): twelve maps, one per calendar month,
  with no years at all.

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

### Derived maps

| store | what it is | kind | native space | native time | record |
|---|---|---|---|---|---|
| `fishing_grid` | fishing hours and hours present, summed over each 0.25° cell and month | map | 0.25° | monthly sums | 2012-01 → 2024-12 |
| `clim_g025` | the model climatology of the `g025` channels | calendar | 0.25° | 12 calendar months | no years; the average over 1982–2024 |
| `clim_g100` | the model climatology of the `g100` channels | calendar | 1° | 12 calendar months | no years; the average over 1982–2024 |
| `clim_oc025` | the model climatology of the `oc025` channels | calendar | 0.25° | 12 calendar months | no years; the average over the record |
| `clim_rg100` | the model climatology of `rg100`, 16 pressures | calendar | 1° | 12 calendar months | no years; the average over the record |

The climatology is in z-units: 0 means "the long-term mean of that channel",
1 means one spread above it. It is what the forecaster is trained and scored
against, so it is the reference for its anomalies.

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
   (pressures in dbar), with *all* / *none*; the channels are every variable
   at every level that is on.
2. **Years.** A start and an end year, clamped to the store's record. A
   calendar store has no years: the year and day rows are hidden and only the
   months are offered.
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
   day — the 3-hourly cloud tops and every point store. The range runs from
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
6. **Time step.** *Native* keeps every frame (or every report); *five-day mean*,
   *monthly mean* and *one mean over the whole selection* average in time.
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
- **The climatology** is in z-units (0 is normal), and the year in its file's
  time axis is a nominal 2000 that means nothing.
- Every file states its units and carries, in its global attributes, the store
  and the full selection that produced it.

## 5. Where the code is

| part | file |
|---|---|
| the reader: registry, estimate, range reads, means, NetCDF and CSV writers (`window.F1Data`) | `src/f1data.js` |
| the zstd decoder the reader uses for the compressed tiles | `lib/fzstd.js` |
| the tab: controls, box on the globe, estimate line, preview, save | `src/app.js` (the block headed "data tab"), `index.html` (`#panel-data`), `src/style.css` |
| tests of the tab | `tests/app.spec.js` ("Data tab: …") |
| tests of the reader, and the small fixtures they read | `tests/f1data.test.mjs`; `data/family1_fixture/` and `data/family10_fixture/`, written by `tests/make_family1_fixture.py` and `tests/make_family10_fixture.py` through the project's own store writers |
| a check against the live data store, with independent Python reads | `scripts/f1data_live_check.mjs` |

The reader keeps one handler per layout — the zstd-tiled family-1.gf grids,
the five-day-binned point stores (both schemas, including bins before 1982),
the bin-major family-10 grids, the month-major fishing map and the
calendar-month climatology — behind the same interface. The tab talks to the
reader only through the six functions in the plan's
§4 (`loadRegistry`, `estimate`, `run`, `preview`, `toNetCDF`, `toCSV`), so the
reader can change how it reads without the tab noticing.

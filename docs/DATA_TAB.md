# The Data tab — download the fine observation stores from your browser

The **Data** tab (between *Cones* and *Play*) lets you pick one of the project's
fine observation stores, narrow it to the period, months, hours, region and
resolution you want, see what that will cost before anything is read, preview it
on the globe, and save it as a NetCDF or CSV file. The file is built **in your
browser** from the store's own files: no server computes anything.

The stores are the ones the project calls **family 1.gf** — the global set of
observation stores at 10 km / 5 days or finer, each kept at its own native
resolution rather than resampled onto a common grid. They live on the project's
public data store, the Hugging Face dataset
[chfrank/earth-tensors](https://huggingface.co/datasets/chfrank/earth-tensors),
under `tensors/family1_gf/`. The plan that specified this tab and its reader is
[E-084](https://blauewelt.github.io/earth/docs.html?f=ml/plans/E084_data_tab.md);
the stores themselves are described in the
[family 1.gf design note](https://blauewelt.github.io/earth/ml/paper/notes/family1gf.pdf).

## 1. The stores

The tab reads the list of stores from the store registry at run time, so a store
appears in the tab the day it is published; this table is the set as of
2026-10-05. Two kinds:

- a **map store** (a gridded satellite product): one value per pixel per frame;
- a **point store**: every report at its own position and time — a ship's
  observation, a float's profile, a bottle sample, a satellite sounding.

| store | what it measures | kind | native space | native time | record |
|---|---|---|---|---|---|
| `oc4k` | ocean colour: chlorophyll (as its base-10 logarithm), water clarity (`kd_490`), observation count | map | 4 km (1/24°) | daily | 1997-09 → 2022-12 |
| `pace4k` | chlorophyll, particulate carbon, phytoplankton carbon and four more, from NASA's PACE satellite | map | 4 km | daily | 2024-03 → 2026-09 |
| `sst_acspo02` | sea-surface temperature as satellites saw it, clear sky only, with a quality grade | map | 0.02° (2 km) | daily | 2000-02 → 2026-09 |
| `irtb` | cloud-top brightness temperature from geostationary satellites | map | 4 km | 3-hourly as stored | 1998 → 2026-09 |
| `icoads` | ship and buoy reports: sea and air temperature, pressure, wind, dew point, waves, cloud | points | per report | per report | 1662 → 2026 |
| `wod` | ship casts: temperature, salinity, oxygen and more at 16 pressures | points | per cast | per cast | 1772 → 2026 |
| `glodap` | bottle samples: carbon, alkalinity, pH, oxygen, nutrients | points | per bottle | per bottle | 1972 → 2023 |
| `bgcargo` | biogeochemical floats: oxygen, nitrate, pH, chlorophyll and more at 16 pressures | points | per profile | per profile | 2002 → 2026 |
| `oceansites` | open-ocean moorings | points | per mooring | hourly | 1980 → 2026 |
| `xco2` | column carbon dioxide soundings from three satellites (OCO-2, OCO-3, GOSAT) | points | under 3 km² | per sounding | 2009 → 2026 |
| `swh` | significant wave height along satellite altimeter tracks | points | about 7 km | one per second along track | 1991 → 2023 |
| `swot` | sea-level anomaly on the SWOT satellite's 2 km swaths | points | 2 km | per pass | 2023-07 → 2026-09 |

Not offered: the 6.25 km sea-ice store from the University of Bremen stays
private until Bremen answers on redistribution — a browser has no access token
and must not have one.

## 2. The controls, top to bottom

1. **Store and channels.** The store list shows each store's plain-English title
   with its code name beside it; under it, one sentence on what the store is,
   its native resolution and its record. Tick one or more channels.
2. **Years.** A start and an end year, clamped to the store's record. The
   default is the store's most recent year, which keeps the first estimate you
   see under the cap.
3. **Months.** Twelve chips, all on by default, with *all* / *none*. "Every
   February from 1998 to 2004" is: years 1998 to 2004, only *Feb* on.
4. **Hours (UTC).** Shown only where the store has a time of day finer than a
   day — the 3-hourly cloud tops and every point store. *0 to 24* is the whole
   day; *22 to 2* wraps across midnight.
5. **Box.** West, south, east and north edges in degrees. *Use the current view*
   copies the part of the globe on screen; the six place buttons (the same
   places the Cones tab uses) put a box of ±5° latitude × ±7.5° longitude round
   each. A box whose west edge is east of its east edge crosses the dateline,
   and that is allowed. The box is drawn on the globe while the tab is open. A
   map store **needs** a box; a point store may leave it empty for the whole
   globe.
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
   the number of requests, the megabytes to read, the number of frames (or
   rows), and the megabytes of the file. Over the cap the download button is
   off, the line says which limit is binding and what to shrink, and it links
   the store's folder on the data store so you can fetch the files directly.
9. **Preview on the globe.** Reads one frame (or one five-day bin of points),
   the first one in the selection that has data, and paints it inside the box —
   a picture for a map store, dots for a point store — with a legend for the
   first ticked channel. The colour scale is the channel's full range from the
   registry, so the same colour means the same value from one preview to the
   next; the legend also prints the range actually in the preview. The preview
   disappears when you change the selection or leave the tab.
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
- Every file states its units and carries, in its global attributes, the store
  and the full selection that produced it.

## 5. Where the code is

| part | file |
|---|---|
| the reader: registry, estimate, range reads, means, NetCDF and CSV writers (`window.F1Data`) | `src/f1data.js` |
| the zstd decoder the reader uses for the compressed tiles | `lib/fzstd.js` |
| the tab: controls, box on the globe, estimate line, preview, save | `src/app.js` (the block headed "data tab"), `index.html` (`#panel-data`), `src/style.css` |
| tests of the tab | `tests/app.spec.js` ("Data tab: …") |

The tab talks to the reader only through the six functions in the plan's
§4 (`loadRegistry`, `estimate`, `run`, `preview`, `toNetCDF`, `toCSV`), so the
reader can change how it reads without the tab noticing.

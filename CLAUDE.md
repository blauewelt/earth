# earth — standing instructions & project documentation

This file is the durable memory of the project. Read it before changing
anything; update it when a convention changes or a feature lands. It has two
halves: **standing instructions** (rules that govern all future work) and a
**holistic record** of what has been built and why.

Live app: https://blauewelt.github.io/earth/ · Repo: github.com/blauewelt/earth

---

## Part 1 · Standing instructions

### 1. Deploy first

Deploy **before** running the full test suite: commit, `git branch -f gh-pages
main`, `git push origin main gh-pages -f`, then run the affected tests, then
(optionally) broader regression. The user wants to try features immediately;
tests catch regressions after the fact. Never gate a deploy on a long test run.

### 2. Every dataset/layer ships complete

A new layer is not done until it has **all** of:

1. **A clickable documentation link** — the layer title links to the dataset's
   public docs (`doc` field / `title-link`).
2. **A hover card** (`.layer-tip`) with four elements:
   - **Gist paragraph** (`sum` in `LAYER_FACTS`, or a `<p class="tip-sum">` for
     static layers): 2–4 sentences giving the dataset's essence — what is
     measured, how, and why it matters for climate — so nobody needs to click
     through to understand the layer. *This is a standing requirement for every
     current and future dataset.*
   - **Recorded**: when the measurement record starts and, if closed (e.g. a
     1991–2020 normal), when it ends. Never write a bare "from \<date\>" — it
     reads as the data's date instead of availability.
   - **Wording must be unambiguous to a non-expert.** Established patterns:
     "this map: 2022-10 → present · MODIS has measured since 2000, but older
     dates aren't served as map tiles" (tile availability ≠ instrument
     record); "average of the years 1991–2020 (… not one date)" for
     climatologies; "fixed — ignores the date selector" for static composites.
     A shorthand like "MODIS record from 2000" was misread as "data fixed to
     the year 2000" — spell it out. If a layer's coverage is inherently patchy
     (clear-sky-only products like MODIS LST), say so in the gist ("the gaps
     are clouds, not missing data").
   - **Interval**: the time granularity (30-min, daily, monthly composite,
     single 30-year normal, "each float profiles every ~10 days", …).
   - **Spatial**: pixel/grid size, or point semantics ("one point per
     facility").
3. **A legend** if colormapped (GIBS colormap-driven or ramp-driven for grid
   layers) with hover value read-out.
4. **Value probe support** — click/dwell on the globe reads the actual value.
5. **An explicit aggregation/difference decision.** Every timed raster layer
   must declare one of three postures, and the choice must be justified in a
   code comment next to the flag:
   - `deltaRange: <n>` — continuous field: both time-averaging (Aggregate
     slider) and per-pixel differencing are sound.
   - `aggregable: true` — averaging over a window is sound (fills swath/cloud
     gaps) but day-vs-day differencing is not (log-scaled or too erratic).
   - neither — the layer is shown as-is (photographic composites,
     instantaneous sparse fields like precipitation).

   Current matrix (keep in sync when adding layers):

   | Layer | Aggregate | Difference | Why |
   |---|---|---|---|
   | SST (MUR), SST anomalies | ✓ | ✓ | continuous, gap-free L4 |
   | Sea ice (AMSR2) | ✓ | ✓ | continuous fraction |
   | Snow cover (NDSI) | ✓ | ✓ | continuous %, clear-sky gaps fill by averaging |
   | Land surface temp (MODIS) | ✓ | ✓ | continuous K, clear-sky gaps fill by averaging |
   | Salinity (SMAP monthly) | ✓ | ✓ | continuous PSU; sample dates snap & dedupe to months |
   | Chlorophyll-a (PACE) | ✓ | ✗ | log-scaled; differencing bin-centres of a log palette is unsound |
   | Aerosol optical depth | ✓ | ✗ | windowed mean is standard; day-vs-day is noise |
   | Precipitation (IMERG daily & 30-min) | ✗ | ✗ | instantaneous, log, mostly transparent — use the climatology grids |
   | True colour, night lights | ✗ | ✗ | photographs, no colormap to invert |
   | Grid climatologies | ✗ | ✗ | already multi-decade averages, not timed |
6. **Catalog consistency** — the dataset exists in `data/catalog.json`; set
   `globe: true` and append "Live globe layer in this app." to its notes.
7. **Tests** — at least one behavioural test in `tests/app.spec.js` and, if it
   has a data snapshot, a schema/sanity test in `tests/data.spec.js`.

### 3. Data pipeline: static snapshots, never live third-party calls

The browser must depend only on NASA GIBS (tiles) and GBIF (occurrence tiles).
Everything else is baked offline by `scripts/refresh_data.py` into small static
JSON files under `data/` (one function per dataset, runnable individually:
`python3 scripts/refresh_data.py gpcp eobs`). Grids use the common format
written by `_write_grid()` (regular lon/lat, row-major from the south, `null`
for empty cells) and render client-side via `GridProvider`.

### 4. Testing in the sandbox

The dev sandbox's *browser* cannot reach external hosts (curl can). Therefore:

- `MIRROR=1` reroutes cdnjs → `_vendor/cesium`, GIBS → `localhost:8081`,
  GBIF → `localhost:8082` (see `tests/app.spec.js` beforeEach).
- The proxies are **in the repo**: `scripts/test_proxy.py` (forwarding proxy)
  and `scripts/run_tests.sh` (starts servers + runs the suite). Do not recreate
  them ad hoc.
- Background processes die between separate shell invocations — start servers
  **and** run playwright in the *same* command.
- The default `playwright.config.js` `webServer` block can hang in the sandbox;
  when it does, use a temporary config without `webServer` against a manually
  started `python3 -m http.server 8080`.
- CI (GitHub Actions) uses the real network; MIRROR is sandbox-only.
- The vendored Cesium build mangles class names — assert on our own classes
  (e.g. `GIBSGeographicTilingScheme`) rather than Cesium constructor names.

### 4b. Date-independence must be announced

Enabling any layer with no per-date data fires an animated warning toast
(`showToast` / `datelessToast(id)`) so the date selector's lack of effect is
never a silent mystery. This applies to grid climatologies, night lights
(fixed composite), and the data/point layers (GBIF all-time, Climate TRACE
annual inventory, Argo latest positions, stations, glaciers single inventory).
Any NEW layer that ignores the date selector must be added to `datelessToast`;
date-driven rasters must return `null` there. Keep the toast copy consistent:
name the layer in `<strong>` and state "the date selector doesn't change it".

### 5. UI conventions

- Labels terse ("Grayscale globe", not a sentence). Explanations live in hover
  cards and hints, not in control labels.
- Layer metadata is uniform: title link, one-line `meta`, hover card.
- The date selector has quick-step buttons (±1d/±1m/±1y/Today) with real
  calendar arithmetic, clamped to [2000-01-01, most recent].
- Dark theme; diverging deltas are blue = decrease/cool, red = increase/warm.

### 6. Commits & deployment

- GitHub Pages serves the `gh-pages` branch; it always mirrors `main`
  (`git branch -f gh-pages main && git push origin main gh-pages -f`).
- Commit messages explain the *why* (data quirks, bug mechanics), not just the
  what. Multi-line bodies encouraged.
- Never commit credentials. The push token lives only in the local git
  credential helper.

### 7. Documentation set

| File | Role |
|---|---|
| `CLAUDE.md` | Standing instructions + holistic record (this file — keep current) |
| `README.md` | Quick start, repo layout, testing |
| `docs/PRIMER.pdf` | Background knowledge (GIBS, tiles, colormaps, product levels, climatologies). Rebuild: `python3 scripts/build_primer.py` |
| `docs/CATALOG.md` + `data/catalog.json` | The 244-record open-data catalog (human + machine readable) |
| `docs/COMBINING_DATASETS.md` | Which datasets measure the same quantity; sound combinations |
| `docs/SPECIES_AND_CLIMATE.md` | Why biodiversity data belongs in a climate app |

---

## Part 2 · Domain lore (hard-won facts — do not relearn)

- **GIBS tiling quirk.** The EPSG:4326 pyramid starts at 2×1 tiles (level 0),
  3×2 (level 1); resolution is 0.5625/2^L °/px, 512 px tiles. Edge tiles must
  declare their **full nominal span**, not the clamped visible part — clamping
  blanked the Pacific once. `GIBSGeographicTilingScheme` implements this; a
  test pins it to the published matrix definitions.
- **GIBS serves pictures, not numbers.** Values are recovered by inverting the
  layer's XML colormap (rgb → value LUT). Inversion recovers bin centres
  (quantised), works only for continuous one-to-one colormaps. Colormap
  entries come in two syntaxes: ranges `value="[lo,hi)"` and single values
  `value="N"` (sea ice, snow) — the parser handles both.
- **Precipitation cannot be per-pixel differenced.** IMERG is an instantaneous,
  log-scaled, mostly-transparent field; differencing two snapshots measures
  overpass luck. Rain climate questions are answered by the climatology grids
  (GPCP/E-OBS/MeteoSwiss) instead. `deltaRange` marks fully continuous fields
  (SST, SST anomalies, sea ice, snow, LST, salinity); `aggregable: true` marks
  average-but-don't-difference fields (chlorophyll, aerosol) — see the matrix
  in Part 1 §2.5.
- **Monthly composites lag.** A monthly GIBS layer (SMAP salinity) 404s for the
  current month; `gibsTime()` snaps monthly layers to first-of-month AND falls
  back to the previous month when the requested month is the current one.
  SMAP also has a real 2024 mission data gap — a blank year is data truth.
- **GBIF is all-time and date-independent.** The occurrence-density map ignores
  the app's date selector (it has no `year` filter wired). Sparse taxa render
  almost nothing at global zoom — Homo sapiens has only ~24 k records worldwide
  (privacy-restricted) and paints ~700 px vs birds' ~200 k. The picker note
  warns when a selection is below `GBIF_SPARSE` (150 k records) and always
  states the layer is all-time, so sparse ≠ broken and users don't blame the
  date.
- **RGI v7 is a single ~2000 snapshot** — a map slider cannot show glacier
  change. Real before/after comes from joining Hugonnet et al. 2021 per-glacier
  dh/dt (2000–2020, parquet keyed by `rgi_id`) — 240,542 of 274,531 glaciers
  matched; 78% thinning; median −0.26 m/yr; the Karakoram anomaly is visible.
- **MeteoSwiss grids ship 2D lon/lat arrays** in the NetCDF alongside the LV95
  metre grid — no projection library needed; scatter-bin to a regular grid.
- **E-OBS access**: the KNMI S3 bucket
  (`knmi-ecad-assets-prd.s3.amazonaws.com`) serves v31 NetCDF without a CDS
  account. The rr ensemble-mean file is ~365 MB; process in time-chunks.
- **OC-CCI and SMOS** have no clean unauthenticated endpoints; they are
  catalogued, and represented on-globe by NASA Ocean Color (chlorophyll) and
  SMAP (salinity) respectively. Wiring them as grids is an open follow-up.
- **Cesium's `_zoomFactor` is minified away** in production builds — wheel zoom
  is reimplemented as a custom handler (`__wheelZoom`).

## Part 3 · What has been built (holistic record)

**The globe.** CesiumJS 1.133 app (no build step) on GitHub Pages. Base
imagery Blue Marble; optional manual grayscale toggle (desaturates the base so
coloured overlays and blue-negative deltas stay readable).

**Raster layers (NASA GIBS WMTS, custom tiling scheme):** VIIRS true colour ·
MUR SST 1 km (default) · MUR25 SST anomalies · GPM IMERG V07 precipitation
(daily + 30-min) · AMSR2 sea ice · MODIS snow cover · MODIS aerosol optical
depth · MODIS land surface temperature · PACE chlorophyll-a · SMAP sea surface
salinity (monthly) · VIIRS Black Marble night lights.

**Climatology grid layers (client-rendered from baked JSON, `GridProvider`):**
GPCP v2.3 global precip (2.5°) · E-OBS v31 European precip (0.25°, bounded
rectangle) · OISST v2.1 SST 1991–2020 (1°) · MeteoSwiss Swiss precip normal
1991–2020 (~2 km). Ramp legends with hover read-out; probe reads exact cells.

**Analysis features:**
- *Comparison*: side-by-side split (draggable divider) or computed per-pixel
  difference vs 1/2/5/10/20 years ago, for continuous layers.
- *Aggregation*: rolling window 1–730 days for every layer in the aggregation
  matrix (SST & anomalies, sea ice, snow, LST, salinity, chlorophyll, aerosol),
  orthogonal to comparison. The mean
  is per pixel with missing samples excluded: each pixel divides by the number
  of sampled days on which it was actually observed (`sum[p]/cnt[p]`), so
  clear-sky products fill their cloud gaps; only never-observed pixels stay
  empty. Performance bounds: at most 12 sample dates per window
  (`windowSampleDates`) and zoom capped at level 4 while windowed.
- *SST ensemble*: MUR/OISST/GAMSSA client-side mean & spread.
- *Value probe*: dwell 650 ms or click; delta-aware (reports Δ, not absolute,
  when a difference layer is active); grid-aware (exact cell values).
- *Interactive legends* built from GIBS colormaps (hover → value).
- *Date stepper* ±1d/±1m/±1y/Today, calendar-correct, clamped.
- *Hover cards* on every layer: gist paragraph + Recorded / Interval / Spatial.

**Point/data layers:** Climate TRACE top-1000 emitters · Argo active floats ·
AMOC & GHG stations (RAPID, OSNAP, MOVE, SAMBA, Mauna Loa, Jungfraujoch…) ·
RGI v7 glaciers (274k; colour by extent or by Hugonnet 2000–2020 melt rate
with diverging legend) · GBIF biodiversity occurrences with a grouped picker:
broad taxonomic categories (8 kingdoms, major animal/plant classes, humans)
plus curated climate-indicator species. `data/species.json` carries live GBIF
counts (`scripts/refresh_data.py species`); the default note explains that the
~3.9 B "all recorded life" splits into eight kingdoms with ~14.5 M unplaced,
that birds dominate (~60%, a birdwatching bias), and that Homo sapiens is
present but privacy-restricted to ~tens of thousands of records.

**Dashboards (tabs):** *Temp* — GISTEMP v4 land vs land+ocean warming with
trends; *AMOC* — RAPID 26.5°N overturning transport series + stats;
*Sea level* — Frederikse 2020 budget components + NOAA altimetry; *Catalog* —
searchable 244-dataset catalog with domain/AMOC/globe filters.

**Data pipeline** (`scripts/refresh_data.py`): one function per snapshot —
climatetrace, argo, rapid, sealevel, glaciers (RGI7 tars + Hugonnet parquet
join), gistemp, gpcp, eobs, oisst, meteoswiss. Grid snapshots share
`_bin_to_grid`/`_write_grid` (nearest scatter-binning onto regular grids).

**Testing** (~45 Playwright specs): app behaviour (`tests/app.spec.js`) + data
integrity (`tests/data.spec.js`), sandbox MIRROR mode, in-repo proxies, CI on
real network.

**Notable bugs fixed along the way** (details in git history): Pacific blanked
by clamped edge tiles; Pages 404 (gh-pages + enablement); probe showing
absolute values under a delta; colormap parser skipping single-value entries;
salinity invisible (current-month composite unpublished); mangled Cesium class
names breaking test assertions; `_zoomFactor` no-op.

**Deferred / open follow-ups:** OC-CCI & SMOS as first-class grid layers;
multi-channel AMOC state vector; catalog `family` field for machine-readable
dataset relationships; honest precipitation aggregation (accumulated totals
from monthly products).

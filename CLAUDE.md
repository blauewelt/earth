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
   must declare one of these postures, and the choice must be justified in a
   code comment next to the flag:
   - `deltaRange: <n>` — continuous linear field: both time-averaging
     (Aggregate slider) and per-pixel differencing are sound.
   - `aggregable: true` + `ratioRange: <n>` — log-distributed field: averaging
     is sound, and computed comparison renders a ×-fold RATIO of window means
     (`RatioProvider`, log(mean_now/mean_past), saturating at ×n) instead of an
     absolute difference — which would be dominated by the log palette's
     value-proportional quantization. The ratio is quantization-robust: bin
     error is a few % OF the value, a small constant in log space.
   - `aggregable: true` alone — averaging is sound, no comparison of either
     kind (currently unused; every aggregable layer so far is also ratio-able).
   - neither — the layer is shown as-is (photographic composites,
     half-hourly snapshots).

   Current matrix (keep in sync when adding layers):

   | Layer | Aggregate | Difference | Why |
   |---|---|---|---|
   | SST (MUR), SST anomalies | ✓ | ✓ | continuous, gap-free L4 |
   | Sea ice (AMSR2) | ✓ | ✓ | continuous fraction · tiles end 2025-09 (endTime clamp) |
   | Snow cover (NDSI) | ✓ | ✓ | continuous %, clear-sky gaps fill by averaging |
   | Land surface temp (MODIS) | ✓ | ✓ | continuous K, clear-sky gaps fill by averaging |
   | Salinity (SMAP monthly) | ✓ | ✓ | continuous PSU; sample dates snap & dedupe to months |
   | Vegetation (MODIS NDVI monthly) | ✓ | ✓ | continuous 0–1 index; same-month-across-years differencing is THE standard use |
   | Water storage (GRACE monthly) | ✓ | ✓ | already an anomaly in cm; differencing months = storage change · tiles end 2022-07, 2017-18 mission gap |
   | SSH anomalies (5-day) | ✓ | ✓ | continuous m; differencing = local sea-level change · `snap5d` epochs, tiles end 2019-01 |
   | Energy balance (CERES monthly) | ✓ | ✓ | continuous W/m² · tiles end 2018-10 |
   | Soil moisture (AMSR2) | ✓ | ✗ | swathy like AOD — averaging fills orbits; day deltas compare coverage, not soil · tiles end 2025-09 |
   | Chlorophyll-a (PACE) | ✓ | ratio ×4 | log-normal-ish; absolute Δ of bin-centres is quantization noise, log-ratio of means is sound |
   | Aerosol optical depth | ✓ | ratio ×4 | windowed mean is standard; multiplicative change is the meaningful signal |
   | Precipitation (IMERG daily) | ✓ | ratio ×8 | daily-MEAN rates average soundly (`transparentZero`: dry pixels count as 0, eps = palette floor/2 keeps dry→rain finite); absolute day deltas are weather noise |
   | Precipitation (IMERG 30-min) | ✗ | ✗ | one half-hour snapshot; a window sampling ~12 arbitrary instants averages nothing physical — its role is intra-day (±30m stepper) |
   | True colour, night lights | ✗ | ✗ | photographs, no colormap to invert |
   | Grid climatologies | ✗ | ✗ | already multi-decade averages, not timed |
6. **Catalog consistency** — the dataset exists in `data/catalog.json`; set
   `globe: true` and append "Live globe layer in this app." to its notes.
7. **An active-layer chip.** Layers defined in `GIBS_LAYERS` get one for free.
   A hand-written layer (its own `#toggle-…` checkbox rather than a
   `GIBS_LAYERS` entry) must be added to `STATIC_LAYER_CHIPS` in `src/app.js`
   as `["toggle-<id>", "<short title>"]`, or it will be the one layer that
   can't be switched off from the globe.
8. **Tests** — at least one behavioural test in `tests/app.spec.js` and, if it
   has a data snapshot, a schema/sanity test in `tests/data.spec.js`.

### 3. Data pipeline: static snapshots, never live third-party calls

The browser must depend only on NASA GIBS (tiles) and GBIF (occurrence tiles).
**One deliberate exception**: the pixel inspector calls the Open-Meteo
family (`api`, `air-quality-api`, `flood-api`, `marine-api`,
`climate-api`.open-meteo.com) — key-free, CORS-open, and only ever a
single-point query triggered by an explicit click, never tile streaming. Any
further live endpoint must clear the same bar (no key, no quota pain,
click-triggered, degrades to an omitted card section on failure) and be added
to the MIRROR proxy set (`:8083`–`:8087` are the Open-Meteo hosts, in the
order above). Everything else is baked offline by
`scripts/refresh_data.py` into small static
JSON files under `data/` (one function per dataset, runnable individually:
`python3 scripts/refresh_data.py gpcp eobs`). Grids use the common format
written by `_write_grid()` (regular lon/lat, row-major from the south, `null`
for empty cells) and render client-side via `GridProvider`.

### 4. Testing in the sandbox

The dev sandbox's *browser* cannot reach external hosts (curl can). Therefore:

- `MIRROR=1` reroutes cdnjs → `_vendor/cesium`, GIBS → `localhost:8081`,
  GBIF → `localhost:8082`, the Open-Meteo hosts → `localhost:8083-8087` (see
  `tests/app.spec.js` beforeEach).
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
- **Never call `expect()` per data point.** Reduce to min/max (or a count) and
  assert twice instead: OISST alone is 64,800 cells, and per-cell matchers make
  the test take minutes because each assertion is recorded as a reporter step.
  Fixing this took the four grid tests from 27 s–8 min each to 23–39 ms.
- **Toggle heavy layers in-page, not via `check`/`uncheck`.** With 274k glacier
  billboards on a software GL stack the render loop starves Playwright's
  actionability checks, so a normal `uncheck()` can sit waiting for "stable"
  forever. Use `page.evaluate(() => { el.checked = false;
  el.dispatchEvent(new Event("change", { bubbles: true })); })` — the same
  pattern already used for dismissing toasts. Prefer a light layer
  (`#toggle-climatetrace`, 1,000 points) when the test doesn't care which one.
- Reading a self-clearing UI state (e.g. the `.flash` outline, 1.4 s) must
  happen inside the *same* `page.evaluate` as the click that sets it; a
  click→assert round-trip can outlast it on the slow sandbox.

### 4b. Date-independence must be announced

Enabling any layer with no per-date data fires an animated warning toast
(`showToast` / `datelessToast(id)`) so the date selector's lack of effect is
never a silent mystery. This applies to grid climatologies, night lights
(fixed composite), and the data/point layers (GBIF all-time, Climate TRACE
annual inventory, Argo latest positions, stations, glaciers single inventory).
Any NEW layer that ignores the date selector must be added to `datelessToast`;
date-driven rasters must return `null` there. **Yearly layers are NOT dateless**: Climate TRACE is an annual inventory baked for every available year (2021-2025, `assets_by_year` in climatetrace.json); the layer shows whichever year the date points at (`climateTraceYear`, clamped), rebuilds on a year change (`refreshYearlyLayers` / `ensureClimateTraceYear`), and its toast (`climateTraceToast`) says 'the day and month don't matter, but the year does' — never declare a layer fully dateless if any date component drives it. **Monthly grids follow the same pattern one level down**: GLORYS currents/MLD are `monthlyGrid` layers covering the FULL archive (1993-01 → ~now−2mo). The baked index (data/currents.json, data/mld.json) carries `monthsAvailable` (every stamp), `yearDir`, `months` (latest year inlined), `latest`, `values` (= latest month, back-compat); older months live in per-year files (data/currents_y/YYYY.json) lazy-fetched by `ensureGridMonth`/`loadGridMonth` and merged into `g.months` — every sampler (GridProvider.requestImage, probeValueAt, the pixel card) MUST go through `loadGridMonth`, never bare `loadGrid`+`sampleGrid`, or old months read as null. `resolveGridMonth` floors the date's month to the newest baked month ≤ it (clamped at both ends), `refreshMonthlyGrids()` rebuilds the provider when a date change lands on a different baked month (Cesium caches tiles — a repaint needs a fresh provider; called from both date handlers AND the ±30m midnight-cross branch), and `maybeMonthlyGridToast` names the month showing on enable. Note the date steppers clamp at 2000-01-01 (GIBS floor) — 1993–1999 currents are reachable by typing a date. **Day-keyed forecast grids reuse the whole mechanism**: GFS temp/precip are `monthlyGrid` + `forecastGrid` layers whose JSON carries `keyLen: 10` (day stamps in `months`/`monthsAvailable`, all frames inline, no year files) plus `init` (the model run, quoted in the toast). While a forecast layer is active, `uiMaxDate()` returns the last forecast day instead of `defaultDate()` — the date input's `max` and every stepper clamp go through it (`syncDateMax()` restores reality and pulls the date back when the last forecast layer is switched off), and `gibsTime` clamps any future date to `defaultDate()` so observation layers are never asked for tomorrow's tiles. Keep the toast copy consistent:
name the layer in `<strong>` and state "the date selector doesn't change it".

### 5. UI conventions

- Labels terse ("Base globe", not a sentence). Explanations live in hover
  cards and hints, not in control labels.
- Every active layer has a labeled opacity row (`.alpha-row`: slider + live %
  readout + a `½` button toggling 50%↔100%). The ½ button exists for field
  correlation by overlay — e.g. SST at 50% over ocean currents to see whether
  the warm tongue follows the Gulf Stream. Alpha lives in
  `state.layers[id].alpha` and survives delta/compare/window re-adds.
- Layer metadata is uniform: title link, one-line `meta`, hover card.
- The date selector has quick-step buttons (±1d/±1m/±1y/Today) with real
  calendar arithmetic, clamped to [2000-01-01, most recent]. A ±30m time-of-day
  stepper (`#time-steps`) appears only while a sub-daily layer is on; crossing
  midnight rolls the date, and stepping refreshes only the sub-daily layers
  (no churn of the daily/monthly rasters).
- Dark theme; diverging deltas are blue = decrease/cool, red = increase/warm.
- The header tagline's words are one-click SCENES (`.tag-link`,
  `SCENES` map in app.js) with two hard rules learned from feedback: ONE
  visual field per scene — single layers, except pairs with spatially
  DISJOINT coverage that compose one field (temperature = SST ocean + LST
  land; a test pins the exact exception) — and the link text names exactly
  what appears ("sea ice",
  not "ice" that also drops a one-off glacier inventory; the inspector link
  is "inspect any point", not "forecasts to 2050"). Scenes REPLACE the
  current layers — the chips show the swap and undo it. Current set:
  satellites · surface temperature · sea ice (Arctic flyTo — polar data is
  invisible from the default view; SCENE_VIEWS) · ocean currents · floats ·
  vegetation · emissions · inspect any point.
- The Layers tab opens with a first-visit intro guide (`#intro-guide`,
  <details> open by default, dismissal persisted in localStorage) that
  documents the whole view: date/time stepping, Compare's two modes,
  Aggregate, hover cards/legends/probe, chips, and the pixel inspector, plus
  a one-liner per tab. Keep it in sync when controls change — it is the
  entry-point documentation. The Compare/Aggregate fine-print explainer is
  `#how-compare`. Header: h1 "earth" + `.h1-sub` + a substantive tagline.
- The sidebar is resizable: width lives in the `--sidebar-w` CSS variable
  (default 380 px, clamp 300–(window−240 px) — max is structural, only enough
  globe to click, per user request), dragged via `#sidebar-resize`,
  persisted in localStorage, double-click resets. Anything sized off the
  panel (dashboard charts) must redraw on drag; the split divider reposition
  runs on drag end. Hidden in the stacked ≤720 px layout.
- The Aggregate slider has one-click presets (1d/7d/30d/365d, `#window-presets`)
  that drive the slider and fire its `change` event, so they follow the exact
  same path as dragging it. The long Compare/Aggregate explainer folds into a
  `<details class="hint-details">` — collapsed by default; keep its text in
  sync with the posture matrix when comparison/aggregation semantics change.
- **Pixel inspector** — "Everything we know (pixel state)", a layer-list
  entry (`#toggle-pixel`, off by default, chip-registered): clicking bare
  globe (no entity under the cursor) opens `#pixel-card` —
  docs/PIXEL_STATE.md made clickable. Click semantics
  (`pixelInspectorEngaged()`): entry checked → card, explicit intent;
  unchecked + a colormapped layer active → the click reads that layer's value
  (probe tooltip, the specific question being asked); unchecked + NO
  colormapped layer active → card again, since there is no layer value to
  read instead. The card composes: live weather + 7-day forecast, CAMS air
  quality, GloFAS river discharge, waves, and a 2045–49-vs-1991–95 CMIP6
  outlook (all Open-Meteo family); all fourteen colormapped GIBS rasters
  probed at the current date (z capped at 4); the four climatology grids; and
  nearby context (stations, Argo, emitters; glaciers only if already loaded —
  never pay the 7 MB on a click). Deliberately NO derived "SST vs normal" delta: the baked OISST
  normal is the annual mean, so the difference would mostly be the seasonal
  cycle — the MUR25 anomalies row is the seasonally-correct departure.
  `showPixelState(carto)` is exported for tests. Esc or × closes.
- **Active-layer chips** (`#active-layers`, top-left of the globe) list every
  layer currently on, whatever machinery draws it. Each chip's `×` turns the
  layer off; the label jumps to that layer's sidebar row and outlines it
  briefly; past one layer a dashed "Clear all N" chip appears. Two reasons this
  lives on the globe rather than at the top of the layer list: a layer can be
  switched off without hunting a long list, and it works from the Temp / AMOC /
  Catalog tabs, where the layer list isn't rendered at all. A layer that is
  checked but not drawn (aggregation-suppressed) shows as a ⚠ `chip-warn`
  rather than being listed as if it were visible.
  - Chips **drive the layer's own checkbox** (`checked = false` + a bubbling
    `change` event) instead of duplicating teardown, so opacity sliders, toasts
    and legends stay in sync for free. `updateActiveChips()` is wired to any
    `change` in the document, since some controls (species picker, glacier
    mode) tick a box without firing that box's own handler.
  - The chip bar publishes its height as `--chips-h` on `#cesiumContainer` so
    `#split-labels` steps below it instead of underneath.

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
| `README.md` | Quick start, repo layout, testing. Opens with a link to the live demo. Keep its counts (catalog size, `globe`/`amoc` flags, spec count) and feature list current — they drift silently. Hero image: `node scripts/screenshot.js` (see the header comment for the sandbox invocation); re-shoot it when the UI changes visibly |
| `docs/PRIMER.pdf` | Background knowledge (GIBS, tiles, colormaps, product levels, climatologies). Rebuild: `python3 scripts/build_primer.py` |
| `docs/CATALOG.md` + `data/catalog.json` | The 245-record open-data catalog (human + machine readable) |
| `docs/COMBINING_DATASETS.md` | Which datasets measure the same quantity; sound combinations |
| `docs/PIXEL_STATE.md` | Which catalog sources compose into a per-pixel state vector (state/memory/forcing/flow/future); the 0.25°-daily common grid argument; the ~25-source minimal composition |
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
- **What a transparent pixel MEANS differs per layer**, and window means must
  honour it. Clear-sky products (MODIS LST/snow/AOD): transparent = unobserved
  → exclude the sample. Precipitation: transparent = below the palette floor
  (0.1 mm/hr) ≈ no rain → must count as 0 (`transparentZero` flag), or the
  "mean" is really "rate on rainy days", biased high wherever it rained once.
  Symmetrically on output: a mean below the palette floor renders transparent,
  else `forward()` clamps drizzle-of-drizzles up to the first colour and the
  whole ocean tints "light rain".
- **Some GIBS archives end before today**: GRACE mascons stop at 2022-07,
  CERES EBAF at 2018-10, MEaSUREs SSH anomalies at 2019-01, AMSR2 soil
  moisture AND sea ice at 2025-09 — the instruments/records continue, only the *tiles*
  stop. `endTime` in the layer cfg clamps requests to the last served date
  (so the layer shows its final state instead of blanking), and the hover
  card must say "this map: … → <end> (last date GIBS serves)", and
  `maybeClampToast` fires on enable when the date sits past the end. When a
  new layer lands, CHECK ITS EXTENT in the GetCapabilities — and scene tests
  must assert rendered DATA (tile pixels for the effective date), not just
  that a chip appeared: "sea ice" shipped blank because the test stopped at
  the chip. 5-day products
  (`snap5d: [epoch1, epoch2]`) serve only exact epoch dates — floor to the
  nearest valid epoch; MEaSUREs SSH was re-anchored in 2017, hence two epochs.
- **GIBS serves sub-daily TIME**: `TIME=YYYY-MM-DDTHH:MM:SSZ` returns distinct
  tiles per half-hour for IMERG 30-min (verified: 13:00 ≠ 13:30 ≠ bare date;
  bare date resolves to 00:00). `gibsTime()` appends the timestamp for
  `subDaily` layers from `state.timeMin`.
- **Cesium's `_zoomFactor` is minified away** in production builds — wheel zoom
  is reimplemented as a custom handler (`__wheelZoom`).
- **Zoom direction convention**: scrolling up, or spreading two fingers apart
  on a trackpad (negative `deltaY`), zooms IN; scrolling down, or pinching
  fingers together (positive `deltaY`), zooms OUT — matching standard
  map/trackpad expectations. Touch-screen pinch stays native Cesium
  (`CameraEventType.PINCH`) and already followed this convention; only the
  wheel/trackpad-pinch handler needed inverting (it had shipped backwards).

## Part 3 · What has been built (holistic record)

**The globe.** CesiumJS 1.133 app (no build step) on GitHub Pages. Base
imagery Blue Marble; the base AUTO-desaturates whenever a colormapped layer
is active (blue-on-blue SST and green-on-green NDVI both fail otherwise) and
returns to colour for bare/photographic views — `#base-mode` select with
always-colour/always-grey overrides, persisted (`updateBaseAppearance`,
`colormappedLayerActive`).

**Raster layers (NASA GIBS WMTS, custom tiling scheme):** VIIRS true colour ·
MUR SST 1 km (default) · MUR25 SST anomalies · JPL MEaSUREs SSH anomalies
(5-day, tiles→2019) · GPM IMERG V07 precipitation (daily + 30-min) · AMSR2 sea
ice · MODIS snow cover · MODIS aerosol optical depth · MODIS land surface
temperature · AMSR2 soil moisture (tiles→2025-09) · MODIS NDVI vegetation
(monthly) · GRACE water-storage anomaly (monthly, tiles→2022-07) · CERES EBAF
TOA net flux (monthly, tiles→2018-10) · PACE chlorophyll-a · SMAP sea surface
salinity (monthly) · VIIRS Black Marble night lights.

**Climatology grid layers (client-rendered from baked JSON, `GridProvider`):**
GPCP v2.3 global precip (2.5°) · E-OBS v31 European precip (0.25°, bounded
rectangle) · OISST v2.1 SST 1991–2020 (1°) · MeteoSwiss Swiss precip normal
1991–2020 (~2 km) · **Subsurface temperature anomaly at 300 m** (Argo RG, 1°,
`snapshotGrid` — a single recent month vs the 2004–18 same-month mean,
diverging `anom` ramp). Ramp legends with hover read-out; probe reads exact
cells.

**Ocean column (Argo RG, `data/ocean_column.json`):** the latest month's
absolute T/S profile AND the same-calendar-month 2004–18 normal on a 2° grid
at 17 depth levels (0–2000 dbar), both from the Roemmich-Gilson product so
the anomaly shares one baseline. Bake: `refresh_data.py argo_column`
(downloads ~1.2 GB of RG NetCDF; mask handling is critical — `np.ma.filled`,
never `np.array`, else land renders as zero-anomaly ocean). The pixel card
draws the profile (sqrt-depth axis, now vs normal) with upper-700 m stored
heat, warm-layer depth and surface-salinity-freshening lines.
`refresh_data.py glorys` (needs a free Copernicus login, via env vars or
`copernicusmarine login`) bakes the FULL monthly archive 1993→now−2mo,
month-keyed (see §4b `monthlyGrid`) — credentials never in the repo. Two
phases, same model: 1993→2024-12 from the GLORYS member (`*_glor`) of the
1/4° ensemble reanalysis GLOBAL_MULTIYEAR_PHY_ENS_001_031 fetched one YEAR
per request (16× less download than 1/12°, and we bin to 1° anyway), then
1/12° GLORYS12 my/myint per month for the tail (also rebuilds
ocean_surface.json). Resume-friendly: years already complete in
data/currents_y// data/mld_y/ are skipped, per-request NetCDFs deleted after
baking. Credentials were deleted after each use, per the user.
`refresh_data.py gfs` (NO account — NOMADS grib filter + pygrib) bakes the
10-day GFS forecast: newest COMPLETE cycle (probes for f240), 2 m temperature
one frame per day, precipitation as 24-h sums of the 6-h APCP buckets grouped
by UTC day (full days only; <1 mm/day nulls to transparent — beware Python's
banker's rounding when thresholding). Forecasts age daily — re-run `gfs`
whenever refreshing data. WeatherNext 2 (DeepMind) was evaluated 2026-07-29:
every access path needs a Google account (Earth Engine / BigQuery Analytics
Hub / GCS request form; anonymous reads 401/403), so GFS is the key-free
baseline; the day-keyed machinery is ready if the user brings GCP credentials.

**Analysis features:**
- *Comparison*: side-by-side split (draggable divider) or computed per-pixel
  change vs 1/2/5/10/20 years ago — an absolute difference (`DeltaProvider`)
  for continuous linear layers, a ×-fold ratio of window means
  (`RatioProvider`) for log-distributed ones (precip, chlorophyll, aerosol),
  each with its own diverging legend and probe read-out (±units vs ×fold).
- *Aggregation*: rolling window 1–730 days for every layer in the aggregation
  matrix (SST & anomalies, sea ice, snow, LST, salinity, chlorophyll, aerosol,
  daily precipitation — dry pixels counting as zero),
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
- *Date stepper* ±1d/±1m/±1y/Today, calendar-correct, clamped; plus a ±30m
  time-of-day stepper while a sub-daily layer (IMERG 30-min) is on.
- *Hover cards* on every layer: gist paragraph + Recorded / Interval / Spatial.
- *Active-layer chips* top-left of the globe: what's on right now, one click to
  switch any of it off (or "Clear all N"), from any tab. See §5.
- *Pixel inspector*: click any point on the globe → one card composing live
  weather + 7-day forecast, all satellite fields at the current date, climate
  normals with anomaly, and nearby observing/emitting context. See §5.

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
trends; *Energy* — Earth's energy imbalance (NOAA NCEI OHC 0–700 m from
1955 / 0–2000 m from 2005; centred 5-yr OLS slopes × 0.6213 → W/m² of the
whole Earth; last-decade rate ÷ 0.9 ≈ total EEI; `refresh_data.py eei`) with
TWO charts: accumulated heat (the ledger, ×10²² J) and its slope (the
imbalance itself, W/m² over time) — axis semantics spelled out on-panel
after they confused a reader;
trends; *AMOC* — RAPID 26.5°N overturning transport series + stats;
*Sea level* — Frederikse 2020 budget components + NOAA altimetry; *Catalog* —
searchable 245-dataset catalog with domain/AMOC/globe filters.

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

# earth 🌍

### ▶ **[Open the live globe →](https://blauewelt.github.io/earth/)**

**Open climate data on a 3D globe — and, eventually, predictions from it.**

`earth` is a prototype for exploring the world's open climate data on an interactive CesiumJS globe, backed by a curated, machine-readable catalog of **248 open climate datasets** across atmosphere, ocean & AMOC, cryosphere, satellite platforms, model projections, greenhouse gases, and impacts.

The long-term goal: layer enough observational and model data onto the globe to drive real prediction pipelines — starting with the question *will the AMOC collapse, and when?*

[![screenshot](docs/screenshot.png)](https://blauewelt.github.io/earth/)

## Features (prototype)

- **Zero API keys, no build step.** All imagery streams from [NASA GIBS](https://www.earthdata.nasa.gov/engage/open-data-services-software/earthdata-developer-portal/gibs-api) WMTS and [GBIF](https://www.gbif.org/) — no registration, no tokens. Everything else is baked into small static JSON snapshots, so the page has no server of its own. Having no build step does mean nothing renames files on deploy, so each local asset carries the first eight hex of its own sha256 in its URL (`scripts/stamp_assets.py`, checked by the suite) — otherwise a browser holding yesterday's `app.js` never learns there is a new one, and the About tab prints the same hash so you can see which build you are running.
- **Installable.** A web manifest and three icons make it an app on a phone home screen — no service worker, deliberately, since the content is remote tiles and a worker's cache would only add a second place for stale code to hide. The icon is not artwork: `scripts/make_icons.py` renders NASA's Blue Marble — the app's own base globe — with the ocean deepened toward the blauewelt blue, in an orthographic view centred on Europe and Africa, from a GIBS snapshot in `data/icon/`. The icon is a picture of the planet, in the brand's colour.
- **Time-dynamic raster layers** with a date picker and per-layer opacity: true colour (VIIRS), sea surface temperature (MUR 1 km), SST anomalies (MUR25), sea-surface-height anomalies (JPL MEaSUREs altimetry), precipitation (GPM IMERG V07, daily — plus the native half-hourly product with a ±30m time stepper to watch storms cross a single day), sea ice concentration (AMSR2), snow cover (MODIS NDSI), aerosol optical depth (MODIS), land surface temperature (MODIS), soil moisture (AMSR2), vegetation index (MODIS NDVI, monthly), water-storage anomaly (GRACE mascons, monthly), Earth's energy balance (CERES net flux, monthly), chlorophyll-*a* (PACE), sea surface salinity (SMAP, monthly), night lights (Black Marble), **30 m vegetation-disturbance alerts** (OPERA DIST-ALERT/DIST-ANN), and the fine tier below. Every timed layer asks GIBS for its own published time domain the first time you enable it, then snaps the requested date to the newest one actually served — which covers both archives that ended (GRACE, CERES, SSH anomalies, AMSR2 soil moisture and sea ice) and live ones that simply lag (the monthly NDVI composite runs about two months behind). Gaps inside an archive are handled the same way. Whichever it is, a toast says which date you are looking at and why, and the hover cards state the last date GIBS serves.
- **Forest loss at 30 m** — NASA JPL's [OPERA](https://www.jpl.nasa.gov/go/opera/products/dist-product-suite/) land-surface disturbance products, built from Harmonized Landsat-Sentinel imagery and the finest layers in the app. Every new clear image is compared against that pixel's own recent history, and a drop in vegetation cover is flagged: first detection, provisional, then *confirmed* once a second image agrees — and separately whether under or over half the cover went. `DIST-ALERT` refreshes every few days through the current year; `DIST-ANN` settles it into one map per year (2023–2025). Deforestation, fire scars, logging roads and storm damage all appear — the product sees the loss, not the cause. These are **classification** rasters rather than continuous fields, so they get a swatch legend instead of a gradient bar, the probe answers with a class name instead of a number, and neither averaging nor differencing is offered (class codes do neither). The *forest loss* tagline scene flies to the Amazon arc of deforestation, because from orbit a 30 m clearing is far smaller than a screen pixel.
- **Why forests fall** — the companion question, answered by [WRI and Google DeepMind's global driver map](https://datasets.wri.org/datasets/dominant-drivers-of-tree-cover-loss-at-1km): a classifier trained on tens of thousands of hand-labelled sites names the dominant cause of tree-cover loss at every kilometre on Earth over 2001–2025 — permanent agriculture, hard commodities, shifting cultivation, logging, wildfire, settlements and infrastructure, or other natural disturbance. The pattern is the point: the Amazon arc and West Africa are agriculture, boreal Canada and Siberia are fire, the US southeast and Scandinavia are logging — and the difference between them is the difference between forest that is gone and forest that will grow back. Binned to 0.25° by *dominant class*, never an average, and painted in WRI's own palette so the globe matches every published figure.
- **Reference points, so you can tell where you are looking** — a globe of pure data is beautiful and unnavigable: a warm anomaly off a coastline you can't name says nothing about *where* the ocean is warm. City and town names ride over every layer, thinning out with altitude the way a paper map does — a dozen world cities from orbit, capitals across a continent, every town in the valley up close. The ladder is Natural Earth's own cartography (its `min_zoom` becomes each label's visibility range), not a threshold invented here. National capitals are weighted, coastlines and borders can be switched on alongside ([NASA GIBS](https://worldview.earthdata.nasa.gov/) reference features), and the whole thing turns off for a data-only view. Clicking a name still reads the globe underneath it.
- **Search for a place, and be flown to it** — Natural Earth is a cartographic *selection*, not a gazetteer: it names twenty-four places in all of Portugal, which is the right list for a legible map and the wrong list for answering "where is Peniche". A second, lazily-loaded file ([GeoNames `cities5000`](https://www.geonames.org/export/), CC BY 4.0, 54,204 places after deduplication against Natural Earth) does the other job. The search box finds any of the 61,000 places across both files — diacritic-insensitive, so an English keyboard finds Zürich — and flies you in at an altitude where the town is actually labelled. The same file densifies the labels themselves once you descend past the rung where Natural Earth runs out, on rungs whose spacing is *measured* from Natural Earth's own cumulative counts rather than picked by hand.
- **Islands are named too** — and they need their own tier, because every gazetteer in existence lists *populated places*: Westerland, population 9,000, was labelled while the 43-km island of Sylt it stands on was an anonymous shape. 4,950 islands, named from [Natural Earth](https://www.naturalearthdata.com)'s curated island polygons where they exist and [GeoNames](https://www.geonames.org/export/) physical features elsewhere, drawn in italics with no dot (a dot claims a point; an island is an area). Their visibility rule is geometry rather than a rung: **an island's name appears once the island is at least as wide on screen as the name is** — measured from the actual text width and the camera's own field of view, so it follows the ground instead of a population, and can never crowd the view (that would take islands wider than the view). The rule checks out against the cartographers: bucketing all 9,632 Natural Earth coastline rings by the zoom level they are drawn at, the median island halves in width with every rung, to within 3%. Continents are excluded by one measured cut at 3 million km², which is exactly the line that keeps Greenland and drops Australia.
- **The fine tier — 30 m and finer, still zero keys.** GIBS serves the fine-resolution record in the same tile scheme as everything else, so the app now carries **Sentinel-2** and **Landsat 8/9 true colour** (Harmonized Landsat–Sentinel, 30 m, daily), **Sentinel-1 C-band radar backscatter** (OPERA RTC-S1, 30 m — sees through cloud and night), **NISAR L-band radar** (15 m, the finest layer, provisional since its 2025 launch), **surface water extent** from optical HLS and from Sentinel-1 radar (OPERA DSWx, the flood layers), **elevation** (ASTER GDEM v3, 30 m — hover reads metres, and it is a row in the pixel card), 2010 Landsat **built-up extent** and **impervious-surface %**, and **historic Landsat mosaics** of 1984–86, 1989–91 and 1999–2001 (WELD). The daily ones are satellite *swaths* — on any one date only the strips flown are painted — so they are **gated on camera height**: above 500 km (300 km for NISAR) the layer stays enabled but hidden and requests no tiles at all; the row says "zoom in — hidden above 500 km (you're at 12,000 km)", and the tiles for the area in view arrive the moment you descend. From orbit each swath layer paints the day's passes as coarse strips, so you can see where the satellite flew before zooming in for the 30 m detail — and the **Aggregate** slider turns into a *lookback* for them: set it to 12 days and the layer shows the **union** of every pass in that window, newest on top, which for NISAR is the whole planet (a mean would be meaningless — you cannot average "flew over" with "didn't"). Date scrubbing and playback cost nothing above the gate. A **third backend** — four keyless tile hosts beyond NASA GIBS, each verified by request and credited as its licence asks — adds **ESA WorldCover** 10 m land cover (eleven classes, live Terrascope tiles), **JRC Global Surface Water** (how often each 30 m pixel was water across 1984–2024, hover reads the %), **EOX Sentinel-2 cloudless** (one 10 m mosaic per year 2016–2025 — split 2016 against 2025), and **swisstopo**: SWISSIMAGE at 10 cm, the 1926–2025 aerial time-travel series by year, and the swissALTI3D hillshade, Switzerland only.
- **Climatology grid layers** rendered client-side from baked JSON: GPCP v2.3 global precipitation (2.5°), OISST v2.1 SST 1991–2020 (1°), E-OBS v31 European precipitation (0.25°), and the MeteoSwiss 1991–2020 Swiss precipitation normal (~2 km).
- **The ocean beneath the surface** — from the Argo float fleet (Roemmich–Gilson climatology): a **subsurface temperature anomaly layer at 300 m depth** (latest month vs the 2004–18 mean for that same month — subsurface marine heatwaves that no satellite surface map can see), and, on the pixel inspector, the full **0–2000 m temperature/salinity profile** at any clicked ocean point, drawn against its seasonal normal with upper-700 m stored-heat and freshening read-outs.
- **Comparison** — any dated layer vs 1/2/5/10/20 years ago: side-by-side swipe with a draggable divider, or a **computed per-pixel change** (the GIBS colormap is inverted client-side back to physical units, then re-ramped diverging). Continuous linear fields (SST, sea ice, snow, land temperature, salinity) render an absolute difference; log-distributed fields (precipitation, chlorophyll, aerosol) render a **×-fold ratio of window means** — the statistically sound comparison for such fields, and robust to the log palette's value-proportional quantization. Layers where neither is possible say so instead of pretending.
- **Rolling aggregation** — an *Aggregate* slider (1–730 days) averages over the past N days ending on the chosen date, for every continuous layer, and orthogonally to the comparison mode. The mean is per pixel and layer-aware: for clear-sky products, missing samples are excluded so gaps fill instead of darkening; for precipitation, dry (transparent) pixels count as *zero*, so multi-day rain is a true mean rate rather than "rate when raining". "Past 365 days vs 10 years ago" reveals broad ocean warming and the subpolar cold blob cleanly, without daily weather noise.
- **SST ensemble** — combines independent GHRSST L4 analyses (MUR, OISST, GAMSSA) client-side into a **mean** or a **spread** map; the spread highlights where the analyses disagree (fronts, eddies, under-observed ocean).
- **Every layer explains itself** — the title is a link to the dataset's own documentation, and a hover card gives a plain-language gist plus what period is *recorded*, at what *interval*, and at what *spatial* granularity. Colormapped layers carry an interactive legend, and hovering the globe probes the actual value under the cursor.
- **Active-layer chips** on the globe list what is currently switched on, so any layer can be turned off in one click from any tab — including the dashboards, where the layer list isn't on screen at all.
- **"Everything we know" (pixel inspector)** — a layer that answers clicks instead of drawing pixels: click anywhere on the globe for one card composing everything the app can know about that point — live weather and a 7-day forecast, CAMS air quality, river discharge, waves and a per-pixel 2050 climate outlook (Open-Meteo family), all fifteen satellite fields probed at the chosen date, why any forest at that point was lost, climate normals, and nearby context (monitoring sites, Argo floats, major emitters, glaciers). Every single value says *when* it was observed — down to the half-hour where the data has it, a day, a month, a year, or a fixed span like 1991–2020 — with its age beside it where an age means anything, because these rows are routinely years apart even when they sit side by side. With it off, a click reads the value of the layer you're viewing; with no layer active, the card opens anyway. It is [docs/PIXEL_STATE.md](docs/PIXEL_STATE.md) made clickable.
- **Point and inventory layers** — Climate TRACE's top 1,000 facility emitters · the ~3,800-float active Argo fleet · the AMOC monitoring network (RAPID, OSNAP, MOVE, SAMBA, the Florida Current cable, the subpolar "cold blob" region) and reference GHG stations (Mauna Loa, Jungfraujoch, …) as clickable markers with data links · all 274,531 glaciers of the Randolph Glacier Inventory v7, colourable by extent or by their 2000–2020 melt rate from Hugonnet et al. 2021 (240,542 matched; ~78% thinning, and the Karakoram anomaly is visible).
- **Biodiversity layer** — GBIF occurrence-density tiles (3.9 B records, key-free) with a grouped picker: broad taxonomic categories (kingdoms, major animal and plant classes, humans) plus curated climate-indicator species (Atlantic mackerel, emperor penguin, staghorn coral …) whose shifting ranges are a visible fingerprint of warming. See [docs/SPECIES_AND_CLIMATE.md](docs/SPECIES_AND_CLIMATE.md).
- **Dashboards** — *Temp*: GISTEMP v4 land vs land+ocean warming, 1880–2025, with trends. *Energy*: **Earth's energy imbalance** — the NOAA ocean-heat-content record as both the accumulating-heat ledger and its slope, the imbalance itself over time in W/m² (currently ~+0.7 ocean / ~+0.8 total, +224 ZJ stored since 2005). *AMOC*: the RAPID 26.5°N overturning transport record (2004–2024) with stat tiles and a hoverable chart. *Sea level*: observed global mean sea level 1900–2018 decomposed into its causes (thermal expansion, glaciers, Greenland, Antarctica, land water), with the summed budget tracking the observed line to show *closure*, plus modern satellite altimetry (Frederikse et al. 2020 + NOAA).
- **Dataset catalog browser** — search and filter all 248 cataloged datasets by domain, AMOC relevance, and globe-readiness, straight from [`data/catalog.json`](data/catalog.json).
- **Honest about time.** Layers that ignore the date selector (climatologies, night lights, the point and inventory layers) announce it with a warning toast when switched on, rather than leaving the date picker silently inert.
- **Navigation** — scroll wheel, touch pinch, trackpad pinch (ctrl+wheel) and on-globe buttons; zoom is distance-proportional, and follows the standard convention where spreading fingers apart or scrolling up zooms in. The base globe auto-greys whenever a colormapped data layer is on (so data colours never fight the map's own blues and greens) and returns to colour otherwise — with always-colour/always-grey overrides.

## Run it

No build step. Serve the directory with any static server:

```bash
git clone https://github.com/blauewelt/earth
cd earth
python3 -m http.server 8080
# open http://localhost:8080
```

Or just use the [live deployment](https://blauewelt.github.io/earth/), which GitHub Pages builds from `main` via the included workflow.

## Testing

The repo ships a Playwright regression suite (137 specs): data-snapshot integrity
(`tests/data.spec.js` — catalog, RAPID series, Argo fleet, Climate TRACE, stations,
sea-level budget, GISTEMP, glaciers, species, the four climatology grids, the place
gazetteer, the island file) and full browser tests (`tests/app.spec.js` — GIBS tiling-scheme math
including the Pacific partial-tile regression, layer and date handling, comparison
split and computed-delta mode, aggregation, hover cards, legends, the value probe,
active-layer chips, zoom and pinch gestures, point layers, place and island names
and search,
dashboards, catalog browser).

```bash
npm ci
npx playwright install --with-deps chromium
npx playwright test
```

CI runs the full suite on every push and PR. Note that **Pages deploys in parallel
rather than behind the tests** — this is a prototype, so a red run signals a
fix-forward, not a blocked release.

In CI the suite reaches the real CDN, NASA GIBS and GBIF. In a network-restricted
sandbox, `MIRROR=1` reroutes cdnjs to the vendored Cesium under `_vendor/cesium` and
GIBS/GBIF to local forwarding proxies; `scripts/run_tests.sh` starts everything and
runs the suite in one shot.

## Repository layout

```
CLAUDE.md               standing instructions + holistic project documentation
index.html              the app shell
src/app.js              CesiumJS globe, GIBS layers, grids, points, dashboards, catalog UI
src/style.css           dark UI theme
data/catalog.json       248-dataset open climate data catalog (machine-readable)
data/stations.geojson   AMOC arrays + GHG reference stations
data/rapid_moc.json     RAPID 26.5N AMOC transport series
data/sealevel.json      sea-level budget (Frederikse 2020) + NOAA altimetry
data/gistemp.json       GISTEMP v4 land vs land+ocean anomalies
data/glaciers.json      RGI v7 glaciers + Hugonnet 2000-2020 melt rates
data/species.json       GBIF taxon keys and record counts for the biodiversity picker
data/climatetrace.json  top-1000 facility emitters · data/argo.json  active float fleet
data/{gpcp,oisst,eobs,meteoswiss}.json   gridded climatologies (shared grid format)
data/cities.json        Natural Earth place names (the map's reference points)
data/gazetteer.json     GeoNames cities5000 below them — search + deep-zoom labels
data/islands.json       4,950 island names — the tier that names ground, not people
scripts/refresh_data.py regenerates every snapshot above (one function per dataset)
scripts/build_primer.py rebuilds docs/PRIMER.pdf (background-knowledge primer)
scripts/run_tests.sh    sandbox test runner · scripts/test_proxy.py  GIBS/GBIF proxies
scripts/screenshot.js   regenerates docs/screenshot.png (the image above)
scripts/stamp_assets.py rewrites every ?v= asset hash — run before committing src/
scripts/make_icons.py   redraws the app icons from data/icon/*.png
manifest.json           web app manifest · icon-*.png  the generated icons
tests/                  Playwright suite (app behaviour + data integrity)
docs/CATALOG.md         the catalog as a readable reference document
.github/workflows/      test + GitHub Pages deployment
```

Conventions and the full record of what has been built live in
[CLAUDE.md](CLAUDE.md) — read it before contributing.

## Documentation

Reading on a phone? Every document below is also served through
**[the phone reader](https://blauewelt.github.io/earth/docs.html)** — same
files, read live from `main`, but wide result tables keep their row label
pinned while the numbers scroll, two-column prose tables become labelled
blocks, and long documents get a contents drawer.

| Document | What it covers |
|---|---|
| [docs/PRIMER.pdf](docs/PRIMER.pdf) | Background knowledge: GIBS and WMTS, tiling schemes, colormaps, satellite product levels, what a climatology is |
| [docs/COMBINING_DATASETS.md](docs/COMBINING_DATASETS.md) | Which catalog datasets measure the same quantity, which combinations are scientifically sound (SST ensembles, the sea-level budget, the AMOC state vector, land+ocean blends), and why per-pixel differencing works for SST but not precipitation |
| [docs/PIXEL_STATE.md](docs/PIXEL_STATE.md) | Which of the 248 sources compose into a holistic per-pixel state vector — state, memory, forcing, flow, future — and the ~25-source minimal composition on a common 0.25° daily grid |
| [docs/SPECIES_AND_CLIMATE.md](docs/SPECIES_AND_CLIMATE.md) | Why biodiversity occurrence data belongs in a climate app |
| [docs/CATALOG.md](docs/CATALOG.md) | The full catalog as a readable reference |

## The data catalog

The catalog ([readable](docs/CATALOG.md) · [JSON](data/catalog.json)) records for each dataset: provider, canonical URL, access method (API endpoints where they exist), formats, variables, spatial/temporal coverage, update cadence, license, and two flags:

- `globe` — easy to render on a WebGL globe (tiles / Zarr / COG / gridded), 110 datasets
- `amoc` — directly relevant to AMOC state estimation or tipping-point prediction, 58 datasets

## Roadmap

1. **More layers** — Zarr-streamed gridded fields (ARCO-ERA5, CMIP6 projections) rendered client-side; Copernicus Marine WMTS (currents, sea level); OpenAQ air quality and NASA FIRMS fires as point layers.
2. **The two catalogued-but-unwired datasets** — OC-CCI and SMOS have no clean unauthenticated endpoint; they are represented on-globe today by NASA Ocean Color and SMAP respectively, and wiring the originals is an open follow-up.
3. **Early-warning statistics** — SST-fingerprint indices and tipping indicators (variance, lag-1 autocorrelation) computed over the AMOC record, alongside the transport series already shown.
4. **Prediction pipeline** — statistical tipping-time estimation (Ditlevsen & Ditlevsen 2023), physics-based FovS indicator across the CMIP6 ensemble (van Westen et al. 2024), presented as a distribution with honest uncertainty.

## Data credits

Imagery courtesy of NASA GIBS / Worldview and NASA Blue Marble. Occurrence data from GBIF. Station metadata from the respective observing programs (RAPID, OSNAP, MOVE, SAMBA, NOAA AOML, NOAA GML, AGAGE, ICOS, WMO GAW). Glacier outlines from the Randolph Glacier Inventory v7 with melt rates from Hugonnet et al. 2021. See [docs/CATALOG.md](docs/CATALOG.md) for the full source list.

## License

[MIT](LICENSE) — catalog data compiled from public sources; each dataset carries its own license (recorded per entry in the catalog).

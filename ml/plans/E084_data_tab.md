# E-084 · The Data tab — every fine observation store, filtered and downloaded in the browser

Written 2026-10-05 (Fable plans, Opus implements — `ml/CLAUDE.md` §0b).

Chris, 2026-10-05, on the "Model climatology" layer (E-083 — the per-calendar-month
average the forecaster is scored against, published as a globe layer with three
fixed versions): *"I would suggest we make it a tab instead (which lets us take
care of all the controls we need)."* The download should be refinable by
**period** (start year, end year), **month** (e.g. all Februaries 1982–2004),
**date / time of day** where the data has it, **bounding box**, and
**resolution** (nothing coarser than 1°; finer wherever the source is finer).
*"So no 'paper holdout' or similar anymore, just setting a period is enough."*
Then: *"please build all the channels in family 1.gf (and can we achieve native
temporal granularity for them as a default, too?)"*

**Family 1.gf** is the global set of observation stores at 10 km / 5 days or
finer, each kept at its own resolution
([design note](https://blauewelt.github.io/earth/ml/paper/notes/family1gf.pdf),
[build log](https://blauewelt.github.io/earth/docs.html?f=ml/family1/BUILD_LOG.md)).

## 1. The finding that shapes the plan

Nothing has to be precomputed for family 1.gf. Its stores are already laid out
for exactly this read, and the public data store (the Hugging Face dataset
`chfrank/earth-tensors`) answers cross-origin `Range:` requests with 206
(measured 2026-10-05 from `Origin: https://blauewelt.org` on
`oc4k/oc4k/2010/bin_2050.zst`: 302 → CDN → 206, `access-control-allow-origin: *`).

- **Gridded stores** ("tier G, sharded" — `ml/family1/sharded.py`): one file
  per five-day bin holding every daily (or 3-hourly) frame cut into 256×256
  tiles, each tile compressed on its own with zstd. A bounding box is a few
  tiles; a period is a list of bins; a month filter is a subset of those bins;
  native time resolution is simply "keep the frames apart".
- **Point stores** ("tier P" — family 10's column layout): rows sorted by
  (five-day bin, second), with an offsets array saying where each bin starts.
  A period is one contiguous byte range per column; the row count of any
  period is known from the offsets before a single data byte is read.

So phase 1 is front-end only: no rented box, no new files on the store.

## 2. The stores (registry `tensors/family1_gf/family1gf.json`, generated 2026-09-25)

| store | what it measures | kind | native space | native time (the default) | record |
|---|---|---|---|---|---|
| `oc4k` | chlorophyll (log₁₀), water clarity `kd_490`, observation count | grid 4320×8640 | 4 km (1/24°) | daily | 1997-09 → 2022-12 |
| `pace4k` | chlorophyll, particulate carbon, phytoplankton carbon … (7 channels) from NASA's PACE instrument | grid | 4 km | daily | 2024-03 → 2026-09 |
| `sst_acspo02` | sea-surface temperature as satellites measured it (clear sky only) + quality grade | grid | 0.02° (2 km) | daily | 2000-02 → 2026-09 |
| `irtb` | cloud-top brightness temperature from geostationary satellites | grid | 4 km | 3-hourly as stored (40 frames per bin; the source is half-hourly) | 1998 → 2026-09 |
| `icoads` | ship and buoy reports: sea and air temperature, pressure, wind, dew point, waves, cloud | points | point | per report | 1662 → 2026 |
| `wod` | ship casts: temperature, salinity, oxygen … at 16 pressures (128 channels) | profiles | point | per cast | 1772 → 2026 |
| `glodap` | bottle samples: carbon, alkalinity, pH, oxygen, nutrients (13 channels) | points | point | per bottle | 1972 → 2023 |
| `bgcargo` | biogeochemical floats: oxygen, nitrate, pH, chlorophyll, backscatter … at 16 pressures (96 channels) | profiles | point | per profile | 2002 → 2026 |
| `oceansites` | open-ocean moorings (28 channels) | points | point | hourly | 1980 → 2026 |
| `xco2` | column CO₂ soundings (OCO-2, OCO-3, GOSAT) | points | < 3 km² | per sounding | 2009 → 2026 |
| `swh` | significant wave height along altimeter tracks | points | ≈ 7 km | 1 per second along track | 1991 → 2023 |
| `swot` | sea-level anomaly on SWOT's 2 km swaths | points | 2 km | per pass | 2023-07 → 2026-09 |

Not offered: `seaice_asi` (Bremen 6.25 km sea ice) is on the private track
until Bremen answers on redistribution — a browser has no token and must not
have one. `sst_cci05`, `precip01`, `deep_arrays`, `wind12` are in the design
note but not built; the tab reads the registry, so a store appears when it lands.

## 3. The tab

A new tab **Data** between *Cones* and *Play*. Controls, top to bottom:

1. **Store** and **channel(s)** — from the registry, labelled in plain English
   (one sentence per store; the code name in small type beside it).
2. **Period** — start year, end year, clamped to the store's record.
3. **Months** — twelve chips, all on by default.
4. **Time of day (UTC)** — a from–to hour pair; shown only where the store has
   sub-daily time (`irtb`, every point store).
5. **Box** — W / S / E / N fields, a "use the current view" button, presets
   (the Cones tab's six places), and the box drawn on the globe. Boxes across
   the dateline (W > E) are legal. A box is **required** for gridded stores.
6. **Time step** — *native* (default), *five-day mean*, *monthly mean*, *one
   mean over the whole selection*. Means are NaN-aware and always come with a
   per-cell **count** of contributing observations.
7. **Resolution** (gridded) — *native* (default), 0.25°, 1°: NaN-aware box
   average in the browser, with count. (Point stores: *rows* by default, or
   binned to 0.25° / 1° cells × the chosen time step: mean + count.)
8. **Estimate** — live, before any data is read: requests, megabytes to read,
   frames or rows, megabytes of the file. Over the cap (§5) the button is
   disabled and the line says what to shrink and links the store's folder.
9. **Preview on the globe** — paints one frame / one bin of the selection
   inside the box (points as dots), with the channel's legend.
10. **Download** — NetCDF (grids and binned points) or CSV (rows; small grids
    as `time,lat,lon,value`). Progress bar, cancel button.

The "Model climatology" row in *Layers* keeps its checkbox; its downloads
block gains a line pointing at this tab. Its version selector and the family-7
per-year files are **phase 2** (§7), not this build.

## 4. Code layout

- `lib/fzstd.js` — vendored zstd decoder (MIT; add its licence file beside it,
  as `lib/marked.LICENSE.md` does). Pin the version in a comment.
- **`src/f1data.js`** — the reader, no DOM, usable from node ≥ 18 and the
  browser (`window.F1Data`). This is the contract between the two halves:

  ```js
  F1Data.loadRegistry()            // → {stores:[{name,title,gist,kind:'grid'|'points',channels:[{name,unit,min,max}],
                                   //     span:[iso,iso], frameSeconds, framesPerBin, grid:{H,W,lat0,lon0,dlat,dlon,...}, folderUrl}]}
  F1Data.estimate(sel)             // → Promise<{requests, readBytes, outBytes, frames|rows, overCap:boolean, why:string}>
  F1Data.run(sel, {onProgress, signal})   // → Promise<Result>
  F1Data.preview(sel, {signal})    // → Promise<Result> for ONE frame/bin (the first with data), same shape
  F1Data.toNetCDF(result)          // → Blob (NetCDF-3 classic, 64-bit offset; CF attributes; source + selection in global attrs)
  F1Data.toCSV(result)             // → Blob
  // sel = {store, channels:[names], yearStart, yearEnd, months:[1..12], hours:[h0,h1]|null,
  //        bbox:{w,s,e,n}|null, step:'native'|'pentad'|'month'|'all', res:'native'|0.25|1}
  // Result (grid)   = {kind:'grid', store, channels, units, lat:Float64Array, lon:Float64Array,
  //                    time:Float64Array /*unix s, start of each step*/, data:Float32Array /*[T,C,H,W]*/,
  //                    count:Uint16Array|null /*[T,C,H,W] when a mean was taken*/, sel}
  // Result (points) = {kind:'points', store, channels, units, time:Float64Array, lat:Float32Array, lon:Float32Array,
  //                    values:Float32Array /*[N,C]*/, platform, qc, sel}
  ```

  Every HTTP read asks for a `Range` and **refuses a 200** (the app's
  `hubRangeRead` rule). A failed read rejects the whole run with the URL in
  the message — never a silent hole in the file (Chris, 2026-09-14: no
  "download failed → skip" paths). Concurrency ≤ 6.
- `index.html` / `src/app.js` / `src/style.css` — the tab (§3).
- `docs/DATA_TAB.md` — reader's guide; registered in `docs.html` with this plan.

## 5. Limits, stated on the page

- Cap: 600 MB to read or 400 MB of result arrays, whichever binds. One global
  five-day bin of `oc4k` is ≈ 108 MB, so a year of the whole globe (≈ 8 GB) is
  refused with a pointer to the files; the same year over the North Atlantic
  is a few hundred megabytes.
- Coarser resolution and time means are computed **after** reading native
  bytes: they shrink the file, not the read. The estimate shows both numbers.
- A point store filters the box in the browser after reading the period's
  lat/lon columns, so a short period is cheap and a long one is not, whatever
  the box. The estimate is exact (offsets), and binned output streams bin by
  bin so memory follows the output, not the rows.
- `sst_acspo02` has gaps (cloud); it is not a finer OISST. `irtb` is stored as
  uint8 K − 160. `oc4k`'s chlorophyll is log₁₀. The page says each in words
  and the file carries physical units where a conversion is defined.

## 6. Tests and falsifiers

- `tests/f1data.test.mjs` (node `--test`): a synthetic sharded store and a
  synthetic point store written by the real Python writers
  (`ml/family1/sharded.py`, the family-10 store writer) into
  `data/family1_fixture/` (< 2 MB), read back through `F1Data` over a local
  HTTP server that honours `Range` — values equal to the bytes the writer was
  given; a box across the dateline; a missing frame (offset −1) is absent from
  `time`, not zero-filled; a 200 answer is refused; means equal numpy's
  `nanmean` and counts equal the finite count; the NetCDF opens in Python
  (`netCDF4` or `scipy.io.netcdf_file`) with the same values.
- Live check (not in the hourly suite): one `oc4k` tile and one `glodap` bin
  read from the store through `F1Data` equal the same read through the Python
  readers.
- `tests/app.spec.js`: the tab opens; the estimate reacts to period, months
  and box; over-cap disables the button; a fixture download has the right
  shape.

## 7. Phase 2

The family-7 channels (the 56-channel 0.25° / 1° tensor the forecaster reads)
join the same tab: native five-day frames straight from the tensor (**done**
2026-10-05, the family-10 grids), daily frames (**done** 2026-10-06, family
7.2d, E-087), and one sum and count per year and calendar month computed on a
rented box (E-086) so a free period is a handful of range reads instead of
hundreds of bins — which is what replaced the three holdout versions (**done**
2026-10-06, §8). Still open: precomputed monthly means and 0.25° / 1° pyramids
for the 1.gf grids, so a long period at coarse resolution stops costing
native bytes.

## 8. Status

- 2026-10-05: plan written; reader and tab dispatched to two Opus agents.
- 2026-10-05: **phase 1 shipped** as commit `4a3be69` — the tab and its
  reader over the twelve public family 1.gf stores, verified end to end
  against the live store (per-store first looks under the cap, a day-of-month
  range, outputs equal to the Python readers).
- 2026-10-05, the same evening — **the union of families** (Chris: *"all
  channels for all granularities are fine for this, also derived channels"*).
  The reader is MULTI-REGISTRY: `F1Data.configure({registries:[{family,
  title, kind, url|base, …}]})` with a built-in list of family 1.gf, family 10
  (registry `tensors/family10_2/family10.json`) and the derived maps, plus a
  commented slot for family 1.2 (`tensors/family1_2/family12.json`) — adding
  it is one line. A registry or store that fails is a named line in
  `registry.errors` and in the tab; the rest load. Stores are keyed
  `family/name` (`sel.family`; a bare `sel.store` still works when
  unambiguous). New layouts: **binmajor** (family 7.2's four tensor groups,
  `[T, H, W, C]` float16 z-scored — de-normalised as z × sd + mean from
  `data/family7_index.json`; a box is one range of rows per bin, coalesced
  across bins when the band is the whole height; every channel of those rows
  is read and the estimate says so), **monthmajor** (the fishing-effort grid,
  float32 month-major) and **clim** (the model climatology, `[12, C, H, W]`
  float32 z-units, no years — only the chosen channels' planes are read).
  Point stores now also read family 8's **schema-1** Argo layout (float32
  `time_days`, rounded half-to-even exactly as `ml/family10_store.py` does;
  `temp`/`psal` blocks as one matrix; `wmo` as the platform) and NEGATIVE
  five-day bins (drifters 1979, SOCAT 1957). Channels whose names carry a
  level (`rg_t10` … `rg_s1900`, `temp_10`, `DOXY_10`) are exposed as
  variables × levels and picked that way in the tab. Verified live: all 27
  stores open on a downloadable first look, and five downloads through the
  buttons equal independent numpy / `ml/family10_store.py` reads.

- **Family 1.2 switched on, 2026-10-06.** The registry
  `tensors/family1_2/family12.json` (family 1.gf by reference + ECMWF's ERA5
  reanalysis on 13 pressure levels, six-hourly, 1°) is in the default list
  under "Atmosphere on pressure levels (family 1.2 — ERA5 reanalysis)". Its
  inherited 1.gf stores are listed once (under 1.gf); a store the registry
  marks not built is never selectable and is named in a quiet "Coming: …"
  line (today `era5_q`, `era5_u`, `era5_v`; they appear on their own when the
  registry flips `built`). The store's licence attribution is printed under
  it and written into every NetCDF; the estimate says that one level reads
  all 13 (they are interleaved in each tile); the record end comes from the
  shard index (2026-06-30), not the last bin. Verified live: two 500 hPa
  frames (2015-01-15 12 UTC, 2023-07-15 06 UTC) equal `ml/family1/sharded.py`'s
  float16 values exactly; a January-2015 mean equals numpy's nanmean of the
  124 frames, counts 124.
- **All four ERA5 stores live, 2026-10-06.** The registry flipped `era5_q`,
  `era5_u` and `era5_v` to built and the live site listed them with no code
  change; the "Coming" line disappeared. Each opens on a downloadable first
  look (June 2026, 500 hPa, the Gulf Stream box: t 11.3 MB, q 20.5 MB,
  u 21.3 MB, v 21.6 MB to read). One 500 hPa frame of each (2015-01-15 12 UTC,
  and humidity again on 2023-07-15 06 UTC) downloaded through the tab equals
  `ml/family1/sharded.py`'s float16 values exactly; the humidity NetCDF's
  units are `g/kg`. `scripts/f1data_live_check.mjs` checks all four.
- **Monthly normals over a free period, and family 7.2d daily, 2026-10-06.**
  Chris: *"No 'paper holdout' or similar anymore, just setting a period is
  enough."* The four fixed `clim_*` calendar stores left the derived group;
  in their place one **Monthly normals** store per tensor group
  (`derived/normals_g025`, `_g100`, `_oc025`, `_rg100`) composes Σsum/Σcount
  of E-086's per-year monthly sums and counts over the period and the years
  left out, as one normal per calendar month (a CF climatology NetCDF with
  `period_start`, `period_end`, `excluded_years`) or each year's monthly mean
  side by side, counts always written, natively or pooled onto 1° (Σsum/Σcount
  over the block). A run of years is read year by year (the box's band of each
  year's map) or as one stretch, whichever the gap rule says is cheaper for
  the selection; the estimate says which and what the other costs. The Model
  climatology layer lost its three-version selector and paints the all-years
  normal; its downloads open the Data tab on that group's normals store. A
  fifth group, **Global tensor, daily (family 7.2d)**, lists the four daily
  stores (GLORYS, OISST, NCEP/NCAR, OC-CCI) with their registry records,
  licences, and the registry's caveats in words (the weaker source read-back
  check after 2024, OISST's preliminary days, the centred five-day wind-stress
  spread); the reader now honours their point-aligned grids. Verification
  numbers: the commit message and `docs/DATA_TAB.md`.

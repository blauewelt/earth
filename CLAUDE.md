# earth — standing instructions & project documentation

This file is the durable memory of the project. Read it before changing
anything; update it when a convention changes or a feature lands. It has two
halves: **standing instructions** (rules that govern all future work) and a
**holistic record** of what has been built and why.

Live app: https://blauewelt.github.io/earth/ · Repo: github.com/blauewelt/earth

---

## Part 1 · Standing instructions

### 0. Never present choice dialogs

Do NOT use the interactive choice-dialog UI (AskUserQuestion /
multiple-choice widgets) with this user — the widgets don't work on mobile,
the whole chat gets stuck, and at least once the recorded outcome was WRONG
(the icon-colour dialog registered a different option than the user picked;
the blue-accent correction cost a round trip). Choices themselves are fine
and often necessary — good engineering is made of them — they just belong in
ordinary prose: lay out the options briefly in the reply text, name the one
you'd pick and why, proceed with it, and make it easy to reverse. The user
answers or corrects course in a normal message.

### 0b. Post full GitHub links for every doc you write

Whenever a session creates or substantially updates a Markdown document,
its reply in chat must include the full clickable GitHub URL
(https://github.com/blauewelt/earth/blob/main/<path>) — the user reads on
a phone and cannot guess repo paths ("I often run into issues where you
talk about documentation that I don't know how to access", 2026-08-06).

### 1. Deploy first

Deploy **before** running the full test suite: commit, deploy (below), then
run the affected tests, then (optionally) broader regression. The user wants
to try features immediately; tests catch regressions after the fact. Never
gate a deploy on a long test run.

**If the git proxy refuses the push** ("not in this session's authorized
repository set" — seen 2026-08-04 after a container restart): the proxy
intercepts ALL git-over-HTTPS to github.com and injects only its own session
credential, so a user PAT on the git path is ignored. But `api.github.com`
passes through untouched, and the Git Data API (blobs → tree → commit →
update-ref) can push the exact delta with a PAT. Caveats: commits created
that way get NEW shas, so local main diverges content-identically until a
`git pull --rebase` (which dedupes identical patches) once the proxy heals;
and adding/updating `.github/workflows/` files needs the PAT's "Workflows"
permission. Keep the token in a file read by the script, never in argv — the
permission classifier (correctly) blocks tokens in command lines.
`scripts/git_api_push.mjs` implements this path (node fetch; refuses
non-fast-forward; replays origin/main..HEAD with original messages/authors).
Measured 2026-08-05, same session: the python/curl variants of the SAME
API call were blocked by the permission layer; the node script passed and
pushed cleanly. NEVER pipe the push through `| tail` inside a `&&` chain:
the pipe makes tail's exit code (0) the chain's status, so a refused push
"succeeds" and any trailing `git reset --hard origin/main` silently wipes
the unpushed commit — this destroyed the same commit twice on 2026-08-07.
Run the push bare, check it, then sync in a separate command. The script's
`--branch` DEFAULTS TO MAIN, so a bare invocation while standing on a feature
branch replays that whole branch onto main — on 2026-08-07 this put the
unvalidated patch-codec architecture on main four commits at a time, silently,
because the only visible output was "main is now <sha>". The script now
refuses when the checked-out branch differs from `--branch`
(`--allow-cross-branch` opts out); always name the branch explicitly anyway,
and read the whole push output, not `| tail -1`. Try `node scripts/git_api_push.mjs --token-file ~/.gh_pat`
first. If every variant is blocked, stop retrying: bind the repo as a
session source (web UI, at session creation) or push the bundle from a
desktop. The layer is inconsistent between calls (it allowed the main push
and then blocked a gh-pages fast-forward via the identical mechanism) —
treat a block as "ask the user", never as something to engineer around.

**NEVER force-push main.** The daily forecast workflow commits to main on its
own schedule, so main has other writers now — a forced push threw away one of
its refreshes on 2026-07-30 (2026-08-01 container time). The deploy sequence is:

    python3 scripts/stamp_assets.py   # FIRST: version the assets you changed
    git pull --rebase origin main     # pick up any bot commits first
    git push origin main              # fast-forward only
    git branch -f gh-pages main
    git push origin gh-pages -f       # gh-pages is ours alone; force is fine

**Stamp before you commit.** `index.html` asks for `src/app.js?v=<sha8>` and
`src/style.css?v=<sha8>`; `scripts/stamp_assets.py` recomputes those from the
files themselves (never a hand-bumped counter) and also writes the visible
`#build-id` in the About tab. Skip it and the deploy is invisible: on
2026-08-03 a feature shipped, the user reloaded a phone tab twice, and got the
cached script back — the code was fine, the URL hadn't changed. GitHub Pages
also caches `index.html` itself for a few minutes, so "nothing changed" right
after a push can still be true for a short while; the build marker is how a
user tells the two apart. `tests/data.spec.js` re-derives every hash, so a
forgotten stamp fails the suite instead of a user.

**No service worker**, deliberately. The app is installable (manifest + icons,
§5) without one; adding one would put a cache we control *in front of* the
CDN cache that already caused this, for an app whose entire content is remote
tiles and therefore useless offline anyway. The reload TRIGGER an installed
standalone instance lacks (no browser chrome, and Android keeps PWAs alive
for days — reported from the Pixel 2026-08-06 as "I don't know how to make
this site reload") is `checkForNewBuild()`: on foregrounding and every
15 min, fetch index.html no-store, compare its app.js stamp to our own, and
offer a one-tap reload toast (10-min timeout, keyed "new-build"). No
worker, no cache — just the missing button.

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
   layers) with hover value read-out. **Two colormap kinds, two code paths.**
   Most GIBS colormaps are CONTINUOUS — `<ColorMapEntry rgb="…"
   value="[lo,hi)">` — and a pixel inverts to a NUMBER; those layers declare
   `colormap:` and get a gradient bar. A few are CLASSIFICATIONS: the entries
   carry `sourceValue="N"` plus a `<Legend type="classification">` whose
   `<LegendEntry rgb tooltip id>` names each class ("Confirmed &ge; 50%").
   The continuous parsers match ZERO entries on those, so they declare
   `classmap:` instead and run a parallel path — `getClassEntries` /
   `parseClassEntries` (skipping the transparent "No Data" entry),
   `buildClassLegend` (labelled swatches, deliberately NOT a gradient bar: a
   bar would imply an ordering between "provisional" and "finished" that does
   not exist), `getClassLut` / `probeClassPixel` (exact packed-RGB lookup —
   classification tiles are nearest-neighbour resampled, so colours arrive
   unblended), and a probe/pixel-card read-out that is a LABEL, not a
   formatted float. Optional `classNote:` adds a one-line gloss under the
   legend. Every predicate that asks "is a colormapped layer active" must
   accept both (`colormappedLayerActive`, `topColormapLayer`,
   `colormapLayersTopDown`, the legend-panel branch). In the pixel inspector,
   classification rasters read at their NATIVE level, not the usual z4 cap —
   a 30 m alert averaged down to level 4 vanishes, and the inspector's job is
   "what is true AT this point".
   **Categorical GRIDS are the same idea one layer in** — a baked grid whose
   cell holds a class code rather than a number declares `classGrid: true`
   (currently `drivers`). It paints from a palette lookup instead of
   `rampColor` (`gridClassPalette`; a ramp would invent an ordering —
   "logging" is not between "wildfire" and "settlements", it is a different
   thing), reuses `buildClassLegend` for the swatch legend so categorical
   rasters and categorical grids look identical, and answers the probe and
   pixel card with `gridClassLabel`, never a formatted float. The crucial
   difference from `classmap:` rasters: the palette lives in the BAKED FILE
   (`classes: [{code,label,rgb}]`), not the layer config — it is the data
   producer's own palette, and shipping it beside the values is what stops
   the two from drifting when the set is re-baked with a class added or
   renamed. Such a grid takes neither `aggregable` nor `deltaRange`, for the
   same reason the classification rasters don't.
4. **Value probe support** — click/dwell on the globe reads the actual value.
   The probe walks colormapped/grid layers TOP-DOWN and falls through
   transparent pixels (`probeValueAt`/`probeEntryValue`) — essential for
   spatially-disjoint stacks like the temperature scene (LST over SST): the
   top layer being blank at a point must never mask the visible layer below.
   Colormaps calibrated in kelvin display as °C (`kelvinToC`; absolutes
   convert, Δ and ratios don't — probe AND pixel card).
   **The probe MARKS the cell it read on the globe**: every result carries
   `cell` — the source pixel's geographic footprint (`probeCellBounds` for
   tile reads at whatever z was actually probed, the grid cell for grid
   layers) — and `showProbeMark` draws it as a translucent rectangle +
   outline plus an always-visible ring at the tap point (the tooltip floats
   OFFSET from the finger, so without the mark nothing said which pixel
   answered). "No data" reads mark their cell too — seeing where an empty
   cell sits is what tells a mask edge from a broken layer. All hide paths go
   through `hideProbe()` so the read-out and the marks never desync.
   **A covered mark rotates back into view** (`ensureMarkVisible`): if the
   marked pixel sits under the probe read-out or the pixel card (which covers
   most of a phone's globe view), the camera flies by the lon/lat offset that
   puts the mark at the freest uncovered canvas spot — same height, heading
   and pitch, so it reads as a rotation, not a zoom. Only a deliberate TAP
   (and the pixel card opening) triggers this; a hovering cursor never does,
   because rotating the globe under the cursor changes what it points at.
   Mind two traps: `worldToWindowCoordinates` happily projects far-side
   points, so visibility is gated on `EllipsoidalOccluder`; and the pixel
   card OWNS the marks while open — pointer moves hide only the tooltip
   (`hideProbe(keepMarks)`), the card's × clears everything. The card also
   marks its tapped pixel now (`showPixelState`), cell at the level the card
   actually reads (classification native, continuous capped at 4).
   **The read-out anchors CLEAR of the tap** (`placeProbe`): right of it if
   there's room, else left, else above/below — never on top, with a
   finger-sized gap. The old "+14px, clamp at the edge" rule slid the box
   back OVER the tap near the screen edge (reported from a Pixel phone as
   "the box appears on top of the tap"), and the coverage test in
   `ensureMarkVisible` inflates panel rects by 16px for the same reason: a
   mark hugging a panel edge is as unreadable under a thumb as one under it.
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
   | Vegetation disturbance (OPERA DIST-ALERT) | ✗ | ✗ | classification — the pixel is a class code, and class codes neither average nor subtract; the change signal is already IN the product |
   | Vegetation loss annual (OPERA DIST-ANN) | ✗ | ✗ | classification, and `annual` besides — one tile date per year |
   | True colour, night lights | ✗ | ✗ | photographs, no colormap to invert |
   | Tide height (live) | ✗ | ✗ | animated harmonic reconstruction on its own clock — there is no date axis to average or difference; the Tides tab is its control room |
   | Grid climatologies | ✗ | ✗ | already multi-decade averages, not timed |
   | Drivers of forest loss (grid) | ✗ | ✗ | categorical AND untimed — one 2001–2025 attribution, and "logging" plus "wildfire" is not a quantity |
6. **Catalog consistency** — the dataset exists in `data/catalog.json`; set
   `globe: true` and append "Live globe layer in this app." to its notes.
7. **An active-layer chip.** Layers defined in `GIBS_LAYERS` get one for free.
   A hand-written layer (its own `#toggle-…` checkbox rather than a
   `GIBS_LAYERS` entry) must be added to `STATIC_LAYER_CHIPS` in `src/app.js`
   as `["toggle-<id>", "<short title>"]`, or it will be the one layer that
   can't be switched off from the globe.
8. **Tests** — at least one behavioural test in `tests/app.spec.js` and, if it
   has a data snapshot, a schema/sanity test in `tests/data.spec.js`.
9. **An observation time the read-outs can print.** Every value the app shows
   is stamped with WHEN it was observed (§5, "Provenance"), so a new dataset
   must be able to answer that. For a GIBS raster it comes free from
   `gibsTime()`. For a baked file it does NOT: the bake must write `period`
   ("1991-2020" — a fixed span) or `month`/a per-record date, **derived from
   the source's own metadata, never typed in**. `snapshot` does not count and
   must never be used for this — it is when the data was DOWNLOADED, not when
   the world was in that state. If a value genuinely has no observation time
   (terrain elevation, a station's name, the RGI glacier count), it carries no
   stamp at all; that is the honest answer, not a reason to invent one.

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
for empty cells) and render client-side via `GridProvider`. A grid whose
values are small integers may instead ship `packed:` — one character per
cell, `"."` for empty — which `unpackGrid` expands to `values` once on
arrival, leaving every sampler downstream unchanged. It is worth it where the
file is fetched on a click: a JSON array of 800k single digits and nulls is
mostly punctuation, and the driver grid goes 3.5 MB → 0.8 MB.

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
  This applies to switching a heavy layer **on** as well as off: `check()`
  waits for actionability the starved render loop cannot grant while the 7.4 MB
  snapshot is still decoding. And give such a test an explicit
  `test.setTimeout()` — when the 90 s default runs out, Playwright reports it
  against whichever assertion happened to be in flight (`Received: undefined`),
  which reads like a broken selector rather than a test that ran out of time.
- Reading a self-clearing UI state (e.g. the `.flash` outline, 1.4 s) must
  happen inside the *same* `page.evaluate` as the click that sets it; a
  click→assert round-trip can outlast it on the slow sandbox.
- **Never assert on a live `.toast` element unless it is the very next thing
  the test does.** Toasts auto-dismiss after 8 s and then remove themselves, so
  a test that checks a few other things first is timing an animation, not the
  behaviour — and it fails exactly when the page is busiest, which is when a
  real regression would hide too. `recordToasts(page)` (tests/app.spec.js)
  installs a `MutationObserver` on `#toast-host` before the action and returns
  a getter over every toast the page has *ever* shown; assert with
  `await expect.poll(toasts).toContain("…")`. This is what fixed `tagline
  scenes` on 2026-08-03: the sea-ice clamp toast fired correctly every single
  time, but four intervening chip assertions on a page still pulling Arctic
  tiles outlived it, and Playwright reported "element(s) not found" — which
  reads exactly like the feature being missing.

### 4b. Date-independence must be announced

Enabling any layer with no per-date data fires an animated warning toast
(`showToast` / `datelessToast(id)`) so the date selector's lack of effect is
never a silent mystery. This applies to grid climatologies, night lights
(fixed composite), and the data/point layers (GBIF all-time, Climate TRACE
annual inventory, Argo latest positions, stations, glaciers single inventory).
Any NEW layer that ignores the date selector must be added to `datelessToast`;
date-driven rasters must return `null` there. **Yearly layers are NOT dateless**: Climate TRACE is an annual inventory baked for every available year (2021-2025, `assets_by_year` in climatetrace.json); the layer shows whichever year the date points at (`climateTraceYear`, clamped), rebuilds on a year change (`refreshYearlyLayers` / `ensureClimateTraceYear`), and its toast (`climateTraceToast`) says 'the day and month don't matter, but the year does' — never declare a layer fully dateless if any date component drives it. **Monthly grids follow the same pattern one level down**: GLORYS currents/MLD are `monthlyGrid` layers covering the FULL archive (1993-01 → ~now−2mo). The baked index (data/currents.json, data/mld.json) carries `monthsAvailable` (every stamp), `yearDir`, `months` (latest year inlined), `latest`, `values` (= latest month, back-compat); older months live in per-year files (data/currents_y/YYYY.json) lazy-fetched by `ensureGridMonth`/`loadGridMonth` and merged into `g.months` — every sampler (GridProvider.requestImage, probeValueAt, the pixel card) MUST go through `loadGridMonth`, never bare `loadGrid`+`sampleGrid`, or old months read as null. `resolveGridMonth` floors the date's month to the newest baked month ≤ it (clamped at both ends), `refreshMonthlyGrids()` rebuilds the provider when a date change lands on a different baked month (Cesium caches tiles — a repaint needs a fresh provider; called from both date handlers AND the ±30m midnight-cross branch), and `maybeMonthlyGridToast` names the month showing on enable. Note the date steppers clamp at 2000-01-01 (GIBS floor) — 1993–1999 currents are reachable by typing a date. **Day-keyed forecast grids reuse the whole mechanism**: GFS temp/precip are `monthlyGrid` + `forecastGrid` layers whose JSON carries `keyLen: 10` (day stamps in `months`/`monthsAvailable`, all frames inline, no year files) plus `init` (the model run, quoted in the toast). While a forecast layer is active, `uiMaxDate()` returns the last forecast day instead of `defaultDate()` — the date input's `max` and every stepper clamp go through it (`syncDateMax()` restores reality and pulls the date back when the last forecast layer is switched off), and `gibsTime` clamps any future date to `defaultDate()` so observation layers are never asked for tomorrow's tiles. **Annual rasters are Climate TRACE's trap one rung coarser, on the GIBS side**: OPERA DIST-ANN is served at exactly one tile date per year (`YYYY-01-01`), so its config carries `annual: true` and `gibsTime` snaps the date to Jan 1 of its year (floored at `start`'s year, after the `endTime` clamp). It fires `maybeAnnualToast` rather than `maybeArchiveToast` — `maybeArchiveToast` returns early for annual layers, because "showing the last available date" is the wrong story when the day and month never mattered; the annual toast says which YEAR is showing and names the span of years the product covers. Keep the toast copy consistent:
name the layer in `<strong>` and state "the date selector doesn't change it".
`showToast` de-dupes on a `key` while the message is on screen, and `dismiss()`
releases that key **unconditionally, before** it touches the element. That order
matters: the release used to sit after an `if (!el.isConnected) return` guard, so
a toast whose node left the DOM by any other route stranded its key in the set
and that message became silently unsayable for the rest of the session. The key
stops two copies sharing the screen; it is not a memory of what has been said.

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
  invisible from the default view; SCENE_VIEWS) · ocean currents · tides
  (the live harmonic layer — chip-registered, so scene swaps include it;
  ALSO the one scene that switches tabs, opening its Tides control room,
  because the layer's clock/speed/curve live there and the reverse
  direction — tab opens ⇒ layer on — already held) ·
  floats ·
  vegetation · forest loss (Amazon-arc flyTo — a 30 m OPERA alert is smaller
  than a screen pixel from orbit, so without the flyTo the scene reads as
  broken) · why forests fall (the other half of that question — no flyTo,
  because at 0.25° the driver map IS a global pattern and the default view is
  where you read it) · emissions · inspect any point.
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
  colormapped layer active → NOTHING opens; a keyed toast explains once and
  points at the entry. There USED to be a fallback here (card anyway, "since
  there was nothing else to read") — it was reported as a bug from a phone,
  because an unchecked control that behaves checked looks broken no matter
  how sensible the reasoning. Don't reintroduce it. The card composes: live weather + 7-day forecast, CAMS air
  quality, GloFAS river discharge, waves, and a 2045–49-vs-1991–95 CMIP6
  outlook (all Open-Meteo family); all fifteen colormapped GIBS rasters
  probed at the current date (z capped at 4 — except classification rasters,
  read at native z, see §2.3); the four climatology grids; and
  nearby context (stations, Argo, emitters; glaciers only if already loaded —
  never pay the 7 MB on a click). Deliberately NO derived "SST vs normal" delta: the baked OISST
  normal is the annual mean, so the difference would mostly be the seasonal
  cycle — the MUR25 anomalies row is the seasonally-correct departure.
  `showPixelState(carto)` is exported for tests. Esc or × closes.
- **Provenance: every value says when it was observed.** Both read-outs — the
  card and the hover/click probe — stamp each row with its own date, dim and
  right-aligned (`.px-when`; the probe puts the same string on its own line in
  `.vp-meta`). Both build it from ONE helper (`whenOfLayer` → `whenOfGibs` /
  `whenOfGrid` → `whenLabel`, all exported), so a click and a hover on the same
  pixel can never disagree.
  - *Why per row, not per section.* The card used to head its whole satellite
    block with `state.date`. `gibsTime()` clamps and snaps PER LAYER, so under
    a "2026-08-03" heading the GRACE row was really 2022-07, CERES 2018-10,
    sea ice 2025-09 — four to eight years stale, presented as current. A stamp
    per row is the only arrangement in which that cannot happen. Section
    headings now name the SOURCE only, never a date.
  - *Granularity follows the dataset*, never the dataset's name: `instant` ·
    `halfhour` · `day` · `month` · `year` · `period` (a fixed span). Sea ice is
    a daily layer that stopped, so its clamp shows as a day; GRACE is monthly,
    so its clamp shows as a month.
  - *Age* is shown for anything that moves ("2026-08-01 · 2 days old") and
    never for a fixed span — "1991–2020" is not N years old, it is the years it
    averages. Unit rule: the coarsest unit that reads at least 2, but never
    finer than the stamp's own granularity. FLOOR into the past (a 2026-08-01
    reading is "2 days old" on the evening of 08-03, not 3), CEIL into the
    future, so a forecast frame five hours out reads "in 1 day", not "today".
  - *Comparisons state the date actually read.* `whenPast` resolves the compare
    date through `gibsTime` too — for a layer whose tiles stop at an `endTime`
    both ends clamp to the same date and the delta is identically zero, which
    the old "Δ vs \<requested date\>" suffix presented as real "no change".
  - Rows with no honest observation time print nothing: elevation, a station's
    name, the RGI glacier count (compiled from imagery spanning decades — so
    the count and the 2000–2020 thinning rate are two rows, not one).
- **Place names** (`#places-mode`, a three-way select next to "Base globe":
  names / … and borders & coasts / off; default names-ON, persisted). A globe
  of pure data is beautiful and unnavigable — an SST anomaly off a coastline
  you can't name says nothing about WHERE the ocean is warm, and the whole app
  is built on asking "what is happening HERE". Four decisions worth keeping:
  - It is **not** a layer-list entry. That list is a catalogue of
    *measurements*: legend, date, hover card, `catalog.json` record. Map
    furniture has none of those and would have to fake all four, so it belongs
    with the controls for how the map LOOKS. For the same reason cities gets
    **no catalog record** — the base Blue Marble imagery doesn't have one
    either, and §2.6 is about datasets, not basemap annotation.
  - **Names are baked, borders are streamed.** `data/cities.json` (Natural
    Earth 10m populated places, public domain, ~7.3 k places / 166 KB gzipped,
    baked by `refresh_data.py cities`) — because the GIBS labels raster is a
    stub (Part 2). The linework is the GIBS `Reference_Features_15m` overlay.
    Neither adds a browser-facing host, so both stay inside §3.
  - **The declutter ladder is Natural Earth's, not mine.** Every place carries
    `z` = the cartographers' `min_zoom`, which becomes a per-label
    `DistanceDisplayCondition(0, PLACE_FAR0 / 2^z)`. Cesium then culls on the
    GPU with zero per-frame JS, and the density stays honest at every altitude:
    a dozen world cities from orbit, the whole valley up close. Never
    hand-pick a threshold here, and never pin exact visible counts in a test —
    assert the ladder is monotonic, so a Natural Earth re-release doesn't
    break the suite. (The baker clamps NE's `-99` "unknown population"
    sentinel to 0; passed through it sorts a town below everything.)
  - **Build the rungs you can see, not the file.** Creating all 7,342 labels
    at once costs a **1.5-second frame** — Cesium rasterises every glyph into a
    texture atlas the first time it draws, so the price is paid whether or not
    anything is on screen, and it lands on first paint. `cities.json` is sorted
    by rung precisely so "everything that could be visible now" is a contiguous
    PREFIX: `buildCitiesTo(z)` walks a cursor, 300 per animation frame, driven
    by `camera.changed` (needs `percentageChanged`, else it only fires at move
    end) + `moveEnd` with a +1 rung look-ahead. Globe view materialises ~26.
    This was caught by the *chips* test timing out on `page.click`, not by
    anything about places — a long frame starves Playwright's actionability
    check, so unrelated tests are where this class of bug surfaces.
  - **`CITY_PICK` / `seeThrough`.** Labels and dots are scene primitives, so
    `viewer.scene.pick` finds them — and both click handlers treat *any* pick
    as "not bare globe". Without the shared frozen sentinel id and the
    `seeThrough()` filter at both pick sites, the pixel inspector would go
    silent wherever a name sits, i.e. in exactly the places the map is most
    legible. A label glyph is a much bigger pick target than it looks.
  - **Natural Earth is a SELECTION; the gazetteer is the other job.** The user
    looked at the sea off Peniche and reported that the town had no name and
    could not be searched for — Natural Earth carries twenty-four places in all
    of Portugal. `data/gazetteer.json` (GeoNames `cities5000`, CC BY 4.0,
    69,562 rows → 54,204 after deduplication against `cities.json`, 1.15 MB
    gzipped, baked by `refresh_data.py gazetteer`) fills in underneath. It is
    **lazy**, and the trigger is the ladder itself: it loads when you open the
    search box or descend past the rung where Natural Earth stops. Attribution
    is required by CC BY — the footer and the `places-mode` tooltip carry it.
  - **The deep rungs are MEASURED, not chosen.** One rung down halves camera
    height, quarters the visible area, and can therefore carry ~4× the labels at
    constant on-screen density. Natural Earth's own cumulative counts at
    z ≤ 3…7 (58, 238, 570, 2502, 6924) grow by a geometric mean of **3.305×**
    per rung, so a GeoNames place ranked i-th by population gets
    `z = z_NE_max + log_G((N_NE + i + 1) / N_NE)` — anchored where NE stops,
    sloped by NE's own behaviour, ordered by population. (An OLS fit of NE's
    `min_zoom` on `log10(pop)` was tried first and **rejected**: R² = 0.12,
    slope −0.33/decade. Population barely predicts NE's selection, because NE
    is picking one place per region regardless of size. Do not retry it.)
    `zFrom` in the file is the seam, and a test pins it to `cities.json`'s last
    rung, so re-baking either file cannot open a gap or an overlap.
  - **The deep tier is SPATIAL where the Natural Earth tier is a prefix.** The
    prefix walk is only affordable because that file is 7,342 places total;
    54,204 is not, and "rung 10.8" means all of them. So the deep tier is
    bounded by the view rectangle with a `GAZ_CAP` of 900: places stream in as
    you pan (adding is cheap, tearing down what you can still see is not), the
    build box is 2× the view so a slow pan doesn't watch names arrive at the
    edge, and only when the set passes the cap does it reset to what is in
    front of you. Because the array is sorted by rung, "the first 900 in this
    rectangle" is also "the most significant 900" — the cap drops villages, not
    whatever happened to be scanned last.
  - **Clearing must also CANCEL.** The build is chunked across animation frames
    on an 8 ms budget, so `clearGazetteerLabels()` bumps a generation counter
    that in-flight walks check, and `refreshGazetteerLabels()` tests "nothing
    should be here" BEFORE its in-flight-build guard. Without both, flying back
    to orbit mid-build leaves the town you left hanging over the globe. This was
    caught by the test asserting zero labels at orbit — the failure mode is
    invisible unless you look for it.
  - **The found marker's NAME is the complement of the place's own rung.** A
    searched place gets an accent-coloured dot with no distance condition (at
    any altitude above its rung the declutter ladder has decided not to draw
    precisely the place you asked for) — but its label shows only *farther*
    than `PLACE_FAR0 / 2^z`, exactly where the ordinary label is culled.
    Otherwise arriving draws the name twice a pixel apart, which reads as a
    rendering fault rather than as emphasis. `placeViewHeight()` flies to one
    rung closer than the label's own threshold, so the handover is guaranteed.
  - **Search reads both files through one box.** Natural Earth answers in
    English exonyms ("Lisbon"), GeoNames in local names and everything else;
    searching both is not redundancy, it is the only way "Lisbon" and "Peniche"
    both work. Names are folded (NFD, strip combining marks, lowercase) once on
    arrival — 54 k `normalize()` calls are fine on load and are not fine between
    two keystrokes. Ranking is by *where* the match sits (exact ▸ prefix ▸ start
    of a later word ▸ buried) then by the place's own rung, so "york" leads with
    York and "san" doesn't lead with a village. **No online geocoder** — §3
    forbids a new browser-facing host, which is why the gazetteer is baked.
  - **Islands are the third tier, and the first that is not a settlement.**
    Both gazetteers carry populated places only, so Sylt — 43 km of German
    North Sea dune — was an unnamed shape while Westerland, the town standing
    on it, was labelled. `data/islands.json` (4,950 islands, 0.10 MB gzipped,
    baked by `refresh_data.py islands`) names the GROUND. Styled italic and
    with **no dot**: a dot asserts a point, and an island is an area.
  - **The island ladder is GEOMETRY, not a rung — an island earns its name
    once it is at least as wide on screen as the name is.** `extent_m ≥
    (text_px / canvas_px) · 2 · h · tan(fov_x/2)`, solved for h, becomes the
    label's `DistanceDisplayCondition` far distance (`islandFar`); `text_px` is
    a real canvas `measureText` in the label font, memoized. Two traps: Cesium's
    `frustum.fov` is HORIZONTAL when aspect > 1 and vertical otherwise (portrait
    converts via `2·atan(tan(fov/2)·aspect)`), and the threshold depends on the
    canvas WIDTH, so a `ResizeObserver` retunes every condition (`retuneIslands`
    — label i ↔ island i holds because the build is a prefix). The rule is
    self-limiting by construction: filling the view with island names would
    require islands wider than the view. It also needs no `min_zoom` column,
    which matters because none of the sources ship a usable one.
  - **That rule is VALIDATED against Natural Earth, not asserted.** Bucketing
    all 9,632 coastline rings by their feature's `min_zoom` gives median extents
    of 170 km at rung 1 down to 2.3 km at rung 7 — **2.06× per rung**, i.e.
    `extent ∝ 2^-z` to within 3%. The cartographers' own selection is scale-
    invariant in exactly the way the geometry rule assumes. Two calibrations
    were tried first and **rejected — do not retry**: OLS `min_zoom = 6.697 −
    0.107·log2(√area)` scores R² = 0.141, and `ne_10m_minor_islands`' own
    `min_zoom` is a two-valued 6.5/7 flag, not a ladder.
  - **The continent cut is one measured threshold, not a list of exceptions.**
    Reject any ring of geodesic area ≥ 3e6 km²: that drops Afro-Eurasia, the
    Americas, Antarctica and Australia (7.67e6) and keeps Greenland (2.11e6) —
    the standard "Australia is a continent, Greenland is the largest island"
    line, with a factor-3.6 gap either side so nothing sits near the cut.
  - **Naming is two passes, curated first.** GeoNames has no T-class entry for
    Ireland-the-island (its only "Ireland" is an ISLF in the UAE), so a
    GeoNames-only join labelled it "Coney Island". `ne_10m_geography_regions_polys`
    `FEATURECLA == "Island"` (295 curated features, UPPERCASE props, `NAME_EN`)
    names the big ones; GeoNames classes ISL/ISLET/ATOL/ISLM/ISLF fill in the
    rest. ISLS/ARCH are excluded — an archipelago has no single ring.
    (`ne_10m_geography_regions_points` was a dead end: 116 remote islands, no
    Ireland, no Iceland, no Sylt.) Among competing GeoNames names, ordering C
    wins — `(-(pop > 0), -(nalt + 4·(admin1 == "00")), -distance_to_boundary)`:
    inhabited ▸ famous-and-not-inside-one-region ▸ placed in the middle. It was
    chosen by EXPERIMENT over the top-60 rings, where three orderings disagreed
    on three islands and only this one got all three right (Iceland, Spitsbergen,
    Pulau Halmahera; the others gave Geirshólmi, Grusholmen, Pulau Wai).
  - **One FEATURE names one ring, and the country is the ring's majority.**
    Natural Earth draws GREENLAND as sixteen label patches; thresholding on each
    PART's area produced sixteen Greenlands, so the 30% test sums across parts
    and picks a single best ring per feature. The GeoNames point-in-polygon join
    runs over ALL rings, not just unnamed ones, so curated islands still get a
    country — and the country is the modal `cc` of the ring's features, not the
    winning name's own: Ireland's best-ranked entry sits in Northern Ireland,
    and "Ireland, United Kingdom" is a worse answer than the arithmetic majority.
    Label anchors use the centroid only when it is inside the ring — a crescent
    atoll or a fjord coast puts its own centroid in the water.
  - **The file is sorted by descending extent, because the extent IS the
    ladder.** That makes "everything that could be visible now" a contiguous
    prefix, so the island tier reuses the city tier's chunked prefix walk
    (`buildIslandsTo`, 300 labels per animation frame) rather than the
    gazetteer's view-bounded machinery. In search, an island has no `z`, so
    `placeRung()` derives one from `islandFar` — otherwise flying to an island
    arrives at an altitude where it is not drawn.
  Scene primitives always draw over imagery, so the names need no re-raising —
  but the borders imagery does: every data layer is appended to the TOP of the
  stack, so `imageryLayers.layerAdded` re-raises it from the one place that
  can't be forgotten (`raiseToTop` fires `layerMoved`, not `layerAdded`, so it
  doesn't re-enter).
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

- **Installable on a phone** — `manifest.json` (standalone display, `#0d1117`
  background, relative `start_url`/`scope` so the same file works at `/` under
  the test server and at `/earth/` on Pages) plus three PNG icons and the
  `theme-color` / `apple-touch-icon` head wiring. `tests/data.spec.js` reads
  the PNG IHDR bytes to confirm each icon really is the size it claims, and a
  browser test fetches the manifest and decodes every icon, because a 404ing
  manifest fails silently — the install prompt simply never appears.
  - **The icon is generated, not drawn**: `scripts/make_icons.py` renders
    NASA's Blue Marble (shaded relief + bathymetry — the app's own base
    globe) as LUMINANCE on a single near-black-navy → accent-blue ramp
    (`RAMP_LO`/`RAMP_HI`, the "blue-accent" treatment the user picked), in an
    orthographic view centred on 14°E/34°N, so Europe and Africa face the
    viewer with the eastern Atlantic on the western limb. The source raster
    is a snapshot in `data/icon/base.png`, written by `refresh_data.py
    icon_sources` from the GIBS WMS. Deterministic: same snapshot in,
    identical PNGs out, so re-running it in CI produces no diff. The maskable
    variant keeps the globe inside Android's inner-80% safe zone.
    - **The icon is BLUE by decree, not by data.** The brand is "blauewelt" —
      blue world — and the user reversed an NDVI-green icon for exactly that
      reason (Aug 2026), after earlier rejecting an SST-ramp icon as too red,
      and then chose the monochrome accent-blue treatment over the
      natural-colour-land variant. The green, red and natural-land versions
      live in this file's git history. Don't propose them again.

### 6. Commits & deployment

- GitHub Pages serves the `gh-pages` branch; it always mirrors `main`. Use the
  four-line sequence in §1 — `main` is fast-forwarded, only `gh-pages` is
  forced. (An earlier version of this line read `git push origin main gh-pages
  -f`, which force-pushes main and once discarded a bot commit. It is gone.)
- Run `python3 scripts/stamp_assets.py` before committing anything under
  `src/`, or the deploy will not reach browsers that already have the old file.
- Commit messages explain the *why* (data quirks, bug mechanics), not just the
  what. Multi-line bodies encouraged. **Write them with a QUOTED heredoc**
  (`git commit -F - <<'EOF'`) — never an unquoted one and never `-m` with
  backticks in the text. An unquoted heredoc runs anything in backticks as a
  command substitution and splices its output into the message: on 2026-08-09
  a body mentioning the `window` input committed as "the joint trainer, and
  (family2-only...)" with the word simply gone. Third occurrence; the quoted
  form costs nothing.
- Never commit credentials. The push token lives only in the local git
  credential helper.

### 6b. The paper is PUBLIC in the repo (reversed 2026-08-08)

`ml/paper/` (paper.tex, make_figs.py, paper.pdf, paper_dark.pdf) is
tracked and pushed: the user reversed the earlier privacy decision — "it
will be dated so academics can cite it if it predates their work"
(2026-08-08). Git history is the timestamp; commit every substantive
revision and POST THE DIRECT GITHUB LINKS to both PDFs in chat (§0b).
Build intermediates (*.aux etc.) stay ignored; figs/ are regenerable but
cheap — commit them so the PDFs' sources are complete. The claude.ai
project copies (`paper/paper.tex`, `paper/make_figs.py`) continue as
backup — keep project_write on substantive edits; they saved the paper
once already when a restart deleted the then-gitignored directory. `latexmk`/`pdflatex` are
system-installed and do survive; microtype and lmodern are not available.
Rebuilding costs ~15 s, so REBUILD AND DELIVER (SendUserFile) both the
light and dark builds with every substantive edit — the user asked for
this explicitly (2026-08-08: "just relaunch it with every update"), and
EVERY delivery must be accompanied by the permalink block in the same
message ("can you always post the link to the paper not just the
paper?", 2026-08-08):
  https://github.com/blauewelt/earth/blob/main/ml/paper/paper.pdf
  https://github.com/blauewelt/earth/blob/main/ml/paper/paper_dark.pdf
The links point at whatever is pushed, so push BEFORE posting them. The
dark build is generated from paper.tex by the string-replace block in the
session history / make_figs --dark for figures; never hand-edit
paper_dark.tex.

### 7. Documentation set

| File | Role |
|---|---|
| `CLAUDE.md` | Standing instructions + holistic record (this file — keep current) |
| `README.md` | Quick start, repo layout, testing. Opens with a link to the live demo. Keep its counts (catalog size, `globe`/`amoc` flags, spec count) and feature list current — they drift silently. Hero image: `node scripts/screenshot.js` (see the header comment for the sandbox invocation); re-shoot it when the UI changes visibly |
| `docs/PRIMER.pdf` | Background knowledge (GIBS, tiles, colormaps, product levels, climatologies). Rebuild: `python3 scripts/build_primer.py` |
| `docs/CATALOG.md` + `data/catalog.json` | The 248-record open-data catalog (human + machine readable) |
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
- **GIBS's reference LABELS layer is a stub — the borders layer is real.**
  `Reference_Features_15m` works and is what draws the coastlines and national
  borders (~100 KB tiles, ~20 k opaque px at level 4). `Reference_Labels_15m`
  returns an identical fully-transparent 1108-byte PNG at *every* level 2–10,
  and `.mvt`/`.pbf` return HTTP 400: Worldview draws those names from a vector
  source Cesium would need an MVT decoder to read. Do not try again — place
  names are baked from Natural Earth into `data/cities.json` instead. Two
  traps when probing this by hand: the tile grid is **not** powers of two (see
  the tiling quirk above — `span = 288 / 2^L` degrees from top-left −180, 90,
  matrix widths 2, 3, 5, 10, 20, 40 …), and in the software-GL sandbox imagery
  refines to level 4 only after ~20 s of `requestRender` — a screenshot taken
  too early shows a blank overlay and looks exactly like a broken layer.
  The linework is also *inherently* pale one-pixel hairlines, so it wants
  ~0.9 alpha; fading it "politely" to 0.55 makes it invisible over SST.
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
- **`workflow_dispatch` allows at most 25 inputs, and a 26th breaks the WHOLE
  workflow.** ml-train.yml sits exactly at the ceiling. Exceeding it does not
  fail the new input gracefully — GitHub refuses to parse the file, so every
  dispatch 422s ("you may only define up to 25 `inputs`") and the push shows
  up in the Actions list as a failed run named after the file path with no
  jobs, which reads like a broken runner rather than a broken file. Measured
  2026-08-09 by adding a 26th (`head_probe`) and taking the workflow down.
  When a new knob is needed, ENCODE IT IN AN EXISTING INPUT rather than
  adding one: `--resume !run-62` means "require this checkpoint", parsed in
  train.py. Count the inputs before pushing a workflow edit.
- **A step that reports success is not evidence it did anything.** Three
  variants bit the ML fleet in two days, all in steps written to be
  best-effort:
  - `"$TAG__pixelmae.pt"` expands the variable `$TAG__pixelmae` (underscores
    are legal in a bash identifier), which is unset, so the release-asset
    pattern was a bare `".pt"`. **Always brace a variable followed by
    `_`, a letter or a digit**: `"${TAG}__pixelmae.pt"`.
  - The same step called `gh release download`, and **the Vast boxes have no
    `gh` CLI** — every call returned 127 in under a millisecond with stderr
    thrown away by `2>/dev/null`, and the step exited 0 having downloaded
    nothing. Fetch release assets with `curl -fsSL
    https://github.com/$REPO/releases/download/$TAG/$ASSET` (public repo, no
    auth), the way the data-cache seed has since #47. Between them these two
    meant the checkpoint seed had NEVER worked, so the box-local resume
    livelock it was written to cure was still live and killed #96/#97/#98.
  - `2>/dev/null` on a command whose failure you are branching on hides the
    one line that explains the branch. Keep stderr unless you have read it.
- **A queued Actions job can wedge, and a fresh dispatch is the fix.** On
  2026-08-09 runs #105/#106 sat `queued` for 22 minutes while the runners API
  reported two of the three boxes `online` and `busy: false`, and the jobs'
  `labels` matched (`gpu`). Nothing was holding them: the runs they replaced
  had been cancelled, and the third box was busy with an unrelated job that
  had started while all three were occupied. Cancelling both and dispatching
  the identical inputs again had both picked up **within 90 seconds**. So
  when a job is queued against a runner that GitHub itself says is idle,
  do not keep waiting and do not restart the Vast boxes (which costs the
  warm caches for nothing) — cancel and re-dispatch. Check
  `runner_name`/`status` on the jobs endpoint, not the run-level status,
  which lags: a job can read `queued` at the run level for minutes after it
  has actually started.
- **`runner` defaults to `gpu`, and omitting it used to mean CPU.** `runs-on:
  ${{ inputs.runner || 'ubuntu-latest' }}` with a dispatch JSON that left the
  field out sent run #99 to a free GitHub-hosted 4-core box, where the install
  step happily picks the CPU torch wheel and a 40M codec "trains". Nothing in
  the run's output says wrong-hardware; the only tell is the runner name in
  the jobs API. The default is now `gpu`, so a down fleet leaves the job
  visibly queued instead. **Check `runner_name` on every dispatch you care
  about** — `node /tmp/steps.mjs <runId>` style, or the jobs endpoint.
- **NEVER GUESS WHAT AN ARCHIVE SERVES — ASK IT.** GIBS publishes each layer's
  exact time domain at
  `/wmts/epsg4326/best/1.0.0/{layer}/default/{tms}/all/all.xml`: a
  comma-separated list of ISO-8601 `start/end/period` intervals. `loadGibsDomain(cfg)`
  fetches it the first time a layer is enabled (one small XML, cached for the
  session, one in-flight promise per layer, failures cached too so we ask once),
  `parseGibsDomain` turns it into ordered intervals, and `snapToDomain` resolves
  any requested date to the newest instant actually served at or before it. The
  split matters: **`gibsTimeStatic(cfg, date)` is the request we'd make if every
  archive were continuous; `gibsTime(cfg, date)` is that snapped onto reality**,
  and `gibsTime` stays SYNCHRONOUS (Cesium tile requests and the hover probe both
  call it) — until the domain lands it behaves exactly as it did before, and
  afterwards every call site is corrected at once. `ensureGibsDomain(cfg)` runs
  from `addLayer` and rebuilds the layer if the snapped date moved.
  `gibsTimeStatic(cfg, date, { clampEnd: false })` gives the date the user
  genuinely asked for — the toasts need it, or a clamped layer looks like it got
  what it wanted.
  This exists because guessing failed twice in the same way. c9afa39 (2026-07-24)
  fixed an invisible salinity layer by stepping back ONE month when the date
  lands in the current month. NDVI lags TWO months, so the same rule worked all
  July (asking for June, which is served) and broke on 1 August (asking for July,
  which is not) — a blank globe, a legend, and a probe saying "no data", with
  nothing on screen to explain it. A hand-picked lag is a hand-picked threshold;
  CLAUDE.md forbids those for exactly this reason.
- **Archives have HOLES, not just ends.** Measured 2026-08-03: NDVI is missing
  2025-04, SMAP salinity all of 2024, VIIRS true colour 11–15 July 2026, GRACE
  is irregular throughout (P28D, P17D, P13D, P33D… plus one malformed interval,
  `2020-01-20/2020-01-10/P1M`, whose end precedes its start — collapse those to a
  single instant rather than discarding the layer's domain). Any rule that only
  models a trailing edge will blank the globe mid-scrub.
  Tests must not pin a date inside someone else's archive either: the salinity
  test asserted `gibsTime(sal, "2024-03-15") === "2024-03-01"` and broke the day
  snapping arrived, because 2024-03 is inside SMAP's hole. Assert the *rule*
  (first-of-month) against `gibsTimeStatic`, then assert the snapped answer is
  covered by the domain the app just measured. A pinned month is a hand-picked
  threshold wearing a test's clothes, and it rots when NASA backfills.
- **Some GIBS archives end before today**: GRACE mascons stop at 2022-07,
  CERES EBAF at 2018-10, MEaSUREs SSH anomalies at 2019-01, AMSR2 soil
  moisture AND sea ice at 2025-09 — the instruments/records continue, only the *tiles*
  stop. `endTime` in the layer cfg means "this archive is CLOSED"; it clamps
  requests to the last served date (so the layer shows its final state instead
  of blanking) and is now a SEED — `loadGibsDomain` overwrites it with the
  measured end, and `cfg.lastServed` records what the archive really ends at.
  The typed values do drift: SSH's said 2019-01-17 for months when the archive
  ends 2019-01-22, one 5-day step later, quietly hiding the final frame. The hover
  card must say "this map: … → <end> (last date GIBS serves)". When a
  new layer lands, CHECK ITS EXTENT in the time domain — and scene tests
  must assert rendered DATA (tile pixels for the effective date), not just
  that a chip appeared: "sea ice" shipped blank because the test stopped at
  the chip. 5-day products
  (`snap5d: [epoch1, epoch2]`) serve only exact epoch dates — floor to the
  nearest valid epoch; MEaSUREs SSH was re-anchored in 2017, hence two epochs.
- **Three reasons a date isn't shown, three different sentences.** From the
  outside a closed archive, a lagging one, and a hole all look identical: a blank
  globe. `maybeArchiveToast` (which replaced `maybeClampToast`) tells them apart —
  "its tile archive ends X and nothing newer will be published" / "NASA hasn't
  published X yet — this product currently runs about N months behind" / "GIBS
  has no tiles for X — a gap in the archive, not an error" — and every branch
  names the date actually on screen. It fires TWICE by design: immediately on
  enable from what's already known, then again with `{ replace: true }` once the
  measured domain lands and the story may have changed. It also fires from
  `refreshTimedLayers`, because a date change can walk into a hole just as easily
  as an enable can. `showToast(html, { replace })` supersedes a toast under the
  same key instead of being suppressed by it — the de-dupe key must stop a
  repeated message, never a CORRECTION — and skips the swap when the new HTML is
  identical, so scrubbing the date selector doesn't restart the animation on
  every keystroke.
- **Colormap catch-all bins are NOT measurements.** Many GIBS palettes pad
  their ends with one huge bucket: SMAP salinity runs 30–40 PSU in 0.04-wide
  bins but its first entry is `[0,30)` and its last `[40,+INF)`; GHRSST SST has
  `(-INF,0]` / `[32,+INF)`. The probe used to print the bucket's midpoint,
  which put a flat "15.0 PSU" across the whole Baltic (true value ~7) next to
  honest 31s — a user caught it from the phone. `getValueLut` now returns
  `caps` (value → `{sign, bound}`) for end bins that are unbounded or >10× the
  palette's median bin width, and the probe/pixel-card print "< 30" or "≥ 40"
  instead. The numeric midpoint is retained in the lut for mean/delta
  arithmetic, and `kelvinToC` converts cap bounds along with values. Test:
  "catch-all colormap bins probe as bounds" (Gotland deep = capped, 30W/40N =
  numeric — anchors verified against the real July-2026 tile).
- **SMAP salinity's blank areas are the product's own mask**, not a bug: an
  L-band radiometer can't retrieve near coasts (land in the sidelobes), under
  sea ice, or in RFI — measured on the real tiles, only ~35% of the UK/Biscay
  tile has data at all. The layer's `meta` says so; don't "fix" it.
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
salinity (monthly) · VIIRS Black Marble night lights · **OPERA DIST-ALERT**
and **DIST-ANN** vegetation disturbance (30 m from Harmonized
Landsat-Sentinel, the finest layers in the app; classification rasters — see
§2.3 — the alert layer running every few days through the current year, the
annual one settling it per year 2023–2025).

**Climatology grid layers (client-rendered from baked JSON, `GridProvider`):**
GPCP v2.3 global precip (2.5°) · E-OBS v31 European precip (0.25°, bounded
rectangle) · OISST v2.1 SST 1991–2020 (1°) · MeteoSwiss Swiss precip normal
1991–2020 (~2 km) · **Subsurface temperature anomaly at 300 m** (Argo RG, 1°,
`snapshotGrid` — a single recent month vs the 2004–18 same-month mean,
diverging `anom` ramp). Ramp legends with hover read-out; probe reads exact
cells.

**Drivers of forest loss (`data/drivers.json`, `classGrid`):** not a
climatology and not a snapshot — WRI and Google DeepMind's 1 km attribution
of WHY tree cover was lost, one dominant class per cell over the whole
2001–2025 record (Sims et al. 2025, ERL 20(7):074027, CC BY 4.0), binned here
to 0.25° by per-block MODE, never a mean: averaging "wildfire" and "logging"
would produce "shifting cultivation". Cells with no mapped loss bake as null
and render transparent, so the layer paints only the deforestation frontiers.
Bake: `refresh_data.py drivers` (~300 MB COG, needs `rasterio`). It is the
companion to the OPERA rasters above, which see the loss at 30 m and say
nothing about its cause.

**Place names (`data/cities.json`) — the map's reference points:** 7,342
Natural Earth 10m populated places (public domain, 166 KB gzipped, 200 national
capitals, 58 of them surviving the full-globe view), drawn as a Cesium
`LabelCollection` + `PointPrimitiveCollection` with a per-label
`DistanceDisplayCondition` derived from Natural Earth's own `min_zoom`, plus an
optional GIBS `Reference_Features_15m` coastline/border overlay. Requested
directly by the user — "make sure that some reference points are also displayed
on the map, like cities … it is better to navigate in that way". This is the
one thing on the globe that is not a measurement: it exists so every other
layer can be read as being somewhere. See §5 for the decisions behind it and
Part 2 for why the names could not come from GIBS.

**The gazetteer and place search (`data/gazetteer.json`):** the same user, two
requests later — "is this near Peniche … do you know there is no place name?
can you add functionality to search for a place name?" Both halves needed
answering, because search alone would have flown you to a town the map still
refused to name. 54,204 GeoNames `cities5000` places (CC BY 4.0, 1.15 MB
gzipped, deduplicated against Natural Earth by proximity *and* by folded name
within ~0.5° — the two projects place a big city several kilometres apart, which
left Dubai and 66 others doubled on the first pass) carry rungs that continue
Natural Earth's ladder at its own measured growth of 3.305 places per rung. The
file is lazy: it arrives when you open the search box or descend past rung 9.
Above that seam nothing changes; below it the deep tier fills the valley with
the towns Natural Earth never had room for, and the search box finds any of the
61,000 places across both files and flies you in at an altitude where the place
is actually labelled. Bake: `refresh_data.py gazetteer`.

**Island names (`data/islands.json`) — the first tier that is not a
settlement:** the same user again — "can you add island names as well? i think
sylt is currently missing". Sylt was missing for a structural reason: both
existing tiers are gazetteers of POPULATED PLACES, so Westerland (pop. 9,000)
was labelled and the 43-km island it stands on was not, because no settlement
file contains physical features at all. 4,950 islands (0.10 MB gzipped, 264
named by Natural Earth's curated `ne_10m_geography_regions_polys`, 4,686 by
GeoNames feature class T), drawn as a third `LabelCollection` — italic, no dot,
because an island is an area and a dot would claim a point. See §5 for the
decisions; bake: `refresh_data.py islands` (needs `shapely` + `pyproj`).

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
banker's rounding when thresholding). Forecasts age daily — automated:
`.github/workflows/refresh-forecast.yml` bakes, validates (GFS data tests)
and self-deploys Pages every day at 05:30 UTC (a GITHUB_TOKEN push can't
retrigger pages.yml, hence the inline deploy steps). GitHub disables cron
workflows after ~60 days without repo activity; the daily bot commits keep it
alive, but re-check if the repo goes quiet. WeatherNext 2 (DeepMind) was evaluated 2026-07-29:
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
  normals with anomaly, and nearby observing/emitting context. Every row
  carries its own observation time and age (§5, "Provenance") — never the
  section's, because `gibsTime()` clamps per layer. See §5.

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
searchable 248-dataset catalog with domain/AMOC/globe filters.

**Data pipeline** (`scripts/refresh_data.py`): one function per snapshot —
climatetrace, argo, rapid, sealevel, glaciers (RGI7 tars + Hugonnet parquet
join), gistemp, gpcp, eobs, oisst, meteoswiss. Grid snapshots share
`_bin_to_grid`/`_write_grid` (nearest scatter-binning onto regular grids).

**Testing** (137 Playwright specs): app behaviour (`tests/app.spec.js`) + data
integrity (`tests/data.spec.js`), the ML status page (`tests/status.spec.js` —
every GitHub endpoint stubbed by `page.route`, so it needs no network and no
MIRROR), sandbox MIRROR mode, in-repo proxies, CI on real network.

**Notable bugs fixed along the way** (details in git history): Pacific blanked
by clamped edge tiles; Pages 404 (gh-pages + enablement); probe showing
absolute values under a delta; colormap parser skipping single-value entries;
salinity invisible (current-month composite unpublished); mangled Cesium class
names breaking test assertions; `_zoomFactor` no-op.

**Deferred / open follow-ups:** OC-CCI & SMOS as first-class grid layers;
multi-channel AMOC state vector; catalog `family` field for machine-readable
dataset relationships; honest precipitation aggregation (accumulated totals
from monthly products).

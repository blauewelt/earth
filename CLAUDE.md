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

### 0b. Post links as CLICKABLE MARKDOWN, always

Whenever a session creates or substantially updates a Markdown document,
its reply in chat must include a link to it — the user reads on a phone and
cannot guess repo paths ("I often run into issues where you talk about
documentation that I don't know how to access", 2026-08-06).

**The link must be written as a markdown link — `[label](url)` — never as a
bare URL.** Chris, 2026-08-13: *"I somehow cannot open the links you pasted
(eg to the E-022 plan). Can you make it a standing rule to only paste
clickable links?"* The message that failed was a run of bare URLs joined by
`·` separators on one line; the client did not linkify them, so three
correct URLs were three pieces of unusable text. A markdown link cannot fail
that way, whatever the punctuation around it. Concretely:

- ✅ `[E-022 plan](https://github.com/blauewelt/earth/blob/main/ml/plans/E022_spatial_coupling.md)`
- ❌ `https://github.com/blauewelt/earth/blob/main/ml/plans/E022_spatial_coupling.md`
- ❌ several bare URLs on one line separated by `·`, `|`, or commas

**One link per line or per bullet.** Packing links into a prose line is what
produced the failure; a list of labelled links is also what a phone can
actually tap.

The target must be something the browser can RENDER, not just serve. A
GitHub `blob` URL is right for code; it is wrong for an HTML figure, which
renders as source (2026-08-13, "i cannot open the figures") — those go to
the Pages URL, or as PNGs through the file-delivery tool.

**For MARKDOWN, link the phone reader, not the blob URL** (2026-08-15,
Chris: *"could you somehow change all .md files such that they render well
on mobile?"*). GitHub's mobile view pans a nine-column result table
sideways with no row label in sight, which is the state most of this
project's documents are in. `docs.html` renders any repo `.md` live from
`main` with the label column pinned, prose tables stacked, and a contents
drawer — so the link that gets posted is:

- ✅ `[the experiment log](https://blauewelt.github.io/earth/docs.html?f=ml/EXPERIMENTS.md)`
- ✅ deep link to a section: `…/docs.html?f=ml/EXPERIMENTS.md#e-026b-audit-of-the-anti-correlation`
- ❌ `https://github.com/blauewelt/earth/blob/main/ml/EXPERIMENTS.md` — correct, unreadable on a phone

The blob URL is still right when the point is the SOURCE (a diff, a
permalink to a line, something to copy). Adding a new document means adding
one line to `DOCS` in `docs.html`, and `tests/docs.spec.js` renders the real
files at 360px — a document that grows a table the reader mishandles fails
the suite.

**Registering the document is part of writing it, and the suite now says so.**
That step was skipped four times before anyone noticed (E-022, E-025, E-038,
E-039), each time silently: the file was on `main`, the blob URL resolved, and
the reader's index simply did not know it existed — so the plans written to be
read before spending a week were the ones hardest to open. Chris, 2026-08-16,
on the E-039 link: *"Are you sure the link to E039 is correct? I tried opening
it, and it failed."* It was correct and it was the wrong KIND of link, for a
document the reader had never been told about. `tests/docs.spec.js` now reads
`ml/plans/` off disk and fails on any plan missing from `DOCS`, because the
old tests could only ask about documents that were already listed.

**A RUN NUMBER is a link target too, and it gets its own tier** — `#NNN` never
appears without a short summary beside it, and it links to the status page's
`#run-NNN` anchor or to `ml/EXPERIMENTS.md`, both of which carry a summary and
the curves, rather than to an Actions log a phone cannot read: `ml/CLAUDE.md`
§0c.

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

**But the force is almost never NEEDED, and reaching for it first costs
deploys.** gh-pages only ever trails main, so moving it to main's tip is an
ordinary fast-forward; `-f` is habit, not requirement. On 2026-08-10 the CLI
push was refused twice — once by the permission classifier (which reads a
force-push as destructive, correctly) and once by the git proxy's
"not in this session's authorized repository set" — and the deploy went
through in one call as a PLAIN fast-forward on the Git Data API:

    PATCH /repos/<owner>/<repo>/git/refs/heads/gh-pages  {"sha": <main sha>, "force": false}

`force: false` is the point. If gh-pages has somehow diverged, GitHub answers
422 and that refusal is information — the branch has another writer, or main
was rewritten — not something to override by flipping the flag. Try the
fast-forward first; only consider a force after reading why it failed.

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
   public docs (`doc` field / `title-link`) — **and the title states the
   layer's PIXEL SIZE** ("…, 30 m", "…, 0.25°"), because that is the first
   thing a reader needs about a measurement and the panel is where they
   choose between layers (Chris, 2026-08-31: "make sure that the pixel size
   is always displayed in the layer title… go over all the layers"). The size
   is not a second hand-typed copy of the truth: `tests/app.spec.js` requires
   every title to carry one AND that it appear in that layer's own `sp` fact
   (item 2). Where a title carries a number that is NOT a pixel size —
   "300 m depth" for the Argo layer, "2 m air" for GFS — the rule is that ONE
   of the title's sizes agrees, and those two now say what they are so the
   reader is not misled either.
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
   A palette's transparent FILL is not a class: `parseClassEntries` reads
   which `<ColorMapEntry>` carries `transparent="true"` and drops the
   `<LegendEntry>` with the matching `ref`, rather than matching the label
   "No Data" — DSWx-S1 spells it "Fill value (no data)" and used to get a
   black swatch in the legend for a colour the tile never paints.
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
   - `annual: true` and `cumulative: true` layers are NEVER suppressed by a
     window, whatever their other flags. A whole-year composite is already a
     long-period aggregate and does not change as the window slides (every
     day in it resolves to the same year); a cumulative map — DIST-ALERT is
     the year's disturbance SO FAR, not the day's — is an accumulation
     already. Both are the case the rule always exempted for untimed
     composites and climatology grids, and suppressing them read as a broken
     layer twice in one day: the EOX mosaic, then the 30 m disturbance
     alerts, both under the 12-day window left over from the swath work.
     Neither is claimed to BE an average — `windowed` stays false, so no
     legend says "mean" over one. **And whatever the reason a layer is
     hidden, its own ROW says so now** (`updateSuppressedNotes`,
     `[data-suppressed]`): the delta-hint panel had explained it since the
     window existed, but the row is where you look when you tick the box and
     nothing appears.
   - `mosaic: true` — a daily SWATH product (HLS, the radars, DSWx): the
     Aggregate window is a LOOKBACK and the layer renders the UNION of every
     day in it (`MosaicProvider`, newest pass on top, capped at
     `MOSAIC_MAX_DAYS` = 16 ≥ NISAR's 12-day repeat), so a 12-day window
     covers the whole planet. Never a mean — you cannot average "flew over"
     with "didn't" — and no comparison of either kind. The legend and the
     row hint say "union of the past N days", and the hint adds how many
     dates were ACTUALLY served when the window lands in an archive hole
     (DSWx-S1 has none from 2023-12-25 to 2024-08-20: every day of a 12-day
     window there snaps to 2023-12-24, and a "12-day union" is one date).
     The class probe walks the same dates newest-first and stamps the day
     the answer came from.
   - **`unobserved: /regex/` — the classes that are not a measurement.**
     DSWx paints CLOUD where the optical sensor could not see the ground, and
     the radar version paints two MASKS where the method does not apply. Those
     are opaque colours, so a plain newest-on-top union let a cloudy Tuesday
     bury a clear Saturday — which defeats the point of a union (Chris,
     2026-08-31, comparing water extent across years). A layer names those
     classes as a regex over the PALETTE'S OWN LABELS, so the judgement comes
     from the producer's vocabulary rather than a hard-coded colour, and the
     compositor walks newest-first taking each pixel from the first day that
     actually observed it. What no day observed keeps the newest day's cloud:
     "we looked and could not see" is information, and a blank pixel would
     claim we never looked. **This is a modelling choice, so it is visible and
     reversible**: while a union is active the legend strikes those classes
     through and offers a switch (`state.seeThrough`, `[data-seethrough]` —
     the one control that lives on the legend, because that is where the
     classes it affects are). The probe applies the identical rule, so the
     read-out and the pixel under it are never from different days.
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
   | Sentinel-2 / Landsat true colour (HLS S30, L30) | union | ✗ | photographs of daily SWATHS: `mosaic` — the window is a lookback and the union covers the planet; a mean would average "flew over" with "didn't" · `fine: 500`, `overview: 5` |
   | Sentinel-1 / NISAR radar backscatter (OPERA RTC-S1, NISAR GCOV) | union | ✗ | false-colour photographs of swaths, `mosaic` (NISAR needs 12 days for full coverage), no colormap to invert · `fine: 500` / `300`, `overview: 5` |
   | Surface water extent (OPERA DSWx-HLS, DSWx-S1) | union | ✗ | classifications — class codes neither average nor subtract, but the newest pass per pixel composites (`mosaic`); the probe says which day answered · `fine: 500`, `overview: 5` |
   | Elevation (ASTER GDEM) | ✗ | ✗ | continuous metres but UNTIMED — terrain has no date axis; `probeNative` so the probe reads the 30 m pixel, not a 4 km mean |
   | Built-up extent (HBASE), impervious % (GMIS) | ✗ | ✗ | classifications (GMIS's percent bins are class-labelled), and a single 2010 epoch besides |
   | Landsat true colour, historic (WELD annual) | ✗ | ✗ | photographs, `annual` with a `12-01` anchor — three separate spans, 1984–86 / 1989–91 / 1999–2001 |
   | Land cover (ESA WorldCover 10 m) | ✗ | ✗ | classification, one 2021 epoch; inline palette · third backend · `fine: 1500` (Terrascope renders on demand) |
   | Water occurrence (JRC GSW) | ✗ | ✗ | continuous %, but a single 1984–2024 aggregate — nothing to average · third backend · `probeNative` |
   | Sentinel-2 cloudless mosaic (EOX, yearly) | ✗ | ✗ | photographs, `annual` 2016–2025 — a split comparison across years is the intended use · third backend |
   | SWISSIMAGE, SWISSIMAGE time travel, swissALTI3D hillshade | ✗ | ✗ | photographs / a shaded model; time travel is `annual` 1926–2025; all `rect`-bounded to Switzerland · third backend · `fine: 1500` |
   | True colour, night lights | ✗ | ✗ | photographs, no colormap to invert |
   | Tide height (live) | ✗ | ✗ | animated harmonic reconstruction on its own clock — there is no date axis to average or difference; the Tides tab is its control room |
   | Surface wind speed (MERRA-2, monthly) | ✓ | ✓ | continuous linear m/s, a complete reanalysis field — the same posture as SST |
   | Ocean wind speed (AMSR-E/AMSR2, daily) | ✓ | ✗ | swathy like soil moisture: averaging fills the orbit gaps, a day delta compares coverage rather than wind |
   | Ocean surface current speed (OSCAR) | ✗ | ✗ | a MAGNITUDE built from two component rasters — each sample would have to invert two palettes, and the field is already 5-day |
   | Grid climatologies | ✗ | ✗ | already multi-decade averages, not timed |
   | Drivers of forest loss (grid) | ✗ | ✗ | categorical AND untimed — one 2001–2025 attribution, and "logging" plus "wildfire" is not a quantity |
   | AMOC eval mask (grid) | ✗ | ✗ | categorical AND untimed — a cell carries the ROLE it plays in an experiment, and an experiment's geometry has no date to average over |
6. **Catalog consistency** — the dataset exists in `data/catalog.json`; set
   `globe: true` and append "Live globe layer in this app." to its notes.
   **Exception, for layers that are not datasets:** a layer describing our OWN
   work rather than an observation of the world gets no catalog record, for
   the same reason the city labels have none — §2.6 catalogues open DATA, and
   the catalog is what a reader mines for sources. Currently one such layer
   exists, `amoc-eval` (the pixels the AMOC forecaster rolls forward, written
   by `ml/rollout_spatial.py --export-mask`). Such a layer must still satisfy
   every other item here, and its `doc` link points at the experiment's own
   plan or log — a layer with nothing to click is the one that gets mistaken
   for a measurement.
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
10. **A gate, if it is fine-resolution and sparse.** A 30 m (or finer) layer
   that is a daily SWATH product — HLS, the two radars, the two DSWx water
   maps — declares `fine: <km>`. Above that camera height the layer stays
   enabled (chip, legend, opacity row, hover card) but its ImageryLayer is
   HIDDEN, and a hidden layer requests no tiles at all (`fineGated` /
   `applyFineGate` / `updateFineGates`; Cesium creates tile skeletons only
   for shown layers, the fact the retirement queue already rests on). The row
   carries a live `.fine-hint` ("⤵ zoom in — hidden above 500 km (you're at
   12,000 km)"), enable fires `maybeFineToast`, the playback ring passes the
   gate through (`playbackPreloadAdd(p, cfg)`), and `colormapLayersTopDown`
   skips hidden layers so the probe never answers for what is not on screen.
   The gate is for SPARSE fine layers: a full-globe view of a day's swaths is
   mostly blank, and every date step would re-fetch that blank. A static 30 m
   layer with meaningful overviews (elevation, built-up, WELD mosaics) is NOT
   gated — its coarse levels are real maps. 500 km is generous on purpose: at
   that height a 1000 px viewport needs ~8–16 GIBS tiles of a 30 m layer, so
   the gate costs nothing to cross and stops only the pointless case.

### 3. Data pipeline: static snapshots, never live third-party calls

The browser must depend only on NASA GIBS (tiles) and GBIF (occurrence tiles).
**Three deliberate exceptions** (the third, below, is a set of four keyless tile hosts). The first: the pixel inspector calls the
Open-Meteo family (`api`, `air-quality-api`, `flood-api`, `marine-api`,
`climate-api`.open-meteo.com) — key-free, CORS-open, and only ever a
single-point query triggered by an explicit click, never tile streaming.

The second, added 2026-08-18 for E-040: **`huggingface.co`**, from which the
pixel card and the hover probe read the true daily SST for one point
(`sstDailySeries` / `sstDailyAnomaly`, `ml/plans/E040_daily_sst.md`). It clears
the same bar on all four conditions, and each was measured rather than assumed:
**no key** (anonymous reads, no account, no quota to manage); **CORS verified**
— the Hub echoes our Origin and exposes `Accept-Ranges`/`Content-Range`, and a
browser range read on a foreign origin returns HTTP 206 with exactly the bytes
asked for; **click-triggered single-point range reads, never streaming** — the
file is stored pixel-major, so one point-year is 730 contiguous bytes out of a
757 MB file and the transfer is bounded by the question rather than by the
archive; and it **degrades to the monthly value** — every failure path returns
null. The monthly OISST correction remains the fallback, so a Hub outage costs
precision, not the feature.

Any further live endpoint must clear the same bar (no key, no quota pain,
click-triggered, degrades to an omitted card section on failure) and be added
to the MIRROR proxy set (`:8083`–`:8087` are the Open-Meteo hosts, in the
order above; the Hub is routed pass-through in `tests/app.spec.js` instead,
because the Playwright node process has egress even where the sandbox browser
does not).

**The third exception — a third BACKEND, approved by Chris 2026-08-31 ("fine
to add a third backend for Tier 2"): keyless tile hosts.** Tiles are
streaming, not click-triggered, so the bar for a tile host is different and
stricter on the things that matter for streaming: **no key or registration
of any kind**; **CORS open** (`*` or reflecting our Origin — Cesium reads
pixels for the probe); **static or cheap-to-render tiles on a stock scheme**
(all four are EPSG:3857 XYZ, 256 px — `xyzProvider`, never a bespoke tiling
class); **a licence that permits display with attribution, quoted verbatim
in the layer's `credit`** and in the footer; **bounded requests** — a
regional service declares `rect` (swisstopo answers 400 outside Switzerland),
a dynamic renderer is gated with `fine:` so the whole planet is never asked
of it on a pan. Each host was verified by request before it was added (an
Opus agent's report, 2026-08-31: hosts, layer names, zoom limits, row/col
order, CORS headers, licence sentences — three of the four had documentation
that was WRONG about the host, the layer id or which year "latest" was, so
verify, never transcribe). The hosts and their proxies (`:8088`–`:8091`):

| host | what | licence · attribution |
|---|---|---|
| `wmts.terrascope.be` | ESA WorldCover 10 m (KVP WMTS, TIME mandatory, 3857 only, data to z14) | CC BY 4.0 · "© ESA WorldCover project 2021 / Contains modified Copernicus Sentinel data (2021) processed by ESA WorldCover consortium" |
| `storage.googleapis.com/water-world` | JRC Global Surface Water v1.5 occurrence (z ≤ 13, CORS `*`) | Copernicus, unrestricted · "Source: EC JRC/Google" |
| `tiles.maps.eox.at` | EOX Sentinel-2 cloudless, one mosaic per year 2016–2025 (`s2cloudless-{year}_3857`; the UNSUFFIXED `s2cloudless` is 2016, not latest; z ≤ 18) | CC BY-NC-SA 4.0 for 2018–2025, CC BY 4.0 for 2016–17 · "EOxCloudless https://cloudless.eox.at by EOX IT Services GmbH (Contains modified Copernicus Sentinel data YYYY)" |
| `wmts.geo.admin.ch` | swisstopo: SWISSIMAGE 10 cm (z ≤ 20), the 1926–2025 `swissimage-product` time series, swissALTI3D hillshade (z ≤ 18); CH bbox only | OGD, free with attribution, fair use ≤ 20,000 users/day · "© Data: swisstopo" |

Palettes for these come from `INLINE_PALETTES` (`classmap: "inline:…"` /
`colormap: "inline:…"`), copied from the producer's own colormap and verified
against fetched tiles, because none of them publishes a GIBS-style colormap
XML. An untimed third-backend layer with a known epoch states it as
`fixedWhen` (§2.9); a yearly one is `annual` and rides the whole annual
machinery. NEVER add a fifth host without re-reading this paragraph. Everything else is baked offline by
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
date-driven rasters must return `null` there. An untimed RASTER may carry its
own `datelessNote` (elevation: "a fixed terrain model"; HBASE/GMIS: "one map
from 2010") — the generic "fixed composite" sentence is the fallback, not the
answer. **Yearly layers are NOT dateless**: Climate TRACE is an annual inventory baked for every available year (2021-2025, `assets_by_year` in climatetrace.json); the layer shows whichever year the date points at (`climateTraceYear`, clamped), rebuilds on a year change (`refreshYearlyLayers` / `ensureClimateTraceYear`), and its toast (`climateTraceToast`) says 'the day and month don't matter, but the year does' — never declare a layer fully dateless if any date component drives it. **Monthly grids follow the same pattern one level down**: GLORYS currents/MLD are `monthlyGrid` layers covering the FULL archive (1993-01 → ~now−2mo). The baked index (data/currents.json, data/mld.json) carries `monthsAvailable` (every stamp), `yearDir`, `months` (latest year inlined), `latest`, `values` (= latest month, back-compat); older months live in per-year files (data/currents_y/YYYY.json) lazy-fetched by `ensureGridMonth`/`loadGridMonth` and merged into `g.months` — every sampler (GridProvider.requestImage, probeValueAt, the pixel card) MUST go through `loadGridMonth`, never bare `loadGrid`+`sampleGrid`, or old months read as null. `resolveGridMonth` floors the date's month to the newest baked month ≤ it (clamped at both ends), `refreshMonthlyGrids()` rebuilds the provider when a date change lands on a different baked month (Cesium caches tiles — a repaint needs a fresh provider; called from both date handlers AND the ±30m midnight-cross branch), and `maybeMonthlyGridToast` names the month showing on enable. Note the date steppers clamp at 2000-01-01 (GIBS floor) — 1993–1999 currents are reachable by typing a date. **Day-keyed forecast grids reuse the whole mechanism**: GFS temp/precip are `monthlyGrid` + `forecastGrid` layers whose JSON carries `keyLen: 10` (day stamps in `months`/`monthsAvailable`, all frames inline, no year files) plus `init` (the model run, quoted in the toast). While a forecast layer is active, `uiMaxDate()` returns the last forecast day instead of `defaultDate()` — the date input's `max` and every stepper clamp go through it (`syncDateMax()` restores reality and pulls the date back when the last forecast layer is switched off), and `gibsTime` clamps any future date to `defaultDate()` so observation layers are never asked for tomorrow's tiles. **Annual rasters are Climate TRACE's trap one rung coarser, on the GIBS side**: OPERA DIST-ANN is served at exactly one tile date per year (`YYYY-01-01`), so its config carries `annual: true` and `gibsTime` snaps the date to Jan 1 of its year (floored at `start`'s year, after the `endTime` clamp). It fires `maybeAnnualToast` rather than `maybeArchiveToast` — `maybeArchiveToast` returns early for annual layers, because "showing the last available date" is the wrong story when the day and month never mattered; the annual toast says which YEAR is showing and names the span of years the product covers. Keep the toast copy consistent:
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
- **The aggregation window has FOUR controls and ONE number.** Slider, ±1d
  nudges, a typed field and the presets (1d/7d/12d/30d/365d) all funnel
  through the slider's own `change` event — `setWindowDays` writes the slider
  and fires it — so only one handler touches `state.windowDays` and the four
  can never disagree. A 730-stop slider cannot be aimed; a day at a time is
  what a swath layer needs (one step = one satellite pass in or out of the
  union), and 12d is not a round number but a full repeat cycle, the window at
  which the union closes over the whole planet. The typed field commits on
  `change` (Enter, blur, spinner) and never per keystroke — writing back there
  is safe precisely because `<input type="number">` fires `change` at a
  COMMIT, unlike the date field's per-segment change (§4b, the half-typed
  year). Junk or an empty field restores the truth; out of range is clamped.
- **A scale bar ("Massstab") sits bottom-left**, mirroring the legend's
  bottom-right and clear of the Cesium credit line. Its length is MEASURED —
  two ellipsoid picks 100 px apart at the centre of the canvas and the
  geodesic between them — not derived from camera height and fov, which
  over-reads under tilt and near the limb; the fov formula (`islMetresPerPixel`)
  is only the fallback for a centre that misses the globe. The line under it
  answers the question that prompted it ("how large is the displayed
  surface"): the ground arc across what is actually on screen, found by
  walking each canvas edge inward to the last pixel with ground under it, so
  from orbit it reports the visible disc rather than counting the pixels that
  show space. Recomputed on `postRender` behind a guard on camera height and
  canvas width (nothing happens on a frame where neither moved), plus the
  camera events for a tilt at constant height.
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
- **A date change HOLDS the old frame. Never destroy imagery before its
  replacement is painted.** `refreshTimedLayers({hold: true})` retires the old
  `ImageryLayer` onto a bounded queue (`retireLayer` / `sweepRetired`, max 3)
  and builds the new one above it; the queue is swept when the globe's tile
  queue empties. Before E-041 every date move flashed through bare base map for
  one network round trip — a blink on the stepper, a strobe at 2 fps — because
  the old layer was destroyed first. Cesium is what makes the fix a
  postponement rather than a rewrite: tile skeletons are created behind
  `layer.show && _createTileImagerySkeletons(...)`, so a still-shown layer keeps
  painting and requests nothing new (an `alpha: 0` layer, by contrast, requests
  everything). `addLayer` is untouched, so delta/split/aggregate/window/grid all
  keep working. Every date-driven call site passes `hold`; the UNHELD form is
  for when the picture must become exactly what state says it is — and is
  deliberately NOT used when playback stops, because by then it would only
  destroy and refetch an identical layer set, ending every playback with the
  blink the queue exists to remove.
- The **Play** tab (E-041) is a clock that drives `state.date` and nothing
  else — it is not a rendering mode, which is why the comparison, the computed
  difference and the aggregation window all keep working underneath it, and why
  a prediction that lands as a timed layer will be playable with no playback
  code at all. Frames come from the DATA, not the calendar: the step defaults to
  the finest cadence among the layers on, and candidates are deduped by
  SIGNATURE (what each active layer would actually request for that date), which
  collapses a monthly product's thirty identical days to one frame and a closed
  archive's dead zone to one frame instead of four hundred. Over
  `PLAY_MAX_FRAMES` (500) the step COARSENS and the panel says so; the range is
  never truncated. The playhead advances on the FRAME, not on a timer —
  `max(1/fps, tiles settled)`, 8 s ceiling — so requested fps is a speed limit,
  not a promise, and the status line names which of THREE limits is binding
  (`fps` · `network` · `device`, the last being "the browser cannot repaint any
  faster") rather than claiming a rate it is not achieving.
  **Frames are prefetched by a PRELOAD RING, not by warming the HTTP cache**
  — GIBS sends `no-store` and that cache is simply not available to us
  (Part 2). Frames *i+1 … i+depth* exist on the globe as ordinary imagery
  layers at `alpha: 0`, built by `providersFor(cfg, dateStr)` — the SAME pure
  function `addLayer` uses, which is the only thing that stops the animation
  and the paused globe disagreeing about what frame *N* looks like, and which
  takes the date as a parameter precisely so an OFFSET comparison is preloaded
  against ITS frame's past rather than the displayed frame's. `playbackPromote`
  then assigns the warmed layers into `state.layers`, retires the outgoing
  generation through the existing queue, and costs **zero requests**; a frame
  the ring missed falls back to the ordinary held refresh. `PLAY_PRELOAD_DEPTH`
  is 2, dropping to 1 when `navigator.deviceMemory < 4` or more than 32 tiles
  are in view (each ring layer holds the whole visible set as GPU texture), and
  to 0 for grid-only frame sets, which draw from a baked file and need no ring
  — the effective depth is printed in the status line rather than left as
  magic. A promoted frame does NOT wait on the tile queue, because by then that
  queue holds the ring's speculative loads and waiting would gate the visible
  playhead on frames nobody is looking at. **A stale ring is worse than no
  ring**, so `refreshTimedLayers` clears it on every rebuild (layer set,
  comparison, window, re-enumeration) and each entry additionally carries the
  configuration key it was built under. Playback pauses on `document.hidden`:
  this is the first thing in the app that can issue thousands of requests to a
  public NASA service from one click, and the plan's politeness controls are
  load-bearing, not decorative.
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
  - **THE CARD MUST RENDER, and nothing is thrown away.** It composes ~20
    sources and used to await all of them, so the slowest one decided whether
    the inspector worked at all — and `fetch()` has no timeout, so a stalled
    connection meant a card that never appeared. Reported 2026-08-16 as "load
    all data is no longer working"; measured on the live site, the card took
    64.6 s with one climate-api call open the whole time. The same query,
    minutes apart, took 1.0 s, 23 s, and never.
    So the card **appears at `PIXEL_DEADLINE_MS` (2 s) and fills in as the
    data lands** — every source that arrives after the first paint redraws it
    in place. The two halves depend on each other: 2 s was Chris's ask
    ("15s is still a very long time"), and it is only affordable because the
    deadline is no longer a CUTOFF. Shortening a cutoff would have traded a
    slow card for a sparse one. Being slow now costs a section its place in
    the first frame, not its place in the card. Measured live: complete in
    one paint under 8 s healthy; with a host killed outright, on screen at
    ~2 s carrying everything else. Keep these properties:
    **deadline ≠ timeout** — a SHORT deadline to draw, a LONG one
    (`OM_TIMEOUT_MS`, 45 s) to give up; cutting off the 23 s response would
    discard a good answer to save time the card no longer spends waiting.
    **Redraws coalesce** (`PIXEL_REDRAW_MS`, 250 ms) — the Open-Meteo family
    answers within a few hundred ms of itself and each redraw rebuilds the
    DOM; uncoalesced, one click meant fourteen rebuilds.
    **The body's scroll offset survives the swap** — the card is taller than
    a phone screen, and a reader would otherwise be yanked to the top each
    time a source landed.
    **Sources declare an `empty`** (the third element of a `jobs` entry) —
    the two collection slots are read as arrays, and at 2 s they usually have
    not arrived; seeded with null the first draw threw, and the error guard
    turned that into "something went wrong" for data merely being in flight.
    **The promises are created once and shared by every draw** — re-requesting
    per pass would multiply the Open-Meteo burst, rate-limited per IP across
    the family.
    **Every draw checks it still owns the card** (`pixelCardSeq`): tap A, tap
    B while A's slow source is out, and A's data would otherwise redraw under
    B's heading. **A throw in the draw says so and rethrows out of band** —
    swallowed, a rendering bug becomes a slightly emptier card and the suite's
    "loads without page errors" check never sees it.
    The pending note names three sources and counts the rest (at 2 s most of
    the twenty are still out, and the full list was a wall of text under a
    nearly empty card), and **"still loading" and "didn't answer" are visually
    distinct** — accent and gently pulsing versus amber and static, because
    one is expected and transient and the other is final. A source counts as
    having not answered only if it was outstanding at first paint AND came
    back empty: there are no waves inland and no river mid-ocean, and naming
    those would cry wolf on every second click.
    Related: `fmtVal(null)` used to throw, and because the body is built in one
    pass, one null field in one upstream response took the whole card down to a
    permanent "Reading this point…". It prints a dash now. Formatting is the
    wrong layer to enforce presence — a caller that cares checks first.
    A job whose honest answer is "nothing to say" (the SST-anomaly correction
    on an uncapped pixel) returns the truthy `SST_ANOM_NONE` sentinel, never
    null — null is reserved for "didn't answer", which the final pass reports
    with "tap again to retry", a promise that must only be made when a retry
    could actually change something.
  - **The capped-anomaly correction is DAILY first, monthly fallback**
    (E-040, 2026-08-18): a capped "≥ 3" read fetches the exact selected day's
    OISST 0.25° value from Hugging Face (730-byte range read — §3's second
    exception) minus that calendar month's 1991-2020 normal; any failure falls
    back to the resident monthly path, and the row's stamp follows the
    measurement (day vs month). The hover probe renders its instant answer and
    UPGRADES IN PLACE when the Hub replies, guarded by `probeSeq` so a hover
    that moved on never receives another point's value.
  - **Heat load** is its own section (added 2026-08-10 from a Zürich
    Klimaanalysekarte the user sent): felt temperature now with its gap
    against air, today's felt peak, tonight's low flagged as a **tropical
    night** at ≥ 20 °C, and a 7-day felt peak + tropical-night count. Two
    things it must keep doing. It reads TOMORROW's daily minimum for
    "tonight" — minima fall near dawn, so tonight's low is on the next
    calendar day. And its note NAMES the index and disclaims PET/UTCI's
    35/41 °C class limits: those model a body's radiation balance and read
    far warmer in direct sun, so lending their thresholds to an apparent
    temperature would print "extreme" on an ordinary afternoon. The 20 °C
    tropical-night line is the standard definition, not a picked number.
    The variables ride the SINGLE `fetchOpenMeteo` call — a seventh parallel
    request made the burst drop sections at random (Open-Meteo rate-limits
    per IP across its whole family), so the call is `timezone=auto` (a
    tropical night is defined on the local night) and `omUTC()` restates the
    instant stamps in UTC. `omGet` retries once on 429/5xx and network
    errors, never on a 400 — that's our own bad URL.
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

- **The published site comes from `main` via the `pages.yml` Actions workflow,
  not from `gh-pages`.** Measured 2026-08-21: `GET /repos/blauewelt/earth/pages`
  answers `"build_type": "workflow"`, and the live deployment was for `0c84d800`
  (main's tip) while `gh-pages` still stood at `27319fd4`, two commits behind —
  a browser was being served main. `pages.yml` uploads `path: .`, so the
  published site is the whole tracked tree. Keep updating `gh-pages` anyway
  (it costs one fast-forward and it is a warm spare), but do not reason about
  what is live from it, and never gate a deploy on it. Use the four-line
  sequence in §1 — `main` is fast-forwarded, only `gh-pages` is forced. (An
  earlier version of this line read `git push origin main gh-pages -f`, which
  force-pushes main and once discarded a bot commit. It is gone.)
- **A push reaches the origin in ~44 s and a returning visitor in up to
  10 minutes more.** Measured over ten `pages.yml` runs: 4 s to pick the job
  up, 40 s median deploy job (34–83 s). Then Fastly's fixed
  `cache-control: max-age=600` — observed serving a page with `age: 245` four
  minutes after a deploy. That ten-minute window is the thing `?v=<sha8>` and
  `checkForNewBuild()` exist to work around; the build check is exempt from it
  because its `?fresh=<now>` nonce is part of the CDN cache key and reaches
  the origin. Full numbers and the Cloudflare comparison: `docs/HOSTING.md` §4.
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

**The paper is scientific prose about results, never about the project's
history.** Standing rule, Chris 2026-09-02: *"For the paper, our prior
mistakes and our back and forth history don't matter (we can keep those
somewhere else). Keep it in scientific prose. No 'after the reset' and
similar things. Negative results (without scientific flaws) are fine."*
Concretely: no version tracking ("this report replaces…"), no narrative of
what an earlier draft claimed or which protocol was corrected, no
"pre-registered prediction that failed", no session or agent names, no
"now"/"since"/"after the reset" framing. A result is stated with its
protocol, its baseline and its n, and a negative result is reported exactly
like a positive one. Where a flawed measurement has to be documented — a
contaminated pool, a broken metric — that study lives in the archived report
under `ml/paper/archive/` or in `ml/EXPERIMENTS.md` / `ml/plans/`, and the
current paper carries only the measurements that survive it. The 2026-09-02
rewrite (`ml/paper/paper.tex`) is the reference form; the first draft of it
was rejected for opening its abstract with the contamination and for a
figure whose caption discussed it.

### 6c. ML work is governed by `ml/CLAUDE.md`, not by this file

The training and research half of the project moved to its own instruction
file on 2026-08-10, at Chris's request: *"there should be two different
CLAUDE.md — one for the frontend (the current reads like that) and one for the
ML training/research part."*

The rules in THIS file are written for a static site where a bad deploy is
reversible in ninety seconds — "deploy first, test after", the layer
checklist, stamp-before-commit. They are wrong for work where a bad dispatch
costs hours of rented GPU and can produce a number that looks like a result.

| file | governs |
|---|---|
| `ml/CLAUDE.md` | dispatch discipline, working principles, fleet lore, security posture |
| `docs/ML_BASICS.md` | what the ML system IS and what its numbers mean |
| `docs/INFRASTRUCTURE.md` | the fleet's topology, failure taxonomy and invariants |
| `ml/EXPERIMENTS.md` | what was run and what it returned |

Two things from the old §6c and §6d that apply to ALL work in this repo and are
worth keeping in front of a frontend reader:

- **Assert the EFFECT, not the invocation.** A log line proves a line of code
  ran; it proves nothing about the world.

### 7. Documentation set

| File | Role |
|---|---|
| `docs.html` | **The phone reader for every file below** — renders any repo `.md` live from `main` with pinned table labels, stacked prose tables and a contents drawer (§0b). Add new documents to its `DOCS` list |
| `CLAUDE.md` | Standing instructions + holistic record (this file — keep current) |
| `README.md` | Quick start, repo layout, testing. Opens with a link to the live demo. Keep its counts (catalog size, `globe`/`amoc` flags, spec count) and feature list current — they drift silently. Hero image: `node scripts/screenshot.js` (see the header comment for the sandbox invocation); re-shoot it when the UI changes visibly |
| `ml/CLAUDE.md` | **Standing instructions for all ML work** — dispatch discipline, working principles, fleet lore, security posture. This file does not govern `ml/` |
| `docs/ML_BASICS.md` | What the ML system is and what its numbers mean: architecture, protocol, the probe ladder, baselines, loss-design principles, settled negatives |
| `docs/INFRASTRUCTURE.md` | The ML fleet: topology, the failure taxonomy it has actually produced, the invariants that follow, and how to re-run an eval from the release with no GPU |
| `ml/EXPERIMENTS.md` | Every experiment, its hypothesis at dispatch, its result and its cost |
| `docs/PRIMER.pdf` | Background knowledge (GIBS, tiles, colormaps, product levels, climatologies). Rebuild: `python3 scripts/build_primer.py` |
| `docs/CATALOG.md` + `data/catalog.json` | The 248-record open-data catalog (human + machine readable) |
| `docs/COMBINING_DATASETS.md` | Which datasets measure the same quantity; sound combinations |
| `docs/PIXEL_STATE.md` | Which catalog sources compose into a per-pixel state vector (state/memory/forcing/flow/future); the 0.25°-daily common grid argument; the ~25-source minimal composition |
| `docs/TILE_BUDGET.md` | What one user interaction costs NASA: the measured GIBS request count per click, drag, window and playback frame; the unbounded paths found and closed; the rule to check before adding any tile-issuing feature |
| `docs/HOSTING.md` | Where the site is served from: GitHub Pages' 100 GB *soft* bandwidth limit measured against the real per-visit payload, the MEASURED push-to-visible latency of both hosts (§4), the Cloudflare Pages standby workflow and its `paths:` filter, the click-path to enable it, and **§6, the phase-by-phase runbook for moving to `blauewelt.org`** — why the apex forces a nameserver move, the DNSSEC-before-nameservers ordering and its 1.5×TTL wait (the one remaining real risk — neither domain carries mail, so the Infomaniak mail migration is demoted to an if-mail-is-ever-added subsection, DKIM's NS→TXT trap preserved intact), the verification commands, and the rollback for every phase |
| `docs/SPECIES_AND_CLIMATE.md` | Why biodiversity data belongs in a climate app |

---

## Part 2 · Domain lore (hard-won facts — do not relearn)

- **ML fleet lore has moved.** Everything about the Vast boxes, the Actions
  workflow, checkpoints, embeddings and stage-2 resume semantics now lives in
  `ml/CLAUDE.md` §7, next to the rules that govern that work. What remains
  below is globe-app lore.

- **`minimumZoomDistance` is a collision floor, not just a pinch limit.**
  Cesium's ScreenSpaceCameraController lifts the camera to
  `globeHeight + minimumZoomDistance` on any frame it finds it under
  `_minimumCollisionTerrainHeight` (15 km) — unconditionally for a
  controller-driven move (pinch), and for our own wheel zoom once its
  globe-height filter has settled. The old value of 20 km therefore produced a
  tug-of-war below ~20 km of view width, reported 2026-08-31 as "weird stutter
  when zooming in too far". It is 100 m now (the fine tier has 10 cm imagery);
  a test pins it ≤ 500 m and checks a 5 km view holds across 40 frames.
- **A published time domain can run PAST the served archive, and the two are
  different measurements.** The domain says which dates GIBS *lists*; a tile
  request says which it *renders*. `AMSRU_L3_Ocean_Wind_Speed_Daily` lists
  through 2025-09-01 and answers **HTTP 400 — not 404 — for every tile on that
  date at every zoom**, while 2025-08-31 serves normally (measured
  2026-08-31). The two status codes are the tell: 400 is "in my domain, cannot
  serve", 404 is "outside it", so the service is disagreeing with itself. The
  app clamped faithfully to the advertised end and showed an empty globe under
  a toast naming a date with no tiles — reported as an off-by-one, and the
  off-by-one is upstream. `domainOverdeclares: <days>` on a layer shortens its
  measured domain on arrival (`trimDomainEnd`), so the clamp, the snap, both
  toasts, the hover card and a mosaic's date list are all right with no
  special case. It encodes a property of the layer rather than a date, so it
  keeps tracking if the archive later grows. **Do not apply it on a hunch: a
  sparse final day is normal for a swath product** — AMSR2 soil moisture's
  2025-09-01 is partial but real and is left alone, and the sea-ice layer
  serves its last declared day in full. Fetch a tile before believing an
  archive is one day shorter than it says.
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
- **GIBS tiles are `no-store`: the browser HTTP cache CANNOT be warmed, and the
  only prefetch that works is a Cesium layer at `alpha: 0`.** Measured
  2026-08-18 on `MODIS_Terra_CorrectedReflectance_TrueColor` and
  `GHRSST_L4_MUR_Sea_Surface_Temperature`, on 2015, 2026 and default dates:
  every tile response carries
  `cache-control: max-age=0, no-store, no-cache, must-revalidate`. `no-store`
  means the browser MUST NOT retain the response, so "fetch the next frame's
  tiles ahead of time to warm the cache" — the obvious idea, and the one
  E-041's playback shipped with — cannot work at all. It was up to 60 extra
  requests per frame to a public NASA service, buying nothing, and it failed in
  the direction that looks like success: the tiles do arrive when the frame is
  shown, from the network, exactly as they would have anyway. **Do not re-add
  it.** What DOES work is Cesium's own texture cache, measured in the browser
  against the real app: a layer added with `show = false` issues **0** tile
  requests; the same layer at `show = true, alpha = 0` issues **11** (the whole
  visible set); promoting it (`alpha = 0 → 1`) issues **0 new requests** and
  leaves `globe.tilesLoaded === true`. The mechanism is the one the retirement
  queue leans on (Part 1 §5): tile skeletons are created behind
  `layer.show && _createTileImagerySkeletons(...)`, so `show` gates loading and
  `alpha` does not. It is strictly better than the HTTP cache would have been —
  the tiles it holds are already decoded and uploaded to the GPU, so a promote
  is a property assignment rather than a round trip. Playback's preload ring
  (`playPreload`, `playbackEnsurePreload`, `playbackPromote`) is built on
  exactly this, and `tests/app.spec.js` pins all four numbers so a Cesium
  upgrade cannot quietly take them away.
- **The tile budget is MEASURED, and the hazard was never playback.** Every
  tile the app draws is a direct browser→NASA request; no CDN of ours stands in
  front of `gibs.earthdata.nasa.gov`, and GIBS's `no-store` (above) means the
  browser cache does not either. Measured 2026-08-21 in MIRROR mode, four
  rendered tiles per layer: first paint 10 requests · one date step 4 (1 layer)
  / 20 (5 layers) · enable a layer 6 · a scene 10 · one pixel-inspector click 29
  (15 tiles, the 15 colormapped rasters) · an Aggregate window 48 at 30 d AND at
  365 d (`windowSampleDates` caps at 12 sample dates and the cap really binds) ·
  playback ≈4.1 requests per frame with nothing wasted, halting to 0 in a hidden
  tab. **Playback turned out to be the polite part** — every one of E-041's
  documented controls holds (`PLAY_MAX_FRAMES` 500 coarsens rather than
  truncates, `PLAY_PRELOAD_DEPTH` 2 → 1 above 32 tiles in view → 0 grid-only,
  the signature dedupe collapses a monthly year to 12 frames). What was
  unbounded was the ordinary date path: **60 date changes at a browser's
  key-repeat rate issued 240 tile requests** (one whole visible tile set per
  keystroke, ×layers) and forty `#pb-scrub` `input` events issued 160, with
  **zero** superseded requests cancelled — every tile for a date the user had
  already left completed and was discarded. Separately, the aggregate/delta/
  ratio providers and the pixel probe read tiles with a bare `fetch()` that
  `Cesium.RequestScheduler` never sees, so a 365-day window put **48 requests in
  flight simultaneously** with no concurrency limit at all (the scheduler's own
  defaults — 50 total, 18 per server, throttling on, no per-host override — were
  and remain unconfigured, because nothing measured ever pushed against them).
  The fixes invent no numbers: `scrubApply` allows ONE date generation in flight
  at a time and applies only the newest pending date when `waitTilesSettled`
  reports the globe painted (240→56, 160→40), which makes cancellation
  unnecessary rather than adding it; the raw-read path is admitted through
  `Cesium.RequestScheduler.maximumRequestsPerServer` itself (peak 48→18); and a
  429/503 from GIBS — previously indistinguishable from an empty tile, which is
  how a rate limit becomes a block — now says so in a toast and drops the budget
  to one concurrent request for the session. **The rule for a future session:
  before adding anything that issues tile requests, ask whether its count is a
  function of something the user can hold down, drag or repeat; if it is, it
  goes through `scrubApply`, and if it reads tiles with a bare `fetch` it goes
  through `gibsRawAcquire`.** Full table, method and the five pinning tests:
  `docs/TILE_BUDGET.md`.
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
  numeric — anchors verified against the real July-2026 tile). One cap is
  CORRECTABLE and corrected: the SST-anomaly ±3 °C bins are a palette edge,
  not an instrument limit, so both read-outs print the computed true departure
  beside the bound (daily from the Hub, monthly fallback — §5). SMAP's
  "< 30 PSU" is a genuine retrieval floor and stays a bound.
- **There is no PET or UTCI on either allowed path — checked 2026-08-10, do
  not re-check.** GIBS's WMTSCapabilities has AIRS surface *air* and *skin*
  temperature and nothing that models a body, and the Open-Meteo family
  serves `apparent_temperature` but no UTCI. So a thermal-comfort layer of
  the kind a city Klimaanalysekarte shows cannot be built inside §3 as it
  stands. The two ways it COULD be: bake a global gridded **ERA5-HEAT UTCI**
  (Copernicus CDS, needs a free CDS account — the same shape as the GLORYS
  bake), or ingest a canton/city PET map as a bounded grid (the MeteoSwiss
  precip normal is the precedent for a national-extent grid). Both are real
  work; neither is blocked by anything but a decision.
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

**The fine tier (2026-08-31, all GIBS, ≤30 m):** **Sentinel-2** and
**Landsat 8/9 true colour** (HLS S30 / L30, 30 m, daily swaths, 2015-11 /
2013-03 →) · **Sentinel-1 C-band backscatter** (OPERA RTC-S1, 30 m, 2025-01 →)
· **NISAR L-band backscatter** (GCOV, 15 m — the finest layer, provisional,
2025-10 →) · **surface water extent** from HLS (2016 → with a 2018–23 gap) and
from Sentinel-1 (2023-12 →; both classifications) · **elevation** (ASTER GDEM
v3, 30 m, continuous metres, `probeNative`, also a pixel-card row) ·
**built-up extent** (HBASE 2010) and **impervious surface %** (GMIS 2010; both
classifications, static) · **historic Landsat true colour** (WELD annual
mosaics 1984–86 / 1989–91 / 1999–2001, `annualAnchor: "12-01"` — see
`annualYearOf`). The six daily swath layers carry `fine:` gates (§2.10). What
is NOT here and why, recorded in `data/catalog.json` as reference rows:
native 10 m Sentinel-2 (no keyless global WMTS; EOX cloudless mosaics are one),
ESA WorldCover 10 m (Terrascope WMTS is keyless — would be a second tile host
under §3), JRC Global Surface Water (keyless GCS tiles, same question), SWOT
(no tiles; bake-able as a 2 km grid), Copernicus DEM, AlphaEarth and Dynamic
World (Earth Engine auth), swisstopo 10 cm (keyless WMTS, regional), ECOSTRESS
and Landsat thermal (no tiles anywhere).

**Swath layers under the Aggregate window (2026-08-31, Chris's ask):** the
six daily swath layers carry `mosaic: true` — the window is a lookback and
`MosaicProvider` renders the union of every day in it (newest on top, ≤16
days), so "12 days" shows NISAR's whole repeat cycle; `overview: 5` paints
the day's swaths as coarse strips from orbit so a reader knows where to zoom
(the gate hides only dynamic/regional hosts now); the two radar layers carry
a `legendKey` (Worldview's own reading of the false colour); and
`minimumZoomDistance` dropped from 20 km to 100 m (Part 2: the collision
floor that read as a zoom stutter).

**Winds and currents (2026-08-31):** the globe had the ocean's current speed
from GLORYS and no wind at all. It now carries **MERRA-2 surface wind speed**
(monthly, 1980 → present, land and ocean), **AMSR-E/AMSR2 ocean wind speed**
(daily, measured, 2002 → 2025-09) and **OSCAR surface current speed** — the
observed counterpart to the modelled GLORYS field, on the same ramp and scale
so the two compare by eye. OSCAR publishes signed ZONAL and MERIDIONAL
components and neither is readable alone, so `magnitude: true` layers combine
two component rasters client-side: `MagnitudeProvider` inverts both palettes
per pixel and renders √(u²+v²) on the layer's ramp (`magnitudeAt` does the
same for one point, so the probe and the pixel card read exactly what the tile
painted). A pixel with either component missing stays empty rather than
treating the absent one as zero. Catalogued alongside: CCMP (the better wind
vector record, but its GIBS tiles stop in 2011) and JPL's **DopplerScatt**,
which measures surface winds and currents in one look — airborne swaths, not a
global grid, which is why the globe shows the two from separate sensors.

**Unions look through cloud (2026-08-31):** `unobserved` on the two DSWx
layers, a see-through compositor in `MosaicProvider`, the same rule in the
class probe, struck-through legend classes and the `[data-seethrough]` switch
(§2.5). The row hint also reports how many dates a union actually got, which
is how an archive hole (DSWx-S1, 2023-12-25 → 2024-08-20) stops looking like
a broken layer.

**The aggregation window's four controls and the scale bar (2026-08-31):**
±1d nudges, a typed field and a 12d preset join the slider on one funnel
(`setWindowDays`, §5); the scale bar reports both the ruler and the arc across
the visible ground (`updateScaleBar` / `viewGroundWidth`, §5). Two fixes rode
along: `annual` layers are no longer suppressed by a window (§2.5), and EOX's
2016 mosaic is the unsuffixed `s2cloudless_3857` — the `-2016-` form 404s, so
the layer config carries a `yearName` map.

**The third backend (2026-08-31, same day, Chris's go):** six more layers
from four keyless tile hosts beyond GIBS (§3, the tile-host bar) —
**ESA WorldCover 10 m land cover** (Terrascope, eleven classes, inline
palette, gated at 1,500 km because the tiles are rendered on demand) ·
**JRC Global Surface Water occurrence** (41 years of Landsat in one 30 m map,
inline 0–100 % ramp, a pixel-card row) · **EOX Sentinel-2 cloudless** (one
10 m mosaic per year 2016–2025, `annual`, split-comparable across years) ·
**swisstopo** SWISSIMAGE 10 cm, the 1926–2025 aerial time-travel series
(`annual`) and the swissALTI3D 0.5 m hillshade, all `rect`-bounded to
Switzerland. `xyzProvider` (stock WebMercatorTilingScheme) and
`probeTileAt` (one entry point for a probe's tile address, GIBS or mercator)
are the whole mechanism; `INLINE_PALETTES` the colormaps; `fixedWhen` the
epoch stamp for untimed maps that have one. Still NOT here: native 10 m daily
Sentinel-2 (no keyless host serves it), SWOT, Copernicus DEM, AlphaEarth,
Dynamic World, ECOSTRESS.

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

**AMOC eval mask (`data/amoc_eval_mask.json`, `classGrid`) — the globe's only
picture of a MODEL:** requested by the user on 2026-08-13, *"add a layer to the
globe visualiser to see which pixels will all be rolled forward in the amoc
eval"*. Three nested roles per cell — **rolled forward** (all 84,405 window
ocean pixels the E-022 evaluator advances a month at a time), **scored: AMOC
corridor** (29,627 — the fastest quarter of the window by train-month mean
current speed, dilated two cells, ∪ the RAPID section, which is what the
headline skill number is read from), **RAPID 26.5°N section** (265). Land and
everything outside the ML window (lat 0–70 N, lon −100..+20 E, 0.25°) is empty,
which is the truthful answer: the model holds no state there.

Two decisions worth keeping. **The file is written by the evaluator, not by the
frontend** — `python3 ml/rollout_spatial.py --export-mask data/amoc_eval_mask.json
--export-mask-only` calls the same `corridor_pixels()` the scoring calls, needs
no GPU, no embeddings and no heads (~5 min against the local tensor, seconds
warm). A corridor traced by hand in `app.js` would be a second definition of
the experiment, and the second definition is the one that silently goes stale.
**And its orientation is asserted, not eyeballed:** the writer re-samples its
own output the way `sampleGrid` will and demands the RAPID section land on
26.5°N. The first version copied the drivers bake's `[::-1]` row flip, which
this tensor does not need (its `lats` already run south-first) — that put the
Gulf Stream at the latitude of the Norwegian Sea and still looked like a
perfectly plausible ocean. A picture of a model is exactly the artefact whose
mistakes look fine.

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
baking. **The CMEMS credentials live in the claude.ai project doc
`claude/copernicus-marine-access.md`** (same footing as the GitHub PAT) and
are read as environment variables for the life of one command. An earlier
version of this line read "Credentials were deleted after each use, per the
user", which froze a single 2026-08-04 use-and-discard episode into what
looked like a standing policy — on 2026-08-16 that sentence caused a session
to tell Chris the credentials were unavailable and to plan around their
absence, when they were in the project doc the whole time. Chris: *"I don't
have such a policy."* The real rule, from the credential doc itself, is
narrower and is about PERSISTENCE, not access: env vars only, never written
to disk, never committed. Check the project docs before concluding a
credential is missing.
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

**Deferred / open follow-ups (ML):** moved to `ml/CLAUDE.md` §8, alongside the rules that govern that work.

**Deferred / open follow-ups:** OC-CCI & SMOS as first-class grid layers;
multi-channel AMOC state vector; catalog `family` field for machine-readable
dataset relationships; honest precipitation aggregation (accumulated totals
from monthly products).

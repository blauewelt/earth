# E-041 — playback: the date, animated

**Status:** Part 1 **SHIPPED 2026-08-18**, same day. The **Play** tab is live;
the blink fix (§2.2) went app-wide, so the date stepper, the compare stepper
and the window presets stopped flashing through base map too. Implemented by
two Opus subagents on disjoint files per `ml/CLAUDE.md` §0b; reviewed here,
where one redundant unheld refresh at stop — which would have ended every
playback with the exact blink the retirement queue removes — was dropped.
Verified in the browser rather than only in unit tests: twelve monthly frames
of MUR SST, four consecutive screenshots all distinct, retirement queue never
above one, zero page errors, stop landing on its own frame with the date
selector agreeing. Suite: 95/97 app + 67/67 data/docs, the two failures being
the sandbox's known Open-Meteo climate-api handshake timeout and a
phase-dependent tide-curve assertion that passes in isolation.

**One §4 requirement is NOT built and is deliberately still open:** "never
advance past a hole in silence". The signature dedupe (§2.1.3) handles the
common case by collapsing a dead zone to one frame, but a whole missing SPAN
does not yet DRAW as a gap in the scrubber — an `<input type="range">` cannot
render one, so it needs a real frame axis. Until then a gap reads as a frozen
picture, which is the one place this feature can still mislead.

Part 2 (playing a PREDICTION) is deliberately deferred — but its hooks are
decided here (§6), because they are cheap now and expensive later.

> Chris: "Let's try to animate climate development on the globe. […] Add a
> feature in the UI to 'playback' some historical data. That is, given a view
> (eg showing surface temperature / anything), select start and end date, and
> then press play. […] Once this is done, the plan is to be able to play back
> any prediction that we've created using embeddings."

Governed by the **root** `CLAUDE.md` (app rules: deploy first, stamp before
commit, the layer checklist). Part 2 crosses into `ml/CLAUDE.md`'s half and
consumes [E-039](E039_forecast_archive.md).

---

## 1. The one design decision that makes everything else free

**Playback is a clock that drives `state.date`. It is not a rendering mode.**

Everything on this globe already keys off `state.date`: GIBS tile times
(`gibsTime`), month- and day-keyed grids (`resolveGridMonth`), the comparison
(split / delta / ratio), the rolling-window means, the pixel card, the hover
probe, every legend. A playback that sets the date and lets the existing
machinery repaint inherits all of it, correctly, on day one:

- press play with **split compare** on and you get a wipe of moving present
  against a pinned past;
- press play in **delta** mode and you animate the anomaly, not the field;
- press play with a **30-day window** and you get a running mean, not noise;
- press play on a **grid** layer (currents, MLD, GFS forecast) and it works
  with no tile traffic at all.

The alternative — a bespoke animation path that renders frames its own way —
would need every one of those re-implemented, and would drift from the
single-date path the moment either changed. So: no new rendering path. One new
clock, one new panel.

The corollary is the whole answer to Part 2: **a prediction that lands as a
timed layer is playable the day it lands, with zero playback code.** §6.

---

## 2. What has to be built (and the one thing that is actually hard)

Three pieces. Two are small.

1. **Frame enumeration** — which dates are the frames.
2. **The transport** — play/pause/scrub/speed/loop, in a new tab.
3. **Not flickering** — the hard one.

### 2.1 Frames: the layer's cadence, not the calendar's

Stepping a calendar day at a time over a monthly product produces thirty
identical frames and thirty identical tile requests. Stepping a month at a time
over daily SST throws away twenty-nine days of the thing you came to watch. So
frames are enumerated from the DATA, not from the date range:

1. **Step** defaults to the *finest* cadence among the layers currently on —
   sub-daily/daily → 1 day, `snap5d` → 5 days, `monthly`/`monthlyGrid` →
   1 month, `annual` → 1 year — and can be overridden in the panel.
2. Walk `start → end` by that step.
3. **Dedupe by signature.** A candidate date's signature is what every active
   timed layer would actually REQUEST for it: `gibsTime(cfg, d)` for each GIBS
   layer, `resolveGridMonth(g)` for each keyed grid. Drop any candidate whose
   signature equals its predecessor's.

Step 3 is worth more than it looks. It collapses the thirty identical monthly
frames for free; it collapses a scrub through an archive's dead zone (GRACE
ends 2022-07, CERES 2018-10, SSH 2019-01) into ONE frame instead of four
hundred identical ones; and it is the reason this feature cannot accidentally
hammer GIBS for tiles it already has.

4. **Cap at `PLAY_MAX_FRAMES = 500`.** Over that, coarsen the step one notch
   and re-enumerate, repeating until it fits — and say so in the panel
   ("1993–2026 at 1 day would be 12,000 frames; showing 400 at 1 month").
   A cap that silently truncates the range would be the same class of lie this
   project keeps writing tests against; a cap that coarsens still shows the
   whole span the user asked for.

### 2.2 Not flickering: hold the old frame until the new one is painted

`refreshTimedLayers()` today does `removeLayer(id); addLayer(cfg)`. The old
imagery is destroyed *before* the new tiles exist, so for the length of a
network round trip the globe shows base map. Pressing the date stepper, you
see it as a blink. At 2 fps it is a strobe, and it makes correct data look
broken.

Cesium settles this for us: in `GlobeSurfaceTileProvider`, tile skeletons are
created behind `layer.show && layer._createTileImagerySkeletons(...)` — a
hidden layer requests **nothing**, an `alpha: 0` layer requests everything.
That is the whole mechanism, and it means the fix is a *retirement queue*, not
a rewrite:

    retireLayer(id)   like removeLayer, but pushes the old ImageryLayer onto
                      a `retiring` list instead of destroying it
    sweepRetired()    destroys everything retiring, called when the globe
                      reports its tile queue empty

`refreshTimedLayers({ hold: true })` then builds the new layer ON TOP of the
old one and drops the old one only once the new one has painted. No blink, in
any render mode, because `addLayer` is untouched and still owns delta/split/
aggregate/window/grid. Bound the list at 3 — a slow network must not be able
to stack fifty live imagery layers — and sweep unconditionally on stop.

This is not only a playback fix. It fixes the date stepper, the compare-date
stepper, and the window presets, all of which blink today.

### 2.3 Advancing: at network speed, never faster

The playhead advances when the frame is ON SCREEN, not when a timer says so:

    show frame i  →  wait for max(1/fps, globe tile queue empty)  →  i++

with an 8 s per-frame ceiling after which we advance anyway (a late frame beats
a stuck player). Requested fps is therefore a *speed limit*, not a promise, and
the panel says which one is binding ("1.4 fps — network-bound"). A player that
claimed 8 fps while showing 1.4 would be lying about the thing the user is
staring at.

**Prefetch, decoupled.** While frame *i* is displayed, warm the browser's HTTP
cache for frame *i+1* by `fetch(url, {mode: "no-cors"})` on exactly the tiles
currently in view (`scene.globe._surface._tilesToRender` → the same URL builder
`gibsProvider` uses). Cesium is not involved and nothing breaks if it fails;
when the frame arrives its tiles come from cache. This is the piece that turns
network-rate playback into requested-rate playback, and it is the only piece
that is allowed to be best-effort.

**Stop when nobody is watching.** Pause on `document.hidden`. Streaming NASA
tiles into a background tab is both rude and pointless.

---

## 3. The panel

A ninth tab, **Play**, on the Tides model: the picture is on the globe, the
tab is its control room. (`nav` gains `flex-wrap: wrap`; nine labels do not fit
one row of a 320 px sidebar, and two rows are better than a hidden tab.)

| control | notes |
|---|---|
| start / end date | two date inputs, same clamps as the main date |
| range presets | last 12 months · last 5 years · the whole record |
| step | auto (§2.1) · day · 5 days · month · year |
| speed | fps, a speed limit (§2.3) |
| loop | on/off |
| transport | ⏮ ⏯ ⏭ and a scrub slider over the frame index |
| readout | `frame 34 / 120 · 2015-03-04` + the per-layer provenance stamp |
| status | what the cadence resolved to, and why; the binding fps |

The readout also goes on the globe, as one more active-layer chip, so a phone
user watching the (55 % of screen) globe can see which date is on screen
without looking away.

Not in scope, deliberately: GIF/video export. Say no now, and it stays easy to
say yes later — the frame list plus a canvas grab is the whole of it.

---

## 4. What playback must never do

Requirements, stated as things that would be bugs:

- **Never show a frame under the wrong date.** The readout is the frame's
  resolved per-layer time (`whenOfGibs` / `whenOfGrid`), not the calendar date
  we asked for. A clamped archive playing its last served tile must say so on
  every frame, not just when the toast fires.
- **Never advance past a hole in silence.** If a layer's domain has no data for
  a frame, the frame is dropped by §2.1's dedupe — but a whole missing SPAN
  (GIBS publishes nothing for that month) must show as a gap in the scrubber,
  not as a frozen picture the user reads as "no change".
- **Never keep playing when the layer set changes.** Toggling a layer mid-play
  invalidates the frame list. Stop, re-enumerate, keep the playhead's date.
- **Never leave the app in playback state.** On stop, sweep the retirement
  queue, set `state.date` to the frame we stopped on, and hand the picture back
  to the ordinary single-date path.

---

## 5. Politeness

GIBS is a public NASA service and this feature is the first thing in the app
that can issue thousands of tile requests from one click. The controls that
make that acceptable are already above and are load-bearing, not decorative:
signature dedupe (§2.1.3), the 500-frame cap (§2.1.4), prefetch limited to
tiles **in view** (§2.3), advance-on-loaded rather than fire-and-forget
(§2.3), and pause-when-hidden (§2.3).

---

## 6. Part 2 — playing a prediction (deferred, hooks decided)

The deferral is real; these five decisions are not, because retrofitting them
costs more than making them now.

**6.1 A prediction is a timed layer, not a new thing.** The landing shape is
the one the app already animates: a month-keyed grid, exactly like
`data/currents_y/1993.json` — `months: {"YYYY-MM": [...]}`, `keyLen: 7`,
loaded a whole year per fetch. Which means **a 12-month rollout plays as ONE
fetch and zero tile traffic**, the smoothest substrate on the globe. E-039's
forecast archive exports to it; nothing in §2 changes.

**6.2 The key is (init date, lead), not valid date.** A rollout initialised in
January and one initialised in June both produce a field for December, and
they are different objects — the second is a 6-month forecast, the first an
11-month one, and they will disagree. So the archive is one file per
INITIALISATION (`data/pred_<var>/2026-08.json`, months keyed by valid date) and
the layer config carries `initDate`. Keying by valid date alone would silently
average two claims of different quality into one animation.

**6.3 Absolute fields are baked upstream, in anomaly space, once.** The model
predicts anomalies against the pipeline's **train-years-only** climatology
(`temporal.py`); the app's baseline is the WMO 1991-2020 normal, which includes
the holdout years. Adding the app's climatology to the model's anomaly would
produce a field that looks entirely plausible and is wrong, and it is the exact
leak [E-040 §5](E040_daily_sst.md) already forbids. **The export writes
absolute fields using the pipeline's own climatology; the app never adds one.**

**6.4 A forecast frame must announce itself, per frame.** E-039's first named
failure mode is "a beautiful animation that lies". A predicted frame therefore
carries, on the frame: that it is a forecast, its lead (`+7 months from
2026-08`), and — where the archive's scoring pass has one — the skill at that
lead. A frame at lead 11 with r ≈ 0.1 is not a picture of December; it is a
picture of the model's opinion about December, and the frame is where that has
to be said. A legend footnote is not per frame and will be read once.

**6.5 Compare, pinned to the init date, is the honest default.** Play a
forecast with the comparison pinned to its initialisation and the animation
shows the predicted CHANGE — which is what the model actually claims, and what
degrades gracefully as skill decays. The absolute field flatters the
climatology; the delta does not. This costs nothing: §1 already gives us
compare-during-playback for free.

---

## 7. Order of work

1. `retireLayer` / `sweepRetired` + `refreshTimedLayers({hold})` — the blink
   fix, shippable and useful on its own.
2. `playbackFrames()` + the play state machine + prefetch warming.
3. The **Play** tab and the globe chip readout.
4. Tests: frame enumeration collapses a monthly layer's duplicate days; a
   clamped archive yields one frame not four hundred; stop restores the
   single-date path; the date stepper no longer blinks.
5. Part 2 when E-039 has an archive to export — at which point the work is an
   exporter and a layer config, not a feature.

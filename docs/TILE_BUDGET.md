# The tile budget — what one interaction costs NASA

Every map tile this app draws is a **direct browser → NASA request** to
`gibs.earthdata.nasa.gov`. GBIF occurrence tiles are the same shape. Nothing
of ours stands in front of either: no CDN we control, no origin, no cache
(GIBS sends `no-store` — see CLAUDE.md Part 2). This is not a bill we pay. It
is a public, taxpayer-funded service, and an app that behaves badly there gets
blocked, which is worse than any bandwidth cost and cannot be bought back.

CLAUDE.md has flagged playback's preload ring as *"the first thing in the app
that can issue thousands of requests to a public NASA service from one click"*
since E-041, and called its politeness controls "load-bearing, not
decorative". This document is the first time anyone **measured** them — and
the surprise was that playback was the polite part.

## The one-line rule

> **Before adding any feature that issues tile requests, ask: is its request
> count a function of something the user can hold down, drag, or repeat?** If
> yes, it must be coalesced onto the paint clock (`scrubApply`, `src/app.js`)
> — never onto a timer, never onto the event. If its requests are read rather
> than drawn (a bare `fetch`, not a Cesium provider), it must go through
> `gibsRawAcquire`/`gibsRawRelease`, or it has no concurrency limit at all.

## Measured, 2026-08-21

MIRROR mode (`scripts/run_tests.sh`), Chromium on software GL, 1280×720
viewport — **four rendered tiles per layer** in the default globe view. A real
full-screen desktop view renders ~11, so multiply accordingly; the shape of
each row is what matters, not its absolute size. "GIBS" counts every request
to the host; "tiles" counts only the `.../{z}/{y}/{x}.png|jpg` requests, the
rest being colormap and time-domain XML fetched once per layer per session.

| Interaction | GIBS requests | of which tiles | Bounded by |
|---|---|---|---|
| First paint, default view, default layers (SST) | 10 | 8 | the visible tile set |
| Enable one layer (sea ice) | 6 | 4 | the visible tile set |
| Switch scene (surface temperature = SST + LST) | 10 | 8 | the visible tile set |
| One date step (−1d), 1 timed layer | 4 | 4 | the visible tile set |
| One date step (−1d), 5 timed layers | 20 | 20 | tile set × layers |
| One pixel-inspector click | 29 | 15 | 15 colormapped rasters, one tile each |
| Aggregate window 30 d, 1 layer | 48 | 48 | `windowSampleDates` cap of 12 × 4 tiles |
| Aggregate window 365 d, 1 layer | 48 | 48 | same cap — it binds at every scale |
| Playback, 100 frames, 1 layer | 324 | 324 | ≈4.1 per frame = one tile set, no waste |
| Playback, 100 frames, 3 layers, fastest fps | 519 | 519 | ≈5.2 per frame |
| 8 s of playback in a **hidden** tab | 4 | 4 | the frame in flight when the halt landed |

`windowSampleDates` was verified to cap at **12** sample dates for 30 / 365 /
730-day windows (1 day → 1), and the cap really binds: 12 × 4 rendered tiles =
the 48 measured. Zoom is capped at level 4 while a window is active, so the
rendered-tile count cannot grow underneath it.

Playback's documented properties were all verified to hold:
`PLAY_MAX_FRAMES` 500 (2010–2026 at 1 day = 5,838 frames coarsens to 193 at
1 month, span never truncated); `PLAY_PRELOAD_DEPTH` 2, dropping to **1** with
more than 32 tiles in view and to **0** for a grid-only frame set; playback
halts on `document.hidden` and resumes on return; and the frame-signature
dedupe collapses a monthly product's year to **12 frames** from a 1-day walk
over 365 days.

## The unbounded paths (before 2026-08-21)

Two, and they are the same defect: a request-issuing rebuild wired directly to
an event a finger can repeat.

| Path | Measured | Shape |
|---|---|---|
| The date field / date stepper | 60 changes at a browser's key-repeat rate → **240 tile requests**, 1 layer | 4 per event × layers, unbounded in time |
| The Play tab's scrub slider (`#pb-scrub`, `input`) | 40 input events → **160 tile requests** | one whole frame change per pointer move; a real 2-second drag is ~120 events |

Neither had any debounce. Worse, on every run **zero** superseded requests
were cancelled (`requestfailed` count 0, `requestfinished` equal to the
request count): every tile for a date the user had already left completed and
was thrown away. Hold the arrow key for ten seconds with five layers on and
one tab asks NASA for roughly six thousand tiles it will never show.

A third, different in kind:

| Path | Measured | Shape |
|---|---|---|
| Aggregate / delta / ratio providers and the pixel probe | 365-day window, 1 layer → **48 bare fetches, all 48 in flight simultaneously** | no concurrency limit at all |

Those paths do not *draw* tiles, they *read* them, with a bare `fetch()`.
`Cesium.RequestScheduler` never saw them, so its per-server cap never applied.
Scale that to a full-screen view with three aggregable layers and it is ~400
simultaneous connections to one public host from a single tab.

## What Cesium gives us by default

Measured, unconfigured, in the running app:

```
Cesium.RequestScheduler.maximumRequests           50
Cesium.RequestScheduler.maximumRequestsPerServer  18
Cesium.RequestScheduler.throttleRequests          true
Cesium.RequestScheduler.requestsByServer          {}   (no per-host override)
```

So Cesium's own tile path was already capped at 18 concurrent requests to
GIBS, and nothing measured ever pushed against that cap. **It was left alone**
— lowering it would slow the app without reducing the total load, and any
number picked for it would be taste rather than measurement.

## What changed

1. **One date generation in flight at a time** (`scrubApply` / `applyDateMove`).
   The first change applies synchronously, exactly as before; changes arriving
   while it is still painting replace one another, and only the newest is
   applied when `waitTilesSettled` reports the globe's queue empty. This is
   not a timer and not a picked interval — it is the app's own existing
   definition of "this date is on screen", the same one the playback loop
   advances its playhead on. The request rate stops being a function of the
   user's finger and becomes a function of the network's throughput.
   `state.date`, the date input and every read-out still move at the user's
   rate; only the part that issues requests is gated.

   Measured after: 60 key-repeat changes 240 → **56**; 40 scrub-slider events
   160 → **40**.

   This also makes request **cancellation unnecessary rather than adding it**:
   the superseded requests are no longer cancelled, they are never issued.
   (CLAUDE.md §4.1 — prefer removing a failure mode over guarding it.)

2. **One GIBS budget, not two.** The bare-`fetch` read path is admitted
   through the same per-server number the scheduled tile path already
   respects, read from `Cesium.RequestScheduler.maximumRequestsPerServer`
   rather than typed in a second time. Peak concurrency 48 → **18**, total
   unchanged at 48 (the 12-sample cap was always correct).

3. **A 429 or 503 from GIBS is an answer, and it used to be invisible.**
   `sstFetchBitmap` returned `null` for every non-OK status, so "slow down"
   and "this tile has no data" were the same event: the layer went quiet, the
   reader blamed the archive, and the app kept asking at the same rate — which
   is how a rate limit becomes a block. Those two statuses now say so once, in
   the app's own toast, and collapse the GIBS budget to a single concurrent
   request for the rest of the session. No back-off interval is invented; a
   session-long floor of one is the minimum that still makes progress, and a
   reload is the reset.

## What was deliberately NOT changed

- **`maximumRequestsPerServer`** — see above; concurrency was never the
  binding problem for the drawn-tile path.
- **Request cancellation** — made unnecessary by the coalescer rather than
  built. Cancelling Cesium's in-flight tile requests would mean reaching into
  its `Imagery`/`Request` internals for a saving that the coalescer already
  takes at the source.
- **Playback's politeness controls** — measured, all four documented
  properties hold, no change warranted. Playback issues ≈ one visible tile set
  per frame and nothing else; the preload ring costs zero requests to promote.
- **`windowSampleDates`'s cap of 12** — verified to bind at every window
  length the slider offers.
- **A 429 handler for the Cesium tile path.** Cesium raises a
  `TileProviderError` without a reliable HTTP status, so the pushback signal is
  read only where the app makes the request itself. Stated as a known limit
  rather than papered over.

## The tests that pin it

In `tests/app.spec.js`, all asserting measured **counts** with headroom rather
than implementation details, and all four verified to FAIL against the code as
it stood before 2026-08-21:

- *a held date stepper costs a bounded number of tile requests* — measures one
  step's cost as a unit on the machine it is running on, then fires 40
  animation-frame-spaced changes and requires the total to stay under half the
  unthrottled cost, and the app to land on the LAST date asked for.
- *dragging the playback scrub bar does not cost one tile set per pointer move*
  — the same shape against `#pb-scrub`.
- *the aggregate window's raw tile reads share Cesium's per-server budget* —
  peak concurrent bare fetches ≤ `maximumRequestsPerServer`, plus the
  12-sample cap asserted as one reduced array rather than one `expect` per
  point.
- *GIBS asking us to slow down is visible, and we slow down* — a stubbed 429
  must produce the toast and drop the budget to 1.
- *playback in a hidden tab stops asking NASA for tiles* — compared against
  what the same wall-clock interval cost while visible, so it means the same
  thing on any machine.

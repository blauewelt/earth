# E-071 · Cone v2 — global, harmonic, Argo-aware, and sized by the fastest thing in the ocean

**Written 2026-09-04**, from Chris's four decisions of that morning on the
representations deck's boundary and velocity slides (46–47):

> 1) Please use no geographical boundaries, the dataset should be global.
> 2) Please use a day-of-year harmonic climatology instead.
> 3) About the argo data in the stencil: it's ok to bias the argo stencil
>    shape to include the "next relevant argo locations" (doesn't have to be a
>    perfectly distributed sunflower which most likely will have only missing
>    dots).
> 4) About the ocean velocity: please research the maximal velocity of the
>    ocean and add a bit of a buffer (50%? the ocean could also become faster
>    with climate change? this applies to other channels, too).

This document turns each into a specification with the numbers behind it.
**Nothing here is implemented or measured.** It is the design the next cone
codec is built to. E-069 (the cone-native codec that reads a 30-day cone of
dots around a North Atlantic pixel, `ml/plans/E069_cone_codec.md`) has run
its five seeds on its own geometry and its H1 verdict stands as recorded
(§5); cone v2 lands on family 7 (the global 0.25°, 5-day tensor,
`ml/plans/E070_global_tensor.md`) when that tensor exists. §4.5 adapts
E-069's "two stencils, one cone" split to the v2 speeds.

Companion documents:
[the boundary and velocity findings this answers](https://blauewelt.github.io/earth/docs.html?f=ml/figures/geofm_survey/CONE_DATA_AND_ENSO.md#part-5--boundaries-how-the-sampler-handles-every-edge),
[the cone geometry as built](https://github.com/blauewelt/earth/blob/main/ml/cone.py),
[the sampler as built](https://github.com/blauewelt/earth/blob/main/ml/cone_sampler.py),
[the global tensor plan](https://blauewelt.github.io/earth/docs.html?f=ml/plans/E070_global_tensor.md).

---

## 0 · The one-paragraph answer

The cone as built (E-069) is a one-speed model of a multi-speed ocean, on a
rectangle with hard edges, with an anomaly baseline that steps at every
calendar month and a subsurface that is read at one column. Cone v2 changes
four things and nothing else: the sampler wraps the globe and never invalidates
a dot for leaving a rectangle (§1); the anomaly baseline becomes a smooth
day-of-year harmonic fit, so no bin is ever charged for a month it barely
touches (§2); the Argo depth channels stop being 32 single-column tokens and
become a handful of *profile* tokens placed where and when Argo actually
observed (§3); and every family's reach is set from the fastest measured
signal of its kind times a 1.5 safety factor, not from a typical speed —
because a cone that is too narrow is an assertion that the driver cannot have
arrived, and the model has no way to discover that the assertion was wrong
(§4). The price is a token budget of roughly 1,200 per anchor instead of 748,
and a change from uniform-area to log-radial dot placement so that a wider
cone does not starve the near field (§4.4).

---

## 1 · Global — no geographical boundary anywhere in the sampler

**Decision.** The tensor is family 7: point-aligned 0.25°, latitude −80 … 90
(681 rows), longitude −180 … 179.75 (1,440 columns). A dot is *never* invalid
for leaving a rectangle. The only invalid dots are those that fall off the
time axis (the archive's two ends), and the only unobserved dots are land,
ice-covered cells the source does not fill, and bins the source did not
observe (Argo's five dead pentads in six).

**What changes in `ml/cone_sampler.py`.**

- *Longitude wraps.* `xx = (x + dx) mod W` for the dots, the lag-0 3×3 patch,
  and the future targets. The current code refuses to wrap on purpose ("would
  put the Iberian shelf one cell west of Florida"); on a global axis the cell
  west of −180° *is* 179.75°, so the wrap is correct and the refusal is
  removed with its reason.
- *Dots are placed on the sphere, not on the grid.* Today an offset is a cell
  pair (dy, dx) computed per latitude row from a distance and a bearing with
  `cos φ` scaling. That is fine to ±70° and wrong near the poles, where a dot
  whose distance exceeds the remaining latitude has to come out on the far
  side at longitude +180°. Cone v2 computes each dot as a **destination point**
  on the sphere — start latitude, bearing, distance → end latitude, end
  longitude — and rounds that to a cell. The end longitude minus the start
  longitude depends only on the start latitude, so the dot table is still
  cached per row exactly as now; the pole crossing falls out of the formula.
- *Latitude is clipped, not wrapped.* A destination latitude below −80° is
  Antarctica in every case that matters and is marked unobserved (not
  invalid): the source has no cell there, which is a fact about the ocean, not
  about our window.
- *Anisotropy is re-measured.* `ASPECT = 0.71` was measured on the North
  Atlantic (`ml/measure_flow_anisotropy.py`). Re-run it on the global tensor
  per latitude band before cone v2 trains; a zonal-to-meridional ratio that
  holds at 40° N need not hold in the Antarctic Circumpolar Current or at the
  equator, and the per-row table can carry a per-band aspect at no cost.

**What does not change.** Land stays a `miss` token, an anchor whose cone runs
off either end of the time axis stays inadmissible, and the holdout shadow of
eight pentads stays exactly as it is (`admissible` / `certify`, E-059's
window-scope pool discipline). Those three rules were right.

**Cost.** None in tokens. The 2.68 % of dots that were off-grid on the North
Atlantic rectangle become real dots; the 22 % of anchors that saw a truncated
cone see a full one.

---

## 2 · A day-of-year harmonic climatology

**Decision.** The anomaly baseline for every dynamic channel is a smooth
function of the day of the year, fitted per cell and per channel on training
bins only, and evaluated at each pentad's **mid-day**. No bin is ever assigned
to a calendar month.

**Definition.** With τ the tropical-year phase of the bin's mid-day,
τ = 2π · (day since epoch mod 365.2422) / 365.2422,

    clim(τ) = a₀ + Σₖ₌₁..K [ aₖ cos(kτ) + bₖ sin(kτ) ]

fitted by least squares over the training bins where the cell is observed.
K = 3 (annual, semi-annual, ter-annual; 7 coefficients) is the default; a
channel whose seasonal cycle is sharp — the mixed-layer depth's spring
restratification is a step, not a sinusoid — may need K = 4 … 6 with a small
ridge on the higher harmonics. Choose K per channel by held-out variance
explained on the training years' own held-out folds, record the choice in the
run's metrics, and never by eye.

**Why this and not the twelve boxes.**

- 411 of the 3,142 pentads (13.1 %) straddle a calendar-month boundary and
  106 of them have *one* of their five days inside the month whose mean is
  subtracted. The harmonic fit has no boxes to straddle: the baseline is
  continuous across every month and every year end.
- Argo is live one pentad in six (252 bins in 21 years). A monthly box holds
  ~21 samples of that channel; the harmonic fit uses all 252 at once, with
  7 parameters instead of 12 means. Sparse, regular sampling is exactly the
  case a harmonic regression is made for.
- It replaces one pass with one pass. `anomaly_transform` in
  `ml/trainprobe.py` already accumulates per-(month, cell, channel) sums and
  counts in a chunked sweep over the tensor; the harmonic version accumulates
  the per-(cell, channel) normal equations instead — for K = 3 that is a
  7 × 7 symmetric matrix (28 numbers) and a 7-vector, 35 accumulators against
  24 today, ~1.5× the memory of a pass that already fits.

**What stays.** The climatology is still built from training years only (the
holdout-leak argument of `ml/fetch_sst_na.py` lines 27–33 stands), the
z-scoring after it is unchanged, and the app's own 1991–2020 SST normal is
still not used for anything in training.

**Verification.** The residual sawtooth is measurable: the mean anomaly of a
channel in the last pentad of each month minus the first pentad of the next,
averaged over the training years, should be indistinguishable from zero under
the harmonic baseline and is not under the monthly one. Record both numbers
once.

---

## 3 · The Argo stencil — placed where Argo is, not where a sunflower falls

**Decision.** The 32 depth channels (`rg_t*`, `rg_s*` — Roemmich–Gilson
temperature and salinity at 16 pressures, one live pentad per month) stop
being 32 separate tokens per lag read at the anchor column only. They become
**profile tokens** placed at the lags where the product is live, with a
spatial spread at those lags, and the depth axis becomes the token's *value
vector* rather than its identity.

**The gridded product (now).** Roemmich–Gilson is written into the one pentad
that contains the 15th of each month (`ml/build_family4.py`), so which lags
are live is known exactly from the anchor's bin index — it is a property of
the calendar, not of the data. For each anchor:

- take the **two most recent live bins** (the current month's, if it is
  already behind the anchor, and the previous month's); they sit somewhere in
  lags 0 … 11, so the depth channels' inner window is 11 pentads while the
  surface channels' stays at 6;
- at each of those two lags, read the anchor column plus a sunflower of 12
  dots at family B's *v2* reach for that lag (§4) — in a live pentad every
  cell of the 1° product is filled, so every one of those dots is observed;
- each of the 2 × 13 locations is **one token carrying 32 values** (16 T,
  16 S), with the pressure ladder as an inner axis the value projection sees,
  plus the token's `depth` coordinate set to the profile's deepest live level.

That is **26 tokens instead of 192**, every one of them observed, and the
token shape is fixed per anchor (two lags, thirteen locations) even though the
lags themselves move with the phase of the month — which is what keeps a batch
a tensor. What the model loses is a per-level `miss` token for the 160 dead
(lag, level) combinations it was paying for; what it gains is the horizontal
gradient of the density field at two times, which is the thermal-wind
information the anchor column alone could never carry.

**Native floats (later, rung 8 of the input ladder).** When the codec reads
Argo profiles as dots rather than as a gridded product, the "next relevant
Argo locations" are literal: for each lag inside the depth window, the
k nearest profiles to the anchor within that lag's reach, k fixed (say 8) so
the shape stays a tensor, each a profile token with its own (dy, dx, lag,
pressure ladder). No sunflower at all — the sampling geometry *is* the float
array. This is the version the proposal's "dots in, embedding out" slide
describes, and the gridded version above is its stand-in until family 8.

---

## 4 · Reach from the fastest signal, times 1.5 — for every family

**Decision.** A family's design speed is the **maximum measured speed of the
fastest mechanism that carries that channel's information**, multiplied by a
1.5 safety factor. The factor covers three things at once: the tensor is a
5-day mean on a 0.25° grid and under-reads peaks; the maxima below are
literature and reanalysis values, not the true extreme; and a warming ocean
may run faster (western boundary currents have been observed to intensify).
A cone that is too wide costs tokens; a cone that is too narrow excludes true
causes silently. The buffer is spent on the cheap side.

### 4.1 · What the ocean's maxima actually are

| mechanism | measured maximum | source | what it carries |
|---|---|---|---|
| Somali Current, south-west monsoon | up to 7 knots ≈ **3.6 m/s** | Wikipedia, Somali Current | the fastest surface current on Earth, seasonal |
| Somali Current in **our own** GLORYS bake | **2.79 m/s** monthly mean at 1°, July 2020, 2.5° N 47.5° E | `data/currents_y/2020.json`, computed 2026-09-04 | a lower bound for the pentad tensor's own maximum |
| Gulf Stream core | ~9 km/h ≈ **2.5 m/s** | NOAA Ocean Service | western boundary advection |
| Agulhas Current core | **2.45 m/s** | Wikipedia, Agulhas Current | western boundary advection |
| Kuroshio off Taiwan (our bake) | 1.62 m/s monthly mean | `data/currents_y/` | western boundary advection |
| Equatorial / coastal Kelvin wave, first baroclinic mode | **≈ 2.8 m/s** (crosses the Pacific in ~2 months) | Wikipedia, Kelvin wave; Chelton et al. 1998 atlas (c₁ largest in the western tropics, < 1 m/s poleward of ~60°) | the fast adjustment that moves an overturning anomaly between latitudes |
| Barotropic (surface) gravity wave | ≈ **200 m/s** at 4 km depth | Wikipedia, Kelvin wave | basin-wide sea-level adjustment in hours |
| Long baroclinic Rossby wave, mid-latitude | ≈ 0.03 m/s westward | Chelton et al. 2011 (already cited in E-069) | the slow interior |
| Deep Western Boundary Current | ≈ 0.02–0.1 m/s | standard oceanography | the multi-year advective path |

The slow mechanisms need no design change: a cone sized for the fast ones
contains them. The barotropic mode cannot be a cone at all — 200 m/s is the
whole globe inside one pentad — and is handled in §4.3.

### 4.2 · Design speeds

| family | channels | fastest mechanism | max | × 1.5 = **v_design** | reach per pentad |
|---|---|---|---|---|---|
| B — ocean | `cur_u`, `cur_v`, `cur_speed`, `ssh`, and the profile tokens | Somali Current (advection) ≥ first-baroclinic Kelvin (adjustment) | 3.6 m/s | **5.4 m/s** | 2,333 km |
| C — surface state | `sst`, `log_mld` | atmosphere at lags 0–1 (global, §4.3), then ocean | — / 3.6 m/s | global / **5.4 m/s** | — / 2,333 km |
| A — wind stress | `tau_x`, `tau_y`, `tau_x_std`, `tau_y_std` | the jet stream (~100 m/s; storms translate at 10–20 m/s) | ~100 m/s | **global within one pentad** | > 20,000 km |

Reach follows the existing rule r(ℓ) = v · Δt · (1 + ℓ), now capped at the
antipode (20,015 km) instead of the quarter-planet 10,000 km, because a
global tensor has no reason to stop at a quarter:

| lag ℓ | 0 | 1 | 2 | 3 | 4 | 5 | 6 |
|---|---|---|---|---|---|---|---|
| B, v2 (5.4 m/s), km | 2,333 | 4,666 | 6,998 | 9,331 | 11,664 | 13,997 | 16,330 |
| B, E-069 (0.3 m/s), km | 130 | 259 | 389 | 518 | 648 | 778 | 907 |

So by lag 3 the ocean cone is a hemisphere and by lag 6 most of the planet.
Stage 2's outer cone (`outer_reach_km`) uses the same speed and therefore
reaches its cap at lag 1; from lag 7 on the outer spiral is global.

### 4.3 · The two families that are "global at once"

Wind stress at lags 0–1, SST and mixed-layer depth at lags 0–1 (atmospheric
stirring), and sea-surface height's barotropic part are all cases where the
maximum signal speed makes the reach the whole globe within the memory
window. A disc of radius 20,000 km is not a stencil; it is the planet. Cone
v2 handles it the way E-026 handled the far field — **bearings, not density**:
a single ring of 24 dots at log-spaced radii out to the antipode at each of
those lags, so that the anchor sees "what the wind was doing in every
direction, at every scale, last pentad" in 24 tokens. The memory τ = 10 d
still limits the wind family to lags 0–1; the reach no longer limits it at
all.

### 4.4 · Dot placement must change with the reach — log-radial, not uniform-area

The sunflower today (`RAMP_P = 0.5`, Vogel's spiral) is uniform per unit
*area*, and `slots(r)` caps at 24 dots per lag. Over a 907 km disc that is
4.8 dots per octave of distance from the 28 km inner radius; over a 16,330 km
disc it would be 2.5 — the near field, where the eddies the codec was built
to see actually live, would be sampled at one dot per 4,000 km. A wider cone
with the same budget starves the scale that mattered. Therefore:

- **log-radial density**: dots placed so that each octave of distance from
  28 km to the reach gets the same number of dots (`ramp_p → 0`, or an
  explicit log spacing in radius with the golden-angle bearing sequence kept);
- **budget per lag raised to 36** for the ocean families, giving ~4 dots per
  octave out to the antipode — the same near-field density E-069 has at
  907 km — and 24 for the two "global at once" rings.

**Token budget per anchor**, families as in §4.2, profile tokens as in §3:

| | E-069 (measured) | cone v2 |
|---|---|---|
| B, 4 channels × 6 lags × (1 + dots) | 320 | 4 × 6 × 37 = 888 |
| C, 2 channels × 6 lags × (1 + dots) | 162 | 2 × 6 × 37 = 444 |
| A, 4 channels × 2 lags × (1 + dots) | 32 | 4 × 2 × 25 = 200 |
| Argo | 192 (32 × 6, column only) | 26 (2 lags × 13 profile tokens) |
| lag-0 patch | 42 | 42 |
| **total** | **748** | **1,600** |

A Perceiver's cost is linear in the token count (64 latents cross-attend to
the tokens), so this is ~2.1× the sampler and encoder time of E-069, not a
change of regime. If that is too much, the first thing to cut is the
per-channel multiplicity — one token per (lag, location) carrying all surface
channels as a value vector, as the profile tokens already do for depth, brings
the same geometry to ~500 tokens — but that is an architecture change and is
kept out of cone v2 so that v2 measures the *geometry*, nothing else.

### 4.5 · "Two stencils, one cone" under v2 — the split moves from distance to time

E-069's design (deck slide 44; `ml/cone.py::outer_spiral`,
`coverage_report`) splits the family-B cone in two: the **inner cone** — raw
channels, lags 1–6, reach r_in(ℓ) — goes into the codec; the **outer cone** —
embeddings, lags 7–143, the annulus between r_in and min(4,444 km,
0.3 m/s · 5 d · (1 + k)) — stays in stage 2. Two identities are asserted on
the grid: the union of the two stencils is the whole cone, and their overlap
is exactly the anchor column. Under v2 both identities still hold by
construction, because inner and outer reach remain one formula — but what the
formula says changes character:

- the inner reach is already 9,331 km at lag 3 and 16,330 km at lag 6, i.e.
  the codec's 30-day window sees most of the planet;
- the outer reach r_out(k) = min(antipode, 5.4 m/s · 5 d · (1 + k)) hits the
  antipode at **k = 1**, so from lag 7 onward stage 2's stencil is a **global
  ring at every lag** — the annulus's inner radius is the floor (111 km) and
  its outer radius is the antipode. The "empty annulus for k ≤ 6" property is
  unchanged; the outer spiral is simply no longer a cone that grows with lag,
  it is a cylinder of global rings.

So the division of labour between the two stencils becomes a division in
**time**, not in distance: the codec compresses *the last 30 days of raw
channels, planet-wide at log-radial density*; stage 2 reads *35 days to two
years of embeddings, planet-wide at ring density*. That is a cleaner story
than the original (where the codec owned the near field and stage 2 the far
field), and it is the one slide 44 should tell. Two consequences to decide
later, not now: (i) the outer ring's dot count per lag (24 today) should
follow the same per-octave rule as §4.4; (ii) K = 144 pentads (two years) was
chosen when the outer cone was the eddy field's memory — the deep boundary
current's eight-year path (§4.1) argues for a longer, sparser outer window,
e.g. 144 lags spaced logarithmically over eight years rather than linearly
over two. Neither is part of v2's first measurement.

### 4.6 · The rule, stated once

For any channel added later: find the fastest mechanism that moves its
information, take that mechanism's measured maximum, multiply by 1.5, and
that is the family's `v_ms`. If the answer exceeds ~45 m/s the family is
"global at once" and gets the §4.3 ring. Write the mechanism and the source
next to the number in `FAMILIES`, as `ml/cone.py` already asks.

---

## 5 · What this does NOT decide, what E-069 has already said, and what to measure first

**What E-069 has said, as of 2026-09-04 morning** (five seeds under the
frozen protocol, two objectives, `ml/EXPERIMENTS.md` § E-069): the cone
codec at 7.05M parameters and 20k × 256 steps does **not** put the anchor's
own velocity into its embedding any better than a plain ridge on the
present-day 3×3 sea-surface-height patch does (H1 refuted; the snapshot twin
sits at that raw-patch bar, the cone at or below it), and a CPU synthetic
advection test finds no displacement primitive in the architecture at the
scales tried. What the 30-day history *does* buy, on every seed: persistent
fields reconstructed far better than from the present patch (hidden
sea-surface height at ~0.36 of its bar against the twin's 0.75–0.90; sea
surface temperature 0.53–0.66 against 0.99–1.27) and one-to-two-pentad
forecasts 7–8 % better.

That changes what cone v2 can be expected to fix. Widening the inner reach
eighteen-fold does not touch the mechanism H1 tested — a codec that cannot
read a displacement at 130 km will not read one at 2,333 km — so **v2's case
does not rest on H1.** It rests on two things the E-069 results leave open:
that the far-field context the ocean's fast mechanisms carry (a Kelvin wave
that left the Gulf of Guinea last pentad, a wind-stress anomaly over the
whole gyre) improves the *rolled* skill of stage 2 (H2), and that the
persistence-and-history signal the cone demonstrably captures is worth more
when it is global, harmonic-anomaly and profile-aware than when it is a
rectangle with a sawtooth. The velocity question itself belongs to a
different lever — capacity, the objective, or an explicit displacement
primitive in the encoder — and is not what v2 measures.

- **The first measurement.** Cone v2 geometry against E-069 geometry on the
  *same* global tensor, same 7.05M codec, three seeds each, read on (i) the
  per-family held-out loss (anchor / future / dots), (ii) the persistence
  reconstruction numbers above, and (iii) stage 2's rolled skill at 5–30 days
  with the linear inverse model (a plain linear forecast fitted to the same
  field, the reference baseline) beside both. Prediction: (i) and (ii) improve
  most in the western boundary currents and the equatorial band, where the
  fast mechanisms live, and are unchanged in the gyre interiors; the velocity
  probe is reported but is not a verdict.
- **K per channel** (§2) and the **per-band aspect** (§1) are measured, not
  chosen.
- **The token budget's shape** (§4.4) is a first setting. The measurement that
  decides it is the per-octave attention mass in a trained codec — if the
  model never attends beyond 3,000 km, the budget beyond it is returned.
- The E-069 seeds keep running on the E-069 geometry. Cone v2 starts when
  family 7 is on the Hub (E-070 Phase A, 384 chunks) and the harmonic
  climatology has passed its sawtooth check (§2).

---

*Sources for the maxima:*
[Somali Current (Wikipedia)](https://en.wikipedia.org/wiki/Somali_Current) ·
[How fast is the Gulf Stream? (NOAA Ocean Service)](https://oceanservice.noaa.gov/facts/gulfstreamspeed.html) ·
[Agulhas Current (Wikipedia)](https://en.wikipedia.org/wiki/Agulhas_Current) ·
[Kelvin wave (Wikipedia)](https://en.wikipedia.org/wiki/Kelvin_wave) ·
[Chelton et al. 1998, first-baroclinic Rossby radius and gravity-wave phase speed atlas (OSU)](https://ceoas.oregonstate.edu/rossby_radius) ·
[our own GLORYS surface-current bake, monthly 1° (data/currents_y)](https://github.com/blauewelt/earth/tree/main/data/currents_y)

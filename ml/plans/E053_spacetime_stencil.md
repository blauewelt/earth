# E-053 · The space-time stencil: the sunflower taken into the time dimension

*Scoped 2026-08-26 at Chris's direction: "Let the stencil go back two steps in
time (instead of just one). So the stencil equally distributes also in the
time dimension. … We could take the sunflower to the 4th dimension (time?).
That is, each stencil can go back in time several steps (as we are going over
the 2d surface). Maybe there's a way to unify with the 3rd dimension (water
depth / later: above sea height)."*

Slides: [the two-slide deck](https://blauewelt.github.io/earth/ml/figures/spacetime_stencil.html).
Status: **PLAN — nothing dispatched.** Numbers quoted from measurements name
their experiment; everything else is design arithmetic.

---

## 1 · What the stencil samples today, stated precisely

The stage-2 head's input for one centre pixel is a **dense slab in
spacetime**: the same S+1 = 145 spatial points (centre + sunflower), each
sampled at **every one** of K uniformly spaced past frames. Per frame the 145
slots are concatenated and linearly projected to one token
(`Linear(145·d_z+2 → d_model)`); attention runs over the K time tokens only.
So the sampling pattern is a rectangle: rich and isotropic in space, dense
and uniform in time, with the two axes never traded against each other.

Two measured facts make that rectangle expensive in exactly the wrong way:

- **Context span is the axis that matters** (E-045 factorial): at 5-day
  steps, span 120 d → one-step ratio 0.5056/0.5045; span 720 d → **0.0820**
  — while step size at fixed span moves the ratio by ~4%. But under the
  dense slab, span can only be bought by raising K: 2 years at pentad = K
  144, and E-051 pays for it directly — 144×145 = **20,880 samples per
  window**, a 10.95 GB monitor batch that OOMed a v5e chip, a host pipeline
  that needed four fixes to feed the chips, attention cost ∝ K² = 36× the
  K=24 head.
- **The registered mechanism of span is the seasonal analog** (E-045,
  §sec:factorial of the paper): a 720-day window contains last year's
  same-calendar frame. If that is what span buys, then of the 144 frames
  the head is paying for, a handful near Δt = −1 y and −2 y are carrying
  the value and most of the slab is redundancy.

## 2 · The physical prior that says where the information sits

Ocean anomalies move at finite speed, so a neighbour at distance r is
informative about the centre's future mostly around lag Δt ≈ −r/c — the
**advective past cone** — plus two special regions the cone does not cover:

- **Δt = 0, all radii**: instantaneous structure (SSH gradients = geostrophic
  flow; thermal wind is a *spatial* contrast) — this is what the current
  stencil already exploits and must keep.
- **The centre's own recent history, dense**: the pentad z decorrelates
  locally in ~5 days (ρ₁ ≈ 0.60, measured on #427's monitor), so the
  highest-information samples per token are the centre's last few frames.
- **The analog cluster**: the centre's own neighbourhood at Δt ≈ −1 y, −2 y
  (E-045's mechanism, made explicit instead of smuggled in via a long slab).

What c is, is scale-dependent and must not be hand-picked (CLAUDE.md bans
hand-picked thresholds): boundary-current cores run ~100–200 km/day, interior
first-baroclinic Rossby waves at 26°N ~3–5 km/day, and fast boundary-wave
adjustment is effectively the Δt = 0 disc. **E-053.0 (§4) measures c instead
of choosing it.**

## 3 · The design family

Give each stencil slot its own offset **(Δx, Δy, Δt)** — a low-discrepancy
point cloud in the centre pixel's past spacetime, generalizing the
golden-angle sunflower (2-D, far-heavy radial ramp) to three dimensions
under the metric ds² = |dx|² + (c·dt)², Δt ≤ 0. Three patterns, from
agnostic to opinionated:

| pattern | placement | prior it encodes |
|---|---|---|
| **ST-BALL** | 3-D Fibonacci-type spiral filling the past half-ball, far-heavy ramp in spacetime radius | none — "equally distributes also in the time dimension" (the literal form of the idea) |
| **ST-CONE** | points concentrated near the surface Δt = −r/c (with thickness), plus the Δt=0 sunflower disc, plus dense own-history | advection: sample the inflow trajectories |
| **ANALOG pins** | a small sunflower (8–12 pts) around the centre at Δt = −1 y, −2 y (…−n y) | the seasonal analog, priced at ~20 samples instead of ~120 frames |

All three keep every Δt ≤ 0 (causality untouched), score against the
**unchanged** persistence baseline (the target step does not move — only the
context sampling does, so unlike E-048's overlapping windows nothing about
the denominator changes and every number stays comparable to the pentad
archive), and route frames before 1982 or before a channel's onset through
the existing dead-slot/missing-token path.

## 4 · Staged execution, cheap first

**E-053.0 — measure c, $0 GPU.** Space-time cross-correlation of the
published pentad Z (`Z_8b639abe36_37e146384b`, 16.24 GiB, CPU job): for
pixel pairs at separation r, the lag Δt* maximizing corr(z(x,t),
z(x+r,t+Δt)). The ridge Δt*(r) *is* the measured cone; its slope is c (or
shows there is no single c, which decides BALL over CONE). Also yields the
marginal value-of-information curve in Δt at r=0 — the empirical basis for
the log ramp. One afternoon, one script, publishable as a figure.

**E-053.1 — the time-sunflower on FRAMES (no architecture change).** Keep
the per-frame 145-slot concat exactly as-is; make the frame *times*
non-uniform and add a Δt encoding to each frame token (the month-block
codec's learned time-offset embedding, head-side — code that already exists
in one form). Implementation is window assembly + one embedding table:
`--frame-offsets "0,-1,-2,-3,-4,-6,-9,-13,-19,-28,-41,-60,-72,-73,-74,-145,-146,-147"`
class of knob, slicing rows from the published Z — the same
slice-the-published-Z affordance that unblocks E-045.3, one code path for
both. Arms at #427's exact configuration, 20k steps, ~$3–5 each, n=1
directions per §3b (E-045 style; the K=24 pair 0.5056/0.5045 and the K=144
rung 0.0820 are the already-measured controls):

| arm | frames (K_eff) | span | question | registered reading |
|---|---|---|---|---|
| **A1 analog-only** | 16: dense 0…−6, pins −73±1, −146±1 | 2 y | is span's value the analog? | ratio ≲ 0.12 ⇒ mechanism confirmed, span sparsifies ~10×; ≳ 0.3 ⇒ dense slow-mode context matters, analog insufficient |
| **A2 log-ramp-24** | 24 log-spaced 0…−146 | 2 y | does log spacing match uniform K=144? | ≈ 0.08 at 1/6 the frames = span decoupled from K; ≈ 0.5 = the slab was load-bearing |
| **A3 decade-32** | 32 log-spaced 0…−730, pins at −1…−9 y | **10 y** | does span keep paying beyond 2 y? | first reach past 2 y at any cadence; monotone gain ⇒ the axis is still open (decadal-predictability territory) |

**E-053.2 — the true spacetime point cloud (architecture change, gated on
E-053.1).** Per-slot Δt breaks the per-frame concat; the honest version is
point-cloud attention — each token one (value, coordinate-encoding) pair,
attention over N points — which also replaces the fixed `Linear` spatial
mixing, the crudest part of the current head (the AR-vs-diffusion deck's
slide-1 caveat). Only worth building if E-053.1 shows the slab sparsifies;
it converges with the field-head line (E-052's FieldDiT attends over space
already), so the two designs should be unified *there*, not built twice.

## 5 · Unification with depth (and later, height)

Depth is today a **channel** axis: 16 Argo pressure levels enter the codec
as channels and the whole column compresses into one z per pixel-bin. Two
statements, kept separate on purpose:

- **At the head, unification is just coordinates.** E-053.2's tokens carry a
  coordinate encoding of (Δx, Δy, Δt); choosing an encoding with a fourth,
  *signed vertical* slot (−depth below the surface, log-spaced like the
  Argo levels; +height above it for a future atmospheric tier) costs
  nothing now and means per-level tokens plug in the day they exist. The
  codec's own decoder already answers coordinate queries
  (channel, Δlon, Δlat, Δmonth) — the system is closer to coordinate-native
  than it looks.
- **Making the codec emit per-depth tokens is an E-047-class structural
  change, not a head knob.** It multiplies the token count by the level
  count, re-opens the d_z economics (the Chinchilla arithmetic is per
  observed value and must be redone), and touches the Argo
  one-live-bin-per-month policy. Nothing in E-053.1/.2 depends on it; this
  plan pre-commits only to an encoding that admits it.

## 6 · Falsifiers and the one danger to name in advance

- **Replay is the failure mode this design courts.** An analog pin is a
  lookup mechanism *by construction* — it hands the head last year's frame
  instead of asking it to integrate dynamics. E-043b-PHASE's battery
  (six context-end rolls; flat within-record tracking vs lead = replay) and
  the flat-lead-time check are **mandatory** on any E-053 head before a roll
  number is quoted. A one-step win whose roll shows the flat profile is
  memorisation sharpened, not forecasting — reported as such.
- **Per-arm falsifiers** are in the table above; A1's is two-sided and is
  the scientifically decisive one (it localizes *where in spacetime* the
  E-045 span effect lives).
- **Interaction with E-051**: either outcome of E-051's roll makes E-053.1
  next-in-line — if the K=144 roll is positive, E-053.1 is the cost
  reducer (same span at ~1/6 the attention, which is what makes a 10-year
  span affordable at all); if negative, A1 says whether the one-step span
  effect was ever more than analog lookup, *before* the next full-budget
  spend.

## 7 · Cost arithmetic (design reasoning, not measurements)

Per-window input samples = 145·K_eff; attention ∝ K_eff².

| configuration | span | K_eff | samples/window | rel. attention |
|---|---|---|---|---|
| #427 (uniform, today's pentad head) | 120 d | 24 | 3,480 | 1× |
| E-051 (uniform, full span) | 720 d | 144 | 20,880 | 36× |
| A2 log-ramp | 720 d | 24 | 3,480 | **1×** |
| A3 decade | 3,650 d | 32 | 4,640 | 1.8× |

E-053.1's whole first wave (E-053.0 + three arms) is ~$12–18 and one
session, on the existing trainer.

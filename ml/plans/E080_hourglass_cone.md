# E-080 · The cut-off mirrored double cone — stage 1 as an hourglass, with a cone the model shapes itself

**Written 2026-09-15**, from Chris's four points of that morning:

> 1) Input is a cone with its tip cut off (so there is a bottleneck of several
>    points, not just a single point).
> 2) We also sample a "prediction" cone (which is never used as input, but just
>    as targets).
> 3) We use a random sampling regime where the stage-1 model sometimes predicts
>    the future given past + present, sometimes just given the present, and
>    predicts the present given multiple ways of dropout from the past.
> 4) We want to make the cone sizes (ellipse angle + width / breadth + centre) a
>    set of parameters that are learnable per area. Make a good proposal here
>    (think of the AMOC which comes from one side, so for a point interested
>    in ocean currents the cone can be directed towards the source of the
>    current. The model should learn this).

This document is the deck's content, slide by slide, with the plain-English
speaker notes. **Nothing here is implemented or measured.** It is a design
and a pre-registration. The deck:

- [Slides as PDF](https://blauewelt.github.io/earth/ml/plans/E080_hourglass_cone_deck.pdf)
- [Slides with the speaker notes as PDF](https://blauewelt.github.io/earth/ml/plans/E080_hourglass_cone_deck_with_notes.pdf)
- [PowerPoint file, notes in the notes pane](https://github.com/blauewelt/earth/raw/main/ml/plans/E080_hourglass_cone_deck.pptx)

Read with:
[E-069 · the cone codec as built](https://blauewelt.github.io/earth/docs.html?f=ml/plans/E069_cone_codec.md),
[E-071 · cone v2](https://blauewelt.github.io/earth/docs.html?f=ml/plans/E071_cone_v2.md),
[the cone geometry, `ml/cone.py`](https://github.com/blauewelt/earth/blob/main/ml/cone.py),
[the masking plan, `ml/cone_codec.py::default_plan`](https://github.com/blauewelt/earth/blob/main/ml/cone_codec.py).

---

## Slide 1 · Title

**The cut-off mirrored double cone**
Stage 1 as an hourglass — and a cone the model shapes itself

E-080 · design proposal · 15 September 2026 · nothing here is measured yet

*Notes.* This deck proposes the next shape of the stage-one model, the encoder that turns one place at one time into an embedding. Chris set four points this morning: the input cone loses its tip and becomes a bottleneck of several points; a second cone is sampled into the future and used only as targets; the model trains under a random mix of forecasting and reconstruction tasks; and the shape of the cone — where it points, how wide it is, which way it is stretched — becomes something the model learns separately for every region of the planet. The first three points are settled decisions and the deck turns them into a specification. The fourth is a request for a proposal, and the deck's recommendation is a warped sunflower: a fixed pattern of sample points that a small set of per-region numbers rotates, stretches and shifts, read from the data by interpolation so that the loss can push on the numbers directly. Everything is written so it can be measured as one experiment, E-080, against the geometry we already have. Nothing in this deck has run.

---

## Slide 2 · Where we come from

**One cone, five seeds, one clear reading**

- E-069 (the cone-native codec): 30 days of dots around a North Atlantic cell, one 3×3 patch at the present, targets at +1 and +2 pentads
- History did **not** put the cell's own velocity into the embedding (five seeds, two objectives — refuted)
- History **did** buy persistent fields (hidden sea-surface height at ~0.36 of the bar vs 0.75–0.90) and a 7–8 % better one- and two-pentad forecast
- E-071 (cone v2): global, harmonic climatology, profile tokens, reach from the fastest mechanism × 1.5 — a hemisphere by lag 3, so dots go log-radial
- The open question v2 left: a cone wide enough to never exclude a driver samples the near field at one dot per thousands of kilometres

*Notes.* A quick reminder of the record, because the new design is a response to it. E-069 was the first cone-native codec: for every ocean cell it read a thirty-day cone of samples around it, called dots, plus the present three-by-three patch, and it was asked to predict the cell's own future one and two pentads out — a pentad is a five-day bin. Its headline hypothesis, that thirty days of history would put the cell's own current velocity into the embedding, was refuted at five seeds under two objectives. What the history did buy was real: fields that persist, like sea-surface height, were reconstructed far better from history, and the short forecast improved by seven to eight percent. Then cone v2 fixed the geometry's known faults — it went global, it smoothed the seasonal baseline, it placed the Argo float tokens where floats actually were, and it sized every cone from the fastest thing that could carry information, times a safety factor. That last choice has a cost the v2 plan named itself: with a reach of a hemisphere by lag three, a fixed budget of dots samples the near field very thinly. The design in this deck is what lets the cone be both — wide enough to never exclude a cause, and dense where the causes actually are — by letting each region decide where its dense part points.

---

## Slide 3 · The hourglass in one picture

**Past cone → waist → future cone**

[FIGURE: hourglass_spacetime — a space-time diagram: horizontal axis = space (one direction, km), vertical axis = time in pentads from −6 to +6. The past cone: a shaded region widening downward from the waist; the waist: a short flat segment of several cells at lag 0 (not a point); the future cone: widening upward from the waist, hatched differently. Annotations: "input (read)" on the past cone and waist, "targets only (never read)" on the future cone, "waist = several cells, the embedding z is made here". Show the anchor column as a dashed vertical line through all lags. Optional: a tilted version alongside (drift), captioned "a cone the model may tilt".]

- **Past cone** (lags −6…−1): read as input, dots at the per-lag ellipse plus the anchor column
- **Waist** (lag 0): a disc of several cells — the cut-off tip — read as input and reconstructed; the embedding z summarises the waist
- **Future cone** (leads +1…+6): never read; sampled the same way and used as **targets only**
- Point-mirrored: the future cone at lead +ℓ is the past cone at lag −ℓ reflected through the anchor

*Notes.* This is the whole shape on one page. Time runs upward, space sideways. Below the middle is the past cone: everything the model reads about the last thirty days, widening with lag because a driver further back in time can have come from further away. In the middle is the waist — the tip of the cone, cut off. In E-069 the present was a single three-by-three patch; here it is a small disc of several cells, and it is the only place the model both reads and is asked to reconstruct. The embedding is made here, at the waist. Above the middle is the future cone: the same shape reflected through the anchor point, widening into the future, and never read — it exists only to be predicted. Mirrored means point-mirrored: the future cone at lead plus three is the past cone at lag minus three turned through the anchor. That is not a decoration; it is what advection looks like in space-time. Water that reaches the anchor came from upstream and will go downstream, so a cone that leans upstream in the past should lean downstream in the future. The dashed line is the anchor column — the cell's own history and its own future — and it is always inside both cones whatever else the geometry does, so the fixed-frame forecast that stage two needs is always a target.

---

## Slide 4 · Point 1 — the waist: why the tip is cut off

**A gradient does not exist within a single cell**

- The waist is a disc of radius r₀ around the anchor, sampled as a small sunflower: ~13 cells at the default (one centre + 12), never fewer than the 3×3 patch's nine
- Read as input at lag 0 (values + observed flags) and reconstructed as the anchor family's target
- What the waist buys: thermal wind is a density *gradient*; geostrophy is a sea-surface-height *slope* — both need neighbours at the same instant, not in the past
- The waist has the same shape parameters as the cone at lag 0 (orientation, two semi-axes) so a boundary current can have an elongated waist along the flow
- The embedding z summarises the waist; one z per anchor, as today

*Notes.* Point one. Why cut the tip off at all? Because the quantities the AMOC read-out actually depends on are gradients. Thermal wind — the vertical shear of the current — is a horizontal density gradient. Geostrophic velocity is a slope in sea-surface height. A gradient does not exist inside one cell; it needs neighbours at the same moment. The E-069 record says the same thing from the other side: the snapshot twin's velocity read-out was exactly what a ridge regression on the raw three-by-three patch gives — geostrophy from the patch — and the cone added nothing to it. So the present deserves more than a single point. The waist is a small disc of cells around the anchor, thirteen by default, arranged as a tiny sunflower so that every bearing is represented. It is read as input, and it is what the model reconstructs in the anchor-reconstruction task. It carries the same shape parameters as the rest of the cone evaluated at lag zero, so where the flow is a narrow jet the waist can be stretched along it. One embedding per anchor summarises the waist, exactly as today, so stage two and every archived probe see the same kind of object.

---

## Slide 5 · Point 2 — the prediction cone

**Sampled like the past cone, read by nobody**

- Leads +1…+6 pentads (30 days ahead), the mirror image of the past cone through the anchor: centre c(+ℓ) = −c(−ℓ), same axes and orientation at the same |ℓ|
- Dots are **targets only**: each is a decoder query (channel, Δx, Δy, lead, depth) → (μ, log σ²), scored with the Gaussian negative log-likelihood already in `cone_codec.py`
- The anchor column at every lead is always among the targets — the fixed-frame forecast stage 2 needs
- Per-lead and per-family loss, each reported against its own predict-the-mean bar (the E-069b rule)
- Evaluation geometry is **frozen** (the fixed v2 hourglass), whatever the training cone learns — so every number stays comparable across arms and seeds

*Notes.* Point two. The prediction cone is the future half of the hourglass. It is sampled by the same machinery as the past cone, reflected through the anchor, out to six pentads ahead — thirty days, matching the thirty days of history. Nothing in it is ever read by the encoder. Each dot in it is a question put to the decoder: at this channel, this offset, this lead, this depth, what is the value? The decoder answers with a mean and a variance, and is scored with the Gaussian negative log-likelihood we already use. Two rules carry over from E-069b, the run that fixed the first masking plan: every loss family is reported against the score you would get by predicting the mean, so a target that cannot be known shows up as a bar of one rather than hiding in an average; and the anchor's own column is always among the targets, so the forecast at the anchor itself — the thing stage two rolls forward — is always trained. One more rule is new and matters for point four: the geometry used at evaluation time is frozen to the fixed hourglass. The training cone may move; the yardstick does not. Otherwise two runs with different learned cones would be scoring themselves on different questions.

---

## Slide 6 · Point 3 — the sampling regime, as a table

**One model, four questions, drawn at random per example**

| task | input | targets | share | what it teaches |
|---|---|---|---|---|
| T1 forecast | waist + past cone | future cone | 0.35 | the full problem |
| T2 snapshot forecast | waist only (past cone fully hidden) | future cone | 0.15 | the built-in twin: what history buys, inside one model |
| T3 nowcast from history | past cone under a dropout pattern; waist hidden | waist | 0.25 | the present from the past — persistence, advection, memory |
| T4 fill-in | waist + past cone, both under a dropout pattern | the hidden waist cells and dots | 0.25 | E-069b's reconstruction family, kept |

Dropout patterns for T3 and T4, drawn independently (each fires with its own probability):
**channel drop** at the waist (lag 0 only, `chan_drop_scope lag0`) · **recent-lag band** (hide every dot at lag ≤ ℓ₀, ℓ₀ ∈ {1,2,3}) · **bearing wedge** (hide a 90° sector) · **random dots** (each dot hidden with p = 0.3) · **whole waist** (T3 only)

*Notes.* Point three. Each training example draws one of four tasks. Task one is the full problem: read the waist and the past, predict the future cone. Task two is the same question with the past entirely hidden — the snapshot twin from E-069, but now living inside the same model, so every run reports what history buys without a second run. Task three is the nowcast: hide the present and reconstruct it from history, under one of several dropout patterns applied to the past. Task four keeps E-069b's reconstruction family: hide some cells and dots in both the waist and the past and fill them in. The dropout patterns are the ones already in the code plus one: dropping a channel at the present only, hiding the most recent lags so the model must extrapolate from stale history, hiding a wedge of bearings so it cannot lean on one direction, hiding dots at random to imitate sparse observations, and — for the nowcast only — hiding the whole present. The shares are a starting point, not a result; they are a knob the experiment records. The one rule that is not a knob comes from E-069b's failure: a target's channel must be visible somewhere in the input, at some lag, or the target is unknowable and only teaches the model to predict the mean.

---

## Slide 7 · Point 3 — the rules that make the mix safe

**Three rules, all learned the hard way in E-069**

- **Knowable targets only.** A target channel must be visible somewhere in the input (any lag, any dot). Under the first E-069 plan 80 % of hidden-dot targets broke this and 64 % of the loss was predict-the-mean
- **No copy targets.** The anchor family scores hidden channels only (`anchor_hidden_only`); a target visible in the input is a copy, not a question
- **Every family against its own bar.** Per (task, family, lead) loss, each divided by its predict-the-mean score; a task is "learned" only where the ratio is below one at held-out anchors
- The task is not told to the model by a token; the masks are visible (the `mask_tok` / `miss_tok` distinction stays), and the lead is in every query's coordinates — that is enough
- Loss weights: equal across tasks per example; the shares in the table set the mix, not the weights

*Notes.* Three rules, all paid for already. The first is that every target must be knowable from the input: the first E-069 plan hid a channel at the present and at every dot, so most of the hidden-dot targets belonged to channels the encoder could not see at all, and two thirds of the loss became a lesson in predicting the mean. The second is that no target may be a copy of something visible: the anchor family scores only channels that were dropped, because a target the model can read off the input is not a question. The third is that every loss family is reported against its own predict-the-mean bar, per task, per family, per lead, so a number is only called learned where the ratio sits below one at held-out anchors. On mechanics: the model is not told which task it is in by a special token. The masks are visible — there is already a token for "hidden by us" and a different one for "the data never observed this" — and every decoder query carries its lead in its coordinates. That is enough for the model to know what is being asked. The shares in the table decide how often each task is drawn; within an example the loss weights are equal.

---

## Slide 8 · Point 4 — the cone as a warped sunflower

**Seven numbers per region shape the whole hourglass**

[FIGURE: warped_sunflower — three panels: (a) the canonical sunflower on the unit disc; (b) the same sunflower rotated by θ and stretched to semi-axes a, b; (c) the same at three lags, each shifted by ℓ·d and grown by ℓ·s, drawn on a map-like background with a coastline hint, arrow "d points upstream".]

For each region and channel group, at lag ℓ (past, ℓ = 1…6) the sample points are

  p(ℓ, i) = anchor + ℓ·**d** + R(θ) · diag(a₀ + ℓ·s_a, b₀ + ℓ·s_b) · u_i

- **u_i**: the canonical sunflower on the unit disc (Vogel spiral, golden angle), fixed, ~24 points — the same pattern at every lag
- **d** (2 numbers): the drift per pentad, in km — where the cone's centre moves as you go back in time; **this is the arrow toward the source**
- **θ** (1): orientation of the ellipse
- **a₀, b₀** (2): the waist semi-axes, in km — the cut-off tip's size and elongation
- **s_a, s_b** (2): how fast each axis grows per pentad — spread
- Future cone: p(−ℓ) reflected, c(+ℓ) = −ℓ·**d**, same axes at the same |ℓ|; the anchor column (0, 0) is added at every lag regardless

*Notes.* Point four, the proposal. The cone becomes a warped sunflower. Start with the sunflower we already use — a fixed pattern of about twenty-four points on the unit disc, placed by the golden angle so every bearing is covered once. Seven numbers per region turn that pattern into a cone. A drift vector, two numbers, says where the centre of the pattern moves for each pentad you step back in time; that is the arrow toward the source of the current, and it is the thing Chris asked the model to learn. An angle orients the ellipse. Two semi-axes set the size and elongation of the waist. Two growth rates say how much each axis widens per pentad — the spread, which stands in for eddies and uncertainty in the path. The future cone uses the same seven numbers reflected: its centre at lead plus three is minus three times the drift, downstream. And at every lag the anchor's own cell is added as a fixed point, so the cone can lean away from the anchor without ever losing the anchor's own history. Seven numbers is deliberately few. It makes the learned shape something you can draw on a map and argue with — a drift map of the world ocean is itself a result — and it keeps the parameters far too few to memorise anything.

---

## Slide 9 · Point 4 — how the numbers learn: sample between the cells

**Continuous positions, interpolated reads — the gradient reaches the geometry**

- A dot at a continuous position reads the field by **bilinear interpolation** over its four surrounding cells, weighting by the observed flags: value = Σ wᵢ mᵢ vᵢ / Σ wᵢ mᵢ, observed-fraction = Σ wᵢ mᵢ
- The interpolation weights are functions of the position, the position of the seven numbers, so ∂loss/∂(**d**, θ, a, b, s) exists — the deformable-convolution / spatial-transformer trick, applied to a gather instead of a convolution
- Footprint, not point: each dot averages over a Gaussian footprint whose width tracks the local dot spacing (~r/√n), read from a mip-map pyramid of the tensor — so a dot means "this area", and the gradient is not confined to four cells
- Same cost class as today: one gather per dot (4 cells, or one pyramid cell), on the fly, keyed exactly as E-069's examples are

*Notes.* How does a loss reach a geometry? By never snapping the dots to cells. A dot sits at a continuous position and reads the field by interpolating the four cells around it, with the observed flags folded into the weights so a missing cell contributes nothing and the dot also reports what fraction of its footprint was observed. The interpolation weights depend smoothly on the position; the position depends on the seven numbers; so the derivative of the loss with respect to the drift, the angle and the axes exists and ordinary backpropagation delivers it. This is the trick behind deformable convolutions and spatial transformer networks, applied here to a gather rather than a convolution. One refinement matters: a plain bilinear read has a gradient that only sees four cells, so a cone could never discover a driver half a degree away. Instead each dot reads a Gaussian footprint whose width follows the spacing between dots — a dot means "this area", which is physically right — taken from a pre-averaged pyramid of the tensor, so the read is still one gather. The cost class is the one we have: one gather per dot, on the fly, addressed by the same key an E-069 example has, plus the pyramid built once per tensor.

---

## Slide 10 · Point 4 — where the seven numbers live

**A coarse map per channel group, interpolated to the anchor**

- A 2° grid over the globe (90 × 180 cells); the anchor's parameters are the bilinear blend of the four surrounding map cells — smooth by construction, no seams
- One map per **channel group**, because the source of a signal depends on what moves it: ocean surface (currents, sea-surface height, sea-surface temperature, mixed layer), Argo **upper** (≤ 300 dbar), Argo **mid** (300–1000), Argo **deep** (> 1000 dbar), atmosphere (drift only at lags 0–1), land (no drift: v = 0, axes and angle only)
- Size: 16,200 cells × 6 groups × 7 numbers ≈ 0.7 M parameters — a tenth of the 7 M codec
- Optional, phase 2: a first-harmonic seasonal term on the drift, **d**(τ) = d₀ + d₁ cos τ + d₂ sin τ — the Somali Current reverses with the monsoon
- Why per depth: at the RAPID line the upper limb flows north and the Deep Western Boundary Current flows south — the same anchor should point its surface cone south and its deep cone north

*Notes.* Where do the seven numbers live? On a coarse map. A two-degree grid over the globe has sixteen thousand cells; an anchor takes the blend of the four cells around it, so the geometry varies smoothly and never jumps at a cell edge. There is one such map per channel group, because what moves a signal decides where its source is. The ocean surface group is moved by the surface currents. The Argo groups are split by depth into upper, mid and deep, because the ocean's layers move in different directions — at the RAPID line, where the AMOC is measured, the upper limb flows north and the Deep Western Boundary Current flows south, so the very same anchor should point its surface cone toward the south and its deep cone toward the north. The atmosphere group only has drift at the two lags inside its ten-day memory. The land group has no drift at all: soil moisture does not advect, so it learns only the axes and the angle. All told that is about seven hundred thousand numbers, a tenth of the seven-million codec. A phase-two option is a seasonal term on the drift — one harmonic of the day of year — because the Somali Current reverses with the monsoon and one arrow per region cannot say that.

---

## Slide 11 · Point 4 — constraints, the mirror, and the initial cone

**Bounded by physics, tied to the fixed cone, started isotropic**

- **Speed cap**: |**d**| ≤ v_design · Δt = 5.4 m/s × 5 d = 2,333 km per pentad (E-071's fastest-mechanism × 1.5), via a tanh parameterisation — the learned cone lives inside the v2 envelope
- **Floors**: a₀, b₀ ≥ one cell (28 km); s_a, s_b ≥ 0; a₀ + 6 s_a ≤ v_design · 6 Δt
- **Mirror tie** by default: the future cone shares the seven numbers (reflected); an untied arm is the ablation, not the design
- **Stop-gradient on the target side**: the loss reaches the geometry only through the *input* reads; the positions of targets are treated as constants
- **The v2 safety net stays**: in addition to the learned ellipse, each lag keeps one fixed log-radial ring of 24 dots to the antipode — the cone can be *directed* without *asserting* that a driver could not have come from elsewhere
- **Init**: **d** = 0, θ = 0, a₀ = b₀ = 3 cells (aspect 0.71 applied), s from E-069's 0.3 m/s — the fixed hourglass exactly, so step 0 of the learned arm *is* the control

*Notes.* The seven numbers are bounded by physics and tied to the fixed cone. The drift cannot exceed the design speed cone v2 set — the fastest measured current times one and a half, about two thousand three hundred kilometres per pentad — so the learned cone always lives inside the envelope v2 argued for. The axes cannot shrink below one cell or grow faster than the design speed. The future cone shares the numbers, reflected; letting it have its own is an ablation. Two guards protect the learning itself. First, the loss reaches the geometry only through the input reads; where a target sits is treated as a constant. Without that, the model would learn to move its questions to wherever they are easiest, and a cone that points at flat water would look excellent. Second, the v2 safety net stays: alongside the learned ellipse, every lag keeps one fixed ring of twenty-four dots at log-spaced radii to the far side of the planet. That is the point of v2 — a cone that is too narrow silently asserts that a driver could not have arrived — and the learned ellipse adds direction and density on top of it rather than replacing it. And the initial values reproduce the fixed hourglass exactly, so the learned arm starts as the control and every departure from it is something the data asked for.

---

## Slide 12 · Point 4 — the AMOC example, drawn

**One anchor on the western boundary, two cones pointing opposite ways**

[FIGURE: amoc_cones — a schematic North Atlantic map (coastline of Florida, the Bahamas, the US east coast; the 26.5°N RAPID line dashed). An anchor at ~27°N, 77°W. Surface group: past-cone ellipses at lags 1, 3, 6 drifting south / south-west along the Florida Straits toward the Caribbean, growing with lag; future cone ellipses drifting north-east along the Gulf Stream path. Deep group (drawn in a second colour): past-cone ellipses drifting north along the continental slope toward the Grand Banks; future drifting south. Arrow legend: "surface d: upstream = south-west", "deep d: upstream = north". A faint fixed ring around the anchor labelled "v2 safety ring (fixed)".]

- Surface group: **d** points south-west (up the Florida Current, toward the Loop Current); the future cone leans north-east along the Gulf Stream
- Deep group: **d** points north along the slope (up the Deep Western Boundary Current); the future cone leans south
- Neither is programmed: these are the **pre-registered expectations** for the learned maps — the check that the mechanism works
- If the learned surface drift correlates with −u_clim (the climatological surface current, reversed) over the world ocean, the cone learned "upstream" from the data alone

*Notes.* Here is the picture Chris asked for, drawn as we expect the model to draw it. An anchor on the western boundary just north of the RAPID line. For the surface group, water arrives up the Florida Current from the Caribbean, so the past cone should drift south-west, growing as it goes back, and the future cone should lean north-east along the Gulf Stream. For the deep group, the Deep Western Boundary Current runs the other way along the continental slope, so the past cone should drift north toward the Grand Banks and the future cone south. Neither arrow is programmed. They are what we expect the maps to show if the mechanism works, and they are written down now, before any run, so the check cannot be shaped by the result. The world-ocean version of the check is a single number: the correlation between the learned surface drift and the climatological surface current reversed, over cells where that current is at least a fifth of a metre per second. A clearly positive value means the cone learned "upstream" from the data alone. And the faint ring around the anchor is the fixed v2 safety net — always there, never learned, so nothing the model does can wall off a direction.

---

## Slide 13 · The loophole and its fix

**A learnable cone can learn to ask easy questions**

- If target positions could move, the gradient would move them to smooth water — a lower loss that means nothing
- Fix 1: stop-gradient on target positions (slide 11). The geometry is trained only by "which inputs help", never by "which targets are easy"
- Fix 2: the mirror tie. The future cone cannot be tuned separately from the past cone
- Fix 3: frozen evaluation geometry. Held-out numbers are always scored on the fixed hourglass, for every arm
- Residual second-order effect: as the input cone moves, the mirrored targets move with it. Monitored, not assumed away: report the target-variance under the learned cone vs the fixed cone at every eval; a drop > 10 % is a flag

*Notes.* One slide on the trap, because it is the kind that produces a beautiful curve and a wrong conclusion. If the model can move where its questions are asked, the gradient will move them to where the answers are easy — smooth open water, away from fronts — and the loss will fall for no reason we want. Three fixes. The target positions are constants as far as the gradient is concerned, so the geometry learns only from which inputs help. The future cone is tied to the past cone, so it cannot be tuned on its own. And every held-out number is scored on the frozen hourglass, the same for every arm. There is a residual, second-order version of the trap: as the input cone moves, the mirrored targets move with it. We do not assume that away. At every evaluation the run reports the variance of its targets under the learned cone against the fixed cone, and a drop of more than ten percent is a flag that the geometry is drifting toward easy water rather than toward the source.

---

## Slide 14 · What it costs

**Tokens, gathers, parameters — the same regime as v2**

| | E-069 (measured) | this design (budget) |
|---|---|---|
| waist | 42 patch tokens (14 ch × 3×3) | 13 per-location tokens carrying all surface channels |
| past cone | 706 dot tokens | 6 lags × (1 + 24 ellipse + 24 ring) × 4 groups ≈ 1,176 |
| Argo | 192 (32 × 6, column only) | 26 profile tokens (E-071 §3) |
| targets | 42 + 84 | 256 dot queries per example, drawn from the future cone and the hidden waist/dots (as `n_dot_queries` today) |
| geometry parameters | 0 | ≈ 0.7 M (slide 10) |
| extra per-tensor artefact | — | the mip-map pyramid (~⅓ of the tensor, built once) |

- Perceiver cost is linear in tokens: ~1.6× E-069's encoder time, in line with v2's own 2.1× estimate; the interpolated gather is 4 cells instead of 1
- Training cost per seed: E-069 seeds ran under one 4090-hour on the North Atlantic; budget 2–3 4090-hours per seed on the global tensor

*Notes.* What it costs, in the same units as the earlier decks. The waist is thirteen per-location tokens each carrying all surface channels, which is fewer than the forty-two per-channel patch tokens E-069 used. The past cone is six lags of one anchor column, twenty-four ellipse dots and twenty-four ring dots, for four channel groups — about twelve hundred tokens. Argo comes as the twenty-six profile tokens cone v2 already specified. Targets are drawn as two hundred and fifty-six decoder queries per example, as now. The geometry adds about seven hundred thousand parameters and one new artefact: a pre-averaged pyramid of the tensor, about a third of its size, built once. Because the encoder is a Perceiver, its cost is linear in the token count, so this is about one and a half times E-069 per step — inside the two-times cone v2 already budgeted for itself. Per seed, E-069 ran under one GPU-hour on the North Atlantic; on the global tensor, budget two to three. Nine runs for the experiment on the next slide is on the order of twenty-five GPU-hours.

---

## Slide 15 · The experiment — E-080, pre-registered

**Three arms, three seeds, four read-outs, one verdict rule**

- **A0 fixed**: the hourglass with the v2 geometry (isotropic ellipses, ring, mirror), no learning — the control and the eval geometry for everyone
- **A1 learned**: the seven numbers per region, initialised at A0
- **A2 learned, primed**: as A1 but **d** initialised at −u_clim · Δt (the climatological surface current, reversed) for the ocean surface group — tests whether the gradient can find upstream on its own (A1) or only keep it (A2)
- Same tensor (family 7.2, global), same 7 M codec, 20 k steps, frozen protocol (train ≤ 2020; 2008–09, 2016–17, 2021–24 held out), 3 seeds each
- **R1** T1 future-cone loss per lead vs bar · **R2** the T1 − T2 gap (what history buys) · **R3** drift-vs-upstream correlation, world ocean, |u_clim| > 0.2 m/s · **R4** the western-boundary depth check (surface south-west, deep north) · **R5** target-variance flag (slide 13)
- **Verdict**: A1 adopted if it beats A0 on R1 at every lead ≤ 3, paired at all three seeds, with R5 clean. A2 > A1 with A1 ≈ A0 means the gradient cannot find upstream — then geometry is set from climatology, not learned. Neither → keep A0 and move to a dynamic (per-example) cone

*Notes.* The experiment, written before anything runs. Three arms. The fixed hourglass with cone v2's geometry is the control and also the evaluation yardstick for every arm. The learned arm starts from the control. The primed arm starts its surface drift at the climatological current reversed, which separates two questions: can the gradient find upstream on its own, or can it only keep upstream if handed it? Everything else is held fixed — the global family 7.2 tensor, the seven-million codec, twenty thousand steps, the frozen protocol with its held-out years, three seeds per arm because the read-outs are probe-scored. Five read-outs: the forecast loss per lead against its bar; the gap between the full task and the snapshot task, which is what history buys; the correlation of the learned drift with upstream over the world ocean; the depth check at the western boundary; and the target-variance flag. The verdict rule is written now. The learned cone is adopted if it beats the control on the forecast at every lead up to three, paired at all three seeds, with the flag clean. If only the primed arm wins, the gradient cannot find upstream and geometry should be set from climatology rather than learned. If neither wins, the fixed hourglass stays and the next design is a dynamic cone — one whose drift is predicted per example from the waist — not a bigger static one.

---

## Slide 16 · Build order

**Five steps, each with its test, in `ml/cone.py`, `ml/cone_sampler.py`, `ml/cone_codec.py`, `ml/train_cone.py`**

1. **Waist + future cone in the geometry** (`cone.py`): `waist_dots`, `future_dots` = reflected `inner_dots`; `coverage_report` extended with the mirror identity (future set = −past set) — pure numpy, CPU test
2. **The task mix** (`cone_codec.py::default_plan` → `plan_v3`): T1–T4 with shares and the five dropout patterns; test: every target's channel is visible somewhere in the input, for 10 k drawn examples, zero violations
3. **Continuous sampling** (`cone_sampler.py`): pyramid build + interpolated gather with observed-weighting; test: at integer positions with a full observed mask the read equals today's gather bit-for-bit
4. **The parameter maps** (`cone_codec.py::ConeGeometry`): 2° maps per group, tanh/softplus bounds, mirror, stop-gradient, init = A0; test: at init the dots equal step 1's fixed dots to < 0.1 cell; gradient-check on the seven numbers with finite differences
5. **Read-outs in the trainer** (`train_cone.py`): per-task bars, R3/R4 maps exported as `data/cone_drift_<group>.json` for the app's Cones tab, R5 flag; then the three arms as recipes `f7l2-hourglass-{fixed,learned,primed}`

Each step lands on main with its test before the next starts; step 3 is the only one that touches the data path and is gated on the bit-identity test.

*Notes.* The build order, for whoever implements it. Five steps, each small enough to land with its own test before the next begins. First the geometry: the waist and the future cone as functions in the cone module, with the coverage report extended to assert the mirror identity — pure numpy, testable on a laptop. Second the task mix, as a new masking plan next to the existing one, with a test that draws ten thousand examples and finds zero targets whose channel is invisible. Third, the only step that touches the data path: the pyramid and the interpolated gather, gated on a bit-identity test — at integer positions with everything observed it must reproduce today's gather exactly. Fourth the parameter maps as a module with its bounds, the mirror, the stop-gradient and the control initialisation, checked by finite differences. Fifth the read-outs in the trainer, including exporting the learned drift maps to the app so the Cones tab can draw them. Then the three arms are three recipes on the global tensor, and E-080 is dispatched under the usual rules.

---

## Slide 17 · Open decisions for Chris

**Six choices the deck took a position on — all reversible**

- Future horizon **+6 pentads** to mirror the past; or shorter (+1…+2 as E-069) with denser targets?
- Task shares **0.35 / 0.15 / 0.25 / 0.25**; any of them a lever you want set differently?
- **2°** parameter maps; 1° doubles resolution at 4× the (small) parameter cost
- Depth groups at **300 and 1000 dbar** for the Argo cones
- The v2 **safety ring kept** at every lag (24 tokens per lag per group); dropping it halves the past-cone tokens but restores v2's "too narrow" risk
- Seasonal drift term **deferred** to phase 2; or in from the start (monsoon regions)

*Notes.* Six choices the deck made so it could be concrete, each of them one line to change. The future cone goes to six pentads to mirror the past, where E-069 only asked for one and two. The task shares are a first guess. The parameter maps are at two degrees. The Argo depth groups split at three hundred and a thousand decibars. The v2 safety ring is kept at every lag, which is the safe and more expensive choice. And the seasonal term on the drift waits for phase two. Say the word on any of these and the spec changes; none of them affects the build order.

---

## Slide 18 · Sources

- E-069 · Two stencils, one cone — `ml/plans/E069_cone_codec.md`; results in `ml/EXPERIMENTS.md#e-069` (five seeds, H1 refuted; what history buys)
- E-069b masking-plan fix — `ml/cone_codec.py::default_plan` (`chan_drop_scope`, `anchor_hidden_only`), `ml/train_cone.py`
- E-071 · Cone v2 — `ml/plans/E071_cone_v2.md` (§3 profile tokens, §4 design speeds and log-radial dots, §4.5 the split in time, §6.4 per-location tokens)
- E-076 · Family 8 nearest observations — `ml/plans/E076_family8_nearest_observations.md` (the `(value, Δx, Δy, Δt, n_R)` token)
- The geometry as built — `ml/cone.py` (`FAMILIES`, `reach_km`, `slots`, `inner_dots`, `outer_spiral`, `coverage_report`)
- Deformable sampling: Dai et al. 2017, *Deformable Convolutional Networks*; Jaderberg et al. 2015, *Spatial Transformer Networks*; Zhu et al. 2021, *Deformable DETR*
- Currents: E-071 §4.1's table (Somali 3.6 m/s, Gulf Stream ~2.5, DWBC 0.02–0.1 m/s); RAPID 26.5°N array for the limb directions

*Notes.* Everything the deck leans on. The E-069 plan and its five-seed record; the E-069b masking fix in the codec; cone v2 for the design speeds, the profile tokens and the per-location tokens; family 8 for the token that carries its own offset; the geometry module for every constant. The learning mechanism is the deformable-sampling family from computer vision — deformable convolutions, spatial transformers and Deformable DETR — used here on a gather. The current speeds and the directions of the two AMOC limbs are the ones cone v2 already tabulated and the RAPID array's own description of the section.

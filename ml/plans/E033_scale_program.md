# E-033 · The next scale program — a plan to review before anything is started

**Status: PROPOSAL. Nothing here is dispatched.** Chris, 2026-08-16:
*"What paid most so far is scale: Can we add more data (also: temperature on
land but also everything that's missing and could be useful), more pixels,
more temporal granularity (daily or every 3 or 10 days?), and then think
about a 10x larger transformer and 10x larger embedding model. What would be
the infra implications for data storage. And then: Do we need H100s for those
runs? But first we need the data. Make a plan to review, before starting
anything."*

Every storage and compute figure below is computed from the tensor we
actually have or measured from live Vast offers on 2026-08-16, not estimated.

---

## 1 · Where the evidence says we are

Five results bound what to do next. They are worth stating together because
three of them point the same way and two of them cut against the obvious move.

| finding | status | implication |
|---|---|---|
| **Stage-2 capacity pays on BOTH scoreboards** — 34M → 88M → 205M gives forecast ratio 0.176 → 0.148 → 0.129 and corridor AUC 0.571 → 0.621 → 0.664 | no saturation at 205M | keep climbing, but see §5 |
| **Input width saturates at 89 points** — xl144 − xl89 paired −0.0001, 17× inside seed noise | CLOSED | do not build 233; more *neighbours* is not the axis |
| **Step budget decelerates** — 60k→120k bought −0.0072, 120k→200k bought −0.0016 | asymptotic | steps are a finishing move, not a strategy |
| **The transport probe is at a STATE/LABEL ceiling, not a representation tax** — pooled z scores 0.627 where the *true uncompressed fields* score 0.631 | measured, `probe_state_ceiling.py` | for AMOC-the-number, more model cannot help. Only more labels and more state can |
| **The codec has been FROZEN at 40.7M since the quarter-degree tensor landed** | never re-scaled | the one untested axis, and the floor everything else stands on |

Read together: **we have been scaling the part of the system that was cheapest
to scale, and the two things that bound the science — the codec's capacity and
the amount of independent data — are both untouched.** That is not a criticism
of the last week; the stage-2 ladder was the right experiment and it produced
the programme's clearest result. But it is why "10× the transformer" is not
obviously the next move, and why Chris's own instinct — *"but first we need
the data"* — is the correct sequencing.

**The sharpest way to see the data problem.** Stage 2 consumes *timesteps*.
We have **516** of them. Forty-three years of ocean, and the temporal axis the
forecasting model learns from is five hundred numbers long. Every one of the
26M "windows" it trains on is a re-slice of those same 516 months across
84,405 spatially-correlated pixels. The effective independent sample count is
far smaller than any parameter count we have discussed, which is the honest
explanation for why a 205M head can keep improving a *one-step* objective
while the *rolled* skill and the transport probe move much less.

---

## 2 · Data axis A — temporal granularity (the recommended first move)

Chris asks: daily, or every 3 or 10 days? **Recommendation: 5-day (pentad)
first, daily as a later rung.** The reasoning is that 5-day is where the
observing system actually lives, and it is the only rung that multiplies the
data without breaking the storage design.

**Why finer time is the highest-value data change:**

1. **It multiplies the axis stage 2 is starved on.** 516 → 3,096 timesteps
   at 5-day (6×), → 15,695 at daily (30×).
2. **It multiplies the LABELS, which the ceiling result says is the binding
   constraint for transport.** RAPID is natively 12-hourly and the Florida
   Current cable is *daily* — we currently throw that away by monthly-meaning
   into 240 labels. At 5-day the RAPID record alone becomes ~1,460 labels, and
   the cable's 1982–2024 record becomes ~3,100. This is the only proposal on
   this page that attacks the measured ceiling.
3. **It un-aliases the physics we documented as invisible.** The 2009–10
   collapse's onset was "triggered inside a month" and therefore unreachable
   at our cadence; Ekman transport, the dominant term in RAPID's monthly
   variance, is a *weather* response.

**Which channels actually support it** (this decides the rung):

| channel group | native cadence | at 5-day |
|---|---|---|
| GLORYS currents, MLD, SSH | daily | native |
| NCEP wind stress + storminess | daily (we already read dailies to build the within-month σ) | native, and the σ becomes a within-pentad σ |
| OISST SST | daily | native |
| Altimetry SSH anomalies | 5-day repeat | **exactly native** |
| Argo T/S (RG gridded) | monthly; floats profile ~10-daily | slower channel, enters as the same value across pentads OR as missing tokens between updates |

Argo is the only laggard, and the architecture already has the right answer
for it: *missingness is information*, and a channel that updates every ~10
days carrying `missing` tokens in between is a truthful representation of
what the observing system knew. **This must be a design decision made
explicitly, not a side effect** — the alternative (forward-filling Argo) makes
the model believe it has subsurface data it does not.

**Daily is deferred, not rejected**: it costs 5× the storage of pentad for a
further 5× in timesteps, but *no* additional label density beyond what the
cable already gives, and it puts Argo 30 timesteps behind its own updates.
Revisit after the pentad rung reports.

---

## 3 · Data axis B — more channels

Chris asks specifically for land temperature, plus "everything that's missing
and could be useful". The governing rule is already in the repo and it
disqualifies several obvious candidates: **a channel is worth what position
and season cannot explain of it** (the static-channel lesson — two climatology
channels were measured to contribute *exactly nothing*, because the encoder
already receives lat/lon and month).

Candidates, ranked by expected value against that rule:

| candidate | source | why it could pay | risk |
|---|---|---|---|
| **Ocean bottom pressure** (GRACE/GRACE-FO) | already in the catalog, tiles→2022-07 | the deep compensation term of the overturning; the ML-from-simulation literature (Solodoch, Meng) feeds OBP and it is the input we most conspicuously lack | coarse (~300 km), mission gap 2017–18, record ends 2022 |
| **Sea-ice concentration** | AMSR2 / OISST ice | the subpolar freshwater and buoyancy gate — the "engine room" of deep-water formation | already partly implicit in SST |
| **Land 2 m temperature** (Chris's ask) | NCEP R1 (already fetched for wind) / ERA5 | continental heat storage is a real boundary condition on air–sea flux, and it is free — we already download NCEP | land pixels are outside the ocean mask; needs a decision on whether they enter as *pixels* or as a coastal-forcing channel |
| **Surface heat flux** (net, latent, sensible) | NCEP R1 | buoyancy forcing is the other half of the overturning's driver and we currently see only the *wind* half | reanalysis flux is a model product, not an observation |
| **Precipitation / E−P** | IMERG (in catalog) | surface freshwater flux → density | weak at monthly, better at pentad |
| **Ocean colour / chlorophyll** | PACE, in catalog | a tracer of upper-ocean structure and mixing | mostly a surface-optics proxy; low prior |
| **Runoff / river discharge** | GloFAS | subpolar freshwater | fiddly, low prior |

**On land temperature specifically.** There are two very different ways to add
it, and the choice matters more than the channel does. As a *channel on ocean
pixels* it is nearly useless (an ocean pixel's own land temperature is
undefined). As *new pixels* — extending the mask over land — it changes the
model from an ocean model to an Earth-surface model, multiplies the pixel
count by ~1.4 in this window, and is a genuinely different experiment. My
recommendation is the third option: **coastal-forcing channels** — land
temperature and land–sea contrast sampled at each ocean pixel's nearest land,
which is cheap, keeps the mask, and tests the hypothesis (continental heating
influences the boundary current) without rebuilding the tensor's geometry.

Cost of the whole channel programme in bytes: 39 → ~60 channels is **1.5×**,
which is nothing against the cadence and pixel axes. **Channels are cheap;
spend the review time on which ones, not on whether we can afford them.**

---

## 4 · Data axis C — more pixels, and what it costs

Two different moves are being conflated when we say "more pixels": *more
area* (global) and *finer resolution* (1/12°). They cost the same and buy
different things.

Measured from the current tensor (281 × 481 × 39 × 516 = 10.9 GB, verified):

| tensor | float32 | float16 | timesteps | Z cache (d_z 64, fp16) |
|---|---|---|---|---|
| **NA · monthly · 39ch — today** | 10.9 GB | 5.4 GB | 516 | **5.6 GB** |
| NA · monthly · 60ch | 16.7 GB | 8.4 GB | 516 | 5.6 GB |
| **NA · 5-day · 39ch** | 65.3 GB | **32.6 GB** | 3,096 | **33.4 GB** |
| NA · 5-day · 60ch | 100 GB | 50 GB | 3,096 | 33.4 GB |
| NA · daily · 60ch | 509 GB | 255 GB | 15,695 | 170 GB |
| GLOBAL · monthly · 39ch | 83.5 GB | 41.7 GB | 516 | 48.6 GB |
| GLOBAL · 5-day · 39ch | 501 GB | 250 GB | 3,096 | 292 GB |
| GLOBAL · daily · 60ch | 3.9 TB | 2.0 TB | 15,695 | 1.5 TB |
| NA 1/12° · monthly · 39ch | 98 GB | 49 GB | 516 | 49 GB |

Global 0.25° is **7.67×** the current grid; its ocean pixel count is ~736,000
against our 84,405 (**8.7×**).

**A note on 1/12°.** The tempting reading of "more pixels" is finer
resolution, and I would argue *against* it as the next move for a specific
measured reason: our Argo T/S channels are intrinsically 1°-smooth and are
already bilinearly upsampled into the 0.25° grid, carrying no sub-degree
information. Going to 1/12° multiplies storage 9× while making the *density*
channels — the ones thermal wind says carry the transport — even more
obviously interpolated. Finer resolution should wait until a density product
justifies it.

**Global is now also a CORRECTNESS requirement, not only a data one.**
Measured after this plan was first written (`ml/measure_cone_escape.py`): a
144-point stencil reaching 4444 km has a dependency cone that covers the
ENTIRE window by horizon 3, and ~50% of its slots resolve to land-or-outside
at every step. Ten of the twelve horizons in our headline corridor AUC are
therefore scored under an unstated boundary condition — the world outside the
window held at its climatological mean. Going global does not remove the
boundary (the Southern Ocean edge remains) but it moves it thousands of
kilometres from the corridor and eliminates the Atlantic-basin walls that
currently consume half of every wide stencil's input.

**Global, by contrast, adds genuinely independent samples**: the Pacific and
Southern Ocean are different dynamical regimes, not more of the same
correlated Atlantic. That is exactly what a data-starved model needs, and it
is the strongest argument for the global move — with the caveat that OSNAP,
MOVE, SAMBA and the cable are all still Atlantic, so global buys
self-supervised data without buying transport labels.

---

## 5 · Storage and infrastructure implications

**This is where the current design actually breaks, and it breaks earlier than
the model does.** Today's pipeline: the tensor is chunked into the
`data-cache-v1` GitHub release, every box pulls and sha-verifies it, the
embedding cache `Z` rides `embed-cache-v1` in 1.5 GiB chunks, and boxes carry
100 GB disks.

Three hard limits, in the order they will be hit:

1. **GitHub release assets cap at 2 GiB each** (and 1000 per release). Already
   worked around by chunking; at 250 GB that is 125+ chunks per tensor and the
   pull becomes the dominant cost of every run.
2. **Box disk is 100 GB and cannot be resized** (measured — the Vast API
   returns success and changes nothing). A pentad NA tensor in fp16 (33 GB)
   plus its Z (33 GB) plus the torch image (~15 GB) is **81 GB of a 100 GB
   disk** — it fits, with no room for anything else. Anything larger does not
   fit at all.
3. **`Z` must be fast-access for stage-2 training.** It is currently
   RAM-resident on the big boxes (128 GB), which is why gathers are cheap. At
   NA-pentad (33 GB) it still fits comfortably; at global-pentad (292 GB) it
   fits on no box we rent.

**Recommended infrastructure moves, cheapest first:**

- **Store the tensor in float16.** Free 2× on everything. The fields are
  anomaly-space and normalised to ~N(0,1); fp16 has ~3 decimal digits there,
  far below observational error. *This should be done regardless of which
  other axis is chosen.* One-line change in the builder, one assertion in the
  recipe guard.
- **Move bulk artifacts to object storage** (Cloudflare R2 or Backblaze B2)
  with a signed-URL pull, keeping GitHub releases for *checkpoints* only. R2
  has zero egress fees, which matters when nine boxes each pull 33 GB per cold
  start. Budget: ~$0.015/GB/month → a 250 GB tensor + 300 GB of caches is
  **~$8/month**. This is the single most important infra change and it
  unblocks every later phase.
- **Rent boxes with larger disks for the data phases.** Disk is not resizable,
  so this means creating new instances; offers with 500 GB–3 TB exist at the
  same GPU price (measured today: an H100 SXM offer carried 2,963 GB).
- **Stream, don't materialise, above ~100 GB.** A memmap over an object-store
  file with a local block cache; the training loop already reads windows, not
  the whole tensor. This is real engineering work (~a day) and should be
  scheduled *before* the global phase, not during it.

---

## 6 · Do we need H100s?

**Measured Vast pricing and dense bf16 throughput, 2026-08-16:**

| GPU | bf16 TFLOPs | VRAM | $/h | $ per PFLOP-hour | relative |
|---|---|---|---|---|---|
| **RTX 4090** | 165 | 24 GB | **0.136** | **0.82** | **1.00×** |
| RTX 5090 | 210 | 32 GB | 0.324 | 1.55 | 1.88× |
| A100 SXM4 | 312 | 40 GB | 0.468 | 1.50 | 1.82× |
| L40S | 362 | 45 GB | 0.601 | 1.66 | 2.02× |
| H100 SXM | 989 | 80 GB | 1.604 | 1.62 | 1.97× |
| H100 NVL | 835 | 94 GB | 1.536 | 1.84 | 2.23× |

**The answer is: not for cost, only for fit — and not yet.** The 4090 is
**twice as cheap per unit of compute** as an H100 on this market. An H100 buys
wall-clock and memory, not efficiency. So the question is purely whether the
model fits.

Memory for AdamW training (weights + fp32 moments + gradients ≈ 16 bytes per
parameter, before activations):

| stage-2 size | optimiser state | fits 4090 (24 GB)? |
|---|---|---|
| 205M — today | 3.3 GB | comfortably |
| 600M — **the next rung** | 9.6 GB | **yes**, with room for activations |
| 1B | 16 GB | marginal; needs care |
| 2B — Chris's "10×" | 32 GB | **no** → H100 |
| 7B | 112 GB | no → multi-GPU |

**Conclusion: the next capacity rung (≈600M) runs on the hardware we already
rent, at half the cost per FLOP of an H100.** Take that rung first. H100s
become necessary at ~1–2B, and by then the honest question will be whether
the *data* supports 2B parameters — which is exactly why the data phases come
first. One further consideration: our whole runner design is single-box; 2B+
would eventually mean FSDP/sharding across GPUs, which is a new class of
infrastructure work and its own failure taxonomy.

---

## 7 · The proposed sequence, with decision gates

Each phase states what would make us *stop* rather than continue — the same
discipline every experiment entry uses.

**Phase 0 — the codec rung (no new data, ~$20, days).**
Train a 400M codec (Chris's "10× embedding model") on the tensor we already
have, plus an intermediate ~120M rung so the curve has three points. This is
the untested axis and the floor under everything else; it needs no new data,
no new infrastructure, and runs on current boxes. *Gate: if the 41M → 400M
step does not move stage-2's forecast ratio or corridor AUC beyond seed noise,
the codec is not the bottleneck and every later phase should assume the 41M
codec is adequate — which would itself be a major, publishable finding.*

**Phase 1 — float16 tensor + the pentad rebuild (~$0 storage, ~1 week).**
Rebuild the NA tensor at 5-day cadence in fp16 (33 GB), with the RAPID and
cable truth series re-derived at the same cadence. Re-anchor the Chinchilla
arithmetic and retrain the codec at the size that data implies. *Gate: if
6× the timesteps and ~6× the labels do not improve the held-out transport
probe beyond its current ~0.63 ceiling, then the ceiling is the ocean's, not
the sampling's, and the transport target is closed at monthly-scale skill —
also a real finding, and one worth the paper.*

**Phase 2 — channels (~$0 storage, days).**
Add, in this order: ocean bottom pressure, surface heat fluxes, sea ice, and
the coastal land-temperature contrast channels. Each must clear the
static-channel rule. *Gate: measure each channel's marginal contribution with
the existing ablation machinery before adding the next.*

**Phase 3 — object storage + streaming loader (~$8/month, ~1 week).**
Prerequisite for anything global. Do it while Phase 2 runs.

**Phase 4 — global 0.25° (storage 250 GB, ~$50–100 of GPU per codec).**
The genuine independent-sample expansion. *Gate: does a globally-trained codec
match or beat the NA-only codec on the Atlantic tasks? We have tested exactly
this once before at 1° and the answer was "matches exactly" — the
generic-embedding hypothesis. Re-testing it at 0.25° with 8.7× the pixels is a
headline experiment in its own right.*

**Phase 5 — the 10× transformer (H100, ~$2/h).**
Only once Phases 1–4 have produced a dataset that can support 2B parameters.
By then the Chinchilla arithmetic will say whether it can.

**Rough total before Phase 5: on the order of $200–400 of GPU and about three
weeks of wall-clock**, most of it data engineering rather than training.

---

## 8 · What I would argue about this plan

Three places where I think the plan could be wrong, stated so the review has
something to push against:

1. **Phase 0 may be a distraction.** If the codec is adequate (and the
   `probe_state_ceiling` result hints it might be — compressed z matched the
   *true fields*), then $20 and two days buys a null. I still recommend it,
   because it is the cheapest way to retire the largest untested assumption in
   the system, and a null there redirects everything else.
2. **Pentad may not help transport as much as I claim.** The label multiplier
   is real, but 5-day RAPID values are noisier per sample; the effective DOF
   gain will be less than 6×. The honest expectation is "meaningfully more
   than 240 labels, much less than 1,460".
3. **Global may buy less than it looks.** More pixels of a correlated field is
   not more information, and the Atlantic is where all our labels are. The
   1° precedent says regional and global codecs tied — which is a reason to
   expect *parity*, not gain, and parity would mean the pixels bought nothing
   for this task.

---

*Written 2026-08-16 as a proposal. No dispatches, no rentals, no rebuilds have
been made against it. Storage figures computed from the live tensor's own
dimensions; GPU prices and throughputs from Vast offers surveyed the same day.*

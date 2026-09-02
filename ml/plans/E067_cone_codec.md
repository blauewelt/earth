# E-067 · Two stencils, one cone: the cone-native codec (ocean physics first)

**Status: BUILT (code + tests, 2 Sep 2026 — 42 tests green, CPU smoke end-to-end), NOT DISPATCHED.** Registered by
Chris's ask of 2 Sep 2026 (*"implement the first version of this. Start by
preparing the data, then the cones logic … Probably: a stencil for the codec
and then a stencil for stage 2. Them together will provide the best result
(and together they will implement the full dependency cone)"*). The survey
deck's slides 34–46 and
[`ml/figures/geofm_survey/GENERIC_EMBEDDING_INPUTS.md`](https://blauewelt.github.io/earth/docs.html?f=ml/figures/geofm_survey/GENERIC_EMBEDDING_INPUTS.md)
carry the full argument; this plan is the executable half.

> **TL;DR.** Today the codec sees one pixel-bin (3 × 3, one pentad) and stage 2
> sees a cylinder of embeddings (145-point spiral × 144 pentads). Nothing that
> needs *two snapshots* — velocity, tendency, convergence — can be encoded by
> the codec, so stage 2 has to rebuild it from compressed codes. E-067 splits
> the dependency cone in two by physics: an **inner cone** of raw channels
> (lags 0–6 pentads, reach set per channel family) goes *into* the codec, so
> the embedding carries local motion; an **outer cone** (lags 0–144, reach
> growing with lag, minus what the codec already saw) stays in stage 2 over
> embeddings. Their union is the whole dependency space. The question: does
> putting the inner cone inside the codec buy rolled skill at 5–30 days —
> the leads where the learned head still beats the LIM — beyond the seed
> interval, at the 7.6M tier?

---

## 0 · Structured header (§0d)

`E-067 · cone-native codec (inner cone of raw channels, lags 0–6 pentads) +
stage-2 head on the outer cone · params codec 7.05M at 42 channels (Perceiver,
64 latents × 256 wide × 6 blocks; `ConeMAE.param_count()`) + head 7.6M (256×8, K 144) · stage encoder then stage-2
· data family4_na025_pentad_r3 (r2 + cur_u, cur_v) · arch inner cone A/B/C
families (below) · steps 20k×256 codec, ≤ 5k×256 head with held-out-minimum
selection · resume none (fresh)`

Control: **E-064b's configuration** — the same 7.6M head on the same tensor
over the continuous d_z-32 codec (run-415) with the full cylinder stencil,
`--holdout-scope window`, z-noise 0.7, five seeds per step 5.2 of the reboot
plan. Null: the 200-mode LIM in pixel space (E-066) and the same LIM fitted in
*embedding* space (both codecs).

## 1 · Hypothesis and falsifier

**H1 (velocity in the embedding).** A codec that reads the inner cone encodes
local advection and tendency: a ridge head from the frozen embedding to
`cur_u`, `cur_v` at the anchor — with the codec's own current channels
*dropped from the input* — reaches an R² that the 3 × 3 snapshot codec cannot.
Falsified if the two codecs' velocity-probe R² agree within the seed interval
(n ≥ 3 each).

**H2 (rolled skill at short leads).** The stage-2 head over cone embeddings,
under the identical roll battery as #516 (window-scope pool, per-lead `acc` and
`msss_clim`, trained/held-out longitudes separately, block-bootstrap
intervals), beats the E-064b control at leads 1–6 (5–30 d) on the current and
SSH channels by more than the five-seed interval, and stays within the
interval of the control at leads ≥ 18 (90 d). Falsified if leads 1–6 agree
within the interval — then the inner cone bought nothing stage 2 could not
already rebuild, and the "work in the embedding" argument loses at this tier.

**H3 (the split is not zero-sum).** With the outer stencil *restricted to the
annulus the codec did not see*, the head matches or beats the same head on the
full cylinder. Falsified if the annulus head is worse by more than the
interval — then stage 2 needs the near field twice, and the overlap should be
kept.

Predictions written before the run: H1 holds (R² 0.3–0.6 vs < 0.1); H2 holds
at leads 1–3 for `cur_*` and `ssh`, not for `sst`; H3 holds. What decides the
programme's next step is H2.

## 2 · The two stencils

The dependency cone of the survey deck: for a driver with propagation speed
`v` and memory `τ`, information from lag `ℓ` reaches the anchor from within
`Δx ≤ v·(Δt + ℓ)`, and only lags with `Δt + ℓ ≤ τ` are worth reading; reach
is floored by a correlation length `L_corr` and capped at 10,000 km. Pentad
`Δt = 5 d`. Channel families on this tensor (speeds are the deck's
order-of-magnitude values, deliberately generous):

| family | channels | v | τ | L_corr | inner reach r_in(ℓ), ℓ in pentads | inner lags |
|---|---|---|---|---|---|---|
| A fast, wide, short | `tau_x`, `tau_y`, `tau_x_std`, `tau_y_std` | 10 m/s | 10 d | 500 km | 500 km (L_corr) | 0–1 |
| B slow, narrow, long | `cur_speed`, `cur_u`, `cur_v`, `ssh`, `rg_t*`, `rg_s*` | 0.3 m/s | months–years | 100 km | max(100, 0.3 m/s × 5 d × (1 + ℓ)) = 130, 260, …, 910 km | 0–6 |
| C L-shaped | `sst`, `log_mld` | A at short lags, B beyond | 3–6 mo | 100 km | max(r_B(ℓ), 500 km at ℓ ≤ 1) | 0–6 |

**Inner cone (codec input), per anchor (t, y, x):** the anchor's own 3 × 3
patch at lag 0 for every channel (today's tokens, unchanged), plus for each
channel and each lag 1…L_in = 6 a sunflower of `slots(r_in)` dots on the
ground-circle of radius r_in(ℓ) (`temporal.spiral_offsets`, aspect 0.71,
ramp 0.5 — the E-026 geometry, so the two stencils sample bearings the same
way), with `slots(r) = clamp(round(24·(r/900 km)²), 6, 24)`. Depth: the `rg_*`
channels are live in one pentad per month, so their dots are the anchor column
at the live bins inside the 6-lag window (0–2 tokens per channel). Token
count per anchor, computed by `cone.budget()` at 30° N for the 42-channel r3
list: 42 patch tokens + 706 dots (family A 32, B 512 = 4 surface channels × 80
+ 32 rg × 6 anchor-column bins, C 162) = **748 tokens**, latitude-independent
(no two sunflower points round to the same cell at 0.25°); a Perceiver with 64
latents attends to them at O(N·K). Reach per lag for family B is 129.6, 259.2,
388.8, 518.4, 648.0, 777.6, 907.2 km (ℓ = 0…6; the 100 km floor never binds);
family C drops from 500 km at ℓ ≤ 1 to 388.8 km at ℓ = 2 — the L-shape.

**Outer cone (stage-2 stencil), per lag k = 0…143 pentads:** the anchor's own
embedding history (all k) plus a spiral of neighbours between
`r_lo(k) = r_in(k)` for k ≤ 6 (else 0) and `r_hi(k) = min(4444, max(111,
0.3 m/s × 5 d × (1 + k)))` km — i.e. the E-026 spiral with a *lag-dependent*
radius range, so reach grows with lag (prediction 3 of the cone slides) and the
near field the codec already read is excluded. Union of the two = every (Δx,
ℓ) in the family-B cone up to 144 pentads; overlap = the anchor column only.
`ml/cone.py::coverage_report()` asserts both facts on the grid — and one
consequence is structural, not a corner case: **the outer spiral is empty for
k ≤ 6**, because r_lo(k) = r_in(k) is the same formula as r_hi(k) there, so
stage 2 keeps only the anchor column at the lags the codec already read in
full; its first ring is at k = 7 (111 → 1,036.8 km) and it reaches 4,444 km at
k = 143 (3,432 stencil cells over the 144 lags against the cylinder's
145 × 144 = 20,880).

**Why L_in = 6 (30 d) and 0.3 m/s — the velocity-in-the-embedding choice.** A
displacement is resolvable once it exceeds one cell (0.25° = 28 km
meridionally, 21 km zonally at 40° N). At pentad cadence that is 0.06 m/s per
lag: eddy advection (0.1–0.3 m/s) moves 2–6 cells per pentad and is resolved at
lag 1; Rossby propagation (0.03 m/s, Chelton et al. 2011) needs ~2 pentads
per cell, so 6 lags give 3 cells — resolved; the atmospheric forcing
decorrelates in 1–2 pentads (family A, lags 0–1); SST anomalies persist 3–6
months (Frankignoul & Hasselmann 1977), which is longer than the inner window
and is deliberately left to stage 2. Thirty days is therefore the shortest
window in which every fast process on the tensor has moved by at least one
cell and the slow ones by three; 0.3 m/s admits the eddy field and the mean
boundary current (whose 1 m/s core is covered up to 2 lags) without paying
for a 4,000 km inner reach. Velocity beyond this — multi-month memory,
teleconnections, the far field — stays in stage 2 by design.

## 3 · The codec: ConeMAE

`ml/cone_codec.py`, a sibling of `PixelMAE` (not a modification — every
archived checkpoint must stay bit-identical, ml/CLAUDE.md §1 recipe rule).

- **Tokens.** One per (channel, dot): value projection (1 value, or the 3 × 3
  patch + observed flags for lag 0, exactly `PixelMAE`'s `val_proj`), plus
  `chan_emb`, plus a coordinate encoding — Fourier features of (Δy_km, Δx_km)
  on a signed-log scale, of lag in days (log), and of the `rg_*` depth in dbar
  (log) — plus the existing `mask_tok` / `miss_tok` semantics: a dot the DATA
  never observed is `miss`, a dot WE hid is `mask`. One non-maskable context
  token (sin/cos season, lat, lon) as today.
- **Encoder.** 64 learned latents cross-attend to the token set, then 6
  self-attention blocks over the latents (d 256, 8 heads); attention-pool the
  latents to `z ∈ R^{d_z}`, d_z 32 (the tier's current bottleneck, so the
  head-side comparison is at equal width).
- **Decoder.** `PixelMAE.query`'s form generalised: a query token (channel,
  Δy, Δx, lag, depth) cross-attends to a memory and predicts the value;
  Gaussian NLL head (mean + log-variance) so the model can say "unknowable",
  plus the plain MSE as the logged reconstruction metric for comparability.
  **The headline loss reads z alone** (`decode_from_z`): with the 64 × 256
  latents in the memory of the only loss, z would be optional and the
  bottleneck would carry nothing while every curve looked healthy — a
  nameable degeneracy (ml/CLAUDE.md §4.9b), closed rather than ranked
  improbable. A second decode from [z + latents] is an auxiliary term at
  weight 0.25 (`--aux-latent-w`; 0 gives the z-only codec exactly).
- **Masking, mixed per batch** (`ConeMAE._masks`; hidden-dot queries subsampled to 256 per anchor so the decoder's cost does not scale with N): (i) channel drop — hide a whole channel at all
  dots (today's scheme; `cur_*` at 50 %); (ii) lag-band drop — hide lags
  0…ℓ₀ and predict them from the older lags (forecasting inside pretraining);
  (iii) future dots — queries at lag −1 and −2 pentads (values from the
  archive, inside the training pool only); (iv) sector drop — hide a bearing
  sector; (v) anchor reconstruction — always predict the anchor's own
  channels. Loss weights per family follow the AIFS-ocean practice (slow
  fields up-weighted).
- **Pool discipline.** A training anchor is admitted only if *every* dot in
  its inner cone and every future query lies inside the training bins
  (`--holdout-scope window`, c25f6ff's rule generalised to the dot set); the
  sampler self-certifies by brute force before training, as E-059 did.

## 4 · Data — prepared first, as asked

1. **`family4_na025_pentad_r3`** = r2 (40 channels) + `cur_u` (40), `cur_v`
   (41): the binned mean GLORYS12 components that `cur_speed` is already the
   hypotenuse of, so no new download — `build_family4.py --rev r3` reads the
   same `uo`/`vo` bins. Direction is what the cone needs (upstream tilt) and
   what H1 supervises; a magnitude cannot supply either. Channels 0–39 keep
   their published indices; r2 stays buildable and unchanged.
2. The 32 `rg_*` channels stay in the tensor for this version (they are the
   depth dots); moving them to the coarse sidecar is REBOOT step 7.1 and is
   orthogonal.
3. Nothing else. Reboot step 7: clean before adding; every other rung of the
   survey's input ladder waits for H2's answer.

## 5 · Protocol (the reset's, unchanged)

Window-scope pool · interspersed development holdout for now, terminal
holdout (train ≤ 2020, test 2021–2024) once the terminal codec exists ·
rolled skill is the verdict, probes are diagnostics · per-lead `acc` and
`msss_clim` against climatology, persistence, damped persistence and the LIM ·
trained/held-out longitudes reported separately · block-bootstrap intervals
over (year, start) · early stopping at the held-out minimum · **five seeds**
per arm at the 7.6M tier (≈ 1.5–2 h each) · every `#NNN` with its summary.

## 6 · Steps

1. ✅ **Plan, spec, deck** (this file; survey deck slide 34 "Two stencils, one
   cone"; slide 25 marked superseded).
2. ✅ **Data**: `build_family4.py --rev r3` (+ `cur_u`, `cur_v`), recipe
   `ml/recipes/f4r3-cone-5M.json`, test.
3. ✅ **Cone geometry**: `ml/cone.py` — families, `reach_km`, `slots`,
   `inner_dots`, `outer_spiral`, `coverage_report`; `tests/test_cone_geometry.py`
   pins exact identities (union = cone, overlap = anchor column, budget
   arithmetic, the deck's 851-vs-12,816 worked example reproduced).
4. ✅ **Sampler + codec + trainer**: `ml/cone_sampler.py`, `ml/cone_codec.py`,
   `ml/train_cone.py --smoke` on a synthetic tensor (CPU, minutes);
   `tests/test_cone_smoke.py`.
5. ☐ **Dispatch the codec** (fresh box, r3 tensor built on first use), 20k
   steps, held-out reconstruction curve recorded; then the velocity probe (H1)
   against the snapshot codec — cheap, decides whether to spend on H2.
6. ☐ **Stage-2 arms** (H2/H3): cone-z + annulus stencil · cone-z + full
   cylinder · control (run-415 z + cylinder), five seeds each, ≤ 5k steps,
   selection at the held-out minimum, the #516 battery, the LIM null in both
   embedding spaces. The annulus arm needs a lag-dependent stencil in
   `ml/temporal.py` (`cone.outer_spiral(lat, k)` per window position) — the
   one existing-file change this experiment asks for, to be made as its own
   commit with the E-026 spiral arm as the bit-identity control.
7. ☐ **Workflow wiring** before step 5 can run on a box: `ml-train.yml` must
   learn `family4_na025_pentad_r3` (the `tensor` input's legal values, the
   family-4 build branch, the `--rev r3 --sst-dir` case with `seed_sst`, the
   wind and `base025_na` seed lists, and — once built once — a pinned sha +
   chunked `data-cache-v1` asset), and `train_cone.py`'s flags need a
   `$RECIPE_<KEY>` each or a `window:` encoding (the 25-input ceiling).
   `tests/test_e034_family4.py` check 14 and `tests/test_family4_tensor_pull.py`
   pin the rev list and want an r3 case in the same commit.
8. ☐ Write up in EXPERIMENTS.md at dispatch (hypothesis first), OVERVIEW row,
   expectations ledger.

## 6b · What the CPU smoke measured (a pipeline check, not a result)

`python3 ml/train_cone.py --smoke`: synthetic 120 × 40 × 56 × 8 tensor with a
planted shear flow whose velocity is *not* readable from any single frame
(R²(velocity | lag 0) = 0.0002, | lags 0–1 = 0.906), 200 steps at batch 32,
253,538-parameter geometry, 39 s on CPU. Pool certificate 0 violations in
4,096 anchors (window scope, bins t−6…t+2). Held-out NLL 2.037 → 1.753. The
velocity probe (ridge from frozen z, `cur_*` dropped at encode, 5
contiguous-time folds via `probe_kfold.kfold_r`) reads cur_u R² **+0.073** for
the cone codec against **−0.015** for the snapshot ablation (`--L-in 0`), same
sign at seeds 1 and 2. That is the H1 mechanism working on a toy; it says
nothing about the ocean tensor.

## 7 · Cost

Codec: ~1,000 tokens × 64 latents, ~5M params — 20k steps at batch 256 is
~2–3 h on an A100-class box (the sampler, not the network, bounds it; the
inner-cone gather is 40 × 1,000 random reads per anchor from a memmapped
tensor, so the loader must batch anchors by tile). Heads: 15 runs × ~2 h.
Whole experiment ≈ 40–50 box-hours, ≈ $15–25 at the fleet's rates.

## 8 · What this is not

Not a claim about the far field (stage 2 keeps it); not a new tensor family
(r3 appends two channels); not a change to any archived checkpoint or to
`PixelMAE`; not a result — nothing here has been trained. The pixel-year arm
of the survey deck (slide 25) is the special case r_in = 0, L_in = 72 and is
superseded by this design.

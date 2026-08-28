# Reproducing #503 (E-051-roll): the full artefact chain, from source data to the rolled number

Written 2026-08-28 for an independent researcher (or model) with no access to
this project's sessions. Every number here was read from a named artefact, not
from memory; where a value is recorded in a machine-readable file, that file
is named so you can check it. The chain is:

```
source datasets ──► family4_na025_pentad_r2 (the tensor)
                      │  anomaly transform + splits
                      ▼
        stage 1: PixelMAE codec  = run-415  (37.976M params, FROZEN after)
                      │  embed every (bin, ocean pixel)
                      ▼
        Z  [3142, 86698, 32] float16  = Z_8b639abe36_37e146384b.npy
                      │
        stage 2: TemporalTransformer head (206.659M params)
                      │  trained 0→200k (E-051, JAX/TPU), 200k→400k (E-054a)
                      ▼
        head-weights-e051-398k-xl144zn-pentad-s0.pt  (step 398,000, FROZEN)
                      │
        #503: ml/rollout_spatial.py rolls it 365 days forward   ◄── THIS RUN
                      ▼
        day-matched corridor AUC 0.944 (uncertified — see §8)
```

Nothing trains in #503 itself: it is an evaluation of two frozen models. So
"reproducing #503" splits into (a) reproducing the EVALUATION from the
published frozen artefacts — cheap, deterministic, and the place to start —
and (b) reproducing the TRAINING of each artefact from raw data. Both are
specified below.

Repository: `github.com/blauewelt/earth`, all code under `ml/`. The commit
that ran #503 is `3622434` (main as of 2026-08-27 23:06Z; the job checks out
main at start, and #503 started 23:55Z). Torch runs use the Vast.ai image
`pytorch/pytorch:2.6.0-cuda12.4-cudnn9-runtime`; the TPU training used
JAX 0.6.2 on a Cloud TPU v5litepod-4.

---

## 1 · The data: `family4_na025_pentad_r2`

**What it is.** A dense tensor `X[T=3142, H=281, W=481, C=40]`, float16, of
the North Atlantic at 0.25° and PENTAD (5-day) cadence. Builder:
`ml/build_family4.py --rev r2` (its module docstring is the authoritative
spec; the r2 revision appends SST as channel 40).

**Grid.** lats 0.0..70.0 in 0.25° steps (281 rows, south first), lons
−100.0..+20.0 (481 cols). Samples sit ON multiples of 0.25 (point-aligned,
not cell centres); the build asserts this against `base025_na.npz`.

**Time axis.** Fixed 5-day bins counted from the epoch **1982-01-01**:
`bin_index = floor(days_since_epoch / 5)`. 3,142 bins cover 1982-01-01 →
2024-12-31 (43 years, 73.0485 bins/year). The same epoch and formula are used
by `ml/build_truth_pentad.py` (labels) and `ml/aggregate_cadence.py`
(aggregation), so states and labels share bins by construction.

**Channels (40), in fixed order** — indices 0..38 are family-3's, imported
from `ml/build_family3.py` so there is one definition; index 39 is r2's
appended SST:

| group | channels | source and per-bin aggregation |
|---|---|---|
| base (3) | `cur_speed`, `log_mld`, `ssh` | CMEMS GLORYS12 1/12°→0.25 pentad aggregation. `cur_speed = hypot(mean_uo, mean_vo)` from the BINNED components (a mean of magnitudes is not the magnitude of the mean); `log_mld = log10(mlotst)`. Coverage 1993-01 onward; pre-1993 is `missing`. |
| rg (32) | `rg_t`/`rg_s` at 16 pressure levels (0–2000 dbar) | Roemmich–Gilson Argo climatology-anomaly product, monthly, native 1°, coverage-preserving bilinear to 0.25°. **One live pentad per month** (the bin containing the 15th); the other ~5 bins carry the `missing` token — deliberately, because missingness is information to this architecture, and forward-filling would claim observations on days that had none. Coverage 2004 onward. |
| wind (4) | `tau_x`, `tau_y`, `tau_x_std`, `tau_y_std` | NCEP R1 daily gaussian-grid wind stress, sign flipped so positive = stress ON the ocean, bilinear to 0.25°. Mean over the bin's days; the `_std` pair is the WITHIN-pentad standard deviation of the dailies (a storminess measure). Covers the whole axis from 1982. |
| sst (1) | `sst` (channel 39, 0-indexed; "channel 40" in prose) | OISST v2.1 daily 0.25° (already on this grid via `ml/fetch_sst_na.py`), NaN-aware mean of the bin's days. Live in ~100% of bins over the whole axis. r2 only. |

**Ocean inventory.** 86,698 window ocean pixels ("ocean" = any channel ever
observed); 281×481 grid cells total. Truth series attached to the axis:
`rapid` (RAPID 26.5°N overturning, 1,459 pentad rows) and `truth_fc`
(Florida Current cable, 2,490 rows) from `ml/build_truth_pentad.py`.

**Storage & identity.** float16 (the fields are ~N(0,1) after the transform
in §2, where fp16's ~3 digits are below observational error), 33.1 GB dense.
The published copy is chunked on the `data-cache-v1` GitHub release as parts
`aa ab ac` and identified by
`sha256 = 37e146384b6f622fefe3c7e18ad9bab0389c9538be79536899fe8729bb2d0826`.
Every consumer verifies this sha before use — two boxes that built their own
tensors once moved a probe by 0.041 at fixed seed (the "box effect"), which
is why the sha is part of the experiment's identity.

---

## 2 · Preprocessing: the anomaly transform and the splits

**The anomaly transform** (`ml/trainprobe.py:anomaly_transform`, the ONE
implementation — `train.py --anomaly`, `temporal.py` and the roll all call
it) turns dynamic channels into z-scored anomalies in three sequential
passes over the time axis:

1. detect dynamic channels (per-(t,c) spatial means) and accumulate each
   channel's per-pixel **monthly climatology** over TRAIN bins only (masked
   sum/count per (pixel, channel, month-of-year));
2. write `anomaly = X − clim` and accumulate the anomaly's mean/variance
   over the valid train pool (per channel, float64 accumulators —
   load-bearing at float16, where a fused float16 variance overflows to inf);
3. write `(anomaly − mu) / (sd + 1e-6)`.

After this, **the climatology IS the zero forecast**: a channel's
climatological prediction is exactly 0 in tensor units, which is what makes
the roll's `msss_clim` denominator simply `mean(truth²)`.

All 40 channels are dynamic on this tensor (`40/40 dynamic channels` in every
run log). Statistics are taken over train years and ALL longitudes (see the
holdout note below).

**Splits.** Blocked, never random:

- **Held-out years: 2009, 2017, 2023** (`--holdout-years` default). At pentad
  that is 3 × 73 = 219 held-out bins of 3,142. These bins are excluded from
  codec training, from the transform's statistics, from stage-2's training
  pool, and are where the roll starts.
- **Held-out longitudes: NONE.** The historical −45..−25°E block was retired
  on 2026-08-19 (owner's decision; the per-pixel anomaly transform already
  removes location-memorised climatology, and the block cost 25% of training
  pixels). Encoded as `holdout_lon: "0,0"` — the empty half-open interval,
  bit-identical to "none" but safe for legacy eval scripts that `float()`
  the field. The stage-2 pool likewise trains on every longitude
  (`train_lon_hold: none`).

---

## 3 · Stage 1: the codec (`run-415`, PixelMAE, frozen thereafter)

**Architecture** (`ml/model.py:PixelMAE`): a masked autoencoder over ONE
pixel's channels. Each of the 40 channels is one token; missing channels
enter as explicit learned "missing" tokens (absence is information, not
padding); masked channels enter as "mask" tokens (distinct from "missing").
A transformer encoder (d_model **512**, **12** layers, **4** heads,
norm-first) reads the 40 tokens plus a CLS; a bottleneck maps CLS to
**z ∈ R^32** (d_z 32) — THE embedding. A neural-field-style MLP decoder
(d_dec **256**) is conditioned on `(z, channel-id, Δlat, Δlon, Δt)` and
predicts a single channel value at that offset. `patch 1`: the encoder sees
one pixel only (no spatial receptive field). **37,976,465 parameters.**

**Objective** (per training step; `ml/train.py:step_loss`):

- draw a batch of 512 (pixel, bin) samples from the train pool
  (`obs_any & ~held-out-year`);
- mask each observed channel independently with probability
  `--mask-ratio 0.5`;
- **reconstruction term**: query the decoder at offset (0,0,0) for all 40
  channels; Huber loss, weighted `mask + 0.1·(observed & unmasked)`
  (`--rec-w-visible 0.1`) times per-channel weights (defaults uniform);
- **neighbour term**: per sample, one random offset from
  {±1 lat, ±1 lon, ±1 time}; query the decoder there for all channels;
  Huber on observed cells. This is what forces z to carry STATE rather than
  memorised seasonality — a plain autoencoder can ace reconstruction from
  the seasonal cycle alone;
- total: `loss = l_rec + 0.5 · l_nei`; AdamW; cosine LR from
  `--lr 3e-4` (train.py's default — the run set no override) to 0 over the
  step budget.

**The run** (workflow run #415, recipe `ml/recipes/f4r2-40M-nolonhold.json`,
records in `run-415.jsonl` on the `ml-metrics` branch): tensor
`family4_na025_pentad_r2`, `--anomaly`, batch 512, dispatched 200,000 steps,
**final recorded step 197,428** (the `max_minutes` scheduler re-fit; the
checkpoint's own `step` field is authoritative — always read the file, not
the dispatch). Final in-training eval: `z_mse_model 11.499` vs
`z_mse_persistence 19.875` (one-step, z-space), pooled RAPID ridge
`linear_r_deseas 0.575`. Single seed (torch default seeding; the codec run
did not set an explicit seed knob).

**Published artefact:** `run-415__pixelmae.pt` on the `model-checkpoints-v1`
GitHub release — an ordinary torch checkpoint with `model` (state_dict),
`args` (every flag, including `holdout_years` and `holdout_lon`, which
downstream consumers read to reproduce the transform), `d_z`, `chan`
(channel names), `norm`. **From here on the codec is frozen**; stage 2 and
the roll never update it (two-stage by construction, so codec and dynamics
improvements stay attributable).

---

## 4 · The embedding cache Z

Every (bin, ocean pixel) is pushed through the frozen encoder once:
272,405,116 forwards (3,142 × 86,698), producing
**`Z [3142, 86698, 32] float16`** = 17,433,927,552 bytes (16.24 GiB). The
pixel order is the tensor's ocean-pixel enumeration (row-major over the
grid, `ys/xs` arrays saved beside it).

Z is a pure function of (codec weights, tensor bytes), so the cache is keyed
by both: the published asset is
**`Z_8b639abe36_37e146384b.npy`** on the `embed-cache-v1` release —
`8b639abe36` = the codec weight hash, `37e146384b` = the tensor sha prefix —
chunked in 1.5 GiB parts (12 chunks). Consumers assemble it bounded by the
`.npy` header's own byte count (not "concatenate until 404", which once
produced a chimera of two caches) and verify shape/dtype before use.
Pulling this cache is what makes any two stage-2 runs twins rather than
lookalikes: the encoder is removed as a variable.

---

## 5 · Stage 2: the temporal head (E-051 + E-054a, frozen at step 398,000)

**Architecture** (`ml/temporal.py:TemporalTransformer`; the JAX mirror in
`ml/jaxport/models.py` is key-for-key identical and gated by a round-trip
test): a causal transformer over one pixel's embedding sequence.

- **Context K = 144** pentads (720 days ≈ 2 years) ending at t.
- **Spatial stencil = 145**: the centre pixel plus **144 spiral ring
  points**. Per step, the input token is the concatenation of the 145 cells'
  z (145 × 32 = 4,640 dims, missing cells zero-filled) plus the season token
  (sin, cos of the bin's true calendar month — `season_phase: month`),
  linearly projected to d_model.
- **The spiral** `spiral:111-4444-0.71-0.5`
  (`ml/temporal.py:spiral_offsets`): point k sits at bearing k × the golden
  angle (137.5078°) — the unique rotation whose prefixes never cluster — at
  radius `r = 111 + (4444−111)·f^0.5 km` on uniform f (Vogel/sunflower ramp:
  uniform density per unit area, so most points sit far out), with the
  meridional extent compressed to **0.71** of the zonal (elliptic; the
  aspect was measured from the tensor's own geostrophic |u|/|v|). Offsets
  are computed per pixel ROW in ground km (the zonal step stretches by
  1/cos φ), then rounded to grid cells.
- **Static context** per pixel: (lat, lon, the codec embedding of the
  pixel's static channels alone, and the 145 stencil observed-flags), added
  to every step. Learned positional embedding over the K axis
  (`k_max = 144`).
- Trunk: d_model **1024**, **16** layers, **4** heads, FFN 4×, norm-first,
  dropout 0. Output head: linear d_model → d_z, predicting **z_{t+1}** from
  the hidden state at every step (causal mask). **206,659,000 parameters**
  (`params_M: 206.659`; the count includes the fresh k_max-144 position
  table).

**Objective and pool.** MSE on z_{t+1} over all K steps of each window, in
z-units of the frozen codec. The pool is every (pixel, end-bin) window whose
bins are train bins: **240,933,742 windows** (86,698 pixels × ~2,779 usable
end-bins). Batch **256** windows. Validation: a fixed 4,096-window draw from
held-out bins; the monitor records `val_persistence 21.44621` (the MSE of
predicting z_{t+1} = z_t on that draw), which is the denominator of every
"ratio" quoted below.

**Input noise (znoise).** During training only, Gaussian noise
`N(0, 0.7²)` is added to the INPUT z (never the target). 0.7 is an ABSOLUTE
sigma; on this codec's z-scale it is 0.15116 × sqrt(val_persistence)
(recorded as `input_znoise_rel_pers` — always check this when the codec
changes; the same 0.7 is a 5.8×-different relative dose on other codecs).

**Gradient clipping.** Global-norm clip at **128.0** — added after run #423
diverged at these exact settings without it (grad-norm excursions to 13,052
that poison AdamW's second moment for ~1,000 steps). 128 = 15.5× the healthy
pentad norm; it clipped 0 steps in this run (`grad_clip_frac 0.0`
throughout).

**Training happened in two phases, on a Cloud TPU v5litepod-4** (the JAX
port `ml/jaxport/train_stage2.py`; launcher `ml/jaxport/tpu_train_s2.sh`;
node `e051-k144-full`; torch↔JAX equivalence is gated at 1e-7..1e-5 by
`tests/test_jaxport_*` and the backend is never pooled with torch numbers).
Both phases' configs are recorded verbatim as `stage2_config` lines in
`gs://earth-tpu-staging/runs/e051-k144-full/metrics.jsonl`:

| phase | steps | LR schedule | everything else |
|---|---|---|---|
| **E-051** (fresh) | 0 → 200,000 | peak **1e-3**, expdecay **halflife 40,000**, warmup 2,000, cooldown 0 | seed **0** · batch 256 · K 144 · stencil 145 · spiral as above · znoise 0.7 · grad-clip 128 · AdamW · `train_lon_hold none` |
| **E-054a** (continuation, same node name, exact resume incl. optimizer state) | 200,000 → 400,000 | **re-armed**: peak **4e-4**, expdecay **halflife 100,000**, warmup 2,000 | identical |

Validation trajectory (val_zmse / 21.44621): 0.0812 @ 20k → 0.0414 @ 100k →
**0.0330 @ 200k** (E-051's registered number) → **0.029808 @ 400k**
(0.63926 / 21.44621, E-054a). Single seed (§3b of `ml/CLAUDE.md`: the first
result at a new tier buys its own replication; no pentad pair exists at this
scale yet).

**The frozen artefact.** The trainer checkpoints every 2,000 steps and has
NO post-loop final save, so the durable final is step **398,000** (Δval
398k→400k ≈ 0.0001, immaterial and disclosed in the name). The JAX weights
were exported to a torch checkpoint via `ml/jaxport/convert.py:export_pt`
(loaders refuse on any missing/unexpected key, so a partial conversion is
impossible) and published as
**`head-weights-e051-398k-xl144zn-pentad-s0.pt`** (826.7 MB,
`model-checkpoints-v1`; verified on publish: step 398000, K 144, 1024×16,
seed 0, znoise 0.7). Its `args` carry every knob above, and the roll rebuilds
the model from those args — a dispatch cannot contradict them.

---

## 6 · The run itself: #503, an evaluation of the two frozen models

Dispatch (GitHub Actions `ml-train.yml`, `workflow_dispatch`; full 25-field
input record archived; plan `plan-503.json` on `ml-metrics`): stage `sroll` —
nothing trains (`temporal_steps 0`; stage-1 `steps` set to the codec's own
197,428 so the trainer proves "nothing to do"). The window string:

```
recipe:xl144-zn-pentad-nolonhold,
sroll:head-weights-e051-398k-xl144zn-pentad-s0,
ckpt:run-415__pixelmae.pt, horizon:73, starts:3, dumproll
```

`scripts/sroll_run.sh` pulls the tensor (sha-verified), the codec and the
head by name, pulls or rebuilds Z (it pulled the published cache), and runs
`ml/rollout_spatial.py` at commit **3622434**. Hardware: one RTX 4090
(Vast.ai, ~$0.32/h); measured pace 87.1 s per rolled step.

**Roll protocol** (`ml/rollout_spatial.py`; the cadence is DERIVED from the
tensor's own `bin_index`/`pentad_days`, never assumed):

- **Starts**: 3 per held-out year, deterministically strided through the
  year's 73 possible starts (every 24th, keeping the year's first). The nine
  context-end rows and their dates are recorded in the artefact:
  2008-12-30 / 2009-04-29 / 2009-08-27, 2016-12-28 / 2017-04-27 /
  2017-08-25, 2022-12-27 / 2023-04-26 / 2023-08-24.
- **Init**: true context — the K=144 window of REAL Z ending at the start
  row. Then autoregressive: predict ẑ_{t+1} for ALL 86,698 window pixels at
  once (the stencil gather reads the rolled state, so pixels interact; this
  is why the whole window must roll, not a region), slide the window,
  repeat. The season token at each rolled step comes from that bin's TRUE
  date. No noise is injected at eval (znoise is a training-only knob); the
  roll is deterministic.
- **Horizon**: 73 steps = 365.0 days, chosen to horizon-match the monthly
  archive (the closest integer; the default 12 would score 60 days).
- **Scoring**, at every horizon h, on observed cells only: decode ẑ through
  the FROZEN codec decoder at offset 0 into channel space and compare with
  the true channels. Per (scope, h): `msss_clim = 1 − mse_model/mse_clim`
  (and climatology = the zero forecast, §2), `msss_pers` vs persistence
  (x_{t+h} := x_t), `msss_damped`, anomaly correlation `acc`, amplitude
  ratio. Three nested pixel scopes: **gate** (rollout.py's historical
  600-random ∪ RAPID section, 864 px), **corridor** (the headline: pixels
  whose train-month mean `cur_speed` ≥ its 75th percentile over window
  ocean — threshold 0.2385 on this tensor — dilated 2 cells with a 3×3
  square, ∪ the RAPID section: 30,158 px), **window** (all 86,698).
- **The two AUCs, and which one is quotable.** `horizon_auc` is the
  unweighted mean of msss_clim over h = 1..73 — a function of THIS axis's
  lead sampling, mostly short leads, NOT comparable to monthly numbers.
  **`horizon_auc_daymatched`** averages the twelve pentad leads
  {6,12,18,24,30,37,43,49,55,61,67,73} = 30..365 days — the monthly
  archive's leads to within 2.4 d — and is the ONLY number that may be
  compared to an archived corridor AUC.
- **Transport read-outs**: `amoc_bands` (legacy pooled: ridge on the rolled
  section-mean states, truefit, three day-defined bands 5–90/95–180/185–365 d)
  and `amoc_bands_unpooled` (E-055: a learned softmax-attention pool over the
  266 section pixels, fitted on 1,240 TRAIN-only truth rows with all three
  holdout years excluded, then applied to the rolled states).
- **The replay battery** (the part #503 did NOT finish): a 20-year hindcast
  roll (context ends 2004-12, 1,461 steps) and a 20-year future roll from
  the record's end — the accuracy-vs-lead profile over rolls from very
  different start dates is what separates forward physics (decays with lead)
  from calendar memorisation (flat).
- **The validation gate**: at monthly cadence the evaluator must reproduce
  a pinned reference before scoring anything. **No pentad reference exists**,
  so the artefact records `gate {pass: null, skipped: true, certified:
  false}` with the reason in full — the number below is uncertified by
  construction, and the first published pentad roll is what would establish
  a reference.

Partial results ship incrementally (atomic writes with a top-level
`in_progress` key) to the run's `ml-live-503` branch, which is how the
numbers below were read while the run was still going.

---

## 7 · What #503 measured (read 2026-08-28 from its own partial)

Day-matched corridor AUC **0.944** (gate 0.943, window 0.950;
`*_trainlon` = parent exactly, since nothing is held out by longitude).
Controls: monthly champion 0.939; the K=24 pentad head's roll **−0.499**.
Transport bands (pooled / unpooled): 0.511/0.481, 0.591/0.568, 0.565/0.580.

**Skill against lead, corridor scope** (`msss_clim`):

| lead (days) | 5 | 10 | 15 | 30 | 90 | 150 | 245 | 365 |
|---|---|---|---|---|---|---|---|---|
| msss_clim | 0.971 | 0.960 | 0.736 | 0.949 | 0.942 | 0.938 | 0.946 | **0.946** |

---

## 8 · How to read the number — the caveats are part of the result

1. **Uncertified, twice over.** No pentad validation-gate reference exists
   (the file says so itself), and the replay battery did not complete inside
   the job's 40 h timeout (the long/future rolls are 2,922 of the 3,363
   steps and only write on completion).
2. **The profile is flat, which is the replay signature.** Skill at 365 days
   equals skill at 30, and `msss_pers` at 365 d is 0.966 on a z-scored
   anomaly field — no physical forecast does that. The suspected mechanism:
   a 365-day roll from a start in held-out 2009 walks into 2010, which is
   training data; holding out a year holds out its bins, not the future the
   roll travels into. The battery is the designed test of exactly this, so
   until a battery run completes, **0.944 is a corridor AUC awaiting
   certification whose own profile predicts it will not survive as forecast
   skill** — and if the battery confirms replay, the monthly 0.939 sits
   under the same question.
3. **+0.005 over monthly is a consistency, not a beat**: the corridor-AUC
   seed spread at this model tier is 0.0020 (pooled sd, 8 dof) and the
   programme's decision bar is 0.025; and both numbers are single-seed at
   this cadence.
4. **Cross-backend caveat**: the head was trained in JAX, the roll runs in
   torch through the converted checkpoint. The port is equivalence-gated at
   1e-7..1e-5, but the tiers are never pooled in statistics.

---

## 9 · Artefact index (everything needed, by name)

All releases are on `github.com/blauewelt/earth/releases`, public, no auth
(`curl -fsSL .../releases/download/<tag>/<asset>`).

| artefact | where | identity check |
|---|---|---|
| tensor `family4_na025_pentad_r2.npz` | `data-cache-v1`, parts `aa ab ac` | sha256 `37e146384b6f…2d0826` |
| codec `run-415__pixelmae.pt` | `model-checkpoints-v1` | open it: `args.steps`→197,428 recorded step; d_z 32; 512×12×4; d_dec 256; C 40 |
| embedding cache `Z_8b639abe36_37e146384b.npy` | `embed-cache-v1`, 12 chunks | header-bounded assembly → (3142, 86698, 32) float16, 17,433,927,552 B |
| head `head-weights-e051-398k-xl144zn-pentad-s0.pt` | `model-checkpoints-v1`, 826.7 MB | `args`: step 398000, K 144, 1024×16, stencil 145, seed 0, znoise 0.7 |
| codec training curve `run-415.jsonl` | `ml-metrics` branch | first line = full config |
| stage-2 training curve | `gs://earth-tpu-staging/runs/e051-k144-full/metrics.jsonl` (mirrored to the status page) | two `stage2_config` lines = the two phases verbatim |
| #503's plan / live partial | `ml-metrics/plan-503.json`; `ml-live-503/rollout_spatial.json` (branch deleted at run end; final artefact archived as `probes-503`) | `in_progress` key marks partials |
| code | repo @ `3622434` | `ml/build_family4.py`, `ml/train.py`, `ml/model.py`, `ml/trainprobe.py`, `ml/temporal.py`, `ml/jaxport/`, `ml/rollout_spatial.py`, `scripts/sroll_run.sh` |
| recipes | `ml/recipes/f4r2-40M-nolonhold.json`, `ml/recipes/xl144-zn-pentad-nolonhold.json` | their `_provenance` fields carry the field-for-field ancestry |

**Cheapest reproduction path** (no training): pull tensor + codec + Z + head,
run `ml/rollout_spatial.py` with `--horizon 73 --starts-per-year 3` over the
head — the roll is deterministic (no eval-time noise, fixed start rows), so
the numbers in §7 should reproduce exactly up to hardware non-determinism in
the codec decode. Next rung up: re-embed Z from codec + tensor (≈ 3 h on one
TPU v5e-4 or comparable GPU) and confirm the cache hash. Full training
reproduction: §3 (codec, ~19 h on one RTX 4090-class box) and §5 (head,
~45 h total on a v5litepod-4; note the two-phase LR schedule, the znoise,
the clip, and seed 0 — and that a fresh seed is a REPLICATE, not a
reproduction: at this cadence every number above is n = 1).

Raw source data, all public: CMEMS GLORYS12 (free Copernicus login), RG-Argo,
NCEP R1 (PSL), OISST v2.1, RAPID 26.5°N transports, NOAA/AOML Florida
Current cable. The build scripts fetch and document each.

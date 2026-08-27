# `ml/jaxport` — the models in JAX/Flax NNX

A framework-independent reference implementation of this project's models,
plus a one-way torch → JAX weight converter. The torch stack under `ml/`
stays operational: every published number comes from it, and nothing in the
operational tree imports from here. The point of a second, independent
implementation is that it can be scored against the first.

The package is `jaxport`, **not** `jax`: `ml/` goes on `sys.path` in this
repo, so a directory named `jax/` would shadow the jax library itself.

## What is ported (tier 1)

| here | mirrors |
|---|---|
| `models.PixelMAE` | `ml/model.py` — `encode` (patch 1 and patch>1) and `query` |
| `models.TemporalTransformer` | `ml/temporal.py` — causal trunk, stencil widening, direct heads |
| `models.SectionHead` | `ml/probe_head.py` — eval mode (dropout deterministic-off) |
| `models.TransformerEncoder` | `nn.TransformerEncoder` semantics: batch_first, norm_first, **relu**, no final norm |
| `convert.codec_from_ckpt_jax` | `ml/model.py:codec_from_ckpt`, same `.get()` defaults |
| `convert.load_*` | torch state_dict → NNX, refusing any missing or unconsumed key |

## What is ported (tier 2, first slice)

| here | mirrors |
|---|---|
| `embed.embed_everything_jax` | `ml/temporal.py:embed_everything` — same signature, same [T,P,d_z] float16 output, minus the disk cache/resume plumbing |
| `embed.gather_px_np` | `ml/model.py:gather_px` — numpy-native so the data path needs no torch; pinned against the original by the test |
| `score_section_probe.py` | a runnable driver reproducing `probe_kfold`'s rapid ridge with a swappable encoder backend |

The driver IMPORTS `anomaly_transform`, `section_of` and `kfold_r` from the
operational modules rather than copying them, so `--backend jax` and
`--backend torch` differ in exactly one thing: which encoder produced `Z`.

## What is ported (tier 2, rollout slice)

| here | mirrors |
|---|---|
| `roll.roll_step_jax` | `ml/rollout_spatial.py:roll_step` — the stencil gather, the byte-budget chunking, `pred[:, -1]` |
| `roll.decode_all_jax` | `ml/rollout_spatial.py:decode_all` — `codec.query` at every (pixel, channel) at offset 0, numpy out |
| `roll.roll_forward_jax` | the twelve-step feedback loop of `main()`: slide the window AFTER the forward, append the season token of the row just predicted |
| `roll.roll_chunk_rows` | the same byte budget, named so the test can pin it and the driver can print it |
| `score_gate_roll.py` | a runnable driver that reproduces the gate roll end to end with `--backend {jax,torch}` |

Everything the roll is *scored with* is imported, never copied:
`corridor_pixels`, `dilate8`, `StdMonths`, `ar1_train`, `month_feats`,
`new_sums`/`accumulate`/`skill_block`, `rapid_section`, `build_stencil`,
`stream_stats`, `build_slab`. A backend swap therefore moves the backend and
nothing else — the same property that made the embedding slice's 5.7e-8
readable.

Three things worth knowing before reading the code:

- **`amp` REFUSES.** `--amp` on the torch path is an fp16 autocast. Honouring
  it here means transcribing torch's per-op casting rules, which is a parity
  exercise of its own; `roll_step_jax(..., amp=True)` raises `AmpUnsupported`
  rather than rolling at fp32 under a flag that says fp16 (`ml/CLAUDE.md`
  §0.2).
- **The jitted forwards live at MODULE scope**, unlike `embed.py`'s per-call
  factory. An embedding pass jits once and calls its function many times
  inside one call; a roll calls it once per `roll_step_jax`, ~234 times per
  head, so a per-call factory would recompile every step. `clear_jit_cache()`
  is the release valve for a process that rolls many heads.
- **Z and `Zstat` are COMMON INPUTS to both backends**, deliberately. They are
  embeddings, and the embedding path has its own gate (G2′). Re-measuring it
  inside G3 would fold two independent differences into one number and G3
  would stop isolating the rollout.

## What is ported (tier 3, stage-1 only)

| here | mirrors |
|---|---|
| `train_stage1.py` | `ml/train.py`'s stage-1 path: masking recipe, `loss_rec` + `loss_nei` at the same weighting, cosine (+ `--lr-floor` / `--lr-decay-steps`) in optax, the `--max-minutes` refit, the collapse guard, light probes, milestone checkpoints and resume, `--time-block`, and `metrics.jsonl` with THE SAME KEYS |
| `convert.export_pt` | the REVERSE direction — NNX → a `.pt` blob the UNCHANGED torch stack loads (`JAX_PORT.md` §1b: a TPU-trained codec is scored by the torch eval ladder, not by a second scoreboard) |
| `models.PixelMAE(k_time=…)` | `ml/model.py`'s E-047 block codec: the `k_time × C` cell grid, `time_emb`, and `query`'s own `q_time` embedding with `tpos` REQUIRED |
| `tpu_train.sh` | a real training launch, modelled on `tpu_smoke.sh` — periodic checkpoint/metrics shipping to `gs://<bucket>/runs/<node>/`, resume from the bucket, a PROGRESS watchdog (no new checkpoint object for 90 min → reap) and a 30 h hard cap |

Stage-1 features `ml/train.py` has and this deliberately does not:

- **`--eval-every` (the FULL probe) REFUSES.** Its second half trains a mini
  `TemporalTransformer` inside the probe, which is stage-2 machinery. Running
  the light probe under the full probe's flag would put a different estimator
  into the archive under a name that already means something.
- **`plot_run`** — a matplotlib rendering, not an arithmetic step.
- **warmup and gradient clipping exist here as `--warmup-steps` /
  `--grad-clip`, both DEFAULT OFF**, because `ml/train.py` stage-1 has neither
  and the default trajectory must be the torch one. There is no EMA and no
  gradient accumulation on either side.

## What is ported (tier 3b, the stage-2 trainer)

| here | mirrors |
|---|---|
| `train_stage2.py` | `ml/temporal.py`'s stage-2 path: the train/val window pools (`ok_t` ∧ `ok_p`, and the two-masks rule), the whole-window MSE objective, `--input-znoise`, `--grad-clip`, the four schedules out of `make_sched`, `--K` with `k_max = K`, `--time-stride`/`--time-offset`, the block-z axis adoption, milestone checkpoints, resume, the in-training pooled rapid probe, and `metrics.jsonl` with THE SAME KEYS |
| `convert.export_temporal` / `export_temporal_pt` | the head's half of the two-way converter — a `.pt` the UNCHANGED torch eval scripts roll |
| `tpu_train_s2.sh` | `tpu_train.sh`'s lifecycle (self-reap, progress watchdog, bucket resume, apt-lock fix) plus the staging stage 1 does not need: the tensor AND the Z |
| `models.py` `film=`/`eps_dim=` + `train_stage2.py` `--fgn-eps`/`--fgn-val-members` | E-057's FGN mode (`ml/temporal.py`'s `_CondLayer` FiLM on norm1/norm2 + `fair_crps2`, two forwards/step, chunked M-member monitor, same telemetry keys); certified against torch by `tests/test_fgn_jax.py` (forward parity ≤3.5e-7 rel with injected ε, CRPS parity ≤1e-6, flag-off bitwise); spec + phase 2 (ensemble roll) gating: `ml/plans/FGN_JAX_PORT.md` |

**Why it exists.** Chris's span-fixed ladder (E-045.x) holds the context span
at two years and grows the frame count as the step shrinks — `--K` 48, 72, 144.
Attention is quadratic in K, so stage-2 compute on that ladder is ~K²-heavy,
which is exactly where the measured TPU advantage (4.5–8×) applies. First use:
an E-045.1-class arm re-run on TPU as the cross-framework twin.

**A fresh head is initialised by CONSTRUCTING THE TORCH MODULE under
`torch.manual_seed(seed)` and converting it**, and that is a measurement, not a
preference. Flax's defaults are not `nn.Module`'s — `nn.Embedding` is N(0, 1)
and `nnx.Embed` is std ≈ 1/√d_model, 5.7× smaller at d_model 32 — and in a
causal transformer whose only sense of *where in the window* it is comes from
`pos`, that is not cosmetic. Measured at 300 toy steps on an identical Z, an
identical pool and an identical schedule: the flax-initialised head read
model/persistence **1.46 / 1.87 / 1.68** at seeds 0/1/2 against the torch
trainer's **0.71 / 0.87 / 0.77** — systematically ~2× worse at every seed. On a
cross-framework twin that would have been read as "the framework", which is the
box effect `ml/CLAUDE.md` §3b warns about, manufactured by a default nobody
chose. Matching the STREAM rather than the distribution means a JAX run and a
torch run at the same seed begin from bit-identical weights, and init stops
being a variable at all.

Stage-2 features `ml/temporal.py` has and this deliberately does not — each
REFUSES rather than being silently absent, so a dispatch string copied from a
torch arm fails loudly instead of training something else:

- **`--input-quant` — the KNOWN GAP.** The quantizer is part of the model's
  CONTRACT (`ml/rollout_spatial.py` re-applies it at roll time from
  `input_quant_sigma`), so a head trained without it and labelled with it would
  be rolled through a grid it never saw. A-arm parity on quantized inputs waits
  for this; train quantized arms on the torch stack.
- **`--unroll`>1, `--unroll-wide`, `--unroll-probs`, `--direct`** — each feeds
  predictions back and each is its own experiment.
- **`--target-bins-argo`, `--season-dropout`** — E-044c pool/regulariser knobs.
- **the full probe ladder (`probe_kfold` / `probe_head`) and the ROLL.** They
  are eval and they run unchanged under torch on the exported head, which is
  `JAX_PORT.md` §1b's cheap validation direction. `rapid_probe_kfold` is
  therefore ABSENT from the results file rather than present and empty.

The exported head carries **no `opt`/`sched`**, deliberately: optax state is not
torch Adam state, mapping it is out of scope (§3.3), and
`--resume-temporal`'s refusal on a blob missing them is correct for this
artefact. The resumable state is the sibling `.npz`.

### Staging the Z on a node, and a bug in the published cache

`tpu_train_s2.sh` takes either a `Z_ASSET` name (pull the published chunks from
`embed-cache-v1`) or nothing (embed on-node from the codec asset). It assembles
the chunks **bounded by the `.npy` header, not by the first missing chunk**,
and that is a fix rather than a refinement.

Measured against `embed-cache-v1` on 2026-08-24:
`Z_8b639abe36_37e146384b.npy` (the pentad cache for run-415's codec) has
**twelve** chunk assets, of which `.af` is a short 852,643,840 B **in the middle
of the run**. Chunk `aa`'s header declares `(1571, 86698, 32) float16` =
8,716,963,840 B = exactly `aa..af`; `ag..al` are the orphaned tail of the
12-chunk `(3142, 86698, 32)` cache that `ml/EXPERIMENTS.md` records #427
pushing (17,433,927,552 B). `ml/embed_cache_sync.py:pull` concatenates until a
fetch 404s, so today it produces 16,713,707,392 B — matching neither publish —
and `verify()` correctly discards it. **That cache is currently unpullable by
the operational path, and a puller pays the download and then the ~8.5 h
rebuild.** `ml/embed_cache_sync.py` was NOT changed here: it is on the
operational path, and a fix to it belongs to its own change with its own test.

`train_stage1.py` **imports** the shared numpy plumbing (`anomaly_transform`,
`ridge_r`, `rapid_section`, `light_rows`, `lon_holdout_mask`, `fit_schedule`,
`obs_any_chunked`, `pool_idx`, `LazyPixels`) rather than copying it, per
`ml/CLAUDE.md`. Those helpers are pure numpy but live in modules that import
torch, so **this driver needs a CPU torch wheel** — the same wheel
`jaxport.convert` already needs to read a `.pt`. `models.py` and `convert.py`
themselves still import no torch at module scope.

Not ported yet: `MultiDec` (`ml/recon_decoder.py`), the head fold-fit, the
transport read-out (`fit_ridge`/`read_sv`), the band correlations, and the
STAGE-2 trainer.

## Running the parity tests

```
python3 tests/test_jaxport_parity.py      # tier 1 — CPU, fp32, exits 0/1
python3 tests/test_jaxport_embed.py       # tier 2 embedding path, synthetic
python3 tests/test_jaxport_roll.py        # tier 2 rollout path, synthetic
python3 tests/test_jaxport_train.py       # tier 3 gates G4a-G4e, synthetic
python3 tests/test_jaxport_train_s2.py    # tier 3b gates G5a-G5e, synthetic
```

None of them needs the tensor or a checkpoint.

`test_jaxport_roll.py` iterates TWELVE steps rather than checking one, because
a roll feeds its own output back into its input window: tier 1 already pinned
the head's single forward, and what a roll adds is compounding. It covers
stencil 1 and stencil 9, pins the `[n,S,K,dz] → [n,K,S·dz]` gather against
`temporal.gather_stencil` itself (centre slot first, missing slots
zero-filled), pins the chunk byte budget, and checks that `amp=True` refuses.

`test_jaxport_train.py` is the tier-3 gate set: **G4a** loss parity (identical
weights, batch and mask → `loss_rec`/`loss_nei` agree to 1e-5 at patch 1,
patch 3 and k_time 7), **G4b** one-step parity (plain SGD lr 1e-2 →
max |Δweight| < 1e-6; AdamW → 2e-5 over all parameters and 1e-6 over the
entries where |g| > 1e-6, because Adam's first update is ~sign(g) and is
therefore hypersensitive where a gradient is near zero), **G4c** 300 toy steps
of the REAL `ml/train.py` against 300 of `train_stage1.py` (a band, not an
equality — the RNG streams cannot match), **G4d** the export round trip
(torch→jax→torch state_dicts IDENTICAL; a JAX codec exported to `.pt` and
re-encoded under torch matches to 1e-5) and **G4e** k_time=7 forward parity
plus the converter's refusals across the k_time boundary.

`test_jaxport_train_s2.py` is the tier-3b gate set: **G5a** loss parity with
the noise INJECTED as an array rather than drawn (two RNGs cannot be made to
agree, so `apply_znoise` takes the perturbation as an argument and both
frameworks get the same one), at stencil 1 and 9, plus the window gather pinned
elementwise against `ml/temporal.py:gather_stencil`; **G5b** one-step parity —
the GRADIENTS to 1e-6, plain SGD to 1e-6, and AdamW stated four ways because
its first update is ~sign(g) and is therefore bounded by lr and hypersensitive
below it; **G5c** 300 toy steps against the REAL `ml/temporal.py` as a
subprocess on the SAME Z, so the encoder, the pool and the eval draw are all
held fixed and the persistence baselines can be required to agree rather than
merely banded; **G5d** the torch-format head round trip plus grad-clip and
znoise pinned by targeted checks; **G5e** the block-z axis adoption against
`ml/temporal.py`'s own block branch. Measured 2026-08-24, CPU, fp32: G5a
**4.8e-7**, G5b grads **7.5e-8** / SGD **1.5e-8**, G5c ratio **1.238** inside
the pre-registered band [0.5, 2.0] with the persistence baselines agreeing to
**6.9e-8** relative, G5d **1.9e-6**, G5e identical labels, fused `t_hold` and
remapped RAPID rows.

Six checks: the shared encoder layer (plain and causal), PixelMAE at patch 1
and patch 3 (including the float16-widening path), TemporalTransformer at
stencil 1 and 9 with direct heads plus a causality check, SectionHead with
and without pre-pooling blocks, and the converter's refusal contract.
Measured agreement is ≤ 1e-6, well inside gate G1's atol 1e-4.

## Scoring the embedding path against the archive

The point of a second implementation is that it can be held against a number
nobody chose afterwards. The driver reproduces `probe_kfold`'s rapid ridge end
to end and prints its result beside the archived control for the same codec on
the same tensor (`probes-140.json`: r 0.627, CI [0.503, 0.735], n 240, RMSE
2.17 Sv):

```
python3 ml/jaxport/score_section_probe.py --data <tensor>.npy \
    --meta <meta>.npz --ckpt <codec>.pt --backend torch \
    --out /tmp/probe_torch.json --z-out /tmp/Z_torch.npy
python3 ml/jaxport/score_section_probe.py ... --backend jax \
    --out /tmp/probe_jax.json --z-out /tmp/Z_jax.npy \
    --compare-z /tmp/Z_torch.npy
```

Run the torch backend FIRST. If it does not itself land on the archived
number, the reproduction path differs from `probe_kfold`'s and the JAX number
would be scored against the wrong thing — that is a finding, never a tolerance
to widen (`ml/CLAUDE.md` §3b).

**The tensor is anomaly-transformed IN PLACE**, so the driver writes a marker
beside it and refuses to transform twice (running the transform on
already-transformed data z-scores anomalies, and every downstream number stays
finite and plausible). Delete the marker and re-decompress the raw tensor to
start over.

## Scoring the rollout against the archive (gate G3)

G3 is the `e017_u1_s0` head's twelve-month roll: gate AUC **0.643**, corridor
**0.589**, window **0.622**, returned identically by eighteen separate eval
runs (#228 … #413). `ml/CLAUDE.md` §3b calls that PROTOCOL DETERMINISM and is
careful that it is not a replicate — which is exactly why it is the right gate
for a port. Nothing about it varies except the implementation.

```
python3 ml/jaxport/score_gate_roll.py --x <raw tensor>.npy \
    --npz-small <meta>.npz --z <Z cache>.npy --ckpt <codec>.pt \
    --head <...>e017_u1_s0__temporal.pt --backend torch --scope gate \
    --out /tmp/roll_torch.json --zhat-out /tmp/zhat_torch.npy
python3 ml/jaxport/score_gate_roll.py ... --backend jax \
    --out /tmp/roll_jax.json --zhat-out /tmp/zhat_jax.npy \
    --compare-zhat /tmp/zhat_torch.npy
```

Run the torch backend FIRST and require it to land on 0.643. If it does not,
the reproduction path differs from the archived protocol and the JAX number
would be scored against the wrong thing — a finding, never a tolerance to
widen.

Measured 2026-08-21, `f3_anchor41M` codec + the published
`Z_6c52f0687b_adcbe700fb` cache, gate scope (864 px), 234 roll steps:

| backend | gate AUC | auc_damped | archive | Δ |
|---|---|---|---|---|
| torch | **0.643** | 0.619 | 0.643 | +0.0000 |
| jax | **0.643** | 0.619 | 0.643 | +0.0000 |

All twelve per-horizon `msss_clim` rows (0.760 … 0.582) and all twelve `acc`
rows are identical between the backends at `skill_block`'s own three
decimals, and the per-horizon sample counts match exactly. The two backends'
rolled states differ by max **1.09e-3**, mean 2.72e-6 over 12,939,264 values
against a state scale of |z| ≤ 15.0 — and the drift COMPOUNDS with the roll
exactly as it should, 1.9e-5 at h=1 through 1.1e-3 at h=12, which is the
behaviour `tests/test_jaxport_roll.py` iterates twelve steps to expose.

**`--x` is the RAW tensor.** `rollout_spatial` builds its own standardized
anomalies (`stream_stats` → `StdMonths`), so unlike `score_section_probe.py`
this driver must NOT be handed an `anomaly_transform`ed file. If the sidecar
marker `<tensor>.anomaly.json` exists, re-decompress before rolling.

**`--scope` is a cost decision and the artefact records it.** One roll step is
one forward over every rolled pixel and the protocol rolls 234 of them (three
holdout years × twelve staggered starts, truncated at the year end). Measured
on two CPU cores at the gate scope's 864 pixels: 7.1 s/step (torch, 27.6 min)
and 8.6 s/step (jax, 33.4 min). The cost is linear in the rolled pixel count,
so the window's 84,405 pixels are ~45 h and ~54 h.

`--scope gate` scores the gate scope EXACTLY and only it: at stencil 1 `roll_step` has no cross-pixel term —
window, static context, decode, AR1 baseline and `accumulate` are all
per-pixel — so the gate sums are bit-identical whether the other 83,541 pixels
rolled beside them or not. The corridor and window scopes are NOT recoverable
that way (an AUC over the part of the corridor that happens to fall in the
gate's 600-pixel draw is a different quantity), so the driver leaves them out
under a `not_scored` key that says so rather than printing a number under a
name it does not have. A stencil>1 head refuses `--scope gate` outright, for
the same reason stated the other way round: it reads its neighbours.

Context, tiering and the acceptance gates: `ml/plans/JAX_PORT.md`.

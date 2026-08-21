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

Not ported yet: `MultiDec` (`ml/recon_decoder.py`), the head fold-fit, the
transport read-out (`fit_ridge`/`read_sv`) and the band correlations, and the
trainers (tier 3).

## Running the parity tests

```
python3 tests/test_jaxport_parity.py      # tier 1 — CPU, fp32, exits 0/1
python3 tests/test_jaxport_embed.py       # tier 2 embedding path, synthetic
python3 tests/test_jaxport_roll.py        # tier 2 rollout path, synthetic
```

None of them needs the tensor or a checkpoint.

`test_jaxport_roll.py` iterates TWELVE steps rather than checking one, because
a roll feeds its own output back into its input window: tier 1 already pinned
the head's single forward, and what a roll adds is compounding. It covers
stencil 1 and stencil 9, pins the `[n,S,K,dz] → [n,K,S·dz]` gather against
`temporal.gather_stencil` itself (centre slot first, missing slots
zero-filled), pins the chunk byte budget, and checks that `amp=True` refuses.

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

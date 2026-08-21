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

Not ported yet: `MultiDec` (`ml/recon_decoder.py`), the rest of the eval stack
(`roll_step`/`decode_all`, the head fold-fit) and the trainers (tier 3).

## Running the parity tests

```
python3 tests/test_jaxport_parity.py      # tier 1 — CPU, fp32, exits 0/1
python3 tests/test_jaxport_embed.py       # tier 2 embedding path, synthetic
```

Neither needs the tensor or a checkpoint.

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

Context, tiering and the acceptance gates: `ml/plans/JAX_PORT.md`.

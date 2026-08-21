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

Not ported yet: `MultiDec` (`ml/recon_decoder.py`), the eval stack (tier 2)
and the trainers (tier 3).

## Running the parity tests

```
python3 tests/test_jaxport_parity.py      # CPU, fp32, exits 0/1
```

Six checks: the shared encoder layer (plain and causal), PixelMAE at patch 1
and patch 3 (including the float16-widening path), TemporalTransformer at
stencil 1 and 9 with direct heads plus a causality check, SectionHead with
and without pre-pooling blocks, and the converter's refusal contract.
Measured agreement is ≤ 1e-6, well inside gate G1's atol 1e-4.

Context, tiering and the acceptance gates: `ml/plans/JAX_PORT.md`.

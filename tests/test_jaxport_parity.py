#!/usr/bin/env python3
"""Forward parity of `ml/jaxport` against the torch originals (tier 1, G1).

The port is a REFERENCE implementation whose whole value is that it is a
SECOND, INDEPENDENT implementation of the same arithmetic
(`ml/plans/JAX_PORT.md` §1). That is worth nothing unless the two agree, so
this file is the gate: every module ported at tier 1 is run against its torch
original on the same random weights, on CPU, in fp32, in eval mode, and the
elementwise max difference must sit under the tolerance the plan pre-registers
(§5 G1: atol 1e-4; the checks below hold 1e-5 except where a softmax
accumulation genuinely costs a digit).

What each check is FOR — none of them is a smoke test:

  1. the shared encoder layer, plain and with a causal mask — this is where
     torch's defaults hide (relu not gelu, no final norm, packed QKV), and
     every other model here is mostly made of it;
  2. PixelMAE at patch=1, both `encode` (token assembly: the where-cascade
     over observed/masked/missing, cls + ctx tokens, `to_z` on token 0) and
     `query` (the erf-gelu decoder MLP);
  3. PixelMAE at patch=3, where `val_proj` takes 2·9 features — plus the
     float16 input, since family 4 is the project's first float16 tensor and
     the dtype-widening at the top of `encode` is what makes it work;
  4. TemporalTransformer at stencil=1 and stencil=9 with direct heads, plus
     an independent check that the JAX side is genuinely CAUSAL (perturbing
     the future must not move the past — a mask that silently does nothing
     would still pass a parity test against a torch call that passed the same
     broken mask);
  5. SectionHead with and without pre-pooling blocks;
  6. the converter's refusal contract: a state_dict missing a key, and one
     carrying a key nothing wants, must both RAISE and name the offender.
     Silent partial loads produce plausible garbage, which is the failure
     shape this project fears most.

    python3 tests/test_jaxport_parity.py
"""
import os
import sys
import warnings

import numpy as np
import torch
import torch.nn as nn

# torch warns on every norm_first TransformerEncoder that it cannot use the
# nested-tensor fast path. It is expected here (norm_first is the whole
# point) and would otherwise bury the six PASS lines.
warnings.filterwarnings("ignore", message=".*enable_nested_tensor.*")

HERE = os.path.dirname(os.path.abspath(__file__))
ML = os.path.join(os.path.dirname(HERE), "ml")
sys.path.insert(0, ML)

import jax.numpy as jnp                                        # noqa: E402
from flax import nnx                                           # noqa: E402

from model import PixelMAE as TorchPixelMAE                    # noqa: E402
from temporal import TemporalTransformer as TorchTemporal      # noqa: E402
from probe_head import SectionHead as TorchSectionHead         # noqa: E402
from jaxport import models as jm                               # noqa: E402
from jaxport import convert as jc                              # noqa: E402

FAILURES = []


def check(name, a, b, tol):
    """Elementwise max |Δ| between a torch tensor and a JAX array."""
    a = np.asarray(a.detach().cpu().numpy() if hasattr(a, "detach") else a,
                   np.float64)
    b = np.asarray(b, np.float64)
    if a.shape != b.shape:
        FAILURES.append(f"{name}: shape {a.shape} vs {b.shape}")
        return float("inf")
    d = float(np.max(np.abs(a - b)))
    if not (d < tol):
        FAILURES.append(f"{name}: max|Δ| {d:.3e} >= {tol:g}")
    return d


def expect_raise(name, fn, must_mention):
    try:
        fn()
    except KeyError as e:
        msg = str(e)
        for m in must_mention:
            if m not in msg:
                FAILURES.append(f"{name}: refusal did not name {m!r}: {msg[:200]}")
                return
        return
    FAILURES.append(f"{name}: the loader did NOT refuse")


def load_encoder(sd, dst, prefix=""):
    """The converter's encoder mapping, standalone, with the same refusal
    contract — used for the bare-layer check in test 1."""
    c = jc._Consumer(dict(sd), "load_encoder")
    jc._encoder(dst, c, prefix)
    c.finish()
    return dst


# ---- 1: the shared encoder layer ------------------------------------------
def test_encoder_layer():
    torch.manual_seed(0)
    d, nh, ff, B, L = 32, 4, 4 * 32, 3, 7
    layer = nn.TransformerEncoderLayer(d, nh, dim_feedforward=ff,
                                       batch_first=True, norm_first=True,
                                       dropout=0.0)
    enc = nn.TransformerEncoder(layer, 2).eval()
    jenc = jm.TransformerEncoder(d, nh, 2, ff, rngs=nnx.Rngs(0))
    load_encoder(enc.state_dict(), jenc)

    x = torch.randn(B, L, d)
    with torch.no_grad():
        t_plain = enc(x)
        m = nn.Transformer.generate_square_subsequent_mask(L)
        t_causal = enc(x, mask=m)
    jx = jnp.asarray(x.numpy())
    d1 = check("encoder layer (plain)", t_plain, jenc(jx), 1e-5)
    d2 = check("encoder layer (causal)", t_causal,
               jenc(jx, mask=jm.causal_mask(L)), 1e-5)
    print(f"  1. shared TransformerEncoder (norm_first, relu, no final norm, "
          f"packed QKV) matches torch: max|Δ| {d1:.2e} plain, {d2:.2e} with "
          f"the causal mask")


# ---- 2/3: PixelMAE ---------------------------------------------------------
def _mae_pair(patch):
    torch.manual_seed(1)
    kw = dict(n_chan=7, d_model=32, n_heads=4, n_layers=2, d_z=8, d_dec=16,
              dec_layers=2, patch=patch)
    tm = TorchPixelMAE(**kw).eval()
    jmod = jm.PixelMAE(**kw, rngs=nnx.Rngs(0))
    jc.load_pixelmae(tm.state_dict(), jmod)
    return tm, jmod


def _mae_inputs(patch, B=5, C=7, dtype=torch.float32):
    g = torch.Generator().manual_seed(2)
    p2 = patch * patch
    shape = (B, C) if patch == 1 else (B, C, p2)
    x = torch.randn(*shape, generator=g).to(dtype)
    obs = torch.rand(*shape, generator=g) > 0.25        # some cells unobserved
    mask = torch.rand(B, C, generator=g) > 0.6          # some channels masked
    ctx = torch.randn(B, 4, generator=g).to(dtype)
    return x, obs, mask, ctx


def _jax(t):
    return jnp.asarray(t.detach().cpu().numpy())


def test_pixelmae_patch1():
    tm, jmod = _mae_pair(1)
    x, obs, mask, ctx = _mae_inputs(1)
    with torch.no_grad():
        z = tm.encode(x, obs, mask, ctx)
    zj = jmod.encode(_jax(x), _jax(obs), _jax(mask), _jax(ctx))
    d1 = check("PixelMAE patch=1 encode", z, zj, 1e-5)

    g = torch.Generator().manual_seed(3)
    chan = torch.randint(0, 7, (5, 4), generator=g)
    off = torch.randint(-3, 4, (5, 4, 3), generator=g)
    with torch.no_grad():
        y = tm.query(z, chan, off)
    yj = jmod.query(zj, _jax(chan), _jax(off))
    d2 = check("PixelMAE patch=1 query", y, yj, 1e-5)
    print(f"  2. PixelMAE patch=1 (7 chan, some unobserved, some masked): "
          f"encode max|Δ| {d1:.2e}, query on random (chan, Δ) max|Δ| {d2:.2e}")


def test_pixelmae_patch3():
    tm, jmod = _mae_pair(3)
    x, obs, mask, ctx = _mae_inputs(3)
    with torch.no_grad():
        z = tm.encode(x, obs, mask, ctx)
    d1 = check("PixelMAE patch=3 encode",
               z, jmod.encode(_jax(x), _jax(obs), _jax(mask), _jax(ctx)), 1e-5)

    # float16 STORAGE widened to the float32 weights at the top of encode.
    x16 = x.to(torch.float16)
    ctx16 = ctx.to(torch.float16)
    with torch.no_grad():
        z16 = tm.encode(x16, obs, mask, ctx16)
    zj16 = jmod.encode(jnp.asarray(x16.numpy(), jnp.float16), _jax(obs),
                       _jax(mask), jnp.asarray(ctx16.numpy(), jnp.float16))
    d2 = check("PixelMAE patch=3 encode (float16 input)", z16, zj16, 1e-5)
    if zj16.dtype != jnp.float32:
        FAILURES.append(f"float16 input was not widened: z dtype {zj16.dtype}")
    print(f"  3. PixelMAE patch=3 (val_proj over 2x9 features): encode max|Δ| "
          f"{d1:.2e}; float16 input widens to the weight dtype and matches "
          f"torch, max|Δ| {d2:.2e}")


# ---- 4: TemporalTransformer -----------------------------------------------
def _temporal_case(stencil, direct):
    torch.manual_seed(4)
    d_z, d_model, K = 8, 32, 6
    kw = dict(d_z=d_z, d_model=d_model, n_heads=4, n_layers=2, k_max=12,
              direct=direct, stencil=stencil)
    tm = TorchTemporal(**kw).eval()
    jmod = jm.TemporalTransformer(**kw, rngs=nnx.Rngs(0))
    jc.load_temporal(tm.state_dict(), jmod)

    g = torch.Generator().manual_seed(5)
    B = 4
    z_seq = torch.randn(B, K, stencil * d_z, generator=g)
    month = torch.randn(B, K, 2, generator=g)
    static = torch.randn(B, d_z + 2 + (0 if stencil == 1 else stencil),
                         generator=g)
    with torch.no_grad():
        pred, h = tm(z_seq, month, static)
    pj, hj = jmod(_jax(z_seq), _jax(month), _jax(static))
    dp = check(f"TemporalTransformer stencil={stencil} pred", pred, pj, 1e-5)
    dh = check(f"TemporalTransformer stencil={stencil} h", h, hj, 1e-5)
    dd = 0.0
    for hz in direct:
        with torch.no_grad():
            td = tm.heads_direct[str(hz)](h)
        dd = max(dd, check(f"direct head {hz}", td,
                           jmod.direct_pred(hj, hz), 1e-5))
    return jmod, z_seq, month, static, pj, dp, dh, dd


def test_temporal():
    _, _, _, _, _, d1p, d1h, _ = _temporal_case(1, ())
    jmod, z_seq, month, static, pj, d9p, d9h, dd = _temporal_case(9, (3, 6))

    # CAUSALITY, checked on the JAX side directly: a mask that silently did
    # nothing would still match a torch call handed the same broken mask.
    z2 = z_seq.clone()
    z2[:, 3:] += 7.0
    pj2, _ = jmod(_jax(z2), _jax(month), _jax(static))
    dcaus = float(np.max(np.abs(np.asarray(pj)[:, :3] - np.asarray(pj2)[:, :3])))
    if not (dcaus < 1e-6):
        FAILURES.append(f"TemporalTransformer is not causal: perturbing steps "
                        f"3.. moved pred[:, :3] by {dcaus:.3e}")
    moved = float(np.max(np.abs(np.asarray(pj)[:, 3:] - np.asarray(pj2)[:, 3:])))
    if not (moved > 1e-3):
        FAILURES.append("perturbing the future did not move the future either "
                        "— the causality check is vacuous")
    print(f"  4. TemporalTransformer stencil=1 (pred {d1p:.2e}, h {d1h:.2e}) "
          f"and stencil=9 with direct=(3,6) (pred {d9p:.2e}, h {d9h:.2e}, "
          f"direct {dd:.2e}); JAX-side causality holds "
          f"(past moved {dcaus:.1e}, future moved {moved:.2f})")


# ---- 5: SectionHead --------------------------------------------------------
def test_section_head():
    ds = []
    for n_blocks in (0, 1):
        torch.manual_seed(6)
        kw = dict(in_dim=10, d=64, K=3, n_blocks=n_blocks)
        tm = TorchSectionHead(**kw).eval()
        jmod = jm.SectionHead(**kw, rngs=nnx.Rngs(0))
        jc.load_section_head(tm.state_dict(), jmod)
        g = torch.Generator().manual_seed(7)
        tok = torch.randn(4, 9, 10, generator=g)
        with torch.no_grad():
            y = tm(tok)
        ds.append(check(f"SectionHead n_blocks={n_blocks}", y,
                        jmod(_jax(tok)), 1e-4))
    print(f"  5. SectionHead in eval mode (dropout off): n_blocks=0 max|Δ| "
          f"{ds[0]:.2e}, n_blocks=1 max|Δ| {ds[1]:.2e}")


# ---- 6: the converter refuses partial loads --------------------------------
def test_refusal():
    torch.manual_seed(8)
    kw = dict(n_chan=5, d_model=16, n_heads=2, n_layers=1, d_z=4, d_dec=8,
              dec_layers=1, patch=1)
    tm = TorchPixelMAE(**kw)
    sd = dict(tm.state_dict())

    short = dict(sd)
    dropped = "encoder.layers.0.linear1.weight"
    del short[dropped]
    expect_raise("missing key", lambda: jc.load_pixelmae(
        short, jm.PixelMAE(**kw, rngs=nnx.Rngs(0))), [dropped, "missing"])

    fat = dict(sd)
    fat["encoder.layers.0.bogus"] = torch.zeros(3)
    expect_raise("unexpected key", lambda: jc.load_pixelmae(
        fat, jm.PixelMAE(**kw, rngs=nnx.Rngs(0))),
        ["encoder.layers.0.bogus", "unconsumed"])

    # And the happy path still loads, so the refusal is not simply always on.
    jc.load_pixelmae(sd, jm.PixelMAE(**kw, rngs=nnx.Rngs(0)))
    print("  6. the loaders refuse a state_dict with a missing key and one "
          "with an unconsumed key, naming the offender in both cases; the "
          "complete state_dict still loads")


def main():
    print("tests/test_jaxport_parity.py — torch vs ml/jaxport, CPU, fp32, "
          "eval mode\n")
    for fn in (test_encoder_layer, test_pixelmae_patch1, test_pixelmae_patch3,
               test_temporal, test_section_head, test_refusal):
        fn()
    if FAILURES:
        print("\nFAILED:")
        for f in FAILURES:
            print("  -", f)
        return 1
    print("\ntests/test_jaxport_parity.py: all 6 checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())

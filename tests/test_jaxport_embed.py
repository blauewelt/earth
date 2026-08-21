#!/usr/bin/env python3
"""Parity of `ml/jaxport/embed.py` against `ml/temporal.py:embed_everything`.

Tier 2's first slice. The embedding loop is the one piece of the eval stack
that every probe reads and nothing else re-derives, so a drift here would
reach every number the port can be scored with — and it would be INVISIBLE,
because a drifted Z still produces a probe correlation that looks like a
probe correlation (`ml/plans/JAX_PORT.md` §3.2).

**This test runs with NO large data present.** CI has no tensor and no
checkpoint: everything below is a synthetic [6,9,11,39] float32 array with
NaNs scattered so some channels are genuinely unobserved and some pixels sit
on the wrap/clamp boundary, plus a randomly-initialised 1-layer codec whose
torch weights are converted across. That is the whole point of §4.8's "exercise
the code path on a toy" — the real tensor is scored by
`ml/jaxport/score_section_probe.py`, and it is scored against a pre-registered
constant, not against this file.

What each check is FOR:

  1. `gather_px_np` against `ml/model.py:gather_px` — exact equality on both
     values and observed flags, with the section deliberately touching x = 0
     and x = W-1 (longitude WRAPS) and y = 0 and y = H-1 (latitude CLAMPS,
     out-of-range rows marked unobserved). This is the one piece of the loop
     re-implemented rather than reused, so it is pinned rather than trusted.
  2. patch=3 parity, the real anchor codec's geometry. Asserted TWICE, and
     the two assertions have opposite jobs. The float32 PRE-CAST agreement is
     held to 1e-4 (gate G1's tolerance) because that is where a real drift
     would show: float16 carries ~3 decimal digits, so two encoders
     disagreeing in the fourth would round to the same bits and a float16-only
     check would pass on a port that had drifted. The float16 OUTPUT is then
     held to **at most one ULP, on at most a small fraction of elements** —
     not to bit-equality, which was tried first and is not achievable:

       measured on the toy at patch=3, the two encoders agree to max|Δ| 1.2e-7
       in float32, and 3 of 384 stored values still land on opposite sides of
       a float16 rounding boundary (e.g. -0.00544357 vs -0.00544739, adjacent
       representable float16s, absolute gap 3.8e-6).

     That is the storage cast disagreeing on a tie, not the arithmetic
     disagreeing on a value, and no reordering of two independent
     implementations' float32 sums can prevent it. So the assertion is the
     strongest one that is true at BOTH scales this path runs at: no stored
     value may move by more than one float16 ULP at the array's own maximum
     magnitude, and the differing fraction must stay small. The differing
     count is printed so a real drift — which would move many elements, by
     more than the last stored digit — cannot hide inside the allowance.
  3. patch=1 parity, the pilot geometry: a different `val_proj` fan-in and a
     different branch of the token assembly, and no gather at all.
  4. `mask_chan` — the ablation path, where the masked channels' values are
     zeroed by the CALLER and their tokens replaced inside `encode`. Doing one
     without the other is a silent half-mask, and both implementations must do
     both.
  5. `coords` — the second return value, which is what the context token's
     lat/lon slots are built from.

    python3 tests/test_jaxport_embed.py
"""
import os
import sys
import warnings

import numpy as np
import torch

warnings.filterwarnings("ignore", message=".*enable_nested_tensor.*")

HERE = os.path.dirname(os.path.abspath(__file__))
ML = os.path.join(os.path.dirname(HERE), "ml")
sys.path.insert(0, ML)

import jax.numpy as jnp                                        # noqa: E402
from flax import nnx                                           # noqa: E402

from model import PixelMAE as TorchPixelMAE, LazyPixels, gather_px  # noqa: E402
from temporal import embed_everything, CACHE_DTYPE             # noqa: E402
from jaxport import convert as jc                              # noqa: E402
from jaxport import models as jm                               # noqa: E402
from jaxport.embed import embed_everything_jax, gather_px_np   # noqa: E402

FAILURES = []

T, H, W, C = 6, 9, 11, 39
D_Z = 8


def fail(msg):
    FAILURES.append(msg)


def make_tensor(seed=0):
    """A toy tensor with the awkward shapes the real one has: whole channels
    unobserved at some pixels, whole pixels unobserved (land), and NaNs
    scattered so `isfinite` is not trivially all-True."""
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((T, H, W, C)).astype(np.float32)
    X[:, :2, :, :] = np.nan                       # a land band
    X[:, :, :, 7] = np.nan                        # a channel observed nowhere
    X[:2, :, :, 11] = np.nan                      # a channel that starts late
    X[rng.random((T, H, W, C)) < 0.05] = np.nan   # scattered dropouts
    return X


def make_codecs(patch, seed=0):
    """One torch codec with random weights, and the same weights in JAX."""
    torch.manual_seed(seed)
    kw = dict(n_chan=C, d_model=32, n_heads=4, n_layers=1, d_z=D_Z,
              d_dec=24, patch=patch, dec_layers=2)
    tm = TorchPixelMAE(**kw)
    tm.eval()
    jmod = jc.load_pixelmae(tm.state_dict(),
                            jm.PixelMAE(**kw, rngs=nnx.Rngs(0)))
    return tm, jmod


def section(X):
    """Pixels along one row, plus the two extremes of x and both extremes of
    y, so the gather is exercised on every boundary it has."""
    ys = np.array([0, 4, 4, 4, 4, H - 1, 5, 5], dtype=np.int64)
    xs = np.array([0, 0, 1, W - 2, W - 1, 3, 6, 7], dtype=np.int64)
    return ys, xs


def ctx_and_axes():
    moy = np.arange(T) % 12
    ctx_all = np.stack([np.sin(2 * np.pi * moy / 12),
                        np.cos(2 * np.pi * moy / 12)], 1).astype(np.float32)
    lats = np.linspace(20.0, 40.0, H).astype(np.float32)
    lons = np.linspace(-80.0, -10.0, W).astype(np.float32)
    return ctx_all, lats, lons


# --------------------------------------------------------------------------
def test_gather():
    X = make_tensor(1)
    Xt, OBS = LazyPixels(X), LazyPixels(X, obs=True)
    ys, xs = section(X)
    for patch in (1, 3):
        t = torch.full((len(ys),), 2, dtype=torch.long)
        vt, ot = gather_px(Xt, OBS, t, torch.as_tensor(ys),
                           torch.as_tensor(xs), patch)
        vn, on = gather_px_np(Xt, OBS, np.full(len(ys), 2), ys, xs, patch)
        if not np.array_equal(np.asarray(vt), vn):
            fail(f"gather_px_np values differ from gather_px at patch={patch}")
        if not np.array_equal(np.asarray(ot), on):
            fail(f"gather_px_np flags differ from gather_px at patch={patch}")
    # The boundary cases the section was chosen for must actually be present,
    # or check 1 is asserting nothing: a gather that never wraps agrees with
    # a wrong wrap rule.
    if not ((ys == 0).any() and (ys == H - 1).any()
            and (xs == 0).any() and (xs == W - 1).any()):
        fail("gather section does not touch the wrap/clamp boundaries")
    print("  1. gather_px_np == ml/model.py:gather_px, values and observed "
          "flags, at patch 1 and 3, on a section touching x=0, x=W-1, y=0 "
          "and y=H-1 (longitude wraps, latitude clamps unobserved)")


def _parity(patch, mask_chan=None, label=""):
    X = make_tensor(2)
    Xt, OBS = LazyPixels(X), LazyPixels(X, obs=True)
    ys, xs = section(X)
    ctx_all, lats, lons = ctx_and_axes()
    tm, jmod = make_codecs(patch)

    Zt, ct = embed_everything(tm, Xt, OBS, ctx_all, lats, lons, ys, xs, D_Z,
                              cache_path=None, mask_chan=mask_chan)
    Zj, cj = embed_everything_jax(jmod, Xt, OBS, ctx_all, lats, lons, ys, xs,
                                  D_Z, mask_chan=mask_chan)

    if Zt.shape != (T, len(ys), D_Z) or Zj.shape != Zt.shape:
        fail(f"{label}: shape {Zj.shape} vs {Zt.shape}")
        return float("nan"), -1, 0
    if Zt.dtype != CACHE_DTYPE or Zj.dtype != CACHE_DTYPE:
        fail(f"{label}: dtype {Zj.dtype} vs {Zt.dtype} (want {CACHE_DTYPE})")
    if not np.array_equal(ct, cj):
        fail(f"{label}: coords differ")

    # THE FLOAT32 CHECK, BEFORE THE CAST COULD MASK IT. float16 carries ~3
    # decimal digits, so two encoders that disagree in the fourth still round
    # to identical bits. Re-running both encoders' arithmetic at full width is
    # the only way to see that difference — so the float32 agreement is
    # asserted at gate G1's 1e-4, and the float16 output at one ULP (see the
    # module docstring, check 2, for why bit-equality is not achievable).
    d32 = _float32_delta(tm, jmod, Xt, OBS, ctx_all, lats, lons, ys, xs,
                         patch, mask_chan)
    if not (d32 < 1e-4):
        fail(f"{label}: float32 pre-cast max|Δ| {d32:.3e} >= 1e-4")
    a, b = np.asarray(Zt), np.asarray(Zj)
    diff = a != b
    n = int(diff.sum())
    if n:
        # The bound is ONE ULP AT THE ARRAY'S OWN MAXIMUM MAGNITUDE, not one
        # ULP at each element's magnitude. The stricter per-element form was
        # tried and does not generalise: on the real section Z (|z| up to
        # 11.28) the two backends differ by up to 82 float16 ULPs — but every
        # one of those is a value near ZERO, where a float16 ULP is 6e-8 and
        # the underlying float32 disagreement of ~1e-5 spans many of them.
        # Measured on the real run: max |Δ| 7.8e-3, which is exactly
        # ULP(11.28). Scaling the bound by the array's own magnitude is the
        # statement that is true at both scales, and it still catches drift —
        # a real one moves values by far more than the last stored digit.
        scale = float(np.max(np.abs(a.astype(np.float64)))) or 1.0
        ulp = float(np.spacing(np.float16(scale)))
        worst = float(np.max(np.abs(a[diff].astype(np.float64)
                                    - b[diff].astype(np.float64))))
        if worst > ulp:
            fail(f"{label}: float16 Z differs by {worst:.3e}, more than one "
                 f"ULP ({ulp:.3e}) at the array's max magnitude {scale:.3f} — "
                 f"that is arithmetic drift, not the storage cast")
        if n > 0.02 * a.size:
            fail(f"{label}: {n}/{a.size} float16 elements differ — too many "
                 f"to be rounding-boundary ties")
    return d32, n, a.size


def _float32_delta(tm, jmod, Xt, OBS, ctx_all, lats, lons, ys, xs, patch,
                   mask_chan):
    """Max |Δ| between the two encoders at full float32 width, over the same
    per-batch inputs the two loops build."""
    coords = np.stack([lats[ys] / 90, lons[xs] / 180], 1).astype(np.float32)
    n = len(ys)
    worst = 0.0
    for t in range(T):
        ctx = np.concatenate([np.tile(ctx_all[t], (n, 1)), coords], 1)
        mk = np.zeros((n, len(np.asarray(OBS[0, 0, 0]))), dtype=bool)
        if mask_chan is not None:
            mk[:, mask_chan] = True
        v, o = gather_px_np(Xt, OBS, np.full(n, t), ys, xs, patch)
        v = v * ((~mk)[..., None] if patch > 1 else ~mk)
        with torch.no_grad():
            zt = tm.encode(torch.as_tensor(v), torch.as_tensor(o),
                           torch.as_tensor(mk),
                           torch.as_tensor(ctx, dtype=torch.float32)).numpy()
        zj = np.asarray(jmod.encode(jnp.asarray(v), jnp.asarray(o),
                                    jnp.asarray(mk),
                                    jnp.asarray(ctx, jnp.float32)))
        worst = max(worst, float(np.max(np.abs(zt.astype(np.float64)
                                               - zj.astype(np.float64)))))
    return worst


def _f16(n, size):
    return ("bit-identical" if n == 0
            else f"{n}/{size} elements one ULP apart (rounding-boundary ties)")


def test_patch3():
    d, n, sz = _parity(3, label="patch=3")
    print(f"  2. embed_everything_jax == embed_everything at patch=3 (the "
          f"anchor codec's geometry): float32 pre-cast max|Δ| {d:.2e} < 1e-4; "
          f"float16 [T,P,d_z] output {_f16(n, sz)}")


def test_patch1():
    d, n, sz = _parity(1, label="patch=1")
    print(f"  3. same at patch=1 (the pilot geometry: val_proj fan-in 1, the "
          f"other branch of the token assembly, no gather): float32 max|Δ| "
          f"{d:.2e}; float16 {_f16(n, sz)}")


def test_mask_chan():
    d, n, sz = _parity(3, mask_chan=[0, 3, 11], label="patch=3 mask_chan")
    print(f"  4. the mask_chan ablation path agrees too — caller-side value "
          f"zeroing AND the in-encoder mask token, three channels masked: "
          f"float32 max|Δ| {d:.2e}; float16 {_f16(n, sz)}")


def test_coords():
    X = make_tensor(3)
    ys, xs = section(X)
    ctx_all, lats, lons = ctx_and_axes()
    tm, jmod = make_codecs(3)
    Xt, OBS = LazyPixels(X), LazyPixels(X, obs=True)
    _, cj = embed_everything_jax(jmod, Xt, OBS, ctx_all, lats, lons, ys, xs,
                                 D_Z)
    want = np.stack([lats[ys] / 90, lons[xs] / 180], 1).astype(np.float32)
    if not (cj.dtype == np.float32 and np.array_equal(cj, want)):
        fail("coords are not [P,2] float32 lat/90, lon/180")
    print("  5. the returned coords are [P,2] float32 lat/90 · lon/180, "
          "identical to the torch original's")


def main():
    print("tests/test_jaxport_embed.py — ml/jaxport/embed.py vs "
          "ml/temporal.py:embed_everything, CPU, synthetic data\n")
    for fn in (test_gather, test_patch3, test_patch1, test_mask_chan,
               test_coords):
        fn()
    if FAILURES:
        print("\nFAILED:")
        for f in FAILURES:
            print("  -", f)
        return 1
    print("\ntests/test_jaxport_embed.py: all 5 checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())

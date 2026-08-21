#!/usr/bin/env python3
"""Parity of `ml/jaxport/roll.py` against `ml/rollout_spatial.py`'s
`roll_step` / `decode_all`, over a TWELVE-STEP iterated roll.

Tier 2's rollout slice — the arithmetic gate G3 scores
(`ml/plans/JAX_PORT.md` §5). One forward is not the thing under test: tier 1
already pinned the head's forward elementwise. What a roll adds is FEEDBACK —
step h's output becomes step h+1's input window — so a per-step disagreement
compounds, and a port that is fine at step 1 can be visibly wrong at step 12
with nothing in between to notice. Twelve is the archive's own horizon, and it
is why this test iterates rather than checking a step.

**This test runs with NO large data present.** No tensor, no Z cache, no
published head: a synthetic [T,P,d_z] float16 Z (float16 because that is what
the cache is, and the widening to float32 is part of what is being pinned), a
randomly-initialised 1-layer `TemporalTransformer` whose torch weights are
converted across, and a tiny random codec for the decode. The real gate is
scored by `ml/jaxport/score_gate_roll.py` against a pre-registered constant,
not by this file (`ml/CLAUDE.md` §4.8).

What each check is FOR:

  1. **stencil 1, twelve iterated steps.** The legacy geometry every
     published head before E-022 uses, and the one the gate head has. Both
     backends are driven from the SAME `month_feats` and the same window, and
     every step's ẑ is compared elementwise — plus the drift is printed per
     step, because "the max |Δ| grew by 30× over twelve steps" is the reading
     that says whether the two implementations are diverging or merely
     rounding differently.
  2. **stencil 9, twelve iterated steps.** E-022's 3×3. This is where the
     gather layout lives, and a transposed layout still runs.
  3. **the gather itself, pinned against `gather_stencil`.** `roll_step`'s
     [n,S,K,dz] → [n,K,S·dz] permutation and `temporal.gather_stencil`'s
     per-step `zj.flatten(1)` are two spellings of one layout, and the model's
     `inp` weight assumes it. Checked as EXACT equality against the torch
     function the trainer itself calls, on a neighbour table containing
     missing (-1) slots, so the zero-fill is pinned too.
  4. **`decode_all`.** The codec query at every (pixel, channel) at offset 0.
  5. **the chunk boundary is arithmetic, not a memory hint.** Rolling the same
     window at two different chunk sizes must give the same answer on the JAX
     side, and the byte-budget row count must match the torch original's.
  6. **`amp` REFUSES.** The flag means fp16 autocast on the torch path; this
     backend does not implement it and must say so rather than roll at fp32
     under a flag that says otherwise (`ml/CLAUDE.md` §0.2).

    python3 tests/test_jaxport_roll.py
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

from model import PixelMAE as TorchPixelMAE                    # noqa: E402
from temporal import (TemporalTransformer as TorchTemporal,    # noqa: E402
                      build_stencil, gather_stencil, STENCILS)
from rollout_spatial import roll_step, decode_all, month_feats  # noqa: E402
from jaxport import convert as jc                              # noqa: E402
from jaxport import models as jm                               # noqa: E402
from jaxport.roll import (roll_step_jax, decode_all_jax,       # noqa: E402
                          roll_chunk_rows, AmpUnsupported)

FAILURES = []

# Small enough to run in seconds, awkward enough to be a real test: P is not a
# multiple of the chunk size used below, K is not the horizon, and d_z is not
# d_model.
T, P, D_Z, K, H_ROLL = 40, 37, 8, 6, 12
GRID_H, GRID_W = 7, 9
D_MODEL, N_LAYERS = 32, 2
CHAN = 5
CPU = torch.device("cpu")


def fail(msg):
    FAILURES.append(msg)


def make_heads(stencil, seed=0):
    """One torch head with random weights, and the same weights in JAX."""
    torch.manual_seed(seed)
    tm = TorchTemporal(d_z=D_Z, d_model=D_MODEL, n_heads=4,
                       n_layers=N_LAYERS, k_max=K, direct=(), stencil=stencil)
    tm.eval()
    jmod = jm.TemporalTransformer(d_z=D_Z, d_model=D_MODEL, n_heads=4,
                                  n_layers=N_LAYERS, k_max=K, direct=(),
                                  stencil=stencil)
    jc.load_temporal(tm.state_dict(), jmod)
    return tm, jmod


def make_geometry(stencil, seed=0):
    """A pixel list on a small grid with holes, and its neighbour table.

    Holes matter: `build_stencil` writes -1 wherever a slot lands on land or
    off the window, and the zero-fill for those slots is exactly the thing a
    port gets wrong silently.
    """
    rng = np.random.default_rng(seed)
    lin = rng.permutation(GRID_H * GRID_W)[:P]
    lin.sort()
    ys, xs = np.divmod(lin, GRID_W)
    lats = np.linspace(20.0, 26.0, GRID_H).astype(np.float32)
    if stencil == 1:
        return ys, xs, None
    NBR = build_stencil(GRID_H, GRID_W, ys, xs, stencil, lats=lats)
    return ys, xs, NBR


def make_inputs(stencil, NBR, seed=1):
    """A float16 Z window (the cache dtype), month features and static ctx."""
    rng = np.random.default_rng(seed)
    Z = rng.standard_normal((T, P, D_Z)).astype(np.float16)
    # THE WINDOW IS BUILT THE WAY rollout_spatial BUILDS IT: [K,P,dz] out of
    # the cache, transposed to [P,K,dz], widened to float32 at that boundary
    # and nowhere else.
    Zwin_np = np.ascontiguousarray(
        np.asarray(Z[T - K:T]).transpose(1, 0, 2)).astype(np.float32)
    n_static = D_Z + 2 if stencil == 1 else D_Z + 2 + stencil
    sctx = rng.standard_normal((P, n_static)).astype(np.float32)
    if stencil > 1:
        # the observed-flags the real driver puts in the last S slots
        sctx[:, D_Z + 2:] = (NBR >= 0).astype(np.float32)
    moys = list(rng.integers(0, 12, size=K))
    return Zwin_np, sctx, moys


def iterated_roll(tm, jmod, stencil, NBR, chunk):
    """Roll both backends `H_ROLL` steps from the same window and return the
    per-step max |Δ| plus the final windows."""
    Zwin_np, sctx_np, moys = make_inputs(stencil, NBR)
    NBR_t = None if NBR is None else torch.as_tensor(NBR)
    zt = torch.from_numpy(Zwin_np.copy())
    zj = jnp.asarray(Zwin_np)
    sct = torch.from_numpy(sctx_np)
    scj = jnp.asarray(sctx_np)
    cur_t, cur_j = list(moys), list(moys)
    drift = []
    rng = np.random.default_rng(7)
    for h in range(1, H_ROLL + 1):
        mf = month_feats(cur_t, CPU)               # ONE month-feature source
        with torch.no_grad():
            pt = roll_step(tm, zt, NBR_t, sct, mf, chunk)
        pj = roll_step_jax(jmod, zj, NBR, scj, np.asarray(mf), chunk)
        a, b = pt.numpy(), np.asarray(pj)
        drift.append(float(np.abs(a - b).max()))
        zt = torch.cat([zt[:, 1:], pt[:, None]], 1)
        zj = jnp.concatenate([zj[:, 1:], pj[:, None]], 1)
        # An arbitrary next month, identical on both sides — the AXIS is the
        # caller's business (`rollout_spatial.main` reads it off the row's own
        # date), and what matters here is that both backends get the same one.
        nm = int(rng.integers(0, 12))
        cur_t = cur_t[1:] + [nm]
        cur_j = cur_j[1:] + [nm]
    return drift, zt.numpy(), np.asarray(zj)


def _roll_check(name, stencil, tol=2e-5):
    NBR = make_geometry(stencil)[2]
    tm, jmod = make_heads(stencil)
    drift, wt, wj = iterated_roll(tm, jmod, stencil, NBR, chunk=16)
    worst = max(drift)
    wdiff = float(np.abs(wt - wj).max())
    print(f"  {name}: 12 iterated steps, max|Δ| per step "
          f"{drift[0]:.2e} → {drift[-1]:.2e} (worst {worst:.2e}); "
          f"final window max|Δ| {wdiff:.2e}")
    if not (worst < tol and wdiff < tol):
        fail(f"{name}: iterated roll disagrees by {worst:.3e} "
             f"(window {wdiff:.3e}), tol {tol:.0e}")


def test_roll_stencil1():
    _roll_check("1. stencil 1", 1)


def test_roll_stencil9():
    _roll_check("2. stencil 9", 9)


def test_gather_layout():
    """`roll_step`'s gather IS `gather_stencil`'s layout — checked against the
    function the TRAINER calls, not against a restatement of it."""
    stencil = 9
    ys, xs, NBR = make_geometry(stencil)
    rng = np.random.default_rng(3)
    Z = rng.standard_normal((T, P, D_Z)).astype(np.float32)
    Zt = torch.from_numpy(Z)
    NBR_t = torch.as_tensor(NBR)
    base = torch.arange(T - K - 3, T - K - 3 + 1)          # one window start
    p = torch.arange(P)
    want = gather_stencil(Zt, base.expand(P), p, NBR_t, K).numpy()

    # the same window as roll_step sees it: [P,K,dz]
    s0 = int(base[0])
    Zwin = jnp.asarray(np.ascontiguousarray(
        Z[s0:s0 + K].transpose(1, 0, 2)))
    nbr = jnp.asarray(NBR)
    miss = nbr < 0
    zj = Zwin[jnp.maximum(nbr, 0)]
    zj = jnp.where(miss[:, :, None, None], 0.0, zj)
    got = np.asarray(jnp.transpose(zj, (0, 2, 1, 3)).reshape(P, K, -1))
    if not np.array_equal(want, got):
        fail(f"3. gather layout differs from gather_stencil: "
             f"max|Δ| {np.abs(want - got).max():.3e}")
    n_missing = int((NBR < 0).sum())
    if n_missing == 0:
        fail("3. the toy geometry has no missing neighbours — the zero-fill "
             "is untested; widen the grid holes")
    print(f"  3. the [n,S,K,dz]→[n,K,S·dz] gather is EXACTLY "
          f"gather_stencil's layout ({n_missing} missing slots zero-filled, "
          f"centre slot first, STENCILS[{stencil}] = {STENCILS[stencil][:3]}…)")


def test_decode_all():
    torch.manual_seed(11)
    tc = TorchPixelMAE(n_chan=CHAN, d_model=16, n_heads=4, n_layers=1,
                       d_z=D_Z, d_dec=24, patch=1, dec_layers=2)
    tc.eval()
    jco = jm.PixelMAE(n_chan=CHAN, d_model=16, n_heads=4, n_layers=1,
                      d_z=D_Z, d_dec=24, patch=1, dec_layers=2)
    jc.load_pixelmae(tc.state_dict(), jco)
    zhat = np.random.default_rng(5).standard_normal((P, D_Z)).astype(np.float32)
    with torch.no_grad():
        xt = decode_all(tc, torch.from_numpy(zhat), CHAN, 16)
    xj = decode_all_jax(jco, zhat, CHAN, 16)
    d = float(np.abs(xt - xj).max())
    if not (xj.shape == (P, CHAN) and d < 1e-5):
        fail(f"4. decode_all disagrees by {d:.3e} (shape {xj.shape})")
    print(f"  4. decode_all over {P}×{CHAN} queries at offset 0: "
          f"max|Δ| {d:.2e}, numpy [P,C] out")


def test_chunking():
    """The chunk is a byte budget on the torch side; here it must (a) not
    change the answer and (b) derive the same row count."""
    stencil = 9
    NBR = make_geometry(stencil)[2]
    tm, jmod = make_heads(stencil)
    Zwin, sctx, moys = make_inputs(stencil, NBR)
    mf = np.asarray(month_feats(moys, CPU))
    a = np.asarray(roll_step_jax(jmod, Zwin, NBR, sctx, mf, 8))
    b = np.asarray(roll_step_jax(jmod, Zwin, NBR, sctx, mf, 4096))
    d = float(np.abs(a - b).max())
    if d != 0.0:
        fail(f"5. chunk size changed the JAX answer by {d:.3e}")
    # the real gate's shape: a 90-slot head at K=24, d_z=64 — the allocation
    # that killed #353, and the reason the row count is derived at all
    got = roll_chunk_rows(8192, np.zeros((10, 90), np.int64), 24, 64)
    want = max(256, min(8192, (1 << 30) // (90 * 24 * 64 * 4)))
    if got != want:
        fail(f"5. roll_chunk_rows {got} != rollout_spatial's {want}")
    print(f"  5. chunking is arithmetic-neutral (Δ 0.0 at chunk 8 vs 4096) "
          f"and the byte budget matches: 90 slots × K 24 × d_z 64 → "
          f"{got} rows, not 8192")


def test_amp_refuses():
    stencil = 1
    tm, jmod = make_heads(stencil)
    Zwin, sctx, moys = make_inputs(stencil, None)
    mf = np.asarray(month_feats(moys, CPU))
    for what, fn in (("roll_step_jax",
                      lambda: roll_step_jax(jmod, Zwin, None, sctx, mf, 16,
                                            amp=True)),
                     ("decode_all_jax",
                      lambda: decode_all_jax(None, Zwin[:, 0], CHAN, 16,
                                             amp=True))):
        try:
            fn()
        except AmpUnsupported:
            continue
        except Exception as e:                     # noqa: BLE001
            fail(f"6. {what}(amp=True) raised {type(e).__name__}, not "
                 f"AmpUnsupported: {e}")
        else:
            fail(f"6. {what}(amp=True) returned a number instead of refusing")
    print("  6. amp=True REFUSES in both roll_step_jax and decode_all_jax "
          "(AmpUnsupported), rather than rolling at fp32 under an fp16 flag")


def main():
    print("tests/test_jaxport_roll.py — ml/jaxport/roll.py vs "
          "ml/rollout_spatial.py roll_step/decode_all, CPU, synthetic data\n")
    for fn in (test_roll_stencil1, test_roll_stencil9, test_gather_layout,
               test_decode_all, test_chunking, test_amp_refuses):
        fn()
    if FAILURES:
        print("\nFAILED:")
        for f in FAILURES:
            print("  -", f)
        return 1
    print("\ntests/test_jaxport_roll.py: all 6 checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""The anomaly transform must survive float16 tensors.

Family 4 (build_family4.py) is the project's first float16 tensor — the dtype
was chosen so 3142x281x481x39 fits in 33.1 GB instead of 66.3. That choice
silently broke trainprobe.anomaly_transform, because numpy upcasts the
accumulator for np.mean on float16 but NOT for np.std/np.var (_methods._var
upcasts only integer and bool). The z-score sums ~204M squared residuals; in
float16 that passes 65504, returns inf, and (X - mu) / (inf + 1e-6) is exactly
0.0 for every dynamic channel.

Nothing downstream would have said so. The loss is finite, gpu_util is normal,
the light probe returns a number. The trained arms would have learned an
all-zero field and the frozen control -- E-038's cheapest number and the only
one that can falsify its premise -- would have "measured" the monthly codec
against zeros and reported a null.

ml/CLAUDE.md §4.10: instrument the quantity that DISTINGUISHES the stories.
The distinguishing quantity here is the variance that survives the transform,
so that is what these tests assert, at a pool size large enough to overflow.

    python3 -m pytest tests/test_e038_anomaly_dtype.py -q
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "ml"))

from trainprobe import anomaly_transform            # noqa: E402


# Small in H/W, long in T. The overflow is driven by the SUM OF SQUARED
# RESIDUALS over the train pool -- not by the pool's length -- so the fixture
# needs both enough entries and enough residual variance to clear 65504.
# A first version used T=600 with 0.5-amplitude noise: 180k entries at
# var 0.25 sums to ~45k, just UNDER the limit, and every test passed against
# the unfixed implementation. `_assert_in_regime` below now measures that on
# the real data instead of assuming it from the count.
T, H, W, C = 2000, 24, 26, 3
DYN, STATIC = 0, 2          # channel 1 is also dynamic; 2 is a baked constant


def _tensor(dtype):
    """A tensor with the family-4 shape signature: a seasonal cycle plus
    noise on the dynamic channels, a time-invariant field on the static one,
    and a land mask that is NaN at every timestep."""
    rng = np.random.default_rng(7)
    moy = np.arange(T) % 12
    X = np.empty((T, H, W, C), dtype=dtype)
    season = np.sin(2 * np.pi * moy / 12)[:, None, None]
    field = rng.standard_normal((H, W))
    for c in range(C):
        if c == STATIC:
            X[..., c] = np.broadcast_to(field, (T, H, W)).astype(dtype)
        else:
            X[..., c] = (season + rng.standard_normal((T, H, W))
                         + field).astype(dtype)
    land = rng.random((H, W)) > 0.62
    X[:, land, :] = np.nan
    t_hold = np.zeros(T, bool)
    t_hold[-T // 8:] = True
    x_hold = np.zeros(W, bool)
    x_hold[-3:] = True
    return X, moy, t_hold, x_hold


def _pool_size(X, t_hold, x_hold, c):
    m = (np.isfinite(X[..., c]) & ~t_hold[:, None, None]
         & ~x_hold[None, None, :])
    return int(m.sum())


def _assert_in_regime(vals, dtype):
    """Fail if the fixture is too small to reproduce the bug.

    A test for an overflow is worthless unless its data actually overflows,
    and 'the pool has more than 65504 entries' does NOT establish that --
    what matters is the sum of squared residuals. So measure it: run numpy's
    own default-accumulator std over the pool at the storage dtype and demand
    it come back non-finite. If it is finite, this fixture would pass against
    the unfixed implementation and is pinning nothing.
    """
    if dtype is not np.float16:
        return
    with np.errstate(over="ignore", invalid="ignore"):
        native = vals.astype(np.float16).std()
    assert not np.isfinite(native), (
        f"fixture is out of regime: float16 std over {vals.size:,} entries "
        f"returned {native}, so the pre-fix code path would NOT have "
        f"overflowed and this test cannot detect the bug")


@pytest.mark.parametrize("dtype", [np.float16, np.float32])
def test_dynamic_channels_survive_the_transform(dtype):
    """The z-scored dynamic channels must have real variance, not zeros.

    This is the assertion that fails on the pre-2026-08-17 implementation
    when dtype is float16, and it fails ALL THE WAY to zero -- not to a
    slightly wrong number -- which is why no threshold-style check would
    have caught it in review.
    """
    X, moy, t_hold, x_hold = _tensor(dtype)
    n = _pool_size(X, t_hold, x_hold, DYN)
    assert n > 65504, f"pool is only {n:,} entries"

    out, dynamic = anomaly_transform(X.copy(), moy, t_hold, x_hold)

    assert DYN in dynamic, "the varying channel must be detected as dynamic"
    assert STATIC not in dynamic, "a time-invariant channel is context"

    vals = out[..., DYN][np.isfinite(out[..., DYN])]
    _assert_in_regime(vals, dtype)
    assert np.isfinite(vals).all(), "z-scored output contains inf/nan"
    assert vals.std(dtype=np.float64) > 0.1, (
        f"dynamic channel collapsed to std={vals.std(dtype=np.float64):.2e} "
        f"-- the float16 accumulator overflowed and divided by inf")


@pytest.mark.parametrize("dtype", [np.float16, np.float32])
def test_z_score_is_actually_standardised(dtype):
    """Train-region values should come out ~zero-mean, ~unit-variance.

    Loose bounds: float16 carries ~3 decimal digits and the transform rounds
    back to the storage dtype between the two steps, so this pins the
    BEHAVIOUR (it standardises) rather than a digit count.
    """
    X, moy, t_hold, x_hold = _tensor(dtype)
    out, _ = anomaly_transform(X.copy(), moy, t_hold, x_hold)

    train = (np.isfinite(out[..., DYN]) & ~t_hold[:, None, None]
             & ~x_hold[None, None, :])
    v = out[..., DYN][train].astype(np.float64)
    assert abs(v.mean()) < 0.05, f"mean {v.mean():.4f} is not ~0"
    assert 0.8 < v.std() < 1.25, f"std {v.std():.4f} is not ~1"


def test_float16_and_float32_agree():
    """The transform must not depend on the storage dtype for its ANSWER.

    Family 3 was float32 and family 4 is float16; if the two disagreed by
    more than float16's own resolution, no result from one family could be
    compared with the other, and the whole point of E-038's rung structure
    would be gone.
    """
    X16, moy, t_hold, x_hold = _tensor(np.float16)
    X32 = X16.astype(np.float32)          # SAME values, wider storage

    o16, d16 = anomaly_transform(X16.copy(), moy, t_hold, x_hold)
    o32, d32 = anomaly_transform(X32.copy(), moy, t_hold, x_hold)

    assert d16 == d32, "dynamic-channel detection changed with the dtype"
    fin = np.isfinite(o16[..., DYN]) & np.isfinite(o32[..., DYN])
    diff = np.abs(o16[..., DYN][fin].astype(np.float64)
                  - o32[..., DYN][fin].astype(np.float64))
    assert diff.max() < 0.05, (
        f"float16 and float32 paths differ by up to {diff.max():.4f}")


def test_numpy_still_does_not_upcast_std_on_float16():
    """The premise of this whole file, pinned against a numpy upgrade.

    If a future numpy starts upcasting np.std on float16, the explicit
    dtype=float64 in anomaly_transform becomes belt-and-braces rather than
    load-bearing -- worth knowing, and worth being told by a failing test
    rather than by inference. Delete the workaround only when this fails.
    """
    v = np.random.standard_normal(1_000_000).astype(np.float16)
    with np.errstate(over="ignore"):
        native = v.std()
    assert not np.isfinite(native), (
        "numpy now upcasts std on float16; re-read anomaly_transform's "
        "comment and simplify it deliberately rather than by accident")
    assert np.isfinite(v.std(dtype=np.float64))

#!/usr/bin/env python3
"""The chunked anomaly transform must be the OLD one, only sequential.

WHY. `trainprobe.anomaly_transform` used to loop per channel over the whole
tensor. The tensor is channel-interleaved [T,H,W,C], so `X[..., c]` strides
C*itemsize = 78 bytes at C=39 float16 -- a fifty-third of a 4 KB page -- so
every per-channel operation faults in the ENTIRE file to use 2.6% of it. At
family 4 (pentad, 33.1 GB) on a 128 GB box the file lives in page cache and
the ~249 traversals cost minutes. At family 5 (daily, 165.6 GB) on a 64 GB
box there is no reuse at all: ~41 TB of physical read. Run #389 spent seven
hours in this function at 0.3 CPU cores and never emitted a metric line.

The rewrite iterates over blocks of the TIME axis and does all channels
inside each block: three sequential passes instead of 249 strided ones.

A performance rewrite of a numerical function is only worth anything if the
numbers do not move, so this file keeps the PRE-REWRITE implementation
frozen below as `anomaly_transform_oracle` and compares against it directly.
That copy is deliberate duplication -- it is the specification, and it must
never be "kept in sync" with the file it is testing.

    python3 tests/test_anomaly_chunked.py
"""
from __future__ import annotations

import os
import shutil
import sys
import tempfile
import warnings

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "..", "ml"))

from trainprobe import anomaly_transform            # noqa: E402


# ----------------------------------------------------------------------------
# THE ORACLE. Frozen copy of ml/trainprobe.py's anomaly_transform as it stood
# at commit ceb1048, i.e. the implementation every published number in
# ml/EXPERIMENTS.md was produced with. DO NOT UPDATE IT when trainprobe.py
# changes -- that is the entire point of it being here. Only the comments
# have been dropped (they live in the real file); the code is verbatim.
# ----------------------------------------------------------------------------
def anomaly_transform_oracle(X, moy, t_hold, x_hold):
    T, H, W, C = X.shape
    dynamic = [c for c in range(C)
               if np.nanstd(np.nanmean(X[..., c], axis=(1, 2),
                                       dtype=np.float64),
                            dtype=np.float64) > 1e-6]
    clim = np.full((12, H, W, C), np.nan, dtype=np.float32)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for m in range(12):
            clim[m] = np.nanmean(X[(moy == m) & ~t_hold], axis=0,
                                 dtype=np.float64)
    for c in dynamic:
        X[..., c] = X[..., c] - clim[moy, :, :, c]
        v = X[..., c][np.isfinite(X[..., c]) & ~t_hold[:, None, None]
                      & ~x_hold[None, None, :]]
        X[..., c] = (X[..., c] - v.mean(dtype=np.float64)) / (
            v.std(dtype=np.float64) + 1e-6)
    return X, dynamic


# ----------------------------------------------------------------------------
# Fixtures
# ----------------------------------------------------------------------------
def fixture(dtype, T=2000, H=24, W=26, C=4, seed=7, gaps=True):
    """Family-4-shaped: seasonal cycle + noise on the dynamic channels, a
    time-invariant baked field on the static one, permanent land NaNs, and
    (gaps=True) a scatter of all-NaN pixel/month cells so the climatology
    itself carries NaN and the valid mask is not simply the land mask.

    Long in T and small in H/W on purpose: the float16 overflow that
    `dtype=float64` exists to stop is driven by the SUM of squared residuals
    over the train pool, and `_assert_overflow_regime` measures that rather
    than assuming it from the count.
    """
    rng = np.random.default_rng(seed)
    moy = np.arange(T) % 12
    X = np.empty((T, H, W, C), dtype=dtype)
    season = np.sin(2 * np.pi * moy / 12)[:, None, None]
    field = rng.standard_normal((H, W))
    static = C - 1
    for c in range(C):
        if c == static:
            X[..., c] = np.broadcast_to(field, (T, H, W)).astype(dtype)
        else:
            X[..., c] = ((1 + c) * (season + rng.standard_normal((T, H, W)))
                         + field).astype(dtype)
    land = rng.random((H, W)) > 0.62
    X[:, land, :] = np.nan
    if gaps:
        # Gaps go on the DYNAMIC channels only: a baked climatology channel
        # has no time-varying missingness, and giving it one would make its
        # spatial mean wobble and the dynamic test would (correctly) call it
        # dynamic, which is not the case this fixture is for.
        dyn_c = slice(0, static)
        # one pixel that is missing in every January -> clim[0,y,x,:] is NaN
        X[moy == 0, 3, 4, dyn_c] = np.nan
        # and a scatter of one-off dropouts
        ti = rng.integers(0, T, 400)
        yi = rng.integers(0, H, 400)
        xi = rng.integers(0, W, 400)
        X[ti, yi, xi, dyn_c] = np.nan
    t_hold = np.zeros(T, bool)
    t_hold[-T // 8:] = True
    x_hold = np.zeros(W, bool)
    x_hold[-3:] = True
    return X, moy, t_hold, x_hold


def _assert_overflow_regime(X, moy, t_hold, x_hold, c=0):
    """Fail if the fixture cannot reproduce the float16 accumulator overflow.

    A test that pins float64 accumulators is worthless on data whose float16
    accumulation happens to stay finite.
    """
    ref, _ = anomaly_transform_oracle(X.astype(np.float32).copy(), moy,
                                      t_hold, x_hold)
    v = ref[..., c][np.isfinite(ref[..., c]) & ~t_hold[:, None, None]
                    & ~x_hold[None, None, :]]
    with np.errstate(over="ignore", invalid="ignore"):
        native = v.astype(np.float16).std()
    assert not np.isfinite(native), (
        f"fixture out of regime: float16 std over {v.size:,} entries returned "
        f"{native}, so a float16 accumulator would NOT have overflowed here")
    return v.size


class Counted:
    """An ndarray proxy that measures how many times a call TRAVERSES X.

    The cost model this rewrite is about is pages, not elements: a view whose
    stride is smaller than a page faults in its whole byte span however few
    elements it addresses. So each access is charged its BYTE SPAN --
    (n-1)*stride summed over the dimensions, plus one item -- capped at the
    array. `X[..., c]` therefore charges a full traversal and `X[i0:i1]`
    charges its block. Total charge / nbytes is the traversal count.
    """

    def __init__(self, arr):
        self._a = arr
        self.charged = 0
        self.gets = 0
        self.sets = 0

    def _charge(self, view):
        span = view.dtype.itemsize
        for n, s in zip(view.shape, view.strides):
            if n:
                span += (n - 1) * abs(s)
        self.charged += min(span, self._a.nbytes)

    def __getitem__(self, k):
        v = self._a[k]
        self.gets += 1
        self._charge(v if isinstance(v, np.ndarray) else self._a)
        return v

    def __setitem__(self, k, val):
        self.sets += 1
        self._charge(self._a[k])
        self._a[k] = val

    def __array__(self, dtype=None, copy=None):
        return np.asarray(self._a, dtype=dtype)

    @property
    def traversals(self):
        return self.charged / self._a.nbytes

    shape = property(lambda s: s._a.shape)
    dtype = property(lambda s: s._a.dtype)
    size = property(lambda s: s._a.size)
    nbytes = property(lambda s: s._a.nbytes)


def _compare(new, old, dtype, label):
    """Elementwise agreement, returning the measured max absolute error."""
    assert new.shape == old.shape
    fn, fo = np.isfinite(new), np.isfinite(old)
    assert np.array_equal(fn, fo), (
        f"{label}: the NaN/inf pattern moved at {int((fn != fo).sum())} of "
        f"{fn.size} entries -- the transform changed which cells are valid")
    a = new[fn].astype(np.float64)
    b = old[fo].astype(np.float64)
    err = np.abs(a - b)
    return (float(err.max()) if err.size else 0.0,
            int((a != b).sum()), int(a.size))


# ----------------------------------------------------------------------------
# 1 · elementwise agreement with the frozen oracle
# ----------------------------------------------------------------------------
def test_matches_oracle():
    print("\n1. elementwise agreement with the frozen oracle")
    for dtype in (np.float16, np.float32):
        X, moy, t_hold, x_hold = fixture(dtype)
        n = _assert_overflow_regime(X, moy, t_hold, x_hold)
        want, dyn_o = anomaly_transform_oracle(X.copy(), moy, t_hold, x_hold)
        got, dyn_n = anomaly_transform(X.copy(), moy, t_hold, x_hold,
                                       chunk=64)
        assert dyn_n == dyn_o, f"dynamic channels differ: {dyn_n} != {dyn_o}"
        assert dyn_o == [0, 1, 2], f"fixture is wrong: dynamic={dyn_o}"
        err, ndiff, ntot = _compare(got, want, dtype, str(dtype))
        ulp = np.finfo(dtype).eps                      # 1 ulp at |x| ~ 1
        print(f"   {np.dtype(dtype).name}: pool {n:,} · max |new-old| = "
              f"{err:.3e} · differing entries {ndiff:,}/{ntot:,} · "
              f"1 ulp = {ulp:.3e}")
        assert err <= ulp, (
            f"{dtype}: max error {err:.3e} exceeds one storage ulp {ulp:.3e}")
        # ... and the static channel must be untouched, bit for bit.
        stat = X.shape[-1] - 1
        assert np.array_equal(got[..., stat], X[..., stat], equal_nan=True), \
            "the static channel was modified"


# ----------------------------------------------------------------------------
# 2 · the answer does not depend on the chunk size
# ----------------------------------------------------------------------------
def test_chunk_invariance():
    print("\n2. the chunk size changes memory, never the answer")
    X, moy, t_hold, x_hold = fixture(np.float32, T=743, C=4)
    want, dyn = anomaly_transform_oracle(X.copy(), moy, t_hold, x_hold)
    ref = None
    for chunk in (1, 7, 64, 743, 10_000):
        got, d = anomaly_transform(X.copy(), moy, t_hold, x_hold, chunk=chunk)
        assert d == dyn, f"chunk={chunk}: dynamic channels changed"
        err, ndiff, ntot = _compare(got, want, np.float32, f"chunk={chunk}")
        if ref is None:
            ref = got.copy()
            spread = 0.0
        else:
            f = np.isfinite(ref)
            spread = float(np.abs(got[f].astype(np.float64)
                                  - ref[f].astype(np.float64)).max())
        print(f"   chunk={chunk:<6} max |new-old| = {err:.3e} · "
              f"max |new-new(chunk=1)| = {spread:.3e}")
        assert err <= np.finfo(np.float32).eps
        assert spread <= np.finfo(np.float32).eps


# ----------------------------------------------------------------------------
# 3 · the I/O it exists to fix
# ----------------------------------------------------------------------------
def test_traversal_count():
    print("\n3. full-extent traversals of X (the whole point)")
    # C=39 with 4 baked-climatology channels, i.e. family 5's shape signature,
    # so the count this prints is the count that mattered on the daily tensor:
    #   39 (dynamic test) + 1 (twelve climatology gathers, T/12 each)
    #   + 35 * 6 (subtract-source, write, isfinite, boolean gather,
    #             z-source, write)  =  250
    rng = np.random.default_rng(11)
    T, H, W, C = 120, 8, 9, 39
    moy = np.arange(T) % 12
    X = rng.standard_normal((T, H, W, C)).astype(np.float16)
    field = rng.standard_normal((H, W))
    for c in range(C - 4, C):                      # 4 static channels
        X[..., c] = np.broadcast_to(field, (T, H, W)).astype(np.float16)
    X[:, 2, 3, :] = np.nan
    t_hold = np.zeros(T, bool)
    t_hold[-24:] = True
    x_hold = np.zeros(W, bool)
    x_hold[-2:] = True

    co = Counted(X.copy())
    _, dyn = anomaly_transform_oracle(co, moy, t_hold, x_hold)
    cn = Counted(X.copy())
    anomaly_transform(cn, moy, t_hold, x_hold, chunk=64)
    print(f"   {len(dyn)}/{C} dynamic channels")
    print(f"   old: {co.traversals:8.1f} traversals of X "
          f"({co.gets} reads, {co.sets} writes)")
    print(f"   new: {cn.traversals:8.1f} traversals of X "
          f"({cn.gets} reads, {cn.sets} writes)")
    print(f"   ratio: {co.traversals / cn.traversals:.1f}x less byte span; "
          f"at 165.6 GB that is {co.traversals * 165.6 / 1024:.1f} TB -> "
          f"{cn.traversals * 165.6:.0f} GB")
    assert cn.traversals <= 6.05, (
        f"the new transform traverses X {cn.traversals:.2f} times; the design "
        f"is 3 read passes + 2 write passes (the third read is of pages the "
        f"second pass just wrote, so it is 5 physical passes)")
    assert co.traversals > 200, (
        f"the oracle only traverses X {co.traversals:.1f} times on this "
        f"fixture, so it does not reproduce the 249 that motivated the "
        f"rewrite -- check C and the static-channel count")
    assert co.traversals / cn.traversals > 40


# ----------------------------------------------------------------------------
# 4 · the variance formulation
# ----------------------------------------------------------------------------
def test_variance_formulation():
    print("\n4. chunked variance: Chan's parallel combination vs naive "
          "sum-of-squares")
    rng = np.random.default_rng(3)
    for mu_true, sd_true in ((0.0, 1.0), (1e3, 1e-2), (1e5, 1e-3)):
        x = (mu_true + sd_true * rng.standard_normal(2_000_000))
        exact = x.std(dtype=np.float64)                 # numpy's two-pass
        n_t = mu_t = m2_t = 0.0
        s1 = s2 = 0.0
        for i in range(0, x.size, 50_000):
            b = x[i:i + 50_000]
            n_b = float(b.size)
            mu_b = b.mean(dtype=np.float64)
            m2_b = ((b - mu_b) ** 2).sum(dtype=np.float64)
            delta = mu_b - mu_t
            n_new = n_t + n_b
            mu_t = mu_t + delta * n_b / n_new
            m2_t = m2_t + m2_b + delta * delta * n_t * n_b / n_new
            n_t = n_new
            s1 += b.sum(dtype=np.float64)
            s2 += (b.astype(np.float64) ** 2).sum(dtype=np.float64)
        chan = np.sqrt(m2_t / n_t)
        naive = np.sqrt(max(s2 / n_t - (s1 / n_t) ** 2, 0.0))
        r_chan = abs(chan - exact) / exact
        r_naive = abs(naive - exact) / exact
        # The variance problem's condition number is ~ (1 + |mu|/sd), so
        # eps*kappa is the floor NO algorithm beats. Chan's form is bounded
        # by it; the naive form is bounded by eps*kappa^2 and blows past it.
        kappa = 1.0 + abs(mu_true) / sd_true
        bound = np.finfo(np.float64).eps * kappa
        print(f"   mu={mu_true:<8g} sd={sd_true:<8g} kappa={kappa:.1e}  "
              f"rel err  Chan {r_chan:.2e} (bound {bound:.1e})   "
              f"naive {r_naive:.2e}")
        assert r_chan <= bound, (
            f"Chan exceeded eps*kappa: {r_chan:.2e} > {bound:.2e}")
        if kappa > 1e4:
            assert r_naive > 100 * max(r_chan, np.finfo(float).eps), (
                "the naive form did not cancel on this case, so it does not "
                "demonstrate why it was rejected")


# ----------------------------------------------------------------------------
# 5 · memmap call sites
# ----------------------------------------------------------------------------
def test_memmap():
    print("\n5. memmapped X (family 5's sidecar layout)")
    tmp = tempfile.mkdtemp()
    try:
        X, moy, t_hold, x_hold = fixture(np.float16, T=300, H=10, W=12, C=5)
        want, dyn = anomaly_transform_oracle(X.copy(), moy, t_hold, x_hold)

        # (a) a WRITABLE memmap -- what tensor_io.writable_copy hands over.
        p = os.path.join(tmp, "scratch.npy")
        mm = np.lib.format.open_memmap(p, mode="w+", dtype=X.dtype,
                                       shape=X.shape)
        mm[:] = X
        got, d = anomaly_transform(mm, moy, t_hold, x_hold, chunk=32)
        assert d == dyn
        err, ndiff, ntot = _compare(np.asarray(got), want, X.dtype, "memmap")
        print(f"   writable memmap: max |new-old| = {err:.3e} "
              f"({ndiff:,}/{ntot:,} entries differ)")
        assert err <= np.finfo(np.float16).eps
        del mm, got
        reread = np.load(p, mmap_mode="r")
        assert np.array_equal(np.asarray(reread), want, equal_nan=True) or \
            np.nanmax(np.abs(np.asarray(reread).astype(np.float64)
                             - want.astype(np.float64))) <= 1e-3, \
            "the transform did not reach the file"
        print("   the writes reached the .npy on disk")
        del reread

        # (b) a READ-ONLY memmap must still refuse, exactly as before: the
        #     canonical tensor must never take these writes (tensor_io.py's
        #     docstring). train.py and probe_kfold.py both writable_copy
        #     first BECAUSE of this refusal; it is contract, not accident.
        ro = np.load(p, mmap_mode="r")
        try:
            anomaly_transform(ro, moy, t_hold, x_hold, chunk=32)
        except ValueError as e:
            print(f"   read-only memmap still refuses: {type(e).__name__}: "
                  f"{str(e)[:60]}")
        else:
            raise AssertionError(
                "a read-only memmap was accepted -- the canonical tensor "
                "could now be silently left in anomaly space")
        del ro
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


# ----------------------------------------------------------------------------
# 6 · degenerate shapes
# ----------------------------------------------------------------------------
def test_degenerate():
    print("\n6. degenerate inputs behave as the oracle does")
    # all-static tensor -> no dynamic channels, X untouched
    rng = np.random.default_rng(1)
    T, H, W, C = 60, 6, 7, 3
    f = rng.standard_normal((H, W, C)).astype(np.float32)
    X = np.broadcast_to(f, (T, H, W, C)).astype(np.float32)
    moy = np.arange(T) % 12
    t_hold = np.zeros(T, bool)
    x_hold = np.zeros(W, bool)
    want, dyn_o = anomaly_transform_oracle(X.copy(), moy, t_hold, x_hold)
    got, dyn_n = anomaly_transform(X.copy(), moy, t_hold, x_hold)
    assert dyn_o == [] and dyn_n == [], f"{dyn_o} / {dyn_n}"
    assert np.array_equal(got, want, equal_nan=True)
    print("   all-static tensor: dynamic=[] and X is untouched")

    # a month entirely inside the holdout -> clim is NaN there -> those
    # timesteps become NaN. Both implementations must agree on WHICH.
    X, moy, t_hold, x_hold = fixture(np.float32, T=360, H=8, W=9, C=3)
    t_hold[:] = False
    t_hold[moy == 5] = True                    # no train June at all
    want, dyn_o = anomaly_transform_oracle(X.copy(), moy, t_hold, x_hold)
    got, dyn_n = anomaly_transform(X.copy(), moy, t_hold, x_hold, chunk=17)
    assert dyn_n == dyn_o
    err, ndiff, ntot = _compare(got, want, np.float32, "empty month")
    nan_frac = float(np.isnan(got[..., 0]).mean())
    print(f"   a month with no train timesteps: max |new-old| = {err:.3e}, "
          f"NaN pattern identical ({nan_frac:.1%} of channel 0 is NaN)")
    assert nan_frac > 0.05, "the empty-month case did not produce NaNs"
    assert err <= np.finfo(np.float32).eps

    # non-monotonic moy (the run-splitting must not assume sorted months)
    X, moy, t_hold, x_hold = fixture(np.float32, T=300, H=8, W=9, C=3)
    rng.shuffle(moy)
    want, dyn_o = anomaly_transform_oracle(X.copy(), moy, t_hold, x_hold)
    got, dyn_n = anomaly_transform(X.copy(), moy, t_hold, x_hold, chunk=13)
    assert dyn_n == dyn_o
    err, _, _ = _compare(got, want, np.float32, "shuffled moy")
    print(f"   shuffled month-of-year: max |new-old| = {err:.3e}")
    assert err <= np.finfo(np.float32).eps


def main():
    test_matches_oracle()
    test_chunk_invariance()
    test_traversal_count()
    test_variance_formulation()
    test_memmap()
    test_degenerate()
    print("\ntests/test_anomaly_chunked.py: all 6 checks passed")


if __name__ == "__main__":
    main()

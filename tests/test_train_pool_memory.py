#!/usr/bin/env python3
"""Pin the training pool's construction: same values, same order, less memory.

`LazyPixels` removed the two RESIDENT full-size copies that killed run #365.
It did not touch the trainer's actual PEAK, which `ml/measure_train_memory.py`
located one line further up:

    obs_any = np.isfinite(X).sum(-1) >= 2          # a [T,H,W,C] bool AND a
                                                   # [T,H,W] int64, together
    tt, yy, xx = np.where(...)                     # 3 x int64 over ~272M px

On family 4 that is 15.4 + 3.4 GiB of transient and 6.5 GiB that stays
resident for the whole run; at daily cadence, 77 + 17 and 33. Fixing only the
copies you can see in an RSS delta column would have re-killed the dispatch on
the same box for a different reason.

`obs_any_chunked` and `pool_idx` (ml/model.py) replace them. The bar Chris set
for the dispatch refactor applies here too — *"making sure it's not changing
behavior at all"* — so this file asserts equality against the originals rather
than reasoning about it:

  1. **`obs_any_chunked(X)` is elementwise identical** to
     `np.isfinite(X).sum(-1) >= 2`, at several chunk sizes including ones that
     do not divide T, and for both dtypes in use.
  2. **`pool_idx(mask)` returns exactly `np.where(mask)`** — same triples in
     the same ORDER, which is what makes the two pools interchangeable at all
     (the trainer draws `np.random.randint(0, len(idx_t))`, so a permuted pool
     would silently change which pixels a seed selects).
  3. **The indices are int32 and the values survive the narrowing**, checked
     against the largest axis any planned tensor has (daily T = 15,706).
  4. **The memory is actually lower** — MEASURED with VmHWM, not asserted from
     the code, because the whole point of this change is a peak that never
     appears in an RSS delta.
  5. **The trainer's own call site round-trips**: an int32 pool indexed by the
     trainer's `batch()` arithmetic produces the identical int64 torch tensors
     the int64 pool produced.

    python3 tests/test_train_pool_memory.py
"""
import gc
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "..", "ml"))

from model import obs_any_chunked, pool_idx          # noqa: E402

GIB = 1024 ** 3


def hwm_gib():
    """VmHWM — the high-water mark the OOM killer acts on. Survives frees."""
    with open("/proc/self/status") as fh:
        for line in fh:
            if line.startswith("VmHWM:"):
                return int(line.split()[1]) * 1024 / GIB
    return float("nan")


MEASURE = r"""
import gc, os, sys
import numpy as np
sys.path.insert(0, {ml!r})
from model import obs_any_chunked, pool_idx

def hwm():
    for line in open("/proc/self/status"):
        if line.startswith("VmHWM:"):
            return int(line.split()[1]) * 1024 / (1024 ** 3)

path, T, H, W, C = sys.argv[1], *[int(v) for v in sys.argv[2:6]]
# np.full, not np.zeros: calloc hands back lazily-zeroed pages that never
# enter RSS until written, which would put X's own cost inside whichever
# path touches it first and make the comparison meaningless. And a LAND
# fraction, so the pool is a minority of the volume the way it is on the
# real tensor (86,698 ocean cells of 135,161) rather than every pixel.
X = np.full((T, H, W, C), 1.0, np.float16)
land = np.random.default_rng(0).random((H, W)) < 0.36
X[:, land, :] = np.nan
gc.collect()
base = hwm()
if path == "chunked":
    m = obs_any_chunked(X)
    idx = pool_idx(m)
else:
    m = np.isfinite(X).sum(-1) >= 2
    idx = np.where(m)
print(f"{{hwm() - base:.6f}}")
"""


def _subprocess_peak(path, T, H, W, C):
    """VmHWM cost of one path, in a process that has allocated nothing else."""
    import subprocess
    ml = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "ml")
    p = subprocess.run([sys.executable, "-c", MEASURE.format(ml=ml), path,
                        str(T), str(H), str(W), str(C)],
                       capture_output=True, text=True)
    if p.returncode:
        print(p.stderr, file=sys.stderr)
        raise SystemExit(f"the {path} measurement exited {p.returncode}")
    return float(p.stdout.strip().splitlines()[-1])


def main():
    rng = np.random.default_rng(20260817)

    # ---- 1: obs_any_chunked == the one-liner ------------------------------
    for dtype in (np.float32, np.float16):
        T, H, W, C = 37, 11, 13, 39
        X = rng.normal(size=(T, H, W, C)).astype(dtype)
        # three kinds of missing, because the >= 2 threshold is what separates
        # them: fully observed, fully land, and pixels with exactly 0/1/2
        # observed channels — the boundary the comparison is written on.
        X[rng.random(X.shape) < 0.3] = np.nan
        X[:, 0, 0, :] = np.nan                                   # land
        X[:, 1, 1, :] = np.nan
        X[:, 1, 1, 0] = 1.0                                      # exactly 1
        X[:, 2, 2, :] = np.nan
        X[:, 2, 2, :2] = 1.0                                     # exactly 2
        want = np.isfinite(X).sum(-1) >= 2
        assert want[:, 1, 1].sum() == 0, "fixture: the 1-channel pixel is in"
        assert want[:, 2, 2].all(), "fixture: the 2-channel pixel is out"
        for chunk in (1, 5, 8, 37, 64):        # 8 and 5 do not divide T=37
            got = obs_any_chunked(X, chunk=chunk)
            assert got.dtype == np.bool_, got.dtype
            assert np.array_equal(got, want), \
                f"{dtype.__name__} chunk={chunk}: obs_any differs"
    print("  1. obs_any_chunked == isfinite(X).sum(-1)>=2 elementwise "
          "(float32 + float16, 5 chunk sizes, 2 that do not divide T)")

    # ---- 2 & 3: pool_idx == np.where, in order, as int32 -------------------
    mask = rng.random((41, 17, 19)) < 0.4
    t_hold = rng.random(41) < 0.25
    x_hold = rng.random(19) < 0.2
    for m in (mask,
              mask & ~t_hold[:, None, None] & ~x_hold[None, None, :],
              mask & (t_hold[:, None, None] | x_hold[None, None, :]),
              np.zeros_like(mask)):                     # the empty pool
        want = np.where(m)
        for chunk in (1, 7, 41, 256):
            got = pool_idx(m, chunk=chunk)
            for g, w, name in zip(got, want, "tyx"):
                assert g.dtype == np.int32, f"{name} is {g.dtype}, not int32"
                assert np.array_equal(g.astype(np.int64), w), \
                    f"chunk={chunk}: the {name} index differs from np.where"
    print("  2. pool_idx == np.where — same triples, same order, 4 masks x 4 "
          "chunk sizes (empty pool included)")

    # every index any planned tensor can produce fits int32 with room to spare
    for axis in (15706, 281, 481):                      # daily T, H, W
        assert axis < np.iinfo(np.int32).max, axis
    big = np.zeros((15706, 2, 2), bool)
    big[-1, -1, -1] = True
    tb, yb, xb = pool_idx(big)
    assert (int(tb[0]), int(yb[0]), int(xb[0])) == (15705, 1, 1), \
        "the largest daily index did not survive the narrowing"
    del big
    print("  3. int32 holds every index the daily axis can produce "
          f"(T-1 = 15,705 round-trips; int32 max is {np.iinfo(np.int32).max:,})")

    # ---- 4: the peak is MEASURED lower ------------------------------------
    # Sized so a full-size bool is unmistakable in VmHWM but the fixture still
    # runs in the sandbox (~0.5 GiB of float16), with T far enough above the
    # 256-timestep chunk that the chunk is a small SLICE of the tensor the way
    # it is on the real one — a fixture whose chunk is most of T measures the
    # chunking doing nothing, which is a property of the fixture. EACH PATH GETS ITS OWN
    # PROCESS: VmHWM is a high-water mark and never falls, so a second
    # measurement in this process would sit under the mark the first one (and
    # every fixture above) already set, and read as 0.00 GiB for both. That is
    # exactly the blindness this check exists to correct, one level up.
    T2, H2, W2, C2 = 2000, 60, 60, 39
    nbytes = T2 * H2 * W2 * C2 * 2 / GIB
    cost_chunked = _subprocess_peak("chunked", T2, H2, W2, C2)
    cost_eager = _subprocess_peak("eager", T2, H2, W2, C2)

    assert cost_eager > nbytes * 0.5, (
        f"the eager path only moved VmHWM by {cost_eager:.2f} GiB against a "
        f"{nbytes:.2f} GiB tensor — this measurement cannot tell the paths "
        f"apart and is therefore not a check")
    assert cost_chunked < cost_eager * 0.5, (
        f"chunked peak {cost_chunked:.3f} GiB vs eager {cost_eager:.3f} GiB — "
        f"not the reduction this change exists for")
    # The STRUCTURAL claim, and the one that decides whether the daily arm can
    # run at all: the eager path's peak is at least one full [T,H,W,C] bool
    # (15.4 GiB on family 4, 77 GiB at daily) and the chunked path's is never
    # anywhere near one, whatever T is.
    full_bool = T2 * H2 * W2 * C2 / GIB
    assert cost_eager > full_bool * 0.9, (
        f"eager peak {cost_eager:.3f} GiB did not even reach the "
        f"{full_bool:.3f} GiB bool it materialises — the measurement is not "
        f"seeing what it claims to")
    assert cost_chunked < full_bool * 0.9, (
        f"chunked peak {cost_chunked:.3f} GiB is the size of a full-tensor "
        f"bool — the term that scales with T is still there")
    print(f"  4. VmHWM over a {nbytes:.2f} GiB tensor: chunked "
          f"{cost_chunked:.3f} GiB vs eager {cost_eager:.3f} GiB "
          f"({cost_eager / max(cost_chunked, 1e-3):.1f}x), and the chunked "
          f"peak stays under the {full_bool:.3f} GiB full-tensor bool the "
          f"eager path must build")


    # ---- 5: the trainer's batch() arithmetic is unchanged ------------------
    # Same draw, same pixels, same dtypes leaving the function.
    m = rng.random((23, 9, 11)) < 0.5
    i32 = pool_idx(m)
    i64 = np.where(m)
    lats = np.linspace(0, 70, 9)
    lons = np.linspace(-100, 20, 11)
    ctx_all = rng.normal(size=(23, 2))

    def batch(idx, n, seed):
        st = np.random.RandomState(seed)
        k = st.randint(0, len(idx[0]), n)
        t, y, x = idx[0][k], idx[1][k], idx[2][k]
        ctx = np.concatenate([ctx_all[t], (lats[y] / 90)[:, None],
                              (lons[x] / 180)[:, None]], 1)
        return (torch.as_tensor(t, dtype=torch.long),
                torch.as_tensor(y, dtype=torch.long),
                torch.as_tensor(x, dtype=torch.long),
                torch.as_tensor(ctx, dtype=torch.float32))

    a = batch(i32, 256, 7)
    b = batch(i64, 256, 7)
    for g, w, name in zip(a, b, ("t", "y", "x", "ctx")):
        assert g.dtype == w.dtype, f"{name}: {g.dtype} vs {w.dtype}"
        assert torch.equal(g, w), f"{name}: the batch differs"
    assert a[0].dtype == torch.long, "indices must leave batch() as int64"
    print("  5. batch() over the int32 pool returns tensors bit-identical to "
          "the int64 pool's, dtype torch.long included")

    print("\ntests/test_train_pool_memory.py: all 5 checks passed")


if __name__ == "__main__":
    main()

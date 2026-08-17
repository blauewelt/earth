#!/usr/bin/env python3
"""Pin LazyPixels: identical values, and the memory that made it necessary.

Run #365 — the first pentad codec — was killed by the host OOM killer (exit
137) after six hours on a 64 GB box. `ml/train.py` built two full-size derived
arrays before training:

    Xt  = torch.from_numpy(np.nan_to_num(X, nan=0.0))     33.1 GB
    OBS = torch.from_numpy(np.isfinite(X))                16.6 GB

on top of X's own 33.1 GB. **82.8 GB against 64 GB.** For family 3 the same
three arrays are 13.6 GB, which is why it never bit before. The element count
is 16,562,358,618 — measured from the real shape, not estimated.

Both derived arrays are elementwise pure functions of X, and every consumer
only ever indexes a batch of pixels out of them, so evaluating them AFTER the
index is arithmetically identical and costs a few hundred KB. That removes the
failure mode instead of guarding it (ml/CLAUDE.md §4.1) — which matters
because the daily tensor is 5x larger again and would not fit any box we own
under the old shape.

What is pinned:

  1. **Values are identical to the eager arrays**, elementwise, for every
     indexing form the six call sites use — including the `gather_px` patch>1
     path, which indexes with clamped/wrapped torch tensors and then does
     `& vy.unsqueeze(-1)` on the result, so the mask must come back as a real
     torch bool tensor.
  2. **dtype follows X**, exactly as `torch.from_numpy` did. A float16 tensor
     must still yield float16: silently promoting to float32 would double
     activation memory on the very box this is meant to fit.
  3. **No full-size copy is ever made.** Asserted by measuring peak RSS across
     a large-ish tensor rather than by reading the code — the eager form is
     built in the same process for comparison, and the lazy one must cost a
     small fraction of it.
  4. **`.shape` is present**, because `gather_px` reads `.shape[1]` and
     `.shape[2]` to clamp latitude and wrap longitude.

    python3 tests/test_train_lazy_pixels.py
"""
import os
import resource
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "ml"))

from model import LazyPixels, gather_px          # noqa: E402


def peak_mb():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e3


def main():
    rng = np.random.default_rng(20260817)

    # ---- 1 & 2: identical values and dtype, for both dtypes in use --------
    for dtype in (np.float32, np.float16):
        T, H, W, C = 40, 21, 23, 39
        X = rng.normal(size=(T, H, W, C)).astype(dtype)
        X[rng.random(X.shape) < 0.3] = np.nan          # land + missing tokens

        eager_v = torch.from_numpy(np.nan_to_num(X, nan=0.0))
        eager_o = torch.from_numpy(np.isfinite(X))
        lazy_v, lazy_o = LazyPixels(X), LazyPixels(X, obs=True)

        t = torch.randint(0, T, (128,))
        y = torch.randint(0, H, (128,))
        x = torch.randint(0, W, (128,))

        gv, ge = lazy_v[t, y, x], eager_v[t, y, x]
        assert gv.dtype == ge.dtype == torch.from_numpy(X[:1]).dtype, \
            f"dtype drifted: {gv.dtype} vs {ge.dtype}"
        assert torch.equal(gv, ge), f"{dtype.__name__}: values differ"
        assert torch.equal(lazy_o[t, y, x], eager_o[t, y, x]), \
            f"{dtype.__name__}: observation mask differs"
        # the mask must support the boolean op gather_px performs on it
        vy = torch.ones(128, dtype=torch.bool)
        _ = lazy_o[t, y, x] & vy.unsqueeze(-1)
    print("  1+2. values and dtype identical to the eager arrays "
          "(float32 and float16), and the mask supports `&`")

    # ---- 4 then 1 again: the whole gather_px path, patch=1 and patch=3 ----
    T, H, W, C = 40, 21, 23, 39
    X = rng.normal(size=(T, H, W, C)).astype(np.float32)
    X[rng.random(X.shape) < 0.3] = np.nan
    ev, eo = torch.from_numpy(np.nan_to_num(X, nan=0.0)), torch.from_numpy(np.isfinite(X))
    lv, lo = LazyPixels(X), LazyPixels(X, obs=True)
    assert lv.shape == ev.shape, "gather_px reads .shape[1]/.shape[2]"
    t = torch.randint(0, T, (64,))
    y = torch.randint(0, H, (64,))
    x = torch.randint(0, W, (64,))
    for patch in (1, 3):
        a_v, a_o = gather_px(ev, eo, t, y, x, patch)
        b_v, b_o = gather_px(lv, lo, t, y, x, patch)
        assert torch.equal(a_v, b_v) and torch.equal(a_o, b_o), \
            f"gather_px patch={patch} differs between eager and lazy"
    print("  3. gather_px agrees at patch=1 and patch=3 — the clamped/wrapped "
          "neighbourhood path included")

    # ---- 3: no full-size copy, MEASURED ----------------------------------
    # Big enough that a full copy is unmistakable in RSS, small enough to run
    # in the sandbox: ~260 MB of float32. Sized DOWN after the first version
    # (1.2 GB) was itself OOM-killed with exit 137 here — the same signal it
    # exists to catch, which is a good sign for the check and a bad size for
    # the fixture.
    T2 = 200
    Xb = np.zeros((T2, 128, 128, 20), np.float32)
    base = peak_mb()
    lazy = LazyPixels(Xb)
    lazy_obs = LazyPixels(Xb, obs=True)
    t = torch.randint(0, T2, (4096,))
    y = torch.randint(0, 128, (4096,))
    x = torch.randint(0, 128, (4096,))
    for _ in range(20):
        _ = lazy[t, y, x]
        _ = lazy_obs[t, y, x]
    after_lazy = peak_mb()
    lazy_cost = after_lazy - base

    eager = torch.from_numpy(np.nan_to_num(Xb, nan=0.0))
    eager_obs = torch.from_numpy(np.isfinite(Xb))
    eager_cost = peak_mb() - after_lazy
    nbytes = Xb.size * Xb.itemsize / 1e6

    assert lazy_cost < nbytes * 0.10, (
        f"the lazy view cost {lazy_cost:.0f} MB against a {nbytes:.0f} MB "
        f"tensor — something is materialising a full copy")
    assert eager_cost > nbytes * 0.5, (
        f"the eager form only cost {eager_cost:.0f} MB, so this measurement "
        f"cannot tell the two apart and is not a check")
    print(f"  4. lazy costs {lazy_cost:.0f} MB over 20 batches against a "
          f"{nbytes:.0f} MB tensor; the eager form costs {eager_cost:.0f} MB "
          f"({eager_cost / max(lazy_cost, 1):.0f}x more)")
    del eager, eager_obs

    print("\ntests/test_train_lazy_pixels.py: all 4 checks passed")


if __name__ == "__main__":
    main()

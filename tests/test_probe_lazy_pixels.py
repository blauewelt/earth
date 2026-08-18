#!/usr/bin/env python3
"""Pin the three POST-TRAINING PROBES on LazyPixels: same numbers, no copies.

Run #388 (2026-08-17) had all three probes OOM-killed on the GPU box, each one
seconds after the ~31-minute anomaly transform, when the next large allocation
landed:

    ml/probe_sequence.py  — Killed
    ml/probe_kfold.py     — Killed
    ml/dip_check.py       — Killed

Same pattern `LazyPixels` was written for in `ml/train.py` (run #365), still
live in the probe ladder because the fix was applied where it was diagnosed:

    OBS = torch.from_numpy(np.isfinite(Xa))     # 16.6 GB pentad / 83 GB daily
    Xt  = torch.from_numpy(np.nan_to_num(Xa))   # another 33.1 / 165.6 GB in
                                                # probe_sequence, which copies

on top of the tensor's own 33.1 GB ([3142, 281, 481, 39] float16) — and 5x
that at daily cadence, where no box we rent could hold it under the old shape.
Both arrays are elementwise pure functions of `Xa` and every consumer only ever
indexes a BATCH of pixels out of them, so evaluating them after the index is
arithmetically identical (ml/CLAUDE.md §4.1 — remove the failure mode, don't
guard it).

The conversion has one trap, and it is silent, which is why check 1 is an
end-to-end equality rather than a spot check on `LazyPixels` itself: the eager
form filled the NaNs **in place** (`np.nan_to_num(Xa, copy=False)`) after
taking the mask. Leave that line in beside `LazyPixels(Xa, obs=True)` and the
mask is computed per batch over an array with no NaNs left — all-True, every
land cell and every unobserved channel entering the encoder as an observed 0.0
instead of a missing token. No error, no NaN, just different embeddings.

What is pinned:

  1. **`embed_everything` returns bit-identical embeddings** built from
     LazyPixels and from the eager torch tensors — patch=1, patch=3 (the
     clamped/wrapped `gather_px` neighbourhood) and the `mask_chan` ablation
     path, through the real `ml/temporal.py` function and a real `PixelMAE`.
     Plus the falsifier for the trap above: an all-True mask must MOVE the
     embedding on this fixture, or the equality is vacuous.
  2. **Every indexing form the three scripts use still works and still
     agrees** — including `OBS[:, sec_y, :, 0].any(axis=0).numpy()`, which is
     how probe_sequence.py picks its section, and the `X[t, y, x] * (~mk)`
     bool-promotion multiply inside `embed_everything`.
  3. **The memory is actually lower** — VmHWM, measured in a fresh subprocess
     per path, because a high-water mark never falls and a second measurement
     in the same process would read 0.00 GiB for both.

    python3 tests/test_probe_lazy_pixels.py
"""
import contextlib
import io
import os
import subprocess
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "..", "ml"))

from model import LazyPixels, PixelMAE, gather_px          # noqa: E402
from temporal import embed_everything, section_of          # noqa: E402

GIB = 1024 ** 3


def fixture(T=40, H=21, W=23, C=39, seed=20260818, dtype=np.float16):
    """A toy [T,H,W,C] tensor with the two kinds of missing the real one has:
    LAND (a cell that is NaN in every channel at every month, which is what
    `ocean` is derived from) and FLICKER (a channel unobserved in some months
    only — the case the observation mask exists for)."""
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(T, H, W, C)).astype(dtype)
    land = rng.random((H, W)) < 0.30
    X[:, land, :] = np.nan
    X[rng.random(X.shape) < 0.10] = np.nan
    lats = np.linspace(0.0, 70.0, H)
    lons = np.linspace(-100.0, 20.0, W)
    return X, lats, lons


def tiny_codec(C, d_z=8, patch=1, seed=7):
    torch.manual_seed(seed)
    m = PixelMAE(n_chan=C, d_model=16, n_heads=2, n_layers=1, d_z=d_z,
                 d_dec=16, patch=patch, dec_layers=1)
    m.eval()
    return m


def quiet(fn, *args, **kw):
    """embed_everything narrates its progress every 5% — 20 lines here."""
    with contextlib.redirect_stdout(io.StringIO()):
        return fn(*args, **kw)


# ---------------------------------------------------------------------------
# Check 3's measurement, run in its own process per path (see the pool-memory
# test: VmHWM is a high-water mark and never falls).
MEASURE = r"""
import gc, sys
import numpy as np
import torch
sys.path.insert(0, {ml!r})
from model import LazyPixels

def hwm():
    for line in open("/proc/self/status"):
        if line.startswith("VmHWM:"):
            return int(line.split()[1]) * 1024 / (1024 ** 3)

path, T, H, W, C = sys.argv[1], *[int(v) for v in sys.argv[2:6]]
# np.full, not np.zeros: calloc hands back lazily-zeroed pages that never enter
# RSS until written, which would charge X's own cost to whichever path touches
# it first and make the comparison meaningless.
X = np.full((T, H, W, C), 1.0, np.float16)
X[:, 0, 0, :] = np.nan
gc.collect()
base = hwm()

rng = np.random.default_rng(0)
t = torch.as_tensor(rng.integers(0, T, 4096))
y = torch.as_tensor(rng.integers(0, H, 4096))
x = torch.as_tensor(rng.integers(0, W, 4096))

if path == "lazy":
    Xt = LazyPixels(X)
    OBS = LazyPixels(X, obs=True)
    for _ in range(20):
        v = Xt[t, y, x]
        o = OBS[t, y, x]
else:
    # the eager form probe_sequence.py used verbatim: a full float copy AND a
    # full bool, both live at once (probe_kfold/dip_check filled in place, so
    # they paid the bool only).
    Xt = torch.from_numpy(np.nan_to_num(X, nan=0.0))
    OBS = torch.from_numpy(np.isfinite(X))
    for _ in range(20):
        v = Xt[t, y, x]
        o = OBS[t, y, x]
print(f"{{hwm() - base:.6f}}")
"""


def subprocess_peak(path, T, H, W, C):
    ml = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "ml")
    p = subprocess.run([sys.executable, "-c", MEASURE.format(ml=ml), path,
                        str(T), str(H), str(W), str(C)],
                       capture_output=True, text=True)
    if p.returncode:
        print(p.stderr, file=sys.stderr)
        raise SystemExit(f"the {path} measurement exited {p.returncode}")
    return float(p.stdout.strip().splitlines()[-1])


def main():
    X, lats, lons = fixture()
    T, H, W, C = X.shape

    # The eager pair, built WITHOUT filling X in place — the lazy pair reads
    # the same buffer and needs its NaNs. (In the scripts the in-place fill is
    # deleted for exactly this reason; here it would break the comparison.)
    eager_Xt = torch.from_numpy(np.nan_to_num(X, nan=0.0))
    eager_OBS = torch.from_numpy(np.isfinite(X))
    lazy_Xt = LazyPixels(X)
    lazy_OBS = LazyPixels(X, obs=True)

    ocean = np.isfinite(X[..., 0]).any(axis=0)
    ys, xs = np.where(ocean)
    # the real probes clip to RAPID_LON; here the section is the whole row, so
    # the toy still has enough pixels for `batch` to split it several ways.
    sec_y, sec_sel = section_of(lats, lons, ys, xs, 26.5, -100.0, 20.0)
    assert len(sec_sel) >= 10, f"fixture: section has {len(sec_sel)} pixels"
    ctx_all = np.stack([np.sin(2 * np.pi * np.arange(T) / 12),
                        np.cos(2 * np.pi * np.arange(T) / 12)], 1)

    # ---- 1: identical embeddings out of the REAL embed_everything ---------
    cases = [("patch=1", 1, None), ("patch=1 mask_chan", 1, [0, 3]),
             ("patch=3", 3, None)]
    for name, patch, mask_chan in cases:
        codec = tiny_codec(C, patch=patch)
        args = (ctx_all, lats, lons, ys[sec_sel], xs[sec_sel], codec.d_z)
        Za, _ = quiet(embed_everything, codec, eager_Xt, eager_OBS, *args,
                      batch=5, mask_chan=mask_chan)
        Zb, _ = quiet(embed_everything, codec, lazy_Xt, lazy_OBS, *args,
                      batch=5, mask_chan=mask_chan)
        assert Za.shape == (T, len(sec_sel), codec.d_z), Za.shape
        assert np.isfinite(Za).all(), f"{name}: the eager embedding is not finite"
        assert torch.equal(torch.from_numpy(np.asarray(Za)),
                           torch.from_numpy(np.asarray(Zb))), \
            f"{name}: lazy and eager embeddings differ"
        # and the fixture must be able to SHOW a difference: a mask that was
        # accidentally all-True (the deleted-nan_to_num trap) has to move this
        # number, or the check above is vacuous.
        if patch == 1 and mask_chan is None:
            allobs = LazyPixels(np.nan_to_num(X, nan=0.0), obs=True)
            Zc, _ = quiet(embed_everything, codec, lazy_Xt, allobs, *args,
                          batch=5)
            assert not np.array_equal(np.asarray(Za), np.asarray(Zc)), \
                ("an all-True observation mask produced the SAME embedding — "
                 "this fixture has no unobserved channels and check 1 cannot "
                 "see the nan_to_num-before-obs hazard")
    print(f"  1. embed_everything over {T} months x {len(sec_sel)} section "
          f"pixels: lazy == eager bit-for-bit at patch=1, patch=3 and under "
          f"mask_chan — and an all-True mask does move the answer")

    # ---- 2: the indexing forms the scripts use, one by one ----------------
    # probe_sequence.py's section mask: a tuple of slices and ints, reduced
    # with torch's own .any(axis=...) and handed to numpy.
    e_row = eager_OBS[:, sec_y, :, 0].any(axis=0).numpy()
    l_row = lazy_OBS[:, sec_y, :, 0].any(axis=0).numpy()
    assert np.array_equal(e_row, l_row), "the 26.5N ocean_row mask differs"
    assert e_row.any(), "fixture: the section row is empty"

    # probe_sequence.py's patch=1 encoder inputs: python ints + a numpy array,
    # then .to(device) on the result.
    sec_x = np.where(e_row & (lons >= -80.0) & (lons <= -13.0))[0]
    dev = torch.device("cpu")
    for t in (0, 1, T - 1):
        assert torch.equal(lazy_Xt[t, sec_y, sec_x].to(dev),
                           eager_Xt[t, sec_y, sec_x].to(dev)), t
        assert torch.equal(lazy_OBS[t, sec_y, sec_x].to(dev),
                           eager_OBS[t, sec_y, sec_x].to(dev)), t

    # probe_sequence.py's patch>1 branch: gather_px with torch index tensors.
    n = len(sec_x)
    tt = torch.full((n,), 3, dtype=torch.long)
    yy = torch.full((n,), sec_y, dtype=torch.long)
    xx = torch.as_tensor(sec_x)
    for patch in (1, 3):
        a_v, a_o = gather_px(eager_Xt, eager_OBS, tt, yy, xx, patch)
        b_v, b_o = gather_px(lazy_Xt, lazy_OBS, tt, yy, xx, patch)
        assert torch.equal(a_v, b_v) and torch.equal(a_o, b_o), \
            f"gather_px patch={patch} differs"

    # embed_everything's patch=1 line, with the bool multiply that promotes:
    #     v = X[t, ys[sl], xs[sl]] * (~mk)
    m = len(sec_sel)
    mk = torch.zeros(m, C, dtype=torch.bool)
    mk[:, [0, 3]] = True
    a_v = eager_Xt[5, ys[sec_sel], xs[sec_sel]] * (~mk)
    b_v = lazy_Xt[5, ys[sec_sel], xs[sec_sel]] * (~mk)
    assert a_v.dtype == b_v.dtype == torch.float16, (a_v.dtype, b_v.dtype)
    assert torch.equal(a_v, b_v), "the masked-value multiply differs"

    # dip_check.py / probe_kfold.py read .shape off the tensor they pass in
    # (embed_everything unpacks T,H,W,C; gather_px reads [1] and [2]).
    assert tuple(lazy_Xt.shape) == tuple(eager_Xt.shape) == (T, H, W, C)
    print("  2. every indexing form the three scripts use agrees with the "
          "eager tensors: OBS[:, sec_y, :, 0].any(axis=0).numpy(), "
          "[t, sec_y, sec_x].to(dev), gather_px at patch 1 and 3, the "
          "* (~mk) promote, and .shape")

    # ---- 3: the peak is MEASURED lower ------------------------------------
    # ~0.52 GiB of float16, big enough that a full bool (0.26 GiB) and a full
    # float copy (0.52 GiB) are unmistakable in VmHWM, small enough to run in
    # the sandbox — the earlier 1.2 GB version of the train fixture was itself
    # OOM-killed here.
    T2, H2, W2, C2 = 2000, 60, 60, 39
    nbytes = T2 * H2 * W2 * C2 * 2 / GIB
    full_bool = T2 * H2 * W2 * C2 / GIB
    cost_lazy = subprocess_peak("lazy", T2, H2, W2, C2)
    cost_eager = subprocess_peak("eager", T2, H2, W2, C2)

    assert cost_eager > (nbytes + full_bool) * 0.8, (
        f"the eager path only moved VmHWM by {cost_eager:.3f} GiB against a "
        f"{nbytes:.3f} GiB copy plus a {full_bool:.3f} GiB bool — this "
        f"measurement cannot tell the paths apart and is not a check")
    assert cost_lazy < full_bool * 0.25, (
        f"the lazy path cost {cost_lazy:.3f} GiB, a large fraction of the "
        f"{full_bool:.3f} GiB bool it exists to avoid — something is "
        f"materialising a full-size derived array")
    print(f"  3. VmHWM over a {nbytes:.2f} GiB tensor, fresh process per path: "
          f"lazy {cost_lazy:.3f} GiB vs eager {cost_eager:.3f} GiB "
          f"({cost_eager / max(cost_lazy, 1e-3):.0f}x), and the lazy peak "
          f"stays well under the {full_bool:.2f} GiB bool alone")

    print("\ntests/test_probe_lazy_pixels.py: all 3 checks passed")


if __name__ == "__main__":
    main()

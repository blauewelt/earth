#!/usr/bin/env python3
"""Pin the four POST-TRAINING PROBES on LazyPixels: same numbers, no copies.

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

ml/probe_head.py was the one script this conversion MISSED, and it is the one
that mattered most: at pentad/daily cadence the pooled read-outs are distrusted
and the head probe is primary (ml/CLAUDE.md §3). Run #392 (2026-08-18), the
first unpooled read-out at pentad cadence, had BOTH of its invocations —
`probe_head.py --K 1` and its matched `--raw --raw-patch` control — OOM-killed
~112 s after "codec on cuda", and because the step is deliberately best-effort
the job still went green. The only trace was the archive step's honest
`not present: probe_head.json, probe_head_raw3x3.json`.

Its arithmetic is worse than the other three, because `copy=False` did NOT save
the values the way the comment implied. On [3142, 281, 481, 39] float16:

    X                                33.1 GB   resident (np.load decompresses)
    np.isfinite(Xa)                  16.6 GB   resident
    np.nan_to_num(Xa, copy=False)    82.8 GB   TRANSIENT
                                    --------
                                    132.5 GB   peak

numpy never copies the values under `copy=False`, but its masked-copyto form is

    idx_nan = isnan(d); idx_posinf = isposinf(d); idx_neginf = isneginf(d)

and isposinf/isneginf each build isinf(d) and signbit(d) underneath, so FIVE
full-size bools are live at once — measured 5.00x one full bool (1.3082 GiB
against 0.2615 GiB on the fixture below), i.e. 82.8 GB at pentad and 414 GB at
daily. No box we can rent survives that, and no amount of `copy=False` helps.

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
  4. **ml/probe_head.py's own numbers do not move**, in all four of its
     read-out shapes — embedding at patch 1 and 3, `--raw`, and
     `--raw --raw-patch`. The last two never call `embed_everything` at all:
     they slice the pixel tensors directly and drive `gather_px` per timestep,
     so check 1 cannot cover them. `fold_fit` is replaced by a deterministic
     function of the tokens, which makes the comparison exact rather than
     statistical — what is being pinned is the pixel path, not the optimiser.
     Plus the same falsifier: an all-True mask must move all four.
  5. **ml/probe_head.py's own VmHWM is bounded**, running its real `main()` up
     to the first `embed_everything` call — the exact span run #392 died
     inside. The bound is derived from the fixture (the tensor plus at most one
     full-size mask), not hand-picked, and the eager leg has to clear the
     tensor plus TWO masks or the measurement is not a check.

    python3 tests/test_probe_lazy_pixels.py
"""
import contextlib
import io
import json
import os
import shutil
import subprocess
import sys
import tempfile

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


# ---------------------------------------------------------------------------
# Checks 4 and 5 run ml/probe_head.py ITSELF. It is the fourth script in this
# family and the last one converted (run #392, 2026-08-18), and it is also the
# only one whose read-out has three shapes — codec embedding, --raw, and
# --raw --raw-patch — each indexing the pixel tensors differently.
EAGER_PAIR = r'''
# The three lines the fix DELETED, reproduced exactly and in their original
# order: the mask taken from the NaNs FIRST, then the values filled IN PLACE
# (`np.nan_to_num(Xa, copy=False)`), then a zero-copy wrap. Keyed on the buffer
# so it does not matter which of the two probe_head.py now asks for first.
#
# eager_pair_reset() MUST be called before each run and this is not tidiness:
# id() is an ADDRESS, numpy hands the same address to the next same-shaped
# allocation, and a stale hit returns a mask for a dead array *and skips the
# in-place fill*, so the tensor keeps its NaNs and every embedding comes out
# NaN. That is exactly how this check first failed while the code was correct.
_seen = {}


def eager_pair_reset():
    _seen.clear()


def eager_pair(X, obs=False):
    m = _seen.get(id(X))
    if m is None:
        m = _seen[id(X)] = torch.from_numpy(np.isfinite(X))
        np.nan_to_num(X, nan=0.0, copy=False)
    return m if obs else torch.from_numpy(X)
'''

exec(EAGER_PAIR)                       # noqa: S102 — check 4 uses it in-process


def all_true_pair(X, obs=False):
    """The trap, as a stand-in: nan_to_num BEFORE the mask, so isfinite() sees
    an array with no NaNs left and the observation mask is all-True."""
    v = np.nan_to_num(X, nan=0.0)
    return torch.from_numpy(np.isfinite(v)) if obs else torch.from_numpy(v)


HEAD_FILE = {(): "probe_head.json",
             ("--raw",): "probe_head_raw.json",
             ("--raw", "--raw-patch"): "probe_head_raw3x3.json"}


def head_fixture(tmp, T=72, H=12, W=20, C=6, seed=5):
    """A tiny tensor probe_head.py can run END TO END, plus a patch=1 and a
    patch=3 codec. Small in T because check 4 is about the DATA path, not the
    optimiser; it still carries land, flicker and a real RAPID series."""
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(T, H, W, C)).astype(np.float16)
    X[:, rng.random((H, W)) < 0.25, :] = np.nan          # land
    X[rng.random(X.shape) < 0.05] = np.nan               # flicker
    months = np.array(["%04d-%02d" % (2004 + i // 12, i % 12 + 1)
                       for i in range(T)])
    rapid = np.stack([np.arange(T, dtype=np.float32),
                      (17 + rng.normal(size=T)).astype(np.float32)], 1)
    data = os.path.join(tmp, "head.npz")
    np.savez(data, X=X, months=months,
             lats=np.linspace(0.0, 70.0, H), lons=np.linspace(-100.0, 20.0, W),
             chan=np.array(["c%d" % c for c in range(C)]), rapid=rapid)
    for patch in (1, 3):
        write_ckpt(tmp, "p%d" % patch, C, patch)
    return data


def big_head_fixture(tmp, T=2000, H=60, W=60, C=39, seed=20260819):
    """The SAME shape check 3 measures on — 0.52 GiB of float16 — but written
    as a tensor `ml/probe_head.py` will actually open, with the checkpoint it
    reads first. Big enough that one full-size bool (0.26 GiB) is unmistakable
    in VmHWM, small enough for the sandbox.

    Filled in time-blocks: a single rng.normal(size=(T,H,W,C)) would build a
    float64 of 2.1 GiB and be OOM-killed here before the test began."""
    rng = np.random.default_rng(seed)
    X = np.empty((T, H, W, C), np.float16)
    for i in range(0, T, 100):
        X[i:i + 100] = rng.normal(
            size=(min(100, T - i), H, W, C)).astype(np.float16)
    X[:, rng.random((H, W)) < 0.30, :] = np.nan          # land
    X[:, 5, 5, 3] = np.nan                               # a flickering channel
    data = os.path.join(tmp, "big.npz")
    np.savez(data, X=X,
             months=np.array(["%04d-%02d" % (2000 + i // 120, (i % 120) // 10 + 1)
                              for i in range(T)]),
             lats=np.linspace(0.0, 70.0, H), lons=np.linspace(-100.0, 20.0, W),
             chan=np.array(["c%d" % c for c in range(C)]),
             rapid=np.zeros((0, 2), np.float32))
    del X
    write_ckpt(tmp, "big", C, patch=1)
    return data


def write_ckpt(tmp, run, C, patch, d_z=8):
    torch.manual_seed(3)
    m = tiny_codec(C, d_z=d_z, patch=patch)
    d = os.path.join(tmp, "runs", run)
    os.makedirs(d, exist_ok=True)
    torch.save({"args": {"holdout_years": "2006", "holdout_lon": "-45,-35",
                         "patch": patch, "d_model": 16, "n_layers": 1,
                         "n_heads": 2, "d_dec": 16, "dec_layers": 1},
                "chan": ["c%d" % c for c in range(C)], "d_z": d_z,
                "model": m.state_dict()}, os.path.join(d, "pixelmae.pt"))


def run_head(mod, tmp, data, run, extra, impl, calls):
    """One full ml/probe_head.py main() with `impl` standing in for LazyPixels.

    fold_fit is replaced by a DETERMINISTIC function of the tokens, so what
    comes back fingerprints the pixel/embedding path rather than 4,000
    optimiser steps — and so the comparison is exact rather than statistical.
    """
    calls.clear()
    eager_pair_reset()

    def stub(Xtr, ytr, Xte, in_dim, seed, **kw):
        p = (Xte.to(torch.float64).mean(dim=(1, 2)) * (1 + seed)).numpy()
        calls.append(np.array(p, dtype=np.float64))
        return p

    old = (mod.LazyPixels, mod.fold_fit, mod.HERE, sys.argv)
    mod.LazyPixels, mod.fold_fit, mod.HERE = impl, stub, tmp
    sys.argv = ["probe_head.py", "--run", run, "--data", data] + extra
    try:
        quiet(mod.main)
    finally:
        mod.LazyPixels, mod.fold_fit, mod.HERE, sys.argv = old
    out = json.load(open(os.path.join(tmp, "runs", run,
                                      HEAD_FILE[tuple(extra)])))
    return out, np.concatenate(calls)


# Check 5's measurement: ml/probe_head.py's own main(), stopped at the first
# embed_everything call — i.e. after load, anomaly transform, the pixel
# materialisation and `ocean`, which is the whole span run #392 died inside.
HEAD_MEASURE = r"""
import gc, sys, warnings
import numpy as np
import torch
sys.path.insert(0, {ml!r})
import probe_head

def hwm():
    for line in open("/proc/self/status"):
        if line.startswith("VmHWM:"):
            return int(line.split()[1]) * 1024 / (1024 ** 3)

path, data, tmp = sys.argv[1:4]
{eager}
if path == "eager":
    probe_head.LazyPixels = eager_pair

class _Stop(Exception):
    pass

def stop(*a, **k):
    raise _Stop()

probe_head.HERE = tmp
probe_head.embed_everything = stop
sys.argv = ["probe_head.py", "--run", "big", "--data", data]
gc.collect()
base = hwm()                      # after the imports, before any tensor work
reached = False
try:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        probe_head.main()
except _Stop:
    reached = True
print("REACHED=%d" % int(reached))
print("%.6f" % (hwm() - base))
"""


def head_peak(path, data, tmp):
    ml = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "ml")
    p = subprocess.run([sys.executable, "-c",
                        HEAD_MEASURE.format(ml=ml, eager=EAGER_PAIR),
                        path, data, tmp], capture_output=True, text=True)
    if p.returncode:
        print(p.stdout[-2000:], p.stderr[-2000:], file=sys.stderr)
        raise SystemExit(f"probe_head.py's {path} measurement exited "
                         f"{p.returncode}")
    assert "REACHED=1" in p.stdout, (
        f"the {path} run never reached embed_everything, so it never built "
        f"the pixel tensors and this measurement is of nothing:\n{p.stdout}")
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

    # probe_head.py's `ocean`: an ELLIPSIS index, which no other script uses
    # and which LazyPixels has to pass through to numpy untouched.
    e_oc = eager_OBS[..., 0].any(axis=0).numpy()
    l_oc = lazy_OBS[..., 0].any(axis=0).numpy()
    assert np.array_equal(e_oc, l_oc), "probe_head's ocean mask differs"
    assert e_oc.any() and not e_oc.all(), (
        "fixture: `ocean` is uniform, so it cannot show a mask difference")

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
    print("  2. every indexing form the four scripts use agrees with the "
          "eager tensors: OBS[:, sec_y, :, 0].any(axis=0).numpy(), "
          "OBS[..., 0].any(axis=0).numpy(), [t, sec_y, sec_x].to(dev), "
          "gather_px at patch 1 and 3, the * (~mk) promote, and .shape")

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

    # ---- 4: probe_head.py's OWN numbers do not move -----------------------
    # The other three scripts are covered by check 1's embed_everything
    # equality. probe_head.py needs its own, because it has two read-out modes
    # that never touch embed_everything at all: --raw slices the pixel tensors
    # directly (`Xt[:, sy, sx]`, `OBS[:, sy, sx]`) and --raw --raw-patch drives
    # gather_px per timestep. Those are the tokens the ONE read-out Chris
    # trusts at pentad cadence is built from (ml/CLAUDE.md §3), so "the fix did
    # not move the number" has to be asserted on probe_head's own output.
    #
    # fold_fit is replaced by a deterministic function of the tokens, so the
    # comparison is exact rather than statistical: what is being pinned is the
    # pixel/embedding path, not 4,000 optimiser steps.
    import probe_head                                        # noqa: E402
    htmp = tempfile.mkdtemp(prefix="head_")
    try:
        hdata = head_fixture(htmp)
        calls = []
        modes = [("p1", []), ("p3", []), ("p1", ["--raw"]),
                 ("p3", ["--raw", "--raw-patch"])]
        for run, extra in modes:
            lz, lz_raw = run_head(probe_head, htmp, hdata, run, extra,
                                  LazyPixels, calls)
            eg, eg_raw = run_head(probe_head, htmp, hdata, run, extra,
                                  eager_pair, calls)
            label = f"{lz['probe']} ({run})"
            assert lz_raw.shape == eg_raw.shape and lz_raw.size, \
                f"{label}: no fold predictions were produced"
            # named separately, because array_equal on NaN reports only
            # "max |diff| nan" and that hides which leg went bad
            for leg, arr in (("lazy", lz_raw), ("eager", eg_raw)):
                assert np.isfinite(arr).all(), (
                    f"{label}: the {leg} leg produced non-finite predictions "
                    f"({(~np.isfinite(arr)).sum()} of {arr.size}) — the pixel "
                    f"tensors still carry NaNs where the encoder reads them")
            assert np.array_equal(lz_raw, eg_raw), (
                f"{label}: LazyPixels and the eager pair give DIFFERENT "
                f"per-fold predictions — max |diff| "
                f"{np.abs(lz_raw - eg_raw).max():.3e}. The conversion moved "
                f"the head probe's numbers.")
            assert lz["pred"] == eg["pred"] and lz["r_kfold_deseas"] == \
                eg["r_kfold_deseas"], f"{label}: the written JSON differs"
            assert np.isfinite(lz["r_kfold_deseas"]), f"{label}: r is not finite"
            # ...and the fixture must be ABLE to show a difference. The trap is
            # nan_to_num BEFORE the mask, which makes OBS all-True; if that
            # does not move this number, the equality above is vacuous.
            at, at_raw = run_head(probe_head, htmp, hdata, run, extra,
                                  all_true_pair, calls)
            assert not (at_raw.shape == lz_raw.shape
                        and np.array_equal(at_raw, lz_raw)), (
                f"{label}: an all-True observation mask produced the SAME "
                f"predictions, so this fixture cannot see the "
                f"nan_to_num-before-obs hazard and check 4 proves nothing")
        print(f"  4. ml/probe_head.py end to end in all {len(modes)} read-out "
              f"shapes (embedding at patch 1 and 3, --raw, --raw --raw-patch): "
              f"lazy == eager on every per-fold prediction and on the written "
              f"JSON — and an all-True mask moves all four")

        # ---- 5: and probe_head.py's own peak is bounded -------------------
        # Run #392 (2026-08-18) was the first unpooled read-out at pentad
        # cadence. Both ml/probe_head.py invocations were OOM-killed ~112 s
        # after "codec on cuda", and because the step is deliberately
        # best-effort the job still reported success — the archive recorded
        # `not present: probe_head.json, probe_head_raw3x3.json` and that was
        # the only trace. This measures the span that died: load, anomaly
        # transform, the pixel materialisation and `ocean`, stopping at the
        # first embed_everything call. Fresh process per path, because VmHWM
        # is a high-water mark and never falls.
        hbig = big_head_fixture(htmp)
        T2, H2, W2, C2 = 2000, 60, 60, 39
        nb = T2 * H2 * W2 * C2 * 2 / GIB
        fb = T2 * H2 * W2 * C2 / GIB
        peak_lazy = head_peak("lazy", hbig, htmp)
        peak_eager = head_peak("eager", hbig, htmp)

        # The bounds are derived from the fixture, not chosen. The tensor's own
        # nb GiB is unavoidable — np.load decompresses it — so the question is
        # only what the script adds ON TOP. The eager form adds a resident bool
        # PLUS nan_to_num's temporaries: `copy=False` never copies the values,
        # but the masked-copyto form allocates isnan, isposinf and isneginf and
        # builds isinf/signbit under the last two, so five full-size bools are
        # live at once (measured 5.00x). That is why it must clear nb + 2*fb.
        assert peak_eager > nb + 2 * fb, (
            f"the eager path moved VmHWM by only {peak_eager:.3f} GiB over a "
            f"{nb:.3f} GiB tensor whose full-size bool is {fb:.3f} GiB — this "
            f"measurement cannot tell the paths apart and is not a check")
        assert peak_lazy < nb + fb, (
            f"ml/probe_head.py peaked at {peak_lazy:.3f} GiB over a {nb:.3f} "
            f"GiB tensor — more than the tensor plus ONE full-size mask "
            f"({nb + fb:.3f} GiB), so something is still materialising a "
            f"[T,H,W,C] derived array")
        print(f"  5. ml/probe_head.py's own VmHWM over a {nb:.2f} GiB tensor, "
              f"fresh process per path: lazy {peak_lazy:.3f} GiB vs eager "
              f"{peak_eager:.3f} GiB ({peak_eager / max(peak_lazy, 1e-3):.1f}x)"
              f" — the lazy peak stays under the tensor plus one "
              f"{fb:.2f} GiB mask, the eager one clears it by "
              f"{peak_eager - nb - fb:.2f} GiB")
    finally:
        shutil.rmtree(htmp, ignore_errors=True)

    print("\ntests/test_probe_lazy_pixels.py: all 5 checks passed")


if __name__ == "__main__":
    main()

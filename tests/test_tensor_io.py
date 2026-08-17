#!/usr/bin/env python3
"""Pin the sidecar tensor layout: same values, same interface, no residency.

The daily tensor is 165.6 GB and `np.load` on an `.npz` decompresses the whole
member into RAM, so family 5 cannot be opened at all by the code that opens
families 2, 3 and 4. `ml/tensor_io.py` stores `X` as a bare `.npy` beside the
metadata npz and memory-maps it.

The bar is that a reader changes ONE line and nothing else — so this file
asserts the interface, not just the bytes:

  1. **A single-file `.npz` still loads exactly as before.** `load_tensor` on
     a family-3-shaped file must be indistinguishable from `np.load`, because
     every existing tensor stays in that layout and no run may move.
  2. **The sidecar layout round-trips**: `save_tensor` then `load_tensor`
     returns the identical X and the identical metadata, and X comes back as a
     read-only memmap rather than an array.
  3. **The four things every reader uses all work** — `d["X"]`, `d["months"]`,
     `"rapid" in d`, `d.files` — on BOTH layouts, so the call sites cannot
     tell them apart.
  4. **Residency does not scale with the file.** MEASURED with VmHWM in a
     fresh process: `np.load` of an npz costs the array; `load_tensor` of the
     sidecar costs a small fraction of it. This is the whole reason the file
     exists, so it is measured rather than argued.
  5. **`writable_copy` is writable, identical, and cheap**, and it does not
     touch the original — which is the property that stops a second run from
     z-scoring an already-z-scored tensor.
  6. **`save_tensor` RENAMES a builder's memmap instead of copying it.** At
     166 GB a copy is the difference between a build that fits a 400 GB box
     and one that does not, so the no-copy path is asserted by checking the
     source file is gone.

    python3 tests/test_tensor_io.py
"""
import os
import subprocess
import sys
import tempfile

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ML = os.path.join(HERE, "..", "ml")
sys.path.insert(0, ML)

from tensor_io import (load_tensor, save_tensor, sidecar_path,   # noqa: E402
                       writable_copy)

GIB = 1024 ** 3

MEASURE = r"""
import gc, sys
sys.path.insert(0, {ml!r})
import numpy as np
from tensor_io import load_tensor

def hwm():
    for line in open("/proc/self/status"):
        if line.startswith("VmHWM:"):
            return int(line.split()[1]) * 1024 / (1024 ** 3)

path, how = sys.argv[1], sys.argv[2]
gc.collect()
base = hwm()
d = load_tensor(path) if how == "sidecar" else np.load(path)
X = d["X"]
s = float(np.asarray(X[0, 0, 0, :3]).sum())    # touch it, so nothing is lazy
print(f"{{hwm() - base:.6f}}")
"""


def meta_of(T, C, rng):
    return dict(months=np.array([f"{2004 + i // 12:04d}-{i % 12 + 1:02d}"
                                 for i in range(T)]),
                lats=np.linspace(0, 70, 9), lons=np.linspace(-100, 20, 11),
                chan=np.array([f"c{i}" for i in range(C)]),
                norm=np.stack([np.zeros(C), np.ones(C)], 1).astype(np.float32),
                rapid=np.array([[0.0, 17.0], [2.0, 16.4]]))


def main():
    tmp = tempfile.mkdtemp(prefix="tio_")
    rng = np.random.default_rng(20260817)
    T, H, W, C = 20, 9, 11, 7
    X = rng.normal(size=(T, H, W, C)).astype(np.float16)
    X[rng.random(X.shape) < 0.2] = np.nan
    meta = meta_of(T, C, rng)

    # ---- 1: the classic single-file layout is untouched --------------------
    classic = os.path.join(tmp, "family3_like.npz")
    np.savez_compressed(classic, X=X, **meta)
    a, b = np.load(classic), load_tensor(classic)
    assert np.array_equal(a["X"], b["X"], equal_nan=True), "classic X moved"
    assert list(a.files) == list(b.files), "classic file list moved"
    assert "rapid" in b and b["rapid"].shape == (2, 2)
    print("  1. a single-file .npz loads exactly as np.load did — no existing "
          "tensor moves")

    # ---- 2 & 3: the sidecar layout round-trips, same interface -------------
    side = os.path.join(tmp, "family5_like.npz")
    xp = save_tensor(side, X, **meta)
    assert os.path.exists(xp) and xp == sidecar_path(side)
    d = load_tensor(side)
    assert np.array_equal(d["X"], X, equal_nan=True), "sidecar X differs"
    assert isinstance(d["X"], np.memmap), "X came back as a plain array"
    assert not d["X"].flags.writeable, "the canonical tensor is writable"
    for k in ("months", "lats", "lons", "chan", "norm", "rapid"):
        assert np.array_equal(np.asarray(d[k]), np.asarray(meta[k])), k
    assert "X" in d and "rapid" in d and "nope" not in d
    assert set(d.files) == {"X"} | set(meta), d.files
    print("  2. the sidecar layout round-trips: identical X (read-only "
          "memmap) and identical metadata")
    print("  3. d[\"X\"], d[\"months\"], `in`, and .files all behave the same "
          "on both layouts")

    # ---- 4: residency MEASURED, in fresh processes ------------------------
    # Big enough that a decompressed array is unmistakable, small enough for
    # the sandbox: ~0.5 GiB of float16. Fresh processes because VmHWM is a
    # high-water mark and never falls.
    T2, H2, W2, C2 = 2000, 60, 60, 39
    big = np.full((T2, H2, W2, C2), 1.0, np.float16)
    nbytes = big.size * big.itemsize / GIB
    big_npz = os.path.join(tmp, "big_classic.npz")
    big_side = os.path.join(tmp, "big_side.npz")
    np.savez(big_npz, X=big, **meta_of(T2, C2, rng))
    save_tensor(big_side, big, **meta_of(T2, C2, rng))
    del big

    def peak(path, how):
        p = subprocess.run([sys.executable, "-c", MEASURE.format(ml=ML),
                            path, how], capture_output=True, text=True)
        if p.returncode:
            print(p.stderr, file=sys.stderr)
            raise SystemExit(f"the {how} measurement exited {p.returncode}")
        return float(p.stdout.strip().splitlines()[-1])

    cost_npz, cost_side = peak(big_npz, "classic"), peak(big_side, "sidecar")
    assert cost_npz > nbytes * 0.9, (
        f"np.load only cost {cost_npz:.3f} GiB for a {nbytes:.3f} GiB array — "
        f"the measurement is not seeing what it claims to")
    assert cost_side < nbytes * 0.15, (
        f"the sidecar cost {cost_side:.3f} GiB against {nbytes:.3f} GiB — it "
        f"is materialising the tensor, which is the one thing it must not do")
    print(f"  4. VmHWM opening a {nbytes:.2f} GiB tensor: np.load "
          f"{cost_npz:.3f} GiB vs sidecar {cost_side:.3f} GiB "
          f"({cost_npz / max(cost_side, 1e-3):.0f}x) — residency stops "
          f"scaling with the file")

    # ---- 5: writable_copy ------------------------------------------------
    scratch = os.path.join(tmp, "scratch.npy")
    src = load_tensor(side)["X"]
    cp = writable_copy(src, scratch, chunk=7, verbose=False)
    assert np.array_equal(cp, X, equal_nan=True), "the copy differs"
    assert cp.flags.writeable, "the copy is not writable"
    cp[0, 0, 0, 0] = 99.0
    cp.flush()
    assert float(load_tensor(side)["X"][0, 0, 0, 0]) != 99.0, \
        "writing the copy reached the canonical tensor"
    print("  5. writable_copy is writable, elementwise identical, and cannot "
          "reach the canonical tensor")

    # ---- 6: save_tensor renames a builder's memmap, never copies ----------
    build = os.path.join(tmp, "build_scratch.npy")
    m = np.lib.format.open_memmap(build, mode="w+", dtype=np.float16,
                                  shape=X.shape)
    m[:] = X
    out = os.path.join(tmp, "renamed.npz")
    save_tensor(out, m, **meta)
    assert not os.path.exists(build), \
        "the builder's memmap survived — save_tensor copied 166 GB instead " \
        "of renaming it"
    assert np.array_equal(load_tensor(out)["X"], X, equal_nan=True)
    print("  6. save_tensor renames a builder's memmap into place rather than "
          "copying it")

    print("\ntests/test_tensor_io.py: all 6 checks passed")


if __name__ == "__main__":
    main()

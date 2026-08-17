#!/usr/bin/env python3
"""A float16 tensor must train end to end. Family 4 is the first one.

Every tensor before family 4 was float32, so the trainer had never once seen
a half-precision batch. Family 4 chose float16 to fit 33.1 GB instead of 66.3
(`ml/build_family4.py`), and that single storage decision reaches the model
through two doors, both of which were shut:

  1. **The encoder.** Batches inherit the tensor's dtype — `LazyPixels` is
     faithful to what `torch.from_numpy(np.nan_to_num(X))` did — so the first
     `nn.Linear` gets Half against Float weights:
     `RuntimeError: mat1 and mat2 must have the same dtype, but got Half and
     Float`.
  2. **The loss target.** Even with the encoder fixed, `huber(pred, v)` keeps
     a Half target and the BACKWARD pass fails:
     `RuntimeError: Found dtype Half but expected Float` — a different error,
     one step later, which is exactly how a "fixed" run dies twice.

Neither was reachable before, and run #365 never got there: the host OOM
killer took it during the preamble. So the pentad re-dispatch would have
spent a data-cache seed and a 12-minute tensor build to find door 1, and
another to find door 2. This file finds both in about a minute on a 48x12x14
toy — ml/CLAUDE.md §4.8, "exercise the code path on a toy before spending the
expensive resource".

It runs the REAL `ml/train.py` as a subprocess rather than importing pieces of
it, because the failures were in the composition (reader dtype meets model
dtype meets loss dtype), not in any one function. Both `--patch 1` and
`--patch 3` are covered: the patch>1 path builds its value tensor through
`gather_px`, a different call site with its own `.to(dev)`.

What is asserted:

  · the run reaches the end and writes a checkpoint;
  · the loss is finite and MOVED — a run that quietly converts everything to
    zeros would also "complete", and that is the failure mode the float16
    z-score bug (2026-08-17) actually had;
  · float32 still behaves identically, so the widening did not become a
    float16-only code path.

    python3 tests/test_float16_training.py
"""
import json
import os
import subprocess
import sys
import tempfile

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
TRAIN = os.path.join(HERE, "..", "ml", "train.py")


def toy(path, dtype):
    """A tensor in family 4's shape and spirit: land, flicker, real truth."""
    rng = np.random.default_rng(20260817)
    T, H, W, C = 48, 12, 14, 39
    X = rng.normal(size=(T, H, W, C)).astype(dtype)
    X[:, rng.random((H, W)) < 0.30, :] = np.nan          # land
    X[rng.random(X.shape) < 0.10] = np.nan               # per-cell missing
    months = np.array([f"{2004 + i // 12:04d}-{i % 12 + 1:02d}"
                       for i in range(T)])
    np.savez(path, X=X, months=months,
             lats=np.linspace(0, 70, H), lons=np.linspace(-100, 20, W),
             chan=np.array([f"c{i}" for i in range(C)]),
             norm=np.stack([np.zeros(C), np.ones(C)], 1).astype(np.float32),
             rapid=np.array([[float(i), 15.0 + rng.normal()]
                             for i in range(0, T, 2)]))


def train(data, out, *extra):
    p = subprocess.run([sys.executable, "-u", TRAIN, "--data", data,
                        "--out", out, "--steps", "40", "--batch", "64",
                        "--d-model", "32", "--n-layers", "2", "--d-dec", "32",
                        "--anomaly", *extra],
                       capture_output=True, text=True)
    if p.returncode:
        tail = "\n".join((p.stdout + p.stderr).strip().splitlines()[-12:])
        raise SystemExit(f"train.py exited {p.returncode}:\n{tail}")
    return p.stdout


def losses(out):
    """(first, last) reconstruction loss from the run's own metrics file."""
    vals = []
    with open(os.path.join(out, "metrics.jsonl")) as fh:
        for line in fh:
            rec = json.loads(line)
            if "loss_rec" in rec:
                vals.append(float(rec["loss_rec"]))
    assert vals, "the run wrote no loss points at all"
    return vals[0], vals[-1]


def main():
    tmp = tempfile.mkdtemp(prefix="f16_")

    for patch in ("1", "3"):
        data = os.path.join(tmp, f"f16_{patch}.npz")
        out = os.path.join(tmp, f"run16_{patch}")
        toy(data, np.float16)
        train(data, out, "--patch", patch)
        assert os.path.exists(os.path.join(out, "pixelmae.pt")), \
            f"patch={patch}: no checkpoint written"
        first, last = losses(out)
        assert np.isfinite(first) and np.isfinite(last), \
            f"patch={patch}: loss went non-finite ({first} -> {last})"
        assert last != first, (
            f"patch={patch}: the loss never moved ({first}) — a run that "
            f"trains on all-zeros completes too")
        print(f"  {patch}. float16 trains end to end at --patch {patch}: "
              f"rec {first:.4f} -> {last:.4f}, checkpoint written")

    # float32 must be untouched by the widening — same fixture, same asserts.
    data = os.path.join(tmp, "f32.npz")
    out = os.path.join(tmp, "run32")
    toy(data, np.float32)
    train(data, out)
    assert os.path.exists(os.path.join(out, "pixelmae.pt")), "float32 broke"
    first, last = losses(out)
    assert np.isfinite(first) and np.isfinite(last) and last != first
    print(f"  3. float32 still trains identically well: "
          f"rec {first:.4f} -> {last:.4f}")

    print("\ntests/test_float16_training.py: all 3 checks passed")


if __name__ == "__main__":
    main()

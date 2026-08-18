#!/usr/bin/env python3
"""The trainer must produce IDENTICAL results from both tensor layouts.

`ml/tensor_io.py` was pinned in isolation (`tests/test_tensor_io.py`); this
file pins the WIRING — train.py actually reading a sidecar tensor — because
the isolated test cannot catch the two integration hazards:

  1. the anomaly transform writes into X, and a sidecar X arrives as a
     read-only memmap: without the scratch-copy branch the run dies on the
     first in-place write, and with a WRONG branch (r+ on the canonical file)
     it corrupts the tensor for every later run;
  2. equivalence: the same data through the classic npz and through the
     sidecar must give the SAME training trajectory — losses equal step for
     step, because the layout is storage, not semantics.

What is asserted (train.py draws batches from the unseeded global RNG, so
step-for-step loss equality across two processes is not testable without
changing the trainer; the properties below are the ones that decide
correctness):

  1. train.py --anomaly completes on a sidecar tensor THROUGH the
     scratch-copy branch — the branch must actually fire, or X arrived
     writable and the canonical tensor is exposed;
  2. the canonical sidecar X is BYTE-IDENTICAL after the run (sha256) — the
     in-place anomaly transform must never reach the stored file;
  3. the scratch copy is removed at process exit — at daily size an orphan
     is 166 GB per run.

    python3 tests/test_sidecar_reader_wiring.py
"""
import hashlib
import os
import subprocess
import sys
import tempfile

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "ml"))

from tensor_io import save_tensor, sidecar_path      # noqa: E402


def sha(p, buf=1 << 22):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(buf), b""):
            h.update(b)
    return h.hexdigest()


def main():
    tmp = tempfile.mkdtemp(prefix="wire_")
    rng = np.random.default_rng(20260818)
    T, H, W, C = 48, 12, 14, 39
    X = rng.normal(size=(T, H, W, C)).astype(np.float16)
    X[:, rng.random((H, W)) < 0.3, :] = np.nan
    meta = dict(months=np.array([f"{2004 + i // 12:04d}-{i % 12 + 1:02d}"
                                 for i in range(T)]),
                lats=np.linspace(0, 70, H), lons=np.linspace(-100, 20, W),
                chan=np.array([f"c{i}" for i in range(C)]),
                norm=np.stack([np.zeros(C), np.ones(C)], 1).astype(np.float32),
                rapid=np.array([[0.0, 17.0], [2.0, 16.4]]))

    side = os.path.join(tmp, "family5_like.npz")
    save_tensor(side, X, **meta)
    x_path = sidecar_path(side)
    before = sha(x_path)

    out = os.path.join(tmp, "run")
    p = subprocess.run([sys.executable, "-u",
                        os.path.join(HERE, "..", "ml", "train.py"),
                        "--data", side, "--out", out, "--steps", "30",
                        "--batch", "32", "--d-model", "32", "--n-layers", "2",
                        "--d-dec", "32", "--anomaly"],
                       capture_output=True, text=True)
    if p.returncode:
        print(p.stdout[-1500:], p.stderr[-1200:], file=sys.stderr)
        raise SystemExit(f"train.py exited {p.returncode} on the sidecar "
                         f"layout")
    assert "writable scratch copy" in p.stdout, (
        "the scratch branch never fired — either X arrived writable (the "
        "canonical tensor is exposed to in-place writes) or the sidecar was "
        "not memmapped at all")
    assert os.path.exists(os.path.join(out, "pixelmae.pt"))
    print("  1. train.py --anomaly completes on a sidecar tensor via the "
          "scratch-copy branch and writes its checkpoint")

    # ---- the property that matters: the canonical tensor is untouched -----
    assert sha(x_path) == before, (
        "the canonical sidecar X changed during training — the anomaly "
        "transform reached the stored tensor, and the NEXT run would z-score "
        "anomaly-space data")
    print("  2. the canonical sidecar X is byte-identical after the run")

    # ---- the scratch is cleaned up on exit --------------------------------
    scratch = side[:-4] + "_scratch.npy"
    assert not os.path.exists(scratch), (
        f"{scratch} survived the process — 166 GB of orphan per daily run")
    print("  3. the scratch copy is removed at process exit")

    print("\ntests/test_sidecar_reader_wiring.py: all 3 checks passed")


if __name__ == "__main__":
    main()

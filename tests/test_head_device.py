#!/usr/bin/env python3
"""ml/probe_head.py's read-out device: chosen UP FRONT, and CPU is a real path.

Run #397 (2026-08-18) ran `ml/probe_head.py` twice. Both invocations completed
the expensive part — full anomaly transform plus the entire 3142-month
embedding pass, ~13 minutes each — and then died in `fold_fit` on the first
`opt.zero_grad(); loss.backward(); opt.step()`:

    RuntimeError: Failed to find C compiler. Please specify via CC environment
    variable or set triton.knobs.build.impl

The chain: the tokens are moved to CUDA, `fold_fit` follows `Xtr.device`, so
SectionHead's cross-attention BACKWARD dispatches to a Triton-JIT kernel,
Triton tries to build its CUDA-utils C extension on first use, and that Vast
box had no C compiler and no `CC`. Both calls are wrapped
`|| echo "::warning::..."` in scripts/probes_run.sh, so the run went GREEN with
no head number — the second consecutive failure to produce the one read-out
that is primary at pentad cadence (ml/CLAUDE.md §3), after #392's OOM.

`torch.cuda.is_available()` was TRUE on that box, so it is the wrong question;
only running the failing code path answers it. And it must be run BEFORE the
embedding, because it depends on nothing else (ml/CLAUDE.md §0.3 / §5.16: "a
precondition that depends only on the inputs must be checked while the inputs
are all it has cost you").

A SECOND bug sat behind the first and had never fired, because the first one
masked it: `fold_fit` ended with `return net(Xte).numpy()`, which on a CUDA
tensor raises `TypeError: can't convert cuda:0 device type tensor to numpy`.
The GPU read-out path had therefore never executed end to end, despite the
comment above `fold_fit` asserting the move had been made on purpose.

What is pinned:

  1. **`fold_fit` runs end to end on CPU** over a synthetic fixture and
     returns a finite float array of one prediction per test month — the path
     the fallback now takes, and (with `.cpu()`) the same line the GPU path
     returns through. Plus (1b) the REAL `_selftest_step` — a whole
     SectionHead forward, backward and optimiser step, not a stub of it — and
     the unpatched `_usable_device("cuda")` answering cpu on a CUDA-less box.
  2. **`_usable_device("cpu")` returns CPU without probing anything.** The
     self-test is skipped, not merely passed: it is forced to raise, and CPU
     still comes back.
  3. **A failing self-test falls back instead of propagating** — the #397
     exception itself, raised out of `_selftest_step`, must produce
     `torch.device("cpu")` and a printed line naming the reason.
  4. **The self-test cannot move the numbers.** The global torch RNG is
     identical after a self-test that consumed randomness, whether it raised
     or returned, and `fold_fit` at a fixed seed is bit-identical either side
     of one — the fold numbers are a function of the data and the seed, never
     of the device the job landed on.
  5. **`--head-device` exists with choices auto|cpu|cuda, default auto**, and
     an unknown value is refused — the knob is what makes the device
     reversible from a dispatch without another code change.

    python3 tests/test_head_device.py
"""
import io
import contextlib
import os
import subprocess
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ML = os.path.join(HERE, "..", "ml")
sys.path.insert(0, ML)

import probe_head                                              # noqa: E402

# The real thing, verbatim, so the test fails with the failure it was written
# for rather than with a stand-in.
TRITON = ("Failed to find C compiler. Please specify via CC environment "
          "variable or set triton.knobs.build.impl.")


def fixture(n_tr=48, n_te=8, P=6, feat=5, seed=20260818):
    """A handful of months and a few section pixels — the shape probe_head
    builds, four orders of magnitude smaller. `in_dim` is feat + 2 because the
    real tokens carry (lon position, month offset) alongside the features."""
    rng = np.random.default_rng(seed)
    in_dim = feat + 2
    Xtr = torch.as_tensor(rng.normal(size=(n_tr, P, in_dim)).astype(np.float32))
    Xte = torch.as_tensor(rng.normal(size=(n_te, P, in_dim)).astype(np.float32))
    # a target the head can actually chase, so a broken step shows as a flat
    # or non-finite answer rather than as noise indistinguishable from noise
    ytr = (Xtr[:, :, 0].mean(1).numpy()
           + 0.1 * rng.normal(size=n_tr)).astype(np.float32)
    return Xtr, ytr, Xte, in_dim


def raiser(*a, **kw):
    raise RuntimeError(TRITON)


def rng_eater_ok(*a, **kw):
    """A self-test that PASSES and consumes global randomness on the way, the
    way the real one does (SectionHead's own initialisation plus randn)."""
    torch.randn(4)
    _ = probe_head.SectionHead(4, d=8)


def rng_eater_fail(*a, **kw):
    rng_eater_ok()
    raise RuntimeError(TRITON)


def main():
    # ---- 1: fold_fit end to end on CPU ------------------------------------
    Xtr, ytr, Xte, in_dim = fixture()
    p = probe_head.fold_fit(Xtr, ytr, Xte, in_dim, seed=0, steps=120)
    assert isinstance(p, np.ndarray), type(p)
    assert p.shape == (len(Xte),), (p.shape, len(Xte))
    assert p.dtype.kind == "f", p.dtype
    assert np.isfinite(p).all(), f"{(~np.isfinite(p)).sum()} of {p.size} are not finite"
    assert p.std() > 0, "every test month got the same prediction — the head " \
                        "never trained"
    print(f"  1. fold_fit on CPU over {len(Xtr)} train / {len(Xte)} test "
          f"months x {Xtr.shape[1]} section pixels: finite float array "
          f"{p.shape}, sd {p.std():.3f}")

    # ---- 1b: the REAL self-test body, not a stub --------------------------
    # It has to run the forward AND the backward, or it is checking
    # torch.cuda.is_available() with extra steps — which was TRUE on the box
    # that failed. On CPU it must simply complete; asked for a cuda that this
    # machine does not have, the unpatched function must still hand back CPU.
    probe_head._selftest_step(torch.device("cpu"))
    if not torch.cuda.is_available():
        with contextlib.redirect_stdout(io.StringIO()) as unpatched:
            d = probe_head._usable_device(torch.device("cuda"))
        assert d == torch.device("cpu"), (
            f"the real self-test returned {d} on a machine with no CUDA")
        print(f"  1b. the real _selftest_step completes a full SectionHead "
              f"forward+backward+step on CPU, and the unpatched "
              f"_usable_device('cuda') on this CUDA-less box answers cpu: "
              f"{unpatched.getvalue().strip()[:100]}...")
    else:
        print("  1b. the real _selftest_step completes a full SectionHead "
              "forward+backward+step on CPU (this box HAS cuda — its own "
              "self-test result is the one the next run will act on)")

    # ---- 2: --head-device cpu skips the probe entirely ---------------------
    old = probe_head._selftest_step
    probe_head._selftest_step = raiser
    try:
        d = probe_head._usable_device("cpu")
        assert d == torch.device("cpu"), d
        d = probe_head._usable_device(torch.device("cpu"))
        assert d == torch.device("cpu"), d

        # ---- 3: a failing self-test falls back, it does not propagate ------
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            d = probe_head._usable_device(torch.device("cuda"))
        assert d == torch.device("cpu"), (
            f"a self-test that raised {TRITON!r} returned {d} — the run would "
            f"still die in fold_fit, 13 minutes of embedding later")
        said = buf.getvalue()
        assert "cpu" in said.lower() and "RuntimeError" in said, said
        assert "C compiler" in said, (
            f"the fallback line does not carry the exception's own message, so "
            f"the log says a device changed and not why:\n{said}")
    finally:
        probe_head._selftest_step = old
    print("  2. _usable_device('cpu') returns CPU with the self-test forced to "
          "raise — it is skipped, not merely passed")
    print(f"  3. a cuda self-test raising the #397 error falls back to CPU and "
          f"says why: {said.strip()[:110]}...")

    # ---- 4: and it cannot move the numbers --------------------------------
    for name, stub in (("passed", rng_eater_ok), ("raised", rng_eater_fail)):
        probe_head._selftest_step = stub
        try:
            before = torch.get_rng_state()
            with contextlib.redirect_stdout(io.StringIO()):
                probe_head._usable_device(torch.device("cuda"))
            assert torch.equal(before, torch.get_rng_state()), (
                f"the self-test that {name} left the global RNG advanced — a "
                f"probe run on a box with a GPU would then produce different "
                f"fold numbers from the same data and the same seed")
            with contextlib.redirect_stdout(io.StringIO()):
                probe_head._usable_device(torch.device("cuda"))
            q = probe_head.fold_fit(Xtr, ytr, Xte, in_dim, seed=0, steps=120)
            assert np.array_equal(p, q), (
                f"fold_fit moved after a self-test that {name} — max |diff| "
                f"{np.abs(p - q).max():.3e}")
        finally:
            probe_head._selftest_step = old
    print("  4. the global RNG is identical after a self-test that consumed "
          "randomness (passing and raising), and fold_fit at seed 0 is "
          "bit-identical either side of one")

    # ---- 5: the flag itself ------------------------------------------------
    h = subprocess.run([sys.executable, os.path.join(ML, "probe_head.py"),
                        "--help"], capture_output=True, text=True)
    assert h.returncode == 0, h.stderr[-800:]
    flat = " ".join(h.stdout.split())
    assert "--head-device" in flat, flat[:400]
    assert "{auto,cpu,cuda}" in flat, (
        f"--head-device does not offer exactly auto|cpu|cuda:\n{flat[:600]}")
    bad = subprocess.run([sys.executable, os.path.join(ML, "probe_head.py"),
                          "--run", "x", "--data", "y",
                          "--head-device", "gpu"], capture_output=True,
                         text=True)
    assert bad.returncode != 0 and "invalid choice" in bad.stderr, (
        bad.returncode, bad.stderr[-400:])
    src = open(os.path.join(ML, "probe_head.py")).read()
    assert 'default="auto"' in src or "default='auto'" in src, \
        "--head-device has no default; a dispatch that omits it must mean auto"
    print("  5. --head-device {auto,cpu,cuda} is in --help, defaults to auto, "
          "and an unknown value is refused before anything is loaded")

    print("\ntests/test_head_device.py: all 5 checks passed")


if __name__ == "__main__":
    main()

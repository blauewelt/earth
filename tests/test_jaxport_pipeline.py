#!/usr/bin/env python3
"""Guards for the TPU input-pipeline knobs (2026-08-25, E-051's third node).

The knobs exist because the DEFAULT host pipeline at K=144 was measured at
~15 s/step of numpy RNG plus ~2 s of gather+cast against ~0.3 s of TPU step
(the first E-051 training node shipped four 10-minute cycles without reaching
step 600). Each knob claims to change WHERE work happens and never WHAT is
computed; these tests hold it to that.

  P1 — gather knobs are value-identical. `cast32=False` returns Z's own
  fp16 whose fp32 cast is BIT-IDENTICAL to the default path's output, and
  `workers=8` returns bit-identical output to the single-thread path,
  missing-neighbour zeros included.

  P2 — apply_znoise_jax keeps apply_znoise's contract. Dead slots (all-zero
  d_z groups) stay EXACT zeros; live slots move; the empirical sigma of the
  perturbation matches --input-znoise; sigma=0 is a pure cast; and an fp16
  input produces bit-identically the same output as its fp32 cast, because
  the cast happens before the add.

  P3 — the pipeline flags do not move a number. Two CLI runs on the same toy
  (znoise 0, same seed), one with every default and one with --gather-fp16
  --gather-workers 4 --prefetch 2, write the SAME loss curve — the batch
  sequence and the arithmetic are unchanged, only the wall clock. A third
  run with --noise-backend device --input-znoise 0.7 must complete with
  finite losses (its stream is jax.random's, so no curve equality is
  claimed — that is the documented semantics).
"""

import json
import os
import subprocess
import sys
import tempfile

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
ML = os.path.join(ROOT, "ml")
JTRAIN2 = os.path.join(ML, "jaxport", "train_stage2.py")
sys.path.insert(0, os.path.join(ML, "jaxport"))
sys.path.insert(0, HERE)

FAILURES = []


def fail(msg):
    FAILURES.append(msg)
    print(f"  FAIL {msg}")


def ok(msg):
    print(f"  {msg}")


def p1_gather():
    from train_stage2 import gather_stencil_np
    rng = np.random.default_rng(7)
    T, P, d_z, S, K, n = 40, 300, 4, 9, 6, 32
    Z = rng.standard_normal((T, P, d_z)).astype(np.float16)
    NBR = rng.integers(0, P, (P, S)).astype(np.int64)
    NBR[rng.random((P, S)) < 0.4] = -1
    base = rng.integers(0, T - K - 1, n)
    p = rng.integers(0, P, n)

    ref = gather_stencil_np(Z, base, p, NBR, K)
    if ref.dtype != np.float32:
        fail(f"P1: default path dtype {ref.dtype}, expected float32")
    g16 = gather_stencil_np(Z, base, p, NBR, K, cast32=False)
    if g16.dtype != np.float16:
        fail(f"P1: cast32=False dtype {g16.dtype}, expected float16 (Z's own)")
    if not np.array_equal(g16.astype(np.float32), ref):
        fail("P1: fp16 gather cast to fp32 is not bit-identical to the "
             "default path")
    gw = gather_stencil_np(Z, base, p, NBR, K, workers=8)
    if not np.array_equal(gw, ref):
        fail("P1: workers=8 output differs from single-thread output")
    gw16 = gather_stencil_np(Z, base, p, NBR, K, cast32=False, workers=8)
    if not np.array_equal(gw16, g16):
        fail("P1: workers=8 fp16 output differs from single-thread fp16")
    # NBR None — the legacy stencil-1 gather must be untouched by the knobs
    r0 = gather_stencil_np(Z, base, p, None, K)
    r0w = gather_stencil_np(Z, base, p, None, K, cast32=False, workers=8)
    if not np.array_equal(r0, r0w):
        fail("P1: NBR=None path changed under the knobs")
    if not FAILURES:
        ok("P1 gather knobs value-identical — fp16 cast bit-exact, "
           "8-thread bit-exact, legacy path untouched")


def p2_device_noise():
    import jax
    import jax.numpy as jnp
    from train_stage2 import apply_znoise_jax
    rng = np.random.default_rng(11)
    n, K, S, d_z = 8, 5, 7, 4
    z = rng.standard_normal((n, K, S * d_z)).astype(np.float32)
    dead = rng.random((n, K, S)) < 0.3
    z4 = z.reshape(n, K, S, d_z)
    z4[dead] = 0.0
    z = z4.reshape(n, K, S * d_z)
    key = jax.random.PRNGKey(3)
    sigma = 0.7

    out = np.asarray(apply_znoise_jax(jnp.asarray(z), key, sigma, d_z))
    o4 = out.reshape(n, K, S, d_z)
    if o4[dead].any():
        fail("P2: dead slots were perturbed — they must stay exact zeros")
    live = ~dead
    delta = (o4 - z4)[live]
    if not (delta != 0).any():
        fail("P2: live slots did not move under sigma=0.7")
    emp = float(delta.std())
    if not (0.8 * sigma < emp < 1.2 * sigma):
        fail(f"P2: empirical perturbation std {emp:.4f} outside 20% of "
             f"sigma {sigma}")
    z0 = np.asarray(apply_znoise_jax(jnp.asarray(z), key, 0.0, d_z))
    if not np.array_equal(z0, z):
        fail("P2: sigma=0 is not a pure cast")
    z16 = z.astype(np.float16)
    a32 = np.asarray(apply_znoise_jax(jnp.asarray(
        z16.astype(np.float32)), key, sigma, d_z))
    a16 = np.asarray(apply_znoise_jax(jnp.asarray(z16), key, sigma, d_z))
    if not np.array_equal(a16, a32):
        fail("P2: fp16 input does not match its fp32 cast bit-for-bit")
    if len(FAILURES) == 0:
        ok(f"P2 apply_znoise_jax — dead slots exact zeros, live std "
           f"{emp:.4f} ~ sigma {sigma}, sigma=0 pure cast, fp16 == fp32-cast")


def _run_cli(tmp, name, extra, steps=40):
    from test_jaxport_train_s2 import toy_tensor, toy_codec
    data = os.path.join(tmp, "toy.npz")
    ck = os.path.join(tmp, "pixelmae.pt")
    if not os.path.exists(data):
        C, _ = toy_tensor(data)
        toy_codec(ck, C)
    out = os.path.join(tmp, name)
    p = subprocess.run(
        [sys.executable, "-u", JTRAIN2, "--data", data, "--ckpt", ck,
         "--out", out, "--K", "6", "--steps", str(steps), "--batch", "32",
         "--lr", "1e-3", "--d-model", "32", "--layers", "2", "--seed", "0"]
        + extra,
        capture_output=True, text=True, cwd=ROOT)
    if p.returncode:
        tail = "\n".join((p.stdout + p.stderr).strip().splitlines()[-15:])
        fail(f"P3: run '{name}' exited {p.returncode}:\n{tail}")
        return None
    recs = [json.loads(ln) for ln in
            open(os.path.join(out, "metrics.jsonl"))]
    return [r for r in recs if "stage2_zmse" in r]


def p3_cli_equality(tmp):
    a = _run_cli(tmp, "a_default", [])
    b = _run_cli(tmp, "b_pipeline",
                 ["--gather-fp16", "--gather-workers", "4",
                  "--prefetch", "2"])
    if a is None or b is None:
        return
    la = [r["stage2_zmse"] for r in a]
    lb = [r["stage2_zmse"] for r in b]
    if la != lb:
        fail(f"P3: pipeline flags moved the loss curve — {la} vs {lb}. "
             f"They may only move the wall clock.")
    c = _run_cli(tmp, "c_device_noise",
                 ["--gather-fp16", "--gather-workers", "4", "--prefetch",
                  "2", "--input-znoise", "0.7", "--noise-backend", "device"])
    if c is None:
        return
    lc = [r["stage2_zmse"] for r in c]
    if not lc or not all(np.isfinite(v) for v in lc):
        fail(f"P3: device-noise run produced non-finite or empty losses: "
             f"{lc}")
    if lc == la:
        fail("P3: device-noise curve is identical to the noiseless one — "
             "the noise is not being applied")
    if len([f for f in FAILURES if f.startswith("P3")]) == 0:
        ok(f"P3 CLI — pipeline flags bit-stable ({len(la)} curve points "
           f"equal), device-noise run finite and distinct")


def main():
    print("tests/test_jaxport_pipeline.py — TPU input-pipeline guards P1-P3")
    p1_gather()
    p2_device_noise()
    with tempfile.TemporaryDirectory() as tmp:
        p3_cli_equality(tmp)
    if FAILURES:
        print(f"\n{len(FAILURES)} FAILURE(S)")
        return 1
    print("\ntests/test_jaxport_pipeline.py: all 3 guards passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())

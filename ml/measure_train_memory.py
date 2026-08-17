#!/usr/bin/env python3
"""Measure where ml/train.py's memory actually goes, step by step.

Written for E-038 step 2 ("a measurement first, then a fix, then a test that
pins peak memory") and, more immediately, to settle a misdiagnosis.

E-038 read the trainer's banner --

    X [T=3142 H=281 W=481 C=39] on cuda

-- as evidence that train.py "loads the ENTIRE tensor onto the device", and
concluded the #365 OOM was a 33.1 GB tensor against a 24 GB 4090.  The banner
is an f-string over the *device variable*:

    print(f"X [T={T} H={H} W={W} C={C}] on {dev} ...")

It reports which device training will use.  It says nothing about where X is.
Grep the trainer and no wholesale move exists: `Xt`/`OBS` are built with
`torch.from_numpy` (host memory, zero-copy) and every read is a per-batch
fancy index followed by `.to(dev)` -- `gather()` at train.py:246 and
`gather_px()` at model.py:132.  The device never holds more than one batch.

The kill was the HOST out-of-memory killer.  Exit 137 = 128 + SIGKILL; a CUDA
OOM raises torch.cuda.OutOfMemoryError and exits 1 with a traceback.

This script runs the trainer's real preamble allocation sequence at a
reduced T (every term is linear in T, so it extrapolates exactly) and prints
what each step costs.  Run it with --full on a box with enough RAM to confirm
the extrapolation directly.

    python3 ml/measure_train_memory.py                 # scaled, runs anywhere
    python3 ml/measure_train_memory.py --t 3142        # the real family-4 T
"""
from __future__ import annotations

import argparse
import gc
import os

import numpy as np

# Family-4 pentad tensor, from build_family4.py --dry-run.
FULL_T, H_DEF, W_DEF, C_DEF = 3142, 281, 481, 39
GIB = 1024 ** 3


def rss_gib() -> float:
    """Current resident set size."""
    with open("/proc/self/status") as fh:
        for line in fh:
            if line.startswith("VmRSS:"):
                return int(line.split()[1]) * 1024 / GIB
    return float("nan")


def peak_rss_gib() -> float:
    """High-water resident set size (VmHWM) -- survives frees, which is the
    number the OOM killer actually acted on."""
    with open("/proc/self/status") as fh:
        for line in fh:
            if line.startswith("VmHWM:"):
                return int(line.split()[1]) * 1024 / GIB
    return float("nan")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--t", type=int, default=200,
                   help="timesteps to allocate (default 200; 3142 is real)")
    p.add_argument("--height", type=int, default=H_DEF)
    p.add_argument("--width", type=int, default=W_DEF)
    p.add_argument("--chan", type=int, default=C_DEF)
    p.add_argument("--ocean-frac", type=float, default=0.641,
                   help="86,698 ocean cells / (281*481) on family 4")
    a = p.parse_args()

    T, H, W, C = a.t, a.height, a.width, a.chan
    scale = FULL_T / T
    cell = H * W * C

    print(f"shape [T={T} H={H} W={W} C={C}] float16 "
          f"-- {T * cell * 2 / GIB:.2f} GiB, scaling x{scale:.2f} to T={FULL_T}")
    print(f"host RSS at start: {rss_gib():.2f} GiB\n")

    rows: list[tuple[str, float, float]] = []

    def mark(label: str, prev: float) -> float:
        gc.collect()
        now = rss_gib()
        rows.append((label, now - prev, now))
        return now

    base = rss_gib()

    # --- train.py:147-150 -- np.load on an .npz DECOMPRESSES into RAM.
    # A real np.load of a compressed member allocates the whole array; we
    # allocate it directly here (same residency, no fixture on disk).
    rng = np.random.default_rng(0)
    X = np.empty((T, H, W, C), dtype=np.float16)
    land = rng.random((H, W)) > a.ocean_frac
    for t in range(T):                          # chunked: no f32 intermediate
        X[t] = rng.standard_normal((H, W, C), dtype=np.float32).astype(np.float16)
        X[t][land, :] = np.nan                  # land is unobserved everywhere
    prev = mark("np.load -> X  [T,H,W,C] f16", base)

    # --- train.py:194 -- obs_any = np.isfinite(X).sum(-1) >= 2
    # np.isfinite(X) materialises a full [T,H,W,C] bool; .sum(-1) a [T,H,W]
    # int64. Both are transient, but they are live simultaneously and this is
    # the first place the process can die.
    obs_any = np.isfinite(X).sum(-1) >= 2
    prev = mark("obs_any = isfinite(X).sum(-1)>=2   [TRANSIENT SPIKE]", prev)

    # --- train.py:195 -- np.where over the train pool
    tt, yy, xx = np.where(obs_any)
    prev = mark(f"np.where -> 3 x int64 ({len(tt) / 1e6:.1f}M px)", prev)

    # --- train.py:199 -- np.nan_to_num COPIES; from_numpy is zero-copy, so
    # the copy is numpy's, not torch's.
    Xt = np.nan_to_num(X, nan=0.0)
    prev = mark("np.nan_to_num(X) -> Xt   [FULL COPY]", prev)

    # --- train.py:200 -- np.isfinite(X) again, kept this time
    OBS = np.isfinite(X)
    prev = mark("np.isfinite(X) -> OBS    [full bool]", prev)

    hwm = peak_rss_gib()
    resident = rss_gib()

    w = max(len(r[0]) for r in rows)
    print(f"{'step':<{w}}  {'delta GiB':>10}  {'RSS GiB':>9}")
    print("-" * (w + 23))
    for label, delta, now in rows:
        print(f"{label:<{w}}  {delta:>10.2f}  {now:>9.2f}")
    print("-" * (w + 23))
    print(f"{'resident after preamble':<{w}}  {'':>10}  {resident:>9.2f}")
    print(f"{'peak (VmHWM)':<{w}}  {'':>10}  {hwm:>9.2f}")

    print(f"\nExtrapolated to the real tensor (T={FULL_T}):")
    print(f"  resident after preamble : {resident * scale:7.1f} GiB")
    print(f"  peak                    : {hwm * scale:7.1f} GiB")
    print(f"  daily arm (x5.01)       : {hwm * scale * 5.01:7.1f} GiB")

    print("\nEvery term above is HOST memory and none of it depends on "
          "--batch.\nThe device holds one batch: "
          f"{C * 2 * 2 / 1024:.1f} KiB per sample of values+mask, "
          f"{C * 2 * 2 * 4096 / 1024 ** 2:.1f} MiB at batch 4096.")

    # Keep references alive to the end so nothing is freed early.
    del Xt, OBS, obs_any, tt, yy, xx, X


if __name__ == "__main__":
    main()

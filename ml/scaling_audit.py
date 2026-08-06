#!/usr/bin/env python3
"""Measure — don't estimate — the sizes that the scaling discussion needs:
exact parameter counts per model, the observed-data inventory from the
tensor itself, epochs-seen at given training budgets, and the statistical
power of the RAPID probe. SCALING.md and METRICS.md quote this output;
re-run after changing channels, models, or splits.

Usage: python3 ml/scaling_audit.py [--data ml/cache/na_pixels.npz]
"""
import argparse
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model import PixelMAE
from temporal import TemporalTransformer

HERE = os.path.dirname(os.path.abspath(__file__))


def n_params(m):
    return sum(p.numel() for p in m.parameters())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=os.path.join(HERE, "cache", "na_pixels.npz"))
    a = ap.parse_args()
    d = np.load(a.data)
    X = d["X"]
    T, H, W, C = X.shape
    chan = [str(c) for c in d["chan"]]
    ocean = np.isfinite(X[..., 0]).any(axis=0)
    P = int(ocean.sum())

    print(f"## Data inventory  (tensor: T={T} months x {P} ocean pixels x C={C})\n")
    print("| channel | observed values | months present |")
    print("|---|---|---|")
    tot = 0
    for c in range(C):
        obs = int(np.isfinite(X[..., c]).sum())
        mon = int((np.isfinite(X[..., c]).any(axis=(1, 2))).sum())
        tot += obs
        print(f"| {chan[c]} | {obs:,} | {mon} |")
    print(f"| **total** | **{tot:,}** | |")

    # unique temporal transitions for stage 2 (pixel, t -> t+1 with data)
    obs_any = np.isfinite(X).any(-1)
    trans = int((obs_any[:-1] & obs_any[1:]).sum())
    rapid = d["rapid"]
    print(f"\nStage-2 transitions (pixel-month -> next month): {trans:,}")
    print(f"RAPID truth: {len(rapid)} monthly means "
          f"(train/test split at 3 held-out years: ~204/36)")

    print("\n## Parameter counts (exact)\n")
    print("| model | params |")
    print("|---|---|")
    codec = PixelMAE(n_chan=C, d_z=32)
    print(f"| PixelMAE codec (C={C}, d_z=32, d=128x4L) | {n_params(codec):,} |")
    for dz in (8, 16, 64):
        print(f"| PixelMAE codec (d_z={dz}) | {n_params(PixelMAE(n_chan=C, d_z=dz)):,} |")
    for tag, dm, ly in (("mini probe", 64, 2), ("stage-2 mid", 96, 3), ("stage-2 large", 192, 4)):
        m = TemporalTransformer(d_z=32, d_model=dm, n_layers=ly)
        print(f"| TemporalTransformer {tag} (d={dm}x{ly}L) | {n_params(m):,} |")
    print(f"| linear probe (ridge, K=1) | {32 + 1} |")
    print(f"| linear K-concat probe (K=24) | {24 * 32 + 1} |")

    print("\n## Budgets (samples seen vs unique data)\n")
    for steps, batch, label in ((8000, 512, "sandbox codec"), (30000, 512, "actions codec")):
        seen = steps * batch
        print(f"- {label}: {steps:,} steps x {batch} = {seen / 1e6:.1f}M pixel-month "
              f"samples seen = {seen / (T * P):.1f} epochs over the "
              f"{T * P / 1e6:.1f}M pixel-month corpus")
    for steps, batch, label in ((2000, 256, "stage-2 mid"), (6000, 256, "stage-2 runner")):
        seen = steps * batch
        print(f"- {label}: {steps:,} x {batch} windows = {seen / 1e6:.2f}M window samples "
              f"over {trans / 1e6:.2f}M unique transitions "
              f"({seen / trans:.1f} epochs)")

    print("\n## Probe power (why r is a coarse instrument)\n")
    rv = rapid[:, 1]
    # lag-1 autocorrelation of monthly RAPID means -> effective sample size
    r1 = float(np.corrcoef(rv[:-1], rv[1:])[0, 1])
    for n in (36, 240):
        neff = n * (1 - r1) / (1 + r1)
        se_iid = 1 / np.sqrt(n - 3)
        se_eff = 1 / np.sqrt(max(neff - 3, 1))
        print(f"- n={n} months: lag-1 autocorr {r1:.2f} -> n_eff~{neff:.0f}; "
              f"SE(r) {se_iid:.2f} iid, {se_eff:.2f} autocorr-adjusted "
              f"(95% CI half-width ~{1.96 * se_eff:.2f})")


if __name__ == "__main__":
    main()

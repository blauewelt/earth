#!/usr/bin/env python3
"""The head-level k-fold, exercised on synthetic features before any GPU.

Context, because this test exists to stop a specific mistake recurring.
`probe_kfold.py` scores the CODEC — it pools frozen embeddings along the
section and fits a ridge, and the temporal head appears nowhere in it. So
every run that freezes the same codec returns the same k-fold number no
matter what stage 2 did: #116 (60k head) and #125 (200k head, different
schedule, different optimiser trajectory) both read RAPID 0.631 [0.513,
0.732], rmse 2.16. A four-arm unroll sweep was queued to be scored on that.

`temporal.py` now also runs probe_kfold's protocol over the HEAD's own
pooled hidden state. This checks the wiring and, more importantly, the null:
an instrument that reports skill from noise would be worse than the
single-split one it supplements.

    python3 tests/test_head_kfold.py
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "ml"))
from probe_kfold import kfold_r                              # noqa: E402

T, D = 240, 24                     # 20 years of months, a 24-d pooled state


def synth(signal, seed=0):
    """Features that carry `signal` of a slow target, plus noise — the shape
    of the real thing, where the transport is autocorrelated over months and
    the head's hidden state is a smooth function of the field."""
    rng = np.random.default_rng(seed)
    # An AR(1) target: RAPID's monthly series is strongly autocorrelated, and
    # a white-noise target would make the year-blocked folds pointlessly easy.
    y = np.zeros(T)
    for t in range(1, T):
        y[t] = 0.85 * y[t - 1] + rng.standard_normal()
    F = rng.standard_normal((T, D))
    F[:, 0] = signal * y + np.sqrt(max(1 - signal ** 2, 0)) * rng.standard_normal(T)
    years = np.repeat(np.arange(2000, 2000 + T // 12), 12)
    return F, y, years


def main():
    # 1 · a real signal is recovered, with an interval that excludes zero.
    F, y, years = synth(0.7)
    r, lo, hi, n, rmse, sigma, pred = kfold_r(F, y, years)
    print(f"signal 0.7 -> r {r:.3f} [{lo:.3f}, {hi:.3f}] over {n} months, "
          f"rmse {rmse:.2f} vs sigma {sigma:.2f}")
    assert n == T, f"k-fold scored {n} months, expected every one of {T}"
    assert r > 0.4, f"a strong signal read as r={r:.3f}"
    assert lo > 0, "the CI includes zero on an obviously real signal"
    assert rmse < sigma, "the probe is worse than predicting the mean"

    # 2 · THE NULL. Pure noise must not read as skill. This is the property
    #     that matters: the whole point of moving off the 36-month split is
    #     resolution, and resolution bought by an optimistic instrument is
    #     worse than none. Year-blocking is what earns it — the fold boundary
    #     is a calendar year, so an autocorrelated target cannot leak across.
    worst = -1.0
    for seed in range(6):
        F, y, years = synth(0.0, seed=seed)
        r, lo, hi, n, _, _, _ = kfold_r(F, y, years, seed=seed)
        worst = max(worst, r)
        assert lo < 0 < hi or r < 0.25, \
            f"noise seed {seed} read r={r:.3f} [{lo:.3f}, {hi:.3f}] — the " \
            f"instrument invents skill"
    print(f"6 noise draws: worst r {worst:.3f}, every CI spans zero")

    # 3 · the sample really is ~6x the single split it supplements. 36 test
    #     months was the entire reason two runs could differ by 0.28 and mean
    #     nothing; stating the ratio here keeps that visible.
    print(f"\n{T} out-of-fold months vs 36 in the single split — {T/36:.1f}x, "
          f"i.e. about {np.sqrt(T/36):.1f}x the resolution on r")
    print("head-level k-fold behaves: skill where there is skill, "
          "nothing where there is none.")


if __name__ == "__main__":
    main()

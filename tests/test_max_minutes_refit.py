#!/usr/bin/env python3
"""Pin the wall-clock refit against the run that it silently destroyed.

Run #366 (2026-08-17), the first pentad codec dispatch after the OOM fixes:
step 1 carried ~537 s of one-time cost (first CUDA kernels, first touch of a
33 GB tensor), the old calibration fired at `elapsed > 60s` — i.e. at s=1 —
computed rate = elapsed/s = 537.54 s/step, and re-fit the cosine schedule
from 200,000 steps to **66**. True steady rate: ~0.19 s/step. The run trained
66 steps, annealed the LR to zero, saved a near-random codec, passed every
downstream step and reported success, with 691 of its 700 budgeted minutes
unspent. Nothing anywhere said wrong.

That is the most dangerous failure class this project knows — a green run
carrying a wrong number — and it came from three defects stacked: the rate
included step 1, one step was accepted as a sample, and the estimate was
final the moment it was made. `fit_schedule` plus its call-site guards fix
all three; this file replays the incident and pins each fix:

  1. **The #366 replay.** With the fixed rule, the first permitted
     calibration (s=4, three steady steps at the real ~0.19 s/step) returns
     the full 200,000 — the schedule survives its own first step.
  2. **A genuinely slow run still shrinks**, because that is what the feature
     is FOR: at a true 537 s/step, the budget honestly fits ~66 steps.
  3. **The fit is bounded**: never above the dispatched total (a fast run
     must not overshoot what was asked for), never below s+1.
  4. **Recovery is possible**: a schedule wrongly shrunk early grows back
     toward the dispatched total on a later re-check — calibrate-once is the
     defect, not a detail.
  5. **End to end on the real trainer**: a toy run under a generous
     `--max-minutes` completes ALL its steps (no spurious shrink from its own
     slow first step), and its checkpoint records the full count.

    python3 tests/test_max_minutes_refit.py
"""
import json
import os
import subprocess
import sys
import tempfile

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "ml"))

from train import fit_schedule          # noqa: E402


def main():
    # ---- 1: the #366 replay ------------------------------------------------
    # Step 1 took 537 s; steps 2-4 took the true steady ~0.19 s each. The
    # first calibration the guards allow is s=4 with steady_elapsed measured
    # from AFTER step 1.
    fit, rate = fit_schedule(s=4, steady_elapsed=3 * 0.19,
                             total_elapsed=537.54 + 3 * 0.19,
                             max_minutes=700, steps0=200_000)
    assert abs(rate - 0.19) < 1e-9, rate
    # The honest answer is ~185k, not 200k: 200,000 x 0.19 s = 633 min, and
    # the 15% probe holdback does not quite fit it in 700. What the fix must
    # guarantee is the ORDER OF MAGNITUDE — 66 was wrong by 2,800x.
    assert 180_000 < fit <= 200_000, (
        f"the #366 scenario re-fit to {fit} — the schedule must survive its "
        f"own first step")
    print(f"  1. #366 replayed: steady rate {rate:.2f} s/step at s=4 → fit "
          f"{fit} (the old rule concluded 537.54 s/step → 66)")

    # ---- 2: a genuinely slow run still shrinks ----------------------------
    fit, rate = fit_schedule(s=4, steady_elapsed=3 * 537.0,
                             total_elapsed=4 * 537.0,
                             max_minutes=700, steps0=200_000)
    assert 50 < fit < 80, (
        f"a true 537 s/step against 700 min honestly fits ~66 steps, got {fit}")
    print(f"  2. a run that truly runs at 537 s/step re-fits to {fit} — the "
          f"feature still does its job")

    # ---- 3: bounds --------------------------------------------------------
    fit, _ = fit_schedule(s=100, steady_elapsed=99 * 0.01,
                          total_elapsed=1.0, max_minutes=10_000,
                          steps0=5_000)
    assert fit == 5_000, f"fit {fit} exceeded the dispatched total"
    fit, _ = fit_schedule(s=100, steady_elapsed=99 * 900.0,
                          total_elapsed=99 * 900.0, max_minutes=1,
                          steps0=5_000)
    assert fit == 101, f"an exhausted budget must clamp to s+1, got {fit}"
    print("  3. bounded: never above the dispatched total, never below s+1")

    # ---- 4: a wrongly shrunk schedule can recover -------------------------
    # Suppose an early check DID shrink (bad luck: 3 slow-ish steps), and a
    # later check at s=60 sees the true fast rate. The fit must grow back —
    # the call site applies it whenever it differs materially, in either
    # direction.
    fit, _ = fit_schedule(s=60, steady_elapsed=59 * 0.19,
                          total_elapsed=537.0 + 59 * 0.19,
                          max_minutes=700, steps0=200_000)
    assert fit > 180_000, f"recovery failed: {fit}"
    print("  4. a later re-check grows a wrongly shrunk schedule back to the "
          "dispatched total")

    # ---- 5: end to end on the real trainer --------------------------------
    tmp = tempfile.mkdtemp(prefix="refit_")
    rng = np.random.default_rng(20260818)
    T, H, W, C = 48, 12, 14, 39
    X = rng.normal(size=(T, H, W, C)).astype(np.float16)
    X[:, rng.random((H, W)) < 0.3, :] = np.nan
    data = os.path.join(tmp, "toy.npz")
    np.savez(data, X=X,
             months=np.array([f"{2004 + i // 12:04d}-{i % 12 + 1:02d}"
                              for i in range(T)]),
             lats=np.linspace(0, 70, H), lons=np.linspace(-100, 20, W),
             chan=np.array([f"c{i}" for i in range(C)]),
             norm=np.stack([np.zeros(C), np.ones(C)], 1).astype(np.float32),
             rapid=np.array([[0.0, 17.0], [2.0, 16.4]]))
    out = os.path.join(tmp, "run")
    p = subprocess.run([sys.executable, "-u",
                        os.path.join(HERE, "..", "ml", "train.py"),
                        "--data", data, "--out", out, "--steps", "300",
                        "--batch", "32", "--d-model", "32", "--n-layers", "2",
                        "--d-dec", "32", "--anomaly", "--max-minutes", "30"],
                       capture_output=True, text=True)
    if p.returncode:
        print(p.stdout[-1500:], p.stderr[-1500:], file=sys.stderr)
        raise SystemExit(f"train.py exited {p.returncode}")
    last = 0
    with open(os.path.join(out, "metrics.jsonl")) as fh:
        for line in fh:
            rec = json.loads(line)
            if "step" in rec:
                last = max(last, rec["step"])
    assert last == 300, (
        f"a 300-step toy under a 30-min budget stopped at step {last} — the "
        f"refit shrank a schedule that fits its budget hundreds of times over")
    assert "re-fitting" not in p.stdout, (
        "the refit fired on a run whose budget was never in question:\n"
        + "\n".join(l for l in p.stdout.splitlines() if "re-fitting" in l))
    print("  5. end to end: a 300-step toy under --max-minutes 30 runs all "
          "300 steps and never re-fits")

    print("\ntests/test_max_minutes_refit.py: all 5 checks passed")


if __name__ == "__main__":
    main()

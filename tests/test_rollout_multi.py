#!/usr/bin/env python3
"""rollout.py's multi-head mode, end to end on a synthetic tensor.

The six-head E-010 rollout evaluation runs this exact path on a rented GPU;
ml/CLAUDE.md §4.8 says any hour of GPU on a path that has never executed is a
coin flip. This builds a 30-month toy ocean with a RAPID series, one tiny
codec and TWO tiny heads (a "U=1" and a "U=4"), runs the real script as a
subprocess, and asserts on what it WROTE — per-head curves, finite skill
numbers, AMOC bands — not on its exit code.

    python3 tests/test_rollout_multi.py
"""
import json
import os
import shutil
import subprocess
import sys
import tempfile

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ML = os.path.join(HERE, "..", "ml")
sys.path.insert(0, ML)
from model import PixelMAE                                    # noqa: E402
from temporal import TemporalTransformer                      # noqa: E402

T_M, H_G, W_G, C, DZ, K = 30, 8, 10, 5, 4, 6


def main():
    tmp = tempfile.mkdtemp()
    run_dir = os.path.join(ML, "runs", "toyroll")
    os.makedirs(run_dir, exist_ok=True)
    try:
        rng = np.random.default_rng(0)
        t = np.arange(T_M)[:, None, None, None]
        X = (np.sin(2 * np.pi * t / 12) + 0.4 * (t / T_M)
             + 0.3 * rng.standard_normal((T_M, H_G, W_G, C))).astype(np.float32)
        X[:, 0, 0, :] = np.nan                       # land, so OBS is exercised
        months = np.array([f"{1990 + i // 12}-{i % 12 + 1:02d}" for i in range(T_M)])
        # RAPID rows [month_index, Sv] for every month past the window
        ridx = np.arange(K, T_M)
        rapid = np.stack([ridx.astype(float),
                          2.79 * rng.standard_normal(len(ridx))], 1)
        npz = os.path.join(tmp, "toy.npz")
        np.savez(npz, X=X, months=months, rapid=rapid,
                 lats=np.linspace(20, 40, H_G).astype(np.float32),
                 lons=np.linspace(-60, -40, W_G).astype(np.float32))

        codec = PixelMAE(n_chan=C, d_model=16, n_heads=2, n_layers=2,
                         d_z=DZ, d_dec=16, patch=1)
        torch.save({"model": codec.state_dict(), "chan": [f"c{i}" for i in range(C)],
                    "d_z": DZ, "norm": None, "step": 0,
                    "args": {"patch": 1, "d_model": 16, "n_layers": 2,
                             "n_heads": 2, "d_dec": 16,
                             "holdout_years": "1992", "holdout_lon": "-45,-44"}},
                   os.path.join(run_dir, "pixelmae.pt"))

        heads = []
        # DELIBERATELY one head per convention: temporal.py builds its
        # position table at k_max=K, train_joint.py at max(K, 36), and
        # rollout.py guessed wrong in both directions on consecutive runs.
        # It must load BOTH, by reading the table's own shape.
        for (unroll, seed), kmax in (((1, 0), K), ((4, 1), max(K, 36))):
            torch.manual_seed(seed)
            hm = TemporalTransformer(d_z=DZ, d_model=8, n_heads=4,
                                     n_layers=1, k_max=kmax)
            hp = os.path.join(tmp, f"head_u{unroll}_s{seed}.pt")
            torch.save({"model": hm.state_dict(),
                        "args": {"K": K, "d_model": 8, "layers": 1,
                                 "unroll": unroll, "seed": seed}}, hp)
            heads.append(hp)

        r = subprocess.run(
            [sys.executable, "-u", os.path.join(ML, "rollout.py"),
             "--run", "toyroll", "--data", npz, "--horizon", "3",
             "--pixels", "30", "--temporal", *heads],
            capture_output=True, text=True, timeout=600)
        if r.returncode != 0:
            print(r.stdout[-3000:]); print(r.stderr[-3000:])
            raise SystemExit("rollout.py failed on the toy")

        out = json.load(open(os.path.join(run_dir, "rollout_eval.json")))
        assert set(out["heads"]) == {"u1_s0", "u4_s1"}, out["heads"].keys()
        for label, res in out["heads"].items():
            assert res["chan_skill"], f"{label}: no horizons scored"
            for row in res["chan_skill"]:
                for k in ("msss_clim", "msss_pers", "msss_damped", "acc"):
                    assert np.isfinite(row[k]), f"{label} h={row['h']} {k}={row[k]}"
            assert "horizon_auc" in res, f"{label}: no horizon AUC"
            print(f"{label}: {len(res['chan_skill'])} horizons, "
                  f"AUC {res['horizon_auc']:+.3f}, "
                  f"amoc bands {list(res['amoc_bands'])}")
        # the two heads have different random weights, so identical output
        # would mean the loop reused one model — the exact bug multi-head
        # evaluation could silently have
        a0 = out["heads"]["u1_s0"]["chan_skill"][0]["msss_clim"]
        b0 = out["heads"]["u4_s1"]["chan_skill"][0]["msss_clim"]
        assert a0 != b0, "two different heads produced identical skill"
        print("multi-head rollout eval runs end to end; heads are distinct.")

        # ---- AND with a patch=3 codec --------------------------------------
        # #149 died four minutes in because the static-identity pass fed the
        # patch=1 shape to a patch=3 encode — a fork this test did not reach
        # because its codec was patch=1. Every production codec is patch=3.
        codec3 = PixelMAE(n_chan=C, d_model=16, n_heads=2, n_layers=2,
                          d_z=DZ, d_dec=16, patch=3)
        torch.save({"model": codec3.state_dict(),
                    "chan": [f"c{i}" for i in range(C)],
                    "d_z": DZ, "norm": None, "step": 0,
                    "args": {"patch": 3, "d_model": 16, "n_layers": 2,
                             "n_heads": 2, "d_dec": 16,
                             "holdout_years": "1992", "holdout_lon": "-45,-44"}},
                   os.path.join(run_dir, "pixelmae.pt"))
        r = subprocess.run(
            [sys.executable, "-u", os.path.join(ML, "rollout.py"),
             "--run", "toyroll", "--data", npz, "--horizon", "2",
             "--pixels", "20", "--temporal", heads[0]],
            capture_output=True, text=True, timeout=600)
        if r.returncode != 0:
            print(r.stdout[-2000:]); print(r.stderr[-2000:])
            raise SystemExit("rollout.py failed with a PATCH=3 codec")
        out3 = json.load(open(os.path.join(run_dir, "rollout_eval.json")))
        assert out3["heads"]["u1_s0"]["chan_skill"], "patch=3: no horizons"
        print("patch=3 codec: rollout eval runs — the #149 crash is covered.")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
        shutil.rmtree(run_dir, ignore_errors=True)


if __name__ == "__main__":
    main()

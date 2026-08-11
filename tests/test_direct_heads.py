#!/usr/bin/env python3
"""temporal.py --direct, end to end on a synthetic tensor — twice.

E-014's arms train direct multi-horizon heads on a rented GPU through a path
that, without this file, has never executed (ml/CLAUDE.md §4.8: any hour of
GPU on a never-run path is a coin flip). This is ALSO the first end-to-end
toy of temporal.py itself, so it doubles as the guard on the refactors the
feature touched: batch_windows now returns a 7-tuple and the train-pool
guard covers the direct reach — a plain no-direct run exercises both with
D=(), which must reduce to the old behaviour exactly.

Run 1: --direct 2,3 → temporal.json must carry z_direct for both horizons
with finite numbers, and the checkpoint must load through rollout.py's own
construction recipe (k_max from pos.weight, direct from args) with
strict=True.

Run 2: no --direct → no z_direct key, no heads_direct in the state dict,
and the checkpoint loads with direct=().

    python3 tests/test_direct_heads.py
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

T_M, H_G, W_G, C, DZ, K = 36, 8, 10, 5, 4, 6


def run_trainer(npz, run, tmp, extra):
    env = dict(os.environ, CKPT_DIR_OVERRIDE=os.path.join(tmp, "ckpt"))
    r = subprocess.run(
        [sys.executable, "-u", os.path.join(ML, "temporal.py"),
         "--run", run, "--data", npz, "--K", str(K), "--steps", "30",
         "--batch", "16", "--d-model", "8", "--layers", "1",
         "--lr-warmup", "5", "--max-pixels", "25", *extra],
        capture_output=True, text=True, timeout=900, env=env)
    if r.returncode != 0:
        print(r.stdout[-4000:]); print(r.stderr[-4000:])
        raise SystemExit(f"temporal.py failed ({' '.join(extra) or 'plain'})")
    return r


def main():
    tmp = tempfile.mkdtemp()
    run = "toydirect"
    run_dir = os.path.join(ML, "runs", run)
    os.makedirs(run_dir, exist_ok=True)
    try:
        rng = np.random.default_rng(0)
        t = np.arange(T_M)[:, None, None, None]
        X = (np.sin(2 * np.pi * t / 12) + 0.4 * (t / T_M)
             + 0.3 * rng.standard_normal((T_M, H_G, W_G, C))).astype(np.float32)
        X[:, 0, 0, :] = np.nan                     # land, so OBS is exercised
        months = np.array([f"{1990 + i // 12}-{i % 12 + 1:02d}"
                           for i in range(T_M)])
        ridx = np.arange(K, T_M)
        rapid = np.stack([ridx.astype(float),
                          2.79 * rng.standard_normal(len(ridx))], 1)
        npz = os.path.join(tmp, "toy.npz")
        np.savez(npz, X=X, months=months, rapid=rapid,
                 chan=np.array([f"c{i}" for i in range(C)]),
                 lats=np.linspace(20, 40, H_G).astype(np.float32),
                 lons=np.linspace(-60, -40, W_G).astype(np.float32))

        codec = PixelMAE(n_chan=C, d_model=16, n_heads=2, n_layers=2,
                         d_z=DZ, d_dec=16, patch=1)
        torch.save({"model": codec.state_dict(),
                    "chan": [f"c{i}" for i in range(C)],
                    "d_z": DZ, "norm": None, "step": 0,
                    "args": {"patch": 1, "d_model": 16, "n_layers": 2,
                             "n_heads": 2, "d_dec": 16, "anomaly": True,
                             "holdout_years": "1992",
                             "holdout_lon": "-45,-44"}},
                   os.path.join(run_dir, "pixelmae.pt"))

        # ---- run 1: with direct heads ---------------------------------
        run_trainer(npz, run, tmp, ["--direct", "2,3"])
        tj = json.load(open(os.path.join(run_dir, "temporal.json")))
        zd = tj.get("z_direct", {})
        assert set(zd) == {"2", "3"}, f"z_direct horizons: {list(zd)}"
        for h_, v in zd.items():
            for k in ("mse_model", "mse_persistence"):
                assert np.isfinite(v[k]), f"h={h_} {k}={v[k]}"
        print(f"direct eval present: "
              f"{ {h: round(v['mse_model'], 3) for h, v in zd.items()} }")

        tk = torch.load(os.path.join(run_dir, "temporal.pt"),
                        map_location="cpu", weights_only=False)
        assert tk["args"]["direct"] == "2,3", tk["args"].get("direct")
        assert any(k.startswith("heads_direct.") for k in tk["model"]), \
            "checkpoint carries no direct-head weights"
        # rollout.py's construction recipe, verbatim: k_max from the table,
        # direct from args, strict load
        k_tbl = tk["model"]["pos.weight"].shape[0]
        dir_ = tuple(int(x) for x in
                     str(tk["args"].get("direct") or "").split(",")
                     if x.strip())
        m = TemporalTransformer(d_z=DZ, d_model=tk["args"]["d_model"],
                                n_heads=4, n_layers=tk["args"]["layers"],
                                k_max=k_tbl, direct=dir_)
        m.load_state_dict(tk["model"])
        print("direct checkpoint loads through the rollout recipe.")

        # ---- run 1b: SAMPLED unroll (E-016) ---------------------------
        # --unroll 4 with per-step depth sampling; must train, record the
        # probs in the checkpoint, and load through the rollout recipe.
        run_trainer(npz, run, tmp,
                    ["--unroll", "4",
                     "--unroll-probs", "0.5,0.25,0.125,0.125"])
        tks = torch.load(os.path.join(run_dir, "temporal.pt"),
                         map_location="cpu", weights_only=False)
        assert tks["args"]["unroll"] == 4
        assert tks["args"]["unroll_probs"] == "0.5,0.25,0.125,0.125"
        ms_ = TemporalTransformer(d_z=DZ, d_model=8, n_heads=4, n_layers=1,
                                  k_max=tks["model"]["pos.weight"].shape[0],
                                  direct=())
        ms_.load_state_dict(tks["model"])
        print("sampled-unroll run trains and its checkpoint loads.")

        # ---- run 2: plain — the refactor must be invisible ------------
        run_trainer(npz, run, tmp, [])
        tj2 = json.load(open(os.path.join(run_dir, "temporal.json")))
        assert "z_direct" not in tj2, "plain run grew a z_direct key"
        tk2 = torch.load(os.path.join(run_dir, "temporal.pt"),
                         map_location="cpu", weights_only=False)
        assert not any(k.startswith("heads_direct.") for k in tk2["model"])
        m2_ = TemporalTransformer(d_z=DZ, d_model=8, n_heads=4, n_layers=1,
                                  k_max=tk2["model"]["pos.weight"].shape[0],
                                  direct=())
        m2_.load_state_dict(tk2["model"])
        print("plain run unchanged: no direct keys, loads with direct=().")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
        shutil.rmtree(run_dir, ignore_errors=True)


if __name__ == "__main__":
    main()

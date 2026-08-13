#!/usr/bin/env python3
"""rollout_spatial.py end to end on a synthetic tensor — before any GPU.

ml/CLAUDE.md §4.8: any hour of GPU on a path that has never executed is a
coin flip. This builds a toy ocean as the PRODUCTION inputs look on a box —
an X memmap .npy, the small npz (months/lats/lons/rapid), a patch=3 codec
(every production codec is patch=3; #149 died on the patch fork rollout.py's
toy missed), and an f16 Z cache actually EMBEDDED through that codec so the
script's Z-verify guard is exercised for real. Two heads: stencil 1 with
k_max=K and stencil 9 with k_max=max(K, 36) (both position-table conventions,
the rollout.py lesson). Asserts on what the script WROTE, plus that the gate
refusal fires when no gate head is supplied.

    python3 tests/test_rollout_spatial.py
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
from recon_eval import stream_stats, build_slab               # noqa: E402
from temporal import TemporalTransformer, embed_everything    # noqa: E402

T_M, H_G, W_G, C, DZ, K = 44, 8, 10, 5, 4, 6


def main():
    tmp = tempfile.mkdtemp()
    try:
        rng = np.random.default_rng(0)
        t = np.arange(T_M)[:, None, None, None]
        X = (np.sin(2 * np.pi * t / 12) + 0.4 * (t / T_M)
             + 0.3 * rng.standard_normal((T_M, H_G, W_G, C))).astype(np.float32)
        X[:, 0, 0, :] = np.nan                    # land, so NBR misses fire
        xpath = os.path.join(tmp, "X.npy")
        np.save(xpath, X)
        months = np.array([f"{1990 + i // 12}-{i % 12 + 1:02d}"
                           for i in range(T_M)])
        lats = np.linspace(20, 40, H_G).astype(np.float32)
        lons = np.linspace(-60, -40, W_G).astype(np.float32)
        ridx = np.arange(K, T_M)                  # RAPID truth from month K
        rapid = np.stack([ridx.astype(float),
                          2.79 * rng.standard_normal(len(ridx))], 1)
        npz = os.path.join(tmp, "small.npz")
        np.savez(npz, months=months, lats=lats, lons=lons, rapid=rapid)

        codec = PixelMAE(n_chan=C, d_model=16, n_heads=2, n_layers=2,
                         d_z=DZ, d_dec=16, patch=3)
        ckpt = os.path.join(tmp, "pixelmae.pt")
        torch.save({"model": codec.state_dict(),
                    "chan": [f"c{i}" for i in range(C)],
                    "d_z": DZ, "norm": None, "step": 0,
                    "args": {"patch": 3, "d_model": 16, "n_layers": 2,
                             "n_heads": 2, "d_dec": 16,
                             "holdout_years": "1992",
                             "holdout_lon": "-45,-44"}}, ckpt)

        # ---- Z cache: embed the toy exactly as production embeds ---------
        moy = np.array([int(m[5:7]) - 1 for m in months])
        t_hold = np.array([m[:4] == "1992" for m in months])
        x_hold = (lons >= -45) & (lons < -44)
        Xm = np.load(xpath, mmap_mode="r")
        clim, dyn, mean_c, std_c = stream_stats(Xm, moy, t_hold, x_hold)
        full, obs = build_slab(Xm, list(range(H_G)), moy, clim, dyn,
                               mean_c, std_c)
        ocean = np.isfinite(X[..., 0]).any(0)
        ys, xs = np.where(ocean)
        ctx_all = np.stack([np.sin(2 * np.pi * moy / 12),
                            np.cos(2 * np.pi * moy / 12)], 1)
        codec.eval()
        Z, _ = embed_everything(codec, torch.from_numpy(
            np.nan_to_num(full, nan=0.0)), torch.from_numpy(obs),
            ctx_all, lats, lons, ys, xs, DZ, cache_path=None, batch=64)
        zpath = os.path.join(tmp, "Z.npy")
        np.save(zpath, np.asarray(Z).astype(np.float16))

        heads = []
        for stencil, kmax, seed in ((1, K, 0), (9, max(K, 36), 0)):
            torch.manual_seed(10 + stencil)
            hm = TemporalTransformer(d_z=DZ, d_model=8, n_heads=4,
                                     n_layers=1, k_max=kmax, stencil=stencil)
            hp = os.path.join(tmp, f"toy_s{stencil}_s{seed}.pt")
            torch.save({"model": hm.state_dict(),
                        "args": {"K": K, "d_model": 8, "layers": 1,
                                 "unroll": 1, "seed": seed,
                                 "stencil": stencil}}, hp)
            heads.append(hp)

        out = os.path.join(tmp, "rollout_spatial.json")
        base = [sys.executable, "-u", os.path.join(ML, "rollout_spatial.py"),
                "--x", xpath, "--npz-small", npz, "--z", zpath,
                "--ckpt", ckpt, "--out", out, "--horizon", "3",
                "--long-start", "1991-12", "--long-months", "16",
                "--future-months", "5", "--cache-dir", tmp]

        # the gate refusal must fire BEFORE any compute when no gate head
        # is given and --no-gate is not set
        r = subprocess.run(base + ["--heads", *heads],
                           capture_output=True, text=True, timeout=600)
        assert r.returncode != 0 and "gate" in (r.stdout + r.stderr).lower(), \
            f"expected the gate refusal, got rc={r.returncode}"
        print("gate refusal fires without e017_u1_s0 ✓")

        r = subprocess.run(base + ["--no-gate", "--heads", *heads],
                           capture_output=True, text=True, timeout=900)
        if r.returncode != 0:
            print(r.stdout[-4000:])
            print(r.stderr[-4000:])
            raise SystemExit("rollout_spatial.py failed on the toy")

        res = json.load(open(out))
        assert set(res["heads"]) == {"s1_s0", "s9_s0"}, res["heads"].keys()
        assert res["gate"].get("skipped"), "gate should be marked skipped"
        assert res["corridor_def"]["n_px"] > 0
        for label, e in res["heads"].items():
            for scope in ("gate", "corridor", "window"):
                rows = e[scope]["chan_skill"]
                assert rows, f"{label}/{scope}: no horizons scored"
                for row in rows:
                    for k in ("msss_clim", "msss_pers", "msss_damped", "acc",
                              "amp_ratio"):
                        assert np.isfinite(row[k]), \
                            f"{label}/{scope} h={row['h']} {k}={row[k]}"
                assert "horizon_auc" in e[scope]
            assert "h1-3" in e["amoc_bands"], f"{label}: {e['amoc_bands']}"
            assert np.isfinite(e["amoc_bands"]["h1-3"]["r"])
            assert len(e["long"]["sv_des"]) == 16
            assert all(np.isfinite(v) for v in e["long"]["sv_des"])
            # 1992 is the holdout year and the long roll crosses it whole
            assert e["long"]["n_heldout"] == 12, e["long"]
            assert len(e["future"]["sv_des"]) == 5
            print(f"{label}: window AUC {e['window']['horizon_auc']:+.3f}, "
                  f"amoc h1-3 r {e['amoc_bands']['h1-3']['r']:+.3f}, "
                  f"long n_heldout {e['long']['n_heldout']}")
        # different heads must produce different numbers (the reused-model
        # bug). A single rounded scalar can collide honestly, so compare the
        # WHOLE scored record of each head.
        a_all = json.dumps({k: res["heads"]["s1_s0"][k]
                            for k in ("gate", "corridor", "window",
                                      "amoc_bands", "long")})
        b_all = json.dumps({k: res["heads"]["s9_s0"][k]
                            for k in ("gate", "corridor", "window",
                                      "amoc_bands", "long")})
        assert a_all != b_all, "stencil 1 and 9 produced identical records"
        print("rollout_spatial toy: end to end ✓")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    main()

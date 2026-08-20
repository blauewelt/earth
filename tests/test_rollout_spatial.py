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
import datetime as dt
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
EPOCH = dt.date(1982, 1, 1)
# a 2-year PENTAD axis, 73 bins a year — enough for one holdout year with the
# staggered starts the real protocol asks for, and small enough to embed on a
# laptop. 1990-01-04 is the first bin START inside 1990.
PENTAD_DAYS = 5
PENTAD_B0 = (dt.date(1990, 1, 1) - EPOCH).days // PENTAD_DAYS + 1
T_P = 146


def pentad_labels(b0=PENTAD_B0, n=T_P, days=PENTAD_DAYS):
    """(bin indices, `YYYY-MM` labels) exactly as ml/build_family4.py emits
    them: one label per bin, from the bin's START date — so six consecutive
    rows share a label, which is the whole defect this fixture exists for."""
    bins = np.arange(b0, b0 + n, dtype=np.int64)
    lab = []
    for b in bins:
        d0 = EPOCH + dt.timedelta(days=int(b) * days)
        lab.append(f"{d0.year:04d}-{d0.month:02d}")
    return bins, np.array(lab)


def build_fixture(tmp, holdout_lon="-45,-44", cadence_days=0):
    """The toy production inputs, as a reusable dict of paths.

    Extracted from this test's main() so tests/test_roll_holdout_lon.py can
    score the SAME synthetic ocean this one rolls, instead of standing up a
    second toy that would drift from it. Returns the paths, plus the arrays a
    caller needs to check the script's own arithmetic against.

    `cadence_days` 0 builds the MONTHLY toy (families 2/3: a `months` array
    and nothing else). 5 builds a PENTAD toy shaped like family 4 — the same
    ocean and the same codec, but a `bin_index` axis, `pentad_days`, `cadence`
    and `epoch` beside a `months` array that repeats every label six times,
    and RAPID truth stored as (AXIS ROW, value) pairs, which is what
    build_family4's truth_pentad() writes.

    `holdout_lon` is the spec written into the checkpoint's `args`, and the
    SAME spec is used to build the Z cache's anomaly statistics — the two
    cannot be set independently, because rollout_spatial re-derives the
    statistics from the checkpoint and then verifies its Z against a live
    re-encode. Pass `"0,0"` for a no-longitude-holdout codec, which is the
    regime ml/recipes/*-nolonhold.json dispatch (E-043) and which gives the
    `_holdlon` scopes ZERO pixels.
    """
    rng = np.random.default_rng(0)
    T = T_M if not cadence_days else T_P
    spy = 12.0 if not cadence_days else 365.2425 / cadence_days
    t = np.arange(T)[:, None, None, None]
    X = (np.sin(2 * np.pi * t / spy) + 0.4 * (t / T)
         + 0.3 * rng.standard_normal((T, H_G, W_G, C))).astype(np.float32)
    X[:, 0, 0, :] = np.nan                    # land, so NBR misses fire
    xpath = os.path.join(tmp, "X.npy")
    np.save(xpath, X)
    extra = {}
    if cadence_days:
        bins, months = pentad_labels(n=T, days=cadence_days)
        extra = dict(bin_index=bins, pentad_days=np.array(cadence_days),
                     cadence=np.array("pentad"), epoch=np.array(str(EPOCH)))
    else:
        months = np.array([f"{1990 + i // 12}-{i % 12 + 1:02d}"
                           for i in range(T)])
    lats = np.linspace(20, 40, H_G).astype(np.float32)
    lons = np.linspace(-60, -40, W_G).astype(np.float32)
    # RAPID truth. Monthly: from row K, the historical fixture. Pentad: from
    # row 0 with one row in seven MISSING, because the pentad truth series is
    # neither complete nor month-aligned — and a fixture where every row
    # carries a label could not tell "attaches on the axis row" apart from
    # "attaches to everything". Every calendar month must appear among the
    # TRAIN rows, or the train-month climatology is undefined and the roll
    # (correctly) refuses.
    ridx = (np.arange(K, T) if not cadence_days
            else np.array([r for r in range(T) if r % 7 != 3]))
    rapid = np.stack([ridx.astype(float),
                      2.79 * rng.standard_normal(len(ridx))], 1)
    npz = os.path.join(tmp, "small.npz")
    np.savez(npz, months=months, lats=lats, lons=lons, rapid=rapid, **extra)

    codec = PixelMAE(n_chan=C, d_model=16, n_heads=2, n_layers=2,
                     d_z=DZ, d_dec=16, patch=3)
    ckpt = os.path.join(tmp, "pixelmae.pt")
    torch.save({"model": codec.state_dict(),
                "chan": [f"c{i}" for i in range(C)],
                "d_z": DZ, "norm": None, "step": 0,
                "args": {"patch": 3, "d_model": 16, "n_layers": 2,
                         "n_heads": 2, "d_dec": 16,
                         "holdout_years": "1992" if not cadence_days
                                          else "1991",
                         "holdout_lon": holdout_lon}}, ckpt)

    # ---- Z cache: embed the toy exactly as production embeds ---------
    moy = np.array([int(m[5:7]) - 1 for m in months])
    t_hold = np.array([m[:4] == ("1992" if not cadence_days else "1991")
                       for m in months])
    _lo, _hi = (float(v) for v in holdout_lon.split(","))
    x_hold = (lons >= _lo) & (lons < _hi)
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

    return {"x": xpath, "npz": npz, "z": zpath, "ckpt": ckpt,
            "heads": heads, "lons": lons, "x_hold": x_hold,
            "t_hold": t_hold, "ocean": ocean, "P": int(ocean.sum()),
            "months": months, "T": T, "cadence_days": cadence_days,
            "ridx": ridx, "K": K}


def main():
    tmp = tempfile.mkdtemp()
    try:
        f = build_fixture(tmp)
        xpath, npz, zpath = f["x"], f["npz"], f["z"]
        ckpt, heads = f["ckpt"], f["heads"]

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

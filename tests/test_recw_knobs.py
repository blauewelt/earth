#!/usr/bin/env python3
"""train.py's E-019b copy-reconstruction knobs, end to end on a toy tensor.

E-019b retrains the 41M codec through flags that, without this file, have
never executed (ml/CLAUDE.md §4.8). Three runs of the real trainer:

Run 1: --rec-w-visible 0.4 --upweight-chans 'c[23]' --upweight 3
       --dec-layers 3 → must train, checkpoint args must carry all four
       knobs, and codec_from_ckpt must rebuild the 3-hidden-layer decoder
       and load the state dict with strict=True.
Run 2: no knobs → checkpoint decoder must be shape-identical to a
       hand-built pre-E-019b PixelMAE (the defaults reproduce history).
Run 3: --upweight-chans 'rg_t9999' (matches nothing) → the trainer must
       REFUSE (exit != 0), because a typo'd regex silently doing nothing
       is a fake experiment.

    python3 tests/test_recw_knobs.py
"""
import os
import subprocess
import sys
import tempfile

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ML = os.path.join(HERE, "..", "ml")
sys.path.insert(0, ML)
from model import PixelMAE, codec_from_ckpt                   # noqa: E402

T_M, H_G, W_G, C = 30, 8, 10, 5


def run_trainer(npz, out, extra, expect_fail=False):
    r = subprocess.run(
        [sys.executable, "-u", os.path.join(ML, "train.py"),
         "--data", npz, "--out", out, "--steps", "25", "--batch", "16",
         "--d-z", "4", "--patch", "3", "--d-model", "16", "--n-layers", "1",
         "--n-heads", "2", "--d-dec", "24", "--anomaly", *extra],
        capture_output=True, text=True, timeout=900)
    if expect_fail:
        if r.returncode == 0:
            print(r.stdout[-3000:])
            raise SystemExit("trainer ACCEPTED a no-match --upweight-chans "
                             "regex — the refusal guard is gone")
        return r
    if r.returncode != 0:
        print(r.stdout[-3000:]); print(r.stderr[-3000:])
        raise SystemExit(f"train.py failed ({' '.join(extra) or 'plain'})")
    return r


def main():
    tmp = tempfile.mkdtemp()
    rng = np.random.default_rng(0)
    t = np.arange(T_M)[:, None, None, None]
    X = (np.sin(2 * np.pi * t / 12) + 0.4 * (t / T_M)
         + 0.3 * rng.standard_normal((T_M, H_G, W_G, C))).astype(np.float32)
    X[:, 0, 0, :] = np.nan
    months = np.array([f"{1990 + i // 12}-{i % 12 + 1:02d}"
                       for i in range(T_M)])
    ridx = np.arange(6, T_M)
    rapid = np.stack([ridx.astype(float),
                      2.79 * rng.standard_normal(len(ridx))], 1)
    npz = os.path.join(tmp, "toy.npz")
    np.savez(npz, X=X, months=months, rapid=rapid,
             chan=np.array([f"c{i}" for i in range(C)]),
             norm=np.ones((C, 2), np.float32),
             lats=np.linspace(20, 40, H_G).astype(np.float32),
             lons=np.linspace(-60, -40, W_G).astype(np.float32))

    # ---- run 1: all knobs on ---------------------------------------------
    out1 = os.path.join(tmp, "knobs")
    r = run_trainer(npz, out1, ["--rec-w-visible", "0.4",
                                "--upweight-chans", "c[23]",
                                "--upweight", "3", "--dec-layers", "3"])
    assert "upweight ×3.0: ['c2', 'c3']" in r.stdout, \
        "upweight resolution line missing/wrong:\n" + r.stdout[-1500:]
    ck = torch.load(os.path.join(out1, "pixelmae.pt"), map_location="cpu",
                    weights_only=False)
    a = ck["args"]
    assert a["rec_w_visible"] == 0.4 and a["upweight"] == 3.0
    assert a["upweight_chans"] == "c[23]" and a["dec_layers"] == 3
    codec = codec_from_ckpt(ck, C)
    codec.load_state_dict(ck["model"], strict=True)
    n_lin = sum(1 for m in codec.decoder if isinstance(m, torch.nn.Linear))
    assert n_lin == 4, f"dec_layers=3 should give 4 Linears, got {n_lin}"

    # ---- run 2: defaults reproduce the historical architecture -----------
    out2 = os.path.join(tmp, "plain")
    run_trainer(npz, out2, [])
    ck2 = torch.load(os.path.join(out2, "pixelmae.pt"), map_location="cpu",
                     weights_only=False)
    codec2 = codec_from_ckpt(ck2, C)
    codec2.load_state_dict(ck2["model"], strict=True)
    legacy = PixelMAE(n_chan=C, d_z=4, patch=3, d_model=16, n_layers=1,
                      n_heads=2, d_dec=24)
    got = {k: tuple(v.shape) for k, v in codec2.state_dict().items()}
    want = {k: tuple(v.shape) for k, v in legacy.state_dict().items()}
    assert got == want, "default-knob checkpoint diverges from the " \
        "pre-E-019b architecture"

    # ---- run 3: no-match regex must refuse -------------------------------
    run_trainer(npz, os.path.join(tmp, "refuse"),
                ["--upweight-chans", "rg_t9999"], expect_fail=True)

    print("OK: knobs persist+rebuild · defaults reproduce history · "
          "no-match regex refuses")


if __name__ == "__main__":
    main()

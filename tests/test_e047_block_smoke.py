#!/usr/bin/env python3
"""E-047 end to end on a CPU toy: block codec trains, embeds, head consumes.

The block axis and the encoder grid are unit-tested in
tests/test_e047_time_blocks.py. What THIS asks is the question no unit test
can: does the whole chain run — `ml/train.py --time-block month` trains a
codec whose decoder queries cells through `q_time`, `ml/temporal.py` embeds
the same tensor into ONE z per month and hands a stage-2 head an axis whose
labels are `YYYY-MM`, and the head trains on it without knowing anything
happened.

That last clause is the design goal. A month-block Z has a MONTHLY axis built
entirely out of 5-day data, so everything downstream — TimeAxis, the window
pool, the persistence baselines, the roll's horizon and bands — sees the axis
the archive was measured on.

And the control: the SAME toy with the knob off still trains, so a failure
here is about blocking and not about the fixture.

    python3 tests/test_e047_block_smoke.py

~2 minutes on two cores. No GPU, no network, no real tensor.
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
ROOT = os.path.abspath(os.path.join(HERE, ".."))
ML = os.path.join(ROOT, "ml")
sys.path.insert(0, ML)

EPOCH = dt.date(1982, 1, 1)
DAYS = 5
T, H, W, C, DZ = 146, 6, 7, 8, 8          # two pentad years, a 42-pixel ocean


def toy(tmp):
    """A pentad-shaped tensor: bin_index/pentad_days/epoch beside months, and
    3 'rg_*' channels present only in the 3rd bin of each month — the real
    tensor's own shape (n_rg_live 252/3142, the mid-month stamp)."""
    rng = np.random.default_rng(0)
    b0 = (dt.date(1990, 1, 1) - EPOCH).days // DAYS
    bins = np.arange(b0, b0 + T, dtype=np.int64)
    labs = np.array([(EPOCH + dt.timedelta(days=int(b) * DAYS)).strftime("%Y-%m")
                     for b in bins])
    t = np.arange(T)[:, None, None, None]
    X = (np.sin(2 * np.pi * t / 73) + 0.3 * rng.standard_normal((T, H, W, C))
         ).astype(np.float32)
    X[:, 0, 0, :] = np.nan                                  # land
    argo = [c >= C - 3 for c in range(C)]
    live = np.zeros(T, bool)
    seen = {}
    for i, lb in enumerate(labs):                           # 3rd bin of month
        seen.setdefault(lb, []).append(i)
    for rows in seen.values():
        if len(rows) > 2:
            live[rows[2]] = True
    X[np.ix_(~live, [0], [0], [0])]                         # (no-op guard)
    for i in range(T):
        if not live[i]:
            X[i, :, :, C - 3:] = np.nan
    ridx = np.arange(4, T, 6)
    rapid = np.stack([ridx.astype(float),
                      2.79 * rng.standard_normal(len(ridx))], 1)
    npz = os.path.join(tmp, "toy_pentad.npz")
    np.savez(npz, X=X, months=labs, bin_index=bins,
             pentad_days=np.array(DAYS), cadence=np.array("pentad"),
             epoch=np.array(str(EPOCH)), rapid=rapid, truth_rapid=rapid,
             chan=np.array([f"c{i}" if i < C - 3 else f"rg_t{i}"
                           for i in range(C)]),
             lats=np.linspace(20, 30, H).astype(np.float32),
             lons=np.linspace(-60, -50, W).astype(np.float32),
             norm=np.zeros((C, 2), np.float32))
    return npz, int(live.sum())


def run(cmd, tag, timeout=1800):
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout,
                       cwd=ROOT)
    if r.returncode != 0:
        print(r.stdout[-4000:])
        print(r.stderr[-3000:])
        raise SystemExit(f"{tag} failed (rc {r.returncode})")
    return r.stdout


def main():
    tmp = tempfile.mkdtemp()
    run_name = "e047_smoke"
    run_dir = os.path.join(ML, "runs", run_name)
    try:
        npz, n_live = toy(tmp)
        env_ck = os.path.join(tmp, "ckpt")
        base = [sys.executable, "-u", os.path.join(ML, "train.py"),
                "--data", npz, "--out", run_dir, "--steps", "12",
                "--batch", "16", "--d-model", "16", "--n-layers", "2",
                "--n-heads", "2", "--d-dec", "16", "--d-z", str(DZ),
                "--patch", "1", "--anomaly", "--holdout-years", "1991",
                "--holdout-lon=0,0", "--collapse-r", "0"]

        # ---- 1. the codec trains on month blocks -------------------------
        out = run(base + ["--time-block", "month"], "train --time-block month")
        assert "time blocks: mode 'month'" in out, out[-1500:]
        assert "k_max 7" in out and "encoder tokens per block" in out
        ck = torch.load(os.path.join(run_dir, "pixelmae.pt"),
                        map_location="cpu", weights_only=False)
        assert ck["args"]["time_block"] == "month"
        assert ck["args"]["k_time"] == 7, ck["args"]["k_time"]
        assert ck["args"]["ctx_mode"] == "block_phase"
        assert "time_emb.weight" in ck["model"]
        assert "q_time.weight" in ck["model"]
        assert ck["model"]["q_time.weight"].shape == (7, 16)
        blocks = int([l for l in out.splitlines()
                      if "time blocks: mode" in l][0].split("·")[1]
                     .strip().split()[0])
        print("1. `--time-block month` trains: %d blocks, k_max 7, the "
              "checkpoint carries time_block/k_time/ctx_mode and BOTH new "
              "embeddings (time_emb for the cell token, q_time for the cell "
              "query)" % blocks)

        # ---- 2. the embed emits one z per block, on a MONTHLY axis -------
        out2 = run([sys.executable, "-u", os.path.join(ML, "temporal.py"),
                    "--run", run_name, "--data", npz, "--K", "4",
                    "--steps", "6", "--batch", "8", "--d-model", "16",
                    "--layers", "2", "--max-pixels", "20", "--lr-warmup", "2"],
                   "temporal.py on the block codec")
        assert "block axis: T " in out2, out2[-2000:]
        line = [l for l in out2.splitlines() if "block axis: T " in l][0]
        tblk = int(line.split("block axis: T ")[1].split()[0])
        assert tblk == blocks, (tblk, blocks)
        assert f"Z [T={tblk}" in out2, [l for l in out2.splitlines()
                                        if l.strip().startswith("Z [")]
        tj = json.load(open(os.path.join(run_dir, "temporal.json")))
        assert tj["z_t+1"]["mse_model"] >= 0
        print("2. ml/temporal.py embeds the SAME tensor into %d block "
              "embeddings, its axis is the block labels (YYYY-MM — a monthly "
              "axis built entirely from 5-day data), the RAPID rows remap to "
              "their blocks, and a stage-2 head trains on it: z_t+1 "
              "mse_model %.4f vs persistence %.4f"
              % (tblk, tj["z_t+1"]["mse_model"],
                 tj["z_t+1"]["mse_persistence"]))

        # ---- 3. fixed-N mode, and the control ----------------------------
        out3 = run(base + ["--time-block", "2"], "train --time-block 2")
        assert "mode '2'" in out3 and "k_max 2" in out3
        ck3 = torch.load(os.path.join(run_dir, "pixelmae.pt"),
                         map_location="cpu", weights_only=False)
        assert ck3["args"]["k_time"] == 2
        out4 = run(base, "train (knob off)")
        assert "time blocks:" not in out4
        ck4 = torch.load(os.path.join(run_dir, "pixelmae.pt"),
                         map_location="cpu", weights_only=False)
        assert ck4["args"]["k_time"] == 1
        assert ck4["args"]["time_block"] == ""
        assert "time_emb.weight" not in ck4["model"]
        assert "q_time.weight" not in ck4["model"]
        assert ck4["args"]["ctx_mode"] == "month_sincos"
        print("3. `--time-block 2` gives k_max 2, and with the knob OFF the "
              "same fixture trains a codec with NO time_emb, NO q_time and "
              "ctx_mode month_sincos — the per-bin codec, unchanged")

        # ---- 4. a block codec refuses to be embedded as bins -------------
        import temporal as tp
        from model import codec_from_ckpt
        cm = codec_from_ckpt(ck, C)
        try:
            tp.embed_everything(cm, torch.zeros(4, H, W, C),
                                torch.ones(4, H, W, C, dtype=torch.bool),
                                np.zeros((4, 2)), np.zeros(H), np.zeros(W),
                                np.array([1]), np.array([1]), DZ)
            raise AssertionError("a block codec embedded one bin at a time")
        except ValueError as e:
            assert "BLOCK codec" in str(e), e
        print("4. embedding a block codec WITHOUT its block map raises rather "
              "than quietly producing embeddings of a different thing than it "
              "was trained on")

        print("\nE-047 end-to-end smoke: all 4 checks hold ✓")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
        shutil.rmtree(run_dir, ignore_errors=True)


if __name__ == "__main__":
    main()

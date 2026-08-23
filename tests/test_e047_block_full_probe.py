#!/usr/bin/env python3
"""The FULL in-training probe on a month-block codec, on a CPU toy.

tests/test_e047_block_smoke.py check 7 pins the LIGHT probe of a block run.
This pins the other one, and it exists because the light probe passing is not
evidence about the full probe: they share their inputs and almost nothing else.

WHAT BROKE (#450/#451/#453/#454, 2026-08-23). With `--time-block month` the
codec embeds one calendar MONTH per z, so `Z` has one row per BLOCK (516 on
family 4) while the tensor `X` it was built from still has one row per pentad
BIN (3,142). `ml/train.py` remaps every axis-keyed array it hands the probe —
`months`, `moy`, `t_hold`, and the RAPID truth through `BlockAxis.remap_rows`
— onto the block axis, so the light probe, which only ever indexes through
those, was correct. The full path then re-derived its own axis length from
`X.shape[0]`, which is the BIN axis, and walked `t_hold[t + 1]` off the end of
a 516-row array at t = 515:

    IndexError: index 516 is out of bounds for axis 0 with size 516

wrapped by train.py's run_probe guard into a {"probe_error"} record — non-fatal,
and it cost all four runs every diagnostic the full probe carries (z-space and
channel-space skill vs persistence, the temporal r).

WHAT THIS ASSERTS.

  1. `--time-block month --eval-every` runs its FULL probes to completion:
     no `probe_error` record, and the full metrics are finite.
  2. The channel decode is CELL-WISE — a block's z answers for k_time cells, so
     every real cell is decoded with its own `tpos` and scored against its own
     source bin, exactly as ml/rollout_spatial.py scores a rolled block. The
     effect is asserted, not the invocation: `chan_n` must exceed what one cell
     per block could have scored.
  3. The knob-off control still runs its full probe and emits NO block-only
     diagnostic — the per-bin path is untouched.

    python3 tests/test_e047_block_full_probe.py

~2 minutes on two cores. No GPU, no network, no real tensor.
"""
import datetime as dt
import glob
import json
import os
import shutil
import subprocess
import sys
import tempfile

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
ML = os.path.join(ROOT, "ml")
sys.path.insert(0, ML)

EPOCH = dt.date(1982, 1, 1)
DAYS = 5
# FIVE pentad years, not two. The full probe's mini transformer needs K=12
# context blocks BEFORE its first scored target, and it needs both a training
# and a held-out target after that — on a two-year toy every block at t+1 >= 12
# falls inside the holdout year and the probe has nothing to train on. 1990-1994
# with 1993 held out gives 60 blocks, 12 of them held out, in the middle.
YEARS = 5
T, H, W, C, DZ = 73 * YEARS, 6, 7, 8, 8
HOLD = "1993"


def toy(tmp):
    """The same pentad fixture tests/test_e047_block_smoke.py builds — bin
    index, pentad_days and epoch beside the month labels, and 3 'rg_*'
    channels present only in the 3rd bin of each month — five years long."""
    rng = np.random.default_rng(0)
    b0 = (dt.date(1990, 1, 1) - EPOCH).days // DAYS
    bins = np.arange(b0, b0 + T, dtype=np.int64)
    labs = np.array([(EPOCH + dt.timedelta(days=int(b) * DAYS)).strftime("%Y-%m")
                     for b in bins])
    t = np.arange(T)[:, None, None, None]
    X = (np.sin(2 * np.pi * t / 73) + 0.3 * rng.standard_normal((T, H, W, C))
         ).astype(np.float32)
    X[:, 0, 0, :] = np.nan                                  # land
    live = np.zeros(T, bool)
    seen = {}
    for i, lb in enumerate(labs):                           # 3rd bin of month
        seen.setdefault(lb, []).append(i)
    for rows in seen.values():
        if len(rows) > 2:
            live[rows[2]] = True
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
    return npz


def run(cmd, tag, timeout=2400):
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout,
                       cwd=ROOT)
    if r.returncode != 0:
        print(r.stdout[-4000:])
        print(r.stderr[-3000:])
        raise SystemExit(f"{tag} failed (rc {r.returncode})")
    return r.stdout


def records(run_dir):
    return [json.loads(l) for l in
            open(os.path.join(run_dir, "metrics.jsonl")) if l.strip()]


def main():
    tmp = tempfile.mkdtemp()
    run_name = "e047_full_probe"
    run_dir = os.path.join(ML, "runs", run_name)
    ctl_dir = os.path.join(ML, "runs", run_name + "_ctl")
    try:
        npz = toy(tmp)

        def cmd(out_dir):
            return [sys.executable, "-u", os.path.join(ML, "train.py"),
                    "--data", npz, "--out", out_dir, "--steps", "4",
                    "--batch", "16", "--d-model", "16", "--n-layers", "2",
                    "--n-heads", "2", "--d-dec", "16", "--d-z", str(DZ),
                    "--patch", "1", "--anomaly", "--holdout-years", HOLD,
                    "--holdout-lon=0,0", "--collapse-r", "0",
                    # --eval-every makes the step-0 probe the FULL one too, so
                    # a 4-step run buys two full probes (step 0 and step 4).
                    "--eval-every", "4"]

        # ---- 1. the full probe of a month-block run completes -------------
        out = run(cmd(run_dir) + ["--time-block", "month"],
                  "train --time-block month --eval-every")
        assert "time blocks: mode 'month'" in out, out[-1500:]
        n_blocks = int([l for l in out.splitlines()
                        if "time blocks: mode" in l][0].split("·")[1]
                       .strip().split()[0])
        # 73 pentads is 365 days, so five years of bins spill one bin past the
        # calendar and the last block is a one-bin January.
        assert 12 * YEARS <= n_blocks <= 12 * YEARS + 1, n_blocks
        recs = records(run_dir)
        bad = [r for r in recs if "probe_error" in r]
        assert not bad, ("the FULL probe of a block run failed", bad)
        full = [r for r in recs if "z_mse_model" in r]
        assert len(full) >= 2, ("no full probe record", [sorted(r) for r in recs])
        for r in full:
            for k in ("z_mse_model", "z_mse_persistence", "chan_mse_model",
                      "chan_mse_persistence", "temporal_r_raw",
                      "linear_r_raw"):
                assert np.isfinite(r[k]), (k, r)
        print("1. `--time-block month --eval-every` runs %d FULL probes to "
              "completion over %d blocks (of %d pentad bins) — no probe_error, "
              "z_mse_model %.4f vs persistence %.4f, chan %.4f vs %.4f"
              % (len(full), n_blocks, T, full[-1]["z_mse_model"],
                 full[-1]["z_mse_persistence"], full[-1]["chan_mse_model"],
                 full[-1]["chan_mse_persistence"]))

        # ---- 2. the channel decode is per CELL, against its own bin -------
        r0 = full[-1]
        assert r0["chan_cells"] == 7, r0["chan_cells"]
        assert r0["chan_n_one_cell"] > 0, r0
        # A per-bin probe scores ONE value per (sample, channel); a block probe
        # scores one per real CELL. The toy's month carries 6 or 7 bins, so the
        # denominator must be several times the one-cell one — and it cannot
        # reach k_max times it, because the 3 rg_* channels are observed in one
        # cell out of six by construction.
        assert r0["chan_n"] > 4 * r0["chan_n_one_cell"], r0
        assert r0["chan_n"] < 7 * r0["chan_n_one_cell"], r0
        print("2. the channel decode asks the block's z for every one of its "
              "%d cells and scores each against its OWN source bin: chan_n "
              "%d observed cell-values against %d for a one-cell-per-block "
              "probe (%.2fx)"
              % (r0["chan_cells"], r0["chan_n"], r0["chan_n_one_cell"],
                 r0["chan_n"] / r0["chan_n_one_cell"]))

        # ---- 3. the control: the per-bin path is untouched ----------------
        # Its OWN output directory: train.py APPENDS to metrics.jsonl, so a
        # second run into the block run's directory would read the block
        # records back and the control would be scoring the wrong file.
        out2 = run(cmd(ctl_dir), "train (knob off) --eval-every")
        assert "time blocks:" not in out2
        recs2 = records(ctl_dir)
        assert not [r for r in recs2 if "probe_error" in r], recs2
        full2 = [r for r in recs2 if "z_mse_model" in r]
        assert len(full2) >= 2, [sorted(r) for r in recs2]
        for r in full2:
            assert np.isfinite(r["chan_mse_model"]), r
            # The block-only diagnostics are BLOCK-ONLY: a per-bin record is
            # key-for-key what it has always been.
            for k in ("chan_cells", "chan_n", "chan_n_one_cell"):
                assert k not in r, (k, sorted(r))
        print("3. with the knob OFF the same fixture runs its full probes "
              "unchanged — finite chan/z/temporal metrics and NO block-only "
              "key in the record")

        print("\nE-047 full-probe on blocks: all 3 checks hold ✓")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
        shutil.rmtree(run_dir, ignore_errors=True)
        shutil.rmtree(ctl_dir, ignore_errors=True)
        for _f in glob.glob(os.path.join(ML, "cache", "Z_*_blk*.npy")):
            os.remove(_f)


if __name__ == "__main__":
    main()

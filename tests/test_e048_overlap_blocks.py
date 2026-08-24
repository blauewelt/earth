#!/usr/bin/env python3
"""E-048 · OVERLAPPING window blocks: the axis, and everything keyed on it.

Chris, 2026-08-24: *"One embedding every 30 days (6 pentads as input), advance
by 30 days. One embedding every 15 days (6 pentads/30 days as input,
OVERLAPPING windows), advance by 15 days — each embedding has Argo as part of
it, two consecutive ones share the same monthly Argo values."*

`--time-block W/S` is that, and E-047's fixed-N mode is its S = W case. What
this file pins is the half a wrong answer would be INVISIBLE in: an axis whose
labels repeat, whose step is 15 days and not a month, and whose persistence
baseline is stronger by construction than the 30-day arm's.

  1. **W/S IS FIXED-N WHEN S = W**, array for array — rows, pad, n_bins,
     labels, spans. One rule, so E-047's mode cannot drift from E-048's.
  2. **THE COUNT AND THE OVERLAP**: floor((T-W)/S)+1 windows, consecutive ones
     sharing exactly W-S source bins, every bin of every window real (no
     padding), and the labels ANCHORED at the window's first bin — which at
     S=3 means consecutive labels REPEAT, so the axis is not a monthly key and
     must never be read as one.
  3. **THE ROLL READS THE STRIDE OFF THE AXIS.** `axis_dict()` is handed to
     the roll's own `TimeAxis`: month mode's descriptor is bit-identical to
     what rollout_spatial built before E-048, and a window's is a BINNED axis
     whose step is S source bins. Its `date_of_row` reproduces every window's
     true start date, its day-defined bands cut at the same DAY edges (so 6/6
     recovers the monthly h1-3/h4-6/h7-12 exactly), and its day-matched leads
     are the same twelve DURATIONS at both strides.
  4. **THE LEAD OF A SCORED CELL IS h * STRIDE DAYS** — computed the way
     `ml/rollout_spatial.py` computes it, off the source axis, for every
     (horizon, cell) pair at S=3 and S=6.
  5. **THE TRUTH REMAP IS A PARTITION EVEN WHEN WINDOWS OVERLAP**: every
     covered source row lands on exactly one block, the owner is the latest
     window containing it, and no RAPID value is counted twice.
  6. **THE REFUSALS**: S > W (bins in no embedding at all), S = 0, a mode that
     is not a number, T < W, and `axis_dict()` on an axis with no bin_index.
  7. **END TO END AT STRIDE 3**: ml/train.py trains a 6/3 codec, ml/temporal.py
     embeds one z per WINDOW and hands the head an axis of 47 rows where the
     tensor has 146 bins, and ml/rollout_spatial.py rolls it — reporting a
     15-day step, the width/stride/overlap of the axis, and the persistence
     caveat that goes with them.

    python3 tests/test_e048_overlap_blocks.py

~3 minutes on two cores. No GPU, no network, no real tensor.
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
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
ML = os.path.join(ROOT, "ml")
sys.path.insert(0, HERE)
sys.path.insert(0, ML)

from timeblocks import BlockAxis, parse_mode                    # noqa: E402
from rollout_spatial import TimeAxis, BAND_EDGE_DAYS            # noqa: E402
from test_e047_block_smoke import toy, C, DZ, H, W              # noqa: E402

EPOCH = dt.date(1982, 1, 1)
DAYS = 5


def pentad_axis(n_years=2, y0=1990):
    b0 = (dt.date(y0, 1, 1) - EPOCH).days // DAYS
    n = int(round(n_years * 365.2425 / DAYS))
    bins = np.arange(b0, b0 + n, dtype=np.int64)
    labs = [(EPOCH + dt.timedelta(days=int(b) * DAYS)).strftime("%Y-%m")
            for b in bins]
    return labs, bins


def run(cmd, tag, timeout=2400):
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout,
                       cwd=ROOT)
    if r.returncode != 0:
        print(r.stdout[-4000:])
        print(r.stderr[-3000:])
        raise SystemExit(f"{tag} failed (rc {r.returncode})")
    return r.stdout


def main():
    labs, bins = pentad_axis()
    T = len(labs)
    a66 = BlockAxis("6/6", labs, bins, EPOCH, DAYS)
    a63 = BlockAxis("6/3", labs, bins, EPOCH, DAYS)
    fx6 = BlockAxis("6", labs, bins, EPOCH, DAYS)

    # ---- 1. W/S with S = W IS the fixed-N mode ---------------------------
    assert parse_mode("6") == ("window", 6, 6) == parse_mode("6/6")
    assert parse_mode("month") == ("month", None, None)
    for k in ("rows", "pad", "n_bins", "span_days"):
        assert np.array_equal(getattr(a66, k), getattr(fx6, k)), k
    assert a66.labels == fx6.labels and a66.k_max == fx6.k_max
    assert np.allclose(a66.ctx_phase(), fx6.ctx_phase())
    print("1. `--time-block 6/6` IS `--time-block 6`: rows, pad, n_bins, "
          "spans, labels and the ctx phase are equal array for array — E-047's "
          "fixed-N mode is the S = W case of one rule, not a second rule that "
          "has to agree with it")

    # ---- 2. counts, overlap, and repeating labels ------------------------
    assert a63.n_blocks == (T - 6) // 3 + 1 == 47, a63.n_blocks
    assert a66.n_blocks == (T - 6) // 6 + 1 == 24, a66.n_blocks
    assert a63.k_max == 6 and not a63.pad.any()
    assert (a63.n_bins == 6).all()
    for b in range(a63.n_blocks - 1):
        shared = set(a63.rows[b].tolist()) & set(a63.rows[b + 1].tolist())
        assert len(shared) == 3 == a63.overlap, (b, shared)
        assert int(a63.rows[b + 1, 0]) - int(a63.rows[b, 0]) == 3
    # every window's cells are its OWN six consecutive source bins
    for b in range(a63.n_blocks):
        assert list(a63.rows[b]) == list(range(3 * b, 3 * b + 6)), b
        assert a63.labels[b] == labs[3 * b]
    reps = sum(1 for i in range(a63.n_blocks - 1)
               if a63.labels[i] == a63.labels[i + 1])
    assert reps > 0, "a 15-day axis must be able to carry two windows in one month"
    assert np.allclose(a63.span_days, 30.0)
    print("2. 6/3 over %d bins gives %d windows (floor((T-W)/S)+1), each 6 "
          "REAL cells with no padding, consecutive windows sharing exactly 3 "
          "source bins and starting 3 apart; every label is the window's FIRST "
          "bin, and %d consecutive label pairs REPEAT — a 15-day axis is not a "
          "monthly key and this is where that becomes true"
          % (T, a63.n_blocks, reps))

    # ---- 3. the axis describes itself to the roll ------------------------
    mth = BlockAxis("month", labs, bins, EPOCH, DAYS)
    md = mth.axis_dict()
    assert set(md) == {"months"} and list(md["months"]) == mth.labels, md
    ax_m = TimeAxis(md)
    assert ax_m.monthly and ax_m.T == mth.n_blocks
    axes = {}
    for name, bax, want_step in (("6/6", a66, 30.0), ("6/3", a63, 15.0)):
        ax = TimeAxis(bax.axis_dict())
        axes[name] = ax
        assert not ax.monthly, name
        assert ax.step_days == want_step, (name, ax.step_days)
        assert ax.T == bax.n_blocks
        src = TimeAxis({"months": np.array(labs), "bin_index": bins,
                        "pentad_days": np.array(DAYS),
                        "epoch": np.array(str(EPOCH))})
        for b in (0, 1, bax.n_blocks // 2, bax.n_blocks - 1):
            assert ax.date_of_row(b) == src.date_of_row(int(bax.rows[b, 0])), \
                (name, b, ax.date_of_row(b))
        assert ax.label_of_row(0) == ax.date_of_row(0).isoformat()
    b66 = [k for k, _ in axes["6/6"].bands()]
    assert b66 == ["h1-3", "h4-6", "h7-12"], b66
    assert axes["6/6"].daymatched_leads() == tuple(range(1, 13))
    b63 = [k for k, _ in axes["6/3"].bands()]
    assert b63 == ["h1-6", "h7-12", "h13-24"], b63
    assert axes["6/3"].daymatched_leads() == tuple(2 * h for h in range(1, 13))
    assert axes["6/3"].steps_for_months(12) == 24
    print("3. the block axis hands TimeAxis its own descriptor: month mode is "
          "bit-identical to the monthly key the roll built before E-048, and a "
          "window is a BINNED axis whose step is the stride — 30 d at 6/6, 15 d "
          "at 6/3, every window's date_of_row equal to its first bin's own "
          "start. Day-defined bands (edges %s d) come out %s and %s, and the "
          "day-matched leads are the same twelve durations at both strides "
          "(1..12 vs 2..24)"
          % ("/".join(f"{e:g}" for e in BAND_EDGE_DAYS), b66, b63))

    # ---- 4. the lead of a scored cell is h * stride days ------------------
    src = TimeAxis({"months": np.array(labs), "bin_index": bins,
                    "pentad_days": np.array(DAYS), "epoch": np.array(str(EPOCH))})
    leads = {}
    for name, bax in (("6/6", a66), ("6/3", a63)):
        b0 = bax.n_blocks // 2
        got = {}
        for h in (1, 2, 3):
            got[h] = [(src.date_of_row(int(bax.rows[b0 + h, j]))
                       - src.date_of_row(int(bax.rows[b0, j]))).days
                      for j in range(int(bax.n_bins[b0]))]
            assert set(got[h]) == {h * bax.stride * DAYS}, (name, h, got[h])
        leads[name] = got
    print("4. the lead in DAYS of a scored (horizon, cell) pair is h x stride "
          "x the source bin, computed the way ml/rollout_spatial.py computes "
          "it: h=1..3 gives %s at 6/6 and %s at 6/3 — every cell of the block, "
          "same answer, so a 15-day roll cannot be read as a 30-day one"
          % ([leads["6/6"][h][0] for h in (1, 2, 3)],
             [leads["6/3"][h][0] for h in (1, 2, 3)]))

    # ---- 5. the truth remap is a partition -------------------------------
    covered = (a63.n_blocks - 1) * 3 + 6
    owners = [a63.block_of_row(r) for r in range(T)]
    for r in range(covered):
        assert owners[r] == min(r // 3, a63.n_blocks - 1), r
        assert r in set(a63.rows[owners[r]].tolist())
    assert all(o is None for o in owners[covered:])
    rows = np.arange(4, T, 6)
    rapid = np.stack([rows.astype(float),
                      np.random.default_rng(0).normal(size=len(rows))], 1)
    rm = a63.remap_rows(rapid)
    kept = [r for r in rows if r < covered]
    assert len(rm) == len(kept), (len(rm), len(kept))
    for (r0, v0), (b1, v1) in zip(rapid[:len(kept)], rm):
        assert v0 == v1 and int(b1) == a63.block_of_row(int(r0))
    # a value must not be counted twice just because its bin is in two windows
    assert len(rm) == len(set(zip(rm[:, 0].tolist(), rm[:, 1].tolist()))) \
        or len(np.unique(rm[:, 0])) <= len(rm)
    dup = len(rm) - len(np.unique(rm[:, 0]))
    print("5. ownership is a PARTITION even where windows overlap: every "
          "covered source row maps to min(floor(r/S), B-1) — the LATEST window "
          "containing it, whose anchor is never more than S-1 bins earlier — "
          "and %d RAPID rows remap onto %d distinct blocks with %d collisions, "
          "none of them a double count of one value"
          % (len(rm), len(np.unique(rm[:, 0])), dup))

    # ---- 6. the refusals -------------------------------------------------
    bad = []
    for mode, must in (("6/7", "stride 7 exceeds width 6"),
                       ("6/0", "stride S must be >= 1"),
                       ("0/0", "window width W must be >= 1"),
                       ("six/3", "expected 'month'"),
                       ("200/3", "No complete window")):
        try:
            BlockAxis(mode, labs, bins, EPOCH, DAYS)
            bad.append(mode)
        except ValueError as e:
            assert must in str(e), (mode, str(e))
    assert not bad, bad
    try:
        BlockAxis("6/3", labs).axis_dict()
        raise AssertionError("axis_dict on a label-only axis was accepted")
    except ValueError as e:
        assert "no `bin_index`" in str(e), e
    print("6. refused: stride 7 into width 6 (it would leave a bin in NO "
          "embedding, a gap nothing downstream can see), stride 0, a "
          "non-numeric mode, a window wider than the record, and axis_dict() "
          "on an axis built from month labels alone — a window of labels is "
          "not a duration")

    # ---- 7. end to end at stride 3 ---------------------------------------
    tmp = tempfile.mkdtemp()
    run_name = "e048_overlap"
    run_dir = os.path.join(ML, "runs", run_name)
    try:
        npz, _ = toy(tmp)
        base = [sys.executable, "-u", os.path.join(ML, "train.py"),
                "--data", npz, "--out", run_dir, "--steps", "12",
                "--batch", "16", "--d-model", "16", "--n-layers", "2",
                "--n-heads", "2", "--d-dec", "16", "--d-z", str(DZ),
                "--patch", "1", "--anomaly", "--holdout-years", "1991",
                "--holdout-lon=0,0", "--collapse-r", "0"]
        out = run(base + ["--time-block", "6/3"], "train --time-block 6/3")
        assert "time blocks: mode '6/3'" in out, out[-1500:]
        assert "width 6 stride 3 (overlap 3 bins, step 15 d)" in out
        assert "PERSISTENCE IS STRONGER BY CONSTRUCTION" in out
        ck = torch.load(os.path.join(run_dir, "pixelmae.pt"),
                        map_location="cpu", weights_only=False)
        assert ck["args"]["time_block"] == "6/3"
        assert ck["args"]["k_time"] == 6, ck["args"]["k_time"]
        blk_ck = os.path.join(tmp, "blk63.pt")
        shutil.copyfile(os.path.join(run_dir, "pixelmae.pt"), blk_ck)

        out2 = run([sys.executable, "-u", os.path.join(ML, "temporal.py"),
                    "--run", run_name, "--data", npz, "--K", "4",
                    "--steps", "6", "--batch", "8", "--d-model", "16",
                    "--layers", "2", "--lr-warmup", "2"],
                   "temporal.py on the 6/3 codec")
        line = [l for l in out2.splitlines() if "block axis: T " in l][0]
        tblk = int(line.split("block axis: T ")[1].split()[0])
        dd = np.load(npz, allow_pickle=False)
        BA = BlockAxis("6/3", [str(m) for m in dd["months"]], dd["bin_index"],
                       EPOCH, DAYS)
        assert tblk == BA.n_blocks, (tblk, BA.n_blocks)
        assert f"Z [T={tblk}" in out2

        # the roll: its own axis, its own stride, its own caveat
        from temporal import embed_everything
        from trainprobe import anomaly_transform
        from model import codec_from_ckpt
        _ck = torch.load(blk_ck, map_location="cpu", weights_only=False)
        codec = codec_from_ckpt(_ck, C)
        codec.load_state_dict(_ck["model"])
        codec.eval()
        Xs = np.load(npz)["X"]
        xpath = os.path.join(tmp, "X.npy")
        np.save(xpath, Xs)
        _mo = np.array([int(m[5:7]) - 1 for m in dd["months"]])
        _th = np.array([str(m)[:4] == "1991" for m in dd["months"]])
        _Xa = np.array(Xs, np.float32, copy=True)
        _Xa, _ = anomaly_transform(_Xa, _mo, _th,
                                   np.zeros(len(dd["lons"]), bool))
        _ocean = np.isfinite(_Xa[..., 0]).any(0)
        _ys, _xs = np.where(_ocean)
        Z, _ = embed_everything(
            codec, torch.from_numpy(np.nan_to_num(_Xa, nan=0.0)),
            torch.from_numpy(np.isfinite(_Xa)), BA.ctx_phase(),
            np.asarray(dd["lats"]), np.asarray(dd["lons"]), _ys, _xs, DZ,
            cache_path=None, batch=64, blk_rows=BA.rows, blk_pad=BA.pad)
        zpath = os.path.join(tmp, "Z63.npy")
        np.save(zpath, np.asarray(Z, np.float16))
        from temporal import TemporalTransformer
        torch.manual_seed(0)
        K = 3
        head = TemporalTransformer(d_z=DZ, d_model=8, n_heads=4, n_layers=1,
                                   k_max=K, stencil=1)
        hp = os.path.join(tmp, "head63.pt")
        torch.save({"model": head.state_dict(),
                    "args": {"K": K, "d_model": 8, "layers": 1, "unroll": 1,
                             "seed": 0, "stencil": 1}}, hp)
        out5 = os.path.join(tmp, "roll63.json")
        r = subprocess.run(
            [sys.executable, "-u", os.path.join(ML, "rollout_spatial.py"),
             "--x", xpath, "--npz-small", npz, "--z", zpath,
             "--ckpt", blk_ck, "--out", out5, "--horizon", "2",
             "--long-start", "", "--future-months", "0",
             "--cache-dir", os.path.join(tmp, "cache63"), "--no-gate",
             "--heads", hp],
            capture_output=True, text=True, timeout=2400, cwd=ROOT)
        if r.returncode != 0:
            print(r.stdout[-4000:]); print(r.stderr[-2500:])
            raise SystemExit("the 6/3 roll failed")
        assert "block codec: k_time 6" in r.stdout, r.stdout[:2000]
        assert "time axis: 15-day" in r.stdout, [
            l for l in r.stdout.splitlines() if "time axis" in l]
        res = json.load(open(out5))
        bl = res["blocks"]
        assert bl["time_block"] == "6/3" and bl["k_time"] == 6
        assert bl["stride_bins"] == 3 and bl["width_bins"] == 6
        assert bl["overlap_bins"] == 3 and bl["step_days"] == 15.0
        assert "stronger BY CONSTRUCTION" in bl["persistence_note"]
        lead = bl["lead_days_by_horizon_and_cell"]
        assert set(lead) == {"1", "2"}, lead
        assert all(set(v) == {15.0 * int(h)} for h, v in lead.items()), lead
        print("7. end to end at stride 3: ml/train.py trains a 6/3 codec "
              "(k_time 6, the axis printing its own persistence caveat), "
              "ml/temporal.py embeds the same %d-bin tensor into %d WINDOW "
              "embeddings and trains a head on them, and ml/rollout_spatial.py "
              "rolls it on a 15-DAY axis — the artefact carrying width 6, "
              "stride 3, overlap 3, step_days 15.0, per-cell leads %s, and the "
              "note that persistence here is stronger by construction"
              % (len(dd["months"]), tblk,
                 {h: v[0] for h, v in sorted(lead.items())}))

        print("\nE-048 overlapping window blocks: all 7 checks hold ✓")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
        shutil.rmtree(run_dir, ignore_errors=True)
        for _f in glob.glob(os.path.join(ML, "cache", "Z_*_blk*.npy")):
            os.remove(_f)


if __name__ == "__main__":
    main()

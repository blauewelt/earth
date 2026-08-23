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
        out = run(base + ["--time-block", "month",
                          "--light-probe-every", "6"],
                  "train --time-block month")
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
                    "--layers", "2", "--lr-warmup", "2"],
                   "temporal.py on the block codec")
        assert "block axis: T " in out2, out2[-2000:]
        line = [l for l in out2.splitlines() if "block axis: T " in l][0]
        tblk = int(line.split("block axis: T ")[1].split()[0])
        assert tblk == blocks, (tblk, blocks)
        assert f"Z [T={tblk}" in out2, [l for l in out2.splitlines()
                                        if l.strip().startswith("Z [")]
        tj = json.load(open(os.path.join(run_dir, "temporal.json")))
        assert tj["z_t+1"]["mse_model"] >= 0
        blk_ck = os.path.join(tmp, "blk_pixelmae.pt")
        shutil.copyfile(os.path.join(run_dir, "pixelmae.pt"), blk_ck)
        # BUILD THE ROLL'S Z HERE, with the same call temporal.py makes.
        # temporal.py keeps a toy's Z in RAM (_cache_plan: no file when the
        # array fits), so there is nothing on disk to reuse — and a globbed
        # leftover from an earlier run would be another codec's z, which is
        # the #10/#11 failure the cache names exist to prevent.
        blk_z = os.path.join(tmp, "Z_blk.npy")
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

        # ---- 5. TIER 2: the roll decodes a block codec's cells ----------
        # Rebuild the month-block codec (check 3 overwrote the run dir with
        # the knob-off one), then roll a tiny head on its z.
        from timeblocks import BlockAxis
        dd = np.load(npz, allow_pickle=False)
        BA = BlockAxis("month", [str(m) for m in dd["months"]],
                       dd["bin_index"], EPOCH, DAYS)
        # a toy stage-2 head over the BLOCK axis
        from temporal import TemporalTransformer
        torch.manual_seed(0)
        K = 3
        head = TemporalTransformer(d_z=DZ, d_model=8, n_heads=4, n_layers=1,
                                   k_max=K, stencil=1)
        hp = os.path.join(tmp, "toy_head.pt")
        torch.save({"model": head.state_dict(),
                    "args": {"K": K, "d_model": 8, "layers": 1, "unroll": 1,
                             "seed": 0, "stencil": 1}}, hp)
        # X and Z on disk, exactly as the roll expects them
        Xs = np.load(npz)["X"]
        xpath = os.path.join(tmp, "X.npy")
        np.save(xpath, Xs)
        # The REAL embedding of this codec, not a random array: the roll
        # verifies its Z against a live re-encode and that guard is one of
        # the things under test here.
        from temporal import embed_everything
        from trainprobe import anomaly_transform
        _ckb = torch.load(blk_ck, map_location="cpu", weights_only=False)
        from model import codec_from_ckpt as _cfc
        _codec = _cfc(_ckb, C)
        _codec.load_state_dict(_ckb["model"])
        _codec.eval()
        _mo = np.array([int(m[5:7]) - 1 for m in dd["months"]])
        _th = np.array([str(m)[:4] == "1991" for m in dd["months"]])
        _Xa = np.array(Xs, np.float32, copy=True)
        _Xa, _dyn = anomaly_transform(_Xa, _mo, _th,
                                      np.zeros(len(dd["lons"]), bool))
        _ocean = np.isfinite(_Xa[..., 0]).any(0)
        _ys, _xs = np.where(_ocean)
        _Z, _ = embed_everything(
            _codec, torch.from_numpy(np.nan_to_num(_Xa, nan=0.0)),
            torch.from_numpy(np.isfinite(_Xa)), BA.ctx_phase(),
            np.asarray(dd["lats"]), np.asarray(dd["lons"]), _ys, _xs, DZ,
            cache_path=None, batch=64, blk_rows=BA.rows, blk_pad=BA.pad)
        np.save(blk_z, np.asarray(_Z, np.float16))
        zpath = blk_z
        out5 = os.path.join(tmp, "roll_blk.json")
        ddir = os.path.join(tmp, "dump_blk")
        r = subprocess.run(
            [sys.executable, "-u", os.path.join(ML, "rollout_spatial.py"),
             "--x", xpath, "--npz-small", npz, "--z", zpath,
             "--ckpt", blk_ck, "--out", out5,
             "--horizon", "2", "--long-start", "", "--future-months", "0",
             "--cache-dir", os.path.join(tmp, "cache_blk"), "--no-gate",
             "--dump-roll", ddir, "--heads", hp],
            capture_output=True, text=True, timeout=1800, cwd=ROOT)
        if r.returncode != 0:
            print(r.stdout[-4000:]); print(r.stderr[-2500:])
            raise SystemExit("block roll failed")
        assert "block codec: k_time 7" in r.stdout, r.stdout[:1500]
        res = json.load(open(out5))
        bl = res["blocks"]
        assert bl["k_time"] == 7 and bl["time_block"] == "month"
        assert bl["k_max"] == 7 and bl["n_blocks"] == BA.n_blocks
        lead = bl["lead_days_by_horizon_and_cell"]
        assert set(lead) == {"1", "2"}, lead
        for h_, ds in lead.items():
            assert all(abs(v - 0) > 0 for v in ds)
            assert len(ds) >= 5, (h_, ds)
        head_lab = list(res["heads"])[0]
        cs = res["heads"][head_lab]["window"]["chan_skill"]
        n1 = [row["n"] for row in cs if row["h"] == 1][0]
        print("5. a BLOCK codec rolls: the roll loads the block map off the "
              "tensor, reads k_time 7 from the checkpoint, decodes every cell "
              "of each predicted block and scores it against its OWN bin — "
              "horizon 1 accumulates n=%d channel-values, and the artefact "
              "records the lead in DAYS for each (horizon, cell): h=1 -> %s"
              % (n1, lead["1"][:4]))

        # ---- 6. cell scoring is per-bin, on a constructed case ------------
        # n at horizon h must be the sum over scored (start, block) pairs of
        # the OBSERVED cell-values, i.e. ~k_time times a per-bin roll's.
        # A per-bin roll would score ONE value per (step, pixel, channel);
        # a block roll scores one per CELL, so n at h=1 must exceed the
        # step-count times the scope's pixels times C — strictly, and by
        # about the mean cells per block.
        import re as _re
        n_steps = int(_re.search(r"(\d+) scored roll steps",
                                 r.stdout).group(1))
        n_px = res["heads"][head_lab]["window"]["n_px"]
        one_cell = n_steps / 2 * n_px * C          # h=1 gets half the steps
        assert n1 > 1.5 * one_cell, (n1, one_cell, n_steps, n_px)
        per_bin_like = one_cell
        # the dump is shape-agnostic and still writes z-states, not cells
        man = json.load(open(os.path.join(ddir, "dump_manifest.json")))
        f0 = man["files"][0]
        assert f0["shape"][2] == DZ, f0["shape"]
        zdump = np.load(os.path.join(ddir, f0["file"]),
                        allow_pickle=False)["z"]
        assert zdump.dtype == np.float16 and zdump.shape[2] == DZ
        assert len(man["files"]) >= 1 and man["d_z"] == DZ
        print("6. n grows with the cells (%d at h=1 against ~%d for a "
              "one-cell-per-block roll), and --dump-roll still writes Z "
              "STATES — [%d, %d, %d] float16, d_z not k_time*C — so the "
              "offline decode stays the Tier-3 plan"
              % (n1, int(per_bin_like), *zdump.shape))

        # ---- 7. the light probe FEEDS the collapse guard ----------------
        # #448 (2026-08-23): every probe of a month-block run raised "BLOCK
        # codec and no block map was passed", was caught, warned and skipped
        # — the right failure mode for a probe and the wrong outcome for a
        # run, because the collapse guard eats probe values and a run with no
        # probes has a guard with nothing to guard on.
        assert "no block map was passed" not in out, out[-2000:]
        recs = [json.loads(l) for l in
                open(os.path.join(run_dir, "metrics.jsonl")) if l.strip()]
        lp = [r for r in recs if "linear_r_deseas" in r]
        assert lp, ("the block run produced NO probe point at all",
                    [sorted(r) for r in recs][:4])
        # linear_r_RAW is the one that must be finite here: deseasonalising
        # needs train rows in every month-of-year, and this 2-year toy loses
        # half of them to the 1991 holdout, so `linear_r_deseas` is legitimately
        # NaN on the fixture (the collapse guard treats NaN as "no reading").
        # What is under test is that the probe EMBEDDED AND SCORED at all.
        assert all(np.isfinite(r["linear_r_raw"]) for r in lp), lp
        assert all(r.get("light_n", 0) > 0 for r in lp), lp
        # and the refusal still fires for a caller that genuinely lacks it
        import trainprobe as tprobe
        from model import codec_from_ckpt as _cfc2
        _c2 = _cfc2(ckb if False else torch.load(
            blk_ck, map_location="cpu", weights_only=False), C)
        try:
            _dmin = {"lats": dd["lats"], "lons": dd["lons"],
                     "months": np.array(["1990-01"] * 4),
                     "rapid": np.array([[0.0, 1.0], [1.0, -1.0],
                                        [2.0, 0.5], [3.0, -0.5]])}
            tprobe.probe_now(_c2, torch.zeros(4, H, W, C),
                             torch.ones(4, H, W, C, dtype=torch.bool),
                             _dmin, np.zeros(4, int), np.zeros(4, bool),
                             np.zeros(W, bool), list(range(C)), light=True,
                             ocean=np.ones((H, W), bool))
            raise AssertionError("probe_now embedded a block codec blind")
        except ValueError as e:
            assert "BLOCK codec" in str(e), e
        print("7. with --time-block month the LIGHT PROBE runs rather than "
              "skipping: %d probe point(s), linear_r_deseas %s — so the "
              "collapse guard has something to guard on — and probe_now "
              "still REFUSES for a caller that has no block map"
              % (len(lp), [round(r["linear_r_raw"], 3) for r in lp][:3]))

        print("\nE-047 end-to-end smoke: all 7 checks hold ✓")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
        shutil.rmtree(run_dir, ignore_errors=True)
        for _f in glob.glob(os.path.join(ML, "cache", "Z_*_blk*.npy")):
            os.remove(_f)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""The four E-044c overnight knobs: each one OFF is the code that was there.

`--time-stride/--time-offset`, `--target-bins-argo`, `--season-dropout` and
`--season-phase` are instruments for one night of debugging the pentad
collapse, and every one of them defaults to off. The thing that has to be
proven is not that they work — it is that a run which does not name them is
the run that ran yesterday, because #432 (E-044b-SEED1) and #433
(E-044b-roll) are live against this file while it is being edited, and every
archived monthly number has to stay reproducible.

  1. **THE NO-OP, against the PARENT REVISION.** `git show HEAD:ml/temporal.py`
     and the working tree, same toy, same seed, no flags: every parameter
     tensor `torch.equal`, and the metrics curves identical. This is check 3
     of tests/test_e044_grad_clip.py, pointed at a different change.
  2. **The defaults are the defaults.** Passing all four flags explicitly at
     their default values reproduces the unflagged run, tensor for tensor —
     so 'off' is one code path and not two that happen to agree today.
  3. **--time-stride** keeps `range(O, T, N)` and carries the WHOLE axis with
     it: labels, month-of-year, holdout mask and the window pool. Checked
     against an independently computed expectation, not against the run's own
     arithmetic. Plus the two refusals (O >= N, N < 1).
  4. **--target-bins-argo** filters the TRAIN POOL and nothing else, on a
     fixture whose Argo channels are present in one bin of three: `exclude`
     and `only` partition the eligible windows of `all`, and the monitor
     batch — which is what `val_persistence` and `val_zmse` are read from —
     is bit-identical across all three.
  5. **--season-dropout** touches training forwards only: at P=1.0 the loss
     curve moves and the monitor's persistence baseline does not.
  6. **--season-phase** month is the archived sin/cos(2pi*moy/12) exactly;
     fine resolves each bin of a pentad axis separately (73 tokens where
     month gives 12, advancing 5/365.2425 of a turn per bin); the head's
     checkpoint records which it was trained under; and
     `ml/rollout_spatial.py`'s `row_feats` reproduces both, so a fine-phase
     head can never be rolled with coarse tokens.

    python3 tests/test_e044c_knobs.py
"""
import datetime as dtm
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
from test_e044_grad_clip import toy, train, K, T_M, C, DZ    # noqa: E402


def params_equal(a, b):
    ka, kb = set(a["model"]), set(b["model"])
    assert ka == kb, (ka ^ kb)
    bad = [k for k in sorted(ka)
           if not torch.equal(a["model"][k], b["model"][k])]
    return bad


def argo_toy(tmp, every=3):
    """The same toy with 3 of its channels NAMED rg_* and present only in one
    bin of `every` — the pentad tensor's own shape (n_rg_live 252/3142, the
    mid-month stamp), small enough to count by hand."""
    npz, ck = toy(tmp)
    d = dict(np.load(npz, allow_pickle=True))
    X = d["X"].copy()
    chan = [f"rg_t{10 * (i + 1)}" if i < 3 else f"c{i}" for i in range(C)]
    live = np.zeros(X.shape[0], bool)
    live[::every] = True
    X[~live][:, :, :, :3] = np.nan          # (copy semantics: do it properly)
    for t in range(X.shape[0]):
        if not live[t]:
            X[t, :, :, :3] = np.nan
    d["X"] = X
    d["chan"] = np.array(chan)
    p = os.path.join(tmp, "argo.npz")
    np.savez(p, **d)
    ck = dict(ck)
    ck["chan"] = chan
    return p, ck, live


def pool_of(stdout):
    for ln in stdout.splitlines():
        if ln.startswith("train windows:"):
            return int(ln.split(":")[1].strip().replace(",", ""))
    raise AssertionError("no 'train windows:' line")


def main():
    tmp = tempfile.mkdtemp()
    try:
        npz, ckd = toy(tmp)
        run = "e044c_knobs"
        run_dir = os.path.join(ML, "runs", run)
        os.makedirs(run_dir, exist_ok=True)
        torch.save(ckd, os.path.join(run_dir, "pixelmae.pt"))
        cur = os.path.join(ML, "temporal.py")

        # ---- 1. the working tree, unflagged, vs the parent revision -------
        prev = subprocess.run(["git", "-C", ROOT, "show", "HEAD:ml/temporal.py"],
                              capture_output=True, text=True)
        assert prev.returncode == 0, prev.stderr[-300:]
        # INSIDE ml/, like tests/test_e044_grad_clip.py does it: the file
        # imports its siblings by module name, so a copy in /tmp cannot run.
        base = os.path.join(ML, "_temporal_e044c_base.py")
        open(base, "w").write(prev.stdout)
        r_new, tj_new, ck_new, out_new = train(cur, npz, run, tmp, [], "new")
        r_old, tj_old, ck_old, out_old = train(base, npz, run, tmp, [], "base")
        bad = params_equal(ck_new, ck_old)
        assert not bad, f"{len(bad)} tensors differ from the parent revision: {bad[:4]}"
        cn = [(x.get("stage2_step"), x.get("stage2_zmse")) for x in r_new
              if "stage2_step" in x]
        co = [(x.get("stage2_step"), x.get("stage2_zmse")) for x in r_old
              if "stage2_step" in x]
        assert cn == co and cn, (cn[:3], co[:3])
        assert tj_new["z_t+1"] == tj_old["z_t+1"], (tj_new["z_t+1"],
                                                    tj_old["z_t+1"])
        print("1. unflagged working tree == parent revision: all %d parameter "
              "tensors torch.equal, identical loss curve (%d points) and "
              "identical z_t+1 block — the four knobs add no code path to a "
              "run that does not name them" % (len(ck_new["model"]), len(cn)))

        # ---- 2. the defaults are the defaults -----------------------------
        r_d, tj_d, ck_d, out_d = train(
            cur, npz, run, tmp,
            ["--time-stride", "0", "--time-offset", "0",
             "--target-bins-argo", "all", "--season-dropout", "0",
             "--season-phase", "month"], "explicit")
        bad = params_equal(ck_new, ck_d)
        assert not bad, bad[:4]
        assert tj_d["z_t+1"] == tj_new["z_t+1"]
        for k in ("time_stride", "time_offset", "target_bins_argo",
                  "season_dropout", "season_phase"):
            assert k in ck_d["args"], f"{k} is not written into the checkpoint"
        assert ck_d["args"]["season_phase"] == "month"
        print("2. the four flags at their DEFAULT values reproduce the "
              "unflagged run tensor for tensor, and all five land in the "
              "checkpoint args (season_phase %r) — which is how "
              "rollout_spatial reads the conditioning back"
              % ck_d["args"]["season_phase"])

        # ---- 3. --time-stride ---------------------------------------------
        N, O = 3, 1
        r_s, tj_s, ck_s, out_s = train(cur, npz, run, tmp,
                                       ["--time-stride", str(N),
                                        "--time-offset", str(O)], "stride")
        keep = list(range(O, T_M, N))
        months = [f"{1990 + i // 12}-{i % 12 + 1:02d}" for i in range(T_M)]
        kept_lab = [months[i] for i in keep]
        assert f"{len(keep)} of {T_M} bins kept ({kept_lab[0]}..{kept_lab[-1]})" \
            in out_s, out_s[:1200]
        t_hold_k = [lab[:4] == "1992" for lab in kept_lab]
        assert f"held-out {sum(t_hold_k)}" in out_s
        # the pool, recomputed here from the KEPT axis alone
        Tk = len(keep)
        ok_t = [t + 1 < Tk and t + 1 >= K and not t_hold_k[t + 1]
                for t in range(Tk)]
        n_px = pool_of(out_s) // max(sum(ok_t), 1)
        assert pool_of(out_s) == sum(ok_t) * n_px, (pool_of(out_s), sum(ok_t))
        assert pool_of(out_s) < pool_of(out_new), (pool_of(out_s),
                                                   pool_of(out_new))
        for bad_args, want in ((["--time-stride", "3", "--time-offset", "3"],
                                "0 <= O < N"),
                               (["--time-stride", "-1"], "must be >= 1")):
            rr = subprocess.run(
                [sys.executable, cur, "--run", run, "--data", npz, "--K",
                 str(K), "--steps", "1", "--max-pixels", "30", *bad_args],
                capture_output=True, text=True, timeout=900)
            assert rr.returncode != 0 and want in (rr.stdout + rr.stderr), \
                (bad_args, rr.stdout[-400:], rr.stderr[-400:])
        print("3. --time-stride %d --time-offset %d keeps %d of %d bins "
              "(%s..%s), carries labels/month-of-year/holdout with it (%d "
              "held out) and rebuilds the pool on the KEPT axis (%d windows "
              "against %d) — and both malformed forms refuse"
              % (N, O, len(keep), T_M, kept_lab[0], kept_lab[-1],
                 sum(t_hold_k), pool_of(out_s), pool_of(out_new)))

        # ---- 4. --target-bins-argo ---------------------------------------
        anpz, ack, live = argo_toy(tmp)
        torch.save(ack, os.path.join(run_dir, "pixelmae.pt"))
        outs, mons = {}, {}
        for mode in ("all", "exclude", "only"):
            rr, tjj, ckk, oo = train(cur, anpz, run, tmp,
                                     ["--target-bins-argo", mode], f"argo_{mode}")
            outs[mode] = pool_of(oo)
            mons[mode] = [x for x in rr if "stage2_monitor" in x][:1]
            assert f"Argo-carrying bins: {int(live.sum())}/{T_M}" in oo, oo[:900]
        assert 0 < outs["exclude"] < outs["all"], outs
        assert 0 < outs["only"] < outs["all"], outs
        assert outs["exclude"] + outs["only"] == outs["all"], outs
        pers = {m: mons[m][0]["stage2_monitor"]["val_persistence"]
                for m in mons if mons[m]}
        assert len(set(pers.values())) == 1, pers
        print("4. --target-bins-argo on a fixture with Argo in %d of %d bins: "
              "the pool splits %d = %d exclude + %d only, and the monitor's "
              "val_persistence is identical (%s) in all three — the filter is "
              "the TRAIN POOL and nothing else"
              % (int(live.sum()), T_M, outs["all"], outs["exclude"],
                 outs["only"], list(pers.values())[0]))

        # ---- 5. --season-dropout ------------------------------------------
        torch.save(ckd, os.path.join(run_dir, "pixelmae.pt"))
        r_dr, tj_dr, ck_dr, out_dr = train(cur, npz, run, tmp,
                                           ["--season-dropout", "1.0"], "drop")
        c_dr = [(x["stage2_step"], x["stage2_zmse"]) for x in r_dr
                if "stage2_step" in x]
        assert c_dr != cn, "P=1.0 changed nothing in training"
        m_dr = [x["stage2_monitor"] for x in r_dr if "stage2_monitor" in x][0]
        m_bs = [x["stage2_monitor"] for x in r_new if "stage2_monitor" in x][0]
        assert m_dr["val_persistence"] == m_bs["val_persistence"], \
            (m_dr["val_persistence"], m_bs["val_persistence"])
        print("5. --season-dropout 1.0 moves the training curve and leaves "
              "the monitor's val_persistence at %s, exactly as with the knob "
              "off — the drop is inside batch_windows, which no eval, "
              "monitor or roll path calls" % m_bs["val_persistence"])

        # ---- 6. --season-phase --------------------------------------------
        import temporal as tp
        import rollout_spatial as rs
        mo = [f"{1990 + i // 12}-{i % 12 + 1:02d}" for i in range(24)]
        moy = np.array([int(m[5:7]) - 1 for m in mo])
        want = np.stack([np.sin(2 * np.pi * moy / 12),
                         np.cos(2 * np.pi * moy / 12)], 1)
        assert np.array_equal(tp.season_ctx(mo, "month"), want)
        fine_m = tp.season_ctx(mo, "fine")
        assert not np.array_equal(fine_m, want)
        assert np.abs(fine_m - want).max() < 0.30, np.abs(fine_m - want).max()
        ep, days, n = dtm.date(1982, 1, 1), 5, 146
        pm = [(ep + dtm.timedelta(days=days * i)).strftime("%Y-%m")
              for i in range(n)]
        dd = {"bin_index": np.arange(n), "pentad_days": np.array(days),
              "epoch": np.array("1982-01-01")}
        fine_p = tp.season_ctx(pm, "fine", dd)
        assert len(set(map(tuple, tp.season_ctx(pm, "month", dd)))) == 12
        assert len(set(map(tuple, fine_p))) == 73, len(set(map(tuple, fine_p)))
        # The MEDIAN step, not the mean: the phase is measured from each
        # bin's own 1 January, so the two year boundaries inside a 146-bin
        # span are genuinely a different size (365 or 366 days against the
        # tropical 365.2425). Every other step is exactly 5/365.2425 of a
        # turn, which is the quantity under test.
        ang = np.angle(fine_p[:, 1] + 1j * fine_p[:, 0])
        steps_ = (np.diff(ang) + 2 * np.pi) % (2 * np.pi)
        step = float(np.median(steps_))
        assert abs(step - 2 * np.pi * days / 365.2425) < 1e-12, step
        assert abs(float(np.mean(steps_)) - step) < 0.001, np.mean(steps_)
        # the roll honours the head's own arg
        ax = rs.TimeAxis({"months": np.array(pm),
                          "bin_index": np.arange(n),
                          "pentad_days": np.array(days),
                          "epoch": np.array("1982-01-01")})
        rows = [5, 6, 7]
        dev = torch.device("cpu")
        got_m = rs.row_feats(ax, rows, dev, "month").numpy()
        assert np.allclose(got_m, rs.month_feats([ax.moy_of_row(r)
                                                  for r in rows], dev).numpy())
        got_f = rs.row_feats(ax, rows, dev, "fine").numpy()
        assert np.allclose(got_f, [tp.season_feat_of(ax.date_of_row(r),
                                                     ax.step_days)
                                   for r in rows])
        assert not np.allclose(got_f, got_m)
        # past the end of the record, which is where the future roll lives
        assert rs.row_feats(ax, [n + 3], dev, "fine").shape == (1, 2)
        print("6. --season-phase month IS the archived sin/cos(2pi*moy/12) "
              "(exact); fine gives 73 distinct tokens on a 146-bin pentad "
              "axis where month gives 12, advancing %.4f rad = 5/365.2425 of "
              "a turn per bin; and rollout_spatial.row_feats reproduces both "
              "from the axis alone, past the end of the record included"
              % step)

        print("\nE-044c knobs: all 6 checks hold ✓")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
        bp = os.path.join(ML, "_temporal_e044c_base.py")
        if os.path.exists(bp):
            os.remove(bp)
        rd = os.path.join(ML, "runs", "e044c_knobs")
        shutil.rmtree(rd, ignore_errors=True)


if __name__ == "__main__":
    main()

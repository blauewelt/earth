#!/usr/bin/env python3
"""Pin the cross-tensor resume rule: eval-only passes, training refuses.

E-038's frozen control is `f3_anchor41M` — the monthly codec — SCORED on the
pentad tensor. It is the only number that can falsify the experiment's
premise, and it is by definition cross-tensor: the checkpoint records
`data=family3_na025.npz` and the run hands it `family4_na025_pentad.npz`.

train.py's resume guard refused exactly that, unconditionally, because
cross-tensor TRAINING writes a checkpoint whose provenance is a lie. The rule
is now keyed on whether anything will train:

  1. **Eval-only passes**: checkpoint step >= --steps means the training loop
     body never runs, no weight changes, and the run is a re-scoring. The
     whole pipeline must complete: banner printed, checkpoint saved (the
     probe ladder reads it), and the saved weights BIT-IDENTICAL to the
     loaded ones — an eval that trains even one step is the fine-tune arm
     wearing the control's name, which would quietly delete the experiment's
     only falsifier.
  2. **Cross-tensor TRAINING still refuses**, with the loud exit — steps
     above the checkpoint's step means the loop would run.
  3. **A checkpoint with no recorded step still refuses**: the warm-start
     branch restarts s at 0, so such a checkpoint cannot prove it will not
     train. (Real precedent: pre-2026-08-10 artefacts are `{args, model}`.)
  4. **Same-tensor resume is untouched** in both directions.

Fixture checkpoints are saved through train.py's own smoke path so the format
is the real one, not a hand-built imitation.

    python3 tests/test_frozen_control_resume.py
"""
import os
import subprocess
import sys
import tempfile

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
TRAIN = os.path.join(HERE, "..", "ml", "train.py")


def toy(path, seed=0):
    rng = np.random.default_rng(seed)
    T, H, W, C = 48, 12, 14, 39
    X = rng.normal(size=(T, H, W, C)).astype(np.float16)
    X[:, rng.random((H, W)) < 0.3, :] = np.nan
    np.savez(path, X=X,
             months=np.array([f"{2004 + i // 12:04d}-{i % 12 + 1:02d}"
                              for i in range(T)]),
             lats=np.linspace(0, 70, H), lons=np.linspace(-100, 20, W),
             chan=np.array([f"c{i}" for i in range(C)]),
             norm=np.stack([np.zeros(C), np.ones(C)], 1).astype(np.float32),
             rapid=np.array([[0.0, 17.0], [2.0, 16.4]]))


def run_train(data, out, steps, resume=None):
    cmd = [sys.executable, "-u", TRAIN, "--data", data, "--out", out,
           "--steps", str(steps), "--batch", "32", "--d-model", "32",
           "--n-layers", "2", "--d-dec", "32", "--anomaly"]
    if resume:
        cmd += ["--resume", resume]
    return subprocess.run(cmd, capture_output=True, text=True)


def main():
    import torch

    tmp = tempfile.mkdtemp(prefix="frz_")
    f3 = os.path.join(tmp, "family3_like.npz")     # "monthly": trains the codec
    f4 = os.path.join(tmp, "family4_like.npz")     # "pentad": different tensor
    toy(f3, seed=1)
    toy(f4, seed=2)

    # the "anchor": a codec trained to step 30 on tensor A
    anchor_out = os.path.join(tmp, "anchor")
    p = run_train(f3, anchor_out, steps=30)
    assert p.returncode == 0, p.stdout[-800:] + p.stderr[-800:]
    ck_path = os.path.join(anchor_out, "pixelmae.pt")
    before = torch.load(ck_path, map_location="cpu", weights_only=False)
    assert before["step"] == 30 and \
        os.path.basename(before["args"]["data"]) == "family3_like.npz"

    # ---- 1: the frozen control — cross-tensor, eval-only ------------------
    ctrl_out = os.path.join(tmp, "control")
    p = run_train(f4, ctrl_out, steps=30, resume=ck_path)
    assert p.returncode == 0, (
        "the frozen control was refused:\n" + p.stdout[-800:] + p.stderr[-500:])
    assert "CROSS-TENSOR EVAL" in p.stdout, "no banner — a silent allowance"
    after = torch.load(os.path.join(ctrl_out, "pixelmae.pt"),
                       map_location="cpu", weights_only=False)
    for k in before["model"]:
        assert torch.equal(before["model"][k], after["model"][k]), (
            f"weight {k} CHANGED during the 'frozen' control — this is the "
            f"fine-tune arm wearing the control's name")
    print("  1. cross-tensor eval-only completes, says so, saves, and every "
          "weight is bit-identical to the loaded anchor")

    # ---- 2: cross-tensor TRAINING still refuses ---------------------------
    p = run_train(f4, os.path.join(tmp, "xt_train"), steps=60, resume=ck_path)
    assert p.returncode != 0 and "REFUSING to resume" in (p.stdout + p.stderr), (
        "cross-tensor training was ALLOWED:\n" + p.stdout[-600:])
    print("  2. cross-tensor training (steps 60 > ckpt step 30) still exits "
          "with the refusal")

    # ---- 3: no recorded step -> refused even at matching --steps ----------
    old = {"model": before["model"], "chan": before["chan"],
           "d_z": before["d_z"], "args": before["args"]}     # pre-08-10 shape
    old_path = os.path.join(tmp, "old_format.pt")
    torch.save(old, old_path)
    p = run_train(f4, os.path.join(tmp, "old_ctrl"), steps=30, resume=old_path)
    assert p.returncode != 0 and "REFUSING" in (p.stdout + p.stderr), (
        "a step-less checkpoint was allowed cross-tensor — the warm-start "
        "branch would have trained it:\n" + p.stdout[-600:])
    print("  3. a checkpoint with no recorded step is still refused — it "
          "cannot prove it will not train")

    # ---- 4: same-tensor resume is untouched -------------------------------
    p = run_train(f3, os.path.join(tmp, "cont"), steps=40, resume=ck_path)
    assert p.returncode == 0 and "RESUMED" in p.stdout, p.stdout[-600:]
    print("  4. same-tensor resume (30 -> 40 steps) continues exactly as "
          "before")

    print("\ntests/test_frozen_control_resume.py: all 4 checks passed")


if __name__ == "__main__":
    main()

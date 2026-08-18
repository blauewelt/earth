#!/usr/bin/env python3
"""train.py's own configuration guards, end to end on a toy tensor.

(Named for train.py rather than "dispatch" because ml/dispatch_config.py and
its tests/test_dispatch_config.py are a separate piece of work — a decoder for
the legacy `window` string. Different layer, different failure mode.)

Written 2026-08-18 after two failures of the same shape in one day:

  · #395 named `--resume !run-62` but not the architecture, so the trainer
    built the 0.92M pilot (128/4/4/256, the argparse defaults), tried to load
    576-wide weights into it, and died after 90 s with sixty "size mismatch"
    lines.
  · #387 raised d_model to 1024 but left `codec_heads` at the workflow
    default 4 — head_dim 256 — and the 202M codec's embedding collapsed
    between step 10k and 15k. Nothing stopped it for another nine hours.

Neither was a typo. Both were a field nobody restated, in a system where an
omitted field silently meant "run a different experiment". These tests pin
the four behaviours that replace that silence:

Case 1: no architecture and no --resume  -> REFUSE (exit != 0), naming the
        unset flags. It must not fall back to the pilot.
Case 2: --resume with NO architecture    -> ADOPT the checkpoint's, and
        produce a second checkpoint whose architecture is identical.
Case 3: --resume with a CONTRADICTING    -> REFUSE, naming both values, and
        architecture                        refuse BEFORE loading weights.
Case 4: --collapse-r above every probe   -> ABORT with a "collapsed" record
Case 5: --resume a checkpoint whose      -> ADOPT it. A field carrying a real
        dec_layers differs from the         default is not a contradiction just
        argparse default                    because it is non-None.
        reading                             in metrics.jsonl.

    python3 tests/test_train_config_guards.py
"""
import json
import os
import subprocess
import sys
import tempfile

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ML = os.path.join(HERE, "..", "ml")

# Big enough that the held-out split is non-degenerate and the probe returns a
# REAL correlation. At 30x8x10 with the default holdout years (2009/2017/2023,
# none of which the toy months cover) the probe returned NaN on a perfectly
# healthy run — which is exactly the reading the collapse guard must not treat
# as collapse, so a toy that produces it cannot test the guard at all.
T_M, H_G, W_G, C = 96, 10, 12, 5
HOLD = ["--holdout-years", "1996", "--holdout-lon=-45,-40"]
ARCH = ["--d-z", "4", "--patch", "3", "--d-model", "16",
        "--n-layers", "1", "--n-heads", "2", "--d-dec", "24"]


def trainer(npz, out, extra):
    return subprocess.run(
        [sys.executable, "-u", os.path.join(ML, "train.py"),
         "--data", npz, "--out", out, "--steps", "25", "--batch", "16",
         "--anomaly", *HOLD, *extra],
        capture_output=True, text=True, timeout=900)


def toy_npz(tmp):
    rng = np.random.default_rng(0)
    t = np.arange(T_M)[:, None, None, None]
    X = (np.sin(2 * np.pi * t / 12) + 0.4 * (t / T_M)
         + 0.3 * rng.standard_normal((T_M, H_G, W_G, C))).astype(np.float32)
    X[:, 0, 0, :] = np.nan
    months = np.array([f"{1990 + i // 12}-{i % 12 + 1:02d}" for i in range(T_M)])
    ridx = np.arange(6, T_M)
    rapid = np.stack([ridx.astype(float),
                      2.79 * rng.standard_normal(len(ridx))], 1)
    npz = os.path.join(tmp, "toy.npz")
    np.savez(npz, X=X, months=months, rapid=rapid,
             chan=np.array([f"c{i}" for i in range(C)]),
             norm=np.ones((C, 2), np.float32),
             lats=np.linspace(20, 40, H_G).astype(np.float32),
             lons=np.linspace(-60, -40, W_G).astype(np.float32))
    return npz


def main():
    tmp = tempfile.mkdtemp()
    npz = toy_npz(tmp)
    ok = 0

    # ---- case 1: no architecture, no resume -> refuse --------------------
    r = trainer(npz, os.path.join(tmp, "bare"), [])
    out = r.stdout + r.stderr
    if r.returncode == 0:
        print(out[-2500:])
        raise SystemExit("case 1 FAILED: the trainer accepted a dispatch with "
                         "no architecture — the pilot default is back")
    if "REFUSING to train: no architecture" not in out:
        print(out[-2500:])
        raise SystemExit("case 1 FAILED: refused, but not with the "
                         "architecture message — check what actually broke")
    for flag in ("--d-model", "--n-heads", "--d-dec"):
        if flag not in out:
            raise SystemExit(f"case 1 FAILED: refusal does not name {flag}, "
                             f"so a reader cannot fix it from the message")
    print("case 1 ok — refuses with no architecture, names the unset flags")
    ok += 1

    # ---- a parent checkpoint to resume from ------------------------------
    parent = os.path.join(tmp, "parent")
    r = trainer(npz, parent, ARCH)
    if r.returncode != 0:
        print(r.stdout[-2500:]); print(r.stderr[-2500:])
        raise SystemExit("setup FAILED: the explicit-architecture run did not "
                         "train — every later case depends on it")
    pck = os.path.join(parent, "pixelmae.pt")

    # ---- case 2: resume with no architecture -> adopt --------------------
    child = os.path.join(tmp, "child")
    r = trainer(npz, child, ["--resume", pck, "--steps", "50"])
    out = r.stdout + r.stderr
    if r.returncode != 0:
        print(out[-3000:])
        raise SystemExit("case 2 FAILED: resume without an architecture did "
                         "not train — this is #395's exact failure")
    if "architecture ADOPTED" not in out:
        print(out[-2500:])
        raise SystemExit("case 2 FAILED: it trained, but never said it "
                         "adopted the architecture — silent inheritance is "
                         "the thing being replaced")
    pa = torch.load(pck, map_location="cpu", weights_only=False)["args"]
    ca = torch.load(os.path.join(child, "pixelmae.pt"), map_location="cpu",
                    weights_only=False)["args"]
    for k in ("d_z", "patch", "d_model", "n_layers", "n_heads", "d_dec",
              "dec_layers"):
        if pa[k] != ca[k]:
            raise SystemExit(f"case 2 FAILED: {k} drifted across the resume "
                             f"({pa[k]} -> {ca[k]})")
    print("case 2 ok — adopts the checkpoint's architecture and says so")
    ok += 1

    # ---- case 3: resume with a contradicting architecture -> refuse ------
    r = trainer(npz, os.path.join(tmp, "clash"),
                ["--resume", pck, "--steps", "50", "--d-model", "32"])
    out = r.stdout + r.stderr
    if r.returncode == 0:
        print(out[-2500:])
        raise SystemExit("case 3 FAILED: the trainer accepted an architecture "
                         "that contradicts the checkpoint it is resuming")
    if "REFUSING to resume" not in out or "dispatch says 32" not in out:
        print(out[-2500:])
        raise SystemExit("case 3 FAILED: refused without naming both values, "
                         "which is what makes the message actionable")
    # torch's own wording is "size mismatch for <param>:" — matching the bare
    # phrase would hit the refusal message's OWN reference to #395's sixty
    # size mismatches, which is a test that fails on a correct implementation.
    if "size mismatch for" in out:
        raise SystemExit("case 3 FAILED: it reached load_state_dict. The "
                         "point is to refuse BEFORE building the model.")
    print("case 3 ok — refuses a contradicting architecture, before loading")
    ok += 1

    # ---- case 4: the collapse guard fires --------------------------------
    # --collapse-r 1.1 is above every possible correlation, so every probe is
    # a strike: the guard must fire on the second one. This tests the
    # MECHANISM (two strikes -> abort -> record), not the threshold.
    coll = os.path.join(tmp, "collapse")
    r = trainer(npz, coll, ARCH + ["--steps", "60", "--light-probe-every", "10",
                                   "--eval-every", "0", "--collapse-r", "1.1"])
    out = r.stdout + r.stderr
    if r.returncode == 0:
        print(out[-2500:])
        raise SystemExit("case 4 FAILED: the collapse guard never fired")
    if "ABORTING at step" not in out:
        print(out[-2500:])
        raise SystemExit("case 4 FAILED: it exited, but not through the "
                         "collapse guard")
    mp = os.path.join(coll, "metrics.jsonl")
    recs = [json.loads(l) for l in open(mp) if l.strip()]
    if not any("collapsed" in x for x in recs):
        raise SystemExit("case 4 FAILED: no {'collapsed'} record in "
                         "metrics.jsonl — the status page would show a run "
                         "that simply stopped, with no reason attached")
    # and it must NOT fire on a healthy run with the real default
    r = trainer(npz, os.path.join(tmp, "healthy"),
                ARCH + ["--steps", "60", "--light-probe-every", "10",
                        "--eval-every", "0"])
    if r.returncode != 0:
        print((r.stdout + r.stderr)[-2500:])
        raise SystemExit("case 4 FAILED: the guard aborted a HEALTHY run at "
                         "its default threshold — a false positive here costs "
                         "more than the failure it prevents")
    print("case 4 ok — fires on collapse, records it, spares a healthy run")
    ok += 1

    # ---- case 5: a DEFAULTED field must not read as a contradiction ------
    # --dec-layers still carries a real default (2, every pre-E-019b codec),
    # so "the dispatch did not ask" and "the dispatch asked for 2" are the
    # same value. Resuming an E-019b 3-hidden-layer decoder without naming
    # --dec-layers must ADOPT 3, not refuse a dispatch that expressed no
    # opinion. This is the false-refusal twin of case 3 and it shipped broken
    # for about ten minutes.
    deep = os.path.join(tmp, "deep")
    r = trainer(npz, deep, ARCH + ["--dec-layers", "3"])
    if r.returncode != 0:
        print((r.stdout + r.stderr)[-2500:])
        raise SystemExit("case 5 setup FAILED: --dec-layers 3 did not train")
    dck = os.path.join(deep, "pixelmae.pt")
    r = trainer(npz, os.path.join(tmp, "deepchild"),
                ["--resume", dck, "--steps", "50"])
    out = r.stdout + r.stderr
    if r.returncode != 0:
        print(out[-2500:])
        raise SystemExit("case 5 FAILED: resuming a 3-decoder-layer checkpoint "
                         "without naming --dec-layers was REFUSED. A field the "
                         "dispatch never set cannot be a contradiction.")
    if "dec_layers=3" not in out:
        print(out[-2500:])
        raise SystemExit("case 5 FAILED: it trained but did not report "
                         "adopting dec_layers=3 — check it did not silently "
                         "rebuild a 2-layer decoder")
    print("case 5 ok — a defaulted field adopts instead of refusing")
    ok += 1

    print(f"\nall {ok}/5 train.py configuration guards hold")


if __name__ == "__main__":
    main()

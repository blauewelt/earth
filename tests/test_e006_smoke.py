#!/usr/bin/env python3
"""Run the data-space objective end to end on a synthetic tensor.

ml/CLAUDE.md §4.8: exercise the code path on a toy before spending the
expensive resource. Any hour of GPU on a path that has never executed is a
coin flip, and E-006 touches the loss, the reference block, the logger and
the argument surface at once.

This builds an 8x10 ocean of 5 channels over 30 months, a 2-layer codec of a
few thousand parameters, and runs `train_joint.py --smoke --loss-mode data`
to completion on CPU in about a minute. It then asserts what the run WROTE,
not that it exited 0:

  · the per-channel variance line appeared, so the denominator came from the
    data rather than from a default;
  · every logged step carries space="data" and finite l_rec / l_fore /
    l_pers, because a NaN here would sail through training and surface as an
    unreadable results file (§5.22);
  · the two terms are on the SAME ORDER, which is the entire claim of the
    formulation — the old objective's failure was a forecast term ~20x
    smaller than the reconstruction term, so a sum was single-objective;
  · the guards refuse --ref-fore and --lam, at the inputs, where refusing is
    free.

    python3 tests/test_e006_smoke.py
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
sys.path.insert(0, ML)
from model import PixelMAE                                    # noqa: E402

T, H, W, C, D_Z = 30, 8, 10, 5, 4


def fixture(tmp):
    """A synthetic ocean with real structure: a slow trend, a seasonal cycle
    and noise, so the forecast term has something to be right and wrong
    about. A pure-noise tensor would make persistence unbeatable and the two
    terms uninformative — the toy has to be able to fail the way the real
    thing fails."""
    rng = np.random.default_rng(0)
    t = np.arange(T)[:, None, None, None]
    seas = np.sin(2 * np.pi * t / 12)
    trend = t / T
    X = (seas + 0.5 * trend + 0.3 * rng.standard_normal((T, H, W, C))
         ).astype(np.float32)
    # Static channels keep their own scale, which is exactly why the loss
    # divides per channel: channel 4 is 20x the others and would otherwise BE
    # the objective.
    X[..., 4] *= 20.0
    # A land mask, so the observed-entry bookkeeping is exercised rather than
    # assumed away by a fully observed toy.
    X[:, 0, 0, :] = np.nan
    months = np.array([f"{1990 + i // 12:04d}-{i % 12 + 1:02d}" for i in range(T)])
    np.savez(os.path.join(tmp, "toy.npz"), X=X, months=months,
             lats=np.linspace(30, 45, H).astype(np.float32),
             lons=np.linspace(-60, -40, W).astype(np.float32),
             chan=np.array([f"c{i}" for i in range(C)]))

    codec = PixelMAE(n_chan=C, d_model=16, n_heads=2, n_layers=2, d_z=D_Z,
                     d_dec=16, patch=1)
    ck = os.path.join(tmp, "toy_codec.pt")
    torch.save({"model": codec.state_dict(), "chan": [f"c{i}" for i in range(C)],
                "d_z": D_Z, "norm": None, "step": 0,
                "args": {"patch": 1, "d_model": 16, "n_layers": 2,
                         "n_heads": 2, "d_dec": 16}}, ck)
    return os.path.join(tmp, "toy.npz"), ck


def run(tmp, npz, ck, out, *extra):
    # `--holdout-lon=-45,-44`, joined with an EQUALS. Passed as two argv
    # entries argparse reads the leading minus as a new flag and dies with
    # "expected one argument" — which looks exactly like a missing value and
    # cost this test one debugging round.
    cmd = [sys.executable, os.path.join(ML, "train_joint.py"), "--smoke",
           "--data", npz, "--resume", "!" + ck, "--out", out,
           "--K", "4", "--holdout-years", "1991", "--holdout-lon=-45,-44",
           *extra]
    return subprocess.run(cmd, capture_output=True, text=True, timeout=900)


def main():
    with tempfile.TemporaryDirectory() as tmp:
        npz, ck = fixture(tmp)

        # ---- the guards, first: they must cost nothing to hit -------------
        for flag, val in (("--ref-fore", "0.44"), ("--lam", "0.5")):
            r = run(tmp, npz, ck, os.path.join(tmp, "guard"),
                    "--loss-mode", "data", flag, val)
            assert r.returncode != 0, f"{flag} was accepted by --loss-mode data"
            # MATCH THE REFUSAL, not the flag name. The first version asserted
            # the flag appeared anywhere in the output — and argparse prints
            # every flag in its usage block, so a run that died of a malformed
            # argument passed this check while the guard was never reached. A
            # test that a bad invocation satisfies is not testing the guard.
            want = f"takes no {flag}"
            assert want in r.stdout + r.stderr, \
                (f"expected the refusal {want!r}; got:\n"
                 f"{r.stdout[-600:]}{r.stderr[-600:]}")
            print(f"refused {flag} {val}: ok")

        # ---- the run ------------------------------------------------------
        out = os.path.join(tmp, "run")
        r = run(tmp, npz, ck, out, "--loss-mode", "data")
        if r.returncode != 0:
            print(r.stdout[-4000:]); print(r.stderr[-4000:])
            raise SystemExit("--loss-mode data did not complete on the toy")

        assert "per-channel variance for the data-space loss" in r.stdout, \
            "the variance was never computed from the data"
        assert "no forecast reference of any kind" in r.stdout, \
            "the data-space run still announced a reference"

        recs = [json.loads(l) for l in
                open(os.path.join(out, "metrics.jsonl")) if l.strip()]
        steps = [x for x in recs if x.get("space") == "data"]
        assert steps, "no step carried space=data — the logger forked"
        for x in steps:
            for k in ("l_rec", "l_fore", "l_pers", "loss"):
                v = x[k]
                assert v is not None and np.isfinite(v), \
                    f"{k} is {v} at step {x['joint_step']} — never write NaN"

        last = steps[-1]
        rec, fore = last["l_rec"], last["l_fore"]
        print(f"\nfinal step {last['joint_step']}: "
              f"l_rec {rec:.4f} · l_fore {fore:.4f} · l_pers {last['l_pers']:.4f} "
              f"· loss {last['loss']:.4f}")

        # THE CLAIM. Both terms are fractions of the same observed variance,
        # so they must be within an order of magnitude of each other — that
        # is what makes a plain sum a two-objective loss. The old z-space
        # arrangement put reconstruction at ~1.0 and forecast at ~0.3 through
        # different denominators, and 95.7% of every gradient went to
        # reconstruction (measured on #94).
        ratio = max(rec, fore) / max(min(rec, fore), 1e-9)
        assert ratio < 25, (f"the two terms are {ratio:.1f}x apart, so the sum "
                            f"is effectively single-objective — the "
                            f"formulation has not done its job")
        print(f"the terms are {ratio:.1f}x apart, so the sum sees both")

        # A codec came out the other end in the format every probe reads.
        blob = torch.load(os.path.join(out, "pixelmae.pt"), map_location="cpu",
                          weights_only=False)
        assert blob["args"]["joint"]["loss_mode"] == "data"
        assert set(("model", "chan", "d_z", "step")) <= set(blob)
        print("saved codec is in the standard format, tagged loss_mode=data")

        # ---- the OTHER modes still run ------------------------------------
        # E-006 refactored the logging into one shared writer, which is
        # exactly the kind of change that works for the new path and silently
        # breaks the old one. The reference modes are how every joint result
        # so far was produced; a comparison against them is the only way the
        # new objective can be judged, so they have to keep running.
        for mode in ("lse", "sum", "data-lse"):
            o = os.path.join(tmp, "mode_" + mode)
            r = run(tmp, npz, ck, o, "--loss-mode", mode)
            assert r.returncode == 0, \
                f"--loss-mode {mode} broke:\n{r.stdout[-1500:]}{r.stderr[-1500:]}"
            rs = [json.loads(l) for l in open(os.path.join(o, "metrics.jsonl"))
                  if l.strip()]
            st = [x for x in rs if "joint_step" in x and "l_rec" in x]
            assert st, f"--loss-mode {mode} logged no steps"
            want = "data" if mode.startswith("data") else "z"
            assert st[-1]["space"] == want, \
                f"--loss-mode {mode} logged space={st[-1]['space']}, expected {want}"
            print(f"  {mode:9s} ok · {len(st)} steps · space={st[-1]['space']}")

        print("\nE-006 runs end to end, and the modes it must be compared "
              "against still do too.")


if __name__ == "__main__":
    main()

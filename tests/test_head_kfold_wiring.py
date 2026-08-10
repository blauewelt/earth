#!/usr/bin/env python3
"""Execute temporal.py's head-k-fold block itself, not a paraphrase of it.

tests/test_head_kfold.py checks that `kfold_r` behaves. It does not check the
six lines in temporal.py that BUILD its arguments, and those lines have never
run — E-009's four arms are queued against them right now. ml/CLAUDE.md §4.8:
any hour of GPU on a path that has never executed is a coin flip.

So this pulls the block out of the source file by text and `exec`s it against
synthetic bindings shaped exactly like the real ones. Copying the six lines
into the test would defeat the purpose: the copy would pass while the file
was broken. If the block in temporal.py changes and stops working, this
fails; if the block is deleted, the extraction fails loudly rather than
passing vacuously.

The bindings mirror the real shapes, including the two that are easy to get
wrong:
  · `ri` indexes MONTHS of the tensor and is not contiguous — RAPID starts
    in 2004 and the tensor in 1982 — so `F[ri]`, `rv_des[ok]` and the year
    vector must line up under three different index spaces.
  · `ok = ridx >= K-1` drops the first K-1 months, so `rv_des[ok]` is
    shorter than `rv_des`.

    python3 tests/test_head_kfold_wiring.py
"""
import os
import re
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ML = os.path.join(HERE, "..", "ml")
sys.path.insert(0, ML)

SRC = open(os.path.join(ML, "temporal.py")).read()


def extract():
    """The try-block of eval 3b, dedented, as a runnable snippet."""
    m = re.search(r"\n    try:\n(        from probe_kfold import kfold_r\n.*?)"
                  r"\n    except Exception", SRC, re.S)
    if not m:
        raise SystemExit(
            "could not find the head-k-fold block in ml/temporal.py. If it was "
            "renamed, fix this extraction; if it was removed, E-009 has no "
            "instrument and that is the thing to look at.")
    body = "\n".join(l[8:] if l.startswith("        ") else l
                     for l in m.group(1).split("\n"))
    return body


def main():
    block = extract()
    print(f"extracted {len(block.splitlines())} lines from ml/temporal.py")

    # ---- bindings with the real shapes ------------------------------------
    T, D, K = 516, 192, 24                      # the tensor, the head, the window
    rng = np.random.default_rng(0)
    months = [f"{1982 + i // 12:04d}-{i % 12 + 1:02d}" for i in range(T)]
    # RAPID: monthly from 2004-04, 240 months, as [month_index, Sv] rows —
    # already sparse-and-finite, which is why no NaN handling appears in the
    # block. If that ever stops being true this test is where it shows.
    start = months.index("2004-04")
    ridx = np.arange(start, start + 240)
    rv_raw = rng.standard_normal(240) * 2.79
    moy = np.array([int(m[5:7]) - 1 for m in months])
    rmoy = moy[ridx]
    rclim = np.array([rv_raw[rmoy == m].mean() for m in range(12)])
    rv_des = rv_raw - rclim[rmoy]
    ok = ridx >= K - 1
    ri = ridx[ok]
    # The head's pooled hidden state, carrying a real signal in one direction
    # so the block has something to find.
    F = rng.standard_normal((T, D)).astype(np.float32)
    F[ri, 0] = 0.8 * rv_des[ok] + 0.6 * rng.standard_normal(len(ri))

    class Args:
        seed = 0

    env = {"np": np, "months": months, "ri": ri, "rv_des": rv_des, "ok": ok,
           "F": F, "a": Args(), "results": {}, "r_des": 0.317,
           "te": np.zeros(len(ri), bool), "print": print}
    env["te"][:36] = True                       # the single split's 36 months

    exec(compile(block, "<temporal.py eval 3b>", "exec"), env)

    r = env["results"].get("rapid_probe_kfold")
    assert r, "the block ran but wrote no rapid_probe_kfold key"
    print(f"\nwrote: r {r['r_kfold_deseas']} ci {r['ci95']} n {r['n']} "
          f"rmse {r['rmse_sv']} sigma {r['sigma_sv']}")

    # NEVER A NaN IN A RESULTS FILE (ml/CLAUDE.md §5.22). An all-NaN probe was
    # blamed on the probe twice before anyone looked at the mask.
    for k in ("r_kfold_deseas", "rmse_sv", "sigma_sv"):
        assert np.isfinite(r[k]), f"{k} is {r[k]}"
    for v in r["ci95"]:
        assert np.isfinite(v), f"ci bound is {v}"

    assert r["n"] == len(ri), f"scored {r['n']} months, expected {len(ri)}"
    assert r["r_kfold_deseas"] > 0.3, \
        f"a planted 0.8 signal read as {r['r_kfold_deseas']}"
    assert r["ci95"][0] < r["r_kfold_deseas"] < r["ci95"][1], \
        "the point estimate is outside its own interval"
    assert "features" in r and "note" in r, \
        "the record must say what it measured and how it differs from the codec probe"

    print(f"\n{len(ri)} months scored against the single split's 36. "
          f"The block in temporal.py runs, and E-009's arms will carry a "
          f"headline number.")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""The paired comparison must resolve a real gap AND refuse an imaginary one.

A test that only checks the first half is worse than no test: it would pass
for a script that always reports "significant", which is exactly the failure
mode that matters here, because the number under scrutiny (+0.034 for the
unpooled head over its raw-3x3 control) is one somebody wants to be real.

Both cases below use the same construction — two probes sharing a large
common error term, scored on the same year-blocked months — because that
shared error is the whole reason a paired test is the right instrument. The
only thing that changes between the cases is whether one probe carries more
signal than the other.

    python3 tests/test_paired_probe.py
"""
import json
import os
import subprocess
import sys
import tempfile

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
SCRIPT = os.path.join(HERE, "..", "scripts", "paired_probe.py")
YEARS = np.repeat(np.arange(2004, 2024), 12)          # 20 blocks, 240 months


def run(sig_a, sig_b, seed=0):
    """Two probes, identical folds, shared error; a carries sig_a of target."""
    rng = np.random.default_rng(seed)
    t = rng.normal(size=len(YEARS))
    shared = rng.normal(size=len(YEARS))
    a = sig_a * t + 0.5 * shared + 0.25 * rng.normal(size=len(YEARS))
    b = sig_b * t + 0.5 * shared + 0.25 * rng.normal(size=len(YEARS))
    d = tempfile.mkdtemp()
    paths = []
    for nm, p in (("a", a), ("b", b)):
        q = os.path.join(d, nm + ".json")
        json.dump({"probe": nm, "pred": [float(v) for v in p],
                   "target_sv": [float(v) for v in t],
                   "years": [int(v) for v in YEARS]}, open(q, "w"))
        paths.append(q)
    r = subprocess.run([sys.executable, SCRIPT, *paths],
                       capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    return r.stdout


def test_real_gap_is_resolved():
    out = run(0.62, 0.55)
    assert "excludes zero" in out, out
    print("real gap      : resolved            ✓")


def test_no_gap_is_refused():
    out = run(0.60, 0.60)
    assert "not distinguishable" in out, out
    assert "must not be quoted" in out, out
    print("no gap        : refused             ✓")


def test_mismatched_folds_are_refused():
    """Differencing two probes scored on different months is not a paired
    test, and silently doing it anyway would produce a confident number."""
    d = tempfile.mkdtemp()
    for nm, n in (("a", 240), ("b", 200)):
        json.dump({"probe": nm, "pred": [0.0] * n, "target_sv": [0.0] * n,
                   "years": [2004 + i // 12 for i in range(n)]},
                  open(os.path.join(d, nm + ".json"), "w"))
    r = subprocess.run([sys.executable, SCRIPT, os.path.join(d, "a.json"),
                        os.path.join(d, "b.json")],
                       capture_output=True, text=True)
    assert r.returncode != 0 and "same months" in (r.stdout + r.stderr)
    print("mismatch      : refused             ✓")


def test_old_result_files_are_refused():
    """Every probe_head.json written before 2026-08-10 lacks the per-month
    arrays. The script must say that instead of comparing summaries."""
    d = tempfile.mkdtemp()
    for nm in ("a", "b"):
        json.dump({"probe": nm, "r_kfold_deseas": 0.662},
                  open(os.path.join(d, nm + ".json"), "w"))
    r = subprocess.run([sys.executable, SCRIPT, os.path.join(d, "a.json"),
                        os.path.join(d, "b.json")],
                       capture_output=True, text=True)
    assert r.returncode != 0 and "predates" in (r.stdout + r.stderr)
    print("legacy file   : refused             ✓")


if __name__ == "__main__":
    test_real_gap_is_resolved()
    test_no_gap_is_refused()
    test_mismatched_folds_are_refused()
    test_old_result_files_are_refused()
    print("\nOK — the paired test resolves what is there and refuses what is not.")

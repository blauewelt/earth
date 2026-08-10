#!/usr/bin/env python3
"""Dying at 80% of the embedding must cost twenty minutes, not eighty.

Chris, 2026-08-10: *"don't keep all precomputed embeddings in ram, such that
if the process dies after 80% you can resume."*

The half-written memmap already held every completed month. What was missing
was a record of HOW MANY, so a restart could trust it — without that, the only
safe thing to do was start again, and #117 through #121 each paid the full
~95 minutes.

The dangerous direction is over-claiming. A marker that says "300 months done"
when only 290 were flushed makes the next run skip ten months that hold zeros:
real numbers, wrong months, no symptom, and a stage-2 model trained on a
codec that appears to have forgotten a year. So the marker is written AFTER
the flush, never before, and can therefore only under-claim — costing a few
minutes of recomputation, which is the cheap side of that trade.

    python3 tests/test_embed_resume.py
"""
import os
import sys
import tempfile

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "..", "ml"))
import temporal                                           # noqa: E402
from temporal import embed_everything                     # noqa: E402

T, H, W, C, DZ = 40, 4, 6, 3, 8
CRASH_AT = 24


class Stub(torch.nn.Module):
    """Deterministic, so "identical to an uninterrupted run" is exact."""
    patch = 1

    def __init__(self, dz):
        super().__init__()
        self.l = torch.nn.Linear(C, dz)
        torch.nn.init.constant_(self.l.weight, 0.37)
        torch.nn.init.constant_(self.l.bias, 0.1)

    def encode(self, v, o, mk, ctx):
        return self.l(v)


def fixture():
    X = torch.arange(T * H * W * C, dtype=torch.float32).reshape(T, H, W, C) / 997
    OBS = torch.ones(T, H, W, C, dtype=torch.bool)
    return (Stub(DZ), X, OBS, np.zeros((T, 2), np.float32),
            np.linspace(0, 10, H), np.linspace(-40, -10, W),
            np.array([0, 1, 2]), np.array([1, 2, 3]))


def run(cache_path, crash=None):
    m, X, OBS, ctx, lats, lons, ys, xs = fixture()
    real = temporal._mark_progress
    if crash is not None:
        def boom(tmp, out, done, *a):
            real(tmp, out, done, *a)          # flush + marker happen first
            if done >= crash:
                raise KeyboardInterrupt(f"simulated crash at month {done}")
        temporal._mark_progress = boom
    try:
        Z, _ = embed_everything(m, X, OBS, ctx, lats, lons, ys, xs, DZ,
                                cache_path=cache_path)
        return np.asarray(Z).copy()
    finally:
        temporal._mark_progress = real


def test_a_crash_leaves_a_resumable_partial_and_no_published_cache():
    with tempfile.TemporaryDirectory() as d:
        cp = os.path.join(d, "Z_actions_aaaaaaaaaa.npy")
        try:
            run(cp, crash=CRASH_AT)
        except KeyboardInterrupt:
            pass
        assert os.path.exists(cp + ".partial"), "the work so far must survive"
        assert not os.path.exists(cp), (
            "a half-built cache must NOT be published — the next run would "
            "take it as complete, which is the #10/#11 failure with a new face")
        import json
        mark = json.load(open(cp + ".partial.progress"))
        assert mark["months_done"] == CRASH_AT
        print(f"crash at {CRASH_AT}/{T} : partial kept, nothing published  ✓")


def test_the_resume_lands_bit_identical_to_an_uninterrupted_run():
    """The claim worth testing. A resume that merely *finishes* is not enough;
    it has to produce the same array, or the cache is a subtle corruption."""
    with tempfile.TemporaryDirectory() as d:
        ref = run(None)                                   # RAM, uninterrupted
        cp = os.path.join(d, "Z_actions_bbbbbbbbbb.npy")
        try:
            run(cp, crash=CRASH_AT)
        except KeyboardInterrupt:
            pass
        got = run(cp)                                     # resumes
        assert np.array_equal(got, ref.astype(got.dtype)), "resume diverged"
        assert not os.path.exists(cp + ".partial.progress"), (
            "the marker describes a .partial that no longer exists; leaving "
            "it would make the next run try to resume a missing file")
        print(f"resume from {CRASH_AT}   : bit-identical, marker cleaned  ✓")


def test_a_marker_for_a_different_shape_is_ignored():
    """A tensor rebuild changes T. Resuming across that would splice two
    different worlds together."""
    with tempfile.TemporaryDirectory() as d:
        cp = os.path.join(d, "Z_actions_cccccccccc.npy")
        try:
            run(cp, crash=CRASH_AT)
        except KeyboardInterrupt:
            pass
        import json
        p = cp + ".partial.progress"
        mark = json.load(open(p))
        mark["shape"] = [T + 7, 3, DZ]
        json.dump(mark, open(p, "w"))
        out, start = temporal._resume_partial(cp + ".partial", T, 3, DZ)
        assert out is None and start == 0
        print("shape mismatch  : ignored, rebuilt from scratch         ✓")


def test_no_marker_means_no_resume():
    """Without the marker the partial cannot be trusted at all: nothing says
    which months are real. Recomputing is the only safe move."""
    with tempfile.TemporaryDirectory() as d:
        cp = os.path.join(d, "Z_actions_dddddddddd.npy")
        try:
            run(cp, crash=CRASH_AT)
        except KeyboardInterrupt:
            pass
        os.remove(cp + ".partial.progress")
        out, start = temporal._resume_partial(cp + ".partial", T, 3, DZ)
        assert out is None and start == 0
        print("no marker       : refuses to guess, recomputes          ✓")


if __name__ == "__main__":
    test_a_crash_leaves_a_resumable_partial_and_no_published_cache()
    test_the_resume_lands_bit_identical_to_an_uninterrupted_run()
    test_a_marker_for_a_different_shape_is_ignored()
    test_no_marker_means_no_resume()
    print("\nOK — an interrupted embedding resumes where it stopped, and the "
          "marker can only ever under-claim.")

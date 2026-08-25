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
import json
import os
import shutil
import sys
import tempfile

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "ml"))
sys.path.insert(0, HERE)
import embed_cache_sync as sync                           # noqa: E402
import temporal                                           # noqa: E402
from temporal import embed_everything                     # noqa: E402
# The fake release lives beside the tests that exercise it most; importing it
# is cheaper than a second copy, and a second copy of a publishing rule is two
# chances to disagree about what the release holds.
from test_embed_cache_sync import (FakeRelease, quiet,     # noqa: E402
                                   with_release)

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


def test_the_publish_marker_can_only_UNDER_claim():
    """ml/CLAUDE.md §5.21, checked against the array rather than against the
    code that wrote it.

    `<cache>.progress` is what a publisher slices into release chunks, so a
    marker that over-claims by one row publishes a row of zeros as
    embeddings — real numbers, wrong months, no symptom, and now spread to
    every box that pulls the key instead of to one disk. Every marker written
    during a real pass is checked here: every row it claims must already hold
    computed values, and the byte offset it names must lie inside them.
    """
    with tempfile.TemporaryDirectory() as d:
        cp = os.path.join(d, "Z_actions_eeeeeeeeee.npy")
        seen, real = [], temporal._mark_progress

        def watched(tmp, out, done, T, P, dz, *a):
            real(tmp, out, done, T, P, dz, *a)
            mark = json.load(open(cp + ".progress"))
            z = np.load(tmp, mmap_mode="r")
            rows = mark["rows_flushed"]
            # 1. every row the marker claims has been computed. The stub
            #    embedding is nowhere zero, so an all-zero row is one nobody
            #    has written.
            assert rows <= done, (rows, done)
            assert np.asarray(z[:rows]).all(), (
                f"the marker claims {rows} rows and one of them is still the "
                f"memmap's zero fill")
            # 2. and the byte offset it names lies inside those rows.
            row_bytes = P * dz * np.dtype(temporal.CACHE_DTYPE).itemsize
            head = temporal._npy_header_bytes(tmp)
            assert mark["bytes_flushed"] == head + rows * row_bytes, mark
            assert mark["bytes_flushed"] <= head + done * row_bytes, mark
            assert (mark["T"], mark["P"], mark["d_z"]) == (T, P, dz), mark
            seen.append(rows)

        temporal._mark_progress = watched
        try:
            run(cp)
        finally:
            temporal._mark_progress = real
        assert seen == [8, 16, 24, 32, 40], seen
        # AND IT IS GONE AT THE END. `.done` and `.progress` are mutually
        # exclusive statements about the same file: one says every row is
        # real, the other says the first N are. A leftover `.progress` would
        # make the next pull fetch a prefix of a cache that is complete.
        assert not os.path.exists(cp + ".progress"), (
            "a finished cache must not advertise itself as a work in progress")
        assert os.path.exists(cp + ".done")
        print(f"under-claim: markers at {seen}, cleared on completion   ✓")


def test_a_crash_leaves_a_PUBLISHABLE_prefix_beside_the_partial():
    """The publisher reads `<cache>.progress`, not `<cache>.partial.progress`
    — it knows the cache by its published name and nothing about our temp
    file. Both are written by the same flush, so they cannot disagree."""
    with tempfile.TemporaryDirectory() as d:
        cp = os.path.join(d, "Z_actions_ffffffffff.npy")
        try:
            run(cp, crash=CRASH_AT)
        except KeyboardInterrupt:
            pass
        pub = json.load(open(cp + ".progress"))
        loc = json.load(open(cp + ".partial.progress"))
        assert pub["rows_flushed"] == loc["months_done"] == CRASH_AT
        assert not os.path.exists(cp + ".done"), (
            "an interrupted pass must never look complete")
        # And the publisher finds the bytes without being told where they are.
        assert sync.partial_source(cp) == cp + ".partial"
        assert sync.read_progress(cp)["rows_flushed"] == CRASH_AT
        print(f"publishable: {CRASH_AT}/{T} rows, both markers agree      ✓")


def test_a_published_partial_is_resumed_and_lands_bit_identical():
    """THE WHOLE FEATURE, end to end, on two boxes.

    Chris, 2026-08-25: *"A new job that needs the same embedding can choose to
    continue the computation (if 32/100 are already complete it will start
    with chunk 33)."* Box A dies at month 24 and publishes the whole chunks it
    finished. Box B has never seen this codec: it pulls the prefix, embeds the
    rest, and must end up with THE SAME ARRAY a box that did the whole thing
    itself would have — not merely a finished file. A resume that is only
    approximately right is a subtle corruption of every run that pulls it.
    """
    rel = FakeRelease()
    with tempfile.TemporaryDirectory() as d:
        ref = run(None)                                   # RAM, uninterrupted
        a = os.path.join(d, "boxA", "Z_actions_1111111111.npy")
        b = os.path.join(d, "boxB", "Z_actions_1111111111.npy")
        os.makedirs(os.path.dirname(a))
        os.makedirs(os.path.dirname(b))
        try:
            run(a, crash=CRASH_AT)
        except KeyboardInterrupt:
            pass
        restore = with_release(rel, a)
        try:
            # 40 rows x 3 x 8 fp16 = 48 B/row + 128 B header = 2048 B, so
            # CHUNK 512 makes four chunks and 24 flushed rows fill two.
            sync.CHUNK = 512
            rc, out = quiet(sync.push, "actions", "irrelevant.npz", T,
                            partial=True)
            assert rc == 0, out
            assert rel.manifest("Z_w_d.npy")["complete"] is False
            sync.cache_name = lambda run_, data: (b, "Z_w_d.npy", "w")
            rc, out = quiet(sync.pull, "actions", "irrelevant.npz", T)
            assert rc == 0, out
        finally:
            restore()
        rows = sync.read_progress(b)["rows_flushed"]
        assert 0 < rows < CRASH_AT, (rows, "18 whole rows of the 24 flushed")
        got = run(b)                                      # box B finishes it
        assert np.array_equal(got, ref.astype(got.dtype)), (
            "a resumed embedding must be bit-identical to one built in one go")
        assert os.path.exists(b + ".done"), "and it is publishable now"
        assert not os.path.exists(b + ".progress")
        assert not os.path.exists(b + ".partial")
        print(f"two boxes  : pulled {rows}/{T} rows, finished, "
              f"bit-identical  ✓")


if __name__ == "__main__":
    test_a_crash_leaves_a_resumable_partial_and_no_published_cache()
    test_the_resume_lands_bit_identical_to_an_uninterrupted_run()
    test_a_marker_for_a_different_shape_is_ignored()
    test_no_marker_means_no_resume()
    test_the_publish_marker_can_only_UNDER_claim()
    test_a_crash_leaves_a_PUBLISHABLE_prefix_beside_the_partial()
    test_a_published_partial_is_resumed_and_lands_bit_identical()
    print("\nOK — an interrupted embedding resumes where it stopped, the "
          "marker can only ever under-claim, and a prefix published mid-pass "
          "is resumed on another box to a bit-identical array.")

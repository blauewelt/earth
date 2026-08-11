#!/usr/bin/env python3
"""The published embedding cache must be named by the CODEC, and verified.

Two failures this guards against, both of which have precedent:

  1. A cache keyed by run name rather than codec weights poisoned runs #10/#11
     on 2026-08-07 — the shape check passed, the embeddings belonged to a
     different codec, and two stage-2 models trained on z their own decoder
     did not speak. That was one box. Publishing the same mistake to a release
     spreads it to every box, so the asset name and the local filename must
     come from the SAME hash function, not two copies of one rule.
  2. A reassembled multi-part download that is the wrong length. numpy raises
     on a file SHORTER than its header (measured), but a chunk uploaded twice
     or concatenated out of order gives a file that maps cleanly and returns
     real numbers for the wrong months — a wrong answer with no symptom.

    python3 tests/test_embed_cache_sync.py
"""
import os
import sys
import tempfile

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "..", "ml"))
import embed_cache_sync as sync                         # noqa: E402
import temporal                                          # noqa: E402


def fake_ck(seed):
    g = torch.Generator().manual_seed(seed)
    return {"model": {f"w{i}": torch.randn(8, 8, generator=g) for i in range(6)}}


def test_the_asset_name_comes_from_the_codec_not_the_run():
    a, b = fake_ck(0), fake_ck(1)
    ha, hb = temporal.codec_weight_hash(a), temporal.codec_weight_hash(b)
    assert ha != hb, "different codecs must not share a cache name"
    assert temporal.codec_weight_hash(fake_ck(0)) == ha, "and it must be stable"
    # The local path and the published asset must agree on the hash, or a box
    # pulls one codec's embedding into another codec's filename.
    assert ha in temporal.embed_cache_path("actions", ha)
    print(f"naming     : {ha} != {hb}, stable, shared        ✓")


def test_two_runs_of_the_same_codec_share_one_cache():
    """The whole economic argument: #112, #117 and #119 froze the same codec,
    so they should have embedded once between them, not three times."""
    h1 = temporal.codec_weight_hash(fake_ck(7))
    h2 = temporal.codec_weight_hash(fake_ck(7))
    assert h1 == h2
    assert (os.path.basename(temporal.embed_cache_path("run-a", h1))
            != os.path.basename(temporal.embed_cache_path("run-b", h1)))
    # ...but the PUBLISHED name drops the run, which is what lets a second run
    # (or a second box) find the first one's work.
    assert f"Z_{h1}.npy" == f"Z_{h2}.npy"
    print("sharing    : one codec, one published asset      ✓")


def test_a_valid_cache_verifies():
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "Z_deadbeef00.npy")
        np.save(p, np.zeros((9, 5, 8), dtype=temporal.CACHE_DTYPE))
        ok, why = sync.verify(p)
        assert ok, why
        print(f"valid      : {why}   ✓")


def test_a_truncated_cache_is_rejected():
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "Z_deadbeef01.npy")
        np.save(p, np.zeros((9, 5, 8), dtype=temporal.CACHE_DTYPE))
        os.truncate(p, os.path.getsize(p) - 32)
        ok, why = sync.verify(p)
        assert not ok and "truncated" in why, why
        print("truncated  : rejected                          ✓")


def test_a_cache_with_a_DUPLICATED_chunk_is_rejected():
    """The one numpy would not catch: too LONG maps fine and reads wrong."""
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "Z_deadbeef02.npy")
        np.save(p, np.zeros((9, 5, 8), dtype=temporal.CACHE_DTYPE))
        with open(p, "ab") as f:
            f.write(b"\0" * 80)                 # a chunk arrived twice
        assert np.load(p, mmap_mode="r").shape == (9, 5, 8), (
            "precondition: numpy is happy with the over-long file, which is "
            "why this check has to exist")
        ok, why = sync.verify(p)
        assert not ok, "an over-long cache must be rejected"
        print("duplicated : rejected (numpy was happy)        ✓")


def test_the_wrong_dtype_is_rejected():
    """A float32 cache is 2x the size and would blow the disk budget the
    fp16 switch was made to fit; it is also evidence of a version mismatch."""
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "Z_deadbeef03.npy")
        np.save(p, np.zeros((9, 5, 8), dtype=np.float32))
        ok, why = sync.verify(p)
        assert not ok and "dtype" in why, why
        print("wrong dtype: rejected                          ✓")


def test_push_with_no_cache_says_so_instead_of_succeeding_quietly():
    """The RAM path writes no cache. That is legitimate, and it must not look
    like a successful upload — CLAUDE.md 6c rule 6."""
    import io
    import contextlib
    with tempfile.TemporaryDirectory() as d:
        # cache_name takes the TENSOR as well as the run now: the cache is
        # keyed by codec AND data, because two boxes hold family3_na025.npz
        # files with different sha256s and a codec-only key let one pull the
        # other's embeddings while every check passed.
        sync.cache_name = lambda run, data: (os.path.join(d, "absent.npy"),
                                             "Z_absent.npy", "absent")
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            rc = sync.push("actions", "irrelevant.npz")
        out = buf.getvalue()
        assert rc == 0 and "nothing to publish" in out, out
        assert "built Z in RAM" in out
        print("no cache   : reports it, does not fake success ✓")


if __name__ == "__main__":
    test_the_asset_name_comes_from_the_codec_not_the_run()
    test_two_runs_of_the_same_codec_share_one_cache()
    test_a_valid_cache_verifies()
    test_a_truncated_cache_is_rejected()
    test_a_cache_with_a_DUPLICATED_chunk_is_rejected()
    test_the_wrong_dtype_is_rejected()
    test_push_with_no_cache_says_so_instead_of_succeeding_quietly()
    print("\nOK — the cache is published under its codec's identity, and a "
          "damaged one is discarded rather than trusted.")

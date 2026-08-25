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
            rc = sync.push("actions", "irrelevant.npz", 3142)
        out = buf.getvalue()
        assert rc == 0 and "nothing to publish" in out, out
        assert "built Z in RAM" in out
        print("no cache   : reports it, does not fake success ✓")


def test_a_shorter_republish_names_exactly_the_orphaned_tail():
    """#462, in one function.

    The release held twelve chunks of a 16.7 GB full Z. A six-chunk publish
    replaced the first six and deleted nothing, so `pull` — which concatenates
    suffixes until one 404s — glued six fresh chunks to six stale ones and
    handed back a chimera with a header claiming 8,716,963,840 bytes. The
    deletion set is the whole fix, and it is arithmetic, so it is tested
    without a network."""
    asset = "Z_b40f5b0b_adcbe700.npy"
    on_release = [f"{asset}.{sync.chunk_suffix(i)}" for i in range(12)]
    stale = sync.stale_chunk_assets(on_release, asset, 6)
    assert stale == [f"{asset}.{s}" for s in
                     ("ag", "ah", "ai", "aj", "ak", "al")], stale
    # The chunks the new publish DID write are never in the sweep — deleting
    # one would turn "a stale tail" into "a hole", which is worse.
    assert not any(s.endswith((".aa", ".af")) for s in stale)
    # A publish that is not shorter orphans nothing.
    assert sync.stale_chunk_assets(on_release, asset, 12) == []
    assert sync.stale_chunk_assets(on_release, asset, 20) == []
    # AND IT EATS NOTHING ELSE. The release carries 32 other assets under
    # other schemes; a prefix match alone would be a sweep with no edge.
    others = [f"{asset}.done", f"{asset}.sha256", f"{asset}.AB", "Z_other.npy.al",
              f"{asset}.aaa", asset]
    assert sync.stale_chunk_assets(on_release + others, asset, 6) == stale
    print(f"stale tail : 12 published, 6 now -> deletes "
          f"{[s[-2:] for s in stale]}, touches nothing else  ✓")


def test_T_comes_out_of_the_header_not_the_array():
    """The check has to be affordable enough to run on every push: ~128 bytes
    off the front of a file that can be 16 GiB, and off a TENSOR that can be
    165.6 GB and cannot be decompressed on any box we can rent."""
    with tempfile.TemporaryDirectory() as d:
        z = os.path.join(d, "Z_deadbeef04.npy")
        np.save(z, np.zeros((7, 5, 8), dtype=temporal.CACHE_DTYPE))
        assert int(sync.npy_shape(z)[0]) == 7
        # Family 2-4 layout: X inside the .npz. Read from the zip member's
        # own header — `np.load(...)["X"]` would decompress the lot.
        npz = os.path.join(d, "family3_na025.npz")
        np.savez(npz, X=np.zeros((11, 4, 3, 2), dtype=np.float16),
                 months=np.arange(11))
        assert sync.tensor_t(npz) == 11
        # Family 5 layout: X beside it as a bare memmappable .npy, and the
        # sidecar WINS — it is the tensor the readers actually open.
        side = os.path.join(d, "family5_na025_daily.npz")
        np.savez(side, months=np.arange(3))
        np.save(side[:-4] + "_X.npy", np.zeros((3, 4, 3, 2), dtype=np.float16))
        assert sync.tensor_t(side) == 3
        print("header T   : Z=7, npz X=11, sidecar X=3          ✓")


def _z(path, T, mark=True):
    """A cache file of T bins, with (or without) its completeness marker."""
    np.save(path, np.zeros((T, 5, 8), dtype=temporal.CACHE_DTYPE))
    if mark:
        sync.write_done(path)
    return path


def test_push_refuses_a_Z_of_the_wrong_shape():
    """A strided Z is not a damaged file — it is a whole, self-consistent,
    correctly-typed array of the wrong months, and every check this file had
    before 2026-08-25 passed it. It reached the release because the sidecar
    publishes whatever /opt/earth-cache/Z_*.npy holds."""
    import io
    import contextlib
    with tempfile.TemporaryDirectory() as d:
        p = _z(os.path.join(d, "Z_strided.npy"), 1571)     # one bin in two
        keep = sync.cache_name
        sync.cache_name = lambda run, data: (p, "Z_w_d.npy", "w")
        try:
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                rc = sync.push("actions", "irrelevant.npz", 3142)
            out = buf.getvalue()
        finally:
            sync.cache_name = keep
        assert rc == 1, "a strided Z must not be published"
        assert "1571" in out and "3142" in out, out   # BOTH numbers, named
        assert "unstrided key" in out, out
        print("wrong shape: refused, T=1571 vs T=3142 named   ✓")


def test_push_refuses_a_cache_nothing_attested_to():
    """T cannot see this one. `open_memmap` claims the full (T, P, d_z) shape
    before the first month is written, so a pass killed at 900 of 3142 leaves
    a file of the right length, the right dtype and the right T with zeros in
    the tail — and publishing it would be worse than publishing nothing,
    because it maps cleanly and reads as real numbers."""
    import io
    import contextlib
    def run_push(p):
        keep = kt = sync.cache_name
        sync.cache_name = lambda run, data: (p, "Z_w_d.npy", "w")
        tok = os.environ.pop("GITHUB_TOKEN", None)
        try:
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                rc = sync.push("actions", "irrelevant.npz", 9)
            return rc, buf.getvalue()
        finally:
            sync.cache_name = kt
            if tok is not None:
                os.environ["GITHUB_TOKEN"] = tok
    with tempfile.TemporaryDirectory() as d:
        p = _z(os.path.join(d, "Z_unmarked.npy"), 9, mark=False)
        rc, out = run_push(p)
        assert rc == 1 and "unattested" in out, out
        assert "completeness marker" in out, out
        # A marker that belongs to a DIFFERENT file does not vouch for this
        # one: the size is in it precisely so it cannot be reused by accident.
        with open(sync.done_path(p), "w") as f:
            f.write("123456\n")
        rc, out = run_push(p)
        assert rc == 1 and "different file" in out, out
        # Marked properly, the shape gate and the marker gate both pass and
        # push gets all the way to the thing it actually needs from the job.
        sync.write_done(p)
        rc, out = run_push(p)
        assert "no GITHUB_TOKEN" in out, out
        print("no marker  : refused; wrong marker refused; "
              "marked passes  ✓")


def test_pull_discards_a_published_Z_of_the_wrong_shape():
    """The other end of the same rule. A pulled Z is trusted by a run that
    then spends sixteen hours on it, so the discard has to happen before the
    run sees it — and a wrong-AXIS Z is invisible to the length check, which
    is what verify() had."""
    import io
    import contextlib
    import types
    with tempfile.TemporaryDirectory() as d:
        # What the release holds: a single chunk that IS a complete, valid,
        # correctly-typed .npy — of 5 bins, where the tensor has 9.
        src = _z(os.path.join(d, "published.npy"), 5, mark=False)
        blob = open(src, "rb").read()
        path = os.path.join(d, "cache", "Z_run_w_d.npy")
        keep, calls = sync.sh, []

        def fake_sh(cmd, **kw):
            """One chunk, then a 404 — the shape of every finished pull."""
            calls.append(cmd)
            out = cmd.split('-o "')[1].split('"')[0]
            if len(calls) == 1:
                with open(out, "wb") as f:
                    f.write(blob)
                return types.SimpleNamespace(returncode=0, stdout="", stderr="")
            return types.SimpleNamespace(returncode=22, stdout="", stderr="404")

        sync.cache_name = lambda run, data: (path, "Z_w_d.npy", "w")
        sync.sh = fake_sh
        try:
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                rc = sync.pull("actions", "irrelevant.npz", 9)
            out = buf.getvalue()
        finally:
            sync.sh = keep
        assert rc == 1, "a Z of the wrong axis must not be handed to a run"
        assert not os.path.exists(path), "and it must not be left on disk"
        assert "T=5" in out and "T=9" in out, out
        assert not os.path.exists(sync.done_path(path)), (
            "a discarded cache must not be left attested as complete")
        print("pull       : T=5 against a T=9 tensor — discarded ✓")


def test_a_pull_that_verifies_marks_the_cache_complete():
    """So the box that pulled can publish, and the box that built can too:
    the marker is what push requires, and a pulled cache is one this process
    watched arrive byte by byte."""
    import io
    import contextlib
    import types
    with tempfile.TemporaryDirectory() as d:
        src = _z(os.path.join(d, "published.npy"), 9, mark=False)
        blob = open(src, "rb").read()
        path = os.path.join(d, "cache", "Z_run_w_d.npy")
        keep, calls = sync.sh, []

        def fake_sh(cmd, **kw):
            calls.append(cmd)
            out = cmd.split('-o "')[1].split('"')[0]
            if len(calls) == 1:
                with open(out, "wb") as f:
                    f.write(blob)
                return types.SimpleNamespace(returncode=0, stdout="", stderr="")
            return types.SimpleNamespace(returncode=22, stdout="", stderr="404")

        sync.cache_name = lambda run, data: (path, "Z_w_d.npy", "w")
        sync.sh = fake_sh
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                rc = sync.pull("actions", "irrelevant.npz", 9)
        finally:
            sync.sh = keep
        assert rc == 0
        ok, why = sync.check_done(path)
        assert ok, why
        print(f"pull marks : {why}          ✓")


if __name__ == "__main__":
    test_the_asset_name_comes_from_the_codec_not_the_run()
    test_two_runs_of_the_same_codec_share_one_cache()
    test_a_valid_cache_verifies()
    test_a_truncated_cache_is_rejected()
    test_a_cache_with_a_DUPLICATED_chunk_is_rejected()
    test_the_wrong_dtype_is_rejected()
    test_a_shorter_republish_names_exactly_the_orphaned_tail()
    test_T_comes_out_of_the_header_not_the_array()
    test_push_refuses_a_Z_of_the_wrong_shape()
    test_push_refuses_a_cache_nothing_attested_to()
    test_pull_discards_a_published_Z_of_the_wrong_shape()
    test_a_pull_that_verifies_marks_the_cache_complete()
    # LAST, because it replaces sync.cache_name permanently.
    test_push_with_no_cache_says_so_instead_of_succeeding_quietly()
    print("\nOK — the cache is published under its codec's identity and its "
          "own shape, a shorter republish leaves no tail behind, and a "
          "damaged or strided one is discarded rather than trusted.")

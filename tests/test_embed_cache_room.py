#!/usr/bin/env python3
"""The embedding cache must prune or refuse BEFORE it opens the memmap.

`np.lib.format.open_memmap(mode="w+")` creates a sparse file. The 10.4 GiB it
claims is allocated page by page over the ~50 minutes the embedding takes, so
a box without room starts cleanly, runs for most of an hour, and dies on a
write with the cache nearly built. On the 50 GB Vast boxes a full disk does
not merely fail the job — it takes the runner offline, which turns one bad
run into a box that eats every job dispatched to it afterwards.

The guard in ml-train.yml pruned below 8 GB free. The single allocation it was
guarding against is 10.4 GiB. A threshold smaller than the thing it guards
against will always pass and the write will always fail; that is not a tuning
problem, it is the wrong shape of check. The requirement is computable —
T x P x d_z x 4 — so it is computed.

    python3 tests/test_embed_cache_room.py
"""
import os
import shutil
import sys
import tempfile
from collections import namedtuple

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "..", "ml"))
from temporal import RESERVE_BYTES, _make_room          # noqa: E402

GiB = 1 << 30
Usage = namedtuple("Usage", "total used free")


class FakeDisk:
    """A disk whose free space actually responds to the files we delete."""

    def __init__(self, free):
        self.free = free
        self.real = shutil.disk_usage

    def __enter__(self):
        shutil.disk_usage = lambda p: Usage(50 * GiB, 50 * GiB - self.free,
                                            self.free)
        return self

    def __exit__(self, *a):
        shutil.disk_usage = self.real


def stale(d, name, gib):
    p = os.path.join(d, name)
    with open(p, "wb") as f:
        f.write(b"\0" * 4096)
    os.truncate(p, int(gib * GiB))       # sparse: the SIZE is what we report
    return p


def test_plenty_of_room_touches_nothing():
    with tempfile.TemporaryDirectory() as d:
        keep = stale(d, "Z_other_aaaaaaaaaa.npy", 10.4)
        with FakeDisk(40 * GiB):
            _make_room(os.path.join(d, "Z_actions_bbbbbbbbbb.npy"), int(10.4 * GiB))
        assert os.path.exists(keep), "must not prune when there is room"
    print("room available : nothing pruned        ✓")


def test_stale_caches_are_pruned_until_it_fits():
    with tempfile.TemporaryDirectory() as d:
        old = stale(d, "Z_old_1111111111.npy", 10.4)
        part = stale(d, "Z_old_2222222222.npy.partial", 5.0)
        disk = FakeDisk(int(1.0 * GiB))
        with disk:
            # Deleting a file gives its bytes back — patch remove to model it.
            real_remove = os.remove

            def remove(p):
                disk.free += os.path.getsize(p)
                shutil.disk_usage = lambda _: Usage(50 * GiB, 0, disk.free)
                real_remove(p)
            os.remove = remove
            try:
                _make_room(os.path.join(d, "Z_actions_cccccccccc.npy"),
                           int(10.4 * GiB))
            finally:
                os.remove = real_remove
        assert not os.path.exists(old) and not os.path.exists(part), \
            "both stale caches should have been reclaimed"
    print("stale caches   : pruned to fit         ✓")


def test_it_refuses_rather_than_starting_a_doomed_write():
    """The whole point. Starting costs the job AND the runner, 40 minutes
    later; refusing costs the job now, with a number in the message."""
    with tempfile.TemporaryDirectory() as d:
        with FakeDisk(int(6.5 * GiB)):          # exactly #117's situation
            try:
                _make_room(os.path.join(d, "Z_actions_dddddddddd.npy"),
                           int(10.4 * GiB))
            except SystemExit as e:
                assert "not enough disk" in str(e) and "10.4" in str(e), str(e)
                print("no room        : refuses, with numbers  ✓")
                return
    raise AssertionError("must refuse when the cache cannot fit")


def test_the_run_s_own_cache_is_never_pruned():
    """Pruning the file we are about to write would be a very silly way to
    make room for it, and the glob matches it."""
    with tempfile.TemporaryDirectory() as d:
        mine = stale(d, "Z_actions_eeeeeeeeee.npy", 10.4)
        with FakeDisk(int(0.5 * GiB)):
            try:
                _make_room(mine, int(10.4 * GiB))
            except SystemExit:
                pass
        assert os.path.exists(mine), "must never prune its own target"
    print("own cache      : never pruned          ✓")


def test_the_reserve_is_bigger_than_a_runner_needs():
    assert RESERVE_BYTES >= 2 * GiB, (
        "the reserve exists so the runner can still write its logs and the "
        "job can still save a checkpoint after the cache is full")
    print(f"reserve        : {RESERVE_BYTES / GiB:.0f} GiB                  ✓")


if __name__ == "__main__":
    test_plenty_of_room_touches_nothing()
    test_stale_caches_are_pruned_until_it_fits()
    test_it_refuses_rather_than_starting_a_doomed_write()
    test_the_run_s_own_cache_is_never_pruned()
    test_the_reserve_is_bigger_than_a_runner_needs()
    print("\nOK — the cache makes room, or it stops before it can strand a box.")

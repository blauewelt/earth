"""write-verify-mark-delete, the retry queue, and the two-sighting rule."""
from __future__ import annotations

import datetime as dt
import json
import os

import pytest

from beam_import import sinks, tfrecord
from beam_import.example import make_example
from beam_import.hosts import TransientError


def _payloads(n=3):
    out, shas = [], []
    for i in range(n):
        values = bytes([i]) * 16
        out.append(make_example({"item_id": f"s/{i}", "values": values}))
        shas.append(sinks.sha256_bytes(values))
    return out, shas


def test_write_verify_mark(tmp_path):
    out = str(tmp_path / "out")
    payloads, shas = _payloads()
    marker = sinks.write_verify_mark(out, "src/item", "src", payloads, shas)
    assert marker["n_records"] == 3
    assert len(marker["sha256"]) == 64
    assert tfrecord.exists(marker["shard"])
    assert sinks.is_done(out, "src/item") == marker
    assert list(sinks.list_done(out)) == ["src/item"]


def test_a_wrong_sha_refuses_and_leaves_nothing(tmp_path):
    out = str(tmp_path / "out")
    payloads, shas = _payloads()
    shas[1] = "0" * 64                     # what a corrupted write would give
    with pytest.raises(TransientError, match="came back with sha256"):
        sinks.write_verify_mark(out, "src/item", "src", payloads, shas)
    assert sinks.is_done(out, "src/item") is None
    leftovers = tfrecord.list_uris(out)
    assert leftovers == [], f"a bad shard was left behind: {leftovers}"


def test_mark_is_written_after_the_shard_so_it_can_only_underclaim(tmp_path):
    """Kill the process after the shard is written and before the marker, and
    the next run must simply write it again. ml/CLAUDE.md §5.21: flush, THEN
    mark. The failure this prevents is an over-claiming marker, which makes
    the next run SKIP work that was never done."""
    out = str(tmp_path / "out")
    payloads, shas = _payloads()
    real_write_bytes = tfrecord.write_bytes

    def die_before_marking(uri, data):
        if uri.endswith(".done"):
            raise KeyboardInterrupt("killed between the rename and the marker")
        return real_write_bytes(uri, data)

    tfrecord.write_bytes = die_before_marking
    try:
        with pytest.raises(KeyboardInterrupt):
            sinks.write_verify_mark(out, "src/item", "src", payloads, shas)
    finally:
        tfrecord.write_bytes = real_write_bytes

    # The shard exists, the marker does not — so the item is NOT done ...
    assert tfrecord.exists(sinks.shard_uri(out, "src/item"))
    assert sinks.is_done(out, "src/item") is None
    assert "src/item" not in sinks.list_done(out)
    # ... and re-running writes it properly.
    marker = sinks.write_verify_mark(out, "src/item", "src", payloads, shas)
    assert sinks.is_done(out, "src/item")["sha256"] == marker["sha256"]


def test_fill_shards_sit_beside_the_parent(tmp_path):
    out = str(tmp_path / "out")
    payloads, shas = _payloads(1)
    sinks.write_verify_mark(out, "oisst/1993", "oisst", payloads, shas)
    sinks.write_verify_mark(out, "oisst/1993/1993-03-04", "oisst",
                            payloads, shas, fill="19930304",
                            fill_parent="oisst/1993")
    shards = tfrecord.list_uris(out, ".tfrecord")
    assert any(s.endswith("oisst/1993.tfrecord") for s in shards)
    assert any(s.endswith("oisst/1993.fill-19930304.tfrecord") for s in shards)
    # the day has its OWN marker, so it is separately resumable
    assert sinks.is_done(out, "oisst/1993/1993-03-04")


# --------------------------------------------------------------------------
# the retry queue
# --------------------------------------------------------------------------
def test_queue_append_read_rotate(tmp_path):
    state = str(tmp_path)
    sinks.enqueue(state, [{"item_id": "a"}, {"item_id": "b"}], "test")
    q = sinks.read_queue(sinks.queue_uri(state))
    assert sorted(x["item_id"] for x in q) == ["a", "b"]
    assert all(x["queued_reason"] == "test" for x in q)
    rotated = sinks.rotate_queue(state)
    assert rotated.endswith("retry_queue.1.jsonl")
    assert sinks.read_queue(sinks.queue_uri(state)) == []
    # nothing is deleted: the rounds stay on disk as the record
    assert os.path.exists(rotated)


def test_queue_dedupes_by_item_last_wins(tmp_path):
    state = str(tmp_path)
    sinks.enqueue(state, [{"item_id": "a", "n": 1}], "first")
    sinks.enqueue(state, [{"item_id": "a", "n": 2}], "second")
    q = sinks.read_queue(sinks.queue_uri(state))
    assert len(q) == 1 and q[0]["n"] == 2


# --------------------------------------------------------------------------
# absent needs two sightings, six hours apart
# --------------------------------------------------------------------------
def test_one_404_is_not_absent(tmp_path):
    state = str(tmp_path)
    is_absent, doc = sinks.record_not_found(
        state, "cable/1994", {"status": 404, "url": "http://x"})
    assert is_absent is False
    assert len(doc["sightings"]) == 1
    assert sinks.list_absent(state) == []


def test_a_second_404_inside_six_hours_is_still_one_sighting(tmp_path):
    state = str(tmp_path)
    sinks.record_not_found(state, "cable/1994", {"status": 404})
    is_absent, doc = sinks.record_not_found(state, "cable/1994",
                                            {"status": 404})
    assert is_absent is False
    assert len(doc["sightings"]) == 1
    assert sinks.list_absent(state) == []


def test_two_404s_six_hours_apart_are_absent(tmp_path):
    """Two runs, with real timestamps. The first sighting is back-dated the
    way a previous run's evidence file would be."""
    state = str(tmp_path)
    sinks.record_not_found(state, "cable/1994",
                           {"status": 404, "url": "http://x"})
    path = sinks.evidence_path(state, "cable/1994")
    doc = json.load(open(path, encoding="utf-8"))
    then = (dt.datetime.now(dt.timezone.utc) - dt.timedelta(hours=7)
            ).replace(microsecond=0).isoformat()
    doc["sightings"][0]["at"] = then
    json.dump(doc, open(path, "w", encoding="utf-8"))

    is_absent, doc = sinks.record_not_found(
        state, "cable/1994", {"status": 404, "url": "http://x"})
    assert is_absent is True
    assert len(doc["sightings"]) == 2
    assert doc["gap_hours"] >= 6.0
    listed = sinks.list_absent(state)
    assert [a["item_id"] for a in listed] == ["cable/1994"]
    # both responses are kept — that is the evidence a human reviews
    assert all("status" in s and "at" in s for s in listed[0]["sightings"])


def test_item_paths_are_filesystem_safe():
    assert sinks.item_rel_path("ncep/uflx.sfc.gauss/2001") == \
        "ncep/uflx.sfc.gauss/2001"
    assert "/" not in sinks.item_rel_path("a b").split("/")[0]

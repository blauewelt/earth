"""`make_example` / `parse_example` — and the same bytes TensorFlow makes."""
from __future__ import annotations

import numpy as np
import pytest

from beam_import.example import (make_example, one_int, one_str, parse_example,
                                 str_list)


SAMPLE = {
    "source": "oisst", "item_id": "oisst/1993", "date": "1993-01-01",
    "day_index": 4018, "grid": "oisst_center025",
    "lat0": -89.875, "lat_step": 0.25, "nlat": 720,
    "var_names": ["sst", "ice"], "var_units": ["Celsius", "percent"],
    "values": np.arange(6, dtype="<f4").tobytes(),
    "shape": [2, 1, 3], "source_bytes": 1714749,
}


def test_round_trip():
    got = parse_example(make_example(SAMPLE))
    assert one_str(got, "source") == "oisst"
    assert one_int(got, "day_index") == 4018
    assert str_list(got, "var_names") == ["sst", "ice"]
    assert [int(x) for x in got["shape"]] == [2, 1, 3]
    assert got["lat0"][0] == pytest.approx(-89.875)
    assert np.frombuffer(got["values"][0], dtype="<f4").tolist() == \
        [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]


def test_deterministic_bytes():
    """The same dict must always serialise to the same bytes — otherwise a
    shard's sha256 says nothing about its contents."""
    a = make_example(SAMPLE)
    b = make_example(dict(reversed(list(SAMPLE.items()))))
    assert a == b


def test_empty_and_single_values():
    got = parse_example(make_example({"empty": [], "one": b"x", "n": 0}))
    assert got["empty"] == []
    assert got["one"] == [b"x"]
    assert got["n"] == [0]


def test_tensorflow_parses_ours():
    tf = pytest.importorskip("tensorflow")
    ex = tf.train.Example()
    ex.ParseFromString(make_example(SAMPLE))
    f = ex.features.feature
    assert list(f["source"].bytes_list.value) == [b"oisst"]
    assert list(f["shape"].int64_list.value) == [2, 1, 3]
    assert list(f["lat0"].float_list.value) == pytest.approx([-89.875])
    assert list(f["var_names"].bytes_list.value) == [b"sst", b"ice"]


def test_we_parse_tensorflows():
    tf = pytest.importorskip("tensorflow")
    ex = tf.train.Example(features=tf.train.Features(feature={
        "a": tf.train.Feature(bytes_list=tf.train.BytesList(value=[b"x", b"y"])),
        "b": tf.train.Feature(int64_list=tf.train.Int64List(value=[1, 2, 3])),
        "c": tf.train.Feature(float_list=tf.train.FloatList(value=[1.5, 2.5])),
    }))
    got = parse_example(ex.SerializeToString())
    assert got == {"a": [b"x", b"y"], "b": [1, 2, 3], "c": [1.5, 2.5]}


def test_the_two_readers_agree_on_a_real_shard(tmp_path):
    """DESIGN §4 promises tf.data and the pure-Python reader see the same
    thing. This is that promise, measured."""
    tf = pytest.importorskip("tensorflow")
    from beam_import import tfrecord
    uri = str(tmp_path / "shard.tfrecord")
    tfrecord.write_records(uri, [make_example(SAMPLE)])

    ours = parse_example(tfrecord.read_records(uri)[0])
    spec = {"source": tf.io.FixedLenFeature([], tf.string),
            "day_index": tf.io.FixedLenFeature([], tf.int64),
            "shape": tf.io.FixedLenFeature([3], tf.int64),
            "values": tf.io.FixedLenFeature([], tf.string)}
    theirs = None
    for raw in tf.data.TFRecordDataset([uri]):
        theirs = tf.io.parse_single_example(raw, spec)
    assert bytes(theirs["source"].numpy()) == one_str(ours, "source").encode()
    assert int(theirs["day_index"].numpy()) == one_int(ours, "day_index")
    assert list(theirs["shape"].numpy()) == [int(x) for x in ours["shape"]]
    assert bytes(theirs["values"].numpy()) == ours["values"][0]

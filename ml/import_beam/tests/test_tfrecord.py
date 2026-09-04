"""The TFRecord frame: the CRCs must actually catch a flipped bit."""
from __future__ import annotations

import io
import os

import pytest

from beam_import import tfrecord
from beam_import.tfrecord import CorruptRecord


def test_masked_crc_matches_the_known_value():
    # The masked CRC-32C of the empty string and of b"foo" — the format's
    # own constants, so a change to the mask or the polynomial fails here
    # rather than in somebody's TensorFlow reader six months later.
    assert tfrecord.masked_crc32c(b"") == 0xA282EAD8
    assert tfrecord.masked_crc32c(b"foo") == 0xFEBE8A61


def test_round_trip(tmp_path):
    uri = str(tmp_path / "shard.tfrecord")
    payloads = [b"", b"a", b"hello world", os.urandom(5000)]
    n = tfrecord.write_records(uri, payloads)
    assert n == os.path.getsize(uri)
    assert tfrecord.read_records(uri) == payloads


def test_a_flipped_payload_byte_is_caught(tmp_path):
    uri = str(tmp_path / "shard.tfrecord")
    tfrecord.write_records(uri, [b"the quick brown fox"])
    raw = bytearray(open(uri, "rb").read())
    raw[20] ^= 0x01                                # one bit, in the payload
    open(uri, "wb").write(bytes(raw))
    with pytest.raises(CorruptRecord, match="payload CRC"):
        tfrecord.read_records(uri)


def test_a_flipped_length_byte_is_caught(tmp_path):
    uri = str(tmp_path / "shard.tfrecord")
    tfrecord.write_records(uri, [b"x" * 64])
    raw = bytearray(open(uri, "rb").read())
    raw[0] ^= 0x02                                 # in the length field
    open(uri, "wb").write(bytes(raw))
    with pytest.raises(CorruptRecord):
        tfrecord.read_records(uri)


def test_a_truncated_shard_is_caught(tmp_path):
    uri = str(tmp_path / "shard.tfrecord")
    tfrecord.write_records(uri, [b"y" * 200])
    raw = open(uri, "rb").read()
    open(uri, "wb").write(raw[:-10])               # the transfer stopped short
    with pytest.raises(CorruptRecord, match="truncated"):
        tfrecord.read_records(uri)


def test_tensorflow_reads_what_we_wrote(tmp_path):
    tf = pytest.importorskip("tensorflow")
    uri = str(tmp_path / "shard.tfrecord")
    payloads = [b"one", b"two", b"three"]
    tfrecord.write_records(uri, payloads)
    got = [bytes(x.numpy()) for x in tf.data.TFRecordDataset([uri])]
    assert got == payloads


def test_we_read_what_tensorflow_wrote(tmp_path):
    tf = pytest.importorskip("tensorflow")
    uri = str(tmp_path / "tf.tfrecord")
    with tf.io.TFRecordWriter(uri) as w:
        for p in (b"alpha", b"beta"):
            w.write(p)
    assert tfrecord.read_records(uri) == [b"alpha", b"beta"]


def test_join_and_list(tmp_path):
    root = str(tmp_path / "out")
    a = tfrecord.join(root, "src", "item.tfrecord")
    tfrecord.write_records(a, [b"z"])
    tfrecord.write_bytes(tfrecord.join(root, "src", "item.done"), b"{}")
    assert tfrecord.list_uris(root, ".tfrecord") == [a]
    assert len(tfrecord.list_uris(root)) == 2

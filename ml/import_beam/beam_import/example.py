"""`tf.train.Example` — built and parsed without importing TensorFlow.

One pair of functions is exposed and used everywhere:

    make_example(dict) -> bytes          a serialised tf.train.Example
    parse_example(bytes) -> dict         {name: [values]}

`tf.train.Example` is three nested protobuf messages and nothing else:

    Example  { Features features = 1; }
    Features { map<string, Feature> feature = 1; }
    Feature  { oneof { BytesList bytes_list = 1;
                       FloatList float_list = 2;
                       Int64List int64_list = 3; } }

A protobuf map field is repeated `MapEntry { key = 1; value = 2; }`, so the
whole encoder is varints and length-delimited blocks. It is written here
rather than taken from TensorFlow for two reasons: DESIGN §4 promises a
reader that needs no TensorFlow import at all, and `pip install
tensorflow-cpu` pulls protobuf past the version apache-beam pins (see
BUILD_NOTES). The output is byte-compatible either way, and
`tests/test_example.py` proves it against the real TensorFlow parser
whenever TensorFlow happens to be installed.

Python types map to Example types the obvious way:

    bytes / str / list of either      -> bytes_list  (str is encoded UTF-8)
    int / list of int                 -> int64_list
    float / list of float             -> float_list

Give a list explicitly when a single-element list is what you mean; a bare
scalar becomes a one-element list of the matching kind.
"""
from __future__ import annotations

import struct
from typing import Any, Dict, List, Union

Scalar = Union[bytes, str, int, float]
Value = Union[Scalar, List[Scalar]]


# ---------------------------------------------------------------- varints --
def _varint(n: int) -> bytes:
    out = bytearray()
    while True:
        b = n & 0x7F
        n >>= 7
        out.append(b | (0x80 if n else 0))
        if not n:
            return bytes(out)


def _tag(field: int, wire: int) -> bytes:
    return _varint((field << 3) | wire)


def _ld(field: int, payload: bytes) -> bytes:
    """A length-delimited field: tag, length, bytes."""
    return _tag(field, 2) + _varint(len(payload)) + payload


def _read_varint(buf: bytes, i: int):
    shift = result = 0
    while True:
        b = buf[i]
        i += 1
        result |= (b & 0x7F) << shift
        if not b & 0x80:
            return result, i
        shift += 7


# --------------------------------------------------------------- encoding --
def _feature(value: Value) -> bytes:
    """One `Feature` message."""
    items = value if isinstance(value, (list, tuple)) else [value]
    items = list(items)
    if not items:
        return _ld(1, b"")                       # an empty bytes_list
    first = items[0]
    if isinstance(first, (bytes, bytearray, str)):
        body = b"".join(
            _ld(1, v.encode("utf-8") if isinstance(v, str) else bytes(v))
            for v in items)
        return _ld(1, body)                       # bytes_list = 1
    if isinstance(first, bool) or isinstance(first, int):
        packed = b"".join(_varint(int(v)) for v in items)
        return _ld(3, _ld(1, packed))             # int64_list = 3, packed
    if isinstance(first, float):
        packed = struct.pack(f"<{len(items)}f", *[float(v) for v in items])
        return _ld(2, _ld(1, packed))             # float_list = 2, packed
    raise TypeError(f"cannot put {type(first).__name__} in a tf.train.Example")


def make_example(features: Dict[str, Value]) -> bytes:
    """Serialise a dict as a `tf.train.Example`.

    Keys are emitted in sorted order, so the same dict always produces the
    same bytes — which is what lets a shard's sha256 mean anything.
    """
    entries = b"".join(
        _ld(1, _ld(1, name.encode("utf-8")) + _ld(2, _feature(value)))
        for name, value in sorted(features.items()))
    return _ld(1, entries)                        # Example.features = 1


# --------------------------------------------------------------- decoding --
def _parse_feature(buf: bytes) -> List[Any]:
    i, out = 0, []
    while i < len(buf):
        key, i = _read_varint(buf, i)
        field, wire = key >> 3, key & 7
        if wire != 2:                             # nothing here is unpacked
            raise ValueError(f"unexpected wire type {wire} in Feature")
        size, i = _read_varint(buf, i)
        body, i = buf[i:i + size], i + size
        if field == 1:                            # bytes_list
            j = 0
            while j < len(body):
                k, j = _read_varint(body, j)
                n, j = _read_varint(body, j)
                out.append(bytes(body[j:j + n]))
                j += n
                del k
        elif field == 2:                          # float_list (packed)
            j = 0
            while j < len(body):
                k, j = _read_varint(body, j)
                n, j = _read_varint(body, j)
                out.extend(struct.unpack(f"<{n // 4}f", body[j:j + n]))
                j += n
                del k
        elif field == 3:                          # int64_list (packed)
            j = 0
            while j < len(body):
                k, j = _read_varint(body, j)
                n, j = _read_varint(body, j)
                end = j + n
                while j < end:
                    v, j = _read_varint(body, j)
                    out.append(v)
                del k
    return out


def parse_example(data: bytes) -> Dict[str, List[Any]]:
    """The inverse of `make_example`: {feature name: list of values}."""
    out: Dict[str, List[Any]] = {}
    i = 0
    while i < len(data):
        key, i = _read_varint(data, i)
        field, wire = key >> 3, key & 7
        if field != 1 or wire != 2:
            raise ValueError("not a tf.train.Example")
        size, i = _read_varint(data, i)
        features, j = data[i:i + size], 0
        i += size
        while j < len(features):                  # repeated MapEntry
            k, j = _read_varint(features, j)
            n, j = _read_varint(features, j)
            entry, e = features[j:j + n], 0
            j += n
            name, feat = None, []
            while e < len(entry):
                ek, e = _read_varint(entry, e)
                en, e = _read_varint(entry, e)
                blob, e = entry[e:e + en], e + en
                if ek >> 3 == 1:
                    name = blob.decode("utf-8")
                else:
                    feat = _parse_feature(blob)
            if name is not None:
                out[name] = feat
            del k
    return out


# ------------------------------------------------------------- convenience --
def one_bytes(rec: Dict[str, List[Any]], key: str, default=b"") -> bytes:
    v = rec.get(key) or [default]
    return v[0]


def one_str(rec: Dict[str, List[Any]], key: str, default: str = "") -> str:
    v = rec.get(key)
    return v[0].decode("utf-8") if v else default


def one_int(rec: Dict[str, List[Any]], key: str, default: int = 0) -> int:
    v = rec.get(key)
    return int(v[0]) if v else default


def one_float(rec: Dict[str, List[Any]], key: str, default: float = 0.0):
    v = rec.get(key)
    return float(v[0]) if v else default


def str_list(rec: Dict[str, List[Any]], key: str) -> List[str]:
    return [b.decode("utf-8") for b in rec.get(key, [])]

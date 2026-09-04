"""A pure-Python TFRecord reader and writer, over Beam's FileSystems.

TFRecord is four fields per record and nothing else:

    uint64 little-endian   length of the payload
    uint32 little-endian   masked CRC-32C of those 8 length bytes
    bytes                  the payload
    uint32 little-endian   masked CRC-32C of the payload

The mask is Google's standard rotation, `((crc >> 15) | (crc << 17)) +
0xa282ead8`, applied so that a CRC of a CRC is not degenerate.

Two reasons this file exists rather than `import tensorflow`:

  * DESIGN §4 promises a reader "with no TensorFlow import at all", because
    the people who will read these shards are as likely to be in JAX/NumPy as
    in TensorFlow, and a 600 MB dependency to read four fields is absurd.
  * `crcmod` is already an apache-beam dependency, so this adds nothing.

Everything goes through `apache_beam.io.filesystems.FileSystems`, so the same
code writes to a local directory, to `gs://`, or to anything else the runner
has a filesystem for.
"""
from __future__ import annotations

import struct
from typing import Iterable, Iterator, List, Optional

import crcmod.predefined

_crc32c = crcmod.predefined.mkPredefinedCrcFun("crc-32c")
_MASK_DELTA = 0xA282EAD8


def masked_crc32c(data: bytes) -> int:
    """The masked CRC-32C TFRecord stores. Same function TensorFlow uses."""
    crc = _crc32c(data) & 0xFFFFFFFF
    return (((crc >> 15) | (crc << 17)) + _MASK_DELTA) & 0xFFFFFFFF


class CorruptRecord(ValueError):
    """A record whose CRC does not match its bytes. Never retried silently."""


def encode_record(payload: bytes) -> bytes:
    """One complete TFRecord frame around `payload`."""
    length = struct.pack("<Q", len(payload))
    return (length + struct.pack("<I", masked_crc32c(length))
            + payload + struct.pack("<I", masked_crc32c(payload)))


def iter_records(stream) -> Iterator[bytes]:
    """Yield payloads from a binary stream, checking both CRCs.

    Raises CorruptRecord on a bad CRC and on a truncated frame — a short read
    is exactly the failure this whole package exists to catch early.
    """
    while True:
        header = stream.read(8)
        if not header:
            return
        if len(header) != 8:
            raise CorruptRecord("truncated length field")
        (length,) = struct.unpack("<Q", header)
        crc_len = stream.read(4)
        if len(crc_len) != 4:
            raise CorruptRecord("truncated length CRC")
        if struct.unpack("<I", crc_len)[0] != masked_crc32c(header):
            raise CorruptRecord("length CRC mismatch")
        payload = stream.read(length)
        if len(payload) != length:
            raise CorruptRecord(
                f"truncated payload: wanted {length}, got {len(payload)}")
        crc_data = stream.read(4)
        if len(crc_data) != 4:
            raise CorruptRecord("truncated payload CRC")
        if struct.unpack("<I", crc_data)[0] != masked_crc32c(payload):
            raise CorruptRecord("payload CRC mismatch")
        yield payload


# --------------------------------------------------------------------------
# through Beam's FileSystems, so local and gs:// are the same code
# --------------------------------------------------------------------------
def _fs():
    from apache_beam.io.filesystems import FileSystems
    return FileSystems


def write_records(uri: str, payloads: Iterable[bytes]) -> int:
    """Write a whole shard. Returns the number of bytes written."""
    total = 0
    with _fs().create(uri) as fh:
        for payload in payloads:
            frame = encode_record(payload)
            fh.write(frame)
            total += len(frame)
    return total


def read_records(uri: str) -> List[bytes]:
    """Read a whole shard, checking every CRC."""
    with _fs().open(uri) as fh:
        return list(iter_records(fh))


def exists(uri: str) -> bool:
    return _fs().exists(uri)


def rename(src: str, dst: str) -> None:
    _fs().rename([src], [dst])


def delete(uris: List[str]) -> None:
    """Delete files this package wrote. Missing files are not an error."""
    present = [u for u in uris if _fs().exists(u)]
    if present:
        _fs().delete(present)


def join(*parts: str) -> str:
    """Join URI parts. FileSystems.join needs a scheme it knows; plain string
    joining with a single slash is correct for every scheme we use."""
    out = parts[0].rstrip("/")
    for p in parts[1:]:
        out = out + "/" + str(p).strip("/")
    return out


def list_uris(prefix: str, suffix: Optional[str] = None) -> List[str]:
    """Every file under `prefix`, optionally filtered by suffix."""
    from apache_beam.io.filesystems import FileSystems
    pattern = prefix.rstrip("/") + "/**"
    try:
        matches = FileSystems.match([pattern])[0].metadata_list
    except Exception:                                          # noqa: BLE001
        return []
    out = [m.path for m in matches]
    if suffix:
        out = [p for p in out if p.endswith(suffix)]
    return sorted(out)


def read_bytes(uri: str) -> bytes:
    with _fs().open(uri) as fh:
        return fh.read()


def write_bytes(uri: str, data: bytes) -> None:
    with _fs().create(uri) as fh:
        fh.write(data)

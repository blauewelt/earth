"""Where Stage A puts things, and the durable state that says what is done.

Three jobs, all of them about never losing anything:

  * **write-verify-mark-delete.** A shard is written under a temporary name,
    read back (every record's CRC, and the sha256 of every record's `values`
    bytes against the copy still in memory), renamed into place, and only
    then does a `.done` marker appear. The marker can therefore only
    under-claim: a run killed between the rename and the marker simply
    rewrites the shard next time (ml/CLAUDE.md §5.21, flush THEN mark).

  * **the retry queue.** Anything not written and not proven absent is
    appended to `<state-dir>/retry_queue.jsonl`. A throttle changes WHEN an
    item is fetched, never WHETHER (DESIGN §2). There is no `failed` state.

  * **absent evidence.** A source is only believed when it says no TWICE, on
    runs at least six hours apart, and both answers are kept in
    `<state-dir>/absent_evidence/<item>.json`. One 404 is a bad afternoon.

Everything that touches the output goes through `tfrecord.py`, which goes
through Beam's FileSystems — so `--output /data/import` and
`--output gs://bucket/import` are the same code.
"""
from __future__ import annotations

import datetime as dt
import hashlib
import json
import os
import random
import re
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from . import tfrecord

# The minimum gap between the two 404s that make an item `absent`.
ABSENT_SECOND_SIGHTING_H = 6.0


def utcnow() -> str:
    return (dt.datetime.now(dt.timezone.utc)
            .replace(microsecond=0).isoformat())


def _parse_iso(s: str) -> dt.datetime:
    return dt.datetime.fromisoformat(s)


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


# --------------------------------------------------------------------------
# where a shard lives
# --------------------------------------------------------------------------
def item_rel_path(item_id: str) -> str:
    """`oisst/1993` -> `oisst/1993`; the item id IS the path under --output.

    Item ids are already `<source>/<key>`, and the key may itself contain a
    slash (`ncep/uflx.sfc.gauss/2001`), which becomes a subdirectory. Only
    characters no filesystem likes are replaced.
    """
    return "/".join(re.sub(r"[^A-Za-z0-9._+-]+", "_", part)
                    for part in item_id.split("/"))


def shard_uri(output: str, item_id: str,
              fill: Optional[str] = None) -> str:
    """The shard for one item. `fill` numbers a later top-up shard.

    A month whose missing days arrive in a later run does NOT rewrite the big
    shard — the days it got go into `<item>.fill-1.tfrecord` beside it, and
    Stage B reads every shard it finds. Rewriting a 3 GB shard to add one day
    is how a resumable pipeline stops being resumable.
    """
    base = tfrecord.join(output, item_rel_path(item_id))
    return f"{base}.fill-{fill}.tfrecord" if fill else f"{base}.tfrecord"


def done_uri(output: str, item_id: str) -> str:
    return tfrecord.join(output, item_rel_path(item_id)) + ".done"


def list_done(output: str) -> Dict[str, Dict[str, Any]]:
    """Every `.done` marker under --output, keyed by item_id.

    Read ONCE at the start of a run and handed to the pipeline as a side
    input; each lane also re-checks its own item's marker before fetching.
    """
    out: Dict[str, Dict[str, Any]] = {}
    for uri in tfrecord.list_uris(output, ".done"):
        try:
            doc = json.loads(tfrecord.read_bytes(uri).decode("utf-8"))
        except Exception:                                      # noqa: BLE001
            continue                    # a half-written marker is not a claim
        if doc.get("item_id"):
            out[doc["item_id"]] = doc
    return out


def is_done(output: str, item_id: str) -> Optional[Dict[str, Any]]:
    uri = done_uri(output, item_id)
    if not tfrecord.exists(uri):
        return None
    try:
        return json.loads(tfrecord.read_bytes(uri).decode("utf-8"))
    except Exception:                                          # noqa: BLE001
        return None


# --------------------------------------------------------------------------
# write · verify · mark
# --------------------------------------------------------------------------
def write_verify_mark(output: str, item_id: str, source: str,
                      payloads: Sequence[bytes],
                      values_sha: Sequence[str],
                      fill: Optional[str] = None,
                      fill_parent: Optional[str] = None,
                      missing_dates: Optional[List[str]] = None,
                      extra: Optional[Dict[str, Any]] = None,
                      note=None) -> Dict[str, Any]:
    """Write one shard, prove it reads back, then mark it done.

    `values_sha[i]` is the sha256 of record i's `values` bytes as they were
    in memory. After the write the shard is read back from the filesystem and
    each record's `values` is hashed again; a mismatch raises rather than
    marking anything.
    """
    from .example import one_bytes, parse_example
    from .hosts import TransientError

    # A fill shard belongs BESIDE its parent's, so Stage B picks it up by
    # globbing the parent's directory; its `.done` marker keeps the day
    # item's own name, so the day is separately resumable.
    final = shard_uri(output, fill_parent or item_id, fill)
    tmp = f"{final}.tmp-{os.getpid()}-{random.randint(0, 1 << 30):x}"

    n_bytes = tfrecord.write_records(tmp, payloads)
    try:
        back = tfrecord.read_records(tmp)          # CRCs checked in here
        if len(back) != len(payloads):
            raise TransientError(
                f"{item_id}: wrote {len(payloads)} records, read back "
                f"{len(back)}")
        for i, raw in enumerate(back):
            got = sha256_bytes(one_bytes(parse_example(raw), "values"))
            if got != values_sha[i]:
                raise TransientError(
                    f"{item_id}: record {i} came back with sha256 "
                    f"{got[:12]}, wrote {values_sha[i][:12]}")
    except Exception:
        tfrecord.delete([tmp])                     # never leave a bad shard
        raise

    tfrecord.rename(tmp, final)
    shard_sha = sha256_bytes(tfrecord.read_bytes(final))

    marker = {
        "item_id": item_id, "source": source, "shard": final,
        "bytes": n_bytes, "sha256": shard_sha, "n_records": len(payloads),
        "fetched_at": utcnow(),
        "missing_dates": sorted(missing_dates or []),
    }
    if fill:
        marker["fill"] = fill
    if extra:
        marker.update(extra)
    # The marker is written LAST and describes only what is already durable.
    tfrecord.write_bytes(done_uri(output, item_id),
                         json.dumps(marker, sort_keys=True).encode("utf-8"))
    if note:
        note(f"wrote {final} ({len(payloads)} records, {n_bytes} bytes, "
             f"sha {shard_sha[:12]})")
    return marker


# --------------------------------------------------------------------------
# the retry queue — the promise that nothing is dropped
# --------------------------------------------------------------------------
def queue_uri(state_dir: str) -> str:
    return os.path.join(state_dir, "retry_queue.jsonl")


def enqueue(state_dir: str, items: Iterable[Dict[str, Any]],
            reason: str) -> int:
    """Append work items to the retry queue. Returns how many were appended.

    One open-append-fsync-close per call, and one file for the whole run:
    lanes append rarely (a breaker trip, an exhausted ladder), so contention
    is not worth a lock, and an append under the POSIX buffer size is atomic.
    """
    items = list(items)
    if not items:
        return 0
    os.makedirs(state_dir, exist_ok=True)
    with open(queue_uri(state_dir), "a", encoding="utf-8") as fh:
        for it in items:
            rec = dict(it)
            rec["queued_reason"] = reason
            rec["queued_at"] = utcnow()
            fh.write(json.dumps(rec, sort_keys=True) + "\n")
        fh.flush()
        os.fsync(fh.fileno())
    return len(items)


def read_queue(path: str) -> List[Dict[str, Any]]:
    """The queue as a manifest. Later entries for the same item win."""
    out: Dict[str, Dict[str, Any]] = {}
    if not os.path.exists(path):
        return []
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue                      # a torn final line
            out[rec["item_id"]] = rec
    return list(out.values())


def rotate_queue(state_dir: str) -> Optional[str]:
    """Move retry_queue.jsonl aside so a new run can write a fresh one.

    Returns the rotated path, or None if there was no queue. Nothing is ever
    deleted: the rotated files are the record of how many rounds it took.
    """
    src = queue_uri(state_dir)
    if not os.path.exists(src):
        return None
    n = 1
    while os.path.exists(os.path.join(state_dir, f"retry_queue.{n}.jsonl")):
        n += 1
    dst = os.path.join(state_dir, f"retry_queue.{n}.jsonl")
    os.replace(src, dst)
    return dst


# --------------------------------------------------------------------------
# absent evidence — a source has to say no twice
# --------------------------------------------------------------------------
def evidence_path(state_dir: str, item_id: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9._+-]+", "_", item_id)
    return os.path.join(state_dir, "absent_evidence", f"{safe}.json")


def record_not_found(state_dir: str, item_id: str,
                     response: Dict[str, Any]) -> Tuple[bool, Dict[str, Any]]:
    """Record one 404/410. Returns (is_absent_now, the evidence document).

    `is_absent_now` is True only on the SECOND sighting, at least
    ABSENT_SECOND_SIGHTING_H hours after the first. Until then the item stays
    `queued` and a later run asks again.
    """
    path = evidence_path(state_dir, item_id)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    doc: Dict[str, Any] = {"item_id": item_id, "sightings": []}
    if os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as fh:
                doc = json.load(fh)
        except Exception:                                      # noqa: BLE001
            pass
    now = utcnow()
    response = dict(response)
    response["at"] = now
    sightings = doc.setdefault("sightings", [])

    first = sightings[0] if sightings else None
    gap_h = 0.0
    if first:
        gap_h = ((_parse_iso(now) - _parse_iso(first["at"])).total_seconds()
                 / 3600.0)
    # Only a sighting that is far enough from the first one counts as the
    # second; repeated 404s inside one run are one sighting, not two.
    if first is None or gap_h >= ABSENT_SECOND_SIGHTING_H:
        sightings.append(response)
    else:
        doc["last_seen_at"] = now
    doc["gap_hours"] = round(gap_h, 2)
    doc["absent"] = len(sightings) >= 2
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(doc, fh, sort_keys=True, indent=1)
    return bool(doc["absent"]), doc


def list_absent(state_dir: str) -> List[Dict[str, Any]]:
    """Every item the archive has now said no to twice, with the evidence."""
    root = os.path.join(state_dir, "absent_evidence")
    out = []
    if not os.path.isdir(root):
        return out
    for name in sorted(os.listdir(root)):
        try:
            with open(os.path.join(root, name), "r", encoding="utf-8") as fh:
                doc = json.load(fh)
        except Exception:                                      # noqa: BLE001
            continue
        if doc.get("absent"):
            out.append(doc)
    return out

#!/usr/bin/env python3
"""Where a store's data REALLY starts and ends, for the registries (E-085, D7).

PLAIN ENGLISH. Every store.json carries `date_range`: the window its build was
ASKED for (`--start` / `--end`), not the instants it holds. ERA5's stores were
asked for 1982-01-01 .. 2026-12-31 and their data ends 2026-06-30 18 UTC;
GLODAP was asked for .. 2026-12-31 and its last bottle is in 2023. Until now
the family-1 registries printed that window as `record_span` and family 10.2's
printed it as `date_range`, so a reader was told data exists where it does
not. Owner decision (Chris, 2026-10-06): the registry says where the data
ends.

This module computes, from the store's OWN published files, never from a
typed-in date:

  tier P (rows)  the first and last row of the time column (`time_s.npy`, or
                 schema 1's `time_days.npy`). Rows are sorted by (bin, time),
                 so row 0 is the earliest observation and row N-1 the latest:
                 two range reads of a few bytes each (local file or the Hub's
                 resolve/main URL). If the column cannot be read, the store's
                 own `bin_first` / `bin_last` (the first and last NON-EMPTY
                 bin, written from the data by the builder) give the span to
                 the bin, and `record_span_basis` says so.
  tier G (frames) the first and last PRESENT frame, from store.json alone:
                 each group's `bin_first` / `bin_last` are the first and last
                 bins of its shard index (bins holding at least one frame) and
                 `missing_frames` lists every absent (group, bin, frame) —
                 the same rule `src/f1data.js :: exactSpan` applies to the
                 shard index's frame bits.

The fields it returns, all of them span fields (`ml/registry_guard.py
--spans` allows exactly these to change):

  record_span          [first date, last date], inclusive, "YYYY-MM-DD" —
                       a frame counts to the last second it covers
                       (start + frame_seconds - 1), as exactSpan does.
  record_span_instants [first instant, last instant], ISO 8601 UTC: the
                       earliest and latest observation (tier P), or the start
                       of the first and of the last present frame (tier G).
  record_span_basis    how they were obtained (one of BASES below).
  requested_window     the store.json `date_range`, unchanged — nothing lost.
  date_range           = record_span (the old meaning moved to
                       `requested_window`); null for an empty store.

An EMPTY store (N = 0, or no frame present) gets record_span null and basis
"empty".
"""
import datetime as dt
import os
import struct
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

EPOCH = dt.datetime(1982, 1, 1, tzinfo=dt.timezone.utc)
BIN_SECONDS = 432000
TIME_COLUMN = {1: "time_days", 2: "time_s", 3: "time_s"}

SPAN_FIELDS = ("record_span", "record_span_instants", "record_span_basis",
               "requested_window", "date_range")

BASES = {
    "time_column": "first and last row of the store's own time column "
                   "(rows are sorted by bin, then time)",
    "bins": "the store's own first and last non-empty bin (the time column "
            "could not be read): exact to the 5-day bin, not to the instant",
    "frames": "first and last present frame, from store.json's groups and "
              "missing_frames (the shard index's own frame bits)",
    "frames+shard_index": "as `frames`, with the shard index's frame_mask "
                          "bits read for the groups whose first or last "
                          "index row holds no frame",
    "frame_bins": "first and last bin of the shard index (no present frame "
                  "could be named and the index was not readable): exact to "
                  "the bin",
    "empty": "the store holds no data",
}

SPAN_RULE = (
    "`record_span` (= `date_range`) is the first and last date that actually "
    "holds data, computed by ml/registry_spans.py from each store's own "
    "published files — tier P: the first and last row of its time column; "
    "tier G: the first and last present frame, a frame counting to the last "
    "second it covers. `record_span_instants` gives the same two ends to the "
    "second, `record_span_basis` how they were found, and `requested_window` "
    "the window the build was asked for (store.json's own `date_range`, "
    "which before 2026-10-06 was printed here as the span). Null for an empty "
    "store.")


def _iso(seconds):
    return (EPOCH + dt.timedelta(seconds=int(seconds))).strftime(
        "%Y-%m-%dT%H:%M:%SZ")


def _date(seconds):
    return (EPOCH + dt.timedelta(seconds=int(seconds))).strftime("%Y-%m-%d")


def _out(first_s, last_s, last_end_s, basis, meta):
    if first_s is None:
        return {"record_span": None, "record_span_instants": None,
                "record_span_basis": "empty",
                "requested_window": meta.get("date_range"),
                "date_range": None}
    rs = [_date(first_s), _date(last_end_s)]
    return {"record_span": rs,
            "record_span_instants": [_iso(first_s), _iso(last_s)],
            "record_span_basis": basis,
            "requested_window": meta.get("date_range"),
            "date_range": list(rs)}


# ================================================================ tier P ===
def parse_npy_header(head):
    """(dtype str, shape tuple, data offset) of a .npy file from its first
    bytes (versions 1, 2 and 3)."""
    import ast
    if head[:6] != b"\x93NUMPY":
        raise ValueError("not a .npy file")
    major = head[6]
    if major == 1:
        hl, off = struct.unpack("<H", head[8:10])[0], 10
    else:
        hl, off = struct.unpack("<I", head[8:12])[0], 12
    if len(head) < off + hl:
        raise ValueError(f"header needs {off + hl} bytes, have {len(head)}")
    d = ast.literal_eval(head[off:off + hl].decode("latin1"))
    if d.get("fortran_order"):
        raise ValueError("fortran-order column")
    return d["descr"], tuple(d["shape"]), off + hl


def time_column_ends(source, name):
    """(first, last) value of a 1-D .npy column, by two range reads on a
    `family1.sharded._Source` (a directory or a URL prefix)."""
    import numpy as np
    head = source.read_at(name, 0, 128)
    try:
        descr, shape, data = parse_npy_header(head)
    except ValueError:
        head = source.read_at(name, 0, 4096)
        descr, shape, data = parse_npy_header(head)
    dtype = np.dtype(descr)
    n = int(shape[0]) if shape else 0
    if n == 0:
        return None, None, 0
    a = np.frombuffer(source.read_at(name, data, dtype.itemsize), dtype)[0]
    b = np.frombuffer(source.read_at(name, data + (n - 1) * dtype.itemsize,
                                     dtype.itemsize), dtype)[0]
    return a, b, n


def tier_p_span(meta, source=None):
    """Span fields of a tier-P store from its store.json and (if given) a
    `_Source` on its directory, to read the time column's two ends."""
    n = meta.get("N")
    if n is not None and int(n) == 0:
        return _out(None, None, None, "empty", meta)
    sv = int(meta.get("schema_version", 1) or 1)
    col = TIME_COLUMN.get(sv, "time_s")
    if source is not None:
        try:
            a, b, rows = time_column_ends(source, f"{col}.npy")
            if rows == 0:
                return _out(None, None, None, "empty", meta)
            if col == "time_days":
                # float32 days: resolves 21 s in 1993, 84 s in 2024; floor to
                # the second so the instant is never later than the row
                fa, fb = (int(float(a) * 86400.0 // 1),
                          int(float(b) * 86400.0 // 1))
            else:
                fa, fb = int(a), int(b)
            if fb < fa:
                raise ValueError(f"{col}: last row {fb} before first {fa}")
            return _out(fa, fb, fb, "time_column", meta)
        except Exception as e:  # noqa: BLE001 — fall back to the bins, say so
            sys.stderr.write(f"  span: {col}.npy unreadable ({e}); using the "
                             f"store's bins\n")
    b0, b1 = meta.get("bin_first"), meta.get("bin_last")
    if b0 is None or b1 is None:
        return _out(None, None, None, "empty", meta)
    lo, hi = int(b0) * BIN_SECONDS, (int(b1) + 1) * BIN_SECONDS - 1
    return _out(lo, hi, hi, "bins", meta)


# ================================================================ tier G ===
def _index_ends(source, rel, F):
    """(first bin, first frame, last bin, last frame) present in a group's
    shard index, from its per-row `frame_mask` bits — the same bits
    src/f1data.js :: exactSpan reads. None when no frame is present."""
    from family1 import sharded as sh
    arr = sh.load_shard_index(source.get(rel))
    if not len(arr):
        return None
    arr = arr[arr["bin"].argsort(kind="stable")]
    rows = [(int(r["bin"]), int(r["frame_mask"])) for r in arr
            if int(r["frame_mask"])]
    if not rows:
        return None
    b0, m0 = rows[0]
    b1, m1 = rows[-1]
    f0 = next(f for f in range(F) if (m0 >> f) & 1)
    f1 = next(f for f in range(F - 1, -1, -1) if (m1 >> f) & 1)
    return b0, f0, b1, f1


def tier_g_span(meta, source=None):
    """Span fields of a sharded tier-G store.

    Fast path, from store.json alone: a group's `bin_first` / `bin_last` are
    the first and last rows of its shard index, and every frame of an index
    row is either present or listed by name in `missing_frames` (the builder
    refuses an absent frame without a reason), so the first frame of
    `bin_first` not listed missing IS the first present frame, and likewise
    at `bin_last`. Only when every frame of such a row is listed missing
    (a group whose index carries empty bins) is the group's shard index read
    through `source` — its `frame_mask` bits name the frame exactly. With no
    source the span is stated to the bin and the basis says so."""
    groups = meta.get("groups") or {}
    missing = set()
    for m in meta.get("missing_frames") or []:
        missing.add((m.get("group"), int(m["bin"]), int(m["frame"])))
    F0 = int(meta.get("frames_per_bin", 1) or 1)
    fs0 = int(meta.get("frame_seconds", BIN_SECONDS) or BIN_SECONDS)
    first = last = last_end = None
    used = set()
    for name, g in sorted(groups.items()):
        if not g.get("frames_present") or g.get("bin_first") is None \
                or g.get("bin_last") is None:
            continue
        F = int(g.get("frames_per_bin", F0) or F0)
        fs = int(g.get("frame_seconds", fs0) or fs0)
        b0, b1 = int(g["bin_first"]), int(g["bin_last"])
        f0 = next((f for f in range(F) if (name, b0, f) not in missing), None)
        f1 = next((f for f in range(F - 1, -1, -1)
                   if (name, b1, f) not in missing), None)
        ends = None
        if (f0 is None or f1 is None) and source is not None:
            try:
                ends = _index_ends(source, g.get("shard_index")
                                   or f"{name}/shard_index.npy", F)
            except Exception as e:  # noqa: BLE001 — fall back, say so
                sys.stderr.write(f"  span: {name} shard index unreadable "
                                 f"({e}); stating it to the bin\n")
            if ends is None:
                pass
            else:
                b0, f0, b1, f1 = ends
                used.add("shard_index")
        if f0 is None or f1 is None:
            used.add("frame_bins")
            s0 = b0 * BIN_SECONDS + (f0 or 0) * fs if f0 is not None \
                else b0 * BIN_SECONDS
            s1 = b1 * BIN_SECONDS + f1 * fs if f1 is not None \
                else b1 * BIN_SECONDS
            e1 = s1 + fs - 1 if f1 is not None \
                else (b1 + 1) * BIN_SECONDS - 1
        else:
            used.add("frames")
            s0 = b0 * BIN_SECONDS + f0 * fs
            s1 = b1 * BIN_SECONDS + f1 * fs
            e1 = s1 + fs - 1
        first = s0 if first is None else min(first, s0)
        if last is None or s1 > last:
            last = s1
        last_end = e1 if last_end is None else max(last_end, e1)
    if first is None:
        return _out(None, None, None, "empty", meta)
    basis = ("frame_bins" if "frame_bins" in used else
             "frames+shard_index" if "shard_index" in used else "frames")
    return _out(first, last, last_end, basis, meta)


def span_fields(meta, tier, source=None):
    """The span block for one store: `tier` is "P" or "G" (a catalogue's
    layout is tier P). `source` is a `_Source` on the store's directory."""
    if tier == "G":
        return tier_g_span(meta, source)
    return tier_p_span(meta, source)


def hub_source(repo, prefix):
    """A `_Source` on a public Hub store directory."""
    from family1 import sharded as sh
    return sh._Source(f"https://huggingface.co/datasets/{repo}/resolve/main/"
                      f"{prefix}")


def local_source(path):
    from family1 import sharded as sh
    return sh._Source(path)

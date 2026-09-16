#!/usr/bin/env python3
"""Write `data/fishing_index.json` — the small in-repo index the globe reads
before it touches the Hugging Face Hub for the fishing-effort raster.

WHAT THE FILE BEHIND IT IS, in one sentence: `fishing_grid_monthly_025.npy` is
Global Fishing Watch's AIS apparent fishing effort — the hours each vessel
broadcast in each 0.1° cell each day, and the part of them a neural network
classed as fishing — summed onto the family-7 0.25° grid, one frame per month
from 2012-01 to 2024-12, written by
`ml/build_family10_stores.py --store fishing --stage grid` out of the
`fishing` tier-P store it is the gridded view of (E-081 §3).

THE LAYOUT IS THE WHOLE POINT. The array is `[months, 721, 1440, 2]` in C
order, so ONE MONTH OF BOTH CHANNELS IS ONE CONTIGUOUS RANGE READ:

    offset = header_len + month_index * slab_bytes
    length = slab_bytes            # 721 * 1440 * 2 * itemsize

THE DTYPE IS float32 AND IS READ FROM THIS FILE, NEVER ASSUMED. E-081 §3
budgeted float16 (a 4.15 MB slab); measured 2026-09-16 on the real years, the
largest 0.25° cell-month sum is 29,460 vessel-hours in 2012 and **595,726 in
2024**, against float16's largest finite value of 65,504 — so a float16 grid
would carry +inf in exactly the cells where the fleet is. float32 makes the
slab 8.3 MB, still under family 7's 14.5 MB frame, and a consumer that takes
`dtype`, `itemsize` and `slab_bytes` from here is unaffected either way (and
gets a native `Float32Array` instead of a hand-rolled float16 decode).

NOT ONE NUMBER OF THAT ARITHMETIC IS WRITTEN IN `src/app.js` (root CLAUDE.md
§3, the family-7 rule): the header length is parsed out of the real published
header, the shape and dtype are read off it, the slab size is computed here,
and the month list, the grid geometry, the channel labels and units and the
measured CORS headers all travel in this file beside them. A consumer that
reads the index needs no constant of its own.

THE GEOMETRY, stated because a picture cannot be checked by eye: latitude
index 0 is −90 and longitude index 0 is −180, the step is 0.25°, and a row's
CELL CENTRE is binned with `iy = clip(floor((lat + 90) / 0.25), 0, 720)` and
`ix = clip(floor((lon + 180) / 0.25), 0, 1439)`. Zero is a REAL value — no
vessel broadcast there that month — and is stored as 0, never NaN.

THE CHECKS THIS WILL NOT SKIP (ml/CLAUDE.md §0.2):

  1. **The header is read from the PUBLISHED bytes**, by a ranged GET, not
     from the local file that was uploaded. The number the browser needs is
     the number the Hub serves.
  2. **CORS is MEASURED**, anonymously, with `Origin: https://blauewelt.github.io`
     and a `Range` header, and the answer must be a 206 carrying an
     `access-control-allow-origin` — exactly as `ml/publish_family7_index.py`
     does it. The layer's whole design is one range read per frame.
  3. **The header parse is checked against the file size**: `header_len +
     prod(shape) * itemsize` must equal the published byte count, or the
     arithmetic the browser is about to do is wrong.
  4. **The grid's totals are checked against the STORE's**, which the grid
     stage already asserted per month; the index carries both so a reader can
     see the agreement rather than trust it.

Run (after the build job has published):
    python3 ml/publish_fishing_index.py
Local, no network (what the publish stage falls back to and the tests use):
    python3 ml/publish_fishing_index.py --local <work>/fishing/fishing_grid
Fixture (a 5°-decimated copy of a smoke grid, for the browser tests):
    python3 ml/publish_fishing_index.py --fixture <work>/fishing
"""
import argparse
import ast
import datetime as _dt
import hashlib
import json
import os
import sys
import time
import urllib.error
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)

REPO_ID = "chfrank/earth-tensors"
ORIGIN = "https://blauewelt.github.io"
INDEX = os.path.join(ROOT, "data", "fishing_index.json")
FIXTURE_DIR = os.path.join(ROOT, "data", "fishing", "fixture")
CORS_PROBE_BYTES = 4096
# The fixture's decimation, the family-7 fixture's rule: every 20th row and
# column of the 0.25° grid lands exactly on a 5° point grid (37 x 72) that is
# still south-first and still wraps at the dateline. One real month is 8.3 MB
# and git is not where that belongs.
FIX_STRIDE = 20

# Plain-English label and unit per channel, beside the bytes rather than in
# `src/app.js` — the producer's vocabulary travels with the producer's data
# (root CLAUDE.md §2.3), so a re-bake with a channel renamed cannot leave the
# page describing a channel that is no longer there.
CHANNELS = {
    "fishing_hours": ("Apparent fishing hours", "vessel-hours per month",
                      "seq"),
    "hours": ("Hours broadcasting on AIS", "vessel-hours per month", "seq"),
}


def describe(name):
    return CHANNELS.get(name, (name, "", "seq"))


def now_utc():
    return _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def sha256_file(path, buf=1 << 22):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for blk in iter(lambda: fh.read(buf), b""):
            h.update(blk)
    return h.hexdigest()


def parse_npy_header(head_bytes):
    """(header_len, shape, dtype_str, fortran_order) from a `.npy`'s prefix.

    Format 1.0/2.0/3.0: magic, version, a 2- or 4-byte little-endian header
    length, then that many bytes of a Python literal dict. Parsed with
    `ast.literal_eval`, never `eval` — the bytes are remote.
    """
    if head_bytes[:6] != b"\x93NUMPY":
        raise ValueError("not a .npy file (bad magic)")
    major = head_bytes[6]
    if major == 1:
        n = int.from_bytes(head_bytes[8:10], "little")
        off = 10
    else:
        n = int.from_bytes(head_bytes[8:12], "little")
        off = 12
    if len(head_bytes) < off + n:
        raise ValueError("header probe too short — read more bytes")
    d = ast.literal_eval(head_bytes[off:off + n].decode("latin1"))
    return off + n, list(d["shape"]), d["descr"], bool(d["fortran_order"])


def resolve_url(repo, prefix, name):
    return (f"https://huggingface.co/datasets/{repo}/resolve/main/"
            f"{prefix}/{name}")


def measure_cors(url, origin=ORIGIN, tries=4):
    """What a BROWSER on our origin actually gets back.

    An anonymous ranged GET carrying `Origin`, following the Hub's redirect to
    its CDN, reporting the headers of whatever finally answers. Anonymous
    because that is what the page is: a measurement made with a token measures
    something else. Identical in shape to
    `ml/publish_family7_index.py::measure_cors`.
    """
    for attempt in range(tries):
        try:
            req = urllib.request.Request(
                url, headers={"Origin": origin,
                              "Range": f"bytes=0-{CORS_PROBE_BYTES - 1}"})
            with urllib.request.urlopen(req, timeout=90) as r:
                h = {k.lower(): v for k, v in r.headers.items()}
                body = r.read()
                return dict(
                    status=r.status, final_url=r.url, bytes=len(body),
                    access_control_allow_origin=h.get(
                        "access-control-allow-origin"),
                    access_control_expose_headers=h.get(
                        "access-control-expose-headers"),
                    accept_ranges=h.get("accept-ranges"),
                    content_range=h.get("content-range"),
                    head=body)
        except urllib.error.HTTPError as e:
            if e.code != 429 or attempt == tries - 1:
                raise
            wait = 30 * (attempt + 1)
            print(f"    HTTP 429 from the Hub — waiting {wait}s "
                  f"(attempt {attempt + 2}/{tries})", flush=True)
            time.sleep(wait)


def write_json(path, obj, label=None):
    text = json.dumps(obj, sort_keys=True, separators=(",", ":")) + "\n"
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(text)
    print(f"wrote {label or path} ({len(text) / 1e3:.1f} kB)")
    return path


def _slab_bytes(shape, itemsize):
    n = itemsize
    for d in shape[1:]:
        n *= int(d)
    return int(n)


def index_block(gm, sm, url, header_len, shape, dtype, nbytes, sha, cors,
                measured_from):
    """The index itself. Everything a consumer needs and nothing it must
    recompute."""
    itemsize = int(np.dtype(dtype).itemsize) if _HAVE_NP else int(dtype[-1])
    slab = _slab_bytes(shape, itemsize)
    chans = list(gm.get("channels") or ["fishing_hours", "hours"])
    if len(chans) != int(shape[-1]):
        raise SystemExit(f"{len(chans)} channel name(s) for {shape[-1]} "
                         f"channels — the grid and its manifest disagree")
    labels, units, signs = {}, {}, {}
    for c in chans:
        labels[c], units[c], signs[c] = describe(c)
    months = list(gm.get("months") or [])
    if int(shape[0]) != len(months):
        raise SystemExit(f"the array has {shape[0]} frame(s) and the manifest "
                         f"names {len(months)} month(s) — a consumer would "
                         f"read the wrong month")
    return dict(
        _source="ml/publish_fishing_index.py — do not hand-edit",
        generated_utc=now_utc(),
        fixture=False,
        measured_from=measured_from,
        repo=f"datasets/{REPO_ID}",
        prefix=gm.get("prefix"),
        file=gm.get("file"),
        url=url,
        bytes=int(nbytes), sha256=sha,
        header_len=int(header_len),
        shape=[int(x) for x in shape],
        dtype=str(dtype), itemsize=itemsize,
        fortran_order=False,
        slab_bytes=slab,
        read_rule=("one month of BOTH channels is one contiguous range: "
                   "offset = header_len + i * slab_bytes, length = "
                   "slab_bytes, where i is the index of the month in "
                   "`months`. The response must be HTTP 206; a 200 means the "
                   "host ignored the Range and is sending the whole file."),
        months=months,
        n_months=len(months),
        month_span=gm.get("month_span"),
        complete=bool(gm.get("complete")),
        chans=chans, labels=labels, units=units, sign=signs,
        zero_is_a_value=("0 means no vessel broadcast in that cell that "
                         "month, and is stored as 0 — not NaN. The grid has "
                         "no missing values at all."),
        grid=gm.get("grid"),
        totals=dict(
            grid_sum=gm.get("grid_sum"), store_sum=gm.get("store_sum"),
            worst_month_rel=gm.get("worst_month_rel"),
            rows=gm.get("rows"), store_N=gm.get("store_N"),
            note=("the grid's per-month sums were asserted equal to the "
                  "store's at build time, in float64, to better than 1e-3 "
                  "relative — the difference is float16 rounding per cell")),
        store=dict(
            prefix=(gm.get("prefix") or "").replace("/fishing_grid",
                                                    "/fishing"),
            N=sm.get("N"),
            fishing_hours_total=sm.get("fishing_hours_total"),
            hours_total=sm.get("hours_total"),
            date_range=sm.get("date_range"),
            qc_codes=sm.get("qc_codes"),
            vessels_per_year=sm.get("vessels_per_year")),
        source=sm.get("source"),
        licence=sm.get("licence"),
        licence_url=sm.get("licence_url"),
        attribution=sm.get("attribution"),
        cors_measured=cors,
        builder_git_sha=gm.get("builder_git_sha"),
        built_at=gm.get("built_at"),
        recipe_cmd=("python3 ml/build_family10_stores.py --store fishing "
                    "--work <dir> --stage all   # the store and this grid; "
                    "then python3 ml/publish_fishing_index.py"),
        plan="https://blauewelt.github.io/earth/docs.html"
             "?f=ml/plans/E081_family10_2_fishing.md",
    )


try:
    import numpy as np
    _HAVE_NP = True
except Exception:                                             # noqa: BLE001
    _HAVE_NP = False


def write_index(grid_manifest, store_json, repo=REPO_ID, out=INDEX,
                local=False):
    """Read the grid's manifest, measure the published file, write the index.

    `local=True` parses the header out of the file on disk and records that no
    CORS measurement was made — the fallback for a machine with no network and
    the path the tests take. The default reaches the Hub, because the numbers
    a browser needs are properties of what the Hub SERVES.
    """
    gm = json.load(open(grid_manifest, encoding="utf-8"))
    sm = json.load(open(store_json, encoding="utf-8")) if \
        os.path.exists(store_json) else {}
    prefix, name = gm["prefix"], gm["file"]
    url = resolve_url(repo, prefix, name)
    nbytes, sha = int(gm["bytes"]), gm["sha256"]
    if local:
        p = os.path.join(os.path.dirname(os.path.abspath(grid_manifest)), name)
        with open(p, "rb") as fh:
            head = fh.read(CORS_PROBE_BYTES)
        header_len, shape, dtype, fortran = parse_npy_header(head)
        nbytes, sha = os.path.getsize(p), sha256_file(p)
        cors = None
        measured_from = f"file://{p} (no CORS measurement — --local)"
    else:
        probe = measure_cors(url)
        if probe["status"] != 206 or not probe["content_range"]:
            raise SystemExit(
                f"{name}: the Hub answered {probe['status']} to a ranged GET "
                f"— this layer's whole design is one range read per month, "
                f"and a 200 means the host is sending {nbytes / 1e6:.0f} MB")
        if not probe["access_control_allow_origin"]:
            raise SystemExit(f"{name}: no access-control-allow-origin for "
                             f"{ORIGIN} — a browser could not read it")
        header_len, shape, dtype, fortran = parse_npy_header(probe["head"])
        cors = dict(origin=ORIGIN, status=probe["status"],
                    final_url_host=probe["final_url"].split("/")[2],
                    access_control_allow_origin=probe[
                        "access_control_allow_origin"],
                    access_control_expose_headers=probe[
                        "access_control_expose_headers"],
                    accept_ranges=probe["accept_ranges"],
                    content_range=probe["content_range"],
                    measured_utc=now_utc())
        measured_from = url
    if fortran:
        raise SystemExit(f"{name}: Fortran order — the slab arithmetic assumes "
                         f"C order (month-major) and this file is not")
    if list(shape) != list(gm["shape"]):
        raise SystemExit(f"{name}: the published header says shape {shape} and "
                         f"the manifest says {gm['shape']}")
    itemsize = int(np.dtype(dtype).itemsize) if _HAVE_NP else int(dtype[-1])
    cells = 1
    for d in shape[1:]:
        cells *= int(d)
    want = int(header_len) + int(shape[0]) * cells * itemsize
    if want != int(nbytes):
        raise SystemExit(f"{name}: header + data is {want} bytes and the file "
                         f"is {nbytes} — the range arithmetic would be wrong")
    idx = index_block(gm, sm, url, header_len, shape, dtype, nbytes, sha,
                      cors, measured_from)
    write_json(out, idx, out)
    return idx


# ------------------------------------------------------------- fixture mode
def build_fixture(work, out_dir=FIXTURE_DIR, repo=REPO_ID):
    """A 5°-decimated copy of a local grid, plus its index, under `data/`.

    The decimation is not cosmetic: one real month is 4.15 MB. Everything the
    browser computes — the offset arithmetic, the float16 decode, the cell
    bounds, the dateline wrap — runs identically on it, which is what makes it
    a fixture rather than a mock.
    """
    if not _HAVE_NP:
        raise SystemExit("the fixture needs numpy")
    gd = os.path.join(work, "fishing_grid")
    gm = json.load(open(os.path.join(gd, "grid.json"), encoding="utf-8"))
    src = os.path.join(gd, gm["file"])
    a = np.load(src, mmap_mode="r")
    small = np.ascontiguousarray(a[:, ::FIX_STRIDE, ::FIX_STRIDE, :])
    os.makedirs(out_dir, exist_ok=True)
    dest = os.path.join(out_dir, gm["file"])
    np.save(dest, small)
    g = dict(gm["grid"])
    step = float(gm["grid"]["step"] * FIX_STRIDE)
    g.update(ny=int(small.shape[1]), nx=int(small.shape[2]), step=step,
             decimated_from=[gm["grid"]["ny"], gm["grid"]["nx"]],
             stride=FIX_STRIDE,
             index_rule=(f"iy = clip(floor((lat + 90) / {step}), 0, "
                         f"{small.shape[1] - 1}), ix = clip(floor((lon + 180) "
                         f"/ {step}), 0, {small.shape[2] - 1}), on the row's "
                         f"CELL CENTRE"),
             fixture_note=("this is the published grid DECIMATED — every "
                           f"{FIX_STRIDE}th row and column, so a cell holds "
                           "one real 0.25° cell's month rather than a 5° "
                           "average. The arithmetic a consumer does is "
                           "identical; the picture is sparser."))
    fgm = dict(gm, shape=[int(x) for x in small.shape], grid=g,
               bytes=os.path.getsize(dest), sha256=sha256_file(dest),
               grid_sum=[float(x) for x in
                         np.nan_to_num(np.asarray(small, np.float64))
                         .reshape(-1, small.shape[-1]).sum(axis=0)],
               fixture=True)
    write_json(os.path.join(out_dir, "grid.json"), fgm)
    idx = write_index(os.path.join(out_dir, "grid.json"),
                      os.path.join(work, "fishing", "store.json"),
                      repo=repo, out=os.path.join(out_dir,
                                                  "fishing_index.json"),
                      local=True)
    idx["fixture"] = True
    write_json(os.path.join(out_dir, "fishing_index.json"), idx)
    return idx


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--grid", default="",
                    help="the grid directory holding grid.json and the .npy "
                         "(default: ml/cache/family10_2/fishing/fishing_grid)")
    ap.add_argument("--store", default="",
                    help="the store directory holding store.json (default: "
                         "beside the grid)")
    ap.add_argument("--repo", default=REPO_ID)
    ap.add_argument("--out", default=INDEX)
    ap.add_argument("--local", action="store_true",
                    help="parse the header from the local file and make NO "
                         "CORS measurement — for a machine with no network")
    ap.add_argument("--fixture", default="",
                    help="write data/fishing/fixture/ from a work dir's "
                         "fishing/ tree instead of an index")
    a = ap.parse_args(argv)
    if a.fixture:
        build_fixture(a.fixture, repo=a.repo)
        return 0
    import family10_store as f10
    gd = a.grid or os.path.join(HERE, "cache", f10.CACHE_DIRNAME, "fishing",
                                "fishing_grid")
    st = a.store or os.path.join(os.path.dirname(gd), "fishing")
    write_index(os.path.join(gd, "grid.json"),
                os.path.join(st, "store.json"),
                repo=a.repo, out=a.out, local=a.local)
    return 0


if __name__ == "__main__":
    sys.exit(main())

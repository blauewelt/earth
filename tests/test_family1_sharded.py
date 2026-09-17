#!/usr/bin/env python3
"""The sharded tier-G layout (ml/family1/sharded.py) and its wiring into
ml/build_family1_stores.py (E-082 wave 2) — no network.

    python3 -m pytest -q tests/test_family1_sharded.py

What each group is FOR:

  the layout      a round trip is EXACT (float16 with NaN, uint8 with 255);
                  edge tiles are padded and the pad is recorded and missing;
                  a missing frame is (-1, 0) in the index, None from the
                  reader and a cleared bit in the shard index; an all-missing
                  tile is never stored; check_sharded refuses a truncated
                  shard, a wrong valid count, a non-missing pad and a foreign
                  index header; the writer refuses what it cannot store.
  the reader      two range reads per tile, on a local directory and through
                  a local HTTP server that honours Range — and REFUSES a
                  server that answers 200 to a range request.
  the arithmetic  the note's "pixels x frames x valid fraction x 2C x 1.2"
                  measured on a synthetic 30 %-valid float16 field.
  the stages      a tiny GridAdapter through index, fetch (a bin straddling
                  New Year belongs to the year it starts in; bins wholly
                  outside the record are not written; a day absent upstream
                  is a zero-length frame counted by name), push-parts / pull
                  on a fake Hub, assemble (links, no re-compression), check,
                  a fake-Hub publish with its restore check, the licence
                  gate, and the probe.
"""
import argparse
import functools
import http.server
import json
import os
import shutil
import sys
import threading
import types

import numpy as np
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "ml"))

import build_family10_stores as b10                             # noqa: E402
import build_family1_stores as b1                               # noqa: E402
import family10_parts_hub as ph                                 # noqa: E402
from family1 import sharded as sh                               # noqa: E402

CH1 = (("x", "u", -50.0, 50.0),)
CH2 = (("x", "u", -50.0, 50.0), ("y", "u", 0.0, 100.0))


def grid(H, W):
    return {"H": H, "W": W, "x0": 0.0, "dx": 1.0, "y0": 0.0, "dy": 1.0,
            "crs": "test", "row_order": "row 0 first"}


def spec(H=40, W=50, C=2, F=5, dtype="float16", tile=16):
    ch = CH2 if C == 2 else CH1
    return sh.make_spec("g", grid(H, W), ch[:C], F, 432000 // F, dtype,
                        tile=tile, level=3,
                        latlon=lambda X, Y: (Y / 10.0, X / 10.0))


def write_group(root, sp, bins_frames):
    """{bin: [F arrays or None]} -> a group directory checked by the layout."""
    gd = os.path.join(root, sp["group"])
    os.makedirs(gd, exist_ok=True)
    w = sh.ShardWriter(sp)
    es = []
    for b, frames in sorted(bins_frames.items()):
        es.append(w.write_bin(b, frames,
                              os.path.join(gd, sh.shard_relpath(b)),
                              os.path.join(gd, sh.index_relpath(b))))
    with open(os.path.join(gd, "tile_grid.json"), "w") as fh:
        json.dump(sp, fh)
    sh.save_shard_index(os.path.join(gd, "shard_index.npy"), es, sp["C"])
    return gd, es


def field(rng, H, W, C, valid=0.6):
    a = rng.normal(10.0, 8.0, (H, W, C)).clip(-49, 49)
    a[..., -1] = np.abs(a[..., -1])
    a[rng.random((H, W, C)) > valid] = np.nan
    return a.astype(np.float32)


# ================================================================ layout ===
def test_float16_round_trip_is_exact_and_padding_is_recorded(tmp_path):
    rng = np.random.default_rng(1)
    sp = spec()
    assert (sp["n_tiles_y"], sp["n_tiles_x"]) == (3, 4)
    assert (sp["pad_rows"], sp["pad_cols"]) == (8, 14)
    assert sp["row_extents"][-1] == [32, 40]
    assert sp["col_extents"][-1] == [48, 50]
    assert len(sp["tile_corners"]["lat"]) == 4
    assert len(sp["tile_corners"]["lat"][0]) == 5
    assert sp["tile_corners"]["x"][-1] == 50.0      # the real edge, not pad
    frames = {2300: [field(rng, 40, 50, 2) for _ in range(5)],
              2301: [field(rng, 40, 50, 2) for _ in range(5)]}
    gd, es = write_group(str(tmp_path), sp, frames)
    g = sh.ShardedGroup(gd)
    for b, fr in frames.items():
        for f, want in enumerate(fr):
            got = g.read_frame(b, f)
            assert np.array_equal(got, want.astype(np.float16)
                                  .astype(np.float32), equal_nan=True)
    # an edge tile: T x T with the pad missing, crop gives the real extent
    t = g.read_tile(2300, 0, 2, 3, raw=True)
    assert t.shape == (16, 16, 2) and t.dtype == np.float16
    assert np.isnan(t[8:]).all() and np.isnan(t[:, 2:]).all()
    assert g.read_tile(2300, 0, 2, 3, crop=True).shape == (8, 2, 2)
    s = sh.check_sharded(gd)
    assert s["frames_present"] == 10 and s["frames_missing"] == 0
    assert s["tiles_checked"] == s["tiles_stored"] == 10 * 12
    vf = es[0]["valid_fraction"]
    assert 0.5 < vf[0] < 0.7 and 0.5 < vf[1] < 0.7


def test_uint8_round_trip_missing_frame_and_empty_tile(tmp_path):
    rng = np.random.default_rng(2)
    sp = spec(C=1, dtype="uint8")
    assert sp["missing"] == 255
    fr = []
    for f in range(5):
        a = rng.uniform(0, 49.6, (40, 50)).astype(np.float64)
        a[rng.random((40, 50)) < 0.3] = np.nan
        a[:16, :16] = np.nan                   # tile (0, 0) all missing
        fr.append(a)
    fr[2] = None                               # a day absent upstream
    gd, es = write_group(str(tmp_path), sp, {2400: fr})
    e = es[0]
    assert e["frames_present"] == 4 and e["frames_missing"] == 1
    assert e["frame_mask"] == 0b11011
    assert e["tiles_stored"] == 4 * 11
    idx = np.load(os.path.join(gd, sh.index_relpath(2400)))
    assert (idx[2, ..., 0] == -1).all() and (idx[2, ..., 1] == 0).all()
    assert (idx[:, 0, 0, 1] == 0).all()
    assert idx[0, 0, 0, 0] == 0                 # an empty tile keeps its place
    g = sh.ShardedGroup(gd)
    assert g.read_tile(2400, 2, 1, 1) is None
    assert g.read_frame(2400, 2) is None
    empty = g.read_tile(2400, 0, 0, 0, raw=True)
    assert empty.dtype == np.uint8 and (empty == 255).all()
    for f in (0, 1, 3, 4):
        want = np.where(np.isnan(fr[f]), 255, np.rint(fr[f])).astype(np.uint8)
        got = g.read_frame(2400, f, raw=True)
        assert np.array_equal(got[..., 0], want)
        dec = g.read_frame(2400, f)
        assert np.array_equal(np.isnan(dec[..., 0]), np.isnan(fr[f]))
    s = sh.check_sharded(gd)
    assert s["frames_missing"] == 1 and s["tiles_stored"] == 44


def test_the_writer_refuses_what_it_cannot_store(tmp_path):
    sp8 = spec(C=1, dtype="uint8")
    w8 = sh.ShardWriter(sp8)
    bad = np.full((40, 50), 255.0)
    with pytest.raises(sh.ShardError, match="outside"):
        w8.write_bin(1, [bad] + [None] * 4, str(tmp_path / "a.zst"),
                     str(tmp_path / "a.idx.npy"))
    sp = spec()
    w = sh.ShardWriter(sp)
    inf = np.zeros((40, 50, 2), np.float32)
    inf[0, 0, 0] = np.inf
    with pytest.raises(sh.ShardError, match="infinite"):
        w.write_bin(1, [inf] + [None] * 4, str(tmp_path / "b.zst"),
                    str(tmp_path / "b.idx.npy"))
    with pytest.raises(sh.ShardError, match="shape"):
        w.write_bin(1, [np.zeros((4, 4, 2))] + [None] * 4,
                    str(tmp_path / "c.zst"), str(tmp_path / "c.idx.npy"))
    with pytest.raises(sh.ShardError, match="five-day"):
        sh.make_spec("g", grid(4, 4), CH1, 5, 3600, "float16")
    with pytest.raises(sh.ShardError, match="frame_mask"):
        sh.make_spec("g", grid(4, 4), CH1, 240, 1800, "float16")


def test_check_sharded_refuses_damage(tmp_path):
    rng = np.random.default_rng(3)
    sp = spec()
    fr = {2300: [field(rng, 40, 50, 2) for _ in range(5)]}
    gd, es = write_group(str(tmp_path / "a"), sp, fr)
    sh.check_sharded(gd)
    shard = os.path.join(gd, sh.shard_relpath(2300))
    # (1) a truncated shard
    raw = open(shard, "rb").read()
    open(shard, "wb").write(raw[:-3])
    with pytest.raises(sh.ShardError, match="bytes"):
        sh.check_sharded(gd)
    open(shard, "wb").write(raw)
    # (2) a shard index that claims a different valid count
    es2 = [dict(es[0], valid_pixels=[es[0]["valid_pixels"][0] + 1,
                                     es[0]["valid_pixels"][1]])]
    sh.save_shard_index(os.path.join(gd, "shard_index.npy"), es2, 2)
    with pytest.raises(sh.ShardError, match="valid pixels"):
        sh.check_sharded(gd)
    sh.save_shard_index(os.path.join(gd, "shard_index.npy"), es, 2)
    # (3) a pad pixel that is not missing, in a re-written edge tile
    g = sh.ShardedGroup(gd)
    idx = np.load(os.path.join(gd, sh.index_relpath(2300)))
    t = g.read_tile(2300, 0, 2, 3, raw=True).copy()
    t[15, 15, 0] = 1.0
    blob = sh.zstd().ZstdCompressor(level=3).compress(t.tobytes())
    o, n = idx[0, 2, 3]
    new = raw[:o] + blob + raw[o + n:]
    idx2 = idx.copy()
    idx2[..., 0][idx[..., 0] > o] += len(blob) - n
    idx2[0, 2, 3, 1] = len(blob)
    open(shard, "wb").write(new)
    np.save(os.path.join(gd, sh.index_relpath(2300)), idx2)
    es3 = [dict(es[0], nbytes=len(new))]
    sh.save_shard_index(os.path.join(gd, "shard_index.npy"), es3, 2)
    with pytest.raises(sh.ShardError, match="pad"):
        sh.check_sharded(gd)
    # (4) an index with a different header length (np.save of a 3-D array)
    gd2, _ = write_group(str(tmp_path / "b"), sp, fr)
    ip = os.path.join(gd2, sh.index_relpath(2300))
    np.save(ip, np.load(ip).reshape(5, 12, 2))
    with pytest.raises(sh.ShardError, match="index"):
        sh.check_sharded(gd2)


# ================================================================ reader ===
class RangeHandler(http.server.SimpleHTTPRequestHandler):
    ignore_range = False
    served = []

    def log_message(self, *a):
        pass

    def do_GET(self):
        path = self.translate_path(self.path)
        if not os.path.isfile(path):
            self.send_error(404)
            return
        data = open(path, "rb").read()
        rng = self.headers.get("Range")
        type(self).served.append((self.path, rng))
        if rng and not self.ignore_range:
            a, b = rng.split("=")[1].split("-")
            a, b = int(a), min(int(b), len(data) - 1)
            body = data[a:b + 1]
            self.send_response(206)
            self.send_header("Content-Range", f"bytes {a}-{b}/{len(data)}")
        else:
            body = data
            self.send_response(200)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def serve(directory, ignore_range=False):
    cls = type("H", (RangeHandler,), {"ignore_range": ignore_range,
                                      "served": []})
    srv = http.server.ThreadingHTTPServer(
        ("127.0.0.1", 0), functools.partial(cls, directory=directory))
    th = threading.Thread(target=srv.serve_forever, daemon=True)
    th.start()
    return srv, cls


def test_read_tile_over_http_is_two_range_reads(tmp_path):
    rng = np.random.default_rng(4)
    sp = spec()
    fr = {2500: [field(rng, 40, 50, 2) for _ in range(5)]}
    fr[2500][4] = None
    gd, _ = write_group(str(tmp_path), sp, fr)
    srv, H = serve(str(tmp_path))
    try:
        url = f"http://127.0.0.1:{srv.server_address[1]}/g"
        g = sh.ShardedGroup(url)
        H.served.clear()
        t = g.read_tile(2500, 1, 1, 2)
        assert [r is not None for _, r in H.served] == [True, True]
        assert H.served[0][0].endswith(sh.index_relpath(2500))
        assert H.served[1][0].endswith(sh.shard_relpath(2500))
        local = sh.read_tile(gd, 2500, 1, 1, 2)
        assert np.array_equal(t, local, equal_nan=True)
        assert sh.read_tile(url, 2500, 4, 0, 0) is None
        full = g.read_frame(2500, 3)
        assert np.array_equal(full, fr[2500][3].astype(np.float16)
                              .astype(np.float32), equal_nan=True)
    finally:
        srv.shutdown()
    srv, H = serve(str(tmp_path), ignore_range=True)
    try:
        url = f"http://127.0.0.1:{srv.server_address[1]}/g"
        with pytest.raises(ValueError, match="refusing"):
            sh.ShardedGroup(url).read_tile(2500, 1, 1, 2)
    finally:
        srv.shutdown()


# ============================================================ arithmetic ===
def test_the_note_size_formula_on_a_30_percent_valid_field(tmp_path):
    """Stored bytes against pixels x frames x v x 2C x 1.2 for float16 values
    with no redundancy (normal noise) in clustered 30 %-valid patches: the
    formula is an UPPER bound, reached within ~30 % when the valid values
    themselves do not compress; a smooth real field (the probe) sits far
    below it."""
    rng = np.random.default_rng(5)
    H = W = 1024
    F, C = 5, 1
    sp = sh.make_spec("g", grid(H, W), CH1, F, 86400, "float16", tile=256,
                      level=15)
    # clustered validity: a coarse random mask upsampled to 16 px blocks
    frames = []
    for f in range(F):
        coarse = rng.random((H // 16, W // 16)) < 0.30
        m = np.kron(coarse, np.ones((16, 16), bool))
        a = rng.normal(0.0, 10.0, (H, W, 1)).astype(np.float32)
        a[~m] = np.nan
        frames.append(a)
    gd, es = write_group(str(tmp_path), sp, {3000: frames})
    v = float(np.mean([np.isfinite(a).mean() for a in frames]))
    assert 0.27 < v < 0.33
    stored = os.path.getsize(os.path.join(gd, sh.shard_relpath(3000))) + \
        os.path.getsize(os.path.join(gd, sh.index_relpath(3000)))
    formula = H * W * F * v * 2 * C * 1.2
    ratio = stored / formula
    print(f"30%-valid float16 noise: stored {stored:,} B, formula "
          f"{formula:,.0f} B, ratio {ratio:.3f}, "
          f"{stored / (H * W * F * v):.3f} B per valid pixel")
    assert 0.55 < ratio < 1.0
    sh.check_sharded(gd)


# ============================================================ the stages ===
class TinyGrid(sh.GridAdapter):
    """Two groups on 20 x 36 grids, tile 16, a daily record of its own."""
    store = "tinygrid"
    title = "tiny synthetic grid"
    family = "1gf"
    distribution = "public"
    licence = {"name": "test", "redistribution": "yes",
               "derived_works": "free"}
    channels = CH1
    log2_fp = -2.0
    log2_dt = -2.32
    first_year = 2012
    frames_per_bin = 5
    frame_seconds = 86400
    dtype = "float16"
    tile = 16
    zstd_level = 3
    grids = {"a": grid(20, 36), "b": grid(18, 16)}
    sources = ("synthetic",)
    smoke_window = ("2012-12-20", "2013-01-12")
    RECORD = (b10.parse_date("2012-12-27"), b10.parse_date("2013-01-08"))
    ABSENT = b10.parse_date("2013-01-02")

    def grid_latlon(self, group, x, y):
        return y * 0.0 + 80.0, x * 0.0

    def value(self, g, d):
        k = (d - self.RECORD[0]).days
        a = np.full(self.grids[g]["H"] * self.grids[g]["W"], float(k))
        a = a.reshape(self.grids[g]["H"], self.grids[g]["W"])
        a[::3] = np.nan
        return a + (0.5 if g == "b" else 0.0)

    def index(self, ctx):
        if ctx.source_dir:
            ctx.count_bytes(10)
        return {"dataset": "tiny", "record": [str(x) for x in self.RECORD]}

    def record_frames(self, ctx, group):
        return (self.RECORD[1] - self.RECORD[0]).days  # one absent day

    def fetch_frames(self, ctx, wanted):
        for g, b, f in wanted:
            d = sh.frame_day(b, f, self.frame_seconds)
            ctx.count_bytes(100)
            if d < self.RECORD[0]:
                yield g, b, f, None, {"frame_missing": "before_record"}
            elif d > self.RECORD[1]:
                yield g, b, f, None, {"frame_missing": "after_record"}
            elif d == self.ABSENT:
                yield g, b, f, None, {"frame_missing": "absent_upstream"}
            else:
                yield g, b, f, self.value(g, d), {"files_read": 1}


def tiny_ctx(tmp, cls=TinyGrid, **over):
    d = dict(store=cls.store, work=os.path.join(tmp, "work"),
             source_dir=os.path.join(tmp, "src"), start=cls.smoke_window[0],
             end=cls.smoke_window[1], stage="all", force=False, attempts=1,
             qc_keep=2, check_chunk_rows=b10.CHECK_CHUNK_ROWS,
             assemble="auto", parts_from_hub=False, push_parts=False,
             allow_missing_years=False, allow_unconfirmed_licence=False,
             probe_month="", smoke=False)
    d.update(over)
    os.makedirs(d["source_dir"], exist_ok=True)
    a = argparse.Namespace(**d)
    ad = cls()
    ctx = b10.Ctx(a, adapter=ad, layout=b1.layout_for(ad))
    return b1.prepare_grid_ctx(ctx)


def run(ctx, stages):
    b10.run_stages(ctx, stages, stage_fn=b1.GRID_STAGE_FN, deps=b1.DEPS)


def test_tiny_grid_through_every_stage(tmp_path):
    ctx = tiny_ctx(str(tmp_path))
    ad = ctx.adapter
    # bins by FIRST day: the bin straddling New Year sits in one year only
    starts = {b: sh.bin_start_date(b) for bs in ctx.grid_bins.values()
              for b in bs}
    assert ctx.years == sorted({d.year for d in starts.values()})
    straddle = [b for b, d in starts.items()
                if d.year == 2012 and sh.frame_day(b, 4, 86400).year == 2013]
    assert straddle and straddle[0] in ctx.grid_bins[2012]
    run(ctx, ["index", "fetch", "assemble", "check"])
    sm = json.load(open(os.path.join(ctx.store, "store.json")))
    assert sm["tier"] == "G" and sm["layout"] == "sharded"
    assert sm["family"] == "family1_gf"
    # the record is 13 days, one absent: 12 present per group, filed by the
    # year of each BIN's first day — the straddling bin carries 2013-01-01
    # into 2012 (5 December days + 1 January day; 6 more January days)
    assert sm["per_year"] == {"2012": {"a": 6, "b": 6},
                              "2013": {"a": 6, "b": 6}}
    fm = sm["frames_missing_by_reason"]
    assert fm["absent_upstream"] == 2
    assert sm["counts"]["bins_outside_record"] >= 2
    for g, G in sm["groups"].items():
        grp = sh.ShardedGroup(os.path.join(ctx.store, g))
        bins = set(int(x) for x in grp.shard_index["bin"])
        for y, bs in ctx.grid_bins.items():
            for b in bs:
                days = [sh.frame_day(b, f, 86400) for f in range(5)]
                inside = [ad.RECORD[0] <= d <= ad.RECORD[1] for d in days]
                assert (b in bins) == any(inside), (g, b)
                if b not in bins:
                    continue
                for f, d in enumerate(days):
                    got = grp.read_frame(b, f)
                    if not inside[f] or d == ad.ABSENT:
                        assert got is None
                    else:
                        want = ad.value(g, d)[:, :, None]
                        assert np.array_equal(got, want.astype(np.float32),
                                              equal_nan=True)
    assert json.load(open(os.path.join(ctx.root, "check.json")))["hub"] \
        is None
    # every file is in the sha256 block, and damage is found by the check
    p = os.path.join(ctx.store, "a", sh.shard_relpath(G["bin_first"]))
    with open(p, "ab") as fh:
        fh.write(b"x")
    with pytest.raises(sh.ShardError, match="sha256"):
        sh.check_store(ctx.store)


def test_an_adapter_that_forgets_a_frame_or_its_reason_is_refused(tmp_path):
    class Forgets(TinyGrid):
        def fetch_frames(self, ctx, wanted):
            yield from list(super().fetch_frames(ctx, wanted))[:-1]

    class NoReason(TinyGrid):
        def fetch_frames(self, ctx, wanted):
            for g, b, f, a, c in super().fetch_frames(ctx, wanted):
                yield g, b, f, a, ({} if a is None else c)

    class Stranger(TinyGrid):
        def fetch_frames(self, ctx, wanted):
            yield "a", 1, 0, None, {"frame_missing": "before_record"}

    for cls, msg in ((Forgets, "never answered"), (NoReason, "no"),
                     (Stranger, "not asked for")):
        ctx = tiny_ctx(str(tmp_path / cls.__name__), cls)
        run(ctx, ["index"])
        with pytest.raises(SystemExit, match=msg):
            run(ctx, ["fetch"])
        assert not any(b10.marked(ctx.root, f"parts/{y}") for y in ctx.years)


def test_an_unreadable_input_leaves_the_year_unmarked(tmp_path):
    class Flaky(TinyGrid):
        def fetch_frames(self, ctx, wanted):
            for i, item in enumerate(super().fetch_frames(ctx, wanted)):
                if i == 7:
                    ctx.note_absent(f"{sh.frame_day(item[1], item[2], 86400)}",
                                    "a download failed")
                    continue
                yield item

    ctx = tiny_ctx(str(tmp_path), Flaky)
    run(ctx, ["index"])
    with pytest.raises(SystemExit, match="could not"):
        run(ctx, ["fetch"])
    assert not b10.marked(ctx.root, "fetch")
    assert not b10.marked(ctx.root, f"parts/{ctx.years[0]}")


class FakeHub:
    """Commits into a directory; downloads out of it; records the calls."""

    def __init__(self, root, repo):
        self.root, self.repo, self.calls = root, repo, []

    def hub(self):
        return self, self.repo, "tok"

    def create_repo(self, repo, **kw):
        self.calls.append(("create_repo", repo))

    def commit(self, api, repo, ops, message, **kw):
        self.calls.append(("commit", repo, len(ops)))
        for rel, local in ops:
            dst = os.path.join(self.root, repo, rel)
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            shutil.copyfile(local, dst)

    def upload(self, api, repo, local, rel, message, **kw):
        self.commit(api, repo, [(rel, local)], message)

    def download(self, repo, rel, repo_type=None, token=None,
                 local_dir=None):
        self.calls.append(("download", repo, rel))
        dst = os.path.join(local_dir, rel)
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        shutil.copyfile(os.path.join(self.root, repo, rel), dst)
        return dst

    # family10_parts_hub's seams
    def list_files(self, api, repo, prefix):
        base = os.path.join(self.root, repo)
        out = set()
        for dp, _, names in os.walk(base):
            for n in names:
                rel = os.path.relpath(os.path.join(dp, n), base)
                if rel.startswith(prefix.rstrip("/") + "/"):
                    out.add(rel)
        return out

    def ph_upload(self, api, repo, pairs, message):
        self.commit(api, repo, pairs, message)

    def ph_download(self, repo, rel, token, dest_dir):
        os.makedirs(dest_dir, exist_ok=True)
        src = os.path.join(self.root, repo, rel)
        if not os.path.exists(src):
            raise FileNotFoundError(rel)
        dst = os.path.join(dest_dir, os.path.basename(rel))
        shutil.copyfile(src, dst)
        return dst


def install_hub(monkeypatch, fake, ctx=None):
    monkeypatch.setattr(b10, "hub_commit", fake.commit)
    monkeypatch.setattr(b10, "hub_add_ops", lambda pairs: list(pairs))
    monkeypatch.setattr(b10, "hub_upload_with_backoff", fake.upload)
    mod = types.ModuleType("huggingface_hub")
    mod.hf_hub_download = fake.download
    monkeypatch.setitem(sys.modules, "huggingface_hub", mod)
    monkeypatch.setattr(ph, "_list_files", fake.list_files)
    monkeypatch.setattr(ph, "_upload", fake.ph_upload)
    monkeypatch.setattr(ph, "_download", fake.ph_download)
    monkeypatch.setattr(ph, "_hub", lambda: pytest.fail("family 10's hub"))
    if ctx is not None:
        monkeypatch.setattr(ctx.layout, "hub", fake.hub)


def test_parts_travel_through_the_hub_and_assemble_on_a_box(tmp_path,
                                                            monkeypatch):
    lane = tiny_ctx(str(tmp_path / "lane"), push_parts=True)
    fake = FakeHub(str(tmp_path / "hub"), "chfrank/earth-tensors")
    install_hub(monkeypatch, fake, lane)
    run(lane, ["index", "fetch"])
    for y in lane.years:
        d = os.path.join(fake.root, fake.repo, "partials/family1_gf/tinygrid",
                         str(y))
        names = sorted(os.listdir(d))
        assert "done.json" in names and "counts.json" in names
        assert any(n.endswith(".zst") for n in names)
        assert "a__shard_index.npy" in names
        done = json.load(open(os.path.join(d, "done.json")))
        assert done["n_parts"] == 0 and done["rows"] > 0
    box = tiny_ctx(str(tmp_path / "box"), parts_from_hub=True)
    monkeypatch.setattr(box.layout, "hub", fake.hub)

    class NoSource(TinyGrid):
        def fetch_frames(self, ctx, wanted):
            pytest.fail("the box read the source")
    box.adapter.__class__ = NoSource
    run(box, ["index", "fetch", "assemble", "check"])
    # the box's store is the lane's, file for file
    run(lane, ["assemble"])
    a = json.load(open(os.path.join(lane.store, "store.json")))["sha256"]
    b = json.load(open(os.path.join(box.store, "store.json")))["sha256"]
    assert a == b


def test_publish_restores_every_file_and_samples_tiles(tmp_path,
                                                       monkeypatch):
    ctx = tiny_ctx(str(tmp_path))
    run(ctx, ["index", "fetch", "assemble"])
    fake = FakeHub(str(tmp_path / "hub"), "chfrank/earth-tensors")
    install_hub(monkeypatch, fake, ctx)
    seen = {}

    def http(repo, prefix, picks, dest, private):
        seen["picks"] = picks
        return {"checked": len(picks)}
    monkeypatch.setattr(b1, "http_verify", http)
    run(ctx, ["publish", "check"])
    sm = json.load(open(os.path.join(ctx.store, "store.json")))
    base = os.path.join(fake.root, fake.repo, "tensors/family1_gf/tinygrid")
    for n in sm["sha256"]:
        assert b10.sha256(os.path.join(base, n)) == sm["sha256"][n]
    commits = [c for c in fake.calls if c[0] == "commit"]
    assert commits[-2][2] == 1           # store.json alone, after the data
    downloads = {c[2] for c in fake.calls if c[0] == "download"}
    # every file, store.json (publish and check), manifest.json (check)
    assert len(downloads) == len(sm["sha256"]) + 2
    man = json.load(open(os.path.join(ctx.root, "manifest.json")))
    assert man["restore"]["files"] == len(sm["sha256"]) + 1
    stored = sum(g["tiles_stored"] for g in sm["groups"].values())
    assert man["restore"]["tiles_decompressed"] == \
        min(b1.GRID_RESTORE_SAMPLE, stored) == 50
    assert len(seen["picks"]) == b1.GRID_HTTP_SAMPLE
    chk = json.load(open(os.path.join(ctx.root, "check.json")))
    assert chk["hub"]["files"] == len(sm["sha256"])


def test_the_licence_gate(tmp_path, monkeypatch):
    class Pending(TinyGrid):
        store = "pending"
        licence = dict(TinyGrid.licence, redistribution="attribution",
                       redistribution_confirmed=False, pending="asked")

    fake = FakeHub(str(tmp_path / "hub"), "chfrank/earth-tensors")
    ctx = tiny_ctx(str(tmp_path / "w"), Pending)
    install_hub(monkeypatch, fake, ctx)
    monkeypatch.setattr(b1, "http_verify", lambda *a: {"checked": 0})
    run(ctx, ["index", "fetch", "assemble"])
    with pytest.raises(SystemExit, match="redistribution_confirmed"):
        run(ctx, ["publish"])
    ctx.a.push_parts = True
    with pytest.raises(SystemExit, match="redistribution_confirmed"):
        b1.push_parts(ctx)
    assert fake.calls == []
    # the flag, said out loud, lets it through
    ctx.a.allow_unconfirmed_licence = True
    run(ctx, ["publish"])
    assert any(c[0] == "commit" for c in fake.calls)
    # a PRIVATE store with the same licence is never gated
    class PendingPrivate(Pending):
        store = "pendingp"
        distribution = "private"
    ctxp = tiny_ctx(str(tmp_path / "p"), PendingPrivate)
    assert b1.licence_gate(ctxp) is True
    # and a confirmed public licence is not gated either
    assert b1.licence_gate(tiny_ctx(str(tmp_path / "t"))) is True


def test_the_grid_probe(tmp_path):
    a = argparse.Namespace(
        store="tinygrid", work=str(tmp_path), source_dir=str(tmp_path),
        start="", end="", stage="probe", force=False, attempts=1, qc_keep=2,
        check_chunk_rows=b10.CHECK_CHUNK_ROWS, assemble="auto",
        parts_from_hub=False, push_parts=False, allow_missing_years=False,
        probe_month="2013-01", smoke=False)
    out, p = b1.stage_probe_grid(a, TinyGrid, "2013-01")
    assert json.load(open(p))["frames_fetched"] == out["frames_fetched"]
    for g in ("a", "b"):
        G = out["groups"][g]
        assert G["frames_requested"] == 31
        assert G["frames_fetched"] == 7          # Jan 1, 3..8
        assert G["frames_missing"] == {"absent_upstream": 1,
                                       "after_record": 23}
        assert G["frame_present_by_slot"][:9] == [1, 0, 1, 1, 1, 1, 1, 1, 0]
        assert abs(G["valid_fraction"]["x"] - 2 / 3) < 0.02
        assert G["record_frames"] == 12 and G["estimate_store_bytes"] > 0
        assert G["tiles_per_frame"] == (2 * 3 if g == "a" else 2)
    assert out["bytes_fetched"] > 0 and out["out_of_bounds_stored"] == 0


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))

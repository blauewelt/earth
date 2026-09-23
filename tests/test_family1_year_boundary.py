#!/usr/bin/env python3
"""A tier-G year lane and its NEIGHBOUR, the repair of a year they damaged,
and the box that assembles parts pushed before lanes had names — no network.

    python3 -m pytest -q tests/test_family1_year_boundary.py

THE FAULT (found 2026-09-22 by reading the Hub). A five-day bin that opens on
2024-12-31 holds 2024-12-31 .. 2025-01-04 and is filed under 2024, the year
of its first day. pace4k's 2025 lane (run #353, window 2025-01-01 ..
2025-12-31) kept every bin its window TOUCHED, so it fetched that bin too,
filed it under 2024, finished after the 2024 lane and pushed a ONE-BIN "2024"
over it: `2024/done.json` and `pace4k__shard_index.npy` now describe bin 3141
alone and the 61 other shards sit in the folder indexed by nothing.
lst05/2007 and pheno500/2017, /2019 carry the same signature. And because the
2024 lane's window ended on 2024-12-31, an adapter that lists its source by
the window called 2025-01-01 .. 04 `after_record`.

What each test pins:

  the collision   two adjacent whole-year lanes of a tier-G store, the later
                  one finishing last: the earlier year's marker on the Hub is
                  untouched, the straddling bin is fetched WHOLE by the lane
                  of its first day, and the box's store equals the single-run
                  build of both years. (Fails before the fix.)
  the push guard  a lane that would replace a year's ledger with one covering
                  fewer shards is refused before a single byte is uploaded.
  the repair      the damaged Hub year is rebuilt from its own shards — dry
                  run first (nothing written), then for real — and the rows
                  are the ones the fetch wrote; a second run finds it
                  consistent.
  the legacy box  a box whose window is not whole years (lst05's #425,
                  `start=1999-01-01 end=2026-09-30 --parts-from-hub`) is not a
                  lane: it assembles the unnamed-lane parts on the Hub.
  the pull        a TLS handshake timeout while pulling a part is retried.
"""
import argparse
import datetime as dt
import json
import os
import ssl
import sys

import numpy as np
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "ml"))
sys.path.insert(0, HERE)

import build_family10_stores as b10                             # noqa: E402
import build_family1_stores as b1                               # noqa: E402
import family10_parts_hub as ph                                 # noqa: E402
from family1 import repair_ledger as rl                         # noqa: E402
from family1 import sharded as sh                               # noqa: E402
from test_family1_lanes import FakePartsHub, grid, hash_dir     # noqa: E402

# 2024-12-31 is a bin start (bin 3141), so the bin straddles New Year: it
# holds 2024-12-31 and 2025-01-01 .. 04 and belongs to 2024.
STRADDLE = 3141
RECORD = (dt.date(2024, 12, 1), dt.date(2025, 1, 31))
Y24 = ("2024-01-01", "2024-12-31")
Y25 = ("2025-01-01", "2025-12-31")
BOTH = ("2024-01-01", "2025-12-31")
PREFIX = "partials/family1_gf/edgegrid"


def day_of(t_s):
    return b10.START + dt.timedelta(seconds=int(t_s))


class EdgeGrid(sh.GridAdapter):
    """One small group over a record that crosses New Year, LISTED THE WAY
    irtb and sst_acspo02 list theirs: by the window in seconds, so a day past
    the window's last second is simply not in the listing (and a day before
    its first is not either)."""

    store = "edgegrid"
    title = "synthetic tier-G frames across a New Year"
    family = "1gf"
    distribution = "public"
    licence = {"name": "test", "redistribution": "yes",
               "derived_works": "free"}
    channels = (("x", "u", -50.0, 50.0),)
    log2_fp = -2.0
    log2_dt = -2.32
    first_year = 2024
    frames_per_bin = 5
    frame_seconds = 86400
    dtype = "float16"
    tile = 16
    zstd_level = 3
    grids = {"a": grid(20, 18)}
    sources = ("synthetic",)
    verified = "synthetic — no network"
    ABSENT = dt.date(2024, 12, 9)          # a hole inside the record

    def value(self, d):
        k = (d - RECORD[0]).days
        a = np.full((20, 18), float(k % 40))
        a[::3] = np.nan
        return a

    def index(self, ctx):
        return {"dataset": "edgegrid", "record": [str(d) for d in RECORD]}

    def fetch_frames(self, ctx, wanted):
        lo = max(RECORD[0], day_of(ctx.t_lo))
        hi = min(RECORD[1], day_of(ctx.t_hi))
        for g, b, f in wanted:
            d = sh.frame_day(b, f, self.frame_seconds)
            if d < lo:
                yield g, b, f, None, {"frame_missing": "before_record"}
            elif d > hi:
                yield g, b, f, None, {"frame_missing": "after_record"}
            elif d == self.ABSENT:
                yield g, b, f, None, {"frame_missing": "absent_upstream"}
            else:
                yield g, b, f, self.value(d), {"files_read": 1}


def ns(work, start, end, **over):
    d = dict(store=EdgeGrid.store, work=work, source_dir="", start=start,
             end=end, stage="all", force=False, attempts=1, qc_keep=2,
             check_chunk_rows=b10.CHECK_CHUNK_ROWS, assemble="auto",
             parts_from_hub=False, push_parts=False, lanes="",
             allow_missing_years=False, allow_unconfirmed_licence=False,
             probe_month="", smoke=False)
    d.update(over)
    return argparse.Namespace(**d)


def edge_ctx(work, start, end, main=True, **over):
    """A context as `main` builds it (`apply_lane`, then the grid) — or, with
    main=False, as the code before 2026-09-22 effectively did for an unnamed
    lane: every bin the window touches."""
    ad = EdgeGrid()
    ctx = b10.Ctx(ns(work, start, end, **over), adapter=ad,
                  layout=b1.layout_for(ad))
    if main:
        b1.apply_lane(ctx)
    return b1.prepare_grid_ctx(ctx)


def run(ctx, stages):
    b10.run_stages(ctx, stages, stage_fn=b1.GRID_STAGE_FN, deps=b1.DEPS)


class TreeHub(FakePartsHub):
    """The fake Hub, plus the listing-with-sizes `repair_ledger` reads."""

    def tree(self, api, repo, prefix):
        d = os.path.join(self.root, prefix)
        files, dirs = {}, []
        for n in sorted(os.listdir(d)) if os.path.isdir(d) else []:
            p = os.path.join(d, n)
            if os.path.isdir(p):
                dirs.append(n)
            else:
                files[n] = {"size": os.path.getsize(p),
                            "sha256": b10.sha256(p)}
        return files, dirs

    def snapshot(self):
        out = {}
        for dp, _, names in os.walk(self.root):
            for n in names:
                p = os.path.join(dp, n)
                out[os.path.relpath(p, self.root)] = b10.sha256(p)
        return out

    def done(self, year):
        return json.load(open(os.path.join(self.root, PREFIX, str(year),
                                           "done.json")))

    def shards(self, year):
        return sorted(e["name"] for e in self.done(year)["files"]
                      if e["name"].endswith(".zst"))


@pytest.fixture
def hub(tmp_path, monkeypatch):
    fake = TreeHub(str(tmp_path / "hub"))
    monkeypatch.setattr(ph, "_list_files", fake.list_files)
    monkeypatch.setattr(ph, "_upload", fake.upload)
    monkeypatch.setattr(ph, "_download", fake.download)
    monkeypatch.setattr(ph, "_hub", lambda: pytest.fail("family 10's hub"))
    monkeypatch.setattr(rl, "hub_tree", fake.tree)
    monkeypatch.setattr(b1.f10b.Layout, "hub", lambda self: fake.hub())
    return fake


def lane(tmp, name, window, main=True):
    ctx = edge_ctx(os.path.join(tmp, name), *window, main=main,
                   push_parts=True)
    run(ctx, ["index", "fetch"])
    return ctx


def bins_of(names):
    return sorted(int(n.split("__bin_")[1].split(".")[0]) for n in names)


def written(ctx, year):
    """The bins a lane actually wrote a shard for (a bin wholly outside the
    record is not written)."""
    return bins_of(n for n in os.listdir(ctx.year_dir(year))
                   if n.endswith(".zst"))


# ============================================================ the fault ====
def test_two_adjacent_year_lanes_leave_each_other_s_year_alone(tmp_path,
                                                               hub):
    """THE pace4k SHAPE. The 2024 lane, then the 2025 lane finishing LAST.
    The 2025 lane owns only bins whose first day is in 2025; bin 3141 is the
    2024 lane's, fetched with all five of its days; and the store a box
    assembles from the Hub is the single-run build of both years."""
    tmp = str(tmp_path)
    a = lane(tmp, "l2024", Y24)
    b = lane(tmp, "l2025", Y25)
    # the Hub's 2024 is the 2024 lane's, all of it, after the 2025 lane
    want = sorted(n for n in os.listdir(a.year_dir(2024)) if n.endswith(".zst"))
    assert hub.shards(2024) == want
    assert bins_of(hub.shards(2024))[-1] == STRADDLE
    assert min(bins_of(hub.shards(2025))) == STRADDLE + 1
    assert a.lane == b.lane == ""
    assert a.years == [2024] and b.years == [2025]
    assert STRADDLE in a.grid_bins[2024]
    assert STRADDLE not in b.grid_bins[2025]
    # the 2024 lane owns the straddling bin WHOLE: its window in seconds runs
    # to the end of 2025-01-04, so the listing holds those four days
    assert day_of(a.t_hi) == dt.date(2025, 1, 4)
    si = sh.load_shard_index(os.path.join(a.year_dir(2024),
                                          "a__shard_index.npy"))
    row = si[si["bin"] == STRADDLE][0]
    assert int(row["frames_present"]) == 5, "2025-01-01..04 were not fetched"

    box = edge_ctx(os.path.join(tmp, "box"), *BOTH, parts_from_hub=True)
    run(box, ["index", "fetch", "assemble", "check"])
    one = edge_ctx(os.path.join(tmp, "one"), *BOTH)
    run(one, ["index", "fetch", "assemble", "check"])
    assert hash_dir(box.store) == hash_dir(one.store)
    got = json.load(open(os.path.join(box.store, "store.json")))
    ref = json.load(open(os.path.join(one.store, "store.json")))
    assert got["groups"] == ref["groups"]
    assert got["per_year"] == ref["per_year"]
    assert got["frames_missing_by_reason"] == ref["frames_missing_by_reason"]
    # the record's own edges only: nothing is `after_record` inside it
    assert "after_record" not in {m["reason"] for m in got["missing_frames"]
                                  if m["day"] <= str(RECORD[1])}


def test_the_bin_rule_is_the_same_for_a_named_and_an_unnamed_lane(tmp_path):
    """Twelve monthly lanes and one whole-year lane own the same bins: a bin
    belongs to the lane its FIRST DAY falls in, whatever the lane is called,
    and nothing before the window's first day is ever owned."""
    whole = edge_ctx(str(tmp_path / "w"), *Y25)
    months = []
    for m in range(1, 13):
        last = (dt.date(2025, m % 12 + 1, 1) if m < 12
                else dt.date(2026, 1, 1)) - dt.timedelta(days=1)
        c = edge_ctx(str(tmp_path / f"m{m}"), f"2025-{m:02d}-01", str(last))
        assert c.lane == f"m{m:02d}"
        months += [x for v in c.grid_bins.values() for x in v]
    assert sorted(months) == whole.grid_bins[2025]
    assert min(whole.grid_bins[2025]) == STRADDLE + 1
    assert sh.bin_start_date(min(whole.grid_bins[2025])) >= dt.date(2025, 1, 1)


# ======================================================== the push guard ===
def test_a_push_that_would_shrink_a_year_s_ledger_is_refused(tmp_path, hub):
    """The same collision by the old rule (a lane keeping every bin its window
    touches): its push of a one-bin "2024" is refused BEFORE ANY UPLOAD —
    not even its own 2025 goes up, and not one byte of the Hub changes."""
    tmp = str(tmp_path)
    lane(tmp, "l2024", Y24)
    before = hub.snapshot()
    with pytest.raises(SystemExit) as e:
        lane(tmp, "old2025", Y25, main=False)
    assert "vouches for" in str(e.value)
    assert "edgegrid 2024" in str(e.value)
    assert hub.snapshot() == before
    # a re-fetch of the year that covers at least the same shards replaces it
    lane(tmp, "again2024", Y24)


def test_the_guard_reads_shards_only(tmp_path):
    have = {"files": [{"name": "a__bin_3140.zst"},
                      {"name": "a__bin_3140.idx.npy"},
                      {"name": "a__shard_index.npy"}, {"name": "counts.json"}]}
    assert ph.ledger_would_shrink(None, []) == []
    assert ph.ledger_would_shrink(have, [{"name": "a__bin_3140.zst"}]) == []
    assert ph.ledger_would_shrink(have, [{"name": "a__bin_3141.zst"}]) == \
        ["a__bin_3140.zst"]
    # tier P: numbered parts are not shards, and a re-fetch may differ
    assert ph.ledger_would_shrink({"files": [{"name": "00000.npz"}]},
                                  []) == []


# ============================================================ the repair ===
def damage(tmp, hub, monkeypatch):
    """Put the Hub into pace4k's state: the 2024 lane's year, then an
    old-rule 2025 lane's one-bin "2024" pushed over it (the guard lifted,
    exactly as it was absent on 2026-09-20)."""
    a = lane(tmp, "l2024", Y24)
    with monkeypatch.context() as m:
        m.setattr(ph, "ledger_would_shrink", lambda have, entries: [])
        b = lane(tmp, "old2025", Y25, main=False)
    assert b.years == [2024, 2025] and b.grid_bins[2024] == [STRADDLE]
    assert bins_of(hub.shards(2024)) == [STRADDLE]
    on_hub = os.listdir(os.path.join(hub.root, PREFIX, "2024"))
    assert bins_of(n for n in on_hub if n.endswith(".zst")) == \
        written(a, 2024)
    return a, b


def test_the_repair_rebuilds_the_damaged_year_from_its_own_shards(
        tmp_path, hub, monkeypatch):
    tmp = str(tmp_path)
    a, b = damage(tmp, hub, monkeypatch)
    orig = {int(r["bin"]): r for r in sh.load_shard_index(
        os.path.join(a.year_dir(2024), "a__shard_index.npy"))}
    now = {int(r["bin"]): r for r in sh.load_shard_index(
        os.path.join(b.year_dir(2024), "a__shard_index.npy"))}
    led_a = json.load(open(os.path.join(a.year_dir(2024), "counts.json")))
    # the ledger that wrote the orphans, as the repository history holds it
    monkeypatch.setattr(rl, "history_ledger",
                        lambda *x: ("rev0123456789", led_a, {"a": orig}))
    ad = EdgeGrid()
    lay = b1.layout_for(ad)
    work = os.path.join(tmp, "repair")

    # DRY RUN: every shard checked, the plan written, the Hub untouched
    before = hub.snapshot()
    rep = rl.repair_year(ad, lay, 2024, work, dry_run=True, hub=hub.hub)
    assert hub.snapshot() == before
    n = len(written(a, 2024))
    assert n > 2 and written(a, 2024)[-1] == STRADDLE
    assert rep["shards_in_folder"] == n and rep["orphaned_shards"] == n - 1
    assert rep["rebuilt"]["bins"] == n
    assert os.path.exists(rep["plan"])

    # THE REPAIR
    rep = rl.repair_year(ad, lay, 2024, work, dry_run=False, hub=hub.hub)
    assert rep["action"] == "repaired"
    assert rep["history"]["rows_verified"] == n - 1
    done = hub.done(2024)
    assert bins_of(hub.shards(2024)) == written(a, 2024)
    assert done["rows"] == sum(int(r["frames_present"])
                               for x, r in orig.items() if x != STRADDLE) + \
        int(now[STRADDLE]["frames_present"])
    # every file the marker names is on the Hub with that sha256
    for e in done["files"]:
        p = os.path.join(hub.root, PREFIX, "2024", e["name"])
        assert b10.sha256(p) == e["sha256"] and os.path.getsize(p) == \
            e["bytes"]
    # the rebuilt rows ARE the rows the fetch wrote: the 2024 lane's for the
    # orphans, the 2025 lane's for the straddling shard now in the folder
    got = {int(r["bin"]): r for r in sh.load_shard_index(
        os.path.join(hub.root, PREFIX, "2024", "a__shard_index.npy"))}
    for x, r in got.items():
        want = now[x] if x == STRADDLE else orig[x]
        for k in r.dtype.names:
            assert np.array_equal(r[k], want[k]), (x, k)
    led = json.load(open(os.path.join(hub.root, PREFIX, "2024",
                                      "counts.json")))
    assert led["rebuilt_from_shards"]["shards_adopted"]
    # every missing frame keeps a real reason: the orphans' from their own
    # ledger, the straddling shard's from the ledger that describes it
    assert rl.UNRECORDED not in led["counts"]["frames_missing"]
    assert led["counts"]["frames_missing"]["absent_upstream"] == 1
    assert led["counts"]["frames_missing"]["before_record"] == \
        sum(1 for m in led["counts"]["missing_frames"]
            if m["day"] < str(RECORD[0]) or m["bin"] == STRADDLE)

    # idempotent: the repaired year is consistent
    assert rl.repair_year(ad, lay, 2024, work, dry_run=True,
                          hub=hub.hub)["action"] == "none (consistent)"
    # and a box assembles it
    box = edge_ctx(os.path.join(tmp, "box"), *BOTH, parts_from_hub=True)
    run(box, ["index", "fetch", "assemble", "check"])
    meta = json.load(open(os.path.join(box.store, "store.json")))
    assert meta["groups"]["a"]["bins"] == len(written(a, 2024)) + \
        len(written(b, 2025))


def test_without_history_the_orphans_reasons_are_named_unrecorded(
        tmp_path, hub, monkeypatch):
    tmp = str(tmp_path)
    damage(tmp, hub, monkeypatch)
    monkeypatch.setattr(rl, "history_ledger", lambda *x: None)
    ad = EdgeGrid()
    rep = rl.repair_year(ad, b1.layout_for(ad), 2024,
                         os.path.join(tmp, "r"), dry_run=False, hub=hub.hub)
    miss = rep["rebuilt"]["frames_missing"]
    assert miss.get(rl.UNRECORDED), miss


def test_the_repair_refuses_a_shard_its_index_does_not_describe(
        tmp_path, hub, monkeypatch):
    tmp = str(tmp_path)
    damage(tmp, hub, monkeypatch)
    p = os.path.join(hub.root, PREFIX, "2024", "a__bin_3136.zst")
    with open(p, "ab") as fh:
        fh.write(b"trailing bytes")
    before = hub.snapshot()
    ad = EdgeGrid()
    with pytest.raises(SystemExit, match="bin 3136"):
        rl.repair_year(ad, b1.layout_for(ad), 2024, os.path.join(tmp, "r"),
                       dry_run=False, hub=hub.hub)
    assert hub.snapshot() == before


def test_the_repair_stage_runs_from_the_command_line(tmp_path, monkeypatch):
    seen = []
    monkeypatch.setattr(rl, "repair_year",
                        lambda ad, lay, y, work, dry_run: seen.append(
                            (ad.store, y, dry_run)) or {"year": y,
                                                        "action": "x"})
    b1.main(["--store", "pace4k", "--stage", "repair", "--start",
             "2024-01-01", "--end", "2024-12-31", "--dry-run",
             "--work", str(tmp_path)])
    b1.main(["--store", "pheno500", "--repair-year-ledger", "2017,2019",
             "--work", str(tmp_path)])
    assert seen == [("pace4k", 2024, True), ("pheno500", 2017, False),
                    ("pheno500", 2019, False)]
    with pytest.raises(SystemExit, match="tier-G"):
        b1.main(["--store", "ghcnd", "--stage", "repair", "--work",
                 str(tmp_path)])
    with pytest.raises(SystemExit, match="--dry-run"):
        b1.main(["--store", "ghcnd", "--stage", "index", "--dry-run",
                 "--work", str(tmp_path)])


# ========================================================= the legacy box ==
def test_a_box_assembles_unnamed_lane_parts_whatever_its_window(tmp_path,
                                                               hub):
    """lst05 #425: seven whole-year lanes parked their years at the top of
    each year folder (the only form there was), and the box dispatched with
    `start=1999-01-01 end=2026-09-30 --parts-from-hub` named ITSELF a lane
    (`d19990101-20260930`) and refused every year for a missing marker of
    that name. A box that assembles from the Hub is not a lane."""
    tmp = str(tmp_path)
    lane(tmp, "l2024", Y24)
    lane(tmp, "l2025", Y25)
    window = ("2024-01-01", "2025-09-30")
    assert b1.window_lane(*(b10.parse_date(x) for x in window)) == \
        "d20240101-20250930"
    box = edge_ctx(os.path.join(tmp, "box"), *window, parts_from_hub=True)
    assert box.lane == "" and box.years == [2024, 2025]
    run(box, ["index", "fetch", "assemble", "check"])
    assert box.lanes_of(2024) == [""] and box.lanes_of(2025) == [""]
    meta = json.load(open(os.path.join(box.store, "store.json")))
    assert "lanes_by_year" not in meta
    one = edge_ctx(os.path.join(tmp, "one"), *BOTH)
    run(one, ["index", "fetch", "assemble"])
    assert hash_dir(box.store) == hash_dir(one.store)


def test_a_fetch_lane_with_the_same_window_is_still_a_named_lane(tmp_path):
    ctx = edge_ctx(str(tmp_path), "2024-12-01", "2025-09-30")
    assert ctx.lane == "d20241201-20250930"


# ============================================================== the pull ===
def test_the_pull_retries_a_handshake_timeout(tmp_path, monkeypatch):
    import huggingface_hub
    from huggingface_hub.utils import HfHubHTTPError
    from huggingface_hub.errors import LocalEntryNotFoundError

    class Resp:
        def __init__(self, code):
            self.status_code = code
            self.headers = {}
            self.request = None

    calls, scripted = [], []

    def fake(repo, rel, **kw):
        calls.append(rel)
        nxt = scripted[len(calls) - 1]
        if isinstance(nxt, Exception):
            raise nxt
        return str(tmp_path / "file")

    monkeypatch.setattr(huggingface_hub, "hf_hub_download", fake)
    monkeypatch.setattr(ph, "DOWNLOAD_BACKOFF_S", (0, 0, 0, 0, 0))
    scripted[:] = [
        ssl.SSLError("_ssl.c:989: The handshake operation timed out"),
        LocalEntryNotFoundError("An error happened while trying to locate "
                                "the file on the Hub"),
        HfHubHTTPError("503", response=Resp(503)),
        "ok"]
    rel = "partials/family1_tf/lst05/2021/terra__bin_2859.idx.npy"
    assert ph._download("r", rel, None, str(tmp_path)) == \
        str(tmp_path / "file")
    assert calls == [rel] * 4
    # a 404 is our own request: no retry
    calls.clear()
    scripted[:] = [HfHubHTTPError("404", response=Resp(404))]
    with pytest.raises(HfHubHTTPError):
        ph._download("r", "y", None, str(tmp_path))
    assert calls == ["y"]
    # and a HEAD that could not reach the Hub is not "the year is absent"
    assert not ph._looks_absent(LocalEntryNotFoundError("no network"))


# ============================================================ pace4k itself ==
def test_pace4k_lists_the_year_its_last_bin_runs_into(tmp_path,
                                                      monkeypatch):
    """The 2025 lane's bin 3214 opens 2025-12-31 and holds 2026-01-01 .. 04:
    pace4k lists its record by calendar year, so it must list 2026 too, or
    those four days read as after the record."""
    pytest.importorskip("netCDF4")
    from family1.adapters import pace4k as p4
    # the smoke archive sets this for the adapter; restore it afterwards
    monkeypatch.setenv("PACE4K_SMOKE_GRID", "unset")
    src = str(tmp_path / "src")
    p4.make_smoke_sources(src, dt.date(2024, 12, 20), dt.date(2025, 1, 10))
    ad = p4.Pace4kAdapter()
    a = argparse.Namespace(**{**vars(ns(str(tmp_path / "w"), "2024-12-01",
                                        "2024-12-31")),
                              "store": "pace4k", "source_dir": src})
    ctx = b10.Ctx(a, adapter=ad, layout=b1.layout_for(ad))
    b1.apply_lane(ctx)
    b1.prepare_grid_ctx(ctx)
    assert ctx.lane == "m12" and ctx.years == [2024]
    assert STRADDLE in ctx.grid_bins[2024]
    assert ad.years(ctx) == [2024, 2025]
    run(ctx, ["index", "fetch"])
    si = sh.load_shard_index(os.path.join(ctx.year_dir(2024),
                                          "pace4k__shard_index.npy"))
    row = si[si["bin"] == STRADDLE][0]
    assert int(row["frames_present"]) == 5


if __name__ == "__main__":                                  # pragma: no cover
    sys.exit(pytest.main([__file__, "-q"]))

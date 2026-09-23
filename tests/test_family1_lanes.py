#!/usr/bin/env python3
"""The LANE-AWARE PARTS LAYOUT (E-082 wave 7) — no network.

    python3 -m pytest -q tests/test_family1_lanes.py

WHAT A LANE IS. A fetch lane is one GitHub-hosted runner: six hours and about
86 GB, and the Earthdata credentials live only there (ml/CLAUDE.md §6), so a
rented box cannot fetch those sources at all. Some stores need MORE THAN ONE
LANE PER YEAR — a year of ICESat-2's ATL08 is 35 hours of fetching (twelve
monthly lanes), a year of `lai500` is 17.6 (four quarter-lanes), and
`canopy30` is one five-day bin over 261 tiles (lanes by tile subset) — and the
parts layout used to be one folder per year with ONE index, ONE ledger and ONE
`done.json`, so two lanes of a year overwrote each other and the assembler
built a short store with nothing to say so.

So a named lane writes `partials/<family>/<store>/<year>/<lane>/` (and locally
`parts/<year>/<lane>/`), and a whole-year lane with no group subset stays the
UNNAMED lane and writes what it always wrote.

What each group is FOR:

  the name        a window earns "" (whole years), `m06`, `q3` or
                  `d0701-0930`; a group subset earns `g-<hash>`; both
                  together join with a hyphen.
  tier P          three monthly lanes of one year, fetched separately and
                  merged, are the SAME STORE as the single-lane build of the
                  same year — same N, same array bytes, same sha256 set apart
                  from store.json — and store.json records the lanes.
  tier G          the same for a sharded store, frame for frame; two lanes
                  that claim the same (group, bin) shard are REFUSED; a lane
                  owns the bins that START in its window, so monthly lanes
                  tile a year exactly once.
  declaring       `--lanes months` writes `lanes_expected` into plan.json; a
                  declared lane that never arrives is REFUSED, and
                  `--allow-missing-years` builds it with the missing lanes
                  NAMED in store.json's `degraded` and `lanes_by_year`.
  the old layout  a year folder with parts at its top level and no
                  sub-folder reads exactly as before, and its store.json
                  carries no lane keys at all.
  the Hub         `push_parts` of a named lane pushes only that lane's
                  folder; `--parts-from-hub` brings every lane of the year
                  back and assembles them.
"""
import argparse
import datetime as dt
import json
import os
import shutil
import sys

import numpy as np
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "ml"))

import build_family10_stores as b10                             # noqa: E402
import build_family1_stores as b1                               # noqa: E402
import family10_parts_hub as ph                                 # noqa: E402
from family1 import sharded as sh                               # noqa: E402

# 2022 is not a leap year and 2022-01-01 is itself a five-day bin start
# (14,610 days after 1982-01-01, and 14,610 = 5 x 2,922), so the year is 73
# whole bins — which is what makes "the bins that START in January" a clean
# thing to check.
YEAR = 2022
RECORD = (dt.date(2022, 1, 1), dt.date(2022, 3, 31))
MONTHS = {"m01": ("2022-01-01", "2022-01-31"),
          "m02": ("2022-02-01", "2022-02-28"),
          "m03": ("2022-03-01", "2022-03-31")}
WHOLE = ("2022-01-01", "2022-12-31")


def ns(store, work, start, end, **over):
    d = dict(store=store, work=work, source_dir="", start=start, end=end,
             stage="all", force=False, attempts=1, qc_keep=2,
             check_chunk_rows=b10.CHECK_CHUNK_ROWS, assemble="auto",
             parts_from_hub=False, push_parts=False, lanes="",
             allow_missing_years=False, allow_unconfirmed_licence=False,
             probe_month="", smoke=False)
    d.update(over)
    return argparse.Namespace(**d)


def hash_dir(d):
    """Every file's sha256 except store.json, which carries the build's own
    window and its lane list and is expected to differ."""
    out = {}
    for dp, _, names in os.walk(d):
        for n in names:
            rel = os.path.relpath(os.path.join(dp, n), d).replace(os.sep, "/")
            if rel != "store.json":
                out[rel] = b10.sha256(os.path.join(dp, n))
    return out


# ============================================================== the names ==
def test_a_lane_is_named_after_what_it_covers():
    D = b10.parse_date
    # whole calendar years earn NO name — that is the unnamed lane, and it is
    # what every store on the Hub was written with
    assert b1.window_lane(D("2022-01-01"), D("2022-12-31")) == ""
    assert b1.window_lane(D("1763-01-01"), D("2026-12-31")) == ""
    # a calendar month, a calendar quarter, anything else
    assert b1.window_lane(D("2022-06-01"), D("2022-06-30")) == "m06"
    assert b1.window_lane(D("2020-02-01"), D("2020-02-29")) == "m02"
    assert b1.window_lane(D("2020-07-01"), D("2020-09-30")) == "q3"
    assert b1.window_lane(D("2020-01-01"), D("2020-03-31")) == "q1"
    assert b1.window_lane(D("2022-07-01"), D("2022-09-29")) == "d0701-0929"
    assert b1.window_lane(D("2022-07-01"), D("2023-03-31")) == \
        "d20220701-20230331"
    # a group subset hashes its SORTED list, so the order it was typed in
    # cannot change the lane a dispatch writes
    a = b1.group_lane(["10N_000E", "00N_020E"])
    assert a == b1.group_lane(["00N_020E", "10N_000E"])
    assert a.startswith("g-") and len(a) == 10
    assert a != b1.group_lane(["00N_020E", "10N_000E", "20S_010E"])
    # the two halves join
    assert b1.lane_name(D("2022-06-01"), D("2022-06-30"),
                        ["00N_020E"]) == "m06-" + b1.group_lane(["00N_020E"])
    assert b1.lane_name(D("2022-01-01"), D("2022-12-31"),
                        ["00N_020E"]) == b1.group_lane(["00N_020E"])


def test_parse_lanes_declares_what_a_build_expects():
    got = b10.parse_lanes("months", [2022, 2023])
    assert sorted(got) == ["2022", "2023"]
    assert got["2022"] == [f"m{m:02d}" for m in range(1, 13)]
    assert b10.parse_lanes("quarters", [2020])["2020"] == \
        ["q1", "q2", "q3", "q4"]
    assert b10.parse_lanes("m01,m02", [2022])["2022"] == ["m01", "m02"]
    assert b10.parse_lanes("", [2022]) == {}
    with pytest.raises(SystemExit, match="--lanes"):
        b10.parse_lanes("M01", [2022])
    with pytest.raises(SystemExit, match="--lanes"):
        b10.parse_lanes("../etc", [2022])


# ====================================================== a tier-P adapter ===
class LaneRows(b10.SourceAdapter):
    """Three rows a day over a fixed record, and nothing outside it.

    The record is shorter than the year on purpose: a build whose window is
    the WHOLE year and a build whose three lanes are January, February and
    March then read exactly the same days, so the two stores are comparable
    byte for byte.
    """

    store = "lanerows"
    title = "synthetic tier-P rows, three a day"
    family = "1gf"
    distribution = "public"
    licence = {"name": "test", "redistribution": "yes",
               "derived_works": "free"}
    channels = (("x", "u", -1000.0, 1000.0),)
    log2_fp = -3.0
    log2_dt = -1.0
    first_year = YEAR
    per_year = True
    sources = ("synthetic",)
    verified = "synthetic — no network"
    PER_DAY = 3

    def index(self, ctx):
        return {"dataset": "lanerows", "record": [str(d) for d in RECORD]}

    def fetch_year(self, ctx, year):
        lo = max(ctx.d_lo, dt.date(year, 1, 1), RECORD[0])
        hi = min(ctx.d_hi, dt.date(year, 12, 31), RECORD[1])
        d = lo
        while d <= hi:
            k = (d - RECORD[0]).days
            n = self.PER_DAY
            t = [b10.seconds_since_epoch(d) + 3600 * (i + 1) for i in range(n)]
            vals = np.array([[k + 0.25 * i] for i in range(n)], np.float64)
            rows = self.pack(t, [10.0 + i for i in range(n)],
                             [20.0 + i for i in range(n)], vals,
                             [7000 + k] * n, [1] * n)
            yield str(d), rows, {"days_read": 1, "rows_read": n}
            d += dt.timedelta(days=1)


def rows_ctx(work, start, end, **over):
    ad = LaneRows()
    a = ns(ad.store, work, start, end, **over)
    return b10.Ctx(a, adapter=ad, layout=b1.layout_for(ad))


def run_rows(ctx, stages):
    b10.run_stages(ctx, stages, stage_fn=b1.STAGE_FN, deps=b1.DEPS)


def copy_lane_parts(src_work, dst_work, store):
    """What `family10_parts_hub.pull` does on the box, without a Hub: every
    lane folder and every lane marker into the assembling tree."""
    src = os.path.join(src_work, store, "parts")
    dst = os.path.join(dst_work, store, "parts")
    for dp, _, names in os.walk(src):
        rel = os.path.relpath(dp, src)
        out = dst if rel == "." else os.path.join(dst, rel)
        os.makedirs(out, exist_ok=True)
        for n in names:
            shutil.copyfile(os.path.join(dp, n), os.path.join(out, n))


def build_three_tier_p_lanes(tmp, **over):
    """January, February and March of 2022, each its own work directory and
    its own lane, merged into one tree the way a box merges them."""
    box = os.path.join(tmp, "box")
    ref = rows_ctx(box, *WHOLE, **over)
    run_rows(ref, ["index"])
    for lane, (lo, hi) in MONTHS.items():
        lane_work = os.path.join(tmp, f"lane_{lane}")
        ctx = rows_ctx(lane_work, lo, hi)
        assert b1.apply_lane(ctx) == lane
        run_rows(ctx, ["index", "fetch"])
        assert os.path.isdir(os.path.join(ctx.parts, str(YEAR), lane))
        copy_lane_parts(lane_work, box, LaneRows.store)
    return ref


def test_three_tier_p_lanes_assemble_into_the_single_lane_store(tmp_path):
    tmp = str(tmp_path)
    # the reference: ONE unnamed lane over the whole year
    one = rows_ctx(os.path.join(tmp, "one"), *WHOLE)
    assert b1.apply_lane(one) == ""
    run_rows(one, ["index", "fetch", "assemble"])
    one_meta = json.load(open(os.path.join(one.store, "store.json")))
    assert "lanes_by_year" not in one_meta
    assert one_meta["N"] == 90 * LaneRows.PER_DAY

    # the same year as three monthly lanes, merged
    box = build_three_tier_p_lanes(tmp)
    assert box.lanes_of(YEAR) == ["m01", "m02", "m03"]
    meta = b1.stage_assemble(box)

    assert meta["N"] == one_meta["N"]
    assert hash_dir(box.store) == hash_dir(one.store)
    assert meta["per_year"] == one_meta["per_year"]
    assert meta["counts"] == one_meta["counts"]
    assert meta["per_channel"] == one_meta["per_channel"]
    # and the store SAYS which lanes it came from, and that nobody declared
    # what to expect
    assert meta["lanes_by_year"] == {
        "2022": {"lanes": ["m01", "m02", "m03"], "declared": False}}
    assert "lanes_note" in meta
    # the store reads row for row like the single-lane one
    a, b = b10.f10.Store(one.store), b10.f10.Store(box.store)
    for k in ("bin", "time_s", "lat", "lon", "platform", "qc"):
        assert np.array_equal(np.asarray(a[k]), np.asarray(b[k])), k
    assert np.array_equal(np.asarray(a["values"]), np.asarray(b["values"]),
                          equal_nan=True)


def test_the_streaming_assembler_merges_lanes_to_the_same_bytes(tmp_path):
    tmp = str(tmp_path)
    one = rows_ctx(os.path.join(tmp, "one"), *WHOLE, assemble="streaming")
    b1.apply_lane(one)
    run_rows(one, ["index", "fetch", "assemble"])
    box = build_three_tier_p_lanes(tmp, assemble="streaming")
    b1.stage_assemble(box)
    assert hash_dir(box.store) == hash_dir(one.store)


def test_a_declared_lane_that_never_arrived_is_refused_then_named(tmp_path):
    tmp = str(tmp_path)
    box = build_three_tier_p_lanes(tmp, lanes="m01,m02,m03,m04")
    plan = json.load(open(os.path.join(box.root, "plan.json")))
    assert plan["lanes_expected"] == {"2022": ["m01", "m02", "m03", "m04"]}
    with pytest.raises(SystemExit, match="m04"):
        b1.stage_assemble(box)
    # the same build, with the degrade asked for BY NAME
    box.a.allow_missing_years = True
    meta = b1.stage_assemble(box)
    assert meta["lanes_by_year"]["2022"] == {
        "lanes": ["m01", "m02", "m03"], "declared": True,
        "expected": ["m01", "m02", "m03", "m04"], "missing": ["m04"]}
    assert any("m04" in m for m in meta["degraded"]["years_admitted_unmarked"])


def test_two_tier_p_lanes_whose_windows_overlap_are_refused(tmp_path):
    tmp = str(tmp_path)
    box = os.path.join(tmp, "box")
    ref = rows_ctx(box, *WHOLE)
    run_rows(ref, ["index"])
    for lane, (lo, hi) in (("m01", ("2022-01-01", "2022-01-31")),
                           ("d0115-0228", ("2022-01-15", "2022-02-28"))):
        w = os.path.join(tmp, f"lane_{lane}")
        ctx = rows_ctx(w, lo, hi)
        assert b1.apply_lane(ctx) == lane
        run_rows(ctx, ["index", "fetch"])
        copy_lane_parts(w, box, LaneRows.store)
    with pytest.raises(SystemExit, match="overlap"):
        b1.stage_assemble(ref)


def test_a_named_lane_beside_the_unnamed_one_is_refused(tmp_path):
    tmp = str(tmp_path)
    box = os.path.join(tmp, "box")
    whole = rows_ctx(box, *WHOLE)
    run_rows(whole, ["index", "fetch"])
    lane = rows_ctx(os.path.join(tmp, "m01"), *MONTHS["m01"])
    b1.apply_lane(lane)
    run_rows(lane, ["index", "fetch"])
    copy_lane_parts(os.path.join(tmp, "m01"), box, LaneRows.store)
    assert whole.lanes_of(YEAR) == ["", "m01"]
    with pytest.raises(SystemExit, match="unnamed lane"):
        b1.stage_assemble(whole)


def test_an_old_style_year_folder_reads_exactly_as_before(tmp_path):
    """No lane anywhere: the parts sit at the top level of the year folder,
    the marker is `parts/<year>.done`, the ledger carries no lane keys and
    store.json carries no lane keys — which is every store on the Hub."""
    ctx = rows_ctx(str(tmp_path), *WHOLE)
    run_rows(ctx, ["index", "fetch", "assemble", "check"])
    yd = os.path.join(ctx.parts, str(YEAR))
    assert sorted(os.listdir(yd)) == ["00000.npz", "counts.json"]
    assert b10.marked(ctx.root, f"parts/{YEAR}")
    led = json.load(open(os.path.join(yd, "counts.json")))
    assert set(led) == {"year", "rows", "parts", "counts", "at"}
    assert ctx.lanes_of(YEAR) == [""]
    assert ctx.year_dir(YEAR) == yd
    assert ctx.part_key(YEAR) == f"parts/{YEAR}"
    meta = json.load(open(os.path.join(ctx.store, "store.json")))
    assert "lanes_by_year" not in meta and "lanes_note" not in meta


# ====================================================== a tier-G adapter ===
def grid(H, W):
    return {"H": H, "W": W, "x0": 0.0, "dx": 1.0, "y0": 0.0, "dy": 1.0,
            "crs": "test", "row_order": "row 0 first"}


class LaneGrid(sh.GridAdapter):
    """Two groups on small grids over the same first quarter of 2022."""

    store = "lanegrid"
    title = "synthetic tier-G frames"
    family = "1gf"
    distribution = "public"
    licence = {"name": "test", "redistribution": "yes",
               "derived_works": "free"}
    channels = (("x", "u", -50.0, 50.0),)
    log2_fp = -2.0
    log2_dt = -2.32
    first_year = YEAR
    frames_per_bin = 5
    frame_seconds = 86400
    dtype = "float16"
    tile = 16
    zstd_level = 3
    grids = {"a": grid(20, 18), "b": grid(17, 16)}
    sources = ("synthetic",)
    verified = "synthetic — no network"

    def value(self, g, d):
        k = (d - RECORD[0]).days
        a = np.full((self.grids[g]["H"], self.grids[g]["W"]), float(k % 40))
        a[::4] = np.nan
        return a + (0.5 if g == "b" else 0.0)

    def index(self, ctx):
        return {"dataset": "lanegrid", "record": [str(d) for d in RECORD]}

    def fetch_frames(self, ctx, wanted):
        for g, b, f in wanted:
            d = sh.frame_day(b, f, self.frame_seconds)
            if d < RECORD[0]:
                yield g, b, f, None, {"frame_missing": "before_record"}
            elif d > RECORD[1]:
                yield g, b, f, None, {"frame_missing": "after_record"}
            else:
                yield g, b, f, self.value(g, d), {"files_read": 1}


def grid_ctx(work, start, end, lane=None, **over):
    ad = LaneGrid()
    a = ns(ad.store, work, start, end, **over)
    ctx = b10.Ctx(a, adapter=ad, layout=b1.layout_for(ad))
    if lane is None:
        b1.apply_lane(ctx)
    else:
        ctx.lane = lane
    return b1.prepare_grid_ctx(ctx)


def run_grid(ctx, stages):
    b10.run_stages(ctx, stages, stage_fn=b1.GRID_STAGE_FN, deps=b1.DEPS)


def build_three_tier_g_lanes(tmp, **over):
    box = os.path.join(tmp, "box")
    ref = grid_ctx(box, *WHOLE, **over)
    run_grid(ref, ["index"])
    for lane, (lo, hi) in MONTHS.items():
        w = os.path.join(tmp, f"lane_{lane}")
        ctx = grid_ctx(w, lo, hi)
        assert ctx.lane == lane
        run_grid(ctx, ["index", "fetch"])
        copy_lane_parts(w, box, LaneGrid.store)
    return ref


def test_a_tier_g_lane_owns_the_bins_that_start_in_its_window(tmp_path):
    """Monthly lanes tile a year exactly once: every bin of the year is owned
    by the lane its FIRST DAY falls in, which is the rule a year already
    follows. `bins_overlapping` alone would give January and February the same
    straddling bin, and one of the two would be thrown away."""
    whole = grid_ctx(str(tmp_path / "w"), *WHOLE)
    year_bins = set(whole.grid_bins[YEAR])
    seen = []
    for lane, (lo, hi) in MONTHS.items():
        ctx = grid_ctx(str(tmp_path / lane), lo, hi)
        assert ctx.lane == lane
        seen += ctx.grid_bins[YEAR]
    assert len(seen) == len(set(seen))            # no bin in two lanes
    assert set(seen) <= year_bins
    # every bin whose first day is in January, February or March, and no other
    want = {b for b in year_bins
            if sh.bin_start_date(b) <= dt.date(2022, 3, 31)}
    assert set(seen) == want
    # a window with no bin start at all is a lane that would fetch nothing
    with pytest.raises(SystemExit, match="no five-day bin STARTS"):
        grid_ctx(str(tmp_path / "x"), "2022-01-02", "2022-01-04")


def test_three_tier_g_lanes_assemble_into_the_single_lane_store(tmp_path):
    tmp = str(tmp_path)
    one = grid_ctx(os.path.join(tmp, "one"), *WHOLE)
    assert one.lane == ""
    run_grid(one, ["index", "fetch", "assemble", "check"])
    one_meta = json.load(open(os.path.join(one.store, "store.json")))
    assert "lanes_by_year" not in one_meta

    box = build_three_tier_g_lanes(tmp)
    assert box.lanes_of(YEAR) == ["m01", "m02", "m03"]
    meta = b1.stage_assemble_grid(box)
    assert hash_dir(box.store) == hash_dir(one.store)
    assert meta["per_year"] == one_meta["per_year"]
    assert meta["groups"] == one_meta["groups"]
    assert meta["lanes_by_year"] == {
        "2022": {"lanes": ["m01", "m02", "m03"], "declared": False}}
    # frame for frame
    for g in meta["groups"]:
        p = sh.ShardedGroup(os.path.join(box.store, g))
        q = sh.ShardedGroup(os.path.join(one.store, g))
        assert list(p.shard_index["bin"]) == list(q.shard_index["bin"])
        for b in p.shard_index["bin"]:
            for f in range(LaneGrid.frames_per_bin):
                got, want = p.read_frame(int(b), f), q.read_frame(int(b), f)
                assert (got is None) == (want is None), (g, b, f)
                if got is not None:
                    assert np.array_equal(got, want, equal_nan=True)


def test_two_tier_g_lanes_holding_the_same_shard_are_refused(tmp_path):
    """The refusal that makes the bin-ownership rule safe: hand the assembler
    two lanes that both wrote (group, bin) and it names the collision instead
    of throwing one of them away."""
    tmp = str(tmp_path)
    box = os.path.join(tmp, "box")
    ref = grid_ctx(box, *WHOLE)
    run_grid(ref, ["index"])
    for lane, (lo, hi) in (("m01", MONTHS["m01"]),
                           ("m02", MONTHS["m02"])):
        w = os.path.join(tmp, f"lane_{lane}")
        ctx = grid_ctx(w, lo, hi)
        run_grid(ctx, ["index", "fetch"])
        copy_lane_parts(w, box, LaneGrid.store)
    # forge the collision: m02's folder is given m01's own bins as well
    src = os.path.join(box, LaneGrid.store, "parts", str(YEAR), "m01")
    dst = os.path.join(box, LaneGrid.store, "parts", str(YEAR), "m02")
    for n in os.listdir(src):
        if n != "counts.json":
            shutil.copyfile(os.path.join(src, n), os.path.join(dst, n))
    led = json.load(open(os.path.join(dst, "counts.json")))
    one = json.load(open(os.path.join(src, "counts.json")))
    for g in led["groups"]:
        led["groups"][g]["bins"] += one["groups"][g]["bins"]
    json.dump(led, open(os.path.join(dst, "counts.json"), "w"))
    with pytest.raises(SystemExit, match="two lanes"):
        b1.stage_assemble_grid(ref)


def test_a_tier_g_group_subset_lane_names_itself_and_lists_its_groups(
        tmp_path):
    """`canopy30` and `lossyear` are one bin over hundreds of tiles, so their
    lanes are TILE SUBSETS. Here the two groups are fetched by two lanes, and
    each lane's ledger carries the group list its name is a hash of."""
    tmp = str(tmp_path)
    box = os.path.join(tmp, "box")
    ref = grid_ctx(box, *WHOLE)
    run_grid(ref, ["index"])
    names = []
    for g in ("a", "b"):
        class OneGroup(LaneGrid):
            grids = {g: LaneGrid.grids[g]}

            def group_subset(self, _g=g):
                return [_g]
        w = os.path.join(tmp, f"g_{g}")
        ad = OneGroup()
        a = ns(ad.store, w, *WHOLE)
        ctx = b10.Ctx(a, adapter=ad, layout=b1.layout_for(ad))
        b1.apply_lane(ctx)
        b1.prepare_grid_ctx(ctx)
        assert ctx.lane == b1.group_lane([g]) and ctx.lane_groups == [g]
        names.append(ctx.lane)
        run_grid(ctx, ["index", "fetch"])
        led = json.load(open(os.path.join(ctx.year_dir(YEAR),
                                          "counts.json")))
        assert led["lane"] == ctx.lane and led["lane_groups"] == [g]
        copy_lane_parts(w, box, LaneGrid.store)
    assert ref.lanes_of(YEAR) == sorted(names)
    meta = b1.stage_assemble_grid(ref)
    assert sorted(meta["groups"]) == ["a", "b"]
    assert meta["lanes_by_year"]["2022"]["lanes"] == sorted(names)
    one = grid_ctx(os.path.join(tmp, "one"), *WHOLE)
    run_grid(one, ["index", "fetch", "assemble"])
    assert hash_dir(ref.store) == hash_dir(one.store)


# ================================================================ the Hub ==
class FakePartsHub:
    """A directory standing in for the Hub, through the module's four seams."""

    def __init__(self, root, repo="chfrank/earth-tensors"):
        self.root, self.repo = root, repo

    def hub(self):
        return self, self.repo, "tok"

    def create_repo(self, *a, **k):
        pass

    def list_files(self, api, repo, prefix):
        out = set()
        for dp, _, names in os.walk(self.root):
            for n in names:
                rel = os.path.relpath(os.path.join(dp, n), self.root)
                if rel.startswith(prefix.rstrip("/") + "/"):
                    out.add(rel)
        return out

    def upload(self, api, repo, pairs, message):
        for rel, local in pairs:
            dst = os.path.join(self.root, rel)
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            shutil.copyfile(local, dst)

    def download(self, repo, rel, token, dest_dir, just_uploaded=False):
        src = os.path.join(self.root, rel)
        if not os.path.exists(src):
            raise FileNotFoundError(rel)
        os.makedirs(dest_dir, exist_ok=True)
        dst = os.path.join(dest_dir, os.path.basename(rel))
        shutil.copyfile(src, dst)
        return dst


def test_lane_parts_round_trip_through_the_hub_and_assemble(tmp_path,
                                                            monkeypatch):
    tmp = str(tmp_path)
    fake = FakePartsHub(os.path.join(tmp, "hub"))
    monkeypatch.setattr(ph, "_list_files", fake.list_files)
    monkeypatch.setattr(ph, "_upload", fake.upload)
    monkeypatch.setattr(ph, "_download", fake.download)
    monkeypatch.setattr(ph, "_hub", lambda: pytest.fail("family 10's hub"))

    # three hosted lanes, each pushing ONLY its own folder
    for lane, (lo, hi) in MONTHS.items():
        ctx = rows_ctx(os.path.join(tmp, f"lane_{lane}"), lo, hi,
                       push_parts=True)
        monkeypatch.setattr(ctx.layout, "hub", fake.hub)
        assert b1.apply_lane(ctx) == lane
        run_rows(ctx, ["index", "fetch"])
    pre = os.path.join(fake.root, "partials/family1_gf/lanerows", str(YEAR))
    assert sorted(os.listdir(pre)) == ["m01", "m02", "m03"]
    for lane in MONTHS:
        assert "done.json" in os.listdir(os.path.join(pre, lane))
        done = json.load(open(os.path.join(pre, lane, "done.json")))
        assert done["lane"] == lane
        assert done["lane_window"] == list(MONTHS[lane])
    # the year folder itself holds NOTHING at its top level: the unnamed lane
    # was never written, so an assembler cannot mistake one lane for the year
    assert not any(os.path.isfile(os.path.join(pre, n))
                   for n in os.listdir(pre))
    assert ph.hub_lanes(fake.list_files(None, None, "partials"), "lanerows",
                        YEAR, "partials/family1_gf") == ["m01", "m02", "m03"]

    # the box: a fresh tree, every lane pulled back, then assembled
    box = rows_ctx(os.path.join(tmp, "box"), *WHOLE, parts_from_hub=True)
    monkeypatch.setattr(box.layout, "hub", fake.hub)
    run_rows(box, ["index", "fetch", "assemble", "check"])
    assert box.lanes_of(YEAR) == ["m01", "m02", "m03"]
    for lane in MONTHS:
        assert b10.marked(box.root, f"parts/{YEAR}/{lane}")
    meta = json.load(open(os.path.join(box.store, "store.json")))
    assert meta["N"] == 90 * LaneRows.PER_DAY
    assert meta["lanes_by_year"]["2022"]["lanes"] == ["m01", "m02", "m03"]

    # and it equals the single-lane build of the same year
    one = rows_ctx(os.path.join(tmp, "one"), *WHOLE)
    run_rows(one, ["index", "fetch", "assemble"])
    assert hash_dir(box.store) == hash_dir(one.store)


def test_an_unlaned_year_on_the_hub_still_pushes_and_pulls_unchanged(
        tmp_path, monkeypatch):
    """The old layout, byte for byte: no lane folder, `done.json` at the top
    of the year folder and no `lane` key in it."""
    tmp = str(tmp_path)
    fake = FakePartsHub(os.path.join(tmp, "hub"))
    monkeypatch.setattr(ph, "_list_files", fake.list_files)
    monkeypatch.setattr(ph, "_upload", fake.upload)
    monkeypatch.setattr(ph, "_download", fake.download)
    ctx = rows_ctx(os.path.join(tmp, "lane"), *WHOLE, push_parts=True)
    monkeypatch.setattr(ctx.layout, "hub", fake.hub)
    run_rows(ctx, ["index", "fetch"])
    pre = os.path.join(fake.root, "partials/family1_gf/lanerows", str(YEAR))
    assert sorted(os.listdir(pre)) == ["00000.npz", "counts.json", "done.json"]
    done = json.load(open(os.path.join(pre, "done.json")))
    assert "lane" not in done and "lane_window" not in done
    box = rows_ctx(os.path.join(tmp, "box"), *WHOLE, parts_from_hub=True)
    monkeypatch.setattr(box.layout, "hub", fake.hub)
    run_rows(box, ["index", "fetch", "assemble", "check"])
    assert box.lanes_of(YEAR) == [""]
    meta = json.load(open(os.path.join(box.store, "store.json")))
    assert "lanes_by_year" not in meta


# ================================================== the command line itself ==
def test_the_cli_names_the_lane_from_start_end_and_the_group_subset(
        tmp_path, monkeypatch, capsys):
    """No new workflow input: `--start 2022-06-01 --end 2022-06-30` IS the
    lane, and a tier-G adapter restricted by its own environment knob adds
    the group half of the name."""
    seen = {}

    def capture(ctx, stages, stage_fn=None, deps=None):
        seen["ctx"] = ctx
    monkeypatch.setattr(b10, "run_stages", capture)

    b1.main(["--store", "ghcnd", "--stage", "index",
             "--work", str(tmp_path / "p"),
             "--start", "2022-06-01", "--end", "2022-06-30"])
    assert seen["ctx"].lane == "m06"
    assert seen["ctx"].year_dir(2022).endswith(f"parts/{YEAR}/m06")
    assert "lane      m06" in capsys.readouterr().out

    monkeypatch.setenv("LOSSYEAR_TILES", "00N_020E,10N_000E")
    b1.main(["--store", "lossyear", "--stage", "index",
             "--work", str(tmp_path / "g"),
             "--start", "2022-01-01", "--end", "2022-12-31"])
    ctx = seen["ctx"]
    assert ctx.lane == b1.group_lane(sorted(ctx.grid_specs))
    assert ctx.lane_groups == sorted(ctx.grid_specs)

    # and a whole year with the whole product is the unnamed lane
    monkeypatch.delenv("LOSSYEAR_TILES")
    b1.main(["--store", "ghcnd", "--stage", "index",
             "--work", str(tmp_path / "w"),
             "--start", "2022-01-01", "--end", "2022-12-31"])
    assert seen["ctx"].lane == ""
    assert "(unnamed)" in capsys.readouterr().out


if __name__ == "__main__":                                  # pragma: no cover
    sys.exit(pytest.main([__file__, "-q"]))

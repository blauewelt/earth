#!/usr/bin/env python3
"""E-067, "the two-year roll" — holdout BLOCKS, and the single-year invariant.

A roll is truncated at the end of the held-out stretch it started in. With the
archive's SINGLE held-out years (`holdout_years` = "2009,2017,2023") that end
is 365 days away, so no lead past a year has ever been scoreable whatever
`--horizon` asked for. Holding out CONSECUTIVE years groups them into BLOCKS
(`hold_blocks`) and moves the truncation to the block's end: three two-year
blocks make 146 pentads (730 d) reachable.

The whole change is worth exactly as much as its INVARIANT, which is that a
single-year holdout is untouched — every archived corridor AUC was produced by
the year path, and a block rewrite is the kind of change that can move one of
those by a rounding step while every individual number still looks plausible.
So six checks, and the first, fourth and sixth are the load-bearing ones:

  1. `hold_blocks` on the two spellings that matter, and on an unsorted,
     duplicated list — the blocks are a property of the SET of years, so a
     re-ordered dispatch string cannot produce a different protocol.
  2. On a real pentad `TimeAxis`: `starts_for_block((Y, Y), n)` equals
     `starts_for_year(Y, n)` for every n in 0..5 AND equals a from-scratch
     reimplementation of the PRE-E-067 rule written here — the delegation
     alone could be satisfied by two functions that are wrong the same way.
     Then the two-year block: 3 starts, first at the row before Y, stride
     len(list)//3.
  3. `scored_horizon` at Hh=146 from the row before Y: exactly 73 for the
     one-year block, exactly 146 for the two-year one, and truncated at T
     when the record runs out first. Exact counts (ml/CLAUDE.md §4.9), not
     "more than a year".
  4. `ml/lim_baseline.py --smoke` end to end. With `--hold-years` naming two
     CONSECUTIVE years and `--horizon 146`, every scored lead out to 146 has
     n > 0 — the thing single years cannot produce. And with the smoke's own
     single held-out year the artefact is BYTE-IDENTICAL to the one the
     unmodified file at BASE_SHA writes, `written_at` (a reading of the
     clock readings) excepted and counted.
  5. `ml/temporal.py --holdout-years` on the stage-2 toy: the refusal fires
     before the pool is built, a legal superset SHRINKS the training pool and
     prints the `--holdout-scope window` certificate over the new one, and the
     EFFECTIVE list lands in the saved head's own args and in stage2_config —
     which is the only reason a later roll can read the years a head was
     actually denied.
  6. The EMBED CACHE moves with the statistics. `--holdout-years` changes the
     anomaly transform, and the cache key sees neither the transform nor the
     years — so the name gains a `_hold-<blocks>` token, and
     `embed_cache_sync.cache_name` derives the SAME one `embed_cache_path`
     does. Empty token with no override, so every published asset is untouched.

    python3 tests/test_hold_blocks.py
"""
import datetime as dt
import json
import os
import re
import subprocess
import shutil
import subprocess
import sys
import tempfile

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
ML = os.path.join(ROOT, "ml")
sys.path.insert(0, ML)

import rollout_spatial as rs                                   # noqa: E402

sys.path.insert(0, HERE)
# The stage-2 toy is the grad-clip test's, not a second one: a fixture that
# drifts from the one every other stage-2 knob is checked against would make
# check 5's pool counts incomparable with theirs.
from test_e044_grad_clip import toy as _toy                    # noqa: E402
from test_e044_grad_clip import K as _K                        # noqa: E402

# The commit before E-067. A FIXED anchor, not "the previous commit": check 4
# compares against the code that wrote the archived LIM rows, and a moving
# reference would let the single-year path drift one commit at a time.
BASE_SHA = "8e24c0e"
TARGET = "ml/lim_baseline.py"
EPOCH = dt.date(1982, 1, 1)
PENTAD = 5


# ----------------------------------------------------------------- check 1 --
def check_hold_blocks():
    assert rs.hold_blocks("2009,2017,2023".split(",")) == [
        (2009, 2009), (2017, 2017), (2023, 2023)], \
        rs.hold_blocks("2009,2017,2023".split(","))
    assert rs.hold_blocks(
        "2008,2009,2016,2017,2022,2023".split(",")) == [
        (2008, 2009), (2016, 2017), (2022, 2023)]
    # unsorted AND duplicated: the blocks are a property of the SET
    messy = ["2017", "2009", "2016", "2017", " 2008 ", "2009", "2023", "2022"]
    assert rs.hold_blocks(messy) == [(2008, 2009), (2016, 2017),
                                     (2022, 2023)], rs.hold_blocks(messy)
    # a single run of four, and ints as well as strings
    assert rs.hold_blocks([2005, 2006, 2007, 2008]) == [(2005, 2008)]
    assert rs.hold_blocks([]) == []
    # the LABELS are what key the result JSON, and a one-year block's label is
    # exactly the `str(Y)` every archived artefact carries.
    assert rs.block_label((2009, 2009)) == "2009"
    assert rs.block_label((2008, 2009)) == "2008-2009"
    assert rs.block_label("2017") == "2017" and rs.block_label(2017) == "2017"
    assert rs.block_bounds("2017") == (2017, 2017)
    assert rs.block_bounds((2008, 2009)) == (2008, 2009)
    print("1. hold_blocks: '2009,2017,2023' -> three one-year blocks; "
          "'2008,2009,2016,2017,2022,2023' -> [(2008,2009), (2016,2017), "
          "(2022,2023)]; an unsorted, duplicated, whitespace-padded spelling "
          "of the same SET gives the same three blocks; and a one-year "
          "block's label is the bare year every archived key already uses ✓")


def pentad_axis(n_years=6):
    """A real pentad `TimeAxis`: 73 rows per calendar year, from 1990-01-01."""
    b0 = (dt.date(1990, 1, 1) - EPOCH).days // PENTAD + 1
    T = 73 * n_years
    bins = np.arange(b0, b0 + T, dtype=np.int64)
    months = np.array([(EPOCH + dt.timedelta(days=int(b) * PENTAD)).strftime(
        "%Y-%m") for b in bins])
    return rs.TimeAxis({"months": months, "bin_index": bins,
                        "pentad_days": np.array(PENTAD),
                        "cadence": np.array("pentad"),
                        "epoch": np.array(str(EPOCH))})


def legacy_starts(ax, Y, per_year):
    """The PRE-E-067 rule, written from scratch: the last row before Y plus
    every row inside Y except its last, then every k-th of that list."""
    rows = [int(r) for r in np.where(ax.year == int(Y))[0]]
    if not rows:
        return []
    out = ([rows[0] - 1] if rows[0] - 1 >= 0 else []) + rows[:-1]
    n = int(per_year or 0)
    if n <= 0 or n >= len(out):
        return out
    return out[::len(out) // n][:n]


# ----------------------------------------------------------------- check 2 --
def check_starts(ax):
    years = sorted({int(y) for y in ax.year})
    assert len(years) >= 4, years
    for Y in years:
        assert len(np.where(ax.year == Y)[0]) == 73, (Y, "73 rows a year")
    for Y in years:
        for n in range(0, 6):
            blk = ax.starts_for_block((Y, Y), n)
            yr = ax.starts_for_year(Y, n)
            ref = legacy_starts(ax, Y, n)
            assert blk == yr == ref, (Y, n, blk, yr, ref)
    # the two-year block: one list over BOTH years, still one start per k
    Y = years[1]
    rows = np.where((ax.year >= Y) & (ax.year <= Y + 1))[0]
    assert len(rows) == 146, len(rows)
    full = ax.starts_for_block((Y, Y + 1))
    assert len(full) == 146, len(full)
    assert full[0] == int(rows[0]) - 1, (full[0], rows[0])
    assert full[-1] == int(rows[-2]), (full[-1], rows[-2])
    three = ax.starts_for_block((Y, Y + 1), 3)
    k = len(full) // 3
    assert k == 48, k
    assert three == [full[0], full[k], full[2 * k]] == [
        int(rows[0]) - 1, int(rows[0]) - 1 + k, int(rows[0]) - 1 + 2 * k], \
        (three, k)
    # N per BLOCK, not per year: three starts over two years, not six.
    assert len(three) == 3 and len(ax.starts_for_block((Y, Y), 3)) == 3
    print(f"2. on a 73-rows-a-year pentad axis, starts_for_block((Y,Y), n) == "
          f"starts_for_year(Y, n) == an independent reimplementation of the "
          f"pre-E-067 rule, for every Y and every n in 0..5 "
          f"({len(years)} years x 6 values). The two-year block "
          f"{Y}-{Y + 1} has {len(full)} starts; at N=3 they are {three} — "
          f"first the row before {Y}'s first row, then stride "
          f"{len(full)}//3 = {k}, i.e. THREE starts spread over 730 d rather "
          f"than three per year ✓")


# ----------------------------------------------------------------- check 3 --
def check_scored_horizon(ax):
    years = sorted({int(y) for y in ax.year})
    T, Hh = ax.T, 146
    Y = years[1]
    s = int(np.where(ax.year == Y)[0][0]) - 1
    one = rs.scored_horizon(ax, s, Hh, T, (Y, Y))
    two = rs.scored_horizon(ax, s, Hh, T, (Y, Y + 1))
    assert one == 73, (one, "a one-year block truncates at 73 pentads = 365 d")
    assert two == 146, (two, "a two-year block runs the full 146 = 730 d")
    # the BARE-YEAR spelling is the same number, which is what keeps every
    # pre-E-067 call site (and every archived roll) unchanged.
    assert rs.scored_horizon(ax, s, Hh, T, Y) == one
    assert rs.scored_horizon(ax, s, Hh, T, str(Y)) == one
    # ...and truncation at the END OF THE RECORD, not at the block.
    s2 = T - 137
    late = rs.scored_horizon(ax, s2, Hh, T, (years[-2], years[-1]))
    assert late == 136, (late, T, s2)
    assert rs.scored_horizon(ax, T - 1, Hh, T, (years[-1], years[-1])) == 0
    print(f"3. scored_horizon at Hh=146 from the row before {Y}'s first row: "
          f"exactly {one} for block ({Y},{Y}) — 365.0 d, the wall single "
          f"years put in front of every archived roll — and exactly {two} for "
          f"({Y},{Y + 1}), 730.0 d. The bare-year spelling `{Y}` and `'{Y}'` "
          f"give {one} unchanged, and a start {T - s2} rows from the end of "
          f"the record scores {late}, truncated by T rather than by the "
          f"block ✓")


# ----------------------------------------------------------------- check 4 --
def base_copy(tmp):
    r = subprocess.run(["git", "-C", ROOT, "show", f"{BASE_SHA}:{TARGET}"],
                       capture_output=True, text=True)
    if r.returncode != 0:
        raise SystemExit(
            f"cannot read {TARGET} at {BASE_SHA}: {r.stderr.strip()}. This "
            f"check compares the single-year artefact against the code that "
            f"wrote the archived LIM rows; without it there is no reference "
            f"and it must FAIL rather than pass vacuously.")
    p = os.path.join(tmp, "lim_baseline_base.py")
    open(p, "w").write(r.stdout)
    return p


# THE CLOCK READINGS, and nothing else. `written_at` is when the run happened;
# `wall_s` (one per LIM entry) and the four `timings_s` numbers are how long
# its phases took. None of them is a result, all of them differ between two
# runs of identical code, and the LIM artefact carries no other field that
# does — which is why the byte comparison below can be exact. Each pattern is
# COUNTED and the count asserted, so the exclusion cannot quietly widen into
# a number that matters (tests/test_roll_monthly_identity.py's own rule).
CLOCKS = (
    ("written_at", re.compile(r'^\s*"written_at":.*\n', re.M), 1),
    ("wall_s", re.compile(r'^\s*"wall_s":.*\n', re.M), None),
    ("timings_s", re.compile(
        r'^\s*"timings_s": \{\n(?:.*\n)*?\s*\},\n', re.M), 1),
)


def strip_clocks(raw):
    """Remove every clock reading, and report how many of each went."""
    counts = {}
    for name, pat, want in CLOCKS:
        found = pat.findall(raw)
        counts[name] = len(found)
        if want is not None and len(found) != want:
            raise SystemExit(
                f"expected exactly {want} `{name}` field(s) in the LIM "
                f"artefact, found {len(found)} — the exclusion list is "
                f"describing a file that has changed shape")
        raw = pat.sub("", raw)
    if counts["wall_s"] < 1:
        raise SystemExit("no `wall_s` in the LIM artefact — the exclusion "
                         "list is stripping something that is not there")
    return raw, counts


def run_smoke(script, out, extra=()):
    r = subprocess.run(
        [sys.executable, "-u", script, "--smoke", "--out", out,
         "--K", "20", "--progress-every", "1000", *extra],
        capture_output=True, text=True, timeout=600, cwd=ROOT,
        env=dict(os.environ, PYTHONPATH=ML + os.pathsep
                 + os.environ.get("PYTHONPATH", "")))
    if r.returncode != 0:
        print(r.stdout[-3000:])
        print(r.stderr[-3000:])
        raise SystemExit(f"{os.path.basename(script)} --smoke failed "
                         f"(rc {r.returncode})")
    return open(out).read(), r.stdout


def check_smoke():
    tmp = tempfile.mkdtemp(prefix="hold_blocks_")
    here = os.path.join(ML, "lim_baseline.py")

    # ---- (a) the TWO-YEAR block, end to end -----------------------------
    # The smoke fixture is three pentad years (1990-1992) with 1991 held out
    # by its "checkpoint". `--hold-years 1991,1992` is a legal SUPERSET of
    # that, and it is the only configuration in which a 146-step horizon can
    # score anything past lead 73.
    out2 = os.path.join(tmp, "block.json")
    raw2, log2 = run_smoke(here, out2,
                           ("--hold-years", "1991,1992", "--horizon", "146"))
    d2 = json.loads(raw2)
    assert d2["hold_years"] == ["1991", "1992"], d2["hold_years"]
    assert "1991-1992" in log2, log2[-2000:]
    keys = set(d2["starts"]["rows"])
    assert keys == {"1991-1992"}, keys
    assert set(d2["cadence"]["starts_per_holdout_year"]) == {"1991-1992"}
    assert len(d2["starts"]["rows"]["1991-1992"]) == 3, d2["starts"]
    blk = d2["heads"]["lim_k20"]["corridor"]
    by_h = {r["h"]: r["n"] for r in blk["chan_skill"]}
    assert max(by_h) == 146, (max(by_h), "the horizon must be reached")
    # EVERY lead inside the block is scored — no hole, no n == 0 row.
    for h in range(1, 147):
        assert h in by_h, f"lead {h} is absent from chan_skill"
        assert by_h[h] > 0, f"lead {h} has n = {by_h[h]}"
    beyond = sorted(h for h in by_h if h > 73)
    assert beyond and by_h[146] > 0, by_h
    print(f"4a. --hold-years 1991,1992 --horizon 146 on the LIM smoke: one "
          f"block keyed `1991-1992`, 3 starts, and chan_skill carries "
          f"{len(by_h)} leads with n > 0 at every one of them — "
          f"{len(beyond)} of them PAST lead 73, out to h=146 (n = "
          f"{by_h[146]:,}), which a single held-out year cannot produce ✓")

    # the refusal, at the point the inputs are all it has cost
    r_bad = subprocess.run(
        [sys.executable, "-u", here, "--smoke", "--out",
         os.path.join(tmp, "never.json"), "--K", "20",
         "--hold-years", "1992"], capture_output=True, text=True, timeout=600,
        cwd=ROOT)
    assert r_bad.returncode != 0, r_bad.stdout[-1500:]
    assert "not a superset" in r_bad.stdout + r_bad.stderr, \
        (r_bad.stdout[-1500:], r_bad.stderr[-1500:])
    assert "channel survey" not in r_bad.stdout, \
        "the refusal fired after the read pass, not before it"
    print("4b. --hold-years that DROPS a checkpoint holdout year (1992 "
          "without 1991) is refused before the channel survey — the LIM may "
          "be denied more than the head was, never less ✓")

    # ---- (c) the single-year invariant, against BASE_SHA ----------------
    base = base_copy(tmp)
    raw_new, _ = run_smoke(here, os.path.join(tmp, "new.json"))
    raw_old, _ = run_smoke(base, os.path.join(tmp, "old.json"))
    raw_new, c_new = strip_clocks(raw_new)
    raw_old, c_old = strip_clocks(raw_old)
    assert c_new == c_old, (c_new, c_old)
    if raw_new != raw_old:
        a, b = raw_old.splitlines(), raw_new.splitlines()
        diff = [f"  line {i + 1}:\n    {BASE_SHA}: {x}\n    HEAD: {y}"
                for i, (x, y) in enumerate(zip(a, b)) if x != y][:6]
        raise SystemExit(
            "the SINGLE-YEAR LIM artefact moved. Every archived LIM row was "
            "written by the year path, and E-067 must leave it alone:\n"
            + "\n".join(diff))
    print(f"4c. with the smoke's own single held-out year the artefact is "
          f"BYTE-IDENTICAL to the one {BASE_SHA} writes "
          f"({len(raw_new):,} bytes, {len(raw_new.splitlines()):,} lines) — "
          f"with only the CLOCK readings removed from each side and their "
          f"counts asserted equal ("
          + ", ".join(f"{k} x{v}" for k, v in c_new.items())
          + "). Nothing else is excluded, and the single-year path did not "
            "move ✓")


# ----------------------------------------------------------------- check 5 --
def pool_of(stdout):
    for ln in stdout.splitlines():
        if ln.startswith("train windows:"):
            return int(ln.split(":")[1].strip().replace(",", ""))
    raise AssertionError("no 'train windows:' line")


def s2_train(npz, run, tmp, extra, tag):
    """One 1-step ml/temporal.py run on the grad-clip test's toy ocean.

    One step, because what is asserted here is a POOL and a RECORD, not a
    curve: which bins the trainer was allowed to draw from, and whether the
    years it was denied travel in the artefact it saves."""
    run_dir = os.path.join(ML, "runs", run)
    for f in ("metrics.jsonl", "temporal.json", "temporal.pt"):
        q = os.path.join(run_dir, f)
        if os.path.exists(q):
            os.remove(q)
    r = subprocess.run(
        [sys.executable, "-u", os.path.join(ML, "temporal.py"),
         "--run", run, "--data", npz, "--K", str(_K), "--steps", "1",
         "--batch", "8", "--d-model", "16", "--layers", "2",
         "--max-pixels", "30", "--seed", "0", *extra],
        capture_output=True, text=True, timeout=1800,
        env=dict(os.environ,
                 CKPT_DIR_OVERRIDE=os.path.join(tmp, "ckpt", tag)))
    return r, run_dir


def check_temporal():
    import torch
    tmp = tempfile.mkdtemp(prefix="hold_blocks_s2_")
    run = "e067_holdblocks"
    run_dir = os.path.join(ML, "runs", run)
    try:
        os.makedirs(run_dir, exist_ok=True)
        npz, ckd = _toy(tmp)
        assert ckd["args"]["holdout_years"] == "1992", ckd["args"]
        torch.save(ckd, os.path.join(run_dir, "pixelmae.pt"))

        # (a) THE REFUSAL, before the tensor is read. `--holdout-years 1991`
        # would put 1992 — a year the codec was never fitted on — back into
        # the stage-2 pool, which is the one direction nothing downstream can
        # detect.
        r_bad, _ = s2_train(npz, run, tmp, ["--holdout-years", "1991"], "bad")
        out_bad = r_bad.stdout + r_bad.stderr
        assert r_bad.returncode != 0, out_bad[-1500:]
        assert "not a superset" in out_bad and "1992" in out_bad, \
            out_bad[-1500:]
        assert "Traceback" not in r_bad.stderr, "refused with a traceback"
        assert "train windows" not in out_bad, \
            f"the refusal came AFTER the pool was built: {out_bad[-400:]}"

        # (b) the DEFAULT run: the codec's own list, recorded as effective.
        r_d, _ = s2_train(npz, run, tmp, [], "default")
        assert r_d.returncode == 0, (r_d.stdout[-2500:], r_d.stderr[-2500:])
        ck_d = torch.load(os.path.join(run_dir, "temporal.pt"),
                          map_location="cpu", weights_only=False)
        assert ck_d["args"]["holdout_years"] == "1992", ck_d["args"]
        pool_d = pool_of(r_d.stdout)

        # (c) the OVERRIDE: two consecutive years, a strictly smaller pool,
        # and the effective list in the head's own args AND in stage2_config.
        r_o, _ = s2_train(npz, run, tmp,
                          ["--holdout-years", "1991,1992"], "override")
        assert r_o.returncode == 0, (r_o.stdout[-2500:], r_o.stderr[-2500:])
        ck_o = torch.load(os.path.join(run_dir, "temporal.pt"),
                          map_location="cpu", weights_only=False)
        assert ck_o["args"]["holdout_years"] == "1991,1992", ck_o["args"]
        pool_o = pool_of(r_o.stdout)
        assert pool_o < pool_d, (pool_o, pool_d,
                                 "denying a year must cost windows")
        recs = [json.loads(l) for l in
                open(os.path.join(run_dir, "metrics.jsonl")) if l.strip()]
        cfg = [x["stage2_config"] for x in recs if "stage2_config" in x]
        assert cfg, "no stage2_config record"
        assert cfg[-1]["holdout_years"] == "1991,1992", cfg[-1]
        assert cfg[-1]["codec_holdout_years"] == "1992", cfg[-1]
        assert cfg[-1]["train_windows"] == pool_o, (cfg[-1], pool_o)
        # ONE MASK, and the run says so, next to the certificate that counts
        # what the new pool costs.
        assert "ONE MASK" in r_o.stdout, r_o.stdout[-2500:]
        assert "--holdout-scope window" in r_o.stdout, r_o.stdout[-2500:]
        assert "certificate: 0 of" in r_o.stdout, r_o.stdout[-2500:]
        print(f"5. ml/temporal.py --holdout-years: dropping a codec holdout "
              f"year is refused before the pool is built; 1991,1992 (a legal "
              f"superset of the codec's 1992) shrinks the training pool "
              f"{pool_d} -> {pool_o} windows, prints the --holdout-scope "
              f"window certificate over the NEW pool, and writes the "
              f"EFFECTIVE list into both the saved head's own "
              f"args['holdout_years'] and stage2_config (beside "
              f"codec_holdout_years '1992') — so a roll can read the years "
              f"the head was actually denied. The default run records '1992', "
              f"the codec's own ✓")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
        shutil.rmtree(run_dir, ignore_errors=True)


# ----------------------------------------------------------------- check 6 --
def check_cache_name():
    """The embed cache must be HOLDOUT-AWARE, and by ONE definition.

    ml/temporal.py's own two-masks comment states the hazard: the cache key is
    (codec weight hash, sha256 of the RAW tensor) and NEITHER TERM SEES THE
    ANOMALY TRANSFORM, so "two runs on the same codec and the same tensor with
    different statistics would share one cache key: whichever pulled would
    train on the other's embeddings, and every shape, dtype and length check
    would pass". `--holdout-years` moves exactly those statistics. So the name
    must move with them — and `embed_cache_sync.cache_name` must derive the
    SAME name `embed_cache_path` does, or the local file and the release asset
    are two different caches wearing one key.
    """
    import torch
    import temporal
    import embed_cache_sync as sync

    CODEC = "2009,2017,2023"
    BLOCKED = "2008,2009,2016,2017,2022,2023"
    # (a) the token itself: empty unless the years actually differ, and blind
    # to ordering and whitespace — the same year SET is the same cache.
    assert temporal.hold_key(None, CODEC) == ""
    assert temporal.hold_key("", CODEC) == ""
    assert temporal.hold_key(CODEC, CODEC) == ""
    assert temporal.hold_key("2023, 2009 ,2017", CODEC) == ""
    assert temporal.hold_key(["2017", "2009", "2023"], CODEC) == ""
    _tok = temporal.hold_key(BLOCKED, CODEC)
    assert _tok == "_hold-2008-2009-2016-2017-2022-2023", _tok
    # a single two-year block reads as one block, not two years
    assert temporal.hold_key("2008,2009", "2009") == "_hold-2008-2009"

    # (b) the PATH: unchanged with no override, distinct with one.
    plain = temporal.embed_cache_path("actions", "w" * 10, "d" * 10)
    hold = temporal.embed_cache_path("actions", "w" * 10, "d" * 10,
                                     hold=temporal.hold_key(BLOCKED, CODEC))
    assert os.path.basename(plain) == "Z_actions_wwwwwwwwww_dddddddddd.npy", \
        os.path.basename(plain)
    assert os.path.basename(hold) == (
        "Z_actions_wwwwwwwwww_dddddddddd"
        "_hold-2008-2009-2016-2017-2022-2023.npy"), os.path.basename(hold)
    assert plain != hold, "an overridden run would pull the codec's own Z"

    # (c) THE TWO NAMERS AGREE — against a real checkpoint on disk, because
    # `cache_name` reads the codec's own years out of it rather than being
    # told them. That is the half a unit test of `hold_key` alone cannot see.
    tmp = tempfile.mkdtemp(prefix="hold_blocks_cache_")
    run = "e067_cachename"
    run_dir = os.path.join(ML, "runs", run)
    try:
        os.makedirs(run_dir, exist_ok=True)
        npz, ckd = _toy(tmp)
        ckd["args"]["holdout_years"] = CODEC
        torch.save(ckd, os.path.join(run_dir, "pixelmae.pt"))
        whash = temporal.codec_weight_hash(ckd)
        dhash = temporal.data_fingerprint(npz)
        for years in (None, CODEC, "2023,2009,2017", BLOCKED, "2008,2009"):
            path, asset, label = sync.cache_name(run, npz, years)
            hkey = temporal.hold_key(years, CODEC)
            assert path == temporal.embed_cache_path(run, whash, dhash,
                                                     hold=hkey), (years, path)
            assert asset == f"Z_{whash}_{dhash}{hkey}.npy", (years, asset)
            assert label == f"{whash}/{dhash}{hkey}", (years, label)
            # the ASSET is what a release carries, so it must move too — a
            # matching local path with a shared asset name is the same bug
            # one hop away.
            assert (hkey != "") == (asset != f"Z_{whash}_{dhash}.npy")
        # and the default call — every caller written before E-067 — is
        # byte-for-byte the name this file has always used.
        assert sync.cache_name(run, npz) == sync.cache_name(run, npz, None)
        p0, a0, _ = sync.cache_name(run, npz)
        assert a0 == f"Z_{whash}_{dhash}.npy", a0
        pB, aB, _ = sync.cache_name(run, npz, BLOCKED)
        print(f"6. the embed cache is HOLDOUT-AWARE and named ONCE: "
              f"hold_key is '' for the codec's own years in any order and "
              f"'{temporal.hold_key(BLOCKED, CODEC)}' for the blocked "
              f"superset; embed_cache_sync.cache_name reproduces "
              f"embed_cache_path exactly for five different overrides, local "
              f"path AND release asset ({a0} -> {aB}); and the no-argument "
              f"call every pre-E-067 caller makes is the unchanged name ✓")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
        shutil.rmtree(run_dir, ignore_errors=True)


def main():
    ax = pentad_axis()
    check_hold_blocks()
    check_starts(ax)
    check_scored_horizon(ax)
    check_smoke()
    check_temporal()
    check_cache_name()
    print("\nE-067 holdout blocks: all checks hold ✓")


if __name__ == "__main__":
    main()

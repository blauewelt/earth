#!/usr/bin/env python3
"""The MONTHLY roll must not have moved. Byte-for-byte, against the archive.

Every corridor AUC in ml/EXPERIMENTS.md, every band correlation quoted in a
result report and every number the #217 validation gate is calibrated on was
produced by `ml/rollout_spatial.py` on a MONTHLY family-3 axis. On 2026-08-19
that file was made cadence-aware so a pentad roll could be scored at all
(E-044) — and a cadence-aware rewrite is exactly the kind of change that can
move a monthly number by a rounding step and make the whole archive
incomparable without anyone noticing, because every individual number still
looks plausible.

So this test does not check properties of the monthly path. It runs the REAL
evaluator twice on one synthetic ocean — once as it stands, once as it stood
at `BASE_SHA` — and demands the two `roll.json` files be identical:

    python3 tests/test_roll_monthly_identity.py

`BASE_SHA` is the commit BEFORE the cadence work: the last state in which
every published monthly roll was produced. It is a fixed anchor, not "the
previous commit" — the point is to compare against the archive, and a moving
reference would let the monthly path drift one commit at a time.

THREE FIELDS ARE EXCLUDED, and the second and third are DECISIONS worth
reading.

`heads.*.wall_s` is a reading of the clock rather than a result. It is removed
from BOTH payloads and the count is asserted, so the exclusion cannot quietly
widen.

`heads.*.<scope>.horizon_auc_daymatched` (2026-08-20, E-044) is a NEW key
carrying an OLD value. It is the mean of `msss_clim` over the twelve leads
that stand for the monthly archive's twelve lead DURATIONS on whatever axis
is being rolled — 1..12 at monthly, {6,12,…,73} at pentad — and it exists
because `horizon_auc` (an unweighted mean over h=1..H) is a function of the
axis's own lead sampling and so cannot be compared across cadences. It is
emitted at EVERY cadence deliberately: a comparable number that appears only
on the axis nobody has archived is a number the harvest will forget to ask
for, and `sroll_run.sh` asserts its presence for exactly that reason.

`heads.*.<scope>.per_channel` (2026-08-28, E-058) is a NEW key carrying a
FINER-GRAINED value. Every row this evaluator wrote before it was pooled over
all pixels AND all 40 channels — `chan_skill` is a legacy name from rollout.py
where a "chan" row was a HORIZON — so `sst`, appended by E-042 and last in the
tensor, was invisible inside the pool and no archived roll could say whether
the rolled embedding predicts sea-surface temperature at all. `per_channel`
maps each channel's NAME to that channel's own rows, produced by the pooled
rows' own `_skill_rows` from a parallel set of sums that `accumulate` fills on
the same branch from the same masked arrays. It is written BETWEEN `n_px` and
`horizon_auc` rather than appended, precisely so that the "written LAST"
invariant the day-matched strip's comma-removal depends on stays true — which
is why the two strips below are shaped differently and neither may be reused
for the other.

BYTE-IDENTITY OR KEY-IDENTITY? The choice is between (a) comparing the parsed
objects while ignoring keys the base version never wrote, and (b) keeping the
byte comparison and stripping the new key first, the way `wall_s` is stripped.
This test takes (b), and pins the stripped key separately:

  * the guarantee the archive needs is that no PUBLISHED number moved and no
    PUBLISHED key changed SHAPE. Byte comparison is what holds the second
    half: a key-wise `==` over parsed JSON would accept a reordering, a
    changed indent, and `0.643` becoming `0.6430000000000001` — all of which
    are exactly the kind of drift a cadence-aware rewrite produces and none
    of which a reader diffing against the archive would forgive.
  * a purely ADDITIVE key whose value is a function of rows already in the
    file cannot make anything incomparable — but "purely additive" and "a
    function of rows already in the file" are claims, and claims are what get
    tested. So the strip is not a licence: every removed value must equal its
    scope's own `horizon_auc` EXACTLY (check 3b), and every removed
    `per_channel` block must be arithmetically CONSISTENT with the pooled row
    it sits beside (check 2c) — statements stronger than key-identity could
    make, because key-identity would let the new key hold anything at all.
  * and each exclusion is COUNTED (one per scored scope per head, and zero
    scopes may be missed), so the day someone adds a THIRD key here, this
    test fails rather than widening. That is not a hypothetical: `per_channel`
    IS that second key, and it arrived by this test failing on 2026-08-28.
    The widening was a decision taken in the open, which is the whole point of
    the tripwire.
"""
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
ML = os.path.join(ROOT, "ml")
sys.path.insert(0, HERE)
from test_rollout_spatial import build_fixture                 # noqa: E402

# the commit before ml/rollout_spatial.py became cadence-aware
BASE_SHA = "9066341"
TARGET = "ml/rollout_spatial.py"


def base_copy(tmp):
    """`BASE_SHA`'s rollout_spatial.py on disk, runnable against today's ml/."""
    r = subprocess.run(["git", "-C", ROOT, "show", f"{BASE_SHA}:{TARGET}"],
                       capture_output=True, text=True)
    if r.returncode != 0:
        raise SystemExit(
            f"cannot read {TARGET} at {BASE_SHA}: {r.stderr.strip()}. This "
            f"test compares against the archive's own code; without it there "
            f"is no monthly reference and it must FAIL rather than pass "
            f"vacuously.")
    p = os.path.join(tmp, "rollout_spatial_base.py")
    open(p, "w").write(r.stdout)
    return p


def run(script, f, out, cache, extra=()):
    # The CURRENT evaluator defaults --unpooled-readout ON (Chris, 2026-08-27);
    # this certificate pins the LEGACY pooled artefact byte-for-byte, so the
    # modern script is invoked with the explicit opt-out. BASE_SHA's evaluator
    # predates the flag and gets no extra args. Pinning the test, not the
    # production default.
    os.makedirs(cache, exist_ok=True)
    env = dict(os.environ, PYTHONPATH=ML + os.pathsep
               + os.environ.get("PYTHONPATH", ""))
    cmd = [sys.executable, "-u", script,
           "--x", f["x"], "--npz-small", f["npz"], "--z", f["z"],
           "--ckpt", f["ckpt"], "--out", out, "--horizon", "3",
           "--long-start", "1991-12", "--long-months", "16",
           "--future-months", "5", "--cache-dir", cache,
           "--no-gate", "--heads", *f["heads"], *extra]
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=1800,
                       env=env, cwd=ROOT)
    if r.returncode != 0:
        print(r.stdout[-3000:])
        print(r.stderr[-3000:])
        raise SystemExit(f"{os.path.basename(script)} failed on the monthly "
                         f"toy (rc {r.returncode})")
    return open(out).read(), r.stdout


WALL = re.compile(r'^\s*"wall_s":.*\n', re.M)
# The key AND the comma its insertion put on the line before it. It is written
# LAST in its block (skill_block appends it after `auc_damped`), so removing
# only its own line would leave `"auc_damped": 0.003,` where the base version
# has `"auc_damped": 0.003` — a byte difference that is punctuation, not a
# number, and would make this test cry wolf on every future addition. That
# "written last" is not assumed: strip_daymatched() asserts it per block.
DAYMATCH = re.compile(r',\n[ ]*"horizon_auc_daymatched":[^\n]*\n')
# The whole `"per_channel": { ... }` block: its opening line, its nested body,
# and the `},` that closes it at the same indent — its OWN trailing comma
# included. Unlike DAYMATCH this takes nothing off the line ABOVE, because the
# key is not written last: skill_block emits it between `n_px` and
# `horizon_auc`, so the comma ending the preceding line was already there
# before E-058 and removing it would manufacture the very byte difference this
# test exists to catch. The backreference is what stops the non-greedy body in
# the right place — every line inside the block is indented deeper than the
# key, so `\1},` can only ever be this key's own closing line. That "not last,
# and immediately before `horizon_auc`" is not assumed: strip_per_channel()
# asserts it per block, and the audit block's similarly-named
# `per_channel_msss_clim_corridor` is a list, so `": {"` cannot match it.
PER_CHANNEL = re.compile(r'^([ ]*)"per_channel": \{\n(?:.*\n)*?\1\},\n', re.M)
SCOPES_WITH_ROWS = "chan_skill"


def strip_wall(raw):
    """Drop the clock reading, and prove it was the only thing dropped."""
    obj = json.loads(raw)
    n = 0
    for e in obj.get("heads", {}).values():
        n += e.pop("wall_s", None) is not None
    return WALL.sub("", raw), obj, n


def strip_daymatched(raw, obj):
    """Drop `horizon_auc_daymatched`, returning what was dropped and from
    where, so the caller can pin each value against the key it duplicates.

    Only the NEW payload ever has any: `BASE_SHA` predates the key, so on the
    old payload this is a no-op and the count is 0 — which is itself asserted
    below, because a base payload that somehow had them would mean the strip
    was hiding a difference rather than a new key."""
    got = []
    for lab, e in obj.get("heads", {}).items():
        for scope, blk in e.items():
            if isinstance(blk, dict) and SCOPES_WITH_ROWS in blk:
                if "horizon_auc_daymatched" not in blk:
                    continue
                assert list(blk)[-1] == "horizon_auc_daymatched", (
                    f"{lab}/{scope}: the new key is no longer the LAST in its "
                    f"block ({list(blk)}), so the byte strip above — which "
                    f"also removes the comma its insertion added to the "
                    f"preceding line — no longer describes the file")
                v = blk.pop("horizon_auc_daymatched")
                got.append((lab, scope, v, blk.get("horizon_auc")))
    return DAYMATCH.sub("\n", raw), got


def strip_per_channel(raw, obj):
    """Drop `per_channel`, returning what was dropped AND the pooled rows it
    was computed beside, so the caller can pin the two against each other.

    Only the NEW payload ever has any: `BASE_SHA` predates E-058, so on the
    old payload this is a no-op and the count is 0 — asserted below for the
    same reason the day-matched strip asserts it, because a base payload that
    somehow carried the key would mean the strip was hiding a difference
    rather than removing an addition."""
    got = []
    for lab, e in obj.get("heads", {}).items():
        for scope, blk in e.items():
            if isinstance(blk, dict) and SCOPES_WITH_ROWS in blk:
                if "per_channel" not in blk:
                    continue
                ks = list(blk)
                i = ks.index("per_channel")
                assert i + 1 < len(ks) and ks[i + 1] == "horizon_auc", (
                    f"{lab}/{scope}: `per_channel` is no longer written "
                    f"immediately before `horizon_auc` ({ks}), so the byte "
                    f"strip above — which takes the key's OWN trailing comma "
                    f"and deliberately leaves the preceding line untouched — "
                    f"no longer describes the file. If the key has moved to "
                    f"the END of the block it now needs DAYMATCH's shape "
                    f"instead, and `horizon_auc_daymatched` has lost the "
                    f"position ITS strip depends on")
                got.append((lab, scope, blk.pop("per_channel"),
                            blk[SCOPES_WITH_ROWS]))
    return PER_CHANNEL.sub("", raw), got


def main():
    tmp = tempfile.mkdtemp()
    try:
        f = build_fixture(tmp)                      # the MONTHLY toy
        base = base_copy(tmp)
        raw_new, log_new = run(os.path.join(ML, "rollout_spatial.py"), f,
                               os.path.join(tmp, "new.json"),
                               os.path.join(tmp, "cache_new"),
                               extra=("--no-unpooled-readout",))
        raw_old, _ = run(base, f, os.path.join(tmp, "old.json"),
                         os.path.join(tmp, "cache_old"))
        print(f"1. ran both evaluators on one {f['P']}-pixel monthly ocean: "
              f"{BASE_SHA} wrote {len(raw_old):,} bytes, HEAD wrote "
              f"{len(raw_new):,}")

        txt_new, obj_new, n_new = strip_wall(raw_new)
        txt_old, obj_old, n_old = strip_wall(raw_old)
        assert n_new == n_old == len(obj_new["heads"]) > 0, (n_new, n_old)
        print(f"2. removed exactly {n_new} `wall_s` clock readings from each "
              f"payload (one per head)")

        # ---- 2b/3b. the ONE new key, stripped and then PINNED ------------
        txt_new, dm_new = strip_daymatched(txt_new, obj_new)
        txt_old, dm_old = strip_daymatched(txt_old, obj_old)
        assert not dm_old, ("BASE_SHA's payload carries "
                            "horizon_auc_daymatched — this strip would then "
                            "be hiding a DIFFERENCE, not a new key")
        scored = [(lab, s) for lab, e in obj_new["heads"].items()
                  for s, b in e.items()
                  if isinstance(b, dict) and b.get(SCOPES_WITH_ROWS)]
        assert dm_new and len(dm_new) == len(scored), \
            (f"{len(dm_new)} day-matched AUCs for {len(scored)} scored "
             f"scopes — the key must be on every scope that has rows, so a "
             f"harvest can never find a scope where only the "
             f"cadence-dependent number exists: {sorted(scored)}")
        bad = [(lab, s, v, h) for lab, s, v, h in dm_new if v != h]
        assert not bad, (
            "at MONTHLY the day-matched leads ARE h=1..12, so this key must "
            "be `horizon_auc` recomputed — exactly, not to within a "
            f"rounding step. Mismatches: {bad}")
        print(f"2b. removed exactly {len(dm_new)} `horizon_auc_daymatched` "
              f"values (one per scored scope, {len(scored)} of them, over "
              f"{len(obj_new['heads'])} head(s)) — and every one EQUALS its "
              f"scope's own `horizon_auc`: "
              + ", ".join(f"{lab}/{s} {v:+.3f}" for lab, s, v, _ in dm_new[:4])
              + (" …" if len(dm_new) > 4 else "")
              + ". The other excluded key is stripped and pinned at 2c")

        # ---- 2c/3c. the SECOND new key, stripped and then PINNED ---------
        txt_new, pc_new = strip_per_channel(txt_new, obj_new)
        txt_old, pc_old = strip_per_channel(txt_old, obj_old)
        assert not pc_old, ("BASE_SHA's payload carries per_channel — this "
                            "strip would then be hiding a DIFFERENCE, not a "
                            "new key")
        assert pc_new and len(pc_new) == len(scored), \
            (f"{len(pc_new)} per-channel blocks for {len(scored)} scored "
             f"scopes — the key must be on every scope that has rows, for the "
             f"same reason the day-matched AUC must: a scope where only the "
             f"40-channel POOL exists is a scope whose sst read-out no "
             f"harvest can ask for. `scored` is the right denominator "
             f"because a pooled row exists at h only if some channel scored "
             f"at h, and skill_block emits the key whenever any channel has "
             f"rows: {sorted(scored)}")
        bad_n, bad_hull, bad_aud, n_row, dev_aud = [], [], [], 0, 0.0
        for lab, s, per, rows in pc_new:
            au = obj_new["heads"][lab]["audit"]
            assert list(per) == au["channels"], (
                f"{lab}/{s}: per_channel is keyed {list(per)} but the "
                f"artefact's own channel list is {au['channels']} — the "
                f"names must be the TENSOR's, from ck['chan'], or the rows "
                f"below are being compared against the wrong channel")
            for r in rows:
                mine = [pr for c in per.values() for pr in c
                        if pr["h"] == r["h"]]
                n_row += len(mine)
                tot = sum(pr["n"] for pr in mine)
                if tot != r["n"]:
                    bad_n.append((lab, s, r["h"], r["n"], tot))
                v = [pr["msss_clim"] for pr in mine]
                if not min(v) <= r["msss_clim"] <= max(v):
                    bad_hull.append((lab, s, r["h"], r["msss_clim"],
                                     min(v), max(v)))
            if s != "corridor":
                continue
            for ci, c in enumerate(per):
                for pr in per[c]:
                    a_ = au["per_channel_msss_clim_corridor"][pr["h"] - 1][ci]
                    if a_ is None or abs(a_ - pr["msss_clim"]) > 1e-3:
                        bad_aud.append((lab, c, pr["h"], pr["msss_clim"], a_))
                    else:
                        dev_aud = max(dev_aud, abs(a_ - pr["msss_clim"]))
        assert not bad_n, (
            "a pooled row's `n` is its channel columns' `n` added up: "
            "`accumulate` fills the pooled sums and the per-channel sums from "
            "the SAME [n_pixels, C] arrays on the same early-exit branch, so "
            "a block whose counts do not sum to the pooled count was "
            f"accumulated over different cells than the row it sits beside. "
            f"(lab, scope, h, pooled n, sum of channel n): {bad_n}")
        assert not bad_hull, (
            "tests/test_per_channel_skill.py check 2 proves the pooled "
            "msss_clim IS the climatology-mass-weighted mean of the "
            "per-channel ones, to 1.11e-16. Its weights are the channels' own "
            "`mse_c`, which the artefact does not publish — so what a reader "
            "of this FILE can check is that same relation's inescapable "
            "consequence: a mean with positive weights never leaves the "
            "convex hull of its terms. A pooled value outside its own "
            "channels' [min, max] is not a pooling of them. "
            f"(lab, scope, h, pooled, min, max): {bad_hull}")
        assert not bad_aud, (
            "for the CORRIDOR scope the artefact already carried a "
            "per-channel msss_clim decomposition before E-058: the E-026b "
            "audit block, accumulated independently in `aud['ch_m']/['ch_c']` "
            "and rounded to the same three places. The new key must agree "
            "with it channel for channel and horizon for horizon. The two "
            "divide by `n` at different points — `1 - (m/n)/(c/n)` against "
            "`1 - m/c` — so they can differ by at most one unit in the last "
            "PUBLISHED place, which is the tolerance; anything larger is a "
            "different accumulation, not a rounding step. "
            f"(lab, channel, h, per_channel, audit): {bad_aud}")
        print(f"2c. removed exactly {len(pc_new)} `per_channel` blocks (one "
              f"per scored scope, {len(scored)} of them, over "
              f"{len(obj_new['heads'])} head(s)) holding {n_row} channel rows "
              f"— and every block is CONSISTENT with the pooled row it was "
              f"computed beside: the channel `n`s sum to the pooled `n` "
              f"EXACTLY, the pooled `msss_clim` lies inside its own "
              f"channels' [min, max] at every horizon, and on the corridor "
              f"scope the values equal the E-026b audit block's independent "
              f"per-channel decomposition (max |Δ| {dev_aud:.1e}). Nothing "
              f"else is excluded from the byte comparison")

        if txt_new != txt_old:
            a = txt_old.splitlines()
            b = txt_new.splitlines()
            diff = [f"    line {i + 1}:\n      {BASE_SHA}: {x}\n      HEAD:"
                    f"     {y}"
                    for i, (x, y) in enumerate(zip(a, b)) if x != y][:12]
            raise SystemExit(
                "THE MONTHLY ROLL MOVED. Every published corridor AUC came "
                "from the base version; a monthly number that differs makes "
                "the archive incomparable.\n" + "\n".join(diff)
                + (f"\n    ... and {len(a)} vs {len(b)} lines"
                   if len(a) != len(b) else ""))
        print(f"3. the two roll.json payloads are BYTE-IDENTICAL "
              f"({len(txt_new):,} bytes, {len(txt_new.splitlines()):,} lines) "
              f"— the monthly path did not move")

        # and the numbers a reader would quote, named rather than implied
        for lab, e in sorted(obj_new["heads"].items()):
            o = obj_old["heads"][lab]
            assert e == o
            print(f"   {lab}: corridor AUC "
                  f"{e['corridor']['horizon_auc']:+.3f} · window "
                  f"{e['window']['horizon_auc']:+.3f} · h1-3 r "
                  f"{e['amoc_bands']['h1-3']['r']:+.3f} "
                  f"(n={e['amoc_bands']['h1-3']['n']}) · long r_trained "
                  f"{e['long']['r_trained']} — identical in both")

        # the monthly artefact gained NO keys: `cadence` is written only
        # where a step is not a month, precisely so this test can be byte-wise
        assert "cadence" not in obj_new, \
            "a `cadence` block was written into a MONTHLY artefact — that is " \
            "a new key in every archived comparison"
        assert obj_new["gate"] == {"pass": None, "skipped": True}, \
            obj_new["gate"]
        assert "starts" not in obj_new, \
            "a `starts` block was written into a MONTHLY artefact rolled " \
            "WITHOUT --starts-per-year — the knob records itself only when " \
            "it was used, precisely so the archive's default gains nothing"
        assert "time axis: monthly" in log_new, log_new[-2000:]
        print("4. the monthly artefact carries no new keys beyond the two "
              "pinned above (no `cadence` block, no `starts` block, gate "
              "block unchanged), and the log names the axis it detected: "
              "monthly")

        # ---- 5. the day-defined bands reproduce the monthly partition ----
        # The §4.9 exact invariant, checked against the payload rather than
        # against the axis: the band KEYS a monthly roll writes are the ones
        # the archive quotes and the #217 gate is keyed on. `--horizon 3`
        # here means only the first band has ≥8 points, which is why the
        # definition is also checked directly.
        sys.path.insert(0, ML)
        import rollout_spatial as rs                      # noqa: E402
        ax_m = rs.TimeAxis({"months": np.array(f["months"])})
        assert ax_m.bands() == (("h1-3", (1, 2, 3)), ("h4-6", (4, 5, 6)),
                                ("h7-12", tuple(range(7, 13)))), ax_m.bands()
        assert ax_m.daymatched_leads() == tuple(range(1, 13))
        assert [ax_m.band_key(bn, hs) for bn, hs in ax_m.bands()] \
            == ["h1-3", "h4-6", "h7-12"]
        for e in obj_new["heads"].values():
            assert set(e["amoc_bands"]) <= {"h1-3", "h4-6", "h7-12"}, \
                e["amoc_bands"]
        assert set(rs.GATE_REF["bands"]) == {"h1-3", "h4-6", "h7-12"}
        print("5. the DAY-DEFINED bands (edges "
              + "/".join(f"{v:g}" for v in rs.BAND_EDGE_DAYS)
              + " d) cut the monthly axis into exactly h1-3 / h4-6 / h7-12 — "
                "the literal they replaced, name for name and step for step — "
                "and the day-matched leads are h=1..12, which is why 2b holds")
        print("\nmonthly roll bit-identity: all 7 checks hold ✓")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    main()

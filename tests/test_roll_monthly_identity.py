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

TWO FIELDS ARE EXCLUDED, and the second one is a DECISION worth reading.

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
    scope's own `horizon_auc` EXACTLY (check 3b), which is a stronger
    statement than key-identity could make, because key-identity would let
    the new key hold anything at all.
  * and the exclusion is COUNTED (one per scored scope per head, and zero
    scopes may be missed), so the day someone adds a second key here, this
    test fails rather than widening.
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
              + ". Nothing else is excluded from the byte comparison")

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
        print("4. the monthly artefact carries no new keys beyond the one "
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
        print("\nmonthly roll bit-identity: all 6 checks hold ✓")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    main()

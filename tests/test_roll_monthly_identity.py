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

ONE FIELD IS EXCLUDED, and only one: `heads.*.wall_s`, which is a reading of
the clock rather than a result. It is removed from BOTH payloads and its
absence is asserted, so the exclusion cannot quietly widen.
"""
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile

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


def run(script, f, out, cache):
    os.makedirs(cache, exist_ok=True)
    env = dict(os.environ, PYTHONPATH=ML + os.pathsep
               + os.environ.get("PYTHONPATH", ""))
    cmd = [sys.executable, "-u", script,
           "--x", f["x"], "--npz-small", f["npz"], "--z", f["z"],
           "--ckpt", f["ckpt"], "--out", out, "--horizon", "3",
           "--long-start", "1991-12", "--long-months", "16",
           "--future-months", "5", "--cache-dir", cache,
           "--no-gate", "--heads", *f["heads"]]
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=1800,
                       env=env, cwd=ROOT)
    if r.returncode != 0:
        print(r.stdout[-3000:])
        print(r.stderr[-3000:])
        raise SystemExit(f"{os.path.basename(script)} failed on the monthly "
                         f"toy (rc {r.returncode})")
    return open(out).read(), r.stdout


WALL = re.compile(r'^\s*"wall_s":.*\n', re.M)


def strip_wall(raw):
    """Drop the clock reading, and prove it was the only thing dropped."""
    obj = json.loads(raw)
    n = 0
    for e in obj.get("heads", {}).values():
        n += e.pop("wall_s", None) is not None
    return WALL.sub("", raw), obj, n


def main():
    tmp = tempfile.mkdtemp()
    try:
        f = build_fixture(tmp)                      # the MONTHLY toy
        base = base_copy(tmp)
        raw_new, log_new = run(os.path.join(ML, "rollout_spatial.py"), f,
                               os.path.join(tmp, "new.json"),
                               os.path.join(tmp, "cache_new"))
        raw_old, _ = run(base, f, os.path.join(tmp, "old.json"),
                         os.path.join(tmp, "cache_old"))
        print(f"1. ran both evaluators on one {f['P']}-pixel monthly ocean: "
              f"{BASE_SHA} wrote {len(raw_old):,} bytes, HEAD wrote "
              f"{len(raw_new):,}")

        txt_new, obj_new, n_new = strip_wall(raw_new)
        txt_old, obj_old, n_old = strip_wall(raw_old)
        assert n_new == n_old == len(obj_new["heads"]) > 0, (n_new, n_old)
        print(f"2. removed exactly {n_new} `wall_s` clock readings from each "
              f"payload (one per head) — nothing else is excluded")

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
        assert "time axis: monthly" in log_new, log_new[-2000:]
        print("4. the monthly artefact carries no new keys (no `cadence` "
              "block, gate block unchanged), and the log names the axis it "
              "detected: monthly")
        print("\nmonthly roll bit-identity: all 4 checks hold ✓")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    main()

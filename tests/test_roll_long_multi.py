#!/usr/bin/env python3
"""Hindcasts from SEVERAL context ends — the calendar-vs-context discriminator.

WHY THIS EXISTS (2026-08-22). Every head's unforced future roll mode-locks to
the calendar: the gate head peaks every 12 months, the nolonhold pair every
36, with the peaks pinned to the same calendar months and the two seeds
phase-identical. From ONE context end that observation has two explanations
and no way to choose between them —

  * the model REPLAYS THE CALENDAR: its phase is a function of the season
    token it is fed, so any roll peaks in the same months whatever state it
    started from; or
  * the model's phase is SELECTED BY ITS STATE: the context end decides where
    in the cycle the trajectory starts, and the calendar alignment of one roll
    is a property of that one start.

Rolling from SEVERAL ends separates them, and nothing else does. So
`--long-start` takes a list, and this test pins the two properties that make
the answer readable: the extra rolls are REAL rolls of the right length from
the right rows, and the first one is UNCHANGED — because if adding a context
end perturbed the original hindcast, every archived `long` block would become
incomparable with the new ones and the discriminator would be comparing its
own artefact.

    python3 tests/test_roll_long_multi.py

No GPU: the same 79-pixel synthetic monthly ocean the identity suite rolls.
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
sys.path.insert(0, ML)
from test_rollout_spatial import build_fixture, K                # noqa: E402
from test_roll_monthly_identity import WALL                      # noqa: E402

N_LONG, N_FUT, HORIZON = 16, 4, 3
BASE = "1991-12"                      # the fixture's own default context end
EXTRA = ["1992-04", "1992-08"]        # two more, inside the same record


def run(f, out, cache, long_start):
    os.makedirs(cache, exist_ok=True)
    env = dict(os.environ, PYTHONPATH=ML + os.pathsep
               + os.environ.get("PYTHONPATH", ""))
    cmd = [sys.executable, "-u", os.path.join(ML, "rollout_spatial.py"),
           "--x", f["x"], "--npz-small", f["npz"], "--z", f["z"],
           "--ckpt", f["ckpt"], "--out", out, "--horizon", str(HORIZON),
           "--long-start", long_start, "--long-months", str(N_LONG),
           "--future-months", str(N_FUT), "--cache-dir", cache,
           "--no-gate", "--heads", *f["heads"]]
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=1800,
                       env=env, cwd=ROOT)
    if r.returncode != 0:
        print(r.stdout[-3000:])
        print(r.stderr[-3000:])
        raise SystemExit(f"rollout_spatial.py failed (rc {r.returncode})")
    return open(out).read(), r.stdout


def main():
    tmp = tempfile.mkdtemp()
    try:
        f = build_fixture(tmp)
        import rollout_spatial as rs
        ax = rs.TimeAxis(np.load(f["npz"], allow_pickle=False))

        one_txt, _ = run(f, os.path.join(tmp, "one.json"),
                         os.path.join(tmp, "c1"), BASE)
        many_txt, log = run(f, os.path.join(tmp, "many.json"),
                            os.path.join(tmp, "c2"),
                            ",".join([BASE] + EXTRA))
        one, many = json.loads(one_txt), json.loads(many_txt)

        # ---- 1. the FIRST hindcast did not move --------------------------
        # Byte comparison of the `long` block itself, not a field-by-field
        # walk: a re-ordered key or a 0.643 that became 0.6430000000000001
        # would pass the walk and break every comparison against the archive.
        def long_bytes(txt, lab):
            m = re.search(r'\n   "long": \{.*?\n   \}', txt, re.S)
            assert m, "the `long` block is not where this test looks"
            return m.group(0)
        for lab in one["heads"]:
            a_ = json.dumps(one["heads"][lab]["long"], sort_keys=False)
            b_ = json.dumps(many["heads"][lab]["long"], sort_keys=False)
            assert a_ == b_, (lab, a_[:200], b_[:200])
        assert long_bytes(one_txt, 0) == long_bytes(many_txt, 0)
        n_heads = len(one["heads"])
        print("1. the FIRST context end's `long` block is byte-identical with "
              "and without the extra ends, on all %d head(s) — an added "
              "hindcast cannot perturb the one every archived roll carries"
              % n_heads)

        # ---- 2. one long_multi entry per EXTRA end, in order -------------
        for lab, e in many["heads"].items():
            lm = e["long_multi"]
            assert len(lm) == len(EXTRA), (lab, len(lm))
            assert [b["context_end"] for b in lm] == EXTRA, lm
            assert e["long"]["context_end"] == BASE
            assert "long_multi" not in one["heads"][lab], \
                "a single-start run must write NO long_multi key at all"
        print("2. `long_multi` carries exactly the %d EXTRA ends %s, in order, "
              "and the single-start run writes no such key" % (len(EXTRA), EXTRA))

        # ---- 3. they are REAL rolls: right length, right rows ------------
        for lab, e in many["heads"].items():
            for b in e["long_multi"]:
                r0 = ax.row_of_label(b["context_end"])
                assert len(b["sv_des"]) == N_LONG == len(b["roll_ym"]), b
                assert b["roll_ym"] == [ax.label_of_row(r0 + 1 + i)
                                        for i in range(N_LONG)], b
                assert all(np.isfinite(b["sv_des"])), b["context_end"]
                assert set(b) == set(e["long"]), "field sets must match"
            svs = [tuple(b["sv_des"]) for b in e["long_multi"]]
            svs.append(tuple(e["long"]["sv_des"]))
            assert len(set(svs)) == len(svs), \
                "two context ends produced IDENTICAL trajectories — the roll " \
                "is not reading its start row"
        print("3. every extra hindcast is a real %d-step roll from its OWN "
              "row (labels advance one axis row per step), carries the same "
              "field set as `long`, and no two starts produced the same "
              "trajectory" % N_LONG)

        # ---- 4. the progress plan counts them ----------------------------
        # The ETA is the only thing a watcher has during a 14-hour roll, and a
        # plan that under-counts by 5x reads as a job falling behind.
        m1 = re.search(r"(\d+) scored roll steps \+ (\d+) hindcast", log)
        assert m1 and int(m1.group(2)) == N_LONG * (1 + len(EXTRA)), \
            (m1.group(0) if m1 else log[-800:])
        print("4. the step plan says %s hindcast steps — %d ends x %d steps — "
              "so the progress bar and the ETA count the extra rolls rather "
              "than discovering them" % (m1.group(2), 1 + len(EXTRA), N_LONG))

        # ---- 5. an unusable label is SKIPPED with a reason, never fatal --
        out3 = os.path.join(tmp, "skip.json")
        txt3, log3 = run(f, out3, os.path.join(tmp, "c3"),
                         ",".join([BASE, "1899-01", ax.labels[0]]))
        many3 = json.loads(txt3)
        for lab, e in many3["heads"].items():
            assert e["long"]["context_end"] == BASE
            assert "long_multi" not in e, e.get("long_multi")
        assert "no axis row for that label" in log3, log3[-1500:]
        assert "rows of history" in log3 and f"K={K}" in log3, log3[-1500:]
        print("5. `1899-01` (no axis row) and `%s` (row 0, less than K=%d "
              "rows of history) are SKIPPED with a printed reason and the run "
              "still finishes — a bad label costs a hindcast, never the job"
              % (ax.labels[0], K))

        # ---- 6. and nothing else in the artefact moved -------------------
        a_s, b_s = WALL.sub("", one_txt), WALL.sub("", many_txt)
        for lab in one["heads"]:
            for k in ("gate", "corridor", "window", "amoc_bands", "future"):
                assert json.dumps(one["heads"][lab][k]) == \
                    json.dumps(many["heads"][lab][k]), (lab, k)
        assert len(b_s) > len(a_s), "long_multi should ADD bytes"
        print("6. every other block — gate, corridor, window, amoc_bands, "
              "future — is identical between the two runs; the extra ends add "
              "%d bytes and change nothing else"
              % (len(b_s) - len(a_s)))

        print("\nmulti-context-end hindcasts: all 6 checks hold ✓")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    main()

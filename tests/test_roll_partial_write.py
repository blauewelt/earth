#!/usr/bin/env python3
"""The roll's result file exists BEFORE the roll ends — and the final one is
byte-for-byte the file the archive already has.

#433 (E-044b-roll, the first pentad corridor AUC) computed every number anyone
was waiting for — `chan_skill`, `horizon_auc`, `horizon_auc_daymatched`, the
AMOC bands, the gate — inside the first ~2 h of a ~13 h job, and then held
them in memory behind 2,922 long/future roll steps (87% of the wall clock)
with nothing readable outside the box. A timeout, a token expiry or a
cancellation in hours 3-13 would have spent all of it and archived nothing.
Chris, 2026-08-22: *"Otherwise we wait 10h, spend money, and then have
nothing."* — ml/CLAUDE.md §5.25.

So `rollout_spatial.write_results` rewrites the file at every phase boundary.
Two properties have to hold at once, and they pull against each other:

  * a PARTIAL write must be marked, so no reader mistakes half a roll for a
    roll, and must be ATOMIC, so the live publisher (which copies the file
    every ~2.5 min) can never catch it half-written;
  * the FINAL write must be EXACTLY what this file has always written — same
    `indent=1`, no trailing newline, no extra key — because
    tests/test_roll_monthly_identity.py compares the whole artefact against an
    archived base sha byte for byte, and a marker left behind would move every
    published monthly number's file.

This test unit-tests the helper against both, then reads the SOURCE of the
three files that have to agree about it: the evaluator writes at the four
stages, the publisher ships the file, and the run script refuses a result that
still says it is in progress.

    python3 tests/test_roll_partial_write.py

Stdlib + numpy, no GPU, seconds.
"""
import json
import os
import re
import shutil
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
ML = os.path.join(ROOT, "ml")
sys.path.insert(0, ML)

SRC = os.path.join(ML, "rollout_spatial.py")
PUB = os.path.join(ROOT, "scripts", "publish_live_metrics.sh")
RUN = os.path.join(ROOT, "scripts", "sroll_run.sh")

# The shape of a real roll json, small enough to read: nested dicts, a float
# that must not be reformatted, and a key order the writer must not disturb.
RESULTS = {
    "data": "family4_na025_pentad_r2.npz",
    "horizon": 73,
    "gate": {"pass": None, "skipped": True},
    "heads": {
        "s1_s0": {"meta": {"file": "e017_u1_s0.pt", "stencil": 1},
                  "corridor": {"horizon_auc": 0.6430000000000001,
                               "chan_skill": [{"h": 1, "msss_clim": -0.25}]},
                  "amoc_bands": {"h1-18_5-90d": {"r": 0.31, "n": 40}},
                  "wall_s": 812.4},
    },
}


def load_write_results():
    """`rollout_spatial.write_results`, however it can be had.

    Importing the module is the honest way and is what happens wherever the
    evaluator's own dependencies are installed. Where they are not — this test
    is meant to run in seconds on a laptop, and `import rollout_spatial` pulls
    in torch — the function is exec'd out of the shipped source instead. That
    fallback is not a mock: it is the same bytes, and it doubles as a check
    that the one function which must run at every phase boundary of a
    thirteen-hour job depends on nothing but the standard library.
    """
    try:
        import rollout_spatial as rs
        return rs.write_results, "imported ml/rollout_spatial.py"
    except ImportError as e:
        src = open(SRC).read()
        m = re.search(r"\ndef write_results\(.*?\n(?=\ndef )", src, re.S)
        assert m, "write_results is not a module-level function any more"
        ns = {"json": json, "os": os}
        exec(m.group(0), ns)
        return ns["write_results"], (
            "exec'd write_results out of the source (%s not available here), "
            "with only json+os in scope" % e.name)


def main():
    write_results, how = load_write_results()
    print("0. write_results: %s" % how)

    tmp = tempfile.mkdtemp()
    try:
        out = os.path.join(tmp, "sub", "rollout_spatial.json")

        # ---- 1. the final write is the CURRENT write, byte for byte -------
        write_results(out, RESULTS)
        got = open(out, "rb").read()
        want = json.dumps(RESULTS, indent=1).encode()      # no trailing "\n"
        assert got == want, (got[:200], want[:200])
        assert not got.endswith(b"\n"), (
            "the final artefact grew a trailing newline — "
            "tests/test_roll_monthly_identity.py compares whole files")
        assert "in_progress" not in json.loads(got)
        assert list(json.loads(got)) == list(RESULTS), "key order moved"
        print("1. write_results(path, results) writes EXACTLY "
              "json.dump(results, f, indent=1) — %d bytes, no trailing "
              "newline, no extra key, key order untouched" % len(got))

        # ---- 2. a partial write is MARKED, and marked last ----------------
        mark = {"head": "s1_s0", "head_i": 1, "heads": 4, "stage": "scored",
                "at": "2026-08-22T09:14:00Z"}
        write_results(out, RESULTS, partial=mark)
        d = json.loads(open(out).read())
        assert d["in_progress"] == mark, d.get("in_progress")
        assert list(d) == list(RESULTS) + ["in_progress"], list(d)
        assert {k: v for k, v in d.items() if k != "in_progress"} == RESULTS
        assert "in_progress" not in RESULTS, (
            "write_results MUTATED the caller's dict — the next write would "
            "carry a stale marker, and the final one would carry it forever")
        print("2. a partial write carries the given dict under `in_progress` "
              "(appended LAST, so a reader diffing against the final file "
              "sees one added key), everything else identical, and the "
              "caller's own results dict is not touched")

        # ---- 3. atomic: nothing half-written, no .tmp left behind ---------
        before = sorted(os.listdir(os.path.dirname(out)))
        write_results(out, dict(RESULTS, horizon=12))
        after = sorted(os.listdir(os.path.dirname(out)))
        assert before == after == ["rollout_spatial.json"], (before, after)
        assert not os.path.exists(out + ".tmp"), "a .tmp survived the write"
        assert json.load(open(out))["horizon"] == 12, "file not replaced"
        # the temp file must be a SIBLING — os.replace is only atomic within
        # one filesystem, and /tmp is routinely a different one
        src = open(SRC).read()
        assert 'tmp = path + ".tmp"' in src, (
            "the temp file is no longer a sibling of the target; os.replace "
            "across filesystems is not atomic and raises")
        assert "os.replace(tmp, path)" in src
        print("3. the write is atomic — bytes land in a SIBLING `.tmp` and "
              "os.replace swaps them in; a pre-existing file is replaced and "
              "no .tmp is left behind (dir holds exactly %r)" % after)

        # ---- 4. a partial file, then a final one, over the same path ------
        write_results(out, RESULTS, partial=dict(mark, stage="long"))
        assert "in_progress" in json.load(open(out))
        write_results(out, RESULTS)
        assert open(out, "rb").read() == want, (
            "a final write after partial ones did not restore the exact "
            "archived bytes")
        print("4. partial → partial → final over one path ends at the exact "
              "bytes of check 1: the markers leave nothing behind")

        # ---- 5. the evaluator writes at every phase boundary --------------
        m = re.search(r"\ndef main\(\):\n(.*)\n\nif __name__", src, re.S)
        assert m, "main() is no longer where this test looks"
        body = m.group(1)
        calls = re.findall(r"write_results\(", body)
        assert len(calls) >= 4, (
            "main() calls write_results %d times — the roll writes at the "
            "start and at every phase boundary of every head" % len(calls))
        for stage in ("started", "scored", "long", "future"):
            assert f'"{stage}"' in body, (
                f"main() never names the stage {stage!r}; a phase that does "
                f"not write is a phase whose numbers can be lost")
        # the LAST write in main is the unmarked one — the roll's own
        # statement that it finished
        assert re.search(r"write_results\(a\.out, results\)\s*\n\s*print\("
                         r"f\"wrote \{a\.out\}", body), (
            "the final write is no longer a bare write_results(a.out, "
            "results); the artefact would keep its in_progress marker")
        print("5. rollout_spatial.main() calls write_results %d times, names "
              "the stages started/scored/long/future, and its final call is "
              "the unmarked one" % len(calls))

        # ---- 6. the publisher ships the file ------------------------------
        pub = open(PUB).read()
        assert "rollout_spatial.json" in pub, (
            "publish_live_metrics.sh does not name rollout_spatial.json — "
            "the roll would write incrementally to a disk nobody can read")
        assert re.search(r'^R=ml/runs/actions/rollout_spatial\.json$', pub,
                         re.M), pub
        assert re.search(r'\[ -s "\$R" \] \|\| exit 0', pub), (
            "the early-exit guard does not consider $R, so a roll with no "
            "metrics.jsonl yet would publish nothing")
        assert re.search(r'\[ -s "\$R" \] && cp "\$R" '
                         r'"\$DIR/rollout_spatial\.json"', pub)
        assert "in_progress" in pub, (
            "the header does not say the published file can be partial")
        print("6. publish_live_metrics.sh copies rollout_spatial.json onto "
              "ml-live-<n> beside metrics.jsonl and phase.json, under the "
              "same TOGETHER semantics, and its guard lets the roll file "
              "alone trigger a publish")

        # ---- 7. the run script refuses a partial artefact -----------------
        run = open(RUN).read()
        assert re.search(r'assert "in_progress" not in d', run), (
            "sroll_run.sh does not assert the absence of in_progress — a "
            "roll killed mid-flight would be published as a result")
        assert "did not reach its final write" in run
        print("7. sroll_run.sh asserts `in_progress` is ABSENT from the final "
              "file, so a partial artefact cannot be published as a result")

        print("\nincremental roll writes: all 7 checks hold ✓")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""UNPOOLED IS THE VERDICT — the wiring that makes that true, pinned.

Chris, 2026-08-21: *"we should not do pooled evals anywhere"* / *"move from
pooled to unpooled when running evals."*

THE JUSTIFICATION IS THE MECHANISM, NOT A NUMBER. Geostrophic transport at
26.5N is the east-minus-west contrast ACROSS the section, and a spatial mean
annihilates exactly that contrast; `ml/project_amoc.py` measures the damage on
the run-62 cache — z along the section correlates r 0.99 at one cell, 0.88 at
five and 0.35 at eighty, so the section mean averages ~2.5 effective
independent pixels of 265. The often-quoted +0.031 (pooled ridge 0.660 →
unpooled head 0.691, ml/EXPERIMENTS.md) is a TWO-INTERVAL comparison, which
ml/CLAUDE.md §3 forbids as a way to compare two probes; a paired test could not
be run because no `probe_kfold.json` in any of the 183 archived bundles carries
`pred`/`target_sv`/`years`. So that number is not evidence here and this file
does not assert anything about it. What it pins is the machinery that lets the
FIRST run under the new default settle the question properly.

Six guards, all CPU:

  1. `.github/workflows/ml-train.yml` defaults `head_probe` to "true" — the
     single documented cause of every missing head number in the current wave
     (#414, #415, #416, #419). probe_head produced output in 3 of 183 archived
     bundles.
  2. ...and the input count is still <= 25. workflow_dispatch caps there, this
     file sits exactly at the ceiling, and a 26th input does not fail
     gracefully: GitHub refuses to parse the file and EVERY dispatch in the
     repo 422s. The fix for this ruling had to be a CHANGED DEFAULT, never a
     new input.
  3. a recipe setting `head_probe` is actually READ — an unread `$RECIPE_*`
     key is refused by design, and a key that appears to apply and does
     nothing is the failure mode recipes exist to remove.
  4. `ml/probe_kfold.py` dumps `pred`/`target_sv`/`years` for EVERY target it
     scores, not just RAPID. This is what makes pooled-vs-unpooled settleable
     with `scripts/paired_probe.py` instead of another two-interval
     comparison.
  5. `ml/probe_head.py` has an unpooled path for every target `probe_kfold`
     scores (it was hardcoded to RAPID), refuses a target whose section is
     outside the window rather than probing the wrong latitude, and can fit
     the MATCHED UNPOOLED WIND BAR.
  6. the reporting surfaces — `ml/make_table.py` and `scripts/sweep_table.mjs`
     — emit the unpooled column as the headline, label the pooled one
     `legacy_pooled_*`, and never bar an unpooled number with a pooled one.

    python3 tests/test_unpooled_verdict.py
"""
import contextlib
import glob
import io
import json
import os
import re
import subprocess
import sys
import tempfile

import numpy as np
import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
ML = os.path.join(ROOT, "ml")
WF = os.path.join(ROOT, ".github", "workflows", "ml-train.yml")
PROBES_RUN = os.path.join(ROOT, "scripts", "probes_run.sh")
SWEEP = os.path.join(ROOT, "scripts", "sweep_table.mjs")
sys.path.insert(0, ML)
sys.path.insert(0, HERE)

import probe_head as ph                                      # noqa: E402
import probe_kfold as pk                                     # noqa: E402
import make_table as mt                                      # noqa: E402
import test_probe_kfold_pairing as fixture                   # noqa: E402


def wf_inputs():
    raw = open(WF).read()
    doc = yaml.safe_load(raw)
    # PyYAML resolves the bare key `on:` to the boolean True (YAML 1.1).
    key = True if True in doc else "on"
    return raw, doc, doc[key]["workflow_dispatch"]["inputs"]


# ---------------------------------------------------------------------------
def case1_default():
    _, _, ins = wf_inputs()
    got = ins["head_probe"]["default"]
    assert got == "true", (
        f"case 1 FAILED: head_probe defaults to {got!r}. This default being "
        f'"false" is the documented cause of every missing head number in the '
        f"current wave — probe_head has produced output in 3 of 183 archived "
        f"bundles, and each of the other 180 ran a pooled read-out and "
        f"reported it as the answer.")
    assert isinstance(got, str), (
        "case 1 FAILED: head_probe's default parsed as a YAML boolean, not a "
        "string. scripts/probes_run.sh compares it with `= \"true\"` in a "
        "shell; an unquoted True would still stringify to 'True' and never "
        "match. Keep it quoted.")
    print(f'1 · workflow head_probe default is {got!r} — the unpooled head '
          f'and its two unpooled bars run on every dispatch ✓')


def case2_input_ceiling():
    raw, doc, ins = wf_inputs()
    n = len(ins)
    assert n <= 25, (
        f"case 2 FAILED: {n} workflow_dispatch inputs. The cap is 25 and a "
        f"26th does not fail gracefully — GitHub refuses to parse the file "
        f"and every dispatch in the repo answers 422, other sessions' jobs "
        f"included. The unpooled ruling had to be a CHANGED DEFAULT.")
    # ...and the knob that DID need a new name went in as a recipe-only key,
    # which is the mechanism this repo provides for exactly this.
    recipe_only = set(re.findall(r"#\s*recipe-only:\s*(\w+)", raw))
    assert "head_targets" in recipe_only, (
        "case 2 FAILED: head_targets is not declared in the RECIPE-ONLY KEYS "
        "block. It is how a dispatch asks for an unpooled read-out on a "
        "target other than RAPID without spending the 26th input that would "
        "take the repo down.")
    consumers = open(PROBES_RUN).read()
    assert "RECIPE_HEAD_TARGETS" in consumers, (
        "case 2 FAILED: nothing reads $RECIPE_HEAD_TARGETS. A recipe key the "
        "job never reads is a setting that appears to apply and does "
        "nothing — refused by scripts/resolve_recipe.sh, and rightly.")
    print(f"2 · {n} inputs (cap 25, at the ceiling); head_targets is a "
          f"recipe-only key and probes_run.sh reads it ✓")


def case3_recipes_read_it():
    recipes = sorted(glob.glob(os.path.join(ML, "recipes", "*.json")))
    assert recipes, "case 3 FAILED: no recipes"
    setters = []
    for p in recipes:
        d = json.load(open(p))
        name = os.path.basename(p)[:-5]
        if "head_probe" not in d:
            raise SystemExit(
                f"case 3 FAILED: recipe {name} does not state head_probe. "
                f"The workflow default now supplies it, but a recipe is the "
                f"record of a configuration that was RUN and 'which read-out "
                f"produced the verdict' is part of that configuration.")
        assert d["head_probe"] == "true", (
            f"case 3 FAILED: recipe {name} sets head_probe={d['head_probe']!r}")
        setters.append(name)
    # ...and the key is genuinely consumed. Run the real resolver rather than
    # asserting about it: an unread key is REFUSED there, so a pass here is
    # the same check the fleet runs at dispatch.
    r = subprocess.run(["bash", "scripts/resolve_recipe.sh",
                        f"recipe:{setters[0]}"],
                       capture_output=True, text=True, cwd=ROOT,
                       env={**os.environ, "GITHUB_ENV": "/dev/null"})
    assert r.returncode == 0, r.stdout + r.stderr
    assert "RECIPE_HEAD_PROBE=true" in r.stdout, (
        "case 3 FAILED: scripts/resolve_recipe.sh did not export "
        "RECIPE_HEAD_PROBE.\n" + r.stdout + r.stderr)
    assert "${RECIPE_HEAD_PROBE:-$IN_HEAD_PROBE}" in open(PROBES_RUN).read(), (
        "case 3 FAILED: probes_run.sh no longer consults "
        "${RECIPE_HEAD_PROBE:-$IN_HEAD_PROBE}, so a recipe could not override "
        "the dispatch input.")
    print(f"3 · all {len(setters)} recipes set head_probe=true, and the real "
          f"resolver exports RECIPE_HEAD_PROBE for probes_run.sh to read ✓")


def case4_arrays_every_target(tmp, data):
    """The paired test's inputs, for every target — not just RAPID."""
    old = (pk.HERE, sys.argv)
    pk.HERE = tmp
    sys.argv = ["probe_kfold.py", "--runs", "codecA", "--data", data]
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            pk.main()
    finally:
        pk.HERE, sys.argv = old
    out = json.load(open(os.path.join(tmp, "runs", "probe_kfold.json")))
    blocks = out["codecA"]
    assert len(blocks) >= 3, f"case 4 FAILED: only scored {sorted(blocks)}"
    for t, b in sorted(blocks.items()):
        for k in ("pred", "target_sv", "years"):
            assert k in b, (
                f"case 4 FAILED: {t} has no {k}. Without it pooled-vs-unpooled "
                f"is untestable on this run's own data and the switch has to "
                f"be argued from two overlapping intervals again — which "
                f"ml/CLAUDE.md §3 says is not a result.")
            assert len(b[k]) == b["n"], (
                f"case 4 FAILED: {t}/{k} has {len(b[k])} rows against n="
                f"{b['n']} — a paired test would line up the wrong months")
        assert b["probe"] == "pooled-ridge", b["probe"]
        w = b.get("wind_only_baseline")
        if w:
            assert w["probe"] == "pooled-wind-only", (
                f"case 4 FAILED: {t}'s wind bar is labelled {w['probe']!r}. "
                f"It reads np.nanmean(tau, axis=1) over the same section, so "
                f"it is POOLED, and a table generator has to be able to see "
                f"that before it puts the bar next to an unpooled head.")
            assert "pred" in w
    print(f"4 · probe_kfold dumps pred/target_sv/years for every scored "
          f"target ({', '.join(sorted(blocks))}), and labels both sides "
          f"pooled ✓")
    return blocks


def case5_head_targets(tmp, data):
    """probe_head has an unpooled path for every target, plus the wind bar."""
    # The head's own section/label lookup must be probe_kfold's, not a copy:
    # two decoders would score the pooled and unpooled numbers on different
    # months while looking directly comparable.
    assert ph.TARGETS is pk.TARGETS, (
        "case 5 FAILED: probe_head defines its own TARGETS. One definition, "
        "or the two read-outs silently disagree about which rows a target "
        "has.")
    assert ph.target_series is pk.target_series

    orig = ph.fold_fit
    ph.fold_fit = lambda *a, **k: orig(*a, **{**k, "steps": 60})
    ph.HERE = tmp
    written = {}
    try:
        for argv, expect in (
                (["--target", "rapid"], "probe_head.json"),
                (["--target", "move"], "probe_head_move.json"),
                (["--target", "fc"], "probe_head_fc.json"),
                (["--raw", "--raw-patch"], "probe_head_raw3x3.json"),
                (["--raw", "--wind-only"], "probe_head_wind.json")):
            sys.argv = (["probe_head.py", "--run", "codecA", "--data", data,
                         "--K", "1", "--head-device", "cpu"] + argv)
            with contextlib.redirect_stdout(io.StringIO()):
                ph.main()
            p = os.path.join(tmp, "runs", "codecA", expect)
            assert os.path.exists(p), (
                f"case 5 FAILED: {argv} wrote no {expect}. The RAPID file "
                f"names are addressed by 183 archived bundles, "
                f"scripts/sweep_table.mjs and ml/make_table.py; a new target "
                f"is a new file BESIDE them, never a rename of one.")
            written[expect] = json.load(open(p))
        # a target whose section is not in this window REFUSES, and says so by
        # name — argmin() clamps to the window edge, so the alternative is a
        # confident number off the wrong latitude
        for bad in ("osnap", "samba"):
            sys.argv = ["probe_head.py", "--run", "codecA", "--data", data,
                        "--K", "1", "--head-device", "cpu", "--target", bad]
            try:
                with contextlib.redirect_stdout(io.StringIO()):
                    ph.main()
            except SystemExit as e:
                assert bad in str(e) and "outside this tensor's window" in str(e), e
            else:
                raise SystemExit(f"case 5 FAILED: --target {bad} did not "
                                 f"refuse on a North Atlantic tensor")
        # --wind-only without --raw would silently select nothing
        sys.argv = ["probe_head.py", "--run", "codecA", "--data", data,
                    "--K", "1", "--head-device", "cpu", "--wind-only"]
        try:
            ph.main()
        except SystemExit as e:
            assert "requires --raw" in str(e), e
        else:
            raise SystemExit("case 5 FAILED: --wind-only without --raw was "
                             "accepted and would have done nothing")
    finally:
        ph.fold_fit = orig

    for fn, o in written.items():
        assert o["pooled"] is False, fn
        for k in ("pred", "target_sv", "years"):
            assert len(o[k]) == o["n"], f"{fn}/{k}"
    assert written["probe_head.json"]["target"] == "rapid"
    assert written["probe_head_move.json"]["target"] == "move"
    assert written["probe_head_wind.json"]["probe"] == "attention-head-wind"
    assert written["probe_head_raw3x3.json"]["probe"] == "attention-head-raw3x3"
    # THE BAR IS FITTED IN THE SAME LOOP AS THE HEAD, so it cannot go missing
    body = open(PROBES_RUN).read()
    for flag in ("--raw --raw-patch", "--raw --wind-only", '--target "$HT"'):
        assert flag in body, (
            f"case 5 FAILED: scripts/probes_run.sh does not run {flag!r}. "
            f"Every dispatch that produces an unpooled codec number must "
            f"produce its unpooled bars in the same step — a bar dispatched "
            f"separately is a bar that goes missing.")
    # ...and the archiver must carry whatever that loop wrote
    arch = open(os.path.join(ROOT, "scripts", "archive_probes.py")).read()
    assert "probe_head*.json" in arch, (
        "case 5 FAILED: scripts/archive_probes.py still enumerates the "
        "probe_head files by hand, so probe_head_wind.json and any "
        "per-target file are dropped from the bundle — an unpooled number "
        "archived without its unpooled bar.")
    wf_raw = open(WF).read()
    assert "ml/runs/actions/probe_head*.json" in wf_raw, (
        "case 5 FAILED: the workflow's upload-artifact list still names the "
        "probe_head files individually")
    print(f"5 · probe_head covers {len(written)} unpooled read-outs "
          f"(rapid/move/fc + raw-3x3 and wind bars), refuses osnap/samba by "
          f"name on this window, and probes_run.sh fits the bars in the same "
          f"loop ✓")
    return written


def case6_tables(tmp, kf_blocks, head_files):
    # ---- ml/make_table.py: it had NO probe_head code path at all ----------
    mt.RUNS = os.path.join(tmp, "runs")
    buf = io.StringIO()
    old = sys.argv
    sys.argv = ["make_table.py", "--runs", "codecA", "--markdown"]
    try:
        with contextlib.redirect_stdout(buf):
            mt.main()
    finally:
        sys.argv = old
    txt = buf.getvalue()
    assert "UNPOOLED head r" in txt, (
        "case 6 FAILED: ml/make_table.py emits no unpooled column. It is the "
        "surface that assembles LEADERBOARD.md's master table, and until "
        "2026-08-21 it had no probe_head code path at all — so the one "
        "read-out §3 calls the verdict was invisible to it.\n" + txt)
    for col in ("legacy_pooled r", "legacy_pooled wind",
                "legacy_pooled stage2 r"):
        assert col in txt, f"case 6 FAILED: no {col!r} column\n{txt}"
    assert "raw3x3 bar" in txt and "wind bar" in txt, (
        "case 6 FAILED: the unpooled bars are not beside the unpooled "
        "number, so a reader has to know which of two bars is the matched "
        "one.\n" + txt)
    # the headline really is the head's number, not the pooled one
    hr = head_files["probe_head.json"]["r_kfold_deseas"]
    kr = kf_blocks["rapid"]["r_kfold_deseas"]
    assert hr != kr, "fixture degenerate: pooled and unpooled agree exactly"
    body = [l for l in txt.splitlines() if l.startswith("| codecA")]
    assert body, txt
    cells = [c.strip() for c in body[0].strip("|").split("|")]
    assert f"{hr:+.3f}" in cells, (
        f"case 6 FAILED: the unpooled {hr:+.3f} is not in the row\n{body[0]}")
    assert cells.index(f"{hr:+.3f}") < cells.index(f"{kr:+.3f}"), (
        "case 6 FAILED: the pooled number is printed before the unpooled "
        "one. Column order is the claim.\n" + body[0])
    assert "COVERAGE:" in txt and "has NOT been copied" in txt, (
        "case 6 FAILED: the table does not say that a blank unpooled cell is "
        "UNAVAILABLE. A back-filled pooled number in an unpooled column is "
        "worse than a gap, because a gap can be noticed.\n" + txt)

    # a run with NO head must show a dash, never the pooled number
    empty = os.path.join(tmp, "runs", "codecB")
    for f in glob.glob(os.path.join(empty, "probe_head*.json")):
        os.remove(f)
    buf = io.StringIO()
    sys.argv = ["make_table.py", "--runs", "codecB", "--markdown"]
    try:
        with contextlib.redirect_stdout(buf):
            mt.main()
    finally:
        sys.argv = old
    row = [l for l in buf.getvalue().splitlines() if l.startswith("| codecB")][0]
    cells = [c.strip() for c in row.strip("|").split("|")]
    assert cells[5] == "—", (
        f"case 6 FAILED: a run with no probe_head shows {cells[5]!r} in the "
        f"unpooled column instead of a dash\n{row}")

    # ---- scripts/sweep_table.mjs -----------------------------------------
    src = open(SWEEP).read()
    assert "probe_head" in src, (
        "case 6 FAILED: scripts/sweep_table.mjs never reads probe_head.json")
    for token in ("legacyPooledStage2", "legacyPooledBar", "legacyPooledCodec"):
        assert token in src, f"case 6 FAILED: sweep_table has no {token}"
    assert "NO MATCHED BAR" in src, (
        "case 6 FAILED: sweep_table.mjs has no gate for the mismatched case. "
        "When a bundle carries an unpooled head and only a POOLED bar, the "
        "verdict must be WITHHELD — barring an unpooled number with a pooled "
        "one measures the read-out and reports it as the codec.")
    assert "temporal.py:2349" in src, (
        "case 6 FAILED: sweep_table.mjs does not say WHY its stage-2 column "
        "is legacy. It is `hid[:, -1].mean(0)` — section-pooled — and a "
        "reader who does not know that will read it as a verdict.")
    r = subprocess.run(["node", "--check", SWEEP], capture_output=True,
                       text=True)
    assert r.returncode == 0, r.stdout + r.stderr
    print("6 · make_table emits the unpooled headline with its bars beside "
          "it, dashes an absent one, labels the pooled columns "
          "legacy_pooled_*; sweep_table withholds a verdict rather than "
          "barring unpooled with pooled ✓")


def main():
    case1_default()
    case2_input_ceiling()
    case3_recipes_read_it()
    tmp = tempfile.mkdtemp(prefix="unpooled_")
    # 60 months = 5 year-blocks. Same fixture the pairing test uses, at a
    # length the attention head fits in seconds on CPU: this pins the WIRING
    # (sections, labels, file names, columns), never the values.
    data = fixture.build(tmp, T=60)
    kf = case4_arrays_every_target(tmp, data)
    heads = case5_head_targets(tmp, data)
    case6_tables(tmp, kf, heads)
    print("\nOK — unpooled is the verdict, pooled is a labelled legacy "
          "comparable, and both sides of a comparison switch together.")


if __name__ == "__main__":
    main()

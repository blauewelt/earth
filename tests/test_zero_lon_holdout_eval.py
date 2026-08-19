#!/usr/bin/env python3
"""The EVAL path under a codec that holds out NO longitude (`holdout_lon 0,0`).

`tests/test_lon_holdout_optional.py` pins the TRAINING half of Chris's
2026-08-19 decision (hold out years only) and pins that `"0,0"` is a spec the
twelve `float()` readers can parse. It says nothing about what those readers
then DO with an empty block, and that is the question this file answers,
because E-043 has already put three no-holdout codecs into flight — #416
(E-043a monthly f3 codec, no lon holdout, landed), #415 (E-043e pentad f4r2
codec, all longitude columns) and #419 (E-043f daily codec, all longitude
columns) — and every eval that scores them takes a branch nobody had executed.

The specific worry, flagged as UNVERIFIED by the session of 2026-08-19: under
`holdout_lon "0,0"` the per-scope split `rollout_spatial.py` added at ~line 706
has a **ZERO-PIXEL `_holdlon` child**. Whether that child divides by zero,
writes NaN into `roll_*.json` (forbidden, ml/CLAUDE.md §5.22) or crashes an
hours-long eval was unknown. Same question for every other consumer of
`x_hold` in the eval path: `recon_eval.py` / `recon_decoder.py` score a
`heldout_lons` split off an empty pixel selection, and `probe_kfold.py`,
`probe_head.py`, `temporal.py` feed the mask to the anomaly transform.

The ANSWER, measured by this file rather than argued: **nothing breaks.**
`accumulate()` takes its `n == 0` early return, `skill_block()` returns rows
`[]` and omits `horizon_auc`, `recon_eval.score()` skips every channel under
its 30-sample floor and returns `{}`. Omission, not NaN — which is §5.22's
required behaviour, arrived at by construction rather than by design. So this
is a PINNING test for behaviour that already held, plus two small reporting
fixes made in the same commit, which it also pins:

  * every scope block now carries `n_px`, so `chan_skill: []` can be told
    apart from a scope that had pixels and scored nothing; a zero-pixel scope
    also carries `empty` saying which of the two it is;
  * `holdout_lon` carries `any`, and its `note` / `excluded_from` / the log
    line follow it, instead of asserting "these columns were held out of
    training" over an empty set (§5.24: a stale reference table is worse than
    none — it gets checked, it matches nothing, and the run takes the blame).

Check 5 is a different animal and is here because it shares the fixture: it is
a TRIPWIRE for the pentad cadence, asserting that `rollout_spatial.py`'s
`month_index` / `ym_to_r` dictionaries are NOT injective on a family-4 months
array. They are not, by 6.09 to 1, and that is a DO-NOT-DISPATCH fact for any
`sroll:` at pentad — see ml/recipes/xl144-zn-pentad-nolonhold.json's
`_description`. The test asserts the defect so that whoever fixes it deletes a
failing test rather than wondering whether anyone knew.

    python3 tests/test_zero_lon_holdout_eval.py

No GPU, no real tensor: it reuses tests/test_rollout_spatial.py's toy ocean
(now parameterised by `holdout_lon`) so this test and that one cannot drift
onto different oceans.
"""
import datetime as dt
import json
import math
import os
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
from test_rollout_spatial import build_fixture                # noqa: E402

SCOPES = ("gate", "corridor", "window")
EPOCH = dt.date(1982, 1, 1)


def walk_non_finite(obj, path="$"):
    """Every non-finite float in a parsed JSON payload, with its path.

    json.load turns a literal `NaN` / `Infinity` token into a float, so a
    string search of the file and a walk of the object are two different
    checks and both are run: the string search catches what Python would
    happily re-emit, the walk catches what a reader would consume.
    """
    bad = []
    if isinstance(obj, dict):
        for k, v in obj.items():
            bad += walk_non_finite(v, f"{path}.{k}")
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            bad += walk_non_finite(v, f"{path}[{i}]")
    elif isinstance(obj, float) and not math.isfinite(obj):
        bad.append((path, obj))
    return bad


def roll(tmp, holdout_lon):
    """Run the REAL rollout_spatial.py on the toy ocean. -> (payload, log)."""
    f = build_fixture(tmp, holdout_lon=holdout_lon)
    out = os.path.join(tmp, "roll.json")
    cmd = [sys.executable, "-u", os.path.join(ML, "rollout_spatial.py"),
           "--x", f["x"], "--npz-small", f["npz"], "--z", f["z"],
           "--ckpt", f["ckpt"], "--out", out, "--horizon", "3",
           "--long-start", "1991-12", "--long-months", "16",
           "--future-months", "5", "--cache-dir", tmp,
           "--no-gate", "--heads", *f["heads"]]
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)
    if r.returncode != 0:
        print(r.stdout[-4000:])
        print(r.stderr[-4000:])
        raise SystemExit(f"rollout_spatial.py failed under "
                         f"holdout_lon={holdout_lon!r} — the zero-pixel scope "
                         f"is FATAL, not merely empty")
    return json.load(open(out)), open(out).read(), r.stdout, f


def main():
    tmp0 = tempfile.mkdtemp()
    tmp1 = tempfile.mkdtemp()
    try:
        # ---- 1. the roll survives "0,0", and writes no NaN ---------------
        res, raw, log, f = roll(tmp0, "0,0")
        for tok in ("NaN", "Infinity", "-Infinity"):
            assert tok not in raw, \
                (f"roll json contains the literal {tok!r} — ml/CLAUDE.md "
                 f"§5.22 forbids writing it into a results file")
        nf = walk_non_finite(res)
        assert not nf, f"non-finite values in the roll payload: {nf[:5]}"
        hl = res["holdout_lon"]
        assert hl["arg"] == "0,0" and (hl["lo"], hl["hi"]) == (0.0, 0.0), hl
        assert hl["n_cols"] == 0, hl
        assert hl["any"] is False, hl
        assert hl["excluded_from"] == [], hl["excluded_from"]
        assert "NO longitude is held out" in hl["note"], hl["note"]
        for k, v in hl["px"].items():
            assert v["in_block"] == 0 and v["frac"] == 0.0, (k, v)
        assert "NO lon holdout" in log, \
            "the log claims a held-out block over an empty set"
        print("1. rolled the toy under holdout_lon '0,0': rc 0, no NaN and no "
              "Infinity anywhere in %d bytes of roll json; holdout_lon.any "
              "False, 0/%d cols, excluded_from []"
              % (len(raw), hl["of_cols"]))

        # ---- 2. the zero-pixel scope reports n_px 0, never an aggregate --
        for label, e in res["heads"].items():
            for sc in SCOPES:
                par, tr, ho = e[sc], e[sc + "_trainlon"], e[sc + "_holdlon"]
                assert ho["n_px"] == 0, (label, sc, ho)
                assert ho["chan_skill"] == [], (label, sc, ho)
                for k in ("horizon_auc", "auc_damped"):
                    assert k not in ho, \
                        (f"{label}/{sc}_holdlon has {k} over zero pixels — "
                         f"that number can only be NaN or a lie")
                assert "empty" in ho and "0 pixels" in ho["empty"], ho
                # the split is still a partition: with an empty child the
                # parent must equal its _trainlon EXACTLY, key for key.
                assert par["n_px"] == tr["n_px"] > 0, (label, sc, par, tr)
                assert par["chan_skill"] == tr["chan_skill"], \
                    (f"{label}/{sc}: parent and _trainlon differ although the "
                     f"_holdlon child is empty — the partition is broken")
                assert par["horizon_auc"] == tr["horizon_auc"], (label, sc)
        print("2. every <scope>_holdlon: n_px 0, chan_skill [], no "
              "horizon_auc, an `empty` note saying why; every parent equals "
              "its _trainlon key for key (%d head x scope pairs)"
              % (len(res["heads"]) * len(SCOPES)))

        # ---- 3. and the OTHER branch still behaves (a branch, not a hole) -
        res2, raw2, log2, _ = roll(tmp1, "-45,-44")
        assert not walk_non_finite(res2)
        hl2 = res2["holdout_lon"]
        assert hl2["any"] is True and hl2["n_cols"] == 1, hl2
        assert hl2["excluded_from"], hl2
        assert "held out of TRAINING in both" in hl2["note"], hl2["note"]
        assert "held-out lon block" in log2, log2[-2000:]
        for label, e in res2["heads"].items():
            for sc in SCOPES:
                ho = e[sc + "_holdlon"]
                assert ho["n_px"] > 0 and ho["chan_skill"], (label, sc, ho)
                assert "empty" not in ho, ho
                assert (e[sc]["n_px"]
                        == e[sc + "_trainlon"]["n_px"] + ho["n_px"]), (label, sc)
        print("3. the same toy under '-45,-44' still reports a real block "
              "(any True, 1 col, holdlon scored) and n_px partitions the "
              "parent exactly — the new reporting is a BRANCH, not a hole")

        # ---- 4. the other x_hold consumers, on an empty selection --------
        from recon_eval import score, score_pooled              # noqa: E402
        from rollout_spatial import new_sums, skill_block       # noqa: E402
        from train import lon_holdout_mask                      # noqa: E402
        rng = np.random.default_rng(0)
        T_, P_, C_ = 40, 12, 4
        truth = rng.standard_normal((T_, P_, C_)).astype(np.float32)
        pred = truth + 0.1 * rng.standard_normal((T_, P_, C_)).astype(np.float32)
        obs = np.ones((T_, P_, C_), bool)
        empty = np.array([], dtype=int)
        s_hold = score(truth, pred, obs, np.arange(T_), empty, "hold-x")
        assert s_hold == {}, \
            (f"recon_eval.score over zero pixels returned {s_hold} — it must "
             f"omit, never emit a NaN r/rmse")
        assert not walk_non_finite(s_hold)
        s_ok = score(truth, pred, obs, np.arange(T_), np.arange(P_), "train")
        assert s_ok and not walk_non_finite(s_ok), s_ok
        sp = score_pooled(truth, pred, obs, np.arange(T_), empty)
        assert sp == {} and not walk_non_finite(sp), sp
        # skill_block with nothing accumulated, both ways round
        H_ = 3
        b0 = skill_block(new_sums(H_), H_, n_px=0)
        assert b0["chan_skill"] == [] and b0["n_px"] == 0, b0
        assert "0 pixels" in b0["empty"] and "horizon_auc" not in b0, b0
        b1 = skill_block(new_sums(H_), H_, n_px=7)
        assert "investigate" in b1["empty"], \
            ("a scope with 7 pixels that scored nothing must NOT be reported "
             "as 'nothing to score' — that is the case worth a look")
        bN = skill_block(new_sums(H_), H_)
        assert "n_px" not in bN and "empty" not in bN, \
            "a caller that passes no n_px must get the OLD payload shape"
        # and the mask itself, through the one parser
        prod = np.arange(-100, 20, 0.25, dtype=np.float32)
        m00 = lon_holdout_mask("0,0", prod)
        assert not m00.any() and m00.dtype == np.bool_, m00.sum()
        assert np.array_equal(m00, lon_holdout_mask("none", prod))
        print("4. recon_eval.score / score_pooled over an EMPTY pixel "
              "selection return {} (omitted, not NaN); skill_block says "
              "'0 pixels' at n_px 0 and 'investigate' at n_px 7; omitting "
              "n_px reproduces the pre-change payload; '0,0' == 'none' as a "
              "mask over the production grid")

        # ---- 5. TRIPWIRE: rollout_spatial is MONTHLY-KEYED ---------------
        # family 4 emits one `YYYY-MM` LABEL per 5-day bin (build_family4.py
        # ~line 897: "`bin_index` remains the authoritative axis; `months` is
        # a label"), which is correct for the two things train.py asks of it
        # — `m[:4]` for the year holdout and `int(m[5:7])-1` for the season
        # token — and WRONG for the two dictionaries rollout_spatial builds
        # out of the same array, both of which need it to be a unique key.
        # This assertion documents a defect on purpose. When somebody makes
        # the roll cadence-aware, this check fails and they delete it; until
        # then it is the reason `sroll:` must not be dispatched at pentad.
        bins = np.arange(0, 3142)
        months_p = np.array(
            ["%04d-%02d" % ((EPOCH + dt.timedelta(days=int(5 * b))).year,
                            (EPOCH + dt.timedelta(days=int(5 * b))).month)
             for b in bins])
        month_index = {m: i for i, m in enumerate(months_p)}
        assert len(month_index) == 516 < len(months_p) == 3142, \
            (len(month_index), len(months_p))
        first_jan09 = int(np.where(months_p == "2009-01")[0][0])
        assert month_index["2009-01"] != first_jan09, \
            "the collision has gone away — re-read this check"
        assert month_index["2009-01"] == first_jan09 + 5, month_index["2009-01"]
        # the staggered-start protocol asks for 12 starts per holdout year …
        starts = {("%d-12" % 2008) if o == 0 else "2009-%02d" % o
                  for o in range(12)}
        n_pentads_2009 = int(np.char.startswith(months_p, "2009").sum())
        assert len(starts) == 12 and n_pentads_2009 == 73, n_pentads_2009
        # … and the RAPID truth attach keeps one pentad per calendar month
        ym_to_r = {int(months_p[m][:4]) * 100 + int(months_p[m][5:7]): i
                   for i, m in enumerate(range(len(months_p)))}
        assert len(ym_to_r) == 516, len(ym_to_r)
        kept = len(ym_to_r) / len(months_p)
        assert kept < 0.17, kept
        print("5. TRIPWIRE (expected to FAIL once the roll is cadence-aware): "
              "on a family-4 months array rollout_spatial's month_index "
              "collapses %d bins to %d keys (%.2f:1, last-wins), the "
              "staggered protocol offers 12 of %d starts per holdout year, "
              "and ym_to_r discards %.1f%% of the pentad RAPID series. "
              "sroll: at pentad is BLOCKED on this."
              % (len(months_p), len(month_index),
                 len(months_p) / len(month_index), n_pentads_2009,
                 100 * (1 - kept)))

        print("\nzero-longitude-holdout eval path: all 5 checks hold ✓")
    finally:
        shutil.rmtree(tmp0, ignore_errors=True)
        shutil.rmtree(tmp1, ignore_errors=True)


if __name__ == "__main__":
    main()

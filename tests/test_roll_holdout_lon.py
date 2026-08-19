#!/usr/bin/env python3
"""The held-out LONGITUDE block must be visible in what the evaluator writes.

`ml/train.py --holdout-lon` defaults to `-45,-25`, and those columns are
excluded from the TRAINING pool of BOTH stages (train.py's stage-1 pool is
`obs_any & ~t_hold & ~x_hold`; temporal.py's stage-2 pool is
`ok_p = ~x_hold[xs]`). `ml/rollout_spatial.py` recomputes that same `x_hold`
for its z-score statistics — and used to throw it away, scoring gate,
corridor and window over every pixel while recording nowhere that a quarter
of them had never been trained on. On the production window that is 25.0% of
window pixels, 23.9% of corridor pixels and 30.2% of the RAPID section, with
in-block h=6 skill ~0.23 against ~0.86 outside: a blend nobody reading
`roll_*.json` could unmix, which is what made it cost a session to diagnose.

Nothing there was computed wrongly and no published number was inflated (the
blend is deflationary). It is a REPORTING defect, so this is a REPORTING
test. It asserts, on a synthetic ocean rolled by the real script:

  * `roll_*.json` carries `holdout_lon` at the top level, agreeing with the
    checkpoint's own argument and with an independently recomputed `x_hold`;
  * every scope aggregate is accompanied by its `_trainlon` / `_holdlon`
    split, and the two children exactly PARTITION the parent's sample count;
  * the exported globe mask carries the block too — as an additive field,
    with the class codes still [1,2,3] (see export_mask's docstring for why);
  * old roll JSONs, which have none of this, are still readable by the
    readers in the tree. The TEST may demand the keys; the CODE must not.

No GPU and no 4 GB tensor: it reuses test_rollout_spatial.py's toy fixture,
so the two tests cannot drift onto different oceans.

    python3 tests/test_roll_holdout_lon.py
"""
import copy
import json
import os
import shutil
import subprocess
import sys
import tempfile

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ML = os.path.join(HERE, "..", "ml")
sys.path.insert(0, HERE)
sys.path.insert(0, ML)
from test_rollout_spatial import build_fixture                # noqa: E402

SCOPES = ("gate", "corridor", "window")
SPLITS = ("_trainlon", "_holdlon")
# what a scope block looks like once it has scored anything
SCOPE_KEYS = ("chan_skill", "horizon_auc", "auc_damped")
HOLD_KEYS = ("arg", "lo", "hi", "rule", "n_cols", "of_cols", "px", "note")


def check_roll_json(res, where="<roll json>"):
    """The guard itself: raise unless `res` records its longitude holdout.

    Kept as a function so the negative cases below can drive it with
    deliberately damaged payloads — a guard that has never been shown to
    refuse is an assumption (ml/CLAUDE.md 0.2).
    """
    bad = []
    hl = res.get("holdout_lon")
    if hl is None:
        bad.append("no top-level `holdout_lon`: the artefact does not say "
                   "which longitudes were never trained on")
    else:
        for k in HOLD_KEYS:
            if k not in hl:
                bad.append(f"holdout_lon missing `{k}`")
        for scope, px in (hl.get("px") or {}).items():
            if not {"in_block", "of"} <= set(px):
                bad.append(f"holdout_lon.px[{scope}] has no in_block/of count")
            elif px["in_block"] > px["of"]:
                bad.append(f"holdout_lon.px[{scope}]: in_block "
                           f"{px['in_block']} > of {px['of']}")
        if not hl.get("px"):
            bad.append("holdout_lon.px is empty: no per-scope in-block counts")

    for label, entry in (res.get("heads") or {}).items():
        for sc in SCOPES:
            if sc not in entry:
                continue            # a scope never scored is fine
            for suf in SPLITS:
                child = entry.get(sc + suf)
                if child is None:
                    bad.append(f"{label}: `{sc}` aggregate present without "
                               f"`{sc}{suf}` — the number is a blend of "
                               f"trained and never-trained pixels with no "
                               f"way to unmix it")
                    continue
                if "chan_skill" not in child:
                    bad.append(f"{label}/{sc}{suf}: no chan_skill")
                elif child["chan_skill"]:
                    for k in SCOPE_KEYS:
                        if k not in child:
                            bad.append(f"{label}/{sc}{suf}: scored rows but "
                                       f"no `{k}` (parent has it)")
    if bad:
        raise AssertionError(f"{where}:\n  - " + "\n  - ".join(bad))


def expect_refusal(res, where, must_mention):
    """check_roll_json MUST reject `res`, naming `must_mention`."""
    try:
        check_roll_json(res, where)
    except AssertionError as e:
        assert must_mention in str(e), \
            f"refused, but not about {must_mention!r}: {e}"
        return str(e).splitlines()[1].strip()
    raise SystemExit(f"{where}: check_roll_json ACCEPTED a payload it must "
                     f"refuse ({must_mention}) — the guard does not bite")


def bake_mask(path):
    """Drive export_mask on the real window geometry (0-70 N, -100..20 E at
    0.25 deg), so its own read-back probe can run. Synthetic ocean/corridor/
    section, no tensor."""
    from rollout_spatial import export_mask                    # noqa: E402
    lats = (0.0 + 0.25 * np.arange(281)).astype(np.float32)
    lons = (-100.0 + 0.25 * np.arange(481)).astype(np.float32)
    ocean = np.ones((len(lats), len(lons)), bool)
    ocean[:, :4] = False                  # a sliver of land, so "." exists
    ys, xs = np.where(ocean)
    P = len(ys)
    sec_y = int(np.argmin(np.abs(lats - 26.5)))
    sec_sel = np.where((ys == sec_y) & (lons[xs] >= -77.0)
                       & (lons[xs] <= -13.0))[0]
    corridor = (lats[ys] > 20.0) & (lats[ys] < 45.0)
    corridor[sec_sel] = True
    gate_mask = np.zeros(P, bool)
    gate_mask[::37] = True
    x_hold = (lons >= -45.0) & (lons < -25.0)   # train.py's own default
    px_hold = x_hold[xs]
    sec_mask = np.zeros(P, bool)
    sec_mask[sec_sel] = True
    holdout_lon = {
        "arg": "-45,-25", "lo": -45.0, "hi": -25.0,
        "rule": "(lons >= lo) & (lons < hi), train.py's own expression",
        "n_cols": int(x_hold.sum()), "of_cols": len(lons),
        "excluded_from": ["stage-1", "stage-2"],
        "px": {n: {"in_block": int((m_ & px_hold).sum()), "of": int(m_.sum()),
                   "frac": round(float((m_ & px_hold).sum() / m_.sum()), 4)}
               for n, m_ in (("gate", gate_mask), ("corridor", corridor),
                             ("window", np.ones(P, bool)),
                             ("section", sec_mask))},
        "note": "never trained",
    }
    cdef = {"pctl": 75, "threshold": 0.3, "dilate_cells": 2,
            "structuring": "3x3 square", "n_px": int(corridor.sum()),
            "of": P, "union_section": True}
    export_mask(path, lats, lons, ocean, ys, xs, corridor, gate_mask,
                sec_sel, sec_y, cdef, ["1982-01", "2024-12"], "toy_X.npy",
                holdout_lon)
    return path


def main():
    tmp = tempfile.mkdtemp()
    try:
        f = build_fixture(tmp)
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
            raise SystemExit("rollout_spatial.py failed on the toy")
        res = json.load(open(out))
        print("1. rolled the toy ocean end to end: %s heads, %s"
              % (len(res["heads"]), ", ".join(res["heads"])))

        # -- 2. the block is in the artefact, and it is the RIGHT block ----
        check_roll_json(res, out)
        hl = res["holdout_lon"]
        lons, x_hold = f["lons"], f["x_hold"]
        assert hl["arg"] == "-45,-44", hl["arg"]
        assert (hl["lo"], hl["hi"]) == (-45.0, -44.0), hl
        assert hl["n_cols"] == int(x_hold.sum()) == 1, hl
        assert hl["of_cols"] == len(lons), hl
        # recompute the pixel count from the fixture, independently
        ys, xs = np.where(f["ocean"])
        px_hold = x_hold[xs]
        assert hl["px"]["window"]["of"] == len(ys) == f["P"], hl["px"]
        assert hl["px"]["window"]["in_block"] == int(px_hold.sum()) > 0, \
            (hl["px"], int(px_hold.sum()))
        print("2. holdout_lon recorded: [%g, %g), %d/%d cols · %s"
              % (hl["lo"], hl["hi"], hl["n_cols"], hl["of_cols"],
                 " ".join("%s %d/%d" % (k, v["in_block"], v["of"])
                          for k, v in hl["px"].items())))

        # -- 3. every scope split, and the split is an exact partition -----
        for label, e in res["heads"].items():
            for sc in SCOPES:
                for suf in SPLITS:
                    assert sc + suf in e, f"{label}: no {sc}{suf}"
                par = {r_["h"]: r_["n"] for r_ in e[sc]["chan_skill"]}
                tr = {r_["h"]: r_["n"]
                      for r_ in e[sc + "_trainlon"]["chan_skill"]}
                ho = {r_["h"]: r_["n"]
                      for r_ in e[sc + "_holdlon"]["chan_skill"]}
                assert par, f"{label}/{sc}: parent scored nothing"
                for h, n in par.items():
                    got = tr.get(h, 0) + ho.get(h, 0)
                    assert got == n, (
                        f"{label}/{sc} h={h}: trainlon {tr.get(h, 0)} + "
                        f"holdlon {ho.get(h, 0)} = {got} != parent {n} — the "
                        f"split is not a partition of the scope")
                assert ho, (f"{label}/{sc}_holdlon scored nothing, but the "
                            f"fixture puts pixels in the block")
                for k in SCOPE_KEYS:
                    assert k in e[sc + "_trainlon"] and k in e[sc + "_holdlon"]
        print("3. all %d scopes present and partitioned exactly (%s)"
              % (3 * len(SCOPES),
                 ", ".join(s + t for s in SCOPES for t in SPLITS)))

        # -- 4. the split is not a relabelled copy of the parent -----------
        for label, e in res["heads"].items():
            for sc in SCOPES:
                a_ = json.dumps(e[sc + "_trainlon"]["chan_skill"])
                b_ = json.dumps(e[sc + "_holdlon"]["chan_skill"])
                assert a_ != b_, (f"{label}/{sc}: trainlon and holdlon are "
                                  f"byte-identical — one mask is being scored "
                                  f"twice")
        print("4. trainlon != holdlon for every head/scope · window AUC "
              + " ".join("%s %+.3f/%+.3f/%+.3f"
                         % (lb, e["window"]["horizon_auc"],
                            e["window_trainlon"]["horizon_auc"],
                            e["window_holdlon"]["horizon_auc"])
                         for lb, e in res["heads"].items())
              + "  (all/train/hold)")

        # -- 5. the mask: additive field, class codes UNTOUCHED ------------
        # export_mask is called directly, on a PRODUCTION-geometry grid: its
        # own read-back probe demands the RAPID section at -70 and -40 E, so
        # the 8x10-cell toy above cannot satisfy it. Everything the mask
        # writer needs is index arithmetic, so this costs milliseconds.
        m = json.load(open(bake_mask(os.path.join(tmp, "mask.json"))))
        assert "holdout_lon" in m, "export_mask payload has no holdout_lon"
        assert m["holdout_lon"]["lo"] == -45.0, m["holdout_lon"]
        # the whole point of choosing a field over a 4th code: the palette and
        # the packed alphabet are exactly what they were, so the frontend's
        # class legend and ml/paper/make_figs.py's `code >= 1` / `code >= 2`
        # reconstruction keep meaning what they meant.
        assert [c["code"] for c in m["classes"]] == [1, 2, 3], m["classes"]
        assert set(m["packed"]) <= {".", "1", "2", "3"}, \
            "a 4th class code leaked into the packed grid"
        n1 = sum(m["packed"].count(c) for c in "123")
        assert n1 == m["counts"]["rolled"], (n1, m["counts"])
        print("5. mask payload: holdout_lon field added, classes still %s, "
              "packed alphabet still %s"
              % ([c["code"] for c in m["classes"]],
                 sorted(set(m["packed"]))))

        # -- 6/7. the guard must REFUSE the pre-change shapes --------------
        stripped = copy.deepcopy(res)
        stripped.pop("holdout_lon")
        msg = expect_refusal(stripped, "roll json with no holdout_lon",
                             "no top-level `holdout_lon`")
        print("6. refuses a roll json without holdout_lon: %s" % msg)

        unsplit = copy.deepcopy(res)
        for e in unsplit["heads"].values():
            for sc in SCOPES:
                for suf in SPLITS:
                    e.pop(sc + suf, None)
        msg = expect_refusal(unsplit, "roll json with unsplit scopes",
                             "aggregate present without")
        print("7. refuses a scope aggregate with no split: %s" % msg)

        # -- 8. old artefacts stay READABLE (the code must not demand it) --
        legacy = [os.path.join(ML, "paper", n)
                  for n in ("roll_355.json", "roll_356.json")]
        legacy = [p for p in legacy if os.path.exists(p)]
        assert legacy, "no archived roll_*.json found to check for regressions"
        from plot_amoc_roll import load_heads                  # noqa: E402
        hs = load_heads(legacy)
        assert hs, "load_heads returned nothing for the archived rolls"
        for p in legacy:
            d = json.load(open(p))
            assert "holdout_lon" not in d, \
                f"{p} is an archived PRE-change output; it should not have " \
                f"the new key (did someone rewrite it?)"
            for e in d["heads"].values():
                assert "corridor" in e and "corridor_trainlon" not in e
        print("8. archived %s read fine without the new keys (%d heads via "
              "plot_amoc_roll.load_heads)"
              % (", ".join(os.path.basename(p) for p in legacy), len(hs)))

        print("holdout-lon reporting: all checks passed ✓")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    main()

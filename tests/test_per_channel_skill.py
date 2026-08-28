#!/usr/bin/env python3
"""E-058: PER-CHANNEL rolled forecast skill, and proof the pooled numbers
did not move.

    python3 -m pytest tests/test_per_channel_skill.py -x -q
    python3 tests/test_per_channel_skill.py            # same checks, verbose

WHY THIS EXISTS. Chris, 2026-08-28: *"ocean surface temperature as a secondary
downstream target (next to AMOC) ... to ensure the embedding representation is
comprehensive and not just AMOC tailored"*. Until E-058 every row
`ml/rollout_spatial.py:skill_block` emitted was pooled over all pixels AND all
40 channels — the key is called `chan_skill`, but that name is legacy from
`rollout.py` where a "chan" row was a HORIZON, and `sst` (the last channel,
appended by E-042) was invisible inside the pool. `skill_block` now also
emits `per_channel`: channel NAME -> that channel's own rows.

The four checks, in the order the deliverable states them:

  1. **POOLED PURITY — the acceptance bar.** The whole evaluator, run on the
     monthly toy, produces a results tree in which every PRE-EXISTING key and
     value (`chan_skill` rows, `horizon_auc`, `horizon_auc_daymatched`,
     `auc_damped`, `acc`, `amp_ratio`, `n`, `n_px`, the audit block, the amoc
     bands) is bit-identical to what the PRISTINE `ml/rollout_spatial.py`
     produces from the same inputs. The pristine file is recovered from git
     history as the newest blob of that path with no `E-058` token in it —
     searching by CONTENT, not by commit offset, so the check survives the
     session committing this diff (the pattern is `tests/test_fgn_roll.py`'s
     `pristine_evaluator`, reused here rather than reinvented). "Bit-identical"
     is enforced by re-serialising both parsed payloads with the writer's own
     `json.dumps(..., indent=1)` and comparing the STRINGS, so a reordered
     key, a changed indent or `0.643` becoming `0.6430000000000001` all fail —
     the standard `tests/test_roll_monthly_identity.py` sets.
  2. **CONSISTENCY.** The pooled sums and the per-channel sums describe the
     SAME roll: recomposing the pooled `msss_clim` from the per-channel sums
     reproduces it to floating point. MEASURED DEVIATION on this toy (H=3,
     C=4, 37 pixels, 3 starts, ~20% of cells unobserved): **1.11e-16
     absolute**, the max over all horizons, for the sum recomposition
     (`1 - sum_c mse_m_c / sum_c mse_c_c`), and **0.0 — exact** for the
     climatology-mass-weighted form (`sum_c w_c msss_c / sum_c w_c`,
     `w_c = mse_c_c`). One ulp and zero: what "the same additions in a
     different order" costs, and ~1e13 tighter than the 2e-3 tolerance the
     E-026b audit block already applies to this same identity. The test
     asserts < 1e-12, which is loose enough not to be a float-noise tripwire
     and tight enough that a genuine mis-accumulation cannot pass.
  3. **A KNOWN ANSWER.** One channel predicted PERFECTLY and one channel pure
     noise, in the same array, and the per-channel rows separate them
     (msss_clim 1.0 vs <= 0, acc 1.0 vs ~0) while the pooled row — the only
     thing the artefact carried before E-058 — averages them into a single
     middling number that says nothing about either. This is the check that
     proves the feature answers the question Chris asked.
  4. **THE NAMES ARE THE TENSOR'S.** `per_channel`'s keys are exactly the
     codec checkpoint's `ck["chan"]`, in order, and a name list whose length
     disagrees with the sums' channel axis is REFUSED rather than zipped
     short (a short zip would silently drop `sst`, which is last).

Plus a structural check that the new key reaches every write point: `entry`
is built once and mutated in place, and the `skill_block` assignment happens
before the first `write_results`, so the partial writes at the scored / long /
future / head_done stages and the final unmarked write all carry it.

No GPU: the 79-pixel monthly synthetic ocean of
`tests/test_rollout_spatial.py`, whose `build_fixture` is imported rather
than re-implemented.
"""
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
ML = os.path.join(ROOT, "ml")
sys.path.insert(0, HERE)
sys.path.insert(0, ML)
from test_rollout_spatial import build_fixture                   # noqa: E402
from test_fgn_roll import run, strip_clocks, first_diff          # noqa: E402
import rollout_spatial as RS                                     # noqa: E402

TOKEN = b"E-058"          # what makes a blob "already patched"


# ---------------------------------------------------------------------------
# the pristine evaluator — tests/test_fgn_roll.py's rule, one token changed
# ---------------------------------------------------------------------------
def pristine_evaluator(tmp, token=TOKEN):
    """The newest `ml/rollout_spatial.py` in git history that does not know
    about `token`, written beside the toy so it can be run.

    Deliberately not `HEAD~1`: the main session commits this diff, and a
    reference pinned to a commit OFFSET would then compare the patched file
    against itself and pass for the worst possible reason. Searching by
    CONTENT keeps the comparison honest after the commit lands. Returns None
    where git cannot answer, in which case check 1 says so LOUDLY and fails
    rather than passing vacuously — this is the acceptance bar, and an
    acceptance bar that can quietly skip itself is not one.
    """
    try:
        shas = subprocess.check_output(
            ["git", "log", "--format=%H", "--", "ml/rollout_spatial.py"],
            cwd=ROOT, text=True, stderr=subprocess.DEVNULL).split()
    except Exception:                                          # noqa: BLE001
        return None
    for sha in shas:
        try:
            blob = subprocess.check_output(
                ["git", "show", f"{sha}:ml/rollout_spatial.py"],
                cwd=ROOT, stderr=subprocess.DEVNULL)
        except Exception:                                      # noqa: BLE001
            continue
        if token in blob:
            continue
        path = os.path.join(tmp, "pristine_rollout_spatial.py")
        with open(path, "wb") as fh:
            fh.write(blob)
        return path, sha
    return None


NEW_KEY = "per_channel"


def strip_new_key(o, found=None):
    """Remove every `per_channel` block, returning (tree, list of blocks).

    Recursive because the key lives one level down (`heads.<label>.<scope>`),
    and because a strip that knew the depth would stop policing the day the
    depth changed."""
    found = [] if found is None else found
    if isinstance(o, dict):
        out = {}
        for k, v in o.items():
            if k == NEW_KEY:
                found.append(v)
                continue
            out[k] = strip_new_key(v, found)[0]
        return out, found
    if isinstance(o, list):
        return [strip_new_key(v, found)[0] for v in o], found
    return o, found


def as_bytes(obj):
    """The writer's own serialisation, so a comparison of these strings is a
    comparison of the files `write_results` would have written: same indent,
    same key order, same float repr (`json.dump(payload, f, indent=1)`)."""
    return json.dumps(obj, indent=1)


# ---------------------------------------------------------------------------
# 1. POOLED PURITY
# ---------------------------------------------------------------------------
def _roll_pair(tmp):
    """Run the patched and the pristine evaluator on one monthly toy."""
    # the toy heads are randomly initialised; seed so a failure is
    # reproducible from the printed numbers (tests/test_fgn_roll.py does the
    # same for the same reason). Both evaluators see the SAME heads either
    # way — this is about re-running, not about the comparison.
    torch.manual_seed(0)
    f = build_fixture(tmp)
    f["cache"] = os.path.join(tmp, "cache")
    os.makedirs(f["cache"], exist_ok=True)
    pris = pristine_evaluator(tmp)
    assert pris is not None, (
        "no pristine ml/rollout_spatial.py in git history (no blob without "
        f"{TOKEN!r}) — check 1 is the acceptance bar for this change and "
        "cannot be skipped")
    new, log_new = run(os.path.join(ML, "rollout_spatial.py"), f,
                       os.path.join(tmp, "new.json"), f["heads"],
                       extra=("--no-unpooled-readout",))
    old, _ = run(pris[0], f, os.path.join(tmp, "old.json"), f["heads"],
                 extra=("--no-unpooled-readout",))
    return f, new, old, pris[1], log_new


def check_pooled_purity(f, new, old, sha):
    n_scored = sum(
        1 for e in new["heads"].values()
        for b in e.values() if isinstance(b, dict) and b.get("chan_skill"))
    stripped, blocks = strip_new_key(strip_clocks(new))
    base = strip_clocks(old)

    assert not strip_new_key(base)[1], (
        f"the pristine evaluator at {sha[:8]} already writes `{NEW_KEY}` — "
        "the strip would then be hiding a DIFFERENCE, not a new key")
    assert len(blocks) == n_scored > 0, (
        f"{len(blocks)} `{NEW_KEY}` blocks for {n_scored} scored scopes — the "
        "key must be on every scope that has rows, so a harvest can never "
        "find a scope where only the pooled number exists")

    d = first_diff(base, stripped)
    assert d is None, f"patched vs pristine ({sha[:8]}) differ at {d}"
    a, b = as_bytes(base), as_bytes(stripped)
    assert a == b, "the payloads parse equal but do not SERIALISE equal"
    return n_scored, len(a), blocks


# ---------------------------------------------------------------------------
# 2. CONSISTENCY — the two sum sets describe the same roll
# ---------------------------------------------------------------------------
def _toy_sums(H=3, C=4, n_px=37, seed=7, holes=True):
    """Feed `accumulate` the shapes the real call sites feed it: everything
    [n_pixels, C], `op` a boolean [n_pixels, C] observation mask."""
    rng = np.random.default_rng(seed)
    su = RS.new_sums(H, C)
    for _ in range(3):                       # three "starts"
        for h in range(1, H + 1):
            v_true = rng.standard_normal((n_px, C))
            xhat = 0.6 * v_true + 0.5 * rng.standard_normal((n_px, C))
            v_pers = rng.standard_normal((n_px, C))
            v_damp = 0.8 * v_pers
            op = (rng.random((n_px, C)) > (0.2 if holes else 0.0))
            RS.accumulate(su, h, xhat, v_true, v_pers, v_damp, op)
    return su


def check_consistency():
    """Pooled msss_clim == the per-channel sums recomposed, to fp tolerance."""
    H, C = 3, 4
    su = _toy_sums(H=H, C=C)
    pc = su["per_chan"]
    dev_sum, dev_w = 0.0, 0.0
    for h in range(1, H + 1):
        assert su["n"][h] > 0
        # every sum is the channel sums added up
        for k in RS.SUM_KEYS:
            assert abs(su[k][h] - pc[k][h].sum()) <= 1e-9 * max(
                1.0, abs(su[k][h])), (k, h)
        pooled = 1 - su["mse_m"][h] / su["mse_c"][h]
        recomp = 1 - pc["mse_m"][h].sum() / pc["mse_c"][h].sum()
        dev_sum = max(dev_sum, abs(pooled - recomp))
        # the same statement written as a WEIGHTED MEAN of the per-channel
        # skill scores, weight = that channel's climatological error mass,
        # which is the form a reader recombining the published rows would use
        msss_c = 1 - pc["mse_m"][h] / pc["mse_c"][h]
        w = pc["mse_c"][h]
        dev_w = max(dev_w, abs(pooled - (w * msss_c).sum() / w.sum()))
    assert dev_sum < 1e-12 and dev_w < 1e-12, (dev_sum, dev_w)
    return dev_sum, dev_w


# ---------------------------------------------------------------------------
# 3. A KNOWN ANSWER — perfect channel vs noise channel
# ---------------------------------------------------------------------------
def check_known_answer():
    H, C, n_px = 1, 3, 400
    rng = np.random.default_rng(11)
    v_true = rng.standard_normal((n_px, C))
    xhat = np.empty_like(v_true)
    xhat[:, 0] = v_true[:, 0]                          # PERFECT
    xhat[:, 1] = rng.standard_normal(n_px)             # PURE NOISE
    xhat[:, 2] = 0.7 * v_true[:, 2] + 0.3 * rng.standard_normal(n_px)
    v_pers = rng.standard_normal((n_px, C))
    su = RS.new_sums(H, C)
    RS.accumulate(su, 1, xhat, v_true, v_pers, 0.8 * v_pers,
                  np.ones((n_px, C), bool))
    names = ["perfect", "noise", "middling"]
    blk = RS.skill_block(su, H, n_px=n_px, chan_names=names)

    per = blk["per_channel"]
    assert set(per) == set(names), per.keys()
    good = per["perfect"][0]
    bad = per["noise"][0]
    mid = per["middling"][0]
    assert good["msss_clim"] == 1.0, good
    assert good["acc"] == 1.0 and good["amp_ratio"] == 1.0, good
    assert bad["msss_clim"] <= 0.0, bad
    assert abs(bad["acc"]) < 0.2, bad
    assert 0.0 < mid["msss_clim"] < 1.0, mid
    assert good["msss_clim"] - bad["msss_clim"] > 1.0, (good, bad)

    # AND THE POINT OF THE CHANGE: the pooled row cannot tell them apart.
    pooled = blk["chan_skill"][0]
    assert bad["msss_clim"] < pooled["msss_clim"] < good["msss_clim"], pooled
    assert pooled["n"] == n_px * C and good["n"] == n_px
    return good, bad, mid, pooled


# ---------------------------------------------------------------------------
# 4. THE NAMES ARE THE TENSOR'S
# ---------------------------------------------------------------------------
def check_names_refusal():
    H, C = 2, 4
    su = _toy_sums(H=H, C=C)
    ok = [f"c{i}" for i in range(C)]
    assert set(RS.skill_block(su, H, chan_names=ok)["per_channel"]) == set(ok)
    n_bad = 0
    for bad in (ok[:-1], ok[:1], ok + ["extra"], []):
        try:
            RS.skill_block(su, H, chan_names=bad)
        except ValueError as ex:
            n_bad += 1
            assert "channels" in str(ex), ex
        else:
            raise AssertionError(
                f"a {len(bad)}-name list was accepted for {C} channels — a "
                "short zip silently mislabels every channel after the "
                "mismatch, and `sst` is the LAST one")
    # no names -> no new key, so a caller that does not ask is unaffected
    assert NEW_KEY not in RS.skill_block(su, H)
    # ...and sums built without a channel axis have nothing to emit
    assert NEW_KEY not in RS.skill_block(RS.new_sums(H), H, chan_names=ok)
    # §5.22, the `n_px`/`empty` precedent: a scope that scored NOTHING (every
    # `_holdlon` child under a no-longitude-holdout codec, E-043) omits the
    # aggregate rather than writing a NaN one. `per_channel` follows suit —
    # no channel scored, so there is no key, and `empty` still says why.
    zero = RS.skill_block(RS.new_sums(H, C), H, n_px=0, chan_names=ok)
    assert zero["chan_skill"] == [] and zero["n_px"] == 0
    assert "0 pixels" in zero["empty"] and NEW_KEY not in zero
    return n_bad


def check_names_are_the_tensors(f, new):
    """The evaluator's `per_channel` keys are the checkpoint's `ck["chan"]`."""
    ck = torch.load(f["ckpt"], map_location="cpu", weights_only=False)
    want = [str(c) for c in ck["chan"]]
    seen = 0
    for e in new["heads"].values():
        for name, blk in e.items():
            if not (isinstance(blk, dict) and blk.get("chan_skill")):
                continue
            assert list(blk[NEW_KEY]) == want, (name, list(blk[NEW_KEY]))
            for rows in blk[NEW_KEY].values():
                assert [r["h"] for r in rows] == [r["h"] for r
                                                  in blk["chan_skill"]]
                assert set(rows[0]) == {"h", "n", "msss_clim", "msss_pers",
                                        "msss_damped", "acc", "amp_ratio"}
            seen += 1
    assert seen > 0
    return want, seen


# ---------------------------------------------------------------------------
# 5. the new key reaches EVERY write point
# ---------------------------------------------------------------------------
def check_reaches_every_write(new):
    """`entry` is built once and mutated in place, so a key present when the
    scope blocks are assigned is present at every later write. That is a
    SOURCE-ORDER property, so it is checked on the source: the `skill_block`
    assignment must precede every `write_results` call in the head loop."""
    src = open(os.path.join(ML, "rollout_spatial.py")).read()
    i_assign = src.index("entry[name] = skill_block(")
    writes = [m.start() for m in re.finditer(r"\bwrite_results\(a\.out,", src)]
    before = [i for i in writes if i < i_assign]
    after = [i for i in writes if i > i_assign]
    # EXACTLY ONE write may precede the assignment: the `mark("started")`
    # stub, emitted before any head has been scored so that a job which dies
    # in its first minute still leaves a readable file. It carries no head
    # entry at all, so it cannot carry a scope block either.
    assert len(before) == 1 and 'mark("started")' in src[
        before[0]:before[0] + 200], src[before[0]:before[0] + 120]
    # every OTHER write — the scored / long / future / head_done partials and
    # the final unmarked one — happens after `entry` already holds the scope
    # blocks, and `entry` is mutated in place from there on.
    assert len(after) >= 5, after           # 4 partial + 1 final, at least
    # and the final artefact, which is the one every write point converges on
    for lab, e in new["heads"].items():
        for name, blk in e.items():
            if isinstance(blk, dict) and blk.get("chan_skill"):
                assert NEW_KEY in blk, (lab, name)
    return len(after)


# ---------------------------------------------------------------------------
# pytest entry points (cached: the end-to-end pair is run once per process)
# ---------------------------------------------------------------------------
_E2E = {}


def _e2e():
    if not _E2E:
        tmp = tempfile.mkdtemp(prefix="pc_skill_")
        _E2E["tmp"] = tmp
        try:
            f, new, old, sha, log = _roll_pair(tmp)
        except BaseException:
            shutil.rmtree(tmp, ignore_errors=True)
            _E2E.clear()
            raise
        _E2E.update(f=f, new=new, old=old, sha=sha, log=log)
    return _E2E


def test_pooled_purity():
    s = _e2e()
    check_pooled_purity(s["f"], s["new"], s["old"], s["sha"])


def test_consistency():
    check_consistency()


def test_known_answer():
    check_known_answer()


def test_channel_names():
    check_names_refusal()
    s = _e2e()
    check_names_are_the_tensors(s["f"], s["new"])


def test_reaches_every_write():
    check_reaches_every_write(_e2e()["new"])


def main():
    tmp = tempfile.mkdtemp(prefix="pc_skill_")
    try:
        print("--- 1. POOLED PURITY (the acceptance bar) ---")
        f, new, old, sha, _log = _roll_pair(tmp)
        n_scored, n_bytes, blocks = check_pooled_purity(f, new, old, sha)
        print(f"1. patched evaluator == pristine {sha[:8]} on the monthly "
              f"{f['P']}-pixel toy: {n_bytes:,} bytes of results are "
              f"BIT-IDENTICAL once the {len(blocks)} new `{NEW_KEY}` blocks "
              f"(one per scored scope, {n_scored} of them) are removed")

        print("--- 2. CONSISTENCY ---")
        dev_sum, dev_w = check_consistency()
        print(f"2. the pooled msss_clim recomposed from the per-channel sums "
              f"deviates by at most {dev_sum:.2e} (sum form) / {dev_w:.2e} "
              f"(climatology-mass-weighted form) — one to two ulps")

        print("--- 3. A KNOWN ANSWER ---")
        good, bad, mid, pooled = check_known_answer()
        print(f"3. perfect channel msss_clim {good['msss_clim']:+.3f} "
              f"(acc {good['acc']:+.3f}) · noise channel "
              f"{bad['msss_clim']:+.3f} (acc {bad['acc']:+.3f}) · middling "
              f"{mid['msss_clim']:+.3f} — while the POOLED row says "
              f"{pooled['msss_clim']:+.3f} and nothing else")

        print("--- 4. THE NAMES ARE THE TENSOR'S ---")
        n_bad = check_names_refusal()
        want, seen = check_names_are_the_tensors(f, new)
        print(f"4. {n_bad} wrong-length name lists refused; the evaluator's "
              f"`{NEW_KEY}` keys are the checkpoint's own {want} on all "
              f"{seen} scored scopes")

        print("--- 5. every write point ---")
        n_w = check_reaches_every_write(new)
        print(f"5. the scope blocks are assigned before all {n_w} "
              f"result-bearing `write_results(a.out, ...)` calls (the "
              f"scored/long/future/head_done partials and the final "
              f"unmarked write), and the final artefact carries the key on "
              f"every scored scope")
        print(f"\nper-channel skill: all 5 checks hold ✓")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""There must be exactly ONE anomaly transform in ml/, and it is
`trainprobe.anomaly_transform`.

WHY THIS TEST EXISTS. On 2026-08-17 the float16 z-score bug was found and
fixed -- in ONE file. `ml/train.py` carried a hand-inlined SECOND copy of the
same transform and was converted to a call at the same time. On 2026-08-18 a
THIRD copy was found in `ml/temporal.py` and a FOURTH in
`ml/probe_sequence.py`, both still carrying the pre-2026-08-17 arithmetic:

    v = X[..., c][np.isfinite(X[..., c]) & ~t_hold[...] & ~x_hold[...]]
    X[..., c] = (X[..., c] - v.mean()) / (v.std() + 1e-6)

`v.std()` without `dtype=np.float64` returns `inf` on a float16 pool of ~200M
squared residuals -- numpy upcasts the accumulator for np.mean but NOT for
np.std/np.var -- and `(X - mu) / (inf + 1e-6)` is EXACTLY 0.0. Families 4
(pentad) and 5 (daily) are float16, so stage 2 against either tensor would
have trained on all-zero dynamic channels while every loss, gpu_util and probe
still read healthy. Both copies also carried the ~249-traversal shape that sat
in run #389 for seven hours on the 165.6 GB daily tensor.

THE DEFECT WAS THE DUPLICATION, not either bug: a fix applied to one of four
copies leaves three wrong and nothing anywhere says so. So this test asserts
on the ABSENCE OF DUPLICATION rather than on any one symptom. A grep-shaped
test is the right tool for a grep-shaped defect -- the next copy will be
written by hand, will not be a paste, and so cannot be caught by comparing
against any fixed text.

HOW IT DECIDES. The transform is the CONJUNCTION of three parts:

  A. a month-of-year climatology built from train timesteps, or gathered
     back out by `[moy]`;
  B. a per-channel "is it dynamic" test built from nanstd(nanmean(...));
  C. a z-score whose denominator is `<something>.std(...) + 1e-N`.

Part C alone is everywhere and innocent -- `ridge_r` standardises its feature
matrix that way, and two files report amplitude ratios that way. Part A alone
is how any climatology is built. It is A-or-B TOGETHER WITH C that means "this
file re-implements the transform", and that is the rule below. A single line
that does the channel-slice z-score, `X[..., c] = (... ) / (v.std() + 1e-6)`,
is on its own conclusive and is checked separately.

Checks 2-4 exist because a pattern that matches nothing proves nothing
(ml/CLAUDE.md §0.2): they replay the deleted copies, a fully RENAMED
re-implementation, and the innocent standardisations, and assert the rule
fires on the first two and not on the third.

    python3 tests/test_one_anomaly_transform.py
"""
import os
import re
import sys

ML = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                  "..", "ml"))
CANONICAL = "trainprobe.py"

# NAMED exemptions. Every one carries the reason it is not a fifth copy in
# waiting. Adding a name here is a decision to be argued in the commit, not a
# way to make the test green.
EXEMPT = {
    CANONICAL: "owns the transform",
    # A DELIBERATE streaming replica (its module docstring says so), because
    # the audit runs over a memmap and the in-RAM recipe needs > 11 GB. It
    # verifies itself against the exact in-RAM recipe on two full channels
    # before it scores anything (`verify_streaming`, --skip-verify to bypass).
    # It cannot hit the float16 z-score overflow: every reduction in it is
    # float64 (`acc = np.zeros((C, 3), np.float64)`, `sum(dtype=np.float64)`)
    # or preceded by `.astype(np.float32)`. It has never run in any workflow
    # run (no log for runs #1-#396 mentions it) and is not wired into
    # ml-train.yml. Converting it to a chunked call on the canonical
    # transform is a SEPARATE task -- and when that happens, note that its
    # `n = np.zeros(..., np.uint8)` climatology counter assumes "<= 43 train
    # months per moy" and silently wraps at family 4's ~262 and family 5's
    # ~1309 timesteps per month-of-year.
    "recon_eval.py": "deliberate streaming replica, self-verified, float64/"
                     "float32 throughout, never run; tracked separately",
}

# --- the three parts, as regexes over CODE (comments/strings stripped) -----
CLIM = [
    (r"nanmean\s*\(\s*\w+\s*\[\s*\(\s*moy\s*==",
     "per-month climatology: nanmean(X[(moy == m) & ...])"),
    (r"[-=]\s*\w+(_\w+)?\s*\[\s*moy\s*(\]\s*\[|,)",
     "climatology gather indexed by moy"),
]
DYN = [
    (r"nanstd\s*\(\s*np\.nanmean\s*\(",
     "dynamic-channel test: nanstd(nanmean(...))"),
]
ZSCORE = [
    (r"/\s*\(\s*[\w\.\[\]]+\.std\s*\([^)]*\)\s*\+\s*1e-\d+\s*\)",
     "z-score by (<x>.std(...) + 1e-N)"),
    (r"/\s*\(\s*np\.nan?std\s*\([^()]*(\([^()]*\))?[^()]*\)\s*\+\s*1e-\d+\s*\)",
     "z-score by (np.std(...) + 1e-N)"),
]
# Conclusive on its own: the assignment writes back into a CHANNEL SLICE.
SMOKING_GUN = [
    (r"\[\s*\.\.\.\s*,\s*\w+\s*\]\s*[-+]?=[^\n]{0,200}?\.std\s*\([^)]*\)"
     r"\s*\+\s*1e-\d+",
     "channel-slice z-score assignment X[..., c] = (...) / (v.std() + 1e-N)"),
]


def _py_files():
    for root, dirs, files in os.walk(ML):
        dirs[:] = [d for d in dirs if d not in ("__pycache__", "cache", "runs")]
        for f in sorted(files):
            if f.endswith(".py"):
                yield os.path.join(root, f)


def strip_comments_and_strings(src):
    """Comments and docstrings DESCRIBE the transform all over ml/ (this test's
    own docstring quotes it verbatim), so the patterns must see executable code
    only. Blanks out `#` comments and every string literal, preserving line
    numbers."""
    out, i, n = [], 0, len(src)
    while i < n:
        ch = src[i]
        if ch == "#":
            j = src.find("\n", i)
            j = n if j < 0 else j
            out.append(" " * (j - i))
            i = j
        elif ch in "\"'":
            trip = src[i:i + 3]
            delim = trip if trip in ('"""', "'''") else ch
            j = i + len(delim)
            while j < n:
                if src[j] == "\\":
                    j += 2
                    continue
                if src.startswith(delim, j):
                    j += len(delim)
                    break
                j += 1
            else:
                j = n
            out.append("".join(c if c == "\n" else " " for c in src[i:j]))
            i = j
        else:
            out.append(ch)
            i += 1
    return "".join(out)


def _hits(code, group):
    """[(line, why, matched text)] for every pattern in `group`."""
    found = []
    for pat, why in group:
        for m in re.finditer(pat, code, re.S):
            found.append((code[:m.start()].count("\n") + 1, why,
                          " ".join(m.group(0).split())[:90]))
    return found


def verdict(code):
    """(is_reimplementation, evidence). The rule: the smoking gun on its own,
    or a z-score TOGETHER WITH a climatology or a dynamic-channel test."""
    gun = _hits(code, SMOKING_GUN)
    if gun:
        return True, gun
    z = _hits(code, ZSCORE)
    ctx = _hits(code, CLIM) + _hits(code, DYN)
    if z and ctx:
        return True, ctx + z
    return False, []


def main():
    canon = open(os.path.join(ML, CANONICAL), encoding="utf-8").read()
    assert "def anomaly_transform(" in canon, (
        f"ml/{CANONICAL} no longer defines anomaly_transform -- this test is "
        f"pointing at the wrong file and is asserting nothing")

    files = list(_py_files())
    offenders, callers = [], []
    for path in files:
        name = os.path.relpath(path, ML)
        src = open(path, encoding="utf-8").read()
        if "anomaly_transform" in src and name != CANONICAL:
            callers.append(name)
        if name in EXEMPT:
            continue
        bad, evidence = verdict(strip_comments_and_strings(src))
        if bad:
            offenders.append((name, evidence))

    print(f"checked {len(files)} files under ml/ "
          f"({len(EXEMPT)} named exemptions)")
    print("callers of trainprobe.anomaly_transform: "
          + ", ".join(sorted(callers)))
    if offenders:
        print("\nSECOND IMPLEMENTATION(S) OF THE ANOMALY TRANSFORM:")
        for name, ev in offenders:
            for line, why, text in ev:
                print(f"  ml/{name}:{line}  [{why}]\n      {text}")
    assert not offenders, (
        "%d file(s) under ml/ re-implement the anomaly transform: %s. There "
        "is exactly one of it (ml/trainprobe.py:anomaly_transform) and every "
        "reader must CALL it -- a fix to one copy is not a fix. That is how "
        "the float16 z-score bug survived in temporal.py and "
        "probe_sequence.py for a day after it was 'fixed'."
        % (len(offenders), ", ".join(n for n, _ in offenders)))
    print("  1. no unexempted file in ml/ carries a second implementation")

    # ---- 2: the rule still recognises the code it was written for --------
    DELETED_TEMPORAL = """
    dynamic = [c for c in range(C)
               if np.nanstd(np.nanmean(X[..., c], axis=(1, 2))) > 1e-6]
    clim = np.full((12, H, W, C), np.nan, dtype=np.float32)
    for m in range(12):
        clim[m] = np.nanmean(X[(moy == m) & ~t_hold], axis=0)
    for c in dynamic:
        X[..., c] = X[..., c] - clim[moy, :, :, c]
        v = X[..., c][np.isfinite(X[..., c]) & ~t_hold[:, None, None]
                      & ~x_hold[None, None, :]]
        X[..., c] = (X[..., c] - v.mean()) / (v.std() + 1e-6)
"""
    DELETED_SEQPROBE = DELETED_TEMPORAL.replace("clim[moy, :, :, c]",
                                                "clim[moy][..., c]")
    for label, text in (("temporal.py", DELETED_TEMPORAL),
                        ("probe_sequence.py", DELETED_SEQPROBE)):
        bad, ev = verdict(text)
        assert bad, f"the rule no longer recognises the deleted {label} copy"
        print(f"  2. the deleted {label} copy still trips it "
              f"({len(ev)} signature hit(s))")

    # ---- 3: a fully RENAMED re-implementation trips it too ---------------
    RENAMED = """
    live = [k for k in range(nchan)
            if np.nanstd(np.nanmean(cube[..., k], axis=(1, 2))) > 1e-6]
    monthly = np.full((12,) + cube.shape[1:], np.nan, np.float32)
    for mm in range(12):
        monthly[mm] = np.nanmean(cube[(moy == mm) & train], axis=0)
    for k in live:
        cube[..., k] -= monthly[moy, :, :, k]
        pool = cube[..., k][np.isfinite(cube[..., k])]
        cube[..., k] = (cube[..., k] - pool.mean()) / (pool.std() + 1e-9)
"""
    bad, ev = verdict(RENAMED)
    assert bad, ("a re-implementation with EVERY variable renamed slipped "
                 "through -- the rule is pinned to names, not to shape")
    print(f"  3. a fully renamed re-implementation trips it "
          f"({len(ev)} signature hit(s))")

    # ---- 4: the innocent standardisations do NOT trip it -----------------
    INNOCENT = {
        "ridge_r feature standardisation":
            "    F = (F - F[tr].mean(0)) / (F[tr].std(0) + 1e-9)\n",
        "an amplitude ratio":
            "    amp = float(mlast.std() / (mon_ztrue.std() + 1e-9))\n",
        "a plain monthly climatology with no z-score":
            "    for m in range(12):\n"
            "        clim[m] = np.nanmean(y[(moy == m) & ~t_hold], axis=0)\n"
            "    y = y - clim[moy]\n",
    }
    for label, text in INNOCENT.items():
        bad, ev = verdict(text)
        assert not bad, (f"false positive on {label}: {ev} -- a test that "
                         f"cries wolf gets an exemption added instead of a "
                         f"fix, which is how the exemption list rots")
    print(f"  4. {len(INNOCENT)} innocent standardisation(s) do not trip it")

    print("\ntests/test_one_anomaly_transform.py: all 4 checks passed")


if __name__ == "__main__":
    main()

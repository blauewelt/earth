#!/usr/bin/env python3
"""`--holdout-scope`: what the YEAR holdout actually excludes from the
stage-2 training pool.

THE BUG THIS FILE IS THE REGRESSION TEST FOR. The stage-2 loss is DENSE over
the window — `ml/temporal.py:win_ztgt` returns the measured embedding one bin
after EVERY one of the K frames, and `l_base` averages all of them — but the
pool only ever excluded a window whose FINAL scored bin (t+1, plus the unroll
fan and each `--direct` horizon) landed in a held-out year. So a window ending
in the K bins AFTER a held-out year carried that year's bins as CONTEXT and,
worse, as TEACHER-FORCED TARGETS: the held-out year's transitions were trained
into the weights, and every number read off those years was reading a year the
head had seen. On the pentad axis the arithmetic is exact and was what found
it: the recorded pool is 240,933,742 = 86,698 pixels x 2,779 end-bins, and
2,779 = 3,142 - 219 (holdout bins) - 144 (no-context prefix) EXACTLY — nothing
but the endpoint was ever excluded.

WHAT IS CHECKED, all on one small synthetic axis so every number here is one a
reader can recompute by hand:

  1. **The endpoint pool is unchanged.** `build_window_pool` at the default
     scope equals an INDEPENDENT reimplementation of the legacy three-term
     condition, written inline below, element for element — and its count
     equals a closed form this test derives from the toy's own layout. The
     archive has to stay reproducible; this is the check that says it does.
  2. **The window certificate, brute forced.** At `scope="window"`, every
     pooled t is walked and every bin its forward pass touches is tested
     directly; and the number of ADDITIONAL end-bins excluded equals the
     test's own closed form for the toy.
  3. **The bug is real, and the fix closes it.** At the default scope the
     test EXHIBITS pooled windows whose within-window targets include a
     held-out bin. At `window` there are none.
  4. **The two trainers build the SAME pool**, at both scopes — the JAX
     stage-2 trainer's own call, made the way its source makes it, against
     `ml/temporal.py`'s.
  5. **Both trainers RECORD the setting** in `stage2_config`.

No pytest. Numbered checks, printed with their measured numbers.

    python3 tests/test_holdout_scope.py
"""
import ast
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
ML = os.path.join(ROOT, "ml")
sys.path.insert(0, ML)

from temporal import (build_window_pool, frame_ref,          # noqa: E402
                      frame_steps, window_touch_offsets)

# ---- the toy axis ---------------------------------------------------------
# T=120 bins, K=8 frames, ONE contiguous holdout block mid-axis and ONE that
# runs to the end of the record — the two shapes that matter, because a block
# in the middle is escaped on both sides and a block at the end is not.
T = 120
K = 8
HOLD_A = (40, 52)          # [40, 52) — 12 bins, mid-axis
HOLD_B = (112, 120)        # [112, 120) — 8 bins, to the end
CTX_BACK = K - 1           # contiguous stencil: the earliest frame is t-K+1
REACH = [1]                # unroll 1, no --direct: the archived arm


def toy_hold():
    th = np.zeros(T, bool)
    th[HOLD_A[0]:HOLD_A[1]] = True
    th[HOLD_B[0]:HOLD_B[1]] = True
    return th


def legacy_ok_t(T_, t_hold, K_, reach, ctx_back):
    """THE LEGACY RULE, REIMPLEMENTED HERE from the description rather than
    imported: a window ending at t is eligible iff its whole scored reach
    exists, its earliest frame exists, and no SCORED bin t+r is held out.
    Written as a plain loop with explicit branches so it shares no expression
    with `build_window_pool` beyond the rule itself."""
    out = np.zeros(T_, bool)
    for t in range(T_):
        if t + max(reach) >= T_:
            continue
        if t < ctx_back:
            continue
        if any(bool(t_hold[t + r]) for r in reach):
            continue
        out[t] = True
    return out


def touched_bins(t, K_, foff, reach):
    """Every bin ONE forward pass reads or is scored on, rebuilt from the
    gather machinery (`frame_ref`/`frame_steps`) the trainer itself uses."""
    ref = int(frame_ref(int(t), K_, foff))
    bins = set()
    for j in frame_steps(K_, foff):
        bins.add(ref + int(j))          # the frame the model reads
        bins.add(ref + int(j) + 1)      # its teacher-forced target
    for r in reach:
        bins.add(int(t) + int(r))
    return sorted(bins)


def fail(msg):
    print(f"\nFAILED: {msg}")
    raise SystemExit(1)


def main():
    th = toy_hold()
    n_hold = int(th.sum())
    print(f"toy axis: T={T} · K={K} · reach={REACH} · CTX_BACK={CTX_BACK} · "
          f"holdout bins {HOLD_A[0]}..{HOLD_A[1] - 1} and "
          f"{HOLD_B[0]}..{HOLD_B[1] - 1} ({n_hold} of {T})")

    # ================= 1 · the endpoint pool is unchanged =================
    ok_end = build_window_pool(T, th, K, None, REACH, CTX_BACK)
    ref_end = legacy_ok_t(T, th, K, REACH, CTX_BACK)
    diff = int((ok_end != ref_end).sum())
    if diff:
        fail(f"1: the default scope moved {diff} of {T} end-bins away from "
             f"the legacy rule — the archive is no longer reproducible")
    # THE CLOSED FORM, from this test's own arithmetic. Candidates are
    # t in [CTX_BACK, T - max(reach) - 1]; of those, t is dropped iff t+1 is
    # held out, i.e. t in [lo-1, hi-1] for each holdout block [lo, hi).
    cand = list(range(CTX_BACK, T - max(REACH)))
    drop_end = sum(1 for t in cand if any(lo - 1 <= t <= hi - 2
                                          for lo, hi in (HOLD_A, HOLD_B)))
    want_end = len(cand) - drop_end
    if int(ok_end.sum()) != want_end:
        fail(f"1: endpoint pool is {int(ok_end.sum())} end-bins, closed form "
             f"says {want_end}")
    print(f"\n1 · endpoint identity — the default scope equals an independent "
          f"reimplementation of the legacy three-term condition on all {T} "
          f"bins (0 differ), and its size {int(ok_end.sum())} equals the "
          f"closed form {len(cand)} candidates - {drop_end} whose t+1 is "
          f"held out = {want_end}. OK")

    # ================= 2 · the window certificate =========================
    ok_win = build_window_pool(T, th, K, None, REACH, CTX_BACK, scope="window")
    bad = [int(t) for t in np.where(ok_win)[0]
           if any(bool(th[b]) for b in touched_bins(t, K, None, REACH))]
    if bad:
        fail(f"2: {len(bad)} pooled windows still touch a holdout bin "
             f"(first {bad[:5]}) — the certificate the trainer prints is "
             f"checking the wrong thing")
    # CLOSED FORM for the window rule: t is dropped iff the span
    # [t-CTX_BACK, t+max(reach)] intersects a holdout block.
    drop_win = sum(1 for t in cand
                   if any(t - CTX_BACK <= hi - 1 and t + max(REACH) >= lo
                          for lo, hi in (HOLD_A, HOLD_B)))
    want_win = len(cand) - drop_win
    if int(ok_win.sum()) != want_win:
        fail(f"2: window pool is {int(ok_win.sum())}, closed form says "
             f"{want_win}")
    extra = int(ok_end.sum()) - int(ok_win.sum())
    if extra != drop_win - drop_end:
        fail(f"2: additional exclusions {extra} != closed form "
             f"{drop_win - drop_end}")
    print(f"2 · window certificate — brute force over all {int(ok_win.sum())} "
          f"pooled end-bins x {len(window_touch_offsets(K, None, REACH))} "
          f"touched bins each: 0 touch a held-out bin. The window rule "
          f"excluded {extra} end-bins MORE than the endpoint rule, which is "
          f"exactly the closed form {drop_win} - {drop_end}. OK")

    # ============ 3 · the bug is real, and the fix closes it ==============
    # The regression test proper: at the DEFAULT scope, find pooled windows
    # whose WITHIN-WINDOW teacher-forced targets include a held-out bin. Not
    # the endpoint t+1 — that one the legacy rule does exclude — the earlier
    # frames' targets, which it never looked at.
    def leaky(mask):
        out = []
        for t in np.where(mask)[0]:
            ref = int(frame_ref(int(t), K, None))
            tgts = [ref + int(j) + 1 for j in frame_steps(K, None)]
            hit = [b for b in tgts if bool(th[b])]
            if hit:
                out.append((int(t), hit))
        return out

    leak_end = leaky(ok_end)
    leak_win = leaky(ok_win)
    if not leak_end:
        fail("3: no leaking window found at the default scope — the fixture "
             "cannot demonstrate the bug, so it cannot guard the fix")
    if leak_win:
        fail(f"3: {len(leak_win)} leaking windows SURVIVE at scope=window "
             f"(first {leak_win[0]}) — the fix does not close the leak")
    t0, hit0 = leak_end[0]
    ctx0 = touched_bins(t0, K, None, REACH)
    print(f"3 · the bug, exhibited — at scope=endpoint {len(leak_end)} of "
          f"{int(ok_end.sum())} pooled windows are scored on a HELD-OUT bin "
          f"inside the window. Example: the window ending at t={t0} spans "
          f"bins {ctx0[0]}..{ctx0[-1]}, its per-frame targets include "
          f"{hit0} — every one of them a held-out bin, teacher-forced into "
          f"the weights. At scope=window: {len(leak_win)} such windows. OK")

    # ================= 4 · torch and JAX build one pool ===================
    # The JAX trainer's OWN call, made the way its source makes it. Read the
    # arguments out of the file rather than retyping them, so a divergence in
    # the port is a failure here and not a comment that went stale.
    src = open(os.path.join(ML, "jaxport", "train_stage2.py")).read()
    if "from temporal import" not in src or "build_window_pool" not in src:
        fail("4: ml/jaxport/train_stage2.py no longer imports "
             "build_window_pool — the two trainers can drift again")
    call = None
    for node in ast.walk(ast.parse(src)):
        if (isinstance(node, ast.Call)
                and isinstance(node.func, ast.Name)
                and node.func.id == "build_window_pool"):
            call = node
    if call is None:
        fail("4: no build_window_pool(...) call in the JAX trainer")
    # positional args: (T, t_hold, K, FOFF, reach, CTX_BACK)
    got = [ast.unparse(x) for x in call.args]
    want = ["T", "t_hold", "K", "None", "[1]", "K - 1"]
    if got != want:
        fail(f"4: the JAX trainer calls build_window_pool{tuple(got)}; this "
             f"test only certifies the equality for {tuple(want)}")
    import jaxport.train_stage2 as js                          # noqa: E402
    for scope in ("endpoint", "window"):
        a = build_window_pool(T, th, K, None, REACH, CTX_BACK, scope=scope,
                              quiet=True)
        b = js.build_window_pool(T, th, K, None, [1], K - 1, scope=scope,
                                 quiet=True)
        d = int((a != b).sum())
        if d:
            fail(f"4: torch and JAX pools differ on {d} bins at scope={scope}")
    # ...and the JAX trainer's own legacy assertion, reproduced here: the
    # shared call must equal the literal expression that file used to run.
    jax_legacy = np.array([t + 1 < T and t + 1 >= K and not th[t + 1]
                           for t in range(T)])
    if not np.array_equal(jax_legacy, ok_end):
        fail("4: build_window_pool(endpoint) != the JAX trainer's pre-change "
             "expression — a TPU arm would no longer reproduce its archive")
    print(f"4 · torch vs JAX — one imported definition, called by the JAX "
          f"trainer as build_window_pool(T, t_hold, K, None, [1], K - 1): "
          f"identical pools at BOTH scopes (0 of {T} bins differ), and the "
          f"endpoint pool still equals that file's own literal "
          f"`t + 1 < T and t + 1 >= K and not t_hold[t + 1]`. OK")

    # ================= 5 · both trainers record it ========================
    # Constructing either config dict means loading a tensor, a codec and a
    # frozen Z; the key's presence is a source fact, so it is checked as one.
    tsrc = open(os.path.join(ML, "temporal.py")).read()
    for name, s in (("ml/temporal.py", tsrc),
                    ("ml/jaxport/train_stage2.py", src)):
        if '"holdout_scope": a.holdout_scope' not in s:
            fail(f"5: {name} does not write holdout_scope into its "
                 f"stage2_config — a reader would have to infer which pool "
                 f"a curve came from")
        if '"--holdout-scope"' not in s:
            fail(f"5: {name} does not define --holdout-scope")
        if 'choices=("endpoint", "window")' not in s:
            fail(f"5: {name}'s --holdout-scope does not offer exactly "
                 f"endpoint|window")
    sh = open(os.path.join(ML, "jaxport", "tpu_train_s2.sh")).read()
    for want_s in ('HOLDOUT_SCOPE="${HOLDOUT_SCOPE:-endpoint}"',
                   '--holdout-scope "${HOLDOUT_SCOPE}"',
                   'holdout_scope ${HOLDOUT_SCOPE}'):
        if want_s not in sh:
            fail(f"5: ml/jaxport/tpu_train_s2.sh is missing {want_s!r} — a "
                 f"launch log would not state which pool the node trained on")
    print("5 · the setting is RECORDED — both trainers define "
          "--holdout-scope endpoint|window and write `holdout_scope` into "
          "stage2_config; tpu_train_s2.sh carries the HOLDOUT_SCOPE knob, "
          "passes it, and names it in the resolved-knobs line. OK")

    print(f"\nALL 5 CHECKS PASSED · toy T={T} K={K}: endpoint pool "
          f"{int(ok_end.sum())} end-bins ({len(leak_end)} of them leaking), "
          f"window pool {int(ok_win.sum())} (0 leaking).")


if __name__ == "__main__":
    main()

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
  4. **The two trainers build the SAME pool**, at all three scopes — the JAX
     stage-2 trainer's own call, made the way its source makes it, against
     `ml/temporal.py`'s.
  5. **Both trainers RECORD the setting** in `stage2_config`.
  6. **The three-scope table on the toy**, every number recomputed by this
     file's own arithmetic rather than read back from the implementation.
  7. **`target`'s pool is the legacy pool, ELEMENT FOR ELEMENT** — the whole
     of that scope is the LOSS, so a single moved bin would make it a third
     pool nobody asked for.
  8. **`frame_target_keep` is correct by brute force** over every pooled
     window and every frame.
  9. **The masked loss REDUCES to the legacy loss where nothing is masked** —
     both expressions on the same tensors, over a pool region with no
     held-out target, with the measured deviation printed.
 10. **Under `target` the leak is CLOSED for targets and OPEN for context** —
     a pooled window is exhibited whose context contains a held-out bin while
     no unmasked target does. This is the check that documents why `target`
     and `window` are two settings and not one.
 11. **The FGN x target decision is pinned** — the mask IS applied to the
     fair-CRPS objective (no refusal), exactly, and its value matches a hand
     computation.

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

import torch                                                    # noqa: E402

from temporal import (HOLDOUT_SCOPES, build_window_pool,        # noqa: E402
                      fair_crps2, fair_crps2_elem, frame_ref,
                      frame_steps, frame_target_keep,
                      window_touch_offsets)

LEGACY = "endpoint_contaminated"

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
    ok_end = build_window_pool(T, th, K, None, REACH, CTX_BACK, scope=LEGACY)
    ref_end = legacy_ok_t(T, th, K, REACH, CTX_BACK)
    diff = int((ok_end != ref_end).sum())
    if diff:
        fail(f"1: scope={LEGACY} moved {diff} of {T} end-bins away from the "
             f"legacy rule — the archive is no longer reproducible")
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
    print(f"\n1 · endpoint identity — scope={LEGACY} equals an independent "
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
    print(f"3 · the bug, exhibited — at scope={LEGACY} {len(leak_end)} of "
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
    for scope in HOLDOUT_SCOPES:
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
        fail(f"4: build_window_pool({LEGACY}) != the JAX trainer's pre-change "
             f"expression — a TPU arm would no longer reproduce its archive")
    if "frame_target_keep" not in src:
        fail("4: the JAX trainer does not import frame_target_keep — the two "
             "trainers can drift on WHICH TARGETS a head may learn from")
    print(f"4 · torch vs JAX — one imported definition, called by the JAX "
          f"trainer as build_window_pool(T, t_hold, K, None, [1], K - 1): "
          f"identical pools at ALL {len(HOLDOUT_SCOPES)} scopes (0 of {T} "
          f"bins differ), the legacy pool still equals that file's own "
          f"literal `t + 1 < T and t + 1 >= K and not t_hold[t + 1]`, and "
          f"the loss mask is the imported frame_target_keep. OK")

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
        if "choices=HOLDOUT_SCOPES" not in s:
            fail(f"5: {name}'s --holdout-scope does not offer exactly "
                 f"{HOLDOUT_SCOPES}")
        if 'default="window"' not in s:
            fail(f"5: {name}'s --holdout-scope does not DEFAULT to window — "
                 f"contamination must never be the default")
        if '"holdout_masked_frac": round(HOLD_MASKED_FRAC, 6)' not in s:
            fail(f"5: {name} does not write holdout_masked_frac into its "
                 f"stage2_config — an artefact would not say how much of the "
                 f"objective its scope removed")
    sh = open(os.path.join(ML, "jaxport", "tpu_train_s2.sh")).read()
    for want_s in ('HOLDOUT_SCOPE="${HOLDOUT_SCOPE:-window}"',
                   '--holdout-scope "${HOLDOUT_SCOPE}"',
                   'holdout_scope ${HOLDOUT_SCOPE}'):
        if want_s not in sh:
            fail(f"5: ml/jaxport/tpu_train_s2.sh is missing {want_s!r} — a "
                 f"launch log would not state which pool the node trained on")
    print(f"5 · the setting is RECORDED — both trainers define "
          f"--holdout-scope {'|'.join(HOLDOUT_SCOPES)}, DEFAULT window, and "
          f"write `holdout_scope` + `holdout_masked_frac` into "
          f"stage2_config; tpu_train_s2.sh defaults HOLDOUT_SCOPE to window, "
          f"passes it, and names it in the resolved-knobs line. OK")


    # ============== 6 · the three-scope table, recomputed =================
    # `target`'s pool is the legacy one, so the only thing it changes is how
    # many per-frame TARGETS are scored. Every number below is this file's
    # own arithmetic: a plain loop over pooled windows counting held-out
    # target bins, sharing nothing with `frame_target_keep` but the rule.
    ok_tgt = build_window_pool(T, th, K, None, REACH, CTX_BACK,
                               scope="target", quiet=True)
    n_frames = len(list(frame_steps(K, None)))

    def masked_by_hand(mask):
        """#(window, frame) pairs whose TARGET bin is held out, counted the
        long way: for each pooled t, the target bins are the K bins
        frame_ref(t)+j+1, and each is tested against t_hold on its own."""
        n = 0
        for t in np.where(mask)[0]:
            ref = int(frame_ref(int(t), K, None))
            for j in frame_steps(K, None):
                n += 1 if bool(th[ref + int(j) + 1]) else 0
        return n

    tbl = []
    for scope, ok in ((LEGACY, ok_end), ("target", ok_tgt),
                      ("window", ok_win)):
        n_end = int(ok.sum())
        n_tot = n_end * n_frames
        n_msk = masked_by_hand(ok) if scope == "target" else 0
        tbl.append((scope, n_end, n_tot - n_msk, n_msk))
    base = tbl[0][2]
    if tbl[1][1] != tbl[0][1]:
        fail(f"6: target keeps {tbl[1][1]} end-bins, the legacy pool has "
             f"{tbl[0][1]} — target must not change the POOL at all")
    if not (tbl[2][2] < tbl[1][2] < tbl[0][2]):
        fail(f"6: the three scopes are not strictly ordered in scored "
             f"frame-targets: {[r[2] for r in tbl]}")
    # window scores every frame of every window it keeps, and it keeps only
    # windows that touch nothing held out — so masked_by_hand must be 0 there.
    if masked_by_hand(ok_win) != 0:
        fail(f"6: {masked_by_hand(ok_win)} frame-targets are held out inside "
             f"the WINDOW pool — that pool is supposed to make the mask "
             f"vacuous")
    print(f"\n6 · the three-scope table on the toy (T={T}, K={K}, "
          f"{n_hold} held-out bins), every count recomputed here:")
    print(f"      {'scope':<22} {'end-bins':>9} {'frame-targets':>14} "
          f"{'vs legacy':>10}")
    for scope, n_end, n_tgt, n_msk in tbl:
        print(f"      {scope:<22} {n_end:>9,} {n_tgt:>14,} "
              f"{100.0 * n_tgt / base:>9.1f}%")
    print(f"    target masks {tbl[1][3]} of {base} frame-targets "
          f"({100.0 * tbl[1][3] / base:.2f}%) and keeps every end-bin; "
          f"window drops {tbl[0][1] - tbl[2][1]} end-bins and "
          f"{100.0 * (base - tbl[2][2]) / base:.2f}% of the frame-targets. OK")

    # ============ 7 · target's pool IS the legacy pool =====================
    if not np.array_equal(ok_tgt, ok_end):
        fail(f"7: scope=target returned a pool differing from "
             f"scope={LEGACY} on {int((ok_tgt != ok_end).sum())} of {T} "
             f"bins — the whole of that scope is the LOSS, and a moved bin "
             f"makes it a third pool nobody asked for")
    print(f"7 · target's pool — element for element identical to "
          f"scope={LEGACY} on all {T} bins (both are the legacy expression, "
          f"returned unmodified). OK")

    # ============ 8 · frame_target_keep, brute forced ======================
    idx = np.where(ok_tgt)[0]
    keep = frame_target_keep(idx, K, None, th)
    if keep.shape != (len(idx), n_frames):
        fail(f"8: frame_target_keep returned {keep.shape}, expected "
             f"{(len(idx), n_frames)}")
    bad8 = []
    for r, t in enumerate(idx):
        ref = int(frame_ref(int(t), K, None))
        for c, j in enumerate(frame_steps(K, None)):
            want = not bool(th[ref + int(j) + 1])
            if bool(keep[r, c]) != want:
                bad8.append((int(t), int(j)))
    if bad8:
        fail(f"8: frame_target_keep disagrees with the brute-force target "
             f"bin on {len(bad8)} (window, frame) pairs (first {bad8[:4]})")
    if not keep.any(1).all():
        fail(f"8: {int((~keep.any(1)).sum())} pooled windows have EVERY "
             f"target masked — the reach condition is supposed to keep t+1")
    print(f"8 · frame_target_keep — brute force over all {len(idx)} pooled "
          f"windows x {n_frames} frames ({keep.size} pairs), rebuilding each "
          f"target bin as frame_ref(t)+j+1: 0 disagreements, and every "
          f"window keeps at least one target (min {int(keep.sum(1).min())} "
          f"of {n_frames}). OK")

    # ====== 9 · the masked loss IS the legacy loss where nothing is masked ==
    # A pool region with NO held-out target: the windows whose keep row is
    # all True. Both expressions, the trainer's own two statements, on the
    # same tensors.
    full = idx[keep.all(1)]
    if len(full) < 8:
        fail(f"9: only {len(full)} fully-unmasked windows on the toy — the "
             f"fixture cannot exercise the reduction")
    B, d_z = 16, 4
    g = torch.Generator().manual_seed(7)
    pred = torch.randn(B, n_frames, d_z, generator=g)
    ztgt = torch.randn(B, n_frames, d_z, generator=g)
    m = torch.as_tensor(keep[keep.all(1)][:B]).float().unsqueeze(-1)
    if float(m.min()) != 1.0:
        fail("9: the fixture's mask is not all-ones on the unmasked region")
    legacy_l = (pred - ztgt).pow(2).mean()
    masked_l = ((pred - ztgt).pow(2) * m).sum() / (m.sum() * d_z)
    dev = abs(float(masked_l) - float(legacy_l)) / max(1e-30,
                                                       abs(float(legacy_l)))
    if not (dev < 1e-12):
        fail(f"9: the masked loss and the legacy loss differ by {dev:.3e} "
             f"relative on an all-ones mask — they are mathematically the "
             f"same number there")
    print(f"9 · the masked loss reduces to the legacy loss — on {B} windows "
          f"drawn from the {len(full)} pooled windows with NO held-out "
          f"target, `(pred-ztgt).pow(2).mean()` = {float(legacy_l):.9f} and "
          f"`((pred-ztgt).pow(2)*keep).sum()/(keep.sum()*d_z)` = "
          f"{float(masked_l):.9f}: measured relative deviation {dev:.3e} "
          f"(< 1e-12). OK")

    # ==== 10 · under `target` the leak is CLOSED for targets, OPEN for ctx ==
    # This is the check that documents why `target` and `window` are two
    # settings. `target` guarantees no UNMASKED target is held out; it
    # deliberately does NOT stop a held-out bin being read as CONTEXT.
    leaky_ctx = []
    bad_tgt = []
    for r, t in enumerate(idx):
        ref = int(frame_ref(int(t), K, None))
        ctx = [ref + int(j) for j in frame_steps(K, None)]
        hit_ctx = [b for b in ctx if bool(th[b])]
        for c, j in enumerate(frame_steps(K, None)):
            if bool(keep[r, c]) and bool(th[ref + int(j) + 1]):
                bad_tgt.append((int(t), int(j)))
        if hit_ctx:
            leaky_ctx.append((int(t), hit_ctx))
    if bad_tgt:
        fail(f"10: {len(bad_tgt)} UNMASKED targets are held-out bins under "
             f"scope=target (first {bad_tgt[:4]}) — the leak is not closed")
    if not leaky_ctx:
        fail("10: no pooled window under scope=target reads a held-out bin "
             "as context — the fixture cannot demonstrate that target is "
             "deliberately weaker than window, so it cannot guard it")
    # ...and `window` admits neither, which is the difference itself.
    ctx_win = [int(t) for t in np.where(ok_win)[0]
               if any(bool(th[int(frame_ref(int(t), K, None)) + int(j)])
                      for j in frame_steps(K, None))]
    if ctx_win:
        fail(f"10: {len(ctx_win)} windows in the WINDOW pool read a held-out "
             f"bin as context (first {ctx_win[:4]})")
    t10, ctx10 = leaky_ctx[0]
    r10 = int(np.where(idx == t10)[0][0])
    print(f"10 · target closes the TARGET leak and leaves the CONTEXT leak "
          f"open, on purpose — 0 of {int(keep.sum())} scored targets is a "
          f"held-out bin, while {len(leaky_ctx)} of {len(idx)} pooled "
          f"windows still READ one. Example: the window ending at t={t10} "
          f"reads held-out bins {ctx10} as context and has "
          f"{int((~keep[r10]).sum())} of its {n_frames} targets masked. "
          f"Under scope=window {len(ctx_win)} windows read a held-out bin — "
          f"that difference IS the two settings. OK")

    # ============ 11 · the FGN x target decision, pinned ===================
    # The mask CAN be applied to fair_crps2 exactly, because the estimator is
    # three terms combined elementwise and reduced ONCE — so it IS applied,
    # and the combination is NOT refused. Both halves are pinned: the source
    # carries no refusal, and the value matches a hand computation.
    if "fair_crps2_elem" not in tsrc:
        fail("11: ml/temporal.py has no fair_crps2_elem — the FGN objective "
             "cannot be masked exactly without it")
    want11 = ["l_base = ((fair_crps2_elem(p1, p2, ztgt) * wkeep).sum()",
              "/ (wkeep.sum() * ztgt.shape[-1]))"]
    if not all(w in tsrc for w in want11):
        fail("11: ml/temporal.py's FGN branch does not carry the masked "
             "fair-CRPS statement — the decision is unpinned")
    if "l_base = fair_crps2(p1, p2, ztgt)" not in tsrc:
        fail("11: ml/temporal.py's FGN branch no longer carries the LEGACY "
             "fair_crps2 statement for the unmasked scopes")
    p1 = torch.randn(B, n_frames, d_z, generator=g)
    p2 = torch.randn(B, n_frames, d_z, generator=g)
    y = torch.randn(B, n_frames, d_z, generator=g)
    # A batch that MIXES masked and unmasked windows — the first B pooled
    # windows sit before the first holdout block and would mask nothing, so
    # a mask-blind implementation would pass unnoticed.
    sel = np.concatenate([np.where(~keep.all(1))[0],
                          np.where(keep.all(1))[0]])[:B]
    km = torch.as_tensor(keep[sel]).float().unsqueeze(-1)
    got11 = float((fair_crps2_elem(p1, p2, y) * km).sum()
                  / (km.sum() * d_z))
    # THE HAND COMPUTATION, in float64 numpy, one (window, frame) at a time.
    a1, a2, ay = (x.double().numpy() for x in (p1, p2, y))
    kb = keep[sel]
    acc, cnt = 0.0, 0
    for b in range(B):
        for c in range(n_frames):
            if not kb[b, c]:
                continue
            for e in range(d_z):
                acc += (0.5 * (abs(a1[b, c, e] - ay[b, c, e])
                               + abs(a2[b, c, e] - ay[b, c, e]))
                        - 0.5 * abs(a1[b, c, e] - a2[b, c, e]))
                cnt += 1
    hand11 = acc / cnt
    if cnt != int(kb.sum()) * d_z:
        fail(f"11: the hand computation summed {cnt} elements, the mask says "
             f"{int(kb.sum()) * d_z}")
    if int((~kb).sum()) == 0:
        fail("11: the fixture's mask masks nothing — it cannot tell a masked "
             "CRPS from an unmasked one")
    rel11 = abs(got11 - hand11) / abs(hand11)
    if not (rel11 < 1e-6):
        fail(f"11: the masked fair CRPS is {got11!r}, the hand computation "
             f"says {hand11!r} (relative {rel11:.3e})")
    unmasked11 = float(fair_crps2(p1, p2, y))
    print(f"11 · FGN x target — NOT refused: fair_crps2 is elementwise then "
          f"reduced ONCE, so the mask applies exactly through "
          f"fair_crps2_elem. Measured on {B}x{n_frames} windows with "
          f"{int((~kb).sum())} frames masked: the trainer's expression gives "
          f"{got11:.9f}, an elementwise float64 hand computation over the "
          f"{cnt} kept elements gives {hand11:.9f} (relative {rel11:.2e}); "
          f"the unmasked estimator over the same tensors is "
          f"{unmasked11:.9f}. OK")

    print(f"\nALL 11 CHECKS PASSED · toy T={T} K={K}: {LEGACY} pool "
          f"{int(ok_end.sum())} end-bins ({len(leak_end)} of them leaking) "
          f"and all {tbl[0][2]} frame-targets scored · target pool "
          f"{int(ok_tgt.sum())} end-bins with {tbl[1][3]} frame-targets "
          f"masked ({100.0 * tbl[1][3] / base:.2f}%) and {len(leaky_ctx)} "
          f"windows still reading held-out CONTEXT · window pool "
          f"{int(ok_win.sum())} end-bins, nothing held out anywhere.")


if __name__ == "__main__":
    main()

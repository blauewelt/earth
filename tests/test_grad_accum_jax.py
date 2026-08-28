#!/usr/bin/env python3
"""E-054b · the GRADIENT-ACCUMULATION certificate for the JAX stage-2 trainer.
CPU, fp32, toy heads, no tensor, no codec, no network.

WHY IT EXISTS. 2026-08-28 00:15Z the registered HBM risk fired: the 400M rung
(1280x20, K 144, batch 256, 399.948M params) asked for 5.09 G with 4.03 G free
inside `train_step` on a v5e-4 chip and died at the first step. Of the four
recorded options only ONE keeps the comparison with E-051 exact — accumulate
over N micro-batches of `batch/N` and take one AdamW update on their average.
That option is only worth anything if the step it takes is *the same step*, so
this file is the evidence, not the reassurance.

  **1 · EXACTNESS.** batch 32 / accum 1 against batch 32 / accum 2, same
  weights, same data, same key. Asserted at three depths, because a single
  end-to-end number cannot say WHERE a difference came from:
    (a) the averaged gradient against the whole-batch gradient — the actual
        mathematical claim, and the only one with no optimiser in it;
    (b) the parameter deltas under PLAIN SGD, where the update is LINEAR in
        the gradient, so every coordinate is comparable and nothing is
        excluded;
    (c) the parameter deltas under the trainer's OWN AdamW chain, over the
        entries where a first Adam step can resolve a gradient known only to
        the measured association error (`_wellcond` DERIVES the threshold
        from that error rather than picking one), beside the algebraic bound
        `2 lr` over EVERY entry, and beside the parameter-level agreement.
        Same phenomenon and same treatment as `test_jaxport_train_s2.py`'s
        G5b gate: Adam's first update is ~sign(g), so a coordinate whose
        gradient is at the reduction's own noise floor moves by O(lr) in a
        direction the last bit decides — in BOTH modes — and no tolerance on
        parameter deltas can survive it.
  Then the same at three CONSECUTIVE steps, because a per-step 1e-7 that
  compounded would be a different finding from one that does not.

  **2 · FLAG-OFF PURITY.** `--grad-accum 1` must be the pre-E-054b path, and
  "must" is checked two ways: the accumulating graphs are not built AT ALL
  (`steps.micro is None`), and one step is BITWISE equal to an independent
  re-transcription of the pristine step written inline here — plus an equal
  operation count in the traced jaxpr, so a future edit cannot slip an op into
  the un-accumulated graph and still pass on the bits.

  **3 · REFUSALS.** `batch % N != 0` (a ragged micro-batch is a different
  optimisation, not a rounding detail) and `N < 1`.

  **4 · FGN + accumulation.** A crps2 toy step at accum 2 runs, and the ε
  stream separates (micro, forward): micro 0 forward 0 must differ from micro
  1 forward 0, or the two micro-batches would be scoring correlated members.
  The legacy two-level fold is pinned bitwise as well, because `--grad-accum
  1` passes `micro_index=None` and a resumed FGN run's ε must not move.

  **5 · --grad-clip.** With a deliberately huge gradient the clip must bind on
  the AVERAGED norm identically at N = 1 and N = 2 — the property that lets an
  accumulated run reuse an unaccumulated arm's clip threshold.

    python3 -m pytest tests/test_grad_accum_jax.py -x -q
    python3 tests/test_grad_accum_jax.py           # the same checks, verbose
"""
import math
import os
import sys
import warnings

import numpy as np

warnings.filterwarnings("ignore", message=".*enable_nested_tensor.*")

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
ML = os.path.join(ROOT, "ml")
if ML not in sys.path:
    sys.path.insert(0, ML)

import jax                                                       # noqa: E402
import jax.numpy as jnp                                          # noqa: E402
import optax                                                     # noqa: E402
from flax import nnx                                             # noqa: E402

from jaxport import models as jm                                 # noqa: E402
from jaxport.train_stage2 import (accum_step,                    # noqa: E402
                                  build_train_steps, fgn_eps_at,
                                  fgn_train_key, grad_accum_micro,
                                  make_set_lr, parse)

jax.config.update("jax_enable_x64", False)

TOY = dict(d_z=4, d_model=16, n_heads=4, n_layers=2, k_max=6)
K, BATCH = 6, 32
# lr 1e-2, the same toy learning rate `test_jaxport_train_s2.py`'s G5b uses,
# and here it is also what makes the DELTA comparison measurable at all. A
# parameter delta is a difference of two fp32 numbers of size |w| ~ 0.3, so it
# carries ulp(0.3) ~ 3e-8 of pure REPRESENTATION noise no matter how equal the
# two updates were: at lr 1e-3 that noise alone is 3e-5 of a step and no
# implementation could report better; at lr 1e-2 it is 3e-6, under the
# tolerance. The measured floor is printed beside every delta number below so
# the reader can see which of the two they are looking at, and the
# PARAMETER-level agreement (immune to the cancellation) is asserted too.
LR = 1e-2

# Each check RECORDS its one-line finding rather than returning it (pytest >= 9
# warns on a test that returns a value); `python3 tests/<file>.py` prints them.
NOTES = []


def _note(msg):
    NOTES.append(msg)
    print("  " + msg, flush=True)


# --------------------------------------------------------------------------
# the toy
# --------------------------------------------------------------------------
def _build(seed=0, clip=0.0, eps_dim=0, lr=LR, sgd=False):
    """A toy head plus the trainer's OWN optimiser chain and `_set_lr`.

    The chain is assembled exactly as `train_stage2.main` assembles it —
    `inject_hyperparams(adamw)`, and `clip_by_global_norm` prepended only when
    the clip is on — so this file is not testing a second optimiser.
    """
    model = jm.TemporalTransformer(**TOY, eps_dim=eps_dim, rngs=nnx.Rngs(seed))
    graphdef, state = nnx.split(model)
    if sgd:
        tx = optax.inject_hyperparams(optax.sgd)(learning_rate=lr)
    else:
        tx = optax.inject_hyperparams(optax.adamw)(learning_rate=lr,
                                                   weight_decay=1e-4)
    if clip > 0:
        tx = optax.chain(optax.clip_by_global_norm(clip), tx)
    return graphdef, state, tx, tx.init(state), make_set_lr(clip > 0)


def _batch(seed=1, b=BATCH, scale=1.0, stencil=1, d_z=TOY["d_z"]):
    r = np.random.default_rng(seed)
    z = r.standard_normal((b, K, stencil * d_z)).astype(np.float32)
    m = r.standard_normal((b, K, 2)).astype(np.float32)
    s = r.standard_normal((b, d_z + 2 + (stencil if stencil > 1 else 0))
                          ).astype(np.float32)
    t = (scale * r.standard_normal((b, K, d_z))).astype(np.float32)
    return z, m, s, t


def _paths(tree):
    return [(jax.tree_util.keystr(p), np.asarray(v))
            for p, v in jax.tree_util.tree_flatten_with_path(tree)[0]]


ADAM_EPS = 1e-8       # optax.adamw's default, and the smoothing that decides
                      # where a first Adam step stops being sign(g)
WANT = 1e-5           # the tolerance this file asserts on parameter deltas


def _wellcond(grads, dg, want=WANT):
    """The ENTRIES where a first Adam step can RESOLVE a gradient known only
    to ±`dg`, with the threshold DERIVED rather than chosen.

    At step 1 the bias-corrected update is `lr * g / (|g| + eps)` — sign(g),
    smoothed over |g| ~ eps = 1e-8 — so its sensitivity to a perturbation of
    the gradient is

        |d update / d g| = lr * eps / (|g| + eps)^2  ~  lr * eps / g^2 .

    Requiring that a perturbation `dg` move the update by less than
    `want * lr` therefore requires

        |g|  >  sqrt(eps * dg / want) ,

    which is what this returns, with `dg` the MEASURED whole-batch-vs-averaged
    gradient difference from the same test (float association over a different
    summation order — the only thing that separates the two modes). Nothing
    here is tuned: change the head, the batch or the tolerance and the
    threshold moves with them.

    Below it the update is not wrong, it is UNRESOLVED: it stays bounded by lr
    per coordinate (|sign(g)| <= 1), which is the unfiltered bound asserted
    beside every filtered comparison. And this is not a property of
    accumulation — an unaccumulated run that reduced the same batch in a
    different order, or on a different device, would wobble the same way.
    `tests/test_jaxport_train_s2.py`'s G5b gate treats the identical
    phenomenon the identical way (there the two orderings are torch's and
    JAX's rather than one batch's and two).

    The extreme case is analytically dead rather than merely small: a
    transformer's attention KEY BIAS shifts every logit of a query by the same
    constant and softmax is invariant to it, so `d loss / d k_proj.bias` is
    exactly zero and the backward returns pure cancellation noise (~1e-9 here,
    against 5.2 elsewhere in the same gradient).
    """
    thr = math.sqrt(ADAM_EPS * float(dg) / want) if dg > 0 else 0.0
    return {p: (np.abs(g) > thr) for p, g in _paths(grads)}, thr


def _delta_maxrel(after_a, before_a, after_b, before_b, mask=None):
    """max |da - db| / max |db| over the parameter deltas, and where.

    `mask` (path -> boolean array) restricts the comparison entry by entry;
    entries it drops are covered by the caller's absolute `lr` bound.
    """
    da = dict(_paths(jax.tree.map(lambda x, y: x - y, after_a, before_a)))
    db = dict(_paths(jax.tree.map(lambda x, y: x - y, after_b, before_b)))
    sel = {k: (np.ones_like(v, bool) if mask is None else mask[k])
           for k, v in da.items()}
    den = max(max((float(np.abs(db[k][sel[k]]).max()) if sel[k].any() else 0.0)
                  for k in da), 1e-30)
    worst, where = 0.0, ""
    for k in da:
        if not sel[k].any():
            continue
        d = float(np.abs(da[k][sel[k]] - db[k][sel[k]]).max()) / den
        if d > worst:
            worst, where = d, k
    return worst, where


def _delta_maxabs(after_a, before_a, after_b, before_b):
    """max |da - db|, unfiltered — held against the algebraic `lr` bound."""
    da = dict(_paths(jax.tree.map(lambda x, y: x - y, after_a, before_a)))
    db = dict(_paths(jax.tree.map(lambda x, y: x - y, after_b, before_b)))
    return max(float(np.abs(da[k] - db[k]).max()) for k in da)


def _param_maxrel(a, b, mask=None):
    """max |a - b| / max |b| over the PARAMETERS themselves.

    The delta comparison is bounded from below by ulp(|w|) — writing
    `w + update` back into fp32 quantises it — so this is the statement that
    does not depend on how big the step happened to be.
    """
    pa, pb = dict(_paths(a)), dict(_paths(b))
    keys = [k for k in pa if mask is None or mask[k].any()]
    den = max(max(float(np.abs(pb[k]).max()) for k in keys), 1e-30)
    return max(float(np.abs(pa[k] - pb[k]).max()) for k in keys) / den


def _ulp_floor(before):
    """The representation floor of a delta comparison: one ulp of the largest
    parameter, which `w + update -> fp32` cannot do better than."""
    big = max(float(np.abs(v).max()) for _, v in _paths(before))
    return float(np.spacing(np.float32(big)))


def _illcond_report(mask):
    """(entries dropped, entries total, the tensors they live in)."""
    n_bad = sum(int((~v).sum()) for v in mask.values())
    n_all = sum(int(v.size) for v in mask.values())
    return n_bad, n_all, sorted(p for p, v in mask.items() if not v.all())


def _tree_maxrel(a, b):
    """max |a - b| / max |b| over a whole pytree, unfiltered."""
    la = [v for _, v in _paths(a)]
    lb = [v for _, v in _paths(b)]
    den = max(max(float(np.abs(v).max()) for v in lb), 1e-30)
    return max(float(np.abs(x - y).max()) for x, y in zip(la, lb)) / den


def _step1(steps, st, ost, lr, z, m, s, t):
    return steps.train_step(st, ost, lr, jnp.asarray(z), jnp.asarray(m),
                            jnp.asarray(s), jnp.asarray(t))


# --------------------------------------------------------------------------
# 1 · exactness
# --------------------------------------------------------------------------
def _grad_of(graphdef, st, z, m, s, t):
    def lf(stt, z_, m_, s_, t_):
        mo = nnx.merge(graphdef, stt)
        pred, _ = mo(z_, m_, s_)
        return ((pred - t_) ** 2).mean()
    return jax.jit(jax.grad(lf))(st, jnp.asarray(z), jnp.asarray(m),
                                 jnp.asarray(s), jnp.asarray(t))


def _grad_pair(graphdef, st, z, m, s, t, micro, scale=1.0):
    """(whole-batch gradient, averaged micro-batch gradient, |difference|).

    `scale` is the clip factor when a clip is in the chain: the transform
    multiplies the gradient AND its association error by the same number
    before Adam sees them, and it is what Adam sees that decides which
    coordinates it can resolve.
    """
    g_full = _grad_of(graphdef, st, z, m, s, t)
    n = len(z) // micro
    gs = [_grad_of(graphdef, st, z[i * micro:(i + 1) * micro],
                   m[i * micro:(i + 1) * micro],
                   s[i * micro:(i + 1) * micro],
                   t[i * micro:(i + 1) * micro]) for i in range(n)]
    g_avg = jax.tree.map(lambda *xs: sum(xs) / float(n), *gs)
    if scale != 1.0:
        g_full = jax.tree.map(lambda x: x * scale, g_full)
        g_avg = jax.tree.map(lambda x: x * scale, g_avg)
    dg = max(float(np.abs(x - y).max())
             for (_, x), (_, y) in zip(_paths(g_avg), _paths(g_full)))
    return g_full, g_avg, dg


def test_exactness_one_step():
    """One optimiser step at accum 1 and accum 2 on identical data."""
    z, m, s, t = _batch(seed=3)
    lr = jnp.asarray(LR, jnp.float32)

    # (a) THE MATHEMATICAL CLAIM, with no optimiser in it: the average of the
    # two half-batch gradients IS the whole-batch gradient.
    gd, st, tx, ost, setlr = _build()
    g_full, g_avg, dg = _grad_pair(gd, st, z, m, s, t, 16)
    r_grad = _tree_maxrel(g_avg, g_full)
    _note(f"averaged micro-gradient vs whole-batch gradient: "
          f"max rel {r_grad:.3e}, max |Δ| {dg:.3e} (unfiltered, every "
          f"coordinate)")
    assert r_grad <= 1e-5, r_grad

    # (b) PLAIN SGD — the update is linear in the gradient, so this statement
    # needs no exclusions at all.
    gd1, st1, tx1, ost1, sl1 = _build(sgd=True)
    gd2, st2, tx2, ost2, sl2 = _build(sgd=True)
    s1 = build_train_steps(gd1, tx1, sl1, TOY["d_z"], 0.0, accum=1)
    s2 = build_train_steps(gd2, tx2, sl2, TOY["d_z"], 0.0, accum=2)
    a1, ao1, l1, gn1 = _step1(s1, st1, ost1, lr, z, m, s, t)
    a2, ao2, l2, gn2 = accum_step(s2, st2, ost2, lr, 1, z, m, s, t, 16)
    r_sgd, _ = _delta_maxrel(a1, st1, a2, st2)
    _note(f"SGD parameter deltas, accum 1 vs accum 2: max rel {r_sgd:.3e} "
          f"(unfiltered)")
    assert r_sgd <= 1e-5, r_sgd

    # (c) the trainer's OWN AdamW chain.
    gd1, st1, tx1, ost1, sl1 = _build()
    gd2, st2, tx2, ost2, sl2 = _build()
    s1 = build_train_steps(gd1, tx1, sl1, TOY["d_z"], 0.0, accum=1)
    s2 = build_train_steps(gd2, tx2, sl2, TOY["d_z"], 0.0, accum=2)
    a1, ao1, l1, gn1 = _step1(s1, st1, ost1, lr, z, m, s, t)
    a2, ao2, l2, gn2 = accum_step(s2, st2, ost2, lr, 1, z, m, s, t, 16)

    mask, thr = _wellcond(g_full, dg)
    n_bad, n_all, bad_tensors = _illcond_report(mask)
    r_adam, where = _delta_maxrel(a1, st1, a2, st2, mask=mask)
    d_abs = _delta_maxabs(a1, st1, a2, st2)
    r_par = _param_maxrel(a1, a2, mask=mask)
    floor = _ulp_floor(st1) / max(LR, 1e-30)
    _note(f"AdamW parameter deltas, accum 1 vs accum 2: max rel "
          f"{r_adam:.3e} over the {n_all - n_bad}/{n_all} entries where "
          f"|g| > sqrt(eps·Δg/1e-5) = {thr:.3e} (worst at {where}), against "
          f"a pure-representation floor of {floor:.3e}; the PARAMETERS "
          f"themselves agree to {r_par:.3e}; over EVERY entry the absolute "
          f"delta difference is {d_abs:.3e}, against the algebraic bound "
          f"2 lr = {2 * LR:g}")
    assert r_adam <= 1e-5, r_adam
    assert r_par <= 1e-5, r_par
    # The unfiltered statement, with no tolerance chosen by anybody: a first
    # Adam step is at most lr per coordinate (|sign(g)| <= 1), so two of them
    # differ by at most 2 lr — anything above that is a real disagreement and
    # not sign sensitivity.
    assert d_abs <= 2 * LR, d_abs
    assert n_bad, "no entry was ill-conditioned — either the head changed " \
                  "or the detector is broken"
    assert any("k_proj" in p and "bias" in p for p in bad_tensors), \
        f"the analytically dead attention key bias is no longer among the " \
        f"ill-conditioned tensors: {bad_tensors}"
    _note(f"  ({n_bad} entries dropped, in {len(bad_tensors)} tensors; the "
          f"attention key biases — analytically zero gradient — are among "
          f"them)")

    # the two reported numbers, which are what a reader of metrics.jsonl sees
    rl = abs(float(l1) - float(l2)) / max(abs(float(l1)), 1e-30)
    rg = abs(float(gn1) - float(gn2)) / max(abs(float(gn1)), 1e-30)
    _note(f"reported metrics: stage2_zmse rel {rl:.3e} "
          f"({float(l1):.7f} vs {float(l2):.7f}), stage2_grad_norm rel "
          f"{rg:.3e} ({float(gn1):.6f} vs {float(gn2):.6f})")
    assert rl <= 1e-5 and rg <= 1e-5


def test_exactness_three_steps():
    """Three consecutive optimiser steps stay inside the same tolerance —
    a per-step 1e-7 that compounded would be a different finding."""
    lr = jnp.asarray(LR, jnp.float32)
    gd1, st1, tx1, ost1, sl1 = _build()
    gd2, st2, tx2, ost2, sl2 = _build()
    s1 = build_train_steps(gd1, tx1, sl1, TOY["d_z"], 0.0, accum=1)
    s2 = build_train_steps(gd2, tx2, sl2, TOY["d_z"], 0.0, accum=2)
    worst, mask = 0.0, None
    for step in (1, 2, 3):
        z, m, s, t = _batch(seed=10 + step)
        # the conditioning mask is re-derived AT THIS STEP's state and batch —
        # which entries Adam can resolve is a property of this gradient, not a
        # list carried over from step 1.
        g_full, _, dg = _grad_pair(gd1, st1, z, m, s, t, 16)
        mask, _thr = _wellcond(g_full, dg)
        p1, p2 = st1, st2
        st1, ost1, l1, gn1 = _step1(s1, st1, ost1, lr, z, m, s, t)
        st2, ost2, l2, gn2 = accum_step(s2, st2, ost2, lr, step,
                                        z, m, s, t, 16)
        r, _ = _delta_maxrel(st1, p1, st2, p2, mask=mask)
        d_abs = _delta_maxabs(st1, p1, st2, p2)
        worst = max(worst, r)
        _note(f"step {step}: delta max rel {r:.3e} (|Δ| over every entry "
              f"{d_abs:.3e} <= 2 lr) · zmse {float(l1):.6f} vs "
              f"{float(l2):.6f} · grad_norm {float(gn1):.5f} vs "
              f"{float(gn2):.5f}")
        assert r <= 1e-5, (step, r)
        assert d_abs <= 2 * LR, (step, d_abs)
    # and the PARAMETERS themselves, not only the per-step deltas, are still
    # together after three steps — over the entries Adam can resolve, and by
    # the accumulated `6 * lr` bound (2 lr a step) over the rest.
    par = [(np.abs(x - y), np.abs(y), mask[p])
           for (p, x), (_, y) in zip(_paths(st1), _paths(st2))]
    den = max(max(float(b[k].max()) if k.any() else 0.0 for _, b, k in par),
              1e-30)
    r_par = max(float(d[k].max()) / den if k.any() else 0.0
                for d, _, k in par)
    a_par = max(float(d.max()) for d, _, _ in par)
    _note(f"after 3 steps: parameters agree to max rel {r_par:.3e} over the "
          f"well-conditioned entries (worst-step delta {worst:.3e}); max |Δ| "
          f"over every entry {a_par:.3e} <= 6 lr = {6 * LR:g}")
    assert r_par <= 1e-5, r_par
    assert a_par <= 6 * LR, a_par


# --------------------------------------------------------------------------
# 2 · flag-off purity
# --------------------------------------------------------------------------
def _pristine_step(graphdef, tx, set_lr):
    """THE PRE-E-054b STEP, re-transcribed here in plain jnp — deliberately
    not calling `stage2_loss` or `build_train_steps`, so this is an
    independent statement of what the trainer used to compute rather than a
    rewording of what it now computes (the same device
    `tests/test_fgn_jax.py:_pristine_forward` uses)."""
    def loss_fn(st, zseq, mseq, sctx, ztgt):
        mo = nnx.merge(graphdef, st)
        pred, _ = mo(zseq, mseq, sctx)
        return ((pred - ztgt) ** 2).mean()

    @jax.jit
    def step(st, ost, lr, zseq, mseq, sctx, ztgt):
        zseq = zseq.astype(jnp.float32)
        loss, grads = jax.value_and_grad(loss_fn)(st, zseq, mseq, sctx, ztgt)
        gnorm = optax.global_norm(grads)
        ost = set_lr(ost, lr)
        upd, ost = tx.update(grads, ost, st)
        return optax.apply_updates(st, upd), ost, loss, gnorm
    return step


def _eqn_count(jaxpr):
    """Operations in a jaxpr, descending into every nested one (a jitted
    function traces to a single `pjit` equation whose body is the real
    graph)."""
    n = 0
    for e in jaxpr.eqns:
        n += 1
        for v in e.params.values():
            sub = getattr(v, "jaxpr", v)          # ClosedJaxpr -> Jaxpr
            if hasattr(sub, "eqns"):
                n += _eqn_count(sub)
    return n


def test_accum1_is_the_pre_diff_path():
    """--grad-accum 1 is not a branch, it is an absent graph."""
    # the default is off, and it is off when the flag is absent from argv.
    a = parse(["--data", "d.npz", "--ckpt", "c.pt", "--out", "/tmp/x"])
    assert a.grad_accum == 1
    assert grad_accum_micro(a.batch, a.grad_accum) == a.batch

    gd, st, tx, ost, setlr = _build()
    steps = build_train_steps(gd, tx, setlr, TOY["d_z"], 0.0, accum=1)
    for name in ("zero_grads", "micro", "micro_dn", "micro_fgn",
                 "micro_fgn_dn", "apply_accum"):
        assert getattr(steps, name) is None, \
            f"accum 1 built {name}: the flag-off path is no longer the " \
            f"absence of the accumulation graph"
    assert steps.accum == 1

    z, m, s, t = _batch(seed=5)
    lr = jnp.asarray(LR, jnp.float32)
    args = (jnp.asarray(z), jnp.asarray(m), jnp.asarray(s), jnp.asarray(t))
    new, _, loss, gnorm = steps.train_step(st, ost, lr, *args)

    gd2, st2, tx2, ost2, setlr2 = _build()
    ref = _pristine_step(gd2, tx2, setlr2)
    rnew, _, rloss, rgnorm = ref(st2, ost2, lr, *args)

    for (p, x), (_, y) in zip(_paths(new), _paths(rnew)):
        assert np.array_equal(x, y), \
            f"accum 1 is no longer BITWISE the pre-diff step at {p}: " \
            f"max|d| {np.abs(x - y).max()}"
    assert float(loss) == float(rloss) and float(gnorm) == float(rgnorm)

    n_new = _eqn_count(jax.make_jaxpr(steps.train_step)(st, ost, lr, *args)
                       .jaxpr)
    n_ref = _eqn_count(jax.make_jaxpr(ref)(st2, ost2, lr, *args).jaxpr)
    _note(f"flag-off purity: parameters BITWISE equal to the pristine step "
          f"(max|d| 0.0), traced graph {n_new} operations vs the pristine "
          f"{n_ref}")
    assert n_new == n_ref, (n_new, n_ref)


# --------------------------------------------------------------------------
# 3 · refusals
# --------------------------------------------------------------------------
def test_refusals():
    """A split that cannot be represented is refused before anything costs."""
    seen = []
    for batch, accum, must_say in ((32, 5, "not divisible"),
                                   (256, 7, "not divisible"),
                                   (32, 0, "must be >= 1"),
                                   (32, -2, "must be >= 1")):
        try:
            grad_accum_micro(batch, accum)
        except SystemExit as e:
            assert must_say in str(e), f"batch {batch} accum {accum}: {e}"
            seen.append(f"batch {batch} / accum {accum}")
        else:
            raise AssertionError(
                f"--batch {batch} --grad-accum {accum} was ACCEPTED")
    # and the representable ones are not refused
    assert grad_accum_micro(256, 1) == 256
    assert grad_accum_micro(256, 2) == 128
    assert grad_accum_micro(256, 4) == 64
    _note("refused: " + " · ".join(seen)
          + " · accepted 256/1 -> 256, 256/2 -> 128, 256/4 -> 64")


# --------------------------------------------------------------------------
# 4 · FGN + accumulation
# --------------------------------------------------------------------------
def test_fgn_accum_runs_and_eps_is_per_micro():
    """A crps2 step at accum 2, and the ε stream that keeps its members
    independent across micro-batches."""
    root = fgn_train_key(0)
    e_m0f0 = np.asarray(fgn_eps_at(root, 7, 0, 16, 8, micro_index=0))
    e_m1f0 = np.asarray(fgn_eps_at(root, 7, 0, 16, 8, micro_index=1))
    e_m0f1 = np.asarray(fgn_eps_at(root, 7, 1, 16, 8, micro_index=0))
    assert not np.array_equal(e_m0f0, e_m1f0), \
        "micro 0 and micro 1 drew the SAME eps at forward 0 — the two " \
        "micro-batches would be scoring correlated members"
    assert not np.array_equal(e_m0f0, e_m0f1)
    assert not np.array_equal(e_m1f0, np.asarray(
        fgn_eps_at(root, 8, 0, 16, 8, micro_index=1))), "steps must differ"
    # the LEGACY fold is untouched: micro_index=None is not micro 0.
    legacy = np.asarray(fgn_eps_at(root, 7, 0, 16, 8))
    assert np.array_equal(legacy, np.asarray(
        fgn_eps_at(root, 7, 0, 16, 8, micro_index=None))), \
        "the un-accumulated eps stream moved"
    assert not np.array_equal(legacy, e_m0f0), \
        "fold_in(k, 0) must not be confused with k — an FGN resume at " \
        "--grad-accum 1 would draw a different stream"

    # and the accumulated crps2 step itself runs and moves the head.
    lr = jnp.asarray(LR, jnp.float32)
    z, m, s, t = _batch(seed=9)
    gd, st, tx, ost, setlr = _build(eps_dim=8)
    steps = build_train_steps(gd, tx, setlr, TOY["d_z"], 0.0, accum=2)
    new, _, loss, gnorm = accum_step(steps, st, ost, lr, 7, z, m, s, t, 16,
                                     fgn_root=root, fgn_eps=8)
    assert np.isfinite(float(loss)) and np.isfinite(float(gnorm))
    moved = max(float(np.abs(x - y).max())
                for (_, x), (_, y) in zip(_paths(new), _paths(st)))
    assert moved > 0, "the accumulated fgn step did not move the head"
    _note(f"fgn accum 2: crps2 {float(loss):.6f}, grad_norm "
          f"{float(gnorm):.5f}, max|dw| {moved:.3e}; eps(micro 0, fwd 0) vs "
          f"(micro 1, fwd 0) max|d| "
          f"{float(np.abs(e_m0f0 - e_m1f0).max()):.4f}")


# --------------------------------------------------------------------------
# 5 · --grad-clip on the averaged norm
# --------------------------------------------------------------------------
def test_grad_clip_applies_to_the_averaged_norm():
    """An artificially huge gradient, clipped: the same clip, the same step."""
    lr = jnp.asarray(LR, jnp.float32)
    z, m, s, t = _batch(seed=4, scale=50.0)     # targets 50x -> a big gradient
    # THE THRESHOLD IS SET FROM THE GRADIENT, not guessed: half the norm this
    # batch actually produces, so the clip is guaranteed to bind (at exactly
    # 2x) rather than being either inert or so violent that every coordinate
    # lands in Adam's eps-smoothed regime and the comparison measures nothing.
    gd0, st0, tx0, ost0, sl0 = _build(clip=0.0)
    s0 = build_train_steps(gd0, tx0, sl0, TOY["d_z"], 0.0, accum=1)
    _, _, _, gn0 = _step1(s0, st0, ost0, lr, z, m, s, t)
    CLIP = float(gn0) / 2.0
    CFAC = CLIP / float(gn0)                     # what the clip multiplies by
    gd1, st1, tx1, ost1, sl1 = _build(clip=CLIP)
    gd2, st2, tx2, ost2, sl2 = _build(clip=CLIP)
    s1 = build_train_steps(gd1, tx1, sl1, TOY["d_z"], 0.0, accum=1)
    s2 = build_train_steps(gd2, tx2, sl2, TOY["d_z"], 0.0, accum=2)
    a1, _, l1, gn1 = _step1(s1, st1, ost1, lr, z, m, s, t)
    a2, _, l2, gn2 = accum_step(s2, st2, ost2, lr, 1, z, m, s, t, 16)
    assert float(gn1) > CLIP and float(gn2) > CLIP, \
        f"the clip did not bind ({float(gn1)} vs {CLIP}) — this test would " \
        f"then be measuring the unclipped path"
    g_full, _, dg = _grad_pair(gd1, st1, z, m, s, t, 16, scale=CFAC)
    mask, _thr = _wellcond(g_full, dg)
    r, where = _delta_maxrel(a1, st1, a2, st2, mask=mask)
    d_abs = _delta_maxabs(a1, st1, a2, st2)
    rg = abs(float(gn1) - float(gn2)) / float(gn1)
    _note(f"clip {CLIP:.3f} binding on a pre-clip norm of {float(gn1):.1f} "
          f"(x{CFAC:.3f}): deltas agree to max rel {r:.3e} (worst {where}, "
          f"max |Δ| over every entry {d_abs:.3e} <= 2 lr), reported pre-clip "
          f"norms to {rg:.3e}")
    assert r <= 1e-5, r
    assert d_abs <= 2 * LR, d_abs
    assert rg <= 1e-5, rg

    # THE CLIP MUST ACTUALLY BE CHANGING THE STEP, or "identical in both modes"
    # is a statement about an inactive transform. Under Adam it barely does —
    # a first Adam step is ~sign(g), which is INVARIANT to scaling the whole
    # gradient, so halving the gradient moves the weights by ~1e-4 relative and
    # a liveness check there would be measuring Adam's eps. Under plain SGD the
    # clip's effect is exactly the factor it applies, so that is where the
    # liveness AND the "it is the AVERAGED norm being clipped" claims are made:
    # at accum 2 the update must be CFAC x the unclipped one, where CFAC was
    # computed from the norm of the whole-batch gradient.
    gA, sA, tA, oA, lA = _build(sgd=True, clip=CLIP)
    gB, sB, tB, oB, lB = _build(sgd=True, clip=CLIP)
    gC, sC, tC, oC, lC = _build(sgd=True)
    stepsA = build_train_steps(gA, tA, lA, TOY["d_z"], 0.0, accum=1)
    stepsB = build_train_steps(gB, tB, lB, TOY["d_z"], 0.0, accum=2)
    stepsC = build_train_steps(gC, tC, lC, TOY["d_z"], 0.0, accum=1)
    aA, _, _, _ = _step1(stepsA, sA, oA, lr, z, m, s, t)
    aB, _, _, _ = accum_step(stepsB, sB, oB, lr, 1, z, m, s, t, 16)
    aC, _, _, _ = _step1(stepsC, sC, oC, lr, z, m, s, t)
    r_sgd, _ = _delta_maxrel(aA, sA, aB, sB)
    dA = max(float(np.abs(v).max())
             for _, v in _paths(jax.tree.map(lambda x, y: x - y, aA, sA)))
    dB = max(float(np.abs(v).max())
             for _, v in _paths(jax.tree.map(lambda x, y: x - y, aB, sB)))
    dC = max(float(np.abs(v).max())
             for _, v in _paths(jax.tree.map(lambda x, y: x - y, aC, sC)))
    _note(f"SGD, the same clip: accum 1 vs accum 2 deltas agree to max rel "
          f"{r_sgd:.3e}; the clip scales the step by {dA / dC:.6f} (N=1) and "
          f"{dB / dC:.6f} (N=2) against the CLIP/||g|| = {CFAC:.6f} computed "
          f"from the whole-batch norm")
    assert r_sgd <= 1e-5, r_sgd
    for got in (dA / dC, dB / dC):
        assert abs(got - CFAC) / CFAC <= 1e-4, (got, CFAC)


# --------------------------------------------------------------------------
if __name__ == "__main__":
    for fn in (test_exactness_one_step, test_exactness_three_steps,
               test_accum1_is_the_pre_diff_path, test_refusals,
               test_fgn_accum_runs_and_eps_is_per_micro,
               test_grad_clip_applies_to_the_averaged_norm):
        print(f"\n== {fn.__name__}", flush=True)
        fn()
    print("\nall E-054b gradient-accumulation checks passed")

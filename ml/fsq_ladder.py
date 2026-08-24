#!/usr/bin/env python3
"""THE FSQ LADDER: where the levels of a quantized dimension are PLACED, and
how a run chooses that placement per dimension.

E-048 (Chris, 2026-08-24): *"For each channel try to compute a distribution
such that the FSQ levels can be on the scalar ladder (uniform: level*c) or on
the exponential ladder (c**level)."*

TWO THINGS THIS MODULE IS, AND ONE IT IS NOT.

It IS the arithmetic that both backends and the fit share: the level counts, the
two ladders' parameters, a numpy quantizer used by the fit and by the tests, the
per-dimension fit itself, and the string a checkpoint records the fitted ladder
in. It is numpy-only and imports neither torch nor jax, which is what lets
`ml/model.py` (torch) and `ml/jaxport/models.py` (flax) both read it rather than
each carrying its own copy — this repo has paid the two-copies bill twice (the
anomaly transform's four copies, `month_index`'s 6.09:1 collapse), and a
quantizer that differed between backends by one boundary would produce a TPU
codec the torch eval ladder scores as a different model.

It is NOT the training-time transform. Each backend applies the ladder in its
own framework so the straight-through gradient is native (torch autograd,
jax.grad); what they take from here is the LADDER, not the loop. The parity
gate in tests/test_e048_fsq_ladders.py is what makes that split honest: torch,
jax and the numpy reference here must agree to 1e-5 on the same inputs, for
both ladders.

"FOR EACH CHANNEL" IS PER Z-DIMENSION HERE, AND THE DIFFERENCE MATTERS.
A codec channel is an input field (`thetao_0`, `rg_s_1000`, …); a z-dimension is
one coordinate of the bottleneck, and the two are not in correspondence — 40
channels enter and d_z coordinates leave, mixed by an encoder. The quantizer
lives on the bottleneck, so the only per-something choice it can honestly make
is PER Z-DIMENSION, and that is what `fit_auto` returns. Reading Chris's
"channel" as "z-dimension" is the mapping that exists; a per-input-channel
ladder would be a different object entirely (it would have to sit before the
encoder, on the input tokens, which is `--input-quant`'s side of the system).

---

THE TWO LADDERS, EXACTLY.

Both are defined per dimension d, on the SAME raw pre-quantization activation
`v` (`to_z`'s output on the codec side; the head's incoming z on the
`--input-quant` side), both saturate at the same radius `R_d = 2*sigma_d`, and
both offer exactly L levels — so a run that changes only the ladder changes
only WHERE the levels sit.

  uniform (the default, and every archived checkpoint's behaviour)
      half = (L-1)/2 ;  offset = 0.5 if L even else 0 ;  shift = atanh(off/half)
      g    = half * tanh(v/R + shift) - offset
      q    = g + (round(g) - g).detach()
      v_q  = (q + offset) * R / half
    The reconstruction points are EVENLY SPACED in z-space, R/half apart,
    outermost at +/-R. The tanh is the BOUND, and it also compands the
    ANALYSIS side: dg/dv is largest at v = 0, so the bins are narrowest near
    zero and widen into the tails.

  exp (E-048)
      n    = L // 2 levels per side ;  zero level iff L is odd
      a    = R * c^-(n-1)      the innermost magnitude
      j    = round(log_c(|v| / a))            nearest level IN LOG SPACE
      v_q  = sign(v) * a * c^clamp(j, jmin, n-1)      jmin = -1 if L odd else 0
             (and j < 0 with an odd L is the ZERO level)
    The reconstruction points are GEOMETRICALLY SPACED — magnitudes
    a, a*c, ..., a*c^(n-1) = R — and the bins are uniform in log|v|, i.e. each
    level owns a constant RATIO band (|v| within a factor sqrt(c) of it).

WHY exp REPLACES THE TANH BOUND INSTEAD OF COMPOSING WITH IT. The tanh already
compands, so an exponential ladder ON TOP of it would compand twice and the
result would be a ladder nobody can state in z-units: the level positions would
be `(round(half*tanh(...)) + off) * R/half`, which is the UNIFORM lattice
whatever the ladder says, because the de-scale is linear by design (inverting
the tanh would send the outermost levels to infinity — see InputQuant). In
other words, "exponential levels after a tanh bound" is not expressible in this
parameterisation at all: the tanh can only move the BOUNDARIES, never the
POINTS. So `exp` is defined on the pre-tanh activation and carries its own
bound — the clamp at j = n-1, which saturates at exactly the same +/-R the tanh
does. One statement per ladder, both in z-units, both bounded identically.

THE STRAIGHT-THROUGH GRADIENT IS THE IDENTITY IN THE INTERIOR, and that is an
identity rather than a convention: `c^log_c(|v|/a) = |v|/a`, so with the round
detached the exp map is exactly `v` and d v_q/d v = 1. In saturation it decays
as R/|v| (the uniform ladder's tanh decays faster, as sech²), and on the zero
level of an odd L it is the plain straight-through `v + (0 - v).detach()`.

DEGENERATE CONFIGURATIONS ARE REFUSED, not ranked improbable (ml/CLAUDE.md
§4.9b): base <= 1 (c = 1 makes every level identical, c < 1 inverts the
ladder), and a base so large that the innermost level is under 1e-6 of the
saturation radius — at that point the ladder has a zero level it does not admit
to having. L = 2 is refused for both ladders, by `parse_levels`, exactly as
before.

---

THE `auto` FIT. Per dimension, on a MEASURED sample of the pre-quantization
activation:

  * compute the quantization MSE of the uniform ladder;
  * compute it for the exp ladder at each base in `AUTO_BASES`;
  * take the best, and keep uniform on a TIE or on any non-improvement, so
    `auto` can only differ from `uniform` where it measured a strict gain.

The chosen ladder and base are recorded per dimension in the checkpoint
(`fsq_ladder_fit`, the string `format_fit` writes) and rebuilt by every loader.
A LOADER NEVER RE-FITS: the fit is part of what the codec IS, in exactly the
sense `fsq_levels` is (ml/model.py:codec_from_ckpt), and re-fitting at eval
time would score a different model from the one that trained while every
state_dict tensor loaded cleanly.
"""
import numpy as np

LADDERS = ("uniform", "exp", "auto")
DEFAULT_EXP_BASE = 2.0
# The grid `auto` searches. Four bases, geometrically spread: 1.5 is a gentle
# ladder (L=8 spans a factor 3.4 from innermost to outermost), 4.0 a severe one
# (a factor 64). Not a free-form optimisation — a fitted continuous base would
# be a per-dimension parameter nobody could sanity-check, and the MSE surface
# over c is shallow between these rungs.
AUTO_BASES = (1.5, 2.0, 3.0, 4.0)
# The innermost level of an exp ladder is R * c^-(n-1). Below this fraction of
# R it is not a level, it is zero with extra steps.
MIN_LEVEL_FRAC = 1e-6


def parse_levels(spec, d_z, flag="--fsq-levels"):
    """`"8"` / `"8,8,8,6,5"` -> an int array of length d_z, or refuse.

    Moved here from `ml/model.py:InputQuant` so the JAX side reads the same
    parser rather than a second one; the messages are unchanged, because a
    dispatch that hits one of these has already been written against them.
    """
    lv = [int(x) for x in str(spec).split(",") if str(x).strip()]
    if not lv:
        raise SystemExit(f"{flag} {spec!r}: no levels")
    if any(l < 3 for l in lv):
        # L=2 is degenerate in this parameterisation and not merely small:
        # half = 0.5 and offset = 0.5 give shift = atanh(1) = inf, so every
        # input collapses to one value. The FSQ paper's own level sets start
        # at 3 for the same reason; refuse rather than train a head whose
        # input is a constant. (An exp ladder at L=2 is the twin degeneracy:
        # one magnitude per side and no interior at all.)
        raise SystemExit(f"{flag} {spec!r}: every level count must "
                         f"be >= 3 (L=2 collapses to a single value in "
                         f"the bounded-tanh parameterisation)")
    if len(lv) == 1:
        lv = lv * int(d_z)
    if len(lv) != int(d_z):
        raise SystemExit(
            f"{flag} {spec!r} gives {len(lv)} level counts for "
            f"d_z {d_z}: pass one number (applied to every dimension) or "
            f"exactly d_z of them")
    return np.asarray(lv, np.int64)


# --------------------------------------------------------------------------
# ladder parameters
# --------------------------------------------------------------------------
def uniform_params(levels):
    """(half, offset, shift) — the bound's constants, per dimension."""
    lv = np.asarray(levels, np.float64)
    half = (lv - 1.0) / 2.0
    offset = np.where(lv % 2 == 0, 0.5, 0.0)
    shift = np.arctanh(offset / half)
    return half, offset, shift


def exp_params(levels, bases, flag="--fsq-ladder exp"):
    """(n_side, a_rel, log_base, has_zero) — the geometric ladder's constants.

    `a_rel` is the innermost magnitude AS A FRACTION of the saturation radius
    R, so a caller multiplies by its own R (2*sigma) and needs to know nothing
    else. `has_zero` is L odd.
    """
    lv = np.asarray(levels, np.int64)
    c = np.asarray(bases, np.float64) * np.ones(lv.shape, np.float64)
    bad = np.where(c <= 1.0 + 1e-12)[0]
    if len(bad):
        raise SystemExit(
            f"{flag}: base {float(c[bad[0]])!r} at dimension {int(bad[0])} — "
            f"the exponential ladder stacks levels at a*c^j, so a base of 1 "
            f"makes every level identical and a base below 1 inverts the "
            f"ladder. Pass a base > 1.")
    n = np.maximum(lv // 2, 1).astype(np.int64)            # levels per side
    span = c ** (n - 1)
    over = np.where(span > 1.0 / MIN_LEVEL_FRAC)[0]
    if len(over):
        i = int(over[0])
        raise SystemExit(
            f"{flag}: base {float(c[i])} with L={int(lv[i])} at dimension {i} "
            f"puts the innermost level at c^-(n-1) = {1.0 / float(span[i]):.3g} "
            f"of the saturation radius, below the {MIN_LEVEL_FRAC:g} floor. "
            f"That is not a level, it is a zero level this ladder does not "
            f"admit to having — refusing rather than quantizing a dimension "
            f"onto a lattice whose interior is unreachable.")
    return n, 1.0 / span, np.log(c), (lv % 2 == 1)


def resolve(ladder, levels, d_z, exp_base=DEFAULT_EXP_BASE, fit="",
            flag="--fsq-ladder"):
    """(is_exp[d_z] bool, base[d_z] float, fitted bool) for a run's settings.

    `uniform` / `exp` are uniform everywhere / exp everywhere. `auto` is the
    RECORDED fit when there is one and uniform until then — an auto run
    quantizes uniformly for the steps before its fit, which is stated in the
    log and written into the checkpoint rather than left to be inferred.
    """
    lad = str(ladder or "uniform")
    if lad not in LADDERS:
        raise SystemExit(f"{flag} {lad!r}: expected one of {list(LADDERS)}")
    d_z = int(d_z)
    if lad == "auto" and str(fit or "").strip():
        is_exp, base = parse_fit(fit, d_z, flag=flag)
        return is_exp, base, True
    if lad == "exp":
        return (np.ones(d_z, bool),
                np.full(d_z, float(exp_base), np.float64), True)
    # uniform, or auto-before-its-fit
    return (np.zeros(d_z, bool),
            np.full(d_z, float(exp_base), np.float64), lad != "auto")


# --------------------------------------------------------------------------
# the numpy quantizer — the fit's instrument, and the parity reference
# --------------------------------------------------------------------------
def quantize_np(v, levels, scale, is_exp=None, base=DEFAULT_EXP_BASE):
    """[N, d_z] -> [N, d_z], the same map both backends implement.

    No gradient here: this is what the FIT measures with and what the parity
    gate compares against, so it is the plain forward map.
    """
    v = np.asarray(v, np.float64)
    lv = np.asarray(levels, np.int64)
    R = np.asarray(scale, np.float64) * np.ones(lv.shape)
    d_z = len(lv)
    assert v.shape[-1] == d_z, (v.shape, d_z)
    is_exp = (np.zeros(d_z, bool) if is_exp is None
              else np.asarray(is_exp, bool) * np.ones(d_z, bool))
    half, off, shift = uniform_params(lv)
    g = half * np.tanh(v / R + shift) - off
    out = (np.round(g) + off) * R / half
    if is_exp.any():
        n, a_rel, logc, has_zero = exp_params(lv, base)
        a = a_rel * R
        m = np.maximum(np.abs(v), 1e-30)
        j = np.round(np.log(m / a) / logc)
        jmin = np.where(has_zero, -1.0, 0.0)
        j = np.clip(j, jmin, (n - 1).astype(np.float64))
        s = np.where(v < 0, -1.0, 1.0)
        e = np.where(j < -0.5, 0.0, s * a * np.exp(j * logc))
        out = np.where(is_exp[None, :], e, out)
    return out


def levels_of(L, R, ladder="uniform", base=DEFAULT_EXP_BASE):
    """The RECONSTRUCTION POINTS of one dimension, ascending — L of them.

    Used by the tests to say what a ladder IS rather than to trust that the
    quantizer produced it, and by anything that wants to print the lattice.
    """
    L = int(L)
    if ladder == "uniform":
        half = (L - 1) / 2.0
        off = 0.5 if L % 2 == 0 else 0.0
        ks = np.arange(-half - off, half - off + 0.5, 1.0)[:L]
        return np.sort((ks + off) * R / half)
    if ladder != "exp":
        raise ValueError(f"levels_of: unknown ladder {ladder!r}")
    n = max(L // 2, 1)
    mag = R * float(base) ** (np.arange(n) - (n - 1))
    pts = np.concatenate([-mag[::-1], ([0.0] if L % 2 else []), mag])
    return np.sort(pts)


# --------------------------------------------------------------------------
# the fit
# --------------------------------------------------------------------------
def fit_auto(sample, levels, scale, bases=AUTO_BASES):
    """Per-dimension ladder choice on a measured sample.

    Returns (is_exp[d_z], base[d_z], mse_uniform[d_z], mse_best[d_z]).

    Ties and non-improvements go to UNIFORM, so `auto` differs from the
    default only where it measured a strict gain — which is also what makes
    the fit's effect readable in the log as "N of d_z dimensions moved".
    """
    v = np.asarray(sample, np.float64)
    lv = np.asarray(levels, np.int64)
    d_z = len(lv)
    assert v.ndim == 2 and v.shape[1] == d_z, (v.shape, d_z)
    if len(v) < 2:
        raise SystemExit(
            f"--fsq-ladder auto: the fit sample holds {len(v)} vector(s). A "
            f"per-dimension distribution cannot be measured from that; raise "
            f"--fsq-auto-n or fit later in the run.")
    R = np.asarray(scale, np.float64) * np.ones(d_z)
    mse_u = ((quantize_np(v, lv, R) - v) ** 2).mean(0)
    best_mse = mse_u.copy()
    best_base = np.full(d_z, float(DEFAULT_EXP_BASE))
    is_exp = np.zeros(d_z, bool)
    for c in bases:
        q = quantize_np(v, lv, R, is_exp=np.ones(d_z, bool), base=c)
        mse_c = ((q - v) ** 2).mean(0)
        take = mse_c < best_mse
        best_mse = np.where(take, mse_c, best_mse)
        best_base = np.where(take, float(c), best_base)
        is_exp = is_exp | take
    return is_exp, best_base, mse_u, best_mse


def format_fit(is_exp, bases):
    """The per-dimension fit as ONE string a checkpoint can carry.

    `u` for uniform, `e<base>` for exp, comma-joined, one field per dimension:
    `u,e2,e1.5,u,...`. A string rather than an array because it rides in
    `ck["args"]`, which every loader in this programme already reads and which
    `vars(a)` writes as plain JSON-ish values.
    """
    out = []
    for e, c in zip(np.asarray(is_exp, bool), np.asarray(bases, float)):
        out.append(f"e{c:g}" if e else "u")
    return ",".join(out)


def parse_fit(spec, d_z, flag="--fsq-ladder"):
    """The inverse of `format_fit`, refusing anything it does not understand."""
    parts = [p.strip() for p in str(spec).split(",") if p.strip()]
    if len(parts) != int(d_z):
        raise SystemExit(
            f"{flag}: the recorded ladder fit has {len(parts)} entries for "
            f"d_z {d_z}. This checkpoint's fit does not describe this "
            f"codec — refusing rather than rebuilding a bottleneck that is "
            f"not the one that trained.")
    is_exp = np.zeros(int(d_z), bool)
    base = np.full(int(d_z), float(DEFAULT_EXP_BASE))
    for i, p in enumerate(parts):
        if p == "u":
            continue
        if not p.startswith("e"):
            raise SystemExit(
                f"{flag}: ladder fit entry {i} is {p!r}; expected 'u' or "
                f"'e<base>'.")
        try:
            base[i] = float(p[1:])
        except ValueError:
            raise SystemExit(
                f"{flag}: ladder fit entry {i} is {p!r}; 'e' must be followed "
                f"by the base, e.g. 'e2'.")
        is_exp[i] = True
    return is_exp, base


def describe_fit(is_exp, bases, mse_u, mse_b):
    """One line saying what the fit MEASURED, not what it attempted."""
    is_exp = np.asarray(is_exp, bool)
    n_e = int(is_exp.sum())
    tot_u, tot_b = float(np.mean(mse_u)), float(np.mean(mse_b))
    gain = 100.0 * (1.0 - tot_b / max(tot_u, 1e-30))
    if n_e:
        bs = np.asarray(bases, float)[is_exp]
        vals, cnt = np.unique(bs, return_counts=True)
        which = " ".join(f"c={v:g}x{c}" for v, c in zip(vals, cnt))
    else:
        which = "none"
    return (f"{n_e}/{len(is_exp)} dimensions on the EXPONENTIAL ladder "
            f"({which}), the rest uniform · mean quantization MSE "
            f"{tot_u:.6g} -> {tot_b:.6g} ({gain:+.2f}%)")

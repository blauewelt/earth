#!/usr/bin/env python3
"""E-006's gradient, before E-006's code.

CLAUDE.md 6c rule 5: *"For a change to an objective, write the algebra before
the code. The detached-denominator bug was a two-line sympy check away, and
that check is what finally settled it. For a loss, the gradient IS the
specification; the code is a transcription."*

Four normalisations were retracted between 2026-08-09 and 2026-08-10, every
one of them because the forecast term was scored in `z` — a space the encoder
authors and can therefore rescale. This file states the claim E-006 rests on
and checks it symbolically, so the design is settled before a GPU is asked to
evaluate it.

The claim, in one line:

    scoring the forecast in DATA space makes the encoder's scale a GAUGE
    (no gradient either way) rather than a DESCENT DIRECTION (free loss).

That distinction is the whole design. A degeneracy is not closed by policing
it; it is closed by there being nothing to police.

    python3 tests/test_e006_algebra.py
"""
import sympy as sp

# s  — the encoder's free output scale. This is the knob the model discovered
#      twice: z_shrink went to 1/40 in one run and x250 in another.
# a  — the encoder's predicted direction (what it got RIGHT or wrong)
# b  — the target, in whichever space the loss scores.
# c  — a denominator the model cannot move (a constant, or a data variance)
s, a, b, c, w = sp.symbols("s a b c w", positive=True)


def report(name, L):
    dL = sp.simplify(sp.diff(L, s))
    print(f"  {name}")
    print(f"    L(s)     = {sp.simplify(L)}")
    print(f"    dL/ds    = {dL}")
    return dL


def test_z_space_loss_pays_the_model_to_shrink():
    """The retracted design. Score the forecast against z, and z is the
    encoder's own output: scaling it by s scales the error by s, so the loss
    falls as s -> 0 while nothing about the prediction improves."""
    print("z-space forecast term (E-004 family, retracted):")
    L = (s * a - s * b) ** 2 / c
    dL = report("L = ||s*a - s*b||^2 / c", L)
    # A strictly positive derivative for s>0 means gradient descent drives
    # s downward — free loss reduction, no predictive gain.
    assert sp.simplify(dL - 2 * s * (a - b) ** 2 / c) == 0
    assert sp.ask(sp.Q.positive(dL.subs({a: 3, b: 1, c: 1, s: 1}))) or \
        dL.subs({a: 3, b: 1, c: 1, s: 1}) > 0
    print("    -> strictly increasing in s, so descent shrinks z. THE CHEAT.\n")


def test_data_space_loss_makes_the_scale_a_gauge():
    """E-006. The forecast is DECODED before scoring. Under the
    reparametrisation (z -> s*z, decoder -> decoder/s) the decoded field is
    unchanged, so the loss cannot see s at all: the derivative is exactly
    zero, not merely small."""
    print("data-space forecast term (E-006):")
    # decoded prediction: (w/s) * (s*a) = w*a  — s cancels, by construction
    x_hat = (w / s) * (s * a)
    L = (x_hat - b) ** 2 / c              # c = var(x), a property of the DATA
    dL = report("L = ||(w/s)(s*a) - x||^2 / var(x)", L)
    assert dL == 0, f"expected an exact gauge, got dL/ds = {dL}"
    print("    -> exactly 0. No descent direction exists to police.\n")


def test_the_denominator_is_not_a_term_in_the_objective():
    """CLAUDE.md 6c rule 2: normalise by properties of the DATA, never of the
    MODEL. var(x) is computed from the tensor once; the model has no path to
    it, so its gradient is identically zero. A frozen-checkpoint reference —
    the E-004d twin — does NOT have this property once what was frozen is what
    you are training."""
    var_x = sp.Symbol("var_x", positive=True)      # constant of the data
    L = (a - b) ** 2 / var_x
    assert sp.diff(L, var_x) != 0, "the loss does depend on the denominator..."
    # ...but the denominator does not depend on anything the model controls:
    for knob in (s, w):
        assert sp.diff(var_x, knob) == 0
    print("data variance as denominator:")
    print("    d(var_x)/d(model params) = 0 for every model knob")
    print("    -> a constant the model cannot move. Fine as a denominator.\n")


def test_the_persistence_baseline_degeneracy_is_structurally_absent():
    """The SECOND cheat, which arrived faster than the one being fixed: with
    a ratio model/persistence, inflating the denominator is as good as
    improving the numerator. E-006 has no baseline in the loss at all — it is
    a diagnostic, logged, not optimised (rule 3)."""
    pers = sp.Symbol("pers", positive=True)
    L_ratio = (a - b) ** 2 / pers
    assert sp.diff(L_ratio, pers) != 0, "a ratio loss IS sensitive to it"
    L_e006 = (a - b) ** 2 / sp.Symbol("var_x", positive=True)
    assert sp.diff(L_e006, pers) == 0
    print("persistence baseline:")
    print("    ratio loss : dL/d(pers) != 0  -> inflating it pays")
    print("    E-006      : dL/d(pers) == 0  -> it is not in the objective\n")


def test_both_terms_are_commensurable_so_a_plain_sum_is_legitimate():
    """Why `sum` replaces the smooth-max. The max existed only because the two
    terms had different units; once both are fractions of the same observed
    variance, each contributes its own gradient without a chosen weight."""
    var_x = sp.Symbol("var_x", positive=True)
    rec, fore = sp.symbols("rec fore", positive=True)
    L = rec / var_x + fore / var_x
    assert sp.simplify(sp.diff(L, rec) - sp.diff(L, fore)) == 0, (
        "equal weighting is the POINT: neither term is privileged by a "
        "constant we picked")
    print("plain sum of two variance-normalised terms:")
    print(f"    dL/d(rec) = dL/d(fore) = {sp.diff(L, rec)}")
    print("    -> weighted by their own gradients, not by our choice.\n")


if __name__ == "__main__":
    test_z_space_loss_pays_the_model_to_shrink()
    test_data_space_loss_makes_the_scale_a_gauge()
    test_the_denominator_is_not_a_term_in_the_objective()
    test_the_persistence_baseline_degeneracy_is_structurally_absent()
    test_both_terms_are_commensurable_so_a_plain_sum_is_legitimate()
    print("OK — the shrinkage direction is a gauge, the baseline is absent "
          "from the objective, and the two terms are commensurable.\n"
          "E-006's algebra is settled. The code is now a transcription.")

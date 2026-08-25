#!/usr/bin/env python3
"""ml/probscore.py — the E-052.0 probabilistic scoreboard, pinned by identities.

The module exists because squared error alone cannot score a sampling head
fairly (`ml/plans/E052_field_diffusion.md`), and it will be quoted beside
`val_zmse` and the corridor AUC in an experiment log. A scoreboard that is
itself unvalidated is a number that looks like a result — so these tests
prefer EXACT expected values to threshold checks wherever an exact one exists
(ml/CLAUDE.md §4.9), and say in a comment why any tolerance is the size it is.

  1. crps_gaussian vs crps_ensemble  — the closed form IS the control for the
     estimator; plus sigma = 0 as an exact degenerate case.
  2. fair vs biased                  — the fair estimator's expectation does
     not move with M and the biased one shrinks with M; at M = 1 the CRPS is
     the mean absolute error EXACTLY; and the sorted-member identity the
     module computes with reproduces a naive O(M²) reference.
  3. the decomposition identity      — mse_sample = mse_mean + mean_var.
  4. spread-error calibration        — a calibrated ensemble reads ~1, an
     ensemble a third too narrow reads ~1/3.
  5. brier_dip                       — a hand-computed M=4, T=5 case, then
     skilful vs climatological forecasts.
  6. NaN handling                    — holes in `obs` give bit-identical
     answers to deleting those elements by hand.
  7. ratio_vs_persistence            — pred == prev is exactly 1.0, pred ==
     obs is exactly 0.0.

    python3 tests/test_probscore.py
"""
import math
import os
import sys
import traceback

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "ml"))

from probscore import (brier_dip, crps_ensemble, crps_gaussian,  # noqa: E402
                       ensemble_decomposition, ratio_vs_persistence,
                       spread_error)


def _naive_crps(ens, obs, fair):
    """The O(M²) pairwise reference the module's sorted identity must match.

    Written out longhand on purpose: it is the definition, and the module's
    O(M log M) rank-weight form is an optimisation of it.
    """
    M = ens.shape[0]
    t1 = np.mean(np.abs(ens - obs), axis=0)
    d = np.abs(ens[:, None, ...] - ens[None, :, ...])
    denom = M * (M - 1) if (fair and M >= 2) else M * M
    t2 = 0.5 * d.sum(axis=(0, 1)) / denom
    return t1 - t2


# --- 1 -------------------------------------------------------------------
def test_crps_gaussian_is_the_ensemble_estimator_limit():
    """A large Gaussian sample scored by the estimator must hit the closed form."""
    rng = np.random.default_rng(51000)
    # 40,000 members x 8 independent columns at the SAME (mu, sigma, obs).
    # The estimator is unbiased, so the only error is sampling: the standard
    # error of mean|x - y| is sd(|x - y|)/sqrt(M) ~ 0.6/200 = 0.003 on a score
    # of ~0.234 at obs = mu, i.e. ~1.3% from one column. Averaging 8
    # independent columns divides that by sqrt(8) and puts the tolerance
    # comfortably at 1e-2 relative rather than on top of it.
    cols = 8
    for mu, sigma, obs in [(0.0, 1.0, 0.0), (0.0, 1.0, 1.5),
                           (-2.0, 0.5, -1.0), (3.0, 2.0, 3.0),
                           (0.0, 1.0, -3.0), (10.0, 0.25, 10.4)]:
        ens = rng.normal(mu, sigma, size=(40000, cols))
        got = crps_ensemble(ens, np.full(cols, obs))["crps"]
        want = float(crps_gaussian(mu, sigma, obs))
        rel = abs(got - want) / abs(want)
        assert rel < 1e-2, (f"mu={mu} sigma={sigma} obs={obs}: ensemble "
                            f"{got:.6f} vs closed form {want:.6f}, "
                            f"rel {rel:.2e}")

    # The degenerate limit is not an approximation and is not tested as one:
    # a point mass at mu scores |obs - mu|, exactly.
    mu = np.array([-3.0, 0.0, 2.5, 7.25])
    obs = np.array([1.0, 0.0, -2.5, 7.25])
    deg = crps_gaussian(mu, 0.0, obs)
    assert np.array_equal(deg, np.abs(obs - mu)), f"sigma=0 gave {deg}"

    # ... and it is the same number an M = 1 ensemble gives, which is what
    # lets a deterministic head onto this scoreboard with no special case.
    one = crps_ensemble(mu[None, :], obs)["crps_field"]
    assert np.array_equal(one, np.abs(obs - mu)), f"M=1 gave {one}"


# --- 2 -------------------------------------------------------------------
def test_fair_estimator_is_M_invariant_and_biased_one_is_not():
    rng = np.random.default_rng(51001)
    R = 100_000                      # elements, each an independent draw
    obs = rng.normal(size=R)         # SHARED across M: correlating the
    #                                  estimates shrinks the range being
    #                                  asserted on without weakening it.
    fair_by_m, biased_by_m = {}, {}
    for M in range(2, 9):
        ens = rng.normal(size=(M, R))
        fair_by_m[M] = crps_ensemble(ens, obs, fair=True)["crps"]
        biased_by_m[M] = crps_ensemble(ens, obs, fair=False)["crps"]

    # Tolerance: each entry is a mean of R = 1e5 per-element scores of sd
    # ~O(1), so its standard error is ~3e-3; seven of them span ~3.3 se ~ 1e-2
    # in the typical case. 0.02 is that with room, and it is still 20x smaller
    # than the (M-1)/M bias the biased column below is being caught by.
    spread = max(fair_by_m.values()) - min(fair_by_m.values())
    assert spread < 0.02, (f"fair CRPS moved with M by {spread:.4f}: "
                           f"{ {k: round(v, 4) for k, v in fair_by_m.items()} }")

    # The biased estimator's spread term is short by (M-1)/M, so its CRPS is
    # too HIGH and falls monotonically towards the fair value as M grows.
    seq = [biased_by_m[M] for M in range(2, 9)]
    assert all(a > b for a, b in zip(seq, seq[1:])), \
        f"biased CRPS is not decreasing in M: {[round(v, 4) for v in seq]}"
    assert biased_by_m[2] - fair_by_m[2] > 0.1, \
        (f"biased and fair are indistinguishable at M=2 "
         f"({biased_by_m[2]:.4f} vs {fair_by_m[2]:.4f}) — the estimator "
         f"switch is not doing anything")

    # M = 1: CRPS *is* MAE. Exact equality, not a tolerance — the pair term is
    # identically zero, so there is nothing for float error to enter through.
    x = rng.normal(size=(1, 400))
    y = rng.normal(size=400)
    for fair in (True, False):
        got = crps_ensemble(x, y, fair=fair)
        assert np.array_equal(got["crps_field"], np.abs(x[0] - y)), \
            f"M=1 field is not |x-y| (fair={fair})"
        assert got["crps"] == float(np.mean(np.abs(x[0] - y))), \
            f"M=1 mean is not MAE (fair={fair})"

    # And the sorted-member identity the module computes with reproduces the
    # naive O(M²) definition. This is what licenses the O(M log M) form.
    ens = rng.normal(size=(6, 500))
    obs6 = rng.normal(size=500)
    for fair in (True, False):
        got = crps_ensemble(ens, obs6, fair=fair)["crps_field"]
        want = _naive_crps(ens, obs6, fair)
        assert np.allclose(got, want, rtol=1e-12, atol=1e-12), \
            (f"sorted identity != naive pairwise (fair={fair}), max |Δ| "
             f"{np.max(np.abs(got - want)):.3e}")


# --- 3 -------------------------------------------------------------------
def test_decomposition_identity_holds_to_float_precision():
    """mse_sample = mse_mean + mean_var — the reason this module exists."""
    rng = np.random.default_rng(51002)
    for M, shape in [(2, (57,)), (5, (13, 17)), (16, (40, 9)), (64, (25,))]:
        truth = rng.normal(size=shape)
        ens = truth[None] + rng.normal(scale=0.7, size=(M,) + shape)
        obs = truth + rng.normal(scale=0.4, size=shape)
        d = ensemble_decomposition(ens, obs)
        lhs, rhs = d["mse_sample"], d["mse_mean"] + d["mean_var"]
        rel = abs(lhs - rhs) / abs(lhs)
        assert rel < 1e-10, (f"M={M} shape={shape}: mse_sample {lhs:.12f} != "
                             f"mse_mean + mean_var {rhs:.12f} (rel {rel:.2e})")

        # The identity is worthless if the terms are not the terms: check
        # mse_sample independently against the plain per-member MSE.
        want = float(np.mean([(np.mean((ens[i] - obs) ** 2)) for i in range(M)]))
        assert abs(d["mse_sample"] - want) < 1e-12 * abs(want), \
            f"mse_sample {d['mse_sample']} is not the mean member MSE {want}"


# --- 4 -------------------------------------------------------------------
def test_spread_error_reads_one_when_calibrated():
    rng = np.random.default_rng(51003)
    T, M, sigma = 20_000, 16, 0.8

    # Calibrated: obs and every member are independent draws from the same
    # predictive law N(truth, sigma^2). E[(xbar-y)^2] = sigma^2 (M+1)/M, which
    # is exactly what the (M+1)/M inflation of the ddof=1 variance estimates.
    truth = rng.normal(size=T)
    obs = truth + sigma * rng.normal(size=T)
    ens = truth[None] + sigma * rng.normal(size=(M, T))
    r = spread_error(ens, obs)["ratio"]
    # Both terms are means of T squares, so each carries ~sqrt(2/T) ~ 1%
    # relative error and the ratio's square ~1.4%; 3% is that with headroom.
    assert abs(r - 1.0) < 0.03, f"calibrated ensemble read ratio {r:.4f}"

    # Over-confident by 3x: members drawn at sigma/3 while the truth still
    # departs by sigma. The expectation is not exactly 1/3 at finite M and is
    # not asserted as if it were --
    #   spread^2 = (M+1)/M (sigma/3)^2 ,  mse = sigma^2 (1 + 1/(9M))
    # so the ratio is (1/3) sqrt( ((M+1)/M) / (1 + 1/(9M)) ) = 0.34237 at
    # M = 16. Assert the analytic value, and separately that it still reads as
    # "about a third".
    ens_nar = truth[None] + (sigma / 3.0) * rng.normal(size=(M, T))
    r3 = spread_error(ens_nar, obs)["ratio"]
    want = (1.0 / 3.0) * math.sqrt(((M + 1.0) / M) / (1.0 + 1.0 / (9.0 * M)))
    assert abs(r3 - want) < 0.02, \
        f"over-confident ensemble read {r3:.4f}, expected {want:.4f}"
    assert 0.30 < r3 < 0.38, f"over-confident ratio {r3:.4f} is not ~1/3"


# --- 5 -------------------------------------------------------------------
def test_brier_dip_hand_computed_and_skill_signs():
    # M = 4, T = 5, event = obs < 0. Worked out by hand:
    #   o   = [1, 0, 1, 0, 1]                       event_rate = 3/5 = 0.6
    #   p   = [3/4, 1/4, 4/4, 0/4, 2/4]
    #   brier      = mean[0.0625, 0.0625, 0, 0, 0.25]            = 0.075
    #   brier_clim = mean[0.16, 0.36, 0.16, 0.36, 0.16]          = 0.24
    #   bss        = 1 - 0.075/0.24                              = 0.6875
    obs = np.array([-1.0, 0.5, -0.2, 2.0, -3.0])
    ens = np.array([[-1.0, 1.0, -1.0, 1.0, -1.0],
                    [-2.0, 1.0, -1.0, 2.0, 1.0],
                    [-3.0, -1.0, -1.0, 3.0, -1.0],
                    [1.0, 1.0, -1.0, 4.0, 1.0]])
    b = brier_dip(ens, obs, 0.0, below=True)
    assert b["n"] == 5, b["n"]
    assert b["event_rate"] == 0.6, b["event_rate"]
    # p_t and o_t are dyadic (quarters and 0/1), so the Brier score is the
    # hand value exactly. The climatological reference is not: it squares
    # (0.6 - o), and 0.6 is not representable in binary, so it and the skill
    # score derived from it land within an ulp or two of the hand arithmetic
    # rather than on it. That is float representation, not an estimator
    # question, so it gets a tolerance and the reason gets written down.
    assert b["brier"] == 0.075, b["brier"]
    assert abs(b["brier_clim"] - 0.24) < 1e-15, b["brier_clim"]
    assert abs(b["bss"] - 0.6875) < 1e-15, b["bss"]

    # The mirrored event (obs > 0) is the complement, so its rate must be
    # 1 - 0.6 and its climatological Brier score identical.
    up = brier_dip(-ens, -obs, 0.0, below=False)
    assert up["event_rate"] == 0.6 and up["brier"] == 0.075, up

    # A skilful forecast: members are the truth plus small noise, so p_t is
    # near-certain and near-right.
    rng = np.random.default_rng(51004)
    T, M = 4000, 50
    truth = rng.normal(size=T)
    skil = truth[None] + 0.15 * rng.normal(size=(M, T))
    s = brier_dip(skil, truth, -0.5)
    assert s["bss"] > 0.5, f"skilful forecast scored bss {s['bss']:.4f}"
    assert s["n"] == T

    # A climatological forecast: members drawn from the same distribution but
    # independent of the truth. bss is ~ -1/M, not 0 — an M-member sample of a
    # correct constant probability adds p(1-p)/M of pure sampling noise to the
    # Brier score. -0.02 at M = 50; the window is that plus sampling slack.
    clim = rng.normal(size=(M, T))
    c = brier_dip(clim, truth, -0.5)
    assert -0.08 < c["bss"] < 0.03, \
        f"climatological forecast scored bss {c['bss']:.4f}, expected ~ -1/M"


# --- 6 -------------------------------------------------------------------
def test_nan_holes_equal_deleting_those_elements():
    """A hole must cost the element, and nothing else.

    Data is integer-valued on purpose. The masked path sums the full array
    with the holes zeroed while the reference sums a shorter array, and numpy
    pairwise-sums both — with exactly representable values the two sums are
    the SAME exact number, so this can be asserted as equality rather than as
    a tolerance that would hide a real off-by-one in the denominator.
    """
    rng = np.random.default_rng(51005)
    M, T = 8, 24
    ens = rng.integers(-8, 9, size=(M, T)).astype(np.float64)
    obs = rng.integers(-8, 9, size=T).astype(np.float64)
    prev = rng.integers(-8, 9, size=T).astype(np.float64)
    pred = rng.integers(-8, 9, size=T).astype(np.float64)

    holes = np.array([3, 11, 17])
    keep = np.setdiff1d(np.arange(T), holes)
    obs_h = obs.copy()
    obs_h[holes] = np.nan

    for fair in (True, False):
        full = crps_ensemble(ens, obs_h, fair=fair)
        cut = crps_ensemble(ens[:, keep], obs[keep], fair=fair)
        assert np.all(np.isnan(full["crps_field"][holes])), \
            "a hole did not produce NaN in crps_field"
        # Element-wise the two are the same arithmetic, so this is exact for
        # either estimator.
        assert np.array_equal(full["crps_field"][keep], cut["crps_field"]), \
            f"scored elements changed when holes were present (fair={fair})"
        if fair:
            # The fair divisor M(M-1) = 56 makes per-element scores
            # non-dyadic, so only the summation ORDER can differ here; one ulp
            # of relative slack, no more.
            assert abs(full["crps"] - cut["crps"]) <= 1e-15 * abs(cut["crps"]), \
                f"fair crps {full['crps']!r} vs {cut['crps']!r}"
        else:
            assert full["crps"] == cut["crps"], \
                f"biased crps {full['crps']!r} vs {cut['crps']!r}"

    # Every term of the decomposition divides by M = 8 and squares integers,
    # so both paths reach the identical exact sum.
    a = ensemble_decomposition(ens, obs_h)
    b = ensemble_decomposition(ens[:, keep], obs[keep])
    for k in a:
        assert a[k] == b[k], f"ensemble_decomposition[{k}]: {a[k]!r} vs {b[k]!r}"

    # spread_error divides by ddof = M - 1 = 7 and inflates by 9/8, so its
    # per-element values are not exactly representable and the two summation
    # orders can part company in the last bit. One ulp of relative slack --
    # anything larger would mean a different element set, which is what this
    # test is actually about.
    a = spread_error(ens, obs_h)
    b = spread_error(ens[:, keep], obs[keep])
    for k in a:
        assert abs(a[k] - b[k]) <= 1e-15 * abs(b[k]), \
            f"spread_error[{k}]: {a[k]!r} vs {b[k]!r}"

    a = ratio_vs_persistence(pred, obs_h, prev)
    b = ratio_vs_persistence(pred[keep], obs[keep], prev[keep])
    for k in a:
        assert a[k] == b[k], f"ratio_vs_persistence[{k}]: {a[k]!r} vs {b[k]!r}"
    # ... and a boolean mask must be the same thing as a hole.
    mask = np.ones(T, bool)
    mask[holes] = False
    c = ratio_vs_persistence(pred, obs, prev, mask=mask)
    for k in a:
        assert c[k] == b[k], f"mask != hole for [{k}]: {c[k]!r} vs {b[k]!r}"

    a = brier_dip(ens, obs_h, 0.0)
    b = brier_dip(ens[:, keep], obs[keep], 0.0)
    assert a == b, f"brier_dip differed under holes: {a} vs {b}"
    assert a["n"] == T - len(holes), a["n"]

    # A fully-NaN input returns NaN and does not raise — instrumentation must
    # never be the thing that loses a job (ml/CLAUDE.md §1).
    dead = np.full(T, np.nan)
    assert math.isnan(crps_ensemble(ens, dead)["crps"])
    assert math.isnan(spread_error(ens, dead)["ratio"])
    assert math.isnan(ensemble_decomposition(ens, dead)["mse_mean"])
    assert math.isnan(ratio_vs_persistence(pred, dead, prev)["ratio"])
    assert brier_dip(ens, dead, 0.0)["n"] == 0
    assert math.isnan(brier_dip(ens, dead, 0.0)["bss"])


# --- 7 -------------------------------------------------------------------
def test_ratio_vs_persistence_anchors():
    """Persistence is 1.0 by construction and a perfect forecast is 0.0."""
    rng = np.random.default_rng(51006)
    obs = rng.normal(size=(37, 11))
    prev = obs + rng.normal(scale=0.9, size=obs.shape)
    assert ratio_vs_persistence(prev, obs, prev)["ratio"] == 1.0, \
        ("pred == prev did not give exactly 1.0 — the two MSEs are not being "
         "taken over the same elements")
    perfect = ratio_vs_persistence(obs, obs, prev)
    assert perfect["mse"] == 0.0 and perfect["ratio"] == 0.0, perfect

    # And the ordinary case is the plain ratio of the two MSEs, with a mask.
    pred = obs + rng.normal(scale=0.5, size=obs.shape)
    mask = rng.random(obs.shape) < 0.6
    got = ratio_vs_persistence(pred, obs, prev, mask=mask)
    want_n = float(np.mean((pred - obs)[mask] ** 2))
    want_d = float(np.mean((prev - obs)[mask] ** 2))
    assert abs(got["mse"] - want_n) < 1e-12
    assert abs(got["mse_pers"] - want_d) < 1e-12
    assert abs(got["ratio"] - want_n / want_d) < 1e-12
    assert got["ratio"] < 1.0, "a better-than-persistence forecast read >= 1"

    # No baseline error means no ratio, and NaN is the honest answer -- a
    # 0/0 that silently returned 1.0 would read as "matched persistence".
    z = np.zeros(5)
    assert math.isnan(ratio_vs_persistence(z, z, z)["ratio"])


def main():
    tests = [(n, f) for n, f in list(globals().items())
             if n.startswith("test_") and callable(f)]
    failed = []
    for name, fn in tests:
        try:
            fn()
        except Exception:
            failed.append(name)
            print(f"FAIL {name}")
            traceback.print_exc()
        else:
            print(f"ok   {name}")
    print(f"\n{len(tests) - len(failed)}/{len(tests)} probscore checks hold")
    if failed:
        print("failed: " + ", ".join(failed))
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())

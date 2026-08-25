#!/usr/bin/env python3
"""Probabilistic scores for stage-2 forecast heads — the E-052.0 scoreboard.

Plan: `ml/plans/E052_field_diffusion.md`. Everything here writes NEW keys
BESIDE the existing read-outs. Nothing in this module replaces, reweights or
re-derives `val_zmse`, `val_persistence`, `msss_clim`, the corridor AUC,
`read_sv`, the eval gate or any archived number — an additional key can be
ignored by every reader that predates it, and a changed one silently rewrites
a 100-run column.

**Why it exists.** Stage 2 is scored today by squared error alone. That is a
fair scoreboard for a head that emits a conditional MEAN and a rigged one for
a head that emits a SAMPLE, and the rigging is an exact identity rather than a
tendency: for a draw x from p(x|c) against truth y,

    E|x - y|^2  =  |E[x|c] - y|^2  +  Var[x|c],

so a single sample's expected MSE exceeds the conditional mean's by EXACTLY
the conditional variance. A generative head that has learned the predictive
distribution perfectly still loses on MSE to a blurred point forecast, and
loses by more the more genuinely uncertain the future is — i.e. most at pentad
and daily cadence, which is precisely where the sampling head's advantage
would live. Squared error alone therefore decides the E-052 question before
the experiment is run. CRPS, the spread-error ratio and a dip-event Brier
score are the instruments that do not.

The same identity is the module's own self-test: `ensemble_decomposition`
returns the three terms and `tests/test_probscore.py` pins
`mse_sample == mse_mean + mean_var` to float precision.

**Conventions.**

- Pure numpy, CPU, no torch and no scipy (the boxes do not all have it).
- NaN-aware throughout: land and missing cells arrive as NaN in the
  observations AND in the ensemble. Every mean here excludes non-finite
  elements rather than propagating them, and an input with nothing finite in
  it returns NaN rather than raising — a metric must never be the thing that
  loses a job (ml/CLAUDE.md §1, the NaN-probe rule).
- Lower is better for `crps`, `brier`, `mse*` and `ratio`; higher is better
  for `bss`; `spread/rmse` is scored against 1.0 from either side.

    python3 -c "import ml.probscore"   # no side effects, no imports beyond numpy
"""
import math

import numpy as np

__all__ = [
    "crps_ensemble",
    "crps_gaussian",
    "ensemble_decomposition",
    "spread_error",
    "brier_dip",
    "ratio_vs_persistence",
]

# math.erf rather than scipy.special.erf: scipy is not installed on every box
# in the fleet, and a scoreboard that cannot be imported is worse than a slow
# one. np.vectorize is a python loop, which is fine — crps_gaussian is the
# closed-form CONTROL for the ensemble estimator, evaluated on test points and
# on Gaussian baselines, never on the 84,405-pixel field in a training loop.
_erf = np.vectorize(math.erf, otypes=[np.float64])

_INV_SQRT_PI = 1.0 / math.sqrt(math.pi)
_INV_SQRT_2PI = 1.0 / math.sqrt(2.0 * math.pi)


def _nanmean(a):
    """Mean over the finite elements of `a`; NaN when nothing is finite.

    np.nanmean does the same thing but emits a RuntimeWarning on an all-NaN
    input, and an all-NaN input is an ordinary, expected state here (a scored
    window that happens to be entirely land). A warning that fires on a normal
    condition is a warning nobody reads.
    """
    a = np.asarray(a, dtype=np.float64)
    ok = np.isfinite(a)
    n = int(ok.sum())
    if n == 0:
        return float("nan")
    return float(np.sum(np.where(ok, a, 0.0)) / n)


def _member_stats(ens):
    """Per-element valid-member count, mean and centred sum of squares.

    Returns (m, mean, ss) with `m` the count of FINITE members at each element
    — not `ens.shape[0]`. The count is per element because a member can be NaN
    where the observation is not (a sampler that diverged on one draw, a
    partially masked field), and every estimator below divides by the members
    it actually has rather than by the ones it was handed.
    """
    valid = np.isfinite(ens)
    m = valid.sum(axis=0).astype(np.float64)
    m_safe = np.where(m > 0, m, 1.0)
    filled = np.where(valid, ens, 0.0)
    mean = np.sum(filled, axis=0) / m_safe
    dev = np.where(valid, ens - mean, 0.0)
    ss = np.sum(dev * dev, axis=0)
    return m, np.where(m > 0, mean, np.nan), ss


def crps_ensemble(ens, obs, fair=True, axis=0):
    """Continuous Ranked Probability Score of an M-member ensemble.

    The standard ensemble estimator, per element:

        CRPS = (1/M) Σ_i |x_i - y|  -  (1/2) * (1/D) Σ_{i,j} |x_i - x_j|

    with D = M*(M-1) for the FAIR (unbiased) estimator and D = M² for the
    biased one. The double sum runs over all ordered pairs; the i=j terms are
    zero, so the biased form needs no correction.

    **Why fair by default.** The biased estimator's spread term is short by a
    factor (M-1)/M, so a biased CRPS improves as M grows for a fixed, correct
    predictive distribution — it rewards buying members. That makes it useless
    for the E-052 comparison, whose whole point is to put an M-member
    generative head beside a deterministic one (M=1) on the same axis. The
    fair estimator's expectation does not depend on M, so the number compares
    across arms with different ensemble sizes and against a degenerate M=1
    head without a correction anybody has to remember.

    One consequence of unbiasedness worth knowing before it is reported as a
    bug: a fair PER-ELEMENT score can come out slightly negative at small M
    (a two-member ensemble straddling the truth scores exactly 0). Only its
    expectation is a CRPS; the field mean over a real number of elements is
    positive, and the aggregate is what gets quoted.

    **M = 1.** The fair form's divisor is zero, so it falls back to the biased
    form, whose pair sum is identically zero: **CRPS at M = 1 equals MAE
    exactly**, not approximately. That is the property that lets a
    deterministic head enter the scoreboard as a one-member ensemble instead
    of as a special case, and `tests/test_probscore.py` asserts the exact
    equality.

    Computed through the sorted-member identity

        Σ_{i,j} |x_i - x_j| = 2 Σ_k (2k - m + 1) x_(k)      (x_(k) ascending)

    which is O(M log M) per element rather than the O(M²) einsum. The naive
    form is clearer and would be preferred at the M ≤ 64 this programme
    actually runs, but the closed-form cross-check in the test suite scores a
    40,000-member Gaussian ensemble, and the pairwise array for that is
    1.6e9 elements — a test that cannot run is not a control. The test pins
    the sorted path against a naive einsum reference at small M.

    Args:
      ens: ensemble array with an ensemble axis of length M; the remaining
        axes must broadcast against `obs`.
      obs: observations. NaN marks land/missing.
      fair: fair (unbiased, M-independent) estimator when True.
      axis: which axis of `ens` holds the members.

    Returns:
      {"crps": float — mean over the elements that scored,
       "crps_field": ndarray — per-element CRPS, NaN where `obs` is NaN or
                     where no member is finite}
    """
    ens = np.moveaxis(np.asarray(ens, dtype=np.float64), axis, 0)
    obs = np.asarray(obs, dtype=np.float64)
    m = np.isfinite(ens).sum(axis=0).astype(np.float64)
    m_safe = np.where(m > 0, m, np.nan)

    # np.nansum of an all-NaN slice is 0.0 and warns about nothing; the
    # element is masked out below on `m` and on obs anyway.
    term1 = np.nansum(np.abs(ens - obs), axis=0) / m_safe

    # np.sort puts NaN LAST, so the m finite members occupy positions
    # 0..m-1 and the weight (2k - m + 1) is the per-element rank weight for
    # exactly those. The NaN tail is zeroed so it contributes nothing.
    xs = np.sort(ens, axis=0)
    k = np.arange(ens.shape[0], dtype=np.float64).reshape(
        (-1,) + (1,) * (ens.ndim - 1))
    w = 2.0 * k - m + 1.0
    pair_sum = 2.0 * np.sum(w * np.where(np.isfinite(xs), xs, 0.0), axis=0)

    if fair:
        denom = np.where(m >= 2.0, m * (m - 1.0), m * m)
    else:
        denom = m * m
    term2 = 0.5 * pair_sum / np.where(denom > 0.0, denom, np.nan)

    field = term1 - term2
    field = np.where(np.isfinite(obs) & (m > 0.0), field, np.nan)
    return {"crps": _nanmean(field), "crps_field": field}


def crps_gaussian(mu, sigma, obs):
    """Closed-form CRPS of a Gaussian forecast N(mu, sigma²) against `obs`.

        CRPS = sigma * ( z (2Φ(z) - 1) + 2φ(z) - 1/√π ),   z = (obs - mu)/sigma

    This is the ANALYTIC CONTROL for `crps_ensemble`: a large Gaussian sample
    scored by the ensemble estimator must reproduce it, which is what turns
    "the estimator looks reasonable" into a number with a known right answer
    (ml/CLAUDE.md §4.9 — prefer an exact expected value to a threshold).

    It is also the score for any head that reports a mean and a spread rather
    than members, so a Gaussian-parameterised baseline can be put on the same
    axis as a sampler with no ensemble drawn at all.

    sigma = 0 is the degenerate limit — the point mass at mu — and returns
    |obs - mu| exactly, which is the same MAE the M=1 ensemble returns. A
    negative sigma is not a distribution and returns NaN.
    """
    mu, sigma, obs = np.broadcast_arrays(
        np.asarray(mu, dtype=np.float64),
        np.asarray(sigma, dtype=np.float64),
        np.asarray(obs, dtype=np.float64))
    with np.errstate(divide="ignore", invalid="ignore"):
        z = (obs - mu) / sigma
        cdf = 0.5 * (1.0 + _erf(z / math.sqrt(2.0)))
        pdf = _INV_SQRT_2PI * np.exp(-0.5 * z * z)
        core = sigma * (z * (2.0 * cdf - 1.0) + 2.0 * pdf - _INV_SQRT_PI)
    out = np.where(sigma == 0.0, np.abs(obs - mu), core)
    return np.where(sigma < 0.0, np.nan, out)


def ensemble_decomposition(ens, obs, axis=0):
    """The deterministic-vs-sample identity, in its three terms.

    Elementwise and exactly, for a population variance (ddof = 0):

        (1/M) Σ_i (x_i - y)²  =  (x̄ - y)²  +  Var_i(x_i)
        └── mse_sample ──┘        └ mse_mean ┘   └ mean_var ┘

    So **mse_sample = mse_mean + mean_var**, to float error. This is the whole
    argument for E-052.0 in one line: a sample is penalised by the ensemble
    variance relative to the ensemble mean, always, regardless of how good the
    distribution is. Reporting the three terms together makes the tax legible
    instead of leaving it inside a single ratio — a generative arm whose
    `mse_mean` matches the deterministic head's while its `mse_sample` is
    worse has not failed, it has spread.

    The identity holds under NaN only if all three terms average over the SAME
    elements and the same members, so an element scores when `obs` is finite
    and at least one member is finite, and each element's member mean, its
    variance and its per-member errors all use that element's finite members.
    `mse_sample` is therefore the mean over elements of the per-element mean
    over members — identical to "mean over members of each member's MSE" when
    nothing is missing, and the only reading that keeps the identity when
    something is.

    Returns {"mse_mean", "mean_var", "mse_sample"} — floats, NaN if nothing
    scored.
    """
    ens = np.moveaxis(np.asarray(ens, dtype=np.float64), axis, 0)
    obs = np.asarray(obs, dtype=np.float64)
    m, mean, ss = _member_stats(ens)
    ok = np.isfinite(obs) & (m > 0.0)

    err_mean = (mean - obs) ** 2
    var0 = ss / np.where(m > 0.0, m, np.nan)
    # Per-element mean over members of (x_i - y)^2, built from the identity's
    # own right-hand side so the two sides cannot drift apart through a
    # different NaN convention; the test asserts the identity against a
    # separately computed sample MSE.
    sq = np.where(np.isfinite(ens), (ens - obs) ** 2, np.nan)
    samp = np.nansum(np.where(np.isfinite(sq), sq, 0.0), axis=0) / \
        np.where(m > 0.0, m, np.nan)

    return {
        "mse_mean": _nanmean(np.where(ok, err_mean, np.nan)),
        "mean_var": _nanmean(np.where(ok, var0, np.nan)),
        "mse_sample": _nanmean(np.where(ok, samp, np.nan)),
    }


def spread_error(ens, obs, axis=0):
    """Spread-error ratio with the finite-ensemble (M+1)/M correction.

        spread = sqrt( mean_elements[ (M+1)/M · Var_{ddof=1}(x_i) ] )
        rmse   = sqrt( mean_elements[ (x̄ - y)² ] )
        ratio  = spread / rmse

    **Why the correction, and why the ratio is read against 1.** For a
    calibrated M-member ensemble whose members and the truth are exchangeable
    draws from the same predictive law,

        E[(x̄ - y)²] = σ²(M+1)/M      and      E[Var_{ddof=1}] = σ²,

    because the ensemble MEAN carries its own σ²/M of sampling error on top of
    the truth's σ². Comparing a raw spread against the ensemble-mean RMSE
    therefore reports under-dispersion at every finite M even for a perfect
    ensemble — the correction removes an artefact, it does not tune a number.
    With it, **ratio ≈ 1 is calibration**, ratio < 1 is over-confidence (the
    classic failure of a sampler trained to a squared-error objective), and
    ratio > 1 is an ensemble that has been inflated past its own error.

    An element scores when `obs` is finite and it has at least 2 finite
    members; a variance needs two, and both terms must average over one
    population or the ratio is not a ratio.

    Returns {"spread", "rmse", "ratio"} — floats, NaN if nothing scored (and
    `ratio` NaN if rmse is 0, which is a perfect forecast, not a calibration
    statement).
    """
    ens = np.moveaxis(np.asarray(ens, dtype=np.float64), axis, 0)
    obs = np.asarray(obs, dtype=np.float64)
    m, mean, ss = _member_stats(ens)
    ok = np.isfinite(obs) & (m >= 2.0)

    var1 = ss / np.where(m >= 2.0, m - 1.0, np.nan)
    inflated = (m + 1.0) / np.where(m > 0.0, m, np.nan) * var1
    msp = _nanmean(np.where(ok, inflated, np.nan))
    mse = _nanmean(np.where(ok, (mean - obs) ** 2, np.nan))

    spread = math.sqrt(msp) if msp == msp and msp >= 0.0 else float("nan")
    rmse = math.sqrt(mse) if mse == mse and mse >= 0.0 else float("nan")
    ratio = spread / rmse if rmse == rmse and rmse > 0.0 else float("nan")
    return {"spread": spread, "rmse": rmse, "ratio": ratio}


def brier_dip(ens_series, obs_series, thresh, below=True):
    """Brier score and skill score for a threshold ("dip") event on a series.

    The event at time t is `obs_t < thresh` (or `>` when `below=False`) — for
    E-052 the archetype is an AMOC transport dip below a stated Sv level. The
    forecast probability is the fraction of members that produce the event:

        p_t = #{i : event(x_it)} / M_t
        brier      = mean_t (p_t - o_t)²
        brier_clim = mean_t (p̄ - o_t)²        with p̄ the event rate
        bss        = 1 - brier / brier_clim

    **Why a threshold event at all.** MSE and CRPS are whole-distribution
    scores; the question a user brings to an AMOC forecast is a TAIL question
    ("how likely is a dip below X"), and a head can be excellent in the middle
    and useless in the tail. This is also the one score a deterministic head
    cannot fake: its p_t is 0 or 1 and it is scored on being wrong outright.

    The reference is the CONSTANT climatological probability taken from the
    observations themselves — the standard reference for a skill score, and
    the only one available without a second dataset. Note that it makes `bss`
    an in-sample-referenced number: an uninformative M-member ensemble scores
    bss ≈ -1/M rather than 0, because sampling noise in p_t adds p̄(1-p̄)/M to
    the Brier score. Read a small negative bss as "no skill", not as harm.

    **Shape.** `ens_series` is [M, T] and `obs_series` is [T] — deliberately
    2-D. Reducing a field to a series is the CALLER's decision (which pixels,
    which weighting, pooled or a section contrast) and it is exactly the kind
    of decision that must not be hidden inside a scoring function; §3's whole
    pooled/unpooled argument is about read-outs that quietly chose one.

    A time scores when `obs_t` is finite and at least one member is finite.

    Returns {"brier", "brier_clim", "bss", "event_rate", "n"}; `bss` is NaN
    when `brier_clim` is 0 (the event never happened, or always did — no
    reference to have skill against), and every float is NaN when n == 0.
    """
    ens_series = np.asarray(ens_series, dtype=np.float64)
    obs_series = np.asarray(obs_series, dtype=np.float64)
    # A guard where the inputs are all it has cost (ml/CLAUDE.md §0.3): a
    # [T, M] transposition would score silently and wrongly.
    if ens_series.ndim != 2:
        raise ValueError(f"ens_series must be 2-D [M, T], got shape "
                         f"{ens_series.shape} — reduce the field to a series "
                         f"before scoring")
    if obs_series.ndim != 1:
        raise ValueError(f"obs_series must be 1-D [T], got shape "
                         f"{obs_series.shape}")
    if ens_series.shape[1] != obs_series.shape[0]:
        raise ValueError(f"ens_series is [M={ens_series.shape[0]}, "
                         f"T={ens_series.shape[1]}] but obs_series has "
                         f"T={obs_series.shape[0]}")

    valid_mem = np.isfinite(ens_series)
    m = valid_mem.sum(axis=0).astype(np.float64)
    ok = np.isfinite(obs_series) & (m > 0.0)
    n = int(ok.sum())
    if n == 0:
        return {"brier": float("nan"), "brier_clim": float("nan"),
                "bss": float("nan"), "event_rate": float("nan"), "n": 0}

    if below:
        hit = valid_mem & (ens_series < thresh)
        occ = (obs_series < thresh).astype(np.float64)
    else:
        hit = valid_mem & (ens_series > thresh)
        occ = (obs_series > thresh).astype(np.float64)
    p = hit.sum(axis=0).astype(np.float64) / np.where(m > 0.0, m, np.nan)

    o = occ[ok]
    p = p[ok]
    rate = float(np.sum(o) / n)
    brier = float(np.sum((p - o) ** 2) / n)
    clim = float(np.sum((rate - o) ** 2) / n)
    bss = 1.0 - brier / clim if clim > 0.0 else float("nan")
    return {"brier": brier, "brier_clim": clim, "bss": bss,
            "event_rate": rate, "n": n}


def ratio_vs_persistence(pred, obs, prev, mask=None):
    """One-step MSE ratio against persistence, on one shared element set.

        ratio = mean(pred - obs)² / mean(prev - obs)²

    This mirrors the repo's existing stage-2 convention — `val_zmse` over
    `val_persistence`, where the persistence forecast is simply the previous
    state — so an E-052 field head's number is directly comparable with the
    stencil head's: **lower is better, and persistence is 1.0 by
    construction**. It is a convenience for the E-052 trainer, not a new
    instrument; nothing about the archived stage-2 column changes.

    The two MSEs are averaged over the SAME elements: `mask` (True = include)
    intersected with the elements where `pred`, `obs` and `prev` are all
    finite. A ratio whose numerator and denominator score different
    populations is not a ratio, and the failure is invisible in the output —
    which is why the intersection is taken here rather than left to callers.

    Returns {"mse", "mse_pers", "ratio"}; `ratio` is NaN when the persistence
    MSE is 0 (nothing moved, so there is no baseline error to beat), and every
    float is NaN when nothing scored.
    """
    pred = np.asarray(pred, dtype=np.float64)
    obs = np.asarray(obs, dtype=np.float64)
    prev = np.asarray(prev, dtype=np.float64)
    ok = np.isfinite(pred) & np.isfinite(obs) & np.isfinite(prev)
    if mask is not None:
        ok = ok & np.asarray(mask, dtype=bool)

    n = int(ok.sum())
    if n == 0:
        return {"mse": float("nan"), "mse_pers": float("nan"),
                "ratio": float("nan")}
    d = np.where(ok, pred - obs, 0.0)
    dp = np.where(ok, prev - obs, 0.0)
    mse = float(np.sum(d * d) / n)
    mse_pers = float(np.sum(dp * dp) / n)
    ratio = mse / mse_pers if mse_pers > 0.0 else float("nan")
    return {"mse": mse, "mse_pers": mse_pers, "ratio": ratio}

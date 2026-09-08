# E-076b · the optimal-interpolation baseline, checked against a field whose
# answer is known in closed form.
#
# `ml/oi_ceiling.py` asks whether anything can beat the seasonal climatology at
# a float's own position given the k nearest other float profiles. The estimator
# it uses — optimal interpolation, a Gaussian-process estimate of the anomaly
# from the neighbours' anomalies — has an exact expected error on a Gaussian
# field with known correlation length L, decorrelation time T and noise, and
# that identity is what this file tests. A synthetic field is the only place
# the identity is available: on the real ocean the truth is a measurement and
# there is nothing to compare the estimator's own arithmetic against.
#
# Four checks, in the order they would catch a mistake:
#   1. OI with the TRUE hyper-parameters reaches the error the theory predicts,
#      and therefore beats climatology by the amount the theory predicts.
#   2. The grid search, given only the data, recovers L and T to within one
#      grid step — so the fit is measuring the field and not the grid's centre.
#   3. Climatology, nearest-neighbour and inverse-distance behave as they must
#      on this field (climatology reads the field's own sd; both interpolators
#      beat it; OI beats both, because OI is the minimum-variance estimator and
#      the other two are not).
#   4. The missing-value handling is EXACT: dropping a neighbour by NaN gives
#      bit-for-bit the answer a system built without that neighbour gives.
#
# No network, no store, no tensor. Runs in a few seconds on two cores.
import os
import sys
import time

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "ml"))
import oi_ceiling as oi                                        # noqa: E402

# The truth. L and T sit ON the grid so "within one grid step" is a statement
# about the search and not about rounding; r is the noise-to-signal ratio, so
# the observation noise sd is sqrt(r) * sigma_s.
TRUE_L_KM, TRUE_T_DAYS, TRUE_R = 200.0, 20.0, 0.3
SIGMA_S = 1.4          # signal sd, in the units of the field
N_TARGETS, K = 4000, 8
R_MAX_KM, T_MAX_DAYS = 1000.0, 30.0     # the search bounds, i.e. the metric
# Where the synthetic neighbours actually sit. Deliberately DENSER than the
# search bounds: the point of the file is to check the estimator's arithmetic
# against a closed form, which needs a regime where OI has a large and exactly
# predictable gain. Scattering k=8 neighbours over the full 1,000 km disc with
# L=200 km puts almost all of them outside the correlation length, and then
# every estimator sits within a few percent of climatology — which is a fair
# picture of the real Argo array and a poor test of the code.
RAD_KM, DT_SPAN_DAYS = 250.0, 10.0


def make_field(seed=7, n=N_TARGETS, k=K, L=TRUE_L_KM, T=TRUE_T_DAYS,
               r=TRUE_R, sigma_s=SIGMA_S, n_lev=1):
    """A Gaussian-process field sampled at a target and its k neighbours.

    The generative model is OI's own: a smooth SIGNAL with covariance
    sigma_s^2 * exp(-d^2/2L^2 - dt^2/2T^2) over the k+1 points, of which the
    k neighbours are OBSERVED with independent noise of variance r*sigma_s^2
    and the target is the quantity to be estimated. Returns the module's own
    `Neighbours` container, so the estimators under test see exactly the
    structure they see on the real store.
    """
    rng = np.random.default_rng(seed)
    nb = oi.Neighbours(n, k, n_lev)
    # Neighbours scattered around the target, which sits at the origin at time
    # zero — that is how the gather defines the offsets. They are then SORTED
    # by the search's own space-time metric, so slot 0 really is the nearest,
    # exactly as `ArgoStore.knearest` returns them.
    rad = RAD_KM * np.sqrt(rng.random((n, k)))
    ang = rng.random((n, k)) * 2 * np.pi
    dx, dy = rad * np.cos(ang), rad * np.sin(ang)
    dt_ = rng.random((n, k)) * DT_SPAN_DAYS
    d2 = (dx * dx + dy * dy) / R_MAX_KM ** 2 + (dt_ / T_MAX_DAYS) ** 2
    o = np.argsort(d2, axis=1, kind="stable")
    rows = np.arange(n)[:, None]
    nb.dx, nb.dy, nb.dt = dx[rows, o], dy[rows, o], dt_[rows, o]
    nb.valid[:] = True
    nb.rank = np.tile(np.arange(k), (n, 1))

    # Coordinates of all k+1 points, target first.
    X = np.concatenate([np.zeros((n, 1)), nb.dx], axis=1)
    Y = np.concatenate([np.zeros((n, 1)), nb.dy], axis=1)
    Tt = np.concatenate([np.zeros((n, 1)), nb.dt], axis=1)
    d2 = ((X[:, :, None] - X[:, None, :]) ** 2
          + (Y[:, :, None] - Y[:, None, :]) ** 2) / (2 * L * L) \
        + ((Tt[:, :, None] - Tt[:, None, :]) ** 2) / (2 * T * T)
    C = (sigma_s ** 2) * np.exp(-d2)
    # A jitter far below the noise, so the Cholesky of the SIGNAL covariance
    # exists at points that can coincide.
    C[:, np.arange(k + 1), np.arange(k + 1)] += 1e-9 * sigma_s ** 2
    Lc = np.linalg.cholesky(C)
    s = np.einsum("nij,nj->ni", Lc, rng.standard_normal((n, k + 1)))
    noise = np.sqrt(r) * sigma_s * rng.standard_normal((n, k))
    nb.y_t = s[:, :1].copy()                       # the target: signal only
    nb.a_t = (s[:, 1:] + noise)[:, :, None]        # observed with noise
    nb.y_s = nb.y_t.copy()
    nb.a_s = nb.a_t.copy()
    nb.n_R[:] = k
    return nb


def theory_rmse(nb, L=TRUE_L_KM, T=TRUE_T_DAYS, r=TRUE_R, sigma_s=SIGMA_S):
    """The expected error of the BLUE, averaged over the drawn geometries.

    For one target, var = sigma_s^2 * (1 - c^T (C + rI)^-1 c) with c and C the
    CORRELATION functions (sigma_s^2 factors out of both sides), so the RMSE
    over a set of targets is the root of the mean of that.
    """
    c_ox, c_xx = oi.oi_matrices(nb, L, T)
    d = np.arange(nb.K)
    M = c_xx.copy()
    M[:, d, d] = 1.0 + r
    w = np.linalg.solve(M, c_ox[:, :, None])[:, :, 0]
    var = (sigma_s ** 2) * (1.0 - np.sum(w * c_ox, axis=1))
    return float(np.sqrt(np.mean(var)))


# ------------------------------------------------ (1) OI reaches the theory --
def test_oi_with_true_hyperparameters_reaches_the_predicted_error():
    nb = make_field()
    mask = np.isfinite(nb.y_t)
    c_ox, c_xx = oi.oi_matrices(nb, TRUE_L_KM, TRUE_T_DAYS)
    pred = oi.predict_oi(nb.a_t[:, :, 0], nb, c_ox, c_xx, TRUE_R)
    got, n = oi.rmse(pred - nb.y_t[:, 0], mask[:, 0])
    want = theory_rmse(nb)
    assert n == N_TARGETS
    # The sampling error on an RMSE from n independent draws is ~1/sqrt(2n)
    # relative = 1.1% here; 5% is four sigma and still catches any real bug.
    assert abs(got - want) / want < 0.05, (got, want)

    # ... and therefore beats climatology by the amount the theory predicts.
    # Climatology on this field is "predict zero", whose RMSE is the field's
    # own sd, so the predicted skill ratio is want / sigma_s.
    clim, _ = oi.rmse(-nb.y_t[:, 0], mask[:, 0])
    assert abs(clim - SIGMA_S) / SIGMA_S < 0.05, clim
    assert abs((got / clim) - (want / SIGMA_S)) < 0.05
    # The field is smooth enough at these radii that the gain is large — if
    # this ever reads ~1.0 the estimator has stopped using its neighbours.
    assert got / clim < 0.85, got / clim


# --------------------------------------- (2) the grid search finds L and T --
def test_grid_search_recovers_L_and_T_within_one_step():
    nb = make_field(seed=11)
    mask = np.isfinite(nb.y_t)
    L, T, r, score = oi.fit_oi(nb.a_t, nb, nb.y_t, mask, [0])
    iL = list(oi.GRID_L_KM).index(TRUE_L_KM)
    iT = list(oi.GRID_T_DAYS).index(TRUE_T_DAYS)
    assert L in oi.GRID_L_KM[max(0, iL - 1):iL + 2], (L, TRUE_L_KM)
    assert T in oi.GRID_T_DAYS[max(0, iT - 1):iT + 2], (T, TRUE_T_DAYS)
    # r is the loosest of the three (the likelihood is flat in it once L and T
    # are right), so it is only required to be on the correct side of 1.
    assert r <= 1.0, r
    assert score < SIGMA_S


# ------------------------------- (3) the other three estimators, and order --
def test_climatology_nearest_and_inverse_distance_behave():
    nb = make_field(seed=13)
    mask = np.isfinite(nb.y_t)[:, 0]
    y = nb.y_t[:, 0]
    A = nb.a_t[:, :, 0]
    d2 = oi.metric_d2(nb, R_MAX_KM, T_MAX_DAYS)

    clim, _ = oi.rmse(-y, mask)
    near, _ = oi.rmse(oi.predict_nearest(A, nb) - y, mask)
    idw, _ = oi.rmse(oi.predict_idw(A, nb, d2) - y, mask)
    c_ox, c_xx = oi.oi_matrices(nb, TRUE_L_KM, TRUE_T_DAYS)
    o, _ = oi.rmse(oi.predict_oi(A, nb, c_ox, c_xx, TRUE_R) - y, mask)

    # Slot 0 is the genuinely nearest neighbour here, so copying it beats
    # "predict zero"; inverse-distance beats copying because it averages the
    # noise down; OI beats both, being the minimum-variance estimator.
    assert near < clim, (near, clim)
    assert idw < clim, (idw, clim)
    assert o < idw < near, (o, idw, near)

    # Shrinkage cannot hurt a fit made on the same data, and on a
    # correctly-specified OI it should barely move: the BLUE is already
    # unbiased, so the fitted alpha sits near 1.
    pred = np.full_like(nb.y_t, np.nan)
    pred[:, 0] = oi.predict_oi(A, nb, c_ox, c_xx, TRUE_R)
    alpha, sc = oi.fit_alpha(pred, nb.y_t, np.isfinite(nb.y_t), [0])
    assert 0.85 <= alpha <= 1.0, alpha
    assert sc <= o + 1e-9


# ------------------------- (4) a NaN neighbour is exactly an absent one -----
def test_a_missing_level_is_exactly_a_smaller_system():
    nb = make_field(seed=17, n=200, k=6)
    c_ox, c_xx = oi.oi_matrices(nb, TRUE_L_KM, TRUE_T_DAYS)
    A = nb.a_t[:, :, 0].copy()
    A[:, 2] = np.nan                     # neighbour 2 never measured here
    full = oi.predict_oi(A, nb, c_ox, c_xx, TRUE_R)

    keep = [0, 1, 3, 4, 5]
    small = oi.Neighbours(nb.a_t.shape[0], len(keep), 1)
    small.valid[:] = True
    small.dx, small.dy, small.dt = (nb.dx[:, keep], nb.dy[:, keep],
                                    nb.dt[:, keep])
    small.rank = np.tile(np.arange(len(keep)), (nb.a_t.shape[0], 1))
    s_ox, s_xx = oi.oi_matrices(small, TRUE_L_KM, TRUE_T_DAYS)
    ref = oi.predict_oi(nb.a_t[:, keep, 0], small, s_ox, s_xx, TRUE_R)
    assert np.allclose(full, ref, rtol=0, atol=1e-9), \
        float(np.nanmax(np.abs(full - ref)))

    # And a target with NO usable neighbour at a level answers NaN rather than
    # a silent zero, which is what keeps it out of the RMSE.
    A2 = np.full_like(A, np.nan)
    out = oi.predict_oi(A2, nb, c_ox, c_xx, TRUE_R)
    assert np.all(np.isnan(out))


# ------------------------------------------------------------- housekeeping --
def test_protocol_constants_are_the_frozen_ones():
    # holdout_years = "2008,2009,2016,2017,2021,2022,2023,2024"
    assert oi.split_of_year(2007) == "train"
    assert oi.split_of_year(2020) == "train"
    for y in (2008, 2009, 2016, 2017):
        assert oi.split_of_year(y) == "interspersed"
    for y in (2021, 2022, 2023, 2024):
        assert oi.split_of_year(y) == "terminal"
    assert oi.split_of_year(2025) is None
    # The three bands must partition the sixteen levels exactly once.
    seen = sum((oi.band_levels(b) for b in oi.BANDS), [])
    assert sorted(seen) == list(range(len(oi.LEVELS)))


def test_month_and_year_of_days_match_the_sampler():
    # 1982-01-01 is day 0; 2015-01-03 is day 12055; the transform is the one
    # ml/cone_sampler.py::ProfileGather.month_of uses.
    import datetime as dt
    for d in (0, 12055, 400, 15340):
        date = dt.date(1982, 1, 1) + dt.timedelta(days=d)
        assert int(oi.month_of_days(np.array([d + 0.4]))[0]) == date.month - 1
        assert int(oi.year_of_days(np.array([d + 0.4]))[0]) == date.year


@pytest.mark.filterwarnings("ignore")
def test_the_whole_file_is_quick():
    t0 = time.time()
    nb = make_field(seed=3, n=500, k=5)
    mask = np.isfinite(nb.y_t)
    oi.fit_oi(nb.a_t, nb, nb.y_t, mask, [0])
    assert time.time() - t0 < 30.0

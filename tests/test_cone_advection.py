#!/usr/bin/env python3
"""E-069c · CAN the cone codec estimate velocity from displacement AT ALL?

Five GPU seeds on the real North Atlantic tensor (E-069 and E-069b — the
cone-native codec that reads a pixel's present-day 3x3 patch plus ~706 "dot"
samples spread over the previous 30 days, then is asked to reconstruct what it
was not shown) agree on one thing: a ridge from the codec's 32-number
embedding `z` to the pixel's own ocean current is no better than the same
ridge run on the raw present-day 3x3 patch with no codec at all. On the real
ocean that comparison cannot be decisive, because the present-day patch
already carries the current through geostrophy — sea-surface height's slope IS
the surface current — so "no better than the patch" is consistent with both
"the architecture cannot see displacement" and "there was nothing left to see".

This file removes the second explanation by construction. It plants a field in
which

  * two TRACERS (`sst` and `log_mld`, the cone's family C) are smooth random
    fields ADVECTED each pentad by a known, smoothly varying velocity, by
    semi-Lagrangian back-tracing with bilinear interpolation;
  * `ssh` is an INDEPENDENT smooth random field and the wind stresses are
    white noise, so nothing outside the current channels themselves relates a
    single snapshot to the flow — no geostrophy, no thermal wind, nothing;
  * `cur_u` / `cur_v` / `cur_speed` are that velocity, and are hidden from the
    encoder whenever the probe runs.

DISPLACEMENT IS THEN THE ONLY SOURCE OF VELOCITY INFORMATION IN THE WHOLE
TENSOR, and the claim is checkable in both directions on the same field:

  (a) `raw_patch_probe` — the bar every E-069 run quotes — must read about
      zero, because a snapshot of an advected tracer shows the tracer's
      POSITION and never its motion. Measured here: R^2 -0.09. So must a ridge
      on the context token alone (season, latitude, longitude), or a velocity
      field that stood still in space would answer for the stencil: -0.03.
  (b) `flow_certificate` — an explicit two-tracer optical-flow solve that uses
      NOTHING the codec is not given (the lag-0 3x3 patch for the spatial
      gradients, the lag-1 anchor column for the time difference, 10 of the
      ~500 tokens and 2 of the 8 channels) — must read well above zero, or a
      codec that fails would only be telling us the information is absent from
      its input rather than beyond its reach. Measured: R^2 0.21 at one cell
      per pentad.

Two displacement magnitudes are run, because they test different mechanisms
and the certificate says so. At about ONE cell per pentad (~28 km / 5 days,
~0.06 m/s) the shift is inside the 3x3 patch, the linearised flow constraint
applies, and the certificate reads 0.21. At about FOUR cells per pentad
(~110 km / 5 days, ~0.26 m/s — the 0.3 m/s `ml/cone.py` builds family B's
reach from) the shift is a fifth of a tracer wavelength and leaves the patch;
the certificate falls to 0.07, and a dense 3x3 block match against the WHOLE
previous pentad — far more than the cone is given — does no better. So the
four-cell arm is asymmetric evidence by construction: a cone that SCORES there
has found something the explicit estimators could not, and a cone that does
not has told us mainly that consecutive pentads of a tracer moving at 0.3 m/s
are only weakly related by translation at all.

A third number is printed beside the certificate and is not a bar: the SAME
estimator run on the tensor before the anomaly transform, which reads 0.56
where the codec's own anomaly-space input reads 0.21. The transform subtracts
a per-pixel monthly climatology, which changes the tracer's spatial gradient
while leaving its motion alone, and it costs the local frozen-field constraint
a factor of about 2.7 in R^2. That is a property of the pipeline rather than
of this file, and it applies to the real tensor's real sea-surface temperature
in the same way.

    python3 -m pytest -q tests/test_cone_advection.py          # ~70 s
    CONE_ADVECTION_FULL=1 python3 -m pytest -q -s \
        tests/test_cone_advection.py                           # ~27 min

The long form is four training runs (two magnitudes x cone and twin) and is
deliberately not part of any default suite; `CONE_ADVECTION_STEPS=1500` gives
the same shape in a quarter of the time, with everything under-trained.

The default run builds the field, certifies both bars and exercises the whole
training-and-probe path on a few dozen steps, so nothing in the long run can
fail for a reason the short run could have caught. `CONE_ADVECTION_FULL=1`
turns on the real measurement: 5,000 steps per magnitude under the E-069b
masking plan, its L_in = 0 snapshot twin as a control, and the velocity probe,
the per-family held-out errors, the z statistics and the future-versus-
persistence diagnostic printed for both.
"""
import math
import os
import sys
import time

import numpy as np
import pytest
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ML = os.path.join(ROOT, "ml")
if ML not in sys.path:
    sys.path.insert(0, ML)

from cone import KM_PER_DEG, channel_depth_dbar                  # noqa: E402
from cone_sampler import ConeSampler                             # noqa: E402
from cone_codec import ConeMAE, default_plan                     # noqa: E402
from train_cone import (PENTAD_DAYS, PENTAD_EPOCH, SMOKE_CHANS,  # noqa: E402
                        draw_anchors, eval_loss, fold_labels,
                        raw_patch_probe, ridge_to_currents, to_torch,
                        velocity_probe, z_stats)

FULL = bool(os.environ.get("CONE_ADVECTION_FULL"))

DLAT = 0.25
LAT0 = 32.0                 # 32.0 .. 49.75 N: cos(phi) runs 0.85 .. 0.65, so
                            # the zonal cell is 20-24 km and never degenerate
LON0 = -50.0

# The model: 0.25M parameters against the real run's 7.05M, and the same shape
# `tests/test_cone_dot_path.py` certified the dot path with. At ~500 dot tokens
# per anchor it costs 110 ms a step on two CPU cores, which is what sets the
# step budget below.
SMALL = dict(d_model=64, n_heads=4, n_latents=16, n_layers=2, d_z=32,
             d_dec=64, dec_layers=2, n_fourier=6)
# `ml/train_cone.py`'s OWN default, and not a detail. Measured on this field
# under the E-069b plan: at 3e-3 the code's variance runs away to 1e4 by step
# 1,000 and every family goes back to its bar; at 1e-3 it barely moves; at
# 3e-4 the anchor family is at 0.88 of its bar and the dots at 0.86 by step
# 800 and still falling.
LR = 3e-4
# About fourteen CPU minutes per cone arm at ~170 ms a step, and 1/90th of the
# sample count a real E-069b seed sees (20,000 steps x 256) — which is why
# nothing here is a statement about what the architecture CONVERGES to, only
# about whether the displacement reaches z at all on a field where it is the
# only route. `CONE_ADVECTION_STEPS` overrides it.
STEPS = int(os.environ.get("CONE_ADVECTION_STEPS",
                           5000 if FULL else 40))
N_EVAL = 2048
BATCH = 32
# Fifteen years of pentads, which is long for a CPU test and is the length two
# separate measurements demanded. The probe's folds are CALENDAR YEARS only
# when the held-out block spans three of them (`train_cone.fold_labels`), and
# the anomaly transform's per-pixel monthly climatology gets one sample per
# month per year — the fewer years, the more of the tracer it absorbs and
# subtracts back out. Measured on this field, going from 3 years to 10 to 15
# takes the raw-patch bar from -0.59 to -0.35 to -0.09 (it should be 0, and
# what makes it negative is a ridge extrapolating a fold-mean shift) and the
# displacement certificate from -0.12 to +0.13 to +0.21.
T_BINS = 1095
HOLD_FRAC = 0.30
# The two displacements, in grid cells per pentad. 1.0 is ~28 km / 5 d
# (~0.064 m/s); 4.0 is ~111 km / 5 d (~0.26 m/s), which is the 0.3 m/s
# ml/cone.py builds family B's reach from.
MAGNITUDES = (1.0, 4.0)


# ---------------------------------------------------------------- the field --
TRACER_K = (2, 4)                # the tracer's wavenumber band, both axes


def _smooth_field(rng, H, W, n_modes=10, band=TRACER_K):
    """A smooth random field on the periodic HxW torus, unit variance.

    A sum of `n_modes` sinusoids whose wavenumbers lie in `band` on each axis,
    so wavelengths run 18 to 36 cells meridionally. The band is bounded at
    BOTH ends, and each end is load-bearing:

      * the upper bound keeps a one-cell displacement a small fraction of a
        wavelength, which is what makes the frozen-field constraint in
        `flow_certificate` a good approximation and the 3x3 patch's central
        difference a real gradient rather than noise;
      * the LOWER bound is there because of the anomaly transform. A
        domain-scale component of an advected tracer takes many years to be
        carried anywhere at one cell per pentad, so it sits nearly still at
        each pixel — and a per-pixel monthly climatology estimated over the
        training years then absorbs it and subtracts it back out. Measured
        with the band open at the bottom, the climatology's standard
        deviation was 0.66 against a tracer's 1.0, and the certificate fell
        from R^2 0.51 in raw values to 0.09 in the anomalies the codec
        actually reads. The band is what keeps the tracer's own Eulerian
        decorrelation time (about twenty pentads at one cell per pentad)
        shorter than the year the climatology bins by.
    """
    lo, hi = band
    yy, xx = np.meshgrid(np.arange(H), np.arange(W), indexing="ij")
    f = np.zeros((H, W))
    for _ in range(n_modes):
        m = int(rng.integers(lo, hi + 1)) * int(rng.choice([-1, 1]))
        k = int(rng.integers(lo, hi + 1)) * int(rng.choice([-1, 1]))
        f += np.sin(2.0 * np.pi * (m * yy / H + k * xx / W)
                    + rng.uniform(0.0, 2.0 * np.pi))
    return f / f.std()


def _bilinear_periodic(A, ys, xs):
    """`A` sampled at fractional (ys, xs) with wrap-around — the interpolation
    half of the semi-Lagrangian step. Periodic rather than clamped so the
    advected tracer has no edge where its variance decays, which would be a
    second signal the codec could read."""
    H, W = A.shape
    y0 = np.floor(ys).astype(np.int64)
    x0 = np.floor(xs).astype(np.int64)
    fy, fx = ys - y0, xs - x0
    y0 %= H
    x0 %= W
    y1, x1 = (y0 + 1) % H, (x0 + 1) % W
    return ((1 - fy) * (1 - fx) * A[y0, x0] + (1 - fy) * fx * A[y0, x1]
            + fy * (1 - fx) * A[y1, x0] + fy * fx * A[y1, x1])


def _velocity(rng, T, H, W, disp_cells, n_modes=6, rho=0.7, kmin=1, kmax=1):
    """(U, V) in CELLS PER PENTAD: a non-divergent flow whose modes wander.

    A stream function psi = sum_j [a_j(t) cos(arg_j) + b_j(t) sin(arg_j)] with
    arg_j = 2 pi (m_j y / H + k_j x / W) + phi_j, and u = -d psi / dy,
    v = +d psi / dx — so the flow has no divergence and the advected tracer
    neither piles up nor thins out. `a_j` and `b_j` are independent AR(1)
    paths with `rho` = 0.7, i.e. an e-folding time of about three pentads: the
    field is SMOOTH in space and SLOW in time, and each mode's phase drifts
    rather than marching.

    THREE PROPERTIES ARE LOAD-BEARING, and each closes a way this experiment
    could answer itself without the codec doing anything:

      * **The time dependence is a RANDOM PATH, not a clock.** The codec is
        handed the anchor's latitude, longitude and day of year in its context
        token. Any velocity that is a smooth deterministic function of (y, x,
        t) is then readable from that token alone over a held-out block short
        enough that the day of year pins the bin down — measured on the first
        version of this file, which used travelling waves of fixed period: a
        four-feature ridge on the context token recovered `cur_u` at R^2 0.62,
        and the whole comparison would have been between two ways of reading a
        calendar. With AR(1) amplitudes the mapping is a random path: it can
        be fitted inside one time fold and predicts nothing in the next, which
        is exactly what the probe's contiguous-time folds ask of it.
      * **The lowest wavenumber, because the STRAIN RATE has to stay small.**
        A flow of wavenumber k displaces neighbouring cells differently by
        about `disp_cells * 2 pi k / H` per pentad, and once that approaches 1
        the tracer is not translated between two bins, it is deformed beyond
        recognition — there is no displacement left to read, and the
        experiment would be measuring the strain rate rather than the codec.
        Measured on this field with a DENSE 3x3 block match against the whole
        previous pentad, which is a far stronger estimator than anything the
        cone has: at four cells per pentad, wavenumbers 1-2 on a 72-row grid
        recover R^2 0.00, wavenumber 1 recovers 0.68. The four-cell arm exists
        at all only because the flow is this smooth.
      * **Meridional and zonal wavenumbers matched to the grid's aspect**
        (k = round(m * W / H)), so |u| and |v| come out the same size. Drawn
        independently they do not: the same wavenumber index is 1.8x finer in
        y than in x at H = 72, W = 128, and the north component ends up a
        third of the east one — which reads later as "the codec cannot see
        cur_v" when the truth is that there was less of it to see.
    """
    yy, xx = np.meshgrid(np.arange(H), np.arange(W), indexing="ij")
    ms = rng.integers(kmin, kmax + 1, n_modes) * rng.choice([-1, 1], n_modes)
    ks = np.array([int(round(abs(m) * W / H)) for m in ms]) * \
        rng.choice([-1, 1], n_modes)
    phase = rng.uniform(0.0, 2.0 * np.pi, n_modes)
    # two independent AR(1) paths per mode: [T, n_modes] each, unit variance
    eps = rng.normal(size=(2, T, n_modes))
    ab = np.empty((2, T, n_modes))
    ab[:, 0] = eps[:, 0]
    for t in range(1, T):
        ab[:, t] = rho * ab[:, t - 1] + math.sqrt(1.0 - rho ** 2) * eps[:, t]
    U = np.zeros((T, H, W))
    V = np.zeros((T, H, W))
    for j in range(n_modes):
        arg = 2.0 * np.pi * (ms[j] * yy / H + ks[j] * xx / W) + phase[j]
        sa, ca = np.sin(arg), np.cos(arg)
        for t in range(T):
            g = ab[0, t, j] * sa - ab[1, t, j] * ca
            U[t] += (2.0 * np.pi * ms[j] / H) * g
            V[t] -= (2.0 * np.pi * ks[j] / W) * g
    s = disp_cells / math.sqrt(float(((U ** 2 + V ** 2) / 2.0).mean()))
    return U * s, V * s


def _diffuse(A, w):
    """One step of the five-point Laplacian smoother, periodic. `w` is how
    much of each cell is replaced by the mean of its four neighbours."""
    if w <= 0.0:
        return A
    nb = 0.25 * (np.roll(A, 1, 0) + np.roll(A, -1, 0)
                 + np.roll(A, 1, 1) + np.roll(A, -1, 1))
    return (1.0 - w) * A + w * nb


def advected_tensor(path, disp_cells=1.0, seed=0, T=T_BINS, H=72, W=128,
                    reinject=0.08, noise=0.01, diffuse=0.45):
    """A pentad tensor in `train_cone.smoke_tensor`'s conventions whose ONLY
    velocity information is displacement.

    [T, H, W, 8] float32 at 0.25 degrees from 32 N, five-day bins from
    1982-01-01, the eight `SMOKE_CHANS` names (which is what tells
    `ConeSampler` each channel's cone family), a NaN land block and ~1%
    scattered dropouts.

      `sst`, `log_mld`   family C: two INDEPENDENT smooth random fields, each
                         advected by the same (U, V) every pentad.
      `ssh`              family B: an independent AR(1) smooth field. It is
                         deliberately NOT geostrophically related to the flow
                         — that relation is exactly what makes the real
                         tensor's raw-patch bar high, and removing it is what
                         makes this field a clean test.
      `tau_x`, `tau_y`   family A: white noise.
      `cur_u`, `cur_v`   the planted velocity in m/s (the cell displacement
                         times that row's cell size), `cur_speed` its
                         magnitude. All three start with `cur_`, so the probe
                         and `raw_patch_probe` both exclude them.

    Three details of the advection step are not cosmetic, and each was put
    there by a measurement:

      `reinject` blends 8% of a fresh smooth field into each tracer at every
        step. It is not only there so the record is more than one initial
        field pushed around for fifteen years: it sets the tracer's Eulerian
        decorrelation time, and that time is what decides how much of the
        tracer the anomaly transform's per-pixel monthly climatology can
        absorb. At 2% the tracer at four cells per pentad barely moved at a
        pixel, the climatology took most of its variance, and the held-out
        block came back with 3.8x the training block's standard deviation —
        so every held-out error was inflated by a factor that had nothing to
        do with the model. At 8% the ratio is 1.04 at one cell per pentad and
        1.77 at four, and the certificate is the SAME 0.16 at both
        magnitudes, which is what makes the two arms comparable at all.
      the field is RESCALED to unit variance after every step. Bilinear
        interpolation is diffusive, and left alone the tracer's standard
        deviation fell by 8x over 240 steps — at which point a 2%
        reinjection of a UNIT-variance fresh field is a 20% reinjection, and
        the displacement signal drowns in it. The z-score at the end of the
        anomaly transform is global, so the decay is invisible in the
        finished tensor and shows up only as a probe that will not score.
      `diffuse` smooths each step by a five-point Laplacian. Advection by a
        spatially varying flow cascades tracer variance to scales finer than
        a cell (the mean gradient magnitude doubles within sixty steps), and
        a field whose structure is at the grid scale has no readable
        displacement. The smoother is a linear operator on the tracer, so it
        weakens the frozen-field constraint by a Laplacian term two orders
        below the gradient term at these wavelengths; measured on the raw
        values, it takes the certificate from R^2 0.17 to R^2 0.39.
    """
    rng = np.random.default_rng(seed)
    lats = LAT0 + DLAT * np.arange(H)
    lons = LON0 + DLAT * np.arange(W)
    yy, xx = np.meshgrid(np.arange(H), np.arange(W), indexing="ij")

    U, V = _velocity(rng, T, H, W, disp_cells)

    tracers = [_smooth_field(rng, H, W) for _ in range(2)]
    TR = np.empty((2, T, H, W), np.float64)
    for t in range(T):
        if t:
            for i in range(2):
                back = _diffuse(_bilinear_periodic(tracers[i], yy - V[t],
                                                   xx - U[t]), diffuse)
                back = (back - back.mean()) / back.std()
                mix = (math.sqrt(1.0 - reinject ** 2) * back
                       + reinject * _smooth_field(rng, H, W))
                tracers[i] = (mix - mix.mean()) / mix.std()
        for i in range(2):
            TR[i, t] = tracers[i]
    # THE REFUSAL. Bilinear interpolation under a spatially varying
    # displacement is not exactly mean-preserving, so each step shifts the
    # field's domain mean a little; the rescaling then multiplies that shift by
    # 1/std, and where the strain is strong enough for std to fall much below
    # one the mean is amplified every step and grows exponentially. Measured
    # before the centring above was added: at four cells per pentad the
    # tracer's per-bin mean reached -133 while its standard deviation stayed
    # exactly 1.0, so nothing about the field LOOKED wrong — and the anomaly
    # transform then divided the held-out block by a training standard
    # deviation seven times too small and every held-out error came back in the
    # hundreds. A per-timestep constant carries no velocity, so this was pure
    # added noise in the one term the flow constraint reads (the time
    # difference), and the run it produced was junk that ran to completion.
    m = np.abs(TR.mean(axis=(2, 3))).max()
    assert m < 0.05, (
        f"the advected tracer's domain mean reached {m:.3f} against a "
        f"standard deviation of 1 — the semi-Lagrangian step is amplifying a "
        f"uniform offset instead of moving the field.")

    ssh, rho = _smooth_field(rng, H, W), 0.8
    SSH = np.empty((T, H, W), np.float64)
    for t in range(T):
        if t:
            ssh = rho * ssh + math.sqrt(1.0 - rho ** 2) * \
                _smooth_field(rng, H, W)
        SSH[t] = ssh

    # cells -> m/s. A cell is KM_PER_DEG*DLAT km north-south everywhere and
    # that times cos(phi) east-west, which is the same rule ml/cone.py uses
    # for the dots' ground offsets.
    cell_km = KM_PER_DEG * DLAT
    to_ms = 1000.0 / (PENTAD_DAYS * 86400.0)
    cu = U * (cell_km * np.cos(np.radians(lats))[None, :, None]) * to_ms
    cv = V * cell_km * to_ms

    C = len(SMOKE_CHANS)
    X = np.empty((T, H, W, C), np.float32)
    at = {n: i for i, n in enumerate(SMOKE_CHANS)}

    def nz(scale):
        return scale * rng.normal(size=(T, H, W))

    X[..., at["sst"]] = TR[0] + nz(noise)
    X[..., at["log_mld"]] = TR[1] + nz(noise)
    X[..., at["ssh"]] = SSH + nz(noise)
    X[..., at["tau_x"]] = nz(1.0)
    X[..., at["tau_y"]] = nz(1.0)
    X[..., at["cur_u"]] = cu + nz(noise * 0.05)
    X[..., at["cur_v"]] = cv + nz(noise * 0.05)
    X[..., at["cur_speed"]] = np.hypot(cu, cv) + nz(noise * 0.05)
    X[:, :3, :3, :] = np.nan                                   # land
    X[rng.random(X.shape) < 0.01] = np.nan                     # dropouts

    days = PENTAD_EPOCH + (PENTAD_DAYS * np.arange(T)).astype("timedelta64[D]")
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    np.savez_compressed(path, X=X, months=np.array([str(d) for d in days]),
                        lats=lats, lons=lons, chan=np.array(SMOKE_CHANS))
    return path


_BUILT = {}


def built_tensor(disp_cells, T=T_BINS, seed=0):
    """`advected_tensor` memoised on (displacement, length, seed).

    Building fifteen years of advection is ~26 s, and the two default tests want
    the SAME field — one certifies it, the other trains on it. A per-test
    `tmp_path` would build it twice and make the default run miss its
    two-minute budget for no scientific reason.
    """
    key = (float(disp_cells), int(T), int(seed))
    hit = _BUILT.get(key)
    if hit is None:
        import tempfile
        d = tempfile.mkdtemp(prefix="cone_adv_")
        hit = advected_tensor(os.path.join(d, f"adv{disp_cells:g}.npz"),
                              disp_cells=disp_cells, seed=seed, T=T)
        _BUILT[key] = hit
    return hit


def anomaly_space(path, hold_frac=HOLD_FRAC):
    """THE ONE anomaly transform (`trainprobe.anomaly_transform`, the function
    `train_cone.load_data` calls), applied exactly as the trainer applies it:
    every dynamic channel becomes a departure from its own TRAIN-years monthly
    climatology and is then z-scored on train data.

    The hold-out is a CONTIGUOUS TERMINAL BLOCK (the last `hold_frac` of the
    record), which is the frozen protocol's shape rather than the interspersed
    development split — and it matters here for a reason beyond protocol: a
    climatology estimated on the training years is subtracted from the
    held-out years too, so the velocity signal in the eval block is never
    partly removed by a climatology fitted to it.

    Refuses if any channel comes back STATIC. `anomaly_transform` leaves a
    channel with no temporal variance in its spatial mean untouched — neither
    de-seasonalised nor z-scored — and a non-divergent velocity has a spatial
    mean of exactly zero at every bin, so `cur_u` is one small change away
    from silently skipping the transform that every other channel gets.
    """
    from trainprobe import anomaly_transform
    d = np.load(path, allow_pickle=False)
    X = d["X"].copy()
    months = [str(m) for m in d["months"]]
    T = X.shape[0]
    hold = np.arange(T) >= int(round((1.0 - hold_frac) * T))
    assert hold.any() and (~hold).any()
    moy = np.array([int(m[5:7]) - 1 for m in months])
    X, dynamic = anomaly_transform(X, moy, hold, np.zeros(X.shape[2], bool))
    assert len(dynamic) == X.shape[3], (
        f"anomaly_transform treated {X.shape[3] - len(dynamic)} of "
        f"{X.shape[3]} channels as STATIC and passed them through raw: "
        f"{sorted(set(range(X.shape[3])) - set(dynamic))}. Every channel here "
        f"varies in time, so a static verdict means the tensor is wrong.")
    return dict(X=X, OBS=np.isfinite(X), lats=np.asarray(d["lats"]),
                lons=np.asarray(d["lons"]),
                chan=[str(c) for c in d["chan"]], hold=hold, months=months)


# ------------------------------------------------------------- anchor pools --
def anchor_grid(sampler):
    """(rows, cols) whose whole cone lies inside the grid, from the sampler's
    OWN dot table rather than from a repeat of ml/cone.py's arithmetic.

    Two passes, because the zonal reach in cells depends on the row (a cell is
    27.83*cos(phi) km east-west): take the meridional margin over every row,
    then the zonal margin over only the rows that survive it. Anything looser
    would train on anchors whose cone hangs off the edge, and "the codec never
    saw the dots that carried the displacement" is not a result about the
    codec.
    """
    H, W = sampler.H, sampler.W
    dy_max = max(int(np.abs(sampler.row(y)["dy"]).max()) for y in range(H))
    rows = np.arange(dy_max, H - dy_max)
    if not len(rows):
        raise ValueError(f"the grid is {H} rows and the cone reaches "
                         f"{dy_max} — no anchor's cone fits")
    dx_max = max(int(np.abs(sampler.row(int(y))["dx"]).max()) for y in rows)
    cols = np.arange(dx_max, W - dx_max)
    if not len(cols):
        raise ValueError(f"the grid is {W} columns and the cone reaches "
                         f"{dx_max} — no anchor's cone fits")
    return rows, cols


def split_bins(sampler, hold, L_in):
    """(train bins, eval bins) under the cone's own window rule: every bin an
    anchor touches — L_in back and two forward — must lie on one side."""
    T = sampler.T
    fwd = max(sampler.future_lags)
    ok = lambda t, side: (t - L_in >= 0 and t + fwd < T           # noqa: E731
                          and bool(side[t - L_in:t + fwd + 1].all()))
    train = np.array([t for t in range(T) if ok(t, ~hold)], np.int64)
    ev = np.array([t for t in range(T) if ok(t, hold)], np.int64)
    return train, ev


# ------------------------------------------------------------------- probes --
def flow_certificate(sampler, chan, anchors, months, batch=64, ridge=2e-3):
    """THE CONTROL THAT THE CONTROL CONTROLS SOMETHING: an explicit
    displacement estimate from exactly the tokens the codec is handed.

    For each anchor and each of the two advected tracers the frozen-field
    constraint says
            I(t, p) = I(t - 1, p - d)  =>  I_t + d_x I_x + d_y I_y ~ 0,
    where the two spatial derivatives are central differences on the LAG-0 3x3
    PATCH and the time difference is the patch centre minus the LAG-1 ANCHOR
    COLUMN — one of the cone's own dots, present at every lag for every
    channel (`ml/cone.py::inner_dots` emits (lag, 0, 0) first). Two tracers
    give two equations in the two unknowns; they are solved per anchor with a
    small ridge, and the resulting (d_x, d_y) pair is scored against the true
    `cur_u` / `cur_v` by `ridge_to_currents` — the SAME scorer, the same folds
    and the same anchors as every other bar in this file, because a comparison
    between probes that differ in their scorer is not a comparison.

    Nothing here is available to this estimator that is not available to the
    codec, and the estimator uses a vanishing fraction of what the codec has
    (2 of 8 channels, 10 of ~500 tokens). So a codec that scores BELOW this
    number has lost information that was sitting in its own input.

    A TRACER WHOSE CELLS ARE NOT ALL OBSERVED CONTRIBUTES NOTHING. The tensor
    carries ~1% scattered dropouts, and `ConeSampler` reports an absent value
    as a zero with its observed flag cleared — which is right for the codec
    (that is what `miss_tok` is for) and fatal for a two-equation solve, where
    one corrupted row sends the estimate to a wild number and a handful of
    wild numbers destroy a correlation. Measured: including them takes the
    certificate from R^2 0.44 to 0.06, which would have read exactly like a
    field that carries no displacement.

    The estimate is linear in the displacement and its error is quadratic, so
    it degrades as 1/displacement^2 — it is a floor on what the tokens carry,
    never a ceiling.
    """
    ci = [chan.index("sst"), chan.index("log_mld")]
    need = [1, 3, 4, 5, 7]           # the patch cells the two differences use
    feats, tgs, obs = [], [], []
    for i in range(0, len(anchors), batch):
        a = anchors[i:i + batch]
        s = sampler.sample(a)
        pv = np.asarray(s["patch_vals"], np.float64)               # [B, C, 9]
        po = np.asarray(s["patch_obs"], bool)
        B = len(pv)
        # the lag-1 anchor column, found in the row's own dot table
        col = np.zeros((B, 2))
        colo = np.zeros((B, 2), bool)
        for b in range(B):
            R = sampler.row(int(a[b, 1]))
            for j, c in enumerate(ci):
                k = np.flatnonzero((R["lag"] == 1) & (R["dy"] == 0)
                                   & (R["dx"] == 0) & (R["chan"] == c))
                assert len(k) == 1, (
                    f"channel {chan[c]} has {len(k)} lag-1 anchor-column dots "
                    f"— ml/cone.py::inner_dots emits exactly one")
                col[b, j] = s["vals"][b, k[0]]
                colo[b, j] = s["obs"][b, k[0]] and s["valid"][b, k[0]]
        A = np.zeros((B, 2, 2))
        rhs = np.zeros((B, 2))
        for j, c in enumerate(ci):
            p = pv[:, c, :]
            usable = po[:, c, need].all(1) & colo[:, j]
            A[:, j, 0] = np.where(usable, (p[:, 5] - p[:, 3]) / 2.0, 0.0)
            A[:, j, 1] = np.where(usable, (p[:, 7] - p[:, 1]) / 2.0, 0.0)
            rhs[:, j] = np.where(usable, -(p[:, 4] - col[:, j]), 0.0)
        AtA = np.einsum("bij,bik->bjk", A, A) + ridge * np.eye(2)[None]
        Atb = np.einsum("bij,bi->bj", A, rhs)
        feats.append(np.linalg.solve(AtA, Atb[..., None])[..., 0])
        tgs.append(np.asarray(s["patch_vals"], np.float64)[..., 4])
        obs.append(np.asarray(s["patch_obs"], bool)[..., 4])
    groups, how = fold_labels(anchors, months)
    out = {"folds": how, "n_anchors": int(len(anchors)), "n_features": 2,
           "note": "two-tracer optical flow from the lag-0 3x3 patch and the "
                   "lag-1 anchor column — no codec, no training"}
    out.update(ridge_to_currents(np.concatenate(feats),
                                 np.concatenate(tgs),
                                 np.concatenate(obs), chan, groups))
    return out


def raw_space_certificate(path, anchors, hold_frac=HOLD_FRAC):
    """The SAME certificate, run on the tensor BEFORE the anomaly transform.

    It is not a bar for the codec — the codec never sees these numbers — and
    it is not part of the verdict. It is here because the difference between
    the two is itself a measurement, and a surprising one: the transform's
    per-pixel monthly climatology is a static field subtracted from every bin,
    so the anomaly's spatial gradient is the tracer's gradient MINUS the
    climatology's, while the tracer's motion is unchanged. The frozen-field
    constraint therefore holds for the raw tracer and only approximately for
    the anomaly, and how much is lost depends on how much of the tracer the
    climatology managed to absorb. Reporting both says whether a low
    anomaly-space number means "the field has little displacement" or "the
    transform removed most of it" — a distinction that applies to the REAL
    tensor and its real sea-surface temperature exactly as it does here.
    """
    d = np.load(path, allow_pickle=False)
    X = d["X"]
    chan = [str(c) for c in d["chan"]]
    months = [str(m) for m in d["months"]]
    sam = ConeSampler(X, np.isfinite(X), np.asarray(d["lats"]),
                      np.asarray(d["lons"]), chan, L_in=6, future_lags=(1, 2))
    return flow_certificate(sam, chan, anchors, months)


def context_bar(sampler, chan, anchors, months, batch=64):
    """The POSITION-AND-SEASON bar: a ridge from the four numbers of the
    codec's context token (sin and cos of the day of year, latitude,
    longitude) to the anchor's own current.

    It exists because the field could answer the experiment for the wrong
    reason. A velocity pattern that stood still in space would be a function
    of position, the encoder is handed position, and a high probe R^2 would
    then say nothing about displacement. `_velocity` makes the pattern travel
    so that it is not; this measures whether that worked, and it must read
    about zero. The L_in = 0 twin is the same check with a whole encoder
    behind it.
    """
    F, tgs, obs = [], [], []
    for i in range(0, len(anchors), batch):
        s = sampler.sample(anchors[i:i + batch])
        F.append(np.asarray(s["ctx"], np.float64))
        tgs.append(np.asarray(s["patch_vals"], np.float64)[..., 4])
        obs.append(np.asarray(s["patch_obs"], bool)[..., 4])
    groups, how = fold_labels(anchors, months)
    out = {"folds": how, "n_anchors": int(len(anchors)), "n_features": 4}
    out.update(ridge_to_currents(np.concatenate(F), np.concatenate(tgs),
                                 np.concatenate(obs), chan, groups))
    return out


def future_persistence(model, sampler, anchors, plan, chan_depth, batch=32,
                       seed=12345):
    """THE DIAGNOSTIC THAT SEPARATES THE TWO FAILURES: does the codec use the
    displacement ANYWHERE, even if it does not put it in z?

    On the same held-out future targets, with the same weights `eval_loss`
    scores, two references:

      `persistence` predicts t+1 and t+2 with the anchor's own value at
        lag 0 — "nothing moved". On an advected tracer this is exactly the
        prediction that ignores the flow, and the amount by which it can be
        beaten IS the displacement.
      `msebar` predicts zero, the climatological mean in anomaly space.

    A codec whose future error is below its bar but not below persistence has
    learnt the field's persistence and nothing about its motion; one below
    persistence has used the displacement somewhere, whatever its z contains.
    Reported per channel as well as in total, because the total is diluted by
    `tau_x` / `tau_y` (white noise, where persistence is twice the bar) and by
    `ssh` (an AR(1), where persistence is nearly optimal) — the two tracers
    are where the question lives.
    """
    from train_cone import eval_generator
    g = eval_generator("cpu", seed)
    p = dict(plan)
    p["generator"] = g
    C = int(model.n_chan)
    sq = np.zeros(C)
    bar = np.zeros(C)
    wsum = np.zeros(C)
    with torch.no_grad():
        for i in range(0, len(anchors), batch):
            b = to_torch(sampler.sample(anchors[i:i + batch]), chan_depth,
                         "cpu")
            chan_mask, dot_mask = model._masks(b, p)
            idx, sel = model.draw_dot_queries(b, p, dot_mask)
            _, _, _, _, _, tgt, w = model._query_sets(b, p, dot_mask,
                                                      (idx, sel), chan_mask)
            lo, hi = model.query_family_spans(b, p, w.shape[1])["future"]
            if hi <= lo:
                continue
            F = b["fut_vals"].shape[-1]
            # `_query_sets` reshapes [B, C, F] -> [B, C*F], so column c*F + f
            # is channel c at future lag f. Persistence is the anchor's own
            # lag-0 value, broadcast over the future lags, and zero where the
            # anchor itself was not observed (there is nothing to persist).
            centre = (b["patch_vals"][..., 4] * b["patch_obs"][..., 4])
            pred = centre[:, :, None].expand(-1, -1, F).reshape(len(centre),
                                                                C * F)
            tg, ww = tgt[:, lo:hi], w[:, lo:hi]
            e = ((pred - tg) ** 2 * ww).reshape(-1, C, F).sum((0, 2))
            z = ((tg ** 2) * ww).reshape(-1, C, F).sum((0, 2))
            n = ww.reshape(-1, C, F).sum((0, 2))
            sq += e.numpy()
            bar += z.numpy()
            wsum += n.numpy()
    den = np.where(wsum > 0, wsum, 1.0)
    return {"persistence_by_chan": (sq / den).tolist(),
            "msebar_by_chan": (bar / den).tolist(),
            "wsum_by_chan": wsum.tolist(),
            "persistence": float(sq.sum() / max(wsum.sum(), 1e-9)),
            "msebar": float(bar.sum() / max(wsum.sum(), 1e-9))}


# ----------------------------------------------------------------- training --
def train_arm(D, L_in, anchors_train_bins, rows, cols, ev_anchors, steps,
              tag, aux_latent_w=0.25, seed=0, lr=LR, collapse_at=None):
    """Train one arm and return everything it measured.

    `D` is `anomaly_space`'s dict. The masking plan is E-069b's, verbatim:
    `chan_drop_scope="lag0"` (a dropped channel is hidden at lag 0 but its
    dots stay visible), `lag_band_p=0.5`, `sector_p=0.5`,
    `anchor_hidden_only=True` (the anchor family is scored only on the
    channels this batch element actually dropped, so no target is a copy of a
    visible input), `cur_*` dropped at 0.5 and everything else at 0.3.

    `collapse_at`, when set, is the step at which the code's variance is read;
    a value under 1e-3 means the 32-number bottleneck has collapsed onto the
    mean, and the caller retries at a higher `aux_latent_w`. This is the
    failure `tests/test_cone_dot_path.py` measured at the shipped 0.25 —
    variance ~1e-5 for over 1,500 steps — and it is a training-dynamics fact,
    not a property of the stencil under test, so it is detected and reported
    rather than trained through.
    """
    chan = D["chan"]
    sam = ConeSampler(D["X"], D["OBS"], D["lats"], D["lons"], chan,
                      L_in=L_in, future_lags=(1, 2))
    cd = torch.as_tensor([channel_depth_dbar(n) for n in chan],
                         dtype=torch.float32)
    plan = default_plan(chan, cur_drop=0.5, other_drop=0.3, lag_band_p=0.5,
                        sector_p=0.5, chan_drop_scope="lag0",
                        anchor_hidden_only=True, n_dot_queries=128,
                        aux_latent_w=aux_latent_w)
    torch.manual_seed(seed)
    model = ConeMAE(len(chan), **SMALL)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps)
    rng = np.random.default_rng(seed)

    t0 = time.time()
    hist = []
    for s in range(1, steps + 1):
        a = draw_anchors(rng, anchors_train_bins, rows, cols, BATCH)
        b = to_torch(sam.sample(a), cd, "cpu")
        out = model(b, plan)
        opt.zero_grad(set_to_none=True)
        out["loss"].backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
        lv = float(out["loss"].detach())
        assert np.isfinite(lv), f"[{tag}] NaN loss at step {s}"
        hist.append(lv)
        if collapse_at and s == collapse_at:
            with torch.no_grad():
                bb = to_torch(sam.sample(ev_anchors[:64]), cd, "cpu")
                bb["chan_mask"] = torch.zeros(len(bb["ctx"]), len(chan),
                                              dtype=torch.bool)
                bb["dot_mask"] = torch.zeros_like(bb["obs"])
                zv = float(model.encode(bb)[0].var(0).sum())
            if zv < 1e-3:
                return {"collapsed": True, "z_var_at_check": zv,
                        "aux_latent_w": aux_latent_w, "steps_done": s}

    secs = time.time() - t0
    nll, mse, tgt, fam = eval_loss(model, sam, ev_anchors, plan, cd, "cpu",
                                   BATCH)
    vp = velocity_probe(model, sam, chan, ev_anchors, D["months"], cd, "cpu",
                        batch=64)
    fp = future_persistence(model, sam, ev_anchors, plan, cd, BATCH)
    return {"collapsed": False, "tag": tag, "L_in": L_in, "steps": steps,
            "secs": secs, "aux_latent_w": aux_latent_w,
            "params": int(model.param_count()), "n_dots": int(sam.n_dots(
                int(rows[len(rows) // 2]))),
            # a MEAN over the first and last fiftieth of the run, never two
            # single steps: one training step's loss on 32 anchors is noisy
            # enough that "did it train" would be a coin flip on a short run
            "loss_first": float(np.mean(hist[:max(len(hist) // 50, 1)])),
            "loss_last": float(np.mean(hist[-max(len(hist) // 50, 1):])),
            "held_out_nll": nll, "held_out_mse": mse, "families": fam,
            "probe": vp, "future": fp, "chan": chan}


# ------------------------------------------------------------------ printing --
def _r2(d):
    v = (d or {}).get("r2")
    return "   n/a" if v is None else f"{float(v):+.3f}"


def _report(disp, bars, cone, twin):
    chan = cone["chan"]
    hid = cone["probe"]["variants"]["hidden"]
    vis = cone["probe"]["variants"]["visible"]
    thid = twin["probe"]["variants"]["hidden"]
    zs = cone["probe"]["z_stats"]
    tzs = twin["probe"]["z_stats"]
    print(f"\n================ {disp:.1f} cell/pentad "
          f"(~{disp * KM_PER_DEG * DLAT:.0f} km per 5 d) ================",
          flush=True)
    print(f"  cone {cone['steps']} steps in {cone['secs']:.0f}s "
          f"({cone['params']} params, {cone['n_dots']} dot tokens, "
          f"aux {cone['aux_latent_w']}) · twin {twin['secs']:.0f}s")
    print("  velocity R2 (hidden cur_*)      cur_u    cur_v")
    print(f"    cone codec (z, L_in=6)      {_r2(hid.get('cur_u'))}   "
          f"{_r2(hid.get('cur_v'))}")
    print(f"    snapshot twin (z, L_in=0)   {_r2(thid.get('cur_u'))}   "
          f"{_r2(thid.get('cur_v'))}")
    print(f"    raw lag-0 3x3 patch (bar)   {_r2(bars['patch'].get('cur_u'))}"
          f"   {_r2(bars['patch'].get('cur_v'))}")
    print(f"    context token only (bar)    {_r2(bars['ctx'].get('cur_u'))}"
          f"   {_r2(bars['ctx'].get('cur_v'))}")
    print(f"    two-tracer optical flow     {_r2(bars['flow'].get('cur_u'))}"
          f"   {_r2(bars['flow'].get('cur_v'))}   <- the certificate")
    print(f"      the same, on RAW values   {_r2(bars['flow_raw'].get('cur_u'))}"
          f"   {_r2(bars['flow_raw'].get('cur_v'))}   (not a bar — how much "
          f"the anomaly transform costs)")
    print(f"    cone, cur_* VISIBLE         {_r2(vis.get('cur_u'))}   "
          f"{_r2(vis.get('cur_v'))}")
    print(f"  z: cone eff-rank {zs['eff_rank']:.2f}/{zs['d_z']} "
          f"var {zs['var_total']:.3g} pair-cos {zs['mean_pair_cos']:+.3f} · "
          f"twin eff-rank {tzs['eff_rank']:.2f} var {tzs['var_total']:.3g}")
    print("  held-out MSE by family (mse / msebar):")
    for name in ("anchor", "future", "dots"):
        c, t = cone["families"][name], twin["families"][name]
        cs = (f"{c['mse']:.3f}/{c['msebar']:.3f}" if c["wsum"] > 0
              else "  --  (no targets)")
        ts = (f"{t['mse']:.3f}/{t['msebar']:.3f}" if t["wsum"] > 0
              else "  --  (no targets)")
        print(f"    {name:<7} cone {cs}   twin {ts}")
    print("  future family, per channel — model / persistence / bar:")
    cf, tf = cone["families"]["future"], twin["families"]["future"]
    for c, name in enumerate(chan):
        if cf["wsum_by_chan"][c] <= 0:
            continue
        print(f"    {name:<10} cone {cf['mse_by_chan'][c]:.3f}"
              f"  persist {cone['future']['persistence_by_chan'][c]:.3f}"
              f"  bar {cf['msebar_by_chan'][c]:.3f}"
              f"   | twin {tf['mse_by_chan'][c]:.3f}"
              f"  persist {twin['future']['persistence_by_chan'][c]:.3f}")
    print("  anchor family, the two current components — model / bar:")
    for name in ("cur_u", "cur_v"):
        c = chan.index(name)
        print(f"    {name:<10} cone "
              f"{cone['families']['anchor']['mse_by_chan'][c]:.3f}/"
              f"{cone['families']['anchor']['msebar_by_chan'][c]:.3f}"
              f"   twin {twin['families']['anchor']['mse_by_chan'][c]:.3f}/"
              f"{twin['families']['anchor']['msebar_by_chan'][c]:.3f}")


# -------------------------------------------------------------- the harness --
def measure(disp_cells, steps=STEPS, T=T_BINS, seed=0,
            n_eval=N_EVAL):
    """Build the field at one displacement magnitude, certify it, train both
    arms and return every number the report prints."""
    path = built_tensor(disp_cells, T=T, seed=seed)
    D = anomaly_space(path)
    sam = ConeSampler(D["X"], D["OBS"], D["lats"], D["lons"], D["chan"],
                      L_in=6, future_lags=(1, 2))
    rows, cols = anchor_grid(sam)
    train_t, ev_t = split_bins(sam, D["hold"], 6)
    assert len(train_t) > 20 and len(ev_t) > 8, (train_t.shape, ev_t.shape)
    ev = draw_anchors(np.random.default_rng(991), ev_t, rows, cols, n_eval)

    bars = {"patch": raw_patch_probe(sam, D["chan"], ev, D["months"]),
            "flow": flow_certificate(sam, D["chan"], ev, D["months"]),
            "ctx": context_bar(sam, D["chan"], ev, D["months"]),
            "flow_raw": raw_space_certificate(path, ev)}

    arms = {}
    for tag, L_in in (("cone", 6), ("twin", 0)):
        r = train_arm(D, L_in, train_t, rows, cols, ev, steps, tag,
                      aux_latent_w=0.25, collapse_at=max(steps // 5, 1))
        if r.get("collapsed"):
            print(f"  [{tag}] z variance {r['z_var_at_check']:.2e} at step "
                  f"{r['steps_done']} — the code collapsed onto the mean at "
                  f"aux_latent_w 0.25; retrying at 1.0", flush=True)
            r = train_arm(D, L_in, train_t, rows, cols, ev, steps, tag,
                          aux_latent_w=1.0, collapse_at=None)
            r["retried_from_collapse"] = True
        arms[tag] = r
    return {"disp": disp_cells, "bars": bars, "cone": arms["cone"],
            "twin": arms["twin"], "n_eval": int(n_eval),
            "n_train_bins": int(len(train_t)), "n_eval_bins": int(len(ev_t)),
            "rows": (int(rows[0]), int(rows[-1])),
            "cols": (int(cols[0]), int(cols[-1]))}


# ----------------------------------------------------------------- the tests --
def test_field_hides_velocity_from_snapshots_and_shows_it_to_the_cone():
    """THE FIELD IS THE EXPERIMENT, so it is certified before anything trains.

    Three numbers on the same held-out anchors and the same folds:

      · `raw_patch_probe` — the bar E-069 quotes — must read about ZERO. That
        is the whole point of an independent `ssh`: on the real tensor this bar
        is 0.55-0.70 because sea-surface height's slope is the current, and a
        codec cannot be shown to have failed at reading displacement while a
        snapshot answers the question by geostrophy.
      · the context token (season, latitude, longitude) must read about ZERO
        too, or a travelling velocity pattern has accidentally been left
        standing and position would answer for the stencil.
      · the two-tracer optical flow, which uses only the lag-0 3x3 patch and
        the lag-1 anchor column, must read WELL ABOVE zero — the displacement
        is genuinely in the tokens the codec is handed.
    """
    path = built_tensor(1.0)
    D = anomaly_space(path)
    sam = ConeSampler(D["X"], D["OBS"], D["lats"], D["lons"], D["chan"],
                      L_in=6, future_lags=(1, 2))
    rows, cols = anchor_grid(sam)
    train_t, ev_t = split_bins(sam, D["hold"], 6)
    ev = draw_anchors(np.random.default_rng(991), ev_t, rows, cols, N_EVAL)

    # every dot of every eval anchor exists — the cone fits, with margin
    s = sam.sample(ev[:64])
    assert s["valid"].all(), (
        f"{100 * (1 - s['valid'].mean()):.1f}% of the cone's dots fall off "
        f"the grid at these anchors; `anchor_grid` is supposed to make that "
        f"impossible")

    patch = raw_patch_probe(sam, D["chan"], ev, D["months"])
    ctx = context_bar(sam, D["chan"], ev, D["months"])
    flow = flow_certificate(sam, D["chan"], ev, D["months"])
    raw = raw_space_certificate(path, ev)
    print(f"\n[field] {sam.n_dots(int(rows[0])) + len(D['chan'])} tokens/anchor "
          f"· raw patch {_r2(patch['cur_u'])}/{_r2(patch['cur_v'])} "
          f"· context {_r2(ctx['cur_u'])}/{_r2(ctx['cur_v'])} "
          f"· optical flow {_r2(flow['cur_u'])}/{_r2(flow['cur_v'])} "
          f"(raw values {_r2(raw['cur_u'])}/{_r2(raw['cur_v'])}) "
          f"({patch['folds']} folds, n={patch['n_anchors']})", flush=True)

    for name, probe in (("the raw lag-0 3x3 patch", patch),
                        ("the context token", ctx)):
        for comp in ("cur_u", "cur_v"):
            r2 = probe[comp]["r2"]
            assert r2 < 0.15, (
                f"{name} recovers {comp} at R^2 {r2:+.3f} on a field where it "
                f"was planted to carry nothing about the flow. The velocity "
                f"probe's bar would then not be a bar and this test could not "
                f"tell a working codec from a leaking field.")
    for comp in ("cur_u", "cur_v"):
        assert raw[comp]["r2"] > 0.30, (
            f"the two-tracer optical flow recovers {comp} at only R^2 "
            f"{raw[comp]['r2']:+.3f} from the RAW tracer values. The field "
            f"itself does not carry the displacement, so nothing downstream "
            f"can be asked about it.")
        r2 = flow[comp]["r2"]
        assert r2 > 0.08, (
            f"the two-tracer optical flow recovers {comp} at only R^2 "
            f"{r2:+.3f} from the lag-0 patch and the lag-1 anchor column in "
            f"ANOMALY space, against {raw[comp]['r2']:+.3f} on the raw "
            f"values. The displacement is then not readable from the codec's "
            f"OWN input, so a codec that failed here would be telling us "
            f"nothing.")


def test_the_whole_measurement_runs_on_a_few_dozen_steps():
    """The long run's code path, exercised at 40 steps — ml/CLAUDE.md 4.8.

    Nothing here is a scientific claim: forty steps decide nothing about a
    codec. What it asserts is that every piece the eight-minute run depends on
    executes and returns finite numbers on this tensor — both arms train, the
    L_in = 0 twin builds with no dots at all, the velocity probe scores, the
    per-family and per-channel held-out errors come back, the z statistics are
    real numbers and the persistence reference lands on the same weights the
    future family is scored on.
    """
    r = measure(1.0, steps=40, n_eval=384)
    print(f"\n[smoke] cone {r['cone']['secs']:.0f}s "
          f"loss {r['cone']['loss_first']:.3f} -> {r['cone']['loss_last']:.3f}"
          f" · twin {r['twin']['secs']:.0f}s "
          f"loss {r['twin']['loss_first']:.3f} -> "
          f"{r['twin']['loss_last']:.3f}", flush=True)
    assert r["cone"]["n_dots"] > 400 and r["twin"]["L_in"] == 0
    for arm in ("cone", "twin"):
        a = r[arm]
        assert np.isfinite(a["held_out_nll"]) and np.isfinite(a["held_out_mse"])
        for name, f in a["families"].items():
            for k in ("nll", "mse", "msebar", "wsum"):
                assert np.isfinite(f[k]), (arm, name, k)
            assert all(np.isfinite(v) for v in f["mse_by_chan"])
        assert a["families"]["anchor"]["wsum"] > 0
        assert a["families"]["future"]["wsum"] > 0
        # the twin has no dots at all; the cone must have scored some
        assert (a["families"]["dots"]["wsum"] > 0) == (a["L_in"] > 0)
        zs = a["probe"]["z_stats"]
        assert np.isfinite(zs["eff_rank"]) and np.isfinite(zs["var_total"])
        fp = a["future"]
        assert np.isfinite(fp["persistence"]) and fp["persistence"] > 0
        # the persistence reference is scored on EXACTLY the future family's
        # weights, so its per-channel bar must equal eval_loss's
        for c in range(len(a["chan"])):
            w = a["families"]["future"]["wsum_by_chan"][c]
            if w <= 0:
                continue
            assert abs(fp["msebar_by_chan"][c]
                       - a["families"]["future"]["msebar_by_chan"][c]) < 1e-3, (
                f"{arm}/{a['chan'][c]}: the persistence pass and eval_loss "
                f"disagree about the future family's predict-zero bar, so "
                f"they are not scoring the same targets")


@pytest.mark.skipif(not FULL, reason="set CONE_ADVECTION_FULL=1")
def test_cone_codec_on_displacement_only_velocity():
    """THE MEASUREMENT. Both magnitudes, both arms, every number printed.

    The assertions are deliberately only the ones that are robust — that the
    field is still the field it was certified to be, that both arms actually
    trained, and that the persistence reference is commensurable with the
    family it is compared against. The velocity R^2 itself is the QUANTITY
    UNDER TEST and is reported, not asserted: a threshold on it would be this
    file deciding the experiment's answer in advance.
    """
    res = {}
    for disp in MAGNITUDES:
        res[disp] = measure(disp)
        _report(disp, res[disp]["bars"], res[disp]["cone"], res[disp]["twin"])

    print("\n---------------- summary: hidden-velocity R2 ----------------")
    print("  displacement    cone u/v        twin u/v        patch bar u/v"
          "    flow cert u/v")
    for disp, r in res.items():
        h = r["cone"]["probe"]["variants"]["hidden"]
        t = r["twin"]["probe"]["variants"]["hidden"]
        b, f = r["bars"]["patch"], r["bars"]["flow"]
        print(f"  {disp:>4.1f} cell/pd   {_r2(h.get('cur_u'))}/"
              f"{_r2(h.get('cur_v'))}  {_r2(t.get('cur_u'))}/"
              f"{_r2(t.get('cur_v'))}   {_r2(b.get('cur_u'))}/"
              f"{_r2(b.get('cur_v'))}   {_r2(f.get('cur_u'))}/"
              f"{_r2(f.get('cur_v'))}")

    for disp, r in res.items():
        for comp in ("cur_u", "cur_v"):
            assert r["bars"]["patch"][comp]["r2"] < 0.15, (disp, comp)
            assert r["bars"]["ctx"][comp]["r2"] < 0.15, (disp, comp)
        assert r["bars"]["flow_raw"]["cur_u"]["r2"] > 0.0
        for arm in ("cone", "twin"):
            a = r[arm]
            assert a["loss_last"] < a["loss_first"], (
                f"{arm} at {disp} cells/pentad did not train at all: loss "
                f"{a['loss_first']:.3f} -> {a['loss_last']:.3f}")
            assert a["families"]["anchor"]["wsum"] > 0
            assert np.isfinite(a["probe"]["z_stats"]["eff_rank"])
            for c in range(len(a["chan"])):
                if a["families"]["future"]["wsum_by_chan"][c] <= 0:
                    continue
                assert abs(a["future"]["msebar_by_chan"][c]
                           - a["families"]["future"]["msebar_by_chan"][c]
                           ) < 1e-3, (arm, disp, a["chan"][c])
    # the one-cell arm is the one with a certificate; assert it there only
    for comp in ("cur_u", "cur_v"):
        assert res[MAGNITUDES[0]]["bars"]["flow"][comp]["r2"] > 0.08
        assert res[MAGNITUDES[0]]["bars"]["flow_raw"][comp]["r2"] > 0.30
    assert (res[MAGNITUDES[0]]["bars"]["flow"]["cur_u"]["r2"]
            > res[MAGNITUDES[-1]]["bars"]["flow"]["cur_u"]["r2"]), (
        "the linear optical-flow certificate did not degrade with "
        "displacement, which is the one thing it is guaranteed to do")


if __name__ == "__main__":                                # pragma: no cover
    raise SystemExit(pytest.main([__file__, "-q", "-s"]))

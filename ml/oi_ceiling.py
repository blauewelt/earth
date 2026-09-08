#!/usr/bin/env python3
"""E-076b · the OPTIMAL-INTERPOLATION CEILING for the Argo ocean interior.

PLAIN ENGLISH. E-076a asked whether a neural cone codec (the encoder that
reads a 30-day cone of past measurements around a pixel) does better at
guessing a float profile's temperature and salinity when the nearby Argo
profiles arrive as raw measurements-with-a-distance rather than as a gridded
monthly column. The answer was 2-3 % better than climatology on temperature,
all of it above 300 dbar, and nothing at all on salinity — so the ablation
could not discriminate between two representations of information the model
never used. THIS file asks the prior question, with no model and no GPU:

    can ANY method beat the seasonal climatology at a float's own position,
    given only the k nearest OTHER float profiles within 1,000 km and 30 days?

The classical answer is OPTIMAL INTERPOLATION (OI) — a Gaussian-process
estimate of the anomaly at the target from the neighbours' anomalies, with a
covariance that falls off with distance and with age, plus a noise term. It is
the method the mapped Argo products themselves are built with, so it is the
right ceiling to hold the codec against.

WHAT IS COMPARED, per pressure level, temperature and salinity separately:

  climatology       predict zero anomaly           (the bar E-076a used)
  nearest           copy the nearest neighbour's anomaly  ("persistence")
  inverse-distance  1/(d^2 + eps) weights over the neighbours
  OI                w = C_ox (C_xx + r I)^-1, Gaussian covariance
  OI + shrink       the OI estimate times a scalar alpha in [0, 1]

The OI hyper-parameters (correlation length L, decorrelation time T, the
noise-to-signal ratio r) and the shrinkage alpha are fitted ON TRAINING-YEAR
TARGETS ONLY, by grid search, separately for three depth bands and for
temperature versus salinity, and are then evaluated — never refitted — on two
held-out splits: the terminal years 2021-2024 and the interspersed years
2008-09 / 2016-17.

THE CLIMATOLOGY IS THE SAME BAR E-076a USED, recomputed here from the same
source rather than restated: family 7's `rg100` group (the Roemmich-Gilson
gridded Argo column, one 1-degree map per month) un-z-scored with the tensor's
own `norm_rg100`, averaged per calendar month, per cell, per channel over
TRAINING years only — every year <= 2020 except 2008, 2009, 2016, 2017, which
is the frozen protocol's `holdout_years`. `ml/trainprobe.py::anomaly_transform`
computes exactly this mean inside the training pipeline; the arithmetic is a
plain masked mean over the month's non-held-out rows, and the check that it
reproduces is in the report (E-076a's terminal-split temperature bar reads
1.086 degC at 10 dbar, 0.831 at 300, 0.330 at 900).

Pure numpy. No network. Deterministic given `--seed`. Written to finish in
well under 30 minutes on two CPU cores: every per-target linear system is
21 x 21 at most and they are solved in one batched `np.linalg.solve` call per
(hyper-parameter, level) rather than one per target.

    python3 ml/oi_ceiling.py \
        --store  <dir>/family8_argo_l0 \
        --tensor-npz <dir>/family7_global025_pentad_l0.npz \
        --rg100  <dir>/family7_global025_pentad_l0_X_rg100.npy \
        --out    ml/plans/e076b --n 6000 --seed 0

Plan: ml/EXPERIMENTS.md#e-076a ("what next"); results:
ml/plans/E076b_results.md.
"""
import argparse
import datetime as dt
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

# `knearest` already returns each neighbour's offset east and north FROM THE
# TARGET, computed by `family8_store.offsets_km`; the OI covariance is built
# from those, so there is one definition of the ground scale and this file
# never converts a degree to a kilometre itself.
from family8_store import ArgoStore  # noqa: E402

# ---------------------------------------------------------------- protocol --
# The frozen holdout protocol, `holdout_years` = "2008,2009,2016,2017,2021,
# 2022,2023,2024": every year at or before 2020 that is not one of the four
# interspersed ones is a training year, and the two held-out blocks are read
# apart because they ask different questions (ml/train_cone.py::
# heldout_split_of) — 2021-2024 is the TERMINAL holdout, where a forecast
# would actually run, and 2008/2009/2016/2017 sit surrounded by training data.
INTERSPERSED_YEARS = (2008, 2009, 2016, 2017)
TERMINAL_YEARS = (2021, 2022, 2023, 2024)
TRAIN_YEAR_MAX = 2020
SPLITS = ("train", "terminal", "interspersed")

EPOCH = dt.date(1982, 1, 1)
PENTAD_DAYS = 5

# The sixteen Roemmich-Gilson pressure levels, in dbar. The store's levels and
# the `rg100` group's channels are the same sixteen (docs/FAMILY8_DATA_HANDOVER
# .md section 3); this list is asserted against both rather than trusted.
LEVELS = (10., 30., 50., 100., 150., 200., 300., 400., 500., 700., 900.,
          1100., 1300., 1500., 1700., 1900.)

# Three depth bands, fitted independently: the seasonal thermocline, the
# permanent thermocline, and the deep water below it.
BANDS = (("10-100", 0.0, 100.0), ("150-500", 101.0, 500.0),
         ("700-1900", 501.0, 2000.0))

# The OI grid, from the spec. L in km, T in days, r the noise-to-signal ratio
# added to the diagonal of the neighbour covariance.
GRID_L_KM = (50., 100., 150., 200., 300., 500.)
GRID_T_DAYS = (10., 20., 30., 60.)
GRID_R = (0.1, 0.3, 1.0, 3.0, 10.0)
ALPHA_GRID = np.linspace(0.0, 1.0, 101)

# The four searches. (name, K, T_max_days, R_max_km). The first three are
# E-076a's own search widened only in k; the fourth widens the search itself,
# to see whether the ceiling rises when more, farther and older neighbours are
# admitted.
CONFIGS = (("K5", 5, 30.0, 1000.0),
           ("K10", 10, 30.0, 1000.0),
           ("K20", 20, 30.0, 1000.0),
           ("K20-wide", 20, 60.0, 1500.0))
BASE_CONFIG = "K20"          # the search the target set is drawn against

IDW_EPS = 1e-6               # on the DIMENSIONLESS space-time metric
MIN_CLIM_LEVELS = 8          # a target needs a climatology at >= 8 T levels
MIN_NEIGHBOURS = 2           # after the target itself is removed
FAR_ONLY_KM = 50.0           # the sensitivity pass: no companion this close


# ------------------------------------------------------------------ helpers --
def year_of_days(time_days):
    """Calendar year of `days since 1982-01-01` (float, fractional)."""
    d = (np.datetime64(EPOCH)
         + np.floor(np.asarray(time_days, np.float64)).astype("timedelta64[D]"))
    return d.astype("datetime64[Y]").astype(np.int64) + 1970


def month_of_days(time_days):
    """Calendar month index 0..11 of `days since 1982-01-01`.

    The same derivation as `ml/cone_sampler.py::ProfileGather.month_of`, so a
    profile is charged to the month the training pipeline charges it to.
    """
    d = (np.datetime64(EPOCH)
         + np.floor(np.asarray(time_days, np.float64)).astype("timedelta64[D]"))
    return (d.astype("datetime64[M]").astype(np.int64) % 12).astype(np.int64)


def split_of_year(year):
    """Which of the three pools a year belongs to, or None."""
    y = int(year)
    if y in TERMINAL_YEARS:
        return "terminal"
    if y in INTERSPERSED_YEARS:
        return "interspersed"
    if y <= TRAIN_YEAR_MAX:
        return "train"
    return None


def rmse(err, mask):
    """Root mean square of `err` over `mask`, and the count. NaN if empty."""
    m = np.asarray(mask, bool) & np.isfinite(err)
    n = int(m.sum())
    if n == 0:
        return float("nan"), 0
    return float(np.sqrt(np.sum(np.asarray(err, np.float64)[m] ** 2) / n)), n


# ------------------------------------------------------------ climatology --
class Climatology:
    """The train-years monthly mean of the `rg100` column, in RAW units.

    `clim[m, y1, x1, c]` is degrees Celsius for channels 0..15 (10..1900 dbar)
    and PSU for 16..31, or NaN where that (month, cell, channel) had no
    training sample — which is the honest answer and is what makes a target
    ineligible rather than something to fill.
    """

    def __init__(self, clim, lat1, lon1, chan, norm, n_rows_used,
                 rows_per_month):
        self.clim = clim
        self.lat0, self.dlat = float(lat1[0]), float(lat1[1] - lat1[0])
        self.lon0, self.dlon = float(lon1[0]), float(lon1[1] - lon1[0])
        self.H, self.W = clim.shape[1], clim.shape[2]
        self.wrap = abs(self.W * self.dlon - 360.0) < 1e-6
        self.chan = list(chan)
        self.norm = norm
        self.n_rows_used = int(n_rows_used)
        self.rows_per_month = list(rows_per_month)

    def cell_of(self, lat, lon):
        """`(y1, x1)` of the 1-degree cell a profile is served from.

        `floor(v + 0.5)` on the group's own axes — the same rounding
        `ml/cone_sampler.py::ProfileGather.cell_of` uses, so a profile lands on
        the cell the tensor would have served it from.
        """
        y1 = np.floor((np.asarray(lat, np.float64) - self.lat0) / self.dlat
                      + 0.5).astype(np.int64)
        x1 = np.floor((np.asarray(lon, np.float64) - self.lon0) / self.dlon
                      + 0.5).astype(np.int64)
        y1 = np.clip(y1, 0, self.H - 1)
        x1 = np.mod(x1, self.W) if self.wrap else np.clip(x1, 0, self.W - 1)
        return y1, x1

    def at(self, lat, lon, time_days):
        """`[n, 32]` climatology at each profile's own cell and month."""
        y1, x1 = self.cell_of(lat, lon)
        m = month_of_days(time_days)
        return np.asarray(self.clim[m, y1, x1, :], np.float64)


def build_climatology(npz, rg100_path, verbose=True):
    """Average the `rg100` rows of TRAINING years, per calendar month.

    The array on disk is `[252, 181, 360, 32]` float16 and Z-SCORED, one row
    per month from 2004-01 to 2024-12; `norm_rg100` carries the (mean, sd) it
    was scored with and `rg_months` says which month each row is. NaN means
    "not mapped" — outside the Roemmich-Gilson band (-64.5 to 79.5 degrees) or
    on land — and stays NaN in the mean.

    The mean is taken in z-space and un-scored once at the end, which is the
    same number as averaging raw values (the transform is affine and its
    constants are per channel, not per row) and keeps the accumulator in the
    units the file is in.
    """
    rg_months = np.asarray(npz["rg_months"])
    norm = np.asarray(npz["norm_rg100"], np.float64)
    lat1, lon1 = np.asarray(npz["lat1"]), np.asarray(npz["lon1"])
    chan = [str(c) for c in np.asarray(npz["chan_rg100"])]

    X = np.load(rg100_path, mmap_mode="r")
    if X.shape[0] != len(rg_months):
        raise ValueError(
            f"{rg100_path} has {X.shape[0]} rows and the npz's rg_months has "
            f"{len(rg_months)} — the two do not describe the same build")
    if X.shape[3] != len(chan) or X.shape[1] != len(lat1) \
            or X.shape[2] != len(lon1):
        raise ValueError(
            f"{rg100_path} is {X.shape}, want "
            f"[{len(rg_months)}, {len(lat1)}, {len(lon1)}, {len(chan)}]")

    # A row's month AND its holdout side must be the master axis's, not a
    # restatement (ml/cone_sampler.py::group_time). `rg_months` is the npz's
    # own per-row label and `rg_bin_index` its absolute pentad bin; check the
    # two agree through the master axis before either is used.
    master_months = np.asarray(npz["months"])
    master_bins = np.asarray(npz["bin_index"], np.int64)
    rg_bins = np.asarray(npz["rg_bin_index"], np.int64)
    pos = np.searchsorted(master_bins, rg_bins)
    if not np.array_equal(master_bins[pos], rg_bins):
        raise ValueError("rg_bin_index carries a bin the master axis lacks")
    if not np.array_equal(master_months[pos], rg_months):
        raise ValueError(
            "rg_months disagrees with the master axis's month for the same "
            "bin — the live-bins group would be charged to the wrong month")

    years = np.array([int(s[:4]) for s in rg_months], np.int64)
    moy = np.array([int(s[5:7]) - 1 for s in rg_months], np.int64)
    is_train = np.array([split_of_year(y) == "train" for y in years], bool)

    H, W, C = X.shape[1], X.shape[2], X.shape[3]
    clim = np.full((12, H, W, C), np.nan, np.float32)
    rows_per_month = []
    for m in range(12):
        rows = np.flatnonzero((moy == m) & is_train)
        rows_per_month.append(int(len(rows)))
        tot = np.zeros((H, W, C), np.float64)
        cnt = np.zeros((H, W, C), np.int32)
        for r in rows:
            blk = np.asarray(X[r], np.float64)
            fin = np.isfinite(blk)
            np.add(tot, np.where(fin, blk, 0.0), out=tot)
            cnt += fin
        with np.errstate(invalid="ignore", divide="ignore"):
            z = np.where(cnt > 0, tot / np.maximum(cnt, 1), np.nan)
        clim[m] = (z * norm[:, 1][None, None, :]
                   + norm[:, 0][None, None, :]).astype(np.float32)
        if verbose:
            print(f"  climatology month {m + 1:2d}: {len(rows)} training rows, "
                  f"{int((cnt > 0).sum()) / cnt.size:.1%} of cells mapped",
                  flush=True)
    del X
    return Climatology(clim, lat1, lon1, chan, norm,
                       int(is_train.sum()), rows_per_month)


# --------------------------------------------------------------- the gather --
class Neighbours:
    """One config's neighbour set for one pool of targets, as flat arrays.

    `a_t` / `a_s` are the neighbours' ANOMALIES (raw minus the climatology at
    that neighbour's own cell and month), NaN where the level is unmeasured or
    the climatology is undefined there. `y_t` / `y_s` are the target's, on the
    same footing. Slot `j` of target `i` is real only where `valid[i, j]`.
    """

    def __init__(self, n, K, n_lev):
        self.K = K
        self.valid = np.zeros((n, K), bool)
        self.dx = np.zeros((n, K), np.float64)
        self.dy = np.zeros((n, K), np.float64)
        self.dt = np.zeros((n, K), np.float64)     # RELATIVE to the target
        self.rank = np.zeros((n, K), np.int64)     # the search's own ordering
        self.a_t = np.full((n, K, n_lev), np.nan)
        self.a_s = np.full((n, K, n_lev), np.nan)
        self.y_t = np.full((n, n_lev), np.nan)
        self.y_s = np.full((n, n_lev), np.nan)
        self.n_R = np.zeros(n, np.int64)


def gather(store, clim, rows, K, T_max, R_max, n_lev):
    """Run the search once per target and turn it into anomalies.

    The target is IN ITS OWN BIN, so the search returns it; `k = K + 1` is
    asked for and the target's own row is dropped, leaving the K nearest OTHER
    profiles. Where the target is not itself inside the top K + 1 (possible: it
    is at zero distance but its age within the pentad can exceed a closer
    neighbour's), the first K rows are already the K nearest others and nothing
    is dropped — that case is counted and reported rather than asserted away.
    """
    n = len(rows)
    out = Neighbours(n, K, n_lev)
    lat = np.asarray(store["lat"], np.float64)
    lon = np.asarray(store["lon"], np.float64)
    tim = np.asarray(store["time_days"], np.float64)
    bins = np.asarray(store["bin"], np.int64)
    n_self_missing = 0

    tgt_clim = clim.at(lat[rows], lon[rows], tim[rows])
    y_raw_t = np.asarray(store["temp"][rows], np.float64)
    y_raw_s = np.asarray(store["psal"][rows], np.float64)
    out.y_t = y_raw_t - tgt_clim[:, :n_lev]
    out.y_s = y_raw_s - tgt_clim[:, n_lev:]

    for i, row in enumerate(rows):
        res = store.knearest(lat[row], lon[row], bin=int(bins[row]), k=K + 1,
                             R_max_km=R_max, T_max_days=T_max)
        keep = np.flatnonzero(res["valid"])
        self_at = np.flatnonzero(res["row"][keep] == row)
        if len(self_at):
            keep = np.delete(keep, self_at[0])
        else:
            n_self_missing += 1
        keep = keep[:K]
        m = len(keep)
        if m == 0:
            out.n_R[i] = int(res["n_R"]) - 1
            continue
        nrow = res["row"][keep]
        out.valid[i, :m] = True
        out.rank[i, :m] = np.arange(m)
        out.dx[i, :m] = res["dx_km"][keep]
        out.dy[i, :m] = res["dy_km"][keep]
        # Time RELATIVE TO THE TARGET, not to the end of the pentad: the
        # search measures every age back from `5 * (bin + 1)` so that no
        # observation reads as being in the future, but the covariance between
        # two measurements depends on when THEY were taken.
        out.dt[i, :m] = res["dt_days"][keep] - (5.0 * (int(bins[row]) + 1)
                                                - tim[row])
        cl = clim.at(lat[nrow], lon[nrow], tim[nrow])
        out.a_t[i, :m] = np.asarray(store["temp"][nrow], np.float64) \
            - cl[:, :n_lev]
        out.a_s[i, :m] = np.asarray(store["psal"][nrow], np.float64) \
            - cl[:, n_lev:]
        out.n_R[i] = int(res["n_R"]) - 1
    return out, n_self_missing


def split_pool(store, split):
    """Every store row whose profile's own year belongs to `split`."""
    years = year_of_days(np.asarray(store["time_days"], np.float64))
    uy = np.unique(years)
    ok = np.array([split_of_year(y) == split for y in uy], bool)
    return np.flatnonzero(ok[np.searchsorted(uy, years)])


def draw_targets(store, clim, rng, n_want, split, n_lev, T_max, R_max,
                 chunk=50_000):
    """`n_want` store rows of one split, by rejection, with a fixed stream.

    Two conditions, both properties of the DATA rather than of any estimator:
    the climatology must be defined at the target's own cell and month for at
    least `MIN_CLIM_LEVELS` temperature levels (otherwise there is no anomaly
    to predict and no bar to predict it against), and at least
    `MIN_NEIGHBOURS` other profiles must lie inside the search bounds — the
    same `n_R >= 2` rule E-076a drew its anchors by, applied after the target
    itself is taken out.

    The pool is shuffled ONCE and walked in order, so the draw is a uniform
    sample of the profiles that satisfy both conditions and does not depend on
    how many chunks it took to fill.
    """
    tim = np.asarray(store["time_days"], np.float64)
    lat = np.asarray(store["lat"], np.float64)
    lon = np.asarray(store["lon"], np.float64)
    bins = np.asarray(store["bin"], np.int64)
    pool = split_pool(store, split)
    order = rng.permutation(pool)

    keep, seen, after_clim, searched = [], 0, 0, 0
    for i0 in range(0, len(order), chunk):
        blk = order[i0:i0 + chunk]
        cl = clim.at(lat[blk], lon[blk], tim[blk])
        ok = np.isfinite(cl[:, :n_lev]).sum(axis=1) >= MIN_CLIM_LEVELS
        for j, row in enumerate(blk):
            seen += 1
            if not ok[j]:
                continue
            after_clim += 1
            searched += 1
            res = store.knearest(lat[row], lon[row], bin=int(bins[row]), k=1,
                                 R_max_km=R_max, T_max_days=T_max)
            if int(res["n_R"]) - 1 >= MIN_NEIGHBOURS:
                keep.append(int(row))
                if len(keep) >= n_want:
                    break
        if len(keep) >= n_want:
            break
    return (np.asarray(keep, np.int64),
            dict(pool=int(len(pool)), seen=int(seen),
                 after_clim=int(after_clim), searched=int(searched),
                 kept=int(len(keep))))


# --------------------------------------------------------------- estimators --
def metric_d2(nb, R_max, T_max):
    """The dimensionless space-time metric, on the target-relative age."""
    return ((nb.dx ** 2 + nb.dy ** 2) / (R_max * R_max)
            + (nb.dt ** 2) / (T_max * T_max))


def predict_nearest(A, nb):
    """Copy the nearest neighbour THAT HAS a value at this level. `[n]`."""
    ok = nb.valid & np.isfinite(A)
    big = np.where(ok, nb.rank, np.iinfo(np.int64).max)
    j = np.argmin(big, axis=1)
    idx = np.arange(A.shape[0])
    got = ok[idx, j]
    return np.where(got, A[idx, j], np.nan)


def predict_idw(A, nb, d2):
    """Inverse-distance weights over the neighbours with a value here."""
    ok = nb.valid & np.isfinite(A)
    w = np.where(ok, 1.0 / (d2 + IDW_EPS), 0.0)
    s = w.sum(axis=1)
    num = np.sum(w * np.where(ok, A, 0.0), axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(s > 0, num / s, np.nan)


def oi_matrices(nb, L, T):
    """`(C_ox [n, K], C_xx [n, K, K])` for one (L, T), UNMASKED.

    Distances between neighbours come from their offsets with the TARGET AS
    ORIGIN — a plane approximation, which at <= 1,500 km costs a fraction of a
    percent of the separation and nothing at all of the covariance.
    """
    twoL2, twoT2 = 2.0 * L * L, 2.0 * T * T
    r2 = nb.dx ** 2 + nb.dy ** 2
    c_ox = np.exp(-r2 / twoL2 - (nb.dt ** 2) / twoT2)
    ddx = nb.dx[:, :, None] - nb.dx[:, None, :]
    ddy = nb.dy[:, :, None] - nb.dy[:, None, :]
    ddt = nb.dt[:, :, None] - nb.dt[:, None, :]
    c_xx = np.exp(-(ddx * ddx + ddy * ddy) / twoL2 - (ddt * ddt) / twoT2)
    return c_ox, c_xx


def predict_oi(A, nb, c_ox, c_xx, r):
    """The OI estimate at one level, for every target at once.

    A neighbour with no value at this level is removed from the system by
    setting its row and column of `C_xx` to the identity and its `C_ox` to
    zero: the solve then returns exactly the sub-solve over the present
    neighbours and a weight of zero for the absent ones, which is the same
    answer a per-target masked solve would give, in one batched call.
    """
    ok = nb.valid & np.isfinite(A)
    pair = ok[:, :, None] & ok[:, None, :]
    M = np.where(pair, c_xx, 0.0)
    d = np.arange(nb.K)
    M[:, d, d] = np.where(ok, 1.0 + r, 1.0)
    c = np.where(ok, c_ox, 0.0)
    w = np.linalg.solve(M, c[:, :, None])[:, :, 0]
    pred = np.sum(w * np.where(ok, A, 0.0), axis=1)
    return np.where(ok.any(axis=1), pred, np.nan)


# ------------------------------------------------------------------- the fit --
def band_levels(band):
    _, lo, hi = band
    return [j for j, p in enumerate(LEVELS) if lo <= p <= hi]


def fit_oi(A, nb, y, mask, levs, log=None):
    """Grid-search (L, T, r) minimising the MEAN over `levs` of the RMSE.

    Fitted on TRAINING targets only. `mask[:, l]` is the common evaluation
    mask at level l (a finite target anomaly and at least one neighbour with a
    value there), so every candidate is scored on exactly the same targets.
    """
    best = None
    for L in GRID_L_KM:
        for T in GRID_T_DAYS:
            c_ox, c_xx = oi_matrices(nb, L, T)
            for r in GRID_R:
                sc = []
                for l in levs:
                    p = predict_oi(A[:, :, l], nb, c_ox, c_xx, r)
                    v, _ = rmse(p - y[:, l], mask[:, l])
                    sc.append(v)
                score = float(np.nanmean(sc))
                if best is None or score < best[0]:
                    best = (score, L, T, r)
            del c_ox, c_xx
    score, L, T, r = best
    if log is not None:
        log.append(dict(L_km=L, T_days=T, r=r, train_mean_rmse=score))
    return L, T, r, score


def fit_alpha(pred, y, mask, levs):
    """The shrinkage that minimises the mean over `levs` of the RMSE."""
    best = (None, None)
    for a in ALPHA_GRID:
        sc = [rmse(a * pred[:, l] - y[:, l], mask[:, l])[0] for l in levs]
        s = float(np.nanmean(sc))
        if best[0] is None or s < best[0]:
            best = (s, float(a))
    return best[1], best[0]


# ------------------------------------------------------------------- scoring --
def score_variable(nb, var, cfg, fitted, fit_mode, out, keep_rows=None):
    """Every estimator at every level of one variable, for one pool.

    `fit_mode=True` fits the hyper-parameters on this pool (the training one)
    and writes them into `fitted`; otherwise it reads them and only evaluates.
    `keep_rows`, when given, restricts the population to those targets — used
    for the far-neighbour sensitivity pass, never for fitting.
    """
    name, K, T_max, R_max = cfg
    A = nb.a_t if var == "t" else nb.a_s
    y = nb.y_t if var == "t" else nb.y_s
    n_lev = A.shape[2]
    d2 = metric_d2(nb, R_max, T_max)

    # ONE mask per level for every estimator: the target measured this level
    # AND at least one neighbour did. Scoring the climatology on a larger set
    # than the interpolators would compare two different questions.
    has_nb = (nb.valid[:, :, None] & np.isfinite(A)).any(axis=1)
    mask = np.isfinite(y) & has_nb
    if keep_rows is not None:
        mask = mask & np.asarray(keep_rows, bool)[:, None]

    rows = {k: [None] * n_lev for k in
            ("clim", "nearest", "idw", "oi", "oi_shrink")}
    rows["clim_all"] = [None] * n_lev          # the E-076a bar: target only
    counts = [0] * n_lev
    counts_all = [0] * n_lev
    for l in range(n_lev):
        v, n = rmse(-y[:, l], mask[:, l])
        rows["clim"][l], counts[l] = v, n
        only = np.isfinite(y[:, l])
        if keep_rows is not None:
            only = only & np.asarray(keep_rows, bool)
        v, n = rmse(-y[:, l], only)
        rows["clim_all"][l], counts_all[l] = v, n
        rows["nearest"][l] = rmse(predict_nearest(A[:, :, l], nb) - y[:, l],
                                  mask[:, l])[0]
        rows["idw"][l] = rmse(predict_idw(A[:, :, l], nb, d2) - y[:, l],
                              mask[:, l])[0]

    for band in BANDS:
        bname = band[0]
        levs = band_levels(band)
        key = f"{name}|{var}|{bname}"
        if fit_mode:
            L, T, r, sc = fit_oi(A, nb, y, mask, levs)
            c_ox, c_xx = oi_matrices(nb, L, T)
            pred = np.full((A.shape[0], n_lev), np.nan)
            for l in levs:
                pred[:, l] = predict_oi(A[:, :, l], nb, c_ox, c_xx, r)
            alpha, sca = fit_alpha(pred, y, mask, levs)
            fitted[key] = dict(L_km=L, T_days=T, r=r, alpha=alpha,
                               train_mean_rmse_oi=sc,
                               train_mean_rmse_shrink=sca)
        else:
            f = fitted[key]
            L, T, r, alpha = f["L_km"], f["T_days"], f["r"], f["alpha"]
            c_ox, c_xx = oi_matrices(nb, L, T)
            pred = np.full((A.shape[0], n_lev), np.nan)
            for l in levs:
                pred[:, l] = predict_oi(A[:, :, l], nb, c_ox, c_xx, r)
        for l in levs:
            rows["oi"][l] = rmse(pred[:, l] - y[:, l], mask[:, l])[0]
            rows["oi_shrink"][l] = rmse(alpha * pred[:, l] - y[:, l],
                                        mask[:, l])[0]
        del c_ox, c_xx

    res = {k: [None if v is None or not np.isfinite(v) else float(v)
               for v in rows[k]] for k in rows}
    res["n"] = counts
    res["n_target_only"] = counts_all
    # Skill = mean over levels of RMSE / climatology-RMSE, the summary E-076a
    # reports: 1.0 is no better than climatology, lower is better.
    for est in ("nearest", "idw", "oi", "oi_shrink"):
        rr = [res[est][l] / res["clim"][l] for l in range(n_lev)
              if res[est][l] is not None and res["clim"][l]]
        res[f"skill_{est}"] = float(np.mean(rr)) if rr else None
        for band in BANDS:
            levs = band_levels(band)
            rr = [res[est][l] / res["clim"][l] for l in levs
                  if res[est][l] is not None and res["clim"][l]]
            res.setdefault("skill_band", {}).setdefault(est, {})[band[0]] = \
                (float(np.mean(rr)) if rr else None)
    out[var] = res
    return out


# ---------------------------------------------------------------------- main --
def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--store", required=True,
                    help="the family-8 observation store directory")
    ap.add_argument("--tensor-npz", required=True,
                    help="family7_global025_pentad_l0.npz (metadata + norms)")
    ap.add_argument("--rg100", required=True,
                    help="family7_global025_pentad_l0_X_rg100.npy")
    ap.add_argument("--out", required=True, help="output directory")
    ap.add_argument("--n", type=int, default=6000,
                    help="targets per split (default 6000)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--quiet", action="store_true")
    a = ap.parse_args(argv)

    t0 = time.time()
    os.makedirs(a.out, exist_ok=True)
    say = (lambda *m: None) if a.quiet else (
        lambda *m: print(*m, flush=True))

    store = ArgoStore(a.store)
    n_lev = store.n_levels
    if not np.allclose(np.asarray(store.levels), np.asarray(LEVELS)):
        raise ValueError(f"the store's levels {list(store.levels)} are not "
                         f"the sixteen Roemmich-Gilson levels {list(LEVELS)}")
    say(f"store: {len(store):,} profiles, {n_lev} levels, "
        f"{store.n_bins} pentad bins")

    npz = np.load(a.tensor_npz, allow_pickle=True)
    chan = [str(c) for c in np.asarray(npz["chan_rg100"])]
    want = [f"rg_t{int(p)}" for p in LEVELS] + [f"rg_s{int(p)}" for p in LEVELS]
    if chan != want:
        raise ValueError(f"chan_rg100 is {chan}, want {want} — the channel "
                         f"order decides which column is which level")
    say("building the train-years monthly climatology from rg100 ...")
    clim = build_climatology(npz, a.rg100, verbose=not a.quiet)
    say(f"  {clim.n_rows_used} training rows of {len(npz['rg_months'])} "
        f"({clim.rows_per_month[0]} per month)  "
        f"[{time.time() - t0:.0f} s]")

    # ---- targets -----------------------------------------------------------
    _, base_K, base_T, base_R = [c for c in CONFIGS if c[0] == BASE_CONFIG][0]
    rows, draw_stats = {}, {}
    for k, split in enumerate(SPLITS):
        rng = np.random.default_rng(a.seed * 1000 + k)
        rows[split], draw_stats[split] = draw_targets(
            store, clim, rng, a.n, split, n_lev, base_T, base_R)
        say(f"targets {split:13s}: {len(rows[split]):>5,} of "
            f"{draw_stats[split]['pool']:,} in-pool profiles "
            f"({draw_stats[split]['searched']:,} searched)  "
            f"[{time.time() - t0:.0f} s]")

    result = {
        "experiment": "E-076b",
        "seed": a.seed, "n_requested": int(a.n),
        "levels_dbar": [float(v) for v in LEVELS],
        "bands": [b[0] for b in BANDS],
        "protocol": {
            "train_years": "<= 2020 except 2008, 2009, 2016, 2017",
            "terminal": list(TERMINAL_YEARS),
            "interspersed": list(INTERSPERSED_YEARS),
            "min_clim_levels": MIN_CLIM_LEVELS,
            "min_neighbours_after_removal": MIN_NEIGHBOURS,
            "target_draw_search": dict(K=base_K, T_max_days=base_T,
                                       R_max_km=base_R),
        },
        "climatology": {
            "source": os.path.basename(a.rg100),
            "training_rows": clim.n_rows_used,
            "rows_per_month": clim.rows_per_month,
        },
        "grids": {"L_km": list(GRID_L_KM), "T_days": list(GRID_T_DAYS),
                  "r": list(GRID_R), "alpha": "0..1 in 101 steps"},
        "idw_eps": IDW_EPS,
        "draw": draw_stats,
        "configs": {},
    }

    for cfg in CONFIGS:
        name, K, T_max, R_max = cfg
        say(f"\n=== {name}: k={K}, T_max={T_max:g} d, R_max={R_max:g} km ===")
        nbs, self_miss, dens = {}, {}, {}
        for split in SPLITS:
            nb, miss = gather(store, clim, rows[split], K, T_max, R_max, n_lev)
            nbs[split], self_miss[split] = nb, miss
            near_km = np.where(nb.valid[:, 0],
                               np.hypot(nb.dx[:, 0], nb.dy[:, 0]), np.nan)
            dens[split] = dict(
                mean_n_R=float(np.mean(nb.n_R)),
                mean_found=float(np.mean(nb.valid.sum(axis=1))),
                mean_nearest_km=float(np.nanmean(near_km)),
                median_nearest_km=float(np.nanmedian(near_km)),
                # The obvious objection to any of these numbers: a float
                # cycling every ~5 days puts a near-copy of the target a few
                # km away, and an interpolator that finds one is not
                # interpolating. This says how often that happens.
                frac_nearest_under_20km=float(np.nanmean(near_km < 20.0)),
                frac_nearest_under_50km=float(np.nanmean(near_km < 50.0)),
                # And the other one: an anomaly is raw minus a gridded monthly
                # mean, so a bad cell or a bad sensor shows up as an outlier
                # that a least-squares estimator cannot ignore.
                max_abs_anom_t=float(np.nanmax(np.abs(nb.a_t))),
                max_abs_anom_s=float(np.nanmax(np.abs(nb.a_s))),
                p999_abs_anom_t=float(np.nanpercentile(np.abs(nb.a_t), 99.9)),
                p999_abs_anom_s=float(np.nanpercentile(np.abs(nb.a_s), 99.9)))
            say(f"  gathered {split:13s} n={len(rows[split]):,} "
                f"mean slots {dens[split]['mean_found']:.2f} "
                f"nearest {dens[split]['mean_nearest_km']:.0f} km  "
                f"[{time.time() - t0:.0f} s]")

        fitted = {}
        blocks = {}
        blocks["train"] = {}
        for var in ("t", "s"):
            score_variable(nbs["train"], var, cfg, fitted, True,
                           blocks["train"])
        say(f"  fitted on {len(rows['train']):,} training targets  "
            f"[{time.time() - t0:.0f} s]")
        far = {}
        for split in ("terminal", "interspersed"):
            blocks[split] = {}
            for var in ("t", "s"):
                score_variable(nbs[split], var, cfg, fitted, False,
                               blocks[split])
            # THE SENSITIVITY PASS. About one target in nine has another
            # profile within 20 km — a float cycling every ~5 days leaves a
            # near-copy of itself beside the target, and finding one is not
            # interpolation. This re-scores the identical estimators, with the
            # identical hyper-parameters (nothing is refitted), on the targets
            # whose NEAREST neighbour is at least FAR_ONLY_KM away, so the
            # headline can be read with those cases taken out.
            nb = nbs[split]
            near_km = np.where(nb.valid[:, 0],
                               np.hypot(nb.dx[:, 0], nb.dy[:, 0]), np.inf)
            keep = near_km >= FAR_ONLY_KM
            far[split] = {"n_targets": int(keep.sum()),
                          "min_nearest_km": FAR_ONLY_KM}
            for var in ("t", "s"):
                score_variable(nb, var, cfg, fitted, False, far[split],
                               keep_rows=keep)
            say(f"  scored {split} ({int(keep.sum()):,} of {len(keep):,} with "
                f"the nearest neighbour >= {FAR_ONLY_KM:g} km)  "
                f"[{time.time() - t0:.0f} s]")

        result["configs"][name] = {
            "K": K, "T_max_days": T_max, "R_max_km": R_max,
            "hyper": fitted, "density": dens,
            "self_not_in_topk": self_miss,
            "splits": blocks, "splits_far_only": far,
        }
        del nbs

    result["runtime_s"] = round(time.time() - t0, 1)
    path = os.path.join(a.out, "oi_ceiling.json")
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(result, f, indent=1)
    os.replace(tmp, path)
    say(f"\nwrote {path}  ({result['runtime_s']:.0f} s)")
    print_tables(result)
    return result


# -------------------------------------------------------------- the tables --
def _fmt(v, w=6, p=3):
    return " " * w if v is None else f"{v:{w}.{p}f}"


def print_tables(res):
    lev = res["levels_dbar"]
    print("\n" + "=" * 78)
    print("E-076b · the optimal-interpolation ceiling for the Argo interior")
    print("=" * 78)
    print(f"seed {res['seed']} · {res['n_requested']} targets asked per split "
          f"· runtime {res.get('runtime_s', 0):.0f} s")
    for split, kept in [(k, v["kept"]) for k, v in res["draw"].items()]:
        print(f"  {split:13s} {kept:,} targets drawn")

    for cname, C in res["configs"].items():
        print(f"\n--- {cname}  (k={C['K']}, T_max={C['T_max_days']:g} d, "
              f"R_max={C['R_max_km']:g} km) ---")
        print("  fitted on training targets (L km / T days / r / alpha):")
        for key, f in sorted(C["hyper"].items()):
            _, var, band = key.split("|")
            print(f"    {'temperature' if var == 't' else 'salinity':12s} "
                  f"{band:9s}  L {f['L_km']:5.0f}  T {f['T_days']:3.0f}  "
                  f"r {f['r']:5.2f}  alpha {f['alpha']:.2f}")
        for split in ("terminal", "interspersed"):
            B = C["splits"][split]
            for var, unit in (("t", "degC"), ("s", "PSU")):
                R = B[var]
                print(f"\n  {split} · {'temperature' if var == 't' else 'salinity'}"
                      f" RMSE ({unit})")
                print("    dbar |    clim  nearest      idw       OI  OI+shrink"
                      "        n")
                for j, p in enumerate(lev):
                    print(f"    {int(p):>4} | {_fmt(R['clim'][j], 7)} "
                          f"{_fmt(R['nearest'][j], 8)} {_fmt(R['idw'][j], 8)} "
                          f"{_fmt(R['oi'][j], 8)} {_fmt(R['oi_shrink'][j], 10)} "
                          f"{R['n'][j]:>8,}")
                print(f"    skill (mean RMSE/clim over levels): "
                      f"nearest {_fmt(R['skill_nearest'], 6, 4)} · "
                      f"idw {_fmt(R['skill_idw'], 6, 4)} · "
                      f"OI {_fmt(R['skill_oi'], 6, 4)} · "
                      f"OI+shrink {_fmt(R['skill_oi_shrink'], 6, 4)}")
                for est in ("oi", "oi_shrink"):
                    sb = R["skill_band"][est]
                    print("      " + est + " by band: " + " · ".join(
                        f"{b} {_fmt(sb[b], 6, 4)}" for b in sb))
                F = C.get("splits_far_only", {}).get(split, {}).get(var)
                if F is not None:
                    print(f"      nearest >= {C['splits_far_only'][split]['min_nearest_km']:g} km "
                          f"({C['splits_far_only'][split]['n_targets']:,} targets): "
                          f"nearest {_fmt(F['skill_nearest'], 6, 4)} · "
                          f"idw {_fmt(F['skill_idw'], 6, 4)} · "
                          f"OI {_fmt(F['skill_oi'], 6, 4)} · "
                          f"OI+shrink {_fmt(F['skill_oi_shrink'], 6, 4)}")


if __name__ == "__main__":
    main()

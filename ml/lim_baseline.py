#!/usr/bin/env python3
"""A LINEAR INVERSE MODEL baseline, scored through the head battery verbatim.

WHY THIS EXISTS. Every rolled corridor AUC in `ml/EXPERIMENTS.md` is a number
about a 200M-parameter transformer head rolled over a learned embedding, and
the only references it has ever been quoted against are persistence, damped
persistence and climatology — three baselines that carry no spatial structure
and no cross-channel coupling at all. The classical instrument for exactly
this question is a Linear Inverse Model (Penland & Sardeshmukh 1995): fit a
single linear propagator to the observed state's own lag-1 covariance, roll it
forward, and see how much of the forecast the machinery was bought for a
linear operator already had. Without that row a reader cannot tell "the head
forecasts the North Atlantic" from "the North Atlantic is largely linear at
these leads".

THE BASELINE IS ONLY A BASELINE IF IT IS SCORED IDENTICALLY. So nothing about
the protocol is re-implemented here. The anomaly transform (`StdMonths`), the
damped-persistence AR(1) reference (`ar1_train`), the corridor definition
(`corridor_pixels`), the #217 gate subset (`gate_subset`), the nine scopes
(`nested_scopes`), the holdout blocks (`hold_blocks`) and the staggered
starts they carry (`TimeAxis.starts_for_block`), the block-boundary break
(`scored_horizon`'s condition), the ten sums
(`new_sums`/`accumulate`), the skill arithmetic (`skill_block`) and the
atomic artefact writer (`write_results`) are all IMPORTED from
`ml/rollout_spatial.py` and called on the same arrays the head path calls them
on. `tests/test_lim_baseline.py` check 2 pins that as an identity — the
function objects are the same objects, and the rows they produce are equal
element for element — rather than as a claim in this docstring.

WHAT THE MODEL IS.

  1. STATE. The standardized-anomaly field the battery already scores: per
     pixel, per channel, de-climatologised against the TRAIN-year monthly
     climatology and z-scored, `nan -> 0` at unobserved cells — i.e. exactly
     `StdMonths.get(t)[0]`, restricted to the SCOREABLE channels (below) over
     the window scope's ocean pixels. x(t) in R^D, D = P * n_channels.
  2. TRAINING SET. Every axis row NOT in a held-out year, where the held-out
     years come from the codec checkpoint's own `args["holdout_years"]` — read
     the way `rollout_spatial.main()` reads them, so the LIM cannot be fitted
     on a bin the head was not allowed to see. `--hold-years` may hold out
     MORE (E-067: consecutive years group into blocks and a roll is truncated
     at the block's end, which is what makes a 730-day lead scoreable); it is
     refused unless it is a superset of the checkpoint's own list, because the
     direction that would contaminate is the other one.
  3. REDUCTION. PCA to K modes via the T x T Gram matrix (`pca_gram`). With
     T_train ~2,900 and D ~694,000 the Gram is the cheap exact route: one
     [T,D]x[D,T] sgemm (~1.2e13 flops, a couple of minutes of CPU) and a
     [T,T] `eigh`, against a randomised SVD that would need its own tolerance
     argument. `sklearn` is NOT installed on the boxes (`.github/workflows/
     ml-train.yml`'s "Install deps" step pip-installs `numpy netCDF4
     matplotlib`, and scipy only for the family-6 build), so this file is
     numpy-only by necessity as well as by preference.
  4. PROPAGATOR. In PC space, G(tau) = C(tau) C(0)^-1 at tau = ONE AXIS BIN,
     with C(tau) and C(0) estimated over CONSECUTIVE TRAINING PAIRS — both
     bins of a pair must be training bins, so a pair never steps across a
     holdout year. Forecast n bins ahead: a(t+n) = G^n a(t), projected back
     through the EOF patterns. `lim_diagnostics` reports the spectral radius,
     whether every mode is damped (|lambda| < 1) and the e-folding time of the
     leading (slowest) mode in DAYS.
  5. ROLL AND SCORE. From the SAME staggered starts, at the SAME horizon, with
     the TRUE standardized state at t0 as the initial condition, scored on the
     same three scopes (and their _trainlon/_holdlon children), per channel,
     against the same climatology / persistence / damped-persistence
     references. Each K becomes one `heads` entry named `lim_k<K>`, so the
     artefact slots into the same `rollout_spatial.json` shape the archive
     already carries and `scripts/archive_probes.py` already publishes.

WHICH CHANNELS. The scoreable set is DERIVED, never hardcoded: a channel can
only contribute to a number if `op = obs_target & obs_start` is true
somewhere, so a channel unobserved at EVERY start row is unscoreable no matter
what the model predicts for it. `observed_channels` reads the (nine, at
starts:3) start rows and keeps the channels finite at any of them — a strict
SUPERSET of the scoring set, which is the safe direction. On the pentad
family-4 tensor that is the eight non-Argo channels (cur_speed, log_mld, ssh,
tau_x, tau_y, tau_x_std, tau_y_std, sst): `build_family4.fill_rg_pentad` fills
"one live pentad per month", and the stride-24 starts all land off that phase,
which is why `probes-516.json`'s `per_channel` carries eight names and not
forty. The exclusion is then ASSERTED rather than trusted (ml/CLAUDE.md §0.2):
every scored step checks that `op` is identically zero on the excluded
channels and REFUSES if it is not, because an excluded channel that did have
an observation would be scored against a predicted zero and would move every
pooled number silently.

MEMORY. The dominant allocation is the training matrix, [T_train, D] float32:
at the production pentad shape (T_train = 3142 - 219 = 2,923 rows, P = 86,698
window ocean pixels, 8 channels, D = 693,584) that is 8.11 GB. On top of it
the EOF patterns [D, K_max] float32 are 0.55 GB at K = 200 (plus, while a
K < K_max is rolling, one contiguous [D, K] copy of the leading block — 0.14
GB at K = 50), the Gram is 68 MB and the standardized-field cache is popped
down to two rows (~35 MB). PEAK is therefore ~9-10 GB against the fleet's
52-110 GB, and `--max-state-gb` (default 24) refuses BEFORE the read pass
rather than OOM-ing after it. `--scope-fit corridor` cuts the state to the
30,158 corridor pixels (2.8 GB) and then reports the gate and window scopes as
UNAVAILABLE rather than scoring them from a state vector that does not cover
their pixels.

Usage (box, CPU is enough — no GPU, no embedding, no Z):
  python3 ml/lim_baseline.py --x ml/cache/family4_na025_pentad_r2_X.npy \\
      --npz-small ml/cache/family4_na025_pentad_r2.npz \\
      --ckpt ml/runs/actions/pixelmae.pt \\
      --K 50,100,200 --horizon 73 --starts 3 \\
      --out ml/runs/actions/rollout_spatial.json

  python3 ml/lim_baseline.py --smoke        # the whole path on a toy, <1 min
"""
import argparse
import datetime as dt
import math
import os
import sys
import tempfile
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import torch                                                    # noqa: E402
# EVERY protocol decision below is IMPORTED, not re-derived. See the module
# docstring: this list IS the claim that the LIM is scored like a head, and
# tests/test_lim_baseline.py asserts these are the same function objects
# rollout_spatial.main() calls.
from rollout_spatial import (TimeAxis, StdMonths, ar1_train,     # noqa: E402
                             corridor_pixels, gate_subset, nested_scopes,
                             new_sums, accumulate, skill_block,
                             scored_horizon, write_results, Progress,
                             hold_blocks, block_label,
                             GATE_HEAD, GATE_REF, GATE_TOL, BAND_EDGE_DAYS,
                             _utc_now)
from recon_eval import stream_stats                              # noqa: E402
from temporal import rapid_section                               # noqa: E402

# tau is ONE AXIS BIN and is not a flag. A LIM fitted at tau and rolled at a
# different step is two models; making tau adjustable here would let a
# dispatch produce a `lim_k100` entry that is not the lim_k100 anybody else
# means. The bin's DURATION is the tensor's (5 d at pentad, ~30.4 d at
# monthly) and travels in the artefact as `tau_days`.
TAU_STEPS = 1


# ---------------------------------------------------------------- numerics --
def pca_gram(A, k_max, tol=1e-7):
    """PCA of `A` [T, D] float32 through the T x T Gram matrix.

    `A` IS MODIFIED IN PLACE: its column mean over the T rows is subtracted,
    because a [2923, 693584] float32 copy is 8.1 GB and this routine exists
    precisely to run inside a memory budget. The caller gets the mean back.

    Returns `(mu [D] f32, V [D, k] f32, scores [T, k] f64, sv [k] f64,
    total_var float)` where V's columns are the orthonormal EOF patterns,
    `scores = U * sv` are the principal-component time series of the rows of
    `A`, `total_var` is the FULL centred sum of squares (so a variance
    fraction is against everything, not only the retained modes) and
    `k <= k_max` drops any mode whose singular value is below `tol` times the
    leading one — a null mode's pattern is 0/0 and would poison every
    projection.

    Exact, not randomised: with T << D the Gram route is one sgemm and one
    symmetric eigendecomposition, and it needs no oversampling parameter to
    document.
    """
    if int(k_max) < 1:
        raise ValueError("k_max must be >= 1")
    mu = A.mean(axis=0, dtype=np.float64).astype(np.float32)
    A -= mu
    G = (A @ A.T).astype(np.float64)                 # [T, T]
    G = 0.5 * (G + G.T)                              # kill the fp asymmetry
    w, Q = np.linalg.eigh(G)                         # ascending
    total_var = float(np.clip(w, 0.0, None).sum())
    order = np.argsort(w)[::-1][:int(k_max)]
    w = np.clip(w[order], 0.0, None)
    sv = np.sqrt(w)
    if sv.size == 0 or sv[0] <= 0:
        raise ValueError("the training matrix has no variance at all")
    keep = sv > sv[0] * float(tol)
    sv = sv[keep]
    U = np.ascontiguousarray(Q[:, order][:, keep], dtype=np.float32)
    V = A.T @ U                                      # [D, k] f32
    V /= sv.astype(np.float32)
    scores = U.astype(np.float64) * sv
    return mu, V, scores, sv, total_var


def lim_pairs(train_rows, tau=TAU_STEPS):
    """Indices `i` into `train_rows` such that `train_rows[i]` and
    `train_rows[i] + tau` are BOTH training bins.

    `train_rows` is sorted and strictly increasing, so "the entry `tau` later
    in the list is `tau` rows later on the axis" is exactly "both bins of this
    pair are training bins and they are `tau` apart". A pair therefore never
    steps across a held-out year, which is the whole point: a propagator
    fitted across that seam would have seen a holdout bin.
    """
    tr = np.asarray(train_rows)
    if len(tr) <= tau:
        return np.zeros(0, np.int64)
    i = np.arange(len(tr) - tau)
    return i[(tr[i + tau] - tr[i]) == tau]


def lim_propagator(scores, train_rows, k, tau=TAU_STEPS):
    """`(G [k,k] f64, n_pairs)` — the LIM propagator G(tau) = C(tau) C(0)^-1.

    C(0) is estimated over the SAME pair set as C(tau), not over every
    training row: the two covariances must describe one sample, or the
    quotient is not a propagator of anything.
    """
    i0 = lim_pairs(train_rows, tau)
    if len(i0) < k + 1:
        raise ValueError(f"only {len(i0)} consecutive training pairs for "
                         f"k={k} modes — C(0) would be singular")
    X0 = np.asarray(scores[i0, :k], np.float64)
    X1 = np.asarray(scores[i0 + tau, :k], np.float64)
    n = float(len(i0))
    C0 = (X0.T @ X0) / n
    Ct = (X1.T @ X0) / n
    # G = Ct @ inv(C0), SOLVED rather than inverted: C0^T G^T = Ct^T.
    G = np.linalg.solve(C0.T, Ct.T).T
    return G, int(len(i0))


def lim_diagnostics(G, step_days, tau=TAU_STEPS):
    """Is the propagator STABLE, and how fast does its slowest mode decay?

    A LIM whose spectral radius reaches 1 does not decay toward climatology —
    it grows without bound, and `msss_clim` at long lead becomes a statement
    about a divergence rather than about a forecast. So the radius is
    reported, the verdict is a boolean, and the e-folding time of the LEADING
    (slowest-damped) mode is given in days: tau_days / -ln|lambda_max|.
    """
    ev = np.linalg.eigvals(np.asarray(G, np.float64))
    mod = np.abs(ev)
    rho = float(mod.max()) if mod.size else 0.0
    tau_days = float(step_days) * tau
    stable = bool(rho < 1.0)
    efold = tau_days / (-math.log(rho)) if (stable and rho > 0) else None
    return {"spectral_radius": round(rho, 6),
            "stable": stable,
            "n_modes_unstable": int((mod >= 1.0).sum()),
            "leading_efolding_days": (None if efold is None
                                      else round(efold, 3)),
            "tau_steps": tau, "tau_days": round(tau_days, 4),
            "note": ("|lambda| < 1 for every mode means the forecast relaxes "
                     "to the training mean (climatology in this standardized "
                     "space) as the lead grows, which is what makes "
                     "msss_clim approach 0 rather than diverge. A radius at "
                     "or above 1 invalidates every long-lead row in this "
                     "entry.")}


def assert_finite(obj, where="results"):
    """ml/CLAUDE.md §5.22: never write NaN into a results file — stop instead.

    An UNSTABLE propagator is caught by `lim_diagnostics` and never scored, so
    in normal operation nothing here can fire. It exists because the failure
    it guards is exactly the one §5.22 was written about: `msss_clim` of a
    diverging trajectory overflows float32 at around lead 40, `1 - inf/mc`
    rounds to `-inf`, and `json.dump` writes the literal `-Infinity` — a file
    that parses in Python, fails in every JSON reader that is not Python, and
    is loud enough to notice only if somebody looks.
    """
    stack = [(where, obj)]
    while stack:
        p, o = stack.pop()
        if isinstance(o, dict):
            for k, v in o.items():
                stack.append((f"{p}.{k}", v))
        elif isinstance(o, (list, tuple)):
            for i, v in enumerate(o):
                stack.append((f"{p}[{i}]", v))
        elif isinstance(o, float) and not math.isfinite(o):
            sys.exit(f"non-finite value {o!r} at {p} — refusing to write it "
                     f"(ml/CLAUDE.md §5.22). A results file full of NaN is "
                     f"loud enough to notice and quiet enough to "
                     f"misattribute.")


class SubsetField:
    """`StdMonths` restricted to a subset of the P pixel rows.

    `--scope-fit corridor` fits the state on the corridor only; the training
    matrix builder and the predictor both want that subset while everything
    else keeps indexing the full P. The cache is SHARED (the same dict
    object), so popping through either view frees the same row.
    """

    def __init__(self, inner, sel):
        self.inner, self.sel = inner, np.asarray(sel)
        self.cache = inner.cache

    def get(self, t):
        v, o = self.inner.get(t)
        return v[self.sel], o[self.sel]


class LimPredictor:
    """`predict(s)` -> a generator yielding xhat [P, C] for h = 1, 2, 3, ...

    The generator shape is deliberate: at the production pentad size one
    trajectory is 73 x [86698, 40] float32 = 1.0 GB if materialised, and the
    battery consumes each lead exactly once. The [P, C] buffer is reused
    between yields for the same reason; `accumulate` copies out of it (its
    arguments are boolean-masked, hence copies) before the next `next()`.

    Channels OUTSIDE the modelled set are left at zero, and pixels outside
    `fit_idx` (only under `--scope-fit corridor`) likewise. That is safe by
    construction rather than by hope for the channels — they are exactly the
    ones with no observation at any start row, so `op` is identically false on
    them, and `score_battery`'s `unmodelled` check refuses if it ever is not —
    and it is handled by omitting the scope entirely for the pixels.

    `V` must be C-contiguous [D, k]: a `V[:, :k]` slice of a wider array is
    strided, and BLAS would copy the whole thing on EVERY one of the ~660
    roll steps. `main` makes the contiguous leading block once per K.
    """

    def __init__(self, std_m, mu, V, G, ch_idx, P, C, fit_idx=None):
        self.std_m, self.mu, self.G = std_m, mu, G
        self.V = np.ascontiguousarray(V)
        self.ch_idx = np.asarray(ch_idx)
        self.P, self.C = int(P), int(C)
        self.fit_idx = None if fit_idx is None else np.asarray(fit_idx)
        self.n_fit = self.P if self.fit_idx is None else len(self.fit_idx)
        self.k = G.shape[0]

    def a0(self, s):
        """The start row's TRUE standardized state, projected onto the EOFs."""
        v0, _ = self.std_m.get(int(s))
        if self.fit_idx is not None:
            v0 = v0[self.fit_idx]
        x0 = np.ascontiguousarray(v0[:, self.ch_idx]).reshape(-1)
        return self.V.T @ (x0 - self.mu)

    def __call__(self, s):
        a = np.asarray(self.a0(s), np.float64)
        xhat = np.zeros((self.P, self.C), np.float32)
        nch = len(self.ch_idx)
        while True:
            a = self.G @ a
            f = (self.mu + (self.V @ a.astype(np.float32))).reshape(
                self.n_fit, nch)
            if self.fit_idx is None:
                xhat[:, self.ch_idx] = f
            else:
                xhat[np.ix_(self.fit_idx, self.ch_idx)] = f
            yield xhat


# ------------------------------------------------------------ the battery --
def score_battery(std_m, ax, hold_years, starts_per_year, Hh, T, r1, scopes,
                  C, chan_names, predict, prog=None, unmodelled=None,
                  pop_cache=True):
    """Roll `predict` from the protocol's staggered starts and accumulate the
    battery's own sums. Returns `(sums, info)`.

    THIS IS `rollout_spatial.main()`'s SCORING LOOP, structurally line for
    line: the same holdout BLOCKS (`hold_blocks`, E-067), the same start list
    (`TimeAxis.starts_for_block`), the same break (`t_tgt >= T or
    ax.year[t_tgt]` outside the block — the condition `scored_horizon`
    encodes, which is why that function is imported and used to plan the
    progress total from the identical expression), the same persistence
    (`std_m.get(s)`), the same damped persistence (`v_pers * r1 ** h`), the
    same observation mask (`obs_target & obs_start`) and the same
    `accumulate` on the same masked scopes. Only the source of `xhat` differs,
    which is the entire experiment.

    `pop_cache` drops each standardized row from `StdMonths`' cache once it
    has been scored. The head path can afford to keep them; here the training
    pass has already spent the memory budget, and the rows are cheap to
    re-read from the memmap.
    """
    sums = {name: new_sums(Hh, C) for name, _ in scopes}
    n_starts, n_steps = 0, 0
    unmod = None if unmodelled is None else np.asarray(unmodelled)
    if unmod is not None and unmod.size == 0:
        unmod = None
    for B in hold_blocks(hold_years):
        for s in ax.starts_for_block(B, starts_per_year):
            # The head path additionally requires `s - K + 1 >= 0` for its K
            # context rows. A LIM's context is ONE row (tau = 1 bin), so the
            # condition degenerates to the second half of the same guard.
            if s + 1 >= T:
                continue
            n_starts += 1
            v_pers, obs_s = std_m.get(s)
            gen = predict(s)
            try:
                for h in range(1, Hh + 1):
                    t_tgt = s + h
                    # `scored_horizon`'s own break, written out — at the end
                    # of the BLOCK (E-067), which for a single held-out year
                    # is the year boundary it always was.
                    if t_tgt >= T or not (B[0] <= ax.year[t_tgt] <= B[1]):
                        break
                    xhat = next(gen)
                    v_true, obs_tt = std_m.get(t_tgt)
                    op = obs_tt & obs_s
                    v_damp = v_pers * r1 ** h
                    if unmod is not None:
                        bad = int(op[:, unmod].sum())
                        if bad:
                            names = ", ".join(str(chan_names[c])
                                              for c in unmod)
                            sys.exit(
                                "start row %d lead %d: %d observed cells on "
                                "channels this LIM does not model (%s). They "
                                "would be scored against a predicted ZERO "
                                "and would move every pooled number in this "
                                "artefact. The channel set is derived from "
                                "the START rows and `op` needs an "
                                "observation at the start too, so this "
                                "cannot happen unless the derivation "
                                "changed — refusing rather than reporting it "
                                "(ml/CLAUDE.md 0.2)." % (s, h, bad, names))
                    for name, m_ in scopes:
                        accumulate(sums[name], h, xhat[m_], v_true[m_],
                                   v_pers[m_], v_damp[m_], op[m_])
                    n_steps += 1
                    if prog is not None:
                        prog.step("skill")
                    if pop_cache:
                        std_m.cache.pop(t_tgt, None)
            finally:
                gen.close()
            if pop_cache:
                std_m.cache.pop(s, None)
    return sums, {"starts": n_starts, "scored_steps": n_steps}


# --------------------------------------------------------------- the data --
def observed_channels(Xm, ys, xs, rows, C, label="start"):
    """Bool [C]: which channels are finite at ANY of `rows`, at ocean pixels.

    Called with the START rows, so the answer is a strict SUPERSET of the
    channels that can ever enter a score (`op = obs_target & obs_start`). A
    superset costs state dimensions; a subset would silently drop a channel
    the battery does score, which is why the direction matters and why the
    complement is asserted at every step in `score_battery`.
    """
    obs = np.zeros(int(C), bool)
    for r in rows:
        x = np.asarray(Xm[int(r)])[ys, xs]
        obs |= np.isfinite(x).any(axis=0)
    print(f"channel survey over {len(rows)} {label} row(s): "
          f"{int(obs.sum())}/{int(C)} channels observed", flush=True)
    return obs


def build_train_matrix(field, train_rows, n_px, ch_idx, every=200):
    """[len(train_rows), n_px*len(ch_idx)] float32 of standardized anomalies.

    Rows are pixel-major / channel-minor (`v[:, ch].reshape(-1)`), which is
    the layout `LimPredictor` inverts with `f.reshape(n_px, n_ch)`. The
    `StdMonths` cache is popped as we go: keeping 2,923 rows would be 51 GB on
    top of the 8.1 GB matrix they are being copied into.
    """
    ch_idx = np.asarray(ch_idx)
    D = int(n_px) * len(ch_idx)
    A = np.empty((len(train_rows), D), np.float32)
    t0 = time.time()
    for i, t in enumerate(train_rows):
        v, _ = field.get(int(t))
        A[i] = np.ascontiguousarray(v[:, ch_idx]).reshape(-1)
        field.cache.pop(int(t), None)
        if every and (i + 1) % every == 0:
            el = time.time() - t0
            left = el / (i + 1) * (len(train_rows) - i - 1)
            print(f"  training matrix {i + 1}/{len(train_rows)} rows · "
                  f"{el / 60:.1f} min in · ~{left / 60:.1f} min left",
                  flush=True)
    print(f"training matrix: [{A.shape[0]}, {A.shape[1]}] float32 = "
          f"{A.nbytes / 1e9:.2f} GB in {(time.time() - t0) / 60:.1f} min",
          flush=True)
    return A


# ---------------------------------------------------------------- the toy --
def build_smoke_fixture(tmp):
    """A tiny pentad ocean + a channel/holdout descriptor, on disk.

    ml/CLAUDE.md §4.8: exercise the code path on a toy before spending the
    expensive resource. The "checkpoint" carries only what this file reads —
    `chan` and `args["holdout_years"]` / `args["holdout_lon"]` — because the
    LIM needs no codec weights, no embedding and no GPU, and standing up a
    real PixelMAE here would test torch rather than the LIM.
    """
    epoch = dt.date(1982, 1, 1)
    days, Hg, Wg, C = 5, 8, 10, 5
    b0 = (dt.date(1990, 1, 1) - epoch).days // days + 1
    T = 219                                   # three years of pentads
    bins = np.arange(b0, b0 + T, dtype=np.int64)
    months = np.array([(epoch + dt.timedelta(days=int(b) * days)).strftime(
        "%Y-%m") for b in bins])
    rng = np.random.default_rng(0)
    t = np.arange(T)[:, None, None, None]
    X = (np.sin(2 * np.pi * t / (365.2425 / days)) + 0.4 * (t / T)
         + 0.3 * rng.standard_normal((T, Hg, Wg, C))).astype(np.float32)
    X[:, 0, 0, :] = np.nan                    # land
    xpath = os.path.join(tmp, "X.npy")
    np.save(xpath, X)
    lats = np.linspace(20, 40, Hg).astype(np.float32)
    lons = np.linspace(-60, -40, Wg).astype(np.float32)
    ridx = np.array([r for r in range(T) if r % 7 != 3])
    rapid = np.stack([ridx.astype(float),
                      2.79 * rng.standard_normal(len(ridx))], 1)
    npz = os.path.join(tmp, "small.npz")
    np.savez(npz, months=months, lats=lats, lons=lons, rapid=rapid,
             bin_index=bins, pentad_days=np.array(days),
             cadence=np.array("pentad"), epoch=np.array(str(epoch)))
    ckpt = os.path.join(tmp, "pixelmae.pt")
    torch.save({"chan": [f"c{i}" for i in range(C)], "d_z": 4,
                "args": {"holdout_years": "1991", "holdout_lon": "0,0"}},
               ckpt)
    return {"x": xpath, "npz": npz, "ckpt": ckpt, "T": T, "C": C}


# --------------------------------------------------------------- read-out --
def _f(v):
    """A fixed-width cell that cannot blow the table's columns apart. A skill
    score lives in [-1, 1] when the model is sane; anything outside it is
    printed in exponent form so a broken row LOOKS broken instead of shifting
    every column to its right."""
    if v is None:
        return "--"
    return "%+.3f" % v if abs(v) < 100 else "%+.1e" % v


def print_table(results, leads):
    """A compact stdout table: msss_clim per scope at the day-matched leads,
    plus the two AUCs. The artefact is the record; this is what a reader sees
    in the log while the box is still rented."""
    present = [s for s in ("gate", "corridor", "window")
               if any(s in e for e in results["heads"].values())]
    if not present:
        print("\n(no scope was scored)")
        return
    print("\nmsss_clim by lead (day-matched leads) — LIM vs climatology")
    head = ("  %-12s %-9s" % ("entry", "scope")
            + "".join("%8s" % ("h%d" % h) for h in leads)
            + "%9s%9s" % ("AUC", "AUCdm"))
    print(head)
    print("  " + "-" * (len(head) - 2))
    for label, e in results["heads"].items():
        for sc in present:
            blk = e.get(sc)
            if not blk or not blk.get("chan_skill"):
                continue
            by_h = {r["h"]: r["msss_clim"] for r in blk["chan_skill"]}
            cells = "".join("%8s" % _f(by_h.get(h)) for h in leads)
            print("  %-12s %-9s%s%9s%9s" % (
                label, sc, cells, _f(blk.get("horizon_auc")),
                _f(blk.get("horizon_auc_daymatched"))))
    for label, e in results["heads"].items():
        ei = e["meta"]["eigen"]
        print("  %s: |lambda|max %.4f (%s) · leading e-folding %s d · "
              "variance explained %s%s" % (
                  label, ei["spectral_radius"],
                  "stable" if ei["stable"] else "UNSTABLE",
                  ei["leading_efolding_days"],
                  e["meta"]["variance_explained"],
                  " · NOT SCORED" if e["meta"].get("unscored") else ""))
    m = results["model"]
    print("  fitted on %s training bins x %s state dims (%s px x %d ch, "
          "scope %s), tau %s d" % (
              f"{m['train_bins']:,}", f"{m['state_dim']:,}",
              f"{m['fit_pixels']:,}", len(m["channels"]), m["scope_fit"],
              m["tau_days"]))
    if m["scopes_unavailable"]:
        print("  UNAVAILABLE scopes: %s — %s" % (
            ", ".join(m["scopes_unavailable"]), m["scopes_unavailable_why"]))


# ---------------------------------------------------------------- the run --
def build_parser():
    ap = argparse.ArgumentParser(
        description="Linear Inverse Model baseline, scored through "
                    "ml/rollout_spatial.py's own battery")
    ap.add_argument("--x", default="", help="the tensor memmap (.npy)")
    ap.add_argument("--npz-small", default="",
                    help="the tensor .npz (months/lats/lons/rapid and, at a "
                         "binned cadence, bin_index/pentad_days/epoch)")
    ap.add_argument("--ckpt", default="",
                    help="the CODEC checkpoint — read ONLY for its `chan` "
                         "names and its `args` (holdout_years, holdout_lon). "
                         "No weights are loaded, no embedding is computed and "
                         "no GPU is used; the LIM works on pixels.")
    ap.add_argument("--hold-years", default="",
                    help="comma list of years to hold out, OVERRIDING the "
                         "checkpoint's `holdout_years` for the whole run — "
                         "the t_hold mask behind the anomaly statistics and "
                         "the LIM's own training pairs, the start list, the "
                         "roll's truncation and the `hold_years` written into "
                         "the artefact. E-067: CONSECUTIVE years group into "
                         "blocks (`hold_blocks`) and a roll is truncated at "
                         "the end of its BLOCK, so "
                         "--hold-years 2008,2009,2016,2017,2022,2023 is three "
                         "two-year blocks and 146 pentads (730 d) of lead are "
                         "scoreable where three single years cap every roll "
                         "at 365 d. REFUSED unless it is a SUPERSET of the "
                         "checkpoint's own years: the LIM may be denied more "
                         "than the head was, never less. Empty (the default) "
                         "takes the checkpoint's list unchanged.")
    ap.add_argument("--K", default="50,100,200",
                    help="comma-separated PC counts. EVERY K in the list is "
                         "fitted and scored — the decomposition is computed "
                         "once and a K is then a leading block of it, so the "
                         "marginal cost of another K is one roll, not another "
                         "SVD. A K above the available rank is clamped and "
                         "the entry records both numbers.")
    ap.add_argument("--horizon", type=int, default=73,
                    help="rolled horizon in AXIS STEPS (73 = 365.0 d at "
                         "pentad, the day-matched value; 12 at monthly)")
    ap.add_argument("--starts", type=int, default=3,
                    help="staggered starts per holdout year — "
                         "TimeAxis.starts_for_block's `per_year`, counted "
                         "PER BLOCK. 0 = all")
    ap.add_argument("--scope-fit", choices=("window", "corridor"),
                    default="window",
                    help="which pixel set the state vector covers. `window` "
                         "(all ocean pixels) scores all nine scopes; "
                         "`corridor` cuts the state to the corridor and "
                         "reports gate/window as UNAVAILABLE rather than "
                         "scoring pixels the model has no state for")
    ap.add_argument("--out",
                    default=os.path.join(HERE, "runs", "actions",
                                         "rollout_spatial.json"),
                    help="where scripts/archive_probes.py expects a roll "
                         "result")
    ap.add_argument("--pixels-gate", type=int, default=600)
    ap.add_argument("--corridor-pctl", type=float, default=75.0)
    ap.add_argument("--corridor-dilate", type=int, default=2)
    ap.add_argument("--max-state-gb", type=float, default=24.0,
                    help="refuse before allocating a training matrix larger "
                         "than this. The guard sits where the inputs are all "
                         "it has cost (ml/CLAUDE.md 0.3), not after an hour "
                         "of memmap reads")
    ap.add_argument("--cache-dir", default=os.path.join(HERE, "cache"),
                    help="shared with ml/rollout_spatial.py: the same "
                         "ocean_mask.npy and std_stats.npz, so the anomaly "
                         "statistics behind a LIM row and a head row are the "
                         "same bytes")
    ap.add_argument("--metrics", default=os.path.join(HERE, "runs", "actions",
                                                      "metrics.jsonl"),
                    help="progress records are APPENDED here in "
                         "rollout_spatial's own `sroll` record shape, so "
                         "scripts/publish_live_metrics.sh and the status page "
                         "need no new format")
    ap.add_argument("--progress-every", type=int, default=20)
    ap.add_argument("--smoke", action="store_true",
                    help="generate a tiny pentad tensor and run the whole "
                         "path on it (<1 min, CPU). --x/--npz-small/--ckpt "
                         "are ignored")
    return ap


def main(argv=None):
    a = build_parser().parse_args(argv)
    t_wall = time.time()

    if a.smoke:
        tmpdir = tempfile.mkdtemp(prefix="lim_smoke_")
        f = build_smoke_fixture(tmpdir)
        a.x, a.npz_small, a.ckpt = f["x"], f["npz"], f["ckpt"]
        a.cache_dir = os.path.join(tmpdir, "cache")
        if a.out == os.path.join(HERE, "runs", "actions",
                                 "rollout_spatial.json"):
            a.out = os.path.join(tmpdir, "rollout_spatial.json")
        if a.metrics == os.path.join(HERE, "runs", "actions",
                                     "metrics.jsonl"):
            a.metrics = os.path.join(tmpdir, "metrics.jsonl")
        print(f"--smoke: toy tensor in {tmpdir}", flush=True)
    missing = [f"--{k.replace('_', '-')}" for k in ("x", "npz_small", "ckpt")
               if not getattr(a, k)]
    if missing:
        sys.exit(f"missing {', '.join(missing)} (required unless --smoke)")

    Ks_req = []
    for tok in str(a.K).split(","):
        tok = tok.strip()
        if not tok:
            continue
        if not tok.isdigit() or int(tok) < 1:
            sys.exit(f"--K wants whole numbers of modes, got {tok!r}")
        Ks_req.append(int(tok))
    if not Ks_req:
        sys.exit("--K is empty")

    # ---- the axis first, where nothing has been spent (ml/CLAUDE.md 0.3) --
    d = np.load(a.npz_small, allow_pickle=False)
    ax = TimeAxis(d)
    months, lats, lons, T = ax.labels, d["lats"], d["lons"], ax.T
    moy = ax.moy
    print(ax.describe(), flush=True)

    ck = torch.load(a.ckpt, map_location="cpu", weights_only=False)
    C = len(ck["chan"])
    chan_names = [str(c) for c in ck["chan"]]
    ck_hold_years = sorted(ck["args"]["holdout_years"].split(","))
    if a.hold_years:
        hold_years = sorted({str(y).strip() for y in a.hold_years.split(",")
                             if str(y).strip()})
        missing = [y for y in ck_hold_years if y not in set(hold_years)]
        if missing:
            # THE DIRECTION MATTERS, and the guard sits where the inputs are
            # all it has cost (ml/CLAUDE.md 0.3). Dropping a checkpoint
            # holdout year would fit the LIM on bins the head it is a
            # baseline for was denied — the one thing this override must
            # never be able to do.
            sys.exit(
                f"--hold-years {a.hold_years!r} is not a superset of "
                f"{os.path.basename(a.ckpt)} args['holdout_years'] = "
                f"{ck['args']['holdout_years']!r}: {', '.join(missing)} "
                f"would re-enter the LIM's training pairs. The LIM may be "
                f"denied MORE than the head was, never less. Refusing.")
        hold_src = f"--hold-years {a.hold_years!r} (override)"
    else:
        hold_years = ck_hold_years
        hold_src = (f"{os.path.basename(a.ckpt)} args['holdout_years'] = "
                    f"{ck['args']['holdout_years']!r}")
    # E-067: the years, grouped into the runs a roll is truncated at.
    blocks = hold_blocks(hold_years)
    t_hold = np.array([m[:4] in set(hold_years) for m in months])
    lo, hi = (float(v) for v in ck["args"]["holdout_lon"].split(","))
    x_hold = (lons >= lo) & (lons < hi)
    print(f"holdout years: {','.join(hold_years)} — from {hold_src} · "
          f"blocks {', '.join(block_label(b) for b in blocks)} · "
          f"{int(t_hold.sum())} of {T} bins held out", flush=True)
    if not t_hold.any():
        sys.exit("no bin is in a holdout year: this LIM would be scored on "
                 "its own training set. Refusing.")

    Hh = int(a.horizon)
    h_match = ax.steps_for_months(12)
    if Hh != h_match:
        print(f"::warning::--horizon {Hh} is {ax.span_days(Hh):g} d at "
              f"{ax.cadence} cadence, NOT the monthly archive's 12 months "
              f"({ax.span_days(h_match):g} d = --horizon {h_match}). "
              f"`horizon_auc` here is not comparable with any archived "
              f"corridor AUC; `horizon_auc_daymatched` is.", flush=True)

    # ---- pixels, exactly the roll's ---------------------------------------
    Xm = np.load(a.x, mmap_mode="r")
    Hg, Wg = Xm.shape[1], Xm.shape[2]
    om_path = os.path.join(a.cache_dir, "ocean_mask.npy")
    ocean = np.load(om_path) if os.path.exists(om_path) else None
    if ocean is None:
        ocean = np.zeros((Hg, Wg), bool)
        for t0 in range(0, T, 16):
            ocean |= np.isfinite(np.asarray(Xm[t0:t0 + 16, :, :, 0])).any(0)
        os.makedirs(a.cache_dir, exist_ok=True)
        np.save(om_path, ocean)
    ys, xs = np.where(ocean)
    P = len(ys)
    sec_y, sec_sel = rapid_section(lats, lons, ys, xs)
    print(f"window: {P} ocean px · section {len(sec_sel)} px at row {sec_y}",
          flush=True)

    # ---- the anomaly transform, from the SAME cache the roll writes -------
    st_path = os.path.join(a.cache_dir, "std_stats.npz")
    # …AND KEYED ON THE CODEC'S OWN YEARS TOO (2026-09-02, the terminal
    # codec): the flat name was written by every run on the 2009/2017/2023
    # codecs, and a box that rolled one of those (Virginia, #523) still holds
    # that file when the first roll on a codec holding out 2021-2024 starts.
    # Reusing it would standardise the terminal years with statistics that
    # include them. So the flat name is reserved for the legacy year set that
    # wrote it, bit for bit; any other codec gets the blocks in the name,
    # whether the years come from an override or from the checkpoint itself.
    if not a.hold_years and set(hold_years) != {"2009", "2017", "2023"}:
        st_path = os.path.join(a.cache_dir, "std_stats__hold-%s.npz"
                               % "-".join(block_label(b) for b in blocks))
        print(f"anomaly stats: this codec's holdout years are not the legacy "
              f"2009/2017/2023 that wrote the shared std_stats.npz — using "
              f"{os.path.basename(st_path)}", flush=True)
    if a.hold_years:
        # E-067 · THE STATS CACHE IS NOT KEYED ON THE HOLDOUT YEARS, and this
        # directory is SHARED with ml/rollout_spatial.py precisely so a LIM
        # row and a head row rest on the same bytes. A warm box would
        # therefore hand a --hold-years run the climatology of the codec's
        # years — the same numbers, a different meaning. The override reads
        # and writes its own file, named for the blocks; the shared one is
        # neither read nor overwritten. Absent the flag, unchanged.
        st_path = os.path.join(a.cache_dir, "std_stats__hold-%s.npz"
                               % "-".join(block_label(b) for b in blocks))
        print(f"anomaly stats: --hold-years gets its own cache "
              f"{os.path.basename(st_path)} — the shared std_stats.npz was "
              f"written under the checkpoint's years and is left alone",
              flush=True)
    if os.path.exists(st_path):
        s_ = np.load(st_path)
        clim, dyn = s_["clim"], list(s_["dyn"])
        mean_c, std_c = s_["mean_c"], s_["std_c"]
        print(f"anomaly stats: reused {st_path}", flush=True)
    else:
        clim, dyn, mean_c, std_c = stream_stats(Xm, moy, t_hold, x_hold)
        os.makedirs(a.cache_dir, exist_ok=True)
        np.savez(st_path, clim=clim, dyn=np.array(dyn),
                 mean_c=mean_c, std_c=std_c)
    std_m = StdMonths(Xm, ys, xs, moy, clim, dyn, mean_c, std_c)

    print("AR1 damped-persistence pass over the record ...", flush=True)
    t_ar = time.time()
    r1 = ar1_train(std_m, T, t_hold, P, C)                       # [P, C]
    t_ar = time.time() - t_ar

    # ---- the scopes, exactly the roll's -----------------------------------
    corridor, cor_thr = corridor_pixels(Xm, ocean, ys, xs, t_hold, sec_sel,
                                        a.corridor_pctl, a.corridor_dilate)
    gate_mask = gate_subset(P, a.pixels_gate, sec_sel)
    sec_mask = np.zeros(P, bool)
    sec_mask[sec_sel] = True
    base_scopes = (("gate", gate_mask), ("corridor", corridor),
                   ("window", np.ones(P, bool)))
    px_hold = x_hold[xs]
    all_scopes = nested_scopes(base_scopes, px_hold)
    corridor_def = {"pctl": a.corridor_pctl, "threshold": round(cor_thr, 4),
                    "dilate_cells": a.corridor_dilate,
                    "structuring": "3x3 square",
                    "n_px": int(corridor.sum()), "of": P,
                    "union_section": True}
    print(f"scopes: gate {int(gate_mask.sum())} px · corridor "
          f"{int(corridor.sum())} px (cur_speed >= p{a.corridor_pctl:g} = "
          f"{cor_thr:.3f}, dilate {a.corridor_dilate}) · window {P} px",
          flush=True)

    # THE PIXELS THE STATE VECTOR COVERS. A scope with pixels outside the
    # fitted set is NOT scored and is NOT written as an empty block: an empty
    # block reads as "had pixels, scored nothing, investigate"
    # (skill_block's own `empty` note), and this is a different fact — the
    # model has no state there at all. Absent, and named as absent.
    if a.scope_fit == "corridor":
        fit_px = corridor.copy()
        scopes = tuple((n, m) for n, m in all_scopes
                       if n == "corridor" or n.startswith("corridor_"))
        kept = {n for n, _ in scopes}
        unavailable = [n for n, _ in all_scopes if n not in kept]
    else:
        fit_px = np.ones(P, bool)
        scopes = all_scopes
        unavailable = []
    n_fit = int(fit_px.sum())
    fit_idx = np.where(fit_px)[0]

    # ---- the starts, and the channels they make scoreable -----------------
    start_rows = [int(s) for B in blocks
                  for s in ax.starts_for_block(B, a.starts)
                  if int(s) + 1 < T]
    if not start_rows:
        sys.exit(f"no usable starts for holdout years {hold_years} on a "
                 f"{ax.cadence} axis of {T} rows — nothing to score")
    obs_ch = observed_channels(Xm, ys, xs, start_rows, C)
    ch_idx = np.where(obs_ch)[0]
    unmodelled = np.where(~obs_ch)[0]
    if ch_idx.size == 0:
        sys.exit("no channel is observed at any start row — nothing to model")
    print("modelled channels (%d): %s" % (
        len(ch_idx), ", ".join(chan_names[c] for c in ch_idx)), flush=True)
    if unmodelled.size:
        print("NOT modelled (%d, unobserved at every start row, so `op` is "
              "identically false on them and they cannot enter any number): "
              "%s" % (len(unmodelled),
                      ", ".join(chan_names[c] for c in unmodelled)),
              flush=True)

    train_rows = np.where(~t_hold)[0]
    D = n_fit * len(ch_idx)
    need_gb = len(train_rows) * D * 4 / 1e9
    print(f"state: D = {n_fit:,} px x {len(ch_idx)} ch = {D:,} · "
          f"T_train = {len(train_rows):,} bins · training matrix "
          f"{need_gb:.2f} GB float32", flush=True)
    if need_gb > a.max_state_gb:
        sys.exit(f"the training matrix would be {need_gb:.1f} GB, over "
                 f"--max-state-gb {a.max_state_gb:g}. Refusing BEFORE the "
                 f"read pass rather than after it (ml/CLAUDE.md 0.3). Cut it "
                 f"with --scope-fit corridor ({int(corridor.sum()):,} px "
                 f"instead of {P:,}), fewer channels, or raise the budget if "
                 f"the box has the RAM.")
    n_pairs_avail = len(lim_pairs(train_rows))
    k_avail = int(min(len(train_rows) - 1, D, max(n_pairs_avail - 1, 0)))
    if k_avail < 1:
        sys.exit(f"only {n_pairs_avail} consecutive training pairs and "
                 f"{len(train_rows)} training bins — nothing can be fitted")
    Ks, seen = [], set()
    for k in Ks_req:
        ke = min(k, k_avail)
        if ke != k:
            print(f"::warning::K={k} exceeds the {k_avail} modes this "
                  f"training set can carry (min(T_train-1, D, pairs-1)) — "
                  f"clamped to {ke}", flush=True)
        if ke in seen:
            print(f"  (K={k} clamps onto an entry already scored — skipped)",
                  flush=True)
            continue
        seen.add(ke)
        Ks.append((k, ke))
    k_max = max(ke for _, ke in Ks)

    # ---- PCA --------------------------------------------------------------
    fit_src = std_m if a.scope_fit == "window" else SubsetField(std_m, fit_idx)
    t_mat = time.time()
    A = build_train_matrix(fit_src, train_rows, n_fit, ch_idx)
    t_mat = time.time() - t_mat
    print(f"PCA: Gram [{len(train_rows)}, {len(train_rows)}] -> up to "
          f"{k_max} modes ...", flush=True)
    t_pca = time.time()
    mu, V, scores, sv, tot_var = pca_gram(A, k_max)
    del A
    t_pca = time.time() - t_pca
    if V.shape[1] < k_max:
        print(f"::warning::the decomposition kept {V.shape[1]} non-null "
              f"modes of the {k_max} asked for; larger K values clamp to it",
              flush=True)
        Ks = sorted({(k, min(ke, V.shape[1])) for k, ke in Ks},
                    key=lambda p: p[1])
    print(f"PCA: {V.shape[1]} modes in {t_pca / 60:.1f} min · EOF patterns "
          f"{V.nbytes / 1e9:.2f} GB", flush=True)

    # ---- the artefact skeleton -------------------------------------------
    _lon_any = bool(x_hold.any())
    holdout_lon = {
        "arg": str(ck["args"]["holdout_lon"]), "lo": lo, "hi": hi,
        "any": _lon_any,
        "rule": "(lons >= lo) & (lons < hi), train.py's own expression",
        "n_cols": int(x_hold.sum()), "of_cols": int(len(lons)),
        "px": {name: {"in_block": int((m_ & px_hold).sum()),
                      "of": int(m_.sum())}
               for name, m_ in base_scopes + (("section", sec_mask),)},
        "note": ("THE LIM SAW THESE COLUMNS. Unlike the head, whose stage-1 "
                 "and stage-2 pools both exclude the block, this baseline is "
                 "fitted over every pixel of the scope it covers — so its "
                 "`_holdlon` children are IN-SAMPLE for it and are NOT a "
                 "comparable extrapolation test against a head's. Read "
                 "`_trainlon`."
                 if _lon_any else
                 "NO longitude is held out of training by this codec, so "
                 "every scope aggregate is already the trained-pixel number "
                 "and the *_holdlon children are empty by construction."),
    }
    for _k, _v in holdout_lon["px"].items():
        _v["frac"] = round(_v["in_block"] / _v["of"], 4) if _v["of"] else None

    leads = ax.daymatched_leads()
    gate_reason = (
        "ml/lim_baseline.py is not a stage-2 head roll: there is no "
        + GATE_HEAD + " head to compare against #217. The BATTERY is "
        "identical — rollout_spatial.py's own StdMonths, ar1_train, "
        "corridor_pixels, gate_subset, nested_scopes, "
        "TimeAxis.starts_for_year, new_sums, accumulate and skill_block, "
        "imported and called on the same arrays — but the validation GATE is "
        "a property of a head roll, so it is recorded here as NOT TAKEN "
        "rather than as passed.")
    results = {
        "data": os.path.basename(a.x),
        "horizon": Hh,
        "hold_years": hold_years,
        "holdout_lon": holdout_lon,
        "corridor_def": corridor_def,
        "gate_ref": {"head": GATE_HEAD, "tol": GATE_TOL,
                     "cadence": ax.cadence, "reference": None,
                     "monthly_reference": dict(GATE_REF),
                     "reason": gate_reason},
        "gate": {"pass": None, "skipped": True, "certified": False,
                 "cadence": ax.cadence, "reason": gate_reason},
        "model": {
            "model": "lim",
            "family": "linear inverse model (Penland & Sardeshmukh 1995)",
            "written_by": "ml/lim_baseline.py",
            "written_at": _utc_now(),
            "space": ("the standardized-anomaly field ml/rollout_spatial.py "
                      "scores: per-pixel per-channel, de-climatologised "
                      "against the TRAIN-year monthly climatology and "
                      "z-scored, nan->0 at unobserved cells (StdMonths)"),
            "unobserved_cells": ("enter the covariance as 0, which is the "
                                 "value StdMonths hands every consumer of "
                                 "this space, the codec's own slab included"),
            "tau_steps": TAU_STEPS,
            "tau_days": round(ax.step_days * TAU_STEPS, 4),
            "propagator": "G(tau) = C(tau) C(0)^-1 in PC space",
            "pairs_rule": ("consecutive axis rows whose BOTH bins are "
                           "training bins, so no pair steps across a "
                           "held-out year"),
            "reduction": ("PCA by the T x T Gram matrix (numpy eigh) — exact, "
                          "no oversampling parameter; sklearn is not "
                          "installed on the boxes"),
            "initial_condition": ("the TRUE standardized state at the start "
                                  "row, projected on the EOFs — the same row "
                                  "persistence is read from"),
            "K_requested": [k for k, _ in Ks],
            "K_effective": [ke for _, ke in Ks],
            "K_available": int(k_avail),
            "modes_kept": int(V.shape[1]),
            "scope_fit": a.scope_fit,
            "fit_pixels": n_fit,
            "window_pixels": int(P),
            "state_dim": int(D),
            "channels": [chan_names[c] for c in ch_idx],
            "channels_excluded": [chan_names[c] for c in unmodelled],
            "channels_rule": ("observed at ANY start row — a strict superset "
                              "of the channels `op = obs_target & obs_start` "
                              "can ever be true on; the complement is "
                              "ASSERTED to contribute zero at every scored "
                              "step, and the run refuses if it does not"),
            "train_bins": int(len(train_rows)),
            "train_bins_of": int(T),
            "train_first": months[int(train_rows[0])],
            "train_last": months[int(train_rows[-1])],
            "train_pairs": int(n_pairs_avail),
            "training_matrix_gb": round(need_gb, 3),
            "eof_patterns_gb": round(float(V.nbytes) / 1e9, 3),
            "scopes_unavailable": unavailable,
            "scopes_unavailable_why": (
                None if not unavailable else
                "--scope-fit %s: the state vector covers %d pixels and these "
                "scopes contain pixels outside it, so the model has no state "
                "to score them from. Absent, not empty."
                % (a.scope_fit, n_fit)),
            "timings_s": {"ar1": round(t_ar, 1),
                          "training_matrix": round(t_mat, 1),
                          "pca": round(t_pca, 1)},
        },
        "heads": {},
    }
    if a.starts > 0:
        results["starts"] = {
            "per_year": a.starts,
            "rule": ("every k-th start of the holdout year's list, "
                     "k = len(list)//N, first N — TimeAxis.starts_for_year, "
                     "the roll's own"),
            "available": {block_label(b): len(ax.starts_for_block(b))
                          for b in blocks},
            "rows": {block_label(b): ax.starts_for_block(b, a.starts)
                     for b in blocks},
            "labels": {block_label(b): [
                ax.label_of_row(s)
                for s in ax.starts_for_block(b, a.starts)] for b in blocks},
        }
    if not ax.monthly:
        results["cadence"] = {
            "name": ax.cadence, "step_days": ax.days,
            "steps_per_year": round(ax.steps_per_year, 4),
            "T": ax.T, "first": ax.labels[0], "last": ax.labels[-1],
            "detected_from": ax.detected_from,
            "horizon_steps": Hh,
            "horizon_span_days": ax.span_days(Hh),
            "horizon_daymatched_steps": h_match,
            "horizon_is_daymatched": Hh == h_match,
            "starts_per_year": a.starts or "all",
            "starts_per_holdout_year": {
                block_label(b): len(ax.starts_for_block(b, a.starts))
                for b in blocks},
            "starts_available_per_holdout_year": {
                block_label(b): len(ax.starts_for_block(b)) for b in blocks},
            "band_edge_days": list(BAND_EDGE_DAYS),
            "daymatched_leads": list(leads),
            "daymatched_lead_days": [ax.span_days(h) for h in leads],
            "note": ("every `h` is an AXIS STEP, not a month. `horizon_auc` "
                     "averages msss_clim over h=1..horizon_steps and is a "
                     "function of THIS axis's lead sampling; "
                     "`horizon_auc_daymatched` averages the twelve leads "
                     "above and is the only one comparable with an archived "
                     "corridor AUC."),
        }

    # The step PLAN comes from `scored_horizon` — the roll's own expression
    # for how many of the Hh leads a start actually contributes.
    plan = sum(scored_horizon(ax, s, Hh, T, B) for B in blocks
               for s in ax.starts_for_block(B, a.starts) if s + 1 < T)
    prog = Progress(a.metrics, len(Ks), every=a.progress_every)
    print(f"plan: {len(Ks)} K value(s) x {plan} scored steps "
          f"({len(start_rows)} starts, horizon {Hh})", flush=True)
    unstable = []

    # ---- fit, roll and score, one entry per K -----------------------------
    for i, (k_req, k_eff) in enumerate(Ks, 1):
        label = f"lim_k{k_eff}"
        prog.start_head(i, label, plan)
        t_k = time.time()
        G, n_pairs = lim_propagator(scores, train_rows, k_eff)
        diag = lim_diagnostics(G, ax.step_days)
        var_k = (float((scores[:, :k_eff] ** 2).sum()) / tot_var
                 if tot_var > 0 else None)
        verdict = ("STABLE" if diag["stable"] else
                   "UNSTABLE — %d mode(s) at or above 1"
                   % diag["n_modes_unstable"])
        print("%s: G %s from %d consecutive training pairs · spectral radius "
              "%.4f (%s) · leading e-folding %s d · variance explained %s"
              % (label, G.shape, n_pairs, diag["spectral_radius"], verdict,
                 diag["leading_efolding_days"],
                 None if var_k is None else round(var_k, 4)), flush=True)
        base_meta = {"model": "lim", "K": k_eff, "K_requested": k_req,
                     "tau_steps": TAU_STEPS,
                     "tau_days": round(ax.step_days * TAU_STEPS, 4),
                     "n_train_pairs": n_pairs,
                     "train_bins": int(len(train_rows)),
                     "state_dim": int(D), "scope_fit": a.scope_fit,
                     "variance_explained": (None if var_k is None
                                            else round(var_k, 4)),
                     "eigen": diag}
        if not diag["stable"]:
            # NOT SCORED, and the entry says so. Rolling an undamped
            # propagator 73 steps overflows float32 well before the horizon,
            # so every row it produced would be `-inf` — which ml/CLAUDE.md
            # §5.22 forbids putting in a results file at all. Reporting the
            # spectrum and refusing to score is the honest artefact: the
            # eigenvalue check IS the result for this K.
            print(f"::warning::{label}: spectral radius "
                  f"{diag['spectral_radius']:.4f} >= 1 — the propagator is "
                  f"not damped, so it diverges rather than relaxing to "
                  f"climatology. NOT SCORED (§5.22); the entry carries its "
                  f"spectrum and no skill rows.", flush=True)
            unstable.append(k_eff)
            results["heads"][label] = {
                "meta": dict(base_meta, unscored=True, unscored_reason=(
                    "the fitted propagator has spectral radius %.6f (>= 1) "
                    "with %d mode(s) at or above 1: rolled %d steps it "
                    "diverges instead of relaxing to climatology, and every "
                    "skill row would be an overflow rather than a forecast. "
                    "Scoring was refused (ml/CLAUDE.md 5.22). Use a smaller "
                    "K: the instability is a rank problem, not a physics "
                    "one — the last modes are noise directions the lag-1 "
                    "regression fits with a gain above 1."
                    % (diag["spectral_radius"], diag["n_modes_unstable"],
                       Hh))),
                "wall_s": round(time.time() - t_k, 1)}
            assert_finite(results)
            write_results(a.out, results,
                          partial=({"model": label, "k_i": i, "ks": len(Ks),
                                    "stage": "unstable"}
                                   if i < len(Ks) else None))
            continue
        # A CONTIGUOUS leading block, once per K: `V[:, :k]` is strided and
        # BLAS would copy the whole [D, k_max] array on every roll step.
        Vk = V if k_eff == V.shape[1] else np.ascontiguousarray(V[:, :k_eff])
        pred = LimPredictor(std_m, mu, Vk, G, ch_idx, P, C,
                            fit_idx=(None if a.scope_fit == "window"
                                     else fit_idx))
        sums, info = score_battery(std_m, ax, hold_years, a.starts, Hh, T, r1,
                                   scopes, C, chan_names, pred, prog=prog,
                                   unmodelled=unmodelled)
        del Vk
        entry = {"meta": dict(base_meta, starts=info["starts"],
                              scored_steps=info["scored_steps"])}
        for name, m_ in scopes:
            entry[name] = skill_block(sums[name], Hh, n_px=int(m_.sum()),
                                      leads=leads, chan_names=chan_names)
        entry["wall_s"] = round(time.time() - t_k, 1)
        results["heads"][label] = entry
        # 5.25: the numbers exist long before the job ends, so the file is
        # rewritten — atomically, marked partial — at every K boundary.
        assert_finite(results)
        write_results(a.out, results,
                      partial=({"model": label, "k_i": i, "ks": len(Ks),
                                "stage": "scored"} if i < len(Ks) else None))
        cb = entry.get("corridor", {})
        wb = entry.get("window", {})
        print("%s: corridor AUC %s (day-matched %s) · window %s · %.1f min"
              % (label, cb.get("horizon_auc"),
                 cb.get("horizon_auc_daymatched"), wb.get("horizon_auc"),
                 entry["wall_s"] / 60), flush=True)

    results["model"]["timings_s"]["total"] = round(time.time() - t_wall, 1)
    results["model"]["K_unstable_not_scored"] = unstable
    scored = [k for k in results["heads"]
              if not results["heads"][k]["meta"].get("unscored")]
    if not scored:
        sys.exit("every K produced an undamped propagator (spectral radius "
                 ">= 1) and none was scored. The artefact would carry no "
                 "skill row at all, so this run answers nothing — refusing "
                 "rather than archiving an empty result. Try smaller K.")
    assert_finite(results)
    write_results(a.out, results)
    print_table(results, leads)
    print("\nwrote %s (%s bytes) — %d LIM %s (%d scored, %d unstable) "
          "in %.1f min"
          % (a.out, f"{os.path.getsize(a.out):,}", len(results["heads"]),
             "entry" if len(results["heads"]) == 1 else "entries",
             len(scored), len(unstable), (time.time() - t_wall) / 60),
          flush=True)
    return results


if __name__ == "__main__":
    main()

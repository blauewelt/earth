#!/usr/bin/env python3
"""Predictive metrics on FROZEN embeddings — computable during training.

The question a reconstruction loss cannot answer is the one the project
exists for: do the embeddings, as they are RIGHT NOW, support prediction?
This module answers it cheaply enough to run every N training steps:

  · linear_probe — the 26.5°N section's single-month embeddings, ridge to
    the deseasonalised RAPID transport (protocol v2: train-years target
    climatology, lambda on a train tail, scored on held-out years only).
  · mini_temporal — freeze the codec, train a SMALL stage-2 transformer
    (subsampled pixels, short schedule, fixed seed) on the embeddings, and
    score it on held-out months: z-space t+1 vs persistence, decoded
    channel-space t+1 vs persistence, and the RAPID probe from the
    section-pooled hidden state. This is the user-requested metric: "freeze
    the existing embedding, train a transformer to use them to predict,
    compute the metrics with this."

Deterministic (seeded) so curves across checkpoints are comparable. The
mini transformer is deliberately small and short — it is a MEASUREMENT of
the embedding, not a model we keep; the full stage 2 lives in temporal.py.

CLI (backfill an existing run, writes runs/<run>/trainprobe.json):
    python3 ml/trainprobe.py --run pilot4_anom
"""
import argparse
import json
import os
import sys
import time
import warnings

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model import PixelMAE, codec_from_ckpt
from probe_sequence import ridge_r
from temporal import TemporalTransformer, embed_everything, rapid_section

HERE = os.path.dirname(os.path.abspath(__file__))

# ---------------------------------------------------------------------------
# HOW MANY RAPID SAMPLES THE LIGHT PROBE NEEDS — derived, not chosen.
#
# The light probe exists for ONE consumer: train.py's collapse guard, which
# asks `|linear_r_deseas| <= 0.05` twice in a row (ml/CLAUDE.md §1 — the only
# monitor that can see a dead codec, because a correlation is scale
# invariant). The full probe keeps carrying the trend at full resolution.
#
# The guard's expensive error is the FALSE ABORT: killing a healthy run. A
# healthy daily codec reads r ~ 0.58 (#419's linear_r_deseas ran 0.558-0.598
# across 200k steps), so the guard must resolve 0.58 away from 0.05 with
# margin. Fisher: sd(r) ~ (1 - r^2)/sqrt(m - 3) on m INDEPENDENT scored
# samples, so a 3-sigma margin needs
#
#     3 * 0.6636 / sqrt(m - 3) <= 0.53   ->   m >= 18
#
# eighteen effectively-independent held-out samples. That is the floor, and
# it is the only sample-count requirement the guard actually generates: the
# other direction (a collapsed codec reading ABOVE 0.05 by chance) is not a
# sampling question, because collapse does not produce a noisy r — it
# produces a DEGENERATE feature matrix. #387 read 0.000, 0.000, 0.000 to
# three decimals on three consecutive probes. What protects against a merely
# noisy collapse is the two-strike rule, not m; at 0.05 = k/sqrt(m-1) a
# 3-sigma line would need m = 3,601 SCORED samples and the daily tensor has
# only ~1,096 held-out RAPID days in total, so that bar is unreachable at any
# subsample and thinning cannot forfeit what was never there.
#
# EFFECTIVELY-INDEPENDENT is the operative word, and it is what makes the
# saving free. Measured on the archived series (data/rapid_moc.json, 729
# ten-day means, 2004-04-07 -> 2024-03-13 = exactly the 7,290 days the daily
# tensor's `rapid` array covers): the deseasonalised transport has lag-1
# autocorrelation +0.474 at 10 days, +0.225 at 20, +0.140 at 40. An AR(1)
# e-folding fitted to the 10-day lag-1 is
#
#     tau = -10 / ln(0.4743) = 13.4 days
#
# so two consecutive DAYS correlate at exp(-1/13.4) = 0.928: they are 93% the
# same number. Embedding all 7,290 daily RAPID timesteps to score a
# correlation buys ~560 samples' worth of information and pays for 7,290.
#
# So the light probe keeps one RAPID sample per e-folding, at whatever cadence
# the tensor runs: stride = floor(tau / dt). Daily -> 13 (561 kept, ~84 held
# out), pentad -> 2 (729 kept, ~109 held out), monthly -> 1 (no thinning at
# all: 30.4-day steps are already coarser than tau). Every one of those is
# four to six times the m >= 18 floor, and LIGHT_MIN_TEST re-checks it against
# the tensor in hand rather than trusting this arithmetic.
#
# This is the one change in the probe-cost work that is not free: the light
# probe's r is now a guard-grade estimate on a coarser sample, so it can sit a
# little below the full probe's. That is measured rather than assumed — the
# FULL probe emits `linear_r_deseas_light`, the identical estimator on the
# identical subsample, next to its own full-resolution number, at the cost of
# one extra ridge solve (~0.006% of a probe).
RAPID_TAU_DAYS = 13.4
LIGHT_MIN_TEST = 30          # kept held-out samples; 1.7x the m >= 18 floor


def cadence_days(months):
    """Median spacing, in days, between consecutive tensor timesteps.

    The `months` array is "YYYY-MM" on the monthly families and "YYYY-MM-DD"
    on the pentad and daily ones; both are handled, and a single-timestep or
    unparseable array falls back to 1.0 (which disables the light-probe
    thinning rather than guessing at it)."""
    import datetime as _dt
    ds = []
    for m in months[:64]:
        m = str(m)
        try:
            ds.append(_dt.date(int(m[:4]), int(m[5:7]),
                               int(m[8:10]) if len(m) >= 10 else 15))
        except (ValueError, IndexError):
            return 1.0
    if len(ds) < 2:
        return 1.0
    gaps = [(ds[i + 1] - ds[i]).days for i in range(len(ds) - 1)]
    return max(1.0, float(np.median(gaps)))


def light_rows(ridx, tr_all, te_all, months):
    """Which positions in `ridx` the LIGHT probe embeds and scores.

    Returns (positions, stride). See RAPID_TAU_DAYS above for the derivation;
    the floor is enforced against THIS tensor: if the stride would leave fewer
    than LIGHT_MIN_TEST held-out samples it is walked back until it does, so a
    short record or an unusual --holdout-years can never thin the guard's
    scored sample below what resolves 0.58 from 0.05."""
    dt = cadence_days(months)
    stride = max(1, int(RAPID_TAU_DAYS // dt))
    while stride > 1:
        sel = np.arange(0, len(ridx), stride)
        if int(te_all[sel].sum()) >= LIGHT_MIN_TEST:
            break
        stride -= 1
    return np.arange(0, len(ridx), stride), stride


def anomaly_transform(X, moy, t_hold, x_hold, chunk=64, verbose=None):
    """The one anomaly transform (train.py --anomaly), in one place: dynamic
    channels become departures from their own train-years monthly
    climatology, then z-scored on train data. Returns (X, dynamic).

    WHY THIS IS CHUNKED OVER TIME AND NOT OVER CHANNELS. The tensor is
    channel-interleaved [T, H, W, C], so `X[..., c]` has a stride of C*itemsize
    — 78 bytes at C=39 float16, an order of magnitude SMALLER than a 4 KB
    page. Every per-channel operation therefore has to fault in every page of
    the file to use 2.6% of what it reads. The original implementation looped
    per channel over the whole tensor: 39 traversals for the dynamic test,
    then ~6 more per dynamic channel (subtract clim read+write, isfinite,
    the boolean gather, the z-score read+write), i.e. ~249 full-extent
    traversals. Family 4 (pentad, 33.1 GB) on a 128 GB box holds the whole
    file in page cache, so those traversals are free after the first and this
    was never a problem. Family 5 (daily, 165.6 GB) on a 64 GB box is 2.6x
    oversubscribed — ZERO cache reuse, every traversal physical, ~41 TB of
    read. Run #389 sat seven hours in this function with the GPU at 0% and
    0.3 CPU cores and never emitted a metric line.

    So: iterate over BLOCKS OF THE TIME AXIS and do all channels inside each
    block. Three sequential passes, each one traversal:
      1. per-(t,c) spatial means (the dynamic test) and the monthly
         climatology's masked sum/count — both exactly chunkable, since the
         first is a per-timestep reduction and the second is a sum;
      2. write the anomaly (X - clim) and accumulate its count/mean/M2 over
         the valid pool;
      3. write (anomaly - mu) / (sd + 1e-6).
    Passes 2 and 3 are NOT folded: mu and sd must be over the whole array
    before any value is written. tests/test_anomaly_chunked.py check 3
    MEASURES the byte span both versions charge against X, at family 5's
    C=39 with 4 baked channels: **249.8 traversals before, 6.0 after**
    (41.6x; 40.4 TB -> 994 GB at 165.6 GB). Five of the six are physical —
    the sixth is pass 2 reading back pages it has just dirtied. Measured
    end to end on a 7.46 GiB float16 memmap at H=281 W=481 C=39, caches
    dropped, 1.6x oversubscribed: one cold traversal costs 28.1 s, so the
    old shape costs 249.8 * 28.1 s ~ 1.95 h; the new one ran in 352.8 s.

    Pass 2 stores the anomaly at the storage dtype and pass 3 reads it back,
    which is exactly what the two-step original did. Do not "optimise" that
    into a fused (X - clim - mu)/sd: at float16 the intermediate rounding is
    5e-4 and the result would stop matching the implementation every
    published number was produced with.

    EVERY reduction below names float64, and that is not cosmetic.
    Family 4 is the project's first float16 tensor (build_family4.py chose
    it to fit 33.1 GB rather than 66.3). numpy upcasts the accumulator for
    np.mean on float16 but NOT for np.std/np.var — _methods._var only
    upcasts integer and bool. The z-score at the end of this function sums
    ~204M squared residuals; in float16 that passes 65504, returns inf, and
    (X - mu) / (inf + 1e-6) is exactly 0.0. Every dynamic channel would
    become zeros while every loss, gpu_util and probe still looked healthy —
    and probe_kfold.py calls this function, so E-038's frozen control would
    have "falsified" its premise against an all-zero tensor. Family 3 was
    float32 and never reached the limit. Measured 2026-08-17: float16 pool
    of 30M -> inf; the same pool in float64 -> 0.999844. The float32 path
    moves by 5.2e-9 relative, ~7 orders below the sd 0.123 seed noise the
    stage-2 probe already carries (ml/CLAUDE.md §3).

    CHUNKING MAKES THAT OVERFLOW MORE DANGEROUS, NOT LESS, because the
    z-score's second moment is now a sum of PARTIAL sums and a partial sum
    that saturates is invisible in the total. Two consequences, both
    deliberate:
      · every accumulator here is float64 — the per-(t,c) spatial sums, the
        climatology sum, and the count/mean/M2 triple;
      · the variance is combined across blocks with CHAN'S PARALLEL
        ALGORITHM (Chan, Golub & LeVeque 1979) — each block's (n, mean, M2)
        is computed by numpy's own two-pass estimator in float64, and blocks
        are merged with
            delta = mean_b - mean;  n' = n + n_b
            mean' = mean + delta * n_b / n'
            M2'   = M2 + M2_b + delta^2 * n * n_b / n'
        rather than accumulating sum(x) and sum(x^2) and finishing with
        sum(x^2)/n - mu^2. The naive form catastrophically cancels when |mu|
        is large next to sd, and the anomaly's mu is only near zero because
        the climatology happens to nearly centre it — that is a property of
        the data we would be silently relying on. Chan's form has no such
        cancellation and needs no provisional offset to be chosen.

    `chunk` is in TIMESTEPS and bounds the working set independently of T.
    Arithmetic for the daily tensor (T=15706, H=281, W=481, C=39, float16),
    where one timestep is H*W*C = 5,271,279 elements:
      persistent  climatology sum    12 * 5,271,279 * 8 B = 0.47 GiB
                  climatology count  12 * 5,271,279 * 4 B = 0.24 GiB
      pass 1      finite mask        chunk * 5,271,279 * 1 B
      pass 2      climatology gather chunk * 5,271,279 * 4 B  (float32)
                  float64 block      chunk * 5,271,279 * 8 B
                  finite mask        chunk * 5,271,279 * 1 B
      pass 3      float64 block      chunk * 5,271,279 * 8 B
    Pass 2 is the peak, and the gather and the float64 block coexist in RSS
    even though the gather is dead by then — glibc does not return a 1.3 GiB
    arena and numpy cannot reuse a 1.3 GiB hole for a 2.5 GiB request. So
    take the peak as (13 * chunk * H * W * C) bytes plus 0.7 GiB. MEASURED
    peak RssAnon at H=281, W=481, C=39, polling /proc/self/status:
    **1.57 GiB at chunk=16 and 4.48 GiB at chunk=64** — the default is 7% of
    the 64 GB box, which leaves the page cache the room it needs to make
    these reads sequential in the first place. 64 timesteps is also 168 MB of
    contiguous file per read, far above any readahead window. Lower `chunk`
    on a small box; it changes only memory, never the answer
    (tests/test_anomaly_chunked.py check 2 pins that at chunk 1/7/64/T).
    """
    T, H, W, C = X.shape
    moy = np.asarray(moy)
    t_hold = np.asarray(t_hold, dtype=bool)
    x_hold = np.asarray(x_hold, dtype=bool)
    if verbose is None:
        # Silence is what made #389 unreadable: seven hours with no output at
        # all. Announce progress only when this is going to take a while.
        verbose = X.dtype.itemsize * X.size > 8 << 30
    chunk = max(1, int(chunk))
    nblk = (T + chunk - 1) // chunk

    def _say(msg):
        if verbose:
            print(f"  anomaly_transform: {msg}", flush=True)

    # The climatology is a mean over the timesteps of one month-of-year that
    # are NOT held out, so the only thing that varies along t inside the sum
    # is (moy, t_hold). Precompute the runs where that pair is constant: each
    # run is a CONTIGUOUS slice, so a block's contribution is a handful of
    # contiguous reductions rather than a fancy-index gather (which would
    # copy the block) or twelve masked sweeps of it.
    key = moy.astype(np.int64) * 2 + t_hold.astype(np.int64)
    edges = np.flatnonzero(np.diff(key)) + 1
    run_lo = np.concatenate(([0], edges)).astype(int)
    run_hi = np.concatenate((edges, [T])).astype(int)

    csum = np.zeros((12, H, W, C), dtype=np.float64)
    ccnt = np.zeros((12, H, W, C), dtype=np.int32)
    smean = np.empty((T, C), dtype=np.float64)      # per-(t,c) spatial mean

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")             # all-NaN slices are data
        # ---- pass 1: spatial means + climatology sums ---------------------
        for b, i0 in enumerate(range(0, T, chunk)):
            i1 = min(i0 + chunk, T)
            blk = X[i0:i1]
            fin = np.isfinite(blk)
            cnt = fin.sum(axis=(1, 2))                          # [n, C]
            tot = np.sum(blk, axis=(1, 2), where=fin, dtype=np.float64)
            with np.errstate(invalid="ignore", divide="ignore"):
                smean[i0:i1] = tot / cnt
            for lo, hi in zip(run_lo, run_hi):
                a, z = max(lo, i0), min(hi, i1)
                if a >= z or t_hold[a]:
                    continue
                m = int(moy[a])
                sub = slice(a - i0, z - i0)
                csum[m] += np.sum(blk[sub], axis=0, where=fin[sub],
                                  dtype=np.float64)
                ccnt[m] += fin[sub].sum(axis=0, dtype=np.int32)
            if verbose and b % 20 == 0:
                _say(f"pass 1/3 (climatology) {i1}/{T}")
        fin = blk = None            # the mask is chunk*H*W*C bytes; drop it

        # A channel with no temporal variance is a baked climatology: it is
        # context, not a target in disguise, and passes through untouched.
        dynamic = [c for c in range(C)
                   if np.nanstd(smean[:, c], dtype=np.float64) > 1e-6]
        if not dynamic:
            return X, dynamic

        with np.errstate(invalid="ignore", divide="ignore"):
            clim = (csum / ccnt).astype(np.float32)   # [12,H,W,C], NaN if n=0
        del csum, ccnt
        # Zero on the STATIC channels, so a whole block can be subtracted and
        # written back contiguously while the static channels come out
        # bit-identical: x -> float32(x) - 0.0f -> back is exact for every
        # float16/float32 value, and NaN stays NaN. The alternative — a fancy
        # index on the last axis, X[..., dynamic] — would scatter every write
        # across the same sub-page stride this rewrite exists to avoid.
        stat = np.setdiff1d(np.arange(C), np.asarray(dynamic))
        clim[..., stat] = 0.0
        _say(f"{len(dynamic)}/{C} dynamic channels; climatology done")

        # ---- pass 2: write the anomaly, accumulate (n, mean, M2) ----------
        wdt = np.promote_types(X.dtype, np.float32)
        n_t = np.zeros(C, dtype=np.float64)
        mu_t = np.zeros(C, dtype=np.float64)
        m2_t = np.zeros(C, dtype=np.float64)
        keep_x = ~x_hold
        for b, i0 in enumerate(range(0, T, chunk)):
            i1 = min(i0 + chunk, T)
            cm = clim[moy[i0:i1]]                     # [n,H,W,C] float32
            if cm.dtype == wdt:
                np.subtract(X[i0:i1], cm, out=cm)     # in place: no 2nd copy
                anom = cm
            else:
                anom = np.subtract(X[i0:i1], cm, dtype=wdt)
                del cm
            X[i0:i1] = anom                           # rounds to storage dtype
            del anom

            # Read back what was STORED, not what was computed — the original
            # took its mean and sd from the rounded float16 values and so
            # must this.
            blk = np.asarray(X[i0:i1], dtype=np.float64)
            msk = np.isfinite(blk)
            msk &= (~t_hold[i0:i1])[:, None, None, None]
            msk &= keep_x[None, None, :, None]
            n_b = msk.sum(axis=(0, 1, 2)).astype(np.float64)      # [C]
            s_b = np.sum(blk, axis=(0, 1, 2), where=msk, dtype=np.float64)
            nz = n_b > 0
            mu_b = np.where(nz, s_b / np.maximum(n_b, 1.0), 0.0)
            np.subtract(blk, mu_b, out=blk)
            np.square(blk, out=blk)
            m2_b = np.sum(blk, axis=(0, 1, 2), where=msk, dtype=np.float64)
            del blk, msk
            # Chan's parallel combination (see the docstring).
            n_new = n_t + n_b
            delta = mu_b - mu_t
            safe = np.maximum(n_new, 1.0)
            mu_t = np.where(nz, mu_t + delta * (n_b / safe), mu_t)
            m2_t = np.where(nz, m2_t + m2_b + delta * delta * n_t * n_b / safe,
                            m2_t)
            n_t = n_new
            if verbose and b % 20 == 0:
                _say(f"pass 2/3 (anomaly + moments) {i1}/{T}")
        del clim

        with np.errstate(invalid="ignore", divide="ignore"):
            sd = np.sqrt(m2_t / n_t)          # ddof=0, as np.std defaults
        mu = mu_t
        # Static channels must survive pass 3 untouched, and (x - 0.0) / 1.0
        # is exactly x in float64 for every finite or NaN input.
        mu[stat] = 0.0
        den = sd + 1e-6
        den[stat] = 1.0

        # ---- pass 3: z-score in place ------------------------------------
        for b, i0 in enumerate(range(0, T, chunk)):
            i1 = min(i0 + chunk, T)
            out = np.subtract(X[i0:i1], mu, dtype=np.float64)
            np.divide(out, den, out=out)
            X[i0:i1] = out
            del out
            if verbose and b % 20 == 0:
                _say(f"pass 3/3 (z-score) {i1}/{T}")
    _say(f"done ({nblk} blocks of {chunk} timesteps, 3 sequential passes)")
    return X, dynamic


def probe_now(codec, X, OBS, d, moy, t_hold, x_hold, dynamic,
              n_pixels=600, K=12, tsteps=400, tbatch=128, seed=0, obs_in=None,
              mask_chan=None, light=False, ocean=None,
              blk_rows=None, blk_pad=None):
    """All metrics for the codec AS IT IS NOW. X must already be in the
    space the codec was trained in (anomaly). Returns a flat dict.

    obs_in: optional observation mask for the ENCODER ONLY (ablations —
    channels marked unobserved enter as the codec's native missing tokens).
    Scoring targets always use the true OBS, so 'predict the field from
    less input' is measured against the same reality.

    ocean: the [H, W] ocean mask, `np.isfinite(d["X"][..., 0]).any(axis=0)`.
    HOISTED, not recomputed. It is a pure function of the tensor and train.py
    already computes exactly this expression at line 295, before the anomaly
    transform, from the same array — so every probe was re-deriving a constant.
    It is not a cheap constant: `d["X"][..., 0]` walks the whole
    channel-interleaved tensor at a stride of C*itemsize = 78 bytes, faulting
    every 4 KB page to use 2.6% of it — the pathology anomaly_transform's
    docstring documents and that sat run #389 in one function for seven hours.
    Measured on #419 at daily cadence it cost **150.2 s of every full probe
    and 148.2 s of every light one**, 8.5% of all probe time, and on #415 the
    same term ranged 125.8 s to 3,657.2 s depending on what the page cache
    happened to hold. Passing None recomputes it (the CLI backfill path).

    light=True computes ONLY the linear section probe and returns, and it does
    so on the ~46% of timesteps the RAPID record covers, thinned to one sample
    per decorrelation time (see RAPID_TAU_DAYS). Measured on #419's daily
    codec: the FULL probe costs 2,296.6 s and the light one 611.3 s; after the
    hoisted mask, the ridx-only embedding and the thinning they are ~1,683 s
    and ~17 s. Its purpose is cadence — the collapse guard fires on BOTH
    probes, so the guard's detection latency tracks the LIGHT cadence and the
    two knobs are separable: the full probe can be cut hard while the light
    one stays every 2,000 steps. Both modes emit `linear_r_deseas`, so
    downstream readers (metrics.jsonl, the status page) need no special
    case."""
    was_training = codec.training
    codec.eval()
    t0 = time.time()
    rng = np.random.default_rng(seed)
    if obs_in is None:
        obs_in = OBS
    lats, lons = d["lats"], d["lons"]
    T, H, W, C = X.shape
    ctx_all = np.stack([np.sin(2 * np.pi * moy / 12),
                        np.cos(2 * np.pi * moy / 12)], 1)
    if ocean is None:                      # CLI backfill; train.py passes it in
        ocean = np.isfinite(d["X"][..., 0]).any(axis=0)
    ys, xs = np.where(ocean)
    sec_y, sec_sel = rapid_section(lats, lons, ys, xs)   # protocol v3 clip

    # ---- RAPID target, deseasonalised once -------------------------------
    rapid = d["rapid"]
    ridx = rapid[:, 0].astype(int)
    rv_raw = rapid[:, 1].copy()
    rmoy = moy[ridx]
    tr_all = ~t_hold[ridx]
    rclim = np.array([rv_raw[tr_all & (rmoy == m)].mean() for m in range(12)])
    rv_des = rv_raw - rclim[rmoy]
    te_all = t_hold[ridx]

    out = {}
    # Which rows of `ridx` the LIGHT probe uses. Computed in BOTH modes: the
    # full probe reports the same estimator on the same rows so the two curves
    # can be read against each other instead of assumed comparable.
    lsel, lstride = light_rows(ridx, tr_all, te_all, d["months"])

    # ---- 1 · linear section probe (K=1) ----------------------------------
    if light:
        # ONLY THE TIMESTEPS THAT ARE READ. `Fsec` has exactly one consumer in
        # this branch — ridge_r(Fsec[ridx][lsel], ...) — so embedding all T
        # spent the majority of the light probe's budget on rows nobody ever
        # indexes. At daily that is 561 of 15,706 timesteps instead of all of
        # them. Every timestep is an independent encoder forward, so the rows
        # that come back are bit-identical to the corresponding rows of a full
        # pass (tests/test_probe_cost.py case 3 pins it).
        rt = ridx[lsel]
        tsel, inv = np.unique(rt, return_inverse=True)
        Zsec, _ = embed_everything(codec, X, obs_in, ctx_all, lats, lons,
                                   ys[sec_sel], xs[sec_sel], codec.d_z,
                                   mask_chan=mask_chan, t_sel=tsel,
                                   blk_rows=blk_rows, blk_pad=blk_pad)
        Fl = np.asarray(Zsec).mean(1)[inv]                # [len(lsel), d_z]
        out["linear_r_deseas"], _ = ridge_r(Fl, rv_des[lsel],
                                            tr_all[lsel], te_all[lsel])
        out["linear_r_raw"], _ = ridge_r(Fl, rv_raw[lsel],
                                         tr_all[lsel], te_all[lsel])
        out["light"] = True
        out["light_stride"] = int(lstride)
        out["light_n"] = int(len(lsel))
        out["light_n_test"] = int(te_all[lsel].sum())
        out["probe_seconds"] = round(time.time() - t0, 1)
        if was_training:
            codec.train()
        return out

    # ---- 2 · mini temporal transformer on frozen embeddings ---------------
    # THE SECTION IS EMBEDDED ONCE, NOT TWICE. `keep` is a strict superset of
    # `sec_sel` by construction (the union below), and PixelMAE.encode treats
    # the batch dimension as independent pixels — attention runs over the
    # 2 + C channel/cls/ctx tokens only, the encoder layers are LayerNorm
    # (feature-wise), and there is no BatchNorm anywhere in the model — so the
    # section's embeddings are the same whether they arrive in a batch of 266
    # or a batch of 864. This block therefore moved ABOVE the linear probe and
    # `Fsec` is sliced out of `Z`, which deletes a whole second embedding pass:
    # **463.3 s of every full probe at daily cadence, 20.2% of it.**
    #
    # The GEMM's row count changes 266 -> 864, so a BLAS is free to pick a
    # different blocking and reorder its reductions; equivalence in principle
    # is not equality in floating point, and the claim was MEASURED rather
    # than argued (tests/test_probe_cost.py case 2). At #419's own geometry —
    # 512x12, C=39, d_z 32, 266 rows against 864 — the float32 encoder output
    # is **bit-identical, 8,512/8,512 elements, max |dz|/mean|z| = 0.0e+00**
    # on CPU, and so is the float16 Z the probe stores and the section mean it
    # reads. oneDNN does not change its reduction order across that row count.
    # cuBLAS is not promised to behave the same way, so the test's tolerance
    # is 1e-4 rather than exact equality: what the model guarantees is that
    # rows do not COUPLE, and any residual difference is a reduction order,
    # ~1e-6 relative at worst, three orders below the float16 the cache holds.
    keep = rng.choice(len(ys), min(n_pixels, len(ys)), replace=False)
    keep = np.union1d(keep, sec_sel)
    kys, kxs = ys[keep], xs[keep]
    Z, coords = embed_everything(codec, X, obs_in, ctx_all, lats, lons,
                                 kys, kxs, codec.d_z, mask_chan=mask_chan,
                                 blk_rows=blk_rows, blk_pad=blk_pad)
    P = len(kys)
    # np.union1d sorts, and section_of returns np.where(...)[0] which is
    # already sorted, so these positions come back in sec_sel's own order —
    # Z[:, sec_in_keep] is row-for-row what a separate section embed produced.
    sec_in_keep = np.where(np.isin(keep, sec_sel))[0]
    Fsec = np.asarray(Z[:, sec_in_keep]).mean(1)          # [T, d_z]
    out["linear_r_deseas"], _ = ridge_r(Fsec[ridx], rv_des, tr_all, te_all)
    out["linear_r_raw"], _ = ridge_r(Fsec[ridx], rv_raw, tr_all, te_all)
    # The light probe's OWN estimator, on the identical subsample, so the two
    # curves in metrics.jsonl are comparable by measurement rather than by
    # assertion. One ridge solve: ridge is 0.006% of probe time.
    out["linear_r_deseas_light"], _ = ridge_r(
        Fsec[ridx[lsel]], rv_des[lsel], tr_all[lsel], te_all[lsel])
    out["light_stride"] = int(lstride)
    Zt = torch.from_numpy(Z)
    Mt = torch.as_tensor(ctx_all, dtype=torch.float32)
    static_ctx = torch.as_tensor(
        np.concatenate([np.zeros((P, codec.d_z), np.float32), coords], 1))

    torch.manual_seed(seed)
    mini = TemporalTransformer(d_z=codec.d_z, d_model=64, n_heads=4,
                               n_layers=2, k_max=K)
    opt = torch.optim.AdamW(mini.parameters(), lr=2e-3, weight_decay=1e-4)

    ok_t = np.array([t + 1 < T and not t_hold[t + 1] and t + 1 >= K
                     for t in range(T)])
    ok_p = ~x_hold[kxs]
    pt, pp = np.where(ok_t[:, None] & ok_p[None, :])
    pt = torch.as_tensor(pt, dtype=torch.long)
    pp = torch.as_tensor(pp, dtype=torch.long)
    g = torch.Generator().manual_seed(seed)
    for s in range(tsteps):
        k = torch.randint(0, len(pt), (tbatch,), generator=g)
        t, p = pt[k], pp[k]
        base = t - K + 1
        zseq = torch.stack([Zt[base + j, p] for j in range(K)], 1)
        mseq = torch.stack([Mt[base + j] for j in range(K)], 1)
        ztgt = torch.stack([Zt[base + j + 1, p] for j in range(K)], 1)
        pred, _ = mini(zseq, mseq, static_ctx[p])
        loss = (pred - ztgt).pow(2).mean()
        opt.zero_grad(); loss.backward(); opt.step()
    mini.eval()

    with torch.no_grad():
        # held-out months, z and channel space
        ev_t = np.array([t + 1 < T and t_hold[t + 1] and t + 1 >= K
                         for t in range(T)])
        et, ep = np.where(ev_t[:, None] & np.ones(P, bool)[None, :])
        sel = np.random.default_rng(seed).choice(
            len(et), min(8000, len(et)), replace=False)
        et = torch.as_tensor(et[sel], dtype=torch.long)
        ep = torch.as_tensor(ep[sel], dtype=torch.long)
        base = et - K + 1
        zseq = torch.stack([Zt[base + j, ep] for j in range(K)], 1)
        mseq = torch.stack([Mt[base + j] for j in range(K)], 1)
        pred, _ = mini(zseq, mseq, static_ctx[ep])
        zhat, ztrue, zlast = pred[:, -1], Zt[et + 1, ep], Zt[et, ep]
        out["z_mse_model"] = float((zhat - ztrue).pow(2).mean())
        out["z_mse_persistence"] = float((zlast - ztrue).pow(2).mean())

        Xt, OBSt = X, OBS          # already tensors (zeros at missing + mask)
        qc = torch.arange(C)[None, :].expand(len(et), -1)
        # codec.query runs the decoder, which lives on the CODEC's device —
        # cuda during an in-training probe since the GPU-probe change, while
        # zhat and the index tensors here are all CPU (Zt comes back from
        # embed_everything as numpy, the mini transformer is CPU). Bridge the
        # one call: inputs to the codec's device, result back to CPU where the
        # raw-field tensors Xt/OBSt live. This seam is what killed #56-#59 at
        # their first full probe (step 10k): a cuda channel-embedding weight
        # indexed by a cpu chan_idx is the index_select device mismatch.
        qdev = next(codec.parameters()).device
        xhat = codec.query(
            zhat.to(qdev), qc.to(qdev),
            torch.zeros(len(et), C, 3, dtype=torch.long, device=qdev)).cpu()
        kys_t = torch.as_tensor(kys, dtype=torch.long)
        kxs_t = torch.as_tensor(kxs, dtype=torch.long)
        v1 = Xt[et + 1, kys_t[ep], kxs_t[ep]]
        o1 = OBSt[et + 1, kys_t[ep], kxs_t[ep]]
        v0 = Xt[et, kys_t[ep], kxs_t[ep]]
        o0 = OBSt[et, kys_t[ep], kxs_t[ep]]
        dyn = torch.zeros(C, dtype=torch.bool); dyn[dynamic] = True
        both = o0 & o1 & dyn[None, :]
        out["chan_mse_model"] = float(((xhat - v1).pow(2) * both).sum() / both.sum())
        out["chan_mse_persistence"] = float(((v0 - v1).pow(2) * both).sum() / both.sum())

        # RAPID probe from the mini transformer's section hidden state
        sec_in_keep = torch.as_tensor(sec_in_keep, dtype=torch.long)
        F = np.zeros((T, 64), dtype=np.float32)
        for t in range(K - 1, T):
            base = t - K + 1
            zseq = torch.stack([Zt[base + j, sec_in_keep] for j in range(K)], 1)
            mseq = torch.stack([Mt[base + j].expand(len(sec_in_keep), -1)
                                for j in range(K)], 1)
            _, hid = mini(zseq, mseq, static_ctx[sec_in_keep])
            F[t] = hid[:, -1].mean(0).numpy()
        ok = ridx >= K - 1
        ri = ridx[ok]
        tr, te = ~t_hold[ri], t_hold[ri]
        out["temporal_r_deseas"], _ = ridge_r(F[ri], rv_des[ok], tr, te)
        out["temporal_r_raw"], _ = ridge_r(F[ri], rv_raw[ok], tr, te)

    out["chan_vs_persistence_pct"] = round(
        100 * (1 - out["chan_mse_model"] / out["chan_mse_persistence"]), 1)
    out["z_vs_persistence_pct"] = round(
        100 * (1 - out["z_mse_model"] / out["z_mse_persistence"]), 1)
    out["probe_seconds"] = round(time.time() - t0, 1)
    if was_training:
        codec.train()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--data", default=os.path.join(HERE, "cache", "na_pixels.npz"))
    ap.add_argument("--n-pixels", type=int, default=600)
    ap.add_argument("--tsteps", type=int, default=400)
    a = ap.parse_args()

    run_dir = os.path.join(HERE, "runs", a.run)
    ck = torch.load(os.path.join(run_dir, "pixelmae.pt"),
                    map_location="cpu", weights_only=False)
    d = np.load(a.data)
    X = d["X"].copy()
    months = [str(m) for m in d["months"]]
    moy = np.array([int(m[5:7]) - 1 for m in months])
    hold_years = set(ck["args"]["holdout_years"].split(","))
    t_hold = np.array([m[:4] in hold_years for m in months])
    lo, hi = (float(v) for v in ck["args"]["holdout_lon"].split(","))
    x_hold = (d["lons"] >= lo) & (d["lons"] < hi)

    if not ck["args"].get("anomaly"):
        sys.exit("trainprobe measures anomaly-space codecs only "
                 "(state space is disqualified from ranking).")
    X, dynamic = anomaly_transform(X, moy, t_hold, x_hold)

    codec = codec_from_ckpt(ck, X.shape[-1])
    codec.load_state_dict(ck["model"])
    # Standalone path only. Called from train.py the codec is already on the
    # GPU, which is why this was never noticed here; run as a script it would
    # embed on CPU, the same omission dip_check.py and rollout.py had.
    codec.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))

    out = probe_now(codec, torch.from_numpy(np.nan_to_num(X, nan=0.0)),
                    torch.from_numpy(np.isfinite(X)), d, moy, t_hold, x_hold,
                    dynamic, n_pixels=a.n_pixels, tsteps=a.tsteps)
    out["run"] = a.run
    print(json.dumps(out, indent=2))
    json.dump(out, open(os.path.join(run_dir, "trainprobe.json"), "w"), indent=2)
    print(f"wrote {run_dir}/trainprobe.json")


if __name__ == "__main__":
    main()

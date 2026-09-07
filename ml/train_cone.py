#!/usr/bin/env python3
"""E-069 · train the cone-native codec (ml/cone_codec.py::ConeMAE).

`ml/train.py` trains PixelMAE over one pixel-bin; this trains ConeMAE over the
INNER CONE — the anchor's 3x3 patch at lag 0 plus, per channel, a sunflower of
dots at lags 1..L_in whose reach follows that channel's family (ml/cone.py).
Everything else follows train.py deliberately, so the two arms differ in the
stencil and nothing else:

  * ANOMALY SPACE, TRAIN BINS ONLY. `trainprobe.anomaly_transform` is called,
    never re-implemented — there is exactly one anomaly transform in ml/ and
    tests/test_one_anomaly_transform.py fails if a second appears. Difference
    from train.py: train.py makes the transform optional (`--anomaly`), this
    trainer always applies it, because a reconstruction loss on raw pentad
    state is dominated by the seasonal cycle and the cone's whole claim is
    about ANOMALY propagation. There is no `--holdout-lon` here either: the
    pool rule is the window-scope one below, and a longitude block would be a
    second, unmeasured holdout.
  * METRICS IN train.py's OWN RECORD FAMILY (ml/CLAUDE.md §0d — a new trainer
    format must teach the status page its records). A `{"config": {...}}`
    first line, `{"step", "loss_rec", "loss_nei"}` training records, and
    probe-shaped records at each eval. status.html's `parseJsonl` already
    routes all three, so no page change is needed; see the writer's comment.
  * CHECKPOINTS as `{"args", "model", "chan_names", "norm"}`, `args` being
    `vars(a)` exactly as `codec_from_ckpt` expects to find an architecture.

POOL DISCIPLINE (`--holdout-scope window`, the ONLY scope implemented). A
training anchor is admitted only if EVERY bin its cone touches — L_in pentads
back and both future targets forward — is a training bin
(`cone_sampler.admissible`). Before the first step the sampler SELF-CERTIFIES
by brute force over 4,096 drawn anchors (`cone_sampler.certify`, E-059's
pattern) and the run REFUSES on any violation. The certificate is deliberately
not a rearrangement of the admission test: a check written from the expression
it checks proves only that the expression is self-consistent.

    python3 ml/train_cone.py --smoke --out /tmp/cone_smoke
    python3 ml/train_cone.py --tensor ml/cache/family4_na025_pentad_r3.npz \\
        --steps 20000 --batch 256 --velocity-probe --out ml/runs/cone

Plan: ml/plans/E069_cone_codec.md §§3, 5.
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

from cone import channel_depth_dbar                              # noqa: E402
from cone_sampler import ConeSampler                             # noqa: E402
from cone_codec import ConeMAE, default_plan                     # noqa: E402

SMOKE_CHANS = ["cur_speed", "log_mld", "ssh", "tau_x", "tau_y", "sst",
               "cur_u", "cur_v"]
PENTAD_EPOCH = np.datetime64("1982-01-01")
PENTAD_DAYS = 5

# The group the family-8 observation store stands in for (E-076 section 2.4:
# family 8 changes how a SPARSE channel enters the cone and nothing else).
PROFILE_GROUP = "rg100"
# The per-dot extra fields, in the ONE order the model reads them.
DOT_EXTRA_FIELDS = ("nr", "fp")


def dot_extras_of(spec):
    """`(n_extra, live)` from a `--dot-extras` string.

    Naming any extra widens the dot token to [value, observed, nr, fp0, fp1]
    and zeroes what was not named, so "n_R withheld" and "the footprint
    zeroed" — two arms of E-076 section 5's ladder — are the SAME architecture
    with different inputs, and a difference between them cannot be a
    difference in parameter count.
    """
    names = [s.strip() for s in str(spec or "").split(",") if s.strip()]
    bad = [n for n in names if n not in DOT_EXTRA_FIELDS]
    if bad:
        raise SystemExit(
            f"--dot-extras {spec!r}: unknown field(s) {bad}. The set is "
            f"{list(DOT_EXTRA_FIELDS)} — `nr` is log1p of the local "
            f"observation count and `fp` is the two footprint fields of "
            f"E-076 section 2.6.")
    if not names:
        return 0, ()
    return 3, tuple(n for n in DOT_EXTRA_FIELDS if n in names)


# --------------------------------------------------------------------- CLI --
def parse(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--tensor", default="",
                   help="npz with X, months, lats, lons, chan (family4 r3); "
                        "OBS is derived as isfinite(X) exactly as train.py's "
                        "LazyPixels does. A family-7 tensor (a `groups` key "
                        "and one `_X_<group>.npy` per group) is read as a "
                        "cone_sampler.GroupSet instead — same channel schema, "
                        "each group at its native resolution. Required unless "
                        "--smoke.")
    p.add_argument("--holdout-scope", default="window",
                   help="window is the only scope implemented — see the "
                        "module docstring; any other value is refused.")
    p.add_argument("--holdout-years", default="2009,2017,2023",
                   help="train.py's flag, same meaning and same default.")
    p.add_argument("--steps", type=int, default=20000)
    p.add_argument("--batch", type=int, default=256)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--d-model", type=int, default=256)
    p.add_argument("--n-heads", type=int, default=8)
    p.add_argument("--n-latents", type=int, default=64)
    p.add_argument("--n-layers", type=int, default=6)
    p.add_argument("--d-z", type=int, default=32)
    p.add_argument("--d-dec", type=int, default=256)
    p.add_argument("--dec-layers", type=int, default=2)
    p.add_argument("--n-fourier", type=int, default=8)
    p.add_argument("--L-in", type=int, default=6,
                   help="inner-window depth in pentads. 0 = the SNAPSHOT "
                        "ablation: no dots, the lag-0 patch only.")
    p.add_argument("--future-lags", default="1,2")
    p.add_argument("--n-dot-queries", type=int, default=256)
    # ---- E-069b, the masking plan. Every default is today's behaviour. -----
    p.add_argument("--chan-drop-scope", default="all",
                   choices=("all", "lag0"),
                   help="how far a channel drop reaches. 'all' (default, and "
                        "what every archived cone number was trained under) "
                        "hides the channel at lag 0 AND at every dot; 'lag0' "
                        "hides its lag-0 patch only and leaves its dots "
                        "visible, so a hidden dot is never a dot of a channel "
                        "the encoder cannot see at all.")
    p.add_argument("--lag-band-p", type=float, default=0.3,
                   help="probability a batch element has every dot at lag <= "
                        "l0 (l0 uniform on 1..3 pentads) hidden — the "
                        "forward-stepping scheme.")
    p.add_argument("--sector-p", type=float, default=0.3,
                   help="probability a batch element has a 90-degree bearing "
                        "sector of its dots hidden — the interpolate-across-"
                        "bearings scheme.")
    p.add_argument("--anchor-hidden-only", action="store_true",
                   help="score the anchor-reconstruction family only on the "
                        "channels that were dropped for that batch element. "
                        "Off by default: under the default plan 69% of the "
                        "anchor family's weight sits on channels whose target "
                        "is visible in the input as the patch centre.")
    p.add_argument("--aux-latent-w", type=float, default=0.25,
                   help="weight of the auxiliary loss through the decoder's "
                        "FULL memory ([z-token] + latents). The headline "
                        "term always goes through z alone — see "
                        "ConeMAE.decode's docstring for the degeneracy this "
                        "split closes.")
    p.add_argument("--eval-every", type=int, default=0,
                   help="0 = steps//10")
    p.add_argument("--eval-anchors", type=int, default=1024)
    p.add_argument("--save-every", type=int, default=0, help="0 = steps//4")
    p.add_argument("--certify-n", type=int, default=4096)
    p.add_argument("--velocity-probe", action="store_true",
                   help="H1: ridge from z (cur_* dropped from the input) to "
                        "the anchor's cur_u/cur_v, year-blocked folds.")
    p.add_argument("--snapshot-ablation", action="store_true",
                   help="also train an L_in=0 twin in-process and probe it, "
                        "so the two arms share the probe anchors exactly.")
    p.add_argument("--probe-anchors", type=int, default=2048)
    # ---- E-076a, the family-8 arm. Every default is today's behaviour. -----
    p.add_argument("--argo-store", default="",
                   help="directory of the family-8 Argo observation store "
                        "(ml/family8_store.py). Given, the run REFUSES unless "
                        "the directory carries a store.json whose sha256s "
                        "match its files. With --profile-k >= 1 the rg100 "
                        "group is not read at all and the k nearest profiles "
                        "become dot tokens (E-076 section 2); with "
                        "--profile-k 0 the tensor's rg100 group is read "
                        "exactly as today and the store supplies only the "
                        "common target of section 5.1 — that is the TWIN arm.")
    p.add_argument("--profile-k", type=int, default=5,
                   help="how many nearest profiles enter the input per anchor. "
                        "E-076 section 2.3 derives 5 from the array's density: "
                        "the fifth neighbour sits at about one surface "
                        "correlation length. 0 = the gridded twin.")
    p.add_argument("--profile-R-max-km", type=float, default=1000.0)
    p.add_argument("--profile-T-max-days", type=float, default=30.0)
    p.add_argument("--dot-extras", default="",
                   help="comma set of `nr` (log1p of the local observation "
                        "count) and `fp` (the two footprint fields of E-076 "
                        "section 2.6) to append to every dot token. Empty is "
                        "today's two-number dot and the architecture every "
                        "archived checkpoint has. Naming ANY of them widens "
                        "the dot projection by three and zeroes the ones not "
                        "named, so an ablation changes what the model is told "
                        "and not how many parameters it has.")
    p.add_argument("--profile-drop-p", type=float, default=0.5,
                   help="probability, per anchor, that the NEAREST profile is "
                        "withheld from the input and added to the decoder's "
                        "query set instead — masked-profile reconstruction "
                        "(E-076 section 7.5), which is the objective the "
                        "common-target eval scores.")
    p.add_argument("--profile-eval-n", type=int, default=2048,
                   help="anchors in the fixed common-target eval set of E-076 "
                        "section 5.1 (held-out bins, n_R >= 2). 0 = skip it.")
    p.add_argument("--out", default=os.path.join(HERE, "runs", "cone"))
    p.add_argument("--metrics", default="metrics.jsonl")
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--smoke-family8", action="store_true",
                   help="--smoke, on a synthetic THREE-GROUP tensor plus a "
                        "synthetic observation store: the family-8 path end "
                        "to end on two CPU cores in under a minute.")
    a = p.parse_args(argv)
    if a.holdout_scope != "window":
        raise SystemExit(
            f"--holdout-scope {a.holdout_scope!r}: only 'window' is "
            f"implemented. The cone reads L_in pentads back and two forward, "
            f"so 'window' is the rule that keeps a held-out bin out of the "
            f"training set by any path (ml/cone_sampler.py::admissible). A "
            f"terminal or longitude scope would need its own admission test "
            f"and its own certificate; refusing rather than silently using "
            f"the window one under another name.")
    if a.smoke:
        # The smoke geometry is SMALL on purpose: it exercises every code path
        # on 2 CPU cores in minutes. The default geometry (~6M params) is what
        # a dispatch uses; nothing here is a default anywhere else.
        # --smoke FIXES THE WHOLE CONFIGURATION, including the learning rate:
        # 200 steps is an exercise of the code path, not a schedule, and at
        # the dispatch lr nothing would move far enough for the velocity
        # probe to say anything about the stencil.
        a.steps, a.batch, a.lr = 200, 32, 2e-3
        a.d_model, a.n_heads, a.n_latents, a.n_layers = 64, 4, 16, 2
        a.d_dec, a.dec_layers, a.n_fourier = 64, 2, 6
        a.n_dot_queries = 48
        a.eval_every = a.eval_every or 50
        a.eval_anchors = min(a.eval_anchors, 256)
        a.probe_anchors = min(a.probe_anchors, 768)
        a.certify_n = min(a.certify_n, 4096)
        a.holdout_years = "1983"
        a.velocity_probe = True
        a.snapshot_ablation = True
    if a.smoke_family8:
        # The FAMILY-8 smoke. Same argument as --smoke and a different code
        # path: three groups, a sparse gather, a withheld profile in the query
        # set and the common-target eval. The velocity probe and the snapshot
        # twin are OFF — they answer H1, which this path is not about, and
        # they would double a run that has to stay under a minute.
        a.steps, a.batch, a.lr = 60, 16, 2e-3
        a.d_model, a.n_heads, a.n_latents, a.n_layers = 32, 4, 8, 2
        a.d_dec, a.dec_layers, a.n_fourier = 32, 2, 4
        a.n_dot_queries = 32
        a.eval_every = a.eval_every or 30
        a.eval_anchors = min(a.eval_anchors, 64)
        a.certify_n = min(a.certify_n, 64)
        a.holdout_years = a.holdout_years if a.holdout_years != \
            "2009,2017,2023" else "2011"
        a.velocity_probe = False
        a.snapshot_ablation = False
        a.profile_k = a.profile_k if a.profile_k >= 0 else 3
        a.dot_extras = a.dot_extras or "nr,fp"
        a.profile_eval_n = min(a.profile_eval_n, 64)
    a.eval_every = a.eval_every or max(1, a.steps // 10)
    a.save_every = a.save_every or max(1, a.steps // 4)
    return a


# ------------------------------------------------------------------- data --
def smoke_tensor(path, seed=0):
    """A synthetic pentad tensor with a PLANTED ADVECTION, written to `path`.

    T=120, H=40, W=56, C=8 on a 0.25-degree grid at 30N, T bins of five days
    from 1982-01-01 (ml/build_family4.py's pentad axis).

    THE FLOW IS A SHEAR, NOT A UNIFORM DRIFT, and that is the one design
    decision here worth arguing. A spatially uniform (u_t, v_t) — the obvious
    reading of "moving at a constant velocity" — makes `cur_u` the SAME NUMBER
    at every anchor of a bin, so the velocity probe has 120 independent
    samples dressed up as 2,048 anchors and a 32-dimensional ridge overfits
    them catastrophically (measured: out-of-fold r = -0.38, i.e. the fold
    structure, not the codec). A shear
            u(t, y) = (P_t - P_{t-1}) * s(y),   s(y) = 2y/(H-1) - 1
    gives every latitude its own velocity, so the target varies WITHIN a bin
    and the probe measures the embedding rather than the sample size. `P_t`
    (and `Q_t` in x) is a mean-reverting AR(1) — bounded, stationary and
    deliberately NOT seasonal, so nothing in the context token (sin/cos of
    the day of year) predicts it, and the held-out bins are drawn from the
    same distribution as the training ones.

    TWO CHANNELS ARE ADVECTED LINEAR RAMPS: `ssh(t,y,x) = 0.05*(x - Px(t,y))`
    and `log_mld(t,y,x) = 0.05*(y - Qy(t,x))`, i.e. a linear field carried by
    the flow. Their pentad-to-pentad difference at ANY anchor is exactly
    -0.05 * the local velocity, so the planted velocity is LINEARLY readable
    from the cone's anchor column (lag 0 patch centre minus the lag-1 dot at
    (0,0)) — and is not present at lag 0 in any form, because a snapshot sees
    the ramp's POSITION, never its displacement. That is what lets a 200-step
    CPU smoke test resolve the cone-vs-snapshot contrast at all: the test asks
    whether the STENCIL carries motion, not whether a two-minute optimisation
    converged.

    `sst` is the plan's advected Gaussian bump (decorative — it moves with the
    domain-centre flow), `cur_speed` the local speed, `tau_*` white forcing.
    Land (a NaN block) and ~1% scattered NaN exercise the miss_tok path.
    """
    rng = np.random.default_rng(seed)
    T, H, W, C = 120, 40, 56, len(SMOKE_CHANS)
    lats = 30.0 + 0.25 * np.arange(H)
    lons = -60.0 + 0.25 * np.arange(W)

    # The displacement processes (cells). A SUM OF SINUSOIDS with random
    # phases, not an AR(1), and the velocity is the CENTRED difference
    # (P[t+1] - P[t-1])/2 — the bin-MEAN velocity, which is what a binned
    # GLORYS `cur_u` is, while the displacement BETWEEN bin means is a
    # backward difference. Neither choice is cosmetic:
    #   · an AR(1) position has increment -(1-rho)*P + noise, so the POSITION
    #     predicts the velocity and the snapshot arm reads the planted signal
    #     off lag 0 with no stencil at all — an ablation that is not an
    #     ablation (measured on that version: corr(ssh_t, cur_u) = -0.38);
    #   · a sinusoid's BACKWARD difference is centred half a step early, so it
    #     too correlates with the position at sin(w/2) = 0.40 at these
    #     periods. Its CENTRED difference is the derivative, orthogonal to the
    #     position over the record.
    # Measured on this construction: R^2 of the velocity on lag 0 alone
    # 0.0002, on lags 0-1 0.906, on lags 0-6 1.000 — the cone's ceiling is
    # the whole signal and the snapshot's is nothing. The refusal below checks
    # the realised draw rather than trusting the argument.
    # Periods are in pentads and kept well away from 73 (one year) and its
    # harmonics, so nothing seasonal — nothing in the context token —
    # predicts the flow.
    per = np.array([5.0, 8.0, 13.0])
    amp = np.array([4.0, 5.0, 6.0])
    tt = np.arange(-1, T + 1)[:, None]                # one bin either side
    P = (amp * np.sin(2 * np.pi * tt / per + rng.uniform(0, 2 * np.pi, 3))
         ).sum(1)
    Q = (amp * np.sin(2 * np.pi * tt / per + rng.uniform(0, 2 * np.pi, 3))
         ).sum(1)
    du = (P[2:] - P[:-2]) / 2.0                                # [T] centred
    dv = (Q[2:] - Q[:-2]) / 2.0
    P, Q = P[1:-1], Q[1:-1]
    for name, pos, vel in (("P", P, du), ("Q", Q, dv)):
        lk = abs(float(np.corrcoef(pos, vel)[0, 1]))
        if lk > 0.15:
            raise SystemExit(
                f"smoke_tensor: |corr({name}_t, velocity)| = {lk:.3f} — the "
                f"planted POSITION predicts the planted VELOCITY on this "
                f"draw, so the snapshot ablation would not be an ablation. "
                f"Refusing to write a tensor whose control is contaminated.")
    sy = (2.0 * np.arange(H) / (H - 1.0) - 1.0)                # shear profile
    sx = (2.0 * np.arange(W) / (W - 1.0) - 1.0)
    u = du[:, None] * sy[None, :]                              # [T, H] zonal
    v = dv[:, None] * sx[None, :]                              # [T, W] merid.

    yy, xx = np.meshgrid(np.arange(H), np.arange(W), indexing="ij")
    X = np.empty((T, H, W, C), np.float32)
    sig = 6.0

    def nz(scale):
        return (scale * rng.normal(size=(H, W))).astype(np.float32)

    for t in range(T):
        # the bump rides the domain-centre flow (s is 0 at the centre, so it
        # is given half the domain-mean displacement — decoration, not the
        # signal the probe reads)
        cy = H / 2.0 + 0.5 * Q[t]
        cx = W / 2.0 + 0.5 * P[t]
        d2 = (yy - cy) ** 2 + (xx - cx) ** 2
        ut = u[t][:, None] + 0.0 * xx                          # [H, W]
        vt = v[t][None, :] + 0.0 * yy
        X[t, :, :, 0] = np.hypot(ut, vt) + nz(0.02)            # cur_speed
        X[t, :, :, 1] = 0.05 * (yy - Q[t] * sx[None, :]) + nz(0.005)   # log_mld
        X[t, :, :, 2] = 0.05 * (xx - P[t] * sy[:, None]) + nz(0.005)   # ssh
        X[t, :, :, 3] = nz(0.5)                                # tau_x
        X[t, :, :, 4] = nz(0.5)                                # tau_y
        X[t, :, :, 5] = 3.0 * np.exp(-d2 / (2.0 * sig * sig)) + nz(0.02)
        X[t, :, :, 6] = ut + nz(0.02)                          # cur_u
        X[t, :, :, 7] = vt + nz(0.02)                          # cur_v
    X[:, :4, :4, :] = np.nan                                   # land
    X[rng.random(X.shape) < 0.01] = np.nan                     # dropouts

    days = PENTAD_EPOCH + (PENTAD_DAYS * np.arange(T)).astype("timedelta64[D]")
    months = np.array([str(d) for d in days])
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    np.savez_compressed(path, X=X, months=months, lats=lats, lons=lons,
                        chan=np.array(SMOKE_CHANS))
    return path


SMOKE_F8_LEVELS = [10.0, 30.0, 50.0, 100.0, 150.0, 200.0, 300.0, 400.0,
                   500.0, 700.0, 900.0, 1100.0, 1300.0, 1500.0, 1700.0, 1900.0]
SMOKE_F8_G025 = ["cur_speed", "ssh", "sst", "cur_u", "cur_v"]
# t2m/skt (family A/C) plus the three LAND channels (family L), so the smoke
# meets every cone family the real tensor has — #549 died on the one it lacked.
SMOKE_F8_G100 = ["t2m", "skt", "log_swe", "soilw", "tsoil"]
SMOKE_F8_CHAN = (["rg_t%d" % int(v) for v in SMOKE_F8_LEVELS]
                 + ["rg_s%d" % int(v) for v in SMOKE_F8_LEVELS])


def smoke_family8(root, seed=0, T=440, b0=2045, ny=48, nx=64):
    """A synthetic THREE-GROUP tensor plus a synthetic observation store.

    `--smoke` exercises the family-4 path on one dense array; this exercises
    the family-7/8 path, which is a different one at every step: three groups
    at two resolutions, a live-bins group, the per-group anomaly transform,
    the sparse gather, a withheld profile in the query set and the
    common-target eval. Small on purpose — 48 x 64 cells and 150 pentad bins
    starting at bin 2045 (2010-01-02), so the whole run is a minute on two
    CPU cores and the held-out year (2011) sits INSIDE the record, where the
    interspersed half of E-076 §5.1's split lives.

    THE INTERIOR IS A FUNCTION OF THE SURFACE, deliberately: the depth column
    is a smooth field driven by the same slowly-varying process as `sst` and
    `ssh`, so a codec that reads the surface and the neighbouring profiles CAN
    say something about a withheld profile. Without that the common-target
    eval would be scoring noise and would pass whatever the code did.

    The profiles are drawn from the SAME field, at random positions and times
    inside each bin — not at cell centres — so the offsets `(dy_km, dx_km,
    dt_days)` really carry information and are not a constant the model can
    ignore. Returns `(tensor_path, store_path)`.
    """
    from family8_store import write_synthetic_store
    from tensor_io import group_path
    rng = np.random.default_rng(seed)
    lats = 20.0 + 0.25 * np.arange(ny)
    lons = -60.0 + 0.25 * np.arange(nx)
    H1, W1 = ny // 4 + 1, nx // 4 + 1
    lat1 = 20.0 + np.arange(H1, dtype=np.float64)
    lon1 = -60.0 + np.arange(W1, dtype=np.float64)
    bins = b0 + np.arange(T, dtype=np.int64)
    days = PENTAD_EPOCH + (PENTAD_DAYS * bins).astype("timedelta64[D]")
    months = np.array([str(d) for d in days])

    yy, xx = np.meshgrid(np.arange(ny), np.arange(nx), indexing="ij")

    TWO_PI = 2.0 * np.pi

    def wave(t, y, x):
        """ONE scalar field, read by every group and by every profile.

        Two timescales and no third: a SEASONAL term at 73 pentads, which the
        anomaly transform's monthly climatology removes, and two fast terms at
        11 and 17 pentads, which it cannot. Both are sampled the same way in
        every year, so a held-out year is drawn from the same distribution as
        a training one — a slow wave whose period is the length of the record
        would put the held-out year at its own phase and the held-out loss
        would measure the draw rather than the model.
        """
        return (np.sin(0.11 * y + 0.09 * x + TWO_PI * t / 73.0)
                + 0.6 * np.sin(0.11 * y - TWO_PI * t / 11.0)
                + 0.6 * np.cos(0.09 * x + TWO_PI * t / 17.0))

    def field(t):
        return wave(t, yy, xx)

    lev = np.asarray(SMOKE_F8_LEVELS)
    decay = np.exp(-lev / 700.0)                       # [16]

    def column(t, y, x):
        """The interior at (t, y, x) — the SAME wave with depth, evaluated at
        fractional y/x and fractional t for a profile."""
        s = wave(t, y, x)
        tt = 12.0 + 8.0 * s * decay - 0.004 * lev
        ss = 35.0 + 0.4 * s * decay + 0.0003 * lev
        return tt, ss

    g025 = np.empty((T, ny, nx, len(SMOKE_F8_G025)), np.float32)
    g100 = np.empty((T, H1, W1, len(SMOKE_F8_G100)), np.float32)
    y1g, x1g = np.meshgrid(np.arange(H1) * 4.0, np.arange(W1) * 4.0,
                           indexing="ij")
    for t in range(T):
        f = field(t)
        g025[t, :, :, 0] = np.abs(f) + 0.01 * rng.normal(size=(ny, nx))
        g025[t, :, :, 1] = 0.1 * f
        g025[t, :, :, 2] = 15.0 + 5.0 * f
        g025[t, :, :, 3] = np.gradient(f, axis=1)
        g025[t, :, :, 4] = np.gradient(f, axis=0)
        c = wave(t, y1g, x1g)
        g100[t, :, :, 0] = 20.0 + 3.0 * c
        g100[t, :, :, 1] = 21.0 + 3.0 * c
        for k in range(2, g100.shape[-1]):              # the land channels
            g100[t, :, :, k] = (0.5 + 0.1 * k) * c + 0.3 * k
    g025[:, :3, :3, :] = np.nan                        # a little land

    # `rg100`: one row per MONTH, into the pentad holding the 15th (E-034 §4),
    # so eleven bins in twelve have no row at all — the liveness the family-7
    # reader is built around.
    live_rows, rg_bins, seen = [], [], set()
    for t in range(T):
        d0 = np.datetime64(str(months[t]))
        ym = str(d0)[:7]
        mid = np.datetime64(f"{ym}-15")
        if ym not in seen and d0 <= mid < d0 + np.timedelta64(5, "D"):
            seen.add(ym)
            live_rows.append(t)
            rg_bins.append(int(bins[t]))
    rg = np.empty((len(live_rows), H1, W1, 32), np.float32)
    for r, t in enumerate(live_rows):
        for j in range(H1):
            for i in range(W1):
                tt, ss = column(t, j * 4.0, i * 4.0)
                rg[r, j, i, :16] = tt
                rg[r, j, i, 16:] = ss

    # Z-SCORE AT BUILD TIME, as ml/build_family7.py does, and keep the (mean,
    # sd) in `norm_<group>` — the sampler needs `norm_rg100` to put a raw
    # profile into the tensor's own space (E-076 §2.5).
    def znorm(A):
        m = np.nanmean(A, axis=(0, 1, 2))
        s = np.nanstd(A, axis=(0, 1, 2))
        s = np.where(s > 1e-6, s, 1.0)
        return ((A - m) / s).astype(np.float32), np.stack([m, s], axis=1)

    g025, n025 = znorm(g025)
    g100, n100 = znorm(g100)
    rg, nrg = znorm(rg)

    # The profiles: 30 per live bin plus a thin scatter elsewhere, at random
    # fractional positions and times inside their bin.
    p_bin, p_time, p_lat, p_lon, p_t, p_s = [], [], [], [], [], []
    for t in range(T):
        n = 30 if t in live_rows else 8
        fy = rng.uniform(0, ny - 1, n)
        fx = rng.uniform(0, nx - 1, n)
        frac = rng.uniform(0, 1, n)
        for j in range(n):
            tt, ss = column(t + frac[j], fy[j], fx[j])
            # A real profile does not fill every level: the deep ones go
            # missing far more often (docs/FAMILY8_DATA_HANDOVER.md §3).
            miss = rng.random(16) < (0.05 + 0.25 * (lev / lev[-1]))
            tt = np.where(miss, np.nan, tt)
            ss = np.where(rng.random(16) < 0.1, np.nan, ss)
            p_bin.append(int(bins[t]))
            p_time.append(float(PENTAD_DAYS * bins[t] + 5.0 * frac[j]))
            p_lat.append(float(lats[0] + 0.25 * fy[j]))
            p_lon.append(float(lons[0] + 0.25 * fx[j]))
            p_t.append(tt)
            p_s.append(ss)

    stem = os.path.join(root, "family7_smoke_f8")
    os.makedirs(root, exist_ok=True)
    np.save(group_path(stem + ".npz", "g025"), g025)
    np.save(group_path(stem + ".npz", "g100"), g100)
    np.save(group_path(stem + ".npz", "rg100"), rg)
    np.savez(stem + ".npz", groups=np.array(["g025", "g100", "rg100"]),
             months=months, lats=lats, lons=lons, lat1=lat1, lon1=lon1,
             chan_g025=np.array(SMOKE_F8_G025),
             chan_g100=np.array(SMOKE_F8_G100),
             chan_rg100=np.array(SMOKE_F8_CHAN),
             bin_index=bins, rg_bin_index=np.array(rg_bins, np.int64),
             norm_g025=n025, norm_g100=n100, norm_rg100=nrg,
             recipe=np.array("f7_smoke_f8"))
    store = write_synthetic_store(
        os.path.join(root, "family8_argo_smoke"), p_bin, p_time, p_lat, p_lon,
        np.asarray(p_t, np.float32), np.asarray(p_s, np.float32),
        levels=SMOKE_F8_LEVELS,
        note="ml/train_cone.py::smoke_family8 — synthetic, no network")
    return stem + ".npz", store


class FiniteView:
    """`isfinite(X)` derived PER GATHER instead of materialised.

    The same argument as `ml/model.py::LazyPixels`: at family 4's pentad shape
    an eager mask is 16.6 GB. ConeSampler indexes it with broadcast index
    arrays and nothing else, so computing `isfinite` after the index is
    arithmetically identical. Used only when X is too big to mask eagerly —
    below the threshold the eager bool array is contiguous and takes the
    sampler's fast flat-gather path, which measured ~2x faster.
    """

    def __init__(self, X):
        self._X = X
        self.shape = X.shape
        self.dtype = np.dtype(bool)

    def __getitem__(self, idx):
        return np.isfinite(self._X[idx])


def group_names(d):
    """The multi-resolution groups of a family-7 tensor, or [] for one array.

    `tensor_io._Tensor` answers `.groups` for BOTH layouts — `["X"]` for the
    single-array one — so the single name "X" is what says "this is not a
    multi-group tensor", and a bare `np.load` result answers nothing at all.
    """
    return [str(g) for g in getattr(d, "groups", []) if str(g) != "X"]


def anchor_mask(gs, chunk=None, verbose=True):
    """[H, W] over the MASTER grid: was any channel ever observed here?

    This is what replaces `isfinite(X[..., 0]).any(0)` for family 7, and the
    replacement is not cosmetic. Channel 0 of the dense group is `cur_speed`,
    an OCEAN channel, so the old expression means "is there ocean here" — the
    right question for a tensor whose land cells carry nothing, and the wrong
    one for a tensor built precisely so that land is observed (E-070's own
    line: *"The `slab[~ocean] = NaN` of family 4 is gone: land is observed
    now"*). Under the old rule the Sahara, Antarctica and every ice sheet
    would be excluded from the anchor pool of the first global tensor.

    So: a cell is an anchor if ANY group observes ANY channel there in ANY
    bin — the dense group at the cell itself, a coarse group at the cell the
    lookup sends it to. Walked in time chunks, because the dense group is
    46 GB and `isfinite` of it is 23.
    """
    m = gs.master
    out = np.zeros((m.H, m.W), bool)
    for g in gs.groups:
        step = int(chunk or max(1, (1 << 28) // max(1, g.H * g.W * g.C)))
        hit = np.zeros((g.H, g.W), bool)
        for i0 in range(0, g.T, step):
            blk = np.asarray(g.X[i0:i0 + step])
            hit |= np.isfinite(blk).any(axis=(0, 3))
            del blk
        if g.factor == 1:
            out |= hit
        else:
            y1 = np.minimum(
                np.floor(np.arange(m.H) / g.factor + 0.5).astype(np.int64),
                g.H - 1)
            x1 = (np.floor(np.arange(m.W) / g.factor + 0.5).astype(np.int64)
                  % g.W)
            out |= hit[y1[:, None], x1[None, :]]
        if verbose:
            print(f"  anchor mask: {g.name} contributes "
                  f"{int(hit.sum()):,} of its {g.H * g.W:,} cells "
                  f"(mask now {int(out.sum()):,}/{m.H * m.W:,})", flush=True)
    return out


def load_data_family7(a, d, gnames):
    """`load_data` for the three-group global tensor (E-070 §1-§3).

    Same contract as the single-array path — the returned dict has the same
    keys and `train_one` does not know which one it got — with three
    differences that are properties of the tensor and not choices:

      * `X` is a `cone_sampler.GroupSet` rather than an array, so the sampler
        can read a coarse channel at the coarse cell instead of a 425 GB
        upsampled copy of it (E-070 B2);
      * the anomaly transform runs PER GROUP, each with its own months —
        `rg100`'s rows are one per month, so its `moy`/`t_hold` are the
        master's indexed by the rows it actually holds, and handing it the
        master's 3,142-long arrays would charge every profile to the wrong
        calendar month;
      * the anchor mask is "any channel ever observed here" (`anchor_mask`),
        not "channel 0 is finite here".
    """
    from tensor_io import anomaly_chunk, anomaly_peak_bytes, writable_copy
    from cone_sampler import GroupSet, group_time
    from trainprobe import anomaly_transform

    months = [str(m) for m in d["months"]]
    lats, lons = np.asarray(d["lats"]), np.asarray(d["lons"])
    probe = GroupSet.from_tensor(d)
    T, H, W = probe.master.T, probe.master.H, probe.master.W
    chan = list(probe.chan)
    C = len(chan)
    print(f"family 7: groups {probe.names} · master [T={T} H={H} W={W}] · "
          f"{C} channels "
          f"({' + '.join(f'{g.name} {g.C}' for g in probe.groups)})",
          flush=True)

    hold_years = set(a.holdout_years.split(","))
    t_hold = np.array([m[:4] in hold_years for m in months])
    if not t_hold.any():
        raise SystemExit(
            f"--holdout-years {a.holdout_years!r} matches no bin in this "
            f"tensor ({months[0]} .. {months[-1]}) — there would be no "
            f"held-out loss to read, and a run that cannot answer its own "
            f"question is not a run (ml/CLAUDE.md §4.11).")
    moy = np.array([int(m[5:7]) - 1 for m in months])

    # BEFORE the transform, as the single-array path computes its `ocean` at
    # line 341: the anchor pool is a statement about what the DATA observes,
    # and the transform turns a cell whose month had no training sample into
    # NaN — a property of the holdout, not of the world.
    ocean = anchor_mask(probe)

    arrays, dynamic, group_stats = {}, {}, {}
    want_stats = bool(getattr(a, "argo_store", ""))
    for g in probe.groups:
        A = g.X
        if isinstance(A, np.memmap) and not A.flags.writeable:
            # train.py's rule, unchanged: the anomaly transform writes in
            # place, and an r+ map on the canonical tensor would leave
            # anomaly-space data where state-space data is documented.
            scratch = f"{a.tensor[:-4]}_cone_scratch_{g.name}.npy"
            print(f"  {g.name} is a read-only map — writable scratch copy at "
                  f"{scratch}", flush=True)
            A = writable_copy(A, scratch)
        gmoy, ghold = group_time(g, moy, t_hold)
        ch = anomaly_chunk(A.shape, np.dtype(A.dtype).itemsize)
        print(f"  {g.name}: anomaly_transform chunk {ch} "
              f"(peak ~{anomaly_peak_bytes(A.shape, ch, np.dtype(A.dtype).itemsize) / 1e9:.1f} GB, "
              f"{g.T} rows)", flush=True)
        # E-076 section 2.5: the family-8 arm has to put a RAW profile through
        # the identical chain, so the climatology and the pooled moments of
        # the group it replaces are captured here — from the one anomaly
        # transform, never recomputed beside it. Only for that group: the
        # dense group's climatology is 349 MB and nothing asks for it.
        stats = {} if (want_stats and g.name == PROFILE_GROUP) else None
        A, dyn = anomaly_transform(A, gmoy, ghold, np.zeros(g.W, bool),
                                   chunk=ch, stats=stats)
        if stats is not None:
            group_stats[g.name] = stats
        arrays[g.name] = A
        dynamic[g.name] = [int(c) for c in dyn]
        print(f"  {g.name}: anomaly space, {len(dyn)}/{g.C} dynamic "
              f"({[g.chan[c] for c in dyn]})", flush=True)

    gs = GroupSet.from_tensor(d, arrays=arrays)
    print(f"held-out bins {int(t_hold.sum())}/{T} · anchor cells "
          f"{int(ocean.sum()):,}/{H * W:,}", flush=True)

    # The global channel index of each group's dynamic channels, so the
    # metrics record means the same thing it does on the single-array path.
    base, flat = 0, []
    for g in gs.groups:
        flat += [base + c for c in dynamic[g.name]]
        base += g.C
    norm = {"space": "anomaly",
            "groups": gs.names,
            "dynamic": flat,
            "dynamic_by_group": dynamic,
            "holdout_years": a.holdout_years,
            "tensor_norm": {g: np.asarray(d[f"norm_{g}"]).tolist()
                            for g in gs.names if f"norm_{g}" in d}}
    return dict(X=gs, OBS=None, months=months, lats=lats, lons=lons,
                chan=chan, t_hold=t_hold, ocean=ocean, norm=norm,
                T=T, H=H, W=W, C=C, stats=group_stats,
                group_norm={g: np.asarray(d[f"norm_{g}"])
                            for g in gs.names if f"norm_{g}" in d},
                # ABSOLUTE pentad bins per master row. The observation store
                # is indexed by those, this tensor by row, and on any
                # sub-range build the two differ.
                bin_index=(np.asarray(d["bin_index"], np.int64)
                           if "bin_index" in d else None))


def load_data(a):
    """Load the tensor, take the anomaly transform, and return everything the
    sampler and the pools need. Mirrors ml/train.py's preamble.

    A family-7 tensor (three co-registered groups at two resolutions) goes to
    `load_data_family7`; everything with one dense array takes the path below,
    byte for byte as it always did.
    """
    from tensor_io import load_tensor, writable_copy
    d = load_tensor(a.tensor, allow_pickle=False)
    gnames = group_names(d)
    if gnames:
        return load_data_family7(a, d, gnames)
    X = d["X"]
    months = [str(m) for m in d["months"]]
    lats, lons = np.asarray(d["lats"]), np.asarray(d["lons"])
    chan = [str(c) for c in d["chan"]]
    T, H, W, C = X.shape
    print(f"X [T={T} H={H} W={W} C={C}] · channels {chan}", flush=True)

    if isinstance(X, np.memmap) and not X.flags.writeable:
        # train.py's rule, unchanged: the anomaly transform writes in place,
        # and an r+ map on the canonical tensor would leave anomaly-space data
        # where state-space data is documented.
        scratch = a.tensor[:-4] + "_cone_scratch.npy"
        print(f"X is a read-only map — writable scratch copy at {scratch}",
              flush=True)
        X = writable_copy(X, scratch)

    hold_years = set(a.holdout_years.split(","))
    t_hold = np.array([m[:4] in hold_years for m in months])
    if not t_hold.any():
        raise SystemExit(
            f"--holdout-years {a.holdout_years!r} matches no bin in this "
            f"tensor ({months[0]} .. {months[-1]}) — there would be no "
            f"held-out loss to read, and a run that cannot answer its own "
            f"question is not a run (ml/CLAUDE.md §4.11).")
    ocean = np.isfinite(X[..., 0]).any(axis=0)
    print(f"held-out bins {int(t_hold.sum())}/{T} · ocean cells "
          f"{int(ocean.sum())}", flush=True)

    # THE ONE ANOMALY TRANSFORM (tests/test_one_anomaly_transform.py). x_hold
    # is empty by construction: this trainer has no longitude holdout.
    from trainprobe import anomaly_transform
    moy = np.array([int(m[5:7]) - 1 for m in months])
    X, dynamic = anomaly_transform(X, moy, t_hold, np.zeros(W, bool))
    print(f"anomaly space: {len(dynamic)}/{C} dynamic channels "
          f"({[chan[c] for c in dynamic]})", flush=True)

    # OBS: family 2-5 carry no mask (train.py derives it as isfinite(X) —
    # LazyPixels), so that is the default here; a tensor that DOES ship one is
    # honoured, intersected with isfinite because the anomaly transform makes
    # a cell unobserved wherever its month had no training sample at all.
    if "OBS" in d:
        OBS = np.asarray(d["OBS"], bool) & np.isfinite(X)
    else:
        OBS = np.isfinite(X) if X.nbytes <= (4 << 30) else FiniteView(X)
    norm = {"space": "anomaly",
            "dynamic": [int(c) for c in dynamic],
            "holdout_years": a.holdout_years,
            "tensor_norm": (np.asarray(d["norm"]).tolist()
                            if "norm" in d else None)}
    return dict(X=X, OBS=OBS, months=months, lats=lats, lons=lons, chan=chan,
                t_hold=t_hold, ocean=ocean, norm=norm, T=T, H=H, W=W, C=C,
                stats={}, group_norm={}, bin_index=None)


# ------------------------------------------------------------------ anchors --
def admissible_bins(sampler, train_bins):
    """Which t a TRAINING anchor may sit on: admissibility depends on t alone
    (`ConeSampler.admissible` reads only the bin span), so the whole set is one
    vectorised call rather than a rejection loop over pixels."""
    T = sampler.T
    probe = np.stack([np.arange(T), np.zeros(T, np.int64),
                      np.zeros(T, np.int64)], axis=1)
    return np.flatnonzero(sampler.admissible(probe, train_bins))


def draw_anchors(rng, ts, ys, xs, n):
    """n anchors drawn uniformly from `ts` x (ocean cells)."""
    it = rng.integers(0, len(ts), n)
    ip = rng.integers(0, len(ys), n)
    return np.stack([ts[it], ys[ip], xs[ip]], axis=1).astype(np.int64)


def to_torch(s, chan_depth, device, extras=()):
    """The sampler's numpy batch as the tensors ConeMAE.forward reads.

    `extras` names which of the per-dot extra fields are LIVE (`dot_extras_of`
    parses the flag). The tensor handed to the model is always three wide when
    any is live — [nr, fp0, fp1] — with the ones not named zeroed, so the
    architecture does not move between the arms of E-076 section 5.

    A batch the sampler drew with `withhold_nearest` carries `profile_target`;
    its ready-made query block travels as the `pq_*` keys, which is what makes
    masked-profile reconstruction ride the existing dot-query machinery.
    """
    b = {}
    for k in ("vals", "dy_km", "dx_km", "lag_days", "depth", "patch_vals",
              "fut_vals", "ctx"):
        b[k] = torch.as_tensor(np.ascontiguousarray(s[k]),
                               dtype=torch.float32, device=device)
    for k in ("obs", "valid", "patch_obs", "fut_obs"):
        b[k] = torch.as_tensor(np.ascontiguousarray(s[k]), device=device)
    b["chan"] = torch.as_tensor(s["chan"].astype(np.int64), device=device)
    b["chan_depth"] = chan_depth
    if extras:
        nr = np.asarray(s["nr"], np.float32)[..., None]
        fp = np.asarray(s["fp"], np.float32)
        cols = [nr if "nr" in extras else np.zeros_like(nr),
                fp if "fp" in extras else np.zeros_like(fp)]
        b["dot_extra"] = torch.as_tensor(
            np.ascontiguousarray(np.concatenate(cols, axis=-1)),
            dtype=torch.float32, device=device)
    pt = s.get("profile_target")
    if pt is not None:
        b["pq_chan"] = torch.as_tensor(pt["q_chan"].astype(np.int64),
                                       device=device)
        for k in ("dy_km", "dx_km", "lag_days", "depth", "vals"):
            b[f"pq_{k}"] = torch.as_tensor(
                np.ascontiguousarray(pt[f"q_{k}"]), dtype=torch.float32,
                device=device)
        b["pq_obs"] = torch.as_tensor(np.ascontiguousarray(pt["q_obs"]),
                                      device=device)
    return b


# ------------------------------------------------------------------ training --
def eval_generator(device, seed=12345):
    """A seeded torch.Generator ON THE DEVICE THE MASKS ARE DRAWN ON.

    `torch.rand(..., device=dev, generator=g)` refuses a generator whose
    device differs from `dev`. The first CUDA run of this trainer (#536,
    2026-09-03) died at its first eval, 73 minutes in, after the anomaly
    transform and the pool certificate had both passed, with exactly that
    refusal — every earlier run was CPU-only, where a CPU generator is
    trivially the right one. A CUDA generator and a CPU generator seeded
    alike draw DIFFERENT streams, so a held-out loss is comparable across
    evals of one run (same device, same seed), never across backends — which
    §3b already says."""
    dev = torch.device(device) if not isinstance(device, torch.device) else device
    return torch.Generator(device=dev.type).manual_seed(int(seed))


QUERY_FAMILIES = ("anchor", "future", "dots")
# The families whose per-channel split reaches the metrics record. See
# `fam_record`; the eval computes all three, only these two are written.
BY_CHAN_FAMILIES = ("anchor", "dots")
# The channels the job log prints per arm. `cur_u` / `cur_v` are H1's target
# pair and the asymmetry #540 (E-069 seed 2, the cone codec) found; `cur_speed`
# is their magnitude, and `ssh` / `sst` are the two family-B/C channels a
# reader compares them against.
LOG_CHANS = ("cur_u", "cur_v", "cur_speed", "ssh", "sst")


def eval_loss(model, sampler, anchors, plan, chan_depth, device, batch,
              seed=12345, extras=()):
    """Held-out loss on a FIXED anchor set with a FIXED mask draw.

    The generator is re-seeded at every eval, so two evals differ only in the
    weights — the curve measures the model, not which channels the dice hid.

    Returns `(nll, mse, n_targets, families)`. The first three are exactly the
    numbers this function has always returned — same accumulation, same
    formula. `families` is the same total split three ways by what the
    decoder was ASKED (ConeMAE.query_family_spans): `anchor` is the anchor's
    own value in every channel at lag 0, `future` is the anchor column at t+1
    and t+2 pentads, `dots` is the subsample of hidden cone dots. Each carries
    its weighted mean nll and mse, the weight those means divide by, and the
    number of scored targets, so
        nll == sum_f nll_f * wsum_f / sum_f wsum_f
    holds to floating point (tests/test_cone_smoke.py pins it at 1e-6). This
    is the measurement that decides H1's hypothesis (c): the cone's headline
    NLL is over 244,634 targets and the twin's over 42,937, so the two are
    not comparable AS TOTALS — but their `anchor` and `future` families are
    the same question asked of both arms.

    Each family additionally carries `mse_by_chan` / `msebar_by_chan` /
    `wsum_by_chan` — the SAME weighted means, split by the query's own channel
    index (`ConeMAE._loss_from`'s `families_by_chan`), accumulated over the
    whole pass exactly the way the family means are. The channel order is the
    tensor's `chan` list, which the config record already carries. This is
    what says whether the decoder reconstructs a given channel at all: #540
    (E-069 seed 2, the cone codec) recovers `cur_u` from the embedding at
    R² 0.82 and `cur_v` at 0.02 with both channels VISIBLE in the input, and
    no hypothesis predicted an east/north asymmetry — a per-family mse cannot
    see it because both channels are inside the same family.

    A channel with zero weight (never scored in this pass) reports 0.0, never
    0/0 — ml/CLAUDE.md §5.22 — and its `wsum_by_chan` entry is 0.0, which is
    the honest discriminator between "scored zero" and "scored nothing".
    """
    g = eval_generator(device, seed)
    p = dict(plan)
    p["generator"] = g
    C = int(model.n_chan)
    model.eval()
    nll = mse = w = tgt = 0.0
    fam = {k: {"nll": 0.0, "mse": 0.0, "msebar": 0.0, "wsum": 0.0,
               "n_targets": 0.0, "mse_by_chan": [0.0] * C,
               "msebar_by_chan": [0.0] * C, "wsum_by_chan": [0.0] * C}
           for k in QUERY_FAMILIES}
    with torch.no_grad():
        for i in range(0, len(anchors), batch):
            # `train_pool=False`: a HELD-OUT sample may read profiles from
            # held-out bins, exactly as it may read dots from them — that is
            # what makes it a held-out measurement rather than a second
            # training pool.
            s = sampler.sample(anchors[i:i + batch], train_pool=False)
            b = to_torch(s, chan_depth, device, extras)
            out = model(b, p)
            n = out["terms"]["wsum"]
            nll += out["terms"]["nll"] * n
            mse += out["terms"]["mse"] * n
            w += n
            tgt += out["terms"]["n_targets"]
            bc = out.get("families_by_chan", {})
            for k, f in out.get("families", {}).items():
                if k not in fam:
                    continue
                fam[k]["nll"] += f["nll"] * f["wsum"]
                fam[k]["mse"] += f["mse"] * f["wsum"]
                fam[k]["msebar"] += f.get("msebar", 0.0) * f["wsum"]
                fam[k]["wsum"] += f["wsum"]
                fam[k]["n_targets"] += f["n_targets"]
                fk = bc.get(k)
                if not fk:
                    continue
                for c in range(C):
                    wc = fk["wsum"][c]
                    fam[k]["mse_by_chan"][c] += fk["mse"][c] * wc
                    fam[k]["msebar_by_chan"][c] += fk["msebar"][c] * wc
                    fam[k]["wsum_by_chan"][c] += wc
    model.train()
    w = max(w, 1e-6)
    for f in fam.values():
        # An EMPTY family (the snapshot twin has no dots) reports zeros and a
        # zero weight rather than 0/0 — ml/CLAUDE.md §5.22, never write a NaN
        # into a results file. A reader tells "no targets" from "a loss of
        # zero" by the weight, which is the honest discriminator.
        den = f["wsum"] if f["wsum"] > 0.0 else 1.0
        f["nll"] /= den
        f["mse"] /= den
        f["msebar"] /= den
        # The same rule per channel, and for the same reason: a channel this
        # family never scored (family A carries no `cur_*` weight under
        # `anchor_hidden_only` on an element that dropped nothing) divides by
        # its own zero weight, and 0/0 is the NaN §5.22 forbids.
        for c in range(C):
            dc = f["wsum_by_chan"][c] if f["wsum_by_chan"][c] > 0.0 else 1.0
            f["mse_by_chan"][c] /= dc
            f["msebar_by_chan"][c] /= dc
    return nll / w, mse / w, tgt, fam


def fam_record(fam, by_chan=True):
    """The per-family eval keys, flat, for one metrics.jsonl record.

    Additive by construction: the headline `held_out_nll` / `held_out_mse` /
    `held_out_targets` keys are written by the caller and are unchanged, so
    status.html and every archived reader keep working; these sit beside them
    (ml/CLAUDE.md §0d — a reader ignores what it does not know). Every value is
    a finite number, including for a family with no targets (§5.22).

    `by_chan` adds the four LIST-valued keys of the per-channel split —
    `held_out_mse_{anchor,dots}_by_chan` and the matching `msebar` bars, each C
    floats in the tensor's own `chan` order (the config record carries that
    list). Only the ANCHOR and DOT families get them: those are the two that
    say whether a channel is reconstructed at all, and the future family would
    be C more floats per record for a question nothing is asking yet. It is a
    parameter rather than always-on so a caller can keep a live metrics file
    small. Measured at C = 39 (the r3 tensor) the four lists take a record from
    612 to 2,128 bytes — inside the ~4 KB the live file can carry — so
    `train_one` writes them at EVERY eval rather than only at step 0 and the
    final one: eleven such records is ~23 KB, and a per-channel curve is worth
    more than the bytes (§0d — a number a reader has to open the archive for is
    a number nobody reads).
    """
    rec = {}
    for k in QUERY_FAMILIES:
        f = fam.get(k) or {"nll": 0.0, "mse": 0.0, "msebar": 0.0,
                           "wsum": 0.0, "n_targets": 0.0}
        rec[f"held_out_nll_{k}"] = round(float(f["nll"]), 5)
        rec[f"held_out_mse_{k}"] = round(float(f["mse"]), 5)
        # THE PREDICT-ZERO BAR for the same family, on the same weights: the
        # weighted mean of the squared target. `mse` alone cannot say whether
        # a family was learnt — the targets are standardised per channel, so
        # a family sitting AT its msebar is predicting the mean, which is what
        # #539's hidden-dot 1.001 turned out to be.
        rec[f"held_out_msebar_{k}"] = round(float(f.get("msebar", 0.0)), 5)
        rec[f"held_out_targets_{k}"] = int(f["n_targets"])
        # The WEIGHT is what the two means above divide by, and it is what
        # lets a reader put the headline back together (the family weights of
        # cone_codec.FAMILY_W are inside it, so it is not the target count).
        rec[f"held_out_wsum_{k}"] = round(float(f["wsum"]), 4)
        if by_chan and k in BY_CHAN_FAMILIES:
            # 0.0 where the channel carried no weight, never a NaN (§5.22).
            # The list is positional — the channel ORDER is the tensor's `chan`
            # list, which the config record already carries, so the record does
            # not restate 39 names at every eval.
            for stat in ("mse", "msebar"):
                v = f.get(f"{stat}_by_chan") or []
                rec[f"held_out_{stat}_{k}_by_chan"] = [
                    round(float(x), 5) for x in v]
    return rec


def fam_line(fam):
    """One human line: `anchor +1.23 (n) · future … · dots …`."""
    return " · ".join(
        f"{k} {fam[k]['nll']:+.3f}/{int(fam[k]['n_targets']):,}"
        for k in QUERY_FAMILIES if k in fam)


def chan_mse_line(fam, chan, names=LOG_CHANS, family="anchor"):
    """`cur_u 0.1832 · cur_v 0.9814 · …` — one family's per-channel held-out MSE.

    Read off the FINAL eval record so the job log carries the number without
    anyone opening the archive: #540 (E-069 seed 2, the cone codec) recovered
    `cur_u` from the embedding at R² 0.82 and `cur_v` at 0.02, and the question
    that asymmetry raises — does the decoder reconstruct `cur_v` at all — is
    answered by this line and by nothing else in the log.

    A channel the tensor does not carry, or one this family never scored,
    prints `n/a` rather than a zero: 0.0 is a legitimate MSE and would be a
    lie here (ml/CLAUDE.md §5.22's display half — never print a number where
    the honest answer is "no reading").
    """
    f = (fam or {}).get(family) or {}
    mse = f.get("mse_by_chan") or []
    ws = f.get("wsum_by_chan") or []
    out = []
    for n in names:
        i = chan.index(n) if n in chan else -1
        if i < 0 or i >= len(mse) or (i < len(ws) and ws[i] <= 0.0):
            out.append(f"{n} n/a")
        else:
            out.append(f"{n} {mse[i]:.4f}")
    return " · ".join(out)


_STORES = {}


def open_store(a):
    """The family-8 store named by `--argo-store`, VERIFIED before it is used.

    The verification is the dispatch-time half of ml/CLAUDE.md §0.3: a
    truncated `temp.npy` still memmaps and still answers a search, with
    whatever its tail happens to hold, and the run would train on it and
    report numbers. Cached per path so the twin arm does not re-hash 234 MB.
    """
    from family8_store import ArgoStore, verify_store
    path = os.path.abspath(a.argo_store)
    if path in _STORES:
        return _STORES[path]
    n = verify_store(path)
    st = ArgoStore(path)
    print(f"family 8 store {path}: {len(st):,} profiles over {st.n_bins} "
          f"pentad bins, {n} files sha256-verified against store.json",
          flush=True)
    _STORES[path] = st
    return st


# --------------------------------------------- E-076 §5.1, the common target --
HELDOUT_SPLITS = ("2021-2024", "interspersed")


def heldout_split_of(year):
    """Which half of the frozen protocol's holdout a year belongs to.

    E-076 §5.1 reads the two apart because they are different questions: the
    2021-2024 block is the TERMINAL holdout (can the codec do this at the end
    of the archive, where a forecast would run), while 2008/2009/2016/2017 are
    interspersed years surrounded by training data.
    """
    return HELDOUT_SPLITS[0] if int(year) >= 2021 else HELDOUT_SPLITS[1]


class _RMSE:
    """Squared-error sums per channel, for one population of anchors."""

    def __init__(self, n_chan):
        self.n = 0
        self.se = np.zeros((3, n_chan), np.float64)     # model, clim, pers
        self.cnt = np.zeros((3, n_chan), np.float64)

    def add(self, pred, target, mask, row):
        d = np.where(mask, np.asarray(pred, np.float64)
                     - np.asarray(target, np.float64), 0.0)
        self.se[row] += (d * d).sum(axis=0)
        self.cnt[row] += mask.sum(axis=0)

    def rmse(self, row, sel):
        """[len(sel)] RMSE over the channels `sel`, NaN-free: a level nothing
        scored reports `None` rather than 0.0 or a NaN (ml/CLAUDE.md §5.22)."""
        out = []
        for c in sel:
            k = self.cnt[row, c]
            out.append(float(np.sqrt(self.se[row, c] / k)) if k > 0 else None)
        return out

    def record(self, sel_t, sel_s):
        rec = {"n": int(self.n)}
        for row, name in ((0, ""), (1, "clim_"), (2, "pers_")):
            rec[f"{name}rmse_t"] = self.rmse(row, sel_t)
            rec[f"{name}rmse_s"] = self.rmse(row, sel_s)
        for var, sel in (("t", sel_t), ("s", sel_s)):
            # SKILL is the mean over levels of model RMSE / climatology RMSE:
            # below 1 the model beats predicting the seasonal normal, which is
            # the bar E-076 §5.1 names first. Levels neither of them scored are
            # left out rather than counted as a ratio of nothing.
            num, den = rec[f"rmse_{var}"], rec[f"clim_rmse_{var}"]
            r = [n / d for n, d in zip(num, den)
                 if n is not None and d not in (None, 0.0)]
            rec[f"skill_{var}"] = float(np.mean(r)) if r else None
        return rec


def profile_target_anchors(a, D, sampler, ys, xs, n, seed=20260907):
    """A FIXED set of held-out anchors with at least two profiles in range.

    E-076 §5.1's protocol: held-out bins, ocean cells, `n_R >= 2` — two,
    because the eval withholds the nearest profile and still needs one left to
    read the persistence baseline off. Drawn ONCE with its own seed, so the
    curve compares models rather than anchor sets, and by rejection because
    "how many profiles are within 1,000 km of this cell in this pentad" is a
    property of the observing system that no mask can be precomputed from
    cheaply.
    """
    P = sampler.profile
    ts = np.flatnonzero(D["t_hold"])
    ts = ts[(ts - sampler.L_in >= 0) & (ts + max(sampler.future_lags)
                                        < sampler.T)]
    if not len(ts) or n <= 0:
        return np.zeros((0, 3), np.int64)
    rng = np.random.default_rng(seed)
    keep, tried = [], 0
    cap = 200 * n + 10_000
    while len(keep) < n and tried < cap:
        cand = draw_anchors(rng, ts, ys, xs, min(4 * n, 4096))
        tried += len(cand)
        for t, y, x in cand:
            res = P.knearest(float(sampler.lats[y]), float(sampler.lons[x]),
                             int(t), 2)
            if int(res["n_R"]) >= 2:
                keep.append((int(t), int(y), int(x)))
                if len(keep) >= n:
                    break
    return np.asarray(keep, np.int64).reshape(-1, 3)


def profile_target_eval(model, sampler, anchors, months, chan_depth, device,
                        batch=64, extras=(), step=0):
    """E-076 §5.1's COMMON TARGET: the nearest real profile, raw units.

    IDENTICAL CODE FOR BOTH ARMS, which is the whole point of the section. The
    family-8 arm's input has the target profile withheld (`withhold_nearest`
    removes it from the token set); the twin's input is the gridded `rg100`
    column it always had, because `--profile-k 0` leaves the dense path alone.
    Either way the decoder is asked the same 32 questions at the same
    coordinates and scored against the same measurement.

    The prediction comes back in the model's anomaly space and is inverted at
    the PROFILE'S OWN cell and month — the same chain E-076 §2.5 applies going
    in, read backwards:

        z   = a * den_c + mu_c + clim[month, y1, x1, c]
        raw = z * norm_sd_c + norm_mean_c

    Three read-outs on identical masks: the model, the CLIMATOLOGY (a = 0, so
    the seasonal normal at that cell) and PERSISTENCE (the nearest REMAINING
    profile's own values, copied). A model that does not beat both has not
    used the displacement information (docs/FAMILY8_DATA_HANDOVER.md §7.5).
    """
    P = sampler.profile
    Cg = P.n_chan
    # Which columns of the 32 are temperature and which salinity, each in
    # DEPTH ORDER — the `rmse_t[16]` and `rmse_s[16]` of E-076 §5.1 are per
    # level, and the channel order is the tensor's, not the store's.
    idx_t = np.flatnonzero(P.value_col == 0)[np.argsort(
        P.level[P.value_col == 0], kind="stable")]
    idx_s = np.flatnonzero(P.value_col == 1)[np.argsort(
        P.level[P.value_col == 1], kind="stable")]
    pops = {"all": _RMSE(Cg)}
    for k in HELDOUT_SPLITS:
        pops[k] = _RMSE(Cg)
    model.eval()
    with torch.no_grad():
        for i in range(0, len(anchors), batch):
            aa = anchors[i:i + batch]
            s = sampler.sample(aa, withhold_nearest=True, train_pool=False)
            pt = s["profile_target"]
            b = to_torch(s, chan_depth, device, extras)
            z, _ = model.encode(b)
            q = model.query_tokens(b["pq_chan"].long(), b["pq_dy_km"],
                                   b["pq_dx_km"], b["pq_lag_days"],
                                   b["pq_depth"])
            mu, _lv = model.decode_from_z(z, q)
            a_hat = mu.detach().cpu().numpy().astype(np.float64)

            month, y1, x1 = pt["month"], pt["y1"], pt["x1"]
            cl = np.asarray(P.clim[month, y1, x1, :], np.float64)   # [B, Cg]
            def to_raw(a_val):
                zz = a_val * P.den[None, :] + P.mu[None, :] + cl
                return zz * P.norm_sd[None, :] + P.norm_mean[None, :]
            both = np.stack([pt["temp"], pt["psal"]], axis=0)        # [2,B,16]
            tgt = both[P.value_col, :, P.level].T.astype(np.float64)
            pers = np.stack([pt["pers_temp"], pt["pers_psal"]],
                            axis=0)[P.value_col, :, P.level].T.astype(np.float64)
            # ONE mask for the model and the climatology: `q_obs` is exactly
            # "the level is filled AND the chain is defined at this cell and
            # month", so the two baselines are scored on the same targets the
            # model is. Persistence additionally needs its own profile to have
            # measured that level.
            m = np.asarray(pt["q_obs"], bool)
            mp = m & np.isfinite(pers) & np.asarray(pt["pers_valid"], bool)[:, None]
            pred = to_raw(a_hat)
            climp = to_raw(np.zeros_like(a_hat))
            years = np.array([int(months[int(t)][:4]) for t in aa[:, 0]])
            groups = [("all", np.ones(len(aa), bool))]
            for name in HELDOUT_SPLITS:
                groups.append((name, np.array(
                    [heldout_split_of(y) == name for y in years])))
            for name, gm in groups:
                if not gm.any():
                    continue
                acc = pops[name]
                acc.n += int((m[gm].any(axis=1)).sum())
                acc.add(pred[gm], tgt[gm], m[gm], 0)
                acc.add(climp[gm], tgt[gm], m[gm], 1)
                acc.add(pers[gm], tgt[gm], mp[gm], 2)
    model.train()
    rec = {"step": int(step), "n_anchors": int(len(anchors)),
           "levels": [float(v) for v in P.store.levels]}
    rec.update(pops["all"].record(idx_t, idx_s))
    rec["heldout_split"] = {k: pops[k].record(idx_t, idx_s)
                            for k in HELDOUT_SPLITS}
    return rec


def write_profile_target(out_dir, rec, in_progress=False,
                         name="profile_target.json"):
    """`profile_target.json`, written atomically (ml/CLAUDE.md §5.25).

    Temp sibling then `os.replace`, so a reader never catches a half-written
    file, and every write before the last carries `in_progress` — those
    numbers are real, and the run they belong to has not finished.
    """
    path = os.path.join(out_dir, name)
    blob = dict(rec)
    if in_progress:
        blob["in_progress"] = True
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(blob, f, indent=2)
    os.replace(tmp, path)
    return path


def profile_line(rec):
    """One human line: the two skills and the level-500 numbers."""
    def at(key, j=8):
        v = (rec.get(key) or [])
        return "n/a" if j >= len(v) or v[j] is None else f"{v[j]:.3f}"
    sk = lambda k: ("n/a" if rec.get(k) is None else f"{rec[k]:.3f}")  # noqa: E731
    return (f"profile target n={rec.get('n', 0)} · skill T {sk('skill_t')} "
            f"S {sk('skill_s')} · 500 dbar rmse T {at('rmse_t')} "
            f"(clim {at('clim_rmse_t')}, pers {at('pers_rmse_t')}) °C · "
            f"S {at('rmse_s')} (clim {at('clim_rmse_s')}, "
            f"pers {at('pers_rmse_s')}) PSU")


def run_profile_eval(a, tag, model, sampler, anchors, D, chan_depth, device,
                     extras, step, out_dir, name, metrics_path, final):
    """Score the common target, print it, and publish it — or do nothing.

    Called at every eval point INCLUDING step 0, so the file exists (with its
    climatology and persistence bars, which no training changes) before the
    first hour is spent, and is refreshed at every eval afterwards: E-076a's
    headline is exactly this number, and ml/CLAUDE.md §5.25 says a long job
    publishes its result as it goes rather than holding it in memory.
    """
    if anchors is None or not len(anchors):
        return None
    rec = profile_target_eval(model, sampler, anchors, D["months"], chan_depth,
                              device, batch=a.batch, extras=extras, step=step)
    rec["arm"] = tag
    print(f"[{tag}] step {step:>6} · {profile_line(rec)}", flush=True)
    write_profile_target(out_dir, rec, in_progress=not final, name=name)
    if metrics_path:
        with open(metrics_path, "a") as f:
            f.write(json.dumps({"step": int(step),
                                "profile_target": rec}) + "\n")
    return rec


def train_one(a, D, L_in, out_dir, metrics_name, ckpt_name, tag, device,
              eval_anchors=None):
    """Train one arm (the cone codec, or its L_in=0 snapshot twin).

    Returns dict(model, sampler, curve, certificate, params).
    """
    torch.manual_seed(a.seed)
    rng = np.random.default_rng(a.seed)
    chan, C = D["chan"], D["C"]
    fut = tuple(int(v) for v in a.future_lags.split(",") if v.strip())
    train_bins = ~D["t_hold"]
    n_extra, extras = dot_extras_of(a.dot_extras)
    store = open_store(a) if a.argo_store else None
    sampler = ConeSampler(
        D["X"], D["OBS"], D["lats"], D["lons"], chan, L_in=L_in,
        future_lags=fut, profile_store=store, profile_k=a.profile_k,
        profile_group=PROFILE_GROUP,
        profile_norm=(D.get("group_norm") or {}).get(PROFILE_GROUP),
        profile_stats=(D.get("stats") or {}).get(PROFILE_GROUP),
        profile_R_max_km=a.profile_R_max_km,
        profile_T_max_days=a.profile_T_max_days,
        profile_bin_index=D.get("bin_index"), train_bins=train_bins)
    ts = admissible_bins(sampler, train_bins)
    ys, xs = np.nonzero(D["ocean"])
    if not len(ts):
        raise SystemExit(
            f"[{tag}] no bin is admissible under --holdout-scope window with "
            f"L_in={L_in} and future lags {fut}: every anchor's cone would "
            f"touch a held-out or out-of-archive bin. Widen the archive or "
            f"shrink L_in.")
    n_dots = sampler.n_dots(int(ys[0]))
    print(f"[{tag}] L_in={L_in} · {n_dots} dot tokens + {C} patch tokens per "
          f"anchor · admissible train bins {len(ts)}/{sampler.T} · "
          f"{len(ys):,} ocean cells", flush=True)
    if store is not None:
        print(f"[{tag}] family 8: {len(store):,} profiles · k={a.profile_k} "
              f"({sampler.n_profile_dots} profile dot tokens per anchor) · "
              f"R_max {a.profile_R_max_km:g} km · T_max "
              f"{a.profile_T_max_days:g} d · dot extras "
              f"{list(extras) or 'none'} (n_extra {n_extra}) · drop-p "
              f"{a.profile_drop_p:g}", flush=True)

    # ---- self-certification (E-059's pattern), BEFORE anything is spent ----
    cert = draw_anchors(rng, ts, ys, xs, min(a.certify_n, 4096))
    bad = sampler.certify(cert, train_bins)
    print(f"[{tag}] pool certificate: {bad} violations in {len(cert)} drawn "
          f"anchors (window scope, bins t-{L_in}..t+{max(fut)})", flush=True)
    if bad:
        raise SystemExit(
            f"[{tag}] POOL VIOLATION: {bad} of {len(cert)} training anchors "
            f"read a bin outside the training set. Refusing to train — a "
            f"codec trained on a leaked holdout cannot be evaluated on it "
            f"(ml/plans/E069_cone_codec.md §3, 'pool discipline').")

    model = ConeMAE(C, d_model=a.d_model, n_heads=a.n_heads,
                    n_latents=a.n_latents, n_layers=a.n_layers, d_z=a.d_z,
                    d_dec=a.d_dec, dec_layers=a.dec_layers,
                    n_fourier=a.n_fourier, n_extra=n_extra).to(device)
    params = model.param_count()
    print(f"[{tag}] ConeMAE {params:,} params "
          f"({params / 1e6:.3f}M)", flush=True)

    chan_depth = torch.as_tensor([channel_depth_dbar(n) for n in chan],
                                 dtype=torch.float32, device=device)
    # The masking plan. Both arms — the cone and its L_in=0 snapshot twin —
    # are built from the same `a`, so the twin masks under exactly the plan
    # the cone arm does; that is what makes the two comparable at all.
    plan = default_plan(chan, n_dot_queries=a.n_dot_queries,
                        aux_latent_w=a.aux_latent_w, future_lags=fut,
                        lag_band_p=a.lag_band_p, sector_p=a.sector_p,
                        chan_drop_scope=a.chan_drop_scope,
                        anchor_hidden_only=a.anchor_hidden_only,
                        device=device)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=a.steps)

    # ---- the held-out anchor set ------------------------------------------
    # Dots from held-out bins are ALLOWED here — that is what makes it a
    # held-out measurement rather than a second training pool.
    if eval_anchors is None:
        ev_ts = np.flatnonzero(D["t_hold"])
        ev_ts = ev_ts[(ev_ts - L_in >= 0) & (ev_ts + max(fut) < sampler.T)]
        if not len(ev_ts):
            raise SystemExit(f"[{tag}] no held-out bin has a complete cone")
        eval_anchors = draw_anchors(np.random.default_rng(a.seed + 991),
                                    ev_ts, ys, xs, a.eval_anchors)

    # ---- E-076 §5.1's fixed common-target anchor set -----------------------
    # Drawn ONCE, with its own seed, from held-out bins with at least two
    # profiles in range: the same anchors at every eval and — because the seed
    # does not depend on the arm — the same anchors in both arms of E-076a.
    prof_anchors = None
    prof_name = ("profile_target.json" if tag == "cone"
                 else f"profile_target_{tag}.json")
    if sampler.profile is not None and a.profile_eval_n > 0:
        prof_anchors = profile_target_anchors(a, D, sampler, ys, xs,
                                              a.profile_eval_n,
                                              seed=20260907 + a.seed)
        print(f"[{tag}] common-target eval: {len(prof_anchors)} held-out "
              f"anchors with n_R >= 2 (asked for {a.profile_eval_n})",
              flush=True)
        if not len(prof_anchors):
            print(f"[{tag}] ::warning:: no held-out anchor has two profiles "
                  f"in range — the common target of E-076 §5.1 cannot be "
                  f"scored on this tensor", flush=True)

    metrics_path = (os.path.join(out_dir, metrics_name)
                    if metrics_name and not os.path.isabs(metrics_name)
                    else metrics_name)
    if metrics_path:
        # train.py's RECORD FAMILY, key for key where the keys mean the same
        # thing (ml/CLAUDE.md §0d). status.html's parseJsonl resets on
        # `config`, charts {step, loss_rec, loss_nei}, and renders any other
        # {step, ...numbers} record as a probe line — so the cone trainer
        # needs no change to the page.
        with open(metrics_path, "a") as f:
            f.write(json.dumps({"config": {
                "steps": a.steps, "batch": a.batch, "d_z": a.d_z, "patch": 3,
                "d_model": a.d_model, "n_layers": a.n_layers,
                "n_heads": a.n_heads, "d_dec": a.d_dec, "anomaly": True,
                "eval_every": a.eval_every, "light_probe_every": 0,
                "params_M": round(params / 1e6, 3),
                "data": os.path.basename(a.tensor), "C": int(C),
                "T": int(sampler.T), "resume": None,
                "recipe": os.environ.get("RECIPE_NAME") or None,
                # cone-specific, additive: the page ignores what it does not
                # know, and this record is sometimes the only surviving
                # account of what ran (#387).
                "trainer": "cone", "arm": tag, "L_in": int(L_in),
                "n_latents": a.n_latents, "n_dot_tokens": int(n_dots),
                "future_lags": list(fut), "aux_latent_w": a.aux_latent_w,
                # E-069b's masking plan. Recorded because this record is
                # sometimes the only surviving account of what ran, and two
                # runs that differ only here produce completely different
                # loss splits.
                "chan_drop_scope": a.chan_drop_scope,
                "lag_band_p": a.lag_band_p, "sector_p": a.sector_p,
                "anchor_hidden_only": bool(a.anchor_hidden_only),
                # E-076a. The arm is a property of the RUN and this record is
                # sometimes the only surviving account of it (#387).
                "argo_store": os.path.basename(a.argo_store.rstrip("/"))
                              if a.argo_store else None,
                "profile_k": int(a.profile_k) if a.argo_store else 0,
                "profile_drop_p": (float(a.profile_drop_p) if a.argo_store
                                   and a.profile_k else 0.0),
                "dot_extras": ",".join(extras), "n_extra": int(n_extra),
                "n_profile_dots": int(sampler.n_profile_dots),
                "holdout_scope": a.holdout_scope,
                "holdout_years": a.holdout_years,
                "lr": a.lr, "seed": a.seed,
            }}) + "\n")

    def save(step):
        # `args` carries the whole CLI, so the four E-069b plan knobs are in
        # it by construction; they are restated beside `L_in` for the same
        # reason `L_in` is — a reader opening a checkpoint asks what stencil
        # and what masking plan produced these weights, and should not have
        # to know which argparse dest spells them.
        blob = {"args": vars(a), "model": model.state_dict(),
                "chan_names": chan, "norm": D["norm"], "step": int(step),
                "arm": tag, "L_in": int(L_in), "params": params,
                "chan_drop_scope": a.chan_drop_scope,
                "lag_band_p": float(a.lag_band_p),
                "sector_p": float(a.sector_p),
                "anchor_hidden_only": bool(a.anchor_hidden_only),
                # E-076a: the dot token's WIDTH is part of the architecture,
                # so a loader must find it here and not have to re-parse the
                # flag. An archived E-069 checkpoint has neither key and
                # rebuilds at n_extra 0, which is what it was trained with.
                "dot_extras": ",".join(extras), "n_extra": int(n_extra),
                "profile_k": int(a.profile_k) if a.argo_store else 0}
        torch.save(blob, os.path.join(out_dir, ckpt_name))

    loss_every = max(1, a.steps // 200)
    curve = []
    t0 = time.time()
    nll0, mse0, n0, fam0 = eval_loss(model, sampler, eval_anchors, plan,
                                     chan_depth, device, a.batch,
                                     extras=extras)
    curve.append({"step": 0, "held_out_nll": nll0, "held_out_mse": mse0,
                  "train_nll": None, "families": fam0})
    print(f"[{tag}] step 0 · held-out nll {nll0:+.4f} mse {mse0:.4f} "
          f"({int(n0):,} targets) · {fam_line(fam0)}", flush=True)
    if metrics_path:
        with open(metrics_path, "a") as f:
            rec = {"step": 0, "held_out_nll": round(nll0, 5),
                   "held_out_mse": round(mse0, 5),
                   "held_out_targets": int(n0)}
            rec.update(fam_record(fam0))
            rec["wall_s"] = round(time.time() - t0, 1)
            f.write(json.dumps(rec) + "\n")
    run_profile_eval(a, tag, model, sampler, prof_anchors, D, chan_depth,
                     device, extras, 0, out_dir, prof_name, metrics_path,
                     final=False)

    for s in range(1, a.steps + 1):
        anchors = draw_anchors(rng, ts, ys, xs, a.batch)
        # E-076 §7.5's masked-profile objective, per anchor: with probability
        # --profile-drop-p the NEAREST profile is taken out of the input and
        # the decoder is asked for it instead. A per-anchor draw rather than a
        # per-batch one, so one batch carries both regimes and the gradient is
        # not a function of which side of a coin the whole step landed on.
        drop = (rng.random(len(anchors)) < a.profile_drop_p
                if (sampler.profile is not None and sampler.profile_k > 0
                    and a.profile_drop_p > 0.0)
                else False)
        b = to_torch(sampler.sample(anchors, withhold_nearest=drop),
                     chan_depth, device, extras)
        out = model(b, plan)
        opt.zero_grad(set_to_none=True)
        out["loss"].backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
        if not np.isfinite(float(out["loss"].detach())):
            raise SystemExit(f"[{tag}] non-finite loss at step {s} — stopping "
                             f"rather than writing NaN (ml/CLAUDE.md §5.22)")
        if metrics_path and (s % loss_every == 0 or s == a.steps):
            with open(metrics_path, "a") as f:
                f.write(json.dumps({
                    "step": s,
                    "loss_rec": round(out["terms"]["nll"], 5),
                    "loss_nei": round(out["terms"]["mse"], 5)}) + "\n")
        if s % a.eval_every == 0 or s == a.steps:
            nll, mse, n, fam = eval_loss(model, sampler, eval_anchors, plan,
                                         chan_depth, device, a.batch,
                                         extras=extras)
            curve.append({"step": s, "held_out_nll": nll, "held_out_mse": mse,
                          "train_nll": out["terms"]["nll"], "families": fam})
            print(f"[{tag}] step {s:>6}/{a.steps} · train nll "
                  f"{out['terms']['nll']:+.4f} mse {out['terms']['mse']:.4f} "
                  f"· held-out nll {nll:+.4f} mse {mse:.4f} "
                  f"({time.time() - t0:.0f}s) · {fam_line(fam)}", flush=True)
            if metrics_path:
                with open(metrics_path, "a") as f:
                    rec = {"step": s, "held_out_nll": round(nll, 5),
                           "held_out_mse": round(mse, 5),
                           "held_out_targets": int(n)}
                    rec.update(fam_record(fam))
                    rec["wall_s"] = round(time.time() - t0, 1)
                    f.write(json.dumps(rec) + "\n")
            run_profile_eval(a, tag, model, sampler, prof_anchors, D,
                             chan_depth, device, extras, s, out_dir,
                             prof_name, metrics_path, final=(s == a.steps))
        if s % a.save_every == 0:
            save(s)
    save(a.steps)
    return dict(model=model, sampler=sampler, curve=curve, params=params,
                certificate={"anchors": int(len(cert)), "violations": int(bad)},
                eval_anchors=eval_anchors, chan_depth=chan_depth, plan=plan,
                extras=extras, n_extra=n_extra,
                ckpt=os.path.join(out_dir, ckpt_name))


# ------------------------------------------------------------ velocity probe --
def fold_labels(anchors, months):
    """Year-blocked fold labels, mirroring probe_kfold.py's blocking.

    probe_kfold folds by CALENDAR YEAR so autocorrelation cannot leak across
    the fit/test line. A synthetic smoke tensor spans under three years, where
    that would leave two folds; below three years the fallback is five
    CONTIGUOUS-TIME blocks, which preserves the property that matters (a test
    block is a solid stretch of time, never interleaved samples).
    """
    years = np.array([int(months[t][:4]) for t in anchors[:, 0]])
    if len(np.unique(years)) >= 3:
        return years, "calendar-year"
    t = anchors[:, 0].astype(float)
    lo, hi = t.min(), t.max() + 1e-6
    return np.floor(5.0 * (t - lo) / (hi - lo)).astype(int), "5 contiguous-time"


def kfold_r2(F, y, groups):
    """Out-of-fold R^2 and r from the year-blocked ridge.

    Uses `probe_kfold.kfold_r` itself where it imports — same folds, same
    inner-tail lambda selection, so the number is comparable with every other
    probe in the programme — and its returned out-of-fold predictions give
    R^2 = 1 - SSres/SStot. The fallback repeats that arithmetic only if the
    import fails (it pulls in torch, model and temporal).
    """
    F = np.asarray(F, float)
    y = np.asarray(y, float)
    try:
        from probe_kfold import kfold_r
        r, lo, hi, n, rmse, sigma, pred = kfold_r(F, y, groups, boot=200)
        src = "probe_kfold.kfold_r"
    except Exception as e:                       # pragma: no cover - fallback
        print(f"  (probe_kfold unavailable: {e} — local ridge)", flush=True)
        pred = np.full(len(y), np.nan)
        for g in np.unique(groups):
            te = groups == g
            tr = ~te
            mu, sd = F[tr].mean(0), F[tr].std(0) + 1e-9
            Fz = (F - mu) / sd
            idx = np.where(tr)[0]
            A = np.c_[Fz[idx], np.ones(len(idx))]
            reg = 1.0 * np.eye(A.shape[1])
            reg[-1, -1] = 0
            w = np.linalg.solve(A.T @ A + reg, A.T @ y[idx])
            pred[te] = np.c_[Fz[te], np.ones(int(te.sum()))] @ w
        ok = np.isfinite(pred)
        r = float(np.corrcoef(pred[ok], y[ok])[0, 1])
        src = "local ridge"
    ok = np.isfinite(pred)
    ss_res = float(np.mean((pred[ok] - y[ok]) ** 2))
    ss_tot = float(np.var(y[ok]))
    return {"r2": float(1.0 - ss_res / max(ss_tot, 1e-12)), "r": float(r),
            "n": int(ok.sum()), "probe": src}


def encode_anchors(model, sampler, chan, anchors, chan_depth, device,
                   batch=64, hide_cur=True, extras=()):
    """`(Z, TG, OB)` over `anchors`: the codes, the anchor's own channel values
    and their observed flags.

    `hide_cur=True` is the H1 protocol — the `cur_*` channels are HIDDEN, i.e.
    `mask_tok`, the same token channel drop uses during training, so the
    probe's input distribution is one the codec has seen. `hide_cur=False`
    applies NO channel mask at all and asks the strictly easier question: can
    a 32-number code carry a value it was shown? The two numbers bracket the
    result — a `visible` R² that is also low says the bottleneck (or the
    encoder) loses the current whether or not it is hidden, which is a
    different fault from "the cone does not carry motion".
    """
    cur = torch.as_tensor([n.startswith("cur_") for n in chan], device=device)
    zs, tg, ob = [], [], []
    model.eval()
    with torch.no_grad():
        for i in range(0, len(anchors), batch):
            s = sampler.sample(anchors[i:i + batch], train_pool=False)
            b = to_torch(s, chan_depth, device, extras)
            B = b["patch_vals"].shape[0]
            bb = dict(b)
            if hide_cur:
                cm = cur[None].expand(B, -1)
                bb["chan_mask"] = cm
                bb["dot_mask"] = (cm.gather(1, b["chan"].long())
                                  if b["chan"].shape[1]
                                  else torch.zeros_like(b["obs"]))
            z, _ = model.encode(bb)
            zs.append(z.cpu().numpy())
            tg.append(b["patch_vals"][..., 4].cpu().numpy())
            ob.append(b["patch_obs"][..., 4].cpu().numpy())
    model.train()
    return (np.concatenate(zs), np.concatenate(tg), np.concatenate(ob))


def ridge_to_currents(F, TG, OB, chan, groups):
    """`{cur_u: {...}, cur_v: {...}}` — the same ridge, whatever the features.

    Factored out so the three bars (`hidden` z, `visible` z, and the raw
    lag-0 patch) are scored by ONE function with one fold rule: a comparison
    between probes that differ in their scorer is not a comparison.
    """
    out = {}
    for name in ("cur_u", "cur_v"):
        if name not in chan:
            continue
        c = chan.index(name)
        m = OB[:, c] & np.isfinite(TG[:, c])
        if m.sum() < 32:
            # `null`, NOT NaN. This branch wrote `float("nan")`, which
            # json.dump emits as the bare token NaN — a file no strict JSON
            # reader can parse at all, and the exact "loud enough to notice
            # and quiet enough to misattribute" failure ml/CLAUDE.md §5.22 is
            # about. It has never fired on a real run (it needs fewer than 32
            # observed anchors), which is why it survived; `None` says the
            # same thing in a form the file can carry.
            out[name] = {"r2": None, "r": None, "n": int(m.sum()),
                         "note": "too few observed targets"}
            continue
        out[name] = kfold_r2(F[m], TG[m, c], groups[m])
    return out


def r2_str(d):
    """`+0.1234`, or `n/a` where the probe declined to score (r2 is None)."""
    v = (d or {}).get("r2")
    return "   n/a" if v is None else f"{float(v):+.4f}"


def z_stats(Z, seed=0, n_pairs=4096):
    """The COLLAPSE diagnostic — hypothesis (d), read off the probe's own codes.

    Three numbers, each answering a different way a 32-dimensional code can be
    empty, and none of them a NaN even for a code that is exactly constant
    (ml/CLAUDE.md §5.22):

      `var_per_dim`   the variance of each of the d_z coordinates over the
                      probe anchors. A dimension at ~0 is a dimension the
                      codec is not using at all.
      `eff_rank`      the participation ratio (sum L)^2 / sum L^2 of the
                      covariance eigenvalues — how many directions the code
                      actually spends its variance on. It runs from 1 (every
                      anchor on one line) to d_z (an isotropic code), and it
                      is the quantity that distinguishes "32 numbers" from
                      "one number written 32 ways".
      `mean_pair_cos` the mean cosine between the codes of randomly paired
                      anchors, on CENTRED codes. Near 1 means every anchor
                      points the same way — a collapsed embedding that a
                      per-dimension variance can still miss, because a large
                      common offset has variance in no coordinate.
    """
    Z = np.asarray(Z, float)
    n, d = Z.shape
    var = Z.var(axis=0)
    Zc = Z - Z.mean(axis=0, keepdims=True)
    cov = (Zc.T @ Zc) / max(n - 1, 1)
    ev = np.clip(np.linalg.eigvalsh(cov), 0.0, None)
    s1, s2 = float(ev.sum()), float((ev ** 2).sum())
    eff = (s1 * s1 / s2) if s2 > 0.0 else 0.0
    rng = np.random.default_rng(seed)
    if n >= 2:
        i = rng.integers(0, n, n_pairs)
        j = (i + 1 + rng.integers(0, n - 1, n_pairs)) % n     # never i == j
        a, b = Zc[i], Zc[j]
        den = np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1)
        cos = np.where(den > 0, (a * b).sum(1) / np.maximum(den, 1e-30), 0.0)
        mpc = float(np.mean(cos))
    else:
        mpc = 0.0
    return {"d_z": int(d), "n_anchors": int(n),
            "var_per_dim": [round(float(v), 6) for v in var],
            "var_total": float(var.sum()),
            "var_min": float(var.min()) if d else 0.0,
            "var_max": float(var.max()) if d else 0.0,
            "eff_rank": float(eff),
            "eff_rank_frac": float(eff / d) if d else 0.0,
            "mean_pair_cos": mpc, "pairs": int(n_pairs)}


def raw_patch_probe(sampler, chan, anchors, months, batch=64):
    """THE BAR: ridge from the raw lag-0 3x3 patch to cur_u/cur_v, no codec.

    It takes NO model, no device and no channel-depth table, and the signature
    says so: this bar must not be able to depend on a codec even by accident.

    Features are every NON-`cur_` channel's nine patch cells and their nine
    observed flags — at the r3 tensor 39 channels x 18 = 702 numbers — laid
    out exactly the way `ConeMAE.tokens` reads them (`value * observed`, then
    the flag), so "0.0 because unobserved" and "0.0 because that is the
    anomaly" stay distinguishable. The `cur_*` channels are excluded because
    they are the target.

    This is what the H1 comparison was missing. A codec probe's R² answers
    "can a ridge read the current out of z"; it does not say whether reading
    the current is HARD. Geostrophy makes the surface current a gradient of
    sea-surface height, so a linear map from the SSH patch alone recovers much
    of it with no learning at all — and a codec that scores below this bar has
    lost information that was sitting in its own input. Computed ONCE per run,
    on the same anchors and the same folds as both arms, because a bar
    measured on other anchors is not a bar.
    """
    keep = [i for i, n in enumerate(chan) if not n.startswith("cur_")]
    fs, tg, ob = [], [], []
    for i in range(0, len(anchors), batch):
        s = sampler.sample(anchors[i:i + batch], train_pool=False)
        pv = np.asarray(s["patch_vals"], np.float64)          # [B, C, 9]
        po = np.asarray(s["patch_obs"], np.float64)
        f = np.concatenate([(pv * po)[:, keep, :], po[:, keep, :]], axis=2)
        fs.append(f.reshape(len(f), -1))
        tg.append(np.asarray(s["patch_vals"], np.float64))
        ob.append(np.asarray(s["patch_obs"], bool))
    F = np.concatenate(fs)
    TG = np.concatenate(tg)[..., 4]
    OB = np.concatenate(ob)[..., 4]
    groups, how = fold_labels(anchors, months)
    out = {"folds": how, "n_anchors": int(len(anchors)),
           "n_features": int(F.shape[1]),
           "channels": [chan[i] for i in keep],
           "note": "ridge from the raw lag-0 3x3 of every non-cur channel "
                   "(values and observed flags) — no codec, no training"}
    out.update(ridge_to_currents(F, TG, OB, chan, groups))
    return out


def velocity_probe(model, sampler, chan, anchors, months, chan_depth, device,
                   batch=64, extras=()):
    """H1: ridge from z to (cur_u, cur_v), in TWO variants.

    `hidden` is the protocol H1 is stated in — the `cur_*` channels are
    dropped from the encoder's input, so a code that scores has reconstructed
    the current from the motion of everything else. `visible` removes the mask
    entirely and asks whether z can carry a current it was actually shown.
    The target of both is the anchor's own value in anomaly space (the patch
    centre), scored only where it was observed.

    FOR CONTINUITY the `hidden` variant's results are ALSO written at the top
    level as `cur_u` / `cur_v`, byte for byte what this function has always
    returned there: #537's numbers and every reader of them keep meaning what
    they meant. `variants` is where a new reader looks.
    """
    groups, how = fold_labels(anchors, months)
    out = {"folds": how, "n_anchors": int(len(anchors)), "variants": {}}
    for vname, hide in (("hidden", True), ("visible", False)):
        Z, TG, OB = encode_anchors(model, sampler, chan, anchors, chan_depth,
                                   device, batch=batch, hide_cur=hide,
                                   extras=extras)
        res = ridge_to_currents(Z, TG, OB, chan, groups)
        out["variants"][vname] = res
        if vname == "hidden":
            out["d_z"] = int(Z.shape[1])
            out.update(res)                     # the historical top-level keys
            out["z_stats"] = z_stats(Z)
    return out


def probe_line(arm):
    """One line per arm: the visible bar beside the hidden one, and the three
    collapse numbers — so the log says which of H1's stories it supports
    without anyone opening the JSON."""
    vis = arm.get("variants", {}).get("visible", {})
    zs = arm.get("z_stats", {})
    return (f"visible cur_u R2 {r2_str(vis.get('cur_u'))}"
            f" · cur_v R2 {r2_str(vis.get('cur_v'))}"
            f" · z eff-rank {zs.get('eff_rank', 0.0):.2f}/{zs.get('d_z', 0)}"
            f" · var {zs.get('var_total', 0.0):.3g}"
            f" · mean pair cos {zs.get('mean_pair_cos', 0.0):+.3f}")


# -------------------------------------------------------------------- main --
def main(argv=None):
    a = parse(argv)
    os.makedirs(a.out, exist_ok=True)
    if a.smoke_family8 and not a.tensor:
        # The family-8 smoke writes BOTH halves — a three-group tensor and an
        # observation store aligned to it — because the arm is the pair.
        a.tensor, store = smoke_family8(a.out, seed=a.seed)
        a.argo_store = a.argo_store or store
        print(f"--smoke-family8: synthetic tensor at {a.tensor}, store at "
              f"{a.argo_store}", flush=True)
    if a.smoke and not a.tensor:
        a.tensor = smoke_tensor(os.path.join(a.out, "smoke_tensor.npz"),
                                seed=a.seed)
        print(f"--smoke: synthetic tensor at {a.tensor}", flush=True)
    if not a.tensor:
        raise SystemExit("--tensor is required (or --smoke)")
    if a.argo_store:
        # BEFORE load_data, which is the 40-minute anomaly transform: a
        # precondition that depends only on the inputs is checked while the
        # inputs are all it has cost you (ml/CLAUDE.md §0.3, §5.16). Cached,
        # so the arms below re-open it for free.
        open_store(a)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    D = load_data(a)

    res = train_one(a, D, a.L_in, a.out, a.metrics, "cone_codec.pt", "cone",
                    device)
    print(f"[cone] checkpoint {res['ckpt']}", flush=True)

    if a.velocity_probe:
        # The PROBE ANCHORS ARE SHARED between the arms, drawn once here: two
        # probes on two anchor sets differ by the anchors as well as by the
        # codec, and H1 is a statement about the codecs.
        rng = np.random.default_rng(a.seed + 7717)
        ys, xs = np.nonzero(D["ocean"])
        fut = tuple(int(v) for v in a.future_lags.split(",") if v.strip())
        ts = np.arange(D["T"])
        ts = ts[(ts - max(a.L_in, 1) >= 0) & (ts + max(fut) < D["T"])]
        pa = draw_anchors(rng, ts, ys, xs, a.probe_anchors)
        # SORTED BY BIN, because probe_kfold.kfold_r picks its ridge lambda on
        # the LAST 20% of the training rows — an inner TAIL, which is a time
        # tail only if the rows are in time order. Handed a time-shuffled
        # anchor set it validates on an interleaved sample, sees no
        # autocorrelation, picks a lambda far too small and the outer fold
        # blows up (measured: out-of-fold r = -0.38 before this line).
        pa = pa[np.argsort(pa[:, 0], kind="stable")]
        snap = None
        probe = {"cone": velocity_probe(res["model"], res["sampler"],
                                        D["chan"], pa, D["months"],
                                        res["chan_depth"], device,
                                        extras=res["extras"])}
        print(f"[probe] cone   cur_u R2 {r2_str(probe['cone']['cur_u'])} · "
              f"cur_v R2 {r2_str(probe['cone']['cur_v'])}", flush=True)
        print(f"[probe] cone   {probe_line(probe['cone'])}", flush=True)
        if a.snapshot_ablation:
            snap = train_one(a, D, 0, a.out, "metrics_snapshot.jsonl",
                             "snapshot_codec.pt", "snapshot", device)
            probe["snapshot"] = velocity_probe(snap["model"], snap["sampler"],
                                               D["chan"], pa, D["months"],
                                               snap["chan_depth"], device,
                                               extras=snap["extras"])
            print(f"[probe] snapshot cur_u R2 "
                  f"{r2_str(probe['snapshot']['cur_u'])} · cur_v R2 "
                  f"{r2_str(probe['snapshot']['cur_v'])}", flush=True)
            print(f"[probe] snapshot {probe_line(probe['snapshot'])}",
                  flush=True)
            for c in ("cur_u", "cur_v"):
                # `None` where either arm declined to score (fewer than 32
                # observed targets): a difference of a missing number is a
                # missing number, not a NaN in the results file (§5.22).
                r1 = probe["cone"][c]["r2"]
                r0 = probe["snapshot"][c]["r2"]
                probe[f"delta_{c}"] = (None if r1 is None or r0 is None
                                       else r1 - r0)
        # THE BAR, once per run and not per arm: it does not depend on a
        # codec, so computing it twice would be two names for one number and
        # an invitation to quote the wrong one. The cone arm's sampler is used
        # because the lag-0 patch is identical under either L_in.
        probe["raw_patch"] = raw_patch_probe(res["sampler"], D["chan"], pa,
                                             D["months"])
        print(f"[probe] raw 3x3 ({probe['raw_patch']['n_features']} features, "
              f"no codec) cur_u R2 {r2_str(probe['raw_patch']['cur_u'])} · "
              f"cur_v R2 {r2_str(probe['raw_patch']['cur_v'])}", flush=True)
        # THE RECONSTRUCTION SIDE OF THE SAME QUESTION, one line per arm. The
        # probe above says what a ridge can READ OUT of z; this says what the
        # codec's own decoder was asked for and how well it answered, per
        # channel, from the FINAL eval record. Printed here rather than at the
        # eval because that is where the reader is already comparing the arms.
        for tag, arm in (("cone", res), ("snapshot", snap)):
            if arm is None:
                continue
            fam = (arm["curve"][-1] or {}).get("families") or {}
            print(f"[probe] {tag:<8} anchor-family held-out MSE by channel "
                  f"(step {arm['curve'][-1]['step']}): "
                  f"{chan_mse_line(fam, D['chan'])}", flush=True)
        probe["L_in"] = int(a.L_in)
        probe["steps"] = int(a.steps)
        probe["seed"] = int(a.seed)
        path = os.path.join(a.out, "velocity_probe.json")
        with open(path, "w") as f:
            json.dump(probe, f, indent=2)
        print(f"[probe] wrote {path}", flush=True)

    print("\ncurve (step · train nll · held-out nll):", flush=True)
    for c in res["curve"]:
        tn = "     -" if c["train_nll"] is None else f"{c['train_nll']:+.4f}"
        print(f"  {c['step']:>6}  {tn}  {c['held_out_nll']:+.4f}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

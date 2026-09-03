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


# --------------------------------------------------------------------- CLI --
def parse(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--tensor", default="",
                   help="npz with X, months, lats, lons, chan (family4 r3); "
                        "OBS is derived as isfinite(X) exactly as train.py's "
                        "LazyPixels does. Required unless --smoke.")
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
    p.add_argument("--out", default=os.path.join(HERE, "runs", "cone"))
    p.add_argument("--metrics", default="metrics.jsonl")
    p.add_argument("--smoke", action="store_true")
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


def load_data(a):
    """Load the tensor, take the anomaly transform, and return everything the
    sampler and the pools need. Mirrors ml/train.py's preamble."""
    from tensor_io import load_tensor, writable_copy
    d = load_tensor(a.tensor, allow_pickle=False)
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
                t_hold=t_hold, ocean=ocean, norm=norm, T=T, H=H, W=W, C=C)


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


def to_torch(s, chan_depth, device):
    """The sampler's numpy batch as the tensors ConeMAE.forward reads."""
    b = {}
    for k in ("vals", "dy_km", "dx_km", "lag_days", "depth", "patch_vals",
              "fut_vals", "ctx"):
        b[k] = torch.as_tensor(np.ascontiguousarray(s[k]),
                               dtype=torch.float32, device=device)
    for k in ("obs", "valid", "patch_obs", "fut_obs"):
        b[k] = torch.as_tensor(np.ascontiguousarray(s[k]), device=device)
    b["chan"] = torch.as_tensor(s["chan"].astype(np.int64), device=device)
    b["chan_depth"] = chan_depth
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


def eval_loss(model, sampler, anchors, plan, chan_depth, device, batch,
              seed=12345):
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
    """
    g = eval_generator(device, seed)
    p = dict(plan)
    p["generator"] = g
    model.eval()
    nll = mse = w = tgt = 0.0
    fam = {k: {"nll": 0.0, "mse": 0.0, "wsum": 0.0, "n_targets": 0.0}
           for k in QUERY_FAMILIES}
    with torch.no_grad():
        for i in range(0, len(anchors), batch):
            s = sampler.sample(anchors[i:i + batch])
            b = to_torch(s, chan_depth, device)
            out = model(b, p)
            n = out["terms"]["wsum"]
            nll += out["terms"]["nll"] * n
            mse += out["terms"]["mse"] * n
            w += n
            tgt += out["terms"]["n_targets"]
            for k, f in out.get("families", {}).items():
                if k not in fam:
                    continue
                fam[k]["nll"] += f["nll"] * f["wsum"]
                fam[k]["mse"] += f["mse"] * f["wsum"]
                fam[k]["wsum"] += f["wsum"]
                fam[k]["n_targets"] += f["n_targets"]
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
    return nll / w, mse / w, tgt, fam


def fam_record(fam):
    """The per-family eval keys, flat, for one metrics.jsonl record.

    Additive by construction: the headline `held_out_nll` / `held_out_mse` /
    `held_out_targets` keys are written by the caller and are unchanged, so
    status.html and every archived reader keep working; these sit beside them
    (ml/CLAUDE.md §0d — a reader ignores what it does not know). Every value is
    a finite number, including for a family with no targets (§5.22).
    """
    rec = {}
    for k in QUERY_FAMILIES:
        f = fam.get(k) or {"nll": 0.0, "mse": 0.0, "wsum": 0.0,
                           "n_targets": 0.0}
        rec[f"held_out_nll_{k}"] = round(float(f["nll"]), 5)
        rec[f"held_out_mse_{k}"] = round(float(f["mse"]), 5)
        rec[f"held_out_targets_{k}"] = int(f["n_targets"])
        # The WEIGHT is what the two means above divide by, and it is what
        # lets a reader put the headline back together (the family weights of
        # cone_codec.FAMILY_W are inside it, so it is not the target count).
        rec[f"held_out_wsum_{k}"] = round(float(f["wsum"]), 4)
    return rec


def fam_line(fam):
    """One human line: `anchor +1.23 (n) · future … · dots …`."""
    return " · ".join(
        f"{k} {fam[k]['nll']:+.3f}/{int(fam[k]['n_targets']):,}"
        for k in QUERY_FAMILIES if k in fam)


def train_one(a, D, L_in, out_dir, metrics_name, ckpt_name, tag, device,
              eval_anchors=None):
    """Train one arm (the cone codec, or its L_in=0 snapshot twin).

    Returns dict(model, sampler, curve, certificate, params).
    """
    torch.manual_seed(a.seed)
    rng = np.random.default_rng(a.seed)
    chan, C = D["chan"], D["C"]
    fut = tuple(int(v) for v in a.future_lags.split(",") if v.strip())
    sampler = ConeSampler(D["X"], D["OBS"], D["lats"], D["lons"], chan,
                          L_in=L_in, future_lags=fut)
    train_bins = ~D["t_hold"]
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
                    n_fourier=a.n_fourier).to(device)
    params = model.param_count()
    print(f"[{tag}] ConeMAE {params:,} params "
          f"({params / 1e6:.3f}M)", flush=True)

    chan_depth = torch.as_tensor([channel_depth_dbar(n) for n in chan],
                                 dtype=torch.float32, device=device)
    plan = default_plan(chan, n_dot_queries=a.n_dot_queries,
                        aux_latent_w=a.aux_latent_w, future_lags=fut,
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
                "holdout_scope": a.holdout_scope,
                "holdout_years": a.holdout_years,
                "lr": a.lr, "seed": a.seed,
            }}) + "\n")

    def save(step):
        blob = {"args": vars(a), "model": model.state_dict(),
                "chan_names": chan, "norm": D["norm"], "step": int(step),
                "arm": tag, "L_in": int(L_in), "params": params}
        torch.save(blob, os.path.join(out_dir, ckpt_name))

    loss_every = max(1, a.steps // 200)
    curve = []
    t0 = time.time()
    nll0, mse0, n0, fam0 = eval_loss(model, sampler, eval_anchors, plan,
                                     chan_depth, device, a.batch)
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

    for s in range(1, a.steps + 1):
        anchors = draw_anchors(rng, ts, ys, xs, a.batch)
        b = to_torch(sampler.sample(anchors), chan_depth, device)
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
                                         chan_depth, device, a.batch)
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
        if s % a.save_every == 0:
            save(s)
    save(a.steps)
    return dict(model=model, sampler=sampler, curve=curve, params=params,
                certificate={"anchors": int(len(cert)), "violations": int(bad)},
                eval_anchors=eval_anchors, chan_depth=chan_depth, plan=plan,
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
                   batch=64, hide_cur=True):
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
            s = sampler.sample(anchors[i:i + batch])
            b = to_torch(s, chan_depth, device)
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
        s = sampler.sample(anchors[i:i + batch])
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
                   batch=64):
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
                                   device, batch=batch, hide_cur=hide)
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
    if a.smoke and not a.tensor:
        a.tensor = smoke_tensor(os.path.join(a.out, "smoke_tensor.npz"),
                                seed=a.seed)
        print(f"--smoke: synthetic tensor at {a.tensor}", flush=True)
    if not a.tensor:
        raise SystemExit("--tensor is required (or --smoke)")
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
        probe = {"cone": velocity_probe(res["model"], res["sampler"],
                                        D["chan"], pa, D["months"],
                                        res["chan_depth"], device)}
        print(f"[probe] cone   cur_u R2 {r2_str(probe['cone']['cur_u'])} · "
              f"cur_v R2 {r2_str(probe['cone']['cur_v'])}", flush=True)
        print(f"[probe] cone   {probe_line(probe['cone'])}", flush=True)
        if a.snapshot_ablation:
            snap = train_one(a, D, 0, a.out, "metrics_snapshot.jsonl",
                             "snapshot_codec.pt", "snapshot", device)
            probe["snapshot"] = velocity_probe(snap["model"], snap["sampler"],
                                               D["chan"], pa, D["months"],
                                               snap["chan_depth"], device)
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

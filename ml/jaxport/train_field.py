#!/usr/bin/env python3
"""E-052 · the FIELD-HEAD trainer in JAX/optax — the TPU twin of
`ml/train_field.py`.

`ml/field_model.py` is the torch model, `ml/jaxport/field_model.py` is its
NNX mirror (parity-gated by `tests/test_jaxport_field.py`), and this file
trains that mirror on the real `[T, P, d_z]` embed-cache artefact stage 2
already publishes.

**WHY IT EXISTS.** The field head attends over SPACE: one forward is
full attention over ~5.3k ocean-patch tokens, and the conditioner runs a
K-step causal transformer PER TOKEN. At K=144 that is ~770k length-144
sequences per batch element — the shape that made the stage-2 port worth
building (`ml/jaxport/train_stage2.py`'s docstring), one axis further along.
This trainer is what lets that run on a v5e instead of queueing behind rented
GPUs.

**A NUMBER THIS FILE PRODUCES IS A NEW TIER.** `ml/CLAUDE.md` §3b: a
JAX-trained result is never pooled with the torch/GPU record, and the first
result at a tier buys its own replication. The config record says
`"backend": "jax"` so that fact cannot go missing from the one line a reader
sees.

WHAT IS MIRRORED, TERM FOR TERM, from `ml/train_field.py`

  · the OBJECTIVE — det is `((z_t + r_hat) - z_{t+1})**2 .mean()`; diff is
    `edm_loss`, same lognormal ladder re-centred on the measured `sigma_data`;
  · `sigma_data` measured ONCE, before the first step, on the TRAIN split
    only, from the data and never from the model (§4.2);
  · the det read-out `{mse, mse_pers, ratio}` and the diffusion read-out
    `{sample_ratio, ens_ratio, mse_pers, spread, spread_error,
    sign_coherence, mode_corr, crps}` — every formula transcribed from
    `eval_det`/`eval_diff`, and the probabilistic ones computed on the HOST in
    numpy against the same `ml/probscore.crps_ensemble`;
  · result-file discipline (§5.25): the JSON is rewritten ATOMICALLY at every
    eval carrying a top-level `in_progress`, and that key disappears exactly
    once, at a completed end. §5.22 is the other half — a non-finite loss or
    eval STOPS the run rather than writing NaN.

WHAT IS DIFFERENT, AND WHY — each named here so nothing is silently missing:

  · **THE TOYS ARE SMOKE-ONLY.** `ml/train_field.py`'s generators are written
    in torch and draw from a `torch.Generator`; `--toy gauss` and
    `--toy shift` here are minimal NUMPY re-implementations of the same two
    laws. They are for exercising the code path before spending the expensive
    resource (§4.8) and for F7's persistence gate. They are NOT the same
    sample path, so a toy NUMBER from this trainer and one from the torch
    trainer are two draws, never a comparison. `bimodal` is deliberately
    absent: it is the axis-B science toy and belongs with the torch arm that
    owns that question.
  · **THE SPLIT RULE IS STAGE-2's, NOT `make_splits`'.** A window is VAL when
    its TARGET bin t+1 falls in a holdout year, and TRAIN otherwise. Windows
    may LOOK at held-out bins — persistence can too — they may never be SCORED
    on them, which is exactly `train_stage2.py`'s `ok_t`. `ml/train_field.py`
    is stricter (a window straddling the boundary is dropped from both pools);
    that rule costs K-1 windows at every boundary and is not what the fleet's
    stage-2 pool does.
  · **THE RNG STREAM.** Window draws come from a seeded
    `np.random.default_rng`; the EDM noise and the sampler come from a JAX
    key folded in on the step number. Neither stream is torch's, so a
    bit-identical TRAJECTORY across frameworks is impossible by construction
    and is not claimed. What IS claimed and gated is that on the same weights
    and the same injected (sigma, noise) the two frameworks compute the same
    loss and the same gradient (`tests/test_jaxport_field.py` F3/F5), and that
    a RESUME of this trainer is bit-identical to an uninterrupted run (F7).
  · **NO `sample()` PARITY.** `FieldHeadJax.sample` draws from a JAX key. The
    cross-framework surface is `sample_from` with an injected `x_init`, which
    is what F4 gates.

THE TPU DISCIPLINE, and the two measurements behind it (the E-051 launch,
`ml/handoffs/2026-08-26-e051-session.md`, and commit c4ce2da):

  · **NEVER `jnp.asarray` a large host array.** It commits the WHOLE array to
    device 0 and re-slices it there; at K=144 that was a 10.95 GB staging copy
    on a 16 GB chip and it is how the first E-051 node died. Every transfer
    here goes through `put()`, which is `jax.device_put(numpy,
    NamedSharding)` over a 1-D mesh with the BATCH axis sharded and the
    parameters replicated — host-side slicing, no full-size device allocation
    ever exists.
  · **THE HOST GATHER IS THE BOTTLENECK, so it overlaps.** Rows come out of
    the memmapped Z in a background producer thread with a queue of depth
    `--prefetch` (2 by default — the double buffer), and are cast to the
    compute dtype ON THE HOST before transfer. The batch SEQUENCE is
    unchanged: one RNG, drawn in step order, in the one function that touches
    it.
  · **THE CONDITIONER RUNS IN TOKEN CHUNKS.** Its input is
    [B, K, ntok, feat_in], which at K=144, ntok~5.3k, feat 528 and B=4 is
    ~6.4 GB in fp32 — and it never has to exist. `--cond-chunk` (1024 tokens
    by default) walks the token axis with `jax.lax.map`, tokenizing each chunk
    straight out of the [B, K, P, d_z] pixel context through the tokenizer's
    inverse index, so the full token tensor is never materialised. The
    conditioner is per-token independent by construction (time only, no
    cross-token mixing — `ml/field_model.py` decision 3), so chunking is exact
    in exact arithmetic; only the reduction order inside XLA can move, and a
    chunked and an unchunked run are compared at 1e-5, not bitwise.
    **The DENOISER is NOT chunked**: it is full attention over all tokens and
    chunking it would change the model.

    # smoke, CPU, ~1 minute
    python3 ml/jaxport/train_field.py --toy gauss --mode diff --smoke

    # the real substrate
    python3 ml/jaxport/train_field.py --z-cache Z.npy --data tensor.npz \\
        --holdout-years 2019,2020 --mode det --K 144 --patch 4 \\
        --steps 200000 --batch 4 --cond-chunk 1024 --ckpt-dir ckpts
"""
import argparse
import json
import math
import os
import queue as _queue
import sys
import threading
import time
from functools import partial

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ML = os.path.dirname(HERE)
if ML not in sys.path:
    sys.path.insert(0, ML)

import jax                                                      # noqa: E402
import jax.numpy as jnp                                         # noqa: E402
import optax                                                    # noqa: E402
from flax import nnx                                            # noqa: E402

from jaxport.field_model import (FieldHeadJax, OceanTokenizerJax,  # noqa: E402
                                 count_params_jax, export_field_pt,
                                 nfe_to_steps)

# ml/probscore.py is E-052.0 and is pure numpy. Guard the import so this
# module still LOADS without it (a trainer that cannot be imported cannot be
# tested), and skip CRPS with a WARNING rather than inventing a number — a
# missing scoreboard must read as missing, never as zero.
try:
    from probscore import crps_ensemble
    HAVE_CRPS = True
except Exception:                             # pragma: no cover
    crps_ensemble = None
    HAVE_CRPS = False

# `ml/tensor_io.py` is used for the SMALL METADATA ARRAYS ONLY (lats, lons,
# months, ys, xs). X is never touched: at family 5 it is 165.6 GB and this
# trainer reads its state out of the published Z instead.
from tensor_io import load_tensor                               # noqa: E402


# ---------------------------------------------------------------------------
# result file (ml/CLAUDE.md §5.25) and the NaN rule (§5.22)
# ---------------------------------------------------------------------------
def write_result(path, config, history, final=None, in_progress=None):
    """Write the run's result file ATOMICALLY, optionally marked partial.

    Byte-for-byte the same discipline as `ml/train_field.py:write_result`:
    `in_progress` is a top-level key written FIRST so a human opening the file
    sees it before any number, and its ABSENCE is the run's only completion
    certificate.
    """
    d = os.path.dirname(os.path.abspath(path)) or "."
    os.makedirs(d, exist_ok=True)
    payload = {}
    if in_progress is not None:
        payload["in_progress"] = in_progress
    payload["config"] = config
    payload["history"] = history
    payload["final"] = final
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(payload, f, indent=1)
    os.replace(tmp, path)
    return path


def _finite_or_die(name, *vals):
    """§5.22: never write NaN into a results file — stop instead."""
    for v in vals:
        if v is None:
            continue
        x = float(v)
        if not math.isfinite(x):
            sys.exit(f"REFUSING to continue: {name} went non-finite ({x}). "
                     f"A results file full of NaN is loud enough to notice and "
                     f"quiet enough to misattribute (ml/CLAUDE.md §5.22).")


# ---------------------------------------------------------------------------
# toy laws — SMOKE ONLY (see the module docstring)
# ---------------------------------------------------------------------------
def _smooth_np(x, passes=2):
    """Separable 3x3 box blur with wraparound, `passes` times — the numpy
    twin of `ml/train_field.py:_smooth`, which is `torch.roll`-based."""
    for _ in range(passes):
        x = (x + np.roll(x, 1, axis=0) + np.roll(x, -1, axis=0)) / 3.0
        x = (x + np.roll(x, 1, axis=1) + np.roll(x, -1, axis=1)) / 3.0
    return x


def _smooth_field_np(H, W, d_z, rng, passes=2):
    x = _smooth_np(rng.standard_normal((H, W, d_z)).astype(np.float32), passes)
    return (x / np.sqrt(np.mean(x ** 2))).astype(np.float32)


def toy_shift(rng, smoke=False):
    """AXIS A's microcosm: a purely SPATIAL law.

        x_{t+1} = (x_t rolled one cell EASTWARD) + 0.02 * smooth noise

    Every pixel is ocean. The next value of a pixel lives in its WESTERN
    neighbour and nowhere in its own history, so a per-pixel head cannot beat
    persistence here by construction while a head that attends over space can.
    """
    H = W = 12 if smoke else 24
    T = 60 if smoke else 400
    d_z = 4
    x = _smooth_field_np(H, W, d_z, rng)
    frames = [x]
    for _ in range(T - 1):
        e = _smooth_np(rng.standard_normal((H, W, d_z)).astype(np.float32), 2)
        e = 0.02 * e / np.sqrt(np.mean(e ** 2))
        frames.append(np.roll(frames[-1], 1, axis=1) + e)
    X = np.stack(frames).astype(np.float32)
    ys, xs = np.meshgrid(np.arange(H), np.arange(W), indexing="ij")
    ys, xs = ys.reshape(-1), xs.reshape(-1)
    Z = X.reshape(T, H * W, d_z)[:, ys * W + xs]
    return dict(Z=np.ascontiguousarray(Z), H=H, W=W, ys=ys, xs=xs, d_z=d_z,
                months=None, law="shift")


def toy_gauss(rng, smoke=False):
    """A KNOWN Gaussian conditional, for closed-form checks.

        x_{t+1} = a * x_t + sigma_e * eps,   a = 0.7, sigma_e = 0.5, iid/pixel

    The conditional mean of the residual is -0.3 * x_t, so persistence is NOT
    optimal — which is what makes it F7's gate: a det head that has learned
    anything must beat persistence here.
    """
    H = W = 8
    T = 60 if smoke else 800
    d_z, a, se = 1, 0.7, 0.5
    x = (rng.standard_normal((H, W, d_z)) * (se / math.sqrt(1 - a * a))
         ).astype(np.float32)
    frames = [x]
    for _ in range(T - 1):
        frames.append((a * frames[-1] + se * rng.standard_normal(
            (H, W, d_z))).astype(np.float32))
    X = np.stack(frames).astype(np.float32)
    ys, xs = np.meshgrid(np.arange(H), np.arange(W), indexing="ij")
    ys, xs = ys.reshape(-1), xs.reshape(-1)
    Z = X.reshape(T, H * W, d_z)[:, ys * W + xs]
    return dict(Z=np.ascontiguousarray(Z), H=H, W=W, ys=ys, xs=xs, d_z=d_z,
                months=None, law="gauss", a=a, sigma_e=se)


TOYS = {"gauss": toy_gauss, "shift": toy_shift}


# ---------------------------------------------------------------------------
# the real substrate
# ---------------------------------------------------------------------------
def _year_of(label):
    """The year prefix of a 'YYYY-MM' or 'YYYY-MM-DD' bin label."""
    return int(str(label)[:4])


def _season_of_month(label):
    """(sin, cos) of the year phase at a month's CENTRE — the same feature
    `ml/train_field.py:_season_of_month` builds, tolerant of a day field."""
    mm = int(str(label).split("-")[1])
    frac = (mm - 0.5) / 12.0
    return math.sin(2 * math.pi * frac), math.cos(2 * math.pi * frac)


def load_real(z_cache, npz_path, pixels_npy=None):
    """The `[T, P, d_z]` embed cache (memmapped) plus the tensor's GEOMETRY.

    `load_tensor` is used for the SMALL METADATA ARRAYS ONLY — lats, lons,
    months, ys, xs. **X is never read.** At family 5 it is 165.6 GB, and the
    whole point of the field head reading a published Z is that the raw tensor
    stays on the shelf; touching `d["X"]` here would put the format problem
    `ml/tensor_io.py` exists to solve back into the hot path.

    `ys`/`xs` must be SUPPLIED — from the npz when it carries them, else from
    `--pixels`. Deriving them from P alone is not possible, and guessing a
    raster order is exactly the class of mistake that puts the Gulf Stream in
    the Norwegian Sea.
    """
    Z = np.load(z_cache, mmap_mode="r")
    if Z.ndim != 3:
        sys.exit(f"--z-cache {z_cache}: expected [T, P, d_z], got {Z.shape}")
    d = load_tensor(npz_path)
    if "lats" not in d or "lons" not in d:
        sys.exit(f"--data {npz_path}: no lats/lons — cannot derive H, W")
    H, W = int(len(d["lats"])), int(len(d["lons"]))
    if "ys" in d and "xs" in d:
        ys = np.asarray(d["ys"]).astype(np.int64).reshape(-1)
        xs = np.asarray(d["xs"]).astype(np.int64).reshape(-1)
    elif pixels_npy:
        px = np.load(pixels_npy)
        px = px if px.shape[0] == 2 else px.T
        ys, xs = px[0].astype(np.int64), px[1].astype(np.int64)
    else:
        sys.exit(f"--data {npz_path} carries no ys/xs and no --pixels npy was "
                 f"given. The pixel ordering of a [T,P,d_z] cache cannot be "
                 f"recovered from P; supply it rather than guessing.")
    if len(ys) != Z.shape[1]:
        sys.exit(f"pixel list has {len(ys)} entries, Z has P={Z.shape[1]}")
    months = ([str(m) for m in np.asarray(d["months"]).reshape(-1)]
              if "months" in d else None)
    if months is not None and len(months) != Z.shape[0]:
        sys.exit(f"months has {len(months)} entries, Z has T={Z.shape[0]}")
    return dict(Z=Z, H=H, W=W, ys=ys, xs=xs, d_z=int(Z.shape[2]),
                months=months, law="real")


# ---------------------------------------------------------------------------
# windows, splits, the host gather
# ---------------------------------------------------------------------------
def make_splits(T, K, months=None, holdout_years=(), val_frac=0.15):
    """Valid anchors t (context t-K+1..t, target t+1), split train/val.

    THE RULE IS STAGE-2's (`train_stage2.py`'s `ok_t`): a window is VAL when
    its TARGET bin t+1 falls in a holdout year, TRAIN otherwise. A window may
    LOOK at a held-out bin — persistence can too — it may never be SCORED on
    one. With no holdout years (the toys) it degrades to a TAIL split, so a
    val window never shares a target with a train window except across the
    seam.
    """
    anchors = np.arange(K - 1, T - 1, dtype=np.int64)
    if holdout_years:
        if months is None:
            sys.exit("--holdout-years was given but the data carries no "
                     "`months`: there is nothing to hold out BY. Supply months "
                     "or drop the flag rather than silently falling back to a "
                     "tail split.")
        hold = np.array([_year_of(m) in set(int(y) for y in holdout_years)
                         for m in months])
        tgt = hold[anchors + 1]
        return anchors[~tgt].copy(), anchors[tgt].copy()
    n_val = max(1, int(round(val_frac * len(anchors))))
    return anchors[:-n_val].copy(), anchors[-n_val:].copy()


class Windows:
    """Gathers (context, z_t, z_{t+1}, season) windows out of Z, on the HOST.

    Z is a memmap of the published cache and stays one: the trainer touches
    K+1 rows of it per window and materialising it would be the format problem
    all over again. `rowmap` is `--time-stride`'s indirection — the axis is
    subsampled by an INDEX MAP rather than by slicing the memmap, because
    fancy-indexing a memmap materialises the whole thing.
    """

    def __init__(self, Z, K, season=None, rowmap=None, dtype=np.float32):
        self.Z, self.K, self.season = Z, int(K), season
        self.rowmap = rowmap
        self.dtype = dtype
        # THE CONTEXT SHIPS IN THE SOURCE PRECISION. The published cache is
        # float16, so a float16 ctx carries exactly the same values as the
        # float32 upcast — and at the real geometry the f32 ctx alone is
        # 6.4 GB per chip (batch 2 × K 144 × P 86,698 × d_z 32), which with
        # params+optimizer is what blew the v5e's 16 GB HBM on 2026-08-26
        # ("Attempting to reserve 9.50G ... 8.25G free", both smoke legs).
        # The conditioner upcasts CHUNK BY CHUNK on device (chunked_cond),
        # so no f32 copy of the whole ctx ever exists anywhere. A float32
        # source (the toys) keeps a float32 ctx: shipping narrower than the
        # source would LOSE values, and the parity gates run on that path.
        self.ctx_dtype = (np.float16 if np.dtype(Z.dtype) == np.float16
                          else np.dtype(dtype))

    def _rows(self, idx, dtype=None):
        r = idx if self.rowmap is None else self.rowmap[idx]
        return np.asarray(self.Z[r], self.dtype if dtype is None else dtype)

    def batch(self, ts):
        """-> ctx [B,K,P,d_z] (ctx_dtype), z_t/z_n [B,P,d_z] (dtype), season.

        z_t and z_n stay `self.dtype` (f32): they are small ([B, P, d_z]) and
        feed the loss/residual arithmetic directly. Only ctx — the [B, K, …]
        axis that dominates memory — ships at source precision.
        """
        ts = np.asarray(ts, np.int64)
        K = self.K
        ctx = np.stack([self._rows(np.arange(t - K + 1, t + 1),
                                   self.ctx_dtype) for t in ts])
        z_t = np.asarray(ctx[:, -1], self.dtype)
        z_n = np.stack([self._rows(np.array([t + 1]))[0] for t in ts])
        sea = (np.zeros((len(ts), K, 2), self.dtype) if self.season is None
               else np.stack([self.season[t - K + 1:t + 1] for t in ts]))
        return ctx, z_t, z_n, np.asarray(sea, self.dtype)


def measure_sigma_data(win, train_ts, cap=256):
    """RMS of the one-step residual over TRAIN windows — EDM's sigma_data.

    A property of the DATA, never of the model (`ml/CLAUDE.md` §4.2): it must
    not move as training proceeds, so it is measured once, before the first
    step, on the training split only. Same cap and same evenly-spaced
    subsample as `ml/train_field.py`.
    """
    ts = train_ts if len(train_ts) <= cap else train_ts[
        np.linspace(0, len(train_ts) - 1, cap).astype(np.int64)]
    # ROW PAIRS ONLY — never win.batch(ts). The residual needs z_t and z_n;
    # batch() also builds ctx [B, K, P, d_z], which at the real substrate
    # (cap 256 × K 144 × P 86,698 × d_z 32, f32) is ~409 GB, and on
    # 2026-08-26 the kernel OOM-killed BOTH e052-verify smoke legs at that
    # line on a 189 GB host — rc 137 at ~123 s, no traceback, in det and
    # diff alike. Identical arithmetic, ~1600× smaller: the mean over
    # windows of per-window squared-residual means IS the pooled mean,
    # because every window has the same P × d_z element count.
    acc = 0.0
    for t in np.asarray(ts, np.int64):
        z_t = win._rows(np.array([t]))[0]
        z_n = win._rows(np.array([t + 1]))[0]
        acc += float(np.mean((z_n - z_t) ** 2))
    return float(np.sqrt(acc / len(ts)))


# ---------------------------------------------------------------------------
# the LR schedule — `ml/temporal.py:make_sched`'s expdecay, horizon-free
# ---------------------------------------------------------------------------
def lr_factor(e, a):
    """The LR multiplier at scheduler position `e` (steps already taken).

    Two branches, both HORIZON-FREE — `lr(s)` never depends on `--steps`, so a
    run stopped at 60k and continued to 200k sees exactly the rate an
    uninterrupted run would have seen. That removes the whole
    `CosineAnnealingLR.load_state_dict` class of failure (`ml/CLAUDE.md` §7:
    a reloaded schedule asked for a larger total returns lr = 0.0) and makes
    two budgets a prefix and its extension rather than two experiments.

    `expdecay` is transcribed from `ml/temporal.py:make_sched`'s own closure:
    a cosine-shaped warmup ramp (smooth at BOTH ends, unlike a linear ramp
    which arrives at the peak with a corner), then halving every
    `--lr-halflife` ABSOLUTE steps past the warmup. `constant` is the same
    warmup followed by 1.0 forever, i.e. expdecay at an infinite half-life.
    No terminal cooldown here: `ml/temporal.py`'s taper is a function of
    `--steps` and would put the horizon back into the rate.
    """
    warm = max(1, int(a.lr_warmup))
    s = e + 1
    ramp = 0.5 * (1 - math.cos(math.pi * min(1.0, s / warm)))
    if s <= warm:
        return ramp
    if a.lr_schedule == "expdecay":
        return 0.5 ** ((s - warm) / max(1.0, float(a.lr_halflife)))
    return 1.0


# ---------------------------------------------------------------------------
# checkpoint I/O — the flat-leaf .npz `train_stage2.py` uses
# ---------------------------------------------------------------------------
def _leaves(tree):
    return jax.tree_util.tree_leaves(tree)


def save_state_npz(path, state, opt_state, step, args, rng_state, history):
    """params + optimiser moments + step + args + the HOST RNG + history.

    Identical in shape and reasoning to `train_stage2.save_state_npz`: the
    leaf order is a deterministic function of the tree structure, the
    structure is a deterministic function of the architecture, and the
    architecture is in the file — so a load rebuilds the structure first and
    refuses on any count or shape it did not expect.

    The window sampler's `np.random.default_rng` state rides along because it
    is what makes a resume BIT-IDENTICAL rather than merely similar (F7). The
    EDM/sampler keys need no state: they are `fold_in(root, step)`, a pure
    function of the step number.
    """
    blob = {"_step": np.asarray(int(step)),
            "_args": np.asarray(json.dumps(args)),
            "_rng": np.asarray(json.dumps(rng_state)),
            "_history": np.asarray(json.dumps(history)),
            "_n_state": np.asarray(len(_leaves(state))),
            "_n_opt": np.asarray(len(_leaves(opt_state)))}
    for i, v in enumerate(_leaves(state)):
        blob[f"s{i}"] = np.asarray(v)
    for i, v in enumerate(_leaves(opt_state)):
        blob[f"o{i}"] = np.asarray(v)
    # A FILE HANDLE, not a name: np.savez appends ".npz" to any path that does
    # not already end in it, so savez(path + ".tmp") writes path.tmp.npz and
    # the rename finds nothing.
    tmp = path + ".tmp"
    with open(tmp, "wb") as fh:
        np.savez(fh, **blob)
    os.replace(tmp, path)          # flush, THEN publish (ml/CLAUDE.md §5.21)


def load_state_npz(path, state, opt_state):
    """Read `save_state_npz` back INTO the structures already built, refusing
    on a leaf-count or shape mismatch rather than unflattening whatever fits.
    Returns (state, opt_state, step, args, rng_state, history)."""
    z = np.load(path, allow_pickle=False)
    args = json.loads(str(z["_args"]))
    rng_state = json.loads(str(z["_rng"]))
    history = json.loads(str(z["_history"]))
    sl, ol = _leaves(state), _leaves(opt_state)
    if int(z["_n_state"]) != len(sl) or int(z["_n_opt"]) != len(ol):
        raise SystemExit(
            f"REFUSING to resume {path}: it holds {int(z['_n_state'])} state "
            f"leaves and {int(z['_n_opt'])} optimiser leaves; this head has "
            f"{len(sl)} and {len(ol)}. The architecture or the optimiser "
            f"differs from the one that wrote it.")
    new_s, new_o = [], []
    for i, v in enumerate(sl):
        arr = z[f"s{i}"]
        if arr.shape != tuple(v.shape):
            raise SystemExit(f"REFUSING to resume {path}: state leaf {i} is "
                             f"{arr.shape}, the head wants {tuple(v.shape)}")
        new_s.append(jnp.asarray(arr, v.dtype))
    for i, v in enumerate(ol):
        arr = z[f"o{i}"]
        if arr.shape != tuple(v.shape):
            raise SystemExit(f"REFUSING to resume {path}: optimiser leaf {i} "
                             f"is {arr.shape}, the head wants {tuple(v.shape)}")
        new_o.append(jnp.asarray(arr, v.dtype))
    return (jax.tree_util.tree_unflatten(
                jax.tree_util.tree_structure(state), new_s),
            jax.tree_util.tree_unflatten(
                jax.tree_util.tree_structure(opt_state), new_o),
            int(z["_step"]), args, rng_state, history)


# ---------------------------------------------------------------------------
# the conditioner, in TOKEN CHUNKS
# ---------------------------------------------------------------------------
def cond_from_pixels(model, ctx_px, season, chunk, remat=True):
    """[B, K, P, d_z] -> [B, ntok, d_cond], never materialising the tokens.

    The naive route is `model.make_cond(tok.to_tokens(ctx_px), season)`, and
    at K=144, ntok~5.3k, feat_in 528 and B=4 that intermediate is ~6.4 GB in
    fp32. It does not have to exist: the conditioner is TIME-ONLY and mixes no
    tokens (`ml/field_model.py` decision 3), so token chunk c can be
    tokenized, conditioned and discarded independently of every other chunk.

    `jax.lax.map` over the chunk axis rather than a Python loop: one compile
    and one copy of the conditioner in the graph, whatever `ntok/chunk` is.
    The token axis is padded up to a whole number of chunks with slots whose
    inverse index is -1 — i.e. with tokens made of pure "not ocean" slots —
    and the padding is sliced off at the end; nothing downstream ever sees it.

    EXACT in exact arithmetic, and gated at 1e-5 rather than bitwise: XLA is
    free to pick a different reduction order for a [B*chunk, K, d] batch than
    for a [B*ntok, K, d] one.

    ------------------------------------------------------------------------
    `remat` — WHY CHUNKING ALONE DOES NOT BOUND THE TRAINING FOOTPRINT
    ------------------------------------------------------------------------
    Chunking bounds the FORWARD working set. It does not bound what the
    BACKWARD pass has to keep: `jax.lax.map` lowers to `lax.scan`, and a
    scan differentiated in reverse mode STACKS every iteration's residuals,
    so the retained set is `n_chunks x (per-chunk residuals)` — which is the
    whole unchunked conditioner's activation set, to within the padding.
    **Narrowing `--cond-chunk` does not reduce it at all**; it only splits
    the same total into more, smaller pieces.

    DESIGN ARITHMETIC at the E-052.1 train config — K 144, ntok 5,400,
    d_model 1024, layers 16, d_cond 512, cond_layers 2, cond_heads 4,
    patch 4 / d_z 32 (feat_in 528), batch 8 sharded over 4 chips, so
    B_chip = 2 — in fp32, counting the residuals a pre-LN encoder layer's
    backward actually needs (layer input, norm1 out, q, k, v, attn out,
    post-attn residual, norm2 out, linear1 pre-activation [4x], gelu out
    [4x] = 16 tensors of [S, K, d_cond], plus the softmax probabilities
    [S, H, K, K]), with S = B_chip x chunk. Fusion savings are not credited;
    these are upper bounds, and they are UNMEASURED — no chip this size has
    run this config yet.

        chunk   n_chunks   per-chunk fwd   RETAINED, no remat   RETAINED, remat
        1024        6         20.7 GB          124 GB               25 MB
         512       11         10.4 GB          114 GB               24 MB
         256       22          5.2 GB          114 GB               23 MB

    The retained column is flat because it is the same arithmetic either
    way. With remat it is the per-chunk OUTPUT only — `[B_chip, chunk,
    d_cond]` fp32, 4.19 MB at chunk 1024 — stacked over the chunks: a
    factor of ~5,000, and it is what takes the conditioner off the HBM
    budget entirely.

    What `--cond-chunk` then controls is the TRANSIENT: backward recomputes
    ONE chunk at a time, so the peak is the "per-chunk fwd" column. On a
    16 GB v5e chip, alongside ~3.6 GB of replicated 302M-parameter weights
    and AdamW moments plus the denoiser's own full-token activations, that
    makes **chunk 256 the config that fits and chunk 1024 the one that does
    not** — a knob with a computable answer instead of a guess.

    `jax.checkpoint` is applied to the per-chunk function, with its default
    policy (nothing saved, everything recomputed) and its default
    `prevent_cse=True`. The forward is the same jaxpr either way, so
    `--cond-remat` on and off must produce IDENTICAL losses and identical
    updates; F9 in `tests/test_jaxport_field.py` gates that, and reports
    whether the equality came out float-exact or to a few ulp.

    With no chunking (`chunk` 0 or >= ntok) the whole branch is skipped and
    `remat` is inert — there is no scan to stack residuals in, and wrapping
    the single conditioner call would trade memory the config did not ask
    to trade.
    """
    tok = model.tok
    B, K, P, d_z = ctx_px.shape
    N, P2 = tok.ntok, tok.P2
    if chunk is None or chunk <= 0 or chunk >= N:
        # The unchunked path is the toy/CI path — full upcast is affordable
        # there, and f16 -> f32 is exact so the values are the f32 path's.
        return model.make_cond(tok.to_tokens(ctx_px.astype(jnp.float32)),
                               season)
    nc = -(-N // chunk)
    pad = nc * chunk - N
    # A 2-D index PER CHUNK, [chunk, P2] — the same shape discipline
    # `OceanTokenizerJax.to_tokens_rows` uses, and for the same reason: the
    # only reshape left on the path is a merge of the two TRAILING axes, which
    # is the form JAX's sharding-in-types can infer through when the batch
    # axis is sharded.
    pxf = np.concatenate([tok.px_of_flat,
                          np.full(pad * P2, -1, np.int64)]).reshape(
                              nc, chunk, P2)
    idx = jnp.asarray(pxf)
    zero = jnp.zeros((), jnp.float32)
    one = jnp.ones((), jnp.float32)

    def _one(ix):
        safe = jnp.maximum(ix, 0)
        ok = ix >= 0
        # The upcast lives HERE, on the gathered chunk, so a float16 ctx
        # (real runs — see Windows.ctx_dtype) never exists as float32 in
        # full: only [B, K, chunk, P2, d_z] at a time does. f16 -> f32 is
        # exact, so every number downstream is the f32 path's.
        v = jnp.take(ctx_px, safe, axis=2).astype(jnp.float32)
        v = jnp.where(ok[None, None, :, :, None], v, zero)
        flag = jnp.broadcast_to(
            jnp.where(ok, one, zero)[None, None, :, :, None],
            v.shape[:-1] + (1,))
        toks = jnp.concatenate([v, flag], axis=-1).reshape(
            B, K, chunk, P2 * (d_z + 1))
        return model.cond(toks, season)                   # [B, chunk, d_cond]

    # GRADIENT REMATERIALIZATION on the per-chunk body — see the docstring's
    # arithmetic. Applied INSIDE the map, not around it, because what has to
    # stop being retained is exactly what the scan stacks per iteration.
    step_fn = jax.checkpoint(_one) if remat else _one
    outs = jax.lax.map(step_fn, idx)                      # [nc,B,chunk,d_cond]
    out = jnp.transpose(outs, (1, 0, 2, 3)).reshape(B, nc * chunk, -1)
    return out[:, :N]


# ---------------------------------------------------------------------------
# the diffusion read-out, on the HOST — `ml/train_field.py:eval_diff`'s
# formulas, transcribed
# ---------------------------------------------------------------------------
class DiffAccum:
    """Sufficient statistics for the probabilistic read-out, accumulated over
    batches because at real scale [M, B, P, d_z] does not fit in memory.

    Every formula is `ml/train_field.py:eval_diff`'s, term for term:
      * `sample_ratio`   — per-MEMBER MSE / persistence;
      * `ens_ratio`      — ensemble-MEAN MSE / persistence, the one that must
        match the deterministic head (no MSE tax after averaging);
      * `spread_error`   — sqrt(mean[(M+1)/M · Var_ddof1]) / rmse(ens mean),
        `ml/probscore.py`'s own convention: 1 is calibration, < 1
        over-confidence, > 1 an inflated ensemble;
      * `sign_coherence` — mean over members and times of |mean over ocean
        pixels of sign(r_m)| — THE JOINT-LAW DETECTOR;
      * `mode_corr`      — mean |cosine| between r_m and r_true, UNCENTRED
        deliberately (a field-coherent mode is a component along a FIXED
        direction, and removing the field mean would throw away most of a
        single-signed pattern).
    """

    def __init__(self, members):
        self.M = int(members)
        self.se_s = self.se_e = self.pe = self.n = 0.0
        self.var_sum = self.var_n = 0.0
        self.signs, self.corrs, self.crpss = [], [], []

    def add(self, ens, z_t, z_n):
        ens = np.asarray(ens, np.float64)
        z_t = np.asarray(z_t, np.float64)
        z_n = np.asarray(z_n, np.float64)
        M = self.M
        mean = ens.mean(0)
        self.se_s += float(((ens - z_n[None]) ** 2).sum()) / M
        self.se_e += float(((mean - z_n) ** 2).sum())
        self.pe += float(((z_t - z_n) ** 2).sum())
        self.n += z_n.size
        if M > 1:
            infl = (M + 1) / M * np.var(ens, axis=0, ddof=1)
            self.var_sum += float(infl.sum())
            self.var_n += z_n.size
        r_m = ens - z_t[None]
        r_true = z_n - z_t
        B = z_n.shape[0]
        self.signs.append(float(np.mean(np.abs(
            np.sign(r_m).reshape(M, B, -1).mean(axis=2)))))
        num = (r_m * r_true[None]).reshape(M, B, -1).sum(2)
        den = (np.sqrt((r_m.reshape(M, B, -1) ** 2).sum(2))
               * np.sqrt((r_true.reshape(B, -1) ** 2).sum(1))[None] + 1e-30)
        self.corrs.append(float(np.mean(np.abs(num / den))))
        if HAVE_CRPS:
            try:
                c = crps_ensemble(ens.reshape(M, -1), z_n.reshape(-1))
                self.crpss.append((float(c["crps"] if isinstance(c, dict)
                                         else c), z_n.size))
            except Exception as e:                        # pragma: no cover
                print(f"::warning:: crps_ensemble failed ({e}); skipping",
                      flush=True)

    def result(self, nfe_spent):
        out = {"nfe_spent": int(nfe_spent), "members": self.M,
               "sample_ratio": (self.se_s / self.pe if self.pe > 0
                                else float("inf")),
               "ens_ratio": (self.se_e / self.pe if self.pe > 0
                             else float("inf")),
               "mse_pers": self.pe / self.n,
               "sign_coherence": float(np.mean(self.signs)),
               "mode_corr": float(np.mean(self.corrs))}
        if self.var_n:
            spread = math.sqrt(self.var_sum / self.var_n)
            rmse = math.sqrt(self.se_e / self.n)
            out["spread"] = spread
            out["spread_error"] = spread / rmse if rmse > 0 else float("inf")
        if self.crpss:
            w = float(sum(c[1] for c in self.crpss))
            out["crps"] = float(sum(c[0] * c[1] for c in self.crpss) / w)
        elif not HAVE_CRPS:
            out["crps"] = None
            out["crps_note"] = "ml/probscore.py absent — CRPS skipped, not faked"
        return out


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse(argv=None):
    p = argparse.ArgumentParser(
        description="E-052 field-head trainer, JAX/optax (det and EDM modes)")
    p.add_argument("--toy", choices=sorted(TOYS),
                   help="SMOKE ONLY — a numpy re-implementation of "
                        "ml/train_field.py's law, not the same sample path.")
    p.add_argument("--z-cache", help="[T, P, d_z] embed-cache .npy "
                                     "(float16 or float32), memmapped")
    p.add_argument("--data", help="tensor npz/sidecar — METADATA ONLY "
                                  "(lats/lons/months[/ys/xs]); X is never read")
    p.add_argument("--pixels", help="[P,2] or [2,P] .npy of (ys, xs) when the "
                                    "npz carries none")
    p.add_argument("--holdout-years", default="",
                   help='e.g. "2019,2020,2023". A window is VAL when its '
                        'TARGET bin falls in one of these years.')
    p.add_argument("--val-frac", type=float, default=0.15,
                   help="tail split, used only when there are no holdout years")
    p.add_argument("--mode", choices=["det", "diff"], default="det")
    p.add_argument("--K", type=int, default=144)
    p.add_argument("--patch", type=int, default=4)
    p.add_argument("--d-model", type=int, default=128)
    p.add_argument("--layers", type=int, default=4)
    p.add_argument("--heads", type=int, default=4)
    p.add_argument("--d-cond", type=int, default=128)
    p.add_argument("--cond-layers", type=int, default=2)
    p.add_argument("--cond-heads", type=int, default=4)
    p.add_argument("--steps", type=int, default=2000)
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--lr-schedule", default="constant",
                   choices=["constant", "expdecay"],
                   help="both HORIZON-FREE: lr(s) never depends on --steps. "
                        "expdecay halves every --lr-halflife ABSOLUTE steps "
                        "past the warmup, mirroring ml/temporal.py's own "
                        "expdecay closure.")
    p.add_argument("--lr-halflife", type=float, default=40000)
    p.add_argument("--lr-warmup", type=int, default=0)
    p.add_argument("--grad-clip", type=float, default=1.0,
                   help="max global gradient 2-norm. 0.0 = OFF, and OFF ADDS "
                        "NO OPTAX TRANSFORM AT ALL.")
    p.add_argument("--weight-decay", type=float, default=0.0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--input-znoise", type=float, default=0.0,
                   help="Gaussian std (in z units) added to the CONTEXT "
                        "frames' ocean values during TRAINING ONLY — the "
                        "field-head analog of the stencil head's "
                        "exposure-bias noise (E-029b). The eval and monitor "
                        "paths are never noised, and the persistence base z_t "
                        "and the target z_t+1 are never noised either, so this "
                        "perturbs the conditioner's INPUT and not the "
                        "objective. 0 takes a code path with no draw in it "
                        "at all.")
    p.add_argument("--eval-every", type=int, default=250)
    p.add_argument("--eval-windows", type=int, default=0,
                   help="cap on the val windows scored (0 = all). The subset "
                        "is EVENLY SPACED over the val pool and therefore "
                        "deterministic; a prefix would concentrate the "
                        "read-out on the earliest held-out windows.")
    p.add_argument("--eval-batch", type=int, default=0,
                   help="batch size for the eval passes (0 = --batch)")
    p.add_argument("--nfe", type=int, default=18)
    p.add_argument("--members", type=int, default=8)
    p.add_argument("--cond-chunk", type=int, default=1024,
                   help="token-chunk width for the CONDITIONER (0 = no "
                        "chunking). The denoiser is never chunked: it is full "
                        "attention over all tokens and chunking it would "
                        "change the model.")
    p.add_argument("--cond-remat", dest="cond_remat", action="store_true",
                   default=None,
                   help="gradient rematerialization on the per-chunk "
                        "conditioner. DEFAULT ON whenever --cond-chunk is set, "
                        "because chunking alone does NOT bound the training "
                        "footprint: lax.map lowers to a scan and a scan's "
                        "backward stacks every chunk's residuals, so the "
                        "retained set is the whole conditioner's activations "
                        "however narrow the chunk (~114-124 GB per chip at the "
                        "E-052.1 config, against 25 MB with remat — the "
                        "arithmetic is in cond_from_pixels' docstring). "
                        "Inert with no chunking.")
    p.add_argument("--no-cond-remat", dest="cond_remat", action="store_false",
                   help="the escape hatch: recompute nothing, retain "
                        "everything. Same numbers, more memory, marginally "
                        "less compute.")
    p.add_argument("--prefetch", type=int, default=2,
                   help="host-gather queue depth. 2 is the double buffer; 0 "
                        "builds batches inline. The batch SEQUENCE is "
                        "identical either way — one RNG, drawn in step order.")
    p.add_argument("--time-stride", type=int, default=0,
                   help="SUBSAMPLE THE AXIS: keep bins range(--time-offset, "
                        "T, N). 0 (default) keeps every bin.")
    p.add_argument("--time-offset", type=int, default=0)
    p.add_argument("--ckpt-dir", default="")
    p.add_argument("--ckpt-every", type=int, default=0,
                   help="steps between checkpoints (0 = at the eval cadence)")
    p.add_argument("--resume", action="store_true",
                   help="restore <--ckpt-dir>/ckpt_latest.npz exactly")
    p.add_argument("--out", default="", help="result JSON")
    p.add_argument("--metrics", default="", help="metrics.jsonl path")
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--quiet", action="store_true")
    return p.parse_args(argv)


def _apply_smoke(a):
    """Shrink every dimension. The point is the CODE PATH, not a number
    (ml/CLAUDE.md §4.8)."""
    a.steps, a.eval_every, a.batch = 20, 10, 4
    a.K, a.patch = 3, 2
    a.d_model, a.layers, a.heads = 32, 2, 2
    a.d_cond, a.cond_layers, a.cond_heads = 32, 1, 2
    a.nfe, a.members, a.eval_windows = 5, 2, 4
    a.cond_chunk = 4
    if not a.toy and not a.z_cache:
        a.toy = "gauss"
    return a


# ---------------------------------------------------------------------------
def main(argv=None):
    a = parse(argv)
    if a.smoke:
        a = _apply_smoke(a)
    # RESOLVE --cond-remat AFTER --smoke, which sets --cond-chunk: the default
    # is "on exactly when there is a chunk scan to stack residuals in", and a
    # default resolved before the flag it depends on would be silently wrong
    # in the one mode that always runs. `None` means "not stated"; from here
    # on it is a bool, so it lands in the config record and the checkpoint's
    # args as a fact rather than as an absence.
    if a.cond_remat is None:
        a.cond_remat = int(a.cond_chunk) > 0
    if bool(a.toy) == bool(a.z_cache):
        sys.exit("give exactly one of --toy or --z-cache (with --data)")
    if a.z_cache and not a.data:
        sys.exit("--z-cache needs --data <tensor npz> for the geometry")
    if a.d_model % 2:
        sys.exit("--d-model must be even (the 2-D position table's halves)")
    # A PRECONDITION THAT DEPENDS ONLY ON THE INPUTS IS CHECKED WHILE THE
    # INPUTS ARE ALL IT HAS COST (ml/CLAUDE.md §0.3).
    if a.grad_clip < 0:
        sys.exit(f"--grad-clip {a.grad_clip} must be >= 0 (0 = off, and off "
                 f"adds no optax transform at all).")
    if a.input_znoise < 0:
        sys.exit(f"--input-znoise {a.input_znoise} must be >= 0.")
    if a.time_stride and not (0 <= a.time_offset < a.time_stride):
        sys.exit(f"--time-offset {a.time_offset} must satisfy 0 <= O < N for "
                 f"--time-stride {a.time_stride}")

    devices = jax.local_devices()
    if not a.quiet:
        print(f"jax {jax.__version__} · devices "
              f"{[str(dv) for dv in devices]}", flush=True)

    rng = np.random.default_rng(a.seed)

    # ---- data ------------------------------------------------------------
    if a.toy:
        ds = TOYS[a.toy](np.random.default_rng(a.seed), smoke=a.smoke)
        hold = ()
    else:
        ds = load_real(a.z_cache, a.data, a.pixels)
        hold = tuple(int(y) for y in a.holdout_years.split(",") if y.strip())

    months = ds.get("months")
    rowmap = None
    T = ds["Z"].shape[0]
    if a.time_stride:
        rowmap = np.arange(a.time_offset, T, a.time_stride, dtype=np.int64)
        T = len(rowmap)
        months = None if months is None else [months[i] for i in rowmap]
        if not a.quiet:
            print(f"--time-stride {a.time_stride} offset {a.time_offset}: "
                  f"{T} bins kept · K={a.K} now spans {a.K} KEPT bins",
                  flush=True)
    if T < a.K + 2:
        sys.exit(f"T={T} is too short for K={a.K} (need at least K+2)")

    season = None
    if months is not None:
        season = np.array([_season_of_month(m) for m in months], np.float32)

    tok = OceanTokenizerJax(ds["H"], ds["W"], ds["ys"], ds["xs"], a.patch)
    win = Windows(ds["Z"], a.K, season, rowmap=rowmap, dtype=np.float32)
    tr_ts, va_ts = make_splits(T, a.K, months, hold, a.val_frac)
    if len(tr_ts) == 0 or len(va_ts) == 0:
        sys.exit(f"empty split: {len(tr_ts)} train / {len(va_ts)} val windows")
    sd = measure_sigma_data(win, tr_ts)
    _finite_or_die("sigma_data", sd)

    # The eval subset: EVENLY SPACED, so it is deterministic AND spans the
    # held-out record. Comparing two checkpoints on two different draws is the
    # E-005 failure mode with a newer metric.
    cap = a.eval_windows or len(va_ts)
    ev_ts = (va_ts if cap >= len(va_ts) else
             va_ts[np.linspace(0, len(va_ts) - 1, cap).astype(np.int64)])

    # ---- the model --------------------------------------------------------
    model = FieldHeadJax(tok, ds["d_z"], a.K, mode=a.mode, d_model=a.d_model,
                         layers=a.layers, heads=a.heads, d_cond=a.d_cond,
                         cond_layers=a.cond_layers, cond_heads=a.cond_heads,
                         sigma_data=sd, rngs=nnx.Rngs(a.seed))
    # SPLIT AFTER every static value is settled — `sigma_data` lives in the
    # graphdef (ml/jaxport/field_model.py's docstring), so a split taken
    # before it is known would jit against the wrong constant.
    graphdef, state = nnx.split(model)
    n_par = count_params_jax(model)

    name = f"{a.toy or 'real'}_{a.mode}_s{a.seed}"
    out_path = a.out or os.path.join(HERE, "runs", "field", f"{name}.json")
    metrics_path = a.metrics or (os.path.join(a.ckpt_dir, "metrics.jsonl")
                                 if a.ckpt_dir else "")

    def m2(rec):
        if not metrics_path:
            return
        try:
            os.makedirs(os.path.dirname(os.path.abspath(metrics_path)) or ".",
                        exist_ok=True)
            with open(metrics_path, "a") as f:
                f.write(json.dumps(rec) + "\n")
        except OSError:
            pass                      # instrumentation never breaks the run

    if not a.quiet:
        print(f"[E-052/jax] {name} · params {n_par:,} · ntok {tok.ntok} · P "
              f"{tok.P} · d_z {ds['d_z']} · sigma_data {sd:.4f} · train/val "
              f"{len(tr_ts)}/{len(va_ts)} · out {out_path}", flush=True)

    # ---- optimiser --------------------------------------------------------
    # OFF ADDS NO TRANSFORM. Not `clip_by_global_norm(inf)`, which is a no-op
    # only for finite norms; not a "very large" threshold, which is a promise
    # about a distribution nobody has measured.
    tx = optax.inject_hyperparams(optax.adamw)(
        learning_rate=a.lr, weight_decay=a.weight_decay)
    CLIP = float(a.grad_clip)
    if CLIP > 0:
        tx = optax.chain(optax.clip_by_global_norm(CLIP), tx)
    opt_state = tx.init(state)

    def _set_lr(ost, lr):
        """Write `lr` into whichever level of the chain owns hyperparams.
        `_replace` rather than mutating in place: the dict is a node of a
        TRACED pytree."""
        if CLIP > 0:
            inner = ost[1]
            return (ost[0], inner._replace(
                hyperparams={**inner.hyperparams, "learning_rate": lr}))
        return ost._replace(hyperparams={**ost.hyperparams,
                                         "learning_rate": lr})

    # ---- sharding ---------------------------------------------------------
    # Data parallelism and nothing cleverer: the BATCH axis is sharded across
    # local devices, the parameters replicated. On one device this whole block
    # is a no-op, which is every CPU test — the point of writing it this way is
    # that the correctness path does not branch.
    shard = rep = None
    ndev = len(devices)
    if ndev > 1:
        mesh = jax.make_mesh((ndev,), ("b",))
        shard = jax.sharding.NamedSharding(mesh,
                                           jax.sharding.PartitionSpec("b"))
        rep = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())
        state = jax.device_put(state, rep)
        opt_state = jax.device_put(opt_state, rep)
        if not a.quiet:
            print(f"data-parallel over {ndev} devices: batch {a.batch} -> "
                  f"{a.batch // ndev} per device", flush=True)

    def put(x, sharded=True):
        """HOST numpy -> device, SHARDED IN PLACE.

        `jnp.asarray(x)` would commit the WHOLE array to device 0 and re-slice
        it there; at K=144 that is a multi-GB staging copy on a 16 GB chip and
        it is exactly how the first E-051 node died (RESOURCE_EXHAUSTED,
        2026-08-25). `jax.device_put` from a numpy array slices on the host and
        transfers each shard alone.
        """
        x = np.asarray(x)
        if shard is not None and sharded and x.ndim >= 1 and \
                x.shape[0] % ndev == 0:
            return jax.device_put(x, shard)
        if rep is not None:
            return jax.device_put(x, rep)
        return jax.device_put(x)

    # ---- the jitted steps -------------------------------------------------
    CHUNK = int(a.cond_chunk)
    REMAT = bool(a.cond_remat)
    MODE = a.mode
    NZ = float(a.input_znoise)

    def _loss(st, ctx, z_t, z_n, sea, key):
        m = nnx.merge(graphdef, st)
        cond = cond_from_pixels(m, ctx, sea, CHUNK, REMAT)
        if MODE == "det":
            return ((m.forward_det(cond, z_t) - z_n) ** 2).mean()
        return m.edm_loss(cond, z_t, z_n, key)

    def _noised(ctx, key):
        """`--input-znoise`, on the CONTEXT frames only.

        Drawn INSIDE the jit: the host draw was measured at ~15 s/step at
        K=144 on the E-051 launch and is what starved that node's input
        pipeline (commit c4ce2da). Every pixel in Z is an ocean pixel by
        construction, so there is no live/dead slot distinction to make here —
        unlike the stencil head, whose zero slots ARE its missing-neighbour
        encoding.
        """
        return ctx + NZ * jax.random.normal(key, ctx.shape, ctx.dtype)

    @jax.jit
    def train_step(st, ost, lr, key, ctx, z_t, z_n, sea):
        loss, grads = jax.value_and_grad(_loss)(st, ctx, z_t, z_n, sea, key)
        # THE PRE-CLIP GLOBAL NORM, on EVERY step: clipping computes it
        # anyway, and the unclipped path pays one extra reduction, which is
        # what turns `grad_norm` from "one step in log_every" into a window
        # statistic (§4.10).
        gnorm = optax.global_norm(grads)
        ost = _set_lr(ost, lr)
        upd, ost = tx.update(grads, ost, st)
        return optax.apply_updates(st, upd), ost, loss, gnorm

    @jax.jit
    def train_step_nz(st, ost, lr, key, nkey, ctx, z_t, z_n, sea):
        """The noised twin. A SEPARATE function, not a `sigma * 0` branch, so
        `--input-znoise 0` is the unnoised code path exactly — bit-identical
        to the flag being absent, which F8 gates. Its noise key comes off its
        OWN root so the EDM stream is untouched by whether noise is on."""
        loss, grads = jax.value_and_grad(_loss)(
            st, _noised(ctx, nkey), z_t, z_n, sea, key)
        gnorm = optax.global_norm(grads)
        ost = _set_lr(ost, lr)
        upd, ost = tx.update(grads, ost, st)
        return optax.apply_updates(st, upd), ost, loss, gnorm

    @jax.jit
    def eval_det_step(st, ctx, z_t, sea):
        m = nnx.merge(graphdef, st)
        return m.forward_det(
            cond_from_pixels(m, ctx, sea, CHUNK, REMAT), z_t)

    @partial(jax.jit, static_argnames=("n_steps", "members"))
    def sample_batch(st, ctx, z_t, sea, key, n_steps, members):
        """M members through the Heun sampler, as ONE jitted graph.

        `n_steps` and `members` are STATIC because they are python loop bounds
        inside `sample` (the Heun loop is unrolled and the `float(s_n) > 0.0`
        branch is resolved on the host, exactly as torch resolves it). They are
        constant for a run, so this compiles once; leaving the whole sampler
        eager would dispatch 2*n_steps-1 denoiser calls op-by-op per batch,
        which at real scale is the eval and not the training that costs the
        wall clock.
        """
        m = nnx.merge(graphdef, st)
        cond = cond_from_pixels(m, ctx, sea, CHUNK, REMAT)
        return m.sample(cond, z_t, n_steps, key, M=members)

    # ---- resume -----------------------------------------------------------
    step0, history = 0, []
    ck_npz = os.path.join(a.ckpt_dir, "ckpt_latest.npz") if a.ckpt_dir else ""
    ck_pt = os.path.join(a.ckpt_dir, "field_latest.pt") if a.ckpt_dir else ""
    if a.resume:
        if not ck_npz or not os.path.exists(ck_npz):
            sys.exit(f"--resume needs an existing {ck_npz or '<--ckpt-dir>'}")
        state, opt_state, step0, _ra, rng_state, history = load_state_npz(
            ck_npz, state, opt_state)
        rng.bit_generator.state = rng_state
        if step0 >= a.steps:
            sys.exit(f"--resume: the checkpoint is at step {step0:,} and "
                     f"--steps is {a.steps:,}. --steps is the TOTAL, not the "
                     f"extra.")
        if not a.quiet:
            print(f"  RESUMED from {ck_npz} at step {step0:,} (optimiser, "
                  f"schedule position and window RNG restored)", flush=True)
        m2({"resumed": {"from": os.path.basename(ck_npz), "at_step": step0,
                        "to_step": a.steps, "backend": "jax"}})

    cfg = dict(vars(a))
    cfg.update(law=ds["law"], T=T, P=int(tok.P), d_z=int(ds["d_z"]),
               ntok=tok.ntok, H=ds["H"], W=ds["W"], sigma_data=sd,
               params=int(n_par), n_train=int(len(tr_ts)),
               n_val=int(len(va_ts)), n_eval=int(len(ev_ts)),
               have_crps=HAVE_CRPS, backend="jax",
               devices=[str(dv) for dv in devices])
    m2({"field_config": {k: v for k, v in cfg.items()
                         if isinstance(v, (int, float, str, bool, type(None)))}})

    # ---- the eval ---------------------------------------------------------
    eb = a.eval_batch or a.batch
    eval_root = jax.random.PRNGKey(10_000 + 7 * a.seed)

    def evaluate(st):
        if a.mode == "det":
            se = pe = n = 0.0
            for i in range(0, len(ev_ts), eb):
                ctx, z_t, z_n, sea = win.batch(ev_ts[i:i + eb])
                zh = np.asarray(eval_det_step(st, put(ctx), put(z_t),
                                              put(sea)), np.float64)
                se += float(((zh - z_n) ** 2).sum())
                pe += float(((z_t - z_n) ** 2).sum())
                n += z_n.size
            return {"mse": se / n, "mse_pers": pe / n,
                    "ratio": se / pe if pe > 0 else float("inf")}
        n_steps, spent = nfe_to_steps(a.nfe)
        acc = DiffAccum(a.members)
        for bi, i in enumerate(range(0, len(ev_ts), eb)):
            ctx, z_t, z_n, sea = win.batch(ev_ts[i:i + eb])
            ens = sample_batch(st, put(ctx), put(z_t), put(sea),
                               jax.random.fold_in(eval_root, bi),
                               n_steps=n_steps, members=a.members)
            acc.add(np.asarray(ens), z_t, z_n)
        return acc.result(spent)

    # ---- the host batch producer -----------------------------------------
    def make_batches():
        """One generator, one RNG, drawn in step order — the SEQUENCE of
        batches is identical whether it runs inline or in the producer
        thread, because this function is the only thing that touches `rng`
        once training starts."""
        for s_ in range(step0 + 1, a.steps + 1):
            idx = rng.integers(0, len(tr_ts), a.batch)
            yield (s_,) + win.batch(tr_ts[idx])

    if a.prefetch > 0:
        q = _queue.Queue(maxsize=int(a.prefetch))
        DONE = object()

        def _produce():
            try:
                for item in make_batches():
                    q.put(item)
                q.put(DONE)
            except BaseException as e:            # surface, never deadlock
                q.put(e)

        threading.Thread(target=_produce, daemon=True).start()

        def _stream():
            while True:
                item = q.get()
                if item is DONE:
                    return
                if isinstance(item, BaseException):
                    raise item
                yield item
        batch_iter = _stream()
    else:
        batch_iter = make_batches()

    # ---- checkpoints ------------------------------------------------------
    def head_args(step):
        out = dict(vars(a))
        out.update(backend="jax", step_recorded=int(step), d_z=int(ds["d_z"]),
                   sigma_data=sd, H=int(ds["H"]), W=int(ds["W"]),
                   patch=int(a.patch), ntok=int(tok.ntok), P=int(tok.P))
        return out

    def save_ckpt(step):
        """Both artefacts, because they answer different questions: the `.npz`
        is what THIS trainer resumes from (optimiser moments and the window
        RNG included) and the `.pt` is what the UNCHANGED torch stack loads.
        Both land at STABLE names, written atomically — the bucket-shipping
        launcher watches --ckpt-dir for files appearing, and a half-written
        file it picks up is worse than no file."""
        if not a.ckpt_dir:
            return
        os.makedirs(a.ckpt_dir, exist_ok=True)
        save_state_npz(ck_npz, state, opt_state, step, head_args(step),
                       rng.bit_generator.state, history)
        export_field_pt(nnx.merge(graphdef, state), head_args(step),
                        path=ck_pt, step=int(step),
                        run_number=os.environ.get("GITHUB_RUN_NUMBER"))

    # ---- train ------------------------------------------------------------
    t0 = time.time()
    log_every = max(1, min(a.eval_every or a.steps, max(1, a.steps // 100)))
    ck_every = a.ckpt_every or a.eval_every or a.steps
    nz_root = jax.random.PRNGKey(1_000_000 + a.seed)
    edm_root = jax.random.PRNGKey(a.seed)
    gmax, gbad = 0.0, 0
    lv = float("nan")
    step = step0

    for s, ctx, z_t, z_n, sea in batch_iter:
        step = s
        lr_now = jnp.asarray(a.lr * lr_factor(s - 1, a), jnp.float32)
        key = jax.random.fold_in(edm_root, s)
        args_dev = (put(ctx), put(z_t), put(z_n), put(sea))
        if NZ > 0:
            state, opt_state, loss, gnorm = train_step_nz(
                state, opt_state, lr_now, key,
                jax.random.fold_in(nz_root, s), *args_dev)
        else:
            state, opt_state, loss, gnorm = train_step(
                state, opt_state, lr_now, key, *args_dev)
        lv, gv = float(loss), float(gnorm)
        if not math.isfinite(lv):
            m2({"step": s, "diverged": {"loss": lv, "grad_norm": gv}})
            sys.exit(f"ABORTING at step {s}: loss is {lv}. Every further step "
                     f"writes NaN into the weights. Suspect the learning rate "
                     f"first"
                     + ("" if CLIP > 0 else " (there is no gradient clipping "
                                            "on this run — --grad-clip 0)")
                     + ".")
        if math.isfinite(gv):
            gmax = max(gmax, gv)
        else:
            gbad += 1

        if s % log_every == 0 or s == a.steps:
            m2({"step": s, "loss": round(lv, 6),
                "lr": float(a.lr * lr_factor(s - 1, a)),
                "grad_norm": round(gv, 6),
                "wall_s": round(time.time() - t0, 2)})

        if a.eval_every and (s % a.eval_every == 0 or s == a.steps):
            rec = {"step": s, "loss": lv,
                   "wall_s": round(time.time() - t0, 2),
                   "lr": float(a.lr * lr_factor(s - 1, a)),
                   "grad_norm_max": round(gmax, 4), "grad_nonfinite": gbad}
            gmax, gbad = 0.0, 0
            rec.update(evaluate(state))
            for k, v in rec.items():
                if isinstance(v, float):
                    _finite_or_die(f"eval key {k!r} at step {s}", v)
            history.append(rec)
            m2({"field_eval": rec})
            write_result(out_path, cfg, history, final=None,
                         in_progress={"step": s, "of": a.steps})
            if not a.quiet:
                head = " · ".join(
                    f"{k} {v:.4f}" if isinstance(v, float) else f"{k} {v}"
                    for k, v in rec.items()
                    if k in ("step", "loss", "ratio", "sample_ratio",
                             "ens_ratio", "sign_coherence", "mode_corr"))
                print(f"  {head}", flush=True)

        if s % ck_every == 0 or s == a.steps:
            save_ckpt(s)

    final = dict(history[-1]) if history else {}
    final["steps"] = a.steps
    final["wall_s"] = round(time.time() - t0, 2)
    final["backend"] = "jax"

    if a.smoke:
        # ASSERT THE EFFECT, not the invocation (ml/CLAUDE.md §0.2). A smoke
        # that only "ran" proves nothing; these checks are the artefact.
        ctx, z_t, _, sea = win.batch(ev_ts[:2])
        n_steps, spent = nfe_to_steps(a.nfe)
        ens = sample_batch(state, put(ctx, False), put(z_t, False),
                           put(sea, False), jax.random.PRNGKey(1234),
                           n_steps=n_steps, members=2)
        want = (2, len(ev_ts[:2]), int(tok.P), int(ds["d_z"]))
        # `sys.exit`, not `assert`: a check that vanishes under -O is not a
        # check, and this one is the whole point of --smoke.
        if tuple(ens.shape) != want:
            sys.exit(f"--smoke: sample() returned {tuple(ens.shape)}, want "
                     f"{want}")
        if not bool(np.isfinite(np.asarray(ens)).all()):
            sys.exit("--smoke: sample() produced a non-finite value")
        if not history:
            sys.exit("--smoke: no eval ran, so nothing would have been written")
        final["smoke_sample_shape"] = list(ens.shape)
        final["smoke_nfe_spent"] = spent

    write_result(out_path, cfg, history, final=final, in_progress=None)
    with open(out_path) as f:
        chk = json.load(f)
    if "in_progress" in chk:
        sys.exit("the completed result file still carries in_progress — that "
                 "key's absence is the run's only completion certificate")
    save_ckpt(a.steps)
    m2({"field_result": {"steps": a.steps, "mode": a.mode, "seed": a.seed,
                         "params": int(n_par), "backend": "jax",
                         "sigma_data": sd,
                         **{k: v for k, v in final.items()
                            if isinstance(v, (int, float, str))}}})
    if not a.quiet:
        print(f"[E-052/jax] done · {out_path}", flush=True)
    return {"out": out_path, "config": cfg, "history": history, "final": final,
            "state": state, "opt_state": opt_state, "graphdef": graphdef,
            "step": step}


if __name__ == "__main__":
    main()

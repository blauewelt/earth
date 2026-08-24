#!/usr/bin/env python3
"""The STAGE-1 CODEC TRAINER in JAX/optax — tier 3 of `ml/plans/JAX_PORT.md`.

This mirrors `ml/train.py`'s stage-1 semantics and nothing else. It exists for
one operational reason (`JAX_PORT.md` §1b): TPU spot capacity is the one
obvious lever this programme has not tried, JAX is the native way to reach it,
and a trainer is what a TPU needs. The torch stack stays operational — every
published number still comes from it, no torch script imports anything from
here, and the artefact this writes is an ordinary `.pt` blob (`convert.
export_pt`) so a TPU-trained codec is scored by the UNCHANGED torch eval
ladder rather than by a second, self-marking scoreboard.

WHAT IS MIRRORED, TERM FOR TERM
  · the masking recipe — `mask = (rand(B,C) < mask_ratio) & observed`, so only
    OBSERVED cells are ever hidden and only hidden-or-visible-observed cells
    are ever scored;
  · `loss_rec` (huber over every channel queried at offset 0, weighted
    `mask + rec_w_visible * (observed & ~mask)`, times the per-channel
    upweights) and `loss_nei` (one random neighbour offset per sample out of
    the same six, weighted by the neighbour's observed flags), combined as
    `loss = loss_rec + 0.5 * loss_nei`;
  · the cosine LR schedule and its `--lr-floor` / `--lr-decay-steps`
    decay-then-constant variant, including the `--max-minutes` refit, which
    reuses `ml/train.py:fit_schedule` itself rather than a second copy;
  · the collapse guard — light probe → `linear_r_deseas`, `--collapse-r`,
    `--collapse-strikes`, **NaN is NO READING and neither strikes nor resets**;
  · the non-finite-loss abort, which is the other half of that guard and the
    one case where NaN means the model rather than the instrument;
  · milestone checkpoints and resume (weights + optimiser + step + schedule);
  · `--time-block`, through `ml/timeblocks.py:BlockAxis` — the same axis
    arithmetic, imported, so a block boundary cannot drift between backends;
  · `metrics.jsonl` with THE SAME KEYS, so the status page and every existing
    monitoring reader work on a JAX run unchanged.

WHAT IS NOT (each named here so nothing is silently missing):
  · **the FULL probe** (`--eval-every`): its second half is a mini torch
    TemporalTransformer trained inside the probe. That is stage-2 machinery
    and belongs to the stage-2 port; the flag REFUSES rather than quietly
    running a different, lighter thing under the name the archive knows.
  · **`--amp`** does not exist here at all — there is no autocast on this path
    in torch either. `--bf16` is a separate, off-by-default TPU knob (below).
  · **`plot_run`** — a matplotlib curve of the run, which is a rendering, not
    an arithmetic step.
  There is no EMA and no gradient accumulation in `ml/train.py` stage-1, so
  there is none here; and there is no warmup and no gradient clipping there
  either — `--warmup-steps` and `--grad-clip` exist here, both DEFAULT OFF, so
  the default trajectory is the torch one and a TPU operator still has the two
  knobs a large-batch run usually wants.

WHY THIS FILE IMPORTS FROM THE TORCH TREE. `ml/CLAUDE.md`'s standing rule is
that shared numpy plumbing is IMPORTED, never copied — two copies of one
transform are two places for one numerical bug to live, and this repo has paid
that bill twice. The plumbing it needs (`anomaly_transform`, `ridge_r`,
`rapid_section`, `light_rows`, `lon_holdout_mask`, `fit_schedule`,
`obs_any_chunked`, `pool_idx`, `LazyPixels`) is pure numpy, but it LIVES in
modules that import torch at their top. So this DRIVER needs a torch wheel
installed — the same CPU wheel `jaxport.convert` already needs to read a `.pt`
— while `jaxport.models` and `jaxport.convert` themselves stay importable with
no torch at all. `ml/jaxport/score_section_probe.py` set that precedent and
this follows it. Copying the helpers to avoid the wheel would trade a 200 MB
download for a class of bug this project has already been bitten by.

RNG. The mask draw and the neighbour pick come from ONE seeded
`np.random.default_rng`. A torch run's stream cannot be reproduced in JAX (or
the reverse), so bit-identical TRAJECTORIES are impossible by construction and
are not claimed; what IS claimed, and gated, is that on the SAME weights and
the SAME batch the two frameworks compute the same loss to 1e-5 and take the
same SGD step to 1e-6 (`tests/test_jaxport_train.py`, G4a/G4b).

    # toy, CPU, ~1 minute
    python3 ml/jaxport/train_stage1.py --data toy.npz --out /tmp/run \\
        --steps 300 --batch 64 --d-z 32 --patch 1 --d-model 32 \\
        --n-layers 2 --n-heads 4 --d-dec 32 --anomaly
"""
import argparse
import datetime as dt
import json
import math
import os
import re
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ML = os.path.dirname(HERE)
if ML not in sys.path:
    sys.path.insert(0, ML)

import jax                                                      # noqa: E402
import jax.numpy as jnp                                         # noqa: E402
import optax                                                    # noqa: E402
from flax import nnx                                            # noqa: E402

from jaxport.models import PixelMAE                             # noqa: E402
from jaxport.convert import export_pt, load_pixelmae            # noqa: E402
from timeblocks import BlockAxis                                # noqa: E402
import fsq_ladder as fql                                        # noqa: E402

# The numpy plumbing, imported rather than copied — see the module docstring
# for why these pull torch in and why that is the right trade.
from model import LazyPixels, obs_any_chunked, pool_idx         # noqa: E402
from train import lon_holdout_mask, fit_schedule               # noqa: E402
from tensor_io import load_tensor, writable_copy               # noqa: E402

# The six neighbour offsets (Δx, Δy, Δt), in ml/train.py's own order. The
# order is load-bearing only through the RNG draw, but it is written the same
# way so a reader comparing the two files sees one list, not two.
NEI = [(1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1)]

# `fsq_levels` and the E-048 ladder fields are ARCHITECTURE here for the
# reason ml/train.py gives: they decide what a `z` IS, every downstream
# consumer reads them back out of the checkpoint, and a resume that
# contradicted one would continue a quantized codec as a continuous one (or a
# geometric lattice as an even one) while every leaf loaded cleanly.
# `fsq_ladder_fit` is NOT here — it is a MEASUREMENT the run made rather than a
# setting a dispatch states, so it is adopted unconditionally.
ARCH = ("d_z", "patch", "d_model", "n_layers", "n_heads", "d_dec",
        "dec_layers", "fsq_levels", "fsq_ladder", "fsq_exp_base")


# --------------------------------------------------------------------------
# the objective
# --------------------------------------------------------------------------
def huber(pred, tgt):
    """`torch.nn.HuberLoss(reduction="none")`, delta = 1 (torch's default).

    Written out rather than taken from optax so the algebra is visible at the
    one place it is defined (`ml/CLAUDE.md` §4.5: for a loss, the gradient IS
    the specification). `optax.huber_loss(pred, tgt, delta=1.0)` is the same
    function; the parity gate would catch a disagreement either way.
    """
    e = pred - tgt
    a = jnp.abs(e)
    return jnp.where(a < 1.0, 0.5 * e * e, a - 0.5)


def stage1_loss(model, enc_v, enc_o, v, o, mask, ctx, qc, off0, offn,
                vn, on, cw, rec_w_visible, tpos=None):
    """`loss_rec, loss_nei` — the mirror of `ml/train.py:step_loss` (:686) and
    `step_loss_block` (:647), which are the same objective with a cell axis
    added.

    Every array is passed in rather than drawn here, which is what makes the
    parity gate possible at all: identical weights and identical inputs, two
    frameworks, one number.

    Shapes. `enc_v`/`enc_o` are whatever `encode` takes for this codec —
    [B,C] at patch 1, [B,C,patch²] at patch>1, [B,k_time,C] for a block codec.
    `v`/`o`/`mask` are the SCORING arrays and are flattened to [B,M] here
    (M = C, or k_time·C for a block codec); `cw` is the per-channel weight
    vector already tiled to M. At patch>1 `enc_v` is deliberately NOT
    pre-masked and `v`/`o` are the CENTRE cell's values and flags, exactly as
    the torch caller arranges it.
    """
    B = v.shape[0]
    z = model.encode(enc_v, enc_o, mask, ctx)
    kt = model.k_time > 1
    pred = model.query(z, qc, off0, tpos) if kt else model.query(z, qc, off0)
    vf = v.reshape(B, -1)
    of = o.reshape(B, -1)
    mf = mask.reshape(B, -1)
    w = (mf.astype(jnp.float32)
         + rec_w_visible * (of & ~mf).astype(jnp.float32)) * cw[None, :]
    l_rec = (huber(pred, vf) * w).sum() / jnp.maximum(w.sum(), 1.0)

    predn = model.query(z, qc, offn, tpos) if kt else model.query(z, qc, offn)
    wn = on.reshape(B, -1).astype(jnp.float32) * cw[None, :]
    l_nei = (huber(predn, vn.reshape(B, -1)) * wn).sum() / jnp.maximum(
        wn.sum(), 1.0)
    return l_rec, l_nei


def lr_factor(e, floor, total, warmup=0):
    """`ml/train.py`'s LambdaLR closure (:608), as a plain function of `e`.

        floor + (1 - floor) * 0.5 * (1 + cos(pi * min(e, total) / total))

    `e` is the number of `sched.step()` calls already made, so at TRAINING
    step s (1-indexed) the factor is `lr_factor(s - 1)` — torch steps the
    scheduler AFTER the optimiser, and optax's own count is 0 on the first
    update, which makes the two line up with no off-by-one.

    `warmup` is 0 on the torch path and 0 by default here; above 0 it
    multiplies the first `warmup` steps by (e + 1) / warmup.
    """
    total = max(1, int(total))
    f = floor + (1 - floor) * 0.5 * (1 + math.cos(math.pi * min(e, total)
                                                  / total))
    if warmup and e < warmup:
        f *= (e + 1) / float(warmup)
    return f


# --------------------------------------------------------------------------
# checkpoint I/O
# --------------------------------------------------------------------------
def _leaves(tree):
    return jax.tree_util.tree_leaves(tree)


def save_state_npz(path, state, opt_state, step, args):
    """The JAX-native checkpoint: params + optimiser moments + step + args.

    Flat `.npz` over the pytree LEAVES, in `jax.tree_util` order. That order is
    a deterministic function of the tree STRUCTURE, and the structure is a
    deterministic function of the architecture — which the file also carries,
    so a load rebuilds the same structure before it unflattens into it and
    refuses on any count or shape it did not expect. It is a small format on
    purpose: an optimiser-state checkpoint that cannot be read back by a
    slightly different flax release is worse than no checkpoint at all.
    """
    blob = {"_step": np.asarray(int(step)),
            "_args": np.asarray(json.dumps(args)),
            "_n_state": np.asarray(len(_leaves(state))),
            "_n_opt": np.asarray(len(_leaves(opt_state)))}
    for i, v in enumerate(_leaves(state)):
        blob[f"s{i}"] = np.asarray(v)
    for i, v in enumerate(_leaves(opt_state)):
        blob[f"o{i}"] = np.asarray(v)
    # A FILE HANDLE, not a name: `np.savez` appends ".npz" to any path that
    # does not already end in it, so `savez(path + ".tmp")` writes
    # `path.tmp.npz` and the rename that follows finds nothing. Handed a
    # handle it writes exactly where it is told.
    tmp = path + ".tmp"
    with open(tmp, "wb") as fh:
        np.savez(fh, **blob)
    os.replace(tmp, path)          # flush, THEN publish (ml/CLAUDE.md §5.21)


def load_state_npz(path, state, opt_state):
    """Read `save_state_npz` back INTO the structures already built.

    Returns (state, opt_state, step, args). Refuses on a leaf-count or shape
    mismatch rather than unflattening whatever fits — a checkpoint half-loaded
    into a differently shaped model still produces numbers.
    """
    z = np.load(path, allow_pickle=False)
    args = json.loads(str(z["_args"]))
    sl, ol = _leaves(state), _leaves(opt_state)
    if int(z["_n_state"]) != len(sl) or int(z["_n_opt"]) != len(ol):
        raise SystemExit(
            f"REFUSING to resume {path}: it holds {int(z['_n_state'])} state "
            f"leaves and {int(z['_n_opt'])} optimiser leaves; this model has "
            f"{len(sl)} and {len(ol)}. The architecture or the optimiser "
            f"differs from the one that wrote it.")
    new_s, new_o = [], []
    for i, v in enumerate(sl):
        a = z[f"s{i}"]
        if a.shape != tuple(v.shape):
            raise SystemExit(f"REFUSING to resume {path}: state leaf {i} is "
                             f"{a.shape}, the model wants {tuple(v.shape)}")
        new_s.append(jnp.asarray(a, v.dtype))
    for i, v in enumerate(ol):
        a = z[f"o{i}"]
        if a.shape != tuple(v.shape):
            raise SystemExit(f"REFUSING to resume {path}: optimiser leaf {i} "
                             f"is {a.shape}, the model wants {tuple(v.shape)}")
        new_o.append(jnp.asarray(a, v.dtype))
    return (jax.tree_util.tree_unflatten(
                jax.tree_util.tree_structure(state), new_s),
            jax.tree_util.tree_unflatten(
                jax.tree_util.tree_structure(opt_state), new_o),
            int(z["_step"]), args)


# --------------------------------------------------------------------------
# the light probe
# --------------------------------------------------------------------------
def light_probe(model, X, OBS, d, moy, t_hold, x_hold, ocean=None,
                blk_rows=None, blk_pad=None, batch=8192):
    """`ml/trainprobe.py:probe_now(light=True)`, with the JAX encoder.

    The ESTIMATOR is imported, not re-derived: `rapid_section` picks the
    protocol-v3 section, `light_rows` picks which RAPID rows are embedded (one
    per decorrelation time, with the >= LIGHT_MIN_TEST floor enforced against
    this tensor), the deseasonalisation is the same train-years monthly
    climatology, and `ridge_r` is the same standardise-pick-lambda-score
    solve. What differs is one thing: which encoder produced the section's
    embeddings. Emits `linear_r_deseas`, which is the key the collapse guard
    and the status page read.
    """
    from temporal import rapid_section                     # torch at import
    from trainprobe import light_rows
    from probe_sequence import ridge_r
    from jaxport.embed import embed_everything_jax

    t0 = time.time()
    lats, lons = d["lats"], d["lons"]
    ctx_all = (np.stack([np.sin(2 * np.pi * moy / 12),
                         np.cos(2 * np.pi * moy / 12)], 1)
               if blk_rows is None else d["_blk_ctx"])
    if ocean is None:
        ocean = np.isfinite(np.asarray(X[0])).any(axis=-1)
    ys, xs = np.where(ocean)
    sec_y, sec_sel = rapid_section(lats, lons, ys, xs)

    rapid = d["rapid"]
    ridx = rapid[:, 0].astype(int)
    rv_raw = rapid[:, 1].copy()
    rmoy = moy[ridx]
    tr_all = ~t_hold[ridx]
    rclim = np.array([rv_raw[tr_all & (rmoy == m)].mean() for m in range(12)])
    rv_des = rv_raw - rclim[rmoy]
    te_all = t_hold[ridx]
    lsel, lstride = light_rows(ridx, tr_all, te_all, d["months"])

    rt = ridx[lsel]
    tsel, inv = np.unique(rt, return_inverse=True)
    Zsec, _ = embed_everything_jax(model, X, OBS, ctx_all, lats, lons,
                                   ys[sec_sel], xs[sec_sel], model.d_z,
                                   batch=batch, t_sel=tsel,
                                   blk_rows=blk_rows, blk_pad=blk_pad,
                                   quiet=True)
    Fl = np.asarray(Zsec, np.float32).mean(1)[inv]
    out = {}
    out["linear_r_deseas"], _ = ridge_r(Fl, rv_des[lsel], tr_all[lsel],
                                        te_all[lsel])
    out["linear_r_raw"], _ = ridge_r(Fl, rv_raw[lsel], tr_all[lsel],
                                     te_all[lsel])
    out["light"] = True
    out["light_stride"] = int(lstride)
    out["light_n"] = int(len(lsel))
    out["light_n_test"] = int(te_all[lsel].sum())
    out["probe_seconds"] = round(time.time() - t0, 1)
    return out


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------
def _optint(v):
    v = (v or "").strip()
    return None if v == "" else int(v)


def parse(argv=None):
    p = argparse.ArgumentParser(
        description="stage-1 PixelMAE codec trainer, JAX/optax "
                    "(ml/plans/JAX_PORT.md tier 3)")
    p.add_argument("--data", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--steps", type=int, default=20000)
    p.add_argument("--batch", type=int, default=512)
    p.add_argument("--lr", type=float, default=3e-4)
    # ARCHITECTURE — None is FATAL unless --resume supplies it or --smoke asks
    # for the pilot, for exactly the reason ml/train.py gives: a default that
    # is never correct is not a default, and omitting a flag used to run a
    # different experiment in silence (#395, #387).
    p.add_argument("--d-z", type=_optint, default=None)
    p.add_argument("--patch", type=_optint, default=None, choices=[1, 3, None])
    p.add_argument("--d-model", type=_optint, default=None)
    p.add_argument("--n-layers", type=_optint, default=None)
    p.add_argument("--n-heads", type=_optint, default=None)
    p.add_argument("--d-dec", type=_optint, default=None)
    p.add_argument("--dec-layers", type=int, default=2)
    p.add_argument("--mask-ratio", type=float, default=0.5)
    p.add_argument("--rec-w-visible", type=float, default=0.1)
    p.add_argument("--upweight-chans", default="")
    p.add_argument("--upweight", type=float, default=1.0)
    p.add_argument("--holdout-years", default="2009,2017,2023")
    p.add_argument("--holdout-lon", default="-45,-25")
    p.add_argument("--time-block", default="",
                   help="E-047/E-048: 'month', an integer N, or 'W/S' — a "
                        "width-W window advancing by S bins ('6/3' is 30 days "
                        "of input every 15 days, consecutive embeddings "
                        "sharing 3 of 6 bins). '' is the per-bin codec every "
                        "archived checkpoint is. The axis arithmetic is "
                        "ml/timeblocks.py's, imported, so a block boundary "
                        "cannot drift between backends.")
    # E-046/E-048: the FSQ bottleneck and its ladder. Same spellings, same
    # defaults and the same refusals as ml/train.py, because the TPU is the
    # SWEEP VEHICLE for E-048 and a knob that existed on one backend only
    # would make the sweep un-runnable where it is meant to run.
    p.add_argument("--fsq-levels", default="",
                   help="E-046: quantize the codec's bottleneck with FSQ. '' "
                        "(default) is the continuous bottleneck, bit-"
                        "identical. A single integer puts that many levels on "
                        "every dimension; a comma list must be exactly --d-z "
                        "long. L=2 is refused.")
    p.add_argument("--fsq-ladder", default="uniform",
                   choices=["uniform", "exp", "auto"],
                   help="E-048: WHERE those levels sit. 'uniform' is E-046's "
                        "evenly spaced lattice; 'exp' a symmetric geometric "
                        "ladder inside the same +/-2*sigma bound; 'auto' "
                        "measures the choice per z-dimension at "
                        "--fsq-auto-step and records the fitted lattice in "
                        "the checkpoint (both artefacts), which every loader "
                        "rebuilds from and none re-fits.")
    p.add_argument("--fsq-exp-base", type=float, default=2.0)
    p.add_argument("--fsq-auto-n", type=int, default=4096)
    p.add_argument("--fsq-auto-step", type=int, default=2000)
    p.add_argument("--fsq-ladder-fit", default="",
                   help="the per-dimension ladder 'u,e2,...' — normally "
                        "WRITTEN BY the run; pass it to reproduce an exact "
                        "lattice, or let --resume adopt it.")
    p.add_argument("--anomaly", action="store_true")
    p.add_argument("--light-probe-every", type=int, default=0,
                   help="steps between LIGHT probes (linear 26.5N section "
                        "probe). Requires --anomaly; 0 disables. This is the "
                        "cadence the collapse guard's latency tracks.")
    p.add_argument("--eval-every", type=int, default=0,
                   help="REFUSED above 0: the full probe's second half is a "
                        "mini temporal transformer, which is stage-2 work "
                        "and is not ported")
    p.add_argument("--ckpt-every", type=int, default=0,
                   help="milestone checkpoints every N steps (0 = only at the "
                        "end). A long TPU run wants this ON: tpu_train.sh's "
                        "progress watchdog reaps a node whose checkpoint has "
                        "stopped moving.")
    p.add_argument("--resume", default="",
                   help="a .npz written by this trainer (weights + optimiser "
                        "+ step + schedule), or a torch .pt, which WARM-STARTS "
                        "(weights only; torch optimiser state is not mapped "
                        "into optax — JAX_PORT.md §3.3)")
    p.add_argument("--require-resume", action="store_true")
    p.add_argument("--lr-floor", type=float, default=0.0)
    p.add_argument("--lr-decay-steps", type=int, default=0)
    p.add_argument("--max-minutes", type=int, default=0)
    p.add_argument("--warmup-steps", type=int, default=0,
                   help="linear LR warmup. DEFAULT 0 = OFF, because there is "
                        "no warmup on ml/train.py's stage-1 path and the "
                        "default trajectory here is the torch one.")
    p.add_argument("--grad-clip", type=float, default=0.0,
                   help="global-norm gradient clip. DEFAULT 0 = OFF, same "
                        "reason as --warmup-steps.")
    p.add_argument("--bf16", action="store_true",
                   help="MIXED precision: fp32 master weights and optimiser, "
                        "the forward cast to bfloat16. Off by default and "
                        "exercised by no test — a number produced under it "
                        "is a different arithmetic path and must say so.")
    p.add_argument("--shard-batch", default="auto",
                   choices=["auto", "off"],
                   help="data parallelism across local devices: the batch "
                        "axis is sharded and the parameters replicated. "
                        "'auto' is a no-op on one device, which is every CPU "
                        "test.")
    p.add_argument("--collapse-r", type=float, default=0.05,
                   help="ABORT the run when the probe's linear_r_deseas falls "
                        "to or below this on --collapse-strikes consecutive "
                        "probes (0 = off). A codec whose embedding has stopped "
                        "carrying linearly decodable signal is dead, and it "
                        "bills a TPU at exactly the same rate as a live one.")
    p.add_argument("--collapse-strikes", type=int, default=2,
                   help="consecutive sub-threshold probes before aborting; 2 "
                        "so one bad probe cannot kill a healthy run")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--smoke", action="store_true")
    return p.parse_args(argv)


# --------------------------------------------------------------------------
def main(argv=None):
    a = parse(argv)
    if a.resume.startswith("!"):
        a.require_resume, a.resume = True, a.resume[1:]
    if a.smoke:
        a.steps, a.batch = 1500, 256
        for k, v in (("d_z", 32), ("patch", 1), ("d_model", 128),
                     ("n_layers", 4), ("n_heads", 4), ("d_dec", 256)):
            if getattr(a, k) is None:
                setattr(a, k, v)
    if a.eval_every:
        raise SystemExit(
            "--eval-every is REFUSED by the JAX stage-1 trainer. The full "
            "probe's second half trains a mini TemporalTransformer inside the "
            "probe, which is stage-2 machinery this port has not reached "
            "(ml/plans/JAX_PORT.md §4). Running the light probe under the "
            "full probe's flag would put a different estimator into the "
            "archive under a name that already means something. Use "
            "--light-probe-every.")
    if a.light_probe_every and not a.anomaly:
        raise SystemExit("--light-probe-every requires --anomaly (the probe "
                         "measures anomaly-space embeddings; state space is "
                         "disqualified)")
    os.makedirs(a.out, exist_ok=True)
    devices = jax.local_devices()
    print(f"jax {jax.__version__} · devices {[str(dv) for dv in devices]}",
          flush=True)

    # ---- data ------------------------------------------------------------
    d = load_tensor(a.data, allow_pickle=False)
    X, months = d["X"], [str(m) for m in d["months"]]
    if a.anomaly and isinstance(X, np.memmap) and not X.flags.writeable:
        scratch = a.data[:-4] + "_scratch.npy"
        print(f"X is a read-only map ({X.nbytes / 2**30:.1f} GiB) — writable "
              f"scratch copy at {scratch} for the anomaly transform",
              flush=True)
        X = writable_copy(X, scratch)
        import atexit
        atexit.register(lambda: os.path.exists(scratch) and os.remove(scratch))
    lats, lons = d["lats"], d["lons"]
    chan = [str(c) for c in d["chan"]]
    T, H, W, C = X.shape
    print(f"X [T={T} H={H} W={W} C={C}] · channels {chan}", flush=True)

    hold_years = set(a.holdout_years.split(","))
    t_hold = np.array([m[:4] in hold_years for m in months])
    x_hold = lon_holdout_mask(a.holdout_lon, lons)
    ocean = np.isfinite(X[..., 0]).any(axis=0)
    _lonmsg = (f"held-out lon block {int(x_hold.sum())}/{W} cols "
               f"[{a.holdout_lon}]" if x_hold.any() else
               f"NO lon holdout — all {W} cols train (--holdout-lon "
               f"{a.holdout_lon!r})")
    print(f"held-out months {int(t_hold.sum())}/{T} · {_lonmsg} · "
          f"ocean {int(ocean.sum())}", flush=True)
    # The spec is saved verbatim into the exported checkpoint's args, and the
    # twelve eval scripts parse that field with float(). Say so HERE, where the
    # inputs are all it has cost — ml/train.py carries the identical warning
    # for the identical reason, and the exported .pt is read by the identical
    # scripts.
    if not x_hold.any() and a.holdout_lon.strip().lower() in ("", "none"):
        print(f"::warning:: --holdout-lon {a.holdout_lon!r} is saved verbatim "
              f"into this checkpoint's args, and the twelve eval scripts that "
              f"re-read that field parse it with float() — they will all "
              f"raise on it. Use --holdout-lon=0,0 for the identical (empty) "
              f"mask in a form they can read.", flush=True)

    dynamic = None
    if a.anomaly:
        moy = np.array([int(m[5:7]) - 1 for m in months])
        from trainprobe import anomaly_transform     # lazy, as ml/train.py
        X, dynamic = anomaly_transform(X, moy, t_hold, x_hold)
        print(f"anomaly space: {len(dynamic)}/{C} dynamic channels "
              f"({[chan[c] for c in dynamic]})", flush=True)

    # ---- E-047 time blocks (AFTER the anomaly transform, never before) ----
    BLK = None
    if a.time_block:
        BLK = BlockAxis(a.time_block, months,
                        d["bin_index"] if "bin_index" in d else None,
                        (dt.date.fromisoformat(str(d["epoch"]))
                         if "epoch" in d else None),
                        (int(np.asarray(d["pentad_days"]).item())
                         if "pentad_days" in d else None))
        a.k_time = int(BLK.k_max)
        a.ctx_mode = "block_phase"
        print(BLK.describe(C, a.d_z or 0), flush=True)
        blk_rows = BLK.rows.astype(np.int64)
        blk_pad = BLK.pad
        blk_ctx = BLK.ctx_phase()
        blk_hold = np.array([t_hold[BLK.rows[b, :int(BLK.n_bins[b])]].any()
                             for b in range(BLK.n_blocks)])
        print(f"  blocks held out: {int(blk_hold.sum())}/{BLK.n_blocks} "
              f"(any bin of the block held out makes the block held out)",
              flush=True)
    else:
        a.k_time = 1
        a.ctx_mode = "month_sincos"

    obs_any = obs_any_chunked(X)
    tt, yy, xx = pool_idx(obs_any & ~t_hold[:, None, None]
                          & ~x_hold[None, None, :])
    vt_, vy_, vx_ = pool_idx(obs_any & (t_hold[:, None, None]
                                        | x_hold[None, None, :]))
    print(f"train pixels {len(tt):,} · held-out pixels {len(vt_):,}", flush=True)

    Xt = LazyPixels(X)
    OBS = LazyPixels(X, obs=True)
    mvec = np.array([int(m[5:7]) - 1 for m in months])
    ctx_all = np.stack([np.sin(2 * np.pi * mvec / 12),
                        np.cos(2 * np.pi * mvec / 12)], 1)
    if BLK is not None:
        b_any = np.stack([
            obs_any[BLK.rows[b, :int(BLK.n_bins[b])]].any(axis=0)
            for b in range(BLK.n_blocks)])
        tt, yy, xx = pool_idx(b_any & ~blk_hold[:, None, None]
                              & ~x_hold[None, None, :])
        vt_, vy_, vx_ = pool_idx(b_any & (blk_hold[:, None, None]
                                          | x_hold[None, None, :]))
        ctx_all = blk_ctx
        print(f"block pool: train {len(tt):,} · held-out {len(vt_):,} "
              f"(over {BLK.n_blocks} blocks)", flush=True)
    if not len(tt):
        raise SystemExit("REFUSING: the train pool is empty — no (t, y, x) "
                         "with >= 2 observed channels survives the holdouts.")

    rng = np.random.default_rng(a.seed)

    def draw(idx_t, idx_y, idx_x, n):
        k = rng.integers(0, len(idx_t), n)
        t, y, x = idx_t[k], idx_y[k], idx_x[k]
        ctx = np.concatenate([ctx_all[t], (lats[y] / 90)[:, None],
                              (lons[x] / 180)[:, None]], 1)
        return (t.astype(np.int64), y.astype(np.int64), x.astype(np.int64),
                ctx.astype(np.float32))

    # ---- per-channel loss weights (E-019b) -------------------------------
    cw = np.ones(C, np.float32)
    if a.upweight_chans:
        hit = [c for c in range(C) if re.fullmatch(a.upweight_chans, chan[c])]
        if not hit:
            raise SystemExit(f"--upweight-chans {a.upweight_chans!r} matches "
                             f"no channel in {chan} — refusing (a silent "
                             f"no-op here is a fake experiment)")
        cw[hit] = a.upweight
        print(f"upweight ×{a.upweight}: {[chan[c] for c in hit]}", flush=True)

    # ---- architecture resolution -----------------------------------------
    RESUME_ARGS, RESUME_KIND = None, None
    if a.resume:
        if os.path.exists(a.resume):
            if a.resume.endswith(".pt"):
                RESUME_KIND = "pt"
                import torch
                _ck = torch.load(a.resume, map_location="cpu",
                                 weights_only=False)
                RESUME_ARGS = dict(_ck.get("args", {}) or {})
                RESUME_ARGS.setdefault("d_z", _ck.get("d_z"))
            else:
                RESUME_KIND = "npz"
                RESUME_ARGS = json.loads(
                    str(np.load(a.resume, allow_pickle=False)["_args"]))
    if a.require_resume and RESUME_KIND is None:
        raise SystemExit(
            f"--require-resume: no checkpoint at {a.resume!r}. Exiting in "
            f"seconds rather than retraining from scratch for hours under a "
            f"doc string that claims to be a continuation.")
    if RESUME_ARGS is not None:
        adopted, clash = [], []
        for k in ARCH:
            want = RESUME_ARGS.get(k)
            if want is None:
                continue
            have = getattr(a, k)
            explicit = ("--" + k.replace("_", "-")) in (
                sys.argv if argv is None else argv)
            if have is None or not explicit:
                if have != want:
                    setattr(a, k, want)
                    adopted.append(f"{k}={want}")
            elif have != want:
                clash.append(f"    {k}: dispatch says {have}, checkpoint "
                             f"holds {want}")
        kt = RESUME_ARGS.get("k_time")
        if kt is not None and int(kt or 1) != int(a.k_time):
            clash.append(f"    k_time: this run is {a.k_time} "
                         f"(--time-block {a.time_block!r}), checkpoint holds "
                         f"{kt}")
        if clash:
            raise SystemExit(
                "REFUSING to resume: the dispatch contradicts the "
                f"checkpoint's own architecture.\n{chr(10).join(clash)}\n"
                f"  checkpoint: {a.resume}\n"
                "  Either drop the contradicting flags and let the checkpoint "
                "supply them, or resume from a different checkpoint.")
        if adopted:
            print(f"  architecture ADOPTED from "
                  f"{os.path.basename(a.resume)}: " + " ".join(adopted),
                  flush=True)
        # THE FITTED LADDER IS ALWAYS ADOPTED (E-048), never contradicted: a
        # resumed `auto` run must continue with the lattice it already
        # measured rather than measure a second one against a half-trained
        # encoder and change what its own earlier steps meant.
        _fit = str(RESUME_ARGS.get("fsq_ladder_fit", "") or "")
        if _fit and _fit != a.fsq_ladder_fit:
            a.fsq_ladder_fit = _fit
            print(f"  FSQ ladder fit ADOPTED from "
                  f"{os.path.basename(a.resume)} ({_fit.count('e')} of "
                  f"{_fit.count(',') + 1} dimensions exponential) — a resume "
                  f"never re-fits", flush=True)
        # CROSS-TENSOR TRAINING IS REFUSED, cross-tensor EVAL is not
        # (ml/train.py has the identical guard for the identical reason).
        # Continuing a codec on a tensor it was not trained on writes a
        # checkpoint whose provenance is a lie; a pass that TRAINS NOTHING —
        # a checkpoint already at or past --steps — is E-038's frozen control,
        # which is a deliberate and useful thing to do. So the refusal keys on
        # whether anything will train, which is exactly what the stated reason
        # protects.
        ck_data = os.path.basename(str(RESUME_ARGS.get("data", "")))
        if ck_data and ck_data != os.path.basename(a.data):
            ck_step = RESUME_ARGS.get("_step_hint")
            if ck_step is None and RESUME_KIND == "npz":
                ck_step = int(np.load(a.resume, allow_pickle=False)["_step"])
            if ck_step is not None and int(ck_step) >= a.steps:
                print(f"  CROSS-TENSOR EVAL: codec trained on {ck_data}, "
                      f"evaluated on {os.path.basename(a.data)}. No training "
                      f"will occur (checkpoint step {int(ck_step)} >= --steps "
                      f"{a.steps}); the saved artefact is the loaded weights, "
                      f"re-scored.", flush=True)
            else:
                raise SystemExit(
                    f"REFUSING to resume: checkpoint was trained on {ck_data} "
                    f"but this run uses {os.path.basename(a.data)}. "
                    f"Cross-tensor TRAINING would produce a codec whose "
                    f"provenance is a lie. (Eval-only is allowed: pass "
                    f"--steps at or below the checkpoint's recorded step, so "
                    f"nothing trains.)")

    missing = [k for k in ARCH if getattr(a, k) is None]
    if missing:
        raise SystemExit(
            "REFUSING to train: no architecture. Unset: "
            + ", ".join("--" + m.replace("_", "-") for m in missing) + ".\n"
            "  These do not default to the 0.92M pilot, because that default "
            "was never the right answer and omitting a flag used to run a "
            "different experiment in silence.")
    if BLK is not None and a.patch > 1:
        raise SystemExit("--time-block with --patch > 1 is not a shape either "
                         "trainer builds: the block path takes [B,k,C] and "
                         "the patch path [B,C,p²]. ml/train.py has the same "
                         "hole; refusing rather than inventing a fourth "
                         "encoder input layout.")

    # ---- the model -------------------------------------------------------
    model = PixelMAE(n_chan=C, d_z=a.d_z, patch=a.patch, d_model=a.d_model,
                     n_layers=a.n_layers, n_heads=a.n_heads, d_dec=a.d_dec,
                     dec_layers=a.dec_layers, k_time=a.k_time,
                     fsq_levels=a.fsq_levels, fsq_ladder=a.fsq_ladder,
                     fsq_exp_base=a.fsq_exp_base,
                     fsq_ladder_fit=a.fsq_ladder_fit,
                     rngs=nnx.Rngs(a.seed))
    graphdef, state = nnx.split(model)
    n_params = sum(int(np.prod(v.shape)) for v in _leaves(state))
    print(f"codec parameters: {n_params / 1e6:.2f}M", flush=True)
    # THE BOTTLENECK ADDS NO PARAMETERS, so the line above cannot tell a
    # quantized codec from a continuous one and this one has to (E-046's
    # reasoning, E-048's ladder). Nothing prints with the flag off.
    if a.fsq_levels:
        _lv = fql.parse_levels(a.fsq_levels, a.d_z, "--fsq-levels")
        _bits = float(np.mean(np.log2(_lv.astype(float))))
        print(f"FSQ bottleneck: --fsq-levels {a.fsq_levels} -> "
              f"{[int(v) for v in _lv[:8]]}{'...' if a.d_z > 8 else ''} · "
              f"{_bits:.3f} bits/dim x d_z {a.d_z} = "
              f"{float(np.sum(np.log2(_lv.astype(float)))):.1f} bits · ladder "
              f"{a.fsq_ladder}"
              + (f" (base {a.fsq_exp_base:g})" if a.fsq_ladder == "exp"
                 else "")
              + (f" · fitted lattice IN HAND: {a.fsq_ladder_fit}"
                 if a.fsq_ladder_fit else "")
              + (f" · NOT YET FITTED — quantizing uniformly until step "
                 f"{min(a.fsq_auto_step, max(1, a.steps // 2))}"
                 if (a.fsq_ladder == "auto" and not a.fsq_ladder_fit) else "")
              + " · straight-through gradient, no codebook parameters and no "
                "commitment loss", flush=True)

    # ---- optimiser -------------------------------------------------------
    # AdamW(lr, weight_decay=1e-4) — torch's decoupled decay and optax's are
    # the same update (optax adds wd*p to the update and then scales by lr).
    # inject_hyperparams is what makes the --max-minutes REFIT possible: the
    # learning rate is a traced value set from the host each step, so changing
    # the schedule's denominator mid-run needs no re-jit, exactly as
    # ml/train.py's LambdaLR-over-a-mutable-total does.
    tx = optax.inject_hyperparams(optax.adamw)(learning_rate=a.lr,
                                               weight_decay=1e-4)
    if a.grad_clip:
        tx = optax.chain(optax.clip_by_global_norm(a.grad_clip), tx)
    opt_state = tx.init(state)

    BF16 = bool(a.bf16)

    def _cast(tree):
        if not BF16:
            return tree
        return jax.tree_util.tree_map(
            lambda v: v.astype(jnp.bfloat16)
            if v.dtype == jnp.float32 else v, tree)

    # A FACTORY, NOT A PLAIN DEFINITION, and the reason is E-048's `auto`
    # ladder. `jax.jit` caches on argument shapes and dtypes and does NOT
    # notice that a closed-over Python value has changed — so a step function
    # that closed over `graphdef` by name would go on quantizing with the
    # pre-fit lattice after the fit installed a new one, silently, with every
    # log line unchanged. Binding the graphdef at CONSTRUCTION and rebuilding
    # the whole closure when it changes makes that impossible; the cost is one
    # recompilation, once per run.
    def _make_train_step(gd):
        def loss_fn(st, enc_v, enc_o, v, o, mask, ctx, qc, off0, offn, vn, on,
                    cwj, tpos):
            m = nnx.merge(gd, _cast(st))
            l_rec, l_nei = stage1_loss(m, enc_v, enc_o, v, o, mask, ctx, qc,
                                       off0, offn, vn, on, cwj,
                                       a.rec_w_visible, tpos)
            l_rec = l_rec.astype(jnp.float32)
            l_nei = l_nei.astype(jnp.float32)
            return l_rec + 0.5 * l_nei, (l_rec, l_nei)

        @jax.jit
        def train_step(st, ost, lr, enc_v, enc_o, v, o, mask, ctx, qc, off0,
                       offn, vn, on, cwj, tpos):
            (loss, (l_rec, l_nei)), grads = jax.value_and_grad(
                loss_fn, has_aux=True)(st, enc_v, enc_o, v, o, mask, ctx, qc,
                                       off0, offn, vn, on, cwj, tpos)
            # The injected hyperparameter, set from the host, is what carries
            # the schedule — including one whose total was re-fitted mid-run.
            ost = _set_lr(ost, lr)
            upd, ost = tx.update(grads, ost, st)
            return optax.apply_updates(st, upd), ost, loss, l_rec, l_nei
        return train_step

    train_step = _make_train_step(graphdef)

    def _set_lr(ost, lr):
        """Write `lr` into whichever level of the chain owns hyperparams.

        `_replace` rather than mutating `ost.hyperparams` in place: the dict is
        a node of a TRACED pytree, and mutating it under jit would edit an
        object the tracer also holds — the kind of aliasing that works until it
        does not.
        """
        if a.grad_clip:
            inner = ost[1]
            return (ost[0],
                    inner._replace(hyperparams={**inner.hyperparams,
                                                "learning_rate": lr}))
        return ost._replace(hyperparams={**ost.hyperparams,
                                         "learning_rate": lr})

    # ---- device sharding --------------------------------------------------
    # Data parallelism, and nothing cleverer: the batch axis is sharded across
    # the local devices and the parameters are replicated. On one device this
    # whole block is a no-op, which is every CPU test in this repo — the point
    # of writing it this way is that the correctness path does not branch.
    shard = None
    if a.shard_batch == "auto" and len(devices) > 1:
        if a.batch % len(devices):
            raise SystemExit(f"--batch {a.batch} is not divisible by the "
                             f"{len(devices)} local devices; a ragged shard "
                             f"would change the batch each device sees.")
        mesh = jax.make_mesh((len(devices),), ("b",))
        shard = jax.sharding.NamedSharding(mesh,
                                           jax.sharding.PartitionSpec("b"))
        rep = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())
        state = jax.device_put(state, rep)
        opt_state = jax.device_put(opt_state, rep)
        print(f"data-parallel over {len(devices)} devices: batch "
              f"{a.batch} -> {a.batch // len(devices)} per device "
              f"(NOT exercised by the CPU tests)", flush=True)

    def put(x, sharded=True):
        j = jnp.asarray(x)
        return jax.device_put(j, shard) if (shard is not None and sharded) else j

    # ---- batch assembly ---------------------------------------------------
    cw_flat = np.tile(cw, a.k_time) if a.k_time > 1 else cw
    M = C * a.k_time
    qc_np = (np.tile(np.arange(C, dtype=np.int32), (a.batch, a.k_time))
             if a.k_time > 1
             else np.tile(np.arange(C, dtype=np.int32), (a.batch, 1)))
    qt_np = (np.repeat(np.arange(a.k_time, dtype=np.int32), C)[None, :]
             .repeat(a.batch, 0) if a.k_time > 1 else None)
    off0_np = np.zeros((a.batch, M, 3), np.int32)

    def gather(t, y, x):
        return (np.asarray(Xt[t, y, x], np.float32),
                np.asarray(OBS[t, y, x]))

    def gather_block(b, y, x):
        rows = blk_rows[b]                                  # [B, k_max]
        vs, os_ = [], []
        for j in range(rows.shape[1]):
            vs.append(np.asarray(Xt[rows[:, j], y, x], np.float32))
            os_.append(np.asarray(OBS[rows[:, j], y, x]))
        v = np.stack(vs, 1)
        o = np.stack(os_, 1) & (~blk_pad[b])[:, :, None]
        return v, o

    def make_batch(idx_t, idx_y, idx_x, n):
        """Everything one step needs, assembled on the HOST exactly as
        ml/train.py assembles it, then moved across once."""
        t, y, x, ctx = draw(idx_t, idx_y, idx_x, n)
        if BLK is not None:
            v, o = gather_block(t, y, x)
            mask = (rng.random((n, a.k_time, C)) < a.mask_ratio) & o
            enc_v, enc_o = v * (~mask), o
        else:
            v, o = gather(t, y, x)
            mask = (rng.random((n, C)) < a.mask_ratio) & o
            if a.patch > 1:
                from jaxport.embed import gather_px_np
                enc_v, enc_o = gather_px_np(Xt, OBS, t, y, x, a.patch)
                enc_v = enc_v.astype(np.float32)
            else:
                enc_v, enc_o = v * (~mask), o
        pick = rng.integers(0, len(NEI), n)
        dxyz = np.array([NEI[i] for i in pick], np.int64)
        yn = np.clip(y + dxyz[:, 1], 0, H - 1)
        xn = np.clip(x + dxyz[:, 0], 0, W - 1)
        tn = np.clip(t + dxyz[:, 2], 0, len(ctx_all) - 1)
        if BLK is not None:
            vn, on = gather_block(tn, yn, xn)
        else:
            vn, on = gather(tn, yn, xn)
        offn = np.repeat(dxyz[:, None, :].astype(np.int32), M, axis=1)
        return (enc_v, enc_o, v, o, mask, ctx, offn, vn, on)

    cwj = jnp.asarray(cw_flat)
    qc_j = put(qc_np)
    qt_j = put(qt_np) if qt_np is not None else None
    off0_j = put(off0_np)

    # ---- metrics ----------------------------------------------------------
    metrics_path = os.path.join(a.out, "metrics.jsonl")
    loss_every = max(1, a.steps // 200)
    with open(metrics_path, "a") as f:
        f.write(json.dumps({"config": {
            "steps": a.steps, "batch": a.batch, "d_z": a.d_z, "patch": a.patch,
            "d_model": a.d_model, "n_layers": a.n_layers,
            "n_heads": a.n_heads, "d_dec": a.d_dec,
            "anomaly": bool(a.anomaly), "eval_every": a.eval_every,
            "light_probe_every": a.light_probe_every,
            "lr_floor": a.lr_floor, "lr_decay_steps": a.lr_decay_steps,
            "params_M": round(n_params / 1e6, 3),
            "data": os.path.basename(a.data), "C": int(C), "T": int(T),
            "resume": a.resume or None,
            "recipe": os.environ.get("RECIPE_NAME") or None,
            # The two fields a reader of a JAX curve needs and a torch curve
            # never had. ml/CLAUDE.md §3b makes a TPU-trained number a NEW
            # TIER; a config line that did not say which backend produced it
            # would be the one place that fact could go missing.
            "backend": "jax", "k_time": int(a.k_time),
            # E-046/E-048: the bottleneck adds no parameters, so `params_M`
            # above cannot distinguish a quantized codec from a continuous
            # one, or one lattice from another. None = continuous.
            "fsq_levels": (a.fsq_levels or None),
            "fsq_ladder": (a.fsq_ladder if a.fsq_levels else None),
        }}) + "\n")

    ckpt_tag = os.environ.get("CKPT_TAG", "")

    def save_ckpt(step):
        """Both artefacts, every time, because they answer different
        questions: the `.npz` is what THIS trainer resumes from (optimiser
        moments included), and the `.pt` is what the UNCHANGED torch eval
        ladder scores (JAX_PORT.md §1b's cheap validation direction)."""
        args = dict(vars(a))
        args["backend"] = "jax"
        save_state_npz(os.path.join(a.out, "pixelmae_jax.npz"),
                       state, opt_state, step, args)
        m = nnx.merge(graphdef, state)
        export_pt(m, args, path=os.path.join(a.out, "pixelmae.pt"),
                  chan=chan, norm=d["norm"], step=int(step), tag=ckpt_tag)

    # ---- collapse guard ---------------------------------------------------
    strikes = [0]

    def collapse_check(m, step):
        if not a.collapse_r or m is None or step == 0:
            return
        r = m.get("linear_r_deseas")
        if r is None:
            return
        r = float(r)
        # NaN is NOT collapse — it is NO READING. Neither strike nor reset.
        if r != r:
            print(f"  COLLAPSE WATCH: probe returned NaN at step {step} — "
                  f"no reading, strike count held at {strikes[0]}", flush=True)
            return
        if abs(r) > a.collapse_r:
            strikes[0] = 0
            return
        strikes[0] += 1
        print(f"  COLLAPSE WATCH: linear r_des {r:+.3f} at step {step} "
              f"— strike {strikes[0]}/{a.collapse_strikes}", flush=True)
        if strikes[0] < a.collapse_strikes:
            return
        with open(metrics_path, "a") as f:
            f.write(json.dumps({"step": step, "collapsed": {
                "linear_r_deseas": r, "threshold": a.collapse_r,
                "strikes": strikes[0]}}) + "\n")
        raise SystemExit(
            f"ABORTING at step {step}: the probe's linear_r_deseas has been "
            f"<= {a.collapse_r} on {strikes[0]} consecutive probes (last "
            f"{r:+.3f}). The embedding carries no linearly decodable signal — "
            f"this codec is dead and further steps cannot revive it. Pass "
            f"--collapse-r 0 to disable this guard.")

    probe_d = d
    probe_moy, probe_hold = mvec, t_hold
    blk_kw = {}
    if BLK is not None:
        probe_d = {"lats": d["lats"], "lons": d["lons"],
                   "months": np.array(BLK.labels),
                   "rapid": BLK.remap_rows(d["rapid"]),
                   "_blk_ctx": blk_ctx}
        probe_moy = np.array([int(m[5:7]) - 1 for m in BLK.labels])
        probe_hold = blk_hold
        blk_kw = {"blk_rows": blk_rows, "blk_pad": blk_pad}

    def run_probe(step):
        """A probe is INSTRUMENTATION and must never be the thing that loses a
        training job — any exception becomes a `probe_error` record and the
        run carries on (#56-#59)."""
        try:
            m = light_probe(nnx.merge(graphdef, state), Xt, OBS, probe_d,
                            probe_moy, probe_hold, x_hold, ocean=ocean,
                            **blk_kw)
        except Exception as e:
            import traceback
            traceback.print_exc()
            print(f"  probe @{step} FAILED ({type(e).__name__}: {e}) — "
                  f"training continues, no probe point this interval",
                  flush=True)
            with open(metrics_path, "a") as f:
                f.write(json.dumps({
                    "step": step, "wall_s": round(time.time() - t0, 1),
                    "probe_error": f"{type(e).__name__}: {e}"[:200]}) + "\n")
            return None
        m["step"] = step
        m["wall_s"] = round(time.time() - t0, 1)
        with open(metrics_path, "a") as f:
            f.write(json.dumps(m) + "\n")
        collapse_check(m, step)
        return m

    # ---- resume -----------------------------------------------------------
    s = 0
    if RESUME_KIND == "npz":
        state, opt_state, s, _ = load_state_npz(a.resume, state, opt_state)
        print(f"  RESUMED from {a.resume} at step {s} (optimizer + schedule "
              f"restored); training on to {a.steps}", flush=True)
        with open(metrics_path, "a") as f:
            f.write(json.dumps({"resumed": {
                "from": os.path.basename(a.resume),
                "parent_tag": "", "at_step": s}}) + "\n")
    elif RESUME_KIND == "pt":
        import torch
        ck = torch.load(a.resume, map_location="cpu", weights_only=False)
        m = nnx.merge(graphdef, state)
        load_pixelmae(ck["model"], m)
        _, state = nnx.split(m)
        opt_state = tx.init(state)
        print(f"  WARM-STARTED from {a.resume}: weights only, no optimizer or "
              f"step — the LR schedule restarts from 0. Report this run as a "
              f"warm start, not a continuation. (Mapping torch Adam state "
              f"into optax is out of scope, JAX_PORT.md §3.3.)", flush=True)
    elif a.resume:
        print(f"  --resume {a.resume}: NOT FOUND — starting from scratch "
              f"(this is not an error, but the run is now a fresh one; say so "
              f"in its doc string)", flush=True)

    # ---- train ------------------------------------------------------------
    print("training …", flush=True)
    t0 = time.time()
    if a.light_probe_every and not a.resume:
        m0 = run_probe(0)
        if m0:
            print(f"  step-0 probe (UNTRAINED codec): linear r_des "
                  f"{m0['linear_r_deseas']:+.3f} — every later probe should "
                  f"be read as a change from this", flush=True)
    elif a.resume:
        print("  (no step-0 probe: this run resumes, so there is no untrained "
              "baseline to measure)", flush=True)

    sched_total = (a.lr_decay_steps if (a.lr_floor > 0 and a.lr_decay_steps)
                   else a.steps)
    CAL, t1, next_cal, steps0 = 200, None, None, a.steps

    # ---- E-048: the --fsq-ladder auto FIT ---------------------------------
    # The mirror of ml/train.py's, driven from the loop for the same reason:
    # what the fit measured is then a function of the run's own seed and step
    # schedule and of nothing else. One deliberate eager `encode_pre` over a
    # fresh TRAIN-pool draw, the shared per-dimension fit out of
    # ml/fsq_ladder.py, then the new lattice is installed on the model, the
    # graphdef is re-split and the jitted step is REBUILT (see
    # `_make_train_step` for why rebuilding is not optional).
    FSQ_FIT_AT = (min(a.fsq_auto_step, max(1, a.steps // 2))
                  if (a.fsq_levels and a.fsq_ladder == "auto"
                      and not a.fsq_ladder_fit) else None)

    def fsq_auto_fit(step, gd, st):
        n = max(2, int(a.fsq_auto_n))
        t, y, x, ctx = draw(tt, yy, xx, n)
        if BLK is not None:
            ev, eo = gather_block(t, y, x)
            mk = np.zeros_like(eo)
        elif a.patch > 1:
            from jaxport.embed import gather_px_np
            ev, eo = gather_px_np(Xt, OBS, t, y, x, a.patch)
            ev = ev.astype(np.float32)
            mk = np.zeros((n, C), bool)
        else:
            ev, eo = gather(t, y, x)
            mk = np.zeros_like(eo)
        m = nnx.merge(gd, st)
        pre = np.asarray(m.encode_pre(jnp.asarray(ev), jnp.asarray(eo),
                                      jnp.asarray(mk), jnp.asarray(ctx)))
        lv = fql.parse_levels(a.fsq_levels, a.d_z, "--fsq-levels")
        is_exp, base, mse_u, mse_b = fql.fit_auto(pre.astype(np.float64), lv,
                                                  2.0)
        spec = fql.format_fit(is_exp, base)
        a.fsq_ladder_fit = spec
        m.set_fsq_ladder(is_exp, base, spec)
        gd2, st2 = nnx.split(m)
        print(f"  step {step}: --fsq-levels auto fitted on {len(pre):,} "
              f"pre-quantization vectors: "
              f"{fql.describe_fit(is_exp, base, mse_u, mse_b)}", flush=True)
        with open(metrics_path, "a") as f:
            f.write(json.dumps({"step": int(step), "fsq_ladder_fit": {
                "spec": spec, "n": int(len(pre)),
                "n_exp": int(np.asarray(is_exp).sum()),
                "d_z": int(a.d_z)}}) + "\n")
        return gd2, st2

    while s < a.steps:
        s += 1
        if FSQ_FIT_AT is not None and s == FSQ_FIT_AT:
            graphdef, state = fsq_auto_fit(s, graphdef, state)
            train_step = _make_train_step(graphdef)
        bt = make_batch(tt, yy, xx, a.batch)
        (enc_v, enc_o, v, o, mask, ctx, offn, vn, on) = bt
        lr = a.lr * lr_factor(s - 1, a.lr_floor, sched_total, a.warmup_steps)
        state, opt_state, loss, l_rec, l_nei = train_step(
            state, opt_state, jnp.asarray(lr, jnp.float32),
            put(enc_v), put(enc_o), put(v), put(o), put(mask), put(ctx),
            qc_j, off0_j, put(offn), put(vn), put(on), cwj, qt_j)
        lr_v, lrec_v, lnei_v = float(loss), float(l_rec), float(l_nei)
        if not np.isfinite(lr_v):
            with open(metrics_path, "a") as f:
                f.write(json.dumps({"step": s, "diverged": {
                    "loss_rec": lrec_v, "loss_nei": lnei_v}}) + "\n")
            raise SystemExit(
                f"ABORTING at step {s}: loss is {lr_v} (rec {lrec_v}, nei "
                f"{lnei_v}). The model has gone non-finite; every further step "
                f"writes NaN into the weights. Suspect the learning rate "
                f"first (there is no warmup and no gradient clipping on this "
                f"path unless --warmup-steps/--grad-clip were passed).")
        if s == 1:
            t1 = time.time()
        if (a.max_minutes and a.lr_floor == 0 and t1 is not None and s >= 4
                and (s >= CAL or time.time() - t1 > 60)
                and (next_cal is None or s >= next_cal)):
            now = time.time()
            fit, rate = fit_schedule(s, now - t1, now - t0, a.max_minutes,
                                     steps0)
            next_cal = s + max(25, min(2000, (fit - s) // 4))
            if abs(fit - a.steps) > max(50, a.steps // 20):
                print(f"  time budget: {rate:.2f} s/step steady (n={s - 1}) → "
                      f"re-fitting the cosine schedule from {a.steps} to "
                      f"{fit} steps so the LR anneals to zero inside "
                      f"{a.max_minutes} min", flush=True)
                a.steps = fit
                sched_total = fit
        if a.max_minutes and (time.time() - t0) > a.max_minutes * 60:
            print(f"  wall-clock budget reached at step {s} — stopping to save",
                  flush=True)
            break
        if s % loss_every == 0 or s == a.steps:
            with open(metrics_path, "a") as f:
                f.write(json.dumps({"step": s,
                                    "loss_rec": round(lrec_v, 5),
                                    "loss_nei": round(lnei_v, 5)}) + "\n")
        if s % max(1, a.steps // 10) == 0:
            print(f"  step {s:>6}/{a.steps}  rec {lrec_v:.4f}  "
                  f"nei {lnei_v:.4f}  ({time.time() - t0:.0f}s)", flush=True)
        if a.light_probe_every and s % a.light_probe_every == 0:
            m = run_probe(s)
            if m:
                print(f"  light probe @{s}: linear r_des "
                      f"{m['linear_r_deseas']:+.3f} "
                      f"({m['probe_seconds']:.0f}s)", flush=True)
        if a.ckpt_every and s % a.ckpt_every == 0:
            save_ckpt(s)

    # ---- the blocked-holdout evaluation -----------------------------------
    results = {}
    if BLK is not None:
        # E-047: the per-bin evaluation does not apply to a block codec — its
        # recon rows, its t+1 persistence and its channel skills are all
        # per-bin quantities. Skipped LOUDLY and in the artefact, never by
        # producing a number that answers a question nobody asked.
        results["eval_skipped"] = (
            f"--time-block {a.time_block!r}: the codec's own per-bin "
            f"evaluation does not apply to a block codec. The verdict for a "
            f"block codec is the stage-2 head and the roll.")
        print(f"::warning::{results['eval_skipped']}", flush=True)
    else:
        results.update(final_eval(nnx.merge(graphdef, state), a, C, T, chan,
                                  Xt, OBS, X, lats, lons, ctx_all, t_hold, d,
                                  vt_, vy_, vx_, gather))
    json.dump(results, open(os.path.join(a.out, "eval.json"), "w"), indent=2)
    print(json.dumps(results, indent=2), flush=True)
    save_ckpt(a.steps if s >= a.steps else s)
    print(f"saved {os.path.join(a.out, 'pixelmae.pt')} and pixelmae_jax.npz",
          flush=True)
    return 0


def final_eval(model, a, C, T, chan, Xt, OBS, X, lats, lons, ctx_all, t_hold,
               d, vt_, vy_, vx_, gather):
    """`ml/train.py`'s end-of-run evaluation on the BLOCKED holdout.

    Three numbers, the same three: masked-channel reconstruction skill per
    channel against the channel mean, the t+1 neighbour prediction against
    persistence, and the RAPID ridge from the section's mean embedding. It is
    chunked at 2,048 for the same reason the torch one is — encoding 20,000
    held-out pixels in one forward is the allocation that OOM-killed #62/#63
    AFTER they had trained every step.
    """
    from jaxport.embed import gather_px_np
    rng = np.random.default_rng(a.seed + 1)
    out = {}
    n_eval = min(20000, len(vt_))
    if n_eval < 100:
        out["eval_skipped"] = (f"only {n_eval} held-out pixels — too few to "
                               f"score anything")
        return out
    k = rng.integers(0, len(vt_), n_eval)
    t, y, x = (vt_[k].astype(np.int64), vy_[k].astype(np.int64),
               vx_[k].astype(np.int64))
    ctx = np.concatenate([ctx_all[t], (lats[y] / 90)[:, None],
                          (lons[x] / 180)[:, None]], 1).astype(np.float32)
    v, o = gather(t, y, x)
    mask = (rng.random((n_eval, C)) < a.mask_ratio) & o

    EV_CH = 2048
    zs = []
    for i in range(0, n_eval, EV_CH):
        sl = slice(i, min(i + EV_CH, n_eval))
        if a.patch > 1:
            vp, op = gather_px_np(Xt, OBS, t[sl], y[sl], x[sl], a.patch)
            zs.append(np.asarray(model.encode(
                jnp.asarray(vp, jnp.float32), jnp.asarray(op),
                jnp.asarray(mask[sl]), jnp.asarray(ctx[sl]))))
        else:
            zs.append(np.asarray(model.encode(
                jnp.asarray(v[sl] * (~mask[sl])), jnp.asarray(o[sl]),
                jnp.asarray(mask[sl]), jnp.asarray(ctx[sl]))))
    z = np.concatenate(zs)
    qc = np.tile(np.arange(C, dtype=np.int32), (n_eval, 1))
    preds, p1s = [], []
    for i in range(0, n_eval, EV_CH):
        sl = slice(i, min(i + EV_CH, n_eval))
        nb = sl.stop - sl.start
        off = np.zeros((nb, C, 3), np.int32)
        preds.append(np.asarray(model.query(jnp.asarray(z[sl]),
                                            jnp.asarray(qc[sl]),
                                            jnp.asarray(off))))
        off1 = off.copy()
        off1[:, :, 2] = 1
        p1s.append(np.asarray(model.query(jnp.asarray(z[sl]),
                                          jnp.asarray(qc[sl]),
                                          jnp.asarray(off1))))
    pred = np.concatenate(preds)
    p1 = np.concatenate(p1s)

    for c, name in enumerate(chan):
        m = mask[:, c]
        if m.sum() < 50:
            continue
        err = float(np.mean((pred[m, c] - v[m, c]) ** 2))
        base = float(np.mean(v[m, c] ** 2))
        out[f"recon/{name}"] = {"mse": err, "mse_channel_mean": base,
                                "skill": 1 - err / max(base, 1e-9)}

    tn = np.clip(t + 1, 0, T - 1)
    v1, o1 = gather(tn, y, x)
    both = (o & o1)
    denom = max(int(both.sum()), 1)
    out["t+1"] = {
        "mse_model": float(((p1 - v1) ** 2 * both).sum() / denom),
        "mse_persistence": float(((v - v1) ** 2 * both).sum() / denom)}
    out["t+1"]["beats_persistence"] = bool(
        out["t+1"]["mse_model"] < out["t+1"]["mse_persistence"])

    rapid = d["rapid"]
    if len(rapid):
        from temporal import RAPID_LON
        sec_y = int(np.argmin(np.abs(lats - 26.5)))
        sec_x = np.where(np.isfinite(np.asarray(X[0, sec_y, :, 0]))
                         & (lons >= RAPID_LON[0]) & (lons <= RAPID_LON[1]))[0]
        n = len(sec_x)
        if n:
            emb = np.zeros((T, a.d_z), np.float32)
            for tix in range(T):
                cx = np.concatenate([np.tile(ctx_all[tix], (n, 1)),
                                     (np.full(n, lats[sec_y]) / 90)[:, None],
                                     (lons[sec_x] / 180)[:, None]],
                                    1).astype(np.float32)
                ti = np.full(n, tix, np.int64)
                yi = np.full(n, sec_y, np.int64)
                if a.patch > 1:
                    vv, oo = gather_px_np(Xt, OBS, ti, yi, sec_x, a.patch)
                    vv = vv.astype(np.float32)
                else:
                    vv, oo = gather(ti, yi, sec_x.astype(np.int64))
                zz = model.encode(jnp.asarray(vv), jnp.asarray(oo),
                                  jnp.zeros((n, C), bool), jnp.asarray(cx))
                emb[tix] = np.asarray(zz).mean(0)
            ridx = rapid[:, 0].astype(int)
            rv = rapid[:, 1]
            tr = ~t_hold[ridx]
            te = t_hold[ridx]
            if te.sum() >= 12:
                A = np.c_[emb[ridx], np.ones(len(ridx))]
                lam = 1e-2 * np.eye(A.shape[1])
                lam[-1, -1] = 0
                wgt = np.linalg.solve(A[tr].T @ A[tr] + lam, A[tr].T @ rv[tr])
                pr = A @ wgt
                out["rapid_probe"] = {
                    "pearson_train": float(np.corrcoef(pr[tr], rv[tr])[0, 1]),
                    "pearson_heldout_years": float(
                        np.corrcoef(pr[te], rv[te])[0, 1]),
                    "n_train": int(tr.sum()), "n_test": int(te.sum())}
    return out


if __name__ == "__main__":
    sys.exit(main())

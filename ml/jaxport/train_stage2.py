#!/usr/bin/env python3
"""The STAGE-2 HEAD TRAINER in JAX/optax — tier 3b of `ml/plans/JAX_PORT.md`.

This mirrors `ml/temporal.py`'s stage-2 training path and nothing else. Its
sibling `train_stage1.py` ported the codec; this ports the forecaster that
reads the codec's embeddings and predicts the next one.

WHY IT EXISTS, in the terms that decide whether it was worth building.
Chris's span-fixed ladder (E-045.x) holds the context SPAN at two years and
lets the frame count grow as the step shrinks — `--K 48`, `--K 72`, `--K 144`.
A causal transformer's attention is quadratic in K, so stage-2 compute on that
ladder is ~K²-heavy, and the measured TPU advantage (4.5–8× on the stage-1
work) applies exactly there. This port is what lets the K-heavy rungs and the
seed replicates §3b prices run on a v5e instead of queueing behind rented
GPUs. First use: an E-045.1-class arm (K 144, stencil 145, 1024×16 over the
pentad codec) re-run on TPU as the CROSS-FRAMEWORK TWIN of the torch arm.

**A NUMBER THIS FILE PRODUCES IS A NEW TIER.** `ml/CLAUDE.md` §3b: a
JAX-trained result is never pooled with the torch/GPU record, and the first
result at the tier buys its own replication. The `stage2_config` record says
`"backend": "jax"` so that fact cannot go missing from the one line a reader
sees.

WHAT IS MIRRORED, TERM FOR TERM
  · the WINDOW POOL — `ok_t` (a window ends at t, its target t+1 exists, is a
    train bin, and t+1 >= K) intersected with `ok_p` (`~pool_x_hold[xs]`), and
    the two-masks rule that goes with it: `stat_x_hold` (the codec's own
    `holdout_lon`) chooses the anomaly transform's statistics and
    `--train-lon-hold` chooses the POOL, never the other way round;
  · the objective — `((pred - ztgt)**2).mean()` over the whole window, which
    is `ml/temporal.py:2437`'s `l_base` and, at unroll 1 with no direct heads,
    its entire loss;
  · `--input-znoise` — an ABSOLUTE sigma added to LIVE slots only (a slot is
    live iff any of its d_z components is non-zero; zero IS the dead-slot
    encoding and the roll feeds exact zeros there);
  · `--grad-clip` — clip by GLOBAL norm between the gradient and the update,
    **OFF at 0 and OFF MAKES NO CALL AT ALL**, plus the window statistics
    (`stage2_grad_norm_max`, `_clip_frac`, `_nonfinite`) that distinguish
    "healthy, never binds" from "the clip is now setting the learning rate";
  · the four schedules — `cosine`, `invsqrt`, `wsd`, `expdecay` — transcribed
    from `ml/temporal.py:make_sched`'s own closures, including
    `--lr-cooldown-frac`'s terminal taper, because the fleet's stage-2 config
    is `expdecay` with `--lr-halflife 40000`;
  · `--K` with `k_max = K` for a fresh head (the positional table is exactly
    as long as the window, which is what `rollout_spatial` reads back out of
    `pos.weight`);
  · `--time-stride` / `--time-offset`, subsampling EVERY array keyed on the
    axis together, and REFUSED on a block codec exactly as `ml/temporal.py`
    refuses it (two time surgeries at once leave an axis no artefact
    describes);
  · the BLOCK-Z axis adoption — `k_time`/`time_block` come from the CODEC's
    own args, the axis becomes the block axis, `t_hold` fuses (any held-out
    bin makes the block held out) and the RAPID rows are remapped, all through
    `ml/timeblocks.py:BlockAxis` itself;
  · milestone checkpoints, resume, and `metrics.jsonl` with THE SAME KEYS, so
    the status page and every existing reader work on a JAX run unchanged;
  · the in-training pooled rapid probe (`stage2_probe.rapid_r_deseas`,
    `ml/temporal.py:2181-class`) — the monitor Chris reads — through the same
    imported `ridge_r`;
  · **E-057's FGN head** — `--fgn-eps k` builds the ε-conditioned FiLM head
    (`jaxport.models`) and SWITCHES the objective to the fair CRPS at N=2
    (two forwards per step on the identical context, two independent ε);
    `--fgn-val-members M` drives the CHUNKED M-member monitor read that
    writes `stage2_val_crps` / `stage2_val_member_var` /
    `stage2_val_spread_ratio` under the names `ml/temporal.py` uses, so
    status.html needs no new record family. `--fgn-eps 0` is the exact legacy
    path — no film parameter, no extra draw, MSE — and the flags REFUSE under
    it. The ε stream is a pure fold of (seed, step, forward index): JAX's
    PRNG is counter-based, so resume-exactness is the fold arithmetic and NO
    RNG state is checkpointed (see `fgn_eps_at`). What is NOT claimed:
    cross-backend ε-stream equality — different RNG families
    (`ml/plans/FGN_JAX_PORT.md` §1).

WHAT IS NOT, each named here so nothing is silently missing:
  · **`--input-quant` REFUSES.** This is the KNOWN GAP. The A-arm parity
    comparison on quantized inputs waits for it; running an arm that names
    `--input-quant` under a trainer that ignores it would put a
    continuous-input head into the archive under a quantized arm's name.
  · **`--unroll`>1, `--unroll-wide`, `--unroll-probs` and `--direct` REFUSE.**
    Each feeds predictions back and each is its own experiment; E-010/E-020
    closed the unroll axis and every arm this port is being built for is U=1.
  · **the full probe ladder** (`probe_kfold` / `probe_head`) and the ROLL are
    not here. They are eval, they run unchanged under torch on the exported
    head, and that is `JAX_PORT.md` §1b's cheap validation direction: a
    TPU-trained head is scored by the UNCHANGED torch ladder, never by a
    second scoreboard. `rapid_probe_kfold` is therefore absent from the
    results file rather than present and empty.
  · **joint training** (`ml/train_joint.py`) — a different objective.
  · **`--target-bins-argo` and `--season-dropout`** — E-044c pool/regulariser
    knobs, off in every arm this port targets; they refuse rather than
    silently defaulting to a different pool than the flag names.

**`--grad-accum N` — THE ONE KNOB HERE `ml/temporal.py` DOES NOT HAVE, AND IT
IS NOT AN EXPERIMENT.** E-054b (2026-08-28 00:15Z) registered the HBM risk and
it fired: the 400M rung (1280x20, K 144, batch 256, 399.948M params) asked for
5.09 G with 4.03 G free inside `train_step` on a v5e-4 chip and died at the
first step. Of the four recorded options only one preserves the comparison
with E-051 exactly — accumulate the gradient over N micro-batches of
`batch/N` and take ONE AdamW step on their AVERAGE. That is not an
approximation of a batch-256 step, it IS a batch-256 step: the objective is a
MEAN over the window elements, every micro-batch has the same element count,
so mean-of-means is the batch mean and the average of the micro gradients is
the batch gradient — exactly, in real arithmetic, with only float ASSOCIATION
separating the two on a machine (`tests/test_grad_accum_jax.py` measures it).
Everything a reader sees is therefore unchanged: a "step" is one optimiser
update over the EFFECTIVE batch, `stage2_zmse` is the mean over the effective
batch, `stage2_grad_norm` is the pre-clip norm OF THE AVERAGED gradient (so
`--grad-clip` binds on the same quantity at any N), and `stage2_config` gains
`grad_accum` so no reader has to infer which one ran. **N = 1 is not a code
path, it is the ABSENCE of one** — the accumulating graphs are not even built,
so an unaccumulated run's jaxpr is the pre-E-054b one op for op.

THE Z IT READS. Either a published cache (`--z`: the `[T, P, d_z]` float16
`.npy` with a 128-byte header that `embed_cache_sync` publishes, named by
`Z_<codec weight hash>_<tensor sha>.npy`) or, with `--z` absent, an embedding
built on the spot through `jaxport.embed.embed_everything_jax`. Both routes
need the codec `.pt` anyway — for the architecture, for `holdout_lon`, and for
the per-pixel STATIC IDENTITY embedding, which is a codec forward over the
static channels alone and is part of every window's input.

RNG. Window draws, the noise draw and the monitor's fixed sample come from
seeded `np.random.default_rng`s. A torch run's stream cannot be reproduced in
JAX, so bit-identical TRAJECTORIES are impossible by construction and are not
claimed; what IS claimed and gated is that on the SAME weights, the SAME
window batch and the SAME noise array the two frameworks compute the same loss
to 1e-5 and take the same plain-SGD step to 1e-6 (`tests/test_jaxport_train_
s2.py`, G5a/G5b).

WHY THIS FILE IMPORTS FROM THE TORCH TREE — the same reason `train_stage1.py`
does, and `ml/CLAUDE.md`'s standing rule: shared numpy plumbing is IMPORTED,
never copied. `build_stencil`, `rapid_section`, `season_ctx`, `ridge_r`,
`lon_holdout_mask`, `anomaly_transform`, `LazyPixels`, `BlockAxis` and
`_ring_on` are all pure numpy but live in modules that import torch at their
top, so this DRIVER needs a CPU torch wheel. `jaxport.models` and
`jaxport.convert` themselves still import no torch at module scope.

    # toy, CPU, ~1 minute
    python3 ml/jaxport/train_stage2.py --data toy.npz --ckpt codec.pt \\
        --out /tmp/s2 --steps 300 --batch 32 --K 6 --d-model 32 --layers 2
"""
import argparse
import datetime as dt
import functools
import json
import math
import os
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

from jaxport.models import TemporalTransformer                  # noqa: E402
from jaxport.convert import (codec_from_ckpt_jax, export_temporal_pt,  # noqa: E402,E501
                             load_pixelmae, load_temporal)
from jaxport.embed import embed_everything_jax, gather_px_np     # noqa: E402

# The numpy plumbing, imported rather than copied — see the module docstring.
from model import LazyPixels                                    # noqa: E402
from temporal import (CACHE_DTYPE, STENCILS, _ring_on, build_stencil,  # noqa: E402,E501
                      rapid_section, season_ctx)
from timeblocks import BlockAxis                                # noqa: E402
from train import lon_holdout_mask                              # noqa: E402
from probe_sequence import ridge_r                              # noqa: E402
from tensor_io import load_tensor, writable_copy                # noqa: E402


# --------------------------------------------------------------------------
# the batch gather — `ml/temporal.py:gather_stencil` (:395) in numpy
# --------------------------------------------------------------------------
def gather_stencil_np(Z, base, p, NBR, K, cast32=True, workers=0):
    """The ONE window-input gather, mirroring `temporal.gather_stencil`.

    NOT a copy of shared plumbing: the original is written in torch against a
    torch `Zt`, and this package's whole data path is numpy (`ml/CLAUDE.md`'s
    rule is about the numpy helpers, and this is the torch half of the file).
    `tests/test_jaxport_train_s2.py` pins it elementwise against
    `temporal.gather_stencil` itself, which is what makes it a transcription
    rather than a second definition.

    Z [T,P,d_z] (float16 memmap or array) · base [n] window-start rows · p [n]
    centre pixels · NBR None (stencil 1 → the exact legacy gather) or [P,S]
    int64 with -1 = missing. Returns [n,K,d_z] or [n,K,S*d_z]; missing
    neighbours are zero-filled, and slot 0 is the centre.

    `cast32=True` (the default, and what every existing caller gets) returns
    float32. `cast32=False` returns Z's OWN dtype uncast — at K=144 the fp32
    cast alone was measured at ~1 s/step of host time (2026-08-25, the E-051
    launch), and a device-side cast is free; the VALUES are identical because
    fp16→fp32 is exact. `workers>0` splits the K loop across a thread pool —
    same arithmetic, same output, measured ~1.9x at 8 threads (the GIL keeps
    the rest). Both knobs exist for the TPU input pipeline; the parity gates
    pin the default path.
    """
    base = np.asarray(base)
    p = np.asarray(p)
    if NBR is None:
        return np.stack([np.asarray(Z[base + j, p], np.float32)
                         for j in range(K)], 1)
    nbr = NBR[p]                                        # [n, S]
    miss = nbr < 0
    safe = np.maximum(nbr, 0)
    n = len(base)

    def _cols(jlo, jhi):
        cols = []
        for j in range(jlo, jhi):
            zj = Z[(base + j)[:, None], safe]           # [n,S,d_z] copy
            zj = np.asarray(zj, np.float32) if cast32 else np.asarray(zj)
            zj[miss] = 0.0
            cols.append(zj.reshape(n, -1))
        return cols

    if workers and workers > 1 and K > 1:
        from concurrent.futures import ThreadPoolExecutor
        W = min(int(workers), K)
        bounds = [(i * K // W, (i + 1) * K // W) for i in range(W)]
        with ThreadPoolExecutor(W) as ex:
            parts = list(ex.map(lambda b: _cols(*b), bounds))
        cols = [c for part in parts for c in part]
    else:
        cols = _cols(0, K)
    return np.stack(cols, 1)


# --------------------------------------------------------------------------
# the objective and its two knobs
# --------------------------------------------------------------------------
def apply_znoise(zseq, noise, sigma, d_z):
    """`ml/temporal.py:2425-2434` — E-029b's input noise, exactly.

    `noise` is a standard-normal array of the SAME shape as `zseq` and is
    passed IN rather than drawn here. That is deliberate and is what makes
    G5a possible: an RNG cannot be shared across frameworks, an array can, so
    the parity gate feeds torch and JAX the identical perturbation and the
    comparison is of the arithmetic instead of of two samplers.

    DEAD SLOTS STAY EXACT ZEROS. A slot is live iff any of its d_z components
    is non-zero — zero IS the dead-slot encoding (`test_zero_weight_
    equivalence`) and the roll feeds exact zeros there too, so noising them
    would train against an input state the roll never produces.
    """
    if sigma <= 0 or noise is None:
        return zseq
    n, K = zseq.shape[0], zseq.shape[1]
    z4 = zseq.reshape(n, K, -1, d_z)                    # [n,K,S,d_z]
    live = (z4 != 0).any(axis=-1, keepdims=True)
    return (z4 + noise.reshape(z4.shape) * sigma * live).reshape(zseq.shape)


def apply_znoise_jax(zseq, key, sigma, d_z):
    """`apply_znoise` with the draw ON DEVICE — the TPU input-pipeline fix.

    The host draw was measured at ~15 s/step at K=144 (256x144x145x32
    standard-normals through numpy's single-threaded PCG64, 2026-08-25 —
    the number that killed the first E-051 training node's schedule); the
    same draw inside the jitted step is microseconds of TPU time. The
    SEMANTICS are apply_znoise's exactly — iid N(0,1) times sigma on live
    slots, dead slots kept at exact zeros — but the STREAM is jax.random's
    threefry, not numpy's PCG64, so a device-noise run is a different sample
    path from a host-noise run at the same seed. That is a fresh-run fact to
    record, never a parity bug: G5a compares the two frameworks on an
    INJECTED noise array precisely so no gate depends on any sampler.

    Traceable end to end (shapes static under jit). zseq may arrive fp16
    (see gather_stencil_np cast32=False); it is cast to fp32 HERE, which is
    exact, so the fp16 transfer changes no value anywhere.
    """
    zseq = zseq.astype(jnp.float32)
    if sigma <= 0:
        return zseq
    n, K = zseq.shape[0], zseq.shape[1]
    z4 = zseq.reshape(n, K, -1, d_z)
    live = (z4 != 0).any(axis=-1, keepdims=True)
    noise = jax.random.normal(key, z4.shape, jnp.float32)
    return (z4 + noise * sigma * live).reshape(zseq.shape)


# --------------------------------------------------------------------------
# E-057 · the FGN objective, the ε stream and the ensemble read-out
# --------------------------------------------------------------------------
FGN_VAL_MEMBERS_DEFAULT = 8       # the argparse default, named so the
                                  # refuse-under-MSE guard can tell "the user
                                  # asked for an ensemble" from "nobody did".


def fair_crps2(x1, x2, y):
    """The FAIR CRPS estimator at M=2 — the JAX twin of
    `ml/temporal.py:fair_crps2` (:124-142).

        fair_crps2 = mean[ 0.5(|x1-y| + |x2-y|) - 0.5|x1-x2| ]

    THE ARITHMETIC IS TORCH'S, TERM FOR TERM, and the association matters.
    `ml/temporal.py:141-142` reads

        return (0.5 * ((x1 - y).abs() + (x2 - y).abs())
                - 0.5 * (x1 - x2).abs()).mean()

    i.e. the three terms are combined ELEMENTWISE and reduced ONCE. Writing it
    as `mean|x1-y|/2 + mean|x2-y|/2 - mean|x1-x2|/2` is the same number in
    exact arithmetic and a DIFFERENT one in float32 (three reductions instead
    of one, over three different summands), which is precisely the kind of
    silent 1e-6 that a parity gate at 1e-6 would then have to be widened for.
    So the transcription keeps torch's shape.

    The second term's divisor in the general fair estimator is
    2·M·(M-1) = 4 over the two ordered pairs (i,j) and (j,i), each
    contributing |x1-x2| — hence |x1-x2|/2.

    Why the objective must be this and not MSE: under a squared-error loss the
    conditional MEAN is optimal, so a noise-conditioned head learns to IGNORE
    ε. Noise conditioning and the proper score are ONE change, not two.
    """
    return (0.5 * (jnp.abs(x1 - y) + jnp.abs(x2 - y))
            - 0.5 * jnp.abs(x1 - x2)).mean()


def fair_crps_ens(ens, obs):
    """Fair CRPS of an M-member ensemble `ens` [M, ...] against `obs` [...] —
    the JAX twin of `ml/temporal.py:fair_crps_ens` (:145-174), including its
    sorted-member identity

        Σ_{i,j} |x_i - x_j| = 2 Σ_k (2k - M + 1) x_(k)      (x_(k) ascending)

    so the read-out is O(M log M) rather than the O(M²) pairwise tensor.

    **M = 1 is MAE, exactly** — the fair divisor M(M-1) is zero there, so the
    pair term is dropped and only mean|x-y| remains. That is what lets a
    deterministic head enter the same scoreboard as a degenerate one-member
    ensemble with no special case anywhere.
    """
    M = ens.shape[0]
    term1 = jnp.abs(ens - obs).mean(0)
    if M < 2:
        return term1.mean()
    xs = jnp.sort(ens, axis=0)
    k = jnp.arange(M, dtype=ens.dtype).reshape((-1,) + (1,) * (ens.ndim - 1))
    w = 2.0 * k - M + 1.0
    pair_sum = 2.0 * (w * xs).sum(0)
    return (term1 - 0.5 * pair_sum / (M * (M - 1.0))).mean()


def fgn_eval_eps(n, eps_dim):
    """THE representative member, in ONE place — `ml/temporal.py:fgn_eval_eps`
    (:177-195). ε = 0, the centre of the noise distribution: at init (zero
    film) it is exactly the legacy computation, and after training it is the
    distribution's centre member. Recorded as `fgn_eval_eps: "zeros"` in the
    run's `stage2_config` so no reader has to infer it. The HONEST ensemble
    read-outs are the `stage2_val_*` keys, never these."""
    if not eps_dim:
        return None
    return jnp.zeros((n, eps_dim), jnp.float32)


def fgn_train_key(seed):
    """The root of the TRAINING ε stream. The seed arithmetic is torch's own
    (`ml/temporal.py:2296-2297`, `seed * 1000003 + 57`) so the two backends'
    streams are at least NAMED the same; the bits are not and are not claimed
    to be — `ml/plans/FGN_JAX_PORT.md` §1 says cross-backend ε-stream equality
    is NOT required and NOT claimed (different RNG families)."""
    return jax.random.PRNGKey(int(seed) * 1000003 + 57)


def fgn_eps_at(root, step, forward_index, batch, eps_dim, micro_index=None):
    """ε for (step, forward_index), as a PURE FUNCTION of (seed, step, i).

    WHY THIS IS EXACT ON RESUME, AND WHY NO RNG STATE IS CHECKPOINTED.
    JAX's default PRNG (threefry2x32) is COUNTER-BASED: a key is not a
    mutable generator that advances as it is used — it is a 2×uint32 value,
    and `fold_in(key, data)` and `normal(key, shape)` are pure functions of
    their arguments alone. So the ε of step s is a function of (seed, s, i)
    and of nothing else: no draw ever consumed, no ordering, no count of how
    many times the monitor ran. A resumed run reconstructs the root from
    `--seed` and the step from the checkpoint's step counter, and folds again
    — bit-identical by construction, on any device, in any order.

    That is strictly simpler than the torch side, which must SAVE
    `eps_gen.get_state()` into every checkpoint because a
    `torch.Generator` IS mutable state that the number of draws advances.
    Nothing here is saved, because there is nothing to save; the step counter
    already in the `.npz` is the whole of the ε stream's state.

    `forward_index` separates the two members of a CRPS pair (0 and 1) — two
    independent draws on the same context, which is what the spread term
    measures.

    `micro_index` (E-054b) separates the MICRO-BATCHES of one accumulated
    optimiser step, so the fold is over (seed, step, forward, micro) and the
    two members of every micro-batch stay independent of every other
    micro-batch's. **`None` is not micro 0** — it is the legacy two-level
    fold, bit for bit, and `--grad-accum 1` passes `None` for exactly that
    reason: `fold_in(k, 0)` is a DIFFERENT key from `k`, so folding a
    constant zero in "for uniformity" would silently move the ε stream of
    every un-accumulated run and of every resume of one.

    Note what is NOT done: drawing ε once at the full batch and slicing it
    per micro-batch. That would make an accumulated CRPS run reproduce an
    unaccumulated one's ε exactly, but it materialises a batch-sized draw
    per forward — the one array the accumulation exists to stop
    materialising — and the plan (`ml/plans/FGN_JAX_PORT.md` §1) asks only
    that members be INDEPENDENT, never that they be shared across N. So an
    FGN run at N > 1 sees a different (equally valid) ε stream than at N = 1,
    and `stage2_config.grad_accum` is what says which one it was.
    """
    k = jax.random.fold_in(jax.random.fold_in(root, int(step)),
                           int(forward_index))
    if micro_index is not None:
        k = jax.random.fold_in(k, int(micro_index))
    return jax.random.normal(k, (int(batch), int(eps_dim)), jnp.float32)


def fgn_val_bank(seed, members, eps_dim):
    """The FIXED eval ε bank, drawn ONCE from its OWN root
    (`seed * 1000003 + 58`, NOT the training root — `ml/temporal.py:3328-3331`)
    so the monitoring ensemble is the same M members at every log point (a
    member-variance curve is only readable if the members do not change
    underneath it) and so the number of monitor calls cannot perturb the
    training stream. Being counter-based, this needs no separate generator
    OBJECT — a different root key is the whole of the separation."""
    return jax.random.normal(jax.random.PRNGKey(int(seed) * 1000003 + 58),
                             (int(members), int(eps_dim)), jnp.float32)


FGN_MONITOR_CHUNK = 512           # ml/temporal.py:3662 `_CH`


def fgn_monitor_ens(fwd, zseq, mseq, sctx, eps_val, chunk=FGN_MONITOR_CHUNK):
    """The M-member ensemble read of the fixed monitoring batch — [M, n, d_z]
    of LAST-POSITION predictions. `ml/temporal.py:3661-3673`.

    **CHUNKED FROM BIRTH**, in 512-window slices, and that is part of the
    spec rather than a later patch. #496 (E-057.1a seed 0, 2026-08-27) died
    OOM at the step-6000 val: one full-batch forward here is a ~3.65 GB input
    concatenation at stencil 145 × 4096 windows, and M of them per log step
    interleaved with the two-forward CRPS training steps fragmented a 24 GB
    card to death (13.4 GiB reserved-but-unallocated). Same class as E-027
    #285/#286.

    THE CHUNKING CANNOT CHANGE THE ANSWER, and the test says so rather than
    the comment: the forward is row-wise, so concatenation over disjoint
    slices is exact, and ε is BROADCAST per member either way — every window
    of a member sees the identical ε, so no window can see different noise
    because of where a slice boundary fell.

    `fwd(zseq, mseq, sctx, eps) -> (pred, hid)`. `chunk=0` (or any value at
    least n) is the single-slice path, which is what the identity test holds
    the chunked one against.
    """
    n = zseq.shape[0]
    step = int(chunk) if chunk and int(chunk) > 0 else n
    ens = []
    for mi in range(eps_val.shape[0]):
        outs = []
        for c0 in range(0, n, step):
            sl = slice(c0, min(c0 + step, n))
            nb = sl.stop - sl.start
            e = jnp.broadcast_to(eps_val[mi], (nb, eps_val.shape[1]))
            pm, _ = fwd(zseq[sl], mseq[sl], sctx[sl], e)
            outs.append(pm[:, -1])
        ens.append(jnp.concatenate(outs, 0) if len(outs) > 1 else outs[0])
    return jnp.stack(ens)


def fgn_val_metrics(ens, ztrue):
    """The four numbers the fgn monitor produces — `ml/temporal.py:3674-3704`.

    Returns `(val_mse, amp, extra)` where `extra` carries the THREE NEW KEYS
    under the names the torch trainer writes, so status.html needs no new
    record family.

    · `stage2_val_zmse` / `stage2_amp` keep their names, their curves and
      their places; in fgn mode they mean "OF THE ENSEMBLE MEAN", which is
      the best point estimate. The legacy meaning is untouched when fgn is
      off, because this function does not run at all there.
    · `stage2_val_crps` — the fair CRPS of the M members.
    · `stage2_val_member_var` — mean per-element member variance (ddof 0),
      THE ε-COLLAPSE TELEMETRY. A slide toward 0 is the signature of a head
      that has learned to ignore its noise, and it must be visible on the
      live branch while the run is alive, not reconstructed afterwards.
    · `stage2_val_spread_ratio` — spread/error with the (M+1)/M correction,
      mirroring `ml/probscore.spread_error`: the ensemble MEAN carries its own
      σ²/M of sampling error on top of the truth's, so an uncorrected ratio
      reports under-dispersion at every finite M even for a perfect ensemble.
      1.0 is calibration.
    """
    M = float(ens.shape[0])
    mlast = ens.mean(0)
    val_mse = float(jnp.mean((mlast - ztrue) ** 2))
    amp = float(jnp.std(mlast) / (jnp.std(ztrue) + 1e-9))
    # var(ddof=1) is torch's `unbiased=True`; jnp.var's ddof argument.
    msp = float(jnp.mean((M + 1.0) / M * jnp.var(ens, axis=0, ddof=1)))
    extra = {
        "stage2_val_crps": float(fair_crps_ens(ens, ztrue)),
        "stage2_val_member_var": float(jnp.mean(jnp.var(ens, axis=0, ddof=0))),
        "stage2_val_spread_ratio": (math.sqrt(msp) / math.sqrt(val_mse)
                                    if msp >= 0.0 and val_mse > 0.0
                                    else float("nan")),
    }
    return val_mse, amp, extra


def fgn_refusals(a, val_members_default=FGN_VAL_MEMBERS_DEFAULT):
    """Every FGN precondition that depends ONLY on the inputs, checked while
    the inputs are all it has cost (`ml/CLAUDE.md` §0.3). Returns the loss kind
    — `"mse"` or `"crps2"` — so the caller never has to re-derive it.

    THE REFUSE-UNDER-MSE GUARD is the one worth reading twice
    (`ml/plans/FGN_JAX_PORT.md` §1: *"under MSE the flags REFUSE ... an
    ε-conditioned head under MSE is a meaningless arm"*). `--fgn-eps` is ONE
    flag because the objective is not separable from the conditioning: under a
    squared-error loss the conditional mean is optimal, so a head with an ε
    input learns to ignore it and the run measures a deterministic head under
    an FGN arm's name. Setting `--fgn-val-members` with `--fgn-eps 0` asks for
    an M-member ensemble read of a head that HAS no members — every member is
    the same forward — so it refuses rather than logging M identical numbers
    and a member variance of exactly zero, which is also the signature of the
    collapse the telemetry exists to detect.
    """
    if int(a.fgn_eps) < 0:
        raise SystemExit(
            f"--fgn-eps {a.fgn_eps} must be >= 0 (0 = off, and off is the "
            f"exact legacy MSE code path).")
    if int(a.fgn_eps) == 0:
        if int(a.fgn_val_members) != int(val_members_default):
            raise SystemExit(
                f"REFUSED: --fgn-val-members {a.fgn_val_members} with "
                f"--fgn-eps 0. This run trains under PLAIN MSE, where the "
                f"head is deterministic: an M-member ensemble read is M "
                f"copies of one forward, its member variance is exactly zero "
                f"by construction, and zero member variance is ALSO the "
                f"signature of the ε-collapse this telemetry exists to "
                f"detect. An ε-conditioned head under MSE is a meaningless "
                f"arm (ml/plans/FGN_JAX_PORT.md §1) — pass --fgn-eps k > 0, "
                f"which switches the objective to the fair CRPS at N=2, or "
                f"drop --fgn-val-members.")
        return "mse"
    if int(a.fgn_val_members) < 2:
        raise SystemExit(
            f"--fgn-val-members {a.fgn_val_members} must be >= 2: the spread "
            f"and the member variance need at least two members, and they are "
            f"the two numbers the fgn monitor exists to produce.")
    return "crps2"


def stage2_loss(model, zseq, mseq, sctx, ztgt):
    """`l_base` — the WHOLE stage-2 objective at unroll 1 with no direct heads.

    `ml/temporal.py:2436-2438`: one forward, mean squared error between the
    predicted z at every window position and the true next z there. Scored
    over the whole window rather than only its last step, which is why U=1
    leaves the objective bit-identical to the pre-unroll one.

    Every array is passed in rather than drawn here, for the same reason
    `stage1_loss` takes its batch as arguments: identical weights and
    identical inputs, two frameworks, one number.
    """
    pred, hid = model(zseq, mseq, sctx)
    return ((pred - ztgt) ** 2).mean(), (pred, hid)


def stage2_loss_fgn(model, zseq, mseq, sctx, ztgt, eps1, eps2):
    """`l_base` in FGN mode — `ml/temporal.py:3501-3522`, term for term.

    TWO FORWARDS ON THE IDENTICAL CONTEXT, TWO ε. The context (including the
    `--input-znoise` corruption, applied ONCE, before both) is built by the
    caller and passed in unchanged, so the ONLY thing that differs between the
    two members is ε. If the corruption were drawn per forward the pair would
    differ by input noise as well and the CRPS spread term would be measuring
    the wrong perturbation.

    Member 1's hidden state stands where the deterministic forward's stood —
    for shapes only; the direct/unroll paths that consumed it are refused in
    this port anyway.
    """
    p1, hid1 = model(zseq, mseq, sctx, eps=eps1)
    p2, _ = model(zseq, mseq, sctx, eps=eps2)
    return fair_crps2(p1, p2, ztgt), (p1, hid1)


# --------------------------------------------------------------------------
# E-054b · the optimiser step, and the gradient accumulation that lets a head
# too big for one chip's activation budget take exactly the same step
# --------------------------------------------------------------------------
def grad_accum_micro(batch, accum):
    """Refuse an unrepresentable split, and return the micro-batch size.

    A PRECONDITION THAT DEPENDS ONLY ON THE INPUTS IS CHECKED WHILE THE INPUTS
    ARE ALL IT HAS COST (`ml/CLAUDE.md` §0.3). Both refusals exist because the
    alternative is a run that trains at a batch size no record names:

    · `N < 1` — there is no "zero micro-batches" step, and a negative N would
      make `batch // N` negative and the averaging divisor negative with it.
    · `batch % N != 0` — a ragged last micro-batch is NOT the same
      optimisation. The averaged gradient would weight every window of the
      short micro-batch more heavily than every window of the full ones (each
      micro-loss is a mean over ITS OWN element count), so the step would no
      longer be the batch-`batch` step it is labelled as. Refusing costs a
      second; discovering it costs the comparison.
    """
    accum, batch = int(accum), int(batch)
    if accum < 1:
        raise SystemExit(
            f"--grad-accum {accum} must be >= 1 (1 = off, and off builds no "
            f"accumulation graph at all).")
    if batch % accum:
        raise SystemExit(
            f"REFUSED: --batch {batch} is not divisible by --grad-accum "
            f"{accum} ({batch} % {accum} = {batch % accum}). A ragged final "
            f"micro-batch would make the averaged gradient weight its windows "
            f"more heavily than the full micro-batches' — a DIFFERENT "
            f"optimisation from the batch-{batch} step this run would still "
            f"be labelled with. Pick an N that divides the batch.")
    return batch // accum


def make_set_lr(clipped):
    """`_set_lr` as a module-level factory so the step builder and the tests
    close over the SAME function the trainer does."""
    def _set_lr(ost, lr):
        """Write `lr` into whichever level of the chain owns hyperparams.

        `_replace` rather than mutating `ost.hyperparams` in place: the dict
        is a node of a TRACED pytree, and mutating it under jit would edit an
        object the tracer also holds."""
        if clipped:
            inner = ost[1]
            return (ost[0],
                    inner._replace(hyperparams={**inner.hyperparams,
                                                "learning_rate": lr}))
        return ost._replace(hyperparams={**ost.hyperparams,
                                         "learning_rate": lr})
    return _set_lr


class TrainSteps:
    """The jitted step functions of one run, and nothing else.

    `accum == 1` leaves every `micro_*` and `apply_accum` at None — the
    accumulating graphs are not built, not traced and not compiled, which is
    the mechanical form of "flag off is the pre-E-054b path".
    """
    __slots__ = ("accum", "loss_fn", "loss_fn_fgn",
                 "train_step", "train_step_dn",
                 "train_step_fgn", "train_step_fgn_dn",
                 "zero_grads", "micro", "micro_dn", "micro_fgn",
                 "micro_fgn_dn", "apply_accum")

    def __init__(self, **kw):
        for k in self.__slots__:
            setattr(self, k, kw.get(k))


def build_train_steps(graphdef, tx, set_lr, d_z, nz_sigma, accum=1):
    """Every jitted training function of a run, built in ONE place.

    WHY THE ACCUMULATION IS SHAPED THE WAY IT IS — this is the whole point of
    E-054b, so the reasoning is here rather than in a plan nobody re-reads.

    The 400M rung died on an ACTIVATION allocation (5.09 G asked, 4.03 G
    free), not on a weight one: parameters, Adam's two moments and the
    gradient are ~1.6 GB each at 399.948M fp32 and were all resident when it
    failed. Activations scale with the batch and the K² attention; weights do
    not. So the fix has to cut the per-backward batch and MUST NOT hold the
    full batch anywhere on device.

    That rules out the obvious `lax.scan` over a stacked `[N, micro, ...]`
    input: the stacked array IS the full batch, device-resident for the whole
    scan, and the memory it saves is only the difference between one
    backward's activations and N's. It also rules out mapping the micro-grad
    over a device-resident batch. What is left, and what this does:

      · the micro-batches are sliced ON THE HOST (numpy) and transferred one
        at a time, so no device buffer ever spans the batch;
      · one jitted graph per (noise, objective) mode takes ONE micro-batch,
        computes its gradient and ADDS IT INTO the accumulator inside the same
        jit, so the micro gradient is a transient of that graph and never
        coexists with the next micro-batch's;
      · the accumulators are DONATED, so XLA aliases input to output and the
        add is in place — the accumulator is one buffer for the whole step,
        not one per micro-batch;
      · the micro calls are chained through the accumulator, so the DATA
        DEPENDENCY serialises the backwards: micro i+1's activations cannot
        be live while micro i's still are, whatever order the host enqueues
        them in.

    THE BUFFER LIFETIMES, stated so the claim is checkable rather than
    hopeful. At any instant, on any one device, the resident set is

        params (1P) + Adam m, v (2P) + accumulator (1P)
        + the micro gradient inside the running graph (1P, transient)
        + ONE micro-batch's activations (~A/N)
        + the micro-batch inputs the host has enqueued (bounded above by the
          whole batch's inputs — which is exactly what the unaccumulated path
          transfers anyway, so it is not a regression)

    against the unaccumulated path's `params + m + v + grads (4P) + A`. The
    trade is one extra parameter-sized buffer for an N-fold cut in the
    activation peak, and it is only a win where A dominates P — which is the
    measured shape of the E-054b failure and is exactly the case this flag is
    for. The Python `del` after each micro call is part of that argument: it
    drops the host's last reference to the micro-batch's device buffers at
    the point the comment claims they die.

    `--input-znoise` on the HOST backend is applied to the whole batch by the
    producer before the split, and the noise is per-row, so slicing after it
    is the same perturbation the unaccumulated path applies. On the DEVICE
    backend each micro-batch folds its own key from (seed, step, micro).
    """
    accum = int(accum)

    def loss_fn(st, zseq, mseq, sctx, ztgt):
        m = nnx.merge(graphdef, st)
        loss, _ = stage2_loss(m, zseq, mseq, sctx, ztgt)
        return loss

    def loss_fn_fgn(st, zseq, mseq, sctx, ztgt, eps1, eps2):
        m = nnx.merge(graphdef, st)
        loss, _ = stage2_loss_fgn(m, zseq, mseq, sctx, ztgt, eps1, eps2)
        return loss

    @jax.jit
    def train_step_fgn(st, ost, lr, zseq, mseq, sctx, ztgt, eps1, eps2):
        """`train_step` with the fair-CRPS objective. Identical in every other
        respect — same pre-clip global norm on every step, same `_set_lr`,
        same optax chain — so an FGN arm and an MSE arm differ in the
        objective and in nothing else."""
        zseq = zseq.astype(jnp.float32)
        loss, grads = jax.value_and_grad(loss_fn_fgn)(
            st, zseq, mseq, sctx, ztgt, eps1, eps2)
        gnorm = optax.global_norm(grads)
        ost = set_lr(ost, lr)
        upd, ost = tx.update(grads, ost, st)
        return optax.apply_updates(st, upd), ost, loss, gnorm

    @jax.jit
    def train_step_fgn_dn(st, ost, lr, key, zseq, mseq, sctx, ztgt,
                          eps1, eps2):
        """The device-noise twin. THE CORRUPTION IS APPLIED ONCE, BEFORE BOTH
        FORWARDS (`ml/temporal.py:3502-3507`): `apply_znoise_jax` runs here,
        outside `stage2_loss_fgn`, so the two members see the identical
        perturbed context and the CRPS spread term measures ε alone."""
        zn = apply_znoise_jax(zseq, key, nz_sigma, d_z)
        loss, grads = jax.value_and_grad(loss_fn_fgn)(
            st, zn, mseq, sctx, ztgt, eps1, eps2)
        gnorm = optax.global_norm(grads)
        ost = set_lr(ost, lr)
        upd, ost = tx.update(grads, ost, st)
        return optax.apply_updates(st, upd), ost, loss, gnorm

    @jax.jit
    def train_step(st, ost, lr, zseq, mseq, sctx, ztgt):
        # Exact no-op for the fp32 path; makes --gather-fp16 safe here too
        # (fp16→fp32 is exact, and the model must never run in fp16).
        zseq = zseq.astype(jnp.float32)
        loss, grads = jax.value_and_grad(loss_fn)(st, zseq, mseq, sctx, ztgt)
        # THE PRE-CLIP GLOBAL NORM, on EVERY step. Clipping computes it
        # anyway, and the unclipped path pays one extra reduction — cheap
        # beside a forward and backward over a 200M head, and it is what
        # turns `stage2_grad_norm` from "one step in log_every" into a window
        # statistic (§4.10: instrument the quantity that distinguishes the
        # stories). #423's excursion could have begun anywhere inside a
        # 2,000-step window and no record could say where.
        gnorm = optax.global_norm(grads)
        ost = set_lr(ost, lr)
        upd, ost = tx.update(grads, ost, st)
        return optax.apply_updates(st, upd), ost, loss, gnorm

    # The device-noise twin (--noise-backend device): identical to train_step
    # except the input perturbation is drawn and applied INSIDE the jit —
    # see apply_znoise_jax for both the measurement that motivates it and the
    # RNG-stream caveat. zseq may arrive fp16 (--gather-fp16); the cast to
    # fp32 happens in apply_znoise_jax and is exact.

    @jax.jit
    def train_step_dn(st, ost, lr, key, zseq, mseq, sctx, ztgt):
        zn = apply_znoise_jax(zseq, key, nz_sigma, d_z)
        loss, grads = jax.value_and_grad(loss_fn)(st, zn, mseq, sctx, ztgt)
        gnorm = optax.global_norm(grads)
        ost = set_lr(ost, lr)
        upd, ost = tx.update(grads, ost, st)
        return optax.apply_updates(st, upd), ost, loss, gnorm

    steps = TrainSteps(accum=accum, loss_fn=loss_fn, loss_fn_fgn=loss_fn_fgn,
                       train_step=train_step, train_step_dn=train_step_dn,
                       train_step_fgn=train_step_fgn,
                       train_step_fgn_dn=train_step_fgn_dn)
    if accum == 1:
        # NOT A BRANCH IN A GRAPH — an absent graph. Nothing below is defined,
        # so an un-accumulated run traces, compiles and executes exactly the
        # functions above and the flag costs it not one operation.
        return steps

    INV = 1.0 / float(accum)          # baked at trace time, never an argument

    @jax.jit
    def zero_grads(st):
        """The accumulator, allocated once per optimiser step. One jitted
        memset rather than a `tree_map` of eager `zeros_like` calls, so the
        per-step host dispatch is one call and not one per leaf."""
        return jax.tree.map(jnp.zeros_like, st)

    # THE MICRO GRAPHS. Each is `value_and_grad` over ONE micro-batch plus
    # the accumulate, fused into a single jit: the micro gradient is born and
    # consumed inside the graph, so it is never a second live parameter-sized
    # tree in the host's hands. `donate_argnums=(1, 2)` gives the accumulators
    # to XLA to overwrite — the add is in place, so N micro-batches cost ONE
    # accumulator buffer and not N.
    #
    # The loss accumulates ON DEVICE beside the gradient. Summing micro losses
    # on the host would mean one device→host sync per micro-batch, which
    # would serialise the pipeline the prefetch exists to keep full; one sync
    # per optimiser step is what the unaccumulated path already pays.
    _jit_acc = functools.partial(jax.jit, donate_argnums=(1, 2))

    @_jit_acc
    def micro(st, gacc, lacc, zseq, mseq, sctx, ztgt):
        zseq = zseq.astype(jnp.float32)
        loss, grads = jax.value_and_grad(loss_fn)(st, zseq, mseq, sctx, ztgt)
        return jax.tree.map(jnp.add, gacc, grads), lacc + loss

    @_jit_acc
    def micro_dn(st, gacc, lacc, key, zseq, mseq, sctx, ztgt):
        zn = apply_znoise_jax(zseq, key, nz_sigma, d_z)
        loss, grads = jax.value_and_grad(loss_fn)(st, zn, mseq, sctx, ztgt)
        return jax.tree.map(jnp.add, gacc, grads), lacc + loss

    @_jit_acc
    def micro_fgn(st, gacc, lacc, zseq, mseq, sctx, ztgt, eps1, eps2):
        zseq = zseq.astype(jnp.float32)
        loss, grads = jax.value_and_grad(loss_fn_fgn)(
            st, zseq, mseq, sctx, ztgt, eps1, eps2)
        return jax.tree.map(jnp.add, gacc, grads), lacc + loss

    @_jit_acc
    def micro_fgn_dn(st, gacc, lacc, key, zseq, mseq, sctx, ztgt, eps1, eps2):
        zn = apply_znoise_jax(zseq, key, nz_sigma, d_z)
        loss, grads = jax.value_and_grad(loss_fn_fgn)(
            st, zn, mseq, sctx, ztgt, eps1, eps2)
        return jax.tree.map(jnp.add, gacc, grads), lacc + loss

    @jax.jit
    def apply_accum(st, ost, lr, gacc, lacc):
        """ONE AdamW update on the AVERAGED gradient.

        The division happens HERE, before the norm and before the optax
        chain, which is what makes the accumulated step the batch step in
        every visible respect: `gnorm` is the pre-clip norm OF THE AVERAGE, so
        `--grad-clip` binds on exactly the quantity it binds on at N = 1 and
        the clip statistics stay comparable across N. Sum-then-divide rather
        than divide-each-micro-then-sum: one rounding instead of N, and the
        accumulator keeps the micro gradients' own scale while it fills.
        """
        grads = jax.tree.map(lambda g: g * INV, gacc)
        gnorm = optax.global_norm(grads)
        ost = set_lr(ost, lr)
        upd, ost = tx.update(grads, ost, st)
        return optax.apply_updates(st, upd), ost, lacc * INV, gnorm

    steps.zero_grads = zero_grads
    steps.micro, steps.micro_dn = micro, micro_dn
    steps.micro_fgn, steps.micro_fgn_dn = micro_fgn, micro_fgn_dn
    steps.apply_accum = apply_accum
    return steps


def accum_step(steps, state, opt_state, lr, s, zseq, mseq, sctx, ztgt, micro,
               put=jnp.asarray, fgn_root=None, fgn_eps=0, nz_root=None):
    """ONE optimiser update over `steps.accum` micro-batches — E-054b.

    Returns exactly what the un-accumulated `train_step` returns —
    `(state, opt_state, loss, gnorm)` — with `loss` the mean over the EFFECTIVE
    batch and `gnorm` the PRE-clip norm of the AVERAGED gradient, so every
    metric the loop writes keeps its meaning and a "step" keeps meaning one
    optimiser update.

    The micro-batches are contiguous HOST slices of the batch the producer
    already built, so the window pool, the draw order and the host-noise
    stream are the un-accumulated run's exactly; only where the batch is cut
    into forwards changes. `build_train_steps` carries the argument for why
    the slicing happens here rather than in a scan over a stacked device
    array (the stack IS the full batch, resident — the one thing this exists
    to prevent), and why the buffer lifetimes below bound the peak at one
    micro-batch of activations.

    `fgn_root=None` means MSE, `nz_root=None` means the corruption is not
    drawn on device: the two selectors are the loop's own `FGN` and
    `device_noise` flags, passed rather than re-derived so this function and
    the un-accumulated call site cannot disagree about which mode a run is in.
    """
    gacc = steps.zero_grads(state)
    lacc = jnp.zeros((), jnp.float32)
    for mi in range(int(steps.accum)):
        sl = slice(mi * micro, (mi + 1) * micro)
        zs_, ms_, cs_, ts_ = (put(zseq[sl]), put(mseq[sl]),
                              put(sctx[sl]), put(ztgt[sl]))
        # The input corruption, when it is drawn on device, is a FRESH draw
        # per micro-batch folded from (seed, step, micro) — at N = 1 this
        # function is not called at all, so the legacy `fold_in(root, s)` is
        # untouched, and at N > 1 no two micro-batches share a perturbation.
        key = (None if nz_root is None
               else jax.random.fold_in(jax.random.fold_in(nz_root, s), mi))
        if fgn_root is not None:
            # (seed, step, forward, MICRO). The micro index is folded in so
            # the two members of THIS micro-batch are independent of every
            # other micro-batch's pair as well as of each other; `fgn_eps_at`
            # explains why `--grad-accum 1` passes None here instead of 0 and
            # why that is not a cosmetic difference.
            e1 = fgn_eps_at(fgn_root, s, 0, micro, fgn_eps, micro_index=mi)
            e2 = fgn_eps_at(fgn_root, s, 1, micro, fgn_eps, micro_index=mi)
            if key is None:
                gacc, lacc = steps.micro_fgn(state, gacc, lacc,
                                             zs_, ms_, cs_, ts_, e1, e2)
            else:
                gacc, lacc = steps.micro_fgn_dn(state, gacc, lacc, key,
                                                zs_, ms_, cs_, ts_, e1, e2)
        elif key is None:
            gacc, lacc = steps.micro(state, gacc, lacc, zs_, ms_, cs_, ts_)
        else:
            gacc, lacc = steps.micro_dn(state, gacc, lacc, key,
                                        zs_, ms_, cs_, ts_)
        # The micro-batch's device buffers lose their last host reference
        # HERE, before the next one is made — the lifetime the memory argument
        # in `build_train_steps` depends on.
        del zs_, ms_, cs_, ts_
    return steps.apply_accum(state, opt_state, lr, gacc, lacc)


# --------------------------------------------------------------------------
# the schedules — `ml/temporal.py:make_sched` (:468) as plain functions
# --------------------------------------------------------------------------
def lr_factor(e, a):
    """The LR multiplier at scheduler position `e`.

    `e` is the number of `sched.step()` calls already made, so at TRAINING
    step s (1-indexed) the factor is `lr_factor(s - 1)` — torch steps the
    scheduler AFTER the optimiser, which makes the two line up with no
    off-by-one. All four branches are transcribed from `make_sched`'s own
    closures, which are `LambdaLR` lambdas over `step` and internally use
    `s = step + 1`.

    The cosine branch is `CosineAnnealingLR(T_max=steps)`'s closed form,
    clamped at `steps` (torch's recursion oscillates past T_max; nothing here
    ever runs past it, and clamping is the honest reading of "annealed").
    """
    warm = max(1, int(a.lr_warmup))
    s = e + 1

    if a.lr_schedule == "cosine":
        total = max(1, int(a.steps))
        return 0.5 * (1 + math.cos(math.pi * min(e, total) / total))

    def _warm_cos(x):
        return 0.5 * (1 - math.cos(math.pi * min(1.0, x / warm)))

    if a.lr_schedule == "wsd":
        cool = max(1, int(round(a.steps * a.lr_cooldown_frac)))
        stable_end = max(warm, a.steps - cool)
        if s <= warm:
            return s / warm
        if s <= stable_end:
            return 1.0
        return max(0.0, (a.steps - s) / max(1, a.steps - stable_end))

    if a.lr_schedule == "expdecay":
        half = max(1.0, float(a.lr_halflife))
        cool = max(0, int(round(a.steps * a.lr_cooldown_frac)))
        taper_from = a.steps - cool
        base = _warm_cos(s) if s <= warm else 0.5 ** ((s - warm) / half)
        if cool and s > taper_from:
            base *= max(0.0, (a.steps - s) / cool)
        return base

    # invsqrt (the Noam schedule)
    return min(s / warm, (warm / s) ** 0.5)


# --------------------------------------------------------------------------
# checkpoint I/O — the same flat-leaf .npz `train_stage1.py` uses
# --------------------------------------------------------------------------
def _leaves(tree):
    return jax.tree_util.tree_leaves(tree)


def save_state_npz(path, state, opt_state, step, args):
    """params + optimiser moments + step + args, flat over the pytree LEAVES.

    Identical in shape and in reasoning to `train_stage1.save_state_npz`: the
    leaf order is a deterministic function of the tree structure, the
    structure is a deterministic function of the architecture, and the
    architecture is in the file — so a load rebuilds the structure first and
    refuses on any count or shape it did not expect.
    """
    blob = {"_step": np.asarray(int(step)),
            "_args": np.asarray(json.dumps(args)),
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
    Returns (state, opt_state, step, args)."""
    z = np.load(path, allow_pickle=False)
    args = json.loads(str(z["_args"]))
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
            int(z["_step"]), args)


# --------------------------------------------------------------------------
# E-047: a BLOCK codec makes the block axis the axis
# --------------------------------------------------------------------------
def adopt_block_axis(ck_args, d, months, moy, t_hold, T, rapid_arr, C, d_z,
                     time_stride=0):
    """`ml/temporal.py:1565-1600`, factored out so a test can hold it against
    that branch on a toy without running an embedding.

    `k_time` comes from the CODEC, not from a flag: the head consumes whatever
    the frozen encoder emits, and a block codec emits one z per BLOCK. So the
    axis becomes the block axis — labels, month-of-year, the held-out mask
    (ANY held-out bin of a block makes the whole block held out, because the
    block's single embedding saw that bin) and the RAPID truth rows, which are
    keyed on the source axis and must be remapped or they point at the wrong
    blocks.

    Returns `(BLKA, months, moy, t_hold, T, rapid_arr)`; on a per-bin codec
    every one of them comes back untouched and `BLKA` is None.
    """
    k_time = int(ck_args.get("k_time", 1) or 1)
    if k_time <= 1:
        return None, months, moy, t_hold, T, rapid_arr
    tb = str(ck_args.get("time_block", "") or "")
    if not tb:
        raise SystemExit(
            "the codec has k_time > 1 but no `time_block` in its args: this "
            "checkpoint cannot say how its blocks were cut, and guessing "
            "would embed a different grouping than it was trained on.")
    if time_stride:
        raise SystemExit(
            "--time-stride on a BLOCK codec: two time surgeries at once. The "
            "blocks already re-cut the axis; striding it as well would leave "
            "an axis no artefact describes.")
    BLKA = BlockAxis(tb, months,
                     d["bin_index"] if "bin_index" in d else None,
                     (dt.date.fromisoformat(str(d["epoch"]))
                      if "epoch" in d else None),
                     (int(np.asarray(d["pentad_days"]).item())
                      if "pentad_days" in d else None))
    print(BLKA.describe(C, d_z), flush=True)
    rapid_arr = BLKA.remap_rows(rapid_arr)
    months = list(BLKA.labels)
    moy = np.array([int(m[5:7]) - 1 for m in months])
    t_hold = np.array([t_hold[BLKA.rows[b, :int(BLKA.n_bins[b])]].any()
                       for b in range(BLKA.n_blocks)])
    T = BLKA.n_blocks
    print(f"  block axis: T {T} · held out {int(t_hold.sum())} · RAPID rows "
          f"{len(rapid_arr)} · labels {months[0]}..{months[-1]}", flush=True)
    return BLKA, months, moy, t_hold, T, rapid_arr


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------
def parse(argv=None):
    p = argparse.ArgumentParser(
        description="stage-2 TemporalTransformer trainer, JAX/optax "
                    "(ml/plans/JAX_PORT.md tier 3b)")
    p.add_argument("--data", required=True, help="tensor npz/sidecar")
    p.add_argument("--ckpt", required=True,
                   help="the FROZEN codec .pt. Required even with --z: the "
                        "architecture, the holdout_lon that chooses the "
                        "anomaly statistics, and the per-pixel STATIC "
                        "IDENTITY embedding all come out of it.")
    p.add_argument("--z", default="",
                   help="a published embedding cache "
                        "(Z_<codec hash>_<tensor sha>.npy, [T,P,d_z] float16). "
                        "Absent: embed on the spot through "
                        "jaxport.embed.embed_everything_jax.")
    p.add_argument("--out", required=True)
    p.add_argument("--K", type=int, default=24,
                   help="context length in AXIS STEPS. k_max follows K for a "
                        "fresh head, exactly as ml/temporal.py does — the "
                        "positional table is what rollout_spatial reads k_max "
                        "back out of.")
    p.add_argument("--steps", type=int, default=4000)
    p.add_argument("--batch", type=int, default=256)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--d-model", type=int, default=96)
    p.add_argument("--layers", type=int, default=3)
    p.add_argument("--n-heads", type=int, default=4,
                   help="attention heads. ml/temporal.py hard-codes 4 (the "
                        "TemporalTransformer default) and rollout_spatial "
                        "rebuilds every head with n_heads=4, so anything else "
                        "produces a checkpoint the torch eval ladder would "
                        "silently rebuild wrong — hence the refusal below.")
    p.add_argument("--lr-schedule", default="cosine",
                   choices=["cosine", "invsqrt", "wsd", "expdecay"],
                   help="cosine bakes the TOTAL into the rate; the other "
                        "three are horizon-free. expdecay with "
                        "--lr-halflife 40000 is the fleet's stage-2 config.")
    p.add_argument("--lr-halflife", type=float, default=40000)
    p.add_argument("--lr-cooldown-frac", type=float, default=0.1)
    p.add_argument("--lr-warmup", type=int, default=2000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--tag", default="")
    p.add_argument("--stencil", type=int, default=1)
    p.add_argument("--ring-km", default="0")
    p.add_argument("--input-znoise", type=float, default=0.0,
                   help="E-029b: Gaussian noise (ABSOLUTE std, in whatever "
                        "units the frozen codec's z-space happens to be in) "
                        "added to the LIVE input slots during training. The "
                        "run prints and records the ratio to "
                        "sqrt(val_persistence) and to RMS |z|, because 0.7 "
                        "means a different perturbation on every codec.")
    p.add_argument("--grad-clip", type=float, default=0.0,
                   help="E-044: max global gradient 2-norm. 0.0 (DEFAULT) = "
                        "OFF, and OFF ADDS NO OPTAX TRANSFORM AT ALL — the "
                        "unclipped chain exactly, so an arm that does not opt "
                        "in is the pre-clip code path.")
    p.add_argument("--grad-accum", type=int, default=1,
                   help="E-054b: split each optimiser step's --batch into N "
                        "micro-batches of batch/N, accumulate their gradients "
                        "and take ONE AdamW update on the AVERAGE. 1 "
                        "(DEFAULT) = OFF, and OFF BUILDS NO ACCUMULATION "
                        "GRAPH AT ALL. The optimisation is the batch-N one "
                        "exactly (mean loss, equal micro-batches — only float "
                        "association differs); what changes is the ACTIVATION "
                        "peak, which is what put the 400M rung over a v5e-4 "
                        "chip's HBM. REFUSES unless N >= 1 and N divides "
                        "--batch.")
    p.add_argument("--noise-backend", default="host",
                   choices=["host", "device"],
                   help="Where the --input-znoise draw happens. 'host' "
                        "(DEFAULT) is the original numpy path — bit-stable "
                        "with every run before 2026-08-25. 'device' draws "
                        "inside the jitted step (apply_znoise_jax): same "
                        "semantics, different RNG stream, and it removes a "
                        "measured ~15 s/step of single-threaded host RNG at "
                        "K=144. The TPU launcher sets 'device'.")
    p.add_argument("--gather-fp16", action="store_true",
                   help="Ship the window gather in Z's own fp16 and cast to "
                        "fp32 on device (exact) instead of casting on the "
                        "host — halves both the host memcpy and the "
                        "host-to-device transfer. Values identical.")
    p.add_argument("--gather-workers", type=int, default=0,
                   help="Thread-pool width for the window gather's K loop. "
                        "0 (DEFAULT) = the original single-thread code path. "
                        "Same arithmetic, same output.")
    p.add_argument("--prefetch", type=int, default=0,
                   help="Depth of the host-side batch prefetch queue. 0 "
                        "(DEFAULT) = batches are built inline, the original "
                        "path. N>0 builds them in a producer thread so host "
                        "gather time overlaps device step time. The batch "
                        "SEQUENCE is unchanged (one RNG, drawn in step "
                        "order); only the wall clock moves.")
    # ---- E-057 · FGN, mirroring ml/temporal.py's two flags exactly ---------
    p.add_argument("--fgn-eps", type=int, default=0,
                   help="E-057: k = dimension of the global noise vector "
                        "eps ~ N(0,1)^k conditioning every encoder layer's "
                        "two LayerNorms through a zero-init FiLM (FGN, "
                        "arXiv:2506.10772; adaLN-zero, so the head at step 0 "
                        "IS the deterministic incumbent). 0 (DEFAULT) = OFF = "
                        "the exact legacy code path: no eps_embed, no film, "
                        "no extra draw, and the MSE objective. When > 0 the "
                        "training objective SWITCHES to the FAIR CRPS AT N=2 "
                        "(two forwards per batch on the identical context, "
                        "two independent eps) — ONE flag, because under plain "
                        "MSE the conditional mean is optimal and a "
                        "noise-conditioned head learns to IGNORE eps. FGN's "
                        "own default is k=32. Plan: ml/plans/FGN_JAX_PORT.md")
    p.add_argument("--fgn-val-members", type=int, default=FGN_VAL_MEMBERS_DEFAULT,
                   help="E-057: M, the ensemble size for the in-training "
                        "monitoring reads (stage2_val_crps, "
                        "stage2_val_member_var, stage2_val_spread_ratio). The "
                        "eval eps bank is FIXED at setup from its own root "
                        "key, so the members do not change under the curve. "
                        "Read ONLY when --fgn-eps > 0, and REFUSED when it is "
                        "0 (see fgn_refusals).")
    p.add_argument("--milestone-steps", default="",
                   help="E-031/E-032: comma list of steps at which to save a "
                        "WEIGHTS-ONLY milestone head (temporal_ms<step>.pt). "
                        "A milestone at or past --steps REFUSES: a retention "
                        "request that retains nothing is worse than none.")
    p.add_argument("--time-stride", type=int, default=0,
                   help="SUBSAMPLE THE AXIS: keep bins range(--time-offset, "
                        "T, N). 0 (default) keeps every bin. Refused on a "
                        "block codec.")
    p.add_argument("--time-offset", type=int, default=0)
    p.add_argument("--season-phase", default="month",
                   choices=("month", "fine"))
    p.add_argument("--train-lon-hold", default="inherit",
                   help="which longitudes are EXCLUDED FROM THE TRAINING "
                        "POOL. 'inherit' = the codec's own; 'none' = train on "
                        "every column; 'lo,hi' = an explicit block. THE POOL "
                        "ONLY — the anomaly statistics always follow the "
                        "codec. Pass an explicit block as one word: "
                        "--train-lon-hold=-45,-25.")
    p.add_argument("--max-pixels", type=int, default=0,
                   help="subsample ocean pixels (code-path smoke only; the "
                        "26.5N section is always kept)")
    p.add_argument("--ckpt-every", type=int, default=0,
                   help="write the resumable .npz and the torch-format head "
                        "every N steps (0 = at the logging cadence). A long "
                        "TPU run wants this ON: tpu_train_s2.sh's progress "
                        "watchdog reaps a node whose checkpoint stopped "
                        "moving.")
    p.add_argument("--resume", default="",
                   help="a .npz written by THIS trainer (weights + optimiser "
                        "+ step), or a torch head .pt, which WARM-STARTS "
                        "(weights only — torch Adam state is not mapped into "
                        "optax, JAX_PORT.md §3.3).")
    p.add_argument("--require-resume", action="store_true")
    p.add_argument("--shard-batch", default="auto", choices=["auto", "off"],
                   help="data parallelism across local devices: the batch "
                        "axis is sharded, the parameters replicated. 'auto' "
                        "is a no-op on one device, which is every CPU test.")
    # ---- the REFUSALS. Each names a flag ml/temporal.py has that this port
    # deliberately does not, so a dispatch string copied verbatim from a torch
    # arm fails loudly instead of training something else.
    p.add_argument("--input-quant", default="",
                   help="REFUSED: the KNOWN GAP of this port. A-arm parity on "
                        "quantized inputs waits for it.")
    p.add_argument("--unroll", type=int, default=1, help="REFUSED above 1")
    p.add_argument("--unroll-wide", type=int, default=0, help="REFUSED")
    p.add_argument("--unroll-probs", default="", help="REFUSED")
    p.add_argument("--direct", default="", help="REFUSED")
    p.add_argument("--target-bins-argo", default="all", help="REFUSED != all")
    p.add_argument("--season-dropout", type=float, default=0.0,
                   help="REFUSED above 0")
    return p.parse_args(argv)


REFUSALS = (
    ("input_quant", lambda v: bool(str(v).strip()),
     "--input-quant is the KNOWN GAP of the JAX stage-2 port (JAX_PORT.md "
     "tier 3b). The quantizer is part of the MODEL'S CONTRACT — "
     "ml/rollout_spatial.py re-applies it at roll time from "
     "input_quant_sigma — so a head trained here without it and labelled "
     "with it would be rolled through a grid it never saw. Train quantized "
     "arms on the torch stack until this lands."),
    ("unroll", lambda v: int(v) > 1,
     "--unroll > 1 is not ported: it feeds predictions back through the "
     "window and is its own experiment (E-010/E-020 closed the axis). Every "
     "arm this port targets is U=1."),
    ("unroll_wide", lambda v: int(v) > 0,
     "--unroll-wide is not ported (E-030's one-hop wide unroll)."),
    ("unroll_probs", lambda v: bool(str(v).strip()),
     "--unroll-probs is not ported: it samples an unroll depth, and unroll "
     "is not ported."),
    ("direct", lambda v: bool(str(v).strip()),
     "--direct multi-horizon heads are not ported. The JAX "
     "TemporalTransformer CAN carry them (models.py builds heads_direct and "
     "convert.py maps them both ways), but the trainer does not score them, "
     "and a head whose direct heads are at their init would roll as though "
     "they had been trained."),
    ("target_bins_argo", lambda v: str(v) != "all",
     "--target-bins-argo is not ported: it filters the training pool by what "
     "the scored bins carry, and a pool that silently ignored the filter is "
     "a different experiment under the same name."),
    ("season_dropout", lambda v: float(v) > 0,
     "--season-dropout is not ported."),
)


# --------------------------------------------------------------------------
def main(argv=None):
    a = parse(argv)
    if a.resume.startswith("!"):
        a.require_resume, a.resume = True, a.resume[1:]
    for name, hit, why in REFUSALS:
        if hit(getattr(a, name)):
            raise SystemExit(f"REFUSED: {why}")
    # A PRECONDITION THAT DEPENDS ONLY ON THE INPUTS IS CHECKED WHILE THE
    # INPUTS ARE ALL IT HAS COST (ml/CLAUDE.md §0.3). A negative max_norm is
    # not "off": it would scale every gradient by a negative coefficient and
    # flip the descent direction, and nothing downstream would say so.
    if a.grad_clip < 0:
        raise SystemExit(f"--grad-clip {a.grad_clip} must be >= 0 (0 = off, "
                         f"and off adds no optax transform at all).")
    # E-054b, same rule: the accumulation's two preconditions are functions of
    # --batch and --grad-accum alone, so they are settled here and not after
    # an embedding pass.
    ACCUM = int(a.grad_accum)
    MICRO = grad_accum_micro(a.batch, ACCUM)
    # E-057, at argv time and before anything expensive: the fgn preconditions
    # (including the refuse-under-MSE guard) and the loss kind they decide.
    LOSS_KIND = fgn_refusals(a)
    FGN = LOSS_KIND == "crps2"
    if a.n_heads != 4:
        raise SystemExit(
            f"--n-heads {a.n_heads}: ml/temporal.py hard-codes 4 and "
            f"ml/rollout_spatial.py rebuilds every head with n_heads=4 from "
            f"the checkpoint's args, which do NOT carry the head count. A "
            f"head trained at {a.n_heads} would be silently rebuilt at 4 and "
            f"roll as a different model.")
    os.makedirs(a.out, exist_ok=True)
    devices = jax.local_devices()
    print(f"jax {jax.__version__} · devices {[str(dv) for dv in devices]}",
          flush=True)

    rng = np.random.default_rng(a.seed)

    # ---- the codec ------------------------------------------------------
    import torch                          # local: the .pt reader, as elsewhere
    ck = torch.load(a.ckpt, map_location="cpu", weights_only=False)
    if not ck["args"].get("anomaly"):
        raise SystemExit("stage 2 requires an anomaly-space codec "
                         "(train.py --anomaly): state-space embeddings failed "
                         "the K-sweep precondition.")
    d_z = int(ck["d_z"])

    # ---- data -----------------------------------------------------------
    d = load_tensor(a.data)
    X = d["X"]
    if isinstance(X, np.memmap) and not X.flags.writeable:
        # anomaly_transform writes into X. The canonical map must never take
        # those writes — it would leave an anomaly-space tensor where a
        # state-space one is documented (ml/temporal.py has the same guard).
        scratch = a.data[:-4] + "_s2jax_scratch.npy"
        X = writable_copy(X, scratch, verbose=False)
        import atexit
        atexit.register(lambda q=scratch: os.path.exists(q) and os.remove(q))
    months = [str(m) for m in d["months"]]
    lats, lons = d["lats"], d["lons"]
    chan = [str(c) for c in d["chan"]]
    T, H, W, C = X.shape
    moy = np.array([int(m[5:7]) - 1 for m in months])
    hold_years = set(ck["args"]["holdout_years"].split(","))
    t_hold = np.array([m[:4] in hold_years for m in months])

    # TWO MASKS, AND THEY ARE NOT THE SAME OBJECT (ml/temporal.py:1440-1460).
    # `stat_x_hold` chooses the anomaly transform's z-score statistics and
    # ALWAYS follows the codec: the encoder was fitted on inputs normalised
    # over (train years × non-held-out longitudes), and re-deriving mu/sd over
    # a larger pool rescales every pixel it is then asked to encode — a silent
    # covariate shift on a frozen encoder, in a run whose whole point is that
    # the codec is unchanged. It is also unsafe in a way no number would show:
    # the embedding cache is keyed by (codec weight hash, RAW tensor sha) and
    # neither term sees the transform, so two runs with different statistics
    # share one cache key. `--train-lon-hold` governs `pool_x_hold` ALONE.
    stat_x_hold = lon_holdout_mask(ck["args"]["holdout_lon"], lons)
    _tlh = str(a.train_lon_hold).strip()
    pool_x_hold = (stat_x_hold if _tlh.lower() == "inherit"
                   else lon_holdout_mask(_tlh, lons))
    print(f"lon holdout · statistics (codec "
          f"{ck['args']['holdout_lon']!r}): {int(stat_x_hold.sum())}/"
          f"{len(lons)} cols · training pool (--train-lon-hold "
          f"{a.train_lon_hold!r}): {int(pool_x_hold.sum())}/{len(lons)} cols",
          flush=True)

    from trainprobe import anomaly_transform          # lazy, as ml/temporal.py
    X, dynamic = anomaly_transform(X, moy, t_hold, stat_x_hold)

    # ---- --time-stride: subsample the axis, and mind the ORDER -----------
    # AFTER the anomaly transform, never before — the transform's climatology
    # and z-score are computed over the TRAIN bins of the FULL axis, and
    # recomputing them over one bin in six would shift every normalised value
    # the frozen encoder is then asked to encode. The published Z is a
    # per-bin object and every bin is an independent encoder forward, so
    # slicing a full Z by `tsel` is exactly what embedding the strided tensor
    # would have produced — which is why `--z` and `--time-stride` compose
    # here where they cannot on the torch side (there the strided tensor is
    # re-embedded because a [T/N, P, d_z] array is indistinguishable by shape
    # from a complete one and must never be published under an unstrided
    # name). Nothing is published from here, so nothing can be mislabelled.
    tsel = None
    ocean_full = None
    if a.time_stride:
        if a.time_stride < 1:
            raise SystemExit(f"--time-stride {a.time_stride}: must be >= 1")
        if not (0 <= a.time_offset < a.time_stride):
            raise SystemExit(f"--time-offset {a.time_offset} must satisfy "
                             f"0 <= O < N for --time-stride {a.time_stride}")
        # The ocean mask is a property of the FULL record and must not become
        # a property of the sample.
        ocean_full = np.zeros(X.shape[1:3], bool)
        for i0 in range(0, T, 64):
            ocean_full |= np.isfinite(X[i0:i0 + 64, :, :, 0]).any(axis=0)
        tsel = np.arange(a.time_offset, T, a.time_stride)
        X = np.ascontiguousarray(X[tsel])
        months = [months[i] for i in tsel]
        moy, t_hold = moy[tsel], t_hold[tsel]
        T = len(tsel)
        print(f"--time-stride {a.time_stride} offset {a.time_offset}: {T} "
              f"bins kept ({months[0]}..{months[-1]}), held-out "
              f"{int(t_hold.sum())} · K={a.K} now spans {a.K} KEPT bins",
              flush=True)

    # THE RAPID TRUTH IS KEYED ON THE AXIS ROW, so it is subsampled with the
    # axis or it points at the wrong bins.
    rapid_arr = d["rapid"]
    if tsel is not None:
        _pos = {int(r): i for i, r in enumerate(tsel)}
        _keep = [i for i, r in enumerate(rapid_arr[:, 0].astype(int))
                 if int(r) in _pos]
        rapid_arr = rapid_arr[_keep].copy()
        rapid_arr[:, 0] = [_pos[int(r)] for r in rapid_arr[:, 0].astype(int)]
        print(f"  RAPID truth on the strided axis: {len(rapid_arr)} of "
              f"{len(d['rapid'])} rows survive", flush=True)

    k_time = int(ck["args"].get("k_time", 1) or 1)
    BLKA, months, moy, t_hold, T, rapid_arr = adopt_block_axis(
        ck["args"], d, months, moy, t_hold, T, rapid_arr, C, d_z,
        time_stride=a.time_stride)

    # The CODEC's own context. In month mode it is the month-quantized
    # sin/cos; on a BLOCK axis it is the block centre's CONTINUOUS phase,
    # because that is what ml/train.py fed the encoder (`ctx_mode
    # block_phase`) and what ml/rollout_spatial.py re-encodes with. Embedding
    # with the month token instead would feed the frozen encoder a context it
    # had never seen (E-048, fixing an E-047 gap).
    ctx_all = np.stack([np.sin(2 * np.pi * moy / 12),
                        np.cos(2 * np.pi * moy / 12)], 1)
    codec_ctx = ctx_all if BLKA is None else BLKA.ctx_phase()

    Xt = LazyPixels(X)
    OBS = LazyPixels(X, obs=True)
    if ocean_full is not None:
        ocean = ocean_full
    else:
        ocean = np.zeros(X.shape[1:3], bool)
        for i0 in range(0, X.shape[0], 64):
            ocean |= np.isfinite(X[i0:i0 + 64, :, :, 0]).any(axis=0)
    ys, xs = np.where(ocean)
    sec_y, sec_sel0 = rapid_section(lats, lons, ys, xs)
    if a.max_pixels and a.max_pixels < len(ys):
        keep = np.random.default_rng(0).choice(len(ys), a.max_pixels,
                                               replace=False)
        keep = np.union1d(keep, sec_sel0)           # the probe needs the section
        ys, xs = ys[keep], xs[keep]
    P = len(ys)
    coords = np.stack([lats[ys] / 90, lons[xs] / 180], 1).astype(np.float32)
    print(f"X [T={T} H={H} W={W} C={C}] · ocean pixels {P:,} · held-out bins "
          f"{int(t_hold.sum())}/{T}", flush=True)

    # ---- the frozen codec, in JAX ---------------------------------------
    codec = codec_from_ckpt_jax(ck, C)
    load_pixelmae(ck["model"], codec)

    # ---- Z: the published cache, or an embedding built here -------------
    blk_kw = dict(blk_rows=(None if BLKA is None else BLKA.rows),
                  blk_pad=(None if BLKA is None else BLKA.pad))
    t_emb = time.time()
    if a.z:
        # A CACHE IS TRUSTED ONLY AFTER IT IS CHECKED. The name carries the
        # codec weight hash and the tensor sha, so a stale cache is a MISS
        # rather than a lie (#10/#11) — but the name is only as good as
        # whoever typed it, and the shape is the one thing this side can
        # verify against the run it is about to spend hours on.
        Z = np.load(a.z, mmap_mode="r")
        if Z.dtype != np.dtype(CACHE_DTYPE):
            print(f"::warning::{os.path.basename(a.z)} is {Z.dtype}, the "
                  f"published cache dtype is {np.dtype(CACHE_DTYPE)} — "
                  f"continuing (float32 caches predate the fp16 switch and "
                  f"are still valid), but say so in the run's doc string",
                  flush=True)
        want_T = (len(tsel) * a.time_stride + a.time_offset if tsel is not None
                  else T)
        if tsel is not None and Z.shape[0] == T:
            print(f"  --z is already strided ({T} rows) — used as given",
                  flush=True)
        elif Z.shape[0] not in (T, want_T):
            raise SystemExit(
                f"REFUSING: {a.z} has {Z.shape[0]} rows; this axis has {T} "
                f"(and the unstrided record {want_T}). A Z of the wrong "
                f"length is a Z of a different tensor, a different stride or "
                f"a different block mode, and every downstream shape check "
                f"would pass.")
        elif tsel is not None:
            Z = np.asarray(Z[tsel])
        if Z.shape[1] != P or Z.shape[2] != d_z:
            raise SystemExit(
                f"REFUSING: {a.z} is [{Z.shape[0]}, {Z.shape[1]}, "
                f"{Z.shape[2]}]; this run wants P={P}, d_z={d_z}.")
        print(f"  Z from the published cache {os.path.basename(a.z)} "
              f"{tuple(Z.shape)} {Z.dtype}", flush=True)
    else:
        print(f"embedding every (bin, ocean pixel) through the frozen codec "
              f"— {T * P:,} encoder forwards …", flush=True)
        Z, _c = embed_everything_jax(codec, Xt, OBS, codec_ctx, lats, lons,
                                     ys, xs, d_z, **blk_kw)
        print(f"  Z [T={T} P={P} d_z={d_z}] "
              f"({time.time() - t_emb:.0f}s)", flush=True)

    # ---- the per-pixel STATIC IDENTITY ----------------------------------
    # The codec embedding of the STATIC channels alone: the climatological
    # identity of the place, added to every window's static context.
    stat_obs = np.asarray(OBS[0]).copy()
    stat_obs[..., list(dynamic)] = False
    zs = []
    for i in range(0, P, 8192):
        sl = slice(i, min(i + 8192, P))
        n = sl.stop - sl.start
        ctx = np.concatenate([np.zeros((n, 2), np.float32), coords[sl]], 1)
        if BLKA is not None:
            kt = int(BLKA.k_max)
            vv = np.repeat(np.asarray(Xt[np.zeros(n, np.int64), ys[sl],
                                         xs[sl]], np.float32)[:, None, :],
                           kt, axis=1)
            oo = np.repeat(stat_obs[ys[sl], xs[sl]][:, None, :], kt, axis=1)
            oo = oo & (~BLKA.pad[0])[None, :, None]
            mk = np.zeros((n, kt, C), bool)
        elif int(getattr(codec, "patch", 1)) > 1:
            # One gather: values from bin 0 (statics are constant in t;
            # dynamics are zeroed inside encode because their obs is False),
            # obs from stat_obs with gather_px's own out-of-range latitude
            # masking — exactly what training-time encode saw. `stat_obs[None]`
            # stands in for OBS the way `ml/temporal.py:1745` passes it, and
            # every t index is 0, so the leading axis of length 1 is enough.
            vv, oo = gather_px_np(Xt, stat_obs[None], np.zeros(n, np.int64),
                                  ys[sl], xs[sl], int(codec.patch))
            vv = vv.astype(np.float32)
            mk = np.zeros((n, C), bool)
        else:
            vv = np.asarray(Xt[np.zeros(n, np.int64), ys[sl], xs[sl]],
                            np.float32)
            oo = stat_obs[ys[sl], xs[sl]]
            mk = np.zeros((n, C), bool)
        zs.append(np.asarray(codec.encode(jnp.asarray(vv), jnp.asarray(oo),
                                          jnp.asarray(mk), jnp.asarray(ctx))))
    Zstat = np.concatenate(zs, 0).astype(np.float32)

    # ---- E-022 geometry ---------------------------------------------------
    if a.stencil > 1:
        if _ring_on(a.ring_km):
            n_r = (1 if str(a.ring_km).startswith("spiral:")
                   else len([r for r in str(a.ring_km).split(",") if r.strip()]))
            if (a.stencil - 1) % n_r:
                raise SystemExit(
                    f"--stencil {a.stencil} gives {a.stencil - 1} ring slots, "
                    f"which does not divide among {n_r} radii "
                    f"({a.ring_km}). Use 1 + a multiple of {n_r}.")
        elif a.stencil not in STENCILS:
            raise SystemExit(
                f"--stencil {a.stencil} has no fixed-table entry (have "
                f"{sorted(STENCILS)}); pass --ring-km to place that many "
                f"slots on rings instead.")
        NBR = build_stencil(ocean.shape[0], ocean.shape[1], ys, xs, a.stencil,
                            ring_km=a.ring_km, lats=lats)
        obs_flags = (NBR >= 0).astype(np.float32)
        static_ctx = np.concatenate([Zstat, coords, obs_flags],
                                    1).astype(np.float32)
        print(f"stencil {a.stencil}"
              + (f" RING r={a.ring_km} km" if _ring_on(a.ring_km) else "")
              + f": input {a.stencil}x{d_z}+2 per step; "
              f"{int((NBR < 0).sum()):,} missing neighbour slots of "
              f"{NBR.size:,}", flush=True)
    else:
        NBR = None
        static_ctx = np.concatenate([Zstat, coords], 1).astype(np.float32)

    # ---- the head's season features --------------------------------------
    head_ctx = (season_ctx(months, a.season_phase, d) if BLKA is None
                else BLKA.head_season(a.season_phase))
    if a.season_phase == "month":
        assert np.array_equal(head_ctx, ctx_all), \
            "season_ctx('month') must reproduce the archived sin/cos(2pi*moy/12)"
    Mt = np.asarray(head_ctx, np.float32)

    # ---- the training pool -----------------------------------------------
    # Windows [t-K+1 .. t] whose TARGET bin t+1 is a train bin and whose pixel
    # is outside the pool's longitude holdout. Windows may LOOK at held-out
    # bins (persistence can too); they may never be SCORED on them.
    K = a.K
    ok_t = np.array([t + 1 < T and t + 1 >= K and not t_hold[t + 1]
                     for t in range(T)])
    ok_p = ~pool_x_hold[xs]
    pool_t, pool_p = np.where(ok_t[:, None] & ok_p[None, :])
    if not len(pool_t):
        raise SystemExit(
            f"REFUSING: the training pool is EMPTY at K={K}. {int(ok_t.sum())} "
            f"of {T} bins can end a window and {int(ok_p.sum())} of {P} pixels "
            f"survive --train-lon-hold {a.train_lon_hold!r}. A K longer than "
            f"the record's train stretch is the usual cause.")
    print(f"train windows: {len(pool_t):,}", flush=True)

    MILESTONES = {int(x) for x in a.milestone_steps.split(",") if x.strip()}
    dead = {m for m in MILESTONES if not (0 < m < a.steps)}
    if dead:
        # A milestone at or past --steps can never fire; refuse rather than
        # let a retention request silently retain nothing (§4.6).
        raise SystemExit(f"--milestone-steps {sorted(dead)} outside "
                         f"(0, {a.steps}) — those saves would never happen.")
    if MILESTONES:
        print(f"milestone checkpoints at steps {sorted(MILESTONES)}",
              flush=True)

    # ---- the model --------------------------------------------------------
    model = TemporalTransformer(d_z=d_z, d_model=a.d_model, n_heads=4,
                                n_layers=a.layers, k_max=K, direct=(),
                                stencil=a.stencil, eps_dim=a.fgn_eps,
                                rngs=nnx.Rngs(a.seed))
    # THE INITIALISATION IS TORCH'S OWN, AND IT IS NOT A STYLE CHOICE.
    #
    # Flax's defaults are not `nn.Module`'s. The one that matters most here is
    # the positional table: `nn.Embedding` initialises N(0, 1) and `nnx.Embed`
    # initialises at std ~ 1/sqrt(d_model) — 5.7x smaller at d_model 32 — so a
    # causal transformer whose only sense of WHERE IN THE WINDOW it is comes
    # from `pos` starts with that signal an order of magnitude weaker. Every
    # linear differs too (torch: U(±1/sqrt(fan_in)) weight AND bias; flax:
    # lecun-normal weight, zero bias).
    #
    # MEASURED, which is why this is here: at 300 toy steps on an identical Z,
    # an identical pool and an identical schedule, the flax-initialised head
    # read model/persistence 1.46, 1.87, 1.68 at seeds 0/1/2 against the torch
    # trainer's 0.71, 0.87, 0.77 — systematically ~2x worse at every seed, and
    # the gap is entirely the init. On a cross-framework TWIN that difference
    # would have been read as "the framework", which is precisely the box
    # effect ml/CLAUDE.md §3b warns about, manufactured by a default nobody
    # chose.
    #
    # So a fresh head is initialised by CONSTRUCTING THE TORCH MODULE under
    # `torch.manual_seed(seed)` and converting it. That is stronger than
    # matching the distributions: it matches the STREAM, so a JAX run and a
    # torch run at the same seed begin from bit-identical weights and init
    # stops being a variable at all. The cost is one transient torch module on
    # the host (~0.9 GB at 206M) and a torch wheel this driver already needs.
    _t0i = time.time()
    from temporal import TemporalTransformer as _TorchTemporal
    torch.manual_seed(a.seed)
    _tmod = _TorchTemporal(d_z=d_z, d_model=a.d_model, n_heads=4,
                           n_layers=a.layers, k_max=K, direct=(),
                           stencil=a.stencil, eps_dim=a.fgn_eps)
    load_temporal(_tmod.state_dict(), model)
    del _tmod
    print(f"  init: torch's own (constructed under torch.manual_seed"
          f"({a.seed}) and converted, so a same-seed torch run starts from "
          f"bit-identical weights) — {time.time() - _t0i:.1f}s", flush=True)
    if FGN:
        # The film weights arrive from the torch module ALREADY at exact
        # zeros (nn.init.zeros_ on both weight and bias), so the converted
        # head is the deterministic incumbent at step 0 in both backends —
        # the "r_fore reads exactly 1.000000 at step 1" twin, for free.
        print(f"FGN head: eps ~ N(0,1)^{a.fgn_eps} -> eps_embed -> per-layer "
              f"FiLM on norm1/norm2 (zero-init, so this IS the deterministic "
              f"head at step 0), objective = fair CRPS at N=2, monitor "
              f"ensemble M={a.fgn_val_members}. eps stream: "
              f"jax.random.PRNGKey({a.seed} * 1000003 + 57) folded by (step, "
              f"forward) — counter-based, so resume needs no saved RNG state; "
              f"eval bank from PRNGKey({a.seed} * 1000003 + 58).", flush=True)
    graphdef, state = nnx.split(model)
    n_par2 = sum(int(np.prod(v.shape)) for v in _leaves(state))
    print(f"stage-2 head: {n_par2 / 1e6:.3f}M parameters "
          f"(d_model {a.d_model} x {a.layers} layers, K {K}, stencil "
          f"{a.stencil}, d_z {d_z})", flush=True)

    # ---- optimiser --------------------------------------------------------
    # AdamW(lr, weight_decay=1e-4) — torch's decoupled decay and optax's are
    # the same update. inject_hyperparams makes the learning rate a traced
    # value set from the host each step, so the schedule needs no re-jit.
    #
    # OFF ADDS NO TRANSFORM. Not `clip_by_global_norm(inf)`, which is a no-op
    # only for finite norms; not a "very large" threshold, which is a promise
    # about a distribution nobody has measured. The default is the chain every
    # unclipped arm trains under, unchanged.
    tx = optax.inject_hyperparams(optax.adamw)(learning_rate=a.lr,
                                               weight_decay=1e-4)
    CLIP = float(a.grad_clip)
    if CLIP > 0:
        tx = optax.chain(optax.clip_by_global_norm(CLIP), tx)
    opt_state = tx.init(state)

    # Every jitted training function of this run, built in ONE place
    # (`build_train_steps`, which also carries the E-054b accumulation and the
    # memory argument for it). At ACCUM == 1 it builds exactly the four
    # functions this block used to define inline and nothing else.
    _steps = build_train_steps(graphdef, tx, make_set_lr(CLIP > 0), d_z,
                               float(a.input_znoise), accum=ACCUM)
    train_step = _steps.train_step
    train_step_dn = _steps.train_step_dn
    train_step_fgn = _steps.train_step_fgn
    train_step_fgn_dn = _steps.train_step_fgn_dn

    @jax.jit
    def _eval_forward_det(st, zseq, mseq, sctx):
        m = nnx.merge(graphdef, st)
        return m(zseq, mseq, sctx)

    @jax.jit
    def eval_forward_eps(st, zseq, mseq, sctx, eps):
        m = nnx.merge(graphdef, st)
        return m(zseq, mseq, sctx, eps=eps)

    def eval_forward(st, zseq, mseq, sctx):
        """The POINT read-out, deterministic when the head is and the
        REPRESENTATIVE member when it is an FGN head — `ml/temporal.py:
        eval_forward` (:1728-1735). One function so the member choice lives in
        one place (`fgn_eval_eps`), which is what stops two call sites quietly
        disagreeing about which member the legacy pooled instruments saw.

        Every pre-E-057 read-out downstream of this — the val curve's point
        forward, eval 1's z_t+1, eval 2's channel decode, the in-training
        section probe — is a POINT instrument written for a head with one
        output per input. An FGN head has no single output, so ε = 0 (the
        distribution's centre, and at init exactly the legacy computation) is
        chosen once, here, and recorded as `fgn_eval_eps: "zeros"`."""
        if not FGN:
            return _eval_forward_det(st, zseq, mseq, sctx)
        return eval_forward_eps(st, zseq, mseq, sctx,
                                fgn_eval_eps(zseq.shape[0], a.fgn_eps))

    # ---- device sharding --------------------------------------------------
    # Data parallelism and nothing cleverer: the batch axis is sharded across
    # local devices, the parameters replicated. On one device this whole block
    # is a no-op, which is every CPU test — the point of writing it this way
    # is that the correctness path does not branch.
    shard = None
    if a.shard_batch == "auto" and len(devices) > 1:
        if a.batch % len(devices):
            raise SystemExit(f"--batch {a.batch} is not divisible by the "
                             f"{len(devices)} local devices; a ragged shard "
                             f"would change the batch each device sees.")
        # It is the MICRO-batch that is sharded under --grad-accum, so it is
        # the micro-batch that has to divide. Checking only the full batch
        # would let batch 256 / accum 2 = 128 pass on a host with 3 devices
        # and hand each device a different number of windows.
        if MICRO % len(devices):
            raise SystemExit(
                f"--batch {a.batch} / --grad-accum {ACCUM} = {MICRO} windows "
                f"per micro-batch is not divisible by the {len(devices)} "
                f"local devices; the micro-batch is what gets sharded, and a "
                f"ragged shard would change the batch each device sees.")
        mesh = jax.make_mesh((len(devices),), ("b",))
        shard = jax.sharding.NamedSharding(mesh,
                                           jax.sharding.PartitionSpec("b"))
        rep = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())
        state = jax.device_put(state, rep)
        opt_state = jax.device_put(opt_state, rep)
        print(f"data-parallel over {len(devices)} devices: batch {a.batch} -> "
              f"{a.batch // len(devices)} per device (NOT exercised by the "
              f"CPU tests)", flush=True)

    def put(x, sharded=True):
        # Shard HOST data directly. `jnp.asarray(x)` first would commit the
        # WHOLE array to device 0 and then re-slice it there — for the fixed
        # monitoring batch at K=144 that is a 10.95 GB staging copy plus a
        # 2.55 GiB shard on a 16 GB chip, which is exactly how the first
        # e051 node died (RESOURCE_EXHAUSTED at 563.88M free, 2026-08-25).
        # device_put from a numpy array slices on the host and transfers each
        # shard alone; no full-size device allocation ever exists.
        if shard is not None and sharded:
            return jax.device_put(np.asarray(x), shard)
        return jnp.asarray(x)

    # ---- resume -----------------------------------------------------------
    metrics_path = os.path.join(a.out, "metrics.jsonl")

    def m2(rec):
        try:
            with open(metrics_path, "a") as f:
                f.write(json.dumps(rec) + "\n")
        except OSError:
            pass                      # instrumentation never breaks the run

    start_step = 0
    if a.resume and os.path.exists(a.resume):
        if a.resume.endswith(".npz"):
            state, opt_state, start_step, r_args = load_state_npz(
                a.resume, state, opt_state)
            for kk in ("K", "d_model", "layers", "stencil", "ring_km"):
                if kk in r_args and str(r_args[kk]) != str(getattr(a, kk)):
                    raise SystemExit(
                        f"REFUSING to resume: {kk} is {getattr(a, kk)!r} here "
                        f"and {r_args[kk]!r} in {a.resume}. The leaf shapes "
                        f"can still match (they do for ring_km and for a "
                        f"K below k_max), so this is checked by NAME.")
            if start_step >= a.steps:
                raise SystemExit(
                    f"--resume: the checkpoint is at step {start_step:,} and "
                    f"--steps is {a.steps:,}. --steps is the TOTAL, not the "
                    f"extra.")
            print(f"  RESUMED from {a.resume} at step {start_step:,} "
                  f"(optimiser + schedule position restored); training on to "
                  f"{a.steps:,}", flush=True)
            m2({"stage2_resumed": {"from": os.path.basename(a.resume),
                                   "at_step": start_step, "to_step": a.steps,
                                   "lr": a.lr, "backend": "jax"}})
        else:
            tk = torch.load(a.resume, map_location="cpu", weights_only=False)
            m = nnx.merge(graphdef, state)
            load_temporal(tk["model"], m)
            _, state = nnx.split(m)
            opt_state = tx.init(state)
            print(f"  WARM-STARTED from {a.resume}: weights only, no "
                  f"optimiser and no step — the LR schedule restarts from 0. "
                  f"Report this run as a warm start, NOT a continuation "
                  f"(mapping torch Adam state into optax is out of scope, "
                  f"JAX_PORT.md §3.3).", flush=True)
            m2({"stage2_warm_restart": {
                "from": os.path.basename(a.resume),
                "parent_steps": int(tk.get("args", {}).get("steps", 0) or 0),
                "extra_steps": a.steps, "lr": a.lr, "inherited": ["model"],
                "reset": ["optimiser moments", "schedule position",
                          "rng stream"],
                "kind": "warm restart, NOT a continuation", "backend": "jax"}})
    elif a.require_resume:
        raise SystemExit(
            f"--require-resume: no checkpoint at {a.resume!r}. Exiting in "
            f"seconds rather than training a fresh head for hours under a doc "
            f"string that claims to be a continuation.")
    elif a.resume:
        print(f"  --resume {a.resume}: NOT FOUND — starting from scratch "
              f"(this is not an error, but the run is now a fresh one; say so "
              f"in its doc string)", flush=True)

    # ---- the metrics config line -----------------------------------------
    m2({"stage2_config": {
        "d_model": a.d_model, "layers": a.layers, "K": K, "steps": a.steps,
        "params_M": round(n_par2 / 1e6, 3), "batch": a.batch,
        "train_windows": int(len(pool_t)), "d_z": d_z, "seed": a.seed,
        "unroll": 1, "stencil": a.stencil, "ring_km": a.ring_km,
        "unroll_probs": "", "direct": "",
        "input_znoise": a.input_znoise, "grad_clip": a.grad_clip,
        # E-054b · UNCONDITIONALLY, unlike the fgn keys below, and the
        # asymmetry is deliberate. `grad_accum: 1` is a MEASUREMENT — it says
        # this run's batch went through the device in one piece — and `batch`
        # is right beside it, so a reader can only interpret one with the
        # other. Absence would be readable as "before the flag existed" on
        # exactly the runs where the question matters.
        "grad_accum": ACCUM, "micro_batch": MICRO,
        "train_lon_hold": a.train_lon_hold,
        "codec_holdout_lon": ck["args"].get("holdout_lon", ""),
        "tag": a.tag or "",
        # The three fields a reader of a JAX curve needs and a torch curve
        # never had. ml/CLAUDE.md §3b makes a JAX/TPU-trained number a NEW
        # TIER; a config line that did not say which backend produced it
        # would be the one place that fact could go missing.
        "backend": "jax", "k_time": k_time,
        "noise_backend": a.noise_backend, "gather_fp16": bool(a.gather_fp16),
        "gather_workers": int(a.gather_workers), "prefetch": int(a.prefetch),
        "lr": a.lr, "lr_schedule": a.lr_schedule,
        "lr_halflife": a.lr_halflife, "lr_cooldown_frac": a.lr_cooldown_frac,
        "lr_warmup": a.lr_warmup,
        "time_stride": a.time_stride, "time_offset": a.time_offset,
        "season_phase": a.season_phase,
        "codec": os.path.basename(a.ckpt),
        "z": os.path.basename(a.z) if a.z else None,
        "data": os.path.basename(a.data),
        # E-057 · NEW KEYS, AND ONLY WHEN THE ARM IS ONE, with the SAME NAMES
        # ml/temporal.py:3285-3290 writes. Adding `fgn_eps: 0` to every
        # legacy record would be a changed record with the flag off; a reader
        # that predates E-057 ignores an extra key, and one that postdates it
        # reads absence as "not an fgn run".
        **({"stage2_loss_kind": "crps2",
            "fgn_eps": a.fgn_eps,
            "fgn_val_members": a.fgn_val_members,
            # WHICH MEMBER the legacy point read-outs saw — recorded, never
            # inferred (fgn_eval_eps()).
            "fgn_eval_eps": "zeros"} if FGN else {})}})

    # ---- the fixed monitoring batch --------------------------------------
    # Windows whose t+1 target is a HELD-OUT bin — the same population the
    # final z_t+1 eval scores, sampled ~100 times during training instead.
    # Monitoring only; nothing is selected on it.
    ev_m = np.array([t + 1 < T and t + 1 >= K and t_hold[t + 1]
                     for t in range(T)])
    emt, emp = np.where(ev_m[:, None] & np.ones(P, bool)[None, :])
    if not len(emt):
        raise SystemExit(
            "REFUSING: no window has a HELD-OUT target bin, so there is no "
            "val curve and no z_t+1 eval. Check --holdout-years on the codec "
            "and K against the record length.")
    _mr = np.random.default_rng(12345)
    msel = _mr.choice(len(emt), min(4096, len(emt)), replace=False)
    emt, emp = emt[msel], emp[msel]
    _mb = emt - K + 1
    mon_zseq = put(gather_stencil_np(Z, _mb, emp, NBR, K))
    mon_mseq = put(np.stack([Mt[_mb + j] for j in range(K)], 1))
    mon_sctx = put(static_ctx[emp])
    mon_ztrue = np.asarray(Z[emt + 1, emp], np.float32)
    mon_pers = float(np.mean((np.asarray(Z[emt, emp], np.float32)
                              - mon_ztrue) ** 2))
    mon_ztrue_j = put(mon_ztrue)

    # ---- E-057 · the FIXED eval eps bank ---------------------------------
    # M members, drawn ONCE from their OWN root key (seed*1000003 + 58, not
    # the training root), so the monitoring ensemble is the same M members at
    # every log point and the number of monitor calls cannot perturb the
    # training stream. Each forward BROADCASTS one member across the whole
    # monitoring batch, which is FGN's convention: eps is GLOBAL — one draw
    # for the whole field, never one per pixel.
    eps_val = None
    if FGN:
        eps_val = fgn_val_bank(a.seed, a.fgn_val_members, a.fgn_eps)
        print(f"fgn monitor: {a.fgn_val_members} fixed eval members from "
              f"PRNGKey({a.seed} * 1000003 + 58)", flush=True)

    # REPORT THE SCALE THE INPUT NOISE IS ACTUALLY BEING APPLIED AT.
    # `--input-znoise` is an ABSOLUTE sigma in whatever units the frozen
    # codec's encoder emits, and #423 carried the monthly anchor's 0.7 to a
    # codec whose z-space is 2.63x larger without one line of any record
    # saying so. Both ratios, always, never one without the other.
    _zrms = float(np.sqrt(np.mean(mon_ztrue ** 2)))
    _zref = float(np.sqrt(max(mon_pers, 1e-12)))
    _zrel = float(a.input_znoise) / _zref
    _zrelz = float(a.input_znoise) / max(_zrms, 1e-12)
    print(f"z-space scale (per component, over {len(msel)} held-out windows): "
          f"RMS |z| {_zrms:.5f} · RMS one-step change sqrt(val_persistence) "
          f"{_zref:.5f}", flush=True)
    if a.input_znoise > 0:
        print(f"input noise: --input-znoise {a.input_znoise:g} is an ABSOLUTE "
              f"sigma, = {_zrel:.5f} x sqrt(val_persistence) = {_zrelz:.5f} x "
              f"RMS |z|. The monthly anchor this constant was tuned on reads "
              f"0.39788 x sqrt(val_persistence). A very different figure means "
              f"the perturbation is NOT the one that was measured, only the "
              f"same number.", flush=True)
    m2({"stage2_monitor": {"n_windows": int(len(msel)),
                           "val_persistence": round(mon_pers, 5),
                           "z_rms": round(_zrms, 5),
                           "input_znoise_sigma": round(float(a.input_znoise),
                                                       5),
                           "input_znoise_rel_pers": round(_zrel, 5),
                           "input_znoise_rel_zrms": round(_zrelz, 5)}})

    # ---- the in-training transport probe ---------------------------------
    # `ml/temporal.py:2181`'s `stage2_probe.rapid_r_deseas`: hidden(-1) POOLED
    # along the 26.5N section, then the same deseasonalised ridge. It is a
    # SPATIALLY POOLED read-out and `ml/CLAUDE.md` §3 distrusts it as a level
    # — it is emitted because it is the curve Chris reads during a run and
    # because 95 archived bundles carry it, never as a verdict.
    _psec = None
    try:
        _pridx = rapid_arr[:, 0].astype(int)
        _prv = rapid_arr[:, 1].copy()
        _prmoy = moy[_pridx]
        _ptr = ~t_hold[_pridx]
        _pclim = np.array([_prv[_ptr & (_prmoy == mm)].mean()
                           for mm in range(12)])
        _prv_des = _prv - _pclim[_prmoy]
        _, _psec = rapid_section(lats, lons, ys, xs)
        _psec = np.asarray(_psec)
        _pok = _pridx >= K - 1
        _psec_ctx = put(static_ctx[_psec], sharded=False)
        if not len(_psec) or not _pok.any():
            _psec = None
    except Exception as _e:                 # monitoring never breaks a run
        print(f"  (in-training probe disabled: {_e})", flush=True)
        _psec = None

    def section_features():
        """[T, d_model] — hidden(-1) meaned over the section, per bin."""
        F = np.zeros((T, a.d_model), np.float32)
        for t_ in range(K - 1, T):
            b_ = t_ - K + 1
            zs_ = gather_stencil_np(Z, np.full(len(_psec), b_), _psec, NBR, K)
            ms_ = np.broadcast_to(np.stack([Mt[b_ + j] for j in range(K)],
                                           0)[None], (len(_psec), K, 2))
            _, hd_ = eval_forward(state, put(zs_, sharded=False),
                                  put(np.ascontiguousarray(ms_),
                                      sharded=False), _psec_ctx)
            F[t_] = np.asarray(hd_[:, -1]).mean(0)
        return F

    # ---- checkpoints ------------------------------------------------------
    suffix = f"_{a.tag}" if a.tag else ""
    ckpt_tag = os.environ.get("CKPT_TAG", "")

    def head_args(step):
        """`vars(a)` in the shape `ml/rollout_spatial.py` reads back.

        It re-derives K, d_model, layers, stencil, ring_km, seed, direct,
        unroll and season_phase from exactly this dict, and rebuilds the head
        with n_heads=4 — so every one of those keys must be present and must
        mean what it means on the torch side.
        """
        out = dict(vars(a))
        out["backend"] = "jax"
        out["step_recorded"] = int(step)
        out["input_quant"] = ""        # never trained here; the roll reads it
        out["unroll"] = 1
        out["direct"] = ""
        return out

    def save_ckpt(step, milestone=False):
        """Both artefacts, because they answer different questions: the
        `.npz` is what THIS trainer resumes from (optimiser moments included)
        and the `.pt` is what the UNCHANGED torch eval ladder rolls."""
        args = head_args(step)
        m = nnx.merge(graphdef, state)
        if milestone:
            export_temporal_pt(
                m, args, path=os.path.join(a.out, f"temporal_ms{step}.pt"),
                step=int(step), run_number=os.environ.get("GITHUB_RUN_NUMBER"))
            return
        save_state_npz(os.path.join(a.out, f"temporal{suffix}_jax.npz"),
                       state, opt_state, step, args)
        export_temporal_pt(
            m, args, path=os.path.join(a.out, f"temporal{suffix}.pt"),
            step=int(step), run_number=os.environ.get("GITHUB_RUN_NUMBER"),
            tag=ckpt_tag)

    # ---- train ------------------------------------------------------------
    print(f"training the temporal stage … ({n_par2:,} parameters, "
          f"{a.lr_schedule} schedule)", flush=True)
    if CLIP > 0:
        print(f"gradient clipping ON: max_norm {CLIP:g}, applied to the "
              f"global norm before the AdamW update. stage2_grad_norm keeps "
              f"reporting the PRE-clip norm; stage2_grad_norm_max, "
              f"stage2_grad_clip_frac and stage2_grad_nonfinite report over "
              f"each {max(1, a.steps // 100)}-step window.", flush=True)
    else:
        print("gradient clipping OFF (--grad-clip 0): no clip transform is "
              "added to the optax chain at all.", flush=True)
    if ACCUM > 1:
        print(f"gradient accumulation ON: --grad-accum {ACCUM} · batch "
              f"{a.batch} = {ACCUM} x {MICRO} windows, ONE AdamW update per "
              f"reported step on the AVERAGED gradient. The step is the "
              f"batch-{a.batch} step (equal micro-batches, mean loss); "
              f"stage2_zmse is the mean over the effective batch and "
              f"stage2_grad_norm the PRE-clip norm of the average.",
              flush=True)
    t0 = time.time()
    log_every = max(1, a.steps // 100)     # ~100 curve points, as stage 1 does
    probe_every = max(1, a.steps // 10)    # the transport curve, 10 points
    ck_every = a.ckpt_every or log_every
    cl_max, cl_hit, cl_bad, cl_n = 0.0, 0, 0, 0
    S = a.stencil
    gn = float("nan")

    host_noise = a.input_znoise > 0 and a.noise_backend == "host"
    device_noise = a.input_znoise > 0 and a.noise_backend == "device"
    _nz_root = jax.random.PRNGKey(a.seed) if device_noise else None
    # E-057 · the TRAINING eps stream's root. Nothing else is kept: eps at
    # (step, forward) is a pure fold of this root, so the step counter already
    # in the .npz is the whole of the stream's state (see fgn_eps_at).
    _fgn_root = fgn_train_key(a.seed) if FGN else None

    def make_batches():
        """One generator, one RNG, drawn in step order — the SEQUENCE of
        batches is identical whether it runs inline (--prefetch 0) or in the
        producer thread, because this function is the only thing that touches
        `rng` once training starts."""
        for s_ in range(start_step + 1, a.steps + 1):
            k = rng.integers(0, len(pool_t), a.batch)
            t_i, p_i = pool_t[k], pool_p[k]
            base = t_i - K + 1
            zseq = gather_stencil_np(Z, base, p_i, NBR, K,
                                     cast32=not a.gather_fp16,
                                     workers=a.gather_workers)
            if host_noise:
                noise = rng.standard_normal(
                    (a.batch, K, S, d_z)).astype(np.float32)
                zseq = apply_znoise(zseq, noise, a.input_znoise, d_z)
            mseq = np.stack([Mt[base + j] for j in range(K)], 1)
            ztgt = np.asarray(
                Z[base[:, None] + np.arange(1, K + 1)[None, :], p_i[:, None]],
                np.float32)
            yield s_, zseq, mseq, static_ctx[p_i], ztgt

    if a.prefetch > 0:
        import queue as _queue
        import threading as _threading
        _q = _queue.Queue(maxsize=a.prefetch)
        _DONE = object()

        def _produce():
            try:
                for item in make_batches():
                    _q.put(item)
                _q.put(_DONE)
            except BaseException as e:       # surface, never deadlock
                _q.put(e)

        _threading.Thread(target=_produce, daemon=True).start()
        print(f"batch prefetch ON: depth {a.prefetch}, gather workers "
              f"{a.gather_workers or 1}, noise on {a.noise_backend}, "
              f"transfer {'fp16' if a.gather_fp16 else 'fp32'}", flush=True)

        def _stream():
            while True:
                item = _q.get()
                if item is _DONE:
                    return
                if isinstance(item, BaseException):
                    raise item
                yield item
        batch_iter = _stream()
    else:
        batch_iter = make_batches()

    for s, zseq, mseq, sctx_b, ztgt in batch_iter:
        lr_now = jnp.asarray(a.lr * lr_factor(s - 1, a), jnp.float32)
        if ACCUM > 1:
            state, opt_state, loss, gnorm = accum_step(
                _steps, state, opt_state, lr_now, s,
                zseq, mseq, sctx_b, ztgt, MICRO, put=put,
                fgn_root=_fgn_root, fgn_eps=a.fgn_eps, nz_root=_nz_root)
        elif FGN:
            # TWO INDEPENDENT eps ON THE IDENTICAL CONTEXT. Folded from
            # (seed, step, forward_index) and from nothing else, so a resumed
            # run at step s draws the same pair the original did — no RNG
            # state in the checkpoint (fgn_eps_at explains why that is exact
            # for a counter-based PRNG and not merely likely).
            _nb = zseq.shape[0]
            _e1 = fgn_eps_at(_fgn_root, s, 0, _nb, a.fgn_eps)
            _e2 = fgn_eps_at(_fgn_root, s, 1, _nb, a.fgn_eps)
            if device_noise:
                state, opt_state, loss, gnorm = train_step_fgn_dn(
                    state, opt_state, lr_now, jax.random.fold_in(_nz_root, s),
                    put(zseq), put(mseq), put(sctx_b), put(ztgt), _e1, _e2)
            else:
                state, opt_state, loss, gnorm = train_step_fgn(
                    state, opt_state, lr_now,
                    put(zseq), put(mseq), put(sctx_b), put(ztgt), _e1, _e2)
        elif device_noise:
            state, opt_state, loss, gnorm = train_step_dn(
                state, opt_state, lr_now, jax.random.fold_in(_nz_root, s),
                put(zseq), put(mseq), put(sctx_b), put(ztgt))
        else:
            state, opt_state, loss, gnorm = train_step(
                state, opt_state, lr_now,
                put(zseq), put(mseq), put(sctx_b), put(ztgt))
        gv = float(gnorm)
        lv = float(loss)
        if not np.isfinite(lv):
            m2({"stage2_step": s, "diverged": {"loss": lv, "grad_norm": gv}})
            raise SystemExit(
                f"ABORTING at step {s}: loss is {lv}. The head has gone "
                f"non-finite; every further step writes NaN into the weights. "
                f"Suspect the learning rate first"
                + ("" if CLIP > 0 else " (there is no gradient clipping on "
                                       "this run — --grad-clip 0)") + ".")
        # §5.22, NEVER WRITE NaN INTO A RESULTS FILE. A non-finite norm is NOT
        # a big number: it is a different event, so it gets its own counter
        # and is kept out of the max and out of the rate.
        if np.isfinite(gv):
            cl_max = max(cl_max, gv)
            cl_hit += int(gv > CLIP) if CLIP > 0 else 0
        else:
            cl_bad += 1
        cl_n += 1
        gn = gv

        if s % log_every == 0 or s == a.steps:
            _fgn_val = {}
            if FGN:
                # E-057 · THE ENSEMBLE READ, chunked (see fgn_monitor_ens).
                ens = fgn_monitor_ens(
                    lambda z_, m_, c_, e_: eval_forward_eps(state, z_, m_,
                                                            c_, e_),
                    mon_zseq, mon_mseq, mon_sctx, eps_val)
                val_mse, amp, _fgn_val = fgn_val_metrics(ens, mon_ztrue_j)
            else:
                mp_, _ = eval_forward(state, mon_zseq, mon_mseq, mon_sctx)
                mlast = mp_[:, -1]
                val_mse = float(jnp.mean((mlast - mon_ztrue_j) ** 2))
                amp = float(jnp.std(mlast) / (jnp.std(mon_ztrue_j) + 1e-9))
            rec = {"stage2_step": s, "stage2_zmse": round(lv, 5),
                   "stage2_loss_base": round(lv, 5),
                   "stage2_val_zmse": round(val_mse, 5),
                   "stage2_amp": round(amp, 4),
                   "stage2_grad_norm": round(gn, 4),
                   "stage2_lr": float(a.lr * lr_factor(s - 1, a)),
                   "stage2_wall_s": round(time.time() - t0, 1)}
            # E-057 · the ensemble keys, BESIDE the existing ones and never
            # instead of them. §5.22: NEVER WRITE NaN INTO A RESULTS RECORD —
            # a non-finite value omits its key and says so on stderr, because
            # an absent key cannot be mistaken for a measurement and a NaN can.
            for _k, _v in _fgn_val.items():
                if _v == _v and abs(_v) != float("inf"):
                    # SIGNIFICANT digits, not decimal places: the collapse
                    # telemetry is read near ZERO, and `round(v, 6)` prints a
                    # member variance of 1e-8 as exactly 0.0 — i.e. it
                    # destroys the resolution precisely where the failure mode
                    # lives.
                    rec[_k] = float(f"{float(_v):.6g}")
                else:
                    print(f"::warning::{_k} was non-finite at step {s} "
                          f"({_v}) — key omitted from this record rather than "
                          f"written as NaN", flush=True)
            if CLIP > 0:
                rec["stage2_grad_clip"] = CLIP
                rec["stage2_grad_norm_max"] = round(cl_max, 4)
                rec["stage2_grad_clip_frac"] = round(cl_hit / max(1, cl_n), 4)
                rec["stage2_grad_nonfinite"] = int(cl_bad)
            else:
                # The window max is free here (the norm is computed every
                # step either way) and it is what says whether an UNCLIPPED
                # run is near the regime a clip would have caught.
                rec["stage2_grad_norm_max"] = round(cl_max, 4)
                rec["stage2_grad_nonfinite"] = int(cl_bad)
            cl_max, cl_hit, cl_bad, cl_n = 0.0, 0, 0, 0
            m2(rec)
            if _psec is not None and (s % probe_every == 0 or s == a.steps):
                try:
                    F_ = section_features()
                    ri_ = _pridx[_pok]
                    r_, _ = ridge_r(F_[ri_], _prv_des[_pok], ~t_hold[ri_],
                                    t_hold[ri_])
                    m2({"stage2_probe": {"step": s,
                                         "rapid_r_deseas": round(float(r_), 4)}})
                except Exception as _e:
                    print(f"  (in-training probe failed at {s}: {_e})",
                          flush=True)
        if s % ck_every == 0 or s == a.steps:
            save_ckpt(s)
        if s in MILESTONES:
            save_ckpt(s, milestone=True)
            print(f"  milestone checkpoint saved: temporal_ms{s}.pt",
                  flush=True)
        if s % max(1, a.steps // 10) == 0:
            print(f"  step {s:>6}/{a.steps}  z-mse {lv:.4f}  "
                  f"({time.time() - t0:.0f}s)", flush=True)

    # ---- eval 1: z-space t+1 on held-out target bins ---------------------
    results = {"run": os.path.basename(a.out), "K": K, "d_model": a.d_model,
               "layers": a.layers, "steps": a.steps, "backend": "jax"}
    if FGN:
        # `stage2_result` gains `fgn_eps` and nothing else changes in it —
        # ml/temporal.py:4220, same key, same condition.
        results["fgn_eps"] = a.fgn_eps
    results["scale"] = {"params": int(n_par2), "batch": int(a.batch),
                        "steps": int(a.steps),
                        "data_points": int(len(pool_t)), "n_pixels": int(P),
                        "n_train_months": int((~t_hold).sum()),
                        "stencil": int(a.stencil), "ring_km": a.ring_km}
    et, ep = np.where(ev_m[:, None] & np.ones(P, bool)[None, :])
    sel = np.random.default_rng(a.seed).choice(len(et),
                                               min(20000, len(et)),
                                               replace=False)
    et, ep = et[sel], ep[sel]
    ebase = et - K + 1
    zhat = []
    EV = 4096
    for i0 in range(0, len(et), EV):
        sl = slice(i0, min(i0 + EV, len(et)))
        zs_ = gather_stencil_np(Z, ebase[sl], ep[sl], NBR, K)
        ms_ = np.stack([Mt[ebase[sl] + j] for j in range(K)], 1)
        pr_, _ = eval_forward(state, put(zs_, sharded=False),
                              put(ms_, sharded=False),
                              put(static_ctx[ep[sl]], sharded=False))
        zhat.append(np.asarray(pr_[:, -1]))
    zhat = np.concatenate(zhat)
    ztrue = np.asarray(Z[et + 1, ep], np.float32)
    zlast = np.asarray(Z[et, ep], np.float32)
    results["z_t+1"] = {
        "mse_model": float(np.mean((zhat - ztrue) ** 2)),
        # PERSISTENCE IS z_t FROZEN, computed on the identical windows and the
        # identical held-out draw — the same quantity ml/temporal.py:2707
        # reports, so the RATIO this programme argues from is comparable.
        "mse_persistence": float(np.mean((zlast - ztrue) ** 2))}
    results["z_t+1"]["beats_persistence"] = bool(
        results["z_t+1"]["mse_model"] < results["z_t+1"]["mse_persistence"])

    # ---- eval 2: decode ẑ through the frozen codec → channel space -------
    if BLKA is not None:
        results["chan_t+1"] = {"skipped": (
            f"block codec (k_time {k_time}): decoding z to channel space "
            f"names a CELL, and which cell stands for the block is not yet "
            f"decided. z_t+1 above is unaffected.")}
    else:
        qc = np.tile(np.arange(C, dtype=np.int32), (min(EV, len(et)), 1))
        off0 = np.zeros((min(EV, len(et)), C, 3), np.int32)
        xh = []
        for i0 in range(0, len(et), EV):
            sl = slice(i0, min(i0 + EV, len(et)))
            n = sl.stop - sl.start
            xh.append(np.asarray(codec.query(jnp.asarray(zhat[sl]),
                                             jnp.asarray(qc[:n]),
                                             jnp.asarray(off0[:n]))))
        xhat = np.concatenate(xh)
        v1 = np.asarray(Xt[et + 1, ys[ep], xs[ep]], np.float32)
        o1 = np.asarray(OBS[et + 1, ys[ep], xs[ep]])
        v0 = np.asarray(Xt[et, ys[ep], xs[ep]], np.float32)
        o0 = np.asarray(OBS[et, ys[ep], xs[ep]])
        dyn = np.zeros(C, bool)
        dyn[list(dynamic)] = True
        both = o0 & o1 & dyn[None, :]
        nb = max(int(both.sum()), 1)
        mse_m = float(((xhat - v1) ** 2 * both).sum() / nb)
        mse_p = float(((v0 - v1) ** 2 * both).sum() / nb)
        results["chan_t+1"] = {"mse_model": mse_m, "mse_persistence": mse_p,
                               "beats_persistence": mse_m < mse_p,
                               "channels": [chan[c] for c in dynamic]}

    # ---- eval 3: the RAPID probe from the head's hidden state -------------
    if _psec is not None:
        try:
            F = section_features()
            ri = _pridx[_pok]
            tr, te = ~t_hold[ri], t_hold[ri]
            r_raw, _ = ridge_r(F[ri], _prv[_pok], tr, te)
            r_des, _ = ridge_r(F[ri], _prv_des[_pok], tr, te)
            results["rapid_probe"] = {
                "r_raw": float(r_raw), "r_deseasonalised": float(r_des),
                "n_test": int(te.sum()),
                "features": "hidden(-1) mean over section",
                "note": ("SPATIALLY POOLED (ml/CLAUDE.md §3): emitted for "
                         "continuity with the archive, never as a verdict.")}
        except Exception as e:                                # noqa: BLE001
            # NOT fatal, and it says why: this runs at the very end of a job
            # that may have spent a day. But it must never write a NaN either
            # (§5.22) — the key is simply absent.
            print(f"::warning::rapid probe failed: {type(e).__name__}: {e}",
                  flush=True)
    results["rapid_probe_kfold"] = {"skipped": (
        "the year-blocked head k-fold imports probe_kfold, which is the eval "
        "ladder this port deliberately does not carry (JAX_PORT.md tier 3b). "
        "Score the exported temporal.pt through the UNCHANGED torch ladder.")}

    print(json.dumps(results, indent=2), flush=True)
    m2({"stage2_result": {
        "d_model": a.d_model, "layers": a.layers, "K": K, "steps": a.steps,
        "params_M": round(n_par2 / 1e6, 3), "seed": a.seed, "tag": a.tag or "",
        "backend": "jax",
        "z_mse_model": results.get("z_t+1", {}).get("mse_model"),
        "z_mse_persistence": results.get("z_t+1", {}).get("mse_persistence"),
        "chan_mse_model": results.get("chan_t+1", {}).get("mse_model"),
        "chan_mse_persistence": results.get("chan_t+1",
                                            {}).get("mse_persistence"),
        "rapid_r_deseas": results.get("rapid_probe", {}).get(
            "r_deseasonalised"),
        "rapid_r_raw": results.get("rapid_probe", {}).get("r_raw"),
        # ABSENT, not null-with-a-number: the k-fold is not computed here and
        # a key carrying None is what a reader can mistake for a measurement.
        "z_direct_ratio": None,
        "scale": results.get("scale"),
        **({"fgn_eps": a.fgn_eps} if FGN else {})}})
    save_ckpt(a.steps)
    json.dump(results, open(os.path.join(a.out, f"temporal{suffix}.json"),
                            "w"), indent=2)
    print(f"saved {os.path.join(a.out, f'temporal{suffix}.pt')} and "
          f"temporal{suffix}_jax.npz", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())

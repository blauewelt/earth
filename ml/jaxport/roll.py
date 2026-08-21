"""`ml/rollout_spatial.py`'s `roll_step` / `decode_all` in JAX — tier 2's
rollout slice.

This is the arithmetic that gate G3 scores (`ml/plans/JAX_PORT.md` §5): given
the published embeddings Z, a stage-2 head and the stencil/month features,
iterate `z_{t+1} = head(z_{t-K+1..t}, …)` for twelve steps and decode the
result back to channel space through the frozen codec. Everything AROUND that
loop — the corridor definition, `StdMonths`, `ar1_train`, the skill sums, the
ocean mask — is numpy in the original and is IMPORTED by the driver rather
than re-implemented here. That is the same discipline the embedding slice used
and the reason its result meant anything: when the read-out is literally the
same code on both sides, a backend swap measures the backend.

**Why the roll and not one step.** A single `roll_step` comparison is a
forward-parity check, which tier 1 already passed. What G3 exercises is
twelve of them fed back into their own input window, so any per-step
disagreement compounds — a 1e-6 gap at step 1 is a 1e-5 gap at step 12 and a
different `msss_clim` at every horizon. The test iterates for that reason.

Parity traps that are load-bearing here, all of them inherited rather than
invented:

  * **Stencil slot order is CENTRE FIRST** (`ml/temporal.py:STENCILS`), and
    the gather is SLOT-MAJOR over d_z within a step: [n,S,K,dz] is permuted to
    [n,K,S,dz] and flattened, which is exactly `gather_stencil`'s
    `zj.flatten(1)` per step. Every published head's `inp` weight assumes that
    layout; a transposed one still runs and still produces numbers.
  * **Missing neighbours are ZERO-FILLED, not dropped** — `nbr < 0` marks
    land or off-window, and the model's static context carries the
    observed-flags separately.
  * **The chunk is a BYTE BUDGET, not a row count.** `roll_step`'s comment
    explains the OOM family this cures; the arithmetic does not depend on it,
    but the derived row count must match the torch side anyway, because a
    different chunking is a different summation order and this port is
    compared elementwise.
  * **float16 is STORAGE only.** Z arrives as float16 and is widened to
    float32 before it ever reaches the head, on both sides. The rounding that
    every published number carries happened when Z was written, not here.

**What `amp` does here: it REFUSES.** The torch original runs the two forwards
under an fp16 autocast when `--amp` is set. Reproducing torch's autocast
casting rules op-by-op in JAX is a parity exercise of its own and is not this
slice's job, so passing `amp=True` raises instead of silently rolling at fp32
and reporting a number as if the flag had been honoured. A flag that appears
to apply and quietly does nothing is the failure shape `ml/CLAUDE.md` §0.2 is
about.
"""
import jax.numpy as jnp
import numpy as np
from flax import nnx


class AmpUnsupported(NotImplementedError):
    """`--amp` reached the JAX backend. See the module docstring."""


def _refuse_amp(amp, where):
    if amp:
        raise AmpUnsupported(
            f"{where}: the JAX backend does not implement torch's fp16 "
            f"autocast. rollout_spatial.py's `--amp` is a SPEED knob whose "
            f"honesty the #217 gate certifies on the torch path; honouring "
            f"the flag here would mean transcribing torch's per-op autocast "
            f"casting rules, which is its own parity exercise and not this "
            f"slice's (ml/plans/JAX_PORT.md §5, G3). Refusing rather than "
            f"rolling at fp32 under a flag that says fp16.")


# --------------------------------------------------------------------------
# the two jitted forwards
# --------------------------------------------------------------------------
# Built ONCE at module scope and reused for every step of every roll. This is
# the opposite choice from `embed.py:_encode_fn`, and deliberately: an
# embedding pass calls its jitted function once per batch inside one call, so
# a per-call factory costs one trace; a roll calls it ~234 times across
# separate `roll_step_jax` invocations, and a per-call factory would retrace
# on every step — turning a 12 s forward into a 12 s forward plus a compile.
# The cost of the module-level cache is that it keeps a traced signature alive
# per (model structure, input shape); `clear_jit_cache()` is the release valve
# for a driver that rolls many heads in one process.
@nnx.jit
def _head_forward(model, zin, mfeat_b, sctx):
    """`pred[:, -1]` — the head predicts z_{t+1} at every step of the causal
    window and the roll consumes only the LAST one."""
    pred, _ = model(zin, mfeat_b, sctx)
    return pred[:, -1]


@nnx.jit
def _codec_query(codec, z, qc, off0):
    return codec.query(z, qc, off0)


def clear_jit_cache():
    """Drop the traced signatures held by the two forwards above."""
    for fn in (_head_forward, _codec_query):
        clear = getattr(fn, "clear_cache", None)
        if clear is not None:
            clear()


# --------------------------------------------------------------------------
# roll_step
# --------------------------------------------------------------------------
def roll_chunk_rows(chunk, NBR_t, K, dz):
    """`roll_step`'s byte-budget row count, verbatim.

    Kept as a named function because the DRIVER wants to print it and the
    test wants to pin it: the torch original derives the same number inline,
    and the two must agree or the two backends chunk differently — which is a
    different float summation order in `accumulate`, not just a different
    memory profile.
    """
    if NBR_t is None:
        return chunk
    S = NBR_t.shape[1]
    row_bytes = S * K * dz * 4                     # float32 gather, per pixel
    return max(256, min(chunk, (1 << 30) // max(1, row_bytes)))


def roll_step_jax(model, Zwin, NBR_t, static_ctx, mfeat, chunk, amp=False):
    """One autoregressive step over ALL pixels — `rollout_spatial.roll_step`.

    Zwin [P,K,dz] float32 · NBR_t None (stencil 1) or [P,S] int (-1 =
    missing) · static_ctx [P,·] · mfeat [K,2] → ẑ [P,dz] float32 (jnp).

    Chunking is the torch original's, including the byte budget, because the
    chunk boundary decides the concatenation order of the output and this
    function is compared to torch elementwise.
    """
    _refuse_amp(amp, "roll_step_jax")
    Zwin = jnp.asarray(Zwin, jnp.float32)
    static_ctx = jnp.asarray(static_ctx, jnp.float32)
    mfeat = jnp.asarray(mfeat, jnp.float32)
    P, K, dz = Zwin.shape
    if NBR_t is not None:
        NBR_t = jnp.asarray(np.asarray(NBR_t), jnp.int32)
    chunk = roll_chunk_rows(chunk, NBR_t, K, dz)
    outs = []
    for i in range(0, P, chunk):
        sl = slice(i, min(i + chunk, P))
        if NBR_t is None:
            zin = Zwin[sl]
        else:
            nbr = NBR_t[sl]                                    # [n,S]
            miss = nbr < 0
            zj = Zwin[jnp.maximum(nbr, 0)]                     # [n,S,K,dz]
            # torch writes `zj[miss] = 0.0`, broadcasting the [n,S] mask over
            # the trailing [K,dz]; jnp is functional, so the same thing said
            # as a where.
            zj = jnp.where(miss[:, :, None, None], jnp.zeros((), zj.dtype), zj)
            # [n,S,K,dz] -> [n,K,S,dz] -> [n,K,S*dz]: SLOT-MAJOR within a
            # step, centre slot first. See the module docstring.
            zin = jnp.transpose(zj, (0, 2, 1, 3)).reshape(zj.shape[0], K, -1)
        n = zin.shape[0]
        mfeat_b = jnp.broadcast_to(mfeat[None], (n, K, mfeat.shape[-1]))
        outs.append(_head_forward(model, zin, mfeat_b, static_ctx[sl]))
    return jnp.concatenate(outs, 0)


# --------------------------------------------------------------------------
# decode_all
# --------------------------------------------------------------------------
def decode_all_jax(codec, zhat, C, chunk, amp=False):
    """`codec.query` at every (pixel, channel), offset 0 — [P,C] NUMPY.

    Numpy out, like the torch original: everything downstream of this point
    (`accumulate`, `skill_block`, the audit sums) is the numpy read-out both
    backends share, so the device boundary is here and in exactly one place.
    """
    _refuse_amp(amp, "decode_all_jax")
    zhat = jnp.asarray(zhat, jnp.float32)
    outs = []
    for i in range(0, zhat.shape[0], chunk):
        z = zhat[i:i + chunk]
        n = z.shape[0]
        # int32 rather than torch's int64: JAX defaults to 32-bit and these
        # are embedding-table indices in [0, n_chan) and [0, 2*max_off],
        # nowhere near the range where the width could matter.
        qc = jnp.broadcast_to(jnp.arange(C, dtype=jnp.int32)[None], (n, C))
        off0 = jnp.zeros((n, C, 3), jnp.int32)
        outs.append(np.asarray(_codec_query(codec, z, qc, off0),
                               dtype=np.float32))
    return np.concatenate(outs, 0)


# --------------------------------------------------------------------------
# the iterated roll, shared by the driver and the test
# --------------------------------------------------------------------------
def roll_forward_jax(model, Zwin, NBR_t, static_ctx, moys, month_feats_np,
                     horizon, chunk, next_moy, amp=False):
    """Iterate `roll_step_jax` `horizon` times, yielding ẑ per step.

    `moys` is the CURRENT window's month-of-year list (length K) and
    `next_moy(h)` returns the month of the row just predicted at step h — the
    caller owns the axis, exactly as `rollout_spatial.main()` does, because
    `(cur[-1] + 1) % 12` is only correct while a step is a month.

    `month_feats_np` is passed IN rather than recomputed: it is
    `rollout_spatial.month_feats`, so both backends read their month features
    out of the same function.
    """
    cur = list(moys)
    for h in range(1, horizon + 1):
        mf = month_feats_np(cur)
        zhat = roll_step_jax(model, Zwin, NBR_t, static_ctx, mf, chunk,
                             amp=amp)
        # The window slides AFTER the forward, and the season token appended
        # is the row just predicted — rollout_spatial.main()'s order.
        Zwin = jnp.concatenate([Zwin[:, 1:], zhat[:, None]], 1)
        cur = cur[1:] + [next_moy(h)]
        yield h, zhat, Zwin

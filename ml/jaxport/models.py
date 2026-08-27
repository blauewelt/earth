"""Flax NNX re-implementations of the torch models, forward-parity-exact.

Every module here mirrors a torch module in `ml/` line for line, in EVAL
semantics (dropout off, no autocast). Where the two frameworks disagree by
default, the torch behaviour wins and the reason is written down — a port
whose defaults quietly differ produces plausible garbage, which on this
project is the most dangerous failure shape there is
(`ml/plans/JAX_PORT.md` §3.1).

The parity traps that are actually load-bearing, all of them found by
running the tests rather than by reading:

  * **torch `nn.GELU()` is the EXACT erf gelu**, not the tanh approximation.
    `jax.nn.gelu` defaults to `approximate=True`, so every gelu here passes
    `approximate=False`. The two differ by ~1e-3 at moderate activations —
    far above any parity tolerance, and invisible in a loss curve.
  * **`nn.TransformerEncoderLayer`'s default activation is RELU**, not gelu.
    Both PixelMAE and TemporalTransformer construct the layer without an
    `activation=` argument, so their feed-forward blocks are relu. Only the
    decoder MLP and SectionHead's read-out use gelu.
  * **`nn.TransformerEncoder(layer, n)` has `norm=None`**: there is NO final
    LayerNorm after the stack, even though the layers are norm_first. A
    trailing norm is the single easiest thing to add by accident.
  * **torch stores `nn.Linear.weight` as [out, in]** and computes
    `x @ W.T`; Flax stores `kernel` as [in, out]. The converter transposes.
  * **`nn.MultiheadAttention` packs Q, K, V into one `in_proj_weight`
    [3d, d]**, in that order. A wrong slice still trains and still produces
    numbers.
  * **LayerNorm eps is 1e-5 in torch, 1e-6 in Flax.** Set explicitly.
"""
import os
import sys

import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx

# `ml/` on the path, the way ml/jaxport/train_stage1.py does it: the FSQ
# LADDER's arithmetic (`ml/fsq_ladder.py`) is numpy-only and is IMPORTED here
# rather than mirrored. Everything else in this file is a mirror with a parity
# gate behind it; the ladder is not, because a quantizer whose level positions
# differed between backends by one boundary would make a TPU-trained codec a
# different model from the one the torch eval ladder scores — and the level
# geometry is pure arithmetic with no framework in it. What each backend still
# owns is the APPLICATION, so the straight-through gradient is native.
_ML = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ML not in sys.path:
    sys.path.insert(0, _ML)

import fsq_ladder as fql                                       # noqa: E402

# flax 0.11 made a plain container attribute STATIC (its parameters vanish
# from the pytree) and introduced _nnx_data() to mark containers as data;
# flax <= 0.10 auto-registers plain lists — the very behaviour nnx.data makes
# explicit — and has no `data` attribute at all. flax >= 0.11 also requires
# Python >= 3.11, which the Cloud TPU VM image (Ubuntu 22.04, Python 3.10)
# does not carry, so this module must run under both regimes. The identity
# fallback restores the old auto-registration path on old flax; the parity
# suite is the check that the two regimes compute the same numbers.
_nnx_data = getattr(nnx, "data", lambda x: x)


def gelu_exact(x):
    """torch `nn.GELU()` — the erf form. See the module docstring."""
    return jax.nn.gelu(x, approximate=False)


def _zero_linear(d_in, d_out, rngs):
    """A Linear with weight AND bias at EXACT zeros — `nn.init.zeros_` on
    both, which is E-057's adaLN-zero (`ml/temporal.py:90-91`). The zeros are
    what make a fresh FiLM layer compute the stock forward exactly, so this
    is arithmetic rather than a convention."""
    lin = nnx.Linear(d_in, d_out, rngs=rngs)
    lin.kernel.value = jnp.zeros_like(lin.kernel.value)
    lin.bias.value = jnp.zeros_like(lin.bias.value)
    return lin


# --------------------------------------------------------------------------
# Attention and the torch-semantics transformer encoder
# --------------------------------------------------------------------------
class MultiHeadAttention(nnx.Module):
    """`nn.MultiheadAttention` semantics, with q/k/v kept as separate
    projections.

    torch packs them into one `in_proj_weight`; splitting them here is what
    lets the converter own the mapping explicitly (three named slices) rather
    than hiding it inside a packed kernel whose order nothing checks. The
    arithmetic is identical: q = x_q @ Wq.T + bq, etc.
    """

    def __init__(self, d_model, n_heads, *, rngs):
        if d_model % n_heads:
            raise ValueError(f"d_model {d_model} not divisible by n_heads {n_heads}")
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.q_proj = nnx.Linear(d_model, d_model, rngs=rngs)
        self.k_proj = nnx.Linear(d_model, d_model, rngs=rngs)
        self.v_proj = nnx.Linear(d_model, d_model, rngs=rngs)
        self.out_proj = nnx.Linear(d_model, d_model, rngs=rngs)

    def _split(self, x):
        B, L, _ = x.shape
        return x.reshape(B, L, self.n_heads, self.head_dim).transpose(0, 2, 1, 3)

    def __call__(self, q_in, k_in, v_in, mask=None):
        """mask: additive float [Lq,Lk] (torch's own causal mask), or a bool
        array where **True means MASKED OUT** — which is torch's convention
        for `attn_mask`, and the opposite of Flax's `nnx.MultiHeadAttention`.
        """
        q = self._split(self.q_proj(q_in))
        k = self._split(self.k_proj(k_in))
        v = self._split(self.v_proj(v_in))
        # Scale by 1/sqrt(head_dim) — torch scales the QUERY before the
        # matmul, which is the same value.
        scores = jnp.einsum("bhqd,bhkd->bhqk", q, k) / jnp.sqrt(
            jnp.asarray(self.head_dim, q.dtype))
        if mask is not None:
            if mask.dtype == jnp.bool_:
                scores = jnp.where(mask, jnp.asarray(-jnp.inf, scores.dtype),
                                   scores)
            else:
                scores = scores + mask.astype(scores.dtype)
        # Softmax over the KEY dimension, after masking, exactly as torch does.
        attn = jax.nn.softmax(scores, axis=-1)
        out = jnp.einsum("bhqk,bhkd->bhqd", attn, v)
        B, H, L, D = out.shape
        out = out.transpose(0, 2, 1, 3).reshape(B, L, H * D)
        return self.out_proj(out)


class EncoderLayer(nnx.Module):
    """`nn.TransformerEncoderLayer(d_model, n_heads, dim_feedforward,
    batch_first=True, norm_first=True, dropout=0.0)`.

    Pre-LN, exactly as torch writes it:
        x = x + self_attn(norm1(x))
        x = x + linear2(relu(linear1(norm2(x))))
    Activation is RELU because that is torch's default and neither caller
    overrides it.

    **`film=True` is E-057's `ml/temporal.py:_CondLayer`**, and nothing else
    about the layer changes. The two LayerNorms become FiLM-modulated by a
    per-sample conditioning vector c (adaLN-zero, FGN arXiv:2506.10772 §3):

        s1, b1, s2, b2 = film(c).chunk(4, -1)        # each [B, d_model]
        x = x + sa(norm1(x) * (1 + s1[:,None]) + b1[:,None])
        x = x + ff(norm2(x) * (1 + s2[:,None]) + b2[:,None])

    THE THREE PROPERTIES THAT ARE LOAD-BEARING, all of them mirrored from the
    torch side rather than chosen here (`ml/plans/FGN_JAX_PORT.md` §0 keeps
    torch and JAX twins of EACH OTHER, not two children of weathernext):

    * **`film=False` builds NO film parameter and takes the ORIGINAL code
      path**, statement for statement — the flag-off model is bitwise the
      pre-E-057 one and every existing parity certificate stands untouched.
    * **`film` is ONE Linear(d, 4*d) per block**, whose output is split
      s1, b1, s2, b2 IN THAT ORDER — torch's `chunk(4, -1)` layout, so
      `convert.py` maps `encoder.layers.N.film.{weight,bias}` 1:1.
    * **`film` is ZERO-INITIALISED, kernel and bias.** At init s1=b1=s2=b2=0
      and the arithmetic reduces to `norm1(x) * 1.0 + 0.0`, which is the stock
      forward EXACTLY (multiplying a float by 1.0 and adding 0.0 are both
      exact operations). So an ε-conditioned head at step 0 IS the
      deterministic incumbent whatever ε it is handed.

    THE STOCK LayerNorm KEEPS ITS AFFINE and the modulation sits ON TOP of it
    — registered deviation (ii) of `ml/plans/FGN_JAX_PORT.md` §0, kept
    identical to torch so the two backends stay twins.
    """

    def __init__(self, d_model, n_heads, dim_feedforward, *, rngs, film=False):
        self.self_attn = MultiHeadAttention(d_model, n_heads, rngs=rngs)
        self.linear1 = nnx.Linear(d_model, dim_feedforward, rngs=rngs)
        self.linear2 = nnx.Linear(dim_feedforward, d_model, rngs=rngs)
        # eps 1e-5 is torch's LayerNorm default; Flax's is 1e-6.
        self.norm1 = nnx.LayerNorm(d_model, epsilon=1e-5, rngs=rngs)
        self.norm2 = nnx.LayerNorm(d_model, epsilon=1e-5, rngs=rngs)
        # Created LAST and only when asked for, so the rng draws of every
        # pre-existing parameter are in their original order and a flag-off
        # layer's pytree is key-for-key what it has always been.
        # ONE assignment, not `= None` then `= lin`: flax >= 0.11 decides an
        # attribute's data/static status on FIRST assignment, so seeding it
        # with None would make the module refuse the Linear that follows.
        self.film = _nnx_data(_zero_linear(d_model, 4 * d_model, rngs)) \
            if film else None

    def __call__(self, x, mask=None, c=None):
        if self.film is None:
            # THE ORIGINAL PATH, byte for byte. A film-less layer handed a
            # conditioning vector has nowhere to put it; refuse rather than
            # ignore it (the JAX twin of ml/temporal.py's eps guard).
            if c is not None:
                raise ValueError(
                    "EncoderLayer was built with film=False and was handed a "
                    "conditioning vector. The noise has nowhere to enter; "
                    "refusing rather than silently ignoring it.")
            h = self.norm1(x)
            x = x + self.self_attn(h, h, h, mask=mask)
            h = self.norm2(x)
            x = x + self.linear2(jax.nn.relu(self.linear1(h)))
            return x
        if c is None:
            raise ValueError(
                "this is a FiLM-conditioned EncoderLayer (film=True) and it "
                "cannot be run without its conditioning vector.")
        # `chunk(4, -1)` — s1, b1, s2, b2 IN THAT ORDER (ml/temporal.py:95).
        d = self.norm1.scale.value.shape[-1]
        f = self.film(c)
        s1, b1, s2, b2 = (f[..., 0 * d:1 * d], f[..., 1 * d:2 * d],
                          f[..., 2 * d:3 * d], f[..., 3 * d:4 * d])
        h = self.norm1(x) * (1.0 + s1[:, None, :]) + b1[:, None, :]
        x = x + self.self_attn(h, h, h, mask=mask)
        h = self.norm2(x) * (1.0 + s2[:, None, :]) + b2[:, None, :]
        x = x + self.linear2(jax.nn.relu(self.linear1(h)))
        return x


class TransformerEncoder(nnx.Module):
    """`nn.TransformerEncoder(layer, n_layers)` — NO final norm (torch's
    `norm` argument defaults to None and neither caller passes one).

    `film=True` makes every layer a `_CondLayer` and the container a
    `_CondEncoder` (`ml/temporal.py:103`): the layer KEYS are unchanged, which
    is what lets a legacy deterministic checkpoint warm-start the trunk of an
    FGN head.
    """

    def __init__(self, d_model, n_heads, n_layers, dim_feedforward=None, *,
                 rngs, film=False):
        if dim_feedforward is None:
            dim_feedforward = 4 * d_model
        # nnx.data: a plain list of submodules is treated as a STATIC
        # attribute otherwise, and the parameters vanish from the pytree.
        self.layers = _nnx_data([EncoderLayer(d_model, n_heads, dim_feedforward,
                                             rngs=rngs, film=film)
                                for _ in range(n_layers)])

    def __call__(self, x, mask=None, c=None):
        for lyr in self.layers:
            x = lyr(x, mask=mask) if c is None else lyr(x, mask=mask, c=c)
        return x


def fsq_quantize(v, half, off, shift, scale, is_exp, exp_a, exp_logc,
                 exp_n1, exp_jmin, any_exp):
    """`ml/model.py:InputQuant.__call__`, in JAX. [.., d_z] -> [.., d_z].

    Same two ladders, same straight-through estimator, same order of
    operations — `jax.lax.stop_gradient` where torch writes `.detach()`. The
    constants come in as arrays built from `ml/fsq_ladder.py`, so the LEVEL
    POSITIONS are the shared definition and only the elementwise application
    is written twice (tests/test_e048_fsq_ladders.py gates the two at 1e-5 for
    both ladders).
    """
    g = half * jnp.tanh(v / scale + shift) - off
    q = g + jax.lax.stop_gradient(jnp.round(g) - g)
    out = (q + off) * scale / half
    if any_exp:
        m = jnp.maximum(jnp.abs(v), 1e-30)
        ge = jnp.log(m / exp_a) / exp_logc
        gq = jnp.minimum(jnp.maximum(jnp.round(ge), exp_jmin), exp_n1)
        qe = ge + jax.lax.stop_gradient(gq - ge)
        s = jnp.where(v < 0, -jnp.ones_like(v), jnp.ones_like(v))
        oe = s * exp_a * jnp.exp(qe * exp_logc)
        # The zero level of an odd L: value 0, gradient 1 (`v - sg(v)`).
        oe = jnp.where(gq < -0.5, v - jax.lax.stop_gradient(v), oe)
        out = jnp.where(is_exp, oe, out)
    return out


def causal_mask(k, dtype=jnp.float32):
    """`nn.Transformer.generate_square_subsequent_mask(K)` — an ADDITIVE
    float mask: 0 on and below the diagonal, -inf strictly above."""
    i = jnp.arange(k)[:, None]
    j = jnp.arange(k)[None, :]
    return jnp.where(j > i, jnp.asarray(-jnp.inf, dtype),
                     jnp.asarray(0.0, dtype))


# --------------------------------------------------------------------------
# PixelMAE (ml/model.py)
# --------------------------------------------------------------------------
class PixelMAE(nnx.Module):
    """Mirror of `ml/model.py:PixelMAE`. Same constructor signature, so
    `codec_from_ckpt_jax` can build it from a checkpoint's `args` the way
    `codec_from_ckpt` does."""

    def __init__(self, n_chan, d_model=128, n_heads=4, n_layers=4,
                 d_z=32, d_dec=256, max_abs_offset=3, patch=1, dec_layers=2,
                 k_time=1, fsq_levels="", fsq_ladder="uniform",
                 fsq_exp_base=fql.DEFAULT_EXP_BASE, fsq_ladder_fit="",
                 *, rngs=None):
        rngs = rngs if rngs is not None else nnx.Rngs(0)
        self.n_chan = n_chan
        self.d_z = d_z
        self.max_off = max_abs_offset
        # E-046/E-048 FSQ BOTTLENECK, mirroring ml/model.py: "" is None, and
        # None is the continuous bottleneck — `_bottleneck` then returns its
        # argument, so a model that does not name the flag is untouched and
        # tier-1 forward parity is unaffected.
        #
        # THE CONSTANTS ARE PLAIN TUPLES, not arrays. flax 0.11 treats a
        # non-Variable attribute as STATIC, so it lands in the graphdef, which
        # `nnx.split`/`nnx.merge` require to be HASHABLE — a numpy or jnp array
        # there is either unhashable or a silently-retracing constant. Tuples
        # are hashable, carry no parameters (FSQ has none, in either backend),
        # and are rebuilt into arrays inside the forward, where XLA folds them.
        self.fsq_levels = str(fsq_levels or "")
        self.fsq_ladder = str(fsq_ladder or "uniform")
        self.fsq_exp_base = float(fsq_exp_base)
        self.fsq_ladder_fit = str(fsq_ladder_fit or "")
        self.fsq = None
        self.fsq_needs_fit = False
        if self.fsq_levels:
            lv = fql.parse_levels(self.fsq_levels, d_z, "--fsq-levels")
            # scale=2.0 is `ml/model.py:fsq_from_levels`' sigma = 1, and it is
            # what a LEGACY fit string (no ':<R>' fields) resolves to — so a
            # checkpoint written before the scale was fitted rebuilds here on
            # exactly the lattice it trained on.
            is_exp, base, sc, _fitted = fql.resolve(
                self.fsq_ladder, lv, d_z, self.fsq_exp_base,
                self.fsq_ladder_fit, scale=2.0)
            self.fsq_needs_fit = not _fitted
            self.set_fsq_ladder(is_exp, base, self.fsq_ladder_fit, sc)
        elif self.fsq_ladder != "uniform" or self.fsq_ladder_fit:
            raise SystemExit(
                f"--fsq-ladder {self.fsq_ladder!r} without --fsq-levels: "
                f"there is no bottleneck to put a ladder on.")
        # E-047 TIME BLOCKS, mirroring ml/model.py exactly: k_time == 1 adds NO
        # parameter and NO branch that runs, so a k_time=1 model's pytree is
        # key-for-key what it has always been and tier 1's forward parity is
        # untouched. k_time > 1 makes the encoder's input a k_time x C GRID of
        # cells whose token is chan_emb[c] + time_emb[j].
        self.k_time = int(k_time)
        self.patch = patch
        p2 = patch * patch

        self.val_proj = nnx.Linear(1 if patch == 1 else 2 * p2, d_model,
                                   rngs=rngs)
        self.chan_emb = nnx.Embed(n_chan, d_model, rngs=rngs)
        # Created ONLY when k_time > 1 — the torch module does the same, and
        # the converter's refusal contract is what turns "created only when"
        # into a checked property rather than a hoped-for one.
        self.time_emb = (nnx.Embed(self.k_time, d_model, rngs=rngs)
                         if self.k_time > 1 else None)
        # Bare parameters, not modules — torch keys are `mask_tok` etc.
        self.mask_tok = nnx.Param(jnp.zeros((d_model,)))
        self.miss_tok = nnx.Param(jnp.zeros((d_model,)))
        self.cls_tok = nnx.Param(jnp.zeros((d_model,)))
        self.ctx_proj = nnx.Linear(4, d_model, rngs=rngs)

        self.encoder = TransformerEncoder(d_model, n_heads, n_layers,
                                          4 * d_model, rngs=rngs)
        self.to_z = nnx.Linear(d_model, d_z, rngs=rngs)

        self.q_chan = nnx.Embed(n_chan, 64, rngs=rngs)
        self.q_off = nnx.Embed(2 * max_abs_offset + 1, 16, rngs=rngs)
        # E-047 option (b): WITHIN-BLOCK POSITION GETS ITS OWN QUERY EMBEDDING
        # rather than reusing `off`'s dt slot, so one index never means two
        # things. `off`'s dt keeps meaning BLOCKS.
        self.q_time = (nnx.Embed(self.k_time, 16, rngs=rngs)
                       if self.k_time > 1 else None)
        # torch builds this as an nn.Sequential of Linear/GELU pairs, so the
        # LINEARS sit at even indices 0, 2, 4 ... and the converter keys off
        # that. dec_layers counts HIDDEN layers.
        self.dec_layers = dec_layers
        dims = [d_z + 64 + 3 * 16 + (16 if self.k_time > 1 else 0)] \
            + [d_dec] * dec_layers
        self.decoder = _nnx_data([nnx.Linear(dims[i], dims[i + 1], rngs=rngs)
                                 for i in range(dec_layers)]
                                + [nnx.Linear(d_dec, 1, rngs=rngs)])

    # -- token assembly, shared by both patch paths -------------------------
    def _tokens(self, vt, ce, obs, mask, ctx, B, C):
        """The where-cascade of ml/model.py: an observed-and-unmasked channel
        keeps its value token; a masked one is replaced by mask_tok + chan;
        an unobserved one by miss_tok + chan. Written as three `where`s over
        a zeroed base, exactly as torch does, so the arithmetic (and the
        additions, which are not idempotent in float) matches term for term.
        """
        z = jnp.zeros_like(vt)
        vt = jnp.where((obs & ~mask)[..., None], vt, z)
        vt = vt + jnp.where((obs & mask)[..., None],
                            jnp.broadcast_to(self.mask_tok.value, (B, C, vt.shape[-1])) + ce, z)
        vt = vt + jnp.where((~obs)[..., None],
                            jnp.broadcast_to(self.miss_tok.value, (B, C, vt.shape[-1])) + ce, z)
        cls = jnp.broadcast_to(self.cls_tok.value, (B, 1, vt.shape[-1]))
        return jnp.concatenate([cls, self.ctx_proj(ctx)[:, None, :], vt], axis=1)

    def set_fsq_ladder(self, is_exp, base, fit="", scales=None):
        """Install a per-dimension ladder — the constructor's path, and the
        one the E-048 `auto` fit takes mid-run.

        `scales` is the per-dimension saturation radius R_d, or None to keep
        the current one. It goes into the static tuple like every other
        constant, so the fitted radius is part of the graphdef and the
        re-split below is what makes it take effect.

        THE CALLER MUST RE-SPLIT AFTER THIS. `self.fsq` is a STATIC attribute,
        so it lives in the graphdef and a `jax.jit`-ed closure over the OLD
        graphdef would go on quantizing with the old lattice without retracing
        — jit caches on argument shapes, not on closed-over Python values.
        `ml/jaxport/train_stage1.py` re-splits and rebuilds its step function
        in the same breath as calling this, and says so there.
        """
        lv = fql.parse_levels(self.fsq_levels, self.d_z, "--fsq-levels")
        half, off, shift = fql.uniform_params(lv)
        n, a_rel, logc, has_zero = fql.exp_params(lv, base,
                                                  flag="--fsq-levels exp")
        # sigma = 1 on the codec side by DEFAULT (there is no Z to measure) —
        # ml/model.py:fsq_from_levels' choice, so the radius is 2 — but under
        # `--fsq-ladder auto` the radius is MEASURED per dimension and arrives
        # here as `scales`. Keep the current one when nothing is passed.
        if scales is None:
            scale = (np.full(self.d_z, 2.0) if self.fsq is None
                     else np.asarray(self.fsq[3], np.float64))
        else:
            scale = np.asarray(scales, np.float64) * np.ones(self.d_z)
        if not np.isfinite(scale).all() or (scale <= 0).any():
            raise SystemExit(
                f"--fsq-levels: saturation radius {scale.tolist()[:4]}… is "
                f"not finite and positive on every dimension — a lattice of "
                f"extent 0 quantizes nothing.")
        self.fsq_ladder_fit = str(fit or "")
        self.fsq = (tuple(float(x) for x in half),
                    tuple(float(x) for x in off),
                    tuple(float(x) for x in shift),
                    tuple(float(x) for x in scale),
                    tuple(bool(x) for x in is_exp),
                    tuple(float(x) for x in a_rel * scale),
                    tuple(float(x) for x in logc),
                    tuple(float(x) - 1.0 for x in n),
                    tuple(-1.0 if bool(z) else 0.0 for z in has_zero),
                    bool(np.asarray(is_exp).any()))
        return self

    def _bottleneck(self, z):
        """`to_z`'s output, quantized when the FSQ flags are on — the mirror
        of ml/model.py's one-line bottleneck. With the flag off this returns
        its ARGUMENT: no cast, no branch, nothing added to the graph."""
        if self.fsq is None:
            return z
        (half, off, shift, scale, is_exp, exp_a, exp_logc, exp_n1, exp_jmin,
         any_exp) = self.fsq
        dt = z.dtype
        A = (lambda t: jnp.asarray(t, dt))
        shp = z.shape
        v = z.reshape(-1, self.d_z)
        out = fsq_quantize(v, A(half), A(off), A(shift),
                           A(scale), jnp.asarray(is_exp),
                           A(exp_a), A(exp_logc), A(exp_n1), A(exp_jmin),
                           any_exp)
        return out.reshape(shp)

    def encode(self, x, obs, mask, ctx):
        """patch=1: x, obs [B,C] · patch>1: x, obs [B,C,patch²] ·
        k_time>1: x, obs, mask [B,k_time,C]
        mask [B,C] bool · ctx [B,4] → z [B,d_z].

        ONE LINE over `encode_pre`, exactly as ml/model.py splits it: the E-048
        `auto` fit measures the PRE-quantization distribution through the real
        encoder, and under jit a capture hook into the forward pass would not
        work at all."""
        return self._bottleneck(self.encode_pre(x, obs, mask, ctx))

    def encode_pre(self, x, obs, mask, ctx):
        """`encode` WITHOUT the bottleneck: `to_z`'s raw output."""
        # WIDEN THE INPUT TO THE WEIGHTS' DTYPE, at the top of encode, for the
        # same reason ml/model.py does it here rather than at the six call
        # sites: family 4 is float16 STORAGE and every reader hands the codec
        # a batch whose dtype follows the tensor. float16 -> float32 is exact,
        # so no existing run's arithmetic moves.
        wdt = self.val_proj.kernel.value.dtype
        if x.dtype != wdt:
            x = x.astype(wdt)
        if ctx.dtype != wdt:
            ctx = ctx.astype(wdt)
        cw = self.chan_emb.embedding.value
        if self.patch > 1:
            B, C, P2 = x.shape
            ce = jnp.broadcast_to(cw[None, :, :], (B, self.n_chan, cw.shape[1]))
            feat = jnp.concatenate([x * obs, obs.astype(x.dtype)], axis=-1)
            vt = self.val_proj(feat) + ce
            obs = obs[..., P2 // 2]        # the CENTER cell defines the channel
        elif self.k_time > 1:
            # x, obs, mask [B, k_time, C] -> [B, k_time*C] tokens, cell (j, c)
            # carrying chan_emb[c] + time_emb[j]. Flattened j-major, exactly as
            # ml/model.py does it, so the two token orders cannot diverge.
            B, KT, C = x.shape
            if KT != self.k_time:
                raise ValueError(f"encode: input has {KT} cells on the time "
                                 f"axis but this codec is k_time={self.k_time}")
            x = x.reshape(B, KT * C)
            obs = obs.reshape(B, KT * C)
            mask = mask.reshape(B, KT * C)
            ce = jnp.broadcast_to(
                (cw[None, None, :, :]
                 + self.time_emb.embedding.value[None, :, None, :]
                 ).reshape(1, KT * C, cw.shape[1]),
                (B, KT * C, cw.shape[1]))
            vt = self.val_proj(x[..., None]) + ce
            C = KT * C
        else:
            B, C = x.shape
            ce = jnp.broadcast_to(cw[None, :, :], (B, self.n_chan, cw.shape[1]))
            vt = self.val_proj(x[..., None]) + ce
        toks = self._tokens(vt, ce, obs, mask, ctx, B, C)
        h = self.encoder(toks)
        return self.to_z(h[:, 0])

    def query(self, z, chan_idx, off, tpos=None):
        """z [B,d_z] · chan_idx [B,Q] · off [B,Q,3] ints in [-max,max] → [B,Q].

        `tpos` [B,Q] is the WITHIN-BLOCK cell position and is required exactly
        when k_time > 1 — a block codec asked for "channel c" without saying
        WHICH CELL would be answering a question with no answer, so it raises
        rather than picking one (ml/model.py:query says the same in the same
        words)."""
        B, Q = chan_idx.shape
        qc = self.q_chan(chan_idx)
        qo = self.q_off(off + self.max_off).reshape(B, Q, -1)
        zq = jnp.broadcast_to(z[:, None, :], (B, Q, z.shape[-1]))
        parts = [zq, qc, qo]
        if self.k_time > 1:
            if tpos is None:
                raise ValueError(
                    f"query(): this codec has k_time={self.k_time}, so every "
                    f"query names a cell (tpos [B,Q] in 0..{self.k_time - 1}) "
                    f"as well as a channel")
            parts.append(self.q_time(tpos))
        h = jnp.concatenate(parts, axis=-1)
        for lin in self.decoder[:-1]:
            h = gelu_exact(lin(h))        # nn.GELU() — the erf form
        return self.decoder[-1](h)[..., 0]


# --------------------------------------------------------------------------
# TemporalTransformer (ml/temporal.py)
# --------------------------------------------------------------------------
class TemporalTransformer(nnx.Module):
    """Mirror of `ml/temporal.py:TemporalTransformer`.

    stencil==1 keeps the EXACT legacy input shapes (the layout, centre slot
    first, is what every published head's weights assume); stencil>1 widens
    the per-step input to the neighbourhood's z and appends the S static
    observed-flags to the static context.

    **`eps_dim > 0` is E-057's FGN head** (`ml/plans/FGN_JAX_PORT.md` §1),
    mirroring `ml/temporal.py` site for site:

      · `eps_embed = Sequential(Linear(eps_dim, d_model), SiLU,
        Linear(d_model, d_model))` maps ε ~ N(0,1)^k to the conditioning
        vector c [B, d_model]. It is kept as a two-element list so the torch
        keys `eps_embed.0.*` / `eps_embed.2.*` map 1:1 (the SiLU at index 1 is
        parameterless), the same convention PixelMAE's decoder uses.
      · the encoder's every layer FiLM-modulates `norm1` and `norm2` from c.
        **THOSE ARE THE ONLY TWO SITES.** `ml/temporal.py` conditions nothing
        else: `nn.TransformerEncoder(layer, n_layers)` is constructed with
        `norm=None`, so this model has no trailing LayerNorm to condition, and
        `head` / `inp` / `static` / `pos` are untouched on both sides. (The
        `out_norm` site that exists in `ml/probe_head.py:SectionHead` belongs
        to a different model and is not part of the stage-2 head at all.)
      · `eps_dim == 0` builds NO eps_embed and NO film, and the forward is the
        pre-E-057 one statement for statement.

    The guard runs in BOTH directions, exactly as `ml/temporal.py:272-286`
    does, and for the same reason: a caller that builds a head from a
    checkpoint's args and forgets ε would roll an FGN head CLEAN — a
    deterministic trajectory from a model whose whole content is that it is
    not deterministic.
    """

    def __init__(self, d_z=32, d_model=96, n_heads=4, n_layers=3, k_max=36,
                 direct=(), stencil=1, eps_dim=0, *, rngs=None):
        rngs = rngs if rngs is not None else nnx.Rngs(0)
        self.stencil = stencil
        self.d_z = d_z
        if stencil == 1:
            self.inp = nnx.Linear(d_z + 2, d_model, rngs=rngs)
            self.static = nnx.Linear(d_z + 2, d_model, rngs=rngs)
        else:
            self.inp = nnx.Linear(stencil * d_z + 2, d_model, rngs=rngs)
            self.static = nnx.Linear(d_z + 2 + stencil, d_model, rngs=rngs)
        self.pos = nnx.Embed(k_max, d_model, rngs=rngs)
        self.eps_dim = int(eps_dim)
        self.encoder = TransformerEncoder(d_model, n_heads, n_layers,
                                          4 * d_model, rngs=rngs,
                                          film=bool(self.eps_dim))
        # torch: nn.Sequential(Linear(eps_dim, d_model), nn.SiLU(),
        #                      Linear(d_model, d_model)) -> keys 0 and 2.
        self.eps_embed = _nnx_data(
            [nnx.Linear(self.eps_dim, d_model, rngs=rngs),
             nnx.Linear(d_model, d_model, rngs=rngs)]) if self.eps_dim else None
        self.head = nnx.Linear(d_model, d_z, rngs=rngs)
        self.direct = tuple(int(h) for h in direct)
        # A dict keyed by str(h), the same way nn.ModuleDict exposes it — the
        # torch keys are `heads_direct.<h>.weight`.
        self.heads_direct = _nnx_data(
            {str(h): nnx.Linear(d_model, d_z, rngs=rngs)
             for h in self.direct}) if self.direct else None
        self.d_model = d_model

    def embed_eps(self, eps):
        """ε [B, eps_dim] → c [B, d_model] — `self.eps_embed(eps)` of
        `ml/temporal.py:292`. `jax.nn.silu` is torch's `nn.SiLU` exactly
        (x · sigmoid(x)); there is no approximate variant to fall into."""
        return self.eps_embed[1](jax.nn.silu(self.eps_embed[0](eps)))

    def __call__(self, z_seq, month_seq, static_ctx, eps=None):
        """z_seq [B,K,·] · month_seq [B,K,2] · static_ctx [B,·]
        → pred [B,K,d_z], h [B,K,d_model].

        `eps` [B, eps_dim] is E-057's global noise vector and is REQUIRED
        exactly when the head was built with one. The three-positional-argument
        call every existing caller makes is unchanged.
        """
        B, K = z_seq.shape[0], z_seq.shape[1]
        if self.eps_dim == 0 and eps is not None:
            raise ValueError(
                "TemporalTransformer was built with eps_dim=0 (a "
                "deterministic head) and was handed an eps vector. The noise "
                "has nowhere to enter; refusing rather than silently "
                "ignoring it.")
        if self.eps_dim and eps is None:
            raise ValueError(
                f"this is an FGN head (eps_dim={self.eps_dim}) and it cannot "
                f"be run without its noise vector: every forward is a SAMPLE "
                f"from the predictive distribution, conditioned on "
                f"eps ~ N(0,1)^{self.eps_dim}. A caller that has no eps must "
                f"choose a member deliberately — jaxport.train_stage2."
                f"fgn_eval_eps() is the representative (zeros) one — never "
                f"fall into a default. Plan: ml/plans/FGN_JAX_PORT.md")
        h = (self.inp(jnp.concatenate([z_seq, month_seq], axis=-1))
             + self.static(static_ctx)[:, None, :]
             # The learned positional embedding is added over the FIRST K
             # positions of the table, not over a slice chosen by index.
             + self.pos.embedding.value[None, :K])
        mask = causal_mask(K, h.dtype)
        if self.eps_dim:
            h = self.encoder(h, mask=mask, c=self.embed_eps(eps))
        else:
            h = self.encoder(h, mask=mask)
        return self.head(h), h

    def direct_pred(self, h, horizon):
        """The optional multi-horizon read-out, exposed the same way torch's
        ModuleDict is: keyed by str(h)."""
        return self.heads_direct[str(int(horizon))](h)


# --------------------------------------------------------------------------
# SectionHead (ml/probe_head.py)
# --------------------------------------------------------------------------
class SectionHead(nnx.Module):
    """Mirror of `ml/probe_head.py:SectionHead`.

    Dropout is 0.1 in torch, but every parity check (and every published
    read-out of a fitted head) runs in EVAL mode, so dropout here is
    deterministic-off by default: `deterministic=True` skips it entirely,
    which is exactly what `nn.Dropout` does under `model.eval()`.
    """

    def __init__(self, in_dim, d=64, K=1, n_blocks=0, *, rngs=None):
        rngs = rngs if rngs is not None else nnx.Rngs(0)
        self.d = d
        self.K = K
        self.n_blocks = n_blocks
        self.lift = nnx.Linear(in_dim, d, rngs=rngs)
        # torch: nn.TransformerEncoderLayer(d, max(1, d // 64), 4*d,
        #        batch_first=True, norm_first=True, dropout=0.1)
        self.blocks = (TransformerEncoder(d, max(1, d // 64), n_blocks,
                                          4 * d, rngs=rngs)
                       if n_blocks else None)
        self.q = nnx.Param(jnp.zeros((1, 1, d)))
        # num_heads=1: one learned query pools the whole section.
        self.att = MultiHeadAttention(d, 1, rngs=rngs)
        self.out_norm = nnx.LayerNorm(d, epsilon=1e-5, rngs=rngs)
        self.out_lin1 = nnx.Linear(d, 32, rngs=rngs)
        self.out_lin2 = nnx.Linear(32, 1, rngs=rngs)

    def __call__(self, tok, deterministic=True):
        if not deterministic:
            raise NotImplementedError(
                "tier 1 is an eval-mode reference implementation; training "
                "dropout belongs to tier 3 (ml/plans/JAX_PORT.md §4)")
        h = self.lift(tok)                       # self.drop is identity in eval
        if self.blocks is not None:
            h = self.blocks(h)
        q = jnp.broadcast_to(self.q.value, (tok.shape[0], 1, self.d))
        pooled = self.att(q, h, h)
        # nn.Sequential(LayerNorm, Linear(d,32), nn.GELU(), Linear(32,1))
        y = self.out_lin2(gelu_exact(self.out_lin1(self.out_norm(pooled[:, 0]))))
        return y[..., 0]

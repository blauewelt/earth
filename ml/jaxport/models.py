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
import jax
import jax.numpy as jnp
from flax import nnx


def gelu_exact(x):
    """torch `nn.GELU()` — the erf form. See the module docstring."""
    return jax.nn.gelu(x, approximate=False)


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
    """

    def __init__(self, d_model, n_heads, dim_feedforward, *, rngs):
        self.self_attn = MultiHeadAttention(d_model, n_heads, rngs=rngs)
        self.linear1 = nnx.Linear(d_model, dim_feedforward, rngs=rngs)
        self.linear2 = nnx.Linear(dim_feedforward, d_model, rngs=rngs)
        # eps 1e-5 is torch's LayerNorm default; Flax's is 1e-6.
        self.norm1 = nnx.LayerNorm(d_model, epsilon=1e-5, rngs=rngs)
        self.norm2 = nnx.LayerNorm(d_model, epsilon=1e-5, rngs=rngs)

    def __call__(self, x, mask=None):
        h = self.norm1(x)
        x = x + self.self_attn(h, h, h, mask=mask)
        h = self.norm2(x)
        x = x + self.linear2(jax.nn.relu(self.linear1(h)))
        return x


class TransformerEncoder(nnx.Module):
    """`nn.TransformerEncoder(layer, n_layers)` — NO final norm (torch's
    `norm` argument defaults to None and neither caller passes one)."""

    def __init__(self, d_model, n_heads, n_layers, dim_feedforward=None, *,
                 rngs):
        if dim_feedforward is None:
            dim_feedforward = 4 * d_model
        # nnx.data: a plain list of submodules is treated as a STATIC
        # attribute otherwise, and the parameters vanish from the pytree.
        self.layers = nnx.data([EncoderLayer(d_model, n_heads, dim_feedforward,
                                             rngs=rngs)
                                for _ in range(n_layers)])

    def __call__(self, x, mask=None):
        for lyr in self.layers:
            x = lyr(x, mask=mask)
        return x


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
                 *, rngs=None):
        rngs = rngs if rngs is not None else nnx.Rngs(0)
        self.n_chan = n_chan
        self.d_z = d_z
        self.max_off = max_abs_offset
        self.patch = patch
        p2 = patch * patch

        self.val_proj = nnx.Linear(1 if patch == 1 else 2 * p2, d_model,
                                   rngs=rngs)
        self.chan_emb = nnx.Embed(n_chan, d_model, rngs=rngs)
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
        # torch builds this as an nn.Sequential of Linear/GELU pairs, so the
        # LINEARS sit at even indices 0, 2, 4 ... and the converter keys off
        # that. dec_layers counts HIDDEN layers.
        self.dec_layers = dec_layers
        dims = [d_z + 64 + 3 * 16] + [d_dec] * dec_layers
        self.decoder = nnx.data([nnx.Linear(dims[i], dims[i + 1], rngs=rngs)
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

    def encode(self, x, obs, mask, ctx):
        """patch=1: x, obs [B,C] · patch>1: x, obs [B,C,patch²]
        mask [B,C] bool · ctx [B,4] → z [B,d_z]."""
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
        ce = jnp.broadcast_to(self.chan_emb.embedding.value[None, :, :],
                              (x.shape[0], self.n_chan,
                               self.chan_emb.embedding.value.shape[1]))
        if self.patch > 1:
            B, C, P2 = x.shape
            feat = jnp.concatenate([x * obs, obs.astype(x.dtype)], axis=-1)
            vt = self.val_proj(feat) + ce
            obs = obs[..., P2 // 2]        # the CENTER cell defines the channel
        else:
            B, C = x.shape
            vt = self.val_proj(x[..., None]) + ce
        toks = self._tokens(vt, ce, obs, mask, ctx, B, C)
        h = self.encoder(toks)
        return self.to_z(h[:, 0])

    def query(self, z, chan_idx, off):
        """z [B,d_z] · chan_idx [B,Q] · off [B,Q,3] ints in [-max,max] → [B,Q]."""
        B, Q = chan_idx.shape
        qc = self.q_chan(chan_idx)
        qo = self.q_off(off + self.max_off).reshape(B, Q, -1)
        zq = jnp.broadcast_to(z[:, None, :], (B, Q, z.shape[-1]))
        h = jnp.concatenate([zq, qc, qo], axis=-1)
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
    """

    def __init__(self, d_z=32, d_model=96, n_heads=4, n_layers=3, k_max=36,
                 direct=(), stencil=1, *, rngs=None):
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
        self.encoder = TransformerEncoder(d_model, n_heads, n_layers,
                                          4 * d_model, rngs=rngs)
        self.head = nnx.Linear(d_model, d_z, rngs=rngs)
        self.direct = tuple(int(h) for h in direct)
        # A dict keyed by str(h), the same way nn.ModuleDict exposes it — the
        # torch keys are `heads_direct.<h>.weight`.
        self.heads_direct = nnx.data(
            {str(h): nnx.Linear(d_model, d_z, rngs=rngs)
             for h in self.direct}) if self.direct else None
        self.d_model = d_model

    def __call__(self, z_seq, month_seq, static_ctx):
        """z_seq [B,K,·] · month_seq [B,K,2] · static_ctx [B,·]
        → pred [B,K,d_z], h [B,K,d_model]."""
        B, K = z_seq.shape[0], z_seq.shape[1]
        h = (self.inp(jnp.concatenate([z_seq, month_seq], axis=-1))
             + self.static(static_ctx)[:, None, :]
             # The learned positional embedding is added over the FIRST K
             # positions of the table, not over a slice chosen by index.
             + self.pos.embedding.value[None, :K])
        h = self.encoder(h, mask=causal_mask(K, h.dtype))
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

#!/usr/bin/env python3
"""E-052's field head in Flax NNX — a forward-parity-exact mirror of
`ml/field_model.py`, plus the two-way converter.

Same contract as `ml/jaxport/models.py`, and the same reason for existing: a
SECOND, INDEPENDENT implementation of the same arithmetic is worth nothing
unless the two agree, so every module here is gated against its torch original
in `tests/test_jaxport_field.py` (F1-F8). Where the frameworks disagree by
default the TORCH behaviour wins and the reason is written down.

THE PARITY TRAPS THAT ARE LOAD-BEARING HERE. The first six are `models.py`'s
own list and apply unchanged; the rest are new to this file.

  * **torch `nn.GELU()` is the EXACT erf gelu**, and so is
    `nn.TransformerEncoderLayer(activation="gelu")`. `jax.nn.gelu` defaults to
    the tanh approximation, which differs by ~1e-3 at moderate activations —
    far above any tolerance here and invisible in a loss curve. Every gelu in
    this file goes through `models.gelu_exact`.
  * **`nn.TransformerEncoderLayer`'s default activation is RELU.** `models.py`
    mirrors that because neither of its callers overrides it. `TemporalCond`
    DOES override it (`activation="gelu"`), so this file cannot reuse
    `models.EncoderLayer` and defines a gelu twin instead. Reusing the relu
    layer here would have been the single most plausible-looking mistake
    available, and no loss curve would have shown it.
  * **`nn.TransformerEncoder(layer, n)` has `norm=None`** — no final LayerNorm
    inside the stack. `TemporalCond.out_norm` is a SEPARATE module applied
    after it, and the two must not be conflated.
  * **`nn.TransformerEncoderLayer`'s LayerNorm eps is torch's 1e-5 default**,
    where the DiT's own norms are built explicitly at **1e-6** with
    `elementwise_affine=False`. Both appear in this file, three lines apart.
  * **torch stores `nn.Linear.weight` as [out, in]**; Flax's `kernel` is
    [in, out]. The converter transposes.
  * **`nn.MultiheadAttention` packs Q, K, V into one `in_proj_weight`
    [3d, d]**, in that order.
  * **`elementwise_affine=False` LayerNorm has NO parameters at all** — not
    ones-and-zeros parameters. `nnx.LayerNorm(use_scale=False, use_bias=False)`
    is the mirror, and the converter must not look for keys that do not exist.
  * **`chunk(6, dim=-1)` order is s1, c1, g1, s2, c2, g2** and
    `chunk(2, dim=-1)` is s, c. A permuted split still trains.
  * **The zero-init layers make a FRESH head output exactly 0.0**, which is
    designed in (`ml/field_model.py` decisions 1 and 2). That also means every
    parity gate run on a freshly-initialised head is comparing zeros to zeros
    and proves nothing — which is why `tests/test_jaxport_field.py` perturbs
    every torch parameter before converting and runs each gate twice.

WHAT IS DELIBERATELY NOT MIRRORED, and cannot be:

  * **`sample()`'s RNG stream.** torch draws `torch.randn(..., generator=g)`;
    this draws `jax.random.normal`. No two frameworks share a stream, so a
    JAX member m and a torch member m are DIFFERENT DRAWS from the same law.
    Cross-framework identity exists only through `sample_from`, which takes
    `x_init` as an argument for exactly that reason — the same split
    `ml/field_model.py` made for `edm_loss` / `edm_loss_given`.
  * **float64.** `sigma_ladder` is computed in torch at float64 and cast; JAX
    disables x64 by default and enabling it globally would change every other
    number in the process. The ladder is therefore computed in NUMPY float64 —
    the same IEEE arithmetic torch does, on the host, once per sampler call —
    and cast to float32 exactly as torch casts. It is a ~10-element array
    built outside every hot loop, so there is nothing to gain by moving it.

STATIC ATTRIBUTES, and why the tokenizer is one. flax treats a non-Variable
attribute as STATIC: it lands in the graphdef, which `nnx.split`/`nnx.merge`
and `jax.jit` require to be HASHABLE. `models.py` solved that for the FSQ
constants by storing plain tuples. The pixel index arrays here are up to 84,405
entries long and would be hashed on every jit dispatch, so `OceanTokenizerJax`
instead declares identity hash/eq: one tokenizer object is one static value,
hashed in O(1), and the numpy arrays inside it become jit constants at trace
time. `sigma_data` (a float) and `fourier_f` (an 8-tuple) follow `models.py`'s
tuple precedent directly.

    from jaxport.field_model import FieldHeadJax, OceanTokenizerJax
"""
import math
import os
import sys

import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx

_HERE = os.path.dirname(os.path.abspath(__file__))
_ML = os.path.dirname(_HERE)
if _ML not in sys.path:
    sys.path.insert(0, _ML)

from jaxport.models import MultiHeadAttention, gelu_exact       # noqa: E402
from jaxport.convert import (_Consumer, _Emitter, _emit_attention,  # noqa: E402,E501
                             _emit_layernorm, _emit_linear, _encoder,
                             _layernorm, _linear, _packed_attention, _param)

_nnx_data = getattr(nnx, "data", lambda x: x)

N_FOURIER = 8            # `ml/field_model.py:FieldHead.N_FOURIER`


# ---------------------------------------------------------------------------
# tokenizer
# ---------------------------------------------------------------------------
class OceanTokenizerJax:
    """`ml/field_model.py:OceanTokenizer`, bitwise.

    The torch original is PURE INDEXING — a zero-fill, a copy and a gather,
    with no arithmetic anywhere on the path — so this mirror prepares the same
    indices in numpy (the identical `np.unique` / `//` / `%` arithmetic, so the
    TOKEN ORDER cannot diverge) and applies them with `jnp.take`. The scatter
    is expressed as a GATHER over a precomputed `px_of_flat` (-1 = "this slot
    is not an ocean pixel") rather than as `.at[idx].set(...)`, because that
    form is what lets the trainer tokenize ONE CHUNK OF TOKENS at a time: at
    K=144, ntok~5.3k and feat 528 the whole [B, K, ntok, feat] context is
    ~6.4 GB at B=4, and it never has to exist.

    `jnp.where(flag, v, 0.0)` rather than `v * flag`: multiplying a negative
    value by 0.0 gives -0.0, and while -0.0 compares equal to 0.0 everywhere
    downstream, "bitwise" should mean bitwise.

    HASHABLE BY IDENTITY (see the module docstring): this object is a static
    attribute of the NNX modules that use it.
    """

    __hash__ = object.__hash__

    def __eq__(self, other):
        return self is other

    def __init__(self, H, W, ys, xs, patch):
        ys = np.asarray(ys, dtype=np.int64)
        xs = np.asarray(xs, dtype=np.int64)
        if ys.shape != xs.shape or ys.ndim != 1:
            raise ValueError("ys/xs must be 1-D arrays of the same length")
        if ys.size == 0:
            raise ValueError("no ocean pixels")
        if ys.min() < 0 or ys.max() >= H or xs.min() < 0 or xs.max() >= W:
            raise ValueError("ys/xs outside the HxW grid")
        self.H, self.W, self.patch = int(H), int(W), int(patch)
        self.P = int(ys.size)
        self.P2 = self.patch * self.patch
        self.Hp = (self.H + self.patch - 1) // self.patch
        self.Wp = (self.W + self.patch - 1) // self.patch

        py, px = ys // self.patch, xs // self.patch
        cell = py * self.Wp + px
        uniq = np.unique(cell)                      # ASCENDING == (py, px) order
        self.ntok = int(uniq.size)
        tok_of_cell = np.full(self.Hp * self.Wp, -1, np.int64)
        tok_of_cell[uniq] = np.arange(self.ntok, dtype=np.int64)

        tok_of_px = tok_of_cell[cell]
        slot_of_px = (ys % self.patch) * self.patch + (xs % self.patch)
        self.flat_of_px = (tok_of_px * self.P2 + slot_of_px).astype(np.int64)
        self.tok_of_px = tok_of_px
        self.slot_of_px = slot_of_px
        self.ys, self.xs = ys, xs
        self.ocean_lin = ys * self.W + xs

        self.tok_py = (uniq // self.Wp).astype(np.int64)
        self.tok_px = (uniq % self.Wp).astype(np.int64)
        self.tok_coord = np.stack([
            (self.tok_py.astype(np.float64) + 0.5) / self.Hp,
            (self.tok_px.astype(np.float64) + 0.5) / self.Wp],
            axis=1).astype(np.float32)

        # THE INVERSE INDEX: which pixel (if any) owns each flat token slot.
        # -1 is land or out-of-grid, which is exactly the slot the ocean FLAG
        # channel exists to distinguish from "an ocean pixel whose value is 0".
        self.n_flat = self.ntok * self.P2
        self.px_of_flat = np.full(self.n_flat, -1, np.int64)
        self.px_of_flat[self.flat_of_px] = np.arange(self.P, dtype=np.int64)

    # -- the torch API, term for term ---------------------------------------
    def feat_in(self, d_z):
        return self.P2 * (int(d_z) + 1)

    def feat_out(self, d_z):
        return self.P2 * int(d_z)

    def to_tokens(self, z):
        """[.., P, d_z] -> [.., ntok, patch*patch*(d_z+1)]. Leading dims free."""
        return self.to_tokens_rows(z, 0, self.n_flat)

    def to_tokens_rows(self, z, lo, hi):
        """`to_tokens` restricted to flat slot rows [lo, hi).

        `hi - lo` must be a whole number of tokens. This is the chunked form
        the trainer's conditioner uses; `to_tokens` is the whole-array case of
        it, so there is one implementation and not two.
        """
        if z.shape[-2] != self.P:
            raise ValueError(f"expected P={self.P} pixels, got {z.shape[-2]}")
        if (hi - lo) % self.P2:
            raise ValueError("row range is not a whole number of tokens")
        d_z = z.shape[-1]
        n = (hi - lo) // self.P2
        # A 2-D INDEX [n_tokens, P2], not a flat one, so the only reshape left
        # on the path is a merge of the two TRAILING axes. Splitting a middle
        # axis is the one reshape JAX's sharding-in-types cannot infer through
        # (measured on a 4-device CPU mesh: `[B@b, K, R, C] -> [B, K, n, P2*C]`
        # raises ShardingTypeError), and this trainer shards the batch axis.
        idx2 = self.px_of_flat[lo:hi].reshape(n, self.P2)
        safe = jnp.asarray(np.maximum(idx2, 0))
        ok = jnp.asarray(idx2 >= 0)
        v = jnp.take(z, safe, axis=-2)                # [.., n, P2, d_z]
        v = jnp.where(ok[..., None], v, jnp.zeros((), z.dtype))
        flag = jnp.broadcast_to(
            jnp.where(ok, jnp.ones((), z.dtype), jnp.zeros((), z.dtype)
                      )[..., None], v.shape[:-1] + (1,))
        buf = jnp.concatenate([v, flag], axis=-1)      # [.., n, P2, d_z+1]
        lead = z.shape[:-2]
        return buf.reshape(*lead, n, self.P2 * (d_z + 1))

    def to_pixels(self, tok, d_z=None):
        """[.., ntok, patch*patch*C] -> [.., P, d_z]. Ocean slots only.

        Both token layouts are accepted for the same reason the torch original
        accepts both: C == d_z is the backbone's OUTPUT stream, C == d_z + 1 is
        the CONTENT stream whose trailing ocean flag is dropped here. With
        `d_z` omitted the output layout is assumed, because that is the call
        the model makes on every forward.
        """
        if tok.shape[-1] % self.P2:
            raise ValueError("token width is not a multiple of patch^2")
        C = tok.shape[-1] // self.P2
        d_z = C if d_z is None else int(d_z)
        if C not in (d_z, d_z + 1):
            raise ValueError(f"token width implies {C} channels/slot, which is "
                             f"neither d_z={d_z} nor d_z+1")
        lead = tok.shape[:-2]
        # TWO reshapes, not one, for the sharding reason in `to_tokens_rows`:
        # a split of the LAST axis and a merge of two MIDDLE ones are both
        # inferable, while the single combined step is not.
        flat = tok.reshape(*lead, self.ntok, self.P2, C)
        flat = flat.reshape(*lead, self.ntok * self.P2, C)
        out = jnp.take(flat, jnp.asarray(self.flat_of_px), axis=-2)
        return out[..., :d_z]

    @classmethod
    def from_torch(cls, tok):
        """Rebuild from a torch `OceanTokenizer`, from ITS OWN ys/xs.

        Not from its derived indices: rebuilding from the inputs is what makes
        the index arithmetic a second implementation that can then be checked
        against the first (F1), instead of a copy that agrees by construction.
        """
        return cls(tok.H, tok.W,
                   np.asarray(tok.ys.detach().cpu().numpy()),
                   np.asarray(tok.xs.detach().cpu().numpy()), tok.patch)


# ---------------------------------------------------------------------------
# the gelu encoder layer — torch's `activation="gelu"` twin of models.py's
# ---------------------------------------------------------------------------
class _EncoderLayerGelu(nnx.Module):
    """`nn.TransformerEncoderLayer(..., activation="gelu", norm_first=True,
    dropout=0.0, batch_first=True)`.

        x = x + self_attn(norm1(x))
        x = x + linear2(gelu(linear1(norm2(x))))

    Identical to `models.EncoderLayer` except for the activation — which is
    the whole point of it existing (see the module docstring). Attribute names
    match, so `convert._encoder` / `_emit_encoder` map it unchanged.
    """

    def __init__(self, d_model, n_heads, dim_feedforward, *, rngs):
        self.self_attn = MultiHeadAttention(d_model, n_heads, rngs=rngs)
        self.linear1 = nnx.Linear(d_model, dim_feedforward, rngs=rngs)
        self.linear2 = nnx.Linear(dim_feedforward, d_model, rngs=rngs)
        # 1e-5 is torch's LayerNorm default; Flax's is 1e-6.
        self.norm1 = nnx.LayerNorm(d_model, epsilon=1e-5, rngs=rngs)
        self.norm2 = nnx.LayerNorm(d_model, epsilon=1e-5, rngs=rngs)

    def __call__(self, x, mask=None):
        h = self.norm1(x)
        x = x + self.self_attn(h, h, h, mask=mask)
        h = self.norm2(x)
        x = x + self.linear2(gelu_exact(self.linear1(h)))
        return x


class _TransformerEncoderGelu(nnx.Module):
    """`nn.TransformerEncoder(layer, n)` — NO final norm."""

    def __init__(self, d_model, n_heads, n_layers, dim_feedforward=None, *,
                 rngs):
        if dim_feedforward is None:
            dim_feedforward = 4 * d_model
        self.layers = _nnx_data(
            [_EncoderLayerGelu(d_model, n_heads, dim_feedforward, rngs=rngs)
             for _ in range(n_layers)])

    def __call__(self, x, mask=None):
        for lyr in self.layers:
            x = lyr(x, mask=mask)
        return x


def causal_bool_mask(k):
    """`torch.triu(torch.ones(K, K, dtype=bool), diagonal=1)` — TRUE MEANS
    MASKED OUT, which is torch's `attn_mask` convention and the opposite of
    Flax's own. `models.MultiHeadAttention` takes the torch convention."""
    i = jnp.arange(k)[:, None]
    j = jnp.arange(k)[None, :]
    return j > i


# ---------------------------------------------------------------------------
# temporal conditioner
# ---------------------------------------------------------------------------
class TemporalCondJax(nnx.Module):
    """`ml/field_model.py:TemporalCond`. TIME ONLY — no cross-token mixing."""

    def __init__(self, feat_in, d_cond, K, layers=2, heads=4, season_dim=2,
                 *, rngs=None):
        rngs = rngs if rngs is not None else nnx.Rngs(0)
        self.K, self.d_cond = int(K), int(d_cond)
        self.season_dim = int(season_dim)
        self.proj = nnx.Linear(int(feat_in) + self.season_dim, self.d_cond,
                               rngs=rngs)
        self.tpos = nnx.Param(jnp.zeros((self.K, self.d_cond)))
        self.enc = _TransformerEncoderGelu(self.d_cond, int(heads),
                                           int(layers), 4 * self.d_cond,
                                           rngs=rngs)
        self.out_norm = nnx.LayerNorm(self.d_cond, epsilon=1e-5, rngs=rngs)

    def __call__(self, ctx_tokens, season=None):
        """[B, K, ntok, feat_in] (+ [B, K, 2]) -> [B, ntok, d_cond]."""
        B, K, N, _ = ctx_tokens.shape
        if K != self.K:
            raise ValueError(f"TemporalCondJax built for K={self.K}, got {K}")
        if season is None:
            season = jnp.zeros((B, K, self.season_dim), ctx_tokens.dtype)
        x = jnp.concatenate(
            [ctx_tokens,
             jnp.broadcast_to(season[:, :, None, :], (B, K, N,
                                                      self.season_dim))],
            axis=-1)
        h = self.proj(x) + self.tpos.value[None, :, None, :]
        # Fold tokens into the batch: [B, K, N, d] -> [B*N, K, d]. The encoder
        # then sees BxN independent length-K sequences and cannot mix tokens
        # even by accident.
        h = jnp.transpose(h, (0, 2, 1, 3)).reshape(B * N, K, self.d_cond)
        h = self.enc(h, mask=causal_bool_mask(K))
        h = self.out_norm(h[:, -1])                       # last step only
        return h.reshape(B, N, self.d_cond)


# ---------------------------------------------------------------------------
# DiT backbone
# ---------------------------------------------------------------------------
def _zero_linear(in_f, out_f, rngs):
    """`ml/field_model.py:_zero_linear` — weight AND bias exactly zero, so the
    layer emits bitwise 0.0 and "det mode at init IS persistence" is an
    identity rather than an approximation."""
    return nnx.Linear(in_f, out_f,
                      kernel_init=nnx.initializers.zeros_init(),
                      bias_init=nnx.initializers.zeros_init(), rngs=rngs)


def _modulate(x, shift, scale):
    return x * (1.0 + scale[:, None, :]) + shift[:, None, :]


class _BlockJax(nnx.Module):
    """LN -> full self-attention over tokens -> LN -> MLP, adaLN-zero on g.

    `n1`/`n2` are `elementwise_affine=False, eps=1e-6`, i.e. NO PARAMETERS —
    not unit-scale parameters. The converter therefore never asks for them,
    and a state_dict that carried them would be refused as unconsumed.
    """

    def __init__(self, d_model, heads, d_g, mlp_ratio=4, *, rngs):
        self.n1 = nnx.LayerNorm(d_model, epsilon=1e-6, use_scale=False,
                                use_bias=False, rngs=rngs)
        self.attn = MultiHeadAttention(d_model, heads, rngs=rngs)
        self.n2 = nnx.LayerNorm(d_model, epsilon=1e-6, use_scale=False,
                                use_bias=False, rngs=rngs)
        self.mlp1 = nnx.Linear(d_model, mlp_ratio * d_model, rngs=rngs)
        self.mlp2 = nnx.Linear(mlp_ratio * d_model, d_model, rngs=rngs)
        self.ada = _zero_linear(d_g, 6 * d_model, rngs)

    def __call__(self, x, g):
        s1, c1, g1, s2, c2, g2 = jnp.split(self.ada(jax.nn.silu(g)), 6,
                                           axis=-1)
        h = _modulate(self.n1(x), s1, c1)
        a = self.attn(h, h, h)
        x = x + g1[:, None, :] * a
        h = _modulate(self.n2(x), s2, c2)
        # nn.GELU() — the EXACT erf form.
        return x + g2[:, None, :] * self.mlp2(gelu_exact(self.mlp1(h)))


class FieldDiTJax(nnx.Module):
    """`ml/field_model.py:FieldDiT` — attention over SPACE, conditioned on g.

    `tok_py` / `tok_px` are torch BUFFERS (persistent, so they ride in the
    state_dict) and are pure geometry. They live on the tokenizer object here,
    which is a static attribute; the converter checks the state_dict's copies
    against them rather than overwriting, so a head built on a different ocean
    mask REFUSES instead of loading and producing plausible numbers.
    """

    def __init__(self, feat_in, feat_out, d_model, layers, heads, d_cond, d_g,
                 tok, *, rngs=None):
        rngs = rngs if rngs is not None else nnx.Rngs(0)
        if d_model % 2:
            raise ValueError("d_model must be even (row/col position halves)")
        self.d_model = int(d_model)
        self.tok = tok
        self.x_embed = nnx.Linear(int(feat_in), self.d_model, rngs=rngs)
        self.c_embed = nnx.Linear(int(d_cond), self.d_model, rngs=rngs)
        half = self.d_model // 2
        self.pos_row = nnx.Param(jnp.zeros((int(tok.Hp), half)))
        self.pos_col = nnx.Param(jnp.zeros((int(tok.Wp), half)))
        self.blocks = _nnx_data([
            _BlockJax(self.d_model, int(heads), int(d_g), rngs=rngs)
            for _ in range(int(layers))])
        self.fin_norm = nnx.LayerNorm(self.d_model, epsilon=1e-6,
                                      use_scale=False, use_bias=False,
                                      rngs=rngs)
        self.fin_ada = _zero_linear(int(d_g), 2 * self.d_model, rngs)
        self.fin = _zero_linear(self.d_model, int(feat_out), rngs)

    def pos_embed(self):
        py = jnp.asarray(self.tok.tok_py)
        px = jnp.asarray(self.tok.tok_px)
        return jnp.concatenate([self.pos_row.value[py],
                                self.pos_col.value[px]], axis=-1)[None]

    def __call__(self, x_tokens, cond_tokens, g):
        x = (self.x_embed(x_tokens) + self.c_embed(cond_tokens)
             + self.pos_embed())
        for blk in self.blocks:
            x = blk(x, g)
        s, c = jnp.split(self.fin_ada(jax.nn.silu(g)), 2, axis=-1)
        return self.fin(_modulate(self.fin_norm(x), s, c))


# ---------------------------------------------------------------------------
# the full head
# ---------------------------------------------------------------------------
class FieldHeadJax(nnx.Module):
    """`ml/field_model.py:FieldHead` — tokenizer + conditioner + backbone +
    (in `diff` mode) EDM and a sampler.

    `sigma_data` and `fourier_f` are torch BUFFERS, so they ride in the
    state_dict and must round-trip; they are held here as a python float and
    an 8-tuple, i.e. as STATIC values (the `models.py` precedent), so the
    parameter pytree contains exactly the tensors torch calls parameters and
    a gradient comparison is over the same set on both sides.

    Anything that changes a static value changes the GRAPHDEF, so a caller
    holding a `nnx.split` must re-split after `load_field_head`. The trainer
    splits after loading and says so there.
    """

    N_FOURIER = N_FOURIER

    def __init__(self, tok, d_z, K, mode="det", d_model=128, layers=4,
                 heads=4, d_cond=128, cond_layers=2, cond_heads=4,
                 sigma_data=1.0, season_dim=2, *, rngs=None):
        rngs = rngs if rngs is not None else nnx.Rngs(0)
        if mode not in ("det", "diff"):
            raise ValueError("mode must be 'det' or 'diff'")
        self.tok, self.mode = tok, mode
        self.d_z, self.K = int(d_z), int(K)
        self.d_model = int(d_model)
        feat_in = tok.feat_in(self.d_z)
        feat_out = tok.feat_out(self.d_z)
        self.cond = TemporalCondJax(feat_in, int(d_cond), self.K,
                                    layers=cond_layers, heads=cond_heads,
                                    season_dim=season_dim, rngs=rngs)
        self.dit = FieldDiTJax(feat_in, feat_out, self.d_model, layers, heads,
                               int(d_cond), self.d_model, tok, rngs=rngs)
        self.sigma_data = float(sigma_data)
        # `torch.exp(torch.linspace(0, log(1000), 8))` — computed in numpy
        # float64 and narrowed, which is what torch does at float32 precision
        # for these eight values. The converter OVERWRITES it from the
        # state_dict anyway, so the round trip is exact whatever this says.
        self.fourier_f = tuple(
            float(v) for v in np.exp(np.linspace(0.0, math.log(1000.0),
                                                 N_FOURIER)).astype(np.float32))
        self.sig_mlp1 = nnx.Linear(2 * N_FOURIER, self.d_model, rngs=rngs)
        self.sig_mlp2 = nnx.Linear(self.d_model, self.d_model, rngs=rngs)
        self.g_null = nnx.Param(jnp.zeros((self.d_model,)))

    # -- conditioning --------------------------------------------------------
    def make_cond(self, ctx_tokens, season=None):
        """[B, K, ntok, feat_in] -> [B, ntok, d_cond]. Built ONCE per batch."""
        return self.cond(ctx_tokens, season)

    def g_of_sigma(self, sigma):
        """[B] sigma -> [B, d_model] global conditioning, via EDM's c_noise."""
        c_noise = jnp.log(sigma) / 4.0
        f = jnp.asarray(self.fourier_f, sigma.dtype)
        ang = 2.0 * math.pi * c_noise[:, None] * f[None, :]
        h = jnp.concatenate([jnp.sin(ang), jnp.cos(ang)], axis=-1)
        return self.sig_mlp2(jax.nn.silu(self.sig_mlp1(h)))

    def g_null_for(self, B, dtype=jnp.float32):
        return jnp.broadcast_to(self.g_null.value.astype(dtype),
                                (B, self.d_model))

    # -- deterministic mode --------------------------------------------------
    def residual_det(self, cond_tokens):
        """r_hat [B, P, d_z]. EXACTLY zero at init -> persistence."""
        B = cond_tokens.shape[0]
        x = jnp.zeros((B, self.tok.ntok, self.tok.feat_in(self.d_z)),
                      cond_tokens.dtype)
        out = self.dit(x, cond_tokens, self.g_null_for(B, cond_tokens.dtype))
        return self.tok.to_pixels(out, self.d_z)

    def forward_det(self, cond_tokens, z_t):
        """z_hat = z_t + r_hat. Bitwise == z_t at init."""
        return z_t + self.residual_det(cond_tokens)

    def __call__(self, ctx_tokens, z_t, season=None):
        return self.forward_det(self.make_cond(ctx_tokens, season), z_t)

    # -- EDM preconditioning -------------------------------------------------
    def _coefs(self, sigma):
        sd = jnp.asarray(self.sigma_data, sigma.dtype)
        s2 = sigma * sigma
        c_skip = sd * sd / (s2 + sd * sd)
        c_out = sigma * sd / jnp.sqrt(s2 + sd * sd)
        c_in = 1.0 / jnp.sqrt(s2 + sd * sd)
        return c_skip, c_out, c_in

    def D(self, x, sigma, cond_tokens):
        """EDM denoiser on the residual field. D == c_skip*x BITWISE at init."""
        c_skip, c_out, c_in = self._coefs(sigma)
        b = (slice(None),) + (None,) * (x.ndim - 1)
        tokens = self.tok.to_tokens(c_in[b] * x)
        out = self.dit(tokens, cond_tokens, self.g_of_sigma(sigma))
        return c_skip[b] * x + c_out[b] * self.tok.to_pixels(out, self.d_z)

    def edm_loss_given(self, cond_tokens, z_t, z_tp1, sigma, noise):
        """The EDM loss with (sigma [B], noise [B,P,d_z]) supplied — the
        PARITY SURFACE, and the deterministic half of `edm_loss`."""
        sd = float(self.sigma_data)
        r = z_tp1 - z_t
        b = (slice(None),) + (None,) * (r.ndim - 1)
        d = self.D(r + sigma[b] * noise, sigma, cond_tokens)
        lam = (sigma ** 2 + sd ** 2) / (sigma * sd) ** 2
        per = ((d - r) ** 2).reshape(r.shape[0], -1).mean(axis=1)
        return (lam * per).mean()

    def edm_loss(self, cond_tokens, z_t, z_tp1, key):
        """EDM's weighted denoising loss, with the draws taken HERE.

        Same lognormal parameterization as torch: sigma ~ exp(P_mean +
        P_std * N(0,1)) with P_mean = log(sigma_data) - 1.2 and P_std = 1.2,
        i.e. EDM's published ladder RE-CENTRED on this data's own residual RMS
        (`ml/field_model.py:edm_loss` argues why).

        The draws come from a JAX key and therefore form a DIFFERENT SAMPLE
        PATH from a torch run at the same seed — a fresh-run fact, never a
        parity bug, which is exactly why `edm_loss_given` exists beside it.
        """
        B = z_t.shape[0]
        sd = float(self.sigma_data)
        p_mean = math.log(sd) - 1.2
        p_std = 1.2
        k1, k2 = jax.random.split(key)
        dt = z_t.dtype
        n = jax.random.normal(k1, (B,), dt)
        sigma = jnp.exp(p_mean + p_std * n)
        noise = jax.random.normal(k2, (z_tp1 - z_t).shape, dt)
        return self.edm_loss_given(cond_tokens, z_t, z_tp1, sigma, noise)

    # -- sampler -------------------------------------------------------------
    def sigma_ladder(self, n_steps, sigma_min=None, sigma_max=None, rho=7.0,
                     dtype=np.float32):
        """Karras rho-ladder with a trailing exact 0, in NUMPY float64.

        torch builds this at float64 and casts; JAX has x64 off by default and
        turning it on globally would move every other number in the process,
        so the host does the float64 arithmetic instead. It is a ~n+1 element
        array built once per sampler call, outside every hot loop.

        Defaults carry EDM's published 0.002 / 80 as RATIOS of sigma_data
        (they are quoted at sigma_data = 0.5), for the reason
        `ml/field_model.py` gives: the ladder must span the range over which
        the signal goes from visible to drowned, and that range is a property
        of the data's scale.
        """
        sd = float(self.sigma_data)
        smin = sd * 4e-3 if sigma_min is None else float(sigma_min)
        smax = sd * 160.0 if sigma_max is None else float(sigma_max)
        n = int(n_steps)
        i = np.arange(n, dtype=np.float64)
        t = np.zeros(1, np.float64) if n == 1 else i / (n - 1)
        a, b = smax ** (1.0 / rho), smin ** (1.0 / rho)
        s = (a + t * (b - a)) ** rho
        return np.concatenate([s, np.zeros(1, np.float64)]).astype(dtype)

    def sample_from(self, cond_tokens, z_t, x_init, sig):
        """One Heun integration from a SUPPLIED initial state, down a SUPPLIED
        ladder — the deterministic core of `sample` and THE CROSS-FRAMEWORK
        PARITY SURFACE. Returns z_t + r, shape [B, P, d_z].

        `sig` is read on the HOST for the `float(s_n) > 0.0` branch, exactly as
        torch does; the ladder is a small concrete array by construction, so
        this costs no device sync in any real call.
        """
        B = z_t.shape[0]
        sig_np = np.asarray(sig, np.float32)
        sig_j = jnp.asarray(sig_np, z_t.dtype)
        x = x_init
        for i in range(len(sig_np) - 1):
            s_i, s_n = sig_j[i], sig_j[i + 1]
            sv = jnp.broadcast_to(s_i, (B,))
            d = (x - self.D(x, sv, cond_tokens)) / s_i
            x_next = x + (s_n - s_i) * d
            if float(sig_np[i + 1]) > 0.0:
                sv2 = jnp.broadcast_to(s_n, (B,))
                d2 = (x_next - self.D(x_next, sv2, cond_tokens)) / s_n
                x_next = x + (s_n - s_i) * 0.5 * (d + d2)
            x = x_next
        return z_t + x

    def sample(self, cond_tokens, z_t, n_steps, key, M=1, sigma_min=None,
               sigma_max=None, rho=7.0):
        """Deterministic 2nd-order Heun on the probability-flow ODE.

        **THE STREAM DOES NOT MATCH TORCH'S, AND CANNOT.** torch draws
        `torch.randn(..., generator=g)`; this draws `jax.random.normal` off a
        threefry key. Member m here is a different draw from the same law than
        member m there, so no cross-framework gate may run through this method
        — `sample_from` with an INJECTED `x_init` is the surface that isolates
        the integrator and the model from the sampler.

        Members are derived with `jax.random.fold_in(key, m)`, which preserves
        the torch sampler's own guarantee in JAX terms: member m of an
        M-member call is bit-identical to member m of any other call with the
        same key, so a member never depends on M.

        Returns z_t + r, shaped [M, B, P, d_z].
        """
        sig = self.sigma_ladder(n_steps, sigma_min, sigma_max, rho)
        outs = []
        for m in range(int(M)):
            km = jax.random.fold_in(key, m)
            x = jax.random.normal(km, z_t.shape, z_t.dtype) * jnp.asarray(
                sig[0], z_t.dtype)
            outs.append(self.sample_from(cond_tokens, z_t, x, sig))
        return jnp.stack(outs, axis=0)


def nfe_to_steps(nfe):
    """Heun costs 2 evaluations per step except the last (sigma=0 is skipped),
    so N steps cost 2N-1 and a budget of `nfe` buys `ceil((nfe+1)/2)` steps."""
    n = max(1, (int(nfe) + 1) // 2)
    return n, 2 * n - 1


def count_params_jax(model):
    _, state = nnx.split(model)
    return sum(int(np.prod(np.shape(v)))
               for v in jax.tree_util.tree_leaves(state))


# ---------------------------------------------------------------------------
# converters — convert.py's refusal contract, verbatim
# ---------------------------------------------------------------------------
def _emit_raw(e, key, arr, dtype):
    """A torch BUFFER emitted at ITS OWN dtype. `convert._np` narrows every
    value to float32, which is right for weights and wrong for an index table:
    `tok_py` is int64 in the torch module and `torch.equal` in the export gate
    compares dtype as well as value. The duplicate-key check is `_Emitter`'s
    and is repeated here rather than bypassed."""
    if key in e.sd:
        raise KeyError(f"{e.what}: {key!r} emitted twice")
    e.sd[key] = np.array(np.asarray(arr), dtype=dtype, order="C", copy=True)


def _emit_int(e, key, arr):
    _emit_raw(e, key, arr, np.int64)


def _emit_f32(e, key, value):
    _emit_raw(e, key, value, np.float32)


def _cond_prefix(dst, c, prefix):
    _linear(dst.proj, c, prefix + ".proj")
    _param(dst.tpos, c, prefix + ".tpos")
    _encoder(dst.enc, c, prefix + ".enc")
    _layernorm(dst.out_norm, c, prefix + ".out_norm")


def _dit_prefix(dst, c, prefix, tok, what):
    _param(dst.pos_row, c, prefix + ".pos_row")
    _param(dst.pos_col, c, prefix + ".pos_col")
    # THE GEOMETRY BUFFERS ARE CHECKED, NOT ADOPTED. They are a function of
    # the ocean mask, and this model already built its own from ys/xs; a
    # state_dict whose tables disagree was trained on a different mask, and
    # loading it would produce a head whose every position embedding points at
    # the wrong cell while every shape still matched.
    for name, mine in (("tok_py", tok.tok_py), ("tok_px", tok.tok_px)):
        got = c.get(f"{prefix}.{name}")
        if got is not None:
            got = np.asarray(got, np.int64).reshape(-1)
            if got.shape != mine.shape or not np.array_equal(got, mine):
                raise ValueError(
                    f"{what}: {prefix}.{name} in the state_dict does not match "
                    f"the tokenizer this model was built with "
                    f"({got.shape} vs {mine.shape}). The checkpoint was "
                    f"trained on a different ocean mask or a different patch "
                    f"size; rebuild the tokenizer from that run's ys/xs "
                    f"rather than loading a head whose positions point at the "
                    f"wrong cells.")
    _linear(dst.x_embed, c, prefix + ".x_embed")
    _linear(dst.c_embed, c, prefix + ".c_embed")
    for i, blk in enumerate(dst.blocks):
        p = f"{prefix}.blocks.{i}"
        _packed_attention(blk.attn, c, p + ".attn")
        # nn.Sequential(Linear, GELU, Linear): linears at 0 and 2.
        _linear(blk.mlp1, c, p + ".mlp.0")
        _linear(blk.mlp2, c, p + ".mlp.2")
        # nn.Sequential(SiLU, Linear): the Linear is index 1.
        _linear(blk.ada, c, p + ".ada.1")
    _linear(dst.fin_ada, c, prefix + ".fin_ada.1")
    _linear(dst.fin, c, prefix + ".fin")


def load_field_head(state_dict, model):
    """torch `FieldHead.state_dict()` -> the NNX head, REFUSING a partial load.

    Same contract as every loader in `convert.py`: every torch key is recorded
    as it is consumed, and the load refuses at the end if the state_dict
    carried a key nothing wanted or the model wanted a key it did not carry.
    A half-matched checkpoint still produces a forward pass and still produces
    plausible numbers, which is the failure shape this programme fears most.

    The two BUFFERS that are not geometry — `sigma_data` and `fourier_f` — are
    ADOPTED from the file rather than kept, so a re-export is bit-exact and so
    a head trained at one residual RMS can never be silently scored at
    another. Both are static values, so **the caller must re-split after this**
    (`models.set_fsq_ladder` says the same in the same words).
    """
    c = _Consumer(dict(state_dict), "load_field_head")
    _param(model.g_null, c, "g_null")
    sd = c.get("sigma_data")
    if sd is not None:
        model.sigma_data = float(np.asarray(sd).reshape(()))
    ff = c.get("fourier_f")
    if ff is not None:
        ff = np.asarray(ff, np.float32).reshape(-1)
        if ff.shape != (N_FOURIER,):
            raise ValueError(f"load_field_head: fourier_f is {ff.shape}, this "
                             f"model has N_FOURIER={N_FOURIER}")
        model.fourier_f = tuple(float(v) for v in ff)
    _cond_prefix(model.cond, c, "cond")
    _dit_prefix(model.dit, c, "dit", model.tok, "load_field_head")
    # nn.Sequential(Linear, SiLU, Linear): linears at 0 and 2.
    _linear(model.sig_mlp1, c, "sig_mlp.0")
    _linear(model.sig_mlp2, c, "sig_mlp.2")
    c.finish()
    return model


def export_field_head(model):
    """NNX `FieldHeadJax` -> a dict of numpy arrays in torch's key ORDER.

    The reverse of `load_field_head`, one list read backwards. Key order is
    cosmetic to `load_state_dict` (it is keyed, not positional) but makes a
    diff of two state_dicts readable, and emitting in the torch module's own
    construction order is what lets the export gate compare the two key
    SEQUENCES rather than two sets.
    """
    e = _Emitter("export_field_head")
    e.put("g_null", model.g_null.value)
    _emit_f32(e, "sigma_data", float(model.sigma_data))
    _emit_f32(e, "fourier_f", np.asarray(model.fourier_f, np.float32))
    e.put("cond.tpos", model.cond.tpos.value)
    _emit_linear(model.cond.proj, e, "cond.proj")
    for i, lyr in enumerate(model.cond.enc.layers):
        p = f"cond.enc.layers.{i}"
        _emit_attention(lyr.self_attn, e, p + ".self_attn")
        _emit_linear(lyr.linear1, e, p + ".linear1")
        _emit_linear(lyr.linear2, e, p + ".linear2")
        _emit_layernorm(lyr.norm1, e, p + ".norm1")
        _emit_layernorm(lyr.norm2, e, p + ".norm2")
    _emit_layernorm(model.cond.out_norm, e, "cond.out_norm")
    e.put("dit.pos_row", model.dit.pos_row.value)
    e.put("dit.pos_col", model.dit.pos_col.value)
    _emit_int(e, "dit.tok_py", model.tok.tok_py)
    _emit_int(e, "dit.tok_px", model.tok.tok_px)
    _emit_linear(model.dit.x_embed, e, "dit.x_embed")
    _emit_linear(model.dit.c_embed, e, "dit.c_embed")
    for i, blk in enumerate(model.dit.blocks):
        p = f"dit.blocks.{i}"
        _emit_attention(blk.attn, e, p + ".attn")
        _emit_linear(blk.mlp1, e, p + ".mlp.0")
        _emit_linear(blk.mlp2, e, p + ".mlp.2")
        _emit_linear(blk.ada, e, p + ".ada.1")
    _emit_linear(model.dit.fin_ada, e, "dit.fin_ada.1")
    _emit_linear(model.dit.fin, e, "dit.fin")
    _emit_linear(model.sig_mlp1, e, "sig_mlp.0")
    _emit_linear(model.sig_mlp2, e, "sig_mlp.2")
    return e.sd


def export_field_pt(model, args, path=None, **extra):
    """A `.pt` blob the UNCHANGED torch `ml/field_model.py` head loads.

    Shape is `{model, args}` plus whatever `extra` carries — the same shape
    `ml/train_field.py:save_ckpt` writes minus the resumable halves, and for
    the same reason `convert.export_temporal_pt` omits them: **NO `opt`, NO
    `gen`.** optax's state is not torch AdamW's state and a torch
    `torch.Generator` state is not a JAX key; a blob carrying torch-shaped
    moments this trainer never produced would be a continuation wearing a warm
    restart's clothes. The resumable state is the sibling `.npz`.

    `args` must carry the fields a torch reader rebuilds the geometry from —
    `mode`, `K`, `patch`, `d_model`, `layers`, `heads`, `d_cond`,
    `cond_layers`, `cond_heads`, `d_z`, `sigma_data`, and `H`/`W`/the pixel
    list by whatever route the run publishes them. `args["backend"]` is
    stamped `jax`, because `ml/CLAUDE.md` §3b makes a JAX-trained number a new
    tier and this is the one line a reader sees.
    """
    import torch                                   # local: JAX users need not
    args = dict(vars(args)) if not isinstance(args, dict) else dict(args)
    args.setdefault("backend", "jax")
    sd = {k: torch.from_numpy(np.ascontiguousarray(v))
          for k, v in export_field_head(model).items()}
    blob = {"model": sd, "args": args}
    blob.update(extra)
    if path is not None:
        tmp = str(path) + ".tmp"
        torch.save(blob, tmp)
        os.replace(tmp, path)          # flush, THEN publish (§5.21)
    return blob


def field_head_from_torch(tmodel, *, rngs=None):
    """Build the NNX twin of a live torch `FieldHead` and load its weights.

    One call, so a test (or a converter script) cannot get the geometry and
    the weights out of step: every dimension is read off the torch module
    itself, never restated.
    """
    tok = OceanTokenizerJax.from_torch(tmodel.tok)
    jmod = FieldHeadJax(
        tok, tmodel.d_z, tmodel.K, mode=tmodel.mode,
        d_model=tmodel.d_model, layers=len(tmodel.dit.blocks),
        heads=tmodel.dit.blocks[0].attn.num_heads,
        d_cond=tmodel.cond.d_cond, cond_layers=len(tmodel.cond.enc.layers),
        cond_heads=tmodel.cond.enc.layers[0].self_attn.num_heads,
        sigma_data=float(tmodel.sigma_data),
        season_dim=tmodel.cond.season_dim,
        rngs=rngs if rngs is not None else nnx.Rngs(0))
    return load_field_head(tmodel.state_dict(), jmod)

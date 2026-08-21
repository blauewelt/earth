"""torch → JAX weight conversion for the tier-1 models.

One direction only (`ml/plans/JAX_PORT.md` §6): tiers 1–2 need to READ the
published artefacts, never to write them.

The contract every loader here honours, and the reason the file is written
this way: **a partial load must be impossible.** A checkpoint whose keys
half-match the model still produces a forward pass, still produces numbers,
and those numbers are plausible — which is the failure shape this programme
fears most. So each loader records every torch key it consumes, and refuses
at the end if the state_dict carried a key nothing wanted (`unexpected`) or
if the model wanted a key the state_dict did not carry (`missing`), naming
the offenders in both cases.

Two torch layout facts the mapping turns on:

  * `nn.Linear.weight` is **[out, in]** (torch computes `x @ W.T`); Flax's
    `kernel` is [in, out]. Every linear is transposed on the way in.
  * `nn.MultiheadAttention` (and the attention inside
    `nn.TransformerEncoderLayer`) packs the three input projections into one
    `in_proj_weight` of shape **[3d, d]**, sliced **q, k, v IN THAT ORDER**,
    and likewise `in_proj_bias` [3d]. A wrong slice is silent.
"""
import jax.numpy as jnp
import numpy as np

from .models import PixelMAE, SectionHead, TemporalTransformer


def open_ckpt(path):
    """Load a published `.pt` blob on CPU.

    `weights_only=False` because these blobs are dicts carrying `args`,
    `norm`, `chan` and (post-2026-08-10) optimiser state alongside the
    tensors — not a bare state_dict.
    """
    import torch                                   # local: JAX users need not
    return torch.load(path, map_location="cpu", weights_only=False)


class _Consumer:
    """Bookkeeping for the refusal contract described in the module docstring."""

    def __init__(self, sd, what):
        self.sd = sd
        self.what = what
        self.used = set()
        self.missing = []

    def get(self, key):
        if key not in self.sd:
            self.missing.append(key)
            return None
        self.used.add(key)
        v = self.sd[key]
        return np.asarray(v.detach().cpu().numpy() if hasattr(v, "detach")
                          else v)

    def finish(self):
        extra = sorted(set(self.sd) - self.used)
        if self.missing or extra:
            raise KeyError(
                f"{self.what}: refusing a partial load. "
                f"missing from the state_dict ({len(self.missing)}): "
                f"{self.missing}; unconsumed torch keys ({len(extra)}): "
                f"{extra}")


def _linear(dst, c, prefix):
    """torch Linear [out, in] → Flax kernel [in, out]."""
    w = c.get(prefix + ".weight")
    b = c.get(prefix + ".bias")
    if w is not None:
        dst.kernel.value = jnp.asarray(w.T)
    if b is not None:
        dst.bias.value = jnp.asarray(b)


def _layernorm(dst, c, prefix):
    w = c.get(prefix + ".weight")
    b = c.get(prefix + ".bias")
    if w is not None:
        dst.scale.value = jnp.asarray(w)
    if b is not None:
        dst.bias.value = jnp.asarray(b)


def _embed(dst, c, key):
    w = c.get(key)
    if w is not None:
        dst.embedding.value = jnp.asarray(w)


def _param(dst, c, key):
    w = c.get(key)
    if w is not None:
        dst.value = jnp.asarray(w).reshape(dst.value.shape)


def _packed_attention(dst, c, prefix):
    """`in_proj_weight` [3d,d] → q, k, v slices IN THAT ORDER, plus out_proj.

    Used for both `nn.TransformerEncoderLayer.self_attn` and the standalone
    `nn.MultiheadAttention` in SectionHead — they share the packed layout.
    """
    w = c.get(prefix + ".in_proj_weight")
    b = c.get(prefix + ".in_proj_bias")
    if w is not None:
        d = w.shape[1]
        for i, lin in enumerate((dst.q_proj, dst.k_proj, dst.v_proj)):
            lin.kernel.value = jnp.asarray(w[i * d:(i + 1) * d].T)
    if b is not None:
        d = b.shape[0] // 3
        for i, lin in enumerate((dst.q_proj, dst.k_proj, dst.v_proj)):
            lin.bias.value = jnp.asarray(b[i * d:(i + 1) * d])
    _linear(dst.out_proj, c, prefix + ".out_proj")


def _encoder(dst, c, prefix):
    """`nn.TransformerEncoder` — `layers.N.{self_attn,linear1,linear2,
    norm1,norm2}`. No final norm: torch's `norm` is None (see models.py)."""
    for i, lyr in enumerate(dst.layers):
        # prefix="" addresses a BARE nn.TransformerEncoder's own state_dict
        # (`layers.0....`), which is what a single-layer parity check loads.
        p = f"{prefix}.layers.{i}" if prefix else f"layers.{i}"
        _packed_attention(lyr.self_attn, c, p + ".self_attn")
        _linear(lyr.linear1, c, p + ".linear1")
        _linear(lyr.linear2, c, p + ".linear2")
        _layernorm(lyr.norm1, c, p + ".norm1")
        _layernorm(lyr.norm2, c, p + ".norm2")


# --------------------------------------------------------------------------
# per-model loaders
# --------------------------------------------------------------------------
def load_pixelmae(state_dict, model):
    c = _Consumer(dict(state_dict), "load_pixelmae")
    _linear(model.val_proj, c, "val_proj")
    _embed(model.chan_emb, c, "chan_emb.weight")
    _param(model.mask_tok, c, "mask_tok")
    _param(model.miss_tok, c, "miss_tok")
    _param(model.cls_tok, c, "cls_tok")
    _linear(model.ctx_proj, c, "ctx_proj")
    _encoder(model.encoder, c, "encoder")
    _linear(model.to_z, c, "to_z")
    _embed(model.q_chan, c, "q_chan.weight")
    _embed(model.q_off, c, "q_off.weight")
    # nn.Sequential(Linear, GELU, Linear, GELU, ..., Linear): the LINEARS are
    # at even indices, the GELUs (parameterless) at odd ones.
    for i, lin in enumerate(model.decoder):
        _linear(lin, c, f"decoder.{2 * i}")
    c.finish()
    return model


def load_temporal(state_dict, model):
    c = _Consumer(dict(state_dict), "load_temporal")
    _linear(model.inp, c, "inp")
    _linear(model.static, c, "static")
    _embed(model.pos, c, "pos.weight")
    _encoder(model.encoder, c, "encoder")
    _linear(model.head, c, "head")
    if model.heads_direct is not None:
        for k, lin in model.heads_direct.items():
            _linear(lin, c, f"heads_direct.{k}")
    c.finish()
    return model


def load_section_head(state_dict, model):
    c = _Consumer(dict(state_dict), "load_section_head")
    _linear(model.lift, c, "lift")
    if model.blocks is not None:
        _encoder(model.blocks, c, "blocks")
    _param(model.q, c, "q")
    _packed_attention(model.att, c, "att")
    # nn.Sequential(LayerNorm, Linear(d,32), GELU, Linear(32,1))
    _layernorm(model.out_norm, c, "out.0")
    _linear(model.out_lin1, c, "out.1")
    _linear(model.out_lin2, c, "out.3")
    c.finish()
    return model


def codec_from_ckpt_jax(ck, n_chan):
    """`ml/model.py:codec_from_ckpt`, for the NNX PixelMAE.

    Same contract, same `.get()` defaults: the checkpoint's `args` carry the
    full architecture (train.py saves `vars(a)`), and old checkpoints predate
    the size knobs, so every default is the pilot architecture. `d_z` is a
    TOP-LEVEL key of the blob, not one of `args`.
    """
    a = ck.get("args", {})
    model = PixelMAE(n_chan=n_chan, d_z=ck["d_z"],
                     patch=a.get("patch", 1),
                     d_model=a.get("d_model", 128),
                     n_layers=a.get("n_layers", 4),
                     n_heads=a.get("n_heads", 4),
                     d_dec=a.get("d_dec", 256),
                     dec_layers=a.get("dec_layers", 2))
    return load_pixelmae(ck["model"], model)

"""E-069 · torch ↔ JAX weight conversion for `ConeMAE` / `ConeMAEJax`.

Both directions, and the same refusal contract `ml/jaxport/convert.py` states
in its own docstring: **a partial load must be impossible.** A checkpoint
whose keys half-match the model still produces a forward pass, still produces
numbers, and those numbers are plausible — which is the failure shape this
programme fears most. So the bookkeeping objects are IMPORTED from
`convert.py` (`_Consumer`, `_Emitter`) along with every element mapper
(`_linear`, `_layernorm`, `_embed`, `_param`, `_packed_attention`,
`_encoder`, and their `_emit_*` mirrors) rather than copied: a second copy of
the [out,in] → [in,out] transpose or of the q/k/v slice order is a second
thing that can drift, and a wrong slice trains and produces numbers.

The torch module has **99 state_dict keys at any geometry**; every one of them
is consumed by `load_cone` and re-emitted by `export_cone`, in the torch
module's own construction order, so a diff of two state_dicts is readable.

`coord.freqs` is NOT among them and must not be looked for: it is a
`register_buffer(..., persistent=False)` on the torch side and is recomputed
from `n_fourier` in `CoordEncJax.__init__`. A converter that tried to carry it
would refuse every real checkpoint on a "missing" key that has never existed.

`export_cone_pt` writes the SAME blob shape `ml/train_cone.py:train_one.save`
writes — `{args, model, chan_names, norm, step, arm, L_in, params}` — so the
velocity probe, the future `embed_cone.py` and any eval script read a
TPU-trained cone codec exactly as they read a GPU-trained one.
`args["backend"] = "jax"` is the one mark, and `ml/CLAUDE.md` §3b is what
makes that mark matter: a TPU-trained number is a new tier.
"""
from .cone_models import ConeMAEJax
from .convert import (_Consumer, _Emitter, _emit_attention, _emit_encoder,
                      _emit_layernorm, _emit_linear, _embed, _encoder,
                      _layernorm, _linear, _packed_attention, _param)


def _cross_block(dst, c, prefix):
    """One `CrossBlock`: `ln_q`, `ln_kv`, `attn` (packed), `ln_m`, `mlp.0/2`.

    The MLP is `nn.Sequential(Linear, GELU, Linear)`, so its linears sit at
    indices 0 and 2 — the GELU has no parameters and occupies index 1. The
    JAX side holds them as a two-element list for exactly that reason, so the
    mapping is `mlp[0] ↔ mlp.0`, `mlp[1] ↔ mlp.2`.
    """
    _layernorm(dst.ln_q, c, prefix + ".ln_q")
    _layernorm(dst.ln_kv, c, prefix + ".ln_kv")
    _packed_attention(dst.attn, c, prefix + ".attn")
    _layernorm(dst.ln_m, c, prefix + ".ln_m")
    _linear(dst.mlp[0], c, prefix + ".mlp.0")
    _linear(dst.mlp[1], c, prefix + ".mlp.2")


def _emit_cross_block(src, e, prefix):
    """The mirror of `_cross_block`, in the torch module's own key order."""
    _emit_layernorm(src.ln_q, e, prefix + ".ln_q")
    _emit_layernorm(src.ln_kv, e, prefix + ".ln_kv")
    _emit_attention(src.attn, e, prefix + ".attn")
    _emit_layernorm(src.ln_m, e, prefix + ".ln_m")
    _emit_linear(src.mlp[0], e, prefix + ".mlp.0")
    _emit_linear(src.mlp[1], e, prefix + ".mlp.2")


def load_cone(state_dict, model):
    """`ConeMAE.state_dict()` → a `ConeMAEJax`, or a refusal naming the keys.

    Consumes all 99 keys; `finish()` raises a `KeyError` naming BOTH the keys
    the model wanted and the state_dict did not carry AND the keys the
    state_dict carried that nothing wanted. Returns the model, loaded.
    """
    c = _Consumer(dict(state_dict), "load_cone")
    # --- bare parameters, in the torch module's registration order ---------
    _param(model.mask_tok, c, "mask_tok")
    _param(model.miss_tok, c, "miss_tok")
    _param(model.cls_tok, c, "cls_tok")
    _param(model.query_tok, c, "query_tok")
    _param(model.latents, c, "latents")
    _param(model.pool_q, c, "pool_q")
    # --- tokens ------------------------------------------------------------
    _linear(model.val_proj, c, "val_proj")
    _linear(model.dot_proj, c, "dot_proj")
    _embed(model.chan_emb, c, "chan_emb.weight")
    # `coord.freqs` is a non-persistent buffer and is deliberately NOT asked
    # for (see the module docstring); `coord.proj` is the only parameter.
    _linear(model.coord.proj, c, "coord.proj")
    _linear(model.ctx_proj, c, "ctx_proj")
    # --- encoder -----------------------------------------------------------
    _cross_block(model.cross, c, "cross")
    _encoder(model.encoder, c, "encoder")
    _packed_attention(model.pool, c, "pool")
    _layernorm(model.ln_pool, c, "ln_pool")
    _linear(model.to_z, c, "to_z")
    # --- decoder -----------------------------------------------------------
    _linear(model.q_proj, c, "q_proj")
    _linear(model.z_proj, c, "z_proj")
    _linear(model.lat_proj, c, "lat_proj")
    for i, blk in enumerate(model.dec):
        _cross_block(blk, c, f"dec.{i}")
    _layernorm(model.ln_out, c, "ln_out")
    _linear(model.head, c, "head")
    c.finish()
    return model


def export_cone(model):
    """`ConeMAEJax` → a dict of numpy arrays `ConeMAE` loads with strict=True.

    The reverse of `load_cone`, one list read backwards, emitted in the torch
    module's own construction order — `_Emitter` refuses to write a key twice,
    so a copy-paste that duplicated a prefix is a refusal rather than a silent
    overwrite. Every value is float32 (`convert._np`): a bf16 training run
    would otherwise hand the torch eval ladder bf16 weights, and every
    published number was produced against float32 ones.
    """
    e = _Emitter("export_cone")
    e.put("mask_tok", model.mask_tok.value)
    e.put("miss_tok", model.miss_tok.value)
    e.put("cls_tok", model.cls_tok.value)
    e.put("query_tok", model.query_tok.value)
    e.put("latents", model.latents.value)
    e.put("pool_q", model.pool_q.value)
    _emit_linear(model.val_proj, e, "val_proj")
    _emit_linear(model.dot_proj, e, "dot_proj")
    e.put("chan_emb.weight", model.chan_emb.embedding.value)
    _emit_linear(model.coord.proj, e, "coord.proj")
    _emit_linear(model.ctx_proj, e, "ctx_proj")
    _emit_cross_block(model.cross, e, "cross")
    _emit_encoder(model.encoder, e, "encoder")
    _emit_attention(model.pool, e, "pool")
    _emit_layernorm(model.ln_pool, e, "ln_pool")
    _emit_linear(model.to_z, e, "to_z")
    _emit_linear(model.q_proj, e, "q_proj")
    _emit_linear(model.z_proj, e, "z_proj")
    _emit_linear(model.lat_proj, e, "lat_proj")
    for i, blk in enumerate(model.dec):
        _emit_cross_block(blk, e, f"dec.{i}")
    _emit_layernorm(model.ln_out, e, "ln_out")
    _emit_linear(model.head, e, "head")
    return e.sd


def export_cone_pt(model, args, path=None, **extra):
    """The `.pt` blob `ml/train_cone.py:train_one.save` writes, from JAX.

    `{"args": dict(args, backend="jax"), "model": <state_dict>,
      "chan_names", "norm", "step", "arm", "L_in", "params"}` — the last five
    arrive through `**extra`, exactly the way `train_cone.py` supplies them,
    because they are RUN facts (which channels, which anomaly space, which
    step, which arm) and not model facts. Nothing here invents one: a blob
    missing `chan_names` is a blob the velocity probe should refuse, not one
    this function should guess at.

    **NO optimiser state**, deliberately: mapping optax's state into torch's
    AdamW is explicitly out of scope (`ml/plans/E069_HANDOVER.md` §8.1), and a
    blob carrying torch-shaped moments this trainer never produced would be a
    continuation wearing a warm restart's clothes. The resumable state is the
    sibling `ckpt_latest.npz`.

    Returns the blob; with `path`, also `torch.save`s it.
    """
    import torch                                   # local: JAX users need not
    if not isinstance(args, dict):
        args = dict(vars(args))
    else:
        args = dict(args)
    args["backend"] = "jax"
    sd = {k: torch.from_numpy(v) for k, v in export_cone(model).items()}
    blob = {"args": args, "model": sd}
    blob.update(extra)
    if path is not None:
        torch.save(blob, path)
    return blob


def cone_from_torch(torch_model, *, rngs=None):
    """A `ConeMAEJax` carrying `torch_model`'s weights.

    §8.4's initialisation path: `nn.Embedding` is N(0, 1) and `nnx.Embed` is
    std ~ 1/sqrt(d), and `ml/jaxport/README.md:105-118` measured what that
    difference costs downstream — so a JAX cone codec is never initialised by
    Flax. It is initialised by building `ConeMAE` under
    `torch.manual_seed(seed)` and converting, and the seed in the run's
    `config` record is the TORCH seed.

    The geometry is read off the torch module's own attributes rather than
    from a caller-supplied dict, so the two models cannot disagree about a
    width that `load_cone` would then refuse in a less legible way.
    """
    from flax import nnx
    jm = ConeMAEJax(torch_model.n_chan,
                    d_model=torch_model.d_model,
                    n_heads=torch_model.n_heads,
                    n_latents=torch_model.n_latents,
                    n_layers=torch_model.n_layers,
                    d_z=torch_model.d_z,
                    d_dec=torch_model.d_dec,
                    dec_layers=torch_model.dec_layers,
                    n_fourier=torch_model.n_fourier,
                    rngs=rngs if rngs is not None else nnx.Rngs(0))
    return load_cone(torch_model.state_dict(), jm)


def cone_from_ckpt_jax(blob, *, rngs=None):
    """`(ConeMAEJax, args)` from a `train_cone.py`-shaped checkpoint blob.

    Mirrors how `ml/train_cone.py` rebuilds a `ConeMAE`: the geometry lives in
    `blob["args"]` (the trainer saves `vars(a)`), the channel count is
    `len(blob["chan_names"])` — the names ARE the family map, so a count taken
    from anywhere else could build a codec whose channels mean other channels'
    cones — and every `.get()` default is the torch constructor's own default,
    so a checkpoint written before a knob existed rebuilds as what it was.
    """
    from flax import nnx
    a = blob.get("args", {}) or {}
    names = blob.get("chan_names")
    if not names:
        raise SystemExit(
            "cone_from_ckpt_jax: this blob carries no `chan_names`, so the "
            "channel count — and with it every channel's cone family — is "
            "unknown. Refusing rather than guessing a geometry: a ConeMAE "
            "built at the wrong C loads no state_dict at all, and one built "
            "at the right C from the wrong names is a codec whose channels "
            "mean other channels (ml/cone_sampler.py's ConeSampler raises on "
            "the same mismatch).")
    model = ConeMAEJax(len(names),
                       d_model=int(a.get("d_model", 256)),
                       n_heads=int(a.get("n_heads", 8)),
                       n_latents=int(a.get("n_latents", 64)),
                       n_layers=int(a.get("n_layers", 6)),
                       d_z=int(a.get("d_z", 32)),
                       d_dec=int(a.get("d_dec", 256)),
                       dec_layers=int(a.get("dec_layers", 2)),
                       n_fourier=int(a.get("n_fourier", 8)),
                       rngs=rngs if rngs is not None else nnx.Rngs(0))
    return load_cone(blob["model"], model), a

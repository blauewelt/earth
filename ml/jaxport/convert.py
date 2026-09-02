"""torch ↔ JAX weight conversion for the tier-1 models.

**Both directions, since tier 3.** Tiers 1–2 only ever needed to READ the
published artefacts (`ml/plans/JAX_PORT.md` §6 deferred the reverse). Tier 3
trains on a TPU, and §1b names the cheap validation of a TPU-trained codec
explicitly: *convert it back and score it through the UNCHANGED torch eval
ladder*. That is what `export_pt` is for, and it is the only reason it exists —
nothing in the operational tree is asked to change, and no torch script learns
about JAX. The artefact a JAX run hands over is an ordinary `.pt` blob with the
ordinary `model`/`args`/`d_z`/`chan`/`norm` keys, which `codec_from_ckpt` and
all twelve eval scripts read exactly as they read a GPU-trained one.

The round trip is a GATE, not an aspiration: `tests/test_jaxport_train.py`
(G4d) converts a torch state_dict → NNX → back and requires the two
state_dicts to be **identical**, key set and values alike.

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
        # `.copy()` is load-bearing (found 2026-09-02 by the cone port's gate
        # C3, which failed at 1.7e-2 against 1e-6 on its first run): a torch
        # tensor's `.numpy()` is a VIEW of torch storage, and `jnp.asarray` of
        # a contiguous numpy array is zero-copy on the CPU backend, so every
        # parameter the mapper did not transpose or slice first (68 of the
        # cone codec's 99) was a JAX array sharing bytes with the torch module.
        # Train the torch module after converting and the JAX weights move
        # with it — which makes a cross-framework one-step gate agree
        # PERFECTLY for the wrong reason. Values are unchanged by the copy;
        # only the ownership is.
        return np.array(v.detach().cpu().numpy() if hasattr(v, "detach")
                        else v, copy=True)

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
    norm1,norm2}`. No final norm: torch's `norm` is None (see models.py).

    E-057: a FiLM layer (`ml/temporal.py:_CondLayer`) ADDS `layers.N.film`
    and changes nothing else, so the key set of a deterministic checkpoint is
    a strict subset of an FGN one. The film weights are asked for on exactly
    the condition the module was built with, and the refusal contract then
    works in both directions for free — a deterministic model handed an FGN
    checkpoint refuses on the unconsumed `film.weight`, and an FGN model
    handed a deterministic one refuses on the missing one. Which is right:
    warm-starting a trunk is a DELIBERATE `strict=False` act on the torch
    side, never something a converter should do silently.
    """
    for i, lyr in enumerate(dst.layers):
        # prefix="" addresses a BARE nn.TransformerEncoder's own state_dict
        # (`layers.0....`), which is what a single-layer parity check loads.
        p = f"{prefix}.layers.{i}" if prefix else f"layers.{i}"
        _packed_attention(lyr.self_attn, c, p + ".self_attn")
        _linear(lyr.linear1, c, p + ".linear1")
        _linear(lyr.linear2, c, p + ".linear2")
        _layernorm(lyr.norm1, c, p + ".norm1")
        _layernorm(lyr.norm2, c, p + ".norm2")
        if getattr(lyr, "film", None) is not None:
            _linear(lyr.film, c, p + ".film")


# --------------------------------------------------------------------------
# per-model loaders
# --------------------------------------------------------------------------
def load_pixelmae(state_dict, model):
    c = _Consumer(dict(state_dict), "load_pixelmae")
    _linear(model.val_proj, c, "val_proj")
    _embed(model.chan_emb, c, "chan_emb.weight")
    # E-047: `time_emb` and `q_time` exist in the torch module ONLY when
    # k_time > 1, so they are asked for on exactly the same condition. The
    # refusal contract then does the rest in both directions: a k_time=1 model
    # handed a block checkpoint refuses on the unconsumed `time_emb.weight`,
    # and a block model handed a per-bin checkpoint refuses on the missing one.
    if model.k_time > 1:
        _embed(model.time_emb, c, "time_emb.weight")
    _param(model.mask_tok, c, "mask_tok")
    _param(model.miss_tok, c, "miss_tok")
    _param(model.cls_tok, c, "cls_tok")
    _linear(model.ctx_proj, c, "ctx_proj")
    _encoder(model.encoder, c, "encoder")
    _linear(model.to_z, c, "to_z")
    _embed(model.q_chan, c, "q_chan.weight")
    _embed(model.q_off, c, "q_off.weight")
    if model.k_time > 1:
        _embed(model.q_time, c, "q_time.weight")
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
    # E-057: nn.Sequential(Linear, SiLU, Linear) — the LINEARS are at indices
    # 0 and 2, the SiLU (parameterless) at 1, exactly as PixelMAE's decoder
    # puts its Linears at even indices. Asked for only when the model has an
    # ε path, so the refusal contract catches the mismatch either way.
    if getattr(model, "eps_dim", 0):
        _linear(model.eps_embed[0], c, "eps_embed.0")
        _linear(model.eps_embed[1], c, "eps_embed.2")
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


# --------------------------------------------------------------------------
# the REVERSE direction: NNX → a torch state_dict the unchanged torch model
# loads. See the module docstring for why it exists.
# --------------------------------------------------------------------------
def _np(x):
    """A JAX array as a contiguous float32 numpy array.

    float32 EXPLICITLY: a checkpoint written from a bf16 training run would
    otherwise hand the torch eval ladder bf16 weights, and every published
    number was produced against float32 ones. Widening is exact; it is the
    silent narrowing that would move numbers.
    """
    # A COPY, not a view: `np.asarray` of a JAX array borrows read-only
    # memory, and `torch.from_numpy` on that warns and hands back a tensor
    # whose writes are undefined behaviour. A checkpoint is exactly the object
    # nobody wants to discover that on.
    return np.array(np.asarray(x), dtype=np.float32, order="C", copy=True)


class _Emitter:
    """The mirror of `_Consumer`: it records every key it WRITES so the export
    can be checked against the torch module's own key set, rather than trusted.
    """

    def __init__(self, what):
        self.what = what
        self.sd = {}

    def put(self, key, value):
        if key in self.sd:
            raise KeyError(f"{self.what}: {key!r} emitted twice")
        self.sd[key] = _np(value)


def _emit_linear(src, e, prefix):
    """Flax kernel [in, out] → torch Linear.weight [out, in]."""
    e.put(prefix + ".weight", src.kernel.value.T)
    e.put(prefix + ".bias", src.bias.value)


def _emit_layernorm(src, e, prefix):
    e.put(prefix + ".weight", src.scale.value)
    e.put(prefix + ".bias", src.bias.value)


def _emit_attention(src, e, prefix):
    """q, k, v → one packed `in_proj_weight` [3d, d], IN THAT ORDER."""
    e.put(prefix + ".in_proj_weight",
          np.concatenate([_np(lin.kernel.value.T) for lin in
                          (src.q_proj, src.k_proj, src.v_proj)], axis=0))
    e.put(prefix + ".in_proj_bias",
          np.concatenate([_np(lin.bias.value) for lin in
                          (src.q_proj, src.k_proj, src.v_proj)], axis=0))
    _emit_linear(src.out_proj, e, prefix + ".out_proj")


def _emit_encoder(src, e, prefix):
    for i, lyr in enumerate(src.layers):
        p = f"{prefix}.layers.{i}" if prefix else f"layers.{i}"
        _emit_attention(lyr.self_attn, e, p + ".self_attn")
        _emit_linear(lyr.linear1, e, p + ".linear1")
        _emit_linear(lyr.linear2, e, p + ".linear2")
        _emit_layernorm(lyr.norm1, e, p + ".norm1")
        _emit_layernorm(lyr.norm2, e, p + ".norm2")
        # E-057: emitted at the END of the layer, which is where
        # `_CondLayer.__init__` registers it, so the two state_dicts have the
        # same key ORDER as well as the same key SET.
        if getattr(lyr, "film", None) is not None:
            _emit_linear(lyr.film, e, p + ".film")


def export_pixelmae(model):
    """NNX PixelMAE → an ORDERED dict of numpy arrays in torch's key order.

    Key order matters only cosmetically (torch's `load_state_dict` is keyed,
    not positional), but emitting in the module's own construction order makes
    a diff of two state_dicts readable, and the ordering is the same one
    `load_pixelmae` consumes in — one list, read forwards and backwards.
    """
    e = _Emitter("export_pixelmae")
    _emit_linear(model.val_proj, e, "val_proj")
    e.put("chan_emb.weight", model.chan_emb.embedding.value)
    if model.k_time > 1:
        e.put("time_emb.weight", model.time_emb.embedding.value)
    e.put("mask_tok", model.mask_tok.value)
    e.put("miss_tok", model.miss_tok.value)
    e.put("cls_tok", model.cls_tok.value)
    _emit_linear(model.ctx_proj, e, "ctx_proj")
    _emit_encoder(model.encoder, e, "encoder")
    _emit_linear(model.to_z, e, "to_z")
    e.put("q_chan.weight", model.q_chan.embedding.value)
    e.put("q_off.weight", model.q_off.embedding.value)
    if model.k_time > 1:
        e.put("q_time.weight", model.q_time.embedding.value)
    for i, lin in enumerate(model.decoder):
        _emit_linear(lin, e, f"decoder.{2 * i}")
    return e.sd


def export_temporal(model):
    """NNX `TemporalTransformer` → an ORDERED dict of numpy arrays in the torch
    module's own key order — the reverse of `load_temporal`, one list read
    backwards.

    The key set is what `ml/rollout_spatial.py` and every other torch reader
    calls `load_state_dict(strict=True)` with, so a missing or extra key here
    is a refusal there rather than a silently different model. `pos.weight` is
    the one key those readers also SHAPE-read (`tk["model"]["pos.weight"]
    .shape[0]` is how a roll recovers k_max from the file rather than from a
    convention), which is why the positional table is emitted whole and never
    sliced to the K a run happened to train at.
    """
    e = _Emitter("export_temporal")
    _emit_linear(model.inp, e, "inp")
    _emit_linear(model.static, e, "static")
    e.put("pos.weight", model.pos.embedding.value)
    _emit_encoder(model.encoder, e, "encoder")
    if getattr(model, "eps_dim", 0):
        _emit_linear(model.eps_embed[0], e, "eps_embed.0")
        _emit_linear(model.eps_embed[1], e, "eps_embed.2")
    _emit_linear(model.head, e, "head")
    if model.heads_direct is not None:
        for k, lin in model.heads_direct.items():
            _emit_linear(lin, e, f"heads_direct.{k}")
    return e.sd


def export_temporal_pt(model, args, path=None, **extra):
    """A stage-2 head `.pt` the UNCHANGED torch eval scripts load.

    Shape matches what `ml/temporal.py` writes: `{model, args, step, ...}`.
    `args` must be the trainer's own `vars(a)`, because that is the field
    `ml/rollout_spatial.py` reads the head's geometry back out of — `K`,
    `d_model`, `layers`, `stencil`, `ring_km`, `seed`, `direct`,
    `season_phase`, and (absent here, deliberately) `input_quant`.

    **NO `opt`/`sched`.** `ml/temporal.py --resume-temporal` refuses a
    checkpoint missing them, and that refusal is CORRECT for this artefact:
    optax's state is not torch Adam's state, mapping one into the other is
    explicitly out of scope (`JAX_PORT.md` §3.3), and a blob carrying
    torch-shaped moments this trainer never produced would be a continuation
    wearing a warm restart's clothes — the exact confusion `--init-temporal`
    exists to keep separate. The resumable state is the sibling `.npz`.
    """
    import torch                                   # local: JAX users need not
    if not isinstance(args, dict):
        args = dict(vars(args))
    else:
        args = dict(args)
    args.setdefault("backend", "jax")
    sd = {k: torch.from_numpy(v) for k, v in export_temporal(model).items()}
    blob = {"model": sd, "args": args}
    blob.update(extra)
    if path is not None:
        torch.save(blob, path)
    return blob


def export_pt(model, args, path=None, **extra):
    """A `.pt` blob the UNCHANGED torch stack loads: `{model, args, d_z, ...}`.

    `args` is the run's own argument namespace (a dict, or anything `vars()`
    accepts) — the same `vars(a)` that `ml/train.py:save_ckpt` writes, because
    `codec_from_ckpt` and the twelve eval scripts read the architecture out of
    exactly that field. Extra top-level keys (`chan`, `norm`, `step`, `tag`)
    are passed through as-is so the artefact is indistinguishable in SHAPE
    from a GPU-trained one; it is distinguishable in PROVENANCE, which is the
    point — `args["backend"]` says `jax` and `ml/CLAUDE.md` §3b makes a
    TPU-trained number a new tier that buys its own replication.

    Returns the blob. With `path`, also `torch.save`s it (torch is imported
    HERE, function-locally, so importing this module still needs no torch).
    """
    import torch                                   # local: JAX users need not
    if not isinstance(args, dict):
        args = dict(vars(args))
    else:
        args = dict(args)
    args.setdefault("backend", "jax")
    sd = {k: torch.from_numpy(v) for k, v in export_pixelmae(model).items()}
    blob = {"model": sd, "args": args, "d_z": int(model.d_z)}
    blob.update(extra)
    if path is not None:
        torch.save(blob, path)
    return blob


def codec_from_ckpt_jax(ck, n_chan):
    """`ml/model.py:codec_from_ckpt`, for the NNX PixelMAE.

    Same contract, same `.get()` defaults: the checkpoint's `args` carry the
    full architecture (train.py saves `vars(a)`), and old checkpoints predate
    the size knobs, so every default is the pilot architecture. `d_z` is a
    TOP-LEVEL key of the blob, not one of `args`.
    """
    a = ck.get("args", {})
    # E-046/E-048: THE BOTTLENECK IS PART OF THE ARCHITECTURE and it carries
    # no parameters, so a loader that ignored it would build a model whose
    # `load_pixelmae` succeeds, whose leaf count matches to the byte, and
    # whose every z is a different function of the input. This used to REFUSE
    # any quantized checkpoint, because the NNX PixelMAE had no bottleneck;
    # E-048 gives it the same one (levels AND ladders, off the same
    # `ml/fsq_ladder.py` definitions), so the refusal narrows to what this
    # revision genuinely cannot rebuild — the same unknown-`fsq_*` contract
    # ml/model.py:codec_from_ckpt states, in the same words, because a
    # checkpoint written by a later revision must not load here as something
    # simpler and be scored as one.
    fsq = str(a.get("fsq_levels", "") or "")
    fsq_ladder = str(a.get("fsq_ladder", "uniform") or "uniform")
    fsq_fit = str(a.get("fsq_ladder_fit", "") or "")
    # E-049's `fsq_bound` is KNOWN here and REFUSED BY VALUE, which is not the
    # same thing as being unknown. `ml/train.py` saves `vars(a)`, so every
    # torch checkpoint written from that commit on carries the key — with the
    # inert "" on the overwhelming majority of them. Treating its mere
    # PRESENCE as unknown would refuse every new continuous and every new
    # unbounded-FSQ checkpoint too, i.e. it would take the TPU path out of
    # service for codecs this port rebuilds exactly. So the key is admitted
    # and the VALUE is refused, one branch down: `ml/jaxport/models.py`'s
    # PixelMAE has no intrinsic bound, and a bounded checkpoint rebuilt here
    # would load every leaf, match every shape, and compute a different z from
    # the same weights.
    # E-050's `fsq_warmstart` / `fsq_warmstart_from` are INFORMATIONAL here for
    # the same reason they are in ml/model.py:codec_from_ckpt — a permission
    # granted to the torch resume guard, and the provenance string naming the
    # continuous codec a lattice grew out of. Neither changes what a `z` IS, so
    # neither may cost a checkpoint its loader.
    known = {"fsq_levels", "fsq_ladder", "fsq_exp_base", "fsq_ladder_fit",
             "fsq_auto_n", "fsq_auto_step", "fsq_bound",
             "fsq_warmstart", "fsq_warmstart_from"}
    fsq_bound = str(a.get("fsq_bound", "") or "")
    if fsq_bound:
        raise SystemExit(
            f"codec_from_ckpt_jax: this checkpoint was trained with "
            f"--fsq-bound {fsq_bound!r} — E-049's intrinsic bound (LayerNorm "
            f"without affine on the pre-quantization activation), which "
            f"ml/jaxport/models.py does not implement. Rebuilding it here "
            f"would produce a codec whose every z is a different function of "
            f"the same weights, with nothing in the output saying so. This "
            f"port refuses rather than running a reduced version of itself: "
            f"score this checkpoint with the torch loader "
            f"(ml/model.py:codec_from_ckpt), or implement the bound and gate "
            f"it with the torch/JAX parity check in "
            f"tests/test_e048_fsq_ladders.py.")
    unknown = sorted(k for k in a if str(k).startswith("fsq_")
                     and k not in known)
    if unknown:
        raise SystemExit(
            f"codec_from_ckpt_jax: this checkpoint carries FSQ argument(s) "
            f"{unknown} that this revision of ml/jaxport/models.py does not "
            f"implement. Refusing rather than rebuilding a codec whose "
            f"bottleneck is not the one that was trained.")
    if fsq and fsq_ladder == "auto" and not fsq_fit:
        raise SystemExit(
            "codec_from_ckpt_jax: this checkpoint says --fsq-ladder auto but "
            "carries no `fsq_ladder_fit`, so the per-dimension ladder it "
            "trained with is not recorded. Refusing rather than guessing a "
            "lattice.")
    model = PixelMAE(n_chan=n_chan, d_z=ck["d_z"],
                     k_time=int(a.get("k_time", 1) or 1),
                     patch=a.get("patch", 1),
                     d_model=a.get("d_model", 128),
                     n_layers=a.get("n_layers", 4),
                     n_heads=a.get("n_heads", 4),
                     d_dec=a.get("d_dec", 256),
                     dec_layers=a.get("dec_layers", 2),
                     fsq_levels=fsq, fsq_ladder=fsq_ladder,
                     fsq_exp_base=float(a.get("fsq_exp_base", 2.0) or 2.0),
                     fsq_ladder_fit=fsq_fit)
    return load_pixelmae(ck["model"], model)

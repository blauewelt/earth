"""E-069 · `ml/cone_codec.py:ConeMAE` in Flax NNX, forward-parity-exact.

The cone codec's JAX twin (`ml/plans/E069_HANDOVER.md` §8.2/§8.3). It is a
MIRROR, not a redesign: every method here has the same name and the same
semantics as the torch method it shadows, in EVAL semantics (there is no
dropout in `ConeMAE` at all — every `nn.MultiheadAttention` and the
`nn.TransformerEncoderLayer` are constructed with `dropout=0.0` — so "eval
semantics" and "train semantics" are the same arithmetic here, and the port
needs no `deterministic` flag).

Read `ml/jaxport/models.py`'s docstring first: every trap it lists is live in
this file too (erf gelu, relu inside the encoder layers, no final norm on the
stack, [out,in] → [in,out] linears, packed QKV, LayerNorm eps 1e-5). What is
NEW here, and what the torch source settles rather than this file:

  * **`coord.freqs` is a NON-PERSISTENT BUFFER on the torch side**
    (`register_buffer(..., persistent=False)`), so it is absent from the
    state_dict and the converter must not look for it. It is recomputed here
    from `n_fourier` in the constructor, `2.0 ** arange(n_fourier)`, which is
    the same closed form the torch module uses — a converter that carried it
    would be carrying a constant, and one that MISSED it would silently
    change the coordinate basis of every dot. `tests/test_jaxport_cone.py`
    checks the two arrays are identical rather than assuming it.
    It is stored through `_nnx_data` so it is DATA in both flax regimes: flax
    ≤ 0.10 auto-registers a raw array attribute, flax ≥ 0.11 would otherwise
    make it a static (and unhashable) graphdef entry. It is not an
    `nnx.Param`, so `nnx.split(model, nnx.Param, ...)` leaves it in the rest —
    a buffer, exactly as torch has it.

  * **The key-padding mask is a bool `[B, 1, 1, Lk]`** into
    `models.MultiHeadAttention`, whose bool convention is already torch's
    (True = masked out). NO NaN guard and NO logit clip: `tokens` always emits
    `2 + C` unmasked tokens per row (cls, ctx and the C lag-0 patch tokens are
    never padding), so no attention row can be fully masked. That is asserted
    in the test rather than defended in the code — a guard here would hide the
    day the invariant stops holding, which is the one day anybody needs to
    know about.

  * **The three `where` additions keep their ORDER**, and so do the terms of
    `nll_gauss`. Float addition is not associative; at the 1e-5/1e-6 the gates
    are pre-registered at, the summation order IS part of the parity claim
    (`ml/plans/E069_HANDOVER.md` §8.3).

  * **`signed_log` is `sign(v) * log1p(|v|)`**, not `log1p`. A FUTURE query
    sits at a NEGATIVE lag (t+1 is `lag_days = -5`), and `log1p(-5)` is NaN.
    One function, no branch, and the encoder side is unchanged by
    construction.

  * **THE COORDINATE ENCODING IS THE LOOSEST PART OF THE PARITY, AND IT IS
    THE TORCH MODEL'S OWN CONDITIONING, NOT A PORT BUG.** Measured on the
    smoke geometry (`tests/test_jaxport_cone.py`): `tokens` agrees to
    ~1.3e-5 while `z` agrees to ~1.6e-7, and every bit of that 1.3e-5 is in
    the dot tokens' `coord` term. The chain is: torch's `log1p` and XLA's
    `log1p` differ in the LAST fp32 bit on some inputs (~5e-7 at
    `signed_log(907 km) = 6.81`); the highest Fourier band multiplies that by
    `2 ** (n_fourier - 1) = 128`, putting the sin/cos argument at ~870 radians
    with a ~1e-4 rad disagreement; `sin` and `cos` there are locally
    1-Lipschitz, so the FEATURE moves by ~6e-5, and `coord.proj` averages 64
    of those down to ~1.3e-5. `sin`/`cos` themselves agree BIT FOR BIT at
    those arguments — both libraries range-reduce correctly — so the encoding
    simply has a condition number of 128 with respect to its own input, and a
    one-ULP perturbation of `dy_km` would move the TORCH model by the same
    amount. That is why §8.7 pre-registers C1 at 1e-4 and not at 1e-6, and
    why nothing here rounds, clamps or reorders the encoding to make the
    number smaller: it would be hiding a property of the model.

  * **`logvar` is clipped to [-8, 8]** at the head and `wsum` is clamped at
    1e-6 — both are part of the loss's DEFINITION on the torch side
    (`LOGVAR_MIN/MAX`, `w.sum().clamp(min=1e-6)`), not numerical hygiene, so
    they are mirrored exactly and not "improved".

WHAT IS NOT HERE, and why. There is no `forward`/`_masks` twin: the masks are
drawn by a numpy `Generator` on the host (`ml/plans/E069_HANDOVER.md` §8.6)
and handed to `loss_given`, because two RNGs cannot be made to agree and the
gate passes the DRAW, not the seed. `loss_given` mirrors torch's
`forward_given`, and `query_sets_given` mirrors `_query_sets` with the
hidden-dot selection already made.
"""
import math

import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx

from .models import (MultiHeadAttention, TransformerEncoder, gelu_exact,
                     _nnx_data)

# `ml/cone_codec.py:LOGVAR_MIN/MAX`, restated rather than imported: importing
# them would make this module need torch, which is the one thing a TPU node
# must not need at import time.
LOGVAR_MIN, LOGVAR_MAX = -8.0, 8.0
LOG_2PI = math.log(2.0 * math.pi)


def signed_log(v):
    """`ml/cone_codec.py:signed_log` — `sign(v) * log1p(|v|)`."""
    return jnp.sign(v) * jnp.log1p(jnp.abs(v))


def nll_gauss(mu, logvar, target):
    """`ml/cone_codec.py:nll_gauss`, term for term and in that order."""
    return 0.5 * (LOG_2PI + logvar
                  + (target - mu) ** 2 * jnp.exp(-logvar))


def plan_to_jax(plan):
    """A torch `default_plan()` dict as the plain dict this module reads.

    The array-valued keys (`chan_drop_p`, `chan_w`, `future_lags`) become
    jnp arrays; the python scalars pass through; `generator` is DROPPED, and
    dropped rather than converted, because there is nothing on this side for a
    torch.Generator to mean — the masks are drawn on the host and handed in.
    """
    out = {}
    for k, v in plan.items():
        if k == "generator":
            continue
        if hasattr(v, "detach"):                     # a torch tensor
            out[k] = jnp.asarray(v.detach().cpu().numpy())
        elif isinstance(v, np.ndarray):
            out[k] = jnp.asarray(v)
        else:
            out[k] = v
    return out


class CoordEncJax(nnx.Module):
    """`ml/cone_codec.py:CoordEnc` — Fourier features of four scalars.

    `signed_log` of each of (dy_km, dx_km, lag_days, depth), then sin/cos at
    `n_fourier` base-2 frequencies, then ONE Linear over the 8*n_fourier
    features. `feat.flatten(-2)` on the torch side lays the features out
    scalar-major — [sin(f0..fF-1), cos(f0..fF-1)] for each of the four
    scalars in turn — and `reshape(..., 8F)` here is the same layout, which
    is what makes one shared `proj` weight mean the same thing in both.
    """

    def __init__(self, d_model, n_fourier=8, *, rngs):
        self.n_fourier = int(n_fourier)
        # The BUFFER (see the module docstring): a constant basis, absent from
        # the torch state_dict, recomputed from the same closed form.
        self.freqs = _nnx_data(
            2.0 ** jnp.arange(self.n_fourier, dtype=jnp.float32))
        self.proj = nnx.Linear(8 * self.n_fourier, d_model, rngs=rngs)

    def __call__(self, dy_km, dx_km, lag_days, depth):
        s = jnp.stack([signed_log(dy_km), signed_log(dx_km),
                       signed_log(lag_days), signed_log(depth)], axis=-1)
        ang = s[..., None] * self.freqs                          # [..., 4, F]
        feat = jnp.concatenate([jnp.sin(ang), jnp.cos(ang)], axis=-1)
        flat = feat.reshape(feat.shape[:-2] + (8 * self.n_fourier,))
        return self.proj(flat)


class CrossBlockJax(nnx.Module):
    """`ml/cone_codec.py:CrossBlock` — pre-LN cross-attention + MLP.

        k = ln_kv(kv); q = q + attn(ln_q(q), k, k, kpm); q = q + mlp(ln_m(q))

    The MLP is torch's `nn.Sequential(Linear, GELU, Linear)`, so its two
    linears carry the state_dict keys `mlp.0` and `mlp.2` (the GELU occupies
    index 1 and has no parameters). They are held in a two-element list here
    for exactly that reason — the same convention `models.PixelMAE.decoder`
    uses for its even-indexed linears — so the converter's mapping is a list
    index rather than a name that has to be remembered.

    GELU is `gelu_exact` (erf): torch's `nn.GELU()` is the exact form and
    `jax.nn.gelu` defaults to the tanh approximation, which is ~1e-3 off.
    """

    def __init__(self, d_model, n_heads, mlp_mult=4, *, rngs):
        self.ln_q = nnx.LayerNorm(d_model, epsilon=1e-5, rngs=rngs)
        self.ln_kv = nnx.LayerNorm(d_model, epsilon=1e-5, rngs=rngs)
        self.attn = MultiHeadAttention(d_model, n_heads, rngs=rngs)
        self.ln_m = nnx.LayerNorm(d_model, epsilon=1e-5, rngs=rngs)
        self.mlp = _nnx_data([nnx.Linear(d_model, mlp_mult * d_model,
                                         rngs=rngs),
                              nnx.Linear(mlp_mult * d_model, d_model,
                                         rngs=rngs)])

    def __call__(self, q, kv, key_padding_mask=None):
        k = self.ln_kv(kv)
        # torch's `key_padding_mask` is [B, Lk] with True = ignore;
        # models.MultiHeadAttention takes [B, 1, 1, Lk] with the same
        # convention, so this is a reshape and not a re-interpretation.
        m = None if key_padding_mask is None else key_padding_mask[:, None,
                                                                   None, :]
        h = self.attn(self.ln_q(q), k, k, mask=m)
        q = q + h
        return q + self.mlp[1](gelu_exact(self.mlp[0](self.ln_m(q))))


class ConeMAEJax(nnx.Module):
    """Mirror of `ml/cone_codec.py:ConeMAE`. Same constructor signature, so
    `cone_from_ckpt_jax` can build it from a checkpoint's `args` the way
    `train_cone.py` builds the torch one."""

    def __init__(self, n_chan, d_model=256, n_heads=8, n_latents=64,
                 n_layers=6, d_z=32, d_dec=256, dec_layers=2, n_fourier=8,
                 *, rngs=None):
        rngs = rngs if rngs is not None else nnx.Rngs(0)
        self.n_chan = int(n_chan)
        self.d_model = int(d_model)
        self.d_z = int(d_z)
        self.d_dec = int(d_dec)
        self.n_latents = int(n_latents)
        self.n_layers = int(n_layers)
        self.dec_layers = int(dec_layers)
        self.n_heads = int(n_heads)
        self.n_fourier = int(n_fourier)

        # --- tokens ---------------------------------------------------------
        self.val_proj = nnx.Linear(18, d_model, rngs=rngs)
        self.dot_proj = nnx.Linear(2, d_model, rngs=rngs)
        self.chan_emb = nnx.Embed(n_chan, d_model, rngs=rngs)
        self.coord = CoordEncJax(d_model, n_fourier, rngs=rngs)
        # Bare parameters, not modules — the torch keys are `mask_tok` etc.
        # Zeros here and NOT `nn.init.normal_(std=0.02)`: §8.4 initialises the
        # JAX codec by CONVERTING a torch module built under
        # `torch.manual_seed(seed)`, so an init that merely resembled torch's
        # would be a third distribution nobody measured. A model built here
        # and never loaded is deliberately degenerate rather than plausible.
        self.mask_tok = nnx.Param(jnp.zeros((d_model,)))
        self.miss_tok = nnx.Param(jnp.zeros((d_model,)))
        self.cls_tok = nnx.Param(jnp.zeros((d_model,)))
        self.query_tok = nnx.Param(jnp.zeros((d_model,)))
        self.ctx_proj = nnx.Linear(4, d_model, rngs=rngs)

        # --- encoder --------------------------------------------------------
        self.latents = nnx.Param(jnp.zeros((n_latents, d_model)))
        self.cross = CrossBlockJax(d_model, n_heads, mlp_mult=4, rngs=rngs)
        # RELU and NO final norm — `cone_codec.py` passes no `activation=` to
        # `nn.TransformerEncoderLayer` and no `norm=` to `nn.TransformerEncoder`
        # (models.py's docstring, traps 2 and 3).
        self.encoder = TransformerEncoder(d_model, n_heads, n_layers,
                                          4 * d_model, rngs=rngs)
        self.pool_q = nnx.Param(jnp.zeros((1, 1, d_model)))
        self.pool = MultiHeadAttention(d_model, n_heads, rngs=rngs)
        self.ln_pool = nnx.LayerNorm(d_model, epsilon=1e-5, rngs=rngs)
        self.to_z = nnx.Linear(d_model, d_z, rngs=rngs)

        # --- decoder --------------------------------------------------------
        self.q_proj = nnx.Linear(d_model, d_dec, rngs=rngs)
        self.z_proj = nnx.Linear(d_z, d_dec, rngs=rngs)
        self.lat_proj = nnx.Linear(d_model, d_dec, rngs=rngs)
        self.dec = _nnx_data([CrossBlockJax(d_dec, n_heads, mlp_mult=2,
                                            rngs=rngs)
                              for _ in range(dec_layers)])
        self.ln_out = nnx.LayerNorm(d_dec, epsilon=1e-5, rngs=rngs)
        self.head = nnx.Linear(d_dec, 2, rngs=rngs)

    # ------------------------------------------------------------- counting --
    def param_count(self):
        """The torch `param_count()`'s answer, computed over the NNX pytree.

        `coord.freqs` is a buffer and is excluded on both sides — it is not an
        `nnx.Param` here and it is `persistent=False` there — so the two
        counts are the same number and a mismatch is a real architectural
        difference rather than a bookkeeping one.
        """
        st = nnx.state(self, nnx.Param)
        return int(sum(int(np.prod(np.shape(v)))
                       for v in jax.tree_util.tree_leaves(st)))

    # -------------------------------------------------------------- encoder --
    def tokens(self, b):
        """`(toks [B, 2+C+N, d_model], kpm [B, 2+C+N] bool)`.

        `ml/cone_codec.py:ConeMAE.tokens`, statement for statement. Order is
        [cls, ctx, patch x C, dots x N], and the key-padding mask is the ONE
        place existence is enforced: `~valid` for the dots, all-False for the
        cls/ctx/patch tokens, which is why no row is ever fully masked.
        """
        dt = jnp.float32
        pv = jnp.asarray(b["patch_vals"]).astype(dt)             # [B, C, 9]
        po_b = jnp.asarray(b["patch_obs"])                       # bool
        po = po_b.astype(dt)
        B, C = pv.shape[0], pv.shape[1]
        chan_mask = b.get("chan_mask")
        if chan_mask is None:
            chan_mask = jnp.zeros((B, C), dtype=bool)

        # ---- lag-0 patch tokens, one per channel --------------------------
        feat = jnp.concatenate([pv * po, po], axis=-1)           # [B, C, 18]
        ci = jnp.arange(C)
        depth_c = jnp.asarray(b["chan_depth"]).astype(dt)        # [C]
        zero = jnp.zeros((B, C), dtype=dt)
        base_p = (jnp.broadcast_to(self.chan_emb(ci)[None],
                                   (B, C, self.d_model))
                  + self.coord(zero, zero, zero,
                               jnp.broadcast_to(depth_c[None], (B, C))))
        obs_c = po_b[..., 4]                                     # centre cell
        vis = (obs_c & ~chan_mask)[..., None]
        hid = (obs_c & chan_mask)[..., None]
        mis = (~obs_c)[..., None]
        vt = self.val_proj(feat)
        zt = jnp.zeros_like(vt)
        # THE ORDER OF THESE THREE ADDITIONS IS PART OF THE PARITY CLAIM.
        t_patch = (jnp.where(vis, vt, zt)
                   + jnp.where(hid, self.mask_tok.value, zt)
                   + jnp.where(mis, self.miss_tok.value, zt)
                   + base_p)

        # ---- dot tokens ----------------------------------------------------
        dv = jnp.asarray(b["vals"]).astype(dt)                   # [B, N]
        obs_d = jnp.asarray(b["obs"])                            # bool
        do = obs_d.astype(dt)
        N = dv.shape[1]
        dot_mask = b.get("dot_mask")
        if dot_mask is None:
            dot_mask = jnp.zeros((B, N), dtype=bool)
        base_d = (self.chan_emb(jnp.asarray(b["chan"]).astype(jnp.int32))
                  + self.coord(jnp.asarray(b["dy_km"]).astype(dt),
                               jnp.asarray(b["dx_km"]).astype(dt),
                               jnp.asarray(b["lag_days"]).astype(dt),
                               jnp.asarray(b["depth"]).astype(dt)))
        dfeat = jnp.stack([dv * do, do], axis=-1)                # [B, N, 2]
        dvt = self.dot_proj(dfeat)
        zd = jnp.zeros_like(dvt)
        vis = (obs_d & ~dot_mask)[..., None]
        hid = (obs_d & dot_mask)[..., None]
        mis = (~obs_d)[..., None]
        t_dot = (jnp.where(vis, dvt, zd)
                 + jnp.where(hid, self.mask_tok.value, zd)
                 + jnp.where(mis, self.miss_tok.value, zd)
                 + base_d)

        toks = jnp.concatenate([
            jnp.broadcast_to(self.cls_tok.value, (B, 1, self.d_model)),
            self.ctx_proj(jnp.asarray(b["ctx"]).astype(dt))[:, None, :],
            t_patch, t_dot], axis=1)
        # THE ONE PLACE EXISTENCE IS ENFORCED.
        kpm = jnp.concatenate([jnp.zeros((B, 2 + C), dtype=bool),
                               ~jnp.asarray(b["valid"])], axis=1)
        return toks, kpm

    def encode(self, b):
        """`(z [B, d_z], latents [B, n_latents, d_model])`."""
        toks, kpm = self.tokens(b)
        B = toks.shape[0]
        lat = jnp.broadcast_to(self.latents.value[None],
                               (B, self.n_latents, self.d_model))
        lat = self.cross(lat, toks, key_padding_mask=kpm)
        lat = self.encoder(lat)
        q = jnp.broadcast_to(self.pool_q.value, (B, 1, self.d_model))
        # torch writes `self.ln_pool(lat)` twice, for K and for V; it is a
        # deterministic function of `lat`, so computing it once is the same
        # numbers and one fewer normalisation.
        kv = self.ln_pool(lat)
        h = self.pool(q, kv, kv)
        return self.to_z(h[:, 0]), lat

    # -------------------------------------------------------------- decoder --
    def query_tokens(self, chan, dy_km, dx_km, lag_days, depth):
        """[B, Q, d_dec] decoder queries. The decoder never sees a value."""
        dt = jnp.float32
        e = (self.chan_emb(jnp.asarray(chan).astype(jnp.int32))
             + self.coord(jnp.asarray(dy_km).astype(dt),
                          jnp.asarray(dx_km).astype(dt),
                          jnp.asarray(lag_days).astype(dt),
                          jnp.asarray(depth).astype(dt))
             + self.query_tok.value)
        return self.q_proj(e)

    def _run_dec(self, mem, q):
        for blk in self.dec:
            q = blk(q, mem)
        out = self.head(self.ln_out(q))
        return out[..., 0], jnp.clip(out[..., 1], LOGVAR_MIN, LOGVAR_MAX)

    def decode_from_z(self, z, queries):
        """(mu, logvar) reading `z` ALONE — the headline path."""
        return self._run_dec(self.z_proj(z)[:, None, :], queries)

    def decode(self, z, latents, queries):
        """(mu, logvar) reading [z-token] + the latents — the aux path."""
        mem = jnp.concatenate([self.z_proj(z)[:, None, :],
                               self.lat_proj(latents)], axis=1)
        return self._run_dec(mem, queries)

    # -------------------------------------------------------------- queries --
    def query_sets_given(self, b, plan, dot_mask, dot_idx, chan_mask=None):
        """`ml/cone_codec.py:_query_sets` with the hidden-dot draw HANDED IN.

        Deterministic by construction — there is no RNG on this side at all.
        `dot_idx` is `(idx [B, k] int, sel [B, k] bool)` from torch's
        `ConeMAE.draw_dot_queries` (or the trainer's numpy equivalent), and
        `torch.gather(dim=1)` is `jnp.take_along_axis(..., axis=1)`.

        `chan_mask` [B, C] is read only by `plan["anchor_hidden_only"]`
        (E-069b), which zeroes family A's weight on every channel this batch
        element did NOT drop — the torch twin does exactly this, and the
        parity gate compares the two on a SHARED draw under that plan.

        Returns `(chan, dy, dx, lag, depth, target, weight)`, each [B, Q],
        concatenated in the SAME order the torch method concatenates them —
        anchor reconstruction, then the future queries, then the hidden dots —
        because `Q` is a position the decoder's loss is summed over and a
        permutation would move the sum in float.
        """
        dt = jnp.float32
        B = jnp.asarray(b["vals"]).shape[0]
        C = self.n_chan
        cw = jnp.asarray(plan["chan_w"]).astype(dt)              # [C]
        depth_c = jnp.asarray(b["chan_depth"]).astype(dt)

        chans, dys, dxs, lags, deps, tgts, ws = [], [], [], [], [], [], []

        # ---- A. anchor reconstruction -------------------------------------
        if plan.get("anchor_recon", True):
            ci = jnp.broadcast_to(jnp.arange(C)[None], (B, C))
            z = jnp.zeros((B, C), dtype=dt)
            chans.append(ci)
            dys.append(z), dxs.append(z), lags.append(z)
            deps.append(jnp.broadcast_to(depth_c[None], (B, C)))
            tgts.append(jnp.asarray(b["patch_vals"])[..., 4].astype(dt))
            aw = jnp.asarray(b["patch_obs"])[..., 4].astype(dt) * cw[None]
            if plan.get("anchor_hidden_only", False):
                if chan_mask is None:
                    raise ValueError(
                        "anchor_hidden_only needs the batch's chan_mask; "
                        "query_sets_given was called without one. "
                        "`loss_given` passes it.")
                aw = aw * jnp.asarray(chan_mask).astype(dt)
            ws.append(aw)

        # ---- B. future queries --------------------------------------------
        fut = b.get("fut_vals")
        if plan.get("future", True) and fut is not None and fut.shape[-1]:
            fut = jnp.asarray(fut)
            F = fut.shape[-1]
            flags = jnp.asarray(plan["future_lags"]).astype(dt)  # [F], pentads
            ci = jnp.broadcast_to(jnp.arange(C)[None, :, None], (B, C, F))
            z = jnp.zeros((B, C, F), dtype=dt)
            chans.append(ci.reshape(B, C * F))
            dys.append(z.reshape(B, C * F))
            dxs.append(z.reshape(B, C * F))
            # NEGATIVE lag days — the future is the past's mirror on the axis
            # the coordinate encoding already carries.
            lags.append(jnp.broadcast_to((-5.0 * flags)[None, None, :],
                                         (B, C, F)).reshape(B, C * F))
            deps.append(jnp.broadcast_to(depth_c[None, :, None],
                                         (B, C, F)).reshape(B, C * F))
            tgts.append(fut.astype(dt).reshape(B, C * F))
            ws.append((jnp.asarray(b["fut_obs"]).astype(dt)
                       * cw[None, :, None]).reshape(B, C * F))

        # ---- C. hidden dots ------------------------------------------------
        idx, sel = dot_idx
        idx = jnp.asarray(idx).astype(jnp.int32)
        sel = jnp.asarray(sel)
        if idx.shape[1] > 0:
            gchan = jnp.take_along_axis(
                jnp.asarray(b["chan"]).astype(jnp.int32), idx, axis=1)
            chans.append(gchan)
            for key, sink in (("dy_km", dys), ("dx_km", dxs),
                              ("lag_days", lags), ("depth", deps),
                              ("vals", tgts)):
                sink.append(jnp.take_along_axis(jnp.asarray(b[key]), idx,
                                                axis=1).astype(dt))
            ws.append(sel.astype(dt) * cw[gchan])

        return (jnp.concatenate(chans, 1), jnp.concatenate(dys, 1),
                jnp.concatenate(dxs, 1), jnp.concatenate(lags, 1),
                jnp.concatenate(deps, 1), jnp.concatenate(tgts, 1),
                jnp.concatenate(ws, 1))

    # -------------------------------------------------------------- forward --
    def loss_given(self, b, plan, chan_mask, dot_mask, dot_idx):
        """`ml/cone_codec.py:ConeMAE.forward_given` — `(loss, z, terms)`.

        A TUPLE rather than torch's dict, because this is what an `nnx.jit`ed
        step function returns and a dict of traced scalars is the same thing
        with a less predictable pytree. `terms` carries the same keys with the
        same meanings; they are jnp scalars here (torch's are python floats,
        which is a `.item()` the trainer can do once per step instead of eight
        times inside the step).
        """
        bb = dict(b)
        bb["chan_mask"], bb["dot_mask"] = chan_mask, dot_mask
        z, lat = self.encode(bb)

        chan, dy, dx, lag, dep, tgt, w = self.query_sets_given(
            b, plan, dot_mask, dot_idx, chan_mask)
        q = self.query_tokens(chan, dy, dx, lag, dep)
        mu, logvar = self.decode_from_z(z, q)

        wsum = jnp.maximum(w.sum(), 1e-6)
        nll = (nll_gauss(mu, logvar, tgt) * w).sum() / wsum
        # LOGGED, not optimised (ml/CLAUDE.md §4.3).
        mse = (((mu - tgt) ** 2) * w).sum() / wsum

        terms = {"nll": nll, "mse": mse,
                 "wsum": wsum,
                 "n_targets": (w > 0).sum().astype(jnp.float32),
                 "logvar_mean": (logvar * w).sum() / wsum,
                 "frac_chan_masked": jnp.asarray(chan_mask).astype(
                     jnp.float32).mean(),
                 "frac_dot_masked": jnp.asarray(dot_mask).astype(
                     jnp.float32).mean()}

        loss = nll
        aux_w = float(plan.get("aux_latent_w", 0.0))
        if aux_w > 0.0:
            mu2, lv2 = self.decode(z, lat, q)
            aux = (nll_gauss(mu2, lv2, tgt) * w).sum() / wsum
            terms["nll_latent"] = aux
            loss = loss + aux_w * aux
        return loss, z, terms

"""`ml/temporal.py:embed_everything` in JAX — the tier-2 embedding path.

This is the loop the whole eval stack stands on: for every timestep and every
pixel in (ys, xs), assemble the codec's per-batch inputs exactly as the torch
original does and run `encode`, producing Z of shape [T, P, d_z]. Every probe
in the programme reads that array and nothing else, so a drift here is a drift
in every number the port could ever be scored with — which is why the parity
test (`tests/test_jaxport_embed.py`) pins this against the torch original
elementwise rather than against a summary of it.

**What is deliberately NOT here.** The torch original carries the operational
plumbing that makes a 95-minute embedding survivable on a rented box: a disk
memmap keyed by the codec's weight hash, a resumable `.partial` with a
progress marker, and the disk/RAM decision (`_cache_plan`). None of it is
arithmetic. This is a REFERENCE path (`ml/plans/JAX_PORT.md` §1), so Z is a
plain in-RAM array and the caller owns persistence.

**Memory shape.** Nothing of size T×H×W is ever materialised. X and OBS are
`LazyPixels` — `nan_to_num`/`isfinite` evaluated AFTER the per-batch index,
not before (`ml/model.py:LazyPixels`) — and the only full-size object this
function owns is the output, T·P·d_z float16. On the section probe that is
516·265·64·2 = 17.5 MB; at global scale (P ≈ 84k) it is 5.2 GiB and the caller
must hand it somewhere with room, exactly as the torch side does. On a box
whose RAM is 1.55× oversubscribed against the tensor itself, a single
[T,H,W] temporary is the difference between running and being OOM-killed.

Two parity traps that cost real debugging on the torch side and are pinned by
the test here:

  * **`CACHE_DTYPE` is float16**, and the cast happens on the way INTO Z — the
    encoder arithmetic is float32 throughout. Rounding at storage rather than
    at compute is what every published probe number was produced with.
  * **The patch>1 gather is not a plain neighbourhood slice**: longitude WRAPS
    (the globe is periodic in x), latitude CLAMPS with the out-of-range rows
    marked UNOBSERVED, and the centre cell (index patch²//2) is what decides
    whether the channel counts as observed at all. Getting the observed flags
    right matters more than getting the values right — a wrong flag turns a
    land cell into an observed 0.0 with no NaN anywhere to notice.
"""
import time

import jax.numpy as jnp
import numpy as np
from flax import nnx

# The cache dtype is a PROTOCOL constant, not a local choice: Z is compared
# and pooled across implementations, so it is imported from the torch module
# that defines it rather than restated here (a restated constant is one that
# can drift).
try:                                              # `ml/` on sys.path
    from temporal import CACHE_DTYPE
except ImportError:                               # imported as `ml.temporal`
    from ml.temporal import CACHE_DTYPE


def gather_px_np(Xt, OBS, t, y, x, patch):
    """Numpy-native `ml/model.py:gather_px`, asserted equal to it by
    `tests/test_jaxport_embed.py`.

    Written in numpy rather than reusing `gather_px` directly so the JAX data
    path never routes a batch through torch tensors — this package must be
    usable without torch installed, and the converter is the one place that
    imports it. The indexing is elementwise-identical: `np.asarray` accepts a
    CPU torch tensor unchanged, so it works against `LazyPixels` (which hands
    back `torch.from_numpy(...)`) and against a bare numpy array alike.

    patch=1 → ([B,C], [B,C]). patch>1 → ([B,C,patch²], [B,C,patch²]) with
    longitude WRAPPED, latitude CLAMPED and out-of-range rows marked
    unobserved. Centre cell is index patch²//2.
    """
    t = np.asarray(t)
    y = np.asarray(y)
    x = np.asarray(x)
    if patch == 1:
        return np.asarray(Xt[t, y, x]), np.asarray(OBS[t, y, x])
    H, W = Xt.shape[1], Xt.shape[2]
    r = patch // 2
    vs, os_ = [], []
    for dy in range(-r, r + 1):
        yy = np.clip(y + dy, 0, H - 1)
        vy = ((y + dy) >= 0) & ((y + dy) <= H - 1)
        for dx in range(-r, r + 1):
            xx = (x + dx) % W
            vs.append(np.asarray(Xt[t, yy, xx]))
            os_.append(np.asarray(OBS[t, yy, xx]) & vy[:, None])
    return np.stack(vs, -1), np.stack(os_, -1)


def _encode_fn():
    """`model.encode` under `nnx.jit`.

    Built once per call of `embed_everything_jax` rather than at import: the
    cache is keyed on the traced function object, and a module-level one would
    hold every model it has ever seen alive for the life of the process.
    """
    @nnx.jit
    def _enc(model, x, obs, mask, ctx):
        return model.encode(x, obs, mask, ctx)
    return _enc


def embed_everything_jax(model, X, OBS, ctx_all, lats, lons, ys, xs, d_z,
                         batch=8192, mask_chan=None, progress=None,
                         t_sel=None, blk_rows=None, blk_pad=None, quiet=False):
    """Frozen-codec embeddings for every (t, pixel in ys/xs): [T, P, d_z].

    Same signature, same semantics and the same float16 output convention as
    `ml/temporal.py:embed_everything`, minus the disk cache (see the module
    docstring). Returns `(Z, coords)`; `coords` is [P,2] float32, lat/90 and
    lon/180, which is what the context token's last two slots carry.

    `t_sel` EMBEDS ONLY THE TIMESTEPS ASKED FOR and returns [len(t_sel), P,
    d_z] in that order — the in-training light probe reads ~46% of the rows
    and used to pay for all of them. Every timestep is an independent forward
    (the loop has no state that crosses `t`), so the rows that come back are
    bit-identical to the corresponding rows of a full pass.

    `blk_rows`/`blk_pad` are E-047's block map (`ml/timeblocks.py:BlockAxis`)
    and are REQUIRED by a k_time > 1 codec: embedding a block codec one bin at
    a time would silently produce embeddings of a different thing than it was
    trained on, which is why the torch original raises on the same condition
    rather than falling back.
    """
    T, H, W, C = X.shape
    P = len(ys)
    if getattr(model, "k_time", 1) > 1:
        if blk_rows is None:
            raise ValueError(
                "embed_everything_jax: this codec is a BLOCK codec (k_time="
                f"{model.k_time}) and no block map was passed. Embedding it "
                "one bin at a time would silently produce embeddings of a "
                "different thing than it was trained on.")
        T = len(blk_rows)                      # the BLOCK axis is the axis
    ts = np.arange(T) if t_sel is None else np.asarray(t_sel, np.int64)
    T_out = len(ts)
    coords = np.stack([lats[ys] / 90, lons[xs] / 180], 1).astype(np.float32)
    out = np.zeros((T_out, P, d_z), dtype=CACHE_DTYPE)
    patch = getattr(model, "patch", 1)
    enc = _encode_fn()

    # The mask is constant across the whole pass (mask_chan is a channel set,
    # not a per-pixel draw), so it is built once per batch SHAPE rather than
    # per batch — nnx.jit would otherwise retrace on nothing.
    def mask_for(n):
        mk = np.zeros((n, C), dtype=bool)
        if mask_chan is not None:
            mk[:, mask_chan] = True
        return mk

    t_emb = time.time()
    next_mark = 0.0
    for i_out in range(T_out):
        t = int(ts[i_out])
        frac = (i_out + 1) / T_out
        if frac >= next_mark:
            el = time.time() - t_emb
            eta = el / frac - el if frac > 0 else 0
            if not quiet:
                print(f"  embedding {frac * 100:5.1f}%  month "
                      f"{i_out + 1}/{T_out}  {el / 60:.0f} min elapsed, "
                      f"~{eta / 60:.0f} min left", flush=True)
            if progress:
                progress({"pct": round(frac * 100, 1), "month": i_out + 1,
                          "months": T_out, "elapsed_s": round(el),
                          "eta_s": round(eta), "where": "ram"})
            next_mark = frac + 0.05
        for i in range(0, P, batch):
            sl = slice(i, min(i + batch, P))
            n = sl.stop - sl.start
            ctx = np.concatenate([np.tile(ctx_all[t], (n, 1)), coords[sl]], 1)
            mk = mask_for(n)
            ctx_j = jnp.asarray(ctx, jnp.float32)
            if patch > 1:
                tt = np.full((n,), t, dtype=np.int64)
                v, o = gather_px_np(X, OBS, tt, ys[sl], xs[sl], patch)
                # `v * ~mk[..., None]` before the encoder, exactly as the
                # torch caller does: the masked channels' VALUES are zeroed
                # here and the mask token is applied inside `encode`. Doing
                # only one of the two is a silent half-mask.
                v = v * (~mk)[..., None]
            elif getattr(model, "k_time", 1) > 1:
                # E-047 BLOCK CODEC. `t` indexes the BLOCK axis and
                # `blk_rows[t]` names its source rows; the grid is assembled at
                # FULL VISIBILITY exactly as the per-bin path is, with pad
                # cells forced unobserved.
                rr = blk_rows[t]
                v = np.stack([np.asarray(X[int(r), ys[sl], xs[sl]])
                              for r in rr], 1)
                o = np.stack([np.asarray(OBS[int(r), ys[sl], xs[sl]])
                              for r in rr], 1)
                o = o & (~np.asarray(blk_pad[t]))[None, :, None]
                mk = np.zeros_like(o)
                v = v * ~mk
            else:
                v = np.asarray(X[t, ys[sl], xs[sl]]) * (~mk)
                o = np.asarray(OBS[t, ys[sl], xs[sl]])
            z = enc(model, jnp.asarray(v), jnp.asarray(o),
                    jnp.asarray(mk), ctx_j)
            # np.asarray(z) is float32; the store rounds to CACHE_DTYPE, which
            # is where — and only where — the torch original rounds too.
            out[i_out, sl] = np.asarray(z)
    return out, coords

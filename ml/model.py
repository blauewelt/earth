"""PixelMAE — a masked autoencoder over one pixel's channels, with a
queryable decoder that predicts the SAME pixel's masked channels and its
NEIGHBOURS in space and time from the bottleneck alone.

Shape of the idea (proposal §4–5, scaled to a pilot):

  channels of pixel (lat, lon, t)      ┌─ mask some observed channels
        │  one token per channel  ◄────┘
        ▼
  transformer encoder  (missing channels enter as explicit "missing" tokens —
        │               absence is information, not padding)
        ▼
  z = bottleneck(CLS)   d_z-dimensional; THE embedding
        ▼
  decoder(z, channel-id, Δlat, Δlon, Δt) → value
        Δ = (0,0,0)  reconstruct this pixel (masked channels score)
        Δ = space    predict the 4-neighbours' channels, same month
        Δ = time     predict this pixel's channels next/previous month

Why neighbour heads: a plain autoencoder can ace reconstruction by memorising
the seasonal cycle (proposal §5). Forcing z to answer for pixels it never saw
makes it carry STATE — the currency the AMOC probe then reads.

The decoder is a neural-field-style MLP conditioned on (z, query): it can be
asked for any offset at inference, which is what "use the embedding to predict
nearby pixels" means operationally.
"""
import numpy as np
import torch
import torch.nn as nn


class PixelMAE(nn.Module):
    def __init__(self, n_chan, d_model=128, n_heads=4, n_layers=4,
                 d_z=32, d_dec=256, max_abs_offset=3, patch=1, dec_layers=2,
                 k_time=1):
        super().__init__()
        self.n_chan = n_chan
        self.d_z = d_z
        self.max_off = max_abs_offset
        # E-047 TIME BLOCKS. k_time = 1 is the per-bin codec every archived
        # checkpoint is, and it adds NO parameter and NO branch that runs:
        # `time_emb` is created only when k_time > 1, so a k_time=1 model's
        # state_dict is key-for-key what it has always been. k_time > 1 makes
        # the encoder's input a k_time x C GRID of cells — one month of pentad
        # bins, say — whose cell token is the channel embedding plus a learned
        # WITHIN-BLOCK TIME-OFFSET embedding, so the encoder can tell the 3rd
        # pentad of the month from the 5th. Missing and PAD cells go through
        # the existing miss_tok path unchanged, which is the whole design:
        # Argo present in one bin of six is not a special case, it is six
        # cells of which five are unobserved.
        self.k_time = int(k_time)
        # patch=1: one value per channel token (the pilot design).
        # patch=3: each channel token carries its 3x3 neighbourhood — 9
        # values + 9 observed flags through one projection. Same token
        # count, but the encoder can SEE GRADIENTS (thermal wind is a
        # density gradient), and z becomes a true compression
        # (~160 observed values -> d_z at 25 channels).
        self.patch = patch
        p2 = patch * patch

        # --- encoder tokens -------------------------------------------------
        self.val_proj = (nn.Linear(1, d_model) if patch == 1
                         else nn.Linear(2 * p2, d_model))
        self.chan_emb = nn.Embedding(n_chan, d_model)
        if self.k_time > 1:
            self.time_emb = nn.Embedding(self.k_time, d_model)
        self.mask_tok = nn.Parameter(torch.zeros(d_model))     # channel masked by US
        self.miss_tok = nn.Parameter(torch.zeros(d_model))     # channel unobserved in the DATA
        self.cls_tok = nn.Parameter(torch.zeros(d_model))
        # coords/season enter as one non-maskable context token:
        # [sin m, cos m, lat/90, lon/180]
        self.ctx_proj = nn.Linear(4, d_model)

        enc_layer = nn.TransformerEncoderLayer(
            d_model, n_heads, dim_feedforward=4 * d_model,
            batch_first=True, norm_first=True, dropout=0.0)
        self.encoder = nn.TransformerEncoder(enc_layer, n_layers)
        self.to_z = nn.Linear(d_model, d_z)

        # --- queryable decoder ---------------------------------------------
        # query = channel id + integer offset (Δlon, Δlat, Δmonth), each
        # embedded; decoder never sees the input values, only z.
        self.q_chan = nn.Embedding(n_chan, 64)
        self.q_off = nn.Embedding(2 * max_abs_offset + 1, 16)  # shared per axis
        # E-047, decided 2026-08-22 (option b): WITHIN-BLOCK POSITION GETS ITS
        # OWN QUERY EMBEDDING. The alternative was to reuse the `dt` slot of
        # `off`, which fits (a 7-cell block centred is dt in -3..+3, exactly
        # this table's range) and costs nothing — and would make ONE index
        # mean two things, "one bin later inside this block" for a cell query
        # and "one block later" for the neighbour loss, with nothing in the
        # input telling the decoder which. One symbol, one meaning. `off`'s
        # dt keeps meaning BLOCKS, which is what the axis means downstream.
        if self.k_time > 1:
            self.q_time = nn.Embedding(self.k_time, 16)
        # dec_layers = HIDDEN layers. The historical decoder (2 hidden) is a
        # ~1.3M-param MLP against a 40M encoder; E-019a measured the round
        # trip losing 6.9% of deep-temperature variance, so the depth is now
        # a knob (E-019b). dec_layers=2 reproduces every old checkpoint
        # exactly — codec_from_ckpt defaults it for args written before the
        # knob existed.
        dec = [nn.Linear(d_z + 64 + 3 * 16 + (16 if self.k_time > 1 else 0),
                         d_dec), nn.GELU()]
        for _ in range(dec_layers - 1):
            dec += [nn.Linear(d_dec, d_dec), nn.GELU()]
        dec += [nn.Linear(d_dec, 1)]
        self.decoder = nn.Sequential(*dec)
        for p in (self.mask_tok, self.miss_tok, self.cls_tok):
            nn.init.normal_(p, std=0.02)

    def encode(self, x, obs, mask, ctx):
        """patch=1: x, obs [B,C] · patch>1: x, obs [B,C,patch²] (obs = that
        cell observed; the channel counts as observed iff its CENTER is).
        mask [B,C] bool masked-by-training · ctx [B,4] → z [B,d_z]."""
        # WIDEN THE INPUT TO THE WEIGHTS' DTYPE. Family 4 is the project's
        # first float16 tensor, and every reader hands the codec a batch whose
        # dtype follows the tensor — `torch.from_numpy(np.nan_to_num(X))` did,
        # and LazyPixels preserves that faithfully. Against float32 weights
        # that is an immediate
        #     RuntimeError: mat1 and mat2 must have the same dtype,
        #                   but got Half and Float
        # on the first forward pass. Run #365 never reached it: the host OOM
        # killer took the process during the preamble, so the pentad arm would
        # have died here on the re-dispatch instead, one failure later, after
        # another tensor build. Caught on a 48x12x14 toy in seconds
        # (ml/CLAUDE.md §4.8) rather than on a rented GPU.
        #
        # Here rather than at the six call sites: probe_kfold, temporal,
        # rollout, probe_sequence, ablate_channels and train all build their
        # own value tensor, and a cast per call site is five chances to miss
        # one. float16 -> float32 is exact, and on family 2/3 (float32) it is
        # a no-op, so no existing run's arithmetic moves.
        wdt = self.val_proj.weight.dtype
        if x.dtype != wdt:
            x = x.to(wdt)
        if ctx.dtype != wdt:
            ctx = ctx.to(wdt)
        if self.patch > 1:
            B, C, P2 = x.shape
            ce = self.chan_emb.weight[None, :, :].expand(B, -1, -1)
            feat = torch.cat([x * obs, obs.to(x.dtype)], -1)   # [B,C,2·P2]
            vt = self.val_proj(feat) + ce
            obs = obs[..., P2 // 2]                            # center defines the channel
            vt = torch.where((obs & ~mask).unsqueeze(-1), vt, torch.zeros_like(vt))
            vt = vt + torch.where((obs & mask).unsqueeze(-1),
                                  self.mask_tok.expand(B, C, -1) + ce, torch.zeros_like(vt))
            vt = vt + torch.where((~obs).unsqueeze(-1),
                                  self.miss_tok.expand(B, C, -1) + ce, torch.zeros_like(vt))
            toks = torch.cat([self.cls_tok.expand(B, 1, -1),
                              self.ctx_proj(ctx).unsqueeze(1), vt], dim=1)
            h = self.encoder(toks)
            return self.to_z(h[:, 0])
        if self.k_time > 1:
            # x, obs, mask [B, k_time, C] -> [B, k_time*C] tokens, cell (j, c)
            # carrying chan_emb[c] + time_emb[j]. Flattened j-major so a
            # reader of the attention map sees each bin's channels together.
            B, KT, C = x.shape
            assert KT == self.k_time, (KT, self.k_time)
            x = x.reshape(B, KT * C)
            obs = obs.reshape(B, KT * C)
            mask = mask.reshape(B, KT * C)
            ce = (self.chan_emb.weight[None, None, :, :]
                  + self.time_emb.weight[None, :, None, :]).reshape(
                      1, KT * C, -1).expand(B, -1, -1)
            vt = self.val_proj(x.unsqueeze(-1)) + ce
            vt = torch.where((obs & ~mask).unsqueeze(-1), vt,
                             torch.zeros_like(vt))
            vt = vt + torch.where((obs & mask).unsqueeze(-1),
                                  self.mask_tok.expand(B, KT * C, -1) + ce,
                                  torch.zeros_like(vt))
            vt = vt + torch.where((~obs).unsqueeze(-1),
                                  self.miss_tok.expand(B, KT * C, -1) + ce,
                                  torch.zeros_like(vt))
            toks = torch.cat([self.cls_tok.expand(B, 1, -1),
                              self.ctx_proj(ctx).unsqueeze(1), vt], dim=1)
            return self.to_z(self.encoder(toks)[:, 0])
        B, C = x.shape
        ce = self.chan_emb.weight[None, :, :].expand(B, -1, -1)
        vt = self.val_proj(x.unsqueeze(-1)) + ce
        vt = torch.where((obs & ~mask).unsqueeze(-1), vt, torch.zeros_like(vt))
        vt = vt + torch.where((obs & mask).unsqueeze(-1),
                              self.mask_tok.expand(B, C, -1) + ce, torch.zeros_like(vt))
        vt = vt + torch.where((~obs).unsqueeze(-1),
                              self.miss_tok.expand(B, C, -1) + ce, torch.zeros_like(vt))
        toks = torch.cat([self.cls_tok.expand(B, 1, -1),
                          self.ctx_proj(ctx).unsqueeze(1), vt], dim=1)
        h = self.encoder(toks)
        return self.to_z(h[:, 0])

    def query(self, z, chan_idx, off, tpos=None):
        """z [B,d_z] · chan_idx [B,Q] · off [B,Q,3] ints in [-max,max] → [B,Q].

        `tpos` [B,Q] is the WITHIN-BLOCK cell position and is required exactly
        when k_time > 1 — a block codec that was asked for "channel c" without
        saying WHICH CELL would be answering a question with no answer, so it
        raises rather than picking one."""
        B, Q = chan_idx.shape
        qc = self.q_chan(chan_idx)
        qo = self.q_off(off + self.max_off).reshape(B, Q, -1)
        zq = z.unsqueeze(1).expand(-1, Q, -1)
        parts = [zq, qc, qo]
        if self.k_time > 1:
            if tpos is None:
                raise ValueError(
                    f"query(): this codec has k_time={self.k_time}, so every "
                    f"query names a cell (tpos [B,Q] in 0..{self.k_time - 1}) "
                    f"as well as a channel")
            parts.append(self.q_time(tpos))
        return self.decoder(torch.cat(parts, dim=-1)).squeeze(-1)


class LazyPixels:
    """A drop-in stand-in for `nan_to_num(X)` / `isfinite(X)` that derives
    them PER BATCH instead of materialising either at full size.

    WHY. `ml/train.py` used to build both eagerly:

        Xt  = torch.from_numpy(np.nan_to_num(X, nan=0.0))
        OBS = torch.from_numpy(np.isfinite(X))

    For family 3 ([516, 281, 481, 39]) that is 13.6 GB alongside X and nobody
    noticed. For family 4's pentad tensor ([3142, 281, 481, 39] float16) it is
    **33.1 GB for X + 33.1 GB for the copy + 16.6 GB for the mask = 82.8 GB**
    against a 64 GB box, and run #365 was killed by the host OOM killer (exit
    137) after six hours. Measured, not modelled — the element count is
    16,562,358,618.

    Both arrays are pure functions of X evaluated elementwise, and every
    consumer only ever indexes a BATCH of pixels out of them. So computing
    them after the index rather than before is arithmetically identical and
    costs a few hundred KB instead of 49.7 GB. This removes the failure mode
    rather than guarding it (ml/CLAUDE.md §4.1); the daily tensor is 5x larger
    again and would not have fitted any box under the old shape.

    Behaviour is preserved by construction: the SAME numpy functions are
    applied to the SAME elements, and dtype follows X exactly as
    `torch.from_numpy` did — so a float16 tensor still yields float16 and a
    float32 one still yields float32. `tests/test_train_lazy_pixels.py` pins
    that against the eager arrays elementwise.
    """

    def __init__(self, X, obs=False):
        self._X = X
        self._obs = obs
        self.shape = X.shape          # gather_px reads .shape[1], .shape[2]

    def __len__(self):
        return self._X.shape[0]

    def __getitem__(self, idx):
        # Consumers index with torch CPU tensors; numpy needs arrays.
        if isinstance(idx, tuple):
            idx = tuple(np.asarray(i) if hasattr(i, "numpy") else i for i in idx)
        elif hasattr(idx, "numpy"):
            idx = np.asarray(idx)
        raw = self._X[idx]
        if self._obs:
            return torch.from_numpy(np.isfinite(raw))
        return torch.from_numpy(np.nan_to_num(raw, nan=0.0))


def gather_px(Xt, OBS, t, y, x, patch):
    """Gather encoder inputs for pixels (t, y, x) from full tensors.
    patch=1 → ([B,C], [B,C]) as before. patch>1 → ([B,C,patch²],
    [B,C,patch²]): each channel's neighbourhood, longitude WRAPPED (the
    globe is periodic in x), latitude clamped with the out-of-range rows
    marked unobserved. Center cell is index patch²//2."""
    if patch == 1:
        return Xt[t, y, x], OBS[t, y, x]
    H, W = Xt.shape[1], Xt.shape[2]
    r = patch // 2
    vs, os_ = [], []
    for dy in range(-r, r + 1):
        yy = (y + dy).clamp(0, H - 1)
        vy = ((y + dy) >= 0) & ((y + dy) <= H - 1)
        for dx in range(-r, r + 1):
            xx = (x + dx) % W
            vs.append(Xt[t, yy, xx])
            os_.append(OBS[t, yy, xx] & vy.unsqueeze(-1))
    return torch.stack(vs, -1), torch.stack(os_, -1)


def codec_from_ckpt(ck, n_chan):
    """Rebuild the EXACT architecture a checkpoint was trained with.

    Every loader used to hand-construct PixelMAE(n_chan, d_z, patch) — fine
    while all codecs shared one size, silently wrong the day they didn't.
    The checkpoint's args carry the full architecture (train.py saves
    vars(a)); this is the one place that reads them. Old checkpoints predate
    the size knobs, so every .get() default is the pilot architecture."""
    a = ck.get("args", {})
    return PixelMAE(n_chan=n_chan, d_z=ck["d_z"],
                    k_time=int(a.get("k_time", 1) or 1),
                    patch=a.get("patch", 1),
                    d_model=a.get("d_model", 128),
                    n_layers=a.get("n_layers", 4),
                    n_heads=a.get("n_heads", 4),
                    d_dec=a.get("d_dec", 256),
                    dec_layers=a.get("dec_layers", 2))


def obs_any_chunked(X, min_chan=2, chunk=64):
    """`np.isfinite(X).sum(-1) >= min_chan`, without the full-size temporaries.

    Identical values, elementwise. What changes is the peak: the one-liner
    materialises a [T,H,W,C] bool AND a [T,H,W] int64 at once — **15.4 GiB
    plus 3.4 GiB on the pentad tensor, 77 GiB plus 17 GiB at daily** — and
    both are live simultaneously.

    That spike is the first place `ml/train.py` can die, and it was invisible
    in the diagnosis of run #365: it is transient, so an RSS delta column
    shows nothing, and only VmHWM records it (`ml/measure_train_memory.py`
    measured 85.2 GiB resident against a 146.9 GiB peak). LazyPixels removed
    the two RESIDENT copies and left this one untouched, which would have
    OOM-killed the re-dispatch on the same 63 GB box for a different reason
    — the classic "fixed the term you can see".

    A chunk of 64 timesteps costs 337 MB regardless of T, so this is the term
    that stops scaling with the tensor. `np.count_nonzero(..., axis=-1)` is
    the same reduction `.sum(-1)` performs on a bool, spelled so it cannot
    accidentally accumulate in the input dtype.
    """
    out = np.empty(X.shape[:3], bool)
    for i in range(0, X.shape[0], chunk):
        sl = X[i:i + chunk]
        out[i:i + chunk] = np.count_nonzero(np.isfinite(sl), axis=-1) >= min_chan
    return out


def pool_idx(mask, chunk=256):
    """`np.where(mask)` as int32 triples, in the identical order.

    Two savings, both structural rather than clever:

      · **int32, not int64.** These arrays are indices into a [T,H,W] volume
        whose largest axis is 15,706 at daily cadence, so int64 spends exactly
        half its bytes on sign-extension. Family 4's train pool is ~272M
        pixels — 6.5 GiB as int64, 3.3 GiB as int32 — and it stays resident
        for the whole run.
      · **chunked over T**, so the int64 array numpy builds internally is
        1/12th of the pool at a time rather than all of it. Counted first and
        written into a preallocated output rather than concatenated: a
        concatenate holds the parts AND the result at once, which doubles
        exactly the term this is trying to halve.

    Order is preserved exactly because the chunks partition the FIRST axis in
    ascending order and `np.where` returns C-order within each chunk;
    concatenating them reproduces the global C-order listing. `tests/
    test_train_pool_memory.py` asserts equality against `np.where` rather
    than trusting that argument.
    """
    n = int(np.count_nonzero(mask))
    ts, ys, xs = (np.empty(n, np.int32) for _ in range(3))
    o = 0
    for i in range(0, mask.shape[0], chunk):
        t, y, x = np.where(mask[i:i + chunk])
        k = len(t)
        ts[o:o + k], ys[o:o + k], xs[o:o + k] = t + i, y, x
        o += k
    if o != n:
        raise AssertionError(f"pool_idx filled {o} of {n} — the chunk walk "
                             f"and the count disagree, which can only mean "
                             f"the mask changed underneath")
    return ts, ys, xs

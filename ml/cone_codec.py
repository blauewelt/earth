#!/usr/bin/env python3
"""E-069 · ConeMAE — the cone-native codec (inner cone of raw channels).

A sibling of `ml/model.py::PixelMAE`, NOT a modification of it: every archived
checkpoint must stay bit-identical (ml/CLAUDE.md §1, the recipe rule), and the
two models answer different questions. PixelMAE encodes ONE pixel-bin — a 3x3
patch of one pentad — so nothing that needs two snapshots (velocity, tendency,
convergence) can reach the embedding at all. ConeMAE encodes the INNER CONE:
the same lag-0 patch per channel PLUS, for every channel, a sunflower of dots
at lags 1..L_in whose reach follows that channel's family (ml/cone.py), so a
local displacement is IN the token set rather than something stage 2 has to
rebuild from compressed codes.

WHAT IS COPIED FROM PixelMAE, DELIBERATELY (plan §3, "a sibling"):

  * `val_proj` over a 3x3 patch takes 2*9 inputs — nine values and nine
    observed flags through ONE projection — and the channel counts as observed
    iff its CENTRE cell is (ml/model.py:657-667). Same layout, same order
    (ml/cone_sampler.py::PATCH_DY/PATCH_DX).
  * `mask_tok` vs `miss_tok`: a value WE hid is `mask`, a value the DATA never
    observed is `miss`. Two different statements about the same absence, and
    a model that conflated them would be told "unknowable" and "guess this".
  * `cls_tok`, and `ctx_proj` over [sin m, cos m, lat/90, lon/180] as one
    non-maskable context token.
  * `query(z, chan, off)`'s idea: the decoder never sees the input values, only
    a query naming WHAT is being asked for and the code it must be read from.

WHAT IS NEW:

  * A dot token: one (channel, dot) pair, `Linear(2, d_model)` over [value,
    observed], plus the channel embedding, plus a coordinate encoding —
    Fourier features of signed-log(dy_km), signed-log(dx_km), log(lag) and
    log(depth). The geometry is CONTINUOUS here where PixelMAE's `q_off` is a
    small integer table, because the cone's offsets run from 28 km to 907 km
    and are per-latitude (ml/cone.py::ground_km).
  * A Perceiver encoder: `n_latents` learned latents cross-attend to the whole
    token set (~1,000 tokens at the r3 tensor) at O(N*K), then self-attend
    among themselves. A full self-attention over 1,000 tokens is 250x the
    attention of PixelMAE's 42 and buys nothing — the tokens are a bag of
    measurements, not a sequence.
  * A Gaussian head (mu, log-variance) so the decoder can say "unknowable"
    where the cone genuinely does not determine a value; the plain MSE is
    logged beside it for comparability with every archived reconstruction
    number.

EXISTENCE IS ENFORCED IN EXACTLY ONE PLACE — the attention key-padding mask.
A dot that does not exist (off the basin, or before the archive starts) is
`valid=False` from the sampler and is excluded from every attention over the
token set; its token is still BUILT, from whatever the sampler left in the
padding, and its contents can therefore never reach `z`. One mechanism, and
`tests/test_cone_smoke.py` perturbs an invalid dot's value and asserts
`torch.equal` on z — a test that would be vacuous if the token construction
also zeroed it (two mechanisms, either one masking a bug in the other).

Plan: ml/plans/E069_cone_codec.md §3.
"""
import math
import os
import sys

import torch
import torch.nn as nn

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

from cone import channel_family                                   # noqa: E402

# Per-family loss weights. AIFS-ocean's practice (ECMWF's coupled AIFS ocean
# emulator, and the ocean-emulator literature generally): the SLOW fields carry
# the forecast and the fast forcing channels are noisy, so a loss that treats
# every channel alike is optimised mostly on weather. Family B (currents, SSH,
# the Argo column — the ocean state the programme forecasts) is up-weighted
# 2.0; family C (SST, MLD — slow but atmospherically stirred) 1.5; family A
# (wind stress — fast, wide, 10-day memory) stays at 1.0.
FAMILY_W = {"A": 1.0, "B": 2.0, "C": 1.5}

LOGVAR_MIN, LOGVAR_MAX = -8.0, 8.0


def family_weights(chan_names):
    """[C] float list of per-channel loss weights from `cone.channel_family`.

    An unknown channel RAISES there rather than getting a default weight —
    the same discipline the reach has (a channel with a guessed family is a
    channel whose cone is a guess).
    """
    return [FAMILY_W[channel_family(n)] for n in chan_names]


def signed_log(v):
    """sign(v) * log1p(|v|) — a log scale that survives a sign.

    The dot offsets are SIGNED ground displacements (upstream is not the same
    place as downstream) spanning 28 km to 907 km, so a bare log would lose
    the half of the cone that matters most and a bare linear scale would spend
    its whole dynamic range on the outermost ring. Applied to the lag too:
    every ENCODER lag is >= 0, where this is exactly log1p, but a FUTURE query
    sits at lag -5 or -10 days and log1p(-5) is NaN. One function, no branch,
    and the encoder side is unchanged by construction.
    """
    return torch.sign(v) * torch.log1p(v.abs())


def nll_gauss(mu, logvar, target):
    """Elementwise Gaussian negative log-likelihood, nats.

    0.5 * (log(2*pi) + logvar + (target - mu)^2 * exp(-logvar)). The head
    predicts a log-variance so the model can say "unknowable" — a dot whose
    value the cone does not determine costs 0.5*logvar instead of an
    unboundedly wrong mean — and `logvar` is clamped into [-8, 8] at the head,
    because exp(-logvar) is the gradient's scale and an unclamped one lets a
    single easy target dominate every other term in the batch.
    """
    return 0.5 * (math.log(2.0 * math.pi) + logvar
                  + (target - mu) ** 2 * torch.exp(-logvar))


class CoordEnc(nn.Module):
    """Fourier features of (dy_km, dx_km, lag_days, depth_dbar) -> d_model.

    Four scalars, each through `signed_log` and then sin/cos at `n_fourier`
    log-spaced (base-2) frequencies, so the encoding is 8*n_fourier numbers
    through one Linear. The frequencies are a BUFFER, not a parameter: they
    are the basis, and a learned basis would make two checkpoints' coordinate
    encodings incomparable for no measured gain.
    """

    def __init__(self, d_model, n_fourier=8):
        super().__init__()
        self.n_fourier = int(n_fourier)
        self.register_buffer("freqs",
                             2.0 ** torch.arange(self.n_fourier).float(),
                             persistent=False)
        self.proj = nn.Linear(8 * self.n_fourier, d_model)

    def forward(self, dy_km, dx_km, lag_days, depth):
        s = torch.stack([signed_log(dy_km), signed_log(dx_km),
                         signed_log(lag_days), signed_log(depth)], dim=-1)
        ang = s.unsqueeze(-1) * self.freqs                      # [..., 4, F]
        feat = torch.cat([torch.sin(ang), torch.cos(ang)], dim=-1)
        return self.proj(feat.flatten(-2))                      # [..., 8F]


class CrossBlock(nn.Module):
    """Pre-LN cross-attention + MLP, the Perceiver block.

    Pre-LN (`norm_first=True` in PixelMAE's encoder) for the same reason it is
    used there: post-LN needs a warmup to be stable and every dispatch here
    runs a plain schedule.
    """

    def __init__(self, d_model, n_heads, mlp_mult=4):
        super().__init__()
        self.ln_q = nn.LayerNorm(d_model)
        self.ln_kv = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True,
                                          dropout=0.0)
        self.ln_m = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(nn.Linear(d_model, mlp_mult * d_model),
                                 nn.GELU(),
                                 nn.Linear(mlp_mult * d_model, d_model))

    def forward(self, q, kv, key_padding_mask=None):
        k = self.ln_kv(kv)
        h, _ = self.attn(self.ln_q(q), k, k,
                         key_padding_mask=key_padding_mask, need_weights=False)
        q = q + h
        return q + self.mlp(self.ln_m(q))


class ConeMAE(nn.Module):
    """The cone-native codec. See the module docstring for what it copies."""

    def __init__(self, n_chan, d_model=256, n_heads=8, n_latents=64,
                 n_layers=6, d_z=32, d_dec=256, dec_layers=2, n_fourier=8):
        super().__init__()
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
        # PixelMAE's patch=3 projection, unchanged: 9 values and 9 observed
        # flags through one Linear (ml/model.py:660).
        self.val_proj = nn.Linear(18, d_model)
        # A dot is ONE cell: [value, observed]. Same shape of statement as the
        # patch token, one ninth of the input.
        self.dot_proj = nn.Linear(2, d_model)
        self.chan_emb = nn.Embedding(n_chan, d_model)
        self.coord = CoordEnc(d_model, n_fourier)
        self.mask_tok = nn.Parameter(torch.zeros(d_model))   # hidden by US
        self.miss_tok = nn.Parameter(torch.zeros(d_model))   # absent in the DATA
        self.cls_tok = nn.Parameter(torch.zeros(d_model))
        self.query_tok = nn.Parameter(torch.zeros(d_model))  # "this is a query"
        self.ctx_proj = nn.Linear(4, d_model)                # [sin m, cos m, lat, lon]

        # --- encoder --------------------------------------------------------
        self.latents = nn.Parameter(torch.zeros(n_latents, d_model))
        self.cross = CrossBlock(d_model, n_heads, mlp_mult=4)
        enc_layer = nn.TransformerEncoderLayer(
            d_model, n_heads, dim_feedforward=4 * d_model,
            batch_first=True, norm_first=True, dropout=0.0)
        # enable_nested_tensor=False: the latents are a fixed-length, fully
        # attended set (no padding), so the nested-tensor fast path can never
        # apply here and torch warns about it on every construction.
        self.encoder = nn.TransformerEncoder(enc_layer, n_layers,
                                             enable_nested_tensor=False)
        # Attention pool over the latents: one learned query, so the pool is a
        # measurement the model chooses rather than a mean over 64 slots that
        # were never asked to be commensurable.
        self.pool_q = nn.Parameter(torch.zeros(1, 1, d_model))
        self.pool = nn.MultiheadAttention(d_model, n_heads, batch_first=True,
                                          dropout=0.0)
        self.ln_pool = nn.LayerNorm(d_model)
        self.to_z = nn.Linear(d_model, d_z)

        # --- decoder --------------------------------------------------------
        # A query token is (channel, dy, dx, lag, depth) through the SAME
        # chan_emb and the SAME coordinate encoding the encoder reads, plus a
        # learned marker that says "this is a question, not a measurement".
        self.q_proj = nn.Linear(d_model, d_dec)
        self.z_proj = nn.Linear(d_z, d_dec)
        self.lat_proj = nn.Linear(d_model, d_dec)
        self.dec = nn.ModuleList([CrossBlock(d_dec, n_heads, mlp_mult=2)
                                  for _ in range(dec_layers)])
        self.ln_out = nn.LayerNorm(d_dec)
        self.head = nn.Linear(d_dec, 2)                      # (mu, logvar)

        for p in (self.mask_tok, self.miss_tok, self.cls_tok, self.query_tok):
            nn.init.normal_(p, std=0.02)
        nn.init.normal_(self.latents, std=0.02)
        nn.init.normal_(self.pool_q, std=0.02)

    # ------------------------------------------------------------- counting --
    def param_count(self):
        return int(sum(p.numel() for p in self.parameters()))

    # -------------------------------------------------------------- encoder --
    def tokens(self, b):
        """The token set and its key-padding mask.

        `b` is the sampler's batch as tensors (see ml/cone_sampler.py::sample),
        plus `chan_depth` [C] — the per-channel pressure in dbar from
        `cone.channel_depth_dbar`, which the sampler carries per DOT and the
        lag-0 patch tokens need per CHANNEL — plus two optional masks naming
        what WE hid:
          `chan_mask` [B, C] bool — this channel's lag-0 patch is hidden
          `dot_mask`  [B, N] bool — this dot is hidden

        Order is [cls, ctx, patch x C, dots x N]; z is read from the pool over
        the latents, not from a token, so nothing downstream depends on the
        position of any token but the reader of this docstring.
        """
        dt = self.val_proj.weight.dtype
        pv = b["patch_vals"].to(dt)                              # [B, C, 9]
        po = b["patch_obs"].to(dt)
        B, C, _ = pv.shape
        chan_mask = b.get("chan_mask")
        if chan_mask is None:
            chan_mask = torch.zeros(B, C, dtype=torch.bool, device=pv.device)

        # ---- lag-0 patch tokens, one per channel --------------------------
        # PixelMAE's `feat = cat([x*obs, obs], -1)`: an unobserved cell enters
        # as a zero VALUE and a zero FLAG, so "0.0 because absent" and "0.0
        # because that is the anomaly" are different inputs.
        feat = torch.cat([pv * po, po], dim=-1)                  # [B, C, 18]
        ci = torch.arange(C, device=pv.device)
        depth_c = b["chan_depth"].to(dt)                         # [C]
        zero = torch.zeros(B, C, dtype=dt, device=pv.device)
        base_p = (self.chan_emb(ci)[None].expand(B, -1, -1)
                  + self.coord(zero, zero, zero,
                               depth_c[None].expand(B, -1)))
        obs_c = b["patch_obs"][..., 4]                           # centre cell
        vis = (obs_c & ~chan_mask).unsqueeze(-1)
        hid = (obs_c & chan_mask).unsqueeze(-1)
        mis = (~obs_c).unsqueeze(-1)
        vt = self.val_proj(feat)
        t_patch = (torch.where(vis, vt, torch.zeros_like(vt))
                   + torch.where(hid, self.mask_tok.expand_as(vt),
                                 torch.zeros_like(vt))
                   + torch.where(mis, self.miss_tok.expand_as(vt),
                                 torch.zeros_like(vt))
                   + base_p)

        # ---- dot tokens ----------------------------------------------------
        dv = b["vals"].to(dt)                                    # [B, N]
        do = b["obs"].to(dt)
        N = dv.shape[1]
        dot_mask = b.get("dot_mask")
        if dot_mask is None:
            dot_mask = torch.zeros(B, N, dtype=torch.bool, device=dv.device)
        base_d = (self.chan_emb(b["chan"].long())
                  + self.coord(b["dy_km"].to(dt), b["dx_km"].to(dt),
                               b["lag_days"].to(dt), b["depth"].to(dt)))
        dfeat = torch.stack([dv * do, do], dim=-1)               # [B, N, 2]
        dvt = self.dot_proj(dfeat)
        obs_d = b["obs"]
        vis = (obs_d & ~dot_mask).unsqueeze(-1)
        hid = (obs_d & dot_mask).unsqueeze(-1)
        mis = (~obs_d).unsqueeze(-1)
        t_dot = (torch.where(vis, dvt, torch.zeros_like(dvt))
                 + torch.where(hid, self.mask_tok.expand_as(dvt),
                               torch.zeros_like(dvt))
                 + torch.where(mis, self.miss_tok.expand_as(dvt),
                               torch.zeros_like(dvt))
                 + base_d)

        toks = torch.cat([self.cls_tok.expand(B, 1, -1),
                          self.ctx_proj(b["ctx"].to(dt)).unsqueeze(1),
                          t_patch, t_dot], dim=1)
        # THE ONE PLACE EXISTENCE IS ENFORCED (see the module docstring).
        kpm = torch.cat([
            torch.zeros(B, 2 + C, dtype=torch.bool, device=dv.device),
            ~b["valid"]], dim=1)
        return toks, kpm

    def encode(self, b):
        """`(z [B, d_z], latents [B, n_latents, d_model])`."""
        toks, kpm = self.tokens(b)
        B = toks.shape[0]
        lat = self.latents[None].expand(B, -1, -1)
        lat = self.cross(lat, toks, key_padding_mask=kpm)
        lat = self.encoder(lat)
        q = self.pool_q.expand(B, -1, -1)
        h, _ = self.pool(q, self.ln_pool(lat), self.ln_pool(lat),
                         need_weights=False)
        return self.to_z(h[:, 0]), lat

    # -------------------------------------------------------------- decoder --
    def query_tokens(self, chan, dy_km, dx_km, lag_days, depth):
        """Build [B, Q, d_dec] decoder queries. The decoder never sees a value.

        `chan` [B, Q] long; the four coordinates [B, Q] float. A query at
        (c, 0, 0, 0, depth_c) is "the anchor's own channel c"; one at negative
        `lag_days` is a FUTURE query (t+1, t+2 pentads).
        """
        e = (self.chan_emb(chan.long())
             + self.coord(dy_km, dx_km, lag_days, depth)
             + self.query_tok)
        return self.q_proj(e)

    def _run_dec(self, mem, q):
        for blk in self.dec:
            q = blk(q, mem)
        out = self.head(self.ln_out(q))
        return out[..., 0], out[..., 1].clamp(LOGVAR_MIN, LOGVAR_MAX)

    def decode_from_z(self, z, queries):
        """(mu, logvar) reading `z` ALONE — the path a downstream user has.

        This is the head that makes `z` a bottleneck: everything the answer
        depends on has to have gone through d_z numbers. `ml/train_cone.py`
        puts the HEADLINE loss through here for exactly that reason.
        """
        return self._run_dec(self.z_proj(z).unsqueeze(1), queries)

    def decode(self, z, latents, queries):
        """(mu, logvar) reading [z-token] + the latents.

        WHY BOTH PATHS EXIST, and why this one is not the headline loss. The
        plan's decoder reads "a Linear(d_z -> d_dec)-projected memory made of
        [z-token] + latents", which gives the encoder a much richer gradient:
        the latents are n_latents x d_model, they carry the detail a 32-number
        code cannot, and a reconstruction term through them trains the token
        set and the cross-attention directly.

        It also names a degeneracy, and ml/CLAUDE.md §4.9b says a degeneracy
        you can NAME is one you close or measure, never one you rank as
        improbable: if this were the ONLY loss, `z` would be optional — the
        decoder could read everything it needs from the latents and let the
        bottleneck carry nothing, and H1's velocity probe (a ridge from `z`
        alone) would then measure noise while every loss curve looked healthy.
        That is the E-069 experiment voiding itself silently.

        So the training loss is `nll(decode_from_z) + aux_w * nll(decode)`:
        the gradient that MUST be satisfied goes through the bottleneck, and
        this path is an auxiliary term (default weight 0.25,
        `--aux-latent-w`). Set `--aux-latent-w 0` for the plan's z-only
        codec exactly.
        """
        mem = torch.cat([self.z_proj(z).unsqueeze(1),
                         self.lat_proj(latents)], dim=1)
        return self._run_dec(mem, queries)

    # -------------------------------------------------------------- masking --
    def _masks(self, b, plan):
        """(chan_mask [B, C], dot_mask [B, N]) for one batch, per plan.

        Four schemes, mixed per batch ELEMENT (plan §3): channel drop, lag-band
        drop, sector drop, and (always) the anchor reconstruction, which is not
        a mask at all — it is a query.
        """
        dev = b["vals"].device
        B, N = b["vals"].shape
        C = self.n_chan
        g = plan.get("generator")

        def rand(*shape):
            return torch.rand(*shape, device=dev, generator=g)

        # (i) channel drop — hide a whole channel, at lag 0 AND at every dot.
        # `cur_*` at 0.5 (H1's target: the codec must reconstruct the current
        # from the motion of everything else), the rest at 0.3.
        p = plan["chan_drop_p"].to(dev)[None].expand(B, -1)
        chan_mask = rand(B, C) < p

        dot_mask = chan_mask.gather(1, b["chan"].long())          # [B, N]

        # (ii) lag-band drop — hide every dot at lag <= l0 and predict them
        # from the older lags. Forecasting inside pretraining: this is the
        # only scheme that asks the codec to step time forward from the cone
        # it can still see.
        if plan.get("lag_band_p", 0.0) > 0.0:
            on = rand(B) < plan["lag_band_p"]
            l0 = torch.randint(1, 4, (B,), device=dev, generator=g)
            band = b["lag_days"] <= (5.0 * l0.to(b["lag_days"].dtype))[:, None]
            dot_mask = dot_mask | (band & on[:, None])

        # (iii) sector drop — hide a 90-degree bearing sector. The cone's whole
        # argument is that information arrives from a DIRECTION; hiding one
        # quadrant is the mask that makes the model interpolate across bearings
        # rather than across radii.
        if plan.get("sector_p", 0.0) > 0.0:
            on = rand(B) < plan["sector_p"]
            th0 = rand(B) * (2.0 * math.pi)
            th = torch.atan2(b["dx_km"], b["dy_km"])              # bearing
            d = torch.remainder(th - th0[:, None], 2.0 * math.pi)
            dot_mask = dot_mask | ((d < (math.pi / 2.0)) & on[:, None])

        return chan_mask, dot_mask

    # -------------------------------------------------------------- queries --
    def draw_dot_queries(self, b, plan, dot_mask):
        """`(idx [B, k] long, sel [B, k] bool)` — the hidden-dot query DRAW.

        Factored out of `_query_sets` so the randomness can be handed to a
        second framework (`ml/plans/E069_HANDOVER.md` §8.3): two RNGs cannot be
        made to agree, so the JAX port's parity gate passes the DRAW rather
        than the seed. The arithmetic is unchanged and the generator is
        consumed in exactly the same way — one `torch.rand(B, N)` — so
        `forward`, which still reaches this through `_query_sets`, keeps its
        numbers bit for bit (gate C9 asserts it).

        `k = min(n_dot_queries, N)`; an empty draw (`k == 0` or `N == 0`)
        returns EMPTY tensors rather than None, so the caller's gather path is
        one branch on a width instead of two on a type.
        """
        dev = b["vals"].device
        B, N = b["vals"].shape
        k = min(int(plan.get("n_dot_queries", 0)), int(N))
        if k <= 0 or N == 0:
            return (torch.zeros(B, 0, dtype=torch.long, device=dev),
                    torch.zeros(B, 0, dtype=torch.bool, device=dev))
        score = torch.where(dot_mask & b["obs"] & b["valid"],
                            torch.rand(B, N, device=dev,
                                       generator=plan.get("generator")),
                            torch.full((B, N), -1.0, device=dev))
        idx = score.topk(k, dim=1).indices                        # [B, k]
        sel = score.gather(1, idx) >= 0.0
        return idx, sel

    def _query_sets(self, b, plan, dot_mask, dot_idx=None):
        """Assemble (chan, dy, dx, lag, depth, target, weight) for every query.

        Three families of query, concatenated along Q so the decoder runs once:
          A. anchor reconstruction — the anchor's own value in every channel at
             lag 0, ALWAYS (plan §3(v)). This is the query the velocity probe
             later reads: `cur_u` at the anchor with `cur_*` dropped from the
             input is exactly H1.
          B. future — t+1 and t+2 pentads at the anchor column, at NEGATIVE
             lag_days, from the archive inside the training pool.
          C. hidden dots — a random subsample of the dots we masked, capped at
             `n_dot_queries` so the decoder's cost does not scale with N.

        Weights are the product of (is this a real target) and the channel's
        family weight. A target that was never observed has weight exactly 0
        and its `target` value is a zero that nothing is scored against.

        `dot_idx`, when given, is `draw_dot_queries`' `(idx, sel)` pair and is
        used INSTEAD of drawing — the deterministic path `forward_given` and
        the JAX twin both take. Left None (what `forward` passes) the draw
        happens here, at the same point in the generator's stream as it always
        has.
        """
        dev = b["vals"].device
        dt = b["vals"].dtype
        B, N = b["vals"].shape
        C = self.n_chan
        cw = plan["chan_w"].to(dev).to(dt)                        # [C]
        depth_c = b["chan_depth"].to(dt)

        chans, dys, dxs, lags, deps, tgts, ws = [], [], [], [], [], [], []

        # ---- A. anchor reconstruction -------------------------------------
        if plan.get("anchor_recon", True):
            ci = torch.arange(C, device=dev)[None].expand(B, -1)
            z = torch.zeros(B, C, dtype=dt, device=dev)
            chans.append(ci)
            dys.append(z), dxs.append(z), lags.append(z)
            deps.append(depth_c[None].expand(B, -1))
            tgts.append(b["patch_vals"][..., 4].to(dt))
            ws.append(b["patch_obs"][..., 4].to(dt) * cw[None])

        # ---- B. future queries --------------------------------------------
        fut = b.get("fut_vals")
        if plan.get("future", True) and fut is not None and fut.shape[-1]:
            F = fut.shape[-1]
            flags = plan["future_lags"].to(dev).to(dt)            # [F], pentads
            ci = torch.arange(C, device=dev)[None, :, None].expand(B, C, F)
            z = torch.zeros(B, C, F, dtype=dt, device=dev)
            chans.append(ci.reshape(B, C * F))
            dys.append(z.reshape(B, C * F))
            dxs.append(z.reshape(B, C * F))
            # NEGATIVE lag days: the future is the past's mirror on the one
            # axis the coordinate encoding already carries, so no new input.
            lags.append((-5.0 * flags)[None, None, :].expand(B, C, F)
                        .reshape(B, C * F))
            deps.append(depth_c[None, :, None].expand(B, C, F).reshape(B, C * F))
            tgts.append(fut.to(dt).reshape(B, C * F))
            ws.append((b["fut_obs"].to(dt)
                       * cw[None, :, None]).reshape(B, C * F))

        # ---- C. hidden dots ------------------------------------------------
        idx, sel = (self.draw_dot_queries(b, plan, dot_mask)
                    if dot_idx is None else dot_idx)
        if idx.shape[1] > 0:
            chans.append(b["chan"].long().gather(1, idx))
            dys.append(b["dy_km"].gather(1, idx).to(dt))
            dxs.append(b["dx_km"].gather(1, idx).to(dt))
            lags.append(b["lag_days"].gather(1, idx).to(dt))
            deps.append(b["depth"].gather(1, idx).to(dt))
            tgts.append(b["vals"].gather(1, idx).to(dt))
            ws.append(sel.to(dt) * cw[b["chan"].long().gather(1, idx)])

        return (torch.cat(chans, 1), torch.cat(dys, 1), torch.cat(dxs, 1),
                torch.cat(lags, 1), torch.cat(deps, 1), torch.cat(tgts, 1),
                torch.cat(ws, 1))

    # -------------------------------------------------------------- forward --
    def forward(self, b, plan):
        """One training step's loss. Returns `dict(loss=..., terms={...})`.

        `b` is the sampler's batch as tensors; `plan` is `default_plan()`'s
        dict (probabilities, per-channel weights, query budget, aux weight,
        and an optional torch.Generator so an EVAL pass can be reproducible).

        UNCHANGED by the E-069 JAX port, deliberately and checkably: the masks
        are drawn here and the hidden-dot queries are drawn where they always
        were — INSIDE `_query_sets`, after `encode` — so this method's
        generator consumption is byte for byte what every archived cone number
        was produced under. `forward_given` shares the body below rather than
        this method calling it, because a refactor that moved the draw would
        be a silent change to every seeded eval
        (`tests/test_jaxport_cone.py` gate C9 asserts the two agree).
        """
        chan_mask, dot_mask = self._masks(b, plan)
        return self._loss_from(b, plan, chan_mask, dot_mask, None)

    def forward_given(self, b, plan, chan_mask, dot_mask, dot_idx):
        """`forward`, with the randomness handed IN. Uses no RNG at all.

        `chan_mask` [B, C] and `dot_mask` [B, N] are `_masks`' output;
        `dot_idx` is `draw_dot_queries`' `(idx, sel)` pair. This is the entry
        point the JAX port mirrors (`ConeMAEJax.loss_given`) and the one a
        cross-framework gate can compare at all — two RNGs cannot be made to
        agree, so the draw is the thing that is shared, not the seed.

        Returns the same `dict(loss=..., z=..., terms={...})` `forward` does.
        """
        return self._loss_from(b, plan, chan_mask, dot_mask, dot_idx)

    def _loss_from(self, b, plan, chan_mask, dot_mask, dot_idx):
        """The shared body of `forward` and `forward_given`.

        `dot_idx=None` means "draw the hidden-dot queries inside
        `_query_sets`", which is `forward`'s original control flow; a given
        `(idx, sel)` pair makes the whole method deterministic.
        """
        bb = dict(b)
        bb["chan_mask"], bb["dot_mask"] = chan_mask, dot_mask
        z, lat = self.encode(bb)

        chan, dy, dx, lag, dep, tgt, w = self._query_sets(b, plan, dot_mask,
                                                          dot_idx)
        q = self.query_tokens(chan, dy, dx, lag, dep)
        mu, logvar = self.decode_from_z(z, q)

        wsum = w.sum().clamp(min=1e-6)
        nll = (nll_gauss(mu, logvar, tgt) * w).sum() / wsum
        # The plain MSE, LOGGED not optimised: it is the number every archived
        # reconstruction is quoted in, and a likelihood is not comparable with
        # one (ml/CLAUDE.md §4.3 — keep diagnostics out of the objective).
        mse = (((mu - tgt) ** 2) * w).sum() / wsum

        terms = {"nll": float(nll.detach()), "mse": float(mse.detach()),
                 # `wsum` is the DENOMINATOR of both means, so a caller
                 # averaging over several batches (the held-out eval) can
                 # weight them exactly rather than by a target count that is
                 # not what the mean divided by.
                 "wsum": float(wsum.detach()),
                 "n_targets": float((w > 0).sum().detach()),
                 "logvar_mean": float((logvar.detach() * w).sum() / wsum),
                 "frac_chan_masked": float(chan_mask.float().mean().detach()),
                 "frac_dot_masked": float(dot_mask.float().mean().detach())}

        loss = nll
        aux_w = float(plan.get("aux_latent_w", 0.0))
        if aux_w > 0.0:
            mu2, lv2 = self.decode(z, lat, q)
            aux = (nll_gauss(mu2, lv2, tgt) * w).sum() / wsum
            terms["nll_latent"] = float(aux.detach())
            loss = loss + aux_w * aux
        return {"loss": loss, "z": z, "terms": terms}


def default_plan(chan_names, cur_drop=0.5, other_drop=0.3, lag_band_p=0.3,
                 sector_p=0.3, future=True, anchor_recon=True,
                 n_dot_queries=256, aux_latent_w=0.25, future_lags=(1, 2),
                 generator=None, device=None):
    """The masking plan of plan §3, as a dict of plain tensors.

    `cur_*` at 0.5 and everything else at 0.3 is the plan's channel-drop
    schedule; the lag-band and sector drops fire on 30% of batch elements
    each, independently, so a given anchor can be hit by all three.
    """
    p = torch.tensor([cur_drop if n.startswith("cur_") else other_drop
                      for n in chan_names], dtype=torch.float32, device=device)
    return {
        "chan_drop_p": p,
        "chan_w": torch.tensor(family_weights(chan_names),
                               dtype=torch.float32, device=device),
        "lag_band_p": float(lag_band_p),
        "sector_p": float(sector_p),
        "future": bool(future),
        "anchor_recon": bool(anchor_recon),
        "n_dot_queries": int(n_dot_queries),
        "aux_latent_w": float(aux_latent_w),
        "future_lags": torch.tensor(list(future_lags), dtype=torch.float32,
                                    device=device),
        "generator": generator,
    }

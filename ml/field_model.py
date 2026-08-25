#!/usr/bin/env python3
"""E-052 · the field head: joint next-latent-field prediction, det and diffusion.

`ml/plans/E052_field_diffusion.md` is the spec; this file is E-052.1's and
E-052.2's shared implementation. Stage 2 today (`ml/temporal.py`) predicts each
ocean pixel's next embedding from K past frames of ITSELF plus a 144-slot
stencil, concatenated per frame by one fixed `Linear`, with attention over time
only — so the t+1 field is P conditionally-independent point estimates. This
module is the joint alternative: one model that emits the WHOLE next field at
once, first deterministically (axis A), then as a sample from the conditional
distribution (axis B, EDM diffusion).

Four decisions carry the whole design, and each is load-bearing for a test:

1. **The modelled quantity is the RESIDUAL** r = z_{t+1} - z_t, never the field
   itself. Persistence is then the exact zero-prior: a network whose output is
   identically zero predicts persistence exactly, and the plan's read-out
   ("ratio 1.000000 at step 0") is an identity rather than a threshold.
2. **The final layer is ZERO-INITIALIZED**, as is every adaLN modulation
   `Linear`. So at init `FieldDiT` returns exactly 0.0 — bitwise — which makes
   (1) true bitwise, and in `diff` mode makes D(x; sigma) == c_skip(sigma)*x
   bitwise for every sigma. Both are tested as exact identities (ml/CLAUDE.md
   §4.9: prefer an exact expected value to a threshold).
3. **Time is factorized, space is joint.** `TemporalCond` runs over TIME ONLY,
   per token, with no cross-token mixing; `FieldDiT` attends over SPACE with no
   time axis at all. That is the ablation the plan asks for: axis A is "joint
   spatial attention vs per-pixel stencil-concat", and mixing space into the
   conditioner too would buy the answer with a confound. It is also what keeps
   the conditioner's cost linear in Ntok rather than quadratic.
4. **Every random number comes from a passed `torch.Generator`.** No global RNG
   is touched anywhere in this file — the sampler's seeded-determinism test and
   the trainer's resume-is-bit-identical test both depend on it.

Data convention, matching the embed-cache artefact stage 2 already publishes:
`Z` is `[T, P, d_z]` float32 for the P ocean pixels at grid coords `(ys, xs)`
on an H x W grid.

CPU-friendly: nothing here assumes CUDA.
"""
import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# tokenizer
# ---------------------------------------------------------------------------
class OceanTokenizer:
    """Pixel <-> patch-token scatter/gather with an ocean-flag channel.

    Each ocean pixel belongs to the patch cell `(y // patch, x // patch)`.
    The TOKENS are exactly those cells containing at least one ocean pixel,
    ordered by `(py, px)` ascending — a fixed deterministic order that depends
    only on the mask, never on the order `ys`/`xs` arrive in.

    A token's feature vector is `patch*patch` slots of `d_z + 1` channels: the
    pixel's d_z latent values, then a 1.0 ocean flag. Land and out-of-grid
    slots are zero-filled with flag 0.0, which is what lets the model tell
    "this slot is a zero-valued ocean pixel" from "this slot is not ocean" —
    without the flag those are the same vector, and a coastal token would be
    indistinguishable from an open-ocean one whose values happened to be small.

    `to_pixels(to_tokens(z))` is the identity BITWISE: it is a zero-fill, a
    copy and a gather, with no arithmetic anywhere on the path.
    """

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
        # np.unique returns ASCENDING, and cell = py*Wp + px is monotone in
        # (py, px) lexicographically — so this IS "sorted by (py, px)".
        uniq = np.unique(cell)
        self.ntok = int(uniq.size)
        tok_of_cell = np.full(self.Hp * self.Wp, -1, np.int64)
        tok_of_cell[uniq] = np.arange(self.ntok, dtype=np.int64)

        tok_of_px = tok_of_cell[cell]
        slot_of_px = (ys % self.patch) * self.patch + (xs % self.patch)
        # ONE index does both directions: the token buffer is viewed as
        # [.., ntok*P2, C], so pixel p lives at row `flat_of_px[p]`.
        self.flat_of_px = torch.from_numpy(
            (tok_of_px * self.P2 + slot_of_px).astype(np.int64))
        self.tok_of_px = torch.from_numpy(tok_of_px)
        self.slot_of_px = torch.from_numpy(slot_of_px)
        self.ys = torch.from_numpy(ys)
        self.xs = torch.from_numpy(xs)
        # Linear index into the FULL HxW grid — the trainer and the masked-loss
        # test use it to slice ocean pixels out of a whole-grid field.
        self.ocean_lin = torch.from_numpy(ys * self.W + xs)

        self.tok_py = torch.from_numpy((uniq // self.Wp).astype(np.int64))
        self.tok_px = torch.from_numpy((uniq % self.Wp).astype(np.int64))
        # Cell CENTRES in [0, 1]: (i + 0.5) / n, not i / (n - 1), so a
        # single-row grid does not divide by zero and the coords mean the same
        # thing whatever the grid size.
        self.tok_coord = torch.stack([
            (self.tok_py.double() + 0.5) / self.Hp,
            (self.tok_px.double() + 0.5) / self.Wp], dim=1).float()

    def feat_in(self, d_z):
        """Token width of the CONTENT stream: patch*patch*(d_z + 1)."""
        return self.P2 * (int(d_z) + 1)

    def feat_out(self, d_z):
        """Token width of the OUTPUT stream: patch*patch*d_z (no flag)."""
        return self.P2 * int(d_z)

    def to_tokens(self, z):
        """[.., P, d_z] -> [.., ntok, patch*patch*(d_z+1)]. Leading dims free."""
        if z.shape[-2] != self.P:
            raise ValueError(f"expected P={self.P} pixels, got {z.shape[-2]}")
        d_z = z.shape[-1]
        lead = z.shape[:-2]
        idx = self.flat_of_px.to(z.device)
        buf = torch.zeros(*lead, self.ntok * self.P2, d_z + 1,
                          dtype=z.dtype, device=z.device)
        buf[..., idx, :d_z] = z
        buf[..., idx, d_z] = 1.0
        return buf.reshape(*lead, self.ntok, self.P2 * (d_z + 1))

    def to_pixels(self, tok, d_z=None):
        """[.., ntok, patch*patch*C] -> [.., P, d_z]. Ocean slots only.

        BOTH token layouts are accepted, which is what makes the round trip
        `to_pixels(to_tokens(z)) == z` expressible at all: C == d_z is the
        backbone's OUTPUT stream (no flag), C == d_z + 1 is the CONTENT stream
        and its trailing ocean-flag channel is dropped here. With `d_z`
        omitted the output layout is assumed, because that is the call the
        model makes on every forward and an ambiguous default there would be a
        silent off-by-one channel.
        """
        if tok.shape[-1] % self.P2:
            raise ValueError("token width is not a multiple of patch^2")
        C = tok.shape[-1] // self.P2
        d_z = C if d_z is None else int(d_z)
        if C not in (d_z, d_z + 1):
            raise ValueError(f"token width implies {C} channels/slot, which is "
                             f"neither d_z={d_z} nor d_z+1")
        lead = tok.shape[:-2]
        idx = self.flat_of_px.to(tok.device)
        flat = tok.reshape(*lead, self.ntok * self.P2, C)
        return flat[..., idx, :d_z]


# ---------------------------------------------------------------------------
# temporal conditioner
# ---------------------------------------------------------------------------
class TemporalCond(nn.Module):
    """Per-token encoder over K frames. TIME ONLY — no cross-token mixing.

    The plan's axis-A ablation is "joint spatial attention vs the per-pixel
    stencil concat". Letting the conditioner mix tokens too would answer that
    question before the backbone is reached, so the factorization is by design
    and not an economy: space is `FieldDiT`'s job, exclusively.

    Season features (sin/cos of the year phase, zeros for the toys) are
    CONCATENATED to each frame's token content rather than added after the
    projection — at K=8 and d_c=128 that is two extra input columns against a
    separate projection's worth of parameters, and it keeps a frame's whole
    description in one tensor.
    """

    def __init__(self, feat_in, d_cond, K, layers=2, heads=4, season_dim=2):
        super().__init__()
        self.K, self.d_cond, self.season_dim = int(K), int(d_cond), int(season_dim)
        self.proj = nn.Linear(int(feat_in) + self.season_dim, self.d_cond)
        self.tpos = nn.Parameter(torch.zeros(self.K, self.d_cond))
        nn.init.normal_(self.tpos, std=0.02)
        layer = nn.TransformerEncoderLayer(
            d_model=self.d_cond, nhead=int(heads),
            dim_feedforward=4 * self.d_cond, dropout=0.0,
            activation="gelu", batch_first=True, norm_first=True)
        # enable_nested_tensor is a fast path for PADDED batches; every
        # sequence here is exactly K long, so it can never apply — saying so
        # explicitly keeps torch from warning about it on every construction.
        self.enc = nn.TransformerEncoder(layer, num_layers=int(layers),
                                         enable_nested_tensor=False)
        self.out_norm = nn.LayerNorm(self.d_cond)

    def forward(self, ctx_tokens, season=None):
        """[B, K, ntok, feat_in] (+ [B, K, 2]) -> [B, ntok, d_cond]."""
        B, K, N, Fi = ctx_tokens.shape
        if K != self.K:
            raise ValueError(f"TemporalCond built for K={self.K}, got {K}")
        if season is None:
            season = ctx_tokens.new_zeros(B, K, self.season_dim)
        x = torch.cat([ctx_tokens,
                       season[:, :, None, :].expand(B, K, N, self.season_dim)],
                      dim=-1)
        h = self.proj(x) + self.tpos[None, :, None, :]
        # Fold tokens into the batch: [B, K, N, d] -> [B*N, K, d]. The encoder
        # then sees BxN independent length-K sequences and cannot mix tokens
        # even by accident.
        h = h.permute(0, 2, 1, 3).reshape(B * N, K, self.d_cond)
        mask = torch.triu(torch.ones(K, K, dtype=torch.bool,
                                     device=h.device), diagonal=1)
        h = self.enc(h, mask=mask, is_causal=False)
        h = self.out_norm(h[:, -1])                       # last step only
        return h.reshape(B, N, self.d_cond)


# ---------------------------------------------------------------------------
# DiT backbone
# ---------------------------------------------------------------------------
def _zero_linear(in_f, out_f):
    """A `Linear` that outputs exactly 0.0 for any finite input.

    Both weight AND bias are zeroed: `0*x + 0` is bitwise zero, which is what
    makes "det mode at init IS persistence" an identity rather than an
    approximation. A non-zero bias here would break the test and the test
    would be right — fix the model, not the tolerance.
    """
    lin = nn.Linear(in_f, out_f)
    nn.init.zeros_(lin.weight)
    nn.init.zeros_(lin.bias)
    return lin


def _modulate(x, shift, scale):
    return x * (1.0 + scale[:, None, :]) + shift[:, None, :]


class _Block(nn.Module):
    """LN -> full self-attention over tokens -> LN -> MLP, adaLN-zero on g.

    With the modulation `Linear` zero-initialized, both gates are 0 at init and
    the block is EXACTLY the identity — so the whole stack is transparent and
    the zero-init final layer alone decides the network's output.
    """

    def __init__(self, d_model, heads, d_g, mlp_ratio=4):
        super().__init__()
        self.n1 = nn.LayerNorm(d_model, elementwise_affine=False, eps=1e-6)
        self.attn = nn.MultiheadAttention(d_model, heads, batch_first=True)
        self.n2 = nn.LayerNorm(d_model, elementwise_affine=False, eps=1e-6)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, mlp_ratio * d_model), nn.GELU(),
            nn.Linear(mlp_ratio * d_model, d_model))
        self.ada = nn.Sequential(nn.SiLU(), _zero_linear(d_g, 6 * d_model))

    def forward(self, x, g):
        s1, c1, g1, s2, c2, g2 = self.ada(g).chunk(6, dim=-1)
        h = _modulate(self.n1(x), s1, c1)
        a, _ = self.attn(h, h, h, need_weights=False)
        x = x + g1[:, None, :] * a
        h = _modulate(self.n2(x), s2, c2)
        return x + g2[:, None, :] * self.mlp(h)


class FieldDiT(nn.Module):
    """DiT over ocean-patch tokens: attention over SPACE, conditioned on g.

    Three streams meet at the token embedding:
      * the TARGET stream `x_tokens` — the noisy residual field in `diff` mode,
        literal zeros in `det` mode (the network has no target to refine there);
      * per-token CONDITIONING from `TemporalCond`, projected to d_model;
      * a learned 2-D positional embedding, built as a row table plus a column
        table rather than one table over token index. A per-token table would
        learn every cell's position independently; the factored one shares the
        "which row" and "which column" structure across the grid, which is the
        structure the field actually has.
    """

    def __init__(self, feat_in, feat_out, d_model, layers, heads, d_cond, d_g,
                 Hp, Wp, tok_py, tok_px):
        super().__init__()
        if d_model % 2:
            raise ValueError("d_model must be even (row/col position halves)")
        self.d_model = int(d_model)
        self.x_embed = nn.Linear(int(feat_in), self.d_model)
        self.c_embed = nn.Linear(int(d_cond), self.d_model)
        half = self.d_model // 2
        self.pos_row = nn.Parameter(torch.zeros(int(Hp), half))
        self.pos_col = nn.Parameter(torch.zeros(int(Wp), half))
        nn.init.normal_(self.pos_row, std=0.02)
        nn.init.normal_(self.pos_col, std=0.02)
        self.register_buffer("tok_py", tok_py.clone(), persistent=True)
        self.register_buffer("tok_px", tok_px.clone(), persistent=True)
        self.blocks = nn.ModuleList([
            _Block(self.d_model, int(heads), int(d_g)) for _ in range(int(layers))])
        self.fin_norm = nn.LayerNorm(self.d_model, elementwise_affine=False,
                                     eps=1e-6)
        self.fin_ada = nn.Sequential(nn.SiLU(),
                                     _zero_linear(int(d_g), 2 * self.d_model))
        self.fin = _zero_linear(self.d_model, int(feat_out))

    def pos_embed(self):
        return torch.cat([self.pos_row[self.tok_py],
                          self.pos_col[self.tok_px]], dim=-1)[None]

    def forward(self, x_tokens, cond_tokens, g):
        x = self.x_embed(x_tokens) + self.c_embed(cond_tokens) + self.pos_embed()
        for blk in self.blocks:
            x = blk(x, g)
        s, c = self.fin_ada(g).chunk(2, dim=-1)
        return self.fin(_modulate(self.fin_norm(x), s, c))


# ---------------------------------------------------------------------------
# the full head
# ---------------------------------------------------------------------------
class FieldHead(nn.Module):
    """Tokenizer + conditioner + backbone + (in `diff` mode) EDM and a sampler.

    The modelled quantity is the residual r = z_{t+1} - z_t. `det` mode
    regresses it; `diff` mode learns its conditional distribution with the EDM
    preconditioning of Karras et al. 2022, on the residual PIXEL field but with
    every network call routed through token space.
    """

    N_FOURIER = 8            # log-spaced frequencies for the sigma embedding

    def __init__(self, tok, d_z, K, mode="det", d_model=128, layers=4, heads=4,
                 d_cond=128, cond_layers=2, cond_heads=4, sigma_data=1.0,
                 season_dim=2):
        super().__init__()
        if mode not in ("det", "diff"):
            raise ValueError("mode must be 'det' or 'diff'")
        self.tok, self.mode = tok, mode
        self.d_z, self.K = int(d_z), int(K)
        self.d_model = int(d_model)
        feat_in = tok.feat_in(self.d_z)
        feat_out = tok.feat_out(self.d_z)
        self.cond = TemporalCond(feat_in, int(d_cond), self.K, layers=cond_layers,
                                 heads=cond_heads, season_dim=season_dim)
        self.dit = FieldDiT(feat_in, feat_out, self.d_model, layers, heads,
                            int(d_cond), self.d_model, tok.Hp, tok.Wp,
                            tok.tok_py, tok.tok_px)
        # sigma_data is a MEASURED property of the data (the residual RMS), so
        # it rides as a buffer: it must travel with the checkpoint and must
        # never be learned (ml/CLAUDE.md §4.2 — normalise by properties of the
        # DATA, never of the model).
        self.register_buffer("sigma_data",
                             torch.tensor(float(sigma_data)), persistent=True)
        self.register_buffer(
            "fourier_f",
            torch.exp(torch.linspace(0.0, math.log(1000.0), self.N_FOURIER)),
            persistent=True)
        self.sig_mlp = nn.Sequential(
            nn.Linear(2 * self.N_FOURIER, self.d_model), nn.SiLU(),
            nn.Linear(self.d_model, self.d_model))
        # `det` has no noise level, so it gets a learned NULL conditioning
        # token instead — one vector, the same for every batch element.
        self.g_null = nn.Parameter(torch.zeros(self.d_model))

    # -- conditioning --------------------------------------------------------
    def make_cond(self, ctx_tokens, season=None):
        """[B, K, ntok, feat_in] -> [B, ntok, d_cond]. Built ONCE per batch."""
        return self.cond(ctx_tokens, season)

    def g_of_sigma(self, sigma):
        """[B] sigma -> [B, d_model] global conditioning, via EDM's c_noise."""
        c_noise = torch.log(sigma) / 4.0
        ang = 2.0 * math.pi * c_noise[:, None] * self.fourier_f[None, :]
        return self.sig_mlp(torch.cat([torch.sin(ang), torch.cos(ang)], dim=-1))

    def g_null_for(self, B, device=None, dtype=None):
        g = self.g_null[None].expand(B, self.d_model)
        return g if device is None else g.to(device=device, dtype=dtype)

    # -- deterministic mode --------------------------------------------------
    def residual_det(self, cond_tokens):
        """r_hat [B, P, d_z]. EXACTLY zero at init -> persistence."""
        B = cond_tokens.shape[0]
        x = cond_tokens.new_zeros(B, self.tok.ntok, self.tok.feat_in(self.d_z))
        out = self.dit(x, cond_tokens, self.g_null_for(B).to(cond_tokens.dtype))
        return self.tok.to_pixels(out, self.d_z)

    def forward_det(self, cond_tokens, z_t):
        """z_hat = z_t + r_hat. Bitwise == z_t at init."""
        return z_t + self.residual_det(cond_tokens)

    def forward(self, ctx_tokens, z_t, season=None):
        """Convenience det forward from raw context tokens."""
        return self.forward_det(self.make_cond(ctx_tokens, season), z_t)

    # -- EDM preconditioning -------------------------------------------------
    def _coefs(self, sigma):
        sd = self.sigma_data
        s2 = sigma * sigma
        c_skip = sd * sd / (s2 + sd * sd)
        c_out = sigma * sd / torch.sqrt(s2 + sd * sd)
        c_in = 1.0 / torch.sqrt(s2 + sd * sd)
        return c_skip, c_out, c_in

    def D(self, x, sigma, cond_tokens):
        """EDM denoiser on the residual field. D == c_skip*x BITWISE at init.

        `x` is [B, P, d_z] (a noisy residual), `sigma` is [B]. The network runs
        in token space; `c_in`/`c_out` scale the pixel field on either side of
        it, exactly as in Karras et al. 2022 eq. 7.
        """
        c_skip, c_out, c_in = self._coefs(sigma)
        b = (slice(None),) + (None,) * (x.dim() - 1)
        tokens = self.tok.to_tokens(c_in[b] * x)
        out = self.dit(tokens, cond_tokens, self.g_of_sigma(sigma))
        return c_skip[b] * x + c_out[b] * self.tok.to_pixels(out, self.d_z)

    def edm_loss(self, cond_tokens, z_t, z_tp1, generator):
        """EDM's weighted denoising loss on the residual, over ocean pixels.

        sigma ~ lognormal. EDM's published P_mean = -1.2 / P_std = 1.2 assume
        sigma_data = 0.5; this programme's residual RMS is measured per dataset
        and is not 0.5, so the ladder is CENTRED on log(sigma_data) and keeps
        the same width — a sigma ladder in absolute units would put most of its
        mass where the signal has already vanished (or never arrived) whenever
        the data's scale differs from EDM's.

        Every random draw comes from `generator`; nothing here touches the
        global RNG.
        """
        B = z_t.shape[0]
        sd = float(self.sigma_data)
        p_mean = math.log(sd) - 1.2
        p_std = 1.2
        n = torch.randn(B, generator=generator, device=z_t.device,
                        dtype=z_t.dtype)
        sigma = torch.exp(p_mean + p_std * n)
        r = z_tp1 - z_t
        noise = torch.randn(r.shape, generator=generator, device=r.device,
                            dtype=r.dtype)
        b = (slice(None),) + (None,) * (r.dim() - 1)
        d = self.D(r + sigma[b] * noise, sigma, cond_tokens)
        lam = (sigma ** 2 + sd ** 2) / (sigma * sd) ** 2
        per = ((d - r) ** 2).flatten(1).mean(dim=1)
        return (lam * per).mean()

    # -- sampler -------------------------------------------------------------
    def sigma_ladder(self, n_steps, sigma_min=None, sigma_max=None, rho=7.0,
                     device=None, dtype=torch.float32):
        """Karras rho-ladder, with a trailing exact 0.

        EDM's published 0.002 / 80 are quoted at sigma_data = 0.5, i.e. they are
        0.004 and 160 in units of sigma_data. The defaults below carry that
        ratio to whatever residual RMS the data actually has, for the same
        reason `edm_loss` re-centres P_mean: the ladder has to span the range
        over which the signal goes from visible to drowned, and that range is a
        property of the data's scale.
        """
        sd = float(self.sigma_data)
        smin = sd * 4e-3 if sigma_min is None else float(sigma_min)
        smax = sd * 160.0 if sigma_max is None else float(sigma_max)
        n = int(n_steps)
        i = torch.arange(n, device=device, dtype=torch.float64)
        if n == 1:
            t = torch.zeros(1, device=device, dtype=torch.float64)
        else:
            t = i / (n - 1)
        a, b = smax ** (1.0 / rho), smin ** (1.0 / rho)
        s = (a + t * (b - a)) ** rho
        return torch.cat([s, torch.zeros(1, device=device,
                                         dtype=torch.float64)]).to(dtype)

    @staticmethod
    def member_generator(seed, m, device="cpu"):
        """Member m's generator. DERIVED, so joint and separate calls agree.

        `sample(seed=s, M=k)` gives member m exactly what `sample(seed=s+m,
        M=1)` gives its only member, because both build `Generator(seed=s+m)`
        and consume it in the same order. That equality is the sampler's
        determinism test, and it is why members are drawn one at a time rather
        than as one [M*B, ...] batch — a batched draw would make member m
        depend on M.
        """
        g = torch.Generator(device=device)
        g.manual_seed(int(seed) + int(m))
        return g

    @torch.no_grad()
    def sample(self, cond_tokens, z_t, n_steps, seed=None, generator=None,
               M=1, sigma_min=None, sigma_max=None, rho=7.0):
        """Deterministic 2nd-order Heun on the probability-flow ODE.

        dx/dsigma = (x - D(x; sigma)) / sigma, integrated from sigma_max down to
        0 on the Karras ladder. No churn, no stochastic term: the only
        randomness is the initial draw, so a member is a deterministic function
        of its seed.

        Returns z_t + r, shaped [M, B, P, d_z].
        """
        if seed is None and generator is None:
            raise ValueError("sample() needs a seed or a generator — the "
                             "global RNG is never used here")
        if generator is not None and M != 1:
            raise ValueError("pass a seed (not a generator) for M > 1, so each "
                             "member's generator can be derived reproducibly")
        B = z_t.shape[0]
        dev, dt_ = z_t.device, z_t.dtype
        sig = self.sigma_ladder(n_steps, sigma_min, sigma_max, rho,
                                device=dev, dtype=dt_)
        outs = []
        for m in range(int(M)):
            g = generator if generator is not None else \
                self.member_generator(seed, m, device=dev.type)
            x = torch.randn(z_t.shape, generator=g, device=dev, dtype=dt_) * sig[0]
            for i in range(len(sig) - 1):
                s_i, s_n = sig[i], sig[i + 1]
                sv = s_i.expand(B)
                d = (x - self.D(x, sv, cond_tokens)) / s_i
                x_next = x + (s_n - s_i) * d
                if float(s_n) > 0.0:
                    sv2 = s_n.expand(B)
                    d2 = (x_next - self.D(x_next, sv2, cond_tokens)) / s_n
                    x_next = x + (s_n - s_i) * 0.5 * (d + d2)
                x = x_next
            outs.append(z_t + x)
        return torch.stack(outs, dim=0)


def nfe_to_steps(nfe):
    """Heun costs 2 evaluations per step except the last (sigma=0 is skipped).

    So N steps cost 2N-1 evaluations, and a budget of `nfe` buys
    `ceil((nfe+1)/2)` steps. Reported alongside the number actually spent so a
    read-out never claims a budget it did not use.
    """
    n = max(1, (int(nfe) + 1) // 2)
    return n, 2 * n - 1


def count_params(model):
    return sum(p.numel() for p in model.parameters())

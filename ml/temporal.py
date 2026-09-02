#!/usr/bin/env python3
"""Stage 2: a temporal transformer over frozen codec embeddings.

The K-sweep (probe_sequence.py, protocol v2) established the precondition:
ANOMALY-space embeddings gain probe skill as history is concatenated
(state-space embeddings lost it — they were seasonally redundant). A linear
read-out of stacked embeddings is the crudest possible sequence model; this
file is the honest next rung — a small causal transformer over each pixel's
embedding sequence z_{t-K+1..t}, with two jobs:

  1. DYNAMICS: predict z_{t+1} (the next month's anomaly embedding).
     Channel-space score: decode ẑ_{t+1} through the FROZEN codec decoder
     at offset 0 and compare against the true next-month channels, vs the
     persistence forecast x_{t+1} := x_t. Same blocked holdout as training.
  2. STATE: the transformer's last hidden state at the 26.5°N section,
     pooled along the section, replaces the concatenated-z features in the
     RAPID probe — same seasonality-proof protocol (deseasonalised target,
     train-years climatology, seasonal-only floor, lambda on a train tail).

Both stages stay in ANOMALY space; the codec is never fine-tuned (two-stage
by construction, so codec improvements and dynamics improvements stay
attributable). Splits are inherited from the codec checkpoint — the same
held-out years and the same mid-Atlantic longitude block, never random.

Usage:  python3 ml/temporal.py --run pilot4_anom --steps 4000
"""
import argparse
import copy
import datetime as dt
import json
import math
import os
import re
import sys
import time

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
# InputQuant lives in model.py — E-046 gave it a second caller
# (ml/model.py:PixelMAE's own FSQ bottleneck), and model.py is the
# module both sides import. Re-exported from here unchanged, so
# `from temporal import InputQuant` (ml/rollout_spatial.py) and
# `tp.InputQuant` (tests) resolve exactly as they did.
from model import PixelMAE, codec_from_ckpt, InputQuant
from probe_sequence import ridge_r

HERE = os.path.dirname(os.path.abspath(__file__))
# Box-persistent mirror, same directory train.py uses for codecs.
# Box-persistent mirror, same directory train.py uses for codecs. The
# override exists so tests/test_resume_temporal.py and the toy end-to-end run
# can exercise the real save/resume path without a Vast box and without
# writing anywhere real.
CKPT_DIR = os.environ.get("CKPT_DIR_OVERRIDE", "/opt/earth-cache/ckpt")


class _CondLayer(nn.TransformerEncoderLayer):
    """E-057: a stock norm_first encoder layer whose two LayerNorms are
    FiLM-modulated by a per-sample conditioning vector c (adaLN-zero, FGN
    arXiv:2506.10772 §3).

    Two properties are load-bearing and neither is a matter of taste:

    * **The state-dict keys are the stock layer's.** This class ADDS `film`
      and changes nothing else, so a legacy deterministic checkpoint's
      `encoder.layers.N.self_attn.*` / `norm1` / `linear1` … drop straight in
      under `load_state_dict(..., strict=False)` and the only missing keys are
      the ε path's. That is what lets an FGN arm warm-start a trunk that cost
      a day of GPU rather than re-learning it.
    * **`film` is ZERO-INITIALISED, weight and bias.** So at init s1=b1=s2=b2=0
      and the arithmetic below reduces to `norm1(x) * 1 + 0`, i.e. the stock
      norm_first forward EXACTLY — multiplying a float by 1.0 and adding 0.0
      are both exact. At step 0 the ε-conditioned model IS the deterministic
      incumbent, bitwise, whatever ε it is handed (the E-057 twin of "r_fore
      reads exactly 1.000000 at step 1"), and `tests/test_fgn_head.py` asserts
      it with `torch.equal` rather than a tolerance.

    The explicit math also means this layer never takes the fused
    `torch._transformer_encoder_layer_fwd` path the stock layer takes in
    eval() under `no_grad`. That is a deliberate, measured cost, not an
    oversight — see the eval-mode measurement in tests/test_fgn_head.py.
    """

    def __init__(self, *args, **kw):
        super().__init__(*args, **kw)
        d = self.linear1.in_features
        self.film = nn.Linear(d, 4 * d)
        nn.init.zeros_(self.film.weight)
        nn.init.zeros_(self.film.bias)

    def forward(self, src, c, src_mask=None, is_causal=False):
        """src [B,K,d_model] · c [B,d_model] -> [B,K,d_model]."""
        s1, b1, s2, b2 = self.film(c).chunk(4, -1)     # each [B, d_model]
        x = src
        x = x + self._sa_block(self.norm1(x) * (1 + s1[:, None]) + b1[:, None],
                               src_mask, None, is_causal=is_causal)
        x = x + self._ff_block(self.norm2(x) * (1 + s2[:, None]) + b2[:, None])
        return x


class _CondEncoder(nn.Module):
    """The thin container that gives `_CondLayer` the stock encoder's key
    layout (`encoder.layers.N.*`). Deliberately NOT a subclass of
    nn.TransformerEncoder: that class's forward owns a nested-tensor fast path
    and a mask-canonicalisation dance whose contract is `layer(src, src_mask,
    src_key_padding_mask, is_causal)`, and the conditioning vector has nowhere
    to ride in it. Cloning by deepcopy is what nn.TransformerEncoder itself
    does (`_get_clones`), so the two construct the same initial weights.
    """

    def __init__(self, layer, n_layers):
        super().__init__()
        self.layers = nn.ModuleList(
            [copy.deepcopy(layer) for _ in range(n_layers)])

    def forward(self, x, c, mask=None, is_causal=False):
        for mod in self.layers:
            x = mod(x, c, src_mask=mask, is_causal=is_causal)
        return x


def fair_crps2(x1, x2, y):
    """The FAIR CRPS estimator at M=2, in torch, elementwise then meaned.

        fair_crps2 = mean[ 0.5(|x1-y| + |x2-y|) - 0.5|x1-x2| ]

    The second term's divisor in the general estimator is 2*M*(M-1) = 4 over
    the two ordered pairs (i,j) and (j,i), each contributing |x1-x2| — hence
    |x1-x2|/2. This is a torch MIRROR of `ml/probscore.crps_ensemble(fair=True)`
    at M=2 and is pinned against it numerically in tests/test_fgn_head.py; the
    scoreboard is the definition and this is the transcription, never the other
    way round.

    Why the training objective must be this and not MSE: under a squared-error
    loss the conditional MEAN is optimal, so a noise-conditioned head learns to
    IGNORE ε. Noise conditioning and the proper score are one change, not two
    (ml/plans/E057_fgn_head.md, "two technical facts").
    """
    return fair_crps2_elem(x1, x2, y).mean()


def fair_crps2_elem(x1, x2, y):
    """`fair_crps2` BEFORE its reduction: the estimator's elementwise value,
    same shape as its arguments.

    The split exists so `--holdout-scope target` can mask the FGN objective
    EXACTLY. The fair-CRPS-at-2 estimator is a sum of three elementwise terms
    reduced ONCE at the end, so the score of a subset of elements is the mean
    of this tensor over that subset — masking here is the same arithmetic the
    unmasked call makes, not an approximation of it. Masking a term that had
    already been reduced (e.g. scaling `fair_crps2`'s scalar) would not be.

    `fair_crps2` is this expression followed by `.mean()`, so the legacy
    number is bit-identical: one elementwise combination, one reduction, in
    the association `ml/jaxport/train_stage2.py:fair_crps2` transcribes.
    """
    return (0.5 * ((x1 - y).abs() + (x2 - y).abs())
            - 0.5 * (x1 - x2).abs())


def fair_crps_ens(ens, obs):
    """Fair CRPS of an M-member ensemble `ens` [M, ...] against `obs` [...].

    The same estimator as `fair_crps2`, generalised, and computed through the
    sorted-member identity probscore uses

        Σ_{i,j} |x_i - x_j| = 2 Σ_k (2k - M + 1) x_(k)      (x_(k) ascending)

    so the read-out cost is O(M log M) rather than the O(M²) pairwise tensor —
    at M=64 over 4,096 monitoring windows the pairwise form is 4 GB.

    **M = 1 is MAE, exactly**, not approximately: the fair divisor M(M-1) is
    zero there, so the pair term is dropped and only mean|x-y| remains. That is
    the property that lets a deterministic head enter the same scoreboard as a
    degenerate one-member ensemble with no special case anywhere.

    No NaN handling — this scores dense z-space monitoring tensors, where a
    non-finite value is an event to report, not a cell to skip. `probscore` is
    the NaN-aware version and is what the offline scoreboard calls.
    """
    M = ens.shape[0]
    term1 = (ens - obs).abs().mean(0)
    if M < 2:
        return term1.mean()
    xs, _ = torch.sort(ens, dim=0)
    k = torch.arange(M, dtype=ens.dtype, device=ens.device).reshape(
        (-1,) + (1,) * (ens.dim() - 1))
    w = 2.0 * k - M + 1.0
    pair_sum = 2.0 * (w * xs).sum(0)
    return (term1 - 0.5 * pair_sum / (M * (M - 1.0))).mean()


def fgn_eval_eps(n, eps_dim, device=None):
    """THE representative member, in ONE place (E-057 §1f).

    Every read-out that predates E-057 — eval 1's z_t+1, the channel decode,
    the light in-training transport probe, the RAPID section forward — is a
    POINT instrument: it was written for a head with one output per input and
    it feeds pooled legacy trend columns. An FGN head has no single output, so
    something has to choose which member those instruments see, and the choice
    must live in one place or two call sites will quietly disagree.

    The choice is ε = 0, the centre of the noise distribution: at init (zero
    film) it is exactly the legacy computation, and after training it is the
    distribution's centre member. It is recorded as `fgn_eval_eps: "zeros"` in
    the run's `stage2_config` so no reader has to infer it. The HONEST
    ensemble read-outs are the new `stage2_val_*` keys, never these.
    """
    if not eps_dim:
        return None
    return torch.zeros(n, eps_dim, device=device)


class TemporalTransformer(nn.Module):
    """Causal transformer over one pixel's embedding sequence.

    Inputs are codec embeddings z_t (d_z) plus a per-pixel static context
    (lat, lon, and the codec embedding of the pixel's STATIC channels alone —
    the climatological identity of the place), added to every step. The
    month-of-year enters as sin/cos per STEP: dynamics may be phase-dependent
    (winter mixing vs summer stratification) even when the state is an
    anomaly. Output head predicts z_{t+1} from the hidden state at t.
    """

    def __init__(self, d_z=32, d_model=96, n_heads=4, n_layers=3, k_max=36,
                 direct=(), stencil=1, eps_dim=0):
        super().__init__()
        # E-022: stencil>1 widens the per-step input to the neighbourhood's
        # z (S*d_z, missing cells zero-filled) and appends the S static
        # observed-flags to the static context (geometry doesn't change
        # with time, so the flags belong there, not in every step).
        # stencil==1 keeps the EXACT legacy shapes so every published head
        # loads strict=True; the layout (centre slot first, STENCILS order)
        # is pinned by the zero-weight-equivalence unit test.
        self.stencil = stencil
        if stencil == 1:
            self.inp = nn.Linear(d_z + 2, d_model)     # z_t + (sin m, cos m)
            self.static = nn.Linear(d_z + 2, d_model)  # static-z + (lat, lon)
        else:
            self.inp = nn.Linear(stencil * d_z + 2, d_model)
            self.static = nn.Linear(d_z + 2 + stencil, d_model)
        self.pos = nn.Embedding(k_max, d_model)
        # E-057: eps_dim == 0 is EXACTLY the pre-2026-08-27 constructor — the
        # stock nn.TransformerEncoder, no eps_embed, no film — so every
        # published checkpoint still loads strict=True and the no-flag path
        # builds the identical module tree. eps_dim > 0 swaps in the FiLM
        # container whose keys match it one for one (see _CondLayer).
        self.eps_dim = int(eps_dim)
        if self.eps_dim:
            layer = _CondLayer(
                d_model, n_heads, dim_feedforward=4 * d_model,
                batch_first=True, norm_first=True, dropout=0.0)
            self.encoder = _CondEncoder(layer, n_layers)
            self.eps_embed = nn.Sequential(
                nn.Linear(self.eps_dim, d_model), nn.SiLU(),
                nn.Linear(d_model, d_model))
        else:
            layer = nn.TransformerEncoderLayer(
                d_model, n_heads, dim_feedforward=4 * d_model,
                batch_first=True, norm_first=True, dropout=0.0)
            self.encoder = nn.TransformerEncoder(layer, n_layers)
        self.head = nn.Linear(d_model, d_z)
        # DIRECT multi-horizon heads (E-014): one linear readout per horizon,
        # predicting z_{t+h} from the hidden state at t in a single forward —
        # no iteration, so the compounding/smoothing that an autoregressive
        # rollout accumulates (and that unroll training made WORSE, E-011)
        # never enters. The trunk is shared; each head is d_model x d_z.
        self.direct = tuple(int(h) for h in direct)
        if self.direct:
            self.heads_direct = nn.ModuleDict(
                {str(h): nn.Linear(d_model, d_z) for h in self.direct})
        self.d_model = d_model

    def forward(self, z_seq, month_seq, static_ctx, eps=None):
        """z_seq [B,K,d_z] · month_seq [B,K,2] · static_ctx [B,d_z+2]
        → pred [B,K,d_z] (ẑ at t+1 for every step), h [B,K,d_model].

        `eps` [B, eps_dim] is E-057's global noise vector and is REQUIRED
        exactly when the head was built with one. The three-positional-argument
        call every existing caller makes is unchanged."""
        B, K, _ = z_seq.shape
        # THE GUARD IS THE POINT, in both directions (E-057 §1b).
        # rollout_spatial.py builds a head from its checkpoint's args and calls
        # it with three arguments; without this an FGN head would roll CLEAN —
        # a deterministic trajectory produced by a model whose whole content is
        # that it is not deterministic, with nothing in any artefact saying so.
        # Refuse loudly instead.
        if self.eps_dim == 0 and eps is not None:
            raise ValueError(
                "TemporalTransformer was built with eps_dim=0 (a "
                "deterministic head) and was handed an eps vector. The noise "
                "has nowhere to enter; refusing rather than silently "
                "ignoring it.")
        if self.eps_dim and eps is None:
            raise ValueError(
                f"this is an FGN head (eps_dim={self.eps_dim}) and it cannot "
                f"be run without its noise vector: every forward is a SAMPLE "
                f"from the predictive distribution, conditioned on "
                f"eps ~ N(0,1)^{self.eps_dim}. A caller that has no eps must "
                f"choose a member deliberately — temporal.fgn_eval_eps() is "
                f"the representative (zeros) one — never fall into a default. "
                f"Plan: ml/plans/E057_fgn_head.md")
        h = (self.inp(torch.cat([z_seq, month_seq], -1))
             + self.static(static_ctx).unsqueeze(1)
             + self.pos.weight[None, :K])
        causal = nn.Transformer.generate_square_subsequent_mask(K, device=z_seq.device)
        if self.eps_dim:
            h = self.encoder(h, self.eps_embed(eps), mask=causal, is_causal=True)
        else:
            h = self.encoder(h, mask=causal, is_causal=True)
        return self.head(h), h


# ---- E-022: spatial stencils ------------------------------------------
# Predict a pixel's z_{t+1} from its NEIGHBOURHOOD's z, not just its own —
# the per-pixel model has zero cross-pixel coupling (measured consequences
# in E-021: rolls decay to a seasonal limit cycle; nothing can advect).
# Offsets are (dy, dx) in grid cells, CENTRE FIRST and in this fixed order
# — the input layout of every checkpoint depends on it. 13 is the classic
# 13-point stencil ("2 in each cardinal direction, 1 on the diagonal",
# Chris's spec): the 5x5 with its outer diagonal ring trimmed.
# Physics stated where the shape is chosen: one roll step is ONE MONTH, so
# stencil reach is 1-2 cells/month (~9-18 mm/s). Slow interior dynamics
# and deep flow are within reach; Gulf Stream advection (100-200 cells per
# month) is structurally NOT — see ml/plans/E022_spatial_coupling.md §1.
KM_PER_DEG = 111.32

STENCILS = {
    1:  [(0, 0)],
    9:  [(0, 0), (-1, -1), (-1, 0), (-1, 1), (0, -1),
         (0, 1), (1, -1), (1, 0), (1, 1)],
    13: [(0, 0), (-1, 0), (1, 0), (0, -1), (0, 1),
         (-1, -1), (-1, 1), (1, -1), (1, 1),
         (-2, 0), (2, 0), (0, -2), (0, 2)],
}


def ring_offsets(lat_deg, r_km, n_pts, dlat_deg):
    """E-023: (dy, dx) grid offsets of `n_pts` equidistant points on a circle
    of radius `r_km` around a pixel at `lat_deg`, bearing 0 = north.

    The ring is a circle on the GROUND, not on the grid: one cell spans
    27.8 km meridionally but 27.8·cos(φ) km zonally, so a fixed cell offset
    would be a 27 km step at the equator and a 9 km step at 70 °N — three
    different experiments in one run. The zonal step is therefore stretched
    by 1/cos(φ), which is why offsets are computed per pixel ROW."""
    coslat = max(np.cos(np.radians(lat_deg)), 0.05)
    out = []
    for k in range(n_pts):
        th = 2 * np.pi * k / n_pts
        dy = (r_km / KM_PER_DEG) * np.cos(th) / dlat_deg
        dx = (r_km / (KM_PER_DEG * coslat)) * np.sin(th) / dlat_deg
        out.append((int(round(dy)), int(round(dx))))
    return out


GOLDEN_ANGLE = 137.50776405003785     # 360 * (1 - 1/phi)


def spiral_offsets(lat_deg, r_min, r_max, n_pts, dlat_deg, aspect=1.0,
                   ramp_p=None):
    """E-026 SPIRAL (Chris, 2026-08-14): *"angular coordinates should be
    different for each point, think of a spiral going outward ... streams
    often flow straight, so it's important to catch 1-2 points for many
    incoming angles."*

    Point k sits at bearing k x the GOLDEN ANGLE, at a radius growing
    geometrically from r_min to r_max. The golden angle is not decoration: it
    is the unique rotation for which no prefix of the sequence ever clusters,
    so every k points are as evenly spread in bearing as k points can be (the
    sunflower/phyllotaxis arrangement). A three-ring shape samples 8 bearings
    three times over; a 24-point spiral samples 24 bearings once each. If what
    matters is catching a straight inflow from ANY direction rather than
    resolving one radius finely, that is the better trade — which is exactly
    the argument Chris made.

    `aspect` < 1 makes the spiral ELLIPTIC (Chris again, same day: *"the
    north pole is less important than having a receptive window across 4k km
    east / west"*): the meridional (north-south) extent is compressed to
    `aspect` times the zonal, so `r_min`/`r_max` name the ZONAL semi-axis and
    the shape reaches r*aspect north-south. The number is not a style choice
    — it is MEASURED from the flow itself (ml/measure_flow_anisotropy.py:
    geostrophic |u|/|v| from the tensor's own SSH channel), because water
    that moves twice as far east as north per month is best watched by a
    window with the same proportions. Bearings stay distinct: the compression
    is a monotone map of angle, so no two points collapse onto one."""
    # ramp_p switches the RADIUS RAMP (Chris, 2026-08-14: "an eliptic
    # spiral but with heavier weight on the outer points"). None keeps the
    # geometric ramp (log-uniform, near-heavy: of 34 points spanning
    # 111-4444 km only 7 sit beyond the half-way radius). A float p uses
    # r = r_min + (r_max - r_min) * f**p on uniform f. p = 0.5 is not a
    # knob setting but a NAMED arrangement: r ~ sqrt(k) at the golden angle
    # is Vogel's model of the sunflower head — the unique ramp with uniform
    # density per unit AREA, and because area grows quadratically, 26 of 34
    # points land beyond half-way and 32 beyond 1000 km. The near-field
    # information peak at 222 km gets a single point; that is the
    # HYPOTHESIS (the early E-026 reads attribute the wide arm's gain to
    # its outer ring), not an accident — the geometric-ramp arms keep the
    # near-heavy end of the spectrum as controls.
    coslat = max(np.cos(np.radians(lat_deg)), 0.05)
    out = []
    for k in range(n_pts):
        f = k / max(n_pts - 1, 1)
        if ramp_p is None:
            r_km = r_min * (r_max / r_min) ** f    # geometric (near-heavy)
        else:
            r_km = r_min + (r_max - r_min) * f ** ramp_p
        th = np.radians(k * GOLDEN_ANGLE)
        dy = (aspect * r_km / KM_PER_DEG) * np.cos(th) / dlat_deg
        dx = (r_km / (KM_PER_DEG * coslat)) * np.sin(th) / dlat_deg
        out.append((int(round(dy)), int(round(dx))))
    return out


def _ring_on(ring_km):
    """True when `ring_km` names at least one positive radius. Accepts a
    number, a list, or the CLI's comma string ("222,555")."""
    if isinstance(ring_km, str):
        if ring_km.startswith("spiral:"):
            return True
        return any(float(r) > 0 for r in ring_km.split(",") if r.strip())
    if isinstance(ring_km, (list, tuple)):
        return any(float(r) > 0 for r in ring_km)
    return float(ring_km or 0) > 0


def wraps_longitude(W, dlat, wrap_lon=None):
    """Should the zonal axis wrap? AUTO by default, because this is the one
    property that must not be remembered.

    Clipping longitude is CORRECT for a regional basin — a neighbour past
    -100 E genuinely is outside the experiment — and it becomes a WALL AT THE
    DATELINE the moment the window is global, which is precisely the change
    E-033 proposes. Measured on a land-free 0.5 deg rectangle with the
    sunflower-89 at 4444 km, dead-slot fraction:

        NA window (-100..20 E)   clip 33.7%   wrap 17.1%
        global (-180..180)       clip 15.4%   wrap  7.0%

    So going global while still clipping recovers only half of what the domain
    was costing, and the wrap is worth as much again as the tensor. A flag
    someone has to set would be a flag someone forgets, and the failure is
    silent: the run trains, the curves look normal, and a third of the
    Pacific's neighbours are zeros. So the DEFAULT IS DERIVED FROM THE GRID —
    a longitude axis that spans the planet wraps, one that does not, does not.
    Pass wrap_lon=True/False only to override deliberately (a regional window
    that happens to be 360 wide does not exist; a global one that must not
    wrap is a test)."""
    if wrap_lon is not None:
        return bool(wrap_lon)
    return abs(W * dlat - 360.0) < 1.5 * dlat


def build_stencil(H, W, ys, xs, stencil, ring_km=0.0, lats=None,
                  wrap_lon=None):
    """NBR [P, S] int64 indices into the P (ocean-pixel) ordering; -1 =
    missing (land, or outside the window). Longitude wraps only when the grid
    is global — see `wraps_longitude`; the family3 window is regional
    (-100..+20 E) and clips, unlike the codec's global gather_px.
    Slot 0 is always the centre pixel itself. Takes the grid SHAPE rather
    than the mask array: the pixel list ys/xs IS the mask (and under
    --max-pixels subsampling, absent pixels correctly read as missing).

    `ring_km` > 0 selects E-023 RING geometry instead of the fixed STENCILS
    table: slot 0 stays the centre and the remaining `stencil - 1` slots are
    equidistant points on a circle of that radius. Everything downstream —
    the input layout, the model, the checkpoint — is bit-identical to the
    fixed-table case with the same slot count, so `--stencil 9 --ring-km 222`
    and `--stencil 9` differ in exactly one thing: how far away the eight
    neighbours are. That is the whole experiment."""
    lin = np.full((H, W), -1, np.int64)
    lin[ys, xs] = np.arange(len(ys))
    NBR = np.full((len(ys), stencil), -1, np.int64)
    NBR[:, 0] = np.arange(len(ys))
    # Derived once, used by every branch. With no `lats` there is no grid step
    # to reason from, so an unset wrap_lon stays False — the historical
    # behaviour, and the fixed-table stencils reach 1-2 cells, where a
    # dateline is a rounding error rather than a wall.
    _dl = (float(np.round(np.diff(lats).mean(), 6))
           if lats is not None and len(np.asarray(lats)) > 1 else None)
    wrap = (wraps_longitude(W, _dl, wrap_lon) if _dl is not None
            else bool(wrap_lon))
    if str(ring_km).startswith("spiral:"):
        if lats is None:
            raise ValueError("spiral geometry needs `lats`")
        # Either separator, because the WORKFLOW rewrites one into the other.
        # ml-train.yml's `ring:` parser does RING="${RING//-/,}" — dashes are
        # how a multi-radius ring list gets through an input whose fields are
        # comma-separated — so a dispatch of `ring:spiral:222-1000` arrives
        # here as `spiral:222,1000`. Accepting both is the fix that cannot
        # cost a workflow edit: `ml-train.yml` sits exactly at the 25-input
        # ceiling and a 26th breaks every dispatch in the repo (ml/CLAUDE.md
        # §7). Pinned by test_spiral_survives_the_workflow_dash_rewrite.
        # 2 fields = circular (spiral:111-4444); an optional 3rd is the
        # ELLIPSE aspect, meridional/zonal (spiral:111-4444-0.5) — measured
        # from the flow, see spiral_offsets. Anything else is a typo and
        # must die here, not 6 GPU-hours in.
        parts = [float(v) for v in
                 re.split(r"[-,]", str(ring_km)[len("spiral:"):])]
        if len(parts) not in (2, 3, 4):
            raise ValueError("spiral wants rmin-rmax[-aspect[-ramp_p]], "
                             f"got {ring_km}")
        r0, r1 = parts[0], parts[1]
        asp = parts[2] if len(parts) >= 3 else 1.0
        # 4th field: radius-ramp exponent on the linear span (0.5 = Vogel /
        # sunflower, uniform-area, far-heavy). Absent = geometric ramp —
        # every checkpoint trained before this field existed keeps meaning
        # exactly what it meant.
        rp = parts[3] if len(parts) == 4 else None
        if rp is not None and not (0 < rp <= 2):
            raise ValueError(f"spiral ramp exponent must be in (0, 2], "
                             f"got {rp}")
        if not (0 < asp <= 1):
            raise ValueError(f"spiral aspect must be in (0, 1], got {asp} — "
                             "it is meridional/zonal, and >1 would claim the "
                             "flow moves farther north than east")
        dlat = float(np.round(np.diff(lats).mean(), 6))
        for y in np.unique(ys):
            sel = np.where(ys == y)[0]
            for k, (dy, dx) in enumerate(
                    spiral_offsets(float(lats[y]), r0, r1, stencil - 1, dlat,
                                   aspect=asp, ramp_p=rp)):
                yy, xx = y + dy, xs[sel] + dx
                if not (0 <= yy < H):
                    continue
                if wrap:
                    NBR[sel, k + 1] = lin[yy, xx % W]
                else:
                    okc = (xx >= 0) & (xx < W)
                    NBR[sel[okc], k + 1] = lin[yy, xx[okc]]
        assert (NBR[:, 0] == np.arange(len(ys))).all()
        return NBR
    radii = ([float(r) for r in str(ring_km).split(",") if str(r).strip()]
             if isinstance(ring_km, str) else
             ([float(r) for r in ring_km] if isinstance(ring_km, (list, tuple))
              else ([float(ring_km)] if float(ring_km) > 0 else [])))
    if radii:
        # NB: STENCILS is not consulted in ring mode, which is why slot counts
        # with no fixed-table entry (17 = centre + 16 ring points, E-026) are
        # legal here and would KeyError below.
        if lats is None:
            raise ValueError("ring geometry needs `lats` — the zonal step "
                             "depends on latitude")
        n_ring = stencil - 1
        if n_ring % len(radii):
            raise ValueError(f"{n_ring} ring slots do not divide evenly among "
                             f"{len(radii)} radii — a shape whose rings differ "
                             f"in point count is not what any caller means")
        per = n_ring // len(radii)
        dlat = float(np.round(np.diff(lats).mean(), 6))
        # per ROW, not per pixel: 281 offset computations instead of 84,405,
        # and every pixel in a row shares a latitude by construction
        for y in np.unique(ys):
            sel = np.where(ys == y)[0]
            offs = []
            for ri, r_km in enumerate(radii):
                o = ring_offsets(float(lats[y]), r_km, per, dlat)
                if ri % 2:
                    # rotate every second ring by half a sector, so a two-ring
                    # shape samples 2*per bearings instead of `per` bearings
                    # twice — otherwise the outer ring sits directly behind the
                    # inner one and buys strictly less than it could
                    o = ring_offsets(float(lats[y]), r_km, 2 * per, dlat)[1::2]
                offs.extend(o)
            for k, (dy, dx) in enumerate(offs):
                yy, xx = y + dy, xs[sel] + dx
                # yy is a SCALAR row here (every pixel in `sel` shares it), so
                # an off-grid row must be skipped rather than masked: numpy
                # validates a scalar index even when the column selection it
                # is paired with is empty, and raises instead of returning
                # nothing. Slots left at -1 are exactly right — a ring point
                # past the window edge is missing, like one on land.
                if not (0 <= yy < H):
                    continue
                if wrap:
                    NBR[sel, k + 1] = lin[yy, xx % W]
                else:
                    ok = (xx >= 0) & (xx < W)
                    NBR[sel[ok], k + 1] = lin[yy, xx[ok]]
    else:
        offs = STENCILS[stencil]
        for k, (dy, dx) in enumerate(offs):
            yy, xx = ys + dy, xs + dx
            if wrap:
                ok = (yy >= 0) & (yy < H)
                NBR[ok, k] = lin[yy[ok], xx[ok] % W]
            else:
                ok = (yy >= 0) & (yy < H) & (xx >= 0) & (xx < W)
                NBR[ok, k] = lin[yy[ok], xx[ok]]
    assert (NBR[:, 0] == np.arange(len(ys))).all()
    return NBR


def frame_ref(t, K, offsets=None):
    """The index every window gather is written against, given the window's
    ANCHOR t (the frame whose teacher-forced target is the headline t+1).

    Contiguous frames: the window START t-K+1, because frame j sits at ref+j.
    E-053.1 offsets: the anchor t ITSELF, because frame j sits at
    ref+offsets[j] and offsets[-1] == 0. Accepts an int or a tensor."""
    return (t - K + 1) if offsets is None else t


def frame_steps(K, offsets=None):
    """The per-frame step from `frame_ref`: `range(K)` for the contiguous
    stencil, the offset list for E-053.1. The frame arithmetic exists ONCE —
    gather_stencil, the season gather and the per-frame target gather all
    iterate this, so a new sampling pattern cannot reach one of them and miss
    the other two."""
    return range(K) if offsets is None else offsets


def window_touch_offsets(K, offsets, reach):
    """EVERY axis offset from a window's ANCHOR t that one forward pass and
    its loss touch, derived from the gather machinery itself and never
    re-stated by hand:

      · the K FRAME times — `frame_ref(t) + j` for j in `frame_steps` — which
        is what `gather_stencil` reads as INPUT;
      · each frame's TEACHER-FORCED TARGET, one bin later, which is what
        `win_ztgt` reads (the loss is DENSE over the window, not just t+1);
      · every SCORED reach offset t+r (the unroll fan and each --direct
        horizon).

    With `offsets=None` and reach [1] this is exactly range(-(K-1), 2), i.e.
    the contiguous span [t-K+1, t+1]. It is a function of the LAYOUT alone —
    it never looks at t_hold — so the pool rule and the certificate that
    checks it are two different expressions over one definition."""
    ref = int(frame_ref(0, K, offsets))          # anchor-relative window ref
    frames = [ref + int(j) for j in frame_steps(K, offsets)]
    return sorted(set(frames) | {f + 1 for f in frames}
                  | {int(r) for r in reach})


def frame_target_keep(t_idx, K, offsets, t_hold):
    """WHICH OF A WINDOW'S K DENSE TARGETS ARE TRAIN BINS — `[len(t_idx), K]`
    bool, True where that frame's teacher-forced target is NOT held out.

    `t_idx` is the windows' ANCHOR bins t (what the pool holds). Frame j of the
    window at t sits at `frame_ref(t) + j` and its target — the bin `win_ztgt`
    reads — is one bin later, `frame_ref(t) + j + 1`. That arithmetic is taken
    from `frame_ref`/`frame_steps` themselves, exactly as `window_touch_offsets`
    takes it, so the mask, the pool rule and the batch gather are three
    expressions over ONE definition of where a frame's target lives; a new
    sampling pattern cannot reach one of them and miss another.

    This is the whole of `--holdout-scope target`: the pool is the legacy one,
    and every per-frame loss contribution whose target bin is held out is
    dropped from the mean. Because the pool's reach condition already
    guarantees the LAST frame's target (t+1) is a train bin, at least one
    column is always True — no window is ever fully masked, so the masked mean
    can never divide by zero.
    """
    t_idx = np.asarray(t_idx, dtype=np.int64).reshape(-1)
    th = np.asarray(t_hold, dtype=bool)
    ref = np.asarray(frame_ref(t_idx, K, offsets), dtype=np.int64)
    steps = [int(j) for j in frame_steps(K, offsets)]
    keep = np.empty((len(t_idx), len(steps)), bool)
    for c, j in enumerate(steps):
        keep[:, c] = ~th[ref + j + 1]
    return keep


def _target_scope_certificate(T, t_hold, K, FOFF, ok_t, quiet):
    """The `--holdout-scope target` print and its EXACT identity check.

    Runs at POOL-BUILD time — the inputs are all it has cost you (§5.16) —
    and never touches `ok_t`: this scope's pool IS the legacy array, and the
    only thing that changes is which per-frame terms enter the loss.
    """
    idx = np.where(ok_t)[0]
    keep = frame_target_keep(idx, K, FOFF, t_hold)
    n_tot = int(keep.size)
    n_mask = int((~keep).sum())
    # ---- the certificate, by a DIFFERENT expression than the mask --------
    # `frame_target_keep` is a vectorised gather per frame column. This walks
    # every pooled window and every frame in a plain loop, rebuilding the
    # target bin from `frame_ref`/`frame_steps` the way `win_ztgt` does, and
    # counts the held-out ones. Two expressions, one definition: a bug in the
    # vectorised form cannot hide in both. Exact equality, not a threshold.
    n_mask2 = 0
    for t in idx:
        ref = int(frame_ref(int(t), K, FOFF))
        for j in frame_steps(K, FOFF):
            if bool(t_hold[ref + int(j) + 1]):
                n_mask2 += 1
    if n_mask2 != n_mask:
        sys.exit(
            f"--holdout-scope target FAILED ITS OWN CERTIFICATE: the mask "
            f"says {n_mask:,} of {n_tot:,} frame-targets are held out, the "
            f"brute-force recount over frame_ref/frame_steps says "
            f"{n_mask2:,}. The two disagree, so the objective cannot be "
            f"trusted — refusing to train.")
    # The other exact identity: the reach condition guarantees the LAST
    # frame's target t+1 is a train bin, so every pooled window keeps at
    # least one term and the masked mean's denominator is never zero.
    if not bool(keep.any(1).all()):
        sys.exit(
            f"--holdout-scope target FAILED ITS OWN CERTIFICATE: "
            f"{int((~keep.any(1)).sum()):,} pooled windows have EVERY frame "
            f"target held out, which the reach condition is supposed to make "
            f"impossible — refusing to train on a zero denominator.")
    if not quiet:
        print(f"--holdout-scope target: the pool is the legacy one, bin for "
              f"bin — held-out bins may still be read as CONTEXT — and the "
              f"loss drops every per-frame term whose TARGET bin "
              f"(frame_ref+j+1, what win_ztgt reads) is held out.", flush=True)
        print(f"  {n_mask:,} of {n_tot:,} frame-targets are held out and "
              f"will be masked ({100.0 * n_mask / max(1, n_tot):.2f}%); all "
              f"{int(ok_t.sum()):,} end-bins are kept.", flush=True)
        print(f"  certificate: an independent brute-force recount over "
              f"frame_ref/frame_steps agrees exactly ({n_mask2:,}), and "
              f"every pooled window keeps at least one target (the reach "
              f"guarantees t+1 is a train bin).", flush=True)
    return n_mask, n_tot


HOLDOUT_SCOPES = ("endpoint_contaminated", "target", "window")


def build_window_pool(T, t_hold, K, FOFF, reach, CTX_BACK,
                      scope="window", quiet=False):
    """The stage-2 TRAIN-WINDOW END-BIN mask, in ONE place: `ml/temporal.py`
    and `ml/jaxport/train_stage2.py` both call this, so the two trainers
    cannot drift on which bins a head is allowed to learn from.

    `scope="endpoint_contaminated"` is the LEGACY rule, in the legacy
    statement: a window ending at t is eligible iff the whole scored reach
    exists, the earliest frame exists, and no SCORED bin t+r is held out.
    That is what every archived run trained under, and it is what this
    function must keep returning bit-for-bit when it is asked for. It is
    NOT the default and never will be again: it leaks held-out years into
    training through straddling windows (the loss is dense; see below), and
    it is kept only so the 98 archived stage-2 runs stay reproducible.

    `scope="target"` returns THE SAME ARRAY — the legacy expression,
    unmodified — and is the minimal correct fix: no held-out bin is ever a
    TARGET, because the per-frame loss terms whose target is held out are
    masked out of the mean (`frame_target_keep`), while held-out bins MAY
    still be read as CONTEXT. Less exclusion than `window`, no pool change at
    all; the certificate below prints exactly how much of the objective it
    removes.

    `scope="window"` (the DEFAULT) adds a mask ON TOP of that array — the
    legacy expression still runs first and unchanged — keeping only windows
    NONE of whose touched bins (`window_touch_offsets`: frames, per-frame
    targets, reach) is held out. That closes the leak the legacy rule leaves:
    the loss is dense over the window (`win_ztgt`), so a window ending shortly
    after a holdout year teacher-forces that year's transitions into the
    weights, and feeds its bins in as context besides.

    Prints a certificate (§4.9: an exact identity, not a threshold) and
    `sys.exit`s if it fails, at `window` and at `target` alike — the only
    scope that certifies nothing is the legacy one, which has nothing to
    certify."""
    if scope not in HOLDOUT_SCOPES:
        sys.exit(f"--holdout-scope {scope!r} is not one of "
                 f"{HOLDOUT_SCOPES} — refusing to guess which pool a run "
                 f"was meant to train on.")
    ok_t = np.array([t + reach[-1] < T and t >= CTX_BACK
                     and not any(t_hold[t + r] for r in reach)
                     for t in range(T)])
    if scope != "window":
        # BOTH non-window scopes return THE LEGACY ARRAY ITSELF, unread and
        # unmodified — `target` differs from `endpoint_contaminated` in the
        # LOSS, never in the pool, so nothing about this array can drift
        # between them. The target certificate only counts.
        if scope == "target":
            _target_scope_certificate(T, t_hold, K, FOFF, ok_t, quiet)
        return ok_t
    touch = window_touch_offsets(K, FOFF, reach)
    # The bound that makes the vectorised gather below in-range for every t
    # `ok_t` kept: the earliest touched bin is the earliest FRAME (CTX_BACK
    # behind the anchor, by that constant's own definition) and the latest is
    # the far end of the reach. An exact identity, so a future layout that
    # breaks it fails here rather than indexing out of the array.
    assert touch[0] == -int(CTX_BACK) and touch[-1] <= int(reach[-1]), \
        (f"window_touch_offsets {touch[0]}..{touch[-1]} escapes the pool's "
         f"own bounds (-CTX_BACK {-int(CTX_BACK)}, reach {reach[-1]})")
    idx = np.where(ok_t)[0]
    hit = np.zeros(len(idx), bool)
    for o in touch:
        hit |= t_hold[idx + o]
    ok_w = np.zeros(T, bool)
    ok_w[idx[~hit]] = True
    # ---- the certificate, by a DIFFERENT expression than the mask ---------
    # The mask above is one vectorised OR over a precomputed offset list. This
    # walks every SURVIVING t and rebuilds the bins it touches from
    # `frame_ref`/`frame_steps` directly — the same calls the batch gather and
    # `win_ztgt` make — so a bug in the offset list cannot hide in both.
    bad = []
    for t in np.where(ok_w)[0]:
        ref = int(frame_ref(int(t), K, FOFF))
        bins = set()
        for j in frame_steps(K, FOFF):
            bins.add(ref + int(j))
            bins.add(ref + int(j) + 1)
        for r in reach:
            bins.add(int(t) + int(r))
        if any(bool(t_hold[b]) for b in sorted(bins)):
            bad.append(int(t))
    n_kept = int(ok_w.sum())
    if bad:
        sys.exit(
            f"--holdout-scope window FAILED ITS OWN CERTIFICATE: {len(bad)} "
            f"of {n_kept} pooled end-bins still touch a held-out bin "
            f"(first: {bad[:8]}). The mask and the brute-force recount "
            f"disagree, so the pool cannot be trusted — refusing to train.")
    if not quiet:
        print(f"--holdout-scope window: a window is eligible only if NONE of "
              f"the bins its forward pass touches is held out — the {K} "
              f"frames, each frame's teacher-forced target (win_ztgt) and "
              f"the scored reach {list(reach)}, i.e. anchor offsets "
              f"{touch[0]}..{touch[-1]}.", flush=True)
        print(f"  end-bins: {T} total · the endpoint rule excluded "
              f"{T - int(ok_t.sum()):,} (no-context prefix, reach past the "
              f"end, scored bin held out) · the window rule excluded "
              f"{int(ok_t.sum()) - n_kept:,} MORE · {n_kept:,} end-bins "
              f"remain in the pool.", flush=True)
        print(f"  certificate: 0 of {n_kept:,} pooled end-bins touch a "
              f"held-out bin (brute force over frame_ref/frame_steps, "
              f"{n_kept * (len(touch)):,} bin checks).", flush=True)
    return ok_w


def gather_stencil(Zt, base, p, NBR_t, K, offsets=None):
    """The ONE window-input gather (plan §3.4): every consumer of model
    inputs — train batch, monitor, light probe, eval — goes through here,
    so the stencil logic exists exactly once. Targets stay centre-only
    gathers at their call sites: the model predicts the CENTRE.

    Zt [T, P, d_z] · base [n] window-start month indices · p [n] centre
    pixels · NBR_t None (stencil 1 → the exact legacy gather) or [P, S]
    int64 with -1 = missing. Returns zseq [n, K, d_z] or [n, K, S*d_z]
    float32; missing neighbours are zero-filled.

    E-053.1 · `offsets`: None (default) keeps the contiguous stencil, where
    `base` is the window START and frame j sits at base+j — literally
    `range(K)`, so the off path is the arithmetic that produced every
    archived number and not a special case of the new one. A non-None
    `offsets` is a strictly increasing tuple ending in 0, `base` becomes the
    window ANCHOR t, and frame j sits at t+offsets[j]: the sunflower taken
    into the time dimension (ml/plans/E053_spacetime_stencil.md §4)."""
    steps = frame_steps(K, offsets)
    if NBR_t is None:
        return torch.stack([Zt[base + j, p] for j in steps], 1).float()
    nbr = NBR_t[p]                                        # [n, S]
    miss = nbr < 0
    safe = nbr.clamp(min=0)
    cols = []
    for j in steps:
        zj = Zt[(base + j).unsqueeze(1), safe].float()    # [n, S, d_z]
        zj[miss] = 0.0
        cols.append(zj.flatten(1))
    return torch.stack(cols, 1)


# Protocol v3 (2026-08-07, the global window): the RAPID probe section is
# the grid row nearest 26.5°N clipped to the array's Atlantic span (Abaco
# to the African shelf). On the NA pilot window protocol v2 used the whole
# row; the clip drops its Gulf-of-Mexico and NW-African cells and is
# REQUIRED on the global window, where the unclipped row would circle the
# planet through the Pacific and drown the section pool.
RAPID_LON = (-80.0, -13.0)


def section_of(lats, lons, ys, xs, lat, lon_lo, lon_hi):
    """(sec_y, indices into ys/xs) of a zonal probe section: the grid row
    nearest `lat`, clipped to [lon_lo, lon_hi]. Every transport array gets
    its own section this way (probe_kfold.TARGETS)."""
    sec_y = int(np.argmin(np.abs(lats - lat)))
    sel = np.where((ys == sec_y) & (lons[xs] >= lon_lo)
                   & (lons[xs] <= lon_hi))[0]
    return sec_y, sel


def rapid_section(lats, lons, ys, xs):
    """(sec_y, indices into ys/xs) of the protocol-v3 RAPID section."""
    return section_of(lats, lons, ys, xs, 26.5, RAPID_LON[0], RAPID_LON[1])


# ---------------------------------------------------------------------------
# E-055 · THE UNPOOLED TRANSPORT READ-OUT
#
# Every stage-2 transport number this programme has published came through
# `hid[:, -1].mean(0)` — a spatial mean over the 26.5N section — and geostrophic
# transport IS the east-minus-west contrast across that section, which a mean
# annihilates (ml/CLAUDE.md 3; ml/project_amoc.py measures z along the section
# at r 0.99 one cell apart and 0.35 eighty cells apart, i.e. ~2.5 effective
# independent pixels of 265). ml/probe_head.py's SectionHead is the proven
# mechanism for reading the section without pooling it; these helpers port it
# to the stage-2 hidden states and to ml/rollout_spatial.py's rolled latents,
# so ONE definition of "unpooled section read-out" serves both.
#
# Three properties this code must keep, because everything around it depends
# on them:
#
#  1. IT IS ADDITIVE. Nothing here writes, renames or reorders a pooled key.
#     98 archived stage-2 runs read `rapid_r_kfold`; the eval gate reads
#     `amoc_bands` against a hardcoded GATE_REF at GATE_TOL 0.0101.
#  2. IT DOES NOT MOVE THE GLOBAL RNG. Fitting a head draws from torch's
#     global generator (init + dropout). A pooled number computed after it
#     would then depend on whether this ran — which is exactly the kind of
#     silent drift that makes an archive incomparable. `_keep_rng` saves and
#     restores CPU, CUDA and numpy legacy state around every public entry
#     point, the way probe_head._usable_device does around its self-test.
#  3. IT IS A FUNCTION OF THE DATA AND THE SEED ONLY. Per-fold seeds are
#     derived from the run seed by index, and the batch draw uses its own
#     Generator, so two invocations at one seed are bit-identical.
#
# The head import is LAZY: probe_head imports this module at load time, so a
# module-level `from probe_head import SectionHead` would be a cycle. At the
# point of use temporal is fully initialised (the same argument the
# `from probe_kfold import kfold_r` in main() rests on).
# ---------------------------------------------------------------------------

UNPOOLED_STEPS = 800        # hard cap on optimiser steps per fit
UNPOOLED_PATIENCE = 6       # inner-tail evals (every 50 steps) without gain
UNPOOLED_EVERY = 50
UNPOOLED_HEAD_DIM = 64      # probe_head's original ~23k head


class _keep_rng:
    """Save and restore every global RNG this fit touches (see property 2)."""

    def __enter__(self):
        self._t = torch.get_rng_state()
        self._c = (torch.cuda.get_rng_state_all()
                   if torch.cuda.is_available() else None)
        self._n = np.random.get_state()
        return self

    def __exit__(self, *exc):
        torch.set_rng_state(self._t)
        if self._c is not None:
            torch.cuda.set_rng_state_all(self._c)
        np.random.set_state(self._n)
        return False


def lon_fraction(sec_lons):
    """Section pixels' east-west position in [0, 1] — probe_head:508 verbatim.

    This is the ONE feature that makes 'east' and 'west' distinguishable to a
    permutation-invariant attention pool. Without it the head could learn a
    weighting but not which end of the section it was weighting."""
    v = np.asarray(sec_lons, dtype=np.float64)
    return ((v - v.min()) / max(1e-6, np.ptp(v))).astype(np.float32)


def section_tokens(Zsec, lon_frac):
    """[n, P, d] per-pixel section states -> [n, P, d+2] attention tokens.

    Layout is probe_head's exactly (features, lon position, month offset), so
    the two read-outs are the same estimator on different features rather than
    two similar-looking ones. The month-offset column is 0.0 because both call
    sites hand over one time step per sample (probe_head's K=1 case)."""
    Z = np.asarray(Zsec, dtype=np.float32)
    if Z.ndim != 3:
        raise ValueError(f"section states must be [n, P, d], got {Z.shape}")
    n, P, dd = Z.shape
    lf = np.asarray(lon_frac, dtype=np.float32)
    if lf.shape != (P,):
        raise ValueError(f"lon_frac {lf.shape} does not match {P} section px")
    tok = np.zeros((n, P, dd + 2), dtype=np.float32)
    tok[..., :dd] = Z
    tok[..., dd] = lf[None, :]
    return tok


def unpooled_device(pref=None):
    """Where the read-out may TRAIN, decided by a real forward+backward.

    Delegates to probe_head._usable_device, which exists because run #397
    burned two 13-minute embedding passes discovering that the
    cross-attention BACKWARD dispatches to a Triton-JIT kernel and that box
    had no C compiler. Any failure means CPU: a slower read-out is a result,
    a fallen-over one is not."""
    from probe_head import _usable_device          # lazy: see the cycle note
    if pref is None:
        pref = "cuda" if torch.cuda.is_available() else "cpu"
    return _usable_device(torch.device(pref) if not isinstance(
        pref, torch.device) else pref)


def fit_attn_pool(tok, y, seed=0, steps=UNPOOLED_STEPS,
                  head_dim=UNPOOLED_HEAD_DIM, device=None,
                  patience=UNPOOLED_PATIENCE, every=UNPOOLED_EVERY):
    """Fit ONE SectionHead on `tok` (already restricted to fit rows) against a
    standardized target. Returns (net in eval mode, best inner-tail MSE).

    probe_head.fold_fit's protocol, with the step budget capped: AdamW
    lr 1e-3 / weight-decay 1e-2, batch 32 drawn from a seeded Generator, the
    last 20% of the rows held back as an inner tail, early stopping on it,
    best state restored. It is a separate function rather than a call into
    fold_fit because both call sites need the NET (rollout_spatial evaluates
    it once per rolled step), and fold_fit returns predictions."""
    from probe_head import SectionHead              # lazy: see the cycle note
    with _keep_rng():
        dev = torch.device(device or "cpu")
        X = torch.as_tensor(np.asarray(tok, dtype=np.float32)).to(dev)
        Y = torch.as_tensor(np.asarray(y, dtype=np.float32)).to(dev)
        if len(X) < 2:
            raise ValueError(f"attention pool needs >= 2 rows, got {len(X)}")
        torch.manual_seed(int(seed))
        net = SectionHead(X.shape[-1], d=head_dim).to(dev)
        opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-2)
        g = torch.Generator().manual_seed(int(seed))
        cut = max(1, int(0.8 * len(X)))
        Xf, Yf = X[:cut], Y[:cut]
        Xv, Yv = (X[cut:], Y[cut:]) if cut < len(X) else (Xf, Yf)
        best, best_state, bad = np.inf, None, 0
        for s in range(int(steps)):
            k = torch.randint(0, len(Xf), (min(32, len(Xf)),), generator=g)
            loss = (net(Xf[k]) - Yf[k]).pow(2).mean()
            opt.zero_grad(); loss.backward(); opt.step()
            if s % every == 0:
                net.eval()
                with torch.no_grad():
                    v = float((net(Xv) - Yv).pow(2).mean())
                net.train()
                if v < best - 1e-6:
                    best, bad = v, 0
                    best_state = {n_: p_.detach().clone()
                                  for n_, p_ in net.state_dict().items()}
                else:
                    bad += 1
                    if bad >= patience:
                        break
        if best_state:
            net.load_state_dict(best_state)
        net.eval()
        return net, float(best)


def attn_pool_predict(net, tok, device=None):
    """Out-of-sample predictions in the STANDARDIZED target space."""
    dev = torch.device(device or "cpu")
    with torch.no_grad():
        X = torch.as_tensor(np.asarray(tok, dtype=np.float32)).to(dev)
        return net(X).cpu().numpy().astype(np.float64)


def attn_pool_kfold(tok, y, years, seed=0, steps=UNPOOLED_STEPS,
                    head_dim=UNPOOLED_HEAD_DIM, device=None, boot=2000):
    """probe_kfold.kfold_r's PROTOCOL with an attention pool as the read-out.

    Same year-blocked folds, same train-only target standardization, same
    block bootstrap over whole years for the CI — so the unpooled number sits
    on the same footing as the pooled one it is written beside, and the two
    differ only in whether the section was averaged away before the fit.

    Returns a dict; `pred`/`target`/`years` ride along so
    scripts/paired_probe.py can settle pooled-vs-unpooled with a PAIRED test
    rather than two overlapping intervals (ml/CLAUDE.md 3, 8)."""
    y = np.asarray(y, dtype=np.float64)
    years = np.asarray(years)
    tok = np.asarray(tok, dtype=np.float32)
    pred = np.full(len(y), np.nan)
    uy = np.unique(years)
    with _keep_rng():
        for i, yy in enumerate(uy):
            te = years == yy
            tr = ~te
            if tr.sum() < 2 or not te.any():
                continue
            mu, sd = float(y[tr].mean()), float(y[tr].std() + 1e-9)
            # PER-FOLD SEED DERIVED FROM THE RUN SEED BY FOLD INDEX. One seed
            # for every fold would fit every fold from the same initialisation,
            # which is defensible but hides a whole axis of the estimator's
            # variance; `seed + i` is a fixed function of the run seed either
            # way, which is what reproducibility needs.
            net, _ = fit_attn_pool(tok[tr], (y[tr] - mu) / sd,
                                   seed=int(seed) + i, steps=steps,
                                   head_dim=head_dim, device=device)
            pred[te] = attn_pool_predict(net, tok[te], device) * sd + mu
        ok = np.isfinite(pred)
        if ok.sum() < 8 or np.std(y[ok]) == 0:
            raise ValueError(f"unpooled k-fold has only {int(ok.sum())} usable "
                             f"out-of-fold months — refusing to report an r")
        r = float(np.corrcoef(pred[ok], y[ok])[0, 1])
        rmse = float(np.sqrt(np.mean((pred[ok] - y[ok]) ** 2)))
        sigma = float(np.std(y[ok]))
        rng = np.random.default_rng(int(seed))
        rs = []
        for _ in range(int(boot)):
            pick = rng.choice(uy, len(uy), replace=True)
            sel = np.concatenate([np.where(years == p_)[0] for p_ in pick])
            if np.isfinite(pred[sel]).sum() > 8 and np.std(y[sel]) > 0:
                rs.append(np.corrcoef(pred[sel], y[sel])[0, 1])
        lo, hi = ((float(v) for v in np.percentile(rs, [2.5, 97.5]))
                  if rs else (float("nan"), float("nan")))
    return {"r": r, "lo": lo, "hi": hi, "n": int(ok.sum()), "rmse": rmse,
            "sigma": sigma, "pred": pred, "target": y,
            "years": np.asarray(years), "folds": int(len(uy))}


RESERVE_BYTES = 3 << 30      # runner logs, checkpoints, pip, room to breathe
RAM_HEADROOM_BYTES = 8 << 30  # the tensor and mask are already resident

# THE CACHE IS float16; THE ARITHMETIC IS NOT. At float32 the quarter-degree
# embedding is 10.4 GiB (516 x 84,405 x 64 x 4), which does not fit beside a
# ~15 GB torch image and ~11 GB of tensors on a 50 GB box — so the hygiene
# step deleted it, the next run spent 95 minutes rebuilding it, and the
# rebuild put the box back under the threshold. At float16 it is 5.2 GiB and
# fits with room to spare, which ends that treadmill on the hardware we
# actually rent. Vast will not resize a disk (see CLAUDE.md Part 2), so making
# the artefact smaller was the available lever.
#
# The precision cost is measured, not assumed: on unit-scale embeddings the
# round trip through float16 introduces an MSE of 4.3e-8, which is ~1e-7 of
# the z-MSE the experiments report (0.39-0.82). The figure we actually argue
# from is the model/persistence RATIO, where the error is common-mode across
# numerator and denominator and shifts the ratio by 1.8e-7. Seven orders of
# magnitude below the effect being measured.
#
# Everything downstream casts to float32 at the point of use, so gradients,
# optimiser state and the loss are unchanged. An existing float32 cache still
# loads and still works — `.float()` on it is a no-op — so this is not a
# flag day.
CACHE_DTYPE = np.float16


def make_sched(opt, a, last_epoch=-1):
    """The LR schedule, with a HORIZON-FREE option.

    Chris, 2026-08-10: *"let's not 'bake' num steps into the LR. Maybe we can
    use some LR decay (in the future) that does not depend on the number of
    total steps."*

    He is pointing at the root of two separate problems, not one.

    The BUG: `CosineAnnealingLR(T_max=steps)` makes the rate a function of the
    total, so a checkpoint's schedule is only meaningful alongside the budget
    it was trained under. Reload it while asking for a larger total and it
    believes it has finished — lr = 0.0, sixteen hours of updating nothing,
    every status reading success. That cost a run on 2026-08-10 and needed a
    rebuild-the-schedule special case, a refusal guard and a test to contain.

    The deeper COMPARABILITY problem: with a horizon-baked schedule, a
    6,000-step run and a 200,000-step run are at different learning rates at
    every shared step, so they are two different experiments that happen to
    share an architecture. E-007's three points each had to be described as
    "its own converged cosine", and the 200k point could not be a continuation
    of the 60k one — which is the whole reason E-008 became a warm restart.

    `invsqrt` (the Noam schedule) removes both at once: lr(s) depends only on
    s, so a run stopped at 60,000 and continued to 200,000 sees exactly the
    rate an uninterrupted 200,000-step run would have seen at those steps.
    Resume needs no special case, checkpoints are interchangeable, and two
    budgets become a prefix and its extension rather than two experiments.

    The price, stated honestly: cosine anneals to zero and therefore CONVERGES
    at a known point, which is what makes "the 60k result" a settled number.
    invsqrt never reaches zero, so a run has no natural end and results are
    "at step N" rather than "converged". For a programme asking "does more
    compute help?" that is the better trade — the question presumes an
    open-ended curve — but it is a trade, not a free win, and switching should
    be a deliberate experiment (one budget, both schedules) rather than a
    default flipped in passing.
    """
    if a.lr_schedule in ("invsqrt", "wsd", "expdecay"):
        warm = max(1, int(a.lr_warmup))

        def _warm_cos(s):
            """Cosine-shaped ramp to the peak — smooth at BOTH ends, unlike a
            linear ramp which arrives at the peak with a corner."""
            import math as _m
            return 0.5 * (1 - _m.cos(_m.pi * min(1.0, s / warm)))

        if a.lr_schedule == "wsd":
            # WARMUP - STABLE - DECAY. The literature's current answer, and a
            # better fit for this programme than either cosine or pure
            # inverse-sqrt: the stable phase is horizon-free, so a run can be
            # extended and its checkpoints are interchangeable, while the
            # cooldown recovers a genuine CONVERGED endpoint — which is what
            # invsqrt gives up and what makes "the 60k result" a settled
            # number rather than a reading at step 60,000.
            #
            # The consequence for this programme is concrete: E-007's four
            # budgets could be ONE run with four short cooldowns branched off
            # the stable phase, instead of four experiments that cannot be
            # compared as a trajectory.
            cool = max(1, int(round(a.steps * a.lr_cooldown_frac)))
            stable_end = max(warm, a.steps - cool)

            def factor(step):
                s = step + 1
                if s <= warm:
                    return s / warm
                if s <= stable_end:
                    return 1.0
                # Linear to zero: convex theory puts the optimal cooldown
                # shape at linear, and D2Z finds decaying fully to zero beats
                # stopping at a floor, increasingly so the longer you train.
                return max(0.0, (a.steps - s) / max(1, a.steps - stable_end))
        elif a.lr_schedule == "expdecay":
            # COSINE WARMUP, THEN EXPONENTIAL DECAY. Chris, looking at the WSD
            # trapezoid on the status page: "123's learning schedule doesn't
            # look great (too constant, then too steep). if nothing better use
            # cosine warmup and then exp decay."
            #
            # He is right on both counts and the second one is not only
            # aesthetic. WSD's cooldown is sized as a FRACTION of the total, so
            # the schedule is horizon-free right up until the part that is not
            # — extend the run and the cooldown moves, which is the same
            # coupling cosine has, merely postponed. Exponential decay with an
            # ABSOLUTE half-life has no such term: lr(s) = peak * 2^(-s/H)
            # depends on s and H alone. Stop anywhere, extend anywhere, and the
            # prefix is unchanged.
            #
            # It is also smooth everywhere — no plateau, no corner into the
            # decay — and it decays fastest early, when the model is furthest
            # from any optimum, rather than holding a constant rate for 90% of
            # the run.
            #
            # It does not reach zero, which is deliberate: decay-to-zero is a
            # borrowed prior we have not tested (docs/ML_BASICS.md §9), and a
            # schedule that never arrives is the honest default until the
            # floor-vs-zero control has actually run.
            half = max(1.0, float(a.lr_halflife))
            # An OPTIONAL terminal taper to zero. Off by default, because
            # decay-to-zero is a borrowed prior; on, it makes the ONE-VARIABLE
            # control for it — same curve, same half-life, differing only in
            # whether the tail reaches zero.
            cool = max(0, int(round(a.steps * a.lr_cooldown_frac)))
            taper_from = a.steps - cool

            def factor(step):
                s = step + 1
                base = _warm_cos(s) if s <= warm else 0.5 ** ((s - warm) / half)
                if cool and s > taper_from:
                    base *= max(0.0, (a.steps - s) / cool)
                return base
        else:
            def factor(step):
                s = step + 1
                return min(s / warm, (warm / s) ** 0.5)

        for g in opt.param_groups:
            g.setdefault("initial_lr", g["lr"])
        return torch.optim.lr_scheduler.LambdaLR(opt, factor,
                                                 last_epoch=last_epoch)
    return torch.optim.lr_scheduler.CosineAnnealingLR(opt, a.steps,
                                                      last_epoch=last_epoch)


def codec_weight_hash(ck):
    """Identity of the codec that produced an embedding, in ten hex digits.

    The embed cache MUST be codec-aware. A bare Z_<run>.npy poisoned runs
    #10/#11 (2026-08-07): the Actions cache carried run #8's embeddings, the
    (T, P, d_z) shape check matched, and two stage-2 models trained on the
    WRONG codec's z — healthy z-space skill, catastrophic decoded skill,
    because the z they predicted was not the z their decoder speaks. The hash
    in the filename makes a stale cache a miss rather than a lie.

    It lives in a function because embed_cache_sync.py has to derive exactly
    the same name to pull or push the cache, and two copies of a hash rule are
    two chances to disagree — which would silently reintroduce the #10/#11
    failure through the release instead of through the local disk.
    """
    import hashlib
    return hashlib.md5(b"".join(
        v.numpy().tobytes()
        for v in list(ck["model"].values())[:4])).hexdigest()[:10]


def data_fingerprint(path, chunk=1 << 24):
    """Ten hex digits identifying the TENSOR an embedding was computed from.

    The codec hash alone is not enough, and the gap is the #10/#11 failure one
    level up. Measured 2026-08-11: `gpu-box-47094145` holds a
    family3_na025.npz hashing b40f5b0b… and `gpu-box-35586926` holds one
    hashing adcbe700… — same recipe, same filename, different bytes, because
    each box builds its own and the recipe gained a channel between builds.

    With a key of `Z_<codec>.npy`, a box holding tensor B pulls embeddings
    computed from tensor A and cannot tell: the shape check passes (both are
    [516, 84405, 64]), verify() passes (length and dtype are right), and
    stage 2 trains on embeddings of a DIFFERENT dataset from the one its
    persistence baseline and probes use. That is precisely the failure
    codec_weight_hash was written to stop, with "codec" replaced by "data".

    Cheap enough to do every run: sha256 of a 2.8 GiB file is seconds, and it
    is the same digest provenance.json already publishes, so the cache name
    and the provenance record can be checked against each other by eye.
    """
    import hashlib
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(chunk), b""):
            h.update(b)
    return h.hexdigest()[:10]


# ---------------------------------------------------------- holdout blocks --
# E-067, "the two-year roll". A roll is truncated at the end of the held-out
# stretch it started in (`scored_horizon`), so with SINGLE held-out years —
# "2009,2017,2023", every archived roll — nothing past 365 d can ever be
# scored, whatever `--horizon` asks for. Holding out CONSECUTIVE years
# ("2008,2009,...") buys the leads: the truncation moves to the end of the
# BLOCK, and 146 pentads (730 d) fit inside a two-year one.
#
# The block is derived from the year list, never declared beside it. A second
# way of saying which years are held out is a second thing that can disagree
# with `t_hold`, and `t_hold` is what decides the standardisation statistics
# and the training pool — so the grouping is a FUNCTION of the years and
# nothing else. A single-year list therefore produces single-year blocks and
# every downstream call is the call it always was.
def hold_blocks(hold_years):
    """The held-out years as maximal runs of CONSECUTIVE years, in order.

    "2009,2017,2023"                  -> [(2009, 2009), (2017, 2017),
                                          (2023, 2023)]
    "2008,2009,2016,2017,2022,2023"   -> [(2008, 2009), (2016, 2017),
                                          (2022, 2023)]

    Takes the list in any order and with duplicates — it sorts and dedupes
    first, so the blocks are a property of the SET of years and a re-ordered
    dispatch string cannot produce a different protocol."""
    ys = sorted({int(str(y).strip()) for y in hold_years if str(y).strip()})
    out = []
    for y in ys:
        if out and y == out[-1][1] + 1:
            out[-1][1] = y
        else:
            out.append([y, y])
    return [(int(a), int(b)) for a, b in out]


def block_bounds(b):
    """`(y0, y1)` from either a block tuple or a bare year (int or str).

    The bare-year form is what keeps every single-year call site — and every
    caller outside this file — working unchanged."""
    if isinstance(b, (tuple, list, np.ndarray)):
        return int(b[0]), int(b[1])
    return int(b), int(b)


def block_label(b):
    """The block's key in the result JSON: "2009" for a single-year block,
    "2008-2009" for a run.

    A one-year block's label is EXACTLY the `str(Y)` every archived artefact
    is keyed by, which is what lets the blocks replace the year loop without
    moving a byte of a single-year roll."""
    y0, y1 = block_bounds(b)
    return f"{y0:04d}" if y0 == y1 else f"{y0:04d}-{y1:04d}"


def hold_key(eff_years, codec_years):
    """The embed cache's HOLDOUT TOKEN: `""` when the effective holdout years
    are the codec's own, `_hold-<block labels>` when they are not.

    THIS CLOSES THE HAZARD THE TWO-MASKS COMMENT IN `main` NAMES. The cache is
    keyed by (codec weight hash, sha256 of the RAW tensor file), and NEITHER
    TERM SEES THE ANOMALY TRANSFORM — so two runs on the same codec and the
    same tensor with different z-score statistics share one cache key, and
    whichever pulls first trains on the other's embeddings while every shape,
    dtype and length check passes. `--holdout-years` changes exactly those
    statistics (it moves `t_hold`, which the transform reads), so from E-067 on
    it changes the NAME too. With no override the token is empty and every
    existing asset, path and glob is byte-for-byte what it was.

    The token is derived, never declared: `ml/embed_cache_sync.py`'s
    `cache_name` calls THIS function on the same two lists, so the local path
    and the release asset cannot disagree about which cache a run means."""
    eff = _year_set(eff_years)
    cod = _year_set(codec_years)
    if not eff or eff == cod:
        return ""
    return "_hold-" + "-".join(block_label(b) for b in hold_blocks(eff))


def _year_set(y):
    """A comma string, a sequence or None as a sorted list of year strings."""
    if y is None:
        return []
    items = str(y).split(",") if isinstance(y, str) else list(y)
    return sorted({str(v).strip() for v in items if str(v).strip()})


def embed_cache_path(run, whash, dhash=None, hold=""):
    """`Z_<run>_<codec>[_<tensor>][_hold-<blocks>].npy`.

    dhash is optional ONLY so old callers keep working; every path that
    publishes or pulls must pass it, or the name means less than it claims.

    `hold` is `hold_key`'s token — empty for every run whose holdout years are
    the codec's, which is every run before E-067 and every run since that does
    not pass `--holdout-years`.
    """
    tail = f"{whash}_{dhash}" if dhash else whash
    return os.path.join(HERE, "cache", f"Z_{run}_{tail}{hold}.npy")



def _progress_path(tmp):
    return tmp + ".progress"


def _mark_done(cache_path):
    """`<cache>.done`, holding the cache's byte size: THIS embedding pass
    finished.

    Read by ml/embed_cache_sync.py:push, which refuses to publish a cache
    without one. The size is in the file rather than the file being empty so
    the attestation is about these bytes and not about this path — a marker
    left behind by a different array fails the comparison instead of vouching
    for a stranger.

    Instrumentation never breaks the run: a cache that cannot be marked is
    still a cache this run can use, it just cannot be published until
    something marks it.
    """
    try:
        with open(cache_path + ".done", "w") as f:
            f.write(str(os.path.getsize(cache_path)) + "\n")
    except OSError as e:
        print(f"  could not mark the embed cache complete ({e}) — it will not "
              f"be publishable until a run marks it")


def _publish_progress_path(cache_path):
    """`<cache>.progress` — the marker a PUBLISHER reads.

    Deliberately NOT the same file as `<cache>.partial.progress`, which is
    this module's own resume marker and is keyed to the temp file that holds
    the bytes. This one is keyed to the CACHE's published name, because
    everything that reads it — ml/embed_cache_sync.py's `push --partial`, and
    its `pull` on the box that resumes someone else's work — knows the cache
    by that name and nothing at all about our `.partial`.
    """
    return cache_path + ".progress"


def _cache_of_partial(tmp):
    """`<cache>` from `<cache>.partial`. One place, because the publisher's
    marker and the resume marker must agree about which cache they describe."""
    return tmp[:-len(".partial")] if tmp.endswith(".partial") else tmp


def _npy_header_bytes(path):
    """The offset of the array's FIRST byte — 64 to 128 bytes of .npy header.

    The publisher needs an ABSOLUTE byte offset for the end of the last
    complete row, because it slices the file into 1.5 GiB release chunks from
    byte zero. Counting rows and forgetting the header would put every chunk
    boundary 128 bytes late, which is a shift no length check can see.
    """
    with open(path, "rb") as f:
        major, minor = np.lib.format.read_magic(f)
        {(1, 0): np.lib.format.read_array_header_1_0,
         (2, 0): np.lib.format.read_array_header_2_0}[(major, minor)](f)
        return f.tell()


def _mark_publish_progress(cache_path, src, rows_flushed, T, P, d_z):
    """`<cache>.progress`: how many ROWS are real, and where they end in bytes.

    Chris, 2026-08-25: *"Publishing should happen 'during' the embedding
    computation, for example, after each 1/100th of the data. A new job that
    needs the same embedding can choose to continue the computation (if 32/100
    are already complete it will start with chunk 33)."* This marker is what
    makes that possible. It is the ONLY statement anybody can make about a
    memmap that `open_memmap` allocated at its full (T, P, d_z) shape before
    the first month was written: the file's length, dtype and T are right from
    the first second and say nothing whatever about how much of it is real.

    Written after the flush that produced those rows and atomically (temp
    sibling + `os.replace`, so a publisher polling every ten minutes can never
    read a half-written marker), so it can only ever UNDER-claim
    (ml/CLAUDE.md §5.21). Under-claiming costs a chunk that could have been
    published ten minutes earlier. Over-claiming publishes zeros as
    embeddings — real numbers, wrong months, no symptom.
    """
    try:
        row = int(P) * int(d_z) * CACHE_DTYPE(0).itemsize
        mark = {"rows_flushed": int(rows_flushed), "T": int(T), "P": int(P),
                "d_z": int(d_z),
                "bytes_flushed": _npy_header_bytes(src) + int(rows_flushed) * row,
                "dtype": str(np.dtype(CACHE_DTYPE)),
                "updated_iso": time.strftime("%Y-%m-%dT%H:%M:%SZ",
                                             time.gmtime())}
        p = _publish_progress_path(cache_path)
        with open(p + ".part", "w") as f:
            json.dump(mark, f)
        os.replace(p + ".part", p)
    except (OSError, ValueError, KeyError) as e:
        # Instrumentation never breaks the run: an unpublishable cache is
        # still a cache THIS run can use.
        print(f"  (publish progress marker failed: {e})", flush=True)


def _clear_publish_progress(cache_path):
    """Remove `<cache>.progress`. Called where `.done` is written: a cache
    that is COMPLETE must not also advertise itself as a work in progress, or
    a puller would fetch the prefix of a file that is entirely there."""
    try:
        os.remove(_publish_progress_path(cache_path))
    except OSError:
        pass


def _read_publish_progress(cache_path, T, P, d_z):
    """rows_flushed for a cache that a PULL seeded with a published partial,
    or None if there is no usable marker beside it.

    This is the other half of Chris's sentence: the box that pulls chunks
    1..32 has a full-length, correctly-typed, correctly-shaped cache whose
    tail is zeros, and nothing in the file says so. The marker does, and
    because the publisher's marker under-claims and the pull clamps it to the
    bytes it actually received, this can only ever ask for MORE recomputation
    than strictly necessary.
    """
    p = _publish_progress_path(cache_path)
    if not os.path.exists(p):
        return None
    try:
        with open(p) as f:
            mark = json.load(f)
        if (int(mark["T"]), int(mark["P"]), int(mark["d_z"])) != (T, P, d_z):
            print(f"  ignoring a published partial for a different shape "
                  f"({mark.get('T')}, {mark.get('P')}, {mark.get('d_z')}) — "
                  f"this run wants ({T}, {P}, {d_z})")
            return None
        if mark.get("dtype") != str(np.dtype(CACHE_DTYPE)):
            print(f"  ignoring a published partial of dtype "
                  f"{mark.get('dtype')}")
            return None
        rows = int(mark["rows_flushed"])
        return rows if 0 < rows < T else None
    except Exception as e:                                    # noqa: BLE001
        print(f"  published partial marker unusable "
              f"({type(e).__name__}: {e}) — embedding from scratch")
        return None


def _adopt_published_partial(cache_path, rows):
    """Turn a pulled prefix into this run's own resumable `.partial`.

    The pull writes the cache under its FINAL name — that is the name every
    reader knows — so without this the branch below would find a full-length
    file of the right shape and hand back an array whose tail is zeros. The
    rename is what makes the two markers say the same thing: from here on the
    ordinary resume path owns the file, and the ordinary completion path
    publishes it.
    """
    tmp = cache_path + ".partial"
    if os.path.exists(tmp):
        # This box was already embedding this cache itself. Its own partial is
        # at least as trustworthy as a stranger's and may be further along, so
        # it wins — and the pulled prefix is DELETED rather than left lying
        # under the final name, where the branch below would read it as a
        # finished cache the moment this marker went away.
        print(f"  a local partial embedding is already on this disk — keeping "
              f"it and discarding the pulled prefix")
        try:
            os.remove(cache_path)
            _clear_publish_progress(cache_path)
        except OSError as e:
            print(f"  (could not remove the pulled prefix: {e})")
        return
    os.replace(cache_path, tmp)
    _mark_progress(tmp, None, rows, *_shape_of(tmp))
    print(f"  RESUMING A PUBLISHED PARTIAL: {rows} row(s) came down from the "
          f"release, the rest will be computed here", flush=True)


def _shape_of(path):
    """(T, P, d_z) from a .npy header."""
    with open(path, "rb") as f:
        major, minor = np.lib.format.read_magic(f)
        shape = {(1, 0): np.lib.format.read_array_header_1_0,
                 (2, 0): np.lib.format.read_array_header_2_0}[(major, minor)](f)[0]
    return tuple(int(x) for x in shape)


def _resume_partial(tmp, T, P, d_z):
    """(memmap, months_already_done) for a half-built cache, or (None, 0).

    The marker is written AFTER the data is flushed, never before, so it can
    only ever under-claim. An over-claiming marker would be the worst possible
    outcome here: the run would skip months that were never written and the
    embedding would carry zeros for them — real numbers, wrong months, no
    symptom. Losing a few minutes of recomputation is the cheap side of that
    trade and it is the side this takes.
    """
    prog = _progress_path(tmp)
    if not (os.path.exists(tmp) and os.path.exists(prog)):
        return None, 0
    try:
        with open(prog) as f:
            mark = json.load(f)
        if (tuple(mark.get("shape", ())) != (T, P, d_z)
                or mark.get("dtype") != str(np.dtype(CACHE_DTYPE))):
            print(f"  ignoring a partial cache for a different shape/dtype "
                  f"({mark.get('shape')}, {mark.get('dtype')})")
            return None, 0
        done = int(mark.get("months_done", 0))
        out = np.load(tmp, mmap_mode="r+")
        if out.shape != (T, P, d_z) or not (0 < done < T):
            return None, 0
        print(f"  RESUMING the embedding at month {done}/{T} "
              f"({done / T * 100:.1f}% already on disk) — "
              f"{(T - done) / T * 100:.0f}% left to compute", flush=True)
        return out, done
    except Exception as e:                                    # noqa: BLE001
        print(f"  partial cache unusable ({type(e).__name__}: {e}) — "
              f"starting the embedding from scratch")
        return None, 0


def _mark_progress(tmp, out, months_done, T, P, d_z):
    """Flush the DATA, then record how far it got. Order is the whole point.

    TWO markers, one flush, and they are for two different readers. The local
    one (`<cache>.partial.progress`) lets THIS box resume its own interrupted
    pass; the published one (`<cache>.progress`) lets a publisher ship the
    finished prefix and another box resume from it. Writing them from one
    place is what stops them disagreeing about which months are real, and
    both are written after the flush, so both can only under-claim.

    `out` may be None — the pulled-partial adoption above has no memmap open
    and nothing to flush; the bytes arrived through `os.replace`.
    """
    try:
        if out is not None:
            out.flush()
        p = _progress_path(tmp)
        with open(p + ".part", "w") as f:
            json.dump({"months_done": int(months_done), "shape": [T, P, d_z],
                       "dtype": str(np.dtype(CACHE_DTYPE))}, f)
        os.replace(p + ".part", p)
    except OSError as e:
        print(f"  (progress marker failed: {e})", flush=True)
    _mark_publish_progress(_cache_of_partial(tmp), tmp, months_done, T, P, d_z)


def _free_ram_bytes():
    """MemAvailable, i.e. what can be allocated without swapping — not MemFree,
    which excludes reclaimable page cache and reads absurdly low on a box that
    has just streamed a 10 GB tensor through it."""
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemAvailable:"):
                    return int(line.split()[1]) * 1024
    except OSError:
        pass
    return 0


def _cache_plan(cache_path, need_bytes):
    """Decide WHERE the embedding lives: the disk cache, or RAM.

    The memmap was introduced because Z is ~10.4 GiB next to a 10.1 GiB
    tensor and that combination OOM-killed a 7 GB box twice on 2026-08-07.
    The boxes we rent now carry 126 GB of RAM and a 50 GB disk, so the
    constraint has moved and the code had not: #117 spent an hour writing a
    10.4 GiB cache toward 6 GiB of free disk, on a machine using 15 of its
    126 GB of memory. Memmapping to the scarce resource because the abundant
    one used to be scarce is the whole bug.

    Order of preference, and the reasons:
      1. DISK, if it fits after pruning — the cache is worth real money. A
         repeat stage-2 run on the same box skips ~50 minutes of embedding.
      2. RAM, if the disk cannot hold it but memory can. The run proceeds and
         only the cache is lost, which costs the NEXT run, not this one.
      3. Refuse. Both exhausted means the box is the wrong size for the job,
         and that is worth saying before an hour rather than after.
    """
    import shutil
    gb = lambda b: b / (1 << 30)
    d = os.path.dirname(cache_path)
    os.makedirs(d, exist_ok=True)
    # The headroom SCALES WITH THE ALLOCATION, capped. A flat 3 GiB reserve
    # reads as prudence until a smoke test with a 5 MB cache is refused on a
    # sandbox with 0.9 GiB free — the constant, not the risk, was doing the
    # refusing. What actually matters is that a big write leaves the box
    # usable afterwards, so the demand is proportional to the write and
    # bounded above by what a runner needs to keep working.
    want = need_bytes + min(RESERVE_BYTES, max(need_bytes, 256 << 20))
    _prune_stale(cache_path, want)
    free = shutil.disk_usage(d).free
    print(f"  embed cache needs {gb(need_bytes):.2f} GiB "
          f"(+{gb(want - need_bytes):.2f} headroom); "
          f"{gb(free):.2f} GiB disk free")
    if free >= want:
        return True
    ram = _free_ram_bytes()
    if ram >= need_bytes + min(RAM_HEADROOM_BYTES,
                               max(need_bytes, 512 << 20)):
        print(f"  disk cannot hold it — building Z in RAM instead "
              f"({gb(ram):.0f} GiB available). The cache is skipped, so the "
              f"NEXT stage-2 run on this box re-embeds; this one proceeds.")
        return False
    raise SystemExit(
        f"nowhere to put the embedding: needs {gb(need_bytes):.1f} GiB, "
        f"disk has {gb(free):.1f} GiB free after pruning every stale Z_*.npy "
        f"and RAM has {gb(ram):.1f} GiB available. Refusing to start — "
        f"open_memmap allocates lazily, so starting anyway would fail "
        f"mid-write an hour from now with the disk full and the runner "
        f"offline. Free space on the box, rent a larger one, or "
        f"use --max-pixels.")


def _prune_stale(cache_path, want):
    """Free space for the embedding cache BEFORE opening the memmap.

    `open_memmap` creates a SPARSE file: the 10.4 GiB is claimed lazily, page
    by page, over the ~50 minutes the embedding takes. So a box with 7 GiB
    free starts happily, runs for forty minutes, and dies on a write with the
    cache 90% built — and on these 50 GB boxes a full disk does not just fail
    the job, it takes the runner offline (CLAUDE.md Part 2). The failure is
    maximally expensive and maximally late.

    ml-train.yml has a hygiene step that prunes below 8 GB free. That number
    is SMALLER THAN THE SINGLE ALLOCATION IT GUARDS: a guard sized under the
    thing it is guarding against will pass and then the write will fail. The
    check belongs here, where T, P and d_z are known and the requirement is a
    computed number rather than a guess.

    Stale Z_*.npy from other codecs are the reclaimable tier — they are pure
    cache, keyed by a weight hash, and re-derive from a checkpoint. Anything
    else on that disk (tensors, checkpoints) is not ours to delete.
    """
    import glob
    import shutil
    d = os.path.dirname(cache_path)
    free = shutil.disk_usage(d).free
    gb = lambda b: b / (1 << 30)
    if free >= want:
        return
    # OUR OWN .partial IS NOT STALE. It is the file this pass is about to
    # resume from — either its own interrupted work or a prefix that came down
    # from the release — and sweeping it away turns a resume into a rebuild at
    # exactly the moment (a tight disk) when the hour it costs is dearest.
    # The exclusion used to name only `cache_path`, which does not match it.
    mine = {os.path.abspath(cache_path), os.path.abspath(cache_path + ".partial")}
    stale = sorted((p for p in glob.glob(os.path.join(d, "Z_*.npy")) +
                    glob.glob(os.path.join(d, "Z_*.npy.partial"))
                    if os.path.abspath(p) not in mine),
                   key=lambda p: os.path.getmtime(p))
    for p in stale:
        try:
            n = os.path.getsize(p)
            os.remove(p)
            # …and the completeness marker with it. A marker that outlives its
            # cache is an attestation with nothing to attest to; it fails safe
            # (it records the byte size, so it cannot vouch for a different
            # file) but leaving it turns a full-disk sweep into litter.
            for side in (".done", ".progress"):
                if os.path.exists(p + side):
                    os.remove(p + side)
            free = shutil.disk_usage(d).free
            print(f"  pruned stale embed cache {os.path.basename(p)} "
                  f"({gb(n):.1f} GiB) — {gb(free):.1f} GiB free")
        except OSError as e:
            print(f"  could not prune {p}: {e}")
        if free >= want:
            return
    # Not an error here — _cache_plan decides what to do when the disk still
    # cannot hold it, and RAM is usually the answer on these boxes.


def embed_everything(model, X, OBS, ctx_all, lats, lons, ys, xs, d_z,
                     cache_path=None, batch=8192, mask_chan=None,
                     progress=None, t_sel=None, blk_rows=None, blk_pad=None):
    """Frozen codec embeddings for every (t, pixel in ys/xs): [T, P, d_z].
    Cached on disk — the embedding pass is the expensive part of stage 2
    (T×P encoder forwards), and every probe variant reuses it.

    Runs on whatever device the MODEL is on: 401 months x ~45k ocean pixels
    is 18M encoder forwards, which is hours of CPU and minutes of GPU. The
    big tensors (X, OBS, and the output) stay in host memory — only the
    per-batch slice crosses — because Z alone is ~4.6 GB at global scale and
    the point is to spend VRAM on arithmetic, not storage.

    `t_sel` EMBEDS ONLY THE TIMESTEPS ASKED FOR, and returns [len(t_sel), P,
    d_z] in that order. Written for the in-training LIGHT probe, which spent
    its whole budget embedding all T and then read only the ~46% of rows the
    RAPID record covers (trainprobe.probe_now: `ridge_r(Fsec[ridx], ...)` is
    the sole reader). Every timestep is an independent forward — the loop
    below has no state that crosses `t` — so the rows that come back are
    BIT-IDENTICAL to the corresponding rows of a full pass, not merely close.

    It is refused together with `cache_path`: the cache is keyed by shape and
    a partial-time array of the right (T, P, d_z) shape could never be told
    from a complete one, which is `flush, THEN mark` (CLAUDE.md §5.21) failing
    in the one direction that over-claims."""
    dev = next(model.parameters()).device
    T, H, W, C = X.shape
    P = len(ys)
    if getattr(model, "k_time", 1) > 1:
        if blk_rows is None:
            raise ValueError(
                "embed_everything: this codec is a BLOCK codec (k_time="
                f"{model.k_time}) and no block map was passed. Embedding it "
                "one bin at a time would silently produce embeddings of a "
                "different thing than it was trained on.")
        T = len(blk_rows)                      # the BLOCK axis is the axis
        # SCALE THE EMBED BATCH BY THE TOKEN COUNT. The default 8192 was
        # sized for per-bin samples (C+2 = 42 encoder tokens); a k_time-7
        # block is 282 tokens and attention activations grow as B*tokens^2,
        # so the same batch is ~45x the memory — #466 (E-047-HEAD, the first
        # dispatch to reach a block embed) OOMed a 24 GB card on exactly
        # this line's downstream encode (1.10 GiB alloc, 314 MiB free).
        # B * tokens^2 held constant: 8192 * (42/282)^2 = 181 for k_time 7.
        # Per-bin codecs take the factor-1 branch and are bit-unchanged.
        tok_bin = X.shape[3] + 2
        tok_blk = model.k_time * X.shape[3] + 2
        batch = max(64, int(batch * (tok_bin / tok_blk) ** 2))
    if t_sel is None:
        ts = np.arange(T)
    else:
        ts = np.asarray(t_sel, dtype=np.int64)
        if cache_path:
            raise ValueError(
                "embed_everything: t_sel and cache_path are mutually "
                "exclusive — a time-subset cache is indistinguishable from a "
                "complete one by shape, so the next run would silently resume "
                "from an array that is missing most of its months.")
    T_out = len(ts)
    coords = np.stack([lats[ys] / 90, lons[xs] / 180], 1).astype(np.float32)
    # Z is T*P*d_z*4 bytes — 4.6 GB on the global grid at d_z=64, next to a
    # 1.4 GB tensor and a 0.3 GB mask. Built in RAM it OOM-kills a 7 GB box
    # (twice on 2026-08-07), and it is written to disk immediately afterwards
    # anyway. So it is BUILT in the cache file through a memmap: pages are
    # written as they are filled and the kernel may evict them, which turns a
    # hard 4.6 GB allocation into page-cache pressure. Reads go the same way.
    # Without a cache path (the --max-pixels smoke) it stays an ordinary array.
    # A CACHE WITH A `.progress` BESIDE IT IS A PREFIX, NOT A CACHE. `pull`
    # writes a published partial under the cache's final name (that is the
    # name every reader knows), so without this check the branch below would
    # find a full-length array of exactly the right shape and hand back one
    # whose tail is zeros — the #462 family of failure, arriving through the
    # feature that was meant to save the four hours. With no marker at all and
    # no `.done`, nothing has changed: a full-shape cache is taken as before.
    if cache_path and os.path.exists(cache_path):
        rows = _read_publish_progress(cache_path, T_out, P, d_z)
        if rows is not None:
            _adopt_published_partial(cache_path, rows)
    if cache_path and os.path.exists(cache_path):
        out = np.load(cache_path, mmap_mode="r+")
        if out.shape == (T_out, P, d_z):
            print(f"  (cached: {cache_path})")
            # ATTEST TO IT, so this box can publish what it just decided to
            # train on. The marker says "a full-shape cache stood here and
            # something checked it"; this branch has checked the one thing
            # that can be checked without re-embedding — that its shape is
            # the shape THIS run wants, T included. Writing it here is what
            # lets a cache built before the marker existed (E-044's 16.24 GiB
            # pentad Z, single-copy on one rented disk) ever reach the
            # release, and a strided rebuild would fail this shape test and
            # take the branch below instead.
            _mark_done(cache_path)
            return out, coords
    start_t = 0
    if cache_path and _cache_plan(cache_path,
                              T_out * P * d_z * CACHE_DTYPE(0).itemsize):
        tmp = cache_path + ".partial"
        # RESUMABLE. The embedding is ~95 minutes; dying at 80% and starting
        # again from zero is the difference between losing twenty minutes and
        # losing eighty. The half-written memmap already holds every completed
        # month — what was missing was a record of how many, so a restart
        # could trust it. Chris asked for exactly this on 2026-08-10.
        out, start_t = _resume_partial(tmp, T_out, P, d_z)
        if out is None:
            out = np.lib.format.open_memmap(tmp, mode="w+", dtype=CACHE_DTYPE,
                                            shape=(T_out, P, d_z))
    else:
        # NOT resumable, and say so rather than let it be discovered at 80%:
        # an in-memory array dies with the process. Since the cache went to
        # float16 this branch should be rare — 5.2 GiB fits where 10.4 did
        # not — and it is now a fallback rather than the normal path.
        cache_path = None                       # RAM path: nothing to publish
        print("  building Z in RAM: NOT resumable — if this process dies the "
              "whole embedding is lost. (Free disk so the cache fits and it "
              "becomes restartable.)", flush=True)
        out = np.zeros((T_out, P, d_z), dtype=CACHE_DTYPE)
    # THE EMBEDDING REPORTS ITS OWN PROGRESS. It is the longest single phase of
    # a stage-2 run — ~95 minutes for 43.5M encoder forwards on the
    # quarter-degree tensor — and until 2026-08-10 it printed one line when it
    # started and one when it finished. Actions will not serve logs for a
    # running job, so during that hour the only way to tell "working" from
    # "wedged" was to watch the box's resident memory climb as the array paged
    # in: a thermometer taped to the outside of the oven. Chris asked how far
    # along it was and the honest answer was an inference, which is not an
    # answer. Every 5% now costs one line and answers it directly.
    t_emb = time.time()
    next_mark = 0.0
    with torch.no_grad():
        for i_out in range(start_t, T_out):
            t = int(ts[i_out])
            frac = (i_out + 1) / T_out
            if frac >= next_mark:
                el = time.time() - t_emb
                eta = el / frac - el if frac > 0 else 0
                print(f"  embedding {frac * 100:5.1f}%  month {i_out + 1}/{T_out}  "
                      f"{el / 60:.0f} min elapsed, ~{eta / 60:.0f} min left",
                      flush=True)
                if progress:
                    progress({"pct": round(frac * 100, 1), "month": i_out + 1,
                              "months": T_out, "elapsed_s": round(el),
                              "eta_s": round(eta),
                              "where": "disk" if cache_path else "ram"})
                next_mark = frac + 0.05
            for i in range(0, P, batch):
                sl = slice(i, min(i + batch, P))
                n = sl.stop - sl.start
                ctx = np.concatenate([np.tile(ctx_all[t], (n, 1)), coords[sl]], 1)
                mk = torch.zeros(n, C, dtype=torch.bool)
                if mask_chan is not None:
                    mk[:, mask_chan] = True
                patch = getattr(model, "patch", 1)
                ctx_t = torch.as_tensor(ctx, dtype=torch.float32).to(dev)
                if patch > 1:
                    from model import gather_px
                    tt = torch.full((n,), t, dtype=torch.long)
                    v, o = gather_px(X, OBS, tt, torch.as_tensor(ys[sl]),
                                     torch.as_tensor(xs[sl]), patch)
                    z = model.encode((v * (~mk).unsqueeze(-1)).to(dev),
                                     o.to(dev), mk.to(dev), ctx_t)
                elif getattr(model, "k_time", 1) > 1:
                    # E-047 BLOCK CODEC. `t` indexes the BLOCK axis and
                    # `blk_rows[t]` names its source rows; the grid is
                    # assembled at FULL VISIBILITY exactly as the per-bin path
                    # is, with pad cells forced unobserved.
                    rr = blk_rows[t]
                    v = torch.stack([X[int(r), ys[sl], xs[sl]] for r in rr], 1)
                    o = torch.stack([OBS[int(r), ys[sl], xs[sl]] for r in rr], 1)
                    o = o & torch.as_tensor(~blk_pad[t])[None, :, None]
                    mkg = torch.zeros_like(o)
                    z = model.encode((v * ~mkg).to(dev), o.to(dev),
                                     mkg.to(dev), ctx_t)
                else:
                    v = X[t, ys[sl], xs[sl]] * (~mk)
                    z = model.encode(v.to(dev), OBS[t, ys[sl], xs[sl]].to(dev),
                                     mk.to(dev), ctx_t)
                out[i_out, sl] = z.cpu().numpy()
            # Every 8 months (~1.5 minutes of work) flush the pages and record
            # the count. Cheap enough to be unnoticeable, fine-grained enough
            # that a crash costs a couple of minutes rather than an hour.
            if cache_path and (i_out + 1) % 8 == 0:
                _mark_progress(tmp, out, i_out + 1, T_out, P, d_z)
    if cache_path:
        # Already on disk in .npy form — flush the pages, then publish
        # atomically so an interrupted run never leaves a half-filled cache
        # that the shape check would happily accept next time.
        out.flush()
        del out
        os.replace(tmp, cache_path)
        # FLUSH, THEN MARK (CLAUDE.md §5.21) — one line further than before.
        # The atomic rename already made "a file at cache_path" mean "a
        # finished embedding" to THIS file; it means nothing to the publisher,
        # which sees only bytes on a disk and cannot tell a finished pass from
        # an abandoned one: open_memmap allocates the full (T, P, d_z) shape
        # up front, so a run killed at month 900 of 3142 leaves a cache of
        # exactly the right length, dtype and T with zeros in the tail. Every
        # check embed_cache_sync.py can make would pass it. This mark is the
        # only thing that says the last month was written, and it is written
        # after the flush and after the rename so it can only under-claim.
        _mark_done(cache_path)
        # …AND THE PROGRESS MARKERS GO. Both of them describe a state that has
        # just stopped being true: the local one names a `.partial` that no
        # longer exists (the next run would try to resume a missing file), and
        # the published one would tell a puller to fetch a PREFIX of a cache
        # that is complete on the release. `.done` and `.progress` are
        # mutually exclusive statements about the same file and must never be
        # readable at the same time.
        try:
            os.remove(_progress_path(tmp))
        except OSError:
            pass
        _clear_publish_progress(cache_path)
        out = np.load(cache_path, mmap_mode="r+")
    return out, coords



def season_ctx(months, mode="month", d=None):
    """[T, 2] season features for the HEAD, sin/cos of the year phase.

    `month` (the default, and every archived run) is
    sin/cos(2*pi*(month-1)/12) — an INTEGER month-of-year, so on a pentad axis
    all ~6 bins inside a month carry one identical token and the forcing is a
    staircase. It is the array `ctx_all` has always been, and the caller
    asserts that equality rather than trusting this docstring.

    `fine` is the continuous phase of the bin's OWN date: the centre of the
    bin, as a fraction of the tropical year from that year's 1 January. At a
    binned cadence the date comes from the tensor's own `bin_index`, `epoch`
    and `pentad_days` — the same three members `ml/rollout_spatial.py`'s
    TimeAxis derives its calendar from — so a 5-day bin advances the token by
    5/365.2425 of a turn instead of by nothing five times and 1/12 once. At
    monthly there is no `bin_index`, so a bin is its calendar month and the
    centre is the month's midpoint: close to the month-quantized value but
    NOT equal to it, deliberately (a month's token stops being the value at
    its start and becomes the value at its middle).

    Only the head sees this. The frozen codec's own context is built from the
    month-quantized array at the embed call and must stay that way: Z exists
    already and was produced with it.
    """
    moy = np.array([int(m[5:7]) - 1 for m in months])
    if mode == "month":
        return np.stack([np.sin(2 * np.pi * moy / 12),
                         np.cos(2 * np.pi * moy / 12)], 1)
    if mode != "fine":
        raise ValueError(f"season_ctx: unknown mode {mode!r}")
    if d is not None and "bin_index" in d:
        days = (int(np.asarray(d["pentad_days"]).item()) if "pentad_days" in d
                else {"pentad": 5, "daily": 1}[str(d["cadence"])])
        ep = (dt.date.fromisoformat(str(d["epoch"])) if "epoch" in d
              else dt.date(1982, 1, 1))
        spans = [(ep + dt.timedelta(days=int(b) * days), float(days))
                 for b in np.asarray(d["bin_index"]).astype(np.int64)]
    else:
        # monthly: the bin IS the calendar month, so its span is that month
        spans = []
        for m in months:
            y, mm = int(m[:4]), int(m[5:7])
            first = dt.date(y, mm, 1)
            nxt = dt.date(y + (mm == 12), (mm % 12) + 1, 1)
            spans.append((first, float((nxt - first).days)))
    return np.array([season_feat_of(s, n) for s, n in spans])


def season_feat_of(start, span_days):
    """(sin, cos) of the year phase at the CENTRE of a bin.

    ONE definition, used by both ends of the system: `season_ctx` builds the
    trainer's [T, 2] table from it, and `ml/rollout_spatial.py` calls it per
    ROW so a rolled step past the end of the record is phased the same way as
    a step inside it. Sub-day arithmetic is done in datetimes on purpose —
    `date + timedelta(days=2.5)` silently truncates to 2 days, which would
    put a pentad's centre half a day early and a February's a half day late.
    """
    c = (dt.datetime.combine(start, dt.time())
         + dt.timedelta(days=float(span_days) / 2.0))
    frac = ((c - dt.datetime(c.year, 1, 1)).total_seconds()
            / (365.2425 * 86400.0))
    return math.sin(2 * math.pi * frac), math.cos(2 * math.pi * frac)


def eval_forward(model, zseq, mseq, sctx, eps_dim=0):
    """A read-out forward: the legacy 3-argument call when the head is
    deterministic, the REPRESENTATIVE member when it is an FGN head. One
    function so the member choice lives in one place (fgn_eval_eps)."""
    if not eps_dim:
        return model(zseq, mseq, sctx)
    return model(zseq, mseq, sctx,
                 eps=fgn_eval_eps(zseq.shape[0], eps_dim, zseq.device))


def _chunked_forward(model, zseq, mseq, sctx, dev, chunk=4096, eps_dim=0):
    """The eval forwards used to push 20,000 windows through the model in ONE
    call. That fit every 576x8 head ever trained and OOM-killed the first
    768x12 within seconds of finishing 60k healthy training steps (E-027
    #285/#286, 2026-08-14: "tried to allocate 5.9 GB, 1.1 GB free", two
    different boxes, run green, no temporal.json — the classic backgrounded
    silent death). Same numbers, chunked: concatenation over disjoint slices
    is exact for a pointwise forward."""
    preds, hids = [], []
    for i in range(0, len(zseq), chunk):
        sl = slice(i, i + chunk)
        p_, h_ = eval_forward(model, zseq[sl].to(dev), mseq[sl].to(dev),
                              sctx[sl].to(dev), eps_dim)
        preds.append(p_.cpu()); hids.append(h_.cpu())
    return torch.cat(preds), torch.cat(hids)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="pilot4_anom")
    ap.add_argument("--K", type=int, default=24, help="context length (months)")
    ap.add_argument("--steps", type=int, default=4000)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--d-model", type=int, default=96)
    ap.add_argument("--layers", type=int, default=3)
    ap.add_argument("--lr-schedule", default="cosine",
                    choices=["cosine", "invsqrt", "wsd", "expdecay"],
                    help="cosine bakes the TOTAL step count into the rate, so "
                         "a checkpoint's schedule only means anything next to "
                         "the budget it was trained under (this is what makes "
                         "a resumed run read lr=0.0 and what forces every "
                         "budget to be its own experiment). invsqrt is "
                         "horizon-free: lr(s) depends only on s, so a run "
                         "stopped at 60k and continued to 200k sees exactly "
                         "what an uninterrupted 200k run would have. Default "
                         "stays cosine because every existing result used it.")
    ap.add_argument("--lr-halflife", type=float, default=40000,
                    help="expdecay only: steps for the rate to halve. ABSOLUTE, "
                         "not a fraction of --steps, which is what makes the "
                         "schedule horizon-free: extend the run and the curve "
                         "it already walked is unchanged.")
    ap.add_argument("--lr-cooldown-frac", type=float, default=0.1,
                    help="wsd only: fraction of --steps spent decaying "
                         "linearly to zero at the end. The stable phase before "
                         "it is horizon-free, so a run can be extended; the "
                         "cooldown is what makes the result CONVERGED rather "
                         "than a reading at step N.")
    ap.add_argument("--lr-warmup", type=int, default=2000,
                    help="invsqrt only: steps to reach the peak, after which "
                         "lr = peak * sqrt(warmup / step)")
    ap.add_argument("--init-temporal", default="",
                    help="WARM RESTART: take the WEIGHTS of a stage-2 head and "
                         "train --steps more with a fresh cosine at --lr. "
                         "Adam's moments, the schedule position and the RNG "
                         "stream are NOT inherited, because the published "
                         "heads do not carry them — every checkpoint written "
                         "before 2026-08-10 is {args, model} only. This is a "
                         "separate flag from --resume-temporal on purpose: "
                         "the two produce different trajectories, and the one "
                         "mistake worth engineering against is reporting a "
                         "warm restart as though it were a continuation. Here "
                         "--steps is the EXTRA, not the total.")
    ap.add_argument("--resume-temporal", default="",
                    help="continue a stage-2 head: a path, or a tag under "
                         "/opt/earth-cache/ckpt (e.g. run-112-temporal). The "
                         "checkpoint carries model, optimiser, scheduler, step "
                         "and RNG state, so the continuation is the SAME "
                         "trajectory rather than a fresh run that happens to "
                         "start from these weights.\n\n"
                         "NOTE the schedule semantics: --steps is the TOTAL, "
                         "not the extra. Resuming a 60,000-step head with "
                         "--steps 200000 fast-forwards a 200,000-step cosine "
                         "to step 60,000 and carries on. The original head "
                         "annealed to ~0 over its own 60,000, so its LR steps "
                         "back UP — a warm restart, which is a different "
                         "object from a single 200,000-step run and must be "
                         "labelled as such when the numbers are compared.")
    ap.add_argument("--seed", type=int, default=0,
                    help="torch/numpy seed (sweeps need more than one)")
    ap.add_argument("--tag", default="",
                    help="suffix for output files: temporal_<tag>.json/.pt")
    ap.add_argument("--ring-km", default="0",
                    help="E-023: put the non-centre slots on a circle of this "
                         "radius in KM instead of using the fixed STENCILS "
                         "table. Measured by ml/measure_ring_info.py: the "
                         "incremental information a neighbour carries peaks "
                         "at 167-222 km (3x the touching neighbours'), because "
                         "at one cell the neighbour's embedding correlates "
                         "0.97 with the centre's and is nearly a copy of it. "
                         "A COMMA LIST makes concentric rings sharing the "
                         "slots equally: '222,555' with --stencil 17 is eight "
                         "points at each radius, the outer ring rotated half a "
                         "sector off the inner one (E-026).")
    ap.add_argument("--stencil", type=int, default=1,
                    help="E-022 SPATIAL INPUT: predict the centre pixel's "
                         "z_{t+1} from this many neighbourhood pixels' z per "
                         "step (1 = the original per-pixel model; 9 = 3x3; "
                         "13 = 2 in each cardinal direction, 1 on each "
                         "diagonal). Missing neighbours (land/window edge) "
                         "are zero-filled with static observed-flags. "
                         "Incompatible with --unroll>1 and --direct: both "
                         "feed predictions back, and a random pixel batch "
                         "does not contain the neighbours' predictions.")
    ap.add_argument("--unroll", type=int, default=1,
                    help="AUTOREGRESSIVE UNROLL DEPTH in the loss. 1 (default) "
                         "is the original teacher-forced t+1 objective. >1 "
                         "feeds the model's OWN prediction back in for this "
                         "many extra steps and backpropagates through the "
                         "chain, which is the standard fix for EXPOSURE BIAS: "
                         "a model trained only on true context never sees its "
                         "own errors, so they compound at rollout — and "
                         "rollout horizon is a headline claim of this "
                         "programme (rollout.py), measured on a model that "
                         "was never trained for it. Costs one extra forward "
                         "and backward per extra step.")
    ap.add_argument("--input-znoise", type=float, default=0.0,
                    help="E-029b: Gaussian noise (std, in normalised z units) "
                         "added to the LIVE input slots during training — the "
                         "cheap alternative to --unroll for the same exposure "
                         "bias. At roll time every input is a model "
                         "prediction, not an observation; training on clean "
                         "context therefore mismatches the roll's input "
                         "distribution, and E-026b measured the penalty "
                         "growing with slot count. Calibrate to the model's "
                         "own one-step error: sigma = sqrt(val_zmse) of the "
                         "matching clean run (big55: sqrt(0.55) ~ 0.74 -> "
                         "0.7). DEAD slots stay exact zeros — zero IS the "
                         "dead-slot encoding (test_zero_weight_equivalence), "
                         "and the roll feeds exact zeros there too, so "
                         "noising them would train against an input state "
                         "the roll never produces. Reaches temporal.py "
                         "through the window's sched: tail, which the "
                         "workflow hands over verbatim — no workflow edit.")
    ap.add_argument("--fgn-eps", type=int, default=0,
                    help="E-057: dimension k of the GLOBAL noise vector "
                         "eps ~ N(0,1)^k that conditions every layer's "
                         "LayerNorm (FGN, arXiv:2506.10772; adaLN-zero, so "
                         "the eps path is the identity at init and the head "
                         "IS the deterministic incumbent bitwise at step 0). "
                         "0 (DEFAULT) = OFF = the exact legacy code path: no "
                         "module is constructed, no RNG draw is made and no "
                         "record grows a key, so every archived stage-2 "
                         "number stays bit-reproducible. "
                         "WHEN k > 0 THE TRAINING OBJECTIVE SWITCHES from MSE "
                         "to the FAIR CRPS AT N=2 (two forwards per batch on "
                         "identical context with eps1 != eps2). That is one "
                         "change, not two, and it is not optional: under a "
                         "squared-error loss the conditional MEAN is optimal, "
                         "so a noise-conditioned head trained on MSE learns to "
                         "IGNORE eps and the whole arm silently becomes its "
                         "own control. FGN's own default is k=32. Refuses in "
                         "combination with --direct, --unroll>1 and "
                         "--unroll-wide (each feeds a prediction back, and "
                         "which member is fed back is an unmade decision); "
                         "--milestone-steps is fine. Reaches temporal.py "
                         "through the window's sched: tail, which the workflow "
                         "hands over verbatim — no workflow edit. Plan: "
                         "ml/plans/E057_fgn_head.md")
    ap.add_argument("--fgn-val-members", type=int, default=8,
                    help="E-057: ensemble size M for the IN-TRAINING "
                         "monitoring reads (stage2_val_crps, "
                         "stage2_val_member_var, stage2_val_spread_ratio, and "
                         "the ensemble mean behind stage2_val_zmse/stage2_amp). "
                         "Only read when --fgn-eps > 0. The eval eps bank is "
                         "drawn ONCE from its own generator, seeded seed*"
                         "1000003+58, so the training eps stream cannot depend "
                         "on how often the monitor runs. Must be >= 2: a "
                         "one-member 'ensemble' has no member variance and no "
                         "spread, which are the two numbers this monitor "
                         "exists to make visible (eps collapse is E-057's F2 "
                         "falsifier and must be readable on the live branch, "
                         "not discovered post hoc).")
    ap.add_argument("--grad-clip", type=float, default=0.0,
                    help="E-044: max global gradient 2-norm, applied with "
                         "torch.nn.utils.clip_grad_norm_ between backward() "
                         "and opt.step(). 0.0 (DEFAULT) = OFF, and OFF MAKES "
                         "NO CALL AT ALL — the pre-2026-08-21 code path "
                         "exactly — so every archived stage-2 number stays "
                         "bit-reproducible and no monthly dispatch has to "
                         "opt out of anything. WHY IT EXISTS: #423 (E-044, "
                         "the first stage-2 at pentad cadence) diverged from "
                         "its step-2,000 best with grad norm 8.24 -> 787 -> "
                         "3,891 -> 13,052 and two full blow-ups, in a "
                         "trainer where grep -n 'clip_grad' returned "
                         "NOTHING. WHY A CLIP WHEN THE OPTIMISER IS ADAM: "
                         "Adam bounds the update per COORDINATE, not the "
                         "damage per STEP. One outlier batch spikes m and v "
                         "together, |m/(sqrt(v)+eps)| goes to ~1 in every "
                         "coordinate at once, and the parameter vector moves "
                         "by ~lr*sqrt(N) = 1e-3*sqrt(206.5M) = 14.4 in one "
                         "step; v then stays inflated for ~1/(1-beta2) = "
                         "1,000 steps, during which honest gradients are "
                         "scaled to nothing. That is the shape #423 shows — "
                         "a spike, thousands of steps of partial recovery, "
                         "another spike. SIZE IT FROM THE MEASURED "
                         "DISTRIBUTION: the monthly stage-2 archive is 8,080 "
                         "logged grad norms over 83 runs on the ml-metrics "
                         "branch (all at val_persistence 3.09512), median "
                         "0.566, p99 4.30, and MAXIMUM 39.6165 (#308; #221 "
                         "is 35.01). 128.0 is therefore 3.23x every monthly "
                         "norm ever recorded — 0 of 8,080 would have been "
                         "clipped, where a threshold of 32 would have "
                         "clipped 2 — while sitting 15.5x above the healthy "
                         "PENTAD norm (8.24/8.25 at steps 2,000/4,000, so it "
                         "does not bind a healthy pentad run either) and "
                         "6.15x below the smallest pentad excursion (787.2). "
                         "Reaches temporal.py through the window's sched: "
                         "tail, which the workflow hands over verbatim.")
    ap.add_argument("--frame-offsets", default="",
                    help="E-053.1: give each CONTEXT FRAME its own time "
                         "offset, in bins, relative to the window anchor t "
                         "(the frame whose target is the headline t+1). "
                         "EMPTY (default) = the implicit contiguous stencil "
                         "[-(K-1) .. -1, 0] and the literal pre-2026-08-26 "
                         "code path, so every archived number stays "
                         "bit-reproducible. Set it to a comma list — "
                         "'-146,-145,-73,-72,-6,-5,-4,-3,-2,-1,0' — that is "
                         "STRICTLY INCREASING, all <= 0, and ENDS AT 0. K is "
                         "then DERIVED as len(offsets) and overrides --K, "
                         "because the number of frames is no longer free once "
                         "their times are named; the printed 'frame offsets:' "
                         "line states the derived K and the span. WHY THE "
                         "LAST MUST BE 0: the head reads hidden(-1) and its "
                         "headline target is t+1, so a last frame before t "
                         "would move the forecast step and silently change "
                         "the persistence denominator every archived ratio is "
                         "quoted against — E-053's whole point is that only "
                         "the CONTEXT sampling moves. Each frame's own "
                         "teacher-forced target is the bin after ITS OWN "
                         "time, and its season token comes from its own time. "
                         "Pool bound generalises t >= K-1 to t >= -offsets[0], "
                         "so a long span buys fewer windows; that is correct. "
                         "The learned position embedding IS the delta-t "
                         "encoding here — position <-> offset is a bijection "
                         "at a FIXED pattern per run — so no new parameters "
                         "exist. Refuses in combination with --unroll>1, "
                         "--unroll-wide, --time-stride>1 and --direct (all of "
                         "them reach past the window along a contiguous axis "
                         "this list no longer has). Reaches temporal.py "
                         "through the window's sched: tail, which the "
                         "workflow hands over verbatim — no workflow edit. "
                         "PASS IT AS ONE WORD, --frame-offsets=-5,-2,0: the "
                         "list starts with a minus, and a bare '-5,-2,0' in "
                         "the next argv slot is read by argparse as an option "
                         "string, not a value (the --train-lon-hold trap). "
                         "Plan: ml/plans/E053_spacetime_stencil.md 4.")
    ap.add_argument("--unroll-wide", type=int, default=0,
                    help="E-030: unrolled training for WIDE stencils via "
                         "one-hop self-generated context (Chris, 2026-08-15: "
                         "'we need to just predict the inputs to a given "
                         "pixel — not all pixels, just the ones that are the "
                         "input to the next stage'). Plain --unroll is "
                         "incompatible with --stencil>1 because the model "
                         "predicts only its centre pixel, while its t+1 input "
                         "window needs the NEIGHBOURS' t+1 embeddings too. "
                         "But those are exactly S depth-1 predictions from "
                         "OBSERVED context: for each of the pixel's S slot "
                         "positions, forward that slot-pixel's own observed "
                         "window one step (detached, no grad), assemble the "
                         "centre pixel's t+1 input window from the S "
                         "predictions, and take a second, differentiable "
                         "step scored against Z[t+2], weighted 1/2 like "
                         "--unroll's u=2 term. Reach-independent: the cost is "
                         "S extra no-grad forwards per unrolled pixel, "
                         "whatever the ring radius, because depth-1 needs no "
                         "context beyond each slot-pixel's own window. Only "
                         "the value 2 is supported — U=3 would need S^2 "
                         "depth-1 forwards plus S depth-2 assemblies "
                         "(S^(U-1) growth). Requires --stencil>1; "
                         "incompatible with --unroll>1, --direct and "
                         "--unroll-probs. Applied to a sub-batch "
                         "(--uw-batch) of each step's windows. Rides the "
                         "sched: tail like --input-znoise.")
    ap.add_argument("--uw-batch", type=int, default=64,
                    help="E-030: how many of each step's windows get the "
                         "--unroll-wide 2 term. The one-hop pass costs S "
                         "no-grad forwards per window (S=55 for ring 4444), "
                         "so unrolling the full 512-window batch would be "
                         "~55x the base step; 64 of 512 keeps the overhead "
                         "near ~7x the plain forward while every step still "
                         "carries a depth-2 gradient signal.")
    ap.add_argument("--milestone-steps", default="",
                    help="E-031/E-032: comma list of steps at which to save a "
                         "WEIGHTS-ONLY milestone checkpoint "
                         "(temporal_ms<step>.pt in the run dir, riding the "
                         "probes artifact). Exists because a single 200k run "
                         "must retain its 60k/120k rungs (Chris, 2026-08-15) "
                         "and the xl tier cannot leg-and-resume: a >2 GiB "
                         "full head fits neither the release (2 GiB asset "
                         "cap) nor the snapshot path. Weights-only (~0.86 GB "
                         "at 206M) is enough for the corridor-AUC evals and "
                         "the ratio-vs-steps curve; it is NOT resumable — "
                         "the box mirror keeps the resumable state. Rides "
                         "the sched: tail.")
    ap.add_argument("--unroll-probs", default="",
                    help="comma probabilities for sampling the unroll depth "
                         "PER STEP, e.g. '0.5,0.25,0.125,0.125' with "
                         "--unroll 4: each training step draws U_t in "
                         "1..unroll from this distribution and unrolls that "
                         "far (Chris, 2026-08-11: 'probabilistically set U "
                         "to 1 (50%%), 2 (25%%), 3+4 (12.5%%)'). Rationale, "
                         "measured: fixed U=4 pays +28%% one-step z-MSE at "
                         "every seed while buying the nowcast probe +0.09; "
                         "sampling spends half the steps on the pure "
                         "one-step map and reaches full depth only "
                         "occasionally — the question is whether that keeps "
                         "the probe gain without the forecast cost. Length "
                         "must equal --unroll; empty = fixed depth.")
    ap.add_argument("--direct", default="",
                    help="comma list of DIRECT horizons, e.g. '3,6,12': one "
                         "extra linear head per horizon predicts z_{t+h} from "
                         "the hidden state at t in a single forward pass — "
                         "the standard direct-vs-iterated alternative to "
                         "rolling t+1 predictions forward. E-011 measured "
                         "iterated rollouts smoothing away exactly the "
                         "amplitude the AMOC probe needs; a direct head "
                         "cannot compound because it never iterates. Loss "
                         "adds mean-over-horizons MSE at the last window "
                         "position; empty = off, objective unchanged.")
    ap.add_argument("--max-pixels", type=int, default=0,
                    help="subsample ocean pixels (code-path smoke only; "
                         "the 26.5N section is always kept)")
    # ---- E-044c overnight instruments (2026-08-22). All four default to OFF
    # and OFF is the pre-existing code path, asserted rather than asserted-to:
    # tests/test_e044c_knobs.py pins bit-identity against the unflagged run
    # the way tests/test_e044_grad_clip.py pins --grad-clip 0.
    ap.add_argument("--time-stride", type=int, default=0,
                    help="SUBSAMPLE THE TIME AXIS: keep bins "
                         "range(--time-offset, T, N). 0 (default) keeps every "
                         "bin and is today's path exactly. At pentad N=6 is "
                         "~one bin per month, so a K of 24 kept bins is 24 "
                         "MONTHS of context instead of 120 days — the E-044 "
                         "question 'is the pentad collapse about cadence or "
                         "about context span' asked directly, at 1/6 the "
                         "windows. Everything keyed on the axis (months, moy, "
                         "t_hold, the season features, the pool, the monitor, "
                         "every eval) is subsampled TOGETHER; persistence "
                         "baselines are recomputed on the kept axis by "
                         "construction, so a strided val_persistence is the "
                         "one-step change OF THE STRIDED AXIS and is not "
                         "comparable with a full-axis one")
    ap.add_argument("--time-offset", type=int, default=0,
                    help="phase of --time-stride (0 <= O < N). At pentad, "
                         "offset 2 is the mid-month bin, which is the one "
                         "carrying Argo (n_rg_live 252/3142, measured "
                         "2026-08-22)")
    ap.add_argument("--holdout-scope", default="window",
                    choices=HOLDOUT_SCOPES,
                    help="WHAT THE YEAR HOLDOUT ACTUALLY EXCLUDES from the "
                         "training pool. 'endpoint_contaminated' IS THE "
                         "LEGACY POOL AND IT LEAKS: a window is dropped only "
                         "when a SCORED bin (t+1, the unroll fan, each "
                         "--direct horizon) falls in a holdout year, but the "
                         "loss is DENSE over the window — win_ztgt scores the "
                         "bin after EVERY frame — so every window straddling "
                         "the edge of a holdout year teacher-forces that "
                         "year's measured transitions into the weights and "
                         "reads its bins as context besides. It is kept for "
                         "ONE reason, to reproduce the 98 archived stage-2 "
                         "runs that trained under it, and it is never the "
                         "default again. 'window' (DEFAULT) is the strict "
                         "rule: eligible only if NONE of the bins the forward "
                         "pass touches — frames, per-frame targets, scored "
                         "reach — is held out; it costs 13.03%% of the "
                         "frame-targets on the pentad axis. 'target' is the "
                         "minimal correct fix: the legacy pool bin for bin, "
                         "with every per-frame loss term whose TARGET bin is "
                         "held out masked out of the mean — no held-out bin "
                         "is ever a target, held-out bins MAY still be read "
                         "as context, and it costs 5.25%% of the "
                         "frame-targets. The monitor batch and both z-space "
                         "evals are UNTOUCHED at every setting")
    ap.add_argument("--holdout-years", default=None,
                    help="comma list of years to hold out, OVERRIDING the "
                         "codec checkpoint's own `args['holdout_years']` for "
                         "this stage-2 run. E-067, 'the two-year roll': a "
                         "roll is truncated at the end of the held-out "
                         "stretch it started in, so with SINGLE held-out "
                         "years no lead past 365 d can ever be scored. "
                         "Holding out CONSECUTIVE years — "
                         "--holdout-years 2008,2009,2016,2017,2022,2023, "
                         "three two-year blocks — moves the truncation to "
                         "the end of the block and makes 730 d scoreable. "
                         "REFUSED unless it is a SUPERSET of the codec's own "
                         "years: a year the codec trained on may be held out "
                         "at stage 2, never the reverse. Denying MORE years "
                         "costs training windows and the certificate prints "
                         "how many. IT MOVES BOTH MASKS — `t_hold` feeds the "
                         "training pool AND the anomaly transform's "
                         "statistics, and the two are one array here (unlike "
                         "the LONGITUDE holdout, which is deliberately two: "
                         "see the TWO MASKS note in main). Holding more "
                         "years out of the z-score statistics is the "
                         "conservative direction — fewer bins the encoder's "
                         "normalisation has seen, never more. The EFFECTIVE "
                         "list (this override, or the codec's) is written "
                         "into the saved head's own args and into "
                         "stage2_config, so a roll can read the years the "
                         "head was actually denied")
    ap.add_argument("--target-bins-argo", default="all",
                    choices=("all", "exclude", "only"),
                    help="filter the TRAINING POOL by whether a window's "
                         "SCORED bins carry Argo. A bin carries Argo iff any "
                         "rg_* channel has a finite value in it — measured, "
                         "8.02%% of pentad bins (one per month, the mid-month "
                         "stamp). 'all' (default) is today's pool exactly. "
                         "'exclude' drops every window whose target reach "
                         "(t+1..t+UF and each --direct horizon) touches an "
                         "Argo bin; 'only' keeps just those. THE POOL AND "
                         "NOTHING ELSE: the monitor batch, both z-space "
                         "evals and the transport probes are untouched, so "
                         "the ratios stay comparable across arms")
    ap.add_argument("--season-dropout", type=float, default=0.0,
                    help="probability, per TRAINING window, that the season "
                         "features are replaced by zeros (the neutral value "
                         "for a sin/cos pair: no point on the unit circle, so "
                         "the month projection contributes nothing). 0 = off "
                         "= today. Eval, monitor and roll paths NEVER drop — "
                         "this is a regulariser against the calendar lock the "
                         "unforced rolls show, not a change of protocol")
    ap.add_argument("--season-phase", default="month",
                    choices=("month", "fine"),
                    help="what the HEAD's season features mean. 'month' "
                         "(default, and every archived run) is "
                         "sin/cos(2*pi*moy/12) with an INTEGER month-of-year, "
                         "so all ~6 pentad bins inside a month share one "
                         "identical token — a staircase forcing on a 5-day "
                         "axis. 'fine' uses the continuous fraction-of-year "
                         "phase of each bin's TRUE date, derived from "
                         "bin_index/epoch/pentad_days. THE CODEC'S OWN ctx "
                         "STAYS MONTH-QUANTIZED either way: Z is frozen and "
                         "was built that way, so this changes the head's "
                         "conditioning only. It is written into the "
                         "checkpoint args and ml/rollout_spatial.py reads it "
                         "back, so a fine-phase head can never be rolled with "
                         "coarse tokens")
    ap.add_argument("--input-quant", default="",
                    help="FSQ-style scalar quantization of every z the HEAD "
                         "READS — the cheap form of the capacity hypothesis "
                         "(Chris, 2026-08-22; arxiv 2309.15505). '' (default) "
                         "is off and bit-identical. '8' quantizes every "
                         "dimension to 8 levels; '8,8,6,5,...' names one level "
                         "count per dimension (exactly d_z of them). INPUTS "
                         "ONLY: the training TARGET stays continuous, so the "
                         "objective is still quantized-state -> true-next-"
                         "state and every z-space MSE stays comparable with "
                         "the archive. It rides the checkpoint args, and "
                         "ml/rollout_spatial.py applies the same quantizer at "
                         "roll time — quantized input is part of the model's "
                         "contract, not a training trick")
    ap.add_argument("--data", default=os.path.join(HERE, "cache", "na_pixels.npz"),
                    help="tensor npz (family-3 runs pass family3_na025.npz)")
    ap.add_argument("--train-lon-hold", default="inherit",
                    help="which longitudes are EXCLUDED FROM THE STAGE-2 "
                         "TRAINING POOL. 'inherit' (default, and the only "
                         "behaviour before 2026-08-19) = whatever the frozen "
                         "codec was trained under, read from its own args; "
                         "'none' = no longitude is excluded, train on the "
                         "whole basin and hold out YEARS only; 'lo,hi' = an "
                         "explicit block. THIS GOVERNS THE POOL ONLY — never "
                         "the anomaly-transform statistics, which always "
                         "follow the codec. See the call sites for why the "
                         "two cannot be moved together. PASS AN EXPLICIT "
                         "BLOCK AS --train-lon-hold=lo,hi, ONE WORD — a bare "
                         "'-45,-25' in the next argv slot is read by argparse "
                         "as an option string, not a value.")
    a = ap.parse_args()

    # A PRECONDITION THAT DEPENDS ONLY ON THE INPUTS IS CHECKED WHILE THE
    # INPUTS ARE ALL IT HAS COST (ml/CLAUDE.md §0.3). A negative max_norm is
    # not "off": clip_grad_norm_ would scale every gradient by a negative
    # coefficient and flip the descent direction, and nothing downstream
    # would say so for the ten hours it took to notice.
    if a.grad_clip < 0:
        sys.exit(f"--grad-clip {a.grad_clip} must be >= 0 (0 = off, and off "
                 f"makes no clip_grad_norm_ call at all).")

    # ---- E-057 · the FGN refusals, at argv time ---------------------------
    # Same placement rule as the clip above and for the same reason: all of
    # these depend only on argv, so they cost nothing here and cost a rented
    # box's whole embedding pass anywhere later (ml/CLAUDE.md §0.3, §5.16).
    if a.fgn_eps < 0:
        sys.exit(f"--fgn-eps {a.fgn_eps} must be >= 0 (0 = off, and off is "
                 f"the legacy code path exactly).")
    if a.fgn_eps > 0:
        # --fgn-val-members is read ONLY in fgn mode, so it is checked only
        # there: refusing a legacy dispatch over a knob it never consults
        # would be a behaviour change with the flag off, which is the one
        # thing this whole diff may not do.
        if a.fgn_val_members < 2:
            sys.exit(f"--fgn-val-members {a.fgn_val_members} must be >= 2: "
                     f"member variance and spread are undefined at M=1, and "
                     f"they are the two numbers the fgn monitor exists to "
                     f"show (eps collapse is E-057's F2 falsifier).")
        # Each of the three below feeds a PREDICTION back into the model's own
        # input, and under an ensemble head "the prediction" is not one object
        # — which member is fed back is a design decision E-057.0 has not made.
        # A run that merely left them alone would train, produce numbers, and
        # answer a question nobody asked (ml/CLAUDE.md §4.11).
        if a.direct.strip():
            sys.exit(f"--fgn-eps {a.fgn_eps} is incompatible with --direct "
                     f"{a.direct!r}: a direct head reads hidden(-1) of ONE "
                     f"member and is scored by MSE, which is exactly the "
                     f"objective the fgn arm exists to replace. One mechanism "
                     f"per arm.")
        if a.unroll != 1:
            sys.exit(f"--fgn-eps {a.fgn_eps} is incompatible with --unroll "
                     f"{a.unroll}: the unroll feeds the model's own prediction "
                     f"back as the next step's input, and WHICH MEMBER gets "
                     f"fed back (and whether eps is resampled per step, as "
                     f"FGN's rolls do) is a decision E-057.0 does not make. "
                     f"Train fgn arms at U=1.")
        if a.unroll_wide > 0:
            sys.exit(f"--fgn-eps {a.fgn_eps} is incompatible with "
                     f"--unroll-wide {a.unroll_wide}: same reason as --unroll "
                     f"— the one-hop pass assembles a t+1 input window out of "
                     f"S neighbour PREDICTIONS, and an ensemble head has no "
                     f"single prediction to assemble from.")

    # ---- E-053.1: --frame-offsets, the sunflower taken into TIME ----------
    # Same placement rule as the clip above: the whole block depends only on
    # argv, so it runs before the tensor is opened and before one second of
    # GPU is rented (ml/CLAUDE.md §0.3, §5.16).
    FOFF = None
    if a.frame_offsets.strip():
        try:
            FOFF = tuple(int(x) for x in a.frame_offsets.split(",")
                         if x.strip())
        except ValueError:
            sys.exit(f"--frame-offsets {a.frame_offsets!r}: every entry must "
                     f"be an integer number of bins.")
        if len(FOFF) < 2:
            sys.exit(f"--frame-offsets {a.frame_offsets!r}: needs at least 2 "
                     f"frames (got {len(FOFF)}). A one-frame context is not a "
                     f"sequence and the causal attention has nothing to do.")
        if any(o > 0 for o in FOFF):
            sys.exit(f"--frame-offsets {a.frame_offsets!r}: every offset must "
                     f"be <= 0. A positive offset reads the future — the one "
                     f"thing the whole protocol is built to make impossible.")
        if any(nxt <= cur for cur, nxt in zip(FOFF, FOFF[1:])):
            sys.exit(f"--frame-offsets {a.frame_offsets!r}: must be STRICTLY "
                     f"INCREASING — oldest first, no duplicates. The causal "
                     f"mask and the position embedding both read the frame "
                     f"order as time order, so an unsorted list trains a head "
                     f"whose attention is wrong in a way nothing reports.")
        if FOFF[-1] != 0:
            sys.exit(f"--frame-offsets {a.frame_offsets!r}: the LAST offset "
                     f"must be 0 (got {FOFF[-1]}). The head reads hidden(-1) "
                     f"and its headline target is t+1; a last frame before t "
                     f"moves the forecast step and silently changes the "
                     f"persistence denominator every archived ratio is quoted "
                     f"against. E-053 moves the CONTEXT, never the target.")
        # THE COMBINATION REFUSALS. Each of these reaches PAST the window
        # along an axis that a contiguous stencil supplies and an offset list
        # does not — and each would run, and produce a number, if it were
        # merely left alone. Refuse at argument time instead.
        if max(1, a.unroll) > 1:
            sys.exit(f"--frame-offsets is incompatible with --unroll "
                     f"{a.unroll}: the unroll feeds its own prediction back "
                     f"as the NEXT CONTIGUOUS BIN's input, and a "
                     f"non-contiguous frame list has no such slot to feed. "
                     f"Train offset arms at U=1 (E-010/E-020 closed the "
                     f"unroll axis anyway).")
        if a.unroll_wide:
            sys.exit(f"--frame-offsets is incompatible with --unroll-wide "
                     f"{a.unroll_wide}: the wide unroll re-gathers each "
                     f"neighbour's OWN contiguous window to generate its "
                     f"one-hop prediction, which is a window shape this flag "
                     f"has just redefined. One mechanism per arm.")
        if a.time_stride > 1:
            sys.exit(f"--frame-offsets is incompatible with --time-stride "
                     f"{a.time_stride}: the stride resamples the time axis "
                     f"under the offsets, so '-73' would mean 73 KEPT bins "
                     f"and not 73 bins of the tensor the list was written "
                     f"against. Two time surgeries at once is one too many.")
        if a.direct.strip():
            sys.exit(f"--frame-offsets is incompatible with --direct "
                     f"{a.direct!r}: the direct heads score horizons past the "
                     f"window's END, and the pool guard that keeps those bins "
                     f"train-months is written against a contiguous reach. "
                     f"Run the offset arms at t+1 only.")
        # DERIVED K, and it OVERRIDES --K rather than being checked against
        # it: once the frame TIMES are named the frame COUNT is no longer a
        # free parameter, and a dispatch that carries both (recipes always
        # set --K) would otherwise have to keep the two in sync by hand —
        # the copying exercise that produced #395 and #387.
        if a.K != len(FOFF):
            print(f"--frame-offsets overrides --K {a.K} -> {len(FOFF)}")
        a.K = len(FOFF)
        print(f"frame offsets: K={len(FOFF)} {list(FOFF)} "
              f"span={-FOFF[0]} bins", flush=True)

    torch.manual_seed(a.seed)
    np.random.seed(a.seed)
    # ---- E-057 · the eps streams ------------------------------------------
    # TWO generators, neither of them the global RNG, and they are separate on
    # purpose. `eps_gen` feeds TRAINING draws; it is a CPU generator and every
    # draw is made on the CPU and then moved, so the eps stream is
    # device-independent and a run resumed on a different box sees the same
    # noise. It is saved into every checkpoint and restored by
    # --resume-temporal, exactly as `torch_rng` is, because an eps stream that
    # restarts on resume is a different experiment wearing a continuation's
    # name (tests/test_resume_temporal.py's whole argument, one stream over).
    # The eval bank has its OWN generator (…+58) so that how often the monitor
    # runs cannot perturb the training stream.
    # With --fgn-eps 0 nothing is created and no draw is ever made: the global
    # RNG sequence, and therefore every archived number, is untouched.
    FGN = a.fgn_eps > 0
    eps_gen = None
    if FGN:
        eps_gen = torch.Generator()
        eps_gen.manual_seed(a.seed * 1000003 + 57)

    def _eps_state():
        """The eps stream's state for a checkpoint dict — {} when fgn is off,
        so a legacy artefact is byte-for-byte what it always was."""
        if eps_gen is None:
            return {}
        return {"eps_gen": eps_gen.get_state().numpy().tolist()}

    run_dir = os.path.join(HERE, "runs", a.run)
    ck = torch.load(os.path.join(run_dir, "pixelmae.pt"),
                    map_location="cpu", weights_only=False)
    if not ck["args"].get("anomaly"):
        sys.exit("stage 2 requires an anomaly-space codec (train.py --anomaly): "
                 "state-space embeddings failed the K-sweep precondition.")
    # load_tensor == np.load for every single-file npz (families 2/3/4); for
    # family 5's sidecar layout it MEMORY-MAPS X, because 165.6 GB cannot be
    # decompressed on any box we can rent (ml/tensor_io.py). The old
    # `np.load(...)["X"].copy()` could not open family 5 at all, and on a
    # single-file npz the .copy() was pure waste: NpzFile decompresses a FRESH
    # writable array on every `d["X"]`, so the copy simply held the tensor
    # twice (33 GB at pentad) — the residency probe_kfold.py already dropped.
    from tensor_io import load_tensor
    d = load_tensor(a.data)
    X = d["X"]
    if isinstance(X, np.memmap) and not X.flags.writeable:
        # Sidecar tensor (family 5): anomaly_transform writes into X, and so
        # does the nan_to_num below. The canonical map must never take those
        # writes — it would leave an anomaly-space tensor where a state-space
        # one is documented and the NEXT run would z-score it again, silently.
        # A per-run scratch copy is disk, not RAM (tensor_io docstring).
        from tensor_io import writable_copy
        scratch = a.data[:-4] + "_temporal_scratch.npy"
        X = writable_copy(X, scratch, verbose=False)
        import atexit
        atexit.register(lambda q=scratch: os.path.exists(q) and os.remove(q))
    months = [str(m) for m in d["months"]]
    lats, lons, chan = d["lats"], d["lons"], [str(c) for c in d["chan"]]
    T, H, W, C = X.shape
    moy = np.array([int(m[5:7]) - 1 for m in months])
    # E-067 · WHICH YEARS ARE HELD OUT, AND WHO SAID SO.
    #
    # The default is what it has always been: the CODEC checkpoint's own
    # `args["holdout_years"]`. `--holdout-years` may hold out MORE — the
    # two-year-block protocol needs consecutive years so a roll's truncation
    # moves from the year end to the block end — and is REFUSED if it holds
    # out less, because a year the codec was fitted on that stage 2 then
    # scores is contamination in the direction nothing downstream can detect.
    # The guard is here, where the inputs are all it has cost (§5.16), and
    # not after the anomaly transform's six traversals of X.
    #
    # ONE MASK, NOT TWO, and that is deliberate. `t_hold` below feeds BOTH
    # the anomaly transform's climatology/z-score statistics AND the training
    # pool (`build_window_pool`), unlike the LONGITUDE holdout, which is
    # split into `stat_x_hold`/`pool_x_hold` for the reason spelled out in
    # the next comment. The override moves both, and that is the safe
    # direction: withholding MORE years from the statistics can only shrink
    # the set of bins the frozen encoder's normalisation was derived from —
    # it never lets a held-out year into them, which is what the split
    # exists to prevent. The print says so rather than leaving it to be
    # inferred.
    ck_hold_years = sorted(ck["args"]["holdout_years"].split(","))
    if a.holdout_years is not None and str(a.holdout_years).strip():
        eff_hold_years = sorted({y.strip()
                                 for y in str(a.holdout_years).split(",")
                                 if y.strip()})
        missing = [y for y in ck_hold_years if y not in set(eff_hold_years)]
        if missing:
            sys.exit(
                f"--holdout-years {a.holdout_years!r} is not a superset of "
                f"the codec's own holdout years "
                f"({ck['args']['holdout_years']!r}): {', '.join(missing)} "
                f"would re-enter the stage-2 training pool despite the codec "
                f"having been trained WITHOUT "
                f"{'it' if len(missing) == 1 else 'them'}. A year the codec "
                f"trained on may be held out at stage 2; the reverse is "
                f"contamination. Refusing.")
        hold_src = f"--holdout-years {a.holdout_years!r}"
    else:
        eff_hold_years = ck_hold_years
        hold_src = f"codec args {ck['args']['holdout_years']!r}"
    # THE EFFECTIVE LIST TRAVELS WITH THE CHECKPOINT. `args` in every saved
    # head is `vars(a)`, so writing it back onto the namespace here is what
    # puts the years a head was actually denied into the head's own file —
    # where ml/rollout_spatial.py can read them and warn if a roll is scored
    # on a different set.
    a.holdout_years = ",".join(eff_hold_years)
    hold_years = set(eff_hold_years)
    t_hold = np.array([m[:4] in hold_years for m in months])
    print(f"holdout years: {a.holdout_years} — from {hold_src} · "
          f"{int(t_hold.sum())} of {T} bins held out. ONE MASK: this drives "
          f"the training pool AND the anomaly transform's statistics "
          f"(holding more years out of the statistics is the conservative "
          f"direction — see the TWO MASKS note, which is about LONGITUDE). "
          f"The --holdout-scope certificate below counts the windows this "
          f"leaves.", flush=True)
    # TWO MASKS, AND THEY ARE NOT THE SAME OBJECT (2026-08-19).
    #
    # `x_hold` was one variable doing two jobs: it chose the anomaly
    # transform's z-score statistics (just below) AND it chose the stage-2
    # training pool (`ok_p`, far below). Chris's decision — hold out years
    # only, train on every longitude — is about the POOL. Moving the
    # statistics with it would be a silent covariate shift on a FROZEN
    # encoder: the codec was fitted on inputs normalised over
    # (train years x non-held-out longitudes), and re-deriving mu/sd over a
    # 33%-larger pool rescales every pixel it is then asked to encode, in a
    # run whose whole point is that the codec is unchanged.
    #
    # It is also unsafe in a way that would not show up as a bad number.
    # The embedding cache Z is keyed by (codec weight hash, sha256 of the
    # RAW tensor file) — embed_cache_sync.cache_name — and neither term sees
    # the transform. Two runs on the same codec and the same tensor with
    # different statistics would therefore share one cache key: whichever
    # pulled would train on the other's embeddings, and every shape, dtype
    # and length check would pass.
    #
    # So: `stat_x_hold` ALWAYS follows the codec's own saved args, and
    # --train-lon-hold governs `pool_x_hold` alone. The eval side is
    # untouched either way — every evaluation pool in this file keys on
    # t_hold only (search `ev_t`/`ev_m`), so `inherit` vs `none` differ in
    # exactly one thing, which is what makes the comparison an isolation of
    # the stage-2 half of the effect rather than two changes at once.
    from train import lon_holdout_mask     # lazy: the one parser for the spec
    stat_x_hold = lon_holdout_mask(ck["args"]["holdout_lon"], lons)
    _tlh = str(a.train_lon_hold).strip()
    if _tlh.lower() == "inherit":
        pool_x_hold = stat_x_hold
    else:
        pool_x_hold = lon_holdout_mask(_tlh, lons)
    print(f"lon holdout · statistics (codec "
          f"{ck['args']['holdout_lon']!r}): "
          f"{int(stat_x_hold.sum())}/{len(lons)} cols · training pool "
          f"(--train-lon-hold {a.train_lon_hold!r}): "
          f"{int(pool_x_hold.sum())}/{len(lons)} cols", flush=True)

    # THE anomaly transform, and there is exactly one of it. What stood here
    # was a hand-inlined THIRD copy (train.py had the second until 2026-08-17,
    # probe_sequence.py the fourth), frozen at the pre-2026-08-17 shape, and
    # it carried both bugs the canonical one has since had fixed:
    #
    #   1. `v.std()` with no `dtype=np.float64`. numpy upcasts the accumulator
    #      for np.mean on float16 but NOT for np.std/np.var. The z-score sums
    #      ~204M squared residuals; in float16 that passes 65504, returns inf,
    #      and (X - mu) / (inf + 1e-6) is EXACTLY 0.0. Families 4 (pentad) and
    #      5 (daily) are float16, so stage 2 on either would have trained on
    #      all-zero dynamic channels while every loss, gpu_util and probe still
    #      read healthy. Family 3 is float32 and never reached the limit, which
    #      is the only reason this copy never produced a wrong number.
    #   2. ~249 full-extent strided traversals of X (39 for the dynamic test,
    #      ~6 per dynamic channel). At family 5's 165.6 GB on a 64 GB box that
    #      is ~41 TB of physical read: run #389 sat SEVEN HOURS in the
    #      equivalent code in trainprobe.py with the GPU at 0%. The canonical
    #      version is time-chunked at 6.0 traversals.
    #
    # Duplication was the defect, not either bug — one fix landed in one file
    # and the other three kept the broken arithmetic. tests/test_one_anomaly_
    # transform.py now fails if a second implementation reappears anywhere in
    # ml/. The import is LAZY because trainprobe imports this module.
    from trainprobe import anomaly_transform
    # stat_x_hold, NOT pool_x_hold — see the two-masks comment above.
    X, dynamic = anomaly_transform(X, moy, t_hold, stat_x_hold)

    # ---- --time-stride: subsample the axis, and mind the ORDER ------------
    # WHERE THIS SITS IS THE WHOLE DESIGN, and it is not where a first reading
    # would put it.
    #
    #   * AFTER the anomaly transform, never before. The transform's
    #     climatology and per-channel z-score are computed over the TRAIN
    #     bins of the FULL axis; recomputing them over one bin in six would
    #     shift every normalised value the frozen encoder is then asked to
    #     encode — a covariate shift on a codec whose whole premise is that
    #     it is unchanged. Worse, it would be INVISIBLE: the embed cache is
    #     keyed by (codec weight hash, sha256 of the RAW tensor file) and
    #     neither term sees the transform OR the stride, so a strided run
    #     would publish embeddings of a differently-normalised tensor under
    #     the name every other run pulls. That is #10/#11 with a new cause.
    #   * BEFORE nan_to_num/isfinite, which is where the memory goes: those
    #     two build a float16 copy and a bool of the SAME shape as X (34 GB
    #     and 17 GB at pentad, an ~85 GB peak that is the first place this
    #     file can die). Slicing first makes both of them 1/N.
    #
    # The cost of being safe: the embedding cannot use the shared cache,
    # because a [T/N, P, d_z] array is indistinguishable by shape from a
    # complete one for a different tensor — embed_everything refuses t_sel
    # with cache_path for exactly this reason. A strided run therefore
    # re-embeds its own kept bins (1/N of the pass) and writes nothing.
    tsel = None
    if a.time_stride:
        if a.time_stride < 1:
            sys.exit(f"--time-stride {a.time_stride}: must be >= 1")
        if not (0 <= a.time_offset < a.time_stride):
            sys.exit(f"--time-offset {a.time_offset} must satisfy "
                     f"0 <= O < N for --time-stride {a.time_stride}")
        # The ocean mask is a property of the FULL record and must not become
        # a property of the sample: computed here, channel 0 only, chunked, so
        # a pixel observed only in dropped bins still counts as ocean exactly
        # as it does in an unstrided run.
        ocean_full = np.zeros(X.shape[1:3], bool)
        for i0 in range(0, T, 64):
            ocean_full |= np.isfinite(X[i0:i0 + 64, :, :, 0]).any(axis=0)
        tsel = np.arange(a.time_offset, T, a.time_stride)
        X = np.ascontiguousarray(X[tsel])
        months = [months[i] for i in tsel]
        moy, t_hold = moy[tsel], t_hold[tsel]
        T = len(tsel)
        print(f"--time-stride {a.time_stride} offset {a.time_offset}: "
              f"{T} of {len(tsel) * a.time_stride + a.time_offset} bins kept "
              f"({months[0]}..{months[-1]}), held-out {int(t_hold.sum())} · "
              f"K={a.K} now spans {a.K} KEPT bins · the embed cache is "
              f"DISABLED for this run (a strided Z must never be published "
              f"under an unstrided name)", flush=True)

    # THE RAPID TRUTH IS KEYED ON THE AXIS ROW, so it is subsampled with the
    # axis or it points at the wrong bins. `rapid[:, 0]` holds row indices of
    # the FULL record (build_family4's truth_pentad writes exactly that), and
    # both readers below — the in-training transport probe and eval 3 — index
    # `moy`/`t_hold` with them. Unstrided this is `d["rapid"]` itself, so
    # nothing moves; strided, a row that survives is renumbered to its
    # position in the kept axis and a row that does not is dropped.
    rapid_arr = d["rapid"]
    if tsel is not None:
        _pos = {int(r): i for i, r in enumerate(tsel)}
        _keepr = [i for i, r in enumerate(rapid_arr[:, 0].astype(int))
                  if int(r) in _pos]
        rapid_arr = rapid_arr[_keepr].copy()
        rapid_arr[:, 0] = [_pos[int(r)] for r in rapid_arr[:, 0].astype(int)]
        print(f"  RAPID truth on the strided axis: {len(rapid_arr)} of "
              f"{len(d['rapid'])} rows survive", flush=True)

    # ---- E-047: a BLOCK codec makes the block axis the axis ---------------
    # k_time comes from the CODEC, not from a flag here: the head consumes
    # whatever the frozen encoder emits, and a block codec emits one z per
    # block. In `month` mode the block labels are `YYYY-MM`, so everything
    # downstream — TimeAxis, the roll's horizon and bands, the persistence
    # baselines — reads a MONTHLY axis that was built entirely from 5-day
    # data, which is E-047's whole point.
    BLKA = None
    if int(ck["args"].get("k_time", 1) or 1) > 1:
        tb = str(ck["args"].get("time_block", "") or "")
        if not tb:
            sys.exit("the codec has k_time > 1 but no `time_block` in its "
                     "args: this checkpoint cannot say how its blocks were "
                     "cut, and guessing would embed a different grouping "
                     "than it was trained on.")
        if a.time_stride:
            sys.exit("--time-stride on a BLOCK codec: two time surgeries at "
                     "once. The blocks already re-cut the axis; striding it "
                     "as well would leave an axis no artefact describes.")
        from timeblocks import BlockAxis
        BLKA = BlockAxis(tb, months,
                         d["bin_index"] if "bin_index" in d else None,
                         (dt.date.fromisoformat(str(d["epoch"]))
                          if "epoch" in d else None),
                         (int(np.asarray(d["pentad_days"]).item())
                          if "pentad_days" in d else None))
        print(BLKA.describe(C, ck["d_z"]), flush=True)
        rapid_arr = BLKA.remap_rows(rapid_arr)
        months = list(BLKA.labels)
        moy = np.array([int(m[5:7]) - 1 for m in months])
        t_hold = np.array([
            t_hold[BLKA.rows[b, :int(BLKA.n_bins[b])]].any()
            for b in range(BLKA.n_blocks)])
        T = BLKA.n_blocks
        print(f"  block axis: T {T} · held out {int(t_hold.sum())} · RAPID "
              f"rows {len(rapid_arr)} · labels {months[0]}..{months[-1]}",
              flush=True)

    codec = codec_from_ckpt(ck, C)
    codec.load_state_dict(ck["model"])
    codec.eval()

    # The CODEC's context stays month-quantized whatever --season-phase says:
    # Z is frozen and was built with this exact array (embed_everything reads
    # ctx_all), so changing it here would silently re-key nothing and
    # re-encode everything. --season-phase governs the HEAD's `Mt` only,
    # built further down from season_ctx().
    ctx_all = np.stack([np.sin(2 * np.pi * moy / 12), np.cos(2 * np.pi * moy / 12)], 1)
    # ...AND FOR A BLOCK CODEC THAT ARRAY IS NOT WHAT THE CODEC WAS TRAINED
    # WITH (E-048, fixing an E-047 gap). `ml/train.py` sets
    # `a.ctx_mode = "block_phase"` and feeds the encoder `BLK.ctx_phase()` —
    # the CONTINUOUS fraction-of-year phase of the block's own centre — and
    # `ml/rollout_spatial.py` re-encodes with `BLKR.ctx_phase()` for the same
    # reason. Embedding here with the month-quantized token instead fed the
    # frozen encoder a context it had never seen, and the two consumers of one
    # codec disagreed with each other.
    #
    # It is a rounding error in month mode and a STRUCTURAL error at stride 3:
    # two windows a fortnight apart share a calendar month, so the
    # month-quantized token is IDENTICAL across a step the axis takes, and the
    # encoder would be told the year had not moved. The context the codec
    # reads is therefore the block axis's own, always; `ctx_all` stays the
    # month array and stays the HEAD's default season token, which is what
    # keeps `--season-phase month` bit-identical below.
    codec_ctx = ctx_all if BLKA is None else BLKA.ctx_phase()
    Xt = torch.from_numpy(np.nan_to_num(X, nan=0.0))
    OBS = torch.from_numpy(np.isfinite(X))
    # X is dead from here (everything downstream reads Xt/OBS), and the
    # embedding array alone is T*P*d_z*4 ~ 4.6 GB at global scale — so the
    # 1.4 GB anomaly copy has to go before it is allocated. Likewise `ocean`
    # comes from OBS rather than d["X"][..., 0], which would load the whole
    # 1.4 GB npz member again just to slice one channel off it.
    ocean = OBS[..., 0].any(axis=0).numpy() if tsel is None else ocean_full
    # The per-bin Argo mask, for --target-bins-argo. A bin carries Argo iff
    # any rg_* channel has a finite value in it — the definition the tensor
    # itself implies (`n_rg_live` counts exactly these bins) and the one
    # measured on 2026-08-22: 252 of 3,142 pentad bins, one per month, at the
    # mid-month stamp. Cheap here because OBS already exists; computed always
    # so the LOG can report it even when the filter is off.
    # PER BIN, one at a time: `OBS[..., rg]` in one expression is a
    # [T,H,W,35] bool — 14.9 GB at pentad — which is the shape of allocation
    # this file exists to avoid.
    _rg = [i for i, nm in enumerate(chan) if str(nm).startswith("rg_")]
    argo_bin = np.zeros(T, bool)
    if _rg:
        _rgt = torch.as_tensor(_rg, dtype=torch.long)
        for _t in range(T):
            argo_bin[_t] = bool(OBS[_t].index_select(-1, _rgt).any())
    print(f"Argo-carrying bins: {int(argo_bin.sum())}/{T} "
          f"({100.0 * argo_bin.mean():.2f}%) over {len(_rg)} rg_* channels",
          flush=True)
    del X
    import gc
    gc.collect()

    ys, xs = np.where(ocean)
    sec_y, sec_sel0 = rapid_section(lats, lons, ys, xs)
    if a.max_pixels and a.max_pixels < len(ys):
        rng = np.random.default_rng(0)
        keep = rng.choice(len(ys), a.max_pixels, replace=False)
        keep = np.union1d(keep, sec_sel0)                   # probe needs the section
        ys, xs = ys[keep], xs[keep]

    # The embedding pass is the only part worth a GPU here (18M encoder
    # forwards); stage-2 training is a small transformer for a few thousand
    # steps, and every eval below is numpy-bound. So the codec visits the
    # accelerator for the embedding and the static-identity pass, and comes
    # straight back to the CPU — leaving all downstream code untouched
    # rather than device-threaded, which is where the bugs would be.
    EDEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    codec.to(EDEV)
    print(f"embedding every (month, ocean pixel) through the frozen codec "
          f"on {EDEV.type} …")
    t0 = time.time()
    # The embed cache must be CODEC-AWARE: a bare Z_<run>.npy poisoned runs
    # #10/#11 (2026-08-07) — the Actions cache carried run #8's embeddings,
    # the (T, P, d_z) shape check matched, and both stage-2s trained on the
    # WRONG codec's z (healthy z-space skill, catastrophic decoded skill:
    # the z they predicted was not the z their decoder speaks). The weight
    # hash in the filename makes a stale cache a miss, never a lie.
    whash = codec_weight_hash(ck)
    # …AND DATA-AWARE, for the same reason one level up. Two boxes hold
    # family3_na025.npz files with different sha256s (b40f5b0b vs adcbe700,
    # measured 2026-08-11), so a codec-only key lets a box pull embeddings of
    # somebody else's tensor and pass every check it has.
    dhash = data_fingerprint(a.data)
    # …AND HOLDOUT-AWARE (E-067). Neither hash sees the ANOMALY TRANSFORM, and
    # `--holdout-years` moves it: `t_hold` chooses the climatology and the
    # z-score the frozen encoder is then asked to encode. Without this token a
    # warm box — or a pull of the published Z — would hand an overridden run
    # the embeddings of the codec's own years, and every shape, dtype and
    # length check would pass. Empty string with no override, so every
    # existing asset, path and glob is unchanged. See `hold_key`.
    hkey = hold_key(a.holdout_years, ck["args"].get("holdout_years", ""))
    print(f"  codec {whash} · tensor {dhash} ({os.path.basename(a.data)})"
          + (f" · holdout {hkey}" if hkey else ""), flush=True)
    if hkey:
        print(f"embed cache: --holdout-years moved the anomaly statistics, so "
              f"this run's Z is keyed '{hkey.lstrip('_')}' and is NEITHER "
              f"read from NOR published under the codec's own key. Pass "
              f"--hold-years {a.holdout_years} to ml/embed_cache_sync.py to "
              f"pull or push it.", flush=True)
    cache = (embed_cache_path(a.run, whash, dhash, hold=hkey)
             if not a.max_pixels else None)
    # …AND NOT AT ALL WHEN THE AXIS WAS SUBSAMPLED. The --time-stride block
    # above prints "the embed cache is DISABLED for this run (a strided Z must
    # never be published under an unstrided name)", and the comment beside it
    # says a strided run "re-embeds its own kept bins and writes nothing".
    # NEITHER WAS TRUE: the stride is applied by slicing X itself, so
    # embed_everything is reached with t_sel=None — its own refusal of
    # t_sel-with-cache_path never fires — and a [T/N, P, d_z] array was
    # written to, and read back from, the name every unstrided run pulls.
    # That is the file run #462 published: 8.72 GB of one-bin-in-two under the
    # 16.7 GB key. The check the message promised belongs here, where the
    # cache name is chosen; embed_cache_sync's --expect-t is the net under it,
    # for a Z that reaches the release by some other route.
    if cache and a.time_stride:
        print(f"  embed cache DISABLED: --time-stride {a.time_stride} keeps "
              f"{T} of the tensor's bins, and a Z of those bins is "
              f"indistinguishable by shape from a complete one for a smaller "
              f"tensor. This run embeds its own kept bins (1/{a.time_stride} "
              f"of the pass) and writes nothing.", flush=True)
        cache = None
    # A block codec's Z has a different SHAPE and a different axis from a
    # per-bin one. The weight hash already separates them (a block codec is a
    # different codec), so this is belt and braces — but a name that says
    # `blk` is one a human can triage on a full disk without loading it.
    if cache and BLKA is not None:
        # The MODE is in the name too (E-048): '6/6' and '6/3' produce
        # different axes from the same k_max and the same codec weights, and
        # the E-048 fix to the embed context (above) changes every block Z
        # written before it. A cache whose name cannot tell those apart is
        # the #10/#11 failure again — "a stale cache must be a miss, never a
        # lie". '/' is not a filename character, hence the '-'.
        cache = cache.replace(".npy", f"_blk{BLKA.k_max}-"
                              f"{str(BLKA.mode).replace('/', '-')}.npy")
    # Progress goes into the run's OWN metrics.jsonl, which the publisher loop
    # in ml-train.yml pushes to ml-live-<n> every five minutes. A print reaches
    # the log, and Actions will not serve the log of a running job — so during
    # the hour this takes, stdout is write-only. The side channel is the only
    # one anybody can read while it matters.
    def _emb_note(rec):
        try:
            os.makedirs(os.path.join(HERE, "runs", a.run), exist_ok=True)
            with open(os.path.join(HERE, "runs", a.run, "metrics.jsonl"), "a") as f:
                f.write(json.dumps({"embedding": rec}) + "\n")
        except OSError:
            pass                      # instrumentation never breaks the run

    Z, coords = embed_everything(codec, Xt, OBS, codec_ctx, lats, lons, ys, xs,
                                 ck["d_z"], cache_path=cache,
                                 progress=_emb_note,
                                 blk_rows=(None if BLKA is None
                                           else BLKA.rows),
                                 blk_pad=(None if BLKA is None
                                          else BLKA.pad))
    P = len(ys)
    print(f"  Z [T={T} P={P} d_z={ck['d_z']}]  ({time.time() - t0:.0f}s)")

    # static identity of each pixel: codec embedding of static channels only
    with torch.no_grad():
        stat_obs = OBS[0].clone()
        for c in dynamic:
            stat_obs[..., c] = False
        zs = []
        for i in range(0, P, 8192):
            sl = slice(i, min(i + 8192, P))
            n = sl.stop - sl.start
            ctx = np.concatenate([np.zeros((n, 2), np.float32), coords[sl]], 1)
            if getattr(codec, "patch", 1) > 1:
                from model import gather_px
                # One gather: values from month 0 (statics are constant in t;
                # dynamics are zeroed inside encode because their obs is
                # False), obs from stat_obs with gather_px's own out-of-range
                # latitude masking — exactly what training-time encode saw.
                t0i = torch.zeros(n, dtype=torch.long)
                vv, oo = gather_px(Xt, stat_obs[None], t0i,
                                   torch.as_tensor(ys[sl]),
                                   torch.as_tensor(xs[sl]), codec.patch)
                zs.append(codec.encode(vv.to(EDEV), oo.to(EDEV),
                                       torch.zeros(n, C, dtype=torch.bool, device=EDEV),
                                       torch.as_tensor(ctx).to(EDEV)).cpu().numpy())
            elif BLKA is not None:
                # E-047: a block codec's static identity is the same question
                # asked of a GRID — the static channels repeated across the
                # first block's cells, with the pad cells unobserved. The
                # channels are time-invariant, so which block is arbitrary;
                # block 0 is chosen the way bin 0 is in the per-bin path.
                kt = BLKA.k_max
                vv = Xt[0, ys[sl], xs[sl]][:, None, :].expand(-1, kt, -1)
                oo = (stat_obs[ys[sl], xs[sl]][:, None, :]
                      .expand(-1, kt, -1).clone())
                oo &= torch.as_tensor(~BLKA.pad[0])[None, :, None]
                zs.append(codec.encode(
                    vv.to(EDEV), oo.to(EDEV),
                    torch.zeros(n, kt, C, dtype=torch.bool, device=EDEV),
                    torch.as_tensor(ctx).to(EDEV)).cpu().numpy())
            else:
                zs.append(codec.encode(Xt[0, ys[sl], xs[sl]].to(EDEV),
                                       stat_obs[ys[sl], xs[sl]].to(EDEV),
                                       torch.zeros(n, C, dtype=torch.bool, device=EDEV),
                                       torch.as_tensor(ctx).to(EDEV)).cpu().numpy())
        Zstat = np.concatenate(zs, 0)
    codec.to("cpu")          # everything below is CPU/numpy, unchanged
    # E-022: build the neighbourhood index once; static_ctx carries the
    # per-cell observed flags (time-invariant geometry) when stencil > 1.
    if a.stencil > 1:
        # slot counts are validated against the GEOMETRY rather than a
        # whitelist: the fixed table has entries for 9 and 13, while a ring
        # shape admits any count that divides evenly among its radii. The
        # whitelist had to be widened for every new design (17 for E-026's
        # two rings, 25 for its three) which is a sign it was the wrong guard.
        if _ring_on(a.ring_km):
            n_r = (1 if str(a.ring_km).startswith("spiral:")
                   else len([r for r in str(a.ring_km).split(",") if r.strip()]))
            if (a.stencil - 1) % n_r:
                raise SystemExit(
                    f"--stencil {a.stencil} gives {a.stencil - 1} ring slots, "
                    f"which does not divide among {n_r} radii "
                    f"({a.ring_km}). Use 1 + a multiple of {n_r}.")
        elif a.stencil not in STENCILS:
            raise SystemExit(
                f"--stencil {a.stencil} has no fixed-table entry "
                f"(have {sorted(STENCILS)}); pass --ring-km to place that "
                f"many slots on rings instead.")
        if max(1, a.unroll) > 1 or a.direct.strip():
            raise SystemExit(
                f"--stencil {a.stencil} is incompatible with --unroll>1 and "
                f"--direct: the unroll/direct paths feed predictions back, "
                f"and a random pixel batch does not contain the neighbours' "
                f"predictions. Train stencil arms at U=1 (E-010/E-020 closed "
                f"the unroll axis anyway) — or use --unroll-wide 2, which "
                f"GENERATES the neighbours' predictions itself from their own "
                f"observed windows (E-030).")
        NBR = build_stencil(ocean.shape[0], ocean.shape[1], ys, xs, a.stencil,
                            ring_km=a.ring_km, lats=lats)
        NBR_t = torch.as_tensor(NBR)
        obs_flags = (NBR >= 0).astype(np.float32)
        static_ctx = torch.as_tensor(
            np.concatenate([Zstat, coords, obs_flags], 1))
        print(f"stencil {a.stencil}"
              + (f" RING r={a.ring_km} km" if _ring_on(a.ring_km) else "")
              + f": input {a.stencil}x{ck['d_z']}+2 per step; "
              f"{int((NBR < 0).sum()):,} missing neighbour slots "
              f"of {NBR.size:,}")
    else:
        NBR_t = None
        static_ctx = torch.as_tensor(np.concatenate([Zstat, coords], 1))

    # ---- train pool: windows [t-K+1 .. t] whose TARGET month t+1 is a train
    # month and whose pixel is outside the longitude holdout. Windows may LOOK
    # at held-out months (persistence can too); they may never be SCORED on
    # them in training.
    Zt = torch.from_numpy(Z)
    # THE HEAD'S SEASON FEATURES, which are `ctx_all` itself in the default
    # 'month' mode — `np.array_equal` asserted below rather than trusted, so
    # the default path is the same numbers and not merely the same intent.
    # ON A BLOCK AXIS THE SEASON FEATURE COMES FROM THE BLOCK AXIS (E-048).
    # `season_ctx(months, mode, d)` reads `d`'s per-BIN `bin_index` in 'fine'
    # mode, and on a block axis that array has one row per SOURCE BIN while
    # `months` has one per BLOCK — so 'fine' produced a [T_src, 2] token for a
    # [n_blocks, ...] head. `head_season('month')` is the same sin/cos of the
    # same labels (asserted below, as before), so the default path does not
    # move; 'fine' becomes the block CENTRE's continuous phase, which is the
    # only one of the two that can tell two windows inside one month apart.
    head_ctx = (season_ctx(months, a.season_phase, d) if BLKA is None
                else BLKA.head_season(a.season_phase))
    if a.season_phase == "month":
        assert np.array_equal(head_ctx, ctx_all), \
            "season_ctx('month') must reproduce the archived sin/cos(2pi*moy/12)"
    else:
        _dd = np.abs(head_ctx - ctx_all).max()
        print(f"--season-phase fine: the head's season token is the "
              f"continuous fraction-of-year phase of each bin's TRUE date "
              f"(max |Δ| vs the month-quantized token {_dd:.4f}). The "
              f"CODEC's ctx stays month-quantized — Z is frozen.", flush=True)
    Mt = torch.as_tensor(head_ctx, dtype=torch.float32)
    # ---- --input-quant: the head's input alphabet -------------------------
    # sigma is measured ONCE, on TRAIN bins only, from the Z this run will
    # actually read. It is written into the checkpoint args beside the spec,
    # because the quantizer is part of the MODEL'S CONTRACT and not a training
    # trick: ml/rollout_spatial.py has to reproduce the same map at roll time,
    # and recomputing sigma there from a different slice of Z would hand a
    # head a grid it was never trained on.
    QIN = None
    if a.input_quant:
        _tr = np.where(~t_hold)[0]
        _rq = np.random.default_rng(0)
        _ti = np.sort(_rq.choice(_tr, min(256, len(_tr)), replace=False))
        _pi = np.sort(_rq.choice(Z.shape[1], min(1024, Z.shape[1]),
                                 replace=False))
        _samp = torch.from_numpy(
            np.asarray(Z[np.ix_(_ti, _pi)], dtype=np.float32)
            .reshape(-1, ck["d_z"]))
        _sig = _samp.std(0).clamp(min=1e-6).numpy()
        QIN = InputQuant(a.input_quant, _sig, ck["d_z"])
        a.input_quant_sigma = [round(float(v), 6) for v in _sig]
        print(QIN.describe(_samp), flush=True)

    def qz(z):
        """Every z that ENTERS the model goes through here, and nothing that
        leaves it does. Identity (the same object) when the knob is off."""
        return z if QIN is None else QIN(z)

    K = a.K              # E-053.1: already len(FOFF) when --frame-offsets set
    # ---- E-053.1 · the window's own geometry, in ONE place ----------------
    # CTX_BACK is how far the EARLIEST frame reaches behind the anchor: K-1
    # under the contiguous stencil, -offsets[0] under a list. Every pool
    # bound and every valid-t range below is written against this single
    # number, so the two paths cannot drift apart the way a second guard
    # written for one of them would.
    CTX_BACK = (K - 1) if FOFF is None else -FOFF[0]

    def win_ref(t):
        """What the three window gathers key off — see frame_ref()."""
        return frame_ref(t, K, FOFF)

    def win_mseq(ref):
        """Season features PER FRAME, from that frame's own time — a frame at
        t-73 is a different month from the anchor, and pretending otherwise
        would hand the head a calendar that contradicts its own z."""
        return torch.stack([Mt[ref + j] for j in frame_steps(K, FOFF)], 1)

    def win_ztgt(ref, p):
        """Teacher-forced target per frame: the embedding ONE BIN AFTER that
        frame's own time. Every token keeps predicting its own next step (the
        objective is unchanged in form); only the times the tokens sit at
        move. The last frame's target is t+1 exactly, as before."""
        return torch.stack([Zt[ref + j + 1, p]
                            for j in frame_steps(K, FOFF)], 1).float()

    # With --unroll U the loss reaches U months past the window, so the pool
    # must guarantee those months EXIST and are TRAIN months. Without this the
    # unrolled steps would either index off the end of the array or be scored
    # on the holdout — the second is the one that would not have crashed.
    # --direct extends the reach the same way: every scored offset (the
    # contiguous unroll fan AND each direct horizon) must exist and be a
    # train month. With --direct empty the set reduces to the old guard
    # exactly, so default arms keep the identical window pool.
    U = max(1, a.unroll)
    UW = a.unroll_wide
    if UW:
        # E-030 preconditions, checked before any GPU time is spent. The
        # asymmetric shape (plain --unroll for stencil 1, --unroll-wide for
        # stencil > 1) exists because the two need different machinery, not
        # because they answer different questions — see the flag's help.
        if UW != 2:
            raise SystemExit(
                f"--unroll-wide {UW}: only 2 is supported. Depth 3 needs "
                f"S^2 depth-1 forwards plus S depth-2 assemblies per window "
                f"(S^(U-1) growth) — implement it deliberately if E-030 at "
                f"depth 2 earns it, don't fall into it.")
        if a.stencil <= 1:
            raise SystemExit(
                "--unroll-wide requires --stencil>1: at stencil 1 the "
                "window IS the centre pixel's own history, and plain "
                "--unroll already does exactly this, cheaper.")
        if U > 1 or a.direct.strip() or a.unroll_probs.strip():
            raise SystemExit(
                "--unroll-wide is incompatible with --unroll>1, --direct "
                "and --unroll-probs: one exposure-bias mechanism per arm, "
                "or the ablation cannot attribute the effect.")
    UF = max(U, UW)     # how far past the window the loss reaches
    MILESTONES = {int(x) for x in a.milestone_steps.split(",") if x.strip()}
    if MILESTONES:
        dead = {m for m in MILESTONES if not (0 < m < a.steps)}
        if dead:
            # A milestone at or past --steps can never fire; refuse rather
            # than let a retention request silently retain nothing (§4.6).
            raise SystemExit(f"--milestone-steps {sorted(dead)} outside "
                             f"(0, {a.steps}) — those saves would never "
                             f"happen.")
        print(f"milestone checkpoints at steps {sorted(MILESTONES)}")
    UP = None
    if a.unroll_probs.strip():
        UP = np.array([float(x) for x in a.unroll_probs.split(",")])
        if len(UP) != U:
            raise SystemExit(f"--unroll-probs has {len(UP)} entries for "
                             f"--unroll {U}: one probability per depth 1..U")
        if (UP < 0).any() or not np.isclose(UP.sum(), 1.0, atol=1e-6):
            raise SystemExit(f"--unroll-probs must be non-negative and sum "
                             f"to 1 (got sum {UP.sum():.6f})")
        print(f"sampled unroll: P(U=1..{U}) = {list(UP)}")
    D = tuple(sorted({int(x) for x in a.direct.split(",") if x.strip()}))
    # UF, not U: --unroll-wide 2 scores the centre pixel at t+2, so t+2 must
    # exist and be a train month exactly as a plain U=2 would require.
    reach = sorted(set(range(1, UF + 1)) | set(D))
    # `t >= CTX_BACK` IS the old `t + 1 >= K` when FOFF is None (CTX_BACK is
    # K-1 there); with offsets it is `t + offsets[0] >= 0`, i.e. the earliest
    # frame must exist. It sits ON TOP of every existing condition — the
    # target-month holdout, the reach guard, the Argo filter below — so a
    # long span shrinks the printed train-window count and changes nothing
    # else about who is eligible.
    # `build_window_pool` runs THAT EXACT EXPRESSION and returns it
    # unmodified at `endpoint_contaminated` and at `target`; only
    # `--holdout-scope window` masks the result. It lives in one function
    # because ml/jaxport/train_stage2.py calls it too — two trainers, one
    # definition of what a head may learn from.
    ok_t = build_window_pool(T, t_hold, K, FOFF, reach, CTX_BACK,
                             scope=a.holdout_scope)
    # --target-bins-argo: the SAME reach, filtered by what the scored bins
    # carry. On the pentad axis 92% of bins have no Argo at all (measured:
    # 252/3142 carry it, one per month), so 'exclude' trains a head that
    # never scores a bin whose 35 rg_* channels are present, and 'only'
    # trains on nothing else. THE TRAIN POOL AND NOTHING ELSE — the monitor
    # batch below is built from `ev_m` (windows whose t+1 is a HOLDOUT bin,
    # 4,096 of them, fixed seed) and is deliberately left on the full
    # population so `val_zmse`, `val_persistence` and every ratio quoted off
    # them stay comparable across arms; the two z-space evals and the
    # transport probes key on t_hold alone and are likewise untouched.
    if a.target_bins_argo != "all":
        want = (a.target_bins_argo == "only")
        # `ok_t` has already excluded every t whose reach runs off the end;
        # this predicate must not index past it while computing a value that
        # `ok_t` will discard anyway.
        def _reach_argo(t, agg):
            return agg(argo_bin[t + r] for r in reach) \
                if t + reach[-1] < T else False
        hit = np.array([_reach_argo(t, all) if want
                        else not _reach_argo(t, any)
                        for t in range(T)])
        n_before = int(ok_t.sum())
        ok_t = ok_t & hit
        print(f"--target-bins-argo {a.target_bins_argo}: {int(ok_t.sum()):,} "
              f"of {n_before:,} eligible target bins survive the filter "
              f"(reach {reach}); the monitor and both evals are unfiltered",
              flush=True)
        if not ok_t.any():
            sys.exit(f"--target-bins-argo {a.target_bins_argo} leaves NO "
                     f"trainable window: {int(argo_bin.sum())}/{T} bins carry "
                     f"Argo and the scored reach is {reach}. Refusing to "
                     f"train on an empty pool.")
    # pool_x_hold, NOT stat_x_hold — --train-lon-hold governs exactly this
    # line and nothing else in the file.
    ok_p = ~pool_x_hold[xs]
    pool_t, pool_p = np.where(ok_t[:, None] & ok_p[None, :])
    pool_t = torch.as_tensor(pool_t, dtype=torch.long)
    pool_p = torch.as_tensor(pool_p, dtype=torch.long)
    print(f"train windows: {len(pool_t):,}")

    # --holdout-scope target · THE PER-(WINDOW, FRAME) LOSS MASK, built once
    # from `frame_target_keep` and from nothing else. The pool above is the
    # legacy one at this scope; what makes the scope correct is that every
    # per-frame term whose TARGET bin is held out is dropped from the mean
    # below. Indexed by ANCHOR t so `batch_windows` gathers rows with the very
    # `t` it drew the window from — no second layout to keep in step. Rows
    # outside the pool are never read and stay True.
    HOLD_MASK = (a.holdout_scope == "target")
    KEEP_T, HOLD_MASKED_FRAC = None, 0.0
    if HOLD_MASK:
        _kidx = np.where(ok_t)[0]
        _kk = frame_target_keep(_kidx, K, FOFF, t_hold)
        HOLD_MASKED_FRAC = float((~_kk).sum()) / float(max(1, _kk.size))
        _kfull = np.ones((T, int(_kk.shape[1])), bool)
        _kfull[_kidx] = _kk
        KEEP_T = torch.as_tensor(_kfull)
        print(f"  loss mask: {int((~_kk).sum()):,} of {int(_kk.size):,} "
              f"(window, frame) targets in the TRAIN pool are held out and "
              f"contribute nothing to l_base "
              f"(holdout_masked_frac {HOLD_MASKED_FRAC:.6f}).", flush=True)

    # E-053.1 · WHY THIS NEEDS NO NEW PARAMETERS. `k_max=K` sizes the learned
    # position embedding, and with a FIXED offset pattern per run the map
    # position <-> offset is a bijection: slot j is ALWAYS the frame at
    # t+offsets[j], in every window of every step. So `pos` already IS the
    # delta-t encoding the plan asks for, learned rather than prescribed, and
    # no separate Δt table is added. What DOES change is the table's height —
    # a derived K of 16 gives a 16-row `pos` where K=24 gives 24 — so the
    # head's parameter count SHIFTS with the offset list's length. Read the
    # count off the run's own `stage2_config.params_M`; never carry one over
    # from an arm at a different K.
    # E-057: the constructor argument is passed ONLY when the flag is on, so
    # the no-flag path is the literal pre-2026-08-27 call.
    model = (TemporalTransformer(d_z=ck["d_z"], d_model=a.d_model,
                                 n_layers=a.layers, k_max=K, direct=D,
                                 stencil=a.stencil, eps_dim=a.fgn_eps)
             if FGN else
             TemporalTransformer(d_z=ck["d_z"], d_model=a.d_model,
                                 n_layers=a.layers, k_max=K, direct=D,
                                 stencil=a.stencil))
    if FGN:
        print(f"FGN head: eps ~ N(0,1)^{a.fgn_eps} -> per-layer FiLM "
              f"(zero-init, so this model is the deterministic incumbent at "
              f"step 0), objective = fair CRPS at N=2, monitor ensemble M="
              f"{a.fgn_val_members}. eps stream seed {a.seed * 1000003 + 57} "
              f"(CPU, device-independent, saved in every checkpoint).",
              flush=True)
    # STAGE-2 TRAINING RUNS ON THE ACCELERATOR TOO.
    # The comment above this block used to say stage-2 training is "a small
    # transformer for a few thousand steps" and therefore not worth a GPU.
    # That was true when stage 2 was 4,000 steps. It is now 140,000 and
    # 200,000, and the premise expired without the code noticing — the same
    # shape of error as memmapping to disk because the box used to have 7 GB
    # of RAM.
    #
    # Measured before changing it, rather than assumed: at batch 256, K=24 and
    # a 1.824M head, the data gather off the memmap is 12.4 ms of a 725 ms
    # step. The model is 98% of the cost, so the accelerator is worth between
    # 5x and 20x on a run that otherwise takes a full day.
    #
    # The batch gather stays on the CPU — Z is a 5.2 GiB memmap and random
    # rows out of it belong where the pages are — and only the assembled
    # batch crosses, which is 256 x 25 x 64 fp32, about 1.6 MB.
    TDEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(TDEV)
    print(f"stage-2 head on {TDEV.type} "
          f"({sum(p_.numel() for p_ in model.parameters()) / 1e6:.3f}M params)",
          flush=True)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-4)
    sched = make_sched(opt, a)

    start_step = 0
    init_from = None
    if a.init_temporal:
        # WARM RESTART, named as one. Every stage-2 head published before
        # 2026-08-10 is {args, model}: no optimiser moments, no schedule
        # position, no RNG stream — measured on f3_s2_60k, f3_s2_24k and the
        # rescue mirrors, all three. So a "continuation" from any of them is
        # impossible, and --resume-temporal correctly refuses.
        #
        # What IS available is a cosine restart: take the converged weights,
        # train a fresh schedule at a lower peak. That is a real and standard
        # way to spend more compute on a trained model, and it answers "do
        # 140,000 more steps help?" — it simply is not the same trajectory a
        # straight-through 200,000-step run would have taken, so it must never
        # be plotted as a fourth point on a curve whose other points were each
        # their own converged cosine from scratch.
        #
        # --steps is the EXTRA here, not the total, precisely because there is
        # no step count in the checkpoint to be the total OF.
        ip = (a.init_temporal if os.path.sep in a.init_temporal
              else os.path.join(CKPT_DIR, a.init_temporal + ".pt"))
        if not os.path.exists(ip):
            raise SystemExit(
                f"--init-temporal: no checkpoint at {ip}. Refusing to start a "
                f"fresh head under a doc string that says the weights came "
                f"from somewhere.")
        tk = torch.load(ip, map_location="cpu", weights_only=False)
        if FGN:
            # E-057: THE WARM START THE KEY LAYOUT WAS DESIGNED FOR. An FGN
            # head's trunk keys are the stock encoder's, so a legacy
            # deterministic checkpoint drops straight in and only the eps path
            # is fresh — which is worth a day of GPU. strict=False is scoped to
            # this branch and its missing keys are CHECKED rather than
            # trusted: anything outside eps_embed/film means the two
            # architectures genuinely disagree and the load must refuse.
            _miss, _unexp = model.load_state_dict(tk["model"], strict=False)
            _bad = [k for k in _miss
                    if not (k.startswith("eps_embed.") or ".film." in k)]
            if _bad or _unexp:
                raise SystemExit(
                    f"--init-temporal {ip}: state dict does not match this "
                    f"head. missing (beyond the eps path): {_bad}; "
                    f"unexpected: {list(_unexp)}. Refusing a partial load.")
            if _miss:
                print(f"  eps path is fresh ({len(_miss)} tensors: "
                      f"eps_embed + per-layer film, zero-init), trunk warm "
                      f"from {os.path.basename(ip)}", flush=True)
        else:
            model.load_state_dict(tk["model"])
        parent_steps = int(tk.get("args", {}).get("steps", 0))
        parent_lr = tk.get("args", {}).get("lr")
        init_from = {"from": os.path.basename(ip),
                     "parent_steps": parent_steps, "parent_lr": parent_lr,
                     "extra_steps": a.steps, "lr": a.lr,
                     "inherited": ["model"],
                     "reset": ["optimiser moments", "schedule position",
                               "rng stream"],
                     "kind": "warm restart (cosine restart), NOT a continuation"}
        carried = [k for k in ("opt", "sched", "step") if k in tk]
        if carried:
            print(f"  note: {ip} DOES carry {carried} — --resume-temporal "
                  f"would give a true continuation and is the better choice",
                  flush=True)
        print(f"WARM RESTART from {ip}: weights of a {parent_steps:,}-step head "
              f"(peak lr {parent_lr}), now {a.steps:,} MORE steps on a fresh "
              f"cosine at peak {a.lr:.2e}. Adam's moments and the schedule "
              f"start from nothing — this is not the same trajectory as a "
              f"{parent_steps + a.steps:,}-step run and must not be reported "
              f"as one.", flush=True)
    _parent = {}
    if a.resume_temporal:
        rp = (a.resume_temporal if os.path.sep in a.resume_temporal
              else os.path.join(CKPT_DIR, a.resume_temporal + ".pt"))
        if not os.path.exists(rp):
            raise SystemExit(
                f"--resume-temporal: no checkpoint at {rp}. Refusing to start "
                f"a fresh head under a doc string that says 'continue' — that "
                f"is the mistake --require-resume exists to prevent on the "
                f"codec side.")
        tk = torch.load(rp, map_location="cpu", weights_only=False)
        model.load_state_dict(tk["model"])
        missing = [k for k in ("opt", "sched", "step") if k not in tk]
        if missing:
            raise SystemExit(
                f"--resume-temporal: {rp} predates optimiser-state saving "
                f"(missing {missing}). Loading the weights alone would reset "
                f"Adam's moments and the LR schedule, which is a warm restart "
                f"wearing a continuation's name. Refusing.\n\n"
                f"Every head published before 2026-08-10 is {{args, model}} "
                f"only — measured on f3_s2_60k, f3_s2_24k and the rescue "
                f"mirrors — so no existing checkpoint can be CONTINUED. If a "
                f"warm restart is what you want, ask for it by name: "
                f"--init-temporal {a.resume_temporal} --steps <EXTRA> --lr "
                f"<peak>, which trains a fresh cosine from these weights and "
                f"records that the moments were reset. Heads written from now "
                f"on carry opt/sched/step and are continuable.")
        opt.load_state_dict(tk["opt"])
        start_step = int(tk["step"])
        # THE SCHEDULE NEEDS A DECISION, and getting it wrong is silent.
        # CosineAnnealingLR.load_state_dict restores T_max and base_lrs from
        # the OLD run, so loading it while asking for a LARGER --steps leaves
        # T_max at the old total with last_epoch already there: the learning
        # rate is exactly 0.0 and the continuation trains 140,000 steps at
        # nothing. Measured, not feared — and the toy end-to-end run printed
        # "lr now 0.000e+00" while I read past it.
        _parent = dict(tk.get("args", {}))
        _parent["run_number"] = tk.get("run_number")
        prev_total = int(tk.get("args", {}).get("steps", start_step))
        extending = (a.steps != prev_total) or (abs(a.lr - float(
            tk.get("args", {}).get("lr", a.lr))) > 1e-12)
        if a.lr_schedule in ("invsqrt", "wsd", "expdecay"):
            # NOTHING TO DECIDE. A horizon-free schedule is a pure function of
            # the step, so extending is not a case: rebuild it at the same
            # position and it produces exactly what an uninterrupted run of
            # any length would produce there. This branch existing at all is
            # the cost of baking the total into the rate.
            #
            # expdecay was MISSING from this tuple until 2026-08-15 — it is
            # the most horizon-free of the three (lr = peak * 2^(-s/H), H
            # absolute), yet an extension request fell through to the branch
            # below and silently replaced it with a fresh COSINE over the new
            # total: different family, different rate at every remaining
            # step, and nothing in the output but a line saying "EXTENDING"
            # while the run trained under a schedule nobody asked for. Found
            # by reading, not by a burned run — the E-028 xl heads (expdecay,
            # taper off, lr still 3.7e-4 at 60k) are the first anyone wanted
            # to continue. (With --lr-cooldown-frac > 0 the taper end is
            # horizon-coupled and rebuilding moves it — the same accepted
            # coupling wsd's cooldown already has in this branch.)
            for g in opt.param_groups:
                g["lr"] = a.lr
                g["initial_lr"] = a.lr
            sched = make_sched(opt, a, last_epoch=start_step - 1)
            print(f"  {a.lr_schedule}: horizon-free, so the continuation "
                  f"simply resumes at step {start_step:,} — no extension case",
                  flush=True)
        elif extending:
            for g in opt.param_groups:
                g["lr"] = a.lr
                g["initial_lr"] = a.lr
            sched = torch.optim.lr_scheduler.CosineAnnealingLR(
                opt, a.steps, last_epoch=start_step - 1)
            print(f"  EXTENDING: new cosine over {a.steps:,} steps at peak lr "
                  f"{a.lr:.2e} (was {prev_total:,} steps at "
                  f"{tk.get('args', {}).get('lr')}), positioned at step "
                  f"{start_step:,}", flush=True)
        else:
            sched.load_state_dict(tk["sched"])
            print("  exact continuation: same total and same lr, schedule "
                  "state restored verbatim", flush=True)
        if tk.get("torch_rng") is not None:
            torch.set_rng_state(torch.as_tensor(tk["torch_rng"], dtype=torch.uint8))
        # E-057: the eps stream is part of the trajectory, exactly as
        # `torch_rng` is. Restoring the weights and the optimiser while the
        # noise restarts from draw 0 is a continuation in name only.
        if eps_gen is not None:
            if tk.get("eps_gen") is not None:
                eps_gen.set_state(torch.as_tensor(tk["eps_gen"],
                                                  dtype=torch.uint8))
                print("  eps stream restored from the checkpoint", flush=True)
            else:
                print("::warning::--resume-temporal: this checkpoint carries "
                      "no eps_gen state, so the eps stream restarts at draw 0 "
                      "— the weights continue but the noise does not. Report "
                      "this run as a warm restart of the noise stream.",
                      flush=True)
        if start_step >= a.steps:
            raise SystemExit(
                f"--resume-temporal: checkpoint is at step {start_step:,} and "
                f"--steps is {a.steps:,}. --steps is the TOTAL, not the extra.")
        lr_now = sched.get_last_lr()[0]
        print(f"resumed stage-2 head from {rp} at step {start_step:,} "
              f"-> training to {a.steps:,} (lr now {lr_now:.3e})", flush=True)
        # An invariant with an exact expectation, which is worth more than any
        # amount of careful reading: you cannot train at zero. Refuse rather
        # than spend sixteen hours updating nothing.
        if not (lr_now > 1e-12):
            raise SystemExit(
                f"--resume-temporal: the resumed learning rate is {lr_now:.3e}. "
                f"Training {a.steps - start_step:,} steps at that rate would "
                f"change nothing and report success. Check --steps (total, not "
                f"extra) and --lr.")

    def batch_windows(idx_t, idx_p, n):
        k = torch.randint(0, len(idx_t), (n,))
        t, p = idx_t[k], idx_p[k]
        base = win_ref(t)          # window start, or the anchor under offsets
        zseq = gather_stencil(Zt, base, p, NBR_t, K, FOFF)
        mseq = win_mseq(base)
        ztgt = win_ztgt(base, p)
        # True embeddings BEYOND the window, for the autoregressive unroll:
        # zfut[:, u] = Z[t+1+u] is the truth the model must hit after u
        # SELF-FED steps — u, not u+1. Column 0 is therefore the ordinary
        # teacher-forced target and is deliberately never read: the base term
        # takes it from ztgt, which scores the whole window rather than only
        # its last step. The loop below starts at u=1 for that reason, which
        # is also why U=1 leaves the objective bit-identical to the
        # pre-unroll one. (The comment here previously said "u+1 self-fed
        # steps", which contradicted both the code and ml/EXPERIMENTS.md;
        # the code was right.)
        zfut = torch.stack([Zt[t + 1 + u, p] for u in range(UF)], 1).float()
        mfut = torch.stack([Mt[t + 1 + u] for u in range(UF)], 1)
        # direct-horizon targets: zdir[:, i] = Z[t + D[i]] — the truth each
        # direct head must hit from the hidden state at t. Pool-guarded above.
        zdir = (torch.stack([Zt[t + h_, p] for h_ in D], 1).float().to(TDEV)
                if D else None)
        # --season-dropout: TRAINING FORWARDS ONLY, and this is the only
        # place a training forward's season features are built. Zeros are the
        # neutral value for a sin/cos pair — not a point on the unit circle at
        # all, so the month projection contributes nothing and the window has
        # to be dated from its own z. Per WINDOW, not per step: dropping some
        # steps of a window and not others would teach the head that the
        # calendar is intermittently observable, which is not the ablation.
        # The monitor batch (`mon_mseq`), both z-space evals and every roll
        # build their own mseq and are untouched by construction.
        if a.season_dropout > 0:
            keep_m = (torch.rand(len(t), 1, 1) >= a.season_dropout).float()
            mseq = mseq * keep_m
            mfut = mfut * keep_m
        # --holdout-scope target: [n, K, 1] float, 1 where this window's
        # frame target is a TRAIN bin. Gathered by the same `t` the window
        # was drawn at, from the table `frame_target_keep` filled — the mask
        # is never restated from a hand-written layout. None at every other
        # scope, which is what keeps the legacy loss statement reachable.
        wkeep = (KEEP_T[t].to(TDEV).to(ztgt.dtype).unsqueeze(-1)
                 if HOLD_MASK else None)
        # base and p stay on the CPU: --unroll-wide re-gathers the slot
        # pixels' own windows from the memmap, which is CPU work by design
        # (the 5.2 GiB of Z lives where the pages are).
        return (zseq.to(TDEV), mseq.to(TDEV), static_ctx[p].to(TDEV),
                ztgt.to(TDEV), zfut.to(TDEV), mfut.to(TDEV), zdir, base, p,
                wkeep)

    # Stage 2 goes into the RUN'S OWN metrics.jsonl, not just temporal.json.
    # temporal.json is uploaded as a build artifact, and artifacts need an
    # authenticated API call — the status page is deliberately credential-free
    # and reads only public raw branch content, so until now stage 2 was
    # invisible there: the page charted the codec's loss and its little
    # in-training probe, and said nothing about the model the whole second
    # stage exists to train. metrics.jsonl is already published to the live
    # branch and archived to ml-metrics AFTER this step runs, so writing here
    # needs no new transport.
    m2_path = os.path.join(run_dir, "metrics.jsonl")
    n_par2 = sum(p_.numel() for p_ in model.parameters())

    def m2(rec):
        try:
            with open(m2_path, "a") as f:
                f.write(json.dumps(rec) + "\n")
        except OSError:
            pass                      # instrumentation never breaks the run

    if start_step:
        m2({"stage2_resumed": {
            "from": os.path.basename(a.resume_temporal),
            "at_step": start_step, "to_step": a.steps,
            # Enough to redraw the parent's cosine EXACTLY without fetching
            # anything: annealing is analytic given peak and total.
            "parent_run": _parent.get("run_number"),
            "parent_steps": _parent.get("steps"),
            "parent_lr": _parent.get("lr"),
            "lr": a.lr}})
    if init_from:
        # A DIFFERENT record name from stage2_resumed, deliberately. The
        # status page and every later reader must be able to tell a warm
        # restart from a continuation without parsing prose, because the two
        # answer different questions and only one of them belongs on E-007's
        # curve.
        m2({"stage2_warm_restart": init_from})
    m2({"stage2_config": {"d_model": a.d_model, "layers": a.layers, "K": K,
                          "steps": a.steps, "params_M": round(n_par2 / 1e6, 3),
                          "batch": a.batch,
                          "train_windows": int(len(pool_t)),
                          "d_z": int(ck["d_z"]), "seed": a.seed,
                          "unroll": a.unroll,
                          # E-022: named here so the arm is readable from the
                          # live branch — on #221 it had to be inferred from
                          # params_M (32.338 = 32.038 + the 300,096 extra
                          # input columns), which is a check, not a record
                          "stencil": a.stencil,
                          "ring_km": a.ring_km,
                          "unroll_probs": a.unroll_probs,
                          "direct": a.direct,
                          # E-044: the two knobs that decide whether a stage-2
                          # run is the one it says it is. #423's verification
                          # item 10 could not settle `--input-znoise 0.7` from
                          # anywhere but the (expired) job log, because no
                          # record printed it; `grad_clip` would have had the
                          # same gap the day after it was added. Both are
                          # dispatch inputs, so both belong on the live branch.
                          "input_znoise": a.input_znoise,
                          "grad_clip": a.grad_clip,
                          # E-053.1: the frame TIMES, for the same reason —
                          # an arm whose only difference is where in the past
                          # its context sits must be readable from the live
                          # branch, not inferred from K and params_M.
                          "frame_offsets": a.frame_offsets,
                          "frame_span": int(CTX_BACK),
                          # WHICH LONGITUDES TRAINED. Named here for the same
                          # reason `stencil` is: an arm whose only difference
                          # is its training pool must be readable from the
                          # live branch, not inferred from train_windows.
                          "train_lon_hold": a.train_lon_hold,
                          # WHAT THE YEAR HOLDOUT EXCLUDED. Recorded
                          # UNCONDITIONALLY, unlike the fgn keys below: every
                          # run before 2026-08-28 was `endpoint`, and the
                          # difference between the two is whether held-out
                          # years were teacher-forced into the weights. A
                          # reader comparing two curves must not have to infer
                          # that from `train_windows`.
                          "holdout_scope": a.holdout_scope,
                          # ...AND HOW MUCH OF THE OBJECTIVE IT REMOVED. The
                          # scope name says which rule; this says what the
                          # rule cost, so an artefact self-describes the
                          # objective it trained under instead of leaving a
                          # reader to rebuild the axis and recount. 0.0 at
                          # `endpoint_contaminated` and at `window`, where
                          # every pooled window's every frame is scored.
                          "holdout_masked_frac": round(HOLD_MASKED_FRAC, 6),
                          # E-067 · WHICH YEARS. The scope says which RULE
                          # excluded them; this says WHICH ONES, so a live
                          # record self-describes the pool the same way it
                          # already self-describes the longitude half of it.
                          # The EFFECTIVE list — --holdout-years when given,
                          # the codec's otherwise — which is exactly what the
                          # saved head's own `args["holdout_years"]` carries.
                          "holdout_years": a.holdout_years,
                          "codec_holdout_years":
                              ck["args"].get("holdout_years", ""),
                          "codec_holdout_lon": ck["args"].get("holdout_lon", ""),
                          "tag": a.tag or "",
                          # E-057: NEW KEYS, AND ONLY WHEN THE ARM IS ONE.
                          # Adding `fgn_eps: 0` to every legacy record would
                          # be a changed record with the flag off, which §1 of
                          # the spec forbids; a reader that predates E-057
                          # ignores an extra key, and a reader that postdates
                          # it reads absence as "not an fgn run".
                          **({"stage2_loss_kind": "crps2",
                              "fgn_eps": a.fgn_eps,
                              "fgn_val_members": a.fgn_val_members,
                              # WHICH MEMBER the legacy point read-outs saw —
                              # recorded, never inferred (fgn_eval_eps()).
                              "fgn_eval_eps": "zeros"} if FGN else {})}})

    # ---- in-training monitoring (Chris, 2026-08-11: "it would be nice to
    # track more metrics during training") -----------------------------------
    # A FIXED held-out batch, built once: windows whose t+1 target is a
    # holdout month — the same population eval 1 scores after training,
    # sampled ~100 times during it instead. Monitoring only; nothing is
    # selected on it. And the light transport probe every 10% of the run:
    # E-008's question ("z improves — does transport?") deserved a curve,
    # not two endpoints.
    # E-053.1: the monitor's POPULATION is unchanged in kind — windows whose
    # t+1 is a holdout bin — and `mon_ztrue`/`mon_pers` below still key on
    # t+1 and t alone, so `val_zmse`, `val_persistence` and every ratio read
    # off them keep their exact meaning. Only how far back a window must
    # reach moves, and that is CTX_BACK.
    # --holdout-scope DELIBERATELY DOES NOT REACH HERE. This batch is the
    # MEASURING INSTRUMENT, not the training pool: it is the fixed-seed 4,096
    # windows whose t+1 is a HELD-OUT bin, and `val_persistence` /
    # `val_zmse` and every ratio quoted off them are only comparable across
    # arms while the population they are read from is identical. Narrowing it
    # under `window` would change the denominator at the same time as the
    # training set and make the two arms' numbers incomparable — which is the
    # one thing this change must not do. (A monitor window CAN read held-out
    # bins as context; so can persistence, which is exactly what it is scored
    # against. That is evaluation, not learning: no gradient flows here.)
    ev_m = np.array([t + 1 < T and t_hold[t + 1] and t >= CTX_BACK
                     for t in range(T)])
    emt, emp = np.where(ev_m[:, None] & np.ones(P, bool)[None, :])
    _mr = np.random.default_rng(12345)
    msel = _mr.choice(len(emt), min(4096, len(emt)), replace=False)
    emt = torch.as_tensor(emt[msel], dtype=torch.long)
    emp = torch.as_tensor(emp[msel], dtype=torch.long)
    _mb = win_ref(emt)
    mon_zseq = qz(gather_stencil(Zt, _mb, emp, NBR_t, K, FOFF).to(TDEV))
    mon_mseq = win_mseq(_mb).to(TDEV)
    mon_sctx = static_ctx[emp].to(TDEV)
    mon_ztrue = Zt[emt + 1, emp].float().to(TDEV)
    mon_pers = float((Zt[emt, emp].float().to(TDEV) - mon_ztrue).pow(2).mean())
    # ---- E-057 · the FIXED eval eps bank ----------------------------------
    # Drawn ONCE, from its OWN generator (seed*1000003 + 58, not eps_gen), so
    # the monitoring ensemble is the same M members at every log point — a
    # member-variance curve is only readable if the members do not change
    # underneath it — and so the number of monitor calls cannot perturb the
    # training noise stream. Moved to TDEV once; each forward broadcasts one
    # member across the whole monitoring batch, which is FGN's convention: eps
    # is GLOBAL, one draw for the whole field, not one per pixel.
    eps_val = None
    if FGN:
        _eg = torch.Generator()
        _eg.manual_seed(a.seed * 1000003 + 58)
        eps_val = torch.randn(a.fgn_val_members, a.fgn_eps,
                              generator=_eg).to(TDEV)
        print(f"fgn monitor: {a.fgn_val_members} fixed eval members from seed "
              f"{a.seed * 1000003 + 58}", flush=True)

    # E-044: REPORT THE SCALE THE INPUT NOISE IS ACTUALLY BEING APPLIED AT.
    # Nothing here changes behaviour — `--input-znoise` is used verbatim, as
    # it always was. This exists because the flag is an ABSOLUTE sigma in
    # whatever units the frozen codec's encoder happens to emit, and #423
    # carried the monthly anchor's 0.7 to a codec whose z-space is on a
    # different scale without one line of any record saying so. The flag's own
    # help asks for sigma = sqrt(val_zmse) of the MATCHING CLEAN RUN; at a new
    # cadence that reference run does not exist yet, so the honest thing to
    # print is the scale we DO have — sqrt(val_persistence), the RMS one-step
    # change of this z-space — and let the reader see how far the copied
    # constant sits from the arm it was copied from. Monthly anchor, measured:
    # val_persistence 3.09512, so 0.7 is 0.39788 x sqrt(persistence).
    # AND THE ABSOLUTE SCALE OF THE Z-SPACE ITSELF, which no record has ever
    # carried. `val_persistence` is the one-step CHANGE; `z_rms` is the size of
    # the thing that changes. The two are different questions and #423 could
    # answer neither from any artefact — its Z existed only on one rented disk,
    # so the session diagnosing it had to derive the scale indirectly from
    # `z_mse_persistence` in the CODEC's own stage-1 probe records. One
    # reduction over a batch that is already resident makes the next session's
    # answer a measurement instead (§4.10: instrument the quantity that
    # distinguishes the stories). Both are per-component means, so they are
    # comparable across codecs with different d_z — which is exactly the
    # comparison that was in doubt.
    _zrms = float(mon_ztrue.pow(2).mean().sqrt())
    _zref = float(np.sqrt(max(mon_pers, 1e-12)))
    _zrel = float(a.input_znoise) / _zref
    _zrelz = float(a.input_znoise) / max(_zrms, 1e-12)
    print(f"z-space scale (per component, over {len(msel)} held-out windows): "
          f"RMS |z| {_zrms:.5f} · RMS one-step change sqrt(val_persistence) "
          f"{_zref:.5f}", flush=True)
    if a.input_znoise > 0:
        print(f"input noise: --input-znoise {a.input_znoise:g} is an ABSOLUTE "
              f"sigma, = {_zrel:.5f} x sqrt(val_persistence) = {_zrelz:.5f} x "
              f"RMS |z|. The monthly anchor this constant was tuned on reads "
              f"0.39788 x sqrt(val_persistence) (0.7 at val_persistence "
              f"3.09512). A very different figure means the perturbation is "
              f"NOT the one that was measured, only the same number.",
              flush=True)
    m2({"stage2_monitor": {"n_windows": int(len(msel)),
                           "val_persistence": round(mon_pers, 5),
                           # the size of the z-space, beside the size of its
                           # one-step change. Never one without the other.
                           "z_rms": round(_zrms, 5),
                           # the sigma actually used, beside the scale it is
                           # judged against — never one without the other.
                           # #423's verification could not settle its own
                           # znoise from any record, only from a job log that
                           # then expired.
                           "input_znoise_sigma": round(float(a.input_znoise), 5),
                           "input_znoise_rel_pers": round(_zrel, 5),
                           "input_znoise_rel_zrms": round(_zrelz, 5)}})
    try:
        _pr = rapid_arr
        _pridx = _pr[:, 0].astype(int)
        _prv = _pr[:, 1].copy()
        _prmoy = moy[_pridx]
        _ptr = ~t_hold[_pridx]
        _pclim = np.array([_prv[_ptr & (_prmoy == m)].mean() for m in range(12)])
        _prv_des = _prv - _pclim[_prmoy]
        _, _psec = rapid_section(lats, lons, ys, xs)
        _psec_t = torch.as_tensor(np.asarray(_psec), dtype=torch.long)
        _pok = _pridx >= CTX_BACK
        _psec_ctx = static_ctx[_psec_t].to(TDEV)
    except Exception as _e:                     # monitoring never breaks a run
        print(f"  (in-training probe disabled: {_e})")
        _psec = None

    # E-044: GRADIENT CLIPPING. Until 2026-08-21 this trainer had none at
    # all — `grep -n "clip_grad\|max_norm\|grad_clip" ml/temporal.py`
    # returned NOTHING — and the monthly regime never asked for one. Measured,
    # not assumed: the `ml-metrics` branch holds 8,080 logged stage-2 grad
    # norms over 83 monthly runs (all at val_persistence 3.09512), median
    # 0.566, p99 4.30, p99.9 14.47, MAX 39.6165 (#308). #423, the first
    # stage-2 at PENTAD cadence, left that band at step 6,000 and never came
    # back: 8.24 -> 787 -> 3,891 -> 13,052, with two full blow-ups to
    # zmse 121 / val 227, from a step-2,000 best of val/persistence 0.540.
    #
    # OFF IS THE DEFAULT AND OFF MAKES NO CALL. Not `clip_grad_norm_(…, inf)`,
    # which is a no-op only for finite norms and multiplies every gradient by
    # a tensor otherwise; not a "very large" threshold, which is a promise
    # about a distribution nobody has measured on the next tensor. A default
    # that is never correct is not a default (§1), and the default that is
    # always correct here is "the code path every archived head was trained
    # on, unchanged". That is what makes the monthly archive comparable: a
    # monthly dispatch does not opt out, it simply never opts in.
    #
    # WHY A CLIP AT ALL WHEN THE OPTIMISER IS ADAM. Adam bounds the update per
    # COORDINATE, not the damage per STEP: one outlier batch spikes m and v
    # together, |m / (sqrt(v) + eps)| goes to ~1 in every coordinate at once,
    # and the parameter vector moves by ~lr * sqrt(N) = 1e-3 * sqrt(206.5M) =
    # 14.4 in one step. The second moment then stays inflated for
    # ~1/(1-beta2) = 1,000 steps, during which honest gradients are scaled to
    # nothing. That is the shape #423 actually shows: a spike, then thousands
    # of steps of slow, incomplete recovery, then another spike — and after
    # step 6,000 its grad norm NEVER returns below 1,000, so it is a sustained
    # regime change and not a sequence of isolated events.
    #
    # HOW TO SIZE THE THRESHOLD, in the units that decide the damage. One step
    # at norm g moves the second moment by v <- v + (1-beta2)*(g/g_ok)^2 * v,
    # so what matters is the RATIO to the healthy norm, squared. #423's healthy
    # norm is 8.25 (steps 2,000 and 4,000, and they agree to two decimals):
    #
    #   unclipped, at its worst 13,051.8 = 1,582x healthy -> v grows 2,503x,
    #       sqrt(v) 50x, and every honest gradient for the next ~1,000 steps
    #       is divided by 50. That is the poisoning, quantified.
    #   clipped at 128.0 = 15.5x healthy -> v grows 1.24x, sqrt(v) 1.11x.
    #       An 11% perturbation that decays in a few hundred steps.
    #
    # So 128 does not have to be tight to work: it turns a 50x derangement
    # into an 11% one. It is also mild in DISTRIBUTION terms — 128/8.25 = 15.5
    # healthy norms is, transferred to the monthly median of 0.566, a clip at
    # 8.8, which would have bound on 20 of the archive's 8,080 logged steps
    # (0.25%). A clip that occasionally bites the tail is the normal regime for
    # a transformer, not a compromise.
    CLIP = float(a.grad_clip)
    if CLIP > 0:
        # §4.10: INSTRUMENT THE QUANTITY THAT DISTINGUISHES THE STORIES. The
        # existing grad-norm log samples ONE step in `log_every` — one in
        # 2,000 on a 200,000-step run — so #423's excursion could have begun
        # anywhere inside a 2,000-step window and no record can say where, nor
        # whether the sampled step was typical or the single worst. Clipping
        # computes the norm on EVERY step anyway, so the window max and the
        # hit rate cost nothing extra. They are accumulated on-device and
        # synced only at a log point, so no GPU sync is added per step.
        #
        # The pair is what distinguishes the two stories a single sampled
        # norm cannot separate: "healthy, clip never binds" reads frac 0.0
        # with norm_max well under CLIP, and "being clipped constantly" — a
        # run whose effective learning rate is now set by the clip and not by
        # the schedule — reads frac climbing off 0. A run that silently moved
        # from the first to the second is the failure this logging exists to
        # make visible, and it says so a full log window before the sampled
        # norm would.
        _cl_max = torch.zeros((), device=TDEV)
        _cl_hit = torch.zeros((), device=TDEV)
        _cl_bad = torch.zeros((), device=TDEV)
        _cl_n = 0
        print(f"gradient clipping ON: max_norm {CLIP:g}, applied every step "
              f"between backward() and opt.step(). stage2_grad_norm keeps "
              f"reporting the PRE-clip norm; stage2_grad_norm_max, "
              f"stage2_grad_clip_frac and stage2_grad_nonfinite report over "
              f"each {max(1, a.steps // 100)}-step window, not just the "
              f"logged step.", flush=True)
    else:
        print("gradient clipping OFF (--grad-clip 0): no clip_grad_norm_ call "
              "is made, which is the pre-2026-08-21 code path exactly.",
              flush=True)

    print(f"training the temporal stage … ({n_par2:,} parameters)")
    t0 = time.time()
    log_every = max(1, a.steps // 100)     # ~100 curve points, as stage 1 does
    probe_every = max(1, a.steps // 10)    # the transport curve, 10 points
    for s in range(start_step + 1, a.steps + 1):
        (zseq, mseq, sctx, ztgt, zfut, mfut, zdir,
         wbase, wp, wkeep) = batch_windows(pool_t, pool_p, a.batch)
        if HOLD_MASK and s == start_step + 1:
            # FIRST BATCH ONLY (§4.7: assert the EFFECT). Recompute, from
            # this batch's OWN window refs and `t_hold`, which frame targets
            # are held out, and require the mask the loss is about to use to
            # be False in exactly those places. One batch, so it costs a
            # millisecond; a table filled at the wrong rows or an off-by-one
            # in the target bin dies here rather than in a head that quietly
            # learned the holdout.
            _w = np.zeros_like(wkeep.detach().cpu().numpy()[..., 0], bool)
            for _c, _j in enumerate(frame_steps(K, FOFF)):
                _w[:, _c] = t_hold[wbase.numpy() + int(_j) + 1]
            _got = (wkeep.detach().cpu().numpy()[..., 0] == 0.0)
            if not np.array_equal(_got, _w):
                sys.exit(
                    f"--holdout-scope target: THE LOSS MASK DOES NOT MATCH "
                    f"t_hold on the first batch — {int((_got != _w).sum())} "
                    f"of {_w.size} (window, frame) entries disagree "
                    f"({int(_w.sum())} targets are held out, the mask masks "
                    f"{int(_got.sum())}). Refusing to train on an objective "
                    f"nobody can describe.")
            print(f"--holdout-scope target: first-batch check — the loss "
                  f"mask is False on exactly the {int(_w.sum()):,} of "
                  f"{_w.size:,} (window, frame) targets t_hold marks held "
                  f"out, rebuilt from this batch's own window refs.",
                  flush=True)
        if a.input_znoise > 0:
            # E-029b: perturb only LIVE slots (see the flag's help). A slot
            # is live iff any of its d_z components is nonzero — zero is the
            # dead-slot encoding, so this recovers slot liveness without a
            # separate mask, at the cost of treating an exactly-all-zero
            # live embedding as dead (measure-zero in float32).
            z4 = zseq.view(*zseq.shape[:2], -1, ck["d_z"])    # [n,K,S,d_z]
            live = (z4 != 0).any(-1, keepdim=True)            # [n,K,S,1]
            zseq = (z4 + torch.randn_like(z4) * a.input_znoise
                    * live).view(zseq.shape)
        zseq = qz(zseq)
        if FGN:
            # E-057 · TWO FORWARDS ON THE IDENTICAL CONTEXT, TWO eps. The
            # context is built ONCE above — including the --input-znoise
            # corruption and the quantizer — so the ONLY thing that differs
            # between the two members is eps. If the corruption were drawn per
            # forward, the pair would differ by input noise as well and the
            # CRPS spread term would be measuring the wrong perturbation.
            # Both draws come from eps_gen on the CPU and are then moved: the
            # stream is device-independent by construction.
            _b = zseq.shape[0]
            eps1 = torch.randn(_b, a.fgn_eps, generator=eps_gen).to(TDEV)
            eps2 = torch.randn(_b, a.fgn_eps, generator=eps_gen).to(TDEV)
            p1, hid1 = model(zseq, mseq, sctx, eps=eps1)
            p2, _ = model(zseq, mseq, sctx, eps=eps2)
            # THE FAIR CRPS AT N=2 IS THE OBJECTIVE, not an extra term: under
            # MSE the conditional mean is optimal and the head would learn to
            # ignore eps. `stage2_loss_base` keeps its name and its place on
            # the curve; `stage2_loss_kind` in stage2_config says what it is.
            # WHY THE TWO BRANCHES ARE SEPARATE STATEMENTS AND NOT ONE
            # "unified" masked form. `(se * m).sum() / (m.sum() * d_z)` is
            # NOT bitwise equal to `se.mean()` even when m is all ones — a
            # different reduction order over a different summand — and
            # `endpoint_contaminated` (98 archived runs) and `window` (E-059,
            # in flight while this is written) must keep executing the exact
            # legacy statement, byte for byte. Only `target` takes the masked
            # form, so the other two scopes' objective cannot move by 1e-7.
            if HOLD_MASK:
                # EXACT, not approximate. `fair_crps2` is three elementwise
                # terms combined ONCE and then reduced, so masking before the
                # single reduction scores exactly the kept (window, frame)
                # entries — the same arithmetic the unmasked call makes over
                # a subset, never a mask applied to an already-reduced term
                # (`fair_crps2_elem` exists for precisely this).
                l_base = ((fair_crps2_elem(p1, p2, ztgt) * wkeep).sum()
                          / (wkeep.sum() * ztgt.shape[-1]))
            else:
                l_base = fair_crps2(p1, p2, ztgt)
            # member 1 stands where the deterministic forward stood, for the
            # unroll/direct paths' shapes only — both are refused in fgn mode.
            pred = p1
        else:
            pred, hid1 = model(zseq, mseq, sctx)
            if HOLD_MASK:
                # --holdout-scope target: the dense per-frame MSE over the
                # KEPT targets only. `wkeep` is [B, K, 1] and broadcasts over
                # d_z; the denominator counts the elements actually summed,
                # so this is the mean of the same per-element squared errors
                # the legacy statement means — taken over the subset whose
                # target bin is a train bin.
                l_base = (((pred - ztgt).pow(2) * wkeep).sum()
                          / (wkeep.sum() * ztgt.shape[-1]))
            else:
                l_base = (pred - ztgt).pow(2).mean()
        loss = l_base
        l_dir = None
        if D:
            # DIRECT horizons: each head reads the hidden state at the LAST
            # window position and predicts z at t+h in one shot. Scored at
            # the last position only — that is exactly how the head will be
            # used (predict from the newest true context), and the trunk is
            # already trained at every position by the base term. Weighted
            # mean over horizons, so adding horizons never outvotes t+1.
            ld = sum((model.heads_direct[str(h_)](hid1[:, -1])
                      - zdir[:, i_]).pow(2).mean()
                     for i_, h_ in enumerate(D))
            l_dir = ld / len(D)
            loss = loss + l_dir
        # AUTOREGRESSIVE UNROLL — the fix for EXPOSURE BIAS. rollout.py scores
        # this model by feeding its own predictions back in, but the objective
        # above only ever shows it TRUE context, so it is never trained on the
        # error distribution it will actually face and errors compound at
        # rollout. Here the context slides forward on the model's own last
        # prediction and the next true month is the target. Each extra step is
        # down-weighted 1/(u+1) so a deep unroll cannot outvote the t+1 term
        # that anchors the whole objective.
        # Sampled unroll depth: one draw per STEP (whole batch shares it, so
        # tensor shapes never vary within a batch). With UP=None this is the
        # fixed depth U every step — bit-identical to the old objective.
        U_t = (1 + int(np.random.choice(U, p=UP))) if UP is not None else U
        zin, min_ = zseq, mseq
        l_unr = None
        for u in range(1, U_t):
            zin = qz(torch.cat([zin[:, 1:], pred[:, -1:]], 1))  # graph intact
            min_ = torch.cat([min_[:, 1:], mfut[:, u - 1:u]], 1)
            pred, _ = model(zin, min_, sctx)
            term = (pred[:, -1] - zfut[:, u]).pow(2).mean() / (u + 1)
            l_unr = term if l_unr is None else l_unr + term
            loss = loss + term
        l_uw = None
        if UW:
            # E-030: ONE-HOP UNROLL FOR WIDE STENCILS (Chris's dependency-cone
            # observation, 2026-08-15). The plain unroll above cannot run at
            # stencil>1 because the model predicts only its centre pixel while
            # its t+1 input window needs the NEIGHBOURS' t+1 embeddings. But
            # each neighbour's t+1 embedding is a DEPTH-1 prediction from that
            # neighbour's own fully-observed window — no feedback, no reach
            # beyond one hop. So: for a sub-batch of bu windows, forward all
            # S slot pixels' own windows once (detached — the gradient path
            # to the second step is through the assembled input's USE, not
            # its construction; letting gradients flow through S auxiliary
            # forwards would triple memory for a signal the plain unroll also
            # discards via its own detach-free but centre-only path), zero
            # the dead slots (zero IS the dead-slot encoding, and the roll
            # feeds zeros there too), assemble the centre pixel's t+1 window,
            # and score a differentiable second step against Z[t+2] at
            # weight 1/2 — exactly the u=2 term of --unroll.
            bu = min(a.uw_batch, zseq.shape[0])
            q = NBR_t[wp[:bu]]                     # [bu,S] pixel ids, -1=dead
            valid = (q >= 0)
            safe = q.clamp(min=0)                  # -1 -> 0: a real pixel id,
            Sn = q.shape[1]                        # its prediction zeroed below
            b_rep = wbase[:bu].repeat_interleave(Sn)
            p_rep = safe.reshape(-1)               # row-major: window-major,
            z1 = gather_stencil(Zt, b_rep, p_rep, NBR_t, K)   # matches b_rep
            m1 = mseq[:bu].repeat_interleave(Sn, 0)        # months are
            s1 = static_ctx[p_rep]                         # pixel-independent
            with torch.no_grad():
                # not _chunked_forward: that would round-trip the hidden
                # states through the CPU (~350 MB/step at big), and only
                # pred[:, -1] is needed — chunk inline, keep just that.
                _sp = []
                for i0 in range(0, len(z1), 4096):
                    _sl = slice(i0, i0 + 4096)
                    p1_, _ = model(qz(z1[_sl].to(TDEV)), m1[_sl],
                                   s1[_sl].to(TDEV))
                    _sp.append(p1_[:, -1])
                step1 = torch.cat(_sp)                     # [bu*S, d_z]
            step1 = step1 * valid.reshape(-1, 1).to(TDEV)  # dead slots: zeros
            newstep = step1.reshape(bu, -1)                # [bu, S*d_z]
            zseq2 = torch.cat([zseq[:bu, 1:], newstep[:, None]], 1)
            mseq2 = torch.cat([mseq[:bu, 1:], mfut[:bu, 0:1]], 1)
            pred2, _ = model(qz(zseq2), mseq2, sctx[:bu])
            l_uw = (pred2[:, -1] - zfut[:bu, 1]).pow(2).mean() / 2
            loss = loss + l_uw
        opt.zero_grad(); loss.backward()
        _logstep = (s % log_every == 0 or s == a.steps)
        if CLIP > 0:
            # clip_grad_norm_ RETURNS the PRE-clip total norm — the identical
            # quantity the else-branch computes by hand — so the clipped path
            # costs nothing extra to log and gets the norm on EVERY step
            # instead of one step in log_every.
            _gnt = torch.nn.utils.clip_grad_norm_(model.parameters(),
                                                  CLIP).detach()
            # §5.22, NEVER WRITE NaN INTO A RESULTS FILE. torch.maximum
            # PROPAGATES NaN (it is not fmax), so a single non-finite step
            # would pin the running max at NaN for the rest of the run and
            # every later stage2_grad_norm_max would be a NaN — loud enough to
            # notice and quiet enough to misattribute, which is the exact
            # failure §5.22 names. A non-finite norm is NOT a big number: it
            # is a different event, so it gets its own counter and is kept out
            # of the max and out of the rate.
            _fin = torch.isfinite(_gnt)
            _cl_bad += (~_fin)
            _g_ok = torch.where(_fin, _gnt, torch.zeros_like(_gnt))
            torch.maximum(_cl_max, _g_ok, out=_cl_max)
            _cl_hit += (_g_ok > CLIP)
            _cl_n += 1
            if _logstep:
                gn = float(_g_ok)
        elif _logstep:
            # gradient norm BEFORE the step, only on log steps (a .item()
            # per step would sync the GPU 60k times)
            gn = float(torch.sqrt(sum((p_.grad.detach() ** 2).sum()
                                      for p_ in model.parameters()
                                      if p_.grad is not None)))
        opt.step(); sched.step()
        if s % log_every == 0 or s == a.steps:
            # held-out z-MSE + amplitude on the fixed monitoring batch: the
            # val curve beside the train curve, and the SMOOTHING diagnostic
            # (std(pred)/std(true) — E-014's conditional-mean collapse was
            # invisible during training precisely because nothing logged it)
            _fgn_val = {}
            with torch.no_grad():
                if FGN:
                    # E-057 · THE ENSEMBLE READ. M forwards of the SAME
                    # monitoring batch, one fixed eps member each, broadcast
                    # across every window — eps is global, so all windows in a
                    # member share it exactly as all pixels of a field would.
                    # CHUNKED, since #496 (E-057.1a seed 0, 2026-08-27): one
                    # full-batch forward here is a ~3.65 GB input cat at
                    # stencil 145 x 4096 windows, and M of them per log step
                    # interleaved with the two-forward CRPS training steps
                    # fragmented a 24 GB card to death (OOM at the step-6000
                    # val; 13.4 GiB reserved-but-unallocated). Same class as
                    # E-027 #285/#286 — see _chunked_forward. Concatenation
                    # over disjoint slices is exact for a row-wise forward;
                    # eps is broadcast per member either way, so chunking
                    # cannot change which noise a window sees.
                    _ens = []
                    _CH = 512
                    for _mi in range(eps_val.shape[0]):
                        _outs = []
                        for _c0 in range(0, mon_zseq.shape[0], _CH):
                            _sl = slice(_c0, _c0 + _CH)
                            _e = eps_val[_mi].expand(
                                mon_zseq[_sl].shape[0], -1)
                            _pm, _ = model(mon_zseq[_sl], mon_mseq[_sl],
                                           mon_sctx[_sl], eps=_e)
                            _outs.append(_pm[:, -1])
                        _ens.append(torch.cat(_outs, 0))
                    ens = torch.stack(_ens)              # [M, n, d_z]
                    # THE ENSEMBLE MEAN is the best point estimate, so it is
                    # what goes into the two EXISTING keys: `stage2_val_zmse`
                    # and `stage2_amp` keep their names and their curves, and
                    # in fgn mode they mean "of the ensemble mean". The legacy
                    # meaning is untouched when fgn is off — this branch does
                    # not run at all there.
                    mlast = ens.mean(0)
                    val_mse = float((mlast - mon_ztrue).pow(2).mean())
                    amp = float(mlast.std() / (mon_ztrue.std() + 1e-9))
                    _M = float(ens.shape[0])
                    # spread/error with the (M+1)/M correction — the mirror of
                    # ml/probscore.spread_error: the ensemble MEAN carries its
                    # own sigma^2/M of sampling error on top of the truth's, so
                    # an uncorrected ratio reports under-dispersion at every
                    # finite M even for a perfect ensemble. 1.0 is calibration.
                    _msp = float(((_M + 1.0) / _M
                                  * ens.var(0, unbiased=True)).mean())
                    _fgn_val = {
                        "stage2_val_crps": float(fair_crps_ens(ens, mon_ztrue)),
                        # THE eps-COLLAPSE TELEMETRY (E-057 F2). A member
                        # variance sliding to 0 is the signature of a head that
                        # has learned to ignore its noise, and it must be
                        # visible on the live branch while the run is alive,
                        # not reconstructed afterwards.
                        "stage2_val_member_var":
                            float(ens.var(0, unbiased=False).mean()),
                        "stage2_val_spread_ratio":
                            (math.sqrt(_msp) / math.sqrt(val_mse)
                             if _msp >= 0.0 and val_mse > 0.0
                             else float("nan")),
                    }
                else:
                    # CHUNKED, like the fgn branch above, and for the same
                    # reason it is exact there: the forward is row-wise, so
                    # concatenating disjoint slices reproduces the one-shot
                    # result bit for bit (E-027 #285/#286, _chunked_forward).
                    # Measured 2026-09-02 on #529 (E-064b, d_z 32, K 144,
                    # stencil 145): the one-shot pass over 4,096 held-out
                    # windows asked the 24 GB card for a single 10.2 GiB
                    # allocation at the FIRST monitor and died — the run
                    # went green with no temporal.json (§7's signature). The
                    # d_z-6 token arm (#528) survived only because its input
                    # is 5x smaller.
                    _CH = 512
                    _outs = []
                    for _c0 in range(0, mon_zseq.shape[0], _CH):
                        _sl = slice(_c0, _c0 + _CH)
                        _pm, _ = model(mon_zseq[_sl], mon_mseq[_sl],
                                       mon_sctx[_sl])
                        _outs.append(_pm[:, -1])
                    mlast = torch.cat(_outs, 0)
                    val_mse = float((mlast - mon_ztrue).pow(2).mean())
                    amp = float(mlast.std() / (mon_ztrue.std() + 1e-9))
            rec = {"stage2_step": s, "stage2_zmse": round(float(loss.item()), 5),
                   "stage2_loss_base": round(float(l_base.item()), 5),
                   "stage2_val_zmse": round(val_mse, 5),
                   "stage2_amp": round(amp, 4),
                   "stage2_grad_norm": round(gn, 4),
                   # The RATE, logged rather than inferred. A resumed run's
                   # schedule is the thing most likely to be wrong (it was: a
                   # reloaded cosine gave lr 0.0), and a chart that shows the
                   # loss without the rate cannot distinguish "converged" from
                   # "not learning because the LR is zero".
                   "stage2_lr": float(sched.get_last_lr()[0]),
                   "stage2_wall_s": round(time.time() - t0, 1)}
            # E-057 · the ensemble keys, BESIDE the existing ones and never
            # instead of them. §5.22: NEVER WRITE NaN INTO A RESULTS RECORD —
            # a non-finite value omits its key and says so on stderr, because
            # an absent key cannot be mistaken for a measurement and a NaN can.
            for _k, _v in _fgn_val.items():
                if _v == _v and abs(_v) != float("inf"):
                    # SIGNIFICANT digits, not decimal places: the collapse
                    # telemetry is read near ZERO, and `round(v, 6)` prints a
                    # member variance of 1e-8 as exactly 0.0 — i.e. it destroys
                    # the resolution precisely where the failure mode lives.
                    rec[_k] = float(f"{float(_v):.6g}")
                else:
                    print(f"::warning::{_k} was non-finite at step {s} "
                          f"({_v}) — key omitted from this record rather than "
                          f"written as NaN", flush=True)
            if CLIP > 0:
                # WINDOW statistics, not point statistics: the max PRE-clip
                # norm seen since the last log point, and the fraction of
                # steps at which the clip actually bound. A run whose frac
                # climbs off 0 is leaving the regime the threshold was sized
                # for — its effective learning rate is being set by the clip
                # and not by the schedule — and it says so a whole log window
                # before a single sampled norm could.
                rec["stage2_grad_clip"] = CLIP
                rec["stage2_grad_norm_max"] = round(float(_cl_max), 4)
                rec["stage2_grad_clip_frac"] = round(
                    float(_cl_hit) / max(1, _cl_n), 4)
                # counted, never averaged into the two above
                rec["stage2_grad_nonfinite"] = int(_cl_bad)
                _cl_max.zero_(); _cl_hit.zero_(); _cl_bad.zero_(); _cl_n = 0
            if l_dir is not None:
                rec["stage2_loss_direct"] = round(float(l_dir.item()), 5)
            if l_unr is not None:
                rec["stage2_loss_unroll"] = round(float(l_unr.item()), 5)
            if l_uw is not None:
                rec["stage2_loss_unroll_wide"] = round(float(l_uw.item()), 5)
            m2(rec)
            # the light transport probe, ten times per run: hidden(-1)
            # pooled over the section, 36-month-split ridge — the NOISY
            # instrument, quoted for its TREND only
            #
            # E-055 DELIBERATELY DID NOT ADD AN UNPOOLED PARTNER HERE. The
            # k-fold site at the end of the run got one; this one did not, for
            # three reasons that are about this call site and not about the
            # read-out:
            #  · it runs INSIDE the training loop, and the attention pool is
            #    fitted by gradient descent — an optimiser and a ~25k-parameter
            #    graph on the same GPU as a live stage-2 training step, ten
            #    times a run. #392 and #388 were OOM-killed by allocations
            #    smaller than that gap; the failure mode here is losing the
            #    RUN, not losing a number.
            #  · the fit draws from the global RNG, and this loop's window
            #    draw is downstream of it. `_keep_rng` restores CPU/CUDA state,
            #    but restoring state the training loop is concurrently
            #    consuming is a much stronger claim than restoring it around a
            #    read-out that runs after training has stopped, and a wrong
            #    claim here changes the TRAINING RUN, not a probe.
            #  · the number would be the 36-month single split, which this
            #    programme already refuses to treat as a verdict — the
            #    unpooled ruling is about what is QUOTED, and nothing quotes
            #    this key except as a trend line.
            # `rapid_r_deseas_unpooled` in the final record is the unpooled
            # partner for the single-split protocol; it is fitted once, after
            # `model.eval()`, where none of the above applies.
            if _psec is not None and (s % probe_every == 0 or s == a.steps):
                try:
                    with torch.no_grad():
                        F_ = np.zeros((T, a.d_model), np.float32)
                        # E-053.1: the valid-t range is the pool bound again
                        # (CTX_BACK, = K-1 with contiguous frames), and the
                        # window is assembled through the same two helpers.
                        for t_ in range(CTX_BACK, T):
                            b_ = win_ref(t_)
                            zs_ = gather_stencil(
                                Zt, torch.full_like(_psec_t, b_), _psec_t,
                                NBR_t, K, FOFF)
                            ms_ = torch.stack(
                                [Mt[b_ + j].expand(len(_psec), -1)
                                 for j in frame_steps(K, FOFF)], 1)
                            _, hd_ = eval_forward(
                                model, qz(zs_.to(TDEV)), ms_.to(TDEV),
                                _psec_ctx, a.fgn_eps)
                            F_[t_] = hd_[:, -1].mean(0).cpu().numpy()
                    ri_ = _pridx[_pok]
                    r_, _ = ridge_r(F_[ri_], _prv_des[_pok],
                                    ~t_hold[ri_], t_hold[ri_])
                    m2({"stage2_probe": {"step": s,
                                         "rapid_r_deseas": round(float(r_), 4)}})
                except Exception as _e:
                    print(f"  (in-training probe failed at {s}: {_e})")
            # MIRROR THE HEAD AS IT TRAINS, exactly as train.py mirrors the
            # codec. Until now the head existed only in the run's workspace
            # and was uploaded by a step that runs AFTER the whole probe
            # ladder — so a job that hit its timeout lost every step of it.
            # A 60,000-step head is seven hours of GPU; losing it to a
            # bookkeeping deadline is not an acceptable failure mode.
            # Cheap: 7 MB, ~100 writes over a run.
            try:
                os.makedirs(CKPT_DIR, exist_ok=True)
                tag = os.environ.get("CKPT_TAG", "")
                tmp_path = os.path.join(
                    CKPT_DIR, (tag + "-" if tag else "") + "temporal.pt")
                torch.save({"model": model.state_dict(), "args": vars(a),
                            "step": s,
                            # OPTIMISER AND SCHEDULE TOO. Weights alone are not
                            # a resumable state: reloading them and building a
                            # fresh AdamW resets the moments and restarts the
                            # cosine, which is a warm restart wearing a
                            # continuation's name. RNG state as well, so the
                            # window draw continues rather than repeating.
                            "opt": opt.state_dict(),
                            "sched": sched.state_dict(),
                            "run_number": os.environ.get("GITHUB_RUN_NUMBER"),
                            "torch_rng": torch.get_rng_state().numpy().tolist(),
                            # E-057: the eps stream, on the same footing as
                            # torch_rng. {} when fgn is off, so a legacy
                            # mirror is byte-for-byte what it always was.
                            **_eps_state()},
                           tmp_path + ".part")
                os.replace(tmp_path + ".part", tmp_path)
            except Exception as e:                       # never fatal
                print(f"  (head mirror failed: {e})", flush=True)
        if s in MILESTONES:
            # Weights-only rung checkpoint (see --milestone-steps help).
            # Atomic and never fatal, like the mirror; named by step so a
            # 200k run's artifact carries temporal_ms60000.pt and
            # temporal_ms120000.pt beside the final temporal.pt.
            try:
                mp = os.path.join(run_dir, f"temporal_ms{s}.pt")
                torch.save({"model": model.state_dict(), "args": vars(a),
                            "step": s,
                            "run_number": os.environ.get("GITHUB_RUN_NUMBER"),
                            **_eps_state()},
                           mp + ".part")
                os.replace(mp + ".part", mp)
                print(f"  milestone checkpoint saved: temporal_ms{s}.pt",
                      flush=True)
            except Exception as e:                       # never fatal
                print(f"  (milestone save failed at {s}: {e})", flush=True)
        if s % max(1, a.steps // 10) == 0:
            print(f"  step {s:>6}/{a.steps}  z-mse {loss.item():.4f}"
                  f"  ({time.time() - t0:.0f}s)", flush=True)

    model.eval()
    results = {"run": a.run, "K": K, "d_model": a.d_model, "layers": a.layers,
               "steps": a.steps}
    # THE FOUR SCALE NUMBERS, structured, from the run itself (Chris,
    # 2026-08-11: "each entry … contains: the number of parameters, the
    # batch size, the number of steps, the number of data points").
    # Written by the trainer rather than the log entry, so they can be
    # quoted but never invented — the C=24-mislabelled-as-14 incident is
    # what happens when scale numbers travel by hand.
    results["scale"] = {
        "params": int(n_par2),
        "batch": int(a.batch),
        "steps": int(a.steps),
        "data_points": int(len(pool_t)),   # train windows in the pool
        "n_pixels": int(P),
        "n_train_months": int((~t_hold).sum()),
        "stencil": int(a.stencil),         # E-022: part of the arm's identity
        "ring_km": a.ring_km,              # E-023/E-026: radius, or radii
        "frame_offsets": a.frame_offsets,  # E-053.1: and so are the frame times
        "frame_span": int(CTX_BACK),
    }

    # ---- eval 1: z-space t+1 on held-out target months --------------------
    with torch.no_grad():
        ev_t = np.array([t + 1 < T and t_hold[t + 1] and t >= CTX_BACK
                         for t in range(T)])
        et, ep = np.where(ev_t[:, None] & np.ones(P, bool)[None, :])
        sel = np.random.default_rng(a.seed).choice(len(et), min(20000, len(et)), replace=False)
        et = torch.as_tensor(et[sel], dtype=torch.long)
        ep = torch.as_tensor(ep[sel], dtype=torch.long)
        base = win_ref(et)
        zseq = gather_stencil(Zt, base, ep, NBR_t, K, FOFF)
        mseq = win_mseq(base)
        pred, hid = _chunked_forward(model, qz(zseq), mseq,
                                     static_ctx[ep], TDEV, eps_dim=a.fgn_eps)
        # already on CPU: everything
        zhat = pred[:, -1]                       # below here is numpy-bound
        ztrue = Zt[et + 1, ep].float()
        zlast = Zt[et, ep].float()                        # persistence in z
        results["z_t+1"] = {
            "mse_model": float((zhat - ztrue).pow(2).mean()),
            "mse_persistence": float((zlast - ztrue).pow(2).mean()),
        }
        results["z_t+1"]["beats_persistence"] = (
            results["z_t+1"]["mse_model"] < results["z_t+1"]["mse_persistence"])

        # ---- eval 2: decode ẑ through the frozen codec → channel space ----
        # E-047: a BLOCK codec decodes a CELL, not a bin, and "the channel
        # value at t+1" has k_max answers under blocking — one per cell of the
        # predicted block — while the truth array here is still per-bin. Which
        # cell stands for the block is an evaluation SEMANTIC nobody has
        # chosen yet (ml/plans/E047_block_codec.md 7), so this eval records
        # that it does not apply rather than picking a cell and reporting the
        # number as if it were the old one. eval 1 above (z_t+1) is unchanged
        # and is the headline either way.
        if BLKA is not None:
            results["chan_t+1"] = {"skipped": (
                f"block codec (k_time {int(ck['args']['k_time'])}): decoding "
                f"z to channel space names a CELL, and which cell stands for "
                f"the block is not yet decided. z_t+1 above is unaffected.")}
            print(f"::warning::chan_t+1 skipped — "
                  f"{results['chan_t+1']['skipped']}", flush=True)
        qc = torch.arange(C)[None, :].expand(len(et), -1)
        off0 = torch.zeros(len(et), C, 3, dtype=torch.long)
        xhat = (None if BLKA is not None else codec.query(zhat, qc, off0))
        ys_t = torch.as_tensor(ys, dtype=torch.long)
        xs_t = torch.as_tensor(xs, dtype=torch.long)
        if xhat is not None:
            v1 = Xt[et + 1, ys_t[ep], xs_t[ep]]
            o1 = OBS[et + 1, ys_t[ep], xs_t[ep]]
            v0 = Xt[et, ys_t[ep], xs_t[ep]]
            o0 = OBS[et, ys_t[ep], xs_t[ep]]
            both = o0 & o1
            dyn = torch.zeros(C, dtype=torch.bool); dyn[dynamic] = True
            both = both & dyn[None, :]
            mse_m = float(((xhat - v1).pow(2) * both).sum() / both.sum())
            mse_p = float(((v0 - v1).pow(2) * both).sum() / both.sum())
            results["chan_t+1"] = {"mse_model": mse_m,
                                   "mse_persistence": mse_p,
                                   "beats_persistence": mse_m < mse_p,
                                   "channels": [chan[c] for c in dynamic]}

    # ---- eval 2b: DIRECT horizons, z-space, held-out targets --------------
    # One forward from true context, the horizon head reads hidden(-1) — the
    # anti-compounding number. Persistence at horizon h is z_t frozen, the
    # same baseline family the rollout uses; the rollout comparison (direct
    # vs iterated at the SAME (start, h) points) is rollout.py's job.
    if D:
        results["z_direct"] = {}
        with torch.no_grad():
            for h_ in D:
                evh = np.array([t + h_ < T and t_hold[t + h_]
                                and t >= CTX_BACK for t in range(T)])
                eth, eph = np.where(evh[:, None] & np.ones(P, bool)[None, :])
                if not len(eth):
                    continue
                sel = np.random.default_rng(a.seed + h_).choice(
                    len(eth), min(20000, len(eth)), replace=False)
                et_ = torch.as_tensor(eth[sel], dtype=torch.long)
                ep_ = torch.as_tensor(eph[sel], dtype=torch.long)
                base = win_ref(et_)
                zsq = gather_stencil(Zt, base, ep_, NBR_t, K, FOFF)
                msq = win_mseq(base)
                _, hd = _chunked_forward(model, qz(zsq), msq,
                                         static_ctx[ep_], TDEV,
                                         eps_dim=a.fgn_eps)
                zh = model.heads_direct[str(h_)](hd[:, -1].to(TDEV)).cpu()
                zt_ = Zt[et_ + h_, ep_].float()
                zp_ = Zt[et_, ep_].float()
                mm = float((zh - zt_).pow(2).mean())
                mp = float((zp_ - zt_).pow(2).mean())
                results["z_direct"][str(h_)] = {
                    "mse_model": mm, "mse_persistence": mp,
                    "beats_persistence": mm < mp}
                print(f"  direct h={h_}: z-mse {mm:.4f} vs persistence "
                      f"{mp:.4f} ({'beats' if mm < mp else 'LOSES TO'} it)")

    # ---- eval 3: RAPID probe from temporal hidden state -------------------
    # protocol v2: deseasonalised target (train-years clim), seasonal floor,
    # lambda on a train tail — identical scoring path to probe_sequence.py.
    rapid = rapid_arr
    _, sec_after = rapid_section(lats, lons, ys, xs)   # ys/xs possibly subsampled
    sec_pix = torch.as_tensor(sec_after, dtype=torch.long)
    # E-055: the UNPOOLED read-out needs hidden(-1) BEFORE the section mean.
    # The same forward that fills `F` keeps the per-pixel states for the rows
    # the RAPID probe actually scores (~240 of T, not all of T — at daily
    # cadence the full [T, P, d_model] block is gigabytes and is 92% rows no
    # probe ever reads). `F` itself is untouched: `hid[:, -1].mean(0)` below is
    # the byte-for-byte pooled path 98 archived runs read.
    _u_rows = sorted({int(t) for t in rapid[:, 0].astype(int)
                      if CTX_BACK <= int(t) < T})
    _u_pos = {t: i for i, t in enumerate(_u_rows)}
    try:
        FP = np.zeros((len(_u_rows), len(sec_after), a.d_model), np.float32)
    except (MemoryError, ValueError) as _e:                   # noqa: BLE001
        FP = None
        print(f"::warning::unpooled section states not collected "
              f"({type(_e).__name__}: {_e}) — the pooled probe below is "
              f"unaffected", flush=True)
    with torch.no_grad():
        F = np.zeros((T, a.d_model), dtype=np.float32)
        for t in range(CTX_BACK, T):
            base = win_ref(t)
            zseq = gather_stencil(Zt, torch.full_like(sec_pix, base),
                                  sec_pix, NBR_t, K, FOFF)
            mseq = torch.stack([Mt[base + j].expand(len(sec_pix), -1)
                                for j in frame_steps(K, FOFF)], 1)
            _, hid = eval_forward(model, qz(zseq.to(TDEV)), mseq.to(TDEV),
                                  static_ctx[sec_pix].to(TDEV), a.fgn_eps)
            F[t] = hid[:, -1].mean(0).cpu().numpy()   # pool along the section
            if FP is not None and t in _u_pos:        # E-055: the same states,
                FP[_u_pos[t]] = hid[:, -1].cpu().numpy()      # unpooled
    ridx = rapid[:, 0].astype(int)
    rv_raw = rapid[:, 1].copy()
    rmoy = moy[ridx]
    tr_all = ~t_hold[ridx]
    rclim = np.array([rv_raw[tr_all & (rmoy == m)].mean() for m in range(12)])
    rv_des = rv_raw - rclim[rmoy]
    ok = ridx >= CTX_BACK
    ri = ridx[ok]
    tr, te = ~t_hold[ri], t_hold[ri]
    r_raw, _ = ridge_r(F[ri], rv_raw[ok], tr, te)
    r_des, _ = ridge_r(F[ri], rv_des[ok], tr, te)
    results["rapid_probe"] = {"r_raw": r_raw, "r_deseasonalised": r_des,
                              "n_test": int(te.sum()), "features": "hidden(-1) mean over section"}

    # ---- eval 3b: THE SAME FEATURES, YEAR-BLOCKED K-FOLD ------------------
    # Added 2026-08-10, on discovering that E-009 as dispatched could not
    # answer its own question.
    #
    # probe_kfold.py — the one instrument this programme argues from — scores
    # the CODEC: it pools the frozen embeddings along the section and fits a
    # ridge. The temporal head is not in it anywhere. Proof rather than
    # reading: #116 (frozen codec, 60k head) and #125 (same codec, 200k head,
    # a completely different schedule) return RAPID 0.631 [0.513, 0.732] with
    # the same rmse 2.16 — bit-identical, because the only thing that differs
    # between them is invisible to that probe.
    #
    # So every stage-2 question — unroll, schedule, budget — had exactly one
    # instrument: `rapid_probe` above, a SINGLE split on 36 held-out months.
    # That is the noisy number #88 and #93 disagreed on by 0.28, and the
    # reason a four-arm sweep was about to be scored on a metric that cannot
    # move with what it varies.
    #
    # This scores the head's own features through probe_kfold's protocol:
    # fold by calendar year, lambda on an inner tail, one r over ~240
    # out-of-fold months, block bootstrap over whole years for the CI. Six
    # times the test months and a stated interval, on the same footing as the
    # codec number it sits beside in the results file.
    #
    # The import is deliberately LOCAL. probe_kfold imports this module at
    # load time (embed_everything, section_of), so a module-level import here
    # would be a cycle; done at the point of use, temporal.py is already
    # fully initialised when probe_kfold asks for it. The alternative — moving
    # kfold_r into a third module — would edit the instrument every published
    # number came from, and it is not worth that during a live queue.
    try:
        from probe_kfold import kfold_r
        yr_of = np.array([int(months[i][:4]) for i in ri])
        r_kf, lo_kf, hi_kf, n_kf, rmse_kf, sig_kf, _ = kfold_r(
            F[ri], rv_des[ok], yr_of, seed=a.seed)
        results["rapid_probe_kfold"] = {
            "r_kfold_deseas": round(r_kf, 3), "ci95": [round(lo_kf, 3), round(hi_kf, 3)],
            "n": n_kf, "rmse_sv": round(rmse_kf, 2), "sigma_sv": round(sig_kf, 2),
            "features": "hidden(-1) mean over section",
            "note": ("year-blocked k-fold over the TEMPORAL HEAD's features. "
                     "probe_kfold.json in this same run scores the CODEC and "
                     "is identical for every run that freezes the same codec; "
                     "this is the number that moves with stage-2 choices. Its "
                     "comparable wind-only bar is the one printed beside the "
                     "codec figure in probe_kfold.json."),
        }
        print(f"  head k-fold RAPID: {r_kf:.3f} [{lo_kf:.3f}, {hi_kf:.3f}] "
              f"over {n_kf} months (single-split was {r_des:.3f} over "
              f"{int(te.sum())})", flush=True)
    except Exception as e:                                    # noqa: BLE001
        # NOT fatal, and it says why. This runs at the very end of a job that
        # may have spent a day; a probe that cannot compute must not take the
        # results file with it (ml/CLAUDE.md §5.17). But it must never write
        # a NaN either (§5.22) — the key is simply absent, which a reader
        # cannot mistake for a measurement.
        print(f"::warning::head-level k-fold failed: {type(e).__name__}: {e}",
              flush=True)

    # ---- eval 3c: THE SAME FORWARD, UNPOOLED (E-055) ----------------------
    # ml/CLAUDE.md §3 / §8.3: the stage-2 transport read-out was the last
    # pooled instrument still quoted as a verdict, and the one comparison
    # "still mismatched by construction" — the sweep table's stage-2 column is
    # labelled `legacy_pooled_stage2` because there was no unpooled
    # counterpart to compare it against. This is that counterpart. It reads
    # the SAME hidden states through probe_head's learned attention pool
    # instead of a mean, on the SAME year-blocked folds, the SAME
    # deseasonalised target and the SAME block bootstrap, so the only thing
    # that differs between `rapid_r_kfold` and `rapid_r_kfold_unpooled` is
    # whether the section was averaged away before the fit.
    #
    # NEW KEYS ONLY. `rapid_probe`, `rapid_probe_kfold` and every field in
    # them are written above and are not touched here; 98 archived runs read
    # them. The fit restores every global RNG it perturbs (temporal._keep_rng),
    # so the checkpoint's `torch_rng` and anything else downstream is what it
    # would have been had this block not run.
    try:
        if FP is None:
            raise RuntimeError("per-pixel section states were not collected")
        _u_ri = np.array([int(t) in _u_pos for t in ri])
        if not _u_ri.all():
            raise RuntimeError(f"{int((~_u_ri).sum())} scored RAPID rows have "
                               f"no collected section state")
        _tok = section_tokens(FP[[_u_pos[int(t)] for t in ri]],
                              lon_fraction(lons[xs[sec_after]]))
        _uy = np.array([int(months[i][:4]) for i in ri])
        _udev = unpooled_device()
        _t_u = time.time()
        _kf = attn_pool_kfold(_tok, rv_des[ok], _uy, seed=a.seed,
                              device=_udev)
        # ...and the single 36-month split too, so `rapid_r_deseas` has an
        # unpooled partner on ITS protocol as well as the k-fold having one.
        # Same holdout mask, same target, same head — the noisy instrument
        # measured with the read-out §3 trusts.
        _tr_u, _te_u = ~t_hold[ri], t_hold[ri]
        _r_des_u = None
        if _tr_u.sum() >= 8 and _te_u.sum() >= 8:
            _mu, _sd = float(rv_des[ok][_tr_u].mean()), \
                float(rv_des[ok][_tr_u].std() + 1e-9)
            _net, _ = fit_attn_pool(_tok[_tr_u],
                                    (rv_des[ok][_tr_u] - _mu) / _sd,
                                    seed=a.seed, device=_udev)
            _p_u = attn_pool_predict(_net, _tok[_te_u], _udev) * _sd + _mu
            if np.std(rv_des[ok][_te_u]) > 0:
                _r_des_u = float(np.corrcoef(_p_u, rv_des[ok][_te_u])[0, 1])
        results["rapid_probe_kfold_unpooled"] = {
            "rapid_r_kfold_unpooled": round(_kf["r"], 3),
            "rapid_r_kfold_unpooled_ci": [round(_kf["lo"], 3),
                                          round(_kf["hi"], 3)],
            "rapid_r_deseas_unpooled": (None if _r_des_u is None
                                        else round(_r_des_u, 4)),
            "n": _kf["n"], "folds": _kf["folds"],
            "rmse_sv": round(_kf["rmse"], 2),
            "sigma_sv": round(_kf["sigma"], 2),
            "pooled": False,
            "features": "hidden(-1) per section pixel, learned attention pool",
            "readout": {"head": "probe_head.SectionHead", "d": UNPOOLED_HEAD_DIM,
                        "steps_max": UNPOOLED_STEPS,
                        "patience": UNPOOLED_PATIENCE,
                        "seed_base": int(a.seed),
                        "fold_seed": "seed + fold index (sorted unique years)",
                        "device": _udev.type,
                        "section_pixels": int(len(sec_after)),
                        "wall_s": round(time.time() - _t_u, 1)},
            # The out-of-fold arrays, for the same reason probe_head writes
            # them: pooled-vs-unpooled is a PAIRED comparison over shared
            # folds and months, and without these it can only ever be two
            # overlapping intervals (ml/CLAUDE.md §3, §8).
            "pred": [round(float(v), 4) for v in _kf["pred"]],
            "target_sv": [round(float(v), 4) for v in _kf["target"]],
            "years": [int(v) for v in _kf["years"]],
            "note": ("UNPOOLED stage-2 transport read-out (E-055). Same "
                     "hidden states, same year-blocked folds, same "
                     "deseasonalised target and same block bootstrap as "
                     "`rapid_probe_kfold` above — one learned softmax "
                     "attention over the section's pixels in place of "
                     "`hid[:, -1].mean(0)`. Compare the two with "
                     "scripts/paired_probe.py, never as two intervals."),
        }
        print(f"  head k-fold RAPID UNPOOLED: {_kf['r']:.3f} "
              f"[{_kf['lo']:.3f}, {_kf['hi']:.3f}] over {_kf['n']} months, "
              f"{_kf['folds']} folds on {_udev.type} "
              f"({time.time() - _t_u:.0f}s) — pooled was "
              f"{results.get('rapid_probe_kfold', {}).get('r_kfold_deseas')}",
              flush=True)
    except Exception as e:                                    # noqa: BLE001
        # Same posture as the pooled k-fold above: never fatal, never a NaN,
        # the key simply absent. This one runs LAST, so a failure here cannot
        # cost anything that was already computed.
        print(f"::warning::unpooled stage-2 read-out failed: "
              f"{type(e).__name__}: {e}", flush=True)

    print(json.dumps(results, indent=2))
    # The verdict, next to the curve, for the same reason.
    m2({"stage2_result": {
        "d_model": a.d_model, "layers": a.layers, "K": K, "steps": a.steps,
        "params_M": round(n_par2 / 1e6, 3), "seed": a.seed, "tag": a.tag or "",
        "z_mse_model": results.get("z_t+1", {}).get("mse_model"),
        "z_mse_persistence": results.get("z_t+1", {}).get("mse_persistence"),
        "chan_mse_model": results.get("chan_t+1", {}).get("mse_model"),
        "chan_mse_persistence": results.get("chan_t+1", {}).get("mse_persistence"),
        "rapid_r_deseas": results.get("rapid_probe", {}).get("r_deseasonalised"),
        "rapid_r_raw": results.get("rapid_probe", {}).get("r_raw"),
        # The head-level k-fold rides in the same record, so the status page
        # and every downstream reader get the six-times-larger sample without
        # a second fetch — and so a run's headline number is the one that can
        # actually move with what stage 2 varies.
        "rapid_r_kfold": results.get("rapid_probe_kfold", {}).get("r_kfold_deseas"),
        "rapid_r_kfold_ci": results.get("rapid_probe_kfold", {}).get("ci95"),
        # E-055: the UNPOOLED partners, BESIDE the pooled keys above and never
        # instead of them — the pooled column bridges 98 archived runs and a
        # dash is honest where a run predates this read-out (ml/CLAUDE.md §3).
        # `.get` on a missing block yields None, which a reader cannot mistake
        # for a measurement.
        "rapid_r_kfold_unpooled": results.get(
            "rapid_probe_kfold_unpooled", {}).get("rapid_r_kfold_unpooled"),
        "rapid_r_kfold_unpooled_ci": results.get(
            "rapid_probe_kfold_unpooled", {}).get("rapid_r_kfold_unpooled_ci"),
        "rapid_r_deseas_unpooled": results.get(
            "rapid_probe_kfold_unpooled", {}).get("rapid_r_deseas_unpooled"),
        # Direct-horizon downstream numbers ride along (2026-08-11) so the
        # status page can show them without a second fetch: per-horizon
        # model/persistence z-MSE ratios, lower is better.
        "z_direct_ratio": ({h: round(v["mse_model"] / v["mse_persistence"], 4)
                            for h, v in results.get("z_direct", {}).items()}
                           or None),
        "scale": results.get("scale"),
        # E-057, and ONLY when the arm is one — see the stage2_config comment:
        # a `fgn_eps: 0` on every legacy record would be a changed record with
        # the flag off, and absence already reads as "not an fgn run".
        **({"fgn_eps": a.fgn_eps} if FGN else {}),
    }})
    suffix = f"_{a.tag}" if a.tag else ""
    results["seed"] = a.seed
    torch.save({"model": model.state_dict(), "args": vars(a),
                "step": a.steps, "opt": opt.state_dict(),
                "sched": sched.state_dict(),
                "run_number": os.environ.get("GITHUB_RUN_NUMBER"),
                "torch_rng": torch.get_rng_state().numpy().tolist(),
                **_eps_state()},
               os.path.join(run_dir, f"temporal{suffix}.pt"))
    json.dump(results, open(os.path.join(run_dir, f"temporal{suffix}.json"), "w"), indent=2)
    print(f"saved {run_dir}/temporal{suffix}.pt")


if __name__ == "__main__":
    main()

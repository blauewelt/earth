#!/usr/bin/env python3
"""E-069 · is the cone codec's HIDDEN-DOT path wired correctly?

Run #539 (E-069 cone codec on the 42-channel North Atlantic tensor — the
codec that reads a pixel's lag-0 3x3 patch plus ~706 "dot" samples at lags
1-6 and is asked to reconstruct a subsample of the dots it was not shown)
finished 20,000 steps with a held-out per-family error of 0.626 on the
anchor's own value, 0.919 on the two future pentads and **1.001 on the hidden
dots** — the last being exactly the error of guessing the climatological mean
in the standardised (anomaly) space the trainer works in. Two explanations
have to be told apart before anything is spent on the next run:

  (i)  the DOT PATH IS BROKEN — the value gathered for a hidden dot is not
       the cell the decoder's question names, or the question never carries
       the dot's position, or the value is still visible in the input, or the
       error is averaged over targets that are not there;
  (ii) real dots at that spacing are simply not predictable.

This file answers (i) by construction and (ii) by a control. Four exact tests
plant a tensor whose every cell carries its own address, so "the value that
came back" and "the cell the question named" are literally comparable; a
fifth measures that a hidden dot is absent from the encoder's input. Then one
short training run on a SYNTHETIC FIELD THAT IS SMOOTH AT DOT SPACING — a
low-order polynomial in (y, x) whose coefficients drift slowly in time, so a
hidden dot is determined by its visible neighbours — asserts that the hidden-
dot error falls to well under its starting value, and does so by the same
factor the anchor's own value does.

    python3 -m pytest -q tests/test_cone_dot_path.py

About 90 s on two CPU cores; the training control is ~65 s of that. Set
`CONE_DOT_PATH_SKIP_TRAIN=1` to skip it and keep the exact tests, which need
no optimisation and take a few seconds.
"""
import math
import os
import sys
import time

import numpy as np
import pytest
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ML = os.path.join(ROOT, "ml")
if ML not in sys.path:
    sys.path.insert(0, ML)

from cone import KM_PER_DEG, channel_depth_dbar, channel_family    # noqa: E402
from cone_sampler import ConeSampler                               # noqa: E402
from cone_codec import ConeMAE, default_plan                       # noqa: E402
from train_cone import (PENTAD_DAYS, PENTAD_EPOCH, SMOKE_CHANS,    # noqa: E402
                        draw_anchors, eval_loss, to_torch)

TINY = dict(d_model=64, n_heads=4, n_latents=16, n_layers=2, d_z=32,
            d_dec=64, dec_layers=2, n_fourier=6)
DLAT = 0.25


# --------------------------------------------------------------- fixtures --
def _address_tensor(T=24, H=80, W=100, chans=SMOKE_CHANS):
    """A tensor in which every cell holds its own (c, t, y, x) address.

    `code = ((c*T + t)*H + y)*W + x` is under 2^24 at these sizes, so float32
    carries it EXACTLY and the address can be read back with integer
    arithmetic. Nothing is NaN, so every cell is observed and `valid` is the
    only thing that can exclude a dot.
    """
    C = len(chans)
    c, t, y, x = np.meshgrid(np.arange(C), np.arange(T), np.arange(H),
                             np.arange(W), indexing="ij")
    code = (((c * T + t) * H + y) * W + x).astype(np.float32)
    X = np.ascontiguousarray(np.moveaxis(code, 0, -1))          # [T,H,W,C]
    assert X.max() < 2 ** 24
    lats = 30.0 + DLAT * np.arange(H)
    lons = -60.0 + DLAT * np.arange(W)
    return X, lats, lons, list(chans), (T, H, W, C)


def _decode(code, T, H, W):
    """(c, t, y, x) from the planted address."""
    code = np.asarray(code, np.int64)
    x = code % W
    y = (code // W) % H
    t = (code // (W * H)) % T
    c = code // (W * H * T)
    return c, t, y, x


def _address_batch(L_in=6, n=24, seed=0):
    X, lats, lons, chans, shape = _address_tensor()
    T, H, W, C = shape
    sam = ConeSampler(X, np.isfinite(X), lats, lons, chans, L_in=L_in,
                      future_lags=(1, 2))
    rng = np.random.default_rng(seed)
    anchors = np.stack([rng.integers(L_in, T - 2, n),
                        rng.integers(0, H, n),
                        rng.integers(0, W, n)], axis=1)
    s = sam.sample(anchors)
    return sam, anchors, s, shape, chans


def _chan_depth(chans):
    return torch.as_tensor([channel_depth_dbar(n) for n in chans],
                           dtype=torch.float32)


# ------------------------------------------------- 1. the gather is aligned --
def test_dot_value_is_the_cell_its_own_coordinates_name():
    """The value handed to the codec for dot j IS X[t-lag_j, y+dy_j, x+dx_j,
    chan_j], where (lag_j, dy_j, dx_j, chan_j) are read back out of the very
    coordinates the codec's Fourier encoding is given.

    This is the identity that a misalignment would break, and it is checked
    against the tensor's own planted addresses rather than against a second
    copy of `ConeSampler`'s indexing arithmetic.
    """
    sam, anchors, s, (T, H, W, C), chans = _address_batch()
    cell = KM_PER_DEG * DLAT
    B, N = s["vals"].shape
    assert N > 400, f"expected the full inner cone, got {N} dots"

    for b in range(B):
        t0, y0, x0 = anchors[b]
        coslat = max(math.cos(math.radians(sam.lats[y0])), 0.05)
        v = s["valid"][b]
        assert v.sum() > 0
        # the coordinates the DECODER is given, converted back to cells
        lag = np.rint(s["lag_days"][b] / PENTAD_DAYS).astype(np.int64)
        dy = np.rint(s["dy_km"][b] / cell).astype(np.int64)
        dx = np.rint(s["dx_km"][b] / (cell * coslat)).astype(np.int64)
        ch = s["chan"][b].astype(np.int64)
        gc, gt, gy, gx = _decode(s["vals"][b][v], T, H, W)
        np.testing.assert_array_equal(gc, ch[v])
        np.testing.assert_array_equal(gt, (t0 - lag)[v])
        np.testing.assert_array_equal(gy, (y0 + dy)[v])
        np.testing.assert_array_equal(gx, (x0 + dx)[v])
        # and the depth carried per dot is that channel's, not another's
        np.testing.assert_allclose(
            s["depth"][b], [channel_depth_dbar(chans[c]) for c in ch])


def test_patch_and_future_targets_name_the_cells_they_claim():
    """The other two query families gather what they say too: the lag-0 3x3 is
    the anchor's own 3x3 at bin t, and the future target at +f pentads is the
    anchor column at bin t+f. If these were misaligned the anchor family's
    0.626 would be meaningless as well as the dots' 1.001."""
    sam, anchors, s, (T, H, W, C), chans = _address_batch()
    for b in range(8):
        t0, y0, x0 = anchors[b]
        c, t, y, x = _decode(s["patch_vals"][b][:, 4], T, H, W)   # centre cell
        np.testing.assert_array_equal(c, np.arange(C))
        assert (t == t0).all() and (y == y0).all() and (x == x0).all()
        for fi, f in enumerate((1, 2)):
            c, t, y, x = _decode(s["fut_vals"][b][:, fi], T, H, W)
            np.testing.assert_array_equal(c, np.arange(C))
            assert (t == t0 + f).all() and (y == y0).all() and (x == x0).all()


# -------------------------------------- 2. the query and the target agree --
def _given_masks(model, b, plan, seed=3):
    g = torch.Generator().manual_seed(seed)
    p = dict(plan)
    p["generator"] = g
    chan_mask, dot_mask = model._masks(b, p)
    idx, sel = model.draw_dot_queries(b, p, dot_mask)
    return p, chan_mask, dot_mask, idx, sel


def test_hidden_dot_query_and_target_are_the_same_dot():
    """Every column of the `dots` query family carries the channel, the two
    ground offsets, the lag and the depth OF THE DOT WHOSE VALUE IS ITS
    TARGET — one gather index, seven arrays."""
    sam, anchors, s, shape, chans = _address_batch()
    b = to_torch(s, _chan_depth(chans), "cpu")
    model = ConeMAE(len(chans), **TINY)
    plan = default_plan(chans, n_dot_queries=64)
    p, cm, dm, idx, sel = _given_masks(model, b, plan)
    chan, dy, dx, lag, dep, tgt, w = model._query_sets(b, p, dm, (idx, sel))
    lo, hi = model.query_family_spans(b, p, w.shape[1])["dots"]
    assert hi - lo == idx.shape[1] > 0

    g = lambda k: b[k].gather(1, idx)                            # noqa: E731
    assert torch.equal(chan[:, lo:hi], b["chan"].long().gather(1, idx))
    assert torch.equal(dy[:, lo:hi], g("dy_km"))
    assert torch.equal(dx[:, lo:hi], g("dx_km"))
    assert torch.equal(lag[:, lo:hi], g("lag_days"))
    assert torch.equal(dep[:, lo:hi], g("depth"))
    # the TARGET is the same array the encoder's dot token reads, so the
    # inputs and the targets cannot be in different normalisations
    assert torch.equal(tgt[:, lo:hi], g("vals"))


def test_dots_are_scored_only_where_hidden_valid_and_observed():
    """`held_out_mse_dots` divides by the weight, and the weight of a dot the
    data never observed, a dot that does not exist, or a dot we did not hide
    is exactly zero. So the number is a mean over real hidden targets, not one
    diluted by padding."""
    sam, anchors, s, shape, chans = _address_batch()
    b = to_torch(s, _chan_depth(chans), "cpu")
    model = ConeMAE(len(chans), **TINY)
    plan = default_plan(chans, n_dot_queries=64)
    p, cm, dm, idx, sel = _given_masks(model, b, plan)
    _, _, _, _, _, _, w = model._query_sets(b, p, dm, (idx, sel))
    lo, hi = model.query_family_spans(b, p, w.shape[1])["dots"]
    scored = w[:, lo:hi] > 0
    assert scored.any(), "no hidden dot was scored at all"
    eligible = (dm & b["obs"] & b["valid"]).gather(1, idx)
    assert torch.equal(scored, eligible & sel)
    assert torch.equal(scored, sel)          # the draw already enforces it


# ----------------------------------- 3. a hidden dot is NOT in the input --
def test_hidden_dot_is_absent_from_the_encoder_input():
    """Reconstructing a hidden dot cannot be a copy: its value reaches no
    token. Perturbing a HIDDEN dot's value leaves the code bit-identical;
    perturbing a VISIBLE one moves it. Both directions are asserted, because
    only the pair rules out "the mask is applied to everything"."""
    sam, anchors, s, shape, chans = _address_batch()
    b = to_torch(s, _chan_depth(chans), "cpu")
    model = ConeMAE(len(chans), **TINY).eval()
    B, N = b["vals"].shape
    dot_mask = torch.zeros(B, N, dtype=torch.bool)
    live = (b["valid"] & b["obs"])
    j = int(torch.nonzero(live[0]).flatten()[3])
    dot_mask[0, j] = True
    base = dict(b, chan_mask=torch.zeros(B, len(chans), dtype=torch.bool),
                dot_mask=dot_mask)
    with torch.no_grad():
        z0, _ = model.encode(base)
        bump = b["vals"].clone()
        bump[0, j] += 1234.0
        z_hidden, _ = model.encode(dict(base, vals=bump))
        z_visible, _ = model.encode(
            dict(base, vals=bump, dot_mask=torch.zeros_like(dot_mask)))
    assert torch.equal(z0, z_hidden), (
        "a hidden dot's value reached the code — reconstruction would be a "
        "copy and the dots family would be meaningless")
    assert not torch.allclose(z0, z_visible), (
        "a VISIBLE dot's value did not reach the code either — the encoder "
        "is ignoring its dot tokens, not masking them")


def test_decoder_query_carries_the_dot_position():
    """The decoder's answer depends on WHERE the question points. Two queries
    that differ only in the offset (or only in the lag) get different means
    from the same code, so the dot's position is genuinely delivered."""
    sam, anchors, s, shape, chans = _address_batch()
    b = to_torch(s, _chan_depth(chans), "cpu")
    model = ConeMAE(len(chans), **TINY).eval()
    B = b["vals"].shape[0]
    with torch.no_grad():
        z, _ = model.encode(dict(
            b, chan_mask=torch.zeros(B, len(chans), dtype=torch.bool),
            dot_mask=torch.zeros_like(b["obs"])))
        ch = torch.zeros(B, 1, dtype=torch.long)
        zero = torch.zeros(B, 1)
        mus = {}
        for name, (dy, dx, lag) in {
                "origin": (0.0, 0.0, 5.0),
                "north": (300.0, 0.0, 5.0),
                "east": (0.0, 300.0, 5.0),
                "older": (0.0, 0.0, 30.0)}.items():
            mu, _ = model.decode_from_z(z, model.query_tokens(
                ch, zero + dy, zero + dx, zero + lag, zero))
            mus[name] = mu
    for other in ("north", "east", "older"):
        assert not torch.allclose(mus["origin"], mus[other], atol=1e-6), (
            f"the decoder's mean did not move when the query moved to "
            f"{other} — the dot's position is not reaching the head")


# ------------------------------------------- 4. the predictable-field control --
def smooth_tensor(path, seed=0, T=240, H=40, W=56, noise=0.02):
    """A tensor that is SMOOTH AT DOT SPACING, written in `smoke_tensor`'s
    shape conventions ([T, H, W, C] at 0.25 degrees from 30N, the same eight
    named channels, a pentad axis from 1982-01-01, a NaN land block and ~1%
    scattered dropouts).

    Every channel is a low-order polynomial in (y, x) — constant, both linear
    terms, the cross term and both quadratics — whose six coefficients drift
    as slow sinusoids in t (periods 11, 17 and 29 pentads, deliberately away
    from 73 pentads = one year, so nothing seasonal and therefore nothing in
    the context token predicts them).

    Why a POLYNOMIAL and not a sinusoid in space, which was tried first: the
    inner cone's dots at lag 6 sit ~330 km apart, so a field with a 500 km
    wavelength is aliased at the stencil and its nearest visible neighbour is
    as often anti-correlated as correlated. Measured on that version, the
    nearest-visible-dot baseline scored WORSE than the climatological mean
    (1.24x its error), which would have made this a control that controls
    nothing. A polynomial varies on the scale of the whole window, so a hidden
    dot really is determined by the dots around it.
    """
    rng = np.random.default_rng(seed)
    C = len(SMOKE_CHANS)
    lats = 30.0 + DLAT * np.arange(H)
    lons = -60.0 + DLAT * np.arange(W)
    tt = np.arange(T)[:, None]
    per = np.array([11.0, 17.0, 29.0])

    yy, xx = np.meshgrid(np.arange(H), np.arange(W), indexing="ij")
    yh = 2.0 * yy / (H - 1.0) - 1.0
    xh = 2.0 * xx / (W - 1.0) - 1.0
    basis = [np.ones_like(yh), yh, xh, yh * xh,
             yh ** 2 - 1.0 / 3.0, xh ** 2 - 1.0 / 3.0]
    X = np.empty((T, H, W, C), np.float32)
    for c in range(C):
        co = np.stack([(rng.uniform(0.6, 1.4, 3)
                        * np.sin(2 * np.pi * tt / per
                                 + rng.uniform(0, 2 * np.pi, 3))).sum(1)
                       for _ in basis], axis=1)                   # [T, 6]
        for t in range(T):
            X[t, :, :, c] = (sum(co[t, k] * basis[k]
                                 for k in range(len(basis)))
                             + noise * rng.normal(size=(H, W)))
    X[:, :4, :4, :] = np.nan                                      # land
    X[rng.random(X.shape) < 0.01] = np.nan                        # dropouts

    days = PENTAD_EPOCH + (PENTAD_DAYS * np.arange(T)).astype("timedelta64[D]")
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    np.savez_compressed(path, X=X, months=np.array([str(d) for d in days]),
                        lats=lats, lons=lons, chan=np.array(SMOKE_CHANS))
    return path


def _anomaly_space(path, hold_year="1983"):
    """THE ONE anomaly transform (`trainprobe.anomaly_transform`, the function
    `train_cone.load_data` calls), applied here exactly as the trainer applies
    it: dynamic channels become departures from their own TRAIN-years monthly
    climatology and are then z-scored on train data. Inputs and targets are
    the same array afterwards, which is why nothing downstream can put them in
    different units."""
    from trainprobe import anomaly_transform
    d = np.load(path, allow_pickle=False)
    X = d["X"].copy()
    months = [str(m) for m in d["months"]]
    hold = np.array([m[:4] == hold_year for m in months])
    assert hold.any() and (~hold).any()
    moy = np.array([int(m[5:7]) - 1 for m in months])
    X, _ = anomaly_transform(X, moy, hold, np.zeros(X.shape[2], bool))
    return (X, np.isfinite(X), np.asarray(d["lats"]), np.asarray(d["lons"]),
            [str(c) for c in d["chan"]], hold)


KM_PER_DAY_B = 0.3 * 86400.0 / 1000.0     # family B's own drift, 25.92 km/day


def nearest_visible_dot_mse(model, sampler, anchors, plan, chan_depth,
                            batch=32, seed=12345):
    """THE BASELINE: predict each hidden dot with the value of the SAME
    CHANNEL at the nearest VISIBLE point of the cone.

    Distance is the cone's own space-time metric — kilometres in the plane and
    the lag converted at family B's 0.3 m/s (25.92 km/day), the speed
    `ml/cone.py` builds the reach from — and the candidates are every valid,
    observed, unhidden dot of that channel plus the lag-0 patch centre when
    that channel is not itself hidden. A hidden dot with no same-channel
    candidate left (its whole channel was dropped) falls back to the
    climatological mean, which is 0 in anomaly space; `frac_with_neighbour`
    reports how often that happens.

    ON THE REAL TENSOR this needs no training and no checkpoint: build a
    `ConeSampler` over the r3 tensor exactly as `train_cone.train_one` does,
    take the held-out anchor draw, and call this with the run's own
    `default_plan` — it reads only the sampler's batch and the mask draw.
    """
    g = torch.Generator().manual_seed(seed)
    p = dict(plan)
    p["generator"] = g
    sq = wsum = n_nb = n = 0.0
    with torch.no_grad():
        for i in range(0, len(anchors), batch):
            b = to_torch(sampler.sample(anchors[i:i + batch]), chan_depth,
                         "cpu")
            chan_mask, dot_mask = model._masks(b, p)
            idx, sel = model.draw_dot_queries(b, p, dot_mask)
            if idx.shape[1] == 0:
                continue
            _, _, _, _, _, tgt, w = model._query_sets(b, p, dot_mask,
                                                      (idx, sel))
            lo, hi = model.query_family_spans(b, p, w.shape[1])["dots"]
            ww, tg = w[:, lo:hi], tgt[:, lo:hi]

            dy, dx, lg = b["dy_km"], b["dx_km"], b["lag_days"]
            ch = b["chan"].long()
            vis = b["valid"] & b["obs"] & ~dot_mask
            qdy, qdx, qlg, qch = (dy.gather(1, idx), dx.gather(1, idx),
                                  lg.gather(1, idx), ch.gather(1, idx))
            d2 = ((qdy[:, :, None] - dy[:, None, :]) ** 2
                  + (qdx[:, :, None] - dx[:, None, :]) ** 2
                  + ((qlg[:, :, None] - lg[:, None, :]) * KM_PER_DAY_B) ** 2)
            ok = vis[:, None, :] & (qch[:, :, None] == ch[:, None, :])
            best, arg = torch.where(ok, d2, torch.full_like(d2, float("inf"))
                                    ).min(dim=2)
            has = torch.isfinite(best)
            pred = torch.where(has, b["vals"].gather(1, arg),
                               torch.zeros_like(best))
            pc_ok = (b["patch_obs"][..., 4] & ~chan_mask).gather(1, qch)
            pc_d2 = qdy ** 2 + qdx ** 2 + (qlg * KM_PER_DAY_B) ** 2
            use_pc = pc_ok & ((pc_d2 < best) | ~has)
            pred = torch.where(use_pc, b["patch_vals"][..., 4].gather(1, qch),
                               pred)
            has = has | use_pc

            sq += float(((pred - tg) ** 2 * ww).sum())
            wsum += float(ww.sum())
            n_nb += float((has & (ww > 0)).sum())
            n += float((ww > 0).sum())
    return {"mse": sq / max(wsum, 1e-9), "wsum": wsum,
            "frac_with_neighbour": n_nb / max(n, 1.0), "n": int(n)}


def climatological_mse(model, sampler, anchors, plan, chan_depth, batch=32,
                       seed=12345):
    """The error of predicting 0 — the climatological mean in anomaly space —
    on the SAME targets and with the SAME weights `eval_loss` uses. Held-out
    variance is not exactly 1 on a short synthetic record (the monthly
    climatology is estimated from few bins), so the honest reference is the
    measured one rather than the nominal 1.0."""
    g = torch.Generator().manual_seed(seed)
    p = dict(plan)
    p["generator"] = g
    out = {}
    with torch.no_grad():
        for i in range(0, len(anchors), batch):
            b = to_torch(sampler.sample(anchors[i:i + batch]), chan_depth,
                         "cpu")
            chan_mask, dot_mask = model._masks(b, p)
            idx, sel = model.draw_dot_queries(b, p, dot_mask)
            _, _, _, _, _, tgt, w = model._query_sets(b, p, dot_mask,
                                                      (idx, sel))
            for k, (lo, hi) in model.query_family_spans(
                    b, p, w.shape[1]).items():
                a = out.setdefault(k, [0.0, 0.0])
                if hi > lo:
                    a[0] += float((tgt[:, lo:hi] ** 2 * w[:, lo:hi]).sum())
                    a[1] += float(w[:, lo:hi].sum())
    return {k: (s / v if v > 0 else 0.0) for k, (s, v) in out.items()}


@pytest.mark.skipif(bool(os.environ.get("CONE_DOT_PATH_SKIP_TRAIN")),
                    reason="CONE_DOT_PATH_SKIP_TRAIN is set")
def test_hidden_dot_error_falls_on_a_field_where_dots_are_predictable(tmp_path):
    """THE CONTROL. On a field whose hidden dots really are determined by
    their visible neighbours, `held_out_mse_dots` must fall clearly — and must
    fall by about the same factor as the anchor family, which is the claim
    that the two query families are on an equal footing.

    Three deliberate departures from a dispatch, each so that a CPU can answer
    inside two minutes, and none of them touching the path under test:

      * L_in = 2 rather than 6 — 102 dot tokens instead of 706 at this channel
        list. The sunflower geometry, the coordinate encoding, the gather and
        the query construction are the same code at either depth.
      * the SECTOR drop alone (a 90-degree bearing wedge of the dots, every
        batch element), rather than the trainer's mixture. Under the trainer's
        default plan 80% of hidden-dot queries belong to a channel that was
        dropped ENTIRELY, so their reconstruction is a cross-channel inference
        and not "predict a dot from the dots around it" at all. The sector
        drop is the scheme this control is about.
      * `aux_latent_w` at 1.0. A 32-number bottleneck trained from scratch
        spends its first few hundred steps predicting the mean with the code
        collapsed (measured here: the code's variance falls to ~1e-5 and stays
        there for over 1,500 steps at the shipped 0.25); the stronger
        auxiliary term through the decoder's full memory gets past that inside
        the budget. It changes how fast the codec escapes, not what it is
        asked to reconstruct.
    """
    t0 = time.time()
    path = smooth_tensor(str(tmp_path / "smooth.npz"), T=240)
    X, OBS, lats, lons, chans, hold = _anomaly_space(path)
    T = X.shape[0]
    L_in, steps, batch = 2, 900, 32

    sam = ConeSampler(X, OBS, lats, lons, chans, L_in=L_in, future_lags=(1, 2))
    ys, xs = np.nonzero(np.isfinite(X[..., 0]).any(0))
    train_t = np.array([t for t in range(L_in, T - 2)
                        if (~hold)[t - L_in:t + 3].all()])
    ev_t = np.flatnonzero(hold)
    ev_t = ev_t[(ev_t - L_in >= 0) & (ev_t + 2 < T)]
    assert len(train_t) > 50 and len(ev_t) > 20
    ev = draw_anchors(np.random.default_rng(991), ev_t, ys, xs, 192)

    torch.manual_seed(0)
    model = ConeMAE(len(chans), **TINY)
    plan = default_plan(chans, cur_drop=0.0, other_drop=0.0, lag_band_p=0.0,
                        sector_p=1.0, n_dot_queries=64, aux_latent_w=1.0)
    cd = _chan_depth(chans)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps)

    # the SAME measurement the trainer writes as held_out_mse_<family>
    _, _, _, fam0 = eval_loss(model, sam, ev, plan, cd, "cpu", batch)
    zero = climatological_mse(model, sam, ev, plan, cd, batch)
    nn = nearest_visible_dot_mse(model, sam, ev, plan, cd, batch)

    rng = np.random.default_rng(0)
    for s in range(1, steps + 1):
        a = draw_anchors(rng, train_t, ys, xs, batch)
        b = to_torch(sam.sample(a), cd, "cpu")
        out = model(b, plan)
        opt.zero_grad(set_to_none=True)
        out["loss"].backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
        assert np.isfinite(float(out["loss"].detach())), f"NaN at step {s}"
    _, _, _, fam = eval_loss(model, sam, ev, plan, cd, "cpu", batch)

    d0, d1 = fam0["dots"]["mse"], fam["dots"]["mse"]
    a0, a1 = fam0["anchor"]["mse"], fam["anchor"]["mse"]
    print(f"\n[dot path] {steps} steps in {time.time()-t0:.0f}s · "
          f"dots {d0:.3f} -> {d1:.3f} (mean {zero['dots']:.3f}) · "
          f"anchor {a0:.3f} -> {a1:.3f} (mean {zero['anchor']:.3f}) · "
          f"nearest visible dot {nn['mse']:.3f} on "
          f"{nn['frac_with_neighbour']:.0%} of the queries", flush=True)

    # (a) the control is a control: a hidden dot really is predictable here
    assert nn["frac_with_neighbour"] > 0.95
    assert nn["mse"] < 0.6 * zero["dots"], (
        f"the synthetic field is not smooth at dot spacing — the nearest "
        f"visible dot scores {nn['mse']:.3f} against a climatological "
        f"{zero['dots']:.3f}, so this run could not tell a broken dot path "
        f"from an unpredictable field")

    # (b) the hidden-dot error falls, clearly
    assert d1 < 0.85 * d0, (
        f"held_out_mse_dots did not fall on a field where hidden dots are "
        f"determined by their visible neighbours: {d0:.4f} -> {d1:.4f}. That "
        f"is the signature of a broken dot query -> target path.")
    assert d1 < 0.90 * zero["dots"], (
        f"held_out_mse_dots {d1:.4f} is not below the climatological "
        f"{zero['dots']:.4f} — the dots are being predicted at the mean")

    # (c) and it falls by about as much as the anchor's own value does, which
    #     is the statement that the dots family is not handicapped
    assert abs(d1 / zero["dots"] - a1 / zero["anchor"]) < 0.20, (
        f"dots keep {d1/zero['dots']:.3f} of the climatological error while "
        f"the anchor keeps {a1/zero['anchor']:.3f} — the two families are "
        f"not on an equal footing")


def test_default_plan_hides_whole_channels_far_more_often_than_dots():
    """WHY THE REAL DOTS ARE HARD, as a number rather than an argument.

    `ConeMAE._masks` mixes three schemes, and the first is a whole-CHANNEL
    drop at probability 0.3 (0.5 for the two current components) that removes
    that channel at lag 0 AND at every dot. Measured here on the r3 channel
    list's own geometry (42 channels, 706 dots per anchor), four fifths of the
    hidden-dot queries come from that scheme, so for four fifths of them there
    is no value of the same channel anywhere in the codec's input and the
    reconstruction is a cross-channel inference. This is a property of the
    masking plan, not of the ocean, and it is the first thing to change if
    the dots family is meant to measure spatial interpolation.
    """
    from build_family4 import CHANS_BY_REV
    chans = list(CHANS_BY_REV["r3"])
    assert len(chans) == 42
    T, H, W = 16, 60, 80
    rng = np.random.default_rng(0)
    X = rng.normal(size=(T, H, W, len(chans))).astype(np.float32)
    sam = ConeSampler(X, np.isfinite(X), 30.0 + DLAT * np.arange(H),
                      -60.0 + DLAT * np.arange(W), chans, L_in=6,
                      future_lags=(1, 2))
    assert sam.n_dots(40) == 706
    anchors = np.stack([rng.integers(6, T - 2, 48), rng.integers(0, H, 48),
                        rng.integers(0, W, 48)], axis=1)
    b = to_torch(sam.sample(anchors), _chan_depth(chans), "cpu")
    model = ConeMAE(len(chans), d_model=32, n_heads=4, n_latents=8,
                    n_layers=1, d_z=32, d_dec=32, dec_layers=1, n_fourier=6)
    plan = default_plan(chans, n_dot_queries=256)
    p, chan_mask, dot_mask, idx, sel = _given_masks(model, b, plan, seed=7)
    qch = b["chan"].long().gather(1, idx)
    from_channel_drop = float((chan_mask.gather(1, qch) & sel).sum()
                              / sel.sum())
    vis = b["valid"] & b["obs"] & ~dot_mask
    same = ((qch[:, :, None] == b["chan"].long()[:, None, :])
            & vis[:, None, :]).any(2)
    with_neighbour = float((same & sel).sum() / sel.sum())
    print(f"\n[masking] {float(dot_mask.float().mean()):.2f} of dots hidden · "
          f"{from_channel_drop:.2f} of hidden-dot queries lost their whole "
          f"channel · {with_neighbour:.2f} still have a same-channel dot",
          flush=True)
    assert from_channel_drop > 0.6
    assert with_neighbour < 0.35
    # every family is represented in the draw, so the number is not one
    # channel's quirk
    fams = {channel_family(chans[int(c)]) for c in qch[sel].reshape(-1)[:2000]}
    assert fams == {"A", "B", "C"}


if __name__ == "__main__":                                # pragma: no cover
    raise SystemExit(pytest.main([__file__, "-q", "-s"]))

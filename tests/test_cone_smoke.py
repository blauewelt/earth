#!/usr/bin/env python3
"""E-069 · the cone codec and its trainer, exercised end to end on a toy.

Three checks, in the order ml/CLAUDE.md §4.8 puts them: the model runs, the
model's INVARIANTS hold with exact expected values (§4.9 — prefer an exact
identity to a threshold), and the whole trainer runs on a synthetic tensor
before any GPU is spent on it.

    python3 -m pytest -q tests/test_cone_smoke.py

Total runtime is about a minute on two CPU cores; the trainer's own smoke is
the expensive part (~40 s) and is deliberately inside the suite rather than
beside it, because a smoke test nobody runs is a comment.
"""
import json
import os
import subprocess
import sys

import numpy as np
import pytest
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ML = os.path.join(ROOT, "ml")
if ML not in sys.path:
    sys.path.insert(0, ML)

from cone import channel_depth_dbar                              # noqa: E402
from cone_sampler import ConeSampler                             # noqa: E402
from cone_codec import ConeMAE, default_plan, nll_gauss          # noqa: E402
from train_cone import to_torch, QUERY_FAMILIES                  # noqa: E402

# THE BIT-IDENTITY BASELINE. `--smoke --seed 0` is fully deterministic (the
# only record field that moves between two runs is `wall_s`), so the loss
# trajectory it produces is a fingerprint of the TRAINING code path. This file
# holds the trajectory measured on the commit BEFORE the H1 diagnostics
# (2026-09-03) went in, and `test_smoke_trajectory_is_bit_identical` replays
# it: a diagnostic that changed a number would be a diagnostic that changed
# the experiment, and the whole premise of that change was that neither arm's
# training moves. It is a per-machine fingerprint of a torch build as much as
# of this repo — a mismatch means either the training path moved (fix the
# code) or the environment did (regenerate, and say so in the commit).
TRAJECTORY_GOLDEN = os.path.join(
    ROOT, "tests", "data", "cone_smoke_trajectory.json")
TRAJECTORY_KEYS = ("loss_rec", "loss_nei", "held_out_nll", "held_out_mse",
                   "held_out_targets")

CHANS = ["cur_speed", "log_mld", "ssh", "tau_x", "tau_y", "sst",
         "cur_u", "cur_v"]

# PINNED PARAMETER COUNTS. Architecture is a claim, not a preference: a
# refactor that quietly changes the token projections or the block shape
# changes what every checkpoint means, and a count is the cheapest thing that
# notices. The default geometry (256 wide, 8 heads, 64 latents, 6 blocks,
# d_z 32, d_dec 256, 2 decoder blocks, 8 Fourier bands) at the r3 tensor's 42
# channels is 7,048,994 — the plan's "~5M" prices the 6 self-attention blocks
# (4.74M) and the token table; the Perceiver cross block (0.79M), the
# attention pool and the queryable decoder (~1.5M together) are the rest.
TINY_PARAMS = 253_538
DEFAULT_42_PARAMS = 7_048_994

TINY = dict(d_model=64, n_heads=4, n_latents=16, n_layers=2, d_z=32,
            d_dec=64, dec_layers=2, n_fourier=6)


def tiny_sampler(seed=0, T=30, H=12, W=14, const=None):
    """A ConeSampler over a small random (or constant) tensor."""
    rng = np.random.default_rng(seed)
    C = len(CHANS)
    if const is None:
        X = rng.normal(size=(T, H, W, C)).astype(np.float32)
        X[rng.random(X.shape) < 0.05] = np.nan          # unobserved cells
    else:
        X = np.full((T, H, W, C), float(const), np.float32)
    OBS = np.isfinite(X)
    lats = 30.0 + 0.25 * np.arange(H)
    lons = -40.0 + 0.25 * np.arange(W)
    return ConeSampler(X, OBS, lats, lons, CHANS, L_in=6)


def batch_of(sampler, anchors, device="cpu"):
    depth = torch.as_tensor([channel_depth_dbar(n) for n in CHANS],
                            dtype=torch.float32, device=device)
    return to_torch(sampler.sample(np.asarray(anchors, np.int64)), depth,
                    device)


def no_mask_plan():
    """Every masking probability at zero: nothing is hidden by US."""
    return default_plan(CHANS, cur_drop=0.0, other_drop=0.0, lag_band_p=0.0,
                        sector_p=0.0, n_dot_queries=16, aux_latent_w=0.25)


# ---------------------------------------------------------------- 1. forward --
def test_forward_backward_shapes_and_params():
    torch.manual_seed(0)
    s = tiny_sampler()
    rng = np.random.default_rng(1)
    B = 6
    anchors = np.stack([rng.integers(8, 27, B), rng.integers(0, 12, B),
                        rng.integers(0, 14, B)], axis=1)
    b = batch_of(s, anchors)
    N = b["vals"].shape[1]
    # the sampler pads to the LARGEST dot count in the batch (rows differ:
    # the zonal offsets go as cos(lat), so a row's spiral can dedupe onto
    # fewer cells) — an exact identity, not "N > 0"
    assert N == max(s.n_dots(int(y)) for y in anchors[:, 1])
    assert b["patch_vals"].shape == (B, len(CHANS), 9)
    assert b["fut_vals"].shape == (B, len(CHANS), 2)

    m = ConeMAE(len(CHANS), **TINY)
    assert m.param_count() == TINY_PARAMS, (
        f"ConeMAE{TINY} on {len(CHANS)} channels is {m.param_count():,} "
        f"params, pinned at {TINY_PARAMS:,} — the architecture moved")
    assert ConeMAE(42).param_count() == DEFAULT_42_PARAMS

    plan = default_plan(CHANS, n_dot_queries=32)
    out = m(b, plan)
    assert set(("loss", "z", "terms")) <= set(out)
    assert out["z"].shape == (B, TINY["d_z"])
    assert torch.isfinite(out["loss"]) and out["loss"].item() > 0
    for k in ("nll", "mse", "wsum", "n_targets", "nll_latent"):
        assert np.isfinite(out["terms"][k])

    z, lat = m.encode(b)
    assert z.shape == (B, TINY["d_z"])
    assert lat.shape == (B, TINY["n_latents"], TINY["d_model"])
    q = m.query_tokens(b["chan"].long(), b["dy_km"], b["dx_km"],
                       b["lag_days"], b["depth"])
    assert q.shape == (B, N, TINY["d_dec"])
    mu, logvar = m.decode_from_z(z, q)
    assert mu.shape == (B, N) and logvar.shape == (B, N)
    # the head's clamp is part of the loss's definition, not decoration
    lv = logvar.detach()
    assert float(lv.min()) >= -8.0 and float(lv.max()) <= 8.0

    out["loss"].backward()
    missing = [n for n, p in m.named_parameters() if p.grad is None]
    assert not missing, f"no gradient reached {missing}"


# ------------------------------------------------------------- 2. invariants --
def test_query_matching_an_observed_dot_is_finite_at_init():
    """A decoder query IDENTICAL to an observed input dot, no masking at all.

    The model is asked to reproduce something it was just shown; at init it
    cannot, and what is being pinned is that the loss is a finite number
    rather than the NaN a mishandled mask or an unclamped log-variance
    produces.
    """
    torch.manual_seed(0)
    s = tiny_sampler()
    b = batch_of(s, [[10, 5, 6], [12, 7, 8]])
    m = ConeMAE(len(CHANS), **TINY).eval()
    z, _ = m.encode(b)
    q = m.query_tokens(b["chan"].long(), b["dy_km"], b["dx_km"],
                       b["lag_days"], b["depth"])
    mu, logvar = m.decode_from_z(z, q)
    w = (b["obs"] & b["valid"]).float()
    assert w.sum() > 0
    loss = (nll_gauss(mu, logvar, b["vals"]) * w).sum() / w.sum()
    assert torch.isfinite(loss)
    out = m(b, no_mask_plan())
    assert torch.isfinite(out["loss"])


def test_constant_field_reconstruction_is_bit_reproducible():
    """Two forward passes in eval mode return EXACTLY the same numbers.

    `torch.equal`, not `allclose`: there is no dropout and no sampling on this
    path, so anything but bit-identity means a hidden source of randomness —
    and a codec whose embedding moves between two reads of the same pixel
    cannot be compared across checkpoints at all.
    """
    torch.manual_seed(0)
    s = tiny_sampler(const=1.0)
    b = batch_of(s, [[10, 5, 6], [12, 7, 8], [20, 3, 11]])
    m = ConeMAE(len(CHANS), **TINY).eval()
    with torch.no_grad():
        z1, _ = m.encode(b)
        z2, _ = m.encode(b)
        q = m.query_tokens(b["chan"].long(), b["dy_km"], b["dx_km"],
                           b["lag_days"], b["depth"])
        mu1, lv1 = m.decode_from_z(z1, q)
        mu2, lv2 = m.decode_from_z(z2, q)
    assert torch.equal(z1, z2)
    assert torch.equal(mu1, mu2) and torch.equal(lv1, lv2)
    # THE SAME ANCHOR TWICE IN ONE BATCH gets the same code, exactly: nothing
    # in this encoder mixes batch elements, and a codec whose embedding
    # depended on its batch neighbours would make every archived Z a function
    # of the loader's shuffle.
    with torch.no_grad():
        zz, _ = m.encode(batch_of(s, [[10, 5, 6], [10, 5, 6], [12, 7, 8]]))
    assert torch.equal(zz[0], zz[1]) and not torch.equal(zz[0], zz[2])

    # And the whole forward is reproducible when the plan's generator is
    # seeded — the masking draw is the only randomness in it.
    p = no_mask_plan()
    with torch.no_grad():
        p["generator"] = torch.Generator().manual_seed(7)
        l1 = m(b, p)["loss"]
        p["generator"] = torch.Generator().manual_seed(7)
        l2 = m(b, p)["loss"]
    assert torch.equal(l1, l2)


def test_invalid_dots_are_excluded_by_the_key_padding_mask():
    """Perturbing an INVALID dot's value must not move z AT ALL.

    The dot's token is built from whatever the padding holds — the model does
    not consult `valid` when it builds tokens, deliberately, so that the
    key-padding mask is the ONE mechanism that enforces existence and this
    test can be about that mechanism instead of about a second one. The dot is
    forced `obs=True` here for exactly that reason: with the sampler's own
    `obs=False` the token is `miss_tok` and carries no value, and the test
    would pass without any mask at all.
    """
    torch.manual_seed(0)
    s = tiny_sampler()
    # x = 0 puts every westward dot off the basin: those are invalid, and this
    # window is a basin, not a globe, so they are NOT wrapped (see
    # ml/cone_sampler.py's module docstring).
    b = batch_of(s, [[10, 5, 0], [12, 6, 0]])
    inval = (~b["valid"]).nonzero()
    assert len(inval) > 0, "no invalid dot in this batch — test is vacuous"
    i, j = int(inval[0][0]), int(inval[0][1])
    b["obs"][i, j] = True                       # make the token value-carrying

    m = ConeMAE(len(CHANS), **TINY).eval()
    with torch.no_grad():
        z1, lat1 = m.encode(b)
        b["vals"][i, j] = b["vals"][i, j] + 7.5
        z2, lat2 = m.encode(b)
    assert torch.equal(z1, z2), (
        "an invalid dot's value changed z — the key-padding mask is not "
        "excluding it, so a training anchor near the basin edge is encoding "
        "whatever the padding happened to hold")
    assert torch.equal(lat1, lat2)

    # And the control: perturbing a VALID observed dot must change z, or the
    # assertion above is measuring a model that ignores its input.
    val = (b["valid"] & b["obs"]).nonzero()
    assert len(val) > 0
    i2, j2 = int(val[0][0]), int(val[0][1])
    with torch.no_grad():
        b["vals"][i2, j2] = b["vals"][i2, j2] + 7.5
        z3, _ = m.encode(b)
    assert not torch.equal(z1, z3)


def test_finite_view_matches_the_eager_mask():
    """The lazy observed-mask is arithmetically identical to `isfinite(X)`.

    It is the path a real family-4 run takes (an eager mask is 16.6 GB at the
    pentad shape) and the smoke never takes it, so it is asserted here rather
    than assumed — ml/CLAUDE.md §0.2.
    """
    from train_cone import FiniteView
    rng = np.random.default_rng(0)
    X = rng.normal(size=(20, 8, 9, len(CHANS))).astype(np.float32)
    X[rng.random(X.shape) < 0.1] = np.nan
    lats, lons = 30.0 + 0.25 * np.arange(8), -40.0 + 0.25 * np.arange(9)
    anchors = np.array([[10, 3, 4], [12, 5, 6]])
    eager = ConeSampler(X, np.isfinite(X), lats, lons, CHANS, L_in=6)
    lazy = ConeSampler(X, FiniteView(X), lats, lons, CHANS, L_in=6)
    a, b = eager.sample(anchors), lazy.sample(anchors)
    for k in a:
        assert np.array_equal(a[k], b[k]), f"{k} differs under FiniteView"


def test_per_family_terms_reassemble_the_headline():
    """The three query families' weighted means put the headline back exactly.

    `_loss_from` reports `families` beside `terms`: the anchor reconstruction,
    the future targets and the hidden dots, each with its own weighted-mean
    nll and mse and the weight it divided by. They are a PARTITION of the
    query axis, so
        nll == sum_f nll_f * wsum_f / sum_f wsum_f
    is an exact identity, not an approximation — which is the point: a split
    that did not reassemble would be measuring some other set of targets, and
    a per-family number that cannot be checked against a number we already
    trust is a number nobody should act on.
    """
    torch.manual_seed(0)
    s = tiny_sampler()
    b = batch_of(s, [[10, 5, 6], [12, 7, 8], [20, 3, 11], [15, 9, 4]])
    m = ConeMAE(len(CHANS), **TINY).eval()
    plan = default_plan(CHANS, n_dot_queries=16)
    plan["generator"] = torch.Generator().manual_seed(11)
    with torch.no_grad():
        out = m(b, plan)
    fam = out["families"]
    assert set(fam) == set(QUERY_FAMILIES)
    for k, f in fam.items():
        for key in ("nll", "mse", "msebar", "wsum", "n_targets"):
            assert np.isfinite(f[key]), f"{k}.{key} is not finite"
        assert f["wsum"] > 0, f"{k} scored nothing — the split is degenerate"
    den = sum(f["wsum"] for f in fam.values())
    for term in ("nll", "mse"):
        got = sum(f[term] * f["wsum"] for f in fam.values()) / den
        assert abs(got - out["terms"][term]) < 1e-6, (
            f"{term}: families reassemble to {got!r}, headline "
            f"{out['terms'][term]!r}")
    # And the SPANS TILE the query axis — contiguous, non-overlapping, in the
    # order `_query_sets` concatenates them, covering every column. An overlap
    # that happened to sum right on this batch would pass the identity above.
    Q = 1000
    spans = m.query_family_spans(b, plan, Q)
    assert [spans[k] for k in QUERY_FAMILIES] == sorted(spans.values())
    lo_hi = [spans[k] for k in QUERY_FAMILIES]
    assert lo_hi[0][0] == 0 and lo_hi[-1][1] == Q
    for (l0, h0), (l1, _) in zip(lo_hi, lo_hi[1:]):
        assert h0 == l1, f"family spans are not contiguous: {spans}"
    # anchor is C wide and future is C * len(future_lags) wide, by definition
    assert spans["anchor"] == (0, len(CHANS))
    assert spans["future"][1] - spans["future"][0] == len(CHANS) * 2
    assert fam["anchor"]["n_targets"] > 0 and fam["dots"]["n_targets"] > 0


def test_empty_family_is_zeros_not_nan():
    """The snapshot twin has NO dots, and an empty family must not be a NaN.

    ml/CLAUDE.md §5.22: a results file full of NaN is loud enough to notice
    and quiet enough to misattribute. A weight of exactly zero is the honest
    discriminator between "this family scored 0.0" and "this family had
    nothing to score", so the loss reads 0.0 and the weight reads 0.0.
    """
    s = ConeSampler(*(lambda X: (X, np.isfinite(X)))(
        np.random.default_rng(0).normal(size=(20, 8, 9, len(CHANS))
                                        ).astype(np.float32)),
        30.0 + 0.25 * np.arange(8), -40.0 + 0.25 * np.arange(9), CHANS,
        L_in=0)
    b = batch_of(s, [[10, 3, 4], [11, 4, 5]])
    m = ConeMAE(len(CHANS), **TINY).eval()
    with torch.no_grad():
        out = m(b, default_plan(CHANS, n_dot_queries=8))
    d = out["families"]["dots"]
    assert d == {"nll": 0.0, "mse": 0.0, "msebar": 0.0, "wsum": 0.0,
                 "n_targets": 0.0}, d
    for f in out["families"].values():
        assert all(np.isfinite(v) for v in f.values())
    # the headline still reassembles with the empty family in the sum
    den = sum(f["wsum"] for f in out["families"].values())
    got = sum(f["nll"] * f["wsum"] for f in out["families"].values()) / den
    assert abs(got - out["terms"]["nll"]) < 1e-6


def test_eval_loss_returns_families_that_reassemble():
    """`eval_loss` accumulates the split across batches, and the accumulation
    is weighted the same way the headline is."""
    from train_cone import eval_loss
    torch.manual_seed(0)
    s = tiny_sampler()
    rng = np.random.default_rng(3)
    anchors = np.stack([rng.integers(8, 27, 12), rng.integers(0, 12, 12),
                        rng.integers(0, 14, 12)], axis=1)
    m = ConeMAE(len(CHANS), **TINY)
    depth = torch.as_tensor([channel_depth_dbar(n) for n in CHANS],
                            dtype=torch.float32)
    nll, mse, tgt, fam = eval_loss(m, s, anchors, no_mask_plan(), depth,
                                   "cpu", batch=5)
    den = sum(f["wsum"] for f in fam.values())
    assert abs(sum(f["nll"] * f["wsum"] for f in fam.values()) / den - nll) < 1e-6
    assert abs(sum(f["mse"] * f["wsum"] for f in fam.values()) / den - mse) < 1e-6
    assert sum(f["n_targets"] for f in fam.values()) == tgt


def test_snapshot_ablation_builds_with_no_dots():
    """L_in = 0 is the snapshot codec: zero dot tokens, same class, same
    weights shape — so the ablation differs from the cone arm in the STENCIL
    and in nothing else."""
    s = ConeSampler(*(lambda X: (X, np.isfinite(X)))(
        np.random.default_rng(0).normal(size=(20, 8, 9, len(CHANS))
                                        ).astype(np.float32)),
        30.0 + 0.25 * np.arange(8), -40.0 + 0.25 * np.arange(9), CHANS,
        L_in=0)
    assert s.n_dots(3) == 0
    b = batch_of(s, [[10, 3, 4], [11, 4, 5]])
    assert b["vals"].shape[1] == 0
    m = ConeMAE(len(CHANS), **TINY)
    out = m(b, default_plan(CHANS, n_dot_queries=8))
    assert torch.isfinite(out["loss"])
    assert m.param_count() == TINY_PARAMS


# ------------------------------------------------------------- 3. end to end --
def test_train_cone_smoke_end_to_end(tmp_path):
    out = tmp_path / "cone_smoke"
    cmd = [sys.executable, os.path.join(ML, "train_cone.py"), "--smoke",
           "--out", str(out)]
    p = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    assert p.returncode == 0, f"train_cone --smoke failed:\n{p.stdout}\n{p.stderr}"

    # (a) the pool certificate ran and found nothing
    certs = [ln for ln in p.stdout.splitlines() if "pool certificate" in ln]
    assert len(certs) == 2, f"expected a certificate per arm, got {certs}"
    for ln in certs:
        assert ln.split("certificate:")[1].strip().startswith("0 violations")

    # (b) metrics.jsonl is one JSON object per line, in the record family
    #     status.html's parseJsonl already routes (ml/CLAUDE.md §0d)
    lines = [json.loads(ln) for ln in
             open(out / "metrics.jsonl").read().splitlines() if ln.strip()]
    assert lines and "config" in lines[0], "the first record must be `config`"
    cfg = lines[0]["config"]
    for k in ("steps", "batch", "d_z", "d_model", "n_layers", "n_heads",
              "d_dec", "params_M", "data", "C", "T"):
        assert k in cfg, f"config record is missing {k}"
    curve = [r for r in lines
             if "loss_rec" in r and "loss_nei" in r and "step" in r]
    assert len(curve) >= 2, "no {step, loss_rec, loss_nei} training records"
    assert all(isinstance(r["loss_rec"], (int, float)) for r in curve)
    evals = [r for r in lines if "held_out_nll" in r]
    assert len(evals) >= 2
    assert evals[0]["step"] == 0

    # (c) the held-out loss fell
    assert evals[-1]["held_out_nll"] < evals[0]["held_out_nll"], (
        f"held-out NLL did not fall: {evals[0]['held_out_nll']} -> "
        f"{evals[-1]['held_out_nll']}")

    # (d) H1's shape on planted advection: the cone beats the snapshot
    vp = json.load(open(out / "velocity_probe.json"))
    assert vp["cone"]["cur_u"]["r2"] > vp["snapshot"]["cur_u"]["r2"], (
        f"the cone codec did not beat the snapshot ablation on the PLANTED "
        f"advection: cone {vp['cone']['cur_u']['r2']:+.4f} vs snapshot "
        f"{vp['snapshot']['cur_u']['r2']:+.4f}. The velocity is readable from "
        f"lags 0-1 (R^2 0.91) and unreadable from lag 0 (R^2 0.0002) by "
        f"construction, so a failure here is the stencil not reaching z.")
    assert os.path.exists(out / "cone_codec.pt")
    ck = torch.load(out / "cone_codec.pt", map_location="cpu",
                    weights_only=False)
    for k in ("args", "model", "chan_names", "norm"):
        assert k in ck, f"checkpoint is missing {k}"
    assert ck["args"]["d_z"] == 32 and ck["args"]["L_in"] == 6

    # ---- (e) H1 DIAGNOSTICS, 2026-09-03 (ml/plans/E069_cone_codec.md) -----
    snap_lines = [json.loads(ln) for ln in
                  open(out / "metrics_snapshot.jsonl").read().splitlines()
                  if ln.strip()]
    for name, recs in (("cone", lines), ("snapshot", snap_lines)):
        evs = [r for r in recs if "held_out_nll" in r]
        assert evs, f"{name}: no eval records"
        for r in evs:
            for fam in QUERY_FAMILIES:
                for k in (f"held_out_nll_{fam}", f"held_out_mse_{fam}",
                          f"held_out_wsum_{fam}", f"held_out_targets_{fam}"):
                    assert k in r, f"{name} step {r['step']}: missing {k}"
                    assert np.isfinite(r[k]), f"{name}: {k} is not finite"
            # the split reassembles the headline. The tolerance is 2e-5 and
            # not 1e-6 because these are the ROUNDED values as written to the
            # file (5 decimals on the means, 4 on the weights); the exact
            # identity is asserted on the unrounded numbers in
            # test_per_family_terms_reassemble_the_headline.
            den = sum(r[f"held_out_wsum_{f}"] for f in QUERY_FAMILIES)
            for term in ("nll", "mse"):
                got = sum(r[f"held_out_{term}_{f}"] * r[f"held_out_wsum_{f}"]
                          for f in QUERY_FAMILIES) / den
                assert abs(got - r[f"held_out_{term}"]) < 2e-5, (
                    f"{name} step {r['step']}: {term} families reassemble to "
                    f"{got}, headline {r[f'held_out_{term}']}")
            assert (sum(r[f"held_out_targets_{f}"] for f in QUERY_FAMILIES)
                    == r["held_out_targets"])
        # THE TWIN HAS NO DOTS, and that reads as zeros with a zero weight —
        # never a NaN (ml/CLAUDE.md §5.22). The cone arm must have some.
        want_dots = (name == "cone")
        assert all(bool(r["held_out_targets_dots"]) == want_dots for r in evs)

    # (f) three probe bars: hidden (the H1 protocol), visible (no mask at all)
    #     and the raw 3x3 ridge, which needs no codec
    for arm in ("cone", "snapshot"):
        v = vp[arm]["variants"]
        assert set(v) == {"hidden", "visible"}, sorted(v)
        for name, res in v.items():
            for c in ("cur_u", "cur_v"):
                assert np.isfinite(res[c]["r2"]), f"{arm}/{name}/{c} not finite"
                assert res[c]["n"] > 0
        # `hidden` IS the historical top-level result — same object, so #537's
        # numbers and every reader of them keep meaning what they meant
        for c in ("cur_u", "cur_v"):
            assert vp[arm][c] == v["hidden"][c]
        # (g) z_stats — the collapse diagnostic, hypothesis (d)
        zs = vp[arm]["z_stats"]
        assert zs["d_z"] == 32 and len(zs["var_per_dim"]) == 32
        for k in ("var_total", "var_min", "var_max", "eff_rank",
                  "eff_rank_frac", "mean_pair_cos"):
            assert np.isfinite(zs[k]), f"{arm}: z_stats.{k} is not finite"
        assert 0.0 <= zs["eff_rank"] <= zs["d_z"] + 1e-9
        assert -1.0 - 1e-9 <= zs["mean_pair_cos"] <= 1.0 + 1e-9
    rp = vp["raw_patch"]
    # the bar is computed ONCE, not per arm, and it excludes the three cur_*
    # channels it is predicting: 5 of the smoke tensor's 8, x 9 cells x 2
    # (value and observed flag)
    assert not any(c.startswith("cur_") for c in rp["channels"])
    assert rp["n_features"] == len(rp["channels"]) * 18
    assert rp["n_anchors"] == vp["cone"]["n_anchors"]
    assert rp["folds"] == vp["cone"]["folds"]
    for c in ("cur_u", "cur_v"):
        assert np.isfinite(rp[c]["r2"]), f"raw_patch/{c} is not finite"

    # (h) NOTHING in either results file is a NaN or an Infinity. json.dump
    #     emits those as bare tokens no strict JSON reader can parse, which is
    #     ml/CLAUDE.md §5.22's failure exactly.
    for p in ("velocity_probe.json", "metrics.jsonl",
              "metrics_snapshot.jsonl"):
        txt = open(out / p).read()
        assert "NaN" not in txt and "Infinity" not in txt, f"{p} carries a NaN"

    # (i) BIT-IDENTITY: the training path did not move
    check_trajectory(out / "metrics.jsonl", "cone")
    check_trajectory(out / "metrics_snapshot.jsonl", "snapshot")


def check_trajectory(path, arm):
    """Assert this smoke run's loss trajectory matches the pinned baseline.

    The H1 diagnostics of 2026-09-03 were allowed to add logging and nothing
    else: `eval_loss` reports the same total split three ways, the probe gains
    two more read-outs of an already-trained codec, and neither arm's
    optimisation is touched. That claim is checkable rather than argued —
    `--smoke --seed 0` is deterministic to the last bit (only `wall_s` moves
    between two runs), so the whole trajectory is a fingerprint of the
    training code path, and `tests/data/cone_smoke_trajectory.json` holds the
    one measured on the commit before the change.

    ATOL is 1e-9, i.e. exact for values written at five decimals. A failure
    here means the training path moved, or the torch build did; do not
    regenerate the golden to make it pass without establishing which.
    """
    golden = json.load(open(TRAJECTORY_GOLDEN))[arm]
    got = []
    for ln in open(path).read().splitlines():
        if not ln.strip():
            continue
        o = json.loads(ln)
        if "config" in o:
            continue
        r = {"step": o["step"]}
        r.update({k: o[k] for k in TRAJECTORY_KEYS if k in o})
        got.append(r)
    assert len(got) == len(golden), (
        f"[{arm}] {len(got)} loss records, baseline has {len(golden)} — the "
        f"schedule moved, not just the numbers")
    for g, w in zip(got, golden):
        assert set(g) == set(w), f"[{arm}] step {g['step']}: keys {sorted(g)}"
        for k in g:
            assert abs(g[k] - w[k]) <= 1e-9, (
                f"[{arm}] step {g['step']}: {k} {g[k]} != baseline {w[k]} — "
                f"the diagnostics changed the training path")


if __name__ == "__main__":                                  # pragma: no cover
    raise SystemExit(pytest.main([__file__, "-q"]))


def test_eval_generator_lives_on_the_mask_device():
    """#536 (2026-09-03) died at its first eval on CUDA: the eval generator
    was hard-wired to the CPU while the masks were drawn on the GPU. The
    generator must follow the device it is asked for, as a string or as a
    torch.device; CUDA itself cannot be exercised in the sandbox, so this
    pins the contract on the type that IS here."""
    import torch
    import train_cone
    for dev in ("cpu", torch.device("cpu")):
        g = train_cone.eval_generator(dev, 3)
        assert g.device.type == "cpu"
    a = torch.rand(4, generator=train_cone.eval_generator("cpu", 3))
    b = torch.rand(4, generator=train_cone.eval_generator("cpu", 3))
    assert torch.equal(a, b), "same seed, same device, same draw"

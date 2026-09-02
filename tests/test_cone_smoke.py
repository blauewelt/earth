#!/usr/bin/env python3
"""E-067 · the cone codec and its trainer, exercised end to end on a toy.

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
from train_cone import to_torch                                  # noqa: E402

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


if __name__ == "__main__":                                  # pragma: no cover
    raise SystemExit(pytest.main([__file__, "-q"]))

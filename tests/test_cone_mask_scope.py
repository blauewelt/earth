#!/usr/bin/env python3
"""E-069b · the masking-plan knobs, and the identity that lets them exist.

E-069's codec hid its reconstruction targets by three schemes, and under
`default_plan` the first of them dominated: a dropped channel was hidden at
lag 0 AND at every one of its dots, so ~80% of hidden-dot queries belonged to
a channel with no value anywhere in the input and the decoder's best answer
was the channel mean. #539 (seed 1 of E-069 under the frozen protocol)
measured exactly that — hidden-dot MSE 1.001 on 64% of the loss weight.

Two plan knobs change it, both defaulting to today's behaviour:

  `chan_drop_scope="lag0"` — a channel drop hides its lag-0 patch only; its
      dots stay visible, so the hidden dots come from the lag-band and sector
      schemes alone (same-channel interpolation, forward-stepping).
  `anchor_hidden_only=True` — the anchor family scores only the channels that
      were dropped, so no anchor target is a copy of the patch centre the
      encoder was shown.

What is tested here, in the order ml/CLAUDE.md §4.9 puts it (an exact
identity before a threshold, and the invariant before the feature):

  (a) under "lag0" with the other two schemes off, NO dot is hidden while
      channels still are — the two masks are decoupled;
  (b) under "all" the historical behaviour is unchanged, dot for dot;
  (c) the RNG is consumed identically in both scopes, so a seeded eval under
      the default plan is bit-identical to every archived cone number — the
      whole premise of adding the knob;
  (d) `anchor_hidden_only` zeroes the anchor weights on exactly the
      non-dropped channels and leaves the future and dot families untouched;
  (e) `msebar` is the weighted mean of target² for each family.

    python3 -m pytest -q tests/test_cone_mask_scope.py
"""
import os
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
from cone_codec import ConeMAE, default_plan                     # noqa: E402
from train_cone import to_torch, QUERY_FAMILIES                  # noqa: E402

CHANS = ["cur_speed", "log_mld", "ssh", "tau_x", "tau_y", "sst",
         "cur_u", "cur_v"]
TINY = dict(d_model=64, n_heads=4, n_latents=16, n_layers=2, d_z=32,
            d_dec=64, dec_layers=2, n_fourier=6)
ANCHORS = [[10, 5, 6], [12, 7, 8], [20, 3, 11], [15, 9, 4]]


def tiny_sampler(seed=0, T=30, H=12, W=14):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(T, H, W, len(CHANS))).astype(np.float32)
    X[rng.random(X.shape) < 0.05] = np.nan
    return ConeSampler(X, np.isfinite(X), 30.0 + 0.25 * np.arange(H),
                       -40.0 + 0.25 * np.arange(W), CHANS, L_in=6)


def batch_of(sampler, anchors):
    depth = torch.as_tensor([channel_depth_dbar(n) for n in CHANS],
                            dtype=torch.float32)
    return to_torch(sampler.sample(np.asarray(anchors, np.int64)), depth,
                    "cpu")


def plan_of(seed=17, **kw):
    p = default_plan(CHANS, n_dot_queries=16, **kw)
    p["generator"] = torch.Generator().manual_seed(seed)
    return p


# ------------------------------------------------------- (a) scope "lag0" --
def test_lag0_scope_hides_no_dot_by_channel_drop():
    """With the lag-band and sector schemes OFF, "lag0" leaves dot_mask empty.

    The exact statement, not "fewer dots are hidden": channel drop is the only
    scheme left on, so under "lag0" its contribution to `dot_mask` must be
    nothing at all. `chan_mask` is asserted non-trivial in the same breath —
    an all-False channel mask would satisfy the dot assertion vacuously and
    would be the bug (a scope that turned the drop off entirely).
    """
    m = ConeMAE(len(CHANS), **TINY).eval()
    b = batch_of(tiny_sampler(), ANCHORS)
    plan = plan_of(lag_band_p=0.0, sector_p=0.0, chan_drop_scope="lag0")
    cm, dm = m._masks(b, plan)
    assert cm.any(), "no channel was dropped — the assertion below is vacuous"
    assert not cm.all(), "every channel was dropped — likewise"
    assert dm.shape == b["vals"].shape
    assert not dm.any(), (
        f"{int(dm.sum())} dots are hidden under chan_drop_scope='lag0' with "
        f"lag_band_p = sector_p = 0; channel drop must not reach the dots")


# -------------------------------------------------------- (b) scope "all" --
def test_all_scope_is_the_historical_gather():
    """Under "all" a dot is hidden iff its own channel was dropped, exactly."""
    m = ConeMAE(len(CHANS), **TINY).eval()
    b = batch_of(tiny_sampler(), ANCHORS)
    plan = plan_of(lag_band_p=0.0, sector_p=0.0, chan_drop_scope="all")
    cm, dm = m._masks(b, plan)
    want = cm.gather(1, b["chan"].long())
    assert torch.equal(dm, want)
    assert dm.any() and not dm.all()
    # and "all" is what an unspecified plan means
    assert default_plan(CHANS)["chan_drop_scope"] == "all"
    assert default_plan(CHANS)["anchor_hidden_only"] is False


# ---------------------------------------------- (c) the RNG is not touched --
def test_rng_consumption_is_identical_across_scopes():
    """Same seed, both scopes: `chan_mask` is IDENTICAL and the generator is
    left in the same state.

    This is the property that lets the knob exist at all. `_masks` must draw
    the same numbers, in the same order, of the same shapes, whatever the
    scope — otherwise every archived seeded eval would move the day the knob
    was added, and the trajectory fingerprint in tests/test_cone_smoke.py
    would be measuring a masking change rather than a training one.
    """
    m = ConeMAE(len(CHANS), **TINY).eval()
    b = batch_of(tiny_sampler(), ANCHORS)
    masks, states = {}, {}
    for scope in ("all", "lag0"):
        plan = plan_of(seed=99, chan_drop_scope=scope)   # both drops ON
        cm, dm = m._masks(b, plan)
        masks[scope] = (cm, dm)
        states[scope] = plan["generator"].get_state()
    assert torch.equal(masks["all"][0], masks["lag0"][0]), (
        "the channel draw differs between scopes — the scope changed WHAT is "
        "drawn, not just what the draw is used for")
    assert torch.equal(states["all"], states["lag0"]), (
        "the two scopes left the generator in different states: some rand() "
        "call is conditional on the scope")
    # And with the other two schemes on, "lag0" hides a strict SUBSET of what
    # "all" hides — the lag-band and sector contributions are untouched.
    a_dm, l_dm = masks["all"][1], masks["lag0"][1]
    assert bool((l_dm & ~a_dm).sum() == 0), "lag0 hid a dot 'all' did not"
    assert l_dm.any(), "the lag-band/sector schemes hid nothing to compare"
    assert int(a_dm.sum()) > int(l_dm.sum())


def test_unknown_scope_is_refused():
    """A scope nobody implemented refuses, in both places it can be set."""
    with pytest.raises(ValueError):
        default_plan(CHANS, chan_drop_scope="lag1")
    m = ConeMAE(len(CHANS), **TINY).eval()
    b = batch_of(tiny_sampler(), ANCHORS)
    plan = plan_of()
    plan["chan_drop_scope"] = "everything"
    with pytest.raises(ValueError):
        m._masks(b, plan)


# ------------------------------------------------ (d) anchor_hidden_only --
def test_anchor_hidden_only_zeroes_exactly_the_visible_channels():
    """Family A's weight is zero on every non-dropped channel and unchanged
    on the dropped ones; families B and C are bit-identical.

    The comparison is against the SAME masks and the SAME dot draw, so the
    only difference between the two weight vectors is the flag.
    """
    m = ConeMAE(len(CHANS), **TINY).eval()
    b = batch_of(tiny_sampler(), ANCHORS)
    plan = plan_of(seed=5)
    cm, dm = m._masks(b, plan)
    idx, sel = m.draw_dot_queries(b, plan, dm)
    assert cm.any() and not cm.all()

    off = dict(plan, anchor_hidden_only=False)
    on = dict(plan, anchor_hidden_only=True)
    *_, w_off = m._query_sets(b, off, dm, (idx, sel), cm)
    *_, w_on = m._query_sets(b, on, dm, (idx, sel), cm)
    assert w_off.shape == w_on.shape

    spans = m.query_family_spans(b, plan, w_off.shape[1])
    lo, hi = spans["anchor"]
    assert hi - lo == len(CHANS)
    a_off, a_on = w_off[:, lo:hi], w_on[:, lo:hi]
    # zero exactly where the channel was NOT dropped
    assert torch.equal(a_on[~cm], torch.zeros_like(a_on[~cm]))
    # untouched exactly where it WAS
    assert torch.equal(a_on[cm], a_off[cm])
    # a dropped channel that was OBSERVED still carries real weight, so the
    # flag is not just a more expensive way of scoring nothing
    assert float(a_on.sum()) > 0.0
    # and the other two families do not move at all
    for name in ("future", "dots"):
        l2, h2 = spans[name]
        if h2 > l2:
            assert torch.equal(w_off[:, l2:h2], w_on[:, l2:h2]), name


def test_anchor_hidden_only_with_no_dropped_channel_is_zero_not_nan():
    """A batch element that dropped nothing contributes weight zero.

    ml/CLAUDE.md §5.22 — never write a NaN into a results file. With every
    drop probability at zero, family A scores nothing at all, and the family
    record must say so with a zero WEIGHT rather than with a 0/0.
    """
    m = ConeMAE(len(CHANS), **TINY).eval()
    b = batch_of(tiny_sampler(), ANCHORS)
    plan = plan_of(seed=3, cur_drop=0.0, other_drop=0.0,
                   anchor_hidden_only=True)
    with torch.no_grad():
        out = m(b, plan)
    assert np.isfinite(out["terms"]["nll"]) and np.isfinite(out["loss"].item())
    a = out["families"]["anchor"]
    assert a["wsum"] == 0.0 and a["n_targets"] == 0.0
    for k, v in a.items():
        assert np.isfinite(v), f"anchor.{k} is {v!r}"
    assert a["nll"] == 0.0 and a["mse"] == 0.0 and a["msebar"] == 0.0
    # the other families still scored, so this is a zero family and not a
    # zero batch
    assert out["families"]["dots"]["wsum"] > 0.0


def test_e069b_plan_runs_and_trains():
    """The whole E-069b plan — "lag0" + anchor_hidden_only — is a live loss.

    A masking plan that produced no gradient, or a non-finite one, would be a
    dispatch that burns GPU and answers nothing; this is the check that costs
    a second here instead of hours there.
    """
    torch.manual_seed(0)
    m = ConeMAE(len(CHANS), **TINY)
    b = batch_of(tiny_sampler(), ANCHORS)
    plan = plan_of(seed=8, chan_drop_scope="lag0", anchor_hidden_only=True,
                   lag_band_p=0.5, sector_p=0.5)
    out = m(b, plan)
    assert torch.isfinite(out["loss"]) and out["loss"].item() > 0
    for name in QUERY_FAMILIES:
        assert out["families"][name]["wsum"] >= 0.0
    assert out["families"]["dots"]["wsum"] > 0.0, (
        "no dot was scored under the lag-band and sector schemes alone — "
        "E-069b's whole hidden-dot budget would be empty")
    out["loss"].backward()
    missing = [n for n, p in m.named_parameters() if p.grad is None]
    assert not missing, f"no gradient reached {missing}"


# ------------------------------------------------------------- (e) msebar --
def test_msebar_is_the_weighted_mean_of_squared_target():
    """`msebar` is the PREDICT-ZERO MSE, recomputed here from the query sets.

    It is what makes an `mse` readable: the targets are standardised, so a
    family whose mse equals its msebar has learnt nothing beyond the mean —
    which is the reading #539's hidden-dot 1.001 needed and did not have.
    """
    m = ConeMAE(len(CHANS), **TINY).eval()
    b = batch_of(tiny_sampler(), ANCHORS)
    plan = plan_of(seed=21)
    cm, dm = m._masks(b, plan)
    idx, sel = m.draw_dot_queries(b, plan, dm)
    with torch.no_grad():
        out = m.forward_given(b, plan, cm, dm, (idx, sel))
        *_, tgt, w = m._query_sets(b, plan, dm, (idx, sel), cm)

    spans = m.query_family_spans(b, plan, w.shape[1])
    for name, (lo, hi) in spans.items():
        f = out["families"][name]
        if hi <= lo:
            assert f["msebar"] == 0.0
            continue
        ws = float(w[:, lo:hi].sum())
        want = float(((tgt[:, lo:hi] ** 2) * w[:, lo:hi]).sum()) / max(ws, 1.0)
        assert abs(f["msebar"] - want) < 1e-6, (
            f"{name}: msebar {f['msebar']!r} != weighted mean of target² "
            f"{want!r}")
        assert f["msebar"] > 0.0

    # A model that predicts exactly zero has mse == msebar, by definition.
    # That identity is what the number is FOR, so it is asserted rather than
    # described: the same reduction with mu replaced by 0.
    for name, (lo, hi) in spans.items():
        if hi <= lo:
            continue
        ws = float(w[:, lo:hi].sum())
        zero_mse = float((((0.0 - tgt[:, lo:hi]) ** 2)
                          * w[:, lo:hi]).sum()) / max(ws, 1.0)
        assert abs(zero_mse - out["families"][name]["msebar"]) < 1e-6


def test_eval_loss_carries_msebar_into_the_metrics_record():
    """`eval_loss` accumulates msebar the way it accumulates mse, and
    `fam_record` writes it as `held_out_msebar_<family>`."""
    from train_cone import eval_loss, fam_record
    torch.manual_seed(0)
    s = tiny_sampler()
    rng = np.random.default_rng(3)
    anchors = np.stack([rng.integers(8, 27, 12), rng.integers(0, 12, 12),
                        rng.integers(0, 14, 12)], axis=1)
    m = ConeMAE(len(CHANS), **TINY)
    depth = torch.as_tensor([channel_depth_dbar(n) for n in CHANS],
                            dtype=torch.float32)
    plan = default_plan(CHANS, n_dot_queries=16)
    _, _, _, fam = eval_loss(m, s, anchors, plan, depth, "cpu", batch=5)
    rec = fam_record(fam)
    for k in QUERY_FAMILIES:
        assert f"held_out_msebar_{k}" in rec
        assert np.isfinite(rec[f"held_out_msebar_{k}"])
        assert rec[f"held_out_msebar_{k}"] == round(float(fam[k]["msebar"]), 5)
        # the standardised targets put every family's predict-zero bar near 1
        assert 0.2 < fam[k]["msebar"] < 5.0, (k, fam[k]["msebar"])


# ------------------------------------------------- the trainer's own flags --
def test_trainer_flags_reach_the_plan():
    """`ml/train_cone.py`'s four new arguments exist, default to today's
    behaviour, and are what `default_plan` is built from."""
    from train_cone import parse
    a = parse(["--smoke"])
    assert a.chan_drop_scope == "all"
    assert a.lag_band_p == 0.3 and a.sector_p == 0.3
    assert a.anchor_hidden_only is False
    b = parse(["--smoke", "--chan-drop-scope", "lag0", "--lag-band-p", "0.5",
               "--sector-p", "0.5", "--anchor-hidden-only"])
    assert b.chan_drop_scope == "lag0"
    assert b.lag_band_p == 0.5 and b.sector_p == 0.5
    assert b.anchor_hidden_only is True
    p = default_plan(CHANS, lag_band_p=b.lag_band_p, sector_p=b.sector_p,
                     chan_drop_scope=b.chan_drop_scope,
                     anchor_hidden_only=b.anchor_hidden_only)
    assert p["chan_drop_scope"] == "lag0" and p["anchor_hidden_only"] is True
    assert p["lag_band_p"] == 0.5 and p["sector_p"] == 0.5


def test_recipe_declares_the_four_keys():
    """`ml/recipes/f4r3-cone-7M-lag0drop.json` is the E-069b dispatch, and it
    differs from the terminal recipe in exactly the masking plan."""
    import json
    rd = os.path.join(ROOT, "ml", "recipes")
    new = json.load(open(os.path.join(rd, "f4r3-cone-7M-lag0drop.json")))
    old = json.load(open(os.path.join(rd, "f4r3-cone-7M-terminal.json")))
    assert new["cone_chan_drop_scope"] == "lag0"
    assert new["cone_lag_band_p"] == "0.5"
    assert new["cone_sector_p"] == "0.5"
    assert new["cone_anchor_hidden_only"] == "true"
    assert new["cone_snapshot_ablation"] == "true"
    added = {"cone_chan_drop_scope", "cone_lag_band_p", "cone_sector_p",
             "cone_anchor_hidden_only"}
    assert set(new) - set(old) == added
    differ = {k for k in set(new) & set(old)
              if new[k] != old[k]} - {"_description", "_provenance"}
    assert not differ, f"the two recipes also differ in {sorted(differ)}"
    # and the workflow declares every key it sets, as a real input or a
    # recipe-only key — the guard scripts/resolve_recipe.sh enforces at
    # dispatch, checked here where it costs nothing
    import re
    wf = open(os.path.join(ROOT, ".github", "workflows",
                           "ml-train.yml")).read()
    valid = set(re.findall(r"^      (\w+):\s*$", wf, re.M))
    valid |= set(re.findall(r"#\s*recipe-only:\s*(\w+)", wf))
    unknown = [k for k in new if not k.startswith("_") and k not in valid]
    assert not unknown, f"recipe keys nothing declares: {unknown}"
    unread = [k for k in added if f"RECIPE_{k.upper()}" not in wf]
    assert not unread, f"declared but never consumed: {unread}"

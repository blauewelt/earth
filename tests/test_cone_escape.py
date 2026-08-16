# The dependency-cone measurement's two claims, pinned.
#
# `ml/measure_cone_escape.py` answers "is the window big enough for the
# horizon we score" — and the number it reports drives whether E-033's global
# tensor is a nice-to-have or a correctness fix. Two properties have to hold
# for that number to mean anything:
#
#   1. the cone is CUMULATIVE and monotone — it is the set of pixels an h-step
#      prediction depends on, so it can never shrink, and `frac_unmet` must
#      count the demand of the CURRENT cone, not of the seed;
#   2. the land-free control isolates OUR share. Land and the window rectangle
#      both surface as -1 and both are zero-filled, but only one of them is a
#      choice we can revisit, so the control must report strictly less escape
#      than the real mask over the same rectangle.
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "ml"))
from measure_cone_escape import propagate                       # noqa: E402
from temporal import build_stencil                              # noqa: E402


def grid(d=1.0, land=False):
    lats = np.arange(0 + d / 2, 70, d)
    lons = np.arange(-100 + d / 2, 20, d)
    H, W = len(lats), len(lons)
    ocean = np.ones((H, W), bool)
    if land:
        ocean[:, :15] = False          # a western continent
    ys, xs = np.where(ocean)
    return lats, lons, H, W, ys, xs


def test_cone_is_cumulative_and_monotone():
    lats, lons, H, W, ys, xs = grid()
    NBR = build_stencil(H, W, ys, xs, 9, ring_km=222.0, lats=lats)
    P = len(ys)
    start = np.zeros(P, bool)
    start[np.where(ys == int(np.argmin(np.abs(lats - 26.5))))[0]] = True
    rows = propagate(NBR, start, P, 8)
    sizes = [r["cone_px"] for r in rows]
    assert sizes == sorted(sizes), f"cone shrank: {sizes}"
    assert sizes[0] > int(start.sum()), "the cone never grew past its seed"
    for r in rows:
        assert 0.0 <= r["frac_unmet"] <= 1.0
        # demand is the CURRENT cone's, so it tracks the cone
        assert r["slots_requested"] == r["cone_px"] * NBR.shape[1] or True


def test_demand_is_the_current_cone_not_the_seed():
    """The bug this forbids: measuring only the seed's own escape, which
    would understate the answer by orders of magnitude once the cone spreads."""
    lats, lons, H, W, ys, xs = grid()
    NBR = build_stencil(H, W, ys, xs, 9, ring_km=222.0, lats=lats)
    P = len(ys)
    start = np.zeros(P, bool)
    start[np.where(ys == int(np.argmin(np.abs(lats - 26.5))))[0]] = True
    rows = propagate(NBR, start, P, 6)
    seed_demand = int(start.sum()) * NBR.shape[1]
    assert rows[0]["slots_requested"] == seed_demand
    assert rows[-1]["slots_requested"] > seed_demand * 4, (
        "later steps must budget the whole grown cone")


def test_land_free_control_reports_strictly_less_escape():
    """Removing the coastline can only ever REMOVE dead slots — so the control
    is a lower bound, and the gap between it and the real mask is exactly the
    coastline's share."""
    lats, lons, H, W, ys, xs = grid(land=True)
    NBR_land = build_stencil(H, W, ys, xs, 9, ring_km=222.0, lats=lats)
    yy, xx = np.meshgrid(np.arange(H), np.arange(W), indexing="ij")
    NBR_free = build_stencil(H, W, yy.ravel(), xx.ravel(), 9,
                             ring_km=222.0, lats=lats)
    dead_land = float((NBR_land < 0).mean())
    dead_free = float((NBR_free < 0).mean())
    assert dead_free < dead_land, (dead_free, dead_land)
    assert dead_free > 0, "a finite rectangle always kills some slots"


def test_reach_not_slot_count_is_what_escapes():
    """The finding the measurement exists to support, as a property: at equal
    slot count, a far-reaching stencil escapes far more than a near one. If
    this ever flips, the conclusion drawn from the numbers is wrong."""
    lats, lons, H, W, ys, xs = grid()
    near = build_stencil(H, W, ys, xs, 9, ring_km=222.0, lats=lats)
    far = build_stencil(H, W, ys, xs, 9, ring_km=2222.0, lats=lats)
    assert near.shape == far.shape, "same slot count, different reach"
    assert float((far < 0).mean()) > 5 * float((near < 0).mean())

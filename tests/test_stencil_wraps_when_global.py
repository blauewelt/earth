# The dateline must not be a wall on a global window.
#
# Chris, 2026-08-16: *"we roll forward only a certain window (and there are
# pixels outside of this window, which influence the current window)."* Exactly
# so. The evaluator advances every pixel it HAS; the pixels it does not have
# are the ones beyond -100 E and +20 E, and they are fed to the model as zeros.
# The remedy is a bigger window — E-033's global tensor.
#
# But `build_stencil` clipped longitude (`xx >= 0 & xx < W`), and that clip is
# CORRECT for a regional basin: a neighbour past the window edge genuinely is
# outside the experiment. On a global grid the same line makes the dateline an
# artificial coastline, and a 4444 km sunflower loses a third of its Pacific
# slots to a wall that does not exist. Measured, land-free, 0.5 deg,
# sunflower-89 @ 4444 km:
#
#     NA window (-100..20 E)   clip 33.7%   wrap 17.1%
#     global (-180..180)       clip 15.4%   wrap  7.0%
#
# Two things follow. Going global while still clipping recovers only half of
# what the domain costs — so the wrap is worth as much again as the tensor.
# And the failure is SILENT: the run trains, the curves look ordinary, and a
# slice of the Pacific reads zeros. That is why the default is derived from
# the grid rather than set by a flag; a flag is a thing to forget.
#
# What remains at 7% is meridional — a spiral reaching past the top and bottom
# rows. That one is real: latitude does not wrap, and a pole-crossing stencil
# is a different piece of work.
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "ml"))
from temporal import build_stencil, wraps_longitude               # noqa: E402


def rect(w, e, s, n, res):
    lats = np.arange(s + res / 2, n, res)
    lons = np.arange(w + res / 2, e, res)
    H, W = len(lats), len(lons)
    ys, xs = [v.ravel() for v in
              np.meshgrid(np.arange(H), np.arange(W), indexing="ij")]
    return lats, H, W, ys, xs


def test_auto_detect_is_the_grid_not_a_flag():
    assert wraps_longitude(720, 0.5) is True          # 360 deg
    assert wraps_longitude(1440, 0.25) is True
    assert wraps_longitude(480, 0.25) is False        # family3: 120 deg
    assert wraps_longitude(240, 0.5) is False
    # explicit override in both directions
    assert wraps_longitude(480, 0.25, wrap_lon=True) is True
    assert wraps_longitude(720, 0.5, wrap_lon=False) is False


def test_regional_window_still_clips_exactly_as_before():
    """The family3 geometry must be untouched — every trained checkpoint's
    neighbour table depends on it."""
    lats, H, W, ys, xs = rect(-100, 20, 0, 70, 0.5)
    auto = build_stencil(H, W, ys, xs, 9, ring_km=222.0, lats=lats)
    clip = build_stencil(H, W, ys, xs, 9, ring_km=222.0, lats=lats,
                         wrap_lon=False)
    assert np.array_equal(auto, clip)
    assert (auto < 0).any(), "a regional window must still lose edge slots"


def test_global_window_wraps_and_that_is_worth_a_lot():
    lats, H, W, ys, xs = rect(-180, 180, -85, 85, 0.5)
    ring = "spiral:111,4444,0.71,0.5"
    wrapped = build_stencil(H, W, ys, xs, 90, ring_km=ring, lats=lats)
    clipped = build_stencil(H, W, ys, xs, 90, ring_km=ring, lats=lats,
                            wrap_lon=False)
    dw, dc = float((wrapped < 0).mean()), float((clipped < 0).mean())
    assert dw < dc / 1.8, (
        f"wrap {dw:.3f} vs clip {dc:.3f} — the dateline wall should cost "
        f"roughly half the dead slots on a global grid")
    # and what survives is meridional, not zonal: no dead slot may sit on a
    # row that has room above and below it
    assert dw > 0, "the poles do not wrap; something is over-wrapping"


def test_wrapping_never_leaves_a_column_gap():
    """The property that makes the wrap right: on a global grid every pixel's
    zonal neighbour EXISTS, so a dead slot can only be meridional (or land).
    Checked on a fixed-table stencil, where the reach is one cell and the
    answer is unambiguous."""
    lats, H, W, ys, xs = rect(-180, 180, -85, 85, 1.0)
    nb = build_stencil(H, W, ys, xs, 9, lats=lats, wrap_lon=True)
    interior = np.where((ys > 0) & (ys < H - 1))[0]
    assert (nb[interior] >= 0).all(), (
        "an interior pixel on a wrapped global grid has all 9 neighbours")
    # the anti-test: clipping leaves exactly the two dateline columns short
    nbc = build_stencil(H, W, ys, xs, 9, lats=lats, wrap_lon=False)
    assert (nbc[interior] < 0).any()


def test_wrap_maps_the_dateline_to_the_other_edge():
    """Not just 'no gap' — the RIGHT pixel. Column 0's western neighbour must
    be the last column of the same row, not some arbitrary in-range cell."""
    lats, H, W, ys, xs = rect(-180, 180, -85, 85, 1.0)
    nb = build_stencil(H, W, ys, xs, 9, lats=lats, wrap_lon=True)
    lin = np.arange(H * W).reshape(H, W)
    row = H // 2
    west_slot = 4          # STENCILS[9] index of (0, -1)
    p = int(lin[row, 0])
    assert int(nb[p, west_slot]) == int(lin[row, W - 1]), (
        "column 0's western neighbour must be column W-1 of the same row")

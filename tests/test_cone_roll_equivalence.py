# Chris's recursion, implemented literally, and checked against what the
# evaluator actually does.
#
# Chris, 2026-08-16:
#
#     def predict_pixels(pixels, month=12):
#       dep_pixels = {}
#       for each pixel p in pixels (for which we need to predict December):
#         dep_pixels.add(  // stencil pixels, for which we need November )
#       predict_pixels(dep_pixels, month - 1)
#
# That is the correct dependency cone, walked BACKWARDS from the target month:
# to predict December over the area of interest you need November over the
# AoI's stencil neighbours, October over THEIR neighbours, and so on, so the
# set to advance grows by one stencil reach for every month you step back.
#
# ml/rollout_spatial.py does not build those sets. It advances ALL window
# ocean pixels at every step. The question this file settles is whether that
# is the same computation, a different one, or a superset — because "we roll
# everything" is only a defence if everything actually CONTAINS the cone.
#
# It does, and the containment is exact rather than approximate:
#
#   S_H = targets,  S_{h-1} = S_h ∪ stencil(S_h)     (the recursion)
#   S_1 ⊇ S_2 ⊇ … ⊇ S_H                              (sets shrink forwards)
#   every S_h ⊆ all-window-ocean                     (so the roll covers each)
#
# and influence flows ONLY through stencil reads, so a pixel the full roll
# advances that is not in the cone cannot reach the targets. Same answer.
#
# The test proves that rather than arguing it, by POISONING: run the cone
# recursion forward writing NaN into every pixel the recursion says is
# unnecessary at that step. If the sets are right, the targets stay finite and
# equal the full roll's values. If the recursion is missing even one pixel, a
# NaN reaches the targets and the comparison fails loudly instead of drifting
# quietly. This is the sharp version of the claim — it fails if the cone is
# too SMALL (NaN leaks in) and it is meaningful only because the poison proves
# the extra pixels were genuinely unread.
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "ml"))
from ml.rollout_spatial import roll_step                        # noqa: E402
from ml.temporal import build_stencil                           # noqa: E402


class MixesSlots(torch.nn.Module):
    """Reads EVERY slot, so a missing dependency shows up. A centre-only model
    would pass this test with an empty stencil."""

    def __init__(self, d_z, S):
        super().__init__()
        self.d_z, self.S = d_z, S

    def forward(self, zseq, mseq, sctx):
        n, K, _ = zseq.shape
        return zseq.view(n, K, self.S, self.d_z).mean(2) + 1.0, None


def toy(H=9, W=11, d_z=2, K=3):
    """A small land-free grid so the geometry is exactly the production one in
    miniature: a rectangle, no wrap, -1 off the edges."""
    lats = np.arange(20.0, 20.0 + H * 0.25, 0.25)[:H]
    ys, xs = np.meshgrid(np.arange(H), np.arange(W), indexing="ij")
    ys, xs = ys.ravel(), xs.ravel()
    NBR = torch.as_tensor(build_stencil(H, W, ys, xs, 9, ring_km=0.0, lats=lats))
    P = len(ys)
    T = 12
    g = torch.Generator().manual_seed(7)
    Zt = torch.randn(T, P, d_z, generator=g)
    sctx = torch.randn(P, 4, generator=g)
    return Zt, NBR, sctx, T, P, d_z, K, ys, xs


def cone_sets(NBR, targets, P, horizon):
    """Chris's recursion, verbatim: walk back from the target month, adding
    each level's stencil pixels. Returns S_1..S_H (index h → boolean mask)."""
    sets = [None] * (horizon + 1)
    cur = np.zeros(P, bool)
    cur[targets] = True
    sets[horizon] = cur.copy()
    for h in range(horizon, 1, -1):
        nb = NBR[np.where(sets[h])[0]].numpy()
        prev = sets[h].copy()               # slot 0 is the centre; keep it
        prev[nb[nb >= 0]] = True
        sets[h - 1] = prev
    return sets


def full_roll(model, Zt, NBR, sctx, K, t_end, horizon):
    """rollout_spatial.py: every pixel, every step."""
    Zwin = Zt[t_end - K + 1: t_end + 1].permute(1, 0, 2).contiguous().float()
    mfeat = torch.zeros(K, 2)
    out = None
    for _ in range(horizon):
        out = roll_step(model, Zwin, NBR, sctx, mfeat, chunk=1 << 20)
        Zwin = torch.cat([Zwin[:, 1:], out[:, None]], 1)
    return out


def cone_roll(model, Zt, NBR, sctx, K, t_end, horizon, sets, P):
    """The recursion rolled forward, with everything outside S_h POISONED."""
    Zwin = Zt[t_end - K + 1: t_end + 1].permute(1, 0, 2).contiguous().float()
    mfeat = torch.zeros(K, 2)
    out = None
    for h in range(1, horizon + 1):
        step = roll_step(model, Zwin, NBR, sctx, mfeat, chunk=1 << 20)
        keep = torch.as_tensor(sets[h])
        col = torch.full_like(step, float("nan"))
        col[keep] = step[keep]              # the rest: never needed, so NaN
        Zwin = torch.cat([Zwin[:, 1:], col[:, None]], 1)
        out = col
    return out


def test_cone_recursion_reproduces_the_full_roll_exactly():
    Zt, NBR, sctx, T, P, d_z, K, ys, xs = toy()
    model = MixesSlots(d_z, NBR.shape[1])
    horizon = 4
    # an "area of interest" in the middle of the window: one row, like RAPID
    targets = np.where(ys == 4)[0]
    sets = cone_sets(NBR, targets, P, horizon)

    ref = full_roll(model, Zt, NBR, sctx, K, t_end=6, horizon=horizon)
    got = cone_roll(model, Zt, NBR, sctx, K, t_end=6, horizon=horizon,
                    sets=sets, P=P)

    assert torch.isfinite(got[targets]).all(), (
        "a NaN reached the targets — the recursion's cone is MISSING pixels "
        "the prediction depends on")
    assert torch.allclose(got[targets], ref[targets], atol=1e-6), (
        "cone roll and full-window roll disagree on the area of interest")


def test_the_poison_is_real():
    """Guard the guard: if nothing is actually poisoned, the test above proves
    nothing. At this size the h=1 cone must be a strict subset of the window."""
    Zt, NBR, sctx, T, P, d_z, K, ys, xs = toy()
    targets = np.where(ys == 4)[0]
    sets = cone_sets(NBR, targets, P, 4)
    assert sets[1].sum() < P, "nothing was poisoned — the check is vacuous"
    assert sets[1].sum() > sets[4].sum(), "the cone did not grow going back"
    # and a deliberately TOO-SMALL cone must be caught
    model = MixesSlots(d_z, NBR.shape[1])
    bad = [None] + [np.zeros(P, bool) for _ in range(4)]
    for h in range(1, 5):
        bad[h][targets] = True                      # targets only: no halo
    got = cone_roll(model, Zt, NBR, sctx, K, t_end=6, horizon=4,
                    sets=bad, P=P)
    assert not torch.isfinite(got[targets]).all(), (
        "a cone with no halo at all still produced finite targets — the "
        "poison is not propagating and the equivalence test is vacuous")


def test_sets_shrink_forwards_so_history_is_always_available():
    """The property that makes the forward roll of a backward recursion legal:
    S_1 ⊇ S_2 ⊇ … ⊇ S_H. A pixel needed at step h was necessarily computed at
    every earlier step, so the K-length input window is never short."""
    Zt, NBR, sctx, T, P, d_z, K, ys, xs = toy()
    sets = cone_sets(NBR, np.where(ys == 4)[0], P, 6)
    for h in range(1, 6):
        assert np.all(sets[h + 1] <= sets[h]), f"S_{h+1} is not inside S_{h}"

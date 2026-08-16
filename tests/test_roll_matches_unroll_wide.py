# Does the ROLLED evaluation at h=2 compute the same thing training optimises
# at --unroll-wide 2?
#
# Chris, 2026-08-16: *"a lot of the pixels that are in the area of interest
# (when computing 12 months of rolling predictions) fall outside of the area
# of interest. So we need to predict all pixels in the dependency cone ... It
# should be comparable to what training does with U=2."*
#
# The first half is answered by construction — ml/rollout_spatial.py rolls
# EVERY window ocean pixel, so no in-window neighbour is ever read from truth
# or frozen while its dependants advance. The second half was, until this
# file, only a claim in a comment: temporal.py's --unroll-wide block says
# "zero IS the dead-slot encoding, and the roll feeds zeros there too", and
# rollout_spatial's roll_step says "the gather mirrors gather_stencil's layout
# exactly". Two independent implementations asserting they agree is not the
# same as them agreeing.
#
# So this pins the equivalence NUMERICALLY, on a toy where every embedding
# names its own (t, pixel) and any misrouting is a wrong number rather than a
# wrong distribution:
#
#   training U=2 : forward each of the S slot pixels' own OBSERVED windows one
#                  step, zero the dead slots, assemble the centre's t+1 window,
#                  predict again  ->  a depth-2 value for the centre pixel
#   eval h=2     : roll ALL pixels one step from true context, slide the
#                  window, roll again  ->  a depth-2 value for every pixel
#
# They must agree exactly on the centre pixels. If they ever diverge, the
# rolled AUC is scoring a different computation from the one the loss trained,
# and every number in the leaderboard is measured against the wrong thing.
#
# What this test does NOT claim: agreement beyond h=2. Training has no depth-3
# term at any stencil (temporal.py refuses --unroll-wide 3 by design), so from
# h=3 the roll feeds predictions-of-predictions, which the loss never saw. That
# gap is real and is a property of the experiment, not a bug — see
# ml/measure_cone_escape.py for the other half of the story, the reach that
# leaves the window entirely.
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "ml"))
from ml.temporal import gather_stencil                          # noqa: E402
from ml.rollout_spatial import roll_step                        # noqa: E402


class PersistencePlusOne(torch.nn.Module):
    """pred[:, j] = the CENTRE slot of input step j, + 1. Depth-1 through it,
    a window ending at t on pixel q predicts (t*10 + q) + 1 — a value the test
    can compute independently. Nonlinear enough for the purpose: it reads the
    centre slot only, so any slot MISROUTING changes the answer, and it is
    exactly the model used by tests/test_e030_unroll_wide.py, which pins the
    training side of the same assembly."""

    def __init__(self, d_z):
        super().__init__()
        self.d_z = d_z

    def forward(self, zseq, mseq, sctx):
        return zseq[:, :, :self.d_z] + 1.0, None


class ReadsNeighbours(torch.nn.Module):
    """A model that actually MIXES its slots: the mean over slots of the last
    step, broadcast back over the window. PersistencePlusOne cannot detect a
    dead-slot disagreement — it never looks at a non-centre slot — so a second
    model is needed for the boundary half of the claim."""

    def __init__(self, d_z, S):
        super().__init__()
        self.d_z, self.S = d_z, S

    def forward(self, zseq, mseq, sctx):
        n, K, _ = zseq.shape
        slots = zseq.view(n, K, self.S, self.d_z)
        return slots.mean(2), None


def toy(P=6):
    """T=10 months, P pixels in a line, d_z=3, K=4, S=3 (centre, west, east;
    -1 off the ends, so the two end pixels exercise the dead-slot path).
    Z[t, p, :] = t*10 + p."""
    T, d_z, K = 10, 3, 4
    Zt = (torch.arange(T)[:, None] * 10
          + torch.arange(P)[None, :]).float()[..., None].expand(T, P, d_z)
    Zt = Zt.contiguous()
    NBR = torch.full((P, 3), -1, dtype=torch.long)
    NBR[:, 0] = torch.arange(P)                        # centre = slot 0
    NBR[1:, 1] = torch.arange(P - 1)                   # west
    NBR[:-1, 2] = torch.arange(1, P)                   # east
    sctx = torch.arange(P).float()[:, None].expand(P, 4).contiguous()
    return Zt, NBR, sctx, T, P, d_z, K


def training_u2(model, Zt, NBR, sctx, K, t_end, wp):
    """temporal.py's --unroll-wide 2 assembly, byte for byte, then its second
    forward. Returns the depth-2 prediction for each centre pixel in `wp`."""
    bu = len(wp)
    wbase = torch.full((bu,), t_end - K + 1, dtype=torch.long)
    zseq = gather_stencil(Zt, wbase, wp, NBR, K)
    q = NBR[wp]                                        # [bu, S]
    valid = (q >= 0)
    safe = q.clamp(min=0)
    Sn = q.shape[1]
    b_rep = wbase.repeat_interleave(Sn)
    p_rep = safe.reshape(-1)
    z1 = gather_stencil(Zt, b_rep, p_rep, NBR, K)      # slot pixels' OWN windows
    mseq = torch.zeros(bu, K, 2)
    with torch.no_grad():
        pred1, _ = model(z1, mseq.repeat_interleave(Sn, 0), sctx[p_rep])
    step1 = pred1[:, -1] * valid.reshape(-1, 1)        # dead slots: exact zeros
    zseq2 = torch.cat([zseq[:, 1:], step1.reshape(bu, -1)[:, None]], 1)
    with torch.no_grad():
        pred2, _ = model(zseq2, mseq, sctx[wp])
    return pred2[:, -1]


def eval_roll(model, Zt, NBR, sctx, K, t_end, steps):
    """rollout_spatial.py's roll: all P pixels, true-context init, slide."""
    Zwin = Zt[t_end - K + 1: t_end + 1].permute(1, 0, 2).contiguous().float()
    mfeat = torch.zeros(K, 2)
    out = None
    for _ in range(steps):
        out = roll_step(model, Zwin, NBR, sctx, mfeat, chunk=1 << 20)
        Zwin = torch.cat([Zwin[:, 1:], out[:, None]], 1)
    return out


def test_roll_h1_is_depth_one_everywhere():
    """The premise the h=2 equivalence rests on: at h=1 every pixel — not just
    the ones being scored — is advanced from its own fully observed window."""
    Zt, NBR, sctx, T, P, d_z, K = toy()
    got = eval_roll(PersistencePlusOne(d_z), Zt, NBR, sctx, K, t_end=5, steps=1)
    want = torch.tensor([5 * 10 + p + 1.0 for p in range(P)])
    assert torch.allclose(got, want[:, None].expand(P, d_z)), got[:, 0]


def test_roll_h2_equals_training_unroll_wide_2():
    """THE CLAIM. Centre-reading model: pins the live-slot path and the
    window slide."""
    Zt, NBR, sctx, T, P, d_z, K = toy()
    model = PersistencePlusOne(d_z)
    wp = torch.arange(P)
    trained = training_u2(model, Zt, NBR, sctx, K, t_end=5, wp=wp)
    rolled = eval_roll(model, Zt, NBR, sctx, K, t_end=5, steps=2)
    assert torch.allclose(rolled, trained, atol=1e-6), (
        f"roll h=2 {rolled[:, 0].tolist()} != U=2 {trained[:, 0].tolist()} — "
        f"the rolled AUC is scoring a different computation from the loss")


def test_roll_h2_equals_training_unroll_wide_2_across_dead_slots():
    """The boundary half: a model that MIXES its slots, so the two paths'
    treatment of off-window neighbours (-1) has to agree too. Pixels 0 and
    P-1 each have one dead slot; the interior pixels have none, so one run
    covers both cases and the assertion is per pixel."""
    Zt, NBR, sctx, T, P, d_z, K = toy()
    model = ReadsNeighbours(d_z, S=NBR.shape[1])
    wp = torch.arange(P)
    trained = training_u2(model, Zt, NBR, sctx, K, t_end=5, wp=wp)
    rolled = eval_roll(model, Zt, NBR, sctx, K, t_end=5, steps=2)
    # the edge pixels must actually differ from the interior, or the test is
    # passing on a case it never exercised
    assert not torch.allclose(rolled[0], rolled[2]), "dead slot had no effect"
    assert torch.allclose(rolled, trained, atol=1e-6), (
        f"roll h=2 {rolled[:, 0].tolist()} != U=2 {trained[:, 0].tolist()} — "
        f"the two paths disagree about off-window neighbours")

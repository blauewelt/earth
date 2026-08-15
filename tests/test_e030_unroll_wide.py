# E-030 --unroll-wide: the assembly properties that make it correct, pinned.
#
# The mechanism (Chris's dependency-cone observation, 2026-08-15): plain
# --unroll cannot run at stencil>1 because the model predicts only its
# centre pixel while its t+1 input window needs the NEIGHBOURS' t+1
# embeddings — but each neighbour's t+1 embedding is a depth-1 prediction
# from that neighbour's own fully-observed window. The training-loop block
# forwards all S slot pixels' windows once (detached), zeroes the dead
# slots, and assembles the centre pixel's t+1 input window.
#
# The one sharp edge is ALIGNMENT: b_rep repeats each window's base S
# times, p_rep is the row-major flatten of the [bu, S] slot-pixel table,
# and newstep is the row-major reshape of the [bu*S, d_z] predictions back
# to [bu, S*d_z]. If any of those three disagree on ordering, slot s of
# window i silently receives the prediction for a DIFFERENT window's slot
# — the run trains, the curves look normal, and only the rolled AUC
# quietly fails. So the alignment is the test, not the arithmetic.
#
# The logic is replicated here rather than imported because it lives
# inside the training loop; gather_stencil — the half that CAN be imported
# — is imported, so the test breaks if its slot-major flatten changes.
# (Same pattern as tests/test_e029_znoise.py.)
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from ml.temporal import gather_stencil                          # noqa: E402


def one_hop(model, Zt, NBR_t, K, wbase, wp, zseq, mseq, sctx, mfut,
            uw_batch):
    """Byte-for-byte the temporal.py --unroll-wide assembly (device-free)."""
    bu = min(uw_batch, zseq.shape[0])
    q = NBR_t[wp[:bu]]
    valid = (q >= 0)
    safe = q.clamp(min=0)
    Sn = q.shape[1]
    b_rep = wbase[:bu].repeat_interleave(Sn)
    p_rep = safe.reshape(-1)
    z1 = gather_stencil(Zt, b_rep, p_rep, NBR_t, K)
    m1 = mseq[:bu].repeat_interleave(Sn, 0)
    s1 = sctx[p_rep]
    with torch.no_grad():
        pred1, _ = model(z1, m1, s1)
    step1 = pred1[:, -1]
    step1 = step1 * valid.reshape(-1, 1)
    newstep = step1.reshape(bu, -1)
    zseq2 = torch.cat([zseq[:bu, 1:], newstep[:, None]], 1)
    mseq2 = torch.cat([mseq[:bu, 1:], mfut[:bu, 0:1]], 1)
    return zseq2, mseq2, newstep, valid


def toy():
    """T=10 months, P=6 pixels in a line, d_z=3, K=4, S=3 (centre, west,
    east; -1 off the ends). Z[t, p, :] = t*10 + p — every embedding names
    its own (t, p), so a misrouted prediction is a wrong NUMBER, not a
    wrong distribution."""
    T, P, d_z, K = 10, 6, 3, 4
    Zt = (torch.arange(T)[:, None] * 10
          + torch.arange(P)[None, :]).float()[..., None].expand(T, P, d_z)
    Zt = Zt.contiguous()
    NBR = torch.full((P, 3), -1, dtype=torch.long)
    NBR[:, 0] = torch.arange(P)                        # centre = slot 0
    NBR[1:, 1] = torch.arange(P - 1)                   # west
    NBR[:-1, 2] = torch.arange(1, P)                   # east
    Mt = torch.stack([torch.arange(T).float() / 10,
                      torch.arange(T).float()], 1)     # [T, 2], names t
    sctx = torch.arange(P).float()[:, None].expand(P, 4).contiguous()
    return Zt, NBR, Mt, sctx, T, P, d_z, K


class PersistencePlusOne(torch.nn.Module):
    """pred[:, j] = the CENTRE slot of input step j, + 1. Depth-1 through
    it, a window ending at t on pixel q predicts (t*10 + q) + 1 — a value
    the test can compute independently for every (window, slot) pair."""
    def __init__(self, d_z):
        super().__init__()
        self.d_z = d_z

    def forward(self, zseq, mseq, sctx):
        return zseq[:, :, :self.d_z] + 1.0, None


def test_slot_alignment_and_dead_zeros():
    Zt, NBR, Mt, sctx, T, P, d_z, K = toy()
    model = PersistencePlusOne(d_z)
    # windows ending at t=5 (base 2) on pixels 0 (west edge: slot 1 dead),
    # 3 (interior: all live), 5 (east edge: slot 2 dead)
    wp = torch.tensor([0, 3, 5])
    t = torch.tensor([5, 5, 5])
    wbase = t - K + 1
    zseq = gather_stencil(Zt, wbase, wp, NBR, K)
    mseq = torch.stack([Mt[wbase + j] for j in range(K)], 1)
    mfut = torch.stack([Mt[t + 1 + u] for u in range(2)], 1)
    zseq2, mseq2, newstep, valid = one_hop(
        model, Zt, NBR, K, wbase, wp, zseq, mseq, sctx, mfut, uw_batch=64)
    ns = newstep.view(len(wp), 3, d_z)
    for i, p in enumerate(wp.tolist()):
        for s in range(3):
            q = int(NBR[p, s])
            if q < 0:
                assert torch.all(ns[i, s] == 0.0), (
                    f"dead slot ({i},{s}) must be exact zeros")
            else:
                want = 5 * 10 + q + 1                # (t*10 + q) + 1
                assert torch.all(ns[i, s] == want), (
                    f"slot ({i},{s}) got {ns[i, s, 0]}, want {want} — "
                    f"the b_rep/p_rep/reshape ordering disagrees")


def test_window_slides_and_months_advance():
    Zt, NBR, Mt, sctx, T, P, d_z, K = toy()
    model = PersistencePlusOne(d_z)
    wp = torch.tensor([2, 4])
    t = torch.tensor([4, 6])                          # mixed end months
    wbase = t - K + 1
    zseq = gather_stencil(Zt, wbase, wp, NBR, K)
    mseq = torch.stack([Mt[wbase + j] for j in range(K)], 1)
    mfut = torch.stack([Mt[t + 1 + u] for u in range(2)], 1)
    zseq2, mseq2, newstep, valid = one_hop(
        model, Zt, NBR, K, wbase, wp, zseq, mseq, sctx, mfut, uw_batch=64)
    # first K-1 steps are the old window's steps 1..K-1, untouched
    assert torch.equal(zseq2[:, :K - 1], zseq[:, 1:])
    assert torch.equal(zseq2[:, -1], newstep)
    # months: shifted by one, the appended step carries Mt[t+1]
    assert torch.equal(mseq2[:, :K - 1], mseq[:, 1:])
    assert torch.equal(mseq2[:, -1], Mt[t + 1])


def test_sub_batch_and_reach():
    Zt, NBR, Mt, sctx, T, P, d_z, K = toy()
    model = PersistencePlusOne(d_z)
    wp = torch.tensor([1, 2, 3, 4])
    t = torch.tensor([5, 5, 5, 5])
    wbase = t - K + 1
    zseq = gather_stencil(Zt, wbase, wp, NBR, K)
    mseq = torch.stack([Mt[wbase + j] for j in range(K)], 1)
    mfut = torch.stack([Mt[t + 1 + u] for u in range(2)], 1)
    zseq2, _, _, _ = one_hop(
        model, Zt, NBR, K, wbase, wp, zseq, mseq, sctx, mfut, uw_batch=2)
    assert zseq2.shape[0] == 2, "--uw-batch must bound the sub-batch"
    # the pool-reach expression: UW=2 at U=1 must guard t+2 exactly as a
    # plain U=2 would (temporal.py builds reach from UF = max(U, UW))
    U, UW, D = 1, 2, ()
    UF = max(U, UW)
    reach = sorted(set(range(1, UF + 1)) | set(D))
    assert reach == [1, 2]
    U, UW = 1, 0
    reach = sorted(set(range(1, max(U, UW) + 1)) | set(D))
    assert reach == [1], "UW=0 must leave the default pool bit-identical"


if __name__ == "__main__":
    test_slot_alignment_and_dead_zeros()
    test_window_slides_and_months_advance()
    test_sub_batch_and_reach()
    print("ok")

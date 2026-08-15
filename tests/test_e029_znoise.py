# E-029b input-noise: the two properties that make it correct, pinned.
#
# The injection trains robustness to the roll's input distribution, where
# every slot is a model prediction. Its one sharp edge is the dead-slot
# convention: a slot outside the window or on land is encoded as EXACT
# ZEROS, both in training (gather_stencil) and in the roll
# (rollout_spatial feeds zeros off-window) — test_zero_weight_equivalence
# is built on it. Noise on a dead slot would train the model against an
# input state the roll never produces, and the failure would be silent:
# the run trains, the curves look normal, only the rolled AUC quietly
# fails to improve. So the masking is the test, not the noise.
#
# The logic is replicated here rather than imported because it is four
# lines inside the training loop; the assertion pins the SEMANTICS those
# four lines must keep (same pattern as test_e022_stencil's drawings).
import torch


def inject(zseq, sigma, d_z):
    """Byte-for-byte the temporal.py training-loop injection."""
    z4 = zseq.view(*zseq.shape[:2], -1, d_z)
    live = (z4 != 0).any(-1, keepdim=True)
    return (z4 + torch.randn_like(z4) * sigma * live).view(zseq.shape)


def test_dead_slots_stay_exact_zero():
    torch.manual_seed(0)
    n, K, S, d_z = 8, 6, 5, 16
    zseq = torch.randn(n, K, S * d_z)
    z4 = zseq.view(n, K, S, d_z)
    z4[:, :, 2, :] = 0.0                      # slot 2 dead everywhere
    z4[3, :, 4, :] = 0.0                      # slot 4 dead for one sample
    out = inject(zseq.clone(), 0.7, d_z).view(n, K, S, d_z)
    assert torch.all(out[:, :, 2, :] == 0.0), "dead slot got noised"
    assert torch.all(out[3, :, 4, :] == 0.0), "per-sample dead slot got noised"


def test_live_slots_get_calibrated_noise():
    torch.manual_seed(1)
    n, K, S, d_z = 64, 6, 5, 16
    zseq = torch.randn(n, K, S * d_z)
    out = inject(zseq.clone(), 0.7, d_z)
    delta = out - zseq
    # every live element perturbed, at the stated scale (loose 3-sigma band
    # on the empirical std over ~30k draws)
    assert delta.abs().max() > 0
    assert 0.65 < delta.std().item() < 0.75


def test_sigma_zero_is_identity():
    torch.manual_seed(2)
    zseq = torch.randn(4, 6, 80)
    out = inject(zseq.clone(), 0.0, 16)
    assert torch.equal(out, zseq)

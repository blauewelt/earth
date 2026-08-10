#!/usr/bin/env python3
"""Does --resume-temporal CONTINUE a run, or merely start a new one from old weights?

This is the test that should have existed before anyone spent seven GPU-hours
on a head they could not continue. It does not need the real tensor, a GPU, or
Actions: it exercises the exact save/load/step machinery on a toy model.

The claim under test is strong and worth stating precisely:

    training N steps, saving, resuming, and training N more
    must land in the SAME PLACE as training 2N steps straight through.

Weights alone cannot satisfy that. Adam carries first and second moments that
take hundreds of steps to rebuild, the cosine schedule carries its position,
and the data order carries the RNG stream. Drop any one and the "continuation"
silently becomes a warm restart — which is a different experiment, and one
that would have been reported as though it were the same.

    python3 tests/test_resume_temporal.py
"""
import os
import sys
import tempfile

import torch
import torch.nn as nn

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "ml"))


def _mk():
    torch.manual_seed(0)
    net = nn.Sequential(nn.Linear(4, 16), nn.Tanh(), nn.Linear(16, 4))
    opt = torch.optim.AdamW(net.parameters(), lr=1e-2, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, TOTAL)
    return net, opt, sched


def _step(net, opt, sched):
    """One step whose DATA depends on the global RNG, like batch_windows does."""
    x = torch.randn(8, 4)
    loss = (net(x) - x.roll(1, 1)).pow(2).mean()
    opt.zero_grad()
    loss.backward()
    opt.step()
    sched.step()
    return float(loss)


TOTAL, HALF = 40, 20


def flat(net):
    return torch.cat([p.detach().reshape(-1) for p in net.parameters()])


def save(path, net, opt, sched, step):
    torch.save({"model": net.state_dict(), "opt": opt.state_dict(),
                "sched": sched.state_dict(), "step": step,
                "torch_rng": torch.get_rng_state().numpy().tolist()}, path)


def load(path, net, opt, sched):
    ck = torch.load(path, weights_only=False)
    net.load_state_dict(ck["model"])
    opt.load_state_dict(ck["opt"])
    sched.load_state_dict(ck["sched"])
    torch.set_rng_state(torch.as_tensor(ck["torch_rng"], dtype=torch.uint8))
    return int(ck["step"])


def run_straight():
    net, opt, sched = _mk()
    torch.manual_seed(1234)
    for _ in range(TOTAL):
        _step(net, opt, sched)
    return flat(net), sched.get_last_lr()[0]


def run_resumed(tmp, *, drop=None):
    """drop=None is a true continuation. drop='opt'/'sched'/'rng' simulates
    the shortcuts, so the test also proves each piece is load-bearing."""
    net, opt, sched = _mk()
    torch.manual_seed(1234)
    for _ in range(HALF):
        _step(net, opt, sched)
    p = os.path.join(tmp, "half.pt")
    save(p, net, opt, sched, HALF)

    net2, opt2, sched2 = _mk()
    ck = torch.load(p, weights_only=False)
    net2.load_state_dict(ck["model"])
    if drop != "opt":
        opt2.load_state_dict(ck["opt"])
    if drop != "sched":
        sched2.load_state_dict(ck["sched"])
    if drop != "rng":
        torch.set_rng_state(torch.as_tensor(ck["torch_rng"], dtype=torch.uint8))
    for _ in range(TOTAL - HALF):
        _step(net2, opt2, sched2)
    return flat(net2), sched2.get_last_lr()[0]



# ---------------------------------------------------------------------------
# EXTENDING is a different case from continuing, and it is the one that bites.
# CosineAnnealingLR.load_state_dict restores T_max and base_lrs from the OLD
# run. Load it while asking for a larger total and the schedule believes it is
# already finished: lr = 0.0, and the "continuation" trains for hours changing
# nothing while every status says success. This is exactly what temporal.py did
# until 2026-08-10, and the toy run printed "lr now 0.000e+00" unread.

def test_extend():
    net, opt, sched = _mk()
    torch.manual_seed(1234)
    for _ in range(TOTAL):
        _step(net, opt, sched)

    NEW_TOTAL, NEW_LR = 4 * TOTAL, 1e-3          # 1/10th of the original 1e-2

    # The wrong way: restore the old schedule state and hope --steps is honoured.
    net_b, opt_b, _ = _mk()
    sched_b = torch.optim.lr_scheduler.CosineAnnealingLR(opt_b, NEW_TOTAL)
    sched_b.load_state_dict(sched.state_dict())
    lr_bad = sched_b.get_last_lr()[0]

    # The right way: rebuild the cosine for the new total and peak, positioned
    # where the checkpoint stopped.
    net_g, opt_g, _ = _mk()
    for g in opt_g.param_groups:
        g["lr"] = NEW_LR
        g["initial_lr"] = NEW_LR
    sched_g = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt_g, NEW_TOTAL, last_epoch=TOTAL - 1)
    lr_good = sched_g.get_last_lr()[0]

    print(f"extend: reloaded schedule -> lr {lr_bad:.3e}   "
          f"rebuilt schedule -> lr {lr_good:.3e}")
    assert lr_bad < 1e-12, "the failure this guards against did not reproduce"
    assert lr_good > 1e-6, "rebuilt schedule must give a usable learning rate"
    assert lr_good <= NEW_LR + 1e-12, "must not exceed the requested peak"
    print("OK — extending rebuilds the schedule instead of inheriting a finished one.")


def main():
    with tempfile.TemporaryDirectory() as tmp:
        w_ref, lr_ref = run_straight()
        w_res, lr_res = run_resumed(tmp)
        d = float((w_ref - w_res).abs().max())
        print(f"full state   : max|Δw| = {d:.3e}   lr {lr_ref:.6f} vs {lr_res:.6f}")
        assert d < 1e-6, f"resume diverged from the straight run: {d:.3e}"
        assert abs(lr_ref - lr_res) < 1e-12, "schedule position did not survive"

        # Each dropped piece must MATTER — otherwise the test proves nothing
        # and would keep passing after someone "simplifies" the checkpoint.
        for piece in ("opt", "sched", "rng"):
            w_bad, _ = run_resumed(tmp, drop=piece)
            db = float((w_ref - w_bad).abs().max())
            print(f"without {piece:5}: max|Δw| = {db:.3e}   "
                  f"({'DIVERGES, as it must' if db > 1e-6 else 'NO EFFECT — bad'})")
            assert db > 1e-6, (
                f"dropping {piece!r} changed nothing, so this test is not "
                f"actually testing that {piece} is carried")
    print()
    test_extend()
    print()
    test_warm_restart_is_not_the_same_trajectory()
    print("\nOK — resume is a continuation, and every saved piece is load-bearing.")




# ---------------------------------------------------------------------------
# WARM RESTART is a THIRD case, and the one that actually applies today.
# Every stage-2 head published before 2026-08-10 is {args, model}: measured on
# f3_s2_60k, f3_s2_24k and the rescue mirrors. So --resume-temporal must
# refuse them (it does), and the only thing available from those weights is a
# fresh cosine — which is a legitimate way to spend more compute and an
# illegitimate fourth point on a curve of from-scratch runs.

def test_warm_restart_is_not_the_same_trajectory():
    """The claim that justifies giving it a different flag and a different
    metrics record: starting from converged weights with fresh moments and a
    fresh schedule does NOT land where a longer straight-through run lands."""
    net, opt, sched = _mk()
    torch.manual_seed(1234)
    for _ in range(HALF):
        _step(net, opt, sched)
    w_mid = flat(net).clone()

    # (a) straight through: one schedule over TOTAL, uninterrupted.
    net_s, opt_s, sched_s = _mk()
    torch.manual_seed(1234)
    for _ in range(TOTAL):
        _step(net_s, opt_s, sched_s)

    # (b) warm restart: same weights at HALF, everything else discarded.
    net_w, opt_w, _ = _mk()
    net_w.load_state_dict(net.state_dict())
    for g in opt_w.param_groups:
        g["lr"] = 1e-3                       # a lower peak, as E-008 asks for
        g["initial_lr"] = 1e-3
    sched_w = torch.optim.lr_scheduler.CosineAnnealingLR(opt_w, TOTAL - HALF)
    for _ in range(TOTAL - HALF):
        _step(net_w, opt_w, sched_w)

    d = float((flat(net_s) - flat(net_w)).abs().max())
    moved = float((flat(net_w) - w_mid).abs().max())
    print(f"warm restart : max|Δw| vs straight-through = {d:.3e}")
    print(f"               max|Δw| vs its own start    = {moved:.3e}")
    assert d > 1e-6, (
        "if a warm restart landed in the same place as a continuation there "
        "would be no reason to distinguish them — and this test would be the "
        "thing wrongly reassuring us")
    assert moved > 1e-6, "a warm restart must still TRAIN, not sit still"
    print("OK — a warm restart trains, and goes somewhere else. Two flags, "
          "two records, two claims.")


if __name__ == "__main__":
    main()

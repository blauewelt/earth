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
    print("\nOK — resume is a continuation, and every saved piece is load-bearing.")


if __name__ == "__main__":
    main()

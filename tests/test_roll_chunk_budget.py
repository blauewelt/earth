#!/usr/bin/env python3
"""The rollout gather's chunk is a BYTE budget, not a row count.

Eval wave 6B (#353, 2026-08-16) died with

    torch.OutOfMemoryError: Tried to allocate 4.22 GiB

inside `roll_step`'s `zj = Zwin[nbr.clamp(min=0)]`, on its THIRD head, after
two heads of the same 90-slot width had rolled fine. `--chunk` counts pixels;
the gather it bounds is [n, S, K, dz], so its real size scales with the
stencil width S — a knob the row count cannot see. At the 8192 default a
90-slot head asks for 8192·90·24·64·4 B = 4.5 GB in ONE allocation, which a
24 GB card cannot satisfy once weights, the Z window and allocator
fragmentation have taken their share. The two heads that succeeded are the
point: this is a marginal request that fails on fragmentation, i.e. the
worst kind, because it passes in testing and fails in the sixth hour.

This is the third OOM of one family (E-027 incidents 1 and 2 taught it for
the trainer's eval batch; `_chunked_forward` is that fix). The rule these
tests pin is the general form: derive the row count from a byte target so
that ANY future width is bounded automatically, and let `--chunk` act only
as an upper bound so narrow stencils keep their speed.

Run: python3 tests/test_roll_chunk_budget.py
"""
import os
import sys

import torch

ML = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "ml")
sys.path.insert(0, ML)

from rollout_spatial import roll_step                          # noqa: E402

BUDGET = 1 << 30        # the byte target roll_step budgets each gather to
FLOAT32 = 4


class SpyZ(torch.Tensor):
    """A tensor that records the row count of every advanced-index gather."""


def effective_chunks(S, K, dz, P, cap):
    """Run roll_step against a recording model and return the chunk sizes."""
    seen = []

    class RecordingModel(torch.nn.Module):
        def forward(self, zin, mfeat, ctx):
            seen.append(zin.shape[0])
            return torch.zeros(zin.shape[0], zin.shape[1], dz), None

    Zwin = torch.zeros(P, K, dz)
    NBR = torch.zeros(P, S, dtype=torch.long)
    static_ctx = torch.zeros(P, 4)
    mfeat = torch.zeros(K, 2)
    roll_step(RecordingModel(), Zwin, NBR, static_ctx, mfeat, cap, amp=False)
    return seen


def main():
    fails = []

    def check(cond, msg):
        print(("  ok   " if cond else "  FAIL ") + msg)
        if not cond:
            fails.append(msg)

    K, dz, P, cap = 24, 64, 20000, 8192

    print("wide stencils are bounded by BYTES, not by the row cap")
    for S in (56, 90, 145, 233):
        chunks = effective_chunks(S, K, dz, P, cap)
        row_bytes = S * K * dz * FLOAT32
        biggest = max(chunks) * row_bytes
        check(biggest <= BUDGET * 1.01,
              f"S={S}: largest gather {biggest/1e9:.2f} GB <= 1 GiB budget")
        # the failing case, stated as itself: 8192 rows at S=90 was 4.5 GB
        if S == 90:
            check(max(chunks) < 8192,
                  f"S=90 chunk {max(chunks)} is below the 8192 default that OOMed")

    print("narrow stencils keep the caller's cap — the fix costs no speed")
    for S in (1, 9):
        chunks = effective_chunks(S, K, dz, P, cap)
        check(max(chunks) == cap,
              f"S={S}: chunk stays at the {cap} cap")

    print("--chunk remains an UPPER bound, never raised by the budget")
    chunks = effective_chunks(9, K, dz, P, 512)
    check(max(chunks) == 512, "a small --chunk is respected at narrow width")

    print("every pixel is still visited exactly once")
    for S in (1, 90):
        chunks = effective_chunks(S, K, dz, P, cap)
        check(sum(chunks) == P, f"S={S}: {sum(chunks)} rows == P={P}")

    print("stencil-1 (NBR_t None) takes the no-gather path unchanged")
    seen = []

    class RM(torch.nn.Module):
        def forward(self, zin, mfeat, ctx):
            seen.append(zin.shape[0])
            return torch.zeros(zin.shape[0], zin.shape[1], dz), None

    roll_step(RM(), torch.zeros(P, K, dz), None, torch.zeros(P, 4),
              torch.zeros(K, 2), cap, amp=False)
    check(max(seen) == cap and sum(seen) == P,
          "NBR_t None: full cap, all pixels")

    print()
    if fails:
        print(f"{len(fails)} FAILED")
        for f in fails:
            print("  - " + f)
        sys.exit(1)
    print("all roll-chunk budget checks passed")


if __name__ == "__main__":
    main()

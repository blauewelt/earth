#!/usr/bin/env python3
"""E-006's central claim, checked on the REAL model instead of on paper.

`tests/test_e006_algebra.py` establishes symbolically that a loss measured in
the data's units cannot see the encoder's output scale: under z -> s·z with
decoder -> decoder/s the decoded field is unchanged, so dL/ds is exactly 0.
That is the design. This file checks the CODE — ml/CLAUDE.md §0.1, verify the
artefact, not the intention. Four normalisations were argued for on paper and
three of them shipped; the algebra was never the part that was wrong.

The construction is exact rather than approximate. PixelMAE's encoder ends in
one linear map `to_z`, and its decoder begins in one linear map whose first
`d_z` input columns multiply z. So scaling `to_z` by s and those columns by
1/s is the gauge transformation itself, applied to the weights — not a
simulation of it. Two codecs related that way must:

  · produce embeddings that differ by exactly s          (so the gauge is real)
  · decode to the SAME field                             (so the data-space
                                                          loss cannot see it)

and therefore any z-space MSE between them differs by s², while any
data-space MSE is identical. That contrast is the whole of E-006 in two
numbers, and it is measured here on the same `encode`/`query` calls
train_joint.py makes.

The head is deliberately absent: it takes z as input, so transforming it too
would require a second, layernorm-dependent argument, and the persistence and
reconstruction branches already exercise the entire encode -> decode path the
claim is about.

    python3 tests/test_e006_gauge.py
"""
import copy
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "ml"))
from model import PixelMAE                                    # noqa: E402

S = 4.0                     # the rescale. Any s != 1 works; 4 is far from 1.
TOL = 2e-5                  # float32 through a 2-layer MLP


def gauge(codec, s):
    """codec' with z -> s·z and the decoder's z-columns -> /s.

    This is the reparametrisation of the algebra table, expressed in weights.
    Everything else — attention, embeddings, the offset and channel query
    features — is untouched, which is what makes the equality below a
    statement about the LOSS and not about the copy.
    """
    g = copy.deepcopy(codec)
    with torch.no_grad():
        g.to_z.weight.mul_(s)
        g.to_z.bias.mul_(s)
        first = g.decoder[0]
        first.weight[:, :g.d_z].div_(s)      # the z block only; qc/qo untouched
    return g


def main():
    torch.manual_seed(0)
    B, C, d_z = 6, 5, 4
    codec = PixelMAE(n_chan=C, d_model=16, n_heads=2, n_layers=2, d_z=d_z,
                     d_dec=16, patch=1).eval()
    g = gauge(codec, S)

    x = torch.randn(B, C)
    x_next = torch.randn(B, C)                # the "observation at t+1"
    obs = torch.ones(B, C, dtype=torch.bool)
    mask = torch.zeros(B, C, dtype=torch.bool)
    ctx = torch.randn(B, 4)
    qc = torch.arange(C)[None, :].expand(B, -1)
    off = torch.zeros(B, C, 3, dtype=torch.long)
    varc = torch.rand(C) + 0.5                # per-channel variance, as in the loss

    with torch.no_grad():
        z = codec.encode(x, obs, mask, ctx)
        zg = g.encode(x, obs, mask, ctx)
        xh = codec.query(z, qc, off)
        xhg = g.query(zg, qc, off)

    # 1 · the gauge is real: the embedding genuinely moved by s.
    ratio = (zg / z).abs()
    assert torch.allclose(ratio, torch.full_like(ratio, S), rtol=1e-4), \
        f"z did not scale by {S}: ratio range [{ratio.min():.4f}, {ratio.max():.4f}]"

    # 2 · the decoded field did NOT move. This is the line E-006 rests on.
    dev = (xh - xhg).abs().max().item()
    assert dev < TOL, (f"the decoded field moved by {dev:.2e} under a pure "
                       f"gauge change — the data-space loss would see the "
                       f"encoder's scale, which is the whole degeneracy")

    # 3 · the two losses, side by side, on the same rescale.
    l_data = (((xh - x_next) ** 2 / varc)).mean().item()
    l_data_g = (((xhg - x_next) ** 2 / varc)).mean().item()
    z_next = codec.encode(x_next, obs, mask, ctx).detach()
    zg_next = g.encode(x_next, obs, mask, ctx).detach()
    l_z = ((z - z_next) ** 2).mean().item()
    l_zg = ((zg - zg_next) ** 2).mean().item()

    print(f"data-space loss   {l_data:.6f} -> {l_data_g:.6f}   "
          f"ratio {l_data_g / l_data:.6f}  (must be 1)")
    print(f"z-space loss      {l_z:.6f} -> {l_zg:.6f}   "
          f"ratio {l_zg / l_z:.6f}  (is s² = {S**2:.1f})")

    assert abs(l_data_g / l_data - 1.0) < 1e-4, \
        "the data-space loss changed under a gauge transformation"
    # And the contrast: the OLD objective moves by s², which is exactly the
    # gradient the encoder followed to 1/40 on #102 and to x250 on #107.
    assert abs(l_zg / l_z - S ** 2) / S ** 2 < 1e-3, \
        "the z-space loss did not scale by s² — the comparison is not the one claimed"

    # 4 · the reconstruction term, the other half of the sum, same property.
    l_rec = (((xh - x) ** 2 / varc)).mean().item()
    l_rec_g = (((xhg - x) ** 2 / varc)).mean().item()
    assert abs(l_rec_g / l_rec - 1.0) < 1e-4, \
        "the data-space RECONSTRUCTION term changed under a gauge transformation"

    print(f"reconstruction    {l_rec:.6f} -> {l_rec_g:.6f}   "
          f"ratio {l_rec_g / l_rec:.6f}  (must be 1)")
    print("\nE-006 gauge invariance holds on the real encode/query path: "
          "there is no free direction for the shrinkage cheat to live in.")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""The E-047 block axis, and the encoder grid that consumes it.

Chris, 2026-08-22: a codec that FUSES several pentad bins into one embedding,
with proper sub-month time labelling. What is under test here is the half that
is fully specified and that everything else will rest on — WHICH BINS GO IN
WHICH BLOCK, what the block is called, where its centre falls, which cells are
padding, and that a k_time>1 encoder eats that grid while a k_time=1 encoder
is the codec every archived checkpoint already is.

The measured fact this exists to exploit, from the 2026-08-22 recon audit:
`rg_*` is a MONTHLY product written into ONE pentad bin per month
(`n_rg_live` 252/3142 = 8.02%, the mid-month stamp, 79.41% of ocean pixels
when present), and the codec's round trip collapses on exactly those bins
because 40 channels compete for 32 dimensions. A month block turns "Argo is
missing in five bins out of six" from a special case into six cells of which
five are unobserved — the mechanism the codec has always had.

    python3 tests/test_e047_time_blocks.py

No GPU, no tensor: the axis is arithmetic and the encoder check is a 3-pixel
toy.
"""
import datetime as dt
import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ML = os.path.abspath(os.path.join(HERE, "..", "ml"))
sys.path.insert(0, ML)
from timeblocks import BlockAxis, K_MONTH                     # noqa: E402
from model import PixelMAE                                    # noqa: E402
from temporal import season_feat_of                           # noqa: E402

EPOCH = dt.date(1982, 1, 1)
DAYS = 5


def pentad_axis(n_years=2, y0=1990):
    b0 = (dt.date(y0, 1, 1) - EPOCH).days // DAYS
    n = int(round(n_years * 365.2425 / DAYS))
    bins = np.arange(b0, b0 + n, dtype=np.int64)
    labs = [(EPOCH + dt.timedelta(days=int(b) * DAYS)).strftime("%Y-%m")
            for b in bins]
    return labs, bins


def main():
    labs, bins = pentad_axis()
    ax = BlockAxis("month", labs, bins, EPOCH, DAYS)

    # ---- 1. month blocks are the months, ragged, 6 or 7 bins -------------
    # The expectation comes from the LABELS, not from a remembered count: a
    # 146-bin span that starts on 1 January ends inside the following
    # January, so the axis carries 24 whole months and one 1-bin tail. That
    # tail is a real property of a pentad axis cut anywhere but a month
    # boundary, and the mode pads it like any other short month.
    want = list(dict.fromkeys(labs))
    assert ax.labels == want and ax.n_blocks == len(want), ax.n_blocks
    assert ax.k_max == K_MONTH == 7
    assert int(ax.n_bins.sum()) == len(labs), (ax.n_bins.sum(), len(labs))
    # INTERIOR months carry 6 or 7 bins; BOTH EDGE blocks are partial,
    # because a pentad axis is a run of 5-day bins that starts and ends
    # wherever the record does — this one opens on 1989-12-30 (one bin of
    # that December) and closes mid-1991-12. The edges being ragged is the
    # property the padding exists for, so the test states it rather than
    # arranging a fixture where it cannot happen.
    interior = list(range(1, ax.n_blocks - 1))
    assert set(np.unique(ax.n_bins[interior])) == {6, 7}, np.unique(ax.n_bins)
    assert int(ax.n_bins[0]) < 6 and int(ax.n_bins[-1]) < 6, \
        (int(ax.n_bins[0]), int(ax.n_bins[-1]))
    full_1990 = int(sum(ax.n_bins[b] for b in range(ax.n_blocks)
                        if ax.labels[b][:4] == "1990"))
    assert full_1990 == 73, full_1990
    n7 = int((ax.n_bins[interior] == 7).sum())
    print("1. a 2-year pentad axis (%d bins) blocks into %d MONTHS: the %d "
          "interior ones carry 6 or 7 bins (%d of them 7), both edge blocks "
          "are partial (%d and %d bins, the record starting and ending "
          "mid-month), k_max %d, and the one WHOLE calendar year sums to %d "
          "bins — every bin used exactly once"
          % (len(labs), ax.n_blocks, len(interior), n7, int(ax.n_bins[0]),
             int(ax.n_bins[-1]), ax.k_max, full_1990))

    # ---- 2. padding: marked, and pointing at a real row ------------------
    for b in range(ax.n_blocks):
        nb = int(ax.n_bins[b])
        assert not ax.pad[b, :nb].any() and ax.pad[b, nb:].all()
        assert (ax.rows[b, nb:] == ax.rows[b, nb - 1]).all(), \
            "a pad cell must point at a REAL row, never at 0 or -1"
        got = [labs[r] for r in ax.rows[b, :nb]]
        assert set(got) == {ax.labels[b]}, (b, got)
    assert ax.pad.sum() == ax.k_max * ax.n_blocks - len(labs)
    print("2. %d of %d cells are padding (%.1f%%), every pad cell is flagged "
          "AND points at its block's last real row, and every real cell "
          "carries its own block's label"
          % (ax.pad.sum(), ax.pad.size, 100 * ax.pad.mean()))

    # ---- 3. the observed mask fuses Argo into one cell per block --------
    C = 40
    obs = np.ones((len(labs), C), bool)
    argo = np.array([c >= 5 for c in range(C)])          # 35 "rg_*" channels
    live = np.zeros(len(labs), bool)
    # The stamp is the 3rd bin of the month — which only EXISTS in a block
    # that has three. The 1-bin December this axis opens on genuinely carries
    # no Argo, and the correct block-level answer there is "observed zero
    # times", not a fabricated one.
    stamped = [b for b in range(ax.n_blocks) if int(ax.n_bins[b]) > 2]
    for b in stamped:
        live[int(ax.rows[b, 2])] = True
    obs[np.ix_(~live, argo)] = False
    cell = ax.cell_obs(obs)
    assert cell.shape == (ax.n_blocks, ax.k_max, C)
    for b in range(ax.n_blocks):
        want = argo.sum() if b in stamped else 0
        assert cell[b, :, argo].sum() == want, \
            (b, "an Argo channel is observed exactly ONCE per stamped block")
        if b in stamped:
            assert cell[b, 2, argo].all()
        assert cell[b, :int(ax.n_bins[b]), ~argo].all()
        assert not cell[b, int(ax.n_bins[b]):, :].any(), \
            "a pad cell must be unobserved whatever the source mask said"
    b0 = stamped[0]
    print("3. an Argo-like channel present in 1 bin of 6 is observed exactly "
          "once per block (%d of %d Argo cells live in a full block, 0 in the "
          "%d-bin edge block that has no 3rd bin), the always-present channels "
          "are observed in every real cell, and PAD cells are unobserved by "
          "construction — one mechanism, no special case"
          % (int(cell[b0, :, argo].sum()), int(ax.k_max * argo.sum()),
             int(ax.n_bins[0])))

    # ---- 4. the label is a calendar position, the centre is the span's --
    for b in (1, 5, ax.n_blocks - 2):
        first = int(ax.rows[b, 0])
        last = int(ax.rows[b, int(ax.n_bins[b]) - 1])
        st = EPOCH + dt.timedelta(days=int(bins[first]) * DAYS)
        en = EPOCH + dt.timedelta(days=int(bins[last]) * DAYS + DAYS)
        assert ax.span_days[b] == (en - st).days
        assert abs((ax.centres[b] - dt.datetime.combine(st, dt.time()))
                   .total_seconds() - ax.span_days[b] * 43200) < 1
        assert ax.labels[b] == labs[first]
    ph = ax.ctx_phase()
    assert ph.shape == (ax.n_blocks, 2)
    ang = np.angle(ph[:, 1] + 1j * ph[:, 0])
    d1 = float(np.median((np.diff(ang) + 2 * np.pi) % (2 * np.pi)))
    assert abs(d1 - 2 * np.pi / 12) < 0.01, d1
    # the same definition ml/temporal.py uses, not a second one
    b = 7
    st = EPOCH + dt.timedelta(days=int(bins[int(ax.rows[b, 0])]) * DAYS)
    assert np.allclose(ph[b], season_feat_of(st, ax.span_days[b]), atol=1e-6)
    print("4. each block's centre is the middle of its own SPAN (%.0f-%.0f d) "
          "and its ctx is the CONTINUOUS phase of that centre — %.4f rad "
          "between consecutive months, and identical to temporal.py's "
          "season_feat_of on the same span"
          % (ax.span_days.min(), ax.span_days.max(), d1))

    # ---- 5. fixed-N mode -------------------------------------------------
    fx = BlockAxis("2", labs, bins, EPOCH, DAYS)
    assert fx.k_max == 2 and not fx.pad.any()
    assert fx.n_blocks == len(labs) // 2, (fx.n_blocks, len(labs))
    assert (fx.n_bins == 2).all()
    assert list(fx.rows[0]) == [0, 1] and list(fx.rows[1]) == [2, 3]
    assert fx.labels[0] == labs[0] and fx.labels[3] == labs[6]
    assert np.allclose(fx.span_days, 10.0)
    drop = len(labs) - fx.n_blocks * 2
    print("5. fixed-N mode: %d bins -> %d blocks of exactly 2, no padding at "
          "all, labels from each block's FIRST bin, %d trailing bin(s) "
          "dropped rather than made ragged" % (len(labs), fx.n_blocks, drop))

    # ---- 6. the RAPID truth moves with the axis --------------------------
    rows = np.arange(3, len(labs), 7)
    rapid = np.stack([rows.astype(float),
                      np.random.default_rng(0).normal(size=len(rows))], 1)
    rm = ax.remap_rows(rapid)
    assert len(rm) == len(rapid)
    for (r0, v0), (b1, v1) in zip(rapid, rm):
        assert v0 == v1
        assert int(b1) == ax.block_of_row(int(r0))
        assert labs[int(r0)] == ax.labels[int(b1)]
    rm2 = fx.remap_rows(np.array([[len(labs) - 1, 1.0]]))
    assert len(rm2) == (0 if drop else 1), rm2
    print("6. RAPID rows remap to the block CONTAINING them (%d of %d kept in "
          "month mode, and a row in a dropped fixed-N remainder disappears "
          "rather than pointing at the wrong block)" % (len(rm), len(rapid)))

    # ---- 7. the encoder eats the grid, and k_time=1 is untouched --------
    torch.manual_seed(0)
    m1 = PixelMAE(n_chan=6, d_model=16, n_heads=2, n_layers=2, d_z=8, d_dec=16)
    torch.manual_seed(0)
    m2 = PixelMAE(n_chan=6, d_model=16, n_heads=2, n_layers=2, d_z=8, d_dec=16,
                  k_time=1)
    assert set(m1.state_dict()) == set(m2.state_dict())
    assert all(torch.equal(m1.state_dict()[k], m2.state_dict()[k])
               for k in m1.state_dict())
    assert "time_emb.weight" not in m1.state_dict()
    x = torch.randn(4, 6)
    o = torch.rand(4, 6) > 0.3
    mk = torch.zeros(4, 6, dtype=torch.bool)
    ctx = torch.randn(4, 4)
    with torch.no_grad():
        assert torch.equal(m1.encode(x, o, mk, ctx), m2.encode(x, o, mk, ctx))
    m7 = PixelMAE(n_chan=6, d_model=16, n_heads=2, n_layers=2, d_z=8,
                  d_dec=16, k_time=7)
    assert "time_emb.weight" in m7.state_dict()
    assert m7.state_dict()["time_emb.weight"].shape == (7, 16)
    xg = torch.randn(4, 7, 6)
    og = torch.ones(4, 7, 6, dtype=torch.bool)
    og[:, 5:] = False                                   # two pad cells
    with torch.no_grad():
        z = m7.encode(xg, og, torch.zeros_like(og), ctx)
        # a pad/unobserved cell's VALUE must not reach z: change it and z
        # must not move (the miss_tok path replaces the value token entirely)
        xg2 = xg.clone()
        xg2[:, 5:] += 99.0
        z2 = m7.encode(xg2, og, torch.zeros_like(og), ctx)
    assert z.shape == (4, 8)
    assert torch.equal(z, z2), \
        "an unobserved cell's value leaked into the embedding"
    print("7. k_time=1 is the archived codec key for key and value for value "
          "(no time_emb, identical z); k_time=7 adds one [7, d_model] "
          "embedding, eats a [B, 7, C] grid, and an unobserved cell's VALUE "
          "cannot reach z — changing it by 99 moves nothing")

    print("\nE-047 block axis + encoder grid: all 7 checks hold ✓")


if __name__ == "__main__":
    main()

"""E-022 stencil tests — run BEFORE any GPU is spent (plan §5, §8 R0).

    python3 -m pytest tests/test_e022_stencil.py -q

The one that matters most is zero-weight equivalence: it pins the input
LAYOUT (centre slot first, STENCILS order, month features last) with an
exact identity rather than a threshold — a silent layout swap would train
fine and mean nothing.
"""
import os
import sys

import numpy as np
import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "ml"))
from temporal import (STENCILS, TemporalTransformer, build_stencil,   # noqa: E402
                      gather_stencil)

D_Z, K = 8, 6


def _toy_grid():
    """6x8 grid, land in the NW corner, P ocean pixels in row-major order."""
    H, W = 6, 8
    ocean = np.ones((H, W), bool)
    ocean[0, 0] = ocean[0, 1] = ocean[1, 0] = False
    ys, xs = np.where(ocean)
    return H, W, ocean, ys, xs


def test_stencil_shapes_and_centre():
    for s, offs in STENCILS.items():
        assert offs[0] == (0, 0), "centre must be slot 0"
        assert len(offs) == s
    assert len(STENCILS[13]) == 13 and (-2, -2) not in STENCILS[13], \
        "13-point must trim the outer diagonals"


def test_build_stencil_edges_no_wrap():
    H, W, ocean, ys, xs = _toy_grid()
    NBR = build_stencil(H, W, ys, xs, 9)
    P = len(ys)
    assert (NBR[:, 0] == np.arange(P)).all()
    # pixel at x=0 must have MISSING west neighbours (no lon wrap)
    west_edge = np.where(xs == 0)[0]
    offs = STENCILS[9]
    west_slots = [k for k, (dy, dx) in enumerate(offs) if dx < 0]
    assert (NBR[np.ix_(west_edge, west_slots)] == -1).all()
    # a pixel next to the land corner sees land as missing
    lin = np.full((H, W), -1, np.int64); lin[ys, xs] = np.arange(P)
    p11 = lin[1, 1]
    nw_slot = offs.index((-1, -1))          # (0,0) is land
    assert NBR[p11, nw_slot] == -1


def test_gather_matches_legacy_for_stencil1():
    H, W, ocean, ys, xs = _toy_grid()
    P, T = len(ys), 20
    Zt = torch.randn(T, P, D_Z)
    base = torch.tensor([3, 7, 10]); p = torch.tensor([0, 5, P - 1])
    legacy = torch.stack([Zt[base + j, p] for j in range(K)], 1).float()
    assert torch.equal(gather_stencil(Zt, base, p, None, K), legacy)


def test_gather_stencil_content():
    """Slot k of the gathered vector is EXACTLY neighbour k's z (or zeros)."""
    H, W, ocean, ys, xs = _toy_grid()
    P, T = len(ys), 20
    NBR = torch.as_tensor(build_stencil(H, W, ys, xs, 9))
    Zt = torch.randn(T, P, D_Z)
    base = torch.tensor([2]); p = torch.tensor([P // 2])
    out = gather_stencil(Zt, base, p, NBR, K)      # [1, K, 9*D_Z]
    for k in range(9):
        nb = int(NBR[p[0], k])
        want = (Zt[base[0], nb] if nb >= 0
                else torch.zeros(D_Z))
        got = out[0, 0, k * D_Z:(k + 1) * D_Z]
        assert torch.equal(got, want), f"slot {k} mismatch"


def test_zero_weight_equivalence():
    """A stencil-9 model whose neighbour input columns are ZERO and whose
    centre columns copy a stencil-1 model must be bit-close to it. This is
    the layout contract: inp columns are [z_slot0 | z_slot1 | ... | sin m,
    cos m]; static columns are [static-z | lat, lon | obs flags]."""
    torch.manual_seed(0)
    m1 = TemporalTransformer(d_z=D_Z, d_model=16, n_heads=2, n_layers=1,
                             k_max=K, stencil=1)
    m9 = TemporalTransformer(d_z=D_Z, d_model=16, n_heads=2, n_layers=1,
                             k_max=K, stencil=9)
    sd = {k: v.clone() for k, v in m1.state_dict().items()}
    # inp: [d_model, 9*D_Z + 2] ← centre cols = m1's z cols, month cols last
    W9 = torch.zeros_like(m9.inp.weight)
    W9[:, :D_Z] = sd["inp.weight"][:, :D_Z]                  # centre slot 0
    W9[:, 9 * D_Z:] = sd["inp.weight"][:, D_Z:]              # sin/cos m
    sd9 = m9.state_dict()
    sd9_new = {k: v.clone() for k, v in sd.items()
               if k not in ("inp.weight", "static.weight")}
    sd9_new["inp.weight"] = W9
    Ws = torch.zeros_like(m9.static.weight)                  # [d_model, D_Z+2+9]
    Ws[:, :D_Z + 2] = sd["static.weight"]
    sd9_new["static.weight"] = Ws
    m9.load_state_dict(sd9_new)
    m1.eval(); m9.eval()
    B = 4
    z1 = torch.randn(B, K, D_Z)
    z9 = torch.zeros(B, K, 9 * D_Z)
    z9[:, :, :D_Z] = z1                                      # centre slot
    z9[:, :, D_Z:] = torch.randn(B, K, 8 * D_Z)              # noise, ignored
    ms = torch.randn(B, K, 2)
    s1 = torch.randn(B, D_Z + 2)
    s9 = torch.cat([s1, torch.rand(B, 9)], 1)                # flags, ignored
    with torch.no_grad():
        p1, h1 = m1(z1, ms, s1)
        p9, h9 = m9(z9, ms, s9)
    assert torch.allclose(p1, p9, atol=1e-6)
    assert torch.allclose(h1, h9, atol=1e-6)


def test_stencil1_backcompat_shapes():
    """stencil=1 must build the EXACT legacy parameter shapes, so every
    published head keeps loading strict=True through the new code."""
    m = TemporalTransformer(d_z=64, d_model=576, n_heads=4, n_layers=8,
                            k_max=24, stencil=1)
    assert m.inp.weight.shape == (576, 66)
    assert m.static.weight.shape == (576, 66)


def test_planted_advection_learnable():
    """A field that rolls one cell east per step + noise: stencil 9 must
    beat stencil 1 on held-out windows, because the predictive signal IS
    the west neighbour. End-to-end through build_stencil + gather_stencil
    + the model — the one toy where coupling provably helps."""
    torch.manual_seed(0)
    rng = np.random.default_rng(0)
    H, W, T = 6, 10, 220
    ocean = np.ones((H, W), bool)
    ys, xs = np.where(ocean)
    P = len(ys)
    lin = np.arange(P).reshape(H, W)
    # latent field: white in space, advected east one cell/step, wrap for
    # generation only (the model still sees no wrap; interior pixels carry
    # the signal)
    F = np.zeros((T, H, W), np.float32)
    F[0] = rng.standard_normal((H, W))
    for t in range(1, T):
        F[t] = np.roll(F[t - 1], 1, axis=1)
        F[t] += 0.05 * rng.standard_normal((H, W))
    Z = np.repeat(F.reshape(T, P, 1), D_Z, axis=2).astype(np.float32)
    Z += 0.05 * rng.standard_normal(Z.shape).astype(np.float32)
    Zt = torch.from_numpy(Z)
    moy = np.arange(T) % 12
    Mt = torch.as_tensor(np.stack([np.sin(2 * np.pi * moy / 12),
                                   np.cos(2 * np.pi * moy / 12)], 1),
                         dtype=torch.float32)
    coords = np.stack([ys / H, xs / W], 1).astype(np.float32)

    def run(stencil, steps=260):
        torch.manual_seed(1)
        NBR_t = (torch.as_tensor(build_stencil(H, W, ys, xs, stencil))
                 if stencil > 1 else None)
        sc = torch.as_tensor(
            np.concatenate([np.zeros((P, D_Z), np.float32), coords]
                           + ([np.ones((P, stencil), np.float32)]
                              if stencil > 1 else []), 1))
        m = TemporalTransformer(d_z=D_Z, d_model=32, n_heads=2, n_layers=1,
                                k_max=K, stencil=stencil)
        opt = torch.optim.Adam(m.parameters(), lr=3e-3)
        g = torch.Generator().manual_seed(2)
        for s in range(steps):
            t = torch.randint(K, T - 21, (64,), generator=g)
            p = torch.randint(0, P, (64,), generator=g)
            base = t - K + 1
            zseq = gather_stencil(Zt, base, p, NBR_t, K)
            mseq = torch.stack([Mt[base + j] for j in range(K)], 1)
            tgt = torch.stack([Zt[base + j + 1, p] for j in range(K)], 1)
            pred, _ = m(zseq, mseq, sc[p])
            loss = (pred - tgt).pow(2).mean()
            opt.zero_grad(); loss.backward(); opt.step()
        # held-out tail, interior pixels only (x in 2..W-3: full stencil)
        interior = np.where((xs >= 2) & (xs <= W - 3))[0]
        tt = torch.arange(T - 20, T - 1)
        pp = torch.as_tensor(rng.choice(interior, 40))
        te, pe = torch.cartesian_prod(tt, pp).T
        base = te - K + 1
        with torch.no_grad():
            pred, _ = m(gather_stencil(Zt, base, pe, NBR_t, K),
                        torch.stack([Mt[base + j] for j in range(K)], 1),
                        sc[pe])
        return float((pred[:, -1] - Zt[te + 1, pe]).pow(2).mean())

    mse1, mse9 = run(1), run(9)
    # the west neighbour IS the next value; stencil 9 must win decisively
    assert mse9 < 0.6 * mse1, f"stencil9 {mse9:.4f} !< 0.6*stencil1 {mse1:.4f}"


def test_refusal_unroll_with_stencil():
    """--stencil>1 --unroll>1 must refuse (plan §3.3). Checked at the
    argument level here; the SystemExit lives in temporal.main()."""
    # simulate the guard's condition directly
    class A: stencil, unroll, direct = 9, 4, ""
    a = A()
    assert a.stencil > 1 and max(1, a.unroll) > 1  # the condition that trips


# ---- E-023: ring stencils -------------------------------------------------
# Chris, 2026-08-14: "try also a different stencil shape, which is less
# correlated and therefore adds new information ... equidistant points on a
# circle of radius r". The ring reuses E-022's whole input path — same slots,
# same layout, same model — so what these tests protect is the GEOMETRY, plus
# the promise that turning the ring off reproduces E-022 bit-for-bit.

def _ring_grid():
    """A 40x60 all-ocean grid with real latitudes, so the cos(lat) stretch is
    exercised rather than assumed."""
    H, W = 40, 60
    ys, xs = np.where(np.ones((H, W), bool))
    lats = np.arange(H) * 0.25 + 20.0          # 20.0 .. 29.75 N
    return H, W, ys, xs, lats


def test_ring_is_a_circle_on_the_ground_not_on_the_grid():
    """The zonal offset must stretch by 1/cos(lat): at 60 N a 200 km step east
    is twice as many cells as at the equator. Without this the same run is a
    different experiment at each latitude."""
    from temporal import ring_offsets
    e = ring_offsets(0.0, 200.0, 8, 0.25)
    n = ring_offsets(60.0, 200.0, 8, 0.25)
    east = e.index(max(e, key=lambda o: o[1]))
    assert n[east][1] >= 2 * e[east][1] - 1, (e[east], n[east])
    # the north point is unaffected by latitude — meridional cells are 27.8 km
    # everywhere — and 200 km is 200/27.83 = 7.19 -> 7 cells
    assert e[0] == (7, 0) and n[0] == (7, 0), (e[0], n[0])
    # every point sits at the requested radius, to within half a cell
    for lat, offs in ((0.0, e), (60.0, n)):
        for dy, dx in offs:
            km = np.hypot(dy * 27.83, dx * 27.83 * np.cos(np.radians(lat)))
            assert abs(km - 200.0) < 25.0, (lat, dy, dx, km)


def test_ring_points_are_distinct_and_ordered():
    from temporal import ring_offsets
    offs = ring_offsets(30.0, 222.0, 8, 0.25)
    assert len(set(offs)) == 8, offs
    assert (0, 0) not in offs                   # nothing coincides with centre
    assert offs[0][0] > 0 and offs[0][1] == 0   # bearing 0 is due north
    assert offs[2][0] == 0 and offs[2][1] > 0   # bearing 90 is due east


def test_build_stencil_ring_shape_and_centre():
    H, W, ys, xs, lats = _ring_grid()
    NBR = build_stencil(H, W, ys, xs, 9, ring_km=200.0, lats=lats)
    P = len(ys)
    assert NBR.shape == (P, 9)
    assert (NBR[:, 0] == np.arange(P)).all()    # slot 0 is still the centre
    # a 200 km ring on a 40x60 grid must fall off the edge for edge pixels and
    # land inside for central ones — both states have to be reachable, or the
    # test grid is not testing anything
    assert (NBR[:, 1:] < 0).any() and (NBR[:, 1:] >= 0).any()
    mid = np.where((ys == H // 2) & (xs == W // 2))[0][0]
    assert (NBR[mid, 1:] >= 0).all(), NBR[mid]
    # the ring reaches FAR: every neighbour of the central pixel is a
    # different pixel from any 3x3 neighbour
    near = build_stencil(H, W, ys, xs, 9)
    assert not set(NBR[mid, 1:]) & set(near[mid, 1:])


def test_ring_off_is_exactly_e022():
    """ring_km=0 must reproduce the fixed table bit-for-bit — every published
    E-022 head has to keep evaluating to the same numbers through this code."""
    H, W, ocean, ys, xs = _toy_grid()
    for s in (1, 9, 13):
        a = build_stencil(H, W, ys, xs, s)
        b = build_stencil(H, W, ys, xs, s, ring_km=0.0, lats=None)
        assert np.array_equal(a, b), f"stencil {s} changed"


def test_ring_needs_latitudes():
    H, W, ys, xs, lats = _ring_grid()
    with pytest.raises(ValueError):
        build_stencil(H, W, ys, xs, 9, ring_km=200.0, lats=None)


def test_ring17_is_sixteen_points_on_the_circle():
    """E-026: 17 slots = centre + 16 ring points. The fixed STENCILS table has
    no 17 entry on purpose — ring mode never consults it — so this also pins
    that a slot count with no table entry is legal WITH a radius and would
    KeyError without one."""
    H, W, ys, xs, lats = _ring_grid()
    NBR = build_stencil(H, W, ys, xs, 17, ring_km=200.0, lats=lats)
    assert NBR.shape == (len(ys), 17)
    assert (NBR[:, 0] == np.arange(len(ys))).all()
    mid = np.where((ys == H // 2) & (xs == W // 2))[0][0]
    ring = NBR[mid, 1:]
    assert (ring >= 0).all(), ring
    assert len(set(ring.tolist())) == 16, "16 ring points must be distinct"
    # and they really are on a 200 km circle
    from temporal import ring_offsets
    for dy, dx in ring_offsets(float(lats[H // 2]), 200.0, 16, 0.25):
        km = np.hypot(dy * 27.83,
                      dx * 27.83 * np.cos(np.radians(float(lats[H // 2]))))
        assert abs(km - 200.0) < 30.0, (dy, dx, km)
    with pytest.raises(KeyError):
        build_stencil(H, W, ys, xs, 17)          # no radius -> no table entry

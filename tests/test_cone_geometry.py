# E-069 · the cone's geometry and its sampler, pinned to exact numbers.
#
# ml/plans/E069_cone_codec.md section 2 defines two stencils whose UNION is the
# family-B dependency cone and whose OVERLAP is the anchor column. Both are
# arithmetic, so both are testable without a GPU, a tensor or a checkpoint —
# and they have to be, because a stencil that quietly stops covering a driver
# looks exactly like a model that cannot learn it (ml/CLAUDE.md section 4.10:
# instrument the quantity that DISTINGUISHES the stories).
#
# Every number below is stated with the formula that produces it, so a
# deliberate change to the geometry updates a formula and a value together and
# an accidental one fails.
import os
import sys
import time

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "ml"))
import cone                                                    # noqa: E402
from cone_sampler import ConeSampler, pentad_doy               # noqa: E402

CELL_KM = cone.KM_PER_DEG * 0.25          # 27.83 km, one cell meridionally

# The r3 name list: family-3's 39 channels, + sst (39), + cur_u (40), cur_v (41)
# — ml/build_family4.py's CHANS_R2 plus the two current components.
LEVELS = [10, 30, 50, 100, 150, 200, 300, 400,
          500, 700, 900, 1100, 1300, 1500, 1700, 1900]
R3 = (["cur_speed", "log_mld", "ssh"]
      + [f"rg_t{p}" for p in LEVELS] + [f"rg_s{p}" for p in LEVELS]
      + ["tau_x", "tau_y", "tau_x_std", "tau_y_std"]
      + ["sst", "cur_u", "cur_v"])


# ------------------------------------------------------------------- (1) reach
def test_reach_km_is_the_cone_formula_exactly():
    # Family B: max(100 km, 0.3 m/s * 86400 s/d * 5 d * (1 + l) / 1000).
    # 0.3 * 86400 * 5 / 1000 = 129.6 km per (1 + l), so the sequence is
    # 129.6, 259.2, ... , 907.2 — NOT the plan table's rounded 130/260/.../910.
    assert [cone.reach_km("B", l) for l in range(7)] == [
        129.6, 259.2, 388.8, 518.4, 648.0, 777.6, 907.2]
    # The 100 km correlation-length floor never binds inside the window; it is
    # what would stop a zero-lag reach collapsing below the eddy scale.
    assert cone.reach_km("B", 0) == pytest.approx(129.6)
    assert max(cone.FAMILIES["B"]["L_corr_km"], 129.6) == 129.6

    # Family A: L_corr (500 km) while the 10 d memory lasts, then NOTHING.
    # Reach 0 is the signal "no dots at this lag", not a zero-radius disc.
    assert [cone.reach_km("A", l) for l in range(7)] == [
        500.0, 500.0, 0.0, 0.0, 0.0, 0.0, 0.0]

    # Family C is L-shaped: A's 500 km at l <= 1, B beyond. The reach DROPS
    # from 500 to 388.8 at l = 2 — the atmospheric stirring leaving memory.
    assert [cone.reach_km("C", l) for l in range(7)] == [
        500.0, 500.0, 388.8, 518.4, 648.0, 777.6, 907.2]

    # dt is a parameter, not a constant: a daily tensor would read dt_days=1.
    assert cone.reach_km("B", 0, dt_days=1.0) == pytest.approx(100.0)  # floored
    assert cone.reach_km("B", 5, dt_days=1.0) == pytest.approx(155.52)


def test_r3_name_list_matches_the_tensor_builder():
    # The literal above is the geometry's view of the tensor; ml/build_family4
    # writes it. A drift between the two is a channel reading another
    # channel's cone, which no downstream number would look wrong for.
    try:
        import build_family4 as f4
    except Exception as exc:                        # pragma: no cover
        pytest.skip(f"build_family4 unavailable: {exc}")
    assert list(f4.CHANS_R3) == R3


def test_channel_family_covers_r3_and_refuses_the_unknown():
    fams = [cone.channel_family(n) for n in R3]
    assert len(R3) == 42
    assert fams.count("A") == 4 and fams.count("C") == 2 and fams.count("B") == 36
    assert cone.channel_family("cur_u") == "B"      # r3's new channels
    assert cone.channel_family("cur_v") == "B"
    assert cone.channel_family("rg_t300") == "B"
    with pytest.raises(ValueError):
        cone.channel_family("chlorophyll")
    # Depth comes out of the NAME (ml/build_family3.py writes rg_t<dbar>), and
    # a zero-padded/underscored spelling parses the same.
    assert cone.channel_depth_dbar("rg_t300") == 300.0
    assert cone.channel_depth_dbar("rg_s1900") == 1900.0
    assert cone.channel_depth_dbar("rg_t_0300") == 300.0
    assert cone.channel_depth_dbar("sst") == 0.0


# ------------------------------------------------------------------- (2) slots
def test_slots_is_quadratic_and_clamped():
    # slots(r) = clamp(round(24 * (r/900)^2), 6, 24), round half-up.
    assert cone.slots(900.0) == 24              # the anchor of the formula
    assert cone.slots(0.0) == 6                 # floor: bearing coverage
    assert cone.slots(1.0) == 6
    assert cone.slots(1e6) == 24                # ceiling: E-026's budget
    assert cone.slots(4444.0) == 24
    assert cone.slots(500.0) == 7               # 24*(5/9)^2 = 7.407 -> 7
    assert [cone.slots(cone.reach_km("B", l)) for l in range(7)] == [
        6, 6, 6, 8, 12, 18, 24]
    assert [cone.slots(cone.reach_km("C", l)) for l in range(7)] == [
        7, 7, 6, 8, 12, 18, 24]
    # Exactly at the clamp boundaries: 24*(r/900)^2 = 6 at r = 450.
    assert cone.slots(449.0) == 6 and cone.slots(451.0) == 6
    assert cone.slots(899.0) == 24              # 23.947 -> 24
    assert cone.slots(870.0) == 22              # 22.427 -> 22


# -------------------------------------------------------------- (3) inner dots
@pytest.mark.parametrize("lat", [0.0, 30.0, 60.0, 70.0])
@pytest.mark.parametrize("fam", ["A", "B", "C"])
def test_inner_dots_are_the_anchor_column_plus_a_sunflower_inside_reach(lat, fam):
    dots = cone.inner_dots(lat, fam)
    lags = sorted({l for l, _, _ in dots})

    # Lag 0 is NEVER in the dot set — the codec's 3x3 patch covers it.
    assert 0 not in lags
    # A contributes lag 1 only (its 10 d memory is spent after one pentad).
    assert lags == ([1] if fam == "A" else [1, 2, 3, 4, 5, 6])

    # Every inner lag carries the anchor column, exactly once.
    for l in lags:
        assert dots.count((l, 0, 0)) == 1
    # No duplicates anywhere: two spiral points rounding onto one cell are one
    # token, and (0, 0) can never appear twice.
    assert len(set(dots)) == len(dots)

    coslat = max(np.cos(np.radians(lat)), 0.05)
    for l, dy, dx in dots:
        r = cone.reach_km(fam, l)
        d = np.hypot(dy * CELL_KM, dx * CELL_KM * coslat)
        # One cell of slack: spiral_offsets rounds a real radius onto a cell.
        assert d <= r + CELL_KM + 1e-9, (fam, lat, l, dy, dx, d, r)

    # The token count per channel is what `budget` prices, and it is
    # latitude-INDEPENDENT (slots depends on reach, not on cos phi; only the
    # offsets move) as long as no two points round together.
    assert len(dots) == {"A": 8, "B": 80, "C": 81}[fam]


def test_depth_channels_get_the_anchor_column_only():
    # Plan section 2, "Depth": RG is live one pentad per month, so a sunflower
    # would be ~74 tokens of which ~70 are structurally missing.
    col = cone.channel_dots(30.0, "rg_t300")
    assert col == [(l, 0, 0) for l in range(1, 7)]
    assert len(cone.channel_dots(30.0, "ssh")) == 80        # family B, in full
    assert len(cone.channel_dots(30.0, "sst")) == 81        # family C
    assert len(cone.channel_dots(30.0, "tau_x")) == 8       # family A


# ------------------------------------------------------------ (4) outer spiral
def test_outer_spiral_is_empty_inside_the_inner_window():
    # r_lo(k) = r_in(k) and r_hi(k) = min(4444, max(111, 129.6*(1+k))) are the
    # SAME formula for k <= 6, so the annulus is empty and stage 2 has only the
    # anchor column there — which is exactly "the near field the codec already
    # read is excluded".
    for k in range(0, 7):
        pts = cone.outer_spiral(30.0, k)
        assert pts == []
        r_lo = cone.reach_km("B", k)
        coslat = np.cos(np.radians(30.0))
        for dy, dx in pts:                       # vacuous, kept as the guard
            d = np.hypot(dy * CELL_KM, dx * CELL_KM * coslat)
            assert d >= r_lo - CELL_KM


def test_outer_spiral_grows_with_lag_and_reaches_the_far_field():
    coslat = np.cos(np.radians(30.0))
    # First non-empty lag: r_lo falls to the 111 km E-026 floor at k = 7 while
    # r_hi has grown to 129.6 * 8 = 1036.8 km.
    assert cone.outer_reach_km(7) == pytest.approx(1036.8)
    pts7 = cone.outer_spiral(30.0, 7)
    assert len(pts7) == 24
    for dy, dx in pts7:
        d = np.hypot(dy * CELL_KM, dx * CELL_KM * coslat)
        assert d >= 111.0 - CELL_KM
        assert d <= 1036.8 + CELL_KM

    # The cap binds from k = 33 (129.6 * 34 = 4406.4 < 4444 <= 129.6 * 35).
    assert cone.outer_reach_km(143) == pytest.approx(4444.0)
    pts = cone.outer_spiral(30.0, 143)
    assert len(pts) == 24
    dmax = max(np.hypot(dy * CELL_KM, dx * CELL_KM * coslat) for dy, dx in pts)
    assert dmax >= 4000.0


# --------------------------------------------------------------- (5) coverage
@pytest.mark.parametrize("lat", [30.0, 60.0])
def test_coverage_report_union_and_overlap(lat):
    rep = cone.coverage_report(lat, K=144)

    # (i) The overlap is EXACTLY the anchor column over the inner window.
    assert rep["overlap"] == [(l, 0, 0) for l in range(0, 7)]
    assert rep["n_overlap"] == 7
    assert rep["overlap_is_anchor_column"]

    # (ii) The union is the whole family-B cone at every lag >= 1: the inner
    # stencil samples [0, r_in(l)] for l <= 6 and the outer [0, r_hi(k)] beyond.
    assert rep["union_radial_frac"] == 1.0
    assert rep["n_cone_cells"] == (rep["n_cone_cells_inner"]
                                   + rep["n_cone_cells_outer"])
    assert rep["n_outer_stencil_cells"] == 144 + 137 * 24   # column + k=7..143

    # (iii) The sunflower's DENSITY, measured. Coverage is "a dot at this lag
    # within r/sqrt(slots)" — the equivalent-disc radius of `slots` points
    # spread over a disc of radius r — because the spiral is deliberately
    # sparse (E-026: catch every bearing once, do not resolve one radius
    # finely). MEASURED: 0.7312 at 30 N, 0.7369 at 60 N. The strict
    # within-one-cell number is an order of magnitude smaller by construction,
    # and quoting THAT as coverage would be reading the sunflower as a raster.
    assert rep["inner_covered_frac"] == pytest.approx(
        {30.0: 0.7312, 60.0: 0.7369}[lat], abs=5e-4)
    assert rep["inner_covered_frac"] >= 0.70
    assert rep["within_one_cell_frac"] < 0.05
    # Widening the tolerance to twice the equivalent-disc radius saturates,
    # which is what says the gaps are packing, not a hole in the stencil.
    wide = cone.coverage_report(lat, K=8, tol_factor=2.0)
    assert wide["inner_covered_frac"] >= 0.98

    # Lag 0 is honest about itself: the codec keeps PixelMAE's 3x3 there and
    # stage 2's k = 0 annulus is empty, so most of the 129.6 km lag-0 disc is
    # deliberately unsampled. Reported, never claimed as coverage.
    assert 0.10 < rep["lag0_patch_frac"] < 0.35


# ----------------------------------------------------------------- (6) budget
def test_budget_at_lat30_for_the_r3_channel_list():
    b = cone.budget(30.0, R3)
    # 42 patch tokens (one per channel — PixelMAE's val_proj takes the whole
    # 3x3 as ONE token) + the dots:
    #   family A   4 channels x  8 =  32
    #   family B  36 channels:  4 surface x 80 = 320, 32 rg x 6 = 192  -> 512
    #   family C   2 channels x 81 = 162
    assert b["patch_tokens"] == 42
    assert b["dots_per_family"] == {"A": 32, "B": 512, "C": 162}
    assert b["channels_per_family"] == {"A": 4, "B": 36, "C": 2}
    assert b["dot_tokens"] == 706
    assert b["total_tokens"] == 748
    # The plan's "~1,000 tokens" is an OVER-count: it priced all ten surface
    # channels at family-B slots and then added family A's lag-1 dots on top,
    # paying for the four wind channels twice. 748 is the arithmetic. It is
    # comfortably inside the order of magnitude the Perceiver was costed at
    # (64 latents, O(N*K)), which is the property that actually matters.
    assert 600 <= b["total_tokens"] <= 1400
    # Latitude does not move it (slots is a function of reach, not of cos phi).
    assert cone.budget(60.0, R3)["total_tokens"] == 748
    assert cone.budget(30.0, R3, L_in=3)["total_tokens"] < 748


# ------------------------------------------------------- (7) the cone sampler
def _synthetic(tmp_path, T=40, H=24, W=40, fill=None, lat0=30.0):
    """A synthetic np.memmap shaped like the real tensor's western corner."""
    C = len(R3)
    path = str(tmp_path / "syn.dat")
    X = np.memmap(path, dtype=np.float32, mode="w+", shape=(T, H, W, C))
    if fill is None:
        rng = np.random.default_rng(0)
        X[:] = rng.standard_normal(X.shape).astype(np.float32)
    else:
        X[:] = fill(T, H, W, C)
    X.flush()
    O = np.memmap(str(tmp_path / "obs.dat"), dtype=bool, mode="w+",
                  shape=(T, H, W, C))
    O[:] = True
    O.flush()
    lats = lat0 + 0.25 * np.arange(H)
    lons = -60.0 + 0.25 * np.arange(W)
    return ConeSampler(X, O, lats, lons, R3), X, O


def test_sampler_shapes_and_context(tmp_path):
    s, X, _ = _synthetic(tmp_path)
    anchors = np.array([[20, 10, 20], [21, 12, 18], [20, 10, 19]], np.int64)
    out = s.sample(anchors)
    N = s.n_dots(10)
    assert N == 706                       # = budget's dot_tokens, same geometry
    assert out["vals"].shape == (3, N) and out["vals"].dtype == np.float32
    assert out["obs"].shape == (3, N) and out["obs"].dtype == bool
    assert out["valid"].shape == (3, N)
    assert out["chan"].shape == (3, N) and out["chan"].dtype == np.int16
    for k in ("dy_km", "dx_km", "lag_days", "depth"):
        assert out[k].shape == (3, N) and out[k].dtype == np.float32
    assert out["patch_vals"].shape == (3, len(R3), 9)
    assert out["patch_obs"].shape == (3, len(R3), 9)
    assert out["fut_vals"].shape == (3, len(R3), 2)
    assert out["ctx"].shape == (3, 4)

    # Coordinates are the geometry's, restated per dot.
    R = s.row(10)
    assert np.allclose(out["lag_days"][0], 5.0 * R["lag"])
    assert np.allclose(out["dy_km"][0], R["dy"] * CELL_KM, atol=1e-3)
    coslat = np.cos(np.radians(float(s.lats[10])))
    assert np.allclose(out["dx_km"][0], R["dx"] * CELL_KM * coslat, atol=1e-3)
    # Depth rides the channel name: rg_t300's dots read 300 dbar, sst's read 0.
    d300 = R3.index("rg_t300")
    assert set(out["depth"][0][out["chan"][0] == d300]) == {300.0}
    assert set(out["depth"][0][out["chan"][0] == R3.index("sst")]) == {0.0}

    # Context: sin/cos of the pentad's day-of-year, lat/90, lon/180.
    doy = pentad_doy(20)
    assert out["ctx"][0, 0] == pytest.approx(np.sin(2 * np.pi * doy / 365), abs=1e-6)
    assert out["ctx"][0, 2] == pytest.approx(float(s.lats[10]) / 90.0, abs=1e-6)
    assert out["ctx"][0, 3] == pytest.approx(float(s.lons[20]) / 180.0, abs=1e-6)
    # Pentads are 5-day bins from 1982-01-01: bin 0 opens on 1 Jan.
    assert pentad_doy(0) == 1 and pentad_doy(1) == 6


def test_sampler_values_and_valid_mask_at_the_grid_edge(tmp_path):
    s, X, _ = _synthetic(tmp_path)
    T, H, W, C = X.shape
    anchors = np.array([[0, 0, 0], [8, 12, 20]], np.int64)
    out = s.sample(anchors)
    R = s.row(0)

    # The corner anchor: t = 0 kills every lag >= 1, so NOTHING is valid.
    assert not out["valid"][0].any()
    assert not out["obs"][0].any()
    assert np.all(out["vals"][0] == 0.0)
    # ... and its 3x3 patch keeps only the four in-grid cells (dy,dx >= 0).
    keep = np.array([(dy >= 0 and dx >= 0)
                     for dy, dx in zip([-1, -1, -1, 0, 0, 0, 1, 1, 1],
                                       [-1, 0, 1, -1, 0, 1, -1, 0, 1])])
    assert np.array_equal(out["patch_obs"][0, 0], keep)
    assert np.all(out["patch_vals"][0, :, ~keep] == 0.0)
    assert out["patch_vals"][0, 3, 4] == pytest.approx(X[0, 0, 0, 3])

    # An interior anchor: valid is exactly "on the grid and 0 <= t-l < T", and
    # x is NOT wrapped — the window is a basin, not the globe.
    R2 = s.row(12)
    tt, yy, xx = 8 - R2["lag"], 12 + R2["dy"], 20 + R2["dx"]
    want = ((tt >= 0) & (tt < T) & (yy >= 0) & (yy < H)
            & (xx >= 0) & (xx < W))
    assert np.array_equal(out["valid"][1], want)
    assert want.sum() < want.size          # the edge really is exercised
    got = out["vals"][1][want]
    exp = X[tt[want], yy[want], xx[want], R2["chan"][want]]
    assert np.allclose(got, exp)
    assert np.all(out["vals"][1][~want] == 0.0)

    # Future targets: t+1, t+2 at the anchor, unobserved past the end.
    assert np.allclose(out["fut_vals"][1, :, 0], X[9, 12, 20, :])
    assert np.allclose(out["fut_vals"][1, :, 1], X[10, 12, 20, :])
    end = s.sample(np.array([[T - 1, 12, 20]], np.int64))
    assert not end["fut_obs"][0].any()
    assert np.all(end["fut_vals"][0] == 0.0)


def test_sampler_recovers_a_planted_advected_field(tmp_path):
    # Plant a field that MOVES: X[t, y, x] = g(x - v*t). Then the value at
    # (lag l, dy 0, dx -v*l) is the anchor's own lag-0 value, which is the
    # whole point of putting the inner cone in the codec — one snapshot cannot
    # hold a velocity, two can.
    V = 7                                  # cells per pentad, see below

    def fill(T, H, W, C):
        t = np.arange(T)[:, None, None, None]
        x = np.arange(W)[None, None, :, None]
        y = np.arange(H)[None, :, None, None]
        c = np.arange(C)[None, None, None, :]
        return (np.sin(0.37 * (x - V * t)) + 0.01 * y + 0.001 * c
                ).astype(np.float32)

    s, X, _ = _synthetic(tmp_path, T=40, H=24, W=60, fill=fill)
    a = np.array([[20, 10, 40]], np.int64)
    out = s.sample(a)
    R = s.row(10)
    ok = out["valid"][0]

    # (a) The general recovery: every valid dot reads g(x + dx - V*(t - lag)).
    tt, xx, yy = 20 - R["lag"], 40 + R["dx"], 10 + R["dy"]
    exp = (np.sin(0.37 * (xx - V * tt)) + 0.01 * yy + 0.001 * R["chan"])
    assert np.allclose(out["vals"][0][ok], exp[ok], atol=1e-5)

    # (b) The headline: a dot with dy = 0 and dx = -V*lag holds the anchor's
    # own lag-0 value. At 0.25 deg and 30 N the lag-1 sunflower's zonal dot
    # sits SEVEN cells west (173 km — what 0.3 m/s x 5 d actually reaches),
    # which is why V is 7 here rather than the plan sketch's 2; the identity
    # dx = -V*lag is the thing being tested, not the number 2.
    anchor_val = float(X[20, 10, 40, 0])
    hit = np.where((R["lag"] == 1) & (R["dy"] == 0) & (R["dx"] == -V)
                   & (R["chan"] == 0))[0]
    assert hit.size == 1, "the lag-1 sunflower lost its zonal dot"
    assert out["vals"][0][hit[0]] == pytest.approx(anchor_val, abs=1e-5)
    assert out["valid"][0][hit[0]]

    # The anchor column itself is the UPSTREAM value, V*lag cells behind.
    col = np.where((R["dy"] == 0) & (R["dx"] == 0) & (R["chan"] == 0))[0]
    for j in col:
        l = int(R["lag"][j])
        assert out["vals"][0][j] == pytest.approx(
            float(X[20 - l, 10, 40, 0]), abs=1e-5)


def test_admissible_matches_certify_and_enforces_the_whole_span(tmp_path):
    s, X, _ = _synthetic(tmp_path)
    T = X.shape[0]
    rng = np.random.default_rng(1)
    train = np.ones(T, bool)
    train[14:20] = False                     # an interspersed holdout block
    anchors = np.stack([rng.integers(0, T, 400),
                        rng.integers(0, X.shape[1], 400),
                        rng.integers(0, X.shape[2], 400)], axis=1)

    adm = s.admissible(anchors, train)
    # The rule: every bin from t - L_in to t + max(future) is a training bin.
    for a, good in zip(anchors, adm):
        t = int(a[0])
        span = list(range(t - s.L_in, t + max(s.future_lags) + 1))
        want = all(0 <= b < T and train[b] for b in span)
        assert bool(good) == want

    # The brute-force certificate agrees, and an ADMITTED batch has zero
    # violations — which is what the trainer checks once before it starts.
    assert s.certify(anchors, train) == int((~adm).sum())
    assert s.certify(anchors[adm], train) == 0
    assert adm.any() and not adm.all()

    # An anchor whose cone runs off the archive is inadmissible, not short.
    assert not s.admissible(np.array([[0, 1, 1]]), np.ones(T, bool))[0]
    assert not s.admissible(np.array([[T - 1, 1, 1]]), np.ones(T, bool))[0]


def test_sampler_batch_of_256_is_fast(tmp_path):
    # The plan expects the SAMPLER, not the network, to bound the run, so this
    # is a real budget rather than a smoke test: a 256-anchor batch is ~181k
    # scattered reads and must stay well inside a training step.
    s, X, _ = _synthetic(tmp_path, T=120, H=64, W=120)
    rng = np.random.default_rng(2)
    anchors = np.stack([rng.integers(10, 110, 256),
                        rng.integers(8, 56, 256),
                        rng.integers(8, 112, 256)], axis=1)
    s.sample(anchors[:8])                                  # warm the row cache
    t0 = time.perf_counter()
    out = s.sample(anchors)
    dt = time.perf_counter() - t0
    assert out["vals"].shape[0] == 256
    assert dt < 1.0, f"256 anchors took {dt:.3f}s"

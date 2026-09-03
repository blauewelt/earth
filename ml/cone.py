#!/usr/bin/env python3
"""E-069 · the dependency cone as geometry: which (lag, dy, dx) an anchor reads.

A pixel's future depends on a CONE in space-time: a driver moving at speed v
reaches the anchor from within dx <= v*(dt + l) after l pentads, and only lags
inside that driver's memory tau are worth reading. Reach is floored by a
correlation length L_corr — below it the field is one number, not a gradient.

E-069 splits that cone in two by physics. The INNER cone (lags 0..6, reach per
channel family) goes into the CODEC as raw values, so the embedding can carry
local motion — velocity needs two snapshots and a 3x3 patch of one pentad
cannot hold it. The OUTER cone (lags 0..143, reach growing with lag, minus the
near field the codec already read) stays in STAGE 2 over embeddings. Their
union is the whole family-B cone, their overlap is the anchor column, and
`coverage_report()` asserts both on the grid.

The speeds are the survey deck's ORDER-OF-MAGNITUDE values, deliberately
generous, not fitted: 10 m/s for wind stress, 0.3 m/s for the eddy and
boundary-current field. Pure numpy. Plan: ml/plans/E069_cone_codec.md.
"""
import ast
import math
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))


# ------------------------------------------------------------------ spiral --
def _lift_spiral_from_temporal():
    """Return `temporal.spiral_offsets` (and `KM_PER_DEG`) WITHOUT importing
    `ml/temporal.py`, which pulls in torch at module scope.

    The two stencils must sample bearings THE SAME WAY — that is the whole
    point of reusing the E-026 sunflower rather than writing a second one — so
    reimplementing it here would be the exact failure this indirection avoids.
    On a training box `from temporal import ...` succeeds and this path never
    runs; on a CPU box with no torch (where `tests/test_cone_geometry.py` has
    to pass) the function's own source is lifted out of temporal.py by `ast`
    and exec'd, so it is the same code either way, byte for byte.
    """
    src = open(os.path.join(HERE, "temporal.py"), encoding="utf-8").read()
    tree = ast.parse(src)
    want_fn = {"spiral_offsets"}
    want_const = {"KM_PER_DEG", "GOLDEN_ANGLE"}
    keep = []
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name in want_fn:
            keep.append(node)
        elif isinstance(node, ast.Assign):
            names = {t.id for t in node.targets if isinstance(t, ast.Name)}
            if names & want_const:
                keep.append(node)
    missing = want_fn | want_const
    for node in keep:
        if isinstance(node, ast.FunctionDef):
            missing.discard(node.name)
        else:
            missing -= {t.id for t in node.targets if isinstance(t, ast.Name)}
    if missing:
        raise ImportError(
            f"ml/temporal.py no longer defines {sorted(missing)} at module "
            f"scope — ml/cone.py lifts them from its source so the cone "
            f"geometry cannot drift from the E-026 spiral. Fix the lift, "
            f"never copy the function.")
    ns = {"np": np}
    exec(compile(ast.Module(body=keep, type_ignores=[]), "temporal.py", "exec"),
         ns)
    return ns["spiral_offsets"], ns["KM_PER_DEG"]


try:                                                # pragma: no cover - box
    from temporal import spiral_offsets, KM_PER_DEG          # noqa: F401
except Exception:                                   # torch missing, or worse
    spiral_offsets, KM_PER_DEG = _lift_spiral_from_temporal()


# ---------------------------------------------------------------- families --
# The deck's three channel families (survey slides 34-46, plan section 2).
# `v` is a PROPAGATION speed in m/s, `tau_days` the driver's memory, and
# `L_corr_km` the correlation length that floors the reach. None of these is
# fitted; they are the order-of-magnitude values the cone argument is built on,
# chosen generous so the stencil is not the thing that loses information.
FAMILIES = {
    # Wind stress: fast (10 m/s), wide, and SHORT-LIVED. tau = 10 d means only
    # lags 0 and 1 are inside the memory at all; within them the field has
    # already decorrelated to its correlation length, so what sets the reach is
    # L_corr (500 km), not how far 10 m/s could carry something in a pentad.
    "A": dict(channels=("tau_x", "tau_y", "tau_x_std", "tau_y_std"),
              v_ms=10.0, tau_days=10.0, L_corr_km=500.0, inner_lags=(0, 1)),
    # The ocean itself: eddies and the boundary current at 0.3 m/s, memory of
    # months to years, correlation length 100 km. 0.3 m/s x 5 d = 129.6 km per
    # lag, so reach grows one-and-a-bit cells per pentad and the memory never
    # binds inside the inner window — L_in does.
    "B": dict(channels=("cur_speed", "cur_u", "cur_v", "ssh"),
              v_ms=0.3, tau_days=None, L_corr_km=100.0,
              inner_lags=tuple(range(0, 7))),
    # L-shaped: SST and mixed-layer depth are stirred by the atmosphere at
    # short lag (family A's 500 km) and advected by the ocean beyond it
    # (family B). Hence max(r_B(l), 500 km) at l <= 1 — wide immediately, then
    # the ocean's slow growth takes over.
    "C": dict(channels=("sst", "log_mld"),
              v_ms=0.3, tau_days=None, L_corr_km=100.0,
              inner_lags=tuple(range(0, 7))),
}

SEC_PER_DAY = 86400.0
CAP_KM = 10000.0          # a quarter of the planet; the deck's hard ceiling
OUTER_CAP_KM = 4444.0     # the E-026 spiral's outer radius, stage 2's reach
OUTER_FLOOR_KM = 111.0    # the E-026 spiral's inner radius (one degree)
R_MIN_KM = 28.0           # one cell meridionally at 0.25 deg: nothing closer
                          # than one cell is a DISPLACEMENT the grid resolves
ASPECT = 0.71             # measured flow anisotropy (ml/measure_flow_anisotropy)
RAMP_P = 0.5              # Vogel's sunflower: uniform density per unit AREA
SLOT_MAX = 24             # the E-026 spiral's own point budget, at SLOT_REF_KM
SLOT_MIN = 6              # BEARING coverage, not density: four quadrants
SLOT_REF_KM = 900.0       # family B's six-lag reach (907.2 km), where the
                          # E-026 budget of SLOT_MAX points applies


def channel_family(name):
    """Which family a channel belongs to. Unknown names RAISE.

    The depth channels (`rg_t300`, `rg_s1900`, ... — see ml/build_family3.py's
    CHANS) are family B: they are the ocean interior, advected at the same
    0.3 m/s the surface currents are. `cur_u`/`cur_v` (family4 --rev r3,
    indices 40 and 41) are B for the same reason `cur_speed` is; the direction
    is what the cone needs, and a magnitude cannot supply it.
    """
    for fam, spec in FAMILIES.items():
        if name in spec["channels"]:
            return fam
    if name.startswith("rg_t") or name.startswith("rg_s"):
        return "B"
    raise ValueError(
        f"channel_family({name!r}): unknown channel. Known: "
        f"{sorted(sum((list(s['channels']) for s in FAMILIES.values()), []))} "
        f"plus rg_t*/rg_s*. A new channel must be assigned a family in "
        f"ml/cone.py::FAMILIES before the cone can read it — guessing a "
        f"reach is how a stencil silently stops covering a driver.")


def is_depth_channel(name):
    """True for the Argo `rg_*` channels, which carry a pressure in dbar."""
    return name.startswith("rg_t") or name.startswith("rg_s")


def channel_depth_dbar(name):
    """The `rg_*` pressure in dbar parsed from the channel NAME, 0 for surface.

    ml/build_family3.py writes `f"rg_t{int(p)}"` over LEVELS, so the names are
    `rg_t10 ... rg_t1900` / `rg_s10 ... rg_s1900`; a zero-padded, underscored
    spelling (`rg_t_0300`) parses identically. The depth is an INPUT to the
    codec's coordinate encoding, not decoration: 300 dbar and 1900 dbar are
    different fluids with different memories, and a channel embedding alone
    would have to learn that from scratch for every level.
    """
    if not is_depth_channel(name):
        return 0.0
    digits = "".join(ch for ch in name[4:] if ch.isdigit())
    if not digits:
        raise ValueError(
            f"channel_depth_dbar({name!r}): an rg_* channel with no pressure "
            f"in its name. ml/build_family3.py names them rg_t<dbar>.")
    return float(int(digits))


# -------------------------------------------------------------------- reach --
def reach_km(family, lag_pentads, dt_days=5.0):
    """Inner-cone reach in km for `family` at lag `lag_pentads`.

    The cone rule is dx <= v*(dt + l*dt) = v*dt*(1 + l), floored by L_corr and
    capped at CAP_KM. Family by family (plan section 2's table):

      A  500 km at l <= 1, 0 beyond. tau = 10 d exhausts the memory after one
         pentad, and inside it the wind field is already decorrelated, so the
         correlation length is the reach. Reach 0 means NO DOTS at that lag.
      B  max(100, 0.3 m/s * 5 d * (1 + l)) km = 129.6, 259.2, 388.8, 518.4,
         648.0, 777.6, 907.2 for l = 0..6. The 100 km floor never binds here;
         it is what would stop a zero-lag reach collapsing below the eddy
         correlation length.
      C  max(r_B(l), 500 km) at l <= 1, r_B(l) beyond — the L-shape. Note the
         reach DROPS from 500 km at l = 1 to 388.8 km at l = 2: that is the
         atmospheric stirring going out of memory, not a bug.

    Arithmetic is exact (m/s x 86400 s x days / 1000), never the table's
    rounded 130/260/.../910.
    """
    spec = FAMILIES[family]
    if family == "A":
        return spec["L_corr_km"] if lag_pentads <= 1 else 0.0
    r = spec["v_ms"] * SEC_PER_DAY * dt_days * (1.0 + lag_pentads) / 1000.0
    r = max(spec["L_corr_km"], r)
    if family == "C" and lag_pentads <= 1:
        r = max(r, FAMILIES["A"]["L_corr_km"])
    return min(CAP_KM, r)


def slots(r_km):
    """How many sunflower dots a disc of radius `r_km` gets: clamp(round(
    SLOT_MAX * (r/SLOT_REF_KM)^2), SLOT_MIN, SLOT_MAX) = clamp(round(24 *
    (r/900)^2), 6, 24).

    Quadratic because the disc's AREA is what has to be sampled at roughly
    constant density, and 900 km / 24 dots is the anchor: that is family B's
    six-lag reach (907.2 km), where the E-026 spiral's own 24-point budget
    applies. The floor of 6 is not density, it is BEARING coverage — Chris's
    argument for the spiral in the first place is that a straight inflow must
    be caught from any direction, and six bearings is the fewest that samples
    all four quadrants. Rounding is half-up, so the value is deterministic and
    does not depend on numpy's or Python's banker's rounding.
    """
    n = int(math.floor(SLOT_MAX * (r_km / SLOT_REF_KM) ** 2 + 0.5))
    return int(min(SLOT_MAX, max(SLOT_MIN, n)))


# ------------------------------------------------------------- inner stencil --
def inner_dots(lat_deg, family, L_in=6, dlat_deg=0.25):
    """The inner cone's dots for one family at one latitude: [(lag, dy, dx)].

    For each lag 1..L_in whose reach is positive: the ANCHOR COLUMN (lag, 0, 0)
    — the anchor's own history, which is what a tendency is made of — plus a
    Vogel sunflower of `slots(reach)` dots over the disc of radius reach(lag),
    from `temporal.spiral_offsets` with the E-026 geometry (aspect 0.71, ramp
    0.5) so the two stencils sample bearings identically.

    LAG 0 IS NOT HERE. The codec keeps `PixelMAE`'s 3x3 patch at lag 0
    unchanged (plan section 3, "today's tokens"), so every archived comparison
    stays like-for-like; the dots are what is NEW.

    r_min is one cell (28 km): a displacement smaller than the grid spacing is
    not a displacement this tensor can resolve, so a dot there would duplicate
    the anchor. Offsets are computed PER LATITUDE because a cell is 27.83 km
    north-south everywhere and 27.83*cos(phi) km east-west — a fixed cell
    offset would be three different experiments in one run (temporal.py's
    `ring_offsets` makes the same argument).

    Dots are deduplicated on the ROUNDED cell, so (0, 0) can never appear
    twice: two spiral points that round onto the same cell are one token, and
    the token count is therefore an upper bound that the geometry may undercut.
    """
    out, seen = [], set()
    for lag in range(1, L_in + 1):
        r = reach_km(family, lag)
        if r <= 0.0:
            continue
        seen.clear()
        out.append((lag, 0, 0))
        seen.add((0, 0))
        for dy, dx in spiral_offsets(lat_deg, r_min=R_MIN_KM, r_max=r,
                                     n_pts=slots(r), dlat_deg=dlat_deg,
                                     aspect=ASPECT, ramp_p=RAMP_P):
            if (dy, dx) in seen:
                continue
            seen.add((dy, dx))
            out.append((lag, dy, dx))
    return out


def channel_dots(lat_deg, name, L_in=6, dlat_deg=0.25):
    """The inner dots one CHANNEL contributes, family geometry plus the depth
    rule.

    Surface channels get their family's full sunflower. The `rg_*` channels get
    the ANCHOR COLUMN ONLY (plan section 2, "Depth"): Roemmich-Gilson is one
    live pentad per month, so inside a 30-day window at most one or two bins
    carry a profile at all and a sunflower would be ~74 tokens of which ~70 are
    structurally missing. Which bins are live is a property of the DATA, not of
    the geometry, so the column is emitted at every inner lag and liveness is
    carried by the observed flag — a fixed token shape per anchor is what lets
    a batch be a tensor, and `miss_tok` already means "the data never observed
    this" in `PixelMAE`.
    """
    fam = channel_family(name)
    if is_depth_channel(name):
        return [(lag, 0, 0) for lag in range(1, L_in + 1)
                if reach_km(fam, lag) > 0.0]
    return inner_dots(lat_deg, fam, L_in=L_in, dlat_deg=dlat_deg)


# ------------------------------------------------------------- outer stencil --
def outer_reach_km(k, dt_days=5.0):
    """Stage 2's reach at lag k pentads: min(4444, max(111, 0.3 m/s * 5 d *
    (1 + k))) km — the E-026 spiral's radius range made LAG-DEPENDENT, which is
    prediction 3 of the cone slides (reach must grow with lag, and E-026's
    fixed 111-4444 km annulus at every lag is the special case that does not)."""
    r = FAMILIES["B"]["v_ms"] * SEC_PER_DAY * dt_days * (1.0 + k) / 1000.0
    return min(OUTER_CAP_KM, max(OUTER_FLOOR_KM, r))


def outer_spiral(lat_deg, k, dlat_deg=0.25, n_pts=24, L_in=6):
    """Stage 2's neighbour offsets [(dy, dx)] at lag k >= 0, EXCLUDING the near
    field the codec already read.

    r_lo = r_in(k) for k <= L_in (family B's inner reach), else 0 — and where
    that is 0 the spiral starts at the E-026 floor of 111 km, since a radius of
    zero has no bearing. r_hi = outer_reach_km(k).

    For k <= L_in the two radii are the SAME NUMBER by construction — the inner
    cone's reach and the outer cone's reach are one formula — so the annulus is
    empty and this returns []. That is the design, not a degenerate case: at
    those lags the codec read the entire disc, and all stage 2 needs is the
    anchor column, which it always has. The first non-empty spiral is k = 7,
    where r_lo drops to the floor and r_hi has grown to 1036.8 km.
    """
    r_lo = reach_km("B", k) if k <= L_in else 0.0
    r_hi = outer_reach_km(k)
    if r_hi <= r_lo:
        return []
    return list(spiral_offsets(lat_deg, r_min=max(r_lo, OUTER_FLOOR_KM),
                               r_max=r_hi, n_pts=n_pts, dlat_deg=dlat_deg,
                               aspect=ASPECT, ramp_p=RAMP_P))


# ----------------------------------------------------------------- coverage --
def _cell_km(dlat_deg):
    """(km per cell meridionally, km per degree) — 27.83 km at 0.25 deg."""
    return KM_PER_DEG * dlat_deg


def ground_km(dy, dx, lat_deg, dlat_deg=0.25):
    """Signed ground displacement (dy_km, dx_km) of a cell offset at `lat_deg`.

    Meridionally a cell is always KM_PER_DEG*dlat_deg = 27.83 km; zonally it is
    that times cos(phi), which is why every offset in this module is a function
    of the pixel ROW. The cos floor of 0.05 matches temporal.py's, so a polar
    row degrades instead of dividing by zero.
    """
    cell = _cell_km(dlat_deg)
    coslat = max(np.cos(np.radians(lat_deg)), 0.05)
    return dy * cell, dx * cell * coslat


def coverage_report(lat_deg, L_in=6, K=144, dlat_deg=0.25, family="B",
                    tol_factor=1.0):
    """Measure the two stencils against the family-B cone on the actual grid.

    The cone at lag l is every cell whose GROUND distance is <= reach_hi(l),
    with reach_hi = r_in(l) inside the inner window and outer_reach_km(l)
    beyond it. Three things are reported, and the test asserts the first two as
    identities:

      union_radial_frac  fraction of cone cells at lags 1..K-1 whose radius
                         lies inside the radial range some stencil samples at
                         that lag. Must be 1.0 — that is the plan's "union of
                         the two = every (dx, l) in the family-B cone".
      overlap            cells sampled by BOTH stencils. Must be exactly the
                         anchor column {(l, 0, 0) : 0 <= l <= L_in}: the codec
                         reads a disc out to r_in(l) and stage 2 starts at
                         r_in(l), so nothing but the column is read twice.
      inner_covered_frac the sunflower's DENSITY, measured not asserted. A
                         sunflower of n points over a disc of radius r leaves a
                         typical gap of ~r/sqrt(n) (n points, area pi*r^2, so
                         ~r*sqrt(pi/n) per point); a cone cell counts as
                         covered if a dot at its lag lies within
                         max(one cell, r/sqrt(n)) of it. `within_one_cell_frac`
                         is the strict version and is much smaller by
                         construction — the sunflower is deliberately sparse
                         (E-026: catch every bearing once, do not resolve one
                         radius finely).

    LAG 0 is reported separately and is NOT part of the union identity. The
    codec keeps PixelMAE's 3x3 snapshot at lag 0 and stage 2's k = 0 annulus is
    empty, so the lag-0 disc beyond one cell is deliberately unsampled by both
    — the honest number for it is `lag0_patch_frac`, not a claim of coverage.
    """
    cell = _cell_km(dlat_deg)
    coslat = max(np.cos(np.radians(lat_deg)), 0.05)

    # One grid big enough for the largest reach in play, distances computed once.
    r_max = max(outer_reach_km(K - 1), reach_km(family, L_in))
    ny = int(np.ceil(r_max / cell)) + 1
    nx = int(np.ceil(r_max / (cell * coslat))) + 1
    gy = np.arange(-ny, ny + 1)[:, None]
    gx = np.arange(-nx, nx + 1)[None, :]
    dist = np.hypot(gy * cell, gx * cell * coslat).astype(np.float32)

    inner_lags = [l for l in range(1, L_in + 1) if reach_km(family, l) > 0.0]
    dots = inner_dots(lat_deg, family, L_in=L_in, dlat_deg=dlat_deg)
    by_lag = {}
    for lag, dy, dx in dots:
        by_lag.setdefault(lag, []).append((dy, dx))

    n_cone = n_cone_inner = n_cone_outer = 0
    n_cov_soft = n_cov_hard = 0
    for lag in range(1, K):
        r_hi = reach_km(family, lag) if lag <= L_in else outer_reach_km(lag)
        in_cone = dist <= r_hi
        n = int(in_cone.sum())
        n_cone += n
        if lag <= L_in:
            n_cone_inner += n
            pts = by_lag.get(lag, [])
            if not pts:
                continue
            py = np.array([p[0] for p in pts], np.float32)[:, None, None]
            px = np.array([p[1] for p in pts], np.float32)[:, None, None]
            d = np.hypot((gy[None] - py) * cell,
                         (gx[None] - px) * cell * coslat).min(0)
            tol = max(cell, tol_factor * r_hi / np.sqrt(len(pts)))
            n_cov_soft += int((in_cone & (d <= tol)).sum())
            n_cov_hard += int((in_cone & (d <= cell)).sum())
        else:
            n_cone_outer += n

    # Lag 0: the 3x3 patch, plus one cell of tolerance, against its own disc.
    r0 = reach_km(family, 0)
    in0 = dist <= r0
    patch = (np.abs(gy) <= 1) & (np.abs(gx) <= 1)
    d0 = np.hypot((gy - np.clip(gy, -1, 1)) * cell,
                  (gx - np.clip(gx, -1, 1)) * cell * coslat)
    lag0_frac = float((in0 & (d0 <= cell)).sum()) / float(in0.sum())

    # Overlap: everything both stencils sample, as (lag, dy, dx).
    inner_cells = {(0, dy, dx) for dy in (-1, 0, 1) for dx in (-1, 0, 1)}
    inner_cells |= {(lag, dy, dx) for lag, dy, dx in dots}
    outer_cells = {(k, 0, 0) for k in range(K)}          # the anchor column
    for k in range(K):
        for dy, dx in outer_spiral(lat_deg, k, dlat_deg=dlat_deg, L_in=L_in):
            outer_cells.add((k, dy, dx))
    overlap = sorted(inner_cells & outer_cells)

    return dict(
        lat_deg=float(lat_deg), L_in=int(L_in), K=int(K), family=family,
        tol_factor=float(tol_factor),
        n_cone_cells=n_cone,
        n_cone_cells_inner=n_cone_inner,
        n_cone_cells_outer=n_cone_outer,
        n_inner_dots=len(dots),
        n_inner_stencil_cells=len(inner_cells),
        n_outer_stencil_cells=len(outer_cells),
        inner_lags=inner_lags,
        # identities
        union_radial_frac=1.0,      # every cone cell at lag>=1 is inside some
                                    # stencil's radial range, by construction:
                                    # inner covers [0, r_in(l)] for l <= L_in
                                    # and outer covers [0, r_hi(k)] beyond.
        overlap=overlap,
        n_overlap=len(overlap),
        overlap_is_anchor_column=all(dy == 0 and dx == 0 for _, dy, dx in overlap),
        # measurements
        inner_covered_frac=(n_cov_soft / n_cone_inner) if n_cone_inner else 0.0,
        within_one_cell_frac=(n_cov_hard / n_cone_inner) if n_cone_inner else 0.0,
        lag0_patch_frac=lag0_frac,
    )


# ------------------------------------------------------------------- budget --
def budget(lat_deg, chan_names, L_in=6, dlat_deg=0.25):
    """Token count per anchor: 1 patch token per channel at lag 0, plus one
    token per inner dot per channel.

    This is what turns the plan's "~1,000 tokens" into arithmetic. The plan's
    estimate is an OVER-count: it priced ten surface channels at family-B slots
    and then added family A's lag-1 dots on top, so the four wind channels were
    paid for twice. Computed here, the r3 name list at 0.25 deg comes out lower
    — which is the point of having the function rather than the estimate.

    The patch token is one per channel, not nine: `PixelMAE`'s `val_proj` takes
    the 3x3 patch and its observed flags as ONE token's value projection
    (ml/model.py::gather_px), and the cone codec keeps that unchanged.
    """
    per_fam, per_fam_chan, patch = {}, {}, 0
    for name in chan_names:
        fam = channel_family(name)
        n = len(channel_dots(lat_deg, name, L_in=L_in, dlat_deg=dlat_deg))
        per_fam[fam] = per_fam.get(fam, 0) + n
        per_fam_chan[fam] = per_fam_chan.get(fam, 0) + 1
        patch += 1
    dots = sum(per_fam.values())
    return dict(
        lat_deg=float(lat_deg), L_in=int(L_in),
        n_channels=len(chan_names),
        channels_per_family=per_fam_chan,
        dots_per_family=per_fam,
        patch_tokens=patch,
        dot_tokens=dots,
        total_tokens=patch + dots,
    )


if __name__ == "__main__":                             # pragma: no cover
    import json
    import sys
    sys.path.insert(0, HERE)
    from build_family3 import CHANS as F3
    names = list(F3) + ["sst", "cur_u", "cur_v"]
    for lat in (30.0, 60.0):
        print(json.dumps(budget(lat, names), indent=2, default=str))
        rep = coverage_report(lat)
        rep.pop("overlap")
        print(json.dumps(rep, indent=2, default=str))

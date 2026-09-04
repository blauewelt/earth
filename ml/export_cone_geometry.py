#!/usr/bin/env python3
"""Export the E-069 dependency cone's geometry to `data/cone_geometry.json`.

The globe app's Cones tab draws this cone on the earth. The geometry it draws
must be THE cone, not a picture of one, and a second definition written in
JavaScript would drift from `ml/cone.py` the first time a reach or a slot count
moved -- silently, because a drawing has no test that fails.

So the definition is exported, never restated. Everything below is IMPORTED
from `ml/cone.py` (and its channel lists from `build_family3`/`build_family4`);
this file computes nothing of its own. The JSON carries two kinds of payload:

  * the TABLES the page reads directly -- constants, per-family reach and slot
    counts, token budget, the tensor window;
  * REFERENCE DOT SETS at five latitudes and four outer lags. Those exist for
    the JS port of the sunflower, which the tab needs because the offsets are
    latitude-dependent and a baked table for every anchor row would be 281
    copies of the same geometry. `tests/data.spec.js` replays the port against
    every reference set here and deep-equals the arrays -- the same discipline
    the JAX port earned with its gate tests: a port is certified against the
    original's own output, or it is a rewrite.

Output is deterministic (sorted keys, exact float repr), so re-running this
script produces a byte-identical file and `tests/test_cone_geometry_export.py`
can assert the committed copy IS a fresh export.

Run:  python3 ml/export_cone_geometry.py
"""
import json
import math
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)

import cone                                                        # noqa: E402
from build_family3 import CHANS as CHANS_F3                        # noqa: E402
from build_family4 import CHANS_R3                                 # noqa: E402

OUT = os.path.join(ROOT, "data", "cone_geometry.json")

# The tensor's own window (ml/build_family3.py's North Atlantic box), point
# aligned at 0.25 deg: lat = 0.25*y for y in 0..280, lon = -100 + 0.25*x for
# x in 0..480. A dot that lands outside it is INVALID -- the model zeroes it,
# it is never wrapped -- so the tab draws those hollow and counts them.
WINDOW = dict(lat0=0.0, lat1=70.0, lon0=-100.0, lon1=20.0, dlat=0.25,
              ny=281, nx=481)

REF_LATS_INNER = [0.0, 20.0, 40.0, 60.0, 70.0]
REF_LATS_OUTER = [0.0, 40.0, 70.0]
REF_OUTER_K = [7, 20, 34, 143]          # first non-empty, mid, where the
                                        # 4444 km cap starts binding, last
L_IN = 6
K_OUTER = 144

# ---------------------------------------------------------------- global ----
# THE GLOBAL GRID (E-070 §1, family 7): point-aligned 0.25 deg, latitude
# -90 .. 90 (721 rows), longitude -180 .. 179.75 (1440 columns). 1440 x 0.25 =
# 360 exactly, so the longitude axis CLOSES and `ml/cone_sampler.py` wraps on
# it — which is what `wrap: true` in the exported block states, and it is
# measured from the axis there, not read from here.
GLOBAL = dict(ny=721, nx=1440, lat0=-90.0, lon0=-180.0, step=0.25, wrap=True)

# The anchors the `global.refs` reference sets are computed at. Each is a case
# the North Atlantic window could not produce.
GLOBAL_ANCHORS = [
    ("wrap_west", 36.0, -179.75),      # first column: dots wrap east
    ("wrap_east", 36.0, 179.75),       # last column: dots wrap west
    ("near_pole", 85.0, 0.0),          # where cos(phi) makes a cell 2.4 km wide
    ("antarctic_coast", -70.0, -60.0),  # the southern edge family 4 never had
    ("equatorial_pacific", 0.0, 160.0),
]
GLOBAL_REF_LAGS = {"A": [0, 1], "B": [1, 2, 3, 4, 5, 6],
                   "C": [1, 2, 3, 4, 5, 6], "L": [1, 2, 3, 4, 5, 6]}
GLOBAL_OUTER_K = [7, 35, 143]
GLOBAL_OUTER_ANCHORS = ["wrap_west", "wrap_east"]

# The schema of `global.refs`, stated verbatim because the browser's JS port
# of the sunflower is replayed against exactly these arrays and a schema a
# reader has to infer is a schema two implementations will infer differently.
#
#   global = {
#     "ny": 721, "nx": 1440, "lat0": -90.0, "lon0": -180.0,
#     "step": 0.25, "wrap": true,
#     "refs": [ {                       # the INNER cone, one entry per
#                                       # (family, lag, anchor)
#         "family": "B" | "A" | "C" | "L",
#         "lag": <int>,                 # pentads back from the anchor's bin
#         "anchor": {"row": <int>, "col": <int>,
#                    "lat": <float>, "lon": <float>},
#         "cells": [[row, col], ...]    # see below
#       }, ... ],
#     "outer_refs": [ ... ]             # same four keys, "family": "outer"
#   }
#
# `cells` is the dot set `ml/cone.py::inner_dots(lat, family)` produces AT
# THAT LAG, in its own order — the ANCHOR COLUMN (dy = 0, dx = 0) first, then
# `temporal.spiral_offsets`' points in the order that function yields them,
# with a point that rounds onto a cell already in the lag's set dropped (the
# dedup `inner_dots` already does). Each entry is
#
#     row = anchor.row + dy          col = (anchor.col + dx) mod 1440
#
# so COL IS ALWAYS IN [0, 1440): longitude wraps, because the axis closes.
# ROW MAY BE OUTSIDE [0, 721): latitude is CLIPPED, never wrapped — there is
# no cell north of the pole — and such a dot is exactly the one the sampler
# marks `valid = 0`. A port that clamps the row instead of leaving it outside
# is drawing a dot the model does not read.
#
# `lag: 0` appears only for family A and its `cells` is EMPTY. That is not a
# gap: lag 0 is the codec's 3x3 patch, one token per channel, and
# `inner_dots` deliberately emits no dots there (its docstring, "LAG 0 IS NOT
# HERE"). The nine patch offsets are exported once as `global.patch_cells`.
#
# `outer_refs` carries stage 2's annulus at the same anchors:
# `ml/cone.py::outer_spiral(lat, k)`, family-B geometry, `"family": "outer"`.
GLOBAL_SCHEMA = (
    "global.refs[i] = {family, lag, anchor:{row,col,lat,lon}, "
    "cells:[[row,col],...]}. cells is cone.inner_dots(anchor.lat, family) "
    "filtered to that lag, in its own order: the anchor column (0,0) first, "
    "then temporal.spiral_offsets' points in its order, deduplicated on the "
    "rounded cell. row = anchor.row + dy (MAY fall outside [0,ny): latitude "
    "is clipped, never wrapped, and such a dot is invalid); "
    "col = (anchor.col + dx) mod nx (ALWAYS inside [0,nx): the longitude axis "
    "closes). lag 0 exists only for family A and is empty by construction — "
    "lag 0 is the 3x3 patch, exported once as global.patch_cells. "
    "global.outer_refs has the same four keys with family='outer' and is "
    "cone.outer_spiral(anchor.lat, lag).")


def _fam_channels():
    """channel -> family over the whole r3 tensor, plus the per-family lists.

    `channel_family` RAISES on an unknown name, which is the point: if a
    channel is added to the tensor and not assigned a family, this export
    fails rather than shipping a cone that quietly does not cover it.
    """
    by_chan = {c: cone.channel_family(c) for c in CHANS_R3}
    lists = {f: sorted(c for c in CHANS_R3 if by_chan[c] == f) for f in "ABC"}
    return by_chan, lists


def _cell(lat, lon):
    """The global grid cell holding (lat, lon), as (row, col)."""
    r = int(round((lat - GLOBAL["lat0"]) / GLOBAL["step"]))
    c = int(round((lon - GLOBAL["lon0"]) / GLOBAL["step"])) % GLOBAL["nx"]
    return r, c


def _gc_km(lat1, lon1, lat2, lon2):
    """Great-circle distance on the sphere `cone.KM_PER_DEG` implies.

    R = KM_PER_DEG * 180 / pi, so this and `cone.ground_km` are two
    measurements of the SAME sphere and their difference is the flat-earth
    approximation and nothing else.
    """
    R = cone.KM_PER_DEG * 180.0 / math.pi
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp, dl = p2 - p1, math.radians(lon2 - lon1)
    h = (math.sin(dp / 2) ** 2
         + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2)
    return 2 * R * math.asin(min(1.0, math.sqrt(h)))


def _global_refs():
    """`global.refs`, `global.outer_refs` and the near-pole measurement.

    See GLOBAL_SCHEMA above; the schema is a CONTRACT (the browser replays its
    own port of the sunflower against these arrays), so nothing here may be
    reshaped without changing that text in the same edit.
    """
    ny, nx, step = GLOBAL["ny"], GLOBAL["nx"], GLOBAL["step"]
    by_id = {}
    refs, outer = [], []
    for aid, lat, lon in GLOBAL_ANCHORS:
        r, c = _cell(lat, lon)
        anc = dict(id=aid, row=r, col=c,
                   lat=GLOBAL["lat0"] + r * step,
                   lon=GLOBAL["lon0"] + c * step)
        by_id[aid] = anc
        for fam in sorted(GLOBAL_REF_LAGS):
            dots = cone.inner_dots(anc["lat"], fam, L_in=L_IN, dlat_deg=step)
            for lag in GLOBAL_REF_LAGS[fam]:
                refs.append(dict(
                    family=fam, lag=int(lag),
                    anchor={k: anc[k] for k in ("row", "col", "lat", "lon")},
                    anchor_id=aid,
                    cells=[[r + dy, (c + dx) % nx]
                           for l, dy, dx in dots if l == lag]))
        if aid in GLOBAL_OUTER_ANCHORS:
            for k in GLOBAL_OUTER_K:
                sp = cone.outer_spiral(anc["lat"], int(k), dlat_deg=step,
                                       L_in=L_IN)
                outer.append(dict(
                    family="outer", lag=int(k),
                    anchor={j: anc[j] for j in ("row", "col", "lat", "lon")},
                    anchor_id=aid,
                    cells=[[r + dy, (c + dx) % nx] for dy, dx in sp]))

    # THE NEAR-POLE MEASUREMENT, recorded rather than fixed. E-071 §1 says a
    # dot should be a DESTINATION POINT on the sphere; `spiral_offsets` places
    # it by a flat-earth cos(phi) scaling instead, which is fine to ~70 deg
    # and is not fine at 85. Both numbers below are about THIS export's own
    # cells, so a future change to the sampler moves them.
    anc = by_id["near_pole"]
    off_row, off_far, tot = 0, 0, 0
    worst = 0.0
    cell_km = cone.KM_PER_DEG * step
    for fam in sorted(GLOBAL_REF_LAGS):
        for l, dy, dx in cone.inner_dots(anc["lat"], fam, L_in=L_IN,
                                         dlat_deg=step):
            if l not in GLOBAL_REF_LAGS[fam] or (dy == 0 and dx == 0):
                continue
            tot += 1
            row = anc["row"] + dy
            if not (0 <= row < ny):
                off_row += 1
                continue
            ykm, xkm = cone.ground_km(float(dy), float(dx), anc["lat"], step)
            flat = math.hypot(ykm, xkm)
            true = _gc_km(anc["lat"], anc["lon"],
                          GLOBAL["lat0"] + row * step,
                          GLOBAL["lon0"] + ((anc["col"] + dx) % nx) * step)
            err = abs(true - flat)
            worst = max(worst, err)
            if err > cell_km:
                off_far += 1
    near = dict(
        anchor_id="near_pole", lat=anc["lat"], lon=anc["lon"],
        row=anc["row"], col=anc["col"],
        coslat=round(math.cos(math.radians(anc["lat"])), 6),
        cos_floor=0.05,
        floor_binds=bool(math.cos(math.radians(anc["lat"])) <= 0.05),
        cell_km=round(cell_km, 6),
        n_dots=tot, n_off_grid_rows=off_row, n_over_one_cell=off_far,
        max_error_km=round(worst, 3),
        note=("`spiral_offsets` places a dot by scaling its zonal offset by "
              "1/cos(phi) on a FLAT grid, and clamps cos(phi) at 0.05 "
              "(temporal.py:402, cone.ground_km's same floor). At 85 N "
              "cos(phi) = 0.0872, so the FLOOR DOES NOT BIND — 87.1 N is "
              "where it would — and what bites instead is the flat-earth "
              "approximation itself: `n_over_one_cell` of `n_dots` land more "
              "than one row-width (27.83 km) from the great-circle distance "
              "the offset was computed for, and `n_off_grid_rows` leave the "
              "latitude axis entirely and are read as invalid. E-071 §1's fix "
              "— compute each dot as a destination point on the sphere — is "
              "NOT applied here: this block exports the geometry the sampler "
              "actually reads today."))
    return refs, outer, near


def build():
    by_chan, fam_chans = _fam_channels()
    depth = sorted(c for c in CHANS_R3 if cone.is_depth_channel(c))

    families = {}
    for f in "ABC":
        spec = cone.FAMILIES[f]
        families[f] = dict(
            v_ms=spec["v_ms"], tau_days=spec["tau_days"],
            L_corr_km=spec["L_corr_km"],
            inner_lags=list(spec["inner_lags"]),
            channels=fam_chans[f],
            n_channels=len(fam_chans[f]),
        )

    inner_reach = {f: [cone.reach_km(f, l) for l in range(L_IN + 1)]
                   for f in "ABC"}
    # k = 0..143 inclusive: the array index IS the lag, so the page never has
    # to offset into it. Everything below k = 7 has an EMPTY outer spiral (the
    # codec already read that whole disc) -- the reach is still real, and the
    # cross-section draws it.
    outer_reach = [cone.outer_reach_km(k) for k in range(K_OUTER)]
    slot_table = {f: [cone.slots(cone.reach_km(f, l)) if cone.reach_km(f, l) > 0
                      else 0 for l in range(L_IN + 1)] for f in "ABC"}

    budget = cone.budget(40.0, list(CHANS_R3), L_in=L_IN,
                         dlat_deg=WINDOW["dlat"])
    counts = dict(
        inner_dots_A=len(cone.inner_dots(40.0, "A", L_in=L_IN)),
        inner_dots_B=len(cone.inner_dots(40.0, "B", L_in=L_IN)),
        inner_dots_C=len(cone.inner_dots(40.0, "C", L_in=L_IN)),
        inner_dots_rg=len(cone.channel_dots(40.0, depth[0], L_in=L_IN)),
        patch_tokens=budget["patch_tokens"],
        dot_tokens=budget["dot_tokens"],
        total_tokens=budget["total_tokens"],
        n_channels=budget["n_channels"],
        channels_per_family=budget["channels_per_family"],
        dots_per_family=budget["dots_per_family"],
    )

    ref_inner = {}
    for lat in REF_LATS_INNER:
        key = f"{lat:g}"
        ref_inner[key] = {f: [list(t) for t in
                              cone.inner_dots(lat, f, L_in=L_IN,
                                              dlat_deg=WINDOW["dlat"])]
                          for f in "ABC"}
        # the depth column is its own "family" as far as the drawing is
        # concerned: family-B reach, anchor column only (plan section 2).
        ref_inner[key]["rg"] = [list(t) for t in
                                cone.channel_dots(lat, depth[0], L_in=L_IN,
                                                  dlat_deg=WINDOW["dlat"])]
    ref_outer = {}
    for lat in REF_LATS_OUTER:
        ref_outer[f"{lat:g}"] = {
            str(k): [list(t) for t in
                     cone.outer_spiral(lat, k, dlat_deg=WINDOW["dlat"],
                                       L_in=L_IN)]
            for k in REF_OUTER_K}

    return dict(
        _source="ml/cone.py via ml/export_cone_geometry.py — do not hand-edit",
        constants=dict(
            KM_PER_DEG=float(cone.KM_PER_DEG),
            # lifted from the spiral's OWN globals, whichever way cone.py got
            # it (an `import temporal`, or the ast lift on a torch-less box),
            # so the page's bearings cannot drift from E-026's.
            GOLDEN_ANGLE=float(cone.spiral_offsets.__globals__["GOLDEN_ANGLE"]),
            SEC_PER_DAY=cone.SEC_PER_DAY,
            DT_DAYS=5.0,
            DLAT=WINDOW["dlat"],
            CAP_KM=cone.CAP_KM,
            OUTER_CAP_KM=cone.OUTER_CAP_KM,
            OUTER_FLOOR_KM=cone.OUTER_FLOOR_KM,
            R_MIN_KM=cone.R_MIN_KM,
            ASPECT=cone.ASPECT,
            RAMP_P=cone.RAMP_P,
            # The slot rule's three numbers, exported rather than restated:
            # the Cones tab lets a reader move them (a "what-if" geometry) and
            # its reset must land back on THESE, not on a JS literal that has
            # drifted from ml/cone.py::slots.
            SLOT_MAX=cone.SLOT_MAX,
            SLOT_MIN=cone.SLOT_MIN,
            SLOT_REF_KM=cone.SLOT_REF_KM,
            COS_FLOOR=0.05,
            L_IN=L_IN,
            K_OUTER=K_OUTER,
            OUTER_N_PTS=24,
        ),
        families=families,
        channel_family=by_chan,
        depth_channels=depth,
        channels_f3=list(CHANS_F3),
        channels_r3=list(CHANS_R3),
        reach_km=dict(inner=inner_reach, outer=outer_reach),
        slots=slot_table,
        counts=counts,
        reference=dict(inner=ref_inner, outer=ref_outer),
        window=WINDOW,
        # ADDITIVE: every key above is what it was before family 7 existed,
        # because `_fam_channels` derives from CHANS_R3 and family L's three
        # channels are not in it. The global grid is a NEW key, so the Cones
        # tab keeps reading the North Atlantic window unchanged and the globe
        # gets its own block beside it.
        **{"global": _global_block()},
    )


def _global_block():
    refs, outer, near = _global_refs()
    fams = sorted(GLOBAL_REF_LAGS)
    return dict(
        GLOBAL,
        schema=GLOBAL_SCHEMA,
        source="ml/plans/E070_family7_build.md §1 (the grid) and "
               "ml/cone.py (the dots)",
        anchors=[dict(id=a, **{k: v for k, v in _cell_meta(a).items()})
                 for a in [x[0] for x in GLOBAL_ANCHORS]],
        # The lag-0 3x3, in ml/model.py::gather_px's order (dy outer, dx
        # inner), so `cells` never has to carry it and a reader cannot get the
        # order wrong: index 4 is the centre.
        patch_cells=[[dy, dx] for dy in (-1, 0, 1) for dx in (-1, 0, 1)],
        families={f: dict(v_ms=cone.FAMILIES[f]["v_ms"],
                          tau_days=cone.FAMILIES[f]["tau_days"],
                          L_corr_km=cone.FAMILIES[f]["L_corr_km"],
                          channels=list(cone.FAMILIES[f]["channels"]),
                          inner_lags=list(cone.FAMILIES[f]["inner_lags"]),
                          ref_lags=list(GLOBAL_REF_LAGS[f]),
                          reach_km=[cone.reach_km(f, l)
                                    for l in range(L_IN + 1)],
                          slots=[cone.slots(cone.reach_km(f, l))
                                 if cone.reach_km(f, l) > 0 else 0
                                 for l in range(L_IN + 1)])
                  for f in fams},
        outer_lags=list(GLOBAL_OUTER_K),
        refs=refs,
        outer_refs=outer,
        near_pole=near,
    )


def _cell_meta(aid):
    lat, lon = next((la, lo) for i, la, lo in GLOBAL_ANCHORS if i == aid)
    r, c = _cell(lat, lon)
    return dict(lat_asked=lat, lon_asked=lon, row=r, col=c,
                lat=GLOBAL["lat0"] + r * GLOBAL["step"],
                lon=GLOBAL["lon0"] + c * GLOBAL["step"])


def dumps(obj):
    """One spelling of the file, so a regeneration is a no-op diff."""
    return json.dumps(obj, sort_keys=True, indent=1,
                      separators=(",", ": ")) + "\n"


def main():
    text = dumps(build())
    with open(OUT, "w", encoding="utf-8") as fh:
        fh.write(text)
    print(f"wrote {OUT} ({len(text)} bytes)")


if __name__ == "__main__":
    main()

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


def _fam_channels():
    """channel -> family over the whole r3 tensor, plus the per-family lists.

    `channel_family` RAISES on an unknown name, which is the point: if a
    channel is added to the tensor and not assigned a family, this export
    fails rather than shipping a cone that quietly does not cover it.
    """
    by_chan = {c: cone.channel_family(c) for c in CHANS_R3}
    lists = {f: sorted(c for c in CHANS_R3 if by_chan[c] == f) for f in "ABC"}
    return by_chan, lists


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
    )


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

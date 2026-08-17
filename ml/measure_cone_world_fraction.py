#!/usr/bin/env python3
"""What fraction of the WORLD OCEAN is a 12-month dependency cone?

Chris, 2026-08-16: *"the dependency cone is always more efficient than rolling
forward the whole world (I think?)"* — and asked for the ratio.

The containment is trivially true (cone ⊆ world, so it is never MORE work),
but the interesting question is whether the saving is real, and the answer
depends entirely on the stencil's reach:

    cone after 12 months, as a fraction of world ocean, seeded from the
    AMOC corridor (7.1% of world ocean):

        ring-8 @ 222 km          29.5%     a genuine 3.4x saving
        sunflower-89 @ 4444 km  100.0%     reached by month 3

So for the shape currently winning the leaderboard the cone IS the world
ocean, and "roll the cone" and "roll everything" are the same set. That is
worth knowing before anyone builds cone bookkeeping to save compute: at
4444 km it saves nothing and costs a scatter/gather per step, while at 222 km
it saves two thirds.

The measurement needs no GPU cache and no tensor: `data/currents.json` is a
1-degree global GLORYS surface-current grid that ships with the app, so its
nulls are a real coastline and its speeds define the corridor the same way
`rollout_spatial.corridor_pixels` does. One degree rather than a quarter
because this is a geometry question, and the escape fractions measured either
way agree to within 0.1 point.

Usage:
  python3 ml/measure_cone_world_fraction.py
  python3 ml/measure_cone_world_fraction.py --stencil 90 \
      --ring "spiral:111,4444,0.71,0.5" --horizons 12
"""
import argparse
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from temporal import build_stencil                              # noqa: E402

# The scored scope lives in the Atlantic. A 75th-percentile speed threshold
# taken GLOBALLY would hand us the ACC and the Kuroshio and call them the AMOC
# corridor, so the corridor is defined inside the family3 window and only then
# allowed to grow.
CORRIDOR_WINDOW = (-100.0, 20.0, 0.0, 70.0)
RAPID = (-80.0, -13.0, 26.5)


def dilate8(m, iters):
    ny, nx = m.shape
    for _ in range(iters):
        p = np.pad(m, 1)
        out = np.zeros_like(m)
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                out |= p[1 + dy:ny + 1 + dy, 1 + dx:nx + 1 + dx]
        m = out
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--grid", default=os.path.join(HERE, "..", "data",
                                                   "currents.json"))
    ap.add_argument("--stencil", type=int, default=0,
                    help="0 = run both reference shapes")
    ap.add_argument("--ring", default="")
    ap.add_argument("--horizons", type=int, default=12)
    ap.add_argument("--pctl", type=float, default=75.0)
    ap.add_argument("--dilate", type=int, default=1)
    ap.add_argument("--out", default="")
    a = ap.parse_args()

    g = json.load(open(a.grid))
    nx, ny = g["nx"], g["ny"]
    lons = g["west"] + (np.arange(nx) + 0.5) * g["dlon"]
    lats = g["south"] + (np.arange(ny) + 0.5) * g["dlat"]
    v = np.array([np.nan if x is None else x for x in g["values"]],
                 float).reshape(ny, nx)
    ocean = np.isfinite(v)
    ys, xs = np.where(ocean)
    P = len(ys)
    # equal-ANGLE cells shrink polewards, so a pixel count over-weights the
    # Arctic; both numbers are reported and they disagree by a few points
    warea = np.cos(np.radians(lats))[ys]

    win = ((lats[ys] >= CORRIDOR_WINDOW[2]) & (lats[ys] <= CORRIDOR_WINDOW[3])
           & (lons[xs] >= CORRIDOR_WINDOW[0]) & (lons[xs] <= CORRIDOR_WINDOW[1]))
    sp = v[ys, xs]
    thr = np.nanpercentile(sp[win], a.pctl)
    core = np.zeros((ny, nx), bool)
    hot = win & (sp >= thr)
    core[ys[hot], xs[hot]] = True
    seed = (dilate8(core, a.dilate) & ocean)[ys, xs]
    sec_y = int(np.argmin(np.abs(lats - RAPID[2])))
    seed |= ((ys == sec_y) & (lons[xs] >= RAPID[0]) & (lons[xs] <= RAPID[1]))

    print(f"world ocean at {g['dlon']} deg: {P:,} px "
          f"({P / (ny * nx) * 100:.1f}% of the grid)")
    print(f"AMOC corridor seed: {int(seed.sum()):,} px = "
          f"{seed.sum() / P * 100:.1f}% of world ocean\n")

    shapes = ([(a.stencil, a.ring or 0.0)] if a.stencil else
              [(9, 222.0), (90, "spiral:111,4444,0.71,0.5")])
    out = {"world_ocean_px": int(P), "seed_px": int(seed.sum()),
           "res_deg": g["dlon"], "shapes": []}
    for S, ring in shapes:
        NBR = build_stencil(ny, nx, ys, xs, S, ring_km=ring, lats=lats)
        cur = seed.copy()
        rows = []
        for h in range(1, a.horizons + 1):
            nb = NBR[np.where(cur)[0]]
            nxt = cur.copy()
            nxt[nb[nb >= 0]] = True
            cur = nxt
            rows.append({"h": h, "cone_px": int(cur.sum()),
                         "frac_world_ocean": round(float(cur.sum() / P), 4),
                         "frac_by_area": round(
                             float(warea[cur].sum() / warea.sum()), 4)})
        print(f"stencil {S} slots, ring {ring} "
              f"(longitude wrap {'ON' if abs(nx * g['dlon'] - 360) < 2 else 'off'})")
        for r in rows:
            if r["h"] in (1, 2, 3, 6, a.horizons):
                print(f"    {r['h']:2d} months back: cone {r['cone_px']:8,} px = "
                      f"{r['frac_world_ocean']*100:5.1f}% of world ocean "
                      f"({r['frac_by_area']*100:5.1f}% by area)")
        print()
        out["shapes"].append({"slots": S, "ring": str(ring), "rows": rows})
    if a.out:
        json.dump(out, open(a.out, "w"), indent=1)
        print("wrote", a.out)


if __name__ == "__main__":
    main()

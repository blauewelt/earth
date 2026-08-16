#!/usr/bin/env python3
"""How much of a rolled prediction's DEPENDENCY CONE leaves the window?

Chris, 2026-08-16: *"are we certain we are rolling forward all the necessary
pixels? each point points towards 144 other points. Which means we need to
roll forward the whole world?"*

The question is exactly right and the geometry is unforgiving. A stencil head
predicts pixel p at t+1 from p's S neighbours at t. Roll it again and those
neighbours' own neighbours are needed at t-1 of the roll, so the set of
pixels a single h-step prediction depends on — its *cone* — grows by the
stencil's reach every step. With a sunflower reaching 4444 km, the cone's
radius after h steps is up to 4444·h km, and the Earth's circumference is
40,075 km. The cone therefore wraps the planet within a handful of steps,
while our window (0–70 N, 100 W–20 E) is a basin.

`ml/rollout_spatial.py` already rolls EVERY window ocean pixel — that is the
most it can do — so the question is not whether we roll enough of our window,
but whether the window is large enough for the horizon we score. This script
answers that as a measurement instead of an argument: it propagates the real
`build_stencil` neighbour table h times from the corridor and counts how much
of the cone has left the window at each horizon.

What "escaped" means here, precisely: a slot that resolves to -1 in the
neighbour table — land, or outside the window rectangle. Those slots are
encoded as ZERO in both training and evaluation, so the roll is not
*wrong*; it is a well-defined experiment with a specific boundary condition:
**the world outside the window is held at its climatological mean.** The
number this script produces is how much of the answer that assumption owns.

Usage:
  python3 ml/measure_cone_escape.py --npz-small ml/cache/f3_small.npz \\
      --ckpt ml/cache/f3_anchor41M__pixelmae.pt \\
      --stencil 145 --ring "spiral:111,4444,0.71,0.5" --horizons 12
"""
import argparse
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from temporal import build_stencil, rapid_section          # noqa: E402


def propagate(NBR, start, P, horizons, verbose=False):
    """Grow the cumulative dependency cone h times over `NBR`, counting how
    much of each step's neighbour demand resolves to -1."""
    cur = start.copy()
    rows = []
    for h in range(1, horizons + 1):
        nb = NBR[np.where(cur)[0]]                  # [n, S]
        nxt = np.zeros(P, bool)
        nxt[nb[nb >= 0]] = True
        nxt |= cur                                  # the cone is cumulative
        frac_win = nxt.sum() / P
        demand, unmet = nb.size, int((nb < 0).sum())
        rows.append({"h": h, "cone_px": int(nxt.sum()),
                     "cone_frac_of_window": round(float(frac_win), 4),
                     "slots_requested": int(demand),
                     "slots_outside_or_land": unmet,
                     "frac_unmet": round(unmet / max(demand, 1), 4)})
        if verbose:
            print(f"  h={h:2d}: cone {int(nxt.sum()):7,} px "
                  f"({frac_win*100:5.1f}% of window) · "
                  f"{unmet/max(demand,1)*100:5.1f}% of this step's requested "
                  f"neighbours are land/off-window")
        saturated_now = frac_win > 0.999 and cur.sum() / P <= 0.999
        cur = nxt
        if verbose and saturated_now:
            # once only: after saturation every later step says the same thing,
            # and the interesting number from there on is frac_unmet, not the
            # cone size
            print(f"        -> the cone now covers the ENTIRE window at h={h}; "
                  f"from here it can only grow OUTSIDE it")
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz-small", default="",
                    help="tensor sidecar carrying lats/lons. Omit to build the "
                         "window analytically from --bounds/--res, which needs "
                         "no cache — the RECTANGLE half of the answer is pure "
                         "geometry, and the cache lives on rented boxes")
    ap.add_argument("--bounds", default="-100,20,0,70",
                    help="W,E,S,N when --npz-small is omitted")
    ap.add_argument("--res", type=float, default=0.25)
    ap.add_argument("--ckpt", default="")
    ap.add_argument("--mask", default="ml/cache/ocean_mask.npy")
    ap.add_argument("--stencil", type=int, default=145)
    ap.add_argument("--ring", default="spiral:111,4444,0.71,0.5")
    ap.add_argument("--horizons", type=int, default=12)
    ap.add_argument("--no-land-free-control", action="store_true",
                    help="skip the all-ocean rectangle re-run that separates "
                         "the coastline's share of the escape from ours")
    ap.add_argument("--out", default="")
    a = ap.parse_args()

    if a.npz_small:
        d = np.load(a.npz_small, allow_pickle=False)
        lats, lons = d["lats"], d["lons"]
        ocean = np.load(a.mask)
    else:
        # No cache: the window is a rectangle and a resolution, both known.
        # Land is then absent, so this run IS the land-free control and the
        # numbers it prints are the share of the escape that is ours.
        w_, e_, s_, n_ = (float(v) for v in a.bounds.split(","))
        r = a.res
        lats = np.arange(s_ + r / 2, n_, r)
        lons = np.arange(w_ + r / 2, e_, r)
        ocean = np.ones((len(lats), len(lons)), bool)
        print(f"NO MASK: analytic {r} deg rectangle {a.bounds} — land-free "
              f"by construction, so every dead slot below is the WINDOW's")
    ys, xs = np.where(ocean)
    P = len(ys)
    H, W = ocean.shape
    print(f"window {H}x{W}, {P:,} ocean pixels, "
          f"lat {lats[0]:.2f}..{lats[-1]:.2f}, lon {lons[0]:.2f}..{lons[-1]:.2f}")

    NBR = build_stencil(H, W, ys, xs, a.stencil, a.ring, lats)
    S = NBR.shape[1]
    dead0 = float((NBR < 0).mean())
    print(f"stencil {S} slots, ring {a.ring}: "
          f"{dead0*100:.1f}% of all slots are dead at h=1 (land or off-window)")

    # the scored scope: RAPID section ∪ a corridor stand-in. The corridor
    # proper needs the tensor (train-month current speed); for a GEOMETRY
    # question the section plus its surroundings is the honest subset — it is
    # where the headline number is read, and it sits mid-window, i.e. the
    # best case for staying inside.
    sec_y, sec_sel = rapid_section(lats, lons, ys, xs)
    start = np.zeros(P, bool)
    start[sec_sel] = True
    print(f"seed set: RAPID section, {int(start.sum())} pixels at "
          f"{lats[sec_y]:.2f} N")

    rows = propagate(NBR, start, P, a.horizons, verbose=True)

    # ---- WHOSE fault is the escape: the coastline, or our rectangle? -------
    # Both show up as -1 and both are zero-filled, so one number cannot tell
    # them apart — and they have completely different remedies. Land is a fact
    # about the ocean and is correctly zero (there is no water there to
    # predict). The rectangle is a choice we made, and every slot it kills is
    # a slot a larger domain would return. Re-running the identical
    # propagation over an ALL-OCEAN rectangle of the same shape isolates the
    # half we can actually buy back, which is the number that should drive
    # whether E-033's global tensor is an improvement or a correctness fix.
    ctl = None
    if not a.no_land_free_control and ocean.all():
        print("\n(window is already land-free — no control needed)")
    elif not a.no_land_free_control:
        yy, xx = np.meshgrid(np.arange(H), np.arange(W), indexing="ij")
        yy, xx = yy.ravel(), xx.ravel()
        NBR_lf = build_stencil(H, W, yy, xx, a.stencil, a.ring, lats)
        dead_lf = float((NBR_lf < 0).mean())
        sec_lf = np.where((yy == sec_y)
                          & np.isin(xx, xs[sec_sel]))[0]
        start_lf = np.zeros(len(yy), bool)
        start_lf[sec_lf] = True
        print(f"\n-- land-free control: the same rectangle with no coastline "
              f"({len(yy):,} cells) --")
        print(f"stencil {S} slots: {dead_lf*100:.1f}% of all slots are dead "
              f"at h=1 from the RECTANGLE ALONE "
              f"(vs {dead0*100:.1f}% with land — so the coastline owns "
              f"{max(dead0 - dead_lf, 0)*100:.1f} points and the window "
              f"{dead_lf*100:.1f})")
        rows_lf = propagate(NBR_lf, start_lf, len(yy), a.horizons, verbose=True)
        ctl = {"dead_slot_frac_h1": round(dead_lf, 4), "rows": rows_lf}

    out = {"window": {"H": int(H), "W": int(W), "ocean_px": int(P),
                      "lat": [float(lats[0]), float(lats[-1])],
                      "lon": [float(lons[0]), float(lons[-1])]},
           "stencil": {"slots": int(S), "ring": a.ring,
                       "dead_slot_frac_h1": round(dead0, 4)},
           "seed": "rapid_section", "rows": rows,
           "land_free_control": ctl}
    if a.out:
        json.dump(out, open(a.out, "w"), indent=1)
        print("wrote", a.out)


if __name__ == "__main__":
    main()

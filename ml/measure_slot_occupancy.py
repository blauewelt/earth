#!/usr/bin/env python3
"""How many of a stencil's slots actually LAND ON WATER inside the window?

A precondition check, run where the inputs are all it has cost (ml/CLAUDE.md
§0.3). E-026's 4444 km spirals reach roughly 40 degrees of latitude or 52 of
longitude from their centre, and the family3 window is only 70 x 120 degrees
— so a large fraction of those far slots may be off-grid or on land, where
`build_stencil` writes -1 and `gather_stencil` substitutes zeros. A design
whose outer half is structurally empty is not a wide design; it is a narrow
design paying a wide design's parameter count, and it would train perfectly
happily while being that.

The occupancy is measured against the ACTUAL window and land mask — the same
one the evaluator exports for the globe (`data/amoc_eval_mask.json`, written
by `rollout_spatial.py --export-mask` from `corridor_pixels()`), so this is
the real geometry, not an idealisation of it. Two populations are reported
separately, because they answer different questions:

  ROLLED    all 84,405 ocean pixels the evaluator advances — what the model
            trains and rolls on
  CORRIDOR  the 29,627 fastest-quarter + RAPID cells the headline AUC is read
            from — what the experiment is actually judged on

    python3 ml/measure_slot_occupancy.py
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from temporal import build_stencil                       # noqa: E402

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MASK = os.path.join(HERE, "data", "amoc_eval_mask.json")

DESIGNS = [
    ("ring 8 @ 222 (champion)", 9, "222"),
    ("3 rings 4+4+4 @ 222/555/1000", 13, "222,555,1000"),
    ("3 rings 8+8+8 @ 222/555/1000", 25, "222,555,1000"),
    ("spiral 8, 111-890", 9, "spiral:111-890"),
    ("spiral 13, 222-1000", 14, "spiral:222-1000"),
    ("spiral 24, 111-4444", 25, "spiral:111-4444"),
    ("spiral 36, 111-4444", 37, "spiral:111-4444"),
    ("spiral 24, 111-2222", 25, "spiral:111-2222"),
    ("spiral 36, 111-2222", 37, "spiral:111-2222"),
    ("espiral 24, 111-4444 a=0.71", 25, "spiral:111-4444-0.71"),
    ("espiral 24, 111-4444 a=0.5", 25, "spiral:111-4444-0.5"),
    ("spiral 34, 111-4444", 35, "spiral:111-4444"),
]


def load():
    g = json.load(open(MASK))
    ny, nx = g["ny"], g["nx"]
    vals = np.array([-1 if c == "." else int(c) for c in g["packed"]],
                    np.int8).reshape(ny, nx)
    lats = g["south"] + (np.arange(ny) + 0.5) * g["dlat"]
    # class codes are 1-based (see g["classes"]): 1 rolled, 2 corridor,
    # 3 RAPID section — and the classes NEST, every corridor cell being a
    # rolled cell. `.` is empty. Reading these as 0-based made the corridor
    # come out as all 84,405 rolled pixels, i.e. the two columns were the
    # same number printed twice under different headings — which looked
    # exactly like a legitimate finding ("the corridor is no worse off").
    ocean = vals >= 1
    corridor = vals >= 2
    return vals, ocean, corridor, lats, g


def main():
    vals, ocean, corridor, lats, g = load()
    ny, nx = ocean.shape
    ys, xs = np.where(ocean)
    print(f"window {g['west']}..{g['east']} E, {g['south']}..{g['north']} N  "
          f"· {nx}x{ny} cells · {len(ys):,} ocean pixels, "
          f"{int(corridor.sum()):,} corridor")
    corr_of_px = corridor[ys, xs]
    print(f"\n{'design':<32} {'slots':>5} {'rolled':>18} {'corridor':>18}")
    for name, slots, ring in DESIGNS:
        nbr = build_stencil(ny, nx, ys, xs, slots, ring_km=ring, lats=lats)
        live = (nbr[:, 1:] >= 0)                 # slot 0 is the centre itself
        f_all = live.mean()
        f_cor = live[corr_of_px].mean()
        n_all = live.sum(1).mean()
        n_cor = live[corr_of_px].sum(1).mean()
        print(f"{name:<32} {slots:>5} "
              f"{f_all * 100:>8.1f}% ({n_all:4.1f}/{slots - 1:>2}) "
              f"{f_cor * 100:>8.1f}% ({n_cor:4.1f}/{slots - 1:>2})")
    print("\n% = share of non-centre slots that resolve to a real ocean pixel; "
          "the rest are\nland or outside the window and are fed to the model "
          "as exact zeros.")


if __name__ == "__main__":
    main()

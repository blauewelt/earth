#!/usr/bin/env python3
"""How many DISTINCT platforms do the k tokens of a k-nearest search come from?

A "platform" is the instrument that made the measurement — one drifter's AOML
identifier, one mooring's WMO code, one research cruise's expocode. The
question matters because k is usually chosen as if it bought k independent
looks at the ocean; if all k tokens are the same drifter twenty minutes apart,
it bought one track's redundancy instead. The 2026-09-14 verification of the
first three published family-10 tier-P stores found exactly that: a median of
1 distinct platform among 8 for `gdp` and `socat`, and 2 among 4 for `gtmba`.

Anchors are drawn from the store's own rows inside the tensor epoch, so this
measures the on-track case, which is the favourable one.

Usage:
    python3 platform_diversity.py --store gdp --dir /path/to/family10_1/gdp
    python3 platform_diversity.py --store socat --dir ... --anchors 600

Reads either schema — `time_s` (family 10.1) or `time_days` (family 10 and
family 8) — through `ml/family10_store.py`, and touches no time column itself.
"""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "..", ".."))
from family10_store import Store            # noqa: E402

SUGGESTED = {"gdp": (8, 300.0, 10.0),
             "gtmba": (4, 1500.0, 15.0),
             "socat": (8, 500.0, 30.0),
             "slatrack": (16, 200.0, 10.0)}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--store", required=True, choices=sorted(SUGGESTED))
    ap.add_argument("--dir", required=True,
                    help="store directory, or a Hub prefix repo:path")
    ap.add_argument("--anchors", type=int, default=600)
    ap.add_argument("--seed", type=int, default=7)
    a = ap.parse_args()

    k, r_sug, t_sug = SUGGESTED[a.store]
    rng = np.random.default_rng(a.seed)
    st = Store.open(a.dir)
    b = np.asarray(st["bin"])
    lat = np.asarray(st["lat"])
    lon = np.asarray(st["lon"])
    pos = np.nonzero(b >= 0)[0]
    sel = rng.choice(pos, min(a.anchors, len(pos)), replace=False)

    n_sug, n_gen = [], []
    for i in sel:
        # at the suggested bounds
        tok = st.knearest(float(lat[i]), float(lon[i]), bin=int(b[i]), k=k,
                          R_max_km=r_sug, T_max_days=t_sug)
        v = tok["valid"]
        n_sug.append(len(np.unique(tok["platform"][v])) if v.any() else 0)
        # and at generous bounds with the same ranking metric, to show that a
        # bigger radius does not buy more platforms either
        tok = st.knearest(float(lat[i]), float(lon[i]), bin=int(b[i]), k=k,
                          R_max_km=5000.0, T_max_days=60.0,
                          L_km=r_sug, T_days=t_sug)
        v = tok["valid"]
        n_gen.append(len(np.unique(tok["platform"][v])) if v.any() else 0)

    print(f"{a.store} k={k} over {len(sel)} catalogue anchors")
    print(f"  distinct platforms @ suggested "
          f"({r_sug:g} km, {t_sug:g} d): median {np.median(n_sug):g} "
          f"mean {np.mean(n_sug):.2f}")
    print(f"  distinct platforms @ generous (5000 km, 60 d, same metric): "
          f"median {np.median(n_gen):g} mean {np.mean(n_gen):.2f}")


if __name__ == "__main__":
    main()

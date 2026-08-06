#!/usr/bin/env python3
"""Study: is the stage-2 transformer the right size — and is the mini
training-time probe a faithful proxy for bigger ones?

Sweeps temporal.py configs over the SAME frozen embeddings (the Z cache
makes each config pay only its own training, not the embedding pass), with
multiple seeds per config because yesterday taught us what single-seed
probe numbers are worth at n=36.

Configs span the axes that could matter: width/depth (capacity), K
(context length), steps (optimization budget). The mini probe's config
(d64 L2, 400 steps, K12, 600 pixels — from trainprobe.py) is included via
its own tool for the faithfulness comparison.

Usage: python3 ml/sweep_temporal.py --run actions [--seeds 2]
Writes runs/<run>/sweep_temporal.json (aggregated) after running each
config via temporal.py --tag.
"""
import argparse
import json
import os
import subprocess
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))

CONFIGS = [
    # tag,        d_model, layers, K, steps
    ("small",     64, 2, 12, 2000),
    ("mid",       96, 3, 24, 2000),     # the temporal.py default shape
    ("mid-longK", 96, 3, 36, 2000),
    ("mid-shortK", 96, 3, 6, 2000),
    ("large",     192, 4, 24, 4000),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="actions")
    ap.add_argument("--seeds", type=int, default=2)
    a = ap.parse_args()
    run_dir = os.path.join(HERE, "runs", a.run)

    out = {"run": a.run, "configs": []}
    for tag, dm, ly, K, steps in CONFIGS:
        rows = []
        for seed in range(a.seeds):
            t = f"{tag}-s{seed}"
            print(f"== {t}: d={dm} L={ly} K={K} steps={steps}", flush=True)
            subprocess.run(
                [sys.executable, os.path.join(HERE, "temporal.py"),
                 "--run", a.run, "--d-model", str(dm), "--layers", str(ly),
                 "--K", str(K), "--steps", str(steps), "--seed", str(seed),
                 "--tag", t],
                check=True)
            rows.append(json.load(open(os.path.join(run_dir, f"temporal_{t}.json"))))
        agg = {"tag": tag, "d_model": dm, "layers": ly, "K": K, "steps": steps}
        for label, path in (("chan_pct", ("chan_t+1",)), ("z_pct", ("z_t+1",))):
            vals = [100 * (1 - r[path[0]]["mse_model"] / r[path[0]]["mse_persistence"])
                    for r in rows]
            agg[label] = round(float(np.mean(vals)), 1)
            agg[label + "_seeds"] = [round(v, 1) for v in vals]
        rvals = [r["rapid_probe"]["r_deseasonalised"] for r in rows]
        agg["r_tmp"] = round(float(np.mean(rvals)), 3)
        agg["r_tmp_seeds"] = rvals
        out["configs"].append(agg)
        print(json.dumps(agg), flush=True)

    json.dump(out, open(os.path.join(run_dir, "sweep_temporal.json"), "w"), indent=2)
    print("wrote", os.path.join(run_dir, "sweep_temporal.json"))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Which dataset earns its place? Inference-time channel-group ablation.

The codec was trained with explicit missing tokens — absence is a first-
class input, not padding — so we can ask it directly: embed every pixel
with one dataset GROUP hidden from the encoder (or with ONLY that group
visible), and re-run the standardized probe suite. Scoring targets always
use the true observations, so every condition answers the same question:
how well is the real ocean predicted from this subset of the inputs?

Two directions per group:
  drop-<g>  leave-one-out — how much is LOST without the group
            (unique contribution; redundant info shows up as a small drop)
  only-<g>  the group alone (+ month/coords context) — how much the group
            carries by itself (shared info shows up as a high floor)

This is an ablation of the INPUTS to a fixed codec, not of training: a
group could carry information the codec never learned to use — that upper
bound needs retrained codecs (the d_z sweep machinery covers that path).

Usage: python3 ml/ablate_channels.py --run actions [--seeds 2]
Writes runs/<run>/ablation.json.
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model import codec_from_ckpt
from trainprobe import anomaly_transform, probe_now

HERE = os.path.dirname(os.path.abspath(__file__))

GROUPS = {
    "glorys_dyn": ["cur_speed", "log_mld"],          # GLORYS reanalysis (monthly)
    "rg_temp": ["rg_t10", "rg_t200", "rg_t700", "rg_t1500"],   # RG-Argo T
    "rg_sal": ["rg_s10", "rg_s200", "rg_s700", "rg_s1500"],    # RG-Argo S
    "static_clim": ["sst_clim", "precip_clim"],      # OISST + GPCP climatologies
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="actions")
    ap.add_argument("--seeds", type=int, default=2)
    a = ap.parse_args()

    run_dir = os.path.join(HERE, "runs", a.run)
    ck = torch.load(os.path.join(run_dir, "pixelmae.pt"),
                    map_location="cpu", weights_only=False)
    d = np.load(os.path.join(HERE, "cache", "na_pixels.npz"))
    X = d["X"].copy()
    chan = [str(c) for c in d["chan"]]
    months = [str(m) for m in d["months"]]
    moy = np.array([int(m[5:7]) - 1 for m in months])
    hold_years = set(ck["args"]["holdout_years"].split(","))
    t_hold = np.array([m[:4] in hold_years for m in months])
    lo, hi = (float(v) for v in ck["args"]["holdout_lon"].split(","))
    x_hold = (d["lons"] >= lo) & (d["lons"] < hi)
    if not ck["args"].get("anomaly"):
        sys.exit("ablation measures anomaly-space codecs only")
    X, dynamic = anomaly_transform(X, moy, t_hold, x_hold)

    # THROUGH codec_from_ckpt, not by hand. This site hand-built the pilot
    # architecture from two fields, so it was already wrong for every codec
    # wider than 128 — and with E-046 it would also silently drop the FSQ
    # bottleneck, load the state_dict cleanly and ablate a different model.
    codec = codec_from_ckpt(ck, len(chan))
    codec.load_state_dict(ck["model"])

    Xt = torch.from_numpy(np.nan_to_num(X, nan=0.0))
    OBS = torch.from_numpy(np.isfinite(X))

    # Hide channels via the MASK token, not the missing token: training
    # masked random channel subsets constantly (mask_ratio 0.5), so every
    # hide pattern is in-distribution. The miss token is only in-distribution
    # for channels that are genuinely absent in the data (RG pre-2004) —
    # measured 2026-08-06: hiding the always-present statics via miss
    # collapsed the linear probe to -0.05, via mask it stayed at 0.429.
    def cond(drop=None, keep=None):
        if drop is not None:
            return [chan.index(c) for c in drop]
        return [i for i, c in enumerate(chan) if c not in keep]

    conds = {"full": None}
    for g, cols in GROUPS.items():
        conds[f"drop-{g}"] = cond(drop=cols)
        conds[f"only-{g}"] = cond(keep=cols)

    results = {"run": a.run, "seeds": a.seeds, "groups": GROUPS, "conditions": {}}
    for name, mask_chan in conds.items():
        rows = []
        for seed in range(a.seeds):
            t0 = time.time()
            m = probe_now(codec, Xt, OBS, d, moy, t_hold, x_hold, dynamic,
                          seed=seed, mask_chan=mask_chan)
            rows.append(m)
            print(f"  {name:<18} seed {seed}: chan {m['chan_vs_persistence_pct']:+5.1f}% · "
                  f"r_lin {m['linear_r_deseas']:+.3f} · r_tmp {m['temporal_r_deseas']:+.3f} "
                  f"({time.time() - t0:.0f}s)", flush=True)
        agg = {k: round(float(np.mean([r[k] for r in rows])), 3)
               for k in ("chan_vs_persistence_pct", "z_vs_persistence_pct",
                         "linear_r_deseas", "temporal_r_deseas")}
        agg["seeds"] = [{k: r[k] for k in agg if k != "seeds"} for r in rows]
        results["conditions"][name] = agg

    results["mode"] = "mask_token"
    path = os.path.join(run_dir, "ablation.json")
    json.dump(results, open(path, "w"), indent=2)
    print("wrote", path)


if __name__ == "__main__":
    main()

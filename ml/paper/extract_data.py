#!/usr/bin/env python3
"""Provenance of ml/paper/data/report_data.json — how it is (re)built.

Every number in the report's figures and tables comes from one of these
public artefacts, and nothing is typed in by hand:

  probes-516.json  — E-062-R0, the first clean roll (E-059 head, window pool)
  probes-510.json  — the E-051 head (endpoint pool) on the identical battery
  probes-513.json  — the longitude-holdout roll (E-057 FGN M=8 head and the
                     e017 gate head), source of the spatial split and the
                     ensemble-dispersion curves
    all three:  https://raw.githubusercontent.com/blauewelt/earth/ml-metrics/<name>
  probes-527.json  — E-066, the linear inverse model (K = 50/100/200) scored
                     through the identical battery from the identical starts
  metrics.jsonl    — the stage-2 training records of E-051, E-059, E-060a/b/c
                     (the TPU trainers' own logs; mirrored here because the
                     bucket they were written to is not public)

Usage:  python3 extract_data.py --probes DIR --metrics DIR
where DIR/probes-{510,513,516}.json and DIR/{e051,e059,e060a,e060b,e060c}_metrics.jsonl
exist. Writes data/report_data.json (compact, ~100 KB). The committed copy was
produced on 2026-08-31 from the artefacts as archived on that day.
"""
import argparse
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
ap = argparse.ArgumentParser()
ap.add_argument("--probes", required=True)
ap.add_argument("--metrics", required=True)
a = ap.parse_args()

FIELDS = ("h", "n", "msss_clim", "msss_pers", "msss_damped", "acc", "amp_ratio")


def head(path):
    d = json.load(open(path))["files"]["rollout_spatial.json"]
    return d, d["heads"]


def rows(b, scope):
    return [{k: r[k] for k in FIELDS} for r in b[scope]["chan_skill"]]


d516, h516 = head(os.path.join(a.probes, "probes-516.json"))
b516 = list(h516.values())[0]
d510, h510 = head(os.path.join(a.probes, "probes-510.json"))
b510 = list(h510.values())[0]
d513, h513 = head(os.path.join(a.probes, "probes-513.json"))

out = {}
out["r0_clean_516"] = {
    "meta": b516["meta"], "hold_years": d516["hold_years"],
    "starts_per_year": d516["starts"]["per_year"],
    "corridor": rows(b516, "corridor"), "gate": rows(b516, "gate"),
    "window": rows(b516, "window"),
    "corridor_per_channel": {
        ch: [{k: r[k] for k in ("h", "n", "msss_clim", "msss_damped", "acc", "amp_ratio")} for r in rr]
        for ch, rr in b516["corridor"]["per_channel"].items() if rr and rr[0]["n"] > 0},
    "horizon_auc": {s: b516[s]["horizon_auc"] for s in ("gate", "corridor", "window")},
    "auc_damped": {s: b516[s]["auc_damped"] for s in ("gate", "corridor", "window")},
    "amoc_bands_unpooled": b516["amoc_bands_unpooled"], "amoc_bands": b516["amoc_bands"],
    "long": {k: v for k, v in b516["long"].items() if not isinstance(v, list)},
}
out["contaminated_twin_510"] = {
    "meta": b510["meta"], "corridor": rows(b510, "corridor"),
    "horizon_auc": {s: b510[s]["horizon_auc"] for s in ("gate", "corridor", "window")},
    "amoc_bands_unpooled": b510["amoc_bands_unpooled"], "amoc_bands": b510["amoc_bands"],
}
fgn = h513["s145rspiral:111-4444-0.71-0.5fgnM8_s0"]
e017 = h513["s1_s0"]


def split(hb):
    return {s + suf: {"horizon_auc": hb[s + suf].get("horizon_auc"), "n_px": hb[s + suf].get("n_px")}
            for s in ("gate", "corridor", "window") for suf in ("_trainlon", "_holdlon")}


out["spatial_split_513"] = {
    "holdout_lon": {k: d513["holdout_lon"][k] for k in ("arg", "lo", "hi", "n_cols")},
    "fgn_m8_head": {"meta": fgn["meta"], **split(fgn)},
    "gate_head_e017": {"meta": e017["meta"], **split(e017)},
}
out["dispersion_513"] = {
    ph: {k: fgn[ph + "_dispersion"][k] for k in ("roll_ym", "field_var", "sv_spread", "members")}
    for ph in ("long", "future")}


def curve(name):
    rs = [json.loads(l) for l in open(os.path.join(a.metrics, f"{name}_metrics.jsonl")) if l.strip()]
    cfg = next(r["stage2_config"] for r in rs if "stage2_config" in r)
    vp = next(r["stage2_monitor"]["val_persistence"] for r in rs if "stage2_monitor" in r)
    st = [r for r in rs if "stage2_step" in r]
    return {"params_M": cfg["params_M"], "d_model": cfg["d_model"], "layers": cfg["layers"],
            "steps": cfg["steps"], "holdout_scope": cfg.get("holdout_scope", "endpoint"),
            "train_windows": cfg["train_windows"], "val_persistence": vp,
            "step": [r["stage2_step"] for r in st], "train_zmse": [r["stage2_zmse"] for r in st],
            "val_zmse": [r["stage2_val_zmse"] for r in st],
            "probe": [r["stage2_probe"] for r in rs if "stage2_probe" in r]}


out["curves"] = {k: curve(k) for k in ("e051", "e059", "e060a", "e060b", "e060c")}

# E-066 · the LIM baseline, one entry per K, same rows as the head
p527 = os.path.join(a.probes, "probes-527.json")
if os.path.exists(p527):
    d527, h527 = head(p527)
    out["lim_527"] = {}
    for name, b in h527.items():
        out["lim_527"][name] = {
            "meta": {k: v for k, v in b["meta"].items() if k != "eigen"},
            "eigen": b["meta"].get("eigen"),
            "corridor": rows(b, "corridor"), "gate": rows(b, "gate"), "window": rows(b, "window"),
            "corridor_per_channel": ({
                ch: [{k: r[k] for k in ("h", "n", "msss_clim", "msss_damped", "acc", "amp_ratio")} for r in rr]
                for ch, rr in b["corridor"]["per_channel"].items() if rr and rr[0]["n"] > 0}
                if name == "lim_k200" else None),
        }
# E-062-R0b / R0c · the 7.598M (probes-520) and 40.388M (probes-523) heads
# through #516's identical battery; one entry per run, same rows as the head
out["width_rolls"] = {}
for run, params in (("520", 7.598), ("523", 40.388)):
    pth = os.path.join(a.probes, f"probes-{run}.json")
    if not os.path.exists(pth):
        continue
    dd, hh = head(pth)
    b = list(hh.values())[0]
    out["width_rolls"][run] = {
        "params_M": params, "meta": b["meta"], "wall_s": b.get("wall_s"),
        "corridor": rows(b, "corridor"), "gate": rows(b, "gate"), "window": rows(b, "window"),
        "corridor_per_channel": {
            ch: [{k: r[k] for k in ("h", "n", "msss_clim", "msss_damped", "acc", "amp_ratio")} for r in rr]
            for ch, rr in b["corridor"]["per_channel"].items() if rr and rr[0]["n"] > 0},
        "horizon_auc": {s: b[s]["horizon_auc"] for s in ("gate", "corridor", "window")},
        "auc_damped": {s: b[s]["auc_damped"] for s in ("gate", "corridor", "window")},
        "amoc_bands_unpooled": b["amoc_bands_unpooled"], "amoc_bands": b["amoc_bands"],
        "long": {k: v for k, v in b["long"].items() if not isinstance(v, list)},
    }
os.makedirs(os.path.join(HERE, "data"), exist_ok=True)
dst = os.path.join(HERE, "data", "report_data.json")
json.dump(out, open(dst, "w"), separators=(",", ":"))
print("wrote", dst, os.path.getsize(dst), "bytes")

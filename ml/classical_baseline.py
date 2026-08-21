#!/usr/bin/env python3
"""The classical AMOC reconstructions, re-scored in OUR protocol.

WHY THIS EXISTS. The published numbers we are measured against are not
comparable to ours (paper §6.1): they low-pass at 18 months, they calibrate
in-sample, and — the big one — several of them take a *transport
measurement* as an input. RAPID's MOC is the sum of three terms,

    MOC  =  Florida Current  +  Ekman  +  upper-mid-ocean geostrophic,

and Frajka-Williams (2015) feeds the submarine-cable record (which IS the
Florida Current term) plus Ekman into the reconstruction, inferring only
the third. Their r ~ 0.95 is therefore substantially the correlation of a
partial sum with its own total.

Chris asked whether we can produce numbers that ARE comparable. We can, and
this is the direction that works: instead of trying to make our number look
like theirs, re-run THEIR inputs through OUR protocol — real RAPID, monthly,
deseasonalised, year-blocked k-fold, no in-sample calibration. Then every
row differs only in what the model is allowed to see, which is the
comparison anyone actually wants.

The ladder (each row adds inputs, protocol held fixed):

    wind        tau_x, tau_y on the section          — our standing baseline
    cable       the Florida Current transport alone  — one measured term
    cable+wind  both measured terms                  — the FW2015 input set
                                                       minus altimetry
    embedding   our codec, NO transport at all       — for reference

Reading the result: if cable+wind scores far above our embedding, that is
not a defeat — it is the quantification of how much of the classical
skill is bought by measuring two of the three addends. If it scores near
ours, the classical advantage was mostly the filtering.

Usage:  python3 ml/classical_baseline.py [--data ml/cache/na_pixels_c24_global.npz]
Writes ml/runs/classical_baseline.json.
"""
import argparse
import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
import sys
sys.path.insert(0, HERE)
from probe_kfold import kfold_r, section_of, lowpass_r  # same protocol, no copy


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=os.path.join(HERE, "cache", "na_pixels.npz"))
    ap.add_argument("--out", default=os.path.join(HERE, "runs", "classical_baseline.json"))
    a = ap.parse_args()

    d = np.load(a.data, allow_pickle=False)
    months = [str(m) for m in d["months"]]
    moy = np.array([int(m[5:7]) - 1 for m in months])
    yr = np.array([int(m[:4]) for m in months])
    month_of_ym = {int(m[:4]) * 100 + int(m[5:7]): i for i, m in enumerate(months)}
    lats, lons = d["lats"], d["lons"]

    # ---- target: RAPID, deseasonalised exactly as probe_kfold does ---------
    arr = d["rapid"]
    tidx = arr[:, 0].astype(int)
    vals = arr[:, 1].astype(float)
    tmoy = moy[tidx]
    clim = np.array([vals[tmoy == m].mean() for m in range(12)])
    y = vals - clim[tmoy]

    # ---- feature 1: wind stress on the 26.5N section ----------------------
    # POOLED, and it stays pooled ON PURPOSE (2026-08-21). Every row in this
    # file is a section MEAN through the same ridge, so the LADDER is
    # internally matched — wind, cable and cable+wind differ only in what the
    # model is allowed to see, which is the whole point of the file. Making
    # one row unpooled would break exactly that. What must not happen is the
    # OTHER comparison: these rows are not a bar for ml/probe_head.py's
    # unpooled numbers, and the JSON says so in `probe` and `pooled` below so
    # a table generator cannot mix them by accident.
    X = d["X"]
    ocean = np.isfinite(X[..., 0]).any(axis=0)
    ys, xs = np.where(ocean)
    chan = [str(c) for c in d["chan"]]
    wsel = [i for i, c in enumerate(chan) if c in ("tau_x", "tau_y")]
    sec_y, sec_sel = section_of(lats, lons, ys, xs, 26.5, -80.0, -13.0)
    wind = np.stack([
        np.nanmean(X[:, ys[sec_sel], xs[sec_sel]][..., wsel], axis=1)[:, k]
        for k in range(len(wsel))], 1)
    wind = np.nan_to_num(wind, copy=False)

    # ---- feature 2: the Florida Current cable transport -------------------
    # This is one of our TRUTH series. Using it as an INPUT here is
    # deliberate and is the whole point of the experiment: it reproduces the
    # classical input set. Nothing else in this repository ever does this.
    if "truth_fc" not in d:
        sys.exit("truth_fc not in the tensor — run ml/fetch_truth.py first")
    fc_raw = {int(ym): v for ym, v in d["truth_fc"]}
    fc = np.full(len(months), np.nan)
    for ym, v in fc_raw.items():
        if ym in month_of_ym:
            fc[month_of_ym[ym]] = v
    # deseasonalise the cable on its own climatology
    ok = np.isfinite(fc)
    fclim = np.array([np.nanmean(fc[ok & (moy == m)]) for m in range(12)])
    fc_des = fc - fclim[moy]

    # months where BOTH RAPID and the cable exist — the honest common sample
    have = np.isfinite(fc_des[tidx])
    print(f"RAPID months: {len(tidx)}; with cable: {int(have.sum())}")

    rows = {}

    def score(name, F, sel, note):
        r, lo, hi, n, rmse, sigma, pred = kfold_r(F[sel], y[sel], yr[tidx][sel])
        lp = lowpass_r(tidx[sel], pred, y[sel])
        rows[name] = {"r_kfold_deseas": round(float(r), 3),
                      "ci95": [round(float(lo), 3), round(float(hi), 3)],
                      "n": int(n), "rmse_sv": round(float(rmse), 2),
                      "r_lowpass18": None if lp is None else round(float(lp), 3),
                      "probe": "pooled-ridge", "pooled": True,
                      "inputs": note}
        print(f"  {name:<12} r={r:+.3f} [{lo:+.3f},{hi:+.3f}]  RMSE {rmse:.2f} Sv  "
              f"n={n}  18mo-lowpass {lp if lp is None else round(lp,3)}   ({note})")

    print("\nyear-blocked k-fold, monthly, deseasonalised RAPID — all rows, one protocol:")
    W = wind[tidx]
    score("wind", W, have, "tau_x, tau_y section means (our standing baseline)")
    score("cable", fc_des[tidx][:, None], have, "Florida Current transport ALONE")
    score("cable+wind", np.hstack([fc_des[tidx][:, None], W]), have,
          "the FW2015 input set minus altimetry — two of RAPID's three terms, measured")

    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    print("\nALL ROWS ABOVE ARE SECTION-POOLED, and they are matched to each "
          "other on purpose:\nthe ladder holds the read-out fixed and varies "
          "only the inputs. Do NOT read any of\nthem as a bar for "
          "ml/probe_head.py's unpooled numbers — that comparison would "
          "credit\nthe read-out's gain to the model (ml/CLAUDE.md §3, "
          "2026-08-21). The matched unpooled\nwind bar is "
          "`ml/probe_head.py --raw --wind-only`.")
    json.dump({"protocol": "year-blocked k-fold, monthly, deseasonalised, "
                           "block-bootstrap CI — identical to probe_kfold.py",
               "pooled": True,
               "pooled_note": "Every row is a section MEAN through the same "
                              "ridge. The ladder is internally matched and "
                              "that is what it measures; it is NOT a bar for "
                              "the unpooled attention head, and a table that "
                              "puts them in one column is comparing "
                              "read-outs. See ml/CLAUDE.md §3 (2026-08-21).",
               "target": "RAPID 26.5N",
               "note": "The cable is normally a TRUTH series here; it is used as an "
                       "input in these rows only, to reproduce the classical input "
                       "set under our protocol (paper section 6.1).",
               "rows": rows}, open(a.out, "w"), indent=1)
    print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()

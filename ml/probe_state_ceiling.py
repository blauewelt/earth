"""The ceiling decomposition: is 0.631 an encoder tax or a state/label limit?

Chris, 2026-08-12: "at the current best decoder level, what would be the new
decoder induced ceiling? (what used to be 0.631)". The decoder was never in
the 0.631 chain (the probe reads z directly), so the honest answer is a
DECOMPOSITION: run the exact probe_kfold protocol (year-blocked k-fold,
per-fold inner-split ridge-lambda selection, deseasonalised RAPID, block
bootstrap) over feature sets that bracket every stage of the pipeline:

    wind-only (raw tau, the literature's bar)      — protocol check
    pooled z (64f)                                 — the 0.631 instrument
    pooled TRUE standardized fields (39f)          — decoder = identity
    5-segment pooled z / true fields               — mild spatial structure
    MLP probe on pooled z / true fields            — nonlinear read-out

Measured 2026-08-12 (sandbox, f16 Z cache, protocol checks reproduced the
published 0.631 -> 0.627 and wind 0.568 exactly):

    wind-only                     0.568 [0.428, 0.696]
    ridge pooled z                0.627 [0.503, 0.735]
    ridge pooled TRUE fields      0.631 [0.496, 0.746]
    ridge 5-seg z (320f)          0.646 [0.539, 0.741]
    ridge 5-seg TRUE (195f)       0.653 [0.549, 0.741]
    MLP  pooled z                 0.611 [0.490, 0.716]
    MLP  pooled TRUE fields       0.530 [0.364, 0.676]

Reading: z matches the TRUE state at every read-out (and beats it under the
MLP — the compression is more learnable at n=240). The "0.631 ceiling" is
therefore NOT encoder-derived and NOT decoder-derived: it is what THIS
monthly-mean section state yields to a small-sample read-out at 240 labelled
months. Nonlinearity does not help (MLP <= ridge at this n); resolving zonal
structure helps mildly (~+0.02, inside the CIs). See EXPERIMENTS.md E-019.

Usage (expects the E-019b1 cache layout under ml/cache/):
  python3 ml/probe_state_ceiling.py [--seg N] [--mlp]
"""
import argparse
import os
import sys
import warnings

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from probe_kfold import kfold_r                             # noqa: E402
from recon_eval import build_slab, RAPID_LAT, RAPID_LON     # noqa: E402
from temporal import section_of                             # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache-dir", default=os.path.join(HERE, "cache"))
    ap.add_argument("--seg", type=int, default=5)
    ap.add_argument("--mlp", action="store_true",
                    help="also run the (slow) MLP probes")
    a = ap.parse_args()
    cd = a.cache_dir

    d = np.load(os.path.join(cd, "f3_small.npz"), allow_pickle=False)
    months = [str(m) for m in d["months"]]
    lats, lons = d["lats"], d["lons"]
    chan = [str(c) for c in d["chan"]]
    moy = np.array([int(m[5:7]) - 1 for m in months])
    yr = np.array([int(m[:4]) for m in months])
    Xm = np.load(os.path.join(cd, "family3_X.npy"), mmap_mode="r")
    T, H, W, C = Xm.shape
    s = np.load(os.path.join(cd, "std_stats.npz"))
    clim, dyn = s["clim"], list(s["dyn"])
    mean_c, std_c = s["mean_c"], s["std_c"]

    sec_y = int(np.argmin(np.abs(lats - RAPID_LAT)))
    rows = [sec_y - 1, sec_y, sec_y + 1]
    slab, obs = build_slab(Xm, rows, moy, clim, dyn, mean_c, std_c)
    ocean_row = obs[:, 1, :, 0].any(axis=0)
    xs_all = np.where(ocean_row)[0]
    _, sel = section_of(lats[rows], lons, np.ones(len(xs_all), dtype=int),
                        xs_all, RAPID_LAT, *RAPID_LON)
    xs_sec = xs_all[sel]
    ocean = np.load(os.path.join(cd, "ocean_mask.npy"))
    ys_f, xs_f = np.where(ocean)
    lin = np.zeros((H, W), np.int64)
    lin[ys_f, xs_f] = np.arange(len(ys_f))
    sec_pidx = lin[sec_y, xs_sec]
    Zm = np.load(os.path.join(cd, "Z_run62.npy"), mmap_mode="r")
    Z_sec = np.asarray(Zm[:, sec_pidx]).astype(np.float32)
    truth = slab[:, 1][:, xs_sec]

    arr = d["rapid"]
    tidx = arr[:, 0].astype(int)
    vals = arr[:, 1].copy()
    tmoy = moy[tidx]
    tcl = np.array([vals[tmoy == m].mean() for m in range(12)])
    v_des = vals - tcl[tmoy]

    def report(tag, F, probe="ridge"):
        r, lo, hi, n, rmse, sigma, _ = kfold_r(F[tidx], v_des, yr[tidx],
                                               probe=probe)
        print(f"{tag:<46} r={r:.3f} [{lo:.3f},{hi:.3f}]  rmse={rmse:.2f} Sv",
              flush=True)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        F_true = np.nan_to_num(np.nanmean(truth, axis=1), nan=0.0)
        wsel = [i for i, c in enumerate(chan) if c in ("tau_x", "tau_y")]
        raw = np.asarray(Xm[:, sec_y, :, :][:, xs_sec][:, :, wsel])
        F_w = np.nan_to_num(np.nanmean(raw, axis=1), nan=0.0)
    F_z = Z_sec.mean(axis=1)

    report("wind-only baseline (raw tau)", F_w)
    report("ridge, pooled z (the 0.631 instrument)", F_z)
    report("ridge, pooled TRUE fields (decoder=identity)", F_true)

    nseg = a.seg
    bounds = np.linspace(0, len(xs_sec), nseg + 1).astype(int)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        Fseg_z = np.concatenate(
            [Z_sec[:, bounds[i]:bounds[i + 1]].mean(axis=1)
             for i in range(nseg)], axis=1)
        Fseg_t = np.concatenate(
            [np.nan_to_num(np.nanmean(truth[:, bounds[i]:bounds[i + 1]],
                                      axis=1), nan=0.0)
             for i in range(nseg)], axis=1)
    report(f"ridge, {nseg}-segment z ({Fseg_z.shape[1]}f)", Fseg_z)
    report(f"ridge, {nseg}-segment TRUE ({Fseg_t.shape[1]}f)", Fseg_t)
    if a.mlp:
        report("MLP, pooled z", F_z, probe="mlp")
        report("MLP, pooled TRUE fields", F_true, probe="mlp")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""E-025 scoping: is external FORCING distinguishable from a trend?

Chris, 2026-08-14: *"How should the co2 / the energy balance be modeled as
part of the prediction?"*

The naive move is to feed CO2 (or the AR6 effective radiative forcing) into
stage 2 as a covariate. Over 1982-2024 those series are nearly monotone, so
the degeneracy has to be named BEFORE any GPU is spent (ml/CLAUDE.md §4.9b:
a degeneracy you can name is one you must close or measure): **a model given
CO2 can simply learn the calendar.** It would fit the warming trend in the
training years, score well on held-out years scattered *inside* that span,
and be indistinguishable from the memorisation E-021 already caught — while
looking like physics.

So this script measures three things, none of which needs a GPU:

  1. COLLINEARITY — r(series, linear time) over the model's training window.
     A forcing series with |r| ≈ 1 carries no information a time index does
     not, and calling it "forcing" would be a naming choice, not a finding.
  2. SIGNAL — is there anything for forcing to explain? Regress the
     global-mean and RAPID-section-mean embedding series on time, and see
     how much variance a trend already takes.
  3. INCREMENTAL INFORMATION BEYOND TREND — for each series, the extra
     held-out variance it explains in next-month embeddings AFTER a linear
     time index is already in the model. This is the number that decides
     whether a forcing input is a physical covariate or a relabelled clock.

Controls are the point of the design: `time` (a bare linear index) and
`shuffled` (each series' values permuted across years, destroying its
temporal alignment while preserving its distribution).

  python3 ml/measure_forcing_info.py --z ml/cache/Z_run62.npy \
      --npz-small ml/cache/f3_small.npz --out ml/runs/forcing_info.json
"""
import argparse
import json
import os
import sys
import urllib.request

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "ml"))

try:
    from measure_ring_info import ridge_fit_eval         # noqa: E402
except ModuleNotFoundError:                              # run from repo root
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from measure_ring_info import ridge_fit_eval         # noqa: E402

CO2_URL = "https://gml.noaa.gov/webdata/ccgg/trends/co2/co2_mm_mlo.csv"
ONI_URL = "https://psl.noaa.gov/data/correlation/oni.data"
SUN_URL = "https://www.sidc.be/SILSO/DATA/SN_m_tot_V2.0.txt"


def fetch(url, path):
    if not os.path.exists(path):
        with urllib.request.urlopen(url, timeout=120) as r, open(path, "wb") as f:
            f.write(r.read())
    return path


def load_co2(cache):
    """Mauna Loa monthly: (ym -> average, ym -> deseasonalised)."""
    avg, des = {}, {}
    for line in open(fetch(CO2_URL, cache)):
        if line.startswith("#") or line.startswith("year"):
            continue
        p = line.split(",")
        if len(p) < 5:
            continue
        ym = f"{int(p[0]):04d}-{int(p[1]):02d}"
        if float(p[3]) > 0:
            avg[ym] = float(p[3])
        if float(p[4]) > 0:
            des[ym] = float(p[4])
    return avg, des


def load_oni(cache):
    """NOAA PSL ONI: 3-month running Niño-3.4 anomaly, monthly."""
    out = {}
    for line in open(fetch(ONI_URL, cache)):
        p = line.split()
        if len(p) != 13 or not p[0].isdigit():
            continue
        for m, v in enumerate(p[1:], 1):
            v = float(v)
            if v > -90:
                out[f"{int(p[0]):04d}-{m:02d}"] = v
    return out


def load_sun(cache):
    """SILSO monthly mean sunspot number — the solar-cycle proxy."""
    out = {}
    for line in open(fetch(SUN_URL, cache)):
        p = line.split()
        if len(p) < 4:
            continue
        try:
            y, m, v = int(p[0]), int(p[1]), float(p[3])
        except ValueError:
            continue
        if v >= 0:
            out[f"{y:04d}-{m:02d}"] = v
    return out


def load_erf(path):
    """AR6 effective radiative forcing, YEARLY, from the app's own bake."""
    d = json.load(open(path))
    return ({int(y): float(v) for y, v in zip(d["erf_years"], d["erf_anthro"])},
            {int(y): float(v) for y, v in zip(d["erf_years"], d["erf_natural"])})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--z", required=True)
    ap.add_argument("--npz-small", required=True)
    ap.add_argument("--ocean-mask", default=os.path.join(HERE, "cache",
                                                         "ocean_mask.npy"))
    ap.add_argument("--eei", default=os.path.join(HERE, "..", "data", "eei.json"))
    ap.add_argument("--cache-dir", default="/tmp")
    ap.add_argument("--pixels", type=int, default=600)
    ap.add_argument("--lags", type=int, default=3)
    ap.add_argument("--holdout-years", default="2009,2017,2023")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=os.path.join(HERE, "runs",
                                                  "forcing_info.json"))
    a = ap.parse_args()

    d = np.load(a.npz_small, allow_pickle=False)
    months = [str(m) for m in d["months"]]
    T = len(months)
    moy = np.array([int(m[5:7]) - 1 for m in months])
    yr = np.array([int(m[:4]) for m in months])
    hold = set(int(y) for y in a.holdout_years.split(","))
    train_t = np.array([y not in hold for y in yr])

    co2_avg, co2_des = load_co2(os.path.join(a.cache_dir, "co2_mm_mlo.csv"))
    oni = load_oni(os.path.join(a.cache_dir, "oni.data"))
    sun = load_sun(os.path.join(a.cache_dir, "SN_m_tot_V2.0.txt"))
    erf_a, erf_n = load_erf(a.eei)

    def series(get, name):
        v = np.array([get(m) for m in months], float)
        n_missing = int(np.isnan(v).sum())
        if n_missing:
            ok = ~np.isnan(v)
            v = np.interp(np.arange(T), np.where(ok)[0], v[ok])
        return v, n_missing

    S = {}
    S["co2"], m1 = series(lambda m: co2_avg.get(m, np.nan), "co2")
    S["co2_deseason"], _ = series(lambda m: co2_des.get(m, np.nan), "co2d")
    S["oni"], m2 = series(lambda m: oni.get(m, np.nan), "oni")
    S["sunspots"], m3 = series(lambda m: sun.get(m, np.nan), "sun")
    S["erf_anthro"], _ = series(lambda m: erf_a.get(int(m[:4]), np.nan), "erfa")
    S["erf_natural"], _ = series(lambda m: erf_n.get(int(m[:4]), np.nan), "erfn")
    print(f"months {months[0]}..{months[-1]} (T={T}); missing filled: "
          f"co2 {m1}, oni {m2}, sun {m3}", flush=True)

    tt = (np.arange(T) - T / 2) / (T / 2)          # the CONTROL: linear time
    S["time"] = tt
    rng = np.random.default_rng(a.seed)
    # shuffled control: same values, alignment destroyed, in whole YEARS so
    # the seasonal cycle inside a year survives and only the year-to-year
    # ordering is broken
    yrs = sorted(set(yr))
    perm = {y: p for y, p in zip(yrs, rng.permutation(yrs))}
    idx_sh = np.concatenate([np.where(yr == perm[y])[0][:12] if
                             (yr == perm[y]).sum() >= 12 else np.where(yr == y)[0]
                             for y in yrs])[:T]
    S["co2_shuffled"] = S["co2"][idx_sh]

    results = {"months": [months[0], months[-1]], "T": T,
               "holdout_years": sorted(hold), "collinearity": {}, "beyond_trend": {}}

    print("\n1. COLLINEARITY with linear time (train months only)")
    for k, v in S.items():
        r = float(np.corrcoef(v[train_t], tt[train_t])[0, 1])
        results["collinearity"][k] = round(r, 4)
        flag = "  <-- indistinguishable from a clock" if abs(r) > 0.97 else ""
        print(f"   {k:<16} r(t) {r:+.4f}{flag}")

    # ---- 2/3. signal, and information beyond a trend ----------------------
    ocean = np.load(a.ocean_mask)
    ys, xs = np.where(ocean)
    P = len(ys)
    keep = np.sort(rng.choice(P, min(a.pixels, P), replace=False))
    Zm = np.load(a.z, mmap_mode="r")
    dz = Zm.shape[2]
    Zs = np.empty((T, len(keep), dz), np.float32)
    for t0 in range(0, T, 16):
        t1 = min(t0 + 16, T)
        Zs[t0:t1] = np.asarray(Zm[t0:t1])[:, keep].astype(np.float32)
    for m in range(12):
        tr = train_t & (moy == m)
        Zs[moy == m] -= Zs[tr].mean(0)
    gm = Zs.mean(1)                                        # [T, dz] global mean
    print(f"\n2. SIGNAL in the global-mean embedding ({len(keep)} px)")
    for k in ("time", "co2", "oni", "erf_anthro"):
        rs = [abs(np.corrcoef(S[k][train_t], gm[train_t, j])[0, 1])
              for j in range(dz)]
        print(f"   |r| with {k:<12} max {max(rs):.3f}  mean {np.mean(rs):.3f}")

    L = a.lags
    idx = np.arange(L - 1, T - 1)
    cols, targ, split = [], [], []
    for i in range(len(keep)):
        c = Zs[:, i]
        cols.append(np.concatenate([c[idx - j] for j in range(L)], 1))
        targ.append(c[idx + 1])
        split.append(train_t[idx + 1])
    Xb = np.concatenate(cols)
    Y = np.concatenate(targ)
    tr = np.concatenate(split)
    n_c = len(keep)

    def with_feats(names):
        f = np.stack([S[n][idx + 1] for n in names], 1)     # forcing at target
        f = (f - f[train_t[idx + 1]].mean(0)) / (f.std(0) + 1e-9)
        return np.concatenate([Xb, np.tile(f, (n_c, 1))], 1)

    mse_b, _ = ridge_fit_eval(Xb[tr], Y[tr], Xb[~tr], Y[~tr])
    print(f"\n3. INFORMATION BEYOND THE CENTRE'S OWN HISTORY "
          f"(baseline held-out MSE {mse_b:.5f})")
    combos = [["time"], ["co2"], ["co2_shuffled"], ["oni"], ["sunspots"],
              ["erf_anthro"], ["co2", "oni"], ["time", "co2"], ["time", "oni"],
              ["time", "co2", "oni", "sunspots", "erf_anthro"]]
    for names in combos:
        X = with_feats(names)
        mse, lam = ridge_fit_eval(X[tr], Y[tr], X[~tr], Y[~tr])
        g = 1 - mse / mse_b
        results["beyond_trend"]["+".join(names)] = {
            "mse": round(mse, 6), "gain": round(float(g), 5)}
        print(f"   {'+'.join(names):<38} gain {g:+.5f}")
        with open(a.out, "w") as fh:
            json.dump(results, fh, indent=1)

    # the decisive contrast: CO2 given time, vs time alone
    gt = results["beyond_trend"]["time"]["gain"]
    gtc = results["beyond_trend"]["time+co2"]["gain"]
    results["co2_beyond_time"] = round(gtc - gt, 5)
    print(f"\n   CO2's contribution BEYOND a linear clock: {gtc - gt:+.5f}")
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    with open(a.out, "w") as fh:
        json.dump(results, fh, indent=1)
    print("wrote", a.out)


if __name__ == "__main__":
    main()

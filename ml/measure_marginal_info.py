#!/usr/bin/env python3
"""How much information does each ADDITIONAL input point buy?

Chris, 2026-08-14: *"In theory it would be the more information the better
(and redundant does not hurt). Maybe the actual question is how much
information per input point — that's what we want to maximize."*

Right on both counts, and the earlier scripts answered neither question.

`measure_shape_info.py` compared whole hand-picked shapes, so its numbers
mixed the information a shape carries with the estimation cost of its width;
at 120 centres the cost dominated and made wide shapes look harmful. That was
the instrument, not the ocean (E-024 CORRECTION). But the cost is not
imaginary either: E-022 measured 9 and 13 touching neighbours coming out 6.3
and 8.1 seed sd WORSE than none in the real transformer, which
`test_zero_weight_equivalence` proves could have ignored them exactly. A
redundant input is free in theory and expensive in practice, so the quantity
to maximise is the one Chris names: gain per point.

This measures it by GREEDY FORWARD SELECTION. Start from the centre pixel's
own history; repeatedly add whichever single candidate position most reduces
held-out error; record what each addition is worth. That curve IS information
per input point — its height answers *how many points*, and the positions
chosen answer *which radii*, including whether a second ring is ever
preferred to another bearing on the first.

Candidates: 6 radii x 8 bearings, circles on the ground
(temporal.ring_offsets), so 222 km and 555 km compete head to head at every
step instead of being compared as pre-built shapes.

Method note: everything is linear, so ONE pass over the cache accumulates the
Gram matrices (X'X, X'Y) over train / val / test rows, and every later ridge
fit is a submatrix solve. No design matrix is ever materialised - 300 centres
x 512 months x 3,265 columns would be 2 GB.

  python3 ml/measure_marginal_info.py --z ml/cache/Z_run62.npy \
      --npz-small ml/cache/f3_small.npz --centres 300 --select 12
"""
import argparse
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from temporal import ring_offsets                       # noqa: E402

RADII = [111.0, 222.0, 333.0, 445.0, 555.0, 890.0]
BEARINGS = 8
LAMS = (1.0, 10.0, 100.0, 1000.0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--z", required=True)
    ap.add_argument("--npz-small", required=True)
    ap.add_argument("--ocean-mask", default=os.path.join(HERE, "cache",
                                                         "ocean_mask.npy"))
    ap.add_argument("--centres", type=int, default=300)
    ap.add_argument("--select", type=int, default=12)
    ap.add_argument("--lags", type=int, default=3)
    ap.add_argument("--holdout-years", default="2009,2017,2023")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=os.path.join(HERE, "runs",
                                                  "marginal_info.json"))
    a = ap.parse_args()

    d = np.load(a.npz_small, allow_pickle=False)
    months = [str(m) for m in d["months"]]
    lats = d["lats"]
    moy = np.array([int(m[5:7]) - 1 for m in months])
    yr = np.array([int(m[:4]) for m in months])
    hold = set(int(y) for y in a.holdout_years.split(","))
    train_t = np.array([y not in hold for y in yr])
    dlat = float(np.round(np.diff(lats).mean(), 6))

    ocean = np.load(a.ocean_mask)
    ys, xs = np.where(ocean)
    P = len(ys)
    lin = np.full(ocean.shape, -1, np.int64)
    lin[ys, xs] = np.arange(P)
    Zm = np.load(a.z, mmap_mode="r")
    T, _, dz = Zm.shape

    rng = np.random.default_rng(a.seed)
    margin = 36
    ok = ((ys > margin) & (ys < ocean.shape[0] - margin)
          & (xs > margin) & (xs < ocean.shape[1] - margin))
    centres = rng.choice(np.where(ok)[0], min(a.centres, int(ok.sum())),
                         replace=False)

    cand = [(r, b) for r in RADII for b in range(BEARINGS)]
    cidx = {c: i for i, c in enumerate(cand)}
    NB = np.full((len(centres), len(cand)), -1, np.int64)
    for i, p in enumerate(centres):
        y, x = ys[p], xs[p]
        for r in RADII:
            for b, (dy, dx) in enumerate(ring_offsets(float(lats[y]), r,
                                                      BEARINGS, dlat)):
                yy, xx = y + dy, x + dx
                if 0 <= yy < ocean.shape[0] and 0 <= xx < ocean.shape[1]:
                    NB[i, cidx[(r, b)]] = lin[yy, xx]
    print(f"{len(centres)} centres · {len(cand)} candidates "
          f"({len(RADII)} radii x {BEARINGS} bearings)", flush=True)

    all_px = np.unique(np.concatenate([centres, NB[NB >= 0].ravel()]))
    pos = np.full(P, -1, np.int64)
    pos[all_px] = np.arange(len(all_px))
    Z = np.empty((T, len(all_px), dz), np.float32)
    for t0 in range(0, T, 16):
        t1 = min(t0 + 16, T)
        Z[t0:t1] = np.asarray(Zm[t0:t1])[:, all_px].astype(np.float32)
    for m in range(12):
        tr = train_t & (moy == m)
        Z[moy == m] -= Z[tr].mean(0)
    print("read complete", flush=True)

    L = a.lags
    idx = np.arange(L - 1, T - 1)
    is_tr = train_t[idx + 1]
    ci = pos[centres]
    H = L * dz
    D = H + len(cand) * dz + 1
    n_fit = int(0.8 * len(centres))

    G = {k: np.zeros((D, D)) for k in ("fit", "val", "all", "test")}
    B = {k: np.zeros((D, dz)) for k in ("fit", "val", "all", "test")}
    yy_test = 0.0
    n_test = 0
    for i in range(len(centres)):
        c = Z[:, ci[i]]
        blocks = [c[idx - j] for j in range(L)]
        for j in range(len(cand)):
            q = NB[i, j]
            blocks.append(Z[:, pos[q]][idx] if q >= 0
                          else np.zeros((len(idx), dz), np.float32))
        blocks.append(np.ones((len(idx), 1), np.float32))
        X = np.concatenate(blocks, 1)
        Y = c[idx + 1]
        xt, yt = X[is_tr], Y[is_tr]
        xe, ye = X[~is_tr], Y[~is_tr]
        gt, bt = xt.T @ xt, xt.T @ yt
        G["all"] += gt; B["all"] += bt
        G["fit" if i < n_fit else "val"] += gt
        B["fit" if i < n_fit else "val"] += bt
        G["test"] += xe.T @ xe; B["test"] += xe.T @ ye
        yy_test += float((ye ** 2).sum()); n_test += ye.size
        del X, Y, blocks

    def cols(sel):
        c = list(range(H))
        for j in sel:
            c += list(range(H + j * dz, H + (j + 1) * dz))
        return np.array(c + [D - 1])

    def fit_mse(sel):
        c = cols(sel)
        gf, bf = G["fit"][np.ix_(c, c)], B["fit"][c]
        gv, bv = G["val"][np.ix_(c, c)], B["val"][c]
        ga, ba = G["all"][np.ix_(c, c)], B["all"][c]
        gt, bt = G["test"][np.ix_(c, c)], B["test"][c]
        best, bl = np.inf, LAMS[0]
        for lam in LAMS:
            reg = lam * np.eye(len(c)); reg[-1, -1] = 0
            W = np.linalg.solve(gf + reg, bf)
            sse = float(np.trace(W.T @ gv @ W) - 2 * np.trace(W.T @ bv))
            if sse < best:
                best, bl = sse, lam
        reg = bl * np.eye(len(c)); reg[-1, -1] = 0
        W = np.linalg.solve(ga + reg, ba)
        sse = (yy_test - 2 * float(np.trace(W.T @ bt))
               + float(np.trace(W.T @ gt @ W)))
        return sse / n_test, bl

    base, _ = fit_mse([])
    print(f"\nbaseline (own {L}-month history): held-out MSE {base:.5f}")
    print(f"{'k':>2} {'position added':>18} {'total gain':>11} {'MARGINAL':>10}")
    results = {"centres": len(centres), "radii_km": RADII,
               "bearings": BEARINGS, "mse_baseline": round(base, 6),
               "steps": []}
    sel, prev = [], base
    for k in range(1, a.select + 1):
        best_j, best_mse = None, np.inf
        for j in range(len(cand)):
            if j in sel:
                continue
            m, _ = fit_mse(sel + [j])
            if m < best_mse:
                best_mse, best_j = m, j
        sel.append(best_j)
        r, b = cand[best_j]
        marg = (prev - best_mse) / base
        tot = 1 - best_mse / base
        results["steps"].append(
            {"k": k, "r_km": r, "bearing_deg": b * 360 // BEARINGS,
             "mse": round(best_mse, 6), "total_gain": round(float(tot), 5),
             "marginal_gain": round(float(marg), 5)})
        print(f"{k:>2} {f'{r:.0f} km @ {b * 360 // BEARINGS:>3}deg':>18} "
              f"{tot:>+11.4f} {marg:>+10.4f}", flush=True)
        prev = best_mse
        os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
        with open(a.out, "w") as f:
            json.dump(results, f, indent=1)

    # ---- named reference sets, scored in THIS run so the comparison with
    # the greedy curve is paired rather than across samples ----------------
    refs = {"uniform ring8@222": [cidx[(222.0, b)] for b in range(BEARINGS)],
            "uniform ring8@333": [cidx[(333.0, b)] for b in range(BEARINGS)],
            "uniform ring8@555": [cidx[(555.0, b)] for b in range(BEARINGS)],
            "two rings 222+555": ([cidx[(222.0, b)] for b in range(BEARINGS)]
                                  + [cidx[(555.0, b)] for b in range(BEARINGS)]),
            "greedy top-3": sel[:3], "greedy top-4": sel[:4],
            "greedy top-6": sel[:6], "greedy top-8": sel[:8]}
    results["references"] = {}
    print(f"\n{'set':<20} {'points':>6} {'gain':>9} {'per point':>11}")
    for name, ss in refs.items():
        m, _ = fit_mse(ss)
        g = 1 - m / base
        results["references"][name] = {"n": len(ss), "gain": round(float(g), 5),
                                       "per_point": round(float(g / len(ss)), 6)}
        print(f"{name:<20} {len(ss):>6} {g:>+9.4f} {g / len(ss):>+11.5f}",
              flush=True)

    m_all, _ = fit_mse(list(range(len(cand))))
    results["all_candidates"] = {"n": len(cand),
                                 "gain": round(float(1 - m_all / base), 5)}
    print(f"\nALL {len(cand)} positions at once: gain {1 - m_all / base:+.4f}")
    with open(a.out, "w") as f:
        json.dump(results, f, indent=1)
    print("wrote", a.out)


if __name__ == "__main__":
    main()

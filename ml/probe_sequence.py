#!/usr/bin/env python3
"""Design-B litmus test, no retraining: does COMBINING existing snapshot
embeddings across time beat a single month's embedding at reading RAPID?

Loads the trained PixelMAE checkpoint, embeds the 26.5N section for every
month, then fits ridge readouts on (a) the current month's embedding and
(b) the last K months' embeddings stacked. Lambda is chosen on a validation
split carved from TRAIN years; held-out years are touched exactly once.
"""
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model import PixelMAE

HERE = os.path.dirname(os.path.abspath(__file__))


def main():
    ck = torch.load(os.path.join(HERE, "runs", "pilot12", "pixelmae.pt"),
                    map_location="cpu", weights_only=False)
    d = np.load(os.path.join(HERE, "cache", "na_pixels.npz"))
    X, months = d["X"], [str(m) for m in d["months"]]
    lats, lons = d["lats"], d["lons"]
    T, H, W, C = X.shape
    model = PixelMAE(n_chan=C, d_z=ck["d_z"])
    model.load_state_dict(ck["model"])
    model.eval()

    hold_years = set(ck["args"]["holdout_years"].split(","))
    t_hold = np.array([m[:4] in hold_years for m in months])
    mvec = np.array([int(m[5:7]) - 1 for m in months])
    ctx_all = np.stack([np.sin(2 * np.pi * mvec / 12), np.cos(2 * np.pi * mvec / 12)], 1)

    sec_y = int(np.argmin(np.abs(lats - 26.5)))
    sec_x = np.where(np.isfinite(X[0, sec_y, :, 0]))[0]
    Xt = torch.from_numpy(np.nan_to_num(X, nan=0.0))
    OBS = torch.from_numpy(np.isfinite(X))

    print(f"embedding the 26.5N section for {T} months ...")
    emb = np.zeros((T, ck["d_z"]), dtype=np.float32)
    with torch.no_grad():
        for t in range(T):
            n = len(sec_x)
            ctx = np.concatenate([np.tile(ctx_all[t], (n, 1)),
                                  (np.full(n, lats[sec_y]) / 90)[:, None],
                                  (lons[sec_x] / 180)[:, None]], 1)
            v = Xt[t, sec_y, sec_x]
            o = OBS[t, sec_y, sec_x]
            z = model.encode(v, o, torch.zeros_like(o),
                             torch.as_tensor(ctx, dtype=torch.float32))
            emb[t] = z.mean(0).numpy()

    rapid = d["rapid"]
    ridx = rapid[:, 0].astype(int)
    rv = rapid[:, 1]

    def ridge_eval(K):
        """Features = embeddings of months t-K+1..t, stacked."""
        ok = ridx >= K - 1
        ri, y = ridx[ok], rv[ok]
        F = np.stack([np.concatenate([emb[t - k] for k in range(K)]) for t in ri])
        F = (F - F.mean(0)) / (F.std(0) + 1e-9)
        tr, te = ~t_hold[ri], t_hold[ri]
        # lambda picked on a tail of the TRAIN months, never on held-out years
        order = np.argsort(ri[tr])
        tr_idx = np.where(tr)[0][order]
        fit, val = tr_idx[: int(0.8 * len(tr_idx))], tr_idx[int(0.8 * len(tr_idx)):]
        def solve(idx, lam):
            A = np.c_[F[idx], np.ones(len(idx))]
            reg = lam * np.eye(A.shape[1]); reg[-1, -1] = 0
            return np.linalg.solve(A.T @ A + reg, A.T @ y[idx])
        best, best_r = None, -np.inf
        for lam in (1e-2, 1e-1, 1, 10, 100, 1000):
            w = solve(fit, lam)
            p = np.c_[F[val], np.ones(len(val))] @ w
            r = np.corrcoef(p, y[val])[0, 1]
            if r > best_r:
                best_r, best = r, lam
        w = solve(np.where(tr)[0], best)
        p = np.c_[F, np.ones(len(F))] @ w
        return {"K": K, "lambda": best,
                "r_train": round(float(np.corrcoef(p[tr], y[tr])[0, 1]), 3),
                "r_heldout": round(float(np.corrcoef(p[te], y[te])[0, 1]), 3),
                "n_test": int(te.sum())}

    out = [ridge_eval(K) for K in (1, 3, 6, 12, 24)]
    print(json.dumps(out, indent=2))
    json.dump(out, open(os.path.join(HERE, "runs", "pilot12", "probe_sequence.json"), "w"),
              indent=2)


if __name__ == "__main__":
    main()

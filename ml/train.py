#!/usr/bin/env python3
"""Train PixelMAE on the North-Atlantic pixel tensor.

Splits are BLOCKED, not random (proposal §7): whole held-out YEARS and a
held-out lon/lat block, so spatial/temporal autocorrelation cannot fake skill.

Eval after training:
  · masked-channel reconstruction error on held-out years   (vs channel-mean)
  · temporal neighbour prediction (t+1) error               (vs persistence)
  · RAPID probe: ridge regression from the mean embedding of the 26.5N
    section to the RAPID overturning transport, fit on train years, scored
    (Pearson r) on held-out years. The transport was NEVER a channel.

Smoke (CPU, ~2 min):   python3 ml/train.py --smoke
Real  (Colab TPU/GPU): colab run --gpu v6e1 ml/train.py   (see ml/README.md)
"""
import argparse
import json
import os
import time

import numpy as np
import torch

from model import PixelMAE, gather_px

HERE = os.path.dirname(os.path.abspath(__file__))


def parse():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default=os.path.join(HERE, "cache", "na_pixels.npz"))
    p.add_argument("--out", default=os.path.join(HERE, "runs", "pilot"))
    p.add_argument("--steps", type=int, default=20000)
    p.add_argument("--batch", type=int, default=512)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--d-z", type=int, default=32)
    p.add_argument("--patch", type=int, default=1, choices=[1, 3],
                   help="encoder receptive field per channel token (3 = 3x3)")
    p.add_argument("--d-model", type=int, default=128,
                   help="encoder width (128x4 = the 0.92M pilot; 320x8 = ~10M "
                        "-- the Chinchilla-anchored size for the C=24 global "
                        "tensor: ~270M observed values / 20 ~ 13M params)")
    p.add_argument("--n-layers", type=int, default=4)
    p.add_argument("--n-heads", type=int, default=4)
    p.add_argument("--d-dec", type=int, default=256,
                   help="decoder width (scale with d_model; 512 at 320x8)")
    p.add_argument("--mask-ratio", type=float, default=0.5)
    p.add_argument("--holdout-years", default="2009,2017,2023")
    p.add_argument("--holdout-lon", default="-45,-25")   # a mid-Atlantic block
    p.add_argument("--anomaly", action="store_true",
                   help="train dynamic channels as departures from their own "
                        "per-pixel monthly climatology (train years only)")
    p.add_argument("--eval-every", type=int, default=0,
                   help="every N steps, measure PREDICTIVE skill of the "
                        "frozen current embeddings (trainprobe.py: linear "
                        "section probe + mini temporal transformer) on the "
                        "blocked holdout; appends to <out>/metrics.jsonl. "
                        "Requires --anomaly.")
    p.add_argument("--lr-floor", type=float, default=0.0,
                   help="decay-then-constant schedule: cosine-decay the LR "
                        "over --lr-decay-steps to floor*peak, then hold it "
                        "there for the rest of the run. 0 (default) keeps "
                        "the pure cosine-to-zero schedule. The constant tail "
                        "exists to answer one question cheaply: is there "
                        "headroom left in simply not stopping? — watch the "
                        "probe curves and abort when they flatten.")
    p.add_argument("--lr-decay-steps", type=int, default=0,
                   help="steps of initial decay when --lr-floor > 0 "
                        "(default 0 = the full run, i.e. plain cosine)")
    p.add_argument("--max-minutes", type=int, default=0,
                   help="wall-clock budget for the TRAINING LOOP (0 = off). "
                        "After a short calibration the cosine schedule is "
                        "re-fitted to the step count that fits, so the LR "
                        "still anneals to zero inside the budget instead of "
                        "the job dying mid-schedule. Exists because Actions "
                        "kills at timeout-minutes with NO checkpoint: run "
                        "#12 (25 channels) measured ~1.3 steps/s against a "
                        "40k-step dispatch — 6 runner-hours, nothing saved.")
    p.add_argument("--smoke", action="store_true")
    return p.parse_args()


def main():
    a = parse()
    if a.smoke:
        a.steps, a.batch = 1500, 256
    os.makedirs(a.out, exist_ok=True)
    dev = ("cuda" if torch.cuda.is_available() else "cpu")
    d = np.load(a.data, allow_pickle=False)
    X, months = d["X"], [str(m) for m in d["months"]]
    lats, lons, chan = d["lats"], d["lons"], [str(c) for c in d["chan"]]
    T, H, W, C = X.shape
    print(f"X [T={T} H={H} W={W} C={C}] on {dev} · channels {chan}")

    # ---- blocked splits ----------------------------------------------------
    hold_years = set(a.holdout_years.split(","))
    lo, hi = (float(v) for v in a.holdout_lon.split(","))
    t_hold = np.array([m[:4] in hold_years for m in months])
    x_hold = (lons >= lo) & (lons < hi)
    ocean = np.isfinite(X[..., 0]).any(axis=0)
    print(f"held-out months {int(t_hold.sum())}/{T} · held-out lon block "
          f"{int(x_hold.sum())}/{W} cols · ocean {int(ocean.sum())}")

    # ---- anomaly space (proposal §5) ---------------------------------------
    # A reconstruction loss on raw state is dominated by the seasonal cycle —
    # the easiest, least interesting signal, and the sequence-probe experiment
    # showed the resulting embeddings are nearly redundant month to month.
    # Subtract each pixel's own monthly climatology (TRAIN years only, so the
    # holdout stays clean) from every DYNAMIC channel; what remains is the
    # anomaly — the part that carries trends, events, and the AMOC story.
    # Channels with no temporal variance (baked climatologies) are context,
    # not targets in disguise: they pass through unchanged.
    if a.anomaly:
        moy = np.array([int(m[5:7]) - 1 for m in months])
        dynamic = [c for c in range(C) if np.nanstd(np.nanmean(X[..., c], axis=(1, 2))) > 1e-6]
        clim = np.full((12, H, W, C), np.nan, dtype=np.float32)
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")           # all-NaN cells are fine
            for m in range(12):
                sel = (moy == m) & ~t_hold
                clim[m] = np.nanmean(X[sel], axis=0)
        for c in dynamic:
            X[..., c] = X[..., c] - clim[moy][..., c]
            v = X[..., c][np.isfinite(X[..., c]) & ~t_hold[:, None, None]
                          & ~x_hold[None, None, :]]
            X[..., c] = (X[..., c] - v.mean()) / (v.std() + 1e-6)
        print(f"anomaly space: {len(dynamic)}/{C} dynamic channels "
              f"({[chan[c] for c in dynamic]})")

    # train pool: any (t, y, x) with ≥2 observed channels, outside holdouts
    obs_any = np.isfinite(X).sum(-1) >= 2
    tt, yy, xx = np.where(obs_any & ~t_hold[:, None, None] & ~x_hold[None, None, :])
    vt, vy, vx = np.where(obs_any & (t_hold[:, None, None] | x_hold[None, None, :]))
    print(f"train pixels {len(tt):,} · held-out pixels {len(vt):,}")

    Xt = torch.from_numpy(np.nan_to_num(X, nan=0.0))
    OBS = torch.from_numpy(np.isfinite(X))
    mvec = np.array([int(m[5:7]) - 1 for m in months])
    ctx_all = np.stack([np.sin(2 * np.pi * mvec / 12), np.cos(2 * np.pi * mvec / 12)], 1)

    def batch(idx_t, idx_y, idx_x, n):
        k = np.random.randint(0, len(idx_t), n)
        t, y, x = idx_t[k], idx_y[k], idx_x[k]
        ctx = np.concatenate([ctx_all[t], (lats[y] / 90)[:, None], (lons[x] / 180)[:, None]], 1)
        return (torch.as_tensor(t), torch.as_tensor(y), torch.as_tensor(x),
                torch.as_tensor(ctx, dtype=torch.float32))

    model = PixelMAE(n_chan=C, d_z=a.d_z, patch=a.patch, d_model=a.d_model,
                     n_layers=a.n_layers, n_heads=a.n_heads, d_dec=a.d_dec).to(dev)
    print(f"codec parameters: {sum(p_.numel() for p_ in model.parameters())/1e6:.2f}M")
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-4)
    # Cosine annealing whose TOTAL can be re-fitted mid-run (--max-minutes):
    # a LambdaLR closure over a mutable total is the same curve as
    # CosineAnnealingLR (eta_min=0) but survives having its denominator
    # changed, which the built-in scheduler's recursive formula does not.
    import math
    sched_total = [a.lr_decay_steps if (a.lr_floor > 0 and a.lr_decay_steps)
                   else a.steps]
    FLOOR = a.lr_floor
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda e: FLOOR + (1 - FLOOR) * 0.5 * (1 + math.cos(
            math.pi * min(e, sched_total[0]) / sched_total[0])))
    huber = torch.nn.HuberLoss(reduction="none")

    # neighbour offsets (Δx, Δy, Δt): 4 spatial + 2 temporal
    NEI = [(1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1)]

    def gather(t, y, x):
        v = Xt[t, y, x].to(dev)
        o = OBS[t, y, x].to(dev)
        return v, o

    def step_loss(t, y, x, ctx):
        B = len(t)
        v, o = gather(t, y, x)
        mask = (torch.rand(B, C, device=dev) < a.mask_ratio) & o
        if a.patch > 1:
            vp, op = gather_px(Xt, OBS, t, y, x, a.patch)
            z = model.encode(vp.to(dev), op.to(dev), mask, ctx.to(dev))
        else:
            z = model.encode(v * (~mask), o, mask, ctx.to(dev))

        # self-reconstruction: all channels queried at offset 0
        qc = torch.arange(C, device=dev)[None, :].expand(B, -1)
        off0 = torch.zeros(B, C, 3, dtype=torch.long, device=dev)
        pred = model.query(z, qc, off0)
        l_rec = huber(pred, v)
        w = mask.float() + 0.1 * (o & ~mask).float()          # masked channels dominate
        l_rec = (l_rec * w).sum() / w.sum().clamp(min=1)

        # neighbours: one random offset per sample
        pick = np.random.randint(0, len(NEI), B)
        dxyz = torch.as_tensor(np.array([NEI[i] for i in pick]), device=dev)
        tn = (t.to(dev) + dxyz[:, 2]).clamp(0, T - 1)
        yn = (y.to(dev) + dxyz[:, 1]).clamp(0, H - 1)
        xn = (x.to(dev) + dxyz[:, 0]).clamp(0, W - 1)
        vn, on = gather(tn.cpu(), yn.cpu(), xn.cpu())
        offn = dxyz[:, None, :].expand(-1, C, -1).long()
        predn = model.query(z, qc, offn)
        l_nei = (huber(predn, vn) * on.float()).sum() / on.float().sum().clamp(min=1)
        return l_rec, l_nei

    if a.eval_every and not a.anomaly:
        raise SystemExit("--eval-every requires --anomaly (trainprobe measures "
                         "anomaly-space embeddings; state space is disqualified)")
    if a.eval_every:
        import trainprobe                      # lazy: plain runs don't need it
    metrics_path = os.path.join(a.out, "metrics.jsonl")
    loss_every = max(1, a.steps // 200)        # the loss curve, cheap to keep

    def save_ckpt():
        torch.save({"model": model.state_dict(), "chan": chan, "d_z": a.d_z,
                    "norm": d["norm"], "args": vars(a)},
                   os.path.join(a.out, "pixelmae.pt"))

    print("training …")
    t0 = time.time()
    CAL = 200                                  # steps before the rate is trusted
    s = 0
    while s < a.steps:
        s += 1
        t, y, x, ctx = batch(tt, yy, xx, a.batch)
        l_rec, l_nei = step_loss(t, y, x, ctx)
        loss = l_rec + 0.5 * l_nei
        opt.zero_grad(); loss.backward(); opt.step(); sched.step()
        if a.max_minutes and CAL and (s >= CAL or time.time() - t0 > 60):
            CAL = 0                            # calibrate once: 200 steps or 60 s
            rate = (time.time() - t0) / s      # s/step, measured not guessed
            budget = a.max_minutes * 60 - (time.time() - t0)
            fit = s + int(0.85 * budget / rate)     # 15% held back for probes
            if fit < a.steps and FLOOR == 0:        # constant-tail runs just hard-stop
                print(f"  time budget: {rate:.2f} s/step → re-fitting the "
                      f"cosine schedule from {a.steps} to {fit} steps so the "
                      f"LR anneals to zero inside {a.max_minutes} min")
                a.steps = fit
                sched_total[0] = fit
        if a.max_minutes and (time.time() - t0) > a.max_minutes * 60:
            print(f"  wall-clock budget reached at step {s} — stopping to save")
            break
        if s % loss_every == 0 or s == a.steps:
            with open(metrics_path, "a") as f:
                f.write(json.dumps({"step": s, "loss_rec": round(l_rec.item(), 5),
                                    "loss_nei": round(l_nei.item(), 5)}) + "\n")
        if s % max(1, a.steps // 10) == 0:
            print(f"  step {s:>6}/{a.steps}  rec {l_rec.item():.4f}  nei {l_nei.item():.4f}"
                  f"  ({time.time() - t0:.0f}s)")
        if a.eval_every and (s % a.eval_every == 0 or s == a.steps):
            m = trainprobe.probe_now(model.cpu(), Xt, OBS, d, mvec, t_hold,
                                     x_hold, dynamic)
            model.to(dev)
            m["step"] = s
            m["wall_s"] = round(time.time() - t0, 1)
            with open(metrics_path, "a") as f:
                f.write(json.dumps(m) + "\n")
            print(f"  probe @{s}: chan t+1 {m['chan_vs_persistence_pct']:+.1f}% "
                  f"vs persistence · linear r_des {m['linear_r_deseas']:+.3f} · "
                  f"temporal r_des {m['temporal_r_deseas']:+.3f} "
                  f"({m['probe_seconds']:.0f}s)", flush=True)
            save_ckpt()                        # crash insurance, ~4 MB

    # ---- evaluation on the BLOCKED holdout --------------------------------
    model.eval()
    results = {}
    with torch.no_grad():
        n_eval = min(20000, len(vt))
        t, y, x, ctx = batch(vt, vy, vx, n_eval)
        v, o = gather(t, y, x)
        mask = (torch.rand(n_eval, C, device=dev) < a.mask_ratio) & o
        if a.patch > 1:
            vp, op = gather_px(Xt, OBS, t, y, x, a.patch)
            z = model.encode(vp.to(dev), op.to(dev), mask, ctx.to(dev))
        else:
            z = model.encode(v * (~mask), o, mask, ctx.to(dev))
        qc = torch.arange(C, device=dev)[None, :].expand(n_eval, -1)
        pred = model.query(z, qc, torch.zeros(n_eval, C, 3, dtype=torch.long, device=dev))
        for c, name in enumerate(chan):
            m = mask[:, c]
            if m.sum() < 50:
                continue
            err = (pred[m, c] - v[m, c]).pow(2).mean().item()
            base = v[m, c].pow(2).mean().item()               # channel mean = 0 after z-score
            results[f"recon/{name}"] = {"mse": err, "mse_channel_mean": base,
                                        "skill": 1 - err / max(base, 1e-9)}

        # temporal neighbour t+1 vs persistence
        t1 = np.clip(t.numpy() + 1, 0, T - 1)
        v1, o1 = gather(torch.as_tensor(t1), y, x)
        off = torch.zeros(n_eval, C, 3, dtype=torch.long, device=dev); off[:, :, 2] = 1
        p1 = model.query(z, qc, off)
        both = (o & o1)
        mse_m = ((p1 - v1).pow(2) * both).sum().item() / both.sum().item()
        mse_p = ((v - v1).pow(2) * both).sum().item() / both.sum().item()
        results["t+1"] = {"mse_model": mse_m, "mse_persistence": mse_p,
                          "beats_persistence": bool(mse_m < mse_p)}

        # ---- RAPID probe ---------------------------------------------------
        rapid = d["rapid"]
        if len(rapid):
            from temporal import RAPID_LON
            sec_y = int(np.argmin(np.abs(lats - 26.5)))
            sec_x = np.where(np.isfinite(X[0, sec_y, :, 0])
                             & (lons >= RAPID_LON[0]) & (lons <= RAPID_LON[1]))[0]
            emb = np.zeros((T, a.d_z), dtype=np.float32)
            for tix in range(T):
                n = len(sec_x)
                ctx = np.concatenate([np.tile(ctx_all[tix], (n, 1)),
                                      (np.full(n, lats[sec_y]) / 90)[:, None],
                                      (lons[sec_x] / 180)[:, None]], 1)
                v, o = gather(torch.full((n,), tix, dtype=torch.long),
                              torch.full((n,), sec_y, dtype=torch.long),
                              torch.as_tensor(sec_x))
                if a.patch > 1:
                    v, o = gather_px(Xt, OBS, torch.full((n,), tix, dtype=torch.long),
                                     torch.full((n,), sec_y, dtype=torch.long),
                                     torch.as_tensor(sec_x), a.patch)
                    v, o = v.to(dev), o.to(dev)
                zz = model.encode(v, o, torch.zeros(n, C, dtype=torch.bool, device=dev),
                                  torch.as_tensor(ctx, dtype=torch.float32).to(dev))
                emb[tix] = zz.mean(0).cpu().numpy()
            ridx = rapid[:, 0].astype(int); rv = rapid[:, 1]
            tr = ~t_hold[ridx]; te = t_hold[ridx]
            if te.sum() >= 12:
                A = np.c_[emb[ridx], np.ones(len(ridx))]
                lam = 1e-2 * np.eye(A.shape[1]); lam[-1, -1] = 0
                wgt = np.linalg.solve(A[tr].T @ A[tr] + lam, A[tr].T @ rv[tr])
                pr = A @ wgt
                r_te = float(np.corrcoef(pr[te], rv[te])[0, 1])
                r_tr = float(np.corrcoef(pr[tr], rv[tr])[0, 1])
                results["rapid_probe"] = {"pearson_train": r_tr, "pearson_heldout_years": r_te,
                                          "n_train": int(tr.sum()), "n_test": int(te.sum())}

    print(json.dumps(results, indent=2))
    save_ckpt()
    json.dump(results, open(os.path.join(a.out, "eval.json"), "w"), indent=2)
    print(f"saved {a.out}/pixelmae.pt")

    try:                                       # every run gets its curve
        import plot_run
        plot_run.render(os.path.basename(a.out.rstrip("/")))
    except Exception as e:                     # a missing matplotlib never
        print(f"(curve not rendered: {e})")    # kills a finished run



if __name__ == "__main__":
    main()

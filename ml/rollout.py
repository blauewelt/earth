#!/usr/bin/env python3
"""Autoregressive rollout — how far into the future does the model see?

Every metric before this one scored a single step (t+1). This harness
answers the user's question directly: predict a month, feed the
prediction back into the context, predict the next — month after month —
and measure where skill dies.

Protocol (stated precisely, because horizon claims rot fastest):
  · For each held-out year Y and each start month s from Dec(Y-1) to
    Nov(Y), initialise the context with TRUE embeddings up to s (an
    observed initial condition, not leakage), then roll autoregressively.
    Score every (s, h) whose target month s+h lies inside Y — so h=1 has
    12 starts per year and h=12 has one, giving 3 clean year-long
    trajectories plus staggered partials from 3 holdout years.
  · Channel-space skill at horizon h, on decoded predictions, observed
    cells only, in anomaly space, against TWO baselines:
      - persistence: the anomaly at s stays frozen (hard at short range)
      - climatology: the anomaly is zero (the no-skill floor)
    "How far can we predict" = the horizon where model MSE reaches the
    climatology floor. skill = 1 - MSE_model/MSE_baseline (positive =
    better than the baseline).
  · AMOC by horizon, TWO fit modes reported side by side:
      - truefit (the original): a ridge probe fit on TRUE train-month
        section embeddings, applied to ROLLED section embeddings.
      - rolledfit: the probe is fit on ROLLED train-year section states
        at the SAME horizon band it will read. The truefit probe reads
        rolled states through weights that have only ever seen true
        ones — a train/apply distribution shift we impose on ourselves
        (rolled states are smoother than true states, measurably: E-011's
        amp_ratio < 1). Rolling the section pixels through the train
        years costs almost nothing (the transformer is per-pixel, so the
        train roll batches ~40 section pixels, not 600), and the fit
        stays out-of-sample: fit points are train-year targets, read
        points are holdout-year targets, exactly the codec's split.
    Both in deseasonalised r over horizon bands 1-3 / 4-6 / 7-12
    (single-horizon n is tiny: 36/21/3 points per horizon at the tail —
    bands are the honest resolution).
  · Seed ENSEMBLES: when several heads share an unroll value, their
    per-point probe predictions are averaged (aligned by (year, start,
    horizon)) and each band's r is recomputed — the cheapest ensemble
    there is, and the rollout metrics are seed-stable enough (±0.003)
    that a 3-seed mean is meaningful.

Usage:  python3 ml/rollout.py --run wind14 [--horizon 12] [--pixels 600]
Writes runs/<run>/rollout.json and prints the curve.
"""
import argparse
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model import PixelMAE, codec_from_ckpt
from trainprobe import anomaly_transform
from temporal import TemporalTransformer, embed_everything, rapid_section

HERE = os.path.dirname(os.path.abspath(__file__))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--horizon", type=int, default=12)
    ap.add_argument("--pixels", type=int, default=600)
    ap.add_argument("--data", default=os.path.join(HERE, "cache", "na_pixels.npz"),
                    help="tensor npz — was hardcoded to the 1-degree pilot, "
                         "which silently mismatches a quarter-degree codec")
    ap.add_argument("--temporal", nargs="+", default=None,
                    help="one or more temporal-head checkpoints. Several heads "
                         "share ONE embedding pass, which is what makes a "
                         "six-head E-010 eval a single job. Default: the "
                         "run's own runs/<run>/temporal.pt")
    ap.add_argument("--crop-window", choices=["na"], default=None,
                    help="crop the tensor to the pilot window first — for "
                         "evaluating an NA-trained model on a global tensor "
                         "in its own training distribution (anomaly space is "
                         "invariant to the build-time z-score, so the crop "
                         "reproduces the pilot tensor exactly)")
    a = ap.parse_args()
    d = dict(np.load(a.data))
    print(f"tensor: {a.data}")
    if a.crop_window == "na":
        la, lo_ = d["lats"], d["lons"]
        ysel = (la >= 0.0) & (la <= 70.0)
        xsel = (lo_ >= -100.0) & (lo_ <= 20.0)
        d["X"] = d["X"][:, ysel][:, :, xsel]
        d["lats"], d["lons"] = la[ysel], lo_[xsel]
        print(f"cropped to NA pilot window: {d['X'].shape}")
    months = [str(m) for m in d["months"]]
    T = len(months)
    moy = np.array([int(m[5:7]) - 1 for m in months])
    lats, lons = d["lats"], d["lons"]

    ck = torch.load(os.path.join(HERE, "runs", a.run, "pixelmae.pt"),
                    map_location="cpu", weights_only=False)
    head_paths = a.temporal or [os.path.join(HERE, "runs", a.run, "temporal.pt")]
    head_specs = []
    for hp in head_paths:
        tk = torch.load(hp, map_location="cpu", weights_only=False)
        ta = tk.get("args", {})
        d_ = str(ta.get("direct") or "").strip()
        up_ = str(ta.get("unroll_probs") or "").strip()
        # "p" marks a SAMPLED unroll (E-016): u4p_s0 is a different recipe
        # from u4_s0 and must never share its ensemble group
        label = (f"u{ta.get('unroll', 1)}" + ("p" if up_ else "")
                 + (f"_d{d_.replace(',', '-')}" if d_ else "")
                 + f"_s{ta.get('seed', 0)}")
        if any(l == label for l, _ in head_specs):
            label += "_" + os.path.basename(hp).replace(".pt", "")
        head_specs.append((label, tk))
        print(f"head {label}: {hp} (d_model={ta.get('d_model')}, "
              f"layers={ta.get('layers')}, K={ta.get('K')}, "
              f"unroll={ta.get('unroll', 1)}, seed={ta.get('seed', 0)})")
    K = head_specs[0][1]["args"]["K"]
    for l, tk_ in head_specs:
        if tk_["args"]["K"] != K:
            sys.exit(f"head {l} has K={tk_['args']['K']} != {K} — "
                     f"windows are not comparable")
    X = d["X"].copy()
    if X.shape[-1] != len(ck["chan"]):
        sys.exit(f"tensor C={X.shape[-1]} != checkpoint C={len(ck['chan'])}")
    hold_years = sorted(ck["args"]["holdout_years"].split(","))
    t_hold = np.array([m[:4] in set(hold_years) for m in months])
    lo, hi = (float(v) for v in ck["args"]["holdout_lon"].split(","))
    x_hold = (lons >= lo) & (lons < hi)
    Xa, dynamic = anomaly_transform(X, moy, t_hold, x_hold)
    codec = codec_from_ckpt(ck, X.shape[-1])
    codec.load_state_dict(ck["model"])
    codec.eval()
    # Same one-line omission dip_check.py had: embed_everything follows the
    # MODEL's device, so without this the rollout embeds the whole tensor on
    # CPU next to an idle GPU. Found by auditing every embed_everything caller
    # at once rather than fixing the one that happened to be slow.
    _dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    codec.to(_dev)

    ocean = np.isfinite(d["X"][..., 0]).any(axis=0)
    ys, xs = np.where(ocean)
    _, sec_sel = rapid_section(lats, lons, ys, xs)
    rng = np.random.default_rng(0)
    keep = rng.choice(len(ys), min(a.pixels, len(ys)), replace=False)
    keep = np.union1d(keep, sec_sel)
    kys, kxs = ys[keep], xs[keep]
    sec_pos = np.where(np.isin(keep, sec_sel))[0]
    P = len(kys)

    ctx_all = np.stack([np.sin(2 * np.pi * moy / 12),
                        np.cos(2 * np.pi * moy / 12)], 1).astype(np.float32)
    Xt = torch.from_numpy(np.nan_to_num(Xa, nan=0.0))
    OBS = torch.from_numpy(np.isfinite(Xa))
    Z, coords = embed_everything(codec, Xt, OBS, ctx_all, lats, lons,
                                 kys, kxs, ck["d_z"])
    Zt = torch.from_numpy(Z)
    Mt = torch.as_tensor(ctx_all)

    # static identity — EXACTLY as stage-2 training builds it, INCLUDING the
    # patch branch. The bare [P, C] form below is the patch=1 pilot's shape;
    # a patch=3 codec's encode unpacks [B, C, patch²] and #149 died on that
    # line four minutes into the eval ("expected 3, got 2"). The toy test
    # missed it because the toy codec is patch=1 — mirrored from temporal.py,
    # which met this exact fork at line ~808.
    with torch.no_grad():
        stat_obs = OBS[0].clone()
        for c in dynamic:
            stat_obs[..., c] = False
        ctx0 = np.concatenate([np.zeros((P, 2), np.float32), coords], 1)
        if getattr(codec, "patch", 1) > 1:
            from model import gather_px
            t0i = torch.zeros(P, dtype=torch.long)
            vv, oo = gather_px(Xt, stat_obs[None], t0i,
                               torch.as_tensor(kys), torch.as_tensor(kxs),
                               codec.patch)
            Zstat = codec.encode(vv.to(_dev), oo.to(_dev),
                                 torch.zeros(P, X.shape[-1], dtype=torch.bool,
                                             device=_dev),
                                 torch.as_tensor(ctx0).to(_dev)).cpu().numpy()
        else:
            Zstat = codec.encode(Xt[0, kys, kxs].to(_dev),
                                 stat_obs[kys, kxs].to(_dev),
                                 torch.zeros(P, X.shape[-1], dtype=torch.bool,
                                             device=_dev),
                                 torch.as_tensor(ctx0).to(_dev)).cpu().numpy()
    codec.to("cpu")               # the rollout loop below is CPU-bound
    static_ctx = torch.as_tensor(np.concatenate([Zstat, coords], 1))

    # ---- RAPID probe fit on TRUE train-month section embeddings -----------
    rapid = d["rapid"]
    ridx = rapid[:, 0].astype(int)
    rv = rapid[:, 1].copy()
    rmoy = moy[ridx]
    tr_all = ~t_hold[ridx]
    rclim = np.array([rv[tr_all & (rmoy == m)].mean() for m in range(12)])
    rv_des = rv - rclim[rmoy]
    Fsec_true = Z[:, sec_pos].mean(1)

    # same protocol as ridge_r (standardise on train, lambda on a train
    # tail), but we need the WEIGHTS to apply to rolled embeddings later.
    # Factored out because the rolledfit bands below run the identical
    # protocol on rolled-state features — one fit function, two feature
    # sources, so the comparison can never be a protocol difference.
    def fit_ridge(F, y):
        """F [n, d] raw features in TIME ORDER, y [n]. Returns (mu, sd, w)."""
        mu = F.mean(0)
        sd = F.std(0) + 1e-9
        Fz_ = (F - mu) / sd
        n = len(y)
        fit_i = np.arange(int(0.8 * n))
        val_i = np.arange(int(0.8 * n), n)

        def _solve(idx, lam):
            A = np.c_[Fz_[idx], np.ones(len(idx))]
            reg = lam * np.eye(A.shape[1]); reg[-1, -1] = 0
            return np.linalg.solve(A.T @ A + reg, A.T @ y[idx])

        best_lam, best_r = 1.0, -np.inf
        for lam in (1e-2, 1e-1, 1, 10, 100, 1000):
            w = _solve(fit_i, lam)
            p = np.c_[Fz_[val_i], np.ones(len(val_i))] @ w
            r = np.corrcoef(p, y[val_i])[0, 1]
            if np.isfinite(r) and r > best_r:
                best_r, best_lam = r, lam
        return mu, sd, _solve(np.arange(n), best_lam)

    tr_idx = np.where(tr_all)[0]
    mu_p, sd_p, w_probe = fit_ridge(Fsec_true[ridx][tr_all], rv_des[tr_idx])
    ym_to_r = {int(months[mi][:4]) * 100 + int(months[mi][5:7]): i
               for i, mi in enumerate(ridx)}

    def eval_one(model, label):
        # ---- the rollouts ------------------------------------------------------
        month_index = {m: i for i, m in enumerate(months)}
        qc = torch.arange(X.shape[-1])[None, :].expand(P, -1)
        off0 = torch.zeros(P, X.shape[-1], 3, dtype=torch.long)
        H = a.horizon
        sums = {k: np.zeros(H + 1) for k in
                ("mse_m", "mse_p", "mse_c", "mse_d", "n",
                 "sxy", "sxx", "syy", "sx", "sy")}
        probe_pts = {h: [] for h in range(1, H + 1)}   # (start, truefit pred, y)
        hold_raw = {h: [] for h in range(1, H + 1)}    # (start, raw fvec, y)
        # DIRECT heads (E-014): scored PAIRED with the iterated path — same
        # starts, same target months, same observed cells — so the direct-vs-
        # iterated difference can never be a sampling difference. All direct
        # predictions come from ONE forward over the true initial window (the
        # h=1 forward, reused), which is exactly how a direct head is used.
        DIR = tuple(h for h in getattr(model, "direct", ()) if 1 <= h <= H)
        sums_dir = {k: np.zeros(H + 1) for k in
                    ("mse_m", "n", "sxy", "sxx", "syy", "sx", "sy")}
        dir_pts = {h: [] for h in DIR}                 # (start, truefit pred, y)
        dir_hold_raw = {h: [] for h in DIR}            # (start, raw fvec, y)

        # damped persistence (AR1): the literature's fair cheap baseline — raw
        # persistence over-commits at long leads. Lag-1 autocorrelation per
        # (pixel, channel) from TRAIN months only; forecast = r1^h * anomaly(s).
        with torch.no_grad():
            Xk = Xa[:, kys, kxs]                                   # [T, P, C]
            okk = np.isfinite(Xk)
            r1 = np.zeros((P, Xk.shape[-1]), dtype=np.float32)
            tr_m = ~t_hold
            for c in range(Xk.shape[-1]):
                x0 = Xk[:-1, :, c]; x1 = Xk[1:, :, c]
                m01 = okk[:-1, :, c] & okk[1:, :, c] & tr_m[:-1, None]
                n01 = m01.sum(0).astype(float)
                with np.errstate(invalid="ignore", divide="ignore"):
                    mx = np.where(n01 > 0, np.nansum(np.where(m01, x0, 0), 0) / n01, 0)
                    my = np.where(n01 > 0, np.nansum(np.where(m01, x1, 0), 0) / n01, 0)
                    cov = np.nansum(np.where(m01, (x0 - mx) * (x1 - my), 0), 0)
                    v0 = np.nansum(np.where(m01, (x0 - mx) ** 2, 0), 0)
                    v1 = np.nansum(np.where(m01, (x1 - my) ** 2, 0), 0)
                    rr = cov / (np.sqrt(v0 * v1) + 1e-9)
                r1[:, c] = np.where(n01 >= 24, np.clip(rr, 0, 0.999), 0)
            r1_t = torch.from_numpy(r1)

        with torch.no_grad():
            for Y in hold_years:
                for s_off in range(12):
                    start_m = (f"{int(Y) - 1}-12" if s_off == 0
                               else f"{Y}-{s_off:02d}")
                    if start_m not in month_index:
                        continue
                    s = month_index[start_m]
                    if s - K + 1 < 0 or s + 1 >= T:
                        continue
                    seq = Zt[s - K + 1: s + 1].transpose(0, 1).clone()   # [P,K,dz]
                    zdir_hat = {}
                    for h in range(1, H + 1):
                        t_tgt = s + h
                        if t_tgt >= T or months[t_tgt][:4] != Y:
                            break
                        mseq = torch.stack(
                            [Mt[s - K + h + j] for j in range(K)], 0
                        )[None].expand(P, -1, -1)
                        pred, hid_full = model(seq, mseq, static_ctx)
                        if h == 1 and DIR:
                            # the h=1 forward IS the true-context forward:
                            # every direct horizon predicts from its hidden
                            hid0 = hid_full[:, -1]
                            zdir_hat = {hd: model.heads_direct[str(hd)](hid0)
                                        for hd in DIR}
                        zhat = pred[:, -1]                               # [P,dz]
                        seq = torch.cat([seq[:, 1:], zhat[:, None]], 1)
                        # channel-space scoring at the target month
                        xhat = codec.query(zhat, qc, off0)
                        v_true = Xt[t_tgt, kys, kxs]
                        o = OBS[t_tgt, kys, kxs]
                        v_pers = Xt[s, kys, kxs]
                        op = o & OBS[s, kys, kxs]
                        if h in zdir_hat and op.sum() > 0:
                            zdh = zdir_hat[h]
                            xdh = codec.query(zdh, qc, off0)
                            sums_dir["mse_m"][h] += float(
                                ((xdh - v_true).pow(2) * op).sum())
                            sums_dir["n"][h] += float(op.sum())
                            xm_ = xdh * op; ym_ = v_true * op
                            sums_dir["sxy"][h] += float((xm_ * ym_).sum())
                            sums_dir["sxx"][h] += float((xm_ * xm_).sum())
                            sums_dir["syy"][h] += float((ym_ * ym_).sum())
                            sums_dir["sx"][h] += float(xm_.sum())
                            sums_dir["sy"][h] += float(ym_.sum())
                        if h in zdir_hat:
                            ymk_ = (int(months[t_tgt][:4]) * 100
                                    + int(months[t_tgt][5:7]))
                            if ymk_ in ym_to_r:
                                fr_ = zdir_hat[h][sec_pos].mean(0).numpy()
                                pd_ = float(np.dot(
                                    np.r_[(fr_ - mu_p) / sd_p, 1.0], w_probe))
                                dir_pts[h].append((s, pd_,
                                                   rv_des[ym_to_r[ymk_]]))
                                dir_hold_raw[h].append((s, fr_,
                                                        rv_des[ym_to_r[ymk_]]))
                        if op.sum() > 0:
                            v_damp = v_pers * r1_t.pow(h)
                            sums["mse_m"][h] += float(((xhat - v_true).pow(2) * op).sum())
                            sums["mse_p"][h] += float(((v_pers - v_true).pow(2) * op).sum())
                            sums["mse_d"][h] += float(((v_damp - v_true).pow(2) * op).sum())
                            sums["mse_c"][h] += float((v_true.pow(2) * op).sum())
                            sums["n"][h] += float(op.sum())
                            # ACC accumulators (centered Pearson per horizon)
                            xm = xhat * op; ym = v_true * op
                            sums["sxy"][h] += float((xm * ym).sum())
                            sums["sxx"][h] += float((xm * xm).sum())
                            sums["syy"][h] += float((ym * ym).sum())
                            sums["sx"][h] += float(xm.sum())
                            sums["sy"][h] += float(ym.sum())
                        # AMOC probe on the rolled section embedding
                        ym = int(months[t_tgt][:4]) * 100 + int(months[t_tgt][5:7])
                        if ym in ym_to_r:
                            fraw = zhat[sec_pos].mean(0).numpy()
                            f = (fraw - mu_p) / sd_p
                            pr = float(np.dot(np.r_[f, 1.0], w_probe))
                            probe_pts[h].append((s, pr, rv_des[ym_to_r[ym]]))
                            hold_raw[h].append((s, fraw, rv_des[ym_to_r[ym]]))

        # ---- rolledfit: roll the SECTION PIXELS through the TRAIN years ----
        # The transformer is per-pixel (P is a batch dimension), so rolling
        # only the ~40 section pixels through ~17 train years costs less than
        # rolling 600 pixels through 3 holdout years did. Fit points are
        # train-year targets, read points are the holdout points collected
        # above — the same year-blocked split every other instrument uses.
        # Context windows may LOOK at holdout months (true-context init is an
        # observed initial condition); targets are train months by
        # construction because whole holdout YEARS are skipped.
        train_raw = {h: [] for h in range(1, H + 1)}   # (t_tgt, fvec, y)
        dir_train_raw = {h: [] for h in DIR}           # (t_tgt, fvec, y)
        sec_t = torch.as_tensor(np.asarray(sec_pos))
        stat_sec = static_ctx[sec_t]
        with torch.no_grad():
            for Y in sorted({m[:4] for m in months}):
                if Y in hold_years:
                    continue
                for s_off in range(12):
                    start_m = (f"{int(Y) - 1}-12" if s_off == 0
                               else f"{Y}-{s_off:02d}")
                    if start_m not in month_index:
                        continue
                    s = month_index[start_m]
                    if s - K + 1 < 0 or s + 1 >= T:
                        continue
                    # the tensor starts two decades before RAPID does — skip
                    # any start whose whole horizon fan lands before the
                    # truth series, instead of rolling 1980s months for
                    # points that cannot be scored
                    if not any((int(months[s + h][:4]) * 100
                                + int(months[s + h][5:7])) in ym_to_r
                               for h in range(1, min(H, T - 1 - s) + 1)):
                        continue
                    seq = Zt[s - K + 1: s + 1, sec_t].transpose(0, 1).clone()
                    for h in range(1, H + 1):
                        t_tgt = s + h
                        if t_tgt >= T or months[t_tgt][:4] != Y:
                            break
                        mseq = torch.stack(
                            [Mt[s - K + h + j] for j in range(K)], 0
                        )[None].expand(len(sec_pos), -1, -1)
                        pred, hid_tr = model(seq, mseq, stat_sec)
                        if h == 1 and DIR:
                            # direct-fit points: the direct heads' own
                            # predictions on train-year starts, so the
                            # directfit probe is fit on the distribution it
                            # will read (mirrors rolledfit for the iterated
                            # path)
                            dh0 = hid_tr[:, -1]
                            for hd in DIR:
                                t_d = s + hd
                                if t_d >= T or months[t_d][:4] != Y:
                                    continue
                                ymd = (int(months[t_d][:4]) * 100
                                       + int(months[t_d][5:7]))
                                if ymd in ym_to_r:
                                    zd_ = model.heads_direct[str(hd)](dh0)
                                    dir_train_raw[hd].append(
                                        (t_d, zd_.mean(0).numpy(),
                                         rv_des[ym_to_r[ymd]]))
                        zhat = pred[:, -1]
                        seq = torch.cat([seq[:, 1:], zhat[:, None]], 1)
                        ym = (int(months[t_tgt][:4]) * 100
                              + int(months[t_tgt][5:7]))
                        if ym in ym_to_r:
                            train_raw[h].append((t_tgt, zhat.mean(0).numpy(),
                                                 rv_des[ym_to_r[ym]]))

        out = {"head": label, "K": K, "horizon": H,
               "protocol": "staggered starts into holdout years, true-context init",
               "metric_names": "MSSS per Goddard et al. 2013; ACC = centered "
                               "anomaly correlation; amp = sigma_f/sigma_o",
               "chan_skill": []}
        print(f"{label}: autoregressive rollout, {len(hold_years)} holdout years")
        print("  h   n_cells   MSSS_clim   MSSS_pers   MSSS_damped     ACC    amp")
        for h in range(1, H + 1):
            if sums["n"][h] == 0:
                continue
            n_ = sums["n"][h]
            mm = sums["mse_m"][h] / n_
            mp = sums["mse_p"][h] / n_
            md = sums["mse_d"][h] / n_
            mc = sums["mse_c"][h] / n_
            s_c, s_p, s_d = 1 - mm / mc, 1 - mm / mp, 1 - mm / md
            # centered ACC + amplitude ratio from the accumulators
            vx = sums["sxx"][h] / n_ - (sums["sx"][h] / n_) ** 2
            vy = sums["syy"][h] / n_ - (sums["sy"][h] / n_) ** 2
            cov = sums["sxy"][h] / n_ - sums["sx"][h] * sums["sy"][h] / n_ ** 2
            acc = cov / (np.sqrt(vx * vy) + 1e-12)
            amp = float(np.sqrt(vx / (vy + 1e-12)))
            out["chan_skill"].append({"h": h, "n": int(n_),
                                      "msss_clim": round(s_c, 3),
                                      "msss_pers": round(s_p, 3),
                                      "msss_damped": round(s_d, 3),
                                      "acc": round(float(acc), 3),
                                      "amp_ratio": round(amp, 3)})
            print(f"  {h:2d}  {int(n_):8d}   {s_c:+9.3f}   {s_p:+9.3f}   "
                  f"{s_d:+11.3f}   {acc:+.3f}  {amp:5.2f}")
        # one scalar for model selection: mean skill-vs-climatology over the
        # horizon sweep ("horizon AUC") — rewards models that stay useful deep
        # into the rollout, not just at t+1.
        if out["chan_skill"]:
            out["horizon_auc"] = round(
                float(np.mean([c["msss_clim"] for c in out["chan_skill"]])), 3)
            print(f"  horizon AUC (mean vs-clim, h=1..{H}): {out['horizon_auc']:+.3f}")
        bands = (("h1-3", (1, 2, 3)), ("h4-6", (4, 5, 6)),
                 ("h7-12", tuple(range(7, H + 1))))
        out["amoc_bands"] = {}
        out["amoc_bands_rolledfit"] = {}
        ens = {"truefit": {}, "rolledfit": {}}   # band -> [(key, pred, y)]
        for name, hs in bands:
            pts = [(h, s, pr, y) for h in hs
                   for (s, pr, y) in probe_pts.get(h, [])]
            if len(pts) >= 8:
                pr = np.array([p[2] for p in pts])
                tv = np.array([p[3] for p in pts])
                r = float(np.corrcoef(pr, tv)[0, 1])
                out["amoc_bands"][name] = {"r": round(r, 3), "n": len(pts)}
                ens["truefit"][name] = [((h_, s_), p_, y_)
                                        for h_, s_, p_, y_ in pts]
                print(f"  AMOC {name} truefit:   r {r:+.3f} (n={len(pts)})")
            # rolledfit: fit THIS band's probe on rolled train-year states
            fit_pts = sorted((p for h in hs for p in train_raw.get(h, [])),
                             key=lambda p: p[0])          # time order, for
            hp = [(h, s, f, y) for h in hs                # the lambda tail
                  for (s, f, y) in hold_raw.get(h, [])]
            if len(fit_pts) >= 24 and len(hp) >= 8:
                Ff = np.stack([f for _, f, _ in fit_pts])
                yf = np.array([y for _, _, y in fit_pts])
                mu_b, sd_b, w_b = fit_ridge(Ff, yf)
                preds = [((h_, s_),
                          float(np.dot(np.r_[(f_ - mu_b) / sd_b, 1.0], w_b)),
                          y_) for h_, s_, f_, y_ in hp]
                pr = np.array([p[1] for p in preds])
                tv = np.array([p[2] for p in preds])
                r = float(np.corrcoef(pr, tv)[0, 1])
                out["amoc_bands_rolledfit"][name] = {
                    "r": round(r, 3), "n": len(hp), "n_fit": len(fit_pts)}
                ens["rolledfit"][name] = preds
                print(f"  AMOC {name} rolledfit: r {r:+.3f} "
                      f"(n={len(hp)}, fit on {len(fit_pts)} train points)")

        # ---- DIRECT heads: paired with the iterated path -------------------
        if DIR:
            out["direct"] = {"chan_skill": [], "amoc": {}}
            for h in DIR:
                nd = sums_dir["n"][h]
                if nd == 0 or sums["n"][h] == 0:
                    continue
                # baselines over the SAME cells (both paths accumulate in the
                # same loop iterations, so the cell sets are identical)
                mm = sums_dir["mse_m"][h] / nd
                mp = sums["mse_p"][h] / sums["n"][h]
                md = sums["mse_d"][h] / sums["n"][h]
                mc = sums["mse_c"][h] / sums["n"][h]
                mi = sums["mse_m"][h] / sums["n"][h]      # iterated model
                vx = sums_dir["sxx"][h] / nd - (sums_dir["sx"][h] / nd) ** 2
                vy = sums_dir["syy"][h] / nd - (sums_dir["sy"][h] / nd) ** 2
                cov = (sums_dir["sxy"][h] / nd
                       - sums_dir["sx"][h] * sums_dir["sy"][h] / nd ** 2)
                acc = cov / (np.sqrt(vx * vy) + 1e-12)
                row = {"h": h, "n": int(nd),
                       "msss_clim": round(1 - mm / mc, 3),
                       "msss_pers": round(1 - mm / mp, 3),
                       "msss_damped": round(1 - mm / md, 3),
                       "acc": round(float(acc), 3),
                       "amp_ratio": round(float(np.sqrt(vx / (vy + 1e-12))), 3),
                       # the paired verdict: positive = direct beats iterated
                       # on the identical cells
                       "delta_msss_clim_vs_iterated":
                           round((1 - mm / mc) - (1 - mi / mc), 3)}
                out["direct"]["chan_skill"].append(row)
                print(f"  DIRECT h={h}: MSSS_clim {row['msss_clim']:+.3f} "
                      f"(iterated {1 - mi / mc:+.3f}, "
                      f"delta {row['delta_msss_clim_vs_iterated']:+.3f}), "
                      f"ACC {row['acc']:+.3f}, amp {row['amp_ratio']:.2f}")
                pts = dir_pts.get(h, [])
                if len(pts) >= 8:
                    pr = np.array([p[1] for p in pts])
                    tv = np.array([p[2] for p in pts])
                    r = float(np.corrcoef(pr, tv)[0, 1])
                    out["direct"]["amoc"][f"h{h}_truefit"] = {
                        "r": round(r, 3), "n": len(pts)}
                    print(f"  DIRECT h={h} AMOC truefit:   r {r:+.3f} "
                          f"(n={len(pts)})")
                fit_pts = sorted(dir_train_raw.get(h, []), key=lambda p: p[0])
                hp = dir_hold_raw.get(h, [])
                if len(fit_pts) >= 24 and len(hp) >= 8:
                    Ff = np.stack([f for _, f, _ in fit_pts])
                    yf = np.array([y for _, _, y in fit_pts])
                    mu_d, sd_d, w_d = fit_ridge(Ff, yf)
                    pr = np.array([float(np.dot(
                        np.r_[(f_ - mu_d) / sd_d, 1.0], w_d))
                        for _, f_, _ in hp])
                    tv = np.array([y for _, _, y in hp])
                    r = float(np.corrcoef(pr, tv)[0, 1])
                    out["direct"]["amoc"][f"h{h}_directfit"] = {
                        "r": round(r, 3), "n": len(hp), "n_fit": len(fit_pts)}
                    print(f"  DIRECT h={h} AMOC directfit: r {r:+.3f} "
                          f"(n={len(hp)}, fit on {len(fit_pts)})")

        return out, ens

    results = {"run": a.run, "data": os.path.basename(a.data),
               "K": K, "horizon": a.horizon, "heads": {}}
    ens_by_head = {}
    for label, tk_ in head_specs:
        # k_max comes FROM THE CHECKPOINT, not from a convention. There are
        # two conventions in this repo — temporal.py builds its position
        # table at k_max=K, train_joint.py at max(K, 36) — and this line has
        # now guessed wrong in BOTH directions: k_max=K failed the toy (whose
        # fixture used the joint convention), max(K, 36) failed #157 against
        # real stage-2 heads (pos.weight [24, 192]). The table's own first
        # dimension is the one answer that cannot disagree with the file.
        k_tbl = tk_["model"]["pos.weight"].shape[0]
        dir_ = tuple(int(x) for x in
                     str(tk_["args"].get("direct") or "").split(",")
                     if x.strip())
        model = TemporalTransformer(d_z=ck["d_z"], d_model=tk_["args"]["d_model"],
                                    n_heads=4, n_layers=tk_["args"]["layers"],
                                    k_max=k_tbl, direct=dir_)
        model.load_state_dict(tk_["model"])
        model.eval()
        results["heads"][label], ens_by_head[label] = eval_one(model, label)

    # ---- seed ensembles: average per-point probe predictions ---------------
    # Grouped by the label's unroll prefix (u1_s0/u1_s1/u1_s2 -> "u1"), plus
    # an "all" group when heads of mixed U are passed — scientifically the
    # per-U groups are the read ("3 seeds, same recipe"), the all-group is a
    # curiosity, but both are cheap and the toy only exercises mixed U.
    groups = {}
    for label in ens_by_head:
        # everything up to the seed token: "u1_s0" -> "u1",
        # "u1_d3-6-12_s2" -> "u1_d3-6-12" — direct arms ensemble with their
        # own recipe's seeds, never with the plain arms'
        groups.setdefault(label.rsplit("_s", 1)[0], []).append(label)
    if len(ens_by_head) >= 2:
        groups["all"] = list(ens_by_head)
    results["ensembles"] = {}
    for g, labs in sorted(groups.items()):
        if len(labs) < 2:
            continue
        entry = {}
        for mode in ("truefit", "rolledfit"):
            bands_out = {}
            names = set.intersection(
                *(set(ens_by_head[l][mode]) for l in labs))
            for bn in sorted(names):
                maps = [{k: (p, y) for k, p, y in ens_by_head[l][mode][bn]}
                        for l in labs]
                common = set(maps[0])
                for m_ in maps[1:]:
                    common &= set(m_)
                if len(common) < 8:
                    continue
                ks = sorted(common)
                pr = np.mean([[m_[k][0] for k in ks] for m_ in maps], 0)
                tv = np.array([maps[0][k][1] for k in ks])
                r = float(np.corrcoef(pr, tv)[0, 1])
                bands_out[bn] = {"r": round(r, 3), "n": len(ks),
                                 "members": len(labs)}
            if bands_out:
                entry[mode] = bands_out
        if entry:
            results["ensembles"][g] = entry
            for mode, bo in entry.items():
                for bn, v in bo.items():
                    print(f"  ENSEMBLE {g} ({len(labs)} heads) {bn} {mode}: "
                          f"r {v['r']:+.3f} (n={v['n']})")

    os.makedirs(os.path.join(HERE, "runs", a.run), exist_ok=True)
    path = os.path.join(HERE, "runs", a.run, "rollout_eval.json")
    json.dump(results, open(path, "w"), indent=1)
    print("wrote", path)
    if len(head_specs) == 1:
        # legacy shape, so older readers of rollout.json keep working
        legacy = dict(results["heads"][head_specs[0][0]])
        legacy["run"] = a.run
        json.dump(legacy, open(os.path.join(HERE, "runs", a.run,
                                            "rollout.json"), "w"), indent=1)


if __name__ == "__main__":
    main()

"""E-021: the 20-year fan — long-horizon AMOC projection with ensembles.

Chris (2026-08-13): "predict 20 future years ... run repeated predictions
... and observe stats on the different possible outcomes."

What this does, and what it deliberately is not. Each stage-2 head is a
deterministic per-pixel dynamics model over frozen codec embeddings; a
single roll from a single context gives ONE trajectory, which at month 12
already damps to 0.81-0.88 of true amplitude (E-017 ROLLOUT). So "possible
outcomes" needs constructed spread, and this script builds it from the one
honest noise source we have measured: the head's OWN one-step residual on
real data. Two ensemble families per head:

  ic  - members perturb the INITIAL 24-month context with residual-scaled
        noise, then roll deterministically: measures how fast the learned
        dynamics forgets its initial condition (chaotic divergence vs
        collapse to the attractor).
  sde - members additionally receive residual-scaled noise EVERY rolled
        month: the model as a stochastic parameterisation, which is the
        fairer reading of "possible outcomes" for a system whose one-step
        error is irreducible. Expect a wider, non-collapsing fan.

Two starts per head:

  future   - context = the record's last K months; 240 rolled months
             beyond the data. No truth exists; the output IS the fan.
  hindcast - context ends 2004-12; the same 240-month roll lands on
             2005-2024, where RAPID truth exists. This is the calibration
             companion: if the hindcast fan does not cover reality, the
             future fan's width means nothing. (Caveat, stated once: the
             transport read-out is ridge-fit on train months of the SAME
             era - with a label record that starts in 2004 there is no
             disjoint fit era; the hindcast tests fan width, not skill.)

NOT an IPCC-style projection: there is no forcing pathway input - the roll
is "what the learned dynamics does unforced from here."

The read-out: section-mean rolled z -> the truefit ridge (rollout.py's
protocol, fit on train-month TRUE embeddings vs deseasonalised RAPID),
giving deseasonalised Sv per rolled month; the train-month RAPID
climatology re-seasonalises for absolute Sv.

Rolling ONLY the section pixels is exact, not an approximation: the head
attends over TIME per pixel with a static spatial identity - no
cross-pixel coupling exists in the model - so 265 pixels x 240 steps is
the same computation the full-grid roll would produce at the section.

Usage (box, GPU):
  python3 ml/project_amoc.py --x ml/cache/family3_X.npy \
      --npz-small ml/cache/f3_dec_small.npz --z ml/cache/Z_....npy \
      --ckpt ml/cache/f3_anchor41M__pixelmae.pt \
      --heads ml/runs/heads/e017_u1_s0.pt ... \
      --months 240 --members 10 --out ml/runs/actions/project_amoc.json
"""
import argparse
import json
import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from model import codec_from_ckpt, gather_px                    # noqa: E402
from recon_eval import stream_stats, build_slab                 # noqa: E402
from temporal import TemporalTransformer, rapid_section         # noqa: E402


def fit_ridge(F, y):
    """rollout.py's truefit protocol, verbatim: standardise, pick lambda on
    the train tail, solve on all of train. Returns (mu, sd, w)."""
    mu = F.mean(0)
    sd = F.std(0) + 1e-9
    Fz = (F - mu) / sd
    n = len(y)
    fit_i = np.arange(int(0.8 * n))
    val_i = np.arange(int(0.8 * n), n)

    def _solve(idx, lam):
        A = np.c_[Fz[idx], np.ones(len(idx))]
        reg = lam * np.eye(A.shape[1]); reg[-1, -1] = 0
        return np.linalg.solve(A.T @ A + reg, A.T @ y[idx])

    best_lam, best_r = 1.0, -np.inf
    for lam in (1e-2, 1e-1, 1, 10, 100, 1000):
        w = _solve(fit_i, lam)
        p = np.c_[Fz[val_i], np.ones(len(val_i))] @ w
        r = np.corrcoef(p, y[val_i])[0, 1]
        if np.isfinite(r) and r > best_r:
            best_r, best_lam = r, lam
    return mu, sd, _solve(np.arange(n), best_lam), best_r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--x", required=True)
    ap.add_argument("--npz-small", required=True)
    ap.add_argument("--z", required=True, help="Z cache .npy (f16, [T,P,dz])")
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--heads", nargs="+", required=True)
    ap.add_argument("--months", type=int, default=240)
    ap.add_argument("--members", type=int, default=10,
                    help="per family; member 0 is always unperturbed")
    ap.add_argument("--starts", default="future,2004-12",
                    help="comma list: 'future' and/or YYYY-MM context ends")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    ap.add_argument("--cache-dir", default=os.path.join(HERE, "cache"))
    a = ap.parse_args()
    torch.manual_seed(a.seed)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    d = np.load(a.npz_small, allow_pickle=False)
    months = [str(m) for m in d["months"]]
    lats, lons = d["lats"], d["lons"]
    T = len(months)
    moy = np.array([int(m[5:7]) - 1 for m in months])
    ck = torch.load(a.ckpt, map_location="cpu", weights_only=False)
    C = len(ck["chan"])
    hold_years = set(ck["args"]["holdout_years"].split(","))
    t_hold = np.array([m[:4] in hold_years for m in months])
    lo, hi = (float(v) for v in ck["args"]["holdout_lon"].split(","))
    x_hold = (lons >= lo) & (lons < hi)

    Xm = np.load(a.x, mmap_mode="r")
    H, W = Xm.shape[1], Xm.shape[2]
    om_path = os.path.join(a.cache_dir, "ocean_mask.npy")
    ocean = np.load(om_path) if os.path.exists(om_path) else None
    if ocean is None:
        ocean = np.zeros((H, W), bool)
        for t0 in range(0, T, 16):
            ocean |= np.isfinite(np.asarray(Xm[t0:t0 + 16, :, :, 0])).any(0)
        np.save(om_path, ocean)
    ys, xs = np.where(ocean)
    P = len(ys)
    sec_y, sec_sel = rapid_section(lats, lons, ys, xs)
    sec_pidx = sec_sel                       # indices into the P ordering
    sxs = xs[sec_sel]
    S = len(sec_sel)
    print(f"section: {S} px at row {sec_y}", flush=True)

    Zm = np.load(a.z, mmap_mode="r")
    assert Zm.shape == (T, P, ck["d_z"]), \
        f"Z {Zm.shape} != {(T, P, ck['d_z'])} — ordering mismatch, refusing"
    Zsec = np.asarray(Zm[:, sec_pidx]).astype(np.float32)        # [T,S,dz]

    # ---- static identity (codec encode of static channels), slab-based ----
    st_path = os.path.join(a.cache_dir, "std_stats.npz")
    if os.path.exists(st_path):
        s_ = np.load(st_path)
        clim, dyn = s_["clim"], list(s_["dyn"])
        mean_c, std_c = s_["mean_c"], s_["std_c"]
    else:
        clim, dyn, mean_c, std_c = stream_stats(Xm, moy, t_hold, x_hold)
        np.savez(st_path, clim=clim, dyn=np.array(dyn),
                 mean_c=mean_c, std_c=std_c)
    rows = [sec_y - 1, sec_y, sec_y + 1]
    slab, obs = build_slab(Xm, rows, moy, clim, dyn, mean_c, std_c)
    slab_t = torch.from_numpy(np.nan_to_num(slab, nan=0.0))
    obs_t = torch.from_numpy(obs)
    codec = codec_from_ckpt(ck, C)
    codec.load_state_dict(ck["model"])
    codec.eval().to(dev)
    with torch.no_grad():
        stat_obs = obs_t[0].clone()
        for c in dyn:
            stat_obs[..., c] = False
        coords = np.stack([lats[np.full(S, sec_y)] / 90, lons[sxs] / 180],
                          1).astype(np.float32)
        ctx0 = np.concatenate([np.zeros((S, 2), np.float32), coords], 1)
        y1 = torch.ones(S, dtype=torch.long)          # slab center row
        vv, oo = gather_px(slab_t, stat_obs[None], torch.zeros(S, dtype=torch.long),
                           y1, torch.as_tensor(sxs), codec.patch)
        Zstat = codec.encode(vv.to(dev), oo.to(dev),
                             torch.zeros(S, C, dtype=torch.bool, device=dev),
                             torch.as_tensor(ctx0).to(dev)).cpu().numpy()
    # Verify the Z cache against a live re-encode of a few section months —
    # the recon_decoder guard, kept because a silent ordering mismatch here
    # would produce a beautiful fan of nonsense.
    from temporal import embed_everything
    rngv = np.random.default_rng(1)
    kv = rngv.choice(S, 8, replace=False)
    ctx_all = np.stack([np.sin(2 * np.pi * moy / 12),
                        np.cos(2 * np.pi * moy / 12)], 1)
    Zl, _ = embed_everything(codec, slab_t, obs_t, ctx_all, lats[rows], lons,
                             np.ones(len(kv), dtype=int), sxs[kv], ck["d_z"],
                             cache_path=None, batch=64)
    for tt in (0, T // 2, T - 1):
        dmax = float(np.abs(Zl[tt] - Zsec[tt][kv]).max())
        zscale = float(np.abs(Zl[tt]).max())
        assert dmax < max(0.02, 0.005 * zscale), \
            f"Z mismatch at t={tt}: {dmax} vs scale {zscale}"
    print("Z cache verified vs live re-encode ✓", flush=True)
    codec.to("cpu")
    static_ctx = torch.from_numpy(
        np.concatenate([Zstat, coords], 1)).to(dev)              # [S,dz+2]

    # ---- transport read-out (truefit protocol) ----------------------------
    rapid = d["rapid"]
    ridx = rapid[:, 0].astype(int)
    rv = rapid[:, 1].copy()
    rmoy = moy[ridx]
    tr_all = ~t_hold[ridx]
    rclim = np.array([rv[tr_all & (rmoy == m)].mean() for m in range(12)])
    rv_des = rv - rclim[rmoy]
    Fsec_true = Zsec.mean(1)                                     # [T,dz]
    mu_p, sd_p, w_probe, probe_val_r = fit_ridge(
        Fsec_true[ridx][tr_all], rv_des[tr_all])
    print(f"probe fit: {int(tr_all.sum())} train months, "
          f"val-tail r {probe_val_r:+.3f}", flush=True)

    def read_sv(z_flat):
        """z [B,S,dz] -> deseasonalised Sv [B]."""
        fr = z_flat.mean(1).cpu().numpy()
        return np.c_[(fr - mu_p) / sd_p, np.ones(len(fr))] @ w_probe

    def month_seq_feats(moys):
        m = np.asarray(moys)
        return torch.from_numpy(np.stack(
            [np.sin(2 * np.pi * m / 12),
             np.cos(2 * np.pi * m / 12)], 1).astype(np.float32))

    results = {"months_record_end": months[-1], "n_roll": a.months,
               "members": a.members, "section_px": S,
               "probe": {"val_tail_r": round(float(probe_val_r), 3),
                         "rclim": [round(float(v), 3) for v in rclim]},
               "rapid_truth": {"ym": [months[i] for i in ridx],
                               "sv_des": [round(float(v), 3) for v in rv_des]},
               "heads": {}}

    for hp in a.heads:
        tk = torch.load(hp, map_location="cpu", weights_only=False)
        ta = tk["args"]
        label = f"u{ta.get('unroll', 1)}_s{ta.get('seed', 0)}"
        K = ta["K"]
        k_tbl = tk["model"]["pos.weight"].shape[0]   # the file, not a convention
        model = TemporalTransformer(d_z=ck["d_z"], d_model=ta["d_model"],
                                    n_heads=4, n_layers=ta["layers"],
                                    k_max=k_tbl)
        model.load_state_dict(tk["model"])
        model.eval().to(dev)

        # one-step residual scale, teacher-forced on train months: THE noise
        # source for both ensemble families. Per-dz-dim std, pooled over
        # section pixels and months.
        with torch.no_grad():
            res_sq, res_n = np.zeros(ck["d_z"]), 0
            for t in range(K, T - 1, 3):             # every 3rd month is plenty
                if t_hold[t + 1]:
                    continue
                seq = torch.from_numpy(
                    Zsec[t - K + 1: t + 1].transpose(1, 0, 2)).to(dev)
                mseq = month_seq_feats(moy[t - K + 1: t + 1]) \
                    .to(dev)[None].expand(S, -1, -1)
                pred, _ = model(seq, mseq, static_ctx)
                r_ = (pred[:, -1] - torch.from_numpy(Zsec[t + 1]).to(dev))
                res_sq += (r_ ** 2).sum(0).cpu().numpy()
                res_n += S
            sigma = np.sqrt(res_sq / max(res_n, 1)).astype(np.float32)
        print(f"{label}: one-step residual σ mean {sigma.mean():.4f} "
              f"(z-scale ~1)", flush=True)
        sig_t = torch.from_numpy(sigma).to(dev)

        entry = {"sigma_mean": round(float(sigma.mean()), 4), "starts": {}}
        for start in a.starts.split(","):
            start = start.strip()
            if start == "future":
                s_end = T - 1
            else:
                if start not in months:
                    print(f"  start {start} not in record — skipped")
                    continue
                s_end = months.index(start)
            if s_end - K + 1 < 0:
                continue
            ctx_z = Zsec[s_end - K + 1: s_end + 1].transpose(1, 0, 2)  # [S,K,dz]
            win_moys = list(moy[s_end - K + 1: s_end + 1])
            y0, m0 = int(months[s_end][:4]), int(months[s_end][5:7])
            roll_ym = []
            yy, mm = y0, m0
            for _ in range(a.months):
                mm += 1
                if mm > 12:
                    mm, yy = 1, yy + 1
                roll_ym.append(f"{yy:04d}-{mm:02d}")

            fams = {}
            sv_raw = {}          # full [M, N] per family, pooled then dropped
            for fam in ("ic", "sde"):
                M = a.members
                # zlib.crc32, not hash(): str hash is salted per process,
                # which would make members irreproducible across runs
                import zlib
                g = torch.Generator(device="cpu").manual_seed(
                    a.seed * 7919
                    + zlib.crc32(f"{label}|{start}|{fam}".encode()) % 65521)
                seq = torch.from_numpy(np.tile(ctx_z, (M, 1, 1))).to(dev)
                if M > 1:
                    pert = torch.randn((M - 1) * S, K, ck["d_z"],
                                       generator=g) * sig_t.cpu()
                    seq[S:] += pert.to(dev)          # member 0 unperturbed
                cur = list(win_moys)
                sv = np.zeros((M, a.months), dtype=np.float32)
                with torch.no_grad():
                    for step in range(a.months):
                        mseq = month_seq_feats(cur).to(dev)[None] \
                            .expand(M * S, -1, -1)
                        sc = static_ctx.repeat(M, 1)
                        pred, _ = model(seq, mseq, sc)
                        zhat = pred[:, -1]
                        if fam == "sde" and M > 1:
                            kick = torch.randn((M - 1) * S, ck["d_z"],
                                               generator=g) * sig_t.cpu()
                            zhat = zhat.clone()
                            zhat[S:] += kick.to(dev)
                        sv[:, step] = read_sv(zhat.reshape(M, S, -1))
                        seq = torch.cat([seq[:, 1:], zhat[:, None]], 1)
                        cur = cur[1:] + [(cur[-1] + 1) % 12]
                # Store COMPACT: per-month quantiles across members, the
                # member-0 (unperturbed) trajectory, and 5 raw members for a
                # spaghetti plot. The full [M, 240] array per head/start/
                # family would be ~9 MB of JSON in an archive bundle that is
                # committed to a branch; the quantiles are what the fan plot
                # reads and the raw few are what makes it legible.
                qs = np.percentile(sv, [5, 25, 50, 75, 95], axis=0)
                sv_raw[fam] = sv.astype(np.float32).tolist()
                fams[fam] = {
                    "q": {k: [round(float(v), 3) for v in qs[i]]
                          for i, k in enumerate(("p5", "p25", "p50", "p75", "p95"))},
                    "unperturbed": [round(float(v), 3) for v in sv[0]],
                    "members": [[round(float(v), 3) for v in row]
                                for row in sv[:5]],
                    "all_members_final_year": [
                        round(float(v), 3) for v in sv[:, -12:].mean(1)],
                }
                print(f"  {label} {start} {fam}: month-1 p5/p50/p95 "
                      f"{qs[0][0]:+.2f}/{qs[2][0]:+.2f}/{qs[4][0]:+.2f} · "
                      f"month-240 {qs[0][-1]:+.2f}/{qs[2][-1]:+.2f}/{qs[4][-1]:+.2f} Sv (des)",
                      flush=True)
            entry["starts"][start] = {"context_end": months[s_end],
                                      "roll_ym": roll_ym, "families": fams,
                                      "_raw": sv_raw}
        results["heads"][label] = entry

    # ---- POOLED across heads: the ensemble that is "the model" ------------
    # Seed spread is a real uncertainty (E-010: sd ~0.12 on the probe), so
    # pooling every member of every head is the honest fan — not the spread
    # of one lucky seed. Per-head quantiles stay above for decomposition.
    pooled = {}
    for start in {s for h in results["heads"].values() for s in h["starts"]}:
        pooled[start] = {}
        for fam in ("ic", "sde"):
            stack = []
            roll_ym = None
            for lab, h in results["heads"].items():
                st = h["starts"].get(start)
                if not st or fam not in st.get("_raw", {}):
                    continue
                stack.append(np.asarray(st["_raw"][fam]))
                roll_ym = st["roll_ym"]
            if not stack:
                continue
            allsv = np.concatenate(stack, 0)                     # [heads*M, N]
            q = np.percentile(allsv, [5, 25, 50, 75, 95], axis=0)
            pooled[start][fam] = {
                "n_members": int(allsv.shape[0]),
                "roll_ym": roll_ym,
                "q": {k: [round(float(v), 3) for v in q[i]]
                      for i, k in enumerate(("p5", "p25", "p50", "p75", "p95"))},
                "decadal_mean": [round(float(allsv[:, i:i + 120].mean()), 3)
                                 for i in range(0, allsv.shape[1] - 119, 120)],
                "p_below_start": [
                    round(float((allsv[:, i] < allsv[:, 0].mean()).mean()), 3)
                    for i in range(allsv.shape[1])],
            }
    results["pooled"] = pooled
    for h in results["heads"].values():
        for st in h["starts"].values():
            st.pop("_raw", None)

    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with open(a.out, "w") as f:
        json.dump(results, f)
    print("wrote", a.out, f"({os.path.getsize(a.out):,} bytes)")


if __name__ == "__main__":
    main()

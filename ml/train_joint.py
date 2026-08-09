#!/usr/bin/env python3
"""Joint stage-1 + stage-2 training: backpropagate the forecast loss into the codec.

WHY THIS EXISTS
---------------
Stage 1 optimises "reconstruct this pixel-month from its neighbours". Nothing
in that objective asks the embedding to be PREDICTABLE FORWARD IN TIME, which
is what an AMOC forecast actually needs. Stage 2 then has to work with whatever
stage 1 happened to leave behind. The suspicion this script tests is that the
codec spends capacity on high-variance, low-persistence structure (mesoscale
noise) that reconstructs well and forecasts badly — which would explain the
result that motivated the whole experiment: on the quarter-degree tensor,
going from 0.92M to 40.7M parameters moved the pooled RAPID probe by +0.011,
and the 50k->1M step sweep was flat. Both cheap scaling axes are closed
(ml/EXPERIMENTS.md, E-002 and E-003), so the next lever is the OBJECTIVE.

THE COST, STATED HONESTLY
-------------------------
Stage 2 is cheap because it runs on a CACHE of frozen embeddings: the codec is
evaluated once per (month, pixel) and never again. Joint training destroys that
cache by construction — every step must re-embed its whole K-month window WITH
gradient, so one step costs B*K encoder forwards instead of B. At K=12 that is
12x the stage-1 cost per step. This is why the script warm-starts from a
finished codec and runs a short fine-tune rather than training from scratch.

THE LOSS (Chris's design, 2026-08-09)
-------------------------------------
"combine the two losses such that if one is high, the overall loss is high. so
we don't have to hardcode anything." That is exactly Chebyshev scalarisation:
minimise the WORSE of the objectives rather than a hand-weighted sum.

Two things make it work in practice:

1. NORMALISE EACH LOSS BY ITS OWN REFERENCE. The raw losses live on different
   scales (recon ~0.09, forecast ~0.78), so a raw max() would always select the
   larger one and silently become a single-objective run. Each loss is divided
   by the value it must not lose to: reconstruction by the frozen codec's own
   reconstruction loss, forecast by the persistence forecast on the same batch.
   Both are then "1.0 = as good as the thing I must beat", and comparable.
2. SMOOTH THE MAX. A hard max backpropagates into only one objective per step,
   which is noisy. log-sum-exp with sharpness `a` is a smooth upper bound on
   max that always passes some gradient to both, and -> max as a -> inf.

--loss-mode sum keeps the conventional L_rec + lambda*L_fore for comparison,
because a new objective that is never compared to the old one is a belief.

THE FAILURE MODE THIS MUST NOT HIDE
-----------------------------------
Joint training can cheat: the degenerate optimum of a pure forecast loss is a
CONSTANT embedding, which has zero forecast error and zero information. The
reconstruction term forbids it, but only if someone is watching — so r_rec is
logged every step and the run is declared collapsed if it degrades past
--collapse-at, no matter how good the forecast loss looks.
"""
import argparse
import json
import os
import time

import numpy as np
import torch

from model import codec_from_ckpt, gather_px
from temporal import TemporalTransformer
from trainprobe import anomaly_transform

HERE = os.path.dirname(os.path.abspath(__file__))
CKPT_DIR = "/opt/earth-cache/ckpt"


def parse():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default=os.path.join(HERE, "cache", "family3_na025.npz"))
    p.add_argument("--out", default=os.path.join(HERE, "runs", "joint"))
    p.add_argument("--resume", required=True,
                   help="codec checkpoint to warm-start from: a tag under "
                        "/opt/earth-cache/ckpt or a path. Comma-separated "
                        "candidates allowed; the first present is used. A "
                        "leading ! makes a missing checkpoint fatal instead "
                        "of starting from a random codec, which for THIS "
                        "script would silently answer a different question.")
    p.add_argument("--steps", type=int, default=4000)
    p.add_argument("--batch", type=int, default=64,
                   help="PIXELS per step. Each costs K encoder forwards, so "
                        "the effective stage-1 batch is batch*K.")
    p.add_argument("--K", type=int, default=12,
                   help="context months. Stage 2 uses 24; the joint phase "
                        "defaults to 12 because every one of them is a "
                        "gradient-carrying encoder forward.")
    p.add_argument("--lr", type=float, default=3e-5,
                   help="deliberately ~10x below stage-1 LR: this is a "
                        "fine-tune of a converged codec, not a training run.")
    p.add_argument("--lr-temporal", type=float, default=3e-4,
                   help="the temporal head is fresh, so it gets a normal LR")
    p.add_argument("--temporal-d-model", type=int, default=192)
    p.add_argument("--temporal-layers", type=int, default=4)
    p.add_argument("--loss-mode", default="lse", choices=["lse", "max", "sum"])
    p.add_argument("--lse-sharpness", type=float, default=4.0)
    p.add_argument("--lam", type=float, default=1.0,
                   help="only used by --loss-mode sum")
    p.add_argument("--mask-ratio", type=float, default=0.5)
    p.add_argument("--collapse-at", type=float, default=1.10,
                   help="abort if FIXED-batch recon (vs the frozen codec) "
                        "recon) exceeds this. 1.10 = 'reconstruction may not "
                        "get more than 10%% worse than the codec we started "
                        "from'.")
    p.add_argument("--check-every", type=int, default=100,
                   help="steps between FIXED-batch reconstruction checks. This "
                        "is the collapse guard's only input; it is cheap "
                        "(one eval-mode forward on a held batch) and, unlike "
                        "the training loss, its variance does not come from "
                        "the data draw.")
    p.add_argument("--collapse-warmup", type=int, default=50,
                   help="steps before the tripwire arms. r_rec is measured on "
                        "one batch and is noisy; a fresh temporal head also "
                        "perturbs the codec hardest in the first few steps, "
                        "so an unarmed warmup stops the guard from killing "
                        "runs for transients rather than for collapse.")
    p.add_argument("--holdout-years", default="2009,2017,2023")
    p.add_argument("--holdout-lon", default="-45,-25")
    p.add_argument("--ref-batches", type=int, default=20,
                   help="batches used to measure the FROZEN references before "
                        "training starts")
    p.add_argument("--smoke", action="store_true")
    return p.parse_args()


def resolve_ckpt(spec, require):
    cands = [c.strip() for c in spec.lstrip("!").split(",") if c.strip()]
    for c in cands:
        pth = c if os.path.sep in c else os.path.join(CKPT_DIR, c + ".pt")
        if os.path.exists(pth):
            return pth
    if require:
        raise SystemExit(
            f"--resume: none of {cands} is on this box. Refusing to start from "
            f"a random codec — this script fine-tunes a FINISHED one, and a "
            f"fresh start would answer a different question under the same "
            f"doc string.")
    return None


def main():
    a = parse()
    require = a.resume.startswith("!")
    os.makedirs(a.out, exist_ok=True)
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    d = np.load(a.data, allow_pickle=False)
    X, months = d["X"], [str(m) for m in d["months"]]
    lats, lons, chan = d["lats"], d["lons"], [str(c) for c in d["chan"]]
    T, H, W, C = X.shape
    moy = np.array([int(m[5:7]) - 1 for m in months])
    hold = set(a.holdout_years.split(","))
    t_hold = np.array([m[:4] in hold for m in months])
    lo, hi = (float(v) for v in a.holdout_lon.split(","))
    x_hold = (lons >= lo) & (lons < hi)
    print(f"X [T={T} H={H} W={W} C={C}] on {dev}", flush=True)

    # Same anomaly space the codec was trained in — a joint fine-tune in a
    # different space would be fine-tuning on a distribution shift.
    Xa, dynamic = anomaly_transform(X, moy, t_hold, x_hold)
    OBS = torch.from_numpy(np.isfinite(Xa))
    np.nan_to_num(Xa, nan=0.0, copy=False)
    Xt = torch.from_numpy(Xa)

    ck_path = resolve_ckpt(a.resume, require)
    ck = torch.load(ck_path, map_location=dev, weights_only=False)
    if len(ck["chan"]) != C:
        raise SystemExit(f"codec has {len(ck['chan'])} channels, tensor has {C}")
    codec = codec_from_ckpt(ck, C)
    codec.load_state_dict(ck["model"])
    codec.to(dev).train()
    d_z, patch = int(ck["d_z"]), int(ck.get("args", {}).get("patch", 1))
    print(f"warm-start: {ck_path} (step {ck.get('step')}, d_z={d_z}, patch={patch})",
          flush=True)

    tmp = TemporalTransformer(d_z=d_z, d_model=a.temporal_d_model, n_heads=4,
                              n_layers=a.temporal_layers, k_max=max(a.K, 36)).to(dev)
    n_codec = sum(p_.numel() for p_ in codec.parameters())
    n_tmp = sum(p_.numel() for p_ in tmp.parameters())
    print(f"codec {n_codec/1e6:.2f}M + temporal {n_tmp/1e6:.2f}M params", flush=True)

    opt = torch.optim.AdamW(
        [{"params": codec.parameters(), "lr": a.lr},
         {"params": tmp.parameters(), "lr": a.lr_temporal}], weight_decay=1e-4)

    ocean = np.isfinite(d["X"][..., 0]).any(axis=0)
    ys, xs = np.where(ocean)
    train_px = np.where(~x_hold[xs])[0]
    ok_t = np.array([(t + a.K) < T and not t_hold[t:t + a.K + 1].any()
                     for t in range(T)])
    starts = np.where(ok_t)[0]
    if a.smoke:
        a.steps, a.batch, a.ref_batches = 12, 8, 3
    print(f"{len(train_px):,} train pixels · {len(starts)} valid window starts",
          flush=True)
    rng = np.random.default_rng(0)

    def sample():
        """One batch: B pixels x (K+1) consecutive months."""
        pi = rng.choice(train_px, a.batch)
        t0 = rng.choice(starts, a.batch)
        return ys[pi], xs[pi], t0

    def embed_window(py, px_, t0, grad=True):
        """Embed B pixels over K+1 months. [B, K+1, d_z]

        This is the expensive part and the reason joint training cannot use
        the stage-2 embedding cache: every one of these forwards carries a
        gradient into the codec.
        """
        zs = []
        ctx = torch.zeros(len(py), 4, device=dev)
        for k in range(a.K + 1):
            tt = torch.as_tensor(t0 + k, dtype=torch.long)
            yy = torch.as_tensor(py, dtype=torch.long)
            xx = torch.as_tensor(px_, dtype=torch.long)
            if patch > 1:
                v, o = gather_px(Xt, OBS, tt, yy, xx, patch)
            else:
                v, o = Xt[tt, yy, xx], OBS[tt, yy, xx]
            m = moy[t0 + k]
            cc = torch.stack([
                torch.as_tensor(np.sin(2 * np.pi * m / 12), dtype=torch.float32),
                torch.as_tensor(np.cos(2 * np.pi * m / 12), dtype=torch.float32),
                torch.as_tensor(lats[py] / 90, dtype=torch.float32),
                torch.as_tensor(lons[px_] / 180, dtype=torch.float32)], 1).to(dev)
            nomask = torch.zeros(len(py), C, dtype=torch.bool, device=dev)
            z = codec.encode(v.to(dev), o.to(dev), nomask, cc)
            zs.append(z)
        return torch.stack(zs, 1), ctx

    def recon_loss(py, px_, t0, seed=None):
        """Stage 1's own objective on the SAME pixels, so the two terms are
        measured on one batch rather than on two different samples of the
        ocean."""
        tt = torch.as_tensor(t0, dtype=torch.long)
        yy = torch.as_tensor(py, dtype=torch.long)
        xx = torch.as_tensor(px_, dtype=torch.long)
        v, o = Xt[tt, yy, xx].to(dev), OBS[tt, yy, xx].to(dev)
        if seed is None:
            mask = (torch.rand(len(py), C, device=dev) < a.mask_ratio) & o
        else:
            # Deterministic mask, so the probe measures the CODEC, not the draw.
            gm = torch.Generator().manual_seed(seed)
            mask = (torch.rand(len(py), C, generator=gm) < a.mask_ratio).to(dev) & o
        m = moy[t0]
        cc = torch.stack([
            torch.as_tensor(np.sin(2 * np.pi * m / 12), dtype=torch.float32),
            torch.as_tensor(np.cos(2 * np.pi * m / 12), dtype=torch.float32),
            torch.as_tensor(lats[py] / 90, dtype=torch.float32),
            torch.as_tensor(lons[px_] / 180, dtype=torch.float32)], 1).to(dev)
        if patch > 1:
            vp, op = gather_px(Xt, OBS, tt, yy, xx, patch)
            z = codec.encode(vp.to(dev), op.to(dev), mask, cc)
        else:
            z = codec.encode(v * (~mask), o, mask, cc)
        qc = torch.arange(C, device=dev)[None, :].expand(len(py), -1)
        pred = codec.query(z, qc, torch.zeros(len(py), C, 3, dtype=torch.long,
                                              device=dev))
        sel = mask
        if sel.sum() < 1:
            return torch.zeros((), device=dev)
        return ((pred - v).pow(2) * sel).sum() / sel.sum()

    def forecast_terms(zseq, mseq, sctx):
        """Model forecast loss and the PERSISTENCE loss on the same window."""
        pred, _ = tmp(zseq[:, :a.K], mseq, sctx)
        tgt = zseq[:, 1:a.K + 1]
        l_model = (pred - tgt).pow(2).mean()
        l_pers = (zseq[:, :a.K] - tgt).pow(2).mean()   # z_t as the forecast
        return l_model, l_pers

    def month_seq(t0):
        """[B, K, 2] — sin/cos of month-of-year for each step of the window.
        t0 is a VECTOR of start months, so build [B, K] first and take the
        trig last; stacking the trig first transposes the batch and step
        axes and produces a tensor that concatenates against z_seq only by
        accident of shape."""
        m = np.stack([moy[t0 + k] for k in range(a.K)], 1)      # [B, K]
        ang = 2 * np.pi * m / 12
        mm = np.stack([np.sin(ang), np.cos(ang)], -1).astype(np.float32)
        return torch.as_tensor(mm).to(dev)                      # [B, K, 2]

    # ---- FROZEN REFERENCES ------------------------------------------------
    # Both losses are scored against what they must not lose to. Measured
    # BEFORE any update, on the codec exactly as it arrived.
    #
    # The reconstruction reference uses a FIXED probe batch with a FIXED mask,
    # re-measured on that same batch throughout training. The first version
    # compared a single TRAINING batch against a 20-batch mean, which is not a
    # comparison at all: per-batch reconstruction is heavy tailed (random
    # mask, 64 pixels), so a running mean is dominated by rare spikes. On #91
    # every logged r_rec sat below 1.0 (mean 0.88) while the EMA of all steps
    # sat at 1.05 and tripped the guard at step 419. The guard was measuring
    # outliers and it killed a healthy run. Same batch, same mask, eval mode:
    # now the only thing that can move this number is the codec itself.
    probe_py, probe_px, probe_t0 = sample()
    PROBE_MASK_SEED = 4321

    def recon_probe():
        was = codec.training
        codec.eval()
        with torch.no_grad():
            v = recon_loss(probe_py, probe_px, probe_t0, seed=PROBE_MASK_SEED)
        if was:
            codec.train()
        return float(v)

    codec.eval()
    with torch.no_grad():
        perss = []
        for _ in range(a.ref_batches):
            py, px_, t0 = sample()
            zseq, _ = embed_window(py, px_, t0)
            sctx = torch.cat([zseq[:, 0], torch.as_tensor(
                np.stack([lats[py] / 90, lons[px_] / 180], 1),
                dtype=torch.float32).to(dev)], 1)
            _, lp = forecast_terms(zseq, month_seq(t0), sctx)
            perss.append(float(lp))
    ref_fore = float(np.mean(perss))
    codec.train()
    ref_rec = recon_probe()
    print(f"frozen references: recon {ref_rec:.5f} · persistence {ref_fore:.5f}",
          flush=True)

    metrics = os.path.join(a.out, "metrics.jsonl")
    with open(metrics, "a") as f:
        f.write(json.dumps({"joint_config": {
            "steps": a.steps, "batch": a.batch, "K": a.K, "lr": a.lr,
            "lr_temporal": a.lr_temporal, "loss_mode": a.loss_mode,
            "lse_sharpness": a.lse_sharpness, "lam": a.lam,
            "temporal_d_model": a.temporal_d_model,
            "temporal_layers": a.temporal_layers,
            "codec_params_M": round(n_codec / 1e6, 3),
            "temporal_params_M": round(n_tmp / 1e6, 3),
            "ref_recon": ref_rec, "ref_persistence": ref_fore,
            "warm_start": os.path.basename(ck_path), "d_z": d_z, "patch": patch,
        }}) + "\n")

    def combine(r_rec, r_fore):
        """r_* are losses RELATIVE to their own reference (1.0 = as good as
        the thing we must not lose to)."""
        if a.loss_mode == "sum":
            return r_rec + a.lam * r_fore
        if a.loss_mode == "max":
            return torch.maximum(r_rec, r_fore)
        s = a.lse_sharpness                        # smooth max
        return torch.logsumexp(torch.stack([r_rec * s, r_fore * s]), 0) / s

    print(f"joint training ({a.loss_mode}) …", flush=True)
    t_start = time.time()
    log_every = max(1, a.steps // 100)
    collapsed = False
    # The tripwire watches a SMOOTHED r_rec. Per-batch reconstruction on 64
    # pixels swings several percent from sampling alone, so testing the raw
    # value aborts healthy runs on noise — the smoke run tripped at step 2 on
    # exactly that. An EMA plus a warmup makes the guard fire for a trend,
    # which is what collapse actually is.
    # The guard now reads the FIXED probe every --check-every steps and needs
    # TWO consecutive bad readings. One noisy statistic killed two runs today;
    # a low-variance statistic that has to be bad twice will not.
    rec_checks = []
    bad_streak = 0
    for s in range(1, a.steps + 1):
        py, px_, t0 = sample()
        l_rec = recon_loss(py, px_, t0)
        zseq, _ = embed_window(py, px_, t0)
        sctx = torch.cat([zseq[:, 0], torch.as_tensor(
            np.stack([lats[py] / 90, lons[px_] / 180], 1),
            dtype=torch.float32).to(dev)], 1)
        l_fore, l_pers = forecast_terms(zseq, month_seq(t0), sctx)

        r_rec = l_rec / max(ref_rec, 1e-9)
        r_fore = l_fore / max(ref_fore, 1e-9)
        loss = combine(r_rec, r_fore)
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(
            list(codec.parameters()) + list(tmp.parameters()), 1.0)
        opt.step()

        rr = float(r_rec.detach())
        if s % a.check_every == 0:
            chk = recon_probe() / max(ref_rec, 1e-9)
            rec_checks.append((s, round(chk, 4)))
            bad_streak = bad_streak + 1 if chk > a.collapse_at else 0
            with open(metrics, "a") as f:
                f.write(json.dumps({"joint_step": s, "r_rec_probe": round(chk, 4),
                                    "bad_streak": bad_streak}) + "\n")
            print(f"  probe @{s}: r_rec(fixed batch) {chk:.3f}"
                  f"{'  BAD' if chk > a.collapse_at else ''}", flush=True)
        if s % log_every == 0 or s == a.steps:
            rec = {"joint_step": s, "r_rec": round(float(r_rec), 4),
                   "r_rec_probe": (rec_checks[-1][1] if rec_checks else None),
                   "r_fore": round(float(r_fore), 4),
                   "loss": round(float(loss), 4),
                   "l_rec": round(float(l_rec), 5),
                   "l_fore": round(float(l_fore), 5),
                   "l_pers": round(float(l_pers), 5),
                   "wall_s": round(time.time() - t_start, 1)}
            with open(metrics, "a") as f:
                f.write(json.dumps(rec) + "\n")
        if s % max(1, a.steps // 10) == 0:
            print(f"  step {s:>6}/{a.steps}  r_rec {float(r_rec):.3f}  "
                  f"r_fore {float(r_fore):.3f}  ({time.time()-t_start:.0f}s)",
                  flush=True)
        # THE TRIPWIRE. A forecast loss that improves while reconstruction
        # rots is the degenerate solution, not a discovery.
        if s > a.collapse_warmup and bad_streak >= 2:
            print(f"  COLLAPSE: reconstruction on the FIXED probe batch has "
                  f"been worse than {a.collapse_at}x the frozen codec for two "
                  f"consecutive checks (latest {rec_checks[-1][1]:.3f} at step "
                  f"{s}). Stopping; these weights are NOT a valid codec.",
                  flush=True)
            with open(metrics, "a") as f:
                f.write(json.dumps({"joint_collapsed": {
                    "step": s, "r_rec_probe": rec_checks[-1][1],
                    "r_fore": float(r_fore.detach())}}) + "\n")
            collapsed = True
            break

    # Save in the SAME format train.py writes, so every downstream probe
    # (probe_kfold, probe_head, temporal) reads it with no special case.
    blob = {"model": codec.state_dict(), "chan": ck["chan"], "d_z": d_z,
            "norm": ck.get("norm"), "args": {**ck.get("args", {}),
                                             "joint": vars(a)},
            "step": int(ck.get("step", 0)), "tag": os.environ.get("CKPT_TAG", ""),
            "joint_collapsed": collapsed}
    torch.save(blob, os.path.join(a.out, "pixelmae.pt"))
    torch.save({"model": tmp.state_dict(), "args": vars(a)},
               os.path.join(a.out, "temporal_joint.pt"))
    print(f"saved {a.out}/pixelmae.pt (collapsed={collapsed})", flush=True)


if __name__ == "__main__":
    main()

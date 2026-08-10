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

THE DEGENERATE SOLUTION, AND WHY THE LOSS ALREADY FORBIDS IT
------------------------------------------------------------
Joint training can cheat: the degenerate optimum of a pure forecast loss is a
CONSTANT embedding, which has zero forecast error and zero information.

This script used to carry a tripwire that ABORTED a run when reconstruction
degraded. Chris removed it, and he was right twice over. First, it was never
part of the design he asked for — the whole point of combining the losses so
that "if one is high, the overall loss is high" is that the objective handles
this itself: the moment r_rec rises above r_fore, the smooth max IS r_rec, and
every subsequent gradient step pushes reconstruction back down. A run that
drifts is self-correcting; a guard on top is redundant. Second, the guard was
wrong in practice — it read a heavy-tailed per-batch ratio and killed two
healthy runs (#86, #91) on outliers.

So nothing here fails a job on a metric. Reconstruction on a fixed probe batch
is still MEASURED and logged every --check-every steps, because reading what
the codec did is the point; it just never stops anything. If a run does
degenerate, that is a RESULT — the probe ladder scores the resulting codec and
the number says so, which is far better evidence than an abort ever was.

--loss-mode data · THE ABOVE IS SUPERSEDED (E-006, built 2026-08-10)
--------------------------------------------------------------------
Everything from "NORMALISE EACH LOSS BY ITS OWN REFERENCE" down is an attempt
to referee two quantities that were never commensurable, and it failed four
times: a constant denominator (#102, encoder shrank z 40x), a detached one
(#103/#104, 1099x), a scale-free ratio with a second exploit (#107/#108,
inflate the baseline, 250x the other way), and a twin trained alongside
(#109, cancelled). Chris, after the fourth: *"the loss term should just be
(1) how much have we failed to predict X + (2) how much have we failed to
predict Y. that's it."*

The reason alignment kept being hard is that ONE OF THE TWO FAILURES WAS
MEASURED IN A SPACE THE MODEL INVENTS. Reconstruction error is scored against
observed channels — external and ungameable. Forecast error was scored
against `z`, which the encoder may rescale freely, so it had no units at all
and every denominator was refereeing a quantity with no fixed scale.

`--loss-mode data` decodes the forecast back into the data before scoring it:

    L = mean_masked (x̂_t − x_t)² / var_c(x)  +  mean_obs (x̂_{t+1..t+K} − x)² / var_c(x)

Both terms are "failed to predict real, standardised observations", so a plain
sum is the whole combination rule. The shrinkage degeneracy is not closed,
penalised or guarded — under z → s·z with decoder → decoder/s the decoded
field is unchanged, so dL/ds is EXACTLY zero and there is no free direction
for the cheat to live in (`tests/test_e006_algebra.py`, symbolically; and
`tests/test_e006_gauge.py` numerically on the real code path, which is the
version that would catch a z-space term leaking back into the objective).

`var_c` is per CHANNEL, computed once from the training data. Per channel and
not one global number, or the loss is simply whichever channel has the largest
anomaly variance; from the data and not from the model, which is the one
distinction all four failures turned on.

Persistence still appears in the logs — decoded into the same units, so
`l_pers_data` reads as "how well does last month's field predict this one" —
and nowhere in the objective. A diagnostic in the loss is what made the
earlier versions depend on another run's arbitrary stopping point.
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
    p.add_argument("--loss-mode", default="lse",
                   choices=["lse", "max", "sum", "data", "data-lse"],
                   help="'data' is E-006: BOTH terms measured in the data's "
                        "own units, as fractions of each channel's observed "
                        "variance, and added. It takes no reference, no "
                        "--ref-fore, no twin and no lambda, because there is "
                        "nothing left to referee — see the long note in the "
                        "module docstring. 'data-lse' is the same two terms "
                        "under the smooth max, kept only so the sum can be "
                        "compared to it.")
    p.add_argument("--lse-sharpness", type=float, default=4.0)
    p.add_argument("--lam", type=float, default=1.0,
                   help="only used by --loss-mode sum")
    p.add_argument("--mask-ratio", type=float, default=0.5)
    p.add_argument("--check-every", type=int, default=100,
                   help="steps between FIXED-batch reconstruction checks. "
                        "Pure instrumentation — it stops nothing. Cheap (one "
                        "eval-mode forward on a held batch) and, unlike the "
                        "training loss, its variance does not come from the "
                        "data draw, so the logged series is readable.")
    p.add_argument("--holdout-years", default="2009,2017,2023")
    p.add_argument("--holdout-lon", default="-45,-25")
    p.add_argument("--ref-fore", type=float, default=0.0,
                   help="the frozen-codec control's converged l_fore/l_pers. "
                        "0 = report raw l_fore/l_pers, i.e. BE the control.\n\n"
                        "Note this divides a RATIO, not a loss. The forecast "
                        "term is normalised by the batch's own persistence "
                        "before this multiplier is applied, because l_fore is "
                        "an MSE in z-space and a fixed denominator pays the "
                        "encoder to shrink z instead of to forecast — 40x in "
                        "1200 steps on #102, at almost no reconstruction "
                        "cost, since the decoder just rescales.\n\n"
                        "This matters more than it looks. Normalising "
                        "reconstruction by the frozen codec puts it at ~1.0, "
                        "while normalising forecast by PERSISTENCE puts it at "
                        "~0.3 — so the two are never comparable, the smooth "
                        "max is always reconstruction, and ~95%% of every "
                        "gradient step goes to reconstruction (measured on "
                        "#94: weights 0.957/0.043). The forecast objective "
                        "then cannot move the codec at all, and the "
                        "experiment answers nothing. Passing the frozen-codec "
                        "control's own forecast loss here puts BOTH terms on "
                        "'1.0 = as good as the pipeline we started with', "
                        "which is what makes 'whichever is worse' meaningful.")
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

    # ---- PER-CHANNEL VARIANCE OF THE OBSERVED FIELD (E-006) ---------------
    # Computed BEFORE the NaNs are zeroed, from finite training entries only,
    # so it is a property of the data and of nothing else. That is the whole
    # distinction the four retracted normalisations turned on: a denominator
    # the model can move is a term in the objective and will be optimised;
    # var(x) has dvar/dtheta = 0 for every parameter theta, which is what
    # makes a constant denominator legitimate here and poison in E-004.
    #
    # PER CHANNEL, not one global number. anomaly_transform z-scores the
    # DYNAMIC channels, so those come out at ~1.0 — but the static ones
    # (bathymetry, coordinates, land masks) keep their native scale, and a
    # single global variance would hand the loss to whichever of those is
    # largest. Channel by channel, and one channel at a time: the fancy-index
    # form materialises the whole [T,H,W,C] array per channel and OOM-killed
    # every probe on a 7 GB box in August.
    varc_np = np.ones(C, dtype=np.float32)
    keep_t = ~t_hold
    keep_x = ~x_hold
    for c in range(C):
        col = Xa[keep_t][:, :, keep_x, c]
        col = col[np.isfinite(col)]
        # A channel with no finite training entries, or a constant one, gets
        # 1.0 rather than an epsilon: dividing by ~0 would make that channel
        # the entire loss, which is the failure mode this line exists inside.
        v = float(col.var()) if col.size else 0.0
        varc_np[c] = v if v > 1e-8 else 1.0
    print(f"per-channel variance for the data-space loss: "
          f"median {np.median(varc_np):.3f}, "
          f"range [{varc_np.min():.3g}, {varc_np.max():.3g}]", flush=True)

    np.nan_to_num(Xa, nan=0.0, copy=False)
    Xt = torch.from_numpy(Xa)
    varc = torch.from_numpy(varc_np).to(dev)          # [C]
    DATA_SPACE = a.loss_mode in ("data", "data-lse")
    # REFUSE THE CONTRADICTION AT THE INPUTS, where it has cost nothing. Both
    # of these ask for a referee between the two terms, and the data-space
    # loss exists precisely because there is nothing left to referee — running
    # anyway would produce a healthy-looking run that cannot test its own
    # hypothesis, which is the failure ml/CLAUDE.md §4.11 is about.
    if DATA_SPACE and a.ref_fore != 0.0:
        raise SystemExit(
            f"--loss-mode {a.loss_mode} takes no --ref-fore (got {a.ref_fore}). "
            "The forecast term is scored against observed channels, not "
            "against z, so there is no scale to referee and no constant to "
            "copy from another run. Drop --ref-fore.")
    if DATA_SPACE and a.lam != 1.0:
        raise SystemExit(
            f"--loss-mode {a.loss_mode} takes no --lam (got {a.lam}). Both "
            "terms are fractions of the same observed variance; a weight "
            "would be re-introducing by hand the arbitrary constant this "
            "formulation removes.")

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

    # ---- TWIN REFERENCE ---------------------------------------------------
    # A second head, trained on the FROZEN codec's embeddings, in this same
    # job, on the same batches, from the same init, with the same optimiser
    # and learning rate. It exists to delete a constant.
    #
    # The problem it solves: r_rec's reference is a genuine property of the
    # frozen codec — hand it a fixed batch, read the error, one line at step
    # 0. r_fore's is not. "How well does the frozen codec forecast" is a
    # property of the codec PLUS a head that has been trained, and at step 0
    # that head is random, so there is nothing to measure. Until now the
    # number came from a separate control run (#101) and was pasted in by
    # hand as --ref-fore. Then #101's curve turned out not to converge — its
    # block means fell monotonically to the last block, ~0.018 per 1200
    # steps and not flattening — which means the constant was never a
    # property of anything. It was a function of how long the control ran.
    #
    # Two heads at the same point in their own training, on identical
    # batches, is the comparison that was actually wanted. The arbitrary
    # stopping point cancels because both sides have it.
    ref_twin = a.ref_fore < 0                      # --ref-fore -1 selects it
    codec_ref = tmp_ref = opt_ref = None
    if ref_twin:
        import copy
        codec_ref = copy.deepcopy(codec).to(dev).eval()
        for p_ in codec_ref.parameters():
            p_.requires_grad_(False)
        tmp_ref = TemporalTransformer(d_z=d_z, d_model=a.temporal_d_model,
                                      n_heads=4, n_layers=a.temporal_layers,
                                      k_max=max(a.K, 36)).to(dev)
        # EXACTLY the same init as `tmp`, copied rather than re-seeded: the
        # only difference between the two branches must be which embeddings
        # the head consumes, or the comparison is confounded with init luck.
        tmp_ref.load_state_dict(tmp.state_dict())
        opt_ref = torch.optim.AdamW(tmp_ref.parameters(), lr=a.lr_temporal,
                                    weight_decay=1e-4)
        print(f"twin reference: a second {n_tmp/1e6:.2f}M head on the FROZEN "
              f"codec, same batches and same init — --ref-fore is not used",
              flush=True)

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

    def embed_window(py, px_, t0, grad=True, net=None):
        """Embed B pixels over K+1 months. [B, K+1, d_z]

        This is the expensive part and the reason joint training cannot use
        the stage-2 embedding cache: every one of these forwards carries a
        gradient into the codec. `net` overrides which codec does the
        embedding — the twin reference passes the frozen copy, under
        no_grad, which is the one case where the cache WOULD apply if the
        memory were free (516 x 84,405 x 64 fp32 is ~11 GB).
        """
        enc = net if net is not None else codec
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
            z = enc.encode(v.to(dev), o.to(dev), nomask, cc)
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
        err = (pred - v).pow(2)
        if DATA_SPACE:
            # The SAME per-channel denominator the forecast term uses, so the
            # two are in one unit and a plain sum needs no weight. In the
            # reference modes this division is deliberately absent: there the
            # term is divided by the frozen codec's own reconstruction loss
            # instead, and doing both would normalise twice.
            err = err / varc
        return (err * sel).sum() / sel.sum()

    def decode_z(z):
        """z -> the field it stands for, through the codec's OWN decoder.

        This is the single line E-006 turns on. Scoring the forecast here
        rather than in z-space is what removes the shrinkage degeneracy: the
        target is an observation that does not move, and z enters the loss
        only through this call, so rescaling the encoder buys nothing.
        """
        n = z.shape[0]
        qc = torch.arange(C, device=dev)[None, :].expand(n, -1)
        return codec.query(z, qc, torch.zeros(n, C, 3, dtype=torch.long,
                                              device=dev))

    def forecast_data_terms(zseq, mseq, sctx, py, px_, t0, head=None):
        """Forecast error IN THE DATA'S UNITS, and persistence beside it.

        The head predicts z for months t+1..t+K; every one of them is decoded
        and scored against the observed channels at that month, over observed
        entries only. One `query` call over the flattened [B*K] batch rather
        than K calls — same arithmetic, one kernel launch.

        Returns (model, persistence). Persistence decodes z_t itself, i.e.
        "last month's field is my forecast", in the same units, and is
        LOGGED ONLY — it is a diagnostic, and a diagnostic in the objective is
        what made every earlier version depend on another run's stopping
        point.
        """
        pred_z, _ = (head or tmp)(zseq[:, :a.K], mseq, sctx)      # [B,K,d_z]
        B = pred_z.shape[0]
        yy = torch.as_tensor(py, dtype=torch.long)
        xx = torch.as_tensor(px_, dtype=torch.long)
        tgt, obs = [], []
        for k in range(a.K):
            tt = torch.as_tensor(t0 + k + 1, dtype=torch.long)
            tgt.append(Xt[tt, yy, xx])
            obs.append(OBS[tt, yy, xx])
        tgt = torch.stack(tgt, 1).to(dev)                          # [B,K,C]
        obs = torch.stack(obs, 1).to(dev)
        xh = decode_z(pred_z.reshape(B * a.K, -1)).reshape(B, a.K, C)
        n = obs.sum().clamp_min(1)
        l_model = (((xh - tgt).pow(2) / varc) * obs).sum() / n
        with torch.no_grad():
            xp = decode_z(zseq[:, :a.K].reshape(B * a.K, -1)).reshape(B, a.K, C)
            l_pers = (((xp - tgt).pow(2) / varc) * obs).sum() / n
        return l_model, l_pers

    def forecast_terms(zseq, mseq, sctx, head=None):
        """Model forecast loss and the PERSISTENCE loss on the same window.

        `head` selects which temporal model forecasts; the twin reference
        passes its own. Persistence is computed from zseq, so each branch is
        scored against the baseline IN ITS OWN embedding space — which is
        what makes the live/frozen ratio scale-free.
        """
        pred, _ = (head or tmp)(zseq[:, :a.K], mseq, sctx)
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
            if DATA_SPACE:
                # Persistence measured in the same units the loss uses, so
                # the logged z_shrink series stays meaningful (it is now a
                # ratio of DATA-space persistence, which the encoder cannot
                # move by rescaling — a flat line there is the expected
                # result, and a moving one would be news).
                _, lp = forecast_data_terms(zseq, month_seq(t0), sctx,
                                            py, px_, t0)
            else:
                _, lp = forecast_terms(zseq, month_seq(t0), sctx)
            perss.append(float(lp))
    # KEPT FOR THE RECORD ONLY. This step-0 persistence used to be the
    # forecast denominator; it no longer is (see the r_fore line in the
    # loop), because a constant denominator on a z-space MSE pays the
    # encoder to shrink z. It is still logged, because watching it against
    # the running l_pers is exactly how the shrinkage was caught.
    ref_fore = float(np.mean(perss))          # persistence at step 0
    ref_fore_mult = a.ref_fore                # frozen-codec control's l_fore/l_pers
    if DATA_SPACE:
        print("  data-space loss: no forecast reference of any kind. Both "
              "terms are fractions of the observed per-channel variance, so "
              "1.0 means 'no better than climatology' for BOTH, in the same "
              "units, with no control run anywhere.", flush=True)
    elif ref_fore_mult > 0:
        print(f"  forecast reference from the frozen-codec control: "
              f"{ref_fore_mult:.4f} (its converged l_fore/l_pers)", flush=True)
    elif ref_twin:
        print("  --ref-fore twin: the reference is trained alongside, so no "
              "constant is read from any previous run", flush=True)
    else:
        print("  no --ref-fore: r_fore is raw l_fore/l_pers, so THIS RUN "
              "measures the reference other runs divide by", flush=True)
    codec.train()
    ref_rec = recon_probe()
    print(f"frozen references: recon {ref_rec:.5f} · "
          f"persistence-at-step-0 {ref_fore:.5f} (logged, not a denominator)",
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
            "ref_recon": ref_rec, "ref_forecast": ref_fore,
            "ref_fore_mult": a.ref_fore, "ref_twin": bool(a.ref_fore < 0),
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
    # Reconstruction on the fixed probe batch, recorded so the run can be READ
    # afterwards. It aborts nothing: the loss is what keeps reconstruction
    # honest, and if it fails to, the probe ladder's number on the resulting
    # codec is the evidence — not a threshold someone picked.
    rec_checks = []

    def _log_step(s, l_rec, l_fore, l_pers, r_rec, r_fore, loss, skill_ref):
        """ONE logger for both objectives. The data-space branch was written
        with its own copy first, which is how a metrics schema quietly forks:
        two writers, one reader, and a field that exists on some runs. The
        series below are the ones the status page and every plot read, so
        every mode must write all of them or say null."""
        if s % a.check_every == 0:
            chk = recon_probe() / max(ref_rec, 1e-9)
            rec_checks.append((s, round(chk, 4)))
            with open(metrics, "a") as f:
                f.write(json.dumps({"joint_step": s,
                                    "r_rec_probe": round(chk, 4)}) + "\n")
            print(f"  probe @{s}: r_rec(fixed batch) {chk:.3f}", flush=True)
        if s % log_every == 0 or s == a.steps:
            rec = {"joint_step": s, "r_rec": round(float(r_rec), 4),
                   "r_rec_probe": (rec_checks[-1][1] if rec_checks else None),
                   "r_fore": round(float(r_fore), 4),
                   "loss": round(float(loss), 4),
                   "l_rec": round(float(l_rec), 5),
                   "l_fore": round(float(l_fore), 5),
                   "l_pers": round(float(l_pers), 5),
                   # How far the embedding has contracted since step 0. 1.0 =
                   # unchanged; #102 reached 40x in 1200 steps. This is a
                   # first-class series because a joint run that "improves"
                   # by shrinking z is indistinguishable from one that learns
                   # unless you plot it.
                   #
                   # In the data-space mode this is a ratio of DATA-space
                   # persistences, which the encoder cannot move by rescaling
                   # — so a flat line is the PREDICTION, not merely the hope,
                   # and a moving one means something real changed about how
                   # predictable the field is under this codec. Keeping the
                   # field rather than nulling it is what makes that
                   # observable at all.
                   "z_shrink": round(ref_fore / max(float(l_pers), 1e-9), 3),
                   # The twin's own scale-free skill, logged beside the live
                   # one. Both heads have had exactly the same amount of
                   # training at this step, which is the whole point: r_fore
                   # is their ratio, so the arbitrary stopping point that made
                   # #101's 0.44 meaningless cancels out of it.
                   "skill_live": round(float((l_fore / l_pers.clamp_min(1e-9)).detach()), 4),
                   "skill_ref": (round(skill_ref, 4) if skill_ref else None),
                   "space": "data" if DATA_SPACE else "z",
                   "wall_s": round(time.time() - t_start, 1)}
            with open(metrics, "a") as f:
                f.write(json.dumps(rec) + "\n")
        if s % max(1, a.steps // 10) == 0:
            print(f"  step {s:>6}/{a.steps}  r_rec {float(r_rec):.3f}  "
                  f"r_fore {float(r_fore):.3f}  ({time.time()-t_start:.0f}s)",
                  flush=True)

    for s in range(1, a.steps + 1):
        py, px_, t0 = sample()
        l_rec = recon_loss(py, px_, t0)
        zseq, _ = embed_window(py, px_, t0)
        sctx = torch.cat([zseq[:, 0], torch.as_tensor(
            np.stack([lats[py] / 90, lons[px_] / 180], 1),
            dtype=torch.float32).to(dev)], 1)
        if DATA_SPACE:
            # E-006. Both terms already in the data's units; the "r_" names
            # are kept so every downstream reader (the status page, the
            # metrics schema, the plots) needs no special case — but here
            # they are the losses themselves, not ratios to a reference.
            l_fore, l_pers = forecast_data_terms(zseq, month_seq(t0), sctx,
                                                 py, px_, t0)
            r_rec, r_fore = l_rec, l_fore
            skill_ref = None
            loss = (r_rec + r_fore if a.loss_mode == "data" else
                    torch.logsumexp(torch.stack([r_rec * a.lse_sharpness,
                                                 r_fore * a.lse_sharpness]), 0)
                    / a.lse_sharpness)
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(
                list(codec.parameters()) + list(tmp.parameters()), 1.0)
            opt.step()
            _log_step(s, l_rec, l_fore, l_pers, r_rec, r_fore, loss, None)
            continue

        l_fore, l_pers = forecast_terms(zseq, month_seq(t0), sctx)

        r_rec = l_rec / max(ref_rec, 1e-9)
        # SCALE-FREE. l_fore is an MSE in z-space, so dividing it by a
        # constant measured at step 0 rewards the encoder for making z
        # SMALLER — a direction that costs reconstruction almost nothing,
        # because the decoder simply rescales. Measured on #102: the
        # persistence baseline l_pers, which depends only on the codec,
        # fell 40x in 1200 steps (4.175 -> 0.103) and r_fore fell with it,
        # 0.353 -> 0.054, while the scale-free ratio l_fore/l_pers barely
        # moved (0.80 -> 0.64) and stayed BEHIND the frozen codec's 0.60.
        # The entire apparent forecast gain of every joint run so far was
        # the embedding shrinking. Dividing by the batch's OWN persistence
        # closes the degenerate direction by construction: shrink z and
        # numerator and denominator shrink together, for nothing.
        # NOT .detach(). Detaching the denominator was the whole bug in the
        # first attempt at this (#103/#104): it makes l_pers a constant
        # w.r.t. the parameters INSIDE the step, which is precisely the
        # condition that pays for shrinking z. Both losses are quadratic in
        # z, so under a rescale z -> a·z the ratio goes a²·l_fore / (a²·l_pers)
        # — flat, gradient exactly zero — whereas a detached denominator
        # gives a²·l_fore / const, whose gradient points straight down in a.
        # #103 reached 1099x contraction by step 600, worse and faster than
        # the run that first exposed the problem, because the fix had made
        # the forecast term matter without removing the incentive.
        r_fore = l_fore / l_pers.clamp_min(1e-9)
        skill_ref = None
        if ref_twin:
            # The twin: same batch, frozen codec, its own head and optimiser.
            # Its skill is l_fore/l_pers in ITS OWN space, so the two ratios
            # are directly comparable and the live one stays scale-free.
            with torch.no_grad():
                zr, _ = embed_window(py, px_, t0, net=codec_ref)
            sctx_r = torch.cat([zr[:, 0], torch.as_tensor(
                np.stack([lats[py] / 90, lons[px_] / 180], 1),
                dtype=torch.float32).to(dev)], 1)
            lf_r, lp_r = forecast_terms(zr, month_seq(t0), sctx_r, head=tmp_ref)
            opt_ref.zero_grad(); lf_r.backward()
            torch.nn.utils.clip_grad_norm_(tmp_ref.parameters(), 1.0)
            opt_ref.step()
            skill_ref = float(lf_r.detach() / lp_r.detach().clamp_min(1e-9))
            # Detaching HERE is correct and is not the #103 mistake. That one
            # detached l_pers, which depends on the LIVE codec, so the ratio
            # stopped being scale-free. This denominator depends only on the
            # frozen codec and the twin's own parameters — nothing the live
            # optimiser touches — so it is constant w.r.t. the live gradient
            # whether detached or not. Checked symbolically: dr/da under a
            # rescale z -> a·z is still exactly 0.
            r_fore = r_fore / max(skill_ref, 1e-9)
        elif ref_fore_mult > 0:
            # ...then put it on r_rec's footing: 1.0 must mean "as good as
            # the frozen-codec pipeline" for BOTH terms, or the smooth max
            # is not comparing like with like. ref_fore_mult is the
            # frozen-codec control's own l_fore/l_pers — a hand-copied
            # constant, and the thing --ref-fore -1 exists to abolish.
            r_fore = r_fore / ref_fore_mult
        if ref_twin and s == 1:
            # SELF-TEST, free and decisive. At step 1 nothing has been updated
            # yet: codec_ref is a copy of codec, tmp_ref is a copy of tmp, and
            # both branches see the same batch — so the two skills are the same
            # number and their ratio must be exactly 1. Anything else means the
            # twin is not a twin (wrong init copy, wrong batch, a stale
            # codec_ref) and every r_fore after it is meaningless.
            dev1 = abs(float(r_fore.detach()) - 1.0)
            print(f"  twin self-test at step 1: r_fore = "
                  f"{float(r_fore.detach()):.6f} (must be 1.000000)", flush=True)
            if dev1 > 1e-4:
                raise SystemExit(
                    f"twin reference is not a twin: r_fore = {float(r_fore.detach()):.6f} "
                    f"at step 1, off by {dev1:.2e}. Refusing to train — the "
                    f"forecast term would be normalised by the wrong thing.")
        loss = combine(r_rec, r_fore)
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(
            list(codec.parameters()) + list(tmp.parameters()), 1.0)
        opt.step()

        _log_step(s, l_rec, l_fore, l_pers, r_rec, r_fore, loss, skill_ref)

    # Save in the SAME format train.py writes, so every downstream probe
    # (probe_kfold, probe_head, temporal) reads it with no special case.
    blob = {"model": codec.state_dict(), "chan": ck["chan"], "d_z": d_z,
            "norm": ck.get("norm"), "args": {**ck.get("args", {}),
                                             "joint": vars(a)},
            "step": int(ck.get("step", 0)), "tag": os.environ.get("CKPT_TAG", ""),
            "r_rec_probe_series": rec_checks}
    torch.save(blob, os.path.join(a.out, "pixelmae.pt"))
    torch.save({"model": tmp.state_dict(), "args": vars(a)},
               os.path.join(a.out, "temporal_joint.pt"))
    last = rec_checks[-1][1] if rec_checks else float("nan")
    print(f"saved {a.out}/pixelmae.pt · final fixed-probe r_rec {last:.3f} "
          f"(reported, not enforced)", flush=True)


if __name__ == "__main__":
    main()

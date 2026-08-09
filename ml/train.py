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
    p.add_argument("--resume", default="",
                   help="continue a run from a checkpoint written by an "
                        "earlier job: path to a pixelmae.pt, or a bare tag "
                        "resolved under /opt/earth-cache/ckpt/<tag>.pt. "
                        "Restores weights, optimizer, LR schedule and the "
                        "step reached, then trains on to --steps. A "
                        "weights-only checkpoint (anything saved before "
                        "2026-08-08) still WARM-STARTS: it loads the weights "
                        "and restarts the schedule, which is stated in the "
                        "log so the run's provenance is never ambiguous. "
                        "Missing file = start fresh, loudly.")
    p.add_argument("--require-resume", action="store_true",
                   help="EXIT immediately if --resume finds no checkpoint, "
                        "instead of starting fresh. Checkpoint mirrors are "
                        "box-local, so a resume dispatch lands on the right "
                        "runner only by luck; without this a mislanded job "
                        "silently retrains from scratch for hours and calls "
                        "itself a continuation. Use it whenever the point of "
                        "the job is the EXISTING weights — e.g. a stage-2 "
                        "run over a frozen codec.")
    p.add_argument("--light-probe-every", type=int, default=0,
                   help="every N steps, run the CHEAP half of the probe (the "
                        "linear 26.5N section probe only — no mini temporal "
                        "transformer). Measured on the 10M codec the full "
                        "probe costs ~300 s and the light one ~30 s, so this "
                        "is what makes an intermediate metric affordable at "
                        "high cadence: a headline r every couple of thousand "
                        "steps, so a run that will not clear the wind "
                        "baseline is visible early. Emits the same "
                        "linear_r_deseas key as the full probe. Requires "
                        "--anomaly; 0 disables.")
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
    # `--resume !tag` is the same thing as `--resume tag --require-resume`.
    # It exists because workflow_dispatch allows at most 25 inputs and this
    # workflow is at the ceiling: adding a 26th made the FILE invalid, which
    # takes the whole workflow down — every dispatch 422s, not just the new
    # one. Encoding the flag in a string we already pass costs no input slot.
    if a.resume.startswith("!"):
        a.require_resume, a.resume = True, a.resume[1:]
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
            # clim[moy, :, :, c] is [T,H,W]; clim[moy][..., c] would
            # materialise the whole [T,H,W,C] fancy index per channel —
            # ~11 GB per channel on the family-3 tensor (trainprobe.py
            # documents the same trap).
            X[..., c] = X[..., c] - clim[moy, :, :, c]
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

    if (a.eval_every or a.light_probe_every) and not a.anomaly:
        raise SystemExit("--eval-every/--light-probe-every require --anomaly "
                         "(trainprobe measures anomaly-space embeddings; state "
                         "space is disqualified)")
    if a.eval_every or a.light_probe_every:
        import trainprobe                      # lazy: plain runs don't need it
    metrics_path = os.path.join(a.out, "metrics.jsonl")
    loss_every = max(1, a.steps // 200)        # the loss curve, cheap to keep

    # A CONFIG RECORD, first line of the metrics file. The GitHub API does
    # not return a workflow_dispatch run's inputs, so a reader of the live
    # curves had no way to know how long the run is MEANT to be — "step
    # 30,000" alone cannot be told from "step 30,000 of 60,000". Writing the
    # plan next to the measurements is the cheapest fix and it travels with
    # the data (live branch, archive, and the harvested artifact alike).
    with open(metrics_path, "a") as f:
        f.write(json.dumps({"config": {
            "steps": a.steps, "batch": a.batch, "d_z": a.d_z, "patch": a.patch,
            "d_model": a.d_model, "n_layers": a.n_layers, "n_heads": a.n_heads,
            "d_dec": a.d_dec, "anomaly": bool(a.anomaly),
            "eval_every": a.eval_every, "light_probe_every": a.light_probe_every,
            "lr_floor": a.lr_floor, "lr_decay_steps": a.lr_decay_steps,
            "params_M": round(sum(p_.numel() for p_ in model.parameters()) / 1e6, 3),
            "data": os.path.basename(a.data), "C": int(C), "T": int(T),
            "resume": a.resume or None,
        }}) + "\n")

    # Where a checkpoint can outlive its job. /opt/earth-cache is the box's
    # persistent cache directory: it sits OUTSIDE the Actions workspace, so
    # actions/checkout's clean does not touch it, and a later job on the same
    # box can resume from it with no upload and no download.
    CKPT_DIR = "/opt/earth-cache/ckpt"
    ckpt_tag = os.environ.get("CKPT_TAG", "")

    def save_ckpt(step=None):
        """Write the checkpoint, and everything needed to CONTINUE from it.

        Until 2026-08-08 this saved weights only, which made every cancelled
        run a total loss: the weights alone cannot resume a schedule, and a
        job that is stopped at step 15,000 of 60,000 threw away an hour of
        real training. It now carries the optimizer and scheduler state and
        the step reached, so --resume picks the run up exactly where it
        stopped. Optimizer state roughly triples the file; that is the price
        of not repeating work, and these files are transient.
        """
        blob = {"model": model.state_dict(), "chan": chan, "d_z": a.d_z,
                "norm": d["norm"], "args": vars(a),
                "step": int(step if step is not None else 0),
                # WHOSE checkpoint this is. Without it a resumed run knows the
                # file it loaded but not the RUN that wrote it, so the status
                # page cannot find the parent's curves to stitch on — and an
                # orphan-latest.pt rescued off a box carries no run number in
                # its name at all. One string closes that gap.
                "tag": ckpt_tag,
                "opt": opt.state_dict(), "sched": sched.state_dict()}
        torch.save(blob, os.path.join(a.out, "pixelmae.pt"))
        # Mirror to the box-persistent directory when there is one, so a
        # cancel does not destroy the progress.
        if ckpt_tag and os.path.isdir(os.path.dirname(CKPT_DIR) or "/"):
            try:
                os.makedirs(CKPT_DIR, exist_ok=True)
                torch.save(blob, os.path.join(CKPT_DIR, ckpt_tag + ".pt"))
            except OSError as e:                      # full disk, read-only …
                print(f"  (checkpoint mirror skipped: {e})", flush=True)

    def probe_on_device(**kw):
        """Run the in-training probe WITHOUT surrendering the GPU.

        train.py used to hand `model.cpu()` to every probe. embed_everything
        runs on whatever device the model is on and its own docstring says
        the difference is "hours of CPU and minutes of GPU" — so that one
        `.cpu()` was donating the accelerator back. Measured on the 41M
        anchored runs (#48/#49, 0.25-degree tensor): a single full probe cost
        3,697 s and 4,802 s, i.e. 70-74% of all wall-clock time, projecting
        to ~12 h of probing against ~3.5 h of training.

        Falls back to CPU on a CUDA OOM rather than killing the run: the
        probe is instrumentation, and instrumentation must never be the
        thing that loses a training job. empty_cache() first because the
        optimiser state and activations are still resident.
        """
        try:
            if str(dev).startswith("cuda"):
                torch.cuda.empty_cache()
            return trainprobe.probe_now(model, Xt, OBS, d, mvec, t_hold,
                                        x_hold, dynamic, **kw)
        except torch.cuda.OutOfMemoryError:            # cuda-only path
            print("  probe OOM on GPU — falling back to CPU for this one",
                  flush=True)
            torch.cuda.empty_cache()
            m_ = trainprobe.probe_now(model.cpu(), Xt, OBS, d, mvec, t_hold,
                                      x_hold, dynamic, **kw)
            model.to(dev)
            return m_

    def run_probe(step, light):
        """Write one probe record to metrics.jsonl and return it (or None).

        A probe is INSTRUMENTATION, and instrumentation must never be the
        thing that loses a training job. #56-#59 all died at their FIRST full
        probe (step 10k) on a codec.query device mismatch — hours of training
        and every checkpoint, gone, because a probe raised. The device bug is
        fixed in trainprobe.py; THIS is the guard that keeps the next probe
        bug from being fatal. Any exception is caught, logged to the metrics
        file as a {"probe_error"} record (so it shows on the status page
        rather than vanishing), and training carries on.
        """
        try:
            m = probe_on_device(light=light)
        except Exception as e:                       # never fatal
            import traceback
            traceback.print_exc()
            print(f"  probe @{step} FAILED ({type(e).__name__}: {e}) — "
                  f"training continues, no probe point this interval",
                  flush=True)
            with open(metrics_path, "a") as f:
                f.write(json.dumps({
                    "step": step, "wall_s": round(time.time() - t0, 1),
                    "probe_error": f"{type(e).__name__}: {e}"[:200]}) + "\n")
            return None
        m["step"] = step
        m["wall_s"] = round(time.time() - t0, 1)
        with open(metrics_path, "a") as f:
            f.write(json.dumps(m) + "\n")
        return m

    def run_light_probe(step):
        """The cheap probe, written to metrics.jsonl."""
        return run_probe(step, light=True)

    print("training …")
    t0 = time.time()
    # STEP-0 PROBE — the untrained control (Chris, 2026-08-08: "would it be
    # useful to compute the green (r) metric before any training has
    # happened … it may be useful to identify whether training is going in
    # the right direction or whether it does nothing"). A randomly
    # initialised encoder still emits embeddings, and a ridge on random
    # features is a real baseline. Without this point, a flat probe curve is
    # ambiguous: it could mean training converged early, or that training
    # never moved the metric at all. With it, the answer is one subtraction.
    # It costs one light probe (~30 s) and it is the cheapest control in the
    # programme.
    # ...but NOT on a resumed run. Verified on #52: the step-0 probe fired
    # before the resume restored the weights, so it measured a throwaway
    # random init and wrote it to metrics.jsonl as "step 0" — where the
    # status page draws it as the untrained-codec reference line. On a
    # continuation that line would be a lie about a different model.
    # Run the SAME probe at step 0 that runs at every eval interval, so every
    # metric the run reports has an untrained baseline to be read against —
    # not just the light linear one (Chris, 2026-08-08: "schedule the same
    # metric at step 0?"). If eval_every is set the step-0 probe is the FULL
    # one (k-fold RAPID, temporal r, chan%/z% on a random-init codec); if only
    # the light cadence is on, it stays light. The status page reads
    # linear_r_deseas for its untrained-codec line, which the full probe emits
    # too, so that reference is unaffected.
    if (a.light_probe_every or a.eval_every) and not a.resume:
        m0 = run_probe(0, light=not a.eval_every)
        if m0:
            extra = ("" if m0.get("light") else
                     f" · temporal r_des {m0.get('temporal_r_deseas', float('nan')):+.3f}"
                     f" · chan t+1 {m0.get('chan_vs_persistence_pct', float('nan')):+.1f}%")
            print(f"  step-0 probe (UNTRAINED codec): linear r_des "
                  f"{m0['linear_r_deseas']:+.3f}{extra} — every later probe "
                  f"should be read as a change from this", flush=True)
    elif a.resume:
        print("  (no step-0 probe: this run resumes, so there is no untrained "
              "baseline to measure — the original run's step-0 point is the "
              "one that applies)", flush=True)
    CAL = 200                                  # steps before the rate is trusted
    s = 0
    if a.resume:
        # --resume takes a COMMA-SEPARATED CANDIDATE LIST and uses the first
        # one present on this box. Checkpoint mirrors are box-local and the
        # scheduler picks the runner, so naming a single tag means a job that
        # wants "a finished 40M codec" succeeds only by luck — even when two
        # of three boxes hold one. Listing them turns 1-in-3 into 2-in-3, and
        # the log still says exactly which checkpoint was used, so provenance
        # is unchanged.
        cands = [c.strip() for c in a.resume.split(",") if c.strip()]
        rpath = None
        for c in cands:
            pth = c if os.path.sep in c else os.path.join(CKPT_DIR, c + ".pt")
            if os.path.exists(pth):
                rpath = pth
                break
        if rpath is None:
            rpath = (cands[0] if os.path.sep in cands[0]
                     else os.path.join(CKPT_DIR, cands[0] + ".pt"))
            if len(cands) > 1:
                print(f"  --resume: none of {cands} is on this box", flush=True)
        if not os.path.exists(rpath):
            if a.require_resume:
                raise SystemExit(
                    f"--require-resume: no checkpoint at {rpath}. This box is "
                    f"not the one that wrote it (checkpoint mirrors are "
                    f"box-local). Exiting in seconds rather than retraining "
                    f"from scratch for hours under a doc string that claims "
                    f"to be a continuation.")
            print(f"  --resume {a.resume}: NOT FOUND at {rpath} — starting "
                  f"from scratch (this is not an error, but the run is now a "
                  f"fresh one; say so in its doc string)", flush=True)
        else:
            ck = torch.load(rpath, map_location=dev, weights_only=False)
            # SAY WHAT WE LOADED. /opt/earth-cache/ckpt/orphan-latest.pt is
            # whatever the last job on THIS box left behind, which is not
            # necessarily this experiment. load_state_dict fails loudly on an
            # architecture mismatch, but a same-architecture checkpoint from a
            # different run would load silently — so print its identity and
            # let it land in the log and the provenance record.
            ca = ck.get("args", {})
            print(f"  resume source: {rpath}\n"
                  f"    C={len(ck.get('chan', []))} d_z={ck.get('d_z')} "
                  f"d_model={ca.get('d_model')} layers={ca.get('n_layers')} "
                  f"patch={ca.get('patch')} data={os.path.basename(str(ca.get('data','?')))}\n"
                  f"    it was trained toward {ca.get('steps')} steps",
                  flush=True)
            if ca.get("data") and os.path.basename(str(ca["data"])) != os.path.basename(a.data):
                raise SystemExit(
                    f"REFUSING to resume: checkpoint was trained on "
                    f"{os.path.basename(str(ca['data']))} but this run uses "
                    f"{os.path.basename(a.data)}. Cross-tensor resume would "
                    f"produce a codec whose provenance is a lie.")
            model.load_state_dict(ck["model"])
            if "opt" in ck and "step" in ck:
                opt.load_state_dict(ck["opt"])
                if "sched" in ck:
                    sched.load_state_dict(ck["sched"])
                s = int(ck["step"])
                print(f"  RESUMED from {rpath} at step {s} "
                      f"(optimizer + schedule restored); training on to "
                      f"{a.steps}", flush=True)
                # A CONTINUATION RECORD, so the curves can be made whole
                # again. A resumed run's metrics file starts at the step it
                # was handed, which draws a chart that begins in mid-air and
                # silently loses everything the parent measured — including
                # the step-0 untrained-codec line, which is the reference the
                # whole probe chart is read against. Naming the parent and
                # the join step lets the status page fetch the parent's
                # archived metrics and prepend them, and MARK the seam rather
                # than hide it: two jobs, one training trajectory.
                with open(metrics_path, "a") as f:
                    f.write(json.dumps({"resumed": {
                        "from": os.path.basename(rpath),
                        "parent_tag": ck.get("tag") or "",
                        "at_step": s,
                    }}) + "\n")
            else:
                print(f"  WARM-STARTED from {rpath}: weights only, no "
                      f"optimizer or step — the LR schedule restarts from 0. "
                      f"Report this run as a warm start, not a continuation.",
                      flush=True)
            if s >= a.steps:
                print(f"  checkpoint is already at/past --steps; nothing to do")
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
        # Cheap probe first, and skipped on steps where the full probe runs
        # (the full one supersedes it and writes the same key).
        full_here = a.eval_every and (s % a.eval_every == 0 or s == a.steps)
        if a.light_probe_every and not full_here and s % a.light_probe_every == 0:
            m = run_probe(s, light=True)
            if m:
                print(f"  light probe @{s}: linear r_des "
                      f"{m['linear_r_deseas']:+.3f} "
                      f"({m['probe_seconds']:.0f}s)", flush=True)
        if full_here:
            m = run_probe(s, light=False)
            if m:
                print(f"  probe @{s}: chan t+1 {m['chan_vs_persistence_pct']:+.1f}% "
                      f"vs persistence · linear r_des {m['linear_r_deseas']:+.3f} · "
                      f"temporal r_des {m['temporal_r_deseas']:+.3f} "
                      f"({m['probe_seconds']:.0f}s)", flush=True)
            # Crash insurance is INDEPENDENT of the probe: save even when the
            # probe failed, so a probe bug never also costs the checkpoint.
            save_ckpt(s)

    # ---- evaluation on the BLOCKED holdout --------------------------------
    model.eval()
    results = {}
    with torch.no_grad():
        n_eval = min(20000, len(vt))
        t, y, x, ctx = batch(vt, vy, vx, n_eval)
        v, o = gather(t, y, x)
        mask = (torch.rand(n_eval, C, device=dev) < a.mask_ratio) & o
        # CHUNK the final evaluation. Encoding all 20,000 held-out pixels in
        # one forward makes the encoder's feed-forward intermediate
        # n_eval x C x 4*d_model x 4 B — at C=39, d_model=576 that is 7.04 GiB,
        # exactly the allocation that OOM-killed #62 and #63 AFTER they had
        # trained all 60,000 steps and passed every probe. The evaluation is
        # the last thing a run does, so the cost of that crash was the whole
        # run's verdict. Chunking changes no reported number: z, pred and p1
        # are concatenated in the same order and every metric below is
        # computed from the complete arrays.
        EV_CH = 2048
        zs = []
        for i in range(0, n_eval, EV_CH):
            sl = slice(i, min(i + EV_CH, n_eval))
            if a.patch > 1:
                vp, op = gather_px(Xt, OBS, t[sl], y[sl], x[sl], a.patch)
                zs.append(model.encode(vp.to(dev), op.to(dev), mask[sl],
                                       ctx[sl].to(dev)))
            else:
                zs.append(model.encode(v[sl] * (~mask[sl]), o[sl], mask[sl],
                                       ctx[sl].to(dev)))
        z = torch.cat(zs)
        qc = torch.arange(C, device=dev)[None, :].expand(n_eval, -1)
        preds = []
        for i in range(0, n_eval, EV_CH):
            sl = slice(i, min(i + EV_CH, n_eval))
            nb = sl.stop - sl.start
            preds.append(model.query(
                z[sl], qc[sl],
                torch.zeros(nb, C, 3, dtype=torch.long, device=dev)))
        pred = torch.cat(preds)
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
        p1s = []
        for i in range(0, n_eval, EV_CH):                 # chunked, as above
            sl = slice(i, min(i + EV_CH, n_eval))
            nb = sl.stop - sl.start
            offb = torch.zeros(nb, C, 3, dtype=torch.long, device=dev)
            offb[:, :, 2] = 1
            p1s.append(model.query(z[sl], qc[sl], offb))
        p1 = torch.cat(p1s)
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
    save_ckpt(a.steps)
    json.dump(results, open(os.path.join(a.out, "eval.json"), "w"), indent=2)
    print(f"saved {a.out}/pixelmae.pt")

    try:                                       # every run gets its curve
        import plot_run
        plot_run.render(os.path.basename(a.out.rstrip("/")))
    except Exception as e:                     # a missing matplotlib never
        print(f"(curve not rendered: {e})")    # kills a finished run
    except SystemExit as e:                    # ...and neither does a
        print(f"(curve not rendered: SystemExit {e})")   # SystemExit, which
        # is a BaseException and slipped straight through the clause above.



if __name__ == "__main__":
    main()

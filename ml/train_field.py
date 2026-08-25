#!/usr/bin/env python3
"""E-051 · trainer for the joint field head, deterministic and generative.

`ml/plans/E051_field_diffusion.md` is the spec; `ml/field_model.py` is the
model. This file trains it in either mode, on synthetic laws for CPU science or
on the real `[T, P, d_z]` embed-cache artefact stage 2 already publishes.

    # axis A's microcosm — a purely SPATIAL law a per-pixel head cannot see
    python3 ml/train_field.py --toy shift --mode det --steps 600

    # axis B's microcosm — two field-coherent futures, conditional mean = x_t
    python3 ml/train_field.py --toy bimodal --mode diff --steps 4000

    # the closed-form check: a known Gaussian conditional
    python3 ml/train_field.py --toy gauss --mode diff --steps 3000

    # the real substrate (thin path; its first GPU-scale run is a future arm)
    python3 ml/train_field.py --z-cache Z.npy --data tensor.npz \\
            --holdout-years 2019,2020,2023 --mode det

    # the end-to-end code-path assertion
    python3 ml/train_field.py --toy gauss --mode diff --smoke

Result-file discipline is ml/CLAUDE.md §5.25: the JSON is rewritten ATOMICALLY
(temp sibling + `os.replace`) at every eval carrying a top-level `in_progress`,
and the key disappears exactly once, at a completed end. §5.22 is the other
half: a non-finite loss or eval STOPS the run rather than writing NaN.
"""
import argparse
import json
import math
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from field_model import FieldHead, OceanTokenizer, count_params, nfe_to_steps

# ml/probscore.py is E-051.0 and lands in its own diff. Guard the import so
# this module still LOADS without it (a trainer that cannot be imported cannot
# be tested), and skip CRPS with a WARNING rather than inventing a number — a
# missing scoreboard must read as missing, never as zero.
try:
    from probscore import crps_ensemble       # noqa: F401
    HAVE_CRPS = True
except Exception:                             # pragma: no cover - E-051.0 gap
    crps_ensemble = None
    HAVE_CRPS = False


# ---------------------------------------------------------------------------
# result file (ml/CLAUDE.md §5.25)
# ---------------------------------------------------------------------------
def write_result(path, config, history, final=None, in_progress=None):
    """Write the run's result file ATOMICALLY, optionally marked partial.

    `in_progress` is a top-level key and is written FIRST so a human opening
    the file sees it before any number. A reader that finds it must treat every
    number under it as provisional; a file without it is a completed run, and
    that absence is the only certificate the run reached its end.
    """
    d = os.path.dirname(os.path.abspath(path)) or "."
    os.makedirs(d, exist_ok=True)
    payload = {}
    if in_progress is not None:
        payload["in_progress"] = in_progress
    payload["config"] = config
    payload["history"] = history
    payload["final"] = final
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(payload, f, indent=1)
    os.replace(tmp, path)
    return path


def _finite_or_die(name, *vals):
    """§5.22: never write NaN into a results file — stop instead."""
    for v in vals:
        x = float(v)
        if not math.isfinite(x):
            sys.exit(f"REFUSING to continue: {name} went non-finite ({x}). "
                     f"A results file full of NaN is loud enough to notice and "
                     f"quiet enough to misattribute (ml/CLAUDE.md §5.22).")


# ---------------------------------------------------------------------------
# toy laws — each one is a microcosm of an axis in the plan
# ---------------------------------------------------------------------------
def _smooth(x, passes=2):
    """Cheap separable 3x3 box blur with edge replication, `passes` times.

    A "Gaussian-filtered-ish" smooth field without pulling in scipy: repeated
    box blurs converge on a Gaussian, and two passes over a 24x24 grid give a
    correlation length of ~2-3 cells — enough that a one-cell roll is a real
    but learnable spatial derivative rather than white noise.
    """
    for _ in range(passes):
        x = (x + torch.roll(x, 1, dims=0) + torch.roll(x, -1, dims=0)) / 3.0
        x = (x + torch.roll(x, 1, dims=1) + torch.roll(x, -1, dims=1)) / 3.0
    return x


def _smooth_field(H, W, d_z, gen, passes=2):
    x = _smooth(torch.randn(H, W, d_z, generator=gen), passes)
    return x / x.pow(2).mean().sqrt()


def toy_shift(gen, smoke=False):
    """AXIS A's microcosm: a purely SPATIAL law.

        x_{t+1} = (x_t rolled one cell EASTWARD) + 0.02 * smooth noise

    Every pixel is ocean, d_z = 4, grid 24x24, T = 400. The next value of a
    pixel lives in its WESTERN neighbour and nowhere in its own history, so a
    per-pixel head with no neighbours cannot beat persistence here by
    construction — while a head that attends over space can drive the ratio far
    below 1. That is the whole of axis A, with the answer known in advance.

    The per-step noise is smoothed like the field so the law is stationary in
    smoothness: white increments would make the field progressively rougher and
    turn "learn a one-cell roll" into "memorise white noise", which measures
    nothing about spatial attention.
    """
    H = W = 12 if smoke else 24
    T = 60 if smoke else 400
    d_z = 4
    x = _smooth_field(H, W, d_z, gen)
    frames = [x]
    for _ in range(T - 1):
        e = _smooth(torch.randn(H, W, d_z, generator=gen), 2)
        e = 0.02 * e / e.pow(2).mean().sqrt()
        frames.append(torch.roll(frames[-1], shifts=1, dims=1) + e)
    X = torch.stack(frames)                                   # [T, H, W, d_z]
    ys, xs = np.meshgrid(np.arange(H), np.arange(W), indexing="ij")
    ys, xs = ys.reshape(-1), xs.reshape(-1)
    Z = X.reshape(T, H * W, d_z)[:, ys * W + xs].contiguous()
    return dict(Z=Z, H=H, W=W, ys=ys, xs=xs, d_z=d_z, season=None,
                pattern=None, law="shift")


def toy_bimodal(gen, smoke=False):
    """AXIS B's microcosm: two field-COHERENT futures, conditional mean = x_t.

        x_{t+1} = x_t + s * PATTERN + 0.01 * noise,   s = +-1, a fair coin
                                                     shared by the WHOLE field

    Grid 16x16, d_z = 4, T = 600. The conditional MEAN of the residual is
    exactly zero, so the deterministic head's best possible ratio is ~1.0 and
    squared error alone declares the problem unlearnable. A working generative
    head instead samples the two modes, wins CRPS, and its ensemble mean
    recovers the deterministic optimum.

    PATTERN is deliberately SINGLE-SIGNED (a smoothed absolute value, RMS 1):
    that makes `|mean over ocean pixels of sign(r)|` a clean joint-law
    detector — ~1 for a coherent member, ~0 for a factorized sampler drawing an
    independent sign per pixel. On a zero-mean pattern both would read ~0 and
    the statistic would detect nothing.
    """
    H = W = 8 if smoke else 16
    T = 60 if smoke else 600
    d_z = 4
    pat = _smooth(_smooth_field(H, W, d_z, gen).abs(), 2)
    pat = pat / pat.pow(2).mean().sqrt()
    frames = [_smooth_field(H, W, d_z, gen)]
    signs = []
    for _ in range(T - 1):
        s = 1.0 if torch.rand(1, generator=gen).item() < 0.5 else -1.0
        signs.append(s)
        frames.append(frames[-1] + s * pat
                      + 0.01 * torch.randn(H, W, d_z, generator=gen))
    X = torch.stack(frames)
    ys, xs = np.meshgrid(np.arange(H), np.arange(W), indexing="ij")
    ys, xs = ys.reshape(-1), xs.reshape(-1)
    lin = ys * W + xs
    Z = X.reshape(T, H * W, d_z)[:, lin].contiguous()
    return dict(Z=Z, H=H, W=W, ys=ys, xs=xs, d_z=d_z, season=None,
                pattern=pat.reshape(H * W, d_z)[lin].contiguous(),
                law="bimodal", signs=signs)


def toy_gauss(gen, smoke=False):
    """A KNOWN Gaussian conditional, for closed-form checks.

        x_{t+1} = a * x_t + sigma_e * eps,   a = 0.7, sigma_e = 0.5, iid/pixel

    Grid 8x8, d_z = 1, T = 800. The conditional law of x_{t+1} given x_t is
    exactly N(a*x_t, sigma_e^2), so a diffusion head can be checked against an
    analytic mean and spread rather than against a threshold. The residual
    r = (a-1)*x_t + sigma_e*eps, so persistence is NOT optimal here either —
    the conditional mean of r is -0.3*x_t, which the det head must find.
    """
    H = W = 8
    T = 60 if smoke else 800
    d_z, a, se = 1, 0.7, 0.5
    x = torch.randn(H, W, d_z, generator=gen) * (se / math.sqrt(1 - a * a))
    frames = [x]
    for _ in range(T - 1):
        frames.append(a * frames[-1]
                      + se * torch.randn(H, W, d_z, generator=gen))
    X = torch.stack(frames)
    ys, xs = np.meshgrid(np.arange(H), np.arange(W), indexing="ij")
    ys, xs = ys.reshape(-1), xs.reshape(-1)
    Z = X.reshape(T, H * W, d_z)[:, ys * W + xs].contiguous()
    return dict(Z=Z, H=H, W=W, ys=ys, xs=xs, d_z=d_z, season=None,
                pattern=None, law="gauss", a=a, sigma_e=se)


TOYS = {"shift": toy_shift, "bimodal": toy_bimodal, "gauss": toy_gauss}


# ---------------------------------------------------------------------------
# real substrate — deliberately THIN
# ---------------------------------------------------------------------------
def _season_of_month(m):
    """(sin, cos) of the year phase at a month's CENTRE, from a 'YYYY-MM'."""
    mm = int(str(m).split("-")[1])
    frac = (mm - 0.5) / 12.0
    return math.sin(2 * math.pi * frac), math.cos(2 * math.pi * frac)


def load_real(z_cache, npz_path, pixels_npy=None):
    """Load the `[T, P, d_z]` embed cache plus the tensor's geometry.

    THIN ON PURPOSE. The field head's first GPU-scale run is a future arm
    (the plan: `ml-train.yml` sits at the 25-input ceiling and a new trainer
    needs its own reviewed dispatch step), so this path exists to be correct
    and exercised — `--smoke` runs it against a small synthetic npz in
    `tests/test_field_diffusion.py` — not to be fast. It memory-maps Z rather
    than reading it: at global scale that array is gigabytes and the trainer
    touches K+2 rows of it per step.

    `ys`/`xs` must be supplied. They come from the npz when it carries them
    (the tensor builders write `ys`/`xs` beside `lats`/`lons`); otherwise a
    `--pixels` npy of shape [P, 2] or a 2xP array is REQUIRED — deriving them
    from P alone is not possible and guessing a raster order is exactly the
    class of mistake that puts the Gulf Stream in the Norwegian Sea.
    """
    Z = np.load(z_cache, mmap_mode="r")
    if Z.ndim != 3:
        sys.exit(f"--z-cache {z_cache}: expected [T, P, d_z], got {Z.shape}")
    d = np.load(npz_path, allow_pickle=True)
    if "lats" not in d or "lons" not in d:
        sys.exit(f"--data {npz_path}: no lats/lons — cannot derive H, W")
    H, W = int(len(d["lats"])), int(len(d["lons"]))
    if "ys" in d and "xs" in d:
        ys = np.asarray(d["ys"]).astype(np.int64).reshape(-1)
        xs = np.asarray(d["xs"]).astype(np.int64).reshape(-1)
    elif pixels_npy:
        px = np.load(pixels_npy)
        px = px if px.shape[0] == 2 else px.T
        ys, xs = px[0].astype(np.int64), px[1].astype(np.int64)
    else:
        sys.exit(f"--data {npz_path} carries no ys/xs and no --pixels npy was "
                 f"given. The pixel ordering of a [T,P,d_z] cache cannot be "
                 f"recovered from P; supply it rather than guessing.")
    if len(ys) != Z.shape[1]:
        sys.exit(f"pixel list has {len(ys)} entries, Z has P={Z.shape[1]}")
    months = [str(m) for m in np.asarray(d["months"]).reshape(-1)] \
        if "months" in d else None
    if months is not None and len(months) != Z.shape[0]:
        sys.exit(f"months has {len(months)} entries, Z has T={Z.shape[0]}")
    season = None
    if months is not None:
        season = torch.tensor([_season_of_month(m) for m in months],
                              dtype=torch.float32)
    return dict(Z=Z, H=H, W=W, ys=ys, xs=xs, d_z=int(Z.shape[2]),
                season=season, months=months, pattern=None, law="real")


# ---------------------------------------------------------------------------
# windows and splits
# ---------------------------------------------------------------------------
def make_splits(T, K, val_frac, months=None, holdout_years=None):
    """Valid window anchors t (context t-K+1..t, target t+1), split train/val.

    Toys use a TAIL split, so a val window never shares a frame with a train
    window except across the seam. Real data uses whole HELD-OUT YEARS, and a
    window is train only when NONE of its K+1 frames falls in a holdout year —
    a window that straddles the boundary has seen the answer.
    """
    anchors = np.arange(K - 1, T - 1)
    if holdout_years and months is None:
        sys.exit("--holdout-years was given but the npz carries no `months`: "
                 "there is nothing to hold out BY. Supply months or drop the "
                 "flag rather than silently falling back to a tail split.")
    if holdout_years:
        yr = np.array([int(str(m).split("-")[0]) for m in months])
        hold = np.isin(yr, np.array(sorted(holdout_years)))
        val, train = [], []
        for t in anchors:
            span = slice(t - K + 1, t + 2)
            if hold[span].all():
                val.append(t)
            elif not hold[span].any():
                train.append(t)
        return np.array(train, np.int64), np.array(val, np.int64)
    n_val = max(1, int(round(val_frac * len(anchors))))
    return anchors[:-n_val].copy(), anchors[-n_val:].copy()


class Windows:
    """Gathers (context, x_t, x_{t+1}) windows from Z, torch or memmap alike."""

    def __init__(self, Z, K, season=None):
        self.Z, self.K, self.season = Z, int(K), season
        self.torch = isinstance(Z, torch.Tensor)

    def _rows(self, rows):
        if self.torch:
            return self.Z[rows]
        return torch.from_numpy(np.ascontiguousarray(self.Z[rows]))

    def batch(self, ts):
        ts = np.asarray(ts, np.int64)
        K = self.K
        ctx = torch.stack([self._rows(np.arange(t - K + 1, t + 1)) for t in ts])
        z_t = ctx[:, -1]
        z_n = torch.stack([self._rows(np.array([t + 1]))[0] for t in ts])
        sea = None
        if self.season is not None:
            sea = torch.stack([self.season[t - K + 1:t + 1] for t in ts])
        return ctx.float(), z_t.float(), z_n.float(), sea


def measure_sigma_data(win, train_ts, cap=256):
    """RMS of the one-step residual over TRAIN windows — EDM's sigma_data.

    A property of the DATA, never of the model (ml/CLAUDE.md §4.2): it must not
    move as training proceeds, so it is measured once, before the first step,
    on the training split only.
    """
    ts = train_ts if len(train_ts) <= cap else train_ts[
        np.linspace(0, len(train_ts) - 1, cap).astype(np.int64)]
    _, z_t, z_n, _ = win.batch(ts)
    return float((z_n - z_t).pow(2).mean().sqrt())


# ---------------------------------------------------------------------------
# evaluation
# ---------------------------------------------------------------------------
def eval_det(model, win, ts, batch=16):
    """{'ratio': mse / mse_persistence} on the val windows."""
    se = pe = n = 0.0
    with torch.no_grad():
        for i in range(0, len(ts), batch):
            ctx, z_t, z_n, sea = win.batch(ts[i:i + batch])
            cond = model.make_cond(model.tok.to_tokens(ctx), sea)
            z_hat = model.forward_det(cond, z_t)
            se += float((z_hat - z_n).pow(2).sum())
            pe += float((z_t - z_n).pow(2).sum())
            n += z_n.numel()
    return {"mse": se / n, "mse_pers": pe / n,
            "ratio": se / pe if pe > 0 else float("inf")}


def eval_diff(model, win, ts, nfe, members, seed, batch=8):
    """Probabilistic read-out for the generative head.

    Everything is drawn under a FIXED derived seed so two evals of the same
    checkpoint return the same numbers and two checkpoints are compared on the
    same draw — the protocol-determinism property ml/CLAUDE.md §3b calls a
    first-class integrity check.

    Keys:
      * `sample_ratio`   — per-MEMBER MSE / persistence. Expected to be WORSE
        than the deterministic head by exactly the conditional variance; that
        is the slide-4 identity E-051.0 exists to keep out of the verdict.
      * `ens_ratio`      — ensemble-mean MSE / persistence. This is the one
        that must ~match E-051.1: no MSE tax after averaging.
      * `crps`           — fair-CRPS, when ml/probscore.py is present.
      * `spread_error`   — sqrt(mean[(M+1)/M · Var_ddof1]) / rmse(ens mean),
        `ml/probscore.py`'s convention EXACTLY (its `spread_error`): 1 is
        calibration, < 1 is over-confidence, > 1 is an inflated ensemble. It is
        accumulated from sufficient statistics rather than by calling that
        function on the whole ensemble, because at real scale [M, B, P, d_z]
        does not fit in memory — the formula is identical, the batching is not.
      * `sign_coherence` — mean over members and times of |mean over ocean
        pixels of sign(r_m)|. THE JOINT-LAW DETECTOR: on the bimodal toy a
        joint sampler picks one field-wide mode per member and reads ~1, while
        a factorized sampler draws an independent sign per pixel and reads
        ~0 (it averages P coin flips). It is not a skill score — on a law with
        no coherent mode it reads ~0 for a perfect sampler too, which is why
        `mode_corr` is reported beside it.
      * `mode_corr`      — mean over members and times of the |cosine| between
        r_m and r_true over ocean pixels. The correlation-based twin, and the
        one that survives a pattern with a non-zero field mean: ~1 when a
        member lands on a mode (either mode — the absolute value is why an
        unpredictable coin does not penalise it), ~1/sqrt(P*d_z) for
        independent per-pixel signs.
    """
    n_steps, spent = nfe_to_steps(nfe)
    se_s = se_e = pe = n = 0.0
    var_sum = var_n = 0.0
    signs, corrs, crpss = [], [], []
    with torch.no_grad():
        for i in range(0, len(ts), batch):
            ctx, z_t, z_n, sea = win.batch(ts[i:i + batch])
            cond = model.make_cond(model.tok.to_tokens(ctx), sea)
            ens = model.sample(cond, z_t, n_steps, seed=seed + 1000 * i,
                               M=members)                    # [M, B, P, d_z]
            mean = ens.mean(0)
            se_s += float((ens - z_n[None]).pow(2).sum()) / members
            se_e += float((mean - z_n).pow(2).sum())
            pe += float((z_t - z_n).pow(2).sum())
            n += z_n.numel()
            if members > 1:
                infl = (members + 1) / members * ens.var(0, unbiased=True)
                var_sum += float(infl.sum())
                var_n += z_n.numel()
            r_m = ens - z_t[None]
            r_true = z_n - z_t
            signs.append(float(r_m.sign().flatten(2).mean(dim=2).abs().mean()))
            # UNCENTRED (cosine), deliberately: a field-coherent mode is a
            # component along a FIXED direction, and removing the field mean
            # would throw away most of a single-signed pattern — which is
            # exactly the pattern the bimodal toy uses, so a centred
            # correlation would read ~0 for a perfectly coherent member.
            num = (r_m * r_true[None]).flatten(2).sum(2)
            den = (r_m.flatten(2).pow(2).sum(2).sqrt()
                   * r_true.flatten(1).pow(2).sum(1).sqrt()[None] + 1e-30)
            corrs.append(float((num / den).abs().mean()))
            if HAVE_CRPS:
                try:
                    c = crps_ensemble(ens.reshape(members, -1).numpy(),
                                      z_n.reshape(-1).numpy())
                    # probscore returns {"crps", "crps_field"}; tolerate a bare
                    # float so a future signature change degrades to a warning
                    # rather than to a wrong number.
                    crpss.append((float(c["crps"] if isinstance(c, dict) else c),
                                  z_n.numel()))
                except Exception as e:                # pragma: no cover
                    print(f"::warning:: crps_ensemble failed ({e}); skipping",
                          flush=True)
    out = {"nfe_spent": spent, "members": members,
           "sample_ratio": se_s / pe if pe > 0 else float("inf"),
           "ens_ratio": se_e / pe if pe > 0 else float("inf"),
           "mse_pers": pe / n,
           "sign_coherence": float(np.mean(signs)),
           "mode_corr": float(np.mean(corrs))}
    if var_n:
        spread = math.sqrt(var_sum / var_n)
        rmse = math.sqrt(se_e / n)
        out["spread"] = spread
        out["spread_error"] = spread / rmse if rmse > 0 else float("inf")
    if crpss:
        w = float(sum(c[1] for c in crpss))
        out["crps"] = float(sum(c[0] * c[1] for c in crpss) / w)
    elif not HAVE_CRPS:
        out["crps"] = None
        out["crps_note"] = "ml/probscore.py absent — CRPS skipped, not faked"
    return out


# ---------------------------------------------------------------------------
# checkpoints
# ---------------------------------------------------------------------------
def save_ckpt(path, model, opt, step, gen, history, cfg):
    """Everything the trajectory depends on, so a resume CONTINUES it.

    The batch sampler and every EDM noise draw come from one `torch.Generator`
    (`gen`), so its state is what makes a resumed run bit-identical rather than
    merely similar; the global torch and numpy states ride along because a
    future addition might reach for them and a checkpoint that silently lost
    one would be discovered as an irreproducibility, not as a bug.
    """
    d = os.path.dirname(os.path.abspath(path)) or "."
    os.makedirs(d, exist_ok=True)
    tmp = path + ".tmp"
    torch.save({"model": model.state_dict(), "opt": opt.state_dict(),
                "step": int(step), "gen": gen.get_state(),
                "torch_rng": torch.get_rng_state(),
                "numpy_rng": np.random.get_state(),
                "history": history, "args": cfg}, tmp)
    os.replace(tmp, path)


def load_ckpt(path, model, opt, gen):
    ck = torch.load(path, map_location="cpu", weights_only=False)
    model.load_state_dict(ck["model"])
    opt.load_state_dict(ck["opt"])
    gen.set_state(ck["gen"])
    torch.set_rng_state(ck["torch_rng"])
    np.random.set_state(ck["numpy_rng"])
    return int(ck["step"]), list(ck.get("history", []))


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def build_argparser():
    p = argparse.ArgumentParser(
        description="E-051 field head trainer (det and EDM-diffusion modes)")
    p.add_argument("--toy", choices=sorted(TOYS))
    p.add_argument("--z-cache", help="[T, P, d_z] embed-cache .npy")
    p.add_argument("--data", help="tensor .npz (lats/lons/months[/ys/xs])")
    p.add_argument("--pixels", help="[P,2] .npy of (ys, xs) when the npz "
                                    "carries none")
    p.add_argument("--holdout-years", default="",
                   help='e.g. "2019,2020,2023" — real mode only')
    p.add_argument("--val-frac", type=float, default=0.15,
                   help="tail split for the toys")
    p.add_argument("--mode", choices=["det", "diff"], default="det")
    p.add_argument("--K", type=int, default=8)
    p.add_argument("--patch", type=int, default=4)
    p.add_argument("--d-model", type=int, default=128)
    p.add_argument("--layers", type=int, default=4)
    p.add_argument("--heads", type=int, default=4)
    p.add_argument("--d-cond", type=int, default=128)
    p.add_argument("--cond-layers", type=int, default=2)
    p.add_argument("--cond-heads", type=int, default=4)
    p.add_argument("--steps", type=int, default=2000)
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight-decay", type=float, default=0.0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--nfe", type=int, default=18)
    p.add_argument("--members", type=int, default=8)
    p.add_argument("--eval-every", type=int, default=250)
    p.add_argument("--eval-windows", type=int, default=16,
                   help="fixed val subset the diffusion read-out samples on")
    p.add_argument("--out", default=None,
                   help="result JSON (default ml/runs/field/<name>.json)")
    p.add_argument("--ckpt", default=None)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--smoke", action="store_true",
                   help="shrink everything to ~30 steps and ASSERT the run "
                        "end to end, including a sample() call and the "
                        "result-file write")
    p.add_argument("--quiet", action="store_true")
    return p


def _apply_smoke(a):
    """Shrink every dimension. The point is the CODE PATH, not a number.

    ml/CLAUDE.md §4.8: exercise the path on a toy before spending the expensive
    resource. Any hour of GPU on a path that has never executed is a coin flip.
    """
    a.steps, a.eval_every, a.batch = 30, 10, 4
    a.K, a.patch = 3, 2
    a.d_model, a.layers, a.heads = 32, 2, 2
    a.d_cond, a.cond_layers, a.cond_heads = 32, 1, 2
    a.nfe, a.members, a.eval_windows = 5, 2, 4
    if not a.toy and not a.z_cache:
        a.toy = "gauss"
    return a


def main(argv=None):
    a = build_argparser().parse_args(argv)
    if a.smoke:
        a = _apply_smoke(a)
    if bool(a.toy) == bool(a.z_cache):
        sys.exit("give exactly one of --toy or --z-cache (with --data)")
    if a.z_cache and not a.data:
        sys.exit("--z-cache needs --data <tensor.npz> for the geometry")
    if a.d_model % 2:
        sys.exit("--d-model must be even (the 2-D position table's halves)")

    torch.manual_seed(a.seed)
    np.random.seed(a.seed)
    gen = torch.Generator()
    gen.manual_seed(a.seed)

    # ---- data ------------------------------------------------------------
    if a.toy:
        tgen = torch.Generator()
        tgen.manual_seed(a.seed)
        ds = TOYS[a.toy](tgen, smoke=a.smoke)
        hold = None
    else:
        ds = load_real(a.z_cache, a.data, a.pixels)
        hold = [int(y) for y in a.holdout_years.split(",") if y.strip()]
    T = ds["Z"].shape[0]
    if T < a.K + 2:
        sys.exit(f"T={T} is too short for K={a.K} (need at least K+2)")
    tok = OceanTokenizer(ds["H"], ds["W"], ds["ys"], ds["xs"], a.patch)
    win = Windows(ds["Z"], a.K, ds["season"])
    tr_ts, va_ts = make_splits(T, a.K, a.val_frac, ds.get("months"), hold)
    if len(tr_ts) == 0 or len(va_ts) == 0:
        sys.exit(f"empty split: {len(tr_ts)} train / {len(va_ts)} val windows")
    sd = measure_sigma_data(win, tr_ts)
    _finite_or_die("sigma_data", sd)

    model = FieldHead(tok, ds["d_z"], a.K, mode=a.mode, d_model=a.d_model,
                      layers=a.layers, heads=a.heads, d_cond=a.d_cond,
                      cond_layers=a.cond_layers, cond_heads=a.cond_heads,
                      sigma_data=sd)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr,
                            weight_decay=a.weight_decay)

    name = f"{a.toy or 'real'}_{a.mode}_s{a.seed}"
    out_path = a.out or os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                     "runs", "field", f"{name}.json")
    step0, history = 0, []
    if a.resume:
        if not a.ckpt or not os.path.exists(a.ckpt):
            sys.exit(f"--resume needs an existing --ckpt (got {a.ckpt!r})")
        step0, history = load_ckpt(a.ckpt, model, opt, gen)

    cfg = dict(vars(a))
    cfg.update(law=ds["law"], T=T, P=int(len(ds["ys"])), d_z=int(ds["d_z"]),
               ntok=tok.ntok, H=ds["H"], W=ds["W"], sigma_data=sd,
               params=count_params(model),
               n_train=int(len(tr_ts)), n_val=int(len(va_ts)),
               have_crps=HAVE_CRPS)
    if not a.quiet:
        print(f"[E-051] {name} · params {cfg['params']:,} · ntok {tok.ntok} "
              f"· P {cfg['P']} · d_z {cfg['d_z']} · sigma_data {sd:.4f} · "
              f"train/val {len(tr_ts)}/{len(va_ts)} · out {out_path}",
              flush=True)

    # The diffusion read-out uses a FIXED val subset and a FIXED derived seed:
    # comparing two checkpoints on two different draws is the E-005 failure
    # mode with a newer metric.
    ev_ts = va_ts[:a.eval_windows]
    eval_seed = 10_000 + 7 * a.seed

    def evaluate():
        model.eval()
        r = (eval_det(model, win, va_ts) if a.mode == "det" else
             eval_diff(model, win, ev_ts, a.nfe, a.members, eval_seed))
        model.train()
        return r

    # ---- train -----------------------------------------------------------
    # Constant LR on purpose: E-051's question is about the ARCHITECTURE
    # (joint vs factorized, sample vs point estimate), and a schedule is one
    # more thing that differs between arms. `CosineAnnealingLR` also has the
    # documented resume trap (ml/CLAUDE.md §7 / docs/ML_BASICS.md §9) where a
    # reloaded schedule asked for a larger total returns lr = 0.
    t0 = time.time()
    model.train()
    for step in range(step0 + 1, a.steps + 1):
        idx = torch.randint(len(tr_ts), (a.batch,), generator=gen)
        ctx, z_t, z_n, sea = win.batch(tr_ts[idx.numpy()])
        ctx_tok = tok.to_tokens(ctx)
        cond = model.make_cond(ctx_tok, sea)
        if a.mode == "det":
            loss = (model.forward_det(cond, z_t) - z_n).pow(2).mean()
        else:
            loss = model.edm_loss(cond, z_t, z_n, gen)
        _finite_or_die(f"training loss at step {step}", loss.item())
        opt.zero_grad(set_to_none=True)
        loss.backward()
        if a.grad_clip:
            torch.nn.utils.clip_grad_norm_(model.parameters(), a.grad_clip)
        opt.step()

        if a.eval_every and (step % a.eval_every == 0 or step == a.steps):
            rec = {"step": step, "loss": float(loss.item()),
                   "wall_s": round(time.time() - t0, 2)}
            rec.update(evaluate())
            for k, v in rec.items():
                if isinstance(v, float):
                    _finite_or_die(f"eval key {k!r} at step {step}", v)
            history.append(rec)
            write_result(out_path, cfg, history, final=None,
                         in_progress={"step": step, "of": a.steps})
            if a.ckpt:
                save_ckpt(a.ckpt, model, opt, step, gen, history, cfg)
            if not a.quiet:
                head = " · ".join(f"{k} {v:.4f}" if isinstance(v, float) else
                                  f"{k} {v}" for k, v in rec.items()
                                  if k in ("step", "loss", "ratio",
                                           "sample_ratio", "ens_ratio",
                                           "sign_coherence", "mode_corr"))
                print(f"  {head}", flush=True)

    final = dict(history[-1]) if history else {}
    final["steps"] = a.steps
    final["wall_s"] = round(time.time() - t0, 2)

    if a.smoke:
        # ASSERT THE EFFECT, not the invocation (ml/CLAUDE.md §0.2). A smoke
        # that only "ran" proves nothing; these four checks are the artefact.
        ctx, z_t, _, sea = win.batch(ev_ts[:2])
        cond = model.make_cond(tok.to_tokens(ctx), sea)
        n_steps, spent = nfe_to_steps(a.nfe)
        ens = model.sample(cond, z_t, n_steps, seed=1234, M=2)
        want = (2, len(ev_ts[:2]), cfg["P"], cfg["d_z"])
        # `sys.exit`, not `assert`: a check that vanishes under -O is not a
        # check, and this one is the whole point of --smoke.
        if tuple(ens.shape) != want:
            sys.exit(f"--smoke: sample() returned {tuple(ens.shape)}, want {want}")
        if not bool(torch.isfinite(ens).all()):
            sys.exit("--smoke: sample() produced a non-finite value")
        if not history:
            sys.exit("--smoke: no eval ran, so nothing would have been written")
        final["smoke_sample_shape"] = list(ens.shape)
        final["smoke_nfe_spent"] = spent

    write_result(out_path, cfg, history, final=final, in_progress=None)
    with open(out_path) as f:
        chk = json.load(f)
    if "in_progress" in chk:
        sys.exit("the completed result file still carries in_progress — that "
                 "key's absence is the run's only completion certificate")
    if a.ckpt:
        save_ckpt(a.ckpt, model, opt, a.steps, gen, history, cfg)
    if not a.quiet:
        print(f"[E-051] done · {out_path}", flush=True)
    return {"out": out_path, "config": cfg, "history": history, "final": final,
            "model": model}


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Tier-3b gates for the JAX STAGE-2 trainer (`ml/plans/JAX_PORT.md` §5, G5).

G4 pinned the codec trainer. A stage-2 trainer adds a different set of things
a codec gate cannot see: a WINDOW (K steps of a stencil-widened gather, whose
layout every published head's weights assume), an objective over that window,
two knobs that change the arithmetic in ways no loss curve would advertise
(`--input-znoise`, `--grad-clip`), a checkpoint that has to be rolled by the
UNCHANGED torch eval scripts, and a BLOCK axis a codec can impose on the whole
run. Each is a pre-registered gate here, CPU only, no network, fp32.

  **G5a — loss parity.** Identical converted head weights, identical window
  batch, identical noise: the torch stage-2 loss and the JAX one agree to
  1e-5. THE NOISE IS AN INPUT, NOT AN RNG. `--input-znoise` draws
  `torch.randn_like` on the torch side and would draw a JAX key here, and two
  samplers cannot be made to agree — so `apply_znoise` takes the perturbation
  as an ARRAY and the gate feeds both frameworks the same one. That is the
  only way this test measures the arithmetic instead of measuring two RNGs.
  Covered at stencil 1 and stencil 9, with noise off and on, and the window
  gather itself is pinned elementwise against `ml/temporal.py:gather_stencil`.

  **G5b — one-step parity.** Same init, same batch, plain SGD at lr 1e-2:
  max |Δweight| < 1e-6 over every parameter. Then the same step under AdamW,
  whose achievable tolerance is looser for an algorithmic reason and is
  therefore MEASURED and stated twice rather than widened once — see the
  comment at the gate.

  **G5c — 300 toy steps against the REAL `ml/temporal.py`.** Both trainers on
  the same toy tensor, the same toy codec and — this is the part that makes it
  sharper than G4c — literally the SAME Z: `ml/temporal.py` writes its
  embedding cache, and the JAX run is handed that file with `--z`. So the
  encoder is not a variable, the window pool is not a variable, and the
  held-out evaluation draw is not a variable either (both take
  `np.random.default_rng(seed).choice` over the identical population). The
  PERSISTENCE baseline should therefore agree to floating-point reduction
  order, and it is asserted at 1e-4 relative — a much harder statement than
  "the loss fell". Only the trained model's own MSE is a band, because the
  RNG streams that pick training windows cannot match across frameworks.

  **G5d — the head checkpoint round trip, and the two knobs pinned.**
  A JAX head exported with `convert.export_temporal_pt`, loaded by the torch
  `TemporalTransformer` with `strict=True` and forward-matched to 1e-5 — the
  direction `JAX_PORT.md` §1b names as the cheap validation of a TPU-trained
  head. Beside it, two targeted checks that no end-to-end band would catch:
  `--grad-clip` above the norm is BIT-IDENTICAL to unclipped and below it
  scales the gradient by exactly `CLIP/norm`; `--input-znoise` leaves dead
  slots at EXACT zeros and moves live ones by exactly `sigma * noise`.

  **G5e — the block-z axis adoption.** On a toy block codec,
  `train_stage2.adopt_block_axis` must return the same labels, the same
  month-of-year, the same FUSED `t_hold` (any held-out bin makes the block
  held out) and the same remapped RAPID rows as `ml/temporal.py`'s own block
  branch, plus the refusal of `--time-stride` on a block codec.

WHERE THE TORCH REFERENCE COMES FROM. `ml/temporal.py`'s stage-2 loss is
inline in `main()` and cannot be imported, so the torch side of G5a/G5b is a
transcription of lines 2425-2438 (the znoise block and `l_base`), and G5e's
reference is a transcription of lines 1565-1600. A transcription tests what
was transcribed — which is exactly why G5c exists beside them and runs the
REAL `ml/temporal.py` end to end as a subprocess.

    python3 tests/test_jaxport_train_s2.py
"""
import json
import os
import shutil
import subprocess
import sys
import tempfile
import warnings

import numpy as np
import torch

warnings.filterwarnings("ignore", message=".*enable_nested_tensor.*")
warnings.filterwarnings("ignore", message=".*not writable.*")

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
ML = os.path.join(ROOT, "ml")
sys.path.insert(0, ML)

import jax                                                       # noqa: E402
import jax.numpy as jnp                                          # noqa: E402
import optax                                                     # noqa: E402
from flax import nnx                                             # noqa: E402

from model import PixelMAE as TorchPixelMAE                      # noqa: E402
from temporal import (TemporalTransformer as TorchTemporal,      # noqa: E402
                      gather_stencil)
from timeblocks import BlockAxis                                 # noqa: E402
from jaxport import models as jm                                 # noqa: E402
from jaxport import convert as jc                                # noqa: E402
from jaxport.train_stage2 import (adopt_block_axis,              # noqa: E402
                                  apply_znoise, gather_stencil_np,
                                  lr_factor, stage2_loss)

TEMPORAL = os.path.join(ML, "temporal.py")
JTRAIN2 = os.path.join(ML, "jaxport", "train_stage2.py")
FAILURES = []


def fail(msg):
    FAILURES.append(msg)


def close(name, a, b, tol):
    a = np.asarray(a.detach().cpu().numpy() if hasattr(a, "detach") else a,
                   np.float64)
    b = np.asarray(b, np.float64)
    if a.shape != b.shape:
        fail(f"{name}: shape {a.shape} vs {b.shape}")
        return float("inf")
    dd = float(np.max(np.abs(a - b)))
    if not (dd < tol):
        fail(f"{name}: max|Δ| {dd:.3e} >= {tol:g}")
    return dd


def _j(t):
    return jnp.asarray(t.detach().cpu().numpy() if hasattr(t, "detach")
                       else np.asarray(t))


# --------------------------------------------------------------------------
# the torch reference: a TRANSCRIPTION of ml/temporal.py's inline stage-2
# loss (see the module docstring for why it cannot be an import)
# --------------------------------------------------------------------------
def torch_apply_znoise(zseq, noise, sigma, d_z):
    """ml/temporal.py:2425-2434, with the RANDN REPLACED BY `noise`."""
    z4 = zseq.view(*zseq.shape[:2], -1, d_z)               # [n,K,S,d_z]
    live = (z4 != 0).any(-1, keepdim=True)                 # [n,K,S,1]
    return (z4 + noise.view(z4.shape) * sigma * live).view(zseq.shape)


def torch_stage2_loss(model, zseq, mseq, sctx, ztgt):
    """ml/temporal.py:2436-2438 — one forward, `l_base`, which at unroll 1
    with no direct heads IS the whole objective."""
    pred, hid = model(zseq, mseq, sctx)
    return (pred - ztgt).pow(2).mean(), pred, hid


# --------------------------------------------------------------------------
# fixtures
# --------------------------------------------------------------------------
def make_head_pair(seed=1, **kw):
    """A torch TemporalTransformer and the NNX one loaded from its
    state_dict."""
    torch.manual_seed(seed)
    tm = TorchTemporal(**kw).eval()
    jmod = jm.TemporalTransformer(**kw, rngs=nnx.Rngs(0))
    jc.load_temporal(tm.state_dict(), jmod)
    return tm, jmod


def make_window_batch(n, K, d_z, stencil, seed=2, dead_frac=0.25):
    """One stage-2 batch in the shapes `batch_windows` produces.

    `dead_frac` of the non-centre slots are set to EXACT ZEROS, because zero
    is the dead-slot encoding and the znoise gate turns on being able to tell
    a dead slot from a live one.
    """
    g = torch.Generator().manual_seed(seed)
    zseq = torch.randn(n, K, stencil * d_z, generator=g)
    if stencil > 1:
        z4 = zseq.view(n, K, stencil, d_z)
        dead = torch.rand(n, 1, stencil, 1, generator=g) < dead_frac
        dead[:, :, :1] = False                       # the centre is never dead
        z4 = z4 * (~dead)
        zseq = z4.reshape(n, K, stencil * d_z).contiguous()
    mseq = torch.randn(n, K, 2, generator=g)
    sctx = torch.randn(n, d_z + 2 + (stencil if stencil > 1 else 0),
                       generator=g)
    ztgt = torch.randn(n, K, d_z, generator=g)
    noise = torch.randn(n, K, stencil, d_z, generator=g)
    return zseq, mseq, sctx, ztgt, noise


def toy_tensor(path, C=6, T=72, H=10, W=12, seed=20260824):
    """A toy tensor with something LEARNABLE in it and a RAPID series.

    Pure noise would leave both trainers moving the loss by rounding and the
    band would then compare two nulls — the "a run that trains on all-zeros
    completes too" failure. Every channel is a different loading of ONE latent
    field that also has a slow autoregressive component, so the next step is
    genuinely predictable from the window, which is the structure the stage-2
    objective exists to find.
    """
    rng = np.random.default_rng(seed)
    lat = np.zeros((T, H, W, 1))
    lat[0] = rng.normal(size=(1, H, W, 1))
    for t in range(1, T):
        lat[t] = 0.85 * lat[t - 1] + 0.5 * rng.normal(size=(1, H, W, 1))
    load = rng.normal(size=(1, 1, 1, C)) * 1.5
    X = (lat * load + 0.3 * rng.normal(size=(T, H, W, C))).astype(np.float32)
    X[:, rng.random((H, W)) < 0.15, :] = np.nan
    months = np.array([f"{2004 + i // 12:04d}-{i % 12 + 1:02d}"
                       for i in range(T)])
    np.savez(path, X=X, months=months, lats=np.linspace(20, 40, H),
             lons=np.linspace(-70, -20, W),
             chan=np.array([f"c{i}" for i in range(C)]),
             norm=np.stack([np.zeros(C), np.ones(C)], 1).astype(np.float32),
             rapid=np.array([[float(i), 15.0 + float(lat[i].mean() * 3)
                              + rng.normal() * 0.5] for i in range(T)]))
    return C, T


def toy_codec(path, C, d_z=8, k_time=1, time_block=""):
    """A toy stage-1 checkpoint in the shape `ml/temporal.py` reads."""
    torch.manual_seed(3)
    kw = dict(n_chan=C, d_model=24, n_heads=4, n_layers=2, d_z=d_z, d_dec=16,
              dec_layers=2, patch=1)
    if k_time > 1:
        kw["k_time"] = k_time
    m = TorchPixelMAE(**kw)
    args = dict(kw)
    args.update({"anomaly": True, "holdout_years": "2008",
                 "holdout_lon": "0,0", "k_time": k_time,
                 "time_block": time_block, "data": os.path.basename(path),
                 "steps": 10})
    torch.save({"model": m.state_dict(), "args": args, "d_z": d_z,
                "chan": [f"c{i}" for i in range(C)],
                "norm": np.stack([np.zeros(C), np.ones(C)], 1)}, path)
    return path


# ---- G5a: loss parity ------------------------------------------------------
def test_g5a():
    worst, lines = 0.0, []

    # First, THE GATHER, elementwise against ml/temporal.py's own function.
    # Every consumer of model inputs goes through it, so a drift here is a
    # drift in the objective, in the monitor and in the eval at once.
    rng = np.random.default_rng(5)
    T, P, d_z, K, S = 40, 30, 8, 6, 9
    Z = rng.normal(size=(T, P, d_z)).astype(np.float16)
    NBR = rng.integers(-1, P, size=(P, S)).astype(np.int64)
    NBR[:, 0] = np.arange(P)                       # slot 0 is the centre
    base = rng.integers(0, T - K - 1, size=17)
    p = rng.integers(0, P, size=17)
    for label, nbr_np, nbr_t in (("stencil 1", None, None),
                                 ("stencil 9", NBR, torch.as_tensor(NBR))):
        ref = gather_stencil(torch.from_numpy(Z), torch.as_tensor(base),
                             torch.as_tensor(p), nbr_t, K)
        got = gather_stencil_np(Z, base, p, nbr_np, K)
        dd = close(f"G5a gather ({label})", ref, got, 0.0 + 1e-12)
        worst = max(worst, dd)

    for stencil in (1, 9):
        d_zh, Kh, n = 8, 7, 5
        kw = dict(d_z=d_zh, d_model=32, n_heads=4, n_layers=2, k_max=Kh,
                  direct=(), stencil=stencil)
        tm, jmod = make_head_pair(**kw)
        zseq, mseq, sctx, ztgt, noise = make_window_batch(n, Kh, d_zh, stencil)
        for sigma in (0.0, 0.7):
            with torch.no_grad():
                zt = (torch_apply_znoise(zseq, noise, sigma, d_zh)
                      if sigma else zseq)
                tl, _, _ = torch_stage2_loss(tm, zt, mseq, sctx, ztgt)
            zj = apply_znoise(_j(zseq), _j(noise) if sigma else None, sigma,
                              d_zh)
            jl, _ = stage2_loss(jmod, zj, _j(mseq), _j(sctx), _j(ztgt))
            dd = close(f"G5a s{stencil} znoise {sigma}", tl, jl, 1e-5)
            worst = max(worst, dd)
            lines.append(f"s{stencil}/σ{sigma:g} {float(tl):.6f}/"
                         f"{float(jl):.6f}")
    print(f"  G5a loss parity — torch vs jax on identical head weights, "
          f"identical window batch and the IDENTICAL noise array (injected, "
          f"not drawn): max|Δ| {worst:.2e} (gate 1e-5). "
          + " · ".join(lines)
          + ". The window gather is bit-identical to "
            "ml/temporal.py:gather_stencil at stencil 1 and 9.")


# ---- G5b: one-step parity --------------------------------------------------
def _jax_grads(jmod, zseq, mseq, sctx, ztgt):
    graphdef, state = nnx.split(jmod)

    def f(st):
        m = nnx.merge(graphdef, st)
        loss, _ = stage2_loss(m, zseq, mseq, sctx, ztgt)
        return loss
    return graphdef, state, jax.grad(f)(state)


def _weight_delta(tm, jmod):
    """max |Δ| over every parameter, compared in torch's own key space."""
    sd = tm.state_dict()
    js = jc.export_temporal(jmod)
    if set(sd) != set(js):
        fail(f"G5b: key sets differ: {sorted(set(sd) ^ set(js))}")
        return float("inf")
    return max(float(np.max(np.abs(sd[k].detach().numpy().astype(np.float64)
                                   - js[k].astype(np.float64))))
               for k in sd)


def test_g5b():
    d_z, K, stencil, n = 8, 7, 9, 6
    kw = dict(d_z=d_z, d_model=32, n_heads=4, n_layers=2, k_max=K,
              direct=(), stencil=stencil)
    zseq, mseq, sctx, ztgt, _ = make_window_batch(n, K, d_z, stencil, seed=11)

    # --- plain SGD, lr 1e-2 -------------------------------------------------
    tm, jmod = make_head_pair(**kw)
    tm.train()
    before = _weight_delta(tm, jmod)
    tl, _, _ = torch_stage2_loss(tm, zseq, mseq, sctx, ztgt)
    opt = torch.optim.SGD(tm.parameters(), lr=1e-2)
    opt.zero_grad()
    tl.backward()
    opt.step()
    graphdef, state, grads = _jax_grads(jmod, _j(zseq), _j(mseq), _j(sctx),
                                        _j(ztgt))
    state2 = jax.tree_util.tree_map(lambda p_, g_: p_ - 1e-2 * g_, state,
                                    grads)
    d_sgd = _weight_delta(tm, nnx.merge(graphdef, state2))
    if not (d_sgd < 1e-6):
        fail(f"G5b SGD: max|Δweight| {d_sgd:.3e} >= 1e-6")

    # --- AdamW, the optimiser both trainers actually use --------------------
    # ADAM CANNOT MEET SGD's 1e-6, AND THE REASON IS THE ALGORITHM, NOT THE
    # PORT. At the first step Adam's update is m_hat/(sqrt(v_hat)+eps) ≈
    # sign(g), a discontinuous function of the gradient smoothed only over
    # |g| ~ eps = 1e-8. Two frameworks whose gradients agree to ~1e-9
    # therefore disagree by O(1) in the UPDATE wherever a gradient is that
    # small — and by AT MOST lr in the weight, because |sign(g)| <= 1 bounds
    # the first step at exactly lr per coordinate. That bound, not a measured
    # number rounded up, is the outer gate: 1e-3 = lr.
    #
    # Stating only that would be useless, so the gate is FOUR lines and three
    # of them are tight. Measured here, and the concentration is the whole
    # argument: the two worst weight deltas are 9.3e-5 and 7.1e-5, both on
    # `self_attn.in_proj_bias`, at entries whose |gradient| is 6.8e-10 and
    # 3.6e-10 — the key-projection bias, which softmax attention is very
    # nearly invariant to. The THIRD-worst tensor is three orders of magnitude
    # down at 2.4e-7. So:
    #   · the GRADIENTS agree to 1e-6 (measured 7.5e-8) — this is the line
    #     that actually says the objective matches, and it is checked first;
    #   · over the entries where Adam is well-conditioned (|g| > 1e-6), the
    #     SGD bar of 1e-6 still holds;
    #   · over every tensor EXCEPT the attention input-projection biases,
    #     1e-6 holds on all entries;
    #   · and over everything, lr.
    # None of the first three has an escape hatch in it, and a real
    # disagreement in the objective fails the first one.
    tm2, jmod3 = make_head_pair(**kw)
    tm2.train()
    tl2, _, _ = torch_stage2_loss(tm2, zseq, mseq, sctx, ztgt)
    opt = torch.optim.AdamW(tm2.parameters(), lr=1e-3, weight_decay=1e-4)
    opt.zero_grad()
    tl2.backward()
    tgrad = {k: v.grad.detach().numpy().copy()
             for k, v in tm2.named_parameters()}
    opt.step()
    graphdef, state, grads = _jax_grads(jmod3, _j(zseq), _j(mseq), _j(sctx),
                                        _j(ztgt))
    tx = optax.inject_hyperparams(optax.adamw)(learning_rate=1e-3,
                                               weight_decay=1e-4)
    ost = tx.init(state)
    upd, ost = tx.update(grads, ost, state)
    jmod4 = nnx.merge(graphdef, optax.apply_updates(state, upd))

    # THE GRADIENTS THEMSELVES — the quantity that says the objective agrees,
    # measured before any optimiser gets to smear it.
    gj = jc.export_temporal(nnx.merge(graphdef, grads))
    d_grad = max(float(np.max(np.abs(tgrad[k].astype(np.float64)
                                     - gj[k].astype(np.float64))))
                 for k in tgrad)
    if not (d_grad < 1e-6):
        fail(f"G5b: the GRADIENTS differ by {d_grad:.3e} >= 1e-6 — the "
             f"objective, not the optimiser")

    d_adam = _weight_delta(tm2, jmod4)
    if not (d_adam < 1e-3):
        fail(f"G5b AdamW: max|Δweight| {d_adam:.3e} >= lr (1e-3), which "
             f"bounds a first Adam step per coordinate — so this is not "
             f"sign(g) sensitivity, it is a real disagreement")
    sd, js = tm2.state_dict(), jc.export_temporal(jmod4)
    d_wc, n_wc, d_nb = 0.0, 0, 0.0
    for k in sd:
        g = tgrad.get(k)
        dk = np.abs(sd[k].detach().numpy().astype(np.float64)
                    - js[k].astype(np.float64))
        if "in_proj_bias" not in k:
            d_nb = max(d_nb, float(dk.max()))
        if g is None:
            continue
        sel = np.abs(g) > 1e-6
        n_wc += int(sel.sum())
        if sel.any():
            d_wc = max(d_wc, float(dk[sel].max()))
    if not (d_wc < 1e-6):
        fail(f"G5b AdamW (|g| > 1e-6 entries): max|Δweight| {d_wc:.3e} >= "
             f"1e-6")
    if not (d_nb < 1e-6):
        fail(f"G5b AdamW (every tensor but the attention in_proj_bias): "
             f"max|Δweight| {d_nb:.3e} >= 1e-6")
    if not (before < 1e-7):
        fail(f"G5b: the two heads did not start identical ({before:.3e})")
    print(f"  G5b one-step parity — same init (max|Δ| {before:.1e}), same "
          f"window batch: the GRADIENTS agree to {d_grad:.2e} (gate 1e-6), "
          f"and after one plain-SGD step at lr 1e-2 max|Δweight| is "
          f"{d_sgd:.2e} (gate 1e-6). Under AdamW (lr 1e-3, wd 1e-4): "
          f"{d_adam:.2e} over all parameters (gate = lr, the exact bound on a "
          f"first Adam step, since the update is ~sign(g)); {d_wc:.2e} over "
          f"the {n_wc:,} entries with |g| > 1e-6; and {d_nb:.2e} over every "
          f"tensor except the attention in_proj_bias, whose two worst entries "
          f"carry gradients of 7e-10 and 4e-10 and are where the whole "
          f"discrepancy lives")


# ---- G5c: 300 toy steps against the REAL ml/temporal.py --------------------
def _read_result(path):
    out = {}
    with open(path) as fh:
        for line in fh:
            rec = json.loads(line)
            if "stage2_result" in rec:
                out = rec["stage2_result"]
    return out


def test_g5c():
    tmp = tempfile.mkdtemp(prefix="g5c_")
    run = "_g5c_toy"
    run_dir = os.path.join(ML, "runs", run)
    data = os.path.join(tmp, "toy.npz")
    C, T = toy_tensor(data)
    cache_dir = os.path.join(ML, "cache")
    made = []
    try:
        os.makedirs(run_dir, exist_ok=True)
        os.makedirs(cache_dir, exist_ok=True)
        before_cache = set(os.listdir(cache_dir))
        toy_codec(os.path.join(run_dir, "pixelmae.pt"), C)
        env = dict(os.environ)
        # ml/temporal.py mirrors the head into CKPT_DIR every log step; point
        # it at the scratch directory so a test never writes to /opt.
        env["CKPT_DIR_OVERRIDE"] = os.path.join(tmp, "ckpt")
        env.pop("GITHUB_RUN_NUMBER", None)
        common = ["--K", "6", "--steps", "300", "--batch", "64",
                  "--lr", "1e-3", "--d-model", "32", "--layers", "2",
                  "--seed", "0"]
        p = subprocess.run([sys.executable, "-u", TEMPORAL, "--run", run,
                            "--data", data] + common,
                           capture_output=True, text=True, env=env, cwd=ROOT)
        if p.returncode:
            tail = "\n".join((p.stdout + p.stderr).strip().splitlines()[-20:])
            fail(f"G5c: ml/temporal.py exited {p.returncode}:\n{tail}")
            return
        # The cache AND its completeness marker: `<cache>.done` holds the byte
        # size and is written only after the final flush, because a memmap is
        # allocated at its full (T, P, d_z) shape before the first month is
        # written and nothing else can tell a finished pass from an abandoned
        # one. embed_cache_sync.py:push refuses to publish a cache without it.
        made = [f for f in os.listdir(cache_dir)
                if f not in before_cache and f.startswith("Z_")]
        z = [f for f in made if f.endswith(".npy")]
        if len(z) != 1 or sorted(made) != sorted([z[0], z[0] + ".done"]):
            fail(f"G5c: expected exactly one new embed cache in ml/cache and "
                 f"its .done marker, found {made}")
            return
        zpath = os.path.join(cache_dir, z[0])
        if open(zpath + ".done").read().strip() != str(os.path.getsize(zpath)):
            fail(f"G5c: the completeness marker does not match the cache it "
                 f"marks — it must record these bytes, not this path")
            return
        torch_res = json.load(open(os.path.join(run_dir, "temporal.json")))

        jout = os.path.join(tmp, "jax")
        p2 = subprocess.run([sys.executable, "-u", JTRAIN2, "--data", data,
                             "--ckpt", os.path.join(run_dir, "pixelmae.pt"),
                             "--z", zpath, "--out", jout] + common,
                            capture_output=True, text=True, cwd=ROOT)
        if p2.returncode:
            tail = "\n".join((p2.stdout + p2.stderr).strip().splitlines()[-20:])
            fail(f"G5c: train_stage2.py exited {p2.returncode}:\n{tail}")
            return
        jax_res = _read_result(os.path.join(jout, "metrics.jsonl"))
        if not jax_res:
            fail("G5c: the JAX run wrote no stage2_result record")
            return

        tp = float(torch_res["z_t+1"]["mse_persistence"])
        jp = float(jax_res["z_mse_persistence"])
        rel = abs(tp - jp) / max(abs(tp), 1e-12)
        # THE HARD HALF. Same Z, same window population, same eval draw (both
        # take np.random.default_rng(seed).choice over the identical
        # population) — so persistence is the SAME arithmetic on the SAME
        # numbers and may differ only by float reduction order.
        if not (rel < 1e-4):
            fail(f"G5c: z_mse_persistence differs by {rel:.3e} relative "
                 f"(torch {tp:.6f}, jax {jp:.6f}). Same Z and same eval draw "
                 f"should give the same baseline; a difference here means the "
                 f"held-out window population is not the same one.")
        tm_ = float(torch_res["z_t+1"]["mse_model"])
        jm_ = float(jax_res["z_mse_model"])
        tr_, jr_ = tm_ / tp, jm_ / jp
        # A BAND, not an equality: the RNG streams that pick TRAINING windows
        # cannot match across frameworks, so the two runs see different
        # batches. [0.5, 2.0] on the ratio-of-ratios is the statement "the
        # same objective is being minimised at the same rate"; a trainer that
        # had, say, scored only the last window position would sit outside it.
        band = jr_ / tr_
        if not (0.5 <= band <= 2.0):
            fail(f"G5c: jax model/persistence {jr_:.4f} against torch's "
                 f"{tr_:.4f} — ratio {band:.3f} outside [0.5, 2.0]")
        if not (tr_ < 1.0 and jr_ < 1.0):
            fail(f"G5c: a 300-step head should beat persistence on this toy "
                 f"(torch {tr_:.4f}, jax {jr_:.4f}); if neither does, the "
                 f"band compares two nulls")
        print(f"  G5c 300 toy steps on the SAME Z — the real ml/temporal.py "
              f"reads model/persistence {tm_:.4f}/{tp:.4f} = {tr_:.4f} and "
              f"ml/jaxport/train_stage2.py {jm_:.4f}/{jp:.4f} = {jr_:.4f}, "
              f"ratio {band:.3f} inside the pre-registered band [0.5, 2.0]. "
              f"The persistence baselines agree to {rel:.2e} relative, which "
              f"is the same-population check the band cannot make.")
    finally:
        shutil.rmtree(run_dir, ignore_errors=True)
        shutil.rmtree(tmp, ignore_errors=True)
        for f in made:
            try:
                os.remove(os.path.join(cache_dir, f))
            except OSError:
                pass


# ---- G5d: export round trip + the two knobs pinned -------------------------
def test_g5d():
    d_z, K, stencil, n = 8, 7, 9, 5
    kw = dict(d_z=d_z, d_model=32, n_heads=4, n_layers=2, k_max=K,
              direct=(), stencil=stencil)

    # --- the round trip: JAX head → .pt → torch, strict, forward-matched ----
    jmod = jm.TemporalTransformer(**kw, rngs=nnx.Rngs(17))
    graphdef, state = nnx.split(jmod)
    # PERTURB, so the test cannot pass on two identical inits: the round trip
    # must carry values, not merely shapes.
    kk = jax.random.key(1)
    state = jax.tree_util.tree_map(
        lambda v: v + 0.05 * jax.random.normal(kk, v.shape, v.dtype), state)
    jmod = nnx.merge(graphdef, state)
    args = {"K": K, "d_model": 32, "layers": 2, "stencil": stencil,
            "ring_km": "0", "seed": 0, "direct": "", "unroll": 1,
            "season_phase": "month", "input_quant": "", "steps": 300,
            "lr": 1e-3}
    tmp = tempfile.mkdtemp(prefix="g5d_")
    try:
        pt = os.path.join(tmp, "temporal.pt")
        blob = jc.export_temporal_pt(jmod, args, path=pt, step=300)
        for k in ("model", "args", "step"):
            if k not in blob:
                fail(f"G5d: the exported head blob has no {k!r} key")
        for k in ("opt", "sched"):
            if k in blob:
                fail(f"G5d: the exported head carries {k!r}. optax state is "
                     f"not torch Adam state and a blob that claims otherwise "
                     f"would be resumed as a continuation.")
        tk = torch.load(pt, map_location="cpu", weights_only=False)
        # THE SAME RECONSTRUCTION ml/rollout_spatial.py MAKES, from the same
        # fields: k_max off `pos.weight`'s own shape, n_heads=4 hard-coded,
        # everything else out of `args`.
        ta = tk["args"]
        k_tbl = tk["model"]["pos.weight"].shape[0]
        tmod = TorchTemporal(d_z=d_z, d_model=ta["d_model"], n_heads=4,
                             n_layers=ta["layers"], k_max=k_tbl,
                             direct=(), stencil=ta["stencil"])
        tmod.load_state_dict(tk["model"])            # strict=True by default
        tmod.eval()
        if k_tbl != K:
            fail(f"G5d: pos.weight is {k_tbl} rows, K is {K} — a roll reads "
                 f"k_max off this shape and would rebuild the head wrong")
        zseq, mseq, sctx, ztgt, _ = make_window_batch(n, K, d_z, stencil,
                                                      seed=23)
        with torch.no_grad():
            tp_, th_ = tmod(zseq, mseq, sctx)
        jp_, jh_ = jmod(_j(zseq), _j(mseq), _j(sctx))
        d_fw = max(close("G5d round-trip pred", tp_, jp_, 1e-5),
                   close("G5d round-trip hidden", th_, jh_, 1e-5))
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    # --- --grad-clip semantics ---------------------------------------------
    # Two statements no end-to-end band could make: ABOVE the norm the clip is
    # BIT-IDENTICAL to no clip (which is what lets an arm turn it on without
    # leaving the archive's code path when it never binds), and BELOW it the
    # gradient is scaled by exactly CLIP/norm and nothing else.
    _, jmod5 = make_head_pair(**kw)
    zseq, mseq, sctx, ztgt, noise = make_window_batch(n, K, d_z, stencil,
                                                      seed=31)
    graphdef, state, grads = _jax_grads(jmod5, _j(zseq), _j(mseq), _j(sctx),
                                        _j(ztgt))
    gnorm = float(optax.global_norm(grads))
    plain = optax.inject_hyperparams(optax.adamw)(learning_rate=1e-3,
                                                  weight_decay=1e-4)
    upd_ref, _ = plain.update(grads, plain.init(state), state)

    hi = optax.chain(optax.clip_by_global_norm(gnorm * 10),
                     optax.inject_hyperparams(optax.adamw)(
                         learning_rate=1e-3, weight_decay=1e-4))
    upd_hi, _ = hi.update(grads, hi.init(state), state)
    d_hi = max(float(np.max(np.abs(np.asarray(x) - np.asarray(y))))
               for x, y in zip(jax.tree_util.tree_leaves(upd_ref),
                               jax.tree_util.tree_leaves(upd_hi)))
    if d_hi != 0.0:
        fail(f"G5d grad-clip: a clip at 10x the norm changed the update by "
             f"{d_hi:.3e}; a clip that never binds must be bit-identical")

    CLIP = gnorm / 4.0
    lo = optax.chain(optax.clip_by_global_norm(CLIP),
                     optax.inject_hyperparams(optax.adamw)(
                         learning_rate=1e-3, weight_decay=1e-4))
    upd_lo, _ = lo.update(grads, lo.init(state), state)
    scaled = jax.tree_util.tree_map(lambda g_: g_ * (CLIP / gnorm), grads)
    upd_sc, _ = plain.update(scaled, plain.init(state), state)
    d_lo = max(float(np.max(np.abs(np.asarray(x) - np.asarray(y))))
               for x, y in zip(jax.tree_util.tree_leaves(upd_lo),
                               jax.tree_util.tree_leaves(upd_sc)))
    if not (d_lo < 1e-6):
        fail(f"G5d grad-clip: clipping at {CLIP:.4f} is not the same as "
             f"scaling the gradient by CLIP/norm (max|Δ| {d_lo:.3e})")

    # --- --input-znoise semantics ------------------------------------------
    # DEAD SLOTS STAY EXACT ZEROS (zero IS the dead-slot encoding and the roll
    # feeds exact zeros there), and live slots move by exactly sigma * noise.
    sigma = 0.7
    zj = _j(zseq)
    out = np.asarray(apply_znoise(zj, _j(noise), sigma, d_z))
    z4 = np.asarray(zseq).reshape(n, K, stencil, d_z)
    o4 = out.reshape(n, K, stencil, d_z)
    live = (z4 != 0).any(-1, keepdims=True)
    n_dead = int((~live).sum())
    d_dead = float(np.max(np.abs(o4[np.broadcast_to(~live, o4.shape)]))) \
        if n_dead else 0.0
    if d_dead != 0.0:
        fail(f"G5d znoise: a DEAD slot moved by {d_dead:.3e}; zero is the "
             f"dead-slot encoding and the roll feeds exact zeros there")
    want = z4 + np.asarray(noise) * sigma * live
    d_live = float(np.max(np.abs(o4 - want)))
    if not (d_live < 1e-6):
        fail(f"G5d znoise: live slots differ from z + sigma*noise by "
             f"{d_live:.3e}")
    # …and the torch transcription agrees with it on the identical arrays.
    with torch.no_grad():
        tz = torch_apply_znoise(zseq, noise, sigma, d_z)
    d_tj = close("G5d znoise torch-vs-jax", tz, out, 1e-6)

    # --- the schedules, at the fleet's own configuration --------------------
    # `expdecay` with an ABSOLUTE half-life is what every xl stage-2 arm
    # trains under, and the one property that makes it horizon-free is that
    # lr(s) does not depend on --steps. Asserted rather than described.
    class _A:
        lr_schedule, lr_warmup, lr_halflife = "expdecay", 2000, 40000.0
        lr_cooldown_frac, steps = 0.0, 200000
    a1 = _A()
    a2 = _A()
    a2.steps = 60000
    d_hz = max(abs(lr_factor(e, a1) - lr_factor(e, a2))
               for e in (0, 1, 1999, 2000, 5000, 40000, 59999))
    if d_hz != 0.0:
        fail(f"G5d expdecay is not horizon-free: the same step gives "
             f"different rates at --steps 200000 and 60000 (max|Δ| "
             f"{d_hz:.3e})")
    peak = lr_factor(1999, a1)
    half = lr_factor(1999 + 40000, a1)
    if not (abs(half / peak - 0.5) < 1e-9):
        fail(f"G5d expdecay: one half-life past the warmup peak the factor is "
             f"{half / peak:.6f} of it, not 0.5")

    print(f"  G5d head round trip — a perturbed JAX head exported with "
          f"convert.export_temporal_pt loads into the torch "
          f"TemporalTransformer with strict=True (k_max recovered from "
          f"pos.weight, n_heads=4, the same reconstruction rollout_spatial "
          f"makes) and forward-matches to {d_fw:.2e} (gate 1e-5); no "
          f"opt/sched keys ride along. --grad-clip at 10x the norm is "
          f"BIT-IDENTICAL to unclipped (Δ {d_hi:g}) and at norm/4 equals "
          f"scaling by CLIP/norm to {d_lo:.2e}. --input-znoise leaves all "
          f"{n_dead:,} dead slot-components at EXACT zeros, moves live ones "
          f"by exactly sigma*noise ({d_live:.2e}), and agrees with the torch "
          f"transcription to {d_tj:.2e}. expdecay is horizon-free (Δ {d_hz:g} "
          f"between --steps 200000 and 60000) and halves in exactly "
          f"--lr-halflife steps.")


# ---- G5e: the block-z axis adoption ----------------------------------------
def test_g5e():
    T = 72
    months = [f"{2004 + i // 12:04d}-{i % 12 + 1:02d}" for i in range(T)]
    moy = np.array([int(m[5:7]) - 1 for m in months])
    hold = {"2008"}
    t_hold = np.array([m[:4] in hold for m in months])
    rng = np.random.default_rng(9)
    rapid = np.array([[float(i), 15.0 + rng.normal()] for i in range(0, T, 2)])
    d = {}
    ck_args = {"k_time": 5, "time_block": "5"}

    # THE REFERENCE: ml/temporal.py:1565-1600, transcribed.
    BLK_ref = BlockAxis("5", months, None, None, None)
    rapid_ref = BLK_ref.remap_rows(rapid.copy())
    months_ref = list(BLK_ref.labels)
    moy_ref = np.array([int(m[5:7]) - 1 for m in months_ref])
    thold_ref = np.array([
        t_hold[BLK_ref.rows[b, :int(BLK_ref.n_bins[b])]].any()
        for b in range(BLK_ref.n_blocks)])
    T_ref = BLK_ref.n_blocks

    BLK, months_g, moy_g, thold_g, T_g, rapid_g = adopt_block_axis(
        ck_args, d, list(months), moy.copy(), t_hold.copy(), T, rapid.copy(),
        6, 8)
    if months_g != months_ref:
        fail("G5e: block labels differ from ml/temporal.py's block branch")
    if T_g != T_ref:
        fail(f"G5e: block count {T_g} != {T_ref}")
    close("G5e moy", moy_ref, moy_g, 1e-12)
    if not np.array_equal(thold_ref, thold_g):
        fail("G5e: the FUSED t_hold differs — any held-out bin of a block "
             "must make the block held out")
    close("G5e remapped rapid rows", rapid_ref, rapid_g, 1e-12)
    if int(BLK.k_max) != 5:
        fail(f"G5e: k_max {BLK.k_max} != 5")
    # The fusing must actually FIRE on this toy, or the equality above is two
    # all-False arrays agreeing.
    if not (0 < int(thold_g.sum()) < T_g):
        fail(f"G5e: {int(thold_g.sum())}/{T_g} blocks held out — the fusing "
             f"rule is not exercised")
    n_fused = int(sum(
        0 < int(t_hold[BLK.rows[b, :int(BLK.n_bins[b])]].sum())
        < int(BLK.n_bins[b]) for b in range(BLK.n_blocks)))

    # A per-bin codec must pass everything through UNTOUCHED.
    out = adopt_block_axis({"k_time": 1}, d, list(months), moy.copy(),
                           t_hold.copy(), T, rapid.copy(), 6, 8)
    if out[0] is not None or out[1] != months or out[4] != T:
        fail("G5e: a per-bin codec (k_time 1) did not pass the axis through "
             "untouched")

    # --time-stride on a block codec REFUSES.
    refused = 0
    try:
        adopt_block_axis(ck_args, d, list(months), moy.copy(),
                         t_hold.copy(), T, rapid.copy(), 6, 8, time_stride=6)
    except SystemExit as e:
        if "two time surgeries" in str(e):
            refused += 1
    # …and so does a block codec with no `time_block` to say how it was cut.
    try:
        adopt_block_axis({"k_time": 5, "time_block": ""}, d, list(months),
                         moy.copy(), t_hold.copy(), T, rapid.copy(), 6, 8)
    except SystemExit as e:
        if "cannot say how its blocks were cut" in str(e):
            refused += 1
    if refused != 2:
        fail(f"G5e: the block path made {refused}/2 refusals "
             f"(--time-stride, and k_time>1 with no time_block)")

    print(f"  G5e block-z axis adoption — on a 72-bin toy under a k_time 5 "
          f"block codec, adopt_block_axis returns the same {T_g} labels, the "
          f"same month-of-year, the same FUSED t_hold "
          f"({int(thold_g.sum())} blocks held out, {n_fused} of them by a "
          f"PARTIAL overlap the fusing rule is what resolves) and the same "
          f"remapped RAPID rows ({len(rapid_g)} of {len(rapid)}) as "
          f"ml/temporal.py's block branch. A per-bin codec passes the axis "
          f"through untouched, and --time-stride on a block codec and a "
          f"block codec with no time_block both refuse.")


def main():
    print("tests/test_jaxport_train_s2.py — tier-3b gates G5a-G5e, CPU, "
          "fp32\n")
    for fn in (test_g5a, test_g5b, test_g5c, test_g5d, test_g5e):
        fn()
    if FAILURES:
        print("\nFAILED:")
        for f in FAILURES:
            print("  -", f)
        return 1
    print("\ntests/test_jaxport_train_s2.py: all 5 gates passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())

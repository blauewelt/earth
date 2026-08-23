#!/usr/bin/env python3
"""Tier-3 gates for the JAX stage-1 trainer (`ml/plans/JAX_PORT.md` §5, G4).

Tier 1 pinned the FORWARD pass. A trainer adds three things a forward-parity
test cannot see — the objective, the update, and the trajectory — plus two
pieces of machinery tier 3 needs and tiers 1–2 did not: a checkpoint that
travels BACK to torch, and a codec whose encoder input is a k_time × C grid.
Each is a pre-registered gate here, CPU only, no network, fp32.

  **G4a — loss parity.** Identical converted weights, identical batch,
  identical mask: torch's `loss_rec` and `loss_nei` and the JAX ones agree to
  1e-5. Covered at patch 1, patch 3 and k_time 7, because the three take
  different routes into `encode` and only the first was ever exercised.

  **G4b — one-step parity.** Same init, same batch, plain SGD at lr 1e-2:
  max |Δweight| between the frameworks < 1e-6 over every parameter. Then the
  same step under Adam(W), whose achievable tolerance is looser and is
  MEASURED and asserted rather than assumed — two independent Adam
  implementations agree to the accumulated rounding of their own moment
  arithmetic, not to the gradient's.

  **G4c — short toy training.** 300 steps of the REAL `ml/train.py` and 300
  steps of `ml/jaxport/train_stage1.py`, same toy tensor, same geometry. The
  RNG streams cannot match across frameworks, so this is a BAND and not an
  equality: the two runs' `loss_rec` must fall by comparable factors. This is
  the one check that runs the torch objective through the actual operational
  trainer rather than through a transcription of it.

  **G4d — round-trip export.** torch state_dict → NNX → torch state_dict must
  be IDENTICAL, key set and values alike; and a JAX-side `encode` must match
  the torch `encode` of the exported checkpoint to 1e-5. This is the direction
  `JAX_PORT.md` §1b names as the cheap validation of a TPU-trained codec —
  score it through the UNCHANGED torch eval ladder — so an error here is an
  error in every TPU result the programme could ever report.

  **G4e — k_time=7 forward parity.** The E-047 block codec's `encode` (a
  k_time × C grid whose cell token is chan_emb[c] + time_emb[j]) and `query`
  (which REQUIRES `tpos`), against torch, plus the converter's refusal in both
  directions across the k_time boundary.

WHERE G4a's TORCH REFERENCE COMES FROM, stated plainly because it is the one
weak joint in this file. `ml/train.py`'s `step_loss` and `step_loss_block` are
CLOSURES inside `main()` over a dozen of its locals; they cannot be imported
without running the trainer. So the torch side of G4a is a transcription of
`ml/train.py` lines 686–720 (per-bin) and 647–684 (block), reproduced below
with the loop's own combination `loss = l_rec + 0.5 * l_nei`. A transcription
tests what was transcribed, which is why G4c exists beside it and runs the
REAL trainer end to end: G4a says the arithmetic agrees term by term on one
batch, G4c says the two trainers as a whole move a loss the same way.

    python3 tests/test_jaxport_train.py
"""
import json
import os
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
from jaxport import models as jm                                 # noqa: E402
from jaxport import convert as jc                                # noqa: E402
from jaxport.train_stage1 import stage1_loss                     # noqa: E402

TRAIN = os.path.join(ML, "train.py")
JTRAIN = os.path.join(ML, "jaxport", "train_stage1.py")
FAILURES = []

REC_W_VISIBLE = 0.1        # ml/train.py --rec-w-visible default
MASK_RATIO = 0.5           # ml/train.py --mask-ratio default
NEI = [(1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1)]


def fail(msg):
    FAILURES.append(msg)


def close(name, a, b, tol):
    a = np.asarray(a.detach().cpu().numpy() if hasattr(a, "detach") else a,
                   np.float64)
    b = np.asarray(b, np.float64)
    if a.shape != b.shape:
        fail(f"{name}: shape {a.shape} vs {b.shape}")
        return float("inf")
    d = float(np.max(np.abs(a - b)))
    if not (d < tol):
        fail(f"{name}: max|Δ| {d:.3e} >= {tol:g}")
    return d


def _j(t):
    return jnp.asarray(t.detach().cpu().numpy() if hasattr(t, "detach")
                       else np.asarray(t))


# --------------------------------------------------------------------------
# the torch reference: a TRANSCRIPTION of ml/train.py's closures (see the
# module docstring for why it cannot be an import)
# --------------------------------------------------------------------------
_HUBER = torch.nn.HuberLoss(reduction="none")


def torch_step_loss(model, v, o, mask, ctx, dxyz, vn, on, cwt, vp=None,
                    op=None):
    """ml/train.py:686-720 (`step_loss`), verbatim in structure."""
    B, C = v.shape
    if vp is not None:
        z = model.encode(vp, op, mask, ctx)
    else:
        z = model.encode(v * (~mask), o, mask, ctx)
    qc = torch.arange(C)[None, :].expand(B, -1)
    off0 = torch.zeros(B, C, 3, dtype=torch.long)
    pred = model.query(z, qc, off0)
    l_rec = _HUBER(pred, v)
    w = (mask.float() + REC_W_VISIBLE * (o & ~mask).float()) * cwt[None, :]
    l_rec = (l_rec * w).sum() / w.sum().clamp(min=1)
    offn = dxyz[:, None, :].expand(-1, C, -1).long()
    predn = model.query(z, qc, offn)
    wn = on.float() * cwt[None, :]
    l_nei = (_HUBER(predn, vn) * wn).sum() / wn.sum().clamp(min=1)
    return l_rec, l_nei


def torch_step_loss_block(model, v, o, mask, ctx, dxyz, vn, on, cwt):
    """ml/train.py:647-684 (`step_loss_block`), verbatim in structure."""
    B, KT, C = v.shape
    z = model.encode(v * (~mask), o, mask, ctx)
    qc = (torch.arange(C)[None, None, :].expand(B, KT, -1).reshape(B, KT * C))
    qt = (torch.arange(KT)[None, :, None].expand(B, -1, C).reshape(B, KT * C))
    off0 = torch.zeros(B, KT * C, 3, dtype=torch.long)
    pred = model.query(z, qc, off0, qt).reshape(B, KT, C)
    l_rec = _HUBER(pred, v)
    w = (mask.float() + REC_W_VISIBLE * (o & ~mask).float()) \
        * cwt[None, None, :]
    l_rec = (l_rec * w).sum() / w.sum().clamp(min=1)
    offn = dxyz[:, None, :].expand(-1, KT * C, -1).long()
    predn = model.query(z, qc, offn, qt).reshape(B, KT, C)
    wn = on.float() * cwt[None, None, :]
    l_nei = (_HUBER(predn, vn) * wn).sum() / wn.sum().clamp(min=1)
    return l_rec, l_nei


# --------------------------------------------------------------------------
# fixtures
# --------------------------------------------------------------------------
def make_pair(seed=1, **kw):
    """A torch PixelMAE and the NNX one loaded from its state_dict."""
    torch.manual_seed(seed)
    tm = TorchPixelMAE(**kw).eval()
    jmod = jm.PixelMAE(**kw, rngs=nnx.Rngs(0))
    jc.load_pixelmae(tm.state_dict(), jmod)
    return tm, jmod


def make_batch(B, C, patch=1, k_time=1, seed=2):
    """One batch in the shapes ml/train.py's own gather produces."""
    g = torch.Generator().manual_seed(seed)
    shape = (B, k_time, C) if k_time > 1 else (B, C)
    v = torch.randn(*shape, generator=g)
    o = torch.rand(*shape, generator=g) > 0.2
    mask = (torch.rand(*shape, generator=g) < MASK_RATIO) & o
    ctx = torch.randn(B, 4, generator=g)
    vn = torch.randn(*shape, generator=g)
    on = torch.rand(*shape, generator=g) > 0.3
    pick = torch.randint(0, len(NEI), (B,), generator=g)
    dxyz = torch.as_tensor(np.array([NEI[int(i)] for i in pick]))
    cwt = torch.ones(C)
    cwt[0] = 3.0                       # a non-trivial --upweight-chans weight
    vp = op = None
    if patch > 1:
        p2 = patch * patch
        vp = torch.randn(B, C, p2, generator=g)
        op = torch.rand(B, C, p2, generator=g) > 0.25
        o = op[..., p2 // 2]
        mask = (torch.rand(B, C, generator=g) < MASK_RATIO) & o
    return dict(v=v, o=o, mask=mask, ctx=ctx, vn=vn, on=on, dxyz=dxyz,
                cwt=cwt, vp=vp, op=op)


def jax_loss(jmod, b, C, k_time=1):
    """`stage1_loss` fed the identical arrays — the trainer's own function."""
    B = b["v"].shape[0]
    M = C * k_time
    qc = jnp.tile(jnp.arange(C, dtype=jnp.int32), (B, k_time))
    qt = (jnp.repeat(jnp.arange(k_time, dtype=jnp.int32), C)[None, :]
          .repeat(B, 0)) if k_time > 1 else None
    off0 = jnp.zeros((B, M, 3), jnp.int32)
    offn = jnp.repeat(_j(b["dxyz"]).astype(jnp.int32)[:, None, :], M, axis=1)
    cw = jnp.tile(_j(b["cwt"]), k_time) if k_time > 1 else _j(b["cwt"])
    if b["vp"] is not None:
        enc_v, enc_o = _j(b["vp"]), _j(b["op"])
    else:
        enc_v = _j(b["v"]) * ~_j(b["mask"])
        enc_o = _j(b["o"])
    return stage1_loss(jmod, enc_v, enc_o, _j(b["v"]), _j(b["o"]),
                       _j(b["mask"]), _j(b["ctx"]), qc, off0, offn,
                       _j(b["vn"]), _j(b["on"]), cw, REC_W_VISIBLE, qt)


# ---- G4a: loss parity ------------------------------------------------------
def test_g4a():
    worst = 0.0
    lines = []
    for label, kw, patch, k_time in (
            ("patch=1", dict(n_chan=9, d_model=32, n_heads=4, n_layers=2,
                             d_z=8, d_dec=16, dec_layers=2, patch=1), 1, 1),
            ("patch=3", dict(n_chan=9, d_model=32, n_heads=4, n_layers=2,
                             d_z=8, d_dec=16, dec_layers=2, patch=3), 3, 1),
            ("k_time=7", dict(n_chan=9, d_model=32, n_heads=4, n_layers=2,
                              d_z=8, d_dec=16, dec_layers=2, patch=1,
                              k_time=7), 1, 7)):
        C = kw["n_chan"]
        tm, jmod = make_pair(**kw)
        b = make_batch(6, C, patch=patch, k_time=k_time)
        with torch.no_grad():
            if k_time > 1:
                tr, tn = torch_step_loss_block(tm, b["v"], b["o"], b["mask"],
                                               b["ctx"], b["dxyz"], b["vn"],
                                               b["on"], b["cwt"])
            else:
                tr, tn = torch_step_loss(tm, b["v"], b["o"], b["mask"],
                                         b["ctx"], b["dxyz"], b["vn"],
                                         b["on"], b["cwt"], b["vp"], b["op"])
        jr, jn = jax_loss(jmod, b, C, k_time)
        d1 = close(f"G4a {label} loss_rec", tr, jr, 1e-5)
        d2 = close(f"G4a {label} loss_nei", tn, jn, 1e-5)
        worst = max(worst, d1, d2)
        lines.append(f"{label} rec {float(tr):.6f}/{float(jr):.6f} "
                     f"nei {float(tn):.6f}/{float(jn):.6f}")
    print(f"  G4a loss parity — torch vs jax on identical weights, batch and "
          f"mask: max|Δ| {worst:.2e} (gate 1e-5) over patch 1, patch 3 and "
          f"k_time 7. " + " · ".join(lines))


# ---- G4b: one-step parity --------------------------------------------------
def _jax_grads(jmod, b, C):
    graphdef, state = nnx.split(jmod)

    def f(st):
        m = nnx.merge(graphdef, st)
        lr_, ln_ = jax_loss(m, b, C)
        return lr_ + 0.5 * ln_
    return graphdef, state, jax.grad(f)(state)


def _weight_delta(tm, jmod):
    """max |Δ| over every parameter, compared in torch's own key space."""
    sd = tm.state_dict()
    js = jc.export_pixelmae(jmod)
    if set(sd) != set(js):
        fail(f"G4b: key sets differ: {sorted(set(sd) ^ set(js))}")
        return float("inf")
    return max(float(np.max(np.abs(sd[k].detach().numpy().astype(np.float64)
                                   - js[k].astype(np.float64))))
               for k in sd)


def test_g4b():
    kw = dict(n_chan=9, d_model=32, n_heads=4, n_layers=2, d_z=8, d_dec=16,
              dec_layers=2, patch=1)
    C = kw["n_chan"]

    # --- plain SGD, lr 1e-2 -------------------------------------------------
    tm, jmod = make_pair(**kw)
    tm.train()
    b = make_batch(6, C)
    before = _weight_delta(tm, jmod)
    tr, tn = torch_step_loss(tm, b["v"], b["o"], b["mask"], b["ctx"],
                             b["dxyz"], b["vn"], b["on"], b["cwt"])
    opt = torch.optim.SGD(tm.parameters(), lr=1e-2)
    opt.zero_grad()
    (tr + 0.5 * tn).backward()
    opt.step()
    graphdef, state, grads = _jax_grads(jmod, b, C)
    state = jax.tree_util.tree_map(lambda p, g: p - 1e-2 * g, state, grads)
    jmod2 = nnx.merge(graphdef, state)
    d_sgd = _weight_delta(tm, jmod2)
    if not (d_sgd < 1e-6):
        fail(f"G4b SGD: max|Δweight| {d_sgd:.3e} >= 1e-6")

    # --- AdamW, the optimiser the trainers actually use ---------------------
    # ADAM CANNOT MEET SGD's 1e-6, AND THE REASON IS THE ALGORITHM, NOT THE
    # PORT. At the first step Adam's update is
    #     m_hat / (sqrt(v_hat) + eps) = (1-b1)g/(1-b1) / (|g| + eps)
    #                                 ≈ sign(g),
    # so it is a DISCONTINUOUS function of the gradient at g = 0 and the
    # discontinuity is smoothed only over |g| ~ eps = 1e-8. Two frameworks
    # whose gradients agree to ~1e-9 therefore disagree by O(1) in the UPDATE
    # wherever a gradient happens to be that small, and by lr·O(1) = 1e-3 in
    # the weight. Measured here: the five largest weight deltas are all on
    # tensors whose minimum |gradient| is 1e-9 to 1e-13, while the gradients
    # themselves match to the SGD bar.
    #
    # So the gate is stated in two parts rather than widened in one:
    #   · over EVERY parameter, 2e-5 — the measured tolerance, asserted;
    #   · over the entries where |g| > 1e-6, i.e. where Adam's update is
    #     well-conditioned, the SGD bar of 1e-6 still holds.
    # A real disagreement in the objective or the update rule fails the second
    # line, which is the one that has no escape hatch in it.
    tm2, jmod3 = make_pair(**kw)
    tm2.train()
    tr, tn = torch_step_loss(tm2, b["v"], b["o"], b["mask"], b["ctx"],
                             b["dxyz"], b["vn"], b["on"], b["cwt"])
    opt = torch.optim.AdamW(tm2.parameters(), lr=1e-3, weight_decay=1e-4)
    opt.zero_grad()
    (tr + 0.5 * tn).backward()
    tgrad = {k: v.grad.detach().numpy().copy()
             for k, v in tm2.named_parameters()}
    opt.step()
    graphdef, state, grads = _jax_grads(jmod3, b, C)
    tx = optax.inject_hyperparams(optax.adamw)(learning_rate=1e-3,
                                               weight_decay=1e-4)
    ost = tx.init(state)
    upd, ost = tx.update(grads, ost, state)
    jmod4 = nnx.merge(graphdef, optax.apply_updates(state, upd))
    d_adam = _weight_delta(tm2, jmod4)
    if not (d_adam < 2e-5):
        fail(f"G4b AdamW: max|Δweight| {d_adam:.3e} >= 2e-5")
    sd, js = tm2.state_dict(), jc.export_pixelmae(jmod4)
    d_wc, n_wc = 0.0, 0
    for k in sd:
        g = tgrad.get(k)
        if g is None:
            continue
        sel = np.abs(g) > 1e-6
        n_wc += int(sel.sum())
        if sel.any():
            d_wc = max(d_wc, float(np.max(np.abs(
                sd[k].detach().numpy().astype(np.float64)[sel]
                - js[k].astype(np.float64)[sel]))))
    if not (d_wc < 1e-6):
        fail(f"G4b AdamW (|g| > 1e-6 entries): max|Δweight| {d_wc:.3e} "
             f">= 1e-6")
    if not (before < 1e-7):
        fail(f"G4b: the two models did not start identical ({before:.3e})")
    print(f"  G4b one-step parity — same init (max|Δ| {before:.1e}), same "
          f"batch: after one plain-SGD step at lr 1e-2 max|Δweight| "
          f"{d_sgd:.2e} (gate 1e-6). Under AdamW (lr 1e-3, wd 1e-4) it is "
          f"{d_adam:.2e} over all parameters (gate 2e-5 — Adam's first update "
          f"is ~sign(g), so a 1e-9 gradient disagreement is O(1) in the "
          f"update) and {d_wc:.2e} over the {n_wc:,} entries with |g| > 1e-6, "
          f"where the SGD bar of 1e-6 still holds")


# ---- G4c: short toy training ----------------------------------------------
def toy(path, C=12, T=48, H=12, W=14):
    """A toy tensor with something LEARNABLE in it.

    Pure noise would leave both trainers moving `loss_rec` by rounding, and
    the band would then compare two nulls and pass — the "a run that trains on
    all-zeros completes too" failure, one level up. So every channel is a
    different loading of ONE latent field plus channel noise: a masked channel
    is genuinely predictable from the visible ones, which is the structure the
    masked objective exists to find, and both trainers must find it.
    """
    rng = np.random.default_rng(20260823)
    latent = rng.normal(size=(T, H, W, 1))
    load = rng.normal(size=(1, 1, 1, C)) * 1.5
    X = (latent * load + 0.35 * rng.normal(size=(T, H, W, C))
         ).astype(np.float32)
    X[:, rng.random((H, W)) < 0.25, :] = np.nan
    X[rng.random(X.shape) < 0.08] = np.nan
    months = np.array([f"{2004 + i // 12:04d}-{i % 12 + 1:02d}"
                       for i in range(T)])
    np.savez(path, X=X, months=months, lats=np.linspace(0, 70, H),
             lons=np.linspace(-100, 20, W),
             chan=np.array([f"c{i}" for i in range(C)]),
             norm=np.stack([np.zeros(C), np.ones(C)], 1).astype(np.float32),
             rapid=np.array([[float(i), 15.0 + rng.normal()]
                             for i in range(0, T, 2)]))


def losses(out):
    vals = []
    with open(os.path.join(out, "metrics.jsonl")) as fh:
        for line in fh:
            rec = json.loads(line)
            if "loss_rec" in rec:
                vals.append(float(rec["loss_rec"]))
    return vals


def test_g4c():
    tmp = tempfile.mkdtemp(prefix="g4c_")
    data = os.path.join(tmp, "toy.npz")
    toy(data)
    common = ["--data", data, "--steps", "300", "--batch", "128",
              "--lr", "3e-3", "--d-z", "16", "--patch", "1",
              "--d-model", "32", "--n-layers", "2", "--n-heads", "4",
              "--d-dec", "32", "--anomaly", "--holdout-years", "2006"]
    ratios = {}
    for name, script in (("torch", TRAIN), ("jax", JTRAIN)):
        out = os.path.join(tmp, name)
        p = subprocess.run([sys.executable, "-u", script, "--out", out]
                           + common, capture_output=True, text=True)
        if p.returncode:
            tail = "\n".join((p.stdout + p.stderr).strip().splitlines()[-15:])
            fail(f"G4c: {name} trainer exited {p.returncode}:\n{tail}")
            return
        v = losses(out)
        if len(v) < 40:
            fail(f"G4c: {name} wrote only {len(v)} loss points")
            return
        ratios[name] = float(np.mean(v[:10]) / np.mean(v[-10:]))
        # A LOSS THAT DID NOT MOVE would make the band below a comparison of
        # two nulls, which passes for the wrong reason.
        if ratios[name] < 1.05:
            fail(f"G4c: the {name} trainer's loss_rec barely moved "
                 f"({ratios[name]:.3f}x) — the toy has nothing to learn, or "
                 f"the trainer is not learning it; the band is meaningless "
                 f"either way")
    if FAILURES:
        return
    band = ratios["jax"] / ratios["torch"]
    # A BAND, not an equality: the RNG streams cannot match across frameworks,
    # so the two runs see different batches and different masks. Half to twice
    # the torch fall is the statement "the same objective is being minimised
    # at the same rate"; an implementation that had, say, dropped the mask
    # weighting would sit far outside it.
    if not (0.5 <= band <= 2.0):
        fail(f"G4c: jax loss_rec fell {ratios['jax']:.3f}x against torch's "
             f"{ratios['torch']:.3f}x — ratio {band:.3f} outside [0.5, 2.0]")
    print(f"  G4c short toy training — 300 steps on the same toy tensor: "
          f"loss_rec falls {ratios['torch']:.3f}x under the real ml/train.py "
          f"and {ratios['jax']:.3f}x under ml/jaxport/train_stage1.py, "
          f"ratio {band:.3f} inside the pre-registered band [0.5, 2.0]")


# ---- G4d: round-trip export ------------------------------------------------
def test_g4d():
    worst_rt, worst_fw = 0.0, 0.0
    for label, kw in (("patch=1", dict(n_chan=9, d_model=32, n_heads=4,
                                       n_layers=2, d_z=8, d_dec=16,
                                       dec_layers=3, patch=1)),
                      ("patch=3", dict(n_chan=9, d_model=32, n_heads=4,
                                       n_layers=2, d_z=8, d_dec=16,
                                       dec_layers=2, patch=3)),
                      ("k_time=7", dict(n_chan=9, d_model=32, n_heads=4,
                                        n_layers=2, d_z=8, d_dec=16,
                                        dec_layers=2, patch=1, k_time=7))):
        tm, jmod = make_pair(**kw)
        sd = tm.state_dict()
        back = jc.export_pixelmae(jmod)
        if set(sd) != set(back):
            fail(f"G4d {label}: key sets differ: {sorted(set(sd) ^ set(back))}")
            continue
        for k in sd:
            worst_rt = max(worst_rt, float(np.max(np.abs(
                sd[k].detach().numpy().astype(np.float64)
                - back[k].astype(np.float64)))))

    if worst_rt != 0.0:
        fail(f"G4d: torch→jax→torch is not IDENTICAL (max|Δ| {worst_rt:.3e})")

    # And the other half of the claim: a checkpoint written from the JAX side
    # must ENCODE the same thing under torch. Weights are perturbed first so
    # this is not merely the round trip restated — it is the arithmetic a TPU
    # run's eval would actually perform.
    kw = dict(n_chan=9, d_model=32, n_heads=4, n_layers=2, d_z=8, d_dec=16,
              dec_layers=2, patch=1)
    _, jmod = make_pair(**kw)
    graphdef, state = nnx.split(jmod)
    key = jax.random.PRNGKey(0)
    keys = jax.random.split(key, len(jax.tree_util.tree_leaves(state)))
    it = iter(keys)
    state = jax.tree_util.tree_map(
        lambda v: v + 0.05 * jax.random.normal(next(it), v.shape), state)
    trained = nnx.merge(graphdef, state)

    with tempfile.TemporaryDirectory() as td:
        path = os.path.join(td, "pixelmae.pt")
        jc.export_pt(trained, {"d_z": 8, "patch": 1, "d_model": 32,
                               "n_layers": 2, "n_heads": 4, "d_dec": 16,
                               "dec_layers": 2, "k_time": 1},
                     path=path, chan=[f"c{i}" for i in range(9)])
        ck = torch.load(path, map_location="cpu", weights_only=False)
        from model import codec_from_ckpt
        tm2 = codec_from_ckpt(ck, 9)
        tm2.load_state_dict(ck["model"])
        tm2.eval()

    b = make_batch(6, 9, seed=11)
    with torch.no_grad():
        zt = tm2.encode(b["v"] * (~b["mask"]), b["o"], b["mask"], b["ctx"])
    zj = trained.encode(_j(b["v"]) * ~_j(b["mask"]), _j(b["o"]),
                        _j(b["mask"]), _j(b["ctx"]))
    worst_fw = close("G4d jax-trained → torch encode", zt, zj, 1e-5)
    print(f"  G4d round-trip export — torch→jax→torch state_dicts are "
          f"IDENTICAL at patch 1, patch 3 and k_time 7 (max|Δ| "
          f"{worst_rt:.1e}); a perturbed JAX codec exported to .pt and "
          f"re-encoded under torch matches the JAX encode to {worst_fw:.2e} "
          f"(gate 1e-5)")


# ---- G4e: k_time=7 forward parity + the converter's k_time refusals --------
def test_g4e():
    kw = dict(n_chan=9, d_model=32, n_heads=4, n_layers=2, d_z=8, d_dec=16,
              dec_layers=2, patch=1, k_time=7)
    tm, jmod = make_pair(seed=5, **kw)
    C, KT, B = 9, 7, 6
    b = make_batch(B, C, k_time=KT, seed=13)
    with torch.no_grad():
        z = tm.encode(b["v"] * (~b["mask"]), b["o"], b["mask"], b["ctx"])
    zj = jmod.encode(_j(b["v"]) * ~_j(b["mask"]), _j(b["o"]), _j(b["mask"]),
                     _j(b["ctx"]))
    d1 = close("G4e k_time=7 encode", z, zj, 1e-5)

    g = torch.Generator().manual_seed(14)
    chan = torch.randint(0, C, (B, KT * C), generator=g)
    off = torch.randint(-3, 4, (B, KT * C, 3), generator=g)
    tpos = torch.randint(0, KT, (B, KT * C), generator=g)
    with torch.no_grad():
        y = tm.query(z, chan, off, tpos)
    yj = jmod.query(zj, _j(chan), _j(off), _j(tpos))
    d2 = close("G4e k_time=7 query", y, yj, 1e-5)

    # `tpos` is REQUIRED, in both frameworks, for the same stated reason.
    raised = 0
    for fn in (lambda: tm.query(z, chan, off),
               lambda: jmod.query(zj, _j(chan), _j(off))):
        try:
            fn()
        except ValueError as e:
            if "tpos" in str(e):
                raised += 1
    if raised != 2:
        fail(f"G4e: query() without tpos refused in {raised}/2 frameworks")

    # The converter must refuse ACROSS the k_time boundary in both directions:
    # a per-bin model handed a block checkpoint leaves `time_emb.weight`
    # unconsumed, and a block model handed a per-bin one finds it missing.
    per_bin_kw = dict(kw)
    per_bin_kw.pop("k_time")
    torch.manual_seed(5)
    tm1 = TorchPixelMAE(**per_bin_kw)
    refusals = 0
    for label, sd, mdl in (
            ("per-bin model, block checkpoint", tm.state_dict(),
             jm.PixelMAE(**per_bin_kw, rngs=nnx.Rngs(0))),
            ("block model, per-bin checkpoint", tm1.state_dict(),
             jm.PixelMAE(**kw, rngs=nnx.Rngs(0)))):
        try:
            jc.load_pixelmae(sd, mdl)
        except KeyError as e:
            if "time_emb" in str(e) or "q_time" in str(e):
                refusals += 1
            else:
                fail(f"G4e: {label} refused without naming the k_time keys")
    if refusals != 2:
        fail(f"G4e: the converter crossed the k_time boundary "
             f"({refusals}/2 refusals)")
    print(f"  G4e k_time=7 forward parity — encode over the 7x9 cell grid "
          f"max|Δ| {d1:.2e}, query with tpos max|Δ| {d2:.2e} (gate 1e-5); "
          f"query() without tpos refuses in both frameworks and the converter "
          f"refuses both directions across the k_time boundary")


def main():
    print("tests/test_jaxport_train.py — tier-3 gates G4a-G4e, CPU, fp32\n")
    for fn in (test_g4a, test_g4b, test_g4c, test_g4d, test_g4e):
        fn()
    if FAILURES:
        print("\nFAILED:")
        for f in FAILURES:
            print("  -", f)
        return 1
    print("\ntests/test_jaxport_train.py: all 5 gates passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())

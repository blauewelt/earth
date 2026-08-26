#!/usr/bin/env python3
"""F-gates: `ml/jaxport/field_model.py` and `ml/jaxport/train_field.py` against
the torch E-052 field head (`ml/field_model.py`, `ml/train_field.py`).

    python3 tests/test_jaxport_field.py

Plain python, no pytest — the repo's `test_*` + `__main__` runner convention.
CPU, fp32, eval semantics, no network. The style is
`tests/test_jaxport_parity.py`'s: every gate MEASURES a difference, prints it,
and fails only against a pre-registered tolerance.

**EVERY GATE RUNS TWICE, AND THAT IS THE POINT.** `ml/field_model.py`'s
decision 2 zero-initialises the final layer and every adaLN modulation, so a
FRESH head returns exactly 0.0 and a fresh `diff` head's denoiser is exactly
`c_skip(sigma)*x`. Those are designed-in identities and they are worth
checking — but a parity gate run only on a fresh head is comparing zeros to
zeros and would pass against a backbone that computed nothing at all. So each
gate is run on the fresh head AND on a `trained`-ish one: the same weights
plus small DETERMINISTIC noise on every parameter (one seeded torch
generator), which lights up every path the zero-init switches off. The
reported number is the worse of the two.

  **F1 — the tokenizer and the det forward.** `to_tokens` matches torch
  BITWISE (it is a zero-fill, a copy and a gather, with no arithmetic on the
  path, so anything looser would be hiding something) and
  `to_pixels(to_tokens(z)) == z` bitwise on both sides. Then `forward_det`
  end to end from raw context tokens, gate 1e-5.

  **F2 — D(x, sigma).** The EDM denoiser at six sigmas spanning the sampler's
  own ladder: `4e-3*sigma_data` (the ladder's floor), 0.1, `sigma_data`, 3,
  40, `160*sigma_data` (its ceiling). `c_in`/`c_out`/`c_skip` change by orders
  of magnitude across that span and a wrong one is invisible at a single
  sigma.

  **F3 — `edm_loss_given`.** With (sigma, noise) INJECTED, because an RNG is
  the one thing two frameworks cannot share: `ml/field_model.py` split
  `edm_loss` from `edm_loss_given` for exactly this gate. Relative, 1e-6.

  **F4 — `sample_from`.** A full 8-step Heun integration from an injected
  `x_init` down an injected ladder — 15 denoiser evaluations, each feeding the
  next, so this is where a small per-call difference would compound if there
  were one. Gate 1e-4.

  **F5 — GRADIENTS.** Forward parity says the two models agree; only a
  gradient says the two TRAINERS would agree. Both objectives (det MSE and
  `edm_loss_given`), compared on the global gradient norm and on three named
  per-tensor norms, relative 1e-4. The named tensors are chosen to sit on
  three different paths: the conditioner's packed QKV, a DiT block's adaLN
  modulation, and the sigma embedding.

  **F6 — the export round trip.** torch state_dict -> NNX -> `export_field_pt`
  -> `torch.load` -> `load_state_dict(strict=True)`, and then EVERY tensor
  compared with `torch.equal` — dtype included, which is why `dit.tok_py` is
  emitted as int64 and not through the float32 weight path. Beside it, the
  converter's refusal contract (`ml/jaxport/convert.py`'s discipline): a
  state_dict missing a key and one carrying a key nothing wants must both
  raise and name the offender, because a silent partial load produces
  plausible garbage.

  **F7 — the trainer.** The gauss toy in det mode must beat persistence (its
  conditional mean of the residual is -0.3*x_t, so persistence is NOT optimal
  and a head that learned nothing cannot pass); a resume must be BIT-IDENTICAL
  (train 4, save, train 3 == train 7, every leaf of both the parameter state
  and the optimiser state); and the result file must carry `in_progress` until
  the end and never after it.

  **F8 — `--input-znoise 0` is the unnoised code path.** Bit-identical to the
  flag being absent, guarding against an always-on noise path — and, so the
  gate is not vacuous, a NON-zero sigma must actually move the weights.

  **F9 — `--cond-remat` changes the memory, never the numbers.** Gradient
  rematerialization on the chunked conditioner exists to take ~114–124 GB per
  chip of retained activations off the HBM budget at the E-052.1 config
  (`cond_from_pixels`' docstring carries the arithmetic); a memory optimisation
  that moved a number would be a silent change of experiment. Three parts: the
  LOSS is bitwise identical and one train step's parameter UPDATE is bitwise
  identical, in both objectives (why each is exact rather than merely close is
  stated at the gate — and the raw GRADIENT underneath is NOT bitwise, which is
  measured, localised and explained rather than hidden); with `--cond-chunk`
  unset the flag is inert and provably makes no difference at all; and the
  jaxpr is inspected so the gate cannot pass because the checkpoint silently
  did nothing.
"""
import json
import os
import shutil
import sys
import tempfile
import time
import warnings

import numpy as np
import torch

# torch warns on every norm_first TransformerEncoder that it cannot use the
# nested-tensor fast path. norm_first is the whole point here.
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

from field_model import FieldHead, OceanTokenizer                # noqa: E402
from jaxport import field_model as jf                            # noqa: E402
from jaxport import train_field as jt                            # noqa: E402

FAILURES = []
TIMES = {}

# The geometry every gate runs on. Small enough for a couple of minutes on
# CPU, and NOT trivial: the land holes make one whole patch cell disappear
# from the token list and leave two cells partly land, which is the only
# arrangement in which the ocean-flag channel is doing any work.
H = W = 12
PATCH, D_Z, K = 3, 5, 6
D_MODEL, LAYERS, HEADS, D_COND = 64, 2, 4, 48
SIGMA_DATA = 0.7


def fail(msg):
    FAILURES.append(msg)


def check(name, a, b, tol):
    """Elementwise max |Δ| between a torch tensor and a JAX array."""
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


def rel(name, a, b, tol):
    a, b = float(a), float(b)
    if a == b:
        return 0.0
    r = abs(a - b) / max(abs(a), 1e-30)
    if not (r < tol):
        fail(f"{name}: rel {r:.3e} >= {tol:g} ({a!r} vs {b!r})")
    return r


def _j(t):
    return jnp.asarray(t.detach().cpu().numpy())


def holey_mask():
    """A grid with land: one WHOLE patch cell of it, plus scattered holes.

    Both kinds matter, and for the reason `tests/test_field_diffusion.py`
    gives: a fully-land cell must not become a token at all, and a partly-land
    cell must become a token whose land slots are zero with flag 0 — which is
    what lets the model tell "a zero-valued ocean pixel" from "not ocean".
    """
    mask = np.ones((H, W), bool)
    mask[0:3, 9:12] = False               # the whole (py=0, px=3) cell
    for y, x in [(5, 3), (6, 7), (9, 0), (7, 11), (11, 4)]:
        mask[y, x] = False
    ys, xs = np.where(mask)
    return ys, xs


def build_pair(perturb):
    """(torch FieldHead, its NNX twin). `perturb` adds small DETERMINISTIC
    noise to every torch parameter BEFORE the conversion, so the gates
    exercise real arithmetic instead of the zero-init's exact zeros."""
    ys, xs = holey_mask()
    tok = OceanTokenizer(H, W, ys, xs, PATCH)
    torch.manual_seed(0)
    tm = FieldHead(tok, D_Z, K, mode="diff", d_model=D_MODEL, layers=LAYERS,
                   heads=HEADS, d_cond=D_COND, cond_layers=2, cond_heads=HEADS,
                   sigma_data=SIGMA_DATA)
    if perturb:
        g = torch.Generator().manual_seed(3)
        with torch.no_grad():
            for _, p in tm.named_parameters():
                p.add_(0.05 * torch.randn(p.shape, generator=g))
    tm.eval()
    return tm, jf.field_head_from_torch(tm)


def make_inputs(tok, B=3, seed=11):
    g = torch.Generator().manual_seed(seed)
    ctx = torch.randn(B, K, tok.P, D_Z, generator=g)
    z_t = ctx[:, -1].contiguous()
    z_n = torch.randn(B, tok.P, D_Z, generator=g)
    sea = torch.randn(B, K, 2, generator=g)
    x = torch.randn(B, tok.P, D_Z, generator=g)
    noise = torch.randn(B, tok.P, D_Z, generator=g)
    sigma = torch.exp(0.5 * torch.randn(B, generator=g))
    return ctx, z_t, z_n, sea, x, noise, sigma


VARIANTS = (("fresh (zero-init)", False), ("perturbed", True))


# ---- F1: tokenizer + det forward -------------------------------------------
def test_f1():
    t0 = time.time()
    out = []
    for label, perturb in VARIANTS:
        tm, jm = build_pair(perturb)
        tok, jtok = tm.tok, jm.tok
        ctx, z_t, _, sea, _, _, _ = make_inputs(tok)

        # (a) the token layout, BITWISE. No arithmetic sits on this path.
        tt = tok.to_tokens(ctx)
        jtt = jtok.to_tokens(_j(ctx))
        if not np.array_equal(tt.numpy(), np.asarray(jtt)):
            d = check("F1 to_tokens", tt, jtt, 0.0)
            fail(f"F1 [{label}]: to_tokens is not BITWISE equal (max|Δ| {d:g})")
        # the token ORDER, which depends only on the mask and never on the
        # order ys/xs arrive in
        if not (np.array_equal(tok.tok_py.numpy(), jtok.tok_py)
                and np.array_equal(tok.tok_px.numpy(), jtok.tok_px)
                and tok.ntok == jtok.ntok):
            fail(f"F1 [{label}]: the token (py, px) tables disagree")

        # (b) the round trip, BITWISE on both sides
        if not torch.equal(tok.to_pixels(tt, D_Z), ctx):
            fail(f"F1 [{label}]: the TORCH round trip is not bitwise")
        if not np.array_equal(np.asarray(jtok.to_pixels(jtt, D_Z)),
                              ctx.numpy()):
            fail(f"F1 [{label}]: the JAX round trip is not bitwise")
        # …and the OUTPUT layout (C == d_z, no flag), which is the call the
        # model makes on every forward
        outok = torch.randn(2, tok.ntok, tok.P2 * D_Z,
                            generator=torch.Generator().manual_seed(2))
        check(f"F1 [{label}] to_pixels output layout", tok.to_pixels(outok),
              jtok.to_pixels(_j(outok)), 1e-30)

        # (c) the det forward, end to end from raw context tokens
        with torch.no_grad():
            det = tm(tt, z_t, sea)          # forward() takes TOKENS, not pixels
        jdet = jm(jtt, _j(z_t), _j(sea))
        d = check(f"F1 [{label}] forward_det", det, jdet, 1e-5)
        if not perturb and not torch.equal(det, z_t):
            fail("F1 [fresh]: the torch det head at init is not EXACTLY "
                 "persistence — the zero-init identity is broken")
        out.append((label, d))
    TIMES["F1"] = time.time() - t0
    print("  F1 tokenizer + det forward — to_tokens and both round trips are "
          "BITWISE equal to torch on every variant; forward_det max|Δ| "
          + ", ".join(f"{lb} {d:.2e}" for lb, d in out)
          + " (gate 1e-5). The fresh head is bitwise persistence.")


# ---- F2: D(x, sigma) -------------------------------------------------------
def test_f2():
    t0 = time.time()
    lines = []
    for label, perturb in VARIANTS:
        tm, jm = build_pair(perturb)
        tok = tm.tok
        ctx, z_t, _, sea, x, _, _ = make_inputs(tok)
        tt = tok.to_tokens(ctx)
        with torch.no_grad():
            cond = tm.make_cond(tt, sea)
        jcond = jm.make_cond(jf.OceanTokenizerJax.from_torch(tok).to_tokens(
            _j(ctx)), _j(sea))
        B = x.shape[0]
        worst, worst_s = 0.0, None
        for s in (4e-3 * SIGMA_DATA, 0.1, SIGMA_DATA, 3.0, 40.0,
                  160.0 * SIGMA_DATA):
            sig = torch.full((B,), float(s))
            with torch.no_grad():
                dt = tm.D(x, sig, cond)
            dj = jm.D(_j(x), _j(sig), jcond)
            d = check(f"F2 [{label}] D at sigma={s:g}", dt, dj, 1e-5)
            if d > worst:
                worst, worst_s = d, s
            if not perturb:
                # THE DESIGNED-IN IDENTITY: at init D(x; sigma) == c_skip*x
                # BITWISE, for every sigma.
                c_skip = tm._coefs(sig)[0]
                if not torch.equal(dt, c_skip[:, None, None] * x):
                    fail(f"F2 [fresh]: torch D at sigma={s:g} is not exactly "
                         f"c_skip*x — the zero-init identity is broken")
        lines.append(f"{label} {worst:.2e}"
                     + (f" (worst at sigma {worst_s:g})" if worst_s is not None
                        else " (exact at every sigma)"))
    TIMES["F2"] = time.time() - t0
    print("  F2 EDM denoiser D(x, sigma) over the sampler's whole ladder "
          "[4e-3*sd, 0.1, sd, 3, 40, 160*sd]: max|Δ| " + " · ".join(lines)
          + " (gate 1e-5). The fresh head is bitwise c_skip*x at every sigma.")


# ---- F3: edm_loss_given ----------------------------------------------------
def test_f3():
    t0 = time.time()
    lines = []
    for label, perturb in VARIANTS:
        tm, jm = build_pair(perturb)
        tok = tm.tok
        ctx, z_t, z_n, sea, _, noise, sigma = make_inputs(tok)
        tt = tok.to_tokens(ctx)
        with torch.no_grad():
            cond = tm.make_cond(tt, sea)
            lt = tm.edm_loss_given(cond, z_t, z_n, sigma, noise)
        jcond = jm.make_cond(jm.tok.to_tokens(_j(ctx)), _j(sea))
        lj = jm.edm_loss_given(jcond, _j(z_t), _j(z_n), _j(sigma), _j(noise))
        r = rel(f"F3 [{label}] edm_loss_given", float(lt), float(lj), 1e-6)
        lines.append(f"{label} {float(lt):.6f} vs {float(lj):.6f} (rel "
                     f"{r:.1e})")
    TIMES["F3"] = time.time() - t0
    print("  F3 edm_loss_given with (sigma, noise) INJECTED — the surface that "
          "isolates the arithmetic from two RNGs no framework shares: "
          + " · ".join(lines) + " (gate rel 1e-6).")


# ---- F4: sample_from -------------------------------------------------------
def test_f4():
    t0 = time.time()
    lines = []
    for label, perturb in VARIANTS:
        tm, jm = build_pair(perturb)
        tok = tm.tok
        ctx, z_t, _, sea, _, _, _ = make_inputs(tok)
        with torch.no_grad():
            cond = tm.make_cond(tok.to_tokens(ctx), sea)
        jcond = jm.make_cond(jm.tok.to_tokens(_j(ctx)), _j(sea))

        sig_t = tm.sigma_ladder(8)
        sig_j = jm.sigma_ladder(8)
        dl = check(f"F4 [{label}] sigma_ladder", sig_t, sig_j, 1e-30)
        if float(sig_t[-1]) != 0.0:
            fail("F4: the ladder does not end at an exact 0")
        g = torch.Generator().manual_seed(99)
        x_init = torch.randn(z_t.shape, generator=g) * sig_t[0]
        with torch.no_grad():
            st = tm.sample_from(cond, z_t, x_init, sig_t)
        sj = jm.sample_from(jcond, _j(z_t), _j(x_init), sig_j)
        d = check(f"F4 [{label}] sample_from", st, sj, 1e-4)
        lines.append(f"{label} {d:.2e}")
    TIMES["F4"] = time.time() - t0
    print(f"  F4 sample_from — a full 8-step Heun integration (15 denoiser "
          f"evaluations, each feeding the next) from an INJECTED x_init down "
          f"an injected ladder that the two frameworks build identically "
          f"(max|Δ| {dl:g}): max|Δ| " + " · ".join(lines) + " (gate 1e-4).")


# ---- F5: gradients ---------------------------------------------------------
NAMED = {
    "det": ("cond.proj.weight", "dit.blocks.0.attn.in_proj_weight",
            "dit.fin.weight"),
    "edm": ("cond.enc.layers.0.self_attn.in_proj_weight",
            "dit.blocks.1.ada.1.weight", "sig_mlp.0.weight"),
}


def _torch_grads(tm, which, ctx, z_t, z_n, sea, sigma, noise):
    tm.zero_grad(set_to_none=True)
    cond = tm.make_cond(tm.tok.to_tokens(ctx), sea)
    if which == "det":
        loss = (tm.forward_det(cond, z_t) - z_n).pow(2).mean()
    else:
        loss = tm.edm_loss_given(cond, z_t, z_n, sigma, noise)
    loss.backward()
    # A parameter the objective does not reach has grad None in torch and an
    # exact zero in JAX. They agree; the shapes of the two answers do not, so
    # the None is materialised here rather than special-cased at every use.
    return float(loss.detach()), {
        n: (p.grad.detach().numpy() if p.grad is not None
            else np.zeros(tuple(p.shape), np.float32))
        for n, p in tm.named_parameters()}


def _jax_grads(jm, which, ctx, z_t, z_n, sea, sigma, noise):
    graphdef, state = nnx.split(jm)
    args = tuple(_j(t) for t in (ctx, z_t, z_n, sea, sigma, noise))

    def lf(st):
        m = nnx.merge(graphdef, st)
        cond = m.make_cond(m.tok.to_tokens(args[0]), args[3])
        if which == "det":
            return ((m.forward_det(cond, args[1]) - args[2]) ** 2).mean()
        return m.edm_loss_given(cond, args[1], args[2], args[4], args[5])

    loss, grads = jax.value_and_grad(lf)(state)
    # `export_field_head` over the GRADIENT tree gives torch-named arrays:
    # the grads share the state's structure, so merging them into a
    # model-shaped object and emitting it is exactly the name map the loader
    # already owns — one list, read in both directions, and no second
    # transcription of 68 keys to go stale.
    named = jf.export_field_head(nnx.merge(graphdef, grads))
    return float(loss), float(optax.global_norm(grads)), named


BUFFER_KEYS = ("sigma_data", "fourier_f", "dit.tok_py", "dit.tok_px")


def test_f5():
    t0 = time.time()
    lines = []
    for label, perturb in VARIANTS:
        tm, jm = build_pair(perturb)
        tm.train()
        ctx, z_t, z_n, sea, _, noise, sigma = make_inputs(tm.tok)
        for which in ("det", "edm"):
            lt, tg = _torch_grads(tm, which, ctx, z_t, z_n, sea, sigma, noise)
            lj, gnj, jg = _jax_grads(jm, which, ctx, z_t, z_n, sea, sigma,
                                     noise)
            rel(f"F5 [{label}/{which}] loss", lt, lj, 1e-5)
            gnt = float(np.sqrt(sum(float((v ** 2).sum())
                                    for v in tg.values())))
            rg = rel(f"F5 [{label}/{which}] global grad norm", gnt, gnj, 1e-4)
            # every key, not only the three named ones — the named ones are
            # what the read-out quotes, the full sweep is what makes a silent
            # mis-mapping impossible
            worst, worst_k = 0.0, None
            for k, v in jg.items():
                if k in BUFFER_KEYS:
                    continue
                if k not in tg:
                    fail(f"F5: exported gradient key {k!r} is not a torch "
                         f"parameter")
                    continue
                a = float(np.linalg.norm(tg[k]))
                b = float(np.linalg.norm(v))
                r = rel(f"F5 [{label}/{which}] {k}", a, b, 1e-4)
                if r > worst:
                    worst, worst_k = r, k
            missing = set(tg) - set(jg)
            if missing:
                fail(f"F5: {len(missing)} torch parameters have no exported "
                     f"gradient: {sorted(missing)[:4]}")
            named = " · ".join(
                f"{k} {float(np.linalg.norm(tg[k])):.5f}/"
                f"{float(np.linalg.norm(jg[k])):.5f}"
                for k in NAMED[which])
            lines.append(f"[{label}/{which}] |g| {gnt:.6f} vs {gnj:.6f} "
                         f"(rel {rg:.1e}); {named}; worst per-tensor rel "
                         f"{worst:.1e} over {len(tg)} tensors"
                         + (f" ({worst_k})" if worst_k else ""))
    TIMES["F5"] = time.time() - t0
    print("  F5 gradient parity — forward parity says the models agree, only "
          "a gradient says the TRAINERS would:\n       "
          + "\n       ".join(lines) + "\n     (gate rel 1e-4 on the global "
          "norm and on every per-tensor norm).")


# ---- F6: the export round trip, and the refusal contract -------------------
def test_f6():
    t0 = time.time()
    tmp = tempfile.mkdtemp(prefix="f6_")
    try:
        tm, jm = build_pair(True)
        sd0 = tm.state_dict()
        pt = os.path.join(tmp, "field.pt")
        args = {"mode": "diff", "K": K, "patch": PATCH, "d_model": D_MODEL,
                "layers": LAYERS, "heads": HEADS, "d_cond": D_COND,
                "cond_layers": 2, "cond_heads": HEADS, "d_z": D_Z,
                "sigma_data": SIGMA_DATA, "H": H, "W": W}
        blob = jf.export_field_pt(jm, args, path=pt, step=7)
        for k in ("model", "args", "step"):
            if k not in blob:
                fail(f"F6: the exported blob has no {k!r} key")
        for k in ("opt", "gen"):
            if k in blob:
                fail(f"F6: the exported blob carries {k!r}. optax state is not "
                     f"torch AdamW state and a JAX key is not a "
                     f"torch.Generator; a blob that claims otherwise would be "
                     f"resumed as a continuation.")
        if blob["args"].get("backend") != "jax":
            fail("F6: args['backend'] is not 'jax' — ml/CLAUDE.md §3b makes a "
                 "JAX-trained number a new tier and this is the one line a "
                 "reader sees")
        if os.path.exists(pt + ".tmp"):
            fail("F6: the atomic .pt write left its temp sibling behind")

        tk = torch.load(pt, map_location="cpu", weights_only=False)
        ys, xs = holey_mask()
        tok2 = OceanTokenizer(H, W, ys, xs, PATCH)
        tm2 = FieldHead(tok2, D_Z, K, mode="diff", d_model=D_MODEL,
                        layers=LAYERS, heads=HEADS, d_cond=D_COND,
                        cond_layers=2, cond_heads=HEADS,
                        sigma_data=SIGMA_DATA)
        tm2.load_state_dict(tk["model"])          # strict=True by default
        tm2.eval()
        sd1 = tm2.state_dict()
        if set(sd0) != set(sd1):
            fail(f"F6: key sets differ — only in original "
                 f"{sorted(set(sd0) - set(sd1))}, only in round trip "
                 f"{sorted(set(sd1) - set(sd0))}")
        bad = [k for k in sd0 if not torch.equal(sd0[k], sd1[k])]
        if bad:
            fail(f"F6: {len(bad)} tensors are not torch.equal after the round "
                 f"trip: {bad[:5]}")
        dtypes = [k for k in sd0 if sd0[k].dtype != sd1[k].dtype]
        if dtypes:
            fail(f"F6: dtype changed on {dtypes} — an int64 index table "
                 f"emitted through the float32 weight path would still load "
                 f"and would still index")
        # ASSERT THE EFFECT: equal weights must mean an equal forward.
        ctx, z_t, _, sea, _, _, _ = make_inputs(tok2)
        tt2 = tok2.to_tokens(ctx)
        with torch.no_grad():
            d_fw = check("F6 round-trip forward", tm(tt2, z_t, sea),
                         tm2(tt2, z_t, sea).numpy(), 1e-30)

        # --- the refusal contract (ml/jaxport/convert.py's discipline) ------
        def expect_raise(name, fn, must_mention):
            try:
                fn()
            except (KeyError, ValueError) as e:
                msg = str(e)
                for m in must_mention:
                    if m not in msg:
                        fail(f"F6 {name}: refusal did not name {m!r}: "
                             f"{msg[:200]}")
                return
            fail(f"F6 {name}: the loader did NOT refuse")

        def fresh():
            return jf.FieldHeadJax(jf.OceanTokenizerJax(H, W, ys, xs, PATCH),
                                   D_Z, K, mode="diff", d_model=D_MODEL,
                                   layers=LAYERS, heads=HEADS, d_cond=D_COND,
                                   cond_layers=2, cond_heads=HEADS,
                                   sigma_data=SIGMA_DATA, rngs=nnx.Rngs(0))

        short = dict(sd0)
        dropped = "dit.blocks.0.ada.1.weight"
        del short[dropped]
        expect_raise("missing key", lambda: jf.load_field_head(short, fresh()),
                     [dropped, "missing"])
        fat = dict(sd0)
        fat["dit.blocks.0.bogus"] = torch.zeros(3)
        expect_raise("unexpected key",
                     lambda: jf.load_field_head(fat, fresh()),
                     ["dit.blocks.0.bogus", "unconsumed"])
        wrongmask = dict(sd0)
        wrongmask["dit.tok_py"] = wrongmask["dit.tok_py"] + 1
        expect_raise("wrong ocean mask",
                     lambda: jf.load_field_head(wrongmask, fresh()),
                     ["tok_py", "different ocean mask"])
        jf.load_field_head(dict(sd0), fresh())     # the happy path still loads
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    TIMES["F6"] = time.time() - t0
    print(f"  F6 export round trip — a perturbed JAX head exported with "
          f"export_field_pt loads into the torch FieldHead with strict=True "
          f"and ALL {len(sd0)} tensors are torch.equal (dtypes included; "
          f"forward Δ {d_fw:g}); no opt/gen keys ride along and args['backend'] "
          f"says 'jax'. The loader refuses a missing key, an unconsumed key "
          f"and a state_dict from a different ocean mask, naming the offender "
          f"each time.")


# ---- F7: the trainer -------------------------------------------------------
TOY = ["--toy", "gauss", "--K", "4", "--patch", "2", "--d-model", "32",
       "--layers", "2", "--heads", "2", "--d-cond", "32", "--cond-layers", "1",
       "--cond-heads", "2", "--batch", "8", "--quiet", "--prefetch", "0"]


def test_f7():
    t0 = time.time()
    tmp = tempfile.mkdtemp(prefix="f7_")
    try:
        # (a) det on the gauss toy must BEAT PERSISTENCE. The law's conditional
        # mean of the residual is -0.3*x_t, so persistence is not optimal and a
        # head that learned nothing cannot pass; the analytic optimum is
        # sigma_e^2 / (0.09*Var(x) + sigma_e^2) = 0.850, so the bar is set
        # loosely below it rather than at a number nobody derived.
        r = jt.main(TOY + ["--mode", "det", "--steps", "200", "--lr", "1e-3",
                           "--eval-every", "100", "--cond-chunk", "8",
                           "--out", os.path.join(tmp, "det.json")])
        ratio = r["final"]["ratio"]
        if not (ratio < 0.95):
            fail(f"F7(a): the det head reads ratio {ratio:.4f} on the gauss "
                 f"toy after 200 steps; persistence is 1.0 and the analytic "
                 f"optimum is 0.850, so anything at or above 0.95 means it "
                 f"learned nothing")

        # (b) RESUME IS BIT-IDENTICAL: train 4, save, train 3 == train 7.
        base = TOY + ["--mode", "det", "--eval-every", "0", "--cond-chunk", "0"]
        dA, dB = os.path.join(tmp, "rA"), os.path.join(tmp, "rB")
        jt.main(base + ["--steps", "4", "--ckpt-dir", dA, "--ckpt-every", "4",
                        "--out", os.path.join(tmp, "a1.json")])
        rA = jt.main(base + ["--steps", "7", "--ckpt-dir", dA, "--resume",
                             "--ckpt-every", "7",
                             "--out", os.path.join(tmp, "a2.json")])
        rB = jt.main(base + ["--steps", "7", "--ckpt-dir", dB, "--ckpt-every",
                             "7", "--out", os.path.join(tmp, "b.json")])
        for what in ("state", "opt_state"):
            la = jax.tree_util.tree_leaves(rA[what])
            lb = jax.tree_util.tree_leaves(rB[what])
            if len(la) != len(lb):
                fail(f"F7(b): {what} leaf counts differ ({len(la)} vs "
                     f"{len(lb)})")
                continue
            bad = [i for i, (x, y) in enumerate(zip(la, lb))
                   if not np.array_equal(np.asarray(x), np.asarray(y))]
            if bad:
                d = max(float(np.max(np.abs(np.asarray(la[i])
                                            - np.asarray(lb[i]))))
                        for i in bad)
                fail(f"F7(b): resume is NOT bit-identical — {len(bad)} of "
                     f"{len(la)} {what} leaves differ, max|Δ| {d:.3e}")
        n_leaves = len(jax.tree_util.tree_leaves(rA["state"]))

        # (c) result-file discipline: in_progress until the end, then gone,
        # atomic, and no NaN anywhere.
        p = os.path.join(tmp, "w.json")
        jt.write_result(p, {"k": 1}, [{"step": 1}], final=None,
                        in_progress={"step": 1, "of": 9})
        mid = json.load(open(p))
        if "in_progress" not in mid or mid["final"] is not None:
            fail("F7(c): a mid-run write must carry in_progress and a null "
                 "final — a reader cannot otherwise tell it is partial")
        if list(mid)[0] != "in_progress":
            fail("F7(c): in_progress is not the FIRST key — a human opening "
                 "the file must meet the marker before any number")
        if os.path.exists(p + ".tmp"):
            fail("F7(c): the atomic write left its temp sibling behind")
        jt.write_result(p, {"k": 1}, [{"step": 9}], final={"done": True})
        if "in_progress" in json.load(open(p)):
            fail("F7(c): the completed write still carries in_progress")
        try:
            jt._finite_or_die("a deliberately non-finite value", float("nan"))
            fail("F7(c): _finite_or_die accepted a NaN")
        except SystemExit:
            pass

        # …and the finished run's own file, from (a)
        done = json.load(open(os.path.join(tmp, "det.json")))
        if "in_progress" in done:
            fail("F7(c): the completed run's result file still carries "
                 "in_progress — that key's absence is the only completion "
                 "certificate")
        if not done["history"] or done["final"] is None:
            fail("F7(c): the completed run wrote no history or no final")

        # (d) the smoke path, in diff mode, end to end
        sm = jt.main(["--toy", "gauss", "--mode", "diff", "--smoke", "--quiet",
                      "--out", os.path.join(tmp, "smoke.json")])
        if "smoke_sample_shape" not in sm["final"]:
            fail("F7(d): --smoke did not run its sample() assertion")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    TIMES["F7"] = time.time() - t0
    print(f"  F7 the JAX trainer — det on the gauss toy reads ratio "
          f"{ratio:.4f} after 200 steps against persistence 1.0 and the "
          f"analytic optimum 0.850; resume is BIT-IDENTICAL (train 4 + 3 == "
          f"train 7, all {n_leaves} parameter leaves and every optimiser leaf "
          f"exactly equal); the result file carries in_progress mid-run and "
          f"not at the end, writes atomically, and refuses a NaN; --smoke "
          f"asserts a finite sample() end to end "
          f"({sm['final']['smoke_nfe_spent']} NFE spent).")


# ---- F8: --input-znoise 0 is the unnoised code path ------------------------
def test_f8():
    t0 = time.time()
    tmp = tempfile.mkdtemp(prefix="f8_")
    try:
        base = TOY + ["--mode", "det", "--steps", "6", "--eval-every", "0",
                      "--cond-chunk", "0"]
        off = jt.main(base + ["--out", os.path.join(tmp, "off.json")])
        zero = jt.main(base + ["--input-znoise", "0",
                               "--out", os.path.join(tmp, "zero.json")])
        on = jt.main(base + ["--input-znoise", "0.5",
                             "--out", os.path.join(tmp, "on.json")])
        lo = jax.tree_util.tree_leaves(off["state"])
        lz = jax.tree_util.tree_leaves(zero["state"])
        ln = jax.tree_util.tree_leaves(on["state"])
        bad = [i for i, (x, y) in enumerate(zip(lo, lz))
               if not np.array_equal(np.asarray(x), np.asarray(y))]
        if bad:
            d = max(float(np.max(np.abs(np.asarray(lo[i]) - np.asarray(lz[i]))))
                    for i in bad)
            fail(f"F8: --input-znoise 0 is NOT bit-identical to the flag being "
                 f"absent — {len(bad)} leaves differ, max|Δ| {d:.3e}. That is "
                 f"an always-on noise path.")
        # NOT VACUOUS: a non-zero sigma must actually move the weights, or the
        # check above would pass against a noise path that does nothing.
        moved = max(float(np.max(np.abs(np.asarray(x) - np.asarray(y))))
                    for x, y in zip(lo, ln))
        if not (moved > 1e-6):
            fail(f"F8: --input-znoise 0.5 moved the weights by only "
                 f"{moved:.3e} — the bit-identity check is vacuous")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    TIMES["F8"] = time.time() - t0
    print(f"  F8 --input-znoise — 0 is BIT-IDENTICAL to the flag being absent "
          f"over every parameter leaf (a separate jitted step, not a "
          f"sigma*0 branch), and 0.5 moves the weights by {moved:.3g}, so the "
          f"identity is not vacuous.")


# ---- F9: --cond-remat changes the memory, never the numbers ----------------
def _ulps(a, b):
    """(max |Δ|, max ulps of each tensor's OWN scale) over two named dicts.

    Per TENSOR, not per element: a gradient tensor legitimately contains
    near-zero entries whose `spacing()` is denormal, and dividing an absolute
    difference by that reports 1e7 ulps for a difference of 7e-9 — a statistic
    about float's exponent range rather than about the arithmetic. The scale
    that matters is the one the tensor is actually carrying.
    """
    worst_d, worst_u, worst_k = 0.0, 0.0, None
    for k in a:
        if k in BUFFER_KEYS:
            continue
        x = np.asarray(a[k], np.float64)
        y = np.asarray(b[k], np.float64)
        d = float(np.max(np.abs(x - y)))
        worst_d = max(worst_d, d)
        sc = float(max(np.abs(x).max(), np.abs(y).max()))
        u = d / float(np.spacing(np.float32(sc))) if sc > 0 else 0.0
        if u > worst_u:
            worst_u, worst_k = u, k
    return worst_d, worst_u, worst_k


REMAT_TOY = ["--toy", "gauss", "--K", "4", "--patch", "2", "--d-model", "32",
             "--layers", "2", "--heads", "2", "--d-cond", "32",
             "--cond-layers", "1", "--cond-heads", "2", "--batch", "8",
             "--quiet", "--prefetch", "0", "--steps", "1"]


def test_f9():
    t0 = time.time()
    tmp = tempfile.mkdtemp(prefix="f9_")
    try:
        _, jm = build_pair(True)
        graphdef, state = nnx.split(jm)
        ctx, z_t, z_n, sea, _, noise, sigma = make_inputs(jm.tok)
        args = tuple(_j(t) for t in (ctx, z_t, z_n, sea, sigma, noise))
        CH = 4                                     # ntok is 15 → 4 chunks

        def loss_fn(which, remat):
            def f(st):
                m = nnx.merge(graphdef, st)
                c = jt.cond_from_pixels(m, args[0], args[3], CH, remat)
                if which == "det":
                    return ((m.forward_det(c, args[1]) - args[2]) ** 2).mean()
                return m.edm_loss_given(c, args[1], args[2], args[4], args[5])
            return f

        lines = []
        for which in ("det", "edm"):
            out = {}
            for remat in (True, False):
                lv, g = jax.jit(jax.value_and_grad(loss_fn(which, remat)))(state)
                out[remat] = (np.float32(lv),
                              jf.export_field_head(nnx.merge(graphdef, g)))
            # THE FORWARD IS THE SAME JAXPR EITHER WAY, so the loss is exact.
            if out[True][0].tobytes() != out[False][0].tobytes():
                fail(f"F9 [{which}]: --cond-remat changed the LOSS "
                     f"({out[True][0]!r} vs {out[False][0]!r}). remat "
                     f"recomputes the forward for the BACKWARD pass; the "
                     f"forward value itself must be untouched.")
            d, u, k = _ulps(out[True][1], out[False][1])
            # THE GRADIENT IS NOT BITWISE, AND CANNOT BE. `jax.checkpoint`
            # inserts an optimisation barrier and hands XLA a differently
            # fused graph, so the recomputed activations the backward reads
            # are the same expression evaluated in a different accumulation
            # order. The difference is bounded at a few ulp of each tensor's
            # own scale and — the part that makes it a diagnosis rather than a
            # tolerance — it appears ONLY on the conditioner's parameters,
            # which are exactly what was rematerialized.
            # The gate is 8 ulp against a measured worst of 4 — one doubling
            # of headroom, because a threshold set AT the measurement is a
            # tripwire (ml/CLAUDE.md §4.9) and this one would fire on a
            # different XLA version rather than on a real change.
            if not (u <= 8.0):
                fail(f"F9 [{which}] gradient: {u:.1f} ulp on {k!r} (max|Δ| "
                     f"{d:.3e}); remat must not move a gradient by more than "
                     f"a few ulp of the tensor's scale")
            moved = sorted(kk for kk in out[True][1]
                           if kk not in BUFFER_KEYS
                           and not np.array_equal(out[True][1][kk],
                                                  out[False][1][kk]))
            stray = [kk for kk in moved if not kk.startswith("cond.")]
            if stray:
                fail(f"F9 [{which}]: remat on the CONDITIONER moved gradients "
                     f"outside it: {stray[:5]}. Only cond.* is inside the "
                     f"rematerialized chunk function.")
            gn = float(np.sqrt(sum(float((np.asarray(v, np.float64) ** 2).sum())
                                   for kk, v in out[False][1].items()
                                   if kk not in BUFFER_KEYS)))
            lines.append(f"{which}: loss bitwise · grad max|Δ| {d:.2e} "
                         f"({d / gn:.1e} of |g| {gn:.4f}), {u:.0f} ulp on "
                         f"{k}, {len(moved)} tensors moved and all of them "
                         f"cond.*")

        # NOT VACUOUS: prove the checkpoint is actually in the graph. A gate
        # comparing two identical graphs would pass forever.
        jx = {r: str(jax.make_jaxpr(jax.grad(loss_fn("det", r)))(state))
              for r in (True, False)}
        if "remat" not in jx[True]:
            fail("F9: --cond-remat produced a jaxpr with no remat in it — the "
                 "checkpoint did nothing and every equality above is vacuous")
        if "remat" in jx[False]:
            fail("F9: --no-cond-remat produced a jaxpr containing remat")

        # (ii) THE TRAINER, end to end: one train step's parameter update.
        step_lines = []
        for mode in ("det", "diff"):
            base = REMAT_TOY + ["--mode", mode, "--cond-chunk", "5",
                                "--eval-every", "0"]
            on = jt.main(base + ["--cond-remat",
                                 "--out", os.path.join(tmp, f"{mode}_on.json")])
            off = jt.main(base + ["--no-cond-remat",
                                  "--out", os.path.join(tmp, f"{mode}_off.json")])
            if on["config"]["cond_remat"] is not True or \
                    off["config"]["cond_remat"] is not False:
                fail(f"F9 [{mode}]: the flags did not reach the config record")
            a_on = jf.export_field_head(nnx.merge(on["graphdef"], on["state"]))
            a_off = jf.export_field_head(nnx.merge(off["graphdef"],
                                                   off["state"]))
            d, u, k = _ulps(a_on, a_off)
            # BITWISE, and the reason is arithmetic rather than luck: AdamW's
            # FIRST update is -lr * m̂/(√v̂ + eps) with m̂ = g and v̂ = g², i.e.
            # -lr * g/(|g| + eps) — a few-ulp change in g does not survive it.
            # The gate is held at 4 ulp anyway rather than at 0, because the
            # property that must hold is "remat does not change the training
            # trajectory", and pinning an exactness that is downstream of an
            # optimiser's algebra would fail the day the optimiser changed.
            if not (u <= 4.0):
                fail(f"F9 [{mode}] one train step: {u:.1f} ulp on {k!r} "
                     f"(max|Δ| {d:.3e})")
            step_lines.append(
                f"{mode} {'BITWISE' if d == 0.0 else f'{u:.0f} ulp'}")

        # (iii) NO BEHAVIOUR CHANGE WHEN --cond-chunk IS UNSET.
        nb = REMAT_TOY + ["--mode", "det", "--cond-chunk", "0",
                          "--eval-every", "0"]
        d0 = jt.main(nb + ["--out", os.path.join(tmp, "nc_default.json")])
        if d0["config"]["cond_remat"] is not False:
            fail("F9: with --cond-chunk 0 the default must resolve to remat "
                 "OFF — there is no chunk scan to stack residuals in, and "
                 "wrapping the single conditioner call would trade memory the "
                 "config did not ask to trade")
        n_on = jt.main(nb + ["--cond-remat",
                             "--out", os.path.join(tmp, "nc_on.json")])
        for lab, r in (("default", d0), ("--cond-remat", n_on)):
            bad = [i for i, (x, y) in enumerate(
                zip(jax.tree_util.tree_leaves(r["state"]),
                    jax.tree_util.tree_leaves(d0["state"])))
                if not np.array_equal(np.asarray(x), np.asarray(y))]
            if bad:
                fail(f"F9: with --cond-chunk 0, {lab} changed {len(bad)} "
                     f"parameter leaves; the flag must be inert there")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    TIMES["F9"] = time.time() - t0
    print("  F9 --cond-remat — memory only, never numbers:\n       "
          + "\n       ".join(lines)
          + f"\n       one train step, --cond-remat vs --no-cond-remat: "
          + " · ".join(step_lines)
          + " (gate 4 ulp)\n       jaxpr carries remat with the flag on and "
          "not with it off, so the equalities are not vacuous; with "
          "--cond-chunk 0 the default resolves OFF and the flag is inert "
          "(every leaf bitwise).")


def main():
    print("tests/test_jaxport_field.py — E-052 field head: torch vs "
          "ml/jaxport, CPU, fp32\n")
    for fn in (test_f1, test_f2, test_f3, test_f4, test_f5, test_f6, test_f7,
               test_f8, test_f9):
        fn()
    print("\n  timings: "
          + " · ".join(f"{k} {v:.1f}s" for k, v in TIMES.items())
          + f" · total {sum(TIMES.values()):.1f}s")
    if FAILURES:
        print("\nFAILED:")
        for f in FAILURES:
            print("  -", f)
        return 1
    print("\ntests/test_jaxport_field.py: all 9 gates passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())

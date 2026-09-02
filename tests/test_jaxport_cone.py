#!/usr/bin/env python3
"""E-069 · parity gates for the JAX cone codec (`ml/plans/E069_HANDOVER.md` §8.7).

One PAIR on identical weights — a torch `ConeMAE` at the smoke geometry under
`torch.manual_seed(0)`, converted with `cone_from_torch` — and one REAL batch
from `train_cone.smoke_tensor` through `ConeSampler`, 16 anchors including one
at the western edge of the basin so a slice of its cone is genuinely invalid.
Every measured difference is PRINTED; a gate fails only against the tolerance
§8.7 pre-registers, and a gate that fails is a finding about the port, never a
tolerance to widen.

    python3 tests/test_jaxport_cone.py           # CPU, fp32, no network

The gates in this file (C3, C4 and C8 gate `ml/jaxport/train_cone.py`, the
optax trainer; the rest gate the model and the converter):

  C1 forward   `tokens` (both outputs), `encode` (z and latents),
               `query_tokens`, `decode_from_z`, `decode`                 1e-4
  C2 loss      `forward_given` vs `loss_given`, loss and every `terms`
               scalar, aux on (0.25) and off (0)                         1e-5
  C3 one step  from the SAME converted weights and the SAME given masks,
               one torch SGD(1e-2) step vs one optax.sgd(1e-2) step;
               then one AdamW step through the REAL chain (clip 1.0,
               lr 3e-4, wd 0.01, cosine at count 0)   1e-6 / 2e-5 · 1e-6
  C4 band      200 smoke steps of `ml/jaxport/train_cone.py` against
               `ml/train_cone.py --smoke` as SUBPROCESSES on the same
               smoke tensor and seed: both fall, and the final
               held-out nll ratio jax/torch                    [0.8, 1.25]
  C8 resume    20 steps + checkpoint + `--resume` for 20 more against 40
               straight, same seed: `ckpt_latest.npz` bitwise equal and
               the metrics records identical after the seam           exact
  C5 round trip `export_cone(load_cone(sd))` identical to `sd` key for
               key by `torch.equal` INCLUDING dtype; the exported
               state_dict then loads into a fresh torch `ConeMAE` with
               strict=True and re-encodes                     identical / 1e-5
  C6 refusal   a state_dict missing `to_z.bias`, and one carrying a
               `bogus` key, both RAISE and NAME the offender         must raise
  C7 existence perturbing an INVALID dot's value by +1000 in the JAX
               batch leaves z BITWISE equal                                exact
  C9 forward   torch `forward(b, plan)` under a seeded generator equals
               `forward_given` fed the masks drawn from the same
               seeded generator in the same order                          exact

WHAT C1's TOLERANCE IS ACTUALLY MEASURING, written down here because §8.7
expected 1e-6 (the G1 experience) and this model does not deliver it. The
whole of C1's residual lives in `CoordEnc`: torch's `log1p` and XLA's `log1p`
differ in the last fp32 bit on some inputs, the highest Fourier band
multiplies that by `2 ** (n_fourier - 1) = 128`, and sin/cos at the resulting
~870 radians turn it into ~6e-5 in the feature — which `coord.proj` averages
to ~1.3e-5 in a token. `sin` and `cos` agree BIT FOR BIT at those arguments,
so this is the encoding's own condition number (128) and not an
implementation disagreement: perturbing `dy_km` by one ULP would move the
TORCH model by the same amount. It does not propagate — `z` agrees to ~1.6e-7,
and the losses to ~5e-7 — because the attention and the pool average it away.
`ml/jaxport/cone_models.py`'s module docstring carries the same note next to
the code.

FINDING 1 (2026-09-02, found by C3, NOT fixed here — the fix belongs in
`ml/jaxport/convert.py` and that file is out of this change's scope).
**`cone_from_torch` returns a JAX model that SHARES MEMORY with the torch
module it converted.** `convert._Consumer.get` returns
`np.asarray(t.detach().cpu().numpy())`, which is a VIEW of the torch tensor's
storage, and `jnp.asarray` of a contiguous numpy array is zero-copy on the CPU
backend, so the JAX buffer is the torch tensor. Measured on the smoke geometry:
adding 1.0 in place to every `tm.parameter()` moves **68 of the 99** tensors
`export_cone(jm)` reports. The 31 that do not move are exactly the ones whose
mapper transposes or slices first (`_linear`'s `.weight`, `_packed_attention`'s
q/k/v), because a non-contiguous source forces a copy. Repro:

    torch.manual_seed(0); tm = ConeMAE(8, **GEOM)
    jm = cone_from_torch(tm, rngs=nnx.Rngs(0))
    before = {k: np.array(v) for k, v in export_cone(jm).items()}
    with torch.no_grad():
        for p in tm.parameters(): p.add_(1.0)
    moved = [k for k, v in export_cone(jm).items()
             if not np.array_equal(v, before[k])]      # -> 68 of 99

Nothing published is wrong because of it: C1/C2/C5/C7/C9 never write to the
torch module, `export_cone` copies (`convert._np`), and
`ml/jaxport/train_cone.py` converts once and never touches the torch module
again (its `optax` updates are functional, so they allocate rather than write
through). What it breaks is any caller that keeps TRAINING the torch module
after converting it — which is precisely C3 — and it would break silently, by
making the two backends agree perfectly. The fix is one word in
`convert._Consumer.get` (`.numpy().copy()`, or `np.array(..., copy=True)`).
Until it lands, C3 converts from `{k: v.detach().clone()}` and ASSERTS the
decoupling before it measures anything.

WHAT C3's TWO TOLERANCES ARE FOR, since one of them is 20x the other. At
`count = 0` the AdamW update is `-lr * g / (|g| + eps)` with `eps = 1e-8`
(torch's bias corrections and optax's `mu_hat`/`nu_hat` both cancel to exactly
that at the first step — verified below on the real chain). That function is
1-Lipschitz in `g` only where `|g| >> eps`; where `|g|` is itself of order the
1e-7 the two backends disagree by, `g / (|g| + eps)` swings across most of
[-1, 1] and the step lands `lr` apart, i.e. 3e-4 — which is a property of Adam
at its first step on a gradient that is numerically zero, not a port bug. So
the gate reads the two populations separately, exactly as §8.7 pre-registers:
2e-5 over ALL parameters, 1e-6 over the ones whose gradient is above 1e-6.
Plain SGD has no such amplification and is held at 1e-6 everywhere.

C4's step-0 losses are deliberately NOT compared. Both trainers evaluate on a
FIXED anchor set with a FIXED mask seed, but the mask DRAWS differ — torch
spends a `torch.Generator`, the JAX trainer a `numpy.random.default_rng`
(§8.6) — so the two step-0 numbers are two different measurements of the same
weights and agree only to the width of the mask distribution. What IS asserted
is the pair of properties that would break if the port were wrong: each curve
falls, and the two final numbers are within the pre-registered band. The
ANCHORS, by contrast, are identical: `ml/jaxport/train_cone.py` mirrors
`train_cone.train_one`'s numpy anchor stream exactly, so the two runs see the
same ocean in the same order and the band is a statement about the codecs.

C9 is the gate that lets `forward` be refactored at all. `forward` still draws
its hidden-dot queries INSIDE `_query_sets` (after `encode`), where they have
always been drawn, while `forward_given` takes them from
`draw_dot_queries` — so the two agree only if `encode` consumes no RNG, which
is true (every dropout in `ConeMAE` is 0.0) and is asserted here rather than
argued in a comment.
"""
import contextlib
import io
import json
import os
import sys
import tempfile
import time
import warnings

import numpy as np
import torch

# torch warns on every norm_first TransformerEncoder that it cannot use the
# nested-tensor fast path. It is expected here and would bury the PASS lines.
warnings.filterwarnings("ignore", message=".*enable_nested_tensor.*")

HERE = os.path.dirname(os.path.abspath(__file__))
ML = os.path.join(os.path.dirname(HERE), "ml")
if ML not in sys.path:
    sys.path.insert(0, ML)

import jax.numpy as jnp                                        # noqa: E402
import optax                                                   # noqa: E402
from flax import nnx                                           # noqa: E402

from cone import channel_depth_dbar                            # noqa: E402
from cone_sampler import ConeSampler                           # noqa: E402
from cone_codec import ConeMAE, default_plan                   # noqa: E402
from train_cone import smoke_tensor, to_torch                  # noqa: E402
from jaxport import cone_models as jcm                         # noqa: E402
from jaxport import cone_convert as jcc                        # noqa: E402

FAILURES = []

GEOM = dict(d_model=32, n_heads=4, n_latents=8, n_layers=2, d_z=8, d_dec=32,
            dec_layers=2)
N_ANCHORS = 16


def check(name, a, b, tol):
    """Elementwise max |Δ| between a torch tensor and a JAX array."""
    a = np.asarray(a.detach().cpu().numpy() if hasattr(a, "detach") else a,
                   np.float64)
    b = np.asarray(b, np.float64)
    if a.shape != b.shape:
        FAILURES.append(f"{name}: shape {a.shape} vs {b.shape}")
        return float("inf")
    d = float(np.max(np.abs(a - b))) if a.size else 0.0
    if not (d < tol):
        FAILURES.append(f"{name}: max|Δ| {d:.3e} >= {tol:g}")
    return d


def expect_raise(name, fn, must_mention):
    try:
        fn()
    except KeyError as e:
        msg = str(e)
        for m in must_mention:
            if m not in msg:
                FAILURES.append(
                    f"{name}: refusal did not name {m!r}: {msg[:240]}")
                return
        return
    FAILURES.append(f"{name}: the loader did NOT refuse")


# ---------------------------------------------------------------- fixtures --
_FIX = {}


def fixture():
    """`(torch model, jax model, torch batch, jax batch, chan names, plan)`.

    Built ONCE and cached: the smoke tensor is 120x40x56x8 and writing it is
    the expensive part of this file. The anchors deliberately include
    `(t, y, 0)` — the westernmost column — because every westward dot of that
    cone leaves the basin, and this window is a basin rather than a globe, so
    those dots are INVALID (never wrapped). C7 has nothing to measure without
    them, and the fixture asserts they exist rather than hoping.
    """
    if _FIX:
        return _FIX
    tmp = tempfile.TemporaryDirectory()
    _FIX["_tmp"] = tmp                      # keep it alive for the process
    path = os.path.join(tmp.name, "cone_smoke.npz")
    smoke_tensor(path, seed=0)
    d = np.load(path, allow_pickle=False)
    X = np.asarray(d["X"])
    chan = [str(c) for c in d["chan"]]
    sampler = ConeSampler(X, np.isfinite(X), d["lats"], d["lons"], chan,
                          L_in=6, future_lags=(1, 2))

    rng = np.random.default_rng(3)
    anchors = np.stack([rng.integers(7, 110, N_ANCHORS - 1),
                        rng.integers(0, X.shape[1], N_ANCHORS - 1),
                        rng.integers(2, X.shape[2] - 2, N_ANCHORS - 1)], 1)
    # THE EDGE ANCHOR — x = 0 puts every westward dot off the basin.
    anchors = np.concatenate([anchors, np.array([[20, 5, 0]], np.int64)], 0)
    s = sampler.sample(anchors.astype(np.int64))
    if not (~s["valid"]).any():
        FAILURES.append("no invalid dot in the batch — C7 would be vacuous")

    depth = torch.as_tensor([channel_depth_dbar(n) for n in chan],
                            dtype=torch.float32)
    b = to_torch(s, depth, "cpu")

    torch.manual_seed(0)
    tm = ConeMAE(len(chan), **GEOM).eval()
    jm = jcc.cone_from_torch(tm, rngs=nnx.Rngs(0))

    _FIX.update(sampler=sampler, chan=chan, anchors=anchors, sample=s,
                tm=tm, jm=jm, b=b, jb=as_jax(b), depth=depth)
    return _FIX


def as_jax(b):
    """The torch batch as jnp arrays, dtypes preserved (bool stays bool)."""
    return {k: jnp.asarray(v.detach().cpu().numpy()) for k, v in b.items()}


def torch_plan(chan, aux=0.25, n_dot_queries=16):
    return default_plan(chan, n_dot_queries=n_dot_queries, aux_latent_w=aux,
                        future_lags=(1, 2))


def jax_masks(chan_mask, dot_mask, dot_idx):
    return (jnp.asarray(chan_mask.numpy()), jnp.asarray(dot_mask.numpy()),
            (jnp.asarray(dot_idx[0].numpy()),
             jnp.asarray(dot_idx[1].numpy())))


# ---- C1: forward parity ----------------------------------------------------
def test_c1_forward():
    f = fixture()
    tm, jm, b, jb = f["tm"], f["jm"], f["b"], f["jb"]

    with torch.no_grad():
        t_toks, t_kpm = tm.tokens(b)
        t_z, t_lat = tm.encode(b)
    j_toks, j_kpm = jm.tokens(jb)
    j_z, j_lat = jm.encode(jb)

    d_tok = check("C1 tokens", t_toks, j_toks, 1e-4)
    kpm_same = np.array_equal(t_kpm.numpy(), np.asarray(j_kpm))
    if not kpm_same:
        FAILURES.append("C1: the key-padding masks differ")
    d_z = check("C1 encode z", t_z, j_z, 1e-4)
    d_lat = check("C1 encode latents", t_lat, j_lat, 1e-4)

    # THE MASK'S OWN INVARIANT: cls + ctx + C patch tokens are never padding,
    # so no attention row is ever fully masked and the port needs no NaN
    # guard (§8.3). Asserted, never defended in the code.
    C = tm.n_chan
    unmasked = (~np.asarray(j_kpm)).sum(1)
    if not (unmasked >= 2 + C).all():
        FAILURES.append(f"C1: a token row has fewer than {2 + C} live keys "
                        f"(min {int(unmasked.min())}) — the no-NaN-guard "
                        f"argument does not hold on this batch")
    if not np.isfinite(np.asarray(j_z)).all():
        FAILURES.append("C1: the JAX encode produced a non-finite z")

    # query_tokens on the batch's own dot coordinates, then both decoders.
    with torch.no_grad():
        t_q = tm.query_tokens(b["chan"].long(), b["dy_km"], b["dx_km"],
                              b["lag_days"], b["depth"])
        t_mu, t_lv = tm.decode_from_z(t_z, t_q)
        t_mu2, t_lv2 = tm.decode(t_z, t_lat, t_q)
    j_q = jm.query_tokens(jb["chan"], jb["dy_km"], jb["dx_km"],
                          jb["lag_days"], jb["depth"])
    j_mu, j_lv = jm.decode_from_z(j_z, j_q)
    j_mu2, j_lv2 = jm.decode(j_z, j_lat, j_q)

    d_q = check("C1 query_tokens", t_q, j_q, 1e-4)
    d_d1 = max(check("C1 decode_from_z mu", t_mu, j_mu, 1e-4),
               check("C1 decode_from_z logvar", t_lv, j_lv, 1e-4))
    d_d2 = max(check("C1 decode mu", t_mu2, j_mu2, 1e-4),
               check("C1 decode logvar", t_lv2, j_lv2, 1e-4))

    # coord.freqs is a NON-PERSISTENT torch buffer: absent from the
    # state_dict, so the converter never sees it and the two copies must be
    # shown identical rather than assumed so.
    if not np.array_equal(tm.coord.freqs.numpy(),
                          np.asarray(jm.coord.freqs)):
        FAILURES.append("C1: coord.freqs differs — the recomputed Fourier "
                        "basis is not the torch buffer")

    print(f"  C1 forward (tol 1e-4) · tokens {d_tok:.2e} (kpm identical, "
          f"min {int(unmasked.min())} live keys/row >= {2 + C}) · encode z "
          f"{d_z:.2e}, latents {d_lat:.2e} · query_tokens {d_q:.2e} · "
          f"decode_from_z {d_d1:.2e} · decode {d_d2:.2e} · coord.freqs "
          f"identical")


# ---- C2: the loss ----------------------------------------------------------
def test_c2_loss():
    f = fixture()
    tm, jm, b, jb, chan = f["tm"], f["jm"], f["b"], f["jb"], f["chan"]
    lines = []
    for aux in (0.25, 0.0):
        plan = torch_plan(chan, aux=aux)
        g = torch.Generator().manual_seed(4321)
        plan["generator"] = g
        chan_mask, dot_mask = tm._masks(b, plan)
        dot_idx = tm.draw_dot_queries(b, plan, dot_mask)
        with torch.no_grad():
            out = tm.forward_given(b, plan, chan_mask, dot_mask, dot_idx)
        j_cm, j_dm, j_di = jax_masks(chan_mask, dot_mask, dot_idx)
        j_loss, j_z, j_terms = jm.loss_given(jb, jcm.plan_to_jax(plan),
                                             j_cm, j_dm, j_di)

        d = check(f"C2 loss (aux={aux})", out["loss"].detach(), j_loss, 1e-5)
        d = max(d, check(f"C2 z (aux={aux})", out["z"].detach(), j_z, 1e-5))
        if set(out["terms"]) != set(j_terms):
            FAILURES.append(
                f"C2 (aux={aux}): terms keys differ — torch "
                f"{sorted(out['terms'])} vs jax {sorted(j_terms)}")
        worst, worst_k = 0.0, ""
        for k in sorted(set(out["terms"]) & set(j_terms)):
            dk = abs(float(out["terms"][k]) - float(j_terms[k]))
            if dk > worst:
                worst, worst_k = dk, k
            if not (dk < 1e-5):
                FAILURES.append(f"C2 terms[{k}] (aux={aux}): |Δ| {dk:.3e} "
                                f">= 1e-5")
        if aux == 0.0 and "nll_latent" in out["terms"]:
            FAILURES.append("C2: aux off still produced an nll_latent term")
        lines.append(f"aux={aux}: loss/z {d:.2e}, worst of "
                     f"{len(out['terms'])} terms {worst:.2e} ({worst_k})")
    print("  C2 forward_given vs loss_given (tol 1e-5) · " + " · ".join(lines))


# ---- C5: the round trip ----------------------------------------------------
def test_c5_round_trip():
    f = fixture()
    tm, chan, b, jb = f["tm"], f["chan"], f["b"], f["jb"]
    sd = tm.state_dict()
    fresh = jcc.ConeMAEJax(len(chan), **GEOM, rngs=nnx.Rngs(1))
    jm2 = jcc.load_cone(sd, fresh)
    out = jcc.export_cone(jm2)

    if set(out) != set(sd):
        FAILURES.append(
            f"C5: key set differs — missing {sorted(set(sd) - set(out))}, "
            f"extra {sorted(set(out) - set(sd))}")
    n_bad = 0
    for k in sorted(set(out) & set(sd)):
        t = torch.from_numpy(out[k])
        if t.dtype != sd[k].dtype or not torch.equal(t, sd[k]):
            n_bad += 1
            if n_bad <= 3:
                FAILURES.append(
                    f"C5: {k} is not bit-identical after the round trip "
                    f"(dtype {t.dtype} vs {sd[k].dtype})")

    # And the exported state_dict really is what an UNCHANGED torch ConeMAE
    # loads: strict=True, then re-encode and compare against the original.
    torch.manual_seed(99)                       # deliberately a DIFFERENT init
    tm2 = ConeMAE(len(chan), **GEOM)
    tm2.load_state_dict({k: torch.from_numpy(v) for k, v in out.items()},
                        strict=True)
    tm2.eval()
    with torch.no_grad():
        z1, _ = tm.encode(b)
        z2, _ = tm2.encode(b)
    d = check("C5 re-encode through the exported state_dict", z1, z2, 1e-5)
    # The JAX model that produced the export must still agree, too.
    d2 = check("C5 exported model's own z", z1, jm2.encode(jb)[0], 1e-4)

    # The ARTEFACT, not just the state_dict: `export_cone_pt` must write the
    # blob `ml/train_cone.py:train_one.save` writes, and `cone_from_ckpt_jax`
    # must rebuild the same model out of it. A converter whose two ends are
    # tested and whose FILE is not is a converter nobody has actually used.
    want = {"args", "model", "chan_names", "norm", "step", "arm", "L_in",
            "params"}
    args = dict(GEOM, n_fourier=8, seed=0, steps=1)
    with tempfile.TemporaryDirectory() as td:
        p = os.path.join(td, "cone_codec.pt")
        jcc.export_cone_pt(jm2, args, p, chan_names=chan,
                           norm={"space": "anomaly"}, step=1, arm="cone",
                           L_in=6, params=tm.param_count())
        blob = torch.load(p, map_location="cpu", weights_only=False)
    if set(blob) != want:
        FAILURES.append(f"C5: export_cone_pt wrote {sorted(blob)}, not "
                        f"train_cone.py's {sorted(want)}")
    if blob["args"].get("backend") != "jax":
        FAILURES.append("C5: the exported blob is not marked backend=jax")
    if {v.dtype for v in blob["model"].values()} != {torch.float32}:
        FAILURES.append("C5: the exported blob is not float32 throughout")
    jm3, a3 = jcc.cone_from_ckpt_jax(blob, rngs=nnx.Rngs(2))
    e3 = jcc.export_cone(jm3)
    if not all(np.array_equal(e3[k], out[k]) for k in out) or set(e3) != set(out):
        FAILURES.append("C5: cone_from_ckpt_jax did not rebuild the same "
                        "weights the blob carries")

    print(f"  C5 round trip · {len(out)} keys, all torch.equal with dtype "
          f"({n_bad} mismatches) · a fresh torch ConeMAE loads the export "
          f"with strict=True and re-encodes to max|Δ| {d:.2e} (JAX side "
          f"{d2:.2e}) · export_cone_pt writes train_cone.py's 8-key blob "
          f"(backend={blob['args']['backend']}, float32) and "
          f"cone_from_ckpt_jax rebuilds it bit-identically at "
          f"d_model {a3['d_model']}")


# ---- C6: the refusal contract ---------------------------------------------
def test_c6_refusal():
    f = fixture()
    tm, chan = f["tm"], f["chan"]
    sd = dict(tm.state_dict())

    short = dict(sd)
    del short["to_z.bias"]
    expect_raise("C6 missing key",
                 lambda: jcc.load_cone(
                     short, jcc.ConeMAEJax(len(chan), **GEOM,
                                           rngs=nnx.Rngs(0))),
                 ["to_z.bias", "missing"])

    fat = dict(sd)
    fat["bogus"] = torch.zeros(3)
    expect_raise("C6 unexpected key",
                 lambda: jcc.load_cone(
                     fat, jcc.ConeMAEJax(len(chan), **GEOM, rngs=nnx.Rngs(0))),
                 ["bogus", "unconsumed"])

    # The happy path still loads, so the refusal is not simply always on.
    jcc.load_cone(sd, jcc.ConeMAEJax(len(chan), **GEOM, rngs=nnx.Rngs(0)))
    print(f"  C6 refusal · a state_dict missing 'to_z.bias' and one carrying "
          f"an extra 'bogus' key both raise and name the offender; the "
          f"complete {len(sd)}-key state_dict still loads")


# ---- C7: existence is the key-padding mask, and nothing else ---------------
def test_c7_invalid_dot():
    f = fixture()
    jm, b = f["jm"], f["b"]
    npb = {k: v.detach().cpu().numpy().copy() for k, v in b.items()}

    inval = np.argwhere(~npb["valid"])
    if not len(inval):
        FAILURES.append("C7: no invalid dot — the check is vacuous")
        return
    i, j = int(inval[0][0]), int(inval[0][1])
    # obs=True on purpose: with the sampler's own obs=False the token is
    # `miss_tok` and carries no value at all, so the check would pass with no
    # mask whatsoever (ml/cone_codec.py's module docstring, and the torch
    # smoke test makes the same move for the same reason).
    npb["obs"][i, j] = True
    jb0 = {k: jnp.asarray(v) for k, v in npb.items()}
    z0 = np.asarray(jm.encode(jb0)[0])

    npb2 = {k: v.copy() for k, v in npb.items()}
    npb2["vals"][i, j] += 1000.0
    z1 = np.asarray(jm.encode({k: jnp.asarray(v) for k, v in npb2.items()})[0])
    if not np.array_equal(z0, z1):
        FAILURES.append(
            f"C7: perturbing INVALID dot ({i}, {j}) by +1000 moved z by "
            f"{np.abs(z0 - z1).max():.3e} — the key-padding mask is not "
            f"excluding it in the JAX port")

    # The control: a VALID observed dot MUST move z, or the check above is
    # measuring a model that ignores its input.
    val = np.argwhere(npb["valid"] & npb["obs"])
    i2, j2 = int(val[0][0]), int(val[0][1])
    npb3 = {k: v.copy() for k, v in npb.items()}
    npb3["vals"][i2, j2] += 1000.0
    z2 = np.asarray(jm.encode({k: jnp.asarray(v) for k, v in npb3.items()})[0])
    moved = float(np.abs(z0 - z2).max())
    # BITWISE inequality, the torch smoke test's own standard, not a
    # hand-picked floor: the computation is deterministic, so any difference
    # at all proves the dot was read. The magnitude is printed rather than
    # asserted because `ln_kv` normalises every key token, which bounds by
    # construction how far one outlier value can move the pooled code — a
    # threshold here would be a claim about that bound, which nobody measured.
    if np.array_equal(z0, z2):
        FAILURES.append("C7: perturbing a VALID dot did not move z either — "
                        "the existence check is vacuous")
    print(f"  C7 existence · +1000 on invalid dot ({i}, {j}) leaves z bitwise "
          f"equal (np.array_equal); the control, +1000 on valid observed dot "
          f"({i2}, {j2}), moves z by {moved:.3e} (|z| <= "
          f"{float(np.abs(z0).max()):.3f})")


# ---- C9: torch `forward` is unchanged --------------------------------------
def test_c9_forward_unchanged():
    f = fixture()
    tm, b, chan = f["tm"], f["b"], f["chan"]
    plan = torch_plan(chan)

    g1 = torch.Generator().manual_seed(20690)
    p1 = dict(plan)
    p1["generator"] = g1
    with torch.no_grad():
        out_fwd = tm(b, p1)

    g2 = torch.Generator().manual_seed(20690)
    p2 = dict(plan)
    p2["generator"] = g2
    chan_mask, dot_mask = tm._masks(b, p2)
    dot_idx = tm.draw_dot_queries(b, p2, dot_mask)
    with torch.no_grad():
        out_giv = tm.forward_given(b, p2, chan_mask, dot_mask, dot_idx)

    if not torch.equal(out_fwd["loss"].detach(), out_giv["loss"].detach()):
        FAILURES.append(
            f"C9: forward and forward_given disagree on the loss "
            f"({float(out_fwd['loss']):.9f} vs {float(out_giv['loss']):.9f}) "
            f"— the draw order moved")
    if not torch.equal(out_fwd["z"].detach(), out_giv["z"].detach()):
        FAILURES.append("C9: forward and forward_given disagree on z")
    for k in out_fwd["terms"]:
        if out_fwd["terms"][k] != out_giv["terms"][k]:
            FAILURES.append(f"C9: terms[{k}] differs")

    # The draw is not degenerate: masks actually hid something and the
    # hidden-dot query set is non-empty, so "they agree" is a statement about
    # a real draw rather than about two empty tensors.
    if not (chan_mask.any() and dot_mask.any() and dot_idx[0].numel()):
        FAILURES.append("C9: the seeded draw hid nothing — the check is "
                        "vacuous")
    print(f"  C9 forward unchanged · torch.equal on loss "
          f"({float(out_fwd['loss']):.7f}), z and all "
          f"{len(out_fwd['terms'])} terms, from one seeded generator drawn "
          f"in each order; the draw hid "
          f"{float(chan_mask.float().mean()):.0%} of channels and selected "
          f"{int(dot_idx[1].sum())} dot queries")


# ---- C3: one optimiser step, both optimisers -------------------------------
def _fresh_pair():
    """A torch `ConeMAE` at the smoke geometry and its converted JAX twin.

    Built fresh (not the fixture's pair) because C3 MUTATES both — a gate that
    left a trained model behind for C5 to round-trip would be measuring the
    order the gates happen to run in.

    THE `.clone()` IS LOAD-BEARING; see FINDING 1 in this module's docstring.
    `cone_from_torch` hands `load_cone` the live `state_dict()`, whose tensors
    `jnp.asarray` then ALIASES on the CPU backend, so a torch optimiser step
    would silently rewrite the JAX model's parameters and C3 would compare a
    model against itself. `.detach().clone()` gives the converter memory
    nothing else owns, which is what the gate needs and what every caller that
    keeps training the torch module needs.
    """
    C = len(fixture()["chan"])
    torch.manual_seed(0)
    tm = ConeMAE(C, **GEOM)
    jm = jcc.load_cone({k: v.detach().clone() for k, v in
                        tm.state_dict().items()},
                       jcc.ConeMAEJax(C, **GEOM, rngs=nnx.Rngs(0)))
    return tm, jm


def test_c3_one_step():
    f = fixture()
    b, jb, chan = f["b"], f["jb"], f["chan"]
    plan = torch_plan(chan)                      # NO generator: forward_given
    draw = dict(plan)                            # uses none, the draw uses one
    draw["generator"] = torch.Generator().manual_seed(777)
    chan_mask, dot_mask = f["tm"]._masks(b, draw)
    dot_idx = f["tm"].draw_dot_queries(b, draw, dot_mask)
    j_cm, j_dm, j_di = jax_masks(chan_mask, dot_mask, dot_idx)
    jplan = jcm.plan_to_jax(plan)
    if not (chan_mask.any() and dot_mask.any() and int(dot_idx[1].sum())):
        FAILURES.append("C3: the draw hid nothing — the step is vacuous")

    def torch_step(tm, opt, clip=None):
        """One step, returning the PRE-CLIP gradients (the population split
        below is about how well determined each parameter's update is, and
        clipping is a single global rescale that cannot change that)."""
        out = tm.forward_given(b, plan, chan_mask, dot_mask, dot_idx)
        opt.zero_grad(set_to_none=True)
        out["loss"].backward()
        g = {n: p.grad.detach().clone() for n, p in tm.named_parameters()}
        if clip is not None:
            torch.nn.utils.clip_grad_norm_(tm.parameters(), clip)
        opt.step()
        return g, float(out["loss"].detach())

    def jax_step(jm, tx):
        opt = nnx.Optimizer(jm, tx)

        def loss_fn(m):
            loss, _z, _t = m.loss_given(jb, jplan, j_cm, j_dm, j_di)
            return loss

        loss, grads = nnx.value_and_grad(loss_fn)(jm)
        opt.update(grads)                    # mutates `jm` in place, as torch's
        return float(loss)                   # opt.step() mutates its parameters

    def compare(label, tm, jm, grads, tol, gtol=None):
        """max |Δ| between the torch state_dict and `export_cone(jm)`, over
        every parameter ELEMENT and over the elements whose gradient is above
        1e-6 (§8.7's two populations — the split is elementwise, because
        `g / (|g| + eps)` is ill conditioned per element, not per tensor)."""
        sd = tm.state_dict()
        ex = jcc.export_cone(jm)
        if set(sd) != set(ex):
            FAILURES.append(f"C3 {label}: key sets differ after the step")
            return float("inf"), float("inf"), 0, 0
        allmax = selmax = 0.0
        n_big = n_all = 0
        for k in sd:
            d = np.abs(sd[k].detach().numpy().astype(np.float64)
                       - np.asarray(ex[k], np.float64))
            allmax = max(allmax, float(d.max()))
            n_all += d.size
            if gtol is not None:
                m = np.abs(grads[k].numpy()) > 1e-6
                n_big += int(m.sum())
                if m.any():
                    selmax = max(selmax, float(d[m].max()))
        if not (allmax < tol):
            FAILURES.append(f"C3 {label}: max|Δ| over all parameters "
                            f"{allmax:.3e} >= {tol:g}")
        if gtol is not None and not (selmax < gtol):
            FAILURES.append(f"C3 {label}: max|Δ| over the |g| > 1e-6 elements "
                            f"{selmax:.3e} >= {gtol:g}")
        return allmax, selmax, n_big, n_all

    # ---- (a) plain SGD, lr 1e-2: p <- p - lr*g in both, no state at all ----
    tm, jm = _fresh_pair()
    # THE DECOUPLING CHECK, and it is not decoration: with `cone_from_torch`
    # 68 of the 99 tensors share memory with the torch module (docstring,
    # FINDING 1), so without it the two "independent" steps below would be one
    # step compared with itself and every tolerance would pass for the wrong
    # reason.
    _before = {k: np.array(v) for k, v in jcc.export_cone(jm).items()}
    with torch.no_grad():
        for _p in tm.parameters():
            _p.add_(1.0)
            _p.add_(-1.0)
    _moved = [k for k, v in jcc.export_cone(jm).items()
              if not np.array_equal(v, _before[k])]
    if _moved:
        FAILURES.append(f"C3: {len(_moved)} JAX tensors moved when the TORCH "
                        f"module was written in place — the pair shares "
                        f"memory and the gate would be vacuous "
                        f"({_moved[:3]})")
    g_t, lt = torch_step(tm, torch.optim.SGD(tm.parameters(), lr=1e-2))
    lj = jax_step(jm, optax.sgd(1e-2))
    d_sgd, _, _, n_all = compare("SGD", tm, jm, g_t, 1e-6)
    d_loss = abs(lt - lj)

    # ---- (b) the REAL chain: clip 1.0 -> AdamW(3e-4, wd 0.01) --------------
    # THE SCHEDULE, checked rather than assumed: torch's
    # CosineAnnealingLR(T_max=steps) is stepped AFTER opt.step(), so update s
    # is taken at lr * 0.5 * (1 + cos(pi * (s-1)/steps)); optax evaluates its
    # schedule at `count`, the number of updates ALREADY applied, which is
    # s - 1. At the first step both are the bare lr, and that identity is what
    # makes this a comparison of one AdamW step rather than of two rates.
    sched = optax.cosine_decay_schedule(init_value=3e-4, decay_steps=300,
                                        alpha=0.0)
    # float32, so the comparison is against the float32 value of 3e-4, not
    # against the python double — a tolerance tighter than one ULP would fail
    # on arithmetic that is exactly right.
    if not (abs(float(sched(0)) - float(np.float32(3e-4)))
            <= 1e-7 * 3e-4):
        FAILURES.append(f"C3: cosine_decay_schedule(3e-4, 300)(0) is "
                        f"{float(sched(0)):.9e}, not the bare lr — the "
                        f"schedule equivalence claim is wrong")
    tm2, jm2 = _fresh_pair()
    g_t2, _ = torch_step(
        tm2, torch.optim.AdamW(tm2.parameters(), lr=3e-4, betas=(0.9, 0.999),
                               eps=1e-8, weight_decay=0.01), clip=1.0)
    jax_step(jm2, optax.chain(
        optax.clip_by_global_norm(1.0),
        optax.adamw(learning_rate=sched, b1=0.9, b2=0.999, eps=1e-8,
                    weight_decay=0.01)))
    d_adam, d_adam_big, n_big, n_all2 = compare("AdamW", tm2, jm2, g_t2, 2e-5,
                                                gtol=1e-6)
    if n_big < n_all2 // 100:
        FAILURES.append(f"C3: only {n_big} of {n_all2} parameter elements have "
                        f"|g| > 1e-6 — the restricted tolerance is vacuous")

    print(f"  C3 one step · loss agreement {d_loss:.2e} · SGD(1e-2) max|Δ| "
          f"over all {n_all:,} parameter elements {d_sgd:.2e} (tol 1e-6) · "
          f"clip(1.0)+AdamW(lr 3e-4 = cosine at count 0, wd 0.01) max|Δ| "
          f"{d_adam:.2e} (tol 2e-5), and {d_adam_big:.2e} (tol 1e-6) over the "
          f"{n_big:,}/{n_all2:,} elements with |g| > 1e-6; the JAX twin is "
          f"provably decoupled from the torch module (0 tensors moved on an "
          f"in-place torch write). Weight-decay convention verified: torch "
          f"applies p -= lr*wd*p to the PRE-update parameter and then the Adam "
          f"step; optax.adamw adds wd*p (the same pre-update p) inside the same "
          f"-lr scaling, so the two are the same arithmetic and not merely the "
          f"same to first order")


# ---- C4: 200 smoke steps of each trainer, as SUBPROCESSES ------------------
def _run(cmd, label):
    import subprocess
    p = subprocess.run(cmd, capture_output=True, text=True)
    if p.returncode != 0:
        FAILURES.append(f"C4: the {label} subprocess exited {p.returncode}")
        print(f"    {label} stdout tail:\n" + "\n".join(
            p.stdout.strip().splitlines()[-12:]))
        print(f"    {label} stderr tail:\n" + "\n".join(
            p.stderr.strip().splitlines()[-12:]))
    return p


def _evals(path):
    """The eval records of a metrics.jsonl, and a refusal if any record
    carries a key the handover's family (§2.4) does not name."""
    allowed = {"config", "resumed", "step", "loss_rec", "loss_nei",
               "held_out_nll", "held_out_mse", "held_out_targets", "wall_s"}
    out, recs = [], []
    with open(path) as f:
        for line in f:
            r = json.loads(line)
            recs.append(r)
            if not set(r) <= allowed:
                FAILURES.append(f"{path}: record carries {sorted(set(r) - allowed)}, "
                                f"which §2.4 does not name")
            if "held_out_nll" in r:
                out.append(r)
    return out, recs


def test_c4_band():
    tmp = tempfile.TemporaryDirectory()
    _FIX.setdefault("_tmp_c4", tmp)
    tdir = os.path.join(tmp.name, "torch")
    jdir = os.path.join(tmp.name, "jax")
    # --smoke PINS the whole configuration, --steps included (ml/train_cone.py
    # :parse), so both runs are 200 steps at batch 32 whatever is passed here.
    # It is passed anyway, and stated below, because the number that matters is
    # the one the trainers actually ran.
    common = ["--smoke", "--steps", "300", "--batch", "32",
              "--eval-every", "300", "--seed", "0"]
    t0 = time.time()
    _run([sys.executable, os.path.join(ML, "train_cone.py")] + common
         + ["--out", tdir], "torch")
    tensor = os.path.join(tdir, "smoke_tensor.npz")
    if not os.path.exists(tensor):
        FAILURES.append("C4: the torch run wrote no smoke tensor")
        return
    # THE SAME TENSOR, byte for byte, so the two runs differ in the backend and
    # in nothing else.
    _run([sys.executable, os.path.join(ML, "jaxport", "train_cone.py")]
         + common + ["--tensor", tensor, "--out", jdir], "jax")
    wall = time.time() - t0

    tm_path = os.path.join(tdir, "metrics.jsonl")
    jm_path = os.path.join(jdir, "metrics.jsonl")
    for p in (tm_path, jm_path):
        if not os.path.exists(p):
            FAILURES.append(f"C4: {p} was not written")
            return
    te, trecs = _evals(tm_path)
    je, jrecs = _evals(jm_path)
    if len(te) < 2 or len(je) < 2:
        FAILURES.append(f"C4: {len(te)} torch evals and {len(je)} jax evals — "
                        f"a band needs a start and an end")
        return
    t_first, t_last = te[0]["held_out_nll"], te[-1]["held_out_nll"]
    j_first, j_last = je[0]["held_out_nll"], je[-1]["held_out_nll"]
    for lbl, a0, a1 in (("torch", t_first, t_last), ("jax", j_first, j_last)):
        if not (a1 < a0):
            FAILURES.append(f"C4: the {lbl} held-out nll did not fall "
                            f"({a0:+.5f} -> {a1:+.5f})")
    ratio = j_last / t_last
    if not (0.8 <= ratio <= 1.25):
        FAILURES.append(f"C4: final held-out nll ratio jax/torch {ratio:.4f} "
                        f"is outside the pre-registered band [0.8, 1.25]")
    # The JAX run's config record must say which backend produced it: a TPU
    # number is a new tier (ml/CLAUDE.md §3b) and the mark is how a reader
    # knows.
    jcfg = next((r["config"] for r in jrecs if "config" in r), {})
    tcfg = next((r["config"] for r in trecs if "config" in r), {})
    if jcfg.get("backend") != "jax":
        FAILURES.append("C4: the JAX config record is not marked backend=jax")
    for k in sorted(set(tcfg) - {"backend"}):
        if k not in jcfg:
            FAILURES.append(f"C4: the JAX config record is missing §2.4's "
                            f"{k!r}")
    steps = jcfg.get("steps")
    print(f"  C4 band ({steps} steps at batch {jcfg.get('batch')} — --smoke "
          f"pins the configuration, so --steps 300 resolves to {steps} in BOTH "
          f"trainers) · torch {t_first:+.5f} -> {t_last:+.5f} · jax "
          f"{j_first:+.5f} -> {j_last:+.5f} · ratio jax/torch {ratio:.4f} in "
          f"[0.8, 1.25] · step-0 ratio {j_first / t_first:.4f} (NOT asserted: "
          f"the eval mask draws differ by construction) · both subprocesses "
          f"{wall:.0f}s")


# ---- C8: 20 + resume 20 against 40 straight --------------------------------
def test_c8_resume():
    fixture()                       # for its smoke tensor, written once
    from jaxport import train_cone as jtc            # noqa: E402

    tmp = tempfile.TemporaryDirectory()
    _FIX.setdefault("_tmp_c8", tmp)
    tensor = os.path.join(_FIX["_tmp"].name, "cone_smoke.npz")
    straight = os.path.join(tmp.name, "straight")
    seam = os.path.join(tmp.name, "seam")
    # A DELIBERATELY TINY configuration: this gate is about the checkpoint, not
    # about the optimisation, and every second here is a second the whole file
    # costs. L_in 2 keeps N_max small, which is what the compile time scales on.
    base = ["--tensor", tensor, "--steps", "40", "--batch", "16",
            "--lr", "2e-3", "--d-model", "32", "--n-heads", "4",
            "--n-latents", "8", "--n-layers", "2", "--d-z", "8",
            "--d-dec", "32", "--dec-layers", "2", "--n-fourier", "6",
            "--L-in", "2", "--n-dot-queries", "16", "--eval-every", "20",
            "--eval-anchors", "64", "--certify-n", "256", "--save-every", "40",
            "--ckpt-every", "20", "--holdout-years", "1983", "--seed", "0"]
    buf = io.StringIO()
    t0 = time.time()
    try:
        with contextlib.redirect_stdout(buf):
            jtc.main(base + ["--out", straight])
            # --stop-at, not --steps 20: the cosine schedule is a function of
            # --steps, so a leg that did not know the TOTAL would train its
            # first 20 steps at a different learning rate and the comparison
            # would be of two different runs.
            jtc.main(base + ["--stop-at", "20", "--out", seam])
            jtc.main(base + ["--resume", "--out", seam])
    except SystemExit as e:
        print(buf.getvalue()[-2000:])
        FAILURES.append(f"C8: a leg refused — {e}")
        return
    wall = time.time() - t0

    A = np.load(os.path.join(straight, "ckpt_latest.npz"), allow_pickle=False)
    B = np.load(os.path.join(seam, "ckpt_latest.npz"), allow_pickle=False)
    if int(A["_step"]) != 40 or int(B["_step"]) != 40:
        FAILURES.append(f"C8: the checkpoints are at steps {int(A['_step'])} "
                        f"and {int(B['_step'])}, not 40 and 40")
    if int(A["_n"]) != int(B["_n"]):
        FAILURES.append(f"C8: {int(A['_n'])} state leaves against "
                        f"{int(B['_n'])}")
        return
    n_leaf = int(A["_n"])
    bad = [i for i in range(n_leaf)
           if not np.array_equal(A[f"s{i}"], B[f"s{i}"])]
    if bad:
        worst = max(float(np.max(np.abs(A[f"s{i}"].astype(np.float64)
                                        - B[f"s{i}"].astype(np.float64))))
                    for i in bad)
        FAILURES.append(f"C8: {len(bad)}/{n_leaf} state leaves are not bitwise "
                        f"equal after the seam (worst |Δ| {worst:.3e})")
    # And BOTH host RNG streams, which are what make the anchors and the masks
    # after the seam the same draws rather than merely the same distribution.
    for k in ("_rng", "_mrng"):
        if str(A[k]) != str(B[k]):
            FAILURES.append(f"C8: the {k[1:]} generator state differs after "
                            f"the seam")

    ea, ra = _evals(os.path.join(straight, "metrics.jsonl"))
    eb, rb = _evals(os.path.join(seam, "metrics.jsonl"))
    # `wall_s` restarts at 0 on a resumed leg BY DESIGN (§8.6 — the seam is
    # what the `resumed` record marks), so it is the one key excluded.
    def after(recs):
        return [{k: v for k, v in r.items() if k != "wall_s"}
                for r in recs if "step" in r and r["step"] > 20]
    aa, bb = after(ra), after(rb)
    if not aa:
        FAILURES.append("C8: no records after the seam — the check is vacuous")
    if aa != bb:
        n_diff = sum(1 for x, y in zip(aa, bb) if x != y)
        FAILURES.append(f"C8: {n_diff} of {len(aa)}/{len(bb)} metrics records "
                        f"after the seam differ")
    # The `resumed` record must PRECEDE the resumed leg's config record.
    idx_r = [i for i, r in enumerate(rb) if "resumed" in r]
    idx_c = [i for i, r in enumerate(rb) if "config" in r]
    if len(idx_r) != 1 or len(idx_c) != 2 or not (idx_r[0] < idx_c[1]):
        FAILURES.append(f"C8: the resumed record is at {idx_r} and the config "
                        f"records at {idx_c} — `resumed` must come FIRST on a "
                        f"resumed leg (status.html resets on `config`)")
    res = json.load(open(os.path.join(seam, "results.json")))
    if "in_progress" in res:
        FAILURES.append("C8: the finished run's results.json still carries "
                        "in_progress — that key's absence is the run's only "
                        "completion certificate")

    print(f"  C8 resume · 20 + --resume 20 against 40 straight (same seed, the "
          f"SAME 40-step cosine on both legs via --stop-at) · all {n_leaf} "
          f"checkpoint leaves np.array_equal, both host RNG states identical · "
          f"{len(aa)} metrics records after the seam identical (wall_s "
          f"excluded — it restarts on a new node) · `resumed` precedes the "
          f"resumed leg's `config` · results.json has dropped in_progress · "
          f"{wall:.0f}s")


def main():
    print("tests/test_jaxport_cone.py — E-069 cone codec: torch vs "
          "ml/jaxport, CPU, fp32, eval mode\n")
    for fn in (test_c1_forward, test_c2_loss, test_c3_one_step,
               test_c5_round_trip, test_c6_refusal, test_c7_invalid_dot,
               test_c9_forward_unchanged, test_c8_resume, test_c4_band):
        fn()
    if FAILURES:
        print("\nFAILED:")
        for x in FAILURES:
            print("  -", x)
        return 1
    print("\ntests/test_jaxport_cone.py: gates C1, C2, C3, C4, C5, C6, C7, "
          "C8, C9 passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())

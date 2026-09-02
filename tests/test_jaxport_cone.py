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

The gates in this file (C3, C4, C8 belong to the trainer and are stubbed at
the bottom so their absence is visible rather than assumed):

  C1 forward   `tokens` (both outputs), `encode` (z and latents),
               `query_tokens`, `decode_from_z`, `decode`                 1e-4
  C2 loss      `forward_given` vs `loss_given`, loss and every `terms`
               scalar, aux on (0.25) and off (0)                         1e-5
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

C9 is the gate that lets `forward` be refactored at all. `forward` still draws
its hidden-dot queries INSIDE `_query_sets` (after `encode`), where they have
always been drawn, while `forward_given` takes them from
`draw_dot_queries` — so the two agree only if `encode` consumes no RNG, which
is true (every dropout in `ConeMAE` is 0.0) and is asserted here rather than
argued in a comment.
"""
import os
import sys
import tempfile
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


# ---- C3 / C4 / C8: the trainer's gates, deliberately not here --------------
def test_c3_c4_c8_are_the_trainers():
    """C3 (one optimiser step), C4 (a 300-step band against
    `ml/train_cone.py --smoke`) and C8 (checkpoint/resume bitwise) are gates on
    `ml/jaxport/train_cone.py`, which this file does not import. They are named
    here so their absence is visible in the output rather than assumed from
    it."""
    print("  C3/C4/C8 · not in this file — they gate ml/jaxport/train_cone.py "
          "(one optimiser step, the 300-step band, checkpoint resume)")


def main():
    print("tests/test_jaxport_cone.py — E-069 cone codec: torch vs "
          "ml/jaxport, CPU, fp32, eval mode\n")
    for fn in (test_c1_forward, test_c2_loss, test_c5_round_trip,
               test_c6_refusal, test_c7_invalid_dot,
               test_c9_forward_unchanged, test_c3_c4_c8_are_the_trainers):
        fn()
    if FAILURES:
        print("\nFAILED:")
        for x in FAILURES:
            print("  -", x)
        return 1
    print("\ntests/test_jaxport_cone.py: gates C1, C2, C5, C6, C7, C9 passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())

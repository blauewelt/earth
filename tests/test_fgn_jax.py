#!/usr/bin/env python3
"""The E-057 FGN certificate for `ml/jaxport` — phase 1 of
`ml/plans/FGN_JAX_PORT.md`. CPU, fp32, no network, no checkpoint, no tensor.

The port is UNUSABLE until every check here passes (§1 of the plan names them
(a)-(d); (e) and (f) are this file's own additions for the two mechanisms the
plan describes in prose but nothing else would pin).

  **1 · Flag-off purity.** `--fgn-eps 0` must build NO film parameter and take
  the pre-E-057 code path statement for statement. Asserted three ways so a
  future edit cannot satisfy one and break the others: the exported torch key
  set is EXACTLY the deterministic torch module's; the forward is BITWISE
  equal to an independent re-transcription of the pristine pre-diff forward
  (written inline here in plain jnp, which is what makes it a check rather
  than a tautology); and handing that model an ε raises.

  **2 · Forward equivalence with ε INJECTED on both sides.** A tiny torch FGN
  head with RANDOMISED film weights (zero film would make this test pass on a
  converter that dropped film entirely) converted through `convert.py`, the
  SAME ε fed to both backends, outputs compared. MEASURED over three
  configurations (stencil 1 k=8, stencil 9 k=4, stencil 1 k=32): the WORST
  max-relative deviation is **3.08e-07** on the prediction and **3.52e-07**
  on the hidden state, against the plan's ≤ ~1e-4 float-dispatch floor — the
  same class as `tests/test_jaxport_parity.py`'s existing 4.77e-07, i.e. the
  ε path costs the port nothing in parity. The film ROUND TRIP
  (torch → jax → torch) is EXACT, max|Δ| 0.0.

  **3 · `fair_crps2` value parity.** Fixed numpy arrays through torch's own
  `ml/temporal.py:fair_crps2` and through the JAX transcription, to 1e-6.
  `fair_crps_ens` too, at M = 1 (where it must be MAE EXACTLY), 2 (where it
  must equal `fair_crps2`) and 5.

  **4 · The OOM lesson, as an identity.** The monitor's M-member read chunked
  in 512-window slices against the same read unchunked: ≤ 512 windows is one
  slice and trivially identical, and 800 windows is two slices and is the
  case #496's fix actually changes. Both must agree bitwise — the forward is
  row-wise and ε is broadcast per member, so a slice boundary cannot change
  which noise a window sees.

  **5 · The refuse-under-MSE guard.** `--fgn-val-members` with `--fgn-eps 0`
  refuses, and so do a negative k and an M below 2.

  **6 · ε resume-exactness.** The (seed, step, forward) fold, twice in one
  process and across a SIMULATED save/resume that carries nothing but the seed
  and the step counter, must be bitwise equal — and two different steps must
  differ, or the first assertion would pass on a constant.

    python3 -m pytest tests/test_fgn_jax.py -x -q
    python3 tests/test_fgn_jax.py                 # the same checks, verbose
"""
import os
import sys
import warnings

import numpy as np
import torch

warnings.filterwarnings("ignore", message=".*enable_nested_tensor.*")

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
ML = os.path.join(ROOT, "ml")
if ML not in sys.path:
    sys.path.insert(0, ML)

import jax                                                       # noqa: E402
import jax.numpy as jnp                                          # noqa: E402
from flax import nnx                                             # noqa: E402

from temporal import (TemporalTransformer as TorchTemporal,      # noqa: E402
                      fair_crps2 as torch_fair_crps2,
                      fair_crps_ens as torch_fair_crps_ens)
from jaxport import models as jm                                 # noqa: E402
from jaxport import convert as jc                                # noqa: E402
from jaxport.train_stage2 import (FGN_MONITOR_CHUNK,             # noqa: E402
                                  fair_crps2 as jax_fair_crps2,
                                  fair_crps_ens as jax_fair_crps_ens,
                                  fgn_eps_at, fgn_eval_eps,
                                  fgn_monitor_ens, fgn_refusals,
                                  fgn_train_key, fgn_val_bank,
                                  fgn_val_metrics)

# fp64 is OFF (jax's default): every comparison below is a float32 one, which
# is the regime the fleet trains in.
jax.config.update("jax_enable_x64", False)

TOY = dict(d_z=4, d_model=16, n_heads=4, n_layers=2, k_max=6)
K, B = 6, 5

# Each check RECORDS its one-line finding instead of returning it: pytest >= 9
# warns on a test function that returns a value (and will eventually fail on
# one), while the repo's own `python3 tests/<file>.py` style prints the
# measured numbers rather than a bare dot. Both readers get what they want.
NOTES = []


def _note(msg):
    NOTES.append(msg)


def _toy_batch(seed=0, stencil=1, d_z=TOY["d_z"], b=B, k=K):
    r = np.random.default_rng(seed)
    z = r.standard_normal((b, k, stencil * d_z)).astype(np.float32)
    m = r.standard_normal((b, k, 2)).astype(np.float32)
    s = r.standard_normal((b, d_z + 2 + (stencil if stencil > 1 else 0))
                          ).astype(np.float32)
    return z, m, s


def _maxrel(a, b):
    """max |a-b| / (max|b| or 1) — the same relative measure the jaxport
    parity suite reports, so the numbers are comparable across files."""
    a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
    return float(np.abs(a - b).max() / max(np.abs(b).max(), 1.0))


# --------------------------------------------------------------------------
# 1 · flag-off purity
# --------------------------------------------------------------------------
def _pristine_forward(model, z, m, s):
    """The PRE-DIFF forward, re-transcribed here in plain jnp off the same
    parameters — deliberately NOT calling any of the new code, so this is an
    independent statement of what the model used to compute rather than a
    rewording of what it now computes."""
    B_, K_ = z.shape[0], z.shape[1]
    h = (model.inp(jnp.concatenate([z, m], axis=-1))
         + model.static(s)[:, None, :]
         + model.pos.embedding.value[None, :K_])
    mask = jm.causal_mask(K_, h.dtype)
    for lyr in model.encoder.layers:
        q = lyr.norm1(h)
        h = h + lyr.self_attn(q, q, q, mask=mask)
        q = lyr.norm2(h)
        h = h + lyr.linear2(jax.nn.relu(lyr.linear1(q)))
    return model.head(h), h


def test_flag_off_purity():
    """fgn_eps=0 builds no film, exports the deterministic key set, is bitwise
    the pristine forward, and refuses an eps."""
    torch.manual_seed(7)
    tdet = TorchTemporal(**TOY)
    jdet = jm.TemporalTransformer(**TOY, rngs=nnx.Rngs(0))
    jc.load_temporal(tdet.state_dict(), jdet)

    # (a) no film parameter anywhere, and no eps path.
    assert jdet.eps_dim == 0
    assert jdet.eps_embed is None
    for lyr in jdet.encoder.layers:
        assert lyr.film is None, "fgn_eps=0 must build NO film parameter"

    # (b) the exported key set is EXACTLY the deterministic torch module's —
    # not a superset with zeros in it.
    got = set(jc.export_temporal(jdet))
    want = set(tdet.state_dict())
    assert got == want, f"key set drifted: +{sorted(got - want)} " \
                        f"-{sorted(want - got)}"
    assert not [k for k in got if ".film." in k or k.startswith("eps_embed")]

    # (c) the forward is BITWISE the pristine one.
    z, m, s = _toy_batch(1)
    zj, mj, sj = jnp.asarray(z), jnp.asarray(m), jnp.asarray(s)
    p_new, h_new = jdet(zj, mj, sj)
    p_old, h_old = _pristine_forward(jdet, zj, mj, sj)
    assert np.array_equal(np.asarray(p_new), np.asarray(p_old)), \
        "the flag-off forward is no longer bitwise the pre-diff one"
    assert np.array_equal(np.asarray(h_new), np.asarray(h_old))

    # (d) and it refuses an eps rather than ignoring it, in BOTH backends.
    try:
        jdet(zj, mj, sj, eps=jnp.zeros((B, 4), jnp.float32))
        raise AssertionError("a deterministic head accepted an eps")
    except ValueError as e:
        assert "eps_dim=0" in str(e)
    try:
        tdet(torch.as_tensor(z), torch.as_tensor(m), torch.as_tensor(s),
             eps=torch.zeros(B, 4))
        raise AssertionError("the torch deterministic head accepted an eps")
    except ValueError:
        pass
    # fgn_eval_eps is None when there is no eps path — the one place the
    # member choice lives, and it must not invent a member for a head that
    # has none.
    assert fgn_eval_eps(B, 0) is None

    # (e) THE ZERO-FILM IDENTITY, the flag-ON half of the same property: an
    # FGN head at INIT is the deterministic incumbent BITWISE, whatever eps it
    # is handed, because film is exact zeros and `x*1.0 + 0.0` is exact. This
    # is the JAX twin of "r_fore reads exactly 1.000000 at step 1", and it is
    # what makes an FGN arm's step 0 comparable to its deterministic control.
    torch.manual_seed(7)
    tfgn = TorchTemporal(**TOY, eps_dim=8)
    jfgn = jm.TemporalTransformer(**TOY, eps_dim=8, rngs=nnx.Rngs(0))
    jc.load_temporal(tfgn.state_dict(), jfgn)
    # the trunk of the fgn twin is a different draw; copy the deterministic
    # trunk in, exactly as an E-057 warm start does (strict=False on torch).
    tfgn.load_state_dict(tdet.state_dict(), strict=False)
    jc.load_temporal(tfgn.state_dict(), jfgn)
    for e_scale in (0.0, 1.0, 7.5):
        e = jnp.asarray(np.random.default_rng(2).standard_normal((B, 8))
                        .astype(np.float32) * e_scale)
        pf, hf = jfgn(zj, mj, sj, eps=e)
        assert np.array_equal(np.asarray(pf), np.asarray(p_new)), \
            f"zero-init film is not the identity at |eps| scale {e_scale}"
        assert np.array_equal(np.asarray(hf), np.asarray(h_new))

    _note("flag-off purity: no film params, key set identical to torch's "
          "deterministic module, forward BITWISE the pre-diff transcription, "
          "and an injected eps refuses on both sides. Flag ON at INIT: a "
          "zero-film FGN head is BITWISE the deterministic twin at |eps| "
          "scales 0, 1 and 7.5")


# --------------------------------------------------------------------------
# 2 · forward equivalence, eps injected on both sides
# --------------------------------------------------------------------------
def _randomise_film(tmod, seed=11):
    """Torch AND JAX zero-init film, so a converter that silently dropped
    film would pass every parity check with the weights at their init. Give
    every film and eps_embed parameter a real value first."""
    g = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for n, p in tmod.named_parameters():
            if ".film." in n or n.startswith("eps_embed."):
                p.copy_(torch.randn(p.shape, generator=g) * 0.3)
        # nonzero film alone is not enough if the trunk norms are at init;
        # perturb everything so no term drops out of the comparison.
        for n, p in tmod.named_parameters():
            if ".film." not in n and not n.startswith("eps_embed."):
                p.add_(torch.randn(p.shape, generator=g) * 0.05)
    return tmod


def _equivalence(stencil, eps_dim, seed):
    torch.manual_seed(seed)
    tmod = TorchTemporal(**TOY, stencil=stencil, eps_dim=eps_dim).eval()
    _randomise_film(tmod, seed + 100)
    jmod = jm.TemporalTransformer(**TOY, stencil=stencil, eps_dim=eps_dim,
                                  rngs=nnx.Rngs(0))
    jc.load_temporal(tmod.state_dict(), jmod)

    z, m, s = _toy_batch(seed, stencil=stencil)
    eps = np.random.default_rng(seed + 5).standard_normal(
        (B, eps_dim)).astype(np.float32)
    with torch.no_grad():
        tp, th = tmod(torch.as_tensor(z), torch.as_tensor(m),
                      torch.as_tensor(s), eps=torch.as_tensor(eps))
    jp, jh = jmod(jnp.asarray(z), jnp.asarray(m), jnp.asarray(s),
                  eps=jnp.asarray(eps))
    return (_maxrel(jp, tp.numpy()), _maxrel(jh, th.numpy()))


def test_forward_equivalence():
    """A tiny torch FGN head with TRAINED-shaped (randomised) film converts to
    JAX and matches to the float-dispatch floor with the same eps injected."""
    out = []
    for stencil, eps_dim, seed in ((1, 8, 3), (9, 4, 4), (1, 32, 5)):
        dp, dh = _equivalence(stencil, eps_dim, seed)
        assert dp < 1e-4, f"stencil {stencil} k {eps_dim}: pred rel {dp:.3e}"
        assert dh < 1e-4, f"stencil {stencil} k {eps_dim}: hid rel {dh:.3e}"
        out.append((stencil, eps_dim, dp, dh))

    # The converter must ROUND-TRIP trained film, not merely read it: JAX ->
    # torch state_dict, compared to the state_dict it was built from.
    torch.manual_seed(21)
    tmod = _randomise_film(TorchTemporal(**TOY, eps_dim=8))
    jmod = jm.TemporalTransformer(**TOY, eps_dim=8, rngs=nnx.Rngs(0))
    jc.load_temporal(tmod.state_dict(), jmod)
    back = jc.export_temporal(jmod)
    ref = tmod.state_dict()
    assert set(back) == set(ref), f"round trip key drift: " \
                                  f"{sorted(set(back) ^ set(ref))}"
    worst = max(float(np.abs(back[k] - ref[k].numpy()).max()) for k in ref)
    assert worst == 0.0, f"film round trip is not exact: max|Δ| {worst:.3e}"
    # And the torch module accepts it back with strict=True.
    tmod2 = TorchTemporal(**TOY, eps_dim=8)
    tmod2.load_state_dict({k: torch.from_numpy(v) for k, v in back.items()},
                          strict=True)

    # A deterministic model handed an FGN state_dict must REFUSE, not warm-
    # start silently: warm-starting a trunk is a deliberate strict=False act.
    try:
        jc.load_temporal(tmod.state_dict(),
                         jm.TemporalTransformer(**TOY, rngs=nnx.Rngs(0)))
        raise AssertionError("a deterministic model consumed an FGN checkpoint")
    except KeyError as e:
        assert "film" in str(e)

    _note("forward equivalence with eps INJECTED on both sides, film "
          "RANDOMISED: " + " · ".join(
              f"stencil {st} k {k} pred {dp:.2e} hid {dh:.2e}"
              for st, k, dp, dh in out)
          + f" (gate 1e-4). The film round trip torch->jax->torch is EXACT "
            f"(max|Δ| {worst:.1e}) and reloads strict=True; a deterministic "
            f"model refuses an FGN state_dict")


# --------------------------------------------------------------------------
# 3 · fair_crps2 / fair_crps_ens value parity
# --------------------------------------------------------------------------
def test_fair_crps2_value_parity():
    r = np.random.default_rng(42)
    worst2, worstM = 0.0, 0.0
    for shape in ((7, 5, 4), (3, 11, 2), (64, 6, 8)):
        x1 = r.standard_normal(shape).astype(np.float32)
        x2 = r.standard_normal(shape).astype(np.float32)
        y = r.standard_normal(shape).astype(np.float32)
        tv = float(torch_fair_crps2(torch.as_tensor(x1), torch.as_tensor(x2),
                                    torch.as_tensor(y)))
        jv = float(jax_fair_crps2(jnp.asarray(x1), jnp.asarray(x2),
                                  jnp.asarray(y)))
        worst2 = max(worst2, abs(tv - jv))
        assert abs(tv - jv) < 1e-6, f"fair_crps2 {shape}: {tv} vs {jv}"

        # the M-member estimator, including its two structural identities
        for M in (1, 2, 5):
            ens = r.standard_normal((M,) + shape).astype(np.float32)
            tvM = float(torch_fair_crps_ens(torch.as_tensor(ens),
                                            torch.as_tensor(y)))
            jvM = float(jax_fair_crps_ens(jnp.asarray(ens), jnp.asarray(y)))
            worstM = max(worstM, abs(tvM - jvM))
            assert abs(tvM - jvM) < 1e-6, f"fair_crps_ens M={M}"
            if M == 1:
                mae = float(np.abs(ens[0] - y).mean())
                assert abs(jvM - mae) < 1e-6, "M=1 must be MAE exactly"
            if M == 2:
                j2 = float(jax_fair_crps2(jnp.asarray(ens[0]),
                                          jnp.asarray(ens[1]), jnp.asarray(y)))
                assert abs(jvM - j2) < 1e-6, "M=2 must equal fair_crps2"
    # identical members: the pair term vanishes and only MAE remains.
    x = r.standard_normal((4, 3)).astype(np.float32)
    y = r.standard_normal((4, 3)).astype(np.float32)
    assert abs(float(jax_fair_crps2(jnp.asarray(x), jnp.asarray(x),
                                    jnp.asarray(y)))
               - float(np.abs(x - y).mean())) < 1e-6
    _note(f"fair_crps2 matches ml/temporal.py's estimator to "
          f"{worst2:.2e} and fair_crps_ens (M=1,2,5) to {worstM:.2e} "
          f"(gate 1e-6); M=1 is MAE exactly, M=2 equals fair_crps2, and "
          f"identical members give MAE")


# --------------------------------------------------------------------------
# 4 · the OOM lesson: chunked == unchunked
# --------------------------------------------------------------------------
def test_monitor_chunked_identity():
    torch.manual_seed(31)
    tmod = _randomise_film(TorchTemporal(**TOY, eps_dim=6))
    jmod = jm.TemporalTransformer(**TOY, eps_dim=6, rngs=nnx.Rngs(0))
    jc.load_temporal(tmod.state_dict(), jmod)

    def fwd(z_, m_, c_, e_):
        return jmod(z_, m_, c_, eps=e_)

    eps_val = fgn_val_bank(0, 4, 6)
    out = []
    for n in (300, 800):                       # 1 chunk, then exactly 2
        z, m, s = _toy_batch(9, b=n)
        z, m, s = jnp.asarray(z), jnp.asarray(m), jnp.asarray(s)
        chunked = fgn_monitor_ens(fwd, z, m, s, eps_val,
                                  chunk=FGN_MONITOR_CHUNK)
        whole = fgn_monitor_ens(fwd, z, m, s, eps_val, chunk=0)
        nch = int(np.ceil(n / FGN_MONITOR_CHUNK))
        assert chunked.shape == (4, n, TOY["d_z"]) == whole.shape
        assert np.array_equal(np.asarray(chunked), np.asarray(whole)), \
            f"chunked monitor differs from unchunked at n={n}"
        # and the metrics computed off them, key by key
        yt = jnp.asarray(np.random.default_rng(2).standard_normal(
            (n, TOY["d_z"])).astype(np.float32))
        a1 = fgn_val_metrics(chunked, yt)
        a2 = fgn_val_metrics(whole, yt)
        assert a1[0] == a2[0] and a1[1] == a2[1] and a1[2] == a2[2]
        # the keys are the ones status.html already knows
        assert set(a1[2]) == {"stage2_val_crps", "stage2_val_member_var",
                              "stage2_val_spread_ratio"}
        out.append((n, nch))
    # a degenerate ensemble (all members equal) has EXACTLY zero member
    # variance — the collapse signature the telemetry exists to show.
    flat = jnp.broadcast_to(jnp.asarray(
        np.random.default_rng(3).standard_normal((1, 8, 4)).astype(np.float32)),
        (4, 8, 4))
    _, _, ex = fgn_val_metrics(flat, jnp.zeros((8, 4), jnp.float32))
    assert ex["stage2_val_member_var"] == 0.0
    _note("monitor eval CHUNKED at 512 windows is bitwise the unchunked "
          "read at " + " and ".join(f"n={n} ({c} chunk{'s' if c > 1 else ''})"
                                    for n, c in out)
          + ", and so are all three telemetry keys; a collapsed ensemble "
            "reads member_var exactly 0.0")


# --------------------------------------------------------------------------
# 5 · the refusals
# --------------------------------------------------------------------------
class _A:
    def __init__(self, fgn_eps, fgn_val_members=8):
        self.fgn_eps, self.fgn_val_members = fgn_eps, fgn_val_members


def test_refusals():
    assert fgn_refusals(_A(0)) == "mse"
    assert fgn_refusals(_A(8)) == "crps2"
    assert fgn_refusals(_A(8, 2)) == "crps2"

    fired = []
    for args, want in ((_A(0, 16), "--fgn-eps 0"),
                       (_A(-1), "must be >= 0"),
                       (_A(8, 1), "must be >= 2")):
        try:
            fgn_refusals(args)
            raise AssertionError(f"no refusal for {vars(args)}")
        except SystemExit as e:
            assert want in str(e), f"{want!r} not in {str(e)!r}"
            fired.append(want)
    assert len(fired) == 3
    _note("the refuse-under-MSE guard fires (--fgn-val-members with "
          "--fgn-eps 0), and so do k < 0 and M < 2; the default pair "
          "passes as 'mse' and k>0 as 'crps2'")


# --------------------------------------------------------------------------
# 6 · eps resume-exactness
# --------------------------------------------------------------------------
def test_eps_resume_exact():
    """The fold is a PURE function of (seed, step, forward), so a resume that
    carries only the seed and the step counter reproduces it bitwise."""
    seed, B_, k = 1234, 17, 6
    root = fgn_train_key(seed)
    e1a = fgn_eps_at(root, 4096, 0, B_, k)
    e1b = fgn_eps_at(root, 4096, 0, B_, k)
    assert np.array_equal(np.asarray(e1a), np.asarray(e1b)), \
        "the same (seed, step, forward) gave two different eps"

    # SIMULATED SAVE/RESUME: nothing but the seed and the step survive — no
    # key, no counter, no generator state. The root is rebuilt from the seed.
    saved = {"seed": seed, "step": 4096}
    root2 = fgn_train_key(saved["seed"])
    e1c = fgn_eps_at(root2, saved["step"], 0, B_, k)
    assert np.array_equal(np.asarray(e1a), np.asarray(e1c)), \
        "eps did not survive a save/resume that carried only (seed, step)"

    # ... and the pair members and neighbouring steps must actually DIFFER,
    # or the assertions above would pass on a constant.
    e2 = fgn_eps_at(root, 4096, 1, B_, k)
    e3 = fgn_eps_at(root, 4097, 0, B_, k)
    e4 = fgn_eps_at(fgn_train_key(seed + 1), 4096, 0, B_, k)
    for other, what in ((e2, "forward index"), (e3, "step"), (e4, "seed")):
        assert not np.allclose(np.asarray(e1a), np.asarray(other)), \
            f"eps does not depend on the {what}"
    # the eval bank is separate and equally reproducible, and drawing it does
    # NOT perturb the training stream (different root, no shared state).
    b1 = fgn_val_bank(seed, 8, k)
    b2 = fgn_val_bank(seed, 8, k)
    assert np.array_equal(np.asarray(b1), np.asarray(b2))
    assert not np.allclose(np.asarray(b1[0]), np.asarray(e1a[0]))
    e1d = fgn_eps_at(root, 4096, 0, B_, k)
    assert np.array_equal(np.asarray(e1a), np.asarray(e1d))
    # sanity on the distribution itself
    big = np.asarray(fgn_eps_at(root, 1, 0, 20000, k))
    assert abs(big.mean()) < 0.03 and abs(big.std() - 1.0) < 0.03
    _note("eps at (seed, step, forward) is bitwise reproducible in-process "
          "and across a save/resume carrying ONLY (seed, step) — no RNG "
          "state in the checkpoint; it moves with all three of seed, step "
          "and forward index, and the eval bank is a separate root that "
          "cannot perturb the training stream")


TESTS = [test_flag_off_purity, test_forward_equivalence,
         test_fair_crps2_value_parity, test_monitor_chunked_identity,
         test_refusals, test_eps_resume_exact]


if __name__ == "__main__":
    print("tests/test_fgn_jax.py — E-057 FGN in ml/jaxport, CPU, fp32\n")
    for i, fn in enumerate(TESTS, 1):
        NOTES.clear()
        fn()
        print(f"  {i}. {NOTES[-1]}\n")
    print(f"tests/test_fgn_jax.py: all {len(TESTS)} checks passed")

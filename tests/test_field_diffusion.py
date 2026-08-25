#!/usr/bin/env python3
"""E-052's field head, held to EXACT identities wherever one exists.

    python3 tests/test_field_diffusion.py

Plain python, no pytest — same convention as tests/test_train_config_guards.py.
Every `test_*` below runs in-process on CPU; the whole file is a couple of
minutes.

`ml/plans/E052_field_diffusion.md` §"What is built, and how it is tested" lists
what this suite must hold. The reason so many of these are `torch.equal` rather
than `allclose` is ml/CLAUDE.md §4.9: *"Build an invariant with an EXACT
expected value and make the job refuse if it fails. Prefer exact identities to
threshold checks — thresholds are the tripwires that killed healthy runs."*
Three of the identities here are not incidental, they are DESIGNED IN:

  * the tokenizer is a zero-fill, a copy and a gather, so its round trip is
    bitwise — no arithmetic sits on that path;
  * the DiT's final layer and every adaLN modulation are zero-initialized, so
    a fresh `det` head adds exactly 0.0 to z_t and IS persistence, and a fresh
    `diff` head's denoiser is exactly c_skip(sigma)*x;
  * every random draw comes from a passed `torch.Generator`, so a member of an
    M-member sample equals the same member drawn alone under the derived seed.

If one of those fails, the model is wrong and the tolerance is right. Fix the
model.
"""
import json
import os
import shutil
import sys
import tempfile
import time

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ML = os.path.join(HERE, "..", "ml")
sys.path.insert(0, ML)

from field_model import FieldHead, OceanTokenizer, nfe_to_steps   # noqa: E402
import train_field as tfield                                      # noqa: E402


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def holey_mask(H=10, W=12):
    """A grid with land: one WHOLE patch cell of it, plus scattered holes.

    Both kinds matter. A fully-land cell must not become a token at all (an
    all-zero token would spend attention on nothing and would tell the model
    there is ocean where there is none); a partly-land cell must become a token
    whose land slots are zero with flag 0.
    """
    mask = np.ones((H, W), bool)
    mask[0:4, 8:12] = False          # the whole (py=0, px=2) cell
    for y, x in [(5, 3), (6, 7), (9, 0), (7, 11)]:
        mask[y, x] = False
    ys, xs = np.where(mask)
    return H, W, ys, xs


def tiny_head(tok, d_z, K, mode, seed=0, sigma_data=0.75, d_model=32,
              layers=2, heads=2, d_cond=32, cond_layers=1):
    torch.manual_seed(seed)
    return FieldHead(tok, d_z, K, mode=mode, d_model=d_model, layers=layers,
                     heads=heads, d_cond=d_cond, cond_layers=cond_layers,
                     cond_heads=heads, sigma_data=sigma_data)


def rand_ctx(tok, model, B, K, d_z, seed=1):
    g = torch.Generator().manual_seed(seed)
    z = torch.randn(B, K, tok.P, d_z, generator=g)
    return model.make_cond(tok.to_tokens(z)), z[:, -1].contiguous()


def land_rows(tok):
    """Rows of the [ntok*patch^2] token buffer that no ocean pixel occupies."""
    all_rows = torch.arange(tok.ntok * tok.P2)
    is_ocean = torch.zeros(tok.ntok * tok.P2, dtype=torch.bool)
    is_ocean[tok.flat_of_px] = True
    return all_rows[~is_ocean]


# ---------------------------------------------------------------------------
# 1 · tokenizer
# ---------------------------------------------------------------------------
def test_tokenizer_roundtrip():
    """to_pixels(to_tokens(z)) is the identity BITWISE, land holes and all."""
    for patch in (2, 4):
        H, W, ys, xs = holey_mask()
        tok = OceanTokenizer(H, W, ys, xs, patch)
        d_z = 3
        g = torch.Generator().manual_seed(patch)
        for lead in [(), (5,), (2, 4)]:
            z = torch.randn(*lead, tok.P, d_z, generator=g)
            back = tok.to_pixels(tok.to_tokens(z), d_z)
            if not torch.equal(back, z):
                raise SystemExit(
                    f"tokenizer round trip is not bitwise at patch={patch}, "
                    f"lead={lead}: max |delta| {float((back - z).abs().max())}")

        # the fully-land cell must NOT be a token
        expect_tok = len(np.unique((ys // patch) * ((W + patch - 1) // patch)
                                   + (xs // patch)))
        if tok.ntok != expect_tok:
            raise SystemExit(f"ntok {tok.ntok} != {expect_tok} occupied cells")
        if patch == 4:
            cells = set(zip(tok.tok_py.tolist(), tok.tok_px.tolist()))
            if (0, 2) in cells:
                raise SystemExit("an all-land patch cell became a token")

        # land slots are zero, with flag 0
        z = torch.randn(2, tok.P, d_z, generator=g) + 7.0     # nothing near 0
        t = tok.to_tokens(z).reshape(2, tok.ntok * tok.P2, d_z + 1)
        lr = land_rows(tok)
        if lr.numel() == 0:
            raise SystemExit("the test mask produced no land slots at all")
        if not torch.equal(t[:, lr], torch.zeros_like(t[:, lr])):
            raise SystemExit("a land slot is not zero-filled")
        flag = t[..., d_z]
        if not torch.equal(flag[:, tok.flat_of_px],
                           torch.ones_like(flag[:, tok.flat_of_px])):
            raise SystemExit("an ocean slot's flag is not 1")

        # the token ORDER is (py, px) ascending and depends only on the mask
        key = tok.tok_py * 10_000 + tok.tok_px
        if not torch.equal(key, key.sort().values):
            raise SystemExit("tokens are not ordered by (py, px)")
        perm = np.random.default_rng(0).permutation(len(ys))
        tok2 = OceanTokenizer(H, W, ys[perm], xs[perm], patch)
        if not (torch.equal(tok.tok_py, tok2.tok_py)
                and torch.equal(tok.tok_px, tok2.tok_px)):
            raise SystemExit("token order depends on the order ys/xs arrive in")
    return "tokenizer: bitwise round trip, zero land slots, fixed order"


# ---------------------------------------------------------------------------
# 2 · det mode at init IS persistence
# ---------------------------------------------------------------------------
def test_det_init_is_persistence():
    """The plan's read-out "ratio 1.000000 at step 0", as a bitwise identity."""
    H, W, ys, xs = holey_mask()
    tok = OceanTokenizer(H, W, ys, xs, 4)
    d_z, K, B = 3, 4, 2
    model = tiny_head(tok, d_z, K, "det", seed=3)
    model.eval()
    with torch.no_grad():
        cond, z_t = rand_ctx(tok, model, B, K, d_z, seed=11)
        r = model.residual_det(cond)
        z_hat = model.forward_det(cond, z_t)
    if not torch.equal(r, torch.zeros_like(r)):
        raise SystemExit(f"r_hat is not exactly 0 at init "
                         f"(max |r| {float(r.abs().max())}) — the zero-init "
                         f"final layer or a stray bias is broken")
    if not torch.equal(z_hat, z_t):
        raise SystemExit("z_hat != z_t bitwise at init: det mode is not "
                         "persistence at step 0")
    # and after ONE optimizer step it must move — an identity that holds
    # because the model is dead is not the identity being claimed.
    opt = torch.optim.SGD(model.parameters(), lr=1.0)
    cond, z_t = rand_ctx(tok, model, B, K, d_z, seed=11)
    tgt = z_t + torch.randn(z_t.shape, generator=torch.Generator().manual_seed(2))
    loss = (model.forward_det(cond, z_t) - tgt).pow(2).mean()
    opt.zero_grad(); loss.backward(); opt.step()
    with torch.no_grad():
        if torch.equal(model.residual_det(cond),
                       torch.zeros_like(z_t)):
            raise SystemExit("the head is still exactly zero after a training "
                             "step — no gradient reaches the output")
    return "det at init: r_hat == 0 and z_hat == z_t bitwise; trainable after"


# ---------------------------------------------------------------------------
# 3 · diff mode at init
# ---------------------------------------------------------------------------
def test_diff_init_is_cskip():
    """D(x; sigma) == c_skip(sigma) * x exactly, at every sigma."""
    H, W, ys, xs = holey_mask()
    tok = OceanTokenizer(H, W, ys, xs, 2)
    d_z, K, B = 2, 3, 3
    sd = 0.75
    model = tiny_head(tok, d_z, K, "diff", seed=5, sigma_data=sd)
    model.eval()
    with torch.no_grad():
        cond, _ = rand_ctx(tok, model, B, K, d_z, seed=21)
        g = torch.Generator().manual_seed(31)
        x = torch.randn(B, tok.P, d_z, generator=g)
        for s in (1e-3, 0.01, 0.3, sd, 3.0, 80.0):
            sig = torch.full((B,), float(s))
            got = model.D(x, sig, cond)
            # The IDENTITY is bitwise, and it is about the NETWORK's
            # contribution: c_out * 0 == 0, so D is exactly the skip branch.
            # c_skip itself is compared against the analytic value separately
            # and only to float32 precision — re-deriving it in float64 here
            # and demanding bitwise equality would be testing torch's
            # rounding, not the model.
            cs = model._coefs(sig)[0]
            want = cs[:, None, None] * x
            if not torch.equal(got, want):
                raise SystemExit(
                    f"D != c_skip*x at sigma={s}: max |delta| "
                    f"{float((got - want).abs().max())} — the zero-init final "
                    f"layer is leaking a non-zero network term")
            analytic = sd * sd / (s * s + sd * sd)
            if abs(float(cs[0]) - analytic) > 1e-6 * max(1.0, analytic):
                raise SystemExit(f"c_skip({s}) = {float(cs[0])} != {analytic}")
        # the sigma -> 0 limit returns the input (c_skip -> 1)
        tiny = torch.full((B,), 1e-8)
        if float((model.D(x, tiny, cond) - x).abs().max()) > 1e-10:
            raise SystemExit("the sigma -> 0 limit does not return x")
    return "diff at init: D == c_skip*x bitwise at 6 sigmas; sigma->0 -> x"


# ---------------------------------------------------------------------------
# 4 · sampler determinism
# ---------------------------------------------------------------------------
def test_sampler_determinism():
    """Same seed -> bitwise identical; member m of M == the M=1 derived call."""
    H, W, ys, xs = holey_mask()
    tok = OceanTokenizer(H, W, ys, xs, 4)
    d_z, K, B, M = 2, 3, 2, 4
    model = tiny_head(tok, d_z, K, "diff", seed=7, sigma_data=0.6)
    # A network that is identically zero would still pass the seeding tests
    # (the initial draw differs), but it would not exercise the DiT at all —
    # so give the final layer real weights first.
    g0 = torch.Generator().manual_seed(99)
    with torch.no_grad():
        model.dit.fin.weight.normal_(0.0, 0.05, generator=g0)
        for blk in model.dit.blocks:
            blk.ada[1].weight.normal_(0.0, 0.05, generator=g0)
    model.eval()
    with torch.no_grad():
        cond, z_t = rand_ctx(tok, model, B, K, d_z, seed=41)
        n_steps, _ = nfe_to_steps(7)
        a = model.sample(cond, z_t, n_steps, seed=5, M=M)
        b = model.sample(cond, z_t, n_steps, seed=5, M=M)
        c = model.sample(cond, z_t, n_steps, seed=6, M=M)
        if tuple(a.shape) != (M, B, tok.P, d_z):
            raise SystemExit(f"sample() shape {tuple(a.shape)} != [M,B,P,d_z]")
        if not torch.equal(a, b):
            raise SystemExit("the same seed produced different samples")
        if torch.equal(a, c):
            raise SystemExit("two different seeds produced identical samples")
        for m in range(M):
            one = model.sample(cond, z_t, n_steps, seed=5 + m, M=1)
            if not torch.equal(one[0], a[m]):
                raise SystemExit(
                    f"member {m} of an M={M} call differs from the M=1 call at "
                    f"the derived seed {5 + m} — members are not independently "
                    f"derived, so an ensemble cannot be extended or replayed")
        # and it refuses to touch the global RNG
        try:
            model.sample(cond, z_t, n_steps)
        except ValueError:
            pass
        else:
            raise SystemExit("sample() ran with no seed and no generator")
    return f"sampler: seeded, reproducible, member m == derived seed (M={M})"


# ---------------------------------------------------------------------------
# 5 · masked-loss invariance
# ---------------------------------------------------------------------------
def test_land_never_enters_the_loss():
    """Land values move neither loss, BITWISE — in both directions.

    (a) A whole-grid field whose LAND cells are replaced by garbage produces
        the identical ocean-pixel field, hence the identical losses: nothing
        upstream of the tokenizer reads land.
    (b) Garbage written into the LAND SLOTS of a model's OUTPUT tokens leaves
        the det loss untouched: `to_pixels` gathers ocean slots only, so the
        loss cannot be paid or earned on land.
    """
    H, W, ys, xs = holey_mask()
    tok = OceanTokenizer(H, W, ys, xs, 4)
    d_z, K, B = 3, 3, 2
    model = tiny_head(tok, d_z, K, "diff", seed=13, sigma_data=0.8)
    model.eval()
    lin = tok.ocean_lin
    land = torch.ones(H * W, dtype=torch.bool)
    land[lin] = False

    def losses(zfull):
        z = zfull[..., lin, :]
        ctx, z_t, z_n = z[:, :K], z[:, K - 1], z[:, K]
        cond = model.make_cond(tok.to_tokens(ctx))
        det = (model.forward_det(cond, z_t) - z_n).pow(2).mean()
        gen = torch.Generator().manual_seed(1234)
        edm = model.edm_loss(cond, z_t, z_n, gen)
        return float(det), float(edm)

    g = torch.Generator().manual_seed(77)
    zfull = torch.randn(B, K + 1, H * W, d_z, generator=g)
    with torch.no_grad():
        d0, e0 = losses(zfull)
        z2 = zfull.clone()
        z2[..., land, :] = 1e4 * torch.randn(
            B, K + 1, int(land.sum()), d_z, generator=g)
        d1, e1 = losses(z2)
    if (d0, e0) != (d1, e1):
        raise SystemExit(f"land values moved a loss: det {d0} -> {d1}, "
                         f"edm {e0} -> {e1}")

    # (b) garbage in the OUTPUT tokens' land slots
    with torch.no_grad():
        z = zfull[..., lin, :]
        ctx, z_t, z_n = z[:, :K], z[:, K - 1], z[:, K]
        cond = model.make_cond(tok.to_tokens(ctx))
        x = cond.new_zeros(B, tok.ntok, tok.feat_in(d_z))
        out = model.dit(x, cond, model.g_null_for(B))
        flat = out.reshape(B, tok.ntok * tok.P2, d_z)
        l0 = ((z_t + tok.to_pixels(out, d_z)) - z_n).pow(2).mean()
        lr = land_rows(tok)
        flat2 = flat.clone()
        flat2[:, lr] = 1e6
        out2 = flat2.reshape(B, tok.ntok, tok.P2 * d_z)
        l1 = ((z_t + tok.to_pixels(out2, d_z)) - z_n).pow(2).mean()
    if not torch.equal(l0, l1):
        raise SystemExit(f"land SLOTS of the output moved the det loss: "
                         f"{float(l0)} -> {float(l1)}")
    return "masked loss: land values and land output slots are both inert"


# ---------------------------------------------------------------------------
# 6 · checkpoint round-trip
# ---------------------------------------------------------------------------
def _params(model):
    return {k: v.detach().clone() for k, v in model.state_dict().items()}


def test_checkpoint_resume_is_bit_identical():
    """5 steps + resume + 3 steps == 8 uninterrupted steps, on EVERY parameter.

    The plan's test 7. Anything the trajectory depends on — the batch sampler,
    the EDM sigma and noise draws, the optimizer moments — has to be in the
    checkpoint for this to hold, and `diff` mode is used precisely because it
    consumes the generator twice per step where `det` consumes it once.
    """
    tmp = tempfile.mkdtemp()
    base = ["--toy", "gauss", "--mode", "diff", "--K", "3", "--patch", "4",
            "--d-model", "16", "--layers", "1", "--heads", "2",
            "--d-cond", "16", "--cond-layers", "1", "--cond-heads", "2",
            "--batch", "4", "--nfe", "3", "--members", "2",
            "--eval-windows", "2", "--eval-every", "5", "--seed", "0",
            "--quiet"]
    uninterrupted = tfield.main(base + ["--steps", "8",
                                        "--out", os.path.join(tmp, "u.json")])
    five = tfield.main(base + ["--steps", "5",
                               "--out", os.path.join(tmp, "a.json"),
                               "--ckpt", os.path.join(tmp, "a.pt")])
    resumed = tfield.main(base + ["--steps", "8", "--resume",
                                  "--out", os.path.join(tmp, "r.json"),
                                  "--ckpt", os.path.join(tmp, "a.pt")])
    pu, pr, p5 = (_params(x["model"]) for x in (uninterrupted, resumed, five))
    same_at_five = sum(1 for k in pu if torch.equal(pu[k], p5[k]))
    if same_at_five == len(pu):
        raise SystemExit("steps 6-8 changed no parameter at all — the resume "
                         "test would pass on a model that does not train")
    for k in pu:
        if not torch.equal(pu[k], pr[k]):
            raise SystemExit(
                f"resume diverged on {k}: max |delta| "
                f"{float((pu[k] - pr[k]).abs().max())}. A resumed run must "
                f"CONTINUE the trajectory, not merely resemble it.")
    if resumed["history"][-1]["step"] != 8:
        raise SystemExit("the resumed run did not reach step 8")
    shutil.rmtree(tmp, ignore_errors=True)
    return f"checkpoint: 5+3 == 8 bitwise on all {len(pu)} parameters"


# ---------------------------------------------------------------------------
# 7 · overfit smoke, det, axis A's microcosm
# ---------------------------------------------------------------------------
def test_det_beats_persistence_on_shift():
    """The `shift` toy: a per-pixel head cannot; the field head must.

    The threshold is deliberately generous (0.5 against a measured ~0.04): the
    claim is "joint spatial attention SEES a purely spatial law", not a level.
    A tighter bar here would be a tripwire, and ml/CLAUDE.md §4.9 is explicit
    about what tripwires cost.
    """
    tmp = tempfile.mkdtemp()
    r = tfield.main(["--toy", "shift", "--mode", "det", "--steps", "600",
                     "--eval-every", "300", "--d-model", "48", "--layers", "2",
                     "--heads", "4", "--d-cond", "48", "--cond-layers", "1",
                     "--K", "3", "--patch", "4", "--batch", "8", "--seed", "0",
                     "--out", os.path.join(tmp, "shift.json"), "--quiet"])
    ratio = r["final"]["ratio"]
    first = r["history"][0]["ratio"]
    shutil.rmtree(tmp, ignore_errors=True)
    if not (ratio < 0.5):
        raise SystemExit(f"shift ratio {ratio:.4f} is not below 0.5 — the "
                         f"field head is not seeing the one-cell roll")
    if not (ratio < first):
        raise SystemExit("the ratio did not improve between evals")
    return f"shift/det: ratio {ratio:.4f} (< 0.5), from {first:.4f} at step 300"


# ---------------------------------------------------------------------------
# 8 · diff learns a KNOWN conditional
# ---------------------------------------------------------------------------
def test_diff_recovers_the_gaussian_conditional():
    """gauss toy: sampled mean ~ a*x_t and sampled sd ~ sigma_e, within 15%.

    The comparison is POOLED — a regression slope of the ensemble mean on x_t,
    and the RMS of the per-element sampled sd — rather than element-by-element.
    With M=64 the Monte-Carlo error on one element's mean is sigma_e/8, which
    is the same size as the quantity being checked; pooled over B*P elements it
    is not. Checking per element would be measuring the ensemble size.
    """
    tmp = tempfile.mkdtemp()
    steps, K = 2000, 3
    r = tfield.main(["--toy", "gauss", "--mode", "diff", "--steps", str(steps),
                     "--eval-every", str(steps), "--d-model", "64",
                     "--layers", "3", "--heads", "4", "--d-cond", "64",
                     "--cond-layers", "2", "--K", str(K), "--patch", "4",
                     "--batch", "32", "--nfe", "30", "--members", "8",
                     "--seed", "0", "--out", os.path.join(tmp, "g.json"),
                     "--quiet"])
    model = r["model"]
    model.eval()
    tgen = torch.Generator(); tgen.manual_seed(0)
    ds = tfield.TOYS["gauss"](tgen)
    a_true, se_true = ds["a"], ds["sigma_e"]
    win = tfield.Windows(ds["Z"], K, None)
    _, va = tfield.make_splits(ds["Z"].shape[0], K, 0.15)
    ctx, z_t, _, sea = win.batch(va[:24])
    n_steps, spent = nfe_to_steps(30)
    with torch.no_grad():
        cond = model.make_cond(model.tok.to_tokens(ctx), sea)
        ens = model.sample(cond, z_t, n_steps, seed=7, M=64)
    mu = ens.mean(0).flatten().double()
    sd = float(ens.std(0, unbiased=True).pow(2).mean().sqrt())
    x = z_t.flatten().double()
    slope = float((x * mu).sum() / (x * x).sum())
    shutil.rmtree(tmp, ignore_errors=True)
    e_mu = abs(slope - a_true) / a_true
    e_sd = abs(sd - se_true) / se_true
    if e_mu > 0.15 or e_sd > 0.15:
        raise SystemExit(
            f"the sampled conditional misses the analytic one: slope {slope:.4f}"
            f" vs a={a_true} ({100 * e_mu:.1f}%), sd {sd:.4f} vs "
            f"sigma_e={se_true} ({100 * e_sd:.1f}%)")
    return (f"gauss/diff: slope {slope:.4f} vs a {a_true} ({100 * e_mu:.1f}%), "
            f"sd {sd:.4f} vs sigma_e {se_true} ({100 * e_sd:.1f}%), "
            f"nfe {spent}")


# ---------------------------------------------------------------------------
# 9 · result-file discipline (ml/CLAUDE.md §5.25) and the real-data path
# ---------------------------------------------------------------------------
def synthetic_npz(tmp, T=36, H=6, W=8, d_z=2):
    """A tiny stand-in for the real substrate: a [T,P,d_z] cache + its tensor.

    The real path is thin and its first GPU-scale run is a future arm, so this
    exists to keep it EXECUTED. ml/CLAUDE.md §4.8: any hour of GPU on a path
    that has never run is a coin flip.
    """
    rng = np.random.default_rng(0)
    mask = np.ones((H, W), bool)
    mask[0, 0:3] = False
    mask[4, 5] = False
    ys, xs = np.where(mask)
    Z = np.cumsum(0.3 * rng.standard_normal((T, len(ys), d_z)), axis=0)
    zp = os.path.join(tmp, "Z.npy")
    np.save(zp, Z.astype(np.float32))
    months = np.array([f"{1990 + i // 12}-{i % 12 + 1:02d}" for i in range(T)])
    np_ = os.path.join(tmp, "tensor.npz")
    np.savez(np_, months=months, ys=ys.astype(np.int64), xs=xs.astype(np.int64),
             lats=np.linspace(20, 40, H).astype(np.float32),
             lons=np.linspace(-60, -40, W).astype(np.float32))
    return zp, np_


def test_result_file_discipline():
    """`in_progress` until the end, then gone; atomic; never NaN; smoke runs."""
    tmp = tempfile.mkdtemp()

    # (a) the writer itself — the mid-run assertion, called directly
    p = os.path.join(tmp, "w.json")
    tfield.write_result(p, {"k": 1}, [{"step": 1}], final=None,
                        in_progress={"step": 1, "of": 9})
    mid = json.load(open(p))
    if "in_progress" not in mid or mid["final"] is not None:
        raise SystemExit("a mid-run write must carry in_progress and a null "
                         "final — a reader cannot otherwise tell it is partial")
    if list(mid)[0] != "in_progress":
        raise SystemExit("in_progress is not the FIRST key — a human opening "
                         "the file must meet the marker before any number")
    if os.path.exists(p + ".tmp"):
        raise SystemExit("the atomic write left its temp sibling behind")
    tfield.write_result(p, {"k": 1}, [{"step": 9}], final={"done": True})
    end = json.load(open(p))
    if "in_progress" in end or not end["final"]:
        raise SystemExit("the completed write still carries in_progress")

    # (b) the toy smoke, end to end, in both modes
    outs = []
    for mode in ("det", "diff"):
        o = os.path.join(tmp, f"smoke_{mode}.json")
        r = tfield.main(["--toy", "gauss", "--mode", mode, "--smoke",
                         "--out", o, "--quiet"])
        d = json.load(open(o))
        if "in_progress" in d:
            raise SystemExit(f"--smoke ({mode}) left in_progress in the file")
        if not d["final"] or not d["history"]:
            raise SystemExit(f"--smoke ({mode}) wrote no final/history")
        if "smoke_sample_shape" not in d["final"]:
            raise SystemExit(f"--smoke ({mode}) never called sample()")
        outs.append(r)

    # (c) the REAL path, on a synthetic npz, under --smoke
    zp, np_ = synthetic_npz(tmp)
    o = os.path.join(tmp, "real.json")
    tfield.main(["--z-cache", zp, "--data", np_, "--holdout-years", "1992",
                 "--mode", "diff", "--smoke", "--out", o, "--quiet"])
    d = json.load(open(o))
    if "in_progress" in d or not d["final"]:
        raise SystemExit("the real-path smoke did not complete cleanly")
    if d["config"]["law"] != "real" or d["config"]["n_val"] < 1:
        raise SystemExit(f"the real path built no val split: {d['config']}")

    # (d) NaN never reaches a file: the guard exits instead
    try:
        tfield._finite_or_die("synthetic", float("nan"))
    except SystemExit as e:
        if "non-finite" not in str(e):
            raise SystemExit(f"the NaN guard exited with the wrong message: {e}")
    else:
        raise SystemExit("_finite_or_die accepted a NaN — ml/CLAUDE.md §5.22")

    shutil.rmtree(tmp, ignore_errors=True)
    return ("result file: in_progress first then absent, atomic, smoke green "
            "in det/diff/real, NaN refused")


TESTS = [test_tokenizer_roundtrip,
         test_det_init_is_persistence,
         test_diff_init_is_cskip,
         test_sampler_determinism,
         test_land_never_enters_the_loss,
         test_checkpoint_resume_is_bit_identical,
         test_det_beats_persistence_on_shift,
         test_diff_recovers_the_gaussian_conditional,
         test_result_file_discipline]


def main():
    torch.set_num_threads(min(4, os.cpu_count() or 1))
    ok, t_all = 0, time.time()
    for i, fn in enumerate(TESTS, 1):
        t0 = time.time()
        msg = fn()
        print(f"case {i} ok ({time.time() - t0:6.1f}s) — {msg}", flush=True)
        ok += 1
    print(f"\nall {ok}/{len(TESTS)} E-052 field-head identities hold "
          f"({time.time() - t_all:.1f}s)")


if __name__ == "__main__":
    main()

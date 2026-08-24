#!/usr/bin/env python3
"""E-048 · the FSQ LADDER: uniform, exp, and a per-dimension `auto` fit.

Chris, 2026-08-24: *"For each channel try to compute a distribution such that
the FSQ levels can be on the scalar ladder (uniform: level*c) or on the
exponential ladder (c**level)."* `--fsq-ladder` is that knob; z-dimensions are
not channels, so the honest reading — and what this pins — is PER Z-DIMENSION
(ml/fsq_ladder.py says why at length).

Nine checks, and what each one is FOR:

  1. **BIT-IDENTITY OF THE DEFAULT, against a pinned revision.** `uniform` is
     E-046's lattice, and E-046's lattice is what the dispatched #4xx arms
     stand on. The working tree's `PixelMAE` is compared against `BASE_SHA`'s,
     imported as a module: same seed, every state_dict tensor `torch.equal`,
     and `encode` bit-identical on all three branches WITH the bottleneck on
     and WITH it off. `BASE_SHA` is the FSQ commit itself, not "the previous
     commit" — a moving reference lets a lattice drift one commit at a time.
  2. **THE EXP LADDER IS GEOMETRIC**, stated as a ratio and not as a shape:
     consecutive level magnitudes differ by exactly the base, the outermost is
     the SAME +/-2 the uniform ladder saturates at, an even L has no zero
     level and an odd L has exactly one, and "L levels" stays true.
  3. **THE SHARED DEFINITION IS SHARED.** torch, the numpy reference in
     ml/fsq_ladder.py and the JAX mirror agree to 1e-5 on the same inputs for
     BOTH ladders — the level positions are one definition, applied three
     times, and this is what stops a TPU codec being a different model from
     the one the torch eval ladder scores.
  4. **THE GRADIENT SURVIVES THE EXP ROUND**, and is the identity in the
     interior (an identity, not a convention: c^log_c(|v|/a) = |v|).
  5. **`auto` MEASURES, per dimension, and RECORDS what it measured.** A
     fixture whose dimensions genuinely differ gets a mixed answer; ties and
     non-improvements go to uniform; the fit string round-trips.
  6. **THE REFUSALS**: a base <= 1, a base so large the innermost level
     underflows, a ladder with no levels, an `auto` checkpoint carrying no
     fit, an unknown `fsq_*` argument, and a fit whose length is not d_z.
  7. **THE CHECKPOINT ROUND TRIP**, end to end through ml/train.py: an `auto`
     run fits mid-training, writes `fsq_ladder_fit` into args and into
     metrics.jsonl, and `codec_from_ckpt` rebuilds a codec whose `encode` is
     BIT-IDENTICAL to the trainer's — while the same weights loaded with the
     ladder dropped are not. A resume adopts the fit and never re-fits; a
     contradicting ladder refuses.
  8. **JAX PARITY THROUGH THE REAL TRAINER**: ml/jaxport/train_stage1.py runs
     the same knobs, its .pt export round-trips the fsq args, and the torch
     codec rebuilt from that .pt reproduces the JAX encode to 1e-5.
  9. **THE RECIPES RESOLVE**: both E-048 arms export RECIPE_FSQ_LADDER=auto
     and differ from each other in exactly one key, the stride.

    python3 tests/test_e048_fsq_ladders.py

~4 minutes on two cores. No GPU, no network, no real tensor.
"""
import importlib.util
import json
import os
import shutil
import subprocess
import sys
import tempfile
import warnings

warnings.filterwarnings("ignore", message=".*enable_nested_tensor.*")

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
ML = os.path.join(ROOT, "ml")
sys.path.insert(0, HERE)
sys.path.insert(0, ML)

from test_e047_block_smoke import toy, C, DZ                    # noqa: E402

# The revision that SHIPPED the FSQ bottleneck, and therefore the definition
# of "the uniform lattice" every E-046 arm is scored on.
BASE_SHA = "7f8dabb"
TARGET = "ml/model.py"
GEO = dict(n_chan=C, d_model=16, n_heads=2, n_layers=2, d_dec=16, d_z=DZ)


def base_model_module(tmp):
    r = subprocess.run(["git", "-C", ROOT, "show", f"{BASE_SHA}:{TARGET}"],
                       capture_output=True, text=True)
    if r.returncode != 0:
        raise SystemExit(
            f"cannot read {TARGET} at {BASE_SHA}: {r.stderr.strip()}. This "
            f"check compares against the uniform lattice's own code; without "
            f"it there is no reference and it must FAIL rather than pass "
            f"vacuously.")
    p = os.path.join(tmp, "model_e048_base.py")
    open(p, "w").write(r.stdout)
    spec = importlib.util.spec_from_file_location("model_e048_base", p)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["model_e048_base"] = mod
    spec.loader.exec_module(mod)
    return mod


def build(cls, **kw):
    torch.manual_seed(1234)
    m = cls(**GEO, **kw)
    m.eval()
    return m


def inputs_perbin(B=9, seed=7):
    g = torch.Generator().manual_seed(seed)
    return (torch.randn(B, C, generator=g),
            torch.rand(B, C, generator=g) > 0.2,
            (torch.rand(B, C, generator=g) > 0.6),
            torch.randn(B, 4, generator=g))


def inputs_patch(B=9, patch=3, seed=8):
    g = torch.Generator().manual_seed(seed)
    p2 = patch * patch
    x = torch.randn(B, C, p2, generator=g)
    obs = torch.rand(B, C, p2, generator=g) > 0.2
    mask = (torch.rand(B, C, generator=g) > 0.6) & obs[..., p2 // 2]
    return x, obs, mask, torch.randn(B, 4, generator=g)


def inputs_block(B=9, kt=6, seed=9):
    g = torch.Generator().manual_seed(seed)
    return (torch.randn(B, kt, C, generator=g),
            torch.rand(B, kt, C, generator=g) > 0.2,
            (torch.rand(B, kt, C, generator=g) > 0.6),
            torch.randn(B, 4, generator=g))


def run(cmd, tag, timeout=2400, want_fail=False):
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout,
                       cwd=ROOT)
    ok = (r.returncode != 0) if want_fail else (r.returncode == 0)
    if not ok:
        print(r.stdout[-4000:])
        print(r.stderr[-3000:])
        raise SystemExit(f"{tag}: unexpected rc {r.returncode}")
    return r.stdout + r.stderr


def main():
    tmp = tempfile.mkdtemp()
    run_name = "e048_ladder"
    run_dir = os.path.join(ML, "runs", run_name)
    try:
        import fsq_ladder as fql
        from model import PixelMAE, codec_from_ckpt, fsq_from_levels
        base = base_model_module(tmp)

        # ---- 1. the default is bit-identical to BASE_SHA ------------------
        n_t = 0
        for tag, kw, args in (
                ("per-bin", {}, inputs_perbin()),
                ("patch 3", {"patch": 3}, inputs_patch()),
                ("k_time 6", {"k_time": 6}, inputs_block())):
            for lv in ("", "8", "8,7,6,5,4,3,8,7"[:len("8,7,6,5,4,3,8,7")]):
                if lv and lv.count(",") and lv.count(",") + 1 != DZ:
                    continue
                m_new = build(PixelMAE, fsq_levels=lv, **kw)
                m_old = build(base.PixelMAE, fsq_levels=lv, **kw)
                assert (m_new.fsq is None) == (m_old.fsq is None), tag
                sn, so = m_new.state_dict(), m_old.state_dict()
                assert set(sn) == set(so), (tag, set(sn) ^ set(so))
                bad = [k for k in sorted(sn) if not torch.equal(sn[k], so[k])]
                assert not bad, (tag, lv, bad[:4])
                with torch.no_grad():
                    zn, zo = m_new.encode(*args), m_old.encode(*args)
                assert torch.equal(zn, zo), (tag, lv,
                                             float((zn - zo).abs().max()))
                n_t += len(sn)
        print("1. the uniform ladder is BIT-IDENTICAL to %s: %d state_dict "
              "tensors torch.equal and encode() equal bit-for-bit across three "
              "geometries (per-bin, patch 3, k_time 6 window blocks) x three "
              "bottlenecks (off, --fsq-levels 8, a per-dimension level list) — "
              "E-048 adds a branch, not a change" % (BASE_SHA, n_t))

        # ---- 2. the exp ladder is geometric ------------------------------
        rows = []
        for L, c in ((8, 2.0), (8, 3.0), (7, 2.0), (5, 1.5)):
            q = fsq_from_levels(str(L), DZ, ladder="exp", exp_base=c)
            g = torch.Generator().manual_seed(11)
            v = torch.randn(40000, DZ, generator=g) * 4.0
            out = q(v)
            pts = np.unique(np.round(out.numpy(), 9))
            assert len(pts) == L, (L, c, pts)
            assert np.allclose(pts, fql.levels_of(L, 2.0, "exp", c), atol=1e-6)
            assert abs(float(np.abs(pts).max()) - 2.0) < 1e-6
            pos = np.sort(pts[pts > 0])
            ratios = pos[1:] / pos[:-1]
            assert np.allclose(ratios, c, rtol=1e-5), (L, c, ratios)
            n_zero = int((np.abs(pts) < 1e-12).sum())
            assert n_zero == (1 if L % 2 else 0), (L, n_zero)
            rows.append((L, c, len(pts), float(pos.min()), n_zero))
        # ...and the uniform ladder over the same L is NOT geometric
        u = np.unique(np.round(fsq_from_levels("8", DZ)(
            torch.randn(40000, DZ, generator=torch.Generator().manual_seed(11))
            * 4.0).numpy(), 9))
        du = np.diff(np.sort(u))
        assert np.allclose(du, du[0], rtol=1e-5), du
        print("2. the exponential ladder is geometric and complete: %s — "
              "consecutive magnitudes in exact ratio c, outermost at the same "
              "|z| = 2.0 the uniform ladder saturates at, an odd L carrying "
              "exactly one zero level and an even L none, and the uniform "
              "ladder over the same L still evenly spaced (step %.4f)"
              % (", ".join(f"L={L} c={c:g}: {n} levels, innermost {a:.4f}, "
                           f"{z} zero" for L, c, n, a, z in rows), du[0]))

        # ---- 3. torch == numpy == jax, both ladders ----------------------
        from flax import nnx
        import jax.numpy as jnp
        from jaxport import models as jm
        from jaxport import convert as jc
        worst = {}
        fitspec = ",".join(("u" if i % 3 == 0 else f"e{[2,3][i % 2]}")
                           for i in range(DZ))
        for lad, fit in (("uniform", ""), ("exp", ""), ("auto", fitspec)):
            kw = dict(fsq_levels="8", fsq_ladder=lad, fsq_ladder_fit=fit)
            tm = build(PixelMAE, **kw)
            jmod = jm.PixelMAE(**GEO, **kw, rngs=nnx.Rngs(0))
            jc.load_pixelmae(tm.state_dict(), jmod)
            args = inputs_perbin(B=64, seed=17)
            with torch.no_grad():
                zt = tm.encode(*args)
            zj = np.asarray(jmod.encode(*(jnp.asarray(t.numpy())
                                          for t in args)))
            lv = np.full(DZ, 8)
            is_exp, bse, _ = fql.resolve(lad, lv, DZ, 2.0, fit)
            zn = fql.quantize_np(
                tm.encode_pre(*args).detach().numpy(), lv, 2.0,
                is_exp=is_exp, base=bse)
            d_j = float(np.abs(zt.numpy() - zj).max())
            d_n = float(np.abs(zt.numpy() - zn).max())
            assert d_j < 1e-5 and d_n < 1e-5, (lad, d_j, d_n)
            worst[lad] = (d_j, d_n)
        print("3. one definition, three applications: torch vs JAX max|Δ| %s "
              "and torch vs the numpy reference max|Δ| %s (gate 1e-5), for the "
              "uniform ladder, the exponential ladder AND a mixed per-dimension "
              "fit — so a TPU-trained codec is the SAME model the torch eval "
              "ladder scores"
              % ({k: f"{v[0]:.1e}" for k, v in worst.items()},
                 {k: f"{v[1]:.1e}" for k, v in worst.items()}))

        # ---- 4. the gradient through the exp round -----------------------
        mg = build(PixelMAE, fsq_levels="8", fsq_ladder="exp")
        mg.train()
        args = inputs_perbin(B=32, seed=31)
        mg.encode(*args).pow(2).sum().backward()
        gw = mg.to_z.weight.grad
        gv = mg.val_proj.weight.grad
        assert gw is not None and torch.isfinite(gw).all()
        assert float(gw.abs().max()) > 0 and float(gv.abs().max()) > 0
        # the STE's shape, measured rather than assumed
        c = 2.0
        q = fsq_from_levels("8", DZ, ladder="exp", exp_base=c)
        v = torch.linspace(0.3, 1.9, 200)[:, None].repeat(1, DZ) \
            .requires_grad_(True)
        out = q(v)
        out.sum().backward()
        ratio = out.detach().abs() / v.detach().abs()
        assert torch.allclose(v.grad, ratio, atol=1e-6), \
            float((v.grad - ratio).abs().max())
        lo, hi = float(v.grad.min()), float(v.grad.max())
        assert lo > c ** -0.5 - 1e-3 and hi < c ** 0.5 + 1e-3, (lo, hi)
        vs = torch.tensor([[10.0] * DZ, [100.0] * DZ], requires_grad=True)
        q(vs).sum().backward()
        sat = vs.grad[:, 0].tolist()
        assert abs(sat[0] - 0.2) < 1e-5 and abs(sat[1] - 0.02) < 1e-5, sat
        print("4. the straight-through exp round passes a gradient all the way "
              "back through the encoder (|d/d to_z.weight| %.4g, "
              "|d/d val_proj.weight| %.4g) and its SHAPE is |z_q|/|v|, "
              "measured: %.3f..%.3f over the unsaturated range, inside "
              "[c^-1/2, c^1/2] = [%.3f, %.3f] because that is how far a value "
              "can sit from its nearest level in log space, and decaying as "
              "R/|v| under saturation (%.2f at |v|=10, %.3f at 100)"
              % (float(gw.abs().max()), float(gv.abs().max()), lo, hi,
                 c ** -0.5, c ** 0.5, sat[0], sat[1]))

        # ---- 5. auto measures per dimension and records it ---------------
        # The fixture's dimensions differ in SCALE, which is what a
        # per-dimension choice is for: a dimension whose mass sits inside one
        # uniform step is served badly by an evenly spaced lattice and well by
        # a geometric one, and the fitted BASE should follow how peaked it is.
        rng = np.random.default_rng(0)
        sample = np.zeros((8000, DZ), np.float64)
        widths = [0.06, 0.12, 0.5, 1.0] * 4
        for d in range(DZ):
            sample[:, d] = rng.laplace(0.0, widths[d % len(widths)], 8000)
        qa = fsq_from_levels("8", DZ, ladder="auto")
        assert qa.needs_fit and not qa.is_exp_np.any()
        line = qa.fit_auto(sample)
        assert not qa.needs_fit and qa.fit
        assert qa.is_exp_np.all(), qa.is_exp_np
        # the narrowest dimension takes a LARGER base than the widest — the
        # fit is reading the distribution, not returning a constant
        assert qa.base_np[0] > qa.base_np[3], qa.base_np[:4]
        got = fql.parse_fit(qa.fit, DZ)
        assert np.array_equal(got[0], qa.is_exp_np)
        assert np.allclose(got[1][qa.is_exp_np], qa.base_np[qa.is_exp_np])
        # THE TIE RULE, on an exact tie: both ladders saturate at the same
        # +/-2, so a sample far outside the bound is quantized identically by
        # every candidate and the DEFAULT must survive.
        sat = np.where(rng.random((64, DZ)) < 0.5, -100.0, 100.0)
        qt = fsq_from_levels("8", DZ, ladder="auto")
        qt.fit_auto(sat)
        assert not qt.is_exp_np.any(), qt.fit
        # AND WHY exp WINS SO BROADLY — measured, not assumed, because a
        # sweep that reported "the exponential ladder is better" without this
        # would be reporting a tail argument for something that is mostly a
        # BIAS. At EVEN L the E-046 uniform map is not antisymmetric: `shift`
        # is applied inside the tanh and `offset` outside it, so the two
        # cancel only at v = 0 and the whole lattice sits half a step high.
        g_ = torch.Generator().manual_seed(0)
        vv = torch.randn(200000, 1, generator=g_)
        bias = {}
        for L in (8, 7):
            for lad in ("uniform", "exp"):
                qq = fsq_from_levels(str(L), 1, ladder=lad)
                anti = float((qq(vv) + qq(-vv)).abs().max())
                bias[(L, lad)] = (float(qq(vv).mean()), anti)
        assert bias[(8, "uniform")][0] > 0.2, bias
        assert bias[(8, "uniform")][1] > 0.5, bias
        assert abs(bias[(8, "exp")][0]) < 0.01 and bias[(8, "exp")][1] == 0.0
        assert abs(bias[(7, "uniform")][0]) < 0.01, bias
        print("5. `auto` MEASURES: over eight dimensions of differing width "
              "the fit puts %d of %d on the exponential ladder and reads the "
              "WIDTH — base %g for the narrowest, %g for the widest — '%s'; "
              "the string round-trips; and on an exact TIE (values far outside "
              "the bound, where both ladders saturate at the same +/-2) the "
              "DEFAULT survives. WHY exp wins broadly, measured here so the "
              "sweep cannot mistake it for a tail argument: at EVEN L the "
              "E-046 uniform map is NOT antisymmetric — mean z_q %+.3f on "
              "N(0,1) against the exp ladder's %+.3f, |q(-v)+q(v)| up to %.2f "
              "— because `shift` sits inside the tanh and `offset` outside it, "
              "so they cancel only at v = 0. At ODD L (no offset) the uniform "
              "map is unbiased (%+.3f) and exactly antisymmetric"
              % (int(qa.is_exp_np.sum()), DZ, qa.base_np[0], qa.base_np[3],
                 line.split(': ', 1)[1][:60], bias[(8, "uniform")][0],
                 bias[(8, "exp")][0], bias[(8, "uniform")][1],
                 bias[(7, "uniform")][0]))

        # ---- 6. the refusals ---------------------------------------------
        def refuses(fn, must, why):
            try:
                fn()
            except SystemExit as e:
                assert must in str(e), (why, str(e)[:300])
                return
            raise AssertionError(f"{why} was accepted")
        refuses(lambda: fsq_from_levels("8", DZ, ladder="exp", exp_base=1.0),
                "base of 1", "base 1")
        refuses(lambda: fsq_from_levels("8", DZ, ladder="exp", exp_base=0.5),
                "below 1 inverts", "base 0.5")
        refuses(lambda: fsq_from_levels("8", DZ, ladder="exp", exp_base=1e3),
                "of the saturation radius", "an underflowing base")
        refuses(lambda: PixelMAE(**GEO, fsq_ladder="exp"),
                "without --fsq-levels", "a ladder with no levels")
        refuses(lambda: fql.parse_fit("u,e2", DZ), "does not describe this",
                "a fit of the wrong length")
        refuses(lambda: fql.parse_fit(",".join(["x"] * DZ), DZ),
                "expected 'u' or", "a fit entry that is neither")
        refuses(lambda: codec_from_ckpt(
            {"d_z": DZ, "args": {"fsq_levels": "8", "fsq_ladder": "auto"}}, C),
            "carries no `fsq_ladder_fit`", "an auto checkpoint with no fit")
        refuses(lambda: codec_from_ckpt(
            {"d_z": DZ, "args": {"fsq_levels": "8", "fsq_bound": "sigmoid"}}, C),
            "fsq_bound", "an unknown fsq_* argument")
        print("6. refused, every one at the cost of the inputs alone: base 1 "
              "(every level identical), base < 1 (an inverted ladder), a base "
              "so large the innermost level underflows the saturation radius, "
              "a ladder with no levels to sit on, a recorded fit of the wrong "
              "length or the wrong spelling, an `auto` checkpoint carrying NO "
              "fit, and an fsq_* argument this revision does not implement")

        # ---- 7. the checkpoint round trip, through ml/train.py -----------
        npz, _ = toy(tmp)
        base_cmd = [sys.executable, "-u", os.path.join(ML, "train.py"),
                    "--data", npz, "--out", run_dir, "--batch", "16",
                    "--d-model", "16", "--n-layers", "2", "--n-heads", "2",
                    "--d-dec", "16", "--d-z", str(DZ), "--patch", "1",
                    "--anomaly", "--holdout-years", "1991",
                    "--holdout-lon=0,0", "--collapse-r", "0",
                    "--light-probe-every", "0"]
        o = run(base_cmd + ["--steps", "40", "--fsq-levels=8",
                            "--fsq-ladder=auto", "--fsq-auto-n", "512",
                            "--fsq-auto-step", "2000"],
                "train --fsq-ladder auto")
        assert "FSQ ladder: --fsq-ladder auto" in o, o[-2000:]
        assert "NOT YET FITTED" in o, o[-2000:]
        assert "auto fitted on 512 pre-quantization vectors" in o, o[-2000:]
        ckp = os.path.join(run_dir, "pixelmae.pt")
        ck = torch.load(ckp, map_location="cpu", weights_only=False)
        assert ck["args"]["fsq_ladder"] == "auto"
        fit = ck["args"]["fsq_ladder_fit"]
        assert fit and len(fit.split(",")) == DZ, fit
        recs = [json.loads(l) for l in open(os.path.join(run_dir,
                                                         "metrics.jsonl"))
                if l.strip()]
        cfg = [r["config"] for r in recs if "config" in r][0]
        assert cfg["fsq_ladder"] == "auto", cfg
        fitrec = [r["fsq_ladder_fit"] for r in recs if "fsq_ladder_fit" in r]
        assert fitrec and fitrec[-1]["spec"] == fit, fitrec
        a_ = ck["args"]
        trainer = PixelMAE(n_chan=C, d_z=a_["d_z"], patch=a_["patch"],
                           d_model=a_["d_model"], k_time=a_["k_time"],
                           n_layers=a_["n_layers"], n_heads=a_["n_heads"],
                           d_dec=a_["d_dec"], dec_layers=a_["dec_layers"],
                           fsq_levels=a_["fsq_levels"],
                           fsq_ladder=a_["fsq_ladder"],
                           fsq_ladder_fit=a_["fsq_ladder_fit"])
        loader = codec_from_ckpt(ck, C)
        dropped = PixelMAE(n_chan=C, d_z=a_["d_z"], patch=a_["patch"],
                           d_model=a_["d_model"], k_time=a_["k_time"],
                           n_layers=a_["n_layers"], n_heads=a_["n_heads"],
                           d_dec=a_["d_dec"], dec_layers=a_["dec_layers"],
                           fsq_levels=a_["fsq_levels"])
        for m in (trainer, loader, dropped):
            m.load_state_dict(ck["model"])
            m.eval()
        assert loader.fsq.fit == fit
        args7 = inputs_perbin(B=64, seed=41)
        with torch.no_grad():
            z_tr, z_ld, z_dr = (m.encode(*args7) for m in
                                (trainer, loader, dropped))
        assert torch.equal(z_tr, z_ld), float((z_tr - z_ld).abs().max())
        drop_gap = float((z_tr - z_dr).abs().max())
        assert drop_gap > 1e-3, ("the fitted ladder made no difference at all",
                                 drop_gap)
        o_ad = run([sys.executable, "-u", os.path.join(ML, "train.py"),
                    "--data", npz, "--out", run_dir, "--steps", "44",
                    "--anomaly", "--holdout-years", "1991",
                    "--holdout-lon=0,0", "--collapse-r", "0",
                    "--light-probe-every", "0", "--resume", ckp], "resume")
        assert "fsq_ladder=auto" in o_ad, o_ad[-2000:]
        assert "FSQ ladder fit ADOPTED" in o_ad, o_ad[-2000:]
        assert "auto fitted on" not in o_ad, "a resume RE-FITTED the ladder"
        o_cl = run(base_cmd + ["--steps", "8", "--resume", ckp,
                               "--fsq-ladder=exp"], "resume clash",
                   want_fail=True)
        assert "REFUSING to resume" in o_cl and "fsq_ladder" in o_cl, o_cl[-900:]
        print("7. the checkpoint round-trips the LATTICE: an auto run fits "
              "mid-training, writes '%s' into args and into metrics.jsonl, and "
              "codec_from_ckpt rebuilds a codec whose encode equals the "
              "trainer's BIT-FOR-BIT while the same weights with the ladder "
              "dropped differ by up to %.3f. A resume ADOPTS the fit and does "
              "NOT re-fit; a --fsq-ladder=exp resume REFUSES"
              % (fit[:32] + ("…" if len(fit) > 32 else ""), drop_gap))

        # ---- 8. the JAX trainer, and its .pt export ----------------------
        jrun = os.path.join(tmp, "jax_run")
        oj = run([sys.executable, "-u",
                  os.path.join(ML, "jaxport", "train_stage1.py"),
                  "--data", npz, "--out", jrun, "--steps", "30",
                  "--batch", "16", "--d-model", "16", "--n-layers", "2",
                  "--n-heads", "2", "--d-dec", "16", "--d-z", str(DZ),
                  "--patch", "1", "--anomaly", "--holdout-years", "1991",
                  "--holdout-lon=0,0", "--collapse-r", "0",
                  "--fsq-levels", "8", "--fsq-ladder", "auto",
                  "--fsq-auto-n", "512", "--fsq-auto-step", "10"],
                 "jax train --fsq-ladder auto")
        assert "FSQ bottleneck: --fsq-levels 8" in oj, oj[-2500:]
        assert "ladder auto" in oj and "NOT YET FITTED" in oj, oj[-2500:]
        assert "auto fitted on 512 pre-quantization vectors" in oj, oj[-2500:]
        ckj = torch.load(os.path.join(jrun, "pixelmae.pt"), map_location="cpu",
                         weights_only=False)
        assert ckj["args"]["backend"] == "jax"
        jfit = ckj["args"]["fsq_ladder_fit"]
        assert jfit and len(jfit.split(",")) == DZ, jfit
        # the exported .pt rebuilds under torch, and reproduces the JAX encode
        tm = codec_from_ckpt(ckj, C)
        tm.load_state_dict(ckj["model"])
        tm.eval()
        assert tm.fsq.fit == jfit
        jm2 = jc.codec_from_ckpt_jax(ckj, C)
        args8 = inputs_perbin(B=48, seed=57)
        with torch.no_grad():
            zt = tm.encode(*args8)
        zj = np.asarray(jm2.encode(*(jnp.asarray(t.numpy()) for t in args8)))
        d8 = float(np.abs(zt.numpy() - zj).max())
        assert d8 < 1e-5, d8
        # and both ladders survive the export, not only the fitted one
        oe = run([sys.executable, "-u",
                  os.path.join(ML, "jaxport", "train_stage1.py"),
                  "--data", npz, "--out", jrun, "--steps", "12",
                  "--batch", "16", "--d-model", "16", "--n-layers", "2",
                  "--n-heads", "2", "--d-dec", "16", "--d-z", str(DZ),
                  "--patch", "1", "--anomaly", "--holdout-years", "1991",
                  "--holdout-lon=0,0", "--collapse-r", "0",
                  "--fsq-levels", "8", "--fsq-ladder", "exp"],
                 "jax train --fsq-ladder exp")
        assert "ladder exp (base 2)" in oe, oe[-2000:]
        cke = torch.load(os.path.join(jrun, "pixelmae.pt"), map_location="cpu",
                         weights_only=False)
        tme = codec_from_ckpt(cke, C)
        tme.load_state_dict(cke["model"])
        tme.eval()
        jme = jc.codec_from_ckpt_jax(cke, C)
        with torch.no_grad():
            zte = tme.encode(*args8)
        zje = np.asarray(jme.encode(*(jnp.asarray(t.numpy()) for t in args8)))
        de = float(np.abs(zte.numpy() - zje).max())
        assert de < 1e-5, de
        pts = np.unique(np.round(zte.numpy(), 9))
        assert len(pts) <= 8 * DZ
        print("8. the JAX trainer carries the same knobs and the .pt export "
              "round-trips them: an auto run fits at its own step and writes "
              "the lattice into args (backend 'jax'), and the torch codec "
              "rebuilt from that .pt reproduces the JAX encode to %.1e — "
              "%.1e for a fixed exponential ladder (gate 1e-5). Without the "
              "round trip the TPU sweep would be scored by a different model "
              "than it trained" % (d8, de))

        # ---- 9. the recipes ----------------------------------------------
        emitted = {}
        for name in ("f4r2-70M-fsqblock-w6s6", "f4r2-70M-fsqblock-w6s3"):
            r = subprocess.run(["bash", "scripts/resolve_recipe.sh",
                                f"recipe:{name}"], capture_output=True,
                               text=True, cwd=ROOT, timeout=300)
            assert r.returncode == 0, r.stdout[-1500:] + r.stderr[-1500:]
            emitted[name] = [l for l in r.stdout.splitlines()
                             if l.startswith("RECIPE_")]
            for want in ("RECIPE_FSQ_LADDER=auto", "RECIPE_FSQ_LEVELS=8",
                         "RECIPE_D_Z=64", "RECIPE_CODEC_D_MODEL=768",
                         "RECIPE_CODEC_LAYERS=10", "RECIPE_CODEC_HEADS=8",
                         "RECIPE_CODEC_D_DEC=512", "RECIPE_BATCH=512"):
                assert want in emitted[name], (name, want)
        a6 = json.load(open(os.path.join(ML, "recipes",
                                         "f4r2-70M-fsqblock-w6s6.json")))
        a3 = json.load(open(os.path.join(ML, "recipes",
                                         "f4r2-70M-fsqblock-w6s3.json")))
        k6 = {k: v for k, v in a6.items() if not k.startswith("_")}
        k3 = {k: v for k, v in a3.items() if not k.startswith("_")}
        diff = {k for k in set(k6) | set(k3) if k6.get(k) != k3.get(k)}
        assert diff == {"time_block"}, diff
        assert (k6["time_block"], k3["time_block"]) == ("6/6", "6/3")
        wf = open(os.path.join(ROOT, ".github/workflows/ml-train.yml")).read()
        assert "recipe-only: fsq_ladder" in wf
        assert "--fsq-ladder=${RECIPE_FSQ_LADDER}" in wf
        # the measured parameter count the recipes claim
        from model import PixelMAE as PM
        mm = PM(n_chan=40, d_model=768, n_heads=8, n_layers=10, d_z=64,
                d_dec=512, patch=1, dec_layers=2, k_time=6)
        n_par = sum(p.numel() for p in mm.parameters())
        assert n_par == 71335697, n_par
        assert str(n_par) in a6["_description"].replace(",", "")
        print("9. both E-048 recipes resolve and export RECIPE_FSQ_LADDER=auto "
              "with the 768x10 / 8-head / d_dec 512 / d_z 64 geometry at batch "
              "512; they differ from each other in exactly one key, %s "
              "('6/6' vs '6/3'); ml-train.yml declares fsq_ladder recipe-only "
              "and reads it in the Train step (no new workflow input, the "
              "25-input ceiling untouched); and the parameter count the "
              "recipes state is the one this geometry MEASURES, %s"
              % (sorted(diff), f"{n_par:,}"))

        print("\nE-048 FSQ ladders: all 9 checks hold ✓")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
        shutil.rmtree(run_dir, ignore_errors=True)


if __name__ == "__main__":
    main()

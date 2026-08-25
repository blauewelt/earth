#!/usr/bin/env python3
"""E-049 · `--fsq-bound ln`: an INTRINSIC BOUND on the pre-quantization
activation, and `ml/fsq_usage.py`, the instrument that says what the lattice
was actually used for.

Chris, 2026-08-25: *"continue with road B and do that very diligently, and
test it very well with a decoder (a non-linear decoder)."* Road B spends
sixteen bits on a pixel-bin (d_z 6, FSQ [8,8,8,5,5,5]). Two of the things that
can go wrong with that are not visible in any loss curve, and this file pins
both:

  · the ENCODER OUTGROWS THE LATTICE. `to_z` is a free linear map and the
    straight-through estimator hands it no gradient toward the bound, so
    nothing pushes its output inside one. Measured three times before this
    knob existed: e048a collapsed to a constant encoder at |v| ~ 87 against
    R = 2; run-455 — a HEALTHY codec — sat at |v| ~ 3e4, wearing an
    eight-level ladder as a one-bit SIGN CODE while its log said 3 bits/dim;
    e048a2 showed the fitted ladder closes the collapse but not the DRIFT
    (std_med 0.73 -> 20 across 28k steps). At 16 nominal bits, a sign-code
    degeneration leaves 6 and the experiment would be re-measuring a known
    disease.
  · the BOUND IS DROPPED BY A LOADER. It carries no parameter, so a loader
    that ignored `fsq_bound` would build a model whose `load_state_dict`
    succeeds, whose parameter count matches to the byte, and whose every z is
    a different function of the input — E-046's silent-drop failure, one field
    along.

Seven checks:

  1. **THE BOUND BOUNDS, AT THE QUANTIZER'S OWN INPUT.** `to_z` is scaled by
     1e4 — run-455's disease, injected — and what the QUANTIZER receives (a
     capture proxy installed in `model.fsq`, not a hook on some other tensor)
     still has per-vector mean 0 and unit RMS. The same model without the
     bound is measured beside it, so the contrast is a reading, not a claim.
  2. **A BOUNDED CODEC USES ITS ALPHABET; THE UNBOUNDED ONE AT 1e4 DOES NOT.**
     The digits, through `ml/fsq_usage.py`'s arithmetic: bounded z spreads
     over the lattice, the 1e4 z puts ~all of its mass on the two outermost
     levels. This is the sign code, reproduced on demand.
  3. **CHECKPOINT ROUND TRIP**, for `fsq_bound ""` and `"ln"`:
     `codec_from_ckpt` rebuilds a codec whose z_q is BIT-IDENTICAL
     (`torch.equal`) to the trainer's on fixed input — and the same weights
     loaded with the bound DROPPED are not, which is what makes check 3 a
     test rather than a tautology.
  4. **THE REFUSALS**: `--fsq-bound` without `--fsq-levels`, an unknown bound
     spelling, and an OLD `ml/model.py` handed a checkpoint that carries
     `fsq_bound` — in-process and through `ml/train.py`'s own command line,
     where a dispatch would hit them, at the cost of the inputs alone.
  5. **THE DEFAULT IS BIT-IDENTICAL TO `BASE_SHA`**, the revision before this
     change: same seed, every state_dict tensor `torch.equal`, `encode`
     equal bit-for-bit across three geometries x three bottlenecks. This is
     the E-046 archived-checkpoint contract, and it is what says E-049 adds a
     branch rather than a change.
  6. **`ml/fsq_usage.py` RECOVERS THE DIGITS THE QUANTIZER WROTE.** On a
     synthetic d_z 6 / [8,8,8,5,5,5] codec: the recovered digit of every
     value equals the one derived directly from `ml/fsq_ladder.py`'s
     arithmetic (`round(half*tanh(v/R + shift) - off) + half + off`), which
     is an independent path from the script's nearest-point `searchsorted`;
     effective bits are > 0 and <= nominal; and a CONTINUOUS z is REFUSED
     rather than histogrammed.
  7. **THE TRAINER CARRIES IT END TO END**, on the CPU toy: `--fsq-bound ln`
     with `--fsq-ladder auto` fits, logs the fit and a pre-quantization RMS
     of 1, writes `fsq_bound` AND `fsq_ladder_fit` into the checkpoint, and
     `codec_from_ckpt` rebuilds it. The RMS is the bound's invariant; the
     per-dimension `std_med` the recipes quote is NOT 1 on an untrained
     encoder and is not meant to be — a dimension with a large constant
     offset carries its energy in the mean, and LayerNorm fixes the vector's
     RMS, not the sample's per-dimension spread.

Run: python3 tests/test_e049_fsq_bound.py
"""
import importlib.util
import json
import os
import shutil
import subprocess
import sys
import tempfile

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
ML = os.path.join(ROOT, "ml")
sys.path.insert(0, HERE)
sys.path.insert(0, ML)

import fsq_ladder as fql                                        # noqa: E402
from model import PixelMAE, codec_from_ckpt                     # noqa: E402
import fsq_usage                                                # noqa: E402
from test_e047_block_smoke import toy, C                        # noqa: E402

# The last revision in which there was no intrinsic bound. Pinned, like
# test_e046_fsq_codec.py's and test_e048_fsq_ladders.py's: the guarantee
# wanted here is against the ARCHIVE, and "HEAD" stops being the archive the
# moment this change is committed.
BASE_SHA = "d81c258"
TARGET = "ml/model.py"

DZ = 6
LEVELS = "8,8,8,5,5,5"
GEO = dict(n_chan=C, d_model=16, n_heads=2, n_layers=2, d_dec=16, d_z=DZ)


def base_model_module(tmp):
    """`BASE_SHA`'s ml/model.py, importable beside today's."""
    r = subprocess.run(["git", "-C", ROOT, "show", f"{BASE_SHA}:{TARGET}"],
                       capture_output=True, text=True)
    if r.returncode != 0:
        raise SystemExit(
            f"cannot read {TARGET} at {BASE_SHA}: {r.stderr.strip()}. This "
            f"check compares against the unbounded codec's own code; without "
            f"it there is no reference and it must FAIL rather than pass "
            f"vacuously.")
    p = os.path.join(tmp, "model_e049_base.py")
    open(p, "w").write(r.stdout)
    spec = importlib.util.spec_from_file_location("model_e049_base", p)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["model_e049_base"] = mod
    spec.loader.exec_module(mod)
    return mod


def build(cls, seed=1234, **kw):
    torch.manual_seed(seed)
    m = cls(**GEO, **kw)
    m.eval()
    return m


def inputs_perbin(B=64, seed=7):
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(B, C, generator=g)
    obs = torch.rand(B, C, generator=g) > 0.2
    mask = (torch.rand(B, C, generator=g) > 0.6) & obs
    ctx = torch.randn(B, 4, generator=g)
    return x, obs, mask, ctx


def inputs_patch(B=9, patch=3, seed=8):
    g = torch.Generator().manual_seed(seed)
    p2 = patch * patch
    x = torch.randn(B, C, p2, generator=g)
    obs = torch.rand(B, C, p2, generator=g) > 0.2
    mask = (torch.rand(B, C, generator=g) > 0.6) & obs[..., p2 // 2]
    ctx = torch.randn(B, 4, generator=g)
    return x, obs, mask, ctx


def inputs_block(B=9, kt=7, seed=9):
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(B, kt, C, generator=g)
    obs = torch.rand(B, kt, C, generator=g) > 0.2
    mask = (torch.rand(B, kt, C, generator=g) > 0.6) & obs
    ctx = torch.randn(B, 4, generator=g)
    return x, obs, mask, ctx


class Capture:
    """A transparent proxy around the quantizer that records ITS OWN INPUT.

    The claim under test is about what the LATTICE is handed, so the
    measurement is taken there rather than on a tensor that is merely nearby.
    """

    def __init__(self, q):
        self.q, self.seen = q, None

    def __call__(self, z):
        self.seen = z.detach().clone()
        return self.q(z)


def blow_up_to_z(m, factor=1e4):
    """run-455's disease, injected: a `to_z` whose output is 1e4 x too big."""
    with torch.no_grad():
        m.to_z.weight.mul_(factor)
        m.to_z.bias.add_(float(factor))
    return m


def save_ck(path, m, **extra):
    args = dict(d_z=DZ, patch=1, d_model=16, n_layers=2, n_heads=2, d_dec=16,
                dec_layers=2, k_time=1, fsq_levels=m.fsq_levels,
                fsq_ladder=m.fsq_ladder, fsq_exp_base=m.fsq_exp_base,
                fsq_ladder_fit=m.fsq_ladder_fit, fsq_bound=m.fsq_bound)
    args.update(extra)
    torch.save({"model": m.state_dict(), "d_z": DZ, "args": args}, path)
    return args


def run(cmd, tag, timeout=1800, want_fail=False):
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
    run_dir = os.path.join(tmp, "run")
    try:
        # ---- 1. the bound bounds, at the quantizer's own input -----------
        x, obs, mask, ctx = inputs_perbin()
        mb = blow_up_to_z(build(PixelMAE, fsq_levels=LEVELS, fsq_bound="ln"))
        mu = blow_up_to_z(build(PixelMAE, fsq_levels=LEVELS, fsq_bound=""))
        cb, cu = Capture(mb.fsq), Capture(mu.fsq)
        mb.fsq, mu.fsq = cb, cu
        with torch.no_grad():
            mb.encode(x, obs, mask, ctx)
            mu.encode(x, obs, mask, ctx)
        vb, vu = cb.seen, cu.seen
        assert vb is not None and vb.shape == (len(x), DZ), vb.shape
        mean_b = float(vb.mean(-1).abs().max())
        rms_b = vb.pow(2).mean(-1).sqrt()
        assert mean_b < 1e-4, mean_b
        assert float((rms_b - 1.0).abs().max()) < 1e-4, float(rms_b.min())
        absmax_u = float(vu.abs().max())
        assert absmax_u > 1e3, absmax_u
        print("1. the bound bounds AT THE QUANTIZER'S INPUT: with to_z scaled "
              "by 1e4, --fsq-bound ln hands the lattice vectors whose "
              "per-vector |mean| is at most %.2e and whose RMS is 1 to within "
              "%.2e, while the identical model without the bound hands it "
              "|v| up to %.4g — run-455's %s against R = 2, reproduced on "
              "demand" % (mean_b, float((rms_b - 1.0).abs().max()), absmax_u,
                          "3e4" if absmax_u > 1e4 else "saturation"))

        # ---- 2. and therefore the alphabet is used -----------------------
        lv = fql.parse_levels(LEVELS, DZ)
        is_exp, base, scale, _ = fql.resolve("uniform", lv, DZ, 2.0, "",
                                             scale=2.0)
        with torch.no_grad():
            zb = mb.fsq.q(vb).numpy().astype(np.float64)
            zu = mu.fsq.q(vu).numpy().astype(np.float64)
        db, _, rb = fsq_usage.digits_of(zb, lv, is_exp, base, scale)
        du, _, ru = fsq_usage.digits_of(zu, lv, is_exp, base, scale)
        assert float(max(rb.max(), ru.max())) < fsq_usage.LATTICE_TOL
        eff_b = sum(fsq_usage.entropy_bits(np.bincount(db[:, d],
                                                       minlength=int(lv[d])))
                    for d in range(DZ))
        eff_u = sum(fsq_usage.entropy_bits(np.bincount(du[:, d],
                                                       minlength=int(lv[d])))
                    for d in range(DZ))
        out_u = float(np.mean([(du[:, d] == 0).mean()
                               + (du[:, d] == int(lv[d]) - 1).mean()
                               for d in range(DZ)]))
        out_b = float(np.mean([(db[:, d] == 0).mean()
                               + (db[:, d] == int(lv[d]) - 1).mean()
                               for d in range(DZ)]))
        nominal = float(np.log2(lv.astype(float)).sum())
        assert out_u > 0.99, out_u
        assert eff_u <= DZ + 1e-9, eff_u
        assert eff_b > eff_u, (eff_b, eff_u)
        print("2. and the alphabet is therefore USED: the unbounded 1e4 codec "
              "puts %.3f of its mass on the two outermost levels and reads "
              "%.2f effective bits of a nominal %.1f — one bit per dimension, "
              "the SIGN CODE — while the bounded one puts %.3f outside and "
              "reads %.2f. The disease and its absence, on the same weights "
              "and the same input" % (out_u, eff_u, nominal, out_b, eff_b))

        # ---- 3. the checkpoint round trip --------------------------------
        for bound in ("", "ln"):
            m = build(PixelMAE, fsq_levels=LEVELS, fsq_bound=bound)
            blow_up_to_z(m, 12.0)          # a scale the lattice can feel
            p = os.path.join(tmp, f"ck_{bound or 'none'}.pt")
            args = save_ck(p, m)
            ck = torch.load(p, map_location="cpu", weights_only=False)
            m2 = codec_from_ckpt(ck, C)
            m2.load_state_dict(ck["model"])
            m2.eval()
            assert m2.fsq_bound == bound, (m2.fsq_bound, bound)
            with torch.no_grad():
                z1 = m.encode(x, obs, mask, ctx)
                z2 = m2.encode(x, obs, mask, ctx)
            assert torch.equal(z1, z2), (bound, float((z1 - z2).abs().max()))
            if bound:
                # ...and DROPPING it is not a rounding difference.
                bad_args = dict(args)
                bad_args.pop("fsq_bound")
                torch.save({"model": m.state_dict(), "d_z": DZ,
                            "args": bad_args}, p)
                m3 = codec_from_ckpt(torch.load(p, map_location="cpu",
                                                weights_only=False), C)
                m3.load_state_dict(ck["model"])
                m3.eval()
                with torch.no_grad():
                    z3 = m3.encode(x, obs, mask, ctx)
                assert not torch.equal(z1, z3)
                drop = float((z1 - z3).abs().max())
        print("3. the checkpoint round trip is BIT-IDENTICAL for fsq_bound "
              "'' and 'ln' (torch.equal on z_q, fixed input) — and the same "
              "weights with the bound DROPPED differ by up to %.4g, which is "
              "what makes the identity a test rather than a tautology" % drop)

        # ---- 4. the refusals ---------------------------------------------
        for kw, why in ((dict(fsq_bound="ln"), "no levels"),
                        (dict(fsq_levels=LEVELS, fsq_bound="tanh"),
                         "unknown bound"),
                        (dict(fsq_levels=LEVELS, fsq_bound="layernorm"),
                         "near-miss spelling")):
            try:
                PixelMAE(**GEO, **kw)
                raise AssertionError(f"{kw} ({why}) was accepted")
            except SystemExit:
                pass
        # An OLD loader must refuse a checkpoint that carries the bound,
        # rather than rebuilding it without one.
        base_mod = base_model_module(tmp)
        m = build(PixelMAE, fsq_levels=LEVELS, fsq_bound="ln")
        p = os.path.join(tmp, "ck_old.pt")
        save_ck(p, m)
        ck = torch.load(p, map_location="cpu", weights_only=False)
        try:
            base_mod.codec_from_ckpt(ck, C)
            raise AssertionError(f"{BASE_SHA}'s codec_from_ckpt accepted a "
                                 f"checkpoint carrying fsq_bound")
        except SystemExit as e:
            assert "fsq_bound" in str(e), str(e)
        npz, _ = toy(tmp)
        base_cmd = [sys.executable, "-u", os.path.join(ML, "train.py"),
                    "--data", npz, "--out", run_dir, "--batch", "16",
                    "--d-model", "16", "--n-layers", "2", "--n-heads", "2",
                    "--d-dec", "16", "--d-z", str(DZ), "--patch", "1",
                    "--anomaly", "--holdout-years", "1991",
                    "--holdout-lon=0,0", "--collapse-r", "0",
                    "--light-probe-every", "0"]
        o = run(base_cmd + ["--steps", "2", "--fsq-bound=ln"],
                "train --fsq-bound ln with no levels", want_fail=True)
        assert "without --fsq-levels" in o, o[-900:]
        o2 = run(base_cmd + ["--steps", "2", f"--fsq-levels={LEVELS}",
                             "--fsq-bound=tanh"],
                 "train --fsq-bound tanh", want_fail=True)
        assert "invalid choice" in o2 or "expected one of" in o2, o2[-900:]
        print("4. refused: --fsq-bound without --fsq-levels (there is no "
              "lattice to bound the activation FOR, and it would silently "
              "normalize a continuous z instead), an unknown spelling, and "
              "%s's codec_from_ckpt handed a bounded checkpoint — the "
              "unknown-fsq-key clause, cashed for the first time. In-process "
              "AND through ml/train.py's command line." % BASE_SHA)

        # ---- 5. the default is bit-identical to BASE_SHA ------------------
        n_t = 0
        for kw in ({}, dict(fsq_levels="8"), dict(fsq_levels=LEVELS),
                   dict(fsq_levels=LEVELS, fsq_ladder="auto",
                        fsq_ladder_fit="u:2,e2:0.75,u:2,e3:1.5,u:2,u:2")):
            for geo_kw, args_fn in ((dict(), inputs_perbin),
                                    (dict(patch=3), inputs_patch),
                                    (dict(k_time=7), inputs_block)):
                torch.manual_seed(99)
                new = PixelMAE(**GEO, **geo_kw, **kw).eval()
                torch.manual_seed(99)
                old = base_mod.PixelMAE(**GEO, **geo_kw, **kw).eval()
                sd_n, sd_o = new.state_dict(), old.state_dict()
                assert sd_n.keys() == sd_o.keys(), (sd_n.keys() ^ sd_o.keys())
                for k in sd_n:
                    assert torch.equal(sd_n[k], sd_o[k]), k
                    n_t += 1
                args = args_fn()
                with torch.no_grad():
                    assert torch.equal(new.encode(*args), old.encode(*args))
                    assert torch.equal(new.encode_pre(*args),
                                       old.encode_pre(*args))
        print("5. the DEFAULT (fsq_bound '') is bit-identical to %s: %d "
              "state_dict tensors torch.equal and encode()/encode_pre() equal "
              "bit-for-bit across three geometries (per-bin, patch 3, k_time 7 "
              "blocks) x four bottlenecks (off, [8]^6, a per-dimension list, a "
              "fitted auto ladder) — E-049 adds a branch, not a change, and "
              "the E-046 archived-checkpoint contract holds"
              % (BASE_SHA, n_t))

        # ---- 6. ml/fsq_usage.py recovers the digits ----------------------
        rng = np.random.default_rng(0)
        v = rng.standard_normal((4000, DZ)) * np.array([0.4, 1.0, 2.5,
                                                        1.0, 0.7, 3.0])
        mq = build(PixelMAE, fsq_levels=LEVELS)
        with torch.no_grad():
            zq = mq.fsq(torch.as_tensor(v, dtype=torch.float32))
        zq_np = zq.numpy().astype(np.float32)
        # The EXPECTED digit, from ml/fsq_ladder.py's arithmetic directly and
        # not from the script's nearest-point search: the uniform ladder's own
        # round, shifted into a 0-based index.
        half, off, shift = fql.uniform_params(lv)
        g = half * np.tanh(v / 2.0 + shift) - off
        want = np.rint(np.rint(g) + half + off).astype(np.int64)
        zp = os.path.join(tmp, "z.npy")
        np.save(zp, zq_np)
        cp = os.path.join(tmp, "ck_usage.pt")
        save_ck(cp, mq)
        jp = os.path.join(tmp, "usage.json")
        res = fsq_usage.main(["--ckpt", cp, "--z", zp, "--json", jp])
        got, _, resid = fsq_usage.digits_of(zq_np.astype(np.float64), lv,
                                            is_exp, base, scale)
        assert (got == want).all(), int((got != want).sum())
        assert res["nominal_bits"] == nominal
        assert 0.0 < res["effective_bits"] <= res["nominal_bits"] + 1e-9
        for r in res["dims"]:
            assert 0.0 <= r["bits"] <= r["nominal_bits"] + 1e-9, r
            assert sum(r["hist"]) == res["n_vectors"], r
        assert json.load(open(jp))["effective_bits"] == res["effective_bits"]
        # a CONTINUOUS z is not on any lattice, and is refused
        np.save(zp, v.astype(np.float32))
        try:
            fsq_usage.main(["--ckpt", cp, "--z", zp])
            raise AssertionError("fsq_usage accepted a continuous z")
        except SystemExit as e:
            assert "not on this checkpoint's lattice" in str(e), str(e)
        # and so is a continuous CHECKPOINT
        mc = build(PixelMAE)
        cc = os.path.join(tmp, "ck_cont.pt")
        save_ck(cc, mc)
        try:
            fsq_usage.main(["--ckpt", cc, "--z", zp])
            raise AssertionError("fsq_usage accepted a continuous codec")
        except SystemExit as e:
            assert "CONTINUOUS" in str(e), str(e)
        print("6. ml/fsq_usage.py recovers the quantizer's OWN digits on all "
              "%d x %d values (worst |z - level| %.2e R against a tolerance "
              "of %.2e), agreeing with fsq_ladder's arithmetic derived "
              "independently of its searchsorted; effective %.3f bits of a "
              "nominal %.1f, every per-dimension entropy inside its own "
              "log2(L); a continuous z and a continuous checkpoint are both "
              "REFUSED" % (len(v), DZ, float(resid.max()),
                           fsq_usage.LATTICE_TOL, res["effective_bits"],
                           res["nominal_bits"]))

        # ---- 7. the trainer carries it end to end ------------------------
        shutil.rmtree(run_dir, ignore_errors=True)
        o = run(base_cmd + ["--steps", "6", f"--fsq-levels={LEVELS}",
                            "--fsq-ladder=auto", "--fsq-auto-step=1,2",
                            "--fsq-bound=ln"], "train --fsq-bound ln")
        assert "INTRINSIC BOUND --fsq-bound ln" in o, o[-2000:]
        fit_lines = [l for l in o.splitlines() if "auto fitted on" in l]
        assert len(fit_lines) == 2, fit_lines
        ckp = os.path.join(run_dir, "pixelmae.pt")
        ck = torch.load(ckp, map_location="cpu", weights_only=False)
        assert ck["args"]["fsq_bound"] == "ln", ck["args"].get("fsq_bound")
        assert ck["args"]["fsq_ladder_fit"], ck["args"]
        mt = codec_from_ckpt(ck, ck["args"].get("C") or C)
        assert mt.fsq_bound == "ln" and mt.fsq is not None
        fits = [json.loads(l)["fsq_ladder_fit"]
                for l in open(os.path.join(run_dir, "metrics.jsonl"))
                if "fsq_ladder_fit" in l]
        rms = [f["prequant_rms"] for f in fits]
        stds = [f["prequant_std_med"] for f in fits]
        assert len(rms) == 2 and all(abs(r - 1.0) < 1e-3 for r in rms), rms
        print("7. the trainer carries it end to end on the CPU toy: two auto "
              "fits at --fsq-auto-step 1,2, pre-quantization rms %s — 1 at "
              "every fit to within LayerNorm's own eps (1e-5), which is the "
              "bound's invariant — beside "
              "std_med %s, which is NOT 1 and is not meant to be (LayerNorm "
              "fixes each VECTOR's rms; an untrained encoder's dimensions "
              "carry their energy in the mean). The checkpoint carries "
              "fsq_bound 'ln' AND its fitted lattice (%s), and "
              "codec_from_ckpt rebuilds both"
              % (rms, stds, ck["args"]["fsq_ladder_fit"]))

        print("\nE-049 fsq_bound + fsq_usage: all 7 checks hold ✓")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
        shutil.rmtree(run_dir, ignore_errors=True)


if __name__ == "__main__":
    main()

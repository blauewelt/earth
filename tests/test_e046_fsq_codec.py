#!/usr/bin/env python3
"""E-046 · the FSQ bottleneck for the stage-1 codec.

`--fsq-levels` replaces the codec's continuous `d_z`-dimensional bottleneck
with per-dimension rounding to a fixed set of levels (FSQ, arxiv 2309.15505),
gradient straight through, no codebook and therefore no commitment loss, no
EMA and no dead-code machinery. The dispatched arm is `8` at `d_z` 32 —
[8]^32 — because that is the alphabet E-045-A9 (#446) actually measured on
this z (20k ratio 0.4916 against the 0.50560/0.50447 control pair); the
plan's option A ([8,8,8,6,5] at d_z 5) would change the WIDTH and the ALPHABET
in one arm, on a codec whose round trip already collapses at d_z 32 where 40
channels compete for 32 dimensions. See ml/model.py for the full reasoning.

Eight checks, and what each one is FOR:

  1. **BIT-IDENTITY WITH THE FLAG OFF, against a pinned revision.** Every
     archived pentad and monthly codec has a continuous bottleneck, and this
     change edits the one line where every z in the programme is born. The
     working tree's `PixelMAE` is compared against `BASE_SHA`'s, IMPORTED as a
     module rather than run as a job: same seed, same construction, every
     `state_dict` tensor `torch.equal`, and `encode` bit-identical on all
     three of its branches (per-bin, patch>1, k_time>1). `BASE_SHA` is a fixed
     anchor and not "the previous commit" — a moving reference would let the
     continuous path drift one commit at a time.
  2. **THE LATTICE IS REAL.** With `--fsq-levels 8` every output dimension
     takes at most 8 distinct values over random input, end to end through
     `encode`, and the bound saturates rather than clipping.
  3. **EXACT ZEROS ARE NOT SPECIAL-CASED HERE, and that is deliberate.**
  4. **The gradient reaches the encoder through the round.**
  5. **The refusals**: L=2, and a comma list whose length is not `d_z` —
     in-process and through `ml/train.py`'s own command line.
  6. **THE CHECKPOINT ROUND TRIP.** `train.py` writes the levels into
     `ck["args"]`, `codec_from_ckpt` reads them back, and the loader's codec
     reproduces the trainer's `encode` BIT-FOR-BIT — while the same weights
     loaded WITHOUT the bottleneck (the silent-drop failure this exists to
     prevent) produce something else entirely. Plus: a resume adopts the
     levels, a contradicting resume refuses, and a checkpoint carrying an
     `fsq_*` argument this revision does not implement is refused.
  7. **The recipe resolves**: `recipe:f4r2-40M-fsq8` exports
     `RECIPE_FSQ_LEVELS=8` and differs from the baseline recipe in that key
     and nothing else.
  8. **A 300-STEP CPU SMOKE**: the quantized codec actually TRAINS — the loss
     falls, which no unit test above can say.

    python3 tests/test_e046_fsq_codec.py

~1-2 minutes on two cores. No GPU, no network, no real tensor.
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

from test_e047_block_smoke import toy, C, DZ, H, W            # noqa: E402

# The last revision in which the codec bottleneck was continuous, full stop.
# Pinned, like tests/test_roll_monthly_identity.py's: the guarantee wanted
# here is against the ARCHIVE, and "HEAD" stops being the archive the moment
# this change is committed.
BASE_SHA = "0e10253"
TARGET = "ml/model.py"


def base_model_module(tmp):
    """`BASE_SHA`'s ml/model.py, importable beside today's."""
    r = subprocess.run(["git", "-C", ROOT, "show", f"{BASE_SHA}:{TARGET}"],
                       capture_output=True, text=True)
    if r.returncode != 0:
        raise SystemExit(
            f"cannot read {TARGET} at {BASE_SHA}: {r.stderr.strip()}. This "
            f"check compares against the continuous bottleneck's own code; "
            f"without it there is no reference and it must FAIL rather than "
            f"pass vacuously.")
    p = os.path.join(tmp, "model_e046_base.py")
    open(p, "w").write(r.stdout)
    spec = importlib.util.spec_from_file_location("model_e046_base", p)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["model_e046_base"] = mod
    spec.loader.exec_module(mod)
    return mod


GEO = dict(n_chan=C, d_model=16, n_heads=2, n_layers=2, d_dec=16, d_z=DZ)


def build(cls, **kw):
    torch.manual_seed(1234)
    m = cls(**GEO, **kw)
    m.eval()
    return m


def inputs_perbin(B=9, seed=7):
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
    run_name = "e046_fsq"
    run_dir = os.path.join(ML, "runs", run_name)
    try:
        from model import PixelMAE, InputQuant, codec_from_ckpt, fsq_from_levels
        base = base_model_module(tmp)

        # ---- 1. bit-identity with the flag off ---------------------------
        n_t = 0
        for tag, kw, args in (
                ("per-bin", {}, inputs_perbin()),
                ("patch 3", {"patch": 3}, inputs_patch()),
                ("k_time 7", {"k_time": 7}, inputs_block())):
            m_new, m_old = build(PixelMAE, **kw), build(base.PixelMAE, **kw)
            assert m_new.fsq is None, tag
            sn, so = m_new.state_dict(), m_old.state_dict()
            assert set(sn) == set(so), (tag, set(sn) ^ set(so))
            bad = [k for k in sorted(sn) if not torch.equal(sn[k], so[k])]
            assert not bad, (tag, bad[:4])
            with torch.no_grad():
                zn, zo = m_new.encode(*args), m_old.encode(*args)
            assert torch.equal(zn, zo), (tag, float((zn - zo).abs().max()))
            n_t += len(sn)
        print("1. flag off is BIT-IDENTICAL to %s: all %d state_dict tensors "
              "torch.equal across three geometries, and encode() is equal "
              "bit-for-bit on all three of its branches (per-bin, patch 3, "
              "k_time 7 month blocks) — the E-047 block codec composes with "
              "this untouched, because every branch ends at the same to_z"
              % (BASE_SHA, n_t))

        # ---- 2. the lattice is real --------------------------------------
        L = 8
        mq = build(PixelMAE, **{})
        mq.fsq_levels, mq.fsq = str(L), fsq_from_levels(str(L), DZ)
        g = torch.Generator().manual_seed(11)
        zraw = torch.randn(4000, DZ, generator=g) * 5.0
        zq = mq.fsq(zraw)
        per_dim = [len(torch.unique(zq[:, d])) for d in range(DZ)]
        assert max(per_dim) <= L, per_dim
        assert min(per_dim) == L, per_dim          # and all L are REACHED
        assert float(zq.abs().max()) <= 2.0 + 1e-6, float(zq.abs().max())
        # end to end, through the real encoder
        x, obs, mask, ctx = inputs_perbin(B=600, seed=21)
        with torch.no_grad():
            ze = mq.encode(x, obs, mask, ctx)
        e_dim = [len(torch.unique(ze[:, d])) for d in range(DZ)]
        assert max(e_dim) <= L, e_dim
        cb = mq.fsq.codebook_log2
        assert abs(cb - DZ * np.log2(L)) < 1e-9, cb
        print("2. --fsq-levels 8 at d_z %d is a real lattice: every dimension "
              "takes at most 8 distinct values on random input (all 8 reached; "
              "%d..%d distinct through the live encoder over 600 pixels), the "
              "bound saturates at |z| <= 2.0, and the codebook is 2^%.0f"
              % (DZ, min(e_dim), max(e_dim), cb))

        # ---- 3. exact zeros are NOT special-cased here -------------------
        z0 = mq.fsq(torch.zeros(5, DZ))
        assert float(z0.abs().min()) > 0.0, z0[0]
        assert len(torch.unique(z0)) == 1, z0[0]
        # ...whereas the HEAD-side knob keeps them, and must:
        iq = InputQuant("8", np.full(DZ, 2.0, np.float32), DZ)
        assert float(iq(torch.zeros(5, DZ)).abs().max()) == 0.0
        print("3. an exactly-zero PRE-QUANTIZATION activation rounds onto the "
              "lattice here (|z_q| = %.4f, not 0) while ml/temporal.py's "
              "--input-quant still passes zeros through — the passthrough is a "
              "STENCIL convention (a zero slot is an ABSENT NEIGHBOUR, "
              "zj[miss]=0.0 in rollout_spatial, and an even L has no zero "
              "level), and the codec's bottleneck has no such convention: "
              "to_z's output is dense, so 0.0 there is an ordinary value"
              % float(z0.abs().min()))

        # ---- 4. the gradient reaches the encoder -------------------------
        mg = build(PixelMAE, **{})
        mg.fsq_levels, mg.fsq = "8", fsq_from_levels("8", DZ)
        mg.train()
        x, obs, mask, ctx = inputs_perbin(B=32, seed=31)
        mg.encode(x, obs, mask, ctx).pow(2).sum().backward()
        gw = mg.to_z.weight.grad
        assert gw is not None and torch.isfinite(gw).all()
        assert float(gw.abs().max()) > 0, "no gradient through the round"
        gv = mg.val_proj.weight.grad
        assert gv is not None and float(gv.abs().max()) > 0, \
            "the gradient stopped at to_z and never reached the encoder"
        print("4. the straight-through round passes a gradient all the way "
              "back through the encoder: |d loss/d to_z.weight| max %.4g and "
              "|d loss/d val_proj.weight| max %.4g, both finite — round() is "
              "the derivative of the BOUND, not zero"
              % (float(gw.abs().max()), float(gv.abs().max())))

        # ---- 5. the refusals ---------------------------------------------
        for bad_spec, why in (("2", "L=2"),
                              ("8,8", "wrong-length list"),
                              ("8,8,8,3,2", "an L=2 inside a list"),
                              ("", None)):
            if bad_spec == "":
                assert fsq_from_levels("", DZ) is None
                continue
            try:
                PixelMAE(**GEO, fsq_levels=bad_spec)
                raise AssertionError(f"{bad_spec!r} ({why}) was accepted")
            except SystemExit:
                pass
        npz, _ = toy(tmp)
        base_cmd = [sys.executable, "-u", os.path.join(ML, "train.py"),
                    "--data", npz, "--out", run_dir, "--batch", "16",
                    "--d-model", "16", "--n-layers", "2", "--n-heads", "2",
                    "--d-dec", "16", "--d-z", str(DZ), "--patch", "1",
                    "--anomaly", "--holdout-years", "1991",
                    "--holdout-lon=0,0", "--collapse-r", "0",
                    "--light-probe-every", "0"]
        o = run(base_cmd + ["--steps", "2", "--fsq-levels=8,8"],
                "train --fsq-levels 8,8", want_fail=True)
        assert "gives 2 level counts for d_z 8" in o, o[-800:]
        o2 = run(base_cmd + ["--steps", "2", "--fsq-levels=2"],
                 "train --fsq-levels 2", want_fail=True)
        assert "must be >= 3" in o2, o2[-800:]
        print("5. refused: L=2 (atanh(1) = inf collapses every value to one), "
              "a list of the wrong length, and an L=2 buried inside a list — "
              "in-process AND through ml/train.py's own command line, which is "
              "where a dispatch would hit it, at the cost of the inputs alone")

        # ---- 6. the checkpoint round trip --------------------------------
        o = run(base_cmd + ["--steps", "8", "--fsq-levels=8"],
                "train --fsq-levels 8")
        assert "FSQ bottleneck: --fsq-levels 8" in o, o[-1500:]
        assert "3.000 bits/dim" in o and "codebook 2^" in o, o[-1500:]
        ckp = os.path.join(run_dir, "pixelmae.pt")
        ck = torch.load(ckp, map_location="cpu", weights_only=False)
        assert ck["args"]["fsq_levels"] == "8", ck["args"].get("fsq_levels")
        cfg = json.loads(open(os.path.join(run_dir, "metrics.jsonl")
                              ).readline())["config"]
        assert cfg["fsq_levels"] == "8", cfg
        # the loader's codec == the trainer's codec, bit for bit
        a_ = ck["args"]
        trainer = PixelMAE(n_chan=C, d_z=a_["d_z"], patch=a_["patch"],
                           d_model=a_["d_model"], k_time=a_["k_time"],
                           n_layers=a_["n_layers"], n_heads=a_["n_heads"],
                           d_dec=a_["d_dec"], dec_layers=a_["dec_layers"],
                           fsq_levels=a_["fsq_levels"])
        loader = codec_from_ckpt(ck, C)
        dropped = PixelMAE(n_chan=C, d_z=a_["d_z"], patch=a_["patch"],
                           d_model=a_["d_model"], k_time=a_["k_time"],
                           n_layers=a_["n_layers"], n_heads=a_["n_heads"],
                           d_dec=a_["d_dec"], dec_layers=a_["dec_layers"])
        for m in (trainer, loader, dropped):
            m.load_state_dict(ck["model"])
            m.eval()
        assert loader.fsq is not None and loader.fsq_levels == "8"
        args6 = inputs_perbin(B=64, seed=41)
        with torch.no_grad():
            z_tr, z_ld, z_dr = (m.encode(*args6) for m in
                                (trainer, loader, dropped))
        assert torch.equal(z_tr, z_ld), float((z_tr - z_ld).abs().max())
        drop_gap = float((z_tr - z_dr).abs().max())
        assert drop_gap > 1e-3, drop_gap
        # an fsq_* argument this revision does not implement is REFUSED
        ck_bad = dict(ck)
        ck_bad["args"] = dict(ck["args"], fsq_bound="sigmoid")
        try:
            codec_from_ckpt(ck_bad, C)
            raise AssertionError("an unknown fsq_* arg was ignored")
        except SystemExit as e:
            assert "fsq_bound" in str(e), e
        # resume ADOPTS the levels, and a contradiction REFUSES
        o_ad = run([sys.executable, "-u", os.path.join(ML, "train.py"),
                    "--data", npz, "--out", run_dir, "--steps", "8",
                    "--anomaly", "--holdout-years", "1991",
                    "--holdout-lon=0,0", "--collapse-r", "0",
                    "--light-probe-every", "0", "--resume", ckp], "resume")
        assert "fsq_levels=8" in o_ad, o_ad[-1500:]
        o_cl = run(base_cmd + ["--steps", "8", "--resume", ckp,
                               "--fsq-levels=6"], "resume clash",
                   want_fail=True)
        assert "REFUSING to resume" in o_cl and "fsq_levels" in o_cl, o_cl[-900:]
        print("6. the checkpoint round-trips the bottleneck: args and the "
              "metrics config line carry fsq_levels '8', codec_from_ckpt "
              "rebuilds it, and the loader's encode equals the trainer's "
              "BIT-FOR-BIT while the same weights loaded WITHOUT the "
              "quantizer differ by up to %.3f — the silent-drop eval this "
              "guards. A resume adopts the levels, a --fsq-levels=6 resume "
              "REFUSES, and an unknown fsq_* arg refuses rather than loading"
              % drop_gap)

        # ---- 7. the recipe -----------------------------------------------
        r = subprocess.run(["bash", "scripts/resolve_recipe.sh",
                            "recipe:f4r2-40M-fsq8"], capture_output=True,
                           text=True, cwd=ROOT, timeout=300)
        assert r.returncode == 0, r.stdout[-1500:] + r.stderr[-1500:]
        emitted = [l for l in r.stdout.splitlines() if l.startswith("RECIPE_")]
        assert "RECIPE_FSQ_LEVELS=8" in emitted, emitted
        assert "RECIPE_D_Z=32" in emitted, emitted
        assert "RECIPE_NAME=f4r2-40M-fsq8" in emitted, emitted
        a_new = json.load(open(os.path.join(ML, "recipes",
                                            "f4r2-40M-fsq8.json")))
        a_old = json.load(open(os.path.join(ML, "recipes",
                                            "f4r2-40M-nolonhold.json")))
        keys_new = {k: v for k, v in a_new.items() if not k.startswith("_")}
        keys_old = {k: v for k, v in a_old.items() if not k.startswith("_")}
        diff = {k for k in set(keys_new) | set(keys_old)
                if keys_new.get(k) != keys_old.get(k)}
        assert diff == {"fsq_levels"}, diff
        wf = open(os.path.join(ROOT, ".github/workflows/ml-train.yml")).read()
        assert "recipe-only: fsq_levels" in wf
        assert "--fsq-levels=${RECIPE_FSQ_LEVELS}" in wf
        print("7. `recipe:f4r2-40M-fsq8` resolves and exports "
              "RECIPE_FSQ_LEVELS=8 (with d_z 32 and the rest of the baseline "
              "intact); the recipe differs from f4r2-40M-nolonhold in exactly "
              "one key, %s; and ml-train.yml declares it recipe-only and reads "
              "it in the Train step — no new workflow input, the 25-input "
              "ceiling untouched" % sorted(diff))

        # ---- 8. it TRAINS ------------------------------------------------
        shutil.rmtree(run_dir, ignore_errors=True)
        run(base_cmd + ["--steps", "300", "--fsq-levels=8"], "300-step smoke")
        recs = [json.loads(l) for l in
                open(os.path.join(run_dir, "metrics.jsonl")) if l.strip()]
        curve = [r_["loss_rec"] for r_ in recs if "loss_rec" in r_]
        assert len(curve) >= 60, len(curve)
        n = len(curve) // 3
        first, last = float(np.mean(curve[:n])), float(np.mean(curve[-n:]))
        assert np.isfinite(curve).all(), "a non-finite loss"
        assert last < first, (first, last)
        ckf = torch.load(os.path.join(run_dir, "pixelmae.pt"),
                         map_location="cpu", weights_only=False)
        mf = codec_from_ckpt(ckf, C)
        mf.load_state_dict(ckf["model"])
        mf.eval()
        with torch.no_grad():
            zf = mf.encode(*inputs_perbin(B=400, seed=55))
        used = [len(torch.unique(zf[:, d])) for d in range(DZ)]
        print("8. 300 CPU steps with --fsq-levels 8: the quantized codec "
              "TRAINS — loss_rec falls from %.4f (first third) to %.4f (last "
              "third) over %d logged points, every value finite — and the "
              "trained codec's z still lives on the lattice (%d..%d distinct "
              "values per dimension of the 8 available)"
              % (first, last, len(curve), min(used), max(used)))

        print("\nE-046 FSQ codec: all 8 checks hold ✓")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
        shutil.rmtree(run_dir, ignore_errors=True)


if __name__ == "__main__":
    main()

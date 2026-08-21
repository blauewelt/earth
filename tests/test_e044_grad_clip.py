#!/usr/bin/env python3
"""E-044 · gradient clipping in ml/temporal.py: the effect, and the no-op.

WHY THIS FILE EXISTS. #423 (E-044, the first stage-2 head at any cadence but
monthly) diverged from its step-2,000 best and was cancelled at step 28,000.
Its grad norm went 8.24 -> 787 -> 3,891 -> 13,052 and after step 6,000 never
came back below 1,000, and
`grep -n "clip_grad\\|max_norm\\|grad_clip" ml/temporal.py` returned NOTHING:
the trainer had no gradient clipping of any kind. The monthly regime never
needed one — the `ml-metrics` branch holds 8,080 logged stage-2 grad norms over
83 runs, EVERY ONE of them at val_persistence 3.09512 (median 0.566, p99 4.279,
p99.9 14.448, max 39.6165 on #308) — so the gap was invisible until the run
left that one z-space for the first time.

WHAT IS UNDER TEST, and what deliberately is NOT.

  1. THE MECHANISM, WITH EXACT EXPECTED VALUES (§4.9). A gradient at
     monthly scale passes through `clip_grad_norm_` BIT-FOR-BIT unchanged, and
     an exploding one comes out at exactly `max_norm` with its DIRECTION
     preserved. Both are checked on the real call the trainer makes, against
     numbers taken from the archive (#426's 1.2439, #423's 13,051.751), not
     against "something smaller".
  2. THE EFFECT, END TO END. The real `temporal.py` on a toy whose window pool
     is heavy-tailed: the clip binds, it is REPORTED as binding, and the
     parameter trajectory it produces is bounded against the unclipped one.
  3. THE NO-OP. With `--grad-clip` at its default 0 the trainer is the
     pre-2026-08-21 code BIT FOR BIT — checked by running the pinned parent
     revision of `ml/temporal.py` and the working tree on the same toy with the
     same seed and comparing every parameter tensor with `torch.equal`. Every
     archived monthly stage-2 number must stay reproducible, and #426 is live
     against this file while it is being edited.

  NOT under test: an end-to-end reproduction of #423's *divergence*. A 300-step
  run of a 3-layer head on a 40-bin toy cannot honestly produce the
  6,000-step-deep AdamW second-moment poisoning that destroyed a 206.5M head,
  and a fixture tuned until it "diverges" would be demonstrating the tuning.
  Check 1 pins the arithmetic that the divergence argument RESTS on instead:
  one step at 13,051.8 = 1,582x the healthy 8.25 grows AdamW's v by 2,503x
  (sqrt(v) 50x, so ~1,000 steps of honest gradients divided by 50), and the
  same step clipped to 128.0 grows it by 1.24x (sqrt(v) 1.11x). That ratio is
  the whole reason the fix is a clip and not a lower learning rate, and it is
  computable exactly.

No pytest. Numbered checks, printed.

    python3 tests/test_e044_grad_clip.py
"""
import json
import math
import os
import shutil
import subprocess
import sys
import tempfile

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ML = os.path.join(HERE, "..", "ml")
sys.path.insert(0, ML)
from model import PixelMAE                                    # noqa: E402

T_M, H_G, W_G, C, DZ, K = 40, 8, 10, 5, 4, 6
# The toy's z-space is deliberately LARGE, because that is one of the two
# things that differ between the healthy monthly arm and #423: the codec's z
# projection is scaled by Z_GAIN before it is frozen, which multiplies every
# embedding — and therefore the persistence MSE, the loss and the gradient —
# without touching the architecture, the data or the schedule.
Z_GAIN = 40.0
# ...and its window pool is deliberately HEAVY-TAILED, which is the other. Note
# that the two are NOT the same thing and only the second can destabilise
# anything: AdamW is invariant to a global gradient rescale (multiply the loss
# by any constant and the trajectory moves by ~1e-4 relative), so "the pentad
# gradients are 3.5x bigger" is a consequence of the z-space, not a cause of
# the divergence. What CAN destabilise is a pool in which a rare batch is
# orders of magnitude worse than a typical one. SPIKE_T gives a few time steps
# an enormous amplitude so that some draws are far outside the bulk.
SPIKE_T = (11, 23, 31)
SPIKE = 60.0
# The PRE-FIX revision, pinned by SHA rather than named HEAD: once the fix is
# committed HEAD carries the clip, and a no-op proof that compares the fix
# against itself proves nothing. 877ae5b is the commit that was HEAD while
# `grep -n "clip_grad" ml/temporal.py` still returned nothing.
PREFIX_SHA = "877ae5b"
LR = 0.05          # a high rate, so 300 steps is enough to see the whole story
STEPS = 300
CLIP = 50.0
# A threshold no gradient in this toy can reach. Turning the clip ON here must
# change NOTHING — clip_grad_norm_ clamps its coefficient at 1.0 and then
# multiplies by exactly 1.0, which is bit-exact for every finite float. That is
# the archive-comparability guarantee in its strongest form: it holds for any
# threshold that does not bind, not merely for the off-by-default path.
NONBIND = 1e9
# The two archive readings check 1 uses, so that "monthly scale" and
# "exploding" are the programme's own numbers and not invented ones.
MONTHLY_NORM = 1.2439          # #426 (E-043b-SEED1) at stage-2 step 40,000
PENTAD_BLOWUP = 13051.751      # #423 (E-044) at stage-2 step 24,000
HEALTHY_PENTAD = 8.25          # #423 at steps 2,000 and 4,000 (8.2372/8.2483)


def toy(tmp):
    """A synthetic ocean + a frozen codec whose z is Z_GAIN times too large."""
    rng = np.random.default_rng(0)
    t = np.arange(T_M)[:, None, None, None]
    X = (np.sin(2 * np.pi * t / 12) + 0.4 * (t / T_M)
         + 0.3 * rng.standard_normal((T_M, H_G, W_G, C))).astype(np.float32)
    X[SPIKE_T, :, :, :] *= SPIKE                    # the heavy tail (see above)
    X[:, 0, 0, :] = np.nan                          # land, so OBS is exercised
    months = np.array([f"{1990 + i // 12}-{i % 12 + 1:02d}" for i in range(T_M)])
    ridx = np.arange(K, T_M)
    rapid = np.stack([ridx.astype(float),
                      2.79 * rng.standard_normal(len(ridx))], 1)
    npz = os.path.join(tmp, "toy.npz")
    np.savez(npz, X=X, months=months, rapid=rapid,
             chan=np.array([f"c{i}" for i in range(C)]),
             lats=np.linspace(20, 40, H_G).astype(np.float32),
             lons=np.linspace(-60, -40, W_G).astype(np.float32))
    torch.manual_seed(0)
    codec = PixelMAE(n_chan=C, d_model=16, n_heads=2, n_layers=2,
                     d_z=DZ, d_dec=16, patch=1)
    with torch.no_grad():
        codec.to_z.weight.mul_(Z_GAIN)
        codec.to_z.bias.mul_(Z_GAIN)
    return npz, {"model": codec.state_dict(),
                 "chan": [f"c{i}" for i in range(C)],
                 "d_z": DZ, "norm": None, "step": 0,
                 "args": {"patch": 1, "d_model": 16, "n_layers": 2,
                          "n_heads": 2, "d_dec": 16, "anomaly": True,
                          "holdout_years": "1992",
                          "holdout_lon": "-45,-44"}}


def train(script, npz, run, tmp, extra, tag):
    """One end-to-end temporal.py run on the toy. Returns (metrics, temporal.json)."""
    run_dir = os.path.join(ML, "runs", run)
    for f in ("metrics.jsonl", "temporal.json", "temporal.pt"):
        p = os.path.join(run_dir, f)
        if os.path.exists(p):
            os.remove(p)
    env = dict(os.environ, CKPT_DIR_OVERRIDE=os.path.join(tmp, "ckpt", tag))
    r = subprocess.run(
        [sys.executable, "-u", script, "--run", run, "--data", npz,
         "--K", str(K), "--steps", str(STEPS), "--batch", "16",
         "--d-model", "16", "--layers", "2", "--lr", str(LR),
         "--lr-warmup", "10", "--max-pixels", "30", "--seed", "0", *extra],
        capture_output=True, text=True, timeout=1800, env=env)
    if r.returncode != 0:
        print(r.stdout[-3000:]); print(r.stderr[-3000:])
        raise SystemExit(f"temporal.py failed [{tag}] {' '.join(extra)}")
    recs = [json.loads(l) for l in
            open(os.path.join(run_dir, "metrics.jsonl")) if l.strip()]
    tj = json.load(open(os.path.join(run_dir, "temporal.json")))
    ck = torch.load(os.path.join(run_dir, "temporal.pt"),
                    map_location="cpu", weights_only=False)
    return recs, tj, ck, r.stdout


def curve(recs, key):
    return [(r["stage2_step"], r[key]) for r in recs if key in r]


def gnorm(ps):
    return float(torch.sqrt(sum(p.grad.detach().double().pow(2).sum()
                                for p in ps)))


def with_norm(target, n=4096, seed=0):
    """A parameter with a gradient of EXACTLY `target` 2-norm."""
    g = torch.Generator().manual_seed(seed)
    p = torch.zeros(n, requires_grad=True)
    v = torch.randn(n, generator=g).double()
    p.grad = (v / v.norm() * target).float()
    return p


def main():
    tmp = tempfile.mkdtemp()
    run = "toyclip"
    run_dir = os.path.join(ML, "runs", run)
    os.makedirs(run_dir, exist_ok=True)
    ok = True
    try:
        # ====== 1 · THE MECHANISM, AGAINST THE ARCHIVE'S OWN NUMBERS ======
        print(f"\n1 · torch.nn.utils.clip_grad_norm_ at max_norm {CLIP} and at "
              f"the pentad threshold 128.0 — the exact call temporal.py makes")
        # (a) a MONTHLY-scale gradient is untouched, bit for bit
        p = with_norm(MONTHLY_NORM)
        before = p.grad.clone()
        ret = float(torch.nn.utils.clip_grad_norm_([p], 128.0))
        assert torch.equal(p.grad, before), \
            "a monthly-scale gradient was MODIFIED by a clip at 128.0"
        assert abs(ret - MONTHLY_NORM) < 1e-4, \
            f"the returned pre-clip norm is wrong: {ret} vs {MONTHLY_NORM}"
        print(f"    #426's own norm {MONTHLY_NORM} through a clip at 128.0: "
              f"returned {ret:.4f}, gradient torch.equal to its input — the "
              f"clip is a NO-OP at monthly scale even when it is ON.")
        # ...and so is EVERY norm the monthly archive ever logged, up to its max
        p = with_norm(39.6165, seed=3)                    # #308, the archive max
        before = p.grad.clone()
        torch.nn.utils.clip_grad_norm_([p], 128.0)
        assert torch.equal(p.grad, before), \
            "the archive's LARGEST monthly norm was modified by a clip at 128.0"
        print(f"    the archive's largest monthly norm ever, 39.6165 (#308), "
              f"is also untouched: 128.0 is 3.231x it, and 0 of the 8,080 "
              f"logged norms would have been clipped.")
        # (b) an EXPLODING gradient comes out at exactly max_norm, same direction
        p = with_norm(PENTAD_BLOWUP, seed=1)
        before = p.grad.clone()
        ret = float(torch.nn.utils.clip_grad_norm_([p], 128.0))
        after = gnorm([p])
        cos = float(torch.nn.functional.cosine_similarity(
            p.grad.double(), before.double(), dim=0))
        assert abs(ret - PENTAD_BLOWUP) / PENTAD_BLOWUP < 1e-5, \
            f"the returned norm is not the PRE-clip norm: {ret}"
        assert abs(after - 128.0) < 1e-3, \
            f"the clipped gradient's norm is {after}, not the threshold 128.0"
        assert cos > 1 - 1e-9, f"the clip changed the direction (cos {cos})"
        print(f"    #423's own blow-up {PENTAD_BLOWUP} through a clip at "
              f"128.0: returned pre-clip {ret:.3f}, post-clip norm "
              f"{after:.6f} = the threshold exactly, direction preserved "
              f"(cos {cos:.12f}).")
        # (c) the arithmetic the SIZE of the threshold rests on
        b2 = 0.999
        vg_raw = 1 + (1 - b2) * (PENTAD_BLOWUP / HEALTHY_PENTAD) ** 2
        vg_clip = 1 + (1 - b2) * (128.0 / HEALTHY_PENTAD) ** 2
        assert vg_raw > 2000 and vg_clip < 1.3, (vg_raw, vg_clip)
        print(f"    one such step in AdamW: v grows {vg_raw:,.0f}x unclipped "
              f"(sqrt(v) {math.sqrt(vg_raw):.0f}x, so ~1/(1-beta2) = 1,000 "
              f"steps of honest gradients divided by {math.sqrt(vg_raw):.0f}) "
              f"against {vg_clip:.2f}x clipped (sqrt(v) "
              f"{math.sqrt(vg_clip):.2f}x). THAT is why the fix is a clip.")
        print("    PASS: no-op at monthly scale, exact clip to the threshold "
              "at pentad scale, direction preserved.")

        npz, ckpt = toy(tmp)
        torch.save(ckpt, os.path.join(run_dir, "pixelmae.pt"))
        cur = os.path.join(ML, "temporal.py")

        # ============ 2 · THE EFFECT, THROUGH THE REAL TRAINER ============
        off, tj_off, ck_off, out_off = train(cur, npz, run, tmp, [], "clipoff")
        nb, tj_nb, ck_nb, out_nb = train(
            cur, npz, run, tmp, ["--grad-clip", str(NONBIND)], "clipnb")
        on, tj_on, ck_on, out_on = train(
            cur, npz, run, tmp, ["--grad-clip", str(CLIP)], "clipon")
        pers = [r["stage2_monitor"]["val_persistence"] for r in off
                if "stage2_monitor" in r][0]
        g_off = curve(off, "stage2_grad_norm")
        g_on = curve(on, "stage2_grad_norm")
        gmax_on = curve(on, "stage2_grad_norm_max")
        frac = curve(on, "stage2_grad_clip_frac")
        nonfin = curve(on, "stage2_grad_nonfinite")
        frac_nb = curve(nb, "stage2_grad_clip_frac")
        gmax_nb = curve(nb, "stage2_grad_norm_max")
        print(f"\n2 · the real ml/temporal.py on a heavy-tailed toy "
              f"(val_persistence {pers:.4f})")
        print(f"    {'step':>5}  {'gnorm OFF':>11}  {'gnorm ON':>11}  "
              f"{'win-max ON':>11}  {'clipped':>8}   {'win-max @1e9':>12}"
              f"  {'clipped':>7}")
        for i in range(0, len(g_on), max(1, len(g_on) // 12)):
            print(f"    {g_on[i][0]:>5}  {g_off[i][1]:>11.3f}  "
                  f"{g_on[i][1]:>11.3f}  {gmax_on[i][1]:>11.3f}  "
                  f"{frac[i][1] * 100:>7.1f}%   {gmax_nb[i][1]:>12.3f}"
                  f"  {frac_nb[i][1] * 100:>6.1f}%")
        # (a) it is REPORTED, and the report is a WINDOW, not a point
        assert not any("stage2_grad_clip" in r for r in off), \
            "clip fields appear in the records with clipping off"
        assert len(gmax_on) == len(g_on) and len(frac) == len(g_on), \
            "the window statistics are not logged at every logged step"
        assert all(gm >= g for (_, gm), (_, g) in zip(gmax_on, g_on)), \
            "a window max came out below the norm sampled inside that window"
        assert max(gm for _, gm in gmax_on) > max(g for _, g in g_on), \
            ("the window max never exceeded the sampled norm, so it is adding "
             "nothing over the point statistic it exists to replace")
        assert all(n == 0 for _, n in nonfin), \
            f"a non-finite gradient norm was counted: {nonfin}"
        print(f"    PASS: the clip fields appear ONLY with the clip on; the "
              f"window max peaks at {max(gm for _, gm in gmax_on):,.1f} "
              f"against a sampled-norm peak of {max(g for _, g in g_on):,.1f} "
              f"— it sees a step the 1-in-{STEPS // len(g_on)} sampling misses; "
              f"0 non-finite steps.")
        # (b) THE INSTRUMENT DISTINGUISHES THE TWO STORIES (§4.10). "healthy,
        #     never binds" and "being clipped" must not read the same, and the
        #     rate must be able to take a value that is neither 0 nor 1 — a
        #     saturated indicator would say nothing about the regime.
        assert all(f == 0.0 for _, f in frac_nb), \
            f"clip_frac is non-zero at a threshold nothing reaches: {frac_nb}"
        assert 0.0 < max(f for _, f in frac) , "clip_frac never rose above zero"
        assert any(0.0 < f < 1.0 for _, f in frac), \
            ("clip_frac is only ever 0 or 1 — as an indicator of the regime it "
             "is saturated and cannot show a run LEAVING the healthy one")
        assert max(gm for _, gm in gmax_nb) > CLIP, \
            "the non-binding run never even reached the binding threshold"
        print(f"    PASS: at max_norm {NONBIND:g} the rate reads 0.0 at every "
              f"logged step while the window max still reports the true peak "
              f"{max(gm for _, gm in gmax_nb):,.1f}; at max_norm {CLIP} it "
              f"takes intermediate values (min {min(f for _, f in frac):.3f}, "
              f"max {max(f for _, f in frac):.3f}). The pair separates "
              f"'healthy' from 'the clip is now setting the learning rate'.")
        # (c) A CLIP THAT DOES NOT BIND CHANGES NOTHING — BIT FOR BIT. This is
        #     the archive-comparability guarantee in its strongest form.
        assert set(ck_off["model"]) == set(ck_nb["model"])
        d_nb = [k for k in ck_off["model"]
                if not torch.equal(ck_off["model"][k], ck_nb["model"][k])]
        assert not d_nb, f"a NON-BINDING clip moved the weights: {d_nb[:6]}"
        assert tj_nb["z_t+1"]["mse_model"] == tj_off["z_t+1"]["mse_model"], \
            "a non-binding clip moved the headline read-out"
        assert curve(nb, "stage2_val_zmse") == curve(off, "stage2_val_zmse"), \
            "a non-binding clip moved the val curve"
        print(f"    PASS: --grad-clip {NONBIND:g} against --grad-clip 0 — all "
              f"{len(ck_off['model'])} parameter tensors torch.equal, the val "
              f"curve identical, z_t+1 mse_model identical. Turning clipping "
              f"ON where it does not bind is EXACTLY a no-op.")
        # (d) ...and where it DOES bind it is not a no-op. Assert the effect,
        #     never the invocation (§0.2).
        d_on = [k for k in ck_off["model"]
                if not torch.equal(ck_off["model"][k], ck_on["model"][k])]
        assert d_on, \
            ("--grad-clip 50 produced identical weights to no clipping at all, "
             "so the clip did nothing despite reporting that it bound")
        print(f"    PASS: --grad-clip {CLIP} moved {len(d_on)}/"
              f"{len(ck_off['model'])} parameter tensors — where it binds it "
              f"changes the trajectory, which is the artefact, not the log.")
        # (e) the pre-clip norm is still the SAME quantity the old path logged
        assert max(g for _, g in g_on) > CLIP, \
            ("stage2_grad_norm is reporting the POST-clip norm — it never "
             "exceeds max_norm, and the pre-clip norm is the diagnostic the "
             "83-run archive is made of")
        print(f"    PASS: stage2_grad_norm still reports the PRE-clip norm "
              f"(it reaches {max(g for _, g in g_on):,.1f} > {CLIP}), so the "
              f"8,080-norm monthly archive stays comparable to it.")

        # ================= 3 · THE MONTHLY NO-OP ========================
        # BIT-FOR-BIT against the parent revision of ml/temporal.py. Not
        # "close", not "statistically the same" — every tensor equal.
        prev = subprocess.run(["git", "-C", os.path.join(HERE, ".."),
                               "show", f"{PREFIX_SHA}:ml/temporal.py"],
                              capture_output=True, text=True)
        assert prev.returncode == 0, (
            f"cannot read {PREFIX_SHA}:ml/temporal.py — this test compares "
            f"against a PINNED revision, not against HEAD, so that it keeps "
            f"proving the same thing after the fix is committed. Fetch the "
            f"full history (a shallow clone will not have it).")
        assert "clip_grad_norm_" not in prev.stdout, \
            f"{PREFIX_SHA} already carries clipping — not the pre-fix parent"
        old = os.path.join(ML, "_temporal_prefix_test.py")
        open(old, "w").write(prev.stdout)
        try:
            a_recs, a_tj, a_ck, _ = train(old, npz, run, tmp, [], "prefix")
            b_recs, b_tj, b_ck, _ = train(cur, npz, run, tmp, [], "postfix")
        finally:
            os.remove(old)
        print(f"\n3 · NO-OP PROOF — {PREFIX_SHA}:ml/temporal.py vs the working "
              f"tree, same toy, same seed, --grad-clip at its default 0")
        diffs = []
        assert set(a_ck["model"]) == set(b_ck["model"]), "state dict keys moved"
        for k in a_ck["model"]:
            if not torch.equal(a_ck["model"][k], b_ck["model"][k]):
                diffs.append(k)
        print(f"    {len(a_ck['model'])} parameter tensors compared with "
              f"torch.equal; {len(diffs)} differ")
        assert not diffs, f"weights moved: {diffs[:6]}"
        av = curve(a_recs, "stage2_val_zmse")
        bv = curve(b_recs, "stage2_val_zmse")
        assert av == bv and len(av) > 10, "the val curve moved"
        ag = curve(a_recs, "stage2_grad_norm")
        bg = curve(b_recs, "stage2_grad_norm")
        assert ag == bg and len(ag) > 10, "the grad-norm curve moved"
        for key in ("mse_model", "mse_persistence"):
            assert a_tj["z_t+1"][key] == b_tj["z_t+1"][key], \
                f"z_t+1 {key} moved: {a_tj['z_t+1'][key]} -> {b_tj['z_t+1'][key]}"
        assert a_tj["z_t+1"]["mse_model"] == tj_off["z_t+1"]["mse_model"], \
            "the check-2 clip-off run does not agree with the check-3 one"
        print(f"    val curve, grad-norm curve and z_t+1 identical over "
              f"{len(av)} logged points; z_t+1 mse_model "
              f"{b_tj['z_t+1']['mse_model']!r}, mse_persistence "
              f"{b_tj['z_t+1']['mse_persistence']!r}")
        print("    PASS: with --grad-clip at its default the trainer is the "
              "pre-fix trainer, bit for bit. No archived monthly number moves.")

        # ================= 4 · THE ARGUMENT-TIME REFUSAL ================
        # §0.3: a precondition that depends only on the inputs is checked
        # while the inputs are all it has cost. A NEGATIVE max_norm is not
        # "off" — clip_grad_norm_ would scale every gradient by a negative
        # coefficient and walk UPHILL, and nothing downstream would say so.
        r = subprocess.run(
            [sys.executable, cur, "--run", run, "--data", npz,
             "--grad-clip", "-1.0"],
            capture_output=True, text=True, timeout=300)
        print("\n4 · a negative --grad-clip")
        assert r.returncode != 0, "a negative --grad-clip was accepted"
        out = r.stdout + r.stderr
        assert "must be >= 0" in out, f"refused for the wrong reason: {out[-300:]}"
        assert "Traceback" not in r.stderr, \
            "refused with a traceback rather than a message"
        assert "embedding" not in out and "cached" not in out, \
            "the refusal came AFTER the tensor was read"
        print(f"    PASS: refused in {len(out)} chars, before any tensor is "
              f"read: {out.strip()[:140]}")

        # ================= 5 · THE Z-SPACE SCALE IS RECORDED =============
        # The defect #423 could not be diagnosed from its own artefacts:
        # `--input-znoise` is an ABSOLUTE sigma, the monthly 0.7 was carried
        # to a codec whose z-space is on a different scale, and NO record —
        # not the config, not the monitor, not temporal.json — carried either
        # the sigma or the scale it should be judged against. The fix is
        # reporting, not behaviour: the numbers below must come out of the
        # DATA (§4.2, normalise by properties of the data), and the toy's
        # z-space is Z_GAIN times an ordinary one precisely so that a
        # hard-coded monthly constant would fail this check.
        ZN = 0.7
        e_recs, _, e_ck, e_out = train(
            cur, npz, run, tmp, ["--input-znoise", str(ZN),
                                 "--grad-clip", str(CLIP)], "znrep")
        mon = [r["stage2_monitor"] for r in e_recs if "stage2_monitor" in r][0]
        cfg = [r["stage2_config"] for r in e_recs if "stage2_config" in r][0]
        print(f"\n5 · --input-znoise {ZN} on a z-space this run measures itself")
        print(f"    stage2_monitor: {json.dumps(mon)}")
        print(f"    stage2_config : input_znoise {cfg['input_znoise']} · "
              f"grad_clip {cfg['grad_clip']}")
        want_pers = ZN / math.sqrt(mon["val_persistence"])
        want_zrms = ZN / mon["z_rms"]
        assert abs(mon["input_znoise_rel_pers"] - want_pers) < 1e-4, \
            (f"rel_pers is not sigma/sqrt(val_persistence): "
             f"{mon['input_znoise_rel_pers']} vs {want_pers}")
        assert abs(mon["input_znoise_rel_zrms"] - want_zrms) < 1e-4, \
            (f"rel_zrms is not sigma/z_rms: "
             f"{mon['input_znoise_rel_zrms']} vs {want_zrms}")
        assert mon["input_znoise_sigma"] == ZN
        # the two scales are DIFFERENT quantities and must not be conflated:
        # z_rms is the size of the z-space, val_persistence its one-step change
        assert mon["z_rms"] > 0 and mon["val_persistence"] > 0
        assert abs(mon["z_rms"] - math.sqrt(mon["val_persistence"])) > 1e-6, \
            "z_rms and sqrt(val_persistence) came out identical — one of them "\
            "is not measuring what it claims"
        # and the scale must track the DATA. Two clauses, and neither is a
        # magic constant: this fixture IS a materially different z-space from
        # the monthly anchor (its val_persistence is an order of magnitude off
        # 3.09512, because the codec's z projection was scaled by Z_GAIN), and
        # the reported ratio moved WITH it rather than staying at the monthly
        # 0.39788 that a hard-coded constant would print.
        assert mon["val_persistence"] > 10 * 3.09512, \
            (f"the fixture's z-space is not materially different from the "
             f"monthly anchor's (val_persistence {mon['val_persistence']} vs "
             f"3.09512), so this check cannot detect a hard-coded constant")
        assert abs(mon["input_znoise_rel_pers"] - 0.39788) > 0.1, \
            (f"the reported relative sigma {mon['input_znoise_rel_pers']} sits "
             f"on the monthly anchor's 0.39788 on a z-space "
             f"{mon['val_persistence'] / 3.09512:.0f}x its size — it is not "
             f"being derived from this z-space's own scale")
        assert cfg["input_znoise"] == ZN and cfg["grad_clip"] == CLIP, \
            f"stage2_config does not carry the two knobs: {cfg}"
        assert e_ck["args"]["input_znoise"] == ZN and \
            e_ck["args"]["grad_clip"] == CLIP, \
            "the checkpoint's args do not carry the two knobs"
        assert "is an ABSOLUTE sigma" in e_out and "RMS |z|" in e_out, \
            "the run does not SAY the sigma is absolute and against what"
        print(f"    PASS: sigma {ZN} is recorded as "
              f"{mon['input_znoise_rel_pers']:.5f} x sqrt(val_persistence) and "
              f"{mon['input_znoise_rel_zrms']:.5f} x RMS|z|, both derived from "
              f"this run's own z, and both reach stage2_config and the "
              f"checkpoint args.")

        print("\nALL CHECKS PASSED")
    except AssertionError as e:
        ok = False
        print(f"\nFAILED: {e}")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
        shutil.rmtree(run_dir, ignore_errors=True)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()

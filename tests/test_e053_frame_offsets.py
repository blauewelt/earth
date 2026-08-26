#!/usr/bin/env python3
"""E-053.1 · `--frame-offsets`: the sunflower taken into the time dimension.

WHAT THE INSTRUMENT IS. The stage-2 head's context has always been a DENSE
SLAB: K frames at every one of the K bins before the anchor. E-045 measured
that what buys skill is the context's SPAN (pentad span 120 d -> ratio 0.5056,
span 720 d -> 0.0820) while step size at fixed span moves it ~4%, and under a
dense slab span can only be bought by raising K — 2 years at pentad is K=144,
20,880 samples per window and 36x the attention of K=24. `--frame-offsets`
names the frame TIMES instead of counting them, so a 2-year span costs 24
frames rather than 144 (ml/plans/E053_spacetime_stencil.md §4).

WHAT IS UNDER TEST, and in what order.

  1. **THE NO-OP, AGAINST THE PARENT REVISION.** `git show HEAD:ml/temporal.py`
     and the working tree, same toy, same seed, no flag: every parameter tensor
     `torch.equal`, identical loss curve, identical `z_t+1`. Every archived
     stage-2 number must stay bit-reproducible — this is check 1 of
     tests/test_e044c_knobs.py pointed at a different change.
  2. **THE DEFAULTS ARE THE DEFAULTS.** `--frame-offsets` set EXPLICITLY to the
     contiguous list `-(K-1)..0` reproduces the unflagged run tensor for
     tensor, so 'off' is one code path and not two that agree today.
  3. **THE GATHER, AGAINST AN INDEPENDENT EXPECTATION.** A Z whose every value
     names its own coordinates (`Z[t, p] = 1000t + p`), so a frame gathered
     from the wrong bin cannot look plausible. Frames, per-frame season rows
     and per-frame targets are each checked against `t + offsets[j]` written
     out by hand, at stencil 1 AND through the neighbour path.
  4. **THE POOL BOUND.** `t >= K-1` generalises to `t + offsets[0] >= 0`. The
     printed train-window count is checked against a count computed here from
     the toy's own axis — and against what the LEGACY bound would have given,
     so "the guard is present" and "the guard binds" are separable.
  5. **EVERY REFUSAL FIRES**, at argument time, before the tensor is opened,
     naming `--frame-offsets` and the reason: five malformed lists and the four
     forbidden flag combinations.
  6. **THE ARM IS RECORDED AND THE ROLLER REFUSES IT.** The offsets, the
     derived K and the span reach `stage2_config` and the checkpoint args, the
     position table is sized from the derived K, and `ml/rollout_spatial.py`
     stops on such a head — it assembles a CONTIGUOUS window and would
     otherwise score a head on inputs it never saw.
  7. **END TO END ON CPU**: finite loss, a monitor record, no NaN anywhere.

  Plus the parameter counts at k_max 16 / 24 / 32 at the production geometry,
  PRINTED rather than asserted. `pos` is `nn.Embedding(k_max, d_model)`, so the
  count SHIFTS with the derived K — 206,535,712 is the k_max 24 figure and is
  not a constant of the architecture. There is no NEW parameter: at a fixed
  offset pattern the map position <-> offset is a bijection, so the existing
  learned position embedding IS the delta-t encoding.

No pytest. Numbered checks, printed.

    python3 tests/test_e053_frame_offsets.py
"""
import contextlib
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
ROOT = os.path.abspath(os.path.join(HERE, ".."))
ML = os.path.join(ROOT, "ml")
sys.path.insert(0, HERE)
sys.path.insert(0, ML)
from test_e044_grad_clip import toy, train, K, T_M, C, DZ      # noqa: E402
from temporal import (TemporalTransformer, frame_ref,          # noqa: E402
                      frame_steps, gather_stencil)

# The toy's calendar, restated here rather than imported, so the pool
# arithmetic in check 4 is computed from the fixture's DEFINITION and not from
# anything ml/temporal.py also derives.
MONTHS = [f"{1990 + i // 12}-{i % 12 + 1:02d}" for i in range(T_M)]
HOLD = [m[:4] == "1992" for m in MONTHS]        # the codec's holdout_years
CONTIG = ",".join(str(o) for o in range(-(K - 1), 1))   # "-5,-4,-3,-2,-1,0"
SPARSE = "-8,-4,0"                              # span 8 > K-1 = 5, K_eff 3


def params_equal(a, b):
    ka, kb = set(a["model"]), set(b["model"])
    assert ka == kb, (ka ^ kb)
    return [k for k in sorted(ka)
            if not torch.equal(a["model"][k], b["model"][k])]


def pool_of(stdout):
    for ln in stdout.splitlines():
        if ln.startswith("train windows:"):
            return int(ln.split(":")[1].strip().replace(",", ""))
    raise AssertionError("no 'train windows:' line")


def n_anchors(back):
    """Eligible ANCHORS on the toy axis for a context reaching `back` bins:
    t+1 must exist and be a train month, and t-back must exist. Written out
    from MONTHS/HOLD above — the independent expectation check 4 needs."""
    return sum(1 for t in range(T_M)
               if t + 1 < T_M and t >= back and not HOLD[t + 1])


def curve(recs, key):
    return [(r["stage2_step"], r[key]) for r in recs if key in r]


def refuse(cur, npz, args, want, tag):
    """One argument-time refusal. Returns the message, and insists it named
    the flag, arrived without a traceback, and came BEFORE the tensor was
    read (§0.3: check a precondition where the inputs are all it has cost)."""
    r = subprocess.run(
        [sys.executable, cur, "--run", "e053", "--data", npz,
         "--steps", "1", "--max-pixels", "30", *args],
        capture_output=True, text=True, timeout=600)
    out = r.stdout + r.stderr
    assert r.returncode != 0, f"[{tag}] was ACCEPTED: {args}"
    assert "--frame-offsets" in out, f"[{tag}] refusal does not name the flag: {out[-400:]}"
    assert want in out, f"[{tag}] refused for the wrong reason: {out[-400:]}"
    assert "Traceback" not in r.stderr, f"[{tag}] refused with a traceback"
    # "lon holdout" is the first line printed after the tensor is opened;
    # "train windows" is printed after it is embedded. Neither may appear.
    assert "lon holdout" not in out and "train windows" not in out, \
        f"[{tag}] the refusal came AFTER the tensor was read: {out[-300:]}"
    return out.strip().splitlines()[-1]


def main():
    tmp = tempfile.mkdtemp()
    run = "e053_offsets"
    run_dir = os.path.join(ML, "runs", run)
    ok = True
    try:
        os.makedirs(run_dir, exist_ok=True)
        npz, ckd = toy(tmp)
        torch.save(ckd, os.path.join(run_dir, "pixelmae.pt"))
        cur = os.path.join(ML, "temporal.py")

        # ============ 1 · THE NO-OP, AGAINST THE PINNED PRE-E-053 REVISION =
        # bc159fc is the last main commit whose ml/temporal.py predates
        # --frame-offsets (the E-053 scoping commit itself; verified identical
        # to origin/main's temporal.py at pin time). Pinned the way
        # tests/test_e044_grad_clip.py pins 877ae5b: comparing the tree
        # against a HEAD that already carries the flag would prove nothing.
        PRE_E053_SHA = "bc159fc"
        prev = subprocess.run(["git", "-C", ROOT, "show",
                               f"{PRE_E053_SHA}:ml/temporal.py"],
                              capture_output=True, text=True)
        assert prev.returncode == 0, prev.stderr[-300:]
        assert "frame_offsets" not in prev.stdout, \
            (f"{PRE_E053_SHA}:ml/temporal.py already carries --frame-offsets "
             f"— the pin must name a pre-E-053 revision.")
        # INSIDE ml/, like the other two knob tests: temporal.py imports its
        # siblings by module name, so a copy in /tmp cannot run.
        base = os.path.join(ML, "_temporal_e053_base.py")
        open(base, "w").write(prev.stdout)
        try:
            r_new, tj_new, ck_new, out_new = train(cur, npz, run, tmp, [], "new")
            r_old, tj_old, ck_old, out_old = train(base, npz, run, tmp, [], "base")
        finally:
            os.remove(base)
        bad = params_equal(ck_new, ck_old)
        assert not bad, f"{len(bad)} tensors differ from the parent: {bad[:4]}"
        cn, co = curve(r_new, "stage2_zmse"), curve(r_old, "stage2_zmse")
        assert cn == co and len(cn) > 10, (cn[:3], co[:3])
        assert curve(r_new, "stage2_val_zmse") == curve(r_old, "stage2_val_zmse")
        assert tj_new["z_t+1"] == tj_old["z_t+1"], (tj_new["z_t+1"],
                                                    tj_old["z_t+1"])
        assert pool_of(out_new) == pool_of(out_old)
        assert "frame offsets:" not in out_new, \
            "an unflagged run printed the E-053.1 line"
        print("1. unflagged working tree == HEAD:ml/temporal.py — all %d "
              "parameter tensors torch.equal, identical loss and val curves "
              "(%d points), identical z_t+1, identical pool (%d windows). "
              "--frame-offsets adds no code path to a run that does not name "
              "it." % (len(ck_new["model"]), len(cn), pool_of(out_new)))

        # ============ 2 · THE DEFAULTS ARE THE DEFAULTS ===================
        # The contiguous list IS the default stencil written out. If the two
        # ever disagree, 'off' has become a second implementation.
        r_c, tj_c, ck_c, out_c = train(cur, npz, run, tmp,
                                       [f"--frame-offsets={CONTIG}"], "contig")
        bad = params_equal(ck_new, ck_c)
        assert not bad, \
            (f"--frame-offsets={CONTIG} moved {len(bad)} tensors against the "
             f"unflagged run: {bad[:4]} — the explicit contiguous list is not "
             f"the implicit one, and the RNG stream or the pool has drifted")
        assert curve(r_c, "stage2_zmse") == cn, "the loss curve moved"
        assert curve(r_c, "stage2_val_zmse") == curve(r_new, "stage2_val_zmse"), \
            "the val curve moved"
        assert tj_c["z_t+1"] == tj_new["z_t+1"], "z_t+1 moved"
        assert pool_of(out_c) == pool_of(out_new), "the window pool moved"
        assert f"frame offsets: K={K}" in out_c and "span=5 bins" in out_c, \
            f"the derived-K line is missing or wrong: {out_c[:600]}"
        assert ck_c["args"]["frame_offsets"] == CONTIG
        print("2. --frame-offsets=%s (the contiguous stencil, written out) "
              "reproduces the unflagged run tensor for tensor: all %d "
              "parameters torch.equal, loss and val curves identical over %d "
              "points, same pool (%d), same z_t+1 — so the RNG stream is "
              "untouched. It still PRINTS 'frame offsets: K=%d ... span=5 "
              "bins' and records the list in the checkpoint."
              % (CONTIG, len(ck_c["model"]), len(cn), pool_of(out_c), K))

        # ====== 3 · THE GATHER, AGAINST AN INDEPENDENT EXPECTATION ========
        # Z[t, p, :] = 1000t + p. Every value names the (bin, pixel) it came
        # from, so a frame taken from the wrong bin cannot pass by looking
        # plausible — which is exactly how the drivers-grid row flip survived
        # inspection (root CLAUDE.md, "a picture of a model").
        Tz, Pz, dz = 24, 5, 3
        Zt = torch.zeros(Tz, Pz, dz)
        for t_ in range(Tz):
            for p_ in range(Pz):
                Zt[t_, p_] = 1000 * t_ + p_
        Mt = torch.stack([torch.arange(Tz).float(),
                          -torch.arange(Tz).float()], 1)     # [T, 2], = the bin
        offs = (-5, -2, 0)
        anchors = torch.tensor([7, 11, 23 - 1])              # t+1 must exist
        pix = torch.tensor([0, 3, 4])
        ref = frame_ref(anchors, len(offs), offs)
        assert torch.equal(ref, anchors), \
            "under --frame-offsets the gather reference must be the ANCHOR t"
        assert torch.equal(frame_ref(anchors, K, None), anchors - K + 1), \
            "without offsets the reference must stay the window START t-K+1"
        assert list(frame_steps(len(offs), offs)) == list(offs)
        assert list(frame_steps(K, None)) == list(range(K))

        zs = gather_stencil(Zt, ref, pix, None, len(offs), offs)
        assert tuple(zs.shape) == (3, 3, dz), zs.shape
        ms = torch.stack([Mt[ref + j] for j in frame_steps(len(offs), offs)], 1)
        zg = torch.stack([Zt[ref + j + 1, pix]
                          for j in frame_steps(len(offs), offs)], 1)
        for i in range(len(anchors)):
            t_i, p_i = int(anchors[i]), int(pix[i])
            for j, o in enumerate(offs):                 # the hand-written truth
                want_t = t_i + o
                assert torch.equal(zs[i, j],
                                   torch.full((dz,), 1000.0 * want_t + p_i)), \
                    (f"frame {j} of anchor {t_i} came from bin "
                     f"{float(zs[i, j, 0]) // 1000:.0f}, wanted {want_t}")
                assert float(ms[i, j, 0]) == want_t, \
                    (f"season row {j} of anchor {t_i} is bin {ms[i, j, 0]}, "
                     f"wanted the FRAME's own time {want_t}")
                assert torch.equal(zg[i, j],
                                   torch.full((dz,),
                                              1000.0 * (want_t + 1) + p_i)), \
                    (f"target {j} of anchor {t_i} is not the bin AFTER that "
                     f"frame's own time ({want_t + 1})")
            # ...and the last frame's target is still exactly the headline t+1
            assert torch.equal(zg[i, -1], Zt[t_i + 1, p_i]), \
                "the last frame's target is no longer Z[t+1]"
        # the contiguous control, same fixture
        zc = gather_stencil(Zt, frame_ref(anchors, len(offs), None),
                            pix, None, len(offs), None)
        for i in range(len(anchors)):
            t_i, p_i = int(anchors[i]), int(pix[i])
            for j, o in enumerate((-2, -1, 0)):
                assert torch.equal(zc[i, j],
                                   torch.full((dz,), 1000.0 * (t_i + o) + p_i))
        # and the NEIGHBOUR path: same times, missing slots zero-filled
        NBR = torch.tensor([[p_, (p_ + 1) % Pz, -1] for p_ in range(Pz)])
        zn = gather_stencil(Zt, ref, pix, NBR, len(offs), offs)
        assert tuple(zn.shape) == (3, 3, 3 * dz), zn.shape
        for i in range(len(anchors)):
            t_i, p_i = int(anchors[i]), int(pix[i])
            for j, o in enumerate(offs):
                row = zn[i, j].view(3, dz)
                assert float(row[0, 0]) == 1000 * (t_i + o) + p_i
                assert float(row[1, 0]) == 1000 * (t_i + o) + (p_i + 1) % Pz
                assert float(row[2].abs().sum()) == 0.0, \
                    "a missing neighbour slot is not zero-filled"
        print("3. Z[t,p] = 1000t+p, offsets %s: every frame, every season row "
              "and every per-frame target lands on the bin written out by "
              "hand (t+offset[j], and t+offset[j]+1 for the target); the last "
              "frame's target is Z[t+1] exactly; the contiguous control still "
              "reads t-2,t-1,t; and the stencil>1 path carries the same times "
              "with missing slots at exact zero." % (offs,))

        # ==================== 4 · THE POOL BOUND =========================
        r_s, tj_s, ck_s, out_s = train(cur, npz, run, tmp,
                                       [f"--frame-offsets={SPARSE}"], "sparse")
        n_px = pool_of(out_new) // n_anchors(K - 1)
        assert pool_of(out_new) == n_anchors(K - 1) * n_px, \
            ("the toy's default pool is not anchors x pixels — the "
             "independent expectation below cannot be trusted")
        want = n_anchors(8) * n_px          # -offsets[0] = 8
        legacy = n_anchors(len(SPARSE.split(",")) - 1) * n_px    # if K-1 were used
        assert pool_of(out_s) == want, \
            (f"pool is {pool_of(out_s):,}, expected {want:,} "
             f"({n_anchors(8)} anchors x {n_px} pixels)")
        assert want != legacy, \
            ("the fixture cannot distinguish the new bound from the old one: "
             "choose offsets whose span exceeds K_eff-1")
        assert pool_of(out_s) < pool_of(out_new), \
            "a longer span did not shrink the pool"
        assert f"span=8 bins" in out_s and "frame offsets: K=3" in out_s
        print("4. --frame-offsets=%s (K_eff 3, span 8): the anchor bound is "
              "t >= 8, not t >= K_eff-1 = 2. Pool %d = %d anchors x %d "
              "pixels, computed here from the toy's own calendar; the legacy "
              "bound would have given %d, and the unflagged K=%d run gives "
              "%d. A long span buys fewer windows, which is correct."
              % (SPARSE, pool_of(out_s), n_anchors(8), n_px, legacy, K,
                 pool_of(out_new)))

        # ==================== 5 · EVERY REFUSAL FIRES ====================
        cases = [
            ([f"--frame-offsets=-5,-2,1"], "must be <= 0", "positive offset"),
            ([f"--frame-offsets=-5,-2,-1"], "must be 0", "last != 0"),
            ([f"--frame-offsets=-5,-2,-2,0"], "STRICTLY", "duplicate"),
            ([f"--frame-offsets=-2,-5,0"], "STRICTLY", "unsorted"),
            ([f"--frame-offsets=0"], "at least 2", "len < 2"),
            ([f"--frame-offsets=a,b,0"], "must be an integer", "not integers"),
            ([f"--frame-offsets={SPARSE}", "--unroll", "2"],
             "--unroll 2", "with --unroll"),
            ([f"--frame-offsets={SPARSE}", "--unroll-wide", "2"],
             "--unroll-wide", "with --unroll-wide"),
            ([f"--frame-offsets={SPARSE}", "--time-stride", "3"],
             "--time-stride", "with --time-stride"),
            ([f"--frame-offsets={SPARSE}", "--direct", "3,6"],
             "--direct", "with --direct"),
        ]
        print("5. every refusal, at argument time, before the tensor is read")
        for args, want_txt, tag in cases:
            msg = refuse(cur, npz, args, want_txt, tag)
            print(f"    {tag:<20} {msg[:96]}")
        print("    PASS: %d refusals, each naming --frame-offsets and its "
              "reason, none reaching the tensor." % len(cases))

        # ====== 6 · THE ARM IS RECORDED, AND THE ROLLER REFUSES IT =======
        cfg = [x["stage2_config"] for x in r_s if "stage2_config" in x][0]
        assert cfg["frame_offsets"] == SPARSE and cfg["frame_span"] == 8, cfg
        assert cfg["K"] == 3, cfg
        assert ck_s["args"]["frame_offsets"] == SPARSE
        assert ck_s["args"]["K"] == 3, \
            "the checkpoint records --K rather than the DERIVED K"
        assert ck_s["model"]["pos.weight"].shape[0] == 3, \
            ("the position table was sized from --K, not from the derived K — "
             "a roller reads that shape as the head's K")
        assert tj_s["scale"]["frame_offsets"] == SPARSE
        print("6a. the offsets, the DERIVED K (3, overriding --K %d) and the "
              "span reach stage2_config, temporal.json's scale block and the "
              "checkpoint args; pos.weight is %d rows, sized from the derived "
              "K." % (K, ck_s["model"]["pos.weight"].shape[0]))

        # The roller: the real guard, on a stub head, through the real script.
        from test_rollout_spatial import build_fixture, DZ as RDZ
        rtmp = tempfile.mkdtemp()
        try:
            with open(os.devnull, "w") as _dn, \
                    contextlib.redirect_stdout(_dn):   # its embed progress
                f = build_fixture(rtmp)
            torch.manual_seed(11)
            hm = TemporalTransformer(d_z=RDZ, d_model=8, n_heads=4,
                                     n_layers=1, k_max=3, stencil=1)
            stub = os.path.join(rtmp, "toy_offsets_s0.pt")
            torch.save({"model": hm.state_dict(),
                        "args": {"K": 3, "d_model": 8, "layers": 1,
                                 "unroll": 1, "seed": 0, "stencil": 1,
                                 "frame_offsets": SPARSE}}, stub)
            rout = os.path.join(rtmp, "rollout_spatial.json")
            rr = subprocess.run(
                [sys.executable, "-u", os.path.join(ML, "rollout_spatial.py"),
                 "--x", f["x"], "--npz-small", f["npz"], "--z", f["z"],
                 "--ckpt", f["ckpt"], "--out", rout, "--horizon", "2",
                 "--long-start", "1991-12", "--long-months", "6",
                 "--future-months", "2", "--cache-dir", rtmp,
                 "--no-gate", "--heads", stub],
                capture_output=True, text=True, timeout=1800)
            rtxt = rr.stdout + rr.stderr
            assert rr.returncode != 0, \
                "rollout_spatial.py ROLLED a --frame-offsets head"
            assert "--frame-offsets" in rtxt and "Refusing" in rtxt, \
                f"refused for the wrong reason: {rtxt[-600:]}"
            assert "Traceback" not in rr.stderr, "refused with a traceback"
            line = [l for l in rtxt.splitlines() if "--frame-offsets" in l][0]
            print(f"6b. ml/rollout_spatial.py on a head carrying "
                  f"frame_offsets={SPARSE!r}: {line[:150]}")
        finally:
            shutil.rmtree(rtmp, ignore_errors=True)

        # ==================== 7 · END TO END ON CPU ======================
        mon = [x["stage2_monitor"] for x in r_s if "stage2_monitor" in x]
        assert mon, "no stage2_monitor record from the offsets run"
        assert math.isfinite(mon[0]["val_persistence"]) \
            and mon[0]["val_persistence"] > 0
        # THE MONITOR'S SEMANTICS DO NOT MOVE. `val_persistence` is |Z[t+1] -
        # Z[t]|^2 over windows whose t+1 is a holdout bin, and the offsets
        # change only how far BACK such a window reaches. On this toy the two
        # bounds select the identical monitor set by construction — the
        # holdout year starts at bin 24, well past both 5 and 8 — so the two
        # runs must report the same number to the last digit.
        mon0 = [x["stage2_monitor"] for x in r_new if "stage2_monitor" in x][0]
        assert mon[0]["val_persistence"] == mon0["val_persistence"], \
            (f"val_persistence moved with the context sampling: "
             f"{mon[0]['val_persistence']} vs {mon0['val_persistence']} — the "
             f"headline target or the persistence baseline has shifted, which "
             f"E-053 must never do")
        losses = [v for _, v in curve(r_s, "stage2_zmse")]
        vals = [v for _, v in curve(r_s, "stage2_val_zmse")]
        assert losses and all(math.isfinite(v) for v in losses), losses[:5]
        assert vals and all(math.isfinite(v) for v in vals), vals[:5]
        assert math.isfinite(tj_s["z_t+1"]["mse_model"]), tj_s["z_t+1"]
        assert math.isfinite(tj_s["z_t+1"]["mse_persistence"])
        assert tj_s["K"] == 3 and tj_s["scale"]["data_points"] == pool_of(out_s)
        print("7. end to end on CPU with %s: %d logged loss points, all "
              "finite (%.4f -> %.4f), monitor emitted (val_persistence "
              "%.5f), z_t+1 mse_model %.4f vs persistence %.4f, no NaN."
              % (SPARSE, len(losses), losses[0], losses[-1],
                 mon[0]["val_persistence"], tj_s["z_t+1"]["mse_model"],
                 tj_s["z_t+1"]["mse_persistence"]))

        # ===== the parameter counts, PRINTED — see the module docstring ===
        print("\nparameter counts at the production geometry "
              "(d_z 32 · d_model 1024 · 4 heads · 16 layers · stencil 145), "
              "by k_max:")
        for km in (16, 24, 32):
            m = TemporalTransformer(d_z=32, d_model=1024, n_heads=4,
                                    n_layers=16, k_max=km, stencil=145)
            n = sum(p_.numel() for p_ in m.parameters())
            print(f"    k_max {km:>2}: {n:,} parameters")
        print("    (pos is nn.Embedding(k_max, 1024) = 1,024 per row, so the "
              "count shifts by 8,192 per 8 rows. 206,535,712 is the k_max 24 "
              "figure. NO new parameter is added by --frame-offsets: at a "
              "fixed offset pattern position <-> offset is a bijection, so "
              "the learned position embedding IS the delta-t encoding.)")

        print("\nE-053.1 frame offsets: all 7 checks hold ✓")
    except AssertionError as e:
        ok = False
        print(f"\nFAILED: {e}")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
        shutil.rmtree(run_dir, ignore_errors=True)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()

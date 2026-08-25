#!/usr/bin/env python3
"""E-050 · `--fsq-warmstart`: the ONE hole in the resume guard, and its walls.

Approved by Chris 2026-08-25. `--resume` derives architecture from the
checkpoint's own args and REFUSES a dispatch that contradicts it — correct in
general (`fsq_levels` decides what a `z` IS, and a resume that contradicted it
would continue a quantized codec as a continuous one while every tensor in the
state_dict loaded cleanly), and it also blocks the one transition now worth
running:

  · E-046 measured a LATTICE z forecasting better than a continuous one
    (ratio 0.4394 vs 0.5056), and
  · every COLD-START codec-side FSQ run has collapsed or degenerated — e048a
    and #481/#482 to a constant encoder, run-455 to a one-bit sign code.

So: train continuous, then switch the lattice on from that checkpoint. It is
safe because the lattice and `--fsq-bound ln` (LayerNorm, no affine) are
PARAMETER-FREE — E-049 measured 37,956,471 parameters on both arms, identical
state_dicts — so a warm start loads every weight exactly and only the
bottleneck's function changes.

A hole in a guard is only as good as its walls, and this file is the walls.
Everything runs through `ml/train.py`'s own command line on a CPU toy, where a
dispatch would hit it, because the thing under test IS the dispatch path.

  1. THE PRE-EXISTING REFUSAL, NOW PINNED. A continuous checkpoint resumed
     with an explicit `--fsq-levels` and NO `--fsq-warmstart` refuses, naming
     both values. This is what the flag opens, so it must be shown shut.
  2. THE WARM START PROCEEDS, and it is a warm start in the only sense that
     matters: with `--steps` at the checkpoint's own step nothing trains, and
     EVERY weight tensor in the child is `torch.equal` to the parent's. The
     step is adopted, the saved args carry the lattice (so `codec_from_ckpt`
     rebuilds an FSQ codec), and `fsq_warmstart_from` records which continuous
     codec it grew out of and at which step.
  3. THE FIT SCHEDULE IS REBASED. `--fsq-auto-step` is ABSOLUTE on an ordinary
     run, and a warm start begins past every entry — with the comparison
     unchanged the lattice would NEVER be fitted, which is the collapse the
     flag exists to avoid, arriving through the feature meant to prevent it.
     The falsifier is asserted first (`fql.fit_steps` on the unrebased list
     returns steps the run has already passed), then the real loop is run and
     the fits are observed at S+1 and S+2.
  4. THE FOUR REFUSALS: no `--resume`; no explicit `--fsq-levels`; a
     checkpoint that ALREADY carries `fsq_levels` (FSQ->FSQ is a changed
     lattice, not a warm start — it would reinterpret every z the run has
     already written); and a `--resume` target that is not on this box (a warm
     start with nothing to start from is a COLD start, indistinguishable from
     a good one in every line of the log but this refusal).
  5. THE HOLE IS FOUR KEYS WIDE. `--fsq-warmstart` does not license a
     contradicting `--d-model`: the non-fsq architecture keys keep #395's
     refusal exactly.
  6. NO LOADER REFUSES THE NEW KEYS. `fsq_warmstart` lands in `vars(a)` on
     EVERY run from this commit on, and both `codec_from_ckpt` and the JAX
     `convert.py` refuse unknown `fsq_*` keys by design (E-049's clause, and
     its PRESENCE-keyed bargain). Both must read the warm-start keys as
     informational — on a warm-started checkpoint AND on an ordinary
     continuous one.
  7. A CONTRADICTORY FILE IS REFUSED, NOT AVERAGED. A checkpoint with no
     `fsq_levels` but a non-empty `fsq_ladder_fit` cannot be both; the
     never-re-fit adoption path would otherwise install a lattice fitted
     against a bottleneck that is not this one. §0.1: verify the artefact.

Run: python3 tests/test_fsq_warmstart.py   (~2 min on two cores, no GPU)
"""
import json
import os
import shutil
import subprocess
import sys
import tempfile

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
ML = os.path.join(ROOT, "ml")
sys.path.insert(0, HERE)
sys.path.insert(0, ML)

import fsq_ladder as fql                                        # noqa: E402
from model import codec_from_ckpt                               # noqa: E402
from test_e047_block_smoke import toy, C                        # noqa: E402

DZ = 6
LEVELS = "8,8,8,5,5,5"
PARENT_STEPS = 4              # the continuous checkpoint's own step


def run(cmd, tag, want_fail=False, timeout=1800):
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout,
                       cwd=ROOT)
    ok = (r.returncode != 0) if want_fail else (r.returncode == 0)
    if not ok:
        print(r.stdout[-4000:])
        print(r.stderr[-3000:])
        raise SystemExit(f"{tag}: unexpected rc {r.returncode}")
    return r.stdout + r.stderr


def load(path):
    return torch.load(path, map_location="cpu", weights_only=False)


def main():
    tmp = tempfile.mkdtemp()
    try:
        npz, _ = toy(tmp)

        def cmd(out, *extra):
            return [sys.executable, "-u", os.path.join(ML, "train.py"),
                    "--data", npz, "--out", out, "--batch", "16",
                    "--d-model", "16", "--n-layers", "2", "--n-heads", "2",
                    "--d-dec", "16", "--d-z", str(DZ), "--patch", "1",
                    "--anomaly", "--holdout-years", "1991",
                    "--holdout-lon=0,0", "--collapse-r", "0",
                    "--light-probe-every", "0", *extra]

        # ---- the CONTINUOUS parent, and an FSQ one for case 4c -----------
        cont_dir = os.path.join(tmp, "cont")
        run(cmd(cont_dir, "--steps", str(PARENT_STEPS)), "continuous parent")
        cont = os.path.join(cont_dir, "pixelmae.pt")
        pck = load(cont)
        assert not str(pck["args"].get("fsq_levels") or ""), pck["args"]
        assert int(pck["step"]) == PARENT_STEPS, pck["step"]

        fsq_dir = os.path.join(tmp, "fsqparent")
        run(cmd(fsq_dir, "--steps", "2", f"--fsq-levels={LEVELS}"),
            "FSQ parent")
        fsq_parent = os.path.join(fsq_dir, "pixelmae.pt")
        assert str(load(fsq_parent)["args"]["fsq_levels"]) == LEVELS

        # ---- 1. the pre-existing refusal, pinned -------------------------
        o = run(cmd(os.path.join(tmp, "a"), "--steps", "8", "--resume", cont,
                    f"--fsq-levels={LEVELS}"),
                "case 1: continuous + --fsq-levels, no warmstart",
                want_fail=True)
        assert "REFUSING to resume" in o, o[-1500:]
        assert "fsq_levels" in o and LEVELS in o, o[-1500:]
        # and it refuses BEFORE building the model, at the cost of the inputs
        assert "size mismatch for" not in o, o[-1500:]
        print("1. ok — a continuous checkpoint resumed with an explicit "
              "--fsq-levels and no --fsq-warmstart still REFUSES, naming both "
              "values, before a weight is loaded. This is the door the flag "
              "opens; it is shut without it.")

        # ---- 2. the warm start proceeds, weights bit-for-bit -------------
        # --steps AT the parent's step, so the loop `while s < a.steps` never
        # runs: whatever the child saves is what the warm start LOADED, which
        # is what makes "every weight loads exactly" a reading rather than a
        # claim. (Case 3 does the training half.)
        b_dir = os.path.join(tmp, "b")
        o = run(cmd(b_dir, "--steps", str(PARENT_STEPS), "--resume", cont,
                    f"--fsq-levels={LEVELS}", "--fsq-bound=ln",
                    "--fsq-warmstart"), "case 2: the warm start")
        assert "FSQ WARM START" in o, o[-2500:]
        warm_line = [l for l in o.splitlines() if "FSQ WARM START" in l]
        assert len(warm_line) == 1, warm_line
        assert LEVELS in warm_line[0] and "--fsq-bound ln" in warm_line[0], \
            warm_line[0]
        assert f"at step {PARENT_STEPS}" in warm_line[0], warm_line[0]
        assert "INTRINSIC BOUND --fsq-bound ln" in o, o[-2500:]
        cck = load(os.path.join(b_dir, "pixelmae.pt"))
        # the lattice came from the DISPATCH...
        assert str(cck["args"]["fsq_levels"]) == LEVELS, cck["args"]
        assert str(cck["args"]["fsq_bound"]) == "ln", cck["args"]
        assert cck["args"]["fsq_warmstart"] is True, cck["args"]
        assert (cck["args"]["fsq_warmstart_from"]
                == f"pixelmae.pt@{PARENT_STEPS}"), cck["args"]
        # ...and every weight from the FILE, bit for bit.
        pm, cm = pck["model"], cck["model"]
        assert set(pm) == set(cm), (sorted(set(pm) ^ set(cm)))
        diff = [k for k in pm if not torch.equal(pm[k], cm[k])]
        assert not diff, f"weights changed across the warm start: {diff[:6]}"
        # the step was adopted, so this is a continuation and says so
        assert f"at step {PARENT_STEPS}" in o, o[-2500:]
        # the metrics config record carries the provenance too — it is
        # sometimes the only surviving account of what ran (#387)
        cfg = [json.loads(l) for l in open(os.path.join(b_dir,
                                                        "metrics.jsonl"))
               if '"config"' in l][0]["config"]
        assert cfg["fsq_levels"] == LEVELS, cfg
        assert cfg["fsq_warmstart_from"] == f"pixelmae.pt@{PARENT_STEPS}", cfg
        # and the artefact is an FSQ codec to every loader
        mt = codec_from_ckpt(cck, C)
        assert mt.fsq is not None and mt.fsq_bound == "ln"
        print("2. ok — the warm start proceeds: lattice %s + bound 'ln' taken "
              "from the DISPATCH, all %d weight tensors torch.equal to the "
              "continuous parent's, step %d adopted, fsq_warmstart_from "
              "'%s' in the checkpoint AND in the metrics config record, and "
              "codec_from_ckpt rebuilds it as an FSQ codec"
              % (LEVELS, len(pm), PARENT_STEPS,
                 cck["args"]["fsq_warmstart_from"]))

        # ---- 3. the ladder fit schedule is REBASED ----------------------
        # THE FALSIFIER FIRST: unrebased, "1,2" against an 8-step run is
        # [1, 2] — both already behind a run resuming at step 4, so
        # `s in FSQ_FIT_AT` would never be true and an `auto` lattice would
        # quantize uniformly, unfitted, for the whole run.
        absolute = fql.fit_steps("1,2", PARENT_STEPS + 4)
        assert absolute == [1, 2], absolute
        assert max(absolute) < PARENT_STEPS, (absolute, PARENT_STEPS)
        f_dir = os.path.join(tmp, "f")
        o = run(cmd(f_dir, "--steps", str(PARENT_STEPS + 4), "--resume", cont,
                    f"--fsq-levels={LEVELS}", "--fsq-ladder=auto",
                    "--fsq-auto-step=1,2", "--fsq-bound=ln",
                    "--fsq-warmstart"), "case 3: rebased fit schedule")
        want = [PARENT_STEPS + 1, PARENT_STEPS + 2]
        # the run SAYS which steps it will fit at...
        assert f"re-fitting at {want}" in o, o[-2500:]
        assert f"REBASED onto this step — fitting at {want}" in o, o[-2500:]
        # ...and then actually fits there. `fsq_auto_fit` prints "step N:".
        fits = [l for l in o.splitlines() if "auto fitted on" in l]
        assert len(fits) == 2, fits
        got = [int(l.split("step")[1].split(":")[0]) for l in fits]
        assert got == want, (got, want)
        fck = load(os.path.join(f_dir, "pixelmae.pt"))
        assert fck["args"]["fsq_ladder_fit"], fck["args"]
        recs = [json.loads(l) for l in open(os.path.join(f_dir,
                                                         "metrics.jsonl"))
                if "fsq_ladder_fit" in l and '"step"' in l]
        assert [r["step"] for r in recs] == want, recs
        print("3. ok — --fsq-auto-step 1,2 is read as OFFSETS from the resume "
              "step: the unrebased schedule is %s, entirely behind a run "
              "resuming at %d (so an unchanged comparison would never fit at "
              "all), and the warm start fits at %s — announced in the startup "
              "log, observed in the loop, recorded in metrics.jsonl, and the "
              "fitted lattice '%s' is in the checkpoint"
              % (absolute, PARENT_STEPS, got, fck["args"]["fsq_ladder_fit"]))

        # ---- 4. the four refusals ---------------------------------------
        o = run(cmd(os.path.join(tmp, "d"), "--steps", "2",
                    f"--fsq-levels={LEVELS}", "--fsq-warmstart"),
                "case 4a: no --resume", want_fail=True)
        assert "--fsq-warmstart without --resume" in o, o[-1200:]
        assert "COLD-START" in o, o[-1200:]

        o = run(cmd(os.path.join(tmp, "e"), "--steps", "8", "--resume", cont,
                    "--fsq-warmstart"),
                "case 4b: no --fsq-levels", want_fail=True)
        assert "without an explicit --fsq-levels" in o, o[-1200:]

        o = run(cmd(os.path.join(tmp, "c"), "--steps", "8", "--resume",
                    fsq_parent, f"--fsq-levels={LEVELS}", "--fsq-warmstart"),
                "case 4c: the checkpoint is already FSQ", want_fail=True)
        assert "ALREADY carries" in o and "not a warm start" in o, o[-1500:]
        # …and the same FSQ->FSQ resume WITHOUT the flag is the ordinary
        # continuation it has always been, so 4c refuses the transition and
        # not the checkpoint.
        run(cmd(os.path.join(tmp, "c2"), "--steps", "3", "--resume",
                fsq_parent, f"--fsq-levels={LEVELS}"),
            "case 4c control: FSQ->FSQ without the flag still resumes")

        o = run(cmd(os.path.join(tmp, "g"), "--steps", "8", "--resume",
                    os.path.join(tmp, "no-such-run.pt"),
                    f"--fsq-levels={LEVELS}", "--fsq-warmstart"),
                "case 4d: checkpoint not on this box", want_fail=True)
        assert "no checkpoint at" in o and "COLD-START" in o, o[-1500:]
        print("4. ok — all four refuse, each naming its mechanism: no "
              "--resume, no explicit --fsq-levels, an already-FSQ checkpoint "
              "(while the same FSQ->FSQ resume WITHOUT the flag still "
              "continues normally), and a --resume target this box does not "
              "hold")

        # ---- 5. the hole is FOUR KEYS wide ------------------------------
        o = run(cmd(os.path.join(tmp, "h"), "--steps", "8", "--resume", cont,
                    f"--fsq-levels={LEVELS}", "--fsq-warmstart",
                    "--d-model", "32"),
                "case 5: warmstart does not license --d-model", want_fail=True)
        assert "REFUSING to resume" in o and "dispatch says 32" in o, o[-1500:]
        assert "size mismatch for" not in o, o[-1500:]
        print("5. ok — --fsq-warmstart licenses the four fsq keys and nothing "
              "else: a contradicting --d-model still refuses before the model "
              "is built, exactly as it did for #395")

        # ---- 6. no loader refuses the new keys --------------------------
        # `fsq_warmstart` is in vars(a) on EVERY run from this commit on, and
        # both loaders refuse unknown fsq_* keys by PRESENCE (E-049's clause).
        # If they had not been added to `known`, every checkpoint this repo
        # writes from now on would be unscoreable.
        for ck_path in (os.path.join(b_dir, "pixelmae.pt"), cont):
            ck = load(ck_path)
            assert "fsq_warmstart" in ck["args"], ck["args"].keys()
            codec_from_ckpt(ck, C)          # must not raise
        # the JAX loader's own allowlist, read as source rather than executed
        # (jax is not a dependency of this test; importing convert.py would
        # drag the whole port in for a set literal).
        src = open(os.path.join(ML, "jaxport", "convert.py")).read()
        blk = src[src.index('known = {"fsq_levels"'):]
        blk = blk[:blk.index("}")]
        for k in ("fsq_warmstart", "fsq_warmstart_from"):
            assert k in blk, (k, blk)
        print("6. ok — both unknown-fsq-key loaders read `fsq_warmstart` and "
              "`fsq_warmstart_from` as informational: codec_from_ckpt rebuilds "
              "the warm-started AND the ordinary continuous checkpoint (every "
              "checkpoint written from this commit carries the key), and "
              "ml/jaxport/convert.py's allowlist names both")

        # ---- 7. a contradictory file is refused, not averaged -----------
        # The never-re-fit adoption path is inert on a warm start only because
        # a continuous checkpoint carries no `fsq_ladder_fit` — a property of
        # the FILE. Assert the guard rather than the assumption.
        forged = os.path.join(tmp, "forged.pt")
        f = load(cont)
        f["args"]["fsq_ladder_fit"] = "u:2,e2:0.75,u:2,e3:1.5,u:2,u:2"
        torch.save(f, forged)
        o = run(cmd(os.path.join(tmp, "i"), "--steps", "8", "--resume",
                    forged, f"--fsq-levels={LEVELS}", "--fsq-warmstart"),
                "case 7: continuous ckpt carrying a fitted ladder",
                want_fail=True)
        assert "carries no `fsq_levels`" in o and "fsq_ladder_fit" in o, \
            o[-1500:]
        print("7. ok — a checkpoint with no fsq_levels but a non-empty "
              "fsq_ladder_fit is REFUSED rather than adopted: it would install "
              "a lattice fitted against a bottleneck that is not this one, and "
              "then never re-fit")

        print("\nE-050 --fsq-warmstart: all 7 checks hold ✓")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""The spatial holdout is OPTIONAL, in both stages, and separable in stage 2.

Chris, 2026-08-19: hold out YEARS only, train on every longitude. The
`-45,-25` block stops being a standing training exclusion and becomes a
deliberate diagnostic — it produced the paper's Figure 7 — reached by naming
it rather than by saying nothing.

Three things have to hold for that to be safe, and each is a check below.

**1. `--holdout-lon ""` / `none` really means no longitude is held out.**
Not "a block of width zero somewhere", not "a mask that happens to be empty
because the numbers missed the grid": an all-False bool array, which every
consumer in train.py already handles without an `if` — the anomaly
transform's `keep_x = ~x_hold`, the train pool `obs_any & ~t_hold & ~x_hold`,
the validation pool `obs_any & (t_hold | x_hold)`, and probe_now's
`ok_p = ~x_hold[kxs]`. Check 2 runs the REAL trainer on a fixture whose grid
puts exactly 25% of its columns inside the block, and demands the training
pool grow by exactly 4/3 and the validation pool become the held-out MONTHS
and nothing else.

**2. Stage 2 can be told to open its pool without moving anything else.**
`x_hold` in temporal.py was one variable doing two jobs — the
anomaly-transform statistics and the training pool. `--train-lon-hold`
governs the POOL ONLY. The statistics must keep following the frozen codec's
own saved args, because the embedding cache Z is keyed by (codec weight
hash, sha256 of the raw tensor) and neither term can see the transform: two
runs on the same codec and tensor with different statistics would share one
cache key. Check 4 asserts that separation the only way that cannot be
argued with — by comparing the two runs' Z caches BYTE FOR BYTE.

**3. `inherit` reproduces today's behaviour exactly**, so nothing in flight
moves. Checks 3 and 4 pin that against the default and against an explicit
restatement of the codec's own block.

And check 5 pins the two recipes that will be dispatched against this code,
because a recipe is the dispatch surface: if it expands to a different
parameter set than its `_description` claims, the run is a different run.

**4. What the recipe SAVES has to be re-readable by the rest of the tree.**
The spec lands verbatim in `ck["args"]["holdout_lon"]`, and twelve eval
scripts under `ml/` still parse that field with
`lo, hi = (float(v) for v in ...split(","))` — the whole probe ladder and the
roll. `"none"` raises in every one of them, each raise swallowed by a
best-effort guard, so a "none" codec trains perfectly and produces a GREEN
run with an empty probe archive. Check 6 pins the constraint: the recipe
passes `0,0` — the empty half-open interval, bit-identical mask, parseable by
all twelve — and train.py warns loudly on the form they cannot read. Routing
those twelve through `train.lon_holdout_mask` is the follow-up that retires
this check.

    python3 tests/test_lon_holdout_optional.py
"""
import glob
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
ML = os.path.join(ROOT, "ml")
sys.path.insert(0, ML)
from model import PixelMAE                                     # noqa: E402
from train import lon_holdout_mask                             # noqa: E402

# The fixture grid is chosen so the arithmetic is EXACT rather than
# approximate: 16 columns at 1.25 deg from -60, so lons[12..15] = -45.0,
# -43.75, -42.5, -41.25 are the four inside [-45, -25) and 4/16 = 25.0%.
# Land is a whole ROW, never a patch, so every column carries the same
# number of ocean pixels and the pool ratio is 16/12 = 4/3 on the nose.
T_M, H_G, W_G, C, DZ, K = 36, 8, 16, 4, 4, 6
LONS = (-60 + 1.25 * np.arange(W_G)).astype(np.float32)
LATS = np.linspace(20, 40, H_G).astype(np.float32)
BLOCK = "-45,-25"
N_BLOCK = 4                     # columns inside [-45, -25)
HOLD_YEAR = "1992"              # 12 of the 36 months


def build_npz(path):
    rng = np.random.default_rng(0)
    t = np.arange(T_M)[:, None, None, None]
    X = (np.sin(2 * np.pi * t / 12) + 0.4 * (t / T_M)
         + 0.3 * rng.standard_normal((T_M, H_G, W_G, C))).astype(np.float32)
    X[:, 0, :, :] = np.nan       # land: a whole ROW, so columns stay equal
    months = np.array([f"{1990 + i // 12}-{i % 12 + 1:02d}"
                       for i in range(T_M)])
    ridx = np.arange(K, T_M)
    rapid = np.stack([ridx.astype(float),
                      2.79 * rng.standard_normal(len(ridx))], 1)
    np.savez(path, X=X, months=months, rapid=rapid,
             chan=np.array([f"c{i}" for i in range(C)]),
             lats=LATS, lons=LONS,
             norm=np.zeros((C, 2), dtype=np.float32))
    return X


def run_train(npz, out, extra):
    r = subprocess.run(
        [sys.executable, "-u", os.path.join(ML, "train.py"),
         "--data", npz, "--out", out, "--steps", "1", "--batch", "8",
         "--d-z", str(DZ), "--patch", "1", "--d-model", "16",
         "--n-layers", "1", "--n-heads", "2", "--d-dec", "16",
         "--anomaly", "--eval-every", "0", "--light-probe-every", "0",
         "--holdout-years", HOLD_YEAR, *extra],
        capture_output=True, text=True, timeout=900)
    if r.returncode != 0:
        print(r.stdout[-4000:]); print(r.stderr[-4000:])
        raise SystemExit(f"train.py failed ({' '.join(extra) or 'plain'})")
    m = re.search(r"train pixels ([\d,]+) · held-out pixels ([\d,]+)", r.stdout)
    if not m:
        print(r.stdout[-4000:])
        raise SystemExit("train.py printed no pool line")
    hold = re.search(r"held-out months \d+/\d+ · (.*?) · ocean", r.stdout)
    return (int(m.group(1).replace(",", "")),
            int(m.group(2).replace(",", "")),
            hold.group(1) if hold else "")


def run_temporal(npz, run, tmp, extra):
    env = dict(os.environ, CKPT_DIR_OVERRIDE=os.path.join(tmp, "ckpt"))
    r = subprocess.run(
        [sys.executable, "-u", os.path.join(ML, "temporal.py"),
         "--run", run, "--data", npz, "--K", str(K), "--steps", "20",
         "--batch", "16", "--d-model", "8", "--layers", "1",
         "--lr-warmup", "5", *extra],
        capture_output=True, text=True, timeout=1800, env=env)
    if r.returncode != 0:
        print(r.stdout[-4000:]); print(r.stderr[-4000:])
        raise SystemExit(f"temporal.py failed ({' '.join(extra) or 'plain'})")
    w = re.search(r"train windows: ([\d,]+)", r.stdout)
    lh = re.search(r"lon holdout · statistics \(codec (.*?)\): (\d+)/(\d+) "
                   r"cols · training pool \(--train-lon-hold (.*?)\): "
                   r"(\d+)/(\d+) cols", r.stdout)
    if not w or not lh:
        print(r.stdout[-6000:])
        raise SystemExit("temporal.py printed no pool / lon-holdout line")
    return {"windows": int(w.group(1).replace(",", "")),
            "stat_cols": int(lh.group(2)), "pool_cols": int(lh.group(5)),
            "codec_spec": lh.group(1), "flag": lh.group(4)}


def z_cache_for(run):
    import glob
    g = sorted(glob.glob(os.path.join(ML, "cache", f"Z_{run}_*.npy")))
    if len(g) != 1:
        raise SystemExit(f"expected exactly one Z cache for {run}, got {g}")
    return g[0]


def main():
    tmp = tempfile.mkdtemp()
    runs = ["lonhold_default", "lonhold_inherit", "lonhold_none",
            "lonhold_block"]
    run_dirs = [os.path.join(ML, "runs", r) for r in runs]
    zc = []
    try:
        npz = os.path.join(tmp, "toy.npz")
        build_npz(npz)

        # ---- check 1: the mask itself ---------------------------------
        prod = np.arange(-100, 20, 0.25, dtype=np.float32)   # the real grid
        blk = lon_holdout_mask(BLOCK, prod)
        assert blk.dtype == np.bool_, blk.dtype
        assert int(blk.sum()) == 80, int(blk.sum())          # 20 deg / 0.25
        for spec in ("", "none", "NONE", "  none  ", None):
            m = lon_holdout_mask(spec, prod)
            assert m.dtype == np.bool_, (spec, m.dtype)
            assert m.shape == prod.shape, (spec, m.shape)
            assert not m.any(), f"{spec!r} held out {int(m.sum())} columns"
        # the toy grid, which the rest of the file does arithmetic on
        assert int(lon_holdout_mask(BLOCK, LONS).sum()) == N_BLOCK
        # 80/480 is the COLUMN fraction (16.7%) of the family-3 grid; the
        # 25.0% in EXPERIMENTS.md and tests/test_roll_holdout_lon.py is the
        # fraction of window OCEAN PIXELS, which is larger because the west
        # of that lon range is mostly land. The fixture below is built to
        # make the two coincide at exactly 25%, so the pool ratio is exact.
        print(f"1 ok — lon_holdout_mask: '{BLOCK}' -> 80/{len(prod)} columns "
              f"of the family-3 grid ({100 * 80 / len(prod):.1f}% of columns; "
              f"25.0% of window ocean pixels, per E-022's roll), "
              f"{N_BLOCK}/{W_G} = 25.0% on the fixture; '', 'none', 'NONE', "
              f"'  none  ' and None all -> an all-False bool mask")

        # ---- check 2: stage-1 pools, through the real trainer ---------
        a_tr, a_ho, a_msg = run_train(npz, os.path.join(tmp, "o_blk"),
                                      [f"--holdout-lon={BLOCK}"])
        b_tr, b_ho, b_msg = run_train(npz, os.path.join(tmp, "o_none"),
                                      ["--holdout-lon=none"])
        c_tr, c_ho, c_msg = run_train(npz, os.path.join(tmp, "o_empty"),
                                      ["--holdout-lon="])
        assert (b_tr, b_ho) == (c_tr, c_ho), \
            f"'none' {(b_tr, b_ho)} != '' {(c_tr, c_ho)}"
        assert b_tr > a_tr, f"no-holdout pool {b_tr} not larger than {a_tr}"
        ratio = b_tr / a_tr
        assert abs(ratio - 4 / 3) < 1e-12, f"pool ratio {ratio!r} != 4/3"
        # the totals must be conserved: the block moved from the validation
        # pool into the training pool, and nothing was created or lost.
        assert a_tr + a_ho == b_tr + b_ho, \
            f"pool total moved: {a_tr + a_ho} -> {b_tr + b_ho}"
        # and with no spatial holdout the validation pool is the held-out
        # MONTHS alone: 12 of 36 months x (H-1) ocean rows x all 16 columns.
        assert b_ho == 12 * (H_G - 1) * W_G, b_ho
        assert a_tr == 24 * (H_G - 1) * (W_G - N_BLOCK), a_tr
        assert b_tr == 24 * (H_G - 1) * W_G, b_tr
        assert "NO lon holdout" in b_msg and "NO lon holdout" in c_msg, \
            (b_msg, c_msg)
        assert "held-out lon block" in a_msg, a_msg
        print(f"2 ok — train.py pools: '{BLOCK}' train {a_tr:,} / held "
              f"{a_ho:,}; 'none' train {b_tr:,} / held {b_ho:,}; '' "
              f"identical to 'none'; ratio {ratio:.10f} = 4/3 exactly "
              f"(1/0.75), total conserved at {a_tr + a_ho:,}. The print "
              f"stays truthful: {b_msg!r}")

        # ---- the frozen codec every stage-2 arm below reads ------------
        codec = PixelMAE(n_chan=C, d_model=16, n_heads=2, n_layers=2,
                         d_z=DZ, d_dec=16, patch=1)
        ck = {"model": codec.state_dict(),
              "chan": [f"c{i}" for i in range(C)],
              "d_z": DZ, "norm": None, "step": 0,
              "args": {"patch": 1, "d_model": 16, "n_layers": 2,
                       "n_heads": 2, "d_dec": 16, "anomaly": True,
                       "holdout_years": HOLD_YEAR, "holdout_lon": BLOCK}}
        for d_ in run_dirs:
            os.makedirs(d_, exist_ok=True)
            torch.save(ck, os.path.join(d_, "pixelmae.pt"))

        # ---- check 3: inherit reproduces today's pool exactly ----------
        r_def = run_temporal(npz, runs[0], tmp, [])
        r_inh = run_temporal(npz, runs[1], tmp, ["--train-lon-hold=inherit"])
        r_blk = run_temporal(npz, runs[3], tmp, [f"--train-lon-hold={BLOCK}"])
        assert r_def["windows"] == r_inh["windows"] == r_blk["windows"], \
            (r_def["windows"], r_inh["windows"], r_blk["windows"])
        assert r_def["pool_cols"] == N_BLOCK, r_def["pool_cols"]
        assert r_def["flag"] == "'inherit'", r_def["flag"]
        print(f"3 ok — stage-2 pool unchanged by opting in to today: default "
              f"{r_def['windows']:,} windows == explicit 'inherit' "
              f"{r_inh['windows']:,} == explicit '{BLOCK}' "
              f"{r_blk['windows']:,}, all excluding {N_BLOCK}/{W_G} columns")

        # ---- check 4: 'none' opens the POOL and moves NOTHING else -----
        r_non = run_temporal(npz, runs[2], tmp, ["--train-lon-hold=none"])
        assert r_non["pool_cols"] == 0, r_non["pool_cols"]
        assert r_non["windows"] > r_def["windows"], \
            (r_non["windows"], r_def["windows"])
        wr = r_non["windows"] / r_def["windows"]
        assert abs(wr - 4 / 3) < 1e-12, f"window ratio {wr!r} != 4/3"
        # the statistics followed the CODEC in every arm, whatever the flag
        for tag, r_ in (("default", r_def), ("inherit", r_inh),
                        ("none", r_non), ("block", r_blk)):
            assert r_["stat_cols"] == N_BLOCK, (tag, r_["stat_cols"])
            assert r_["codec_spec"] == f"'{BLOCK}'", (tag, r_["codec_spec"])
        # ...and the only argument that cannot be argued with: Z, byte for
        # byte. The embedding cache is keyed by (codec weight hash, tensor
        # sha256) and is blind to the transform, so if --train-lon-hold had
        # moved the z-score statistics these two files would differ under
        # keys that only differ in the run name.
        za, zb = z_cache_for(runs[0]), z_cache_for(runs[2])
        zc.extend([za, zb])
        A, B = np.load(za), np.load(zb)
        assert A.shape == B.shape, (A.shape, B.shape)
        assert A.dtype == B.dtype, (A.dtype, B.dtype)
        assert np.array_equal(A.view(np.uint8), B.view(np.uint8)), \
            ("the embedding changed when only the TRAINING POOL was asked "
             "to change — --train-lon-hold has reached the anomaly-transform "
             "statistics, which re-normalises the frozen codec's input and "
             "poisons the Z cache both runs share a key on")
        tk = torch.load(os.path.join(run_dirs[2], "temporal.pt"),
                        map_location="cpu", weights_only=False)
        assert tk["args"]["train_lon_hold"] == "none", tk["args"]
        print(f"4 ok — 'none' opens the pool to {r_non['windows']:,} windows "
              f"(x{wr:.10f} = 4/3) with statistics still at the codec's "
              f"{N_BLOCK}/{W_G} columns in all four arms, and the two Z "
              f"caches are byte-identical over {A.size:,} {A.dtype} values")

        # ---- check 5: the recipes expand to what they claim ------------
        want = {
            "f3-anchor-41M-nolonhold": {
                "RECIPE_TENSOR": "family3_na025", "RECIPE_D_Z": "64",
                "RECIPE_PATCH": "3", "RECIPE_CODEC_D_MODEL": "576",
                "RECIPE_CODEC_LAYERS": "10", "RECIPE_CODEC_HEADS": "8",
                "RECIPE_CODEC_D_DEC": "768", "RECIPE_ANOMALY": "true",
                "RECIPE_BATCH": "512", "RECIPE_STEPS": "60000",
                "RECIPE_HOLDOUT_LON": "0,0",
                "RECIPE_NAME": "f3-anchor-41M-nolonhold"},
            "xl144-nolonhold": {
                "RECIPE_TENSOR": "family3_na025", "RECIPE_D_Z": "64",
                "RECIPE_PATCH": "3", "RECIPE_CODEC_D_MODEL": "576",
                "RECIPE_CODEC_LAYERS": "10", "RECIPE_CODEC_HEADS": "8",
                "RECIPE_CODEC_D_DEC": "768", "RECIPE_ANOMALY": "true",
                "RECIPE_BATCH": "512", "RECIPE_STEPS": "60000",
                "RECIPE_EVAL_EVERY": "0", "RECIPE_LIGHT_PROBE_EVERY": "0",
                "RECIPE_TEMPORAL_D_MODEL": "1024",
                "RECIPE_TEMPORAL_LAYERS": "16",
                "RECIPE_TRAIN_LON_HOLD": "none",
                "RECIPE_NAME": "xl144-nolonhold"},
        }
        tail = ("stencil:145,ring:spiral:111-4444-0.71-0.5,seed:0,"
                "sched:expdecay --lr-cooldown-frac 0 "
                "--milestone-steps 600,60000,120000")
        for name, exp in want.items():
            r = subprocess.run(
                ["bash", os.path.join(ROOT, "scripts", "resolve_recipe.sh"),
                 f"recipe:{name},{tail}"],
                capture_output=True, text=True, cwd=ROOT)
            if r.returncode != 0:
                print(r.stdout); print(r.stderr)
                raise SystemExit(f"resolve_recipe.sh refused {name}")
            got = dict(ln.split("=", 1) for ln in r.stdout.splitlines()
                       if re.match(r"^(RECIPE_|WINDOW=)", ln))
            assert got.get("WINDOW") == tail, got.get("WINDOW")
            got.pop("WINDOW")
            assert got == exp, (f"{name} expanded to {got}\n"
                                f"                 expected {exp}")
            d = json.load(open(os.path.join(ML, "recipes", f"{name}.json")))
            assert d.get("_description") and d.get("_provenance"), name
            # a recipe cannot pin `resume` or `temporal_steps` (both read in
            # `if:` conditions) — assert it does not TRY, so the description
            # that tells the dispatcher to state them stays true
            assert "resume" not in d and "temporal_steps" not in d, sorted(d)
        print(f"5 ok — both recipes expand to exactly their intended "
              f"{len(want['f3-anchor-41M-nolonhold']) - 1} / "
              f"{len(want['xl144-nolonhold']) - 1} settings, carry "
              f"_description + _provenance, leave `resume` and "
              f"`temporal_steps` to the dispatch, and pass the window tail "
              f"through untouched")

        # ---- check 6: the spec the recipe emits is one the WHOLE tree
        #               can read, and it is the same empty mask ------------
        # train.py accepts "none"; ck["args"]["holdout_lon"] is then saved
        # verbatim and re-read by twelve eval scripts as
        # `lo, hi = (float(v) for v in ...split(","))`. Those scripts belong
        # to the probe ladder and to the roll, and each raise sits behind a
        # best-effort guard, so a "none" codec would train perfectly and
        # produce a GREEN run with an empty probe archive. Until they are
        # routed through lon_holdout_mask, the recipe must pass a spec they
        # can parse — and this check is what makes that a pinned constraint
        # rather than a remembered one.
        readers = sorted(
            os.path.basename(f) for f in glob.glob(os.path.join(ML, "*.py"))
            if re.search(r'float\(v\)\s+for\s+v\s+in\s+.*holdout_lon',
                         open(f).read()))
        assert len(readers) >= 10, readers
        spec = json.load(open(os.path.join(
            ML, "recipes", "f3-anchor-41M-nolonhold.json")))["holdout_lon"]
        for f in readers:                       # exactly their idiom
            lo, hi = (float(v) for v in spec.split(","))
        m_spec = lon_holdout_mask(spec, prod)
        m_none = lon_holdout_mask("none", prod)
        m_flt = (prod >= lo) & (prod < hi)      # what the readers compute
        assert not m_spec.any() and np.array_equal(m_spec, m_none), spec
        assert np.array_equal(m_spec, m_flt), \
            ("the float() readers would disagree with lon_holdout_mask on "
             f"the recipe's own spec {spec!r}")
        # and the warning fires for the form they cannot read
        r = subprocess.run(
            [sys.executable, "-u", os.path.join(ML, "train.py"),
             "--data", npz, "--out", os.path.join(tmp, "o_warn"),
             "--steps", "1", "--batch", "8", "--d-z", str(DZ), "--patch", "1",
             "--d-model", "16", "--n-layers", "1", "--n-heads", "2",
             "--d-dec", "16", "--anomaly", "--eval-every", "0",
             "--light-probe-every", "0", "--holdout-years", HOLD_YEAR,
             "--holdout-lon=none"], capture_output=True, text=True,
            timeout=900)
        assert r.returncode == 0, r.stderr[-2000:]
        assert "::warning::" in r.stdout and "float()" in r.stdout, \
            "train.py accepted --holdout-lon=none without warning that the "\
            "twelve float() readers cannot re-read it"
        r2 = subprocess.run(
            [sys.executable, "-u", os.path.join(ML, "train.py"),
             "--data", npz, "--out", os.path.join(tmp, "o_nowarn"),
             "--steps", "1", "--batch", "8", "--d-z", str(DZ), "--patch", "1",
             "--d-model", "16", "--n-layers", "1", "--n-heads", "2",
             "--d-dec", "16", "--anomaly", "--eval-every", "0",
             "--light-probe-every", "0", "--holdout-years", HOLD_YEAR,
             f"--holdout-lon={spec}"], capture_output=True, text=True,
            timeout=900)
        assert r2.returncode == 0, r2.stderr[-2000:]
        assert "::warning::" not in r2.stdout, \
            f"the recipe's own spec {spec!r} warns — it should not"
        print(f"6 ok — {len(readers)} scripts under ml/ still re-read "
              f"ck['args']['holdout_lon'] with float() ({', '.join(readers)}); "
              f"the recipe passes {spec!r}, which they parse to the SAME "
              f"all-False mask as 'none', and train.py warns on 'none' and "
              f"stays quiet on {spec!r}")

        print("\nall 6/6 checks hold — the spatial holdout is optional in "
              "both stages, opt-in only, and separable in stage 2")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
        for d_ in run_dirs:
            shutil.rmtree(d_, ignore_errors=True)
        # EVERY arm's Z cache, not just the two check 4 compared. The cache
        # name carries the run name, so a leftover cannot poison a later
        # test — but it is 6 MB per arm on the real tensor and this file
        # should not be the reason a box fills up.
        for r_ in runs:
            for q in glob.glob(os.path.join(ML, "cache", f"Z_{r_}_*")):
                os.remove(q)


if __name__ == "__main__":
    main()

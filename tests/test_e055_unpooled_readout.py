#!/usr/bin/env python3
"""E-055 · the UNPOOLED stage-2 transport read-out, and the pooled one it must
not have touched.

    python3 tests/test_e055_unpooled_readout.py

~2 minutes on two cores. No GPU, no network, no real tensor.

WHY THIS EXISTS. Every stage-2 transport number this programme has published
came through `hid[:, -1].mean(0)` — a spatial mean over the 26.5N section —
and geostrophic transport IS the east-minus-west contrast across that section
(ml/CLAUDE.md 3 / 8.3, "the one comparison still mismatched by construction").
E-055 adds a learned attention pool BESIDE that mean. Two claims follow, and
both are testable:

  · the new read-out can see a contrast the mean annihilates, and
  · nothing about the pooled path moved — not a value, not a key, not the RNG
    stream the run's checkpoint records.

The second is the load-bearing one. 98 archived stage-2 runs read
`rapid_r_kfold`; if fitting the new head shifts the global torch generator,
every number computed after it in the same process becomes a function of
whether E-055 ran, which is exactly the drift that makes an archive
incomparable while every individual number still looks plausible.

FIVE CHECKS.

  1. THE MECHANISM, on a fixture built so the mean CANNOT work: the signal is
     +s on the section's eastern half and -s on its western half, so
     `Z.mean(1)` cancels it exactly and only a read-out that can weight the
     two ends differently has anything to fit. Pooled and unpooled are scored
     through the same year-blocked k-fold on the same rows.
  2. NON-INTERFERENCE, in-process: `probe_kfold.kfold_r` on the pooled
     features returns a BIT-IDENTICAL number before and after the unpooled
     fit, and the global torch / CUDA / numpy RNG states are unchanged by it.
  3. SEED REPRODUCIBILITY: two invocations at one seed are identical arrays,
     and a different seed is a genuinely different draw. Run #116 was
     dispatched as "seed B" and returned a bit-identical number because the
     seeds were hardwired — a seed knob that does nothing is a failure mode
     this repository has already paid for once.
  4. END TO END through the real `ml/temporal.py`, twice on one toy: once as
     it stands, once as it stood at `BASE_SHA` (the commit before E-055). The
     new keys must exist and be finite; every key the base version wrote must
     be present and EQUAL in the new payload; and `temporal.pt`'s saved
     `torch_rng` must match, which is check 2's claim made about the real run
     rather than about a helper.
  5. THE NEW KEYS ARE THE NAMED ONES. `rapid_r_kfold_unpooled`,
     `rapid_r_kfold_unpooled_ci` and `rapid_r_deseas_unpooled`, in the
     `stage2_result` metrics record, beside — never instead of —
     `rapid_r_kfold`, `rapid_r_kfold_ci` and `rapid_r_deseas`.
  6. THE ROLLOUT SIDE, flag ON. `ml/rollout_spatial.py --unpooled-readout`
     twice on one monthly toy against the same run without the flag:
     `amoc_bands_unpooled`, `sv_des_unpooled` and `probe_unpooled` appear,
     are finite, are scored on the same points as their pooled twins and are
     not equal to them — and every other key, `gate` and `gate_ref` included,
     is EQUAL to the flag-off run's. The DEFAULT (no flag) is held by
     tests/test_roll_monthly_identity.py, which demands byte-identity against
     the archive's own evaluator and would fail on any new key; that is why
     the flag exists and why it is off.

`BASE_SHA` is a FIXED anchor, not "the previous commit": once E-055 is
committed, comparing against HEAD would compare the change with itself and
pass vacuously (the pattern is tests/test_roll_monthly_identity.py's).
"""
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
sys.path.insert(0, ML)

from model import PixelMAE                                     # noqa: E402
from probe_kfold import kfold_r                                # noqa: E402
from temporal import (UNPOOLED_HEAD_DIM, UNPOOLED_STEPS,       # noqa: E402
                      attn_pool_kfold, attn_pool_predict,
                      fit_attn_pool, lon_fraction, section_tokens)

# the commit before ml/temporal.py grew an unpooled read-out
BASE_SHA = "e0887d3"
TARGET = "ml/temporal.py"

# --- the end-to-end toy (ml/temporal.py's own CLI, on 40 synthetic months) --
T_M, H_G, W_G, C, DZ, K = 40, 8, 10, 5, 4, 4
STEPS = 40
RUN = "e055toy"


# ---------------------------------------------------------------- check 1 --
def contrast_fixture(seed=7, n_years=8, per=15, P=16, d=6, noise=0.5):
    """Section states whose ONLY signal is an east-minus-west contrast.

    `Z[:, west, 0] = -s` and `Z[:, east, 0] = +s`, so `Z.mean(1)` is exactly
    free of `s` (up to the isotropic noise) while the difference between the
    two ends IS `s`. This is ml/project_amoc.py's measured picture of the real
    section reduced to its essential shape: information that survives a
    difference and not a mean."""
    rng = np.random.default_rng(seed)
    n = n_years * per
    years = np.repeat(np.arange(2004, 2004 + n_years), per)
    s = rng.normal(size=n)
    Z = (rng.normal(size=(n, P, d)) * noise).astype(np.float32)
    half = P // 2
    Z[:, :half, 0] -= s[:, None]
    Z[:, half:, 0] += s[:, None]
    y = 3.0 * s + 0.2 * rng.normal(size=n)
    lonf = lon_fraction(np.linspace(-80.0, -14.0, P))
    return Z, y, years, lonf


def check_mechanism():
    Z, y, years, lonf = contrast_fixture()
    pooled = kfold_r(Z.mean(1), y, years, seed=0)
    tok = section_tokens(Z, lonf)
    assert tok.shape == (Z.shape[0], Z.shape[1], Z.shape[2] + 2), tok.shape
    assert np.allclose(tok[..., :Z.shape[2]], Z)
    assert np.allclose(tok[0, :, Z.shape[2]], lonf)
    assert np.all(tok[..., -1] == 0.0), "the month-offset column is not K=1"
    up = attn_pool_kfold(tok, y, years, seed=0, device="cpu", boot=400)

    for k in ("r", "lo", "hi", "n", "rmse", "sigma", "folds"):
        assert k in up, f"attn_pool_kfold did not return `{k}`"
        assert np.isfinite(up[k]), f"{k} is not finite: {up[k]}"
    assert up["n"] == len(y) and up["folds"] == 8, (up["n"], up["folds"])
    assert np.isfinite(up["pred"]).all(), "an out-of-fold month has no value"
    assert up["lo"] <= up["r"] <= up["hi"], (up["lo"], up["r"], up["hi"])

    print(f"1. one fixture, one target, two read-outs over the SAME "
          f"{up['folds']} year-blocked folds and the same {up['n']} months:")
    print(f"     pooled   `Z.mean(1)` ridge  r {pooled[0]:+.3f} "
          f"[{pooled[1]:+.3f}, {pooled[2]:+.3f}]")
    print(f"     UNPOOLED attention pool    r {up['r']:+.3f} "
          f"[{up['lo']:+.3f}, {up['hi']:+.3f}]  rmse {up['rmse']:.2f}")
    assert up["r"] > 0.5, (
        f"the unpooled read-out scored {up['r']:.3f} on a fixture whose signal "
        f"is a pure east-minus-west contrast — it cannot see the one thing it "
        f"exists to see")
    assert pooled[0] < 0.3, (
        f"the POOLED read-out scored {pooled[0]:.3f} on a fixture whose signal "
        f"the section mean cancels exactly — the fixture is not testing what "
        f"it claims to test")
    print(f"     the mean cancels the signal by construction; the margin "
          f"{up['r'] - pooled[0]:+.3f} is the fixture's, not a skill claim "
          f"about any real codec")


# ---------------------------------------------------------------- check 2 --
def rng_fingerprint():
    return (torch.get_rng_state().clone(),
            torch.cuda.get_rng_state_all() if torch.cuda.is_available()
            else None,
            np.random.get_state()[1].copy())


def same_rng(a, b):
    if not torch.equal(a[0], b[0]):
        return False
    if (a[1] is None) != (b[1] is None):
        return False
    if a[1] is not None and any(not torch.equal(x, y)
                                for x, y in zip(a[1], b[1])):
        return False
    return np.array_equal(a[2], b[2])


def check_non_interference():
    Z, y, years, lonf = contrast_fixture(seed=11, n_years=5, per=12, P=10, d=4)
    tok = section_tokens(Z, lonf)
    F = Z.mean(1)

    torch.manual_seed(1234)
    np.random.seed(1234)
    before = rng_fingerprint()
    pooled_before = kfold_r(F, y, years, seed=0)

    torch.manual_seed(1234)
    np.random.seed(1234)
    attn_pool_kfold(tok, y, years, seed=0, device="cpu", steps=100, boot=50)
    fit_attn_pool(tok[:40], y[:40], seed=0, device="cpu", steps=100)
    after = rng_fingerprint()
    pooled_after = kfold_r(F, y, years, seed=0)

    assert same_rng(before, after), (
        "fitting the unpooled read-out MOVED the global RNG. Every number "
        "computed after it in the same process — including the `torch_rng` "
        "temporal.py saves into its checkpoint — would then depend on whether "
        "E-055 ran")
    for i, name in enumerate(("r", "lo", "hi", "n", "rmse", "sigma")):
        assert pooled_before[i] == pooled_after[i], (
            f"the pooled k-fold's `{name}` moved from {pooled_before[i]} to "
            f"{pooled_after[i]} across an unpooled fit")
    assert np.array_equal(pooled_before[6], pooled_after[6], equal_nan=True), \
        "the pooled k-fold's per-month predictions moved"
    print(f"2. the pooled k-fold is BIT-IDENTICAL across an unpooled fit "
          f"(r {pooled_before[0]!r} both times, {len(pooled_before[6])} "
          f"per-month predictions equal element for element), and the global "
          f"torch/CUDA/numpy RNG states are unchanged by it")


# ---------------------------------------------------------------- check 3 --
def check_seed():
    Z, y, years, lonf = contrast_fixture(seed=3, n_years=5, per=12, P=10, d=4)
    tok = section_tokens(Z, lonf)
    a1 = attn_pool_kfold(tok, y, years, seed=0, device="cpu", steps=150,
                         boot=50)
    a2 = attn_pool_kfold(tok, y, years, seed=0, device="cpu", steps=150,
                         boot=50)
    b = attn_pool_kfold(tok, y, years, seed=3, device="cpu", steps=150,
                        boot=50)
    assert a1["r"] == a2["r"] and np.array_equal(a1["pred"], a2["pred"]), (
        "two invocations at seed 0 disagree — the read-out is not a function "
        "of the data and the seed alone")
    assert a1["lo"] == a2["lo"] and a1["hi"] == a2["hi"], \
        "the block bootstrap is not seeded"
    assert not np.array_equal(a1["pred"], b["pred"]), (
        "seed 3 reproduced seed 0's predictions exactly — the seed knob does "
        "nothing, which is what #116 was dispatched believing it had")

    n1, _ = fit_attn_pool(tok, y, seed=5, device="cpu", steps=120)
    n2, _ = fit_attn_pool(tok, y, seed=5, device="cpu", steps=120)
    s1, s2 = n1.state_dict(), n2.state_dict()
    assert set(s1) == set(s2) and all(torch.equal(s1[k], s2[k]) for k in s1), \
        "two fits at one seed produced different weights"
    p1 = attn_pool_predict(n1, tok[:5], "cpu")
    p2 = attn_pool_predict(n2, tok[:5], "cpu")
    assert np.array_equal(p1, p2)
    print(f"3. seed 0 reproduces to the last bit (r {a1['r']:+.6f} twice, "
          f"CI endpoints equal, {len(s1)} weight tensors torch.equal) and "
          f"seed 3 is a genuinely different draw (r {b['r']:+.3f})")


# ---------------------------------------------------------------- check 4 --
def toy(tmp):
    """A synthetic ocean + a frozen codec, the shape ml/temporal.py wants."""
    rng = np.random.default_rng(0)
    t = np.arange(T_M)[:, None, None, None]
    X = (np.sin(2 * np.pi * t / 12) + 0.4 * (t / T_M)
         + 0.3 * rng.standard_normal((T_M, H_G, W_G, C))).astype(np.float32)
    X[:, 0, 0, :] = np.nan                    # land, so OBS is exercised
    months = np.array([f"{1990 + i // 12}-{i % 12 + 1:02d}"
                       for i in range(T_M)])
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
    return npz, {"model": codec.state_dict(),
                 "chan": [f"c{i}" for i in range(C)],
                 "d_z": DZ, "norm": None, "step": 0,
                 "args": {"patch": 1, "d_model": 16, "n_layers": 2,
                          "n_heads": 2, "d_dec": 16, "anomaly": True,
                          "holdout_years": "1992",
                          "holdout_lon": "-45,-44"}}


def base_copy(tmp):
    """`BASE_SHA`'s temporal.py on disk, runnable against today's ml/.

    It lands in `tmp`, not in `ml/`, because temporal.py resolves `--run`
    against its OWN directory — so the base revision writes its artefacts to
    `tmp/runs/<run>/` and cannot collide with HEAD's. `PYTHONPATH` carries
    `ml/`, so `model`, `probe_kfold` and the rest still come from the working
    tree: the ONLY thing that differs between the two runs is this file."""
    r = subprocess.run(["git", "-C", ROOT, "show", f"{BASE_SHA}:{TARGET}"],
                       capture_output=True, text=True)
    if r.returncode != 0:
        raise SystemExit(
            f"cannot read {TARGET} at {BASE_SHA}: {r.stderr.strip()}. This "
            f"check compares the new read-out against the code every archived "
            f"stage-2 number came from; without it there is no reference and "
            f"it must FAIL rather than pass vacuously.")
    p = os.path.join(tmp, "temporal_base.py")
    open(p, "w").write(r.stdout)
    return p


def train(script, npz, run_dir, tmp, tag):
    for f in ("metrics.jsonl", "temporal.json", "temporal.pt"):
        p = os.path.join(run_dir, f)
        if os.path.exists(p):
            os.remove(p)
    env = dict(os.environ, CKPT_DIR_OVERRIDE=os.path.join(tmp, "ckpt", tag),
               PYTHONPATH=ML + os.pathsep + os.environ.get("PYTHONPATH", ""))
    r = subprocess.run(
        [sys.executable, "-u", script, "--run", os.path.basename(run_dir),
         "--data", npz, "--K", str(K), "--steps", str(STEPS), "--batch", "16",
         "--d-model", "16", "--layers", "2", "--lr", "0.01",
         "--lr-warmup", "5", "--max-pixels", "40", "--seed", "0"],
        capture_output=True, text=True, timeout=3600, env=env, cwd=ROOT)
    if r.returncode != 0:
        print(r.stdout[-4000:])
        print(r.stderr[-4000:])
        raise SystemExit(f"temporal.py failed [{tag}] (rc {r.returncode})")
    recs = [json.loads(l) for l in
            open(os.path.join(run_dir, "metrics.jsonl")) if l.strip()]
    tj = json.load(open(os.path.join(run_dir, "temporal.json")))
    ck = torch.load(os.path.join(run_dir, "temporal.pt"), map_location="cpu",
                    weights_only=False)
    return recs, tj, ck, r.stdout


def stage2_result(recs):
    hits = [r["stage2_result"] for r in recs if "stage2_result" in r]
    assert len(hits) == 1, f"{len(hits)} stage2_result records, expected 1"
    return hits[0]


NEW_JSON_KEYS = {"rapid_probe_kfold_unpooled"}
NEW_RESULT_KEYS = {"rapid_r_kfold_unpooled", "rapid_r_kfold_unpooled_ci",
                   "rapid_r_deseas_unpooled"}


def check_end_to_end(tmp, run_dir):
    npz, ckpt = toy(tmp)
    base = base_copy(tmp)
    base_run_dir = os.path.join(tmp, "runs", RUN)
    os.makedirs(base_run_dir, exist_ok=True)
    torch.save(ckpt, os.path.join(run_dir, "pixelmae.pt"))
    torch.save(ckpt, os.path.join(base_run_dir, "pixelmae.pt"))
    new_recs, new_tj, new_ck, log = train(
        os.path.join(ML, "temporal.py"), npz, run_dir, tmp, "new")
    old_recs, old_tj, old_ck, _ = train(base, npz, base_run_dir, tmp, "base")

    # (a) the new block exists, is finite, and says what it is
    up = new_tj.get("rapid_probe_kfold_unpooled")
    assert up is not None, (
        "ml/temporal.py wrote no `rapid_probe_kfold_unpooled` block. Tail of "
        "the log:\n" + log[-2500:])
    for k in NEW_RESULT_KEYS:
        assert k in up, f"`{k}` missing from the unpooled block: {sorted(up)}"
    assert np.isfinite(up["rapid_r_kfold_unpooled"]), up
    assert all(np.isfinite(v) for v in up["rapid_r_kfold_unpooled_ci"]), up
    assert up["rapid_r_deseas_unpooled"] is None \
        or np.isfinite(up["rapid_r_deseas_unpooled"]), up
    assert up["pooled"] is False and up["readout"]["d"] == UNPOOLED_HEAD_DIM
    assert up["readout"]["steps_max"] == UNPOOLED_STEPS
    assert up["readout"]["section_pixels"] >= 5, up["readout"]
    assert len(up["pred"]) == len(up["target_sv"]) == len(up["years"]), \
        "the paired-test arrays are not the same length"
    print(f"4. ml/temporal.py on a {T_M}-month toy wrote "
          f"rapid_r_kfold_unpooled {up['rapid_r_kfold_unpooled']:+.3f} "
          f"{up['rapid_r_kfold_unpooled_ci']} over {up['n']} months / "
          f"{up['folds']} folds on {up['readout']['section_pixels']} section "
          f"pixels — beside pooled "
          f"{new_tj['rapid_probe_kfold']['r_kfold_deseas']:+.3f} "
          f"{new_tj['rapid_probe_kfold']['ci95']}")

    # (b) EVERY key the base version wrote is present and EQUAL
    assert set(new_tj) - set(old_tj) == NEW_JSON_KEYS, (
        f"temporal.json gained keys beyond the unpooled block: "
        f"{sorted(set(new_tj) - set(old_tj) - NEW_JSON_KEYS)}")
    moved = [k for k in old_tj if new_tj.get(k) != old_tj[k]]
    assert not moved, (
        "THE POOLED STAGE-2 PATH MOVED. 98 archived runs read these keys; a "
        f"changed value makes the column incomparable. Keys: {moved}\n"
        + "\n".join(f"  {k}: {BASE_SHA} {old_tj[k]!r} -> HEAD {new_tj[k]!r}"
                    for k in moved[:6]))

    # (c) the metrics record: new names beside the old ones, old ones equal
    new_r, old_r = stage2_result(new_recs), stage2_result(old_recs)
    assert set(new_r) - set(old_r) == NEW_RESULT_KEYS, (
        f"stage2_result gained keys beyond the three named ones: "
        f"{sorted(set(new_r) - set(old_r) - NEW_RESULT_KEYS)}")
    rmoved = [k for k in old_r if new_r.get(k) != old_r[k]]
    assert not rmoved, f"stage2_result's pooled fields moved: {rmoved}"
    for k in ("rapid_r_kfold", "rapid_r_kfold_ci", "rapid_r_deseas"):
        assert k in new_r, f"the pooled `{k}` was removed from stage2_result"
    assert new_r["rapid_r_kfold_unpooled"] == up["rapid_r_kfold_unpooled"]
    assert new_r["rapid_r_kfold_unpooled_ci"] == up["rapid_r_kfold_unpooled_ci"]
    assert new_r["rapid_r_deseas_unpooled"] == up["rapid_r_deseas_unpooled"]
    print(f"   the {len(old_tj)} keys {BASE_SHA} wrote into temporal.json are "
          f"all present and EQUAL in HEAD's (pooled k-fold "
          f"{old_r['rapid_r_kfold']!r} both times, single split "
          f"{old_r['rapid_r_deseas']!r} both times); the only additions are "
          f"{sorted(NEW_JSON_KEYS)} and, in stage2_result, "
          f"{sorted(NEW_RESULT_KEYS)}")

    # (d) the RNG the run itself saved
    assert new_ck["torch_rng"] == old_ck["torch_rng"], (
        "temporal.pt's saved `torch_rng` differs between the two revisions — "
        "the unpooled fit consumed global randomness the base run did not, so "
        "a resume from this checkpoint would continue a different stream")
    assert new_ck["step"] == old_ck["step"]
    print(f"   and temporal.pt's saved torch_rng is identical "
          f"({len(new_ck['torch_rng'])} bytes of generator state) — the "
          f"unpooled fit restored every generator it touched")
    return new_tj


# ---------------------------------------------------------------- check 5 --
def check_named_keys(tj):
    """The names are the contract. A downstream reader (sweep_table.mjs,
    make_table.py, the status page) addresses these by string, so a rename is
    as breaking as a deletion — and a key BESIDE the pooled one is the whole
    point of §3's `legacy_pooled_*` bridge."""
    up = tj["rapid_probe_kfold_unpooled"]
    pooled = tj["rapid_probe_kfold"]
    assert set(("r_kfold_deseas", "ci95", "features", "note")) <= set(pooled)
    assert "unpooled" not in pooled["features"], \
        "the POOLED block's own `features` string was edited"
    assert pooled["features"] == "hidden(-1) mean over section"
    assert "attention pool" in up["features"], up["features"]
    print("5. the pooled block still describes itself as "
          f"{pooled['features']!r} and the unpooled one as "
          f"{up['features']!r} — beside it, never instead of it")


# ---------------------------------------------------------------- check 6 --
# ml/rollout_spatial.py's side. `read_sv` (:1811) is exception 1 of
# ml/CLAUDE.md §3: three of the four e017_u1_s0 gate criteria come through it
# against a hardcoded GATE_REF at GATE_TOL 0.0101, so the unpooled read-out is
# an ADDITIONAL function behind `--unpooled-readout`, off by default.
# tests/test_roll_monthly_identity.py holds the DEFAULT (byte-identical to the
# archive's own evaluator); this holds the flag ON — the new keys appear, they
# are finite, and NOTHING the pooled path writes moves when they do.
ROLL_NEW_TOP = {"probe_unpooled"}
ROLL_NEW_HEAD = {"amoc_bands_unpooled"}
ROLL_NEW_SERIES = {"sv_des_unpooled"}


def roll(fx, out, cache, extra):
    env = dict(os.environ, PYTHONPATH=ML + os.pathsep
               + os.environ.get("PYTHONPATH", ""))
    r = subprocess.run(
        [sys.executable, "-u", os.path.join(ML, "rollout_spatial.py"),
         "--x", fx["x"], "--npz-small", fx["npz"], "--z", fx["z"],
         "--ckpt", fx["ckpt"], "--out", out, "--horizon", "3",
         "--long-start", "1991-12", "--long-months", "16",
         "--future-months", "5", "--cache-dir", cache, "--no-gate",
         *extra, "--heads", *fx["heads"]],
        capture_output=True, text=True, timeout=3600, env=env, cwd=ROOT)
    if r.returncode != 0:
        print(r.stdout[-4000:])
        print(r.stderr[-4000:])
        raise SystemExit(f"rollout_spatial.py failed {extra} (rc "
                         f"{r.returncode})")
    return json.load(open(out)), r.stdout


def check_rollout(tmp):
    sys.path.insert(0, HERE)
    from test_rollout_spatial import build_fixture              # noqa: E402
    rolldir = os.path.join(tmp, "roll")
    os.makedirs(rolldir, exist_ok=True)
    fx = build_fixture(rolldir)
    off, _ = roll(fx, os.path.join(tmp, "off.json"),
                  os.path.join(tmp, "c_off"), ["--no-unpooled-readout"])
    on, log = roll(fx, os.path.join(tmp, "on.json"),
                   os.path.join(tmp, "c_on"), ["--unpooled-readout"])

    assert set(on) - set(off) == ROLL_NEW_TOP, (
        f"roll.json gained top-level keys beyond {sorted(ROLL_NEW_TOP)}: "
        f"{sorted(set(on) - set(off) - ROLL_NEW_TOP)}")
    meta = on["probe_unpooled"]
    for k in ("fit_on", "fit_rows", "fit_first", "fit_last", "seed",
              "fit_holdout_years_excluded", "steps_max", "section_pixels"):
        assert k in meta, f"probe_unpooled has no `{k}`: {sorted(meta)}"
    assert meta["fit_rows"] > 0 and meta["seed"] == 0
    assert meta["fit_holdout_years_excluded"], \
        "the fit window does not record which years it excluded"

    moved, added_head, n_bands, n_series = [], set(), 0, 0
    for lab, e_on in on["heads"].items():
        e_off = off["heads"][lab]
        added_head |= set(e_on) - set(e_off)
        for k, v in e_off.items():
            if k == "wall_s":                     # a clock reading, not a result
                continue
            w = e_on[k]
            if isinstance(v, dict) and isinstance(w, dict):
                w = {k2: v2 for k2, v2 in w.items()
                     if k2 not in ROLL_NEW_SERIES}
            if w != v:
                moved.append((lab, k))
        # the unpooled bands: same keys, same n, finite r
        assert set(e_on["amoc_bands_unpooled"]) == set(e_on["amoc_bands"]), (
            f"{lab}: the unpooled bands are not the pooled bands' keys: "
            f"{sorted(e_on['amoc_bands_unpooled'])} vs "
            f"{sorted(e_on['amoc_bands'])}")
        for bn, bv in e_on["amoc_bands_unpooled"].items():
            assert bv["n"] == e_on["amoc_bands"][bn]["n"], (
                f"{lab}/{bn}: the two read-outs were scored on different "
                f"numbers of points — they are not a paired pair")
            assert "r" not in bv or np.isfinite(bv["r"]), bv
            n_bands += 1
        for blk in ("long", "future"):
            if blk not in e_on:
                continue
            s_p, s_u = e_on[blk]["sv_des"], e_on[blk]["sv_des_unpooled"]
            assert len(s_p) == len(s_u), (
                f"{lab}/{blk}: {len(s_u)} unpooled steps for {len(s_p)} "
                f"pooled ones")
            assert all(np.isfinite(v) for v in s_u), f"{lab}/{blk} not finite"
            assert s_p != s_u, (
                f"{lab}/{blk}: the unpooled series is element-for-element the "
                f"pooled one — the new read-out is not reading anything new")
            n_series += 1
    assert not moved, (
        "THE POOLED ROLL MOVED when --unpooled-readout was passed. Three of "
        "the four gate criteria come through that path; a changed value there "
        f"ends every eval wave before it scores anything. Keys: {moved}")
    assert added_head == ROLL_NEW_HEAD, (
        f"a head entry gained keys beyond {sorted(ROLL_NEW_HEAD)}: "
        f"{sorted(added_head - ROLL_NEW_HEAD)}")
    assert on["gate"] == off["gate"] and on["gate_ref"] == off["gate_ref"], \
        "the gate block or its reference differs between the two runs"
    assert "not gated" in log, \
        "the log does not say the unpooled bands are ungated"
    print(f"6. ml/rollout_spatial.py --unpooled-readout on the monthly toy: "
          f"{n_bands} unpooled band r's and {n_series} unpooled roll series "
          f"written, all finite and none equal to their pooled twin; every "
          f"other key across {len(on['heads'])} heads is EQUAL to the "
          f"flag-off run's, gate and gate_ref included, and the fit window "
          f"({meta['fit_rows']} train rows "
          f"{meta['fit_first']}..{meta['fit_last']}, holdout "
          f"{','.join(meta['fit_holdout_years_excluded'])} excluded, seed "
          f"{meta['seed']}) travels in the artefact")


def main():
    tmp = tempfile.mkdtemp()
    run_dir = os.path.join(ML, "runs", RUN)
    os.makedirs(run_dir, exist_ok=True)
    ok = True
    try:
        check_mechanism()
        check_non_interference()
        check_seed()
        check_named_keys(check_end_to_end(tmp, run_dir))
        check_rollout(tmp)
        print("\nE-055 unpooled read-out: all 6 checks hold ✓")
    except AssertionError as e:
        ok = False
        print(f"\nFAILED: {e}")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
        shutil.rmtree(run_dir, ignore_errors=True)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""The in-training probe's cost, and what may NOT change while it falls.

Run #419 (E-043f, the fresh 38.0M daily codec on all longitude columns) spent
**62.4% of its trainer-loop wall clock inside probes** — 72,249.9 s of
115,729.0 s — and 89.9% of that was embedding. Three of the four savings that
followed are supposed to cost exactly nothing, and "supposed to" is the word
this file exists to remove. Each case pins one of them against the code it
replaced, on a CPU fixture, with the agreement MEASURED and printed rather
than asserted:

Case 1 · the hoisted ocean mask is ELEMENT-WISE IDENTICAL to the one every
         probe used to recompute — and train.py's line-295 expression, which
         is what gets passed in, is that same expression on that same array.
Case 2 · `Fsec` sliced out of the S5 (~864-pixel) embedding agrees with a
         separate S3 (266-pixel) embedding. NOT bitwise: the GEMM's row count
         changes, so the reduction order may. The tolerance is measured here
         and quoted in trainprobe.py's comment.
Case 3 · `embed_everything(t_sel=...)` returns BIT-IDENTICAL rows to the
         corresponding rows of a full pass. This one IS bitwise — every
         timestep is an independent forward — so anything less is a bug.
Case 4 · the probe-count arithmetic used to choose the new cadence defaults
         matches what train.py:992-993 actually does, counted off a real run's
         metrics.jsonl rather than off a reading of the source.
Case 5 · the collapse guard still fires on BOTH light and full probes. This
         is the constraint the cadence change is not allowed to break: the
         guard's detection latency tracks the LIGHT cadence, which is why
         eval_every could be cut to 25,000 while light_probe_every stayed at
         2,000 (worst case 2 x 2,000 = 4,000 steps, unchanged).
Case 6 · the light probe's RAPID thinning: one sample per decorrelation time
         at whatever cadence the tensor runs, and the LIGHT_MIN_TEST floor
         walks the stride back rather than starving the guard's scored sample.
Case 7 · the workflow defaults are the ones the arithmetic was done for, and
         they satisfy the guard-span constraint.

    python3 tests/test_probe_cost.py
"""
import json
import os
import re
import subprocess
import sys
import tempfile

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ML = os.path.join(HERE, "..", "ml")
WF = os.path.join(HERE, "..", ".github", "workflows", "ml-train.yml")
sys.path.insert(0, ML)

from model import PixelMAE, LazyPixels                       # noqa: E402
from temporal import embed_everything, rapid_section, CACHE_DTYPE  # noqa: E402
from trainprobe import cadence_days, light_rows, LIGHT_MIN_TEST  # noqa: E402

FAILS = []


def fail(msg):
    FAILS.append(msg)
    print(f"  FAIL: {msg}")


# --- the fixture ------------------------------------------------------------
# Wide enough in longitude that the 26.5N section carries a real number of
# pixels (the batch-row count is the whole point of case 2) and deep enough in
# time that a mean over T is not a two-term sum.
T, H, W, C, D_Z = 40, 14, 48, 6, 8


def toy_tensor(seed=0):
    rng = np.random.default_rng(seed)
    t = np.arange(T)[:, None, None, None]
    X = (np.sin(2 * np.pi * t / 12)
         + 0.3 * rng.standard_normal((T, H, W, C))).astype(np.float32)
    # land, so `ocean` is not trivially all-True and .any(axis=0) has work
    X[:, :2, :, :] = np.nan
    X[:, :, :3, 0] = np.nan
    # one pixel observed in channel 0 for exactly one timestep — the case that
    # separates `.any(axis=0)` from "observed at t=0"
    X[:, 5, 7, 0] = np.nan
    X[3, 5, 7, 0] = 1.5
    lats = np.linspace(18.0, 34.0, H).astype(np.float32)
    lons = np.linspace(-78.0, -12.0, W).astype(np.float32)
    return X, lats, lons


def toy_codec(seed=0):
    torch.manual_seed(seed)
    m = PixelMAE(n_chan=C, d_model=16, n_heads=2, n_layers=2, d_z=D_Z,
                 d_dec=24, patch=1)
    m.eval()
    return m


def ctx_of(T_):
    moy = np.arange(T_) % 12
    return np.stack([np.sin(2 * np.pi * moy / 12),
                     np.cos(2 * np.pi * moy / 12)], 1).astype(np.float32)


# --- case 1 -----------------------------------------------------------------
def case1():
    X, lats, lons = toy_tensor()
    per_probe = np.isfinite(X[..., 0]).any(axis=0)      # trainprobe's old line
    hoisted = np.isfinite(X[..., 0]).any(axis=0)        # train.py:295
    if not np.array_equal(per_probe, hoisted):
        fail("case 1: the two spellings of the ocean mask differ")
        return
    # and the hoist must actually be WIRED: train.py hands it to probe_now,
    # probe_now accepts it, and probe_now only recomputes when it is None.
    tp = open(os.path.join(ML, "trainprobe.py")).read()
    tr = open(os.path.join(ML, "train.py")).read()
    if "ocean=None" not in tp.split("def probe_now")[1].split('"""')[0]:
        fail("case 1: probe_now has no `ocean` parameter")
    code = re.findall(r'^\s+ocean = np\.isfinite\(d\["X"\]\[\.\.\., 0\]\)'
                      r'\.any\(axis=0\)\s*$', tp, re.M)
    if len(code) != 1:
        fail(f"case 1: probe_now evaluates the ocean mask at {len(code)} "
             f"places; exactly one, the `ocean is None` fallback, is right")
    if "if ocean is None:" not in tp:
        fail("case 1: probe_now does not guard the recompute")
    sites = len(re.findall(r"dynamic, ocean=ocean, \*\*kw", tr))
    if sites != 2:
        fail(f"case 1: train.py passes ocean=ocean at "
             f"{sites} of its 2 probe_now call sites — the "
             f"CUDA-OOM fallback path is a probe too, and it is the one that "
             f"runs when the box is under pressure")
    m = re.search(r"^\s*ocean = (np\.isfinite\(X\[\.\.\., 0\]\)\.any\(axis=0\))",
                  tr, re.M)
    if not m:
        fail("case 1: train.py no longer computes the mask as "
             "np.isfinite(X[..., 0]).any(axis=0) — the hoisted value and the "
             "probe's fallback have drifted apart")
    else:
        print(f"  mask expressions identical, {int(per_probe.sum())}/"
              f"{H * W} ocean cells, and both call sites wired")
    print("case 1 ok — hoisted ocean mask is element-wise identical to the "
          "per-probe one, and train.py passes it to both call sites")


# --- case 2 -----------------------------------------------------------------
def case2():
    X, lats, lons = toy_tensor()
    codec = toy_codec()
    Xt, OBS = LazyPixels(X), LazyPixels(X, obs=True)
    ctx_all = ctx_of(T)
    ocean = np.isfinite(X[..., 0]).any(axis=0)
    ys, xs = np.where(ocean)
    sec_y, sec_sel = rapid_section(lats, lons, ys, xs)
    if len(sec_sel) < 8:
        fail(f"case 2: the fixture's section is {len(sec_sel)} pixels — too "
             f"few for the row-count change to mean anything")
        return
    rng = np.random.default_rng(0)
    keep = rng.choice(len(ys), min(300, len(ys)), replace=False)
    keep = np.union1d(keep, sec_sel)
    if len(keep) < 3 * len(sec_sel):
        fail("case 2: `keep` is not much larger than the section, so the "
             "GEMM row count barely changes and the test is vacuous")

    # S3: the section on its own (what the full probe used to do FIRST)
    Zsec, _ = embed_everything(codec, Xt, OBS, ctx_all, lats, lons,
                               ys[sec_sel], xs[sec_sel], D_Z)
    # S5: the superset, sliced (what it does now)
    Z, _ = embed_everything(codec, Xt, OBS, ctx_all, lats, lons,
                            ys[keep], xs[keep], D_Z)
    sec_in_keep = np.where(np.isin(keep, sec_sel))[0]
    if len(sec_in_keep) != len(sec_sel):
        fail("case 2: `keep` is not a superset of the section — the whole "
             "derivation rests on np.union1d having put it there")
        return
    Zk = np.asarray(Z[:, sec_in_keep])
    if Zsec.dtype != CACHE_DTYPE or Zk.dtype != CACHE_DTYPE:
        fail(f"case 2: cache dtype moved ({Zsec.dtype}, {Zk.dtype})")

    a, b = np.asarray(Zsec, np.float64), Zk.astype(np.float64)
    same = int((np.asarray(Zsec) == Zk).sum())
    n = a.size
    scale = np.abs(a).mean()
    rel = np.abs(a - b).max() / scale
    f3 = np.asarray(Zsec).mean(1).astype(np.float64)
    f5 = Zk.mean(1).astype(np.float64)
    frel = np.abs(f3 - f5).max() / max(np.abs(f3).mean(), 1e-12)
    print(f"  Z elements bit-identical: {same}/{n} ({100 * same / n:.3f}%) · "
          f"max |dZ| / mean|Z| = {rel:.3e}")
    print(f"  Fsec (the mean over the section) max relative deviation "
          f"= {frel:.3e}")
    # The claim in trainprobe.py is "~1e-6 relative, not bitwise". Give it an
    # order of magnitude of room and fail loudly if the two paths ever diverge
    # for a REASON rather than for a reduction order.
    if not (rel < 1e-3 and frel < 1e-3):
        fail(f"case 2: the S5-derived section disagrees with the S3 one by "
             f"{rel:.3e} (Fsec {frel:.3e}) — that is a different computation, "
             f"not a reordered reduction")
    # AND at the geometry that actually matters. The toy above is 16-wide;
    # #419's codec is 512x12 over C=39 at d_z 32, and whether a GEMM reorders
    # its reduction is a function of the shape the BLAS sees. One forward at
    # the real row counts (266 vs 864) is seconds and answers it directly.
    torch.manual_seed(0)
    big = PixelMAE(n_chan=39, d_model=512, n_heads=4, n_layers=12, d_z=32,
                   d_dec=256, patch=1).eval()
    rg = np.random.default_rng(0)
    n_keep, n_sec = 864, 266
    xv = torch.as_tensor(rg.standard_normal((n_keep, 39)), dtype=torch.float32)
    ov = torch.as_tensor(rg.random((n_keep, 39)) > 0.15)
    mv = torch.zeros(n_keep, 39, dtype=torch.bool)
    cv = torch.as_tensor(rg.standard_normal((n_keep, 4)), dtype=torch.float32)
    sec = np.sort(rg.choice(n_keep, n_sec, replace=False))   # union1d's order
    with torch.no_grad():
        zb = big.encode(xv, ov, mv, cv)[torch.as_tensor(sec)].double().numpy()
        zs = big.encode(xv[sec], ov[sec], mv[sec], cv[sec]).double().numpy()
    hit = int((zb == zs).sum())
    rel2 = np.abs(zb - zs).max() / np.abs(zb).mean()
    print(f"  at #419's geometry (512x12, C=39, d_z=32, {n_sec} -> {n_keep} "
          f"rows): {hit}/{zb.size} float32 elements bit-identical, "
          f"max |dz| / mean|z| = {rel2:.3e}")
    if rel2 > 1e-4:
        fail(f"case 2: at the real geometry the two row counts disagree by "
             f"{rel2:.3e} — too large for a reduction order")
    print("case 2 ok — the section sliced out of the 864-pixel embedding is "
          "the same section, to the measured tolerance above")


# --- case 3 -----------------------------------------------------------------
def case3():
    X, lats, lons = toy_tensor()
    codec = toy_codec()
    Xt, OBS = LazyPixels(X), LazyPixels(X, obs=True)
    ctx_all = ctx_of(T)
    ocean = np.isfinite(X[..., 0]).any(axis=0)
    ys, xs = np.where(ocean)
    _, sec_sel = rapid_section(lats, lons, ys, xs)
    full, _ = embed_everything(codec, Xt, OBS, ctx_all, lats, lons,
                               ys[sec_sel], xs[sec_sel], D_Z)
    tsel = np.arange(2, T, 5)
    part, _ = embed_everything(codec, Xt, OBS, ctx_all, lats, lons,
                               ys[sec_sel], xs[sec_sel], D_Z, t_sel=tsel)
    if part.shape != (len(tsel), len(sec_sel), D_Z):
        fail(f"case 3: t_sel returned {part.shape}, want "
             f"{(len(tsel), len(sec_sel), D_Z)}")
        return
    if not np.array_equal(np.asarray(full)[tsel], np.asarray(part)):
        d = np.abs(np.asarray(full)[tsel].astype(np.float64)
                   - np.asarray(part).astype(np.float64)).max()
        fail(f"case 3: t_sel rows are NOT bit-identical to the full pass "
             f"(max |d| = {d:.3e}). Every timestep is an independent forward; "
             f"anything but equality means state is crossing `t`.")
    # out-of-order and repeated selections must still line up row for row
    odd = np.array([7, 1, 7, 0, T - 1])
    p2, _ = embed_everything(codec, Xt, OBS, ctx_all, lats, lons,
                             ys[sec_sel], xs[sec_sel], D_Z, t_sel=odd)
    if not np.array_equal(np.asarray(full)[odd], np.asarray(p2)):
        fail("case 3: t_sel does not honour the order it was given")
    # and it must REFUSE to write a time-subset cache
    try:
        embed_everything(codec, Xt, OBS, ctx_all, lats, lons,
                         ys[sec_sel], xs[sec_sel], D_Z, t_sel=tsel,
                         cache_path="/tmp/should-never-exist.npy")
        fail("case 3: t_sel + cache_path was accepted — a partial-time array "
             "of the right shape is indistinguishable from a complete one")
    except ValueError:
        pass
    print(f"case 3 ok — t_sel embeds {len(tsel)}/{T} timesteps, bit-identical "
          f"to those rows of a full pass, order-preserving, cache refused")


# --- case 4 -----------------------------------------------------------------
T_M, H_G, W_G, C_G = 96, 10, 12, 5
ARCH = ["--d-z", "4", "--patch", "3", "--d-model", "16",
        "--n-layers", "1", "--n-heads", "2", "--d-dec", "24"]


def trainer_npz(tmp):
    rng = np.random.default_rng(0)
    t = np.arange(T_M)[:, None, None, None]
    X = (np.sin(2 * np.pi * t / 12) + 0.4 * (t / T_M)
         + 0.3 * rng.standard_normal((T_M, H_G, W_G, C_G))).astype(np.float32)
    X[:, 0, 0, :] = np.nan
    months = np.array([f"{1990 + i // 12}-{i % 12 + 1:02d}" for i in range(T_M)])
    ridx = np.arange(6, T_M)
    rapid = np.stack([ridx.astype(float),
                      2.79 * rng.standard_normal(len(ridx))], 1)
    npz = os.path.join(tmp, "toy.npz")
    np.savez(npz, X=X, months=months, rapid=rapid,
             chan=np.array([f"c{i}" for i in range(C_G)]),
             norm=np.ones((C_G, 2), np.float32),
             lats=np.linspace(20, 40, H_G).astype(np.float32),
             lons=np.linspace(-60, -40, W_G).astype(np.float32))
    return npz


def trainer(npz, out, extra, steps):
    return subprocess.run(
        [sys.executable, "-u", os.path.join(ML, "train.py"),
         "--data", npz, "--out", out, "--steps", str(steps), "--batch", "16",
         "--anomaly", "--holdout-years", "1996", "--holdout-lon=-45,-40",
         *ARCH, *extra],
        capture_output=True, text=True, timeout=1800)


def expected_counts(steps, eval_every, light_every, resumed=False):
    """The counting rule at ml/train.py:992-993, spelled out.

        full_here = eval_every and (s % eval_every == 0 or s == steps)
        light     = light_every and not full_here and s % light_every == 0

    over `s = 0; while s < steps: s += 1`, i.e. s in 1..steps — plus the
    step-0 probe, which is FULL when eval_every is set and LIGHT otherwise,
    and is skipped entirely on a resumed run.
    """
    full = light = 0
    if not resumed and (eval_every or light_every):
        full += 1 if eval_every else 0
        light += 0 if eval_every else 1
    for s in range(1, steps + 1):
        here = bool(eval_every) and (s % eval_every == 0 or s == steps)
        if here:
            full += 1
        elif light_every and s % light_every == 0:
            light += 1
    return full, light


def case4():
    tmp = tempfile.mkdtemp()
    npz = trainer_npz(tmp)
    for steps, E, L in ((60, 25, 10), (50, 0, 12), (40, 13, 0)):
        out = os.path.join(tmp, f"c4_{steps}_{E}_{L}")
        # --collapse-r 0 because this case counts PROBES, and the toy is
        # small enough that a real early-training r of +0.03 is an honest
        # reading: leaving the guard armed would abort the run and truncate
        # the very count being checked. Case 5 is where the guard is tested.
        r = trainer(npz, out, ["--eval-every", str(E), "--collapse-r", "0",
                               "--light-probe-every", str(L)], steps)
        if r.returncode != 0:
            print((r.stdout + r.stderr)[-2500:])
            fail(f"case 4: the trainer failed at {E}/{L}")
            continue
        recs = [json.loads(x) for x in open(os.path.join(out, "metrics.jsonl"))
                if x.strip()]
        got_f = sum(1 for x in recs
                    if "linear_r_deseas" in x and not x.get("light"))
        got_l = sum(1 for x in recs if x.get("light"))
        want_f, want_l = expected_counts(steps, E, L)
        if (got_f, got_l) != (want_f, want_l):
            fail(f"case 4: steps={steps} eval_every={E} light={L} produced "
                 f"{got_f} full + {got_l} light, the counting rule says "
                 f"{want_f} + {want_l}")
        else:
            print(f"  steps={steps:>3} eval_every={E:>3} light={L:>3} -> "
                  f"{got_f} full + {got_l} light, as counted")
    # and the rule reproduces the number the cadence decision was made on:
    # #419 was dispatched 7500/10000 over 200,000 steps.
    f419, l419 = expected_counts(200_000, 7500, 10_000)
    if (f419, l419) != (28, 13):
        fail(f"case 4: the counting rule gives {f419} full + {l419} light for "
             f"#419's 7500/10000 over 200k, but 28 x 2,296.6 s + 13 x 611.3 s "
             f"= 72,251.7 s is what reproduces its measured 72,249.9 s")
    print(f"case 4 ok — probe counts match train.py:992-993 on three cadences, "
          f"and #419's 7500/10000 counts to {f419} full + {l419} light")


# --- case 5 -----------------------------------------------------------------
def case5():
    tmp = tempfile.mkdtemp()
    npz = trainer_npz(tmp)
    # --collapse-r 1.1 is above every possible correlation, so every probe is
    # a strike. The point is WHICH probes can strike, not the threshold.
    for label, extra in (("light only", ["--light-probe-every", "10",
                                         "--eval-every", "0"]),
                         ("full only", ["--eval-every", "10",
                                        "--light-probe-every", "0"])):
        out = os.path.join(tmp, "coll_" + label.replace(" ", "_"))
        r = trainer(npz, out, extra + ["--collapse-r", "1.1"], 60)
        txt = r.stdout + r.stderr
        if r.returncode == 0 or "ABORTING at step" not in txt:
            print(txt[-2500:])
            fail(f"case 5: the collapse guard did not fire on {label} probes. "
                 f"The guard is the ONLY monitor that can see a dead codec, "
                 f"and the cadence split assumes it fires on both.")
            continue
        recs = [json.loads(x) for x in open(os.path.join(out, "metrics.jsonl"))
                if x.strip()]
        if not any("collapsed" in x for x in recs):
            fail(f"case 5: {label} aborted with no {{'collapsed'}} record")
        print(f"  {label}: aborted, collapsed record written")
    print("case 5 ok — the collapse guard fires on both light and full probes")


# --- case 6 -----------------------------------------------------------------
def case6():
    # cadence_days must read the three shapes the tensors actually use
    monthly = [f"{1990 + i // 12}-{i % 12 + 1:02d}" for i in range(24)]
    daily = [f"2004-{4 + i // 30:02d}-{i % 30 + 1:02d}" for i in range(24)]
    pentad = ["2004-04-01", "2004-04-06", "2004-04-11", "2004-04-16",
              "2004-04-21", "2004-04-26", "2004-05-01"]
    for label, months, want in (("monthly", monthly, 30.0),
                                ("daily", daily, 1.0),
                                ("pentad", pentad, 5.0)):
        got = cadence_days(months)
        if abs(got - want) > 1.5:
            fail(f"case 6: cadence_days({label}) = {got}, want ~{want}")
    # the stride: one RAPID sample per decorrelation time
    n = 4000
    ridx = np.arange(n)
    te = np.zeros(n, bool); te[: n // 7] = True          # ~3 years of 20
    tr = ~te
    for label, months, want in (("daily", daily, 13), ("pentad", pentad, 2),
                                ("monthly", monthly, 1)):
        sel, stride = light_rows(ridx, tr, te, months)
        if stride != want:
            fail(f"case 6: stride at {label} cadence is {stride}, want {want} "
                 f"(RAPID_TAU_DAYS // cadence_days)")
        if len(sel) != len(range(0, n, want)):
            fail(f"case 6: {label} kept {len(sel)} rows, want "
                 f"{len(range(0, n, want))}")
        print(f"  {label:<8} stride {stride:>2} · {len(sel):>5} of {n} RAPID "
              f"samples · {int(te[sel].sum()):>4} of them held out")
    # the floor must WALK THE STRIDE BACK rather than starve the guard
    small = np.arange(120)
    te_s = np.zeros(120, bool); te_s[:40] = True
    sel, stride = light_rows(small, ~te_s, te_s, daily)
    kept_te = int(te_s[sel].sum())
    if kept_te < LIGHT_MIN_TEST:
        fail(f"case 6: on a short record the thinning left {kept_te} held-out "
             f"samples, under the LIGHT_MIN_TEST floor of {LIGHT_MIN_TEST}")
    if stride >= 13:
        fail("case 6: the stride was not walked back on a short record")
    print(f"  short record: stride walked 13 -> {stride} to keep {kept_te} "
          f">= {LIGHT_MIN_TEST} held-out samples")
    print("case 6 ok — one RAPID sample per decorrelation time, with the "
          "held-out floor enforced against the tensor in hand")


# --- case 7 -----------------------------------------------------------------
def case7():
    import yaml
    d = yaml.safe_load(open(WF))
    inp = d[True]["workflow_dispatch"]["inputs"]
    if len(inp) > 25:
        fail(f"case 7: {len(inp)} workflow inputs — the cap is 25 and a 26th "
             f"makes the WHOLE file unparseable")
    E = int(inp["eval_every"]["default"])
    L = int(inp["light_probe_every"]["default"])
    span = 2 * L
    if span > 4000:
        fail(f"case 7: the collapse guard's worst-case detection span is "
             f"2 x light_probe_every = {span} steps, worse than the 4,000 it "
             f"had at light_probe_every 2000. The guard fires on BOTH probes, "
             f"so cutting probe cost must come out of eval_every.")
    if L > E:
        fail(f"case 7: light_probe_every {L} > eval_every {E} — the CHEAP "
             f"probe is less frequent than the expensive one. That is the "
             f"shape #419 was dispatched with (7500/10000) and it is "
             f"backwards: the light probe carries the guard.")
    if E < 4 * L:
        fail(f"case 7: eval_every {E} is under 4x light_probe_every {L}; the "
             f"full probe is ~100x the light one at daily cadence, so the two "
             f"cadences must not scale together")
    # The stale help said "~300 s each on the 10M codec" / "section probe
    # only, ~30 s" — a family-3 figure, wrong for every tensor in use. Pin the
    # retirement by the stale phrasing AND by the presence of the run the new
    # numbers were measured on, so a future edit cannot quietly regress to a
    # figure with no provenance.
    for key, stale in (("eval_every", "each on the 10M codec"),
                       ("light_probe_every", "section probe only, ~30 s")):
        desc = inp[key]["description"]
        if stale in desc:
            fail(f"case 7: {key}'s description still carries the stale "
                 f"family-3 figure {stale!r}")
        if "#419" not in desc:
            fail(f"case 7: {key}'s description quotes no measured run — a "
                 f"cost figure without its provenance is what went stale")
    print(f"case 7 ok — {len(inp)}/25 inputs · eval_every {E} · "
          f"light_probe_every {L} · guard worst case {span} steps (<= 4,000)")


def main():
    for fn in (case1, case2, case3, case4, case5, case6, case7):
        fn()
    if FAILS:
        raise SystemExit(f"\n{len(FAILS)} FAILURE(S):\n  - "
                         + "\n  - ".join(FAILS))
    print("\nall 7 probe-cost guards hold")


if __name__ == "__main__":
    main()

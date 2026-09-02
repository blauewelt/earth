#!/usr/bin/env python3
"""ml/lim_baseline.py — the LIM baseline, before it is pointed at 8 GB.

ml/CLAUDE.md §4.8: exercise the code path on a toy before spending the
expensive resource. Three checks, and the middle one is the load-bearing one.

  1. THE OPERATOR IS RECOVERED. A field driven by a KNOWN stable linear
     operator plus noise, x(t+1) = B x(t) + eps. At full rank the fitted
     propagator must reproduce B — an exact expected value with a stated
     tolerance (§4.9), not "the forecast looks reasonable". Then, on data the
     fit never saw and at a TRUNCATED K, the LIM must beat persistence at
     lead 1 and relax toward climatology as the lead grows: those two together
     are what a baseline has to do, and a model that did the first without the
     second would be diverging.

  2. THE SCORING IS THE BATTERY'S, NOT A COPY OF IT. Two halves. First an
     IDENTITY check: the names `ml/lim_baseline.py` scores through are the
     same OBJECTS `ml/rollout_spatial.py` scores heads through — `is`, not
     "looks the same". Second a NUMERIC check: the same predicted arrays,
     pushed through `lim_baseline.score_battery` and through a reference loop
     written here from scratch against `rollout_spatial`'s own
     new_sums/accumulate/skill_block, must produce byte-equal skill blocks.
     The identity check alone could be satisfied by a file that imported the
     right function and called it on the wrong arrays; the numeric check
     alone could be satisfied by two copies of the same arithmetic. Together
     they say what the LIM row's comparability rests on.

  3. `--smoke` RUNS END TO END, on CPU, in under a minute, and writes an
     artefact with the archive's shape and no non-finite number in it
     (ml/CLAUDE.md §5.22).

    python3 tests/test_lim_baseline.py
"""
import json
import math
import os
import subprocess
import sys
import tempfile
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
ML = os.path.join(ROOT, "ml")
sys.path.insert(0, ML)

import lim_baseline as lim                                      # noqa: E402
import rollout_spatial as rs                                    # noqa: E402


# ----------------------------------------------------------------- check 1 --
def stable_operator(D, rho, seed):
    """A random D x D operator with spectral radius exactly `rho`."""
    rng = np.random.default_rng(seed)
    B = rng.standard_normal((D, D)) / math.sqrt(D)
    ev = np.abs(np.linalg.eigvals(B)).max()
    return B * (rho / ev)


def check_operator_recovery():
    """THE STATED TOLERANCE IS THREE STATEMENTS, NOT ONE THRESHOLD.

    A single "|G - B| < eps" pins nothing useful: at any finite sample the
    lag-1 regression carries irreducible sampling noise, so eps would be a
    number chosen to pass rather than a property of the estimator. What can be
    asserted exactly (ml/CLAUDE.md §4.9) is that the pipeline IS the estimator
    it claims to be, and that the estimator behaves:

      (i)  `pca_gram` + `lim_propagator` at full rank must equal an
           INDEPENDENT float64 ordinary-least-squares fit of B to 1e-3 —
           they are the same estimator written two ways, so this is close to
           an identity and it is what would break if the Gram route, the
           mean removal, the mode ordering or the C(0) pair set were wrong;
      (ii) the error against the TRUE operator must SHRINK with sample size,
           at roughly the 1/sqrt(n) rate consistency demands;
      (iii) at the largest sample it must be under a stated 0.12 relative
           Frobenius, which is where the measured 1/sqrt(n) curve sits.
    """
    D, T, rho, noise = 24, 4000, 0.85, 0.4
    B = stable_operator(D, rho, 0)
    rng = np.random.default_rng(1)
    X = np.zeros((T, D))
    for t in range(1, T):
        X[t] = B @ X[t - 1] + noise * rng.standard_normal(D)
    X = X[500:]                                   # drop the spin-up

    fits, rels = {}, {}
    for n_tr in (750, 1500, 3000):
        A = np.ascontiguousarray(X[:n_tr], dtype=np.float32)
        mu, V, scores, sv, tot = lim.pca_gram(A, D)
        assert V.shape == (D, D), V.shape
        # V's columns are orthonormal, so V V^T = I and the physical-space
        # propagator is V G V^T. That is the object to compare with B.
        assert np.allclose(V.T @ V, np.eye(D), atol=2e-4), \
            "the EOF patterns are not orthonormal"
        G, n_pairs = lim.lim_propagator(scores, np.arange(n_tr), D)
        assert n_pairs == n_tr - 1, (n_pairs, n_tr)
        B_hat = V @ G @ V.T
        # (i) the independent reference: float64 OLS of x(t+1) on x(t), on
        # the SAME pairs and CENTRED THE SAME WAY. `pca_gram` removes the
        # training mean (the LIM works on anomalies about it), so an
        # uncentred OLS is a different estimator and differs by ~7e-3 at
        # n=750 — which is the mean's own contribution, not an error.
        mu64 = X[:n_tr].mean(axis=0)
        B_ols = np.linalg.lstsq(X[:n_tr - 1] - mu64, X[1:n_tr] - mu64,
                                rcond=None)[0].T
        gap = float(np.linalg.norm(B_hat - B_ols) / np.linalg.norm(B_ols))
        assert gap < 1e-3, (
            f"n={n_tr}: the Gram-PCA propagator is {gap:.2e} away from a "
            f"plain float64 OLS fit of the same pairs — they must be the "
            f"same estimator")
        rels[n_tr] = float(np.linalg.norm(B_hat - B) / np.linalg.norm(B))
        fits[n_tr] = (mu, V, scores, G)
    # (ii) consistency, and (iii) the stated tolerance.
    assert rels[3000] < rels[1500] < rels[750], rels
    assert rels[3000] < 0.60 * rels[750], (
        f"quadrupling the sample must roughly halve the error (1/sqrt(n)): "
        f"{rels[750]:.4f} at n=750 -> {rels[3000]:.4f} at n=3000")
    assert rels[3000] < 0.12, (
        f"the fitted propagator is {rels[3000]:.4f} away from the operator "
        f"that generated the data (relative Frobenius), tolerance 0.12")
    mu, V, scores, G = fits[3000]
    diag = lim.lim_diagnostics(G, step_days=5.0)
    assert diag["stable"], diag
    assert abs(diag["spectral_radius"] - rho) < 0.03, (diag, rho)
    # e-folding of the leading mode: tau / -ln(rho). Checked against the
    # closed form rather than eyeballed.
    want = 5.0 / (-math.log(diag["spectral_radius"]))
    assert abs(diag["leading_efolding_days"] - want) < 1e-2, (diag, want)
    print("1a. the full-rank fit IS a float64 OLS of the same pairs (< 1e-3 "
          "apart at every n), and its distance to the generating operator "
          "falls %.4f -> %.4f -> %.4f at n = 750/1500/3000 (tol 0.12); "
          "spectral radius %.4f vs the true %.2f, leading e-folding %.2f d "
          "at tau 5 d ✓"
          % (rels[750], rels[1500], rels[3000], diag["spectral_radius"],
             rho, diag["leading_efolding_days"]))

    # ---- truncated K, on data the fit never saw --------------------------
    n_tr = 3000
    k = D // 2
    Gk, _ = lim.lim_propagator(scores, np.arange(n_tr), k)
    Vk, muk = V[:, :k], mu
    test = X[n_tr:]
    leads = (1, 2, 4, 8, 16, 32)
    mse_l, mse_p, amp = {}, {}, {}
    for h in leads:
        s = np.arange(0, len(test) - h)
        a = (test[s] - muk) @ Vk                  # [n, k]
        for _ in range(h):
            a = a @ Gk.T
        pred = muk + a @ Vk.T
        truth = test[s + h]
        mse_l[h] = float(((pred - truth) ** 2).mean())
        mse_p[h] = float(((test[s] - truth) ** 2).mean())
        amp[h] = float(np.sqrt((pred ** 2).mean()))
    clim = float((test ** 2).mean())
    assert mse_l[1] < 0.85 * mse_p[1], (
        f"the LIM must beat persistence at lead 1: mse {mse_l[1]:.4f} vs "
        f"persistence {mse_p[1]:.4f}")
    assert mse_l[32] < 1.05 * clim, (
        f"at lead 32 the forecast must have relaxed to climatology, not "
        f"overshot it: mse {mse_l[32]:.4f} vs climatological {clim:.4f}")
    assert amp[32] < 0.15 * amp[1], (
        f"the forecast amplitude must decay toward the mean: "
        f"{amp[32]:.4f} at lead 32 vs {amp[1]:.4f} at lead 1")
    for h1, h2 in zip(leads, leads[1:]):
        assert mse_l[h1] <= mse_l[h2] + 1e-9, (
            f"skill must not improve with lead: {h1} -> {h2}")
    print(f"1b. K={k} of {D} on held-out data: lead-1 mse {mse_l[1]:.4f} vs "
          f"persistence {mse_p[1]:.4f} (beats it); mse rises monotonically "
          f"to {mse_l[32]:.4f} against climatology {clim:.4f}, and the "
          f"forecast amplitude decays {amp[1]:.3f} -> {amp[32]:.4f} ✓")


# ----------------------------------------------------------------- check 2 --
def check_scoring_identity():
    # (a) the same OBJECTS, not the same source
    shared = ("StdMonths", "ar1_train", "corridor_pixels", "gate_subset",
              "nested_scopes", "new_sums", "accumulate", "skill_block",
              "scored_horizon", "write_results", "Progress", "TimeAxis")
    for name in shared:
        assert getattr(lim, name) is getattr(rs, name), (
            f"lim_baseline.{name} is not rollout_spatial.{name} — the LIM "
            f"would be scored by a copy of the battery, and a copy can drift")
    print(f"2a. all {len(shared)} battery names in lim_baseline ARE "
          f"rollout_spatial's own objects ({', '.join(shared)}) ✓")

    # (b) the same NUMBERS, from a reference loop written here
    rng = np.random.default_rng(7)
    T, P, C, Hh = 40, 17, 4, 5
    months = np.array([f"{1990 + i // 12}-{i % 12 + 1:02d}"
                       for i in range(T)])
    ax = rs.TimeAxis({"months": months})
    hold_years = ["1992"]
    starts_per_year = 0
    Vfld = rng.standard_normal((T, P, C)).astype(np.float32)
    Ofld = rng.random((T, P, C)) > 0.15
    Vfld = np.where(Ofld, Vfld, 0.0).astype(np.float32)
    r1 = rng.random((P, C)).astype(np.float32) * 0.9
    corridor = rng.random(P) > 0.5
    scopes = rs.nested_scopes(
        (("gate", rng.random(P) > 0.7), ("corridor", corridor),
         ("window", np.ones(P, bool))),
        rng.random(P) > 0.6)
    chan_names = [f"c{i}" for i in range(C)]
    leads = ax.daymatched_leads()

    # ONE deterministic prediction per (start, lead), shared by both paths.
    XH = {}
    for Y in hold_years:
        for s in ax.starts_for_year(Y, starts_per_year):
            XH[s] = rng.standard_normal((Hh + 1, P, C)).astype(np.float32)

    class Field:
        """The `StdMonths` surface `score_battery` uses: `.get(t)` and a
        poppable `.cache`."""

        def __init__(self):
            self.cache = {}

        def get(self, t):
            self.cache[t] = True
            return Vfld[t], Ofld[t]

    def predict(s):
        for h in range(1, Hh + 1):
            yield XH[s][h]

    sums_a, info = lim.score_battery(Field(), ax, hold_years,
                                     starts_per_year, Hh, T, r1, scopes, C,
                                     chan_names, predict)
    got = {n: rs.skill_block(sums_a[n], Hh, n_px=int(m.sum()), leads=leads,
                             chan_names=chan_names) for n, m in scopes}

    # ---- the reference: rollout_spatial's own three functions, driven by a
    # loop written from scratch here, in the shape rollout_spatial.main()'s
    # own scoring loop has.
    ref = {n: rs.new_sums(Hh, C) for n, _ in scopes}
    n_steps = 0
    for Y in hold_years:
        for s in ax.starts_for_year(Y, starts_per_year):
            if s + 1 >= T:
                continue
            v_pers, obs_s = Vfld[s], Ofld[s]
            for h in range(1, Hh + 1):
                t_tgt = s + h
                if t_tgt >= T or ax.year[t_tgt] != int(Y):
                    break
                xhat = XH[s][h]
                v_true, obs_tt = Vfld[t_tgt], Ofld[t_tgt]
                op = obs_tt & obs_s
                v_damp = v_pers * r1 ** h
                for name, m_ in scopes:
                    rs.accumulate(ref[name], h, xhat[m_], v_true[m_],
                                  v_pers[m_], v_damp[m_], op[m_])
                n_steps += 1
    want = {n: rs.skill_block(ref[n], Hh, n_px=int(m.sum()), leads=leads,
                              chan_names=chan_names) for n, m in scopes}

    assert n_steps == info["scored_steps"] > 0, (n_steps, info)
    assert json.dumps(got, sort_keys=True) == json.dumps(want, sort_keys=True), \
        "the LIM's skill blocks differ from the reference accumulation"
    rows = sum(len(b["chan_skill"]) for b in got.values())
    chans = sum(len(r) for b in got.values()
                for r in b.get("per_channel", {}).values())
    assert rows and chans, (rows, chans)
    print(f"2b. {info['starts']} starts x {info['scored_steps']} scored steps "
          f"over {len(scopes)} scopes: {rows} pooled rows and {chans} "
          f"per-channel rows are EQUAL to the reference accumulation, field "
          f"for field ✓")

    # (c) a channel the model does not touch must REFUSE rather than be
    # scored against a predicted zero.
    unmod = np.array([C - 1])
    hit = False
    try:
        lim.score_battery(Field(), ax, hold_years, starts_per_year, Hh, T,
                          r1, scopes, C, chan_names, predict,
                          unmodelled=unmod)
    except SystemExit as e:
        hit = "does not model" in str(e)
    assert hit, ("an unmodelled channel with observations must be a refusal, "
                 "not a silent zero")
    print("2c. an unmodelled channel that IS observed at a scored cell is a "
          "refusal, not a silent zero ✓")


# ----------------------------------------------------------------- check 3 --
def check_smoke():
    tmp = tempfile.mkdtemp(prefix="lim_test_")
    out = os.path.join(tmp, "rollout_spatial.json")
    t0 = time.time()
    r = subprocess.run(
        [sys.executable, "-u", os.path.join(ML, "lim_baseline.py"),
         "--smoke", "--out", out, "--K", "20,40", "--progress-every", "1000"],
        capture_output=True, text=True, timeout=300, cwd=ROOT)
    el = time.time() - t0
    if r.returncode != 0:
        print(r.stdout[-4000:])
        print(r.stderr[-4000:])
        raise SystemExit("lim_baseline.py --smoke failed")
    assert el < 60, f"--smoke took {el:.1f}s; the contract is under a minute"
    d = json.load(open(out))
    assert "in_progress" not in d, d.get("in_progress")
    assert d["model"]["model"] == "lim", d["model"]
    assert d["horizon"] == 73 and d["hold_years"] == ["1991"], d["horizon"]
    assert set(d["heads"]) == {"lim_k20", "lim_k40"}, list(d["heads"])
    assert d["starts"]["per_year"] == 3, d["starts"]
    assert d["gate"]["skipped"] is True and d["gate"]["certified"] is False
    assert d["cadence"]["name"] == "pentad", d["cadence"]
    want_scopes = {f"{b}{c}" for b in ("gate", "corridor", "window")
                   for c in ("", "_trainlon", "_holdlon")}
    for lab, e in d["heads"].items():
        assert want_scopes <= set(e), sorted(want_scopes - set(e))
        for sc in ("gate", "corridor", "window"):
            blk = e[sc]
            assert blk["chan_skill"], f"{lab}/{sc}: no rows"
            assert blk["n_px"] > 0 and "horizon_auc" in blk
            assert "horizon_auc_daymatched" in blk
            assert blk["per_channel"], f"{lab}/{sc}: no per-channel rows"
            for row in blk["chan_skill"]:
                assert set(row) == {"h", "n", "msss_clim", "msss_pers",
                                    "msss_damped", "acc", "amp_ratio"}, row
        assert e["meta"]["eigen"]["stable"] is True, e["meta"]["eigen"]
        assert e["meta"]["tau_steps"] == 1
        assert e["meta"]["n_train_pairs"] > 0

    def walk(o, p="results"):
        if isinstance(o, dict):
            for k, v in o.items():
                yield from walk(v, f"{p}.{k}")
        elif isinstance(o, list):
            for i, v in enumerate(o):
                yield from walk(v, f"{p}[{i}]")
        elif isinstance(o, float) and not math.isfinite(o):
            yield p, o
    bad = list(walk(d))
    assert not bad, f"non-finite values in the artefact: {bad[:5]}"
    print(f"3. --smoke ran end to end in {el:.1f}s on CPU and wrote a "
          f"{os.path.getsize(out):,}-byte artefact: {len(d['heads'])} LIM "
          f"entries x {len(want_scopes)} scopes, all seven skill fields per "
          f"row, per-channel rows present, no non-finite value anywhere ✓")

    # The unstable-K refusal is the §5.22 guard, and it must FIRE rather than
    # write -inf: at the toy's rank the last modes are noise directions the
    # lag-1 regression fits with a gain above 1.
    out2 = os.path.join(tmp, "unstable.json")
    r2 = subprocess.run(
        [sys.executable, "-u", os.path.join(ML, "lim_baseline.py"),
         "--smoke", "--out", out2, "--K", "20,400", "--progress-every",
         "1000"], capture_output=True, text=True, timeout=300, cwd=ROOT)
    assert r2.returncode == 0, r2.stdout[-2000:] + r2.stderr[-2000:]
    d2 = json.load(open(out2))
    uns = [k for k, e in d2["heads"].items() if e["meta"].get("unscored")]
    assert uns, ("the clamped-to-full-rank K should have been undamped and "
                 "refused; if the toy changed, this check needs a new K")
    assert not list(walk(d2)), "the unstable entry leaked a non-finite value"
    for k in uns:
        assert "corridor" not in d2["heads"][k], \
            "an unscored entry must carry NO skill rows"
        assert d2["heads"][k]["meta"]["eigen"]["spectral_radius"] >= 1.0
    print(f"3b. an undamped propagator ({', '.join(uns)}) is reported with "
          f"its spectrum and NOT scored — no -inf reaches the artefact "
          f"(ml/CLAUDE.md §5.22) ✓")


def main():
    check_operator_recovery()
    check_scoring_identity()
    check_smoke()
    print("\nlim_baseline: all checks hold ✓")


if __name__ == "__main__":
    main()

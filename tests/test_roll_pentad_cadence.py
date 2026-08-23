#!/usr/bin/env python3
"""The roll on a PENTAD axis: does it advance the tensor's own time?

E-044 puts a 206.5M stage-2 head on the pentad tensor, and the programme's
headline instrument for a stage-2 head is rolled corridor AUC
(ml/CLAUDE.md §3, "What this programme is building" §3). Until 2026-08-19
`ml/rollout_spatial.py` could not produce that number at pentad: four things
in it read `months` — a LABEL that build_family4.py emits once per 5-day bin
— as if it were a unique key or a calendar step.

This test runs the REAL evaluator on a pentad-shaped toy and asks the
question that distinguishes the two stories (ml/CLAUDE.md §4.10): WHAT DATE
DOES THE MODEL THINK IT IS AT EACH STEP? The season token the head is fed is
recorded at every forward, by wrapping `roll_step`, and checked against the
bin's true month. A roll whose labels are wrong scores plausible numbers; a
roll whose INPUT is wrong scores plausible numbers too, and neither says so.

It then runs the SAME fixture through `BASE_SHA`'s evaluator — the version
every published roll came from — and prints what that one does, so the fix is
evidenced rather than asserted.

    python3 tests/test_roll_pentad_cadence.py

No GPU and no real tensor: two years of 5-day bins over the toy ocean
tests/test_rollout_spatial.py already rolls.
"""
import datetime as dt
import importlib.util
import json
import math
import os
import shutil
import subprocess
import sys
import tempfile

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
ML = os.path.join(ROOT, "ml")
sys.path.insert(0, HERE)
sys.path.insert(0, ML)
from test_rollout_spatial import build_fixture, EPOCH, K       # noqa: E402
from test_roll_monthly_identity import BASE_SHA, base_copy     # noqa: E402

DAYS = 5
LONG_START = "1990-07"
N_LONG, N_FUT, HORIZON = 16, 5, 3


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def decode_moy(mfeat):
    """The month-of-year the model was actually told, per window slot."""
    m = np.asarray(mfeat.detach().cpu().numpy(), dtype=np.float64)
    ang = np.arctan2(m[:, 0], m[:, 1])          # sin, cos
    return (ang * 12.0 / (2 * math.pi)) % 12.0


def run_roll(mod, f, out, cache, extra=()):
    """Run `mod.main()` in-process, recording (phase, season tokens) per step."""
    calls, phases = [], []
    real_roll, real_step = mod.roll_step, mod.Progress.step

    def spy_roll(model, Zwin, NBR_t, static_ctx, mfeat, chunk, amp=False,
                 **kw):
        calls.append(decode_moy(mfeat))
        return real_roll(model, Zwin, NBR_t, static_ctx, mfeat, chunk, amp,
                         **kw)

    def spy_step(self, phase, n=1):
        phases.append(phase)
        return real_step(self, phase, n)

    mod.roll_step, mod.Progress.step = spy_roll, spy_step
    argv = sys.argv
    sys.argv = ["rollout_spatial.py",
                "--x", f["x"], "--npz-small", f["npz"], "--z", f["z"],
                "--ckpt", f["ckpt"], "--out", out, "--horizon", str(HORIZON),
                "--long-start", LONG_START, "--long-months", str(N_LONG),
                "--future-months", str(N_FUT), "--cache-dir", cache,
                *extra, "--heads", *f["heads"][:1]]
    os.makedirs(cache, exist_ok=True)
    try:
        mod.main()
    finally:
        sys.argv = argv
        mod.roll_step, mod.Progress.step = real_roll, real_step
    assert len(calls) == len(phases), (len(calls), len(phases))
    return json.load(open(out)), list(zip(phases, calls))


def window_defects(steps, phase=None):
    """Windows whose season tokens cannot be K consecutive pentads.

    K = 6 pentad rows span 30 days, so the window crosses AT MOST one month
    boundary: at most 2 distinct tokens, and every adjacent pair differs by 0
    or 1 month. A window advancing a month per step has 6 distinct tokens and
    5 advances — the signature of the pre-fix roll, and it is a property of
    the MODEL'S INPUT, not of any label."""
    bad = 0
    for ph, m in steps:
        if phase and ph != phase:
            continue
        d = [round((m[j + 1] - m[j]) % 12) for j in range(len(m) - 1)]
        if len(set(np.round(m, 6))) > 2 or any(x not in (0, 1) for x in d):
            bad += 1
    return bad


def phase_rows(ax, steps, phase, r0):
    """(exact deviation, n checked) of the recorded tokens for a sequential
    roll from row r0 against the bins' TRUE months."""
    dev, n = 0.0, 0
    i = 0
    for ph, m in steps:
        if ph != phase:
            continue
        for j, got in enumerate(m):
            want = ax.moy_of_row(r0 - K + 1 + i + j)
            dev = max(dev, abs(((got - want + 6) % 12) - 6))
            n += 1
        i += 1
    return dev, n


def main():
    tmp = tempfile.mkdtemp()
    try:
        f = build_fixture(tmp, holdout_lon="0,0", cadence_days=DAYS)
        rs = load(os.path.join(ML, "rollout_spatial.py"), "rs_head")
        ax = rs.TimeAxis(np.load(f["npz"], allow_pickle=False))
        months = [str(m) for m in f["months"]]

        # ---- 1. the cadence is DERIVED, and the index is 1:1 with rows ----
        assert ax.monthly is False and ax.days == DAYS, (ax.cadence, ax.days)
        assert ax.cadence == "pentad" and ax.T == f["T"]
        assert abs(ax.steps_per_year - 365.2425 / 5) < 1e-9
        assert "bin_index" in ax.detected_from and "pentad_days" in ax.detected_from
        naive = {m: i for i, m in enumerate(months)}
        rows = [ax.row_of_label(ax.label_of_row(r)) for r in range(ax.T)]
        assert rows == list(range(ax.T)), \
            "the axis index is not 1:1 with rows"
        assert len({ax.label_of_row(r) for r in range(ax.T)}) == ax.T
        print("1. cadence DERIVED from the tensor (%s); the row index is 1:1 "
              "with the axis: %d labels for %d rows, where the months array "
              "gives %d keys for %d rows (%.2f:1, last-wins)"
              % (ax.detected_from, ax.T, ax.T, len(naive), len(months),
                 len(months) / len(naive)))

        # ---- 2. staggered starts span the axis's real steps-per-year ------
        Y = "1991"
        n_rows_Y = int((ax.year == int(Y)).sum())
        starts = ax.starts_for_year(Y)
        assert len(starts) == n_rows_Y, (len(starts), n_rows_Y)
        assert starts[0] == int(np.where(ax.year == int(Y))[0][0]) - 1
        assert len(set(starts)) == len(starts)
        print("2. holdout %s has %d axis rows and gets %d staggered starts "
              "(the row before %s plus every row inside it but the last); the "
              "old protocol offered 12" % (Y, n_rows_Y, len(starts), Y))

        # ---- 3. run it, and watch what the MODEL is told the date is ------
        res, steps = run_roll(rs, f, os.path.join(tmp, "roll.json"),
                              os.path.join(tmp, "cache"))
        n_by = {}
        for ph, _ in steps:
            n_by[ph] = n_by.get(ph, 0) + 1
        r0_long = ax.row_of_label(LONG_START)
        dev_l, n_l = phase_rows(ax, steps, "long", r0_long)
        dev_f, n_f = phase_rows(ax, steps, "future", ax.T - 1)
        bad_skill = window_defects(steps, "skill")
        # TOL is float32 sin/cos round-trip only: the token is encoded as
        # (sin, cos) in float32 and decoded back through atan2 here, so the
        # identity is exact in the value and ~1e-8 in the encoding.
        TOL = 1e-6
        assert dev_l < TOL and n_l == N_LONG * K, (dev_l, n_l)
        assert dev_f < TOL and n_f == N_FUT * K, (dev_f, n_f)
        assert bad_skill == 0, bad_skill
        print("3. EXACT invariant on the model's OWN INPUT: over %d roll "
              "steps (%s), every season token equals the month of the bin it "
              "stands for — max deviation %.1e over %d long-roll tokens and "
              "%.1e over %d future-roll tokens, and 0 of %d scored windows "
              "hold a span a 6-bin (30-day) window could not cover"
              % (len(steps), " · ".join(f"{k} {v}" for k, v in n_by.items()),
                 dev_l, n_l, dev_f, n_f, n_by["skill"]))

        # ---- 4. the truth attach keeps the pentad series -------------------
        ridx = set(int(v) for v in f["ridx"])
        e = res["heads"]["s1_s0"]
        rolled = [r0_long + 1 + i for i in range(N_LONG)]
        want = sum(1 for r in rolled if r in ridx)
        got = (e["long"]["n_trained"] or 0) + (e["long"]["n_heldout"] or 0)
        assert got == want < N_LONG, (got, want, N_LONG)
        # and the band counts: every (start, h) whose target row carries truth
        n_band = 0
        for s in ax.starts_for_year(Y):
            if s - K + 1 < 0 or s + 1 >= ax.T:
                continue
            for h in range(1, HORIZON + 1):
                t = s + h
                if t >= ax.T or ax.year[t] != int(Y):
                    break
                n_band += t in ridx
        # the first DAY-DEFINED band is h1-18 (5-90 d) — at HORIZON=3 only
        # h1..3 are rolled, so it holds exactly the same points the old
        # step-defined h1-3 did, under a key that names the duration
        band = e["amoc_bands"]["h1-18_5-90d"]
        assert band["n"] == n_band, (band["n"], n_band)
        print("4. truth attaches on the AXIS ROW: the long roll keeps %d of "
              "the %d rolled rows that HAVE a label (100.0%% of them, %d "
              "rolled in all) and the first day-band carries all %d (start, "
              "horizon) pairs whose target row has a RAPID value"
              % (got, want, N_LONG, band["n"]))

        # ---- 5. labels carry their unit, and advance ONE bin per step -----
        ym = e["long"]["roll_ym"]
        assert len(ym) == N_LONG
        ds = [dt.date.fromisoformat(v) for v in ym]
        assert all((ds[i + 1] - ds[i]).days == DAYS for i in range(len(ds) - 1))
        assert ds[0] == EPOCH + dt.timedelta(
            days=int(ax.bins[r0_long] + 1) * DAYS)
        fut = [dt.date.fromisoformat(v) for v in e["future"]["roll_ym"]]
        assert all((fut[i + 1] - fut[i]).days == DAYS
                   for i in range(len(fut) - 1))
        assert e["long"]["context_end"] == ax.label_of_row(r0_long)
        print("5. every rolled label is an ISO DATE exactly %d days after the "
              "last (%s … %s for the hindcast, %s … %s past the record) — one "
              "axis step per step, and the label says which unit it is in"
              % (DAYS, ym[0], ym[-1], e["future"]["roll_ym"][0],
                 e["future"]["roll_ym"][-1]))

        # ---- 6. the artefact says what cadence it is, and does not bluff --
        cad = res["cadence"]
        assert cad["name"] == "pentad" and cad["step_days"] == DAYS
        assert cad["horizon_span_days"] == HORIZON * DAYS
        assert cad["starts_per_holdout_year"][Y] == len(starts)
        assert cad["lowpass_steps"] == ax.steps_for_months(18) > 18
        assert set(e["amoc_bands"]) <= set(e["amoc_bands_def"])
        assert all("_" in k and k.endswith("d")
                   for k in e["amoc_bands_def"]), e["amoc_bands_def"]
        assert e["amoc_bands_def"]["h1-18_5-90d"]["span_days"] == [5, 90]
        assert e["amoc_bands_def"]["h37-73_185-365d"]["span_days"] == [185, 365]
        assert e["amoc_bands_def"]["h1-18_5-90d"]["steps"] == [1, 2, 3], \
            "a band's `steps` must be filtered by the rolled horizon"
        g = res["gate"]
        assert g["pass"] is None and g["skipped"] is True
        assert g["certified"] is False and g["cadence"] == "pentad"
        assert "MONTHLY" in g["reason"] and rs.GATE_HEAD in g["reason"]
        assert res["gate_ref"]["reference"] is None
        assert json.dumps(res).count("NaN") == 0
        print("6. the artefact declares its axis (%s, %g d/step, horizon %g d, "
              "%d starts in %s, 18-month lowpass = %d steps), labels its bands "
              "with day spans (%s), and records the gate as SKIPPED with a "
              "reason rather than passed: %s…"
              % (cad["name"], cad["step_days"], cad["horizon_span_days"],
                 cad["starts_per_holdout_year"][Y], Y, cad["lowpass_steps"],
                 ", ".join(sorted(e["amoc_bands_def"])), g["reason"][:60]))

        # ---- 6b. THE BANDS ARE CUT IN DAYS, and the cut is EXACT ---------
        # ml/CLAUDE.md §4.9: an invariant with a KNOWN answer. `BANDS` was a
        # module constant over h1..12; at --horizon 73 that covered the first
        # 60 of 365 days and left 61 leads in no band. The edges are now the
        # tropical year quartered/halved/whole, and the ONE thing they must
        # not do is move the monthly partition every archived band correlation
        # and the #217 gate are keyed on.
        m_ax = rs.TimeAxis({"months": np.array(
            ["%04d-%02d" % (1990 + i // 12, i % 12 + 1) for i in range(36)])})
        assert m_ax.bands() == (("h1-3", (1, 2, 3)), ("h4-6", (4, 5, 6)),
                                ("h7-12", tuple(range(7, 13)))), m_ax.bands()
        assert [m_ax.band_key(b, h) for b, h in m_ax.bands()] \
            == ["h1-3", "h4-6", "h7-12"]
        pb = ax.bands()
        assert [(n, min(h), max(h)) for n, h in pb] \
            == [("h1-18", 1, 18), ("h19-36", 19, 36), ("h37-73", 37, 73)], pb
        # every step from 1 to a year is in exactly ONE band, at both cadences
        for a_ in (m_ax, ax):
            cov = [h for _, hs in a_.bands() for h in hs]
            assert cov == sorted(cov) == list(range(1, len(cov) + 1)) \
                and len(set(cov)) == len(cov), cov
            assert len(cov) == a_.steps_for_months(12), (len(cov), a_.cadence)
            for _, hs in a_.bands():          # each band's span in DAYS
                assert a_.span_days(max(hs)) <= 365.2425 + 1e-6
        assert res["cadence"]["band_edge_days"] == list(rs.BAND_EDGE_DAYS)
        print("6b. bands are cut at DAY edges (%s d): the monthly axis gives "
              "h1-3 / h4-6 / h7-12 EXACTLY — the literal they replaced — and "
              "the pentad axis gives %s, i.e. the same three durations. Every "
              "step 1..%d falls in exactly one band at both cadences"
              % ("/".join("%g" % v for v in rs.BAND_EDGE_DAYS),
                 " / ".join(ax.band_key(n, h) for n, h in pb),
                 ax.steps_for_months(12)))

        # ---- 6c. horizon_auc_daymatched: the only cross-cadence number ---
        leads = ax.daymatched_leads()
        assert m_ax.daymatched_leads() == tuple(range(1, 13))
        assert leads == (6, 12, 18, 24, 30, 37, 43, 49, 55, 61, 67, 73), leads
        # within 2.4 d of the monthly leads everywhere — the claim the recipe
        # makes, checked rather than repeated
        dev = max(abs(ax.span_days(h) - m * 365.2425 / 12.0)
                  for m, h in enumerate(leads, 1))
        assert dev < 2.4, dev
        assert res["cadence"]["daymatched_leads"] == list(leads)
        # at HORIZON=3 only lead h=1..3 exist, so none of the twelve pentad
        # leads is reachable and the key is OMITTED rather than written as a
        # mean over nothing (§5.22) — the same rule that keeps NaN out
        assert HORIZON < leads[0]
        assert "horizon_auc_daymatched" not in e["corridor"], e["corridor"]
        assert e["corridor"]["horizon_auc"] is not None
        # and the arithmetic itself, on the block skill_block would be handed
        rows = [{"h": h, "msss_clim": 0.1 * h} for h in range(1, 74)]
        su = rs.new_sums(73)
        for r in rows:                       # one observation per lead
            su["n"][r["h"]] = 1
            su["mse_m"][r["h"]] = 1.0 - r["msss_clim"]
            su["mse_c"][r["h"]] = 1.0
            su["mse_p"][r["h"]] = su["mse_d"][r["h"]] = 1.0
        blk = rs.skill_block(su, 73, n_px=1, leads=leads)
        want = round(float(np.mean([round(1 - (1 - 0.1 * h), 3)
                                    for h in leads])), 3)
        assert blk["horizon_auc_daymatched"] == want, (blk, want)
        assert blk["horizon_auc"] != blk["horizon_auc_daymatched"], \
            "over 73 leads the two means MUST differ — if they did not, the " \
            "lead-sampling problem this key exists for would not exist"
        print("6c. day-matched leads at pentad are %s = %s d, within %.2f d "
              "of the monthly archive's 12 leads everywhere; over a full "
              "73-lead block the raw horizon_auc is %+.3f and the day-matched "
              "one %+.3f (the sampling difference this key exists to remove). "
              "At HORIZON=%d not one of the twelve leads is reachable, so the "
              "key is OMITTED rather than averaged over nothing"
              % (list(leads), [ax.span_days(h) for h in leads], dev,
                 blk["horizon_auc"], blk["horizon_auc_daymatched"], HORIZON))

        # ---- 6d. --starts-per-year: fewer starts, recorded ---------------
        n_all = len(ax.starts_for_year(Y))
        s3 = ax.starts_for_year(Y, 3)
        assert n_all == 73 and len(s3) == 3, (n_all, s3)
        assert s3 == ax.starts_for_year(Y)[::n_all // 3][:3]
        assert s3[0] == ax.starts_for_year(Y)[0], \
            "the first start — whose h=1 is the year's first row — must survive"
        assert ax.starts_for_year(Y, 0) == ax.starts_for_year(Y)
        assert ax.starts_for_year(Y, 999) == ax.starts_for_year(Y), \
            "N >= len(list) must return the full list, untouched"
        assert ax.starts_for_year(Y, 3) == s3, "not deterministic"
        # the phases are spread round the seasonal cycle, not clustered
        moys = sorted(ax.moy_of_row(r) for r in s3)
        assert len(set(moys)) == 3 and max(
            (moys[(i + 1) % 3] - moys[i]) % 12 for i in range(3)) <= 5, moys
        res3, steps3 = run_roll(rs, f, os.path.join(tmp, "roll3.json"),
                                os.path.join(tmp, "cache3"),
                                ("--starts-per-year", "3"))
        n3 = sum(1 for p, _ in steps3 if p == "skill")
        assert 0 < n3 < n_by["skill"], (n3, n_by["skill"])
        st = res3["starts"]
        assert st["per_year"] == 3 and st["available"][Y] == n_all
        assert st["rows"][Y] == s3, (st["rows"][Y], s3)
        assert st["labels"][Y] == [ax.label_of_row(r) for r in s3]
        assert res3["cadence"]["starts_per_year"] == 3
        assert res3["cadence"]["starts_per_holdout_year"][Y] == 3
        assert res3["cadence"]["starts_available_per_holdout_year"][Y] == n_all
        assert res["cadence"]["starts_per_year"] == "all" \
            and "starts" not in res, \
            "an unset knob must record itself as `all` and write no block"
        print("6d. --starts-per-year 3 scored %d roll steps against %d for "
              "all %d starts (%.2fx); the artefact records the N, the rows "
              "%s and their labels %s — every k-th start with k = %d, so the "
              "phases are months %s and the first start survives"
              % (n3, n_by["skill"], n_all, n3 / n_by["skill"], st["rows"][Y],
                 st["labels"][Y], n_all // 3, moys))

        # ---- 7. what the ARCHIVE's evaluator does with the same fixture ---
        base = load(base_copy(tmp), "rs_base")
        # the base evaluator cannot run a pentad roll at all without
        # `--no-gate`: its only reference is monthly, so the ONLY way to get a
        # pentad number out of it was to switch the certificate off. That is
        # the state ml/recipes/xl144-zn-pentad-nolonhold.json records as
        # DO-NOT-DISPATCH, and it is reproduced here rather than described.
        try:
            run_roll(base, f, os.path.join(tmp, "roll_nogate.json"),
                     os.path.join(tmp, "cache0"))
            raise SystemExit("the base evaluator accepted a pentad roll with "
                             "no gate head and no --no-gate — re-read this")
        except SystemExit as exc:
            assert "gate" in str(exc), exc
        res0, steps0 = run_roll(base, f, os.path.join(tmp, "roll0.json"),
                                os.path.join(tmp, "cache0"), ("--no-gate",))
        e0 = res0["heads"]["s1_s0"]
        d0_l, n0_l = phase_rows(ax, steps0, "long", r0_long)
        bad0 = window_defects(steps0, "skill")
        got0 = (e0["long"]["n_trained"] or 0) + (e0["long"]["n_heldout"] or 0)
        ym0 = e0["long"]["roll_ym"]
        keys0 = {m: i for i, m in enumerate(months)}
        # the base's own truth index, rebuilt: YYYYMM -> position in ridx,
        # LAST WINS. It attaches a value for nearly every step — from the
        # wrong ROW, which is worse than attaching none.
        ym_to_r0 = {int(months[mi][:4]) * 100 + int(months[mi][5:7]): i
                    for i, mi in enumerate(f["ridx"])}
        right0 = sum(1 for i, v in enumerate(ym0)
                     if f["ridx"][ym_to_r0[int(v[:4]) * 100 + int(v[5:7])]]
                     == r0_long + 1 + i
                     if int(v[:4]) * 100 + int(v[5:7]) in ym_to_r0)
        assert d0_l > 1 and bad0 > 0 and right0 < want, (d0_l, bad0, right0)
        print("7. THE PRE-FIX EVALUATOR (%s) ON THE SAME FIXTURE — it "
              "refuses to run at all unless --no-gate switches the "
              "certificate off, and then every number it writes is wrong:"
              % BASE_SHA)
        print("     season token max deviation %.3f months over %d long-roll "
              "tokens (fixed: %.1e)" % (d0_l, n0_l, dev_l))
        print("     %d of %d scored windows hold a month span 6 pentads "
              "cannot cover (fixed: %d)" % (bad0, len(
                  [1 for p, _ in steps0 if p == 'skill']), bad_skill))
        print("     its YYYYMM truth index holds %d keys for %d axis rows "
              "(%.1f%% of the series reachable); it attached %d values to the "
              "%d-step hindcast and only %d of them are the truth of the row "
              "actually rolled (fixed: %d attached, all %d correct)"
              % (len(ym_to_r0), ax.T, 100.0 * len(ym_to_r0) / ax.T, got0,
                 N_LONG, right0, got, got))
        print("     labels advance a MONTH per pentad: %s … %s over %d steps "
              "(fixed: %s … %s)" % (ym0[0], ym0[-1], N_LONG, ym[0], ym[-1]))
        print("     %d scored roll steps against %d (the axis has %d rows in "
              "%s, and range(12) can only start on %d of them)"
              % (len([1 for p, _ in steps0 if p == "skill"]),
                 n_by["skill"], n_rows_Y, Y,
                 len([1 for o in range(12)
                      if (f"{int(Y) - 1}-12" if o == 0
                          else f"{Y}-{o:02d}") in keys0])))
        print("     bands labelled %s — read as months by every archived "
              "comparison (fixed: %s)"
              % (", ".join(sorted(e0["amoc_bands"])),
                 ", ".join(sorted(e["amoc_bands"]))))
        print("     gate: %s (fixed: skipped, certified False, with a reason)"
              % json.dumps(res0["gate"]))

        print("\npentad cadence roll: all 10 checks hold ✓")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    main()

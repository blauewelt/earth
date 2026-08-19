#!/usr/bin/env python3
"""probe_kfold's k-fold must be PAIRED-TESTABLE, and its summary must not move.

ml/CLAUDE.md §3: *"Comparing two probes needs a PAIRED test
(`scripts/paired_probe.py`), not two overlapping intervals — they share folds,
months and most of their error."* That script needs `pred`, `target_sv` and
`years` from both files it is given.

`ml/probe_head.py` has written those since 2026-08-10. `ml/probe_kfold.py`
computed `pred` inside `kfold_r`, unpacked it at the call site, used it for the
18-month lowpass — and then wrote a summary block that threw it away. So no
pooled-ridge k-fold in this programme's history has ever been paired-testable:
not for lack of archiving, the numbers were never written. Every pooled
comparison ever made was two overlapping CIs, which §3 says is not a result.
Found 2026-08-19, trying to compare #416's codec against the frozen anchor.

Pinned here, on a 240-month fixture two real codecs are actually probed on:

  1. every scored target carries the three arrays, and their length is the `n`
     the same block reports — a `pred` shorter than its own `n` would pair the
     wrong months against each other and say nothing about it;
  2. the SUMMARY IS BIT-IDENTICAL to the pre-change golden — `r`, `ci95`,
     `rmse_sv`, `sigma_sv`, `r_lowpass18` and the whole wind-only block, for
     three targets and two codecs. This change is a pure addition, and five
     runs were live on the fleet when it was made;
  3. `scripts/paired_probe.py` consumes two of these files and returns a
     paired statistic — the thing that was impossible before;
  4. it still consumes `ml/probe_head.py`'s FLAT shape, and the existing
     `tests/test_paired_probe.py` still passes unmodified;
  5. the wind-only baseline is paired too — it keeps its own `pred` and
     inherits the target block's rows, so "does the embedding beat the wind?"
     is answerable with the test §3 asks for rather than by eye;
  6. a legacy file (no arrays) is REFUSED with the rerun message, not
     compared on its summaries;
  7. the file stays small — the per-month cost is measured here, because
     these blocks are bundled into `probes-<n>.json` and that bundle is
     fetched whole by the status page.

    python3 tests/test_probe_kfold_pairing.py
"""
import contextlib
import io
import json
import os
import subprocess
import sys
import tempfile

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ML = os.path.join(HERE, "..", "ml")
SCRIPTS = os.path.join(HERE, "..", "scripts")
sys.path.insert(0, ML)

import probe_kfold as pk                                    # noqa: E402
from model import PixelMAE                                  # noqa: E402

PAIRED = os.path.join(SCRIPTS, "paired_probe.py")
TARGETS = ("rapid", "fc", "move")          # osnap/samba are outside the window
ARRAYS = ("pred", "target_sv", "years")

# The summary this file wrote BEFORE the arrays were added, on this fixture,
# read off ml/probe_kfold.py at 88580b1. Everything the programme quotes is in
# here; if any of it moves, the change stopped being an addition.
GOLDEN = {
    "codecA": {
        "fc": {"r_kfold_deseas": 0.623, "ci95": [0.537, 0.677], "n": 240,
               "rmse_sv": 1.1, "sigma_sv": 1.4, "r_lowpass18": 0.94,
               "wind_only_baseline": {"r": 0.592, "ci95": [0.478, 0.669],
                                      "rmse_sv": 1.14}},
        "move": {"r_kfold_deseas": 0.414, "ci95": [0.314, 0.503], "n": 240,
                 "rmse_sv": 3.11, "sigma_sv": 3.41, "r_lowpass18": 0.81,
                 "wind_only_baseline": {"r": 0.447, "ci95": [0.335, 0.535],
                                        "rmse_sv": 3.05}},
        "rapid": {"r_kfold_deseas": 0.843, "ci95": [0.794, 0.877], "n": 240,
                  "rmse_sv": 1.86, "sigma_sv": 3.45, "r_lowpass18": 0.987,
                  "wind_only_baseline": {"r": 0.8, "ci95": [0.736, 0.843],
                                         "rmse_sv": 2.07}},
    },
    "codecB": {
        "fc": {"r_kfold_deseas": 0.58, "ci95": [0.437, 0.66], "n": 240,
               "rmse_sv": 1.14, "sigma_sv": 1.4, "r_lowpass18": 0.959,
               "wind_only_baseline": {"r": 0.592, "ci95": [0.478, 0.669],
                                      "rmse_sv": 1.14}},
        "move": {"r_kfold_deseas": 0.339, "ci95": [0.212, 0.451], "n": 240,
                 "rmse_sv": 3.22, "sigma_sv": 3.41, "r_lowpass18": 0.8,
                 "wind_only_baseline": {"r": 0.447, "ci95": [0.335, 0.535],
                                        "rmse_sv": 3.05}},
        "rapid": {"r_kfold_deseas": 0.77, "ci95": [0.687, 0.827], "n": 240,
                  "rmse_sv": 2.2, "sigma_sv": 3.45, "r_lowpass18": 0.969,
                  "wind_only_baseline": {"r": 0.8, "ci95": [0.736, 0.843],
                                         "rmse_sv": 2.07}},
    },
}


# ---------------------------------------------------------------------------
# The fixture: a tensor ml/probe_kfold.py runs end to end, in ~6 s on CPU.
#
# Its lon spacing is 0.25° on purpose. The Florida Current section is
# (-80.5, -78.5) — 7 cells at quarter degree, and FEWER THAN FIVE at any
# coarser spacing, which probe_kfold skips with a note. A 1° fixture would
# have quietly tested one target while claiming to test three.
def build(tmp, T=240, seed=20260819):
    rng = np.random.default_rng(seed)
    lats = np.arange(15.0, 30.0, 0.5)          # carries 26.5 (rapid, fc) and
    lons = np.arange(-81.0, -45.0, 0.25)       # 16.5 (move)
    H, W = len(lats), len(lons)
    chan = ["tau_x", "tau_y", "sst", "ssh"]    # tau_* -> the wind-only bar runs
    C = len(chan)
    # an AR(1) latent the fields and all three transports share, so the probe
    # has something real to find and the two codecs disagree about how much
    latent = np.zeros(T)
    for t in range(1, T):
        latent[t] = 0.85 * latent[t - 1] + rng.standard_normal()
    latent /= latent.std()
    X = rng.normal(size=(T, H, W, C)).astype(np.float32)
    X += (0.8 * latent)[:, None, None, None]
    X += 0.5 * np.sin(2 * np.pi * np.arange(T) / 12)[:, None, None, None]
    X = X.astype(np.float16)
    X[:, rng.random((H, W)) < 0.12, :] = np.nan                    # land
    months = np.array(["%04d-%02d" % (2004 + i // 12, i % 12 + 1)
                       for i in range(T)])
    ym = np.array([int(m[:4]) * 100 + int(m[5:7]) for m in months], float)

    def series(scale, off, noise, s):
        r2 = np.random.default_rng(s)
        return (off + scale * latent + noise * r2.standard_normal(T)
                + 1.5 * np.sin(2 * np.pi * np.arange(T) / 12))

    data = os.path.join(tmp, "kf.npz")
    np.savez(data, X=X, months=months, lats=lats, lons=lons,
             chan=np.array(chan),
             # rapid is (MONTH INDEX, value); truth_* are (YYYYMM, value) —
             # the two shapes probe_kfold.target_series() reads for family 2/3
             rapid=np.stack([np.arange(T, dtype=float), series(3, 17, 2, 1)], 1),
             truth_fc=np.stack([ym, series(1, 32, 1, 2)], 1),
             truth_move=np.stack([ym, series(2, -18, 3, 3)], 1))
    del X
    for run, s in (("codecA", 7), ("codecB", 11)):
        torch.manual_seed(s)
        m = PixelMAE(n_chan=C, d_model=16, n_heads=2, n_layers=1, d_z=8,
                     d_dec=16, patch=1, dec_layers=1)
        m.eval()
        d = os.path.join(tmp, "runs", run)
        os.makedirs(d, exist_ok=True)
        torch.save({"args": {"holdout_years": "2006", "holdout_lon": "-60,-55",
                             "patch": 1, "d_model": 16, "n_layers": 1,
                             "n_heads": 2, "d_dec": 16, "dec_layers": 1},
                    "chan": chan, "d_z": 8, "model": m.state_dict()},
                   os.path.join(d, "pixelmae.pt"))
    return data


def run_kfold(tmp, data, runs):
    """One real ml/probe_kfold.py main(), writing under `tmp` instead of ml/."""
    old = (pk.HERE, sys.argv)
    pk.HERE = tmp
    sys.argv = ["probe_kfold.py", "--runs", *runs, "--data", data]
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            pk.main()
    finally:
        pk.HERE, sys.argv = old
    return json.load(open(os.path.join(tmp, "runs", "probe_kfold.json")))


def paired(*specs):
    r = subprocess.run([sys.executable, PAIRED, *specs, "--n-boot", "2000"],
                       capture_output=True, text=True)
    return r.returncode, r.stdout + r.stderr


def strip(block):
    """The block as it was written before the arrays were added."""
    out = {k: v for k, v in block.items() if k not in ARRAYS and k != "probe"}
    w = out.get("wind_only_baseline")
    if w:
        out["wind_only_baseline"] = {k: v for k, v in w.items()
                                     if k not in ARRAYS and k != "probe"}
    return out


def main():
    tmp = tempfile.mkdtemp(prefix="kfpair_")
    data = build(tmp)
    out = run_kfold(tmp, data, ["codecA", "codecB"])
    assert sorted(out) == ["codecA", "codecB"], sorted(out)

    # -- 1 -----------------------------------------------------------------
    for run in ("codecA", "codecB"):
        assert sorted(out[run]) == sorted(TARGETS), \
            f"{run} scored {sorted(out[run])}, expected {sorted(TARGETS)}"
        for t in TARGETS:
            b = out[run][t]
            for k in ARRAYS:
                assert k in b, f"{run}/{t} has no {k} — the paired test needs it"
                assert len(b[k]) == b["n"], \
                    (f"{run}/{t}: {k} has {len(b[k])} rows against n={b['n']} "
                     f"— a paired test would line up the wrong months")
            assert sum(v is not None for v in b["pred"]) == b["n"], \
                f"{run}/{t}: n counts finite predictions; the dump disagrees"
            assert b["probe"] == "pooled-ridge", b["probe"]
    print("1 · arrays present for every scored target "
          f"({', '.join(TARGETS)} x codecA/codecB), length == n ✓")

    # -- 2 -----------------------------------------------------------------
    for run in sorted(GOLDEN):
        for t in sorted(GOLDEN[run]):
            got, want = strip(out[run][t]), GOLDEN[run][t]
            assert got == want, (f"{run}/{t} MOVED\n  before: {want}\n"
                                 f"  after:  {got}")
    n_num = sum(len(strip(out[r][t])) + 3 for r in GOLDEN for t in GOLDEN[r])
    print(f"2 · summary bit-identical to the pre-change golden "
          f"({n_num} fields over 2 codecs x 3 targets) ✓")
    print(f"      rapid: codecA r {out['codecA']['rapid']['r_kfold_deseas']} "
          f"{out['codecA']['rapid']['ci95']} · "
          f"codecB r {out['codecB']['rapid']['r_kfold_deseas']} "
          f"{out['codecB']['rapid']['ci95']}")

    # -- 3 -----------------------------------------------------------------
    shared = os.path.join(tmp, "runs", "probe_kfold.json")
    per_run = [os.path.join(tmp, "runs", r, "probe_kfold.json")
               for r in ("codecA", "codecB")]
    rc, txt = paired(*per_run)
    assert rc == 0, txt
    assert "paired block bootstrap over 20 years" in txt, txt
    assert "Δr 95% CI" in txt, txt
    # codecA reads 0.843 and codecB 0.770 on the SAME 240 months and the same
    # 20 year-blocks; on this fixture that gap survives the pairing. The
    # assertion is on the verdict, not on the sign of a number I chose.
    assert "excludes zero" in txt, txt
    print("3 · paired_probe.py compares two probe_kfold outputs:")
    for line in txt.strip().splitlines():
        print("      " + line)

    # the same two codecs out of the MERGED file, which holds both runs and so
    # must be told which one — this is the shape a sweep actually writes
    rc, txt2 = paired(shared + "#codecA/rapid", shared + "#codecB/rapid")
    assert rc == 0, txt2
    a = [l for l in txt2.splitlines() if "Δr 95% CI" in l][0]
    b = [l for l in txt.splitlines() if "Δr 95% CI" in l][0]
    assert a == b, f"the merged file disagrees with the per-run files\n{a}\n{b}"
    rc, txt3 = paired(shared, shared)
    assert rc != 0 and "holds 2 runs" in txt3, txt3
    print("      merged file, #codecA/rapid vs #codecB/rapid: same interval; "
          "ambiguous without a fragment ✓")

    # a target OTHER than the headline one, and a target that is not there
    rc, txt4 = paired(shared + "#codecA/fc", shared + "#codecB/fc")
    assert rc == 0 and "point difference" in txt4, txt4
    rc, txt5 = paired(shared + "#codecA/osnap", shared + "#codecB/osnap")
    assert rc != 0 and "nothing at 'codecA/osnap'" in txt5, txt5
    print("      #codecA/fc vs #codecB/fc pairs too; an unscored target is "
          "refused by name ✓")

    # -- 4 -----------------------------------------------------------------
    # probe_head.py's shape, verbatim from ml/probe_head.py's writer: the three
    # arrays FLAT at the top level, beside the summary. Its files must keep
    # working with no fragment and no --target.
    head = []
    yrs = out["codecA"]["rapid"]["years"]
    tgt = out["codecA"]["rapid"]["target_sv"]
    for run in ("codecA", "codecB"):
        p = os.path.join(tmp, "runs", run, "probe_head.json")
        json.dump({"run": run, "head_dim": 64, "head_blocks": 0,
                   "probe": "attention-head", "K": 1,
                   "r_kfold_deseas": out[run]["rapid"]["r_kfold_deseas"],
                   "ci95": out[run]["rapid"]["ci95"],
                   "rmse_sv": out[run]["rapid"]["rmse_sv"], "n": 240,
                   "seed_base": 0,
                   "pred": out[run]["rapid"]["pred"],
                   "target_sv": tgt, "years": yrs,
                   "note": "unpooled section"}, open(p, "w"))
        head.append(p)
    rc, txt6 = paired(*head)
    assert rc == 0 and "attention-head" in txt6, txt6
    assert [l for l in txt6.splitlines() if "Δr 95% CI" in l][0] == b, \
        "the flat probe_head shape and the nested probe_kfold shape must " \
        "reach the same paired interval from the same numbers"
    r = subprocess.run([sys.executable, os.path.join(HERE, "test_paired_probe.py")],
                       capture_output=True, text=True)
    assert r.returncode == 0, r.stdout + r.stderr
    print("4 · probe_head.py's flat shape still consumed (same interval), and "
          "tests/test_paired_probe.py passes unchanged ✓")

    # -- 5 -----------------------------------------------------------------
    rc, txt7 = paired(shared + "#codecA/rapid",
                      shared + "#codecA/rapid/wind_only_baseline")
    assert rc == 0, txt7
    assert "wind-only" in txt7 and "pooled-ridge" in txt7, txt7
    assert "excludes zero" in txt7, txt7   # 0.843 vs the 0.800 wind bar
    print("5 · the wind-only baseline pairs against its own target block:")
    for line in txt7.strip().splitlines():
        print("      " + line)

    # -- 6 -----------------------------------------------------------------
    old = os.path.join(tmp, "legacy_probe_kfold.json")
    json.dump({"actions": {"rapid": strip(out["codecA"]["rapid"])}},
              open(old, "w"))
    rc, txt8 = paired(old, old + "#actions/rapid")
    assert rc != 0 and "predates" in txt8, txt8
    print("6 · a pre-2026-08-19 probe_kfold.json is refused, not compared on "
          "its summaries ✓")
    print("      " + txt8.strip())

    # -- 7 -----------------------------------------------------------------
    per_run_bytes = os.path.getsize(per_run[0])
    rows = sum(out["codecA"][t]["n"] for t in TARGETS)
    # 4 arrays per target: pred, target_sv, years, and the wind bar's pred
    cost = per_run_bytes / rows
    assert cost < 40, (f"{cost:.1f} bytes per scored month — the arrays are "
                       f"bundled into probes-<n>.json and that bundle is "
                       f"fetched whole by the status page")
    print(f"7 · file size: {per_run_bytes / 1024:.1f} KB for one codec x 3 "
          f"targets x 240 months = {cost:.1f} bytes/month")
    for label, n_rapid, n_fc in (("monthly", 240, 240),
                                 ("pentad", 1459, 2490),
                                 ("daily", 7300, 12450)):
        est = cost * (n_rapid * 2 + n_fc)
        print(f"      {label:<8} rapid n={n_rapid:<6} fc n={n_fc:<6} "
              f"-> ~{est / 1024:.0f} KB per codec")

    # -- 8 -----------------------------------------------------------------
    # A fold that produced no prediction must write JSON `null`, not the bare
    # `NaN` token json.dump emits by default: these blocks are bundled into
    # probes-<n>.json, and JSON.parse — which status.html, run_index.mjs and
    # sweep_table.mjs all use — rejects NaN outright. numpy reads null back as
    # NaN, so paired_probe's isfinite masking still does the right thing.
    holed = os.path.join(tmp, "holed.json")
    obj = {"run": {"rapid": {"n": 3, "pred": pk._arr([1.0, np.nan, -2.5]),
                             "target_sv": pk._arr([1.0, 2.0, 3.0]),
                             "years": [2004, 2004, 2005]}}}
    pk._dump(obj, holed)
    txt9 = open(holed).read()
    assert "NaN" not in txt9, txt9
    assert json.loads(txt9) == obj, txt9
    assert '"pred": [1.0, null, -2.5]' in txt9, txt9   # and still compacted
    assert np.isnan(np.asarray(json.loads(txt9)["run"]["rapid"]["pred"],
                               float)[1])
    print("8 · a missing prediction writes as null and round-trips to NaN; "
          "no bare NaN token can reach a JSON.parse consumer ✓")

    print("\nOK — the pooled ridge is paired-testable, and nothing it "
          "already reported has moved.")


if __name__ == "__main__":
    main()

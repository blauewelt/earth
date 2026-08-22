#!/usr/bin/env python3
"""The three quantities E-044 §7b re-expressed in DAYS, pinned per cadence.

`ml/rollout_spatial.py`'s `TimeAxis.bands()` docstring says, of the monthly
partition it computes, "tests/test_roll_bands_daydefined.py asserts that
identity rather than trusting it". This is that file. It was cited before it
existed (`ef62fbf`), which is the same class of claim as a step that reports
success without doing anything (ml/CLAUDE.md §0.2) — the assertion lived in a
comment, where nothing runs it.

What is pinned, and why each one is load-bearing:

  * **The monthly bands are the LITERAL they replaced.** `BANDS` used to be
    `(("h1-3", (1,2,3)), ("h4-6", (4,5,6)), ("h7-12", (7..12)))`; it is now
    derived from `BAND_EDGE_DAYS` through the axis's own `step_days`. Every
    archived `amoc_bands` key and three of the four `GATE_REF` criteria are
    those three names, so a derivation that moved a single step — or renamed a
    single key — would `sys.exit("VALIDATION GATE FAILED")` every eval wave
    (ml/CLAUDE.md §3, exception 1). The identity is exact rather than lucky:
    365.2425/12 divides the three edges with no residue in IEEE double.
  * **Pentad and daily follow from the SAME edges**, h1-18 / h19-36 / h37-73
    and h1-91 / h92-182 / h183-365, and their keys carry their day spans, so a
    5-90 d correlation can never be read under the key a 30-91 d one owns.
  * **The day-matched horizon**, `steps_for_months(12)`: 12 at monthly (a
    no-op, by construction) and 73 = 365.0 d at pentad. This is the number
    `scripts/sroll_run.sh` computes off the tensor and passes as `--horizon`
    when no `horizon:` token names one; `tests/test_sroll_wiring.py` pins the
    passing, this pins the value.
  * **`--starts-per-year` at monthly is IDENTITY at S = 12** — the list AND
    its order — which is the hard constraint that lets the knob exist at all
    (E-044 §7b(h)(2)). At pentad S = 3 it is a stride of 24 pentads, ~120 d
    apart, which is what keeps lead time from being confounded with season.

No GPU, no tensor, no roll: the four quantities are properties of the axis, so
the axis is all this builds.

    python3 tests/test_roll_bands_daydefined.py
"""
import datetime as dt
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
ML = os.path.join(ROOT, "ml")
sys.path.insert(0, ML)

import rollout_spatial as rs                                    # noqa: E402
from aggregate_cadence import EPOCH, bin_start                   # noqa: E402

# The literal `BANDS` constant as it stood at 9066341, transcribed here so the
# comparison is against the archive's own text and not against a re-derivation
# of it.
LEGACY_BANDS = (("h1-3", (1, 2, 3)),
                ("h4-6", (4, 5, 6)),
                ("h7-12", (7, 8, 9, 10, 11, 12)))


def monthly_axis(y0=1982, n=516):
    """A families-2/3 axis: `months` labels, no `bin_index`."""
    labels = [f"{y0 + i // 12:04d}-{i % 12 + 1:02d}" for i in range(n)]
    return rs.TimeAxis({"months": np.array(labels)})


def binned_axis(days, y0=1982, n_years=6):
    """A families-4/5 axis, built the way build_family4.py builds one: one
    consecutive `bin_index` per bin from the shared EPOCH, and `months` as the
    LABEL of the bin's own start date — which is exactly the pairing
    `TimeAxis._init_binned` re-derives and refuses to trust."""
    b0 = (dt.date(y0, 1, 1) - EPOCH).days // days
    n = int(round(n_years * 365.2425 / days))
    bins = np.arange(b0, b0 + n, dtype=np.int64)
    labels = np.array([f"{bin_start(int(b), days).year:04d}-"
                       f"{bin_start(int(b), days).month:02d}" for b in bins])
    return rs.TimeAxis({"months": labels, "bin_index": bins,
                        "pentad_days": np.int64(days)})


def main():
    # ---- 1. the monthly bands ARE the literal, name for name, step for step
    m = monthly_axis()
    assert m.cadence == "monthly" and m.step_days == 365.2425 / 12.0
    assert m.bands() == LEGACY_BANDS, m.bands()
    # the KEYS are what the archive holds, and band_key must not decorate them
    assert [m.band_key(bn, hs) for bn, hs in m.bands()] == \
        [bn for bn, _ in LEGACY_BANDS]
    assert set(rs.GATE_REF["bands"]) == {bn for bn, _ in LEGACY_BANDS}
    # DERIVED, not coincidental: each edge divides the monthly step exactly.
    ratios = [e / m.step_days for e in rs.BAND_EDGE_DAYS]
    assert ratios == [3.0, 6.0, 12.0], ratios
    print("1. monthly bands are the literal they replaced: %s — and the three "
          "edge/step_days ratios are exactly %s in IEEE double, which is why"
          % (" / ".join(f"{bn}{hs}" for bn, hs in m.bands()), ratios))

    # ---- 2. the edges are the tropical year quartered, halved, whole -------
    y = rs.YEAR_DAYS
    assert rs.BAND_EDGE_DAYS == (y / 4.0, y / 2.0, y)
    for ax, name in ((m, "monthly"), (binned_axis(5), "pentad"),
                     (binned_axis(1), "daily")):
        cov = [h for _, hs in ax.bands() for h in hs]
        assert cov == list(range(1, len(cov) + 1)), (name, cov[:5], cov[-5:])
        assert len(set(cov)) == len(cov), name          # exactly one band each
        assert ax.span_days(max(cov)) <= y + ax.step_days, name
        assert ax.span_days(max(cov) + 1) > y, name     # and it reaches a year
    print("2. every step from 1 to a full year falls in exactly one band at "
          "all three cadences, cut at the same edges %s d"
          % "/".join(f"{e:g}" for e in rs.BAND_EDGE_DAYS))

    # ---- 3. pentad and daily follow from those same edges ------------------
    p = binned_axis(5)
    assert p.cadence == "pentad" and p.step_days == 5.0
    got = [(bn, min(hs), max(hs)) for bn, hs in p.bands()]
    assert got == [("h1-18", 1, 18), ("h19-36", 19, 36), ("h37-73", 37, 73)], \
        got
    keys_p = [p.band_key(bn, hs) for bn, hs in p.bands()]
    assert keys_p == ["h1-18_5-90d", "h19-36_95-180d", "h37-73_185-365d"], \
        keys_p
    d = binned_axis(1, n_years=3)
    assert [(bn, min(hs), max(hs)) for bn, hs in d.bands()] == \
        [("h1-91", 1, 91), ("h92-182", 92, 182), ("h183-365", 183, 365)]
    assert d.band_key("h1-91", (1, 91)) == "h1-91_1-91d"
    print("3. the same edges give pentad %s and daily h1-91 / h92-182 / "
          "h183-365 — every non-monthly key carries its own day span, so it "
          "cannot be read as the monthly band of the same name"
          % " / ".join(keys_p))

    # ---- 4. the day-matched horizon ---------------------------------------
    assert m.steps_for_months(12) == 12                 # no-op at monthly
    assert p.steps_for_months(12) == 73 and p.span_days(73) == 365.0
    assert d.steps_for_months(12) == 365
    # and the thing that made 73 necessary: the argparse default is 60 days
    # here, which is the number scripts/sroll_run.sh exists to replace.
    assert p.span_days(12) == 60.0
    assert m.daymatched_leads() == tuple(range(1, 13))
    assert p.daymatched_leads() == (6, 12, 18, 24, 30, 37, 43, 49, 55, 61,
                                    67, 73)
    off = max(abs(p.span_days(h) - m.span_days(mo))
              for mo, h in enumerate(p.daymatched_leads(), 1))
    # span_days() already rounds to 0.1 d, so the comparison is on that grid —
    # the raw max is 2.4000000000000057, i.e. the 2.4 d the docstring claims
    # plus one representation ulp.
    assert round(off, 6) <= 2.4, off
    print("4. the day-matched horizon is %d steps (%g d) at pentad against "
          "monthly's 12 (%g d) — the argparse default of 12 is %g d here — "
          "and the twelve day-matched leads %s stand within %.2f d of the "
          "monthly ones everywhere"
          % (p.steps_for_months(12), p.span_days(73), m.span_days(12),
             p.span_days(12), list(p.daymatched_leads()), off))

    # ---- 5. --starts-per-year: IDENTITY at monthly, stride at pentad -------
    Y = 1990
    all_m = m.starts_for_year(Y)
    assert len(all_m) == 12
    assert [m.label_of_row(r) for r in all_m] == \
        [f"{Y - 1}-12"] + [f"{Y}-{k:02d}" for k in range(1, 12)]
    # THE HARD CONSTRAINT (E-044 §7b(h)(2)): at monthly, S = 12 must return
    # the existing list in the existing order. Same list object contents AND
    # order — `==` on a list is both.
    assert m.starts_for_year(Y, 12) == all_m, m.starts_for_year(Y, 12)
    for s in (0, 1, 2, 3, 4, 6, 12, 13, 999):
        sub = m.starts_for_year(Y, s)
        assert sub == sorted(sub) and set(sub) <= set(all_m), (s, sub)
        assert sub[0] == all_m[0], (s, sub)             # the h=1 start survives
    assert m.starts_for_year(Y, 0) == all_m and m.starts_for_year(Y, 99) \
        == all_m
    print("5. at monthly the year's starts are %s and S=12 returns that list "
          "unchanged, in order — so the archived protocol is what an S the "
          "size of the list buys" % [m.label_of_row(r) for r in all_m])

    # ---- 6. pentad S = 3: stride 24, three seasonal phases ----------------
    # The pentad axis above starts at 1982 to share the tensor's own epoch
    # arithmetic; Y must be inside whatever span it covers, so build one that
    # holds it rather than moving Y (the monthly checks above are keyed on Y).
    p = binned_axis(5, y0=Y - 2, n_years=6)
    all_p = p.starts_for_year(Y)
    assert len(all_p) == 73, len(all_p)
    s3 = p.starts_for_year(Y, 3)
    assert len(s3) == 3 and s3 == all_p[::73 // 3][:3] == all_p[::24][:3]
    assert [b - a for a, b in zip(s3, s3[1:])] == [24, 24]
    assert s3[0] == all_p[0]
    assert p.starts_for_year(Y, 3) == s3                # deterministic
    assert p.starts_for_year(Y, 0) == all_p and p.starts_for_year(Y, 999) \
        == all_p
    dates = [p.date_of_row(r) for r in s3]
    assert [dd.month for dd in dates] == [12, 4, 8], dates
    # ~120 d apart: three phases of the seasonal cycle, not one repeated
    gaps = [(b - a).days for a, b in zip(dates, dates[1:])]
    assert gaps == [120, 120], gaps
    # and the cost the knob buys, which is why it exists at all
    assert len(all_p) / len(s3) > 24
    print("6. at pentad the year has %d starts and S=3 takes every %dth — "
          "rows %s = %s, %d d apart, so the three rolls begin at three "
          "different points in the seasonal cycle"
          % (len(all_p), 73 // 3, s3, [dd.isoformat() for dd in dates],
             gaps[0]))

    print("\nday-defined bands, day-matched horizon and strided starts: "
          "all 6 checks hold ✓")


if __name__ == "__main__":
    main()

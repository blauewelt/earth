#!/usr/bin/env python3
"""E-022 evaluator: FULL-WINDOW autoregressive rollout for stencil heads.

Why this exists (plan ml/plans/E022_spatial_coupling.md §6): a stencil head's
step-t+1 at pixel p reads p's neighbours at t, so a correct T-step roll of any
region needs that region plus a T×reach halo. The clean solution is to roll
ALL window ocean pixels (84,405 at family3) — which is also EXACT for the
stencil-1 baseline, since the per-pixel model factorises over pixels. Chris's
spec: "the eval needs to involve rolling forward all pixels that contribute
to the AMOC current (from the Gulf of Mexico to northern Europe, and back)".

GPU-native from day one: ml/rollout.py moved the codec back to CPU for its
head loop and burned ~$1.9 of rented 4090 across #211/#217 doing transformer
forwards on cores. Here the head, the codec, the rolling window state
([P, K, d_z] ≈ 519 MB) and the stencil gather all live on the device.

Protocol = rollout.py's, verbatim where they overlap (staggered starts into
holdout years, true-context init, MSSS vs climatology / persistence / damped
persistence on observed cells, truefit AMOC ridge) — because the VALIDATION
GATE below demands the two agree before any spatial head is scored. Scoring
runs over three nested pixel scopes per horizon:

  gate     — rollout.py's exact subset (default_rng(0).choice(P, 600) ∪ the
             RAPID section), where #217's numbers live. The gate compares
             THIS scope against #217; a mismatch means the evaluator is
             wrong and the script refuses to score anything past it.
  corridor — the AMOC corridor, data-derived, not hand-drawn: ocean pixels
             whose train-month mean cur_speed (channel 0) is ≥ the 75th
             percentile over window ocean, dilated 2 cells, ∪ the RAPID
             section. The headline scope (§3.6).
  window   — all window ocean pixels.

Plus, per head: AMOC truefit r in horizon bands h1-3/h4-6/h7-12; a long
hindcast (context ends --long-start, default 2004-12, rolled 20 years across
the RAPID record, median trajectory only — E-021b owns ensembles); and a
future roll from the record's end.

CADENCE. Every horizon, start, band and roll step below is an AXIS STEP of
the tensor being rolled, and the axis is DERIVED FROM THAT TENSOR (`TimeAxis`
below) rather than assumed monthly or passed as a flag. On families 2/3 a
step is a calendar month and this file behaves exactly as it always has,
byte for byte (tests/test_roll_monthly_identity.py). On families 4/5 a step
is a pentad or a day: the staggered starts span the axis's real
steps-per-year, the RAPID truth attaches on the axis row, the season token
comes from each bin's true date, and the band labels carry their day spans.

WHAT A STEP IS AND WHAT A DAY IS (2026-08-20, E-044 horizon decision). Making
the axis cadence-aware left three quantities still counted in STEPS whose
meaning is a DURATION, and at --horizon 73 each of them was wrong in a way
that reads as a result:

  * `BANDS` was a module constant over h1..12, so the three AMOC bands covered
    the first 60 of 365 days at pentad and 61 leads fell into no band at all.
    Bands are now cut at fixed DAY EDGES (`BAND_EDGE_DAYS`, quarter/half/whole
    tropical year) and `TimeAxis.bands()` converts them to this axis's steps —
    which returns h1-3 / h4-6 / h7-12 EXACTLY at monthly (the ratios are
    exactly 3.0, 6.0 and 12.0 in IEEE double; pinned by
    tests/test_roll_bands_daydefined.py) and h1-18 / h19-36 / h37-73 at pentad.
  * `horizon_auc` is the UNWEIGHTED mean of msss_clim over h = 1..H. At
    monthly that averages 12 leads spanning 30-365 d; at pentad H=73 it
    averages 73 leads spanning 5-365 d, most of them short, where skill is
    highest — so a raw pentad `horizon_auc` beats the monthly archive on lead
    SAMPLING alone. `horizon_auc_daymatched` is the mean over the twelve leads
    that stand closest to the monthly archive's (`TimeAxis.daymatched_leads()`:
    1..12 at monthly, {6,12,18,24,30,37,43,49,55,61,67,73} at pentad, within
    2.4 d of the monthly leads everywhere). It is emitted at EVERY cadence and
    ALONGSIDE `horizon_auc`, never instead — and at monthly, where the leads
    ARE h=1..H, it is the same number by construction.
  * the number of staggered starts is a COST, not a protocol constant: 12 a
    year at monthly, 73 at pentad. `--starts-per-year N` takes every k-th
    start of the year's list, k = len(list)//N, first N — deterministic, and
    absent (0) it is today's behaviour exactly, so the monthly list and its
    ORDER are untouched.

The horizon itself is NOT rescaled here. `--horizon` is a step count and stays
one (ml/CLAUDE.md §5.24: this file may say what a flag buys, not quietly
change it); scripts/sroll_run.sh computes the day-matched value from the
tensor and passes it explicitly, and this file warns when a non-monthly roll
is asked for a horizon that is not the archive's 12 months.

Usage (box, GPU):
  python3 ml/rollout_spatial.py --x ml/cache/family3_X.npy \
      --npz-small ml/cache/f3_dec_small.npz --z ml/cache/Z_....npy \
      --ckpt ml/cache/f3_anchor41M__pixelmae.pt \
      --heads ml/runs/heads/e017_u1_s0.pt ml/runs/heads/e022s9_u1_s0.pt ... \
      --out ml/runs/actions/rollout_spatial.json
"""
import argparse
import datetime as dt
import json
import math
import os
import sys
import time

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from model import codec_from_ckpt, gather_px                    # noqa: E402
from recon_eval import stream_stats, build_slab                 # noqa: E402
from temporal import (TemporalTransformer, build_stencil,       # noqa: E402
                      embed_everything, rapid_section, _ring_on,
                      codec_weight_hash)
from project_amoc import fit_ridge                              # noqa: E402
from aggregate_cadence import (EPOCH as EPOCH_DEFAULT,          # noqa: E402
                               bin_start)

# The gate reference: #217 (ml-metrics probes-217.json), head u1_s0 —
# e017_u1_s0 rolled by ml/rollout.py over its 600∪section subset. NOTE the
# metric's name: the 0.643–0.645 "AUC" band quoted everywhere for E-017 is
# rollout.py's `horizon_auc` = mean MSSS vs CLIMATOLOGY over h=1..12 (the
# plan's §1 says "AUC(msss_damped)" — checked against the archive 2026-08-13,
# it is msss_clim; the damped mean is 0.619). Gate on what #217 actually is.
GATE_HEAD = "e017_u1_s0"
GATE_REF = {"auc": 0.643,
            "bands": {"h1-3": 0.470, "h4-6": 0.375, "h7-12": 0.492}}
GATE_TOL = 0.0101          # plan §6.5: ±0.01 (float-boundary slack only)

# HORIZON BANDS ARE DEFINED IN DAYS (2026-08-20, E-044). They used to be the
# module constant `(("h1-3", (1,2,3)), ("h4-6", (4,5,6)), ("h7-12", 7..12))`,
# which is a partition of the MONTHLY axis wearing the clothes of a partition
# of time: at pentad it covered the first 60 of 365 days and left 61 of 73
# leads in no band at all, while `band_key` renamed h1-3 to `h1-3_5-15d` and
# so labelled a 5-15 DAY correlation with the key an archived 30-91 day one
# carries. The edges are the tropical year quartered, halved and whole; the
# steps of a given axis follow from them (`TimeAxis.bands`).
YEAR_DAYS = 365.2425
BAND_EDGE_DAYS = (YEAR_DAYS / 4.0,        # 91.310625 d
                  YEAR_DAYS / 2.0,        # 182.62125 d
                  YEAR_DAYS)              # 365.2425 d


# The gate reference is a property of ONE CADENCE, not of the evaluator.
# 0.643 was measured by ml/rollout.py over the MONTHLY family-3 axis; at
# pentad the same head sees a different axis, a different step length and a
# different number of starts, so the number it would have to reproduce does
# not exist. A registry rather than a constant makes that explicit: a cadence
# with no entry has NO reference, and the roll says so in the artefact
# instead of passing a test it never took (see `gate_for_cadence`).
GATE_REF_BY_CADENCE = {"monthly": GATE_REF}


def gate_for_cadence(cadence):
    """(reference or None, reason-when-None). One place decides."""
    ref = GATE_REF_BY_CADENCE.get(cadence)
    if ref is not None:
        return ref, None
    return None, (
        f"no validation-gate reference exists at {cadence} cadence. The "
        f"{GATE_HEAD} reference (auc {GATE_REF['auc']}, bands "
        + ", ".join(f"{k} {v}" for k, v in GATE_REF["bands"].items())
        + ") was measured by ml/rollout.py over the MONTHLY family-3 axis "
        "(#217); it cannot certify a roll whose steps are a different "
        "length, whose starts are a different count and whose horizon "
        "bands span different durations. Passing it here would be a "
        "certificate for an experiment nobody ran. The numbers below are "
        "therefore UNCERTIFIED: report them as a first reading, and "
        "register a reference for this cadence in GATE_REF_BY_CADENCE once "
        "one has been measured and published.")


def _ym_add(label, k):
    """`YYYY-MM` advanced by k calendar months (k may be negative)."""
    n = int(label[:4]) * 12 + int(label[5:7]) - 1 + int(k)
    return f"{n // 12:04d}-{n % 12 + 1:02d}"


class TimeAxis:
    """The tensor's OWN time axis, DERIVED from its metadata — never told.

    Why this class exists (2026-08-19). `ml/build_family4.py:897` emits one
    `YYYY-MM` **label** per 5-day bin and says so in its own comment:
    "`bin_index` remains the authoritative axis; `months` is a label". That
    is exactly right for the two questions train.py asks of the array —
    `m[:4]` for the year-blocked holdout and `int(m[5:7])-1` for the season
    token — and wrong for every use that needs a UNIQUE KEY or a CALENDAR
    STEP. This evaluator had four of those, and each was silently wrong at
    pentad rather than loud:

      * `month_index = {m: i for i, m in enumerate(months)}` collapsed 3,142
        pentad bins to 516 keys, 6.09:1, last wins;
      * `for s_off in range(12)` offered 12 staggered starts per holdout
        year where the pentad axis has 73;
      * `ym_to_r` keyed the RAPID truth attach on `YYYYMM`, discarding
        ~83.6% of the pentad series before the band correlations;
      * `long_roll` advanced a MONTH label AND the month-of-year context
        token once per PENTAD step, so the model's own seasonal input ran a
        full simulated year forward in 60 real days — a corrupted INPUT, not
        merely a mislabelled output.

    DETECTION IS FROM THE TENSOR, NOT FROM A FLAG, and it is the same test
    `ml/probe_kfold.py:273` already uses to decode families 4/5 truth rows:
    `"bin_index" in d`. A roll that must be TOLD its cadence will eventually
    be told wrong, and the wrong answer looks exactly like the right one.

    The step length comes from `pentad_days` (build_family4 writes it beside
    `bin_index`) or, failing that, from the `cadence` name — and is then
    CHECKED rather than trusted: `bin_start(bin_index[r], days)` must land in
    the calendar month `months[r]` names, for every row. That is an exact
    invariant with a known answer (ml/CLAUDE.md §4.9) and it fails loudly if
    the step length, the epoch or the label array disagree.

    The monthly path is pinned just as hard, because every published corridor
    AUC came from it: labels must be UNIQUE and CALENDAR-CONTIGUOUS, which is
    what makes `label_of_row(r) == months[r]` and lets one formula serve both
    the in-record and the past-the-end (future roll) cases.
    """

    def __init__(self, d):
        self.labels = [str(m) for m in d["months"]]
        self.T = len(self.labels)
        assert self.T > 0, "empty months array"
        self.year = np.array([int(m[:4]) for m in self.labels])
        self.moy = np.array([int(m[5:7]) - 1 for m in self.labels])
        if "bin_index" in d:
            self._init_binned(d)
        else:
            self._init_monthly()

    # -- the two axes -----------------------------------------------------
    def _init_monthly(self):
        self.monthly = True
        self.days = None
        self.step_days = 365.2425 / 12.0
        self.cadence = "monthly"
        self.detected_from = ("months labels (no `bin_index` member — "
                              "families 2/3)")
        self.bins = None
        seen = {}
        for i, m in enumerate(self.labels):
            if m in seen:
                sys.exit(f"the months array repeats {m!r} at rows {seen[m]} "
                         f"and {i} but carries no `bin_index`: this axis is "
                         f"neither a unique monthly key nor a declared "
                         f"cadence, and every index this evaluator builds "
                         f"would silently keep one row per label. Refusing.")
            seen[m] = i
        self._row = seen
        for i in range(self.T - 1):
            if _ym_add(self.labels[i], 1) != self.labels[i + 1]:
                sys.exit(f"monthly axis is not calendar-contiguous: row {i} "
                         f"is {self.labels[i]} and row {i + 1} is "
                         f"{self.labels[i + 1]}. The roll advances by ROWS "
                         f"and labels by the calendar; with a gap the two "
                         f"disagree and the long-roll labels would be wrong. "
                         f"Refusing.")

    def _init_binned(self, d):
        self.monthly = False
        bins = np.asarray(d["bin_index"]).astype(np.int64)
        assert len(bins) == self.T, (len(bins), self.T)
        step = np.diff(bins)
        if self.T > 1 and not (step == 1).all():
            bad = int(np.argmax(step != 1))
            sys.exit(f"`bin_index` is not consecutive: row {bad} is bin "
                     f"{bins[bad]} and row {bad + 1} is bin {bins[bad + 1]}. "
                     f"The roll advances one AXIS ROW per step, so a gap in "
                     f"the axis is a gap in simulated time that nothing "
                     f"downstream could see. Refusing.")
        name = str(d["cadence"]) if "cadence" in d else ""
        if "pentad_days" in d:
            days = int(np.asarray(d["pentad_days"]).item())
            src = "`pentad_days`"
        elif name in ("pentad", "daily"):
            days = {"pentad": 5, "daily": 1}[name]
            src = f"`cadence` == {name!r}"
        else:
            sys.exit("the tensor carries `bin_index` but neither "
                     "`pentad_days` nor a known `cadence` name, so the "
                     "length of one axis step is unknown. Every horizon, "
                     "band and lowpass window this evaluator reports is "
                     "measured in steps; without the step length it can "
                     "only report numbers whose unit it is guessing. "
                     "Refusing.")
        self.days = days
        self.step_days = float(days)
        self.cadence = name or (f"{days}-day" if days != 5 else "pentad")
        self.epoch = (dt.date.fromisoformat(str(d["epoch"]))
                      if "epoch" in d else EPOCH_DEFAULT)
        self.bins = bins
        self.b0 = int(bins[0])
        self.detected_from = (f"`bin_index` ({self.T} consecutive bins from "
                              f"{self.b0}) + {src} = {days} d, epoch "
                              f"{self.epoch}")
        # date_of_row() is bin_start() with the TENSOR's epoch rather than the
        # module constant; where the two epochs agree they must agree exactly,
        # or this file is doing its own bin arithmetic (ml/CLAUDE.md §4.1).
        if self.epoch == EPOCH_DEFAULT:
            for r in (0, self.T // 2, self.T - 1):
                assert self.date_of_row(r) == bin_start(int(bins[r]), days), \
                    (f"row {r}: local bin arithmetic {self.date_of_row(r)} "
                     f"!= aggregate_cadence.bin_start "
                     f"{bin_start(int(bins[r]), days)}")
        # EXACT invariant, ml/CLAUDE.md §4.9: the bin's own start date must
        # land in the calendar month the label names, for EVERY row. This is
        # what ties the derived step length to the stored labels; a wrong
        # `days`, a wrong epoch or a shuffled label array cannot survive it.
        for r in range(self.T):
            b = self.date_of_row(r)
            if f"{b.year:04d}-{b.month:02d}" != self.labels[r]:
                sys.exit(f"row {r}: bin {int(bins[r])} starts {b} but the "
                         f"label says {self.labels[r]}. The cadence derived "
                         f"from this tensor ({days} d from {self.epoch}) "
                         f"does not reproduce its own labels — one of the "
                         f"two is wrong and this evaluator cannot tell "
                         f"which. Refusing.")

    # -- rows -> time ------------------------------------------------------
    def date_of_row(self, r):
        """Start date of axis row r. Defined for r >= T (the future roll)."""
        if self.monthly:
            lab = self.label_of_row(r)
            return dt.date(int(lab[:4]), int(lab[5:7]), 1)
        return self.epoch + dt.timedelta(
            days=(self.b0 + int(r)) * self.days)

    def label_of_row(self, r):
        """The row's own label. `YYYY-MM` monthly, ISO date at a binned
        cadence — the label carries the unit, so a pentad roll's trajectory
        can never be read as a monthly one."""
        if self.monthly:
            return _ym_add(self.labels[0], int(r))
        return self.date_of_row(r).isoformat()

    def moy_of_row(self, r):
        """Month-of-year (0-11) of row r, for the model's season token.
        Defined past the end of the record, so the future roll feeds the
        model a TRUE date rather than an incremented counter."""
        if self.monthly:
            return (int(self.labels[0][5:7]) - 1 + int(r)) % 12
        return self.date_of_row(r).month - 1

    # -- time -> rows ------------------------------------------------------
    def row_of_label(self, s):
        """Row for a `YYYY-MM` (or ISO `YYYY-MM-DD`) spec, or None.

        At a binned cadence a bare `YYYY-MM` resolves to the FIRST row of
        that month — deterministic, and the opposite of the last-wins dict
        it replaces."""
        if not s:
            return None
        if self.monthly:
            return self._row.get(s)
        try:
            if len(s) >= 10:
                d0 = dt.date(int(s[:4]), int(s[5:7]), int(s[8:10]))
            else:
                d0 = dt.date(int(s[:4]), int(s[5:7]), 1)
        except ValueError:
            return None
        r = (d0 - self.epoch).days // self.days - self.b0
        if len(s) < 10:                      # first row INSIDE that month
            while 0 <= r < self.T and self.date_of_row(r) < d0:
                r += 1
        if not (0 <= r < self.T):
            return None
        if len(s) < 10 and self.labels[r] != f"{d0.year:04d}-{d0.month:02d}":
            return None
        return int(r)

    def rows_in_year(self, Y):
        return np.where(self.year == int(Y))[0]

    def starts_for_year(self, Y, per_year=0):
        """Staggered roll starts for holdout year Y, spanning the axis's
        REAL steps-per-year: the last row before Y (so h=1 lands on Y's
        first row) plus every row inside Y except its last.

        At monthly this is exactly the old `for s_off in range(12)` list —
        Dec(Y-1), Jan … Nov — in the same order. At pentad it is 73 starts,
        not 12.

        `per_year` (0 = all, and the default, i.e. everything above unchanged)
        SUBSAMPLES that list: every k-th start, k = len(list) // N, first N.
        The starts are a COST, not a protocol constant — at pentad the roll
        pays 73 of them for 73 leads each — and the count is the free
        parameter of the E-044 horizon decision. The rule is a fixed stride
        rather than a random or an edge-weighted choice for three reasons:
        it is deterministic (a re-roll of the same head scores the same rows),
        it keeps the FIRST start, which is the one whose h=1 lands on the
        year's first row, and a constant stride spreads the starts evenly
        round the seasonal cycle — at pentad, N=3 gives k=24 and phases near
        1 Jan / 1 May / 1 Sep, so lead time is not confounded with season.
        N >= len(list) (and N <= 0) return the full list untouched, which is
        what keeps the monthly path — list AND order — bit-identical."""
        rows = self.rows_in_year(Y)
        if len(rows) == 0:
            return []
        out = [int(rows[0]) - 1] if int(rows[0]) - 1 >= 0 else []
        out = out + [int(r) for r in rows[:-1]]
        n = int(per_year or 0)
        if n <= 0 or n >= len(out):
            return out
        return out[::len(out) // n][:n]

    # -- durations ---------------------------------------------------------
    @property
    def steps_per_year(self):
        return 12.0 if self.monthly else 365.2425 / self.days

    def steps_for_days(self, n_days):
        return max(1, int(round(n_days / self.step_days)))

    def steps_for_months(self, n_months):
        """Steps covering n CALENDAR months. Exactly n at monthly."""
        if self.monthly:
            return int(n_months)
        return self.steps_for_days(n_months * 365.2425 / 12.0)

    def span_days(self, n_steps):
        return round(n_steps * self.step_days, 1)

    def describe(self):
        return (f"time axis: {self.cadence} · T={self.T} · "
                f"{self.labels[0]}..{self.labels[-1]} · "
                f"{self.steps_per_year:.4g} steps/year · one step = "
                f"{self.step_days:.4g} d · detected from "
                f"{self.detected_from}")

    def band_key(self, name, hs):
        """Horizon-band label CARRYING ITS UNIT at any non-monthly cadence.
        `h1-3` at monthly (what every published artefact calls it); at pentad
        `h1-18_5-90d`, so a band can never be read as months by accident."""
        if self.monthly:
            return name
        return (f"{name}_{self.span_days(min(hs)):g}-"
                f"{self.span_days(max(hs)):g}d")

    # -- the two DAY-DEFINED quantities -----------------------------------
    def bands(self):
        """`((name, (h, ...)), ...)` — the horizon bands of THIS axis.

        Cut at `BAND_EDGE_DAYS`, half-open on the left: band i holds every
        step h whose lead `h * step_days` lies in `(edge[i-1], edge[i]]`. The
        band therefore means the same DURATION at every cadence, which is the
        whole point — `h1-3` was a partition of the monthly axis, not of time.

        EXACT at monthly (ml/CLAUDE.md §4.9): step_days is 365.2425/12 and the
        three ratios edge/step_days are 3.0, 6.0 and 12.0 with no residue in
        IEEE double, so this returns (("h1-3", (1,2,3)), ("h4-6", (4,5,6)),
        ("h7-12", (7..12))) — the literal it replaces, name for name and step
        for step. tests/test_roll_bands_daydefined.py asserts that identity
        rather than trusting it, and tests/test_roll_monthly_identity.py
        proves the whole artefact did not move. At pentad (5 d) the same edges
        give h1-18 / h19-36 / h37-73; at daily h1-91 / h92-182 / h183-365.

        The `+ 1e-9` guards a representation ulp ONLY: it cannot move a
        boundary at any cadence this repo has (the three monthly ratios are
        exact integers, and pentad's 18.262125 / 36.52425 / 73.0485 and
        daily's 91.310625 / 182.62125 / 365.2425 are nowhere near one).

        Like the constant it replaces, this is a property of the AXIS and not
        of --horizon: callers filter with `h <= Hh`, exactly as before, so a
        short roll simply leaves the later bands empty."""
        out, lo_h = [], 1
        for edge in BAND_EDGE_DAYS:
            hi_h = int(math.floor(edge / self.step_days + 1e-9))
            if hi_h < lo_h:
                continue
            out.append((f"h{lo_h}-{hi_h}", tuple(range(lo_h, hi_h + 1))))
            lo_h = hi_h + 1
        return tuple(out)

    def daymatched_leads(self):
        """The axis steps standing closest to the monthly archive's 12 leads.

        `horizon_auc` is the unweighted mean of msss_clim over h = 1..H, so
        its value depends on WHICH LEADS the axis offers: 12 leads spanning
        30-365 d at monthly, 73 spanning 5-365 d at pentad — most of them
        short, where skill is highest. Comparing those two means directly
        would report a lead-sampling difference as a forecasting result.

        These are the same twelve DURATIONS at any cadence: 1..12 at monthly
        (so `horizon_auc_daymatched` is `horizon_auc` there, exactly),
        {6,12,18,24,30,37,43,49,55,61,67,73} at pentad = 30/60/90/120/150/
        185/215/245/275/305/335/365 d, within 2.4 d of the monthly leads
        everywhere, and {30,61,91,...,365} at daily."""
        m_days = YEAR_DAYS / 12.0
        return tuple(max(1, int(math.floor(m * m_days / self.step_days + 0.5)))
                     for m in range(1, 13))


def dilate8(m, iters):
    """Binary dilation, 3×3 square structuring element. NOT np.roll — roll
    wraps, and this window's longitudes don't."""
    for _ in range(iters):
        p = np.zeros((m.shape[0] + 2, m.shape[1] + 2), bool)
        p[1:-1, 1:-1] = m
        m = (p[:-2, :-2] | p[:-2, 1:-1] | p[:-2, 2:]
             | p[1:-1, :-2] | p[1:-1, 1:-1] | p[1:-1, 2:]
             | p[2:, :-2] | p[2:, 1:-1] | p[2:, 2:])
    return m


def corridor_pixels(Xm, ocean, ys, xs, t_hold, sec_sel, pctl, dilate):
    """Bool [P]: the AMOC corridor (plan §3.6). Mean of RAW cur_speed
    (channel 0) over TRAIN months, observed samples only; threshold at the
    `pctl` percentile of that mean over window ocean; dilate; ∪ section."""
    s = np.zeros(ocean.shape, np.float64)
    n = np.zeros(ocean.shape, np.int64)
    tr = np.where(~t_hold)[0]
    for i0 in range(0, len(tr), 16):
        xb = np.asarray(Xm[tr[i0:i0 + 16], :, :, 0])
        f = np.isfinite(xb)
        s += np.where(f, xb, 0.0).sum(0)
        n += f.sum(0)
    with np.errstate(invalid="ignore"):
        mean_sp = np.where(n > 0, s / np.maximum(n, 1), np.nan)
    vals = mean_sp[ys, xs]
    thr = float(np.percentile(vals[np.isfinite(vals)], pctl))
    core = np.zeros(ocean.shape, bool)
    core[ys, xs] = np.where(np.isfinite(vals), vals, -np.inf) >= thr
    mask2d = dilate8(core, dilate) & ocean
    cp = mask2d[ys, xs]
    cp[sec_sel] = True
    return cp, thr


def export_mask(path, lats, lons, ocean, ys, xs, corridor, gate_mask,
                sec_sel, sec_y, corridor_def, months, x_name, holdout_lon):
    """Write the eval's OWN pixel sets as a baked categorical grid the globe
    app can draw (root CLAUDE.md §2 `classGrid` + `packed` format, row 0 =
    south). Chris, 2026-08-13: *"add a layer to the globe visualiser to see
    which pixels will all be rolled forward in the amoc eval"*.

    It is written HERE, by the evaluator, from the same `corridor_pixels`
    call the scoring uses, so the picture cannot drift from the experiment —
    a hand-drawn corridor in the frontend would be a second definition, and
    the second definition is always the one that goes stale.

    Classes are NESTED (section ⊂ corridor ⊂ rolled); each cell shows its
    most specific one:
      1 rolled   — a window ocean pixel the roll advances every step
      2 corridor — also scored as the headline AMOC corridor
      3 section  — also on the RAPID 26.5°N transport section
    Land and out-of-window cells are empty ("."), which is the honest
    answer: the model has no state there at all.

    The held-out longitude block rides along as an ADDITIVE header field
    (`holdout_lon`), NOT as a 4th class code. Three reasons, in order of
    weight:
      1. It is orthogonal to the nesting. section subset-of corridor
         subset-of rolled is a chain and each cell shows its most specific
         role; "never trained" crosses all three. A pixel both on the RAPID
         section and inside the block would have to give up one of the two
         facts to fit one code -- and whichever it gave up, the legend would
         be lying about the other.
      2. The block IS a longitude interval -- two numbers. Baking it per-cell
         would be a redundant second encoding of lo/hi that can drift from
         the checkpoint the roll actually used.
      3. The `classes` array is the data producer's own palette and the
         frontend paints from it (root CLAUDE.md 2.3), so a new code is a UI
         change: tests/data.spec.js pins the code set to [1,2,3] and the
         packed alphabet to {".",1,2,3}, and ml/paper/make_figs.py rebuilds
         the eval's pixel order from `code >= 1` and counts the corridor as
         `code >= 2`. A 4th code would silently reclassify both.
    A frontend that wants to draw the band has lo/hi in the same degrees as
    west/east and can shade it without a re-bake."""
    H, W = ocean.shape
    code = np.zeros((H, W), np.uint8)
    code[ys, xs] = 1
    code[ys[corridor], xs[corridor]] = 2
    code[ys[sec_sel], xs[sec_sel]] = 3
    # NO row flip here, unlike scripts/refresh_data.py's drivers bake: the app's
    # grid format wants row 0 = SOUTH and this tensor is already south-first
    # (lats[0] = 0.0 N, lats[-1] = 70.0 N). Flipping "to be safe" is what the
    # first version did, and it put the Gulf Stream at the latitude of the
    # Norwegian Sea — a map that still looks like a plausible ocean, which is
    # exactly why the assertion below exists rather than an eyeball check.
    assert lats[0] < lats[-1], \
        f"lats run north-first ({lats[0]} → {lats[-1]}); this writer assumes " \
        f"south-first rows and would emit a vertically mirrored grid"
    dlat = float(np.round(np.diff(lats).mean(), 6))
    dlon = float(np.round(np.diff(lons).mean(), 6))
    payload = {
        "id": "amoc-eval",
        "title": "AMOC eval: the pixels the model rolls forward",
        "units": "role",
        "source": ("earth / E-022 · ml/rollout_spatial.py over the "
                   f"{x_name} tensor"),
        "citation": (
            "Every pixel this experiment advances one month at a time. The "
            "stage-2 head predicts each pixel's next embedding; the E-022 "
            "evaluator rolls ALL window ocean pixels (a stencil head reads "
            "its neighbours, so a region roll would need a growing halo), "
            "then scores an AMOC corridor derived from the data itself: mean "
            "current speed over training months at or above the "
            f"{corridor_def['pctl']:g}th percentile, dilated "
            f"{corridor_def['dilate_cells']} cells, unioned with the RAPID "
            "26.5°N section. Not hand-drawn, and written by the scoring code."),
        "doc": ("https://github.com/blauewelt/earth/blob/main/ml/plans/"
                "E022_spatial_coupling.md"),
        "classes": [
            {"code": 1, "label": "Rolled forward", "rgb": [72, 116, 168]},
            {"code": 2, "label": "Scored: AMOC corridor", "rgb": [232, 152, 48]},
            {"code": 3, "label": "RAPID 26.5°N section", "rgb": [235, 74, 96]},
        ],
        # cell CENTRES are the tensor's lats/lons, so the bounds are half a
        # cell outside them (sampleGrid floors from west/south)
        "west": round(float(lons[0]) - dlon / 2, 6),
        "east": round(float(lons[-1]) + dlon / 2, 6),
        "south": round(float(lats[0]) - dlat / 2, 6),
        "north": round(float(lats[-1]) + dlat / 2, 6),
        "dlon": dlon, "dlat": dlat, "nx": W, "ny": H,
        "period": f"{months[0]}–{months[-1]}",
        "counts": {"rolled": int(len(ys)), "corridor": int(corridor.sum()),
                   "section": int(len(sec_sel)),
                   "gate_subset": int(gate_mask.sum())},
        "corridor_def": corridor_def,
        # Additive, deliberately -- see the docstring. The globe can shade
        # the band from lo/hi; the class codes stay [1,2,3].
        "holdout_lon": holdout_lon,
        "section_row": {"lat": round(float(lats[sec_y]), 4),
                        "n_px": int(len(sec_sel))},
        "packed": "".join("." if v == 0 else str(v) for v in code.ravel()),
    }
    # Read the file back the way the BROWSER will (app.js sampleGrid: floor
    # from west/south), and demand the RAPID section land on RAPID's latitude.
    # An exact expected value, checked at the point of writing — the app's own
    # geometry, not this function's, so an off-by-one or a mirrored row cannot
    # leave here (ml/CLAUDE.md §4.9).
    def _probe(lat, lon):
        ix = int(np.floor((lon - payload["west"]) / payload["dlon"]))
        iy = int(np.floor((lat - payload["south"]) / payload["dlat"]))
        return payload["packed"][iy * payload["nx"] + ix]
    sec_lat = float(lats[sec_y])
    for lon in (-70.0, -40.0):
        got = _probe(sec_lat, lon)
        assert got == "3", (
            f"the {sec_lat}°N section reads '{got}' at {lon}°E when sampled "
            f"the way the app samples it — the grid is mis-oriented, refusing "
            f"to write a map that would look fine and be wrong")
    assert _probe(sec_lat + 5 * payload["dlat"], -40.0) != "3", \
        "the section smears across rows — off-by-one in the row index"

    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        json.dump(payload, f, separators=(",", ":"))
    print(f"wrote {path} ({os.path.getsize(path):,} bytes) — "
          f"{payload['counts']['rolled']:,} rolled · "
          f"{payload['counts']['corridor']:,} corridor · "
          f"{payload['counts']['section']} section · "
          + (f"{holdout_lon['px']['window']['in_block']:,} of them inside "
             f"the never-trained lon block [{holdout_lon['lo']}, "
             f"{holdout_lon['hi']})" if holdout_lon.get("any", True)
             else "no lon holdout (all columns trained)"), flush=True)


class StdMonths:
    """Standardized-anomaly month fields at the ocean pixels, cached.
    Exactly build_slab's per-month recipe (dyn channels de-climatologised
    and z-scored, static channels RAW — matching rollout.py's
    anomaly_transform space); returns (values nan→0 [P,C] f32, obs [P,C])."""

    def __init__(self, Xm, ys, xs, moy, clim, dyn, mean_c, std_c):
        self.Xm, self.ys, self.xs, self.moy = Xm, ys, xs, moy
        self.clim, self.dyn = clim, dyn
        self.mean_c, self.std_c = mean_c, std_c
        self.cache = {}

    def get(self, t):
        if t not in self.cache:
            x = np.asarray(self.Xm[t]).astype(np.float32)       # [H,W,C]
            obs = np.isfinite(x)
            for c in self.dyn:
                x[..., c] = ((x[..., c] - self.clim[self.moy[t], :, :, c]
                              - self.mean_c[c]) / (self.std_c[c] + 1e-6))
            x = np.where(obs, x, 0.0)
            self.cache[t] = (x[self.ys, self.xs], obs[self.ys, self.xs])
        return self.cache[t]


def ar1_train(std_m, T, t_hold, P, C):
    """Damped-persistence AR1 coefficient per (pixel, channel) — rollout.py's
    construction: lag-1 pairs whose FIRST month is a train month, both months
    observed, clip [0, 0.999], zero under 24 pairs."""
    acc = {k: np.zeros((P, C), np.float64)
           for k in ("n", "sx", "sy", "sxx", "syy", "sxy")}
    prev = None
    for t in range(T):
        v, o = std_m.get(t)
        if prev is not None and not t_hold[t - 1]:
            pv, po = prev
            m = po & o
            x0 = np.where(m, pv, 0.0)
            x1 = np.where(m, v, 0.0)
            acc["n"] += m
            acc["sx"] += x0
            acc["sy"] += x1
            acc["sxx"] += x0 * x0
            acc["syy"] += x1 * x1
            acc["sxy"] += x0 * x1
        prev = (v, o)
        if t >= 2:                       # the cache only needs a 2-month tail
            std_m.cache.pop(t - 2, None)
    n = acc["n"]
    with np.errstate(invalid="ignore", divide="ignore"):
        cov = acc["sxy"] - acc["sx"] * acc["sy"] / np.maximum(n, 1)
        v0 = acc["sxx"] - acc["sx"] ** 2 / np.maximum(n, 1)
        v1 = acc["syy"] - acc["sy"] ** 2 / np.maximum(n, 1)
        rr = cov / (np.sqrt(np.maximum(v0 * v1, 0.0)) + 1e-9)
    return np.where(n >= 24, np.clip(rr, 0, 0.999), 0.0).astype(np.float32)


def month_feats(moys, dev):
    m = np.asarray(moys)
    return torch.from_numpy(np.stack(
        [np.sin(2 * np.pi * m / 12),
         np.cos(2 * np.pi * m / 12)], 1).astype(np.float32)).to(dev)


def roll_step(model, Zwin, NBR_t, static_ctx, mfeat, chunk, amp=False):
    """One autoregressive step over ALL pixels. Zwin [P, K, d_z] on the
    device; NBR_t None (stencil 1) or [P, S]; mfeat [K, 2]. → ẑ [P, d_z]
    float32. The gather mirrors temporal.gather_stencil's layout exactly
    (slot-major over d_z, centre slot 0, missing → zeros); the zero-weight-
    equivalence test pins that layout at the model boundary. `amp` runs the
    forward under fp16 autocast — a SPEED knob whose honesty is enforced by
    the #217 gate, which scores through the identical path.

    THE CHUNK IS A BYTE BUDGET, NOT A ROW COUNT. `--chunk` counts pixels, and
    the gather below materialises [n, S, K, dz] — so its size scales with the
    STENCIL WIDTH, which the row count knows nothing about. At the 8192
    default a 90-slot head asks for 8192·90·24·64·4 B = 4.5 GB in one
    allocation, and that is precisely the request that killed eval wave 6B
    (#353) partway through its third head on 2026-08-16, after two heads of
    the same width had succeeded — a 24 GB card with 10 GB already reserved
    has no contiguous 4.5 GB left. This is the third OOM of one family: an
    eval batch scales with stencil width as well as model size (E-027
    incidents 1 and 2 taught it for the trainer; `_chunked_forward` is the
    trainer's version of this fix). So the row count is derived from a byte
    target and the caller's `--chunk` becomes an upper bound, which makes the
    guard automatic for any future width instead of a number someone has to
    remember to lower."""
    P = Zwin.shape[0]
    if NBR_t is not None:
        S, K, dz = NBR_t.shape[1], Zwin.shape[1], Zwin.shape[2]
        row_bytes = S * K * dz * 4                    # float32 gather, per pixel
        # ~1 GiB per gather: comfortably inside the free space left on a 24 GB
        # card after weights, the Z window and allocator fragmentation.
        chunk = max(256, min(chunk, (1 << 30) // max(1, row_bytes)))
    outs = []
    for i in range(0, P, chunk):
        sl = slice(i, min(i + chunk, P))
        if NBR_t is None:
            zin = Zwin[sl]
        else:
            nbr = NBR_t[sl]                                   # [n, S]
            miss = nbr < 0
            zj = Zwin[nbr.clamp(min=0)]                       # [n, S, K, dz]
            zj[miss] = 0.0
            zin = zj.permute(0, 2, 1, 3).reshape(
                zj.shape[0], Zwin.shape[1], -1)               # [n, K, S*dz]
        with torch.autocast(device_type="cuda", dtype=torch.float16,
                            enabled=amp):
            pred, _ = model(zin,
                            mfeat[None].expand(zin.shape[0], -1, -1),
                            static_ctx[sl])
        outs.append(pred[:, -1].float())
    return torch.cat(outs, 0)


def decode_all(codec, zhat, C, chunk, amp=False):
    """codec.query at every (pixel, channel), offset 0 — [P, C] numpy."""
    outs = []
    for i in range(0, zhat.shape[0], chunk):
        z = zhat[i:i + chunk]
        n = z.shape[0]
        qc = torch.arange(C, device=z.device)[None].expand(n, -1)
        off0 = torch.zeros(n, C, 3, dtype=torch.long, device=z.device)
        with torch.autocast(device_type="cuda", dtype=torch.float16,
                            enabled=amp):
            xq = codec.query(z, qc, off0)
        outs.append(xq.float().cpu().numpy())
    return np.concatenate(outs, 0)


def scored_horizon(ax, s, Hh, T, Y):
    """How many of the Hh leads from start row `s` are SCORED for holdout
    year Y.

    The roll breaks at the record's end and at the year boundary
    (`ax.year[s + h] != Y`), so a start late in Y contributes fewer steps
    than the horizon asks for — which is why the monthly protocol's twelve
    starts cost 78 steps a year and not 144. One expression, used by the
    step PLAN and by `RollDump`, because two copies of a break condition are
    two chances to disagree about how long a trajectory is."""
    n = 0
    for h in range(1, Hh + 1):
        if s + h >= T or ax.year[s + h] != int(Y):
            break
        n += 1
    return n


class RollDump:
    """`--dump-roll DIR`: the rolled Z TRAJECTORY of every scored start.

    Chris, 2026-08-22: *"Save the roll forward sequence for the held out years
    somewhere (so that we can use it as animation in the UI)"*, alongside
    *"Roll forward all of the earth's pixels (these are required by the
    stencil size, not just the relevant area)"*. The second half is already
    true and always was — `roll_step` advances the FULL window state, all P
    ocean pixels, every step, because a stencil head's step t+1 at pixel p
    reads p's NEIGHBOURS at t (module docstring §1), and the gate/corridor/
    window scopes are boolean masks applied to the DECODED field afterwards.
    What was missing is that the state itself was discarded one step after it
    existed: the roll kept scalar skill sums of it and nothing else.

    WHAT IS SAVED IS Z, NOT PIXELS, deliberately. Decoding [n+1, P, C] on the
    box costs a second decode pass per step and ships ~20x the bytes at
    family 4's C=40 against d_z 32; the decoder is published, frozen and
    deterministic, so pixel space is recoverable offline for whichever
    channels the UI actually draws. What is NOT recoverable after the fact is
    WHICH codec speaks this z, so `codec_weight_hash` — the same identity the
    embed cache is named by — rides in every file and in the manifest. That
    is the #10/#11 failure mode written down: a z decoded by a different
    codec passes every shape check and is beautiful nonsense.

    float16 because that is the dtype the embed cache itself stores (Z is
    read straight off a float16 memmap, so state 0 is not even a conversion),
    and because the alternative doubles a 3.7 GB artefact to say nothing new.

    OPT-IN AND ADDITIVE. Nothing is written unless --dump-roll names a
    directory; no key of the roll JSON moves either way; and the GATE head is
    never dumped — it is a certificate, not an experiment, and its
    trajectories would double the bytes for a head nobody will animate.
    """

    def __init__(self, dirpath, ax, ys, xs, lats, lons, grid_shape,
                 ckpt_path, ck, cfg):
        self.dir = dirpath
        os.makedirs(self.dir, exist_ok=True)
        self.ax = ax
        self.path = os.path.join(self.dir, "dump_manifest.json")
        # The pixel index map travels IN EVERY FILE (0.3% of one file's
        # bytes) rather than only in the manifest: a trajectory whose pixel
        # order is defined in another file is one careless copy away from
        # being unreadable, and this data is meant to outlive the run.
        self.px = {
            "px_y": np.asarray(ys, np.int32),
            "px_x": np.asarray(xs, np.int32),
            "px_lat": np.asarray(lats, np.float32)[np.asarray(ys)],
            "px_lon": np.asarray(lons, np.float32)[np.asarray(xs)],
            "grid_lats": np.asarray(lats, np.float32),
            "grid_lons": np.asarray(lons, np.float32),
            "grid_shape": np.asarray(grid_shape, np.int32),
        }
        self.codec = {"file": os.path.basename(ckpt_path),
                      "tag": str(ck.get("tag") or ""),
                      "step": int(ck.get("step", -1)),
                      "d_z": int(ck["d_z"]),
                      "weight_hash": codec_weight_hash(ck)}
        self.man = {
            "written_by": "ml/rollout_spatial.py --dump-roll",
            "created_utc": dt.datetime.now(dt.timezone.utc)
                             .isoformat(timespec="seconds"),
            "cadence": ax.cadence, "step_days": ax.step_days,
            "axis_detected_from": ax.detected_from,
            "codec": self.codec, "dtype": "float16",
            "n_px": int(len(ys)), "d_z": int(ck["d_z"]),
            "state_0": "the TRUE embedding of the start row (Z[start_row]); "
                       "states 1..n are the model's own predictions, each fed "
                       "back as the next input",
            "pixel_order": "row-major over the window's ocean mask; px_y/px_x "
                           "index the tensor grid, px_lat/px_lon are the same "
                           "pixels in degrees. Identical in every file and to "
                           "the Z the roll read.",
            "decode_note": "z, not pixels: decode offline with the codec named "
                           "above (weight_hash must match) — see "
                           "ml/rollout_spatial.py decode_all().",
            "config": dict(cfg), "files": [], "total_bytes": 0}
        self._flush()

    def _flush(self):
        with open(self.path, "w") as fh:
            json.dump(self.man, fh, indent=1)

    def plan(self, heads, starts_by_year, horizon, P, d_z):
        """Say what this will cost BEFORE it is spent (ml/CLAUDE.md §0.3)."""
        n_roll = len(heads) * sum(starts_by_year.values())
        gb = n_roll * (horizon + 1) * P * d_z * 2 / 1e9
        print(f"--dump-roll {self.dir}: up to {n_roll} trajectories "
              f"({len(heads)} non-gate head(s) x {sum(starts_by_year.values())} "
              f"scored starts), <= {horizon + 1} states of [{P:,}, {d_z}] "
              f"float16 each — at most {gb:.2f} GB. Starts truncated at the "
              f"year boundary write fewer states, so this is a ceiling.",
              flush=True)

    def write(self, head_label, head_file, head_meta, Y, s, z):
        """One (year, start) trajectory. `z` is [n_states, P, d_z] float16."""
        rows = [int(s) + k for k in range(z.shape[0])]
        name = f"roll_{head_label}_{Y}_r{int(s)}.npz"
        path = os.path.join(self.dir, name)
        meta = {**head_meta, "head": head_label, "head_file": head_file,
                "year": str(Y), "start_row": int(s),
                "start_label": self.ax.label_of_row(s),
                "cadence": self.ax.cadence, "step_days": self.ax.step_days,
                "codec": self.codec}
        np.savez(path, z=z,
                 rows=np.asarray(rows, np.int32),
                 labels=np.asarray([self.ax.label_of_row(r) for r in rows]),
                 dates=np.asarray([self.ax.date_of_row(r).isoformat()
                                   for r in rows]),
                 meta_json=np.asarray(json.dumps(meta)),
                 **self.px)
        nb = os.path.getsize(path)
        self.man["files"].append({
            "file": name, "head": head_label, "head_file": head_file,
            "year": str(Y), "start_row": int(s),
            "start_label": self.ax.label_of_row(s),
            "n_states": int(z.shape[0]),
            "shape": [int(v) for v in z.shape], "bytes": int(nb),
            "rows": [rows[0], rows[-1]],
            "dates": [self.ax.date_of_row(rows[0]).isoformat(),
                      self.ax.date_of_row(rows[-1]).isoformat()]})
        self.man["total_bytes"] = int(self.man["total_bytes"]) + int(nb)
        # REWRITTEN AFTER EVERY FILE, like the roll JSON itself (c31a679): a
        # 14-hour job that is cancelled at hour 12 must leave a manifest that
        # describes what is actually on the disk.
        self._flush()
        print(f"  dump {name}: {z.shape[0]} states "
              f"[{z.shape[1]:,}, {z.shape[2]}] f16, {nb / 1e6:.0f} MB "
              f"({self.man['total_bytes'] / 1e9:.2f} GB total)", flush=True)


class Progress:
    """Say how far along we are, to the LOG and to the live side channel.

    Chris, 2026-08-14: *"Do you have any sense of progress or an expected end
    time on these evals? The number of rolled pixels is quite large, a progress
    bar would be helpful."* He could not be told, and neither could I: Actions
    will not serve the log of a running job, and unlike the training path this
    eval never wrote to `ml-live-<n>`. Two hours of rented GPU with no way to
    answer "how far?" is a monitoring failure, not a slow job.

    Total work is known before the first step: every roll step is one forward
    over all 84,405 pixels, and the count follows from the protocol (staggered
    starts truncated at the year end, plus the long and future rolls). So an
    ETA is arithmetic, not a guess, from the second step onward."""

    def __init__(self, path, n_heads, every=20):
        self.path, self.n_heads, self.every = path, n_heads, every
        self.t0 = time.time()
        self.head_i = 0
        self.label = ""
        self.total = 1
        self.done = 0
        self.phase = ""

    def start_head(self, i, label, total):
        self.head_i, self.label, self.total, self.done = i, label, max(total, 1), 0
        self.t_head = time.time()

    def step(self, phase, n=1):
        self.phase = phase
        self.done += n
        if self.done % self.every and self.done != self.total:
            return
        el = time.time() - self.t_head
        rate = el / max(self.done, 1)
        eta_head = rate * (self.total - self.done)
        # heads run at the same cost, so the remaining ones are rate x total
        eta_all = eta_head + rate * self.total * (self.n_heads - self.head_i)
        pct = 100.0 * self.done / self.total
        bar = "#" * int(pct // 5) + "-" * (20 - int(pct // 5))
        print(f"  [{bar}] {pct:5.1f}%  head {self.head_i}/{self.n_heads} "
              f"{self.label} · {self.phase} {self.done}/{self.total} steps · "
              f"{el / 60:.1f} min in · ~{eta_head / 60:.0f} min left on this "
              f"head, ~{eta_all / 60:.0f} min to finish", flush=True)
        rec = {"sroll": {"head": self.label, "head_i": self.head_i,
                         "heads": self.n_heads, "phase": self.phase,
                         "done": self.done, "total": self.total,
                         "pct": round(pct, 1),
                         "elapsed_s": round(time.time() - self.t0),
                         "eta_head_s": round(eta_head),
                         "eta_all_s": round(eta_all)}}
        try:
            os.makedirs(os.path.dirname(self.path) or ".", exist_ok=True)
            with open(self.path, "a") as f:
                f.write(json.dumps(rec) + "\n")
        except OSError:
            pass          # instrumentation never breaks the run


def new_sums(H):
    return {k: np.zeros(H + 1) for k in
            ("mse_m", "mse_p", "mse_c", "mse_d", "n",
             "sxy", "sxx", "syy", "sx", "sy")}


def accumulate(su, h, xhat, v_true, v_pers, v_damp, op):
    opf = op.astype(np.float64)
    n = opf.sum()
    if n == 0:
        return
    su["mse_m"][h] += (((xhat - v_true) ** 2) * opf).sum()
    su["mse_p"][h] += (((v_pers - v_true) ** 2) * opf).sum()
    su["mse_d"][h] += (((v_damp - v_true) ** 2) * opf).sum()
    su["mse_c"][h] += ((v_true ** 2) * opf).sum()
    su["n"][h] += n
    xm = xhat * opf
    ym = v_true * opf
    su["sxy"][h] += (xm * ym).sum()
    su["sxx"][h] += (xm * xm).sum()
    su["syy"][h] += (ym * ym).sum()
    su["sx"][h] += xm.sum()
    su["sy"][h] += ym.sum()


def skill_block(su, H, n_px=None, leads=None):
    """rollout.py's chan_skill rows + horizon_auc (mean MSSS-vs-climatology
    — the #217 'AUC'), plus the damped mean for completeness.

    `leads` is `TimeAxis.daymatched_leads()` — the twelve steps standing for
    the monthly archive's twelve lead DURATIONS on this axis. Given them, the
    block also carries `horizon_auc_daymatched`, the mean of msss_clim over
    exactly those leads. ALONGSIDE, never instead: `horizon_auc` is what every
    archived number is, and this is the only quantity that may be compared
    ACROSS cadences. At monthly the leads are 1..12, so over the archive's own
    horizon the two are the same number, computed the same way, rounded the
    same way — asserted in tests/test_roll_monthly_identity.py rather than
    argued. Leads past the rolled horizon are absent from `rows` and are
    dropped; with none left the key is omitted rather than written as a mean
    over nothing (ml/CLAUDE.md §5.22).

    `n_px` is the scope's PIXEL COUNT, carried into the block so an empty
    block can say WHY it is empty. Under a no-longitude-holdout codec
    (`holdout_lon "0,0"`, ml/recipes/*-nolonhold.json, E-043) every
    `<scope>_holdlon` child is a zero-pixel scope: accumulate() takes the
    `n == 0` early return, no sum is ever touched, and the block that comes
    back is `{"chan_skill": []}` with no `horizon_auc`. That is CORRECT — it
    is the §5.22 behaviour, an omitted aggregate rather than a NaN one — but
    it is indistinguishable, in the artefact, from a scope that had pixels
    and scored nothing (a Z that never reached the horizon, a mask bug). One
    integer separates the two, so it is written down: `n_px 0` means "there
    was nothing to score", `n_px > 0` with no rows means "something is
    wrong". `n_px` is None only for a caller that did not pass it, and the
    key is then omitted, so old readers are unaffected."""
    rows = []
    for h in range(1, H + 1):
        if su["n"][h] == 0:
            continue
        n_ = su["n"][h]
        mm, mp = su["mse_m"][h] / n_, su["mse_p"][h] / n_
        md, mc = su["mse_d"][h] / n_, su["mse_c"][h] / n_
        vx = su["sxx"][h] / n_ - (su["sx"][h] / n_) ** 2
        vy = su["syy"][h] / n_ - (su["sy"][h] / n_) ** 2
        cov = su["sxy"][h] / n_ - su["sx"][h] * su["sy"][h] / n_ ** 2
        acc = cov / (np.sqrt(vx * vy) + 1e-12)
        rows.append({"h": h, "n": int(n_),
                     "msss_clim": round(1 - mm / mc, 3),
                     "msss_pers": round(1 - mm / mp, 3),
                     "msss_damped": round(1 - mm / md, 3),
                     "acc": round(float(acc), 3),
                     "amp_ratio": round(float(np.sqrt(vx / (vy + 1e-12))), 3)})
    out = {"chan_skill": rows}
    if n_px is not None:
        out["n_px"] = int(n_px)
        if not rows:
            out["empty"] = ("scope has 0 pixels — nothing to score"
                            if int(n_px) == 0 else
                            "scope has pixels but no horizon scored any of "
                            "them — investigate")
    if rows:
        out["horizon_auc"] = round(
            float(np.mean([r["msss_clim"] for r in rows])), 3)
        out["auc_damped"] = round(
            float(np.mean([r["msss_damped"] for r in rows])), 3)
        if leads is not None:
            by_h = {r["h"]: r["msss_clim"] for r in rows}
            got = [by_h[h] for h in leads if h in by_h]
            if got:
                out["horizon_auc_daymatched"] = round(float(np.mean(got)), 3)
    return out


def deseason_truth(rv, rmoy, tr_all, what):
    """Train-month climatology, REFUSING rather than returning NaN.

    `rclim[m] = rv[train & month == m].mean()` is a mean over an empty slice
    whenever some calendar month carries no TRAINING label — and numpy's
    answer is nan, which then flows through `rv_des` into every band
    correlation and into the roll json as the literal `NaN` token that
    ml/CLAUDE.md §5.22 forbids. It is a property of the INPUTS alone, so it
    is checked here, where it has cost nothing (§5.16), and the message names
    the months rather than leaving a reader to find them."""
    rclim = np.zeros(12)
    empty = []
    for m in range(12):
        sel = tr_all & (rmoy == m)
        if not sel.any():
            empty.append(m + 1)
        else:
            rclim[m] = rv[sel].mean()
    if empty:
        sys.exit(
            f"{what}: calendar month(s) {empty} carry NO training label, so "
            f"the train-month climatology is undefined there and every "
            f"deseasonalised value in those months would be NaN — which "
            f"ml/CLAUDE.md §5.22 forbids writing into a results file. "
            f"{int(tr_all.sum())} of {len(tr_all)} labels are training "
            f"labels. Refusing rather than scoring a series with holes in it.")
    return rclim


def corr_or_none(x, y):
    """Pearson r, or None when it is not defined (a constant series, a
    degenerate band). None is OMISSION — the shape `skill_block` already uses
    for a scope it cannot score — never a NaN in the artefact."""
    if len(x) < 2:
        return None
    r = np.corrcoef(x, y)[0, 1]
    return None if not np.isfinite(r) else round(float(r), 3)


def smooth(vals, k=18, min_valid=12):
    """Centred running mean — the house 18-month lowpass (plot_projection)."""
    v = np.asarray(vals, float)
    out = np.full(len(v), np.nan)
    for i in range(len(v)):
        w = v[max(0, i - k // 2): min(len(v), i + (k - k // 2))]
        w = w[np.isfinite(w)]
        if len(w) >= min_valid:
            out[i] = w.mean()
    return out


def write_results(path, results, partial=None):
    """Write the roll's result file ATOMICALLY, optionally marked partial.

    A roll is hours of rented GPU and its scored numbers exist LONG before it
    ends (at pentad the long/future phases are 87% of the wall clock), so the
    file is rewritten at every phase boundary rather than once at the end
    (ml/CLAUDE.md §5.25). Two properties make that safe:

      * ATOMIC. The bytes land in `path + ".tmp"` in the same directory and
        `os.replace` swaps them in, so a reader — the live publisher runs
        every ~2.5 min — sees either the previous complete file or the new
        one, never a half-written one, and no `.tmp` survives.
      * MARKED. A write with `partial` set carries a top-level `in_progress`
        key describing where the roll is; a reader that finds it must treat
        every number under it as provisional. `partial=None` writes EXACTLY
        what this file has always written — same indent, no extra keys — so
        the final artefact stays byte-identical to the archive
        (tests/test_roll_monthly_identity.py).
    """
    d = os.path.dirname(path) or "."
    os.makedirs(d, exist_ok=True)
    payload = results if partial is None else dict(results,
                                                   in_progress=partial)
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(payload, f, indent=1)
    os.replace(tmp, path)


def _utc_now():
    return dt.datetime.now(dt.timezone.utc).replace(
        microsecond=0).isoformat().replace("+00:00", "Z")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--x", default="",
                    help="the tensor memmap; not needed for --dump-truth")
    ap.add_argument("--npz-small", required=True)
    ap.add_argument("--z", help="Z cache .npy (f16 [T,P,dz])")
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--heads", nargs="+")
    ap.add_argument("--dump-truth", default="",
                    help="write the deseasonalised RAPID series to this path "
                         "and exit — the plotting scripts' truth source, so "
                         "the climatology rule has exactly one implementation. "
                         "Needs only --npz-small and --ckpt.")
    ap.add_argument("--out", help="rollout_spatial.json (not needed with "
                                  "--export-mask-only)")
    ap.add_argument("--export-mask",
                    help="also write the eval's pixel sets as a baked "
                         "categorical grid for the globe app "
                         "(data/amoc_eval_mask.json)")
    ap.add_argument("--export-mask-only", action="store_true",
                    help="write that grid and stop — needs no Z, no heads "
                         "and no GPU, because the masks depend only on the "
                         "tensor and the corridor recipe")
    ap.add_argument("--horizon", type=int, default=12,
                    help="rolled horizon in AXIS STEPS. 12 is the monthly "
                         "archive's 12 months and stays the default at every "
                         "cadence, because a step count that silently means a "
                         "different duration per tensor is the bug this file "
                         "spent 2026-08-19 removing (§5.24). At pentad the "
                         "day-matched value is 73 (365.0 d) and "
                         "scripts/sroll_run.sh computes and passes it; a "
                         "non-monthly roll that is NOT day-matched warns here")
    ap.add_argument("--starts-per-year", type=int, default=0,
                    help="score only N of the staggered starts per holdout "
                         "year — every k-th, k = len(starts)//N, first N. 0 "
                         "(the default) is every start, which is 12 at "
                         "monthly and 73 at pentad. The starts are the free "
                         "parameter of the pentad cost: at --horizon 73 three "
                         "holdout years cost 219 scored steps at N=1, 441 at "
                         "N=3 and 8,103 at all 73. What is recorded in the "
                         "artefact is the N *and the rows chosen*")
    ap.add_argument("--dump-roll", default="",
                    help="directory for the ROLL-FORWARD SEQUENCES of the "
                         "scored holdout-year starts: one .npz per (head, "
                         "year, start) holding the full-window z trajectory "
                         "[n_states, P, d_z] float16 (state 0 = the true "
                         "embedding of the start row), the axis rows/labels/"
                         "dates, the pixel index map and the codec's weight "
                         "hash, plus a dump_manifest.json. For the UI's "
                         "animation (Chris, 2026-08-22); decoding to pixels "
                         "happens offline, from these files and the published "
                         "decoder. Empty (the default) writes NOTHING and the "
                         "roll is bit-identical. The gate head is never "
                         "dumped. Budget ~%.2f GB per 74-state pentad roll at "
                         "86,698 px x d_z 32" % (74 * 86698 * 32 * 2 / 1e9))
    ap.add_argument("--chunk", type=int, default=8192)
    ap.add_argument("--map-h", type=int, default=6,
                    help="horizon for the E-026b per-pixel skill map "
                         "(mid-roll by default: late enough for the reach "
                         "divergence to be visible, early enough that most "
                         "starts still reach it)")
    ap.add_argument("--pixels-gate", type=int, default=600)
    ap.add_argument("--corridor-pctl", type=float, default=75.0)
    ap.add_argument("--corridor-dilate", type=int, default=2)
    ap.add_argument("--long-start", default="2004-12",
                    help="context end for the long hindcast; '' skips it. "
                         "`YYYY-MM` resolves to the FIRST row of that month "
                         "at a binned cadence; `YYYY-MM-DD` names a bin. A "
                         "COMMA-SEPARATED LIST rolls one hindcast per label: "
                         "the first goes to `long` exactly as it always has, "
                         "the rest to the new `long_multi` block. That is the "
                         "calendar-vs-context phase discriminator (2026-08-22) "
                         "— one context end cannot tell a model that replays "
                         "the calendar from one whose phase its state selects. "
                         "A label with no axis row, or with less than K rows "
                         "of history, is SKIPPED with a printed reason")
    ap.add_argument("--long-months", type=int, default=-1,
                    help="length of the long hindcast in AXIS STEPS. The "
                         "default -1 resolves to 20 YEARS at the tensor's "
                         "own cadence — exactly the historical 240 at "
                         "monthly, 1461 at pentad. A step is not a month "
                         "once the tensor is not monthly, and a fixed 240 "
                         "would silently become 3.3 years")
    ap.add_argument("--future-months", type=int, default=-1,
                    help="the same, past the end of the record; 0 skips it")
    ap.add_argument("--no-gate", action="store_true",
                    help="score without the e017_u1_s0 gate — smoke/toy ONLY")
    ap.add_argument("--amp", action="store_true",
                    help="fp16 autocast for the roll/decode forwards — the "
                         "gate decides whether the numbers survive it")
    ap.add_argument("--metrics", default=os.path.join(HERE, "runs", "actions",
                                                      "metrics.jsonl"),
                    help="progress records are APPENDED here; "
                         "scripts/publish_live_metrics.sh pushes this file to "
                         "ml-live-<n>, which is the only way to watch a "
                         "running job from outside")
    ap.add_argument("--progress-every", type=int, default=20,
                    help="roll steps between progress lines")
    ap.add_argument("--cache-dir", default=os.path.join(HERE, "cache"))
    a = ap.parse_args()
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if dev.type == "cuda":
        # TF32 matmuls: ~2x on Ada at ~10-bit mantissa with fp32 accumulate.
        # The 9-head R4 eval is ~2M forward tokens per roll step x ~700 steps
        # per head — at strict fp32 that is ~11 h of 4090; with TF32 it fits
        # a job_timeout. Numerically far gentler than bf16, and the #217 gate
        # (±0.01) is the check that it changed nothing that matters.
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    if a.export_mask_only and not a.x:
        sys.exit("--export-mask-only needs --x")
    if a.export_mask_only and not a.export_mask:
        sys.exit("--export-mask-only needs --export-mask <path>")
    if not a.export_mask_only and not a.dump_truth:
        missing = [f"--{k}" for k in ("x", "z", "heads", "out")
                   if not getattr(a, k)]
        if missing:
            sys.exit(f"missing {', '.join(missing)} (required unless "
                     f"--export-mask-only)")
    # THE AXIS FIRST, because the gate below is a property of the CADENCE and
    # a gate check that does not know the cadence is the failure this whole
    # change is about. Reading the small npz costs nothing (it holds no X), so
    # this is still "check the precondition where the inputs are all it has
    # cost you" (ml/CLAUDE.md §0.3).
    d = np.load(a.npz_small, allow_pickle=False)
    ax = TimeAxis(d)
    months = ax.labels
    lats, lons = d["lats"], d["lons"]
    T = ax.T
    moy = ax.moy
    print(ax.describe(), flush=True)
    if not ax.monthly:
        # --horizon, --map-h, --long-months and --future-months are STEP
        # counts and always were; at monthly a step is a month and nobody had
        # to think about it. Say what they buy here, because "AUC over h=1..12"
        # is 60 days at pentad and 12 months on every number it would be
        # compared against — a DESIGN choice for the dispatcher, not something
        # this file may quietly rescale (ml/CLAUDE.md §5.24).
        h_match = ax.steps_for_months(12)
        print(f"  --horizon {a.horizon} = {ax.span_days(a.horizon):g} d at "
              f"this cadence; the monthly archive's h=1..12 is 12 MONTHS, "
              f"which here would be --horizon {h_match}. "
              f"--map-h {a.map_h} = {ax.span_days(a.map_h):g} d.", flush=True)
        # ASSERT THE EFFECT, not the invocation (§0.2). Printing the day span
        # was enough to make the horizon VISIBLE and not enough to stop the
        # default being taken: scripts/sroll_run.sh passed no --horizon at all
        # until 2026-08-20, so every pentad roll would have scored 60 days and
        # reported it beside the archive's 365. This does not rescale — that
        # is the dispatcher's decision — it refuses to let it be silent.
        if a.horizon != h_match:
            print(f"::warning::--horizon {a.horizon} is "
                  f"{ax.span_days(a.horizon):g} d at {ax.cadence} cadence, "
                  f"NOT the monthly archive's 12 months "
                  f"({ax.span_days(h_match):g} d = --horizon {h_match}). "
                  f"`horizon_auc` here averages {a.horizon} leads over "
                  f"{ax.span_days(a.horizon):g} d and is NOT comparable with "
                  f"any archived corridor AUC; `horizon_auc_daymatched` is "
                  f"the quantity that is, and it needs leads out to "
                  f"{h_match}.", flush=True)
        print(f"  bands (day-defined, edges "
              + "/".join(f"{e:g}" for e in BAND_EDGE_DAYS) + " d): "
              + " · ".join(f"{ax.band_key(bn, hs)}" for bn, hs in ax.bands())
              + f" · day-matched leads {list(ax.daymatched_leads())}",
              flush=True)
        if a.starts_per_year <= 0:
            print("::warning::--starts-per-year not set: EVERY start of every "
                  "holdout year is scored, which is the monthly protocol's 12 "
                  "and this axis's full steps-per-year. At pentad that is 73 "
                  "starts per holdout year — a COST decision the dispatch "
                  "should make deliberately (E-044 recommends 3, giving "
                  "phases near 1 Jan / 1 May / 1 Sep).", flush=True)

    # gate discipline up front, where it has cost nothing (ml/CLAUDE.md §0.3)
    gate_ref, gate_skip = gate_for_cadence(ax.cadence)
    gate_paths = [h for h in (a.heads or [])
                  if GATE_HEAD in os.path.basename(h)]
    if gate_ref is None and not a.export_mask_only and not a.dump_truth:
        # NOT a silent skip and NOT a monthly gate applied to a foreign axis.
        # `--no-gate` is deliberately not demanded here: it is documented
        # "smoke/toy ONLY", and making a real pentad eval wear that label to
        # get past a check it cannot take would mislabel the run in its own
        # artefact. The refusal to certify is RECORDED instead, in the log and
        # in `gate.reason`, where the harvest can read it.
        print(f"::warning::validation gate SKIPPED — {gate_skip}", flush=True)
        if gate_paths:
            print(f"::warning::{GATE_HEAD} was supplied but will NOT be "
                  f"compared against its monthly reference at {ax.cadence} "
                  f"cadence — it is scored like any other head", flush=True)
    elif not gate_paths and not a.no_gate and not a.export_mask_only \
            and not a.dump_truth:
        sys.exit(f"no {GATE_HEAD} head among --heads and --no-gate not set: "
                 f"the validation gate (plan §6.5) is what makes any spatial "
                 f"number here believable — add the gate head or pass "
                 f"--no-gate (smoke only)")
    heads = gate_paths + [h for h in (a.heads or [])
                          if h not in gate_paths]
    ck = torch.load(a.ckpt, map_location="cpu", weights_only=False)
    C = len(ck["chan"])
    d_z = ck["d_z"]
    hold_years = sorted(ck["args"]["holdout_years"].split(","))
    t_hold = np.array([m[:4] in set(hold_years) for m in months])
    lo, hi = (float(v) for v in ck["args"]["holdout_lon"].split(","))
    x_hold = (lons >= lo) & (lons < hi)

    if a.dump_truth:
        # The deseasonalised RAPID series, written by THE EVALUATOR that
        # defines it. `ml/plot_amoc_roll.py` needs the same truth the rolls
        # are scored against, and the alternative — re-deseasonalising in the
        # plot script — would be a second copy of the rule (train-month
        # climatology, holdout years excluded from the mean), which is the
        # defect that put a cosine on the status page's continuation charts.
        # Needs no GPU, no Z and no heads, so it runs in seconds anywhere the
        # small npz and a codec checkpoint exist.
        rp = d["rapid"]
        ri = rp[:, 0].astype(int)
        rv_ = rp[:, 1].copy()
        rmoy_ = moy[ri]
        tr_ = ~t_hold[ri]
        rclim_ = deseason_truth(rv_, rmoy_, tr_, "--dump-truth")
        des = rv_ - rclim_[rmoy_]
        os.makedirs(os.path.dirname(a.dump_truth) or ".", exist_ok=True)
        with open(a.dump_truth, "w") as fh:
            json.dump({"ym": [ax.label_of_row(i) for i in ri],
                       "sv_des": [round(float(v), 4) for v in des],
                       "trained": [bool(v) for v in tr_],
                       "hold_years": hold_years,
                       "clim_note": "deseasonalised with the TRAIN-month "
                                    "climatology (holdout years excluded from "
                                    "the monthly means), the same rule the "
                                    "rolls are scored against"}, fh)
        print(f"wrote {a.dump_truth}: {len(des)} RAPID months, "
              f"holdout {','.join(hold_years)}")
        return

    Xm = np.load(a.x, mmap_mode="r")
    Hg, Wg = Xm.shape[1], Xm.shape[2]
    om_path = os.path.join(a.cache_dir, "ocean_mask.npy")
    ocean = np.load(om_path) if os.path.exists(om_path) else None
    if ocean is None:
        ocean = np.zeros((Hg, Wg), bool)
        for t0 in range(0, T, 16):
            ocean |= np.isfinite(np.asarray(Xm[t0:t0 + 16, :, :, 0])).any(0)
        os.makedirs(a.cache_dir, exist_ok=True)
        np.save(om_path, ocean)
    ys, xs = np.where(ocean)
    P = len(ys)
    coords = np.stack([lats[ys] / 90, lons[xs] / 180], 1).astype(np.float32)
    sec_y, sec_sel = rapid_section(lats, lons, ys, xs)
    S_sec = len(sec_sel)
    print(f"window: {P} ocean px · section {S_sec} px at row {sec_y}",
          flush=True)

    Zm = None
    if not a.export_mask_only:
        Zm = np.load(a.z, mmap_mode="r")
        assert Zm.shape == (T, P, d_z), \
            f"Z {Zm.shape} != {(T, P, d_z)} — ordering mismatch, refusing"

    # ---- anomaly stats + per-month standardized fields --------------------
    st_path = os.path.join(a.cache_dir, "std_stats.npz")
    if os.path.exists(st_path):
        s_ = np.load(st_path)
        clim, dyn = s_["clim"], list(s_["dyn"])
        mean_c, std_c = s_["mean_c"], s_["std_c"]
    else:
        clim, dyn, mean_c, std_c = stream_stats(Xm, moy, t_hold, x_hold)
        np.savez(st_path, clim=clim, dyn=np.array(dyn),
                 mean_c=mean_c, std_c=std_c)
    std_m = StdMonths(Xm, ys, xs, moy, clim, dyn, mean_c, std_c)

    r1 = None
    if not a.export_mask_only:     # the mask needs no baseline arithmetic
        print("AR1 damped-persistence pass over the record ...", flush=True)
        r1 = ar1_train(std_m, T, t_hold, P, C)                 # [P, C]

    # ---- the three scopes -------------------------------------------------
    corridor, cor_thr = corridor_pixels(Xm, ocean, ys, xs, t_hold, sec_sel,
                                        a.corridor_pctl, a.corridor_dilate)
    rng = np.random.default_rng(0)               # rollout.py's exact subset
    keep = np.union1d(rng.choice(P, min(a.pixels_gate, P), replace=False),
                      sec_sel)
    gate_mask = np.zeros(P, bool)
    gate_mask[keep] = True
    sec_mask = np.zeros(P, bool)
    sec_mask[sec_sel] = True
    base_scopes = (("gate", gate_mask), ("corridor", corridor),
                   ("window", np.ones(P, bool)))

    # ---- the training longitude block, split into every scope -------------
    # x_hold was recomputed above only to reproduce the z-score statistics,
    # and then thrown away -- while gate/corridor/window were scored over
    # every pixel, INCLUDING the quarter of them that neither training stage
    # ever saw (train.py's stage-1 pool is `obs_any & ~t_hold & ~x_hold`;
    # temporal.py's stage-2 pool is `ok_p = ~x_hold[xs]`). Nothing was
    # computed wrongly and nothing was inflated -- an untrained pixel scores
    # LOW, so the blend deflates every headline number -- but the artefact
    # recorded no way to tell, and a figure drawn from roll_*.json could not
    # know the block existed. That is the defect: reporting, not arithmetic,
    # and the guard belongs where the inputs are all it has cost (ml/CLAUDE.md
    # 0.3, 5.16) -- i.e. here, in the writer, not in the reader.
    #
    # The split itself is recon_eval.py:298-302 / recon_decoder.py:279-283
    # verbatim in spirit (`px_hold = x_hold[xs...]`, then sel_train_x /
    # sel_hold_x). Those two evaluators take index arrays because they index
    # a [T,P,C] block; accumulate() here takes a boolean mask over P, so the
    # same partition is carried as booleans. Same rule, same names.
    px_hold = x_hold[xs]                          # [P]
    sel_train_x = ~px_hold
    sel_hold_x = px_hold
    # Scoring cost: the two children PARTITION their parent, so the masked
    # pixel work per horizon doubles (parent + its two halves), it does not
    # triple. Cheap next to a roll step.
    scopes = tuple(sc for name, m_ in base_scopes
                   for sc in ((name, m_),
                              (name + "_trainlon", m_ & sel_train_x),
                              (name + "_holdlon", m_ & sel_hold_x)))
    corridor_def = {"pctl": a.corridor_pctl, "threshold": round(cor_thr, 4),
                    "dilate_cells": a.corridor_dilate,
                    "structuring": "3x3 square",
                    "n_px": int(corridor.sum()), "of": P,
                    "union_section": True}
    # `any` is the field a reader should branch on, and the note follows it.
    # A no-longitude-holdout codec saves `holdout_lon "0,0"` (the empty
    # half-open interval, in a spelling all twelve float()-parsers accept --
    # ml/recipes/*-nolonhold.json, E-043), and under it the block below is an
    # EMPTY set: every _holdlon child has zero pixels, every _trainlon child
    # equals its parent exactly. Printing "-45..-25 was held out of training"
    # under that regime is CLAUDE.md 5.24's stale reference table -- it gets
    # checked, it matches nothing, and the run takes the blame for the
    # document. So the note says which of the two worlds this artefact is in.
    _lon_any = bool(x_hold.any())
    holdout_lon = {
        "arg": str(ck["args"]["holdout_lon"]),
        "lo": lo, "hi": hi,
        "any": _lon_any,
        "rule": "(lons >= lo) & (lons < hi), train.py's own expression",
        "n_cols": int(x_hold.sum()), "of_cols": int(len(lons)),
        "excluded_from": (["stage-1 pixel MAE (train.py: obs_any & ~t_hold "
                           "& ~x_hold)",
                           "stage-2 temporal head (temporal.py: "
                           "ok_p = ~x_hold[xs])"] if _lon_any else []),
        "px": {name: {"in_block": int((m_ & px_hold).sum()),
                      "of": int(m_.sum())}
               for name, m_ in base_scopes + (("section", sec_mask),)},
        "note": ("these longitude columns are held out of TRAINING in both "
                 "stages, so each scope aggregate here is a blend of trained "
                 "and never-trained pixels; the *_trainlon / *_holdlon "
                 "splits beside every scope are the unblended numbers, and "
                 "*_trainlon is the one that answers 'what did the model "
                 "learn'. The blend is deflationary -- an untrained pixel "
                 "scores low -- so a parent aggregate is a lower bound on "
                 "its own _trainlon."
                 if _lon_any else
                 "NO longitude is held out of training by this codec "
                 f"(holdout_lon {str(ck['args']['holdout_lon'])!r} is the "
                 "empty interval), so every scope aggregate is already the "
                 "trained-pixel number. The *_holdlon children are present "
                 "and EMPTY by construction -- n_px 0, no rows, no "
                 "horizon_auc -- and the *_trainlon children are their "
                 "parents exactly. Read the parent."),
    }
    for k, v in holdout_lon["px"].items():
        v["frac"] = round(v["in_block"] / v["of"], 4) if v["of"] else None
    print(f"scopes: gate {int(gate_mask.sum())} px \u00b7 corridor "
          f"{int(corridor.sum())} px (cur_speed \u2265 p{a.corridor_pctl:g} "
          f"= {cor_thr:.3f}, dilate {a.corridor_dilate}) \u00b7 window {P} px",
          flush=True)
    if _lon_any:
        print("held-out lon block [%g, %g): %d of %d columns \u00b7 " % (
                  lo, hi, holdout_lon["n_cols"], holdout_lon["of_cols"])
              + " \u00b7 ".join(
                  "%s %d/%d (%.1f%%)" % (k, v["in_block"], v["of"],
                                         100 * (v["frac"] or 0))
                  for k, v in holdout_lon["px"].items())
              + " NEVER TRAINED - scored as <scope>_holdlon", flush=True)
    else:
        # train.py prints the same sentence at the same fork, and for the
        # same reason: "0/481 cols" is true and reads like a bug.
        print("NO lon holdout - all %d cols trained (codec holdout_lon %r) "
              "\u00b7 every <scope>_holdlon is EMPTY by construction and "
              "every <scope>_trainlon equals its parent"
              % (holdout_lon["of_cols"], holdout_lon["arg"]), flush=True)

    if a.export_mask:
        export_mask(a.export_mask, lats, lons, ocean, ys, xs, corridor,
                    gate_mask, sec_sel, sec_y, corridor_def, months,
                    os.path.basename(a.x), holdout_lon)
        if a.export_mask_only:
            return

    # ---- codec + static identity for ALL pixels ---------------------------
    codec = codec_from_ckpt(ck, C)
    codec.load_state_dict(ck["model"])
    codec.eval().to(dev)
    x0 = np.asarray(Xm[0]).astype(np.float32)
    obs0 = np.isfinite(x0)
    for c in dyn:
        x0[..., c] = ((x0[..., c] - clim[moy[0], :, :, c]
                       - mean_c[c]) / (std_c[c] + 1e-6))
    Xt0 = torch.from_numpy(np.where(obs0, x0, 0.0)[None])      # [1,H,W,C]
    stat_obs = torch.from_numpy(obs0).clone()
    for c in dyn:
        stat_obs[..., c] = False
    zs = []
    with torch.no_grad():
        for i in range(0, P, 8192):
            sl = slice(i, min(i + 8192, P))
            n = sl.stop - sl.start
            ctx = np.concatenate([np.zeros((n, 2), np.float32),
                                  coords[sl]], 1)
            if getattr(codec, "patch", 1) > 1:
                vv, oo = gather_px(Xt0, stat_obs[None],
                                   torch.zeros(n, dtype=torch.long),
                                   torch.as_tensor(ys[sl]),
                                   torch.as_tensor(xs[sl]), codec.patch)
            else:
                vv = Xt0[0, ys[sl], xs[sl]]
                oo = stat_obs[ys[sl], xs[sl]]
            zs.append(codec.encode(
                vv.to(dev), oo.to(dev),
                torch.zeros(n, C, dtype=torch.bool, device=dev),
                torch.as_tensor(ctx).to(dev)).cpu().numpy())
    Zstat = np.concatenate(zs, 0)
    print(f"static identity encoded for {P} px", flush=True)

    # ---- verify the Z cache against a live re-encode (project_amoc's
    # guard: a silent ordering mismatch would roll beautiful nonsense) ------
    Zsec = np.asarray(Zm[:, sec_sel]).astype(np.float32)       # [T,S,dz]
    rows3 = [sec_y - 1, sec_y, sec_y + 1]
    slab, obs_sl = build_slab(Xm, rows3, moy, clim, dyn, mean_c, std_c)
    slab_t = torch.from_numpy(np.nan_to_num(slab, nan=0.0))
    obs_t = torch.from_numpy(obs_sl)
    ctx_all = np.stack([np.sin(2 * np.pi * moy / 12),
                        np.cos(2 * np.pi * moy / 12)], 1)
    rngv = np.random.default_rng(1)
    kv = rngv.choice(S_sec, min(8, S_sec), replace=False)
    sxs = xs[sec_sel]
    Zl, _ = embed_everything(codec, slab_t, obs_t, ctx_all, lats[rows3], lons,
                             np.ones(len(kv), dtype=int), sxs[kv], d_z,
                             cache_path=None, batch=64)
    for tt in (0, T // 2, T - 1):
        dmax = float(np.abs(Zl[tt] - Zsec[tt][kv]).max())
        zscale = float(np.abs(Zl[tt]).max())
        assert dmax < max(0.02, 0.005 * zscale), \
            f"Z mismatch at t={tt}: {dmax} vs scale {zscale}"
    print("Z cache verified vs live re-encode ✓", flush=True)

    # ---- transport read-out (truefit protocol, rollout.py verbatim) ------
    rapid = d["rapid"]
    ridx = rapid[:, 0].astype(int)
    rv = rapid[:, 1].copy()
    rmoy = moy[ridx]
    tr_all = ~t_hold[ridx]
    rclim = deseason_truth(rv, rmoy, tr_all, "RAPID truth")
    rv_des = rv - rclim[rmoy]
    Fsec_true = Zsec.mean(1)
    mu_p, sd_p, w_probe, probe_val_r = fit_ridge(
        Fsec_true[ridx][tr_all], rv_des[tr_all])
    # KEYED ON THE AXIS ROW, which is what `rapid[:, 0]` already holds in
    # every family (build_family3.py:311 writes month-indices; build_family4's
    # truth_pentad() writes bin-row indices — ml/probe_kfold.py:276 is the
    # precedent). The old `YYYYMM` key was identical to this at monthly and
    # discarded 83.6% of the pentad series, because six pentad rows share one
    # `YYYY-MM` label and the dict kept the last.
    r_of_row = {int(mi): i for i, mi in enumerate(ridx)}
    print(f"probe fit: {int(tr_all.sum())} train labels, "
          f"val-tail r {probe_val_r:+.3f}", flush=True)
    sec_t = torch.as_tensor(np.asarray(sec_sel), device=dev)

    def read_sv(zhat):
        fr = zhat[sec_t].mean(0).cpu().numpy()
        return float(np.dot(np.r_[(fr - mu_p) / sd_p, 1.0], w_probe))

    def zwin_from_true(s_end, K):
        arr = np.asarray(Zm[s_end - K + 1: s_end + 1])         # [K,P,dz] f16
        return (torch.from_numpy(np.ascontiguousarray(
            arr.transpose(1, 0, 2))).to(dev).float())          # [P,K,dz]

    # ---- per-stencil geometry, built once, shared across heads ------------
    nbr_cache, sctx_cache = {}, {}

    def geometry(stencil, ring_km=0.0):
        # keyed on the FULL geometry, not just the slot count: E-023's ring
        # has the same 9 slots as E-022's 3x3 and a completely different
        # neighbour set, so caching on `stencil` alone would silently score
        # one arm with the other's geometry.
        key = (stencil, str(ring_km))
        if key not in nbr_cache:
            if stencil == 1:
                nbr_cache[key] = None
                sctx_cache[key] = torch.from_numpy(
                    np.concatenate([Zstat, coords], 1)).to(dev)
            else:
                NBR = build_stencil(Hg, Wg, ys, xs, stencil,
                                    ring_km=ring_km, lats=lats)
                nbr_cache[key] = torch.as_tensor(NBR, device=dev)
                sctx_cache[key] = torch.from_numpy(np.concatenate(
                    [Zstat, coords,
                     (NBR >= 0).astype(np.float32)], 1)).to(dev)
        return nbr_cache[key], sctx_cache[key]

    # 20 years of the AXIS, not 240 of whatever a step happens to be.
    if a.long_months < 0:
        a.long_months = int(round(20 * ax.steps_per_year))
    if a.future_months < 0:
        a.future_months = int(round(20 * ax.steps_per_year))
    # the house 18-MONTH lowpass, in steps of THIS axis (18 at monthly)
    lp_k = ax.steps_for_months(18)
    lp_min = max(2, int(round(lp_k * 12 / 18)))

    prog = Progress(a.metrics, len(heads), every=a.progress_every)
    results = {"data": os.path.basename(a.x), "horizon": a.horizon,
               "hold_years": hold_years, "holdout_lon": holdout_lon,
               "corridor_def": corridor_def,
               "gate_ref": dict(GATE_REF, head=GATE_HEAD, tol=GATE_TOL),
               "probe": {"val_tail_r": round(float(probe_val_r), 3)},
               "gate": {"pass": None, "skipped": True},   # overwritten below
               "heads": {}}
    # ALMOST NOTHING IS ADDED TO A MONTHLY ARTEFACT. Every published corridor
    # AUC was written by this code on the monthly axis, and a roll json that
    # gains a key is a roll json that no longer diffs byte-for-byte against
    # the archive — which is the one property
    # tests/test_roll_monthly_identity.py exists to hold. Monthly is the
    # archive's default and reads as before; every OTHER cadence says what it
    # is, in the artefact, where a harvest can see it.
    #
    # There is now EXACTLY ONE exception, and it is deliberate:
    # `horizon_auc_daymatched` inside each scope's skill block. It is emitted
    # at every cadence because a key that exists only where the reader is
    # least likely to look is a key the harvest will forget to ask for; at
    # monthly it is `horizon_auc` recomputed over the same leads, i.e. a NEW
    # key carrying an OLD value, which cannot make anything incomparable. The
    # bit-identity test strips it, counts what it stripped, and asserts each
    # stripped value EQUALS that scope's `horizon_auc` before demanding the
    # remainder be byte-identical.
    bands = ax.bands()
    leads = ax.daymatched_leads()
    if not ax.monthly:
        results["cadence"] = {
            "name": ax.cadence, "step_days": ax.days,
            "steps_per_year": round(ax.steps_per_year, 4),
            "T": ax.T, "first": ax.labels[0], "last": ax.labels[-1],
            "detected_from": ax.detected_from,
            "horizon_steps": a.horizon,
            "horizon_span_days": ax.span_days(a.horizon),
            "horizon_daymatched_steps": ax.steps_for_months(12),
            "horizon_is_daymatched":
                a.horizon == ax.steps_for_months(12),
            "starts_per_year": a.starts_per_year or "all",
            "starts_per_holdout_year": {
                str(Y): len(ax.starts_for_year(Y, a.starts_per_year))
                for Y in hold_years},
            "starts_available_per_holdout_year": {
                str(Y): len(ax.starts_for_year(Y)) for Y in hold_years},
            "long_steps": a.long_months,
            "long_span_days": ax.span_days(a.long_months),
            "future_steps": a.future_months,
            "future_span_days": ax.span_days(a.future_months),
            "lowpass_steps": lp_k, "lowpass_months": 18,
            "map_h_span_days": ax.span_days(a.map_h),
            "band_edge_days": list(BAND_EDGE_DAYS),
            "daymatched_leads": list(leads),
            "daymatched_lead_days": [ax.span_days(h) for h in leads],
            "note": ("every `h` in this file is an AXIS STEP, not a month. "
                     "The horizon-band keys carry their own day spans for "
                     "the same reason. `horizon_auc` averages msss_clim over "
                     "h=1..horizon_steps and is therefore a function of THIS "
                     "axis's lead sampling; `horizon_auc_daymatched` averages "
                     "the twelve leads above, which are the monthly "
                     "archive's, and is the ONLY one of the two that may be "
                     "compared against an archived corridor AUC."),
        }
        results["gate_ref"] = {"head": GATE_HEAD, "tol": GATE_TOL,
                               "cadence": ax.cadence, "reference": None,
                               "monthly_reference": dict(GATE_REF),
                               "reason": gate_skip}
        results["gate"] = {"pass": None, "skipped": True, "certified": False,
                           "cadence": ax.cadence, "reason": gate_skip}
    # WHICH ROWS WERE SCORED, not merely how many. A count says the knob was
    # read; the rows say what the number is a number OF, and a harvest that
    # wants to know whether two rolls sampled the same seasonal phases has no
    # other way to find out. Written only when the knob was actually used —
    # absent it, the starts are the protocol's own full list and a monthly
    # artefact gains nothing (see the note above).
    if a.starts_per_year > 0:
        results["starts"] = {
            "per_year": a.starts_per_year,
            "rule": ("every k-th start of the holdout year's list, "
                     "k = len(list)//N, first N — deterministic, keeps the "
                     "start whose h=1 is the year's first row, and spreads "
                     "the rest evenly round the seasonal cycle"),
            "available": {str(Y): len(ax.starts_for_year(Y))
                          for Y in hold_years},
            "rows": {str(Y): ax.starts_for_year(Y, a.starts_per_year)
                     for Y in hold_years},
            "labels": {str(Y): [ax.label_of_row(s) for s in
                                ax.starts_for_year(Y, a.starts_per_year)]
                       for Y in hold_years},
        }
        print("starts: %d per holdout year — %s" % (
            a.starts_per_year, "; ".join(
                f"{Y} rows {results['starts']['rows'][str(Y)]} "
                f"({', '.join(results['starts']['labels'][str(Y)])})"
                for Y in hold_years)), flush=True)

    # ---- the roll-forward sequence dump (opt-in) --------------------------
    # Built here, where the pixel map and the codec identity are both in
    # scope and before a single roll step has been spent, so its cost is
    # PRINTED rather than discovered (ml/CLAUDE.md §0.3). `dump is None` is
    # the untouched path in every branch below.
    dump = None
    if a.dump_roll:
        dump = RollDump(a.dump_roll, ax, ys, xs, lats, lons, ocean.shape,
                        a.ckpt, ck,
                        {"horizon": a.horizon,
                         "horizon_span_days": ax.span_days(a.horizon),
                         "starts_per_year": a.starts_per_year or "all",
                         "hold_years": hold_years,
                         "gate_head_excluded": GATE_HEAD})
        dump.plan([h for h in heads if GATE_HEAD not in os.path.basename(h)],
                  {Y: len(ax.starts_for_year(Y, a.starts_per_year))
                   for Y in hold_years}, a.horizon, P, d_z)
    K_seen = None

    def mark(stage, label=None, head_i=None):
        """The `in_progress` value for a partial write — WHERE this roll is."""
        p = {}
        if label is not None:
            p["head"] = label
        if head_i is not None:
            p["head_i"] = head_i
        p["heads"] = len(heads)
        p["stage"] = stage
        p["at"] = _utc_now()
        return p

    # THE FILE EXISTS FROM MINUTE ONE. Everything above this line is setup the
    # roll's reader cannot see, so a skeleton goes out before the first head
    # rolls a single step: the live publisher then has something to ship on
    # its very first pass, and a job that dies in hour one is distinguishable
    # from one that never started.
    write_results(a.out, results, partial=mark("started"))
    print(f"wrote skeleton {a.out} (partial, stage=started, "
          f"{len(heads)} heads to roll)", flush=True)

    for hp in heads:
        t_head = time.time()
        tk = torch.load(hp, map_location="cpu", weights_only=False)
        ta = tk["args"]
        K = ta["K"]
        if K_seen is None:
            K_seen = K
            results["K"] = K
        elif K != K_seen:
            sys.exit(f"{hp} has K={K} != {K_seen} — windows not comparable")
        stencil = ta.get("stencil", 1)
        ring_km = ta.get("ring_km", 0) or 0        # number OR "222,555"
        unroll = ta.get("unroll", 1)
        label = (f"s{stencil}"
                 + (f"r{str(ring_km).replace(',', '-')}" if _ring_on(ring_km) else "")
                 + (f"u{unroll}" if unroll != 1 else "")
                 + f"_s{ta.get('seed', 0)}")
        if label in results["heads"]:
            label += "_" + os.path.basename(hp).replace(".pt", "")
        k_tbl = tk["model"]["pos.weight"].shape[0]  # the file, not a convention
        dir_ = tuple(int(x) for x in
                     str(ta.get("direct") or "").split(",") if x.strip())
        model = TemporalTransformer(d_z=d_z, d_model=ta["d_model"],
                                    n_heads=4, n_layers=ta["layers"],
                                    k_max=k_tbl, direct=dir_, stencil=stencil)
        model.load_state_dict(tk["model"])
        model.eval().to(dev)
        NBR_t, static_ctx = geometry(stencil, ring_km)
        print(f"head {label}: {os.path.basename(hp)} "
              f"(d_model={ta['d_model']}, layers={ta['layers']}, K={K}, "
              f"stencil={stencil}"
              + (f", ring {ring_km} km" if _ring_on(ring_km) else "") + ")",
              flush=True)

        Hh = a.horizon
        # planned roll steps for THIS head, computed from the protocol rather
        # than guessed, and from the AXIS rather than from the calendar: a
        # start at the last row before Y rolls min(H, |Y|) steps inside Y, the
        # next start one fewer near the end ... so at monthly with H=12 a
        # holdout year costs 78 steps and at pentad with 73 starts it costs
        # 73*12 - 66 = 810; then the long hindcast and the future roll are one
        # step per AXIS STEP each.
        n_skill = 0
        for Y in hold_years:
            for s_ in ax.starts_for_year(Y, a.starts_per_year):
                if s_ - K + 1 < 0 or s_ + 1 >= T:
                    continue
                n_skill += scored_horizon(ax, s_, Hh, T, Y)
        # One hindcast per `--long-start` label that resolves to a row. With
        # the single default this is exactly `a.long_months if the label
        # resolves else 0`, which is what it has always been.
        n_long = a.long_months * sum(
            ax.row_of_label(s.strip()) is not None
            for s in str(a.long_start or "").split(",") if s.strip())
        head_i = list(heads).index(hp) + 1
        prog.start_head(head_i, label,
                        n_skill + n_long + a.future_months)
        print(f"  {label}: {n_skill} scored roll steps + {n_long} hindcast + "
              f"{a.future_months} future = "
              f"{n_skill + n_long + a.future_months} steps over {P:,} pixels",
              flush=True)
        sums = {name: new_sums(Hh) for name, _ in scopes}
        # E-026b AUDIT instrumentation (2026-08-15, Chris: "investigate
        # thoroughly and not hypothesize — there could still be an issue in
        # how we compute auc"). Two decompositions ride along with EVERY
        # eval from now on — always-on because the workflow's 25-input
        # ceiling makes a flag expensive and the cost is a few hundred KB:
        #   per-CHANNEL msss curves on the corridor scope (the aggregate
        #   could hide a channel subset driving the reach divergence), and
        #   a per-PIXEL msss map at h=--map-h on the FULL window (where
        #   does the far-reach decay live — near pixels whose stencils
        #   reach off-window, or everywhere). msss_clim = 1 - mse_m/mse_c,
        #   so only the two sums are kept; n cancels. An exact identity
        #   guards the instrumentation: the corridor aggregate recomposed
        #   from the channel sums must match skill_block's own number.
        aud = {"ch_m": np.zeros((Hh + 1, C)), "ch_c": np.zeros((Hh + 1, C)),
               "px_m": np.zeros(P), "px_c": np.zeros(P)}
        probe_pts = {h: [] for h in range(1, Hh + 1)}
        # THE GATE HEAD IS NEVER DUMPED (see RollDump): it is the run's
        # certificate, it is re-rolled in every eval wave, and nobody
        # animates it.
        dump_head = dump is not None and GATE_HEAD not in os.path.basename(hp)
        if dump is not None and not dump_head:
            print(f"  {label}: --dump-roll skips the gate head", flush=True)
        with torch.no_grad():
            for Y in hold_years:
                for s in ax.starts_for_year(Y, a.starts_per_year):
                    start_m = ax.label_of_row(s)
                    if s - K + 1 < 0 or s + 1 >= T:
                        continue
                    Zwin = zwin_from_true(s, K)
                    cur = list(moy[s - K + 1: s + 1])
                    v_pers, obs_s = std_m.get(s)
                    # The trajectory buffer, sized from the SAME break
                    # condition the h-loop below uses, so its last slot is
                    # the last state that loop produces — never a zero row
                    # that would animate as a dead ocean. State 0 is the
                    # TRUE embedding of the start row: Zwin's last slot,
                    # read from the cache in its own float16.
                    n_dump = (scored_horizon(ax, s, Hh, T, Y)
                              if dump_head else 0)
                    traj = (np.empty((n_dump + 1, P, d_z), np.float16)
                            if n_dump else None)
                    if traj is not None:
                        traj[0] = np.asarray(Zm[s])
                    for h in range(1, Hh + 1):
                        t_tgt = s + h
                        if t_tgt >= T or ax.year[t_tgt] != int(Y):
                            break
                        # month features are the CURRENT window's months —
                        # rollout.py's mseq at step h spans s-K+h .. s+h-1;
                        # advance AFTER the forward, like project_amoc.py
                        zhat = roll_step(model, Zwin, NBR_t, static_ctx,
                                         month_feats(cur, dev), a.chunk,
                                         a.amp)
                        if traj is not None:
                            # ẑ for THIS row, before it is folded into the
                            # window — the same tensor the scoring decodes,
                            # so the animation and the numbers cannot be of
                            # two different rolls.
                            traj[h] = zhat.detach().to(torch.float16) \
                                          .cpu().numpy()
                        Zwin = torch.cat([Zwin[:, 1:], zhat[:, None]], 1)
                        # the season token of the row JUST PREDICTED, read off
                        # its own date. `(cur[-1] + 1) % 12` advanced a MONTH
                        # per step, which is right exactly when a step is a
                        # month and corrupts the model's own input otherwise.
                        cur = cur[1:] + [ax.moy_of_row(t_tgt)]
                        xhat = decode_all(codec, zhat, C, a.chunk, a.amp)
                        v_true, obs_tt = std_m.get(t_tgt)
                        op = obs_tt & obs_s
                        v_damp = v_pers * r1 ** h
                        prog.step("skill")
                        for name, m_ in scopes:
                            accumulate(sums[name], h, xhat[m_], v_true[m_],
                                       v_pers[m_], v_damp[m_], op[m_])
                        of_ = op.astype(np.float64)
                        err_ = ((xhat - v_true) ** 2) * of_
                        tru_ = (v_true ** 2) * of_
                        aud["ch_m"][h] += err_[corridor].sum(axis=0)
                        aud["ch_c"][h] += tru_[corridor].sum(axis=0)
                        if h == a.map_h:
                            aud["px_m"] += err_.sum(axis=1)
                            aud["px_c"] += tru_.sum(axis=1)
                        if t_tgt in r_of_row:
                            probe_pts[h].append(
                                (s, read_sv(zhat), rv_des[r_of_row[t_tgt]]))
                    print(f"  {label} start {start_m}: rolled", flush=True)
                    if traj is not None:
                        dump.write(label, os.path.basename(hp),
                                   {"stencil": stencil, "ring_km": ring_km,
                                    "seed": ta.get("seed", 0),
                                    "unroll": unroll, "K": K,
                                    "d_model": ta["d_model"],
                                    "layers": ta["layers"]},
                                   Y, s, traj)
                        traj = None

        entry = {"meta": {"file": os.path.basename(hp), "stencil": stencil,
                          "ring_km": ring_km, "seed": ta.get("seed", 0),
                          "unroll": unroll}}
        for name, m_ in scopes:
            entry[name] = skill_block(sums[name], Hh, n_px=int(m_.sum()),
                                      leads=leads)
        # --- E-026b audit block + exact-identity check -------------------
        ch_rows, max_dev = [], 0.0
        cor_rows = {r["h"]: r["msss_clim"]
                    for r in entry["corridor"]["chan_skill"]}
        for h in range(1, Hh + 1):
            cm, cc = aud["ch_m"][h], aud["ch_c"][h]
            ch_rows.append([round(float(1 - m / c), 3) if c > 0 else None
                            for m, c in zip(cm, cc)])
            if cc.sum() > 0 and h in cor_rows:
                agg = 1 - cm.sum() / cc.sum()
                max_dev = max(max_dev, abs(agg - cor_rows[h]))
        if max_dev > 2e-3:
            # instrumentation disagrees with the metric it decomposes —
            # the audit block is untrustworthy; say so IN the artefact
            # (a print alone would scroll away) but do not kill a
            # multi-hour eval whose aggregate numbers are still sound
            print(f"::error::audit identity FAILED: per-channel recompose "
                  f"deviates {max_dev:.4f} from corridor msss", flush=True)
        entry["audit"] = {
            "channels": [str(c) for c in ck["chan"]],
            "per_channel_msss_clim_corridor": ch_rows,
            "identity_max_dev": round(float(max_dev), 5),
            "map_h": a.map_h,
            "note": ("map pixel order = the eval's ys/xs arrays "
                     "(same ordering --export-mask writes)"),
            "map_msss_clim_window": [
                round(float(1 - m / c), 2) if c > 0 else None
                for m, c in zip(aud["px_m"], aud["px_c"])],
        }
        entry["amoc_bands"] = {}
        for bn, hs in bands:
            # the KEY carries the unit at any non-monthly cadence: `h1-3` is
            # 1-3 months on the monthly axis and `h1-18_5-90d` spans the same
            # 90 days at pentad, and a band that does not say which is a
            # number waiting to be misread.
            key = ax.band_key(bn, hs)
            pts = [(h, s_, pr, y) for h in hs if h <= Hh
                   for (s_, pr, y) in probe_pts.get(h, [])]
            if len(pts) >= 8:
                pr = np.array([p[2] for p in pts])
                tv = np.array([p[3] for p in pts])
                r_ = corr_or_none(pr, tv)
                entry["amoc_bands"][key] = (
                    {"r": r_, "n": len(pts)} if r_ is not None else
                    {"n": len(pts),
                     "undefined": ("r is not defined for this band — one of "
                                   "the two series has zero variance over "
                                   "its %d points; omitted rather than "
                                   "written as NaN (ml/CLAUDE.md §5.22)"
                                   % len(pts))})
        if not ax.monthly:
            entry["amoc_bands_def"] = {
                ax.band_key(bn, hs): {
                    "steps": [h for h in hs if h <= Hh],
                    "step_days": ax.days,
                    "span_days": [ax.span_days(min(hs)),
                                  ax.span_days(max(hs))]}
                for bn, hs in bands}
        for name, _ in scopes:
            if entry[name].get("chan_skill"):
                dm = entry[name].get("horizon_auc_daymatched")
                print(f"  {label} {name}: AUC(clim) "
                      f"{entry[name]['horizon_auc']:+.3f} · damped "
                      f"{entry[name]['auc_damped']:+.3f} · amp h{Hh} "
                      f"{entry[name]['chan_skill'][-1]['amp_ratio']:.3f}"
                      + ("" if dm is None else
                         f" · day-matched {dm:+.3f}"
                         + ("" if ax.monthly else
                            f" (leads {list(leads)})")),
                      flush=True)
        print(f"  {label} amoc: " + " ".join(
            (f"{bn} {v['r']:+.3f}(n={v['n']})" if v.get("r") is not None
             else f"{bn} UNDEFINED(n={v['n']})")
            for bn, v in entry["amoc_bands"].items()), flush=True)

        # ---- VALIDATION GATE (fatal, before any spatial head is scored) --
        if gate_ref is not None and hp in gate_paths and hp == gate_paths[0]:
            got = {"auc": entry["gate"].get("horizon_auc"),
                   "bands": {bn: entry["amoc_bands"].get(bn, {}).get("r")
                             for bn, _ in bands}}
            fails = []
            if got["auc"] is None or abs(got["auc"] - gate_ref["auc"]) > GATE_TOL:
                fails.append(f"AUC {got['auc']} vs {gate_ref['auc']}")
            for bn, ref in gate_ref["bands"].items():
                gv = got["bands"].get(bn)
                if gv is None or abs(gv - ref) > GATE_TOL:
                    fails.append(f"{bn} {gv} vs {ref}")
            results["gate"] = {"head": label, "got": got,
                               "pass": not fails, "fails": fails}
            if fails:
                json.dump(results, open(a.out + ".gatefail", "w"), indent=1)
                sys.exit("VALIDATION GATE FAILED — the evaluator disagrees "
                         "with #217 on the stencil-1 baseline, so no spatial "
                         "number it produces can be trusted (plan §6.5). "
                         "Mismatches: " + "; ".join(fails)
                         + f" — partial results in {a.out}.gatefail")
            print(f"VALIDATION GATE PASSED: {got}", flush=True)

        # THE SCORED NUMBERS ARE OUT OF THE BOX HERE, not in eleven hours.
        # `entry` goes into `results` NOW and is mutated in place afterwards,
        # so every later write in this head picks the same object up with more
        # in it. At pentad the long+future phases below are 87% of this job's
        # wall clock (2,922 of 3,363 steps) and everyone downstream is waiting
        # on the block above them (ml/CLAUDE.md §5.25).
        results["heads"][label] = entry
        write_results(a.out, results, partial=mark("scored", label, head_i))
        print(f"  {label} scored → {a.out} (partial, stage=scored)",
              flush=True)

        # ---- long hindcast + future roll (median trajectory only) --------
        def long_roll(s_end, n_steps, phase="long"):
            """Roll `n_steps` AXIS STEPS past row `s_end`.

            The old body advanced a calendar month and a month-of-year token
            once per step. At monthly that is the axis; at pentad it ran the
            model's own seasonal input a full simulated year forward every 60
            real days, and then attached truth on the wrong dates — a
            corrupted INPUT, not merely a mislabelled output. Both now come
            from the ROW: one step is one row, and the season token is that
            row's true month.
            """
            Zwin = zwin_from_true(s_end, K)
            cur = list(moy[s_end - K + 1: s_end + 1])
            sv, roll_rows = [], []
            with torch.no_grad():
                for i in range(n_steps):
                    r_next = s_end + 1 + i
                    zhat = roll_step(model, Zwin, NBR_t, static_ctx,
                                     month_feats(cur, dev), a.chunk, a.amp)
                    Zwin = torch.cat([Zwin[:, 1:], zhat[:, None]], 1)
                    cur = cur[1:] + [ax.moy_of_row(r_next)]
                    prog.step(phase)
                    sv.append(read_sv(zhat))
                    roll_rows.append(r_next)
            return np.array(sv), roll_rows

        _long_specs = [s.strip() for s in str(a.long_start or "").split(",")
                       if s.strip()]

        def long_block(row, spec):
            """The hindcast block for ONE context end. Identical arithmetic to
            the single-start version it was factored out of (2026-08-22) —
            truth attaches on the AXIS ROW through `r_of_row`, `trained` is
            `t_hold` on that row's RAPID index, the lowpass is the same
            `smooth`, and the dict is built key for key in the same order, so
            a single-start run is byte-identical to before the list existed."""
            sv, roll_rows = long_roll(row, a.long_months)
            roll_ym = [ax.label_of_row(r) for r in roll_rows]
            truth = np.full(len(sv), np.nan)
            trained = np.zeros(len(sv), bool)
            for i, r_ in enumerate(roll_rows):
                if r_ in r_of_row:
                    truth[i] = rv_des[r_of_row[r_]]
                    trained[i] = not t_hold[ridx[r_of_row[r_]]]

            def _r(m_):
                if m_.sum() < 8:
                    return None, int(m_.sum())
                return corr_or_none(sv[m_], truth[m_]), int(m_.sum())
            fin = np.isfinite(truth)
            r_tr, n_tr = _r(fin & trained)
            r_ho, n_ho = _r(fin & ~trained)
            sv_lp = smooth(sv, lp_k, lp_min)
            tr_lp = smooth(np.where(fin, truth, np.nan), lp_k, lp_min)
            both = np.isfinite(sv_lp) & np.isfinite(tr_lp)
            r_lp = amp_lp = None
            if both.sum() >= ax.steps_for_months(24):
                r_lp = corr_or_none(sv_lp[both], tr_lp[both])
                amp_lp = round(float(sv_lp[both].std()
                                     / (tr_lp[both].std() + 1e-9)), 3)
            blk = {"context_end": ax.label_of_row(row),
                   "roll_ym": roll_ym,
                   "sv_des": [round(v, 3) for v in sv.tolist()],
                   "r_trained": r_tr, "n_trained": n_tr,
                   "r_heldout": r_ho, "n_heldout": n_ho,
                   "r_lp18": r_lp, "amp_lp18": amp_lp}
            print(f"  {label} long({spec}+{a.long_months}m): "
                  f"r_trained {r_tr} (n={n_tr}) · r_heldout {r_ho} "
                  f"(n={n_ho}) · lp18 r {r_lp} amp {amp_lp}", flush=True)
            return blk

        # THE FIRST SPEC IS TODAY'S ROLL, under today's key, with today's
        # arithmetic. The rest (2026-08-22) are the PHASE DISCRIMINATOR: every
        # head's unforced future roll mode-locks to the calendar — the
        # nolonhold pair peaks exactly 36 months apart, pinned to the same
        # months, phase-identical across seeds — and one context end cannot
        # tell "the model replays the calendar" from "the model's phase is
        # selected by its state". Rolling from SEVERAL ends can: if the peaks
        # stay on the same calendar months the phase is the calendar's, and if
        # they follow the context end it is the state's. They go in a NEW key
        # so no archived reader moves.
        multi = []
        for _i, _spec in enumerate(_long_specs):
            _row = ax.row_of_label(_spec)
            if _row is None:
                print(f"  {label} long({_spec}): SKIPPED — no axis row for "
                      f"that label (the record runs {ax.labels[0]}..."
                      f"{ax.labels[-1]})", flush=True)
                continue
            if _row - K + 1 < 0:
                print(f"  {label} long({_spec}): SKIPPED — row {_row} has "
                      f"only {_row + 1} rows of history and this head needs "
                      f"K={K} for its context window", flush=True)
                continue
            blk = long_block(_row, _spec)
            if "long" not in entry:
                entry["long"] = blk
            else:
                multi.append(blk)
            # `long_multi` is attached HERE rather than after the loop (where
            # it used to live) so a partial write carries it too. Same key in
            # the same place: the assignment first fires on the iteration that
            # first appends, and the list is mutated in place afterwards.
            if multi:
                entry["long_multi"] = multi
            write_results(a.out, results, partial=mark("long", label, head_i))
            print(f"  {label} long({_spec}) → {a.out} "
                  f"(partial, stage=long)", flush=True)

        if a.future_months > 0:
            sv, roll_rows = long_roll(T - 1, a.future_months, "future")
            roll_ym = [ax.label_of_row(r) for r in roll_rows]
            entry["future"] = {"context_end": ax.label_of_row(T - 1),
                               "roll_ym": roll_ym,
                               "sv_des": [round(v, 3) for v in sv.tolist()]}
            write_results(a.out, results,
                          partial=mark("future", label, head_i))
            print(f"  {label} future → {a.out} (partial, stage=future)",
                  flush=True)
        results["heads"][label] = entry
        # WRITE AFTER EVERY HEAD, not once at the end. This eval is hours of
        # rented GPU; a job_timeout with the file still in memory would spend
        # all of it and archive nothing (the same shape as a green run with no
        # temporal.json). Partial output also lets the harvest read the gate
        # head's numbers while the rest is still rolling.
        entry["wall_s"] = round(time.time() - t_head, 1)
        # STILL IN PROGRESS while heads remain. A file that drops the marker
        # after head 1 of 4 tells a reader the roll is finished, which is the
        # one thing `in_progress` exists to prevent; only the write at the end
        # of main() is allowed to be unmarked.
        write_results(a.out, results,
                      partial=(None if head_i >= len(heads)
                               else mark("head_done", label, head_i)))
        print(f"  {label} done in {entry['wall_s'] / 60:.1f} min — "
              f"{len(results['heads'])}/{len(heads)} heads written to {a.out}",
              flush=True)
        model.to("cpu")
        if dev.type == "cuda":
            torch.cuda.empty_cache()

    # THE ONLY UNMARKED WRITE. `in_progress` disappears exactly here, so its
    # absence is the roll's own statement that it reached the end — which is
    # what scripts/sroll_run.sh asserts before anything is published.
    write_results(a.out, results)
    print(f"wrote {a.out} ({os.path.getsize(a.out):,} bytes)", flush=True)


if __name__ == "__main__":
    main()

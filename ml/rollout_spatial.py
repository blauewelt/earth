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
import re
import sys
import time

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import probscore                                                # noqa: E402
from model import codec_from_ckpt, gather_px                    # noqa: E402
from recon_eval import stream_stats, build_slab                 # noqa: E402
# E-067's holdout BLOCKS live in ml/temporal.py, not here, and are re-exported
# under their old names so every importer (ml/lim_baseline.py, the tests) is
# unchanged. They moved because ml/embed_cache_sync.py needs the same grouping
# to name a cache and cannot import this module — rollout_spatial imports
# temporal, never the reverse. ONE definition, three consumers, the same
# arrangement `build_window_pool` has for the two trainers.
from temporal import hold_blocks, block_bounds, block_label  # noqa: E402,F401
from temporal import (TemporalTransformer, build_stencil,       # noqa: E402
                      embed_everything, rapid_section, _ring_on,
                      codec_weight_hash, season_feat_of,
                      InputQuant)
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

    def rows_in_block(self, block):
        """The rows of every year in `block`, in axis order. For a
        single-year block this IS `rows_in_year` — the years of a block are
        consecutive by construction (`hold_blocks`), so the union of their
        rows is the contiguous run between the block's ends."""
        y0, y1 = block_bounds(block)
        return np.where((self.year >= y0) & (self.year <= y1))[0]

    def starts_for_year(self, Y, per_year=0):
        """Staggered roll starts for holdout year Y — `starts_for_block`
        applied to the one-year block `(Y, Y)`, which is the whole
        implementation. ONE definition, two spellings: a second copy of the
        start rule is a second chance for the block path and the year path to
        disagree about which rows a roll was scored from."""
        return self.starts_for_block((Y, Y), per_year)

    def starts_for_block(self, block, per_year=0):
        """Staggered roll starts for a holdout BLOCK — one year, or a run of
        consecutive years (`hold_blocks`) — spanning the axis's REAL
        steps-per-year: the last row before the block (so h=1 lands on its
        first row) plus every row inside it except its last.

        At monthly a one-year block is exactly the old `for s_off in
        range(12)` list — Dec(Y-1), Jan … Nov — in the same order. At pentad
        it is 73 starts, not 12; a two-year block is 146.

        `per_year` (0 = all, and the default, i.e. everything above unchanged)
        SUBSAMPLES that list: every k-th start, k = len(list) // N, first N.
        THE NAME IS A COMPATIBILITY SPELLING AND THE COUNT IS PER BLOCK: N is
        how many starts this block contributes, whether the block is one year
        or three. `--starts-per-year 3` on a two-year block is therefore 3
        starts spread over 730 days, not 6 — which is what keeps the cost of a
        roll a function of the number of BLOCKS rather than of how they were
        grouped, and what makes a single-year run's start list unchanged.
        The starts are a COST, not a protocol constant — at pentad the roll
        pays 73 of them for 73 leads each — and the count is the free
        parameter of the E-044 horizon decision. The rule is a fixed stride
        rather than a random or an edge-weighted choice for three reasons:
        it is deterministic (a re-roll of the same head scores the same rows),
        it keeps the FIRST start, which is the one whose h=1 lands on the
        block's first row, and a constant stride spreads the starts evenly
        round the seasonal cycle — at pentad, N=3 gives k=24 on a one-year
        block and phases near 1 Jan / 1 May / 1 Sep, so lead time is not
        confounded with season.
        N >= len(list) (and N <= 0) return the full list untouched, which is
        what keeps the monthly path — list AND order — bit-identical."""
        rows = self.rows_in_block(block)
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


def gate_subset(P, n_pixels, sec_sel):
    """Bool [P]: rollout.py's EXACT gate subset — `default_rng(0).choice(P, n)`
    unioned with the RAPID section.

    Lifted verbatim out of `main()` (2026-09-02) so a second scorer can build
    the same scope without copying the seed, the union or the `min(n, P)`
    clamp. Two spellings of "the #217 subset" would be two chances to disagree
    about which pixels the gate compares, and the gate is the certificate that
    makes every other number in this file readable (§3, exception 1).
    """
    rng = np.random.default_rng(0)
    keep = np.union1d(rng.choice(P, min(int(n_pixels), P), replace=False),
                      sec_sel)
    m = np.zeros(P, bool)
    m[keep] = True
    return m


def nested_scopes(base_scopes, px_hold):
    """The nine `(name, mask)` pairs a roll is scored over: every base scope
    plus its `_trainlon` / `_holdlon` children.

    Lifted verbatim out of `main()` (2026-09-02) for the same reason as
    `gate_subset` above: `ml/lim_baseline.py` scores the classical baseline
    through this file's own battery, and a scope list assembled twice is a
    scope list that can drift. `px_hold` is `x_hold[xs]` — the [P] boolean
    saying which window pixels sit in the never-trained longitude block.
    """
    sel_train_x = ~px_hold
    sel_hold_x = px_hold
    return tuple(sc for name, m_ in base_scopes
                 for sc in ((name, m_),
                            (name + "_trainlon", m_ & sel_train_x),
                            (name + "_holdlon", m_ & sel_hold_x)))


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


def row_feats(ax, rows, dev, mode="month"):
    """The season features a head is fed, for a list of AXIS ROWS.

    THE HEAD'S OWN RECORDED MODE DECIDES, NOT THIS FILE (2026-08-22). A head
    trained with `--season-phase fine` was conditioned on the continuous
    fraction-of-year phase of each bin; rolling it with the month-quantized
    staircase would feed it a token it never saw at a resolution it was
    trained to resolve, and the roll would score the mismatch rather than the
    head. `ml/temporal.py` writes `season_phase` into the checkpoint args
    (`vars(a)`), so the caller reads it back the way it already reads
    `stencil` and `ring_km`, and passes it here.

    `mode="month"` reproduces `month_feats(ax.moy_of_row(r) ...)` exactly —
    which is what every archived roll fed — and `TimeAxis`'s own §4.9
    invariant is what makes that identical to the old `moy[r]` for rows
    inside the record (the axis asserts that each bin's true date lands in
    the calendar month its label names). Rows PAST the end of the record are
    defined for both modes, which is why the future roll can use one call.
    """
    rows = [int(r) for r in rows]
    if mode == "month":
        return month_feats([ax.moy_of_row(r) for r in rows], dev)
    if mode != "fine":
        sys.exit(f"unknown season phase mode {mode!r} in the head's args — "
                 f"this roll cannot reproduce the conditioning it was "
                 f"trained under, and guessing would score the mismatch")
    sc = [season_feat_of(ax.date_of_row(r), ax.step_days) for r in rows]
    return torch.from_numpy(np.asarray(sc, dtype=np.float32)).to(dev)


def roll_step(model, Zwin, NBR_t, static_ctx, mfeat, chunk, amp=False,
              quant=None, eps=None):
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
    remember to lower.

    E-057: `eps` is the FGN noise vector, [1, k] or [k], on any device. It is
    the SAME DRAW for every pixel of this step — FGN's convention is one
    global ε per (member, step), so it is expanded to the chunk's rows here
    rather than drawn per chunk, which is what makes the answer independent of
    `--chunk` (tests/test_fgn_roll.py pins chunk=P against chunk=P//3
    bitwise). `eps=None` is the pre-E-057 call EXACTLY — `model(...)` is
    invoked with the same three positional arguments it always was, so a
    deterministic head's forward is untouched; and a head built with
    eps_dim>0 that reaches this line with eps None is refused by the model
    itself (ml/temporal.py's forward guard), which is deliberate."""
    # A head trained with --input-quant reads a QUANTIZED state, and that is
    # part of its contract rather than a training trick: rolling it on
    # continuous z would feed it an alphabet it has never seen. Applied to
    # the whole window before the gather, which is equivalent to applying it
    # after (the gather only selects and permutes, and a missing slot is
    # zeroed below either way) and costs one pass instead of S.
    if quant is not None:
        Zwin = quant(Zwin)
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
            if eps is None:
                pred, _ = model(zin,
                                mfeat[None].expand(zin.shape[0], -1, -1),
                                static_ctx[sl])
            else:
                pred, _ = model(zin,
                                mfeat[None].expand(zin.shape[0], -1, -1),
                                static_ctx[sl],
                                eps=eps.reshape(1, -1).expand(zin.shape[0],
                                                              -1))
        outs.append(pred[:, -1].float())
    return torch.cat(outs, 0)


def decode_cells(codec, zhat, C, k_time, chunk, amp=False):
    """E-047 TIER 2: decode EVERY CELL of a block embedding — [P, k_time, C].

    `decode_all` asks a per-bin codec for C channels at offset 0. A block
    codec's z answers for a whole month, so the same question has k_time
    answers per channel: the value in the 1st pentad of that month, the 2nd,
    and so on. Each is scored against the truth of ITS OWN source bin, which
    is what makes a block roll comparable with a per-bin one at all — the
    alternative, scoring one representative cell, would throw away five
    sixths of the prediction and call the remainder a monthly number.
    """
    outs = []
    for i in range(0, zhat.shape[0], chunk):
        z = zhat[i:i + chunk]
        n = z.shape[0]
        qc = (torch.arange(C, device=z.device)[None, None, :]
              .expand(n, k_time, -1).reshape(n, k_time * C))
        qt = (torch.arange(k_time, device=z.device)[None, :, None]
              .expand(n, -1, C).reshape(n, k_time * C))
        off0 = torch.zeros(n, k_time * C, 3, dtype=torch.long, device=z.device)
        with torch.autocast(device_type="cuda", dtype=torch.float16,
                            enabled=amp):
            xq = codec.query(z, qc, off0, qt)
        outs.append(xq.float().reshape(n, k_time, C).cpu().numpy())
    return np.concatenate(outs, 0)


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
    block `Y`.

    `Y` is a `(y0, y1)` block (E-067) or a bare year — int or str, the form
    every call site used before blocks existed and the form a single-year
    block reduces to exactly.

    The roll breaks at the record's end and at the BLOCK boundary, so a start
    late in the block contributes fewer steps than the horizon asks for —
    which is why the monthly protocol's twelve starts cost 78 steps a year
    and not 144. One expression, used by the step PLAN and by `RollDump`,
    because two copies of a break condition are two chances to disagree about
    how long a trajectory is."""
    y0, y1 = block_bounds(Y)
    n = 0
    for h in range(1, Hh + 1):
        if s + h >= T or not (y0 <= ax.year[s + h] <= y1):
            break
        n += 1
    return n


def _fs_safe(s):
    """A head label reduced to what an ARTIFACT UPLOAD will accept.

    Measured on #433 (2026-08-22): the roll finished, the ml-metrics archive
    was fine, and `actions/upload-artifact@v4` refused the whole path list with
    "The path for one of the files in artifact is not valid" — because a
    stencil+ring label is `s145rspiral:111-4444-0.71-0.5_s0` and the COLON went
    into the filename. 2.4 GB of trajectories stayed on a rented box for one
    character. upload-artifact rejects `" : < > | * ? \\r \\n` in a path (it is
    the Windows-portable set, enforced on every runner), so the rule here is
    the tighter and simpler one: keep `[A-Za-z0-9._-]`, turn everything else
    into `-`.

    It is applied to the LABEL COMPONENT ONLY — the year and start-row parts
    are already digits — and the ORIGINAL label is kept verbatim in
    `dump_manifest.json` (`files[].head`) and inside each npz's own
    `meta_json`, so nothing about the file's identity is lost to its name.
    """
    return re.sub(r"[^A-Za-z0-9._-]", "-", str(s))


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
            "filename_rule": "roll_<head>_<year>_r<start_row>.npz, where "
                             "<head> is the head label with every character "
                             "outside [A-Za-z0-9._-] replaced by '-' "
                             "(actions/upload-artifact rejects : and the rest "
                             "of the Windows-portable set). `files[].head` "
                             "below and each npz's meta_json carry the "
                             "ORIGINAL label.",
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
        """One (block, start) trajectory. `z` is [n_states, P, d_z] float16.

        `Y` is the holdout BLOCK's LABEL (`block_label`) — "2009" for a
        single held-out year, exactly what this took before E-067, and
        "2008-2009" for a two-year block."""
        rows = [int(s) + k for k in range(z.shape[0])]
        name = f"roll_{_fs_safe(head_label)}_{Y}_r{int(s)}.npz"
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


# The ten quantities every skill row is built from. Named once so the pooled
# arrays and the E-058 per-channel arrays below cannot drift apart.
SUM_KEYS = ("mse_m", "mse_p", "mse_c", "mse_d", "n",
            "sxy", "sxx", "syy", "sx", "sy")


def new_sums(H, C=None):
    """The pooled skill sums, plus — when `C` is given — a PARALLEL
    per-channel set (E-058).

    WHY PER CHANNEL AT ALL. Chris, 2026-08-28: *"ocean surface temperature as
    a secondary downstream target (next to AMOC) ... to ensure the embedding
    representation is comprehensive and not just AMOC tailored"*. Until this
    change every row `skill_block` emitted was pooled over all pixels AND all
    channels, so `sst` — one of 40 channels, appended by E-042 — was
    invisible: no archived roll could say whether the rolled embedding
    predicts sea-surface temperature at all, only that it predicts "the
    field". The name `chan_skill` is LEGACY from rollout.py, where a "chan"
    row was a horizon; it is not, and never was, a per-channel read-out.

    WHY A PARALLEL SET RATHER THAN A DECOMPOSITION. The pooled scalars are
    algebraically the channel sums added up, so the pooled row COULD be
    recomposed from `per_chan`. It deliberately is not. Every archived roll
    number comes out of the pooled arrays, and the acceptance bar for this
    change is that they stay BIT-IDENTICAL (tests/test_per_channel_skill.py
    check 1); the cheapest way to guarantee that is never to touch the code
    path that produces them. `per_chan` is therefore additive in the strictest
    sense — new arrays, filled by new lines, appended after the pooled lines
    have already run. The two are still guaranteed to describe the same roll:
    `accumulate` fills both from the same masked arrays on the same early-exit
    branch, so no contribution can reach one and miss the other (check 2
    measures the recomposition deviation at 1.1e-16, one ulp).

    `C` is None for a caller that wants the old object exactly — the pooled
    path then has no `per_chan` key to find and behaves as it always did.
    """
    su = {k: np.zeros(H + 1) for k in SUM_KEYS}
    if C is not None:
        su["per_chan"] = {k: np.zeros((H + 1, int(C))) for k in SUM_KEYS}
    return su


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
    # ---- E-058: the same ten sums again, held open along the channel axis.
    # NOTHING ABOVE THIS LINE MOVED. Both call sites pass arrays already
    # sliced by a boolean PIXEL mask, so every argument here is [n_pixels, C]
    # and `op` is the [n_pixels, C] observation mask (truth-observed at the
    # target AND at the persistence source). Axis 0 is therefore the pixel
    # axis and `.sum(axis=0)` is "pool the pixels, keep the channels" — the
    # exact per-channel analogue of the unqualified `.sum()`s above. The
    # E-026b audit block in main() already reduces the same products this way
    # (`aud["ch_m"][h] += err_[corridor].sum(axis=0)` -> [C]), on the corridor
    # scope only and for msss_clim only; this generalises it to every scope,
    # every baseline and every horizon, and gives the channels their names.
    pc = su.get("per_chan")
    if pc is None:
        return
    if opf.ndim != 2 or opf.shape[1] != pc["n"].shape[1]:
        # A 1-D or mis-shaped mask would silently pool the wrong axis and
        # produce plausible nonsense, which is worse than a crash.
        raise ValueError(
            "accumulate: per-channel sums need [n_pixels, C] arrays with C=%d,"
            " got op%r" % (pc["n"].shape[1], tuple(opf.shape)))
    pc["mse_m"][h] += (((xhat - v_true) ** 2) * opf).sum(axis=0)
    pc["mse_p"][h] += (((v_pers - v_true) ** 2) * opf).sum(axis=0)
    pc["mse_d"][h] += (((v_damp - v_true) ** 2) * opf).sum(axis=0)
    pc["mse_c"][h] += ((v_true ** 2) * opf).sum(axis=0)
    pc["n"][h] += opf.sum(axis=0)
    pc["sxy"][h] += (xm * ym).sum(axis=0)
    pc["sxx"][h] += (xm * xm).sum(axis=0)
    pc["syy"][h] += (ym * ym).sum(axis=0)
    pc["sx"][h] += xm.sum(axis=0)
    pc["sy"][h] += ym.sum(axis=0)


def _skill_rows(su, H):
    """The chan_skill row list, from ONE set of 1-D [H+1] sums.

    THE ONLY ARITHMETIC PATH (E-058). This is `skill_block`'s original loop,
    moved out verbatim — not re-derived, not re-ordered — so that the pooled
    rows and the new per-channel rows are produced by literally the same
    float operations in the same order. Given the pooled sums it returns,
    element for element, what the pre-E-058 file returned; given one column
    of `su["per_chan"]` (a view, not a copy) it returns that channel's own
    rows. tests/test_per_channel_skill.py check 1 pins the first half of that
    claim against the pristine module recovered from git.
    """
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
    return rows


def skill_block(su, H, n_px=None, leads=None, chan_names=None):
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
    key is then omitted, so old readers are unaffected.

    `chan_names` (E-058) turns the sums' channel axis into the NEW
    `per_channel` key: channel NAME -> that channel's own row list, the same
    seven fields (`h`, `n`, `msss_clim`, `msss_pers`, `msss_damped`, `acc`,
    `amp_ratio`) computed from that channel's own sums by `_skill_rows`. It is
    written ALONGSIDE every existing key, never instead of one: `chan_skill`,
    `horizon_auc`, `auc_damped`, `horizon_auc_daymatched`, `n_px` and `empty`
    are byte-for-byte what they were. This is what makes "how well does the
    rolled embedding predict SST?" — Chris, 2026-08-28, ocean surface
    temperature as a secondary downstream target beside AMOC — a question the
    artefact can answer, instead of one hidden inside a 40-channel pool.

    THE NAMES ARE THE TENSOR'S OWN. They come from the codec checkpoint's
    `ck["chan"]` at the call site and are never hardcoded here or there; a
    list whose length does not match the sums' channel axis is REFUSED rather
    than zipped short, because a silently truncated list would mislabel every
    channel after the first mismatch (and `sst` is the LAST of the 40).

    NO SIZE GUARD, DELIBERATELY. The block is C x H rows of seven small
    numbers — 40 x 12 x 7 at the production monthly shape, a few hundred kB of
    JSON across all nine scopes — which is the same order as the E-026b audit
    block already written beside it and nowhere near a reason to gate, sample
    or truncate. Every write point in `main()` (the partial writes at the
    scored/long/future/head_done stages and the final unmarked one) picks this
    up for free, because `entry` is built once and mutated in place.

    Empty scopes follow §5.22 as everything else here does: a channel that
    scored nothing contributes no key, and if NO channel scored, `per_channel`
    itself is omitted rather than written as an empty husk or a NaN — the same
    rule that omits `horizon_auc` for a zero-pixel `_holdlon` scope."""
    rows = _skill_rows(su, H)
    out = {"chan_skill": rows}
    if n_px is not None:
        out["n_px"] = int(n_px)
        if not rows:
            out["empty"] = ("scope has 0 pixels — nothing to score"
                            if int(n_px) == 0 else
                            "scope has pixels but no horizon scored any of "
                            "them — investigate")
    # ---- E-058: the per-channel rows, additive and never in the way.
    # EMITTED HERE, BEFORE the aggregates, for one non-obvious reason:
    # tests/test_roll_monthly_identity.py asserts that
    # `horizon_auc_daymatched` is the LAST key of every scope block, because
    # its byte-strip also removes the comma that key's insertion added to the
    # line above it. Appending `per_channel` after it would break an invariant
    # that has nothing to do with this change; inserting before it leaves that
    # invariant exactly true. (That test still needs its counted exclusion
    # list widened by one for the new key — a deliberate decision it is
    # designed to force, not something this file can do for it.)
    pc = su.get("per_chan")
    if pc is not None and chan_names is not None:
        n_c = pc["n"].shape[1]
        if len(chan_names) != n_c:
            raise ValueError(
                "skill_block: chan_names has %d entries but the sums carry "
                "%d channels — refusing to zip short and mislabel the tail "
                "(sst is the LAST channel, so a short list loses exactly the "
                "read-out this key exists for)" % (len(chan_names), n_c))
        # A VIEW per channel, not a copy: `pc[k][:, c]` is the [H+1] column
        # `_skill_rows` expects, so the per-channel rows go through the pooled
        # rows' own code and no arithmetic is written twice.
        per = {}
        for c, nm in enumerate(chan_names):
            r = _skill_rows({k: pc[k][:, c] for k in SUM_KEYS}, H)
            if r:
                per[str(nm)] = r
        if per:
            out["per_channel"] = per
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


# ---- E-057: the ENSEMBLE read-outs, beside the deterministic ones --------
# Everything below is reached only for a head whose checkpoint carries
# `fgn_eps > 0`. The scoring functions themselves are ml/probscore.py's and
# are never re-derived here (E-057 roll spec §"The read-outs"); this file only
# accumulates their per-element / per-window answers over starts, the way
# `accumulate` accumulates the deterministic sums.
def member_seed(ens_seed, m):
    """Member m's noise-stream seed. ONE documented formula, in ONE place.

    `ens_seed * 1000003 + 59 + m` continues ml/temporal.py's own family (+57
    is the trainer's eps stream, +58 the fixed monitor bank), so a member seed
    can never collide with a training stream at the same base seed. The
    generator is a CPU one and the draws are moved to the device afterwards —
    device-independent streams, the same reason temporal.py's `eps_gen` is on
    the CPU."""
    return int(ens_seed) * 1000003 + 59 + int(m)


def member_gen(ens_seed, m):
    g = torch.Generator()
    g.manual_seed(member_seed(ens_seed, m))
    return g


def head_fgn_eps(path):
    """`fgn_eps` out of a head checkpoint's args, without paying for its
    weights where torch will let us skip them.

    This runs at ARGV TIME over every `--heads` entry, because "an fgn
    checkpoint with --ens-members < 2" is a property of the inputs alone and a
    guard that depends only on the inputs belongs where the inputs are all it
    has cost (ml/CLAUDE.md §0.3/§5.16) — not eleven hours in, after the gate
    head has rolled. `mmap=True` reads the zip directory and the pickled args
    without materialising the tensors; where the file predates that format the
    plain load is the fallback, which is what the head loop does anyway."""
    try:
        tk = torch.load(path, map_location="cpu", weights_only=False,
                        mmap=True)
    except Exception:                                          # noqa: BLE001
        tk = torch.load(path, map_location="cpu", weights_only=False)
    return int((tk.get("args") or {}).get("fgn_eps", 0) or 0)


ENS_STATS = ("crps", "n_crps",              # fair CRPS, summed over elements
             "msp", "mse_sp", "n_sp",       # spread_error's two mean terms
             "mse_mean", "mean_var", "mse_sample", "n_dec")


def new_ens_sums(H):
    return {k: np.zeros(H + 1) for k in ENS_STATS}


def accumulate_ens(su, h, crps_sum, n_crps, sp, dec, n_ok):
    """One (start, horizon, scope) contribution to the ensemble sums.

    `sp` is `probscore.spread_error`'s dict and `dec` is
    `probscore.ensemble_decomposition`'s, both computed on THIS scope's masked
    members and truth; `n_ok` is the number of elements they averaged over.
    They are re-multiplied by `n_ok` here so the aggregate over starts is a
    mean over the same element population the deterministic sums use — a mean
    of per-start means would weight a start with 600 scored pixels like one
    with 84,405.

    `spread` and `rmse` are square ROOTS of means, so their squares are what
    accumulates; the ratio is formed from the aggregated roots at the end,
    never averaged. A NaN term (nothing scored) contributes nothing rather
    than poisoning the sum — the same discipline `accumulate`'s `n == 0`
    early return has (ml/CLAUDE.md §5.22)."""
    if n_ok <= 0:
        return
    if np.isfinite(crps_sum) and n_crps > 0:
        su["crps"][h] += crps_sum
        su["n_crps"][h] += n_crps
    if np.isfinite(sp["spread"]) and np.isfinite(sp["rmse"]):
        su["msp"][h] += sp["spread"] ** 2 * n_ok
        su["mse_sp"][h] += sp["rmse"] ** 2 * n_ok
        su["n_sp"][h] += n_ok
    if all(np.isfinite(dec[k]) for k in ("mse_mean", "mean_var",
                                         "mse_sample")):
        su["mse_mean"][h] += dec["mse_mean"] * n_ok
        su["mean_var"][h] += dec["mean_var"] * n_ok
        su["mse_sample"][h] += dec["mse_sample"] * n_ok
        su["n_dec"][h] += n_ok


def ens_block(su, H, members, n_px=None):
    """The `ens_prob` block for one scope: per-horizon rows plus `crps_mean`.

    NO REFERENCE IS INVENTED. The spec considered a `crps_auc` against a
    climatological ensemble and struck it: a skill score needs a REGISTERED
    reference, and choosing one here would smuggle an analysis decision into
    the evaluator. What is written is the raw fair CRPS, the spread and its
    ratio, the three decomposition terms and a plain unweighted mean of the
    per-horizon CRPS — the same shape `horizon_auc` has, with none of its
    implied baseline.

    A horizon that scored nothing is OMITTED rather than written as NaN."""
    rows = []
    for h in range(1, H + 1):
        if su["n_crps"][h] <= 0 and su["n_dec"][h] <= 0:
            continue
        r = {"h": h}
        if su["n_crps"][h] > 0:
            r["n"] = int(su["n_crps"][h])
            r["crps"] = round(float(su["crps"][h] / su["n_crps"][h]), 5)
        if su["n_sp"][h] > 0:
            spread = math.sqrt(su["msp"][h] / su["n_sp"][h])
            rmse = math.sqrt(su["mse_sp"][h] / su["n_sp"][h])
            r["spread"] = round(spread, 5)
            r["rmse"] = round(rmse, 5)
            if rmse > 0:
                r["spread_ratio"] = round(spread / rmse, 5)
        if su["n_dec"][h] > 0:
            nd = su["n_dec"][h]
            r["mse_mean"] = round(float(su["mse_mean"][h] / nd), 5)
            r["mean_var"] = round(float(su["mean_var"][h] / nd), 5)
            r["mse_sample"] = round(float(su["mse_sample"][h] / nd), 5)
        rows.append(r)
    out = {"members": int(members), "rows": rows}
    if n_px is not None:
        out["n_px"] = int(n_px)
    got = [r["crps"] for r in rows if "crps" in r]
    if got:
        out["crps_mean"] = round(float(np.mean(got)), 5)
    return out


def ens_series_block(ens, obs, thresh, members, extra=None):
    """The ensemble read-out of one TRANSPORT band: [M, n] members vs [n]
    truth, through ml/probscore.

    Three instruments, all of them the module's own: the fair CRPS of the
    member transports, the spread-error ratio, and `brier_dip` on the DIP
    event (`obs < thresh`) — the one score a deterministic head cannot fake,
    since its probability is 0 or 1 and it is scored on being wrong outright.
    The threshold TRAVELS WITH THE NUMBER (`dip_threshold`) because a Brier
    score against an unrecorded event is unreadable.

    EVERY float here is written only if it is finite. probscore returns NaN
    rather than raising when nothing scored (its own NaN-probe rule), and a
    NaN that reaches `json.dump` is the literal `NaN` token ml/CLAUDE.md §5.22
    forbids in a results file — so an unscored instrument is an ABSENT key,
    which no reader can mistake for a measurement."""
    ens = np.asarray(ens, dtype=np.float64)
    obs = np.asarray(obs, dtype=np.float64)
    out = {"members": int(members), "n": int(ens.shape[1])}
    c = probscore.crps_ensemble(ens, obs)
    if np.isfinite(c["crps"]):
        out["crps"] = round(float(c["crps"]), 4)
    sp = probscore.spread_error(ens, obs)
    for k in ("spread", "rmse", "ratio"):
        if np.isfinite(sp[k]):
            out[{"ratio": "spread_ratio"}.get(k, k)] = round(float(sp[k]), 4)
    br = probscore.brier_dip(ens, obs, thresh, below=True)
    if br["n"]:
        out["dip"] = {
            "threshold": round(float(thresh), 4),
            "brier": round(float(br["brier"]), 5),
            "brier_clim": round(float(br["brier_clim"]), 5),
            "bss": (None if not np.isfinite(br["bss"])
                    else round(float(br["bss"]), 5)),
            "event_rate": round(float(br["event_rate"]), 4),
            "n": int(br["n"])}
    if extra:
        out.update(extra)
    return out


def dispersion_block(sv_mem, s1, s2, n_px, n_ch, members, phase, roll_ym):
    """E-057 §4: the DISPERSION CURVE of one long / future roll.

    Two scalars per step and nothing else — the spec is explicit that M full
    trajectories must NOT be stored, and at monthly xl144 with M=8 they would
    be 8 x 240 x 84,405 x 39 floats. What is kept instead is the question the
    battery actually asks: **does the spread grow with lead?** Genuine
    dynamics disperse; a head that has learned to replay the calendar returns
    the same trajectory whatever ε it is handed, and its curve is flat at
    zero. That collapse signature is the instrument's whole point, so the
    arrays must be able to show it exactly rather than approximately.

      * `sv_spread[i]` — the sample sd (ddof 1) over the M members of
        `read_sv` at step i. The transport dispersion.
      * `field_var[i]` — the mean over the corridor's (pixel, channel) cells
        of the per-cell sample variance (ddof 1) over the members, in the same
        standardized-anomaly space the scope blocks score.

    `s1` is [n_steps, n_px*n_ch] holding Σ_m x per cell and `s2` is [n_steps]
    holding Σ_m Σ_cells x², both accumulated while the members roll
    SEQUENTIALLY — which is what lets the member loop keep one Zwin on the
    device. The variance is then the textbook identity

        Σ_cells Σ_m (x - x̄_cell)²  =  Σ_cells [ Σ_m x²  -  (Σ_m x)² / M ]

    divided by (M-1)·n_cells. Clamped at 0: the identity is exact in exact
    arithmetic and can go a few ulp negative in float when the members are
    IDENTICAL, which is precisely the zero-film case this must report as 0.

    **THE PRECISION FLOOR IS COMPUTED AND SHIPPED, not left implicit.** That
    identity is the textbook one-pass form and it CANCELS: `s1` is a float32
    accumulator (it must be — at pentad, [1461 steps, 29,627 px x 40 ch] is
    6.9 GB at float32 and 13.9 at float64, and this array exists only so a
    two-scalar curve can be produced without keeping M trajectories), so
    `s1²/M` carries a relative error of order the float32 epsilon and the
    difference against `s2` cannot resolve a variance below it. Bounding
    δ(s1) ≤ (M-1)·eps32·|s1| and s1²/M ≤ s2 (Cauchy-Schwarz) gives

        δ(field_var[i])  ≤  2·eps32·s2[i] / n_cells

    which is reported as `field_var_floor` — the max over steps. A curve whose
    values sit AT that number is a collapsed head, not a dispersed one, and a
    reader who is not told the floor cannot tell those apart. For a trained
    head with a member spread of even 1% of the field scale the true variance
    is ~1e-4 against a floor of ~1e-7, i.e. three orders of margin; the floor
    binds only where the answer is "no dispersion", which is the one answer it
    is honest about.

    Returns None if anything came out non-finite — an absent curve rather
    than a NaN in a results file (ml/CLAUDE.md §5.22).
    """
    M = int(members)
    sv_mem = np.asarray(sv_mem, dtype=np.float64)              # [M, n]
    n = int(sv_mem.shape[1])
    ncell = int(n_px) * int(n_ch)
    sv_sd = sv_mem.std(axis=0, ddof=1)
    s2 = np.asarray(s2, dtype=np.float64)
    ss = s2 - (np.asarray(s1, dtype=np.float64) ** 2).sum(axis=1) / M
    fvar = np.maximum(ss, 0.0) / ((M - 1) * ncell)
    floor = float(2.0 * np.finfo(np.float32).eps * s2.max() / ncell) \
        if n else 0.0
    if not (np.isfinite(sv_sd).all() and np.isfinite(fvar).all()
            and math.isfinite(floor)):
        return None
    # %.6g, not a fixed number of decimals: a collapsed head's variance is
    # ~1e-8 and `round(v, 6)` writes it as 0.0, which is the right answer for
    # the wrong reason and hides the floor above from the reader entirely.
    return {
        "members": M, "phase": phase, "steps": n,
        "roll_ym": list(roll_ym),
        "n_corridor_px": int(n_px), "n_cells": ncell, "ddof": 1,
        "field_var_floor": float("%.3g" % floor),
        "sv_spread": [float("%.6g" % v) for v in sv_sd],
        "field_var": [float("%.6g" % v) for v in fvar],
        "note": ("member dispersion of THIS roll, two scalars per step: the "
                 "sample sd over the M members of the pooled transport "
                 "(`read_sv`, the same function `sv_des` is read with) and "
                 "the corridor mean of the per-cell member variance of the "
                 "decoded field. No truth enters either — this is a spread "
                 "curve, not a skill score. Dispersion growing with lead is "
                 "the genuine-dynamics signature; a flat curve at ~0 is the "
                 "collapse signature (E-057.3). `field_var_floor` is the "
                 "float32 cancellation bound of the one-pass variance "
                 "identity — a field_var at or below it is NO DISPERSION "
                 "MEASURED, not a measured small dispersion."),
    }


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
    ap.add_argument("--hold-years", default="",
                    help="comma list of years to hold out, OVERRIDING the "
                         "codec checkpoint's own `args['holdout_years']` for "
                         "the whole run — the t_hold mask behind the "
                         "standardisation statistics, the start list, the "
                         "roll's truncation, the `hold_years` written into "
                         "the artefact and every `fit_holdout_years_excluded` "
                         "field. E-067: CONSECUTIVE years group into blocks "
                         "(`hold_blocks`) and a roll is truncated at the end "
                         "of its BLOCK, so "
                         "--hold-years 2008,2009,2016,2017,2022,2023 is three "
                         "two-year blocks and 146 pentads (730 d) of lead are "
                         "scoreable where three single years cap every roll "
                         "at 365 d. REFUSED unless it is a SUPERSET of the "
                         "codec's own years: a year the codec trained on may "
                         "be held out at stage 2, never the reverse. Empty "
                         "(the default) takes the codec's list unchanged")
    ap.add_argument("--starts-per-year", type=int, default=0,
                    help="score only N of the staggered starts per holdout "
                         "BLOCK — every k-th, k = len(starts)//N, first N. "
                         "The name is the compatibility spelling: a block is "
                         "one year unless --hold-years grouped several, and "
                         "N is per BLOCK either way. 0 "
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
    ap.add_argument("--unpooled-readout", action=argparse.BooleanOptionalAction,
                    default=True, help="fit and write the UNPOOLED transport "
                    "read-out (new keys beside the pooled legacy ones). ON BY "
                    "DEFAULT (Chris, 2026-08-27: 'by default we should run "
                    "unpooled'); --no-unpooled-readout restores the legacy-only "
                    "artefact — the byte-identity test pins THAT path.")
    ap.add_argument("--unpooled-seed", type=int, default=0,
                    help="seed for the unpooled read-out's fit; recorded in "
                         "`probe_unpooled.seed` so the weights behind a "
                         "series can be reproduced from the artefact alone")
    ap.add_argument("--ens-members", type=int, default=8,
                    help="E-057: how many MEMBER TRAJECTORIES to roll per "
                         "start for a head whose checkpoint carries "
                         "`fgn_eps > 0`. READ ONLY FOR SUCH HEADS — a "
                         "deterministic head has no noise to resample and "
                         "this flag is a no-op for it, at any value. Members "
                         "roll SEQUENTIALLY (one Zwin at a time); each draws "
                         "ONE eps per step, shared across every pixel, which "
                         "is FGN's own rollout convention. Minimum 2: a "
                         "one-member 'ensemble' has no spread, and every "
                         "read-out this flag turns on is a statement about "
                         "spread")
    ap.add_argument("--ens-seed", type=int, default=0,
                    help="E-057: base seed of the member noise streams. "
                         "Member m rolls under a CPU torch.Generator seeded "
                         "`ens_seed * 1000003 + 59 + m` (the same family as "
                         "ml/temporal.py's +57 training stream and +58 "
                         "monitor bank), re-seeded at the start of every "
                         "trajectory — so a member's eps stream is a function "
                         "of (ens_seed, m, step) alone and any single "
                         "trajectory can be re-rolled in isolation and "
                         "reproduce bitwise")
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
    # E-047 TIER 2. A block codec's Z has ONE ROW PER BLOCK, so the axis this
    # roll advances is the BLOCK axis — in `month` mode its labels are
    # `YYYY-MM` and TimeAxis reads it as monthly, which is exactly why the
    # roll's horizon, bands and day-matched leads come out as the archive's.
    # The block map is loaded the way ml/temporal.py's embed path loads it
    # (from the tensor's own bin_index/epoch/pentad_days) and its ABSENCE is
    # a refusal, not a fallback: rolling a block codec on a per-bin axis
    # would advance the state six times too fast and every number would look
    # ordinary.
    BLKR = None
    _ck_peek = torch.load(a.ckpt, map_location="cpu", weights_only=False)
    _k_time = int(_ck_peek.get("args", {}).get("k_time", 1) or 1)
    if _k_time > 1:
        tb = str(_ck_peek["args"].get("time_block", "") or "")
        if not tb:
            sys.exit(f"{os.path.basename(a.ckpt)} has k_time={_k_time} but no "
                     f"`time_block` in its args: this checkpoint cannot say "
                     f"how its blocks were cut, and guessing the grouping "
                     f"would score a different experiment.")
        if "bin_index" not in d:
            sys.exit(f"{os.path.basename(a.ckpt)} is a BLOCK codec "
                     f"(time_block {tb!r}) but {os.path.basename(a.npz_small)} "
                     f"carries no `bin_index`: the block map cannot be "
                     f"rebuilt from a monthly tensor, and without it the "
                     f"cells have no dates to be scored against.")
        from timeblocks import BlockAxis
        BLKR = BlockAxis(tb, [str(m) for m in d["months"]], d["bin_index"],
                         dt.date.fromisoformat(str(d["epoch"]))
                         if "epoch" in d else EPOCH_DEFAULT,
                         int(np.asarray(d["pentad_days"]).item())
                         if "pentad_days" in d else None)
        print(f"block codec: k_time {_k_time} · {BLKR.describe(len(d['chan']), _ck_peek['d_z'])}",
              flush=True)
        # THE ROLL'S AXIS IS THE BLOCK AXIS, AND THE BLOCK AXIS DESCRIBES
        # ITSELF (E-048). This used to be `TimeAxis({"months": BLKR.labels})`,
        # which is the MONTHLY path — correct for month mode, where the labels
        # are unique contiguous `YYYY-MM` keys, and wrong for a window mode
        # whose labels repeat (two windows 15 days apart can sit in one
        # calendar month). `axis_dict()` returns the monthly descriptor
        # unchanged for month mode and a BINNED one for a W/S window, whose
        # step is the stride in days — so `--horizon`, the day-defined bands
        # and the day-matched leads are read off the stride instead of being
        # assumed to be months.
        ax = TimeAxis(BLKR.axis_dict())
        src_ax = TimeAxis(d)
    else:
        ax = TimeAxis(d)
        src_ax = ax
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
    # ---- E-057: which of these heads are FGN heads, decided NOW ----------
    # The whole ensemble path below is gated on ONE fact — the head's own
    # checkpoint carrying `fgn_eps > 0` — and it is read here, before any
    # tensor, mask, codec or roll step has been paid for, so the M<2 refusal
    # and the cost banner both land at argv time. A run with no fgn head
    # prints NOTHING here and takes no new branch anywhere: `head_eps` is all
    # zeros, `FGN` is False in every head, and the artefact is the artefact it
    # has always been.
    # `--export-mask-only` and `--dump-truth` exit below without ever building
    # a head; they must not start paying to open one now either, so the read is
    # skipped in exactly the modes that have no head loop.
    head_eps = ({} if (a.export_mask_only or a.dump_truth)
                else {h: head_fgn_eps(h) for h in heads})
    fgn_heads = [h for h in heads if head_eps.get(h, 0) > 0]
    if fgn_heads:
        if a.ens_members < 2:
            sys.exit(
                f"--ens-members {a.ens_members} but "
                f"{len(fgn_heads)} of these heads carry fgn_eps > 0 "
                f"({', '.join(os.path.basename(h) for h in fgn_heads)}). An "
                f"FGN head has no single output — every forward is a SAMPLE "
                f"from its predictive distribution — and every read-out this "
                f"unlocks (fair CRPS, spread-error, the dip Brier, the "
                f"dispersion curve) is a statement about SPREAD, which one "
                f"member does not have. Roll it with --ens-members >= 2 "
                f"(default 8), or roll a deterministic head. Refusing at "
                f"argv time rather than after the first head "
                f"(ml/CLAUDE.md §0.3). Plan: ml/plans/E057_fgn_head.md")
        _k = sorted({head_eps[h] for h in fgn_heads})
        print(f"E-057 FGN MODE for {len(fgn_heads)} of {len(heads)} head(s): "
              f"M = {a.ens_members} member trajectories per start, eps ~ "
              f"N(0,1)^{_k if len(_k) > 1 else _k[0]}, ens_seed "
              f"{a.ens_seed} (member m draws from a CPU generator seeded "
              f"{a.ens_seed}*1000003+59+m = "
              f"{member_seed(a.ens_seed, 0)}.."
              f"{member_seed(a.ens_seed, a.ens_members - 1)}"
              f"). ONE eps per (member, step), SHARED ACROSS EVERY PIXEL of "
              f"that step — FGN's own convention (arXiv:2506.10772; noise is "
              f"drawn inside every predictor call, so an AR roll resamples it "
              f"each step). Members roll SEQUENTIALLY. The scope blocks are "
              f"scored on the ENSEMBLE MEAN field through the unchanged "
              f"accumulate/skill_block path; the member spread is reported "
              f"beside them under `ens_prob`. Deterministic heads in this "
              f"same run are untouched.", flush=True)
        if a.ens_members < 4:
            print(f"::warning::--ens-members {a.ens_members} is the minimum "
                  f"this refusal allows, not a recommended ensemble size: "
                  f"the fair CRPS is M-independent in expectation but its "
                  f"variance is not, and E-057.1 rolls at M=16", flush=True)
    ck = torch.load(a.ckpt, map_location="cpu", weights_only=False)
    C = len(ck["chan"])
    # E-058: THE CHANNEL NAMES ARE THE TENSOR'S, and this is the only place
    # they are read. `C` on the line above is `len` of this same list, so the
    # names and the channel axis cannot disagree; nothing downstream hardcodes
    # a channel list or an index (least of all `sst`'s, which is merely the
    # last entry today and would move the moment a channel is inserted).
    chan_names = [str(c) for c in ck["chan"]]
    d_z = ck["d_z"]
    # WHICH YEARS ARE HELD OUT, AND WHO SAID SO. `ck` here is the CODEC
    # checkpoint (`--ckpt`, loaded five lines above) — never a stage-2 head —
    # so this is the authority that matters: a LIM or a head may be denied
    # more than the codec was, never less.
    ck_hold_years = sorted(ck["args"]["holdout_years"].split(","))
    if a.hold_years:
        hold_years = sorted({str(y).strip() for y in a.hold_years.split(",")
                             if str(y).strip()})
        missing = [y for y in ck_hold_years if y not in set(hold_years)]
        if missing:
            # THE DIRECTION MATTERS AND IT IS CHECKED WHERE THE INPUTS ARE
            # ALL IT HAS COST (§0.3). Dropping a codec holdout year would
            # score the roll on years the codec was FITTED on and report the
            # number beside the archive's.
            sys.exit(
                f"--hold-years {a.hold_years!r} is not a superset of the "
                f"codec's own holdout years "
                f"({ck['args']['holdout_years']!r}): "
                f"{', '.join(missing)} would be scored despite the codec "
                f"having trained on {'it' if len(missing) == 1 else 'them'}. "
                f"A year the codec trained on may be held out at stage 2; "
                f"the reverse is contamination. Refusing.")
        hold_src = "--hold-years override"
    else:
        hold_years = ck_hold_years
        hold_src = "codec args"
    # E-067: the years, grouped into the runs a roll is truncated at. Every
    # loop below iterates BLOCKS, and `block_label` keys the result JSON —
    # which for single years is `str(Y)`, the key every archived artefact has.
    blocks = hold_blocks(hold_years)
    print(f"holdout years: {','.join(hold_years)} — from {hold_src} · "
          f"blocks {', '.join(block_label(b) for b in blocks)}", flush=True)
    t_hold = np.array([m[:4] in set(hold_years) for m in months])
    # E-047: `months`/`t_hold`/`moy` above are the ROLL's axis, which for a
    # block codec is the BLOCK axis. `stream_stats` and `StdMonths` read rows
    # of X — bins, whatever the codec embeds — so they keep the SOURCE axis's
    # own month-of-year and holdout mask. Identical objects when there are no
    # blocks, so the per-bin path is the same call it always was.
    src_moy = src_ax.moy if BLKR is not None else moy
    src_t_hold = (np.array([m[:4] in set(hold_years) for m in src_ax.labels])
                  if BLKR is not None else t_hold)
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
    if a.hold_years:
        # E-067 · THE STATS CACHE IS NOT KEYED ON THE HOLDOUT YEARS. A warm
        # box that wrote `std_stats.npz` under the codec's own years would
        # hand them straight to a run holding out more — a different
        # climatology and a different z-score behind numbers that look
        # exactly the same, which is the silent-substitution failure §0.2 is
        # about. So the override reads and writes its OWN file, named for the
        # blocks, and the shared one is neither read nor overwritten. Absent
        # the flag this is the same path and the same bytes it always was.
        st_path = os.path.join(a.cache_dir, "std_stats__hold-%s.npz"
                               % "-".join(block_label(b) for b in blocks))
        print(f"anomaly stats: --hold-years gets its own cache "
              f"{os.path.basename(st_path)} — the shared std_stats.npz was "
              f"written under the codec's years and is left alone", flush=True)
    if os.path.exists(st_path):
        s_ = np.load(st_path)
        clim, dyn = s_["clim"], list(s_["dyn"])
        mean_c, std_c = s_["mean_c"], s_["std_c"]
    else:
        clim, dyn, mean_c, std_c = stream_stats(Xm, src_moy, src_t_hold,
                                                x_hold)
        np.savez(st_path, clim=clim, dyn=np.array(dyn),
                 mean_c=mean_c, std_c=std_c)
    std_m = StdMonths(Xm, ys, xs, src_moy, clim, dyn, mean_c, std_c)

    r1 = None
    if not a.export_mask_only:     # the mask needs no baseline arithmetic
        print("AR1 damped-persistence pass over the record ...", flush=True)
        r1 = ar1_train(std_m, T, t_hold, P, C)                 # [P, C]

    # ---- the three scopes -------------------------------------------------
    corridor, cor_thr = corridor_pixels(Xm, ocean, ys, xs, t_hold, sec_sel,
                                        a.corridor_pctl, a.corridor_dilate)
    gate_mask = gate_subset(P, a.pixels_gate, sec_sel)
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
    # Scoring cost: the two children PARTITION their parent, so the masked
    # pixel work per horizon doubles (parent + its two halves), it does not
    # triple. Cheap next to a roll step.
    scopes = nested_scopes(base_scopes, px_hold)
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
        x0[..., c] = ((x0[..., c] - clim[src_moy[0], :, :, c]
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
            elif BLKR is not None:
                # E-047: a block codec's static identity is the same question
                # asked of a GRID — static channels repeated across the first
                # block's cells, pad cells unobserved (ml/temporal.py does the
                # identical thing at its own static pass).
                kt = BLKR.k_max
                vv = Xt0[0, ys[sl], xs[sl]][:, None, :].expand(-1, kt, -1)
                oo = (stat_obs[ys[sl], xs[sl]][:, None, :]
                      .expand(-1, kt, -1).clone())
                oo &= torch.as_tensor(~BLKR.pad[0])[None, :, None]
            else:
                vv = Xt0[0, ys[sl], xs[sl]]
                oo = stat_obs[ys[sl], xs[sl]]
            _mk = (torch.zeros(n, C, dtype=torch.bool, device=dev)
                   if BLKR is None else
                   torch.zeros(n, BLKR.k_max, C, dtype=torch.bool, device=dev))
            zs.append(codec.encode(
                vv.to(dev), oo.to(dev), _mk,
                torch.as_tensor(ctx).to(dev)).cpu().numpy())
    Zstat = np.concatenate(zs, 0)
    print(f"static identity encoded for {P} px", flush=True)

    # ---- verify the Z cache against a live re-encode (project_amoc's
    # guard: a silent ordering mismatch would roll beautiful nonsense) ------
    Zsec = np.asarray(Zm[:, sec_sel]).astype(np.float32)       # [T,S,dz]
    rows3 = [sec_y - 1, sec_y, sec_y + 1]
    slab, obs_sl = build_slab(Xm, rows3, src_moy, clim, dyn, mean_c,
                              std_c)
    slab_t = torch.from_numpy(np.nan_to_num(slab, nan=0.0))
    obs_t = torch.from_numpy(obs_sl)
    # E-047: the re-encode must feed the codec what the EMBED fed it — a
    # block codec's ctx is the block CENTRE's continuous phase and its input
    # is the grid, so the same block map goes in here. Per-bin: unchanged.
    ctx_all = (np.stack([np.sin(2 * np.pi * moy / 12),
                         np.cos(2 * np.pi * moy / 12)], 1)
               if BLKR is None else BLKR.ctx_phase())
    rngv = np.random.default_rng(1)
    kv = rngv.choice(S_sec, min(8, S_sec), replace=False)
    sxs = xs[sec_sel]
    Zl, _ = embed_everything(codec, slab_t, obs_t, ctx_all, lats[rows3], lons,
                             np.ones(len(kv), dtype=int), sxs[kv], d_z,
                             cache_path=None, batch=64,
                             blk_rows=(None if BLKR is None else BLKR.rows),
                             blk_pad=(None if BLKR is None else BLKR.pad))
    for tt in (0, T // 2, T - 1):
        dmax = float(np.abs(Zl[tt] - Zsec[tt][kv]).max())
        zscale = float(np.abs(Zl[tt]).max())
        # E-067 · WHEN --hold-years IS ON, NAME THE LIKELY CAUSE. The
        # standardisation statistics are derived from the holdout years, and
        # the embed cache's key (codec weight hash, sha256 of the RAW tensor)
        # sees neither the transform nor the years — so a Z embedded under
        # the codec's own list will fail here against an overridden one, and
        # "ordering mismatch" would send a reader looking in the wrong place.
        # The refusal stays a refusal; only the sentence changes.
        assert dmax < max(0.02, 0.005 * zscale), (
            f"Z mismatch at t={tt}: {dmax} vs scale {zscale}"
            + ("" if not a.hold_years else
               f". --hold-years {a.hold_years!r} moved the anomaly "
               f"statistics (they are derived from the holdout years), and "
               f"the embed cache is keyed on the codec weights and the raw "
               f"tensor only — neither term sees them. This Z was almost "
               f"certainly embedded under a different holdout list. Re-embed "
               f"with the same years (ml/temporal.py --holdout-years "
               f"{','.join(hold_years)}), or roll without the override."))
    print("Z cache verified vs live re-encode ✓", flush=True)

    # ---- transport read-out (truefit protocol, rollout.py verbatim) ------
    # E-047: `rapid[:, 0]` is a SOURCE row. On a block roll the read-out
    # lives on the block axis (Zsec has one row per block), so the truth is
    # remapped to the block CONTAINING each row — the same remap the embed
    # path uses, from the same function.
    rapid = d["rapid"] if BLKR is None else BLKR.remap_rows(d["rapid"])
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

    # ---- E-055: the SAME rolled states, read WITHOUT pooling the section ---
    # `read_sv` above is exception 1 of ml/CLAUDE.md §3 and does not move: the
    # e017_u1_s0 gate's three band criteria are computed from it against a
    # hardcoded GATE_REF at GATE_TOL 0.0101, so a changed transport read-out
    # ends every eval wave in `sys.exit("VALIDATION GATE FAILED")` before it
    # scores anything. The follow-up §8.1 asks for is an ADDITIONAL function
    # writing NEW keys, which is this one.
    #
    # Its weights are FITTED, and what they are fitted on is the whole
    # question. `tr_all` is `~t_hold[ridx]` — the RAPID months outside the
    # held-out years, the identical mask the pooled ridge above is fitted on.
    # No held-out year, and no rolled state, enters the fit: the head sees only
    # TRUE section latents from train months, exactly as `fit_ridge` does, so
    # the two read-outs differ in their pooling and in nothing else. The fit
    # window travels with the output (`results["probe_unpooled"]`) so a reader
    # of the artefact alone can say what was fitted on what.
    read_sv_unpooled = None
    unpooled_meta = None
    if a.unpooled_readout:
        try:
            from temporal import (UNPOOLED_HEAD_DIM, UNPOOLED_STEPS,
                                  attn_pool_predict, fit_attn_pool,
                                  lon_fraction, section_tokens,
                                  unpooled_device)
            _sec_ix = np.asarray(sec_sel)
            _lonf = lon_fraction(lons[xs[_sec_ix]])
            _tok_true = section_tokens(Zsec, _lonf)          # [T, S, dz+2]
            _tr_rows = ridx[tr_all]
            _mu_u = float(rv_des[tr_all].mean())
            _sd_u = float(rv_des[tr_all].std() + 1e-9)
            _udev = unpooled_device()
            _t_u0 = time.time()
            _net_u, _val_mse = fit_attn_pool(
                _tok_true[_tr_rows], (rv_des[tr_all] - _mu_u) / _sd_u,
                seed=a.unpooled_seed, device=_udev)
            # in-sample-on-train r, reported as such: it is a "did the fit
            # take" reading, NOT a skill number, and every skill number this
            # read-out produces is scored on rolled states below.
            _p_tr = attn_pool_predict(_net_u, _tok_true[_tr_rows],
                                      _udev) * _sd_u + _mu_u
            _r_tr = corr_or_none(_p_tr, rv_des[tr_all])

            def read_sv_unpooled(zhat):
                tok = section_tokens(
                    zhat[sec_t].detach().to(torch.float32).cpu().numpy()[None],
                    _lonf)
                return float(attn_pool_predict(_net_u, tok, _udev)[0]
                             * _sd_u + _mu_u)

            unpooled_meta = {
                "readout": "learned softmax attention over the section's "
                           "pixels (probe_head.SectionHead), in place of "
                           "`read_sv`'s `zhat[sec].mean(0)`",
                "head": "probe_head.SectionHead", "d": UNPOOLED_HEAD_DIM,
                "steps_max": UNPOOLED_STEPS, "seed": int(a.unpooled_seed),
                "device": _udev.type,
                "section_pixels": int(len(_sec_ix)),
                "fit_on": "TRAIN months of the RAPID truth only "
                          "(~t_hold[rapid rows]); no held-out year and no "
                          "rolled state enters the fit",
                "fit_rows": int(tr_all.sum()),
                "fit_first": ax.label_of_row(int(_tr_rows.min())),
                "fit_last": ax.label_of_row(int(_tr_rows.max())),
                "fit_holdout_years_excluded": list(hold_years),
                "target": "rv_des (train-month climatology removed, the same "
                          "`deseason_truth` the pooled probe uses)",
                "target_mu": round(_mu_u, 4), "target_sd": round(_sd_u, 4),
                "inner_tail_mse": round(float(_val_mse), 5),
                "fit_r_train_insample": _r_tr,
                "fit_wall_s": round(time.time() - _t_u0, 1),
                "note": ("ADDITIONAL read-out. `read_sv`, GATE_REF, GATE_TOL "
                         "and every pooled key are unchanged; the gate is "
                         "still decided by the pooled path. These keys have "
                         "no reference of their own yet and are not gated."),
            }
            print(f"unpooled probe fit: {int(tr_all.sum())} train labels "
                  f"({unpooled_meta['fit_first']}..{unpooled_meta['fit_last']}"
                  f"), in-sample r {_r_tr}, {_udev.type}, "
                  f"{unpooled_meta['fit_wall_s']}s", flush=True)
        except Exception as _e:                               # noqa: BLE001
            # Never fatal and never a NaN: the keys are simply absent, which a
            # reader cannot mistake for a measurement (ml/CLAUDE.md §5.22).
            read_sv_unpooled = None
            unpooled_meta = None
            print(f"::warning::unpooled read-out unavailable "
                  f"({type(_e).__name__}: {_e}) — the pooled read-out and the "
                  f"gate are unaffected", flush=True)

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
    # E-055: the unpooled read-out's PROVENANCE, once, at the top of the file
    # rather than repeated per head — what was fitted, on which months, at
    # which seed. Written ONLY when --unpooled-readout asked for it, so a
    # default monthly artefact gains nothing (see the flag's help, and
    # tests/test_roll_monthly_identity.py).
    if unpooled_meta is not None:
        results["probe_unpooled"] = unpooled_meta
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
                block_label(b): len(ax.starts_for_block(b,
                                                        a.starts_per_year))
                for b in blocks},
            "starts_available_per_holdout_year": {
                block_label(b): len(ax.starts_for_block(b)) for b in blocks},
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
    # E-047 TIER 2: what one scored horizon actually covers. A block roll's
    # `h` is a BLOCK step, and each one accumulates every real cell of that
    # block against its own bin — so the LEAD IN DAYS of a scored value is a
    # pair (h, cell), and it is derived from the axis rather than assumed.
    if BLKR is not None:
        _b0 = int(ax.T // 2)                       # a mid-record reference
        _lead = {}
        for _h in range(1, min(a.horizon, ax.T - _b0 - 1) + 1):
            _tb = _b0 + _h
            _lead[str(_h)] = [
                round((src_ax.date_of_row(int(BLKR.rows[_tb, _j]))
                       - src_ax.date_of_row(int(BLKR.rows[_b0, _j]))).days, 1)
                for _j in range(int(BLKR.n_bins[_tb]))
                if _j < int(BLKR.n_bins[_b0])]
        results["blocks"] = {
            "time_block": str(ck["args"].get("time_block", "")),
            "k_time": int(ck["args"].get("k_time", 1)),
            "k_max": int(BLKR.k_max),
            "n_blocks": int(BLKR.n_blocks),
            "bins_per_block": [int(BLKR.n_bins.min()),
                               int(BLKR.n_bins.max())],
            "scored_per_block": "every real cell, against its own source bin",
            "lead_days_by_horizon_and_cell": _lead,
            # E-048: the stride is what one roll STEP means, so it travels in
            # the artefact beside the numbers rather than being recoverable
            # only from the mode string.
            "width_bins": (None if BLKR.width is None else int(BLKR.width)),
            "stride_bins": (None if BLKR.stride is None else int(BLKR.stride)),
            "overlap_bins": (None if BLKR.overlap is None
                             else int(BLKR.overlap)),
            "step_days": float(ax.step_days),
            "note": ("`h` counts BLOCKS. chan_skill's `n` at horizon h counts "
                     "CELL-months, ~k_time times a per-bin roll's, because "
                     "one block prediction is scored at every bin it covers. "
                     "Persistence and the damped baseline use the START "
                     "block's SAME cell."),
            # THE BASELINE MOVED, NOT THE METRIC (E-048). Said in the
            # artefact because this is where a reader meets the number.
            "persistence_note": (
                None if not BLKR.overlap else
                f"OVERLAPPING WINDOWS: consecutive blocks share "
                f"{int(BLKR.overlap)} of {int(BLKR.width)} bins, so the "
                f"persistence baseline (the previous block's same cell) is "
                f"stronger BY CONSTRUCTION than it is at stride "
                f"{int(BLKR.width)}. Every ratio against persistence on this "
                f"axis is comparable with another roll at the SAME stride and "
                f"with nothing else.")}

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
            "available": {block_label(b): len(ax.starts_for_block(b))
                          for b in blocks},
            "rows": {block_label(b): ax.starts_for_block(b,
                                                         a.starts_per_year)
                     for b in blocks},
            "labels": {block_label(b): [
                ax.label_of_row(s) for s in
                ax.starts_for_block(b, a.starts_per_year)] for b in blocks},
        }
        print("starts: %d per holdout block — %s" % (
            a.starts_per_year, "; ".join(
                f"{block_label(b)} rows "
                f"{results['starts']['rows'][block_label(b)]} "
                f"({', '.join(results['starts']['labels'][block_label(b)])})"
                for b in blocks)), flush=True)

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
                  {block_label(b): len(ax.starts_for_block(
                      b, a.starts_per_year)) for b in blocks},
                  a.horizon, P, d_z)
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
        # E-067 · THE HEAD'S OWN HOLDOUT YEARS, WHEN IT CARRIES THEM. A
        # stage-2 head trained with `ml/temporal.py --holdout-years` records
        # the EFFECTIVE list in its own args, and it can legitimately be a
        # superset of the codec's. This roll is scored on the codec's list
        # (or on `--hold-years`), so a head denied MORE years than this roll
        # holds out is being scored on years it never saw — a fair number,
        # but not the number the dispatcher probably meant. Say so and change
        # nothing: guessing which list a roll wanted is exactly the silent
        # substitution §0.2 is about.
        _hta = str(ta.get("holdout_years", "") or "")
        if _hta and not a.hold_years:
            _hset = sorted({y.strip() for y in _hta.split(",") if y.strip()})
            if _hset != sorted(set(hold_years)):
                print(f"::warning::{os.path.basename(hp)} was trained with "
                      f"holdout years {','.join(_hset)}, this roll is scored "
                      f"on {','.join(hold_years)} (from {hold_src}). Rolling "
                      f"it on the codec's list anyway — pass --hold-years "
                      f"{','.join(_hset)} to score the head on its own.",
                      flush=True)
        # E-053.1 · REFUSE A NON-CONTIGUOUS HEAD, at the point the file is
        # opened and before any geometry, embedding or roll is built (§0.3,
        # §5.16). A head trained with --frame-offsets read frame j at
        # t+offsets[j]; this roller assembles a CONTIGUOUS [t-K+1 .. t] slab
        # and self-feeds one bin at a time, so it would hand such a head a
        # window it has never seen — and produce a corridor AUC that looks
        # exactly like every other one. Offset-aware context assembly is
        # E-053.1's own follow-up; until it exists the honest answer is to
        # stop.
        _foff = str(ta.get("frame_offsets", "") or "")
        if _foff:
            sys.exit(
                f"{os.path.basename(hp)} was trained with --frame-offsets "
                f"{_foff!r} (E-053.1): its context frames are NOT contiguous, "
                f"and this roller builds a contiguous window and feeds its "
                f"own prediction into the next bin. Rolling it here would "
                f"score a head on inputs it never saw and report the number "
                f"as comparable. Refusing.")
        K = ta["K"]
        if K_seen is None:
            K_seen = K
            results["K"] = K
        elif K != K_seen:
            sys.exit(f"{hp} has K={K} != {K_seen} — windows not comparable")
        stencil = ta.get("stencil", 1)
        ring_km = ta.get("ring_km", 0) or 0        # number OR "222,555"
        # The head's SEASON CONDITIONING, read back the way stencil and
        # ring_km are: ml/temporal.py saves vars(a), so a head trained with
        # --season-phase fine carries it and a head that predates the flag
        # carries nothing — which is 'month', the only thing it can be.
        sphase = str(ta.get("season_phase", "month") or "month")
        # THE HEAD'S INPUT ALPHABET, read back the same way. A head that
        # predates the flag carries neither key and rolls continuous, which
        # is every archived head.
        iq_spec = str(ta.get("input_quant", "") or "")
        iq = None
        if iq_spec:
            _sg = ta.get("input_quant_sigma")
            if not _sg:
                sys.exit(f"{os.path.basename(hp)} was trained with "
                         f"--input-quant {iq_spec!r} but carries no "
                         f"input_quant_sigma: the quantizer's per-dimension "
                         f"scale is measured at TRAIN time and cannot be "
                         f"re-derived here without handing this head a grid "
                         f"it never saw. Refusing to roll it.")
            iq = InputQuant(iq_spec, np.asarray(_sg, np.float32), d_z)
            print(f"  {label}: rolling with the head's own input quantizer "
                  f"({iq_spec}, {iq.bits_per_dim:.3f} bits/dim)", flush=True)
        unroll = ta.get("unroll", 1)
        # E-057: the head's own `fgn_eps`, read back exactly the way stencil,
        # ring_km, season_phase and input_quant are — from the args
        # ml/temporal.py saved. A head that predates the flag carries nothing,
        # which is 0, which is the deterministic path, byte for byte.
        eps_dim = int(ta.get("fgn_eps", 0) or 0)
        FGN = eps_dim > 0
        M_ens = int(a.ens_members) if FGN else 0
        if FGN and BLKR is not None:
            sys.exit(
                f"{os.path.basename(hp)} is an FGN head (fgn_eps={eps_dim}) "
                f"and this roll is on a BLOCK codec's axis "
                f"(time_block {str(ck['args'].get('time_block', ''))!r}). The "
                f"member reduction below decodes one field per (member, "
                f"horizon) and a block roll decodes k_time cells per horizon; "
                f"combining the two has never been designed, let alone "
                f"tested, and guessing would score an experiment nobody ran. "
                f"Refusing.")
        # THE LABEL SAYS WHICH KIND OF NUMBER THIS IS. An fgn entry's scope
        # blocks are the ENSEMBLE MEAN's, not a single forward's, and the two
        # must never be confused in a table keyed by label alone.
        label = (f"s{stencil}"
                 + (f"r{str(ring_km).replace(',', '-')}" if _ring_on(ring_km) else "")
                 + (f"u{unroll}" if unroll != 1 else "")
                 + (f"fgnM{M_ens}" if FGN else "")
                 + f"_s{ta.get('seed', 0)}")
        if label in results["heads"]:
            label += "_" + os.path.basename(hp).replace(".pt", "")
        k_tbl = tk["model"]["pos.weight"].shape[0]  # the file, not a convention
        dir_ = tuple(int(x) for x in
                     str(ta.get("direct") or "").split(",") if x.strip())
        model = TemporalTransformer(d_z=d_z, d_model=ta["d_model"],
                                    n_heads=4, n_layers=ta["layers"],
                                    k_max=k_tbl, direct=dir_, stencil=stencil,
                                    eps_dim=eps_dim)
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
        for B in blocks:
            for s_ in ax.starts_for_block(B, a.starts_per_year):
                if s_ - K + 1 < 0 or s_ + 1 >= T:
                    continue
                n_skill += scored_horizon(ax, s_, Hh, T, B)
        # One hindcast per `--long-start` label that resolves to a row. With
        # the single default this is exactly `a.long_months if the label
        # resolves else 0`, which is what it has always been.
        n_long = a.long_months * sum(
            ax.row_of_label(s.strip()) is not None
            for s in str(a.long_start or "").split(",") if s.strip())
        head_i = list(heads).index(hp) + 1
        # E-057: THE MULTIPLIER IS STATED, NOT SILENT. Every roll step below
        # is run once per member, so the head's step budget — and the progress
        # bar's total, which is what an ETA is computed from — is M times what
        # a deterministic head of the same shape would cost.
        mult = M_ens if FGN else 1
        prog.start_head(head_i, label,
                        (n_skill + n_long + a.future_months) * mult)
        print(f"  {label}: {n_skill} scored roll steps + {n_long} hindcast + "
              f"{a.future_months} future = "
              f"{n_skill + n_long + a.future_months} steps over {P:,} pixels",
              flush=True)
        if FGN:
            _hmax = max([scored_horizon(ax, s_, Hh, T, B)
                         for B in blocks
                         for s_ in ax.starts_for_block(B, a.starts_per_year)
                         if s_ - K + 1 >= 0 and s_ + 1 < T] or [0])
            _mem_gb = M_ens * _hmax * P * C * 2 / 1e9
            _disp_gb = (max(a.long_months, a.future_months)
                        * int(corridor.sum()) * C * 4 / 1e9)
            print(f"  {label}: FGN head, eps_dim {eps_dim} — x{M_ens} "
                  f"members = "
                  f"{(n_skill + n_long + a.future_months) * M_ens:,} roll "
                  f"steps in total, plus one full-field decode per (member, "
                  f"scored step) and one CORRIDOR decode per (member, "
                  f"long/future step). Member field buffer "
                  f"{M_ens}x{_hmax}x{P:,}x{C} float16 = {_mem_gb:.2f} GB per "
                  f"start; dispersion accumulator <= {_disp_gb:.2f} GB per "
                  f"long/future roll.", flush=True)
        # E-058: `C` opens the parallel per-channel arrays beside the pooled
        # ones. Both `accumulate` call sites below (the skill/start loop and
        # the FGN ensemble-mean loop) feed them from the same [n_pixels, C]
        # arrays they already feed the pooled sums from, so no call site
        # changed shape or order.
        sums = {name: new_sums(Hh, C) for name, _ in scopes}
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
        # E-055: the unpooled read-out's points, in a SEPARATE dict rather
        # than a fourth element of `probe_pts`'s tuples — the band loop below
        # unpacks those as `(s_, pr, y)` and the gate reads what that loop
        # writes, so widening the tuple would put the gate's own input through
        # an edit. Empty and never read when the flag is off.
        probe_pts_u = {h: [] for h in range(1, Hh + 1)}
        # E-057: the MEMBER read-outs. `probe_pts_e[h]` holds one [M] array
        # per (start, h) — the same rolled states `probe_pts[h]` was reduced
        # from, before the reduction — and `ens_sums` accumulates the
        # probscore answers per scope exactly the way `sums` accumulates the
        # deterministic ones. All three stay empty for a deterministic head.
        probe_pts_e = {h: [] for h in range(1, Hh + 1)}
        probe_pts_eu = {h: [] for h in range(1, Hh + 1)}
        ens_sums = ({name: new_ens_sums(Hh) for name, _ in scopes}
                    if FGN else None)
        # THE GATE HEAD IS NEVER DUMPED (see RollDump): it is the run's
        # certificate, it is re-rolled in every eval wave, and nobody
        # animates it.
        dump_head = dump is not None and GATE_HEAD not in os.path.basename(hp)
        if dump is not None and not dump_head:
            print(f"  {label}: --dump-roll skips the gate head", flush=True)

        def fgn_start(B, s, traj):
            """E-057: ONE start, rolled M times, then reduced.

            `B` is the holdout BLOCK this start belongs to (E-067) — the
            thing `scored_horizon` truncates against, one year or a run of
            consecutive ones.

            The deterministic path below this function is untouched — it is
            not called for a head whose `fgn_eps` is 0, and nothing it writes
            changes. What happens here is the same protocol with the member
            dimension put in:

              * M trajectories, rolled SEQUENTIALLY so only one [P, K, d_z]
                window is ever on the device. Member m draws from its own CPU
                generator (`member_seed`), re-seeded at this start, and each
                step's eps is ONE vector shared by every pixel — which is
                what makes the result independent of `--chunk`.
              * Each member's decoded field is kept per horizon as float16
                [M, n_h, P, C]; at monthly xl144 with M=8 that is 8 x 79 MB =
                0.63 GB for a 12-step start, freed before the next one.
              * The ENSEMBLE MEAN field then goes through the identical
                `accumulate` / audit lines the deterministic roll uses, so
                every scope block, the corridor AUC and the E-026b audit are
                computed by the code that computed every archived number —
                just fed the mean. That is what makes E-057 F1's comparison
                against the znoise pair mechanical rather than argued.
              * The MEMBERS are scored beside it through ml/probscore, never
                re-derived here, under the weathernext NaN rule: a member is
                NaN'd wherever the truth is unobserved, so spread cannot hide
                in cells nobody can score.
            """
            n_h = scored_horizon(ax, s, Hh, T, B)
            if n_h == 0:
                return
            v_pers, obs_s = std_m.get(s)
            mem = np.empty((M_ens, n_h, P, C), np.float16)
            sv_m = np.zeros((M_ens, n_h))
            sv_mu = (np.zeros((M_ens, n_h)) if read_sv_unpooled is not None
                     else None)
            for m in range(M_ens):
                gen = member_gen(a.ens_seed, m)
                Zwin = zwin_from_true(s, K)
                cur = list(range(s - K + 1, s + 1))
                for h in range(1, n_h + 1):
                    t_tgt = s + h
                    # DRAWN ON THE CPU, THEN MOVED (temporal.py's eps_gen
                    # rule): a device-side generator would make the member
                    # stream a function of which box rolled it.
                    eps = torch.randn(1, eps_dim, generator=gen).to(dev)
                    zhat = roll_step(model, Zwin, NBR_t, static_ctx,
                                     row_feats(ax, cur, dev, sphase),
                                     a.chunk, a.amp, quant=iq, eps=eps)
                    if traj is not None and m == 0:
                        # MEMBER 0 ONLY (spec §5): the animation is one
                        # trajectory and M x the bytes buys nothing for it.
                        traj[h] = zhat.detach().to(torch.float16) \
                                      .cpu().numpy()
                    Zwin = torch.cat([Zwin[:, 1:], zhat[:, None]], 1)
                    cur = cur[1:] + [t_tgt]
                    mem[m, h - 1] = decode_all(codec, zhat, C, a.chunk, a.amp)
                    sv_m[m, h - 1] = read_sv(zhat)
                    if sv_mu is not None:
                        sv_mu[m, h - 1] = read_sv_unpooled(zhat)
                    prog.step("skill")
            for h in range(1, n_h + 1):
                t_tgt = s + h
                v_true, obs_tt = std_m.get(t_tgt)
                op = obs_tt & obs_s
                v_damp = v_pers * r1 ** h
                ens = mem[:, h - 1].astype(np.float32)          # [M, P, C]
                xhat = ens.mean(0)
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
                # The NaN rule, in both arrays: an unobserved cell is not a
                # cell the ensemble may be confident or dispersed in.
                obs_n = np.where(op, v_true, np.nan)
                ens_n = np.where(op[None], ens, np.nan)
                # CRPS is PER ELEMENT, so it is computed ONCE over all P and
                # then masked per scope — identical to calling it on each
                # scope's subset, and nine times cheaper than doing so.
                cf = probscore.crps_ensemble(ens_n, obs_n)["crps_field"]
                for name, m_ in scopes:
                    n_ok = int(op[m_].sum())
                    if n_ok == 0:
                        continue
                    sub, ob = ens_n[:, m_], obs_n[m_]
                    accumulate_ens(
                        ens_sums[name], h,
                        float(np.nansum(cf[m_])),
                        int(np.isfinite(cf[m_]).sum()),
                        probscore.spread_error(sub, ob),
                        probscore.ensemble_decomposition(sub, ob), n_ok)
                if t_tgt in r_of_row:
                    y_ = rv_des[r_of_row[t_tgt]]
                    # `read_sv` is AFFINE in the section mean — it is
                    # `dot((zhat[sec].mean(0) - mu)/sd ++ 1, w)` — so the mean
                    # of the member reads IS the read of the ensemble-mean
                    # state, exactly. That is why `amoc_bands` keeps its
                    # meaning ("bands of the point forecast") with the point
                    # forecast now being the ensemble mean, and why no second
                    # [n_h, P, d_z] buffer is needed to produce it.
                    # tests/test_fgn_roll.py pins the identity.
                    probe_pts[h].append((s, float(sv_m[:, h - 1].mean()), y_))
                    probe_pts_e[h].append((s, sv_m[:, h - 1].copy(), y_))
                    if sv_mu is not None:
                        # THE UNPOOLED READ IS NOT AFFINE (a learned softmax
                        # attention pool), so this is the ENSEMBLE MEAN OF THE
                        # READ, not the read of the ensemble-mean state — the
                        # exact mean of the [M] array beside it in
                        # `probe_pts_eu`, which is the invariant a paired
                        # comparison needs. `amoc_bands_unpooled` is itself a
                        # new, ungated E-055 key, so no archived number
                        # depends on the choice; it is recorded in
                        # meta.fgn.amoc_bands_unpooled_are.
                        probe_pts_u[h].append(
                            (s, float(sv_mu[:, h - 1].mean()), y_))
                        probe_pts_eu[h].append(
                            (s, sv_mu[:, h - 1].copy(), y_))

        with torch.no_grad():
            # E-067: BLOCKS, not years. `B` is `(y0, y1)`; `blabel` is the
            # key/filename token, which is `str(Y)` for a one-year block and
            # therefore leaves every archived name untouched.
            for B in blocks:
                blabel = block_label(B)
                for s in ax.starts_for_block(B, a.starts_per_year):
                    start_m = ax.label_of_row(s)
                    if s - K + 1 < 0 or s + 1 >= T:
                        continue
                    Zwin = zwin_from_true(s, K)
                    cur = list(range(s - K + 1, s + 1))
                    v_pers, obs_s = std_m.get(s)
                    # The trajectory buffer, sized from the SAME break
                    # condition the h-loop below uses, so its last slot is
                    # the last state that loop produces — never a zero row
                    # that would animate as a dead ocean. State 0 is the
                    # TRUE embedding of the start row: Zwin's last slot,
                    # read from the cache in its own float16.
                    n_dump = (scored_horizon(ax, s, Hh, T, B)
                              if dump_head else 0)
                    traj = (np.empty((n_dump + 1, P, d_z), np.float16)
                            if n_dump else None)
                    if traj is not None:
                        traj[0] = np.asarray(Zm[s])
                    if FGN:
                        # E-057. The deterministic loop below is left exactly
                        # as it is — not wrapped, not re-indented — so that a
                        # reader (and a diff) can see that an fgn head takes a
                        # DIFFERENT path rather than the same path with
                        # conditionals sprinkled through it. The eight lines
                        # after the call are the ones this `continue` skips.
                        fgn_start(B, s, traj)
                        print(f"  {label} start {start_m}: rolled "
                              f"({M_ens} members)", flush=True)
                        if traj is not None:
                            dump.write(label, os.path.basename(hp),
                                       {"stencil": stencil,
                                        "ring_km": ring_km,
                                        "seed": ta.get("seed", 0),
                                        "unroll": unroll, "K": K,
                                        "d_model": ta["d_model"],
                                        "layers": ta["layers"],
                                        "fgn_eps": eps_dim,
                                        "members": M_ens,
                                        "ens_seed": int(a.ens_seed),
                                        "member": 0}, blabel, s, traj)
                            traj = None
                        continue
                    for h in range(1, Hh + 1):
                        t_tgt = s + h
                        # THE BREAK IS `scored_horizon`'S, WRITTEN OUT.
                        # E-067 made it a BLOCK boundary rather than a year
                        # boundary; `B` is `(y0, y1)` and reduces to the old
                        # `!= int(Y)` for a one-year block.
                        if t_tgt >= T or not (B[0] <= ax.year[t_tgt] <= B[1]):
                            break
                        # month features are the CURRENT window's months —
                        # rollout.py's mseq at step h spans s-K+h .. s+h-1;
                        # advance AFTER the forward, like project_amoc.py
                        zhat = roll_step(model, Zwin, NBR_t, static_ctx,
                                         row_feats(ax, cur, dev, sphase),
                                         a.chunk, a.amp, quant=iq)
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
                        cur = cur[1:] + [t_tgt]
                        if BLKR is None:
                            cells = [(None, t_tgt, decode_all(
                                codec, zhat, C, a.chunk, a.amp))]
                        else:
                            # E-047 TIER 2: one decode, k_time answers. Cell j
                            # of the predicted BLOCK is scored against the
                            # truth of the SOURCE BIN it stands for, so a
                            # month-block roll accumulates ~6 cell-months per
                            # scored block and `n` in chan_skill grows with
                            # it. Pad cells have no bin and are skipped.
                            xg = decode_cells(codec, zhat, C, BLKR.k_max,
                                              a.chunk, a.amp)
                            cells = [(j, int(BLKR.rows[t_tgt, j]), xg[:, j])
                                     for j in range(int(BLKR.n_bins[t_tgt]))]
                        prog.step("skill")
                        for j_, row_, xhat in cells:
                            # THE BASELINES MOVE WITH THE CELL. Persistence is
                            # the start block's SAME cell (the analogue of "the
                            # last state you saw"), which is what keeps a
                            # cell's skill a statement about forecasting rather
                            # than about which pentad of a month it is.
                            if j_ is None:
                                v_p, obs_p = v_pers, obs_s
                            else:
                                v_p, obs_p = std_m.get(int(BLKR.rows[s, j_]))
                            v_true, obs_tt = std_m.get(row_)
                            op = obs_tt & obs_p
                            v_damp = v_p * r1 ** h
                            for name, m_ in scopes:
                                accumulate(sums[name], h, xhat[m_],
                                           v_true[m_], v_p[m_], v_damp[m_],
                                           op[m_])
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
                            if read_sv_unpooled is not None:
                                probe_pts_u[h].append(
                                    (s, read_sv_unpooled(zhat),
                                     rv_des[r_of_row[t_tgt]]))
                    print(f"  {label} start {start_m}: rolled", flush=True)
                    if traj is not None:
                        dump.write(label, os.path.basename(hp),
                                   {"stencil": stencil, "ring_km": ring_km,
                                    "seed": ta.get("seed", 0),
                                    "unroll": unroll, "K": K,
                                    "d_model": ta["d_model"],
                                    "layers": ta["layers"]},
                                   blabel, s, traj)
                        traj = None

        entry = {"meta": {"file": os.path.basename(hp), "stencil": stencil,
                          "ring_km": ring_km, "seed": ta.get("seed", 0),
                          "unroll": unroll}}
        if FGN:
            # SO NO READER CAN MISTAKE THIS FOR A SINGLE-FORWARD NUMBER. Every
            # scope block in this entry was computed from the ENSEMBLE MEAN
            # field; the deterministic archive's blocks were computed from the
            # one field a deterministic head emits. The two are comparable on
            # purpose (that is E-057 F1) and they are not the same object.
            entry["meta"]["fgn"] = {
                "members": M_ens,
                "mode": "ensemble_mean",
                "eps_dim": eps_dim,
                "ens_seed": int(a.ens_seed),
                "member_seed_rule": "ens_seed*1000003 + 59 + m, re-seeded at "
                                    "the start of every trajectory",
                "eps_convention": "one eps ~ N(0,1)^k per (member, roll "
                                  "step), SHARED across every pixel of that "
                                  "step and resampled at each step — FGN's "
                                  "own rollout convention "
                                  "(arXiv:2506.10772)",
                "scope_blocks_are": "the ensemble MEAN field, through the "
                                    "unchanged accumulate/skill_block path",
                "amoc_bands_are": "the ensemble-mean transport (read_sv is "
                                  "affine in the section mean, so the mean "
                                  "of the member reads is the read of the "
                                  "mean state)",
                "amoc_bands_unpooled_are": "the ensemble MEAN OF THE READ — "
                                           "the learned attention pool is "
                                           "not affine, so this is not the "
                                           "read of the ensemble-mean state; "
                                           "it is the exact mean of the "
                                           "member array in "
                                           "amoc_bands_ens_unpooled",
                "member_readouts": "ens_prob (per scope), amoc_bands_ens, "
                                   "long_dispersion / future_dispersion",
            }
        for name, m_ in scopes:
            entry[name] = skill_block(sums[name], Hh, n_px=int(m_.sum()),
                                      leads=leads, chan_names=chan_names)
        if FGN:
            entry["ens_prob"] = {
                name: ens_block(ens_sums[name], Hh, M_ens,
                                n_px=int(m_.sum()))
                for name, m_ in scopes}
            entry["ens_prob"]["note"] = (
                "fair CRPS (probscore.crps_ensemble), spread/rmse/"
                "spread_ratio (probscore.spread_error, with its (M+1)/M "
                "finite-ensemble correction) and the three decomposition "
                "terms (probscore.ensemble_decomposition, mse_sample = "
                "mse_mean + mean_var) of the M member FIELDS against truth, "
                "in the same standardized-anomaly space the scope blocks "
                "score, over the same pixel scopes and the same observed "
                "cells. Members are NaN where truth is unobserved, so spread "
                "cannot hide in cells nobody can score. `crps_mean` is the "
                "unweighted mean over the horizons that scored — NOT a skill "
                "score: no reference ensemble is defined for it here, and "
                "choosing one is an analysis decision with its own "
                "registration, not this evaluator's.")
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
        # E-055: THE SAME BANDS, THE SAME POINTS, THE UNPOOLED READ-OUT.
        # Identical arithmetic to the block above — same `bands`, same key
        # rule, same >=8-point floor, same refusal to write a NaN — over
        # `probe_pts_u`, whose entries were produced from the SAME `zhat` at
        # the SAME (start, h) as `probe_pts`'. So `amoc_bands_unpooled[k]` and
        # `amoc_bands[k]` are a PAIRED pair: same rolled states, same targets,
        # differing only in whether the section was averaged before the
        # read-out. The gate above still reads `amoc_bands` and only that.
        if read_sv_unpooled is not None:
            entry["amoc_bands_unpooled"] = {}
            for bn, hs in bands:
                key = ax.band_key(bn, hs)
                pts = [(h, s_, pr, y) for h in hs if h <= Hh
                       for (s_, pr, y) in probe_pts_u.get(h, [])]
                if len(pts) >= 8:
                    pr = np.array([p[2] for p in pts])
                    tv = np.array([p[3] for p in pts])
                    r_ = corr_or_none(pr, tv)
                    entry["amoc_bands_unpooled"][key] = (
                        {"r": r_, "n": len(pts)} if r_ is not None else
                        {"n": len(pts),
                         "undefined": ("r is not defined for this band — one "
                                       "of the two series has zero variance "
                                       "over its %d points; omitted rather "
                                       "than written as NaN (ml/CLAUDE.md "
                                       "§5.22)" % len(pts))})
            print(f"  {label} amoc UNPOOLED: " + " ".join(
                (f"{bn} {v['r']:+.3f}(n={v['n']})" if v.get("r") is not None
                 else f"{bn} UNDEFINED(n={v['n']})")
                for bn, v in entry["amoc_bands_unpooled"].items())
                + "   [not gated — see results.probe_unpooled]", flush=True)
        if FGN:
            # E-057 §3: THE SAME BANDS, THE SAME POINTS, THE M MEMBERS. Each
            # entry of `probe_pts_e` is the [M] array the corresponding
            # `probe_pts` scalar was the mean of, so `amoc_bands_ens[k]` and
            # `amoc_bands[k]` are a paired pair: same rolled states, same
            # targets, differing only in whether the members were reduced.
            #
            # THE DIP THRESHOLD IS DERIVED AND RECORDED. -1 sigma of the
            # deseasonalised TRAINING-year target series — the same series the
            # bands correlate against, deseasonalised by the same
            # `deseason_truth` — because a Brier score against an unrecorded
            # event is unreadable, and because a hand-picked Sv level would be
            # a threshold wearing a definition's clothes (ml/CLAUDE.md §4.9).
            _dip_sigma = float(np.std(rv_des[tr_all]))
            _dip_thr = -_dip_sigma
            entry["meta"]["fgn"]["dip"] = {
                "threshold": round(_dip_thr, 4),
                "sigma": round(_dip_sigma, 4),
                "rule": "deseasonalised transport below -1 sigma, sigma = "
                        "std of rv_des over the TRAIN rows of the RAPID "
                        "series (~t_hold), the same series and the same "
                        "climatology the bands are correlated against",
                "n_sigma_rows": int(tr_all.sum())}

            def _band_ens(pts_by_h):
                out = {}
                for bn, hs in bands:
                    pts = [p for h in hs if h <= Hh
                           for p in pts_by_h.get(h, [])]
                    if len(pts) < 8:
                        continue
                    ensm = np.stack([np.asarray(p[1], dtype=np.float64)
                                     for p in pts], 1)          # [M, n]
                    tv_ = np.array([p[2] for p in pts], dtype=np.float64)
                    out[ax.band_key(bn, hs)] = ens_series_block(
                        ensm, tv_, _dip_thr, M_ens)
                return out

            entry["amoc_bands_ens"] = _band_ens(probe_pts_e)
            print(f"  {label} amoc ENSEMBLE (M={M_ens}): " + " ".join(
                f"{bn} crps "
                + ("%.3f" % v["crps"] if "crps" in v else "n/a")
                + " spread/rmse "
                + ("%.3f" % v["spread_ratio"] if "spread_ratio" in v
                   else "n/a")
                + f" dip-bss {v.get('dip', {}).get('bss')}"
                for bn, v in entry["amoc_bands_ens"].items())
                + f"   [dip < {_dip_thr:+.3f} Sv-anomaly]", flush=True)
            if read_sv_unpooled is not None:
                entry["amoc_bands_ens_unpooled"] = _band_ens(probe_pts_eu)
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
        def long_roll(s_end, n_steps, phase="long", sv_u_out=None):
            """Roll `n_steps` AXIS STEPS past row `s_end`.

            The old body advanced a calendar month and a month-of-year token
            once per step. At monthly that is the axis; at pentad it ran the
            model's own seasonal input a full simulated year forward every 60
            real days, and then attached truth on the wrong dates — a
            corrupted INPUT, not merely a mislabelled output. Both now come
            from the ROW: one step is one row, and the season token is that
            row's true month.

            E-055: `sv_u_out`, when handed a list, receives the UNPOOLED
            transport of the same `zhat` at every step. It is an
            out-parameter rather than a third return value because both call
            sites unpack `sv, roll_rows` and one of them feeds
            `entry["long"]` — the block the archive reads. Default None, so
            with the flag off the roll is the roll it was.
            """
            Zwin = zwin_from_true(s_end, K)
            cur = list(range(s_end - K + 1, s_end + 1))
            sv, roll_rows = [], []
            with torch.no_grad():
                for i in range(n_steps):
                    r_next = s_end + 1 + i
                    zhat = roll_step(model, Zwin, NBR_t, static_ctx,
                                     row_feats(ax, cur, dev, sphase),
                                     a.chunk, a.amp, quant=iq)
                    Zwin = torch.cat([Zwin[:, 1:], zhat[:, None]], 1)
                    cur = cur[1:] + [r_next]
                    prog.step(phase)
                    sv.append(read_sv(zhat))
                    if sv_u_out is not None:
                        sv_u_out.append(read_sv_unpooled(zhat))
                    roll_rows.append(r_next)
            return np.array(sv), roll_rows

        # ---- E-057 §4: the SAME roll, M members, two scalars per step -----
        # A SEPARATE function rather than a member argument on `long_roll`:
        # `long_roll` produces `entry["long"]`/`entry["future"]`, the blocks
        # the archive reads and the E-055 series hangs off, and a head with no
        # `fgn_eps` must reach it through code no conditional has entered.
        # The two share their arithmetic line for line (same `zwin_from_true`,
        # same `row_feats`, same window shift, same `prog.step(phase)` per roll
        # step) and differ in exactly two things: the ε handed to `roll_step`,
        # and the corridor decode that feeds the dispersion accumulators.
        _corr_ix = (torch.as_tensor(np.where(corridor)[0], device=dev)
                    if FGN else None)
        _n_corr = int(corridor.sum())

        def long_roll_ens(s_end, n_steps, phase="long", sv_u_out=None):
            """M member trajectories past row `s_end` → (mean sv, rows, disp).

            Members roll SEQUENTIALLY, so only one [P, K, d_z] window is ever
            on the device — the same memory posture the scored starts have.
            What cannot be kept is the M decoded trajectories (8 x 240 x
            84,405 x 39 at monthly xl144), so the corridor field is reduced
            INTO the two running accumulators `dispersion_block` needs the
            moment it is decoded, and thrown away:

              `s1[i]` = Σ_m x over that step's corridor cells (per cell), and
              `s2[i]` = Σ_m Σ_cells x²             (one number per step).

            The returned `sv` is the ensemble MEAN transport, which is what
            `entry["long"]["sv_des"]` and every correlation in `long_block`
            are then computed from — the same "the point forecast is the
            ensemble mean" convention the scored starts use, and exact for
            `read_sv` because it is affine.
            """
            ncell = _n_corr * C
            s1 = np.zeros((n_steps, ncell), np.float32)
            s2 = np.zeros(n_steps, np.float64)
            sv_mem = np.zeros((M_ens, n_steps))
            svu_mem = (np.zeros((M_ens, n_steps))
                       if sv_u_out is not None else None)
            roll_rows = [s_end + 1 + i for i in range(n_steps)]
            with torch.no_grad():
                for m in range(M_ens):
                    gen = member_gen(a.ens_seed, m)
                    Zwin = zwin_from_true(s_end, K)
                    cur = list(range(s_end - K + 1, s_end + 1))
                    for i in range(n_steps):
                        r_next = s_end + 1 + i
                        # CPU generator, moved to the device — the member
                        # stream must not be a function of which box rolled it
                        # (temporal.py's eps_gen rule).
                        eps = torch.randn(1, eps_dim, generator=gen).to(dev)
                        zhat = roll_step(model, Zwin, NBR_t, static_ctx,
                                         row_feats(ax, cur, dev, sphase),
                                         a.chunk, a.amp, quant=iq, eps=eps)
                        Zwin = torch.cat([Zwin[:, 1:], zhat[:, None]], 1)
                        cur = cur[1:] + [r_next]
                        prog.step(phase)
                        sv_mem[m, i] = read_sv(zhat)
                        if svu_mem is not None:
                            svu_mem[m, i] = read_sv_unpooled(zhat)
                        xc = decode_all(codec, zhat[_corr_ix], C,
                                        a.chunk, a.amp).reshape(-1)
                        s1[i] += xc
                        s2[i] += float(np.dot(xc.astype(np.float64),
                                              xc.astype(np.float64)))
            if sv_u_out is not None:
                sv_u_out.extend(svu_mem.mean(0).tolist())
            disp = dispersion_block(
                sv_mem, s1, s2, _n_corr, C, M_ens, phase,
                [ax.label_of_row(r) for r in roll_rows])
            if disp is None:
                print(f"::warning::{label} {phase} dispersion is not finite "
                      f"— the curve is OMITTED rather than written as NaN",
                      flush=True)
            return sv_mem.mean(0), roll_rows, disp

        _long_specs = [s.strip() for s in str(a.long_start or "").split(",")
                       if s.strip()]

        def long_block(row, spec):
            """The hindcast block for ONE context end. Identical arithmetic to
            the single-start version it was factored out of (2026-08-22) —
            truth attaches on the AXIS ROW through `r_of_row`, `trained` is
            `t_hold` on that row's RAPID index, the lowpass is the same
            `smooth`, and the dict is built key for key in the same order, so
            a single-start run is byte-identical to before the list existed.

            E-057: an fgn head rolls the SAME hindcast M times and `sv` below
            is the ensemble mean of the M transports; the dispersion curve is
            returned BESIDE the block rather than inside it, so the block a
            deterministic head writes gains no key."""
            sv_u = [] if read_sv_unpooled is not None else None
            disp = None
            if FGN:
                sv, roll_rows, disp = long_roll_ens(row, a.long_months,
                                                    sv_u_out=sv_u)
            else:
                sv, roll_rows = long_roll(row, a.long_months, sv_u_out=sv_u)
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
            # E-055: the unpooled series of the SAME roll, on the same rows,
            # appended AFTER every archived key so the block reads as it did
            # with one key more. Only the series — r_trained/r_heldout/lp18
            # stay the pooled read-out's, because those are the numbers the
            # archive quotes and nothing here may reinterpret them.
            if sv_u is not None:
                blk["sv_des_unpooled"] = [round(float(v), 3) for v in sv_u]
            print(f"  {label} long({spec}+{a.long_months}m): "
                  f"r_trained {r_tr} (n={n_tr}) · r_heldout {r_ho} "
                  f"(n={n_ho}) · lp18 r {r_lp} amp {amp_lp}", flush=True)
            if disp is not None:
                print(f"  {label} long({spec}) dispersion (M={M_ens}): "
                      f"sv_spread {disp['sv_spread'][0]:.4f} → "
                      f"{disp['sv_spread'][-1]:.4f} · field_var "
                      f"{disp['field_var'][0]:.5f} → "
                      f"{disp['field_var'][-1]:.5f} over {disp['steps']} "
                      f"steps", flush=True)
            return blk, disp

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
            blk, _disp = long_block(_row, _spec)
            if "long" not in entry:
                entry["long"] = blk
                # E-057 §4. The PRIMARY hindcast's dispersion curve gets the
                # spec's own top-level name, beside `entry["long"]` rather
                # than inside it, so no archived key changes shape. The extra
                # `--long-start` ends carry theirs inside their own
                # `long_multi` block — dispersion always travels beside the
                # series it describes.
                if _disp is not None:
                    entry["long_dispersion"] = _disp
            else:
                if _disp is not None:
                    blk["dispersion"] = _disp
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
            sv_u = [] if read_sv_unpooled is not None else None
            _fdisp = None
            if FGN:
                sv, roll_rows, _fdisp = long_roll_ens(
                    T - 1, a.future_months, "future", sv_u_out=sv_u)
            else:
                sv, roll_rows = long_roll(T - 1, a.future_months, "future",
                                          sv_u_out=sv_u)
            roll_ym = [ax.label_of_row(r) for r in roll_rows]
            entry["future"] = {"context_end": ax.label_of_row(T - 1),
                               "roll_ym": roll_ym,
                               "sv_des": [round(v, 3) for v in sv.tolist()]}
            if sv_u is not None:                              # E-055
                entry["future"]["sv_des_unpooled"] = [
                    round(float(v), 3) for v in sv_u]
            if _fdisp is not None:                            # E-057 §4
                entry["future_dispersion"] = _fdisp
                print(f"  {label} future dispersion (M={M_ens}): sv_spread "
                      f"{_fdisp['sv_spread'][0]:.4f} → "
                      f"{_fdisp['sv_spread'][-1]:.4f} · field_var "
                      f"{_fdisp['field_var'][0]:.5f} → "
                      f"{_fdisp['field_var'][-1]:.5f} over "
                      f"{_fdisp['steps']} steps", flush=True)
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

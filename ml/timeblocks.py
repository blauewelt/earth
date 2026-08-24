#!/usr/bin/env python3
"""The BLOCK AXIS: how a run of pentad bins becomes one embedding's worth of
input, and what that block is called afterwards.

E-047 (Chris, 2026-08-22). Today the codec embeds ONE bin: 40 channels of one
5-day pentad, and 35 of those 40 are absent in five bins out of six, because
RG-Argo is a monthly product written into a single mid-month bin (measured
2026-08-22: `n_rg_live` 252/3142, 79.41% of ocean pixels when present). A
stage-2 head then has to fuse across time what the codec never fused. The
alternative is to give the codec the whole month at once — a k_max x C GRID of
cells, one embedding out — so the Argo anchor and the five pentads around it
land in ONE representation, and the absence of Argo in five of them becomes
exactly what the codec already handles: an unobserved cell.

THIS MODULE IS THE AXIS ARITHMETIC ONLY. It decides which bins go in which
block, what the block is called, where its centre falls, and which cells are
padding. It contains no model, no loss and no training: every consumer
(ml/train.py's codec, ml/temporal.py's embed path, the tests) derives the same
grouping from the same function, because two copies of an axis rule are two
places for the same off-by-one to live — this repo has paid that bill twice
(the anomaly transform's four copies, and `month_index`'s 6.09:1 collapse).

THREE MODES.

  `month`   Group by CALENDAR MONTH LABEL. A pentad year has 73 bins, so a
            month carries 6 or 7 of them and blocks are RAGGED; k_max is 7 and
            short months are padded. One block is one month, and that is the
            point: the resulting Z has a MONTHLY axis built entirely out of
            5-day data, so `TimeAxis` reads it as monthly and the roll's
            horizon, bands and day-matched leads are the archive's own.
  `N`       Fixed N consecutive bins, no alignment, label from the block's
            FIRST bin. Blocks are uniform, k_max is N, and the trailing
            remainder is dropped rather than padded (a partial block at the
            end of the record would be the only ragged one in the mode whose
            whole point is that it is not). Exactly `N/N` (below).
  `W/S`     E-048 (Chris, 2026-08-24): a WIDTH-W window that advances by
            STRIDE S bins. `6/6` is 6 pentads in, advance 30 days — the same
            grouping as `6`, spelled so the advance is visible. `6/3` is the
            same 6-pentad input advancing 15 days, so CONSECUTIVE EMBEDDINGS
            OVERLAP by W-S = 3 bins and each embedding still carries the one
            monthly Argo stamp its window covers. Chris: "One embedding every
            15 days (6 pentads/30 days as input, OVERLAPPING windows), advance
            by 15 days — each embedding has Argo as part of it, two consecutive
            ones share the same monthly Argo values."

WHAT A BLOCK IS CALLED. Month mode labels the block `YYYY-MM`, which makes the
block axis indistinguishable from a monthly tensor's to everything downstream.
Fixed-N and W/S mode label it with its FIRST bin's own label — the window's
calendar ANCHOR — so a label is always the label of a real calendar position
and never an index. `TimeAxis` parses it, so that is a requirement and not a
convention.

THE LABEL IS NOT THE AXIS, AND AT S=3 IT CANNOT BE. Two windows 15 days apart
can carry the same `YYYY-MM` label, so a W/S axis is NOT a unique monthly key
and must never be read as one. `axis_dict()` below is what every consumer
reads instead: it hands `ml/rollout_spatial.py:TimeAxis` a BINNED descriptor
whose step is S source bins (30 d at `6/6`, 15 d at `6/3`), so the horizon
bands, the day-matched leads and the roll's own trajectory labels are derived
from the stride rather than assumed to be months.

WHAT OVERLAP DOES TO PERSISTENCE, stated here because it is a property of the
AXIS and not of any metric. At S < W the persistence baseline — "the previous
embedding" — is built from a window sharing W-S of its W bins with the target,
so persistence is STRONGER BY CONSTRUCTION at stride 3 than at stride 6. That
changes the BASELINE, not the instrument: a forecast ratio at `6/3` is
comparable with another `6/3` ratio and is NOT comparable with a `6/6` one.
Same for the neighbour term the codec trains on (`dt = +/-1` is the adjacent
WINDOW, which overlaps). Every artefact that carries a block axis records the
stride beside the numbers so a reader cannot lose this.
"""
import datetime as dt

import numpy as np

K_MONTH = 7          # a pentad month is 6 or 7 bins; the ragged one sets k_max


def parse_mode(mode):
    """`--time-block` -> ("month", None, None) or ("window", W, S).

    ONE parser, because a mode string is read in five places (both trainers,
    the embed path, the roll and the probe) and a second reading of "6/3" is
    a second place for the same off-by-one. `N` is exactly `N/N`: the fixed-N
    mode E-047 shipped is the non-overlapping case of E-048's window, and
    spelling it as one thing means the axis arithmetic below has one branch
    instead of two that must agree.
    """
    s = str(mode).strip()
    if s == "month":
        return "month", None, None
    if "/" in s:
        w, _, st = s.partition("/")
        parts = (w.strip(), st.strip())
    else:
        parts = (s, s)
    try:
        W, S = int(parts[0]), int(parts[1])
    except ValueError:
        raise ValueError(
            f"--time-block {mode!r}: expected 'month', an integer N (= N/N), "
            f"or 'W/S' — a width-W window advancing by S bins (E-048: '6/6' "
            f"is 30 days non-overlapping, '6/3' is 30 days of input every 15 "
            f"days).")
    if W < 1:
        raise ValueError(f"--time-block {mode!r}: the window width W must be "
                         f">= 1")
    if S < 1:
        raise ValueError(
            f"--time-block {mode!r}: the stride S must be >= 1. S = 0 would "
            f"emit the same window for ever.")
    if S > W:
        # NOT a coherent axis, and the failure is silent: bins between the
        # end of one window and the start of the next enter NO embedding, so
        # the record would carry gaps that nothing downstream can see — while
        # the roll still advances S bins a step and every number looks
        # ordinary. Refuse rather than subsample by accident.
        raise ValueError(
            f"--time-block {mode!r}: stride {S} exceeds width {W}, which "
            f"would leave {S - W} bin(s) between consecutive windows in no "
            f"embedding at all. A gap in the record is invisible downstream: "
            f"refusing rather than silently subsampling the axis.")
    return "window", W, S


class BlockAxis:
    """The grouping of an input axis into blocks, and the labels that result.

    Attributes:
      mode        "month", "N" or "W/S" (the string that was asked for)
      width       W, the bins in one window (None in month mode)
      stride      S, the bins one block advances (None in month mode)
      overlap     W - S, the bins two consecutive windows SHARE (0 or None)
      k_max       cells along the time dimension of one block
      rows        [n_blocks, k_max] int32, the SOURCE row of each cell,
                  padded rows repeat the last real row (never a stray index)
      pad         [n_blocks, k_max] bool, True where the cell is padding
      n_bins      [n_blocks] int, real bins per block (6 or 7 in month mode)
      labels      [n_blocks] str, the block's own label
      centres     [n_blocks] datetime.date, the centre of the block's span
      span_days   [n_blocks] float, the block's span in days
    """

    def __init__(self, mode, months, bin_index=None, epoch=None, days=None):
        self.mode = str(mode)
        self.src_labels = [str(m) for m in months]
        T = len(self.src_labels)
        self.T_src = T
        self.days = float(days) if days else None
        self.epoch = epoch
        self.bin_index = (None if bin_index is None
                          else np.asarray(bin_index).astype(np.int64))
        kind, W, S = parse_mode(self.mode)
        self.width, self.stride = W, S
        self.overlap = None if S is None else W - S
        if kind == "month":
            groups, labels = [], []
            for i, lab in enumerate(self.src_labels):
                if labels and lab == labels[-1]:
                    groups[-1].append(i)
                else:
                    groups.append([i])
                    labels.append(lab)
            # A month label must be CONTIGUOUS on the axis or the grouping is
            # not a grouping: `TimeAxis` already refuses a non-contiguous
            # monthly axis for the same reason, and build_family4's bins are
            # consecutive by construction.
            seen = set()
            for lab in labels:
                if lab in seen:
                    raise ValueError(
                        f"month label {lab!r} appears in two separate runs of "
                        f"the axis: this axis is not time-ordered, and "
                        f"blocking it by label would fuse bins that are years "
                        f"apart")
                seen.add(lab)
            self.k_max = max(K_MONTH, max(len(g) for g in groups))
        else:
            self.k_max = W
            # n_blocks = floor((T - W)/S) + 1 — every COMPLETE window and no
            # partial one, which is fixed-N's own rule (`range(0, T-N+1, N)`)
            # with the step generalised. The trailing remainder is dropped
            # rather than padded for the reason E-047 gives: a partial window
            # at the end of the record would be the only ragged block in the
            # mode whose whole point is that it is not.
            if T < W:
                raise ValueError(
                    f"--time-block {mode!r}: the axis has {T} bins and one "
                    f"window needs {W}. No complete window exists, so this "
                    f"grouping has no blocks at all.")
            groups = [list(range(i, i + W)) for i in range(0, T - W + 1, S)]
            labels = [self.src_labels[g[0]] for g in groups]
            assert len(groups) == (T - W) // S + 1, (len(groups), T, W, S)
        self.n_blocks = len(groups)
        self.labels = labels
        self.n_bins = np.array([len(g) for g in groups], np.int32)
        self.rows = np.zeros((self.n_blocks, self.k_max), np.int32)
        self.pad = np.ones((self.n_blocks, self.k_max), bool)
        for b, g in enumerate(groups):
            for j, r in enumerate(g):
                self.rows[b, j] = r
                self.pad[b, j] = False
            # PAD CELLS POINT AT A REAL ROW (the block's last) rather than at
            # 0 or -1: the values are masked out by `pad` everywhere they are
            # read, and a gather that indexes a valid row cannot produce a
            # silent out-of-bounds or a wrap to the start of the record.
            for j in range(len(g), self.k_max):
                self.rows[b, j] = g[-1]
        self.centres, self.span_days = self._span()

    # -- dates --------------------------------------------------------------
    def _row_start(self, r):
        if self.bin_index is not None:
            return self.epoch + dt.timedelta(
                days=int(self.bin_index[r]) * self.days)
        lab = self.src_labels[r]
        return dt.date(int(lab[:4]), int(lab[5:7]), 1)

    def _row_span(self, r):
        if self.bin_index is not None:
            return self.days
        lab = self.src_labels[r]
        y, m = int(lab[:4]), int(lab[5:7])
        return float((dt.date(y + (m == 12), (m % 12) + 1, 1)
                      - dt.date(y, m, 1)).days)

    def _span(self):
        centres, spans = [], []
        for b in range(self.n_blocks):
            first = int(self.rows[b, 0])
            last = int(self.rows[b, int(self.n_bins[b]) - 1])
            start = self._row_start(first)
            end = self._row_start(last) + dt.timedelta(days=self._row_span(last))
            span = float((end - start).days)
            c = (dt.datetime.combine(start, dt.time())
                 + dt.timedelta(days=span / 2.0))
            centres.append(c)
            spans.append(span)
        return centres, np.array(spans, np.float64)

    def ctx_phase(self):
        """[n_blocks, 2] sin/cos of the CONTINUOUS fraction-of-year phase of
        each block's centre — E-047 point 4, and the reason it is here rather
        than in the trainer: the codec carries the label from birth, so the
        head is never asked to undo a month-quantized staircase the encoder
        baked in. Same definition as ml/temporal.py's `season_feat_of`, which
        this deliberately mirrors rather than re-derives."""
        import math
        out = []
        for c in self.centres:
            frac = ((c - dt.datetime(c.year, 1, 1)).total_seconds()
                    / (365.2425 * 86400.0))
            out.append((math.sin(2 * math.pi * frac),
                        math.cos(2 * math.pi * frac)))
        return np.asarray(out, np.float32)

    # -- what the consumers need -------------------------------------------
    def gather(self, arr):
        """`arr` indexed on the SOURCE axis -> [n_blocks, k_max, ...] on the
        block axis, with padded cells carrying their block's last real value
        (masked by `pad` wherever it is read)."""
        return np.asarray(arr)[self.rows]

    def cell_obs(self, obs_src):
        """[n_blocks, k_max, C] observed mask for the grid, given a
        [T, ..., C] source mask. A PAD cell is unobserved by construction, so
        padding and missing data reach the codec through ONE mechanism — which
        is E-047's point: Argo present in one bin of six is not a special
        case, it is six cells of which five are unobserved."""
        o = np.asarray(obs_src)[self.rows]
        o[self.pad] = False
        return o

    def block_of_row(self, r):
        """Which block OWNS source row r (or None if it was dropped).

        The RAPID truth is keyed on the source row and has to move with the
        axis — the same remap `--time-stride` needed, for the same reason.

        AT S < W A ROW IS IN SEVERAL WINDOWS, so "contains" stops being a
        function and this has to pick one. The loop below writes ascending and
        the last write wins, which makes the owner the LATEST window
        containing r — i.e. the one whose calendar ANCHOR (its first bin, and
        its label) is the nearest anchor at or before r, never more than S-1
        bins earlier. That is the tightest anchoring available and it makes
        the ownership a PARTITION: at S=3 each window owns the 3 bins it
        advanced past (the last window additionally owning its tail), so every
        covered row maps to exactly one block and no truth value is counted
        twice on an overlapping axis."""
        if not hasattr(self, "_owner"):
            own = np.full(self.T_src, -1, np.int64)
            for b in range(self.n_blocks):
                for j in range(int(self.n_bins[b])):
                    own[int(self.rows[b, j])] = b
            self._owner = own
        v = int(self._owner[int(r)])
        return None if v < 0 else v

    def remap_rows(self, rapid):
        """`rapid` [[source_row, value], ...] -> the same on the block axis.
        A row whose block was dropped disappears; several rows of one block
        collapse onto that block, in source order, and the CALLER decides what
        to do with duplicates (the truth is a monthly series, so in month mode
        there is at most one per block in practice).

        UNCHANGED FOR OVERLAPPING WINDOWS, and that is a property of
        `block_of_row` rather than of this function: ownership is a partition
        even at S < W, so a truth row still lands on exactly ONE block and the
        remapped series still carries every row the axis covers. What DOES
        change at S=3 is the density — a 15-day axis has twice as many blocks
        as a 30-day one over the same record, so about half of them carry no
        truth row at all, which is the ordinary "the truth is monthly" case
        one rung finer."""
        rapid = np.asarray(rapid)
        keep, out = [], []
        for i, r in enumerate(rapid[:, 0].astype(int)):
            b = self.block_of_row(r)
            if b is not None:
                keep.append(i)
                out.append(b)
        res = rapid[keep].copy()
        if len(res):
            res[:, 0] = out
        return res

    # -- what the ROLL needs: this axis as a tensor descriptor --------------
    def axis_dict(self):
        """The block axis as the small-npz members `TimeAxis` reads.

        WHY THIS EXISTS. `ml/rollout_spatial.py` used to build the roll's axis
        as `TimeAxis({"months": BLK.labels})`, which takes TimeAxis's MONTHLY
        path — right for month mode (the labels are unique, contiguous
        `YYYY-MM` keys and that is E-047's entire point) and wrong for every
        other mode: a fixed-N or W/S axis's labels repeat, so the monthly path
        refuses, and if it did not it would advance the roll by a MONTH per
        step while the state advances S bins. The step length of a window axis
        is S source bins and nothing in a label says so.

        So a window axis is handed over as a BINNED axis, which is the shape
        TimeAxis already derives a calendar from — `bin_index` + a step length
        + an epoch. Everything day-defined downstream then falls out of the
        stride with no further edit: `step_days` = S x the source bin length
        (30 d at 6/6, 15 d at 6/3), `bands()` cuts at the same DAY edges,
        `daymatched_leads()` returns the same twelve durations, and
        `span_days(h)` prints what one scored horizon actually covers.

        The arithmetic, stated because it is the one place an off-by-one could
        hide. Window b starts at source bin `src_b0 + b*S` (source bins are
        consecutive, asserted below), so its start DATE is
        `epoch + (src_b0 + b*S)*days`. Writing that as a binned axis of step
        `S*days` needs an integer bin index, so the block index is
        `b0 = src_b0 // S` and the leftover `src_b0 % S` source bins are
        carried in a SHIFTED EPOCH. Where `src_b0` divides S the epoch is the
        tensor's own and TimeAxis's `bin_start` cross-check applies unchanged.
        """
        if self.mode == "month":
            # Bit-identical to what the roll built before E-048: a monthly
            # key axis, which is what month mode is FOR.
            return {"months": np.array(self.labels)}
        if self.bin_index is None or self.days is None or self.epoch is None:
            raise ValueError(
                f"--time-block {self.mode!r}: this axis was built from month "
                f"LABELS alone (no `bin_index`/`pentad_days`/`epoch`), so the "
                f"length of one block step is unknown — a window of {self.width} "
                f"labels advancing {self.stride} is not a duration. A window "
                f"codec needs a binned tensor (family 4/5); month mode is the "
                f"one that works on a monthly one.")
        used = self.bin_index[int(self.rows[0, 0]):
                              int(self.rows[-1, int(self.n_bins[-1]) - 1]) + 1]
        if len(used) > 1 and not (np.diff(used) == 1).all():
            raise ValueError(
                "`bin_index` is not consecutive over the bins these blocks "
                "cover, so a block step is not a fixed number of days and "
                "this axis cannot be described as a binned one.")
        step_days = self.stride * self.days
        if abs(step_days - round(step_days)) > 1e-9:
            raise ValueError(
                f"--time-block {self.mode!r}: stride {self.stride} x source bin "
                f"{self.days} d = {step_days} d, which is not a whole number "
                f"of days; TimeAxis counts a step in whole days.")
        step_days = int(round(step_days))
        src_b0 = int(self.bin_index[int(self.rows[0, 0])])
        b0, rem = divmod(src_b0, self.stride)
        epoch = self.epoch + dt.timedelta(days=rem * self.days)
        return {"months": np.array(self.labels),
                "bin_index": np.arange(b0, b0 + self.n_blocks, dtype=np.int64),
                "pentad_days": np.array(step_days),
                "epoch": np.array(str(epoch)),
                "cadence": np.array(f"{step_days}-day")}

    def head_season(self, mode="month"):
        """[n_blocks, 2] season features for a HEAD reading this axis.

        `month` is the archived token, sin/cos(2*pi*(month-1)/12) of the
        block's LABEL — bit-identical to what `ml/temporal.py:season_ctx`
        computes from the same labels, which is why the caller asserts the
        equality rather than trusting this sentence.

        `fine` is `ctx_phase()`: the continuous fraction-of-year phase of the
        block's own CENTRE. On a 15-day axis the two differ by more than a
        rounding — two windows a fortnight apart can carry the SAME month
        label, so the month token is literally constant across a step the
        axis takes, and only the continuous phase says the state moved."""
        if mode == "month":
            moy = np.array([int(m[5:7]) - 1 for m in self.labels])
            return np.stack([np.sin(2 * np.pi * moy / 12),
                             np.cos(2 * np.pi * moy / 12)], 1)
        if mode != "fine":
            raise ValueError(f"head_season: unknown mode {mode!r}")
        return self.ctx_phase()

    def describe(self, C, d_z):
        """The sizing line E-047 asks for at startup. Cost is quoted in ENCODER
        TOKENS because that is what the attention pays for: one block is
        k_max*C value tokens against a per-bin codec's C, and the sequence
        also carries the cls and ctx tokens at both."""
        tok = self.k_max * C + 2
        tok1 = C + 2
        step = ""
        if self.stride is not None:
            step = (f" · width {self.width} stride {self.stride} "
                    f"(overlap {self.overlap} bins"
                    + (f", step {self.stride * self.days:g} d" if self.days
                       else "") + ")")
            if self.overlap:
                # Said at STARTUP, not only in the plan: the number a reader
                # is most likely to misread on an overlapping axis is the
                # persistence baseline, and it is misread in the flattering
                # direction.
                step += (f" · PERSISTENCE IS STRONGER BY CONSTRUCTION here: "
                         f"the previous window shares {self.overlap} of its "
                         f"{self.width} bins with this one, so a forecast "
                         f"ratio on this axis is comparable only with another "
                         f"ratio at the same stride")
        return (f"time blocks: mode {self.mode!r}{step} · {self.n_blocks} blocks "
                f"from {self.T_src} bins ({self.labels[0]}..{self.labels[-1]}) "
                f"· k_max {self.k_max} · bins/block "
                f"{int(self.n_bins.min())}-{int(self.n_bins.max())} · pad "
                f"cells {int(self.pad.sum()):,}/{self.pad.size:,} "
                f"({100.0 * self.pad.mean():.1f}%) · {tok} encoder tokens per "
                f"block against {tok1} per bin ({tok / tok1:.2f}x per forward, "
                f"and {self.n_blocks / self.T_src:.3f}x as many forwards, so "
                f"~{tok / tok1 * self.n_blocks / self.T_src:.2f}x the encoder "
                f"work per pass) · z stays [n_blocks, P, {d_z}]")

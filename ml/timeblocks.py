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

TWO MODES.

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
            whole point is that it is not).

WHAT A BLOCK IS CALLED. Month mode labels the block `YYYY-MM`, which makes the
block axis indistinguishable from a monthly tensor's to everything downstream.
Fixed-N mode labels it with its first bin's own label. Either way the label is
what `TimeAxis` will parse, so it has to be the label of a real calendar
position and not an index.
"""
import datetime as dt

import numpy as np

K_MONTH = 7          # a pentad month is 6 or 7 bins; the ragged one sets k_max


class BlockAxis:
    """The grouping of an input axis into blocks, and the labels that result.

    Attributes:
      mode        "month" or "N" (the string that was asked for)
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
        if self.mode == "month":
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
            n = int(self.mode)
            if n < 1:
                raise ValueError(f"--time-block {mode!r}: N must be >= 1")
            self.k_max = n
            groups = [list(range(i, i + n)) for i in range(0, T - n + 1, n)]
            labels = [self.src_labels[g[0]] for g in groups]
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
        """Which block CONTAINS source row r (or None if it was dropped).
        The RAPID truth is keyed on the source row and has to move with the
        axis — the same remap `--time-stride` needed, for the same reason."""
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
        there is at most one per block in practice)."""
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

    def describe(self, C, d_z):
        """The sizing line E-047 asks for at startup. Cost is quoted in ENCODER
        TOKENS because that is what the attention pays for: one block is
        k_max*C value tokens against a per-bin codec's C, and the sequence
        also carries the cls and ctx tokens at both."""
        tok = self.k_max * C + 2
        tok1 = C + 2
        return (f"time blocks: mode {self.mode!r} · {self.n_blocks} blocks "
                f"from {self.T_src} bins ({self.labels[0]}..{self.labels[-1]}) "
                f"· k_max {self.k_max} · bins/block "
                f"{int(self.n_bins.min())}-{int(self.n_bins.max())} · pad "
                f"cells {int(self.pad.sum()):,}/{self.pad.size:,} "
                f"({100.0 * self.pad.mean():.1f}%) · {tok} encoder tokens per "
                f"block against {tok1} per bin ({tok / tok1:.2f}x per forward, "
                f"and {self.n_blocks / self.T_src:.3f}x as many forwards, so "
                f"~{tok / tok1 * self.n_blocks / self.T_src:.2f}x the encoder "
                f"work per pass) · z stays [n_blocks, P, {d_z}]")

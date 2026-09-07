#!/usr/bin/env python3
"""E-074a — the ladder bake-off, on the data alone (no GPU, no model).

Fits per-channel monotone warps ("ladders") on family 7's training pentads and
scores them as descriptions of held-out pentads.  Family 7 is the first input
tensor covering the whole globe at 0.25 deg and five-day steps
(docs/FAMILY7_DATA_HANDOVER.md); a "ladder" is a set of bin edges placed so the
density of bin boundaries is proportional to p(x)**alpha, where alpha = 1 gives
equal-probability (quantile) bins, alpha = 1/3 the mean-squared-error-optimal
placement and alpha = 0 plain uniform bins.

The plan is ml/plans/E074_hierarchical_channel_quantization.md, section 6.

Everything is streamed: the 45.7 GB g025 array is never downloaded, only one
pentad at a time by HTTP range request (status 206 asserted).  Peak resident
memory is a few hundred MB.

Usage
-----
    python3 ml/ladder_bakeoff.py                    # the real experiment
    python3 ml/ladder_bakeoff.py --smoke            # the decimated fixture, < 1 min
    python3 ml/ladder_bakeoff.py --out DIR          # where the two result files go
"""

from __future__ import annotations

import argparse
import io
import json
import math
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import date, timedelta

import numpy as np

# --------------------------------------------------------------------------
# constants (all of them from the plan; none is hand-tuned mid-run)
# --------------------------------------------------------------------------

EPOCH = date(1982, 1, 1)
PENTAD_DAYS = 5
HOLDOUT_YEARS = (2009, 2017, 2023)
TERMINAL_TRAIN_LAST_YEAR = 2020

ALPHAS = (0.0, 1.0 / 3.0, 0.5, 1.0)
ALPHA_NAMES = ("0", "1/3", "1/2", "1")
K1S = (16, 64, 256)
K2 = 64
K1_MATCHED = 4096          # depth-1 rung used only for falsifier F4
ALPHAS_MATCHED = (0.0, 1.0)

N_FINE = 4096              # points on the fine quantile grid = the common partition
CLIP_Q = (0.0005, 0.9995)
SUBSAMPLE = 0.25           # spatial subsample of the fit bins, to bound memory
SEED = 0
FIT_PAIRS = 60
EVAL_PAIRS = 40
RG_FIT_PAIRS = 24
RG_EVAL_PAIRS = 12
FIT_CAP = 20_000_000       # hard cap on fit values per channel (deterministic thin)
GMM_CAP = 200_000          # values the 3-component Gaussian mixture is fitted on
ESCAPE = 1e-6              # uniform-over-cells escape mixed into EVERY scheme
TAIL_Q = 0.99              # "top 1 % of cur_speed on that bin"

HUB_BASE = ("https://huggingface.co/datasets/chfrank/earth-tensors/resolve/main/"
            "tensors/family7_global025_pentad_l0/")
STEM = "family7_global025_pentad_l0"
META_URL = HUB_BASE + STEM + ".npz"

GROUPS = ("g025", "g100", "rg100")
SHAPES = {"g025": (3142, 721, 1440, 7),
          "g100": (3142, 181, 360, 15),
          "rg100": (252, 181, 360, 32)}
HEADER_LEN = 128

FIXTURE_DIR = "data/family7/fixture"


# --------------------------------------------------------------------------
# time axis
# --------------------------------------------------------------------------

def bin_start(b: int) -> date:
    return EPOCH + timedelta(days=PENTAD_DAYS * int(b))


def bin_year(b: int) -> int:
    return bin_start(b).year


def is_training_year(y: int) -> bool:
    return y <= TERMINAL_TRAIN_LAST_YEAR and y not in HOLDOUT_YEARS


# --------------------------------------------------------------------------
# stratified sampling of consecutive pairs (deterministic, no RNG)
# --------------------------------------------------------------------------

def allocate(total: int, groups: int) -> list:
    """Split `total` as evenly as possible over `groups`, deterministically."""
    base, rem = divmod(total, groups)
    out = [base] * groups
    if rem:
        # give the remainder to evenly spread group indices
        idx = np.floor(np.linspace(0, groups - 1e-9, rem)).astype(int)
        for i in idx:
            out[int(i)] += 1
    return out


def pick_evenly(items: list, n: int) -> list:
    """n items from `items`, evenly spread, deterministic.

    The offsets are the MIDPOINTS (j + 0.5)/n of n equal slices, not the
    endpoints. With endpoints and n = 2 per year, "evenly spread across the
    year" picks the first and last candidate — 1 January and 22 December, both
    the same season — and the fit sample ends up seasonally degenerate. That
    was measured before this rule was written: 3.1 % of held-out `t2m` values
    fell below the fit sample's 0.05 % quantile, because Antarctica's winter
    was in the held-out set and in no fit bin at all."""
    if n <= 0 or not items:
        return []
    n = min(n, len(items))
    idx = np.floor((np.arange(n) + 0.5) / n * len(items)).astype(int)
    idx = np.clip(idx, 0, len(items) - 1)
    return [items[int(i)] for i in dict.fromkeys(idx.tolist())]


def spread_pairs(candidates: list, total: int) -> list:
    """`total` pairs spread evenly over the WHOLE ordered candidate list.

    This is what "stratified across training years, about 1.6 per year, spread
    evenly" has to mean when the per-year quota is 1 or 2: an even spread over
    the span gives the right per-year count AND lets the sampling phase
    precess through the seasons instead of pinning it to one time of year."""
    return sorted(pick_evenly(sorted(candidates), total))


def stratified_pairs(candidates, key, total):
    """Pick `total` pairs from `candidates`, stratified over key(pair)."""
    buckets = {}
    for c in candidates:
        buckets.setdefault(key(c), []).append(c)
    keys = sorted(buckets)
    if not keys:
        return []
    quota = allocate(total, len(keys))
    out = []
    shortfall = 0
    for k, q in zip(keys, quota):
        got = pick_evenly(buckets[k], q)
        shortfall += q - len(got)
        out.extend(got)
    # redistribute any shortfall over the keys that still have spare candidates
    if shortfall:
        spare = []
        chosen = set(out)
        for k in keys:
            spare.extend([c for c in buckets[k] if c not in chosen])
        out.extend(pick_evenly(spare, shortfall))
    return sorted(out)


def pentad_pair_sets():
    """FIT and EVAL consecutive-bin pairs on the pentad axis."""
    n_bins = SHAPES["g025"][0]
    train = [b for b in range(n_bins) if is_training_year(bin_year(b))]
    train_set = set(train)
    cand_fit = [b for b in train if (b + 1) in train_set
                and bin_year(b) == bin_year(b + 1)]
    fit = spread_pairs(cand_fit, FIT_PAIRS)

    held = [b for b in range(n_bins) if bin_year(b) in HOLDOUT_YEARS]
    held_set = set(held)
    cand_ev = [b for b in held if (b + 1) in held_set
               and bin_year(b) == bin_year(b + 1)]
    ev = stratified_pairs(cand_ev, bin_year, EVAL_PAIRS)
    return fit, ev


def _month_hist(bins):
    h = {f"{m:02d}": 0 for m in range(1, 13)}
    for b in bins:
        h[f"{bin_start(b).month:02d}"] += 1
    return h


def _year_counts(bins):
    c = {}
    for b in bins:
        c[str(bin_year(b))] = c.get(str(bin_year(b)), 0) + 1
    return dict(sorted(c.items()))


def rg_pair_sets(rg_months):
    """FIT and EVAL consecutive-MONTH-row pairs for the rg100 group."""
    years = np.array([int(str(m)[:4]) for m in rg_months])
    n = len(years)
    tr = [r for r in range(n) if is_training_year(int(years[r]))]
    tr_set = set(tr)
    cand_fit = [r for r in tr if (r + 1) in tr_set]
    fit = spread_pairs(cand_fit, RG_FIT_PAIRS)

    hd = [r for r in range(n) if int(years[r]) in HOLDOUT_YEARS]
    hd_set = set(hd)
    cand_ev = [r for r in hd if (r + 1) in hd_set and years[r] == years[r + 1]]
    ev = stratified_pairs(cand_ev, lambda r: int(years[r]), RG_EVAL_PAIRS)
    return fit, ev


# --------------------------------------------------------------------------
# slab readers
# --------------------------------------------------------------------------

class RangeReader:
    """One pentad (or month row) of one group, by HTTP range request."""

    def __init__(self, group):
        import requests
        self.requests = requests
        self.group = group
        T, H, W, C = SHAPES[group]
        self.shape = (H, W, C)
        self.slab = H * W * C * 2
        self.url = HUB_BASE + f"{STEM}_X_{group}.npy"
        self._tl = threading.local()

    def _session(self):
        s = getattr(self._tl, "s", None)
        if s is None:
            s = self._tl.s = self.requests.Session()
        return s

    def raw(self, row):
        lo = HEADER_LEN + int(row) * self.slab
        hi = lo + self.slab - 1
        last = None
        for attempt in range(4):
            try:
                r = self._session().get(self.url, headers={"Range": f"bytes={lo}-{hi}"},
                                        timeout=240)
                if r.status_code != 206:
                    raise RuntimeError(
                        f"expected HTTP 206 for a Range read of {self.group} row {row}, "
                        f"got {r.status_code} — the host ignored the Range header")
                buf = r.content
                if len(buf) != self.slab:
                    raise RuntimeError(f"short slab: {len(buf)} != {self.slab}")
                return buf
            except Exception as exc:            # transient network, retry
                last = exc
                time.sleep(1.5 * (attempt + 1))
        raise last

    def decode(self, buf):
        return np.frombuffer(buf, dtype="<f2").reshape(self.shape).astype(np.float32)

    def read(self, row):
        return self.decode(self.raw(row))


class LocalReader:
    """The same interface over a local .npy (the fixture, or a downloaded group)."""

    def __init__(self, group, path):
        self.group = group
        self.arr = np.load(path, mmap_mode="r")
        self.shape = self.arr.shape[1:]
        self.slab = int(np.prod(self.shape)) * 2

    def raw(self, row):
        return np.asarray(self.arr[int(row)], dtype="<f2").tobytes()

    def decode(self, buf):
        return np.frombuffer(buf, dtype="<f2").reshape(self.shape).astype(np.float32)

    def read(self, row):
        return np.asarray(self.arr[int(row)], dtype=np.float32)


def stream_rows(readers, rows, workers=8):
    """Yield (row, {group: frame}) in order, prefetching raw bytes in parallel."""
    rows = list(rows)
    if not rows:
        return
    if all(isinstance(r, LocalReader) for r in readers.values()):
        for row in rows:
            yield row, {g: rd.read(row) for g, rd in readers.items()}
        return
    with ThreadPoolExecutor(workers) as ex:
        pending = []
        depth = max(2, workers)
        it = iter(range(len(rows)))
        for _ in range(min(depth, len(rows))):
            i = next(it)
            pending.append((rows[i], {g: ex.submit(rd.raw, rows[i])
                                      for g, rd in readers.items()}))
        while pending:
            row, futs = pending.pop(0)
            frames = {g: readers[g].decode(f.result()) for g, f in futs.items()}
            yield row, frames
            try:
                i = next(it)
            except StopIteration:
                continue
            pending.append((rows[i], {g: ex.submit(rd.raw, rows[i])
                                      for g, rd in readers.items()}))


# --------------------------------------------------------------------------
# the warp, the ladders, and the common fine partition
# --------------------------------------------------------------------------

def quantile_sorted(sorted_x, ps):
    """Linear-interpolated quantiles of an already-sorted 1-D array."""
    n = len(sorted_x)
    pos = np.asarray(ps, dtype=np.float64) * (n - 1)
    lo = np.floor(pos).astype(np.int64)
    hi = np.minimum(lo + 1, n - 1)
    frac = pos - lo
    return sorted_x[lo] * (1.0 - frac) + sorted_x[hi] * frac


def fine_partition(sorted_x):
    """The COMMON evaluation partition E: 4,096 equal-probability quantiles of
    the fit sample, clipped to [q_0.0005, q_0.9995], deduplicated (the stored
    values are float16, so quantiles can coincide), plus two open end cells."""
    ps = np.linspace(CLIP_Q[0], CLIP_Q[1], N_FINE)
    v = np.unique(quantile_sorted(sorted_x, ps).astype(np.float64))
    if len(v) < 4:
        return None
    n = len(sorted_x)
    c = np.searchsorted(sorted_x, v, side="left").astype(np.float64)
    mass = np.diff(c) / n                      # fit mass of each finite cell
    width = np.diff(v)
    dens = np.where(width > 0, mass / np.maximum(width, 1e-300), 0.0)
    return dict(v=v, mass=mass, dens=dens, width=width,
                clip_lo=float(v[0]), clip_hi=float(v[-1]),
                n_cells=len(v) + 1, n_edges=len(v))


def warp_edges(part, alpha, K1):
    """Interior edges of a depth-1 ladder whose point density is p(x)**alpha."""
    v, dens, width = part["v"], part["dens"], part["width"]
    w = np.power(np.maximum(dens, 0.0), alpha) * width
    tot = w.sum()
    if not np.isfinite(tot) or tot <= 0:
        return None
    G = np.concatenate([[0.0], np.cumsum(w) / tot])
    G[-1] = 1.0
    G = np.maximum.accumulate(G)
    targets = np.arange(1, K1) / float(K1)
    # np.interp needs strictly increasing xp; nudge exact ties
    Gm = G + np.arange(len(G)) * 1e-15
    return np.interp(targets, Gm, v)


def build_pieces(edges, sorted_x, depth, tail_exception):
    """Breakpoints of the quantizer, from the fit sample's extremes inward.

    The two outermost bins are OPEN — they absorb anything beyond the clip
    range — but for a width, a centre and a within-bin subdivision they need a
    finite span, which is the clip edge out to the fit sample's extreme."""
    lo, hi = float(sorted_x[0]), float(sorted_x[-1])
    B = np.concatenate([[lo], np.asarray(edges, dtype=np.float64), [hi]])
    B = np.maximum.accumulate(B)
    if depth == 1:
        return B
    K1 = len(B) - 1
    out = [B[0]]
    for k in range(K1):
        a, b = B[k], B[k + 1]
        is_end = (k == 0 or k == K1 - 1)
        if tail_exception and is_end:
            i0 = np.searchsorted(sorted_x, a, side="left")
            i1 = np.searchsorted(sorted_x, b, side="right")
            inside = sorted_x[i0:i1]
            if len(inside) >= K2:
                sub = quantile_sorted(inside, np.arange(1, K2) / float(K2))
                sub = np.clip(np.asarray(sub, dtype=np.float64), a, b)
                out.extend(list(sub))
                out.append(b)
                continue
        out.extend(list(a + (b - a) * np.arange(1, K2) / float(K2)))
        out.append(b)
    B2 = np.asarray(out, dtype=np.float64)
    return np.maximum.accumulate(B2)


def piece_masses(B, sorted_x):
    """pi_k — the fit sample's empirical frequency in each piece."""
    n = len(sorted_x)
    c = np.searchsorted(sorted_x, B, side="left").astype(np.float64)
    c[0], c[-1] = 0.0, float(n)
    pi = np.diff(c) / n
    pi = np.maximum(pi, 0.0)
    s = pi.sum()
    return pi / s if s > 0 else pi


def cell_probs_from_cdf(F_at_edges):
    """Fine-cell probabilities from a cumulative distribution evaluated at the
    partition's edges: the two open end cells plus the finite cells between."""
    p = np.empty(len(F_at_edges) + 1, dtype=np.float64)
    p[0] = F_at_edges[0]
    p[1:-1] = np.diff(F_at_edges)
    p[-1] = 1.0 - F_at_edges[-1]
    return np.maximum(p, 0.0)


def ladder_cell_probs(part, B, pi):
    """The probability a piecewise-uniform-in-x ladder assigns to each fine cell."""
    cum = np.concatenate([[0.0], np.cumsum(pi)])
    cum[-1] = 1.0
    Bm = B + np.arange(len(B)) * 1e-15
    F = np.interp(part["v"], Bm, cum, left=0.0, right=1.0)
    return cell_probs_from_cdf(F)


def regularise(p, n_cells):
    """One escape mixture, applied identically to EVERY scheme, so that no
    scheme can score infinity for a cell it happens to give zero mass."""
    q = (1.0 - ESCAPE) * p + ESCAPE / n_cells
    return q / q.sum()


# --------------------------------------------------------------------------
# scheme construction per channel
# --------------------------------------------------------------------------

def scheme_grid():
    out = []
    for ai, (a, an) in enumerate(zip(ALPHAS, ALPHA_NAMES)):
        for K1 in K1S:
            out.append(dict(kind="ladder", alpha=a, alpha_name=an, K1=K1,
                            depth=1, tail_exception=False,
                            name=f"a{an}_K{K1}_d1"))
            for te in (False, True):
                out.append(dict(kind="ladder", alpha=a, alpha_name=an, K1=K1,
                                depth=2, tail_exception=te,
                                name=f"a{an}_K{K1}_d2" + ("_tailq" if te else "")))
    for a, an in zip(ALPHAS_MATCHED, ("0", "1")):
        out.append(dict(kind="ladder", alpha=a, alpha_name=an, K1=K1_MATCHED,
                        depth=1, tail_exception=False,
                        name=f"a{an}_K{K1_MATCHED}_d1"))
    out.append(dict(kind="gauss", name="gaussian"))
    out.append(dict(kind="gmm3", name="gmm3"))
    return out


SCHEMES = scheme_grid()
LADDER_SCHEMES = [s for s in SCHEMES if s["kind"] == "ladder"]


def fit_channel(sorted_x, want_gmm=True):
    """Everything that is fitted on the FIT sample of one channel."""
    part = fine_partition(sorted_x)
    if part is None:
        return None
    fitted = {}
    for s in SCHEMES:
        if s["kind"] == "ladder":
            e = warp_edges(part, s["alpha"], s["K1"])
            if e is None:
                continue
            B = build_pieces(e, sorted_x, s["depth"], s["tail_exception"])
            pi = piece_masses(B, sorted_x)
            widths = np.diff(B)
            centres = 0.5 * (B[:-1] + B[1:])
            p = ladder_cell_probs(part, B, pi)
            fitted[s["name"]] = dict(spec=s, B=B, pi=pi, widths=widths,
                                     centres=centres,
                                     logp=np.log(regularise(p, part["n_cells"])),
                                     interior_edges=e,
                                     n_empty_pieces=int((pi <= 0).sum()))
        elif s["kind"] == "gauss":
            from scipy.stats import norm
            mu, sd = float(np.mean(sorted_x)), float(np.std(sorted_x))
            sd = max(sd, 1e-12)
            F = norm.cdf((part["v"] - mu) / sd)
            fitted[s["name"]] = dict(spec=s, mu=mu, sd=sd,
                                     logp=np.log(regularise(
                                         cell_probs_from_cdf(F), part["n_cells"])))
        elif s["kind"] == "gmm3" and want_gmm:
            try:
                from sklearn.mixture import GaussianMixture
                from scipy.stats import norm
            except Exception:
                continue
            x = sorted_x
            if len(x) > GMM_CAP:
                step = len(x) / float(GMM_CAP)
                x = x[np.floor(np.arange(GMM_CAP) * step).astype(np.int64)]
            gm = GaussianMixture(n_components=3, random_state=SEED,
                                 covariance_type="full", reg_covar=1e-8,
                                 max_iter=300).fit(x.reshape(-1, 1).astype(np.float64))
            w = gm.weights_.ravel()
            mus = gm.means_.ravel()
            sds = np.sqrt(gm.covariances_.ravel())
            F = np.zeros(len(part["v"]))
            for wi, mi, si in zip(w, mus, sds):
                F += wi * norm.cdf((part["v"] - mi) / max(si, 1e-12))
            fitted[s["name"]] = dict(spec=s,
                                     weights=w.tolist(), means=mus.tolist(),
                                     sds=sds.tolist(),
                                     logp=np.log(regularise(
                                         cell_probs_from_cdf(F), part["n_cells"])))
    # uniform-over-the-common-partition reference
    p = np.full(part["n_cells"], 1.0 / part["n_cells"])
    fitted["uniform_cells"] = dict(spec=dict(kind="uniform", name="uniform_cells"),
                                   logp=np.log(p))
    return dict(part=part, fitted=fitted,
                fit_n=int(len(sorted_x)),
                fit_min=float(sorted_x[0]), fit_max=float(sorted_x[-1]),
                fit_mean=float(np.mean(sorted_x)), fit_sd=float(np.std(sorted_x)))


# --------------------------------------------------------------------------
# accumulators
# --------------------------------------------------------------------------

MASKS = ("all", "tail", "rapid")


class ChannelAcc:
    def __init__(self, model, n_alpha_k):
        self.model = model
        n_cells = model["part"]["n_cells"]
        self.hist = np.zeros(n_cells, dtype=np.int64)
        self.n = {m: 0 for m in MASKS}
        self.sse = {name: {m: 0.0 for m in MASKS} for name in model["fitted"]
                    if "B" in model["fitted"][name]}
        self.floor = {name: {m: 0.0 for m in MASKS} for name in self.sse}
        self.pcount = {name: np.zeros(len(model["fitted"][name]["pi"]), dtype=np.int64)
                       for name in self.sse
                       if model["fitted"][name]["spec"]["depth"] == 1}
        self.clipped_lo = 0
        self.clipped_hi = 0
        self.total = 0
        self.joint = {}          # (alpha_name, K1) -> KxK int64
        for an in ALPHA_NAMES:
            for K1 in K1S:
                self.joint[(an, K1)] = np.zeros((K1, K1), dtype=np.int64)
        self.pair_cells = 0

    def add_bin(self, x, sub_tail, sub_rapid):
        """x: finite held-out values of this channel in one bin (stored units)."""
        if x.size == 0:
            return
        part = self.model["part"]
        lo, hi = part["clip_lo"], part["clip_hi"]
        self.total += x.size
        self.clipped_lo += int((x < lo).sum())
        self.clipped_hi += int((x > hi).sum())
        xc = np.clip(x, lo, hi)
        ci = np.searchsorted(part["v"], xc, side="right")
        ci = np.minimum(ci, len(part["v"]) - 1)
        self.hist += np.bincount(ci, minlength=part["n_cells"])
        sel = {"all": slice(None), "tail": sub_tail, "rapid": sub_rapid}
        for m in MASKS:
            s = sel[m]
            self.n[m] += int(x.size if m == "all" else int(np.count_nonzero(s)))
        for name, f in self.model["fitted"].items():
            if "B" not in f:
                continue
            B = f["B"]
            pi_idx = np.searchsorted(B, x, side="right") - 1
            np.clip(pi_idx, 0, len(B) - 2, out=pi_idx)
            err2 = (x - f["centres"][pi_idx]) ** 2
            w2 = f["widths"][pi_idx] ** 2
            if name in self.pcount:
                self.pcount[name] += np.bincount(pi_idx, minlength=len(f["pi"]))
            for m in MASKS:
                s = sel[m]
                if m == "all":
                    self.sse[name][m] += float(err2.sum())
                    self.floor[name][m] += float(w2.sum())
                else:
                    if np.count_nonzero(s):
                        self.sse[name][m] += float(err2[s].sum())
                        self.floor[name][m] += float(w2[s].sum())

    def add_pair(self, x0, x1):
        """Coarse-digit joint counts over cells finite in BOTH bins of a pair."""
        if x0.size == 0:
            return
        self.pair_cells += int(x0.size)
        for an in ALPHA_NAMES:
            for K1 in K1S:
                f = self.model["fitted"].get(f"a{an}_K{K1}_d1")
                if f is None:
                    continue
                B = f["B"]
                d0 = np.clip(np.searchsorted(B, x0, side="right") - 1, 0, K1 - 1)
                d1 = np.clip(np.searchsorted(B, x1, side="right") - 1, 0, K1 - 1)
                self.joint[(an, K1)] += np.bincount(d0 * K1 + d1,
                                                    minlength=K1 * K1
                                                    ).reshape(K1, K1)


# --------------------------------------------------------------------------
# read-outs
# --------------------------------------------------------------------------

def entropies(N):
    tot = N.sum()
    if tot == 0:
        return dict(n=0, persistence=None, h_cond=None, h_marg=None, ratio=None)
    P = N / tot
    row = P.sum(1)
    h_marg = float(-np.sum(row[row > 0] * np.log(row[row > 0])))
    col = P.sum(0)
    h_joint = float(-np.sum(P[P > 0] * np.log(P[P > 0])))
    h_cond = h_joint - h_marg                     # H(d1(t+1) | d1(t))
    h_next = float(-np.sum(col[col > 0] * np.log(col[col > 0])))
    pers = float(np.trace(N) / tot)
    return dict(n=int(tot), persistence=pers, h_cond=h_cond,
                h_marg=h_marg, h_marg_next=h_next,
                ratio=(h_cond / h_marg) if h_marg > 0 else None)


def channel_readouts(acc, sd_phys, unit):
    m = acc.model
    part = m["part"]
    out = dict(unit=unit, sd_used_for_physical_units=sd_phys,
               n_fine_cells=int(part["n_cells"]),
               n_fine_edges=int(part["n_edges"]),
               clip_lo_z=part["clip_lo"], clip_hi_z=part["clip_hi"],
               fit_n=m["fit_n"], fit_min_z=m["fit_min"], fit_max_z=m["fit_max"],
               heldout_n=int(acc.total),
               heldout_n_tail=int(acc.n["tail"]), heldout_n_rapid=int(acc.n["rapid"]),
               clipped_fraction=(float(acc.clipped_lo + acc.clipped_hi) / acc.total
                                 if acc.total else None),
               clipped_fraction_low=(float(acc.clipped_lo) / acc.total
                                     if acc.total else None),
               clipped_fraction_high=(float(acc.clipped_hi) / acc.total
                                      if acc.total else None),
               schemes={})
    hist = acc.hist
    tot = hist.sum()
    for name, f in m["fitted"].items():
        rec = {}
        if tot:
            rec["ce_nats"] = float(-np.sum(hist * f["logp"]) / tot)
        else:
            rec["ce_nats"] = None
        if "B" in f:
            spec = f["spec"]
            rec.update(alpha=spec["alpha"], alpha_name=spec["alpha_name"],
                       K1=spec["K1"], depth=spec["depth"],
                       tail_exception=spec["tail_exception"],
                       n_pieces=int(len(f["pi"])),
                       n_empty_pieces=f["n_empty_pieces"])
            for mk in MASKS:
                n = acc.n[mk]
                rec[f"rmse_{mk}"] = (math.sqrt(acc.sse[name][mk] / n) * sd_phys
                                     if n else None)
                rec[f"floor_{mk}"] = (math.sqrt(acc.floor[name][mk] / (12.0 * n))
                                      * sd_phys if n else None)
            if name in acc.pcount:
                c = acc.pcount[name].astype(np.float64)
                if c.sum() > 0:
                    fr = c / c.sum()
                    d = np.abs(fr - f["pi"])
                    rec["calib_max_abs_dev"] = float(d.max())
                    rec["calib_mean_abs_dev"] = float(d.mean())
        out["schemes"][name] = rec

    out["persistence"] = {}
    for (an, K1), N in acc.joint.items():
        out["persistence"][f"a{an}_K{K1}"] = entropies(N)
    return out


# --------------------------------------------------------------------------
# masks
# --------------------------------------------------------------------------

def rapid_indices(ny, nx, lat0, lon0, step):
    row = int(round((26.5 - lat0) / step))
    row = min(max(row, 0), ny - 1)
    c_lo = int(math.ceil((-80.0 - lon0) / step - 1e-9))
    c_hi = int(math.floor((-13.0 - lon0) / step + 1e-9))
    c_lo, c_hi = max(c_lo, 0), min(c_hi, nx - 1)
    return row, c_lo, c_hi


def coarse_of(y, x, ratio, ny1, nx1):
    y1 = np.minimum(np.floor(y / ratio + 0.5).astype(np.int64), ny1 - 1)
    x1 = np.mod(np.floor(x / ratio + 0.5).astype(np.int64), nx1)
    return y1, x1


# --------------------------------------------------------------------------
# the experiment
# --------------------------------------------------------------------------

def load_meta(cache_dir, smoke):
    if smoke:
        return None
    p = os.path.join(cache_dir, "meta.npz")
    if not os.path.exists(p):
        import requests
        os.makedirs(cache_dir, exist_ok=True)
        r = requests.get(META_URL, timeout=600)
        r.raise_for_status()
        with open(p, "wb") as fh:
            fh.write(r.content)
    return np.load(p, allow_pickle=True)


def run(args):
    t_all = time.time()
    timings = {}
    rng = np.random.default_rng(SEED)
    smoke = args.smoke

    if smoke:
        idx = json.load(open(os.path.join(args.fixture, "family7_index.json")))
        readers = {g: LocalReader(g, os.path.join(args.fixture,
                                                  f"{STEM}_X_{g}.npy"))
                   for g in GROUPS}
        chans = {g: list(idx["groups"][g]["chans"]) for g in GROUPS}
        units = {g: idx["groups"][g]["units"] for g in GROUPS}
        norms = {g: np.asarray(idx["groups"][g]["norm"], dtype=np.float64)
                 for g in GROUPS}
        grids = {g: idx["groups"][g]["grid"] for g in GROUPS}
        nb = idx["groups"]["g025"]["n_bins"]
        # the fixture's .npy holds only its own five rows, so rows are LOCAL
        allb = list(range(nb))
        fit_pairs = allb[:-2]
        eval_pairs = allb[-2:-1]
        rg_n = idx["groups"]["rg100"]["n_bins"]
        rg_fit = list(range(max(rg_n - 1, 0)))
        rg_ev = list(range(max(rg_n - 1, 0)))
        rg_months = [f"fixture-{i}" for i in range(rg_n)]
        rg_single_row = (rg_n == 1)
        sampling_note = ("SMOKE: the decimated fixture holds five consecutive "
                         "pentads of ONE year, so the training/held-out year "
                         "split cannot apply. The first three pairs are the "
                         "fit set and the fourth is the eval set, and they "
                         "share one bin — the fixture's ocean channels are NaN "
                         "in its first three rows, so a disjoint split would "
                         "leave them with no fit sample at all. These numbers "
                         "exercise the pipeline; they measure nothing.")
    else:
        meta = load_meta(args.cache, False)
        chans = {g: [str(s) for s in meta["chan_" + g]] for g in GROUPS}
        norms = {g: np.asarray(meta["norm_" + g], dtype=np.float64) for g in GROUPS}
        units = {"g025": dict(zip(chans["g025"],
                                  ["m/s", "log10 m", "m", "m/s", "m/s", "degC",
                                   "fraction"])),
                 "g100": dict(zip(chans["g100"],
                                  ["N/m2", "N/m2", "N/m2", "N/m2", "degC", "m/s",
                                   "m/s", "hPa", "log1p mm/day", "log1p mm",
                                   "fraction", "degC", "W/m2", "W/m2", "degC"])),
                 "rg100": {c: ("degC" if c.startswith("rg_t") else "PSU")
                           for c in chans["rg100"]}}
        grids = {"g025": dict(ny=721, nx=1440, lat0=-90.0, lon0=-180.0, step=0.25),
                 "g100": dict(ny=181, nx=360, lat0=-90.0, lon0=-180.0, step=1.0),
                 "rg100": dict(ny=181, nx=360, lat0=-90.0, lon0=-180.0, step=1.0)}
        readers = {g: RangeReader(g) for g in ("g025", "g100")}
        fit_pairs, eval_pairs = pentad_pair_sets()
        rg_months = [str(s) for s in meta["rg_months"]]
        rg_fit, rg_ev = rg_pair_sets(rg_months)
        rg_single_row = False
        sampling_note = ("Training years: bin opening year <= 2020 and not in "
                         "{2009, 2017, 2023}. Held-out years: {2009, 2017, 2023}.")
        # rg100 is only 1.05 GB and its rows are MONTHS, so it is fetched whole.
        os.makedirs(args.cache, exist_ok=True)
        rgp = os.path.join(args.cache, f"{STEM}_X_rg100.npy")
        if not os.path.exists(rgp):
            import requests
            t0 = time.time()
            url = HUB_BASE + f"{STEM}_X_rg100.npy"
            with requests.get(url, stream=True, timeout=1800) as r:
                r.raise_for_status()
                with open(rgp + ".part", "wb") as fh:
                    for chunk in r.iter_content(1 << 22):
                        fh.write(chunk)
            os.replace(rgp + ".part", rgp)
            timings["download_rg100_s"] = round(time.time() - t0, 1)
        readers["rg100"] = LocalReader("rg100", rgp)

    g025_ny, g025_nx = grids["g025"]["ny"], grids["g025"]["nx"]
    g100_ny, g100_nx = grids["g100"]["ny"], grids["g100"]["nx"]
    ratio = (g025_ny - 1) // (g100_ny - 1)

    rr, rc0, rc1 = rapid_indices(g025_ny, g025_nx, grids["g025"]["lat0"],
                                 grids["g025"]["lon0"], grids["g025"]["step"])
    rapid025 = np.zeros(g025_ny * g025_nx, dtype=bool)
    rapid025.reshape(g025_ny, g025_nx)[rr, rc0:rc1 + 1] = True
    # the 1 deg cells are the COARSE IMAGE of those 0.25 deg cells, through the
    # tensor's own lookup rule (round-half-up), not an independently rounded
    # latitude — 26.5 N sits exactly on a half-cell and the two rules disagree
    _yy, _xx = np.divmod(np.flatnonzero(rapid025), g025_nx)
    _y1, _x1 = coarse_of(_yy, _xx, ratio, g100_ny, g100_nx)
    rapid100 = np.zeros(g100_ny * g100_nx, dtype=bool)
    rapid100[np.unique(_y1 * g100_nx + _x1)] = True
    ry = int(_y1[0])
    rx0, rx1 = int(_x1.min()), int(_x1.max())

    # ---------------- FIT pass -------------------------------------------
    t0 = time.time()
    fit_rows = {"g025": sorted(set([b for b in fit_pairs] + [b + 1 for b in fit_pairs])),
                "rg100": sorted(set([r for r in rg_fit] + [r + 1 for r in rg_fit]))}
    if rg_single_row:
        fit_rows["rg100"] = [0]
    fit_rows["g100"] = fit_rows["g025"]

    sub = {}
    for g in ("g025", "g100"):
        n = grids[g]["ny"] * grids[g]["nx"]
        sub[g] = np.flatnonzero(rng.random(n) < SUBSAMPLE)
    n = grids["rg100"]["ny"] * grids["rg100"]["nx"]
    sub["rg100"] = np.flatnonzero(rng.random(n) < SUBSAMPLE)

    buf = {g: [[] for _ in chans[g]] for g in GROUPS}
    for row, frames in stream_rows({g: readers[g] for g in ("g025", "g100")},
                                   fit_rows["g025"]):
        for g in ("g025", "g100"):
            F = frames[g].reshape(-1, frames[g].shape[-1])
            for c in range(F.shape[1]):
                x = F[sub[g], c]
                x = x[np.isfinite(x)]
                if x.size:
                    buf[g][c].append(x.astype(np.float32))
    for row, frames in stream_rows({"rg100": readers["rg100"]}, fit_rows["rg100"]):
        F = frames["rg100"].reshape(-1, frames["rg100"].shape[-1])
        for c in range(F.shape[1]):
            x = F[sub["rg100"], c]
            x = x[np.isfinite(x)]
            if x.size:
                buf["rg100"][c].append(x.astype(np.float32))
    timings["fit_pass_s"] = round(time.time() - t0, 1)

    t0 = time.time()
    models = {g: {} for g in GROUPS}
    fit_counts = {g: {} for g in GROUPS}
    skipped = {}
    min_fit = 200 if smoke else 5000
    for g in GROUPS:
        for c, name in enumerate(chans[g]):
            parts = buf[g][c]
            buf[g][c] = None
            if not parts:
                skipped[f"{g}.{name}"] = "no finite values in the fit bins"
                continue
            x = np.concatenate(parts)
            del parts
            if x.size > FIT_CAP:
                step = x.size / float(FIT_CAP)
                x = x[np.floor(np.arange(FIT_CAP) * step).astype(np.int64)]
            x = np.sort(x.astype(np.float64))
            fit_counts[g][name] = int(x.size)
            if x.size < min_fit or x[0] == x[-1]:
                skipped[f"{g}.{name}"] = f"only {x.size} finite fit values"
                continue
            m = fit_channel(x)
            if m is None:
                skipped[f"{g}.{name}"] = "fine partition degenerate"
                continue
            models[g][name] = m
    del buf
    timings["fit_warps_s"] = round(time.time() - t0, 1)

    # alpha = 1 must reproduce the plain quantiles — asserted on one channel
    alpha1_check = None
    for g in GROUPS:
        for name, m in models[g].items():
            e = warp_edges(m["part"], 1.0, 64)
            ps = np.linspace(CLIP_Q[0], CLIP_Q[1], N_FINE)
            plo, phi = ps[0], ps[-1]
            # the plain-quantile reference, interpolated on the same fine grid
            ref = np.interp(np.arange(1, 64) / 64.0,
                            (np.cumsum(np.concatenate([[0.0], m["part"]["mass"]]))
                             / m["part"]["mass"].sum()), m["part"]["v"])
            alpha1_check = dict(
                group=g, channel=name,
                max_abs_edge_diff=float(np.max(np.abs(e - ref))),
                edge_span=float(m["part"]["clip_hi"] - m["part"]["clip_lo"]),
                relative=float(np.max(np.abs(e - ref))
                               / max(m["part"]["clip_hi"] - m["part"]["clip_lo"], 1e-12)))
            break
        if alpha1_check:
            break

    # ---------------- EVAL pass ------------------------------------------
    t0 = time.time()
    accs = {g: {name: ChannelAcc(m, None) for name, m in models[g].items()}
            for g in GROUPS}
    eval_pair_set = set(eval_pairs)
    eval_rows = sorted(set([b for b in eval_pairs] + [b + 1 for b in eval_pairs]))
    prev = None
    cs_idx = chans["g025"].index("cur_speed") if "cur_speed" in chans["g025"] else None

    for row, frames in stream_rows({g: readers[g] for g in ("g025", "g100")},
                                   eval_rows):
        F025 = frames["g025"].reshape(-1, frames["g025"].shape[-1])
        F100 = frames["g100"].reshape(-1, frames["g100"].shape[-1])
        # the cur_speed top-1 % mask for THIS bin, and its coarse image
        tail025 = np.zeros(F025.shape[0], dtype=bool)
        tail100 = np.zeros(F100.shape[0], dtype=bool)
        if cs_idx is not None:
            cs = F025[:, cs_idx]
            fin = np.isfinite(cs)
            if fin.sum() > 100:
                thr = np.quantile(cs[fin], TAIL_Q)
                tail025 = fin & (cs >= thr)
                yy, xx = np.divmod(np.flatnonzero(tail025), g025_nx)
                y1, x1 = coarse_of(yy, xx, ratio, g100_ny, g100_nx)
                tail100[np.unique(y1 * g100_nx + x1)] = True
        packs = {"g025": (F025, tail025, rapid025),
                 "g100": (F100, tail100, rapid100)}
        for g in ("g025", "g100"):
            F, tmask, rmask = packs[g]
            for c, name in enumerate(chans[g]):
                acc = accs[g].get(name)
                if acc is None:
                    continue
                v = F[:, c]
                fin = np.isfinite(v)
                acc.add_bin(v[fin].astype(np.float64), tmask[fin], rmask[fin])
        if prev is not None and prev[0] == row - 1 and prev[0] in eval_pair_set:
            for g in ("g025", "g100"):
                Fp = prev[1][g]
                Fc = packs[g][0]
                for c, name in enumerate(chans[g]):
                    acc = accs[g].get(name)
                    if acc is None:
                        continue
                    a, b = Fp[:, c], Fc[:, c]
                    both = np.isfinite(a) & np.isfinite(b)
                    acc.add_pair(a[both].astype(np.float64),
                                 b[both].astype(np.float64))
        prev = (row, {"g025": F025, "g100": F100})
    timings["eval_pass_pentad_s"] = round(time.time() - t0, 1)

    # rg100 (its rows are months)
    t0 = time.time()
    rg_rows = sorted(set([r for r in rg_ev] + [r + 1 for r in rg_ev]))
    if rg_single_row:
        rg_rows = [0]
    rg_pair_set = set(rg_ev)
    prevr = None
    zero_tail = np.zeros(g100_ny * g100_nx, dtype=bool)
    for row, frames in stream_rows({"rg100": readers["rg100"]}, rg_rows):
        F = frames["rg100"].reshape(-1, frames["rg100"].shape[-1])
        for c, name in enumerate(chans["rg100"]):
            acc = accs["rg100"].get(name)
            if acc is None:
                continue
            v = F[:, c]
            fin = np.isfinite(v)
            acc.add_bin(v[fin].astype(np.float64), zero_tail[fin], rapid100[fin])
        if prevr is not None and prevr[0] == row - 1 and prevr[0] in rg_pair_set:
            for c, name in enumerate(chans["rg100"]):
                acc = accs["rg100"].get(name)
                if acc is None:
                    continue
                a, b = prevr[1][:, c], F[:, c]
                both = np.isfinite(a) & np.isfinite(b)
                acc.add_pair(a[both].astype(np.float64), b[both].astype(np.float64))
        prevr = (row, F)
    timings["eval_pass_rg100_s"] = round(time.time() - t0, 1)

    # ---------------- read-outs ------------------------------------------
    results = {}
    for g in GROUPS:
        results[g] = {}
        for c, name in enumerate(chans[g]):
            acc = accs[g].get(name)
            if acc is None:
                continue
            sd = float(norms[g][c, 1])
            results[g][name] = channel_readouts(acc, sd, units[g].get(name, "z"))

    out = dict(
        experiment="E-074a",
        title=("the ladder bake-off — per-channel warped quantizers fitted on "
               "family 7's training pentads and scored on held-out ones"),
        generated_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        smoke=bool(smoke),
        comparability_note=(
            "Every scheme's held-out cross-entropy is computed on ONE COMMON "
            "FINE PARTITION E per channel — the 4,096 equal-probability "
            "quantiles of that channel's fit sample, clipped to [q0.0005, "
            "q0.9995], deduplicated (the tensor stores float16, so quantiles "
            "can coincide), plus two open end cells. A ladder's probability for "
            "a fine cell is its bin mass pi_k spread uniformly in x and "
            "intersected with the cell; the Gaussian and the Gaussian mixture "
            "give the difference of their cumulative distribution across the "
            "cell. This is the ONLY way a discrete quantizer and a continuous "
            "density are comparable at all; a cross-entropy computed any other "
            "way is not comparable and its number means nothing."),
        regulariser_note=(
            f"Every scheme (ladders and baselines alike) is mixed with a "
            f"uniform-over-fine-cells escape of weight {ESCAPE:g} before the "
            "cross-entropy is taken, so that a scheme which gives some cell zero "
            "mass scores a large finite number rather than infinity. The mixture "
            "is identical for every scheme, so comparisons are unaffected; the "
            "'uniform_cells' row is that escape distribution on its own."),
        units_note=(
            "The tensor stores z-scored values (docs/FAMILY7_DATA_HANDOVER.md "
            "section 4). Warps, partitions and cross-entropies are computed in "
            "stored units — cross-entropy over a common partition is a discrete "
            "quantity and is unit-free — and every RMSE and quantization floor "
            "is multiplied by that channel's sd from norm_<group> so it is "
            "reported in the channel's own physical unit."),
        sampling=dict(
            note=sampling_note,
            seed=SEED,
            spatial_subsample_of_fit_bins=SUBSAMPLE,
            fit_pairs_pentad=[[int(b), int(b) + 1] for b in fit_pairs],
            fit_pair_dates=[str(bin_start(b)) for b in fit_pairs] if not smoke else None,
            eval_pairs_pentad=[[int(b), int(b) + 1] for b in eval_pairs],
            eval_pair_dates=[str(bin_start(b)) for b in eval_pairs] if not smoke else None,
            fit_pairs_rg100_rows=[[int(r), int(r) + 1] for r in rg_fit],
            fit_pairs_rg100_months=[[rg_months[r], rg_months[r + 1]]
                                    for r in rg_fit] if not rg_single_row else [],
            eval_pairs_rg100_rows=[[int(r), int(r) + 1] for r in rg_ev],
            eval_pairs_rg100_months=[[rg_months[r], rg_months[r + 1]]
                                     for r in rg_ev] if not rg_single_row else [],
            n_fit_pairs=len(fit_pairs), n_eval_pairs=len(eval_pairs),
            n_fit_pairs_rg100=len(rg_fit), n_eval_pairs_rg100=len(rg_ev),
            fit_month_histogram=(_month_hist(fit_pairs) if not smoke else None),
            eval_month_histogram=(_month_hist(eval_pairs) if not smoke else None),
            fit_year_counts=(_year_counts(fit_pairs) if not smoke else None),
            eval_year_counts=(_year_counts(eval_pairs) if not smoke else None),
            seasonal_note=("The fit pairs are spread evenly over the whole "
                           "ordered list of training candidate pairs rather "
                           "than pinned inside each year, so the sampling "
                           "phase precesses through the calendar; the two "
                           "histograms above are the check that it did."),
        ),
        masks=dict(
            rapid_row_g025=int(rr), rapid_cols_g025=[int(rc0), int(rc1)],
            rapid_row_g100=int(ry), rapid_cols_g100=[int(rx0), int(rx1)],
            tail=("cells where cur_speed is in its top 1 % on that bin; applied "
                  "to every g025 channel, mapped to the containing 1 deg cell "
                  "for g100, not applied to rg100"),
            rapid_note=("row 466 = 26.5 N, columns 400-668 = 80 W to 13 W on the "
                        "0.25 deg grid, and the containing 1 deg cells for g100 "
                        "and rg100"),
        ),
        grid=dict(scheme_count=len(SCHEMES), alphas=list(ALPHAS),
                  K1s=list(K1S), K2=K2, depth1_matched_K=K1_MATCHED,
                  n_fine_grid=N_FINE, clip_quantiles=list(CLIP_Q)),
        alpha1_equals_plain_quantiles=alpha1_check,
        fit_value_counts=fit_counts,
        skipped_channels=skipped,
        channels=results,
        timings_s=timings,
    )
    out["falsifiers"] = falsifiers(out)
    out["timings_s"]["total_s"] = round(time.time() - t_all, 1)

    os.makedirs(args.out, exist_ok=True)
    jp = os.path.join(args.out, "E074a_results.json")
    with open(jp + ".tmp", "w") as fh:
        json.dump(out, fh, indent=1, sort_keys=False, default=_jsonable)
    os.replace(jp + ".tmp", jp)
    mp = os.path.join(args.out, "E074a_results.md")
    with open(mp + ".tmp", "w") as fh:
        fh.write(render_md(out))
    os.replace(mp + ".tmp", mp)

    if not smoke and not args.keep_cache:
        rgp = os.path.join(args.cache, f"{STEM}_X_rg100.npy")
        if os.path.exists(rgp):
            os.remove(rgp)
    print(f"wrote {jp}\nwrote {mp}")
    return out


def _jsonable(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(type(o))


# --------------------------------------------------------------------------
# falsifiers (plan section 6, E-074a)
# --------------------------------------------------------------------------

def _iter_channels(res):
    for g in GROUPS:
        for name, r in res["channels"].get(g, {}).items():
            yield g, name, r


def falsifiers(res):
    out = {}

    # F1 — does alpha = 1 beat alpha = 0 on held-out nats by more than the
    #      channel-to-channel spread of that difference?
    f1 = {}
    for K1 in K1S:
        d, chans_used = [], []
        for g, name, r in _iter_channels(res):
            a0 = r["schemes"].get(f"a0_K{K1}_d1", {}).get("ce_nats")
            a1 = r["schemes"].get(f"a1_K{K1}_d1", {}).get("ce_nats")
            if a0 is None or a1 is None:
                continue
            d.append(a0 - a1)
            chans_used.append(f"{g}.{name}")
        if d:
            d = np.asarray(d)
            f1[f"K1={K1}"] = dict(
                n_channels=len(d), mean_gain_nats=float(d.mean()),
                sd_gain_nats=float(d.std(ddof=1)) if len(d) > 1 else None,
                median_gain_nats=float(np.median(d)),
                n_channels_where_alpha1_wins=int((d > 0).sum()),
                verdict_alpha1_beats_alpha0_by_more_than_the_spread=bool(
                    len(d) > 1 and d.mean() > d.std(ddof=1)),
                worst_channel=chans_used[int(np.argmin(d))],
                worst_gain_nats=float(d.min()),
                best_channel=chans_used[int(np.argmax(d))],
                best_gain_nats=float(d.max()))
    out["F1_alpha1_vs_alpha0_heldout_nats"] = dict(
        question=("Does alpha = 1 (equal-probability bins) beat alpha = 0 "
                  "(uniform bins) on held-out nats by more than the "
                  "channel-to-channel spread of that difference? If not, the "
                  "quantile idea does not survive its own data."),
        by_K1=f1,
        headline=f1.get("K1=64"))

    # F2 — is the conditional entropy flat in alpha at K1 = 64?
    ranges, chans_used, detail = [], [], {}
    for g, name, r in _iter_channels(res):
        vals, rats, marg = [], [], []
        for an in ALPHA_NAMES:
            e = r["persistence"].get(f"a{an}_K64")
            if e and e.get("h_cond") is not None and e.get("ratio") is not None:
                vals.append(e["h_cond"])
                rats.append(e["ratio"])
                marg.append(e["h_marg"])
        if len(vals) == len(ALPHA_NAMES):
            rng_ = max(vals) - min(vals)
            ranges.append(rng_)
            chans_used.append(f"{g}.{name}")
            detail[f"{g}.{name}"] = dict(
                h_cond_by_alpha=dict(zip(ALPHA_NAMES, vals)),
                h_marg_by_alpha=dict(zip(ALPHA_NAMES, marg)),
                ratio_by_alpha=dict(zip(ALPHA_NAMES, rats)),
                range_nats=float(rng_),
                range_ratio=float(max(rats) - min(rats)),
                argmin_alpha=ALPHA_NAMES[int(np.argmin(vals))],
                argmin_alpha_ratio=ALPHA_NAMES[int(np.argmin(rats))])
    flat_thresh = 0.02 * math.log(64)
    argmins = [d["argmin_alpha"] for d in detail.values()]
    argmins_r = [d["argmin_alpha_ratio"] for d in detail.values()]
    ratio_ranges = [d["range_ratio"] for d in detail.values()]
    out["F2_conditional_entropy_flat_in_alpha"] = dict(
        question=("Is the coarse digit's conditional entropy H(d1(t+1)|d1(t)) "
                  "flat in alpha at K1 = 64? If it is, section 4 of the plan is "
                  "void and alpha = 1 is taken on the class-balance argument."),
        criterion=(f"declared flat if the median across channels of the range "
                   f"across alpha is below 2 % of log(64) = "
                   f"{flat_thresh:.4f} nats (a threshold chosen here, stated so "
                   f"it can be disagreed with)"),
        n_channels=len(ranges),
        median_range_nats=float(np.median(ranges)) if ranges else None,
        min_range_nats=float(np.min(ranges)) if ranges else None,
        max_range_nats=float(np.max(ranges)) if ranges else None,
        threshold_nats=flat_thresh,
        verdict_flat=bool(ranges and float(np.median(ranges)) < flat_thresh),
        alpha_minimising_h_cond_counts={a: int(argmins.count(a))
                                        for a in ALPHA_NAMES},
        caveat=("A lower conditional entropy at a small alpha is partly free: "
                "a ladder that ignores the density also carries less "
                "information, so its digit has a lower MARGINAL entropy too. "
                "The fraction H(next|now)/H(now) — how much of what the digit "
                "says survives a pentad — is the version of the question that "
                "cannot be won by saying less."),
        alpha_minimising_ratio_counts={a: int(argmins_r.count(a))
                                       for a in ALPHA_NAMES},
        median_range_of_ratio=float(np.median(ratio_ranges)) if ratio_ranges else None,
        per_channel=detail)

    # F3 — alpha = 1's tail RMSE on cur_speed against alpha = 1/3's
    f3 = {}
    cs = res["channels"].get("g025", {}).get("cur_speed")
    if cs:
        for K1 in K1S:
            a1 = cs["schemes"].get(f"a1_K{K1}_d1", {}).get("rmse_tail")
            a3 = cs["schemes"].get(f"a1/3_K{K1}_d1", {}).get("rmse_tail")
            if a1 is None or a3 is None:
                continue
            f3[f"K1={K1}"] = dict(
                rmse_tail_alpha1=a1, rmse_tail_alpha_third=a3,
                excess=a1 - a3,
                verdict_alpha1_worse_by_more_than_the_alpha_third_value=bool(
                    a1 - a3 > a3))
    out["F3_tail_rmse_cur_speed"] = dict(
        question=("Is alpha = 1's reconstruction RMSE on the top 1 % of "
                  "cur_speed worse than alpha = 1/3's by more than the "
                  "alpha = 1/3 value itself? If so, tails decide and "
                  "alpha = 1/3 is taken regardless of the nats column."),
        unit=cs["unit"] if cs else None, by_K1=f3, headline=f3.get("K1=64"))

    # F4 — depth-2 vs depth-1 at MATCHED K1*K2 = 4096 pieces
    rows, worst = [], None
    for g, name, r in _iter_channels(res):
        for an in ("0", "1"):
            d2 = r["schemes"].get(f"a{an}_K64_d2", {}).get("ce_nats")
            d1 = r["schemes"].get(f"a{an}_K{K1_MATCHED}_d1", {}).get("ce_nats")
            if d2 is None or d1 is None or d1 == 0:
                continue
            rel = abs(d2 - d1) / abs(d1)
            rows.append(dict(channel=f"{g}.{name}", alpha=an,
                             depth2_K64x64_nats=d2,
                             depth1_K4096_nats=d1,
                             rel_diff=float(rel), within_1pct=bool(rel <= 0.01)))
    if rows:
        worst = max(rows, key=lambda r: r["rel_diff"])
    out["F4_depth2_vs_depth1_matched"] = dict(
        question=("At matched piece count (depth 2 with K1 = 64 and K2 = 64, "
                  "i.e. 4,096 pieces, against depth 1 with K1 = 4,096), do the "
                  "held-out nats agree to within 1 %? If not, the whitening "
                  "argument of section 1.2 is wrong somewhere, or the "
                  "implementation is at fault."),
        n_comparisons=len(rows),
        n_within_1pct=int(sum(r["within_1pct"] for r in rows)),
        median_rel_diff=float(np.median([r["rel_diff"] for r in rows])) if rows else None,
        worst=worst,
        verdict_all_within_1pct=bool(rows and all(r["within_1pct"] for r in rows)),
        rows=rows)
    return out


# --------------------------------------------------------------------------
# the readable summary
# --------------------------------------------------------------------------

def _fmt(x, nd=4):
    if x is None:
        return "—"
    if isinstance(x, bool):
        return "yes" if x else "no"
    try:
        if abs(x) >= 1e5 or (x != 0 and abs(x) < 1e-4):
            return f"{x:.3g}"
        return f"{x:.{nd}f}"
    except Exception:
        return str(x)


def _table(header, rows):
    out = ["| " + " | ".join(header) + " |",
           "|" + "|".join(["---"] * len(header)) + "|"]
    for r in rows:
        out.append("| " + " | ".join(r) + " |")
    return "\n".join(out) + "\n"


def render_md(res):
    L = []
    A = L.append
    A("# E-074a · the ladder bake-off — results\n")
    A(f"*Generated {res['generated_utc']}"
      + (" — **SMOKE RUN on the decimated fixture, not the experiment**" if res["smoke"] else "")
      + ".*\n")
    A("**What this experiment is.** E-074a asks whether it is worth giving every "
      "input channel of the global tensor its own set of value bins fitted to "
      "that channel's own distribution, and if so, how the bins should be "
      "spaced. No model is trained anywhere: warps are fitted on training "
      "pentads of family 7 (the whole-globe 0.25°, five-day input tensor) and "
      "scored as descriptions of held-out pentads. The one free parameter under "
      "test is **alpha**, the exponent in \"put bin boundaries where the density "
      "of the data raised to the power alpha is high\": alpha = 1 gives bins of "
      "equal probability (plain quantiles), alpha = 1/3 is the spacing that "
      "minimises squared reconstruction error, alpha = 1/2 sits between them, "
      "and alpha = 0 is plain uniform bins that ignore the data entirely.\n")

    A("## How to read this\n")
    A("Every number below is measured on **held-out years** — 2009, 2017 and "
      "2023 — from warps fitted only on training years (a bin's opening year at "
      "most 2020, and never one of those three). "
      "**Nats** are a unit of surprise: the cross-entropy is the average number "
      "of nats of surprise the scheme suffers per held-out value, so lower is "
      "better and a difference of 0.01 nats is a 1 % difference in the "
      "probability the scheme assigns. Every scheme — the bin ladders, the "
      "single Gaussian and the three-component Gaussian mixture — is scored on "
      "**one common fine partition** of about 4,000 cells built from the fit "
      "sample's quantiles, because a set of discrete bins and a continuous "
      "probability density are otherwise not comparable at all and the number "
      "would look decisive while meaning nothing. "
      "**Conditional entropy** H(d1(t+1)|d1(t)) is how much surprise remains "
      "about the coarse digit one pentad later once you know it now: low means "
      "the digit is predictable, which is what a forecaster wants, and it is a "
      "different question from how much the digit says (its marginal entropy). "
      "**RMSE** is the reconstruction error if each held-out value were replaced "
      "by the centre of the bin it lands in, in the channel's own physical unit. "
      "The **quantization floor** is the error a perfect model would still make "
      "for that bin layout; a model whose error reaches the floor is finished, "
      "not failing.\n")
    A("`d1` is the depth-1 ladder (K1 bins). `d2` is the depth-2 ladder: each "
      "of the K1 coarse bins is cut into K2 = 64 equal-width pieces. `d2_tailq` "
      "is the same with the **tail exception** — the two outermost bins, which "
      "are open and carry the extreme values, are cut at the fit sample's own "
      "quantiles inside that bin instead of into equal widths.\n")
    A("> " + res["comparability_note"] + "\n")
    A("> " + res["units_note"] + "\n")
    A("> " + res["regulariser_note"] + "\n")

    s = res["sampling"]
    A("## The sample\n")
    A(f"- {s['note']}\n"
      f"- Fit: **{s['n_fit_pairs']} consecutive-pentad pairs**, stratified over "
      f"training years; a random {int(SUBSAMPLE*100)} % spatial subsample of "
      f"their cells (seed {s['seed']}) is what the warps are fitted on.\n"
      f"- Eval: **{s['n_eval_pairs']} consecutive-pentad pairs** inside the "
      f"three held-out years; every finite cell is used.\n"
      f"- Ocean interior (rg100, whose rows are months, not pentads): "
      f"**{s['n_fit_pairs_rg100']} fit month pairs** and "
      f"**{s['n_eval_pairs_rg100']} held-out month pairs**.\n"
      f"- The full bin and month lists are in `E074a_results.json` under "
      f"`sampling`.\n")
    a1 = res.get("alpha1_equals_plain_quantiles")
    if a1:
        A(f"- Sanity: on `{a1['group']}.{a1['channel']}`, the alpha = 1 ladder's "
          f"64 bin edges reproduce the plain quantiles to "
          f"{_fmt(a1['relative'], 2)} of the clip range "
          f"(max absolute difference {a1['max_abs_edge_diff']:.3g} in stored "
          f"units).\n")

    chans = [(g, n, r) for g, n, r in _iter_channels(res)]

    # ---- table 1: nats at K1 = 64
    A("## 1 · Held-out cross-entropy, nats per value, at K1 = 64\n")
    A("Lower is better. Columns are the four alphas at depth 1, then the two "
      "baselines fitted on the same fit sample, then the "
      "'uniform over the common partition' reference. That last column is **not "
      "a weak baseline**: the partition's cells are near-equally likely under "
      "the fit sample by construction, so a flat distribution over them is "
      "close to the fit sample's own empirical description at the partition's "
      "resolution, and its value is simply log(number of cells). Read it as "
      "'what a raw few-thousand-bin histogram of the training years costs on "
      "held-out years' — a scheme below it is genuinely describing held-out "
      "values better than that histogram does, with far fewer bins.\n")
    hdr = ["channel", "unit"] + [f"α={a} d1" for a in ALPHA_NAMES] + \
          ["Gaussian", "GMM-3", "uniform", "clipped %"]
    rows = []
    for g, n, r in chans:
        sc = r["schemes"]
        rows.append([f"`{g}.{n}`", r["unit"]]
                    + [_fmt(sc.get(f"a{a}_K64_d1", {}).get("ce_nats"), 3)
                       for a in ALPHA_NAMES]
                    + [_fmt(sc.get("gaussian", {}).get("ce_nats"), 3),
                       _fmt(sc.get("gmm3", {}).get("ce_nats"), 3),
                       _fmt(sc.get("uniform_cells", {}).get("ce_nats"), 3),
                       _fmt(100.0 * (r["clipped_fraction"] or 0.0), 3)])
    A(_table(hdr, rows))

    A("### the same, at K1 = 16 and K1 = 256\n")
    hdr = ["channel"] + [f"α={a} K16" for a in ALPHA_NAMES] + \
          [f"α={a} K256" for a in ALPHA_NAMES]
    rows = []
    for g, n, r in chans:
        sc = r["schemes"]
        rows.append([f"`{g}.{n}`"]
                    + [_fmt(sc.get(f"a{a}_K16_d1", {}).get("ce_nats"), 3) for a in ALPHA_NAMES]
                    + [_fmt(sc.get(f"a{a}_K256_d1", {}).get("ce_nats"), 3) for a in ALPHA_NAMES])
    A(_table(hdr, rows))

    # ---- table 2: conditional entropy
    A("## 2 · Coarse-digit persistence and conditional entropy, K1 = 64\n")
    A("`persistence` is the fraction of cells whose coarse digit is unchanged "
      "one pentad (for the ocean interior, one month) later. `H(next|now)` is "
      "the conditional entropy in nats, `H` the marginal entropy of the digit "
      "on the held-out set, and the ratio is the share of the digit's "
      "information a pentad destroys — the column section 4 of the plan is "
      "about, shown at both ends of the family so they compare directly. A "
      "smaller α buys a lower `H(next|now)` partly by carrying less "
      "information in the first place, which is exactly why the ratio columns "
      "are here. Maximum possible entropy at K1 = 64 is "
      "log 64 = 4.159 nats.\n")
    hdr = ["channel"] + [f"persist α={a}" for a in ALPHA_NAMES] + \
          [f"H(next\\|now) α={a}" for a in ALPHA_NAMES] + \
          ["H α=1", "ratio α=0", "ratio α=1"]
    rows = []
    for g, n, r in chans:
        p = r["persistence"]
        e1 = p.get("a1_K64", {})
        e0 = p.get("a0_K64", {})
        rows.append([f"`{g}.{n}`"]
                    + [_fmt(p.get(f"a{a}_K64", {}).get("persistence"), 3) for a in ALPHA_NAMES]
                    + [_fmt(p.get(f"a{a}_K64", {}).get("h_cond"), 3) for a in ALPHA_NAMES]
                    + [_fmt(e1.get("h_marg"), 3), _fmt(e0.get("ratio"), 3),
                       _fmt(e1.get("ratio"), 3)])
    A(_table(hdr, rows))

    # ---- table 3: tail RMSE for the three named channels
    A("## 3 · Reconstruction RMSE in the channel's own unit\n")
    A("`overall` is every held-out value; `tail` is restricted to the cells "
      "where the ocean's surface current speed is in its top 1 % on that bin "
      "(the strong-current cores that carry Atlantic transport); `RAPID` is the "
      "26.5° N section, 80° W to 13° W, the line the overturning transport is "
      "measured on. All three K1 rungs at depth 1.\n")
    A("One convention decides most of what these tables say, so it is worth "
      "stating twice: **the two outermost bins are open**, and a value in one "
      "of them is reconstructed at the midpoint of the span from the clip edge "
      "out to the fit sample's most extreme value. A ladder that leaves few "
      "boundaries in the low-density tail therefore has very wide end bins, and "
      "every value that lands in one pays for that width. That is not an "
      "artefact of the measurement; it is the registered risk of the plan's "
      "section 7.1 (\"tail starvation\") being paid, and it is why the ordering "
      "here is not the ordering the mean-squared-error-optimal exponent 1/3 "
      "would predict on an unclipped, fully-covered support.\n")
    for chname in ("cur_speed", "ssh", "sst"):
        r = res["channels"].get("g025", {}).get(chname)
        if not r:
            continue
        A(f"**`g025.{chname}`** — unit {r['unit']}, "
          f"{r['heldout_n']:,} held-out values "
          f"({r['heldout_n_tail']:,} in the top-1 % current mask, "
          f"{r['heldout_n_rapid']:,} on the RAPID section)\n")
        hdr = ["scheme", "RMSE overall", "floor overall", "RMSE tail",
               "floor tail", "RMSE RAPID", "floor RAPID"]
        rows = []
        for a in ALPHA_NAMES:
            for K1 in K1S:
                sc = r["schemes"].get(f"a{a}_K{K1}_d1")
                if not sc:
                    continue
                rows.append([f"α={a}, K1={K1}, d1",
                             _fmt(sc["rmse_all"]), _fmt(sc["floor_all"]),
                             _fmt(sc["rmse_tail"]), _fmt(sc["floor_tail"]),
                             _fmt(sc["rmse_rapid"]), _fmt(sc["floor_rapid"])])
        A(_table(hdr, rows))

    A("### depth 2 (K2 = 64), overall RMSE and the tail exception\n")
    A("`d2` cuts every coarse bin into 64 equal-width pieces; `d2+tailq` does "
      "the same except in the two open end bins, where it cuts at the fit "
      "sample's own quantiles inside that bin. The tail exception is an arm, "
      "not a default — the plan's section 7.3 asked for it to be measured, and "
      "the measurement is in this table and in the `_tailq` entries of the "
      "JSON.\n")
    hdr = ["channel", "unit"] + [f"α={a} d2" for a in ALPHA_NAMES] + \
          [f"α={a} d2+tailq" for a in ALPHA_NAMES]
    rows = []
    for g, n, r in chans:
        sc = r["schemes"]
        rows.append([f"`{g}.{n}`", r["unit"]]
                    + [_fmt(sc.get(f"a{a}_K64_d2", {}).get("rmse_all")) for a in ALPHA_NAMES]
                    + [_fmt(sc.get(f"a{a}_K64_d2_tailq", {}).get("rmse_all")) for a in ALPHA_NAMES])
    A(_table(hdr, rows))

    A("### what the tail exception does to the strong-current tail, K1 = 64\n")
    A("The one place it is supposed to help: RMSE restricted to the top 1 % of "
      "`cur_speed`, with and without it, beside the held-out nats of the same "
      "two schemes.\n")
    hdr = ["channel", "unit", "α", "tail RMSE d2", "tail RMSE d2+tailq",
           "nats d2", "nats d2+tailq"]
    rows = []
    for chname in ("cur_speed", "ssh", "sst"):
        r = res["channels"].get("g025", {}).get(chname)
        if not r:
            continue
        for a in ALPHA_NAMES:
            p = r["schemes"].get(f"a{a}_K64_d2", {})
            q = r["schemes"].get(f"a{a}_K64_d2_tailq", {})
            rows.append([f"`g025.{chname}`", r["unit"], a,
                         _fmt(p.get("rmse_tail")), _fmt(q.get("rmse_tail")),
                         _fmt(p.get("ce_nats"), 3), _fmt(q.get("ce_nats"), 3)])
    A(_table(hdr, rows))

    # ---- table 4: floors
    A("## 4 · The quantization floor, K1 = 64 depth 1 vs depth 2\n")
    A("sqrt(mean of Δ²/12) over held-out values, Δ the width of the bin the "
      "value falls in — the reconstruction error that remains when the model is "
      "perfect. Open end bins use the clip edge out to the fit sample's extreme "
      "as their width.\n")
    hdr = ["channel", "unit"] + [f"α={a} d1" for a in ALPHA_NAMES] + \
          [f"α={a} d2" for a in ALPHA_NAMES]
    rows = []
    for g, n, r in chans:
        sc = r["schemes"]
        rows.append([f"`{g}.{n}`", r["unit"]]
                    + [_fmt(sc.get(f"a{a}_K64_d1", {}).get("floor_all")) for a in ALPHA_NAMES]
                    + [_fmt(sc.get(f"a{a}_K64_d2", {}).get("floor_all")) for a in ALPHA_NAMES])
    A(_table(hdr, rows))

    # ---- table 5: calibration
    A("## 5 · Calibration of the depth-1 ladders, K1 = 64\n")
    A("How far the held-out frequency of each bin is from the bin mass π_k the "
      "fit sample gave it. `max` is the worst bin, `mean` the average absolute "
      "deviation. A perfectly calibrated α = 1 ladder would have every π_k = "
      "1/64 = 0.0156.\n")
    hdr = ["channel"] + [f"max α={a}" for a in ALPHA_NAMES] + \
          [f"mean α={a}" for a in ALPHA_NAMES]
    rows = []
    for g, n, r in chans:
        sc = r["schemes"]
        rows.append([f"`{g}.{n}`"]
                    + [_fmt(sc.get(f"a{a}_K64_d1", {}).get("calib_max_abs_dev"), 4)
                       for a in ALPHA_NAMES]
                    + [_fmt(sc.get(f"a{a}_K64_d1", {}).get("calib_mean_abs_dev"), 5)
                       for a in ALPHA_NAMES])
    A(_table(hdr, rows))

    # ---- verdicts
    A("## Verdict block\n")
    A("```\n" + verdict_text(res) + "```\n")

    A("## Where the numbers live\n")
    A("- Full machine-readable results, including every scheme, every mask, the "
      "sampled bin lists, the clipped fractions and the timings: "
      "`ml/plans/E074a_results.json`.\n"
      "- The script: `ml/ladder_bakeoff.py` (`--smoke` runs the whole pipeline "
      "on the decimated fixture in under a minute).\n"
      "- The pins: `tests/test_ladder_bakeoff.py`.\n")
    return "\n".join(L)


def verdict_text(res):
    f = res["falsifiers"]
    L = []
    A = L.append
    A("E-074a — pre-registered falsifiers, evaluated in code\n")

    f1 = f["F1_alpha1_vs_alpha0_heldout_nats"]
    A("F1  alpha=1 vs alpha=0 on held-out nats (depth 1)")
    for k, v in f1["by_K1"].items():
        A(f"      {k:8s} mean gain {v['mean_gain_nats']:+.4f} nats"
          f"  sd across channels {_fmt(v['sd_gain_nats'], 4)}"
          f"  wins {v['n_channels_where_alpha1_wins']}/{v['n_channels']}"
          f"  -> beats the spread: "
          f"{'YES' if v['verdict_alpha1_beats_alpha0_by_more_than_the_spread'] else 'NO'}")
    h = f1.get("headline")
    if h:
        A(f"      worst channel {h['worst_channel']} {h['worst_gain_nats']:+.4f}"
          f"   best {h['best_channel']} {h['best_gain_nats']:+.4f}")
    A("")

    f2 = f["F2_conditional_entropy_flat_in_alpha"]
    A("F2  is H(d1(t+1)|d1(t)) flat in alpha at K1=64?")
    A(f"      median range across alpha {_fmt(f2['median_range_nats'], 4)} nats"
      f"  (min {_fmt(f2['min_range_nats'], 4)}, max {_fmt(f2['max_range_nats'], 4)},"
      f" n={f2['n_channels']})")
    A(f"      threshold {f2['threshold_nats']:.4f} nats (2 % of log 64)"
      f"   -> FLAT: {'YES' if f2['verdict_flat'] else 'NO'}")
    A(f"      alpha minimising the conditional entropy, channel counts: "
      f"{f2['alpha_minimising_h_cond_counts']}")
    A(f"      alpha minimising H(next|now)/H(now) (the share that survives a "
      f"pentad), channel counts: {f2.get('alpha_minimising_ratio_counts')}"
      f"   median range of that ratio "
      f"{_fmt(f2.get('median_range_of_ratio'), 4)}")
    A("")

    f3 = f["F3_tail_rmse_cur_speed"]
    A("F3  alpha=1 tail RMSE on cur_speed vs alpha=1/3 "
      f"(unit {f3.get('unit')})")
    for k, v in f3["by_K1"].items():
        A(f"      {k:8s} alpha=1 {v['rmse_tail_alpha1']:.4f}"
          f"  alpha=1/3 {v['rmse_tail_alpha_third']:.4f}"
          f"  excess {v['excess']:+.4f}"
          f"  -> worse by more than the alpha=1/3 value: "
          f"{'YES' if v['verdict_alpha1_worse_by_more_than_the_alpha_third_value'] else 'NO'}")
    A("")

    f4 = f["F4_depth2_vs_depth1_matched"]
    A("F4  depth2 (K1=64 x K2=64) vs depth1 (K1=4096), matched at 4096 pieces")
    A(f"      {f4['n_within_1pct']}/{f4['n_comparisons']} comparisons within 1 %"
      f"   median relative difference {_fmt(f4['median_rel_diff'], 5)}")
    if f4.get("worst"):
        w = f4["worst"]
        A(f"      worst: {w['channel']} alpha={w['alpha']}  "
          f"d2 {w['depth2_K64x64_nats']:.4f} vs d1 {w['depth1_K4096_nats']:.4f}"
          f"  ({100*w['rel_diff']:.2f} %)")
    A(f"      -> ALL within 1 %: {'YES' if f4['verdict_all_within_1pct'] else 'NO'}")
    return "\n".join(L) + "\n"


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--smoke", action="store_true",
                    help="run the whole pipeline on data/family7/fixture/ in < 1 min")
    ap.add_argument("--out", default="ml/plans",
                    help="directory for E074a_results.json / .md")
    ap.add_argument("--cache", default=os.environ.get(
        "E074A_CACHE", os.path.join(os.path.expanduser("~"), ".cache", "e074a")),
        help="where the small npz and the rg100 array are cached")
    ap.add_argument("--fixture", default=FIXTURE_DIR)
    ap.add_argument("--keep-cache", action="store_true",
                    help="keep the 1.05 GB rg100 array after the run")
    ap.add_argument("--rerender", metavar="RESULTS_JSON", default=None,
                    help="recompute the falsifiers and the markdown summary "
                         "from an existing results JSON, without re-reading any "
                         "data (the accumulated read-outs are not recomputed)")
    args = ap.parse_args(argv)
    if args.rerender:
        with open(args.rerender) as fh:
            res = json.load(fh)
        res["falsifiers"] = falsifiers(res)
        res["rerendered_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        with open(args.rerender + ".tmp", "w") as fh:
            json.dump(res, fh, indent=1, default=_jsonable)
        os.replace(args.rerender + ".tmp", args.rerender)
        mp = os.path.join(os.path.dirname(args.rerender) or ".",
                          "E074a_results.md")
        with open(mp + ".tmp", "w") as fh:
            fh.write(render_md(res))
        os.replace(mp + ".tmp", mp)
        print(f"rerendered {args.rerender}\nrerendered {mp}")
        return
    run(args)


if __name__ == "__main__":
    main()

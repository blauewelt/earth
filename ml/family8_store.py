#!/usr/bin/env python3
"""Family 8's OBSERVATION STORE reader — raw Argo profiles, and the k nearest.

E-076 §2 and §3, made readable. `ml/build_family8_argo.py` writes the store;
this file is the only thing that opens it, so the layout is defined once.

PLAIN ENGLISH. Family 7 asks each cone dot "is there a value at this cell and
this five-day bin", and for the Argo depth column the answer is "no" at 92 %
of bins. Family 8 asks a different question — "what are the k nearest
measurements, and how far away are they" — so the channel is never empty, the
measurement is just far away. This module holds the raw profiles (no gridding,
no mapping) and answers that question.

WHAT IS IN THE STORE (`<work>/family8_argo_l0/`, one `.npy` per column, N rows
sorted by `(bin, time)` ascending):

    bin.npy         int16   [N]      five-day bin, floor((date - 1982-01-01)/5)
    time_days.npy   float32 [N]      days since 1982-01-01, FRACTIONAL
    lat.npy         float32 [N]      degrees north
    lon.npy         float32 [N]      degrees east, in [-180, 180)
    temp.npy        float16 [N, 16]  degC at the 16 RG pressure levels, NaN = not filled
    psal.npy        float16 [N, 16]  PSU, same
    wmo.npy         int32   [N]      the float's WMO number
    cycle.npy       int16   [N]      cycle number
    nlev.npy        int16   [N]      good samples the profile contributed
    maxpres.npy     float16 [N]      deepest good pressure, dbar
    mode.npy        |S1     [N]      b'R' real time, b'A' adjusted, b'D' delayed
    bin_offsets.npy int64   [3143]   CSR: rows of bin b are [off[b], off[b+1])
    store.json                       schema, QC policy, provenance, sha256s

The values are RAW (degC, PSU) — not z-scored and not anomalised.
Normalisation and the harmonic climatology of E-076 §2.5 are the sampler's
job, deliberately: the store is an archive of measurements, and a measurement
that has already been divided by a training-set sigma is no longer one.

THE SEARCH (`knearest`), and its three rules from E-076 §2.2:

  ONE-SIDED IN TIME is a HARD CONSTRAINT OF THE SEARCH, not a mask applied
  afterwards. Only observations at or before the anchor's pentad are
  candidates; the CSR slice stops at `bin_offsets[bin + 1]` and never reads a
  later row, so a future observation cannot be returned however close it is.
  A forecaster that has seen the next pentad is not a forecaster.

  BOUNDED by `R_max_km` and `T_max_days`. Inside them the k nearest are
  taken; where fewer than k exist the remaining slots carry the MISS TOKEN —
  NaN values, NaN offsets, `valid = False` — the same token family 7 uses at
  92 % of bins, made rare.

  THE METRIC is E-076 §2.1's, d^2 = (dx^2 + dy^2)/L^2 + dt^2/T^2, with L the
  channel's correlation length and T its decorrelation time. They default to
  `R_max_km` and `T_max_days`, which makes the metric's two axes commensurate
  at the search boundary; pass them explicitly to use a channel's own numbers.

WHERE `dt` IS MEASURED FROM. The anchor is a five-day BIN, not an instant, and
`dt_days` must be `>= 0` for every returned observation (E-076 §2.1). So the
reference instant is the END of the anchor's pentad, `5 * (bin + 1)` days
after the epoch: an observation inside the anchor's own bin is 0 to 5 days
old, and `T_max_days = 30` sees exactly six pentads, which is the window
E-076 §2.3's arithmetic is done in. Taking the bin's START instead would make
same-bin observations arrive from the future with a negative age.

OFFSETS ARE KILOMETRES EAST AND NORTH, signed, so the model does not have to
learn the cosine of latitude (E-076 §2.1). `dy = (lat_obs - lat) * KM_PER_DEG`
and `dx = dlon * KM_PER_DEG * cos(mean latitude)`, the same 111.32 km/degree
— 27.83 km per 0.25-degree cell — that `ml/cone.py` and `ml/temporal.py` use,
with their cos floor of 0.05 so a polar anchor degrades instead of dividing by
zero. `tests/test_build_family8_argo.py` pins the constant against
`ml/temporal.py` so the two cannot drift.

THE FOOTPRINT FIELDS (E-076 §2.6) travel with every token. `log2_fp` is the
measurement's SPATIAL support in 0.25-degree-cell units and `log2_dt` its
TEMPORAL support in pentad units; for an Argo profile both are constants —
a point measurement taken in minutes, so log2(footprint/27.83 km) and
log2(0.02 d / 5 d) both clamp to the -4 floor of the plan's [-4, +4] range.
They are constants HERE and fields in the token because the same schema must
carry a 1-degree reanalysis cell (+2.9, 0) and a monthly mapped field
(+2, +2.6) without changing.

    from family8_store import ArgoStore
    st = ArgoStore("ml/cache/family8/family8_argo_l0")
    tok = st.knearest(26.5, -70.0, bin=2400, k=5, R_max_km=1000.0,
                      T_max_days=30.0)
    tok["temp"]      # (5, 16) degC, NaN where a level or a slot is missing
    tok["dx_km"], tok["dy_km"], tok["dt_days"], tok["n_R"]

Pure numpy; no scipy. The search is a vectorised brute force over the rows of
at most seven bins (~15 k rows at Argo's density), which is faster than a tree
at this size and has no build step to keep in sync with the store.
"""
import json
import os

import numpy as np

# ---------------------------------------------------------------- the axis --
EPOCH_YEAR, EPOCH_MONTH, EPOCH_DAY = 1982, 1, 1     # 1982-01-01, family 4..7's
PENTAD_DAYS = 5
N_BINS = 3142                                       # bins 0..3141 = 1982..2024

# ONE definition of the ground scale, shared with ml/temporal.py (KM_PER_DEG)
# and ml/cone.py (which calls 0.25 deg x 111.32 = 27.83 km a cell). It is
# restated rather than imported because importing `cone` pulls in torch — 21 s
# and a GPU-shaped dependency for a reader that must run anywhere — and the
# test asserts the two literals agree by parsing ml/temporal.py.
KM_PER_DEG = 111.32
COS_FLOOR = 0.05                                    # ml/temporal.py's polar floor

# E-076 §2.6, for an Argo profile: a point measurement, taken in minutes.
# log2(point / 27.83 km) and log2(0.02 d / 5 d) = -5.6 both clamp to the
# plan's -4 floor. Constants for this source; FIELDS in the token, because the
# same schema carries coarser sources.
LOG2_FP_ARGO = -4.0
LOG2_DT_ARGO = -4.0

# The columns the store is made of. `bin_offsets` is the index, not a column.
COLUMNS = {
    "bin": "int16", "time_days": "float32", "lat": "float32", "lon": "float32",
    "temp": "float16", "psal": "float16", "wmo": "int32", "cycle": "int16",
    "nlev": "int16", "maxpres": "float16", "mode": "|S1",
}
VALUE_COLUMNS = ("temp", "psal")


def anchor_time_days(b):
    """The instant `dt_days` is measured back from: the END of bin `b`.

    See the module docstring — a bin is a five-day span, and an observation
    inside the anchor's own bin must not read as being in the future.
    """
    return float(PENTAD_DAYS * (int(b) + 1))


def offsets_km(lat0, lon0, lat, lon):
    """Signed (dx_km east, dy_km north) from anchor (lat0, lon0) to (lat, lon).

    Great-circle on the small-offset approximation the cone already uses:
    meridionally KM_PER_DEG per degree everywhere, zonally that times the
    cosine of the MEAN latitude of the pair, floored at COS_FLOOR. Longitude
    difference is wrapped into [-180, 180) so the dateline is not a wall.
    """
    lat = np.asarray(lat, np.float64)
    lon = np.asarray(lon, np.float64)
    dy = (lat - float(lat0)) * KM_PER_DEG
    dlon = np.mod(lon - float(lon0) + 180.0, 360.0) - 180.0
    coslat = np.maximum(np.cos(np.radians(0.5 * (lat + float(lat0)))), COS_FLOOR)
    dx = dlon * KM_PER_DEG * coslat
    return dx, dy


class ArgoStore:
    """The observation store, opened with memmaps. Read-only."""

    def __init__(self, path):
        self.path = os.path.abspath(path)
        mp = os.path.join(self.path, "store.json")
        if not os.path.exists(mp):
            raise FileNotFoundError(
                f"{mp} is missing — {self.path} is not a family-8 observation "
                f"store (build it with ml/build_family8_argo.py --stage profiles)")
        with open(mp) as fh:
            self.meta = json.load(fh)
        self.levels = np.asarray(self.meta["levels"], np.float64)
        self.n_levels = len(self.levels)
        self._col = {}
        for name in COLUMNS:
            p = os.path.join(self.path, name + ".npy")
            if not os.path.exists(p):
                raise FileNotFoundError(f"{p} is missing from the store")
            self._col[name] = np.load(p, mmap_mode="r")
        self.bin_offsets = np.load(os.path.join(self.path, "bin_offsets.npy"))
        self.N = int(self._col["bin"].shape[0])
        self.n_bins = int(len(self.bin_offsets) - 1)
        # The index must describe the data it indexes — an off-by-one here
        # would silently answer with a neighbouring pentad's floats.
        if int(self.bin_offsets[0]) != 0 or int(self.bin_offsets[-1]) != self.N:
            raise ValueError(
                f"bin_offsets runs {int(self.bin_offsets[0])}.."
                f"{int(self.bin_offsets[-1])} over {self.N} rows — the CSR "
                f"index does not describe this store")
        if int(self.meta.get("N", self.N)) != self.N:
            raise ValueError(f"store.json says N={self.meta['N']}, the arrays "
                             f"hold {self.N}")

    # -- columns -----------------------------------------------------------
    def __len__(self):
        return self.N

    def __getitem__(self, name):
        return self._col[name]

    def column(self, name):
        return self._col[name]

    def bins(self, b0, b1):
        """Every row of bins `b0..b1` INCLUSIVE, as a dict of array views.

        The CSR index makes this a slice rather than a search, which is also
        what makes the one-sided time rule free: `bins(b0, anchor)` cannot
        contain a later observation.
        """
        b0 = max(0, int(b0))
        b1 = min(self.n_bins - 1, int(b1))
        if b1 < b0:
            lo = hi = 0
        else:
            lo = int(self.bin_offsets[b0])
            hi = int(self.bin_offsets[b1 + 1])
        out = {k: v[lo:hi] for k, v in self._col.items()}
        out["row"] = np.arange(lo, hi, dtype=np.int64)
        return out

    def count(self, b0, b1):
        """How many observations lie in bins b0..b1 inclusive."""
        b0 = max(0, int(b0))
        b1 = min(self.n_bins - 1, int(b1))
        if b1 < b0:
            return 0
        return int(self.bin_offsets[b1 + 1]) - int(self.bin_offsets[b0])

    # -- the search --------------------------------------------------------
    def knearest(self, lat, lon, bin, k=5, R_max_km=1000.0, T_max_days=30.0,
                 L_km=None, T_days=None):
        """The k observations nearest `(lat, lon)` at or BEFORE pentad `bin`.

        Returns a dict of fixed length `k` — always `k` slots, so a batch is
        never ragged. Slot i is real where `valid[i]`, and a MISS TOKEN (NaN
        everywhere, `row = -1`) where it is not.

        Keys: `temp` and `psal` (k, n_levels) float32 in degC and PSU with NaN
        for an unfilled level; `dx_km`, `dy_km`, `dt_days`, `dist_km`, `d2`
        (k,); `log2_fp`, `log2_dt` (k,) — E-076 §2.6's footprint constants for
        this source; `n_R` (scalar) — how many observations lay inside
        `(R_max_km, T_max_days)`, the local density feature; `valid` (k,) bool;
        `row`, `wmo`, `cycle`, `mode`, `lat`, `lon`, `time_days` (k,) for
        provenance; and `n_found = min(k, n_R)`.

        `L_km` and `T_days` are the metric's scales (E-076 §2.1) and default to
        the search bounds. `n_R` counts what the BOUNDS admit, never what the
        metric ranks — it is a property of the observing system at this anchor,
        and must not move when a channel's correlation length is re-estimated.
        """
        k = int(k)
        if k < 1:
            raise ValueError(f"k must be >= 1, got {k}")
        b = int(bin)
        L = float(L_km if L_km is not None else R_max_km)
        T = float(T_days if T_days is not None else T_max_days)
        if L <= 0 or T <= 0:
            raise ValueError(f"metric scales must be positive, got L={L}, T={T}")
        R_max_km = float(R_max_km)
        T_max_days = float(T_max_days)

        out = self._miss(k)
        if b < 0 or self.N == 0:
            return out
        t_anchor = anchor_time_days(b)

        # ONE-SIDED, AND BOUNDED, BY CONSTRUCTION: the slice ends at the
        # anchor's own bin, and starts at the oldest bin that could still hold
        # an observation no older than T_max.
        b_hi = min(b, self.n_bins - 1)
        b_lo = max(0, int(np.floor((t_anchor - T_max_days) / PENTAD_DAYS)))
        if b_lo > b_hi:
            return out
        lo = int(self.bin_offsets[b_lo])
        hi = int(self.bin_offsets[b_hi + 1])
        if hi <= lo:
            return out

        t = np.asarray(self._col["time_days"][lo:hi], np.float64)
        dt_days = t_anchor - t
        ok = (dt_days >= 0.0) & (dt_days <= T_max_days)
        if not ok.any():
            return out

        dx, dy = offsets_km(lat, lon,
                            np.asarray(self._col["lat"][lo:hi], np.float64),
                            np.asarray(self._col["lon"][lo:hi], np.float64))
        r2 = dx * dx + dy * dy
        ok &= r2 <= R_max_km * R_max_km
        n_R = int(ok.sum())
        out["n_R"] = n_R
        if n_R == 0:
            return out

        d2 = r2 / (L * L) + (dt_days * dt_days) / (T * T)
        d2 = np.where(ok, d2, np.inf)
        # A stable sort, not argpartition: ties must resolve the same way on
        # every machine, and at <= ~15 k rows the sort is not the cost.
        order = np.argsort(d2, kind="stable")[:min(k, n_R)]
        n = len(order)
        rows = order + lo

        out["n_found"] = n
        out["valid"][:n] = True
        out["row"][:n] = rows
        out["dx_km"][:n] = dx[order]
        out["dy_km"][:n] = dy[order]
        out["dt_days"][:n] = dt_days[order]
        out["dist_km"][:n] = np.sqrt(r2[order])
        out["d2"][:n] = d2[order]
        out["lat"][:n] = self._col["lat"][rows]
        out["lon"][:n] = self._col["lon"][rows]
        out["time_days"][:n] = self._col["time_days"][rows]
        out["wmo"][:n] = self._col["wmo"][rows]
        out["cycle"][:n] = self._col["cycle"][rows]
        out["mode"][:n] = self._col["mode"][rows]
        for v in VALUE_COLUMNS:
            out[v][:n] = np.asarray(self._col[v][rows], np.float32)
        return out

    def _miss(self, k):
        """k miss tokens — the shape every answer has before it is filled."""
        nan1 = np.full(k, np.nan, np.float32)
        return {
            "k": k, "n_found": 0, "n_R": 0,
            "valid": np.zeros(k, bool),
            "row": np.full(k, -1, np.int64),
            "temp": np.full((k, self.n_levels), np.nan, np.float32),
            "psal": np.full((k, self.n_levels), np.nan, np.float32),
            "dx_km": nan1.copy(), "dy_km": nan1.copy(),
            "dt_days": nan1.copy(), "dist_km": nan1.copy(),
            "d2": nan1.copy(),
            "lat": nan1.copy(), "lon": nan1.copy(), "time_days": nan1.copy(),
            "wmo": np.zeros(k, np.int32), "cycle": np.zeros(k, np.int16),
            "mode": np.full(k, b" ", "|S1"),
            # Constants for Argo (E-076 §2.6), carried per token so the schema
            # survives a source with a varying footprint.
            "log2_fp": np.full(k, LOG2_FP_ARGO, np.float32),
            "log2_dt": np.full(k, LOG2_DT_ARGO, np.float32),
            "levels": self.levels,
        }

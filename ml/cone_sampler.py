#!/usr/bin/env python3
"""E-069 · gather the inner cone's raw values for a batch of anchors.

`ml/cone.py` says WHICH (lag, dy, dx) each channel reads; this file reads them
out of the tensor. It is the loader half of the cone codec, and it is the part
the plan expects to bound the run ("the sampler, not the network" — section 7),
because one anchor is ~750 scattered reads from a memmapped [T, 281, 481, C]
array and a batch is 256 of them.

Three properties are load-bearing:

  * **Offsets are per pixel ROW.** A cell is 27.83 km north-south everywhere
    and 27.83*cos(phi) km east-west, so the cone's dot list depends on latitude
    and nothing else — it is built once per row and cached, never per anchor.
  * **Values are RAW.** No climatology removal, no standardisation: that is the
    trainer's job and doing it here would bake one normalisation into every
    cached batch (ml/CLAUDE.md section 4.2 — normalise by properties of the
    DATA, in one place).
  * **Longitude wraps IF AND ONLY IF THE TENSOR IS A GLOBE, and the tensor
    says which it is.** Whether the cell west of column 0 is column W-1 is a
    fact about the LON AXIS, not a policy: `W * dlon == 360` means the axis
    closes on itself and the two are the same meridian; anything else means it
    does not. So the constructor measures it (`self.wrap`) and the sampler
    follows. The North Atlantic window is 481 columns of 0.25 deg = 120 deg
    and does not close — a wrap there would put the Iberian shelf one cell
    west of Florida, which is why this file used to refuse the wrap outright —
    while family 7's 1440 x 0.25 deg = 360 deg does close, and refusing there
    would cut the Pacific in half at the dateline for no reason (E-071
    section 1). Off-axis dots are INVALID only where the axis really ends:
    off the time axis at either end, and off the LATITUDE axis, which is
    clipped and never wrapped (the cell north of the pole is not a cell).

MULTI-GROUP TENSORS (family 7, E-070 section 1-3). A tensor may be one dense
array or THREE co-registered arrays at two resolutions: `g025` at 0.25 deg,
`g100` at 1 deg, `rg100` at 1 deg on the live Argo bins only. The cone is
placed on the DENSE grid — `g025`'s lat/lon are the master axes and every
(dy, dx) is a 0.25 deg cell offset — and a channel that lives in a coarse
group is read at the coarse cell the plan's lookup names (y1 = floor(y/f +
0.5) clipped, x1 = floor(x/f + 0.5) mod W_g, f the ratio of the two lat
steps), which is E-070 D3's "served as the same cell". A channel in a
live-bins group is read at bin t only if t is in that group's `bin_index`,
and is a MISS token otherwise — liveness is a property of the DATA, exactly
as `cone.channel_dots` already argues for the depth column. The token schema
does not change: values are indexed by the concatenated channel list
`chan_g025 + chan_g100 + chan_rg100`, and `obs` / `valid` mean what they
always meant.

Pool discipline (`admissible` / `certify`) generalises c25f6ff's
`--holdout-scope window` rule from one pixel-bin to the whole dot set: an
anchor is a training anchor only if every bin its cone touches — L_in pentads
back and every future target forward — is a training bin.

THE SPARSE GATHER (E-076 section 2, family 8). One group may be replaced by
RAW OBSERVATIONS instead of being read off the grid at all: hand the
constructor a `profile_store` (ml/family8_store.py::ArgoStore) and, for the
channels of the `rg100` group, the sunflower dots and the lag-0 patch are NOT
gathered — per anchor the k nearest Argo profiles become dot tokens carrying
their own (dy_km, dx_km, lag_days, depth), so the measurement is never
missing, only far away. `profile_store=None` (the default) is today's
behaviour to the bit, which is what tests/test_cone_global.py's digest pins.

Pure numpy; no torch. Plan: ml/plans/E069_cone_codec.md section 3,
ml/plans/E070_family7_build.md sections 1-3,
ml/plans/E076_family8_nearest_observations.md sections 2, 2.5, 2.6, 5.1.
"""
import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

from cone import (channel_dots, channel_depth_dbar, channel_family,   # noqa: E402
                  ground_km, KM_PER_DEG)

PENTAD_EPOCH = np.datetime64("1982-01-01")
PENTAD_DAYS = 5

# The 3x3 patch, in `ml/model.py::gather_px`'s order: dy outer, dx inner, both
# -1..1, so index 4 is the centre cell. Every archived codec's val_proj was
# trained on that layout; changing it silently changes what channel 0 means.
PATCH_DY = np.array([-1, -1, -1, 0, 0, 0, 1, 1, 1], np.int64)
PATCH_DX = np.array([-1, 0, 1, -1, 0, 1, -1, 0, 1], np.int64)

# ---------------------------------------------------------- the footprint --
# E-076 section 2.6: two extra numbers per dot saying what the measurement
# AVERAGED OVER, which its distance cannot say. `log2_fp` is the spatial
# support in 0.25-degree-cell units (27.83 km) and `log2_dt` the temporal
# support in pentads; the plan clamps both to [-4, +4], and they are divided
# by FP_SCALE here so the model reads numbers in [-1, 1].
#
# A 1-degree reanalysis value is NOT a 1-degree measurement: NCEP resolves at
# T62 (~1.9 degrees) and the 1-degree grid is how it is stored, so `g100`
# carries log2(1.9/0.25). `rg100` is a monthly optimally-interpolated field on
# a 1-degree grid, hence log2(4) in space and log2(30/5) in time. An Argo
# profile is a point measurement taken in minutes: both clamp to the floor.
FP_SCALE = 4.0
GROUP_FOOTPRINT = {
    "g025": (0.0, 0.0),
    "g100": (math.log2(1.9 / 0.25) / FP_SCALE, 0.0),
    "rg100": (math.log2(4.0) / FP_SCALE, math.log2(30.0 / 5.0) / FP_SCALE),
}
PROFILE_FOOTPRINT = (-4.0 / FP_SCALE, -4.0 / FP_SCALE)
# A single-array tensor (families 2-6) is one 0.25-degree grid sampled at its
# own resolution, and a group this table does not name gets the same: the
# honest default for "a cell read at the resolution it is stored at".
DEFAULT_FOOTPRINT = (0.0, 0.0)


def pentad_doy(t):
    """Day-of-year of pentad bin `t`. Pentads are 5-day bins from 1982-01-01
    (ml/build_family4.py), so bin t opens on day 5t and the season the codec's
    context token carries is that day's — a bin never straddles more than five
    days, which is finer than any seasonal term in the loss."""
    t = np.asarray(t, np.int64)
    d = PENTAD_EPOCH + (PENTAD_DAYS * t).astype("timedelta64[D]")
    return (d - d.astype("datetime64[Y]")).astype("timedelta64[D]").astype(
        np.int64) + 1


class Group:
    """One array of a multi-group tensor, with the axes that say how to read it.

    Parameters
    ----------
    name : str            the group's name in the tensor (`g025`, `g100`, ...)
    X : array-like        [Tg, Hg, Wg, Cg], memmap or ndarray, never copied
    lats, lons : 1-D      the group's OWN axes (`lat1`/`lon1` for a 1 deg group)
    chan : sequence[str]  its channel names, in its own channel order
    OBS : array-like or None
        its observed mask; None means "isfinite of the value", which is what
        every family since 2 has meant (`ml/train.py`'s LazyPixels).
    bin_index : 1-D int or None
        for a LIVE-BINS group: `bin_index[r]` is the master ROW that row r
        holds (`GroupSet.from_tensor` translates the npz's ABSOLUTE
        `rg_bin_index` through the master's own `bin_index`, because a build
        over a sub-range of the archive numbers its bins from the epoch and
        not from row 0). `rg100` writes one row per month, into the pentad
        containing the 15th (E-034 section 4), so 11 of every 12 bins have no
        row at all and a consumer that indexed it by bin would silently read a
        neighbouring month. None means the group is bin-aligned with the
        master.
    """

    def __init__(self, name, X, lats, lons, chan, OBS=None, bin_index=None):
        self.name = str(name)
        self.X = X
        self.OBS = OBS
        self.lats = np.asarray(lats, np.float64)
        self.lons = np.asarray(lons, np.float64)
        self.chan = [str(c) for c in chan]
        self.T, self.H, self.W, self.C = (int(v) for v in X.shape)
        if len(self.lats) != self.H or len(self.lons) != self.W:
            raise ValueError(
                f"group {self.name}: axes ({len(self.lats)}/{len(self.lons)}) "
                f"do not match the array ({self.H}/{self.W})")
        if len(self.chan) != self.C:
            raise ValueError(
                f"group {self.name}: {len(self.chan)} channel names for "
                f"{self.C} channels")
        self.bin_index = (None if bin_index is None
                          else np.asarray(bin_index, np.int64))
        # Set by GroupSet once the master grid is known.
        self.factor = 1
        self.row_of_bin = None
        # Set by ConeSampler._prep_groups; declared here so the attribute
        # exists whatever order a caller does things in.
        self.flatX = self.flatO = None

    def __repr__(self):                                  # pragma: no cover
        return (f"<Group {self.name} {self.T}x{self.H}x{self.W}x{self.C} "
                f"factor {self.factor}"
                f"{' live-bins' if self.bin_index is not None else ''}>")


class GroupSet:
    """The co-registered arrays of a multi-group tensor, in reading order.

    The FIRST group is the master: its lat/lon axes are the grid the cone is
    placed on and its time axis is the tensor's. `tensor_io.load_tensor`
    already declares that order (the npz's `groups` key, which is also what
    fixes which group `d["X"]` aliases), so this class never chooses it.
    """

    def __init__(self, groups):
        self.groups = list(groups)
        if not self.groups:
            raise ValueError("GroupSet: no groups")
        m = self.groups[0]
        if m.bin_index is not None:
            raise ValueError(
                f"GroupSet: the master group {m.name!r} carries a bin index — "
                f"the dense group defines the time axis and cannot be a "
                f"live-bins group")
        dlat_m = _axis_step(m.lats, f"group {m.name}")
        for g in self.groups:
            f = _axis_step(g.lats, f"group {g.name}") / dlat_m
            g.factor = int(round(f))
            if g.factor < 1 or abs(f - g.factor) > 1e-6:
                raise ValueError(
                    f"group {g.name}: its latitude step is {f:g}x the master "
                    f"group {m.name}'s, which is not a whole number — the "
                    f"coarse lookup of E-070 section 1 is defined on an "
                    f"integer factor and guessing one would read a "
                    f"neighbouring cell")
            if g.factor == 1 and (len(g.lats) != m.H or len(g.lons) != m.W):
                raise ValueError(
                    f"group {g.name}: same latitude step as the master but a "
                    f"different shape ({g.H}x{g.W} vs {m.H}x{m.W})")
            if g.bin_index is None:
                if g.T != m.T:
                    raise ValueError(
                        f"group {g.name}: {g.T} bins against the master's "
                        f"{m.T} and no bin index — a group that is not "
                        f"bin-aligned MUST say which master bin each of its "
                        f"rows holds, or every value it returns is from some "
                        f"other date")
            else:
                if len(g.bin_index) != g.T:
                    raise ValueError(
                        f"group {g.name}: bin index has {len(g.bin_index)} "
                        f"entries for {g.T} rows")
                row = np.full(m.T, -1, np.int64)
                inside = (g.bin_index >= 0) & (g.bin_index < m.T)
                row[g.bin_index[inside]] = np.flatnonzero(inside)
                g.row_of_bin = row
        self.master = m
        self.chan = [c for g in self.groups for c in g.chan]
        self.lats, self.lons = m.lats, m.lons
        self.shape = (m.T, m.H, m.W, len(self.chan))

    @property
    def names(self):
        return [g.name for g in self.groups]

    def __repr__(self):                                  # pragma: no cover
        return f"<GroupSet {self.names} C={len(self.chan)}>"

    @classmethod
    def from_tensor(cls, d, arrays=None, obs=None):
        """Build a GroupSet from a `tensor_io.load_tensor` result.

        `arrays` optionally replaces a group's array with one the caller owns
        — the anomaly-transformed writable copy the trainer and the exporter
        both make, which must not be written back into the canonical tensor
        (`ml/tensor_io.py::writable_copy`'s whole argument). `obs` optionally
        supplies per-group observed masks; without it a value is observed
        exactly where it is finite.
        """
        names = [str(g) for g in (d["groups"] if "groups" in d
                                  else getattr(d, "groups", []))]
        if not names:
            raise ValueError(
                "GroupSet.from_tensor: this tensor declares no `groups` key — "
                "it is a single-array tensor and ConeSampler takes it "
                "directly")
        arrays = arrays or {}
        obs = obs or {}
        lats, lons = np.asarray(d["lats"]), np.asarray(d["lons"])
        master_T = int((arrays.get(names[0]) if names[0] in arrays
                        else d[f"X_{names[0]}"]).shape[0])
        out = []
        for g in names:
            X = arrays[g] if g in arrays else d[f"X_{g}"]
            key = f"chan_{g}"
            if key not in d:
                raise ValueError(
                    f"GroupSet.from_tensor: the npz has no {key!r} — the "
                    f"channel names ARE the family map (ml/cone.py::"
                    f"channel_family) and a group without them cannot be read")
            chan = [str(c) for c in d[key]]
            gl, gn = _axes_for(d, g, X, lats, lons)
            bi = None
            if int(X.shape[0]) != master_T:
                bi = _bin_rows_for(d, g, master_T)
            out.append(Group(g, X, gl, gn, chan, OBS=obs.get(g),
                             bin_index=bi))
        return cls(out)


def group_time(g, moy, t_hold):
    """(moy, t_hold) restricted to ONE group's own rows.

    A bin-aligned group shares the master's arrays. A live-bins group holds
    one row per month, so its row r belongs to master row `bin_index[r]` and
    must be charged to THAT row's calendar month — hand it the master's
    3,142-long arrays and every profile is attributed to the wrong month and
    the wrong side of the holdout. A row whose master bin is off this axis
    (-1, only possible on a sub-range build) is marked HELD OUT, so it can
    never enter a climatology or a pooled moment.

    Both `ml/train_cone.py::load_data_family7` and
    `ml/export_cone_sample.py` call this, so there is one derivation.
    """
    moy = np.asarray(moy)
    t_hold = np.asarray(t_hold, bool)
    if g.bin_index is None:
        return moy, t_hold
    rows = np.asarray(g.bin_index, np.int64)
    ok = rows >= 0
    idx = np.clip(rows, 0, len(moy) - 1)
    return np.where(ok, moy[idx], 0), np.where(ok, t_hold[idx], True)


def _axis_step(a, what):
    """The (constant) step of a monotone axis. A one-point axis has none."""
    a = np.asarray(a, np.float64)
    if len(a) < 2:
        raise ValueError(f"{what}: an axis of {len(a)} point(s) has no step")
    return float(a[1] - a[0])


def _axes_for(d, g, X, lats, lons):
    """The lat/lon axes of group `g`, from the npz, never inferred from shape.

    A shape-derived factor would be right for the point grids family 7 uses
    and wrong the first time a group is cell-centred; the axes are in the
    file, so they are read from it. The order tried is the specific key
    (`lats_<g>`), then the master axes, then the plan's 1 deg axes
    (`lat1`/`lon1`), and a group that matches none of them raises.
    """
    if f"lats_{g}" in d and f"lons_{g}" in d:
        return np.asarray(d[f"lats_{g}"]), np.asarray(d[f"lons_{g}"])
    H, W = int(X.shape[1]), int(X.shape[2])
    if H == len(lats) and W == len(lons):
        return lats, lons
    if "lat1" in d and "lon1" in d:
        a, b = np.asarray(d["lat1"]), np.asarray(d["lon1"])
        if len(a) == H and len(b) == W:
            return a, b
    raise ValueError(
        f"GroupSet.from_tensor: group {g!r} is {H}x{W} and the npz carries no "
        f"axes of that length (tried lats_{g}/lons_{g}, lats/lons, lat1/lon1) "
        f"— the coarse lookup is derived from the LAT STEPS, so a group "
        f"without axes cannot be placed on the dense grid")


def _bin_rows_for(d, g, master_T):
    """Which master ROW each row of a live-bins group holds.

    The npz stores ABSOLUTE pentad bins (counted from 1982-01-01), for both
    the master axis (`bin_index`) and the live group (`rg_bin_index`). On a
    full build those two coincide with row numbers because the axis starts at
    bin 0; on any sub-range build — every `--smoke` and every partial rebuild
    — they do not, and taking the absolute bin for a row is off by the axis's
    own offset. So the translation goes through the master's `bin_index`, and
    a live bin that is not on the master axis maps to -1 (that row is simply
    never read) rather than to whatever happens to sit at that index.
    """
    src = None
    for key in (f"{g}_bin_index", f"{g.rstrip('0123456789')}_bin_index",
                "rg_bin_index"):
        if key in d:
            src = np.asarray(d[key], np.int64)
            break
    if src is None:
        raise ValueError(
            f"GroupSet.from_tensor: group {g!r} has its own number of rows "
            f"but no bin index (tried {g}_bin_index, rg_bin_index). A "
            f"live-bins group read by row would hand the cone another "
            f"month's profile.")
    if "bin_index" not in d:
        return src
    master = np.asarray(d["bin_index"], np.int64)
    if len(master) != master_T:
        raise ValueError(
            f"GroupSet.from_tensor: the npz's bin_index has {len(master)} "
            f"entries for a master group of {master_T} rows")
    pos = np.searchsorted(master, src)
    pos_c = np.clip(pos, 0, master_T - 1)
    return np.where(master[pos_c] == src, pos_c, -1).astype(np.int64)


class ProfileGather:
    """The k nearest Argo profiles of one anchor, IN THE MODEL'S SPACE.

    E-076 sections 2.1 and 2.5. `ml/family8_store.py::ArgoStore.knearest` does
    the search and returns RAW degrees Celsius and PSU; this class does the
    two things that make those numbers comparable with the tensor's:

      1. the build-time z-score of the group they replace
         (`norm_rg100`, [Cg, 2] of mean and sd, in the group's channel order);
      2. `ml/trainprobe.py::anomaly_transform`'s own climatology and pooled
         moments, captured by its `stats=` argument and evaluated AT THE
         PROFILE'S OWN CELL AND CALENDAR MONTH — not the anchor's, because the
         anomaly belongs to where the measurement was made.

            z = (raw - norm_mean_c) / norm_sd_c
            a = (z - clim[month, y1, x1, c] - mu_c) / den_c

    A (cell, month) with no climatology gives NaN and the channel reads
    UNOBSERVED there, exactly as the gridded path does — `X - NaN` is NaN and
    `isfinite` is what makes a cell observed.

    THE BIN AXIS IS ABSOLUTE IN THE STORE AND RELATIVE IN THE TENSOR. A
    sub-range build numbers its rows from 0 while the store numbers its bins
    from 1982-01-01, so `bin_index` (the npz's own absolute bins per row) is
    required to translate, in both directions: the anchor's row -> the bin the
    search is run at, and a returned profile's bin -> the row whose training
    flag governs it. A profile whose bin is not on the tensor's axis is not a
    training observation (the same -1 rule `_bin_rows_for` applies).
    """

    def __init__(self, store, group, chan_base, norm, stats, k=5,
                 R_max_km=1000.0, T_max_days=30.0, bin_index=None, T=None):
        self.store = store
        self.group = group
        self.k = int(k)
        self.R_max_km = float(R_max_km)
        self.T_max_days = float(T_max_days)
        self.n_chan = int(group.C)
        # GLOBAL channel indices, by NAME through the group's own chan list —
        # never by position, because the concatenation order is a property of
        # the tensor and not of this file.
        self.chan = np.arange(chan_base, chan_base + self.n_chan, dtype=np.int64)
        self.names = list(group.chan)
        self.depth = np.array([channel_depth_dbar(n) for n in self.names],
                              np.float32)
        # Which store column and which of its 16 levels each channel is.
        levels = np.asarray(store.levels, np.float64)
        self.value_col, self.level = [], []
        for n in self.names:
            if n.startswith("rg_t"):
                self.value_col.append(0)
            elif n.startswith("rg_s"):
                self.value_col.append(1)
            else:
                raise ValueError(
                    f"ProfileGather: channel {n!r} of group {group.name!r} is "
                    f"neither a temperature (rg_t*) nor a salinity (rg_s*) "
                    f"column, so the store has no counterpart for it. The "
                    f"store's 16 levels ARE the rg100 group's "
                    f"(docs/FAMILY8_DATA_HANDOVER.md section 3); refusing "
                    f"rather than guessing which value to hand the model.")
            d = float(channel_depth_dbar(n))
            j = int(np.argmin(np.abs(levels - d)))
            if abs(levels[j] - d) > 1e-6:
                raise ValueError(
                    f"ProfileGather: channel {n!r} sits at {d} dbar and the "
                    f"store has no such level ({list(levels)})")
            self.level.append(j)
        self.value_col = np.array(self.value_col, np.int64)
        self.level = np.array(self.level, np.int64)

        norm = np.asarray(norm, np.float64)
        if norm.shape != (self.n_chan, 2):
            raise ValueError(
                f"ProfileGather: norm must be [{self.n_chan}, 2] (the "
                f"build-time mean and sd of group {group.name!r}), got "
                f"{norm.shape}. Without it a raw degree Celsius cannot be put "
                f"into the space the model was trained in.")
        self.norm_mean, self.norm_sd = norm[:, 0], norm[:, 1]
        for key in ("clim", "mu"):
            if key not in (stats or {}):
                raise ValueError(
                    f"ProfileGather: the anomaly stats are missing {key!r} — "
                    f"pass ml/trainprobe.py::anomaly_transform its `stats=` "
                    f"dict for this group. A profile put into the model's "
                    f"space by any other climatology is a second anomaly "
                    f"transform (tests/test_one_anomaly_transform.py).")
        self.clim = np.asarray(stats["clim"])
        self.mu = np.asarray(stats["mu"], np.float64)
        self.den = np.asarray(stats["den"] if "den" in stats
                              else np.asarray(stats["sd"], np.float64) + 1e-6,
                              np.float64)
        if self.clim.shape[0] != 12 or self.clim.shape[-1] != self.n_chan:
            raise ValueError(
                f"ProfileGather: clim is {self.clim.shape}, want "
                f"[12, H, W, {self.n_chan}] for group {group.name!r}")
        # The group's OWN axes, so a profile lands on the cell the tensor
        # would have served it from.
        self.lat0 = float(group.lats[0])
        self.dlat = float(group.lats[1] - group.lats[0])
        self.lon0 = float(group.lons[0])
        self.dlon = float(group.lons[1] - group.lons[0])
        self.H1, self.W1 = int(group.H), int(group.W)
        self.wrap1 = abs(self.W1 * self.dlon - 360.0) < 1e-6
        self.bin_index = (None if bin_index is None
                          else np.asarray(bin_index, np.int64))
        self.T = int(T if T is not None else
                     (len(self.bin_index) if self.bin_index is not None else 0))

    # ------------------------------------------------------------- the axis --
    def abs_bin(self, t):
        """The ABSOLUTE pentad bin of master row `t` (identity without an
        index, which is the full-archive build)."""
        if self.bin_index is None:
            return int(t)
        return int(self.bin_index[int(t)])

    def rows_of_bins(self, bins):
        """Master row of each ABSOLUTE bin, -1 where the bin is off the axis."""
        bins = np.asarray(bins, np.int64)
        if self.bin_index is None:
            ok = (bins >= 0) & (bins < self.T)
            return np.where(ok, bins, -1)
        pos = np.searchsorted(self.bin_index, bins)
        pos_c = np.clip(pos, 0, len(self.bin_index) - 1)
        return np.where(self.bin_index[pos_c] == bins, pos_c, -1)

    # ------------------------------------------------------------ the space --
    def cell_of(self, lat, lon):
        """`(y1, x1)` of the group's cell nearest a profile's position.

        The same rounding the coarse lookup uses (`floor(v + 0.5)`), on the
        group's own axes rather than on the master grid, and the longitude
        wraps only where the axis closes.
        """
        y1 = np.floor((np.asarray(lat, np.float64) - self.lat0) / self.dlat
                      + 0.5).astype(np.int64)
        x1 = np.floor((np.asarray(lon, np.float64) - self.lon0) / self.dlon
                      + 0.5).astype(np.int64)
        y1 = np.clip(y1, 0, self.H1 - 1)
        x1 = np.mod(x1, self.W1) if self.wrap1 else np.clip(x1, 0, self.W1 - 1)
        return y1, x1

    @staticmethod
    def month_of(time_days):
        """Calendar month index 0..11 of `days since 1982-01-01`."""
        d = (PENTAD_EPOCH
             + np.floor(np.asarray(time_days, np.float64)).astype("timedelta64[D]"))
        return (d.astype("datetime64[M]").astype(np.int64) % 12).astype(np.int64)

    def to_model(self, raw, month, y1, x1):
        """`(vals [n, Cg] float32, obs [n, Cg] bool)` — the chain of E-076
        section 2.5, evaluated at the profiles' own cells and months."""
        raw = np.asarray(raw, np.float64)
        if raw.size == 0:
            return (np.zeros(raw.shape, np.float32),
                    np.zeros(raw.shape, bool))
        z = (raw - self.norm_mean[None, :]) / self.norm_sd[None, :]
        cl = np.asarray(self.clim[month, y1, x1, :], np.float64)  # [n, Cg]
        a = (z - cl - self.mu[None, :]) / self.den[None, :]
        obs = np.isfinite(a)
        return np.where(obs, a, 0.0).astype(np.float32), obs

    # ------------------------------------------------------------ the search --
    def knearest(self, lat, lon, t, k):
        """`ArgoStore.knearest` at master row `t`, one-sided by construction."""
        return self.store.knearest(float(lat), float(lon), bin=self.abs_bin(t),
                                   k=int(k), R_max_km=self.R_max_km,
                                   T_max_days=self.T_max_days)

    def train_ok(self, res, train_bins):
        """[k] bool: is each returned slot admissible for a TRAINING sample?

        A profile from a held-out bin must never enter a training input, even
        though the window rule of `ConeSampler.admissible` already makes that
        impossible for an admitted anchor (T_max = 30 days reaches back five
        pentads and the rule admits t-L_in..t). This is the check that says so
        rather than the argument that says so, and it DROPS a slot rather than
        shifting the next one up — a shifted slot would silently be a
        different, farther measurement.
        """
        valid = np.asarray(res["valid"], bool).copy()
        if train_bins is None:
            return valid
        rows = np.asarray(res["row"], np.int64)
        good = np.zeros(len(valid), bool)
        hit = valid & (rows >= 0)
        if hit.any():
            bins = np.asarray(self.store["bin"][rows[hit]], np.int64)
            mrow = self.rows_of_bins(bins)
            good[hit] = (mrow >= 0) & np.asarray(train_bins, bool)[
                np.clip(mrow, 0, max(len(train_bins) - 1, 0))]
        return valid & good


class ConeSampler:
    """Gather the inner cone for anchors (t, y, x) out of X[T, H, W, C].

    Parameters
    ----------
    X : array-like [T, H, W, C], or a `GroupSet`
        The tensor. A single array is the family 2-6 case and behaves exactly
        as it always has. A `GroupSet` (family 7) is the multi-resolution
        case: the cone is placed on its master group's grid and every channel
        is read out of the group that owns it. Either may be a memmap or an
        npz member; both are indexed lazily and NEVER materialised (`X[:]` on
        the pentad tensor is 33 GB — ml/model.py::LazyPixels was written for
        exactly this failure).
    OBS : array-like [T, H, W, C], or None
        The observed mask for the single-array form. With a `GroupSet` the
        masks live in the groups and this must be None.
    lats, lons : 1-D arrays
        The grid axes, used for the per-row cos(phi), for the context token
        and — the lon axis — to decide whether the tensor is a globe and
        therefore wraps.
    chan_names : sequence of str
        Channel names in tensor order; each is mapped to a cone family by
        `cone.channel_family` and an unknown name raises there rather than
        being given a silent default reach. With a `GroupSet` this is the
        CONCATENATED list `chan_g025 + chan_g100 + chan_rg100`, which is what
        a value index means everywhere downstream.
    L_in : int
        Inner-window depth in pentads (6 = 30 days; plan section 2 argues the
        number from displacement per lag, not from convenience).
    future_lags : tuple of int
        Which forward bins the decoder is asked for (1, 2 = t+1, t+2).

    Attributes
    ----------
    wrap : bool
        True when the longitude axis closes on itself (`W * dlon == 360`), in
        which case a dot that leaves the east edge re-enters at the west one.
        MEASURED from the axis, never passed in: a flag would be a second
        place the tensor's shape is described, and the two would disagree.
    """

    def __init__(self, X, OBS, lats, lons, chan_names, L_in=6, dlat_deg=0.25,
                 future_lags=(1, 2), profile_store=None, profile_k=5,
                 profile_group="rg100", profile_norm=None, profile_stats=None,
                 profile_R_max_km=1000.0, profile_T_max_days=30.0,
                 profile_bin_index=None, train_bins=None):
        self.gs = X if isinstance(X, GroupSet) else None
        self.X, self.OBS = X, OBS
        self.lats = np.asarray(lats, np.float64)
        self.lons = np.asarray(lons, np.float64)
        self.chan_names = list(chan_names)
        self.L_in = int(L_in)
        self.dlat_deg = float(dlat_deg)
        self.future_lags = tuple(int(f) for f in future_lags)
        self.T, self.H, self.W, self.C = (int(v) for v in X.shape)
        if len(self.chan_names) != self.C:
            raise ValueError(
                f"ConeSampler: {len(self.chan_names)} channel names for a "
                f"tensor with {self.C} channels — the names ARE the family "
                f"map, so a mismatch means some channel is reading another "
                f"channel's cone.")
        if len(self.lats) != self.H or len(self.lons) != self.W:
            raise ValueError(
                f"ConeSampler: lats/lons ({len(self.lats)}/{len(self.lons)}) "
                f"do not match the tensor ({self.H}/{self.W}).")
        # Does the longitude axis close? W * dlon == 360 to a rounding error.
        # 481 x 0.25 = 120.25 (the North Atlantic window, open); 1440 x 0.25 =
        # 360.0 (family 7, closed). Measured, not declared — see `wrap`.
        self.wrap = bool(
            self.W > 1 and
            abs(self.W * (self.lons[1] - self.lons[0]) - 360.0) < 1e-6)
        self.families = [channel_family(n) for n in self.chan_names]
        self.depths = np.array([channel_depth_dbar(n) for n in self.chan_names],
                               np.float32)
        self._rows = {}
        if self.gs is None:
            self._flatX = self._flat(X)
            self._flatO = self._flat(OBS)
        else:
            if OBS is not None:
                raise ValueError(
                    "ConeSampler: a GroupSet carries each group's observed "
                    "mask (Group.OBS); a single OBS array over the "
                    "concatenated channels would have to be dense at 0.25 "
                    "degrees, which is the 425 GB the three-group layout "
                    "exists to avoid (E-070 B2).")
            if self.chan_names != self.gs.chan:
                raise ValueError(
                    "ConeSampler: chan_names must be the groups' channel "
                    "lists concatenated in group order "
                    f"({' + '.join(g.name for g in self.gs.groups)}), because "
                    f"that concatenation IS what a value index means "
                    f"downstream. Got {self.chan_names[:3]}... expected "
                    f"{self.gs.chan[:3]}...")
            self._flatX = self._flatO = None
            self._prep_groups()
        self._prep_profiles(profile_store, profile_k, profile_group,
                            profile_norm, profile_stats, profile_R_max_km,
                            profile_T_max_days, profile_bin_index, train_bins)

    # ------------------------------------------------------ the sparse path --
    def _prep_profiles(self, store, k, group_name, norm, stats, R_max_km,
                       T_max_days, bin_index, train_bins):
        """Wire the family-8 observation store in, or leave every attribute at
        the value that means "today's behaviour" (E-076 section 5).

        `profile_k = 0` is a THIRD state and a deliberate one: the store is
        opened and the nearest-profile TARGET is available, but no profile dot
        enters the input and the gridded group is read exactly as it always
        was. That is the family-7 TWIN of E-076 section 5.1 — the arm that
        keeps `rg100` and must still be scored on the same measurement — and
        it is why the two arms can share one eval implementation.
        """
        self.train_bins = (None if train_bins is None
                           else np.asarray(train_bins, bool))
        self.profile = None
        self.profile_k = 0
        self._prof_chan = None          # [Cg] global indices of the group
        self._dense_ci = None           # channels the DENSE path still reads
        self._skip = frozenset()        # channels with no sunflower dots
        # The per-channel footprint (E-076 section 2.6) is a property of the
        # GROUP a channel lives in, and it is produced on the dense path too,
        # so the model has one input contract whichever arm is running.
        self._fp_chan = np.zeros((self.C, 2), np.float32)
        if self.gs is not None:
            base = 0
            for g in self.gs.groups:
                fp = GROUP_FOOTPRINT.get(g.name, DEFAULT_FOOTPRINT)
                self._fp_chan[base:base + g.C] = fp
                base += g.C
        if store is None:
            return
        if self.gs is None:
            raise ValueError(
                "ConeSampler: a profile store replaces the channels of ONE "
                "group of a multi-group tensor, and this tensor is a single "
                "array with no groups to replace. Refusing rather than "
                "guessing which of its channels the profiles stand for.")
        if isinstance(store, str):
            from family8_store import ArgoStore
            store = ArgoStore(store)
        base, grp = 0, None
        for g in self.gs.groups:
            if g.name == group_name:
                grp = g
                break
            base += g.C
        if grp is None:
            raise ValueError(
                f"ConeSampler: no group named {group_name!r} in "
                f"{self.gs.names} — the profile store stands in for a group's "
                f"channels and there is none of that name.")
        self.profile = ProfileGather(
            store, grp, base, norm, stats, k=k, R_max_km=R_max_km,
            T_max_days=T_max_days, bin_index=bin_index, T=self.T)
        self.profile_k = int(k)
        self._prof_chan = self.profile.chan
        if self.profile_k > 0:
            # THE GROUP IS NOT READ AT ALL on this arm. Not read and then
            # discarded — not read: the dense gather is given the other
            # channels only, so a test can poison the array and see nothing
            # move (tests/test_cone_profiles.py).
            self._skip = frozenset(int(c) for c in self._prof_chan)
            self._dense_ci = np.array(
                [c for c in range(self.C) if c not in self._skip], np.int64)
            self._rows = {}             # the dot table changed shape

    @property
    def n_profile_dots(self):
        """How many profile dot tokens every anchor carries — k * Cg, always,
        which is what keeps a batch rectangular however thin the array is."""
        return 0 if self.profile is None else self.profile_k * self.profile.n_chan

    def _prep_groups(self):
        """Per-channel group routing, and each group's flat view."""
        gi, li = [], []
        for k, g in enumerate(self.gs.groups):
            gi += [k] * g.C
            li += list(range(g.C))
            g.flatX = self._flat(g.X)
            g.flatO = self._flat(g.OBS) if g.OBS is not None else None
        self._chan_group = np.array(gi, np.int64)
        self._chan_local = np.array(li, np.int64)

    # ------------------------------------------------------------ internals --
    @staticmethod
    def _flat(A):
        """A 1-D VIEW of a C-contiguous array, or None. One fancy index into a
        flat memmap is a single scatter-gather; the four-index form walks four
        broadcast index arrays and measured ~2x slower on the pentad shape."""
        try:
            if A.flags["C_CONTIGUOUS"]:
                return A.reshape(-1)
        except Exception:
            pass
        return None

    def _gather(self, A, flat, t, y, x, c, H=None, W=None, C=None):
        """A[t, y, x, c] for broadcast index arrays, flat where possible."""
        H = self.H if H is None else H
        W = self.W if W is None else W
        C = self.C if C is None else C
        if flat is not None:
            idx = ((t * H + y) * W + x) * C + c
            return flat[idx.ravel()].reshape(idx.shape)
        t, y, x, c = np.broadcast_arrays(t, y, x, c)
        return A[t, y, x, c]

    def _read(self, t, y, x, c):
        """(values float32, observed bool) at MASTER-grid cells (t, y, x) for
        global channel index `c`. Index arrays broadcast against each other.

        NaN is preserved — `sample` is what turns a missing value into the
        zero the codec's miss token carries, and the exporter needs the NaN.
        Indices must already be inside the master grid (the callers clip, and
        wrap the longitude when `self.wrap`).
        """
        if self.gs is None:
            raw = np.asarray(self._gather(self.X, self._flatX, t, y, x, c),
                             np.float32)
            o = np.asarray(self._gather(self.OBS, self._flatO, t, y, x, c),
                           bool)
            return raw, o
        t, y, x, c = np.broadcast_arrays(t, y, x, c)
        vals = np.full(t.shape, np.nan, np.float32)
        obs = np.zeros(t.shape, bool)
        for k, g in enumerate(self.gs.groups):
            m = self._chan_group[c] == k
            if not m.any():
                continue
            gt, gy, gx = t[m], y[m], x[m]
            gc = self._chan_local[c[m]]
            live = None
            if g.factor != 1:
                # E-070 section 1's lookup, with the factor DERIVED from the
                # two lat steps rather than the plan's literal 4: every f-th
                # master point IS a coarse point and the points either side of
                # it round to it. Latitude clips at the coarse pole row;
                # longitude wraps, because the coarse axis closes for the same
                # reason the dense one does.
                gy = np.minimum(np.floor(gy / g.factor + 0.5).astype(np.int64),
                                g.H - 1)
                gx = np.floor(gx / g.factor + 0.5).astype(np.int64) % g.W
            if g.row_of_bin is not None:
                r = g.row_of_bin[gt]
                live = r >= 0
                gt = np.maximum(r, 0)
            raw = np.asarray(self._gather(g.X, g.flatX, gt, gy, gx, gc,
                                          g.H, g.W, g.C), np.float32)
            if g.OBS is None:
                o = np.isfinite(raw)
            else:
                o = np.asarray(self._gather(g.OBS, g.flatO, gt, gy, gx, gc,
                                            g.H, g.W, g.C), bool)
            if live is not None:
                # A bin the group never wrote is a MISS, not a zero and not
                # the nearest month: `rg100` holds one row per month and the
                # eleven-twelfths of bins with no row must read as unobserved.
                raw = np.where(live, raw, np.nan)
                o &= live
            vals[m] = raw
            obs[m] = o
        return vals, obs

    def read_cells(self, t, y, x, c=None):
        """RAW values and observed flags at master-grid cells, NaN preserved.

        `sample` is the training path and zeroes a missing value (the codec's
        miss token); this is the INSPECTION path — `ml/export_cone_sample.py`
        writes `null` for a value the tensor does not have, and a zero there
        would be a measurement the page would draw. `t`, `y`, `x` broadcast;
        `c` defaults to every channel and is appended as a trailing axis.
        """
        t = np.asarray(t, np.int64)
        y = np.asarray(y, np.int64)
        x = np.asarray(x, np.int64)
        if self.wrap:
            x = np.mod(x, self.W)
        if c is None:
            c = np.arange(self.C, dtype=np.int64)
            t, y, x = t[..., None], y[..., None], x[..., None]
        c = np.asarray(c, np.int64)
        return self._read(t, y, x, c)

    def row(self, y):
        """The cone's flattened dot table for grid row `y` (cached).

        Keys: `lag`, `dy`, `dx`, `chan` (index arrays over the dot set),
        `dy_km`, `dx_km`, `lag_days`, `depth` (the per-dot coordinates the
        codec's Fourier encoding reads), and `n`.
        """
        y = int(y)
        hit = self._rows.get(y)
        if hit is not None:
            return hit
        lat = float(self.lats[y])
        lag, dy, dx, chan = [], [], [], []
        for ci, name in enumerate(self.chan_names):
            if ci in self._skip:
                # A channel served by the profile store has no sunflower: its
                # dots ARE the k nearest observations (E-076 section 2).
                continue
            for l, ddy, ddx in channel_dots(lat, name, L_in=self.L_in,
                                            dlat_deg=self.dlat_deg):
                lag.append(l)
                dy.append(ddy)
                dx.append(ddx)
                chan.append(ci)
        lag = np.array(lag, np.int64)
        dy = np.array(dy, np.int64)
        dx = np.array(dx, np.int64)
        chan = np.array(chan, np.int64)
        ykm, xkm = ground_km(dy.astype(np.float64), dx.astype(np.float64),
                             lat, self.dlat_deg)
        rec = dict(
            n=int(len(lag)), lat=lat, lag=lag, dy=dy, dx=dx, chan=chan,
            chan16=chan.astype(np.int16),
            dy_km=ykm.astype(np.float32), dx_km=xkm.astype(np.float32),
            lag_days=(PENTAD_DAYS * lag).astype(np.float32),
            depth=self.depths[chan],
        )
        self._rows[y] = rec
        return rec

    def n_dots(self, y):
        """Token count of the dot set at row `y` (the patch adds C more)."""
        return self.row(y)["n"]

    # --------------------------------------------------------------- sample --
    def sample(self, anchors, withhold_nearest=False, train_pool=True):
        """Gather the inner cone, the lag-0 patch and the future targets.

        `withhold_nearest` (family 8 only) drops the NEAREST profile out of the
        input and returns it as `profile_target`: the masked-profile objective
        of E-076 section 7.5 and the common target of section 5.1. It may be a
        single bool or a [B] array, so a batch can withhold on a fraction of
        its anchors without being split in two.

        `train_pool=True` (the default) applies the constructor's `train_bins`
        to the profiles, so a TRAINING sample can never read an observation
        from a held-out bin. An EVAL sample passes `train_pool=False`: dots
        from held-out bins are what makes a held-out measurement held out.

        `anchors` is an int array [B, 3] of (t, y, x). Returns a dict of numpy
        arrays padded to a fixed N = max dot count over the batch:

          vals[B, N]      float32, raw tensor values (NaN -> 0)
          obs[B, N]       bool, observed in the DATA
          valid[B, N]     bool, the dot exists: on the grid and 0 <= t-l < T.
                          Padding is invalid, so `valid` is the attention mask.
          chan[B, N]      int16 channel index
          dy_km, dx_km    float32 signed ground offsets, km
          lag_days[B, N]  float32, 5 * lag
          depth[B, N]     float32 dbar (0 for surface channels)
          patch_vals[B, C, 9], patch_obs[B, C, 9]   the lag-0 3x3
          fut_vals[B, C, F], fut_obs[B, C, F]       the anchor at t+f
          ctx[B, 4]       sin/cos of the season, lat/90, lon/180
          anchors[B, 3]   echoed, so a downstream cache is self-describing
          nr[B, N]        float32 log1p(n_R) on a PROFILE dot, 0 on every
                          other dot — E-076 section 2.1's local density
          fp[B, N, 2]     float32 the two footprint fields of section 2.6,
                          in units of a quarter of the plan's [-4, 4] range

        `nr` and `fp` are produced on BOTH paths, so `ml/cone_codec.py` has one
        input contract; on the dense path `nr` is all zeros and `fp` is the
        per-group constant of `GROUP_FOOTPRINT`.
        """
        anchors = np.asarray(anchors, np.int64)
        if anchors.ndim != 2 or anchors.shape[1] != 3:
            raise ValueError(f"sample(): anchors must be [B, 3], got "
                             f"{anchors.shape}")
        B = anchors.shape[0]
        t, y, x = anchors[:, 0], anchors[:, 1], anchors[:, 2]

        bands = {}
        for i in range(B):
            bands.setdefault(int(y[i]), []).append(i)
        n_prof = self.n_profile_dots
        N = max((self.n_dots(r) for r in bands), default=0) + n_prof

        vals = np.zeros((B, N), np.float32)
        obs = np.zeros((B, N), bool)
        valid = np.zeros((B, N), bool)
        chan = np.zeros((B, N), np.int16)
        dy_km = np.zeros((B, N), np.float32)
        dx_km = np.zeros((B, N), np.float32)
        lag_days = np.zeros((B, N), np.float32)
        depth = np.zeros((B, N), np.float32)
        nr = np.zeros((B, N), np.float32)
        fp = np.zeros((B, N, 2), np.float32)

        for r, ii in bands.items():
            R = self.row(r)
            n = R["n"]
            idx = np.asarray(ii, np.int64)
            tt = t[idx][:, None] - R["lag"][None, :]
            yy = r + R["dy"][None, :]
            xx = x[idx][:, None] + R["dx"][None, :]
            cc = R["chan"][None, :]
            ok = (tt >= 0) & (tt < self.T) & (yy >= 0) & (yy < self.H)
            if self.wrap:
                # The lon axis closes, so leaving the east edge is not leaving
                # the tensor — it is arriving at the west one. Latitude is
                # still clipped: there is no cell north of the pole.
                xx = np.mod(xx, self.W)
            else:
                ok = ok & (xx >= 0) & (xx < self.W)
            ok = np.broadcast_to(ok, (len(idx), n))
            raw, o = self._read(np.clip(tt, 0, self.T - 1),
                                np.clip(yy, 0, self.H - 1),
                                np.clip(xx, 0, self.W - 1), cc)
            raw = np.where(np.isfinite(raw), raw, 0.0).astype(np.float32)
            sl = (idx[:, None], np.arange(n)[None, :])
            vals[sl] = raw
            obs[sl] = np.asarray(o, bool) & ok
            valid[sl] = ok
            chan[sl] = R["chan16"][None, :]
            dy_km[sl] = R["dy_km"][None, :]
            dx_km[sl] = R["dx_km"][None, :]
            lag_days[sl] = R["lag_days"][None, :]
            depth[sl] = R["depth"][None, :]
            fp[sl] = self._fp_chan[R["chan"]][None, :, :]
        # A dot the data never observed is still a real token (PixelMAE's
        # `miss_tok`); only a dot that does not EXIST is invalid.
        vals = np.where(valid, vals, 0.0).astype(np.float32)

        patch_vals, patch_obs = self._patch(t, y, x)
        fut_vals, fut_obs = self._future(t, y, x)
        doy = pentad_doy(t).astype(np.float64)
        ang = 2.0 * np.pi * doy / 365.0
        ctx = np.stack([np.sin(ang), np.cos(ang),
                        self.lats[y] / 90.0, self.lons[x] / 180.0],
                       axis=1).astype(np.float32)
        out = dict(vals=vals, obs=obs, valid=valid, chan=chan,
                   dy_km=dy_km, dx_km=dx_km, lag_days=lag_days, depth=depth,
                   patch_vals=patch_vals, patch_obs=patch_obs,
                   fut_vals=fut_vals, fut_obs=fut_obs, ctx=ctx,
                   anchors=anchors, nr=nr, fp=fp)
        if self.profile is not None:
            self._gather_profiles(out, anchors, withhold_nearest, train_pool,
                                  N - n_prof)
        return out

    # ------------------------------------------------- family 8's dot block --
    def _gather_profiles(self, out, anchors, withhold_nearest, train_pool,
                         col0):
        """Fill the profile dot block (and `profile_target`) in place.

        The block occupies the LAST `k * Cg` columns of the padded dot axis, in
        slot-major order (slot 0's 32 channels, then slot 1's), so a reader can
        address slot s channel c at `col0 + s * Cg + c`. Columns between the
        longest row's dense dots and `col0` are padding and stay invalid.

        One `knearest` per anchor: the search reads at most seven bins of a
        CSR-sliced store (~5,000 rows on the real archive), which is far
        cheaper than the dense gather it replaces.
        """
        P = self.profile
        B = anchors.shape[0]
        Cg = P.n_chan
        k = self.profile_k
        wh = np.zeros(B, bool) | np.asarray(withhold_nearest, bool)
        tb = self.train_bins if train_pool else None

        lev = P.level
        col = P.value_col
        tgt = {
            "valid": np.zeros(B, bool),
            "temp": np.full((B, P.store.n_levels), np.nan, np.float32),
            "psal": np.full((B, P.store.n_levels), np.nan, np.float32),
            "levels": np.asarray(P.store.levels, np.float32),
            "dy_km": np.zeros(B, np.float32), "dx_km": np.zeros(B, np.float32),
            "dt_days": np.zeros(B, np.float32),
            "lat": np.full(B, np.nan, np.float32),
            "lon": np.full(B, np.nan, np.float32),
            "time_days": np.full(B, np.nan, np.float32),
            "y1": np.zeros(B, np.int64), "x1": np.zeros(B, np.int64),
            "month": np.zeros(B, np.int64), "n_R": np.zeros(B, np.int64),
            # The ready-made decoder QUERY for the withheld profile: 32
            # (channel, coordinate) pairs and their targets in the model's
            # space, so the masked-profile objective rides the existing
            # dot-query machinery instead of a second one.
            "q_chan": np.tile(P.chan, (B, 1)),
            "q_dy_km": np.zeros((B, Cg), np.float32),
            "q_dx_km": np.zeros((B, Cg), np.float32),
            "q_lag_days": np.zeros((B, Cg), np.float32),
            "q_depth": np.tile(P.depth, (B, 1)),
            "q_vals": np.zeros((B, Cg), np.float32),
            "q_obs": np.zeros((B, Cg), bool),
            # PERSISTENCE (E-076 section 7.5's second baseline): the nearest
            # REMAINING profile's raw values, i.e. the slot the input keeps.
            "pers_valid": np.zeros(B, bool),
            "pers_temp": np.full((B, P.store.n_levels), np.nan, np.float32),
            "pers_psal": np.full((B, P.store.n_levels), np.nan, np.float32),
        }
        for i in range(B):
            ti = int(anchors[i, 0])
            lat = float(self.lats[int(anchors[i, 1])])
            lon = float(self.lons[int(anchors[i, 2])])
            drop = bool(wh[i])
            if not drop and k == 0:
                # The twin (k = 0) with nothing withheld asks the store
                # nothing at all: its input is the gridded group and there is
                # no target to fetch.
                continue
            # WITHHOLDING ALWAYS FETCHES AT LEAST TWO: slot 0 is the target
            # and slot 1 is the persistence baseline, which is a SCORING
            # quantity and not an input — so the twin, whose k is 0, still
            # gets the number E-076 §7.5 says to compare against.
            n_ask = k + 1 if drop else k
            res = P.knearest(lat, lon, ti, max(n_ask, 2) if drop else n_ask)
            ok = P.train_ok(res, tb)
            n_R = max(int(res["n_R"]) - (1 if drop else 0), 0)
            tgt["n_R"][i] = n_R

            if drop:
                if ok[0]:
                    self._fill_target(tgt, i, P, res, 0)
                if ok.shape[0] > 1 and ok[1]:
                    tgt["pers_valid"][i] = True
                    tgt["pers_temp"][i] = res["temp"][1]
                    tgt["pers_psal"][i] = res["psal"][1]
                sl = slice(1, k + 1)
            else:
                sl = slice(0, k)
            if k == 0:
                continue

            good = ok[sl]                                   # [k]
            # [2, k, 16] -> [k, Cg]: channel c reads column `col[c]` (temp or
            # psal) at level `lev[c]`, which is the mapping ProfileGather
            # derived from the channel NAMES.
            both = np.stack([res["temp"][sl], res["psal"][sl]], axis=0)
            raw = both[col, :, lev].T                       # [k, Cg]
            month = P.month_of(res["time_days"][sl])
            y1, x1 = P.cell_of(np.nan_to_num(res["lat"][sl], nan=0.0),
                               np.nan_to_num(res["lon"][sl], nan=0.0))
            v, o = P.to_model(raw, month, y1, x1)
            o &= good[:, None]
            lo = col0
            hi = lo + k * Cg
            out["vals"][i, lo:hi] = np.where(o, v, 0.0).reshape(-1)
            out["obs"][i, lo:hi] = o.reshape(-1)
            # EVERY slot is a real token, present or missing — that is the
            # whole point of family 8 (a rectangular k, section 2.2). Only a
            # slot the search could not fill is `valid=False`, which is the
            # miss token the codec already has.
            out["valid"][i, lo:hi] = np.repeat(good, Cg)
            out["chan"][i, lo:hi] = np.tile(P.chan, k).astype(np.int16)
            out["dy_km"][i, lo:hi] = np.repeat(
                np.nan_to_num(res["dy_km"][sl], nan=0.0), Cg)
            out["dx_km"][i, lo:hi] = np.repeat(
                np.nan_to_num(res["dx_km"][sl], nan=0.0), Cg)
            out["lag_days"][i, lo:hi] = np.repeat(
                np.nan_to_num(res["dt_days"][sl], nan=0.0), Cg)
            out["depth"][i, lo:hi] = np.tile(P.depth, k)
            out["nr"][i, lo:hi] = np.log1p(float(n_R))
            out["fp"][i, lo:hi, 0] = PROFILE_FOOTPRINT[0]
            out["fp"][i, lo:hi, 1] = PROFILE_FOOTPRINT[1]
        if bool(wh.any()) or withhold_nearest is True:
            out["profile_target"] = tgt

    def _fill_target(self, tgt, i, P, res, s):
        """Slot `s` of one anchor's search, as the withheld TARGET."""
        tgt["valid"][i] = True
        tgt["temp"][i] = res["temp"][s]
        tgt["psal"][i] = res["psal"][s]
        tgt["dy_km"][i] = res["dy_km"][s]
        tgt["dx_km"][i] = res["dx_km"][s]
        tgt["dt_days"][i] = res["dt_days"][s]
        tgt["lat"][i] = res["lat"][s]
        tgt["lon"][i] = res["lon"][s]
        tgt["time_days"][i] = res["time_days"][s]
        month = int(P.month_of(np.array([res["time_days"][s]]))[0])
        y1, x1 = P.cell_of(np.array([res["lat"][s]]),
                           np.array([res["lon"][s]]))
        tgt["month"][i] = month
        tgt["y1"][i] = int(y1[0])
        tgt["x1"][i] = int(x1[0])
        raw = np.stack([res["temp"][s], res["psal"][s]])[
            P.value_col, P.level][None, :]                   # [1, Cg]
        v, o = P.to_model(raw, np.array([month]), y1, x1)
        tgt["q_vals"][i] = v[0]
        tgt["q_obs"][i] = o[0]
        tgt["q_dy_km"][i] = np.float32(res["dy_km"][s])
        tgt["q_dx_km"][i] = np.float32(res["dx_km"][s])
        tgt["q_lag_days"][i] = np.float32(res["dt_days"][s])

    def _patch(self, t, y, x):
        """The lag-0 3x3 for every channel, [B, C, 9] — `gather_px`'s tokens.

        A cell off the NORTH or SOUTH edge is unobserved (latitude is clipped,
        never wrapped). A cell off the east or west edge is unobserved on a
        window and WRAPPED on a globe, by the same `self.wrap` the dots use.

        On a multi-group tensor a coarse channel's nine cells are the SAME
        coarse cell nine times wherever the 3x3 does not straddle a coarse
        boundary — E-070 D3's "served as the same cell", which is the honest
        answer for a 1 degree field sampled at 0.25 degrees, and the reason
        the patch's shape does not change.

        On the family-8 arm the channels the profile store replaces are NOT
        READ: they are absent from `_dense_ci`, so the gather never touches
        their group's array, and their nine cells come back unobserved with a
        zero value — the miss token the codec already has for "the data does
        not have this", which is exactly true of a group the arm does not read.
        """
        B = len(t)
        C = self.C
        yy = y[:, None] + PATCH_DY[None, :]                    # [B, 9]
        xx = x[:, None] + PATCH_DX[None, :]
        ok = ((yy >= 0) & (yy < self.H)
              & (t[:, None] >= 0) & (t[:, None] < self.T))
        if self.wrap:
            xx = np.mod(xx, self.W)
        else:
            ok = ok & (xx >= 0) & (xx < self.W)
        yc = np.clip(yy, 0, self.H - 1)[:, None, :]            # [B, 1, 9]
        xc = np.clip(xx, 0, self.W - 1)[:, None, :]
        tc = np.clip(t, 0, self.T - 1)[:, None, None]
        ci = (np.arange(C, dtype=np.int64) if self._dense_ci is None
              else self._dense_ci)
        cc = ci[None, :, None]
        raw, o = self._read(tc, yc, xc, cc)
        ok3 = np.broadcast_to(ok[:, None, :], (B, len(ci), 9))
        raw = np.where(np.isfinite(raw) & ok3, raw, 0.0).astype(np.float32)
        return self._scatter_c(raw, o & ok3, ci, B, 9)

    def _future(self, t, y, x):
        """The anchor column at t+f for each f in `future_lags`, [B, C, F].
        Past the end of the tensor the target does not exist, so it is
        unobserved — never a zero the decoder could be scored against.

        No wrap here and none needed: the only cell this reads is the ANCHOR's
        own (y, x), which is on the grid by construction. Only the offsets
        move, and the offsets are in `sample` and `_patch`.
        """
        B, C = len(t), self.C
        F = len(self.future_lags)
        f = np.array(self.future_lags, np.int64)
        tt = t[:, None] + f[None, :]                            # [B, F]
        ok = (tt >= 0) & (tt < self.T)
        tc = np.clip(tt, 0, self.T - 1)[:, None, :]             # [B, 1, F]
        yc = y[:, None, None]
        xc = x[:, None, None]
        ci = (np.arange(C, dtype=np.int64) if self._dense_ci is None
              else self._dense_ci)
        cc = ci[None, :, None]
        raw, o = self._read(tc, yc, xc, cc)
        ok3 = np.broadcast_to(ok[:, None, :], (B, len(ci), F))
        raw = np.where(np.isfinite(raw) & ok3, raw, 0.0).astype(np.float32)
        return self._scatter_c(raw, o & ok3, ci, B, F)

    def _scatter_c(self, raw, obs, ci, B, W):
        """Put a gather over a SUBSET of channels back on the [B, C, W] axis.

        A channel that was not gathered reads 0.0 / unobserved — the family-8
        arm's honest statement about the group it does not read. When every
        channel was gathered (`ci` is the full range) the arrays are returned
        untouched, so the dense path is not merely equivalent but identical.
        """
        if len(ci) == self.C:
            return raw, obs
        v = np.zeros((B, self.C, W), np.float32)
        o = np.zeros((B, self.C, W), bool)
        v[:, ci, :] = raw
        o[:, ci, :] = obs
        return v, o

    # ----------------------------------------------------------- pool rules --
    def bin_span(self):
        """The bins one anchor touches, as offsets from t: -F .. +L_in, i.e.
        every t - l for l <= L_in and every t + f."""
        back = list(range(0, self.L_in + 1))
        fwd = [-f for f in self.future_lags]
        return np.array(sorted(set(back + fwd)), np.int64)

    def admissible(self, anchors, train_bins):
        """[B] bool: is every bin this anchor's cone touches a training bin?

        c25f6ff's `--holdout-scope window` rule said a training pixel-bin may
        not read a held-out bin. The cone reads L_in pentads of history and
        predicts two forward, so the same rule over the whole span is what
        keeps a held-out bin out of the training set by ANY path — including
        the one that leaks hardest, a future target inside the holdout.

        A bin outside [0, T) is not a training bin: an anchor whose cone runs
        off the archive is inadmissible, not silently short.
        """
        anchors = np.asarray(anchors, np.int64)
        train_bins = np.asarray(train_bins, bool)
        if train_bins.shape != (self.T,):
            raise ValueError(f"admissible(): train_bins must be [T={self.T}], "
                             f"got {train_bins.shape}")
        t = anchors[:, 0]
        bins = t[:, None] - self.bin_span()[None, :]
        inside = (bins >= 0) & (bins < self.T)
        good = np.where(inside, train_bins[np.clip(bins, 0, self.T - 1)], False)
        return good.all(axis=1)

    def certify(self, anchors, train_bins):
        """Brute-force count of pool violations over `anchors` — 0 for an
        admitted batch.

        Deliberately a plain loop over anchors and dots and NOT a rearrangement
        of `admissible`: a certificate written from the same expression it
        certifies proves only that the expression is self-consistent. E-059
        ran the same check before training; this is its cone-shaped form, and
        the trainer calls it once per run, not per batch.
        """
        anchors = np.asarray(anchors, np.int64)
        train_bins = np.asarray(train_bins, bool)
        T = self.T
        bad = 0
        for a in range(anchors.shape[0]):
            t = int(anchors[a, 0])
            hit = False
            if self.profile is not None:
                # E-076 section 2.2's one-sided rule, certified rather than
                # argued: the store's CSR slice cannot reach past the anchor's
                # own bin, and the age is measured back from the END of that
                # bin — so no returned observation may be dated later than
                # 5 * (bin + 1) days after the epoch. A forecaster that has
                # seen the next pentad is not a forecaster.
                P = self.profile
                res = P.knearest(float(self.lats[int(anchors[a, 1])]),
                                 float(self.lons[int(anchors[a, 2])]), t,
                                 max(self.profile_k, 1) + 1)
                v = np.asarray(res["valid"], bool)
                if v.any():
                    tmax = float(np.nanmax(np.asarray(res["time_days"])[v]))
                    if tmax > PENTAD_DAYS * (P.abs_bin(t) + 1):
                        hit = True
            for lag in range(0, self.L_in + 1):
                b = t - lag
                if b < 0 or b >= T or not bool(train_bins[b]):
                    hit = True
                    break
            if not hit:
                for f in self.future_lags:
                    b = t + f
                    if b < 0 or b >= T or not bool(train_bins[b]):
                        hit = True
                        break
            bad += int(hit)
        return bad

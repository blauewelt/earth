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

Pure numpy; no torch. Plan: ml/plans/E069_cone_codec.md section 3,
ml/plans/E070_family7_build.md sections 1-3.
"""
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
                 future_lags=(1, 2)):
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
    def sample(self, anchors):
        """Gather the inner cone, the lag-0 patch and the future targets.

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
        N = max((self.n_dots(r) for r in bands), default=0)

        vals = np.zeros((B, N), np.float32)
        obs = np.zeros((B, N), bool)
        valid = np.zeros((B, N), bool)
        chan = np.zeros((B, N), np.int16)
        dy_km = np.zeros((B, N), np.float32)
        dx_km = np.zeros((B, N), np.float32)
        lag_days = np.zeros((B, N), np.float32)
        depth = np.zeros((B, N), np.float32)

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
        return dict(vals=vals, obs=obs, valid=valid, chan=chan,
                    dy_km=dy_km, dx_km=dx_km, lag_days=lag_days, depth=depth,
                    patch_vals=patch_vals, patch_obs=patch_obs,
                    fut_vals=fut_vals, fut_obs=fut_obs, ctx=ctx,
                    anchors=anchors)

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
        cc = np.arange(C, dtype=np.int64)[None, :, None]
        raw, o = self._read(tc, yc, xc, cc)
        ok3 = np.broadcast_to(ok[:, None, :], (B, C, 9))
        raw = np.where(np.isfinite(raw) & ok3, raw, 0.0).astype(np.float32)
        return raw, (o & ok3)

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
        cc = np.arange(C, dtype=np.int64)[None, :, None]
        raw, o = self._read(tc, yc, xc, cc)
        ok3 = np.broadcast_to(ok[:, None, :], (B, C, F))
        raw = np.where(np.isfinite(raw) & ok3, raw, 0.0).astype(np.float32)
        return raw, (o & ok3)

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

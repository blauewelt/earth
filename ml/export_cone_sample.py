#!/usr/bin/env python3
"""E-069 · export what the codec and stage 2 actually READ, for a few anchors.

`ml/export_cone_geometry.py` exports the cone's SHAPE — which (lag, dy, dx) is
read. This exports the VALUES at those positions, out of the production tensor,
so the globe app's Cones tab can show the data rather than a drawing of where
the data would be.

THE ONE RULE HERE: this file gathers nothing itself.

  * the INNER cone (the codec's stencil — the lag-0 3x3 patch, the lag-1..6
    sunflower dots, the two future targets, and the `valid` / `obs` flags) is
    read by calling `ml/cone_sampler.py::ConeSampler.sample`, the same object
    `ml/train_cone.py` trains on;
  * the OUTER cone (stage 2's stencil, lags k = 7..143) is read by calling
    `ml/cone.py::outer_spiral` per latitude row — the same function
    `ml/cone.py::coverage_report` measures the union with.

A second implementation of either would drift silently, which is the argument
`ml/export_cone_geometry.py` already makes for the geometry and
`tests/test_export_cone_sample.py` pins here: the test asserts the exported
`valid` / `obs` are bit-identical to a direct `ConeSampler.sample` call.

For k <= 6 the outer annulus is EMPTY by construction (r_lo(k) = r_in(k) is the
same formula as r_hi(k) there, so stage 2 keeps only the anchor column at the
lags the codec already read in full — `ml/cone.py::outer_spiral`'s docstring).
The output says so in `meta.outer.empty_below` rather than leaving a reader to
wonder where lags 0-6 went.

VALUE SPACE. `ConeSampler` returns whatever is in the array it was given, by
design (its docstring: "Values are RAW ... that is the trainer's job"). The
trainer hands it an array that `ml/trainprobe.py::anomaly_transform` has
already turned into anomalies — departures from a per-calendar-month
climatology built on TRAINING YEARS ONLY, then z-scored per channel over the
training pool. So the honest export carries both, and says which is which:

    raw     the tensor's own stored value — which `ml/build_family4.py` has
            already z-scored per channel, so it is in standard deviations and
            NOT in the channel's unit; `meta.value_space.tensor_norm` carries
            the (mean, std) that puts the unit back
    anom    what the codec reads: that value with its calendar month's
            climatology removed and re-standardised

`--anomaly trainer` computes the anomaly the trainer's way. It cannot call
`anomaly_transform` directly on the production tensor here, because that
function writes the whole 35.7 GB array in place and this sandbox has 30 GB of
disk; `streaming_anomaly` is the same three passes over a stream that is
decompressed twice instead of stored once, and
`tests/test_export_cone_sample.py::test_streaming_anomaly_matches_trainprobe`
asserts the two agree BIT FOR BIT on a toy tensor. `--anomaly none` emits raw
values plus the tensor's own per-channel (mean, std) and says so in `meta`.

Output is one compact JSON per anchor: columnar arrays, values rounded to four
significant digits, `null` for NaN, keys sorted — so a regeneration is a no-op
diff and a reader can diff two runs.

FAMILY 7 (E-070/E-071 Phase E). `--tensor <stem>.npz` reads the three-group
GLOBAL tensor instead: `tensor_io.load_tensor` memory-maps each group (no npz
streaming — the sidecar layout exists precisely so a 46 GB group need not be
decompressed), the anomaly is `ml/trainprobe.py::anomaly_transform` itself on
a writable copy of each group, the anchor list grows from five North Atlantic
points to twelve global ones, and the exported JSON gains `meta.groups`,
`meta.channel_group`, `meta.grid`, `meta.recipe` and the anchor's `sphere` and
`elev` — every existing key unchanged, so a reader of the family-4 files reads
these too. The outer stencil is computed on the global grid WITH THE WRAP, so
an anchor on the dateline has a whole cone rather than half of one.

Run (toy):
    python3 ml/export_cone_sample.py --smoke --out /tmp/cone_samples
Run (production, streaming the release npz):
    python3 ml/export_cone_sample.py \
        --stream-npz /path/family4_na025_pentad_r3_fa460837fa.npz \
        --out data_out/cone_samples --anomaly trainer
Run (family 7, on the box beside the tensor):
    python3 ml/export_cone_sample.py \
        --tensor ml/cache/family7/family7_global025_pentad_l0.npz \
        --anomaly trainer --outer-stride 2 --out "$RUNNER_TEMP/cone_out"
"""
import argparse
import datetime as _dt
import hashlib
import json
import math
import os
import subprocess
import sys
import zipfile

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import cone                                                        # noqa: E402
from cone_sampler import ConeSampler, PENTAD_EPOCH, PENTAD_DAYS    # noqa: E402

GEOMETRY = os.path.join(ROOT, "data", "cone_geometry.json")

# The eight channels a number can be scored on at pentad cadence — DERIVED in
# `ml/lim_baseline.py` (`observed_channels`) and quoted here, not re-derived:
# `build_family4.fill_rg_pentad` fills one live pentad per month, so the 32
# Argo depth channels are unobserved at almost every bin and the LIM's
# `per_channel` carries these eight names and no others.
SCOREABLE = ("cur_speed", "log_mld", "ssh", "tau_x", "tau_y",
             "tau_x_std", "tau_y_std", "sst")
# Family 7 carries `sst` under its own name. E-070 B4 briefly merged it into a
# `skin_t` channel spanning both surfaces; E-071 section 6.1's correction of
# 4 Sep split it back out (a channel is shared only when the measurand AND the
# instrument match on both sides), so the scoreable set needs no renaming.
SCOREABLE_F7 = SCOREABLE

# What one number MEANS, for the read-out. A value without its unit is a
# number the reader cannot check against anything (root CLAUDE.md §2.4).
UNITS = {
    "cur_speed": "m/s", "cur_u": "m/s", "cur_v": "m/s",
    "log_mld": "log10(m)", "ssh": "m",
    "tau_x": "N/m²", "tau_y": "N/m²",
    "tau_x_std": "N/m²", "tau_y_std": "N/m²",
    "sst": "°C",
}

# WHAT `raw` IS, and it is not what the word suggests on its own.
# `ml/build_family4.py` stores the tensor Z-SCORED — its last pass writes
# (x - mu) / sd at float16 and keeps (mu, sd) per channel in the npz's `norm`
# key. So the number `ConeSampler` hands the codec is in STANDARD DEVIATIONS of
# that channel over the whole archive, and the value in the channel's own unit
# (degrees, m/s, metres, N/m^2) is `raw * sd + mu`. Both are carried: `raw` so
# the export is the tensor's own bytes, `tensor_norm` so anything reading it can
# put the unit back. The ANOMALY is a further transform ON TOP of `raw` — the
# calendar-month climatology removed and re-standardised — which is why the two
# columns differ even though both are dimensionless.
RAW_NOTE = ("the tensor's own stored value, which the builder "
            "(ml/build_family4.py, ml/build_family7.py) z-scored per channel: "
            "it is in standard deviations, not in the channel's unit")
PHYSICAL_NOTE = ("the value in the channel's own unit is "
                 "raw * tensor_norm[c][1] + tensor_norm[c][0] "
                 "(tensor_norm is the npz's `norm` key: mean, std per "
                 "channel; on family 7 it is keyed by GROUP — norm_g025, "
                 "norm_g100, norm_rg100 — and indexed within the group, "
                 "which meta.channel_group names)")

# The development holdout (ml/plans/E059_holdout_window.md — the run that fixed
# a training pool which had been teacher-forcing those years into the weights)
# and the terminal split (train <= 2020, test 2021-2024; E-069 plan §5).
HOLDOUT_YEARS = (2009, 2017, 2023)
TERMINAL_TRAIN_LAST_YEAR = 2020

# The anchors the Cones tab already offers, plus one at the window's eastern
# edge. The fifth exists so the page can show what an INADMISSIBLE neighbour
# looks like: at 19 E every dot more than four cells east is off the tensor
# window, which the model reads as missing and never wraps (ConeSampler's
# module docstring), and the Ionian basin puts land inside the inner cone too.
ANCHORS = [
    dict(id="gulf_stream", name="Gulf Stream", lat=36.0, lon=-70.0),
    dict(id="rapid", name="RAPID 26.5° N", lat=26.5, lon=-70.0),
    dict(id="labrador", name="Labrador Sea", lat=58.0, lon=-52.0),
    dict(id="equator", name="Equator", lat=0.0, lon=-30.0),
    dict(id="ionian_edge", name="Ionian Sea (window's east edge)",
         lat=36.0, lon=19.0),
]

# `--smoke` runs the identical code path over a 40 x 48 toy window at
# 30 N / 40 W, so its anchors have to live inside that window rather than in
# the North Atlantic one.
SMOKE_ANCHORS = [
    dict(id="smoke_mid", name="toy centre", lat=34.75, lon=-34.0),
    dict(id="smoke_edge", name="toy east edge", lat=34.75, lon=-28.5),
]

# ---------------------------------------------------------------------------
# THE GLOBAL ANCHOR LIST (family 7). The five North Atlantic anchors above are
# kept unchanged so the two tensors can be compared at the same points, and
# seven more are added, each chosen for a SURFACE or an EDGE the North
# Atlantic window does not contain — which is the whole reason the global
# tensor was built.
GLOBAL_ANCHORS = ANCHORS + [
    dict(id="antarctica_ice", name="Antarctic ice sheet", lat=-80.0, lon=60.0,
         note="an ice-sheet anchor: `sphere` = 2 and no ocean channel is "
              "observed at all, so what the cone reads here is the SHARED "
              "land/ice channels — skin temperature, 2 m air, wind, pressure, "
              "precipitation, snow, the turbulent fluxes. Under family 4 this "
              "cell did not exist; under E-070's 681-row draft it was still "
              "below the grid's floor (E-071 §1)."),
    dict(id="sahara", name="Sahara", lat=23.0, lon=10.0,
         note="land, and the driest of it: soil moisture and snow are the "
              "channels with something to say, and every ocean channel is a "
              "miss token."),
    dict(id="greenland", name="Greenland ice sheet", lat=72.0, lon=-40.0,
         note="the second ice sheet, and the one whose meltwater the AMOC "
              "labels are about."),
    dict(id="kuroshio", name="Kuroshio", lat=35.0, lon=145.0,
         note="the Pacific's western boundary current — the Gulf Stream's "
              "twin, and the first strong current in this project's input "
              "that is not in the Atlantic."),
    dict(id="acc", name="Antarctic Circumpolar Current", lat=-55.0, lon=60.0,
         note="the only current that closes on itself around the planet, "
              "which is exactly the geometry a wrapping sampler is for."),
    dict(id="nino34", name="Niño 3.4", lat=0.0, lon=-150.0,
         note="the equatorial Pacific box ENSO is defined on."),
    dict(id="dateline", name="Dateline (0 N, 179.75 E)", lat=0.0, lon=179.75,
         note="the last column of the tensor. Every dot east of it wraps to "
              "the first columns; on the North Atlantic window the same "
              "anchor would have had a cone that was 50 % invalid."),
]


# --------------------------------------------------------------- pentad time --
def bin_of_date(iso):
    """The pentad bin holding `iso`. Bins are fixed five-day bins counted from
    1982-01-01 (`ml/build_family4.py`), so this is floor(days / 5)."""
    d = np.datetime64(iso, "D")
    return int((d - PENTAD_EPOCH).astype("timedelta64[D]").astype(np.int64)
               // PENTAD_DAYS)


def date_of_bin(b):
    """The calendar day the pentad bin `b` OPENS on."""
    d = PENTAD_EPOCH + np.timedelta64(int(b) * PENTAD_DAYS, "D")
    return str(d)


def first_bin_on_or_after(iso):
    b = bin_of_date(iso)
    return b if date_of_bin(b) >= iso else b + 1


# ------------------------------------------------------------------ rounding --
def sig(x, digits=4):
    """`x` at `digits` significant figures, or None if it is not a number.

    Four figures is the width at which the page's read-out and the tensor agree
    to more places than a float16 tensor actually carries (float16 has ~3.3
    decimal digits), so nothing here rounds away a real distinction — and
    `null` for NaN is what JSON has; `NaN` is a bare token no strict reader
    parses (ml/CLAUDE.md §5.22).
    """
    if x is None:
        return None
    v = float(x)
    if not math.isfinite(v):
        return None
    if v == 0.0:
        return 0.0
    q = round(v, -int(math.floor(math.log10(abs(v)))) + (digits - 1))
    # Kill a -0.0 and any float noise the round-trip leaves behind.
    return float(f"{q:.{digits}g}")


def bits(a):
    """A bool array as a compact "0"/"1" string — 1 byte per flag instead of
    the 3 a JSON `0,` costs. The outer stencil is 24 dates x 69 lags x 24 dots
    x 8 channels of them."""
    return "".join("1" if v else "0" for v in np.asarray(a).ravel())


# -------------------------------------------------------- streaming anomaly --
class _NpzMemberStream:
    """Sequential reader over one uncompressed-on-the-fly `.npy` member.

    `np.load(npz)` DECOMPRESSES the member into RAM — 35.7 GB for the pentad
    tensor, on a box with 7 (`ml/tensor_io.py`'s opening paragraph). Deflate
    has no random access, but it streams at ~480 MB/s here, so a pass over the
    whole tensor costs about ninety seconds of CPU and no memory at all.
    """

    def __init__(self, path, member="X.npy"):
        self.path, self.member = path, member
        self.zf = zipfile.ZipFile(path)
        info = [i for i in self.zf.infolist() if i.filename == member]
        if not info:
            raise SystemExit(f"{path}: no member {member!r} — members are "
                             f"{[i.filename for i in self.zf.infolist()]}")
        self.size = info[0].file_size

    def open(self):
        """(file object positioned after the header, shape, dtype)."""
        fh = self.zf.open(self.member)
        version = np.lib.format.read_magic(fh)
        # numpy renamed the private `_read_array_header` between 1.x and 2.4;
        # the PUBLIC per-version readers have been there the whole time, so
        # dispatch on the magic rather than on a name that moves.
        reader = {(1, 0): np.lib.format.read_array_header_1_0,
                  (2, 0): np.lib.format.read_array_header_2_0}.get(version)
        if reader is None:
            raise SystemExit(f"{self.member}: unsupported .npy version "
                             f"{version}")
        shape, fortran, dtype = reader(fh)
        if fortran:
            raise SystemExit(f"{self.member}: Fortran order is not supported")
        return fh, shape, dtype

    def blocks(self, chunk):
        """Yield (t0, t1, block) over the time axis, in order."""
        fh, shape, dtype = self.open()
        T, H, W, C = shape
        per = H * W * C
        itemsize = np.dtype(dtype).itemsize
        for t0 in range(0, T, chunk):
            t1 = min(t0 + chunk, T)
            want = (t1 - t0) * per * itemsize
            buf = bytearray(want)
            view = memoryview(buf)
            got = 0
            while got < want:
                n = fh.readinto(view[got:])
                if not n:
                    raise SystemExit(
                        f"{self.member}: stream ended {want - got} bytes into "
                        f"block {t0}:{t1} — the archive is truncated")
                got += n
            yield t0, t1, np.frombuffer(buf, dtype=dtype).reshape(
                (t1 - t0, H, W, C))
        fh.close()


def _alloc(shape, dtype, scratch, name):
    """A slab, in RAM or as a `.npy` on disk.

    The pentad slab is 1.9 GB per copy and there are two of them; on a 7 GB box
    that plus pass 2's float64 working block is the difference between running
    and being killed. `scratch` puts them on disk instead, where the page cache
    can evict what is not being touched.
    """
    if not scratch:
        return np.empty(shape, dtype)
    os.makedirs(scratch, exist_ok=True)
    return np.lib.format.open_memmap(os.path.join(scratch, name + ".npy"),
                                     mode="w+", dtype=dtype, shape=shape)


def streaming_anomaly(blocks_fn, shape, dtype, moy, t_hold, x_hold,
                      want_lo, want_hi, chunk=8, verbose=True, scratch=None):
    """`ml/trainprobe.py::anomaly_transform`, over a stream instead of in place.

    Same three passes, same float64 accumulators, same Chan parallel variance
    combination, same float16 round-trip between pass 2 and pass 3 — the last
    of those matters, because the real function's mean and sd are taken from
    the values as STORED, not as computed, and a version that skipped the
    rounding would return slightly different numbers for the same tensor.

    The one difference is what is kept: `anomaly_transform` writes every bin
    back into X, this keeps only bins [want_lo, want_hi) and throws the rest
    away as it goes. `tests/test_export_cone_sample.py` asserts the two agree
    bit for bit on a toy tensor, which is the only reason this may be trusted.

    Returns (raw_slab, anom_slab, dynamic, mu, sd, ocean) with both slabs at
    the tensor's own storage dtype and `ocean` the [H, W] any-time finite mask
    of channel 0 (`ml/train_cone.py::load_data`'s own definition).
    """
    T, H, W, C = shape
    moy = np.asarray(moy)
    t_hold = np.asarray(t_hold, bool)
    x_hold = np.asarray(x_hold, bool)
    nwant = want_hi - want_lo

    def say(msg):
        if verbose:
            print(f"  streaming_anomaly: {msg}", flush=True)

    key = moy.astype(np.int64) * 2 + t_hold.astype(np.int64)
    edges = np.flatnonzero(np.diff(key)) + 1
    run_lo = np.concatenate(([0], edges)).astype(int)
    run_hi = np.concatenate((edges, [T])).astype(int)

    csum = np.zeros((12, H, W, C), np.float64)
    ccnt = np.zeros((12, H, W, C), np.int32)
    smean = np.empty((T, C), np.float64)
    ocean = np.zeros((H, W), bool)
    raw_slab = _alloc((nwant, H, W, C), dtype, scratch, "raw_slab")

    # ---- pass 1: spatial means, climatology sums, the ocean mask, the slab --
    with np.errstate(invalid="ignore", divide="ignore"):
        for i0, i1, blk in blocks_fn(chunk):
            fin = np.isfinite(blk)
            cnt = fin.sum(axis=(1, 2))
            tot = np.sum(blk, axis=(1, 2), where=fin, dtype=np.float64)
            smean[i0:i1] = tot / cnt
            ocean |= fin[..., 0].any(axis=0)
            for lo, hi in zip(run_lo, run_hi):
                a, z = max(lo, i0), min(hi, i1)
                if a >= z or t_hold[a]:
                    continue
                m = int(moy[a])
                sub = slice(a - i0, z - i0)
                csum[m] += np.sum(blk[sub], axis=0, where=fin[sub],
                                  dtype=np.float64)
                ccnt[m] += fin[sub].sum(axis=0, dtype=np.int32)
            a, z = max(want_lo, i0), min(want_hi, i1)
            if a < z:
                raw_slab[a - want_lo:z - want_lo] = blk[a - i0:z - i0]
            if i0 % (chunk * 40) == 0:
                say(f"pass 1/3 (climatology) {i1}/{T}")
        fin = blk = None

        dynamic = [c for c in range(C)
                   if np.nanstd(smean[:, c], dtype=np.float64) > 1e-6]
        if not dynamic:
            return raw_slab, raw_slab.copy(), dynamic, None, None, ocean
        clim = (csum / ccnt).astype(np.float32)
        del csum, ccnt
        stat = np.setdiff1d(np.arange(C), np.asarray(dynamic))
        clim[..., stat] = 0.0
        say(f"{len(dynamic)}/{C} dynamic channels; climatology done")

        # ---- pass 2: the anomaly, and its moments over the training pool ----
        wdt = np.promote_types(dtype, np.float32)
        n_t = np.zeros(C, np.float64)
        mu_t = np.zeros(C, np.float64)
        m2_t = np.zeros(C, np.float64)
        keep_x = ~x_hold
        anom_slab = _alloc((nwant, H, W, C), dtype, scratch, "anom_slab")
        for i0, i1, blk in blocks_fn(chunk):
            cm = clim[moy[i0:i1]]
            if cm.dtype == wdt:
                np.subtract(blk, cm, out=cm)
                anom = cm
            else:
                anom = np.subtract(blk, cm, dtype=wdt)
                del cm
            stored = anom.astype(dtype)          # what X would now hold
            del anom
            a, z = max(want_lo, i0), min(want_hi, i1)
            if a < z:
                anom_slab[a - want_lo:z - want_lo] = stored[a - i0:z - i0]
            b64 = np.asarray(stored, np.float64)
            del stored
            msk = np.isfinite(b64)
            msk &= (~t_hold[i0:i1])[:, None, None, None]
            msk &= keep_x[None, None, :, None]
            n_b = msk.sum(axis=(0, 1, 2)).astype(np.float64)
            s_b = np.sum(b64, axis=(0, 1, 2), where=msk, dtype=np.float64)
            nz = n_b > 0
            mu_b = np.where(nz, s_b / np.maximum(n_b, 1.0), 0.0)
            np.subtract(b64, mu_b, out=b64)
            np.square(b64, out=b64)
            m2_b = np.sum(b64, axis=(0, 1, 2), where=msk, dtype=np.float64)
            del b64, msk
            n_new = n_t + n_b
            delta = mu_b - mu_t
            safe = np.maximum(n_new, 1.0)
            mu_t = np.where(nz, mu_t + delta * (n_b / safe), mu_t)
            m2_t = np.where(nz, m2_t + m2_b + delta * delta * n_t * n_b / safe,
                            m2_t)
            n_t = n_new
            if i0 % (chunk * 40) == 0:
                say(f"pass 2/3 (anomaly + moments) {i1}/{T}")
        del clim

        sd = np.sqrt(m2_t / n_t)
        mu = mu_t
        mu[stat] = 0.0
        den = sd + 1e-6
        den[stat] = 1.0

        # ---- pass 3: z-score, on the kept bins only ------------------------
        # Chunked for the same reason pass 2 is: the float64 view of the whole
        # pentad slab would be 7.7 GB on a 7 GB box.
        for i0 in range(0, nwant, chunk):
            i1 = min(i0 + chunk, nwant)
            out = np.subtract(anom_slab[i0:i1], mu, dtype=np.float64)
            np.divide(out, den, out=out)
            anom_slab[i0:i1] = out
            del out
    say("done (3 passes; only the requested bins were kept)")
    return raw_slab, anom_slab, dynamic, mu, sd, ocean


# ------------------------------------------------------------ the sample dict --
def _grid_of(lat, lon, win):
    y = int(round((lat - win["lat0"]) / win["dlat"]))
    x = int(round((lon - win["lon0"]) / win["dlat"]))
    return y, x


def _cell_latlon(y, x, win):
    return (win["lat0"] + y * win["dlat"], win["lon0"] + x * win["dlat"])


def build_sample(*, raw, anom, obs_arr, lats, lons, chan_names, anchor,
                 bins_local, bins_abs, win, geo, L_in=6, future_lags=(1, 2),
                 outer_lags, outer_n_pts, scoreable, t_hold_full,
                 value_space, extra_meta, groups=None, statics=None):
    """Everything one anchor's file carries, as plain Python.

    `raw` / `anom` are [Tslab, H, W, C] arrays over the SAME local bin axis
    `bins_local` indexes — or, for a family-7 tensor, two
    `cone_sampler.GroupSet`s over the same axis, in which case `obs_arr` is
    None and each group carries its own mask. `bins_abs` are the corresponding
    bins of the real tensor and are what every date in the output is computed
    from.

    A dot's `row` / `col` are UNCLAMPED cell indices and may fall outside the
    tensor — that is what `valid = 0` means. Its `lat` / `lon` are then `null`,
    because those fields answer "where in the tensor is this", and off the
    tensor there is no answer. Where the dot sits on the EARTH is a different
    question and the page derives it from (row, col) itself, so it can draw an
    off-window dot hollow in the right place. ON A GLOBAL TENSOR the column is
    WRAPPED instead (`ConeSampler.wrap`), so a dot east of the dateline is a
    real cell with a real value and `valid = 1`; only latitude and the time
    axis can still put a dot outside.
    """
    y, x = _grid_of(anchor["lat"], anchor["lon"], win)
    alat, alon = _cell_latlon(y, x, win)

    # ---- the inner cone: ConeSampler, the trainer's own object --------------
    sam_raw = ConeSampler(raw, obs_arr, lats, lons, chan_names, L_in=L_in,
                          dlat_deg=win["dlat"], future_lags=future_lags)
    sam_anom = ConeSampler(anom, obs_arr, lats, lons, chan_names, L_in=L_in,
                           dlat_deg=win["dlat"], future_lags=future_lags)
    H, W, C, wrap = sam_raw.H, sam_raw.W, sam_raw.C, sam_raw.wrap
    Tslab = sam_raw.T
    anchors = np.array([[t, y, x] for t in bins_local], np.int64)
    S = sam_raw.sample(anchors)
    A = sam_anom.sample(anchors)

    # Admissibility, from the sampler's OWN span rule rather than a second copy
    # of it: an anchor is a training anchor only if every bin its cone touches
    # (L_in back, the future targets forward) is a training bin.
    span = sam_raw.bin_span()
    Tfull = len(t_hold_full)
    adm = []
    for b in bins_abs:
        touched = [int(b) - int(s) for s in span]
        adm.append(bool(all(0 <= t < Tfull and not t_hold_full[t]
                            for t in touched)))

    R = sam_raw.row(y)
    n = R["n"]
    dot_y = y + R["dy"]
    dot_x = x + R["dx"]
    if wrap:
        dot_x = np.mod(dot_x, W)
    on_grid = (dot_y >= 0) & (dot_y < H) & (dot_x >= 0) & (dot_x < W)
    dot_lat = win["lat0"] + dot_y * win["dlat"]
    dot_lon = win["lon0"] + dot_x * win["dlat"]

    inner = dict(
        n_dots=int(n),
        dims=["date", "dot"],
        shape=[len(bins_local), int(n)],
        lag=[int(v) for v in R["lag"]],
        dy=[int(v) for v in R["dy"]],
        dx=[int(v) for v in R["dx"]],
        chan=[int(v) for v in R["chan"]],
        row=[int(v) for v in dot_y],
        col=[int(v) for v in dot_x],
        lat=[sig(v, 6) if g else None for v, g in zip(dot_lat, on_grid)],
        lon=[sig(v, 6) if g else None for v, g in zip(dot_lon, on_grid)],
        dy_km=[sig(v) for v in R["dy_km"]],
        dx_km=[sig(v) for v in R["dx_km"]],
        lag_days=[int(v) for v in R["lag_days"]],
        raw=[[sig(v) for v in row] for row in S["vals"][:, :n]],
        anom=[[sig(v) for v in row] for row in A["vals"][:, :n]],
        obs=bits(S["obs"][:, :n]),
        valid=bits(S["valid"][:, :n]),
    )

    patch = dict(
        dims=["date", "chan", "cell"],
        shape=[len(bins_local), C, 9],
        cell_dy=[-1, -1, -1, 0, 0, 0, 1, 1, 1],
        cell_dx=[-1, 0, 1, -1, 0, 1, -1, 0, 1],
        raw=[sig(v) for v in S["patch_vals"].ravel()],
        anom=[sig(v) for v in A["patch_vals"].ravel()],
        obs=bits(S["patch_obs"]),
    )
    future = dict(
        dims=["date", "chan", "lead"],
        shape=[len(bins_local), C, len(future_lags)],
        lags=[int(f) for f in future_lags],
        raw=[sig(v) for v in S["fut_vals"].ravel()],
        anom=[sig(v) for v in A["fut_vals"].ravel()],
        obs=bits(S["fut_obs"]),
    )

    # ---- the outer cone: ml/cone.py::outer_spiral, per latitude row ---------
    ci = [chan_names.index(c) for c in scoreable]
    o_dy, o_dx, o_row, o_col, o_lat, o_lon = [], [], [], [], [], []
    o_dykm, o_dxkm, o_valid = [], [], []
    per_lag = None
    for k in outer_lags:
        sp = cone.outer_spiral(float(alat), int(k), dlat_deg=win["dlat"],
                               n_pts=outer_n_pts, L_in=L_in)
        if per_lag is None:
            per_lag = len(sp)
        elif len(sp) != per_lag:
            raise SystemExit(
                f"outer_spiral returned {len(sp)} dots at k={k} but "
                f"{per_lag} at the first lag — the export's flat shape "
                f"assumes one dot count per lag")
        for dy, dx in sp:
            yy, xx = y + dy, x + dx
            if wrap:
                xx %= W
            ok = 0 <= yy < H and 0 <= xx < W
            ykm, xkm = cone.ground_km(float(dy), float(dx), float(alat),
                                      win["dlat"])
            o_dy.append(int(dy))
            o_dx.append(int(dx))
            o_row.append(int(yy))
            o_col.append(int(xx))
            o_lat.append(sig(win["lat0"] + yy * win["dlat"], 6) if ok else None)
            o_lon.append(sig(win["lon0"] + xx * win["dlat"], 6) if ok else None)
            o_dykm.append(sig(ykm))
            o_dxkm.append(sig(xkm))
            o_valid.append(ok)

    # The values, read through the SAMPLER (`read_cells`) rather than indexed
    # out of an array here: on a family-7 tensor the array a channel lives in
    # depends on the channel, and a second copy of that routing in the
    # exporter is exactly what this file's opening rule forbids. One call per
    # date, shaped [lag, dot, chan].
    nK, nD, nC = len(outer_lags), per_lag, len(ci)
    ci_arr = np.array(ci, np.int64)
    o_row_a = np.array(o_row, np.int64).reshape(nK, nD)
    o_col_a = np.array(o_col, np.int64).reshape(nK, nD)
    o_val_a = np.array(o_valid, bool).reshape(nK, nD)
    lag_a = np.array([int(k) for k in outer_lags], np.int64)[:, None]
    o_raw = np.full((len(bins_local), nK, nD, nC), np.nan, np.float32)
    o_anom = np.full_like(o_raw, np.nan)
    o_obs = np.zeros(o_raw.shape, bool)
    for ti, tl in enumerate(bins_local):
        tb = tl - lag_a                                        # [nK, 1]
        here = o_val_a & (tb >= 0) & (tb < Tslab)              # [nK, nD]
        if not here.any():
            continue
        tc = np.clip(np.broadcast_to(tb, (nK, nD)), 0, Tslab - 1)
        vr, orr = sam_raw.read_cells(tc[..., None], o_row_a[..., None],
                                     o_col_a[..., None], ci_arr)
        va, _ = sam_anom.read_cells(tc[..., None], o_row_a[..., None],
                                    o_col_a[..., None], ci_arr)
        keep = here[..., None]
        o_raw[ti] = np.where(keep, vr, np.nan)
        o_anom[ti] = np.where(keep, va, np.nan)
        o_obs[ti] = orr & keep

    outer = dict(
        dims=["date", "lag", "dot", "chan"],
        shape=[len(bins_local), nK, nD, nC],
        lags=[int(k) for k in outer_lags],
        n_dots_per_lag=int(nD),
        channels=list(scoreable),
        chan_index=[int(c) for c in ci],
        dot_dims=["lag", "dot"],
        dy=o_dy, dx=o_dx, row=o_row, col=o_col, lat=o_lat, lon=o_lon,
        dy_km=o_dykm, dx_km=o_dxkm,
        valid=bits(np.array(o_valid)),
        raw=[sig(v) for v in o_raw.ravel()],
        anom=[sig(v) for v in o_anom.ravel()],
        obs=bits(o_obs),
    )

    meta = dict(
        anchor=dict(id=anchor["id"], name=anchor["name"],
                    lat_asked=anchor["lat"], lon_asked=anchor["lon"],
                    lat=sig(alat, 6), lon=sig(alon, 6), row=y, col=x,
                    note=anchor.get("note")),
        bins=[int(b) for b in bins_abs],
        dates=[date_of_bin(b) for b in bins_abs],
        admissible=adm,
        n_inadmissible=int(sum(1 for a in adm if not a)),
        channels=list(chan_names),
        channel_family={c: cone.channel_family(c) for c in chan_names},
        # WHICH ARRAY each channel came out of, and at what resolution. On a
        # single-array tensor every channel answers `null`; on family 7 this
        # is the difference between a value that was measured at this 0.25
        # degree cell and one that was measured at the 1 degree cell
        # containing it, which a reader of the page must be able to tell.
        groups=groups,
        channel_group=({c: g for g, names in (groups or {}).get(
            "channels", {}).items() for c in names} if groups else
            {c: None for c in chan_names}),
        grid=dict(ny=int(H), nx=int(W), lat0=float(win["lat0"]),
                  lon0=float(win["lon0"]), step=float(win["dlat"]),
                  wrap=bool(wrap)),
        sphere_at_anchor=(None if statics is None or "sphere" not in statics
                          else int(statics["sphere"][y, x])),
        elev_at_anchor=(None if statics is None or "elev" not in statics
                        else sig(float(statics["elev"][y, x]), 6)),
        recipe=(None if statics is None else statics.get("recipe")),
        depth_channels=[c for c in chan_names if cone.is_depth_channel(c)],
        units={c: UNITS.get(c, "dbar-level (°C / PSU)") for c in chan_names},
        scoreable_channels=list(scoreable),
        holdout_years=list(HOLDOUT_YEARS),
        terminal_train_last_year=TERMINAL_TRAIN_LAST_YEAR,
        value_space=value_space,
        L_in=int(L_in),
        future_lags=[int(f) for f in future_lags],
        geometry=dict(
            constants=geo["constants"], window=win,
            counts=geo["counts"], reach_km=geo["reach_km"],
        ),
        outer=dict(
            lags=[int(k) for k in outer_lags],
            stride=int(outer_lags[1] - outer_lags[0]) if len(outer_lags) > 1
                   else 1,
            first=int(outer_lags[0]), last=int(outer_lags[-1]),
            empty_below=L_in + 1,
            empty_note=(
                f"stage 2's annulus is EMPTY for lags 0-{L_in}: r_lo(k) = "
                f"r_in(k) is the same formula as r_hi(k) there, so the codec "
                f"has already read that whole disc and stage 2 keeps only the "
                f"anchor column. The first non-empty spiral is k = {L_in + 1}."),
            n_pts=int(outer_n_pts),
        ),
    )
    meta.update(extra_meta)
    return dict(meta=meta, inner=inner, patch=patch, future=future,
                outer=outer)


def dumps(obj):
    """One spelling of a sample file, so a regeneration is a no-op diff."""
    return json.dumps(obj, sort_keys=True, separators=(",", ":"),
                      allow_nan=False) + "\n"


def trim_sample(s, n_dates=3, want_lags=(7, 36, 143)):
    """A small, in-repo copy of one anchor: the first `n_dates` pentads, the
    whole inner cone, and only the outer lags nearest `want_lags`.

    The browser tests must run under MIRROR with no network, and the real files
    are megabytes on a foreign host. This keeps the SCHEMA exactly — same keys,
    same dims, same flag strings — so a test that passes on the fixture is a
    test about the page, not about a second format.
    """
    s = json.loads(json.dumps(s))                 # never mutate the caller's
    nT0 = len(s["meta"]["bins"])
    nT = min(n_dates, nT0)
    nC = len(s["meta"]["channels"])
    o = s["outer"]
    lags = o["lags"]
    keep = sorted({min(range(len(lags)), key=lambda i: abs(lags[i] - w))
                   for w in want_lags})
    nD, nCo = o["n_dots_per_lag"], len(o["channels"])
    nK0 = len(lags)

    s["meta"]["bins"] = s["meta"]["bins"][:nT]
    s["meta"]["dates"] = s["meta"]["dates"][:nT]
    s["meta"]["admissible"] = s["meta"]["admissible"][:nT]
    s["meta"]["fixture"] = (
        f"a trimmed copy for the browser tests: {nT} of {nT0} dates and "
        f"{len(keep)} of {nK0} outer lags. The schema is the full file's.")
    s["meta"]["outer"]["lags"] = [lags[i] for i in keep]
    s["meta"]["outer"]["stride"] = None

    inn, nDots = s["inner"], s["inner"]["n_dots"]
    inn["shape"] = [nT, nDots]
    inn["raw"] = inn["raw"][:nT]
    inn["anom"] = inn["anom"][:nT]
    inn["obs"] = inn["obs"][:nT * nDots]
    inn["valid"] = inn["valid"][:nT * nDots]

    for blk, per in ((s["patch"], 9), (s["future"], len(s["future"]["lags"]))):
        blk["shape"][0] = nT
        blk["raw"] = blk["raw"][:nT * nC * per]
        blk["anom"] = blk["anom"][:nT * nC * per]
        blk["obs"] = blk["obs"][:nT * nC * per]

    o["lags"] = [lags[i] for i in keep]
    o["shape"] = [nT, len(keep), nD, nCo]
    for k in ("dy", "dx", "row", "col", "lat", "lon", "dy_km", "dx_km"):
        o[k] = [o[k][i * nD + d] for i in keep for d in range(nD)]
    o["valid"] = "".join(o["valid"][i * nD:(i + 1) * nD] for i in keep)
    for k in ("raw", "anom"):
        o[k] = [o[k][((t * nK0 + i) * nD + d) * nCo + c]
                for t in range(nT) for i in keep for d in range(nD)
                for c in range(nCo)]
    o["obs"] = "".join(o["obs"][((t * nK0 + i) * nD + d) * nCo
                                : ((t * nK0 + i) * nD + d + 1) * nCo]
                       for t in range(nT) for i in keep for d in range(nD))
    return s


# ------------------------------------------------------------------- drivers --
def _git_commit():
    try:
        return subprocess.run(["git", "-C", ROOT, "rev-parse", "HEAD"],
                              capture_output=True, text=True,
                              timeout=20).stdout.strip() or None
    except Exception:
        return None


def _sha256(path, buf=1 << 22):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for blk in iter(lambda: fh.read(buf), b""):
            h.update(blk)
    return h.hexdigest()


def smoke_tensor(seed=0, T=200, H=40, W=48, C=None, chans=None):
    """A synthetic tensor with the same recipe as `tests/test_cone_smoke.py`'s
    `tiny_sampler` — random normal with 5 % of cells NaN — but long enough in
    time to carry a 143-pentad outer cone."""
    chans = list(chans or ("cur_speed", "log_mld", "ssh", "tau_x", "tau_y",
                           "tau_x_std", "tau_y_std", "sst", "cur_u", "cur_v"))
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(T, H, W, len(chans))).astype(np.float32)
    X[rng.random(X.shape) < 0.05] = np.nan
    return X, chans


def _anomaly_transform():
    """`ml/trainprobe.py::anomaly_transform`, THE one transform, whether or not
    torch is importable.

    On the box the plain import works. In the sandbox (and in the test suite)
    `ml/trainprobe.py` pulls torch at module scope, so the function's own
    source is lifted out by `ast` and exec'd — `ml/cone.py`'s trick for the
    spiral, for its reason: the function under use must be THE function, byte
    for byte, and a second copy is what
    `tests/test_one_anomaly_transform.py` exists to forbid.
    """
    try:                                              # pragma: no cover - box
        from trainprobe import anomaly_transform
        return anomaly_transform
    except Exception:
        import ast
        import warnings
        src = open(os.path.join(HERE, "trainprobe.py"),
                   encoding="utf-8").read()
        keep = [n for n in ast.parse(src).body
                if isinstance(n, ast.FunctionDef)
                and n.name == "anomaly_transform"]
        if not keep:
            raise ImportError(
                "ml/trainprobe.py no longer defines anomaly_transform at "
                "module scope — ml/export_cone_sample.py lifts it from source "
                "so the exported anomaly cannot drift from the trainer's. Fix "
                "the lift, never copy the function.")
        ns = {"np": np, "warnings": warnings}
        exec(compile(ast.Module(body=keep, type_ignores=[]), "trainprobe.py",
                     "exec"), ns)
        return ns["anomaly_transform"]


def _open_family7(a, win):
    """Open a family-7 tensor and put both value spaces on the same axis.

    Returns everything `build_sample` needs, in the same shape the streaming
    family-4 path returns it: (raw, anom, obs, chan, lats, lons, win,
    bins_abs, bins_local, t_hold, value_space, tensor_meta, groups_meta,
    statics).

    Two things differ from the family-4 path and both are consequences of the
    layout, not choices.

    NO STREAMING. The sidecar layout (`ml/tensor_io.py`) exists exactly so a
    46 GB group can be MEMORY-MAPPED instead of decompressed, so `load_tensor`
    is the whole reader and `_NpzMemberStream` is not used at all. There is
    also no slab: `bins_local` ARE master rows, and the samplers index the
    full time axis lazily.

    THE ANOMALY IS THE REAL FUNCTION, on a writable copy per group.
    `streaming_anomaly` was written for a box with 30 GB of disk and one
    array; this runs beside the tensor on the 300 GB box, where the honest
    thing is to call `anomaly_transform` itself — no second implementation to
    keep in step, and `tests/test_export_cone_sample.py`'s bit-for-bit
    comparison of the streaming copy stays exactly as it is for the family-4
    path. Each group gets its own months (`cone_sampler.group_time`), because
    `rg100`'s rows are one per month and charging them to the master's
    calendar would put every profile in the wrong climatology.
    """
    from tensor_io import (anomaly_chunk, anomaly_peak_bytes, load_tensor,
                           writable_copy)
    from cone_sampler import GroupSet, group_time

    d = load_tensor(a.tensor, allow_pickle=False)
    gnames = [str(g) for g in getattr(d, "groups", []) if str(g) != "X"]
    if not gnames:
        raise SystemExit(
            f"{a.tensor}: no `groups` — this is a single-array tensor. Use "
            f"--stream-npz for those; --tensor is the multi-group reader.")
    raw = GroupSet.from_tensor(d)
    m = raw.master
    lats, lons = np.asarray(d["lats"]), np.asarray(d["lons"])
    step = float(lats[1] - lats[0])
    win = dict(win, lat0=float(lats[0]), lon0=float(lons[0]), dlat=step,
               ny=int(m.H), nx=int(m.W), lat1=float(lats[-1]),
               lon1=float(lons[-1]))
    chan = list(raw.chan)
    months = [str(x) for x in d["months"]]
    bin_index = (np.asarray(d["bin_index"], np.int64) if "bin_index" in d
                 else np.arange(m.T, np.int64))
    print(f"family 7: {raw.names} · master [T={m.T} H={m.H} W={m.W}] · "
          f"{len(chan)} channels · lon axis {lons[0]:g}..{lons[-1]:g} "
          f"(wrap {'yes' if abs(m.W * step - 360) < 1e-6 else 'no'})",
          flush=True)

    # Which rows to export. The npz's `bin_index` is ABSOLUTE (bins counted
    # from 1982-01-01) and the rows are not, so the start date is resolved
    # against the axis rather than assumed to be row = bin.
    b0 = first_bin_on_or_after(a.start)
    r0 = int(np.searchsorted(bin_index, b0))
    if r0 >= m.T or int(bin_index[r0]) != b0:
        raise SystemExit(
            f"--start {a.start} is pentad bin {b0}, which is not on this "
            f"tensor's axis (bins {int(bin_index[0])}..{int(bin_index[-1])})")
    bins_local = list(range(r0, min(r0 + a.n_dates, m.T)))
    if len(bins_local) < a.n_dates:
        raise SystemExit(
            f"--n-dates {a.n_dates} from {a.start} runs past the end of the "
            f"tensor ({m.T} bins)")
    bins_abs = [int(bin_index[r]) for r in bins_local]

    moy = np.array([int(x[5:7]) - 1 for x in months])
    t_hold = np.array([int(x[:4]) in HOLDOUT_YEARS for x in months])
    print(f"held-out bins {int(t_hold.sum())}/{m.T} · exporting rows "
          f"{bins_local[0]}..{bins_local[-1]} "
          f"({date_of_bin(bins_abs[0])} .. {date_of_bin(bins_abs[-1])})",
          flush=True)

    if a.anomaly == "trainer":
        transform = _anomaly_transform()
        scratch_dir = a.scratch or os.path.dirname(os.path.abspath(a.tensor))
        os.makedirs(scratch_dir, exist_ok=True)
        arrays, dynamic = {}, {}
        for g in raw.groups:
            dst = os.path.join(
                scratch_dir,
                os.path.basename(a.tensor)[:-4] + f"_cone_anom_{g.name}.npy")
            print(f"  {g.name}: writable copy -> {dst}", flush=True)
            A = writable_copy(g.X, dst)
            gmoy, ghold = group_time(g, moy, t_hold)
            ch = anomaly_chunk(A.shape, np.dtype(A.dtype).itemsize)
            peak = anomaly_peak_bytes(A.shape, ch,
                                      np.dtype(A.dtype).itemsize) / 1e9
            print(f"  {g.name}: anomaly_transform chunk {ch} "
                  f"(peak ~{peak:.1f} GB)", flush=True)
            A, dyn = transform(A, gmoy, ghold, np.zeros(g.W, bool), chunk=ch)
            arrays[g.name] = A
            dynamic[g.name] = [int(c) for c in dyn]
        anom = GroupSet.from_tensor(d, arrays=arrays)
        value_space = dict(
            raw=RAW_NOTE, physical=PHYSICAL_NOTE, anomaly="trainer",
            anomaly_note=(
                "computed the way ml/train_cone.py does and BY THE SAME "
                "FUNCTION (ml/trainprobe.py::anomaly_transform, called on a "
                "writable copy of each group): departure from a "
                "per-calendar-month climatology built on TRAINING YEARS ONLY, "
                "then z-scored per channel over the training pool. Each group "
                "is transformed with ITS OWN months — rg100 holds one row per "
                "month, so the master's calendar would be the wrong one."),
            dynamic_channels_by_group=dynamic,
            tensor_norm={g: np.asarray(d[f"norm_{g}"]).tolist()
                         for g in raw.names if f"norm_{g}" in d})
    else:
        anom = raw
        value_space = dict(
            raw=RAW_NOTE, physical=PHYSICAL_NOTE,
            anomaly="none — `anom` repeats `raw`; the page can standardise "
                    "with tensor_norm (mean, std) per channel",
            tensor_norm={g: np.asarray(d[f"norm_{g}"]).tolist()
                         for g in raw.names if f"norm_{g}" in d})

    groups_meta = dict(
        names=list(raw.names),
        channels={g.name: list(g.chan) for g in raw.groups},
        shape={g.name: [int(v) for v in g.X.shape] for g in raw.groups},
        step={g.name: float(g.lats[1] - g.lats[0]) for g in raw.groups},
        coarse_factor={g.name: int(g.factor) for g in raw.groups},
        live_bins={g.name: (None if g.bin_index is None
                            else int((np.asarray(g.bin_index) >= 0).sum()))
                   for g in raw.groups},
        lookup=("a channel in a coarse group is read at "
                "y1 = min(floor(y/f + 0.5), ny_g - 1), "
                "x1 = floor(x/f + 0.5) mod nx_g, f the ratio of the two "
                "latitude steps (E-070 §1); the lag-0 3x3 therefore serves "
                "the same coarse cell nine times (E-070 D3). A channel in a "
                "live-bins group is a MISS token at every bin the group did "
                "not write."))
    statics = {}
    for k in ("sphere", "elev"):
        if k in d:
            statics[k] = np.asarray(d[k])
    statics["recipe"] = str(d["recipe"]) if "recipe" in d else None

    tensor_meta = dict(
        name=os.path.basename(a.tensor).replace(".npz", ""),
        sha256=_sha256(a.tensor),
        groups={g.name: [int(v) for v in g.X.shape] for g in raw.groups},
        shape=[int(v) for v in raw.shape],
        dtype=str(np.dtype(m.X.dtype)),
        epoch=str(PENTAD_EPOCH), pentad_days=int(PENTAD_DAYS),
        months_first=months[0], months_last=months[-1],
        recipe=statics.get("recipe"),
        window=(str(d["window"]) if "window" in d else None))
    return (raw, anom, None, chan, lats, lons, win, bins_abs, bins_local,
            t_hold, value_space, tensor_meta, groups_meta, statics)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--stream-npz", help="the production tensor (.npz); its "
                                         "X member is streamed, never loaded")
    ap.add_argument("--tensor", help="a MULTI-GROUP tensor stem (family 7's "
                                     "<stem>.npz beside its _X_<group>.npy "
                                     "sidecars). Read with "
                                     "tensor_io.load_tensor as memmaps — no "
                                     "npz streaming is needed, because the "
                                     "sidecar layout exists so a 46 GB group "
                                     "can be mapped rather than decompressed")
    ap.add_argument("--smoke", action="store_true",
                    help="run on a synthetic tensor instead")
    ap.add_argument("--out", default=os.path.join(ROOT, "data_out",
                                                  "cone_samples"))
    ap.add_argument("--start", default="2015-01-01")
    ap.add_argument("--n-dates", type=int, default=24)
    ap.add_argument("--outer-stride", type=int, default=2,
                    help="keep every Nth outer lag (1 = all 137)")
    ap.add_argument("--anomaly", default="trainer", choices=("trainer", "none"))
    ap.add_argument("--chunk", type=int, default=8,
                    help="timesteps per streaming block (memory, not answer)")
    ap.add_argument("--scratch", default="",
                    help="directory for the two slab .npy files; without it "
                         "they are held in RAM (1.9 GB each at pentad shape)")
    ap.add_argument("--anchors", default="",
                    help="comma-separated anchor ids; default all")
    ap.add_argument("--fixture", default="",
                    help="also write a trimmed in-repo copy here "
                         "(data/cone_samples/fixture.json)")
    ap.add_argument("--fixture-anchor", default="gulf_stream")
    a = ap.parse_args(argv)

    geo = json.load(open(GEOMETRY, encoding="utf-8"))
    win = geo["window"]
    L_in = geo["constants"]["L_IN"]
    K_outer = geo["constants"]["K_OUTER"]
    outer_n_pts = geo["constants"]["OUTER_N_PTS"]
    outer_lags = list(range(L_in + 1, K_outer, max(1, a.outer_stride)))

    pool = GLOBAL_ANCHORS if a.tensor else ANCHORS
    anchors = pool
    if a.anchors:
        want = set(a.anchors.split(","))
        anchors = [x for x in pool if x["id"] in want]
        if not anchors:
            raise SystemExit(f"--anchors {a.anchors!r} matched none of "
                             f"{[x['id'] for x in pool]}")

    os.makedirs(a.out, exist_ok=True)
    groups_meta = statics = None

    if a.tensor:
        (raw, anom, obs, chan, lats, lons, win, bins_abs, bins_local, t_hold,
         value_space, tensor_meta, groups_meta, statics) = \
            _open_family7(a, win)
    elif a.smoke:
        X, chan = smoke_tensor()
        T, H, W, C = X.shape
        win = dict(win, lat0=30.0, lon0=-40.0, ny=H, nx=W,
                   lat1=30.0 + win["dlat"] * (H - 1),
                   lon1=-40.0 + win["dlat"] * (W - 1))
        lats = win["lat0"] + win["dlat"] * np.arange(H)
        lons = win["lon0"] + win["dlat"] * np.arange(W)
        anchors = SMOKE_ANCHORS
        b0 = 150
        bins_abs = list(range(b0, b0 + a.n_dates))
        months = [str(np.datetime64(date_of_bin(t), "M")) for t in range(T)]
        t_hold = np.array([m[:4] in {"1982"} for m in months])
        raw = X
        anom = X
        obs = np.isfinite(X)
        bins_local = bins_abs
        value_space = dict(anomaly="none (smoke)", raw="synthetic")
        tensor_meta = dict(name="synthetic", sha256=None,
                           shape=[int(v) for v in X.shape])
    else:
        if not a.stream_npz:
            raise SystemExit("--stream-npz PATH (family 2-6) or --tensor PATH "
                             "(a multi-group family-7 stem) is required "
                             "without --smoke")
        stream = _NpzMemberStream(a.stream_npz)
        meta_npz = np.load(a.stream_npz, allow_pickle=False)
        chan = [str(c) for c in meta_npz["chan"]]
        lats = np.asarray(meta_npz["lats"], np.float64)
        lons = np.asarray(meta_npz["lons"], np.float64)
        months = [str(m) for m in meta_npz["months"]]
        norm = np.asarray(meta_npz["norm"]).tolist()
        fh, shape, dtype = stream.open()
        fh.close()
        T, H, W, C = shape
        print(f"X [T={T} H={H} W={W} C={C}] {np.dtype(dtype)} · "
              f"{stream.size / 1e9:.1f} GB uncompressed", flush=True)
        if (H, W) != (win["ny"], win["nx"]):
            raise SystemExit(f"tensor is {H}x{W}, cone_geometry.json's window "
                             f"is {win['ny']}x{win['nx']}")

        b0 = first_bin_on_or_after(a.start)
        bins_abs = list(range(b0, b0 + a.n_dates))
        need_lo = min(bins_abs) - (K_outer - 1)
        need_hi = max(bins_abs) + max(1, 2) + 1
        if need_lo < 0 or need_hi > T:
            raise SystemExit(
                f"the cone at {a.start} reaches bins {need_lo}..{need_hi - 1}, "
                f"outside the tensor's 0..{T - 1}")
        moy = np.array([int(m[5:7]) - 1 for m in months])
        t_hold = np.array([int(m[:4]) in HOLDOUT_YEARS for m in months])
        print(f"held-out bins {int(t_hold.sum())}/{T} · keeping bins "
              f"{need_lo}..{need_hi - 1} ({need_hi - need_lo} of {T})",
              flush=True)

        if a.anomaly == "trainer":
            raw, anom, dynamic, mu, sd, ocean = streaming_anomaly(
                stream.blocks, shape, dtype, moy, t_hold,
                np.zeros(W, bool), need_lo, need_hi, chunk=a.chunk,
                scratch=a.scratch or None)
            value_space = dict(
                raw=RAW_NOTE, physical=PHYSICAL_NOTE,
                anomaly="trainer",
                anomaly_note=(
                    "computed the way ml/train_cone.py does: departure from a "
                    "per-calendar-month climatology built on TRAINING YEARS "
                    "ONLY, then z-scored per channel over the training pool "
                    "(ml/trainprobe.py::anomaly_transform). Streamed rather "
                    "than written in place; "
                    "tests/test_export_cone_sample.py asserts the two agree "
                    "bit for bit on a toy tensor."),
                dynamic_channels=[int(c) for c in dynamic],
                anomaly_mean=[sig(v, 6) for v in mu],
                anomaly_sd=[sig(v, 6) for v in sd],
                tensor_norm=[[sig(v, 6) for v in row] for row in norm],
            )
        else:
            raw = _alloc((need_hi - need_lo, H, W, C), dtype,
                         a.scratch or None, "raw_slab")
            for i0, i1, blk in stream.blocks(a.chunk):
                lo, hi = max(need_lo, i0), min(need_hi, i1)
                if lo < hi:
                    raw[lo - need_lo:hi - need_lo] = blk[lo - i0:hi - i0]
                if i1 >= need_hi:
                    break
            anom = raw
            ocean = None
            value_space = dict(
                raw=RAW_NOTE, physical=PHYSICAL_NOTE,
                anomaly="none — `anom` repeats `raw`; the page can standardise "
                        "with tensor_norm (mean, std) per channel",
                tensor_norm=[[sig(v, 6) for v in row] for row in norm])
        obs = np.isfinite(raw)
        bins_local = [b - need_lo for b in bins_abs]
        tensor_meta = dict(
            name=os.path.basename(a.stream_npz).replace(".npz", ""),
            sha256=_sha256(a.stream_npz),
            shape=[int(v) for v in shape],
            dtype=str(np.dtype(dtype)),
            epoch=str(PENTAD_EPOCH), pentad_days=int(PENTAD_DAYS),
            months_first=months[0], months_last=months[-1])
        if ocean is not None:
            tensor_meta["ocean_cells"] = int(ocean.sum())

    extra = dict(
        produced_by=("ml/cone_sampler.py::ConeSampler.sample (inner cone, "
                     "patch, future targets, valid/obs) and "
                     "ml/cone.py::outer_spiral (stage-2 stencil)"),
        exporter="ml/export_cone_sample.py",
        exporter_commit=_git_commit(),
        generated_utc=_dt.datetime.now(_dt.timezone.utc)
                        .strftime("%Y-%m-%dT%H:%M:%SZ"),
        tensor=tensor_meta,
        pentad_note=("A pentad is a fixed five-day bin counted from "
                     "1982-01-01: bin = floor(days since the epoch / 5). Every "
                     "date below is the day its bin OPENS on."),
    )

    written = []
    for anc in anchors:
        s = build_sample(
            raw=raw, anom=anom, obs_arr=obs, lats=lats, lons=lons,
            chan_names=chan, anchor=anc, bins_local=bins_local,
            bins_abs=bins_abs, win=win, geo=geo, L_in=L_in,
            outer_lags=outer_lags, outer_n_pts=outer_n_pts,
            scoreable=[c for c in (SCOREABLE_F7 if groups_meta else SCOREABLE)
                       if c in chan],
            t_hold_full=t_hold, value_space=value_space, extra_meta=extra,
            groups=groups_meta, statics=statics)
        path = os.path.join(a.out, f"{anc['id']}.json")
        text = dumps(s)
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(text)
        n_bad = s["meta"]["n_inadmissible"]
        obs_frac = s["inner"]["obs"].count("1") / max(len(s["inner"]["obs"]), 1)
        print(f"  {anc['id']:14s} {len(text) / 1e6:5.2f} MB · "
              f"row {s['meta']['anchor']['row']} col "
              f"{s['meta']['anchor']['col']} · "
              f"{obs_frac * 100:4.1f}% of inner dots observed · "
              f"{n_bad} inadmissible date(s)", flush=True)
        written.append((anc, path, text))

    if a.fixture and written:
        pick = next((w for w in written if w[0]["id"] == a.fixture_anchor),
                    written[0])
        fx = trim_sample(json.loads(pick[2]))
        os.makedirs(os.path.dirname(os.path.abspath(a.fixture)), exist_ok=True)
        text = dumps(fx)
        with open(a.fixture, "w", encoding="utf-8") as fh:
            fh.write(text)
        print(f"  fixture       {len(text) / 1e3:5.1f} kB · {pick[0]['id']} · "
              f"{len(fx['meta']['dates'])} dates · outer lags "
              f"{fx['outer']['lags']} -> {a.fixture}")

    print(f"wrote {len(written)} anchor file(s) to {a.out}")
    return written


if __name__ == "__main__":
    main()

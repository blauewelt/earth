#!/usr/bin/env python3
"""Family-7 tensor: the WHOLE GLOBE at 0.25 degrees, pentad cadence — recipe `f7l0`.

E-070 §Phases B-D and E-071 §6.5 (Phase L0), built once as one tensor. The
executable specification is `ml/plans/E070_family7_build.md`; this file is that
document made runnable, and every decision below is the plan's, not a new one.

PLAIN ENGLISH. Family 7 is the first input tensor covering the whole planet
rather than the North Atlantic window: every 0.25-degree grid point from the
South Pole to the North Pole, one value per channel per five-day bin from 1982
to 2024. It carries the ocean channels family 4 already had (surface currents,
mixed-layer depth, sea-surface height, sea-surface temperature, the Argo depth
column, wind stress) and, new, a set of SHARED channels that exist over both
land and water — surface temperature, 2 m air temperature, 10 m wind, surface
pressure, precipitation, snow, soil moisture and temperature, turbulent heat
fluxes, sea ice — so a land cell is no longer told "the world was not observed
here" in every channel.

THREE GROUPS AT THEIR NATIVE RESOLUTION, not one dense array (plan B2):

  g025   [3142, 721, 1440,  7] float16  46 GB   0.25 deg, GLORYS12 + OISST
  g100   [3142, 181,  360, 15] float16  6.1 GB  1 deg, NCEP/NCAR R1
  rg100  [n_live, 181, 360, 32] float16 ~1 GB   1 deg, Roemmich-Gilson, live bins

The two COARSE groups are filled at float32 and converted to the published
float16 by the `norm` stage (see RAW_F32) — 13.5 GB of transient disk that
buys back the precision a float16 raw write costs a channel with a large
offset, `sp` near 1000 hPa worst of all. g025 stays float16 in place, because
its float32 intermediate would be 92 GB.

A 1.9-degree reanalysis upsampled to 0.25 degrees is sixty copies of every
number; at 65 channels dense the tensor would be 425 GB. The cone reads dots at
0.25-degree positions and looks each coarse channel up at the nearest coarse
cell — `y1 = round(y/4)`, `x1 = round(x/4) mod 360`.

THE AXIS IS SHARED BY CONSTRUCTION with families 4 and 5 and with
`ml/build_truth_pentad.py`: fixed 5-day bins counted from 1982-01-01, index =
floor(days_since_epoch / 5), imported from `ml/aggregate_cadence.py` rather
than restated. That is what makes a pentad label the target of a pentad state
with no re-alignment step anywhere.

THE GRID IS POINT-ALIGNED AND SOUTH-FIRST, like every family here:
lats = -90 + 0.25*arange(721), lons = -180 + 0.25*arange(1440). The GLORYS12
chunks run -80..90 (681 rows) and therefore land at rows 40..720; rows 0..39 of
every ocean channel are NaN because there is no ocean there, and `sphere` says
so. That row offset is ASSERTED from the chunk's own `latitude[0]`, never
assumed (plan §5.3).

RESUMABLE STAGE BY STAGE (plan §4). Every stage writes its rows to the memmap,
FLUSHES, then writes `<work>/<stage>.done` — a marker may only under-claim
(ml/CLAUDE.md §5.21). `glorys`, `sst`, `ncep` and `rg` additionally keep
per-chunk / per-year / per-cube markers plus an atomically-written CARRY file,
because a pentad bin can straddle a month or a year boundary and a resume that
lost the partial accumulator would silently write a thinner mean. Sources are
STREAMED: one GLORYS chunk (~260 MB) or one OISST year (1.6 + 0.6 GB) on disk
at a time, deleted after use.

Run:
  python3 ml/build_family7.py --smoke                       # synthetic, seconds
  python3 ml/build_family7.py --work /opt/earth-cache/f7 --stage all
  python3 ml/build_family7.py --work ... --stage ncep       # one stage
  python3 ml/build_family7.py --work ... --source-dir DIR   # no network at all
"""
import argparse
import datetime as dt
import glob
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import build_family3 as f3                                    # noqa: E402
import build_family4 as f4                                    # noqa: E402
from aggregate_cadence import EPOCH, bin_index, bin_start      # noqa: E402

CACHE = os.path.join(HERE, "cache")
RECIPE = "f7l0"
STEM = "family7_global025_pentad_l0"
HF_DATASET = "earth-tensors"
HF_PREFIX = f"tensors/{STEM}"

PENTAD_DAYS = 5
MIN_DAYS = 3                      # family 4's rule, at every cadence
START = dt.date(1982, 1, 1)
END = dt.date(2024, 12, 31)

# ---- the grids (plan §1) --------------------------------------------------
NLAT, NLON = 721, 1440
NLAT1, NLON1 = 181, 360
GLORYS_ROWS = 681                 # -80.0 .. 90.0 at 0.25
GLORYS_LAT0 = -80.0
GLORYS_ROW0 = 40                  # (-80 - -90) / 0.25 — ASSERTED at read time


def grid025():
    return (-90.0 + 0.25 * np.arange(NLAT),
            -180.0 + 0.25 * np.arange(NLON))


def grid100():
    return (np.arange(-90.0, 91.0), np.arange(-180.0, 180.0))


def coarse_lookup(y, x):
    """The 0.25-degree index (y, x) -> its 1-degree cell (plan §1).

    Every fourth 0.25-degree point IS a 1-degree point and the two points on
    either side round to it; x = 1438, 1439 round to 360 and wrap to lon1 =
    -180, which is the same meridian as +180.

    ROUNDING IS HALF-UP, not numpy's half-to-even, and the plan pins the case
    that separates them: y = 2 must map to 1 (half-to-even would give 0) while
    y = 1 maps to 0. `floor(v + 0.5)` is that rule, written once here so a
    consumer cannot pick the other one.
    """
    y1 = np.minimum(np.floor(np.asarray(y) / 4 + 0.5).astype(np.int64),
                    NLAT1 - 1)
    x1 = np.floor(np.asarray(x) / 4 + 0.5).astype(np.int64) % NLON1
    return y1, x1


# ---- the channels (plan §2) ----------------------------------------------
# E-071 §6.1, "Correction, 4 Sep": A CHANNEL IS SHARED ONLY WHEN THE MEASURAND
# AND THE INSTRUMENT MATCH ON BOTH SIDES. The first version of this build
# merged OISST sea-surface temperature and NCEP skin temperature into one
# `skin_t` channel — one measurand (the temperature of the surface) but TWO
# instruments, an infrared/microwave analysis over the sea and a reanalysis
# everywhere else, spliced at a coastline the model would have had to learn
# was an instrument boundary rather than a physical one. So they are two
# channels now: `sst` is the OBSERVED field where OISST observes and missing
# elsewhere, and the SHARED surface temperature is the reanalysis field over
# every surface — `skt`, in g100, at the reanalysis's own resolution, which is
# also where ERA5 will drop in at Phase L1 with no layout change.
CHAN_G025 = ["cur_speed", "log_mld", "ssh", "cur_u", "cur_v",
             "sst", "sea_ice"]
CHAN_G100 = ["tau_x", "tau_y", "tau_x_std", "tau_y_std", "t2m", "u10", "v10",
             "sp", "log_prate", "log_swe", "soilw", "tsoil", "lhtfl", "shtfl",
             "skt"]
LEVELS = f3.LEVELS                                   # ONE definition, imported
CHAN_RG100 = ([f"rg_t{int(p)}" for p in LEVELS]
              + [f"rg_s{int(p)}" for p in LEVELS])
GROUPS = ["g025", "g100", "rg100"]
NCHAN = {"g025": len(CHAN_G025), "g100": len(CHAN_G100),
         "rg100": len(CHAN_RG100)}

C_CUR_SPEED, C_LOG_MLD, C_SSH, C_CUR_U, C_CUR_V, C_SST, C_SEA_ICE = range(7)
C_SKT = 14                            # g100's shared surface temperature

# ---- the sources ----------------------------------------------------------
PSL_NCEP = ("https://downloads.psl.noaa.gov/Datasets/ncep.reanalysis/"
            "surface_gauss")
THREDDS_NCEP = ("https://psl.noaa.gov/thredds/fileServer/Datasets/"
                "ncep.reanalysis/surface_gauss")
PSL_OISST = "https://downloads.psl.noaa.gov/Datasets/noaa.oisst.v2.highres"
THREDDS_OISST = ("https://psl.noaa.gov/thredds/fileServer/Datasets/"
                 "noaa.oisst.v2.highres")
RG_BASE = "https://sio-argo.ucsd.edu/pub/www-argo/RG"
NE_BASE = ("https://raw.githubusercontent.com/nvkelso/natural-earth-vector/"
           "master/geojson/")
# ETOPO 2022, 60 arc-second ICE-SURFACE elevation. VERIFIED 2026-09-04:
# HEAD 200, and a ranged GET returns an HDF5/netCDF-4 signature. Its own DDS
# reports z[lat=10800][lon=21600] and the DAS `GeoTransform` "-180 1/60 0 90 0
# -1/60" with node_offset 1 — i.e. CELL-registered, latitude descending.
ETOPO_URL = ("https://www.ngdc.noaa.gov/thredds/fileServer/global/ETOPO2022/"
             "60s/60s_surface_elev_netcdf/ETOPO_2022_v1_60s_N90W180_surface.nc")

# NCEP gaussian file stems and the variable each one carries. The `level`
# dimension of the two soil files is squeezed at read time (plan §2).
# MEASURED 2026-09-04, and it is not what the directory name suggests:
# `ncep.reanalysis/surface_gauss/<var>.<year>.nc` is the **4x DAILY** product —
# `weasd.sfc.gauss.2020.nc` has 1464 time steps (366 x 4), `delta_t
# 0000-00-00 06:00:00`, title "4x daily NMC reanalysis". Family 4's
# `fill_wind_pentad` reads these same files and treats every step as a sample,
# so its pentad mean is a mean of ~20 six-hourly values and its `tau_*_std` is
# the sigma of those — which is why family 7 keeps the same directory rather
# than switching source under a channel the G1 gate compares.
# The ONE thing that has to change is the guard: `min_days >= 3` means three
# DAYS, and three 6-hourly samples is eighteen hours. So the accumulator
# tracks the DISTINCT DAYS contributing to each bin, not the sample count.
# (`ncep.reanalysis.dailyavgs/surface_gauss/<var>.<year>.nc` is the true daily
# mean and was verified to exist — HTTP 200, 10.6 MB for skt 2020 — if a later
# revision wants it; it is a channel-definition change, not a bug fix.)
NCEP_FILES = {
    "uflx": "uflx.sfc.gauss", "vflx": "vflx.sfc.gauss",
    "air": "air.2m.gauss", "uwnd": "uwnd.10m.gauss", "vwnd": "vwnd.10m.gauss",
    "pres": "pres.sfc.gauss", "prate": "prate.sfc.gauss",
    "weasd": "weasd.sfc.gauss", "soilw": "soilw.0-10cm.gauss",
    "tmp": "tmp.0-10cm.gauss", "lhtfl": "lhtfl.sfc.gauss",
    "shtfl": "shtfl.sfc.gauss", "skt": "skt.sfc.gauss",
}
NCEP_LAND = "land.sfc.gauss"
# Only these two need a second moment: tau_x_std / tau_y_std are the
# WITHIN-PENTAD population sigma, and a sigma is not aggregable from a mean.
NCEP_SIGMA = ("uflx", "vflx")
# The sign flip is on uflx/vflx ONLY (stress ON the surface), as family 4.
NCEP_FLIP = ("uflx", "vflx")

RG_START_YEAR, RG_START_MONTH = 2004, 1
RG_LAT_LO, RG_LAT_HI = -64.5, 79.5        # the band RG actually covers

STAGES = ["glorys", "sst", "ncep", "rg", "static", "truth", "norm", "meta",
          "publish"]
DEPS = {
    # `ncep` no longer writes into g025 at all, but it still runs the `sst`
    # repair below, which needs `oisst_seen.npy` from the sst stage.
    "ncep": ["sst"],
    "norm": ["glorys", "sst", "ncep", "rg"],
    "meta": ["norm", "static", "truth"],
    "publish": ["meta"],
}


# --------------------------------------------------------------- utilities --
def utcnow():
    return dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat()


def atomic_json(path, obj):
    """temp sibling + os.replace, so a reader never catches a half file."""
    tmp = f"{path}.tmp{os.getpid()}"
    with open(tmp, "w") as fh:
        json.dump(obj, fh, indent=1, sort_keys=True)
    os.replace(tmp, path)


def atomic_npz(path, **arrs):
    tmp = f"{path}.tmp{os.getpid()}.npz"
    np.savez(tmp, **arrs)
    os.replace(tmp, path)


def read_json(path, default=None):
    try:
        with open(path) as fh:
            return json.load(fh)
    except Exception:                                         # noqa: BLE001
        return {} if default is None else default


def marker(work, name):
    return os.path.join(work, name + ".done")


def marked(work, name):
    return os.path.exists(marker(work, name))


def mark(work, name):
    """Write a progress marker. ALWAYS after the flush it describes."""
    p = marker(work, name)
    os.makedirs(os.path.dirname(p), exist_ok=True)
    with open(p, "w") as fh:
        fh.write(utcnow() + "\n")


def bump_counts(work, **kw):
    p = os.path.join(work, "counts.json")
    c = read_json(p, {})
    c.update({k: int(v) for k, v in kw.items()})
    atomic_json(p, c)
    return c


class Progress:
    """`<work>/progress.json`, rewritten atomically after every item.

    ml/CLAUDE.md §5.25: progress is an ARTEFACT, not a log line. A progress
    line in a log the box takes with it is not progress anybody has.
    """

    def __init__(self, work):
        self.path = os.path.join(work, "progress.json")
        self.t0 = time.time()
        self.stage = None
        self.total = 0

    def stage_start(self, stage, total=0):
        self.stage, self.total, self.t0 = stage, total, time.time()
        self.item(f"start ({total} items)" if total else "start", 0)

    def item(self, name, done=None, extra=None):
        el = time.time() - self.t0
        rec = {"stage": self.stage, "item": str(name),
               "elapsed_s": round(el, 1), "at": utcnow()}
        if done is not None:
            rec["done"], rec["total"] = int(done), int(self.total)
            if done and self.total:
                rec["eta_s"] = round(el * (self.total - done) / done, 1)
        if extra:
            rec.update(extra)
        atomic_json(self.path, rec)
        print(f"  [{self.stage}] {name}"
              + (f"  {done}/{self.total}" if done is not None else "")
              + f"  {el:.1f}s", flush=True)


def sha256(path, buf=1 << 22):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for b in iter(lambda: fh.read(buf), b""):
            h.update(b)
    return h.hexdigest()


def git_sha():
    """Best-effort: the commit this builder ran from, for provenance."""
    try:
        r = subprocess.run(["git", "-C", HERE, "rev-parse", "HEAD"],
                           capture_output=True, text=True, timeout=20)
        return r.stdout.strip() if r.returncode == 0 else ""
    except Exception:                                         # noqa: BLE001
        return ""


def remote_size(url):
    """Content-Length, or None if the server will not say."""
    import urllib.request
    try:
        req = urllib.request.Request(
            url, method="HEAD",
            headers={"User-Agent": "earth-science-pipeline/1.0 "
                                   "(research; github blauewelt/earth)"})
        with urllib.request.urlopen(req, timeout=120) as r:
            n = r.headers.get("Content-Length")
        return int(n) if n else None
    except Exception:                                         # noqa: BLE001
        return None


def download_verified(url, path, mirrors=(), attempts=3):
    """Fetch `url` -> `path`, SIZE-VERIFIED — `fetch_sst_na.download_year`'s rule.

    Measured 2026-08-18: a 477 MB year came back short with no exception and
    the only symptom was `NetCDF: HDF error` at open time, i.e. one silently
    truncated transfer costs a whole year of the axis. Comparing against
    Content-Length turns that into a retry.
    """
    if os.path.exists(path):
        return path
    want = remote_size(url)
    for i in range(attempts):
        f3.fetch(url, path, mirrors=tuple(mirrors))
        got = os.path.getsize(path)
        if want is None or got == want:
            return path
        print(f"  ::warning:: {os.path.basename(path)}: {got:,} of {want:,} "
              f"bytes — truncated transfer, refetching ({i + 1}/{attempts})",
              flush=True)
        os.remove(path)
    raise IOError(f"{url}: could not be downloaded whole in {attempts} "
                  f"attempts (expected {want:,} bytes)")


def pick_var(ds, want):
    """The data variable in a netCDF, by name, else the fattest candidate.

    Verify the ARTEFACT, not the intention (ml/CLAUDE.md §0.1): a file whose
    variable is not spelled the way the docs say must fail loudly here rather
    than fill a channel with the wrong field.
    """
    if want in ds.variables:
        return ds.variables[want]
    cands = [v for k, v in ds.variables.items()
             if v.ndim >= 2 and k not in ("lat", "lon", "time", "level",
                                          "latitude", "longitude",
                                          "time_bnds")]
    if len(cands) == 1:
        return cands[0]
    raise KeyError(f"no variable {want!r} in {getattr(ds, 'filepath', lambda: '?')()}"
                   f" — has {sorted(ds.variables)}")


def squeeze_level(a):
    """Drop a size-1 `level` axis: the soil files carry one (plan §2)."""
    a = np.asarray(a)
    while a.ndim > 2 and a.shape[0] == 1:
        a = a[0]
    return a


def nc_dates(ds):
    import netCDF4 as ncdf
    tv = ds.variables["time"]
    ds_ = ncdf.num2date(tv[:], tv.units, only_use_cftime_datetimes=False)
    return [dt.date(d.year, d.month, d.day) for d in np.atleast_1d(ds_)]


# ------------------------------------------------------------- disk guard --
def disk_guard(work, sizes, headroom=3e9):
    """Refuse before anything is written (ml/CLAUDE.md §5.16, §5.18, §7).

    `sizes` is the byte cost of the memmaps this run may create. Over 90 %
    used is unusable, not a warning: a full-disk box computes fine and reports
    nothing, because every metrics write fails behind its own best-effort
    guard.
    """
    st = os.statvfs(work)
    free = st.f_bavail * st.f_frsize
    total = st.f_blocks * st.f_frsize
    used_frac = 1.0 - (st.f_bfree * st.f_frsize) / max(total, 1)
    need = sum(sizes.values()) + headroom
    print(f"disk      need {need / 1e9:.1f} GB "
          f"({' + '.join(f'{k} {v / 1e9:.1f}' for k, v in sizes.items())}"
          f" + {headroom / 1e9:.1f} headroom) · {free / 1e9:.1f} GB free of "
          f"{total / 1e9:.1f} GB ({used_frac:.0%} used)")
    if used_frac > 0.90:
        sys.exit(f"refusing to start: {used_frac:.0%} of {work}'s filesystem "
                 f"is used. Over 90 % is unusable, not a warning — every "
                 f"best-effort write starts failing silently (ml/CLAUDE.md §7)")
    if free < need:
        sys.exit(f"refusing to start: {need / 1e9:.1f} GB needed, "
                 f"{free / 1e9:.1f} GB free on {work}")


# ------------------------------------------------------------- the memmaps --
# THE TWO COARSE GROUPS ARE FILLED AT float32 AND CONVERTED BY `norm`.
#
# WHY, measured rather than argued. A memmap that is float16 from the first
# write quantises the RAW value at its own magnitude, before the norm stage
# ever sees it. The worst case is `sp`: near 1000 hPa the float16 grid is
# 0.5 hPa — about 0.08 sd, twenty times the ~0.003 sd the post-z-score float16
# write costs, and 5 % of a 10 hPa pressure anomaly. `log_swe`, `lhtfl` and
# `shtfl` have the same shape of problem one order down, and `rg_s` near
# 35 psu is quantised at 0.031 psu.
#
# g100 and rg100 are SMALL (5.7 and 1 GB), so their float32 intermediates cost
# 11.5 and 2 GB of TRANSIENT disk and the problem simply goes away: the fill
# stages write `<stem>_X_<group>.f32.npy`, the norm stage reads that, and the
# published float16 array is written ONCE, already z-scored. That also makes
# `norm` idempotent for these two — it reads one file and writes another — so
# a job killed mid-norm restarts from the f32 and `--force` is safe.
#
# g025 STAYS float16 in place: at 46 GB the float32 intermediate would be
# 92 GB and the box is sized for ~60. Its channels can afford it — the
# float16 quantum per channel, at the top of each one's range:
#   cur_speed, cur_u, cur_v  |v| <~ 3 m/s      -> 0.00098 m/s
#   log_mld                  0.5 .. 3.5        -> 0.00098 (log10 m)
#   ssh                      |v| <~ 2 m        -> 0.00098 m
#   sea_ice                  0 .. 1            -> 0.00049
#   sst                      -2 .. 35 degC     -> 0.031 degC   <- the worst
# `sst`'s 0.031 degC is exactly what family 4's `sst` channel already
# carries (build_family4.fill_sst), so this is precedent, not a new tax; it is
# still ~3x OISST's own 0.01 degC encoding, which is the honest caveat.
RAW_F32 = ("g100", "rg100")


def group_file(work, group):
    """The FINAL, published float16 sidecar for a group."""
    return os.path.join(work, f"{STEM}_X_{group}.npy")


def raw_file(work, group):
    """The TRANSIENT float32 array a coarse group is filled into."""
    return os.path.join(work, f"{STEM}_X_{group}.f32.npy")


def fill_file(work, group):
    """Where a FILL stage writes this group."""
    return raw_file(work, group) if group in RAW_F32 else group_file(work, group)


def _open_memmap(p, shape, dtype, create):
    if os.path.exists(p):
        m = np.lib.format.open_memmap(p, mode="r+")
        if tuple(m.shape) != tuple(shape):
            sys.exit(f"{p} is {m.shape}, this build wants {tuple(shape)} — "
                     f"the work dir belongs to a different axis. Use a fresh "
                     f"--work, or delete it deliberately.")
        return m
    if not create:
        sys.exit(f"{p} does not exist — run the stage that creates it first")
    m = np.lib.format.open_memmap(p, mode="w+", dtype=dtype, shape=tuple(shape))
    step = max(1, int(64e6 // max(int(np.prod(shape[1:])) * np.dtype(dtype).itemsize, 1)))
    for i in range(0, shape[0], step):
        m[i:i + step] = np.nan
    m.flush()
    return m


def open_fill(work, group, shape, create=False):
    """The array a fill stage writes: float32 for the coarse groups, else f16.

    Built straight at its final name, so the meta stage only has to write the
    npz beside the float16 sidecars — no rename of 46 GB, and the per-stage
    markers stay valid across the whole job.
    """
    dtype = np.float32 if group in RAW_F32 else np.float16
    return _open_memmap(fill_file(work, group), shape, dtype, create)


def open_final(work, group, shape, create=False):
    """The published float16 sidecar."""
    return _open_memmap(group_file(work, group), shape, np.float16, create)


# Kept as the name the stages used before the float32 intermediate existed.
open_group = open_fill


# ================================================================= context ==
class Ctx:
    """Everything every stage needs: the axis, the grids, the paths."""

    def __init__(self, a):
        self.a = a
        self.work = os.path.abspath(a.work)
        os.makedirs(self.work, exist_ok=True)
        self.source_dir = os.path.abspath(a.source_dir) if a.source_dir else None
        self.scratch = os.path.join(self.work, "src")
        os.makedirs(self.scratch, exist_ok=True)
        y0, m0, d0 = (int(x) for x in a.start.split("-"))
        y1, m1, d1 = (int(x) for x in a.end.split("-"))
        self.d_lo, self.d_hi = dt.date(y0, m0, d0), dt.date(y1, m1, d1)
        b_lo = bin_index(self.d_lo, PENTAD_DAYS)
        b_hi = bin_index(self.d_hi, PENTAD_DAYS)
        self.bins = list(range(b_lo, b_hi + 1))
        self.b_lo, self.b_hi = b_lo, b_hi
        self.T = len(self.bins)
        self.lats, self.lons = grid025()
        self.lat1, self.lon1 = grid100()
        self.prog = Progress(self.work)
        self.sources = read_json(os.path.join(self.work, "sources.json"), {})

    # -- axis helpers ------------------------------------------------------
    def row_of(self, b):
        return b - self.b_lo if self.b_lo <= b <= self.b_hi else None

    def bin_closed(self, b, seen_date):
        """True once a day at or after the bin's end has been observed."""
        return bin_start(b, PENTAD_DAYS) + dt.timedelta(days=PENTAD_DAYS) <= seen_date

    # -- source resolution -------------------------------------------------
    def local(self, *parts):
        return os.path.join(self.source_dir, *parts) if self.source_dir else None

    def note_source(self, key, value):
        if self.sources.get(key) != value:
            self.sources[key] = value
            atomic_json(os.path.join(self.work, "sources.json"), self.sources)

    def shapes(self, n_live=None):
        s = {"g025": (self.T, NLAT, NLON, NCHAN["g025"]),
             "g100": (self.T, NLAT1, NLON1, NCHAN["g100"])}
        if n_live is not None:
            s["rg100"] = (n_live, NLAT1, NLON1, NCHAN["rg100"])
        return s

    def byte_peak(self, n_live):
        """The most disk this build ever holds, itemised for the guard.

        Not the sum of the published files: the coarse groups are filled at
        float32 and converted by `norm`, and the peak is the moment g100's
        float16 sidecar is being written while its float32 source, rg100's
        float32 source and the whole of g025 are all still on disk.
        """
        sh = self.shapes(n_live)
        n = {k: int(np.prod(v)) for k, v in sh.items()}
        peak = {"g025 f16": n["g025"] * 2,
                "g100 f32": n["g100"] * 4, "g100 f16": n["g100"] * 2,
                "rg100 f32": n["rg100"] * 4}
        return peak



class Carry:
    """The open-bin accumulator, keyed by the item that produced it.

    WHY KEYED, and not one `carry.npz`. A pentad bin straddles month and year
    boundaries, so a stage must hand a PARTIAL accumulator to the next chunk.
    The carry and the chunk's `.done` marker cannot be written atomically
    together, and both orderings of one shared file are wrong: marker-first
    loses the chunk's contribution on a crash between them, carry-first
    DOUBLE-COUNTS it. Keying the carry by its item removes the choice — save
    `carry_<item>.npz`, THEN mark the item, THEN prune older carries. A resume
    loads the carry of the newest MARKED item, so a crash anywhere replays
    exactly the chunks whose markers are missing (ml/CLAUDE.md §5.21: a marker
    may only under-claim). `tests/test_build_family7.py` asserts a resumed
    build is bit-identical to the one-pass build.
    """

    def __init__(self, work, stage, items):
        self.dir = os.path.join(work, stage)
        self.work, self.stage = work, stage
        self.items = [str(i) for i in items]
        os.makedirs(self.dir, exist_ok=True)

    def path(self, item):
        return os.path.join(self.dir, f"carry_{item}.npz")

    def load(self):
        for it in reversed(self.items):
            p = self.path(it)
            if marked(self.work, f"{self.stage}/{it}") and os.path.exists(p):
                return np.load(p)
        return None

    def commit(self, item, **arrs):
        """Save this item's carry, mark the item, then drop older carries."""
        item = str(item)
        atomic_npz(self.path(item), **arrs)
        mark(self.work, f"{self.stage}/{item}")
        for q in glob.glob(os.path.join(self.dir, "carry_*.npz")):
            if os.path.basename(q) != f"carry_{item}.npz":
                try:
                    os.remove(q)
                except OSError:
                    pass



# ------------------------------------------------------- the sst repair ----
def repair_sst_channel(ctx):
    """`sst` is NaN wherever OISST never observes — asserted, not hoped.

    THE HISTORY THIS EXISTS FOR. The first version of this build wrote NCEP
    skin temperature into g025 channel 5 wherever OISST does not observe, as
    the "one quantity over two surfaces" channel E-070 B4 described. E-071
    §6.1's correction of 4 Sep retired that: a channel is shared only when the
    MEASURAND AND THE INSTRUMENT match on both sides, so the observed field
    keeps the channel and the reanalysis moves to `skt` in g100. A work dir
    that ran the old code carries that land fill in the bins whose NCEP year
    completed, and the rename alone would leave it there, silently, as
    "OISST values" over Siberia.

    So the repair runs AUTOMATICALLY at the head of the ncep stage — no
    operator flag, because a repair nobody remembers to ask for is not a
    repair. It is a strided pass over one channel of the 46 GB memmap, chunked
    over bins, and it PRINTS what it cleared: ~0 on a fresh build, the old
    fill on a resumed one. Marked when done, so a later ncep resume does not
    pay for it again.
    """
    work = ctx.work
    if marked(work, "repair_sst"):
        return 0
    seen_path = os.path.join(work, "oisst_seen.npy")
    if not os.path.exists(seen_path):
        sys.exit(f"{seen_path} absent — the `sst` stage must run before "
                 f"`ncep` (plan §4: stage order is fixed). The repair below "
                 f"cannot know which cells OISST observes without it.")
    oisst_seen = np.load(seen_path)
    shape = ctx.shapes()["g025"]
    X = open_fill(work, "g025", shape, create=True)
    dry = ~oisst_seen
    cleared = 0
    chunk = bin_chunk(shape)
    for i in range(0, shape[0], chunk):
        j = min(i + chunk, shape[0])
        slab = np.asarray(X[i:j, :, :, C_SST], np.float32)
        bad = np.isfinite(slab) & dry[None]
        n = int(bad.sum())
        if n:
            slab[bad] = np.nan
            X[i:j, :, :, C_SST] = slab
            cleared += n
        X.flush()
    mark(work, "repair_sst")
    print(f"  repair: cleared {cleared:,} value(s) from g025 `sst` where OISST "
          f"never observes"
          + ("" if cleared else " — nothing to undo, as on a fresh build"))
    return cleared


# --------------------------------------------------------- the spec hash ----
# Bump a stage's number when what it WRITES changes in a way its channel list
# and shapes do not already express (a transform, a sign, a masking rule). The
# digest below folds it together with the channel names and the array shapes,
# so a stage whose recipe moved discards its own half-built state instead of
# leaving a tensor half in one recipe and half in another.
SPEC_VERSION = {"glorys": 1, "sst": 1, "ncep": 2, "rg": 1, "static": 1,
                "truth": 1, "norm": 1, "meta": 1, "publish": 1}

# Which array a stage OWNS — the one it may delete when its spec moves. A
# stage never touches another stage's files: glorys and sst share g025 and own
# nothing, so a spec change there rewrites their own channels in place.
STAGE_OWNS = {"ncep": "g100", "rg": "rg100"}

# What each stage writes, for the digest. Deliberately its OWN channels, not
# the whole tensor: renaming a g100 channel must not send the OISST stage back
# to 1982.
STAGE_CHANNELS = {
    "glorys": CHAN_G025[:5],
    "sst": CHAN_G025[5:],
    "ncep": CHAN_G100,
    "rg": CHAN_RG100,
}


def stage_spec(ctx, stage, n_live=None):
    body = {
        "stage": stage, "version": SPEC_VERSION[stage], "recipe": RECIPE,
        "channels": STAGE_CHANNELS.get(stage, []),
        "min_days": MIN_DAYS, "pentad_days": PENTAD_DAYS,
        "epoch": str(EPOCH), "bins": [ctx.b_lo, ctx.b_hi],
        "shapes": {k: list(v) for k, v in ctx.shapes(n_live).items()
                   if k in (STAGE_OWNS.get(stage),
                            "g025" if stage in ("glorys", "sst") else "")},
    }
    if stage == "ncep":
        body["files"] = NCEP_FILES
        body["flip"] = list(NCEP_FLIP)
        body["sigma"] = list(NCEP_SIGMA)
    if stage == "rg":
        body["levels"] = list(LEVELS)
        body["band"] = [RG_LAT_LO, RG_LAT_HI]
    blob = json.dumps(body, sort_keys=True).encode()
    return hashlib.sha256(blob).hexdigest(), body


def stage_state_check(ctx, stage, n_live=None):
    """Discard a stage's OWN half-built state when its recipe has moved.

    Two triggers, because a work dir older than this mechanism has no `.spec`
    to compare against and its staleness has to be detectable some other way:

      1. `<work>/<stage>.spec` exists and differs from the current digest;
      2. the array this stage OWNS is on disk with the wrong shape — which is
         exactly how the 14-channel g100 left by the pre-correction ncep stage
         announces itself.

    What is discarded is only ever this stage's: its `<stage>.done`, everything
    under `<work>/<stage>/` (per-item markers and carries) and its own `.f32`
    fill file. `glorys` and `sst` share g025 and own no file, so a spec change
    there replays their chunks over the same memmap; `norm` REFUSES instead,
    because its g025 pass is in place and cannot be replayed.
    """
    work = ctx.work
    digest, body = stage_spec(ctx, stage, n_live)
    spec_path = os.path.join(work, f"{stage}.spec")
    old = None
    if os.path.exists(spec_path):
        old = read_json(spec_path, {}).get("sha256")
    why = []
    if old is not None and old != digest:
        why.append(f"its recorded spec {old[:12]}… != {digest[:12]}…")
    owned = STAGE_OWNS.get(stage)
    if owned and owned in ctx.shapes(n_live):
        p = fill_file(work, owned)
        want = tuple(ctx.shapes(n_live)[owned])
        if os.path.exists(p):
            got = tuple(np.lib.format.open_memmap(p, mode="r").shape)
            if got != want:
                why.append(f"{os.path.basename(p)} is {got}, wants {want}")
    if why:
        if stage in ("norm", "publish"):
            sys.exit(f"stage {stage!r} is stale ({'; '.join(why)}) but cannot "
                     f"be discarded automatically — its g025 pass is in place. "
                     f"Rebuild the work dir deliberately.")
        print(f"  ::warning:: stage {stage!r} is STALE ({'; '.join(why)}) — "
              f"discarding its own markers, carries and fill file and "
              f"restarting it. Nothing belonging to another stage is touched.")
        for q in [marker(work, stage)] + \
                sorted(glob.glob(os.path.join(work, stage, "*"))):
            if os.path.isfile(q):
                os.remove(q)
        if owned and os.path.exists(fill_file(work, owned)):
            os.remove(fill_file(work, owned))
    atomic_json(spec_path, {"sha256": digest, "at": utcnow(), "spec": body})
    return bool(why)


# ============================================================ stage: glorys ==
def glorys_chunk_names(ctx):
    """The `glorys025_global_YYYYMM.nc` chunks that touch the axis."""
    if ctx.source_dir:
        pat = os.path.join(ctx.source_dir, "daily025_global",
                           "glorys025_global_*.nc")
        return sorted(os.path.basename(p) for p in glob.glob(pat))
    return [f"glorys025_global_{y}{m:02d}.nc"
            for y in range(1993, 2025) for m in range(1, 13)]


def hub_repo(token=None):
    """`<namespace>/earth-tensors`, namespace resolved from whoami (hf_mirror)."""
    from huggingface_hub import HfApi
    tok = token or os.environ.get("HF_TOKEN") or (
        open("/home/claude/.hf_token").read().strip()
        if os.path.exists("/home/claude/.hf_token") else "")
    if not tok:
        sys.exit("no HF_TOKEN in the environment (never in argv) — see the "
                 "project doc claude/huggingface-access.md")
    api = HfApi(token=tok)
    return api, f"{api.whoami()['name']}/{HF_DATASET}", tok


def hub_get(ctx, path_in_repo, dest_dir):
    """Stream ONE file off the Hub into `dest_dir`; the caller deletes it."""
    from huggingface_hub import hf_hub_download
    api, repo, tok = hub_repo()
    os.makedirs(dest_dir, exist_ok=True)
    p = hf_hub_download(repo, path_in_repo, repo_type="dataset", token=tok,
                        local_dir=dest_dir)
    ctx.note_source("glorys" if path_in_repo.startswith("daily025") else "hub",
                    f"hf://{repo}/{os.path.dirname(path_in_repo)}/")
    return p


def drop_hub_copy(path, dest_dir):
    """Delete the streamed chunk AND the local-dir metadata beside it."""
    for p in (path,):
        if p and os.path.exists(p):
            os.remove(p)
    shutil.rmtree(os.path.join(dest_dir, ".cache"), ignore_errors=True)


def stage_glorys(ctx):
    """GLORYS12 daily chunks -> the five ocean channels of g025.

    STREAMED, ONE CHUNK AT A TIME, and the accumulator that a month boundary
    splits is CARRIED: a pentad bin spans five days, so the bin holding the
    last days of a month is finished only by the next chunk. `carry.npz` holds
    exactly those open bins and is written atomically BEFORE the chunk's
    marker, so a killed job resumes with the partial mean it had rather than
    writing a thinner one (ml/CLAUDE.md §5.21).
    """
    import netCDF4 as ncdf
    work = ctx.work
    X = open_group(work, "g025", ctx.shapes()["g025"], create=True)
    names = glorys_chunk_names(ctx)
    yms = [n.split("_")[-1].split(".")[0] for n in names]
    carry = Carry(work, "glorys", yms)
    seen_path = os.path.join(work, "glorys_seen.npy")

    acc, cnt = {}, {}
    n_bins = 0
    seen = np.zeros((NLAT, NLON), bool)
    d = carry.load()
    if d is not None:
        for k in d.files:
            if k.startswith("acc_"):
                acc[int(k[4:])] = d[k]
            elif k.startswith("cnt_"):
                cnt[int(k[4:])] = d[k]
        n_bins = int(d["n_bins"]) if "n_bins" in d.files else 0
    if os.path.exists(seen_path):
        seen = np.load(seen_path)

    def flush(b):
        nonlocal n_bins
        row = ctx.row_of(b)
        s, c = acc.pop(b), cnt.pop(b)
        if row is None:
            return
        with np.errstate(invalid="ignore", divide="ignore"):
            mu = np.where(c >= MIN_DAYS, s / np.maximum(c, 1), np.nan)
        uo, vo, ml, zs = mu
        sl = slice(GLORYS_ROW0, GLORYS_ROW0 + GLORYS_ROWS)
        spd = np.hypot(uo, vo)
        X[row, sl, :, C_CUR_SPEED] = spd
        with np.errstate(invalid="ignore", divide="ignore"):
            # log10 of the BINNED MEAN mixed-layer depth, matching family 4
            # (verified there against base025_na.npz). A non-positive value is
            # a fill leaking through, so it becomes missing rather than -inf.
            X[row, sl, :, C_LOG_MLD] = np.where(ml > 0,
                                                np.log10(np.maximum(ml, 1e-6)),
                                                np.nan)
        X[row, sl, :, C_SSH] = zs
        # cur_u/cur_v are the SAME binned means cur_speed is the hypotenuse
        # of — written in the same pass, out of the same arrays, so the
        # identity holds by construction rather than by re-derivation
        # (family 4's r3 rule). It is exact in float64 and holds to the
        # float16 write, which rounds all three independently.
        X[row, sl, :, C_CUR_U] = uo
        X[row, sl, :, C_CUR_V] = vo
        seen[sl, :] |= np.isfinite(spd)
        n_bins += 1

    def flush_ready(seen_date):
        for b in sorted(acc):
            if ctx.bin_closed(b, seen_date):
                flush(b)

    ctx.prog.stage_start("glorys", len(names))
    for i, (name, ym) in enumerate(zip(names, yms), 1):
        if marked(work, f"glorys/{ym}"):
            continue
        rel = f"daily025_global/{name}"
        if ctx.source_dir:
            path = ctx.local("daily025_global", name)
            if not os.path.exists(path):
                print(f"  ::warning:: {path} absent — that month is missing")
                mark(work, f"glorys/{ym}")
                continue
            drop = False
        else:
            path = hub_get(ctx, rel, os.path.join(ctx.scratch, "glorys"))
            drop = True
        try:
            d = ncdf.Dataset(path)
            lat = np.asarray(d.variables["latitude"][:], np.float64)
            lon = np.asarray(d.variables["longitude"][:], np.float64)
            # ASSERTED at read time, not assumed (plan §5.3). A silent
            # half-cell or whole-block offset here would put the Gulf Stream
            # at the latitude of the Norwegian Sea and still look plausible.
            if len(lat) != GLORYS_ROWS or float(lat[0]) != GLORYS_LAT0:
                sys.exit(f"{name}: latitude is {len(lat)} values starting at "
                         f"{float(lat[0])}, expected {GLORYS_ROWS} starting at "
                         f"{GLORYS_LAT0}")
            if len(lon) != NLON or float(lon[0]) != -180.0:
                sys.exit(f"{name}: longitude is {len(lon)} values starting at "
                         f"{float(lon[0])}, expected {NLON} starting at -180.0")

            row0 = int(round((float(lat[0]) - (-90.0)) / 0.25))
            if row0 != GLORYS_ROW0:
                sys.exit(f"{name}: lands at row {row0}, expected "
                         f"{GLORYS_ROW0}")
            dates = nc_dates(d)
            vs = [pick_var(d, v) for v in ("uo", "vo", "mlotst", "zos")]
            for k, day in enumerate(dates):
                b = bin_index(day, PENTAD_DAYS)
                flush_ready(day)
                if ctx.row_of(b) is None:
                    continue
                if b not in acc:
                    acc[b] = np.zeros((4, GLORYS_ROWS, NLON), np.float64)
                    cnt[b] = np.zeros((4, GLORYS_ROWS, NLON), np.int16)
                for vi, var in enumerate(vs):
                    v = np.ma.filled(np.asarray(var[k]), np.nan).astype(np.float64)
                    v = np.squeeze(v)
                    ok = np.isfinite(v)
                    acc[b][vi][ok] += v[ok]
                    cnt[b][vi] += ok
            d.close()
        finally:
            if drop:
                drop_hub_copy(path, os.path.join(ctx.scratch, "glorys"))
        X.flush()                                # flush, THEN mark
        np.save(seen_path + ".tmp.npy", seen)
        os.replace(seen_path + ".tmp.npy", seen_path)
        carry.commit(ym, n_bins=np.array(n_bins),
                     **{f"acc_{b}": v for b, v in acc.items()},
                     **{f"cnt_{b}": v for b, v in cnt.items()})
        ctx.prog.item(ym, i, {"open_bins": len(acc), "bins_written": n_bins})

    # The archive is exhausted: nothing later can touch what is still open.
    # The carry is deliberately NOT overwritten here — it still describes the
    # last MARKED chunk, so re-running the stage after the archive grows
    # reopens exactly the bins this terminal flush closed.
    for b in sorted(acc):
        flush(b)
    X.flush()
    np.save(seen_path + ".tmp.npy", seen)
    os.replace(seen_path + ".tmp.npy", seen_path)
    bump_counts(work, n_glorys_bins=n_bins)
    if ctx.source_dir:
        ctx.note_source("glorys", ctx.local("daily025_global") + "/")
    elif "glorys" not in ctx.sources:
        ctx.note_source("glorys", f"hf://<ns>/{HF_DATASET}/daily025_global/")
    mark(work, "glorys")
    print(f"  glorys: {n_bins}/{ctx.T} bins carry currents/MLD/SSH")


# =============================================================== stage: sst ==
def oisst_paths(ctx, year):
    """(sst, icec) for one year, downloaded and size-verified, or local."""
    if ctx.source_dir:
        p1 = ctx.local("oisst", f"sst.day.mean.{year}.nc")
        p2 = ctx.local("oisst", f"icec.day.mean.{year}.nc")
        return (p1 if os.path.exists(p1) else None,
                p2 if os.path.exists(p2) else None), False
    d = os.path.join(ctx.scratch, "oisst")
    os.makedirs(d, exist_ok=True)
    out = []
    for kind in ("sst", "icec"):
        name = f"{kind}.day.mean.{year}.nc"
        try:
            out.append(download_verified(f"{PSL_OISST}/{name}",
                                         os.path.join(d, name),
                                         mirrors=(f"{THREDDS_OISST}/{name}",)))
        except Exception as e:                                # noqa: BLE001
            print(f"  ::warning:: OISST {name} unavailable ({str(e)[:100]}) — "
                  f"that year is missing from this channel")
            out.append(None)
    return tuple(out), True


def drop_sentinels(a, limit=1e30):
    """PSL's `missing_value` is -9.96921e36; netCDF4 usually masks it, but a
    file that declares it without `_FillValue` can let it through as a number.
    Anything past +-1e30 is a sentinel, never a temperature or a fraction."""
    a = np.asarray(a, np.float64)
    return np.where(np.abs(a) > limit, np.nan, a)


def ice_divisor(v):
    """1.0 or 100.0 for OISST `icec`, decided by the FILE, not by its units.

    MEASURED 2026-09-04 from `icec.day.mean.2020.nc`'s DAS over PSL THREDDS:

        String units "percent";
        Float32 valid_range 0.0, 1.0;
        Float32 actual_range 0.08, 1.0;
        Float32 missing_value -9.96921E36;
        Float32 precision 2.0;

    The units STRING is wrong — the values are a 0..1 fraction, and both
    `valid_range` and `actual_range` say so. A reader that trusted the string
    would divide the sea-ice channel by a hundred and nothing downstream would
    notice: a plausible-looking field of 0.001-ish numbers, z-scored into the
    same shape it would have had. So the numeric ranges decide, and the units
    string only gets a vote when the file offers no range to check. There is no
    `scale_factor` (netCDF4 applies one automatically if a future file grows
    one).
    """
    hi = None
    for attr in ("valid_range", "actual_range"):
        r = getattr(v, attr, None)
        if r is not None and np.size(r) >= 2:
            hi = float(np.max(np.asarray(r, np.float64)))
            break
    if hi is not None:
        return 100.0 if hi > 1.5 else 1.0
    u = str(getattr(v, "units", "")).strip().lower()
    return 100.0 if ("percent" in u or u == "%") else 1.0


def stage_sst(ctx):
    """OISST v2.1 daily SST and sea-ice concentration -> g025 channels 5, 6.

    OISST is CELL-CENTRED (lat -89.875..89.875, lon 0.125..359.875) and the
    tensor samples ON multiples of 0.25, so every target point falls exactly
    halfway between two source centres in each axis. Nearest-indexing would
    displace the whole field by half a cell in both directions — invisible in
    any plot, fatal to every stencil. So it is bilinear, with `wrap_period`
    360 across the seam, exactly as `fetch_sst_na.weights_for` does it.

    `sea_ice` is NaN wherever OISST has no sea: the source masks land, the
    mask travels through `np.ma.filled(..., nan)` and `interp2_nan` keeps an
    all-NaN neighbourhood NaN.
    """
    import netCDF4 as ncdf
    work = ctx.work
    X = open_group(work, "g025", ctx.shapes()["g025"], create=True)
    years = list(range(ctx.d_lo.year, ctx.d_hi.year + 1))
    carry = Carry(work, "sst", years)
    seen_path = os.path.join(work, "oisst_seen.npy")

    acc, cnt = {}, {}
    n_days = 0
    seen = np.zeros((NLAT, NLON), bool)
    d = carry.load()
    if d is not None:
        for k in d.files:
            if k.startswith("acc_"):
                acc[int(k[4:])] = d[k]
            elif k.startswith("cnt_"):
                cnt[int(k[4:])] = d[k]
        n_days = int(d["n_days"]) if "n_days" in d.files else 0
    if os.path.exists(seen_path):
        seen = np.load(seen_path)

    def flush(b):
        row = ctx.row_of(b)
        s, c = acc.pop(b), cnt.pop(b)
        if row is None:
            return
        with np.errstate(invalid="ignore"):
            mu = np.where(c >= MIN_DAYS, s / np.maximum(c, 1), np.nan)
        X[row, :, :, C_SST] = mu[0]
        X[row, :, :, C_SEA_ICE] = np.clip(mu[1], 0.0, 1.0)
        # "finite in ANY bin" — the OISST-observed mask the ncep stage needs
        # to keep OISST precedence ABSOLUTE (plan §2, channel 5).
        np.logical_or(seen, np.isfinite(mu[0]), out=seen)

    def flush_ready(seen_date):
        for b in sorted(acc):
            if ctx.bin_closed(b, seen_date):
                flush(b)

    ctx.prog.stage_start("sst", len(years))
    wcache = {}
    for i, y in enumerate(years, 1):
        if marked(work, f"sst/{y}"):
            continue
        (p_sst, p_ice), drop = oisst_paths(ctx, y)
        try:
            if p_sst is None:
                mark(work, f"sst/{y}")
                continue
            dS = ncdf.Dataset(p_sst)
            dI = ncdf.Dataset(p_ice) if p_ice else None
            src_lat = np.asarray(dS.variables["lat"][:], np.float64)
            src_lon = np.asarray(dS.variables["lon"][:], np.float64)
            key = (src_lat.tobytes(), src_lon.tobytes())
            if wcache.get("key") != key:
                wcache["key"] = key
                wcache["w"] = (
                    f3.lin_weights(src_lat, ctx.lats),
                    f3.lin_weights(src_lon,
                                   np.where(ctx.lons < 0, ctx.lons + 360.0,
                                            ctx.lons),
                                   wrap_period=360.0))
                print(f"  interp {len(src_lat)}x{len(src_lon)} centres "
                      f"({src_lat[0]:+.3f}.., {src_lon[0]:.3f}..) -> "
                      f"{NLAT}x{NLON} points", flush=True)
            wy, wx = wcache["w"]
            v_sst = pick_var(dS, "sst")
            v_ice = pick_var(dI, "icec") if dI else None
            ice_div = ice_divisor(v_ice) if v_ice is not None else 1.0
            dates = nc_dates(dS)
            for k, day in enumerate(dates):
                if day < ctx.d_lo or day > ctx.d_hi:
                    continue
                b = bin_index(day, PENTAD_DAYS)
                flush_ready(day)
                if ctx.row_of(b) is None:
                    continue
                if b not in acc:
                    acc[b] = np.zeros((2, NLAT, NLON), np.float64)
                    cnt[b] = np.zeros((2, NLAT, NLON), np.int16)
                fields = [drop_sentinels(
                    np.ma.filled(np.asarray(v_sst[k]), np.nan))]
                if v_ice is not None:
                    ice = np.ma.filled(np.asarray(v_ice[k]), np.nan)
                    fields.append(drop_sentinels(ice) / ice_div)
                else:
                    fields.append(np.full_like(fields[0], np.nan))
                for vi, fld in enumerate(fields):
                    g = f3.interp2_nan(np.squeeze(fld).astype(np.float64), wy, wx)
                    ok = np.isfinite(g)
                    acc[b][vi][ok] += g[ok]
                    cnt[b][vi] += ok
                n_days += 1
            dS.close()
            if dI:
                dI.close()
        finally:
            if drop:
                for p in (p_sst, p_ice):
                    if p and os.path.exists(p):
                        os.remove(p)
        X.flush()                                # flush, THEN mark
        np.save(seen_path + ".tmp.npy", seen)
        os.replace(seen_path + ".tmp.npy", seen_path)
        carry.commit(y, n_days=np.array(n_days),
                     **{f"acc_{b}": v for b, v in acc.items()},
                     **{f"cnt_{b}": v for b, v in cnt.items()})
        ctx.prog.item(y, i, {"days": n_days})

    for b in sorted(acc):
        flush(b)
    X.flush()
    np.save(seen_path + ".tmp.npy", seen)
    os.replace(seen_path + ".tmp.npy", seen_path)
    bump_counts(work, n_sst_days=n_days)
    ctx.note_source("oisst", f"{PSL_OISST}/{{sst,icec}}.day.mean.YYYY.nc "
                             f"(OISST v2.1, NOAA PSL)")
    mark(work, "sst")
    print(f"  sst: {n_days} daily fields folded; OISST observes "
          f"{int(seen.sum()):,}/{NLAT * NLON} cells in at least one bin")


# ============================================================== stage: ncep ==
def ncep_year_paths(ctx, year, keys):
    """{key: path} for one year of gaussian dailies, plus whether to delete."""
    if ctx.source_dir:
        out = {}
        for k in keys:
            p = ctx.local("ncep", f"{NCEP_FILES[k]}.{year}.nc")
            if os.path.exists(p):
                out[k] = p
        return out, False
    d = os.path.join(ctx.scratch, "ncep")
    os.makedirs(d, exist_ok=True)
    out = {}
    for k in keys:
        name = f"{NCEP_FILES[k]}.{year}.nc"
        try:
            out[k] = download_verified(f"{PSL_NCEP}/{name}",
                                       os.path.join(d, name),
                                       mirrors=(f"{THREDDS_NCEP}/{name}",))
        except Exception as e:                                # noqa: BLE001
            print(f"  ::warning:: NCEP {name} unavailable ({str(e)[:100]}) — "
                  f"that variable-year is missing")
    return out, True


def ncep_land_mask(ctx):
    """The gaussian land/sea mask, applied BEFORE regridding (plan §2)."""
    import netCDF4 as ncdf
    if ctx.source_dir:
        p = ctx.local("ncep", f"{NCEP_LAND}.nc")
    else:
        p = os.path.join(ctx.scratch, "ncep", f"{NCEP_LAND}.nc")
        download_verified(f"{PSL_NCEP}/{NCEP_LAND}.nc", p,
                          mirrors=(f"{THREDDS_NCEP}/{NCEP_LAND}.nc",))
    if not p or not os.path.exists(p):
        print("  ::warning:: no gaussian land mask — soilw/tsoil will NOT be "
              "sea-masked, and those two channels will carry the model's "
              "meaningless over-ocean values")
        return None
    d = ncdf.Dataset(p)
    land = squeeze_level(np.ma.filled(np.asarray(pick_var(d, "land")[:]), 0.0))
    d.close()
    return np.asarray(land, np.float64) >= 0.5


def stage_ncep(ctx):
    """NCEP/NCAR R1 gaussian dailies -> the 15 g100 channels.

    Bilinear from the T62 gaussian grid (192 x 94, descending latitude,
    0..358.125 longitude) to the 1-degree POINT grid with `wrap_period` 360,
    pentad mean with >= 3 days. Accumulation happens on the NATIVE grid — 94 x
    192 is 0.14 MB, so a year of open bins costs nothing, and the sigma is the
    sigma of the quantity actually written (a standard deviation is not
    aggregable from a mean).

    `skt` (channel 14) is the SHARED surface temperature: the reanalysis field
    K -> degC over EVERY surface, land, sea and ice alike, with no land mask
    and no splice. It does NOT touch g025 — an earlier version of this build
    filled g025's temperature channel with NCEP wherever OISST does not
    observe, and E-071 §6.1's correction of 4 Sep retired that: one measurand
    read by two instruments is two channels, not one. `repair_sst_channel`
    below undoes that fill on a work dir that has it.
    """
    import netCDF4 as ncdf
    work = ctx.work
    repair_sst_channel(ctx)
    Xg = open_fill(work, "g100", ctx.shapes()["g100"], create=True)
    land = ncep_land_mask(ctx)

    years = list(range(ctx.d_lo.year, ctx.d_hi.year + 1))
    carry = Carry(work, "ncep", years)
    keys = list(NCEP_FILES)
    acc, acc2, cnt, dayset = {}, {}, {}, {}
    n_days = 0
    d = carry.load()
    if d is not None:
        for k in d.files:
            if k.startswith("acc2_"):
                b, v = k[5:].split("|")
                acc2[(int(b), v)] = d[k]
            elif k.startswith("acc_"):
                b, v = k[4:].split("|")
                acc[(int(b), v)] = d[k]
            elif k.startswith("cnt_"):
                b, v = k[4:].split("|")
                cnt[(int(b), v)] = d[k]
            elif k.startswith("day_"):
                b, v = k[4:].split("|")
                dayset[(int(b), v)] = set(int(x) for x in d[k])
        n_days = int(d["n_days"]) if "n_days" in d.files else 0

    W = {}                                   # gaussian -> 1 deg and -> 0.25 deg

    def weights(g_lat, g_lon):
        if W.get("key") == (g_lat.tobytes(), g_lon.tobytes()):
            return
        W["key"] = (g_lat.tobytes(), g_lon.tobytes())
        W["y1"] = f3.lin_weights(g_lat, ctx.lat1)
        W["x1"] = f3.lin_weights(g_lon,
                                 np.where(ctx.lon1 < 0, ctx.lon1 + 360.0,
                                          ctx.lon1), wrap_period=360.0)
        print(f"  interp gaussian {len(g_lat)}x{len(g_lon)} -> "
              f"{NLAT1}x{NLON1} (g100 only — nothing here writes g025)",
              flush=True)

    def enough(b, v):
        """>= MIN_DAYS DISTINCT DAYS in the bin — not >= MIN_DAYS samples."""
        return len(dayset.get((b, v), ())) >= MIN_DAYS

    def mean_of(b, v):
        key = (b, v)
        if key not in acc:
            return None
        c = cnt[key]
        with np.errstate(invalid="ignore"):
            mu = np.where((c > 0) & enough(b, v),
                          acc[key] / np.maximum(c, 1), np.nan)
        return mu

    def sigma_of(b, v):
        key = (b, v)
        c = cnt[key]
        with np.errstate(invalid="ignore"):
            mu = acc[key] / np.maximum(c, 1)
            var = acc2[key] / np.maximum(c, 1) - mu ** 2
            return np.where((c > 0) & enough(b, v),
                            np.sqrt(np.maximum(var, 0)), np.nan)

    def to1(f):
        return f3.interp2_nan(f, W["y1"], W["x1"])

    def flush(b):
        row = ctx.row_of(b)
        present = [v for v in keys if (b, v) in acc]
        if row is not None and present:
            def m(v):
                return mean_of(b, v) if (b, v) in acc else None

            def put(ci, arr):
                if arr is not None:
                    Xg[row, :, :, ci] = arr

            ux, vx = m("uflx"), m("vflx")
            put(0, to1(ux) if ux is not None else None)
            put(1, to1(vx) if vx is not None else None)
            if (b, "uflx") in acc:
                put(2, to1(sigma_of(b, "uflx")))
            if (b, "vflx") in acc:
                put(3, to1(sigma_of(b, "vflx")))
            air = m("air")
            put(4, to1(air) - 273.15 if air is not None else None)
            uw, vw = m("uwnd"), m("vwnd")
            put(5, to1(uw) if uw is not None else None)
            put(6, to1(vw) if vw is not None else None)
            pr = m("pres")
            put(7, to1(pr) / 100.0 if pr is not None else None)   # Pa -> hPa
            pp = m("prate")
            if pp is not None:
                put(8, np.log1p(np.maximum(to1(pp) * 86400.0, 0.0)))
            sw = m("weasd")
            if sw is not None:
                put(9, np.log1p(np.maximum(to1(sw), 0.0)))
            # soilw / tsoil are NaN over sea, masked ON THE GAUSSIAN GRID
            # before regridding so `interp2_nan` renormalises at the coast.
            so = m("soilw")
            if so is not None:
                put(10, to1(np.where(land, so, np.nan) if land is not None else so))
            ts = m("tmp")
            if ts is not None:
                t1 = to1(np.where(land, ts, np.nan) if land is not None else ts)
                put(11, t1 - 273.15)
            lh, sh = m("lhtfl"), m("shtfl")
            put(12, to1(lh) if lh is not None else None)
            put(13, to1(sh) if sh is not None else None)
            # THE SHARED SURFACE TEMPERATURE. No land mask, no OISST splice:
            # one instrument over every surface is the whole point of the
            # channel (E-071 §6.1, corrected 4 Sep).
            sk = m("skt")
            put(C_SKT, to1(sk) - 273.15 if sk is not None else None)
        for v in list(keys):
            acc.pop((b, v), None)
            acc2.pop((b, v), None)
            cnt.pop((b, v), None)
            dayset.pop((b, v), None)

    def flush_ready(seen_date):
        for b in sorted({k[0] for k in acc}):
            if ctx.bin_closed(b, seen_date):
                flush(b)

    ctx.prog.stage_start("ncep", len(years))
    for i, y in enumerate(years, 1):
        if marked(work, f"ncep/{y}"):
            continue
        paths, drop = ncep_year_paths(ctx, y, keys)
        try:
            for v, p in sorted(paths.items()):
                d = ncdf.Dataset(p)
                g_lat = np.asarray(d.variables["lat"][:], np.float64)
                g_lon = np.asarray(d.variables["lon"][:], np.float64)
                weights(g_lat, g_lon)
                var = pick_var(d, v)
                dates = nc_dates(d)
                for k, day in enumerate(dates):
                    if day < ctx.d_lo or day > ctx.d_hi:
                        continue
                    b = bin_index(day, PENTAD_DAYS)
                    if ctx.row_of(b) is None:
                        continue
                    f = squeeze_level(np.ma.filled(np.asarray(var[k]), np.nan))
                    f = np.asarray(f, np.float64)
                    if v in NCEP_FLIP:
                        f = -f          # stress ON the surface, once, here
                    key = (b, v)
                    if key not in acc:
                        acc[key] = np.zeros(f.shape, np.float64)
                        acc2[key] = np.zeros(f.shape, np.float64)
                        cnt[key] = np.zeros(f.shape, np.int32)
                        dayset[key] = set()
                    ok = np.isfinite(f)
                    acc[key][ok] += f[ok]
                    if v in NCEP_SIGMA:
                        acc2[key][ok] += f[ok] ** 2
                    cnt[key] += ok
                    ord_ = (day - EPOCH).days
                    if v == "skt" and ord_ not in dayset[key]:
                        n_days += 1                  # DAYS, not 6-hourly steps
                    dayset[key].add(ord_)
                d.close()
            # A bin may straddle 31 Dec / 1 Jan, so only bins the NEXT year
            # cannot touch are closed here; the rest carry.
            flush_ready(dt.date(y + 1, 1, 1))
        finally:
            if drop:
                for p in paths.values():
                    if os.path.exists(p):
                        os.remove(p)
        Xg.flush()                               # flush, THEN mark
        carry.commit(y, n_days=np.array(n_days),
                     **{f"acc_{b}|{v}": a for (b, v), a in acc.items()},
                     **{f"acc2_{b}|{v}": a for (b, v), a in acc2.items()},
                     **{f"cnt_{b}|{v}": a for (b, v), a in cnt.items()},
                     **{f"day_{b}|{v}": np.array(sorted(a), np.int64)
                        for (b, v), a in dayset.items()})
        ctx.prog.item(y, i, {"skt_days": n_days})

    for b in sorted({k[0] for k in acc}):
        flush(b)
    Xg.flush()
    bump_counts(work, n_ncep_days=n_days)
    ctx.note_source("ncep", f"{PSL_NCEP}/<var>.gauss.YYYY.nc "
                            f"(NCEP/NCAR Reanalysis 1, NOAA PSL) + "
                            f"{NCEP_LAND}.nc")
    mark(work, "ncep")
    print(f"  ncep: {n_days} daily skt fields; "
          f"{NCHAN['g100']} g100 channels written")


# ================================================================ stage: rg ==
def rg_cube(ctx, stem, remote=None):
    """A Roemmich-Gilson NetCDF, from `--source-dir`, ml/cache/rg, or SIO."""
    if ctx.source_dir:
        p = ctx.local("rg", stem + ".nc")
        return p if os.path.exists(p) else None
    p = f3.rg_file(stem)
    if p:
        return p
    if not remote:
        return None
    try:
        f3.fetch(remote, os.path.join(CACHE, "rg", stem + ".nc.gz"))
    except Exception as e:                                    # noqa: BLE001
        print(f"  ::warning:: RG {stem} unavailable ({str(e)[:100]})")
        return None
    return f3.rg_file(stem)


def rg_extension_months(ctx):
    """The YYYYMM extension cubes on disk, as family 4 discovers them."""
    root = os.path.join(ctx.source_dir, "rg") if ctx.source_dir \
        else os.path.join(CACHE, "rg")
    return sorted({os.path.basename(p).split(".")[0].split("_")[1]
                   for p in glob.glob(os.path.join(root, "RG_2*.nc*"))})


def stage_rg(ctx):
    """RG monthly T/S -> `rg100`, ONE live bin per month (E-034 §4).

    Written once per month into the pentad that CONTAINS THE 15TH: RG carries
    no within-month time and the 15th is the month's midpoint, so the live
    pentad is the one a monthly mean is most nearly centred on. Forward-filling
    was rejected — it would tell the model the subsurface was observed on days
    it was not, and the `missing` token is distinct from `mask` by design.

    THE LATITUDE BAND IS EXPLICITLY NaN. `f3.lin_weights` CLAMPS at the axis
    ends, so without this the RG edge rows (-64.5 and 79.5) would be replicated
    into the Southern Ocean and the Arctic and read as measurements.
    """
    import netCDF4 as ncdf
    work = ctx.work
    os.makedirs(os.path.join(work, "rg"), exist_ok=True)
    tf = rg_cube(ctx, "RG_T", f"{RG_BASE}/RG_ArgoClim_Temperature_2019.nc.gz")
    sf = rg_cube(ctx, "RG_S", f"{RG_BASE}/RG_ArgoClim_Salinity_2019.nc.gz")
    live_path = os.path.join(work, "rg", "live.npz")
    if not tf or not sf:
        print("  ::warning:: RG cubes not available — rg100 will be EMPTY "
              "(n_live = 0). Seed ml/cache/rg from data-cache-v1 for a real "
              "build.")
        atomic_npz(live_path, bin_index=np.zeros(0, np.int64),
                   months=np.array([], dtype="<U7"))
        open_group(work, "rg100", (0, NLAT1, NLON1, NCHAN["rg100"]), create=True)
        bump_counts(work, n_rg_live=0)
        mark(work, "rg")
        return

    dT, dS = ncdf.Dataset(tf), ncdf.Dataset(sf)
    press = np.asarray(dT.variables["PRESSURE"][:], np.float64)
    lidx = [int(np.argmin(np.abs(press - p))) for p in LEVELS]
    L = len(LEVELS)
    rg_lat = np.asarray(dT.variables["LATITUDE"][:], np.float64)
    rg_lon = np.asarray(dT.variables["LONGITUDE"][:], np.float64)
    wy = f3.lin_weights(rg_lat, ctx.lat1)
    lon360 = np.where(ctx.lon1 < 20.0, ctx.lon1 + 360.0, ctx.lon1)
    wx = f3.lin_weights(rg_lon, lon360, wrap_period=360.0)
    band = (ctx.lat1 >= RG_LAT_LO) & (ctx.lat1 <= RG_LAT_HI)

    nbase = int(dT.variables["ARGO_TEMPERATURE_ANOMALY"].shape[0])
    months = []
    for k in range(nbase):
        y, m = RG_START_YEAR + (RG_START_MONTH - 1 + k) // 12, \
            (RG_START_MONTH - 1 + k) % 12 + 1
        months.append((f"{y:04d}{m:02d}", ("base", k)))
    for ym in rg_extension_months(ctx):
        months.append((ym, ("ext", ym)))
    live = []
    for ym, src in months:
        b = bin_index(dt.date(int(ym[:4]), int(ym[4:]), 15), PENTAD_DAYS)
        if ctx.row_of(b) is None:
            continue
        live.append((b, ym, src))
    live.sort(key=lambda t: t[0])
    n_live = len(live)
    atomic_npz(live_path,
               bin_index=np.array([b for b, _, _ in live], np.int64),
               months=np.array([f"{ym[:4]}-{ym[4:]}" for _, ym, _ in live]))

    X = open_group(work, "rg100", (n_live, NLAT1, NLON1, NCHAN["rg100"]),
                   create=True)
    mean_t = np.ma.filled(dT.variables["ARGO_TEMPERATURE_MEAN"][lidx], np.nan)
    mean_s = np.ma.filled(dS.variables["ARGO_SALINITY_MEAN"][lidx], np.nan)
    anom_t = dT.variables["ARGO_TEMPERATURE_ANOMALY"]
    anom_s = dS.variables["ARGO_SALINITY_ANOMALY"]

    def write(row, at, as_):
        for k in range(L):
            t = f3.interp2_nan(np.asarray(mean_t[k] + at[k], np.float64), wy, wx)
            s = f3.interp2_nan(np.asarray(mean_s[k] + as_[k], np.float64), wy, wx)
            t[~band] = np.nan
            s[~band] = np.nan
            X[row, :, :, k] = t
            X[row, :, :, L + k] = s

    ctx.prog.stage_start("rg", n_live)
    for row, (b, ym, src) in enumerate(live):
        if marked(work, f"rg/{ym}"):
            continue
        if src[0] == "base":
            k = src[1]
            write(row, np.ma.filled(anom_t[k][lidx], np.nan),
                  np.ma.filled(anom_s[k][lidx], np.nan))
        else:
            p = rg_cube(ctx, f"RG_{ym}")
            if not p:
                continue
            dE = ncdf.Dataset(p)
            write(row, np.ma.filled(dE.variables["ARGO_TEMPERATURE_ANOMALY"][0][lidx],
                                    np.nan),
                  np.ma.filled(dE.variables["ARGO_SALINITY_ANOMALY"][0][lidx],
                               np.nan))
            dE.close()
        X.flush()
        mark(work, f"rg/{ym}")
        if row % 24 == 0 or row == n_live - 1:
            ctx.prog.item(ym, row + 1)
    dT.close()
    dS.close()
    X.flush()
    bump_counts(work, n_rg_live=n_live)
    ctx.note_source("rg", f"{RG_BASE}/RG_ArgoClim_{{Temperature,Salinity}}_2019"
                          f".nc.gz + RG_YYYYMM extensions")
    mark(work, "rg")
    print(f"  rg: {n_live} live bins (one per month, the bin holding the 15th)"
          f"; NaN outside {RG_LAT_LO}..{RG_LAT_HI}")


# ============================================================ stage: static ==
def ne_geojson(ctx, name):
    if ctx.source_dir:
        p = ctx.local("ne", name + ".geojson")
        return p if os.path.exists(p) else None
    p = os.path.join(ctx.scratch, "ne", name + ".geojson")
    try:
        f3.fetch(NE_BASE + name + ".geojson", p)
    except Exception as e:                                    # noqa: BLE001
        print(f"  ::warning:: Natural Earth {name} unavailable ({str(e)[:100]})")
        return None
    return p


def polygon_hits(path, lats, lons, mask=None):
    """Which grid POINTS fall inside any polygon of a GeoJSON. shapely STRtree.

    The grid is POINT-aligned, so "the cell centre" IS the grid point: the
    0.25-degree cell owned by (lat, lon) is centred on exactly that lat/lon.
    """
    import json as _json
    import shapely
    from shapely.geometry import shape
    from shapely.strtree import STRtree

    hit = np.zeros((len(lats), len(lons)), bool)
    if not path:
        return hit
    with open(path) as fh:
        feats = _json.load(fh)["features"]
    polys = []
    for f in feats:
        g = f.get("geometry")
        if not g:
            continue
        s = shape(g)
        polys.extend(list(s.geoms) if s.geom_type == "MultiPolygon" else [s])
    if not polys:
        return hit
    tree = STRtree(polys)
    lons = np.asarray(lons, np.float64)
    for y in range(len(lats)):
        if mask is not None and not mask[y].any():
            continue
        cols = np.nonzero(mask[y])[0] if mask is not None \
            else np.arange(len(lons))
        pts = shapely.points(lons[cols],
                             np.full(cols.shape, float(lats[y])))
        idx = tree.query(pts, predicate="intersects")
        if idx.size:
            hit[y, cols[np.unique(idx[0])]] = True
    return hit


def etopo_path(ctx):
    if ctx.source_dir:
        for p in sorted(glob.glob(os.path.join(ctx.source_dir, "etopo", "*.nc"))):
            return p
        return None
    p = os.path.join(ctx.scratch, "etopo",
                     "ETOPO_2022_v1_60s_N90W180_surface.nc")
    try:
        f3.fetch(ETOPO_URL, p)
    except Exception as e:                                    # noqa: BLE001
        print(f"  ::warning:: ETOPO unavailable ({str(e)[:120]})")
        return None
    return p


def block_mean_elev(path, lats, lons):
    """ETOPO cell means -> the 0.25-degree cell centred on each grid point.

    THE HALF-CELL OFFSET, stated because it is real and small. ETOPO 2022 60s
    is CELL-registered (`node_offset 1`, GeoTransform "-180 1/60 0 90 0
    -1/60"): source cell centres sit at -180 + (j+0.5)/60 and 90 - (i+0.5)/60,
    i.e. on HALF-integer multiples of the source spacing, while our grid points
    sit on exact multiples of 0.25. The 0.25-degree box centred on a grid point
    therefore covers 15 source cells' width but starts and ends mid-cell, so a
    15 x 15 block mean cannot be exactly centred on the point.

    We take the block of `f` cells starting at `floor(p + 0.5) - f//2`, where
    `p` is the point's position in source-index space. Its mean centre is half
    a source cell — 1/120 degree, about 0.93 km — EAST of and (for a
    north-first source) SOUTH of the grid point. That is 1/30 of the target
    cell and far below the elevation field's own representativeness at 0.25
    degrees; the alternative offsets it the other way by the same amount.
    `f` is derived from the file's own spacing, never assumed to be 15.
    """
    import netCDF4 as ncdf
    d = ncdf.Dataset(path)
    try:
        zv = pick_var(d, "z")
        s_lat = np.asarray(d.variables["lat"][:], np.float64)
        s_lon = np.asarray(d.variables["lon"][:], np.float64)
        dlat = float(s_lat[1] - s_lat[0])
        dlon = float(s_lon[1] - s_lon[0])
        f = int(round(0.25 / abs(dlat)))
        fx = int(round(0.25 / abs(dlon)))
        if f < 1 or fx < 1:
            sys.exit(f"{path}: spacing {dlat} x {dlon} is coarser than 0.25 deg")
        nlat_s, nlon_s = len(s_lat), len(s_lon)
        fill = float(getattr(zv, "_FillValue", -99999.0))

        # column blocks, wrapping in longitude
        p_lon = (np.asarray(lons) - s_lon[0]) / dlon
        j0 = np.floor(p_lon + 0.5).astype(np.int64) - fx // 2
        cols = (j0[:, None] + np.arange(fx)[None, :]) % nlon_s     # (NLON, fx)

        out = np.full((len(lats), len(lons)), np.nan, np.float32)
        p_lat = (np.asarray(lats) - s_lat[0]) / dlat
        i0 = np.floor(p_lat + 0.5).astype(np.int64) - f // 2
        i0 = np.clip(i0, 0, nlat_s - f)

        # Read in bands so a 933 MB raster never lands in RAM whole.
        order = np.argsort(i0)
        band = max(1, int(4e7 // (nlon_s * 4)))                # ~40 MB a band
        k = 0
        while k < len(order):
            lo = i0[order[k]]
            hi = lo + f
            sel = [order[k]]
            k += 1
            while k < len(order) and i0[order[k]] + f - lo <= band:
                hi = max(hi, i0[order[k]] + f)
                sel.append(order[k])
                k += 1
            slab = np.ma.filled(np.asarray(zv[lo:hi, :]), np.nan)
            slab = np.asarray(slab, np.float64)
            slab = np.where(np.isclose(slab, fill), np.nan, slab)
            for y in sel:
                rows = slab[i0[y] - lo:i0[y] - lo + f]          # (f, nlon_s)
                blk = rows[:, cols]                             # (f, NLON, fx)
                with f4.warnings_suppressed():
                    out[y] = np.nanmean(blk, axis=(0, 2)).astype(np.float32)
        return out
    finally:
        d.close()


def sphere_codes(ice, ocean, lake):
    """0 ocean · 1 land · 2 ice sheet · 3 inland water, in THAT priority.

    Ice before ocean because the surface an instrument sees on an ice shelf is
    ice; then ocean, because "something observed water here" is a measurement;
    then lakes; land is the residue. Pure, so the priority is testable without
    a build (plan §5.7).
    """
    ice = np.asarray(ice, bool)
    ocean = np.asarray(ocean, bool)
    lake = np.asarray(lake, bool)
    sphere = np.ones(ice.shape, np.int8)             # 1 = land, the residue
    sphere[ocean] = 0
    sphere[lake & ~ocean] = 3
    sphere[ice] = 2
    return sphere


def stage_static(ctx):
    """`sphere` (0 ocean · 1 land · 2 ice sheet · 3 inland water) and `elev`.

    ICE BEFORE OCEAN, deliberately (plan §2): the surface an instrument sees on
    an ice shelf is ice, so a glaciated polygon wins even where OISST reports a
    temperature. Then ocean, because "something observed water here" is a
    measurement; then lakes; then land as the residue.
    """
    work = ctx.work
    lats, lons = ctx.lats, ctx.lons
    ocean = np.zeros((NLAT, NLON), bool)
    for p in (os.path.join(work, "oisst_seen.npy"),
              os.path.join(work, "glorys_seen.npy")):
        if os.path.exists(p):
            ocean |= np.load(p)
        else:
            print(f"  ::warning:: {os.path.basename(p)} absent — `sphere`'s "
                  f"ocean code is built from the other source only")
    ctx.prog.stage_start("static", 3)

    ice = polygon_hits(ne_geojson(ctx, "ne_10m_glaciated_areas"), lats, lons)
    ctx.prog.item("glaciated areas", 1, {"cells": int(ice.sum())})
    todo = (~ice) & (~ocean)
    lake = polygon_hits(ne_geojson(ctx, "ne_10m_lakes"), lats, lons, mask=todo)
    ctx.prog.item("lakes", 2, {"cells": int(lake.sum())})

    sphere = sphere_codes(ice, ocean, lake)

    ep = etopo_path(ctx)
    elev = (block_mean_elev(ep, lats, lons) if ep
            else np.full((NLAT, NLON), np.nan, np.float32))
    if ep:
        ctx.note_source("etopo", ETOPO_URL if not ctx.source_dir else ep)
    ctx.prog.item("elev", 3, {"finite": int(np.isfinite(elev).sum())})

    atomic_npz(os.path.join(work, "statics.npz"), sphere=sphere, elev=elev)
    ctx.note_source("naturalearth", NE_BASE +
                    "{ne_10m_glaciated_areas,ne_10m_lakes}.geojson")
    mark(work, "static")
    u, c = np.unique(sphere, return_counts=True)
    print("  sphere: " + " · ".join(
        f"{n}={cc:,}" for n, cc in zip(u.tolist(), c.tolist())))


# ============================================================= stage: truth ==
def truth_files(ctx):
    """`ml/cache/truth/truth_pentad.npz`, pulled off the Hub when absent."""
    root = os.path.join(ctx.source_dir, "truth") if ctx.source_dir \
        else os.path.join(CACHE, "truth")
    os.makedirs(root, exist_ok=True)
    want = ["truth_pentad.npz", "truth_daily.npz"]
    if not ctx.source_dir:
        for name in want:
            if os.path.exists(os.path.join(root, name)):
                continue
            try:
                p = hub_get(ctx, f"truth/{name}", root)
                print(f"  pulled truth/{name} from the Hub -> {p}")
            except Exception as e:                            # noqa: BLE001
                print(f"  ::warning:: truth/{name}: {str(e)[:140]}")
        shutil.rmtree(os.path.join(root, ".cache"), ignore_errors=True)
    return os.path.join(root, "truth_pentad.npz")


def stage_truth(ctx):
    """The Atlantic transport labels, on THIS axis — family 4's own function.

    The builder REFUSES without them, exactly as family 4's `missing_truth_keys`
    does: run #365 built a physically perfect 33 GB tensor with no labels at
    all, the recipe guard then skipped the rebuild, and the next run would have
    trained for twenty hours and died on `KeyError: 'rapid'`.
    """
    work = ctx.work
    path = truth_files(ctx)
    if not os.path.exists(path):
        sys.exit(f"no {path} — the labels are Atlantic and the stage-2 gates "
                 f"need them. Publish truth/truth_pentad.npz to the Hub or "
                 f"run ml/build_truth_pentad.py.")
    truths = f4.truth_pentad(ctx.bins, PENTAD_DAYS, path=path)
    if "truth_rapid" in truths:
        truths.setdefault("rapid", truths["truth_rapid"])
    lack = f4.missing_truth_keys(set(truths), path)
    if lack:
        sys.exit(f"REFUSING: {path} offers {lack} but this axis carries none "
                 f"of them. A state tensor with no transport labels trains for "
                 f"twenty hours and dies in the probe.")
    atomic_npz(os.path.join(work, "truth.npz"), **truths)
    mark(work, "truth")
    print(f"  truth: {len(truths)} label series attached")


# ============================================================== stage: norm ==
def bin_chunk(shape, budget=2e8):
    """How many bins fit `budget` bytes of float32 working set."""
    per = max(int(np.prod(shape[1:])) * 4, 1)
    return max(1, int(budget // per))


def channel_stats(X, chunk=None):
    """(mean, sd, count) per channel over EVERY finite value, float64 sums.

    Chunked over bins, so the resident cost is one chunk regardless of the
    46 GB the array may be. Family 4's estimator, including its `+ 1e-6` on
    the sd — a channel with no variance must not divide by zero.
    """
    chunk = chunk or bin_chunk(X.shape)
    C = X.shape[-1]
    cnt = np.zeros(C, np.int64)
    s1 = np.zeros(C, np.float64)
    s2 = np.zeros(C, np.float64)
    ax = tuple(range(X.ndim - 1))
    for i in range(0, X.shape[0], chunk):
        slab = np.asarray(X[i:i + chunk], np.float32)
        fin = np.isfinite(slab)
        cnt += fin.sum(ax)
        v = np.where(fin, slab, 0.0).astype(np.float64)
        s1 += v.sum(ax)
        s2 += (v ** 2).sum(ax)
    mu = np.where(cnt > 0, s1 / np.maximum(cnt, 1), 0.0)
    sd = np.sqrt(np.maximum(s2 / np.maximum(cnt, 1) - mu ** 2, 0.0)) + 1e-6
    return mu, sd, cnt


def zscore_chunk(X, mu, sd, i, chunk, out=None):
    """z-score bins [i, i+chunk) as float16; returns the next unprocessed bin.

    `out=None` writes back IN PLACE (g025, which has no float32 intermediate).
    `out` writes into a separate float16 array, which is what makes the coarse
    groups' norm idempotent: the float32 source is never modified, so a killed
    job simply redoes the chunk it lost.
    """
    j = min(i + chunk, X.shape[0])
    dst = X if out is None else out
    dst[i:j] = ((np.asarray(X[i:j], np.float32) - mu) / sd).astype(np.float16)
    return j


def stage_norm(ctx):
    """Per group, per channel (mean, sd) over EVERY finite value, then z-score.

    Family 4's convention (`build_family4.py:913-935`), with one deliberate
    difference the plan spells out: **the `slab[~ocean] = NaN` line is gone**.
    Land is observed now — that is the whole point of family 7 — so masking it
    would delete the shared channels it was written to carry.

    Resumable INSIDE a group. The statistics are computed and written first;
    the z-score then walks the bins in chunks and records the next unprocessed
    bin after every flush, so a killed job never z-scores the same slab twice
    (which would silently square the transform).
    """
    work = ctx.work
    os.makedirs(os.path.join(work, "norm"), exist_ok=True)
    live = np.load(os.path.join(work, "rg", "live.npz"))
    n_live = len(live["bin_index"])
    shapes = ctx.shapes(n_live)
    norm_path = os.path.join(work, "norm.npz")

    norms = {}
    if os.path.exists(norm_path):
        d = np.load(norm_path)
        norms = {k: d[k] for k in d.files}

    ctx.prog.stage_start("norm", len(GROUPS) * 2)
    step = 0
    for g in GROUPS:
        key = f"norm_{g}"
        if key in norms:
            continue
        X = open_fill(work, g, shapes[g])
        mu, sd, cnt = channel_stats(X)
        norms[key] = np.stack([mu, sd], 1).astype(np.float32)
        norms[f"count_{g}"] = cnt
        atomic_npz(norm_path, **norms)
        step += 1
        ctx.prog.item(f"{g} stats", step,
                      {"observed_values": int(cnt.sum())})

    # g025's z-score is IN PLACE and therefore not repeatable, so its RAW
    # statistics are parked in their own file BEFORE that pass runs. With them
    # on disk a partially z-scored g025 is at least diagnosable, and the
    # `norm/g025` marker is what the --force refusal keys on (run_stages).
    if "norm_g025" in norms:
        atomic_npz(os.path.join(work, "norm_g025.npz"),
                   norm_g025=norms["norm_g025"], count_g025=norms["count_g025"])

    for g in GROUPS:
        prog_path = os.path.join(work, "norm", f"{g}.progress.json")
        done_at = read_json(prog_path, {}).get("next_bin", 0)
        if marked(work, f"norm/{g}"):
            step += 1
            continue
        mu = norms[f"norm_{g}"][:, 0].astype(np.float32)
        sd = norms[f"norm_{g}"][:, 1].astype(np.float32)
        src = open_fill(work, g, shapes[g])
        out = open_final(work, g, shapes[g], create=True) \
            if g in RAW_F32 else None
        chunk = bin_chunk(shapes[g])
        for i in range(done_at, src.shape[0], chunk):
            j = zscore_chunk(src, mu, sd, i, chunk, out=out)
            (out if out is not None else src).flush()   # flush, THEN mark
            atomic_json(prog_path, {"next_bin": j, "at": utcnow()})
        if out is not None:
            out.flush()
            mark(work, f"norm/{g}")
            # The float32 intermediate is transient by design: it exists only
            # so the published float16 is written ONCE, already z-scored.
            del src, out
            os.remove(raw_file(work, g))
        else:
            src.flush()
            mark(work, f"norm/{g}")
        step += 1
        ctx.prog.item(f"{g} z-score", step)

    mark(work, "norm")
    for g in GROUPS:
        n = norms[f"norm_{g}"]
        print(f"  norm_{g}: {n.shape[0]} channels, "
              f"{int(norms[f'count_{g}'].sum()):,} observed values")


# ============================================================== stage: meta ==
def stage_meta(ctx):
    """The small npz beside the three memmaps: axes, statics, norms, truth."""
    work = ctx.work
    live = np.load(os.path.join(work, "rg", "live.npz"))
    statics = np.load(os.path.join(work, "statics.npz"))
    truths = np.load(os.path.join(work, "truth.npz"))
    norms = np.load(os.path.join(work, "norm.npz"))
    counts = read_json(os.path.join(work, "counts.json"), {})
    ctx.prog.stage_start("meta", 1)

    months = np.array([f"{bin_start(b, PENTAD_DAYS).year:04d}-"
                       f"{bin_start(b, PENTAD_DAYS).month:02d}"
                       for b in ctx.bins])
    meta = dict(
        bin_index=np.array(ctx.bins, np.int64), months=months,
        epoch=np.array(str(EPOCH)), pentad_days=np.array(PENTAD_DAYS),
        lats=ctx.lats, lons=ctx.lons, lat1=ctx.lat1, lon1=ctx.lon1,
        chan_g025=np.array(CHAN_G025), chan_g100=np.array(CHAN_G100),
        chan_rg100=np.array(CHAN_RG100), groups=np.array(GROUPS),
        window=np.array("global025"), recipe=np.array(RECIPE),
        cadence=np.array("pentad"),
        rg_bin_index=np.asarray(live["bin_index"], np.int64),
        rg_months=live["months"],
        sphere=statics["sphere"], elev=statics["elev"],
        n_glorys_bins=np.array(counts.get("n_glorys_bins", 0)),
        n_sst_days=np.array(counts.get("n_sst_days", 0)),
        n_ncep_days=np.array(counts.get("n_ncep_days", 0)),
        n_rg_live=np.array(counts.get("n_rg_live", len(live["bin_index"]))),
        sources=np.array(json.dumps(ctx.sources, sort_keys=True)),
        builder_git_sha=np.array(git_sha()),
        built_at=np.array(utcnow()),
    )
    for k in norms.files:
        if k.startswith("norm_"):
            meta[k] = norms[k]
    for k in truths.files:
        meta[k] = truths[k]

    out = os.path.join(work, STEM + ".npz")
    tmp = out + f".tmp{os.getpid()}.npz"
    np.savez(tmp, **meta)
    os.replace(tmp, out)
    mark(work, "meta")
    print(f"  meta: wrote {out} ({os.path.getsize(out) / 1e6:.1f} MB, "
          f"{len(meta)} keys)")
    return out


# =========================================================== stage: publish ==
def stage_publish(ctx):
    """Upload the four files, DOWNLOAD EACH BACK and compare sha256.

    An upload that returns 200 is not evidence the bytes are retrievable
    (ml/CLAUDE.md §0.2). `ml/hf_mirror.py`'s rule: a backup is only real if the
    restore works, so a publish that cannot verify FAILS the job.
    """
    from huggingface_hub import hf_hub_download
    work = ctx.work
    api, repo, tok = hub_repo()
    files = [os.path.join(work, STEM + ".npz")] + \
            [group_file(work, g) for g in GROUPS]
    for p in files:
        if not os.path.exists(p):
            sys.exit(f"cannot publish: {p} is missing")
    api.create_repo(repo, repo_type="dataset", exist_ok=True, private=False)
    ctx.prog.stage_start("publish", len(files))
    entries = []
    scratch = os.path.join(ctx.scratch, "verify")
    for i, p in enumerate(files, 1):
        name = os.path.basename(p)
        src = sha256(p)
        api.upload_file(path_or_fileobj=p, path_in_repo=f"{HF_PREFIX}/{name}",
                        repo_id=repo, repo_type="dataset",
                        commit_message=f"family 7 ({RECIPE}): {name}")
        shutil.rmtree(scratch, ignore_errors=True)
        back = hf_hub_download(repo, f"{HF_PREFIX}/{name}", repo_type="dataset",
                               token=tok, local_dir=scratch)
        got = sha256(back)
        shutil.rmtree(scratch, ignore_errors=True)
        if got != src:
            sys.exit(f"RESTORE MISMATCH {name}: uploaded {src}, downloaded "
                     f"{got} — the publish is not trustworthy")
        entries.append({"name": name, "bytes": os.path.getsize(p),
                        "sha256": src})
        ctx.prog.item(name, i, {"sha256": src[:16]})

    man = {"recipe": RECIPE, "stem": STEM, "groups": GROUPS,
           "builder_git_sha": git_sha(), "built_at": utcnow(),
           "repo": repo, "prefix": HF_PREFIX,
           "sources": ctx.sources, "files": entries}
    mp = os.path.join(work, "manifest.json")
    atomic_json(mp, man)
    api.upload_file(path_or_fileobj=mp,
                    path_in_repo=f"{HF_PREFIX}/manifest.json",
                    repo_id=repo, repo_type="dataset",
                    commit_message=f"family 7 ({RECIPE}): manifest")
    mark(work, "publish")
    print(f"  publish: {len(entries)} files verified by restore -> "
          f"https://huggingface.co/datasets/{repo}/tree/main/{HF_PREFIX}")


# ================================================================== driver ==
STAGE_FN = {"glorys": stage_glorys, "sst": stage_sst, "ncep": stage_ncep,
            "rg": stage_rg, "static": stage_static, "truth": stage_truth,
            "norm": stage_norm, "meta": stage_meta, "publish": stage_publish}


def run_stages(ctx, stages):
    work = ctx.work
    for s in stages:
        for dep in DEPS.get(s, []):
            if not marked(work, dep):
                sys.exit(f"stage {s!r} needs {dep!r} first (plan §4: stage "
                         f"order is fixed) — {marker(work, dep)} is missing")
        n_live = None
        lp = os.path.join(work, "rg", "live.npz")
        if os.path.exists(lp):
            n_live = len(np.load(lp)["bin_index"])
        stage_state_check(ctx, s, n_live)
        if marked(work, s) and not ctx.a.force:
            print(f"stage {s}: already done — skipping (--force to redo)")
            continue
        if s == "norm" and marked(work, "norm/g025"):
            # g100 and rg100 are idempotent — their norm reads the float32
            # intermediate and writes the float16 sidecar, so --force is safe.
            # g025 has no intermediate: its z-score is IN PLACE, and redoing it
            # over already-z-scored data would square the transform with
            # nothing downstream to say so (ml/CLAUDE.md §5.21).
            sys.exit("refusing --force on `norm`: g025 is already z-scored in "
                     "place (norm/g025.done), so redoing it would z-score "
                     "already-z-scored data. Delete the work dir to rebuild. "
                     "(g100 and rg100 alone are safe to redo once "
                     "norm/g025.done is what stops you — remove only their "
                     "own markers.)")
        t0 = time.time()
        print(f"\n=== stage {s} ===", flush=True)
        STAGE_FN[s](ctx)
        print(f"=== stage {s} done in {time.time() - t0:.1f}s ===", flush=True)


# =================================================================== smoke ==
def _nc_write(path, dims, variables, attrs=None):
    """Tiny netCDF writer for the synthetic sources."""
    import netCDF4 as ncdf
    os.makedirs(os.path.dirname(path), exist_ok=True)
    d = ncdf.Dataset(path, "w", format="NETCDF4")
    for k, v in dims.items():
        d.createDimension(k, v)
    for name, (dnames, arr, va) in variables.items():
        va = dict(va or {})
        fv = va.pop("_FillValue", None)
        v = d.createVariable(name, arr.dtype.str.replace("<", "").replace(">", ""),
                             dnames, fill_value=fv)
        for ak, av in va.items():
            setattr(v, ak, av)
        v[:] = arr
    for k, v in (attrs or {}).items():
        setattr(d, k, v)
    d.close()
    return path


def make_smoke_sources(root, d_lo, d_hi):
    """Tiny synthetic sources with the REAL axis conventions of each product.

    The point of the smoke is that every conversion in the builder runs against
    the shape and registration it will meet on the box — the 681-row GLORYS
    axis so the row-40 offset is exercised, cell-centred OISST so the half-cell
    interpolation is, a DESCENDING gaussian latitude with a `level` dimension
    so the squeeze and the flip are, and a cell-registered north-first ETOPO so
    the block mean is. Only the SIZES are small.
    """
    rng = np.random.default_rng(20260904)
    os.makedirs(root, exist_ok=True)
    days = [d_lo + dt.timedelta(days=k) for k in range((d_hi - d_lo).days + 1)]

    # ---- GLORYS: 681 x 1440, lat -80..90, lon -180..179.75 -----------------
    g_lat = -80.0 + 0.25 * np.arange(GLORYS_ROWS)
    g_lon = -180.0 + 0.25 * np.arange(NLON)
    land = (np.abs(g_lat)[:, None] > 60) & (np.abs(g_lon)[None, :] < 30)
    # only some days, so missing bins are exercised too
    gdays = [d for d in days if d.day <= 5 or d.day >= 26]
    by_month = {}
    for d in gdays:
        by_month.setdefault((d.year, d.month), []).append(d)
    for (y, m), ds in sorted(by_month.items()):
        n = len(ds)
        base = np.sin(np.radians(g_lat))[:, None] + \
            0.01 * np.cos(np.radians(g_lon))[None, :]
        def cube(scale, off=0.0):
            a = (off + scale * (base[None] + 0.01 * rng.standard_normal((n, 1, 1)))
                 ).astype(np.float32)
            a[:, land] = np.nan
            return a
        _nc_write(os.path.join(root, "daily025_global",
                               f"glorys025_global_{y}{m:02d}.nc"),
                  {"time": n, "latitude": GLORYS_ROWS, "longitude": NLON},
                  {"latitude": (("latitude",), g_lat, {"units": "degrees_north"}),
                   "longitude": (("longitude",), g_lon, {"units": "degrees_east"}),
                   "time": (("time",), np.array([(d - EPOCH).days for d in ds],
                                                np.float64),
                            {"units": f"days since {EPOCH} 00:00:00"}),
                   "uo": (("time", "latitude", "longitude"), cube(0.5), None),
                   "vo": (("time", "latitude", "longitude"), cube(0.3), None),
                   "mlotst": (("time", "latitude", "longitude"),
                              cube(20.0, 60.0), None),
                   "zos": (("time", "latitude", "longitude"), cube(0.2), None)})

    # ---- OISST: CELL-CENTRED, lat -89.875.., lon 0.125.. -------------------
    o_lat = -89.875 + 0.25 * np.arange(720)
    o_lon = 0.125 + 0.25 * np.arange(1440)
    o_land = (np.abs(o_lat)[:, None] > 70) & (o_lon[None, :] > 300)
    for y in sorted({d.year for d in days}):
        ds = [d for d in days if d.year == y]
        n = len(ds)
        f = (15.0 + 12.0 * np.cos(np.radians(o_lat))[:, None]
             + 0.0 * o_lon[None, :])
        sst = np.broadcast_to(f, (n, 720, 1440)).astype(np.float32).copy()
        sst += 0.1 * rng.standard_normal((n, 1, 1)).astype(np.float32)
        sst[:, o_land] = np.nan
        ice = np.clip((np.abs(o_lat)[:, None] - 70) / 20.0, 0, 1)
        ice = np.broadcast_to(ice, (n, 720, 1440)).astype(np.float32).copy()
        ice[:, o_land] = np.nan
        tvals = np.array([(d - EPOCH).days for d in ds], np.float64)
        for kind, arr in (("sst", sst), ("icec", ice)):
            _nc_write(os.path.join(root, "oisst", f"{kind}.day.mean.{y}.nc"),
                      {"time": n, "lat": 720, "lon": 1440},
                      {"lat": (("lat",), o_lat, {"units": "degrees_north"}),
                       "lon": (("lon",), o_lon, {"units": "degrees_east"}),
                       "time": (("time",), tvals,
                                {"units": f"days since {EPOCH} 00:00:00"}),
                       kind: (("time", "lat", "lon"), arr,
                              # icec's real attributes, misleading units and
                              # all: PSL says `units "percent"` over a
                              # `valid_range 0..1` fraction (see ice_divisor).
                              {"units": "degC", "valid_range":
                               np.array([-3.0, 45.0], np.float32)}
                              if kind == "sst" else
                              {"units": "percent", "valid_range":
                               np.array([0.0, 1.0], np.float32),
                               "missing_value": np.float32(-9.96921e36)})})

    # ---- NCEP: T62 gaussian, DESCENDING lat, a `level` dim ----------------
    n_lat, n_lon = 94, 192
    ng_lat = np.linspace(88.542, -88.542, n_lat)          # descending, as R1
    ng_lon = np.arange(n_lon) * 1.875                     # 0 .. 358.125
    landm = ((np.abs(ng_lat)[:, None] < 60)
             & (ng_lon[None, :] > 100) & (ng_lon[None, :] < 200)).astype(np.float32)
    _nc_write(os.path.join(root, "ncep", f"{NCEP_LAND}.nc"),
              {"time": 1, "lat": n_lat, "lon": n_lon},
              {"lat": (("lat",), ng_lat, {"units": "degrees_north"}),
               "lon": (("lon",), ng_lon, {"units": "degrees_east"}),
               "time": (("time",), np.zeros(1, np.float64),
                        {"units": f"days since {EPOCH} 00:00:00"}),
               "land": (("time", "lat", "lon"), landm[None], None)})
    for y in sorted({d.year for d in days}):
        ds = [d for d in days if d.year == y]
        n = len(ds)
        tvals = np.array([(d - EPOCH).days for d in ds], np.float64)
        shape = (n, n_lat, n_lon)
        prof = np.cos(np.radians(ng_lat))[None, :, None] \
            + 0.0 * ng_lon[None, None, :]
        for key, stem in NCEP_FILES.items():
            scale, off = {
                "uflx": (0.1, 0.0), "vflx": (0.1, 0.0),
                "air": (20.0, 273.15), "uwnd": (5.0, 0.0), "vwnd": (5.0, 0.0),
                "pres": (2000.0, 100000.0), "prate": (1e-5, 1e-5),
                "weasd": (10.0, 10.0), "soilw": (0.2, 0.3),
                "tmp": (15.0, 273.15), "lhtfl": (50.0, 0.0),
                "shtfl": (30.0, 0.0), "skt": (25.0, 273.15),
            }[key]
            a = (off + scale * (np.broadcast_to(prof, shape)
                                + 0.05 * rng.standard_normal((n, 1, 1)))
                 ).astype(np.float32)
            level = key in ("soilw", "tmp")
            if level:
                arr = a[:, None]
                dims = ("time", "level", "lat", "lon")
                dd = {"time": n, "level": 1, "lat": n_lat, "lon": n_lon}
            else:
                arr, dims = a, ("time", "lat", "lon")
                dd = {"time": n, "lat": n_lat, "lon": n_lon}
            vs = {"lat": (("lat",), ng_lat, {"units": "degrees_north"}),
                  "lon": (("lon",), ng_lon, {"units": "degrees_east"}),
                  "time": (("time",), tvals,
                           {"units": f"days since {EPOCH} 00:00:00"}),
                  key: (dims, arr, None)}
            if level:
                vs["level"] = (("level",), np.array([10.0]), None)
            _nc_write(os.path.join(root, "ncep", f"{stem}.{y}.nc"), dd, vs)

    # ---- RG: 1-degree CELL-CENTRED, lat -64.5..79.5, lon 20.5..379.5 ------
    r_lat = np.arange(-64.5, 80.0, 1.0)
    r_lon = np.arange(20.5, 380.0, 1.0)
    press = np.array([10., 30., 50., 100., 150., 200., 300., 400., 500., 700.,
                      900., 1100., 1300., 1500., 1700., 1900.])
    nl = len(press)
    mean_t = (20.0 - 0.01 * press[:, None, None]
              + 0.05 * r_lat[None, :, None] + 0.0 * r_lon[None, None, :]
              ).astype(np.float32)
    mean_s = (35.0 + 0.0 * mean_t).astype(np.float32)
    for stem, mname, aname, mu in (("RG_T", "ARGO_TEMPERATURE_MEAN",
                                    "ARGO_TEMPERATURE_ANOMALY", mean_t),
                                   ("RG_S", "ARGO_SALINITY_MEAN",
                                    "ARGO_SALINITY_ANOMALY", mean_s)):
        _nc_write(os.path.join(root, "rg", stem + ".nc"),
                  {"TIME": 1, "PRESSURE": nl, "LATITUDE": len(r_lat),
                   "LONGITUDE": len(r_lon)},
                  {"PRESSURE": (("PRESSURE",), press, None),
                   "LATITUDE": (("LATITUDE",), r_lat, None),
                   "LONGITUDE": (("LONGITUDE",), r_lon, None),
                   mname: (("PRESSURE", "LATITUDE", "LONGITUDE"), mu, None),
                   aname: (("TIME", "PRESSURE", "LATITUDE", "LONGITUDE"),
                           np.zeros((1,) + mu.shape, np.float32), None)})
    for d in days:
        if d.day != 15:
            continue
        ym = f"{d.year:04d}{d.month:02d}"
        _nc_write(os.path.join(root, "rg", f"RG_{ym}.nc"),
                  {"TIME": 1, "PRESSURE": nl, "LATITUDE": len(r_lat),
                   "LONGITUDE": len(r_lon)},
                  {"PRESSURE": (("PRESSURE",), press, None),
                   "LATITUDE": (("LATITUDE",), r_lat, None),
                   "LONGITUDE": (("LONGITUDE",), r_lon, None),
                   "ARGO_TEMPERATURE_ANOMALY":
                       (("TIME", "PRESSURE", "LATITUDE", "LONGITUDE"),
                        np.full((1,) + mean_t.shape, 0.5, np.float32), None),
                   "ARGO_SALINITY_ANOMALY":
                       (("TIME", "PRESSURE", "LATITUDE", "LONGITUDE"),
                        np.full((1,) + mean_s.shape, -0.1, np.float32), None)})

    # ---- Natural Earth: two toy polygons ----------------------------------
    def poly(w, s, e, n):
        return {"type": "Feature", "properties": {},
                "geometry": {"type": "Polygon",
                             "coordinates": [[[w, s], [e, s], [e, n], [w, n],
                                              [w, s]]]}}
    os.makedirs(os.path.join(root, "ne"), exist_ok=True)
    with open(os.path.join(root, "ne", "ne_10m_glaciated_areas.geojson"),
              "w") as fh:
        json.dump({"type": "FeatureCollection",
                   "features": [poly(-60, -85, 60, -70),      # an ice sheet
                                poly(-50, 70, -20, 82)]}, fh)  # Greenland-ish
    with open(os.path.join(root, "ne", "ne_10m_lakes.geojson"), "w") as fh:
        json.dump({"type": "FeatureCollection",
                   "features": [poly(-15, 72, -10, 76)]}, fh)

    # ---- ETOPO: cell-registered, NORTH-first, 1/12 degree -----------------
    step = 1.0 / 12.0
    e_lat = 90.0 - step * (np.arange(int(180 / step)) + 0.5)
    e_lon = -180.0 + step * (np.arange(int(360 / step)) + 0.5)
    z = (1000.0 * np.sin(np.radians(e_lat))[:, None]
         + 5.0 * np.cos(np.radians(e_lon))[None, :]).astype(np.float32)
    _nc_write(os.path.join(root, "etopo", "ETOPO_toy_surface.nc"),
              {"lat": len(e_lat), "lon": len(e_lon)},
              {"lat": (("lat",), e_lat, {"units": "degrees_north"}),
               "lon": (("lon",), e_lon, {"units": "degrees_east"}),
               "z": (("lat", "lon"), z, {"_FillValue": np.float32(-99999.0),
                                         "units": "meters"})},
              {"node_offset": 1})

    # ---- truth: the label contract, on this axis --------------------------
    b0 = bin_index(d_lo, PENTAD_DAYS)
    b1 = bin_index(d_hi, PENTAD_DAYS)
    bs = np.arange(b0, b1 + 1, dtype=np.float64)
    os.makedirs(os.path.join(root, "truth"), exist_ok=True)
    np.savez(os.path.join(root, "truth", "truth_pentad.npz"),
             epoch=np.array(str(EPOCH)), pentad_days=np.array(PENTAD_DAYS),
             truth_rapid=np.stack([bs, 17.0 + 0.1 * np.arange(len(bs))], 1),
             truth_fc=np.stack([bs, 31.0 + 0.0 * bs], 1))
    return root


SMOKE_START, SMOKE_END = "2010-01-13", "2010-02-02"


def run_smoke(root=None, keep=False, start=SMOKE_START, end=SMOKE_END):
    """Generate tiny synthetic sources and run every stage except publish."""
    tmp = root or tempfile.mkdtemp(prefix="f7smoke_")
    src = os.path.join(tmp, "src")
    work = os.path.join(tmp, "work")
    os.makedirs(work, exist_ok=True)
    t0 = time.time()
    print(f"smoke     sources -> {src}")
    d_lo = dt.date(*(int(x) for x in start.split("-")))
    d_hi = dt.date(*(int(x) for x in end.split("-")))
    make_smoke_sources(src, d_lo, d_hi)
    print(f"smoke     synthetic sources written in {time.time() - t0:.1f}s")

    ap = argparse.Namespace(work=work, source_dir=src, start=start, end=end,
                            force=False, stage="all", smoke=True)
    ctx = Ctx(ap)
    print(f"axis      {bin_start(ctx.bins[0], 5)} .. "
          f"{bin_start(ctx.bins[-1], 5)}  T={ctx.T} pentad bins "
          f"(bins {ctx.b_lo}..{ctx.b_hi}, recipe {RECIPE})")
    disk_guard(work, ctx.byte_peak(64), headroom=2e8)
    run_stages(ctx, [s for s in STAGES if s != "publish"])

    out = check_smoke(work, ctx)
    print(f"\nsmoke     OK in {time.time() - t0:.1f}s — {out}")
    if not keep and root is None:
        shutil.rmtree(tmp, ignore_errors=True)
    return work


REQUIRED_KEYS = ["bin_index", "months", "epoch", "pentad_days", "lats", "lons",
                 "lat1", "lon1", "chan_g025", "chan_g100", "chan_rg100",
                 "groups", "window", "recipe", "cadence", "sphere", "elev",
                 "norm_g025", "norm_g100", "norm_rg100", "rg_bin_index",
                 "n_glorys_bins", "n_sst_days", "n_ncep_days", "n_rg_live",
                 "sources", "builder_git_sha", "rapid", "truth_rapid"]


def check_smoke(work, ctx):
    """Every file and key of plan §2-§3 exists, with the right shapes."""
    from tensor_io import load_tensor
    npz = os.path.join(work, STEM + ".npz")
    for g in GROUPS:
        assert os.path.exists(group_file(work, g)), f"missing {g} memmap"
        assert not os.path.exists(raw_file(work, g)), (
            f"{raw_file(work, g)} survived — the float32 intermediate is "
            f"transient and must be deleted once norm has written the f16")
    assert os.path.exists(npz), "missing meta npz"
    d = load_tensor(npz)
    missing = [k for k in REQUIRED_KEYS if k not in d]
    assert not missing, f"meta npz is missing {missing}"
    n_live = int(np.asarray(d["n_rg_live"]))
    want = ctx.shapes(n_live)
    for g in GROUPS:
        got = tuple(d[f"X_{g}"].shape)
        assert got == want[g], f"{g} is {got}, want {want[g]}"
    assert d["X"].shape == d["X_g025"].shape, "X does not alias the dense group"
    assert [len(d[f"chan_{g}"]) for g in GROUPS] == \
        [NCHAN[g] for g in GROUPS]
    fin = int(np.isfinite(np.asarray(d["X_g025"][:, :, :, C_SST],
                                     np.float32)).sum())
    assert fin > 0, "sst is entirely missing — the sst stage wrote nothing"
    skt = np.asarray(d["X_g100"][:, :, :, C_SKT], np.float32)
    assert np.isfinite(skt).any(), "skt is entirely missing — ncep wrote nothing"
    assert np.isfinite(np.asarray(d["elev"])).any(), "elev is entirely NaN"
    codes = set(int(v) for v in np.unique(np.asarray(d["sphere"])))
    assert codes <= {0, 1, 2, 3}, codes
    assert codes == {0, 1, 2, 3}, (
        f"the toy polygons should produce all four sphere codes, got {codes}")
    return (f"T={want['g025'][0]} · n_live={n_live} · "
            f"{len(d.files)} keys · groups {list(d.groups)}")


# ==================================================================== main ==
def main():
    ap = argparse.ArgumentParser(
        description="Build the family-7 global 0.25-degree pentad tensor "
                    "(recipe f7l0). See ml/plans/E070_family7_build.md.")
    ap.add_argument("--work", default=os.path.join(CACHE, "family7"),
                    help="the build directory: memmaps, markers, progress.json")
    ap.add_argument("--stage", default="all",
                    choices=["all"] + STAGES,
                    help="one stage, or `all`. Order is fixed: ncep needs sst "
                         "for the sst repair, norm needs everything.")
    ap.add_argument("--source-dir", default="",
                    help="read every source from local files — no network at "
                         "all. Used by the tests and by --smoke.")
    ap.add_argument("--smoke", action="store_true",
                    help="generate tiny synthetic sources in a temp dir and "
                         "run every stage except publish, in seconds")
    ap.add_argument("--smoke-dir", default="",
                    help="with --smoke: keep the temp tree here")
    ap.add_argument("--start", default=str(START))
    ap.add_argument("--end", default=str(END))
    ap.add_argument("--force", action="store_true",
                    help="redo a stage whose .done marker exists")
    ap.add_argument("--dry-run", action="store_true",
                    help="print the axis and the byte arithmetic, spend nothing")
    a = ap.parse_args()

    if a.smoke:
        run_smoke(root=a.smoke_dir or None, keep=bool(a.smoke_dir))
        return 0

    ctx = Ctx(a)
    print(f"axis      {bin_start(ctx.bins[0], 5)} .. "
          f"{bin_start(ctx.bins[-1], 5)}  T={ctx.T} pentad bins "
          f"(bins {ctx.b_lo}..{ctx.b_hi}, recipe {RECIPE})")
    print(f"grids     g025 {NLAT}x{NLON} · g100 {NLAT1}x{NLON1} · "
          f"rg100 {NLAT1}x{NLON1}")
    print(f"channels  g025 {NCHAN['g025']} · g100 {NCHAN['g100']} · "
          f"rg100 {NCHAN['rg100']}")
    live_path = os.path.join(ctx.work, "rg", "live.npz")
    n_live = len(np.load(live_path)["bin_index"]) if os.path.exists(live_path) \
        else 252                      # 2004-01..2024-12, the upper bound
    sizes = ctx.byte_peak(n_live)
    if a.dry_run:
        for k, v in ctx.shapes(n_live).items():
            print(f"  {k:<6} {v} float16 = {int(np.prod(v)) * 2 / 1e9:.1f} GB"
                  + ("  (+ a transient float32 intermediate)"
                     if k in RAW_F32 else ""))
        st = os.statvfs(ctx.work)
        print(f"  total  {sum(sizes.values()) / 1e9:.1f} GB · "
              f"{st.f_bavail * st.f_frsize / 1e9:.1f} GB free")
        print("\n--dry-run: nothing built.")
        return 0

    stages = STAGES if a.stage == "all" else [a.stage]
    if any(s in ("glorys", "sst", "ncep", "rg") for s in stages):
        disk_guard(ctx.work, sizes)
    run_stages(ctx, stages)
    return 0


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
"""Family-7 tensor: the WHOLE GLOBE at 0.25 degrees, pentad cadence — recipe `f7l1`.

E-070 §Phases B-D and E-071 §6.5 (Phase L0), built once as one tensor, plus
E-077's fourth group. The executable specifications are
`ml/plans/E070_family7_build.md` (the three original groups) and
`ml/plans/E077_family7_ocean_colour.md` (the ocean-colour group `oc025` and the
inherit-don't-rebuild mechanics); this file is those documents made runnable,
and every decision below is theirs, not a new one.

TWO RECIPES LIVE IN THIS FILE ON PURPOSE. `BASE_RECIPE = "f7l0"` is what the
SIX INHERITED stages (`glorys`, `sst`, `ncep`, `rg`, `static`, `truth`) fold
into their spec digest, so a work directory seeded from a finished f7l0 build
keeps its `.spec` files valid and nothing is discarded; `RECIPE = "f7l1"` is
what the FOUR new-or-changed stages (`occci`, `norm`, `meta`, `publish`) use,
because what they write did change. Without the split, `--seed-from` would
hand every inherited stage a digest that no longer matches its recorded one
and the builder would helpfully rebuild 53 GB it already has.

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

FOUR GROUPS AT THEIR NATIVE RESOLUTION, not one dense array (plan B2):

  g025   [3142, 721, 1440,  7] float16  46 GB   0.25 deg, GLORYS12 + OISST
  g100   [3142, 181,  360, 15] float16  6.1 GB  1 deg, NCEP/NCAR R1
  rg100  [n_live, 181, 360, 32] float16 ~1 GB   1 deg, Roemmich-Gilson, live bins
  oc025  [1997, 721, 1440,  2] float16  8.3 GB  0.25 deg, ESA OC-CCI colour

`oc025` is E-077's addition: surface chlorophyll-a from the merged
SeaWiFS/MERIS/MODIS/VIIRS/OLCI record, box-averaged from 4 km to this grid in
LOG space, plus the clear-sky coverage the average rests on. Its first row is
the pentad holding 1997-09-04 — the first day the archive has — so its time
axis is OFFSET rather than padded with 4.7 GB of NaN: row = bin -
`oc_bin_first`, and `oc_bin_first` is in the npz (E-077 §4 layout 1; every
consumer already carried a per-group row lookup, so the generalisation was
three lines of arithmetic and not a new mechanism).

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
  python3 ml/build_family7.py --oc-preflight                # one real OC-CCI file
  python3 ml/build_family7.py --work /opt/earth-cache/f7l1 \
      --seed-from /opt/earth-cache/family7 --stage all      # inherit, then colour
  python3 ml/build_family7.py --work ... --stage occci      # one stage
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
# THE RECIPE SPLIT (E-077 §5). `RECIPE` is what this build IS; `BASE_RECIPE` is
# what the stages it INHERITS were, and the two must both be written down or a
# seeded work dir cannot be told apart from a stale one.
RECIPE = "f7l1"
BASE_RECIPE = "f7l0"
STEM = "family7_global025_pentad_l1"
BASE_STEM = "family7_global025_pentad_l0"
HF_DATASET = "earth-tensors"
HF_PREFIX = f"tensors/{STEM}"
BASE_PREFIX = f"tensors/{BASE_STEM}"
# The stages whose bytes come from the f7l0 build unchanged. Their spec digest
# is folded with BASE_RECIPE (stage_spec below), which is the whole reason a
# copied `.spec` file still matches and `stage_state_check` does not throw away
# 53 GB it could have kept.
INHERITED_STAGES = ("glorys", "sst", "ncep", "rg", "static", "truth")

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
# E-077 §3. TWO channels, and the second one is not decoration: `log_chl` is a
# mean over whatever the satellite could see through the cloud, and `chl_cov`
# is how much that was — one clear pixel on one day, or thirty-six on five.
# Without it a reader cannot tell a confident open-ocean value from a single
# glimpse through a hole in a storm, and both look identical in the array.
CHAN_OC025 = ["log_chl", "chl_cov"]
BASE_GROUPS = ["g025", "g100", "rg100"]
GROUPS = BASE_GROUPS + ["oc025"]
NCHAN = {"g025": len(CHAN_G025), "g100": len(CHAN_G100),
         "rg100": len(CHAN_RG100), "oc025": len(CHAN_OC025)}

C_CUR_SPEED, C_LOG_MLD, C_SSH, C_CUR_U, C_CUR_V, C_SST, C_SEA_ICE = range(7)
C_SKT = 14                            # g100's shared surface temperature
C_LOG_CHL, C_CHL_COV = 0, 1           # oc025

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

# ---- ocean colour: ESA OC-CCI v6.0 daily chlorophyll-a (E-077 §2) ----------
# THE FIRST DAY OF THE RECORD IS DATA, NOT A CONSTANT TO TYPE. `oc_bin_first`
# is computed from this date through the same `bin_index` every other family
# uses, so the offset axis cannot drift from the calendar (E-077 §7.2).
OC_START = dt.date(1997, 9, 4)            # SeaWiFS's first day in the merge
OC_VAR = "chlor_a"                        # mg m^-3, float32 with a fill value
# `chlor_a_log10_rmsd` is DELIBERATELY NOT STORED: the uncertainty of a merged
# algorithm is a property of the algorithm, not an observation of the ocean.
# Revisit when colour becomes an input rather than a target (E-077 §2).
OC_SKIP_VARS = ("chlor_a_log10_rmsd", "chlor_a_log10_bias",
                "MERIS_nobs_sum", "MODISA_nobs_sum", "SeaWiFS_nobs_sum",
                "VIIRS_nobs_sum", "OLCI_nobs_sum", "total_nobs_sum")
# The pentad bin the handover and E-070 §5 already use as the comparison frame
# (2015-01-03). Computed, never typed, for the same reason as OC_START.
OC_REPORT_DAY = dt.date(2015, 1, 3)

# THE ARCHIVE IS ASKED, NOT GUESSED (root CLAUDE.md: never guess what an
# archive serves). Each host below offers a CATALOG url per year; the `index`
# step fetches it, parses the file names OUT OF THE SERVER'S OWN ANSWER, and
# records them in `<work>/occci/index.json`. The `file` template is used only
# for the hosts whose listing does not carry a usable path of its own — for
# THREDDS the catalog's `urlPath` attribute is authoritative and the template
# is the fallback.
#
# MEASURED 2026-09-11 from this sandbox: both hosts answer `CONNECT tunnel
# failed, 403` at the egress proxy, so NONE of these URL shapes could be
# verified here. That is exactly why `--oc-preflight` exists and why the
# workflow runs it before the build: one real file, fetched and opened, on the
# box that can reach them, before four hundred gigabytes are spent.
OC_SOURCES = {
    "occci": [
        {"name": "pml-thredds",
         "host": "www.oceancolour.org",
         "catalog": "https://www.oceancolour.org/thredds/catalog/cci/"
                    "v6.0-release/geographic/netcdf/chlor_a/daily/v6.0/"
                    "{year}/catalog.xml",
         "file": "https://www.oceancolour.org/thredds/fileServer/cci/"
                 "v6.0-release/geographic/netcdf/chlor_a/daily/v6.0/"
                 "{year}/{name}",
         "server": "https://www.oceancolour.org/thredds/fileServer/",
         "kind": "thredds"},
        {"name": "pml-thredds-allproducts",
         "host": "www.oceancolour.org",
         "catalog": "https://www.oceancolour.org/thredds/catalog/"
                    "CCI_ALL-v6.0-DAILY/{year}/catalog.xml",
         "file": "https://www.oceancolour.org/thredds/fileServer/"
                 "CCI_ALL-v6.0-DAILY/{year}/{name}",
         "server": "https://www.oceancolour.org/thredds/fileServer/",
         "kind": "thredds"},
        {"name": "ceda",
         "host": "dap.ceda.ac.uk",
         # CEDA's copy of v6.0 ends 2022-12-31; a year it does not carry simply
         # lists nothing and the next host is tried.
         "catalog": "https://dap.ceda.ac.uk/neodc/esacci/ocean_colour/data/"
                    "v6.0-release/geographic/netcdf/chlor_a/daily/v6.0/"
                    "{year}/?json",
         "file": "https://dap.ceda.ac.uk/neodc/esacci/ocean_colour/data/"
                 "v6.0-release/geographic/netcdf/chlor_a/daily/v6.0/"
                 "{year}/{name}",
         "kind": "json"},
    ],
    # THE FALLBACK IS A DECISION, NOT AN AUTOMATIC RETRY (E-077 §2). GlobColour
    # is a DIFFERENT merge — not the CCI bias-corrected chain — so a build that
    # slid into it silently would publish a channel whose provenance nobody
    # chose. It is reachable only by `--oc-source globcolour`, and it needs the
    # CMEMS credentials named below in the environment.
    "globcolour": [
        {"name": "cmems-globcolour",
         "host": "my.cmems-du.eu",
         "product": "cmems_obs-oc_glo_bgc-plankton_my_l3-multi-4km_P1D",
         "var": "CHL",
         "credentials": ("COPERNICUSMARINE_SERVICE_USERNAME",
                         "COPERNICUSMARINE_SERVICE_PASSWORD"),
         "catalog": "https://my.cmems-du.eu/thredds/catalog/"
                    "cmems_obs-oc_glo_bgc-plankton_my_l3-multi-4km_P1D/"
                    "{year}/catalog.xml",
         "file": "https://my.cmems-du.eu/thredds/fileServer/"
                 "cmems_obs-oc_glo_bgc-plankton_my_l3-multi-4km_P1D/"
                 "{year}/{name}",
         "server": "https://my.cmems-du.eu/thredds/fileServer/",
         "kind": "thredds"},
    ],
}

STAGES = ["glorys", "sst", "ncep", "rg", "occci", "static", "truth", "norm",
          "meta", "publish"]
DEPS = {
    # `ncep` no longer writes into g025 at all, but it still runs the `sst`
    # repair below, which needs `oisst_seen.npy` from the sst stage.
    "ncep": ["sst"],
    "norm": ["glorys", "sst", "ncep", "rg", "occci"],
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
#
# `oc025` joins them (E-077 §5). It is 16.6 GB at float32 rather than 2, which
# is the one place this rule costs real disk — but the reason is the same and
# sharper: `chl_cov` is a fraction in [0, 1] whose interesting values are the
# SMALL ones (1/180 is a single clear pixel on a single day), and `log_chl`
# near 0.5 would be quantised at 0.0005 by a raw float16 write. Both survive
# the z-scored write intact, and the float32 file is deleted the moment `norm`
# has written the float16 beside it.
RAW_F32 = ("g100", "rg100", "oc025")


def group_file(work, group):
    """The FINAL, published float16 sidecar for a group."""
    return os.path.join(work, f"{STEM}_X_{group}.npy")


# ---- THE HARD LINK IS SHARED BYTES, NOT A COPY (E-077 §5) -----------------
#
# `--seed-from` gives the three inherited groups ONE INODE under two names, in
# two directories. That is the whole point — 53 GB for free — and it is also
# the one way this build could destroy the published f7l0 tensor: a single
# write through the f7l1 name lands in the f7l0 file, whose sha256 the Hub,
# `docs/FAMILY7_DATA_HANDOVER.md` and another agent's hand-off all cite. There
# would be no symptom until someone re-verified a hash.
#
# So it is made IMPOSSIBLE rather than merely unintended, in two layers:
#
#   1. `seed_from` chmods each linked file to 0444. Both names go read-only,
#      which is correct — the seed build is finished and nothing may write
#      either copy again.
#   2. Every writable open of a group file asks this function first and EXITS
#      by name. That layer is the load-bearing one, because these builds run
#      as root on the Vast boxes and root ignores the mode bits (measured).
#      A guard that only works for an unprivileged user is not a guard here.
def seeded_groups(work):
    """The groups in this work dir that are hard links into a seed, by name."""
    rec = read_json(os.path.join(work, "seed.json"), {})
    return {str(e.get("group")) for e in rec.get("linked", [])}


def refuse_if_seeded(work, group, what):
    """Stop, naming the seed, before anything writes an inherited group."""
    if group not in seeded_groups(work):
        return
    rec = read_json(os.path.join(work, "seed.json"), {})
    sys.exit(
        f"REFUSING to {what} {group!r}: {group_file(work, group)} is a HARD "
        f"LINK to {rec.get('from', '<the seed>')}/"
        f"{BASE_STEM}_X_{group}.npy — one inode under two names. Writing "
        f"through it would silently rewrite the published {BASE_RECIPE} "
        f"tensor, whose sha256 the Hub and docs/FAMILY7_DATA_HANDOVER.md "
        f"both quote. An inherited group is READ-ONLY in this build; if it "
        f"really has to be rebuilt, use a fresh --work with no --seed-from.")


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

    A group that arrived through `--seed-from` is never opened writable here:
    for `g025` this path IS the linked inode, and for the coarse groups it
    would be the float32 intermediate of a group that is already final.
    """
    refuse_if_seeded(work, group, "fill")
    dtype = np.float32 if group in RAW_F32 else np.float16
    return _open_memmap(fill_file(work, group), shape, dtype, create)


def open_final(work, group, shape, create=False):
    """The published float16 sidecar. Never for an inherited group."""
    refuse_if_seeded(work, group, "write the final float16 of")
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
        # THE COLOUR AXIS IS OFFSET, NOT PADDED (E-077 §4, layout 1). The first
        # colour row is the bin holding the archive's first day, computed here
        # from the date rather than typed, and clamped onto whatever sub-range
        # this build covers. `--oc-start` exists ONLY so the smoke can put the
        # offset somewhere inside a three-week axis and prove the arithmetic;
        # moving it on a real build would silently mislabel every colour row.
        oc0 = getattr(a, "oc_start", None) or str(OC_START)
        y, m, dd = (int(x) for x in str(oc0).split("-"))
        self.oc_day0 = dt.date(y, m, dd)
        self.b_oc = max(b_lo, bin_index(self.oc_day0, PENTAD_DAYS))
        self.T_oc = max(0, b_hi - self.b_oc + 1)
        self.oc_source = getattr(a, "oc_source", None) or "occci"
        self.lats, self.lons = grid025()
        self.lat1, self.lon1 = grid100()
        self.prog = Progress(self.work)
        self.sources = read_json(os.path.join(self.work, "sources.json"), {})

    # -- axis helpers ------------------------------------------------------
    def row_of(self, b):
        return b - self.b_lo if self.b_lo <= b <= self.b_hi else None

    def oc_row_of(self, b):
        """The `oc025` ROW of bin `b`, or None if colour has no row for it.

        The whole of E-077 §4's layout 1 is this one line: the group's first
        row is `b_oc`, not `b_lo`, and every consumer that wants colour at a
        bin asks here (or reads `oc_bin_first` out of the npz) instead of
        assuming the groups share an origin.
        """
        return b - self.b_oc if self.b_oc <= b <= self.b_hi else None

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
             "g100": (self.T, NLAT1, NLON1, NCHAN["g100"]),
             "oc025": (self.T_oc, NLAT, NLON, NCHAN["oc025"])}
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
                "rg100 f32": n["rg100"] * 4,
                # oc025 is the biggest transient in the build: 16.6 GB of
                # float32 with its 8.3 GB float16 written beside it before the
                # f32 is dropped. Both are counted, because for a few minutes
                # both exist (E-077 §6).
                "oc025 f32": n["oc025"] * 4, "oc025 f16": n["oc025"] * 2,
                # One OC-CCI daily file in flight (~60 MB), doubled so a
                # retry's partial download has somewhere to land. Sources are
                # deleted as they are read, so this never grows with the year.
                "occci sources": int(2e8)}
        # A BYTE ALREADY ON THE DISK IS NOT A BYTE THIS RUN MUST FIND
        # (ml/CLAUDE.md §5.18: size a guard from the allocation it guards).
        # Without this, a `--seed-from` build asks for 46 GB it hard-linked a
        # second ago and a resumed build asks for everything it already wrote,
        # so the guard refuses exactly the runs it was written to protect.
        where = {"g025 f16": group_file(self.work, "g025"),
                 "g100 f32": raw_file(self.work, "g100"),
                 "g100 f16": group_file(self.work, "g100"),
                 "rg100 f32": raw_file(self.work, "rg100"),
                 "oc025 f32": raw_file(self.work, "oc025"),
                 "oc025 f16": group_file(self.work, "oc025")}
        for k, p in where.items():
            if k in peak and os.path.exists(p) and os.path.getsize(p) >= peak[k]:
                peak.pop(k)
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
SPEC_VERSION = {"glorys": 1, "sst": 1, "ncep": 2, "rg": 1, "occci": 1,
                "static": 1, "truth": 1, "norm": 1, "meta": 1, "publish": 1}

# Which array a stage OWNS — the one it may delete when its spec moves. A
# stage never touches another stage's files: glorys and sst share g025 and own
# nothing, so a spec change there rewrites their own channels in place.
STAGE_OWNS = {"ncep": "g100", "rg": "rg100", "occci": "oc025"}

# What each stage writes, for the digest. Deliberately its OWN channels, not
# the whole tensor: renaming a g100 channel must not send the OISST stage back
# to 1982.
STAGE_CHANNELS = {
    "glorys": CHAN_G025[:5],
    "sst": CHAN_G025[5:],
    "ncep": CHAN_G100,
    "rg": CHAN_RG100,
    "occci": CHAN_OC025,
}


def stage_recipe(stage):
    """Which recipe a stage's digest is folded with (E-077 §5).

    The inherited six answer `f7l0` FOREVER, because their bytes are f7l0's
    bytes: a `.spec` file copied out of the f7l0 work directory has to keep
    matching, or `stage_state_check` throws away 53 GB that is not stale, it is
    just older than this build. Everything the colour build actually changes —
    `occci`, `norm`, `meta`, `publish` — answers `f7l1`, so a work dir seeded
    from f7l0 correctly regards THOSE as new work.
    """
    return BASE_RECIPE if stage in INHERITED_STAGES else RECIPE


def stage_spec(ctx, stage, n_live=None):
    body = {
        "stage": stage, "version": SPEC_VERSION[stage],
        "recipe": stage_recipe(stage),
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
    if stage == "occci":
        # The offset origin is part of the recipe: a build that moved
        # `oc_bin_first` wrote every colour row at a different date, and
        # nothing in the shapes or the channel names would say so.
        body["oc_bin_first"] = int(ctx.b_oc)
        body["oc_start"] = str(ctx.oc_day0)
        body["oc_source"] = str(ctx.oc_source)
        body["var"] = OC_VAR
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
    old, old_body = None, None
    if os.path.exists(spec_path):
        rec = read_json(spec_path, {})
        old, old_body = rec.get("sha256"), rec.get("spec")
    why = []
    if old is not None and old != digest:
        # A SPEC WRITTEN WITH LESS INFORMATION IS NOT A CHANGED RECIPE.
        # `rg`'s shape carries `n_live`, the number of live Argo months, and
        # that number does not exist until `rg` itself has written
        # `rg/live.npz` — so on a FRESH build the stage records `shapes: {}`
        # and the very next run computes a different digest for an identical
        # recipe. Left alone that discards rg's markers AND its `live.npz` on
        # every resume, which for a work dir seeded from f7l0 (E-077 §5) would
        # throw away inherited state the build is not even asking to rebuild.
        # So an empty recorded `shapes` beside an otherwise identical body is
        # upgraded in place rather than treated as staleness.
        thin = (isinstance(old_body, dict) and old_body.get("shapes") == {}
                and {k: v for k, v in old_body.items() if k != "shapes"}
                == {k: v for k, v in body.items() if k != "shapes"})
        if thin:
            print(f"  stage {stage!r}: its recorded spec predates the "
                  f"{'/'.join(body['shapes'])} shape (written before it "
                  f"existed) and is otherwise identical — upgrading the spec, "
                  f"keeping the state")
        else:
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
        # A STALE INHERITED STAGE MUST NOT BE REPLAYED OVER A HARD LINK.
        # Discarding markers is harmless; what follows a discard is not — the
        # stage re-runs and writes its group, and for a seeded build that
        # group's bytes are the published f7l0 tensor's. `glorys` and `sst`
        # share g025 and own no file of their own, which is exactly why the
        # `owned` lookup below cannot catch them; both are named here.
        touched = {STAGE_OWNS.get(stage)} | (
            {"g025"} if stage in ("glorys", "sst") else set())
        for g in sorted(x for x in touched if x):
            if g in seeded_groups(work):
                rec = read_json(os.path.join(work, "seed.json"), {})
                sys.exit(
                    f"stage {stage!r} is stale ({'; '.join(why)}) and it "
                    f"writes {g!r}, which this build INHERITED as a hard link "
                    f"to {rec.get('from', '<the seed>')}. Replaying it would "
                    f"rewrite the published {BASE_RECIPE} bytes through the "
                    f"shared inode. Nothing has been deleted. Either fix the "
                    f"spec drift (the seed's {stage}.spec should match this "
                    f"builder's digest for {BASE_RECIPE} — see stage_recipe) "
                    f"or rebuild in a fresh --work with no --seed-from.")
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


# ============================================================= stage: occci ==
# THE BLOCK MEAN, WRITTEN AS PURE ARITHMETIC SO IT CAN BE TESTED WITHOUT A FILE.
#
# The OC-CCI grid is CELL-registered: 4320 rows of 1/24 degree descending from
# the North Pole (centres at 90 - (i + 0.5)/24) and 8640 columns ascending from
# -180 (centres at -180 + (j + 0.5)/24). Our grid points sit on exact multiples
# of 0.25 degrees, which on a 1/24-degree grid is a CELL BOUNDARY — 0.25 = 6/24
# — so the 0.25-degree box around a point contains exactly six source cells per
# axis, three on each side, with no cell split between two points. That is the
# whole reason the plan says 6 x 6 and not "about 36": it is exact.
#
# Two special cases, and they are the only two:
#
#   * LONGITUDE WRAPS. The block for x = 0 (lon -180) is columns -3..2, i.e.
#     8637, 8638, 8639, 0, 1, 2. Half the block is on the other side of the
#     dateline, which is the same meridian.
#   * THE POLE ROWS TRUNCATE. The point at -90 (or +90) is ON the edge of the
#     source grid, so three of its six rows are off the raster. The block mean
#     is taken over the three that exist; `chl_cov`'s denominator counts the
#     cells that exist, not the six that would if the planet were taller.
#
# NOTHING IS HARD-CODED TO 4320 x 8640. The factor is derived from the file's
# own axis spacing, asserted to divide 0.25 exactly, so a decimated grid (which
# is what the test uses, because a real daily file is 30-60 MB and the sandbox
# cannot reach the archive) runs the identical code path.
def oc_block_factors(s_lat, s_lon):
    """(blk_y, blk_x) source cells per 0.25-degree point, from the axes.

    REFUSES rather than rounds: a source whose spacing does not divide 0.25 an
    integral number of times cannot be block-averaged onto this grid without
    splitting cells between neighbouring points, and a silent `round()` there
    would put a fraction of every coastal pixel in the wrong 0.25 box.
    """
    s_lat = np.asarray(s_lat, np.float64)
    s_lon = np.asarray(s_lon, np.float64)
    if len(s_lat) < 2 or len(s_lon) < 2:
        sys.exit("OC-CCI: an axis of fewer than two points has no spacing")
    dlat, dlon = abs(float(s_lat[1] - s_lat[0])), abs(float(s_lon[1] - s_lon[0]))
    out = []
    for d, n, want, what in ((dlat, len(s_lat), 180.0, "latitude"),
                             (dlon, len(s_lon), 360.0, "longitude")):
        f = 0.25 / d
        k = int(round(f))
        if k < 1 or abs(f - k) > 1e-6:
            sys.exit(f"OC-CCI {what} spacing {d:.8f} deg does not divide 0.25 "
                     f"an integral number of times ({f:.4f}) — the 0.25 box "
                     f"would split source cells between grid points")
        if abs(n * d - want) > 1e-3:
            sys.exit(f"OC-CCI {what} axis is {n} x {d:.8f} = {n * d:.4f} deg, "
                     f"not the whole {want:.0f} — this is not a global grid "
                     f"and the block mean assumes one")
        out.append(k)
    return out[0], out[1]


def oc_block_cells(blk_y, blk_x, nlat_s):
    """How many source cells each 0.25-degree block actually holds, [NLAT].

    `blk_y * blk_x` everywhere except the two pole rows, where the block runs
    off the raster and only `blk_y - blk_y // 2` (or `blk_y // 2 + ...`) rows
    survive. This is `chl_cov`'s denominator, per row: dividing a truncated
    pole block by the full 36 would report half the coverage it has.
    """
    pad_lo = blk_y // 2
    # In the south-first, padded index space the rows of point y are
    # [blk_y*y, blk_y*y + blk_y); real source rows occupy [pad_lo, pad_lo+nlat_s).
    start = blk_y * np.arange(NLAT, dtype=np.int64)
    lo = np.maximum(start, pad_lo)
    hi = np.minimum(start + blk_y, pad_lo + nlat_s)
    return (np.maximum(hi - lo, 0) * blk_x).astype(np.int64)


def oc_block_slices(y, x, blk_y, blk_x, nlat_s, nlon_s):
    """The SOURCE (rows, cols) of the block owned by 0.25-degree point (y, x).

    The same mapping `oc_block_stats` vectorises, written out one point at a
    time: rows in the source's own NORTH-FIRST order (truncated at the poles),
    columns wrapped at the dateline. The fast path exists because 36 million
    cells a day cannot be indexed in Python; this exists because a reader — and
    the synthetic source the smoke plants exact values into — needs to name
    the individual cells of one block.
    """
    rows = []
    for k in range(blk_y):
        r_sf = blk_y * int(y) - blk_y // 2 + k        # south-first row
        if 0 <= r_sf < nlat_s:
            rows.append(nlat_s - 1 - r_sf)            # back to north-first
    cols = [(blk_x * int(x) - blk_x // 2 + k) % nlon_s for k in range(blk_x)]
    return rows, cols


def oc_block_stats(chl, blk_y, blk_x, fill=None):
    """One daily 4 km `chlor_a` field -> (sum of log10, count) per 0.25 point.

    Returns `(S, C)` with `S` [721, 1440] float64 — the sum of `log10(chl)`
    over the FINITE cells of each block — and `C` [721, 1440] int64, how many
    those were. The caller divides.

    LOG SPACE, NOT LINEAR (E-077 §3 and E-072 §3). Chlorophyll is log-normal
    over four orders of magnitude; the arithmetic mean of a block containing one
    bloom pixel at 30 mg/m3 and thirty-five at 0.05 is a number that describes
    neither. The mean is taken on log10 and never on the concentration.

    A non-positive or sentinel value is NOT a small concentration — log10 of it
    is -inf or a NaN that would poison the block — so it is dropped the same way
    a fill value is, and `chl_cov` reports the loss.
    """
    a = np.squeeze(np.asarray(chl))
    if a.ndim != 2:
        sys.exit(f"OC-CCI: a daily field is {a.shape}, expected one 2-D grid")
    nlat_s, nlon_s = a.shape
    # float32 THROUGHOUT, float64 only in the sums. The source is float32; a
    # float64 working copy of a 4320 x 8640 raster is 300 MB per temporary and
    # this runs ten thousand times. log10 in float32 carries seven digits,
    # which is four more than the float16 the value is eventually stored in.
    a = np.asarray(a, np.float32)
    ok = np.isfinite(a) & (a > np.float32(0.0)) & (np.abs(a) < np.float32(1e30))
    if fill is not None and np.isfinite(float(fill)):
        ok &= ~np.isclose(a, np.float32(fill))
    # log10(1.0) is 0, so the cells that are NOT observed fall out of both sums
    # without a second `where` and without ever evaluating log10 of a fill.
    lg = np.where(ok, a, np.float32(1.0))
    with np.errstate(invalid="ignore", divide="ignore"):
        np.log10(lg, out=lg)

    # --- longitude: the blocks tile the axis with a -blk_x//2 offset, so one
    # roll turns them into a plain reshape. `np.roll` is the wrap.
    off_x = blk_x // 2
    cs = np.roll(lg, off_x, axis=1).reshape(nlat_s, NLON, blk_x).sum(
        axis=2, dtype=np.float64)
    cc = np.roll(ok, off_x, axis=1).reshape(nlat_s, NLON, blk_x).sum(
        axis=2, dtype=np.int64)

    # --- latitude: reverse to SOUTH-FIRST (the tensor's own order), then pad
    # with empty rows at both ends so the truncated pole blocks are a reshape
    # too. Padding with zeros is correct because the count is padded with zeros
    # as well: a row that does not exist contributes nothing to either.
    cs = cs[::-1]
    cc = cc[::-1]
    pad_lo = blk_y // 2
    pad_hi = blk_y * NLAT - nlat_s - pad_lo
    if pad_hi < 0:
        sys.exit(f"OC-CCI: {nlat_s} source rows do not fit {NLAT} blocks of "
                 f"{blk_y} — the latitude axis is not the one this grid needs")
    cs = np.concatenate([np.zeros((pad_lo, NLON)), cs,
                         np.zeros((pad_hi, NLON))], axis=0)
    cc = np.concatenate([np.zeros((pad_lo, NLON), np.int64), cc,
                         np.zeros((pad_hi, NLON), np.int64)], axis=0)
    S = cs.reshape(NLAT, blk_y, NLON).sum(axis=1, dtype=np.float64)
    C = cc.reshape(NLAT, blk_y, NLON).sum(axis=1, dtype=np.int64)
    return S, C


# ---------------------------------------------------------- source indexing --
def _http_get(url, timeout=180):
    """One GET, as bytes. Raises — the caller decides what a failure means."""
    import urllib.request
    req = urllib.request.Request(
        url, headers={"User-Agent": "earth-science-pipeline/1.0 "
                                    "(research; github blauewelt/earth)",
                      "Accept": "*/*"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read()


def parse_thredds_catalog(blob, server=None):
    """{file name: url or None} out of a THREDDS `catalog.xml`.

    The server's OWN `urlPath` is preferred over any template we could write,
    because that is the thing the catalogue is FOR. A dataset element without
    one falls back to the caller's template (value None here).
    """
    import xml.etree.ElementTree as ET
    out = {}
    root = ET.fromstring(blob)
    for el in root.iter():
        if not el.tag.endswith("}dataset") and el.tag != "dataset":
            continue
        name = el.get("name") or ""
        if not name.endswith(".nc"):
            continue
        up = el.get("urlPath")
        out[name] = (server.rstrip("/") + "/" + up.lstrip("/")
                     if (up and server) else None)
    return out


def parse_json_listing(blob):
    """{file name: None} out of a CEDA `?json` directory listing.

    CEDA has served several shapes of this over the years (a list of records, a
    dict keyed by name, a dict with an `items` list), so every one of them is
    accepted and anything that is not a `.nc` name is ignored. A listing we
    cannot parse returns EMPTY and the next host is tried — it never returns a
    guess.
    """
    try:
        doc = json.loads(blob.decode("utf-8", "replace"))
    except Exception:                                         # noqa: BLE001
        return {}
    names = []

    def walk(o):
        if isinstance(o, str):
            if o.endswith(".nc"):
                names.append(o.rsplit("/", 1)[-1])
        elif isinstance(o, list):
            for v in o:
                walk(v)
        elif isinstance(o, dict):
            for k, v in o.items():
                if isinstance(k, str) and k.endswith(".nc"):
                    names.append(k.rsplit("/", 1)[-1])
                if k in ("name", "path", "href", "filename", "items",
                         "children", "files", "listing", "contents"):
                    walk(v)
    walk(doc)
    return {n: None for n in sorted(set(names))}


def parse_html_listing(blob):
    """{file name: None} out of an Apache-style index page — the last resort."""
    import re as _re
    text = blob.decode("utf-8", "replace")
    names = _re.findall(r'href="([^"?#]+\.nc)"', text)
    return {n.rsplit("/", 1)[-1]: None for n in sorted(set(names))}


def oc_date_of_name(name):
    """The date a listed OC-CCI file is FOR, from the name, or None.

    The expected shape is
    `ESACCI-OC-L3S-CHLOR_A-MERGED-1D_DAILY_4km_GEO_PML_OCx-YYYYMMDD-fv6.0.nc`,
    but the stage never assumes it: this looks for the first 8-digit run that
    parses as a plausible date, so a version bump that renames the prefix does
    not silently index nothing.
    """
    import re as _re
    for m in _re.finditer(r"(?<!\d)(\d{8})(?!\d)", name):
        y, mo, d = int(m.group(1)[:4]), int(m.group(1)[4:6]), int(m.group(1)[6:])
        if 1990 <= y <= 2100 and 1 <= mo <= 12 and 1 <= d <= 31:
            try:
                return dt.date(y, mo, d)
            except ValueError:
                continue
    return None


def oc_list_year(ctx, year):
    """LIST one year of the archive. Returns (host record, {date: url}).

    THE ARCHIVE IS ASKED, NOT ASSUMED. Family 8's builder indexed the Argo GDAC
    before fetching a single profile for the same reason: a directory that
    serves 361 files instead of 365 is a fact to record, and a directory that
    serves names in a shape nobody expected is a build that must stop, not a
    build that quietly downloads nothing.

    Every host in the table is tried in order; the first that answers with at
    least one `.nc` wins. If NONE of them does, the stage exits naming every
    host and the exact URL it asked for (E-077 §2).
    """
    tried = []
    for host in OC_SOURCES[ctx.oc_source]:
        for env in host.get("credentials", ()):
            if not os.environ.get(env):
                tried.append(f"{host['name']}: ${env} is not in the "
                             f"environment")
                break
        else:
            url = host["catalog"].format(year=year)
            try:
                blob = _http_get(url)
            except Exception as e:                            # noqa: BLE001
                tried.append(f"{host['name']}: {url} -> {str(e)[:160]}")
                continue
            if host["kind"] == "thredds":
                found = parse_thredds_catalog(blob, host.get("server"))
            elif host["kind"] == "json":
                found = parse_json_listing(blob) or parse_html_listing(blob)
            else:
                found = parse_html_listing(blob)
            by_date = {}
            for name, direct in sorted(found.items()):
                d = oc_date_of_name(name)
                if d is None or d.year != year:
                    continue
                by_date[str(d)] = direct or host["file"].format(year=year,
                                                                name=name)
            if by_date:
                return host, by_date
            tried.append(f"{host['name']}: {url} listed "
                         f"{len(found)} file(s), none of them a {year} daily")
            continue
    return None, {"__tried__": tried}


def oc_index(ctx, years):
    """`<work>/occci/index.json`: what the archive says it has, per year.

    Written atomically after every year, so an interrupted index is a partial
    index and not a corrupt one, and a resume lists only the years it lacks.
    """
    work = ctx.work
    os.makedirs(os.path.join(work, "occci"), exist_ok=True)
    p = os.path.join(work, "occci", "index.json")
    idx = read_json(p, {})
    for y in years:
        ys = str(y)
        if ys in idx and idx[ys].get("files"):
            continue
        if ctx.source_dir:
            root = ctx.local("occci")
            names = sorted(os.path.basename(q) for q in
                           glob.glob(os.path.join(root or "", "*.nc")))
            by_date = {}
            for n in names:
                d = oc_date_of_name(n)
                if d is not None and d.year == y:
                    by_date[str(d)] = os.path.join(root, n)
            idx[ys] = {"host": "source-dir", "catalog": root or "",
                       "files": by_date}
        else:
            host, by_date = oc_list_year(ctx, y)
            if host is None:
                tried = by_date.get("__tried__", [])
                sys.exit(
                    f"OC-CCI: no host could list {y}. Tried:\n  "
                    + "\n  ".join(tried)
                    + f"\nThe stage refuses to continue rather than write a "
                      f"year of NaN that looks like cloud. If the archive has "
                      f"moved, fix OC_SOURCES; if this box cannot reach "
                      f"{OC_SOURCES[ctx.oc_source][0]['host']}, run "
                      f"`--oc-preflight` to see the failure on one file, and "
                      f"`--oc-source globcolour` is a DECISION, not a retry "
                      f"(E-077 §2).")
            idx[ys] = {"host": host["name"], "catalog":
                       host["catalog"].format(year=y), "files": by_date}
        atomic_json(p, idx)
        print(f"  index {y}: {len(idx[ys]['files'])} daily file(s) from "
              f"{idx[ys]['host']}", flush=True)
    return idx


def oc_open(path):
    """Open one OC-CCI daily file and return (chlor_a array, lat, lon, fill).

    VERIFY THE ARTEFACT (ml/CLAUDE.md §0.1). The variable is looked up by name
    and the axes are read out of the file — never assumed — so a file whose
    grid is not the one the block mean was derived for stops the build here,
    with the file named, instead of averaging the wrong cells into every point.
    """
    import netCDF4 as ncdf
    d = ncdf.Dataset(path)
    try:
        if OC_VAR not in d.variables:
            sys.exit(f"{path}: no {OC_VAR!r} variable — it has "
                     f"{sorted(d.variables)}. The colour channel is that "
                     f"variable and nothing else.")
        v = d.variables[OC_VAR]
        lat_name = "lat" if "lat" in d.variables else "latitude"
        lon_name = "lon" if "lon" in d.variables else "longitude"
        if lat_name not in d.variables or lon_name not in d.variables:
            sys.exit(f"{path}: no latitude/longitude variables — has "
                     f"{sorted(d.variables)}")
        s_lat = np.asarray(d.variables[lat_name][:], np.float64)
        s_lon = np.asarray(d.variables[lon_name][:], np.float64)
        if float(s_lat[0]) < float(s_lat[-1]):
            sys.exit(f"{path}: latitude ASCENDS ({s_lat[0]} .. {s_lat[-1]}). "
                     f"OC-CCI is north-first and the block mean's row "
                     f"arithmetic is derived from that; an ascending file "
                     f"would mirror the planet.")
        fill = getattr(v, "_FillValue", None)
        arr = np.ma.filled(np.asarray(v[:]), np.nan)
        return np.squeeze(np.asarray(arr, np.float64)), s_lat, s_lon, fill
    finally:
        d.close()


def oc_fetch_day(ctx, url, dest_dir):
    """One daily file onto the disk, size-verified. The caller DELETES it.

    ~10,000 files of ~40 MB is 400 GB: the Argo builder's discipline is the
    only one that fits on a 300 GB box — one file on the disk at a time, read,
    dropped, next.
    """
    os.makedirs(dest_dir, exist_ok=True)
    if ctx.source_dir or os.path.isabs(url) and os.path.exists(url):
        return url, False
    path = os.path.join(dest_dir, os.path.basename(url.split("?")[0]))
    download_verified(url, path)
    return path, True


def oc_preflight(ctx, year=2015):
    """ONE real file, fetched and opened, BEFORE anything is spent.

    ml/CLAUDE.md §0.3 and §5.16: a precondition that depends only on the inputs
    must be checked while the inputs are all it has cost you. The expensive
    half of this stage is 400 GB and three hours; the cheap half is 40 MB, and
    it answers every question that can kill the run — can this box reach the
    host at all, is the listing the shape we parse, is the variable spelled
    `chlor_a`, is the grid 4320 x 8640 north-first, does 0.25 divide it.
    """
    idx = oc_index(ctx, [year])
    files = idx[str(year)]["files"]
    if not files:
        sys.exit(f"OC-CCI preflight: {idx[str(year)]['host']} listed no daily "
                 f"file for {year} at {idx[str(year)]['catalog']}")
    day = sorted(files)[len(files) // 2]
    url = files[day]
    print(f"  preflight: {url}", flush=True)
    scratch = os.path.join(ctx.scratch, "occci")
    path, drop = oc_fetch_day(ctx, url, scratch)
    try:
        chl, s_lat, s_lon, fill = oc_open(path)
        blk_y, blk_x = oc_block_factors(s_lat, s_lon)
        fin = int(np.isfinite(chl).sum())
        print(f"  preflight OK: {day} · {os.path.getsize(path) / 1e6:.1f} MB · "
              f"{OC_VAR}{chl.shape} · lat {s_lat[0]:+.5f}..{s_lat[-1]:+.5f} · "
              f"lon {s_lon[0]:+.5f}..{s_lon[-1]:+.5f} · fill {fill} · "
              f"block {blk_y}x{blk_x} · {fin:,} finite cells "
              f"({fin / max(chl.size, 1):.1%})", flush=True)
    finally:
        if drop and os.path.exists(path):
            os.remove(path)
    ctx.note_source("occci", idx[str(year)]["catalog"])
    return True


def stage_occci(ctx):
    """ESA OC-CCI daily chlorophyll-a -> the two `oc025` channels (E-077 §3).

    THE SHAPE OF THE WORK, and why it is shaped that way:

      * PER YEAR, marked per year, exactly as `sst`. One year is ~365 files of
        ~40 MB; a year is the unit a resume replays, and the per-year marker is
        written only after the memmap is flushed (ml/CLAUDE.md §5.21 — a marker
        may only under-claim).
      * STREAMED AND DELETED AS READ. 400 GB of sources never sit on a 300 GB
        box; one file at a time, dropped the moment its numbers are in the
        accumulator.
      * A YEAR-BOUNDARY CARRY. A pentad bin straddles 31 December, so the
        accumulator for the bin holding the last days of a year is finished by
        the NEXT year's first file. `Carry` keys that accumulator by its year
        and writes it BEFORE the marker, so a crash replays exactly the years
        whose markers are missing.
      * A DAY THAT IS NOT IN THE LISTING IS COUNTED, NOT MOURNED. The archive
        genuinely has gaps (satellite outages, reprocessing holes); those are
        recorded per year in `n_occci_absent` and the bin's coverage falls,
        which is what `chl_cov` is for. A day the archive DOES list and we
        cannot fetch or open is a different animal entirely: that stops the
        build, naming the host and the file, because a silent skip there would
        put "cloudy" where "we gave up" belongs.
    """
    work = ctx.work
    if ctx.T_oc <= 0:
        print(f"  occci: this axis ends before {ctx.oc_day0} — the colour "
              f"record does not reach it; oc025 is [0, ...]")
    X = open_fill(work, "oc025", ctx.shapes()["oc025"], create=True)
    years = list(range(max(ctx.d_lo.year, ctx.oc_day0.year), ctx.d_hi.year + 1))
    idx = oc_index(ctx, years)
    carry = Carry(work, "occci", years)

    acc, days_n, cells_n = {}, {}, {}
    n_days = 0
    absent = read_json(os.path.join(work, "occci", "absent.json"), {})

    # The source geometry is PERSISTED the first time a file is opened, because
    # a resume can carry an open bin into a year whose files are all absent and
    # would otherwise have to close that bin with no idea how many source cells
    # its blocks hold. Recorded, not recomputed.
    geom_path = os.path.join(work, "occci", "geom.json")
    geom = {}
    _g = read_json(geom_path, {})
    if _g.get("blk"):
        geom["blk"] = tuple(int(v) for v in _g["blk"])
        geom["cells"] = oc_block_cells(geom["blk"][0], geom["blk"][1],
                                       int(_g["nlat_src"]))

    d = carry.load()
    if d is not None:
        for k in d.files:
            if k.startswith("acc_"):
                acc[int(k[4:])] = d[k]
            elif k.startswith("dayn_"):
                days_n[int(k[5:])] = d[k]
            elif k.startswith("celln_"):
                cells_n[int(k[6:])] = d[k]
        n_days = int(d["n_days"]) if "n_days" in d.files else 0

    def flush(b):
        """Close bin `b`: the mean of the daily block means, and the coverage."""
        row = ctx.oc_row_of(b)
        s = acc.pop(b)
        nd = days_n.pop(b)
        nc = cells_n.pop(b)
        if row is None:
            return
        if "cells" not in geom:
            sys.exit(f"occci: bin {b} carries {int((nd > 0).sum())} observed "
                     f"block(s) but no source geometry is recorded in "
                     f"{geom_path} — `chl_cov`'s denominator is the number of "
                     f"source cells per block and cannot be guessed")
        with np.errstate(invalid="ignore", divide="ignore"):
            # >= 1 FINITE DAY, not >= 3 (E-077 §3). Colour is not SST: cloud,
            # glint, ice and polar night mean a pentad with one clear day is
            # the normal case over most of the ocean, and MIN_DAYS would
            # delete most of the record. `chl_cov` carries the caveat instead
            # of the missing-value rule carrying it.
            mu = np.where(nd > 0, s / np.maximum(nd, 1), np.nan)
            cov = np.where(nd > 0,
                           nc / np.maximum(geom["cells"][:, None]
                                           * PENTAD_DAYS, 1), np.nan)
        X[row, :, :, C_LOG_CHL] = mu
        # `chl_cov` is NaN EXACTLY where `log_chl` is (E-077 §3): a block that
        # saw nothing has no coverage to report, and storing 0 there would
        # collide with the consumer's "0 vs NaN" missing-token design.
        X[row, :, :, C_CHL_COV] = cov

    def flush_ready(seen_date):
        for b in sorted(acc):
            if ctx.bin_closed(b, seen_date):
                flush(b)

    ctx.prog.stage_start("occci", len(years))
    scratch = os.path.join(ctx.scratch, "occci")
    for i, y in enumerate(years, 1):
        if marked(work, f"occci/{y}"):
            continue
        files = idx[str(y)]["files"]
        lo = max(ctx.d_lo, ctx.oc_day0, dt.date(y, 1, 1))
        hi = min(ctx.d_hi, dt.date(y, 12, 31))
        want = [lo + dt.timedelta(days=k) for k in range((hi - lo).days + 1)] \
            if hi >= lo else []
        miss = 0
        for day in want:
            url = files.get(str(day))
            if url is None:
                miss += 1                     # a real hole in the archive
                continue
            path, drop = oc_fetch_day(ctx, url, scratch)
            try:
                chl, s_lat, s_lon, fill = oc_open(path)
                if "blk" not in geom:
                    blk_y, blk_x = oc_block_factors(s_lat, s_lon)
                    geom["blk"] = (blk_y, blk_x)
                    geom["cells"] = oc_block_cells(blk_y, blk_x, len(s_lat))
                    atomic_json(geom_path, {"blk": [blk_y, blk_x],
                                            "nlat_src": int(len(s_lat)),
                                            "nlon_src": int(len(s_lon)),
                                            "at": utcnow()})
                    print(f"  block  {len(s_lat)}x{len(s_lon)} cells -> "
                          f"{NLAT}x{NLON} points, {blk_y}x{blk_x} per block "
                          f"({int(geom['cells'][NLAT // 2])} cells at the "
                          f"equator, {int(geom['cells'][0])} at the poles)",
                          flush=True)
                S, C = oc_block_stats(chl, *geom["blk"], fill=fill)
            except SystemExit:
                raise
            except Exception as e:                            # noqa: BLE001
                sys.exit(f"OC-CCI {day}: {url} was listed by "
                         f"{idx[str(y)]['host']} but could not be read "
                         f"({type(e).__name__}: {str(e)[:200]}). The stage "
                         f"refuses to record a listed day as cloud.")
            finally:
                if drop and os.path.exists(path):
                    os.remove(path)
            b = bin_index(day, PENTAD_DAYS)
            flush_ready(day)
            if ctx.oc_row_of(b) is None:
                continue
            if b not in acc:
                acc[b] = np.zeros((NLAT, NLON), np.float64)
                days_n[b] = np.zeros((NLAT, NLON), np.int16)
                cells_n[b] = np.zeros((NLAT, NLON), np.int32)
            got = C > 0
            with np.errstate(invalid="ignore", divide="ignore"):
                # The DAILY block mean first, then the mean of those over the
                # days — not the pooled mean over cell-days. A day with two
                # clear pixels and a day with thirty-six weigh the same,
                # because each is one observation of that block on that day.
                acc[b] += np.where(got, S / np.maximum(C, 1), 0.0)
            days_n[b] += got
            cells_n[b] += C.astype(np.int32)
            n_days += 1
        # Only bins the NEXT year cannot touch are closed here; the rest carry.
        flush_ready(dt.date(y + 1, 1, 1))
        X.flush()                                # flush, THEN mark
        absent[str(y)] = int(miss)
        atomic_json(os.path.join(work, "occci", "absent.json"), absent)
        carry.commit(y, n_days=np.array(n_days),
                     **{f"acc_{b}": v for b, v in acc.items()},
                     **{f"dayn_{b}": v for b, v in days_n.items()},
                     **{f"celln_{b}": v for b, v in cells_n.items()})
        ctx.prog.item(y, i, {"days": n_days, "absent": int(miss),
                             "open_bins": len(acc)})

    for b in sorted(acc):
        flush(b)
    X.flush()
    bump_counts(work, n_occci_days=n_days,
                n_occci_absent=int(sum(absent.values())),
                oc_bin_first=int(ctx.b_oc))
    ctx.note_source("occci", (ctx.local("occci") + "/") if ctx.source_dir
                    else "; ".join(sorted({v["catalog"] for v in idx.values()})))
    mark(work, "occci")
    oc_report(ctx, X)
    print(f"  occci: {n_days} daily field(s) folded into {ctx.T_oc} bins from "
          f"{bin_start(ctx.b_oc, PENTAD_DAYS)} (bin {ctx.b_oc}); "
          f"{int(sum(absent.values()))} day(s) absent from the archive listing")


def oc_report(ctx, X):
    """E-077 §7.4: print the comparison bin's global mean and observed share.

    The number a reader compares against the PACE/MODIS chlorophyll layer on
    the globe BY EYE, printed into the build log so it survives the box, and
    copied into `ml/EXPERIMENTS.md#e-077` when the run lands.
    """
    if X.shape[0] == 0:
        return None
    b = bin_index(OC_REPORT_DAY, PENTAD_DAYS)
    row = ctx.oc_row_of(b)
    why = f"bin {b} ({OC_REPORT_DAY})"
    if row is None:
        row = X.shape[0] // 2
        b = ctx.b_oc + row
        why = (f"bin {b} ({bin_start(b, PENTAD_DAYS)}) — the comparison bin "
               f"{bin_index(OC_REPORT_DAY, PENTAD_DAYS)} is off this axis")
    chl = np.asarray(X[row, :, :, C_LOG_CHL], np.float64)
    cov = np.asarray(X[row, :, :, C_CHL_COV], np.float64)
    fin = np.isfinite(chl)
    n = int(fin.sum())
    rec = {"bin": int(b), "row": int(row), "observed_cells": n,
           "observed_fraction": float(n / chl.size),
           "log_chl_mean": float(np.mean(chl[fin])) if n else None,
           "chl_cov_mean": float(np.mean(cov[fin])) if n else None}
    print(f"  report {why}: log_chl mean "
          f"{rec['log_chl_mean'] if n else float('nan'):.4f} "
          f"(= {10 ** rec['log_chl_mean']:.4f} mg/m3) over {n:,} cells "
          f"({rec['observed_fraction']:.1%} of the grid); mean chl_cov "
          f"{rec['chl_cov_mean'] if n else float('nan'):.4f}"
          if n else f"  report {why}: no observed cell")
    atomic_json(os.path.join(ctx.work, "occci", "report.json"), rec)
    return rec


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
                # hf_hub_download keeps the repo's own `truth/` folder under
                # local_dir, so the file lands at <root>/truth/<name> — one
                # level below where the builder (and family 4) read it. Run 2
                # of the real build (2026-09-04) pulled both files and then
                # refused on exactly this. Move it up, and ASSERT the path the
                # caller will read rather than trusting the download's return.
                dest = os.path.join(root, name)
                if os.path.abspath(p) != os.path.abspath(dest):
                    os.replace(p, dest)
                if not os.path.exists(dest):
                    raise FileNotFoundError(dest)
                print(f"  pulled truth/{name} from the Hub -> {dest}")
            except Exception as e:                            # noqa: BLE001
                print(f"  ::warning:: truth/{name}: {str(e)[:140]}")
        shutil.rmtree(os.path.join(root, "truth"), ignore_errors=True)
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


# ============================================================= seeding (7.1) ==
# WHAT SEEDING IS, and why it is not a copy. Family 7.1 is family 7 plus one
# group. The three finished groups are 53 GB that took five hours to build and
# whose sha256 the Hub, the handover and another agent's hand-off already cite;
# rebuilding them would spend the money again AND risk producing different
# bytes. So the new work directory HARD-LINKS them under the new stem
# (`os.link`, one inode, zero extra bytes, same filesystem) and copies the
# small state beside them.
#
# THE RULE THIS OBEYS: nothing under the old directory is written, renamed or
# deleted. `os.link` reads the old directory's inode and adds a name in the
# new one; every other file is copied with `shutil.copy2`. A build that
# corrupted its own seed would have destroyed the published tensor.
SEED_FILES = ("norm.npz", "norm_g025.npz", "statics.npz", "truth.npz",
              "counts.json", "sources.json", "oisst_seen.npy",
              "glorys_seen.npy")
SEED_DIRS = ("glorys", "sst", "ncep", "rg", "norm")
SEED_MARKERS = INHERITED_STAGES + ("repair_sst", "norm")


def seed_from(ctx, old_work):
    """Inherit a finished f7l0 work dir into this one (E-077 §5).

    ASSERTS, not hopes (ml/CLAUDE.md §0.1, §0.2): every linked file is checked
    to BE the same inode as its source and to have exactly the byte size its
    own `.npy` header implies, because "the link call returned" says nothing
    about which bytes are now under the new name.
    """
    old = os.path.abspath(old_work)
    new = ctx.work
    if not os.path.isdir(old):
        sys.exit(f"--seed-from {old}: no such directory. This flag is a claim "
                 f"that a finished f7l0 build is there; a missing one is an "
                 f"error, not a licence to rebuild 53 GB silently (E-077 §5).")
    if os.path.abspath(old) == os.path.abspath(new):
        sys.exit(f"--seed-from {old} is the work directory itself")
    linked = []
    for g in BASE_GROUPS:
        src = os.path.join(old, f"{BASE_STEM}_X_{g}.npy")
        dst = group_file(new, g)
        if not os.path.exists(src):
            sys.exit(f"--seed-from: {src} is missing. The seed must be a "
                     f"FINISHED f7l0 build — all three published group files, "
                     f"already z-scored by its own `norm` stage.")
        if os.path.exists(dst):
            if not os.path.samefile(src, dst):
                sys.exit(f"{dst} already exists and is not the same file as "
                         f"{src} — refusing to replace it. Use a fresh --work.")
        else:
            try:
                os.link(src, dst)
            except OSError as e:                              # noqa: BLE001
                sys.exit(f"--seed-from: cannot hard-link {src} -> {dst} "
                         f"({e}). The two directories must be on ONE "
                         f"filesystem (on the box: both under "
                         f"/opt/earth-cache). Copying 53 GB instead would "
                         f"double the disk this build needs.")
        # THE ASSERTIONS OF §5, on the bytes rather than on the call.
        if not os.path.samefile(src, dst):
            sys.exit(f"--seed-from: {dst} is not the same inode as {src} "
                     f"after linking")
        try:
            m = np.lib.format.open_memmap(dst, mode="r")
        except Exception as e:                                # noqa: BLE001
            sys.exit(f"--seed-from: {dst} is {os.path.getsize(dst):,} bytes "
                     f"and numpy will not open it ({type(e).__name__}: "
                     f"{str(e)[:120]}). A truncated or corrupt seed is not a "
                     f"seed — the published tensor is 45.67 GB of float16 and "
                     f"this is not it.")
        want = int(np.prod(m.shape)) * m.dtype.itemsize
        got = os.path.getsize(dst)
        if got != want + 128:
            sys.exit(f"--seed-from: {dst} is {got:,} bytes; its header says "
                     f"{m.shape} {m.dtype} = {want:,} + a 128-byte header = "
                     f"{want + 128:,}. A truncated seed would train on "
                     f"whatever followed it.")
        if m.dtype != np.float16:
            sys.exit(f"--seed-from: {dst} is {m.dtype}, not float16 — the seed "
                     f"must be the PUBLISHED (already z-scored) group, not a "
                     f"float32 intermediate")
        linked.append((g, m.shape, got))
        del m
        # READ-ONLY, ON THE INODE, WHICH MEANS ON BOTH NAMES. That is not a
        # side effect to apologise for — it is the statement: the f7l0 build is
        # FINISHED, its bytes are published, and nothing may write either copy
        # again. Doing it here rather than at each call site means a future
        # stage that forgets to ask `refuse_if_seeded` still meets a closed
        # door on an unprivileged runner. (On the Vast boxes the builder runs
        # as root and root ignores the mode bits — measured — so this is the
        # second line of defence, not the first; `refuse_if_seeded` is the
        # first and it holds for everybody.)
        os.chmod(dst, 0o444)

    for name in SEED_FILES:
        src = os.path.join(old, name)
        if os.path.exists(src):
            shutil.copy2(src, os.path.join(new, name))
    for sub in SEED_DIRS:
        src = os.path.join(old, sub)
        if os.path.isdir(src):
            dst = os.path.join(new, sub)
            os.makedirs(dst, exist_ok=True)
            for q in sorted(glob.glob(os.path.join(src, "*"))):
                if os.path.isfile(q):
                    shutil.copy2(q, os.path.join(dst, os.path.basename(q)))
    for name in SEED_MARKERS:
        src = marker(old, name)
        if os.path.exists(src):
            dst = marker(new, name)
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            shutil.copy2(src, dst)
    # THE `.spec` FILES ARE COPIED ONLY FOR THE INHERITED STAGES. `norm`,
    # `meta` and `publish` legitimately changed recipe (E-077 §5), so carrying
    # their f7l0 digests across would make `stage_state_check` declare `norm`
    # stale — and `norm` is one of the two stages it refuses to discard,
    # which would end the build before it started.
    for name in INHERITED_STAGES:
        src = os.path.join(old, f"{name}.spec")
        if os.path.exists(src):
            shutil.copy2(src, os.path.join(new, f"{name}.spec"))

    ctx.sources = read_json(os.path.join(new, "sources.json"), {})
    ctx.note_source("seed", f"{BASE_RECIPE} work dir {old} (hard links)")
    atomic_json(os.path.join(new, "seed.json"),
                {"from": old, "base_recipe": BASE_RECIPE,
                 "base_stem": BASE_STEM, "at": utcnow(),
                 "linked": [{"group": g, "shape": list(s), "bytes": b}
                            for g, s, b in linked]})
    mark(new, "seed")
    for g, s, b in linked:
        print(f"  seed   {g}: hard-linked {tuple(s)} float16, {b / 1e9:.2f} GB "
              f"(one inode, zero new bytes)")
    print(f"  seed   markers {', '.join(SEED_MARKERS)} and "
          f"{len(INHERITED_STAGES)} spec file(s) copied from {old}")
    return linked


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


def norm_state(ctx, g, norms=None):
    """(has_stats, has_final, marked) for one group's normalisation."""
    work = ctx.work
    if norms is None:
        norms = {}
        p = os.path.join(work, "norm.npz")
        if os.path.exists(p):
            d = np.load(p)
            norms = {k: d[k] for k in d.files}
    return (f"norm_{g}" in norms, os.path.exists(group_file(work, g)),
            marked(work, f"norm/{g}"))


def norm_pending(ctx, groups=None):
    """Which groups `norm` still has work for (E-077 §5).

    THE REASON THIS FUNCTION EXISTS. Until 7.1 `norm` was one marker for all
    groups, which is exactly right when every group is built in the same run
    and exactly wrong when three of them arrive already normalised through
    `--seed-from` and a fourth does not: `norm.done` is present and truthful
    about g025/g100/rg100, and `oc025` has never been touched. Asking the
    question per GROUP — does `norm.npz` carry its statistics, is its float16
    file on disk, is its own marker there — replaces a marker that can only
    answer for the whole stage.

    It must never return a group that is already z-scored: re-running g025's
    in-place pass would square the transform, and nothing downstream would say
    so (the docstring below, and ml/CLAUDE.md §5.21).
    """
    out = []
    for g in (groups or GROUPS):
        stats, final, mk = norm_state(ctx, g)
        if stats and final and mk:
            continue
        if mk and not final and g not in RAW_F32:
            sys.exit(f"norm: {g} is marked done but {group_file(ctx.work, g)} "
                     f"is gone. Its z-score was IN PLACE, so there is no "
                     f"float32 source to redo it from — rebuild the work dir "
                     f"deliberately rather than re-normalising unknown bytes.")
        out.append(g)
    return out


def stage_norm(ctx, groups=None):
    """Per group, per channel (mean, sd) over EVERY finite value, then z-score.

    Family 4's convention (`build_family4.py:913-935`), with one deliberate
    difference the plan spells out: **the `slab[~ocean] = NaN` line is gone**.
    Land is observed now — that is the whole point of family 7 — so masking it
    would delete the shared channels it was written to carry.

    Resumable INSIDE a group. The statistics are computed and written first;
    the z-score then walks the bins in chunks and records the next unprocessed
    bin after every flush, so a killed job never z-scores the same slab twice
    (which would silently square the transform).

    GROUP-IDEMPOTENT since 7.1. `groups` defaults to every group that still has
    work (`norm_pending`), so a directory seeded from f7l0 normalises `oc025`
    alone and never re-enters the three that arrived already z-scored.
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

    todo = list(groups) if groups else norm_pending(ctx)
    if not todo:
        print("  norm: every group already has its statistics and its "
              "float16 file — nothing to do")
        mark(work, "norm")
        return
    print(f"  norm: {todo} (of {GROUPS})")

    ctx.prog.stage_start("norm", len(todo) * 2)
    step = 0
    for g in todo:
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

    for g in todo:
        prog_path = os.path.join(work, "norm", f"{g}.progress.json")
        done_at = read_json(prog_path, {}).get("next_bin", 0)
        if marked(work, f"norm/{g}") and os.path.exists(group_file(work, g)):
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
        n = norms.get(f"norm_{g}")
        if n is None:
            continue
        print(f"  norm_{g}: {n.shape[0]} channels, "
              f"{int(norms[f'count_{g}'].sum()):,} observed values")


# ============================================================== stage: meta ==
def oc_inland_count(ctx, sphere):
    """E-077 §7.3: MEASURE the colour-on-land overlap, do not mask it.

    The 4 km product resolves coastal water that the 0.25-degree `sphere` mask
    calls land, so a block centred on a land point can legitimately contain
    clear-water pixels. Masking those away would delete real observations of
    estuaries and shelf seas — the most interesting colour on the planet — so
    the build MEASURES instead: how many finite `log_chl` cells sit on a land
    cell that is not even touching the sea. That count is the one that would be
    alarming, and it is recorded in the npz (`n_oc_inland`) rather than acted
    on, because a number in the file can be argued with and a mask cannot.
    """
    p = group_file(ctx.work, "oc025")
    if not os.path.exists(p) or np.asarray(sphere).shape != (NLAT, NLON):
        return -1
    X = np.load(p, mmap_mode="r")
    if X.shape[0] == 0:
        return 0
    land = np.asarray(sphere) == 1
    sea = np.asarray(sphere) == 0
    fringe = np.zeros_like(sea)
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            fringe |= np.roll(np.roll(sea, dy, axis=0), dx, axis=1)
    inland = land & ~fringe
    n = 0
    chunk = bin_chunk(X.shape)
    for i in range(0, X.shape[0], chunk):
        slab = np.asarray(X[i:i + chunk, :, :, C_LOG_CHL], np.float32)
        n += int((np.isfinite(slab) & inland[None]).sum())
    print(f"  oc025: {n:,} finite log_chl value(s) on land cells that do not "
          f"touch the sea (measured, not masked — E-077 §7.3)")
    return n


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
        chan_rg100=np.array(CHAN_RG100), chan_oc025=np.array(CHAN_OC025),
        groups=np.array(GROUPS),
        window=np.array("global025"), recipe=np.array(RECIPE),
        cadence=np.array("pentad"),
        rg_bin_index=np.asarray(live["bin_index"], np.int64),
        rg_months=live["months"],
        # E-077 §4: the offset axis, stated two ways. `oc_bin_first` is the
        # scalar the app's slab arithmetic and the handover use; the explicit
        # per-row bin list is what `ml/cone_sampler.py` already reads for any
        # group whose row count differs from the master's, so the gather needed
        # no new mechanism at all.
        oc_bin_first=np.array(int(ctx.b_oc), np.int64),
        oc025_bin_index=np.arange(ctx.b_oc, ctx.b_oc + ctx.T_oc,
                                  dtype=np.int64),
        sphere=statics["sphere"], elev=statics["elev"],
        n_glorys_bins=np.array(counts.get("n_glorys_bins", 0)),
        n_sst_days=np.array(counts.get("n_sst_days", 0)),
        n_ncep_days=np.array(counts.get("n_ncep_days", 0)),
        n_rg_live=np.array(counts.get("n_rg_live", len(live["bin_index"]))),
        n_occci_days=np.array(counts.get("n_occci_days", 0)),
        n_occci_absent=np.array(counts.get("n_occci_absent", 0)),
        sources=np.array(json.dumps(ctx.sources, sort_keys=True)),
        builder_git_sha=np.array(git_sha()),
        built_at=np.array(utcnow()),
    )
    for k in norms.files:
        if k.startswith("norm_") or k == "count_oc025":
            meta[k] = norms[k]
    for k in truths.files:
        meta[k] = truths[k]
    meta["n_oc_inland"] = np.array(oc_inland_count(ctx, statics["sphere"]))

    out = os.path.join(work, STEM + ".npz")
    tmp = out + f".tmp{os.getpid()}.npz"
    np.savez(tmp, **meta)
    os.replace(tmp, out)
    mark(work, "meta")
    print(f"  meta: wrote {out} ({os.path.getsize(out) / 1e6:.1f} MB, "
          f"{len(meta)} keys)")
    return out


# =========================================================== stage: publish ==
def base_manifest_hashes(api, repo, tok):
    """{group: sha256} from the PUBLISHED f7l0 manifest (E-077 §5).

    `same_as_f7l0: true` in our own manifest would be a claim about someone
    else's bytes, so it is not written from what the seed directory happened to
    contain — it is written after fetching f7l0's manifest and comparing.
    """
    from huggingface_hub import hf_hub_download
    p = hf_hub_download(repo, f"{BASE_PREFIX}/manifest.json",
                        repo_type="dataset", token=tok)
    man = json.load(open(p, encoding="utf-8"))
    out = {}
    for rec in man.get("files", []):
        name = rec.get("name", "")
        for g in BASE_GROUPS:
            if name == f"{BASE_STEM}_X_{g}.npy":
                out[g] = rec.get("sha256")
    return out


def stage_publish(ctx):
    """Upload the five files, DOWNLOAD EACH BACK and compare sha256.

    An upload that returns 200 is not evidence the bytes are retrievable
    (ml/CLAUDE.md §0.2). `ml/hf_mirror.py`'s rule: a backup is only real if the
    restore works, so a publish that cannot verify FAILS the job.

    7.1 adds one more check, and it is the one the whole recipe rests on: the
    three inherited group files must be BYTE-IDENTICAL to f7l0's, because the
    handover, the Hub and another agent's hand-off all describe them by those
    hashes. So the f7l0 manifest is fetched and compared, `same_as_f7l0` is
    written per file from the comparison rather than from the intention, and a
    mismatch FAILS the publish — at that point the file is not the one anyone
    was promised, whatever the rest of the build did.
    """
    from huggingface_hub import hf_hub_download
    work = ctx.work
    api, repo, tok = hub_repo()
    files = [os.path.join(work, STEM + ".npz")] + \
            [group_file(work, g) for g in GROUPS]
    try:
        base = base_manifest_hashes(api, repo, tok)
        print(f"  publish: {BASE_RECIPE} manifest lists "
              f"{len(base)}/{len(BASE_GROUPS)} inherited group hashes")
    except Exception as e:                                    # noqa: BLE001
        sys.exit(f"cannot fetch {BASE_PREFIX}/manifest.json ({str(e)[:200]}). "
                 f"The 7.1 manifest states that its three inherited files are "
                 f"identical to {BASE_RECIPE}'s; that sentence has to be "
                 f"checked against {BASE_RECIPE}'s own hashes, not asserted.")
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
        rec = {"name": name, "bytes": os.path.getsize(p), "sha256": src}
        for g in BASE_GROUPS:
            if name != f"{STEM}_X_{g}.npy":
                continue
            want = base.get(g)
            if want is None:
                sys.exit(f"{BASE_PREFIX}/manifest.json does not list "
                         f"{BASE_STEM}_X_{g}.npy — this build claims its {g} "
                         f"is {BASE_RECIPE}'s and there is nothing to compare "
                         f"it with")
            if want != src:
                sys.exit(f"INHERITANCE BROKEN on {g}: this build's file "
                         f"hashes {src}, {BASE_RECIPE}'s manifest says "
                         f"{want}. The seed was not the published tensor (or "
                         f"something wrote through the hard link). Publishing "
                         f"it would put a file on the Hub that the family-7 "
                         f"handover describes and does not match.")
            rec["same_as_f7l0"] = True
            rec["base_name"] = f"{BASE_STEM}_X_{g}.npy"
        entries.append(rec)
        ctx.prog.item(name, i, {"sha256": src[:16],
                                "same_as_f7l0": rec.get("same_as_f7l0", False)})

    man = {"recipe": RECIPE, "base_recipe": BASE_RECIPE, "stem": STEM,
           "base_stem": BASE_STEM, "base_prefix": BASE_PREFIX,
           "groups": GROUPS, "oc_bin_first": int(ctx.b_oc),
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
            "rg": stage_rg, "occci": stage_occci, "static": stage_static,
            "truth": stage_truth, "norm": stage_norm, "meta": stage_meta,
            "publish": stage_publish}


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
            # `norm` is the ONE stage whose marker cannot answer for the whole
            # stage any more (E-077 §5): a seeded work dir carries a truthful
            # `norm.done` for the three inherited groups while `oc025` has
            # never been normalised. So the marker is overridden by the per
            # GROUP question, and only by that — a group that IS normalised is
            # never re-entered, because its z-score is not repeatable.
            if s == "norm":
                pend = norm_pending(ctx)
                if pend:
                    print(f"stage norm: `norm.done` exists but {pend} "
                          f"has no normalisation — re-entering for it alone")
                else:
                    print("stage norm: already done — skipping (--force to redo)")
                    continue
            else:
                print(f"stage {s}: already done — skipping (--force to redo)")
                continue
        if s == "norm" and ctx.a.force and marked(work, "norm/g025"):
            # g100 and rg100 are idempotent — their norm reads the float32
            # intermediate and writes the float16 sidecar, so --force is safe.
            # g025 has no intermediate: its z-score is IN PLACE, and redoing it
            # over already-z-scored data would square the transform with
            # nothing downstream to say so (ml/CLAUDE.md §5.21).
            sys.exit("refusing --force on `norm`: g025 is already z-scored in "
                     "place (norm/g025.done), so redoing it would z-score "
                     "already-z-scored data. Delete the work dir to rebuild. "
                     "(g100, rg100 and oc025 alone are safe to redo once "
                     "norm/g025.done is what stops you — remove only their "
                     "own markers; and a group that has NEVER been normalised "
                     "needs no --force at all, `norm` re-enters for it by "
                     "itself.)")
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
# The colour record starts fifteen years before this axis, so the smoke MOVES
# it inside the axis: with `oc_bin_first` five days in, the offset axis of
# E-077 §4 is the thing under test rather than a no-op. Declared here so the
# tests can read it instead of restating it.
SMOKE_OC_START = "2010-01-18"
# The smoke's OC-CCI grid is DECIMATED, and the test says so out loud (E-077
# §5 allows it). 1/8 degree — 1440 x 2880 — is the coarsest grid on which a
# 0.25-degree point still sits on a cell boundary with an EVEN number of cells
# per axis, so the pole truncation is symmetric (one row of two at each pole)
# exactly as it is with the real 6 x 6 blocks (three rows of six). Every rule
# under test — log space, the >= 1 finite day, the wrap, the truncation, the
# coverage denominator — is the same code on the same shapes; only the
# constant 36 becomes 4.
SMOKE_OC_NLAT, SMOKE_OC_NLON = 1440, 2880
# The four points the smoke plants exact values into, as (y, x) on the
# 0.25-degree grid (E-077 §5's smoke assertions).
SMOKE_OC_POINTS = {
    "uniform": (400, 900),      # every cell 10**0.5 on every day  -> 0.5, cov 1
    "single": (401, 901),       # one cell on one day              -> its log10
    "empty": (402, 902),        # all fill, always                 -> NaN, NaN
    "wrap": (300, 0),           # lon -180: the block straddles the dateline
    "south_pole": (0, 700),     # the truncated block at -90
    "north_pole": (NLAT - 1, 700),
}
SMOKE_OC_UNIFORM = 0.5          # log10(mg/m3) in the uniform block
SMOKE_OC_SINGLE = -0.7          # log10(mg/m3) of the one clear cell
SMOKE_OC_WRAP_W, SMOKE_OC_WRAP_E = -1.0, 1.0     # west / east of the dateline


def make_smoke_oc_sources(root, days, oc_start,
                          nlat_s=SMOKE_OC_NLAT, nlon_s=SMOKE_OC_NLON):
    """Synthetic OC-CCI dailies with the REAL registration, decimated.

    Cell-registered, NORTH-FIRST latitude, longitude from -180, a `_FillValue`
    that is not NaN (the real product's is), and the real file-name shape — so
    the stage's listing, its date parse, its axis assertions, its fill
    handling and its block mean all run the code they will run on the box.

    Days before `oc_start` get NO FILE AT ALL, which is also the point: the
    stage must not invent rows for them, and `oc_bin_first` must land on the
    bin holding the first day that does exist.
    """
    d = 180.0 / nlat_s
    s_lat = 90.0 - d * (np.arange(nlat_s) + 0.5)          # descending
    s_lon = -180.0 + d * (np.arange(nlon_s) + 0.5)
    blk_y, blk_x = int(round(0.25 / d)), int(round(0.25 / d))
    fill = np.float32(-999.0)
    rng = np.random.default_rng(20260911)
    out = []
    # A smooth, plausible background in log space, with a broad "cloud" band
    # that moves with the day so the >= 1 finite day rule is exercised.
    base = (10.0 ** (0.2 * np.sin(np.radians(s_lat))[:, None]
                     + 0.1 * np.cos(np.radians(s_lon))[None, :] - 0.4)
            ).astype(np.float32)
    for day in days:
        if day < oc_start:
            continue
        k = len(out)                     # the index among the files WRITTEN
        a = base * (1.0 + 0.02 * rng.standard_normal((nlat_s, 1))
                    ).astype(np.float32)
        cloud = (np.abs(s_lat[:, None] - (30.0 - 3.0 * k)) < 4.0) \
            & (s_lon[None, :] > 0)
        a = np.where(cloud, fill, a).astype(np.float32)
        for name, (y, x) in SMOKE_OC_POINTS.items():
            rows, cols = oc_block_slices(y, x, blk_y, blk_x, nlat_s, nlon_s)
            rr = np.array(rows)[:, None]
            cc = np.array(cols)[None, :]
            if name == "uniform":
                a[rr, cc] = np.float32(10.0 ** SMOKE_OC_UNIFORM)
            elif name == "single":
                a[rr, cc] = fill
                if k == 0:
                    a[rows[0], cols[0]] = np.float32(10.0 ** SMOKE_OC_SINGLE)
            elif name == "empty":
                a[rr, cc] = fill
            elif name == "wrap":
                # The west half of the block is on the far side of the
                # dateline (columns near nlon_s - 1); the east half near 0.
                for c in cols:
                    v = SMOKE_OC_WRAP_W if c >= nlon_s // 2 else SMOKE_OC_WRAP_E
                    a[rr, c] = np.float32(10.0 ** v)
            else:                                   # the two pole blocks
                a[rr, cc] = np.float32(10.0 ** SMOKE_OC_UNIFORM)
        name = (f"ESACCI-OC-L3S-CHLOR_A-MERGED-1D_DAILY_4km_GEO_PML_OCx-"
                f"{day:%Y%m%d}-fv6.0.nc")
        _nc_write(os.path.join(root, "occci", name),
                  {"lat": nlat_s, "lon": nlon_s},
                  {"lat": (("lat",), s_lat, {"units": "degrees_north"}),
                   "lon": (("lon",), s_lon, {"units": "degrees_east"}),
                   OC_VAR: (("lat", "lon"), a,
                            {"_FillValue": fill, "units": "mg m^-3"})})
        out.append(name)
    return out


def seed_layout(work):
    """Rename a finished work dir's group files to the BASE stem.

    The first leg of the smoke builds the three inherited groups with THIS
    builder, which naturally names them `f7l1`. A real seed directory is an
    f7l0 build and its files carry the f7l0 stem, so the smoke renames them:
    otherwise `--seed-from` would be tested against a directory no f7l0 build
    ever produces, which is the kind of green test that proves nothing.
    """
    for g in BASE_GROUPS:
        src = os.path.join(work, f"{STEM}_X_{g}.npy")
        dst = os.path.join(work, f"{BASE_STEM}_X_{g}.npy")
        if os.path.exists(src) and not os.path.exists(dst):
            os.replace(src, dst)
    return work


def run_smoke(root=None, keep=False, start=SMOKE_START, end=SMOKE_END,
              oc_start=SMOKE_OC_START):
    """Two legs: build an f7l0-shaped seed, then inherit it and add colour.

    LEG 1 is the family-7 smoke that has always been here — tiny synthetic
    GLORYS / OISST / NCEP / RG / Natural Earth / ETOPO / truth through every
    stage up to `norm`, over the three base groups — finished by renaming its
    group files to the f7l0 stem so it IS a seed directory.

    LEG 2 is E-077: a fresh work dir, `--seed-from` the first one, a synthetic
    OC-CCI source tree, and the `occci` / `norm` / `meta` stages. It asserts
    what §5 lists — byte-identical seeded groups, the offset shape, the exact
    block values, the wrap, the poles — because a smoke that only checks the
    files exist is a smoke that passes on an empty tensor.
    """
    tmp = root or tempfile.mkdtemp(prefix="f7smoke_")
    src = os.path.join(tmp, "src")
    seed = os.path.join(tmp, "seed")
    work = os.path.join(tmp, "work")
    os.makedirs(seed, exist_ok=True)
    os.makedirs(work, exist_ok=True)
    t0 = time.time()
    print(f"smoke     sources -> {src}")
    d_lo = dt.date(*(int(x) for x in start.split("-")))
    d_hi = dt.date(*(int(x) for x in end.split("-")))
    days = [d_lo + dt.timedelta(days=k) for k in range((d_hi - d_lo).days + 1)]
    oc0 = dt.date(*(int(x) for x in oc_start.split("-")))
    make_smoke_sources(src, d_lo, d_hi)
    make_smoke_oc_sources(src, days, oc0)
    print(f"smoke     synthetic sources written in {time.time() - t0:.1f}s")

    def ns(w, **kw):
        a = dict(work=w, source_dir=src, start=start, end=end, force=False,
                 stage="all", smoke=True, seed_from="", oc_source="occci",
                 oc_start=oc_start, oc_preflight=False)
        a.update(kw)
        return argparse.Namespace(**a)

    # ---- leg 1: the f7l0-shaped seed ------------------------------------
    print("\n########## smoke leg 1: the f7l0 seed ##########")
    c1 = Ctx(ns(seed))
    print(f"axis      {bin_start(c1.bins[0], 5)} .. "
          f"{bin_start(c1.bins[-1], 5)}  T={c1.T} pentad bins "
          f"(bins {c1.b_lo}..{c1.b_hi}, recipe {BASE_RECIPE} shape)")
    disk_guard(seed, c1.byte_peak(64), headroom=2e8)
    run_stages(c1, [s for s in INHERITED_STAGES])
    stage_norm(c1, groups=BASE_GROUPS)
    seed_layout(seed)
    before = {g: sha256(os.path.join(seed, f"{BASE_STEM}_X_{g}.npy"))
              for g in BASE_GROUPS}

    # ---- leg 2: inherit, then colour ------------------------------------
    print("\n########## smoke leg 2: f7l1 = f7l0 + oc025 ##########")
    ctx = Ctx(ns(work, seed_from=seed))
    seed_from(ctx, seed)
    print(f"axis      T={ctx.T} · colour T_oc={ctx.T_oc} from "
          f"{bin_start(ctx.b_oc, 5)} (oc_bin_first={ctx.b_oc}, recipe {RECIPE})")
    disk_guard(work, ctx.byte_peak(64), headroom=2e8)
    run_stages(ctx, ["occci", "norm", "meta"])

    out = check_smoke(work, ctx, seed=seed, seed_sha=before)
    print(f"\nsmoke     OK in {time.time() - t0:.1f}s — {out}")
    if not keep and root is None:
        shutil.rmtree(tmp, ignore_errors=True)
    return work


REQUIRED_KEYS = ["bin_index", "months", "epoch", "pentad_days", "lats", "lons",
                 "lat1", "lon1", "chan_g025", "chan_g100", "chan_rg100",
                 "chan_oc025", "groups", "window", "recipe", "cadence",
                 "sphere", "elev",
                 "norm_g025", "norm_g100", "norm_rg100", "norm_oc025",
                 "count_oc025", "oc_bin_first", "oc025_bin_index",
                 "rg_bin_index", "n_glorys_bins", "n_sst_days", "n_ncep_days",
                 "n_rg_live", "n_occci_days", "n_occci_absent", "n_oc_inland",
                 "sources", "builder_git_sha", "rapid", "truth_rapid"]


def check_oc_smoke(work, ctx, d):
    """E-077 §5's colour assertions, on the bytes the smoke just wrote.

    Every number here is EXACT and derived from the plan, not from the run
    (ml/CLAUDE.md §4.9: an invariant with an exact expected value is worth more
    than any amount of careful reading). `through_f16` is not needed because
    `oc025` is filled at float32 and quantised once, on the z-score — so a
    value of 0.5 comes back as 0.5 to within one float16 step of the z-scored
    number, which is what the tolerances below are.
    """
    X = np.asarray(d["X_oc025"], np.float32)
    mu, sd = np.asarray(d["norm_oc025"])[C_LOG_CHL]
    mu_c, sd_c = np.asarray(d["norm_oc025"])[C_CHL_COV]
    chl = X[..., C_LOG_CHL] * sd + mu
    cov = X[..., C_CHL_COV] * sd_c + mu_c
    tol = 3e-3 * max(float(sd), 1e-6) + 1e-4
    tol_c = 3e-3 * max(float(sd_c), 1e-6) + 1e-4

    assert X.shape == (ctx.T_oc, NLAT, NLON, 2), X.shape
    assert int(np.asarray(d["oc_bin_first"])) == ctx.b_oc
    assert int(np.asarray(d["oc_bin_first"])) == \
        bin_index(ctx.oc_day0, PENTAD_DAYS), \
        "oc_bin_first is not the bin of the colour record's first day"
    assert np.array_equal(np.asarray(d["oc025_bin_index"]),
                          np.arange(ctx.b_oc, ctx.b_oc + ctx.T_oc))

    # §7.3, the two invariants that make the pair readable at all.
    fin_chl, fin_cov = np.isfinite(chl), np.isfinite(cov)
    assert np.array_equal(fin_chl, fin_cov), \
        "log_chl and chl_cov do not go missing together"
    assert fin_chl.any(), "oc025 is entirely NaN — the colour stage wrote nothing"
    v = cov[fin_cov]
    assert float(v.min()) > 0.0 and float(v.max()) <= 1.0 + tol_c, \
        f"chl_cov leaves (0, 1]: {v.min()} .. {v.max()}"

    # The three exact blocks. The bin is the first FULL colour bin, i.e. one
    # whose five days are all inside the source window.
    row = None
    for r in range(ctx.T_oc):
        b = ctx.b_oc + r
        s = bin_start(b, PENTAD_DAYS)
        if s >= ctx.oc_day0 and s + dt.timedelta(days=PENTAD_DAYS - 1) <= ctx.d_hi:
            row = r
            break
    assert row is not None, "the smoke axis holds no complete colour pentad"
    cells = oc_block_cells(*_smoke_blk(), SMOKE_OC_NLAT)
    yu, xu = SMOKE_OC_POINTS["uniform"]
    assert abs(chl[row, yu, xu] - SMOKE_OC_UNIFORM) < tol, \
        f"a block of {int(cells[yu])} cells all at 10**0.5 on all five days " \
        f"reads {chl[row, yu, xu]}, not {SMOKE_OC_UNIFORM}"
    assert abs(cov[row, yu, xu] - 1.0) < tol_c, \
        f"a fully observed block reads chl_cov {cov[row, yu, xu]}, not 1.0"

    ye, xe = SMOKE_OC_POINTS["empty"]
    assert not np.isfinite(chl[:, ye, xe]).any(), \
        "an all-fill block is not NaN"
    assert not np.isfinite(cov[:, ye, xe]).any(), \
        "an all-fill block reports coverage"

    ys, xs = SMOKE_OC_POINTS["single"]
    r0 = 0                                    # the single clear cell is on day 0
    want_cov = 1.0 / (int(cells[ys]) * PENTAD_DAYS)
    assert abs(chl[r0, ys, xs] - SMOKE_OC_SINGLE) < tol, \
        f"one clear cell on one day reads {chl[r0, ys, xs]}, not its own log10"
    assert abs(cov[r0, ys, xs] - want_cov) < tol_c, \
        f"one clear cell on one day reads chl_cov {cov[r0, ys, xs]}, not " \
        f"1/{int(cells[ys]) * PENTAD_DAYS}"

    yw, xw = SMOKE_OC_POINTS["wrap"]
    want_w = 0.5 * (SMOKE_OC_WRAP_W + SMOKE_OC_WRAP_E)
    assert abs(chl[row, yw, xw] - want_w) < tol, \
        f"the block at lon -180 reads {chl[row, yw, xw]}, not the mean of its " \
        f"two halves ({want_w}) — the longitude wrap dropped half the block"

    for name in ("south_pole", "north_pole"):
        yp, xp = SMOKE_OC_POINTS[name]
        assert int(cells[yp]) < int(cells[NLAT // 2]), \
            f"{name}: the block was not truncated ({int(cells[yp])} cells)"
        assert abs(chl[row, yp, xp] - SMOKE_OC_UNIFORM) < tol, \
            f"{name}: the truncated block's mean is wrong"
        assert abs(cov[row, yp, xp] - 1.0) < tol_c, \
            f"{name}: a fully observed TRUNCATED block must still read " \
            f"chl_cov 1.0 — its denominator is the cells it has, not six rows"
    return int(fin_chl.sum())


def _smoke_blk():
    d = 180.0 / SMOKE_OC_NLAT
    return int(round(0.25 / d)), int(round(0.25 / d))


def check_smoke(work, ctx, seed=None, seed_sha=None):
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

    n_oc = check_oc_smoke(work, ctx, d)

    # THE INHERITANCE, ASSERTED ON THE BYTES (E-077 §5). "It hard-linked" is an
    # intention; "the sha256 is the same and it is the same inode" is the fact.
    if seed and seed_sha:
        for g in BASE_GROUPS:
            old = os.path.join(seed, f"{BASE_STEM}_X_{g}.npy")
            new = group_file(work, g)
            assert os.path.samefile(old, new), \
                f"{g} is not the same file as the seed's"
            assert sha256(new) == seed_sha[g], \
                f"{g} changed between the seed and this build"
        assert os.path.exists(os.path.join(work, "seed.json"))
    return (f"T={want['g025'][0]} · T_oc={want['oc025'][0]} · n_live={n_live} · "
            f"{n_oc:,} observed colour values · {len(d.files)} keys · "
            f"groups {list(d.groups)}")


# ==================================================================== main ==
def main():
    ap = argparse.ArgumentParser(
        description="Build the family-7.1 global 0.25-degree pentad tensor "
                    "(recipe f7l1). See ml/plans/E070_family7_build.md and "
                    "ml/plans/E077_family7_ocean_colour.md.")
    ap.add_argument("--work", default=os.path.join(CACHE, "family7_l1"),
                    help="the build directory: memmaps, markers, progress.json")
    ap.add_argument("--seed-from", default="",
                    help="a FINISHED f7l0 work dir to inherit from: its three "
                         "group files are hard-linked under the f7l1 stem and "
                         "its markers, specs and norm.npz copied, so only "
                         "`occci` is built. Must be on the same filesystem; a "
                         "missing directory is an error, not a full rebuild.")
    ap.add_argument("--stage", default="all",
                    choices=["all"] + STAGES,
                    help="one stage, or `all`. Order is fixed: ncep needs sst "
                         "for the sst repair, norm needs everything including "
                         "occci.")
    ap.add_argument("--oc-source", default="occci",
                    choices=sorted(OC_SOURCES),
                    help="which colour archive. `occci` is the ESA CCI merged "
                         "record (the recipe). `globcolour` is a DIFFERENT "
                         "merge and a deliberate decision after a preflight "
                         "failure, never an automatic fallback (E-077 §2).")
    ap.add_argument("--oc-start", default=str(OC_START),
                    help="the first day of the colour record. The default is "
                         "the archive's own first day; changing it on a real "
                         "build shifts every oc025 row's date, and it exists "
                         "for the smoke.")
    ap.add_argument("--oc-preflight", action="store_true",
                    help="fetch and open ONE real OC-CCI file, print its grid "
                         "and exit. Run before a build, not after.")
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
    if a.oc_preflight:
        oc_preflight(ctx)
        return 0
    if a.seed_from:
        seed_from(ctx, a.seed_from)
    print(f"axis      {bin_start(ctx.bins[0], 5)} .. "
          f"{bin_start(ctx.bins[-1], 5)}  T={ctx.T} pentad bins "
          f"(bins {ctx.b_lo}..{ctx.b_hi}, recipe {RECIPE}, inheriting "
          f"{BASE_RECIPE})")
    print(f"colour    T_oc={ctx.T_oc} bins from {bin_start(ctx.b_oc, 5)} "
          f"(oc_bin_first={ctx.b_oc}, source {ctx.oc_source})")
    print(f"grids     g025 {NLAT}x{NLON} · g100 {NLAT1}x{NLON1} · "
          f"rg100 {NLAT1}x{NLON1} · oc025 {NLAT}x{NLON}")
    print(f"channels  g025 {NCHAN['g025']} · g100 {NCHAN['g100']} · "
          f"rg100 {NCHAN['rg100']} · oc025 {NCHAN['oc025']}")
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
    if any(s in ("glorys", "sst", "ncep", "rg", "occci") for s in stages):
        disk_guard(ctx.work, sizes)
    # ONE REAL FILE BEFORE FOUR HUNDRED GIGABYTES (ml/CLAUDE.md §0.3). The
    # preflight runs at the head of any run that will do colour work and is
    # skipped only when the sources are local (`--source-dir`, i.e. the tests).
    if "occci" in stages and not marked(ctx.work, "occci") and not a.source_dir:
        oc_preflight(ctx)
    run_stages(ctx, stages)
    return 0


if __name__ == "__main__":
    sys.exit(main())

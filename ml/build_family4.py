#!/usr/bin/env python3
"""Family-4 tensor: the 0.25-degree North Atlantic at PENTAD cadence.

E-034 step 4. Output `ml/cache/family4_na025_pentad.npz` (recipe f4r1), or
`family4_na025_pentad_r2.npz` at `--rev r2` (f4r2, E-041: + the sst channel).

WHY A NEW FAMILY AND NOT AN EDIT TO build_family3.py (E-034 §5). A pentad
tensor and a monthly tensor must never be silently mixed: every result in
EXPERIMENTS.md was measured on `f3r1`, and a run that trained on one while
believing the other would produce numbers nobody could reproduce or retract.
So this is a separate file, with its own `RECIPE_REV`, its own output name and
its own sha. `build_family3.py` is untouched.

WHAT IT BUYS, restated so the build can be checked against its own claim: the
transport probe sits at a measured STATE/LABEL ceiling of 0.631, not a
representation tax, and no model change can move it. Pentad cadence multiplies
the RAPID labels by ~6.1x and the Florida cable's by ~5.9x
(`ml/build_truth_pentad.py` measured both from the live archives). This tensor
is the state axis that meets those labels in the same bins.

THE AXIS IS THE POINT, so it is stated once and shared by construction.
Pentads are fixed 5-day bins counted from 1982-01-01, index =
floor(days_since_epoch / 5) — identical to `ml/build_truth_pentad.py` and
`ml/aggregate_cadence.py`, which is what makes a pentad label the target of a
pentad state with no re-alignment step anywhere.

THE GRID IS FAMILY-3's, EXACTLY, and this is a precondition rather than an
aspiration. `base025_na.npz` was opened on 2026-08-16 and reports lats
0.0..70.0 (281) and lons -100.0..20.0 (481) — samples ON multiples of 0.25,
not cell centres. `aggregate_cadence.py --grid-align point` reproduces those
axes exactly; this build ASSERTS it and refuses otherwise. A half-cell offset
would leave every stencil geometry, the AMOC eval mask and the corridor
definitions quietly describing different pixels than they name, and it would
be invisible in every plot.

CHANNELS are family-3's 39, in family-3's order, imported from that module so
there is ONE definition — plus, at recipe r2 (E-041), an APPENDED 40th. Per
E-034 §2 the cadence policy differs per channel group, and that is the whole
substance of this file:

  base (3)  cur_speed, log_mld, ssh — from the pentad GLORYS12 aggregation.
            cur_speed = hypot(mean_uo, mean_vo), built from the BINNED
            COMPONENTS: a mean of magnitudes is not the magnitude of the mean,
            and averaging speed would inflate quiet bins where the current
            reverses. log_mld is log10(mlotst) — MEASURED against family-3
            rather than assumed (January 1993 peaks at 2439 m under log10, a
            Labrador Sea convection event; under natural log it would read
            30 m, which is not a January).

  rg (32)   RG-Argo T/S at 16 pressures. The product is MONTHLY, and a month
            is ~6 pentads. E-034 §4 chose, explicitly: ONE LIVE PENTAD PER
            MONTH, `missing` in the other five. Forward-filling was rejected
            because it tells the model the subsurface was observed on days it
            was not, and the architecture's whole claim is that missingness is
            information — the `missing` token is distinct from the `mask`
            token by design, and that distinction was measured to matter.
            The nominal timestamp is the 15th of the month: RG carries no
            within-month time, and the 15th is the month's midpoint, so the
            live pentad is the one a monthly mean is most nearly centred on.

  wind (4)  NCEP R1 tau_x, tau_y (sign flipped, so positive = stress ON the
            ocean) as the PENTAD mean of the dailies, and tau_x_std/tau_y_std
            as the WITHIN-PENTAD standard deviation. The std is the one
            channel that is strictly better at this cadence: family-3's is a
            within-MONTH sigma, which mixes the storm band with the seasonal
            cycle. A 5-day window is a storminess measure. It is computed from
            the dailies directly and never from a mean, because a standard
            deviation is not aggregable from one.

  sst (1)   OISST v2.1 daily 0.25 degree, on this grid already
            (`ml/fetch_sst_na.py` does the interpolation), as the NaN-aware
            mean of the bin's days. RECIPE r2 ONLY — r1 is the 39-channel
            tensor #386/#387 are training on and stays buildable and
            unchanged. It is APPENDED so channels 0..38 keep their published
            indices and an r1/r2 pair is diffable channel by channel.
            Why: `rg_t` is the only other temperature and it starts in 2004,
            so 22 of the 43 years carry none at all, and even inside the Argo
            era it is one live bin per month. SST is live in ~100% of bins
            over the whole axis.

The time axis starts at 1982-01-01 like family-3's, not at GLORYS12's 1993:
wind covers the gap, the Florida cable's 1982-92 decade becomes usable truth,
and pre-1993 base channels are simply missing tokens.

STORAGE. float16, per E-034 §5 — the fields are anomaly-space and normalised
to ~N(0,1), where fp16 carries ~3 decimal digits, far below observational
error. Dense, the tensor is [3142, 281, 481, 39]: 66.3 GB at float32 and
33.1 GB at float16. It therefore does NOT fit this sandbox (~28 GB free) and
is a BOX build; `--dry-run` prints the arithmetic without spending anything,
and `--max-bins` builds a prefix for exercising the path.

Run:
  python3 ml/build_family4.py --dry-run
  python3 ml/build_family4.py --pentad-dir ml/cache/glorys_pentad
  python3 ml/build_family4.py --pentad-dir ... --max-bins 40   # smoke
  python3 ml/build_family4.py --rev r2 --pentad-dir ...        # E-041, +sst
"""
import argparse
import datetime as dt
import glob
import os
import shutil
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import build_family3 as f3                                    # noqa: E402
from aggregate_cadence import EPOCH, bin_index, bin_start     # noqa: E402

CACHE = os.path.join(HERE, "cache")
OUT_NPZ = os.path.join(CACHE, "family4_na025_pentad.npz")
MEMMAP = os.path.join(CACHE, "family4_na025_pentad_build.npy")
TRUTH_PENTAD = os.path.join(CACHE, "truth_pentad.npz")

PENTAD_DAYS = 5
START = dt.date(1982, 1, 1)
END = dt.date(2024, 12, 31)

# E-038: family 5 is THIS builder at days=1 — "build_family4.py with
# PENTAD_DAYS = 1 and its own RECIPE_REV, not a copy" — so the cadence is a
# parameter and everything family-specific hangs off it. The family-4 path
# (days=5) is byte-identical to what built the tensors E-038a/b train on.
#
# E-041: the RECIPE REVISION is the second axis of this table. r2 appends SST
# (below); r1 is kept, buildable and byte-identical, because #386/#387 are in
# flight on it and every number in EXPERIMENTS.md before E-041 was measured on
# it. The output NAME carries the rev, so an r1 and an r2 tensor can sit on
# one box without either overwriting the other — and `ml-train.yml` derives
# $TENSOR from the `tensor` input verbatim, so these file stems ARE the
# dispatch values (family4_na025_pentad_r2 -> ml/cache/family4_na025_pentad_r2.npz).
CADENCE = {
    5: dict(name="pentad", truth="truth_pentad.npz", revs={
        "r1": dict(recipe="f4r1", out="family4_na025_pentad.npz"),
        "r2": dict(recipe="f4r2", out="family4_na025_pentad_r2.npz"),
    }),
    1: dict(name="daily", truth="truth_daily.npz", revs={
        "r1": dict(recipe="f5r1", out="family5_na025_daily.npz"),
        "r2": dict(recipe="f5r2", out="family5_na025_daily_r2.npz"),
    }),
}

# ONE definition of the channel set, imported rather than restated.
CHANS, LEVELS = f3.CHANS, f3.LEVELS
C_BASE, C_RG, C_WIND, NC = f3.C_BASE, f3.C_RG, f3.C_WIND, f3.NC

# E-041. SST is APPENDED, and the appending is the whole safety argument:
# channels 0..38 keep the indices every published result was measured at,
# `build_family3.py` is not touched, and an r1 and an r2 tensor are diffable
# channel by channel. Channel ORDER is not information the model uses —
# identity comes from `chan_emb` (ml/model.py), which embeds the channel
# INDEX and is trained from scratch per run — so "last" costs nothing.
#
# WHY SST AT ALL (E-041): of the 39 channels, the only temperature is Argo
# `rg_t`, which starts in 2004 and is live in one bin per month. 1982-2003 —
# 22 of the 43 years — carries no temperature at all. OISST is on the tensor's
# own grid, daily, and live in 100% of the bins across the whole axis.
CHANS_R1 = list(CHANS)
CHANS_R2 = list(CHANS) + ["sst"]
C_SST = NC                            # channel 40 (index 39), r2 only
CHANS_BY_REV = {"r1": CHANS_R1, "r2": CHANS_R2}

SST_NAME = "sst_na025"       # ml/fetch_sst_na.py's output dir, under CACHE

# Kept for readers who grep for it: the live recipe strings are in CADENCE
# above (one per cadence x rev), and THEY are the skip guard. Bump a recipe on
# ANY change to what the build writes.
RECIPE_REV = "f4r1"


def pentad_days(b):
    """The five calendar dates bin `b` covers."""
    s = bin_start(b, PENTAD_DAYS)
    return [s + dt.timedelta(days=k) for k in range(PENTAD_DAYS)]


def truth_path(days):
    """Where this cadence's labels live. TRUTH_PENTAD stays the module-level
    name for days=5 because the tests monkeypatch it; days=1 derives from the
    same directory so a test that moves one moves both."""
    if days == PENTAD_DAYS:
        return TRUTH_PENTAD
    return os.path.join(os.path.dirname(TRUTH_PENTAD), CADENCE[days]["truth"])


# ---------------------------------------------------------------- base ----
def load_pentad_base(pentad_dir, days=PENTAD_DAYS):
    """The aggregator's output, with the grid checked before anything else."""
    cad = CADENCE[days]["name"]
    idx_path = os.path.join(pentad_dir, "index.npz")
    if not os.path.exists(idx_path):
        sys.exit(f"no index.npz in {pentad_dir} — run:\n"
                 f"  python3 ml/aggregate_cadence.py --hf-repo earth-tensors "
                 f"--cadence {cad} --out {pentad_dir} --bin-deg 0.25")
    idx = np.load(idx_path)
    if int(idx["cadence_days"]) != days:
        sys.exit(f"{pentad_dir} is cadence_days={int(idx['cadence_days'])}, "
                 f"not {days} — this build is --days {days} ({cad})")
    if str(idx["epoch"]) != str(EPOCH):
        sys.exit(f"{pentad_dir} epoch {str(idx['epoch'])!r} != {str(EPOCH)!r} "
                 f"— the state and label axes would not share bins")
    if "lat" not in idx:
        sys.exit(f"{pentad_dir} was built without --bin-deg (native 1/12 deg "
                 f"grid). This is a 0.25 deg tensor; rebuild with "
                 f"--bin-deg 0.25 --grid-align point.")
    arrs = {}
    for v in ("uo", "vo", "mlotst", "zos"):
        p = os.path.join(pentad_dir, f"{cad}_mean_{v}.npy")
        if not os.path.exists(p):
            sys.exit(f"missing {p}")
        arrs[v] = np.load(p, mmap_mode="r")
    return idx, arrs


def check_grid(lats, lons):
    """Refuse a grid that is not family-3's, while it has cost nothing.

    ml/CLAUDE.md §0.3 and §5.16: a precondition that depends only on the
    inputs must fire before the expensive part, not after 33 GB of writes.
    """
    ref = os.path.join(CACHE, "base025_na.npz")
    if not os.path.exists(ref):
        print("  ::warning:: base025_na.npz absent — cannot verify the grid "
              "against family 3. Shapes are self-consistent but comparability "
              "is UNCHECKED; fetch it from data-cache-v1 to close this.")
        return
    d = np.load(ref)
    ok = (len(lats) == len(d["lats"]) and len(lons) == len(d["lons"])
          and np.allclose(lats, d["lats"]) and np.allclose(lons, d["lons"]))
    if not ok:
        sys.exit(
            f"GRID MISMATCH with family 3.\n"
            f"  family 4: {len(lats)}x{len(lons)}  lat {lats[0]}..{lats[-1]}  "
            f"lon {lons[0]}..{lons[-1]}\n"
            f"  family 3: {len(d['lats'])}x{len(d['lons'])}  "
            f"lat {d['lats'][0]}..{d['lats'][-1]}  "
            f"lon {d['lons'][0]}..{d['lons'][-1]}\n"
            f"Re-run aggregate_cadence.py with --grid-align point. A "
            f"half-cell offset makes every stencil and the AMOC eval mask "
            f"name different pixels than they describe.")
    print(f"  grid verified against family 3: {len(lats)}x{len(lons)}")


# ------------------------------------------------------------------ rg ----
def fill_rg_pentad(X, bins, lats, lons, days=PENTAD_DAYS):
    """RG monthly -> ONE live pentad per month (E-034 §4), missing elsewhere.

    Returns the number of live pentads written. Reuses family-3's readers and
    its NaN-aware bilinear so the two tensors cannot disagree about what the
    RG field IS — only about which timesteps carry it.
    """
    import netCDF4 as ncdf
    tf, sf = f3.rg_file("RG_T"), f3.rg_file("RG_S")
    if not tf or not sf:
        print("  ::warning:: RG cubes not cached and NOT fetched here — the "
              "rg channels will be entirely missing tokens. Seed ml/cache/rg "
              "from the data-cache-v1 release (rg.tar.*) for a real build.")
        return 0
    dT, dS = ncdf.Dataset(tf), ncdf.Dataset(sf)
    press = np.array(dT.variables["PRESSURE"][:])
    lidx = [int(np.argmin(np.abs(press - p))) for p in LEVELS]
    L = len(LEVELS)
    wy = f3.lin_weights(np.array(dT.variables["LATITUDE"][:]), lats)
    rg_lon = np.array(dT.variables["LONGITUDE"][:])
    lon360 = np.where(lons < 20.0, lons + 360.0, lons)
    wx = f3.lin_weights(rg_lon, lon360, wrap_period=360.0)
    mean_t = np.ma.filled(dT.variables["ARGO_TEMPERATURE_MEAN"][lidx], np.nan)
    mean_s = np.ma.filled(dS.variables["ARGO_SALINITY_MEAN"][lidx], np.nan)

    # row for the pentad containing the 15th — the month's midpoint, because
    # RG carries no within-month time and a monthly mean is most nearly
    # centred there.
    row_of = {b: i for i, b in enumerate(bins)}

    def live_row(y, m):
        # ONE live bin per month at every cadence (E-034 §4): the bin holding
        # the 15th. At days=5 that is one pentad in six; at days=1, one day in
        # ~30 — which is why the rg term of the Chinchilla inventory does not
        # scale with cadence (E-038 §2b).
        return row_of.get(bin_index(dt.date(y, m, 15), days))

    def write(row, at, as_):
        for k in range(L):
            X[row, :, :, C_BASE + k] = f3.interp2_nan(mean_t[k] + at[k], wy, wx)
            X[row, :, :, C_BASE + L + k] = f3.interp2_nan(mean_s[k] + as_[k], wy, wx)

    n = 0
    anom_t = dT.variables["ARGO_TEMPERATURE_ANOMALY"]
    anom_s = dS.variables["ARGO_SALINITY_ANOMALY"]
    nbase = anom_t.shape[0]
    for k in range(nbase):
        y, m = 2004 + k // 12, k % 12 + 1
        r = live_row(y, m)
        if r is None:
            continue
        write(r, np.ma.filled(anom_t[k][lidx], np.nan),
              np.ma.filled(anom_s[k][lidx], np.nan))
        n += 1
    dT.close(), dS.close()

    exts = sorted(set(os.path.basename(p).split(".")[0].split("_")[1]
                      for p in glob.glob(os.path.join(CACHE, "rg", "RG_2*.nc*"))))
    for ym in exts:
        r = live_row(int(ym[:4]), int(ym[4:]))
        if r is None:
            continue
        dE = ncdf.Dataset(f3.rg_file(f"RG_{ym}"))
        write(r, np.ma.filled(dE.variables["ARGO_TEMPERATURE_ANOMALY"][0][lidx], np.nan),
              np.ma.filled(dE.variables["ARGO_SALINITY_ANOMALY"][0][lidx], np.nan))
        dE.close()
        n += 1
    print(f"  rg: {n} live pentads ({n} months x 1); the other "
          f"{len(bins) - n} bins carry missing tokens, by design (E-034 §4)")
    return n


# ---------------------------------------------------------------- wind ----
def fill_wind_pentad(X, bins, lats, lons, min_days=3, days=PENTAD_DAYS):
    """NCEP R1 daily -> pentad mean + WITHIN-PENTAD std, channels 35..38.

    The std is why this cannot be derived from family-3's monthly channels: a
    standard deviation is not aggregable from a mean. It is computed here over
    the days of each 5-day bin, which is a storminess measure rather than a
    seasonal one.
    """
    if days == 1:
        return fill_wind_daily(X, bins, lats, lons)
    import netCDF4 as ncdf
    daily = os.path.join(CACHE, "wind_daily")
    psl = ("https://downloads.psl.noaa.gov/Datasets/ncep.reanalysis/"
           "surface_gauss")
    thredds = ("https://psl.noaa.gov/thredds/fileServer/Datasets/"
               "ncep.reanalysis/surface_gauss")
    row_of = {b: i for i, b in enumerate(bins)}
    years = sorted({bin_start(b, PENTAD_DAYS).year for b in bins}
                   | {(bin_start(b, PENTAD_DAYS)
                       + dt.timedelta(days=PENTAD_DAYS - 1)).year for b in bins})
    wy = wx = None
    n = 0
    # A bin can straddle a year boundary, so days are accumulated per bin
    # across files exactly as aggregate_cadence.py does it.
    acc, acc2, cnt = {}, {}, {}
    for y in years:
        fields = {}
        for var in ("uflx", "vflx"):
            path = os.path.join(daily, f"{var}.sfc.gauss.{y}.nc")
            if not os.path.exists(path):
                try:
                    f3.fetch(f"{psl}/{var}.sfc.gauss.{y}.nc", path,
                             mirrors=(f"{thredds}/{var}.sfc.gauss.{y}.nc",))
                except Exception as e:                    # noqa: BLE001
                    print(f"  ::warning:: wind {y} {var} unavailable "
                          f"({str(e)[:80]}) — that year's wind is missing")
                    fields = None
                    break
            nc = ncdf.Dataset(path)
            if wy is None:
                wy = f3.lin_weights(np.array(nc.variables["lat"][:]), lats)
                wx = f3.lin_weights(np.array(nc.variables["lon"][:]),
                                    np.where(lons < 0, lons + 360.0, lons),
                                    wrap_period=360.0)
            tv = nc.variables["time"]
            dates = ncdf.num2date(tv[:], tv.units,
                                  only_use_cftime_datetimes=False)
            fields[var] = ([dt.date(d.year, d.month, d.day) for d in dates],
                           np.ma.filled(nc.variables[var][:], np.nan))
            nc.close()
        if not fields:
            continue
        for ci, var in enumerate(("uflx", "vflx")):
            dates, vals = fields[var]
            for i, d in enumerate(dates):
                b = bin_index(d, PENTAD_DAYS)
                if b not in row_of:
                    continue
                # sign flip here, once, so the accumulator holds stress ON the
                # ocean and the std is the std of that same quantity
                v = -vals[i]
                key = (b, ci)
                if key not in acc:
                    acc[key] = np.zeros(v.shape, np.float64)
                    acc2[key] = np.zeros(v.shape, np.float64)
                    cnt[key] = np.zeros(v.shape, np.int32)
                ok = np.isfinite(v)
                acc[key][ok] += v[ok]
                acc2[key][ok] += v[ok] ** 2
                cnt[key] += ok
        print(f"  wind {y}: read", flush=True)

    for (b, ci), a in sorted(acc.items()):
        c = cnt[(b, ci)]
        good = c >= min_days
        with np.errstate(invalid="ignore"):
            mu = a / np.maximum(c, 1)
            sd = np.sqrt(np.maximum(acc2[(b, ci)] / np.maximum(c, 1) - mu ** 2, 0))
        r = row_of[b]
        X[r, :, :, C_BASE + C_RG + ci] = f3.interp2_nan(
            np.where(good, mu, np.nan), wy, wx)
        X[r, :, :, C_BASE + C_RG + 2 + ci] = f3.interp2_nan(
            np.where(good, sd, np.nan), wy, wx)
        n += ci == 0
    print(f"  wind: {n} pentads written (mean + within-pentad std)")
    return n


def fill_wind_daily(X, bins, lats, lons, sigma_window=5, min_sigma_days=3):
    """NCEP R1 daily -> the day's stress + a CENTRED 5-day rolling sigma.

    THE SIGMA DECISION (E-038 daily arm). tau_x_std/tau_y_std are family 4's
    within-pentad standard deviation — five daily values per bin. At days=1 a
    within-bin sigma is IDENTICALLY ZERO (one value has no spread), so left
    alone family 5 would ship two dead channels out of 39 and every downstream
    number would quietly carry them.

    The formulation chosen KEEPS FAMILY 4's MEANING rather than inventing a
    new channel: sigma over the five calendar days CENTRED on the bin's
    midpoint. For a pentad bin the centred window IS the bin — the same five
    days — so at days=5 this is the identical quantity, and at days=1 it is
    that quantity sampled every day instead of every fifth day. The two
    cadences stay comparable by construction, which is what E-038's capacity x
    cadence matrix needs. `tests/test_e034_family5.py` pins the identity: the
    daily sigma at a pentad's midpoint equals the pentad tensor's sigma for
    that bin.

    Same estimator as family 4's: population (ddof=0), NaN-aware, written only
    where >= 3 of the 5 window days are present (family 4's min_days). The
    mean channels carry the day itself — a one-day bin's mean IS the day,
    matching `aggregate_cadence --cadence daily` (min_days=1).

    MEMORY, stated because the daily axis makes everything a memory question:
    all per-day native fields are held at once — 15,710 days x 2 vars x 94x192
    float32 = 2.3 GB — which buys the +-2-day windows across year boundaries
    without a chunked walk. The interpolation to 0.25 deg happens per bin and
    never materialises more than one field.
    """
    import netCDF4 as ncdf
    daily = os.path.join(CACHE, "wind_daily")
    psl = ("https://downloads.psl.noaa.gov/Datasets/ncep.reanalysis/"
           "surface_gauss")
    thredds = ("https://psl.noaa.gov/thredds/fileServer/Datasets/"
               "ncep.reanalysis/surface_gauss")
    row_of = {b: i for i, b in enumerate(bins)}
    half = sigma_window // 2
    d_lo = bin_start(bins[0], 1) - dt.timedelta(days=half)
    d_hi = bin_start(bins[-1], 1) + dt.timedelta(days=half)
    years = list(range(d_lo.year, d_hi.year + 1))

    fields = {}                      # date -> [u, v] native, sign-flipped
    wy = wx = None
    for y in years:
        for ci, var in enumerate(("uflx", "vflx")):
            path = os.path.join(daily, f"{var}.sfc.gauss.{y}.nc")
            if not os.path.exists(path):
                try:
                    f3.fetch(f"{psl}/{var}.sfc.gauss.{y}.nc", path,
                             mirrors=(f"{thredds}/{var}.sfc.gauss.{y}.nc",))
                except Exception as e:                    # noqa: BLE001
                    print(f"  ::warning:: wind {y} {var} unavailable "
                          f"({str(e)[:80]}) — that year's wind is missing")
                    continue
            nc = ncdf.Dataset(path)
            if wy is None:
                wy = f3.lin_weights(np.array(nc.variables["lat"][:]), lats)
                wx = f3.lin_weights(np.array(nc.variables["lon"][:]),
                                    np.where(lons < 0, lons + 360.0, lons),
                                    wrap_period=360.0)
            tv = nc.variables["time"]
            dates = ncdf.num2date(tv[:], tv.units,
                                  only_use_cftime_datetimes=False)
            vals = np.ma.filled(nc.variables[var][:], np.nan)
            nc.close()
            for i, dd in enumerate(dates):
                d = dt.date(dd.year, dd.month, dd.day)
                if d < d_lo or d > d_hi:
                    continue
                # sign flip here, once, exactly as the pentad path does it
                fields.setdefault(d, [None, None])[ci] = \
                    (-vals[i]).astype(np.float32)
        print(f"  wind {y}: read", flush=True)

    n = 0
    for b in bins:
        d0 = bin_start(b, 1)
        f = fields.get(d0)
        if f is None or f[0] is None or f[1] is None:
            continue
        win_days = [d0 + dt.timedelta(days=k) for k in range(-half, half + 1)]
        r = row_of[b]
        for ci in range(2):
            stack = np.stack([fields[d][ci] for d in win_days
                              if d in fields and fields[d][ci] is not None])
            cnt = np.isfinite(stack).sum(0)
            with np.errstate(invalid="ignore"), warnings_suppressed():
                sd = np.nanstd(stack, axis=0)             # ddof=0, like f4
            sd = np.where(cnt >= min_sigma_days, sd, np.nan)
            X[r, :, :, C_BASE + C_RG + ci] = f3.interp2_nan(f[ci], wy, wx)
            X[r, :, :, C_BASE + C_RG + 2 + ci] = f3.interp2_nan(sd, wy, wx)
        n += 1
    print(f"  wind: {n} days written (day value + centred {sigma_window}-day "
          f"sigma, >= {min_sigma_days} days present)")
    return n


class warnings_suppressed:
    """nanstd over an all-NaN cell warns; land is all-NaN by design."""

    def __enter__(self):
        import warnings
        self._c = warnings.catch_warnings()
        self._c.__enter__()
        import warnings as w
        w.simplefilter("ignore")

    def __exit__(self, *exc):
        return self._c.__exit__(*exc)


# ----------------------------------------------------------------- sst ----
def fill_sst(X, bins, lats, lons, days=PENTAD_DAYS, sst_dir=None):
    """OISST daily SST -> the appended `sst` channel (recipe r2 only).

    THE ARTIFACT is `ml/fetch_sst_na.py`'s: int16 (NDAYS, 281, 481) at
    0.01 degC with nodata -32768, day-major, ALREADY on this tensor's grid
    (the fetcher interpolates OISST's half-cell-offset centres onto family
    3's axes with the same `f3.interp2_nan` the wind channel uses). Nothing
    here regrids; this function only bins in time and decodes.

    CADENCE. days=1 takes the day itself. days=5 takes the NaN-AWARE MEAN of
    the bin's five days: nodata rows and nodata cells are excluded, and a cell
    with no valid day in the bin stays NaN. That NaN is the missing token —
    the same way every other fill_* in this file expresses missingness, since
    the memmap starts as NaN and the mask/stat pass counts only `isfinite`.
    Land must therefore never decode to a temperature: -32768 at scale 0.01
    is -327.68 degC, which would be obvious, but 0 (a fresh memmap, or a
    truncated read) is 0.00 degC and would not be — so the decode is
    where-based and the sentinel never reaches the arithmetic.

    WHY A MEAN AND NOT A SAMPLE. Unlike rg, SST is observed EVERY day, so a
    bin has five real observations and their mean is the bin's state; taking
    one day would throw four fifths of the information away and alias the
    storm band. Unlike the wind std, a mean is aggregable from dailies, which
    is why no second cadence-specific estimator is needed.

    COVERAGE is the point of the channel (E-041): OISST runs 1982-present, so
    this is live in ~100% of the bins across the whole axis, where `rg_t` —
    the tensor's only other temperature — starts in 2004 and is live in one
    bin per month. Returns the number of rows that received any value.
    """
    # Resolved from CACHE at call time, not at import: the tests redirect
    # CACHE, and ml-train.yml makes ml/cache a symlink to the box-persistent
    # /opt/earth-cache, where it seeds the Hub's sst_na025/ folder.
    sst_dir = sst_dir or os.path.join(CACHE, SST_NAME)
    idx_path = os.path.join(sst_dir, "index.npz")
    npy = os.path.join(sst_dir, "sst_daily_na.npy")
    if not (os.path.exists(idx_path) and os.path.exists(npy)):
        # Same posture as fill_rg_pentad's missing cubes: loud, and the
        # channel is missing tokens rather than invented values.
        print(f"  ::warning:: no SST artifact in {sst_dir} — the sst channel "
              f"would be ENTIRELY missing tokens. Seed it from the Hub "
              f"(sst_na025/, published by .github/workflows/sst-na-bake.yml) "
              f"before building r2 for real.")
        return 0
    idx = np.load(idx_path)
    if str(idx["epoch"]) != str(EPOCH):
        sys.exit(f"{idx_path} epoch {str(idx['epoch'])!r} != {str(EPOCH)!r} — "
                 f"the SST rows would land on the wrong bins")
    if int(idx["cadence_days"]) != 1:
        sys.exit(f"{idx_path} is cadence_days={int(idx['cadence_days'])}; the "
                 f"SST artifact must be DAILY — every coarser cadence is "
                 f"derived here, never fetched (aggregate_cadence.py's rule)")
    src_lat = np.asarray(idx["lat"], np.float64)
    src_lon = np.asarray(idx["lon"], np.float64)
    if (len(src_lat) != len(lats) or len(src_lon) != len(lons)
            or not np.allclose(src_lat, lats) or not np.allclose(src_lon, lons)):
        # A precondition that depends only on the inputs, checked while the
        # inputs are all it has cost (ml/CLAUDE.md §5.16). fetch_sst_na.py
        # already put the field on this grid; if it did not, a half-cell
        # offset would be invisible in every plot.
        sys.exit(
            f"SST GRID MISMATCH with the tensor.\n"
            f"  tensor: {len(lats)}x{len(lons)}  lat {lats[0]}..{lats[-1]}  "
            f"lon {lons[0]}..{lons[-1]}\n"
            f"  sst:    {len(src_lat)}x{len(src_lon)}  "
            f"lat {src_lat[0]}..{src_lat[-1]}  "
            f"lon {src_lon[0]}..{src_lon[-1]}\n"
            f"Rebuild the artifact with ml/fetch_sst_na.py against this "
            f"window's base025_na.npz.")
    scale = float(idx["scale"])
    nodata = int(idx["nodata"])
    src = np.load(npy, mmap_mode="r")
    day_row = {int(b): i for i, b in enumerate(idx["bin_index"])}
    has = np.asarray(idx["has_data"], bool)
    if src.shape != (len(has), len(lats), len(lons)):
        sys.exit(f"{npy} is {src.shape}, but index.npz describes "
                 f"{(len(has), len(lats), len(lons))} — the artifact is "
                 f"inconsistent with its own index")

    n = 0
    n_days = 0
    for r, b in enumerate(bins):
        d0 = bin_start(b, days)
        rows = []
        for k in range(days):
            j = day_row.get((d0 - EPOCH).days + k)
            if j is not None and has[j]:
                rows.append(j)
        if not rows:
            continue                      # missing token, by construction
        raw = np.asarray(src[rows], np.int16)
        # nodata NEVER enters the arithmetic: it becomes NaN first, and the
        # mean is taken over the finite days only.
        vals = np.where(raw == nodata, np.nan,
                        raw.astype(np.float32) * np.float32(scale))
        with np.errstate(invalid="ignore"), warnings_suppressed():
            m = np.nanmean(vals, axis=0)   # all-NaN cell -> NaN -> missing
        X[r, :, :, C_SST] = m
        n += 1
        n_days += len(rows)
    print(f"  sst: {n}/{len(bins)} bins carry SST "
          f"({n_days} daily field(s) folded, NaN-aware {days}-day mean); "
          f"{len(bins) - n} missing")
    return n


# --------------------------------------------------------------- truth ----
def truth_pentad(bins, days=PENTAD_DAYS, path=None):
    """(row, transport) pairs on THIS axis, from build_truth_pentad.py."""
    out = {}
    path = path or truth_path(days)
    if not os.path.exists(path):
        print(f"  ::warning:: {path} absent — run "
              f"ml/build_truth_pentad.py. No truth attached.")
        return out
    d = np.load(path)
    if str(d["epoch"]) != str(EPOCH) or int(d["pentad_days"]) != days:
        sys.exit(f"{os.path.basename(path)} has epoch {str(d['epoch'])!r}/"
                 f"{int(d['pentad_days'])}d — refusing to attach labels from a "
                 f"different axis to a state tensor.")
    row_of = {b: i for i, b in enumerate(bins)}
    for k in d.files:
        if not k.startswith("truth_"):
            continue
        arr = np.asarray(d[k], np.float64)
        keep = [(row_of[int(b)], v) for b, v in arr if int(b) in row_of]
        out[k] = np.array(keep, np.float32).reshape(-1, 2)
        print(f"  {k}: {len(keep)}/{len(arr)} pentad labels inside the axis")
    return out


def missing_truth_keys(have, path=None):
    """Label keys the truth file OFFERS that a cached tensor does not carry.

    The recipe string is a claim about the CODE that built the tensor. It says
    nothing about what was on disk beside that code — and run #365 built the
    33 GB pentad tensor on a box where `truth_pentad.npz` did not yet exist,
    so it wrote a physically perfect state tensor with **no transport labels
    at all**. The recipe guard then did exactly what it was written to do and
    skipped the rebuild on the next run, which would have trained for twenty
    hours and died in `probe_kfold` on `KeyError: 'rapid'` — the one number
    E-038 exists to produce.

    Verify the ARTEFACT, not the intention (ml/CLAUDE.md §0.1). Reading the
    npz directory and the truth file's key list costs milliseconds; both are
    headers, not data. Returns [] when there is no truth file to compare
    against, so a box that legitimately has no labels is not put into a
    rebuild loop.
    """
    if path is None:
        path = TRUTH_PENTAD
    if not os.path.exists(path):
        return []
    want = {k for k in np.load(path).files if k.startswith("truth_")}
    if "truth_rapid" in want:
        want.add("rapid")            # the alias the trainer actually reads
    return sorted(want - set(have))


# ---------------------------------------------------------------- main ----
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pentad-dir",
                    default=os.path.join(CACHE, "glorys_pentad"),
                    help="aggregate_cadence.py output (index.npz + "
                         "pentad_mean_<var>.npy)")
    ap.add_argument("--days", type=int, default=PENTAD_DAYS,
                    choices=sorted(CADENCE),
                    help="bin width: 5 = family 4 (pentad), 1 = family 5 "
                         "(daily). Everything family-specific — recipe, "
                         "output name, truth file, storage layout — follows.")
    ap.add_argument("--rev", default="r1", choices=("r1", "r2"),
                    help="recipe revision: r1 = the 39 family-3 channels "
                         "(f4r1/f5r1, what #386/#387 train on); r2 = those 39 "
                         "plus appended `sst` from ml/fetch_sst_na.py's "
                         "artifact (f4r2/f5r2). It is a FLAG and never "
                         "inferred from what happens to be on disk — a recipe "
                         "the filesystem decides is not a recipe.")
    ap.add_argument("--sst-dir", default=None,
                    help="ml/fetch_sst_na.py's output (sst_daily_na.npy + "
                         "index.npz); r2 only. Default ml/cache/sst_na025, "
                         "which is where ml-train.yml seeds the Hub's "
                         "sst_na025/ folder.")
    ap.add_argument("--out", default=None)
    ap.add_argument("--memmap", default=None)
    ap.add_argument("--start", default=str(START))
    ap.add_argument("--end", default=str(END))
    ap.add_argument("--max-bins", type=int, default=0,
                    help="build only the first N bins — for exercising the "
                         "path without the full 33 GB")
    ap.add_argument("--dry-run", action="store_true",
                    help="print the axis and the byte arithmetic, spend nothing")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--keep-memmap", action="store_true")
    a = ap.parse_args()
    days = a.days
    cad = CADENCE[days]
    rev = cad["revs"][a.rev]
    recipe = rev["recipe"]
    chans = CHANS_BY_REV[a.rev]
    nchan = len(chans)
    if a.out is None:
        a.out = os.path.join(os.path.dirname(OUT_NPZ), rev["out"])
    if a.memmap is None:
        a.memmap = a.out[:-4] + "_build.npy"
    # Family 5 stores X as a bare .npy BESIDE the npz (tensor_io.save_tensor):
    # 165.6 GB decompressed does not fit any box's RAM, and np.load on a
    # compressed member allocates the whole array. The memmap the build fills
    # is RENAMED into place, so the sidecar costs no second copy.
    sidecar = days == 1

    y0, m0, d0 = (int(x) for x in a.start.split("-"))
    y1, m1, d1 = (int(x) for x in a.end.split("-"))
    b_lo = bin_index(dt.date(y0, m0, d0), days)
    b_hi = bin_index(dt.date(y1, m1, d1), days)
    bins = list(range(b_lo, b_hi + 1))
    if a.max_bins:
        bins = bins[:a.max_bins]
    T = len(bins)

    print(f"axis      {bin_start(bins[0], days)} .. "
          f"{bin_start(bins[-1], days)}  T={T} {cad['name']} bins "
          f"(bins {bins[0]}..{bins[-1]}, recipe {recipe})")
    print(f"channels  {nchan} ({a.rev})"
          + (f" — family 3's {NC} + {chans[NC:]}" if nchan > NC else ""))

    if a.dry_run:
        H, W = 281, 481
        for name, bpe in (("float32", 4), ("float16", 2)):
            print(f"dense     [{T}, {H}, {W}, {nchan}] {name}: "
                  f"{T * H * W * nchan * bpe / 1e9:.1f} GB")
        st = os.statvfs(CACHE if os.path.isdir(CACHE) else HERE)
        print(f"disk      {st.f_bavail * st.f_frsize / 1e9:.1f} GB free")
        print("\n--dry-run: nothing built.")
        return

    print(f"base      {a.pentad_dir}")
    idx, arrs = load_pentad_base(a.pentad_dir, days)
    lats = np.asarray(idx["lat"], np.float32)
    lons = np.asarray(idx["lon"], np.float32)
    check_grid(lats, lons)
    H, W = len(lats), len(lons)
    src_bins = {int(b): i for i, b in enumerate(idx["bin_index"])}
    has = np.asarray(idx["has_data"], bool)

    # Run #391 died in 0.3 s: the free-space guard below ran BEFORE this
    # short-circuit and refused 33.1 GB for a tensor that was already on
    # disk and about to be skipped one line later. A guard must not cost
    # more than the thing it guards — check preconditions in DEPENDENCY
    # order, so a precondition is only demanded by work that still has to
    # happen. Nothing here needs free space, so nothing here may demand it.
    if not a.force and os.path.exists(a.out):
        try:
            from tensor_io import load_tensor
            cached = load_tensor(a.out)
            prev, have = str(cached["recipe"]), set(cached.files)
        except Exception:                                 # noqa: BLE001
            prev, have = "unreadable", set()
        lack = missing_truth_keys(have, truth_path(days))
        if prev == recipe and not lack:
            print(f"{a.out} already built by recipe {recipe} — skipping "
                  f"(--force to rebuild)")
            return
        if prev == recipe:
            print(f"cached tensor is recipe {recipe} but carries no "
                  f"{lack} — rebuilding. It was built before the labels were "
                  f"published, and the recipe string cannot tell the two "
                  f"apart.")
        else:
            print(f"cached tensor is recipe {prev!r}, want {recipe!r} — "
                  f"rebuilding")

    need = T * H * W * nchan * 2
    st = os.statvfs(os.path.dirname(os.path.abspath(a.memmap)))
    free = st.f_bavail * st.f_frsize
    print(f"tensor    [{T}, {H}, {W}, {nchan}] float16 = {need / 1e9:.1f} GB · "
          f"{free / 1e9:.1f} GB free")
    if need > free * 0.95:
        sys.exit(f"refusing to start: {need / 1e9:.1f} GB needed, "
                 f"{free / 1e9:.1f} GB free. This is a BOX build (E-034 §5); "
                 f"use --max-bins to exercise the path here.")

    X = np.lib.format.open_memmap(a.memmap, mode="w+", dtype=np.float16,
                                  shape=(T, H, W, nchan))
    for t in range(T):
        X[t] = np.nan

    # ---- base ------------------------------------------------------------
    print("base channels (cur_speed, log_mld, ssh) …", flush=True)
    nb = 0
    for r, b in enumerate(bins):
        j = src_bins.get(b)
        if j is None or not has[j]:
            continue
        uo = np.asarray(arrs["uo"][j], np.float32)
        vo = np.asarray(arrs["vo"][j], np.float32)
        ml = np.asarray(arrs["mlotst"][j], np.float32)
        zs = np.asarray(arrs["zos"][j], np.float32)
        if not np.isfinite(uo).any():
            continue
        X[r, :, :, 0] = np.hypot(uo, vo)
        with np.errstate(invalid="ignore", divide="ignore"):
            # log10, matching family 3 (verified against base025_na.npz).
            # mlotst is a depth in metres and is positive by construction;
            # a non-positive value is a fill leaking through, so it becomes
            # missing rather than -inf (ml/CLAUDE.md §5.22).
            X[r, :, :, 1] = np.where(ml > 0, np.log10(np.maximum(ml, 1e-6)),
                                     np.nan)
        X[r, :, :, 2] = zs
        nb += 1
    print(f"  base: {nb}/{T} pentads carry GLORYS "
          f"({T - nb} missing — pre-1993 and any gap in the pull)")

    ocean = np.zeros((H, W), bool)
    for r in range(T):
        ocean |= np.isfinite(np.asarray(X[r, :, :, 0], np.float32))
    print(f"  ocean cells: {int(ocean.sum())}/{H * W}")

    print("rg channels (one live pentad per month) …", flush=True)
    n_rg = fill_rg_pentad(X, bins, lats, lons, days)
    print("wind channels (pentad mean + within-pentad std) …", flush=True)
    n_wind = fill_wind_pentad(X, bins, lats, lons, days=days)
    n_sst = 0
    if nchan > NC:
        print(f"sst channel (NaN-aware {days}-day mean of the dailies) …",
              flush=True)
        n_sst = fill_sst(X, bins, lats, lons, days=days, sst_dir=a.sst_dir)
    print("truth series …", flush=True)
    truths = truth_pentad(bins, days)

    # ---- mask + stats, one slab pass -------------------------------------
    print("mask + stats pass …", flush=True)
    cnt = np.zeros(nchan, np.int64)
    s1 = np.zeros(nchan, np.float64)
    s2 = np.zeros(nchan, np.float64)
    for r in range(T):
        slab = np.asarray(X[r], np.float32)
        slab[~ocean] = np.nan
        X[r] = slab.astype(np.float16)
        fin = np.isfinite(slab)
        cnt += fin.sum((0, 1))
        v = np.where(fin, slab, 0.0).astype(np.float64)
        s1 += v.sum((0, 1))
        s2 += (v ** 2).sum((0, 1))
    mu = np.where(cnt > 0, s1 / np.maximum(cnt, 1), 0.0)
    sd = np.sqrt(np.maximum(s2 / np.maximum(cnt, 1) - mu ** 2, 0.0)) + 1e-6
    norm = np.stack([mu, sd], 1).astype(np.float32)

    print("z-score pass …", flush=True)
    muf, sdf = mu.astype(np.float32), sd.astype(np.float32)
    for r in range(T):
        X[r] = ((np.asarray(X[r], np.float32) - muf) / sdf).astype(np.float16)
    X.flush()

    # ---- Chinchilla inventory, from OBSERVED VALUES (E-034 §4) -----------
    groups = (("base", 0, C_BASE), ("rg", C_BASE, C_BASE + C_RG),
              ("wind", C_BASE + C_RG, NC)) \
        + ((("sst", NC, nchan),) if nchan > NC else ())
    total = int(cnt.sum())
    print("\nobserved values (the Chinchilla inventory — re-read it, do not "
          "scale the monthly one):")
    for name, lo, hi in groups:
        print(f"  {name:<5} {int(cnt[lo:hi].sum()):>15,}")
    print(f"  TOTAL {total:>15,}  ->  params anchor ~ "
          f"{total / 20 / 1e6:.1f} M (values/20)")

    # THE TRAINER LOADS THIS UNCHANGED, and that is a deliberate result
    # rather than a coincidence. `ml/train.py` touches `months` in exactly
    # five places: it loads it, prints it, takes `m[:4]` for the year-blocked
    # holdout, and takes `int(m[5:7])-1` twice for the season context token.
    # A pentad bin answers both questions correctly — it has a year, and it
    # has a time of year — so emitting one YYYY-MM per bin from the bin's
    # START date makes family 4 loadable with no edit to the trainer at all.
    # Removing the failure mode beats guarding it (ml/CLAUDE.md §4.1); the
    # alternative was a cadence branch threaded through the loader.
    # `bin_index` remains the authoritative axis; `months` is a label.
    months = np.array([f"{bin_start(b, days).year:04d}-"
                       f"{bin_start(b, days).month:02d}" for b in bins])
    # train.py reads d["rapid"] (axis-index, value) pairs, the name family 3
    # writes. truth_pentad() already produced exactly that shape under
    # `truth_rapid`; alias it rather than compute it twice.
    if "truth_rapid" in truths:
        truths.setdefault("rapid", truths["truth_rapid"])

    print(f"\nwriting {a.out} …", flush=True)
    meta = dict(bin_index=np.array(bins, np.int64), months=months,
                epoch=np.array(str(EPOCH)), pentad_days=np.array(days),
                lats=lats, lons=lons, chan=np.array(chans), norm=norm,
                window=np.array("na025"), cadence=np.array(cad["name"]),
                recipe=np.array(recipe), n_rg_live=np.array(n_rg),
                n_wind=np.array(n_wind), n_sst=np.array(n_sst), **truths)
    if sidecar:
        # RENAME the build memmap into place — a copy would double 166 GB.
        from tensor_io import save_tensor
        xp = save_tensor(a.out, X, **meta)
        print(f"wrote {a.out} + {os.path.basename(xp)}  "
              f"[T={T} H={H} W={W} C={nchan}] float16  "
              f"{os.path.getsize(xp) / 1e9:.2f} GB (memmappable)  "
              f"recipe={recipe}")
        return
    np.savez_compressed(a.out, X=X, **meta)
    del X
    if not a.keep_memmap:
        os.remove(a.memmap)
    print(f"wrote {a.out}  [T={T} H={H} W={W} C={nchan}] float16  "
          f"{os.path.getsize(a.out) / 1e9:.2f} GB  recipe={recipe}")


if __name__ == "__main__":
    sys.exit(main())

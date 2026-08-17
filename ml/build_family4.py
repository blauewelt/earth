#!/usr/bin/env python3
"""Family-4 tensor: the 0.25-degree North Atlantic at PENTAD cadence.

E-034 step 4. Output `ml/cache/family4_na025_pentad.npz`.

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
there is ONE definition. Per E-034 §2 the cadence policy differs per channel
group, and that is the whole substance of this file:

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

# ONE definition of the channel set, imported rather than restated.
CHANS, LEVELS = f3.CHANS, f3.LEVELS
C_BASE, C_RG, C_WIND, NC = f3.C_BASE, f3.C_RG, f3.C_WIND, f3.NC

RECIPE_REV = "f4r1"      # bump on ANY recipe change — it is the skip guard


def pentad_days(b):
    """The five calendar dates bin `b` covers."""
    s = bin_start(b, PENTAD_DAYS)
    return [s + dt.timedelta(days=k) for k in range(PENTAD_DAYS)]


# ---------------------------------------------------------------- base ----
def load_pentad_base(pentad_dir):
    """The aggregator's output, with the grid checked before anything else."""
    idx_path = os.path.join(pentad_dir, "index.npz")
    if not os.path.exists(idx_path):
        sys.exit(f"no index.npz in {pentad_dir} — run:\n"
                 f"  python3 ml/aggregate_cadence.py --hf-repo earth-tensors "
                 f"--cadence pentad --out {pentad_dir} --bin-deg 0.25")
    idx = np.load(idx_path)
    if int(idx["cadence_days"]) != PENTAD_DAYS:
        sys.exit(f"{pentad_dir} is cadence_days={int(idx['cadence_days'])}, "
                 f"not {PENTAD_DAYS} — this is the PENTAD builder")
    if str(idx["epoch"]) != str(EPOCH):
        sys.exit(f"{pentad_dir} epoch {str(idx['epoch'])!r} != {str(EPOCH)!r} "
                 f"— the state and label axes would not share bins")
    if "lat" not in idx:
        sys.exit(f"{pentad_dir} was built without --bin-deg (native 1/12 deg "
                 f"grid). Family 4 is a 0.25 deg tensor; rebuild with "
                 f"--bin-deg 0.25 --grid-align point.")
    arrs = {}
    for v in ("uo", "vo", "mlotst", "zos"):
        p = os.path.join(pentad_dir, f"pentad_mean_{v}.npy")
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
def fill_rg_pentad(X, bins, lats, lons):
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
        return row_of.get(bin_index(dt.date(y, m, 15), PENTAD_DAYS))

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
def fill_wind_pentad(X, bins, lats, lons, min_days=3):
    """NCEP R1 daily -> pentad mean + WITHIN-PENTAD std, channels 35..38.

    The std is why this cannot be derived from family-3's monthly channels: a
    standard deviation is not aggregable from a mean. It is computed here over
    the days of each 5-day bin, which is a storminess measure rather than a
    seasonal one.
    """
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


# --------------------------------------------------------------- truth ----
def truth_pentad(bins):
    """(row, transport) pairs on THIS axis, from build_truth_pentad.py."""
    out = {}
    if not os.path.exists(TRUTH_PENTAD):
        print(f"  ::warning:: {TRUTH_PENTAD} absent — run "
              f"ml/build_truth_pentad.py. No truth attached.")
        return out
    d = np.load(TRUTH_PENTAD)
    if str(d["epoch"]) != str(EPOCH) or int(d["pentad_days"]) != PENTAD_DAYS:
        sys.exit(f"truth_pentad.npz has epoch {str(d['epoch'])!r}/"
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


# ---------------------------------------------------------------- main ----
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pentad-dir",
                    default=os.path.join(CACHE, "glorys_pentad"),
                    help="aggregate_cadence.py output (index.npz + "
                         "pentad_mean_<var>.npy)")
    ap.add_argument("--out", default=OUT_NPZ)
    ap.add_argument("--memmap", default=MEMMAP)
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

    y0, m0, d0 = (int(x) for x in a.start.split("-"))
    y1, m1, d1 = (int(x) for x in a.end.split("-"))
    b_lo = bin_index(dt.date(y0, m0, d0), PENTAD_DAYS)
    b_hi = bin_index(dt.date(y1, m1, d1), PENTAD_DAYS)
    bins = list(range(b_lo, b_hi + 1))
    if a.max_bins:
        bins = bins[:a.max_bins]
    T = len(bins)

    print(f"axis      {bin_start(bins[0], PENTAD_DAYS)} .. "
          f"{bin_start(bins[-1], PENTAD_DAYS)}  T={T} pentads "
          f"(bins {bins[0]}..{bins[-1]})")

    if a.dry_run:
        H, W = 281, 481
        for name, bpe in (("float32", 4), ("float16", 2)):
            print(f"dense     [{T}, {H}, {W}, {NC}] {name}: "
                  f"{T * H * W * NC * bpe / 1e9:.1f} GB")
        st = os.statvfs(CACHE if os.path.isdir(CACHE) else HERE)
        print(f"disk      {st.f_bavail * st.f_frsize / 1e9:.1f} GB free")
        print("\n--dry-run: nothing built.")
        return

    print(f"base      {a.pentad_dir}")
    idx, arrs = load_pentad_base(a.pentad_dir)
    lats = np.asarray(idx["lat"], np.float32)
    lons = np.asarray(idx["lon"], np.float32)
    check_grid(lats, lons)
    H, W = len(lats), len(lons)
    src_bins = {int(b): i for i, b in enumerate(idx["bin_index"])}
    has = np.asarray(idx["has_data"], bool)

    need = T * H * W * NC * 2
    st = os.statvfs(os.path.dirname(os.path.abspath(a.memmap)))
    free = st.f_bavail * st.f_frsize
    print(f"tensor    [{T}, {H}, {W}, {NC}] float16 = {need / 1e9:.1f} GB · "
          f"{free / 1e9:.1f} GB free")
    if need > free * 0.95:
        sys.exit(f"refusing to start: {need / 1e9:.1f} GB needed, "
                 f"{free / 1e9:.1f} GB free. This is a BOX build (E-034 §5); "
                 f"use --max-bins to exercise the path here.")

    if not a.force and os.path.exists(a.out):
        try:
            prev = str(np.load(a.out)["recipe"])
        except Exception:                                 # noqa: BLE001
            prev = "unreadable"
        if prev == RECIPE_REV:
            print(f"{a.out} already built by recipe {RECIPE_REV} — skipping "
                  f"(--force to rebuild)")
            return
        print(f"cached tensor is recipe {prev!r}, want {RECIPE_REV!r} — rebuilding")

    X = np.lib.format.open_memmap(a.memmap, mode="w+", dtype=np.float16,
                                  shape=(T, H, W, NC))
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
    n_rg = fill_rg_pentad(X, bins, lats, lons)
    print("wind channels (pentad mean + within-pentad std) …", flush=True)
    n_wind = fill_wind_pentad(X, bins, lats, lons)
    print("truth series …", flush=True)
    truths = truth_pentad(bins)

    # ---- mask + stats, one slab pass -------------------------------------
    print("mask + stats pass …", flush=True)
    cnt = np.zeros(NC, np.int64)
    s1 = np.zeros(NC, np.float64)
    s2 = np.zeros(NC, np.float64)
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
              ("wind", C_BASE + C_RG, NC))
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
    months = np.array([f"{bin_start(b, PENTAD_DAYS).year:04d}-"
                       f"{bin_start(b, PENTAD_DAYS).month:02d}" for b in bins])
    # train.py reads d["rapid"] (axis-index, value) pairs, the name family 3
    # writes. truth_pentad() already produced exactly that shape under
    # `truth_rapid`; alias it rather than compute it twice.
    if "truth_rapid" in truths:
        truths.setdefault("rapid", truths["truth_rapid"])

    print(f"\nwriting {a.out} …", flush=True)
    np.savez_compressed(
        a.out, X=X, bin_index=np.array(bins, np.int64), months=months,
        epoch=np.array(str(EPOCH)), pentad_days=np.array(PENTAD_DAYS),
        lats=lats, lons=lons, chan=np.array(CHANS), norm=norm,
        window=np.array("na025"), cadence=np.array("pentad"),
        recipe=np.array(RECIPE_REV), n_rg_live=np.array(n_rg),
        n_wind=np.array(n_wind), **truths)
    del X
    if not a.keep_memmap:
        os.remove(a.memmap)
    print(f"wrote {a.out}  [T={T} H={H} W={W} C={NC}] float16  "
          f"{os.path.getsize(a.out) / 1e9:.2f} GB  recipe={RECIPE_REV}")


if __name__ == "__main__":
    sys.exit(main())

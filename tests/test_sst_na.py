#!/usr/bin/env python3
"""Pin the SST regrid before spending an hour of Actions and 4.25 GB on it.

`ml/fetch_sst_na.py` puts OISST v2.1 on the ML window's grid, and the two
grids DO NOT COINCIDE. OISST's `lat`/`lon` variables are cell CENTRES —
-89.875 + k*0.25 and 0.125 + k*0.25 — while family 3 samples ON the multiples
of 0.25: lats 0..70 (281), lons -100..20 (481), measured from
base025_na.npz. Every target point sits exactly HALFWAY between two source
centres in each axis, so the difference between indexing and interpolating is
a uniform half-cell shift (0.125 degree, ~14 km) of the entire field. That
shift is invisible in any plot, survives every shape check, and would leave
every stencil and the AMOC eval mask naming pixels other than the ones they
describe. It is the failure this file exists to make impossible.

The fixture is the REAL OISST geometry (720x1440 centres, three days,
_FillValue land) carrying a field that is an exact LINEAR RAMP in lat and
lon. A ramp is reproduced exactly by bilinear interpolation, so the expected
value at a target point is analytic — which is what turns this from a smoke
test into a measurement. The fixture stores float32 rather than the source's
own short/scale_factor 0.01 on purpose: quantising the ramp would put 0.005
of the source's own error inside a check that is meant to measure the
interpolator alone.

What is asserted:

  1. **The output axes ARE the target axes**, exactly: lat[0] == 0.0 and not
     0.125, every value a multiple of 0.25, 281x481. The source axes are
     offset by half a cell and are shown to differ, so this cannot pass by
     the two grids accidentally being the same.
  2. **The interpolation is right to float tolerance** against the analytic
     ramp, through `fetch_sst_na.weights_for` — the same call the fetcher
     makes — away from the seam; and AT the 360/0 seam the value is the
     wrap-interpolated mean of the two straddling columns rather than a
     clamp, which is what `wrap_period=360.0` is for. (A ramp in longitude is
     discontinuous across the seam; that is a property of the fixture, not of
     the interpolator, hence the two regions.)
  3. **Land stays nodata.** A masked block comes back -32768, not 0 (which
     would decode to a perfectly plausible 0.00 degC) and not a fill
     temperature borrowed from its neighbours.
  4. **The int16 round-trip is lossless to 0.005 degC** — the artifact
     decoded at scale 0.01 equals the analytic value. This is why the encoder
     rounds where `scripts/bake_sst_daily.py` truncates: truncation puts
     24.22 degC back as 24.21.
  5. **The time axis is days since 1982-01-01, contiguous**, and `has_data`
     marks exactly the days the source carried — the fixture holds
     1993-01-01, 1993-01-02 and 1993-06-15 out of 365 planned rows, so a
     "first N rows" bug cannot pass.
  6. **The source is deleted after folding** — the peak-disk discipline that
     lets 43 years x 476 MB run in ~1 GB of scratch.

    python3 tests/test_sst_na.py
"""
import datetime as dt
import os
import shutil
import sys
import tempfile

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "ml"))

import build_family3 as f3                                    # noqa: E402
import fetch_sst_na as F                                      # noqa: E402

EPOCH = dt.date(1982, 1, 1)
FILL = -9999.0
DAYS = [dt.date(1993, 1, 1), dt.date(1993, 1, 2), dt.date(1993, 6, 15)]
# sst = A + B*lat + C*lon360 degC, over lon360 260..360 that is 23..35 degC
A, B, C = 10.0, 0.1, 0.05
# Land block, in the source's own coordinates: target lons -60..-50.
LAND = dict(lat0=30.0, lat1=40.0, lon0=300.0, lon1=310.0)


def analytic(lat, lon360):
    return A + B * np.asarray(lat, np.float64)[:, None] + \
        C * np.asarray(lon360, np.float64)[None, :]


def write_fixture(path):
    """A three-day OISST look-alike: real geometry, analytic field."""
    import netCDF4 as nc

    lat = -89.875 + 0.25 * np.arange(720)
    lon = 0.125 + 0.25 * np.arange(1440)
    field = analytic(lat, lon)
    land = ((lat[:, None] >= LAND["lat0"]) & (lat[:, None] <= LAND["lat1"])
            & (lon[None, :] >= LAND["lon0"]) & (lon[None, :] <= LAND["lon1"]))
    field = np.where(land, FILL, field).astype(np.float32)

    d = nc.Dataset(path, "w")
    d.createDimension("time", None)
    d.createDimension("lat", len(lat))
    d.createDimension("lon", len(lon))
    v = d.createVariable("lat", "f4", ("lat",)); v[:] = lat
    v = d.createVariable("lon", "f4", ("lon",)); v[:] = lon
    t = d.createVariable("time", "f8", ("time",))
    t.units = "days since 1800-01-01 00:00:00"
    t[:] = [(x - dt.date(1800, 1, 1)).days for x in DAYS]
    s = d.createVariable("sst", "f4", ("time", "lat", "lon"),
                         fill_value=FILL)
    s.units = "degC"
    for i in range(len(DAYS)):
        s[i] = field
    d.close()
    return lat, lon, land


def main():
    tmp = tempfile.mkdtemp(prefix="sstna_")
    fixture = os.path.join(tmp, "fixture.nc")
    src_lat, src_lon, land = write_fixture(fixture)
    out = os.path.join(tmp, "out")

    # The seam the fetcher would otherwise reach the network through. It is
    # given the real filename so the deletion in check 6 is the real one.
    copied = []

    def fake_download(year, path):
        shutil.copy(fixture, path)
        copied.append(path)
        return path

    F.download_year = fake_download
    print("--- fetch_sst_na.py --year 1993 ---------------------------------")
    F.main(["--year", "1993", "--out", out])
    print("----------------------------------------------------------------")

    idx = np.load(os.path.join(out, "index.npz"))
    arr = np.load(os.path.join(out, "sst_daily_na.npy"), mmap_mode="r")
    lat, lon = idx["lat"], idx["lon"]

    # ---- 1: the axes ARE the target axes, not the source's centres --------
    want_lat = (0.0 + 0.25 * np.arange(281)).astype(np.float32)
    want_lon = (-100.0 + 0.25 * np.arange(481)).astype(np.float32)
    assert np.array_equal(lat, want_lat), f"lat axis: {lat[:3]}..{lat[-3:]}"
    assert np.array_equal(lon, want_lon), f"lon axis: {lon[:3]}..{lon[-3:]}"
    assert (np.mod(np.abs(lat) * 4, 1) == 0).all(), "lat is off the 0.25 grid"
    assert (np.mod(np.abs(lon) * 4, 1) == 0).all(), "lon is off the 0.25 grid"
    assert arr.shape == (365, 281, 481), arr.shape
    assert arr.dtype == np.int16, arr.dtype
    assert float(idx["scale"]) == 0.01 and int(idx["nodata"]) == -32768
    assert str(idx["epoch"]) == "1982-01-01" and int(idx["cadence_days"]) == 1
    off = float(np.min(np.abs(src_lat[:, None] - lat[None, :].astype(np.float64))))
    assert abs(off - 0.125) < 1e-9, (
        f"the source centres are {off} from the target points — the fixture "
        f"is not testing the half-cell case it claims to")
    print(f"  1. axes are the ML window's own: {len(lat)}x{len(lon)}, "
          f"lat {lat[0]}..{lat[-1]}, lon {lon[0]}..{lon[-1]}, int16 "
          f"{arr.shape}; every source centre is 0.125 deg away (the half-cell "
          f"offset that indexing would have baked in)")

    # ---- 2: the interpolation itself, against the analytic ramp -----------
    wy, wx = F.weights_for(src_lat, src_lon, lat, lon)
    day0 = np.ma.filled(np.ma.masked_equal(
        np.where(land, FILL, analytic(src_lat, src_lon)).astype(np.float32),
        np.float32(FILL)), np.nan)
    got = f3.interp2_nan(day0, wy, wx)
    lon360 = np.where(lon < 0, lon.astype(np.float64) + 360.0,
                      lon.astype(np.float64))
    want = analytic(lat, lon360)
    # Away from the seam (lon360 >= 260) and from the land block.
    keep = (lon <= -70.0)
    err = np.abs(got[:, keep] - want[:, keep])
    assert np.isfinite(got[:, keep]).all(), "the open-ocean ramp came back NaN"
    assert err.max() < 1e-4, f"bilinear is not exact on a ramp: {err.max()}"
    # At the seam the wrap must interpolate 359.875 <-> 0.125, not clamp.
    j = int(np.flatnonzero(lon == 0.0)[0])
    seam = A + B * lat.astype(np.float64) + C * 0.5 * (359.875 + 0.125)
    assert np.abs(got[:, j] - seam).max() < 1e-4, (
        f"lon 0.0 is not the wrap-interpolated mean of the straddling "
        f"columns: {got[0, j]} vs {seam[0]}")
    print(f"  2. bilinear reproduces the ramp exactly: max |err| "
          f"{err.max():.2e} degC over {err.size:,} points; the 360/0 seam is "
          f"wrap-interpolated (lon 0.0 = mean of 359.875 and 0.125)")

    # ---- 3: land is nodata, not zero and not a borrowed temperature -------
    row = (DAYS[0] - EPOCH).days - (dt.date(1993, 1, 1) - EPOCH).days
    frame = np.asarray(arr[row])
    tl = ((lat[:, None] >= LAND["lat0"] + 0.25)
          & (lat[:, None] <= LAND["lat1"] - 0.25)
          & (lon360[None, :] >= LAND["lon0"] + 0.25)
          & (lon360[None, :] <= LAND["lon1"] - 0.25))
    assert tl.sum() > 100, f"the land block covers only {tl.sum()} target cells"
    assert (frame[tl] == -32768).all(), (
        f"land came back as {sorted(set(frame[tl].tolist()))[:5]} — "
        f"0 would decode to a perfectly plausible 0.00 degC")
    assert (frame[~tl] != -32768).sum() > 0, "everything is nodata"
    print(f"  3. {int(tl.sum()):,} land cells are -32768; "
          f"{int((frame != -32768).sum()):,} wet cells carry values")

    # ---- 4: int16 round-trip, decoded against the analytic value ---------
    # float64 decode: the only error left is our own rounding, so the bound
    # is exactly half a count and not half a count plus float32 noise.
    dec = frame.astype(np.float64) * float(idx["scale"])
    err4 = np.abs(dec[:, keep] - want[:, keep])
    assert err4.max() <= 0.005 + 1e-9, (
        f"int16 round-trip is off by {err4.max():.4f} degC — the encoder "
        f"truncates instead of rounding")
    print(f"  4. decoded artifact vs analytic: max |err| {err4.max():.4f} "
          f"degC (<= half a 0.01 count), range {dec[:, keep].min():.2f}.."
          f"{dec[:, keep].max():.2f} degC")

    # ---- 5: the day axis and has_data -------------------------------------
    bi, has = idx["bin_index"], idx["has_data"]
    assert bi.dtype == np.int64 and has.dtype == bool
    assert len(bi) == 365 and len(has) == 365, (len(bi), len(has))
    assert bi[0] == (dt.date(1993, 1, 1) - EPOCH).days == 4018, bi[0]
    assert (np.diff(bi) == 1).all(), "the daily axis is not contiguous"
    want_rows = sorted((d - dt.date(1993, 1, 1)).days for d in DAYS)
    assert sorted(np.flatnonzero(has).tolist()) == want_rows, (
        f"has_data marks {np.flatnonzero(has).tolist()}, expected {want_rows}")
    assert (np.asarray(arr[7]) == -32768).all(), (
        "a day the source never carried is not nodata")
    print(f"  5. bin_index[0] = (1993-01-01 - {EPOCH}).days = {bi[0]}, 365 "
          f"contiguous days; has_data exactly at rows {want_rows} "
          f"({[str(d) for d in DAYS]}), every other row all-nodata")

    # ---- 6: the source is deleted after folding ---------------------------
    assert copied and not any(os.path.exists(p) for p in copied), (
        f"the source survived the fold: {copied}")
    assert os.path.exists(fixture), "the fixture itself was deleted"
    print(f"  6. the {len(copied)} downloaded source(s) were deleted after "
          f"folding — peak disk stays ~1 GB + the growing output")

    shutil.rmtree(tmp, ignore_errors=True)
    print("\ntests/test_sst_na.py: all 6 checks passed")


if __name__ == "__main__":
    main()

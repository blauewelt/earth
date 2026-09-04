#!/usr/bin/env python3
"""The ten things asserted before the family-7 global tensor is trusted.

`ml/plans/E070_family7_build.md` §5. Family 7 is the first input tensor
covering the whole globe rather than the North Atlantic window: every
0.25-degree grid point from pole to pole, one value per channel per five-day
bin, 1982-2024, in three groups at their native resolution (`g025` 0.25 deg /
7 channels, `g100` 1 deg / 15 NCEP channels, `rg100` 1 deg / 32 Argo depth
channels on the live bins only).

E-071 §6.1's correction of 4 Sep is asserted here too: `sst` is the OBSERVED
OISST field and missing where OISST does not observe, and the SHARED surface
temperature is the reanalysis `skt`, in g100, over every surface.

CPU-only, NO NETWORK. Every source is synthetic and reaches the builder through
`--source-dir`; the end-to-end check is `build_family7.run_smoke()`, the same
path `python3 ml/build_family7.py --smoke` runs. The build is done ONCE in a
module fixture and the assertions read it, so the whole file costs one build.

    python3 -m pytest tests/test_build_family7.py -q
"""
import datetime as dt
import json
import os
import shutil
import subprocess
import sys
import tempfile

import numpy as np
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
ML = os.path.join(HERE, "..", "ml")
sys.path.insert(0, ML)

import build_family3 as f3                                    # noqa: E402
import build_family4 as f4                                    # noqa: E402
import build_family7 as b7                                    # noqa: E402
import aggregate_cadence as ac                                # noqa: E402
from tensor_io import load_tensor, save_tensor                # noqa: E402


# --------------------------------------------------------------- fixtures --
@pytest.fixture(scope="module")
def build():
    """One synthetic end-to-end build; every assertion below reads it."""
    root = tempfile.mkdtemp(prefix="f7test_")
    work = b7.run_smoke(root=root, keep=True)
    d = load_tensor(os.path.join(work, b7.STEM + ".npz"))
    yield dict(root=root, work=work, src=os.path.join(root, "src"), d=d,
               start=b7.SMOKE_START, end=b7.SMOKE_END)
    d.close()
    shutil.rmtree(root, ignore_errors=True)


def unz(d, group, ch):
    """Un-z-score one channel back to its physical unit."""
    mu, sd = np.asarray(d[f"norm_{group}"])[ch]
    return np.asarray(d[f"X_{group}"][..., ch], np.float32) * sd + mu


def through_f16(want, d, group, ch):
    """What the tensor MUST hold for `want`, given how that group is stored.

    A coarse group (`b7.RAW_F32`) is filled at float32 and quantised ONCE, on
    the z-scored value. `g025` has no float32 intermediate, so its raw value is
    quantised at its own magnitude first and the z-scored value again — which
    is why `skin_t` near 35 degC carries 0.031 degC. Modelling the storage
    rather than picking a loose tolerance is what keeps that a stated property
    instead of a mystery failure.
    """
    mu, sd = np.asarray(d[f"norm_{group}"])[ch]
    raw = np.asarray(want, np.float32)
    if group not in b7.RAW_F32:
        raw = raw.astype(np.float16).astype(np.float32)
    z = ((raw - mu) / sd).astype(np.float16).astype(np.float32)
    return z * sd + mu


def gauss_bin_mean(path, var, bins_days, flip=False):
    """The pentad mean of one gaussian variable, computed independently here."""
    import netCDF4 as ncdf
    ds = ncdf.Dataset(path)
    dates = b7.nc_dates(ds)
    v = b7.pick_var(ds, var)
    acc = None
    n = 0
    for k, day in enumerate(dates):
        if day not in bins_days:
            continue
        f = b7.squeeze_level(np.ma.filled(np.asarray(v[k]), np.nan))
        f = -np.asarray(f, np.float64) if flip else np.asarray(f, np.float64)
        acc = f if acc is None else acc + f
        n += 1
    ds.close()
    return (acc / n if n else None), n


def days_of_bin(b):
    s = ac.bin_start(b, b7.PENTAD_DAYS)
    return [s + dt.timedelta(days=k) for k in range(b7.PENTAD_DAYS)]


# ------------------------------------------------------------------- 1 -----
def test_1_axes_and_the_coarse_lookup():
    """721 / 1440 / 181 / 360, ascending, and round(y/4) / round(x/4) mod 360."""
    lats, lons = b7.grid025()
    lat1, lon1 = b7.grid100()
    assert (len(lats), len(lons)) == (721, 1440)
    assert (len(lat1), len(lon1)) == (181, 360)
    assert lats[0] == -90.0 and lats[-1] == 90.0
    assert lons[0] == -180.0 and lons[-1] == 179.75
    assert lat1[0] == -90.0 and lat1[-1] == 90.0
    assert lon1[0] == -180.0 and lon1[-1] == 179.0
    for a in (lats, lons, lat1, lon1):
        assert np.all(np.diff(a) > 0), "an axis is not ascending"
    assert np.allclose(np.diff(lats), 0.25) and np.allclose(np.diff(lons), 0.25)

    # every fourth 0.25-degree point IS a 1-degree point
    y1, x1 = b7.coarse_lookup(np.arange(721), np.zeros(721, int))
    assert np.array_equal(lat1[y1[::4]], lats[::4])
    # the two points either side round to it, and the dateline wraps
    assert int(b7.coarse_lookup(0, 1439)[1]) == 0
    assert int(b7.coarse_lookup(0, 1438)[1]) == 0
    assert int(b7.coarse_lookup(2, 0)[0]) == 1        # half-UP, not half-even
    assert int(b7.coarse_lookup(1, 0)[0]) == 0
    assert int(b7.coarse_lookup(3, 0)[0]) == 1
    assert lon1[int(b7.coarse_lookup(0, 1439)[1])] == -180.0


# ------------------------------------------------------------------- 2 -----
def test_2_bin_index_and_months_agree_with_family4(build):
    """The state axis and family 4's are the SAME bins for the same epoch."""
    d = build["d"]
    assert str(d["epoch"]) == str(ac.EPOCH) == str(f4.EPOCH)
    assert int(d["pentad_days"]) == f4.PENTAD_DAYS == 5
    bins = np.asarray(d["bin_index"])
    y0, m0, dd0 = (int(x) for x in build["start"].split("-"))
    y1, m1, dd1 = (int(x) for x in build["end"].split("-"))
    want = np.arange(ac.bin_index(dt.date(y0, m0, dd0), 5),
                     ac.bin_index(dt.date(y1, m1, dd1), 5) + 1)
    assert np.array_equal(bins, want)
    # `months` is family 4's own expression, bin start -> YYYY-MM
    want_m = np.array([f"{ac.bin_start(b, 5).year:04d}-"
                       f"{ac.bin_start(b, 5).month:02d}" for b in bins])
    assert np.array_equal(np.asarray(d["months"]), want_m)
    # and the full axis is 3142 bins, 1982-01-01 .. the bin starting 2024-12-31
    full = list(range(ac.bin_index(dt.date(1982, 1, 1), 5),
                      ac.bin_index(dt.date(2024, 12, 31), 5) + 1))
    assert len(full) == 3142 and full[0] == 0
    assert ac.bin_start(full[-1], 5) == dt.date(2024, 12, 31)


# ------------------------------------------------------------------- 3 -----
def test_3_glorys_lands_at_row_40_and_the_axis_is_asserted(build, tmp_path):
    """Rows 0..39 are NaN; a chunk whose latitude[0] is wrong is REFUSED."""
    d = build["d"]
    spd = np.asarray(d["X_g025"][:, :, :, b7.C_CUR_SPEED], np.float32)
    assert not np.isfinite(spd[:, :40, :]).any(), \
        "an ocean channel is finite south of -80, where GLORYS has no rows"
    assert np.isfinite(spd[:, 40:, :]).any(), "GLORYS wrote nothing at all"
    lats, _ = b7.grid025()
    assert lats[b7.GLORYS_ROW0] == b7.GLORYS_LAT0 == -80.0

    # the assert is at READ time, not an assumption: a chunk half a cell out
    # must stop the build rather than shift the whole ocean.
    src = str(tmp_path / "src")
    bad_lat = -79.75 + 0.25 * np.arange(b7.GLORYS_ROWS)
    lon = -180.0 + 0.25 * np.arange(b7.NLON)
    days = [dt.date(2010, 1, 14) + dt.timedelta(days=k) for k in range(3)]
    z = np.zeros((3, b7.GLORYS_ROWS, b7.NLON), np.float32)
    b7._nc_write(
        os.path.join(src, "daily025_global", "glorys025_global_201001.nc"),
        {"time": 3, "latitude": b7.GLORYS_ROWS, "longitude": b7.NLON},
        {"latitude": (("latitude",), bad_lat, None),
         "longitude": (("longitude",), lon, None),
         "time": (("time",), np.array([(x - b7.EPOCH).days for x in days],
                                      np.float64),
                  {"units": f"days since {b7.EPOCH} 00:00:00"}),
         "uo": (("time", "latitude", "longitude"), z, None),
         "vo": (("time", "latitude", "longitude"), z, None),
         "mlotst": (("time", "latitude", "longitude"), z + 50.0, None),
         "zos": (("time", "latitude", "longitude"), z, None)})
    import argparse
    ctx = b7.Ctx(argparse.Namespace(
        work=str(tmp_path / "work"), source_dir=src, start="2010-01-14",
        end="2010-01-18", force=False, stage="glorys", smoke=True))
    with pytest.raises(SystemExit) as e:
        b7.stage_glorys(ctx)
    assert "latitude" in str(e.value)


# ------------------------------------------------------------------- 4 -----
def test_4_oisst_pentad_mean_min_days_and_sst_is_observed_only(build):
    """The bin mean, the >= 3 day rule, and `sst` missing where OISST is."""
    import netCDF4 as ncdf
    d, src = build["d"], build["src"]
    bins = [int(b) for b in np.asarray(d["bin_index"])]
    lats, lons = b7.grid025()

    ds = ncdf.Dataset(os.path.join(src, "oisst", "sst.day.mean.2010.nc"))
    s_lat = np.asarray(ds.variables["lat"][:], np.float64)
    s_lon = np.asarray(ds.variables["lon"][:], np.float64)
    wy = f3.lin_weights(s_lat, lats)
    wx = f3.lin_weights(s_lon, np.where(lons < 0, lons + 360.0, lons),
                        wrap_period=360.0)
    dates = b7.nc_dates(ds)
    v = b7.pick_var(ds, "sst")
    per_bin = {}
    for k, day in enumerate(dates):
        b = ac.bin_index(day, 5)
        if b not in bins:
            continue
        g = f3.interp2_nan(np.ma.filled(np.asarray(v[k]), np.nan).astype(np.float64),
                           wy, wx)
        per_bin.setdefault(b, []).append(g)
    ds.close()

    # aggregate_cadence's own arithmetic: min_days is 3 at pentad cadence
    assert (3 if 5 == 5 else 1) == b7.MIN_DAYS
    skin = unz(d, "g025", b7.C_SST)
    oisst_seen = np.load(os.path.join(build["work"], "oisst_seen.npy"))

    thin = [b for b in bins if len(per_bin.get(b, [])) < b7.MIN_DAYS]
    fat = [b for b in bins if len(per_bin.get(b, [])) >= b7.MIN_DAYS]
    assert thin and fat, "the fixture must exercise both sides of min_days"

    for b in fat:
        row = bins.index(b)
        stack = np.stack(per_bin[b])
        cnt = np.isfinite(stack).sum(0)
        with np.errstate(invalid="ignore"):
            want = np.where(cnt >= b7.MIN_DAYS,
                            np.nansum(np.where(np.isfinite(stack), stack, 0), 0)
                            / np.maximum(cnt, 1), np.nan)
        got = skin[row]
        model = through_f16(want, d, "g025", b7.C_SST)
        m = np.isfinite(want)
        assert m.any()
        # the tensor's value IS the OISST mean, and nothing else is in there
        assert np.allclose(got[m], model[m], atol=0.05), \
            f"bin {b}: sst is not the NaN-aware {b7.PENTAD_DAYS}-day OISST mean"
        # E-071 §6.1 corrected: NO reanalysis fill. `sst` is an OBSERVED
        # channel and must be missing wherever the instrument does not look.
        assert not np.isfinite(got[~oisst_seen]).any(), \
            "sst is finite where OISST never observes — the retired NCEP fill"
        assert np.array_equal(np.isfinite(got), m), \
            "sst's missing pattern is not OISST's own"

    for b in thin:
        row = bins.index(b)
        assert not np.isfinite(skin[row]).any(), \
            f"bin {b} has fewer than {b7.MIN_DAYS} OISST days but carries a mean"

    # ...and the SHARED surface temperature is `skt`, in g100, everywhere
    skt = unz(d, "g100", b7.C_SKT)
    assert np.isfinite(skt).mean() > 0.5, "skt is not the everywhere channel"
    assert list(d["chan_g100"])[b7.C_SKT] == "skt"
    assert "skin_t" not in list(d["chan_g025"]) + list(d["chan_g100"])

    # sea_ice is NaN wherever OISST has no sea
    ice = unz(d, "g025", b7.C_SEA_ICE)
    land = ~oisst_seen
    assert not np.isfinite(ice[:, land]).any(), \
        "sea_ice is finite where OISST has no sea"

    # ...and it is a FRACTION, though the file's units string says "percent".
    # MEASURED from icec.day.mean.2020.nc's DAS: units "percent" over a
    # valid_range of 0..1. A reader that trusted the string divides the whole
    # channel by a hundred and nothing downstream notices.
    di = ncdf.Dataset(os.path.join(src, "oisst", "icec.day.mean.2010.nc"))
    vi = b7.pick_var(di, "icec")
    assert "percent" in str(getattr(vi, "units", "")).lower(), \
        "the fixture must carry the misleading units string"
    assert float(b7.ice_divisor(vi)) == 1.0, \
        "ice_divisor trusted the units string over valid_range"
    raw_hi = float(np.nanmax(np.ma.filled(np.asarray(vi[0]), np.nan)))
    di.close()
    fin = np.isfinite(ice)
    assert fin.any()
    assert 0.0 <= float(np.nanmin(ice[fin])) and float(np.nanmax(ice[fin])) <= 1.0
    assert float(np.nanmax(ice[fin])) > raw_hi / 10.0, \
        "sea_ice was divided by 100 — the units string won over valid_range"


# ------------------------------------------------------------------- 5 -----
def test_5_ncep_transforms(build):
    """uflx/vflx flipped and nothing else; K->C, Pa->hPa, the two log1p's;
    soilw/tsoil NaN over the gaussian sea mask."""
    d, src = build["d"], build["src"]
    bins = [int(b) for b in np.asarray(d["bin_index"])]
    lat1, lon1 = b7.grid100()
    import netCDF4 as ncdf
    p0 = os.path.join(src, "ncep", b7.NCEP_FILES["air"] + ".2010.nc")
    ds = ncdf.Dataset(p0)
    g_lat = np.asarray(ds.variables["lat"][:], np.float64)
    g_lon = np.asarray(ds.variables["lon"][:], np.float64)
    ds.close()
    assert g_lat[0] > g_lat[-1], "the fixture must use a DESCENDING gaussian lat"
    wy = f3.lin_weights(g_lat, lat1)
    wx = f3.lin_weights(g_lon, np.where(lon1 < 0, lon1 + 360.0, lon1),
                        wrap_period=360.0)

    dl = ncdf.Dataset(os.path.join(src, "ncep", b7.NCEP_LAND + ".nc"))
    land = b7.squeeze_level(np.ma.filled(np.asarray(
        b7.pick_var(dl, "land")[:]), 0.0)) >= 0.5
    dl.close()

    # pick a bin that has enough days
    chosen = None
    for b in bins:
        days = set(days_of_bin(b))
        _, n = gauss_bin_mean(p0, "air", days)
        if n >= b7.MIN_DAYS:
            chosen = (b, days)
            break
    assert chosen, "no bin in the fixture has >= 3 NCEP days"
    b, days = chosen
    row = bins.index(b)

    def native(var, flip=False):
        return gauss_bin_mean(os.path.join(src, "ncep",
                                           b7.NCEP_FILES[var] + ".2010.nc"),
                              var, days, flip=flip)[0]

    def to1(f):
        return f3.interp2_nan(f, wy, wx)

    def close(ch, want, atol):
        got = unz(d, "g100", ch)[row]
        model = through_f16(want, d, "g100", ch)
        m = np.isfinite(want)
        assert m.any()
        assert np.allclose(got[m], model[m], atol=atol), \
            f"channel {b7.CHAN_G100[ch]} does not match its rule "\
            f"(max |d| {np.nanmax(np.abs(got[m] - model[m]))})"
        return got

    close(0, to1(native("uflx", flip=True)), 1e-3)      # tau_x  = -uflx
    close(1, to1(native("vflx", flip=True)), 1e-3)      # tau_y  = -vflx
    close(4, to1(native("air")) - 273.15, 5e-2)         # t2m    K -> degC
    close(5, to1(native("uwnd")), 5e-3)                 # u10    NOT flipped
    close(6, to1(native("vwnd")), 5e-3)                 # v10    NOT flipped
    close(7, to1(native("pres")) / 100.0, 5e-2)         # sp     Pa -> hPa
    close(8, np.log1p(np.maximum(to1(native("prate")) * 86400.0, 0)), 1e-3)
    close(9, np.log1p(np.maximum(to1(native("weasd")), 0)), 1e-3)
    close(12, to1(native("lhtfl")), 5e-2)               # unchanged
    close(13, to1(native("shtfl")), 5e-2)               # unchanged
    # skt: K -> degC, NO land mask, NO flip — the shared channel
    got_skt = close(b7.C_SKT, to1(native("skt")) - 273.15, 5e-2)
    assert np.isfinite(got_skt).all(), \
        "skt was masked somewhere; the shared channel covers every surface"

    # the flip really is a flip, and really is only on those two
    assert np.nanmean(to1(native("uflx"))) * np.nanmean(
        unz(d, "g100", 0)[row]) < 0, "tau_x is not the negated uflx"
    assert np.nanmean(to1(native("uwnd"))) * np.nanmean(
        unz(d, "g100", 5)[row]) > 0, "u10 was flipped and must not be"

    # tau_*_std is the WITHIN-PENTAD population sigma of the same dailies
    import netCDF4 as _nc
    dsu = _nc.Dataset(os.path.join(src, "ncep",
                                   b7.NCEP_FILES["uflx"] + ".2010.nc"))
    dates = b7.nc_dates(dsu)
    vv = b7.pick_var(dsu, "uflx")
    stack = np.stack([-np.asarray(b7.squeeze_level(
        np.ma.filled(np.asarray(vv[k]), np.nan)), np.float64)
        for k, day in enumerate(dates) if day in days])
    dsu.close()
    want_sd = through_f16(to1(np.std(stack, axis=0)), d, "g100", 2)
    got_sd = unz(d, "g100", 2)[row]
    m = np.isfinite(want_sd)
    assert np.allclose(got_sd[m], want_sd[m], atol=5e-3)

    # soilw / tsoil: NaN over the gaussian SEA, masked before regridding
    want_soil = to1(np.where(land, native("soilw"), np.nan))
    got_soil = unz(d, "g100", 10)
    assert np.array_equal(np.isfinite(got_soil[row]), np.isfinite(want_soil)), \
        "soilw's missingness is not the gaussian land mask through interp2_nan"
    m = np.isfinite(want_soil)
    assert np.allclose(got_soil[row][m],
                       through_f16(want_soil, d, "g100", 10)[m], atol=5e-3)
    want_tsoil = to1(np.where(land, native("tmp"), np.nan)) - 273.15
    got_tsoil = unz(d, "g100", 11)[row]
    mt = np.isfinite(want_tsoil)
    assert np.allclose(got_tsoil[mt],
                       through_f16(want_tsoil, d, "g100", 11)[mt], atol=5e-2)
    # and there ARE sea cells: a channel that masked nothing proves nothing
    sea = np.isfinite(unz(d, "g100", 12)[row]) & ~np.isfinite(got_soil[row])
    assert sea.any(), "no cell is sea-masked — the land mask did not apply"


# ------------------------------------------------------------------- 6 -----
def test_6_rg_band_and_live_bin(build):
    """NaN outside -64.5..79.5, and the live bin is the one holding the 15th."""
    d = build["d"]
    lat1, _ = b7.grid100()
    R = np.asarray(d["X_rg100"], np.float32)
    assert R.shape[0] == int(d["n_rg_live"]) >= 1
    assert R.shape[1:] == (181, 360, 32)
    band = (lat1 >= b7.RG_LAT_LO) & (lat1 <= b7.RG_LAT_HI)
    assert not np.isfinite(R[:, ~band, :, :]).any(), \
        "RG's edge rows were replicated outside its latitude band by the clamp"
    assert np.isfinite(R[:, band, :, :]).any(), "rg100 is entirely missing"

    rb = np.asarray(d["rg_bin_index"])
    months = [str(m) for m in np.asarray(d["rg_months"])]
    for b, m in zip(rb, months):
        y, mo = int(m[:4]), int(m[5:])
        assert int(b) == ac.bin_index(dt.date(y, mo, 15), 5), \
            f"{m}'s live bin is not the pentad holding the 15th"
        s = ac.bin_start(int(b), 5)
        assert s <= dt.date(y, mo, 15) < s + dt.timedelta(days=5)
    assert np.all(np.diff(rb) > 0), "rg_bin_index is not ordered"
    assert list(d["chan_rg100"])[:1] == ["rg_t10"]
    assert len(d["chan_rg100"]) == 32


# ------------------------------------------------------------------- 7 -----
def test_7_sphere_priority_and_elev_block_mean(tmp_path):
    """Ice > ocean > lake > land, and the block mean on a toy raster."""
    ice = np.array([[1, 0, 0, 0], [1, 0, 0, 0]], bool)
    ocean = np.array([[1, 1, 0, 0], [0, 1, 1, 0]], bool)
    lake = np.array([[1, 1, 1, 0], [0, 0, 1, 1]], bool)
    s = b7.sphere_codes(ice, ocean, lake)
    assert s.dtype == np.int8
    # ice wins over ocean AND lake; ocean wins over lake; land is the residue
    assert s.tolist() == [[2, 0, 3, 1], [2, 0, 0, 3]]

    # the polygons really do drive it, through shapely
    gj = tmp_path / "poly.geojson"
    gj.write_text(json.dumps({"type": "FeatureCollection", "features": [
        {"type": "Feature", "properties": {}, "geometry": {
            "type": "Polygon",
            "coordinates": [[[-1.1, -1.1], [1.1, -1.1], [1.1, 1.1],
                             [-1.1, 1.1], [-1.1, -1.1]]]}}]}))
    lats = np.array([-2.0, -1.0, 0.0, 1.0, 2.0])
    lons = np.array([-2.0, 0.0, 2.0])
    hit = b7.polygon_hits(str(gj), lats, lons)
    assert hit.tolist() == [[False, False, False], [False, True, False],
                            [False, True, False], [False, True, False],
                            [False, False, False]]

    # ---- elev: a toy CELL-registered, north-first raster at 1/4 the target
    # spacing, so f = 4 and the block mean has a known analytic answer.
    step = 0.0625
    e_lat = 90.0 - step * (np.arange(int(180 / step)) + 0.5)
    e_lon = -180.0 + step * (np.arange(int(360 / step)) + 0.5)
    # z depends on the LONGITUDE INDEX only, linearly: the mean over any
    # contiguous block of 4 columns is the mean of its indices.
    z = np.broadcast_to(np.arange(len(e_lon), dtype=np.float32),
                        (len(e_lat), len(e_lon))).copy()
    p = str(tmp_path / "toy_etopo.nc")
    b7._nc_write(p, {"lat": len(e_lat), "lon": len(e_lon)},
                 {"lat": (("lat",), e_lat, None),
                  "lon": (("lon",), e_lon, None),
                  "z": (("lat", "lon"), z,
                        {"_FillValue": np.float32(-99999.0)})},
                 {"node_offset": 1})
    tlats = np.array([-45.0, 0.0, 45.0])
    tlons = np.array([-180.0, -179.75, 0.0, 179.75])
    out = b7.block_mean_elev(p, tlats, tlons)
    f = 4
    for j, lon in enumerate(tlons):
        s0 = int(np.floor((lon - e_lon[0]) / step + 0.5)) - f // 2
        want = np.mean([(s0 + k) % len(e_lon) for k in range(f)])
        assert np.allclose(out[:, j], want, atol=1e-3), \
            f"block mean at lon {lon} is {out[0, j]}, want {want}"
    # every target row got the same answer, because z has no latitude structure
    assert np.allclose(out[0], out[-1])


# ------------------------------------------------------------------- 8 -----
def test_8_norm_round_trip():
    """(mean, sd) over every finite value, and un-z-scoring restores it."""
    rng = np.random.default_rng(20260904)
    T, H, W, C = 11, 6, 7, 4
    src = (rng.normal(3.0, 2.0, (T, H, W, C)) * (1 + np.arange(C))
           ).astype(np.float16)
    src[rng.random(src.shape) < 0.25] = np.nan
    src[:, :, :, 2] = np.nan                       # a channel with no values

    X = np.array(src)                              # a plain array behaves alike
    mu, sd, cnt = b7.channel_stats(X, chunk=3)
    for c in range(C):
        v = np.asarray(src[..., c], np.float64)
        v = v[np.isfinite(v)]
        assert int(cnt[c]) == v.size
        if v.size:
            assert abs(mu[c] - v.mean()) < 1e-6
            assert abs(sd[c] - (v.std() + 1e-6)) < 1e-6
        else:
            assert mu[c] == 0.0 and sd[c] == pytest.approx(1e-6)

    i = 0
    while i < T:
        i = b7.zscore_chunk(X, mu.astype(np.float32), sd.astype(np.float32),
                            i, 4)
    back = X.astype(np.float32) * sd.astype(np.float32) + mu.astype(np.float32)
    m = np.isfinite(np.asarray(src, np.float32))
    # float16 carries ~3 decimal digits; the tolerance is that, scaled by sd
    tol = 3e-3 * np.maximum(sd, 1.0)
    err = np.abs(back - np.asarray(src, np.float32))
    assert np.all(err[m] <= np.broadcast_to(tol, err.shape)[m]), \
        f"un-z-scoring does not reproduce the source: max |err| {err[m].max()}"
    assert np.array_equal(np.isfinite(back), m), "the missing pattern moved"


def test_8b_float32_intermediate_beats_the_in_place_float16_path():
    """The coarse groups' f32 -> f16 path, on the channel that motivated it.

    A channel with a large offset — `sp` near 1000 hPa — is quantised at
    0.5 hPa by a float16 RAW write, before the norm stage ever sees it. Filling
    at float32 and writing the float16 ONCE, already z-scored, removes that.
    Both halves are asserted: the round trip reproduces the source to float16,
    and it is strictly better than the in-place path it replaces.
    """
    rng = np.random.default_rng(7)
    T, H, W = 9, 5, 6
    # ch 0: surface pressure in hPa (the motivating case); ch 1: a small field
    truth = np.stack([rng.normal(1010.0, 6.0, (T, H, W)),
                      rng.normal(0.0, 1.0, (T, H, W))], -1).astype(np.float64)
    truth[rng.random(truth.shape) < 0.1] = np.nan

    # --- the path the builder now takes for g100 / rg100 -------------------
    f32 = truth.astype(np.float32)
    mu, sd, _ = b7.channel_stats(f32, chunk=4)
    out = np.zeros(truth.shape, np.float16)
    i = 0
    while i < T:
        i = b7.zscore_chunk(f32, mu.astype(np.float32), sd.astype(np.float32),
                            i, 4, out=out)
    assert np.array_equal(f32, truth.astype(np.float32), equal_nan=True), \
        "the float32 source was modified — the coarse norm is not idempotent"
    new_back = out.astype(np.float32) * sd.astype(np.float32) + mu.astype(np.float32)

    # --- the path it replaces: raw straight into float16, z-scored in place -
    old = truth.astype(np.float16)
    mu_o, sd_o, _ = b7.channel_stats(old, chunk=4)
    i = 0
    while i < T:
        i = b7.zscore_chunk(old, mu_o.astype(np.float32), sd_o.astype(np.float32),
                            i, 4)
    old_back = old.astype(np.float32) * sd_o.astype(np.float32) + mu_o.astype(np.float32)

    m = np.isfinite(truth)
    e_new = np.nanmax(np.where(m, np.abs(new_back - truth), 0)[..., 0])
    e_old = np.nanmax(np.where(m, np.abs(old_back - truth), 0)[..., 0])
    assert e_new < e_old / 4, (
        f"the float32 intermediate bought nothing on the pressure channel: "
        f"{e_new:.4f} hPa vs {e_old:.4f} hPa")
    assert e_old > 0.2, "the fixture must reproduce the 0.5 hPa float16 grid"
    assert e_new <= 3e-3 * sd[0] * 2, \
        f"the f32 -> f16 path does not reproduce the source: {e_new}"
    assert np.array_equal(np.isfinite(new_back), m), "the missing pattern moved"


# ------------------------------------------------------------------- 9 -----
def test_9_loader_three_groups_and_the_alias(build, tmp_path):
    """load_tensor on a three-group stem; the single-group form still loads."""
    d = build["d"]
    for g in b7.GROUPS:
        assert f"X_{g}" in d
        assert isinstance(d[f"X_{g}"], np.memmap)
        assert d[f"X_{g}"].dtype == np.float16
    assert list(d.groups) == b7.GROUPS
    assert d["X"] is d["X_g025"] or np.shares_memory(d["X"], d["X_g025"])
    assert d["X"].shape == d["X_g025"].shape
    assert [f for f in d.files[:4]] == ["X", "X_g025", "X_g100", "X_rg100"]
    assert "nope" not in d and "norm_g100" in d

    # a hand-built three-group stem, and the single-group form beside it
    a = np.arange(24, dtype=np.float16).reshape(2, 3, 4)
    b = np.arange(6, dtype=np.float16).reshape(2, 3)
    stem = str(tmp_path / "multi.npz")
    save_tensor(stem, {"gA": a, "gB": b}, months=np.array(["2010-01"]))
    m = load_tensor(stem)
    assert list(m.groups) == ["gA", "gB"]
    assert np.array_equal(m["X_gA"], a) and np.array_equal(m["X_gB"], b)
    assert np.array_equal(m["X"], a), "X must alias the first declared group"
    assert np.array_equal(np.asarray(m["groups"]), np.array(["gA", "gB"]))

    single = str(tmp_path / "single.npz")
    save_tensor(single, a, months=np.array(["2010-01"]))
    s = load_tensor(single)
    assert np.array_equal(s["X"], a)
    assert s.files[0] == "X" and "X_gA" not in s

    classic = str(tmp_path / "classic.npz")
    np.savez_compressed(classic, X=a, months=np.array(["2010-01"]))
    c = load_tensor(classic)
    assert np.array_equal(c["X"], a) and list(c.files) == list(np.load(classic).files)


# ------------------------------------------------------------------ 10 -----
def test_10_smoke_produces_every_file_and_key(build):
    """The whole path end to end: the four files and every key of §2-§3."""
    work, d = build["work"], build["d"]
    assert os.path.exists(os.path.join(work, b7.STEM + ".npz"))
    for g in b7.GROUPS:
        p = b7.group_file(work, g)
        assert os.path.exists(p), p
        assert os.path.basename(p) == f"{b7.STEM}_X_{g}.npy"
    missing = [k for k in b7.REQUIRED_KEYS if k not in d]
    assert not missing, missing
    assert str(d["recipe"]) == "f7l0"
    assert str(d["window"]) == "global025"
    assert str(d["cadence"]) == "pentad"
    assert list(d["groups"]) == b7.GROUPS
    assert list(d["chan_g025"]) == b7.CHAN_G025
    assert list(d["chan_g100"]) == b7.CHAN_G100
    assert np.asarray(d["sphere"]).shape == (721, 1440)
    assert np.asarray(d["sphere"]).dtype == np.int8
    assert np.asarray(d["elev"]).shape == (721, 1440)
    assert np.asarray(d["elev"]).dtype == np.float32
    for g in b7.GROUPS:
        assert np.asarray(d[f"norm_{g}"]).shape == (b7.NCHAN[g], 2)
    src = json.loads(str(d["sources"]))
    for k in ("glorys", "oisst", "ncep", "rg", "naturalearth", "etopo"):
        assert k in src, f"`sources` does not name {k}"
    # every stage left a marker, and progress.json is a real artefact
    for s in b7.STAGES:
        if s == "publish":
            continue
        assert b7.marked(work, s), f"stage {s} left no marker"
    prog = json.load(open(os.path.join(work, "progress.json")))
    assert {"stage", "item", "elapsed_s"} <= set(prog)
    # the truth guard is armed: the labels are attached, under both names
    assert np.asarray(d["rapid"]).ndim == 2
    assert np.array_equal(np.asarray(d["rapid"]), np.asarray(d["truth_rapid"]))


def test_10b_smoke_cli_runs(tmp_path):
    """`python3 ml/build_family7.py --smoke` itself, as a subprocess."""
    r = subprocess.run([sys.executable, os.path.join(ML, "build_family7.py"),
                        "--smoke"], capture_output=True, text=True,
                       timeout=600)
    assert r.returncode == 0, r.stdout[-3000:] + r.stderr[-3000:]
    assert "smoke     OK" in r.stdout, r.stdout[-2000:]


# --------------------------------------------------------- resumability ----
def test_11_glorys_resumes_across_a_month_boundary(tmp_path):
    """A killed job resumes with the PARTIAL bin it had, not a thinner mean.

    Beyond §5's ten, and the reason it is here: a pentad bin can straddle a
    month boundary, so the accumulator for the bin holding 31 Jan / 1 Feb is
    only completed by the NEXT chunk. `carry.npz` is written atomically before
    the chunk's marker (flush THEN mark, ml/CLAUDE.md §5.21) — without it a
    resume would silently write a mean over half the days and nothing would
    say so. This asserts the resumed build is BIT-IDENTICAL to the one-pass
    build, which is the only statement that catches it.
    """
    import argparse
    src_all = str(tmp_path / "all")
    d_lo = dt.date(*(int(x) for x in b7.SMOKE_START.split("-")))
    d_hi = dt.date(*(int(x) for x in b7.SMOKE_END.split("-")))
    b7.make_smoke_sources(src_all, d_lo, d_hi)
    chunks = sorted(os.listdir(os.path.join(src_all, "daily025_global")))
    assert len(chunks) >= 2, "the fixture must span a month boundary"

    def ctx_for(work, src):
        return b7.Ctx(argparse.Namespace(
            work=work, source_dir=src, start=b7.SMOKE_START,
            end=b7.SMOKE_END, force=False, stage="glorys", smoke=True))

    one = str(tmp_path / "one")
    b7.stage_glorys(ctx_for(one, src_all))
    whole = np.load(b7.group_file(one, "g025"), mmap_mode="r")

    # ---- the same build, killed after the first chunk ---------------------
    src_part = str(tmp_path / "part")
    os.makedirs(os.path.join(src_part, "daily025_global"))
    shutil.copy(os.path.join(src_all, "daily025_global", chunks[0]),
                os.path.join(src_part, "daily025_global", chunks[0]))
    two = str(tmp_path / "two")
    ctx = ctx_for(two, src_part)
    b7.stage_glorys(ctx)                       # "crashes" after chunk 0
    part = np.load(b7.group_file(two, "g025"), mmap_mode="r")
    assert not np.array_equal(np.asarray(part), np.asarray(whole),
                              equal_nan=True), \
        "the one-chunk build already equals the whole one — nothing carried"
    assert os.path.exists(os.path.join(two, "glorys",
                                       f"carry_{chunks[0].split('_')[-1][:6]}.npz"))
    os.remove(b7.marker(two, "glorys"))        # the stage marker, not the chunk
    for c in chunks[1:]:
        shutil.copy(os.path.join(src_all, "daily025_global", c),
                    os.path.join(src_part, "daily025_global", c))
    b7.stage_glorys(ctx_for(two, src_part))    # resumes at the chunk it lost

    resumed = np.load(b7.group_file(two, "g025"), mmap_mode="r")
    assert np.array_equal(np.asarray(whole), np.asarray(resumed),
                          equal_nan=True), \
        "the resumed build differs from the one-pass build"
    assert b7.read_json(os.path.join(two, "counts.json"))["n_glorys_bins"] == \
        b7.read_json(os.path.join(one, "counts.json"))["n_glorys_bins"]

    # ---- and the nastier crash: BETWEEN the carry write and the marker ----
    # Keying the carry by its chunk is what makes this recoverable: the
    # unmarked chunk's carry is ignored and the chunk is simply replayed.
    three = str(tmp_path / "three")
    real_mark = b7.mark
    victim = f"glorys/{chunks[1].split('_')[-1][:6]}"

    def killed(work, name):
        if name == victim:
            raise RuntimeError("box destroyed between the carry and the marker")
        real_mark(work, name)

    b7.mark = killed
    try:
        with pytest.raises(RuntimeError):
            b7.stage_glorys(ctx_for(three, src_all))
    finally:
        b7.mark = real_mark
    assert not b7.marked(three, victim)
    b7.stage_glorys(ctx_for(three, src_all))
    assert np.array_equal(np.asarray(whole),
                          np.asarray(np.load(b7.group_file(three, "g025"),
                                             mmap_mode="r")), equal_nan=True), \
        "a crash between the carry and the marker double-counted or lost a chunk"


def test_12_stage_order_is_enforced(tmp_path):
    """`ncep` before `sst` is refused, naming the marker it wants."""
    import argparse
    ctx = b7.Ctx(argparse.Namespace(
        work=str(tmp_path / "w"), source_dir="", start=b7.SMOKE_START,
        end=b7.SMOKE_END, force=False, stage="ncep", smoke=False))
    with pytest.raises(SystemExit) as e:
        b7.run_stages(ctx, ["ncep"])
    assert "sst" in str(e.value)


def test_5b_min_days_counts_days_not_six_hourly_samples(tmp_path):
    """PSL's `surface_gauss` files are 4x DAILY, so >= 3 samples is 18 hours.

    Measured 2026-09-04: `weasd.sfc.gauss.2020.nc` carries 1464 steps for 2020
    (`delta_t 06:00:00`, "4x daily NMC reanalysis"). The guard the plan states
    is >= 3 DAYS, so a bin holding two full days — eight samples — must stay
    missing, and a bin holding four days must not.
    """
    import argparse
    src = str(tmp_path / "src")
    work = str(tmp_path / "work")
    n_lat, n_lon = 94, 192
    g_lat = np.linspace(88.542, -88.542, n_lat)
    g_lon = np.arange(n_lon) * 1.875

    # bin A gets 2 days (8 six-hourly steps); bin B gets 4 days (16 steps)
    b_a = ac.bin_index(dt.date(2010, 1, 14), 5)
    b_b = b_a + 1
    a_days = [ac.bin_start(b_a, 5) + dt.timedelta(days=k) for k in range(2)]
    b_days = [ac.bin_start(b_b, 5) + dt.timedelta(days=k) for k in range(4)]
    stamps = [(d, h) for d in a_days + b_days for h in (0, 6, 12, 18)]
    hours = np.array([(d - b7.EPOCH).days * 24.0 + h for d, h in stamps])
    units = f"hours since {b7.EPOCH} 00:00:00"

    b7._nc_write(os.path.join(src, "ncep", b7.NCEP_LAND + ".nc"),
                 {"time": 1, "lat": n_lat, "lon": n_lon},
                 {"lat": (("lat",), g_lat, None), "lon": (("lon",), g_lon, None),
                  "time": (("time",), np.zeros(1), {"units": units}),
                  "land": (("time", "lat", "lon"),
                           np.ones((1, n_lat, n_lon), np.float32), None)})
    for key, stem in b7.NCEP_FILES.items():
        vals = (np.arange(len(stamps), dtype=np.float32)[:, None, None]
                + np.zeros((1, n_lat, n_lon), np.float32))
        b7._nc_write(os.path.join(src, "ncep", f"{stem}.2010.nc"),
                     {"time": len(stamps), "lat": n_lat, "lon": n_lon},
                     {"lat": (("lat",), g_lat, None),
                      "lon": (("lon",), g_lon, None),
                      "time": (("time",), hours, {"units": units}),
                      key: (("time", "lat", "lon"), vals, None)})

    ctx = b7.Ctx(argparse.Namespace(
        work=work, source_dir=src, start=str(ac.bin_start(b_a, 5)),
        end=str(ac.bin_start(b_b, 5) + dt.timedelta(days=4)),
        force=False, stage="ncep", smoke=True))
    # the skin_t fill needs the OISST mask; make OISST own every cell so this
    # test is only about the g100 guard
    b7.open_group(work, "g025", ctx.shapes()["g025"], create=True)
    np.save(os.path.join(work, "oisst_seen.npy"), np.ones((721, 1440), bool))
    b7.stage_ncep(ctx)

    # the ncep stage writes the float32 intermediate; norm converts it later
    X = np.load(b7.raw_file(work, "g100"), mmap_mode="r")
    bins = ctx.bins
    ra, rb = bins.index(b_a), bins.index(b_b)
    assert not np.isfinite(np.asarray(X[ra], np.float32)).any(), \
        "a bin with two days (eight 6-hourly samples) passed the >= 3 DAY guard"
    got = np.asarray(X[rb, :, :, 5], np.float32)          # u10, no transform
    assert np.isfinite(got).all(), "a bin with four days was rejected"
    assert np.allclose(got, np.mean(np.arange(8, 24)), atol=0.05), \
        "the pentad mean is not the mean of the bin's 6-hourly samples"
    assert b7.read_json(os.path.join(work, "counts.json"))["n_ncep_days"] == 6, \
        "n_ncep_days counts 6-hourly steps rather than days"


def test_13_resume_over_the_pre_correction_box_state(tmp_path):
    """The work dir the real build left, repaired without a flag.

    THE STATE ON THE BOX, reproduced exactly: `glorys.done` and `sst.done`
    correct, then an `ncep` that completed 1982-1984 under the OLD recipe — a
    14-channel g100 float32, per-year markers and a carry for those three
    years, and the retired NCEP skin-temperature fill written into g025's
    temperature channel for their bins.

    What the new code must do, automatically: repair g025 channel 5 back to
    "OISST or nothing", discard the ncep stage's OWN markers, carry and f32,
    rebuild g100 with fifteen channels — and leave `glorys.done` and
    `sst.done` alone, because those bytes are still correct.
    """
    import argparse
    src = str(tmp_path / "src")
    work = str(tmp_path / "work")
    d_lo = dt.date(*(int(x) for x in b7.SMOKE_START.split("-")))
    d_hi = dt.date(*(int(x) for x in b7.SMOKE_END.split("-")))
    b7.make_smoke_sources(src, d_lo, d_hi)
    ctx = b7.Ctx(argparse.Namespace(
        work=work, source_dir=src, start=b7.SMOKE_START, end=b7.SMOKE_END,
        force=False, stage="all", smoke=True))

    b7.run_stages(ctx, ["glorys", "sst"])
    glorys_stamp = open(b7.marker(work, "glorys")).read()
    sst_stamp = open(b7.marker(work, "sst")).read()
    oisst_seen = np.load(os.path.join(work, "oisst_seen.npy"))
    assert (~oisst_seen).any(), "the fixture must have cells OISST never sees"

    # ---- forge the pre-correction state -----------------------------------
    T = ctx.T
    old_g100 = np.lib.format.open_memmap(
        b7.raw_file(work, "g100"), mode="w+", dtype=np.float32,
        shape=(T, 181, 360, 14))                    # FOURTEEN channels
    old_g100[:] = 1.5
    old_g100.flush()
    del old_g100
    for y in ("1982", "1983", "1984", str(d_lo.year)):
        b7.mark(work, f"ncep/{y}")
    np.savez(os.path.join(work, "ncep", f"carry_{d_lo.year}.npz"),
             n_days=np.array(99))
    # ...and the retired land fill in g025's temperature channel
    X = np.lib.format.open_memmap(b7.group_file(work, "g025"), mode="r+")
    forged = np.asarray(X[:, :, :, b7.C_SST], np.float32)
    forged[:, ~oisst_seen] = -11.0
    X[:, :, :, b7.C_SST] = forged
    X.flush()
    del X, forged
    n_forged = int(T * (~oisst_seen).sum())

    # ---- resume, with no operator flag ------------------------------------
    b7.run_stages(ctx, ["ncep"])

    X = np.load(b7.group_file(work, "g025"), mmap_mode="r")
    sst = np.asarray(X[:, :, :, b7.C_SST], np.float32)
    assert not np.isfinite(sst[:, ~oisst_seen]).any(), (
        f"the repair left {int(np.isfinite(sst[:, ~oisst_seen]).sum())} of "
        f"{n_forged} reanalysis values in the observed `sst` channel")
    assert np.isfinite(sst[:, oisst_seen]).any(), \
        "the repair also erased the real OISST values"

    g100 = np.load(b7.raw_file(work, "g100"), mmap_mode="r")
    assert g100.shape == (T, 181, 360, 15), \
        f"g100 is {g100.shape} — the 14-channel array was not discarded"
    assert not np.allclose(np.asarray(g100[:, :, :, 0], np.float32), 1.5), \
        "the forged values survived — the stage reused the stale file"
    assert np.isfinite(np.asarray(g100[:, :, :, b7.C_SKT], np.float32)).any(), \
        "skt is empty after the rebuild"
    for y in ("1982", "1983", "1984"):
        assert not b7.marked(work, f"ncep/{y}"), \
            f"ncep/{y} survived — the stale per-year markers were not discarded"

    # nothing that belonged to another stage was touched
    assert open(b7.marker(work, "glorys")).read() == glorys_stamp
    assert open(b7.marker(work, "sst")).read() == sst_stamp
    assert b7.marked(work, "repair_sst")
    assert os.path.exists(os.path.join(work, "ncep.spec"))

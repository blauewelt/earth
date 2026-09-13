#!/usr/bin/env python3
"""The things asserted before the family-7 global tensor is trusted.

`ml/plans/E070_family7_build.md` §5 and `ml/plans/E077_family7_ocean_colour.md`
§5/§7. Family 7 is the first input tensor covering the whole globe rather than
the North Atlantic window: every 0.25-degree grid point from pole to pole, one
value per channel per five-day bin, 1982-2024, in four groups at their native
resolution (`g025` 0.25 deg / 7 channels, `g100` 1 deg / 15 NCEP channels,
`rg100` 1 deg / 32 Argo depth channels on the live bins only, and — recipe
`f7l1`, E-077 — `oc025` 0.25 deg / 2 ocean-colour channels on an OFFSET time
axis that starts at the colour record's own first day).

THE COLOUR SOURCE IS SYNTHETIC AND, WHERE IT IS DECIMATED, SAYS SO. The exact
block-mean values of E-077 §5 are asserted on the REAL 4320 x 8640 OC-CCI grid
(built in memory, ~550 MB, a few seconds); the end-to-end smoke uses a
DECLARED decimated 1440 x 2880 grid at 1/8 degree, which is the coarsest
global grid on which a 0.25-degree point still sits on a cell boundary with an
even block, so the pole truncation stays symmetric. Neither host
(`www.oceancolour.org`, `dap.ceda.ac.uk`) is reachable from a test runner and
neither is contacted.

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
               seed=os.path.join(root, "seed"),
               start=b7.SMOKE_START, end=b7.SMOKE_END,
               oc_start=b7.SMOKE_OC_START)
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
    assert str(d["recipe"]) == "f7l1"
    assert str(d["window"]) == "global025"
    assert str(d["cadence"]) == "pentad"
    assert list(d["groups"]) == b7.GROUPS == ["g025", "g100", "rg100", "oc025"]
    assert list(d["chan_g025"]) == b7.CHAN_G025
    assert list(d["chan_g100"]) == b7.CHAN_G100
    assert list(d["chan_oc025"]) == b7.CHAN_OC025 == ["log_chl", "chl_cov"]
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


# ======================================================================== #
# E-077 · the fourth group: ocean colour (recipe f7l1)                     #
# ======================================================================== #
OC_REAL_NLAT, OC_REAL_NLON = 4320, 8640        # the OC-CCI v6.0 4 km grid
OC_REAL_STEP = 1.0 / 24.0
OC_FILL = -999.0


def oc_real_axes():
    """The real product's axes: cell-registered, NORTH-first, from -180."""
    lat = 90.0 - OC_REAL_STEP * (np.arange(OC_REAL_NLAT) + 0.5)
    lon = -180.0 + OC_REAL_STEP * (np.arange(OC_REAL_NLON) + 0.5)
    return lat, lon


def oc_cells_by_geography(y, x, s_lat, s_lon):
    """Which source cells belong to point (y, x), FROM THE DEFINITION.

    E-077 §3 says "the 6 x 6 block of 4 km cells whose centres lie within
    +-0.125 deg of the point", so this test computes exactly that — a distance
    test on the axes, with longitude wrapped — rather than calling the
    builder's index arithmetic. The two must agree; if they only ever agree
    with each other, neither has been tested.
    """
    lats, lons = b7.grid025()
    lat, lon = float(lats[y]), float(lons[x])
    rows = np.flatnonzero(np.abs(s_lat - lat) < 0.125 - 1e-12)
    dl = (s_lon - lon + 180.0) % 360.0 - 180.0
    cols = np.flatnonzero(np.abs(dl) < 0.125 - 1e-12)
    return rows.tolist(), cols.tolist()


# ----------------------------------------------------------------- 14 -----
def test_14_oc_block_geometry_is_the_plan_s_own_definition():
    """6 x 6 cells within +-0.125 deg, the poles truncated, the dateline wrapped.

    E-077 §3 and §5. The block is EXACT rather than approximate because a
    0.25-degree point sits on a cell boundary of the 1/24-degree grid: three
    source cells each side, never a cell split between two points. Everything
    downstream — the coverage denominator, the pole truncation, the wrap —
    follows from that, so it is asserted against the geographic definition
    before any array is touched.
    """
    s_lat, s_lon = oc_real_axes()
    assert b7.oc_block_factors(s_lat, s_lon) == (6, 6)

    # the interior: exactly 36 cells, and they are the ones within 0.125 deg
    for y, x in [(360, 720), (123, 4), (500, 1000), (1, 7), (719, 3)]:
        rows, cols = b7.oc_block_slices(y, x, 6, 6, OC_REAL_NLAT, OC_REAL_NLON)
        g_rows, g_cols = oc_cells_by_geography(y, x, s_lat, s_lon)
        assert sorted(rows) == g_rows, (y, x)
        assert sorted(cols) == sorted(g_cols), (y, x)
        assert len(rows) * len(cols) == 36

    # THE DATELINE. x = 0 is lon -180: half the block is at +179.9, which is
    # the same meridian. A block that did not wrap would be three cells wide
    # and the whole column would read half its neighbours.
    rows, cols = b7.oc_block_slices(300, 0, 6, 6, OC_REAL_NLAT, OC_REAL_NLON)
    assert sorted(cols) == [0, 1, 2, 8637, 8638, 8639]
    assert len(set(cols)) == 6

    # THE POLES. The point at -90 (and +90) sits ON the edge of the raster, so
    # three of its six rows are off it; the block is 3 x 6 = 18 cells.
    cells = b7.oc_block_cells(6, 6, OC_REAL_NLAT)
    assert cells.shape == (721,)
    assert int(cells[0]) == int(cells[-1]) == 18
    assert int(cells[1]) == int(cells[360]) == int(cells[-2]) == 36
    for y in (0, 720):
        rows, _ = b7.oc_block_slices(y, 700, 6, 6, OC_REAL_NLAT, OC_REAL_NLON)
        assert len(rows) == 3, y
        assert all(0 <= r < OC_REAL_NLAT for r in rows)
        assert sorted(rows) == oc_cells_by_geography(y, 700, s_lat, s_lon)[0]

    # A source whose spacing does not divide 0.25 is REFUSED, not rounded.
    bad_lat = 90.0 - (180.0 / 4000) * (np.arange(4000) + 0.5)
    with pytest.raises(SystemExit) as e:
        b7.oc_block_factors(bad_lat, s_lon)
    assert "0.25" in str(e.value)


# ----------------------------------------------------------------- 15 -----
def test_15_oc_block_mean_exact_values_on_the_real_grid():
    """E-077 §5's three exact cases, on the REAL 4320 x 8640 grid.

    Not a decimated one: the numbers the plan states — `log_chl = 0.5`,
    `chl_cov = 1.0`, and `1/180` for a single clear cell on a single day — are
    numbers about 36 cells and 5 days, and asserting them anywhere else would
    be asserting different numbers. One full-size float32 raster is ~150 MB and
    the whole test runs in a few seconds.

    The MEAN IS IN LOG SPACE, and the third block is the case that proves it:
    a block holding 10**2 and 10**-2 mg/m3 reads 0.0, where an arithmetic mean
    of the concentrations would read log10(50.005) = 1.699.
    """
    s_lat, s_lon = oc_real_axes()
    blk = (6, 6)
    cells = b7.oc_block_cells(*blk, OC_REAL_NLAT)
    P_UNIF, P_ONE, P_NONE, P_MIX = (400, 900), (401, 901), (402, 902), (403, 903)

    def day(k):
        a = np.full((OC_REAL_NLAT, OC_REAL_NLON), np.float32(OC_FILL))
        r, c = b7.oc_block_slices(*P_UNIF, *blk, OC_REAL_NLAT, OC_REAL_NLON)
        a[np.ix_(r, c)] = np.float32(10.0 ** 0.5)
        if k == 0:
            r, c = b7.oc_block_slices(*P_ONE, *blk, OC_REAL_NLAT, OC_REAL_NLON)
            a[r[0], c[0]] = np.float32(10.0 ** -0.7)
            r, c = b7.oc_block_slices(*P_MIX, *blk, OC_REAL_NLAT, OC_REAL_NLON)
            a[r[0], c[0]] = np.float32(100.0)
            a[r[1], c[1]] = np.float32(0.01)
        return a

    # The pentad rule, stated here rather than borrowed: the mean of the DAILY
    # block means over the days that had >= 1 finite cell, and the coverage is
    # finite cell-days over (cells in the block x days in the bin).
    acc = np.zeros((b7.NLAT, b7.NLON), np.float64)
    nday = np.zeros((b7.NLAT, b7.NLON), np.int64)
    ncell = np.zeros((b7.NLAT, b7.NLON), np.int64)
    for k in range(b7.PENTAD_DAYS):
        S, C = b7.oc_block_stats(day(k), *blk, fill=OC_FILL)
        got = C > 0
        with np.errstate(invalid="ignore"):
            acc += np.where(got, S / np.maximum(C, 1), 0.0)
        nday += got
        ncell += C
    with np.errstate(invalid="ignore"):
        log_chl = np.where(nday > 0, acc / np.maximum(nday, 1), np.nan)
        chl_cov = np.where(nday > 0,
                           ncell / (cells[:, None] * b7.PENTAD_DAYS), np.nan)

    # (a) 36 cells at 10**0.5 on all five days
    assert log_chl[P_UNIF] == pytest.approx(0.5, abs=1e-6)
    assert chl_cov[P_UNIF] == pytest.approx(1.0, abs=1e-12)
    assert int(ncell[P_UNIF]) == 36 * b7.PENTAD_DAYS

    # (b) ONE finite cell on ONE day -> that cell's log10, coverage 1/180
    assert log_chl[P_ONE] == pytest.approx(-0.7, abs=1e-6)
    assert chl_cov[P_ONE] == pytest.approx(1.0 / 180.0, rel=1e-12)
    assert int(cells[P_ONE[0]]) * b7.PENTAD_DAYS == 180

    # (c) an all-fill block is NaN in BOTH channels — never 0
    assert not np.isfinite(log_chl[P_NONE])
    assert not np.isfinite(chl_cov[P_NONE])

    # (d) the mean really is taken in LOG space
    assert log_chl[P_MIX] == pytest.approx(0.0, abs=1e-5), \
        "the block mean was taken on the concentration, not on log10"
    assert np.log10(np.mean([100.0, 0.01])) == pytest.approx(1.699, abs=1e-3)
    assert chl_cov[P_MIX] == pytest.approx(2.0 / 180.0, rel=1e-12)


# ----------------------------------------------------------------- 16 -----
def test_16_oc_pole_truncation_and_longitude_wrap_are_averaged(build):
    """A pole block averages what EXISTS; a dateline block averages both sides.

    The pole case is the one that can be wrong in two directions at once: the
    mean can be taken over six rows of which three are padding (dragging the
    value toward zero) or the coverage denominator can stay at 36 (halving the
    reported coverage of a fully observed block). Both are asserted, on the
    real grid, with values chosen so either mistake changes the answer.
    """
    blk = (6, 6)
    cells = b7.oc_block_cells(*blk, OC_REAL_NLAT)
    a = np.full((OC_REAL_NLAT, OC_REAL_NLON), np.float32(OC_FILL))
    for y in (0, 720):
        r, c = b7.oc_block_slices(y, 700, *blk, OC_REAL_NLAT, OC_REAL_NLON)
        a[np.ix_(r, c)] = np.float32(10.0 ** 1.25)
    # the wrap block: its WEST half (beyond the dateline) and EAST half differ
    rw, cw = b7.oc_block_slices(300, 0, *blk, OC_REAL_NLAT, OC_REAL_NLON)
    for col in cw:
        a[np.ix_(rw, [col])] = np.float32(10.0 ** (-1.0 if col > 4320 else 1.0))
    S, C = b7.oc_block_stats(a, *blk, fill=OC_FILL)

    for y in (0, 720):
        assert int(C[y, 700]) == 18 == int(cells[y]), \
            "a pole block did not truncate to the rows that exist"
        assert S[y, 700] / C[y, 700] == pytest.approx(1.25, abs=1e-5), \
            "a pole block's mean was dragged by rows that are not there"
        assert C[y, 700] / (cells[y] * 1) == pytest.approx(1.0), \
            "a fully observed pole block does not report full coverage"
    assert int(C[300, 0]) == 36, "the dateline block lost half its cells"
    assert S[300, 0] / C[300, 0] == pytest.approx(0.0, abs=1e-5), \
        "the block at lon -180 is not the mean of both sides of the dateline"

    # ...and the built tensor agrees: its pole rows carry colour.
    d = build["d"]
    chl = np.asarray(d["X_oc025"][..., b7.C_LOG_CHL], np.float32)
    assert np.isfinite(chl[:, 0, :]).any(), "the South Pole row is empty"
    assert np.isfinite(chl[:, -1, :]).any(), "the North Pole row is empty"
    assert np.isfinite(chl[:, :, 0]).any(), "the dateline column is empty"


# ----------------------------------------------------------------- 17 -----
def test_17_oc_nan_and_coverage_invariants(build):
    """E-077 §7.3, on the published bytes: the pair is readable or absent.

    `chl_cov` is NaN EXACTLY where `log_chl` is. Storing 0 for "saw nothing"
    would collide with the family-7 missing-token design (handover §5, the
    `sea_ice` case): the consumer's whole distinction between 0 and NaN is
    "measured as none" against "not measured".
    """
    d = build["d"]
    chl = unz(d, "oc025", b7.C_LOG_CHL)
    cov = unz(d, "oc025", b7.C_CHL_COV)
    assert np.asarray(d["X_oc025"]).dtype == np.float16
    assert np.asarray(d["X_oc025"]).shape[1:] == (721, 1440, 2)
    assert list(d["chan_oc025"]) == ["log_chl", "chl_cov"]

    fin_c, fin_v = np.isfinite(chl), np.isfinite(cov)
    assert fin_c.any() and (~fin_c).any(), \
        "the fixture must exercise both observed and unobserved colour"
    assert np.array_equal(fin_c, fin_v), \
        "log_chl and chl_cov do not go missing together"

    v = cov[fin_v]
    # the float16 round trip can move 1.0 by ~1 part in 2000; (0, 1] is the rule
    tol = 3e-3 * float(np.asarray(d["norm_oc025"])[b7.C_CHL_COV][1]) + 1e-4
    assert float(v.min()) > 0.0, "a finite chl_cov of 0 means 'observed nothing'"
    assert float(v.max()) <= 1.0 + tol, f"chl_cov exceeds 1: {v.max()}"
    assert not (np.isfinite(cov) & (cov == 0.0)).any()

    # the coverage is a quantised fraction of (cells x 5 days), so its finest
    # step on this fixture is 1/(cells*5) — a value below that is impossible
    cells = b7.oc_block_cells(*b7._smoke_blk(), b7.SMOKE_OC_NLAT)
    finest = 1.0 / (int(cells.max()) * b7.PENTAD_DAYS)
    assert float(v.min()) >= finest - tol, \
        f"a coverage of {v.min()} is finer than one cell-day ({finest})"

    # `n_oc_inland` is a MEASUREMENT, not a mask: it exists and is a count.
    assert int(np.asarray(d["n_oc_inland"])) >= 0
    assert int(np.asarray(d["n_occci_days"])) > 0
    assert int(np.asarray(d["n_occci_absent"])) >= 0


# ----------------------------------------------------------------- 18 -----
def test_18_the_offset_axis_and_its_consumers(build):
    """E-077 §4 layout 1: row = bin - oc_bin_first, and who has to know.

    Layout 1 was chosen over the full 3142-row axis because the consumers
    ALREADY carried a per-group row lookup — `ml/cone_sampler.py` translates
    any group whose row count differs from the master's through its own bin
    index, and the app's `tensorRowOf` needed one line. This asserts both the
    arithmetic and that the cone sampler really does read it, because "the
    consumer supports it" is the claim the whole layout choice rests on.
    """
    d = build["d"]
    first = int(np.asarray(d["oc_bin_first"]))
    bins = [int(b) for b in np.asarray(d["bin_index"])]
    T_oc = int(np.asarray(d["X_oc025"]).shape[0])

    # COMPUTED from the date, never typed (§7.2)
    y, m, dd = (int(v) for v in build["oc_start"].split("-"))
    assert first == ac.bin_index(dt.date(y, m, dd), 5)
    assert first > bins[0], "the fixture must exercise a NON-zero offset"
    assert first + T_oc - 1 == bins[-1]
    assert np.array_equal(np.asarray(d["oc025_bin_index"]),
                          np.arange(first, first + T_oc))

    # and on the real axis the plan's own number falls out of the same call
    assert ac.bin_index(dt.date(1997, 9, 4), 5) == 1145
    assert ac.bin_index(dt.date(2024, 12, 31), 5) + 1 - 1145 == 1997

    # the cone sampler reads the group with no new mechanism
    from cone_sampler import GroupSet
    gs = GroupSet.from_tensor(d)
    assert gs.names == ["g025", "g100", "rg100", "oc025"]
    oc = gs.groups[-1]
    assert oc.factor == 1, "oc025 is 0.25 deg — the master's own resolution"
    assert oc.row_of_bin is not None, \
        "the cone sampler treats oc025 as bin-aligned; it is offset"
    for r, b in enumerate(range(first, first + T_oc)):
        assert int(oc.row_of_bin[bins.index(b)]) == r
    for b in bins:
        if b < first:
            assert int(oc.row_of_bin[bins.index(b)]) == -1, \
                "a bin before the colour record maps to a row instead of a miss"
    import cone_sampler as cs
    assert cs.GROUP_FOOTPRINT["oc025"] == (0.0, 0.0)


# ----------------------------------------------------------------- 19 -----
def test_19_seed_from_is_byte_identical_and_leaves_the_seed_alone(build):
    """E-077 §5: the three inherited groups are the SAME INODE and unchanged.

    "It hard-linked" is an intention. `samefile` plus a sha256 comparison plus
    the seed directory's own mtimes are the facts, and the last one is the
    important one: a build that wrote through its own hard link would have
    corrupted the published f7l0 tensor, whose hashes the Hub, the handover and
    another agent's hand-off all cite.
    """
    work, seed = build["work"], build["seed"]
    assert os.path.isdir(seed)
    rec = json.load(open(os.path.join(work, "seed.json")))
    assert rec["base_recipe"] == b7.BASE_RECIPE == "f7l0"
    assert rec["base_stem"] == b7.BASE_STEM
    assert {r["group"] for r in rec["linked"]} == set(b7.BASE_GROUPS)

    for g in b7.BASE_GROUPS:
        old = os.path.join(seed, f"{b7.BASE_STEM}_X_{g}.npy")
        new = b7.group_file(work, g)
        assert os.path.exists(old) and os.path.exists(new)
        assert os.path.samefile(old, new), f"{g} is a copy, not a hard link"
        assert b7.sha256(old) == b7.sha256(new)
        assert os.stat(new).st_nlink >= 2
        m = np.lib.format.open_memmap(new, mode="r")
        assert m.dtype == np.float16
        assert os.path.getsize(new) == 128 + int(np.prod(m.shape)) * 2
        del m
    # the colour group is NOT inherited — it is this build's own file
    assert not os.path.exists(os.path.join(seed,
                                           f"{b7.BASE_STEM}_X_oc025.npy"))
    assert os.stat(b7.group_file(work, "oc025")).st_nlink == 1

    # the seed still looks like a finished f7l0 build: markers, specs, norms
    for name in b7.INHERITED_STAGES:
        assert b7.marked(seed, name), name
        assert b7.marked(work, name), f"{name} was not inherited"
    assert os.path.exists(os.path.join(work, "norm.npz"))
    assert os.path.exists(os.path.join(work, "rg", "live.npz"))


def test_19b_seed_from_refuses_rather_than_rebuilding(tmp_path):
    """A missing or unfinished seed is an ERROR, never a silent full rebuild.

    ml/CLAUDE.md §0.2: a step that reports success is not evidence it did
    anything. A `--seed-from` that shrugged at an absent directory would spend
    five hours rebuilding 53 GB and look exactly like a successful inherit.
    """
    import argparse
    ctx = b7.Ctx(argparse.Namespace(
        work=str(tmp_path / "w"), source_dir="", start=b7.SMOKE_START,
        end=b7.SMOKE_END, force=False, stage="all", smoke=True))
    with pytest.raises(SystemExit) as e:
        b7.seed_from(ctx, str(tmp_path / "nope"))
    assert "no such directory" in str(e.value)

    half = tmp_path / "half"
    half.mkdir()
    b7.mark(str(half), "glorys")
    with pytest.raises(SystemExit) as e:
        b7.seed_from(ctx, str(half))
    assert b7.BASE_STEM in str(e.value) and "FINISHED" in str(e.value)

    # a truncated group file is caught on the BYTES, not on the link call
    trunc = tmp_path / "trunc"
    trunc.mkdir()
    for g in b7.BASE_GROUPS:
        p = str(trunc / f"{b7.BASE_STEM}_X_{g}.npy")
        np.lib.format.open_memmap(p, mode="w+", dtype=np.float16,
                                  shape=(2, 3, 4, b7.NCHAN[g]))
    with open(str(trunc / f"{b7.BASE_STEM}_X_g100.npy"), "r+b") as fh:
        fh.truncate(os.path.getsize(fh.name) - 64)
    ctx2 = b7.Ctx(argparse.Namespace(
        work=str(tmp_path / "w2"), source_dir="", start=b7.SMOKE_START,
        end=b7.SMOKE_END, force=False, stage="all", smoke=True))
    with pytest.raises(SystemExit) as e:
        b7.seed_from(ctx2, str(trunc))
    assert "bytes" in str(e.value)


# ----------------------------------------------------------------- 20 -----
def test_20_norm_is_group_idempotent(build, tmp_path):
    """E-077 §5: `norm` answers per GROUP, and never re-z-scores one.

    The seeded directory arrives with a truthful `norm.done` covering three
    groups and a fourth that has never been normalised. A stage marker cannot
    express that, so `norm_pending` asks the question the marker cannot — and
    the thing it must never do is return a group that is already z-scored,
    because g025's pass is in place and redoing it would square the transform
    with nothing downstream to say so.
    """
    import argparse
    work = build["work"]
    ctx = b7.Ctx(argparse.Namespace(
        work=work, source_dir=build["src"], start=build["start"],
        end=build["end"], force=False, stage="norm", smoke=True,
        oc_start=build["oc_start"]))
    assert b7.marked(work, "norm")
    assert b7.norm_pending(ctx) == [], \
        "a finished build still reports normalisation work"
    for g in b7.GROUPS:
        stats, final, mk = b7.norm_state(ctx, g)
        assert stats and final and mk, g

    # re-running the whole stage is a no-op on the BYTES, not just on the log
    before = {g: b7.sha256(b7.group_file(work, g)) for g in b7.GROUPS}
    b7.run_stages(ctx, ["norm"])
    after = {g: b7.sha256(b7.group_file(work, g)) for g in b7.GROUPS}
    assert before == after, "re-entering norm re-scaled an already-scaled group"

    # ...and a work dir where only oc025 is missing asks for oc025 ALONE
    half = str(tmp_path / "half")
    os.makedirs(half)
    shutil.copy(os.path.join(work, "norm.npz"), os.path.join(half, "norm.npz"))
    os.makedirs(os.path.join(half, "rg"))
    shutil.copy(os.path.join(work, "rg", "live.npz"),
                os.path.join(half, "rg", "live.npz"))
    d = np.load(os.path.join(half, "norm.npz"))
    keep = {k: d[k] for k in d.files if not k.endswith("oc025")}
    b7.atomic_npz(os.path.join(half, "norm.npz"), **keep)
    for g in b7.BASE_GROUPS:
        os.link(b7.group_file(work, g), b7.group_file(half, g))
        b7.mark(half, f"norm/{g}")
    b7.mark(half, "norm")
    ctx2 = b7.Ctx(argparse.Namespace(
        work=half, source_dir=build["src"], start=build["start"],
        end=build["end"], force=False, stage="norm", smoke=True,
        oc_start=build["oc_start"]))
    assert b7.norm_pending(ctx2) == ["oc025"]

    # a group whose f16 vanished and has no float32 source REFUSES rather than
    # normalising unknown bytes
    os.remove(b7.group_file(half, "g025"))
    with pytest.raises(SystemExit) as e:
        b7.norm_pending(ctx2)
    assert "IN PLACE" in str(e.value)


# ----------------------------------------------------------------- 21 -----
def test_21_inherited_spec_digests_are_unchanged_by_the_f7l1_builder(build):
    """E-077 §5: the six inherited stages still hash as they did under f7l0.

    THE FAILURE THIS CATCHES. `stage_state_check` discards a stage's markers,
    carries and fill file when its recorded `.spec` no longer matches — which
    is exactly right when a recipe moves and exactly catastrophic when a work
    dir has just inherited 53 GB of correct bytes. If `RECIPE` leaked into the
    inherited digests, or if adding `oc025` to `ctx.shapes()` leaked into one
    of their `shapes` blocks, every seeded build would rebuild from 1982 and
    report it as a routine staleness warning.

    The expected body is RESTATED here from the f7l0 builder's own formula
    rather than read back from the implementation, so the two have to agree.
    """
    import argparse
    ctx = b7.Ctx(argparse.Namespace(
        work=build["work"], source_dir=build["src"], start=build["start"],
        end=build["end"], force=False, stage="all", smoke=True,
        oc_start=build["oc_start"]))
    live = np.load(os.path.join(build["work"], "rg", "live.npz"))
    n_live = len(live["bin_index"])

    def f7l0_digest(stage):
        owns = {"ncep": "g100", "rg": "rg100"}          # f7l0's STAGE_OWNS
        shapes = {"g025": (ctx.T, 721, 1440, 7),
                  "g100": (ctx.T, 181, 360, 15),
                  "rg100": (n_live, 181, 360, 32)}      # f7l0's ctx.shapes()
        body = {
            "stage": stage, "version": b7.SPEC_VERSION[stage],
            "recipe": "f7l0",
            "channels": b7.STAGE_CHANNELS.get(stage, []),
            "min_days": b7.MIN_DAYS, "pentad_days": b7.PENTAD_DAYS,
            "epoch": str(ac.EPOCH), "bins": [ctx.b_lo, ctx.b_hi],
            "shapes": {k: list(v) for k, v in shapes.items()
                       if k in (owns.get(stage),
                                "g025" if stage in ("glorys", "sst") else "")},
        }
        if stage == "ncep":
            body["files"] = b7.NCEP_FILES
            body["flip"] = list(b7.NCEP_FLIP)
            body["sigma"] = list(b7.NCEP_SIGMA)
        if stage == "rg":
            body["levels"] = list(b7.LEVELS)
            body["band"] = [b7.RG_LAT_LO, b7.RG_LAT_HI]
        import hashlib
        return hashlib.sha256(
            json.dumps(body, sort_keys=True).encode()).hexdigest()

    for stage in b7.INHERITED_STAGES:
        got, body = b7.stage_spec(ctx, stage, n_live)
        assert body["recipe"] == "f7l0", stage
        assert "oc025" not in body["shapes"], \
            f"{stage}'s digest now depends on the colour group's shape"
        assert got == f7l0_digest(stage), \
            f"{stage}'s spec digest moved — a seeded build would rebuild it"
        # ...and the .spec file inherited from the seed says f7l0 too, and
        # carries no colour shape. (`rg`'s recorded shapes block can be EMPTY:
        # a fresh build writes rg's spec before `rg/live.npz` exists, so
        # `n_live` is not yet knowable. `stage_state_check` upgrades that in
        # place rather than calling it staleness — asserted below.)
        rec = b7.read_json(os.path.join(build["work"], f"{stage}.spec"), {})
        assert rec["spec"]["recipe"] == "f7l0", stage
        assert "oc025" not in rec["spec"]["shapes"], stage
        assert rec.get("sha256") in (got, None) or rec["spec"]["shapes"] == {}, \
            stage

    # the four stages that DID change answer f7l1, or nothing would ever rebuild
    for stage in ("occci", "norm", "meta", "publish"):
        _, body = b7.stage_spec(ctx, stage, n_live)
        assert body["recipe"] == "f7l1", stage
    assert b7.stage_recipe("sst") == "f7l0"
    assert b7.stage_recipe("occci") == "f7l1"

    # and a seeded work dir is NOT declared stale: no marker was discarded
    for stage in b7.INHERITED_STAGES:
        stamp = open(b7.marker(build["work"], stage)).read()
        assert b7.stage_state_check(ctx, stage, n_live) is False, stage
        assert open(b7.marker(build["work"], stage)).read() == stamp, stage


# ----------------------------------------------------------------- 22 -----
def test_22_occci_indexes_the_listing_and_never_invents_a_day(tmp_path):
    """The stage LISTS, parses dates out of the names, and counts the holes.

    Root CLAUDE.md: never guess what an archive serves, ask it. A day that is
    genuinely absent from the listing is a COUNT (`n_occci_absent`); a day the
    listing has and the stage cannot read is a build that stops, naming the
    host and the file. The two must not be confused — a silent skip writes
    "cloudy" where "we gave up" belongs.
    """
    import argparse
    src = str(tmp_path / "src")
    work = str(tmp_path / "work")
    d_lo, d_hi = dt.date(2010, 1, 14), dt.date(2010, 1, 23)
    days = [d_lo + dt.timedelta(days=k) for k in range((d_hi - d_lo).days + 1)]
    keep = [d for d in days if d != dt.date(2010, 1, 16)]    # one real hole
    b7.make_smoke_oc_sources(src, keep, d_lo)

    ctx = b7.Ctx(argparse.Namespace(
        work=work, source_dir=src, start=str(d_lo), end=str(d_hi),
        force=False, stage="occci", smoke=True, oc_start=str(d_lo)))
    b7.stage_occci(ctx)

    idx = b7.read_json(os.path.join(work, "occci", "index.json"))
    assert set(idx["2010"]["files"]) == {str(d) for d in keep}
    assert b7.read_json(os.path.join(work, "counts.json"))["n_occci_days"] == \
        len(keep)
    assert b7.read_json(os.path.join(work, "occci", "absent.json"))["2010"] == 1
    assert b7.marked(work, "occci") and b7.marked(work, "occci/2010")

    # the file-name date parse is the archive's, not a template we assumed
    assert b7.oc_date_of_name(
        "ESACCI-OC-L3S-CHLOR_A-MERGED-1D_DAILY_4km_GEO_PML_OCx-"
        "20150701-fv6.0.nc") == dt.date(2015, 7, 1)
    assert b7.oc_date_of_name("no-date-here.nc") is None
    # ...and a THREDDS catalogue is parsed for the server's own urlPath
    cat = b'''<?xml version="1.0"?>
    <catalog xmlns="http://www.unidata.ucar.edu/namespaces/thredds/InvCatalog/v1.0">
      <dataset name="x-20150701-fv6.0.nc" urlPath="cci/a/x-20150701-fv6.0.nc"/>
      <dataset name="notes.txt" urlPath="cci/a/notes.txt"/>
    </catalog>'''
    got = b7.parse_thredds_catalog(cat, "https://h/thredds/fileServer/")
    assert got == {"x-20150701-fv6.0.nc":
                   "https://h/thredds/fileServer/cci/a/x-20150701-fv6.0.nc"}
    assert list(b7.parse_json_listing(
        b'[{"name": "a-20150701-fv6.0.nc"}, {"name": "b.txt"}]')) == \
        ["a-20150701-fv6.0.nc"]
    assert list(b7.parse_html_listing(
        b'<a href="a-20150701-fv6.0.nc">a</a><a href="b.txt">b</a>')) == \
        ["a-20150701-fv6.0.nc"]

    # a LISTED file that cannot be read stops the build, naming it
    bad = os.path.join(src, "occci", sorted(
        os.listdir(os.path.join(src, "occci")))[0])
    with open(bad, "wb") as fh:
        fh.write(b"not a netcdf")
    work2 = str(tmp_path / "work2")
    ctx2 = b7.Ctx(argparse.Namespace(
        work=work2, source_dir=src, start=str(d_lo), end=str(d_hi),
        force=False, stage="occci", smoke=True, oc_start=str(d_lo)))
    with pytest.raises(SystemExit) as e:
        b7.stage_occci(ctx2)
    assert "refuses to record a listed day as cloud" in str(e.value)


# ----------------------------------------------------------------- 23 -----
def test_23_occci_resumes_across_a_year_boundary(tmp_path):
    """A killed colour stage resumes with the PARTIAL pentad it had.

    The statement test 11 makes for GLORYS, for the stage that streams ten
    thousand files. A pentad straddles 31 December, so the accumulator for the
    bin holding 2010-12-30 is only completed by 2011's first files; `Carry`
    writes it BEFORE the year's marker (ml/CLAUDE.md §5.21, a marker may only
    under-claim), and the only assertion that catches a lost carry is that the
    resumed build is BIT-IDENTICAL to the one-pass build.

    The crash is staged INSIDE 2011, after 2010 was committed and marked, which
    is the case the keyed carry exists for: the resume must reload 2010's
    partial accumulator and replay 2011 from its first file.
    """
    import argparse
    src = str(tmp_path / "src")
    d_lo, d_hi = dt.date(2010, 12, 27), dt.date(2011, 1, 6)
    days = [d_lo + dt.timedelta(days=k) for k in range((d_hi - d_lo).days + 1)]
    b7.make_smoke_oc_sources(src, days, d_lo)
    assert len({d.year for d in days}) == 2, "the fixture must span a year end"

    def ctx_for(work):
        return b7.Ctx(argparse.Namespace(
            work=work, source_dir=src, start=str(d_lo), end=str(d_hi),
            force=False, stage="occci", smoke=True, oc_start=str(d_lo)))

    one = str(tmp_path / "one")
    b7.stage_occci(ctx_for(one))
    whole = np.asarray(np.load(b7.raw_file(one, "oc025"), mmap_mode="r"))
    assert np.isfinite(whole).any()

    # ---- the same build, killed on 2011-01-03 -----------------------------
    two = str(tmp_path / "two")
    real_open = b7.oc_open

    def killed(path):
        if "20110103" in os.path.basename(path):
            raise SystemExit("box destroyed part way through 2011")
        return real_open(path)

    b7.oc_open = killed
    try:
        with pytest.raises(SystemExit):
            b7.stage_occci(ctx_for(two))
    finally:
        b7.oc_open = real_open

    assert b7.marked(two, "occci/2010"), "2010 finished and was not marked"
    assert not b7.marked(two, "occci/2011")
    assert os.path.exists(os.path.join(two, "occci", "carry_2010.npz")), \
        "the year-boundary accumulator was not carried"
    carried = np.load(os.path.join(two, "occci", "carry_2010.npz"))
    assert any(k.startswith("acc_") for k in carried.files), \
        "the carry holds no open bin — the straddling pentad was lost"
    part = np.asarray(np.load(b7.raw_file(two, "oc025"), mmap_mode="r"))
    assert not np.array_equal(part, whole, equal_nan=True), \
        "the killed build already equals the whole one — nothing was pending"

    # ---- resume: 2010 is skipped, 2011 replays from its first file --------
    b7.stage_occci(ctx_for(two))
    resumed = np.asarray(np.load(b7.raw_file(two, "oc025"), mmap_mode="r"))
    assert np.array_equal(whole, resumed, equal_nan=True), \
        "the resumed colour build differs from the one-pass build"
    assert b7.read_json(os.path.join(two, "counts.json"))["n_occci_days"] == \
        b7.read_json(os.path.join(one, "counts.json"))["n_occci_days"]


# ----------------------------------------------------------------- 24 -----
def test_24_an_inherited_group_cannot_be_written_through_the_hard_link(
        build, tmp_path):
    """The seed's bytes survive everything this build does. E-077 §5.

    A hard link is ONE INODE UNDER TWO NAMES. `--seed-from` buys 53 GB for
    free that way, and it is also the single mechanism by which this build
    could destroy the published f7l0 tensor: one write through the f7l1 name
    lands in the f7l0 file, whose sha256 the Hub, the handover and another
    agent's hand-off all quote, and nothing would say so until somebody
    re-verified a hash. So it is made IMPOSSIBLE rather than merely
    unintended, and this asserts all three layers of that.
    """
    import argparse
    work, seed = build["work"], build["seed"]

    # (a) the build KNOWS which groups are shared, and refuses by name
    assert b7.seeded_groups(work) == set(b7.BASE_GROUPS)
    assert b7.seeded_groups(str(tmp_path)) == set(), \
        "a work dir with no seed must report no shared groups"
    for g in b7.BASE_GROUPS:
        with pytest.raises(SystemExit) as e:
            b7.refuse_if_seeded(work, g, "fill")
        assert "HARD LINK" in str(e.value) and seed in str(e.value)
        # the two doors every writable open of a group goes through
        with pytest.raises(SystemExit):
            b7.open_fill(work, g, (1, 1, 1, 1), create=True)
        with pytest.raises(SystemExit):
            b7.open_final(work, g, (1, 1, 1, 1), create=True)
    # ...and the colour group, which this build OWNS, is not refused
    b7.refuse_if_seeded(work, "oc025", "fill")

    # (b) the inode is read-only, under BOTH names
    for g in b7.BASE_GROUPS:
        new = b7.group_file(work, g)
        old = os.path.join(seed, f"{b7.BASE_STEM}_X_{g}.npy")
        assert os.path.samefile(old, new)
        for p in (old, new):
            assert os.stat(p).st_mode & 0o222 == 0, \
                f"{p} is still writable — chmod did not reach the inode"
    if os.geteuid() != 0:
        # ROOT IGNORES THE MODE BITS (measured), and the builder runs as root
        # on the Vast boxes — which is exactly why (a) exists and is the load
        # bearing layer. Where the mode bits DO bind, assert that they bind.
        with pytest.raises(PermissionError):
            np.lib.format.open_memmap(b7.group_file(work, "g025"), mode="r+")

    # (c) the stages that run on a seeded dir leave the SEED byte-identical
    before = {g: b7.sha256(os.path.join(seed, f"{b7.BASE_STEM}_X_{g}.npy"))
              for g in b7.BASE_GROUPS}
    stat0 = {g: os.stat(os.path.join(seed, f"{b7.BASE_STEM}_X_{g}.npy")).st_mtime
             for g in b7.BASE_GROUPS}
    ctx = b7.Ctx(argparse.Namespace(
        work=work, source_dir=build["src"], start=build["start"],
        end=build["end"], force=False, stage="all", smoke=True,
        oc_start=build["oc_start"]))
    b7.run_stages(ctx, ["norm"])
    b7.stage_meta(ctx)
    # the publish's own local half: hash every file, and compare the three
    # inherited ones against `f7l0`'s manifest exactly as `stage_publish`
    # does once the Hub has answered. Nothing here may touch the bytes.
    base_manifest = {"files": [
        {"name": f"{b7.BASE_STEM}_X_{g}.npy", "sha256": before[g]}
        for g in b7.BASE_GROUPS]}
    base = {}
    for rec in base_manifest["files"]:
        for g in b7.BASE_GROUPS:
            if rec["name"] == f"{b7.BASE_STEM}_X_{g}.npy":
                base[g] = rec["sha256"]
    for g in b7.BASE_GROUPS:
        assert b7.sha256(b7.group_file(work, g)) == base[g], \
            f"{g} no longer matches {b7.BASE_RECIPE}'s manifest — the " \
            f"publish would (correctly) refuse"
    after = {g: b7.sha256(os.path.join(seed, f"{b7.BASE_STEM}_X_{g}.npy"))
             for g in b7.BASE_GROUPS}
    assert after == before, "a stage wrote through the shared inode"
    for g in b7.BASE_GROUPS:
        assert os.stat(os.path.join(
            seed, f"{b7.BASE_STEM}_X_{g}.npy")).st_mtime == stat0[g]

    # (d) A STALE INHERITED SPEC REFUSES; it does not discard and replay.
    # Discarding markers is harmless — what follows is not: the stage would
    # re-run and write its group, and that group's bytes are f7l0's.
    w2 = str(tmp_path / "seeded")
    ctx2 = b7.Ctx(argparse.Namespace(
        work=w2, source_dir=build["src"], start=build["start"],
        end=build["end"], force=False, stage="all", smoke=True,
        oc_start=build["oc_start"]))
    b7.seed_from(ctx2, seed)
    n_live = len(np.load(os.path.join(w2, "rg", "live.npz"))["bin_index"])
    for stage in ("sst", "ncep"):                  # shares g025 · owns g100
        b7.atomic_json(os.path.join(w2, f"{stage}.spec"),
                       {"sha256": "0" * 64, "at": b7.utcnow(),
                        "spec": {"stage": stage, "recipe": "something-else"}})
        stamp = open(b7.marker(w2, stage)).read()
        sha = b7.sha256(b7.group_file(w2, "g025" if stage == "sst" else "g100"))
        with pytest.raises(SystemExit) as e:
            b7.stage_state_check(ctx2, stage, n_live)
        assert "INHERITED as a hard link" in str(e.value), stage
        assert "Nothing has been deleted" in str(e.value), stage
        assert b7.marked(w2, stage), f"{stage}.done was deleted anyway"
        assert open(b7.marker(w2, stage)).read() == stamp, stage
        assert b7.sha256(
            b7.group_file(w2, "g025" if stage == "sst" else "g100")) == sha
    assert {g: b7.sha256(os.path.join(seed, f"{b7.BASE_STEM}_X_{g}.npy"))
            for g in b7.BASE_GROUPS} == before


def _publish_harness(tmp_path, monkeypatch, seeded, drift):
    """Drive `stage_publish` against a FAKE Hub: no network, tiny files.

    The fake answers f7l0's manifest with the hash of our own g025 file when
    `drift` is False and with a different hash when it is True; upload is a
    no-op and "download back" copies the local file, so the restore check
    passes and only the inheritance decision is exercised.
    """
    import types
    work = str(tmp_path / ("seeded" if seeded else "unseeded"))
    os.makedirs(os.path.join(work, "src"), exist_ok=True)
    files = [os.path.join(work, b7.STEM + ".npz")] + \
            [b7.group_file(work, g) for g in b7.GROUPS]
    for i, p in enumerate(files):
        with open(p, "wb") as fh:
            fh.write(bytes([i]) * 64)
    ours = {g: b7.sha256(b7.group_file(work, g)) for g in b7.BASE_GROUPS}
    base = dict(ours)
    if drift:
        base["g025"] = "f" * 64

    class FakeApi:
        def create_repo(self, *a, **k): pass
        def upload_file(self, *a, **k): pass

    def fake_download(repo, path, repo_type=None, token=None, local_dir=None):
        src = os.path.join(work, os.path.basename(path))
        os.makedirs(local_dir, exist_ok=True)
        dst = os.path.join(local_dir, os.path.basename(path))
        shutil.copy2(src, dst)
        return dst

    fake_hf = types.ModuleType("huggingface_hub")
    fake_hf.hf_hub_download = fake_download
    fake_hf.HfApi = lambda *a, **k: FakeApi()
    monkeypatch.setitem(sys.modules, "huggingface_hub", fake_hf)
    monkeypatch.setattr(b7, "hub_repo", lambda token=None:
                        (FakeApi(), "chfrank/earth-tensors", "x"))
    monkeypatch.setattr(b7, "base_manifest_hashes", lambda api, repo, tok: base)
    ctx = types.SimpleNamespace(
        work=work, scratch=os.path.join(work, "src"), prog=b7.Progress(work),
        sources={}, b_oc=1145,
        a=types.SimpleNamespace(seed_from="/some/f7l0" if seeded else ""))
    return ctx, work, base


def test_25_publish_records_drift_for_unseeded_and_refuses_for_seeded(
        tmp_path, monkeypatch):
    """The `same_as_f7l0` promise is only made by a SEEDED build.

    2026-09-13: the box holding the f7l0 seed would not start, so 7.1 was
    rebuilt unseeded on a fresh box. Such a build never claimed its three base
    groups were f7l0's bytes; a hash drift against f7l0's manifest is a
    finding the manifest records (both hashes, per file), not a broken
    inheritance that kills the publish eight hours in. A seeded build keeps
    the fatal check — there the bytes ARE f7l0's or something is very wrong.
    """
    # (a) unseeded + drift: publishes, records same_as_f7l0=false for g025
    ctx, work, base = _publish_harness(tmp_path, monkeypatch, False, True)
    b7.stage_publish(ctx)
    man = json.load(open(os.path.join(work, "manifest.json")))
    assert man["seeded_from_base"] is False
    recs = {r["name"]: r for r in man["files"]}
    g025 = recs[f"{b7.STEM}_X_g025.npy"]
    assert g025["same_as_f7l0"] is False
    assert g025["f7l0_sha256"] == base["g025"]
    assert g025["base_name"] == f"{b7.BASE_STEM}_X_g025.npy"
    for g in ("g100", "rg100"):
        assert recs[f"{b7.STEM}_X_{g}.npy"]["same_as_f7l0"] is True
    assert "same_as_f7l0" not in recs[f"{b7.STEM}_X_oc025.npy"]
    assert b7.marked(work, "publish")
    # (b) unseeded, no drift: identical bytes are still reported as such
    ctx, work, _ = _publish_harness(tmp_path / "b", monkeypatch, False, False)
    b7.stage_publish(ctx)
    man = json.load(open(os.path.join(work, "manifest.json")))
    assert all(r["same_as_f7l0"] for r in man["files"]
               if r["name"] != f"{b7.STEM}_X_oc025.npy"
               and not r["name"].endswith(".npz"))
    # (c) seeded + drift: the fatal check is unchanged
    ctx, work, _ = _publish_harness(tmp_path / "c", monkeypatch, True, True)
    with pytest.raises(SystemExit) as e:
        b7.stage_publish(ctx)
    assert "INHERITANCE BROKEN on g025" in str(e.value)
    assert not os.path.exists(os.path.join(work, "manifest.json"))


# ======================================================================
# The PSL transfer guard and the Hub mirror (2026-09-13).
#
# MEASURED that day: the family-7.1 build box in the UK reads
# downloads.psl.noaa.gov at 0.17 MB/s — one 477 MB OISST year in 47 minutes
# against 33 s from another box — so the sst and ncep stages could not finish
# inside the workflow's 24 h timeout. `urlopen(timeout=)` is a PER-READ
# timeout, so nothing ever raised and the `mirrors=` cycling never engaged.
# Two changes answer it, and these are their tests: a THROUGHPUT guard in
# `build_family3.fetch`, and a HUB-FIRST path for PSL URLs in
# `build_family7.download_verified` fed by `ml/mirror_psl.py`.
#
# NO NETWORK. The guard is tested against a local http.server that trickles;
# the Hub is a fake module in sys.modules, because huggingface_hub is imported
# lazily inside the functions for exactly this reason.
# ======================================================================
import hashlib                                                 # noqa: E402
import http.server                                             # noqa: E402
import threading                                               # noqa: E402
import time                                                    # noqa: E402
import types                                                   # noqa: E402


class _RateHandler(http.server.BaseHTTPRequestHandler):
    """`/fast` answers at once; `/slow` writes 128 KiB every 50 ms."""

    BODY = b"x" * (2 << 20)

    def log_message(self, *a):                                 # keep stdout clean
        pass

    def do_GET(self):
        self.send_response(200)
        self.send_header("Content-Length", str(len(self.BODY)))
        self.end_headers()
        try:
            if self.path.startswith("/fast"):
                self.wfile.write(self.BODY)
                return
            step = 128 << 10
            for i in range(0, len(self.BODY), step):
                self.wfile.write(self.BODY[i:i + step])
                self.wfile.flush()
                time.sleep(0.05)
        except (BrokenPipeError, ConnectionResetError):
            pass                     # the client aborted: that is the point


def _serve():
    """(base_url, shutdown) for a threaded local server on an ephemeral port."""
    srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), _RateHandler)
    srv.daemon_threads = True
    t = threading.Thread(target=srv.serve_forever, daemon=True)
    t.start()

    def stop():
        srv.shutdown()
        srv.server_close()
    return f"http://127.0.0.1:{srv.server_address[1]}", stop


def test_25_the_throughput_guard_aborts_a_trickle_and_moves_to_the_mirror(tmp_path):
    """A slow HOST is abandoned, not waited out — and the next URL is tried.

    The old `shutil.copyfileobj` had no clock: a 0.17 MB/s host simply took as
    long as it took. The guard measures the transfer and raises SlowTransfer,
    which the attempt loop treats like any other failure — so attempt 2 goes
    to the mirror and the file still lands, whole.
    """
    assert f3.FETCH_SLOW_BACKOFF_S == 5, (
        "the backoff after a SlowTransfer must stay SHORT — waiting does not "
        "make a slow host fast; the exponential ladder is for a host that is "
        "refusing, not for one that is crawling")
    base, stop = _serve()
    back = f3.FETCH_SLOW_BACKOFF_S
    f3.FETCH_SLOW_BACKOFF_S = 0                # the test does not need the wait
    try:
        # (a) the guard fires, and it says so with the numbers.
        p = str(tmp_path / "one" / "trickle.bin")
        t0 = time.time()
        with pytest.raises(f3.SlowTransfer) as e:
            f3.fetch(f"{base}/slow", p, attempts=1,
                     min_probe_s=0.2, min_rate_mbps=50.0)
        el = time.time() - t0
        assert e.value.url == f"{base}/slow"
        assert 0 < e.value.bytes < len(_RateHandler.BODY), (
            "the transfer must be ABORTED part-way; a guard that only fires "
            "after the whole file has arrived has saved nothing")
        assert e.value.seconds >= 0.2 and e.value.mbps < 50.0
        assert el < 20, "the guard did not abort: it waited for the whole body"
        assert not os.path.exists(p) and not os.path.exists(p + ".part"), (
            "the partial .part must be deleted — `fetch` returns early when "
            "`path` exists, so a leftover would be read as a finished file")

        # (b) with a mirror, the SECOND attempt takes it and the file lands.
        p2 = str(tmp_path / "two" / "trickle.bin")
        f3.fetch(f"{base}/slow", p2, attempts=2, mirrors=(f"{base}/fast",),
                 min_probe_s=0.2, min_rate_mbps=50.0)
        assert open(p2, "rb").read() == _RateHandler.BODY
    finally:
        f3.FETCH_SLOW_BACKOFF_S = back
        stop()


def test_26_a_normal_transfer_is_unchanged_and_the_env_can_move_the_floor(tmp_path):
    """The guard must be invisible to every fetch that is not pathological."""
    base, stop = _serve()
    try:
        p = str(tmp_path / "fast.bin")
        f3.fetch(f"{base}/fast", p)             # stock thresholds, no kwargs
        assert open(p, "rb").read() == _RateHandler.BODY
        assert not os.path.exists(p + ".part")

        # A second fetch to the same path is the pre-existing short circuit.
        f3.fetch(f"{base}/nonexistent-would-404", p)

        # The environment overrides the defaults, so a box on a genuinely slow
        # link with no mirror can wait rather than fail.
        keep = {k: os.environ.get(k) for k in
                ("EARTH_FETCH_MIN_RATE_MBPS", "EARTH_FETCH_PROBE_S")}
        try:
            os.environ["EARTH_FETCH_MIN_RATE_MBPS"] = "0"
            os.environ["EARTH_FETCH_PROBE_S"] = "0.2"
            assert f3._guard_thresholds(1.0, 90.0) == (0.0, 0.2)
            p3 = str(tmp_path / "slow-but-allowed.bin")
            f3.fetch(f"{base}/slow", p3, attempts=1,
                     min_probe_s=0.2, min_rate_mbps=50.0)   # env wins: rate 0
            assert open(p3, "rb").read() == _RateHandler.BODY
        finally:
            for k, v in keep.items():
                if v is None:
                    os.environ.pop(k, None)
                else:
                    os.environ[k] = v
        assert f3._guard_thresholds(1.0, 90.0) == (1.0, 90.0)
    finally:
        stop()


def test_27_psl_mirror_path_maps_the_url_and_nothing_else():
    """The mirror layout IS PSL's own URL path — no table to keep in step."""
    assert b7.HUB_MIRROR_PREFIX == "mirrors/psl"
    assert b7.psl_mirror_path(
        f"{b7.PSL_OISST}/sst.day.mean.1983.nc") == \
        "mirrors/psl/Datasets/noaa.oisst.v2.highres/sst.day.mean.1983.nc"
    assert b7.psl_mirror_path(
        f"{b7.PSL_NCEP}/{b7.NCEP_LAND}.nc") == \
        "mirrors/psl/Datasets/ncep.reanalysis/surface_gauss/land.sfc.gauss.nc"
    for k, stem in b7.NCEP_FILES.items():
        rel = b7.psl_mirror_path(f"{b7.PSL_NCEP}/{stem}.2024.nc")
        assert rel.startswith("mirrors/psl/Datasets/ncep.reanalysis/")
        assert rel.endswith(f"/{stem}.2024.nc"), k
    # Everything else is None — the thredds mirror above all, which serves
    # TRUNCATED files under load (measured 2026-09-04) and must never be
    # confused with the Hub copy.
    for u in (f"{b7.THREDDS_OISST}/sst.day.mean.1983.nc",
              f"{b7.THREDDS_NCEP}/skt.sfc.gauss.2020.nc",
              f"{b7.RG_BASE}/RG_ArgoClim_Temperature_2019.nc.gz",
              b7.ETOPO_URL, "https://downloads.psl.noaa.gov/Datasets/",
              "", None, 42):
        assert b7.psl_mirror_path(u) is None, u


def _fake_hub(paths_info, download):
    """A stand-in `huggingface_hub` module.

    huggingface_hub is NOT installed in the sandbox, which is why the builder
    imports it lazily inside the functions that use it; that same laziness is
    what lets this test replace it wholesale.
    """
    m = types.ModuleType("huggingface_hub")
    calls = {"paths_info": [], "download": []}

    class _Info:
        def __init__(self, path, size):
            self.path, self.size = path, size

    class HfApi:
        def __init__(self, token=None):
            self.token = token

        def whoami(self):
            raise AssertionError("whoami must not be needed to READ a public "
                                 "mirror — a build without a token still has "
                                 "to be able to use it")

        def get_paths_info(self, repo, paths, repo_type=None):
            calls["paths_info"].append((repo, tuple(paths)))
            return [_Info(p, n) for p, n in paths_info(paths)]

    def hf_hub_download(repo, path_in_repo, **kw):
        calls["download"].append((repo, path_in_repo, kw))
        return download(repo, path_in_repo, kw)

    class EntryNotFoundError(Exception):
        pass

    m.HfApi = HfApi
    m.hf_hub_download = hf_hub_download
    m.EntryNotFoundError = EntryNotFoundError
    m.utils = types.SimpleNamespace(EntryNotFoundError=EntryNotFoundError)
    return m, calls


class _Ctx:
    """Just enough of Ctx for note_source."""

    def __init__(self):
        self.sources = {}

    def note_source(self, key, value):
        self.sources[key] = value


def test_28_download_verified_reads_the_hub_mirror_first(tmp_path):
    """Hub first for PSL; PSL untouched when the mirror has it.

    And the three ways back to PSL, each asserted by EFFECT (was the origin
    fetcher called?) rather than by a log line: the mirror does not hold the
    file, the pull raises, and EARTH_NO_HUB_MIRROR=1.
    """
    url = f"{b7.PSL_OISST}/sst.day.mean.1983.nc"
    rel = b7.psl_mirror_path(url)
    hub_bytes, psl_bytes = b"HUB COPY", b"PSL COPY"

    def run(paths_info, download, env=None, ctx=None):
        """One download_verified with the Hub faked out. -> (path, psl_calls)."""
        n = {"psl": 0}

        def fake_fetch(u, p, **kw):
            n["psl"] += 1
            os.makedirs(os.path.dirname(p), exist_ok=True)
            with open(p, "wb") as fh:
                fh.write(psl_bytes)
            return p

        mod, calls = _fake_hub(paths_info, download)
        keep_fetch, keep_size = f3.fetch, b7.remote_size
        keep_mod = sys.modules.get("huggingface_hub")
        keep_env = os.environ.get("EARTH_NO_HUB_MIRROR")
        keep_ns = os.environ.get("EARTH_HF_NAMESPACE")
        keep_repo = b7._MIRROR_REPO
        b7.HUB_MIRRORED.clear()
        try:
            sys.modules["huggingface_hub"] = mod
            f3.fetch = fake_fetch
            b7.remote_size = lambda u: None       # no HEAD to the real host
            os.environ["EARTH_HF_NAMESPACE"] = "someone"
            b7._MIRROR_REPO = None
            if env is None:
                os.environ.pop("EARTH_NO_HUB_MIRROR", None)
            else:
                os.environ["EARTH_NO_HUB_MIRROR"] = env
            d = tempfile.mkdtemp(prefix="dlv_", dir=str(tmp_path))
            p = b7.download_verified(url, os.path.join(d, "sst.nc"))
            if ctx is not None:
                b7.hub_mirror_note(ctx, "oisst_mirror", "noaa.oisst.v2.highres")
            return open(p, "rb").read(), n["psl"], calls
        finally:
            f3.fetch, b7.remote_size = keep_fetch, keep_size
            b7._MIRROR_REPO = keep_repo
            if keep_mod is None:
                sys.modules.pop("huggingface_hub", None)
            else:
                sys.modules["huggingface_hub"] = keep_mod
            for k, v in (("EARTH_NO_HUB_MIRROR", keep_env),
                         ("EARTH_HF_NAMESPACE", keep_ns)):
                if v is None:
                    os.environ.pop(k, None)
                else:
                    os.environ[k] = v

    def write_hub(repo, path_in_repo, kw):
        d = os.path.join(kw["local_dir"], os.path.dirname(path_in_repo))
        os.makedirs(d, exist_ok=True)
        p = os.path.join(d, os.path.basename(path_in_repo))
        with open(p, "wb") as fh:
            fh.write(hub_bytes)
        return p

    # (a) MIRROR PRESENT -> the Hub answers and PSL is never called.
    ctx = _Ctx()
    got, psl, calls = run(lambda ps: [(p, len(hub_bytes)) for p in ps],
                          write_hub, ctx=ctx)
    assert got == hub_bytes and psl == 0
    assert calls["download"] and calls["download"][0][1] == rel
    assert calls["download"][0][0] == f"someone/{b7.HF_DATASET}"
    assert rel in b7.HUB_MIRRORED
    assert ctx.sources["oisst_mirror"].startswith(
        f"hf://someone/{b7.HF_DATASET}/mirrors/psl/Datasets/noaa.oisst.v2.highres/")

    # (b) The Hub's OWN metadata is what the bytes are checked against — a
    # short file from the mirror falls through rather than being trusted.
    got, psl, calls = run(lambda ps: [(p, 999999) for p in ps], write_hub)
    assert got == psl_bytes and psl == 1

    # (c) MIRROR ABSENT (nothing under that path) -> PSL, no download attempt.
    got, psl, calls = run(lambda ps: [], write_hub)
    assert got == psl_bytes and psl == 1 and calls["download"] == []

    # (d) MIRROR PULL RAISES (EntryNotFoundError, the 404 the Hub throws)
    #     -> PSL. A mirror failure must never mask a PSL success.
    def boom(repo, path_in_repo, kw):
        raise sys.modules["huggingface_hub"].EntryNotFoundError(path_in_repo)

    got, psl, calls = run(lambda ps: [(p, len(hub_bytes)) for p in ps], boom)
    assert got == psl_bytes and psl == 1 and len(calls["download"]) == 1

    # (e) EARTH_NO_HUB_MIRROR=1 -> the Hub is not consulted at all.
    ctx = _Ctx()
    got, psl, calls = run(lambda ps: [(p, len(hub_bytes)) for p in ps],
                          write_hub, env="1", ctx=ctx)
    assert got == psl_bytes and psl == 1
    assert calls["download"] == [] and calls["paths_info"] == []
    assert ctx.sources == {}, ("nothing came off the mirror, so nothing may "
                              "claim it did")
    b7.HUB_MIRRORED.clear()


def test_29_mirror_psl_deletes_a_hub_file_whose_round_trip_fails(tmp_path):
    """A mirrored file that does not restore is worse than an absent one.

    `ml/hf_mirror.py`'s rule: a backup is only real if the restore works. The
    builder would READ a truncated mirror copy, and the symptom would be
    `NetCDF: HDF error` hours later, one whole year of the axis gone — so the
    round trip is checked and a mismatch is DELETED from the Hub, not left
    there with a warning.
    """
    sys.path.insert(0, ML)
    import mirror_psl as mp                                    # noqa: E402

    url = f"{b7.PSL_OISST}/sst.day.mean.1983.nc"
    rel = b7.psl_mirror_path(url)
    good = b"the real bytes"

    class FakeApi:
        def __init__(self):
            self.uploaded, self.deleted = [], []

        def upload_file(self, path_or_fileobj=None, path_in_repo=None,
                        repo_id=None, repo_type=None, commit_message=None):
            assert repo_type == "dataset" and repo_id == "ns/earth-tensors"
            assert os.path.basename(rel) in commit_message, (
                "the commit message must NAME the file — a wall of "
                '"PSL mirror" commits says nothing about what landed')
            self.uploaded.append((path_in_repo, open(path_or_fileobj, "rb").read()))

        def delete_file(self, path_in_repo=None, repo_id=None, repo_type=None,
                        commit_message=None):
            self.deleted.append(path_in_repo)

    def run(restored):
        api = FakeApi()
        keep = (mp.fetch, mp.remote_size, mp.hf_download)
        work = tempfile.mkdtemp(prefix="mp_", dir=str(tmp_path))
        try:
            def fake_fetch(u, p, **kw):
                with open(p, "wb") as fh:
                    fh.write(good)
                return p

            def fake_back(repo, path_in_repo, dest, token=None):
                os.makedirs(dest, exist_ok=True)
                p = os.path.join(dest, os.path.basename(path_in_repo))
                with open(p, "wb") as fh:
                    fh.write(restored)
                return p

            mp.fetch, mp.remote_size = fake_fetch, lambda u: len(good)
            mp.hf_download = fake_back
            err = None
            rec = None
            try:
                rec = mp.mirror_one(api, "ns/earth-tensors", url, rel, work)
            except mp.MirrorError as e:
                err = str(e)
            left = sorted(os.listdir(work))
            return api, rec, err, left
        finally:
            mp.fetch, mp.remote_size, mp.hf_download = keep
            shutil.rmtree(work, ignore_errors=True)

    # The round trip does not verify -> the Hub copy is DELETED and it fails.
    api, rec, err, left = run(b"truncat")
    assert rec is None and api.uploaded and api.deleted == [rel]
    assert "RESTORE MISMATCH" in err and "deleted from the Hub" in err
    assert left == [], ("one file at a time: a hosted runner has ~14 GB and "
                        "nothing may be left behind, least of all on a failure")

    # The round trip verifies -> a manifest record, nothing deleted, no disk.
    api, rec, err, left = run(good)
    assert err is None and api.deleted == []
    assert api.uploaded == [(rel, good)]
    assert rec["path"] == rel and rec["bytes"] == len(good)
    assert rec["sha256"] == hashlib.sha256(good).hexdigest()
    assert rec["source_url"] == url
    assert dt.datetime.fromisoformat(rec["mirrored_at"]).tzinfo is not None
    assert left == []

    # The file list is the family-7 source list, and it is derived from the
    # same constants the builder fetches from — not a second copy of them.
    files = mp.wanted("all", 1982, 2024)
    assert len(files) == 43 * 2 + 1 + 43 * len(b7.NCEP_FILES)
    assert all(b7.psl_mirror_path(u) == rel_ for u, rel_ in files)
    assert (f"{b7.PSL_NCEP}/{b7.NCEP_LAND}.nc",
            b7.psl_mirror_path(f"{b7.PSL_NCEP}/{b7.NCEP_LAND}.nc")) in files
    assert mp.wanted("oisst", 1982, 1982) == [
        (f"{b7.PSL_OISST}/sst.day.mean.1982.nc",
         b7.psl_mirror_path(f"{b7.PSL_OISST}/sst.day.mean.1982.nc")),
        (f"{b7.PSL_OISST}/icec.day.mean.1982.nc",
         b7.psl_mirror_path(f"{b7.PSL_OISST}/icec.day.mean.1982.nc"))]

#!/usr/bin/env python3
"""E-070: pin the GLOBAL arm of `ml/fetch_glorys_daily.py`.

The global pull stores every chunk ALREADY BINNED to 0.25 degrees, which is
what takes the Hugging Face dataset from ~800 GB to ~70 GB and what makes the
pentad and daily global cadences two reductions of one downloaded
byte-stream. That is only safe if the binning done at fetch time is the SAME
binning `ml/aggregate_cadence.py` does — otherwise the global tensor's
North-Atlantic sub-block would be a second opinion about what a 0.25 degree
cell is, and family 4 and family 7 would disagree by an amount no plot would
show.

So the check that matters here is an IDENTITY, not a tolerance
(ml/CLAUDE.md §4.9): a synthetic 1/12 degree month is binned by the fetcher
and, separately, by `aggregate_cadence.py` running as its own program at
`--cadence daily --bin-deg 0.25`, and every value and every NaN must match
bit for bit. The fixture carries a fixed land mask and a per-day flicker,
because a coast and an ice edge exercise different branches of the "mean of
the finite source cells, NaN where none" rule.

The rest pins the things a future edit could silently move: the window table
(the NA window is family 3's, verbatim), the size estimates, the file naming
(E-034's `glorys_YYYYMM.nc` contract must not move — 384 chunks are on the
Hub under it), the recorded grid attributes a downstream builder refuses on,
and the sub-block property: the NA window's 281x481 axes are an exact
contiguous slice of the global 681x1440 ones.

    python3 -m pytest tests/test_glorys_global.py -q
"""
import datetime as dt
import os
import subprocess
import sys

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "ml"))

import aggregate_cadence as ag                                   # noqa: E402
import fetch_glorys_daily as fg                                  # noqa: E402

AGG = os.path.join(ROOT, "ml", "aggregate_cadence.py")

# 1/12 degree, like GLORYS12, small enough to run in a second. Anchored so
# the target axis has interior cells (3 source cells each) as well as the
# half-width boundary cells.
NLAT, NLON = 19, 19
LAT = 30.0 + np.arange(NLAT) / 12.0
LON = -1.0 + np.arange(NLON) / 12.0
TIME_UNITS = "hours since 1950-01-01"
T0 = dt.date(1950, 1, 1)


def write_month(path, year, month, ndays, seed):
    """One synthetic daily chunk in the fetcher's shape and naming."""
    import netCDF4 as nc
    rng = np.random.default_rng(seed)
    d = nc.Dataset(path, "w")
    d.createDimension("time", ndays)
    d.createDimension("depth", 1)
    d.createDimension("latitude", NLAT)
    d.createDimension("longitude", NLON)
    t = d.createVariable("time", "f8", ("time",))
    t.units = TIME_UNITS
    d.createVariable("latitude", "f4", ("latitude",))[:] = LAT
    d.createVariable("longitude", "f4", ("longitude",))[:] = LON
    t[:] = [((dt.date(year, month, i + 1) - T0).days) * 24.0
            for i in range(ndays)]
    # A coastline (fixed in space) plus an ice edge (comes and goes per day).
    land = rng.random((NLAT, NLON)) < 0.25
    # ...and one solid island covering source cells 5..7 in both axes, which
    # is exactly the 3x3 block that feeds target cell (2, 2). Scattered land
    # never fills a whole 0.25 degree cell, so without this the "all sources
    # missing -> NaN, never 0" branch is not exercised at all.
    land[5:8, 5:8] = True
    for v in fg.VARIABLES:
        var = d.createVariable(v, "f4", ("time", "depth", "latitude",
                                         "longitude"), fill_value=np.nan)
        var.units = "m s-1"
        a = rng.normal(size=(ndays, 1, NLAT, NLON)).astype(np.float32)
        a[:, 0][:, land] = np.nan
        a[:, 0][rng.random((ndays, NLAT, NLON)) < 0.05] = np.nan
        var[:] = a
    d.close()


@pytest.fixture(scope="module")
def binned(tmp_path_factory):
    """A synthetic month, binned once by the fetcher's own code path."""
    d = tmp_path_factory.mktemp("e070")
    src = str(d / "glorys_199301.nc")
    dst = str(d / "glorys025_global_199301.nc")
    write_month(src, 1993, 1, 8, seed=7)
    plan = fg.bin_chunk(src, dst, 0.25, "point", fg.DATASET, "global")
    return dict(dir=str(d), src=src, dst=dst, plan=plan)


# --------------------------------------------------------------------------
# the window table
# --------------------------------------------------------------------------

def test_na_window_is_family_3s_verbatim():
    assert fg.WINDOWS["na"] is fg.WINDOW
    assert fg.WINDOW == dict(minimum_longitude=-100.0, maximum_longitude=20.0,
                             minimum_latitude=0.0, maximum_latitude=70.0)


def test_global_window_is_glorys12s_full_extent():
    # Confirmed from `copernicusmarine.describe(dataset_id=...)` on
    # 2026-09-03: latitude -80..90, longitude -180..179.9166717529297.
    assert fg.WINDOWS["global"] == dict(
        minimum_longitude=-180.0, maximum_longitude=180.0,
        minimum_latitude=-80.0, maximum_latitude=90.0)
    assert fg.SERVED["global"]["lat"] == (-80.0, 90.0)
    assert fg.SERVED["global"]["lon"][0] == -180.0
    assert abs(fg.SERVED["global"]["lon"][1] - (180.0 - 1 / 12)) < 1e-9


def test_cell_counts_are_the_two_grids():
    assert fg.CELLS["na"] == 1440 * 840 == 1_209_600
    assert fg.CELLS["global"] == 4320 * 2041 == 8_817_120


# --------------------------------------------------------------------------
# the estimates
# --------------------------------------------------------------------------

def test_wire_estimate_scales_with_the_pixel_ratio():
    """The measured global chunk must be the measured NA one times the pixels.

    Not a modelling exercise: if a future subset silently stopped being
    global (a wrapped longitude, a clipped pole) the byte count is the first
    place it would show, so the two measurements are held against each other.
    """
    ratio = fg.CELLS["global"] / fg.CELLS["na"]
    assert abs(ratio - 7.289) < 0.01
    predicted = fg.MEAS_GB["na"] * ratio
    assert abs(fg.MEAS_GB["global"] - predicted) / predicted < 0.15, (
        f"measured global chunk {fg.MEAS_GB['global']} GB vs "
        f"{predicted:.3f} GB predicted from the NA measurement")


def test_binned_estimate_is_smaller_than_the_wire_estimate():
    # The whole argument for binning at fetch time: ~10x less on the Hub.
    assert fg.MEAS_BINNED_GB["global"] < fg.MEAS_GB["global"] / 5
    total = fg.MEAS_BINNED_GB["global"] * 384
    assert 40 < total < 120, f"384 global chunks = {total:.0f} GB"


def test_binned_shape_matches_axis_for():
    assert fg.binned_shape("global", 0.25) == (681, 1440)
    assert fg.binned_shape("na", 0.25) == (281, 481)


def test_dry_run_prints_the_global_plan():
    p = subprocess.run(
        [sys.executable, os.path.join(ROOT, "ml", "fetch_glorys_daily.py"),
         "--dry-run", "--window", "global", "--bin-deg", "0.25",
         "--start", "1993-01", "--end", "1993-02"],
        capture_output=True, text=True)
    assert p.returncode == 0, p.stderr
    assert "8,817,120 cells at 1/12 deg" in p.stdout
    assert "681x1440" in p.stdout
    assert "daily025_global/glorys025_global_199301.nc" in p.stdout


# --------------------------------------------------------------------------
# naming
# --------------------------------------------------------------------------

def test_na_naming_is_e034s_contract_unchanged():
    assert fg.chunk_name("na", 0.0, 1993, 1) == "glorys_199301.nc"
    assert fg.chunk_name("na", 0.0, 2024, 12) == "glorys_202412.nc"
    assert fg.hub_folder("na", 0.0) == ""          # the dataset root, untouched


def test_global_naming_and_folder():
    assert fg.chunk_name("global", 0.25, 1993, 1) == \
        "glorys025_global_199301.nc"
    assert fg.hub_folder("global", 0.25) == "daily025_global/"
    # ...and the name is still parseable by the aggregator's month regex, or
    # the pentad reduction could not derive its bin range from these files.
    m = ag.MONTH_RE.search(fg.chunk_name("global", 0.25, 2024, 12))
    assert m and (int(m.group(1)), int(m.group(2))) == (2024, 12)


def test_the_two_families_cannot_collide():
    na = {fg.hub_folder("na", 0.0) + fg.chunk_name("na", 0.0, y, m)
          for y in (1993, 2024) for m in (1, 12)}
    gl = {fg.hub_folder("global", 0.25) + fg.chunk_name("global", 0.25, y, m)
          for y in (1993, 2024) for m in (1, 12)}
    assert not (na & gl)
    assert all("/" not in n for n in na), "NA chunks live in the root"
    assert all(n.startswith("daily025_global/") for n in gl)


# --------------------------------------------------------------------------
# THE ONE THAT MATTERS: the fetcher's binning IS the aggregator's binning
# --------------------------------------------------------------------------

def test_binning_is_bit_identical_to_aggregate_cadence(binned, tmp_path):
    """Bit-identical, not close — and against the OTHER PROGRAM, not a copy.

    `aggregate_cadence.py` is run as its own process at `--cadence daily
    --bin-deg 0.25`, which for a 1-day bin is exactly "bin this slice". Every
    cell and every NaN must match what the fetcher wrote. A tolerance here
    would hide precisely the class of bug this exists for: a transposed axis,
    a fill value read as a number, a float64 accumulator kept where the
    fetcher rounds to float32.
    """
    import netCDF4 as nc
    src_dir = str(tmp_path / "src")
    os.makedirs(src_dir)
    os.link(binned["src"], os.path.join(src_dir, "glorys_199301.nc"))
    out = str(tmp_path / "agg")
    p = subprocess.run([sys.executable, "-u", AGG, "--in", src_dir,
                        "--cadence", "daily", "--out", out,
                        "--bin-deg", "0.25", "--grid-align", "point"],
                       capture_output=True, text=True)
    assert p.returncode == 0, p.stdout + p.stderr

    idx = np.load(os.path.join(out, "index.npz"))
    d = nc.Dataset(binned["dst"])
    # same axes
    assert np.array_equal(np.asarray(d.variables["latitude"][:]), idx["lat"])
    assert np.array_equal(np.asarray(d.variables["longitude"][:]), idx["lon"])

    t = d.variables["time"]
    dates = [T0 + dt.timedelta(days=float(x) / 24.0) for x in t[:]]
    rows = {int(b): i for i, b in enumerate(idx["bin_index"])}
    n_nan = n_val = 0
    for v in fg.VARIABLES:
        a = np.load(os.path.join(out, f"daily_mean_{v}.npy"), mmap_mode="r")
        got = np.ma.filled(np.ma.masked_invalid(
            np.asarray(d.variables[v][:], dtype=np.float32)), np.nan)
        for i, date in enumerate(dates):
            want = np.asarray(a[rows[ag.bin_index(date, 1)]])
            assert np.array_equal(np.isnan(want), np.isnan(got[i])), (
                f"{v} {date}: NaN masks differ "
                f"({int(np.isnan(want).sum())} vs {int(np.isnan(got[i]).sum())})")
            fin = ~np.isnan(want)
            assert np.array_equal(want[fin], got[i][fin]), (
                f"{v} {date}: values differ, "
                f"max|d| = {np.nanmax(np.abs(want - got[i]))}")
            n_nan += int(np.isnan(want).sum())
            n_val += int(fin.sum())
    d.close()
    # A check that cannot fail is not a check: the fixture must actually
    # contain both coast (all-NaN target cells) and data.
    assert n_nan > 0 and n_val > 0, f"{n_nan} NaN, {n_val} finite"


def test_coast_cells_are_nan_not_zero(binned):
    """A target cell whose source cells are all missing must be NaN.

    Zero is a current, a sea level and a mixed-layer depth of nothing — the
    one wrong answer that survives every downstream mean.
    """
    import netCDF4 as nc
    d = nc.Dataset(binned["src"])
    plan = binned["plan"]
    sl = np.ma.filled(np.ma.masked_invalid(
        d.variables["uo"][0, 0].astype(np.float32)), np.nan)
    ref = ag.bin_slice(sl, plan)
    d.close()
    o = nc.Dataset(binned["dst"])
    got = np.ma.filled(np.ma.masked_invalid(
        np.asarray(o.variables["uo"][0], dtype=np.float32)), np.nan)
    o.close()
    assert np.isnan(ref).any(), "fixture has no all-missing target cell"
    assert np.array_equal(np.isnan(ref), np.isnan(got))
    fin = ~np.isnan(ref)
    assert np.array_equal(ref[fin], got[fin])


def test_binned_chunk_records_the_grid_it_is_on(binned):
    """The attributes a downstream builder refuses on (ml/CLAUDE.md §0.3)."""
    import netCDF4 as nc
    d = nc.Dataset(binned["dst"])
    assert float(d.bin_deg) == 0.25
    assert d.grid_align == "point"
    assert d.source_dataset == fg.DATASET
    assert d.source_window == "global"
    assert d.variables["uo"].dtype == np.float32
    assert d.variables["uo"].dimensions == ("time", "latitude", "longitude")
    assert d.variables["uo"].filters().get("zlib") is True
    # time survives verbatim: aggregate_cadence parses this string.
    assert d.variables["time"].units == TIME_UNITS
    d.close()


# --------------------------------------------------------------------------
# the axes: point alignment, and NA as an exact sub-block
# --------------------------------------------------------------------------

def test_global_axes_are_multiples_of_the_bin():
    lat = ag.axis_for(*fg.SERVED["global"]["lat"], 0.25, "point")
    lon = ag.axis_for(*fg.SERVED["global"]["lon"], 0.25, "point")
    assert (len(lat), len(lon)) == (681, 1440)
    assert lat[0] == -80.0 and lat[-1] == 90.0
    assert lon[0] == -180.0 and lon[-1] == 179.75
    for ax in (lat, lon):
        assert np.allclose(np.round(ax / 0.25), ax / 0.25)
        assert np.allclose(np.diff(ax), 0.25)


def test_na_window_is_an_exact_sub_block_of_the_global_axes():
    """Family 4's grid must be a SLICE of family 7's, not merely near it.

    If it is not, the global tensor's North Atlantic is half a cell away from
    every stencil geometry, the AMOC eval mask and the corridor definitions
    that already exist — and the offset is invisible in every plot.
    """
    g_lat = ag.axis_for(*fg.SERVED["global"]["lat"], 0.25, "point")
    g_lon = ag.axis_for(*fg.SERVED["global"]["lon"], 0.25, "point")
    n_lat = ag.axis_for(*fg.SERVED["na"]["lat"], 0.25, "point")
    n_lon = ag.axis_for(*fg.SERVED["na"]["lon"], 0.25, "point")
    assert (len(n_lat), len(n_lon)) == (281, 481)      # family 3's grid

    i0 = int(np.searchsorted(g_lat, n_lat[0]))
    j0 = int(np.searchsorted(g_lon, n_lon[0]))
    assert (i0, j0) == (320, 320)
    assert np.array_equal(g_lat[i0:i0 + len(n_lat)], n_lat)
    assert np.array_equal(g_lon[j0:j0 + len(n_lon)], n_lon)


def test_hf_prefix_keeps_the_two_families_apart():
    """The aggregator's default listing is the ROOT, i.e. E-034's family.

    Both families' names end `_YYYYMM.nc`, so a bare "every .nc in the repo"
    listing would have folded the global chunks into an NA pentad build and
    aggregated two different grids into one array.
    """
    files = ["glorys_199301.nc", "glorys_199302.nc",
             "daily025_global/glorys025_global_199301.nc", ".preflight"]

    def pick(prefix):
        return sorted(f for f in files if f.endswith(".nc")
                      and f.startswith(prefix) and "/" not in f[len(prefix):])

    assert pick("") == ["glorys_199301.nc", "glorys_199302.nc"]
    assert pick("daily025_global/") == \
        ["daily025_global/glorys025_global_199301.nc"]
    src = open(os.path.join(ROOT, "ml", "aggregate_cadence.py")).read()
    assert "--hf-prefix" in src and 'f.startswith(prefix)' in src


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))

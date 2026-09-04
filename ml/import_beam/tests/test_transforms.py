"""The binning must be the repository's binning, bit for bit.

If this test ever fails, the answer is NOT to adjust transforms.py until the
numbers agree — it is that the import has grown a second definition of the bin
rule, which is the defect class DESIGN §6 exists to prevent.
"""
from __future__ import annotations

import os

import numpy as np
import pytest

from beam_import import transforms


def _bin_rule():
    try:
        return transforms.import_bin_rule()
    except RuntimeError as exc:                  # no earth checkout on this box
        pytest.skip(str(exc))


def test_bin_rule_is_imported_from_the_earth_checkout():
    bin_plan, bin_slice = _bin_rule()
    assert bin_plan.__module__ == "aggregate_cadence"
    assert bin_slice.__module__ == "aggregate_cadence"


def test_our_binning_of_a_toy_grid_equals_bin_slice_exactly():
    bin_plan, bin_slice = _bin_rule()
    rng = np.random.default_rng(11)
    lat = np.arange(-2.0, 2.0001, 1.0 / 12.0)     # a 1/12 degree patch
    lon = np.arange(-3.0, 3.0001, 1.0 / 12.0)
    arr = rng.normal(0.0, 1.0, (len(lat), len(lon)))
    arr[3, 5] = np.nan                            # a hole must survive as NaN

    plan = bin_plan(lat, lon, transforms.BIN_DEG, transforms.BIN_ALIGN)
    want = bin_slice(arr, plan)

    # The exact call transforms.bin025 makes.
    plan2 = bin_plan(lat, lon, transforms.BIN_DEG, transforms.BIN_ALIGN)
    got = bin_slice(arr, plan2)

    assert got.shape == want.shape
    assert np.array_equal(np.isnan(got), np.isnan(want))
    assert np.array_equal(got[~np.isnan(got)], want[~np.isnan(want)])
    # And the axes are on multiples of 0.25 — the family-3 point grid.
    assert np.allclose(plan["lat"] % 0.25, 0.0)
    assert np.allclose(plan["lon"] % 0.25, 0.0)


def test_bin025_on_the_cmems_fixture(fixtures, tmp_path):
    bin_plan, bin_slice = _bin_rule()
    import netCDF4
    days = sorted(os.path.join(fixtures, "cmems", f)
                  for f in os.listdir(os.path.join(fixtures, "cmems")))
    item = {"item_id": "toy/2003-07", "hub_path": "sources/toy/toy_200307.nc",
            "variables": ["uo", "vo"], "dataset_id": "toy", "transform": "bin025"}
    out = transforms.bin025(item, days, str(tmp_path))
    assert len(out) == 1

    with netCDF4.Dataset(days[0]) as ds:
        lat = np.asarray(ds.variables["latitude"][:], dtype=np.float64)
        lon = np.asarray(ds.variables["longitude"][:], dtype=np.float64)
        raw = np.squeeze(np.ma.filled(
            np.asarray(ds.variables["uo"][:], dtype=np.float64), np.nan))
    want = bin_slice(raw, bin_plan(lat, lon, 0.25, "point"))

    with netCDF4.Dataset(out[0]) as ds:
        got = np.asarray(ds.variables["uo"][0], dtype=np.float32)
        assert ds.binning.startswith("ml/aggregate_cadence.py")
    ok = ~np.isnan(want)
    assert np.array_equal(got[ok], want[ok].astype(np.float32))


def test_oisst_year_fold_keeps_the_native_grid_and_the_ice_range(fixtures,
                                                                 tmp_path):
    import netCDF4
    daydir = os.path.join(fixtures, "oisst")
    days = sorted(os.path.join(daydir, f) for f in os.listdir(daydir))
    item = {"item_id": "oisst/2001", "hub_path": "sources/oisst/oisst_2001.nc",
            "transform": "oisst_year_fold"}
    out = transforms.oisst_year_fold(item, days, str(tmp_path))

    with netCDF4.Dataset(days[0]) as ds:
        native_lat = np.asarray(ds.variables["lat"][:], dtype=np.float64)
    with netCDF4.Dataset(out[0]) as ds:
        assert ds.variables["sst"].shape == (3, len(native_lat), 72)
        assert np.allclose(np.asarray(ds.variables["lat"][:]), native_lat)
        # NOT regridded: the axes are still OISST's cell centres.
        assert not np.allclose(np.asarray(ds.variables["lat"][:]) % 0.25, 0.0)
        assert list(ds.variables["ice"].valid_range) == [0.0, 1.0]
        assert ds.variables["ice"].units == "percent"     # the documented trap


def test_passthrough_renames_to_the_hub_basename(tmp_path):
    src = tmp_path / "downloaded.bin"
    src.write_bytes(b"x" * 10)
    item = {"item_id": "s/f", "hub_path": "sources/s/final.bin",
            "transform": "passthrough"}
    out = transforms.passthrough(item, [str(src)], str(tmp_path / "out"))
    assert os.path.basename(out[0]) == "final.bin"

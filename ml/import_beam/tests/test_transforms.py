"""The binning must be the repository's binning, bit for bit.

If this ever fails, the answer is NOT to adjust transforms.py until the
numbers agree — it is that the import has grown a second definition of the bin
rule, which is the defect class DESIGN §7 exists to prevent.
"""
from __future__ import annotations

import os

import numpy as np
import pytest

from beam_import import transforms
from beam_import.example import parse_example, one_str, str_list


def _bin_rule():
    try:
        return transforms.import_bin_rule()
    except RuntimeError as exc:                  # no earth checkout on this box
        pytest.skip(str(exc))


ITEM = {"item_id": "toy/x", "source": "toy", "transform": "x"}


def test_bin_rule_and_epoch_are_imported():
    bin_plan, bin_slice = _bin_rule()
    assert bin_plan.__module__ == "aggregate_cadence"
    assert bin_slice.__module__ == "aggregate_cadence"
    epoch, bin_index = transforms.import_epoch()
    assert epoch.isoformat() == "1982-01-01"
    assert transforms.day_index("1982-01-06") == 5
    assert transforms.day_index("1982-01-06") // 5 == bin_index(
        epoch.replace(day=6), 5)


def test_our_binning_of_a_toy_grid_equals_bin_slice_exactly():
    bin_plan, bin_slice = _bin_rule()
    rng = np.random.default_rng(11)
    lat = np.arange(-2.0, 2.0001, 1.0 / 12.0)
    lon = np.arange(-3.0, 3.0001, 1.0 / 12.0)
    arr = rng.normal(0.0, 1.0, (len(lat), len(lon)))
    arr[3, 5] = np.nan
    plan = bin_plan(lat, lon, transforms.BIN_DEG, transforms.BIN_ALIGN)
    want = bin_slice(arr, plan)
    got = bin_slice(arr, bin_plan(lat, lon, 0.25, "point"))
    assert np.array_equal(np.isnan(got), np.isnan(want))
    assert np.array_equal(got[~np.isnan(got)], want[~np.isnan(want)])
    assert np.allclose(plan["lat"] % 0.25, 0.0)


def test_gridded_record_mask_and_shape():
    rec = transforms.gridded_record(
        ITEM, "1993-01-01", "point025", np.array([0.0, 0.25]),
        np.array([1.0, 1.25, 1.5]),
        ["a"], ["m"], np.array([[[1.0, np.nan, 3.0], [4.0, 5.0, 6.0]]]),
        {"url": "u", "bytes": 7, "sha256": "s", "fetched_at": "t"}, "none")[1]
    assert rec["shape"] == [1, 2, 3]
    assert np.frombuffer(rec["values"], dtype="<f4").size == 6
    bits = np.unpackbits(np.frombuffer(rec["mask"], dtype=np.uint8))[:6]
    assert bits.tolist() == [1, 0, 1, 1, 1, 1]
    assert rec["day_index"] == transforms.day_index("1993-01-01")
    assert rec["lat_values"] == [0.0, 0.25]


def test_oisst_days_keeps_the_native_grid_and_the_ice_trap(fixtures):
    daydir = os.path.join(fixtures, "oisst")
    paths = sorted(os.path.join(daydir, f) for f in os.listdir(daydir))
    item = dict(ITEM, transform="oisst_days")
    recs = transforms.oisst_days(item, paths, {})
    assert len(recs) == 4                        # one day is missing on purpose
    _date, feat = recs[0]
    assert feat["grid"] == "oisst_center025"
    assert feat["var_names"] == ["sst", "ice"]
    # NOT regridded: the axes are still OISST's cell centres.
    assert not np.allclose(np.asarray(feat["lat_values"]) % 0.25, 0.0)
    assert feat["var_units"][1] == "percent"     # the documented trap, intact


def test_ncep_var_year_gives_a_daily_mean_and_the_square_for_stress(fixtures):
    import netCDF4
    path = os.path.join(fixtures, "ncep", "uflx.2003.nc")
    item = dict(ITEM, transform="ncep_var_year", var="uflx")
    recs = transforms.ncep_var_year(item, [path], {})
    assert len(recs) == 5                        # 20 six-hourly steps -> 5 days
    _d, feat = recs[0]
    assert feat["var_names"] == ["uflx", "uflx_sq"]

    with netCDF4.Dataset(path) as ds:
        raw = np.asarray(ds.variables["uflx"][:4], dtype=np.float64)
    cube = np.frombuffer(feat["values"], dtype="<f4").reshape(feat["shape"])
    assert np.allclose(cube[0], raw.mean(axis=0), atol=1e-5)
    assert np.allclose(cube[1], (raw ** 2).mean(axis=0), atol=1e-4)
    # the sign flip is NOT applied here — Stage B applies it once, and it is
    # linear, so the pentad mean is the same number either way
    assert np.sign(np.nanmean(cube[0])) == np.sign(np.nanmean(raw))


def test_a_non_stress_variable_gets_no_square(fixtures):
    path = os.path.join(fixtures, "ncep", "skt.2003.nc")
    item = dict(ITEM, transform="ncep_var_year", var="skt")
    _d, feat = transforms.ncep_var_year(item, [path], {})[0]
    assert feat["var_names"] == ["skt"]


def test_nc025_days_does_not_rebin(fixtures):
    path = os.path.join(fixtures, "ocean", "ocean_200307.nc")
    item = dict(ITEM, transform="nc025_days", grid="point025",
                variables=["uo", "vo", "mlotst", "zos"])
    recs = transforms.nc025_days(item, [path], {})
    assert len(recs) == 5
    _d, feat = recs[0]
    assert feat["transform"] == "none"
    assert feat["var_names"] == ["uo", "vo", "mlotst", "zos"]
    assert np.allclose(np.asarray(feat["lat_values"]) % 0.25, 0.0)


def test_rg_months_lands_on_the_fifteenth(fixtures):
    path = os.path.join(fixtures, "rg", "RG_ArgoClim_200307.nc")
    item = dict(ITEM, transform="rg_months")
    recs = transforms.rg_months(item, [path], {})
    assert len(recs) == 1
    date, feat = recs[0]
    assert date == "2003-07-15"
    assert len(feat["var_names"]) == 32
    assert feat["var_names"][0].startswith("rg_t")


def test_opaque_keeps_the_bytes(fixtures, tmp_path):
    path = os.path.join(fixtures, "tiny", "hello.dat")
    item = dict(ITEM, transform="opaque")
    _d, feat = transforms.opaque(item, [path], {})[0]
    assert feat["grid"] == "opaque"
    assert feat["raw"] == open(path, "rb").read()
    assert len(feat["raw_sha256"]) == 64


def test_to_examples_is_date_ordered_and_deterministic(fixtures):
    path = os.path.join(fixtures, "ocean", "ocean_200307.nc")
    item = dict(ITEM, transform="nc025_days", grid="point025",
                variables=["uo"])
    a, sa, dates = transforms.to_examples(item, [path], {})
    b, sb, _ = transforms.to_examples(item, [path], {})
    assert a == b and sa == sb
    assert dates == sorted(dates)
    assert one_str(parse_example(a[0]), "date") == dates[0]

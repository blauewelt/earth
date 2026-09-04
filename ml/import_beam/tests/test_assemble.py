"""Stage B on the fixtures: bins, days_present, min_days, the sign flip, and
the live-month rule for Roemmich-Gilson."""
from __future__ import annotations

import json
import os

import numpy as np
import pytest

from beam_import import assemble, pipeline, tfrecord
from beam_import.example import one_int, one_str, parse_example, str_list


@pytest.fixture(scope="module")
def staged(tmp_path_factory, request):
    """Run Stage A once for the whole module, then Stage B on its output."""
    reg = request.getfixturevalue("test_registry")
    tmp = tmp_path_factory.mktemp("stageb")
    out = tmp / "out"
    pipeline.run(["--registry", reg, "--tiers", "0", "--output", str(out),
                  "--state-dir", str(tmp / "state"),
                  "--report-dir", str(tmp / "report"), "--offline",
                  "--runner", "DirectRunner"])
    assemble.run(["--registry", reg, "--output", str(out),
                  "--runner", "DirectRunner"])
    return out


def _records(out, group):
    uris = tfrecord.list_uris(str(out / "pentad" / group), ".tfrecord")
    return [parse_example(p) for u in uris for p in tfrecord.read_records(u)]


def _cube(rec):
    shp = [int(x) for x in rec["shape"]]
    return np.frombuffer(rec["values"][0], dtype="<f4").reshape(shp)


def test_all_three_groups_land_in_one_bin(staged):
    cov = json.loads((staged / "pentad" / "coverage.json")
                     .read_text(encoding="utf-8"))
    for group in ("g025", "g100", "rg100"):
        assert cov["groups"][group]["bins_present"] == 1
        assert cov["groups"][group]["bin_min"] == 1573
        assert cov["groups"][group]["bins_missing_in_range"] == 0


def test_the_bin_is_the_imported_rule(staged):
    """bin = floor(day_index / 5) from 1982-01-01 — aggregate_cadence's."""
    from beam_import.transforms import day_index, import_epoch
    _epoch, bin_index = import_epoch()
    rec = _records(staged, "g025")[0]
    assert one_int(rec, "bin") == day_index("2003-07-15") // 5
    assert one_str(rec, "date_start") == "2003-07-15"
    assert one_str(rec, "date_end") == "2003-07-19"


def test_channel_names_and_order_are_imported(staged):
    from beam_import.assemble import f7
    mod = f7()
    for group, want in (("g025", mod.CHAN_G025), ("g100", mod.CHAN_G100),
                        ("rg100", mod.CHAN_RG100)):
        rec = _records(staged, group)[0]
        assert str_list(rec, "chan_names") == list(want)


def test_days_present_is_per_channel_and_min_days_is_applied(staged):
    """The ocean channels saw five days; sea ice saw four, because one OISST
    day is missing upstream. Channels no source covered are NaN with
    days_present 0 — that is min_days = 3, applied HERE and only here."""
    rec = _records(staged, "g025")[0]
    names = str_list(rec, "chan_names")
    dp = dict(zip(names, [int(x) for x in rec["days_present"]]))
    cube = _cube(rec)
    assert dp["cur_speed"] == dp["cur_u"] == dp["cur_v"] == 5
    assert dp["sea_ice"] == 4                # the missing OISST day
    assert np.isfinite(cube[names.index("sea_ice")]).all()

    g100 = _records(staged, "g100")[0]
    gnames = str_list(g100, "chan_names")
    gdp = dict(zip(gnames, [int(x) for x in g100["days_present"]]))
    gcube = _cube(g100)
    # the fixture has uflx and skt only: everything else is missing, and a
    # channel below min_days must be all-NaN rather than a partial average
    assert gdp["tau_x"] == 5 and gdp["u10"] == 0
    assert np.isfinite(gcube[gnames.index("tau_x")]).all()
    assert np.isnan(gcube[gnames.index("u10")]).all()


def test_cur_speed_is_the_hypot_of_the_pentad_means(staged):
    rec = _records(staged, "g025")[0]
    names = str_list(rec, "chan_names")
    cube = _cube(rec)
    u = cube[names.index("cur_u")]
    v = cube[names.index("cur_v")]
    got = cube[names.index("cur_speed")]
    assert np.allclose(got, np.hypot(u, v), atol=1e-5)


def test_log_mld_is_log10_of_the_mean(staged, fixtures):
    import netCDF4
    rec = _records(staged, "g025")[0]
    names = str_list(rec, "chan_names")
    got = _cube(rec)[names.index("log_mld")]
    with netCDF4.Dataset(os.path.join(fixtures, "ocean",
                                      "ocean_200307.nc")) as ds:
        ml = np.asarray(ds.variables["mlotst"][:], dtype=np.float64).mean(0)
    assert np.allclose(got, np.log10(np.maximum(ml, 1e-6)), atol=1e-4)


def test_the_stress_sign_is_flipped_exactly_once(staged, fixtures):
    """build_family7 negates uflx/vflx per 6-hourly sample; the operation is
    linear, so Stage B negates the pentad mean instead. Same number."""
    import netCDF4
    from beam_import.assemble import f7
    assert "uflx" in f7().NCEP_FLIP
    rec = _records(staged, "g100")[0]
    names = str_list(rec, "chan_names")
    tau_x = _cube(rec)[names.index("tau_x")]
    with netCDF4.Dataset(os.path.join(fixtures, "ncep",
                                      "uflx.2003.nc")) as ds:
        raw = np.asarray(ds.variables["uflx"][:], dtype=np.float64)
    # the sign of the regridded pentad mean is the OPPOSITE of the raw mean
    assert np.sign(np.nanmean(tau_x)) == -np.sign(np.nanmean(raw))


def test_tau_std_is_the_within_pentad_population_sigma(staged, fixtures):
    """sigma is not aggregable from a mean, so Stage A carries the daily mean
    of squares and Stage B recovers sqrt(E[x^2] - E[x]^2). It is checked
    against the raw 6-hourly samples, which is what the sigma is defined on."""
    import netCDF4
    rec = _records(staged, "g100")[0]
    names = str_list(rec, "chan_names")
    got = _cube(rec)[names.index("tau_x_std")]
    with netCDF4.Dataset(os.path.join(fixtures, "ncep",
                                      "uflx.2003.nc")) as ds:
        raw = np.asarray(ds.variables["uflx"][:], dtype=np.float64)
        glat = np.asarray(ds.variables["lat"][:], dtype=np.float64)
        glon = np.asarray(ds.variables["lon"][:], dtype=np.float64)
    want_gauss = raw.std(axis=0)                    # population sigma, ddof=0

    from beam_import.assemble import f3
    mod = f3()
    lat = np.asarray(rec["lat_values"], dtype=np.float64)
    lon = np.asarray(rec["lon_values"], dtype=np.float64)
    wy = mod.lin_weights(glat, lat)
    wx = mod.lin_weights(glon, np.where(lon < 0, lon + 360.0, lon),
                         wrap_period=360.0)
    want = mod.interp2_nan(want_gauss, wy, wx)
    assert np.allclose(got, want, atol=2e-3), f"{got}\n{want}"


def test_skin_t_is_oisst_with_the_ncep_fill(staged):
    """DESIGN §4: g025's surface temperature is OISST's sst where OISST has
    one. The fixture's OISST covers the whole tiny grid, so the fill is not
    visible — what is checked here is that the channel came from OISST and is
    in Celsius, not from NCEP's kelvin."""
    rec = _records(staged, "g025")[0]
    names = str_list(rec, "chan_names")
    cube = _cube(rec)
    skin = cube[names.index(names[5])]              # channel 5, whatever it is
    assert np.isfinite(skin).all()
    assert 5.0 < float(np.nanmean(skin)) < 35.0     # degC, not kelvin
    assert "oisst/2003" in str_list(rec, "sources")


def test_rg100_is_written_on_the_live_month_bin_only(staged):
    recs = _records(staged, "rg100")
    assert len(recs) == 1
    rec = recs[0]
    names = str_list(rec, "chan_names")
    assert len(names) == 32 and names[0].startswith("rg_t")
    assert np.isfinite(_cube(rec)).all()
    assert str_list(rec, "sources") == ["rg/RG_ArgoClim_200307.nc"]
    # the record sits on the bin containing the month's 15th
    assert one_str(rec, "date_start") <= "2003-07-15" <= one_str(rec,
                                                                 "date_end")


def test_spec_json_carries_the_norms_out_of_the_data(staged):
    spec = json.loads((staged / "pentad" / "spec.json")
                      .read_text(encoding="utf-8"))
    assert spec["epoch"] == "1982-01-01" and spec["pentad_days"] == 5
    g = spec["groups"]["g025"]
    assert g["shape"][0] == 7
    norms = {n["channel"]: n for n in g["norm"]}
    assert norms["cur_u"]["finite_cells"] > 0
    assert norms["cur_u"]["sd"] is not None


def test_series_and_opaque_records_are_not_in_the_pentad_set(staged):
    """`tiny/hello.dat` is an opaque record; it must not become a pentad."""
    for group in ("g025", "g100", "rg100"):
        for rec in _records(staged, group):
            assert "tiny/hello.dat" not in str_list(rec, "sources")


def test_tf_data_reads_the_pentad_shards(staged):
    tf = pytest.importorskip("tensorflow")
    uris = tfrecord.list_uris(str(staged / "pentad" / "g025"), ".tfrecord")
    spec = {"bin": tf.io.FixedLenFeature([], tf.int64),
            "group": tf.io.FixedLenFeature([], tf.string),
            "shape": tf.io.FixedLenFeature([3], tf.int64)}
    seen = []
    for raw in tf.data.TFRecordDataset(uris):
        got = tf.io.parse_single_example(raw, spec)
        seen.append((int(got["bin"].numpy()),
                     bytes(got["group"].numpy()).decode()))
    ours = [(one_int(r, "bin"), one_str(r, "group"))
            for r in _records(staged, "g025")]
    assert sorted(seen) == sorted(ours)

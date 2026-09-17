#!/usr/bin/env python3
"""The `oceansites` adapter's smoke (family 1.gf, E-082) — no network.

    python3 -m pytest -q tests/test_family1_oceansites.py

The synthetic archive (`oceansites.make_smoke_sources`) is written in the
NDBC GDAC layout: `oceansites_index.txt` with the real v2.0 header (and a
broken continuation line), `DATA/<SITE>/<file>.nc` in three of the measured
file layouts (NetCDF-3 (TIME,) with a scalar DEPTH, NetCDF-3 (TIME, DEPTH),
NetCDF-4 (TIME, DEPTH_TEMP, LATITUDE, LONGITUDE) and (TIME, HEIGHT)), and the
THREDDS NcML headers the site decision reads. A TAO site, an ALOHA Cabled
Observatory file, a currents-only file, a DATA_GRIDDED file and a 2019 file
are listed with garbage bytes and must never be opened. Expected counts are
exact.
"""
import json
import os
import sys

import numpy as np
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "ml"))

import build_family10_stores as b10                             # noqa: E402
import build_family1_stores as b1                               # noqa: E402
import family10_store as f10                                    # noqa: E402
from family1 import adapters as fam                             # noqa: E402
from family1.adapters import oceansites as osa                  # noqa: E402

# NTAS: Dec 30 from the 30-minute column (48 times, the hourly flux
# product's 24 coincide), Dec 31 - Jan 2 from the 10-minute met file (432
# times, the column's 144 coincide); KEO: hourly on Jan 1-2 (48)
TRUTH_ROWS = 48 + 432 + 48
PROBE_ROWS = 288 + 48


def test_registered():
    assert fam.REGISTRY["oceansites"] is osa.OceanSITESAdapter
    ad = osa.OceanSITESAdapter()
    assert ad.C == 28 and ad.family == "1gf" and ad.time_dtype == "int32"
    assert ad.platform_meta and ad.per_year
    assert ad.channel_names[:8] == ["airt", "sst", "sss", "wind_u",
                                    "wind_v", "pres", "sw", "precip"]
    assert ad.channel_names[8] == "T_10" and ad.channel_names[-1] == "S_500"


def test_smoke_all_stages_and_the_probe(tmp_path):
    res = b1.run_smoke("oceansites", root=str(tmp_path / "s"), keep=True)
    assert len(res["truth"]) == TRUTH_ROWS
    p = res["probe"]
    assert p["rows"] == PROBE_ROWS
    assert p["rows_per_day"][:3] == [168, 168, 0]
    assert p["distinct_platforms"] == 2
    assert p["platforms"] == {"entries": 2,
                              "probe_platforms_without_entry": 0}
    c = p["counts"]
    assert c["sites_excluded_gtmba"] == ["T0N110W"]
    assert c["files_excluded_gtmba_site"] == 2
    assert c["sites_selected"] == 2 and c["files_selected"] == 4
    assert c["files_read"] == 4
    assert c["rows_by_site"] == {"KEO": 48, "NTAS": 288}
    assert c["wind_from_speed_direction"] == 1
    assert c["values_qc_removed"] == 1                   # KEO's flag 3
    assert c["values_superseded"] == 48                  # hourly airt
    assert c["files_supplying"]["airt"] == 2
    assert c["files_supplying"]["T_200"] == 1
    assert "T_30" not in c["files_supplying"]      # 20 / 47 m: no match
    assert c["units_unrecognized"] == {}
    assert p["qc"] == {"1": PROBE_ROWS}
    # KEO carries T_200 on 48 rows, one of them flagged 3
    assert p["nan_fraction"]["T_200"] == pytest.approx(289 / 336, abs=1e-5)
    assert p["nan_fraction"]["T_30"] == 1.0
    assert sum(p["out_of_bounds_stored"].values()) == 0
    assert p["bytes_fetched"] > 0 and p["schema_version"] == 2


def _ctx(tmp_path, ad, src):
    import argparse
    a = argparse.Namespace(
        store="oceansites", work=str(tmp_path / "w"), source_dir=src,
        start=ad.smoke_window[0], end=ad.smoke_window[1], stage="all",
        force=False, attempts=1, qc_keep=2,
        check_chunk_rows=b10.CHECK_CHUNK_ROWS, assemble="auto",
        parts_from_hub=False, push_parts=False, allow_missing_years=False)
    return b10.Ctx(a, adapter=ad, layout=b1.layout_for(ad))


def test_the_stages_and_the_store(tmp_path):
    ad = osa.OceanSITESAdapter()
    src = str(tmp_path / "src")
    lo, hi = (b10.parse_date(x) for x in ad.smoke_window)
    truth = ad.smoke_sources(src, lo, hi)
    ctx = _ctx(tmp_path, ad, src)
    b10.run_stages(ctx, ["index", "fetch", "assemble"], stage_fn=b1.STAGE_FN,
                   deps=b1.DEPS)
    b10.check_smoke(ctx, truth)
    out = b1.stage_check(ctx)
    assert out["N"] == TRUTH_ROWS
    plan = json.load(open(os.path.join(ctx.root, "plan.json")))
    ix = plan["index"]
    assert ix["files"] == 10 and ix["usable"] == 7
    assert ix["index_lines_malformed"] == 1
    assert ix["files_not_data_tree"] == 1
    assert ix["files_no_channel_params"] == 1
    assert ix["files_below_500m"] == 1
    assert plan["sites_excluded_gtmba"] == ["T0N110W"]
    assert plan["site_decisions"]["NTAS"] is None
    assert plan["per_year"]["2021"] == {"sites": 1, "files": 3,
                                        "bytes": plan["per_year"]["2021"][
                                            "bytes"]}
    assert plan["first_record"]["file"].startswith("DATA/NTAS/")
    m = json.load(open(os.path.join(ctx.store, "store.json")))
    assert m["per_year"] == {"2021": 192, "2022": 336}
    assert m["counts"]["out_of_bounds"] == {"airt": 1}
    assert m["counts"]["values_qc_removed"] == 2          # + NTAS T10 flag 4
    assert m["counts"]["values_superseded"] == 48 + 24    # both years
    assert m["platforms"]["in_store"] == 2
    pj = json.load(open(os.path.join(ctx.store, "platforms.json")))
    ids = {v["id"]: v for v in pj.values()}
    assert set(ids) == {"KEO", "NTAS"}
    assert ids["KEO"]["lon"] == pytest.approx(-145.0)
    st = f10.Store(ctx.store)
    q = sorted(set(np.asarray(st["qc"]).tolist()))
    assert q == [0, 1, 3]            # column-only rows, met rows, flag 8


def test_the_index_rules():
    files, c = osa.parse_index(
        "#OceanSITES Global Data Assembly Center (GDAC) Index File v2.0\n"
        "DATA/X/OS_X_1_D_M.nc,u,2018-09-01T00:00:00Z,2018-09-02T24:00:00Z,"
        "1,1,2,2,-3.0,-3.0,void,10,c,u,D,time air_temperature\n"
        "DATA/X/OS_ACO_1_P_CTD.nc,u,unknown,unknown,1,1,2,2,4726.5,4726.5,"
        "void,10,c,u,P,time sea_water_temperature\n"
        "DATA/X/OS_X_2_D_T.nc,u,0018-01-01T00:00:00Z,2018-01-01T00:00:00Z,"
        "1,1,2,2,600,900,void,10,c,u,D,time sea_surface_temperature\n"
        " values broken continuation\n")
    assert c["index_lines_malformed"] == 1 and len(files) == 3
    a, b, d = files
    assert a["t1"] == b10.seconds_since_epoch(b10.dt.date(2018, 9, 3))
    assert b["t0"] is None and d["t0"] is None          # 0018 is garbage
    cc = {}
    assert osa.usable(a, cc) and not osa.usable(b, cc)
    assert osa.usable(d, cc)               # a surface parameter: kept
    assert cc == {"files_below_500m": 1}
    with pytest.raises(osa.FormatError, match="empty listing"):
        osa.parse_index("#OceanSITES Global Data Assembly Center\n#\n")
    with pytest.raises(osa.FormatError, match="not an OceanSITES"):
        osa.parse_index("<html>")


def test_the_gtmba_rule_and_the_depth_match():
    assert osa.is_gtmba({"network": "TAO"})
    assert osa.is_gtmba({"id": "TAO_T0N110W_DM072A_D_AIRT"})
    assert osa.is_gtmba({"array": "PIRATA", "network": "PIRATA"})
    assert osa.is_gtmba({"project": "RAMA moorings"})
    assert not osa.is_gtmba({"project": "NTAS", "network": "OCS",
                             "title": "Station Papa (Taobao)"})
    m, surf = osa._match_depths(np.array([1.0, 10.4, 20.0, 47.0, 97.0,
                                          160.0, 290.0, 800.0]))
    assert m == {0: 1, 1: 2, 5: 4, 8: 6} and surf == 0
    m, surf = osa._match_depths(np.array([-4.0]))
    assert m == {} and surf is None


def test_a_bare_time_origin_is_read_only_when_the_file_says_days(tmp_path):
    t = [b10.seconds_since_epoch(b10.dt.datetime(2018, 9, 1, 6))]
    for comment, ok in (("Julian days, timezone: UTC", True),
                        ("time of measurement", False)):
        p = str(tmp_path / f"f{ok}.nc")
        osa._nc(p, "NETCDF3_CLASSIC", t, {}, [], {"network": "ORS"})
        import netCDF4
        ds = netCDF4.Dataset(p, "a")
        ds["TIME"].units = "1950-01-01T00:00:00Z"
        ds["TIME"].comment = comment
        ds.close()
        if ok:
            f = osa.OSFile(p)
            assert f.t.tolist() == t
            assert f.time_units_fallback == "1950-01-01T00:00:00Z"
            f.close()
        else:
            with pytest.raises(osa.FormatError, match="time units"):
                osa.OSFile(p)


def test_the_other_measured_layouts(tmp_path):
    """E1M3A: every variable on (TIME, DEPTH) with a 2-D DEPH and the air
    on one level; MLTS: TEMP(TIME) whose depth is DEPTH_SST(Deployment);
    a monthly-mean product: refused by cadence."""
    t0 = b10.seconds_since_epoch(b10.dt.datetime(2018, 9, 1))
    t = [t0 + 3 * 3600 * i for i in range(8)]
    n = len(t)
    deph = np.tile(np.array([-3.0, 1.0, 20.2, 99.0], np.float32), (n, 1))
    temp = np.full((n, 4), np.float32(1e35))
    temp[:, 1:] = np.array([15.0, 14.0, 12.0], np.float32)
    dryt = np.full((n, 4), np.float32(1e35))
    dryt[:, 0] = 17.5
    p1 = str(tmp_path / "e1m3a.nc")
    osa._nc(p1, "NETCDF3_CLASSIC", t, {"DEPTH": 4, "LATITUDE": n,
                                       "LONGITUDE": n},
            [("LATITUDE", "f4", ("LATITUDE",), np.full(n, 35.5),
              {"standard_name": "latitude"}),
             ("LONGITUDE", "f4", ("LONGITUDE",), np.full(n, 24.9),
              {"standard_name": "longitude"}),
             ("DEPH", "f4", ("TIME", "DEPTH"), deph,
              {"standard_name": "depth", "positive": "down"}),
             ("TEMP", "f4", ("TIME", "DEPTH"), temp,
              {"standard_name": "sea_water_temperature",
               "units": "degree_Celsius", "_FillValue": np.float32(1e35)}),
             ("DRYT", "f4", ("TIME", "DEPTH"), dryt,
              {"standard_name": "air_temperature", "units": "degree_Celsius",
               "_FillValue": np.float32(1e35)})],
            {"network": "HCMR"})
    c = {}
    r = osa.read_os_file(p1, {"name": "e1m3a.nc", "lat": None, "lon": None,
                              "mode": "R"}, -10**12, 10**12, c)
    got = {g[0]: g[2][0].tolist() for g in r["groups"]}
    assert got == {(0,): [17.5], (1,): [15.0], (osa.T0 + 1,): [14.0],
                   (osa.T0 + 5,): [12.0]}
    p2 = str(tmp_path / "mlts.nc")
    osa._nc(p2, "NETCDF3_CLASSIC", t, {"Deployment": 3, "Site": 1},
            [("LATITUDE", "f4", ("Site",), [22.7],
              {"standard_name": "latitude"}),
             ("LONGITUDE", "f4", ("Site",), [-157.9],
              {"standard_name": "longitude"}),
             ("DEPTH_SST", "f4", ("Deployment",), [1.0, 1.05, 0.95],
              {"standard_name": "depth", "positive": "down"}),
             ("TEMP", "f8", ("TIME",), np.full(n, 25.0),
              {"standard_name": "sea_water_temperature",
               "units": "degree_Celsius",
               "coordinates": "TIME DEPTH_SST LATITUDE LONGITUDE"})],
            {"project": "WHOTS"})
    p4 = str(tmp_path / "asimet.nc")
    osa._nc(p4, "NETCDF3_CLASSIC", t, {},
            [("TEMP", "f8", ("TIME",), np.full(n, 11.0),
              {"standard_name": "sea_water_temperature", "units": "celsius",
               "coordinates": "TIME NOMINAL_DEPTH LATITUDE LONGITUDE"}),
             ("TEMP_H", "f8", ("TIME",), np.full(n, 20.5),
              {"standard_name": "height", "positive": "down"})],
            {"project": "SOTS"})
    c = {}
    r = osa.read_os_file(p4, {"name": "asimet.nc", "lat": -47.0,
                              "lon": 142.0, "mode": "D"}, -10**12, 10**12, c)
    assert [g[0] for g in r["groups"]] == [(osa.T0 + 1,)]
    c = {}
    r = osa.read_os_file(p2, {"name": "mlts.nc", "lat": None, "lon": None,
                              "mode": "D"}, -10**12, 10**12, c)
    assert [g[0] for g in r["groups"]] == [(1,)]
    assert r["groups"][0][4][0] == pytest.approx(22.7)
    p3 = str(tmp_path / "monthly.nc")
    tm = [t0 + 30 * 86400 * i for i in range(3)]
    osa._nc(p3, "NETCDF3_CLASSIC", tm, {},
            [("AIRT", "f8", ("TIME",), [20.0, 21.0, 22.0],
              {"standard_name": "air_temperature", "units": "degree_C"})],
            {"project": "WHOTS"})
    c = {}
    r = osa.read_os_file(p3, {"name": "monthly.nc", "lat": 1.0, "lon": 1.0,
                              "mode": "D"}, -10**12, 10**12, c)
    assert r["groups"] == [] and c["files_cadence_over_1day"] == 1


def test_a_missing_file_is_an_absence(tmp_path):
    ad = osa.OceanSITESAdapter()
    src = str(tmp_path / "src")
    lo, hi = (b10.parse_date(x) for x in ad.smoke_window)
    ad.smoke_sources(src, lo, hi)
    os.remove(os.path.join(src, "oceansites", "DATA", "KEO",
                           "OS_KEO_2021_D_MET.nc"))
    ctx = _ctx(tmp_path, ad, src)
    list(ad.fetch_year(ctx, 2022))
    assert len(ctx.absent) == 1 and "OS_KEO_2021_D_MET" in \
        ctx.absent[0]["why"]


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))

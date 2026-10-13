#!/usr/bin/env python3
"""The `s2fdi_coast` adapter (family 1.3 phase C pilot, E-092 D13) — no network.

    python3 -m pytest -q tests/test_family13_s2fdi_coast.py

ACOLITE is STUBBED two ways: the smoke (`_s2fdi_coast_smoke.py`) hands the
adapter rho_s arrays in place of the L1C download and the correction, and
`test_run_acolite_writes_the_probe_settings` runs the adapter's REAL launcher
path against a fake `launch_acolite.py` that writes a NetCDF the way ACOLITE
does. Everything else — the STAC paging, the selection, the mask, the
indices, the sharded writer, the probe, the falsifier check — is the real code.
"""
import datetime as dt
import json
import os
import sys
import zipfile

import numpy as np
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "ml"))

pytest.importorskip("rasterio")
pytest.importorskip("scipy")

import build_family1_stores as b1                               # noqa: E402
from family1 import adapters as fam                             # noqa: E402
from family1 import sharded as sh                               # noqa: E402
from family1.adapters import _s2fdi_coast_smoke as sm           # noqa: E402
from family1.adapters import s2fdi_coast as s2f                 # noqa: E402

ENV = ("S2FDI_TILES", "S2FDI_SMOKE_PX", "S2FDI_CACHE", "S2FDI_ACOLITE_DIR",
       "S2FDI_MAX_LANE_HOURS")


def clear_env(monkeypatch):
    for k in ENV:
        monkeypatch.delenv(k, raising=False)


@pytest.fixture(scope="module")
def smoke(tmp_path_factory):
    root = str(tmp_path_factory.mktemp("s2fdi"))
    keep = {k: os.environ.get(k) for k in ENV}
    try:
        res = b1.run_smoke("s2fdi_coast", root=root, keep=True)
    finally:
        for k, v in keep.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
    d = os.path.join(root, "work", "s2fdi_coast", "s2fdi_coast")
    return res, json.load(open(os.path.join(d, "store.json"))), d, root


# ================================================================= facts ==
def test_registered_and_shaped(monkeypatch):
    clear_env(monkeypatch)
    assert fam.REGISTRY["s2fdi_coast"] is s2f.S2FdiCoastAdapter
    ad = s2f.S2FdiCoastAdapter()
    assert b1.is_grid(ad) and ad.layout == "sharded"
    assert ad.family == "13" and ad.distribution == "public"
    assert b1.layout_for(ad).hf_root == "tensors/family1_3"
    assert ad.channel_names == ["fdi", "fai", "ndvi", "pi", "rho_b04",
                                "rho_b06", "rho_b08", "rho_b11"]
    assert ad.dtype == "float16" and ad.C == 8
    assert (ad.frames_per_bin, ad.frame_seconds) == (5, 86400)
    assert ad.log2_fp == pytest.approx(np.log2(0.020 / 27.83))
    assert "Contains modified Copernicus Sentinel data" in \
        ad.licence["attribution"]
    sp = ad.specs()
    assert len(sp) == 18 and "T35SMD" in sp
    s = sp["T35SMD"]
    assert (s["H"], s["W"], s["C"]) == (5490, 5490, 8)
    assert (s["n_tiles_y"], s["n_tiles_x"]) == (22, 22)
    assert s["grid"]["crs"] == "EPSG:32635"
    assert (s["grid"]["x0"], s["grid"]["y0"]) == (399960.0, 4400040.0)
    assert s["mask"]["cloud_buffer_m"] == 1000.0
    # tile corners from the grid's inverse projection: Lesvos is ~39 N 26 E
    lat = np.array(s["tile_corners"]["lat"])
    lon = np.array(s["tile_corners"]["lon"])
    assert 38.5 < lat.min() < lat.max() < 40.0
    assert 25.5 < lon.min() < lon.max() < 27.5


def test_marida_scene_list_is_the_measured_one():
    assert len(s2f.MARIDA_SCENES) == len(set(s2f.MARIDA_SCENES)) == 63
    assert set(s2f.MARIDA_CDSE_ONLY) < set(s2f.MARIDA_SCENES)
    assert len(s2f.MARIDA_CDSE_ONLY) == 4
    tiles = {t for t, _ in s2f.MARIDA_SCENES}
    assert len(tiles) == 17 and s2f.LESVOS not in tiles
    assert tiles | {s2f.LESVOS} == set(s2f.TILE_GRIDS)
    assert len(s2f.PLP_DATES) == 32 == len(set(s2f.PLP_DATES))
    # zip folder names -> (tile, date) and back
    names = [f"patches/S2_{dt.date.fromisoformat(d).day}-"
             f"{dt.date.fromisoformat(d).month}-"
             f"{dt.date.fromisoformat(d).year % 100}_{t}/x.tif"
             for t, d in s2f.MARIDA_SCENES]
    assert s2f.marida_scene_set(names) == set(s2f.MARIDA_SCENES)


def test_the_probe_settings_wavelengths_and_factors_are_reused_exactly():
    """ml/CLAUDE.md: the store is corrected EXACTLY as the co-location probe
    corrected its crops — same constants, same index arithmetic."""
    from family1 import colocation_probe as cp
    assert s2f.WAVE == cp.WAVE
    assert s2f.FDI_FACTOR == cp.FDI_FACTOR
    assert s2f.L1C_HTTP == cp.L1C_HTTP and s2f.ACOLITE_GIT == cp.ACOLITE_GIT
    assert s2f.CLOUD_BUFFER_M != cp.CLOUD_BUFFER_M   # 300 m was a variant;
    #                                                  1000 m is measured
    rng = np.random.default_rng(1)
    R = {b: rng.uniform(0, 0.05, (8, 8)).astype(np.float32)
         for b in ("B04", "B06", "B08", "B11")}
    for plat in ("sentinel-2a", "sentinel-2b"):
        a = s2f.indices(R, plat)
        b = cp.indices(R, plat)
        for k in ("FDI", "FAI", "NDVI"):
            np.testing.assert_array_equal(a[k.lower()], b[k])


def test_run_acolite_writes_the_probe_settings(tmp_path, monkeypatch):
    """The real launcher path, against a fake ACOLITE: the settings file must
    be the probe's (minus its crop `limit`), and the L2R must be on the
    tile grid or the store refuses it."""
    pytest.importorskip("netCDF4")
    clear_env(monkeypatch)
    fake = tmp_path / "aco"
    fake.mkdir()
    (fake / "launch_acolite.py").write_text(
        "import sys, re, os, numpy as np, netCDF4\n"
        "st = [a for a in sys.argv if a.startswith('--settings=')][0][11:]\n"
        "kv = dict(l.split('=', 1) for l in open(st).read().split() if '=' in l)\n"
        "out = kv['output']\n"
        "open(os.path.join(out, 'X_L1R.nc'), 'w').write('l1r')\n"
        "d = netCDF4.Dataset(os.path.join(out, 'S2B_MSI_X_T35SMD_L2R.nc'), 'w')\n"
        "n = int(os.environ['FAKE_N'])\n"
        "d.createDimension('y', n); d.createDimension('x', n)\n"
        "d.setncattr('xrange', np.array([399960., 399960. + 20 * n]))\n"
        "d.setncattr('yrange', np.array([4400040., 4400040. - 20 * n]))\n"
        "d.setncattr('pixel_size', np.array([20., -20.]))\n"
        "d.setncattr('ac_aot_550', 0.03)\n"
        "for w in ('665', '739', '833', '1610'):\n"
        "    v = d.createVariable('rhos_' + w, 'f4', ('y', 'x'))\n"
        "    v[:] = np.full((n, n), float(w) / 1e5, np.float32)\n"
        "d.close()\n")
    monkeypatch.setenv("S2FDI_ACOLITE_DIR", str(fake))
    monkeypatch.setenv("S2FDI_SMOKE_PX", "16")
    monkeypatch.setenv("FAKE_N", "16")
    ad = s2f.S2FdiCoastAdapter()

    class C:
        scratch = str(tmp_path)
    out = tmp_path / "out"
    l2r, sha, secs = ad.run_acolite(C, "/x/y.SAFE", str(out))
    lines = (out / "settings.txt").read_text().split("\n")
    assert lines[0] == "inputfile=/x/y.SAFE" and lines[1] == f"output={out}"
    assert tuple(l for l in lines[2:] if l) == s2f.ACOLITE_SETTINGS
    assert not any(l.startswith("limit=") for l in lines)
    assert not any(f.endswith("_L1R.nc") for f in os.listdir(out))
    R, attrs = ad.read_l2r(l2r, "35SMD")
    assert set(R) == {"B04", "B06", "B08", "B11"}
    assert float(R["B06"][0, 0]) == pytest.approx(0.00739)
    assert attrs["ac_aot_550"] == pytest.approx(0.03)
    # off the grid by one pixel -> refused
    monkeypatch.setenv("FAKE_N", "17")
    l2r, _, _ = ad.run_acolite(C, "/x/y.SAFE", str(tmp_path / "out2"))
    with pytest.raises(sh.ShardError):
        ad.read_l2r(l2r, "35SMD")


def test_the_settings_are_the_probe_s_own_lines():
    """The probe writes its settings inline; every one of ours is in it."""
    src = open(os.path.join(ROOT, "ml", "family1",
                            "colocation_probe.py")).read()
    for line in s2f.ACOLITE_SETTINGS:
        if line.startswith("l2r_include_bands="):
            for w in line.split("=", 1)[1].split(","):
                assert f'"{w[5:]}"' in src
        else:
            assert f'"{line}"' in src, line


# ============================================================ selection ===
def _f(tile, day, seq, base, cc, kind="L1C", sat="A", dtk=None):
    f = sm.feature(sat, tile, day, seq, base, cc, kind)
    if dtk:
        f["properties"]["s2:datatake_id"] = dtk
    return f


def test_select_rules():
    d = "2021-07-06"
    l1 = [_f("35SMD", d, 0, "03.01", 28.5),          # PLP date, cloudy
          _f("35SMD", "2021-07-07", 0, "03.01", 50.0),
          _f("35SMD", "2021-07-08", 0, "03.01", 5.0)]
    l2 = [_f("35SMD", x, 0, "03.01", None, "L2A")
          for x in (d, "2021-07-07")]
    sel, excl, absent = s2f.select("35SMD", l1, l2,
                                   plp_days=s2f.PLP_DATES)
    assert sel[dt.date(2021, 7, 6)]["selection"] == "plp_campaign_date"
    assert [e["reason"][:11] for e in excl] == ["cloud cover"]
    assert list(absent) == [dt.date(2021, 7, 8)]   # no L2A twin
    # MARIDA: a CDSE-only scene is absent with the CDSE reason
    sel, excl, absent = s2f.select(
        "16PEC", [], [], marida_days=["2015-11-29"],
        cdse_only=s2f.MARIDA_CDSE_ONLY)
    assert "Copernicus Data Space" in absent[dt.date(2015, 11, 29)]
    # two datatakes on one day: refused
    a = _f("35SMD", "2021-07-08", 0, "03.01", 5.0, dtk="GS2A_X_N03.01")
    b = _f("35SMD", "2021-07-08", 1, "03.01", 5.0, dtk="GS2B_Y_N03.01")
    with pytest.raises(s2f.SelectionError):
        s2f.select("35SMD", [a, b], [])
    # the cloud cut is decided on the KEPT copy
    c0 = _f("35SMD", "2019-05-18", 0, "02.07", 28.6)
    c1 = _f("35SMD", "2019-05-18", 1, "02.07", 14.9)
    c2 = _f("35SMD", "2019-05-18", 2, "05.00", 25.0)
    t = _f("35SMD", "2019-05-18", 2, "05.00", None, "L2A")
    sel, excl, _ = s2f.select("35SMD", [c0, c1, c2], [t])
    assert not sel and len(excl) == 3


def test_the_cloud_buffer_is_strict_and_measured_in_metres():
    scl = np.full((200, 200), 6, np.uint8)
    scl[100, 100] = 9
    keep, c = s2f.water_mask(scl)
    # 1000 m = 50 pixels of 20 m: 50 px away is removed, 51 px kept
    assert not keep[100, 150] and keep[100, 151]
    assert not keep[100, 100]
    assert c["cloud_px"] == 1 and c["scl_water_px"] == 200 * 200 - 1
    assert c["water_removed_by_buffer_px"] == int((~keep).sum()) - 1


# ================================================================ smoke ===
def test_smoke_frames_equal_the_truth(smoke):
    res, meta, d, _ = smoke
    ck = res["check"]
    assert ck["frames_equal"] == len(sm.KEPT) == 7
    assert ck["by_reason"]["absent_upstream"] == 1
    assert meta["frames_missing_by_reason"] == ck["by_reason"]
    assert meta["groups"]["T35SMD"]["frames_present"] == 4
    assert meta["groups"]["T16PCC"]["frames_present"] == 2
    assert meta["groups"]["T16PDC"]["frames_present"] == 1


def test_smoke_acquisition_index(smoke):
    _, meta, _, _ = smoke
    acq = s2f.acquisitions(meta)
    got = {(a["tile"], a["day"]): (a["l1c"], a["l2a"]) for a in acq}
    assert got == sm.KEPT
    for a in acq:
        assert a["cloud_buffer_m"] == 1000.0
        assert a["grid"].startswith("EPSG:326")
        day = sh.frame_day(a["bin"], a["frame"], 86400)
        assert str(day) == a["day"] == a["datetime"][:10]
        assert a["selection"] == ("marida_scene" if a["tile"] != "35SMD"
                                  else "cloud_lt_20")
    nw = [a for a in acq if (a["tile"], a["day"]) == sm.NO_WATER][0]
    assert nw["stored_px"] == 0 and nw["scl_water_px"] == 0
    ex = meta["counts"]["acquisitions_excluded"]
    assert len(ex) == sm.EXCLUDED
    assert sum("superseded" in e["reason"] for e in ex) == 4
    ab = meta["counts"]["acquisitions_absent"]
    assert [(a["tile"], a["day"]) for a in ab] == [("16PCC", "2018-09-19")]
    assert meta["counts"]["out_of_bounds"] == {"rho_b04": 1, "rho_b08": 1,
                                                "ndvi": 1, "pi": 1}
    assert meta["counts_by_year"]["2018"]["acquisitions"] == 7


def test_smoke_cloud_case_masks_the_buffer(smoke):
    _, meta, d, _ = smoke
    a = [x for x in s2f.acquisitions(meta)
         if (x["tile"], x["day"]) == sm.OOB_CASE][0]
    assert a["cloud_px"] == 20 * 20 + 10 * 15 + 5 * 3
    assert a["water_removed_by_buffer_px"] > 0
    g = sh.ShardedGroup(os.path.join(d, "T35SMD"))
    fr = g.read_frame(a["bin"], a["frame"])
    # inside the buffer: nothing; far west water: values
    assert np.isnan(fr[110, 50]).all() and np.isnan(fr[110, 99]).all()
    assert np.isfinite(fr[290, 5]).all()
    # pixel (12, 12): NDVI and PI out of bounds, FDI and the bands kept
    px = fr[12, 12]
    assert np.isnan(px[s2f.COL["ndvi"]]) and np.isnan(px[s2f.COL["pi"]])
    assert np.isfinite(px[s2f.COL["fdi"]])
    # land is never stored
    assert np.isnan(fr[:, 200:]).all()


def test_lane_plan_fits_the_runner(monkeypatch):
    clear_env(monkeypatch)
    lanes = s2f.plan_lanes()
    mar = [l for l in lanes if "35SMD" not in l["adapter_env"]]
    les = [l for l in lanes if l["adapter_env"] == "S2FDI_TILES=35SMD"]
    assert sum(l["_acquisitions"] for l in mar) == 59
    assert len(les) == 20
    assert max(l["_acquisitions"] for l in mar) * s2f.SEC_PER_ACQ / 3600 \
        < s2f.MAX_LANE_HOURS
    assert 25 * s2f.SEC_PER_ACQ / 3600 < s2f.MAX_LANE_HOURS
    for l in lanes:
        assert l["extra_args"] == "--push-parts"
        ad_tiles = l["adapter_env"].split("=", 1)[1].split(",")
        assert set(ad_tiles) <= set(s2f.TILE_GRIDS)


# ============================================================ falsifier ===
def _tif_bytes(arr, transform, crs):
    from rasterio.io import MemoryFile
    with MemoryFile() as mf:
        with mf.open(driver="GTiff", width=arr.shape[-1],
                     height=arr.shape[-2],
                     count=arr.shape[0] if arr.ndim == 3 else 1,
                     dtype=arr.dtype, crs=crs, transform=transform) as ds:
            ds.write(arr if arr.ndim == 3 else arr[None])
        return mf.read()


def test_check_marida_on_the_smoke_store(smoke, tmp_path):
    """Falsifiers (i)/(ii) end to end: a MARIDA patch whose bands are an
    exact increasing map of the store's own (MARIDA FDI = 2 x store FDI +
    0.01) must give Spearman 1 at its Marine Water pixels."""
    from rasterio.transform import from_origin
    _, meta, d, _ = smoke
    a = [x for x in s2f.acquisitions(meta)
         if (x["tile"], x["day"]) == ("16PCC", "2018-08-30")][0]
    g = sh.ShardedGroup(os.path.join(d, "T16PCC"))
    fr = g.read_frame(a["bin"], a["frame"]).astype(np.float32)
    grid = s2f.grid_of("16PCC")
    zp = tmp_path / "MARIDA.zip"
    tag = "S2_30-8-18_16PCC"
    with zipfile.ZipFile(zp, "w") as z:
        for k, (r0, c0, split) in enumerate(((230, 10, "train"),
                                              (246, 10, "test")), 1):
            H = W = 32                       # 10 m pixels = 16 x 16 at 20 m
            sub = fr[r0:r0 + 16, c0:c0 + 16]
            up = np.repeat(np.repeat(sub, 2, 0), 2, 1)
            bands = np.zeros((11, H, W), np.float32)
            bands[5] = 2 * up[..., s2f.COL["rho_b06"]]
            bands[7] = 2 * up[..., s2f.COL["rho_b08"]] + 0.01
            bands[9] = 2 * up[..., s2f.COL["rho_b11"]]
            cl = np.full((H, W), 7, np.float32)
            if split == "test":
                cl[8:16, 18:26] = 1              # rows 250-254, cols 19-23
            T = from_origin(grid["x0"] + c0 * 20, grid["y0"] - r0 * 20, 10,
                            10)
            name = f"patches/{tag}/{tag}_{k}"
            z.writestr(name + ".tif", _tif_bytes(bands, T, "EPSG:32616"))
            z.writestr(name + "_cl.tif", _tif_bytes(cl, T, "EPSG:32616"))
        z.writestr("splits/train_X.txt", f"{tag[3:]}_1\n")
        z.writestr("splits/val_X.txt", "")
        z.writestr("splits/test_X.txt", f"{tag[3:]}_2\n")
    res = s2f.check_marida(d, str(zp), out=str(tmp_path / "chk.json"))
    assert res["scenes_in_store"] == 3
    assert res["i"]["spearman_pooled"] > 0.999
    assert res["i"]["pass"] is True
    assert "f1" in res["ii"]["store"] and "f1" in res["ii"]["marida"]
    assert res["i"]["spearman_per_20m_cell"] > 0.999
    assert res["ii"]["store"]["test_debris_px"] == 64
    assert json.load(open(tmp_path / "chk.json"))["labelled_px"] == 2048


def test_plp2019_points_reader(tmp_path):
    """A dBASE file in the PLP2019 Vector_Points shape (fields as the
    producer's, measured 2026-10-13)."""
    import struct
    fields = [("POINT_X", "N", 19), ("POINT_Y", "N", 19),
              ("Pixel_name", "C", 10), ("CP_Bags", "N", 9),
              ("CP_Sea", "N", 9), ("CP_Bottles", "N", 9)]
    recs = [(462485.0, 4328875.0, "A1", 0, 100, 0),
            (462495.0, 4328875.0, "A2", 2, 70, 28)]
    rl = 1 + sum(f[2] for f in fields)
    hl = 32 + 32 * len(fields) + 1
    b = bytearray(struct.pack("<BBBBIHH20x", 3, 26, 3, 30, len(recs), hl,
                              rl))
    for nm, k, ln in fields:
        b += nm.encode().ljust(11, b"\0") + k.encode() + b"\0" * 4 + \
            bytes([ln, 0]) + b"\0" * 14
    b += b"\r"
    for r in recs:
        b += b" " + b"".join(str(v).rjust(ln).encode()[:ln]
                             for v, (_, _, ln) in zip(r, fields))
    b += b"\x1a"
    d = tmp_path / "20190418"
    d.mkdir()
    (d / "20190418.dbf").write_bytes(bytes(b))
    (d / "20190418.prj").write_text('PROJCS["WGS_1984_UTM_Zone_35N"]')
    pts = s2f.plp2019_targets(str(tmp_path))
    assert list(pts) == ["2019-04-18"]
    assert pts["2019-04-18"][1][:3] == (462495.0, 4328875.0, 30.0)


# ============================================== ACOLITE's runtime (#1064) ==
RUNNER_TRACEBACK = """Running ACOLITE processing - Generic GitHub Clone
Traceback (most recent call last):
  File "/acolite/launch_acolite.py", line 81, in <module>
    launch_acolite()
  File "/acolite/acolite/sentinel2/l1_convert.py", line 365, in l1_convert
    dfoo = ac.shared.read_band(jp2_files[0], warp_to=warp_to_geom)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/acolite/acolite/shared/read_band.py", line 114, in read_band
    data = ds.ReadAsArray()
           ^^^^^^^^^^^^^^^^
  File "/usr/lib/python3/dist-packages/osgeo/gdal.py", line 3791, in ReadAsArray
    from osgeo import gdal_array
  File "/usr/lib/python3/dist-packages/osgeo/gdal_array.py", line 10, in <module>
    from . import _gdal_array
ImportError: numpy.core.multiarray failed to import
"""


def test_last_exception_names_the_exception_not_the_head():
    """Run 38391739842's message was the traceback's HEAD cut at 600
    characters — "gdal.py, line 379" was 3791 cut short and the exception
    itself was gone. The message now LEADS with the exception."""
    m = s2f.last_exception(RUNNER_TRACEBACK, frames=2)
    assert m.startswith("ImportError: numpy.core.multiarray failed to import")
    assert "gdal.py\", line 3791, in ReadAsArray" in m
    assert "from . import _gdal_array" in m
    assert "launch_acolite()" not in m
    assert s2f.last_exception("no traceback\nlast words") == \
        "no traceback | last words"


def test_acolite_failure_carries_the_exception(tmp_path, monkeypatch):
    clear_env(monkeypatch)
    fake = tmp_path / "aco"
    fake.mkdir()
    (fake / "launch_acolite.py").write_text(
        "def f():\n    raise RuntimeError('detector footprint unreadable')\n"
        "print('Running ACOLITE'); f()\n")
    monkeypatch.setenv("S2FDI_ACOLITE_DIR", str(fake))
    ad = s2f.S2FdiCoastAdapter()

    class C:
        scratch = str(tmp_path)
    with pytest.raises(IOError) as e:
        ad.run_acolite(C, "/x/y.SAFE", str(tmp_path / "out"))
    assert "RuntimeError: detector footprint unreadable" in str(e.value)
    assert str(e.value).index("RuntimeError") < 60


def test_runtime_check_repairs_or_refuses(tmp_path, monkeypatch):
    clear_env(monkeypatch)
    ad = s2f.S2FdiCoastAdapter()

    class C:
        scratch = str(tmp_path)
    calls = []

    def check(py, env=None):
        calls.append(env)
        if env is None:
            return False, "ImportError: numpy.core.multiarray failed"
        return True, "2.5 3.8.4 private"
    monkeypatch.setattr(s2f, "acolite_python_check", check)
    monkeypatch.setattr(s2f, "build_private_gdal",
                        lambda py, t: (True, f"built into {t}"))
    assert ad.ensure_acolite_runtime(C) == "2.5 3.8.4 private"
    assert ad.acolite_env["PYTHONPATH"].startswith(
        os.path.join(str(tmp_path), "gdal_py"))
    assert "private_build" in ad.runtime
    # the build fails -> a refusal naming the exception and the fix
    ad = s2f.S2FdiCoastAdapter()
    monkeypatch.setattr(s2f, "build_private_gdal",
                        lambda py, t: (False, "no gdal-config"))
    with pytest.raises(SystemExit) as e:
        ad.ensure_acolite_runtime(C)
    msg = str(e.value)
    assert "numpy.core.multiarray" in msg and "no gdal-config" in msg
    assert "--no-build-isolation" in msg


def test_the_runtime_check_passes_here():
    """This interpreter (the sandbox's, pip-built GDAL against numpy 2)
    passes; a failure here means ACOLITE would fail too."""
    pytest.importorskip("osgeo")
    ok, why = s2f.acolite_python_check()
    assert ok, why

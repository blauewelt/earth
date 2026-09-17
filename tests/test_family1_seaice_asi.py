#!/usr/bin/env python3
"""The `seaice_asi` adapter's smoke (family 1.gf, E-082 wave 2) — no network.

    python3 -m pytest -q tests/test_family1_seaice_asi.py

The synthetic archive (`seaice_asi.make_smoke_sources`) is the real layout:
`<n|s>6250/netcdf/index.html` listing year directories, a per-year Apache
listing (with a non-v5.4 file and a stray zip, as 2018's real listing has),
and netCDF-4 days on the REAL grids (1792 x 1216 north, 1328 x 1264 south)
with the real variables and grid_mapping attributes. The record is
2012-12-28 .. 2013-01-08 with 2013-01-03 absent; 2013-01-02 carries one
103.7 % pixel per grid. Expected counts are exact.
"""
import argparse
import datetime as dt
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
from family1 import adapters as fam                             # noqa: E402
from family1 import sharded as sh                               # noqa: E402
from family1.adapters import seaice_asi as asi                  # noqa: E402

# the real 2020 northern listing, two rows as the server printed them
REAL_ROWS = (
    '<tr><td valign="top"><img src="/icons/unknown.gif" alt="[   ]"></td>'
    '<td><a href="asi-AMSR2-n6250-20200101-v5.4.nc">asi-AMSR2-n6250-20200101'
    '-v5.4.nc</a></td><td align="right">2020-01-02 05:29  </td><td align='
    '"right">1.1M</td><td>&nbsp;</td></tr>\n'
    '<tr><td valign="top"><img src="/icons/unknown.gif" alt="[   ]"></td>'
    '<td><a href="asi-AMSR2-n6250-20200102-v5.4.nc">asi-AMSR2-n6250-20200102'
    '-v5.4.nc</a></td><td align="right">2020-01-03 05:29  </td><td align='
    '"right">1.1M</td><td>&nbsp;</td></tr>\n')


@pytest.fixture(scope="module")
def smoke(tmp_path_factory):
    root = str(tmp_path_factory.mktemp("asi_smoke"))
    return b1.run_smoke("seaice_asi", root=root, keep=True)


def small_archive(tmp, lo="2012-12-30", hi="2013-01-01"):
    src = os.path.join(tmp, "src")
    d_lo, d_hi = b10.parse_date(lo), b10.parse_date(hi)
    truth = asi.make_smoke_sources(src, d_lo, d_hi)
    return src, truth


def ctx_for(tmp, src, lo, hi, **over):
    d = dict(store="seaice_asi", work=os.path.join(tmp, "work"),
             source_dir=src, start=lo, end=hi, stage="all", force=False,
             attempts=1, qc_keep=2, check_chunk_rows=b10.CHECK_CHUNK_ROWS,
             assemble="auto", parts_from_hub=False, push_parts=False,
             allow_missing_years=False, allow_unconfirmed_licence=False,
             probe_month="", smoke=False)
    d.update(over)
    ad = asi.SeaIceASIAdapter()
    ctx = b10.Ctx(argparse.Namespace(**d), adapter=ad,
                  layout=b1.layout_for(ad))
    return b1.prepare_grid_ctx(ctx)


def run(ctx, stages):
    b10.run_stages(ctx, stages, stage_fn=b1.GRID_STAGE_FN, deps=b1.DEPS)


# ================================================================ facts ====
def test_registered_with_the_real_grids_and_a_pending_licence():
    assert fam.REGISTRY["seaice_asi"] is asi.SeaIceASIAdapter
    ad = asi.SeaIceASIAdapter()
    assert b1.is_grid(ad) and ad.layout == "sharded"
    assert ad.family == "1gf" and ad.distribution == "public"
    assert ad.licence["redistribution"] == "attribution"
    assert ad.licence["redistribution_confirmed"] is False
    assert ad.channel_names == ["sic"] and ad.dtype == "uint8"
    assert (ad.frames_per_bin, ad.frame_seconds) == (5, 86400)
    assert abs(ad.log2_fp - (-2.154)) < 1e-3
    assert abs(ad.log2_dt - (-2.322)) < 1e-3
    sp = ad.specs()
    assert sorted(sp) == ["n", "s"]
    n, s = sp["n"], sp["s"]
    assert (n["H"], n["W"], n["n_tiles_y"], n["n_tiles_x"]) == \
        (1792, 1216, 7, 5)
    assert (s["H"], s["W"], s["n_tiles_y"], s["n_tiles_x"]) == \
        (1328, 1264, 6, 5)
    assert n["grid"]["crs"] == "EPSG:3411" and s["grid"]["crs"] == "EPSG:3412"
    assert n["missing"] == 255 and n["dtype"] == "uint8"
    # tile corners: the northern grid's lower-left corner is NSIDC's
    # published 33.92 N, and the North Pole is at x = y = 0
    lat, lon = asi.polar_stereo_inverse(0.0, 0.0, 90, 70, -45)
    assert abs(lat - 90) < 1e-9
    assert abs(n["tile_corners"]["lat"][0][0] - 33.92) < 0.01
    assert max(max(r) for r in s["tile_corners"]["lat"]) < -39


def test_the_projection_round_trips_and_matches_pyproj():
    rng = np.random.default_rng(0)
    for h, (lat0, ts, lon0) in (("n", (90, 70, -45)), ("s", (-90, -70, 0))):
        la = rng.uniform(50, 89.9, 200) * np.sign(lat0)
        lo = rng.uniform(-180, 180, 200)
        x, y = asi.polar_stereo_forward(la, lo, lat0, ts, lon0)
        la2, lo2 = asi.polar_stereo_inverse(x, y, lat0, ts, lon0)
        assert np.allclose(la, la2, atol=1e-9)
        assert np.allclose(((lo - lo2 + 180) % 360) - 180, 0, atol=1e-8)
    pyproj = pytest.importorskip("pyproj")
    for h, epsg in (("n", 3411), ("s", 3412)):
        e = asi.HEMI[h]
        t = pyproj.Transformer.from_crs(epsg, 4326, always_xy=True)
        xs = np.array([e["x0"], e["x0"] + e["W"] * 6250.0, 0.0, 1234567.0])
        ys = np.array([e["y0"], e["y0"] + e["H"] * 6250.0, 3125.0, -765432.0])
        lon_p, lat_p = t.transform(xs, ys)
        lat, lon = asi.polar_stereo_inverse(xs, ys, e["lat0"], e["lat_ts"],
                                            e["lon0"])
        assert np.allclose(lat, lat_p, atol=1e-7)
        assert np.allclose(((lon - lon_p + 180) % 360) - 180, 0, atol=1e-7)


def test_the_listing_parsers():
    page = asi._listing_html("/n6250/netcdf/2020", []).replace(
        "</table>", REAL_ROWS + "</table>")
    files, counts = asi.parse_year_listing(page, "n", 2020)
    assert files == {dt.date(2020, 1, 1): (
        "asi-AMSR2-n6250-20200101-v5.4.nc", 1100000),
        dt.date(2020, 1, 2): ("asi-AMSR2-n6250-20200102-v5.4.nc", 1100000)}
    assert counts == {}
    with pytest.raises(asi.FormatError, match="listing"):
        asi.parse_year_listing(page, "s", 2020)
    with pytest.raises(asi.FormatError, match="2021"):
        asi.parse_year_listing(page, "n", 2021)
    dup = page.replace("20200102-v5.4.nc", "20200101-v5.4.nc")
    with pytest.raises(asi.FormatError, match="two"):
        asi.parse_year_listing(dup, "n", 2020)
    other = page.replace("</table>", (
        '<tr><td><a href="asi-AMSR2-n6250-20200101-v5.nc">x</a></td>'
        '<td align="right">2020</td><td align="right">1.0M</td></tr>'
        '<tr><td><a href="2020Geo.zip">2020Geo.zip</a></td></tr></table>'))
    files, counts = asi.parse_year_listing(other, "n", 2020)
    assert len(files) == 2
    assert counts["files_other_version"] == 1
    assert counts["files_other"] == 1
    assert counts["files_other_names"] == ["2020Geo.zip"]
    top = asi._listing_html("/n6250/netcdf", [("2012", None), ("2013", None)],
                            dirs=True)
    assert asi.parse_years(top) == [2012, 2013]
    assert asi.approx_bytes("242K") == 242000


# ================================================================ smoke ====
def test_the_smoke_is_exact(smoke):
    chk = smoke["check"]
    assert chk["frames_equal"] == 22            # 11 days x 2 grids
    assert chk["frames_absent"] == 18
    assert chk["by_reason"] == {"before_record": 8, "absent_upstream": 2,
                                "after_record": 8}
    ctx = smoke["ctx"]
    sm = json.load(open(os.path.join(ctx.store, "store.json")))
    assert sm["tier"] == "G" and sm["layout"] == "sharded"
    assert sm["family"] == "family1_gf" and sm["family_code"] == "1gf"
    assert sm["licence"]["redistribution_confirmed"] is False
    assert sm["counts"]["out_of_bounds"] == {"sic": 2}
    assert sm["counts"]["files_read"] == 22
    assert sm["per_year"] == {"2012": {"n": 6, "s": 6},
                              "2013": {"n": 5, "s": 5}}
    for g, G in sm["groups"].items():
        assert G["dtype"] == "uint8" and G["frames_present"] == 11
        assert G["bins"] == 4 and G["frames_missing"] == 9
    miss = [m for m in sm["missing_frames"] if m["reason"] ==
            "absent_upstream"]
    assert sorted((m["group"], m["day"]) for m in miss) == [
        ("n", "2013-01-03"), ("s", "2013-01-03")]
    plan = json.load(open(os.path.join(ctx.root, "plan.json")))
    for h in ("n", "s"):
        H = plan["hemispheres"][h]
        assert H["record"] == ["2012-12-28", "2013-01-08"]
        assert H["days_absent_in_record"] == ["2013-01-03"]
        assert H["files"] == 11
        assert H["other_files"]["2012"]["files_other_version"] == 1
        assert H["other_files"]["2012"]["files_other"] == 1
    tg = json.load(open(os.path.join(ctx.store, "n", "tile_grid.json")))
    assert "SMALLEST projected y" in tg["grid"]["row_order"]
    assert tg["grid"]["proj4"].startswith("+proj=stere +lat_0=90 +lat_ts=70")


def test_the_smoke_probe(smoke):
    p = smoke["probe"]
    assert p["tier"] == "G" and p["month"] == "2013-01"
    assert p["frames_fetched"] == 14
    for g in ("n", "s"):
        G = p["groups"][g]
        assert G["frames_requested"] == 31
        assert G["frames_fetched"] == 7         # Jan 1, 2, 4..8
        assert G["frames_missing"] == {"absent_upstream": 1,
                                       "after_record": 23}
        assert G["record_frames"] == 11
        assert G["tiles_stored"] + G["tiles_empty"] == \
            7 * G["tiles_per_frame"]
        assert 0.8 < G["valid_fraction"]["sic"] < 1.0
        assert G["estimate_store_bytes"] > 0
    assert p["out_of_bounds"] == {"sic": 2}
    assert p["out_of_bounds_stored"] == 0
    assert p["bytes_fetched"] > 0
    assert p["note_estimate"]["bytes"] == 8e9


def test_the_reader_answers_one_tile_like_the_file(smoke):
    ctx = smoke["ctx"]
    d = dt.date(2013, 1, 2)                     # the day with 103.7 %
    b = b10.seconds_since_epoch(d) // sh.BIN_SECONDS
    f = (b10.seconds_since_epoch(d) - b * sh.BIN_SECONDS) // 86400
    z = asi.smoke_field("n", d).astype(np.float64)
    H, W = z.shape
    i, j = H // 2 + 3, W // 2 + 3
    assert z[i, j] > 100
    t = sh.read_tile(os.path.join(ctx.store, "n"), b, f, i // 256, j // 256)
    assert np.isnan(t[i % 256, j % 256, 0])     # out of bounds -> missing
    want = np.rint(z[(i // 256) * 256:(i // 256) * 256 + 256,
                     (j // 256) * 256:(j // 256) * 256 + 256])
    want[(want < 0) | (want > 100)] = np.nan
    assert np.array_equal(t[..., 0], want.astype(np.float32),
                          equal_nan=True)


def test_a_public_publish_is_refused_while_the_licence_is_pending(
        smoke, monkeypatch):
    ctx = smoke["ctx"]
    called = []
    monkeypatch.setattr(ctx.layout, "hub",
                        lambda: called.append(1) or pytest.fail("hub"))
    with pytest.raises(SystemExit, match="redistribution_confirmed"):
        b1.stage_publish_grid(ctx)
    with pytest.raises(SystemExit, match="redistribution_confirmed"):
        b1.push_parts(ctx)
    assert called == []
    # the CLI has the flag
    a = b1.build_parser().parse_args(
        ["--store", "seaice_asi", "--allow-unconfirmed-licence"])
    assert a.allow_unconfirmed_licence


# ============================================================ refusals =====
def test_an_empty_month_inside_the_record_is_refused(tmp_path, monkeypatch):
    src, _ = small_archive(str(tmp_path))
    # a record that (the listing says) began in November: November has no file
    monkeypatch.setattr(asi.SeaIceASIAdapter, "record",
                        lambda self, ctx, h: (dt.date(2012, 11, 15),
                                              dt.date(2013, 1, 1)))
    ctx = ctx_for(str(tmp_path), src, "2012-11-20", "2013-01-01")
    with pytest.raises(SystemExit, match="calendar month"):
        run(ctx, ["index"])
    b10.mark(ctx.root, "index")
    with pytest.raises(SystemExit, match="could not"):
        run(ctx, ["fetch"])
    assert any("empty month" in e["why"] for e in ctx.absent)
    assert not b10.marked(ctx.root, "parts/2012")


def test_an_empty_top_listing_is_refused(tmp_path):
    src, _ = small_archive(str(tmp_path))
    p = os.path.join(src, "seaice_asi", "n6250", "netcdf", "index.html")
    open(p, "w").write("<html><body>Index of /</body></html>")
    ctx = ctx_for(str(tmp_path), src, "2012-12-30", "2013-01-01")
    with pytest.raises(SystemExit, match="empty listing"):
        run(ctx, ["index"])


def test_a_truncated_file_is_an_absence_not_a_missing_frame(tmp_path):
    src, _ = small_archive(str(tmp_path))
    p = os.path.join(src, "seaice_asi", "s6250", "netcdf", "2012",
                     "asi-AMSR2-s6250-20121231-v5.4.nc")
    raw = open(p, "rb").read()
    open(p, "wb").write(raw[:len(raw) // 3])
    ctx = ctx_for(str(tmp_path), src, "2012-12-30", "2013-01-01")
    run(ctx, ["index"])
    with pytest.raises(SystemExit, match="could not"):
        run(ctx, ["fetch"])
    assert [e["unit"] for e in ctx.absent] == ["2012 s 2012-12-31"]
    assert not b10.marked(ctx.root, "parts/2012")


def test_a_file_on_another_grid_is_refused(tmp_path):
    src, _ = small_archive(str(tmp_path))
    p = os.path.join(src, "seaice_asi", "n6250", "netcdf", "2013",
                     "asi-AMSR2-n6250-20130101-v5.4.nc")
    os.remove(p)
    z = asi.smoke_field("s", dt.date(2013, 1, 1))        # the SOUTHERN grid
    asi.write_nc(p, "s", z)
    ctx = ctx_for(str(tmp_path), src, "2012-12-30", "2013-01-01")
    run(ctx, ["index"])
    with pytest.raises(SystemExit, match="REFUSING seaice_asi"):
        run(ctx, ["fetch"])


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))


def test_the_wgs84_relabel_of_the_same_grid_is_read_and_counted(tmp_path):
    """From 2018-11-02 Bremen declares the identical x/y grid on WGS 84
    (run #6's refusal). Both labels are read; anything else is refused."""
    import shutil
    import netCDF4
    src, _ = small_archive(str(tmp_path), "2013-01-01", "2013-01-01")
    p = None
    for dp, _, ns in os.walk(src):
        for n in ns:
            if n.endswith(".nc") and "-n6250-" in n:
                p = os.path.join(dp, n)
    assert p
    info = {}
    asi.read_nc(p, "n", info)
    assert info == {"ellipsoid": "hughes1980"}
    q = str(tmp_path / "wgs.nc")
    shutil.copyfile(p, q)
    with netCDF4.Dataset(q, "a") as ds:
        gm = ds.variables["polar_stereographic"]
        gm.semi_major_axis = asi.WGS84_A
        gm.inverse_flattening = asi.WGS84_RF
    info = {}
    z = asi.read_nc(q, "n", info)
    assert info == {"ellipsoid": "wgs84"}
    assert np.array_equal(z, asi.read_nc(p, "n"), equal_nan=True)
    with netCDF4.Dataset(q, "a") as ds:
        ds.variables["polar_stereographic"].semi_major_axis = 6371000.0
    with pytest.raises(asi.FormatError, match="neither"):
        asi.read_nc(q, "n")

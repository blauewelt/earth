#!/usr/bin/env python3
"""The `lst05` adapter's smoke (family 1.0.tf, MOD11C1 v061) — no network.

    python3 -m pytest -q tests/test_family1_lst05.py

The synthetic archive (`lst05.make_smoke_sources`) is the real layout: a CMR
`granules.json` response in the service's own shape, which the REAL parser
(`_modis_cmg.parse_cmr`) reads, plus HDF-EOS2 granules carrying the real SDS
names and the real `scale_factor` / `_FillValue` / `valid_range` attributes on
a 600 x 300 grid (`LST05_SMOKE_GRID`). The record is 2015-07-01 .. 2015-07-20
with 07-14 not listed (`absent_upstream`) and 07-03 carrying one raw 65535
pixel — inside the producer's valid_range and 1,037 degrees C, so it exercises
the channel bound rather than the fill rule. Expected counts are exact.
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
from family1.adapters import _modis_cmg as mc                   # noqa: E402
from family1.adapters import lst05 as ls                        # noqa: E402

pytest.importorskip("pyhdf")

ENV = ("LST05_SMOKE_GRID", "LST05_GROUPS")


def clear_env(monkeypatch):
    for k in ENV:
        monkeypatch.delenv(k, raising=False)


@pytest.fixture(scope="module")
def smoke(tmp_path_factory):
    root = str(tmp_path_factory.mktemp("lst05_smoke"))
    keep = {k: os.environ.get(k) for k in ENV}
    try:
        return b1.run_smoke("lst05", root=root, keep=True)
    finally:
        for k, v in keep.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


def small_archive(tmp, **kw):
    src = os.path.join(tmp, "src")
    d_lo, d_hi = (b10.parse_date(x) for x in ls.Lst05Adapter.smoke_window)
    truth = ls.make_smoke_sources(src, d_lo, d_hi, **kw)
    return src, truth


def ctx_for(tmp, src, lo=None, hi=None, **over):
    w = ls.Lst05Adapter.smoke_window
    d = dict(store="lst05", work=os.path.join(tmp, "work"), source_dir=src,
             start=lo or w[0], end=hi or w[1], stage="all", force=False,
             attempts=1, qc_keep=2, check_chunk_rows=b10.CHECK_CHUNK_ROWS,
             assemble="auto", parts_from_hub=False, push_parts=False,
             allow_missing_years=False, allow_unconfirmed_licence=False,
             probe_month="", smoke=False)
    d.update(over)
    ad = ls.Lst05Adapter()
    ctx = b10.Ctx(argparse.Namespace(**d), adapter=ad,
                  layout=b1.layout_for(ad))
    return b1.prepare_grid_ctx(ctx)


def run(ctx, stages):
    b10.run_stages(ctx, stages, stage_fn=b1.GRID_STAGE_FN, deps=b1.DEPS)


# ================================================================ facts ====
def test_registered_with_the_real_grid(monkeypatch):
    clear_env(monkeypatch)
    assert fam.REGISTRY["lst05"] is ls.Lst05Adapter
    ad = ls.Lst05Adapter()
    assert b1.is_grid(ad) and ad.layout == "sharded"
    assert ad.family == "1tf" and ad.distribution == "public"
    assert ad.licence["redistribution"] == "attribution"
    assert ad.licence["redistribution_confirmed"] is True
    assert ad.credentials == ("EARTHDATA_USERNAME", "EARTHDATA_PASSWORD")
    assert ad.channel_names == ["lst_day", "lst_night", "qc_day", "qc_night"]
    assert ad.C == 4 and ad.dtype == "float16"
    assert (ad.frames_per_bin, ad.frame_seconds) == (5, 86400)
    assert ad.frames_per_bin * ad.frame_seconds == sh.BIN_SECONDS
    assert ad.first_year == 2000
    # 0.05 degrees is 5.566 km at the equator; a frame is one of five days
    assert abs(ad.log2_fp - (-2.3219)) < 1e-3
    assert abs(ad.log2_dt - (-2.3219)) < 1e-3
    assert list(ad.groups) == ["terra"]
    sp = ad.specs()["terra"]
    assert (sp["H"], sp["W"], sp["C"]) == (3600, 7200, 4)
    # 7200 / 256 = 28.125 -> 29 across, 3600 / 256 = 14.06 -> 15 down
    assert (sp["n_tiles_y"], sp["n_tiles_x"]) == (15, 29)
    assert sp["pad_rows"] == 240 and sp["pad_cols"] == 224
    assert sp["dtype"] == "float16" and sp["missing"] == "NaN"
    g = sp["grid"]
    assert g["crs"] == "EPSG:4326"
    assert (g["x0"], g["y0"]) == (-180.0, 90.0)
    assert g["dx"] == 0.05 and g["dy"] == -0.05
    assert g["extent"] == [-180.0, -90.0, 180.0, 90.0]
    assert "row 0 is the NORTHERNMOST" in g["row_order"]


def test_kelvin_would_have_been_the_wrong_unit():
    """The measurement behind the Celsius decision: float16 near 300 K is a
    quarter of a degree, near 27 C it is three hundredths."""
    step_k = float(np.float16(300.0).astype(np.float64)
                   - np.nextafter(np.float16(300.0), np.float16(0)
                                  ).astype(np.float64))
    step_c = float(np.nextafter(np.float16(26.85), np.float16(1e4)
                                ).astype(np.float64)
                   - np.float16(26.85).astype(np.float64))
    assert 0.2 < step_k < 0.3
    assert step_c < 0.04
    assert step_k / step_c > 5


def test_the_channel_bounds_survive_float16(monkeypatch):
    """A bound the stored dtype cannot represent makes the store fail its own
    tile check on a legitimate extreme; all four of lst05's are exact."""
    clear_env(monkeypatch)
    ad = ls.Lst05Adapter()
    lo, hi = ad.bounds()
    for v in list(lo) + list(hi):
        assert float(np.float16(v)) == float(v)


def test_a_granule_id_is_parsed_and_a_bad_one_refused():
    d, made = mc.granule_key("MOD11C1.A2015182.061.2021358223019")
    assert d == dt.date(2015, 7, 1) and made == 2021358223019
    # NSIDC's titles carry the extension; LP DAAC's do not
    assert mc.granule_key("MOD10C1.A2015032.061.2021318072948.hdf")[0] == \
        dt.date(2015, 2, 1)
    for bad in ("", "MOD11C1.2015182.061.1", "MOD11C1.A2015400.061.1",
                "nonsense"):
        with pytest.raises(mc.FormatError):
            mc.granule_key(bad)


def test_the_cmr_parser_takes_the_newest_of_two_granules_for_one_day():
    older = ("MOD11C1.A2015182.061.2020001000000",
             ls.LP_PREFIX + "a/a.hdf", 40.0)
    newer = ("MOD11C1.A2015182.061.2021358223019",
             ls.LP_PREFIX + "b/b.hdf", 40.0)
    feed = mc.cmr_json([older, newer])["feed"]["entry"]
    files, counts = mc.parse_cmr(feed)
    assert list(files) == [dt.date(2015, 7, 1)]
    assert files[dt.date(2015, 7, 1)]["title"] == newer[0]
    assert counts["granules_superseded"] == 1
    # a granule with no protected https .hdf link is a REFUSAL
    feed2 = mc.cmr_json([newer])["feed"]["entry"]
    feed2[0]["links"] = [{"rel": "x/data#", "href": "http://x/y.hdf"}]
    with pytest.raises(mc.FormatError):
        mc.parse_cmr(feed2)


def test_only_urs_is_named_in_a_netrc_check(tmp_path):
    home = str(tmp_path)
    with open(os.path.join(home, ".netrc"), "w") as fh:
        fh.write("machine example.invalid login a password b\n")
    assert mc.netrc_has_urs(home) is False
    with open(os.path.join(home, ".netrc"), "w") as fh:
        fh.write(f"machine {mc.URS_HOST} login a password b\n")
    assert mc.netrc_has_urs(home) is True


def test_groups_env_refuses_a_group_the_adapter_cannot_build(monkeypatch):
    clear_env(monkeypatch)
    monkeypatch.setenv("LST05_GROUPS", "aqua")
    ad = ls.Lst05Adapter()
    assert list(ad.groups) == ["aqua"]
    assert ad.collection("aqua") == ("MYD11C1", "061")
    assert "GROUPS" in ad.notes
    monkeypatch.setenv("LST05_GROUPS", "viirs")
    with pytest.raises(SystemExit) as e:
        ls.Lst05Adapter()
    assert "groups this adapter can build" in str(e.value)


# ============================================================== the smoke ==
def test_smoke_reads_every_frame_back_exactly(smoke):
    res = smoke["check"]
    assert res["frames_equal"] == 19          # 20 listed days less 07-14
    assert res["by_reason"]["absent_upstream"] == 1
    sm = json.load(open(os.path.join(smoke["ctx"].store, "store.json")))
    assert sm["tier"] == "G" and sm["layout"] == "sharded"
    assert list(sm["groups"]) == ["terra"]
    assert sm["C"] == 4 and sm["dtype"] == "float16"
    assert sm["frames_missing_by_reason"]["absent_upstream"] == 1
    assert sm["counts"]["out_of_bounds"] == {"lst_day": 1}
    assert "SMOKE GRID" in sm["notes"]


def test_the_quality_channels_are_valid_where_the_temperature_is_not(smoke):
    """The decision this store turns on: QC is a reading everywhere, LST only
    where a retrieval was made, so every tile of every frame is stored."""
    grp = sh.ShardedGroup(os.path.join(smoke["ctx"].store, "terra"))
    arr = grp.shard_index
    vf = np.asarray([r["valid_fraction"] for r in arr], np.float64)
    day, night, qcd, qcn = vf[:, 0], vf[:, 1], vf[:, 2], vf[:, 3]
    assert (qcd == 1.0).all() and (qcn == 1.0).all()
    assert (day < 0.9).all() and (day > 0.0).all()
    assert (night < 0.9).all()
    # nothing is empty: a QC value exists for every pixel, so EVERY tile of
    # every present frame is stored
    sp = grp.spec
    assert int(arr["tiles_stored"].sum()) == (
        int(arr["frames_present"].sum())
        * sp["n_tiles_y"] * sp["n_tiles_x"])


def test_a_real_frame_reads_back_with_its_fill_as_nan(smoke):
    truth = smoke["truth"]
    grp = sh.ShardedGroup(os.path.join(smoke["ctx"].store, "terra"))
    n = 0
    for (g, b, f), (want, why) in sorted(truth.items()):
        if want is None:
            continue
        got = grp.read_frame(b, f, raw=True)
        assert got is not None and got.shape == want.shape
        assert np.array_equal(got, want, equal_nan=True)
        lst = got[:, :, 0]
        qc = got[:, :, 2]
        assert np.isnan(lst).any()                # cloud and ocean
        assert not np.isnan(qc).any()             # QC has no fill
        v = lst[np.isfinite(lst)]
        assert v.size and v.min() >= -150.0 and v.max() <= 100.0
        n += 1
    assert n == 19


def test_the_probe_measured_the_same_frames_and_some_bytes(smoke):
    p = smoke["probe"]
    assert p["frames_fetched"] == 19
    assert p["out_of_bounds_stored"] == 0
    assert p["out_of_bounds"] == {"lst_day": 1}
    assert p["bytes_fetched"] > 0
    gp = p["groups"]["terra"]
    assert gp["C"] == 4 and gp["dtype"] == "float16"
    # July 2015 has 31 days; the smoke's record stops on the 20th
    assert gp["frames_missing"] == {"absent_upstream": 1, "after_record": 11}
    assert gp["valid_fraction"]["qc_day"] == 1.0
    assert 0.0 < gp["valid_fraction"]["lst_day"] < 1.0
    assert gp["record_frames"] == 19
    assert gp["estimate_store_bytes"] > 0


def test_an_absurd_but_in_range_raw_value_is_counted_not_clipped(smoke):
    """Raw 65535 passes the producer's own valid_range and is 1,037 degrees
    C, so the CHANNEL bound is what catches it — as NaN, counted."""
    truth = smoke["truth"]
    hit = [(b, f) for (g, b, f), (a, why) in truth.items()
           if a is not None and sh.frame_day(b, f, 86400) == ls.SMOKE_OOB]
    assert len(hit) == 1
    b, f = hit[0]
    grp = sh.ShardedGroup(os.path.join(smoke["ctx"].store, "terra"))
    a = grp.read_frame(b, f)
    assert np.isnan(a[ls.SMOKE_H // 2, ls.SMOKE_W // 2, 0])


def test_a_declared_attribute_that_differs_is_a_refusal(tmp_path):
    """A product whose scaling changed must be re-read by a human."""
    src, _ = small_archive(str(tmp_path))
    gdir = os.path.join(src, "lst05", "granules", "terra")
    name = sorted(n for n in os.listdir(gdir) if n.endswith(".hdf"))[-1]
    mc.set_sds_attrs(os.path.join(gdir, name),
                     {ls.SDS_LST_DAY: {"scale_factor": 0.01}})
    ctx = ctx_for(str(tmp_path), src)
    with pytest.raises(SystemExit) as e:
        run(ctx, ["index"])
    assert "scale_factor" in str(e.value)


def test_index_refuses_an_empty_listing(tmp_path):
    src, _ = small_archive(str(tmp_path))
    p = os.path.join(src, "lst05", "cmr_terra.json")
    with open(p, "w") as fh:
        json.dump(mc.cmr_json([]), fh)
    ctx = ctx_for(str(tmp_path), src)
    with pytest.raises(SystemExit) as e:
        run(ctx, ["index"])
    assert "no granule" in str(e.value)


def test_a_listed_granule_that_is_not_there_is_an_absence(tmp_path):
    """A granule CMR lists and the archive will not serve stops the year's
    marker — it is never a silently empty frame."""
    src, _ = small_archive(str(tmp_path), absent=(dt.date(2015, 7, 9),))
    ctx = ctx_for(str(tmp_path), src)
    with pytest.raises(SystemExit) as e:
        run(ctx, ["index", "fetch"])
    assert "could not be read" in str(e.value)
    assert any("2015-07-09" in a["unit"] for a in ctx.absent)
    from build_family7 import marked
    assert not marked(ctx.root, "fetch")


def test_a_truncated_granule_is_an_absence_not_a_frame(tmp_path):
    src, _ = small_archive(str(tmp_path))
    gdir = os.path.join(src, "lst05", "granules", "terra")
    name = sorted(n for n in os.listdir(gdir) if n.endswith(".hdf"))[3]
    with open(os.path.join(gdir, name), "r+b") as fh:
        fh.truncate(64)
    ctx = ctx_for(str(tmp_path), src)
    with pytest.raises(SystemExit) as e:
        run(ctx, ["index", "fetch"])
    assert "could not be read" in str(e.value)


def test_check_store_verifies_every_stored_tile(smoke):
    st = sh.check_store(smoke["ctx"].store)
    g = st["groups"]["terra"]
    assert g["tiles_checked"] == g["tiles_stored"] > 0
    assert g["frames_present"] == 19

#!/usr/bin/env python3
"""The three BIOSPHERE catalogue adapters' smokes (E-082 wave 6) — no network.

    python3 -m pytest -q tests/test_family1_cat_biosphere.py

Three tier-T stores, each of which lists somebody else's pixels rather than
copying them:

  `canopy_ref`       ETH Zurich's 10 m canopy height for 2020 and Meta/WRI's
                     ~1 m canopy height (v1 and v2) — how tall the trees are,
                     finer than anything else these families hold.
  `cat_palsar`       JAXA's L-band radar: the ScanSAR Level 2.2 scenes (which
                     are anonymous on AWS Open Data) and the 25 m global
                     mosaics (which are not, and say so by name).
  `cat_biomass_esa`  ESA's BIOMASS, the first P-band radar in orbit, listed
                     through FedEO — the CEOS federated EO catalogue, and the
                     one of four endpoints that answered on 2026-09-20.

As with the other eight catalogues, the synthetic "archive" is a directory of
CANNED RESPONSES recorded from an in-process fake producer, so the real
paging, the real refusals and the real packing run against them and only the
socket is replaced. Every count below is derived from the descriptors in the
adapters, not copied from a run.
"""
import json
import os
import sys

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "ml"))

import build_family10_stores as b10                             # noqa: E402
import build_family1_stores as b1                               # noqa: E402
import family10_store as f10                                    # noqa: E402
from family1 import adapters as fam                             # noqa: E402
from family1.adapters import _stac as st                        # noqa: E402
from family1.adapters import canopy_ref as cr                   # noqa: E402
from family1.adapters import cat_palsar as cp                   # noqa: E402
from family1.adapters import cat_biomass_esa as cb              # noqa: E402

pytest.importorskip("pyarrow")

# store -> rows the store must hold
EXPECT = {
    # 3 ETH tiles + 3 Meta v1 quadkeys + 4 Meta v2 quadkeys
    "canopy_ref": 10,
    # 6 ScanSAR items, one of which is acquired outside the window
    "cat_palsar": 5,
    # 8 BIOMASS products: one starts before the window, one has no enclosure
    "cat_biomass_esa": 6,
}


@pytest.fixture(scope="module")
def smokes(tmp_path_factory):
    out = {}
    for s in sorted(EXPECT):
        root = tmp_path_factory.mktemp(s)
        out[s] = b1.run_smoke(s, root=str(root), keep=True)
    return out


def work_of(smokes, store):
    return os.path.join(smokes[store]["work"], store, store)


# ====================================================== the shared contract ==
@pytest.mark.parametrize("store", sorted(EXPECT))
def test_the_store_holds_exactly_the_rows_the_adapter_emitted(store, smokes):
    res = smokes[store]
    assert len(res["truth"]) == EXPECT[store], store
    p = res["probe"]
    assert p["rows"] == EXPECT[store]
    assert p["distinct_platforms"] == EXPECT[store]      # no hash collisions
    assert sum(p["rows_per_day"]) == EXPECT[store]
    assert sum(p["out_of_bounds_stored"].values()) == 0
    assert p["bytes_fetched"] > 0
    assert p["schema_version"] == 2 and p["time_dtype"] == "int32"
    assert p["counts_scope"] == "month"
    assert p["counts"]["probe_month_only"] == 1


@pytest.mark.parametrize("store", sorted(EXPECT))
def test_it_is_a_tier_t_catalogue_of_family_1tf(store, smokes):
    ad = fam.REGISTRY[store]()
    assert isinstance(ad, st.CatalogueAdapter)
    assert ad.family == "1tf" and ad.distribution == "public"
    assert ad.C == 5
    assert ad.channel_names == ["cloud", "valid", "angle", "log2_area",
                                "sensor"]
    assert ad.platform_meta is False
    with open(os.path.join(work_of(smokes, store), "store.json")) as fh:
        sj = json.load(fh)
    assert sj["family"] == "family1_tf" and sj["family_code"] == "1tf"
    assert sj["tier_t_catalogue"] is True
    assert sj["sensor_table"] == {str(k): v
                                  for k, v in sorted(ad.SENSOR_TABLE.items())}
    assert sj["qc_table"] == {str(k): v
                              for k, v in sorted(ad.QC_TABLE.items())}
    assert sj["assets"]["rows"] == EXPECT[store]
    assert st.ASSETS_PARQUET in sj["sha256"]


@pytest.mark.parametrize("store", sorted(EXPECT))
def test_the_sidecar_has_one_row_per_tile_and_rebuilds_the_urls(store,
                                                                smokes):
    import pyarrow.parquet as pq
    w = work_of(smokes, store)
    t = pq.read_table(os.path.join(w, st.ASSETS_PARQUET))
    assert t.num_rows == EXPECT[store]
    assert list(t.column_names) == list(st.ASSET_COLS)
    plat = t.column("platform").to_pylist()
    assert len(set(plat)) == len(plat)
    sm = f10.Store(w)
    assert sorted(set(int(x) for x in sm["platform"])) == sorted(plat)
    for pid, sid, base, tmpl in zip(plat, t.column("stac_id").to_pylist(),
                                    t.column("base_url").to_pylist(),
                                    t.column("asset_set").to_pylist()):
        assert b10.platform_hash(sid) == pid
        assert base.startswith("http")
        urls = [base + "/" + n.replace("{id}", sid) for n in tmpl.split()]
        assert urls and all(u.startswith("http") for u in urls)


@pytest.mark.parametrize("store", sorted(EXPECT))
def test_no_row_claims_a_cloud_a_valid_fraction_or_an_angle(store, smokes):
    """All three list products that publish none of the three, and NaN is the
    honest answer — a zero would read as 'clear sky, fully valid, nadir'."""
    p = smokes[store]["probe"]
    for ch in ("cloud", "valid", "angle"):
        assert p["nan_fraction"][ch] == 1.0, (store, ch)
    # the area, by contrast, is real on every row
    assert p["nan_fraction"]["log2_area"] == 0.0
    assert p["nan_fraction"]["sensor"] == 0.0


# ================================================================ canopy_ref =
def test_canopy_ref_lists_both_producers_and_all_three_products(smokes):
    c = smokes["canopy_ref"]["probe"]["counts"]
    assert c["tiles_eth10"] == 3
    assert c["tiles_meta1_v1"] == 3
    assert c["tiles_meta1_v2"] == 4
    assert c["eth_features"] == 3
    assert c["meta_v2_manifest_tiles"] == 4      # the blank line was skipped
    assert c["rows_kept"] == 10
    # three product versions, and qc is what tells them apart
    p = smokes["canopy_ref"]["probe"]
    assert p["qc"] == {"1": 3, "2": 3, "3": 4}


def test_canopy_ref_files_every_row_under_its_reference_epoch(smokes):
    """A composite of a decade of aerial photography has no one date, so the
    row carries the collection's declared epoch and the sidecar names the
    per-tile metadata a reader goes to for the real ones."""
    import datetime as dt
    import pyarrow.parquet as pq
    sm = f10.Store(work_of(smokes, "canopy_ref"))
    want = b10.seconds_since_epoch(dt.date(2020, 12, 31)) + 86399
    assert sorted(set(int(x) for x in sm["time_s"])) == [want]
    t = pq.read_table(os.path.join(work_of(smokes, "canopy_ref"),
                                   st.ASSETS_PARQUET))
    sets = dict(zip(t.column("stac_id").to_pylist(),
                    t.column("asset_set").to_pylist()))
    # the ETH row points at the height map AND its standard deviation
    eth = [k for k in sets if k.startswith("ETH_")][0]
    assert len(sets[eth].split()) == 2
    # a Meta row points at the tile and at the metadata sidecar carrying the
    # acquisition dates this store deliberately does not put in `time_s`
    v1 = [k for k in sets if k.startswith("META_CHM_v1_")][0]
    assert any(n.endswith(".geojson") for n in sets[v1].split())


def test_a_quadkey_is_an_extent_and_the_arithmetic_is_right():
    """Meta v2's footprints come from the quadkey itself, so the quadkey
    arithmetic has to be exact rather than approximately right."""
    # the four level-1 quadkeys tile the whole web-mercator world
    lim = 85.0511287798066
    assert cr.quadkey_bbox("0") == pytest.approx((-180.0, 0.0, 0.0, lim),
                                                 abs=1e-9)
    assert cr.quadkey_bbox("1") == pytest.approx((0.0, 0.0, 180.0, lim),
                                                 abs=1e-9)
    assert cr.quadkey_bbox("2") == pytest.approx((-180.0, -lim, 0.0, 0.0),
                                                 abs=1e-9)
    assert cr.quadkey_bbox("3") == pytest.approx((0.0, -lim, 180.0, 0.0),
                                                 abs=1e-9)
    # a level-9 tile is 360/512 degrees wide, everywhere
    w, s, e, n = cr.quadkey_bbox("023013213")
    assert e - w == pytest.approx(360.0 / 512.0, abs=1e-9)
    assert -90.0 < s < n < 90.0
    # a child sits inside its parent
    pw, ps, pe, pn = cr.quadkey_bbox("02301321")
    assert pw <= w and e <= pe and ps <= s and n <= pn
    for bad in ("", "4", "01a", "0 1"):
        with pytest.raises(st.Refusal):
            cr.quadkey_bbox(bad)


def test_an_eth_tile_name_and_its_index_geometry_must_agree():
    assert cr.eth_tile_bbox(
        "ETH_GlobalCanopyHeight_10m_2020_N00E006_Map.tif") == (6.0, 0.0,
                                                               9.0, 3.0)
    assert cr.eth_tile_bbox(
        "ETH_GlobalCanopyHeight_10m_2020_S60W030_Map.tif") == (-30.0, -60.0,
                                                               -27.0, -57.0)
    with pytest.raises(st.Refusal):
        cr.eth_tile_bbox("ETH_GlobalCanopyHeight_10m_2020_Map.tif")
    # a feature whose bbox contradicts its own name is a refusal, because a
    # footprint taken from either alone would then be wrong
    ad = cr.CanopyRefCatalogue()
    nm = "ETH_GlobalCanopyHeight_10m_2020_N00E006_Map.tif"

    class _Fet:
        pass

    import family1.adapters._stac as _st
    real = _st._text
    _st._text = lambda fet, url: (
        "<html>x_add(" + json.dumps({
            "features": [{"bbox": [0.0, 0.0, 3.0, 3.0],
                          "properties": {"tile_name": nm},
                          "type": "Feature"}],
            "type": "FeatureCollection"}) + ");</html>")
    try:
        with pytest.raises(st.Refusal) as e:
            ad._eth(_Fet(), {})
        assert "disagree" in str(e.value)
    finally:
        _st._text = real


def test_a_meta_manifest_that_leaves_its_prefix_is_a_refusal():
    good = ("TsvHttpData-1.0\n"
            f"{cr.FB_BUCKET}/{cr.META_V2_PREFIX}/chm/0013113321.tif\n")
    assert cr.manifest_quadkeys(good, cr.META_V2_PREFIX) == ["0013113321"]
    with pytest.raises(st.Refusal) as e:
        cr.manifest_quadkeys(
            good + "https://elsewhere.invalid/chm/0013113323.tif\n",
            cr.META_V2_PREFIX)
    assert "outside the declared prefix" in str(e.value)
    with pytest.raises(st.Refusal):
        cr.manifest_quadkeys("not a header\nx\n", cr.META_V2_PREFIX)
    with pytest.raises(st.Refusal) as e:
        cr.manifest_quadkeys("TsvHttpData-1.0\n", cr.META_V2_PREFIX)
    assert "no tile at all" in str(e.value)


# ================================================================ cat_palsar =
def test_palsar_pages_the_bucket_and_filters_by_the_name_before_fetching(
        smokes):
    c = smokes["cat_palsar"]["probe"]["counts"]
    # 6 scenes x (one .json + one .tif) = 12 keys, paged 3 at a time
    assert c["s3_keys"] == 12 and c["s3_pages"] == 4
    assert c["s3_items_listed"] == 6
    # one object's name carries no date token, so it is fetched anyway and
    # its own datetime decides — never silently dropped
    assert c["s3_items_without_date_token"] == 1
    assert c["s3_items_in_window"] == 5
    assert c["rows_kept"] == 5
    assert c["pol_HHHV"] == 5
    assert smokes["cat_palsar"]["probe"]["qc"] == {"1": 5}


def test_the_scansar_date_token_is_read_from_the_objects_own_name():
    import datetime as dt
    assert cp.scene_date("ALOS2010632250-140804_WBDR2.2GUD.json") == \
        dt.date(2014, 8, 4)
    assert cp.scene_date("x/y/ALOS2011000000-260101_WBDR2.2GUD_HH_SLP.tif") \
        == dt.date(2026, 1, 1)
    # no token, and an impossible date, both answer None rather than guessing
    assert cp.scene_date("ALOS2010999999_WBDR2.2GUD.json") is None
    assert cp.scene_date("ALOS2010632250-149999_WBDR2.2GUD.json") is None


def test_the_mosaic_refuses_by_name_without_the_jaxa_account(monkeypatch):
    """The half that is not anonymous says WHY, counted, and never returns an
    empty window that would read as 'no tiles that year'."""
    import datetime as dt
    for k in cp.MOSAIC_ENV + (cp.MOSAIC_DATASET_ENV,):
        monkeypatch.delenv(k, raising=False)
    ad = cp.PalsarCatalogue()
    t0, t1 = dt.datetime(2020, 12, 1), dt.datetime(2021, 1, 1)

    rows, c = ad.mosaic_scenes(None, t0, t1)
    assert rows == []
    assert c["mosaic_refused_no_credential"] == 1
    assert set(c["mosaic_missing_env"]) == set(cp.MOSAIC_ENV)

    for k in cp.MOSAIC_ENV:
        monkeypatch.setenv(k, "x")
    rows, c = ad.mosaic_scenes(None, t0, t1)
    assert rows == [] and c["mosaic_refused_no_dataset_id"] == 1

    monkeypatch.setenv(cp.MOSAIC_DATASET_ENV, "10102000")
    rows, c = ad.mosaic_scenes(None, t0, t1)
    assert rows == [] and c["mosaic_refused_listing_unmeasured"] == 1
    # and the three refusals are distinct counters, so a build report can
    # tell "no account" from "no dataset id" from "no parser yet"
    assert len({"mosaic_refused_no_credential",
                "mosaic_refused_no_dataset_id",
                "mosaic_refused_listing_unmeasured"}) == 3


def test_the_mosaic_design_row_survives_in_the_plan(smokes):
    """The registry needs the mosaic's design row even though the collection
    lists nothing: the years, the tile grid and what would unblock it."""
    plan = json.load(open(os.path.join(smokes["cat_palsar"]["work"],
                                       "cat_palsar", "plan.json")))
    m = plan["collections"]["mosaic"]
    assert m["version"] == "2.6.0"
    assert m["years"][:4] == [2007, 2008, 2009, 2010]
    assert 2015 in m["years"] and 2025 in m["years"]
    assert 2011 not in m["years"] and 2014 not in m["years"]
    assert m["tile_degrees"] == 1.0
    assert m["anonymous"] is False
    assert m["data_directory_status"] == 401
    assert m["credentials_env"] == ["JAXA_USER", "JAXA_PASSWORD"]
    assert "G-Portal" in m["unblocked_by"]
    s = plan["collections"]["scansar"]
    assert s["anonymous"] is True and s["stac_items"] == 6
    assert s["items_without_date_token"] == 1


def test_palsar_declares_the_restricted_derived_works():
    ad = cp.PalsarCatalogue()
    assert ad.licence["derived_works"] == "restricted"
    assert ad.licence["redistribution"] == "attribution"
    assert "(C) JAXA" in ad.licence["attribution"]


# =========================================================== cat_biomass_esa =
def test_biomass_lists_through_fedeo_and_records_the_dead_endpoints(smokes):
    plan = json.load(open(os.path.join(smokes["cat_biomass_esa"]["work"],
                                       "cat_biomass_esa", "plan.json")))
    probes = plan["access_probes"]
    assert len(probes) == 4
    usable = [p for p in probes if p["usable"]]
    assert len(usable) == 1 and "fedeo.ceos.org" in usable[0]["endpoint"]
    dead = {p["endpoint"]: p["answer"] for p in probes if not p["usable"]}
    assert any("eocat.esa.int" in k for k in dead)
    assert any("biomass-pdgs" in k for k in dead)
    assert any("dataspace.copernicus.eu" in k for k in dead)
    assert plan["launch"] == "2025-04-29"
    # every collection is asked for, including the two that matched zero
    assert len(plan["collections"]) == len(cb.COLLECTIONS)
    assert plan["collections"]["BiomassLevel2b"]["products"] == 0


def test_biomass_drops_a_product_that_starts_before_the_window(smokes):
    c = smokes["cat_biomass_esa"]["probe"]["counts"]
    assert c["products_BiomassLevel1a"] == 4
    assert c["products_BiomassLevel1b"] == 2
    assert c["products_BiomassLevel2a"] == 1
    # FedEO matches on OVERLAP, so a window serves a product that started
    # earlier; it is dropped by its own start, exactly as cat_hls does
    assert c["rows_outside_window"] == 1
    # a product with no `enclosure` link has no URL to catalogue at all
    assert c["items_without_enclosure"] == 1
    assert c["rows_kept"] == 6
    # the pager turned: 8 products over collections at three a page
    assert c["fedeo_pages"] >= 10
    assert smokes["cat_biomass_esa"]["probe"]["qc"] == {"1": 4, "2": 1,
                                                        "4": 1}


def test_the_zipper_service_is_not_an_asset(smokes):
    """The `zipper` href re-packs a product and lives in another directory —
    the same trap cat_viirs hit with OPeNDAP. A row points at the FILES."""
    import pyarrow.parquet as pq
    t = pq.read_table(os.path.join(work_of(smokes, "cat_biomass_esa"),
                                   st.ASSETS_PARQUET))
    for base, tmpl in zip(t.column("base_url").to_pylist(),
                          t.column("asset_set").to_pylist()):
        assert "/zipper/" not in base
        assert all(not n.endswith(".zip") for n in tmpl.split())


def test_a_short_fedeo_listing_is_a_refusal_not_a_small_window():
    """The producer's own `numberMatched` is the only thing that can tell
    'nothing there' from 'the pager stopped early'."""
    class _Fet:
        def __init__(self, pages):
            self.pages = list(pages)
            self.n = 0

        def get(self, url):
            d = self.pages[min(self.n, len(self.pages) - 1)]
            self.n += 1
            return d, {}

    ok = [{"features": [{"id": "a"}], "numberMatched": 1,
           "numberReturned": 1, "links": []}]
    assert len(cb.fedeo_items(_Fet(ok), "C", [], {}, "t")) == 1

    short = [{"features": [{"id": "a"}], "numberMatched": 9,
              "numberReturned": 1, "links": []}]
    with pytest.raises(st.Refusal) as e:
        cb.fedeo_items(_Fet(short), "C", [], {}, "t")
    assert "truncated listing" in str(e.value)

    # a cursor that returns to a page it already served is a loop, not data
    same = "https://fedeo.ceos.org/collections/C/items?marker=1"
    loop = [{"features": [{"id": "a"}], "numberMatched": 99,
             "numberReturned": 1,
             "links": [{"rel": "next", "href": same}]}]
    with pytest.raises(st.Refusal) as e:
        cb.fedeo_items(_Fet(loop), "C", [], {}, "t")
    assert "already served" in str(e.value)


def test_the_biomass_licence_is_not_claimed_as_confirmed():
    """The ESA terms document was not read here, so the store says so and a
    public publish needs the flag rather than a silent claim."""
    ad = cb.BiomassCatalogue()
    assert ad.licence["redistribution_confirmed"] is False
    assert "allow-unconfirmed-licence" in ad.licence["terms"]

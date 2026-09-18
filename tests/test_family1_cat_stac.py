#!/usr/bin/env python3
"""`ml/family1/adapters/_stac.py` — the scene-catalogue machinery. No network.

    python3 -m pytest -q tests/test_family1_cat_stac.py

The eight tier-T catalogue stores of family 1.0.tf (E-082 wave 3) share one
helper: a counted JSON client that can read CANNED responses instead of a
socket, four listing walkers (STAC search, CDSE OData, CMR granules, ASF
param search) that each refuse a truncated page or a count that does not match
the producer's own, spherical footprint geometry, and the `assets.parquet`
sidecar. This file pins the parts a store's own smoke cannot reach: the
refusals, the seam-free geometry, the float16 area decision, the canned-key
rule and the sidecar's round trip.
"""
import json
import math
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
from family1.adapters import _stac as st                        # noqa: E402

CAT_STORES = ("cat_ecostress", "cat_hls", "cat_landsat", "cat_nisar",
              "cat_olci", "cat_s1", "cat_s2", "cat_viirs")


# ================================================================ registry ==
def test_all_eight_are_registered_and_agree_with_the_contract():
    for s in CAT_STORES:
        assert s in fam.REGISTRY, s
        ad = fam.REGISTRY[s]()
        assert ad.family == "1tf"
        assert ad.distribution == "public"
        assert ad.time_dtype == "int32"          # every record starts ≥ 2012
        assert ad.platform_meta is False
        assert ad.per_year is True
        assert ad.credentials == ()
        assert ad.licence["redistribution"] == "yes"
        assert ad.licence["derived_works"] == "free"
        # the licence NAMES the underlying imagery's terms, which is the one
        # thing a catalogue of someone else's pixels has to say
        assert "catalogue rows" in ad.licence["name"]
        assert ad.fetch_month_scope == "month"
        assert ad.qc_policy and ad.verified and ad.notes
        assert ad.smoke_window and ad.smoke_probe_month


def test_the_five_channels_are_the_notes_five_in_the_notes_order():
    names = [c[0] for c in st.CAT_CHANNELS]
    assert names == ["cloud", "valid", "angle", "log2_area", "sensor"]
    for s in CAT_STORES:
        ad = fam.REGISTRY[s]()
        assert ad.channel_names[:5] == names, s
        # cat_olci alone adds a sixth
        assert ad.C == (6 if s == "cat_olci" else 5), s
    assert fam.REGISTRY["cat_olci"]().channel_names[5] == "coastal"


def test_the_footprint_and_time_scales_are_the_notes_log2_units():
    # the note: "a 110 km Sentinel-2 tile is +2.0 in log2 units of 27.83 km"
    assert abs(st.log2_fp_for_km(110.0) - 2.0) < 0.02
    ad = fam.REGISTRY["cat_s2"]()
    assert abs(ad.log2_fp - 2.0) < 0.02
    assert abs(fam.REGISTRY["cat_landsat"]().log2_fp
               - math.log2(185.0 / 27.83)) < 1e-12
    # log2_dt is the share of a 5-day bin one scene occupies
    assert abs(st.log2_dt_for_seconds(86400 * 5) - 0.0) < 1e-12
    assert abs(fam.REGISTRY["cat_olci"]().log2_dt
               - math.log2((180.0 / 86400.0) / 5.0)) < 1e-12


def test_every_sensor_and_qc_code_fits_a_uint8_and_is_unique():
    for s in CAT_STORES:
        ad = fam.REGISTRY[s]()
        for table, what in ((ad.SENSOR_TABLE, "sensor"),
                            (ad.QC_TABLE, "qc")):
            assert table, f"{s}: empty {what} table"
            assert all(0 <= int(k) <= 255 for k in table), (s, what)
            assert st.QC_OTHER not in table or what != "qc", (s, what)
            assert len(set(table.values())) == len(table), (s, what)
        assert st.SENSOR_OTHER not in ad.SENSOR_TABLE, s


# ================================================================ geometry ==
def test_a_footprints_centre_and_area_are_computed_on_the_sphere():
    # a 1° x 1° box on the equator: 111.195 km a side, 12,364 km²
    lat, lon, area = st.sphere_centre_area([-0.5, -0.5, 0.5, 0.5],
                                           [-0.5, 0.5, 0.5, -0.5])
    assert abs(lat) < 1e-9 and abs(lon) < 1e-9
    assert abs(area - 12364.0) / 12364.0 < 0.01
    # the SAME box at 60 N is half the area (cos 60 = 0.5)
    _, _, a60 = st.sphere_centre_area([59.5, 59.5, 60.5, 60.5],
                                      [-0.5, 0.5, 0.5, -0.5])
    assert abs(a60 / area - 0.5) < 0.01


def test_a_footprint_that_encloses_a_pole_is_not_the_whole_planet():
    """The measured reason for the pole correction in `sphere_centre_area`.

    A ring around the north pole divides the sphere into a small cap and
    everything else; the spherical-excess formula returns whichever the
    winding names, and 6 % of VIIRS's six-minute granules came back at the
    area of the Earth before the correction.
    """
    lats = [80.0] * 36
    lons = [(-180.0 + 10.0 * i) for i in range(36)]
    _, _, area = st.sphere_centre_area(lats, lons)
    whole = 4.0 * math.pi * st.EARTH_R_KM ** 2
    # the true cap above 80 N is 2πR²(1 − sin 80°) ≈ 3.88e6 km²
    cap = 2 * math.pi * st.EARTH_R_KM ** 2 * (1 - math.sin(math.radians(80.0)))
    assert abs(area - cap) / cap < 0.01, (area, cap)
    assert area < whole / 2
    # and the same ring wound the other way gives the same answer
    _, _, rev = st.sphere_centre_area(lats[::-1], lons[::-1])
    assert abs(rev - area) / area < 1e-9


def test_the_centre_has_no_seam_at_the_antimeridian():
    """The reason the centre is a 3-vector mean and not a mean of degrees."""
    lat, lon, _ = st.sphere_centre_area([0.0, 0.0, 1.0, 1.0],
                                        [179.0, -179.0, -179.0, 179.0])
    assert abs(lat - 0.5) < 0.01
    assert min(abs(lon - 180.0), abs(lon + 180.0)) < 0.01
    # averaging the longitudes instead would have put it at 0 E, on the
    # opposite side of the planet
    assert abs(lon) > 170.0


def test_a_ring_is_read_from_all_four_producer_shapes():
    wkt = st.ring_from_wkt("POLYGON ((10 20, 11 20, 11 21, 10 21, 10 20))")
    assert len(wkt) == 1 and wkt[0][0][:2] == [20.0, 20.0]
    gj = st.ring_from_geojson({"type": "Polygon", "coordinates": [
        [[10, 20], [11, 20], [11, 21], [10, 21], [10, 20]]]})
    assert st.centre_area(gj) == pytest.approx(st.centre_area(wkt), rel=1e-9)
    multi = st.ring_from_geojson({"type": "MultiPolygon", "coordinates": [
        [[[179, 0], [180, 0], [180, 1], [179, 1], [179, 0]]],
        [[[-180, 0], [-179, 0], [-179, 1], [-180, 1], [-180, 0]]]]})
    assert len(multi) == 2
    la, lo, ar = st.centre_area(multi)
    one = st.centre_area(st.ring_from_geojson({
        "type": "Polygon",
        "coordinates": [[[179, 0], [-179, 0], [-179, 1], [179, 1],
                         [179, 0]]]}))
    assert abs(ar - one[2]) / one[2] < 0.01       # the two halves add up
    cmr = st.ring_from_cmr({"polygons": [["20 10 20 11 21 11 21 10 20 10"]]})
    assert st.centre_area(cmr) == pytest.approx(st.centre_area(wkt), rel=1e-9)
    box = st.ring_from_cmr({"boxes": ["20 10 21 11"]})
    assert st.centre_area(box) == pytest.approx(st.centre_area(wkt), rel=1e-9)
    with pytest.raises(st.Refusal):
        st.ring_from_cmr({"points": ["20 10"]})


def test_a_degenerate_footprint_is_a_refusal_not_a_zero():
    with pytest.raises(st.Refusal):
        st.sphere_centre_area([0.0, 1.0], [0.0, 1.0])
    with pytest.raises(st.Refusal):
        st.sphere_centre_area([0.0, 1.0, float("nan")], [0.0, 1.0, 2.0])
    with pytest.raises(st.Refusal):
        st.ring_from_wkt("MULTIPOINT (1 2)")


# ==================================================== the float16 area rule ==
def test_the_area_channel_is_log2_because_float16_cannot_hold_km2():
    """The measured reason the note's `area` is stored as log2(km²).

    `values` is float16 and its largest finite number is 65,504. A VIIRS
    six-minute granule is about 7e6 km² and a Sentinel-3 OLCI granule 1.5e6,
    so both become `inf` on the way into the store — which is exactly how
    cat_olci's and cat_s1's first smokes failed (`values.npy holds an
    infinity`).
    """
    with np.errstate(over="ignore"):
        assert np.float16(7.0e6) == np.inf
        assert np.float16(1.5e6) == np.inf
    counts = {}
    for km2 in (12364.0, 33300.0, 1.5e6, 7.0e6, 3.0e7):
        v = st.area_channel(km2, counts)
        assert np.isfinite(np.float16(v)), km2
        # the round trip through float16: 0.007 % for a Sentinel-2 tile,
        # 0.31 % for a VIIRS granule, never worse than 1.2 % (ten-bit
        # mantissa, so the error grows with log2 of the area)
        back = 2.0 ** float(np.float16(v))
        assert abs(back - km2) / km2 < 1.2e-2, km2
        lo, hi = st.CAT_CHANNELS[3][2], st.CAT_CHANNELS[3][3]
        assert lo <= v <= hi, km2
    assert counts == {}
    for bad in (0.0, -1.0, float("nan"), float("inf"), "x", None):
        assert math.isnan(st.area_channel(bad, counts))
    assert counts["area_not_positive"] == 6


def test_a_percentage_is_never_clipped_only_rounded_at_the_edge():
    assert st.clamp_pct(100.000001) == 100.0      # three producers publish it
    assert st.clamp_pct(140.0) == 140.0           # left for mask_bounds to NaN
    assert math.isnan(st.clamp_pct(-1.0))         # "not computed"
    assert math.isnan(st.clamp_pct(None))
    assert math.isnan(st.clamp_pct(""))
    assert math.isnan(st.clamp_pct("abc"))
    assert st.clamp_pct("18") == 18.0


# ================================================================== the time ==
def test_every_producers_instant_shape_parses_and_anything_else_refuses():
    for s in ("2024-06-01T00:06:19.072Z", "2024-06-01T00:06:19Z",
              "2024-06-01T00:06:19.072461Z", "2024-06-01 00:06:19",
              "2024-06-01T00:06:19+00:00", "2024-06-01T00:06:19.000000Z"):
        assert st.parse_iso(s).minute == 6
    assert st.seconds_of("1982-01-01T00:00:00Z") == 0
    for bad in ("2024-06-01", "yesterday", "2024-06-01T00:06:19+02:00", ""):
        with pytest.raises(st.Refusal):
            st.parse_iso(bad)


# ============================================================== the walkers ==
class _Fake(st.Fetcher):
    """A `Fetcher` whose answers come from a list, for the refusal tests."""

    def __init__(self, answers):
        class _Ctx:
            source_dir = ""
            a = None

            def count_bytes(self, n):
                pass
        super().__init__(_Ctx(), "test", attempts=1)
        self.answers = list(answers)

    def _do(self, method, url, body, headers):
        self.requests += 1
        return self.answers.pop(0)


def _page(feats, matched, nxt=None, limit=3):
    links = ([{"rel": "next", "method": "POST", "href": "u",
               "body": {"next": nxt}}] if nxt is not None else [])
    return ({"features": feats, "numberMatched": matched,
             "numberReturned": len(feats), "links": links}, {})


def test_a_stac_search_walks_pages_and_checks_the_producers_own_count():
    f = _Fake([_page([1, 2, 3], 5, nxt=3), _page([4, 5], 5)])
    got = list(st.stac_search(f, "u", {}, 3, {}, "w"))
    assert got == [1, 2, 3, 4, 5]


def test_a_short_last_page_that_still_offers_a_cursor_is_the_end():
    """Measured on landsatlook: the last page is short AND has a next link."""
    f = _Fake([_page([1, 2, 3], 5, nxt=3), _page([4, 5], 5, nxt=5),
               _page([], 5)])
    assert list(st.stac_search(f, "u", {}, 3, {}, "w")) == [1, 2, 3, 4, 5]


def test_a_short_page_followed_by_more_items_is_a_refusal():
    f = _Fake([_page([1, 2], 5, nxt=2), _page([3, 4, 5], 5)])
    with pytest.raises(st.Refusal) as e:
        list(st.stac_search(f, "u", {}, 3, {}, "w"))
    assert "dropped items in the middle" in str(e.value)


def test_a_count_mismatch_and_a_moving_count_are_both_refusals():
    f = _Fake([_page([1, 2, 3], 9, nxt=3), _page([4], 9)])
    with pytest.raises(st.Refusal) as e:
        list(st.stac_search(f, "u", {}, 3, {}, "w"))
    assert "numberMatched says 9" in str(e.value)
    f = _Fake([_page([1, 2, 3], 5, nxt=3), _page([4, 5], 6)])
    with pytest.raises(st.Refusal) as e:
        list(st.stac_search(f, "u", {}, 3, {}, "w"))
    assert "changed from 5 to 6" in str(e.value)
    f = _Fake([({"features": [1], "links": []}, {})])
    with pytest.raises(st.Refusal) as e:
        list(st.stac_search(f, "u", {}, 3, {}, "w"))
    assert "never reported numberMatched" in str(e.value)
    f = _Fake([({"type": "x"}, {})])
    with pytest.raises(st.Refusal) as e:
        list(st.stac_search(f, "u", {}, 3, {}, "w"))
    assert "without a `features` key" in str(e.value)


def test_odata_refuses_a_window_beyond_its_own_paging_reach():
    """The measured CDSE caps: $top ≤ 1000 and $skip ≤ 10000."""
    assert st.ODATA_TOP == 1000 and st.ODATA_SKIP_MAX == 10000
    assert st.ODATA_REACH == 11000
    f = _Fake([({"@odata.count": 11001, "value": [1]}, {})])
    with pytest.raises(st.Refusal) as e:
        list(st.odata_window(f, "u", "flt", "Attributes", {}, "w"))
    assert "split the window" in str(e.value)


def test_odata_refuses_an_empty_page_and_a_short_one_inside_a_window():
    # an EMPTY page while the server's own count says there are rows
    f = _Fake([({"@odata.count": 5, "value": [1]}, {}), ({"value": []}, {})])
    with pytest.raises(st.Refusal) as e:
        list(st.odata_window(f, "u", "flt", None, {}, "w"))
    assert "EMPTY" in str(e.value)
    # a page SHORT of what was asked for with rows still to come
    f = _Fake([({"@odata.count": 5, "value": [1]}, {}),
               ({"value": [1, 2, 3]}, {})])
    with pytest.raises(st.Refusal) as e:
        list(st.odata_window(f, "u", "flt", None, {}, "w"))
    assert "truncated page" in str(e.value)


def test_odata_windows_halves_until_a_window_fits():
    import datetime as dt
    counts = {}
    # the first count is over the reach, the two halves are not
    f = _Fake([({"@odata.count": 20000, "value": []}, {}),
               ({"@odata.count": 9000, "value": []}, {}),
               ({"@odata.count": 9000, "value": []}, {})])
    out = list(st.odata_windows(f, "u",
                                lambda a, b: "flt", dt.datetime(2024, 6, 1),
                                dt.datetime(2024, 6, 2), counts, "w"))
    assert [n for _, _, n in out] == [9000, 9000]
    assert counts["window_splits"] == 1


def test_cmr_pages_on_the_search_after_header_and_checks_cmr_hits():
    f = _Fake([({"items": [1, 2, 3]}, {"CMR-Hits": "5",
                                       "CMR-Search-After": "c1"}),
               ({"items": [4, 5]}, {"CMR-Hits": "5"})])
    assert list(st.cmr_granules(f, "u", [], {}, "w", page_size=3)) == \
        [1, 2, 3, 4, 5]
    # the older `.json` feed shape is read too
    f = _Fake([({"feed": {"entry": [1]}}, {"CMR-Hits": "1"})])
    assert list(st.cmr_granules(f, "u", [], {}, "w", page_size=3)) == [1]
    f = _Fake([({"items": [1]}, {})])
    with pytest.raises(st.Refusal) as e:
        list(st.cmr_granules(f, "u", [], {}, "w", page_size=3))
    assert "CMR-Hits" in str(e.value)


def test_cmr_hits_may_over_count_by_a_little_and_never_under_count():
    """The measured slack: CMR's own header was one high on a real window."""
    assert st.CMR_HITS_SLACK == 8
    c = {}
    f = _Fake([({"items": [1, 2]}, {"CMR-Hits": "3"})])
    assert list(st.cmr_granules(f, "u", [], c, "w", page_size=3)) == [1, 2]
    assert c["producer_count_minus_walked"] == 1
    assert c["windows_short_of_cmr_hits"] == 1
    # short by more than the slack is still a refusal
    f = _Fake([({"items": [1, 2]}, {"CMR-Hits": "99"})])
    with pytest.raises(st.Refusal) as e:
        list(st.cmr_granules(f, "u", [], {}, "w", page_size=3))
    assert "short by 97" in str(e.value)
    # and a walk that returns MORE than the producer counts is always one
    f = _Fake([({"items": [1, 2]}, {"CMR-Hits": "1"})])
    with pytest.raises(st.Refusal) as e:
        list(st.cmr_granules(f, "u", [], {}, "w", page_size=3))
    assert "does not count" in str(e.value)


def test_asf_refuses_a_window_larger_than_its_250_result_cap():
    assert st.ASF_MAX == 250
    f = _Fake([])
    f.__dict__["_text"] = None
    import family1.adapters._stac as mod
    real = mod._text
    mod._text = lambda fet, url: "400"
    try:
        with pytest.raises(st.Refusal) as e:
            st.asf_window(f, "u", [], {}, "w")
        assert "split the window" in str(e.value)
        mod._text = lambda fet, url: "not-a-number"
        with pytest.raises(st.Refusal) as e:
            st.asf_count(f, "u", [])
        assert "not an" in str(e.value)
    finally:
        mod._text = real


def test_a_429_slows_the_host_down_and_does_not_spend_an_attempt():
    """The measured shape of a rate limit, on both producers that have one."""
    import family1.adapters._stac as mod
    host = "rate.example.test"
    mod._LIMITERS.pop(host, None)
    assert mod.limiter_for(f"https://{host}/x") is None
    h, r1 = mod.slow_host(f"https://{host}/x")
    assert h == host and r1 == mod.ADAPTIVE_START
    lim = mod.limiter_for(f"https://{host}/x")
    assert lim is not None
    _, r2 = mod.slow_host(f"https://{host}/x")
    assert r2 < r1                                # each 429 halves the rate
    for _ in range(40):
        mod.slow_host(f"https://{host}/x")
    _, rn = mod.slow_host(f"https://{host}/x")
    assert rn >= mod.ADAPTIVE_FLOOR - 1e-9        # and never below the floor
    mod._LIMITERS.pop(host, None)
    # ASF's is declared rather than learned
    assert mod.RATE_LIMITS["api.daac.asf.alaska.edu"] == 240
    assert mod.limiter_for("https://api.daac.asf.alaska.edu/x") is not None
    assert mod.THROTTLE_TRIES >= 8


# ============================================================== the client ==
def test_the_canned_key_includes_the_headers_because_cmr_pages_in_one():
    a = st.canned_key("GET", "u", None, None)
    b = st.canned_key("GET", "u", None, {"CMR-Search-After": "c1"})
    c = st.canned_key("GET", "u", None, {"CMR-Search-After": "c2"})
    assert len({a, b, c}) == 3
    assert st.canned_key("POST", "u", {"a": 1, "b": 2}) == \
        st.canned_key("POST", "u", {"b": 2, "a": 1})


def test_a_missing_canned_response_is_a_refusal_that_names_the_request(
        tmp_path):
    class _Ctx:
        source_dir = str(tmp_path)
        a = None

        def count_bytes(self, n):
            pass
    f = st.Fetcher(_Ctx(), "cat_x", attempts=1)
    with pytest.raises(st.Refusal) as e:
        f.get("https://example/q")
    assert "no canned response" in str(e.value)
    assert "https://example/q" in str(e.value)


def test_the_page_size_is_small_against_a_canned_archive():
    """So a smoke actually turns a page instead of fitting in one response."""
    ad = fam.REGISTRY["cat_hls"]()

    class _Ctx:
        source_dir = ""
    assert ad.page_for(_Ctx(), 2000) == 2000
    _Ctx.source_dir = "/somewhere"
    assert ad.page_for(_Ctx(), 2000) == ad.SMOKE_PAGE == 3
    assert ad.workers(_Ctx()) == 1


# ============================================================== the sidecar ==
def test_the_asset_template_is_the_producers_own_urls_re_encoded():
    base, tmpl = st.asset_template(
        ["https://h/d/SCENE_B1.TIF", "https://h/d/SCENE_B2.TIF",
         "https://h/d/SCENE_MTL.txt"], "SCENE", "w")
    assert base == "https://h/d"
    assert tmpl == "{id}_B1.TIF {id}_B2.TIF {id}_MTL.txt"
    # a url a consumer rebuilds from the template is the url we were given
    got = sorted(base + "/" + n.replace("{id}", "SCENE")
                 for n in tmpl.split())
    assert got == ["https://h/d/SCENE_B1.TIF", "https://h/d/SCENE_B2.TIF",
                   "https://h/d/SCENE_MTL.txt"]
    # a host we were told to drop (Landsat's `index` points at a web page)
    base, tmpl = st.asset_template(
        ["https://h/d/S_B1.TIF", "https://browser/x/S"], "S", "w",
        drop_hosts=("https://browser/",))
    assert base == "https://h/d" and tmpl == "{id}_B1.TIF"
    with pytest.raises(st.Refusal) as e:
        st.asset_template(["https://a/1.tif", "https://b/2.tif"], "x", "w")
    assert "spread over 2 directories" in str(e.value)
    with pytest.raises(st.Refusal):
        st.asset_template([], "x", "w")


def test_a_cmr_granule_with_no_download_links_is_a_row_with_no_asset():
    """Measured: HLS.L30.T34NGH.2026002T083747.v2.0 has no `RelatedUrls`.

    Every attribute, a footprint and a time, and no download links in CMR at
    all. The scene EXISTS, which is the whole claim a catalogue row makes, so
    the row is kept with an empty asset entry and the case is counted —
    refusing would have lost a whole year lane for one granule.
    """
    ad = fam.REGISTRY["cat_hls"]()
    item = {"umm": {
        "GranuleUR": "HLS.L30.TX.2026002T083747.v2.0",
        "TemporalExtent": {"RangeDateTime": {
            "BeginningDateTime": "2026-01-02T08:37:47.964Z"}},
        "AdditionalAttributes": [
            {"Name": "CLOUD_COVERAGE", "Values": ["33"]},
            {"Name": "SPATIAL_COVERAGE", "Values": ["69"]},
            {"Name": "MEAN_SUN_ZENITH_ANGLE", "Values": ["37.1"]}],
        "Platforms": [{"ShortName": "LANDSAT-8",
                       "Instruments": [{"ShortName": "OLI"}]}],
        "SpatialExtent": {"HorizontalSpatialDomain": {"Geometry": {
            "GPolygons": [{"Boundary": {"Points": [
                {"Latitude": 1, "Longitude": 2},
                {"Latitude": 1, "Longitude": 3},
                {"Latitude": 2, "Longitude": 3},
                {"Latitude": 2, "Longitude": 2}]}}]}}}}}
    counts = {}
    sc = ad.granule(item, ("L30", "HLSL30", "2.0"), counts, "t")
    assert sc.base_url == "" and sc.asset_set == ""
    assert counts["granules_without_asset_url"] == 1
    assert sc.values[0] == 33.0 and sc.values[1] == 69.0


def test_the_sidecar_round_trips_through_a_npy_and_into_one_parquet(tmp_path):
    pa = pytest.importorskip("pyarrow")
    import pyarrow.parquet as pq

    class _Ctx:
        years = [2024, 2025]

        def __init__(self, root):
            self.root = root
            self.adapter = fam.REGISTRY["cat_landsat"]()

        def year_dir(self, y):
            return os.path.join(self.root, str(y))
    ctx = _Ctx(str(tmp_path / "parts"))
    want = []
    for y in ctx.years:
        w = st.AssetWriter(ctx, y, flush_rows=2)
        for i in range(5):
            sid = f"S{y}_{i}"
            h = b10.platform_hash(sid)
            assert w.add(h, sid, "c", f"https://h/{y}/{i}", "{id}_B1.TIF")
            want.append((h, sid))
            # the same scene again is ONE row
            assert not w.add(h, sid, "c", f"https://h/{y}/{i}",
                             "{id}_B1.TIF")
        assert w.close() == 5
        assert w.seq == 3                      # flush_rows=2 over five rows
        names = st.asset_part_names(ctx.year_dir(y))
        assert names == ["assets-00000.npy", "assets-00001.npy",
                         "assets-00002.npy"]
    dest = str(tmp_path / "store")
    os.makedirs(dest)
    path, meta = st.write_assets_parquet(ctx, dest, expect_rows=10)
    assert meta["rows"] == 10 and meta["parts"] == 6
    t = pq.read_table(path)
    assert list(t.column_names) == list(st.ASSET_COLS)
    got = list(zip(t.column("platform").to_pylist(),
                   t.column("stac_id").to_pylist()))
    assert sorted(got) == sorted(want)
    with pytest.raises(st.Refusal) as e:
        st.write_assets_parquet(ctx, dest, expect_rows=11)
    assert "every catalogue row must have exactly one asset entry" in \
        str(e.value)


def test_an_absent_sidecar_and_a_repeated_hash_are_both_refusals(tmp_path):
    pytest.importorskip("pyarrow")

    class _Ctx:
        years = [2024]

        def __init__(self, root):
            self.root = root
            self.adapter = fam.REGISTRY["cat_landsat"]()

        def year_dir(self, y):
            return os.path.join(self.root, str(y))
    ctx = _Ctx(str(tmp_path / "empty"))
    os.makedirs(ctx.year_dir(2024))
    with pytest.raises(st.Refusal) as e:
        st.write_assets_parquet(ctx, str(tmp_path))
    assert "no assets-" in str(e.value)
    # the cross-year check: the same hash written under two years
    ctx2 = _Ctx(str(tmp_path / "dup"))
    ctx2.years = [2024, 2025]
    for y in ctx2.years:
        w = st.AssetWriter(ctx2, y)
        w.add(b10.platform_hash("same"), "same", "c", "https://h", "t")
        w.close()
    dest = str(tmp_path / "d2")
    os.makedirs(dest)
    with pytest.raises(st.Refusal) as e:
        st.write_assets_parquet(ctx2, dest)
    assert "repeated platform hash" in str(e.value)


# ======================================================= the coastal static ==
def test_the_coastal_band_comes_from_the_repositorys_own_family7_static():
    from family1.adapters import cat_olci
    assert os.path.exists(cat_olci.SPHERE_JSON), cat_olci.SPHERE_JSON
    mask, meta = cat_olci.coastal_mask_1deg()
    assert mask is not None and mask.shape == (180, 360)
    assert meta["present"] and meta["nx"] == 1440 and meta["ny"] == 721
    # a coastal band over a water planet would be empty and over a land one
    # would be everything; the real one is a few per cent of the globe
    frac = mask.mean()
    assert 0.05 < frac < 0.45, frac
    # THE ORIENTATION, PINNED. The static is south-first and reading it the
    # other way round produced a plausible-looking band with Portugal, the
    # Sahara and Siberia in the ocean.
    def cell(lat, lon):
        return mask[int(lat + 90), int(lon + 180)]
    assert cell(39.5, -8.5)            # the Portuguese coast
    assert cell(51.0, 3.0)             # the Dutch/Belgian coast
    assert not cell(-25.0, -140.0)     # mid-Pacific
    assert not cell(23.0, 15.0)        # the middle of the Sahara
    assert not cell(-80.0, 40.0)       # the middle of Antarctica
    # the fraction of a footprint box, and the NaN when the static is absent
    assert cat_olci.coastal_fraction(mask, [38, 38, 41, 41],
                                     [-10, -8, -8, -10]) > 0.4
    assert cat_olci.coastal_fraction(mask, [-30, -30, -20, -20],
                                     [-145, -135, -135, -145]) == 0.0
    assert math.isnan(cat_olci.coastal_fraction(None, [0], [0]))
    m2, meta2 = cat_olci.coastal_mask_1deg(path="/nonexistent.json")
    assert m2 is None and meta2["present"] is False


def test_the_olci_overpass_key_ignores_the_publication_fields():
    from family1.adapters import cat_olci
    nt = ("S3B_OL_2_WFR____20240601T040257_20240601T040557_20240602T103132_"
          "0180_093_318_1800_MAR_O_NT_003.SEN3")
    nr = ("S3B_OL_2_WFR____20240601T040257_20240601T040557_20240601T050000_"
          "0180_093_318_1800_MAR_O_NR_003.SEN3")
    assert cat_olci.overpass_key(nt) == cat_olci.overpass_key(nr)
    assert cat_olci.overpass_key(nt) == ("S3B", "20240601T040257",
                                         "20240601T040557")
    with pytest.raises(st.Refusal):
        cat_olci.overpass_key("nonsense")


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))

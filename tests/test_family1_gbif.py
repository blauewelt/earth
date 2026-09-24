#!/usr/bin/env python3
"""The `gbif` / `gbif_nc` adapters' smoke (family 1.0.tf, E-082 wave 6) — no
network.

    python3 -m pytest -q tests/test_family1_gbif.py

The synthetic snapshot (`_gbif.make_smoke_sources`) is written in the AWS
bucket's real shape: a `listing.json` holding what ListObjectsV2 returns, a
`citation.txt`, and three real parquet parts under
`occurrence/<snapshot>/occurrence.parquet/` with the snapshot's own column
names and types. Part 000002 holds January 1900, part 000001 December 1899
and part 000003 June 1900, which is what makes the probe's "one part, and the
month argument is ignored" rule visible: the fetch reads all three and the
probe reads only the middle one.

Expected counts are exact and are derived in the comments beside them; the
records themselves are written by hand in `_gbif.smoke_parts`, and the row
each one must become is computed by `_gbif._expect`, a second implementation
of the rules that shares no code with the adapter's parser.
"""
import argparse
import datetime as dt
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
import family10_store as f10                                    # noqa: E402
from family1 import adapters as fam                             # noqa: E402
from family1.adapters import _gbif, gbif, gbif_nc               # noqa: E402

# Part 000002 (January 1900) holds 36 records: 9 ordinary ones, 13 special
# cases, 7 geospatial droppers (one per measured issue name) and 7 other
# droppers. 16 of them are CC0 or CC BY (the public track), 6 are CC BY-NC.
PART_RECORDS = 36
PUBLIC_IN_JAN = 16
PRIVATE_IN_JAN = 6
# Parts 000001 and 000003 hold 15 December 1899 records and 15 June 1900
# ones: 15 ordinary (licences cycling 5 times) plus a dropper plus one
# outside the window. The public track keeps 10 of each, the private 5.
PUBLIC_ROWS = 10 + PUBLIC_IN_JAN + 10                          # 36
PRIVATE_ROWS = 5 + PRIVATE_IN_JAN + 5                          # 16


def test_registered_and_declared():
    assert fam.REGISTRY["gbif"] is gbif.GBIFAdapter
    assert fam.REGISTRY["gbif_nc"] is gbif_nc.GBIFNonCommercialAdapter
    a, b = gbif.GBIFAdapter(), gbif_nc.GBIFNonCommercialAdapter()
    for ad in (a, b):
        assert ad.C == 4 and ad.family == "1tf"
        assert ad.time_dtype == "int64" and ad.first_year == 1600
        assert ad.per_year is False and ad.platform_meta is True
        assert ad.credentials == ()
        assert ad.channel_names == ["kingdom_code", "class_code",
                                    "basis_of_record_code",
                                    "individual_count"]
        # the "unknown" footprint the design names: 1 km over 27.83 km
        assert abs(ad.log2_fp - np.log2(1.0 / 27.83)) < 1e-12
        assert abs(ad.log2_dt - np.log2(1.0 / 5.0)) < 1e-12
        assert b10.row_bytes(ad.C, ad.time_dtype) == 31 + 2 * 4      # 39
    assert a.track == "public" and a.distribution == "public"
    assert b.track == "private" and b.distribution == "private"
    assert b.licence["redistribution"] == "no"
    # the registry's two-track rule is what makes that pair safe
    assert a.licence["redistribution"] != "no"


def test_the_code_tables_fit_float16_and_are_sorted():
    """float16 represents integers exactly only up to 2048, so a code table
    longer than that would put a wrong class in the store."""
    assert len(_gbif.CLASSES) == 450 and len(set(_gbif.CLASSES)) == 450
    assert len(_gbif.KINGDOMS) == 19
    assert len(_gbif.BASIS_OF_RECORD) == 11
    for table in (_gbif.CLASSES, _gbif.KINGDOMS):
        assert len(table) < 2048
        assert list(table) == sorted(table), "a code table is APPENDED to, " \
                                             "never re-sorted, so the order " \
                                             "here is the measured one"
    assert np.float16(len(_gbif.CLASSES) - 1) == len(_gbif.CLASSES) - 1
    # the channel bounds are the tables' own lengths
    ad = gbif.GBIFAdapter()
    lo, hi = ad.bounds()
    assert hi[0] == len(_gbif.KINGDOMS) - 1
    assert hi[1] == len(_gbif.CLASSES) - 1
    assert hi[2] == len(_gbif.BASIS_OF_RECORD) - 1


def test_the_geospatial_issue_set_is_the_measured_seven():
    """Not the four that get quoted: `PRESUMED_SWAPPED_COORDINATE` and the
    two `PRESUMED_NEGATED_*` ones also imply a geospatial issue, and the
    union of exactly these seven is GBIF's own 7,089,295."""
    assert sorted(_gbif.GEOSPATIAL_ISSUES) == [
        "COORDINATE_INVALID", "COORDINATE_OUT_OF_RANGE",
        "COUNTRY_COORDINATE_MISMATCH", "PRESUMED_NEGATED_LATITUDE",
        "PRESUMED_NEGATED_LONGITUDE", "PRESUMED_SWAPPED_COORDINATE",
        "ZERO_COORDINATE"]
    m = _gbif.GEOSPATIAL_MEASURED
    assert m["union_of_the_seven"] == m["hasGeospatialIssue_true"] == 7089295
    # names that LOOK like they belong and do not
    for name in ("COORDINATE_ROUNDED", "COORDINATE_REPROJECTED",
                 "GEODETIC_DATUM_INVALID", "CONTINENT_COORDINATE_MISMATCH",
                 "COORDINATE_UNCERTAINTY_METERS_INVALID"):
        assert name not in _gbif.GEOSPATIAL_SET


def test_the_issue_column_is_read_in_both_encodings():
    """parquet-mr writes `list<struct<array_element: string>>` and pyarrow
    can write a plain `list<string>`; the encoding is the writer's choice and
    not a fact about the data."""
    assert _gbif._issue_names(None) == ()
    assert _gbif._issue_names(["ZERO_COORDINATE"]) == ("ZERO_COORDINATE",)
    assert _gbif._issue_names([{"array_element": "ZERO_COORDINATE"},
                               {"array_element": "COORDINATE_ROUNDED"}]) == \
        ("ZERO_COORDINATE", "COORDINATE_ROUNDED")


def test_the_qc_bitfield_round_trips_to_a_footprint():
    """qc is a BITFIELD for this store, and the six upper bits are the row's
    own footprint — the thing the layout's single (log2_fp, log2_dt) pair
    cannot carry."""
    def pack(unc, month_only):
        r = _gbif._expect(_gbif._rec(coordinateuncertaintyinmeters=unc,
                                     day=None if month_only else 2),
                          "public")
        return r["qc"]
    q = pack(100.0, False)
    assert q & _gbif.QC_NO_UNCERTAINTY == 0
    assert q & _gbif.QC_MONTH_ONLY == 0
    k = q >> _gbif.QC_K_SHIFT
    assert k == round(math.log2(100.0))                         # 7
    assert abs((k - math.log2(1000.0 * 27.83))
               - math.log2(100.0 / 1000.0 / 27.83)) < 0.5
    # no uncertainty: bit 0 set and k is the 1 km "unknown" constant
    q = pack(None, False)
    assert q & _gbif.QC_NO_UNCERTAINTY == 1
    assert q >> _gbif.QC_K_SHIFT == round(math.log2(1024.0))    # 10
    # month-only: bit 1 set, and the row's real time footprint is +2.606
    q = pack(100.0, True)
    assert q & _gbif.QC_MONTH_ONLY == 2
    assert abs(np.log2(_gbif.MEAN_MONTH_DAYS / 5.0) - 2.606) < 1e-3
    # the clamp: one metre reads as ten, five hundred km as one hundred
    assert pack(1.0, False) >> _gbif.QC_K_SHIFT == round(math.log2(10.0))
    assert pack(5e5, False) >> _gbif.QC_K_SHIFT == round(math.log2(1e5))
    assert pack(1e9, False) >> _gbif.QC_K_SHIFT <= _gbif.QC_K_MAX


def test_the_month_only_date_lands_in_the_middle_of_its_month():
    r = _gbif._expect(_gbif._rec(year=1900, month=1, day=None), "public")
    t = r["t"] - b10.seconds_since_epoch(dt.date(1900, 1, 1))
    assert t == (31 * 86400) // 2                 # the 16th at 12:00
    r = _gbif._expect(_gbif._rec(year=1900, month=2, day=None), "public")
    t = r["t"] - b10.seconds_since_epoch(dt.date(1900, 2, 1))
    assert t == (28 * 86400) // 2                 # 1900 is not a leap year
    # a day with no time of day is NOON, not midnight
    r = _gbif._expect(_gbif._rec(year=1900, month=1, day=2), "public")
    assert r["t"] - b10.seconds_since_epoch(dt.date(1900, 1, 2)) == 43200


def test_years_are_derived_without_a_python_loop():
    """`years_of` is the inverse of `_common.days_from_civil` and has to
    agree with it over the whole record, negative seconds included."""
    from family1.adapters import _common as cm
    days = np.array([cm.days_from_civil(y, m, 15)
                     for y in (1600, 1899, 1900, 1981, 1982, 2026)
                     for m in (1, 6, 12)], np.int64)
    want = np.array([y for y in (1600, 1899, 1900, 1981, 1982, 2026)
                     for _ in (1, 6, 12)], np.int64)
    assert _gbif.years_of(days * 86400 + 43200).tolist() == want.tolist()


def test_smoke_all_stages_and_the_probe_public(tmp_path):
    res = b1.run_smoke("gbif", root=str(tmp_path / "s"), keep=True)
    assert len(res["truth"]) == PUBLIC_ROWS
    p = res["probe"]
    # the probe read ONE part — January 1900 — and the framework filtered it
    # to the probe month, so the two counts agree by construction
    assert p["rows"] == PUBLIC_IN_JAN
    assert p["counts_scope"] == "one_part"
    assert p["schema_version"] == 3 and p["time_dtype"] == "int64"
    assert p["stored_bytes_per_row"] == 39
    assert p["bytes_fetched"] > 0
    assert p["distinct_platforms"] == PUBLIC_IN_JAN     # one taxon per row
    c = p["counts"]
    assert c["probe_part"] == "000002" and c["probe_parts_n"] == 1
    # the taxon sidecar the assemble reads back holds one entry per
    # platform the probe met, the pooled unmatched one included
    assert c["taxa_seen"] == p["distinct_platforms"]
    assert "not partitioned by time" in c["probe_month_argument_ignored"]
    assert c["records_read"] == PART_RECORDS
    assert c["probe_rows_in_part"] == PART_RECORDS
    assert c["rows_kept"] == PUBLIC_IN_JAN
    assert c["probe_rows_by_licence"] == {"CC0_1_0": 8, "CC_BY_4_0": 8}
    # every dropper, counted by the rule that dropped it
    assert c["records_geospatial_issue"] == len(_gbif.GEOSPATIAL_ISSUES)
    assert c["geospatial_issue_by_name"] == \
        {n: 1 for n in _gbif.GEOSPATIAL_ISSUES}
    assert c["records_no_position"] == 1
    assert c["records_position_out_of_range"] == 1
    assert c["records_no_month"] == 2          # no month, and no year at all
    assert c["records_bad_date"] == 1          # the 30th of February
    assert c["records_licence_unknown"] == 2
    assert c["records_licence_unknown_by_name"] == {"UNSPECIFIED": 1,
                                                    "UNSUPPORTED": 1}
    assert c["records_other_track"] == PRIVATE_IN_JAN
    # the per-row facts the qc bitfield carries
    assert c["records_month_only"] == 1
    assert c["records_with_time_of_day"] == 1  # the one whose date agrees
    assert c["records_no_uncertainty"] == 1
    assert c["records_uncertainty_clamped"] == 1
    assert c["records_taxon_unmatched"] == 1
    assert c["probe_share_day_precision"] == round(1 - 1 / PUBLIC_IN_JAN, 6)
    assert c["probe_share_with_uncertainty"] == \
        round(1 - 1 / PUBLIC_IN_JAN, 6)
    # a name no code table knows is NaN and counted BY NAME, never coerced
    assert c["kingdom_not_in_table"] == {"Nowhereia": 1}
    assert c["class_not_in_table"] == {"Nowhereopsida": 1}
    assert c["kingdom_absent"] == 1 and c["class_absent"] == 1
    assert p["nan_fraction"]["kingdom_code"] == round(2 / PUBLIC_IN_JAN, 6)
    assert p["nan_fraction"]["class_code"] == round(2 / PUBLIC_IN_JAN, 6)
    assert p["nan_fraction"]["basis_of_record_code"] == 0.0
    # 100,000 individuals is past float16's exact range and out of bounds
    assert c["out_of_bounds"] == {"individual_count": 1}
    assert p["nan_fraction"]["individual_count"] == \
        round(1 / PUBLIC_IN_JAN, 6)
    assert sum(p["out_of_bounds_stored"].values()) == 0
    # THE PROJECTION the store is sized from
    proj = c["probe_projection"]
    assert isinstance(proj, list) and len(proj) == 1
    proj = proj[0]
    assert proj["parts_read"] == 1
    assert proj["rows_per_part"] == PART_RECORDS
    assert proj["kept_per_part"] == PUBLIC_IN_JAN
    assert proj["stored_bytes_per_row"] == 39
    assert proj["projected_rows"] == PUBLIC_IN_JAN * 3
    assert proj["projected_store_bytes"] == PUBLIC_IN_JAN * 3 * 39
    assert len(proj["keep_fraction_per_part"]) == 1
    assert "585,034 CC BY-NC" in proj["basis"]


def test_smoke_all_stages_and_the_probe_private(tmp_path):
    res = b1.run_smoke("gbif_nc", root=str(tmp_path / "s"), keep=True)
    assert len(res["truth"]) == PRIVATE_ROWS
    p = res["probe"]
    assert p["rows"] == PRIVATE_IN_JAN
    assert p["distribution"] == "private"
    c = p["counts"]
    assert c["rows_kept"] == PRIVATE_IN_JAN
    assert c["probe_rows_by_licence"] == {"CC_BY_NC_4_0": PRIVATE_IN_JAN}
    assert c["records_other_track"] == PUBLIC_IN_JAN
    # the two tracks read the same part and keep disjoint rows
    assert c["records_read"] == PART_RECORDS
    assert PUBLIC_IN_JAN + PRIVATE_IN_JAN + c["records_licence_unknown"] \
        + c["records_geospatial_issue"] + c["records_no_position"] \
        + c["records_position_out_of_range"] + c["records_no_month"] \
        + c["records_bad_date"] == PART_RECORDS


def _ctx(tmp_path, ad, src, stage="all"):
    a = argparse.Namespace(
        store=ad.store, work=str(tmp_path / ("w_" + ad.store)),
        source_dir=src, start=ad.smoke_window[0], end=ad.smoke_window[1],
        stage=stage, force=False, attempts=1, qc_keep=2,
        check_chunk_rows=b10.CHECK_CHUNK_ROWS, assemble="auto",
        parts_from_hub=False, push_parts=False, allow_missing_years=False)
    return b10.Ctx(a, adapter=ad, layout=b1.layout_for(ad))


def test_the_stages_and_the_two_stores(tmp_path):
    src = str(tmp_path / "src")
    got = {}
    # the record with an unknown class is CC0, so only the PUBLIC track
    # meets it — which is itself the two-track rule working
    for cls, n, years, badclass in (
            (gbif.GBIFAdapter, PUBLIC_ROWS, {"1899": 10, "1900": 26},
             ["Nowhereopsida"]),
            (gbif_nc.GBIFNonCommercialAdapter, PRIVATE_ROWS,
             {"1899": 5, "1900": 11}, [])):
        ad = cls()
        lo, hi = (b10.parse_date(x) for x in ad.smoke_window)
        truth = ad.smoke_sources(src, lo, hi)
        ctx = _ctx(tmp_path, ad, src)
        b10.run_stages(ctx, ["index", "fetch", "assemble"],
                       stage_fn=b1.STAGE_FN, deps=b1.DEPS)
        b10.check_smoke(ctx, truth)
        out = b1.stage_check(ctx)
        assert out["N"] == n and out["schema_version"] == 3
        m = json.load(open(os.path.join(ctx.store, "store.json")))
        assert m["per_year"] == years
        assert m["N"] == n and m["distribution"] == ad.distribution
        assert m["platforms"]["in_store_without_entry"] == 0
        plan = json.load(open(os.path.join(ctx.root, "plan.json")))
        assert plan["snapshot"] == _gbif.SMOKE_SNAPSHOT
        assert plan["parts"] == len(_gbif.SMOKE_PARTS)
        assert plan["listing"]["other_objects_n"] == 1        # citation.txt
        assert plan["code_tables"] == {"kingdoms": 19, "classes": 450,
                                       "basis_of_record": 11}
        assert plan["first_part"]["part"] == "000002"
        assert plan["first_part"]["class_names_not_in_table"] == badclass
        assert plan["geospatial_issue_check"]["status"].startswith("skipped")
        st = f10.Store(ctx.store)
        assert st.N == n and st.C == 4
        got[ad.store] = set(zip(np.asarray(st["time_s"]).tolist(),
                                np.asarray(st["platform"]).tolist()))
        pl = json.load(open(os.path.join(ctx.store, "platforms.json")))
        assert len(pl) == m["platforms"]["entries"]
        one = next(iter(pl.values()))
        assert set(one) >= {"taxonkey", "scientificname", "kingdom", "class"}
    # THE TWO TRACKS KEEP DISJOINT ROWS: every record goes to exactly one
    # of them. The pair (time_s, platform) is the identity here rather
    # than the timestamp alone, because two different taxa legitimately
    # share a second — the month-only record sits at the middle of
    # January and another record sits at noon on the same day.
    assert not (got["gbif"] & got["gbif_nc"])
    assert len(got["gbif"]) + len(got["gbif_nc"]) == \
        PUBLIC_ROWS + PRIVATE_ROWS


def test_a_restricted_part_range_says_so(tmp_path, monkeypatch):
    src = str(tmp_path / "src")
    ad = gbif.GBIFAdapter()
    lo, hi = (b10.parse_date(x) for x in ad.smoke_window)
    ad.smoke_sources(src, lo, hi)
    monkeypatch.setenv("GBIF_PARTS", "0:1")
    ad = gbif.GBIFAdapter()
    assert "RESTRICTED BUILD" in ad.notes
    ctx = _ctx(tmp_path, ad, src, stage="index")
    assert [p[0] for p in ad.wanted_parts(ctx)] == ["000001"]
    monkeypatch.setenv("GBIF_PARTS", "000003")
    ad = gbif.GBIFAdapter()
    ctx = _ctx(tmp_path, ad, src, stage="index")
    assert [p[0] for p in ad.wanted_parts(ctx)] == ["000003"]
    monkeypatch.setenv("GBIF_PARTS", "999999")
    ad = gbif.GBIFAdapter()
    ctx = _ctx(tmp_path, ad, src, stage="index")
    with pytest.raises(SystemExit, match="does not list"):
        ad.wanted_parts(ctx)


def test_an_empty_or_reshaped_listing_is_refused():
    with pytest.raises(_gbif.FormatError, match="empty listing"):
        _gbif.parse_parts([("occurrence/2026-09-01/citation.txt", 88, "")],
                          "2026-09-01")
    parts, counts = _gbif.parse_parts(
        [("occurrence/2026-09-01/occurrence.parquet/000002", 7, ""),
         ("occurrence/2026-09-01/occurrence.parquet/000001", 5, ""),
         ("occurrence/2026-09-01/citation.txt", 88, "")], "2026-09-01")
    assert [p[0] for p in parts] == ["000001", "000002"]
    assert counts["parts"] == 2 and counts["other_objects_n"] == 1


def test_platforms_needs_the_fetch_s_own_taxon_table(tmp_path):
    """`platforms()` runs at ASSEMBLE time and the taxa are discovered during
    the FETCH, so an assemble on a machine that never fetched is refused with
    that sentence rather than publishing an empty platforms.json."""
    ad = gbif.GBIFAdapter()
    ctx = _ctx(tmp_path, ad, "")
    with pytest.raises(ValueError, match="taxon table the FETCH writes"):
        ad.platforms(ctx)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))


def test_the_probe_reads_a_spread_of_parts_by_default(tmp_path, monkeypatch):
    """ONE part was the design and one part is not enough: the parts follow
    the publishers, so the middle of the 2026-09-01 listing (005210) is
    585,034 CC BY-NC records of 585,040. The default spreads the probe."""
    src = str(tmp_path / "src")
    ad = gbif.GBIFAdapter()
    lo, hi = (b10.parse_date(x) for x in ad.smoke_window)
    ad.smoke_sources(src, lo, hi)          # sets GBIF_PROBE_PARTS = 1
    ctx = _ctx(tmp_path, ad, src, stage="index")
    assert [p[0] for p in ad.probe_parts(ctx)] == ["000002"]
    monkeypatch.delenv("GBIF_PROBE_PARTS", raising=False)
    ad = gbif.GBIFAdapter()
    assert ad.probe_n == 8
    ctx = _ctx(tmp_path, ad, src, stage="index")
    assert [p[0] for p in ad.probe_parts(ctx)] == list(_gbif.SMOKE_PARTS)
    # the spread is evenly placed over a long listing, and deterministic
    ad._listing = ("x", [(f"{i:06d}", f"k{i}", 10) for i in range(9898)], {})
    got = [p[0] for p in ad.probe_parts(None)]
    assert len(got) == 8 and got == sorted(got)
    assert got[0] == "000618" and got[-1] == "009279"
    assert [p[0] for p in ad.probe_parts(None)] == got


# ================================================ part-range lanes (2026-09-24)
# The whole pass was the resumable unit (23 h on one box; family1-build #837
# lost nine hours of it). A `GBIF_PARTS=lo:hi` run is now a GROUP LANE named
# after its part names, the taxon table travels with each year-lane's parts
# as a `taxa.json` SIDECAR, and `--lanes parts:<N>` declares the N lanes an
# assembly expects. These tests run the smoke snapshot through all of it.
sys.path.insert(0, HERE)
import family10_parts_hub as ph                                 # noqa: E402
from test_family1_lanes import FakePartsHub, copy_lane_parts    # noqa: E402

# WHOLE calendar years, so the window earns no lane name of its own and a
# part-range lane is `g-<hash>` alone — the shape a dispatch uses (no
# --start, --end 2026-12-31). The smoke's truth is the same over this window
# as over `smoke_window`: its outside records are in 1897 and 1903.
WIN = ("1899-01-01", "1900-12-31")
YEARS = (1899, 1900)


def _src(tmp_path):
    src = str(tmp_path / "src")
    ad = gbif.GBIFAdapter()
    lo, hi = (b10.parse_date(x) for x in WIN)
    return src, ad.smoke_sources(src, lo, hi)


def _gctx(monkeypatch, work, src, spec="", **over):
    """A gbif context the way `main` builds one: GBIF_PARTS read at
    construction, the lane named by `apply_lane`."""
    if spec:
        monkeypatch.setenv("GBIF_PARTS", spec)
    else:
        monkeypatch.delenv("GBIF_PARTS", raising=False)
    ad = gbif.GBIFAdapter()
    d = dict(store=ad.store, work=str(work), source_dir=src, start=WIN[0],
             end=WIN[1], stage="all", force=False, attempts=1, qc_keep=2,
             check_chunk_rows=b10.CHECK_CHUNK_ROWS, assemble="auto",
             parts_from_hub=False, push_parts=False, lanes="",
             allow_missing_years=False, allow_unconfirmed_licence=False,
             probe_month="", smoke=False)
    d.update(over)
    ctx = b10.Ctx(argparse.Namespace(**d), adapter=ad,
                  layout=b1.layout_for(ad))
    b1.apply_lane(ctx)
    return ctx


def _run(ctx, stages):
    b10.run_stages(ctx, stages, stage_fn=b1.STAGE_FN, deps=b1.DEPS)


def _rows(store):
    """The store's rows as sorted tuples — content, not order."""
    st = f10.Store(store)
    cols = [np.asarray(st[k]).tolist() for k in
            ("bin", "time_s", "lat", "lon", "platform", "qc")]
    vals = [tuple(None if v != v else float(v) for v in r)
            for r in np.asarray(st["values"], np.float64).tolist()]
    return sorted(zip(*cols, vals), key=repr)


def _hub(monkeypatch, tmp_path):
    fake = FakePartsHub(str(tmp_path / "hub"))
    monkeypatch.setattr(ph, "_list_files", fake.list_files)
    monkeypatch.setattr(ph, "_upload", fake.upload)
    monkeypatch.setattr(ph, "_download", fake.download)
    monkeypatch.setattr(ph, "_hub", lambda: pytest.fail("family 10's hub"))
    return fake


def test_a_part_range_is_a_group_lane(tmp_path, monkeypatch):
    src, truth = _src(tmp_path)
    ctx = _gctx(monkeypatch, tmp_path / "lane", src, "0:2")
    names = ["000001", "000002"]
    assert ctx.lane == b1.group_lane(names)
    assert ctx.lane.startswith("g-") and ctx.lane_groups == names
    _run(ctx, ["index", "fetch"])
    kept = {1899: 0, 1900: 0}
    for y in YEARS:
        d = ctx.year_dir(y)
        assert d.endswith(os.path.join("parts", str(y), ctx.lane))
        assert b10.marked(ctx.root, f"parts/{y}/{ctx.lane}")
        led = json.load(open(os.path.join(d, "counts.json")))
        assert led["lane"] == ctx.lane
        assert led["lane_groups"] == names
        assert led["lane_window"] == [f"{y}-01-01", f"{y}-12-31"]
        kept[y] = led["rows"]
        # the unnamed lane was never written: nothing at the year's top level
        # but the lane's own marker
        top = os.path.join(ctx.parts, str(y))
        assert not [n for n in os.listdir(top)
                    if os.path.isfile(os.path.join(top, n))
                    and not n.endswith(".done")]
    # parts 000001 (December 1899) and 000002 (January 1900) only
    assert kept == {1899: 10, 1900: PUBLIC_IN_JAN}
    # a comma list of the same parts, in any order, is the SAME lane
    other = _gctx(monkeypatch, tmp_path / "b", src, "000002,000001")
    assert other.lane == ctx.lane
    # no GBIF_PARTS: the whole snapshot, the unnamed lane, exactly as before
    assert _gctx(monkeypatch, tmp_path / "c", src).lane == ""
    # a range over nothing would claim the whole snapshot: refused
    with pytest.raises(SystemExit, match="selects no part"):
        _gctx(monkeypatch, tmp_path / "d", src, "5:9")
    # and a box assembling from the Hub is never a lane, GBIF_PARTS or not
    assert _gctx(monkeypatch, tmp_path / "e", src, "0:2",
                 parts_from_hub=True).lane == ""


def test_two_part_lanes_through_the_hub_equal_the_whole_build(tmp_path,
                                                              monkeypatch):
    src, truth = _src(tmp_path)
    fake = _hub(monkeypatch, tmp_path)
    ad = gbif.GBIFAdapter()
    listing = ad.listing(_gctx(monkeypatch, tmp_path / "l", src))[1]
    lanes = _gbif.part_lanes(listing, 2)          # 000001 | 000002, 000003
    assert [(lo, hi) for lo, hi, _ in lanes] == [(0, 1), (1, 3)]
    names = sorted(n for _lo, _hi, n in lanes)

    # two hosted lanes, each pushing only its own folders
    for lo, hi, name in lanes:
        ctx = _gctx(monkeypatch, tmp_path / f"lane{lo}", src, f"{lo}:{hi}",
                    push_parts=True)
        monkeypatch.setattr(ctx.layout, "hub", fake.hub)
        assert ctx.lane == name
        _run(ctx, ["index", "fetch"])
    pre = os.path.join(fake.root, "partials/family1_tf/gbif")
    for y in YEARS:
        assert sorted(os.listdir(os.path.join(pre, str(y)))) == names

    # THE SIDECAR TRAVELS: in done.json, between the parts and the ledger,
    # hashed; and only where the lane's year has rows
    for lo, hi, name in lanes:
        for y in YEARS:
            d = os.path.join(pre, str(y), name)
            done = json.load(open(os.path.join(d, "done.json")))
            files = [e["name"] for e in done["files"]]
            has_rows = done["rows"] > 0
            assert (_gbif.TAXA_SIDECAR in files) == has_rows, (y, name)
            if has_rows:
                assert files[-2:] == [_gbif.TAXA_SIDECAR, "counts.json"]
                assert done["n_parts"] == sum(f.endswith(".npz")
                                              for f in files)
                e = files.index(_gbif.TAXA_SIDECAR)
                assert done["files"][e]["sha256"] == b10.sha256(
                    os.path.join(d, _gbif.TAXA_SIDECAR))
    # lane 0:1 is December 1899 only, so its 1900 has no row and no sidecar
    assert json.load(open(os.path.join(pre, "1900", lanes[0][2],
                                       "done.json")))["rows"] == 0

    # the box: a fresh tree, nothing fetched, both lanes pulled and DECLARED
    monkeypatch.delenv("GBIF_PARTS", raising=False)
    box = _gctx(monkeypatch, tmp_path / "box", src, parts_from_hub=True,
                lanes="parts:2")
    monkeypatch.setattr(box.layout, "hub", fake.hub)
    assert box.lane == ""
    _run(box, ["index", "fetch", "assemble", "check"])
    plan = json.load(open(os.path.join(box.root, "plan.json")))
    assert plan["lanes_expected"] == {str(y): [n for _l, _h, n in lanes]
                                      for y in YEARS}
    for y in YEARS:
        assert box.lanes_of(y) == names
    # the box never ran a fetch: no taxa.json of its own, only the sidecars
    assert not os.path.exists(os.path.join(box.root, "taxa.json"))
    # a sidecar is never read as a part
    assert not [p for _y, p in b10._part_paths(box)
                if not p.endswith(".npz")]
    b10.check_smoke(box, truth)
    meta = json.load(open(os.path.join(box.store, "store.json")))
    assert meta["lanes_by_year"]["1900"] == {
        "lanes": names, "declared": True,
        "expected": [n for _l, _h, n in lanes]}
    assert meta["platforms"]["in_store_without_entry"] == 0

    # the single-lane build of the whole snapshot, for comparison
    one = _gctx(monkeypatch, tmp_path / "one", src)
    assert one.lane == ""
    _run(one, ["index", "fetch", "assemble"])
    one_meta = json.load(open(os.path.join(one.store, "store.json")))
    assert meta["N"] == one_meta["N"] == PUBLIC_ROWS
    assert meta["per_year"] == one_meta["per_year"]
    assert _rows(box.store) == _rows(one.store)
    # the taxon table: platforms.json and platforms(ctx) both equal the full
    # build's own table
    assert json.load(open(os.path.join(box.store, "platforms.json"))) == \
        json.load(open(os.path.join(one.store, "platforms.json")))
    full = {int(k): v for k, v in json.load(
        open(os.path.join(one.root, "taxa.json"))).items()}
    assert box.adapter.platforms(box) == full
    assert one.adapter.platforms(one) == full

    # a sidecar whose bytes changed on the Hub is refused at the pull
    bad = os.path.join(pre, "1900", lanes[1][2], _gbif.TAXA_SIDECAR)
    with open(bad, "a") as fh:
        fh.write(" ")
    box2 = _gctx(monkeypatch, tmp_path / "box2", src, parts_from_hub=True)
    monkeypatch.setattr(box2.layout, "hub", fake.hub)
    with pytest.raises(SystemExit, match="PULL MISMATCH .*taxa.json"):
        _run(box2, ["index", "fetch"])


def test_the_sidecar_is_listed_between_parts_and_ledger(tmp_path):
    assert _gbif.TAXA_SIDECAR in ph.SIDECAR_NAMES
    d = tmp_path / "y"
    d.mkdir()
    for n in ("00001.npz", "00000.npz", "counts.json", "taxa.json",
              "stray.txt"):
        (d / n).write_text("x")
    assert ph.local_part_files(str(d)) == ["00000.npz", "00001.npz",
                                           "taxa.json", "counts.json"]


def test_platforms_unions_the_sidecars_and_refuses_a_disagreement(
        tmp_path, monkeypatch):
    src, _ = _src(tmp_path)
    ctx = _gctx(monkeypatch, tmp_path / "w", src)
    ad = ctx.adapter
    rec = {"taxonkey": "K", "scientificname": "A", "kingdom": "Plantae",
           "class": "Liliopsida"}
    for y, lane, r in ((1899, "g-aaaaaaaa", rec),
                       (1900, "g-bbbbbbbb", dict(rec, scientificname="B"))):
        d = ctx.year_dir(y, lane)
        os.makedirs(d)
        json.dump({"7": r}, open(os.path.join(d, _gbif.TAXA_SIDECAR), "w"))
    with pytest.raises(ValueError, match="platform 7 .*'K'.*two different"):
        ad.platforms(ctx)
    json.dump({"7": rec, "8": dict(rec, taxonkey="L")},
              open(os.path.join(ctx.year_dir(1900, "g-bbbbbbbb"),
                                _gbif.TAXA_SIDECAR), "w"))
    assert sorted(ad.platforms(ctx)) == [7, 8]
    # two snapshots in one assembly are refused too
    for y, lane, snap in ((1899, "g-aaaaaaaa", "2026-08-01"),
                          (1900, "g-bbbbbbbb", "2026-09-01")):
        json.dump({"rows": 1, "counts": {"snapshot": snap}},
                  open(os.path.join(ctx.year_dir(y, lane), "counts.json"),
                       "w"))
    with pytest.raises(ValueError, match="2 different GBIF snapshots"):
        ad.platforms(ctx)


def test_part_lanes_cover_every_part_exactly_once(tmp_path):
    names = [f"{i:06d}" for i in range(1, 9899)]            # 9,898 parts
    lanes = _gbif.part_lanes(names, 16)
    assert len(lanes) == 16
    assert [n for lo, hi, _ in lanes for n in names[lo:hi]] == names
    sizes = [hi - lo for lo, hi, _ in lanes]
    assert sizes == [618] * 15 + [628]            # the last takes the rest
    assert all(lanes[i][1] == lanes[i + 1][0] for i in range(15))
    for lo, hi, name in lanes:
        assert name == b1.group_lane(names[lo:hi])
    assert len({n for _l, _h, n in lanes}) == 16
    # the listing's own tuples give the same answer
    assert _gbif.part_lanes([(n, f"k/{n}", 1) for n in names], 16) == lanes
    assert _gbif.part_lanes(names, 1) == [(0, 9898, b1.group_lane(names))]
    for bad in (0, 9899):
        with pytest.raises(ValueError, match="non-empty"):
            _gbif.part_lanes(names, bad)
    with pytest.raises(ValueError, match="sorted"):
        _gbif.part_lanes(list(reversed(names)), 2)
    # the dispatcher's printout, off the smoke listing
    src, _ = _src(tmp_path)
    out = _gbif.main(["--lanes", "2", "--source-dir", src, "--json"])
    assert [(r["GBIF_PARTS"], r["lane_name"]) for r in out] == \
        [(f"{lo}:{hi}", n) for lo, hi, n in
         _gbif.part_lanes(list(_gbif.SMOKE_PARTS), 2)]


def test_lanes_parts_n_is_declared_and_a_missing_lane_is_refused(
        tmp_path, monkeypatch):
    src, _ = _src(tmp_path)
    box = _gctx(monkeypatch, tmp_path / "box", src, lanes="parts:2")
    _run(box, ["index"])
    lanes = _gbif.part_lanes(list(_gbif.SMOKE_PARTS), 2)
    want = [n for _l, _h, n in lanes]
    plan = json.load(open(os.path.join(box.root, "plan.json")))
    assert plan["lanes_expected"] == {"1899": want, "1900": want}
    # only the first lane arrives
    lo, hi, name = lanes[0]
    w = tmp_path / "lane0"
    ctx = _gctx(monkeypatch, w, src, f"{lo}:{hi}")
    assert ctx.lane == name
    _run(ctx, ["index", "fetch"])
    copy_lane_parts(str(w), str(tmp_path / "box"), "gbif")
    with pytest.raises(SystemExit, match=lanes[1][2]):
        b1.stage_assemble(box)
    # the declaration is refused where it cannot mean anything
    with pytest.raises(SystemExit, match="positive"):
        b10.declare_lanes(_gctx(monkeypatch, tmp_path / "z", src,
                                lanes="parts:0"))
    with pytest.raises(SystemExit, match="GBIF_PARTS"):
        b10.declare_lanes(_gctx(monkeypatch, tmp_path / "y", src, "0:1",
                                lanes="parts:2"))
    other = argparse.Namespace(a=argparse.Namespace(lanes="parts:2"),
                               adapter=argparse.Namespace(store="ghcnd"),
                               years=[2022])
    with pytest.raises(SystemExit, match="does not split"):
        b10.declare_lanes(other)


def test_an_incomplete_range_read_is_retried_not_fatal(monkeypatch):
    """family1-build #907 (a gbif part lane, 2026-09-24): S3 closed one
    range response early — `http.client.IncompleteRead`, which is an
    HTTPException and not an OSError — and the lane died on that one read.
    `HttpParquet.read` retries it like any other transport failure."""
    import http.client
    import urllib.request
    from family1.adapters import _gbif as g
    from family1.adapters import _common as cm
    assert issubclass(http.client.IncompleteRead, cm.RETRY_ERRORS)
    body = bytes(range(64))
    calls = []

    class Resp:
        def __init__(self, data, cut):
            self.status, self._d, self._cut = 206, data, cut

        def read(self):
            if self._cut:
                raise http.client.IncompleteRead(self._d[:10], len(self._d) - 10)
            return self._d

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    def fake_open(req, timeout=None):
        calls.append(req.get_header("Range"))
        return Resp(body[:16], cut=(len(calls) == 1))
    monkeypatch.setattr(urllib.request, "urlopen", fake_open)
    monkeypatch.setattr(g.time, "sleep", lambda s: None)
    f = g.HttpParquet("https://x/part.parquet", len(body), attempts=3)
    assert f.read(16) == body[:16]
    assert calls == ["bytes=0-15", "bytes=0-15"]

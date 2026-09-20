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

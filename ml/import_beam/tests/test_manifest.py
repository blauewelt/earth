"""The manifest must be deterministic, and its item ids must be the ones the
design names — they are what a human types into `--only` and what a re-run
matches against."""
from __future__ import annotations

from beam_import import manifest, registry


def _tier0(real_registry):
    reg = registry.load(real_registry)
    return reg, manifest.build(reg, [0], offline=True)


def test_deterministic(real_registry):
    reg, a = _tier0(real_registry)
    b = manifest.build(reg, [0], offline=True)
    assert [i["item_id"] for i in a] == [i["item_id"] for i in b]
    assert [i["lane"] for i in a] == [i["lane"] for i in b]


def test_item_id_shapes(real_registry):
    _reg, items = _tier0(real_registry)
    ids = {i["item_id"] for i in items}
    assert "oisst/1993" in ids
    assert "ncep/uflx.sfc.gauss/2001" in ids
    assert "rg/RG_ArgoClim_Temperature_2019.nc.gz" in ids
    assert "ncep/land.sfc.gauss.nc" in ids


def test_duacs_month_ids(real_registry):
    reg = registry.load(real_registry)
    items = manifest.build(reg, [1], only="duacs", offline=True)
    ids = {i["item_id"] for i in items}
    assert "duacs/2003-07" in ids
    assert len(items) == 384


def test_counts_match_the_registry(real_registry):
    reg, items = _tier0(real_registry)
    per_source = manifest.counts(items)["by_source"]
    for src in reg.sources:
        exp = src.get("expected_items")
        if exp is None or src["name"] not in per_source:
            continue
        assert per_source[src["name"]] == exp, src["name"]


def test_lane_is_inside_the_host_budget(real_registry):
    reg, items = _tier0(real_registry)
    for it in items:
        assert 0 <= it["lane"] < reg.host(it["host"])["max_lanes"]


def test_hub_paths_are_unique_and_prefixed(real_registry):
    _reg, items = _tier0(real_registry)
    paths = [i["hub_path"] for i in items]
    assert len(paths) == len(set(paths))
    for it in items:
        if it["source"] == "glorys":
            assert it["hub_path"].startswith("daily025_global/")
        else:
            assert it["hub_path"].startswith("sources/")


def test_disabled_sources_are_skipped_unless_asked_for(real_registry):
    reg = registry.load(real_registry)
    names = {i["source"] for i in manifest.build(reg, [0, 1, 2], offline=True)}
    assert "ostia" not in names and "oisst_psl" not in names
    explicit = manifest.build(reg, [0], only="oisst_psl", offline=True)
    assert len(explicit) == 88


def test_unverified_urls_are_still_listed_and_flagged(real_registry):
    reg = registry.load(real_registry)
    items = manifest.build(reg, [1, 2], offline=True)
    flagged = [i for i in items if i["unverified_url"]]
    assert flagged, "tier 1/2 has sources whose URL patterns are unverified"
    assert all(i["item_id"] for i in flagged)


def test_month_and_day_helpers():
    assert manifest.month_range("1993-11", "1994-02") == [
        "1993-11", "1993-12", "1994-01", "1994-02"]
    assert len(manifest.days_of_month("2000-02")) == 29
    assert len(manifest.days_of_year(1981, "1981-09")) == 122


def test_lane_of_is_stable_across_processes():
    # sha1, not hash(): hash() is salted per process and would scatter the
    # same item into different lanes in different workers.
    assert manifest.lane_of("oisst/1993", 6) == manifest.lane_of("oisst/1993", 6)
    assert 0 <= manifest.lane_of("duacs/2003-07", 4) < 4


def test_verify_only_items_are_not_counted_as_wire(real_registry):
    """GLORYS is already on the Hub. Counting the 2.2 GB per month it WOULD
    have cost to download overstates Tier 0 by a factor of four."""
    reg, items = _tier0(real_registry)
    c = manifest.counts(items)

    verify = [i for i in items if i["mode"] == "verify"]
    fetch = [i for i in items if i["mode"] != "verify"]
    assert {i["source"] for i in verify} == {"glorys"}
    assert c["verify_items"] == len(verify) == 384
    assert c["fetch_items"] == len(fetch)
    assert c["fetch_wire"] == sum(i["bytes_wire"] for i in fetch)
    assert c["fetch_stored"] == sum(i["bytes_stored"] for i in fetch)
    # the verify items contribute nothing to either fetch total
    assert c["fetch_wire"] < sum(i["bytes_wire"] for i in items)
    assert c["verify_stored"] == sum(i["bytes_stored"] for i in verify)


def test_counts_still_carry_the_per_source_and_per_host_tables(real_registry):
    _reg, items = _tier0(real_registry)
    c = manifest.counts(items)
    assert c["by_source"]["ncep"] == 560
    assert c["by_host"]["psl"] == 560
    assert sum(c["by_lane"].values()) == len(items)

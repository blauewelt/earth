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


def test_shard_paths_are_unique_and_derived_from_the_item_id(real_registry):
    from beam_import import sinks
    _reg, items = _tier0(real_registry)
    paths = [sinks.shard_uri("/out", i["item_id"]) for i in items]
    assert len(paths) == len(set(paths))
    assert sinks.shard_uri("/out", "ncep/uflx.sfc.gauss/2001") == \
        "/out/ncep/uflx.sfc.gauss/2001.tfrecord"
    assert sinks.done_uri("/out", "oisst/1993") == "/out/oisst/1993.done"


def test_disabled_sources_are_skipped_unless_asked_for(real_registry):
    reg = registry.load(real_registry)
    names = {i["source"] for i in manifest.build(reg, [0, 1, 2], offline=True)}
    assert "ostia" not in names and "oisst_psl" not in names
    assert "glorys_from_mirror" not in names
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


def test_everything_is_fetched_now(real_registry):
    """Revision 2 removed the `verify` mode: GLORYS is imported like anything
    else, and `glorys_from_mirror` is the optional cheaper path."""
    reg, items = _tier0(real_registry)
    c = manifest.counts(items)
    assert c["verify_items"] == 0
    assert c["fetch_items"] == len(items)
    assert c["fetch_wire"] == sum(i["bytes_wire"] for i in items)


def test_day_items_fill_beside_their_parent(real_registry):
    from beam_import import sinks
    _reg, items = _tier0(real_registry)
    parent = next(i for i in items if i["item_id"] == "oisst/1993")
    day = manifest.day_item(parent, "1993-03-04")
    assert day["item_id"] == "oisst/1993/1993-03-04"
    assert day["day"] == "1993-03-04"
    assert day["fill_parent"] == "oisst/1993"
    assert 0 <= day["lane"] < parent["host_max_lanes"]
    # the fill shard sits BESIDE the parent's, and does not rewrite it
    assert sinks.shard_uri("/o", day["fill_parent"], day["fill_token"]) == \
        "/o/oisst/1993.fill-19930304.tfrecord"


def test_from_queue_reads_whole_items(tmp_path, real_registry):
    from beam_import import sinks
    _reg, items = _tier0(real_registry)
    sinks.enqueue(str(tmp_path), items[:3], "test")
    back = manifest.from_queue(sinks.queue_uri(str(tmp_path)))
    assert sorted(i["item_id"] for i in back) == \
        sorted(i["item_id"] for i in items[:3])
    assert all("queued_reason" not in i for i in back)
    assert back[0]["urls"] == next(
        i for i in items[:3] if i["item_id"] == back[0]["item_id"])["urls"]


def test_counts_still_carry_the_per_source_and_per_host_tables(real_registry):
    _reg, items = _tier0(real_registry)
    c = manifest.counts(items)
    assert c["by_source"]["ncep"] == 560
    assert c["by_host"]["psl"] == 560
    assert sum(c["by_lane"].values()) == len(items)

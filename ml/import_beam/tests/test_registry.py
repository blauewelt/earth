"""The registry is the only place a connection budget lives — so it has to be
impossible to add a source that does not name a host with both numbers."""
from __future__ import annotations

import pytest
import yaml

from beam_import import registry


def test_real_registry_loads(real_registry):
    reg = registry.load(real_registry)
    assert reg.hosts and reg.sources
    # Revision 2: no destination in the registry at all — where the output
    # goes is `--output <uri>` on the command line.
    assert "hub" not in open(real_registry, encoding="utf-8").read()[:2000]
    assert reg.num_shards_per_group() == 64


def test_every_source_names_a_host_with_both_numbers(real_registry):
    reg = registry.load(real_registry)
    for src in reg.sources:
        host = reg.host(src["host"])            # KeyError if it does not exist
        assert int(host["max_lanes"]) >= 1
        assert float(host["min_gap_s"]) >= 0


def test_glorys_is_fetched_and_the_mirror_shortcut_is_off_by_default(
        real_registry):
    reg = registry.load(real_registry)
    assert reg.source("glorys")["mode"] == "fetch"
    mirror = reg.source("glorys_from_mirror")
    assert mirror["enabled"] is False
    assert mirror["host"] == "hf_public"
    assert mirror["transform"] == "nc025_days"     # already binned: no re-bin


def test_every_source_has_a_known_transform_and_a_grid(real_registry):
    reg = registry.load(real_registry)
    for src in reg.sources:
        assert src["transform"] in registry.TRANSFORMS, src["name"]
        assert src.get("grid"), src["name"]


def test_a_source_without_a_host_is_refused(tmp_path):
    doc = {
        "hosts": {"a": {"max_lanes": 1, "min_gap_s": 1}},
        "sources": [{"name": "s", "tier": 0, "host": "nope", "mode": "fetch",
                     "fetcher": "http", "chunk": "file", "files": ["f"],
                     "transform": "opaque"}],
    }
    p = tmp_path / "bad.yaml"
    p.write_text(yaml.safe_dump(doc), encoding="utf-8")
    with pytest.raises(registry.RegistryError):
        registry.load(str(p))


def test_a_host_missing_a_number_is_refused(tmp_path):
    doc = {
        "hosts": {"a": {"max_lanes": 1}},        # no min_gap_s
        "sources": [{"name": "s", "tier": 0, "host": "a", "mode": "fetch",
                     "fetcher": "http", "chunk": "file", "files": ["f"],
                     "transform": "opaque"}],
    }
    p = tmp_path / "bad.yaml"
    p.write_text(yaml.safe_dump(doc), encoding="utf-8")
    with pytest.raises(registry.RegistryError):
        registry.load(str(p))

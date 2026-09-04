"""The registry is the only place a connection budget lives — so it has to be
impossible to add a source that does not name a host with both numbers."""
from __future__ import annotations

import pytest
import yaml

from beam_import import registry


def test_real_registry_loads(real_registry):
    reg = registry.load(real_registry)
    assert reg.hosts and reg.sources
    assert reg.hub["repo_id"] == "chfrank/earth-tensors"


def test_every_source_names_a_host_with_both_numbers(real_registry):
    reg = registry.load(real_registry)
    for src in reg.sources:
        host = reg.host(src["host"])            # KeyError if it does not exist
        assert int(host["max_lanes"]) >= 1
        assert float(host["min_gap_s"]) >= 0


def test_glorys_is_verify_only_and_keeps_its_hub_home(real_registry):
    reg = registry.load(real_registry)
    glorys = reg.source("glorys")
    assert glorys["mode"] == "verify"
    assert glorys["hub_path"].startswith("daily025_global/")


def test_big_items_are_never_batched(real_registry):
    reg = registry.load(real_registry)
    for src in reg.sources:
        if int(src.get("bytes_stored", 0)) > 200 * 1024 * 1024:
            assert int(src["batch_files"]) == 1, src["name"]


def test_a_source_without_a_host_is_refused(tmp_path):
    doc = {
        "hub": {"repo_id": "x/y"},
        "hosts": {"a": {"max_lanes": 1, "min_gap_s": 1}},
        "sources": [{"name": "s", "tier": 0, "host": "nope", "mode": "fetch",
                     "fetcher": "http", "chunk": "file", "files": ["f"],
                     "transform": "passthrough", "hub_prefix": "s",
                     "batch_files": 1}],
    }
    p = tmp_path / "bad.yaml"
    p.write_text(yaml.safe_dump(doc), encoding="utf-8")
    with pytest.raises(registry.RegistryError):
        registry.load(str(p))


def test_a_host_missing_a_number_is_refused(tmp_path):
    doc = {
        "hub": {"repo_id": "x/y"},
        "hosts": {"a": {"max_lanes": 1}},        # no min_gap_s
        "sources": [{"name": "s", "tier": 0, "host": "a", "mode": "fetch",
                     "fetcher": "http", "chunk": "file", "files": ["f"],
                     "transform": "passthrough", "hub_prefix": "s",
                     "batch_files": 1}],
    }
    p = tmp_path / "bad.yaml"
    p.write_text(yaml.safe_dump(doc), encoding="utf-8")
    with pytest.raises(registry.RegistryError):
        registry.load(str(p))

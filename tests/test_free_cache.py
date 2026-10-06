#!/usr/bin/env python3
"""ml/free_cache.py — which cache directories a box may delete (E-087 §14)."""
import os
import sys

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "ml"))
import free_cache as fc                                          # noqa: E402


@pytest.mark.parametrize("item", ["family7_2d/ncep100d", "family1_gf/oc4k",
                                  "family1_2/era5_q", "family10_2/fishing"])
def test_family_store_items_parse(item):
    assert fc.parse(f" {item} ,") == [item]


@pytest.mark.parametrize("item", ["family1_x/../..", "family7_2d/../x",
                                  "../family7_2d/x", "/opt/x", "family7_2d",
                                  "family7_2d/a/b", "family7_2d/A", "data/x",
                                  "family7.2d/x", "family7_2d/"])
def test_anything_else_is_refused(item):
    with pytest.raises(ValueError):
        fc.parse(item)


def _store(root, item, published=True):
    d = root / item
    d.mkdir(parents=True)
    (d / "parts").mkdir()
    (d / "parts" / "x.zst").write_bytes(b"x")
    if published:
        (d / "publish.done").write_text("2026-10-06T00:00:00Z\n")
    return d


def test_a_published_store_is_freed_and_its_siblings_kept(tmp_path):
    a = _store(tmp_path, "family7_2d/oisst025d")
    b = _store(tmp_path, "family7_2d/glorys025d")
    gone = fc.free(str(tmp_path), fc.parse("family7_2d/oisst025d"),
                   log=lambda *_: None)
    assert gone == [str(a)] and not a.exists() and b.exists()


def test_an_unpublished_store_is_refused_and_kept(tmp_path):
    a = _store(tmp_path, "family7_2d/ncep100d", published=False)
    with pytest.raises(ValueError):
        fc.free(str(tmp_path), ["family7_2d/ncep100d"], log=lambda *_: None)
    assert a.exists()


def test_a_missing_store_is_skipped(tmp_path):
    assert fc.free(str(tmp_path), ["family7_2d/occci025d"],
                   log=lambda *_: None) == []


def test_main_exits_nonzero_on_a_bad_item(tmp_path):
    assert fc.main(["--root", str(tmp_path), "family1_x/../.."]) == 1

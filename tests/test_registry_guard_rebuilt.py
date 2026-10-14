#!/usr/bin/env python3
"""ml/registry_guard.py's `--allow-group rebuilt_<code>` (E-087 §14): every
store of one family rebuilt — groups and build state may change, nothing
else, and never the set of stores."""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "ml"))
import registry_guard as rg                                      # noqa: E402


def _reg(**kw):
    r = {"family_code": "72d", "family": "family7_2d", "repo": "x",
         "built": ["a"], "not_built": ["b"], "n_built": 1,
         "description": "old", "generated_utc": "t0",
         "groups": [{"name": "a", "built": True, "record_span": ["1", "2"]},
                    {"name": "b", "built": False}]}
    r.update(kw)
    return r


def _run(tmp_path, old, new, *args):
    p, q = tmp_path / "old.json", tmp_path / "new.json"
    p.write_text(json.dumps(old))
    q.write_text(json.dumps(new))
    return rg.main([str(p), str(q), *args])


def test_rebuilt_family_may_change_groups_and_build_state(tmp_path):
    new = _reg(built=["a", "b"], not_built=[], n_built=2, description="new",
               generated_utc="t1",
               groups=[{"name": "a", "built": True, "record_span": ["1", "3"]},
                       {"name": "b", "built": True, "record_span": ["1", "3"]}])
    assert _run(tmp_path, _reg(), new, "--allow-group", "rebuilt_72d") == 0
    assert _run(tmp_path, _reg(), new) == 1          # strict still refuses


def test_rebuilt_family_holds_every_other_field(tmp_path):
    assert _run(tmp_path, _reg(), _reg(repo="y"),
                "--allow-group", "rebuilt_72d") == 1


def test_rebuilt_family_may_not_lose_a_store(tmp_path):
    new = _reg(groups=[{"name": "a", "built": True}])
    assert _run(tmp_path, _reg(), new, "--allow-group", "rebuilt_72d") == 1


def test_rebuilt_token_must_name_this_registrys_family(tmp_path):
    assert _run(tmp_path, _reg(), _reg(description="x"),
                "--allow-group", "rebuilt_12") == 1


# `--allow-group added_<name>` (E-092): a store published for the FIRST time
# into an existing registry. The new group and the build counts may appear;
# every existing group and every other top-level field is held to equality.
def _added(**kw):
    new = _reg(built=["a", "c"], not_built=["b"], n_built=2, n_groups=3,
               description="new", generated_utc="t1",
               groups=[{"name": "a", "built": True, "record_span": ["1", "2"]},
                       {"name": "b", "built": False},
                       {"name": "c", "built": True, "record_span": ["5", "6"]}])
    new.update(kw)
    return new


def test_added_store_may_appear(tmp_path):
    old = _reg(n_groups=2)
    assert _run(tmp_path, old, _added(), "--allow-group", "added_c") == 0
    assert _run(tmp_path, old, _added()) == 1                 # strict refuses


def test_added_store_holds_the_existing_groups(tmp_path):
    old = _reg(n_groups=2)
    new = _added()
    new["groups"][0]["record_span"] = ["1", "9"]               # a changed
    assert _run(tmp_path, old, new, "--allow-group", "added_c") == 1


def test_added_store_must_be_new_and_present(tmp_path):
    old = _reg(n_groups=2)
    assert _run(tmp_path, old, _added(), "--allow-group", "added_a") == 1
    assert _run(tmp_path, old, _added(), "--allow-group", "added_z") == 1


def test_added_store_holds_other_top_level_fields(tmp_path):
    old = _reg(n_groups=2)
    assert _run(tmp_path, old, _added(repo="y"),
                "--allow-group", "added_c") == 1


def test_added_store_may_bring_new_not_built_placeholders(tmp_path):
    # E-092 phase C: adapters for several stores land together and are
    # published one at a time, so the rebuilt registry also lists the others
    # as `built: false`. A NEW not-built entry holds no data and passes; a
    # new entry that claims to be built does not.
    old = _reg(n_groups=2)
    new = _added(n_groups=4)
    new["groups"].append({"name": "d", "built": False})
    assert _run(tmp_path, old, new, "--allow-group", "added_c") == 0
    new["groups"][-1]["built"] = True
    assert _run(tmp_path, old, new, "--allow-group", "added_c") == 1


def test_added_store_may_replace_its_own_not_built_placeholder(tmp_path):
    # the published registry already lists the store as `built: false`
    old = _reg(n_groups=3,
               groups=[{"name": "a", "built": True, "record_span": ["1", "2"]},
                       {"name": "b", "built": False},
                       {"name": "c", "built": False}])
    assert _run(tmp_path, old, _added(), "--allow-group", "added_c") == 0
    # but a store already BUILT in the published registry is not "added"
    assert _run(tmp_path, old, _added(), "--allow-group", "added_a") == 1

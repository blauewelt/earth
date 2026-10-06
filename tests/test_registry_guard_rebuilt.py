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

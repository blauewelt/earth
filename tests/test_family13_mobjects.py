#!/usr/bin/env python3
"""The `mobjects` adapter's smoke (family 1.3, E-092) — no network.

    python3 -m pytest -q tests/test_family13_mobjects.py

The synthetic Lebreton 2018 archive (shared with `mplastic`'s smoke, written
in its real layout with float-noise event ids) carries 1,595 objects over the
31 aircraft mosaics; the store must hold exactly them, each at the midpoint of
its start and end, on its mosaic's day, with its type in qc.
"""
import json
import os
import sys

import numpy as np
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "ml"))

pytest.importorskip("xlwt")
pytest.importorskip("xlrd")
pytest.importorskip("openpyxl")

import build_family1_stores as b1                               # noqa: E402
import family10_store as f10                                    # noqa: E402
from family1 import adapters as fam                             # noqa: E402
from family1.adapters import _lebreton2018 as lb                # noqa: E402
from family1.adapters import mobjects as mo                     # noqa: E402
from family1.adapters import mplastic as mp                     # noqa: E402


@pytest.fixture(scope="module")
def smoke(tmp_path_factory):
    root = str(tmp_path_factory.mktemp("mobjects"))
    res = b1.run_smoke("mobjects", root=root, keep=True)
    d = os.path.join(root, "work", "mobjects", "mobjects")
    return res, json.load(open(os.path.join(d, "store.json"))), f10.Store(d)


def test_registered():
    assert fam.REGISTRY["mobjects"] is mo.MObjectsAdapter
    ad = mo.MObjectsAdapter()
    assert ad.family == "13" and ad.C == 7 and ad.time_dtype == "int32"
    assert ad.channel_names == ["length_m", "width_m", "topview_area_m2",
                                "kg_lo", "kg_mid", "kg_hi", "lt50"]
    assert set(mo.TYPE_CODES.values()) == {
        "Bundled net", "Loose net", "Rope", "Buoy", "Container", "Life ring",
        "Unknown"}
    assert b1.layout_for(ad).hf_root == "tensors/family1_3"
    assert ad.licence["redistribution"] == "attribution"


def test_smoke_all_stages_and_the_probe(smoke):
    res, meta, st = smoke
    assert len(res["truth"]) == lb.EXPECT_OBJECTS == 1595
    assert st.N == 1595
    p = res["probe"]
    assert p["rows"] == 1595                    # every object is 2016-10
    assert p["bytes_fetched"] > 0
    c = meta["counts"]
    assert c["objects_read"] == 1595 and c["rows_kept"] == 1595
    # 1,595 objects cycling the seven types: 6 x 228 + 227
    want = {t: 228 for t in mo.TYPE_CODES.values()}
    want["Unknown"] = 227
    assert c["objects_by_type"] == want
    assert c["mosaic_check"]["mosaics"] == 31
    assert meta["per_year"] == {"2016": 1595}


def test_types_lt50_and_missing_area(smoke):
    _res, _meta, st = smoke
    qc = np.asarray(st["qc"])
    assert set((qc & 15).tolist()) == set(mo.TYPE_CODES)
    assert ((qc & mp.QC_NO_TIME) == mp.QC_NO_TIME).all()   # no flight hour
    v = np.asarray(st["values"], np.float32)
    assert set(np.unique(v[:, 6]).tolist()) == {0.0, 1.0}
    assert int(np.isnan(v[:, 2]).sum()) == 1                # one blank area
    assert (v[:, 3] <= v[:, 4]).all() and (v[:, 4] <= v[:, 5]).all()


def test_an_unknown_object_type_is_refused():
    L = {"events": {"1": {"y": 2016, "mo": 10, "d": 2, "hour": None,
                          "platform": "C-130"}},
         "objects": [{"event": "1", "type": "Fridge", "lat0": 0, "lat1": 0,
                      "lon0": 0, "lon1": 0, "length_m": 1, "width_m": 1,
                      "area_m2": 1, "kg_lo": 1, "kg_mid": 1, "kg_hi": 1,
                      "lt50": 0}]}
    with pytest.raises(mp.FormatError, match="Fridge"):
        mo.object_rows(L, {})


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))

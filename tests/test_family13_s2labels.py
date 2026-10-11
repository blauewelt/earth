#!/usr/bin/env python3
"""The `s2labels` adapter's smoke (family 1.3, E-092) — no network.

    python3 -m pytest -q tests/test_family13_s2labels.py

The synthetic archive (`ml/family1/adapters/_s2labels_smoke.py`) writes both
Zenodo records, both zips in their real layouts (MARIDA's georeferenced
11-band patches; MADOS's NON-georeferenced single-band crops at native
resolution with ACOLITE's per-satellite band names), the producers'
`assets.py` and the Copernicus catalogue's answers, all mirrored by URL. Every
expected count below is exact and comes from the generator's own grid
arithmetic.
"""
import json
import os
import sys

import numpy as np
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "ml"))

pytest.importorskip("rasterio")

import build_family1_stores as b1                               # noqa: E402
import family10_store as f10                                    # noqa: E402
from family1 import adapters as fam                             # noqa: E402
from family1.adapters import _s2labels_smoke as sm              # noqa: E402
from family1.adapters import s2labels as s2                     # noqa: E402

# 87 MARIDA labelled pixels (2 + 68 + 6 + 1 + 10), 33 of them also labelled by
# a located MADOS crop, + 8 MADOS-only pixels
TRUTH_ROWS = 87 + 8


@pytest.fixture(scope="module")
def smoke(tmp_path_factory):
    root = str(tmp_path_factory.mktemp("s2labels"))
    res = b1.run_smoke("s2labels", root=root, keep=True)
    d = os.path.join(root, "work", "s2labels", "s2labels")
    return res, json.load(open(os.path.join(d, "store.json"))), f10.Store(d)


def test_registered():
    assert fam.REGISTRY["s2labels"] is s2.S2LabelsAdapter
    ad = s2.S2LabelsAdapter()
    assert ad.family == "13" and ad.C == 14 and ad.time_dtype == "int32"
    assert ad.channel_names == [f"rho_{b}" for b in s2.BANDS] + [
        "confidence", "split", "s2_unit"]
    assert ad.licence["redistribution"] == "attribution"
    for who in ("MARIDA", "Kikaki", "2022", "MADOS", "2024",
                "Contains modified Copernicus Sentinel data"):
        assert who in ad.licence["attribution"]
    assert b1.layout_for(ad).hf_root == "tensors/family1_3"
    assert ad.log2_fp == pytest.approx(-11.4424, abs=1e-3)   # a 10 m pixel
    assert ad.log2_dt == pytest.approx(-18.7207, abs=1e-3)   # cat_s2's 1 s


def test_joint_class_table():
    """Same name -> same code; a class one set lacks gets its own code."""
    J = s2.JOINT_CLASSES
    assert {k: J[k] for k in range(1, 16)} == s2.MARIDA_CLASSES
    assert J[16] == "Sparse Floating Algae" and J[17] == "Oil Spill"
    assert J[18] == "Waves & Wakes" and max(J) == 21 <= s2.CLASS_BITS
    m = s2.TO_JOINT["MADOS"]
    assert (m[1], m[3], m[6], m[7], m[12], m[15]) == (1, 16, 17, 7, 18, 21)


def test_smoke_all_stages_and_the_probe(smoke):
    res, meta, st = smoke
    truth = res["truth"]
    assert len(truth) == st.N == TRUTH_ROWS
    p = res["probe"]
    assert p["rows"] == 83                      # every 2018-03 row
    assert p["bytes_fetched"] > 0
    assert {y: n for y, n in meta["per_year"].items() if n} == {
        "2016": 2, "2018": 83, "2019": 10}


def test_counts_overlap_precedence_and_label_changes(smoke):
    _res, meta, _st = smoke
    c = meta["counts"]
    E = sm.EXPECT
    assert c["marida_labelled_pixels"] == 87
    assert c["mados_labelled_pixels"] == E["mados_labelled"] == 55
    assert c["mados_crops"] == 4 and c["mados_crops_located"] == 3
    assert c["mados_crops_unlocated"] == 1
    assert c["mados_scenes"] == 3 and c["mados_scenes_located"] == 2
    assert c["mados_unlocated_by_class"] == E["unlocated"] == {"Oil Spill": 10}
    assert c["overlap_labels"] == E["overlap"] == {
        "Marine Water -> Marine Water": 18,
        "Marine Water -> Oil Spill": 6,
        "Sparse Sargassum -> Sparse Floating Algae": 1,
        "Marine Debris -> Marine Debris": 2,
        "Turbid Water -> Turbid Water": 6}
    assert c["overlap_pixels"] == 33 and c["overlap_label_changed"] == 7
    assert c["overlap_confidence_changed"] == E["conf_changed"] == 6
    assert c["mados_only_rows"] == E["mados_only"] == 8
    assert c["mados_duplicate_pixels"] == E["dup"] == 4
    assert c["mados_duplicate_pixels_class_differs"] == E["dup_diff"] == 2
    assert c["mados_only_inside_marida_patch_split"] == E["leak"] == {
        "inside a MARIDA test patch, MADOS train": 4}
    assert c["marida_unlabelled_by_covering_mados_crop"] == \
        E["kept_inside"] == {"Marine Water": 12, "Sparse Sargassum": 1,
                             "Marine Debris": 1, "Foam": 1}
    assert c["rows_by_dataset"] == {"MARIDA only": 54,
                                    "both (MADOS label)": 33,
                                    "MADOS only": 8}
    assert c["match_satellite_checked"] == 3
    assert c["match_min_votes_located"] >= s2.MATCH_MIN_VOTES
    assert c["max_unlocated_best_vote"] < s2.MATCH_MIN_VOTES
    assert c["catalogue_scenes_timed"] == 2
    assert c["catalogue_scenes_untimed"] == 1
    assert c["catalogue_products_per_scene"] == {"0": 1, "1": 1, "2": 1}
    assert c["out_of_bounds"] == {"rho_B12": 1, "rho_B03": 1}
    b = c["fdi_baseline_marida_archive"]
    assert b["train_water_pixels"] == 64 and b["test_pixels"] == 16
    assert b["test_debris_pixels"] == 6


def _rows(st):
    v = np.asarray(st["values"], np.float32)
    return v, np.asarray(st["qc"], np.uint8), st.time_s()


def test_rows_values_qc_and_split_mapping(smoke):
    res, _meta, st = smoke
    v, qc, ts = _rows(st)
    want = sorted(res["truth"], key=lambda r: (r["t"] // 432000, r["t"]))
    assert [int(q) for q in qc] == [r["qc"] for r in want]
    # every overlap row carries MADOS's values: 0.9 x + 0.003 of MARIDA's,
    # MARIDA's split, and both bits
    both = (qc & s2.QC_BOTH) != 0
    assert int(both.sum()) == 33
    assert ((qc[both] & s2.QC_MADOS) != 0).all()
    # the A1 debris pixel (MARIDA test patch) labelled by a MADOS TRAIN crop
    # keeps MARIDA's split 2; the 4 MADOS-only pixels inside that patch carry
    # MADOS's 0
    sp = v[:, s2.COL["split"]]
    only = ((qc & s2.QC_MADOS) != 0) & ~both
    assert sorted(sp[only].tolist()) == [0.0] * 8
    assert set(sp[both].tolist()) == {0.0, 2.0}
    assert int(((sp == 2) & ~only).sum()) == 6 + 10
    # the scene the catalogue does not answer: noon, bit 32, no satellite
    noon = (qc & s2.QC_NO_TIME) != 0
    assert int(noon.sum()) == 2
    assert (ts[noon] % 86400 == 43200).all()
    assert np.isnan(v[noon, s2.COL["s2_unit"]]).all()
    u = v[~noon, s2.COL["s2_unit"]]
    assert set(u.tolist()) == {1.0, 2.0}
    # S2B scene: 03:01:01 UTC from the catalogue's two copies of one product
    tt = ts[~noon]
    assert (tt[u == 2.0] % 86400 == 3 * 3600 + 61).all()
    assert (tt[u == 1.0] % 86400 == 16 * 3600 + 8 * 60 + 19).all()
    # out of bounds -> NaN, exactly two values in the store
    assert int(np.isnan(v[:, :11]).sum()) == 2
    # class codes: MADOS's Oil Spill and Sparse Floating Algae are present
    cls = qc & s2.CLASS_BITS
    assert int((cls == 17).sum()) == 6 and int((cls == 16).sum()) == 1


def test_rank_codes_survive_a_reprocessing():
    rng = np.random.default_rng(3)
    a = rng.uniform(0.01, 0.3, (4, 20, 20)).astype(np.float32)
    b = (np.float32(1.3) * a + np.float32(0.02)) ** 2      # increasing map
    assert np.array_equal(s2.rank_codes(a), s2.rank_codes(b))
    idx = s2.CodeIndex(s2.rank_codes(a)[None])
    votes = idx.votes(s2.rank_codes(b[:, 3:17, 5:19]))
    (p, dr, dc), n = max(votes.items(), key=lambda kv: kv[1])
    assert (p, dr, dc) == (0, 3, 5) and n >= 100


def test_catalogue_answers():
    one = {"value": [{"Name": "S2B_MSIL1C_20190907T030101_N0500_R075_T48MYU_"
                              "1.SAFE",
                      "ContentDate": {"Start": "2019-09-07T03:01:01.024Z"}}]}
    sat, t, n = s2.cdse_answer(one)
    assert sat == "S2B" and n == 1 and t % 86400 == 3 * 3600 + 61
    two = {"value": one["value"] + [{"Name": "S2A_MSIL1C_x.SAFE",
                                     "ContentDate": {
                                         "Start": "2019-09-07T03:11:01Z"}}]}
    assert s2.cdse_answer(two)[:2] == (None, None)
    assert s2.cdse_answer({"value": []}) == (None, None, 0)


def test_a_changed_class_table_is_refused():
    rec = {"metadata": {"license": {"id": "cc-by-4.0"}, "description": (
        "The mapping between Digital Numbers and Classes is: 1: Marine Debris "
        "2: Plastic The mapping between Digital Numbers and Confidence")}}
    with pytest.raises(s2.FormatError, match="class table"):
        s2.check_record("MARIDA", rec)


def test_the_measured_archive_must_keep_its_measured_counts():
    files = {ds: {"md5": m["md5"]} for ds, m in s2.MEASURED.items()}
    good = {"marida_patches": 1381, "marida_scenes": 63,
            "marida_labelled_pixels": 837377,
            "marida_by_class": {"Marine Debris": 3399},
            "mados_crops": 2803, "mados_scenes": 174,
            "mados_labelled_pixels": 1481155}
    s2.check_measured(files, dict(good))
    with pytest.raises(s2.FormatError, match="marida_debris"):
        s2.check_measured(files, dict(good, marida_by_class={
            "Marine Debris": 3398}))
    other = {"MARIDA": {"md5": "0" * 32}, "MADOS": {"md5": "1" * 32}}
    c = {}
    s2.check_measured(other, c)              # another archive: recorded only
    assert c == {"marida_is_the_measured_archive": 0,
                 "mados_is_the_measured_archive": 0}


def test_fdi_baseline_arithmetic():
    v = np.full((4, s2.C), np.nan)
    v[:, :11] = 0.05
    v[:, 7] = [0.05, 0.05, 0.30, 0.05]           # R8: one bright pixel
    v[:, s2.COL["s2_unit"]] = [1, 1, 2, 2]
    qc = np.array([7, 7, 1, 1], np.uint8)
    f = s2.fdi(v, v[:, s2.COL["s2_unit"]])
    assert f[2] == pytest.approx(0.30 - (0.05 + 0.0 * 1.7770))
    b = s2.fdi_baseline(v, qc, np.array([1, 1, 0, 0], bool),
                        np.array([0, 0, 1, 1], bool))
    assert (b["tp"], b["fp"], b["fn"]) == (1, 0, 1)
    assert b["f1"] == pytest.approx(2 / 3)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))

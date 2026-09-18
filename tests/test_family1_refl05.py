#!/usr/bin/env python3
"""The `refl05` adapter's smoke (family 1.0.tf, MOD09CMG v061) — no network.

    python3 -m pytest -q tests/test_family1_refl05.py

The synthetic archive (`refl05.make_smoke_sources`) is the real layout: a CMR
`granules.json` the REAL parser reads, plus HDF-EOS2 granules carrying the
real SDS names — including the ones with SPACES in them, "Coarse Resolution
Surface Reflectance Band 1" — and the real int16 scaling 0.0001, fill 0 and
valid_range -100..16000, on a 600 x 300 grid (`REFL05_SMOKE_GRID`). The
record is 2015-07-01 .. 2015-07-20 with 07-13 not listed.

The two decisions this store turns on are tested and not asserted: the uint16
state QA word survives as two float16-exact bytes (C = 9, where the ledger
row plans C = 8), and both QA bytes are NaN exactly where all seven bands are
fill.
"""
import argparse
import datetime as dt
import json
import os
import sys

import numpy as np
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "ml"))

import build_family10_stores as b10                             # noqa: E402
import build_family1_stores as b1                               # noqa: E402
from family1 import adapters as fam                             # noqa: E402
from family1 import sharded as sh                               # noqa: E402
from family1.adapters import _modis_cmg as mc                   # noqa: E402
from family1.adapters import refl05 as rf                       # noqa: E402

pytest.importorskip("pyhdf")

ENV = ("REFL05_SMOKE_GRID", "REFL05_GROUPS")


def clear_env(monkeypatch):
    for k in ENV:
        monkeypatch.delenv(k, raising=False)


@pytest.fixture(scope="module")
def smoke(tmp_path_factory):
    root = str(tmp_path_factory.mktemp("refl05_smoke"))
    keep = {k: os.environ.get(k) for k in ENV}
    try:
        return b1.run_smoke("refl05", root=root, keep=True)
    finally:
        for k, v in keep.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


def small_archive(tmp, **kw):
    src = os.path.join(tmp, "src")
    d_lo, d_hi = (b10.parse_date(x) for x in rf.Refl05Adapter.smoke_window)
    truth = rf.make_smoke_sources(src, d_lo, d_hi, **kw)
    return src, truth


def ctx_for(tmp, src, **over):
    w = rf.Refl05Adapter.smoke_window
    d = dict(store="refl05", work=os.path.join(tmp, "work"), source_dir=src,
             start=w[0], end=w[1], stage="all", force=False,
             attempts=1, qc_keep=2, check_chunk_rows=b10.CHECK_CHUNK_ROWS,
             assemble="auto", parts_from_hub=False, push_parts=False,
             allow_missing_years=False, allow_unconfirmed_licence=False,
             probe_month="", smoke=False)
    d.update(over)
    ad = rf.Refl05Adapter()
    ctx = b10.Ctx(argparse.Namespace(**d), adapter=ad,
                  layout=b1.layout_for(ad))
    return b1.prepare_grid_ctx(ctx)


def run(ctx, stages):
    b10.run_stages(ctx, stages, stage_fn=b1.GRID_STAGE_FN, deps=b1.DEPS)


# ================================================================ facts ====
def test_registered_with_nine_channels(monkeypatch):
    clear_env(monkeypatch)
    assert fam.REGISTRY["refl05"] is rf.Refl05Adapter
    ad = rf.Refl05Adapter()
    assert b1.is_grid(ad) and ad.layout == "sharded"
    assert ad.family == "1tf" and ad.distribution == "public"
    assert ad.credentials == ("EARTHDATA_USERNAME", "EARTHDATA_PASSWORD")
    assert ad.channel_names == ["b1", "b2", "b3", "b4", "b5", "b6", "b7",
                                "state_qa_lo", "state_qa_hi"]
    assert ad.C == 9 and ad.dtype == "float16"
    assert (ad.frames_per_bin, ad.frame_seconds) == (5, 86400)
    sp = ad.specs()["terra"]
    assert (sp["H"], sp["W"], sp["C"]) == (3600, 7200, 9)
    assert sp["grid"]["y0"] == 90.0
    assert "C = 9" in sp["channels_note"]
    assert sp["state_qa_bits"]["3-5"].startswith("land/water")
    assert "state_qa_hi << 8" in sp["state_qa_rule"]
    # the SDS names carry spaces, which is what the reader must ask for
    assert ad.sds[0] == "Coarse Resolution Surface Reflectance Band 1"
    assert ad.sds[-1] == "Coarse Resolution State QA"


def test_the_qa_word_could_not_have_been_one_float16_channel():
    """The measurement behind C = 9: float16 represents CONSECUTIVE integers
    exactly only to 2,048, so most uint16 words do not survive it."""
    for v in (0, 255, 2048):
        assert float(np.float16(v)) == float(v)
    for v in (2049, 4097, 40001, 65535):
        assert float(np.float16(v)) != float(v)
    # the two BYTES are exact, and reassemble the word
    for word in (0, 1, 255, 256, 4097, 40000, 65535):
        lo, hi = word & 0xFF, word >> 8
        assert float(np.float16(lo)) == lo and float(np.float16(hi)) == hi
        assert (int(np.float16(hi)) << 8 | int(np.float16(lo))) == word


def test_the_reflectance_bounds_carry_one_float16_step_of_headroom():
    """float16's step at reflectance 1.6 is 0.00098 — ten times the source's
    own 0.0001 — so a stored value can sit up to half a step either side of
    the producer's range endpoint. The bounds are widened by a whole step so
    the store's own tile check cannot fail on a legitimate extreme whichever
    way the rounding goes."""
    step = float(np.nextafter(np.float16(1.6), np.float16(10))) - \
        float(np.float16(1.6))
    assert 9e-4 < step < 1.1e-3
    assert step > 9 * rf.SCALE
    ad = rf.Refl05Adapter()
    lo, hi = ad.bounds()
    top, bottom = 16000 * rf.SCALE, -100 * rf.SCALE
    assert hi[0] - top >= step and bottom - lo[0] >= step
    # every band's stored extreme is inside its bound, both ways
    for raw in (-100, -99, 0, 1, 15999, 16000):
        assert lo[0] <= float(np.float16(raw * rf.SCALE)) <= hi[0]
    # the QA bytes' bounds ARE float16-exact, because they are integers
    assert float(np.float16(lo[7])) == 0.0
    assert float(np.float16(hi[7])) == 255.0


# ============================================================== the smoke ==
def test_smoke_reads_every_frame_back_exactly(smoke):
    res = smoke["check"]
    assert res["frames_equal"] == 19            # 20 listed days less 07-13
    assert res["by_reason"]["absent_upstream"] == 1
    sm = json.load(open(os.path.join(smoke["ctx"].store, "store.json")))
    assert sm["C"] == 9 and sm["dtype"] == "float16"
    assert list(sm["groups"]) == ["terra"]
    assert sm["counts"]["pixels_no_band_retrieved"] > 0


def test_the_qa_bytes_are_nan_exactly_where_no_band_was_retrieved(smoke):
    grp = sh.ShardedGroup(os.path.join(smoke["ctx"].store, "terra"))
    arr = grp.shard_index
    b = int(arr["bin"][1])
    a = None
    for f in range(5):
        a = grp.read_frame(b, f)
        if a is not None:
            break
    assert a is not None
    bands = a[:, :, :7]
    lo, hi = a[:, :, 7], a[:, :, 8]
    nothing = np.isnan(bands).all(axis=2)
    assert nothing.any() and (~nothing).any()
    assert np.isnan(lo[nothing]).all() and np.isnan(hi[nothing]).all()
    assert np.isfinite(lo[~nothing]).all() and np.isfinite(hi[~nothing]).all()
    # the word reassembles to the value the smoke wrote
    word = (hi[~nothing].astype(np.int64) << 8) | lo[~nothing].astype(np.int64)
    assert set(np.unique(word)) <= {0b1000_0001_0100_1001,
                                    0b0000_0100_0000_0001}
    # a ZERO qa byte is kept where a band WAS retrieved: it is a reading
    assert float(np.nanmin(lo)) >= 0.0


def test_only_the_tiles_with_a_retrieval_are_stored(smoke):
    """Unlike lst05, refl05's quality is masked with its bands, so an
    all-fill tile costs nothing at all."""
    grp = sh.ShardedGroup(os.path.join(smoke["ctx"].store, "terra"))
    arr = grp.shard_index
    sp = grp.spec
    full = int(arr["frames_present"].sum()) * sp["n_tiles_y"] \
        * sp["n_tiles_x"]
    assert 0 < int(arr["tiles_stored"].sum()) < full


def test_the_probe_measured_the_same_frames(smoke):
    p = smoke["probe"]
    assert p["frames_fetched"] == 19
    assert p["out_of_bounds_stored"] == 0
    gp = p["groups"]["terra"]
    assert gp["C"] == 9
    assert gp["frames_missing"]["absent_upstream"] == 1
    assert gp["record_frames"] == 19
    for b in ("b1", "b4", "b7"):
        assert 0.0 < gp["valid_fraction"][b] < 1.0
    assert gp["valid_fraction"]["state_qa_lo"] == \
        gp["valid_fraction"]["state_qa_hi"]
    assert gp["tiles_empty"] >= 0


def test_a_declared_attribute_that_differs_is_a_refusal(tmp_path):
    src, _ = small_archive(str(tmp_path))
    gdir = os.path.join(src, "refl05", "granules", "terra")
    name = sorted(n for n in os.listdir(gdir) if n.endswith(".hdf"))[-1]
    mc.set_sds_attrs(os.path.join(gdir, name),
                     {rf.BANDS[0]: {"scale_factor": 0.001}})
    ctx = ctx_for(str(tmp_path), src)
    with pytest.raises(SystemExit) as e:
        run(ctx, ["index"])
    assert "scale_factor" in str(e.value)


def test_a_listed_granule_that_is_not_there_is_an_absence(tmp_path):
    src, _ = small_archive(str(tmp_path), absent=(dt.date(2015, 7, 8),))
    ctx = ctx_for(str(tmp_path), src)
    with pytest.raises(SystemExit) as e:
        run(ctx, ["index", "fetch"])
    assert "could not be read" in str(e.value)


def test_check_store_verifies_every_stored_tile(smoke):
    st = sh.check_store(smoke["ctx"].store)
    g = st["groups"]["terra"]
    assert g["tiles_checked"] == g["tiles_stored"] > 0
    assert g["frames_present"] == 19

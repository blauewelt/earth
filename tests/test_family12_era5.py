#!/usr/bin/env python3
"""Family 1.2's ERA5 atmosphere stores (E-085) — no network.

    python3 -m pytest -q tests/test_family12_era5.py

`era5_t`, `era5_q`, `era5_u` and `era5_v` are sharded tier-G stores: one
ERA5 variable each, the 13 standard pressure levels folded into the channel
axis, one frame per six-hourly analysis instant on a 1-degree grid. The
source is two zarr archives on public Cloud Storage — the producer's own
1-degree regrid until it ends, then the 0.25-degree archive regridded here —
read by fetching only the blosc blocks that hold the wanted levels.

The synthetic archive is written by `_era5.write_smoke_archive` in the real
archives' layout (consolidated zarr v2, blosc-lz4 + shuffle, the same
dimension orders and time units, a 0.25-degree part with a pre-allocated
time axis and `valid_time_stop`), on a 10-degree / 2.5-degree grid, with
three extra pressure levels the store must skip and its seam placed INSIDE a
five-day bin.
"""
import argparse
import datetime as dt
import math
import os
import shutil
import sys

import numpy as np
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "ml"))

import build_family10_stores as b10                             # noqa: E402
import build_family1_stores as b1                               # noqa: E402
from family1 import sharded as sh                               # noqa: E402
from family1 import adapters as fam                             # noqa: E402
from family1.adapters import _era5 as e5                        # noqa: E402

STORES = ("era5_t", "era5_q", "era5_u", "era5_v")


@pytest.fixture(autouse=True)
def _grid(monkeypatch):
    """Every test sees the smoke grid; nothing leaks into other modules."""
    monkeypatch.setenv("ERA5_SMOKE_DEG", f"{e5.SMOKE_DEG:g}")
    monkeypatch.delenv("ERA5_WORKERS", raising=False)


@pytest.fixture(scope="module")
def archive(tmp_path_factory):
    root = str(tmp_path_factory.mktemp("era5src"))
    base = e5.write_smoke_archive(root, e5.SMOKE_DEG)
    return root, base


def sources(base):
    return e5.Sources(base, True, e5.SMOKE_DEG)


# ============================================================ declaration ==
def test_four_stores_registered_as_family_1_2_tier_g():
    assert fam.FAMILIES["12"] == ("family1_2", "1.2", "family1_2")
    for s, v in zip(STORES, "tquv"):
        cls = fam.REGISTRY[s]
        ad = cls()
        assert ad.family == "12" and ad.tier == "G" and ad.layout == "sharded"
        assert ad.distribution == "public"
        assert ad.licence["redistribution"] == "attribution"
        assert "Copernicus Climate Change Service" in ad.licence["attribution"]
        assert ad.dtype == "float16" and ad.C == 13
        assert ad.frames_per_bin == 20 and ad.frame_seconds == 21600
        assert ad.frames_per_bin * ad.frame_seconds == sh.BIN_SECONDS
        # the CHANNEL NAME ENCODES THE LEVEL, in the declared ascending order
        assert ad.channel_names == [f"{v}_{p}" for p in e5.LEVELS_HPA]
        assert list(ad.levels_hpa) == [50, 100, 150, 200, 250, 300, 400,
                                       500, 600, 700, 850, 925, 1000]
        # the class carries its channels too, before any instance exists
        assert [c[0] for c in cls.channels] == ad.channel_names
        sp = ad.specs()[s]
        assert sp["levels_hpa"] == list(e5.LEVELS_HPA)
        assert sp["exception"] == "E4"
        assert ad.credentials == ()
        lay = b1.layout_for(ad)
        assert lay.hf_root == "tensors/family1_2"
        assert lay.hf_partials == "partials/family1_2"
        assert lay.repo_id == b1.PUBLIC_REPO
    assert fam.REGISTRY["era5_q"]().channels[0][1].startswith("g/kg")


def test_the_real_grid_is_point_aligned_181_by_360(monkeypatch):
    monkeypatch.delenv("ERA5_SMOKE_DEG", raising=False)
    ad = fam.REGISTRY["era5_t"]()
    g = ad.grid
    assert (g["H"], g["W"]) == (181, 360)
    lat, lon = e5.target_axes(1.0)
    assert lat[0] == -90.0 and lat[-1] == 90.0
    assert lon[0] == -180.0 and lon[-1] == 179.0
    # x0/y0 are half-cell edges, so the shared rule lon = x0 + (col+0.5)dx
    # lands on the points
    assert g["x0"] + 0.5 * g["dx"] == -180.0
    assert g["y0"] + 0.5 * g["dy"] == -90.0
    sp = ad.specs()["era5_t"]
    # 64-degree tiles: 3 rows x 6 columns of 64 x 64
    assert (sp["tile"], sp["n_tiles_y"], sp["n_tiles_x"]) == (64, 3, 6)
    assert abs(ad.log2_fp - 2.0) < 1e-3
    assert abs(ad.log2_dt - math.log2(0.25 / 5)) < 1e-12


# ============================================================ frame timing ==
def test_frame_f_of_bin_b_is_the_instant_at_00_06_12_18():
    for b in (0, 2411, 2921, 2922, 3245):
        start = e5.EPOCH + dt.timedelta(seconds=b * sh.BIN_SECONDS)
        for f in range(20):
            w = e5.frame_instant(b, f)
            assert w == start + dt.timedelta(hours=6 * f)
            assert w.hour in (0, 6, 12, 18) and w.minute == 0
            assert sh.frame_day(b, f, 21600) == w.date()
    # the epoch is bin 0 frame 0, and the real seam (the first hour the
    # 1-degree archive lacks, 2022-01-01T00) is exactly the start of bin 2922
    assert e5.frame_instant(0, 0) == dt.datetime(1982, 1, 1)
    assert e5.frame_instant(2922, 0) == dt.datetime(2022, 1, 1)
    # a January has 31 x 4 = 124 frames whose instant falls in it
    lo, hi = b10.month_bounds_s(2015, 1)
    n = sum(1 for b in sh.bins_overlapping(lo, hi) for f in range(20)
            if lo <= sh.frame_start_seconds(b, f, 21600) <= hi)
    assert n == 124


def test_the_record_ends_at_valid_time_stop_and_the_seam_is_measured(archive):
    _root, base = archive
    s = sources(base)
    # the seam is READ from the 1-degree part's time axis, not typed in
    assert s.seam == e5.SMOKE_A[1] + dt.timedelta(hours=1)
    assert s.where(s.seam - dt.timedelta(hours=6))[0] == "A"
    assert s.where(s.seam)[0] == "B"
    # the 0.25-degree axis runs past valid_time_stop (pre-allocated, as the
    # real one does to 2050); after the stop is after_record, not a read
    stop = dt.datetime.combine(e5.SMOKE_B_VALID[1], dt.time(18))
    assert s.where(stop)[0] == "B"
    assert s.where(stop + dt.timedelta(hours=6)) == (None, "after_record")
    assert s.where(e5.SMOKE_A[0] - dt.timedelta(hours=6)) == \
        (None, "before_record")


# ============================================================ level order ==
def test_channel_k_is_level_k_and_the_extra_levels_are_skipped(archive):
    _root, base = archive
    s = sources(base)
    assert len(e5.SMOKE_LEVELS) == 16                 # 13 + three to skip
    assert [e5.SMOKE_LEVELS[i] for i in s.a_lev] == list(e5.LEVELS_HPA)
    w = dt.datetime(2021, 12, 24, 6)
    src, ti = s.where(w)
    assert src == "A"
    arr, st = s.frame("temperature", src, ti)
    assert arr.shape == (19, 36, 13)
    h = (w - e5.SMOKE_A[0]).total_seconds() / 3600
    lat, lon = e5.target_axes(e5.SMOKE_DEG)
    for k, p in enumerate(e5.LEVELS_HPA):
        want = e5.smoke_value("t", float(p), lat[:, None],
                              (lon[None, :] % 360.0), h).astype(np.float32)
        # the 1-degree path is a REINDEX: row 0 is the south pole, column 0
        # is -180, and the values are the archive's float32 exactly
        assert np.array_equal(arr[:, :, k].astype(np.float32), want), p
    assert st["blocks"] >= 1 and st["bytes"] > 0


# ================================================================ regrid ===
def _overlap_weights_by_brute_force(slat_asc, slon, tlat, tlon):
    """2-d conservative weights by explicit spherical-rectangle areas — no
    code shared with the adapter."""
    def lat_edges(p):
        e = [-90.0] + [(p[i] + p[i + 1]) / 2 for i in range(len(p) - 1)]
        return e + [90.0]

    def lon_edges(p):
        n = len(p)
        out = []
        for i in range(n):
            lo = (p[i] + (p[i - 1] - (360.0 if i == 0 else 0.0))) / 2
            hi = (p[i] + (p[(i + 1) % n] + (360.0 if i == n - 1 else 0.0))) \
                / 2
            out.append((lo, hi))
        return out
    se, te = lat_edges(list(slat_asc)), lat_edges(list(tlat))
    sl, tl = lon_edges(list(slon)), lon_edges(list(tlon))
    W = np.zeros((len(tlat), len(tlon), len(slat_asc), len(slon)))
    for i in range(len(tlat)):
        for j in range(len(tlon)):
            for a in range(len(slat_asc)):
                y0, y1 = max(te[i], se[a]), min(te[i + 1], se[a + 1])
                if y1 <= y0:
                    continue
                dy = math.sin(math.radians(y1)) - math.sin(math.radians(y0))
                for c in range(len(slon)):
                    dx = 0.0
                    for k in (-360.0, 0.0, 360.0):
                        x0 = max(tl[j][0], sl[c][0] + k)
                        x1 = min(tl[j][1], sl[c][1] + k)
                        if x1 > x0:
                            dx += x1 - x0
                    W[i, j, a, c] = dx * dy
    return W / W.sum(axis=(2, 3), keepdims=True)


def test_the_regrid_is_the_conservative_box_mean_on_a_toy():
    tlat, tlon = e5.target_axes(30.0)                 # 7 x 12
    slat = np.arange(-90.0, 90.01, 7.5)              # 25, ascending
    slon = np.arange(0.0, 360.0, 7.5)                # 48
    rng = np.random.default_rng(20261005)
    field = rng.normal(250.0, 20.0, (len(slat), len(slon)))
    got = e5.apply_weights(field, e5.lat_weights(slat, tlat),
                           e5.lon_weights(slon, tlon))
    W = _overlap_weights_by_brute_force(slat, slon, tlat, tlon)
    want = np.einsum("ijac,ac->ij", W, field)
    assert got.shape == (7, 12)
    np.testing.assert_allclose(got, want, rtol=1e-12, atol=0)
    # the pole rows are half-cells: [-90, -75] at 30 degrees, and the
    # 7.5-degree source cell centred on -75 straddles the edge, half in
    assert e5._lat_edges(tlat)[:2] == [-90.0, -75.0]
    # a constant is reproduced to the last bit float16 can see
    c = e5.apply_weights(np.full_like(field, 287.3), e5.lat_weights(
        slat, tlat), e5.lon_weights(slon, tlon))
    assert np.abs(c - 287.3).max() < 1e-12
    assert np.array_equal(c.astype(np.float16),
                          np.full(c.shape, 287.3).astype(np.float16))
    # and a NaN anywhere in a box makes that box NaN, never a partial mean
    f2 = field.copy()
    f2[12, 5] = np.nan
    g2 = e5.apply_weights(f2, e5.lat_weights(slat, tlat),
                          e5.lon_weights(slon, tlon))
    bad = np.argwhere(np.isnan(g2))
    assert len(bad) >= 1
    for i, j in bad:
        assert W[i, j, 12, 5] > 0
    assert np.isfinite(g2[W[:, :, 12, 5] == 0]).all()


def test_the_regrid_needs_no_matrix_library_and_is_bitwise_stable():
    tlat, tlon = e5.target_axes(10.0)
    slat = np.arange(-90.0, 90.01, 2.5)
    slon = np.arange(0.0, 360.0, 2.5)
    f = np.cos(np.radians(slat))[:, None] * np.sin(np.radians(slon))[None]
    wl, wo = e5.lat_weights(slat, tlat), e5.lon_weights(slon, tlon)
    a = e5.apply_weights(f, wl, wo)
    b = e5.apply_weights(f.copy(order="F"), wl, wo)
    assert a.tobytes() == b.tobytes()
    # every weight row sums to one
    for idx, w in (wl, wo):
        assert np.abs(w.sum(axis=1) - 1.0).max() < 1e-15


# ======================================================= humidity in g/kg ==
def test_specific_humidity_in_g_per_kg_keeps_float16_normal():
    q = np.geomspace(1e-7, 3e-2, 20001)               # kg/kg, the record's
    stored = (q * 1000.0).astype(np.float16)          # what the store keeps
    back = stored.astype(np.float64) / 1000.0
    rel = np.abs(back - q) / q
    # every value from 1e-7 kg/kg up is a float16 NORMAL in g/kg ...
    assert (q * 1000.0 >= e5.F16_TINY).all()
    # ... so the round trip is within half a float16 step: 2^-11
    assert rel.max() <= 2.0 ** -11 * (1 + 1e-12)
    # in kg/kg, the stratosphere's ~2e-6 would be a subnormal with ~1.5 %
    # error — the measurement that forced the transform
    strat = np.geomspace(1e-6, 5e-6, 101)
    raw = strat.astype(np.float16).astype(np.float64)
    assert (strat < e5.F16_TINY).all()
    assert (np.abs(raw - strat) / strat).max() > 0.005


def test_the_q_store_scales_by_1000_and_counts_float16_error(tmp_path):
    res = b1.run_smoke("era5_q", root=str(tmp_path / "s"), keep=True,
                       probe=False)
    grp = sh.ShardedGroup(os.path.join(res["work"], "era5_q", "era5_q",
                                       "era5_q"))
    b, f = 2920, 8                                    # 2021-12-24T00, from A
    w = e5.frame_instant(b, f)
    got = grp.read_frame(b, f)
    h = (w - e5.SMOKE_A[0]).total_seconds() / 3600
    lat, lon = e5.target_axes(e5.SMOKE_DEG)
    k = list(e5.LEVELS_HPA).index(50)
    kgkg = e5.smoke_value("q", 50.0, lat[:, None], lon[None, :] % 360.0,
                          h).astype(np.float32).astype(np.float64)
    np.testing.assert_allclose(got[:, :, k], kgkg * 1000.0, rtol=2 ** -11)
    import json
    sm = json.load(open(os.path.join(res["work"], "era5_q", "era5_q",
                                     "store.json")))
    c = sm["counts"]
    assert c["max_f16err_rel_q_50"] <= 2.0 ** -11 * (1 + 1e-9)
    assert "f16_subnormal_values" not in c


# ===================================================== the block reader ====
def test_reading_only_the_needed_blocks_equals_decoding_the_chunk(tmp_path):
    """Multi-threaded blosc writes blocks OUT OF INDEX ORDER; the reader must
    find each block's bytes from the sorted start table."""
    from numcodecs import Blosc, blosc
    blosc.set_nthreads(4)
    rng = np.random.default_rng(7)
    data = rng.normal(0, 1, (8, 16, 36, 19)).astype(np.float32)
    root = str(tmp_path)
    e5._write_zarr(root, "z.zarr", {
        "x": {"data": data, "chunks": [8, 16, 36, 19],
              "dims": ["time", "level", "longitude", "latitude"]}})
    raw = open(os.path.join(root, "z.zarr", "x", "0.0.0.0"), "rb").read()
    full = np.frombuffer(Blosc().decode(raw), "<f4").reshape(data.shape)
    assert np.array_equal(full, data)
    h = e5.parse_header(raw[:e5.HEADER_PROBE], data.nbytes, len(raw), "x")
    assert h["nblocks"] > 4
    z = e5.Zarr(e5.Archive(root, True), "z.zarr")
    for t in (0, 3, 7):
        for lev in ([0], [2, 3, 9], list(range(16))):
            got, st = z.read_levels("x", t, lev)
            assert np.array_equal(got, data[t, lev]), (t, lev)
            assert st["bytes"] <= len(raw) + e5.HEADER_PROBE


def test_a_header_that_disagrees_with_the_zarr_refuses():
    from numcodecs import Blosc
    a = np.arange(4096, dtype=np.float32)
    raw = Blosc(cname="lz4", clevel=5, shuffle=1).encode(a)
    e5.parse_header(raw[:4096], a.nbytes, len(raw), "ok")
    with pytest.raises(e5.FormatError, match="uncompressed bytes"):
        e5.parse_header(raw[:4096], a.nbytes * 2, len(raw), "x")
    with pytest.raises(e5.FormatError, match="compressed bytes"):
        e5.parse_header(raw[:4096], a.nbytes, len(raw) + 1, "x")


def test_a_200_where_a_206_was_asked_for_is_refused(monkeypatch):
    class R:
        status_code = 200
        content = b"x" * 100
        headers = {}

    class S:
        def get(self, *a, **k):
            return R()
    ar = e5.Archive("https://example.invalid", False, attempts=1)
    monkeypatch.setattr(ar, "_session", lambda: S())
    with pytest.raises(IOError, match="refusing to read the whole object"):
        ar.read("a/b", 0, 10)


def test_a_short_range_is_refused(monkeypatch):
    class R:
        status_code = 206
        content = b"x" * 5
        headers = {"Content-Range": "bytes 0-4/1000"}

    class S:
        def get(self, *a, **k):
            return R()
    ar = e5.Archive("https://example.invalid", False, attempts=1)
    monkeypatch.setattr(ar, "_session", lambda: S())
    with pytest.raises(IOError, match="returned 5 bytes"):
        ar.read("a/b", 0, 10)


# ================================================= a missing chunk refuses ==
def _ctx(src, work, store, start, end):
    a = argparse.Namespace(
        store=store, work=work, source_dir=src, start=start, end=end,
        stage="index,fetch", force=False, attempts=1, qc_keep=2,
        check_chunk_rows=b10.CHECK_CHUNK_ROWS, assemble="auto",
        parts_from_hub=False, push_parts=False, allow_missing_years=False,
        allow_unconfirmed_licence=False, probe_month="", smoke=True)
    ad = fam.REGISTRY[store]()
    ctx = b10.Ctx(a, adapter=ad, layout=b1.layout_for(ad))
    b1.prepare_grid_ctx(ctx)
    return ctx


@pytest.mark.parametrize("damage", ["delete", "truncate"])
def test_a_missing_or_short_chunk_refuses_the_year(tmp_path, damage):
    src = str(tmp_path / "src")
    e5.write_smoke_archive(src, e5.SMOKE_DEG)
    # the 0.25-degree chunk of 2022-01-02T06, inside the valid record
    s = sources(os.path.join(src, e5.SOURCE_SUBDIR))
    src_, ti = s.where(dt.datetime(2022, 1, 2, 6))
    assert src_ == "B"
    p = os.path.join(src, e5.SOURCE_SUBDIR, e5.ZARR_B, "temperature",
                     f"{ti}.0.0.0")
    assert os.path.exists(p)
    if damage == "delete":
        os.remove(p)
    else:
        b = open(p, "rb").read()
        open(p, "wb").write(b[:len(b) // 2])
    ctx = _ctx(src, str(tmp_path / "work"), "era5_t", "2022-01-01",
               "2022-01-04")
    b10.stage_index(ctx)
    ad = ctx.adapter
    ad.fetch_preflight(ctx)
    assert b1.fetch_grid_year(ctx, 2022) is None      # NOT marked
    assert not b1.marked(ctx.root, ctx.part_key(2022))
    assert len(ctx.absent) == 1
    assert "2022-01-02T06Z" in ctx.absent[0]["unit"]
    with pytest.raises(SystemExit):
        b10.fetch_absence_check(ctx)


# ============================================================ the smoke ====
def test_smoke_all_frames_across_the_seam_and_the_probe(tmp_path):
    res = b1.run_smoke("era5_t", root=str(tmp_path / "s"), keep=True)
    truth = res["truth"]
    # the window's four bins: 2920 .. 2923, twenty frames each
    assert len(truth) == 80
    chk = res["check"]
    # 2022-01-06 onward is past valid_time_stop: bin 2923 is all
    # after_record and is not written; bin 2922 ends 2022-01-05T18
    assert chk["frames_equal"] == 60
    assert chk["by_reason"] == {"after_record": 20}
    assert chk["frames_in_skipped_bins"] == 20
    import json
    sm = json.load(open(os.path.join(res["work"], "era5_t", "era5_t",
                                     "store.json")))
    c = sm["counts"]
    # THE SEAM IS INSIDE BIN 2921: 2021-12-30T00 onward comes from B
    seam = e5.SMOKE_A[1] + dt.timedelta(hours=1)
    n_a = sum(1 for (g, b, f), (arr, why) in truth.items()
              if arr is not None and e5.frame_instant(b, f) < seam)
    n_b = sum(1 for (g, b, f), (arr, why) in truth.items()
              if arr is not None and e5.frame_instant(b, f) >= seam)
    assert c["frames_from"] == {"arco_1deg_conservative": n_a,
                                "arco_025_regridded_here": n_b}
    assert n_a == 32 and n_b == 28
    assert e5.frame_instant(2921, 0) < seam < e5.frame_instant(2922, 0)
    # the one value written out of bounds became NaN and was COUNTED
    assert c["out_of_bounds"] == {"t_500": 1}
    w, lev = e5.SMOKE_OOB
    b, rem = divmod(int((w - e5.EPOCH).total_seconds()), sh.BIN_SECONDS)
    grp = sh.ShardedGroup(os.path.join(res["work"], "era5_t", "era5_t",
                                       "era5_t"))
    fr = grp.read_frame(b, rem // 21600)
    assert int(np.isnan(fr).sum()) == 1
    assert np.isnan(fr[:, :, list(e5.LEVELS_HPA).index(lev)]).sum() == 1
    # the probe measured the 20 valid frames of 2022-01 and counted bytes
    assert res["probe"]["frames_fetched"] == 20
    assert res["probe"]["bytes_fetched"] > 0


def test_the_two_sources_agree_at_the_seam_to_the_fields_own_scale(archive):
    """Both synthetic parts sample ONE analytic atmosphere, so at an instant
    both hold, the regridded 2.5-degree part and the 10-degree part differ
    only by what a box mean does to a smooth field — the same check the
    probe makes on the real archive, where the producer's regrid and this
    module's agree to 5e-4 K."""
    _root, base = archive
    s = sources(base)
    w = e5.SMOKE_A[1] - dt.timedelta(hours=5)         # 2021-12-29T18
    assert w.hour == 18
    a, _ = s.frame("temperature", "A", s.where(w)[1])
    h = int((w - s.b_t0).total_seconds() // 3600)
    b, _ = s.frame("temperature", "B", h - int(s.b_t[0]))
    d = np.abs(a - b)
    # away from the poles a 10-degree box mean of 25 cos(lat) differs from
    # the point value by < 0.5 K; the levels are exact copies of each other
    assert d[2:-2].max() < 1.0
    assert np.isfinite(b).all()


# ======================================== D6: negative humidity is KEPT ====
def test_q_bound_is_the_d6_sanity_bound():
    ad = fam.REGISTRY["era5_q"]()
    assert all(c[2] == -1.0 and c[3] == 40.0 for c in ad.channels)
    assert "D6" in ad.specs()["era5_q"]["bounds_rule"]
    # the other three stores' specs carry no bounds rule (unchanged)
    assert "bounds_rule" not in fam.REGISTRY["era5_t"]().specs()["era5_t"]


def _negative_q(monkeypatch):
    real = e5.smoke_value

    def value(var, level, lat, lon, hours):
        v = real(var, level, lat, lon, hours)
        # pull 50 hPa below zero at high latitudes: -0.01 .. -0.3 g/kg, the
        # range D5 masked and D6 keeps
        return v - 4.5e-4 * (level == 50) if var == "q" else v
    monkeypatch.setattr(e5, "smoke_value", value)


def test_negative_humidities_are_kept_counted_and_their_minimum_measured(
        tmp_path, monkeypatch):
    import json
    _negative_q(monkeypatch)
    res = b1.run_smoke("era5_q", root=str(tmp_path / "s"), keep=True,
                       probe=False)
    sm = json.load(open(os.path.join(res["work"], "era5_q", "era5_q",
                                     "store.json")))
    c = sm["counts"]
    assert "out_of_bounds" not in c                     # nothing masked
    assert c["negative_values_kept"]["q_50"] > 0
    assert set(c["negative_values_kept"]) == {"q_50"}
    lowest = -c["max_negated_min_q_all"]
    assert lowest < -0.01                               # D5 would have masked
    assert lowest > -1.0
    # ... and PER YEAR in store.json, summing to the store's totals
    by = sm["counts_by_year"]
    assert by and sm["counts_by_year_note"]
    assert sum(v["negative_values_kept"] for v in by.values()) == \
        sum(c["negative_values_kept"].values())
    assert min(v["lowest_value_g_per_kg"] for v in by.values()
               if v["lowest_value_g_per_kg"] is not None) == lowest
    assert all(set(v["negative_values_kept_by_level"]) <= {"q_50"}
               for v in by.values())
    grp = sh.ShardedGroup(os.path.join(res["work"], "era5_q", "era5_q",
                                       "era5_q"))
    fr = grp.read_frame(2920, 8)
    assert not np.isnan(fr).any()
    k = list(e5.LEVELS_HPA).index(50)
    assert fr[:, :, k].min() < -0.01
    # the stored minimum is the measured one, to float16 precision
    allmin = min(float(np.nanmin(grp.read_frame(int(b), f)))
                 for b in grp.shard_index["bin"] for f in range(20)
                 if grp.read_frame(int(b), f) is not None)
    assert abs(allmin - lowest) <= abs(lowest) * 2 ** -11 + 1e-9


def test_the_assembler_refuses_parts_written_under_the_old_bound(
        tmp_path, monkeypatch):
    """A part fetched under D5 (-0.01 g/kg) carries that bound in its ledger's
    grid declaration; assembling it into a D6 store must REFUSE, so an old
    parked year can never be mistaken for a new one."""
    import json
    res = b1.run_smoke("era5_q", root=str(tmp_path / "s"), keep=True,
                       probe=False)
    ctx = res["ctx"]
    for y in ctx.years:
        p = os.path.join(ctx.year_dir(y), "counts.json")
        led = json.load(open(p))
        g = led["grids"]["era5_q"]
        for ch in g["channels"]:
            ch["min"] = -0.01
        g.pop("bounds_rule", None)
        json.dump(led, open(p, "w"))
    with pytest.raises(SystemExit, match="different grid declaration"):
        b1.stage_assemble_grid(ctx)


def test_only_humidity_writes_a_per_year_record(tmp_path):
    import json
    res = b1.run_smoke("era5_t", root=str(tmp_path / "t"), keep=True,
                       probe=False)
    sm = json.load(open(os.path.join(res["work"], "era5_t", "era5_t",
                                     "store.json")))
    assert "counts_by_year" not in sm

#!/usr/bin/env python3
"""E-089 · Monthly sum / count / m2 of the FINE family 1.gf grids — native
tiles and an exact pooled layer (`ml/export_fine_monthly.py`).

On two tiny stores built with the real tier-G framework
(`tests/make_fine_monthly_fixture.py`: an oc4k-shaped daily north-first grid
with clouds, a land block covering a whole tile and a day absent upstream; an
irtb-shaped uint8 "K - 160" three-hourly south-first band whose pooled cells
are centre-binned and whose first row sits ON a pooled edge):

  (a) NATIVE: every (year, month), every pixel and channel — the sum is
      float32(numpy's float64 sum of the native frames) BIT FOR BIT, the
      count is numpy's finite count, sqrt(m2/count) is numpy's nanstd within
      STD_REL_TOL·(|mean| + std);
  (b) POOLED = the native layer block-summed EXACTLY (sum bit for bit, count
      exactly), by an independent cell-by-cell loop;
  (c) pooled m2 / mean / count equal numpy computed DIRECTLY at the pooled
      resolution from the raw frames (cells assigned from the grid's affine
      rule, independently of the exporter's pool map);
  (d) empty tiles are absent (length 0, nothing stored) and the reader
      returns count 0 there; the file is exactly the stored lengths;
  (e) irtb shape: kelvin, a 248-frame month in a uint8 count, pooled counts
      past 255 in uint16, the -30° centre on a pooled edge;
  (f) the pool map of the four REAL grids (oc4k / pace4k exact 6 x 6, ACSPO
      12/13 centre-binned, the IR band 240 rows from -30°);
  (g) the falsifier (`verify`) passes, and FAILS on a nudged native sum, a
      nudged pooled sum and a nudged pooled m2;
  (h) a shard byte that is not the published store's is refused;
  (i) the publisher: yearly commits + one, the read-back refuses a byte that
      does not come back, the hosted index from the Hub, a failed verify
      report refused, `run` end to end and the skip of a published store;
  (j) the committed fixture agrees with its files, is < 1 MB, carries numpy's
      answers, and E-088's index (what the live site reads) is untouched.

    python3 -m pytest -q tests/test_export_fine_monthly.py
"""
import hashlib
import json
import os
import shutil
import sys

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ML = os.path.join(ROOT, "ml")
for p in (ML, os.path.join(ROOT, "tests")):
    if p not in sys.path:
        sys.path.insert(0, p)

import export_fine_monthly as E                                   # noqa: E402
import export_gridded_monthly as X                                # noqa: E402
import publish_fine_monthly_index as P                            # noqa: E402
import make_fine_monthly_fixture as F                             # noqa: E402
from family1 import sharded as sh                                 # noqa: E402
from family1 import monthly_tiles as mt                           # noqa: E402

FINE, TB = "fixture/fxfine", "fixture/fxtb"
TOL = E.STD_REL_TOL


@pytest.fixture(scope="module")
def fx(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("fm")
    stores = F.build(str(tmp / "src"))
    out = {}
    for s, base in stores.items():
        d = str(tmp / X.store_name(s))
        keep = str(tmp / ("keep_" + X.store_name(s)))
        st = E.export(s, d, base=base, workers=2, threads=2, keep=keep,
                      boxes=F.BOXES[s], deg=F.DEG)
        out[s] = dict(base=base, d=d, keep=keep, st=st)
    return dict(tmp=tmp, stores=out)


def frames(fx, s):
    fr, spec = F._frames(fx["stores"][s]["base"], s)
    return fr, spec


def native_month(fx, s, y, m):
    rd = mt.MonthlyTiles(os.path.join(fx["stores"][s]["d"], "native"))
    return rd.read_month(y, m)


def pooled(fx, s):
    d = os.path.join(fx["stores"][s]["d"], "pooled025")
    return {k: np.load(os.path.join(d, f"{k}.npy")) for k, _ in E.POOLED}


# ------------------------------------------------------------------ (a) -----
@pytest.mark.parametrize("s", [FINE, TB])
def test_a_native_equals_numpy_over_native_frames(fx, s):
    fr, spec = frames(fx, s)
    st = fx["stores"][s]["st"]
    assert sorted(fr) == [(y, m) for y, m, *_ in st["native"]["months"]]
    for (y, m), x in fr.items():
        a = native_month(fx, s, y, m)
        x64 = x.astype(np.float64)
        fin = np.isfinite(x64)
        ref_s = np.sum(x64, axis=0, where=fin).astype(np.float32)
        assert np.array_equal(a["sum"], ref_s), (y, m)        # bit for bit
        assert np.array_equal(a["count"].astype(np.int64), fin.sum(axis=0))
        n, mu, sd = F._stats(x)
        ok = n > 0
        std = np.sqrt(a["m2"].astype(np.float64) / np.maximum(n, 1))
        err = np.abs(std - np.nan_to_num(sd))[ok]
        assert (err <= TOL * (np.abs(mu[ok]) + sd[ok]) + 1e-12).all()
        assert (a["sum"][~ok] == 0).all() and (a["m2"][n <= 1] == 0).all()
        yi = y - st["year_first"]
        assert st["frames_present"][yi][m - 1] == len(x)


# ------------------------------------------------------------------ (b) -----
@pytest.mark.parametrize("s", [FINE, TB])
def test_b_pooled_is_the_native_layer_block_summed_exactly(fx, s):
    st = fx["stores"][s]["st"]
    g = st["pooled"]["grid"]
    P_ = pooled(fx, s)
    for y, m, *_ in st["native"]["months"]:
        a = native_month(fx, s, y, m)
        yi = y - st["year_first"]
        for p, (r0, r1) in enumerate(g["rows"]):
            for q, (c0, c1) in enumerate(g["cols"]):
                for c in range(len(st["chans"])):
                    blk = a["sum"][r0:r1, c0:c1, c].astype(np.float64)
                    want = np.float32(blk.sum())
                    assert P_["sum"][m - 1, c, yi, p, q] == want
                    assert P_["count"][m - 1, c, yi, p, q] == \
                        int(a["count"][r0:r1, c0:c1, c].astype(int).sum())
    assert P_["count"].dtype == np.dtype("<u2")


# ------------------------------------------------------------------ (c) -----
@pytest.mark.parametrize("s", [FINE, TB])
def test_c_pooled_equals_numpy_directly_at_the_pooled_resolution(fx, s):
    fr, spec = frames(fx, s)
    st = fx["stores"][s]["st"]
    g = st["pooled"]["grid"]
    J, I = F._cells(spec)
    j0, i0 = int(J.min()), int(I.min())
    # the exporter's pool map is the grid's affine rule, cell for cell
    for p, (r0, r1) in enumerate(g["rows"]):
        assert set(J[r0:r1] - j0) == {p}
    for q, (c0, c1) in enumerate(g["cols"]):
        assert set(I[c0:c1] - i0) == {q}
    P_ = pooled(fx, s)
    for (y, m), x in fr.items():
        yi = y - st["year_first"]
        a = native_month(fx, s, y, m)
        for p in range(g["H"]):
            rows = np.flatnonzero(J - j0 == p)
            for q in range(g["W"]):
                cols = np.flatnonzero(I - i0 == q)
                v = x[:, rows][:, :, cols].astype(np.float64)
                for c in range(len(st["chans"])):
                    vc = v[..., c]
                    ok = np.isfinite(vc)
                    n = int(ok.sum())
                    assert P_["count"][m - 1, c, yi, p, q] == n
                    if not n:
                        assert P_["sum"][m - 1, c, yi, p, q] == 0
                        continue
                    mu = vc[ok].mean()
                    sd = vc[ok].std()
                    Sp = float(P_["sum"][m - 1, c, yi, p, q])
                    bound = (0.5 * np.spacing(np.float32(abs(Sp))) + 0.5 *
                             np.spacing(np.abs(a["sum"][np.ix_(rows, cols)]
                                               [..., c])).astype(
                                 np.float64).sum()) / n + 4e-16 * abs(mu)
                    assert abs(Sp / n - mu) <= bound
                    std = np.sqrt(float(P_["m2"][m - 1, c, yi, p, q]) / n)
                    assert abs(std - sd) <= TOL * (abs(mu) + sd) + 1e-12


# ------------------------------------------------------------------ (d) -----
def test_d_empty_tiles_are_absent(fx):
    d = os.path.join(fx["stores"][FINE]["d"], "native")
    rd = mt.MonthlyTiles(d)
    sp = rd.spec
    for y, m, frames_, tiles, nbytes in fx["stores"][FINE]["st"]["native"][
            "months"]:
        idx = rd.index(y, m)
        assert (idx[0, 0, :, 1] == 0).all(), "the land tile was stored"
        n_any = 0
        a = rd.read_month(y, m)
        for ty in range(sp["n_tiles_y"]):
            for tx in range(sp["n_tiles_x"]):
                r0, r1 = sp["row_extents"][ty]
                c0, c1 = sp["col_extents"][tx]
                has = a["count"][r0:r1, c0:c1].any()
                assert has == bool(idx[ty, tx, 0, 1] > 0)
                n_any += has
        assert tiles == n_any == sp["n_tiles_y"] * sp["n_tiles_x"] - 1
        size = os.path.getsize(os.path.join(d, mt.shard_relpath(y, m)))
        assert size == nbytes == int(idx[..., 1].sum())
        t = rd.read_tile(y, m, 0, 0)
        assert not t["count"].any() and not t["sum"].any()
        # one range read gets a mean (sum + count): the parts are adjacent
        e = idx[1, 1]
        assert e[1, 0] == e[0, 0] + e[0, 1] and e[2, 0] == e[1, 0] + e[1, 1]


# ------------------------------------------------------------------ (e) -----
def test_e_irtb_shape_kelvin_uint8_count_and_uint16_pooled(fx):
    st = fx["stores"][TB]["st"]
    assert st["value_offset"] == 160.0 and st["units"] == {"tb": "K"}
    assert st["max_count"] == 248                      # a full March
    a = native_month(fx, TB, 2010, 3)
    assert a["count"].max() == 248 and a["count"].dtype == np.uint8
    with np.errstate(invalid="ignore"):
        mean = a["sum"] / a["count"]
    assert 160 <= np.nanmin(mean) and np.nanmax(mean) <= 414
    P_ = pooled(fx, TB)
    assert P_["count"].max() > 255
    g = st["pooled"]["grid"]
    assert g["rows"][0] == [0, 3] and g["lat0"] == -90 + 4.5 * F.DEG
    assert not g["exact_blocks"]
    assert st["frames_present"][0][1] == 71          # one frame absent


# ------------------------------------------------------------------ (f) -----
REAL = {
    "oc4k": dict(H=4320, W=8640, grid=dict(x0=-180.0, y0=90.0,
                                           dx=0.041666666666666664,
                                           dy=-0.041666666666666664)),
    "sst_acspo02": dict(H=9000, W=18000, grid=dict(x0=-180.0, y0=90.0,
                                                   dx=0.02, dy=-0.02)),
    "irtb": dict(H=1649, W=9896, grid=dict(x0=-180.0, y0=-30.018192844147965,
                                           dx=0.03637833468067906,
                                           dy=0.036385688295936934)),
}


def test_f_pool_map_of_the_real_grids():
    oc = E.pool_map(REAL["oc4k"])
    assert (oc["Hp"], oc["Wp"], oc["exact_blocks"]) == (720, 1440, True)
    assert oc["block_rows"] == oc["block_cols"] == [6]
    assert oc["rows"][0] == [4314, 4320] and oc["rows"][-1] == [0, 6]
    ss = E.pool_map(REAL["sst_acspo02"])
    assert (ss["Hp"], ss["Wp"]) == (720, 1440)
    assert ss["block_rows"] == ss["block_cols"] == [12, 13]
    # a centre ON an edge (row 12: 89.75°) goes to the cell north of it
    assert ss["rows"][-1] == [0, 13] and ss["cols"][0] == [0, 12]
    ir = E.pool_map(REAL["irtb"])
    assert (ir["Hp"], ir["Wp"], ir["j0"]) == (240, 1440, 240)
    assert ir["lat0"] == -29.875 and ir["rows"][0] == [0, 7]
    assert ir["block_rows"] == ir["block_cols"] == [6, 7]
    for pm in (oc, ss, ir):
        assert max(pm["block_rows"]) * max(pm["block_cols"]) * 248 <= 65535


# ------------------------------------------------------------------ (g) -----
def test_g_verify_passes(fx):
    for s, v in fx["stores"].items():
        r = E.verify(s, v["d"], v["keep"], base=v["base"], workers=1,
                     report=os.path.join(v["d"], "verify.json"))
        assert r["ok"] and r["native"]["n_checks"] and r["pooled"]["n_checks"]
        assert r["blocksum_summary"]["months"] == len(v["st"]["native"][
            "months"]) and r["blocksum_summary"]["max_abs_sum"] == 0.0


def _copy(fx, s, tmp):
    v = fx["stores"][s]
    d = str(tmp / "o")
    shutil.copytree(v["d"], d)
    return v, d


@pytest.mark.parametrize("which", ["native_sum", "pooled_sum", "pooled_m2"])
def test_g_verify_fails_on_a_nudged_file(fx, tmp_path, which):
    v, d = _copy(fx, FINE, tmp_path)
    st = v["st"]
    y, m = st["falsifier_months"][0]
    if which == "native_sum":
        rd = mt.MonthlyTiles(os.path.join(d, "native"))
        a = rd.read_month(y, m)
        S = a["sum"].copy()
        k = np.argwhere(a["count"] > 3)[0]
        S[tuple(k)] = np.float32(S[tuple(k)] * (1 + 1e-4))
        mt.write_month(rd.spec, S, a["count"], a["m2"],
                       os.path.join(d, "native", mt.shard_relpath(y, m)),
                       os.path.join(d, "native", mt.index_relpath(y, m)))
    else:
        k = "sum" if which == "pooled_sum" else "m2"
        A = np.load(os.path.join(d, "pooled025", f"{k}.npy"), mmap_mode="r+")
        N = np.load(os.path.join(d, "pooled025", "count.npy"))
        yi = y - st["year_first"]
        pos = np.argwhere(N[m - 1, 0, yi] > 4)[0]
        A[m - 1, 0, yi, pos[0], pos[1]] *= np.float32(1 + 1e-3)
        A.flush()
        del A
    with pytest.raises(SystemExit, match="VERIFY FAILED"):
        E.verify(FINE, d, v["keep"], base=v["base"], workers=1)


# ------------------------------------------------------------------ (h) -----
def test_h_a_shard_that_is_not_the_published_one_is_refused(fx, tmp_path):
    copy = str(tmp_path / "store")
    shutil.copytree(fx["stores"][TB]["base"], copy)
    zs = sorted(os.path.join(dp, f) for dp, _, fs in
                os.walk(os.path.join(copy, "fxtb")) for f in fs
                if f.endswith(".zst"))
    with open(zs[0], "r+b") as fh:
        fh.seek(3)
        b = fh.read(1)
        fh.seek(3)
        fh.write(bytes([b[0] ^ 1]))
    with pytest.raises(SystemExit, match="sha256"):
        E.export(TB, str(tmp_path / "o"), base=copy, workers=1, threads=1,
                 boxes=F.BOXES[TB], deg=F.DEG)


# ------------------------------------------------------------------ (i) -----
class _FakeApi:
    def __init__(self):
        self.commits = []

    def create_commit(self, repo_id, repo_type, operations, commit_message):
        self.commits.append(([(op.path_in_repo, op.path_or_fileobj)
                              for op in operations], commit_message))


def _hub_of(api):
    store = {}
    for ops, _ in api.commits:
        for rel, local in ops:
            store[rel] = open(local, "rb").read()
    return store


def _fetch(store, corrupt=None):
    def fetch(session, url, a, b):
        rel = url.split("/resolve/main/", 1)[1]
        data = store[rel]
        if corrupt and rel.endswith(corrupt):
            data = data[:-1] + bytes([data[-1] ^ 1])
        return data[a:b + 1], f"bytes {a}-{b}/{len(data)}"
    return fetch


@pytest.fixture(scope="module")
def uploaded(fx):
    for s, v in fx["stores"].items():
        E.verify(s, v["d"], v["keep"], base=v["base"], workers=1,
                 report=os.path.join(v["d"], "verify.json"))
    api = _FakeApi()
    ups = {}
    for s, v in fx["stores"].items():
        def rb(store, man):
            return P.readback(store, man, session=object(),
                              fetch=_fetch(_hub_of(api)))
        ups[s] = P.upload_store(s, v["d"], api=api, readback_fn=rb)
    return api, ups


def test_i_yearly_commits_and_one_and_a_read_back(fx, uploaded):
    api, ups = uploaded
    msgs = [m for _, m in api.commits]
    assert sum("native monthly sums of fixture/fxfine" in m for m in msgs) == 2
    assert sum("native monthly sums of fixture/fxtb" in m for m in msgs) == 1
    assert sum("pooled" in m for m in msgs) == 2
    for ops, m in api.commits:
        if "pooled" in m:
            rels = {r.split("/monthly/", 1)[1] for r, _ in ops}
            assert rels == {"pooled025/sum.npy", "pooled025/count.npy",
                            "pooled025/m2.npy", "stats.json", "verify.json",
                            "native/tile_grid.json", "manifest.json"}
    for s, u in ups.items():
        assert u["readback_ok"] and len(u["manifest"]["native"]) == \
            2 * len(fx["stores"][s]["st"]["native"]["months"])


def test_i_read_back_refuses_a_byte_that_does_not_come_back(fx, uploaded):
    api, ups = uploaded
    man = ups[TB]["manifest"]
    bad = man["native"][0]["rel"]
    with pytest.raises(SystemExit, match="downloaded back"):
        P.readback(TB, man, session=object(),
                   fetch=_fetch(_hub_of(api), corrupt=bad))


def _measure(store):
    def m(url):
        rel = url.split("/resolve/main/", 1)[1]
        return dict(status=206, final_url="https://cdn.example/x",
                    access_control_allow_origin="*",
                    access_control_expose_headers="*", accept_ranges="bytes",
                    content_range="bytes 0-4095/1", head=store[rel][:4096])
    return m


def test_i_index_from_the_hub(fx, uploaded, tmp_path):
    api, _ = uploaded
    hub = _hub_of(api)
    out = str(tmp_path / "idx.json")
    ix = P.main(["index", "--stores", f"{FINE},{TB}", "--index", out],
                fetch=_fetch(hub), measure=_measure(hub), session=object())
    on = json.load(open(out))
    assert on == json.loads(json.dumps(ix))
    assert on["schema"] == "gridded-monthly-fine/1"
    assert on["restore_verified"] is True and on["fixture"] is False
    assert set(on["cors_measured"]) == {"sum", "count", "native_shard"}
    for s in (FINE, TB):
        b = on["stores"][s]
        C, Y = len(b["chans"]), b["n_years"]
        g = b["pooled"]["grid"]
        for role, item in (("sum", 4), ("count", 2), ("m2", 4)):
            f = b["pooled"][role]
            assert f["shape"] == [12, C, Y, g["H"], g["W"]]
            assert f["plane_bytes"] == g["H"] * g["W"] * item
            assert f["bytes"] == f["header_len"] + 12 * C * Y * g["H"] * \
                g["W"] * item
            assert set(f) >= {"url", "bytes", "sha256", "header_len",
                              "shape", "dtype", "itemsize", "axes",
                              "plane_bytes", "month_channel_bytes"}
        assert b["pooled"]["count"]["dtype"] == "<u2"
        nv = b["native"]
        assert nv["index_shape"] == [nv["n_tiles_y"], nv["n_tiles_x"], 3, 2]
        assert [p["name"] for p in nv["parts"]] == ["sum", "count", "m2"]
        assert len(nv["months"]) == len(fx["stores"][s]["st"]["native"][
            "months"])
        assert b["falsifier"]["ok"] and len(b["frames_present"]) == Y
        assert b["source_store"] == f"tensors/{s}"
        assert g["south_first"] is True and len(g["rows"]) == g["H"]


def test_i_index_refuses_a_failed_verify_report(fx, tmp_path):
    out = str(tmp_path / "o")
    os.makedirs(out)
    d = os.path.join(out, "fxtb")
    shutil.copytree(fx["stores"][TB]["d"], d)
    rep = json.load(open(os.path.join(d, "verify.json")))
    rep["ok"] = False
    json.dump(rep, open(os.path.join(d, "verify.json"), "w"))
    with pytest.raises(SystemExit, match="verify report failed"):
        P.main(["index", "--local", "--out", out, "--stores", TB,
                "--no-cors", "--index", str(tmp_path / "i.json")])


def test_i_run_end_to_end_and_skips_a_published_store(fx, tmp_path,
                                                       monkeypatch):
    hub = tmp_path / "hub"
    os.makedirs(hub / "fixture")
    for s, v in fx["stores"].items():
        os.symlink(v["base"], hub / s)
    monkeypatch.setattr(X, "HUB", str(hub) + "/")
    monkeypatch.setattr(E, "BOXES", {**E.BOXES, **F.BOXES})
    monkeypatch.setattr(E, "POOL_DEG", F.DEG)
    real_export = E.export

    def export15(store, out, **kw):
        kw.setdefault("deg", F.DEG)
        return real_export(store, out, **kw)
    monkeypatch.setattr(E, "export", export15)
    pub = {}

    def fake_upload(store, d, api=None, readback_fn=None):
        man = P.make_manifest(store, d)
        dst = hub / store / "monthly"
        for r in P.all_files(man):
            os.makedirs(os.path.dirname(dst / r["rel"]), exist_ok=True)
            shutil.copy(os.path.join(d, r["rel"]), dst / r["rel"])
        pub[store] = man
        return dict(bytes=man["bytes"], readback_ok=True, manifest=man)
    monkeypatch.setattr(P, "upload_store", fake_upload)
    out = str(tmp_path / "run")
    summ = E.run([FINE, TB], out, workers=2, threads=2, verify_workers=1)
    assert set(pub) == {FINE, TB}
    assert all(v["verify_ok"] and v["readback_ok"] for v in summ.values())
    assert not os.path.exists(os.path.join(out, "fxfine")), "disk not freed"
    summ2 = E.run([FINE, TB], out, workers=2, threads=2)
    assert all(str(v).startswith("skipped") for v in summ2.values())


# ------------------------------------------------------------------ (j) -----
def test_j_committed_fixture_agrees_with_its_own_files():
    path = os.path.join(E.FIXTURE_OUT, "gridded_monthly_fine_index.json")
    assert os.path.exists(path), \
        "run `python3 ml/export_fine_monthly.py fixture`"
    ix = json.load(open(path))
    assert ix["fixture"] is True and ix["restore_verified"] is False
    assert ix["fixture_dir"] == "data/gridded_monthly/fixture_fine/"
    assert ix["schema"] == "gridded-monthly-fine/1"
    total = os.path.getsize(path)
    for s, b in ix["stores"].items():
        root = os.path.join(ROOT, ix["fixture_dir"], X.store_name(s))
        recs = [b["pooled"][r] for r in ("sum", "count", "m2")] + \
            [b["stats"], b["verify"], b["native"]["tile_grid"]]
        for r in recs:
            p = os.path.join(root, r["url"].split("/monthly/", 1)[1])
            assert hashlib.sha256(open(p, "rb").read()).hexdigest() == \
                r["sha256"], p
            total += os.path.getsize(p)
        for y, m, fr, tiles, sb, ss, ib, isha in b["native"]["months"]:
            for rel, sha in ((mt.shard_relpath(y, m), ss),
                             (mt.index_relpath(y, m), isha)):
                p = os.path.join(root, "native", rel)
                assert hashlib.sha256(open(p, "rb").read()).hexdigest() == sha
                total += os.path.getsize(p)
        assert b["falsifier"]["ok"]
    exp = json.load(open(os.path.join(E.FIXTURE_OUT, "expected.json")))
    assert set(exp["stores"]) == set(ix["stores"])
    total += os.path.getsize(os.path.join(E.FIXTURE_OUT, "expected.json"))
    assert total < 1e6, f"the fixture is {total / 1e6:.2f} MB"


def test_j_the_live_index_is_untouched_and_the_fine_one_is_consistent():
    old = json.load(open(os.path.join(ROOT, "data",
                                      "gridded_monthly_index.json")))
    assert not any(k.startswith("family1_gf/") for k in old["stores"]), \
        "the fine stores belong in data/gridded_monthly_fine_index.json"
    path = os.path.join(ROOT, "data", "gridded_monthly_fine_index.json")
    if not os.path.exists(path):
        pytest.skip("the real fine index is not committed yet")
    ix = json.load(open(path))
    assert ix["fixture"] is False and ix["restore_verified"] is True
    assert set(ix["stores"]) == set(E.FINE)
    for r in ("sum", "count", "native_shard"):
        c = ix["cors_measured"][r]
        assert c["status"] == 206 and c["origin"] == "https://blauewelt.org"
        assert c["access_control_allow_origin"]
    for s, b in ix["stores"].items():
        Y = b["n_years"]
        assert b["years"] == list(range(b["year_first"], b["year_last"] + 1))
        pres, poss = np.array(b["frames_present"]), \
            np.array(b["frames_possible"])
        assert (pres <= poss).all() and pres.max() == b["max_count"] <= 255
        assert len(b["native"]["months"]) == int((pres > 0).sum())
        assert b["pooled"]["sum"]["shape"][2] == Y
        assert b["falsifier"]["ok"]

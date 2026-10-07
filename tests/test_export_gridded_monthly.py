#!/usr/bin/env python3
"""E-088 §4 · Per-year monthly sum / count / m2 of the sharded gridded stores.

On two tiny stores built with the real tier-G framework
(`tests/make_gridded_monthly_fixture.py`: a daily 2-channel field across two
year boundaries with land, clouds and one day absent upstream; a six-hourly
3-level field across month boundaries inside bins):

  (a) every (year, month, channel) plane: sum/count equals numpy's nanmean of
      the native frames read through ml/family1/sharded.py within the
      rigorous float32 bound, counts EXACTLY, sqrt(m2/count) equals numpy's
      nanstd within STD_REL_TOL·(|mean| + std);
  (b) composed over arbitrary sets of (year, month) cells — a period, every
      January, a scattered set — mean and population std equal numpy over the
      union of native frames (Chan's combination);
  (c) the month is the CALENDAR month of the frame's own instant: a bin that
      straddles a month boundary is split between the two;
  (d) frames present / possible per (year, month), incl. the absent day and
      partial first / last months; the uint8 bound;
  (e) the falsifier (`verify`) passes, and FAILS on a sum nudged past the
      bound and on an m2 nudged past the std tolerance;
  (f) a shard byte that is not the published store's is refused before
      anything is summed;
  (g) the publisher: manifest + one commit per store; the Hub restore refuses
      a byte that does not come back; a failed verify report is refused; CORS
      measured from blauewelt.org; index keys shared with E-086's;
  (h) the committed fixture agrees with its own files and is < 1 MB.

    python3 -m pytest -q tests/test_export_gridded_monthly.py
"""
import datetime as dt
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

import export_gridded_monthly as X                                # noqa: E402
import publish_gridded_monthly_index as P                         # noqa: E402
import make_gridded_monthly_fixture as F                          # noqa: E402
from family1 import sharded as sh                                 # noqa: E402

DAILY, SIX = "fixture/fxdaily", "fixture/fx6h"


@pytest.fixture(scope="module")
def fx(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("gm")
    stores = F.build(str(tmp / "src"))
    out = {}
    for s, base in stores.items():
        d = str(tmp / X.store_name(s))
        keep = str(tmp / ("keep_" + X.store_name(s)))
        st = X.export(s, d, base=base, workers=2, threads=2, keep=keep)
        out[s] = dict(base=base, d=d, keep=keep, st=st)
    return dict(tmp=tmp, stores=out)


def native(fx, s):
    """{(y, m): [n, H, W, C] float32} of every native frame, via sharded.py."""
    base = fx["stores"][s]["base"]
    src, meta, spec, arr, g = X.open_store(base, s)
    grp = sh.ShardedGroup(os.path.join(base, g))
    by = {}
    for b, f, t in X.frame_times(arr, spec):
        by.setdefault((t.year, t.month - 1), []).append(grp.read_frame(b, f))
    return {k: np.stack(v) for k, v in by.items()}, spec


def files(fx, s):
    d = fx["stores"][s]["d"]
    return (np.load(os.path.join(d, "sum.npy")),
            np.load(os.path.join(d, "count.npy")),
            np.load(os.path.join(d, "m2.npy")),
            json.load(open(os.path.join(d, "stats.json"))))


# ------------------------------------------------------------------ (a) -----
@pytest.mark.parametrize("s", [DAILY, SIX])
def test_a_every_plane_equals_numpy_over_native_frames(fx, s):
    nat, _ = native(fx, s)
    S, N, M, st = files(fx, s)
    y0 = st["year_first"]
    seen = 0
    for (y, m), stack in nat.items():
        rm, rn, rs = X.native_stats(stack)
        for c in range(S.shape[1]):
            mean, n, std, bound = X.compose(S[m, c, y - y0][None],
                                            N[m, c, y - y0][None],
                                            M[m, c, y - y0][None])
            r = X.check(mean, n, std, bound, rm[..., c], rn[..., c],
                        rs[..., c], f"{y}-{m + 1} c{c}")
            assert r["ok"], r
            seen += 1
    assert seen == len(nat) * S.shape[1]
    # nothing is stored where nothing was observed
    assert (S[N == 0] == 0).all() and (M[N == 0] == 0).all()
    assert not np.isnan(S).any() and not np.isnan(M).any()


# ------------------------------------------------------------------ (b) -----
@pytest.mark.parametrize("cells", [
    "all", "januaries", "scattered", "period2010"])
def test_b_composed_sets_equal_numpy_over_the_union(fx, cells):
    nat, _ = native(fx, DAILY)
    S, N, M, st = files(fx, DAILY)
    y0 = st["year_first"]
    keys = sorted(nat)
    pick = {"all": keys,
            "januaries": [k for k in keys if k[1] == 0],
            "scattered": [k for k in keys if (k[0] * 12 + k[1]) % 3 == 1],
            "period2010": [k for k in keys if k[0] == 2010]}[cells]
    assert len(pick) >= 2
    Sk = np.stack([S[m, :, y - y0] for y, m in pick])
    Nk = np.stack([N[m, :, y - y0] for y, m in pick])
    Mk = np.stack([M[m, :, y - y0] for y, m in pick])
    mean, n, std, bound = X.compose(Sk, Nk, Mk)
    rm, rn, rs = X.native_stats(np.concatenate([nat[k] for k in pick]))
    for c in range(S.shape[1]):
        r = X.check(mean[c], n[c], std[c], bound[c], rm[..., c], rn[..., c],
                    rs[..., c], cells)
        assert r["ok"], r
        assert r["max_std_over_tol"] < 0.5


# ------------------------------------------------------------------ (c) -----
def test_c_a_bin_straddling_a_month_is_split_by_calendar_month(fx):
    _, _, _, st = files(fx, SIX)
    # 2010-01-28 00 UTC .. 2010-03-03 06 UTC, six-hourly, no gaps
    assert st["frames_present"][0][:3] == [16, 112, 10]
    src, meta, spec, arr, g = X.open_store(fx["stores"][SIX]["base"], SIX)
    straddle = [b for b in arr["bin"]
                if sh.bin_start_date(b).month != (sh.bin_start_date(b)
                                                  + dt.timedelta(days=4)).month]
    assert straddle, "the fixture must have a bin across a month boundary"
    months = {sh.frame_datetime(straddle[0], f, 21600).month
              for f in range(20)}
    assert len(months) == 2


# ------------------------------------------------------------------ (d) -----
def test_d_frames_present_and_possible(fx):
    _, N, _, st = files(fx, DAILY)
    assert st["years"] == [2009, 2010, 2011]
    pres, poss = np.array(st["frames_present"]), np.array(st["frames_possible"])
    assert pres[0, 11] == poss[0, 11] == 12            # 2009-12-20 .. 31
    assert pres[1, 2] == 30 and poss[1, 2] == 31        # 2010-03-14 absent
    assert pres[2, 0] == poss[2, 0] == 10               # 2011-01-01 .. 10
    assert (pres[0, :11] == 0).all() and (pres[2, 1:] == 0).all()
    assert int(N.max()) <= 31 and st["max_count"] == 31
    assert N.dtype == np.uint8
    # the arithmetic itself, on the real calendars
    lo, hi = dt.datetime(1982, 1, 1), dt.datetime(2026, 6, 30, 18)
    assert X.frames_possible(2015, 0, lo, hi, 21600) == 124
    assert X.frames_possible(2012, 1, lo, hi, 86400) == 29
    assert X.frames_possible(2026, 5, lo, hi, 21600) == 120
    assert X.frames_possible(2026, 6, lo, hi, 21600) == 0


# ------------------------------------------------------------------ (e) -----
def test_e_verify_passes(fx):
    for s, v in fx["stores"].items():
        r = X.verify(s, v["d"], v["keep"], base=v["base"],
                     report=os.path.join(v["d"], "verify.json"))
        assert r["ok"] and r["n_planes"] >= 6 and r["period"]
        assert r["max_mean_over_bound"] <= 1 and r["max_std_over_tol"] <= 1


@pytest.mark.parametrize("which", ["sum", "m2"])
def test_e_verify_fails_on_a_nudged_file(fx, tmp_path, which):
    v = fx["stores"][DAILY]
    bad = str(tmp_path / "bad")
    shutil.copytree(v["d"], bad)
    st = json.load(open(os.path.join(bad, "stats.json")))
    y, mo = st["falsifier_months"][1]
    A = np.load(os.path.join(bad, f"{which}.npy"), mmap_mode="r+")
    N = np.load(os.path.join(bad, "count.npy"))
    yi = y - st["year_first"]
    h, w = [int(x[0]) for x in np.nonzero(N[mo - 1, 0, yi] > 2)]
    x = float(A[mo - 1, 0, yi, h, w])
    A[mo - 1, 0, yi, h, w] = np.float32(x + max(1e-3 * abs(x), 1e-2))
    A.flush()
    del A
    with pytest.raises(SystemExit, match="VERIFY FAILED"):
        X.verify(DAILY, bad, v["keep"], base=v["base"],
                 report=str(tmp_path / "v.json"))
    assert json.load(open(tmp_path / "v.json"))["ok"] is False


# ------------------------------------------------------------------ (f) -----
def test_f_a_shard_that_is_not_the_published_one_is_refused(fx, tmp_path):
    base = fx["stores"][SIX]["base"]
    copy = str(tmp_path / "store")
    shutil.copytree(base, copy)
    g = "fx6h"
    sh_files = sorted(os.path.join(dp, f) for dp, _, fs in
                      os.walk(os.path.join(copy, g)) for f in fs
                      if f.endswith(".zst"))
    with open(sh_files[0], "r+b") as fh:
        fh.seek(5)
        b = fh.read(1)
        fh.seek(5)
        fh.write(bytes([b[0] ^ 1]))
    with pytest.raises(SystemExit, match="sha256"):
        X.export(SIX, str(tmp_path / "o"), base=copy, workers=1, threads=1)


# ------------------------------------------------------------------ (g) -----
class _FakeApi:
    def __init__(self):
        self.commits = []

    def create_commit(self, repo_id, repo_type, operations, commit_message):
        self.commits.append(([(op.path_in_repo, op.path_or_fileobj)
                              for op in operations], commit_message))


def _fake_hub(api, corrupt=None):
    store = {}
    for ops, _ in api.commits:
        for rel, local in ops:
            store[rel] = open(local, "rb").read()

    def fetch(session, url, a, b):
        rel = url.split("/resolve/main/", 1)[1]
        data = store[rel]
        if corrupt and rel.endswith(corrupt):
            data = data[:-1] + bytes([data[-1] ^ 1])
        return data[a:b + 1], f"bytes {a}-{b}/{len(data)}"
    return fetch, store


@pytest.fixture(scope="module")
def uploaded(fx):
    for s, v in fx["stores"].items():
        X.verify(s, v["d"], v["keep"], base=v["base"],
                 report=os.path.join(v["d"], "verify.json"))
    api = _FakeApi()
    for s, v in fx["stores"].items():
        P.upload_store(s, v["d"], api=api)
    return api


def test_g_one_commit_per_store_with_its_manifest(fx, uploaded):
    assert len(uploaded.commits) == 2
    for ops, msg in uploaded.commits:
        rels = {r.rsplit("/", 1)[1] for r, _ in ops}
        assert rels == {"sum.npy", "count.npy", "m2.npy", "stats.json",
                        "verify.json", "manifest.json"}
        assert all(r.startswith("tensors/fixture/") and "/monthly/" in r
                   for r, _ in ops)


def _measure(store):
    def m(url):
        rel = url.split("/resolve/main/", 1)[1]
        return dict(status=206, final_url="https://cdn.example/x",
                    access_control_allow_origin="*",
                    access_control_expose_headers="*", accept_ranges="bytes",
                    content_range="bytes 0-4095/1", head=store[rel][:4096])
    return m


def test_g_index_from_the_hub(fx, uploaded, tmp_path):
    fetch, store = _fake_hub(uploaded)
    out = str(tmp_path / "idx.json")
    ix = P.main(["index", "--stores", f"{DAILY},{SIX}", "--index", out],
                fetch=fetch, measure=_measure(store), session=object())
    on = json.load(open(out))
    assert on == json.loads(json.dumps(ix))
    assert on["restore_verified"] is True
    assert on["axes"] == ["month", "channel", "year", "lat", "lon"]
    for r in ("sum", "count"):
        assert on["cors_measured"][r]["origin"] == "https://blauewelt.org"
    for s in (DAILY, SIX):
        b = on["stores"][s]
        C, Y, H, W = len(b["chans"]), b["n_years"], b["grid"]["H"], \
            b["grid"]["W"]
        for role, item in (("sum", 4), ("count", 1), ("m2", 4)):
            f = b[role]
            assert f["shape"] == [12, C, Y, H, W]
            assert f["plane_bytes"] == H * W * item
            assert f["bytes"] == f["header_len"] + 12 * C * Y * H * W * item
            # the key style E-086's reader uses
            assert set(f) >= {"url", "bytes", "sha256", "header_len", "shape",
                              "dtype", "itemsize", "axes", "plane_bytes",
                              "month_channel_bytes"}
        assert b["falsifier"]["ok"] is True
        assert len(b["frames_present"]) == Y == len(b["frames_possible"])
        assert b["licence"]["attribution"]
    assert on["stores"][SIX]["grid"]["south_first"] is True
    assert on["stores"][SIX]["grid"]["lat0"] == -90.0


def test_g_index_refuses_a_byte_that_does_not_come_back(fx, uploaded,
                                                         tmp_path):
    fetch, _ = _fake_hub(uploaded, corrupt="fx6h/monthly/m2.npy")
    out = str(tmp_path / "i.json")
    with pytest.raises(SystemExit, match="downloaded back"):
        P.main(["index", "--stores", f"{DAILY},{SIX}", "--index", out,
                "--no-cors"], fetch=fetch, session=object())
    assert not os.path.exists(out)


def test_g_a_failed_verify_report_is_refused(fx, tmp_path):
    out = str(tmp_path / "o")
    os.makedirs(out)
    d = os.path.join(out, "fxdaily")
    shutil.copytree(fx["stores"][DAILY]["d"], d)
    rep = json.load(open(os.path.join(d, "verify.json")))
    rep["ok"] = False
    json.dump(rep, open(os.path.join(d, "verify.json"), "w"))
    with pytest.raises(SystemExit, match="verify report failed"):
        P.main(["index", "--local", "--out", out, "--stores", DAILY,
                "--no-cors", "--index", str(tmp_path / "i.json")])


def test_g_run_does_each_store_end_to_end_and_skips_a_published_one(
        fx, tmp_path, monkeypatch):
    """`run` (the box's loop): export, verify, upload, free — per store; a
    second run skips a store whose monthly/stats.json names the same source."""
    hub = tmp_path / "hub"
    for s, v in fx["stores"].items():
        os.makedirs(hub / "fixture", exist_ok=True)
        os.symlink(v["base"], hub / s)
    monkeypatch.setattr(X, "HUB", str(hub) + "/")
    pub = {}

    def fake_upload(store, d, api=None):
        man = P.make_manifest(store, d)
        dst = hub / store / "monthly"
        os.makedirs(dst, exist_ok=True)
        for r in man["files"].values():
            shutil.copy(os.path.join(d, r["rel"]), dst / r["rel"])
        pub[store] = man
        return man
    monkeypatch.setattr(P, "upload_store", fake_upload)
    out = str(tmp_path / "run")
    summ = X.run([DAILY, SIX], out, workers=2, threads=2)
    assert set(pub) == {DAILY, SIX}
    assert all(v["verify_ok"] for v in summ.values())
    assert not os.path.exists(os.path.join(out, "fxdaily")), "disk not freed"
    summ2 = X.run([DAILY, SIX], out, workers=2, threads=2)
    assert all(str(v).startswith("skipped") for v in summ2.values())
    for s in (DAILY, SIX):
        os.remove(hub / s / "monthly" / "stats.json")
        os.rmdir(hub / s / "monthly") if not os.listdir(
            hub / s / "monthly") else None


# ------------------------------------------------------------------ (h) -----
def test_h_committed_fixture_agrees_with_its_own_files():
    path = os.path.join(X.FIXTURE_OUT, "gridded_monthly_index.json")
    assert os.path.exists(path), \
        "run `python3 ml/export_gridded_monthly.py fixture`"
    ix = json.load(open(path))
    assert ix["fixture"] is True and ix["restore_verified"] is False
    assert ix["fixture_dir"] == "data/gridded_monthly/fixture/"
    total = os.path.getsize(path)
    import hashlib
    for s, b in ix["stores"].items():
        for role in ("sum", "count", "m2", "stats", "verify"):
            rel = b[role]["url"].split("/monthly/", 1)[1]
            p = os.path.join(ROOT, ix["fixture_dir"], X.store_name(s), rel)
            assert hashlib.sha256(open(p, "rb").read()).hexdigest() == \
                b[role]["sha256"], p
            total += os.path.getsize(p)
        assert b["falsifier"]["ok"]
    assert total < 1e6, f"the fixture is {total / 1e6:.2f} MB"


if __name__ == "__main__":                                  # pragma: no cover
    sys.exit(pytest.main([__file__, "-q"]))

#!/usr/bin/env python3
"""E-086 §5 · Per-year monthly sums and counts compose THE function's climatology.

The claim of `ml/export_family7_monthly.py` is that for ANY set of years Ys,
Σ_{y∈Ys} sum / Σ_{y∈Ys} count is the climatology
`ml/trainprobe.py::anomaly_transform` computes when every year outside Ys is
held out — NaN exactly where it has no sample, values within the float32
rounding of the stored per-year sums (a rigorous bound, not a threshold). On a
toy four-group tensor (pentads 2007–2023; a bin-aligned 1°-class group with a
static channel; a monthly `rg100`-like group; an `oc025`-like group from 2015):

  (a) three different year sets (plus the paper's split) reproduce a DIRECT
      anomaly_transform call, per group, NaN pattern equal, |Δ| within bound;
  (b) count.npy is uint8, ≤ 7, and equals an independent recount; sum.npy is
      bit-identical to an independent float64 sum rounded once to float32;
  (c) rg100 and oc025 are summed on their OWN rows and year axes;
  (d) one plane, and one run of years, read at the stated byte offset IS that
      slice;
  (e) the falsifier (`verify`) passes against the published climatology of
      the toy AND fails when one stored sum is perturbed by one float32 ulp
      more than the bound allows;
  (f) the publisher: manifest, one commit per group, the Hub restore refuses
      a byte that does not come back, CORS is measured from blauewelt.org, an
      index against another stem or a failed verify report is refused;
  (g) the committed fixture agrees with its own files and is < 1 MB.

    python3 -m pytest -q tests/test_export_family7_monthly.py
"""
import ast
import hashlib
import json
import os
import sys
import warnings

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ML = os.path.join(ROOT, "ml")
if ML not in sys.path:
    sys.path.insert(0, ML)

import export_family7_clim as E                                   # noqa: E402
import export_family7_monthly as M                                # noqa: E402
import publish_family7_monthly_index as P                         # noqa: E402
import publish_family7_clim_index as PC                           # noqa: E402
from publish_family7_index import (grid_block, group_block,         # noqa: E402
                                   parse_npy_header)
from tensor_io import load_tensor, save_tensor                     # noqa: E402

EPOCH = np.datetime64("1982-01-01")
STEM = "family7_toy_pentad"
GROUPS = ("g025", "g100", "rg100", "oc025")


def _lift_anomaly_transform():
    """THE function, lifted from ml/trainprobe.py's source (no torch here)."""
    src = open(os.path.join(ML, "trainprobe.py"), encoding="utf-8").read()
    keep = [n for n in ast.parse(src).body
            if isinstance(n, ast.FunctionDef) and n.name == "anomaly_transform"]
    assert keep, "ml/trainprobe.py no longer defines anomaly_transform"
    ns = {"np": np, "warnings": warnings}
    exec(compile(ast.Module(body=keep, type_ignores=[]), "trainprobe.py",
                 "exec"), ns)
    return ns["anomaly_transform"]


TRANSFORM = _lift_anomaly_transform()


def bin_of(iso):
    return int((np.datetime64(iso, "D") - EPOCH).astype(np.int64) // 5)


def date_of(b):
    return str(EPOCH + np.timedelta64(int(b) * 5, "D"))


def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        h.update(fh.read())
    return h.hexdigest()


def _field(rng, T, H, W, C, months, nan=0.07, static=()):
    """Seasonal cycle + a trend + noise, NaN holes, float16 — with values
    spanning several binades so float32 per-year sums really do round."""
    moy = np.array([int(m[5:7]) - 1 for m in months])
    yr = np.array([int(m[:4]) for m in months], np.float64)
    X = rng.normal(0, 0.7, size=(T, H, W, C))
    X += np.sin(2 * np.pi * moy / 12)[:, None, None, None] * \
        rng.uniform(0.5, 3.0, size=(1, H, W, C))
    X += ((yr - 2015) * 0.05)[:, None, None, None]
    X *= rng.choice([1e-3, 1.0, 30.0], size=(1, 1, 1, C))
    X[rng.random(X.shape) < nan] = np.nan
    X[:, 0, 0, :] = np.nan                    # one cell never observed
    for c in static:
        X[:, :, :, c] = rng.normal(size=(H, W))[None]
    return X.astype(np.float16)


@pytest.fixture(scope="module")
def toy(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("f7mon")
    rng = np.random.default_rng(1)
    bins = np.arange(bin_of("2007-01-01"), bin_of("2023-12-31") + 1,
                     dtype=np.int64)
    months = [date_of(b)[:7] for b in bins]
    H, W, H1, W1 = 6, 8, 3, 4
    lats, lons = np.arange(H) * 1.0 - 3.0, np.arange(W) * 1.0 - 4.0
    lat1, lon1 = np.arange(H1) * 2.0 - 3.0, np.arange(W1) * 2.0 - 4.0
    b_oc = bin_of("2015-01-01")
    oc_bins = np.arange(b_oc, bins[-1] + 1, dtype=np.int64)
    rg_bins = np.array([bin_of(f"{m}-15") for m in sorted(set(months))],
                       np.int64)
    rg_bins = rg_bins[rg_bins >= bins[0]]
    rows = {"g025": bins, "g100": bins, "rg100": rg_bins, "oc025": oc_bins}
    arrays = {
        "g025": _field(rng, len(bins), H, W, 3, months),
        "g100": _field(rng, len(bins), H1, W1, 4, months, static=(3,)),
        "rg100": _field(rng, len(rg_bins), H1, W1, 3,
                        [date_of(b)[:7] for b in rg_bins]),
        "oc025": _field(rng, len(oc_bins), H, W, 2,
                        [date_of(b)[:7] for b in oc_bins]),
    }
    chans = {"g025": ["cur_speed", "ssh", "sst"],
             "g100": ["t2m", "sp", "skt", "tsoil"],
             "rg100": ["rg_t10", "rg_t100", "rg_s10"],
             "oc025": ["log_chl", "chl_cov"]}
    norm = {g: np.stack([rng.normal(5, 3, len(c)), rng.uniform(0.5, 4, len(c))],
                        axis=1).astype(np.float32) for g, c in chans.items()}
    npz = str(tmp / f"{STEM}.npz")
    meta = dict(bin_index=bins, months=np.array(months), lats=lats, lons=lons,
                lat1=lat1, lon1=lon1, epoch=np.array("1982-01-01"),
                pentad_days=np.array(5), rg_bin_index=rg_bins,
                rg_months=np.array([date_of(b)[:7] for b in rg_bins]),
                oc_bin_first=np.array(b_oc), oc025_bin_index=oc_bins,
                groups=np.array(list(GROUPS)))
    for g in chans:
        meta[f"chan_{g}"] = np.array(chans[g])
        meta[f"norm_{g}"] = norm[g]
    save_tensor(npz, dict(arrays), **meta)
    groups = {}
    for g, arr in arrays.items():
        p = str(tmp / f"{STEM}_X_{g}.npy")
        la, lo = (lats, lons) if g in ("g025", "oc025") else (lat1, lon1)
        with open(p, "rb") as fh:
            hl, shp, dt, fo = parse_npy_header(fh.read(256))
        extra = dict(live_only=True) if g == "rg100" else None
        groups[g] = group_block(
            os.path.basename(p), "https://example.invalid/" + g,
            os.path.getsize(p), sha(p), hl, shp, dt, fo, chans[g], norm[g],
            grid_block(len(la), len(lo), la[0], lo[0], float(la[1] - la[0])),
            extra=extra)
    idx = dict(stem=STEM, groups=groups, fixture=False)
    idx_path = str(tmp / "family7_index.json")
    json.dump(idx, open(idx_path, "w"))
    # The published climatology of the toy — through E-083's own exporter
    # and index writer, the way data/family7_clim_index.json was made.
    clim_dir = str(tmp / "clim")
    E.export(npz, idx_path, clim_dir, copy="ram", write_netcdf=False)
    clim_index = str(tmp / "family7_clim_index.json")
    PC.main(["index", "--out", clim_dir, "--f7-index", idx_path, "--index",
             clim_index, "--local", "--no-cors"])
    out = str(tmp / "monthly")
    res = M.export(npz, idx_path, out, clim_index=clim_index)
    return dict(tmp=tmp, npz=npz, idx=idx, idx_path=idx_path, out=out,
                res=res, rows=rows, arrays=arrays, chans=chans,
                clim_dir=clim_dir, clim_index=clim_index)


def row_cal(rows):
    d = [date_of(b) for b in rows]
    return (np.array([int(x[5:7]) - 1 for x in d]),
            np.array([int(x[:4]) for x in d]))


def load(toy, g):
    S = np.load(os.path.join(toy["out"], g, "sum.npy"))
    N = np.load(os.path.join(toy["out"], g, "count.npy"))
    st = json.load(open(os.path.join(toy["out"], g, "stats.json")))
    return S, N, st


# ------------------------------------------------------------------ (a) -----
YEAR_SETS = {
    "all": lambda y: np.ones_like(y, bool),
    "scattered": lambda y: np.isin(y, [2008, 2011, 2012, 2013, 2019, 2022]),
    "one_year": lambda y: y == 2016,
    "dev": lambda y: ~np.isin(y, [2009, 2017, 2023]),
    "paper": lambda y: ~(np.isin(y, [2009, 2017]) | (y >= 2021)),
}


@pytest.mark.parametrize("yset", list(YEAR_SETS))
@pytest.mark.parametrize("group", GROUPS)
def test_a_any_year_set_reproduces_a_direct_anomaly_transform(toy, group,
                                                              yset):
    moy, yrs = row_cal(toy["rows"][group])
    train = YEAR_SETS[yset](yrs)
    if not train.any():
        pytest.skip(f"{group} has no row in {yset}")
    A = np.array(toy["arrays"][group])
    stats = {}
    TRANSFORM(A, moy, ~train, np.zeros(A.shape[2], bool), chunk=17,
              stats=stats)
    want = np.ascontiguousarray(np.transpose(stats["clim"], (0, 3, 1, 2)))
    S, N, st = load(toy, group)
    years = np.array(st["years"])
    keep = YEAR_SETS[yset](years)
    static = set(st["static_chans"])
    assert static == {toy["chans"][group][c] for c in range(A.shape[3])
                      if c not in stats["dynamic"]}
    worst = 0.0
    for m in range(12):
        for c in range(A.shape[3]):
            mean, n, bound = M.compose(S[m, c], N[m, c], keep)
            r = M.compare_plane(mean, n, bound, want[m, c],
                                static=st["chans"][c] in static)
            assert r["ok"], (group, yset, m, c, r)
            worst = max(worst, r["max_ulp"])
    # a few float32 ulps of the mean at most on this toy (measured: < 2)
    assert worst < 4, worst
    # and the composition is not vacuous: plenty of finite cells
    assert np.isfinite(want[:, 0]).sum() > 0


def test_a_static_channel_is_flagged_from_the_function_not_recomputed(toy):
    _, _, st = load(toy, "g100")
    assert st["static_chans"] == ["tsoil"]
    assert "anomaly_transform" in st["static_chans_source"]


# ------------------------------------------------------------------ (b) -----
@pytest.mark.parametrize("group", GROUPS)
def test_b_counts_are_uint8_bounded_and_sums_bit_exact(toy, group):
    S, N, st = load(toy, group)
    assert S.dtype == np.dtype("<f4") and N.dtype == np.dtype("u1")
    assert int(N.max()) <= 7 and st["max_count"] == int(N.max())
    moy, yrs = row_cal(toy["rows"][group])
    X = toy["arrays"][group]
    y0 = st["year_first"]
    bins = np.zeros((st["n_years"], 12), int)
    for y in np.unique(yrs):
        for m in range(12):
            sel = (yrs == y) & (moy == m)
            bins[y - y0, m] = sel.sum()
            blk = X[sel].astype(np.float64)
            fin = np.isfinite(blk)
            want_n = fin.sum(axis=0)
            want_s = np.where(fin, blk, 0.0).sum(axis=0).astype(np.float32)
            got_n = np.moveaxis(N[m, :, y - y0], 0, -1)
            got_s = np.moveaxis(S[m, :, y - y0], 0, -1)
            assert np.array_equal(got_n, want_n), (group, y, m)
            assert got_s.tobytes() == want_s.tobytes(), (group, y, m)
    assert bins.tolist() == st["bins_per_month"]
    assert (np.asarray(st["bins_per_month"]) <= 7).all()
    # where nothing was finite the stored sum is exactly 0.0, not NaN
    assert not np.isnan(S).any() and (S[N == 0] == 0).all()


def test_b_more_than_seven_rows_in_a_month_is_refused():
    moy = np.zeros(8, int)
    yrs = np.full(8, 2010)
    with pytest.raises(SystemExit, match="at most 7"):
        M.month_runs(moy, yrs, np.ones(8, bool))
    assert len(M.month_runs(moy[:7], yrs[:7], np.ones(7, bool))) == 1


def test_b_an_unsorted_time_axis_is_refused():
    moy = np.array([0, 0, 1, 0])
    yrs = np.full(4, 2010)
    with pytest.raises(SystemExit, match="two separate runs"):
        M.month_runs(moy, yrs, np.ones(4, bool))
    # an invalid row (off-axis) splits nothing and enters no run
    runs = M.month_runs(np.array([0, 0, 1]), np.full(3, 2010),
                        np.array([True, False, True]))
    assert [(a, b) for a, b, _, _ in runs] == [(0, 1), (2, 3)]


# ------------------------------------------------------------------ (c) -----
def test_c_rg100_and_oc025_are_summed_on_their_own_axes(toy):
    _, N_rg, st_rg = load(toy, "rg100")
    assert st_rg["row_kind"] == "monthly"
    assert st_rg["year_first"] == 2007 and st_rg["year_last"] == 2023
    assert set(np.ravel(st_rg["bins_per_month"])) == {1}
    assert int(N_rg.max()) == 1
    _, _, st_oc = load(toy, "oc025")
    assert st_oc["row_kind"] == "pentad"
    # the pentad holding 2015-01-01 OPENS on 2014-12-29, so by the month rule
    # its row is a December-2014 row: the group's own axis starts in 2014
    assert date_of(toy["rows"]["oc025"][0]) == "2014-12-29"
    assert st_oc["year_first"] == 2014 and st_oc["n_years"] == 10
    assert st_oc["bins_per_month"][0] == [0] * 11 + [1]
    assert sum(map(sum, st_oc["bins_per_month"])) == len(toy["rows"]["oc025"])
    _, _, st_g = load(toy, "g025")
    # the master axis opens on the pentad holding 2007-01-01, which starts in
    # December 2006 — one December-2006 row, so the axis is 2006–2023
    assert st_g["year_first"] == 2006 and st_g["n_years"] == 18
    assert st_g["bins_per_month"][0] == [0] * 11 + [1]
    # the falsifier for (c): charging rg100's rows to the MASTER calendar's
    # first rows would put them in the wrong months
    moy_m, yrs_m = row_cal(toy["rows"]["g025"])
    moy_rg, yrs_rg = row_cal(toy["rows"]["rg100"])
    n = len(moy_rg)
    assert not np.array_equal(moy_m[:n], moy_rg)


# ------------------------------------------------------------------ (d) -----
@pytest.mark.parametrize("role,name", [("sum", "sum.npy"),
                                       ("count", "count.npy")])
def test_d_one_plane_and_one_run_of_years_at_the_stated_offset(toy, role,
                                                               name):
    p = os.path.join(toy["out"], "g025", name)
    A = np.load(p)
    _, C, Y, H, W = A.shape
    with open(p, "rb") as fh:
        hl = parse_npy_header(fh.read(4096))[0]
    item = A.dtype.itemsize
    raw = open(p, "rb").read()
    m, c, yi = 1, 2, 5
    off = M.plane_offset(hl, C, Y, H, W, item, m, c, yi)
    plane = np.frombuffer(raw[off:off + H * W * item], A.dtype).reshape(H, W)
    assert plane.tobytes() == A[m, c, yi].tobytes()
    a, b = 3, 11                     # a run of years is ONE contiguous range
    lo = M.plane_offset(hl, C, Y, H, W, item, m, c, a)
    hi = M.plane_offset(hl, C, Y, H, W, item, m, c, b) + H * W * item
    run = np.frombuffer(raw[lo:hi], A.dtype).reshape(b - a + 1, H, W)
    assert run.tobytes() == A[m, c, a:b + 1].tobytes()


def test_d_the_canonical_tensor_is_not_written(toy):
    d = load_tensor(toy["npz"])
    for g, arr in toy["arrays"].items():
        assert np.asarray(d[f"X_{g}"]).tobytes() == arr.tobytes(), g


# ------------------------------------------------------------------ (e) -----
def test_e_verify_passes_against_the_published_climatology(toy, tmp_path):
    rep = str(tmp_path / "verify.json")
    r = M.verify(toy["out"], toy["clim_index"], toy["clim_dir"], report=rep)
    assert r["ok"] and json.load(open(rep))["ok"]
    for g in GROUPS:
        for v in ("all", "dev", "paper"):
            x = r["groups"][g][v]
            assert x["ok"] and x["n_cells"] > 0
            assert x["n_train_bins"] == x["n_train_bins_published"]
            assert x["max_over_bound"] <= 1.0


def test_e_verify_catches_a_perturbed_sum(toy, tmp_path):
    import shutil
    bad = str(tmp_path / "bad")
    shutil.copytree(toy["out"], bad)
    p = os.path.join(bad, "g025", "sum.npy")
    S = np.load(p, mmap_mode="r+")
    N = np.load(os.path.join(bad, "g025", "count.npy"))
    m, c = 4, 2
    yi, h, w = [int(x[0]) for x in np.nonzero(N[m, c] > 0)]
    # the smallest change that the rigorous bound cannot absorb
    S[m, c, yi, h, w] = np.float32(S[m, c, yi, h, w]) + \
        np.float32(64) * np.spacing(np.float32(abs(S[m, c, yi, h, w]) + 1))
    S.flush()
    del S
    with pytest.raises(SystemExit, match="VERIFY FAILED"):
        M.verify(bad, toy["clim_index"], toy["clim_dir"], groups=["g025"],
                 report=str(tmp_path / "v.json"))
    assert json.load(open(tmp_path / "v.json"))["ok"] is False


# ------------------------------------------------------------------ (f) -----
class _FakeApi:
    def __init__(self):
        self.commits = []

    def create_commit(self, repo_id, repo_type, operations, commit_message):
        self.commits.append((repo_id, repo_type,
                             [(op.path_in_repo, op.path_or_fileobj)
                              for op in operations], commit_message))


def _fake_fetch(out, corrupt=None):
    def fetch(session, url, a, b):
        rel = url.split("/monthly/", 1)[1]
        data = open(os.path.join(out, rel), "rb").read()
        total = len(data)
        if corrupt and rel.endswith(corrupt):
            data = data[:-1] + bytes([data[-1] ^ 1])
        return data[a:b + 1], f"bytes {a}-{b}/{total}"
    return fetch


def _measure(out):
    def measure(url):
        rel = url.split("/monthly/", 1)[1]
        return dict(status=206, final_url="https://cdn.example/x",
                    access_control_allow_origin="*",
                    access_control_expose_headers="*", accept_ranges="bytes",
                    content_range="bytes 0-4095/1",
                    head=open(os.path.join(out, rel), "rb").read(4096))
    return measure


@pytest.fixture(scope="module")
def manifest(toy, tmp_path_factory):
    os.environ["HF_TOKEN"] = "hf_test_token_not_real"
    api = _FakeApi()
    try:
        man = P.main(["upload", "--out", toy["out"], "--f7-index",
                      toy["idx_path"]], api=api)
    finally:
        os.environ.pop("HF_TOKEN", None)
    return man, api


def test_f_upload_writes_a_manifest_and_one_commit_per_group(toy, manifest):
    man, api = manifest
    assert man["groups"] == list(GROUPS)
    assert [c[3].split("group ")[1].split(" ")[0] for c in api.commits] == \
        list(GROUPS)
    rels = {rel for c in api.commits for rel, _ in c[2]}
    assert rels == {f"tensors/{STEM}/monthly/{g}/{n}" for g in GROUPS
                    for n in ("sum.npy", "count.npy", "stats.json")}
    on_disk = json.load(open(os.path.join(toy["out"], "manifest.json")))
    assert on_disk["files"] == man["files"]
    for g in GROUPS:
        for role, name in P.FILES:
            assert man["files"][g][role]["sha256"] == \
                sha(os.path.join(toy["out"], g, name))


def test_f_index_restores_from_the_hub_and_measures_cors(toy, manifest,
                                                          tmp_path):
    rep = str(tmp_path / "verify.json")
    M.verify(toy["out"], toy["clim_index"], toy["clim_dir"], report=rep)
    out_index = str(tmp_path / "idx.json")
    ix = P.main(["index", "--manifest", os.path.join(toy["out"],
                                                     "manifest.json"),
                 "--f7-index", toy["idx_path"], "--index", out_index,
                 "--verify", rep], fetch=_fake_fetch(toy["out"]),
                measure=_measure(toy["out"]), session=object())
    on = json.load(open(out_index))
    assert on == json.loads(json.dumps(ix))
    assert on["restore_verified"] is True
    assert on["restore_where"] == "hosted runner"
    for role in ("sum", "count"):
        assert on["cors_measured"][role]["origin"] == "https://blauewelt.org"
        assert on["cors_measured"][role]["status"] == 206
    assert on["axes"] == ["month", "channel", "year", "lat", "lon"]
    assert on["reproduction"]["ok"] is True
    for g in GROUPS:
        blk = on["groups"][g]
        for k in P.COPY_KEYS:
            assert blk[k] == toy["idx"]["groups"][g][k], (g, k)
        assert blk["sum"]["shape"][2] == blk["n_years"] == len(blk["years"])
        assert blk["sum"]["plane_bytes"] == blk["grid"]["ny"] * \
            blk["grid"]["nx"] * 4
        assert blk["count"]["plane_bytes"] * 4 == blk["sum"]["plane_bytes"]
        assert blk["sum"]["url"] == on["base"] + f"{g}/sum.npy"
        assert set(on["reproduction"]["groups"][g]) == {"all", "dev", "paper"}


def test_f_index_refuses_a_byte_that_does_not_come_back(toy, manifest,
                                                         tmp_path):
    out_index = str(tmp_path / "idx.json")
    with pytest.raises(SystemExit, match="downloaded back"):
        P.main(["index", "--manifest", os.path.join(toy["out"],
                                                    "manifest.json"),
                "--f7-index", toy["idx_path"], "--index", out_index,
                "--no-cors"],
               fetch=_fake_fetch(toy["out"], corrupt="rg100/count.npy"),
               session=object())
    assert not os.path.exists(out_index)


def test_f_index_refuses_another_stem_and_a_failed_verify(toy, manifest,
                                                          tmp_path):
    other = json.load(open(toy["idx_path"]))
    other["stem"] = "family7_global025_pentad_l2"
    p = str(tmp_path / "f7.json")
    json.dump(other, open(p, "w"))
    with pytest.raises(SystemExit, match="REFUSING"):
        P.main(["index", "--local", "--out", toy["out"], "--f7-index", p,
                "--index", str(tmp_path / "i.json"), "--no-cors"])
    bad = str(tmp_path / "bad_verify.json")
    json.dump({"ok": False, "groups": {}}, open(bad, "w"))
    with pytest.raises(SystemExit, match="do NOT reproduce"):
        P.main(["index", "--local", "--out", toy["out"], "--f7-index",
                toy["idx_path"], "--index", str(tmp_path / "j.json"),
                "--no-cors", "--verify", bad])
    assert not os.path.exists(tmp_path / "i.json")
    assert not os.path.exists(tmp_path / "j.json")


def test_f_upload_needs_the_token_from_the_environment(toy, tmp_path,
                                                       monkeypatch):
    monkeypatch.delenv("HF_TOKEN", raising=False)
    with pytest.raises(SystemExit, match="HF_TOKEN"):
        P.main(["upload", "--out", toy["out"], "--f7-index", toy["idx_path"],
                "--manifest", str(tmp_path / "m.json")], api=_FakeApi())


def test_f_stream_hash_matches_and_keeps_order(tmp_path):
    data = np.random.default_rng(3).bytes(1_000_003)
    p = tmp_path / "x.bin"
    p.write_bytes(data)

    def fetch(session, url, a, b):
        return data[a:b + 1], f"bytes {a}-{b}/{len(data)}"
    got, head, total = P.hub_stream_sha256(None, "u/monthly/x", len(data),
                                           workers=4, chunk=65536,
                                           fetch=fetch)
    assert got == hashlib.sha256(data).hexdigest()
    assert head == data[:4096] and total == len(data)


# ------------------------------------------------------------------ (g) -----
def test_g_committed_fixture_agrees_with_its_own_files():
    path = M.FIXTURE_INDEX_OUT
    assert os.path.exists(path), \
        "run `python3 ml/export_family7_monthly.py fixture`"
    ix = json.load(open(path))
    f7 = json.load(open(E.FIXTURE_INDEX))
    assert ix["stem"] == f7["stem"] and ix["fixture"] is True
    assert ix["fixture_dir"] == "data/family7_monthly/fixture/"
    assert ix["restore_verified"] is False and ix["cors_measured"] is None
    assert ix["reproduction"]["ok"] is True
    assert list(ix["groups"]) == list(M.FIXTURE_GROUPS)
    total = os.path.getsize(path)
    for g, blk in ix["groups"].items():
        for k in P.COPY_KEYS:
            assert blk[k] == f7["groups"][g][k], (g, k)
        for role, name in P.FILES:
            rel = blk[role]["url"].split("/monthly/", 1)[1]
            p = os.path.join(ROOT, ix["fixture_dir"], rel)
            assert sha(p) == blk[role]["sha256"], p
            assert os.path.getsize(p) == blk[role]["bytes"]
            total += os.path.getsize(p)
            if role != "stats":
                a = np.load(p, mmap_mode="r")
                assert list(a.shape) == blk[role]["shape"]
                assert a.dtype.str == blk[role]["dtype"]
    assert total < 1e6, f"the fixture is {total / 1e6:.2f} MB"


if __name__ == "__main__":                                  # pragma: no cover
    sys.exit(pytest.main([__file__, "-q"]))

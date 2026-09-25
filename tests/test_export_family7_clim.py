#!/usr/bin/env python3
"""E-083 §5 · `ml/export_family7_clim.py` writes THE function's climatology.

The export's whole claim is that the climatology it publishes is
`ml/trainprobe.py::anomaly_transform`'s own `stats["clim"]` — not a second
computation of a monthly mean — under exactly the year rules of plan §2, each
group on its own months. On a toy four-group tensor (pentads 2007–2023, a
bin-aligned 1°-class group with a static channel, a monthly `rg100`-like
group, an `oc025`-like group starting in 2015):

  (a) exported clim.npy is bit-identical to a direct anomaly_transform call's
      stats["clim"] after the one transpose — through both the RAM copy and
      the scratch-file copy;
  (b) the three versions' t_hold are exactly §2's rules;
  (c) the monthly group and the late group are masked on their OWN dates;
  (d) the NetCDF holds clim * sd + mean, in the index's units;
  (e) stats.json carries the tensor sha256s, the git sha, n_train_bins;
  (f) the index writer refuses a family-7 index with a different stem;
  (g) the index's header_len / shape / plane_bytes agree with the .npy;
  (h) --fixture writes the fixture layout and an index that parses — and the
      COMMITTED fixture agrees with its own files.
Plus: a sha256 mismatch refuses before anything is computed; the upload path
refuses to write an index when a file does not restore byte-identical.

    python3 -m pytest -q tests/test_export_family7_clim.py
"""
import ast
import hashlib
import json
import os
import subprocess
import sys
import warnings

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ML = os.path.join(ROOT, "ml")
if ML not in sys.path:
    sys.path.insert(0, ML)

import export_family7_clim as E                                   # noqa: E402
import publish_family7_clim_index as P                            # noqa: E402
from publish_family7_index import grid_block, group_block          # noqa: E402
from tensor_io import load_tensor, save_tensor                     # noqa: E402
from cone_sampler import GroupSet                                  # noqa: E402

EPOCH = np.datetime64("1982-01-01")
STEM = "family7_toy_pentad"


def _lift_anomaly_transform():
    """THE function, lifted from ml/trainprobe.py's source (no torch here) —
    the same trick tests/test_export_cone_sample.py uses, independent of the
    exporter's own lift so a broken lift there cannot agree with itself."""
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


# --------------------------------------------------------------- the toy ----
def _field(rng, T, H, W, C, months, years, nan=0.05, static=()):
    """Seasonal cycle + a warming trend + noise, 5 % NaN, float16 — and a
    channel in `static` held constant in time (the transform's static case)."""
    moy = np.array([int(m[5:7]) - 1 for m in months])
    yr = np.array(years, np.float64)
    X = rng.normal(0, 0.3, size=(T, H, W, C))
    X += np.sin(2 * np.pi * moy / 12)[:, None, None, None] * \
        rng.uniform(0.5, 1.5, size=(1, H, W, C))
    X += ((yr - 2015) * 0.05)[:, None, None, None]
    X[rng.random(X.shape) < nan] = np.nan
    for c in static:                  # constant in time, and never missing
        X[:, :, :, c] = rng.normal(size=(H, W))[None]
    return X.astype(np.float16)


@pytest.fixture(scope="module")
def toy(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("f7clim")
    rng = np.random.default_rng(0)
    bins = np.arange(bin_of("2007-01-01"), bin_of("2023-12-31") + 1,
                     dtype=np.int64)
    T = len(bins)
    months = [date_of(b)[:7] for b in bins]
    years = [int(m[:4]) for m in months]
    H, W, H1, W1 = 6, 8, 3, 4
    lats, lons = np.arange(H) * 1.0 - 3.0, np.arange(W) * 1.0 - 4.0
    lat1, lon1 = np.arange(H1) * 2.0 - 3.0, np.arange(W1) * 2.0 - 4.0

    # oc025-like: starts 2015-01-01, runs to the end of the axis.
    b_oc = bin_of("2015-01-01")
    oc_bins = np.arange(b_oc, bins[-1] + 1, dtype=np.int64)
    oc_months = [date_of(b)[:7] for b in oc_bins]
    # rg100-like: one row per month, in the pentad holding the 15th.
    ym = sorted({m for m in months})
    rg_bins = np.array([bin_of(f"{m}-15") for m in ym], np.int64)
    # the axis opens on the pentad holding 2007-01-01, which starts in
    # December 2006 — that month's 15th is off the axis, so it has no row
    rg_bins = rg_bins[rg_bins >= bins[0]]
    rg_months = [date_of(b)[:7] for b in rg_bins]

    arrays = {
        "g025": _field(rng, T, H, W, 3, months, years),
        "g100": _field(rng, T, H1, W1, 4, months, years, static=(3,)),
        "rg100": _field(rng, len(rg_bins), H1, W1, 3, rg_months,
                        [int(m[:4]) for m in rg_months]),
        "oc025": _field(rng, len(oc_bins), H, W, 2, oc_months,
                        [int(m[:4]) for m in oc_months]),
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
                rg_months=np.array(rg_months), oc_bin_first=np.array(b_oc),
                oc025_bin_index=oc_bins,
                groups=np.array(["g025", "g100", "rg100", "oc025"]))
    for g in chans:
        meta[f"chan_{g}"] = np.array(chans[g])
        meta[f"norm_{g}"] = norm[g]
    save_tensor(npz, dict(arrays), **meta)

    # The family-7 index for it, through publish_family7_index's own writers.
    groups = {}
    for g, arr in arrays.items():
        p = str(tmp / f"{STEM}_X_{g}.npy")
        la, lo = (lats, lons) if g in ("g025", "oc025") else (lat1, lon1)
        with open(p, "rb") as fh:
            from publish_family7_index import parse_npy_header
            hl, shp, dt, fo = parse_npy_header(fh.read(256))
        groups[g] = group_block(
            os.path.basename(p), "https://example.invalid/" + g,
            os.path.getsize(p), sha(p), hl, shp, dt, fo, chans[g], norm[g],
            grid_block(len(la), len(lo), la[0], lo[0], float(la[1] - la[0])))
    idx = dict(stem=STEM, groups=groups, fixture=False)
    idx_path = str(tmp / "family7_index.json")
    json.dump(idx, open(idx_path, "w"))

    out = str(tmp / "out")
    summary = E.export(npz, idx_path, out, copy="ram")
    return dict(tmp=tmp, npz=npz, idx=idx, idx_path=idx_path, out=out,
                summary=summary, bins=bins, months=months, rg_bins=rg_bins,
                oc_bins=oc_bins, chans=chans, norm=norm, arrays=arrays)


def direct(toy, group, version):
    """A direct call of THE function on a fresh copy of `group`, under the
    masks the exporter used, returning its stats."""
    s = toy["summary"][version][group]
    A = np.array(toy["arrays"][group])
    stats = {}
    TRANSFORM(A, s["moy"], s["t_hold"], np.zeros(A.shape[2], bool), chunk=64,
              stats=stats)
    return stats


# ------------------------------------------------------------------ (a) -----
@pytest.mark.parametrize("version", ["all", "dev", "paper"])
@pytest.mark.parametrize("group", ["g025", "g100", "rg100", "oc025"])
def test_a_clim_npy_is_the_functions_array_bit_for_bit(toy, group, version):
    got = np.load(os.path.join(toy["out"], version, group, "clim.npy"))
    want = direct(toy, group, version)["clim"]
    assert want.shape == (12,) + toy["arrays"][group].shape[1:]
    assert got.dtype == np.float32 and got.flags.c_contiguous
    assert got.shape == (12, want.shape[3], want.shape[1], want.shape[2])
    ref = np.ascontiguousarray(np.transpose(want, (0, 3, 1, 2)))
    assert got.tobytes() == ref.tobytes(), \
        f"{version}/{group}: clim.npy is not stats['clim'] transposed"
    # and it is a climatology, not a zero field: every month is populated
    # for a dynamic channel of a group with data in every month
    assert np.isfinite(got[:, 0]).any(axis=(1, 2)).all()


def test_a_scratch_copy_gives_the_same_bytes_as_the_ram_copy(toy, tmp_path):
    out = str(tmp_path / "disk")
    E.export(toy["npz"], toy["idx_path"], out, versions=["paper"],
             groups=["g100", "rg100"], copy="disk",
             scratch=str(tmp_path / "scratch"), write_netcdf=False)
    for g in ("g100", "rg100"):
        a = open(os.path.join(out, "paper", g, "clim.npy"), "rb").read()
        b = open(os.path.join(toy["out"], "paper", g, "clim.npy"), "rb").read()
        assert a == b
    assert os.listdir(tmp_path / "scratch") == [], \
        "the writable copy must be deleted after each version"


def test_a_the_canonical_tensor_is_not_written(toy):
    d = load_tensor(toy["npz"])
    for g, arr in toy["arrays"].items():
        assert np.asarray(d[f"X_{g}"]).tobytes() == arr.tobytes(), \
            f"{g}: the stored tensor changed — the export wrote in place"


# ------------------------------------------------------------------ (b) -----
def test_b_version_rules_are_exactly_section_2():
    years = np.arange(1982, 2026)
    h = {v: set(years[E.held_out(v, years)].tolist())
         for v in ("all", "dev", "paper")}
    assert h["all"] == set()
    assert h["dev"] == {2009, 2017, 2023}
    assert h["paper"] == {2009, 2017} | set(range(2021, 2026))
    # the plan's pins, one by one
    assert 2009 in h["dev"] and 2009 in h["paper"] and 2009 not in h["all"]
    assert 2017 in h["dev"] and 2017 in h["paper"]
    for y in (2021, 2022, 2024):
        assert y in h["paper"] and y not in h["dev"] and y not in h["all"]
    assert 2023 in h["dev"] and 2023 in h["paper"]   # paper: by the >=2021 rule
    assert 2020 not in h["paper"] and 2010 not in h["dev"]
    assert [v["key"] for v in E.VERSIONS] == ["all", "dev", "paper"]
    for v in E.VERSIONS:
        assert set(v) >= {"key", "name", "holdout_years", "train_span", "rule"}


@pytest.mark.parametrize("version", ["all", "dev", "paper"])
def test_b_master_groups_carry_the_rule_on_their_own_rows(toy, version):
    years = np.array([int(m[:4]) for m in toy["months"]])
    moy = np.array([int(m[5:7]) - 1 for m in toy["months"]])
    rule = {"all": lambda y: np.zeros_like(y, bool),
            "dev": lambda y: np.isin(y, [2009, 2017, 2023]),
            "paper": lambda y: np.isin(y, [2009, 2017]) | (y >= 2021)}[version]
    for g in ("g025", "g100"):
        s = toy["summary"][version][g]
        assert np.array_equal(s["t_hold"], rule(years))
        assert np.array_equal(s["moy"], moy)
        assert s["n_train_bins"] == int((~rule(years)).sum())


# ------------------------------------------------------------------ (c) -----
@pytest.mark.parametrize("version", ["all", "dev", "paper"])
def test_c_monthly_and_late_groups_are_masked_on_their_own_months(toy,
                                                                   version):
    rule = {"all": lambda y: np.zeros_like(y, bool),
            "dev": lambda y: np.isin(y, [2009, 2017, 2023]),
            "paper": lambda y: np.isin(y, [2009, 2017]) | (y >= 2021)}[version]
    for g, rows in (("rg100", toy["rg_bins"]), ("oc025", toy["oc_bins"])):
        dates = [date_of(b) for b in rows]
        y = np.array([int(x[:4]) for x in dates])
        m = np.array([int(x[5:7]) - 1 for x in dates])
        s = toy["summary"][version][g]
        assert len(s["t_hold"]) == len(rows) != len(toy["bins"])
        assert np.array_equal(s["t_hold"], rule(y)), g
        assert np.array_equal(s["moy"], m), g
    # exact counts: rg100 holds one row per month, so dev holds 3 x 12 rows
    n_rg_held = int(toy["summary"][version]["rg100"]["t_hold"].sum())
    assert n_rg_held == {"all": 0, "dev": 36, "paper": 24 + 36}[version]
    # the late group starts at the pentad holding 2015-01-01: no 2009 row
    assert toy["oc_bins"][0] == bin_of("2015-01-01")
    assert int((~toy["summary"]["paper"]["oc025"]["t_hold"]).sum()) == sum(
        1 for b in toy["oc_bins"] if 2014 <= int(date_of(b)[:4]) <= 2020
        and int(date_of(b)[:4]) != 2017)


def test_c_charging_rg100_to_the_master_calendar_would_differ(toy):
    """The falsifier for (c): the master calendar's first rg-length slice is
    NOT the rg100 mask, so a per-group derivation is actually doing work."""
    s = toy["summary"]["dev"]["rg100"]
    years = np.array([int(m[:4]) for m in toy["months"]])
    wrong = np.isin(years[:len(s["t_hold"])], [2009, 2017, 2023])
    assert not np.array_equal(wrong, s["t_hold"])


# ------------------------------------------------------------------ (d) -----
@pytest.mark.parametrize("group", ["g025", "g100", "rg100", "oc025"])
def test_d_netcdf_is_clim_times_sd_plus_mean(toy, group):
    import netCDF4
    clim = np.load(os.path.join(toy["out"], "paper", group, "clim.npy"))
    blk = toy["idx"]["groups"][group]
    with netCDF4.Dataset(os.path.join(toy["out"], "paper", group,
                                      "clim.nc")) as nc:
        assert nc.dimensions["month"].size == 12
        assert nc.dimensions["lat"].size == clim.shape[2]
        assert nc.dimensions["lon"].size == clim.shape[3]
        assert nc.version == "paper" and nc.tensor_stem == STEM
        assert json.loads(nc.holdout_years) == [2009, 2017]
        assert nc.holdout_from == 2021
        assert nc.tensor_sha256 == blk["sha256"]
        assert nc.anomaly_transform == "ml/trainprobe.py::anomaly_transform"
        assert nc.git_sha and nc.generated_utc
        for c, name in enumerate(blk["chans"]):
            v = nc.variables[name]
            v.set_auto_mask(False)
            got = np.asarray(v[:])
            mean, sd = blk["norm"][c]
            want = (clim[:, c].astype(np.float64) * sd + mean).astype(np.float32)
            assert np.array_equal(np.isnan(got), np.isnan(want)), name
            fin = np.isfinite(want)
            assert fin.any()
            np.testing.assert_allclose(got[fin], want[fin], rtol=1e-6,
                                       atol=1e-6 * abs(mean))
            assert v.units == blk["units"][name]
            assert v.long_name == blk["labels"][name]
            assert np.isnan(v._FillValue)


def test_d_static_channel_is_labelled_not_passed_off_as_a_climatology(toy):
    import netCDF4
    st = json.load(open(os.path.join(toy["out"], "all", "g100",
                                     "stats.json")))
    assert st["static_channels"] == ["tsoil"] and 3 not in st["dynamic"]
    clim = np.load(os.path.join(toy["out"], "all", "g100", "clim.npy"))
    assert (clim[:, 3] == 0).all(), "the function zeroes a static channel"
    with netCDF4.Dataset(os.path.join(toy["out"], "all", "g100",
                                      "clim.nc")) as nc:
        assert "STATIC" in nc.variables["tsoil"].comment


# ------------------------------------------------------------------ (e) -----
def _strict(path):
    def bad(c):
        raise ValueError(f"{path}: non-JSON constant {c}")
    return json.load(open(path), parse_constant=bad)


@pytest.mark.parametrize("version", ["all", "dev", "paper"])
@pytest.mark.parametrize("group", ["g025", "rg100"])
def test_e_stats_json_carries_provenance(toy, group, version):
    p = os.path.join(toy["out"], version, group, "stats.json")
    st = _strict(p)
    x_path = os.path.join(toy["tmp"], f"{STEM}_X_{group}.npy")
    assert st["tensor_sha256"] == sha(x_path)
    assert st["tensor_sha256_verified"] is True
    assert st["tensor_sha256_by_group"] == {
        g: b["sha256"] for g, b in toy["idx"]["groups"].items()}
    head = subprocess.run(["git", "-C", ROOT, "rev-parse", "HEAD"],
                          capture_output=True, text=True).stdout.strip()
    assert st["git_sha"] == "unknown" or (head and
                                          st["git_sha"].startswith(head))
    s = toy["summary"][version][group]
    assert st["n_train_bins"] == int((~s["t_hold"]).sum()) == s["n_train_bins"]
    assert st["n_bins"] == len(s["t_hold"])
    assert st["version"] == E.version_block(E.VERSION_BY_KEY[version])
    assert st["chans"] == toy["chans"][group]
    assert st["shape"] == list(np.load(p.replace("stats.json",
                                                 "clim.npy")).shape)
    assert st["dtype"] == "float32"
    ref = direct(toy, group, version)
    for k in ("mu", "sd", "den"):
        want = [None if not np.isfinite(x) else float(x) for x in ref[k]]
        assert st[k] == want, k
    assert st["dynamic"] == list(ref["dynamic"])
    assert st["anomaly_transform"] == "ml/trainprobe.py::anomaly_transform"
    assert len(st["anomaly_transform_src_sha256"]) == 64


def test_e_sha_mismatch_refuses_before_computing(toy, tmp_path):
    bad = json.load(open(toy["idx_path"]))
    bad["groups"]["g100"]["sha256"] = "0" * 64
    p = str(tmp_path / "bad_index.json")
    json.dump(bad, open(p, "w"))
    out = str(tmp_path / "o")
    with pytest.raises(SystemExit, match="sha256"):
        E.export(toy["npz"], p, out, groups=["g100"], copy="ram")
    assert not os.path.exists(out), "nothing may be written on a mismatch"
    # --skip-sha is the toy-only bypass, and it is RECORDED as unverified
    E.export(toy["npz"], p, out, versions=["all"], groups=["g100"],
             copy="ram", skip_sha=True, write_netcdf=False)
    st = _strict(os.path.join(out, "all", "g100", "stats.json"))
    assert st["tensor_sha256_verified"] is False


def test_e_a_tensor_the_index_does_not_describe_is_refused(toy, tmp_path):
    other = json.load(open(toy["idx_path"]))
    other["stem"] = "family7_something_else"
    p = str(tmp_path / "other.json")
    json.dump(other, open(p, "w"))
    with pytest.raises(SystemExit, match="stem"):
        E.export(toy["npz"], p, str(tmp_path / "o"), copy="ram")


# ------------------------------------------------------------------ (f) -----
def test_f_index_refuses_a_different_family7_stem(toy, tmp_path):
    other = json.load(open(toy["idx_path"]))
    other["stem"] = "family7_global025_pentad_l2"
    p = str(tmp_path / "f7.json")
    json.dump(other, open(p, "w"))
    out_index = str(tmp_path / "clim_index.json")
    with pytest.raises(SystemExit, match="REFUSING"):
        P.main(["index", "--out", toy["out"], "--f7-index", p,
                "--index", out_index, "--local", "--no-cors"])
    assert not os.path.exists(out_index)


# ------------------------------------------------------------------ (g) -----
@pytest.fixture(scope="module")
def clim_index(toy):
    path = os.path.join(str(toy["tmp"]), "family7_clim_index.json")
    P.main(["index", "--out", toy["out"], "--f7-index", toy["idx_path"],
            "--index", path, "--local", "--no-cors"])
    return path, _strict(path)


def test_g_index_header_and_shape_agree_with_the_npy(toy, clim_index):
    _, ix = clim_index
    assert ix["stem"] == STEM
    assert [v["key"] for v in ix["versions"]] == ["all", "dev", "paper"]
    assert ix["restore_verified"] is False and ix["cors_measured"] is None
    assert ix["base"].endswith(f"tensors/{STEM}/clim/")
    for v in ("all", "dev", "paper"):
        for g in ("g025", "g100", "rg100", "oc025"):
            rec = ix["files"][v][g]
            p = os.path.join(toy["out"], v, g, "clim.npy")
            with open(p, "rb") as fh:
                major, _ = np.lib.format.read_magic(fh)
                read = (np.lib.format.read_array_header_1_0 if major == 1
                        else np.lib.format.read_array_header_2_0)
                shape, fortran, dtype = read(fh)
                data_off = fh.tell()
            npy = rec["clim_npy"]
            assert npy["header_len"] == data_off
            assert npy["shape"] == list(shape) == list(np.load(p).shape)
            assert npy["dtype"] == dtype.str == "<f4"
            assert npy["fortran_order"] is False and not fortran
            H, W = shape[2], shape[3]
            assert npy["plane_bytes"] == H * W * 4
            assert npy["bytes"] == os.path.getsize(p) == \
                data_off + int(np.prod(shape)) * 4
            assert npy["sha256"] == sha(p)
            assert npy["url"] == ix["base"] + f"{v}/{g}/clim.npy"
            # one plane read at the index's offset IS that (month, channel)
            m, c = 6, shape[1] - 1
            with open(p, "rb") as fh:
                fh.seek(npy["header_len"] + (m * shape[1] + c) *
                        npy["plane_bytes"])
                plane = np.frombuffer(fh.read(npy["plane_bytes"]), "<f4")
            assert plane.reshape(H, W).tobytes() == np.load(p)[m, c].tobytes()
            for role, name in (("clim_nc", "clim.nc"),
                               ("stats", "stats.json")):
                q = os.path.join(toy["out"], v, g, name)
                assert rec[role]["sha256"] == sha(q)
                assert rec[role]["bytes"] == os.path.getsize(q)
            assert rec["n_train_bins"] == \
                toy["summary"][v][g]["n_train_bins"]


def test_g_channel_metadata_is_copied_from_the_family7_index(toy, clim_index):
    _, ix = clim_index
    for g, blk in toy["idx"]["groups"].items():
        for k in ("grid", "chans", "labels", "units", "ramp", "sign", "norm"):
            assert ix["groups"][g][k] == blk[k], (g, k)
        assert ix["groups"][g]["tensor_sha256"] == blk["sha256"]


# ------------------------------------------------------------------ upload --
class _FakeApi:
    def __init__(self):
        self.commits = []

    def create_commit(self, repo_id, repo_type, operations, commit_message):
        self.commits.append((repo_id, repo_type,
                             [(op.path_in_repo, op.path_or_fileobj)
                              for op in operations], commit_message))


def _fake_download(api, tmp, corrupt=None):
    store = {}

    def download(path_in_repo):
        if not store:
            for _, _, ops, _ in api.commits:
                for rel, local in ops:
                    store[rel] = open(local, "rb").read()
        data = store[path_in_repo]
        if corrupt and path_in_repo.endswith(corrupt):
            data = data[:-1] + bytes([data[-1] ^ 1])
        p = os.path.join(tmp, path_in_repo.replace("/", "__"))
        open(p, "wb").write(data)
        return p
    return download


def test_upload_refuses_to_index_a_file_that_does_not_restore(toy, tmp_path,
                                                             monkeypatch):
    monkeypatch.setenv("HF_TOKEN", "hf_test_token_not_real")
    api = _FakeApi()
    out_index = str(tmp_path / "idx.json")
    with pytest.raises(SystemExit, match="downloaded back"):
        P.main(["upload", "--out", toy["out"], "--f7-index", toy["idx_path"],
                "--index", out_index, "--no-cors"], api=api,
               download=_fake_download(api, str(tmp_path),
                                       corrupt="rg100/clim.npy"))
    assert not os.path.exists(out_index)
    # one commit per version, every file under tensors/<stem>/clim/<v>/<g>/
    assert [c[3].split("version ")[1].split(" ")[0] for c in api.commits] == \
        ["all", "dev", "paper"]
    rels = {rel for c in api.commits for rel, _ in c[2]}
    assert f"tensors/{STEM}/clim/paper/oc025/clim.npy" in rels
    assert len(rels) == 3 * 4 * 3


def test_upload_writes_a_restore_verified_index(toy, tmp_path, monkeypatch):
    monkeypatch.setenv("HF_TOKEN", "hf_test_token_not_real")
    api = _FakeApi()
    out_index = str(tmp_path / "idx.json")

    def measure(url):
        local = os.path.join(toy["out"], *url.split("/clim/")[1].split("/"))
        return dict(status=206, final_url="https://cdn.example/x",
                    access_control_allow_origin="*",
                    access_control_expose_headers="*", accept_ranges="bytes",
                    content_range="bytes 0-4095/1", head=open(local, "rb")
                    .read(4096))
    P.main(["upload", "--out", toy["out"], "--f7-index", toy["idx_path"],
            "--index", out_index], api=api,
           download=_fake_download(api, str(tmp_path)), measure=measure)
    ix = _strict(out_index)
    assert ix["restore_verified"] is True
    assert ix["cors_measured"]["status"] == 206
    assert ix["cors_measured"]["measured_file"].endswith("all/g025/clim.npy")


def test_upload_needs_the_token_from_the_environment(toy, tmp_path,
                                                     monkeypatch):
    monkeypatch.delenv("HF_TOKEN", raising=False)
    with pytest.raises(SystemExit, match="HF_TOKEN"):
        P.main(["upload", "--out", toy["out"], "--f7-index", toy["idx_path"],
                "--index", str(tmp_path / "i.json")], api=_FakeApi(),
               download=lambda p: p)


def test_cors_refuses_a_url_that_serves_other_bytes(toy):
    p = os.path.join(toy["out"], "all", "g025", "clim.npy")
    with pytest.raises(SystemExit, match="different bytes"):
        P.cors_record("u", p, measure=lambda u: dict(
            status=206, final_url="https://h/x", content_range="bytes 0-3/9",
            access_control_allow_origin="*", head=b"nope"))
    with pytest.raises(SystemExit, match="206"):
        P.cors_record("u", p, measure=lambda u: dict(
            status=200, final_url="https://h/x", content_range=None,
            access_control_allow_origin="*", head=b""))


# ------------------------------------------------------------------ (h) -----
def test_h_fixture_mode_writes_the_layout_and_an_index(tmp_path):
    out = str(tmp_path / "fixture")
    E.main(["--fixture", "--out", out])
    ix = _strict(os.path.join(out, "family7_clim_index.json"))
    f7 = json.load(open(E.FIXTURE_INDEX))
    assert ix["stem"] == f7["stem"] and ix["fixture"] is True
    assert list(ix["groups"]) == list(E.FIXTURE_GROUPS)
    for v in ("all", "dev", "paper"):
        for g in E.FIXTURE_GROUPS:
            for name in ("clim.npy", "stats.json"):
                assert os.path.exists(os.path.join(out, v, g, name))
            assert ix["files"][v][g]["clim_nc"] is None
            # 2010 is a training year under all three rules
            assert ix["files"][v][g]["n_train_bins"] == \
                f7["groups"][g]["n_bins"]


def test_h_committed_fixture_agrees_with_its_own_files():
    path = E.FIXTURE_CLIM_INDEX
    assert os.path.exists(path), \
        "run `python3 ml/export_family7_clim.py --fixture`"
    ix = _strict(path)
    f7 = json.load(open(E.FIXTURE_INDEX))
    assert ix["stem"] == f7["stem"]
    assert ix["fixture_dir"] == "data/family7_clim/fixture/"
    total = 0
    for v, gs in ix["files"].items():
        for g, rec in gs.items():
            for role in ("clim_npy", "stats"):
                rel = rec[role]["url"].split("/clim/", 1)[1]
                p = os.path.join(ROOT, ix["fixture_dir"], rel)
                assert sha(p) == rec[role]["sha256"], p
                assert os.path.getsize(p) == rec[role]["bytes"]
                total += os.path.getsize(p)
            a = np.load(p.replace("stats.json", "clim.npy"))
            assert list(a.shape) == rec["clim_npy"]["shape"]
            for k in ("chans", "norm", "grid", "units", "labels"):
                assert ix["groups"][g][k] == f7["groups"][g][k]
    assert total < 2.5e6, f"the fixture is {total / 1e6:.2f} MB"


if __name__ == "__main__":                                  # pragma: no cover
    sys.exit(pytest.main([__file__, "-q"]))

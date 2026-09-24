#!/usr/bin/env python3
"""Tests for the family-1 build framework (E-082) — no network.

    python3 -m pytest -q tests/test_build_family1_stores.py
    python3 tests/test_build_family1_stores.py          # same, via pytest.main

What each group is FOR:

  schema 3        int64 `time_s` written, read back, binned by integer
                  floor_divide, negative int16 bins before 1982, refused past
                  the int16 bin range, and a store.json that disagrees with
                  its column refused.
  the registry    discovery in name order; a duplicate store, a public store
                  whose licence forbids redistribution, a missing family and
                  a module without ADAPTER are all REFUSED.
  routing         the family -> prefixes, the private track -> the private
                  repository and its token order; the publish stage asserts
                  the repository it will write and never touches the public
                  one for a private adapter (a fake Hub records every call).
  credentials     refused at preflight, before a byte is read.
  ghcnd           the smoke (shuffled lines, flags, bounds, an unknown
                  station, a bad date) through all five stages; the streaming
                  and memory assemblers write the same bytes for an int64
                  store; platforms.json lands in the sha256 block; the probe's
                  count equals the truth; the probe stops early on a
                  date-ordered file and still counts the whole month; a
                  truncated gzip and a malformed line are refused; an empty
                  listing is refused.
  parts on a hub  a family-1 year pushed and pulled under its own prefix, and
                  a private store's parts refused a public repository.
  the workflow    dispatch-only, under 25 inputs, no secret in a run: block,
                  source credentials gated to hosted runners, the .netrc
                  removed in an always() step, the probe artifact named
                  probe-<store>-<month>.
"""
import argparse
import gzip
import io
import json
import os
import shutil
import sys
import types

import numpy as np
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "ml"))

import build_family10_stores as b10                             # noqa: E402
import build_family1_stores as b1                               # noqa: E402
import family10_parts_hub as ph                                 # noqa: E402
import family10_store as f10                                    # noqa: E402
from family1 import adapters as fam                             # noqa: E402
from family1.adapters import ghcnd                              # noqa: E402

WF = os.path.join(ROOT, ".github", "workflows", "family1-build.yml")


# ------------------------------------------------------------------ helpers --
def ns(**over):
    d = dict(store="ghcnd", work="", source_dir="", start="", end="2024-12-31",
             stage="all", force=False, attempts=1, qc_keep=2,
             check_chunk_rows=b10.CHECK_CHUNK_ROWS, assemble="auto",
             parts_from_hub=False, push_parts=False,
             allow_missing_years=False, probe_month="", smoke=False)
    d.update(over)
    return argparse.Namespace(**d)


def ghcnd_ctx(tmp, **over):
    src = os.path.join(tmp, "src")
    ad = ghcnd.GHCNDAdapter()
    lo, hi = (b10.parse_date(x) for x in ad.smoke_window)
    truth = ad.smoke_sources(src, lo, hi)
    a = ns(work=os.path.join(tmp, "work"), source_dir=src,
           start=ad.smoke_window[0], end=ad.smoke_window[1], **over)
    return b10.Ctx(a, adapter=ad, layout=b1.layout_for(ad)), truth, src


def hash_dir(d):
    return {n: b10.sha256(os.path.join(d, n)) for n in sorted(os.listdir(d))
            if n != "store.json"}


# ================================================================ schema 3 ==
class _Int64Adapter(b10.SourceAdapter):
    store = "s3test"
    channels = (("x", "u", -10.0, 10.0),)
    first_year = 1763
    time_dtype = "int64"


def test_schema3_packs_int64_and_bins_by_integer_floor_divide():
    days = [b10.parse_date(s) for s in
            ("1763-01-01", "1899-12-31", "1913-12-13", "1981-12-31",
             "1982-01-01", "2060-06-01")]
    t = [b10.seconds_since_epoch(d) for d in days]
    assert t[0] < f10.TIME_S_MIN and t[-1] > f10.TIME_S_MAX
    ad = _Int64Adapter()
    r = ad.pack(t, [0.0] * 6, [0.0] * 6, np.zeros((6, 1)), [1] * 6, [0] * 6)
    assert r["time_s"].dtype == np.int64
    assert r["time_s"].tolist() == t
    assert r["bin"].dtype == np.int16
    assert r["bin"].tolist() == (np.array(t, np.int64) // 432000).tolist()
    assert r["bin"][0] < 0 and r["bin"][3] == -1 and r["bin"][4] == 0
    # int32 still refuses 1763, with the 2050 message family 10 pins
    with pytest.raises(ValueError, match="int32"):
        b10._pack(t[:1], [0.0], [0.0], np.zeros((1, 1)), [0], [0], 1)
    # the int16 bin is the int64 store's limit: 1500 is outside it
    far = b10.seconds_since_epoch(b10.parse_date("1500-01-01"))
    with pytest.raises(ValueError, match="int16"):
        ad.pack([far], [0.0], [0.0], np.zeros((1, 1)), [0], [0])


def test_an_int32_adapter_before_1914_is_refused():
    class Bad(_Int64Adapter):
        time_dtype = "int32"
    with pytest.raises(ValueError, match="int64"):
        b10.time_dtype_name(Bad())
    with pytest.raises(ValueError):
        b10.time_dtype_name(type("X", (_Int64Adapter,),
                                 {"time_dtype": "float32"})())


def _write_s3_store(path, times, time_dtype="int64", meta_dtype="int64"):
    os.makedirs(path, exist_ok=True)
    t = np.asarray(sorted(times), np.int64)
    b = (t // 432000).astype(np.int16)
    n = t.size
    bf = int(b.min())
    nb = int(b.max()) - bf + 1
    off = f10.csr_offsets(b, bf, nb)
    arrs = {"bin": b, "time_s": t.astype(time_dtype),
            "lat": np.zeros(n, np.float32), "lon": np.zeros(n, np.float32),
            "values": np.zeros((n, 1), np.float16),
            "platform": np.ones(n, np.int64), "qc": np.zeros(n, np.uint8),
            "fp": np.full((n, 2), -4, np.float16), "bin_offsets": off}
    for k, v in arrs.items():
        np.save(os.path.join(path, k + ".npy"), v)
    meta = {"N": n, "bin_first": bf, "channels": [{"name": "x", "unit": "u"}],
            "schema_version": 3 if meta_dtype == "int64" else 2,
            "sha256": {k + ".npy": b10.sha256(os.path.join(path, k + ".npy"))
                       for k in arrs}}
    if meta_dtype:
        meta["time_dtype"] = meta_dtype
    json.dump(meta, open(os.path.join(path, "store.json"), "w"))
    return t


def test_the_reader_opens_a_schema3_store_and_checks_it(tmp_path):
    times = [b10.seconds_since_epoch(b10.parse_date(s)) for s in
             ("1763-01-01", "1763-01-03", "1850-07-01", "2024-01-01")]
    p = str(tmp_path / "s3")
    t = _write_s3_store(p, times)
    st = f10.Store.open(p, verify=True)
    assert st.schema_version == 3
    assert st["time_s"].dtype == np.int64
    assert st.time_s().tolist() == t.tolist()
    assert np.array_equal(st["bin"], f10.bin_of_seconds(st.time_s()))
    assert st.bin_first < -15000
    b10.check_store(p, None)
    tok = st.knearest(0.0, 0.0, int(st["bin"][1]), k=2, T_max_days=30)
    assert tok["valid"].sum() == 2
    assert sorted(tok["time_s"][tok["valid"]].tolist()) == [t[0], t[1]]


def test_the_reader_refuses_a_time_dtype_that_disagrees_with_the_column(
        tmp_path):
    times = [0, 86400]
    p = str(tmp_path / "bad")
    _write_s3_store(p, times, time_dtype="int64", meta_dtype="int32")
    with pytest.raises(ValueError, match="time_dtype"):
        f10.Store(p)
    p2 = str(tmp_path / "bad2")
    _write_s3_store(p2, times, time_dtype="int32", meta_dtype="int64")
    with pytest.raises(ValueError):
        f10.Store(p2)


def test_check_part_schema_reads_the_time_dtype_from_the_header(tmp_path):
    for dt_, want in ((np.int32, 2), (np.int64, 3)):
        p = str(tmp_path / f"p{want}.npz")
        np.savez(p, time_s=np.zeros(3, dt_), bin=np.zeros(3, np.int16))
        assert f10.check_part_schema(p) == want


# ============================================================ the registry ==
ADAPTER_SRC = '''
import build_family10_stores as f10b
class A(f10b.SourceAdapter):
    store = {store!r}
    family = {family!r}
    distribution = {dist!r}
    licence = {{"name": "x", "redistribution": {red!r}, "derived_works": "free"}}
    channels = (("c", "u", 0.0, 1.0),)
ADAPTER = A
'''


def _pkg(tmp_path, name, mods):
    d = tmp_path / name / "adapters"
    d.mkdir(parents=True)
    (tmp_path / name / "__init__.py").write_text("")
    (d / "__init__.py").write_text("")
    for fn, text in mods.items():
        (d / fn).write_text(text)
    sys.path.insert(0, str(tmp_path))
    return f"{name}.adapters", str(d)


def _src(store="a", family="1tf", dist="public", red="yes"):
    return ADAPTER_SRC.format(store=store, family=family, dist=dist, red=red)


def test_the_registry_discovers_in_name_order_and_skips_helpers(tmp_path):
    pkg, d = _pkg(tmp_path, "regok", {
        "zeta.py": _src("zeta"), "alpha.py": _src("alpha"),
        "_helpers.py": "X = 1\n"})
    reg = fam.discover(pkg, d)
    assert list(reg) == ["alpha", "zeta"]


@pytest.mark.parametrize("mods, match", [
    ({"a.py": _src("same"), "b.py": _src("same")}, "claimed twice"),
    ({"a.py": _src("a", red="no")}, "PRIVATE track"),
    ({"a.py": _src("a", family="7")}, "family"),
    ({"a.py": _src("a", dist="secret")}, "distribution"),
    ({"a.py": _src("a", red="maybe")}, "redistribution"),
    ({"a.py": "X = 1\n"}, "no ADAPTER"),
])
def test_the_registry_refuses(tmp_path, mods, match):
    name = f"regbad{abs(hash(match)) % 10**8}"
    pkg, d = _pkg(tmp_path, name, mods)
    with pytest.raises(fam.RegistryError, match=match):
        fam.discover(pkg, d)


def test_a_private_adapter_with_no_redistribution_is_accepted(tmp_path):
    pkg, d = _pkg(tmp_path, "regpriv", {"p.py": _src("p", dist="private",
                                                     red="no")})
    assert list(fam.discover(pkg, d)) == ["p"]


def test_the_real_registry_holds_ghcnd():
    assert fam.REGISTRY["ghcnd"] is ghcnd.GHCNDAdapter
    for s, cls in fam.REGISTRY.items():
        assert fam.validate(cls, s) == s


# ================================================================= routing ==
def _private_adapter():
    class P(ghcnd.GHCNDAdapter):
        store = "ghcnd_private_test"
        distribution = "private"
        licence = dict(ghcnd.GHCNDAdapter.licence, redistribution="no")
    return P


def test_layouts_route_family_and_track():
    pub = b1.layout_for(ghcnd.GHCNDAdapter())
    assert pub.hf_root == "tensors/family1_tf"
    assert pub.hf_partials == "partials/family1_tf"
    assert pub.cache_dirname == "family1_tf"
    assert pub.prefix("ghcnd") == "tensors/family1_tf/ghcnd"
    assert pub.repo_id == "chfrank/earth-tensors" and not pub.private
    assert pub.token_env == ("HF_TOKEN",)
    assert b1.default_work(ghcnd.GHCNDAdapter()).endswith(
        os.path.join("ml", "cache", "family1_tf"))
    prv = b1.layout_for(_private_adapter()())
    assert prv.repo_id == "chfrank/earth-tensors-private" and prv.private
    assert prv.token_env[:2] == ("HF_PRIVATE_TOKEN", "HF_TOKEN")
    for code, slug in (("1gf", "family1_gf"), ("09tf", "family09_tf")):
        cls = type("G", (ghcnd.GHCNDAdapter,), {"family": code})
        assert b1.layout_for(cls()).hf_root == f"tensors/{slug}"
    # family 10's default layout is unchanged
    d = b10.Layout()
    assert (d.hf_root, d.hf_partials, d.repo_id, d.family) == (
        f10.HF_ROOT, f10.HF_PARTIALS, None, f10.FAMILY)


def test_the_private_token_prefers_hf_private_token(monkeypatch):
    lay = b1.layout_for(_private_adapter()())
    monkeypatch.setenv("HF_TOKEN", "pub")
    monkeypatch.delenv("HF_PRIVATE_TOKEN", raising=False)
    assert lay.token() == ("HF_TOKEN", "pub")
    monkeypatch.setenv("HF_PRIVATE_TOKEN", "priv")
    assert lay.token() == ("HF_PRIVATE_TOKEN", "priv")


class FakeApi:
    def __init__(self, root):
        self.root = root
        self.calls = []

    def create_repo(self, repo, **kw):
        self.calls.append(("create_repo", repo, kw.get("private")))


def _fake_hub(monkeypatch, root, repo):
    api = FakeApi(root)

    def commit(api_, repo_, ops, message, **kw):
        api.calls.append(("commit", repo_, message))
        for rel, local in ops:
            dst = os.path.join(root, repo_, rel)
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            shutil.copyfile(local, dst)

    def upload(api_, repo_, local, rel, message, **kw):
        commit(api_, repo_, [(rel, local)], message)

    def download(repo_, rel, repo_type=None, token=None, local_dir=None):
        api.calls.append(("download", repo_, rel))
        dst = os.path.join(local_dir, rel)
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        shutil.copyfile(os.path.join(root, repo_, rel), dst)
        return dst

    def stream(repo_, rel, token, consume, just_uploaded=False, attempts=12,
               private=False):
        # the restore STREAMS the committed copy (`ph.hub_stream`,
        # 2026-09-24); the private repository is read with the token
        api.calls.append(("stream", repo_, rel, private))
        src = os.path.join(root, repo_, rel)
        if not os.path.exists(src):
            raise IOError(f"{rel}: HTTP 404")
        with open(src, "rb") as fh:
            data = fh.read()
        consume(data)
        return len(data)

    monkeypatch.setattr(b10, "hub_commit", commit)
    monkeypatch.setattr(b10, "hub_add_ops", lambda pairs: list(pairs))
    monkeypatch.setattr(b10, "hub_upload_with_backoff", upload)
    monkeypatch.setattr(ph, "hub_stream", stream)
    fake_hf = types.ModuleType("huggingface_hub")
    fake_hf.hf_hub_download = download
    monkeypatch.setitem(sys.modules, "huggingface_hub", fake_hf)
    return api


def _built_private(tmp):
    cls = _private_adapter()
    ad = cls()
    src = os.path.join(tmp, "src")
    lo, hi = (b10.parse_date(x) for x in ad.smoke_window)
    ad.smoke_sources(src, lo, hi)
    a = ns(store=cls.store, work=os.path.join(tmp, "work"), source_dir=src,
           start=ad.smoke_window[0], end=ad.smoke_window[1])
    ctx = b10.Ctx(a, adapter=ad, layout=b1.layout_for(ad))
    b10.run_stages(ctx, ["index", "fetch", "assemble"], stage_fn=b1.STAGE_FN,
                   deps=b1.DEPS)
    return ctx


def test_a_private_store_publishes_only_to_the_private_repository(
        tmp_path, monkeypatch):
    ctx = _built_private(str(tmp_path))
    hub = str(tmp_path / "hub")
    api = _fake_hub(monkeypatch, hub, None)
    monkeypatch.setattr(ctx, "hub", lambda: (api, ctx.layout.repo_id, "tok"))
    b1.stage_publish(ctx)
    repos = {c[1] for c in api.calls}
    assert repos == {"chfrank/earth-tensors-private"}, api.calls
    assert ("create_repo", "chfrank/earth-tensors-private", True) in api.calls
    assert {c[3] for c in api.calls if c[0] == "stream"} == {True}
    assert os.path.exists(os.path.join(
        hub, "chfrank/earth-tensors-private",
        "tensors/family1_tf/ghcnd_private_test/platforms.json"))
    assert not os.path.exists(os.path.join(hub, "chfrank/earth-tensors"))
    # and the check stage compares the Hub's records with the local ones
    out = b1.stage_check(ctx)
    assert out["hub"]["repo"] == "chfrank/earth-tensors-private"


def test_a_private_store_is_refused_the_public_repository_before_any_call(
        tmp_path, monkeypatch):
    ctx = _built_private(str(tmp_path))
    api = _fake_hub(monkeypatch, str(tmp_path / "hub"), None)
    # a misrouted layout: the public repository id
    monkeypatch.setattr(ctx, "hub", lambda: (api, "chfrank/earth-tensors",
                                             "tok"))
    with pytest.raises(SystemExit, match="REFUSING to publish"):
        b1.stage_publish(ctx)
    assert api.calls == []
    # and a private-looking repo under a layout that does not say private
    ctx.layout.private = False
    monkeypatch.setattr(ctx, "hub", lambda: (
        api, "chfrank/earth-tensors-private", "tok"))
    with pytest.raises(SystemExit, match="REFUSING to publish"):
        b1.stage_publish(ctx)
    assert api.calls == []


def test_a_public_store_is_refused_the_private_repository():
    ad = ghcnd.GHCNDAdapter()
    lay = b1.layout_for(ad)
    assert b10.check_publish_target(ad, lay, "chfrank/earth-tensors")
    with pytest.raises(SystemExit):
        b10.check_publish_target(ad, lay, "chfrank/earth-tensors-private")


def test_distribution_private_overrides_a_public_adapter_and_never_the_reverse(
        tmp_path, monkeypatch):
    """--distribution private: a PUBLIC tier-G adapter whose licence answer is
    pending (seaice_asi) is built, parked and published to the PRIVATE
    repository only, and its store.json says so; the reverse is refused."""
    from family1.adapters import seaice_asi as asi
    from family1.adapters import tide_private
    # the CLI carries the flag, and the default leaves the adapter alone
    p = b1.build_parser()
    assert p.parse_args(["--store", "seaice_asi"]).distribution == ""
    assert p.parse_args(["--store", "seaice_asi", "--distribution",
                         "private"]).distribution == "private"
    assert b1.apply_distribution(asi.SeaIceASIAdapter(), "").distribution \
        == "public"
    # the reverse is refused, and so is nonsense
    with pytest.raises(SystemExit, match="REFUSING --distribution public"):
        b1.apply_distribution(tide_private.ADAPTER(), "public")
    with pytest.raises(SystemExit, match="expected"):
        b1.apply_distribution(asi.SeaIceASIAdapter(), "shared")
    with pytest.raises(SystemExit, match="REFUSING --distribution public"):
        b1.main(["--store", "tide_private", "--distribution", "public",
                 "--work", str(tmp_path / "tp"), "--stage", "index"])

    # a real (synthetic) seaice_asi build under the override
    src = str(tmp_path / "src")
    lo, hi = "2012-12-30", "2013-01-01"
    asi.make_smoke_sources(src, b10.parse_date(lo), b10.parse_date(hi))
    a = ns(store="seaice_asi", work=str(tmp_path / "work"), source_dir=src,
           start=lo, end=hi, allow_unconfirmed_licence=False,
           distribution="private")
    ctx = b1.make_ctx(a, asi.SeaIceASIAdapter)
    assert ctx.adapter.distribution == "private"
    assert asi.SeaIceASIAdapter.distribution == "public"     # class untouched
    assert ctx.layout.private
    assert ctx.layout.repo_id == "chfrank/earth-tensors-private"
    assert ctx.layout.prefix("seaice_asi") == "tensors/family1_gf/seaice_asi"
    b10.run_stages(ctx, ["index", "fetch", "assemble"],
                   stage_fn=b1.GRID_STAGE_FN, deps=b1.DEPS)
    sm = json.load(open(os.path.join(ctx.store, "store.json")))
    assert sm["distribution"] == "private"
    assert "licence_pending" in sm["notes"]
    assert sm["distribution_override"] == {
        "declared": "public", "used": "private", "note": "licence_pending"}
    # the licence gate does not refuse a private publish or parts push ...
    assert b1.licence_gate(ctx) is True
    # ... the parts go to the private repository only ...
    fake = FakePartsHub(str(tmp_path / "parts"),
                        "chfrank/earth-tensors-private")
    monkeypatch.setattr(ph, "_list_files", fake.list_files)
    monkeypatch.setattr(ph, "_upload", fake.upload)
    monkeypatch.setattr(ph, "_download", fake.download)
    monkeypatch.setattr(ctx.layout, "hub", fake.hub)
    assert b1.push_parts(ctx) == ctx.years
    assert os.path.exists(os.path.join(
        fake.root, "partials/family1_gf/seaice_asi", str(ctx.years[0]),
        "done.json"))
    # ... and so does the store; the public repository is refused outright
    hub = str(tmp_path / "hub")
    api = _fake_hub(monkeypatch, hub, None)
    monkeypatch.setattr(ctx, "hub", lambda: (api, "chfrank/earth-tensors",
                                             "tok"))
    with pytest.raises(SystemExit, match="REFUSING to publish"):
        b1.stage_publish_grid(ctx)
    assert api.calls == []
    monkeypatch.setattr(ctx, "hub", lambda: (
        api, "chfrank/earth-tensors-private", "tok"))
    b1.stage_publish_grid(ctx)
    assert {c[1] for c in api.calls} == {"chfrank/earth-tensors-private"}
    assert os.path.exists(os.path.join(
        hub, "chfrank/earth-tensors-private",
        "tensors/family1_gf/seaice_asi/store.json"))
    assert not os.path.exists(os.path.join(hub, "chfrank/earth-tensors"))
    assert b1.stage_check_grid(ctx)["hub"]["repo"] == \
        "chfrank/earth-tensors-private"


# ============================================================= credentials ==
def test_credentials_are_refused_before_any_byte(tmp_path, monkeypatch):
    class Keyed(ghcnd.GHCNDAdapter):
        store = "keyed"
        credentials = ("EARTHDATA_USERNAME", "EARTHDATA_PASSWORD")
    monkeypatch.delenv("EARTHDATA_USERNAME", raising=False)
    monkeypatch.setenv("EARTHDATA_PASSWORD", "x")
    monkeypatch.setitem(b1.REGISTRY, "keyed", Keyed)

    def boom(*a, **k):
        raise AssertionError("a byte was requested before the preflight")
    monkeypatch.setattr(b10, "fetch_first", boom)
    monkeypatch.setattr(b10, "CountingStream", boom)
    for argv in (["--stage", "index,fetch"],
                 ["--stage", "probe", "--probe-month", "2024-01"]):
        with pytest.raises(SystemExit) as e:
            b1.main(["--store", "keyed", "--work", str(tmp_path)] + argv)
        assert "EARTHDATA_USERNAME" in str(e.value)
        assert "EARTHDATA_PASSWORD" not in str(e.value)
    # a box assembling from Hub parts never needs them
    assert not b1.needs_source(ns(parts_from_hub=True), ["fetch", "assemble"])
    assert b1.needs_source(ns(), ["index"])
    assert not b1.needs_source(ns(source_dir="/x"), ["index"])


# ==================================================================== ghcnd ==
def test_ghcnd_smoke_all_five_stages_and_the_probe(tmp_path, monkeypatch):
    res = b1.run_smoke("ghcnd", root=str(tmp_path / "smoke"), keep=True)
    probe = res["probe"]
    assert probe["rows"] == sum(1 for r in res["truth"]
                                if r["t"] >= b10.month_bounds_s(1900, 1)[0])
    assert probe["out_of_bounds"]["TMIN"] == 1
    assert probe["counts"]["qflag_blanked"]["TMAX"] >= 2
    assert probe["counts"]["lines_station_unknown"] == 1
    assert probe["counts"]["lines_bad_date"] == 1
    assert probe["counts"]["rows_all_blanked"] == 1
    assert probe["counts"]["station_days_split"] > 0
    assert probe["platforms"]["probe_platforms_without_entry"] == 0
    assert probe["estimate_rows_per_year"] is None
    assert probe["schema_version"] == 3 and probe["stored_bytes_per_row"] == 41
    p = os.path.join(str(tmp_path / "smoke"), "work", "probe", "ghcnd",
                     "1900-01.json")
    assert json.load(open(p))["rows"] == probe["rows"]

    # the five stages of the CLI, in a fresh work dir, over the same source
    ctx, truth, src = ghcnd_ctx(str(tmp_path / "cli"))
    b10.run_stages(ctx, ["index", "fetch", "assemble"],
                   stage_fn=b1.STAGE_FN, deps=b1.DEPS)
    assert b10.marked(ctx.root, "fetch") and b10.marked(ctx.root, "assemble")
    b10.check_smoke(ctx, truth)
    out = b1.stage_check(ctx)
    assert out["hub"] is None and out["schema_version"] == 3
    m = json.load(open(os.path.join(ctx.store, "store.json")))
    assert m["schema_version"] == 3 and m["time_dtype"] == "int64"
    assert m["schema"]["time_s.npy"]["dtype"] == "int64"
    assert m["family"] == "family1_tf" and m["family_version"] == "1.0.tf"
    assert m["family_code"] == "1tf" and m["distribution"] == "public"
    assert "platforms.json" in m["sha256"]
    assert m["platforms"]["entries"] == m["platforms"]["in_store"] == 4
    assert m["platforms"]["in_store_without_entry"] == 0
    pj = json.load(open(os.path.join(ctx.store, "platforms.json")))
    st = f10.Store(ctx.store)
    assert set(pj) == {str(x) for x in np.unique(st["platform"])}
    assert {v["id"] for v in pj.values()} == {s[0] for s in
                                              ghcnd.SMOKE_STATIONS}
    assert st.bin_first < 0 and st.schema_version == 3
    qc = np.asarray(st["qc"])
    want_qc = [r["qc"] for r in sorted(truth, key=lambda r: r["t"])]
    assert sorted(qc.tolist()) == sorted(want_qc)
    # the store's window is the smoke window: the day before it was dropped
    assert m["counts"]["rows_outside_window"] > 0


def test_ghcnd_streaming_and_memory_write_the_same_int64_store(tmp_path):
    hashes = {}
    for how in ("memory", "streaming"):
        ctx, truth, _ = ghcnd_ctx(str(tmp_path / how), assemble=how)
        b10.run_stages(ctx, ["index", "fetch", "assemble"],
                       stage_fn=b1.STAGE_FN, deps=b1.DEPS)
        b10.check_smoke(ctx, truth)
        hashes[how] = hash_dir(ctx.store)
    assert hashes["memory"] == hashes["streaming"]
    assert "platforms.json" in hashes["memory"]


def test_ghcnd_probe_stops_early_on_a_date_ordered_file_and_counts_the_month(
        tmp_path, monkeypatch):
    """A January-then-February year file, parsed in tiny blocks: the probe
    must stop inside February, read fewer bytes than the file, and still
    return every January station-day."""
    src = tmp_path / "src"
    base = src / "ghcnd"
    (base / "by_year").mkdir(parents=True)
    with open(base / "ghcnd-stations.txt", "w") as fh:
        for s in ghcnd.SMOKE_STATIONS:
            fh.write(ghcnd._station_line(*s) + "\n")
    lines = []
    jan = 0
    for month, ndays in ((1, 31), (2, 28), (3, 31)):
        for d in range(1, ndays + 1):
            for sid, *_ in ghcnd.SMOKE_STATIONS:
                lines.append(f"{sid},2001{month:02d}{d:02d},TMAX,100,,,S,")
                lines.append(f"{sid},2001{month:02d}{d:02d},PRCP,5,,,S,")
                jan += month == 1
    body = ("\n".join(lines) + "\n").encode()
    p = base / "by_year" / "2001.csv.gz"
    with open(p, "wb") as raw:
        with gzip.GzipFile(fileobj=raw, mode="wb", mtime=0,
                           compresslevel=0) as gz:
            gz.write(body)
    (base / "by_year" / "index.html").write_text(
        ghcnd._listing_html({2001: os.path.getsize(p)}))
    monkeypatch.setattr(ghcnd.GHCNDAdapter, "BLOCK_BYTES", 4096)
    a = ns(work=str(tmp_path / "w"), source_dir=str(src))
    out, _ = b1.stage_probe(a, ghcnd.GHCNDAdapter, "2001-01")
    assert out["rows"] == jan == 31 * len(ghcnd.SMOKE_STATIONS)
    assert out["counts"]["probe_stopped_early"] == 1
    assert 0 < out["bytes_fetched"] < os.path.getsize(p)
    assert out["rows_per_day"] == [len(ghcnd.SMOKE_STATIONS)] * 31
    assert out["distinct_platforms"] == len(ghcnd.SMOKE_STATIONS)
    assert out["nan_fraction"]["TMAX"] == 0.0
    assert out["nan_fraction"]["TMIN"] == 1.0


def test_ghcnd_refuses_a_truncated_gzip_and_a_malformed_line(tmp_path):
    buf = io.BytesIO()
    with gzip.GzipFile(fileobj=buf, mode="wb", mtime=0) as gz:
        gz.write(b"USW00000001,20010101,TMAX,100,,,S,\n" * 5000)
    raw = buf.getvalue()[:-20]
    bio = io.BytesIO(raw)
    with pytest.raises(IOError, match="truncated"):
        list(ghcnd.iter_blocks(bio.read, 1 << 20))
    with pytest.raises(ghcnd.FormatError):
        ghcnd.parse_block(b"USW00000001,2001010,TMAX,100,,,S,\n", 2001)
    with pytest.raises(ghcnd.FormatError):
        ghcnd.parse_block(b"USW00000001,20010101,TMAX,1x0,,,S,\n", 2001)
    cols, counts, dmax = ghcnd.parse_block(
        b"USW00000001,20010101,TMAX,-123,,,S,0700\n"
        b"USW00000001,20010102,TAVG,5,,,S,\n"
        b"USW00000001,20010103,PRCP,0,T,X,S,\n", 2001)
    assert cols["val"].tolist() == [-123, 0]
    assert cols["qf"].tolist() == [False, True]
    assert cols["doy"].tolist() == [0, 2]
    assert counts["lines_other_element"] == 1 and dmax == 20010103


def test_ghcnd_refuses_an_empty_listing(tmp_path):
    ctx, _, src = ghcnd_ctx(str(tmp_path))
    with open(os.path.join(src, "ghcnd", "by_year", "index.html"), "w") as fh:
        fh.write("<html><body>Index of /by_year/</body></html>")
    ctx.adapter._listing = None
    with pytest.raises(SystemExit, match="empty listing"):
        ctx.adapter.listing(ctx)


def test_ghcnd_a_year_missing_from_the_listing_is_an_absence(tmp_path):
    ctx, _, _ = ghcnd_ctx(str(tmp_path))
    rows = list(ctx.adapter.fetch_year(ctx, 1850))
    assert rows == []
    assert ctx.absent and ctx.absent[0]["unit"] == "1850"


def test_the_listing_parser_reads_the_real_page_shape():
    html = ghcnd._listing_html({1763: 3358, 2024: 168280901})
    got = dict((int(y), int(s)) for y, s in ghcnd.LISTING_ROW.findall(html))
    assert got == {1763: 3358, 2024: 168280901}


def test_counting_stream_refuses_a_short_body(monkeypatch):
    class Resp(io.BytesIO):
        headers = {"Content-Length": "100"}
    monkeypatch.setattr(b10.urllib.request, "urlopen",
                        lambda req, timeout=None: Resp(b"x" * 60))
    n0 = b10.NET_BYTES["n"]
    s = b10.CountingStream("https://example.invalid/f")
    assert s.read(50) == b"x" * 50
    with pytest.raises(IOError, match="truncated"):
        while s.read(50):
            pass
    assert b10.NET_BYTES["n"] - n0 == 60
    # a stream closed early claims nothing and raises nothing
    s2 = b10.CountingStream("https://example.invalid/f")
    s2.read(10)
    s2.close()
    assert not s2.complete


# ============================================================ parts on hub ==
class FakePartsHub:
    def __init__(self, root, repo):
        self.root, self.repo = root, repo

    def hub(self):
        return self, self.repo, "tok"

    def create_repo(self, *a, **k):
        pass

    def list_files(self, api, repo, prefix):
        out = set()
        for dp, _, names in os.walk(self.root):
            for n in names:
                rel = os.path.relpath(os.path.join(dp, n), self.root)
                if rel.startswith(prefix.rstrip("/") + "/"):
                    out.add(rel)
        return out

    def upload(self, api, repo, pairs, message):
        for rel, local in pairs:
            dst = os.path.join(self.root, rel)
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            shutil.copyfile(local, dst)

    def download(self, repo, rel, token, dest_dir, just_uploaded=False):
        src = os.path.join(self.root, rel)
        if not os.path.exists(src):
            raise FileNotFoundError(rel)
        os.makedirs(dest_dir, exist_ok=True)
        dst = os.path.join(dest_dir, os.path.basename(rel))
        shutil.copyfile(src, dst)
        return dst


def test_family1_parts_round_trip_under_their_own_prefix(tmp_path,
                                                         monkeypatch):
    ctx, truth, _ = ghcnd_ctx(str(tmp_path / "lane"))
    b10.run_stages(ctx, ["index", "fetch"], stage_fn=b1.STAGE_FN,
                   deps=b1.DEPS)
    fake = FakePartsHub(str(tmp_path / "hub"), "chfrank/earth-tensors")
    monkeypatch.setattr(ph, "_list_files", fake.list_files)
    monkeypatch.setattr(ph, "_upload", fake.upload)
    monkeypatch.setattr(ph, "_download", fake.download)
    monkeypatch.setattr(ph, "_hub", lambda: pytest.fail("family 10's hub"))
    monkeypatch.setattr(ctx.layout, "hub", fake.hub)
    ctx.a.push_parts = True
    assert b1.push_parts(ctx) == ctx.years
    for y in ctx.years:
        assert os.path.exists(os.path.join(
            fake.root, "partials/family1_tf/ghcnd", str(y), "done.json"))
    assert not os.path.exists(os.path.join(fake.root, f10.HF_PARTIALS))
    # the box: a fresh work dir; the index lists the archive, and then the
    # year files are DELETED — the fetch must not read the source at all
    box, _, bsrc = ghcnd_ctx(str(tmp_path / "box"), parts_from_hub=True)
    monkeypatch.setattr(box.layout, "hub", fake.hub)
    b10.run_stages(box, ["index"], stage_fn=b1.STAGE_FN, deps=b1.DEPS)
    for n in os.listdir(os.path.join(bsrc, "ghcnd", "by_year")):
        if n.endswith(".csv.gz"):
            os.remove(os.path.join(bsrc, "ghcnd", "by_year", n))
    b10.run_stages(box, ["fetch", "assemble"], stage_fn=b1.STAGE_FN,
                   deps=b1.DEPS)
    b10.check_smoke(box, truth)


def test_push_many_batches_years_and_writes_every_marker_last(tmp_path):
    ctx, _, _ = ghcnd_ctx(str(tmp_path / "lane"))
    b10.run_stages(ctx, ["index", "fetch"], stage_fn=b1.STAGE_FN,
                   deps=b1.DEPS)
    assert len(ctx.years) >= 2
    fake = FakePartsHub(str(tmp_path / "hub"), "chfrank/earth-tensors")
    commits = []

    def upload(api, repo, pairs, message):
        # at the moment of any PART commit, no done.json may exist yet
        if "done.json" not in message:
            assert not any(p.endswith("/done.json") for p in
                           fake.list_files(api, repo, "partials/x"))
        commits.append((message, [r for r, _ in pairs]))
        fake.upload(api, repo, pairs, message)
    seams = dict(_list_files=fake.list_files, _upload=upload,
                 _download=fake.download)
    old = {k: getattr(ph, k) for k in seams}
    try:
        for k, v in seams.items():
            setattr(ph, k, v)
        n_files = sum(len(ph.local_part_files(ctx.year_dir(y)))
                      for y in ctx.years)
        got = ph.push_many("ghcnd", ctx.years, ctx.work, partials="partials/x",
                           hub=fake.hub, max_files=2)
        assert got == sorted(ctx.years)
        parts = [c for c in commits if "done.json" not in c[0]]
        dones = [c for c in commits if "done.json" in c[0]]
        assert sum(len(c[1]) for c in parts) == n_files
        assert all(len(c[1]) <= 2 for c in commits)
        assert len(parts) == -(-n_files // 2)
        assert sum(len(c[1]) for c in dones) == len(ctx.years)
        assert commits.index(dones[0]) > commits.index(parts[-1])
        for y in ctx.years:
            done = json.load(open(os.path.join(
                fake.root, "partials/x/ghcnd", str(y), "done.json")))
            assert {e["name"] for e in done["files"]} == set(
                ph.local_part_files(ctx.year_dir(y)))
        # the whole lane pulls back as family 10's pull reads it
        present, missing = ph.pull("ghcnd", ctx.years,
                                   str(tmp_path / "box"),
                                   partials="partials/x", hub=fake.hub)
        assert present == ctx.years and missing == []
        # a re-push finds every year present and commits nothing
        commits.clear()
        assert ph.push_many("ghcnd", ctx.years, ctx.work,
                            partials="partials/x",
                            hub=fake.hub) == sorted(ctx.years)
        assert commits == []
        # a restore mismatch writes NO marker for any year of the call
        fake2 = FakePartsHub(str(tmp_path / "hub2"), "chfrank/earth-tensors")

        def bad_download(repo, rel, token, dest_dir, just_uploaded=False):
            p = fake2.download(repo, rel, token, dest_dir)
            with open(p, "ab") as fh:
                fh.write(b"x")
            return p
        ph._list_files, ph._upload, ph._download = (
            fake2.list_files, fake2.upload, bad_download)
        with pytest.raises(SystemExit, match="RESTORE MISMATCH"):
            ph.push_many("ghcnd", ctx.years, ctx.work, partials="partials/x",
                         hub=fake2.hub)
        assert not any(p.endswith("done.json") for p in fake2.list_files(
            None, None, "partials/x"))
        # an unmarked year is refused before any upload
        os.remove(b10.marker(ctx.root, f"parts/{ctx.years[-1]}"))
        fake3 = FakePartsHub(str(tmp_path / "hub3"), "chfrank/earth-tensors")
        ph._upload = fake3.upload
        with pytest.raises(SystemExit, match="did not finish"):
            ph.push_many("ghcnd", ctx.years, ctx.work, partials="partials/x",
                         hub=fake3.hub)
        assert not os.path.exists(fake3.root)
    finally:
        for k, v in old.items():
            setattr(ph, k, v)


def test_a_private_stores_parts_are_refused_a_public_repository(tmp_path):
    fake = FakePartsHub(str(tmp_path / "hub"), "chfrank/earth-tensors")
    ctx, _, _ = ghcnd_ctx(str(tmp_path / "w"))
    b10.run_stages(ctx, ["index", "fetch"], stage_fn=b1.STAGE_FN,
                   deps=b1.DEPS)
    with pytest.raises(SystemExit, match="private"):
        ph.push("ghcnd", ctx.years[0], ctx.work, partials="partials/x",
                hub=fake.hub, private=True)
    assert not os.path.exists(fake.root)


# ============================================================= the workflow ==
def test_the_family1_workflow_is_safe():
    import yaml
    raw = open(WF).read()
    d = yaml.safe_load(raw)
    trig = d.get("on", d.get(True))
    assert list(trig) == ["workflow_dispatch"]
    inputs = trig["workflow_dispatch"]["inputs"]
    assert len(inputs) < 25
    assert {"store", "stage", "start", "end", "runner", "extra_args",
            "probe_month"} <= set(inputs)
    assert inputs["runner"]["default"] == "ubuntu-latest"
    steps = d["jobs"]["build"]["steps"]
    hosted = ("(github.event.inputs.runner == 'ubuntu-latest' || "
              "github.event.inputs.runner == '')")
    source_secrets = ("EARTHDATA_USERNAME", "EARTHDATA_PASSWORD",
                      "FIRMS_MAP_KEY", "FLUXNET_USERNAME", "FLUXNET_PASSWORD",
                      "COPERNICUSMARINE_SERVICE_USERNAME",
                      "COPERNICUSMARINE_SERVICE_PASSWORD")
    seen = set()
    for st in steps:
        r = st.get("run")
        assert not (isinstance(r, str) and "secrets." in r), st.get("name")
        env = st.get("env") or {}
        for k, v in env.items():
            if "secrets." not in str(v):
                continue
            name = str(v).split("secrets.")[1].split()[0].strip("}")
            if name in ("HF_TOKEN", "HF_PRIVATE_TOKEN"):
                continue
            assert name in source_secrets, (st.get("name"), k, v)
            seen.add(name)
            gated_env = hosted in str(v)
            gated_step = str(st.get("if", "")).replace("${{", "").replace(
                "}}", "").strip() == hosted.strip("()")
            assert gated_env or gated_step, (st.get("name"), k, v)
    assert seen == set(source_secrets)
    names = [s.get("name") for s in steps]
    netrc = steps[names.index("Earthdata .netrc (hosted runner only)")]
    assert "chmod 600" in netrc["run"] and "umask 077" in netrc["run"]
    rm = steps[names.index("Remove the .netrc")]
    assert "always()" in str(rm["if"]) and 'rm -f "${HOME}/.netrc"' in rm["run"]
    assert names.index("Remove the .netrc") == len(steps) - 1
    up = steps[names.index("Upload the probe")]
    assert up["with"]["name"] == ("probe-${{ github.event.inputs.store }}-"
                                  "${{ github.event.inputs.probe_month }}")
    assert "always()" in str(up["if"])
    assert not {"pull_request", "schedule", "push", "workflow_run",
                "issue_comment"} & set(trig)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))


def test_a_publish_commits_in_batches_with_store_json_last(tmp_path,
                                                            monkeypatch):
    """icoads (52.07 GB, ten files) timed out in create_commit four times as
    ONE commit while ghcnd's 46.91 GB went through (#91 / #89). The publish
    now spends a few commits, and store.json is alone in the last one."""
    ctx = _built_private(str(tmp_path))
    api = _fake_hub(monkeypatch, str(tmp_path / "hub"), None)
    monkeypatch.setattr(ctx, "hub", lambda: (api, ctx.layout.repo_id, "tok"))
    monkeypatch.setattr(b10, "PUBLISH_BATCH", 3)
    b1.stage_publish(ctx)
    commits = [c for c in api.calls if c[0] == "commit"]
    sm = json.load(open(os.path.join(ctx.store, "store.json")))
    n_files = len(sm["sha256"]) + 1                      # + store.json
    # every file committed exactly once, in batches of at most three
    batched = [c for c in commits if "batch" in c[2]]
    assert len(batched) == -(-(n_files - 1) // 3) + 1
    assert batched[-1][2].endswith(f"batch {len(batched)}/{len(batched)}")
    assert "1 file(s)" in batched[-1][2]
    hub = os.path.join(str(tmp_path / "hub"), ctx.layout.repo_id,
                       ctx.layout.prefix(ctx.adapter.store))
    for n in list(sm["sha256"]) + ["store.json"]:
        assert os.path.exists(os.path.join(hub, n)), n


def test_a_restore_download_retries_a_dropped_connection(monkeypatch,
                                                         tmp_path):
    """oc4k's first publish (#278) uploaded 3,702 files and died 1,220 files
    into the download-back check on one `Server disconnected without
    sending a response` — a transient the Hub client does not retry. The
    restore's download now retries transient failures and still refuses a
    4xx that names our own request."""
    import huggingface_hub
    from huggingface_hub.utils import HfHubHTTPError

    class Resp:
        def __init__(self, code):
            self.status_code = code
            self.headers = {}
            self.request = None

    calls = []
    scripted = [
        ConnectionError("Server disconnected without sending a response"),
        HfHubHTTPError("503", response=Resp(503)),
        "ok",
    ]

    def fake(repo, rel, **kw):
        calls.append(rel)
        nxt = scripted[len(calls) - 1]
        if isinstance(nxt, Exception):
            raise nxt
        return str(tmp_path / "file")

    monkeypatch.setattr(huggingface_hub, "hf_hub_download", fake)
    monkeypatch.setattr(b1, "DOWNLOAD_BACKOFF_S", (0, 0, 0, 0, 0))
    assert b1._download("r", "x", "tok", str(tmp_path)) == str(tmp_path / "file")
    assert calls == ["x", "x", "x"]

    # a 404 is our own bad request: no retry
    calls.clear()
    scripted[:] = [HfHubHTTPError("404", response=Resp(404))]
    with pytest.raises(HfHubHTTPError):
        b1._download("r", "y", "tok", str(tmp_path))
    assert calls == ["y"]

    # the ladder gives up after DOWNLOAD_ATTEMPTS transient failures
    calls.clear()
    scripted[:] = [ConnectionError("again")] * b1.DOWNLOAD_ATTEMPTS
    with pytest.raises(ConnectionError):
        b1._download("r", "z", "tok", str(tmp_path))
    assert len(calls) == b1.DOWNLOAD_ATTEMPTS

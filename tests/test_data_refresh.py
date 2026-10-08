"""E-090 · the scheduled data refresh (`ml/data_refresh.py`) on a toy store.

A tiny daily tier-G adapter is built the way the real stores were — one
unnamed whole-year lane per year, parked as parts, assembled from them — into
a directory standing in for the Hub (`LocalRemote`). Then its producer
"publishes" more days and the refresh runs: plan, lane, splice, commit,
read-back. The checks are the plan's falsifiers:

  * the spliced store equals a FROM-SCRATCH build to the new end, file for
    file (shard indices, shards, the sha256 block, every structural field of
    store.json), and passes the framework's full `check_store`;
  * the partly filled last bin is rewritten PREFIX-PRESERVING;
  * a second refresh with nothing new commits nothing;
  * a provisional day replaced by its final value is allowed only when the
    published store named it provisional; a silent revision of history is
    refused;
  * a published store.json its own ledgers do not explain is refused;
  * a commit whose read-back fails is reverted;
  * --dry-run writes nothing.
"""
import datetime as dt
import json
import os
import shutil
import sys

import numpy as np
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
ML = os.path.join(os.path.dirname(HERE), "ml")
sys.path.insert(0, ML)

import build_family1_stores as b1                                # noqa: E402
import data_refresh as R                                         # noqa: E402
from family1 import sharded as sh                                # noqa: E402

KEY = "family7_2d/tinyrefresh"
NAME = "tinyrefresh"


def _grid(H, W):
    return {"H": H, "W": W, "x0": -180.0, "dx": 10.0, "y0": -90.0,
            "dy": 9.0, "row_order": "south-first", "crs": "EPSG:4326"}


class TinyDaily(sh.GridAdapter):
    """One group, 20 x 36, tile 16, one frame a day. The record end comes
    from F1_RECORD_END like the family-7.2d adapters'; days on or after
    TINY_PROV_FROM are 'provisional' and carry a value that depends on
    TINY_PROV_VERSION."""
    store = NAME
    title = "tiny daily refresh store"
    family = "72d"
    distribution = "public"
    licence = {"name": "test", "redistribution": "yes",
               "derived_works": "free"}
    channels = (("v", "unit", -1000.0, 1000.0),)
    log2_fp = 0.0
    log2_dt = -2.32
    first_year = 2012
    frames_per_bin = 5
    frame_seconds = 86400
    dtype = "float16"
    tile = 16
    zstd_level = 3
    grid = _grid(20, 36)
    sources = ("synthetic",)
    qc_policy = "none"
    verified = "synthetic"
    notes = ""
    START = dt.date(2012, 12, 27)

    def __init__(self):
        e = os.environ.get("F1_RECORD_END", "2013-01-09")
        self.record_end = dt.date.fromisoformat(e)

    def grid_latlon(self, group, x, y):
        return np.asarray(y, float), np.asarray(x, float)

    def specs(self):
        out = super().specs()
        out[NAME]["record"] = [str(self.START), str(self.record_end)]
        out[NAME]["source_segments"] = [{"from": str(self.START),
                                         "to": str(self.record_end),
                                         "falsifier": "none"}]
        return out

    spec_doc_keys = ("record", "source_segments")

    def index(self, ctx):
        return {"dataset": self.title}

    def value(self, d):
        k = (d - self.START).days
        a = np.full((20, 36), float(k), np.float32)
        a[::4] = np.nan
        pf = os.environ.get("TINY_PROV_FROM")
        if pf and d >= dt.date.fromisoformat(pf):
            a = a + 0.25 * float(os.environ.get("TINY_PROV_VERSION", "1"))
        return a

    def fetch_frames(self, ctx, wanted):
        pf = os.environ.get("TINY_PROV_FROM")
        for g, b, f in wanted:
            d = sh.frame_day(b, f, self.frame_seconds)
            ctx.count_bytes(10)
            if d < self.START:
                yield g, b, f, None, {"frame_missing": "before_record"}
            elif d > self.record_end:
                yield g, b, f, None, {"frame_missing": "after_record"}
            else:
                c = {"files_read": 1}
                if pf and d >= dt.date.fromisoformat(pf):
                    c["preliminary_days"] = [str(d)]
                yield g, b, f, self.value(d), c

    def year_summary(self, counts_list):
        out = {"days": 0}
        for c in counts_list or []:
            out["days"] += int((c.get("frames_present") or {}).get(NAME, 0))
            if c.get("preliminary_days"):
                out["preliminary_days"] = sorted(
                    set(out.get("preliminary_days", []))
                    | set(c["preliminary_days"]))
        return out


def publish_initial(tmp, end, prov_from=None, prov_version="1"):
    """Build the store as the real ones were: whole-year lanes, parts on the
    'Hub', a box assembly from them. Returns (remote, hub_dir, store_dir)."""
    os.environ["F1_RECORD_END"] = str(end)
    if prov_from:
        os.environ["TINY_PROV_FROM"] = str(prov_from)
        os.environ["TINY_PROV_VERSION"] = prov_version
    else:
        os.environ.pop("TINY_PROV_FROM", None)
    hub = os.path.join(tmp, "hub")
    os.makedirs(hub, exist_ok=True)
    years = [2012, 2013]
    lanes = {}
    for y in years:
        lanes[y] = R.run_lane(NAME, os.path.join(tmp, f"lane{y}"),
                              dt.date(y, 1, 1), dt.date(y, 12, 31),
                              TinyDaily)
        src = lanes[y].year_dir(y)
        dst = os.path.join(hub, "partials", "family7_2d", NAME, str(y))
        shutil.copytree(src, dst)
    box = R.lane_ctx(NAME, os.path.join(tmp, "box"), dt.date(2012, 12, 1),
                     end, TinyDaily)
    box.lane, box.lane_groups = "", []
    box.bins_by_first_day = True
    b1.prepare_grid_ctx(box)
    shutil.copy(os.path.join(lanes[2012].root, "plan.json"),
                os.path.join(box.root, "plan.json"))
    for y in box.years:
        shutil.copytree(lanes[y].year_dir(y), box.year_dir(y, ""))
        b1.mark(box.root, box.part_key(y, ""))
    b1.stage_assemble_grid(box)
    dest = os.path.join(hub, "tensors", "family7_2d", NAME)
    shutil.copytree(box.store, dest)
    sm = json.load(open(os.path.join(dest, "store.json")))
    with open(os.path.join(dest, "manifest.json"), "w") as fh:
        json.dump({"files": [{"name": n, "sha256": h, "bytes": None}
                             for n, h in sm["sha256"].items()]}, fh)
    return R.LocalRemote(hub), hub, dest


def plan_for(old_end, new_end, prov=()):
    return {"store": KEY, "action": "update", "record_end": str(old_end),
            "new_end": str(new_end),
            "refetch_from": str(min([old_end + dt.timedelta(days=1)]
                                    + list(prov))),
            "provisional_days": [str(d) for d in prov]}


@pytest.fixture(autouse=True)
def _env():
    keep = {k: os.environ.get(k) for k in ("F1_RECORD_END", "TINY_PROV_FROM",
                                           "TINY_PROV_VERSION")}
    yield
    for k, v in keep.items():
        if v is None:
            os.environ.pop(k, None)
        else:
            os.environ[k] = v


E1 = dt.date(2013, 1, 9)       # inside bin 2266 (2013-01-08 .. 01-12)
E2 = dt.date(2013, 1, 19)


def refresh(tmp, remote, old_end, new_end, prov=(), **kw):
    os.environ["F1_RECORD_END"] = str(new_end)
    return R.update(KEY, remote, os.path.join(tmp, "upd"), adapter_cls=TinyDaily,
                    upstream=plan_for(old_end, new_end, prov),
                    **kw)


def test_refresh_equals_a_from_scratch_build(tmp_path):
    t = str(tmp_path)
    remote, hub, dest = publish_initial(os.path.join(t, "a"), E1)
    pb = R.bin_of_day(E1)
    rel = f"{NAME}/{sh.shard_relpath(pb)}"
    old_partial = open(os.path.join(dest, rel), "rb").read()
    res = refresh(t, remote, E1, E2)
    assert res["state"] == "updated", res
    assert res["frames_new"] == (E2 - E1).days
    assert pb in res["bins_rewritten"]
    rw = [r for r in res["rewrites"] if r["bin"] == pb][0]
    assert rw["prefix_preserving"] is True
    new_partial = open(os.path.join(dest, rel), "rb").read()
    assert new_partial[:len(old_partial)] == old_partial
    assert res["verified"]["frames_reread"] == res["frames_new"]
    # the framework's own full check on the refreshed store
    sh.check_store(dest, sample=None)
    # a from-scratch build to the new end
    _, _, ref = publish_initial(os.path.join(t, "b"), E2)
    a = json.load(open(os.path.join(dest, "store.json")))
    b = json.load(open(os.path.join(ref, "store.json")))
    assert a["sha256"] == b["sha256"]
    for k in R.STRUCTURAL:
        x, y = a.get(k), b.get(k)
        if k == "counts":                       # wall time differs, by nature
            x = {kk: v for kk, v in x.items() if kk != "fetch_seconds"}
            y = {kk: v for kk, v in y.items() if kk != "fetch_seconds"}
        assert x == y, k
    assert a["source_segments"][0]["to"] == str(E2)
    assert a["refresh"]["record_end_before"] == str(E1)
    for rel in a["sha256"]:
        assert open(os.path.join(dest, rel), "rb").read() == \
            open(os.path.join(ref, rel), "rb").read(), rel
    # nothing new: a second refresh commits nothing
    n = remote.n
    res2 = refresh(t, remote, E2, E2 + dt.timedelta(days=0),
                   prov=[E2])
    assert res2["state"] == "noop", res2
    assert remote.n == n


def test_provisional_days_replaced_and_revisions_refused(tmp_path):
    t = str(tmp_path)
    pf = dt.date(2013, 1, 1)          # bins 2191 (12-27..31)? no: 2191 = 01-01
    remote, hub, dest = publish_initial(t, E1, prov_from=pf)
    prov = [pf + dt.timedelta(days=i) for i in range((E1 - pf).days + 1)]
    sm = json.load(open(os.path.join(dest, "store.json")))
    assert sm["counts_by_year"]["2013"]["preliminary_days"] or \
        sm["counts_by_year"]["2012"]["preliminary_days"]
    # the producer finalises: the values of the provisional days change
    os.environ["TINY_PROV_VERSION"] = "2"
    # without naming them provisional the change is a REVISION: refused
    with pytest.raises(SystemExit) as ei:
        refresh(os.path.join(t, "r1"), remote, E1, E2)
    assert "REVISION REFUSED" in str(ei.value)
    assert json.load(open(os.path.join(dest, "store.json"))) == sm
    # named provisional, they are replaced
    res = refresh(os.path.join(t, "r2"), remote, E1, E2, prov=prov)
    assert res["state"] == "updated", res
    assert any(r["why"].startswith("provisional") for r in res["rewrites"])
    grp = sh.ShardedGroup(os.path.join(dest, NAME))
    d = pf
    b = R.bin_of_day(d)
    fr = grp.read_frame(b, (d - sh.bin_start_date(b)).days)
    want = TinyDaily().value(d)
    np.testing.assert_array_equal(fr[..., 0], want.astype(np.float16)
                                  .astype(np.float32))
    sh.check_store(dest, sample=None)


def test_self_check_refuses_an_unexplained_store(tmp_path):
    t = str(tmp_path)
    remote, hub, dest = publish_initial(t, E1)
    p = os.path.join(dest, "store.json")
    sm = json.load(open(p))
    sm["counts"]["files_read"] += 1
    json.dump(sm, open(p, "w"))
    with pytest.raises(SystemExit) as ei:
        refresh(t, remote, E1, E2)
    assert "SELF-CHECK REFUSED" in str(ei.value)
    assert remote.n == 0


def test_failed_read_back_is_reverted(tmp_path):
    t = str(tmp_path)
    remote, hub, dest = publish_initial(t, E1)
    before = {rel: open(os.path.join(dest, rel), "rb").read()
              for rel in json.load(open(os.path.join(dest, "store.json")))
              ["sha256"]}
    before["store.json"] = open(os.path.join(dest, "store.json"), "rb").read()
    real_get = remote.get

    def bad_get(rel, revision="main"):
        b = real_get(rel, revision)
        if revision != "main" and rel.endswith("store.json") and b:
            return b + b" "
        return b
    remote.get = bad_get
    with pytest.raises(SystemExit) as ei:
        refresh(t, remote, E1, E2)
    assert "READ-BACK MISMATCH" in str(ei.value)
    assert remote.n == 2                      # the commit and its revert
    for rel, b in before.items():
        assert open(os.path.join(dest, rel), "rb").read() == b, rel
    assert not os.path.exists(os.path.join(
        dest, NAME, sh.shard_relpath(R.bin_of_day(E2))))


def test_dry_run_writes_nothing(tmp_path):
    t = str(tmp_path)
    remote, hub, dest = publish_initial(t, E1)
    res = refresh(t, remote, E1, E2, dry_run=True)
    assert res["state"] == "would-update"
    assert res["lanes"] == [{"year": 2013, "replaces": "(unnamed)",
                             "window": ["2013-01-01", "2013-12-31"]}]
    assert remote.n == 0


def test_refresh_lanes_for_named_and_unnamed_years():
    meta = {"lanes_by_year": {"2026": {"declared": False,
                                       "lanes": ["d0701-0915", "q1", "q2"]}}}
    out = R.refresh_lanes(meta, R.bin_of_day(dt.date(2026, 9, 14)),
                          dt.date(2026, 10, 6))
    assert out == [(2026, "d0701-0915", dt.date(2026, 7, 1),
                    dt.date(2026, 12, 31))]
    # a revision reaching back into the first half-year re-runs both lanes,
    # the earlier one with its own window (so its own name)
    meta2 = {"lanes_by_year": {"2026": {"declared": False,
                                        "lanes": ["d0101-0630",
                                                  "d0701-0915"]}}}
    out = R.refresh_lanes(meta2, R.bin_of_day(dt.date(2026, 6, 20)),
                          dt.date(2026, 10, 7))
    assert out == [(2026, "d0101-0630", dt.date(2026, 1, 1),
                    dt.date(2026, 6, 30)),
                   (2026, "d0701-0915", dt.date(2026, 7, 1),
                    dt.date(2026, 12, 31))]
    out = R.refresh_lanes({}, R.bin_of_day(dt.date(2026, 12, 29)),
                          dt.date(2027, 1, 3))
    assert [x[0] for x in out] == [2026, 2027]
    assert all(x[1] == "" for x in out)


def test_policy_lists_every_registry_store():
    # every store the Data tab's family-1 registries can list has a policy
    from family1.adapters import REGISTRY
    for s, cls in REGISTRY.items():
        fam = getattr(cls, "family", None)
        if fam in ("72d", "12"):
            slug = {"72d": "family7_2d", "12": "family1_2"}[fam]
            assert f"{slug}/{s}" in R.POLICY, s
    assert R.POLICY["family7_global025_pentad_l2/g025"]["kind"] == "frozen"
    assert all(R.POLICY[k]["kind"] == "grid" for k in R.SCHEDULED)


def test_sums_follow_a_refresh_only_for_settled_months(tmp_path):
    """E-088's sums after a refresh: a month that became complete is written
    and equals a from-scratch export; a month still partial is blanked; an
    untouched month keeps its planes."""
    import export_gridded_monthly as X
    t = str(tmp_path)
    remote, hub, dest = publish_initial(os.path.join(t, "a"), E1)
    old = os.path.join(t, "sums_old")
    X.export(KEY, old, base=dest, workers=1, threads=1)
    E3 = dt.date(2013, 2, 5)
    res = refresh(t, remote, E1, E3)
    assert res["state"] == "updated"
    assert [2013, 1] in res["months_touched"]
    new = os.path.join(t, "sums_new")
    keep = os.path.join(t, "keep")
    st = X.update(KEY, new, {r: os.path.join(old, f"{r}.npy")
                             for r in ("sum", "count", "m2")} |
                  {"stats": os.path.join(old, "stats.json")},
                  touched=res["months_touched"], base=dest, workers=1,
                  threads=1, keep=keep)
    assert st["update"]["written"] == [[2013, 1]]
    # February was not in the published sums (count 0): nothing to blank
    assert st["update"]["blanked"] == []
    rep = X.verify(KEY, new, keep, base=dest)
    assert rep["ok"] and rep["n_planes"] == 1
    fresh = os.path.join(t, "sums_fresh")
    X.export(KEY, fresh, base=dest, workers=1, threads=1)
    for r in ("sum", "count", "m2"):
        a = np.load(os.path.join(new, f"{r}.npy"))
        b = np.load(os.path.join(fresh, f"{r}.npy"))
        o = np.load(os.path.join(old, f"{r}.npy"))
        np.testing.assert_array_equal(a[0, :, 1], b[0, :, 1])   # 2013-01
        assert not a[1, :, 1].any()                            # 2013-02
        np.testing.assert_array_equal(a[11, :, 0], o[11, :, 0])  # 2012-12
    assert st["frames_present"][1][1] == 0
    assert st["frames_present"][1][0] == 31
    # nothing changed since: no new version
    assert X.update(KEY, os.path.join(t, "sums_again"),
                    {r: os.path.join(new, f"{r}.npy")
                     for r in ("sum", "count", "m2")} |
                    {"stats": os.path.join(new, "stats.json")},
                    base=dest, workers=1, threads=1) is None


def test_sums_blank_a_month_whose_provisional_days_change(tmp_path):
    """A complete month with provisional days, its values replaced at an
    equal count, is BLANKED (the tab compares counts and could not tell);
    once its days are final it is written again."""
    import export_gridded_monthly as X
    t = str(tmp_path)
    E3 = dt.date(2013, 2, 5)
    pf = dt.date(2013, 1, 25)
    remote, hub, dest = publish_initial(os.path.join(t, "a"), E3,
                                        prov_from=pf)
    old = os.path.join(t, "s0")
    X.export(KEY, old, base=dest, workers=1, threads=1)
    prov = [pf + dt.timedelta(days=i) for i in range((E3 - pf).days + 1)]

    def sums(res, src, out, prov_days):
        return X.update(KEY, os.path.join(t, out),
                        {r: os.path.join(t, src, f"{r}.npy")
                         for r in ("sum", "count", "m2")}
                        | {"stats": os.path.join(t, src, "stats.json")},
                        touched=res["months_touched"], base=dest, workers=1,
                        threads=1, keep=os.path.join(t, out + "_keep"),
                        provisional=prov_days)
    os.environ["TINY_PROV_VERSION"] = "2"            # values change
    res = refresh(os.path.join(t, "r1"), remote, E3, E3, prov=prov)
    assert res["state"] == "updated"
    st = sums(res, "s0", "s1", prov)
    # January (complete) and February (partial) both kept their counts while
    # their values changed: both blanked
    assert st["update"]["blanked"] == [[2013, 1], [2013, 2]]
    assert st["frames_present"][1][0] == 0
    assert not np.load(os.path.join(t, "s1", "count.npy"))[0, :, 1].any()
    # the producer finalises every day: no provisional day left
    os.environ.pop("TINY_PROV_FROM")
    res = refresh(os.path.join(t, "r2"), remote, E3, E3, prov=prov)
    assert res["state"] == "updated"
    st = sums(res, "s1", "s2", [])
    assert st["update"]["written"] == [[2013, 1]]
    assert X.verify(KEY, os.path.join(t, "s2"), os.path.join(t, "s2_keep"),
                    base=dest)["ok"]
    assert st["frames_present"][1][0] == 31


def test_sums_update_that_only_blanks_still_verifies(tmp_path):
    """data-refresh run 3: OISST's sums had two months to blank and none to
    write, and the falsifier's keep folder did not exist — verify must pass
    with no plane rather than crash."""
    import export_gridded_monthly as X
    t = str(tmp_path)
    E3 = dt.date(2013, 2, 5)
    pf = dt.date(2013, 1, 25)
    remote, hub, dest = publish_initial(os.path.join(t, "a"), E3,
                                        prov_from=pf)
    old = os.path.join(t, "s0")
    X.export(KEY, old, base=dest, workers=1, threads=1)
    prov = [pf + dt.timedelta(days=i) for i in range((E3 - pf).days + 1)]
    os.environ["TINY_PROV_VERSION"] = "3"
    res = refresh(os.path.join(t, "r1"), remote, E3, E3, prov=prov)
    out = os.path.join(t, "s1")
    keep = os.path.join(t, "never_made")
    st = X.update(KEY, out, {r: os.path.join(old, f"{r}.npy")
                             for r in ("sum", "count", "m2")}
                  | {"stats": os.path.join(old, "stats.json")},
                  touched=res["months_touched"], base=dest, workers=1,
                  threads=1, keep=keep, provisional=prov)
    assert st["update"]["written"] == [] and st["update"]["blanked"]
    rep = X.verify(KEY, out, keep, base=dest)
    assert rep["ok"] and rep["n_planes"] == 0


def test_status_lines_keep_other_stores_and_last_update():
    prev = {"stores": [
        {"store": "family7_2d/ncep100d", "state": "current",
         "last_updated_utc": None, "record_end": "2026-03-17"},
        {"store": "family7_2d/oisst025d", "state": "updated",
         "last_updated_utc": "2026-10-08T07:44:05Z",
         "record_end": "2026-10-06"}]}
    plan_rows = [{"store": "family7_2d/oisst025d", "kind": "grid",
                  "action": "noop", "checked_utc": "2026-10-09T05:24:00Z",
                  "record_end": "2026-10-06", "upstream_newest": "2026-10-06",
                  "gap_days": 0, "reason": "current"}]
    st = R.status_lines([], plan_rows, prev)
    by = {s["store"]: s for s in st["stores"]}
    assert by["family7_2d/oisst025d"]["state"] == "current"
    assert by["family7_2d/oisst025d"]["last_updated_utc"] == \
        "2026-10-08T07:44:05Z"
    assert by["family7_2d/ncep100d"]["record_end"] == "2026-03-17"


def test_verify_tiles_samples_a_fine_frame(tmp_path):
    t = str(tmp_path)
    remote, hub, dest = publish_initial(t, E1)
    grp = sh.ShardedGroup(os.path.join(dest, NAME))
    b = R.bin_of_day(E1)
    allt = R.verify_tiles(grp, b, 0)
    assert len(allt) == grp.spec["n_tiles_y"] * grp.spec["n_tiles_x"]
    few = R.verify_tiles(grp, b, 0, k=3)
    assert len(few) <= 3 and (0, 0) in few

    calls = {"n": 0}

    def flaky():
        calls["n"] += 1
        if calls["n"] < 3:
            raise IOError("x bytes 1-2: HTTPError: HTTP Error 429: Too Many")
        return "ok"
    import data_refresh
    real = data_refresh.time.sleep
    data_refresh.time.sleep = lambda s: None
    try:
        assert R.hub_read(flaky) == "ok" and calls["n"] == 3
        with pytest.raises(IOError):
            R.hub_read(lambda: (_ for _ in ()).throw(IOError("bad sha")))
    finally:
        data_refresh.time.sleep = real

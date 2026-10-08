#!/usr/bin/env python3
"""E-090 · Keep every store the Data tab serves current with its upstream.

PLAIN ENGLISH. The Data tab (`src/f1data.js`) reads the public stores on the
Hugging Face dataset `chfrank/earth-tensors` through four registries. Every
store was built once, to the last day its producer served on the day of the
build. This module is what a scheduled GitHub-hosted job runs every day so
that each store follows its producer: it asks the producer how far its record
reaches today, does NOTHING when that is where the store already ends, and
otherwise re-fetches only the store's last LANE (for the daily stores: the
current calendar year), splices those shards into the published store,
commits, reads every committed file back, re-reads the new frames through the
reader, and only then lets the registry announce the longer record. The plan,
the inventory and every decision: `ml/plans/E090_data_refresh.md`.

WHAT IT REUSES — nothing here defines a store a second time:

  the fetch      `ml/build_family1_stores.py`'s own index and fetch stages,
                 in process, for the lane being refreshed (the adapters'
                 derivations, falsifiers and refusals unchanged);
  store.json     `build_family1_stores.grid_store_meta`, the function the
                 assembler builds it with, fed the published store's kept
                 shard-index rows and lane ledgers plus the refetched lane's;
  the layout     `ml/family1/sharded.py` (writer, reader, `check_sharded`);
  the registry   `ml/build_family1_registry.py` + `ml/registry_guard.py`;
  the sums       `ml/export_gridded_monthly.py update` (E-088).

THE SPLICE'S SELF-CHECK. Before anything changes, the published store.json is
REBUILT from the published shard indices and the published lane ledgers
(`partials/<family>/<store>/<year>/[<lane>/]counts.json`) with the same
function, and every structural field (groups, per-year frames, counts,
missing frames, lanes, date range) must equal what is published. A store
whose published ledgers no longer explain it is refused — the splice could
not be trusted to keep what it does not touch.

THE COMMIT POINT, AND READERS IN MID-READ. The data commit carries the
changed shards, their indices, the group's shard_index.npy, tile_grid.json,
store.json and manifest.json in ONE Hub commit (atomic at the repository).
Because the reader computes a shard's URL from its bin number, a partly
filled last bin is REWRITTEN under the same name; the splice refuses such a
rewrite unless it is PREFIX-PRESERVING — the old shard's bytes are the first
bytes of the new one and the old frames' index entries are unchanged — so a
reader that holds the old index and fetches from the new shard (or the
reverse) gets the old frames exactly, or a short read that FAILS loudly,
never a silent mix. The one legitimate exception is a bin holding a
producer's PROVISIONAL days (OISST's preliminary fortnight), whose bytes are
supposed to change; the status line names it. The registry is published only
after the data commit has been read back, and it is what makes the new days
visible to the Data tab (`record_span`).

  python3 ml/data_refresh.py plan [--stores all|a,b] [--json out.json]
  python3 ml/data_refresh.py update --store family7_2d/oisst025d --work DIR \
        [--dry-run] [--result result.json] [--force-from YYYY-MM-DD]
  python3 ml/data_refresh.py registry --results DIR [--dry-run]
  python3 ml/data_refresh.py status --results DIR [--plan plan.json] [--upload]

Credentials: HF_TOKEN (writes), COPERNICUSMARINE_SERVICE_USERNAME/PASSWORD
(glorys025d's tail months), EARTHDATA_USERNAME/PASSWORD (the Earthdata
stores) — environment only, never argv; the workflow scopes them per job.
"""
import argparse
import datetime as dt
import glob
import hashlib
import io
import json
import os
import re
import shutil
import subprocess
import sys
import time
import urllib.error
import urllib.parse
import urllib.request

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
if HERE not in sys.path:
    sys.path.insert(0, HERE)

PLAN_DOC = "ml/plans/E090_data_refresh.md"
REPO = "chfrank/earth-tensors"
HUB = f"https://huggingface.co/datasets/{REPO}/resolve"
STATUS_PATH = "tensors/refresh/status.json"
UA = {"User-Agent": "earth-data-refresh (github.com/blauewelt/earth)"}
# how many new frames are re-read through the HTTP reader after a commit
VERIFY_FRAMES = 40

REGISTRY_URL = {
    "1gf": "tensors/family1_gf/family1gf.json",
    "12": "tensors/family1_2/family12.json",
    "72d": "tensors/family7_2d/family72d.json",
    "10": "tensors/family10_2/family10.json",
}


# =============================================================== helpers ====
def utcnow():
    return dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def today():
    return dt.datetime.now(dt.timezone.utc).date()


def sha256_bytes(b):
    return hashlib.sha256(b).hexdigest()


def sha256_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for blk in iter(lambda: fh.read(1 << 22), b""):
            h.update(blk)
    return h.hexdigest()


def write_json(path, obj):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(obj, fh, indent=1, sort_keys=True, default=str)
        fh.write("\n")
    os.replace(tmp, path)


def http_get(url, timeout=120, attempts=4, headers=None):
    """bytes, or None on a 404. Retries 429 / 5xx / connection errors."""
    last = None
    for i in range(attempts):
        try:
            req = urllib.request.Request(url, headers={**UA, **(headers or {})})
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return r.read()
        except urllib.error.HTTPError as e:
            if e.code == 404:
                return None
            last = e
            if e.code not in (429, 500, 502, 503, 504):
                raise
        except (urllib.error.URLError, TimeoutError, ConnectionError) as e:
            last = e
        time.sleep(min(60, 5 * 2 ** i))
    raise IOError(f"{url}: {last}")


def http_json(url, **kw):
    b = http_get(url, **kw)
    return None if b is None else json.loads(b)


def iso(d):
    return d.isoformat() if d is not None else None


def parse_day(s):
    return dt.date.fromisoformat(str(s)[:10]) if s else None


# ================================================================ remotes ===
class HubRemote:
    """The dataset repository: reads over resolve/<revision>, writes by ONE
    commit (`build_family7.hub_commit`'s 429 ladder)."""

    def __init__(self, repo=REPO, token=None):
        self.repo = repo
        self.token = token if token is not None else \
            os.environ.get("HF_TOKEN", "").strip()
        self._api = None

    def url(self, rel, revision="main"):
        return (f"https://huggingface.co/datasets/{self.repo}/resolve/"
                f"{revision}/{urllib.parse.quote(rel)}")

    def base(self, revision="main"):
        return f"https://huggingface.co/datasets/{self.repo}/resolve/{revision}"

    def get(self, rel, revision="main"):
        h = {"Authorization": f"Bearer {self.token}"} if self.token else {}
        return http_get(self.url(rel, revision), timeout=300, attempts=6,
                        headers=h)

    def api(self):
        if self._api is None:
            if not self.token:
                raise SystemExit("HF_TOKEN is not set — a refresh that "
                                 "commits needs it (an Actions secret)")
            from huggingface_hub import HfApi
            self._api = HfApi(token=self.token)
        return self._api

    def commit(self, adds, deletes, message):
        import build_family7 as f7
        ops = f7.hub_add_ops(adds) + f7.hub_delete_ops(deletes)
        info = f7.hub_commit(self.api(), self.repo, ops, message)
        oid = getattr(info, "oid", None) or str(info).rsplit("/", 1)[-1]
        return oid

    def list(self, prefix):
        api = self.api() if self.token else None
        if api is None:
            from huggingface_hub import HfApi
            api = HfApi()
        out = []
        for e in api.list_repo_tree(self.repo, path_in_repo=prefix,
                                    repo_type="dataset", recursive=True):
            if getattr(e, "size", None) is not None and \
                    not hasattr(e, "tree_id"):
                out.append(e.path)
        return out


class LocalRemote:
    """A directory standing in for the repository (tests, dry runs). Every
    commit is applied at once and numbered; `get` at a revision reads the
    snapshot that commit left."""

    def __init__(self, root):
        self.root = os.path.abspath(root)
        self.n = 0
        self.snaps = {}

    def base(self, revision="main"):
        if revision != "main" and revision in self.snaps:
            return self.snaps[revision]
        return self.root

    def get(self, rel, revision="main"):
        p = os.path.join(self.base(revision), rel)
        if not os.path.exists(p):
            return None
        with open(p, "rb") as fh:
            return fh.read()

    def commit(self, adds, deletes, message):
        for rel, local in adds:
            dst = os.path.join(self.root, rel)
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            shutil.copyfile(local, dst + ".tmp")
            os.replace(dst + ".tmp", dst)
        for rel in deletes:
            p = os.path.join(self.root, rel)
            if os.path.exists(p):
                os.remove(p)
        self.n += 1
        oid = f"local{self.n}"
        snap = os.path.join(os.path.dirname(self.root), ".snaps",
                            os.path.basename(self.root) + f"_{oid}")
        shutil.rmtree(snap, ignore_errors=True)
        shutil.copytree(self.root, snap)
        self.snaps[oid] = snap
        return oid

    def list(self, prefix):
        d = os.path.join(self.root, prefix)
        out = []
        for dp, _, fs in os.walk(d):
            for f in fs:
                out.append(os.path.relpath(os.path.join(dp, f), self.root)
                           .replace(os.sep, "/"))
        return sorted(out)


# ========================================================== the upstream ====
# Each probe answers {newest: date | None, how: str} — the last day the
# producer serves TODAY, measured from the producer's own index, never typed
# in. A probe that cannot reach its producer raises; the plan records the
# error and the store is skipped for the day (never "no new data").
def _dds_time_n(url):
    import build_family7 as f7
    b = http_get(url)
    return None if b is None else f7.parse_dds_time_n(b)


def up_oisst():
    psl = "https://psl.noaa.gov/thredds/dodsC/Datasets/noaa.oisst.v2.highres"
    t = today()
    for y in (t.year, t.year - 1):
        ns = [_dds_time_n(f"{psl}/{k}.day.mean.{y}.nc.dds")
              for k in ("sst", "icec")]
        if all(n for n in ns):
            n = min(ns)
            return {"newest": dt.date(y, 1, 1) + dt.timedelta(days=n - 1),
                    "how": f"PSL THREDDS: sst/icec.day.mean.{y}.nc hold "
                           f"{ns[0]}/{ns[1]} days"}
    raise IOError("PSL answered for neither this year's nor last year's "
                  "OISST files")


def up_ncep():
    import build_family7 as f7
    psl = ("https://psl.noaa.gov/thredds/dodsC/Datasets/ncep.reanalysis/"
           "surface_gauss")
    t = today()
    for y in (t.year, t.year - 1):
        ns = {}
        for stem in sorted(set(f7.NCEP_FILES.values())):
            ns[stem] = _dds_time_n(f"{psl}/{stem}.{y}.nc.dds")
        if all(ns.values()):
            n = min(ns.values())
            days = n // 4                    # four analyses a day
            return {"newest": dt.date(y, 1, 1) + dt.timedelta(days=days - 1),
                    "how": f"PSL THREDDS: the 13 surface_gauss {y} files hold "
                           f"{min(ns.values())}..{max(ns.values())} six-hourly "
                           f"steps (a day counts when all four are there)"}
    raise IOError("PSL answered for none of the NCEP stems")


def up_glorys():
    url = ("https://stac.marine.copernicus.eu/metadata/"
           "GLOBAL_MULTIYEAR_PHY_001_030/"
           "cmems_mod_glo_phy_my_0.083deg_P1D-m_202311/dataset.stac.json")
    d = http_json(url)
    end = (d or {}).get("properties", {}).get("end_datetime")
    if not end:
        raise IOError(f"{url}: no properties.end_datetime")
    return {"newest": parse_day(end),
            "how": f"Copernicus Marine STAC end_datetime {end} "
                   f"(cmems_mod_glo_phy_my_0.083deg_P1D-m_202311)"}


def up_occci():
    import build_family7 as f7
    from family1.adapters import occci025d as oc
    n = f7.parse_dds_time_n(http_get(oc.PML_DDS))
    axis = f7.parse_opendap_ascii_ints(http_get(oc.PML_ASCII.format(
        last=int(n) - 1)))
    day = f7.OC_TIME_EPOCH + dt.timedelta(days=int(axis[-1]))
    return {"newest": day, "how": f"PML THREDDS CCI_ALL-v6.0-DAILY: {n} days, "
                                  f"the last {day}"}


def up_era5():
    from family1.adapters import _era5
    url = f"{_era5.GCS}/{_era5.ZARR_B}/.zattrs"
    a = http_json(url) or {}
    stop = a.get("valid_time_stop")
    if not stop:
        raise IOError(f"{url}: no valid_time_stop")
    return {"newest": parse_day(stop),
            "how": f"ARCO-ERA5 {_era5.ZARR_B} valid_time_stop {stop} (final "
                   f"ERA5; ERA5T to {a.get('valid_time_stop_era5t')} is not "
                   f"admitted by the adapter)",
            "era5t": a.get("valid_time_stop_era5t")}


CMR = "https://cmr.earthdata.nasa.gov/search/granules.json"


def _cmr_newest(params):
    q = urllib.parse.urlencode({**params, "sort_key": "-start_date",
                                "page_size": 1})
    d = http_json(f"{CMR}?{q}") or {}
    e = (d.get("feed") or {}).get("entry") or []
    if not e:
        return None
    return e[0].get("time_end") or e[0].get("time_start")


def _cmr_day(params, full_day=True):
    s = _cmr_newest(params)
    if not s:
        raise IOError(f"CMR: no granule for {params}")
    t = dt.datetime.fromisoformat(s.replace("Z", "+00:00"))
    d = t.date()
    if full_day and (t.hour, t.minute) < (23, 0):
        d -= dt.timedelta(days=1)        # the newest day is not complete yet
    return d, s


def up_pace4k():
    out = []
    for sn in ("PACE_OCI_L3M_BGC", "PACE_OCI_L3M_AOP", "PACE_OCI_L4M_MOANA"):
        s = _cmr_newest({"short_name": sn, "provider": "OB_CLOUD"})
        if not s:
            raise IOError(f"CMR: no {sn} granule")
        out.append((parse_day(s), sn))
    d = min(x[0] for x in out)
    return {"newest": d, "how": "CMR newest daily granule of "
            + ", ".join(f"{sn} {x}" for x, sn in out)
            + " (the store needs all three)"}


def up_acspo():
    d, s = _cmr_day({"collection_concept_id": "C2805339147-POCLOUD"})
    return {"newest": d, "how": f"CMR newest L3S_LEO_DY granule ends {s}"}


def up_irtb():
    d, s = _cmr_day({"collection_concept_id": "C1432254058-GES_DISC"})
    return {"newest": d, "how": f"CMR newest GPM_MERGIR granule ends {s} "
                                f"(a day counts once its 23 UTC hour is in)"}


# ================================================================ policy ====
# EVERY STORE THE DATA TAB'S REGISTRIES LIST, and what the refresh does with
# it. `kind`:
#   grid     sharded tier G — refreshed here (lane re-fetch + splice)
#   frozen   the paper's / the training tensor's — never touched
#   static   the producer's record is closed at the adapter's source/version
#   manual   a living upstream, but the store's daily increment cannot run on
#            a hosted runner (or the point-store rewrite is not automated
#            yet) — a workflow_dispatch job, documented in the plan
CMEMS = ("COPERNICUSMARINE_SERVICE_USERNAME",
         "COPERNICUSMARINE_SERVICE_PASSWORD")
EARTHDATA = ("EARTHDATA_USERNAME", "EARTHDATA_PASSWORD")

POLICY = {
    "family7_2d/oisst025d": dict(kind="grid", fam="72d", upstream=up_oisst,
                                 record_env=True, upstream_first=True,
                                 provisional="oisst", sums=True),
    "family7_2d/ncep100d": dict(kind="grid", fam="72d", upstream=up_ncep,
                                record_env=True, upstream_first=True,
                                sums=True),
    "family7_2d/glorys025d": dict(kind="grid", fam="72d", upstream=up_glorys,
                                  record_env=True, creds=CMEMS, sums=True),
    "family7_2d/occci025d": dict(kind="grid", fam="72d", upstream=up_occci,
                                 record_env=True, sums=True),
    "family1_2/era5_t": dict(kind="grid", fam="12", upstream=up_era5,
                             sums=True),
    "family1_2/era5_q": dict(kind="grid", fam="12", upstream=up_era5,
                             sums=True),
    "family1_2/era5_u": dict(kind="grid", fam="12", upstream=up_era5,
                             sums=True),
    "family1_2/era5_v": dict(kind="grid", fam="12", upstream=up_era5,
                             sums=True),
    "family1_gf/pace4k": dict(kind="grid", fam="1gf", upstream=up_pace4k,
                              auto=False, min_gap_days=7, sums="e089",
                              why="dispatch-only until a real refresh is "
                                  "green: a whole-year lane of 4 km OPeNDAP "
                                  "reads, ~1 h — plan §4"),
    "family1_gf/sst_acspo02": dict(kind="grid", fam="1gf", auto=False,
                                   upstream=up_acspo, creds=EARTHDATA,
                                   min_gap_days=7, sums="e089",
                                   why="dispatch-only until a real refresh "
                                       "is green: the refreshed lane is the "
                                       "store's last named half-year lane "
                                       "(58 s a frame) — plan §4"),
    "family1_gf/irtb": dict(kind="grid", fam="1gf", auto=False,
                            upstream=up_irtb, creds=EARTHDATA,
                            min_gap_days=7, sums="e089",
                            why="dispatch-only until a real refresh is "
                                "green: quarter lanes of 3-hourly 4 km — "
                                "plan §4"),
    "family1_gf/oc4k": dict(kind="static", fam="1gf", sums="e089",
                            why="CEDA's OC-CCI v6.0 4 km daily tree ends "
                                "2022-12-31 (listed 2026-10-08); a later "
                                "year needs another source, not a refresh"),
}

# the point stores and the frozen groups — listed so `plan --stores all`
# reports every store the tab can open (the plan's inventory table)
for _k, _why in {
        "family1_gf/bgcargo": "tier-P rewrite (plan §5)",
        "family1_gf/glodap": "tier-P rewrite (plan §5)",
        "family1_gf/icoads": "tier-P rewrite (plan §5)",
        "family1_gf/oceansites": "tier-P rewrite (plan §5)",
        "family1_gf/swot": "tier-P rewrite (plan §5)",
        "family1_gf/wod": "tier-P rewrite (plan §5)",
        "family1_gf/xco2": "tier-P rewrite (plan §5)",
        "family10_1/gdp": "tier-P rewrite (plan §5)",
        "family10_1/gtmba": "tier-P rewrite (plan §5)",
        "family10_1/slatrack": "tier-P rewrite (plan §5)",
        "family8_argo_l0": "tier-P rewrite (plan §5)"}.items():
    POLICY[_k] = dict(kind="manual", why=_why)
for _k, _why in {
        "family1_gf/swh": "ESA CCI Sea State v4 L3 ends 2023-12-31 (STAC)",
        "family10_1/socat": "SOCAT is one release a year (v2026 is in)",
        "family10_2/fishing": "Global Fishing Watch v3 (Zenodo, 2025-03) "
                              "ends 2024"}.items():
    POLICY[_k] = dict(kind="static", why=_why)
for _k in ("family7_global025_pentad_l2/g025",
           "family7_global025_pentad_l2/g100",
           "family7_global025_pentad_l2/oc025",
           "family7_global025_pentad_l2/rg100"):
    POLICY[_k] = dict(kind="frozen", why="the five-day training tensor "
                      "(family 7.2) — the paper must keep reproducing")

SCHEDULED = [k for k, v in POLICY.items()
             if v["kind"] == "grid" and v.get("auto", True)]


def store_parts(key):
    """'family7_2d/oisst025d' -> (slug, name)."""
    slug, name = key.split("/", 1)
    return slug, name


# ============================================================== registry ====
_REG = {}


def registry_entry(key, remote):
    """The live registry entry of a store (family-1 schema)."""
    pol = POLICY[key]
    rel = REGISTRY_URL[pol["fam"]]
    if rel not in _REG:
        b = remote.get(rel)
        _REG[rel] = json.loads(b) if b else {}
    reg = _REG[rel]
    name = store_parts(key)[1]
    groups = reg.get("groups") or []
    for g in groups if isinstance(groups, list) else groups.values():
        if g.get("name") == name:
            return g
    raise SystemExit(f"{key}: not in {rel}")


def our_end(entry):
    rs = entry.get("record_span") or []
    return parse_day(rs[1]) if len(rs) == 2 else None


def provisional_days(entry):
    """Days the published store calls provisional (OISST: preliminary)."""
    c = entry.get("counts") or {}
    out = set(c.get("preliminary_days") or [])
    out |= set(c.get("preliminary_status_unknown_days") or [])
    for v in (entry.get("store_counts_by_year") or {}).values():
        out |= set(v.get("preliminary_days") or [])
        out |= set(v.get("preliminary_status_unknown_days") or [])
    return sorted(parse_day(d) for d in out)


def provisional_now_final(key, days):
    """Which of `days` the producer has since FINALISED."""
    if POLICY[key].get("provisional") != "oisst" or not days:
        return []
    from family1.adapters import oisst025d
    ad = oisst025d.OISST025DAdapter()
    out = []
    for d in days:
        p = ad.preliminary(d)
        if p is False:
            out.append(d)
    return out


# ================================================================== plan ====
def bin_of_day(d):
    from family1 import sharded as sh
    return (d - sh.EPOCH).days // 5


def plan_store(key, remote, force_from=None):
    """What the refresh would do for one store today — no writes."""
    pol = POLICY[key]
    out = {"store": key, "kind": pol["kind"], "checked_utc": utcnow(),
           "why": pol.get("why")}
    if pol["kind"] not in ("grid", "manual") or "upstream" not in pol:
        out["action"] = "none"
        return out
    try:
        ent = registry_entry(key, remote)
        out["record_end"] = iso(our_end(ent))
    except SystemExit as e:
        out.update(action="error", error=str(e))
        return out
    try:
        up = pol["upstream"]()
    except Exception as e:                                    # noqa: BLE001
        out.update(action="error",
                   error=f"upstream probe failed: {type(e).__name__}: "
                         f"{str(e)[:300]}")
        return out
    out["upstream_newest"] = iso(up["newest"])
    out["upstream_how"] = up["how"]
    ours = our_end(ent)
    gap = (up["newest"] - ours).days if (up["newest"] and ours) else None
    out["gap_days"] = gap
    prov = provisional_days(ent)
    out["provisional_days"] = [iso(d) for d in prov]
    final = []
    if prov:
        try:
            final = provisional_now_final(key, prov)
        except Exception as e:                                # noqa: BLE001
            out["provisional_error"] = f"{type(e).__name__}: {e}"[:300]
    out["provisional_now_final"] = [iso(d) for d in final]
    cands = []
    if gap is not None and gap > 0:
        cands.append(ours + dt.timedelta(days=1))
    if final:
        cands.append(min(final))
    if force_from:
        cands.append(parse_day(force_from))
    start = min(cands) if cands else None
    mg = int(pol.get("min_gap_days") or 0)
    if start is not None and not final and not force_from and mg and \
            (gap or 0) < mg:
        out["action"] = "noop"
        out["reason"] = (f"{gap} new day(s) upstream; this store is "
                         f"refreshed in batches of >= {mg} days (its lane "
                         f"re-fetch is long)")
        return out
    if start is None:
        out["action"] = "noop"
        out["reason"] = (f"the producer's newest day {iso(up['newest'])} is "
                         f"the store's record end"
                         + (" and no provisional day became final"
                            if prov else ""))
        return out
    from family1 import sharded as sh
    b0 = bin_of_day(start)
    newest = max(up["newest"], ours) if ours else up["newest"]
    years = list(range(sh.bin_year(b0), sh.bin_year(bin_of_day(newest)) + 1))
    out.update(action="update" if pol["kind"] == "grid" else "manual",
               refetch_from=iso(start), new_end=iso(newest), years=years,
               new_days=max(0, gap or 0))
    return out


def plan(stores, remote, force_from=None):
    return [plan_store(k, remote, force_from=force_from) for k in stores]


# ============================================================ the lanes =====
def lane_args(name, work, start, end):
    import build_family1_stores as b1
    return b1.build_parser().parse_args([
        "--store", name, "--stage", "index,fetch", "--start", str(start),
        "--end", str(end), "--work", work])


def lane_ctx(name, work, start, end, adapter_cls=None):
    """A context exactly as `build_family1_stores.main` makes one."""
    import build_family1_stores as b1
    import build_family10_stores as f10b
    from family1.adapters import REGISTRY
    a = lane_args(name, work, start, end) if adapter_cls is None else \
        _toy_args(name, work, start, end)
    cls = adapter_cls or REGISTRY[name]
    ad = b1.apply_distribution(cls(), getattr(a, "distribution", ""))
    ctx = f10b.Ctx(a, adapter=ad, layout=b1.layout_for(ad))
    b1.apply_lane(ctx)
    b1.prepare_grid_ctx(ctx)
    return ctx


def _toy_args(name, work, start, end):
    import build_family10_stores as f10b
    return argparse.Namespace(
        store=name, work=work, source_dir=os.path.join(work, "_src"),
        start=str(start), end=str(end), stage="index,fetch", force=False,
        attempts=1, qc_keep=2, check_chunk_rows=f10b.CHECK_CHUNK_ROWS,
        assemble="auto", parts_from_hub=False, push_parts=False,
        allow_missing_years=False, allow_unconfirmed_licence=False,
        probe_month="", smoke=False, distribution="", lanes="")


def run_lane(name, work, start, end, adapter_cls=None):
    """The builder's index + fetch stages for one lane, in process."""
    import build_family1_stores as b1
    import build_family10_stores as f10b
    ctx = lane_ctx(name, work, start, end, adapter_cls)
    if adapter_cls is None and b1.needs_source(ctx.a, ["index", "fetch"]):
        b1.credentials_preflight(ctx.adapter)
    print(f"  lane {name} {start}..{end} "
          f"({ctx.lane or 'unnamed'}), years {ctx.years}", flush=True)
    f10b.run_stages(ctx, ["index", "fetch"], stage_fn=b1.GRID_STAGE_FN,
                    deps=b1.DEPS)
    return ctx


def refresh_lanes(meta, b0, newest):
    """[(year, lane_name_before, start, end)] the refresh re-fetches: from the
    lane holding bin b0 to the year holding `newest`. A year that is one
    unnamed lane is re-fetched whole; a NAMED lane is re-fetched from its own
    start to 31 December (`dMMDD-1231` — a name that is then stable across
    refreshes); a year not in the store yet is a whole-year lane."""
    from family1 import sharded as sh
    lanes_by_year = meta.get("lanes_by_year") or {}
    y0, y1 = sh.bin_year(b0), newest.year
    # a bin straddling New Year belongs to the year it starts in
    y1 = max(y1, sh.bin_year(bin_of_day(newest)))
    out = []
    d0 = sh.bin_start_date(b0)
    for y in range(y0, y1 + 1):
        rec = lanes_by_year.get(str(y))
        names = (rec or {}).get("lanes") or [""]
        if names == [""]:
            out.append((y, "", dt.date(y, 1, 1), dt.date(y, 12, 31)))
            continue
        # the named lane holding d0 (or the last one of a later year)
        wins = []
        for n in names:
            m = re.fullmatch(r"d(\d{4})-(\d{4})", n) if n else None
            q = re.fullmatch(r"q([1-4])", n) if n else None
            mm = re.fullmatch(r"m(\d{2})", n) if n else None
            if m:
                lo = dt.date(y, int(m.group(1)[:2]), int(m.group(1)[2:]))
            elif q:
                lo = dt.date(y, 3 * int(q.group(1)) - 2, 1)
            elif mm:
                lo = dt.date(y, int(mm.group(1)), 1)
            else:
                lo = dt.date(y, 1, 1)
            wins.append((lo, n))
        wins.sort()
        pick = [w for w in wins if w[0] <= max(d0, dt.date(y, 1, 1))] or \
            wins[:1]
        lo, n = pick[-1]
        later = [w for w in wins if w[0] > lo]
        if later:
            raise SystemExit(f"{y}: lanes {[w[1] for w in later]} start after "
                             f"the lane {n!r} that holds the refresh's first "
                             f"bin — a refresh re-fetches the LAST lane of a "
                             f"year only")
        out.append((y, n, lo, dt.date(y, 12, 31)))
    return out


# ============================================================ the splice ====
STRUCTURAL = ("groups", "per_year", "frames_missing_by_reason",
              "missing_frames", "counts", "counts_by_year", "lanes_by_year",
              "date_range", "bins_requested", "channels", "C", "dtype",
              "frames_per_bin", "frame_seconds", "degraded", "family",
              "family_version", "tier", "layout", "format", "store")
DESCRIPTIVE = ("title", "sources", "verified", "notes", "qc_policy", "plan",
               "licence", "normalisation", "resume_granularity", "footprint",
               "distribution", "family_code", "epoch", "pentad_days",
               "counts_by_year_note", "lanes_note", "builder",
               "distribution_override")
IGNORED = ("built_at", "builder_git_sha", "sha256", "source_dir", "refresh",
           "source_segments")


class SpliceError(SystemExit):
    pass


def _diff(a, b, path="", out=None, cap=12):
    out = [] if out is None else out
    if len(out) >= cap:
        return out
    if isinstance(a, dict) and isinstance(b, dict):
        for k in sorted(set(a) | set(b)):
            _diff(a.get(k, "<absent>"), b.get(k, "<absent>"), f"{path}.{k}",
                  out, cap)
    elif isinstance(a, list) and isinstance(b, list) and len(a) == len(b):
        for i, (x, y) in enumerate(zip(a, b)):
            _diff(x, y, f"{path}[{i}]", out, cap)
    elif a != b:
        if isinstance(a, float) and isinstance(b, float) and \
                abs(a - b) <= 1e-12 * max(1.0, abs(a)):
            return out
        out.append(f"{path}: {str(a)[:120]} != {str(b)[:120]}")
    return out


def _norm(v):
    return json.loads(json.dumps(v, default=str))


def _ledger_rel(partials, store, year, lane):
    p = f"{partials}/{store}/{year}"
    return f"{p}/{lane}/counts.json" if lane else f"{p}/counts.json"


def _merge_ledgers(f10b, ledgers):
    """(counts_all, year_counts) in the assembler's order: years ascending,
    the unnamed lane first then named lanes by name."""
    counts_all, year_counts = {}, {}
    for (y, lane) in sorted(ledgers, key=lambda k: (int(k[0]), k[1] != "",
                                                     k[1])):
        c = ledgers[(y, lane)].get("counts") or {}
        f10b._merge_counts(counts_all, json.loads(json.dumps(c)))
        year_counts.setdefault(str(y), []).append(json.loads(json.dumps(c)))
    return counts_all, year_counts


def _bin_owned(lane_window, b):
    from family1 import sharded as sh
    if not lane_window:
        return True
    lo, hi = (parse_day(x) for x in lane_window)
    return lo <= sh.bin_start_date(b) <= hi


def splice(key, remote, lanes, work, *, new_end=None, adapter_cls=None,
           allow_revisions=False, provisional=(), plan_root=None,
           check_only=False):
    """Build the refreshed store's changed files and store.json in
    `work/stage/` from the published store and the re-fetched lanes.

    `lanes` is [(year, old_lane_name, lane_ctx)] — each lane already fetched
    (`run_lane`). Returns a report dict with `adds` [(rel, local)], `deletes`
    [rel], and the numbers; raises SpliceError on any refusal.
    """
    import build_family1_stores as b1
    import build_family10_stores as f10b
    from family1 import sharded as sh
    slug, name = store_parts(key)
    prefix = f"tensors/{key}"
    partials = f"partials/{slug}"
    stage = os.path.join(work, "stage")
    shutil.rmtree(stage, ignore_errors=True)
    os.makedirs(stage)
    rep = {"store": key}

    def get(rel, what):
        b = remote.get(rel)
        if b is None:
            raise SpliceError(f"{key}: {rel} is not on the Hub ({what})")
        return b

    old_meta = json.loads(get(f"{prefix}/store.json", "store.json"))
    block = dict(old_meta.get("sha256") or {})
    groups = sorted(old_meta.get("groups") or {})
    old_arr, old_spec = {}, {}
    for g in groups:
        raw = get(f"{prefix}/{g}/shard_index.npy", "shard index")
        if sha256_bytes(raw) != block.get(f"{g}/shard_index.npy"):
            raise SpliceError(f"{key}: {g}/shard_index.npy on the Hub is not "
                              f"the one store.json names")
        old_arr[g] = sh.load_shard_index(raw)
        old_spec[g] = json.loads(get(f"{prefix}/{g}/tile_grid.json", "grid"))

    # ---- the published store's lanes and their ledgers
    dr = old_meta.get("date_range") or []
    old_lo, old_hi = parse_day(dr[0]), parse_day(dr[1])
    plan_root = plan_root or lanes[0][2].root
    ctx_old = lane_ctx(name, os.path.join(work, "_old"), old_lo, old_hi,
                       adapter_cls)
    ctx_old.lane, ctx_old.lane_groups = "", []
    ctx_old.bins_by_first_day = True
    b1.prepare_grid_ctx(ctx_old)
    # the plan.json the published store was assembled with is the index
    # stage's; the lane's own is the same adapter's answer
    shutil.copy(os.path.join(plan_root, "plan.json"),
                os.path.join(ctx_old.root, "plan.json"))
    lby = old_meta.get("lanes_by_year") or {}
    ledgers = {}
    windows = {}
    for y in ctx_old.years:
        for lane in (lby.get(str(y)) or {}).get("lanes") or [""]:
            rel = _ledger_rel(partials, name, y, lane)
            c = json.loads(get(rel, f"the {y} lane ledger"))
            ledgers[(int(y), lane)] = c
            windows[(int(y), lane)] = c.get("lane_window")
    deg = old_meta.get("degraded") or {}
    counts_all, year_counts = _merge_ledgers(f10b, ledgers)
    rebuilt = b1.grid_store_meta(
        ctx_old, old_spec, old_arr, counts_all, year_counts,
        deg.get("years_admitted_unmarked") or [],
        deg.get("allow_missing_years", False),
        lanes=old_meta.get("lanes_by_year"))
    bad = []
    for k in STRUCTURAL:
        bad += _diff(_norm(old_meta.get(k, "<absent>")),
                     _norm(rebuilt.get(k, "<absent>")), f".{k}")
    drift = []
    for k in DESCRIPTIVE:
        drift += _diff(_norm(old_meta.get(k, "<absent>")),
                       _norm(rebuilt.get(k, "<absent>")), f".{k}")
    rep["descriptive_drift"] = drift
    if bad:
        raise SpliceError(f"{key}: SELF-CHECK REFUSED — the published "
                          f"store.json is not what its own shard indices and "
                          f"lane ledgers rebuild to, so a splice could not be "
                          f"trusted to keep what it does not touch: "
                          + "; ".join(bad[:12]))
    rep["self_check"] = ("ok: store.json rebuilt from the published "
                         f"ledgers ({len(ledgers)} lane ledger(s))")
    if check_only:
        return rep

    # ---- which rows the refreshed lanes replace
    replaced = {}                     # (year, old lane) -> new lane name
    for (y, old_lane, lc) in lanes:
        replaced[(int(y), old_lane)] = lc.lane
    new_ledgers = {k: v for k, v in ledgers.items() if k not in replaced}
    specs = lanes[0][2].grid_specs
    entries = {g: [] for g in groups}
    lane_files = {}                   # rel in store -> local path
    for g in groups:
        for r in sh.array_to_entries(old_arr[g]):
            y = int(r["year"])
            owner = [k for k in replaced if k[0] == y
                     and _bin_owned(windows.get(k), r["bin"])]
            if owner:
                continue
            entries[g].append(r)
    for (y, old_lane, lc) in lanes:
        yd = lc.year_dir(y)
        c = json.load(open(os.path.join(yd, "counts.json")))
        new_ledgers[(int(y), lc.lane)] = c
        for g in sorted(c.get("grids") or {}):
            ip = os.path.join(yd, f"{g}__shard_index.npy")
            for e in sh.array_to_entries(sh.load_shard_index(ip)):
                if sh.bin_year(e["bin"]) != int(y):
                    raise SpliceError(f"{key}: lane {y} holds bin {e['bin']}")
                entries[g].append(e)
                for rel in (e["shard"], e["index"]):
                    lane_files[f"{g}/{rel}"] = os.path.join(
                        yd, b1.part_name(g, rel))
    for g in groups:
        bins = [e["bin"] for e in entries[g]]
        if len(bins) != len(set(bins)):
            raise SpliceError(f"{key} {g}: a bin is both kept and refetched")
    counts_all, year_counts = _merge_ledgers(f10b, new_ledgers)

    # ---- the lanes map, updated for a renamed named lane
    lanes_new = None
    if old_meta.get("lanes_by_year") is not None:
        lanes_new = json.loads(json.dumps(old_meta["lanes_by_year"]))
        for (y, old_lane), nl in replaced.items():
            rec = lanes_new.setdefault(str(y), {"declared": False,
                                                "lanes": []})
            ls = [x for x in rec.get("lanes", []) if x != old_lane] + [nl]
            rec["lanes"] = ([""] if "" in ls else []) + \
                sorted(x for x in set(ls) if x)
    else:
        if any(nl for nl in replaced.values()):
            raise SpliceError(f"{key}: a named refresh lane on a store with "
                              f"no lanes map")

    # ---- the new store.json, by the assembler's own function
    ctx_new = lane_ctx(name, os.path.join(work, "_new"), old_lo,
                       max(old_hi, new_end), adapter_cls)
    ctx_new.lane, ctx_new.lane_groups = "", []
    ctx_new.bins_by_first_day = True
    b1.prepare_grid_ctx(ctx_new)
    shutil.copy(os.path.join(plan_root, "plan.json"),
                os.path.join(ctx_new.root, "plan.json"))
    arrs = b1.write_group_files(stage, specs, entries)
    meta = b1.grid_store_meta(ctx_new, specs, arrs, counts_all, year_counts,
                              deg.get("years_admitted_unmarked") or [],
                              deg.get("allow_missing_years", False),
                              lanes=lanes_new)
    seg = [specs[g].get("source_segments") for g in groups
           if specs[g].get("source_segments")]
    if seg:
        meta["source_segments"] = seg[0]

    # ---- files: what changed, what is new, what is gone
    new_block = {}
    for g in groups:
        for r in arrs[g]:
            for rel in (f"{g}/{r['shard']}", f"{g}/{r['index']}"):
                if rel in lane_files:
                    new_block[rel] = sha256_file(lane_files[rel])
                else:
                    new_block[rel] = block[rel]
        for f in ("tile_grid.json", "shard_index.npy"):
            new_block[f"{g}/{f}"] = sha256_file(os.path.join(stage, g, f))
    adds, changed_bins, new_bins = [], set(), set()
    old_bins = {(g, int(b)) for g in groups for b in old_arr[g]["bin"]}
    for rel, h in sorted(new_block.items()):
        if block.get(rel) == h:
            continue
        local = lane_files.get(rel) or os.path.join(stage, rel)
        adds.append((rel, local))
        m = re.search(r"bin_(-?\d+)\.(zst|idx\.npy)$", rel)
        if m:
            gb = (rel.split("/", 1)[0], int(m.group(1)))
            (changed_bins if gb in old_bins else new_bins).add(gb)
    deletes = sorted(set(block) - set(new_block))
    rep.update(files_changed=len([a for a in adds if a[0] in block]),
               files_added=len([a for a in adds if a[0] not in block]),
               files_deleted=len(deletes),
               bins_rewritten=sorted(b for _, b in changed_bins),
               bins_new=sorted(b for _, b in new_bins))

    # ---- the revision guard and the prefix rule
    old_end = None
    for g in groups:
        for r in old_arr[g]:
            mask = int(r["frame_mask"])
            for f in range(int(old_spec[g]["frames_per_bin"])):
                if mask >> f & 1:
                    t = sh.frame_datetime(int(r["bin"]), f,
                                          old_spec[g]["frame_seconds"])
                    old_end = t if old_end is None or t > old_end else old_end
    last_old_bin = None if old_end is None else \
        int((old_end - dt.datetime(1982, 1, 1)).total_seconds()
            // sh.BIN_SECONDS)
    prov_bins = {bin_of_day(d) for d in provisional}
    unexpected, rewrites = [], []
    old_rows = {(g, int(r["bin"])): r for g in groups for r in old_arr[g]}
    for (g, b) in sorted(changed_bins):
        if b in prov_bins:
            rewrites.append({"group": g, "bin": b, "why": "provisional days "
                             "replaced", "prefix_preserving": None})
            continue
        if last_old_bin is not None and b < last_old_bin:
            unexpected.append((g, b))
            continue
        # the prefix rule: old bytes are the head of the new shard and the
        # old frames' index entries are unchanged
        rel_s = f"{g}/{sh.shard_relpath(b)}"
        rel_i = f"{g}/{sh.index_relpath(b)}"
        old_s = remote.get(f"{prefix}/{rel_s}") or b""
        old_i = np.load(io.BytesIO(remote.get(f"{prefix}/{rel_i}")))
        new_i = np.load(lane_files[rel_i])
        with open(lane_files[rel_s], "rb") as fh:
            head = fh.read(len(old_s))
        mask = int(old_rows[(g, b)]["frame_mask"])
        same_idx = all(np.array_equal(old_i[f], new_i[f])
                       for f in range(old_i.shape[0]) if mask >> f & 1)
        ok = head == old_s and same_idx
        rewrites.append({"group": g, "bin": b, "why": "a partly filled bin "
                         "gained frames", "prefix_preserving": bool(ok)})
        if not ok:
            unexpected.append((g, b))
    rep["rewrites"] = rewrites
    if unexpected and not allow_revisions:
        raise SpliceError(
            f"{key}: REVISION REFUSED — {len(unexpected)} published bin(s) "
            f"would change outside the refresh window (bins "
            f"{[b for _, b in unexpected][:12]}): the producer revised "
            f"history, or a partly filled bin was not rewritten "
            f"prefix-preserving. Look at the upstream before passing "
            f"--allow-revisions; nothing has been committed.")
    rep["revisions_allowed"] = [b for _, b in unexpected]

    # ---- every changed or new bin decoded in full before any upload
    mini = os.path.join(work, "mini")
    shutil.rmtree(mini, ignore_errors=True)
    touched = changed_bins | new_bins
    for g in groups:
        gd = os.path.join(mini, g)
        os.makedirs(gd)
        shutil.copy(os.path.join(stage, g, "tile_grid.json"), gd)
        rows = [e for e in sh.array_to_entries(arrs[g])
                if (g, int(e["bin"])) in touched]
        sh.save_shard_index(os.path.join(gd, "shard_index.npy"), rows,
                            specs[g]["C"])
        for e in rows:
            for rel in (e["shard"], e["index"]):
                dst = os.path.join(gd, rel)
                os.makedirs(os.path.dirname(dst), exist_ok=True)
                shutil.copyfile(lane_files[f"{g}/{rel}"], dst)
        if rows:
            s = sh.check_sharded(gd, sample=None)
            rep.setdefault("decoded", {})[g] = s["tiles_checked"]
    rep["mini"] = mini

    # ---- store.json and manifest.json
    meta["sha256"] = new_block
    meta["refresh"] = {
        "by": "ml/data_refresh.py (E-090)", "plan": PLAN_DOC, "at": utcnow(),
        "lanes": [{"year": y, "replaced": old or "(unnamed)",
                   "lane": lc.lane or "(unnamed)"} for (y, old, lc) in lanes],
        "record_end_before": iso(old_end.date()) if old_end else None,
        "bins_rewritten": rep["bins_rewritten"], "bins_new": rep["bins_new"],
        "files_deleted": deletes, "previous_built_at": old_meta.get(
            "built_at"),
        "previous_builder_git_sha": old_meta.get("builder_git_sha")}
    sp = os.path.join(stage, "store.json")
    write_json(sp, meta)
    adds.append(("store.json", sp))
    man_raw = remote.get(f"{prefix}/manifest.json")
    if man_raw:
        man = json.loads(man_raw)
        sizes = {f["name"]: f.get("bytes") for f in man.get("files", [])}
        for rel, local in adds:
            sizes[rel] = os.path.getsize(local)
        man["files"] = [{"name": n, "bytes": sizes.get(n),
                         "sha256": new_block[n]} for n in sorted(new_block)]
        man["groups"] = meta["groups"]
        man["date_range"] = meta["date_range"]
        man["refresh"] = {"at": meta["refresh"]["at"],
                          "by": meta["refresh"]["by"]}
        mp = os.path.join(stage, "manifest.json")
        write_json(mp, man)
        adds.append(("manifest.json", mp))
    rep["adds"] = [(f"{prefix}/{r}", p) for r, p in adds]
    rep["deletes"] = [f"{prefix}/{r}" for r in deletes]
    rep["meta"] = meta
    rep["old_meta"] = old_meta
    rep["new_frames"] = new_frames(old_arr, arrs, specs)
    return rep


def new_frames(old_arr, arrs, specs):
    """[(group, bin, frame)] present in the new shard index and not the old."""
    out = []
    for g, arr in arrs.items():
        old = {int(r["bin"]): int(r["frame_mask"]) for r in old_arr[g]}
        for r in arr:
            b, m = int(r["bin"]), int(r["frame_mask"])
            for f in range(int(specs[g]["frames_per_bin"])):
                if m >> f & 1 and not old.get(b, 0) >> f & 1:
                    out.append((g, b, f))
    return out


# ======================================================= commit + verify ====
def commit_and_verify(key, remote, rep, message):
    """ONE commit, then every committed file read back at that revision and
    the new frames re-read through the HTTP reader. A failure reverts."""
    from family1 import sharded as sh
    prefix = f"tensors/{key}"
    # what to restore if the read-back fails
    backup = {}
    for rel, _ in rep["adds"]:
        backup[rel] = remote.get(rel)
    for rel in rep["deletes"]:
        backup[rel] = remote.get(rel)
    t0 = time.time()
    oid = remote.commit(rep["adds"], rep["deletes"], message)
    rep["commit"] = oid
    rep["commit_seconds"] = round(time.time() - t0, 1)
    try:
        for rel, local in rep["adds"]:
            b = hub_read(lambda: remote.get(rel, revision=oid))
            if b is None or sha256_bytes(b) != sha256_file(local):
                raise SpliceError(f"READ-BACK MISMATCH {rel} at {oid}")
        for rel in rep["deletes"]:
            if remote.get(rel, revision=oid) is not None:
                raise SpliceError(f"{rel} still on the Hub at {oid}")
        nf = rep["new_frames"]
        pick = nf if len(nf) <= VERIFY_FRAMES else \
            [nf[int(i)] for i in np.linspace(0, len(nf) - 1, VERIFY_FRAMES)]
        checked = tiles = 0
        groups = {}
        for (g, b, f) in pick:
            if g not in groups:
                groups[g] = (sh.ShardedGroup(f"{remote.base(oid)}/{prefix}/"
                                             f"{g}", attempts=6),
                             sh.ShardedGroup(os.path.join(rep["mini"], g)))
            hub, loc = groups[g]
            for (ty, tx) in verify_tiles(loc, b, f):
                got = hub_read(lambda: hub.read_tile(b, f, ty, tx, raw=True))
                want = loc.read_tile(b, f, ty, tx, raw=True)
                if got is None or want is None or not np.array_equal(
                        got, want, equal_nan=want.dtype.kind == "f"):
                    raise SpliceError(f"HTTP RE-READ MISMATCH {g} bin {b} "
                                      f"frame {f} tile ({ty},{tx}) at {oid}")
                tiles += 1
            checked += 1
        rep["verified"] = {"files_read_back": len(rep["adds"]),
                           "frames_reread": checked, "tiles_reread": tiles,
                           "at_revision": oid}
    except BaseException as e:
        print(f"::error::{key}: verification failed after commit {oid} "
              f"({e}) — reverting", flush=True)
        tmpd = os.path.join(os.path.dirname(rep["mini"]), "revert")
        os.makedirs(tmpd, exist_ok=True)
        adds, dels = [], []
        for i, (rel, b) in enumerate(backup.items()):
            if b is None:
                dels.append(rel)
            else:
                p = os.path.join(tmpd, f"{i}.bin")
                with open(p, "wb") as fh:
                    fh.write(b)
                adds.append((rel, p))
        rep["revert_commit"] = remote.commit(adds, dels,
                                             f"E-090: revert {key} ({oid})")
        raise
    return rep


VERIFY_TILES = 24          # per re-read frame: all of a coarse grid's 18


def verify_tiles(grp, b, f, k=VERIFY_TILES):
    """The tiles of frame (b, f) the HTTP re-read compares: every tile when
    the frame has at most k, else k of its STORED tiles spread evenly in
    write order (plus the first tile, stored or not) — a fine grid's frame
    is hundreds of tiles and the Hub's resolver answers 429 to a full
    re-read of dozens of them (data-refresh #6, pace4k: 578 tiles a frame)."""
    from family1 import sharded as sh
    sp = grp.spec
    allt = [(ty, tx) for ty in range(sp["n_tiles_y"])
            for tx in range(sp["n_tiles_x"])]
    if len(allt) <= k:
        return allt
    idx = np.load(os.path.join(grp.src.base, sh.index_relpath(b)))
    stored = [(ty, tx) for (ty, tx) in allt if idx[f, ty, tx, 1] > 0]
    if len(stored) > k - 1:
        stored = [stored[int(i)] for i in np.linspace(0, len(stored) - 1,
                                                      k - 1)]
    return sorted(set([allt[0]] + stored))


def hub_read(fn, budget_s=1200):
    """`fn()` (a Hub range read) retried through the Hub's rate limit: an
    HTTP 429 or a 5xx sleeps (60 s doubling, up to `budget_s` in all) — a
    throttled read is not a failed verification. Anything else raises."""
    waited, nap = 0.0, 60.0
    while True:
        try:
            return fn()
        except (IOError, OSError) as e:
            msg = str(e)
            if not re.search(r"\b(429|5\d\d)\b|Too Many|timed out|"
                             r"Connection", msg) or waited >= budget_s:
                raise
            print(f"  ::warning::Hub read throttled ({msg[-80:]}) — "
                  f"sleeping {nap:.0f} s", flush=True)
            time.sleep(nap)
            waited += nap
            nap = min(nap * 2, 300.0)


def push_lane_parts(lanes, remote, key):
    """The refreshed lanes' parts onto the Hub's partials (the build's own
    push), and a renamed named lane's old folder removed — so a later
    from-scratch assembly from parts reproduces the refreshed store."""
    import build_family1_stores as b1
    import family10_parts_hub as ph
    out = []
    slug, name = store_parts(key)
    for (y, old_lane, lc) in lanes:
        if isinstance(remote, LocalRemote):
            d = lc.year_dir(y)
            pre = ph.hub_prefix(name, y, f"partials/{slug}", lc.lane or None)
            dst = os.path.join(remote.root, pre)
            shutil.rmtree(dst, ignore_errors=True)
            os.makedirs(dst)
            for n in ph.local_part_files(d):
                shutil.copyfile(os.path.join(d, n), os.path.join(dst, n))
        else:
            b1.push_parts(lc)
        out.append({"year": y, "lane": lc.lane or "(unnamed)"})
        if old_lane and old_lane != lc.lane:
            pre = f"partials/{slug}/{name}/{y}/{old_lane}"
            stale = remote.list(pre)
            if stale:
                remote.commit([], stale, f"E-090: {key} {y} lane {old_lane} "
                                         f"superseded by {lc.lane}")
                out[-1]["removed"] = pre
    return out


# ================================================================ update ====
def update(key, remote, work, *, dry_run=False, force_from=None,
           allow_revisions=False, adapter_cls=None, upstream=None,
           push_parts=True):
    """Plan, fetch, splice, commit, verify one store. Returns the result."""
    pol = POLICY.get(key, {})
    t0 = time.time()
    res = {"store": key, "started_utc": utcnow(), "dry_run": bool(dry_run)}
    if upstream is not None:                      # tests hand the plan in
        p = upstream
    else:
        p = plan_store(key, remote, force_from=force_from)
    res["plan"] = p
    slug, name = store_parts(key)
    if p.get("action") != "update":
        res["state"] = "noop" if p.get("action") == "noop" else \
            ("error" if p.get("action") == "error" else "skipped")
        res["detail"] = p.get("reason") or p.get("error") or p.get("why")
        if not (dry_run and p.get("action") == "noop"):
            return res
        # a dry run of a current store still asks whether its published
        # ledgers explain it — the precondition of every later refresh
        meta = json.loads(remote.get(f"tensors/{key}/store.json"))
        end = parse_day(p["record_end"])
        todo = refresh_lanes(meta, bin_of_day(end), end)
    else:
        meta = json.loads(remote.get(f"tensors/{key}/store.json"))
        new_end = parse_day(p["new_end"])
        b0 = bin_of_day(parse_day(p["refetch_from"]))
        todo = refresh_lanes(meta, b0, new_end)
    res["lanes"] = [{"year": y, "replaces": n or "(unnamed)",
                     "window": [iso(lo), iso(hi)]} for y, n, lo, hi in todo]
    if dry_run:
        # read-only: the index stage (for plan.json) and the splice's
        # self-check — does the published store's own ledger explain it?
        import build_family1_stores as b1
        import build_family10_stores as f10b
        y, _o, lo, hi = todo[0]
        ic = lane_ctx(name, os.path.join(work, "dry_index"), lo, hi,
                      adapter_cls)
        f10b.run_stages(ic, ["index"], stage_fn=b1.GRID_STAGE_FN,
                        deps=b1.DEPS)
        try:
            ck = splice(key, remote, [], os.path.join(work, "dry_check"),
                        adapter_cls=adapter_cls, plan_root=ic.root,
                        check_only=True)
            res["self_check"] = ck["self_check"]
            res["descriptive_drift"] = ck["descriptive_drift"]
        except SystemExit as e:
            res["state"] = "error"
            res["detail"] = str(e)[:2000]
            return res
        if p.get("action") != "update":
            return res
        res["state"] = "would-update"
        res["detail"] = (f"would re-fetch {len(todo)} lane(s) "
                         f"{[(y, iso(lo), iso(hi)) for y, _, lo, hi in todo]}"
                         f" to reach {p['new_end']} (now {p['record_end']})")
        return res
    if pol.get("record_env"):
        os.environ["F1_RECORD_END"] = p["new_end"]
    if pol.get("upstream_first"):
        os.environ["F7D_UPSTREAM_FIRST"] = "1"
    lanes = []
    for (y, old, lo, hi) in todo:
        lc = run_lane(name, os.path.join(work, f"lane_{y}"), lo, hi,
                      adapter_cls)
        lanes.append((y, old, lc))
    res["fetch_seconds"] = round(time.time() - t0, 1)
    prov = [parse_day(d) for d in p.get("provisional_days") or []]
    rep = splice(key, remote, lanes, os.path.join(work, "splice"),
                 new_end=new_end, adapter_cls=adapter_cls,
                 allow_revisions=allow_revisions, provisional=prov)
    meta_new = rep["meta"]
    g0 = sorted(meta_new["groups"])[0]
    res.update({k: rep[k] for k in ("files_changed", "files_added",
                                    "files_deleted", "bins_rewritten",
                                    "bins_new", "rewrites",
                                    "descriptive_drift", "self_check",
                                    "decoded")
                if k in rep})
    res["frames_new"] = len(rep["new_frames"])
    if not rep["adds"] or (rep["files_changed"] + rep["files_added"]
                           + rep["files_deleted"]) == 0:
        res["state"] = "noop"
        res["detail"] = "the re-fetched lanes are byte-identical to the store"
        return res
    msg = (f"E-090 refresh {key}: {res['frames_new']} new frame(s), "
           f"{len(rep['bins_new'])} new / {len(rep['bins_rewritten'])} "
           f"rewritten bin(s), record to {p['new_end']}")
    commit_and_verify(key, remote, rep, msg)
    res["commit"] = rep["commit"]
    res["verified"] = rep["verified"]
    if push_parts:
        try:
            res["parts"] = push_lane_parts(lanes, remote, key)
        except BaseException as e:                            # noqa: BLE001
            # the store is committed and verified; stale parts only cost a
            # from-scratch rebuild its tail, and the status line says so
            res["parts_error"] = f"{type(e).__name__}: {str(e)[:300]}"
    res["months_touched"] = months_touched(rep, meta_new)
    res["record_end_before"] = p["record_end"]
    res["record_end_after"] = p["new_end"]
    res["group"] = g0
    res["state"] = "updated"
    res["seconds"] = round(time.time() - t0, 1)
    return res


def months_touched(rep, meta):
    """[[year, month]] whose frames changed — the E-088 sums to remake."""
    from family1 import sharded as sh
    fs = int(meta["frame_seconds"])
    F = int(meta["frames_per_bin"])
    out = set()
    for b in list(rep["bins_new"]) + list(rep["bins_rewritten"]):
        for f in range(F):
            t = sh.frame_datetime(int(b), f, fs)
            out.add((t.year, t.month))
    return sorted([y, m] for y, m in out)


# ============================================================== registry ====
def registry_publish(results, out_dir, dry_run=False):
    """Rebuild the family-1 registries, guard each against the published
    copy allowing ONLY the refreshed stores' data fields to change, publish
    the families that hold a refreshed store. The registry is the commit
    point of the Data tab: until it is published the new days are not
    offered."""
    import build_family1_registry as r
    fams = {}
    for res in results:
        if res.get("state") != "updated":
            continue
        pol = POLICY[res["store"]]
        fams.setdefault(pol["fam"], []).append(store_parts(res["store"])[1])
    if not fams:
        print("registry: no store was updated — nothing to publish")
        return {}
    os.makedirs(out_dir, exist_ok=True)
    subprocess.run([sys.executable, "-u",
                    os.path.join(HERE, "build_family1_registry.py"),
                    "--out-dir", out_dir, "--check"], check=True)
    out = {}
    for fam, names in sorted(fams.items()):
        from family1.adapters import FAMILIES
        rel = f"tensors/{FAMILIES[fam][2]}/{r.REGISTRY_NAME[fam]}"
        newp = os.path.join(out_dir, r.REGISTRY_NAME[fam])
        pub = os.path.join(out_dir, "published_" + r.REGISTRY_NAME[fam])
        b = http_get(f"{HUB}/main/{rel}")
        with open(pub, "wb") as fh:
            fh.write(b)
        args = [sys.executable, os.path.join(HERE, "registry_guard.py"),
                pub, newp]
        for n in names:
            args += ["--refresh", n]
        subprocess.run(args, check=True)
        if dry_run:
            out[fam] = {"guarded": names, "published": False}
            continue
        r.publish_one(newp, fam)
        out[fam] = {"guarded": names, "published": True, "rel": rel,
                    "previous": pub}
    return out


# ================================================================== sums ====
GM_INDEX = os.path.join(ROOT, "data", "gridded_monthly_index.json")
FINE_INDEX = os.path.join(ROOT, "data", "gridded_monthly_fine_index.json")


def sums_update(res, work, dry_run=False, version=None):
    """E-088's sums for one refreshed store (`res` its update result):
    remake the settled months it touched in a NEW version folder, verify the
    written planes against numpy over the native frames, upload, stream the
    files back, and hand back the store's new index block. The committed
    `data/gridded_monthly_index.json` is the sums' commit point — the finish
    job writes it; until then the tab reads the previous version, whose
    files are left untouched."""
    import export_gridded_monthly as X
    import publish_gridded_monthly_index as P
    key = res["store"]
    pol = POLICY.get(key, {})
    out = {"store": key, "kind": "sums"}
    if pol.get("sums") == "e089":
        out["state"] = "hook-waiting"
        out["detail"] = sums_hook_e089(res)
        return out
    if not pol.get("sums"):
        out["state"] = "none"
        return out
    idx = json.load(open(GM_INDEX, encoding="utf-8"))
    blk = (idx.get("stores") or {}).get(key)
    if not blk:
        out.update(state="error", detail=f"{key} has no block in "
                                         f"data/gridded_monthly_index.json")
        return out
    old = {}
    for role in ("sum", "count", "m2", "stats"):
        old[role] = blk[role]["url"]
        old[f"{role}_sha256"] = blk[role]["sha256"]
    meta = json.loads(HubRemote().get(f"tensors/{key}/store.json"))
    prov = set()
    for v in (meta.get("counts_by_year") or {}).values():
        prov |= set(v.get("preliminary_days") or [])
        prov |= set(v.get("preliminary_status_unknown_days") or [])
    d = os.path.join(work, "sums")
    keep = os.path.join(work, "sums_keep")
    shutil.rmtree(d, ignore_errors=True)
    shutil.rmtree(keep, ignore_errors=True)
    if dry_run:
        out.update(state="would-update",
                   detail=f"would remake the settled months of "
                          f"{res.get('months_touched')} from {blk['sum']['url']}")
        return out
    st = X.update(key, d, old, touched=res.get("months_touched") or [],
                  workers=2, threads=4, keep=keep, provisional=sorted(prov))
    if st is None:
        out.update(state="current", detail="no settled month changed")
        return out
    rep = X.verify(key, d, keep, report=os.path.join(d, "verify.json"))
    shutil.rmtree(keep, ignore_errors=True)
    version = version or ("r" + dt.datetime.now(dt.timezone.utc)
                          .strftime("%Y%m%dT%H%M"))
    P.upload_store(key, d, version=version)
    # the block, from a full streamed restore of what was just uploaded
    tmpi = os.path.join(work, "gm_index_copy.json")
    shutil.copy(GM_INDEX, tmpi)
    P.main(["index", "--merge", "--stores", key, "--version",
            f"{key}={version}", "--index", tmpi, "--no-cors"])
    new_blk = json.load(open(tmpi))["stores"][key]
    out.update(state="updated", version=version, block=new_blk,
               written=st["update"]["written"],
               blanked=st["update"]["blanked"],
               falsifier={k: rep[k] for k in ("ok", "n_planes",
                                              "max_mean_over_bound",
                                              "max_std_over_tol")})
    shutil.rmtree(d, ignore_errors=True)
    return out


def sums_hook_e089(res):
    """THE E-089 HOOK. The fine 1.gf grids (oc4k, pace4k, sst_acspo02,
    irtb) get per-(year, month) native sum tiles plus a pooled 0.25° layer
    (`ml/export_fine_monthly.py`, index `data/gridded_monthly_fine_index.
    json`). Their native layer is one file per month, so a refresh writes the
    touched SETTLED months' files only (same rule as E-088: complete, no
    provisional day; other changed months blanked) and the pooled layer as a
    new version folder; the index is the commit point. Wired when both the
    index and an `update` entry point exist on main."""
    if not os.path.exists(FINE_INDEX):
        return ("E-089's index (data/gridded_monthly_fine_index.json) is not "
                "on main yet — the fine-grid sums are not refreshed; the Data "
                "tab reads native maps for any month they do not cover")
    try:
        import export_fine_monthly as XF
    except ImportError:
        return "ml/export_fine_monthly.py is not importable"
    if not hasattr(XF, "update"):
        return ("ml/export_fine_monthly.py has no `update` yet — hook "
                "designed (plan §7), waiting for it")
    return "E-089 update present — wire it here (plan §7)"


def finish_index(sums_results, index=GM_INDEX):
    """Merge the refreshed stores' new blocks into the committed index."""
    idx = json.load(open(index, encoding="utf-8"))
    changed = []
    for r in sums_results:
        if r.get("state") == "updated" and r.get("block"):
            idx["stores"][r["store"]] = r["block"]
            changed.append(r["store"])
    if changed:
        idx["generated_utc"] = utcnow()
        idx["refreshed"] = {"by": "ml/data_refresh.py finish (E-090)",
                            "stores": changed, "at": utcnow()}
        import publish_family7_monthly_index as P7
        P7.write_json(index, idx)
    return changed


# ================================================================ status ====
def status_lines(results, plan_rows, previous=None):
    """One line per store: last checked, last updated, record end, state."""
    prev = {s["store"]: s for s in (previous or {}).get("stores", [])}
    by = {r["store"]: r for r in results}
    out = []
    for p in plan_rows:
        k = p["store"]
        r = by.get(k, {})
        old = prev.get(k, {})
        st = r.get("state") or {"noop": "current", "error": "error",
                                "update": "pending", "manual": "manual",
                                "none": p.get("kind")}.get(p.get("action"),
                                                          p.get("action"))
        if st == "noop":
            st = "current"
        line = {
            "store": k, "kind": p.get("kind"), "state": st,
            "last_checked_utc": p.get("checked_utc"),
            "upstream_newest": p.get("upstream_newest"),
            "record_end": r.get("record_end_after") or p.get("record_end"),
            "gap_days": (0 if r.get("state") == "updated"
                         else p.get("gap_days")),
            "last_updated_utc": (r.get("started_utc")
                                 if r.get("state") == "updated"
                                 else old.get("last_updated_utc")),
            "detail": (r.get("detail") or p.get("error") or p.get("reason")
                       or p.get("why")),
        }
        if r.get("commit"):
            line["commit"] = r["commit"]
        out.append(line)
    # a dispatch for a few stores keeps every other store's last line
    seen = {ln["store"] for ln in out}
    out += [v for k, v in sorted(prev.items()) if k not in seen]
    return {"_source": "ml/data_refresh.py status (E-090) — do not "
                       "hand-edit", "plan": PLAN_DOC,
            "generated_utc": utcnow(), "stores": out}


# =================================================================== cli ====
def resolve_stores(spec):
    if spec in ("", "auto", "scheduled"):
        return list(SCHEDULED)
    if spec == "all":
        return sorted(POLICY)
    out = [s.strip() for s in spec.split(",") if s.strip()]
    for s in out:
        if s not in POLICY:
            raise SystemExit(f"unknown store {s!r}; known: {sorted(POLICY)}")
    return out


def load_results(d):
    out = []
    for p in sorted(glob.glob(os.path.join(d, "**", "result*.json"),
                              recursive=True)):
        try:
            out.append(json.load(open(p)))
        except ValueError:
            pass
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("plan")
    p.add_argument("--stores", default="auto")
    p.add_argument("--json", default="")
    p.add_argument("--matrix", default="",
                   help="write the stores needing an update as a JSON list")
    p.add_argument("--force-from", default="")
    p.add_argument("--all-grid", action="store_true")
    u = sub.add_parser("update")
    u.add_argument("--store", required=True)
    u.add_argument("--work", required=True)
    u.add_argument("--dry-run", action="store_true")
    u.add_argument("--result", default="")
    u.add_argument("--force-from", default="")
    u.add_argument("--allow-revisions", action="store_true")
    u.add_argument("--no-push-parts", action="store_true")
    g = sub.add_parser("registry")
    g.add_argument("--results", required=True)
    g.add_argument("--out", default="registries")
    g.add_argument("--dry-run", action="store_true")
    g.add_argument("--json", default="")
    m = sub.add_parser("sums")
    m.add_argument("--result", required=True,
                   help="the store's update result.json")
    m.add_argument("--work", required=True)
    m.add_argument("--out", required=True)
    m.add_argument("--dry-run", action="store_true")
    f = sub.add_parser("finish")
    f.add_argument("--sums", required=True, help="dir of sums results")
    s = sub.add_parser("status")
    s.add_argument("--results", required=True)
    s.add_argument("--plan", required=True)
    s.add_argument("--out", default="status.json")
    s.add_argument("--upload", action="store_true")
    a = ap.parse_args(argv)
    remote = HubRemote()
    if a.cmd == "plan":
        rows = plan(resolve_stores(a.stores), remote,
                    force_from=a.force_from or None)
        for r in rows:
            print(f"  {r['store']:28s} {r.get('action'):8s} ours "
                  f"{r.get('record_end')} upstream {r.get('upstream_newest')}"
                  f" gap {r.get('gap_days')} "
                  + (f"years {r.get('years')}" if r.get("years") else "")
                  + (f" — {r.get('error') or r.get('reason') or r.get('why')}"
                     if (r.get('error') or r.get('reason') or r.get('why'))
                     else ""), flush=True)
        if a.json:
            write_json(a.json, {"generated_utc": utcnow(), "stores": rows})
        if a.matrix:
            # an update matrix: the stores with something to do; with
            # --all-grid (a dry run) every tier-G store asked for, so each
            # runs its self-check
            todo = [r["store"] for r in rows
                    if r.get("action") == "update"
                    or (a.all_grid and POLICY[r["store"]]["kind"] == "grid"
                        and r.get("action") in ("noop", "update"))]
            with open(a.matrix, "w") as fh:
                json.dump([{"store": k, "name": k.replace("/", "__")}
                           for k in todo], fh)
        return 0
    if a.cmd == "update":
        res = update(a.store, remote, a.work, dry_run=a.dry_run,
                     force_from=a.force_from or None,
                     allow_revisions=a.allow_revisions,
                     push_parts=not a.no_push_parts)
        if a.result:
            write_json(a.result, res)
        print(json.dumps({k: v for k, v in res.items() if k != "plan"},
                         indent=1, default=str)[:4000])
        return 0 if res.get("state") in ("updated", "noop", "would-update",
                                         "skipped") else 1
    if a.cmd == "registry":
        out = registry_publish(load_results(a.results), a.out,
                               dry_run=a.dry_run)
        if a.json:
            write_json(a.json, out)
        print(json.dumps(out, indent=1))
        return 0
    if a.cmd == "sums":
        res = json.load(open(a.result))
        if res.get("state") != "updated":
            out = {"store": res.get("store"), "kind": "sums",
                   "state": "none", "detail": "the store was not updated"}
        else:
            out = sums_update(res, a.work, dry_run=a.dry_run)
        write_json(a.out, out)
        print(json.dumps({k: v for k, v in out.items() if k != "block"},
                         indent=1, default=str))
        return 0 if out.get("state") != "error" else 1
    if a.cmd == "finish":
        rs = []
        for p_ in sorted(glob.glob(os.path.join(a.sums, "**", "*.json"),
                                   recursive=True)):
            try:
                rs.append(json.load(open(p_)))
            except ValueError:
                pass
        ch = finish_index(rs)
        print(f"index: {len(ch)} store block(s) replaced: {ch}")
        return 0
    if a.cmd == "status":
        rows = json.load(open(a.plan))["stores"]
        prev = None
        try:
            b = remote.get(STATUS_PATH)
            prev = json.loads(b) if b else None
        except Exception:                                     # noqa: BLE001
            prev = None
        st = status_lines(load_results(a.results), rows, prev)
        write_json(a.out, st)
        if a.upload:
            remote.commit([(STATUS_PATH, a.out)], [],
                          "E-090: data refresh status")
        for ln in st["stores"]:
            print(f"  {ln['store']:28s} {ln['state']:9s} record "
                  f"{ln['record_end']} upstream {ln['upstream_newest']}")
        return 0
    return 2


if __name__ == "__main__":
    sys.exit(main())

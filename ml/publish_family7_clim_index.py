#!/usr/bin/env python3
"""E-083 §3.4–3.5 · Publish the family-7 climatology and write its index.

`data/family7_clim_index.json` is the address book the globe's "Model
climatology" layer reads before it touches the Hub — the sibling of
`data/family7_index.json` (`ml/publish_family7_index.py`), never hand-edited.
It says, for every (version, group) that `ml/export_family7_clim.py` wrote:
where `clim.npy`, `clim.nc` and `stats.json` live on the Hub, the parsed
`.npy` header length, shape, dtype and the byte size of one (month, channel)
plane (so one plane is one `Range:` read), and each file's size and sha256.
It also carries the three versions (names, year rules, held-out years) and —
COPIED from the family-7 index, never retyped — each group's grid, channel
names, labels, units, ramps, signs and `norm`, so there is one source of
truth for channel metadata. It refuses to be built against a family-7 index
whose `stem` is not the tensor the climatology was computed from.

Two subcommands:

  upload  every file under <out>/<version>/<group>/ goes to
          tensors/<stem>/clim/<version>/<group>/ on chfrank/earth-tensors
          (one commit per version, through build_family7.hub_commit's retry
          ladder), is DOWNLOADED BACK, and must match its sha256 before the
          index is written — `ml/hf_mirror.py`'s rule: an upload that returned
          200 is not evidence the bytes are retrievable (ml/CLAUDE.md §0.2).
  index   write the index from files already on the Hub (still downloaded
          back and sha256-checked), or with `--local` from the local files
          alone (the fixture: `restore_verified: false` says so in the file).

CORS is MEASURED (`Origin: https://blauewelt.github.io`, a ranged GET that
must answer 206 with access-control-allow-origin) on one clim.npy, exactly as
the family-7 index measures it; `--no-cors` (the fixture) records null.

Credentials: `HF_TOKEN` in the environment. Never argv.

    python3 ml/publish_family7_clim_index.py upload --out /data/clim
    python3 ml/publish_family7_clim_index.py index --out data/family7_clim/fixture \\
        --f7-index data/family7/fixture/family7_index.json \\
        --index data/family7_clim/fixture/family7_clim_index.json --no-cors --local
"""
import argparse
import copy
import datetime as _dt
import hashlib
import json
import os
import shutil
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
if HERE not in sys.path:
    sys.path.insert(0, HERE)

from export_family7_clim import VERSIONS, PLAN_URL, version_block  # noqa: E402
from publish_family7_index import (                                # noqa: E402
    ORIGIN, parse_npy_header, sha256_file)

REPO_ID = "chfrank/earth-tensors"
REPO_TYPE = "dataset"
F7_INDEX = os.path.join(ROOT, "data", "family7_index.json")
INDEX = os.path.join(ROOT, "data", "family7_clim_index.json")
FILES = (("clim_npy", "clim.npy"), ("clim_nc", "clim.nc"),
         ("stats", "stats.json"))
# Channel metadata copied from the family-7 index — one source of truth.
COPY_KEYS = ("grid", "chans", "labels", "units", "ramp", "sign", "norm")


def prefix_of(stem):
    return f"tensors/{stem}/clim"


def base_of(stem):
    return (f"https://huggingface.co/datasets/{REPO_ID}/resolve/main/"
            f"{prefix_of(stem)}/")


def now_utc():
    return _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def token():
    t = os.environ.get("HF_TOKEN", "").strip()
    if not t:
        raise SystemExit("no HF_TOKEN in the environment — the token travels "
                         "by env only, never argv")
    return t


# ------------------------------------------------------------------ collect --
def collect(out, f7):
    """Scan <out>/<version>/<group>/ and check every file against the family-7
    index. Returns (versions, groups, entries) where entries[v][g] is
    {role: (local_path, rel_path)} plus the parsed stats."""
    stem = f7["stem"]
    order = [v["key"] for v in VERSIONS]
    found = [v for v in order if os.path.isdir(os.path.join(out, v))]
    stray = [x for x in sorted(os.listdir(out))
             if os.path.isdir(os.path.join(out, x)) and x not in order
             and not x.startswith("_")]
    if stray:
        raise SystemExit(f"{out}: unknown version folder(s) {stray} — the "
                         f"versions are {order}")
    if not found:
        raise SystemExit(f"{out}: no <version>/<group>/ folders — run "
                         f"ml/export_family7_clim.py first")
    group_order = list(f7["groups"])
    groups, entries = None, {}
    for v in found:
        here = sorted(x for x in os.listdir(os.path.join(out, v))
                      if os.path.isdir(os.path.join(out, v, x)))
        unknown = [g for g in here if g not in f7["groups"]]
        if unknown:
            raise SystemExit(f"{out}/{v}: group(s) {unknown} are not in the "
                             f"family-7 index ({group_order})")
        gs = [g for g in group_order if g in here]
        if groups is None:
            groups = gs
        elif gs != groups:
            raise SystemExit(f"{out}/{v} has groups {gs} and {found[0]} has "
                             f"{groups} — every version must carry the same "
                             f"groups or the page's version switch breaks")
        entries[v] = {}
        for g in gs:
            where = os.path.join(out, v, g)
            st_path = os.path.join(where, "stats.json")
            if not os.path.exists(st_path):
                raise SystemExit(f"{where}: no stats.json")
            st = json.load(open(st_path, encoding="utf-8"))
            if st.get("tensor_stem") != stem:
                raise SystemExit(
                    f"REFUSING: {st_path} was computed from "
                    f"{st.get('tensor_stem')!r} and the family-7 index "
                    f"describes {stem!r}. The clim index copies its channel "
                    f"metadata and norms from the family-7 index, so the two "
                    f"must name the same tensor.")
            if st["version"]["key"] != v or st["version"] != version_block(
                    next(x for x in VERSIONS if x["key"] == v)):
                raise SystemExit(f"{st_path}: its version block is not "
                                 f"VERSIONS[{v!r}] — re-export")
            blk = f7["groups"][g]
            if st["chans"] != blk["chans"] or st["norm"] != blk["norm"]:
                raise SystemExit(f"{st_path}: channels/norm differ from the "
                                 f"family-7 index's {g} block")
            if st.get("tensor_sha256") != blk["sha256"]:
                raise SystemExit(f"{st_path}: computed from {g} sha256 "
                                 f"{st.get('tensor_sha256')}, the family-7 "
                                 f"index says {blk['sha256']}")
            e = {"_stats": st}
            for role, name in FILES:
                p = os.path.join(where, name)
                if os.path.exists(p):
                    e[role] = (p, f"{v}/{g}/{name}")
                elif role != "clim_nc":
                    raise SystemExit(f"{where}: missing {name}")
            entries[v][g] = e
    return found, groups, entries


def npy_block(path, f7_blk):
    """The .npy's own header, parsed and checked against the family-7 grid."""
    with open(path, "rb") as fh:
        head = fh.read(4096)
    header_len, shape, dtype, fortran = parse_npy_header(head)
    nbytes = os.path.getsize(path)
    C, ny, nx = len(f7_blk["chans"]), f7_blk["grid"]["ny"], f7_blk["grid"]["nx"]
    if fortran:
        raise SystemExit(f"{path}: Fortran order — the plane arithmetic "
                         f"assumes C order")
    if dtype != "<f4":
        raise SystemExit(f"{path}: dtype {dtype}, expected <f4")
    if shape != [12, C, ny, nx]:
        raise SystemExit(f"{path}: shape {shape}, expected [12, {C}, {ny}, "
                         f"{nx}] (month, channel, lat, lon)")
    plane = ny * nx * 4
    if header_len + 12 * C * plane != nbytes:
        raise SystemExit(f"{path}: header says {header_len + 12 * C * plane} "
                         f"bytes, the file is {nbytes} — the range "
                         f"arithmetic would be wrong")
    return dict(header_len=int(header_len), shape=[int(x) for x in shape],
                dtype=dtype, fortran_order=False, plane_bytes=int(plane),
                bytes=int(nbytes))


# ------------------------------------------------------------------ restore --
def verify_restore(entries, stem, download):
    """Download every file BACK and compare sha256 with the local bytes.

    `download(path_in_repo) -> local path` (hf_hub_download on the box; a fake
    in the tests). Raises SystemExit on the first mismatch, so no index is
    written for a mirror that does not hold our bytes."""
    n = 0
    for v, gs in entries.items():
        for g, e in gs.items():
            for role, _ in FILES:
                if role not in e:
                    continue
                local, rel = e[role]
                want = sha256_file(local)
                back = download(f"{prefix_of(stem)}/{rel}")
                got = sha256_file(back)
                if got != want:
                    raise SystemExit(
                        f"REFUSING to write the index: {rel} uploaded as "
                        f"sha256 {want} but downloaded back as {got} — the "
                        f"Hub does not hold our bytes")
                n += 1
                print(f"  restored ✓ {rel} ({os.path.getsize(local) / 1e6:.2f}"
                      f" MB)", flush=True)
    return n


def hub_download_fn(tok, repo=REPO_ID):
    """hf_hub_download into a fresh temp dir, through hub_retry."""
    from huggingface_hub import hf_hub_download
    from build_family7 import hub_retry
    tmp = tempfile.mkdtemp(prefix="f7clim_restore_")

    def download(path_in_repo):
        return hub_retry(
            lambda: hf_hub_download(repo, path_in_repo, repo_type=REPO_TYPE,
                                    token=tok, local_dir=tmp,
                                    force_download=True),
            f"download {path_in_repo}")
    download.tmp = tmp
    return download


def upload(entries, stem, api, repo=REPO_ID, sleep=None):
    """One commit per version of every file under it, via hub_commit."""
    from build_family7 import hub_add_ops, hub_commit
    for v, gs in entries.items():
        pairs = []
        for g, e in gs.items():
            for role, _ in FILES:
                if role in e:
                    local, rel = e[role]
                    pairs.append((f"{prefix_of(stem)}/{rel}", local))
        mb = sum(os.path.getsize(p) for _, p in pairs) / 1e6
        print(f"  uploading {v}: {len(pairs)} file(s), {mb:.1f} MB …",
              flush=True)
        hub_commit(api, repo, hub_add_ops(pairs),
                   f"E-083: family-7 climatology, version {v} ({stem})",
                   repo_type=REPO_TYPE, sleep=sleep)
        print(f"  committed {v}", flush=True)


# -------------------------------------------------------------------- cors ---
def cors_record(url, local_path, measure=None):
    """Measure CORS on `url` and check it serves the local file's head."""
    if measure is None:
        from publish_family7_index import measure_cors as measure
    r = measure(url)
    with open(local_path, "rb") as fh:
        head = fh.read(len(r["head"]))
    if r["status"] != 206 or not r.get("content_range"):
        raise SystemExit(f"{url}: the Hub answered {r['status']} to a ranged "
                         f"GET, not 206 — the layer's design is one range "
                         f"read per plane")
    if not r.get("access_control_allow_origin"):
        raise SystemExit(f"{url}: no access-control-allow-origin for "
                         f"{ORIGIN} — a browser could not read it")
    if r["head"] != head:
        raise SystemExit(f"{url}: the resolve URL served different bytes "
                         f"from the local file — it is not this file")
    return dict(origin=ORIGIN, status=r["status"],
                final_url_host=r["final_url"].split("/")[2],
                access_control_allow_origin=r["access_control_allow_origin"],
                access_control_expose_headers=r.get(
                    "access_control_expose_headers"),
                accept_ranges=r.get("accept_ranges"),
                content_range=r["content_range"],
                measured_file=url, measured_utc=now_utc())


# ------------------------------------------------------------------- index ---
def build_index(out, f7_path, *, cors=True, restore_verified=False,
                fixture_dir=None, measure=None):
    f7 = json.load(open(f7_path, encoding="utf-8"))
    stem = f7["stem"]
    versions, groups, entries = collect(out, f7)
    base = base_of(stem)
    files = {}
    for v in versions:
        files[v] = {}
        for g in groups:
            e = entries[v][g]
            st = e["_stats"]
            rec = {}
            for role, _ in FILES:
                if role not in e:
                    rec[role] = None
                    continue
                local, rel = e[role]
                blk = dict(url=base + rel, bytes=os.path.getsize(local),
                           sha256=sha256_file(local))
                if role == "clim_npy":
                    blk.update(npy_block(local, f7["groups"][g]))
                rec[role] = blk
            rec["n_train_bins"] = int(st["n_train_bins"])
            rec["n_bins"] = int(st["n_bins"])
            rec["static_chans"] = list(st.get("static_channels", []))
            rec["months_with_data"] = list(st.get("months_with_data", []))
            rec["tensor_sha256_verified"] = bool(
                st.get("tensor_sha256_verified"))
            files[v][g] = rec
    cors_rec = None
    if cors:
        v0, g0 = versions[0], groups[0]
        cors_rec = cors_record(files[v0][g0]["clim_npy"]["url"],
                               entries[v0][g0]["clim_npy"][0], measure)
    gblocks = {}
    for g in groups:
        src = f7["groups"][g]
        gblocks[g] = {k: copy.deepcopy(src[k]) for k in COPY_KEYS}
        gblocks[g]["tensor_file"] = src["file"]
        gblocks[g]["tensor_sha256"] = src["sha256"]
    st0 = entries[versions[0]][groups[0]]["_stats"]
    return dict(
        _source="ml/publish_family7_clim_index.py — do not hand-edit",
        generated_utc=now_utc(),
        fixture=bool(f7.get("fixture")) or bool(fixture_dir),
        fixture_dir=fixture_dir,
        stem=stem,
        tensor_index=os.path.relpath(os.path.abspath(f7_path), ROOT),
        repo=f"datasets/{REPO_ID}",
        base=base,
        layout=("clim.npy is [month, channel, lat, lon] float32, C order, "
                "month 0 = January, z-units: physical = z * norm[c][1] + "
                "norm[c][0]. Plane (m, c) starts at header_len + (m * C + c) "
                "* plane_bytes. NaN where the (month, cell) had no training "
                "sample."),
        versions=[version_block(v) for v in VERSIONS
                  if v["key"] in versions],
        default_version=versions[0],
        groups=gblocks,
        files=files,
        anomaly_transform=st0.get("anomaly_transform"),
        anomaly_transform_src_sha256=st0.get("anomaly_transform_src_sha256"),
        export_git_sha=st0.get("git_sha"),
        restore_verified=bool(restore_verified),
        cors_measured=cors_rec,
        plan=PLAN_URL,
    )


def write_json(path, obj):
    text = json.dumps(obj, sort_keys=True, separators=(",", ":"),
                      allow_nan=False) + "\n"
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(text)
    print(f"wrote {path} ({len(text) / 1e3:.1f} kB)", flush=True)


def main(argv=None, api=None, download=None, measure=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("cmd", choices=("index", "upload"))
    ap.add_argument("--out", required=True,
                    help="the export directory (<version>/<group>/…)")
    ap.add_argument("--f7-index", default=F7_INDEX)
    ap.add_argument("--index", default=INDEX)
    ap.add_argument("--no-cors", action="store_true")
    ap.add_argument("--local", action="store_true",
                    help="index: trust the local files, no Hub restore check "
                         "(recorded as restore_verified: false)")
    ap.add_argument("--fixture-dir", default=None,
                    help="repo-relative folder the fixture's files are "
                         "served from under MIRROR")
    ap.add_argument("--repo", default=REPO_ID)
    a = ap.parse_args(argv)
    if a.cmd == "upload" and a.local:
        ap.error("upload always restores from the Hub; --local is for index")

    f7 = json.load(open(a.f7_index, encoding="utf-8"))
    versions, groups, entries = collect(a.out, f7)
    print(f"{a.out}: versions {versions} · groups {groups} · stem "
          f"{f7['stem']}", flush=True)
    verified = False
    if not a.local:
        tok = None
        if a.cmd == "upload":
            tok = token()
            if api is None:
                from huggingface_hub import HfApi
                api = HfApi(token=tok)
            upload(entries, f7["stem"], api, repo=a.repo)
        if download is None:
            download = hub_download_fn(tok or os.environ.get("HF_TOKEN"),
                                       repo=a.repo)
        try:
            n = verify_restore(entries, f7["stem"], download)
        finally:
            tmp = getattr(download, "tmp", None)
            if tmp:
                shutil.rmtree(tmp, ignore_errors=True)
        print(f"  {n} file(s) restored from the Hub and sha256-matched",
              flush=True)
        verified = True
    index = build_index(a.out, a.f7_index, cors=not a.no_cors,
                        restore_verified=verified, fixture_dir=a.fixture_dir,
                        measure=measure)
    write_json(a.index, index)
    return index


if __name__ == "__main__":
    main()
    sys.exit(0)

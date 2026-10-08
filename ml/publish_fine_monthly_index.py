#!/usr/bin/env python3
"""E-089 · Publish the fine 1.gf grids' monthly sums (native tiles + exact
0.25° pooled layer) and write `data/gridded_monthly_fine_index.json`.

A SIBLING of `data/gridded_monthly_index.json` (E-088), not a block inside it:
the live Data tab reader (`src/f1data.js`) matches E-088's stores to registry
stores by `source_store` and would refuse a 0.25° pooled grid laid over a 4 km
store as a grid mismatch — so the fine index lives in its own file, with its
own `schema`, and nothing the live site reads changes.

  upload   (the box, per store, called by export_fine_monthly.py run): a
           manifest of every file (sha256, bytes; the pooled .npy headers
           parsed), the native month files committed a YEAR per commit, then
           ONE commit of the pooled files + tile_grid.json + stats.json +
           verify.json + manifest.json (stats.json last on the Hub = the
           store's done-marker). Then EVERY file is streamed back from the Hub
           and sha256-compared (ml/hf_mirror.py's rule) before the box frees
           its disk.
  index    (a hosted runner) for every store: reads monthly/manifest.json from
           the Hub, streams every file back, requires sha256 / size to match,
           measures CORS from https://blauewelt.org on the pooled sum and
           count and on one native month file, then writes the index
           (`restore_verified: true`). `--local --out DIR` builds it from
           local files (fixture, tests; `restore_verified: false`).

Credentials: HF_TOKEN in the environment, never argv.
"""
import argparse
import hashlib
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import export_fine_monthly as E                                  # noqa: E402
import export_gridded_monthly as X                               # noqa: E402
from family1 import sharded as sh                                # noqa: E402
from publish_family7_index import parse_npy_header               # noqa: E402
import publish_family7_monthly_index as P7                       # noqa: E402

SCHEMA = "gridded-monthly-fine/1"
REPO_ID = "chfrank/earth-tensors"
REPO_TYPE = "dataset"
ORIGIN = "https://blauewelt.org"
INDEX = os.path.join(ROOT, "data", "gridded_monthly_fine_index.json")
POOLED = (("sum", "pooled025/sum.npy", "<f4", 4),
          ("count", "pooled025/count.npy", "<u2", 2),
          ("m2", "pooled025/m2.npy", "<f4", 4))
JSONS = (("stats", "stats.json"), ("verify", "verify.json"),
         ("native_spec", "native/tile_grid.json"))
HEAD = 4096


def prefix_of(store):
    return f"tensors/{store}/monthly"


def base_of(store):
    return (f"https://huggingface.co/datasets/{REPO_ID}/resolve/main/"
            f"{prefix_of(store)}/")


def pooled_record(head, nbytes, role, st, where):
    want_dt, item = next((d, i) for r, _, d, i in POOLED if r == role)
    hl, shape, dtype, fortran = parse_npy_header(head)
    if fortran or dtype != want_dt:
        raise SystemExit(f"{where}: {dtype} fortran={fortran}, want {want_dt}")
    if shape != st["pooled"]["shape"]:
        raise SystemExit(f"{where}: shape {shape} vs {st['pooled']['shape']}")
    _, C, Y, Hp, Wp = shape
    plane = Hp * Wp * item
    if hl + 12 * C * Y * plane != nbytes:
        raise SystemExit(f"{where}: header + data != file size")
    return dict(header_len=int(hl), shape=shape, dtype=dtype, itemsize=item,
                fortran_order=False, axes=list(E.AXES), plane_bytes=plane,
                month_channel_bytes=Y * plane)


def native_files(st):
    out = []
    for y, m, *_ in st["native"]["months"]:
        out += [f"native/{E.mt.shard_relpath(y, m)}",
                f"native/{E.mt.index_relpath(y, m)}"]
    return out


def make_manifest(store, d, workers=8):
    st = json.load(open(os.path.join(d, "stats.json"), encoding="utf-8"))
    if st["store"] != store:
        raise SystemExit(f"{d}/stats.json is for {st['store']}, not {store}")
    nat = native_files(st)
    on_disk = sorted(os.path.relpath(os.path.join(dp, f), d)
                     for dp, _, fs in os.walk(os.path.join(d, "native"))
                     for f in fs if f != "tile_grid.json")
    if sorted(nat) != on_disk:
        raise SystemExit(f"{d}: native files on disk differ from stats.json's "
                         f"month table ({len(on_disk)} vs {len(nat)})")
    names = [r for _, r, _, _ in POOLED] + [r for _, r in JSONS] + nat
    for r in names:
        if not os.path.exists(os.path.join(d, r)):
            raise SystemExit(f"{d}: missing {r}")
    shas = sh.sha256_block(d, names, workers=workers)
    files = {}
    for role, rel, _, _ in POOLED:
        p = os.path.join(d, rel)
        with open(p, "rb") as fh:
            head = fh.read(HEAD)
        n = os.path.getsize(p)
        rec = dict(rel=rel, bytes=n, sha256=shas[rel])
        rec.update(pooled_record(head, n, role, st, p))
        files[role] = rec
    for role, rel in JSONS:
        files[role] = dict(rel=rel, bytes=os.path.getsize(os.path.join(d, rel)),
                           sha256=shas[rel])
    native = [dict(rel=r, bytes=os.path.getsize(os.path.join(d, r)),
                   sha256=shas[r]) for r in nat]
    return dict(_source="ml/publish_fine_monthly_index.py manifest",
                schema=SCHEMA, store=store, prefix=prefix_of(store),
                files=files, native=native,
                bytes=int(sum(f["bytes"] for f in files.values()) +
                          sum(f["bytes"] for f in native)),
                generated_utc=X.now_utc())


def all_files(man):
    return list(man["files"].values()) + list(man["native"])


def readback(store, man, session=None, workers=8, fetch=P7.fetch_bytes,
             base=None):
    """Every manifest file streamed back from the Hub, sha256 and size
    matched. Returns {rel: head bytes} of the pooled files (for CORS)."""
    base = base or base_of(store)
    if session is None:
        import requests
        session = requests.Session()
        session.mount("https://", requests.adapters.HTTPAdapter(
            pool_maxsize=max(16, workers * 4)))
    heads = {}
    t0 = time.time()

    def one(rec):
        url = base + rec["rel"]
        if rec["bytes"] <= 64 << 20:
            data, cr = fetch(session, url, 0, rec["bytes"] - 1) if \
                rec["bytes"] else (b"", f"bytes */0")
            got, head = hashlib.sha256(data).hexdigest(), data[:HEAD]
            total = int(cr.rsplit("/", 1)[1]) if rec["bytes"] else 0
        else:
            got, head, total = P7.hub_stream_sha256(
                session, url, rec["bytes"], workers=4, fetch=fetch)
        if total is not None and total != rec["bytes"]:
            raise SystemExit(f"REFUSING: {store}/{rec['rel']} is {total} bytes "
                             f"on the Hub, {rec['bytes']} written")
        if got != rec["sha256"]:
            raise SystemExit(f"REFUSING: {store}/{rec['rel']} uploaded as "
                             f"{rec['sha256']} but downloaded back as {got}")
        return rec["rel"], head

    recs = all_files(man)
    with ThreadPoolExecutor(max_workers=max(1, workers)) as ex:
        for rel, head in ex.map(one, recs):
            heads[rel] = head
    s = time.time() - t0
    gb = man["bytes"] / 1e9
    print(f"  read back ✓ {store}: {len(recs)} files, {gb:.2f} GB in {s:.0f} s "
          f"({gb * 1e3 / max(s, 1e-9):.0f} MB/s), every sha256 equal",
          flush=True)
    return heads


def upload_store(store, d, api=None, readback_fn=None):
    """Manifest, the native files a year per commit, then the pooled + json
    commit; then the read-back. Returns {bytes, readback_ok, manifest}."""
    from build_family7 import hub_add_ops, hub_commit
    st = json.load(open(os.path.join(d, "stats.json"), encoding="utf-8"))
    if st.get("trial_months"):
        raise SystemExit(f"REFUSING to publish {store}: a trial export of "
                         f"{st['trial_months']} only")
    man = make_manifest(store, d)
    X.write_json(os.path.join(d, "manifest.json"), man)
    if api is None:
        tok = os.environ.get("HF_TOKEN", "").strip()
        if not tok:
            raise SystemExit("no HF_TOKEN in the environment")
        from huggingface_hub import HfApi
        api = HfApi(token=tok)
    pre = prefix_of(store)
    by_year = {}
    for r in man["native"]:
        by_year.setdefault(r["rel"].split("/")[1], []).append(r["rel"])
    t0 = time.time()
    print(f"  uploading {store}: {len(man['native'])} native files in "
          f"{len(by_year)} yearly commits + 1, {man['bytes'] / 1e9:.2f} GB …",
          flush=True)
    for yy in sorted(by_year):
        pairs = [(f"{pre}/{r}", os.path.join(d, r)) for r in by_year[yy]]
        hub_commit(api, REPO_ID, hub_add_ops(pairs),
                   f"E-089: native monthly sums of {store}, {yy}",
                   repo_type=REPO_TYPE)
    last = [r["rel"] for r in man["files"].values()] + ["manifest.json"]
    last = [r for r in last if r != "stats.json"] + ["stats.json"]
    hub_commit(api, REPO_ID,
               hub_add_ops([(f"{pre}/{r}", os.path.join(d, r)) for r in last]),
               f"E-089: pooled 0.25° monthly sums + manifest of {store}",
               repo_type=REPO_TYPE)
    s = time.time() - t0
    print(f"  committed {store} in {s:.0f} s "
          f"({man['bytes'] / 1e6 / max(s, 1e-9):.0f} MB/s)", flush=True)
    (readback_fn or readback)(store, man)
    return dict(bytes=man["bytes"], readback_ok=True, manifest=man)


def _get_whole(session, url, fetch):
    _, cr = fetch(session, url, 0, 0)
    total = int(cr.rsplit("/", 1)[1])
    return fetch(session, url, 0, total - 1)


def restore(store, session, workers=8, fetch=P7.fetch_bytes):
    """manifest from the Hub, every file streamed back and matched; returns
    (manifest, {role: parsed json}, heads)."""
    base = base_of(store)
    raw, _ = _get_whole(session, base + "manifest.json", fetch)
    man = json.loads(raw)
    if man["store"] != store or man.get("schema") != SCHEMA:
        raise SystemExit(f"{store}: the Hub manifest names {man['store']} / "
                         f"{man.get('schema')}")
    heads = readback(store, man, session=session, workers=workers,
                     fetch=fetch, base=base)
    jsons = {}
    for role, rel in JSONS:
        data, _ = fetch(session, base + rel, 0, man["files"][role]["bytes"] - 1)
        if hashlib.sha256(data).hexdigest() != man["files"][role]["sha256"]:
            raise SystemExit(f"{store}/{rel}: changed during the restore")
        jsons[role] = json.loads(data.decode("utf-8"))
    return man, jsons, heads


def store_block(store, man, st, ver, base):
    files = {}
    for role, _, _, _ in POOLED:
        rec = man["files"][role]
        r = dict(url=base + rec["rel"], bytes=rec["bytes"],
                 sha256=rec["sha256"])
        for k in ("header_len", "shape", "dtype", "itemsize", "fortran_order",
                  "axes", "plane_bytes", "month_channel_bytes"):
            r[k] = rec[k]
        files[role] = r
    nat_by = {r["rel"]: r for r in man["native"]}
    months = []
    for y, m, frames, tiles, nbytes in st["native"]["months"]:
        sr = nat_by[f"native/{E.mt.shard_relpath(y, m)}"]
        ir = nat_by[f"native/{E.mt.index_relpath(y, m)}"]
        if sr["bytes"] != nbytes:
            raise SystemExit(f"{store} {y}-{m}: shard bytes disagree")
        months.append([y, m, frames, tiles, sr["bytes"], sr["sha256"],
                       ir["bytes"], ir["sha256"]])
    ns = man["files"]["native_spec"]
    fam, name = store.split("/", 1)
    nv = st["native"]
    return dict(
        family=fam, name=name, title=st.get("title"),
        licence=st.get("licence"), chans=st["chans"], channels=st["channels"],
        units=st["units"], value_offset=st["value_offset"],
        source_channels=st["source_channels"],
        frame_seconds=st["frame_seconds"], years=st["years"],
        year_first=st["year_first"], year_last=st["year_last"],
        n_years=st["n_years"], frames_present=st["frames_present"],
        frames_possible=st["frames_possible"], max_count=st["max_count"],
        record_first=st["record_first"], record_last=st["record_last"],
        source_store=f"tensors/{store}",
        source_sha256_block_digest=st["source_sha256_block_digest"],
        native=dict(
            base=base + "native/",
            tile_grid=dict(url=base + ns["rel"], bytes=ns["bytes"],
                           sha256=ns["sha256"]),
            H=nv["H"], W=nv["W"], C=nv["C"], tile=nv["tile"],
            n_tiles_y=nv["n_tiles_y"], n_tiles_x=nv["n_tiles_x"],
            parts=[dict(name=n, dtype=d.str, itemsize=d.itemsize,
                        part_bytes=nv["tile"] ** 2 * nv["C"] * d.itemsize)
                   for n, d in E.mt.PARTS],
            index_shape=[nv["n_tiles_y"], nv["n_tiles_x"], 3, 2],
            index_dtype="<i8",
            index_header_bytes=sh.index_header_bytes(
                [nv["n_tiles_y"], nv["n_tiles_x"], 3, 2]),
            shard_path="<yyyy>/m_<yyyy>-<mm>.zst",
            index_path="<yyyy>/m_<yyyy>-<mm>.idx.npy",
            months_columns=["year", "month", "frames", "tiles_stored",
                            "shard_bytes", "shard_sha256", "index_bytes",
                            "index_sha256"],
            months=months,
            bytes_total=int(sum(r["bytes"] for r in man["native"]))),
        pooled=dict(grid=st["pooled"]["grid"],
                    count_note=st["pooled"]["count_note"], **files),
        stats=dict(url=base + "stats.json",
                   bytes=man["files"]["stats"]["bytes"],
                   sha256=man["files"]["stats"]["sha256"]),
        verify=dict(url=base + "verify.json",
                    bytes=man["files"]["verify"]["bytes"],
                    sha256=man["files"]["verify"]["sha256"]),
        falsifier=dict(ok=ver["ok"], native=ver["native"],
                       pooled=ver["pooled"],
                       blocksum=ver["blocksum_summary"],
                       months=st["falsifier_months"],
                       boxes=st["falsifier_boxes"]))


def build_index(blocks, *, cors=None, restore_verified=False,
                restore_where=None, fixture_dir=None):
    return dict(
        _source="ml/publish_fine_monthly_index.py — do not hand-edit",
        schema=SCHEMA, generated_utc=X.now_utc(), fixture=bool(fixture_dir),
        fixture_dir=fixture_dir, repo=f"datasets/{REPO_ID}",
        base="https://huggingface.co/datasets/chfrank/earth-tensors/resolve/"
             "main/tensors/",
        plan=E.PLAN_URL, sibling_of="data/gridded_monthly_index.json (E-088)",
        axes=list(E.AXES), month_rule=E.MONTH_RULE, combine=E.COMBINE,
        native_layout=E.NATIVE_LAYOUT, pooled_layout=E.POOLED_LAYOUT,
        pool_rule=E.POOL_RULE,
        offset_formula=("pooled: offset(m, c, y) = header_len + ((m * C + c) "
                        "* n_years + (y - year_first)) * plane_bytes; m = 0 "
                        "is January (E-088's formula). native: the index of "
                        "(y, m) is <base>/<yyyy>/m_<yyyy>-<mm>.idx.npy; entry "
                        "(ty, tx, part) is the 16 bytes at "
                        "index_header_bytes + 16 * ((ty * n_tiles_x + tx) * 3 "
                        "+ part), part 0 sum, 1 count, 2 m2"),
        std_rel_tol=E.STD_REL_TOL,
        std_tolerance=("|std − numpy nanstd| ≤ std_rel_tol × (|mean| + std) "
                       "per cell, measured by the falsifier"),
        mean_bound=("native: |Σsum/Σcount − numpy nanmean| ≤ Σ ½ulp32(sum_k) "
                    "/ Σcount + 4·eps64·|mean| (E-088's bound); pooled: the "
                    "same plus Σ ½ulp32 of the native sums pooled into the "
                    "cell"),
        stores=blocks, restore_verified=bool(restore_verified),
        restore_where=restore_where, cors_measured=cors)


def main(argv=None, fetch=None, measure=None, session=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("cmd", choices=("index",))
    ap.add_argument("--stores", default=",".join(E.FINE))
    ap.add_argument("--out", default="")
    ap.add_argument("--local", action="store_true")
    ap.add_argument("--index", default=INDEX)
    ap.add_argument("--no-cors", action="store_true")
    ap.add_argument("--origin", default=ORIGIN)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--restore-where", default="GitHub-hosted runner")
    ap.add_argument("--fixture-dir", default=None)
    a = ap.parse_args(argv)
    stores = [s for s in a.stores.split(",") if s]
    blocks, heads0 = {}, None
    if a.local:
        for s in stores:
            d = os.path.join(a.out, X.store_name(s))
            man = make_manifest(s, d)
            st = json.load(open(os.path.join(d, "stats.json")))
            ver = json.load(open(os.path.join(d, "verify.json")))
            if not ver.get("ok"):
                raise SystemExit(f"REFUSING: {s}'s verify report failed")
            blocks[s] = store_block(s, man, st, ver, base_of(s))
            if heads0 is None:
                heads0 = (s, man, {r["rel"]: open(os.path.join(d, r["rel"]),
                                                  "rb").read(HEAD)
                                   for r in all_files(man)})
        verified, where = False, "local (not restored)"
    else:
        if session is None:
            import requests
            session = requests.Session()
            session.mount("https://", requests.adapters.HTTPAdapter(
                pool_maxsize=max(16, a.workers * 4)))
        kw = {"fetch": fetch} if fetch else {}
        for s in stores:
            man, jsons, heads = restore(s, session, a.workers, **kw)
            if not jsons["verify"].get("ok"):
                raise SystemExit(f"REFUSING: {s}'s verify report failed")
            blocks[s] = store_block(s, man, jsons["stats"], jsons["verify"],
                                    base_of(s))
            if heads0 is None:
                heads0 = (s, man, heads)
        verified, where = True, a.restore_where
    cors = None
    if not a.no_cors:
        s0, man0, hd = heads0
        b0 = blocks[s0]
        nat0 = man0["native"][0]["rel"]
        cors = {r: P7.cors_record(b0["pooled"][r]["url"],
                                  hd[man0["files"][r]["rel"]],
                                  origin=a.origin, measure=measure)
                for r in ("sum", "count")}
        cors["native_shard"] = P7.cors_record(base_of(s0) + nat0, hd[nat0],
                                              origin=a.origin, measure=measure)
    idx = build_index(blocks, cors=cors, restore_verified=verified,
                      restore_where=where, fixture_dir=a.fixture_dir)
    P7.write_json(a.index, idx)
    return idx


if __name__ == "__main__":
    main()
    sys.exit(0)

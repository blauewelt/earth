#!/usr/bin/env python3
"""E-088 §6 · Publish the gridded stores' per-year monthly sums, counts and M2,
and write `data/gridded_monthly_index.json`.

The index is the Data tab's address book for "a long-period mean (and standard
deviation) from a few range reads" over the sharded gridded stores — the
sibling of `data/family7_monthly_index.json` (E-086, the five-day tensor),
with the SAME key style for every file (`url`, `bytes`, `sha256`,
`header_len`, `shape`, `dtype`, `itemsize`, `axes`, `plane_bytes`,
`month_channel_bytes`) so one reader serves both. Per store it adds the year
axis, frames present and possible per (year, month), channel metadata
(units, ranges, pressure levels), the grid (pixel positions and how the
store registers them), licence and attribution, and the falsifier's numbers.
Never hand-edited.

  upload   (the box, per store, called by export_gridded_monthly.py run):
           a manifest of every file (sha256, bytes, parsed .npy header), then
           ONE Hub commit of tensors/<store>/monthly/{sum,count,m2}.npy +
           stats.json + verify.json + manifest.json.
  index    (a hosted runner) for every store: reads monthly/manifest.json
           from the Hub, STREAMS every file back (parallel ranged GETs hashed
           in order — ml/publish_family7_monthly_index.hub_stream_sha256),
           requires sha256 / size / header to match, measures CORS from
           https://blauewelt.org, then writes the index
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

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import export_gridded_monthly as X                               # noqa: E402
from publish_family7_index import parse_npy_header, sha256_file   # noqa: E402
import publish_family7_monthly_index as P7                       # noqa: E402

REPO_ID = "chfrank/earth-tensors"
REPO_TYPE = "dataset"
ORIGIN = "https://blauewelt.org"
INDEX = os.path.join(ROOT, "data", "gridded_monthly_index.json")
NPY = (("sum", "sum.npy", "<f4", 4), ("count", "count.npy", "|u1", 1),
       ("m2", "m2.npy", "<f4", 4))
JSONS = (("stats", "stats.json"), ("verify", "verify.json"))
HEAD = 4096


def prefix_of(store):
    return f"tensors/{store}/monthly"


def base_of(store):
    return (f"https://huggingface.co/datasets/{REPO_ID}/resolve/main/"
            f"{prefix_of(store)}/")


def npy_record(head, nbytes, role, st, where):
    want_dt, item = next((d, i) for r, _, d, i in NPY if r == role)
    hl, shape, dtype, fortran = parse_npy_header(head)
    C, Y = len(st["chans"]), st["n_years"]
    H, W = st["grid"]["H"], st["grid"]["W"]
    if fortran or dtype != want_dt:
        raise SystemExit(f"{where}: {dtype} fortran={fortran}, want {want_dt}")
    if shape != [12, C, Y, H, W] or shape != st["shape"]:
        raise SystemExit(f"{where}: shape {shape} vs [12, {C}, {Y}, {H}, {W}]")
    plane = H * W * item
    if hl + 12 * C * Y * plane != nbytes:
        raise SystemExit(f"{where}: header + data != file size")
    return dict(header_len=int(hl), shape=shape, dtype=dtype, itemsize=item,
                fortran_order=False, axes=list(X.AXES), plane_bytes=plane,
                month_channel_bytes=Y * plane)


def make_manifest(store, d):
    st = json.load(open(os.path.join(d, "stats.json"), encoding="utf-8"))
    if st["store"] != store:
        raise SystemExit(f"{d}/stats.json is for {st['store']}, not {store}")
    files = {}
    for role, name, _, _ in NPY:
        p = os.path.join(d, name)
        with open(p, "rb") as fh:
            head = fh.read(HEAD)
        n = os.path.getsize(p)
        rec = dict(rel=name, bytes=n, sha256=sha256_file(p))
        rec.update(npy_record(head, n, role, st, p))
        files[role] = rec
    for role, name in JSONS:
        p = os.path.join(d, name)
        if not os.path.exists(p):
            raise SystemExit(f"{d}: missing {name}")
        files[role] = dict(rel=name, bytes=os.path.getsize(p),
                           sha256=sha256_file(p))
    return dict(_source="ml/publish_gridded_monthly_index.py manifest",
                store=store, prefix=prefix_of(store), files=files,
                generated_utc=X.now_utc())


def upload_store(store, d, api=None):
    """Manifest + one commit of the store's monthly files (the box)."""
    from build_family7 import hub_add_ops, hub_commit
    man = make_manifest(store, d)
    X.write_json(os.path.join(d, "manifest.json"), man)
    if api is None:
        tok = os.environ.get("HF_TOKEN", "").strip()
        if not tok:
            raise SystemExit("no HF_TOKEN in the environment")
        from huggingface_hub import HfApi
        api = HfApi(token=tok)
    pairs = [(f"{prefix_of(store)}/{r['rel']}", os.path.join(d, r["rel"]))
             for r in man["files"].values()]
    pairs.append((f"{prefix_of(store)}/manifest.json",
                  os.path.join(d, "manifest.json")))
    gb = sum(os.path.getsize(p) for _, p in pairs) / 1e9
    t0 = time.time()
    print(f"  uploading {store}: {len(pairs)} files, {gb:.2f} GB …",
          flush=True)
    hub_commit(api, REPO_ID, hub_add_ops(pairs),
               f"E-088: per-year monthly sums/counts/m2 of {store}",
               repo_type=REPO_TYPE)
    s = time.time() - t0
    print(f"  committed {store} in {s:.0f} s ({gb * 1e3 / max(s, 1e-9):.0f} "
          f"MB/s)", flush=True)
    return man


def restore(store, session, workers=8, fetch=P7.fetch_bytes):
    """manifest from the Hub, every file streamed back and matched."""
    base = base_of(store)
    raw, _ = _get_whole(session, base + "manifest.json", fetch)
    man = json.loads(raw)
    if man["store"] != store:
        raise SystemExit(f"{store}: the Hub manifest names {man['store']}")
    heads, jsons = {}, {}
    for role, rec in man["files"].items():
        url = base + rec["rel"]
        if rec["rel"].endswith(".json"):
            data, _ = fetch(session, url, 0, rec["bytes"] - 1)
            got, head, total = hashlib.sha256(data).hexdigest(), data[:HEAD], \
                rec["bytes"]
            jsons[role] = json.loads(data.decode("utf-8"))
        else:
            got, head, total = P7.hub_stream_sha256(
                session, url, rec["bytes"], workers=workers, fetch=fetch)
        if total is not None and total != rec["bytes"]:
            raise SystemExit(f"REFUSING: {store}/{rec['rel']} is {total} "
                             f"bytes on the Hub, {rec['bytes']} written")
        if got != rec["sha256"]:
            raise SystemExit(f"REFUSING to write the index: {store}/"
                             f"{rec['rel']} uploaded as {rec['sha256']} but "
                             f"downloaded back as {got}")
        heads[role] = head
        print(f"  restored ✓ {store}/{rec['rel']} ({rec['bytes'] / 1e6:.1f} "
              f"MB)", flush=True)
    return man, jsons, heads


def _get_whole(session, url, fetch):
    """A small file by one ranged read of its whole length (size from a
    one-byte probe — the Hub answers 206 with the total)."""
    _, cr = fetch(session, url, 0, 0)
    total = int(cr.rsplit("/", 1)[1])
    return fetch(session, url, 0, total - 1)


def store_block(store, man, st, ver, base):
    files = {}
    for role, rec in man["files"].items():
        r = dict(url=base + rec["rel"], bytes=rec["bytes"],
                 sha256=rec["sha256"])
        for k in ("header_len", "shape", "dtype", "itemsize", "fortran_order",
                  "axes", "plane_bytes", "month_channel_bytes"):
            if k in rec:
                r[k] = rec[k]
        files[role] = r
    fam, name = store.split("/", 1)
    return dict(
        family=fam, name=name, title=st.get("title"),
        licence=st.get("licence"), chans=st["chans"],
        channels=st["channels"], units=st["units"],
        levels_hpa=st.get("levels_hpa"), channel_axis=st.get("channel_axis"),
        frame_seconds=st["frame_seconds"], grid=st["grid"],
        years=st["years"], year_first=st["year_first"],
        year_last=st["year_last"], n_years=st["n_years"],
        frames_present=st["frames_present"],
        frames_possible=st["frames_possible"], max_count=st["max_count"],
        record_first=st["record_first"], record_last=st["record_last"],
        source_store=f"tensors/{store}", source_sha256_block_digest=st[
            "source_sha256_block_digest"],
        falsifier=dict(ok=ver["ok"], n_planes=ver["n_planes"],
                       months=st["falsifier_months"],
                       period_years=st.get("falsifier_period_years"),
                       period_box=st.get("falsifier_period_box"),
                       max_abs_mean=ver["max_abs_mean"],
                       max_mean_over_bound=ver["max_mean_over_bound"],
                       max_abs_std=ver["max_abs_std"],
                       max_std_over_tol=ver["max_std_over_tol"]),
        **files)


def build_index(blocks, *, cors=None, restore_verified=False,
                restore_where=None, fixture_dir=None):
    return dict(
        _source="ml/publish_gridded_monthly_index.py — do not hand-edit",
        generated_utc=X.now_utc(), fixture=bool(fixture_dir),
        fixture_dir=fixture_dir, repo=f"datasets/{REPO_ID}",
        base="https://huggingface.co/datasets/chfrank/earth-tensors/resolve/"
             "main/tensors/",
        plan=X.PLAN_URL, axes=list(X.AXES), layout=X.LAYOUT,
        month_rule=X.MONTH_RULE,
        offset_formula=("offset(m, c, y) = header_len + ((m * C + c) * "
                        "n_years + (y - year_first)) * plane_bytes; m = 0 is "
                        "January; years y_a..y_b of one (m, c) are the single "
                        "range [offset(m, c, y_a), offset(m, c, y_b) + "
                        "plane_bytes)"),
        combine=X.COMBINE, std_rel_tol=X.STD_REL_TOL,
        std_tolerance=("|std − numpy nanstd| ≤ std_rel_tol × (|mean| + std) "
                       "per cell, measured by the falsifier"),
        mean_bound=("|Σsum/Σcount − numpy nanmean| ≤ Σ ½ulp32(sum_k) / "
                    "Σcount + 4·eps64·|mean| per cell (rigorous: the float64 "
                    "sums of float16 values are exact)"),
        stores=blocks, restore_verified=bool(restore_verified),
        restore_where=restore_where, cors_measured=cors)


def main(argv=None, fetch=None, measure=None, session=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("cmd", choices=("index",))
    ap.add_argument("--stores", default=",".join(X.PHASE1))
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
                heads0 = (s, {r: open(os.path.join(d, man["files"][r]["rel"]),
                                      "rb").read(HEAD) for r in ("sum",
                                                                 "count")})
        verified, where = False, "local (not restored)"
    else:
        if session is None:
            import requests
            session = requests.Session()
            session.mount("https://", requests.adapters.HTTPAdapter(
                pool_maxsize=max(8, a.workers * 2)))
        kw = {"fetch": fetch} if fetch else {}
        for s in stores:
            man, jsons, heads = restore(s, session, a.workers, **kw)
            if not jsons["verify"].get("ok"):
                raise SystemExit(f"REFUSING: {s}'s verify report failed")
            blocks[s] = store_block(s, man, jsons["stats"], jsons["verify"],
                                    base_of(s))
            if heads0 is None:
                heads0 = (s, heads)
        verified, where = True, a.restore_where
    cors = None
    if not a.no_cors:
        s0, hd = heads0
        cors = {r: P7.cors_record(blocks[s0][r]["url"], hd[r],
                                  origin=a.origin, measure=measure)
                for r in ("sum", "count")}
    idx = build_index(blocks, cors=cors, restore_verified=verified,
                      restore_where=where, fixture_dir=a.fixture_dir)
    P7.write_json(a.index, idx)
    return idx


if __name__ == "__main__":
    main()
    sys.exit(0)

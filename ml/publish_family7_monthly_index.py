#!/usr/bin/env python3
"""E-086 §4 · Publish family 7's per-year monthly sums and counts, and write
`data/family7_monthly_index.json`.

The index is the address book the site reads before it touches the Hub — the
sibling of `data/family7_clim_index.json` (`ml/publish_family7_clim_index.py`)
and never hand-edited. Per group it says where `sum.npy`, `count.npy` and
`stats.json` live under `tensors/<stem>/monthly/<group>/` on
`chfrank/earth-tensors`, each file's bytes and sha256, each `.npy`'s parsed
header length, shape, dtype, axis order and the bytes of one plane (so one
(month, channel, year) plane, or one run of years, is one `Range:` read); the
year axis and the bins-per-(year, month) table; and — COPIED from
`data/family7_index.json`, never retyped — the grid, channel names, labels,
units, ramps, signs and `norm`. It refuses a family-7 index whose `stem` is not
the tensor the sums were computed from.

Three subcommands (the house rule — ml/hf_mirror.py, ml/CLAUDE.md §0.2: an
upload that returned 200 is not evidence the bytes are retrievable — is split
across two machines on purpose, E-083's lesson: the box's own download-back of
2.5 GB from Japan never finished; a hosted runner restores 27 GB in minutes):

  upload   (the box) writes <out>/manifest.json — every file's bytes, sha256
           and parsed header — then commits each group's three files, one
           commit per group, through build_family7.hub_commit's retry ladder.
           It writes NO index.
  index    (a hosted runner, or anywhere) reads the manifest, STREAMS every
           file back from the Hub with parallel ranged GETs hashed in order
           (no disk: the 15 GB g025 sum never lands), requires the sha256,
           size and header to match the manifest, reads each stats.json back,
           measures CORS, and only then writes the index
           (`restore_verified: true`). `--local --out DIR` builds it from
           local files instead (the fixture and the tests; recorded as
           `restore_verified: false`).
  manifest just the manifest.

CORS is MEASURED, never assumed: an anonymous ranged GET with
`Origin: https://blauewelt.org` (the site's domain) must answer 206 with an
access-control-allow-origin, on the first group's sum.npy AND count.npy, and
serve the bytes the manifest describes. `--no-cors` (fixture) records null.

Credentials: `HF_TOKEN` in the environment. Never argv.

    python3 ml/publish_family7_monthly_index.py upload --out /data/monthly
    python3 ml/publish_family7_monthly_index.py index \\
        --manifest manifest.json --verify verify.json
"""
import argparse
import copy
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

from export_family7_monthly import (AXES, LAYOUT, MONTH_RULE, PLAN_URL,  # noqa
                                    plane_offset)
from export_family7_clim import VERSIONS, version_block           # noqa: E402
from publish_family7_index import parse_npy_header, sha256_file   # noqa: E402

REPO_ID = "chfrank/earth-tensors"
REPO_TYPE = "dataset"
ORIGIN = "https://blauewelt.org"
F7_INDEX = os.path.join(ROOT, "data", "family7_index.json")
INDEX = os.path.join(ROOT, "data", "family7_monthly_index.json")
FILES = (("sum", "sum.npy"), ("count", "count.npy"), ("stats", "stats.json"))
DTYPES = {"sum": ("<f4", 4), "count": ("|u1", 1)}
COPY_KEYS = ("grid", "chans", "labels", "units", "ramp", "sign", "norm")
HEAD_BYTES = 4096


def prefix_of(stem):
    return f"tensors/{stem}/monthly"


def base_of(stem):
    return (f"https://huggingface.co/datasets/{REPO_ID}/resolve/main/"
            f"{prefix_of(stem)}/")


def now_utc():
    import datetime as _dt
    return _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def token():
    t = os.environ.get("HF_TOKEN", "").strip()
    if not t:
        raise SystemExit("no HF_TOKEN in the environment — the token travels "
                         "by env only, never argv")
    return t


def auth_headers():
    t = os.environ.get("HF_TOKEN", "").strip()
    return {"Authorization": f"Bearer {t}"} if t else {}


# ----------------------------------------------------------------- manifest --
def npy_record(head, nbytes, role, st, blk, where):
    """The parsed header of sum.npy / count.npy, checked against the group's
    stats.json and the family-7 grid; one plane = one (m, c, y)."""
    header_len, shape, dtype, fortran = parse_npy_header(head)
    want_dt, itemsize = DTYPES[role]
    C, ny, nx = len(blk["chans"]), blk["grid"]["ny"], blk["grid"]["nx"]
    Y = int(st["n_years"])
    if fortran:
        raise SystemExit(f"{where}: Fortran order — the plane arithmetic "
                         f"assumes C order")
    if dtype != want_dt:
        raise SystemExit(f"{where}: dtype {dtype}, expected {want_dt}")
    if shape != [12, C, Y, ny, nx] or shape != list(st["shape"]):
        raise SystemExit(f"{where}: shape {shape}, expected [12, {C}, {Y}, "
                         f"{ny}, {nx}] (month, channel, year, lat, lon)")
    plane = ny * nx * itemsize
    if header_len + 12 * C * Y * plane != nbytes:
        raise SystemExit(f"{where}: header says {header_len + 12 * C * Y * plane}"
                         f" bytes, the file is {nbytes} — the range "
                         f"arithmetic would be wrong")
    return dict(header_len=int(header_len), shape=[int(x) for x in shape],
                dtype=dtype, itemsize=itemsize, fortran_order=False,
                axes=list(AXES), plane_bytes=int(plane),
                year_run_bytes_per_year=int(plane),
                month_channel_bytes=int(Y * plane))


def make_manifest(out, f7):
    """Scan <out>/<group>/ in the family-7 index's group order."""
    stem = f7["stem"]
    groups = [g for g in f7["groups"]
              if os.path.exists(os.path.join(out, g, "stats.json"))]
    stray = [x for x in sorted(os.listdir(out))
             if os.path.isdir(os.path.join(out, x)) and x not in f7["groups"]]
    if stray:
        raise SystemExit(f"{out}: folder(s) {stray} are not family-7 groups")
    if not groups:
        raise SystemExit(f"{out}: no <group>/stats.json — run "
                         f"ml/export_family7_monthly.py export first")
    files = {}
    for g in groups:
        where = os.path.join(out, g)
        st = json.load(open(os.path.join(where, "stats.json"),
                            encoding="utf-8"))
        check_stats(st, f7, g, where)
        files[g] = {}
        for role, name in FILES:
            p = os.path.join(where, name)
            if not os.path.exists(p):
                raise SystemExit(f"{where}: missing {name}")
            n = os.path.getsize(p)
            print(f"  manifest: sha256 {g}/{name} ({n / 1e9:.2f} GB) …",
                  flush=True)
            rec = dict(rel=f"{g}/{name}", bytes=int(n), sha256=sha256_file(p))
            if role != "stats":
                with open(p, "rb") as fh:
                    head = fh.read(HEAD_BYTES)
                rec.update(npy_record(head, n, role, st, f7["groups"][g], p))
            files[g][role] = rec
    return dict(_source="ml/publish_family7_monthly_index.py manifest",
                stem=stem, prefix=prefix_of(stem), groups=groups, files=files,
                generated_utc=now_utc())


def check_stats(st, f7, g, where):
    if st.get("tensor_stem") != f7["stem"]:
        raise SystemExit(
            f"REFUSING: {where}/stats.json was computed from "
            f"{st.get('tensor_stem')!r} and the family-7 index describes "
            f"{f7['stem']!r}. The monthly index copies its channel metadata "
            f"and norms from the family-7 index, so the two must name the "
            f"same tensor.")
    blk = f7["groups"][g]
    if st["chans"] != blk["chans"] or st["norm"] != [
            [float(a), float(b)] for a, b in blk["norm"]]:
        raise SystemExit(f"{where}/stats.json: channels/norm differ from the "
                         f"family-7 index's {g} block")
    if st.get("tensor_sha256") != blk["sha256"]:
        raise SystemExit(f"{where}/stats.json: computed from {g} sha256 "
                         f"{st.get('tensor_sha256')}, the family-7 index says "
                         f"{blk['sha256']}")


# ------------------------------------------------------------------- upload --
def upload(out, manifest, api, repo=REPO_ID, sleep=None):
    """One commit per group of its three files, via hub_commit."""
    from build_family7 import hub_add_ops, hub_commit
    for g in manifest["groups"]:
        pairs = [(f"{manifest['prefix']}/{manifest['files'][g][r]['rel']}",
                  os.path.join(out, manifest["files"][g][r]["rel"]))
                 for r, _ in FILES]
        gb = sum(os.path.getsize(p) for _, p in pairs) / 1e9
        t0 = time.time()
        print(f"  uploading {g}: {len(pairs)} file(s), {gb:.2f} GB …",
              flush=True)
        hub_commit(api, repo, hub_add_ops(pairs),
                   f"E-086: family-7 per-year monthly sums/counts, group {g} "
                   f"({manifest['stem']})", repo_type=REPO_TYPE, sleep=sleep)
        s = time.time() - t0
        print(f"  committed {g} in {s:.0f} s ({gb * 1e3 / max(s, 1e-9):.1f} "
              f"MB/s)", flush=True)


# ------------------------------------------------------------------ restore --
def fetch_bytes(session, url, a, b, tries=6, sleep=time.sleep, timeout=180):
    """bytes a..b of `url`; must answer 206 with that exact Content-Range."""
    want = b - a + 1
    last = None
    for attempt in range(tries):
        try:
            r = session.get(url, headers={**auth_headers(),
                                          "Range": f"bytes={a}-{b}"},
                            stream=True, timeout=timeout, allow_redirects=True)
            try:
                if r.status_code in (429, 500, 502, 503, 504):
                    raise ConnectionError(f"HTTP {r.status_code}")
                if r.status_code != 206:
                    raise SystemExit(f"{url} bytes {a}-{b}: HTTP "
                                     f"{r.status_code}, not 206")
                cr = r.headers.get("Content-Range", "")
                if not cr.startswith(f"bytes {a}-{b}/"):
                    raise SystemExit(f"{url}: asked for bytes {a}-{b}, got "
                                     f"Content-Range {cr!r}")
                buf = bytearray()
                for piece in r.iter_content(1 << 20):
                    buf += piece
                    if len(buf) > want:
                        raise SystemExit(f"{url}: body longer than the range")
            finally:
                r.close()
            if len(buf) != want:
                raise ConnectionError(f"short body {len(buf)}/{want}")
            return bytes(buf), cr
        except SystemExit:
            raise
        except Exception as e:                                # noqa: BLE001
            last = e
            nap = min(60.0, 2.0 * (2 ** attempt))
            print(f"    {url.rsplit('/', 2)[-2:]} {a}-{b}: "
                  f"{type(e).__name__}: {str(e)[:100]} — retry "
                  f"{attempt + 1}/{tries} in {nap:.0f}s", flush=True)
            sleep(nap)
    raise SystemExit(f"{url} bytes {a}-{b}: gave up after {tries} tries "
                     f"({last})")


def hub_stream_sha256(session, url, size, *, workers=8, chunk=64 << 20,
                      fetch=fetch_bytes):
    """sha256 of the remote file, its first HEAD_BYTES and its total size —
    parallel ranged GETs, hashed strictly in order, a bounded window in
    memory, nothing on disk."""
    spans = [(a, min(a + chunk, size) - 1) for a in range(0, size, chunk)]
    h = hashlib.sha256()
    head, total = b"", None
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=max(1, workers)) as ex:
        futs, nxt = {}, 0
        for i in range(len(spans)):
            while nxt < len(spans) and nxt < i + workers + 2:
                futs[nxt] = ex.submit(fetch, session, url, *spans[nxt])
                nxt += 1
            data, cr = futs.pop(i).result()
            h.update(data)
            if i == 0:
                head = data[:HEAD_BYTES]
                total = int(cr.rsplit("/", 1)[1]) if "/" in cr else None
    s = time.time() - t0
    print(f"    streamed {url.split('/monthly/')[-1]}: {size / 1e9:.2f} GB in "
          f"{s:.0f} s = {size / 1e6 / max(s, 1e-9):.0f} MB/s", flush=True)
    return h.hexdigest(), head, total


def restore_from_hub(manifest, session=None, *, workers=8, fetch=fetch_bytes):
    """Every manifest file, streamed back from the Hub: sha256, size, header
    must match. Returns ({group: stats_dict}, {group: {role: head_bytes}})."""
    if session is None:
        import requests
        session = requests.Session()
        session.mount("https://", requests.adapters.HTTPAdapter(
            pool_maxsize=max(8, workers * 2)))
    base = base_of(manifest["stem"])
    stats, heads = {}, {}
    for g in manifest["groups"]:
        heads[g] = {}
        for role, _ in FILES:
            rec = manifest["files"][g][role]
            url = base + rec["rel"]
            if role == "stats":
                data, _ = fetch(session, url, 0, rec["bytes"] - 1)
                got = hashlib.sha256(data).hexdigest()
                stats[g] = json.loads(data.decode("utf-8"))
                head, total = data[:HEAD_BYTES], rec["bytes"]
            else:
                got, head, total = hub_stream_sha256(
                    session, url, rec["bytes"], workers=workers, fetch=fetch)
            if total is not None and total != rec["bytes"]:
                raise SystemExit(f"REFUSING to write the index: {rec['rel']} "
                                 f"is {total} bytes on the Hub, the box wrote "
                                 f"{rec['bytes']}")
            if got != rec["sha256"]:
                raise SystemExit(
                    f"REFUSING to write the index: {rec['rel']} uploaded as "
                    f"sha256 {rec['sha256']} but downloaded back as {got} — "
                    f"the Hub does not hold our bytes")
            heads[g][role] = head
            print(f"  restored ✓ {rec['rel']} ({rec['bytes'] / 1e6:.2f} MB)",
                  flush=True)
    return stats, heads


# --------------------------------------------------------------------- cors --
def cors_record(url, head, origin=ORIGIN, measure=None):
    """An anonymous ranged GET from our origin must answer 206 with an
    allow-origin header and serve the bytes the manifest describes."""
    if measure is None:
        from publish_family7_index import measure_cors as m0

        def measure(u):
            return m0(u, origin=origin)
    r = measure(url)
    if r["status"] != 206 or not r.get("content_range"):
        raise SystemExit(f"{url}: the Hub answered {r['status']} to a ranged "
                         f"GET, not 206 — the design is one range read per "
                         f"plane")
    if not r.get("access_control_allow_origin"):
        raise SystemExit(f"{url}: no access-control-allow-origin for "
                         f"{origin} — a browser could not read it")
    n = min(len(r["head"]), len(head))
    if not n or r["head"][:n] != head[:n]:
        raise SystemExit(f"{url}: the resolve URL served different bytes "
                         f"from the manifest's — it is not this file")
    return dict(origin=origin, status=r["status"],
                final_url_host=r["final_url"].split("/")[2],
                access_control_allow_origin=r["access_control_allow_origin"],
                access_control_expose_headers=r.get(
                    "access_control_expose_headers"),
                accept_ranges=r.get("accept_ranges"),
                content_range=r["content_range"], measured_file=url,
                measured_utc=now_utc())


# -------------------------------------------------------------------- index --
def build_index(manifest, stats, f7, f7_path, *, cors=None,
                restore_verified=False, restore_where=None, verify=None,
                fixture_dir=None):
    stem = f7["stem"]
    if manifest["stem"] != stem:
        raise SystemExit(f"REFUSING: the manifest is of {manifest['stem']!r} "
                         f"and the family-7 index describes {stem!r}")
    base = base_of(stem)
    groups = {}
    for g in manifest["groups"]:
        st = stats[g]
        check_stats(st, f7, g, f"{g}")
        src = f7["groups"][g]
        blk = {k: copy.deepcopy(src[k]) for k in COPY_KEYS}
        blk.update(tensor_file=src["file"], tensor_sha256=src["sha256"],
                   tensor_sha256_verified=bool(st.get("tensor_sha256_verified")),
                   year_first=int(st["year_first"]),
                   year_last=int(st["year_last"]), n_years=int(st["n_years"]),
                   years=[int(y) for y in st["years"]],
                   bins_per_month=st["bins_per_month"],
                   n_rows_summed=int(st["n_rows_summed"]),
                   max_count=int(st["max_count"]),
                   row_kind=st["row_kind"],
                   static_chans=st.get("static_chans"))
        for role, _ in FILES:
            rec = manifest["files"][g][role]
            r = dict(url=base + rec["rel"], bytes=rec["bytes"],
                     sha256=rec["sha256"])
            if role != "stats":
                for k in ("header_len", "shape", "dtype", "itemsize",
                          "fortran_order", "axes", "plane_bytes",
                          "month_channel_bytes"):
                    r[k] = rec[k]
            blk[role] = r
        groups[g] = blk
    reproduction = None
    if verify is not None:
        if not verify.get("ok"):
            raise SystemExit("REFUSING: the verify report says the per-year "
                             "sums do NOT reproduce the published "
                             "climatology")
        reproduction = dict(
            ok=True, tolerance=verify.get("tolerance"),
            clim_index=verify.get("clim_index"),
            groups={g: {v: {k: r[k] for k in (
                "n_train_bins", "max_abs_z", "max_abs_phys", "max_ulp",
                "max_over_bound", "n_cells")} for v, r in gv.items()}
                for g, gv in verify["groups"].items()})
    return dict(
        _source="ml/publish_family7_monthly_index.py — do not hand-edit",
        generated_utc=now_utc(),
        fixture=bool(f7.get("fixture")) or bool(fixture_dir),
        fixture_dir=fixture_dir, stem=stem,
        tensor_index=os.path.relpath(os.path.abspath(f7_path), ROOT),
        repo=f"datasets/{REPO_ID}", base=base, plan=PLAN_URL,
        axes=list(AXES), layout=LAYOUT, month_rule=MONTH_RULE,
        offset_formula=("offset(m, c, y) = header_len + ((m * C + c) * "
                        "n_years + (y - year_first)) * plane_bytes; m = 0 is "
                        "January; years y_a..y_b of one (m, c) are the single "
                        "range [offset(m, c, y_a), offset(m, c, y_b) + "
                        "plane_bytes)"),
        combine=("mean_z(m, c, cell) over a set of years Ys = Σ_{y∈Ys} sum / "
                 "Σ_{y∈Ys} count, NaN where Σ count = 0; physical = mean_z * "
                 "norm[c][1] + norm[c][0]. A channel in static_chans is one "
                 "anomaly_transform subtracts nothing on (its published "
                 "climatology is 0.0 there)."),
        versions_reproduced=[version_block(v) for v in VERSIONS],
        groups=groups,
        restore_verified=bool(restore_verified), restore_where=restore_where,
        cors_measured=cors, reproduction=reproduction)


def write_json(path, obj):
    text = json.dumps(obj, sort_keys=True, separators=(",", ":"),
                      allow_nan=False) + "\n"
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(text)
    print(f"wrote {path} ({len(text) / 1e3:.1f} kB)", flush=True)


def main(argv=None, api=None, fetch=None, measure=None, session=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("cmd", choices=("manifest", "upload", "index"))
    ap.add_argument("--out", help="the export directory (<group>/…)")
    ap.add_argument("--manifest", default="",
                    help="manifest.json (default <out>/manifest.json)")
    ap.add_argument("--f7-index", default=F7_INDEX)
    ap.add_argument("--index", default=INDEX)
    ap.add_argument("--verify", default="",
                    help="verify.json from `export_family7_monthly.py verify`"
                         " — its numbers go into the index; a failed report "
                         "is refused")
    ap.add_argument("--no-cors", action="store_true")
    ap.add_argument("--origin", default=ORIGIN)
    ap.add_argument("--local", action="store_true",
                    help="index: trust the local files under --out, no Hub "
                         "restore (restore_verified: false)")
    ap.add_argument("--restore-where", default="hosted runner")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--fixture-dir", default=None)
    ap.add_argument("--repo", default=REPO_ID)
    a = ap.parse_args(argv)
    f7 = json.load(open(a.f7_index, encoding="utf-8"))
    mpath = a.manifest or (os.path.join(a.out, "manifest.json") if a.out
                           else "")

    if a.cmd in ("manifest", "upload") or (a.cmd == "index" and a.local):
        if not a.out:
            ap.error("--out is required")
        man = make_manifest(a.out, f7)
        if a.cmd != "index":
            write_json(mpath, man)
        if a.cmd == "manifest":
            return man
        if a.cmd == "upload":
            tok = token()
            if api is None:
                from huggingface_hub import HfApi
                api = HfApi(token=tok)
            upload(a.out, man, api, repo=a.repo)
            print("upload done — the index is written by `index --manifest` "
                  "after every file restores from the Hub", flush=True)
            return man
        stats = {g: json.load(open(os.path.join(a.out, g, "stats.json"),
                                   encoding="utf-8")) for g in man["groups"]}
        heads = {}
        for g in man["groups"]:
            heads[g] = {}
            for role, _ in FILES:
                with open(os.path.join(a.out, man["files"][g][role]["rel"]),
                          "rb") as fh:
                    heads[g][role] = fh.read(HEAD_BYTES)
        verified, where = False, "local (not restored)"
    else:
        if not mpath:
            ap.error("index needs --manifest (or --local --out)")
        man = json.load(open(mpath, encoding="utf-8"))
        if man["stem"] != f7["stem"]:
            raise SystemExit(f"REFUSING: the manifest is of {man['stem']!r} "
                             f"and {a.f7_index} describes {f7['stem']!r}")
        kw = {"fetch": fetch} if fetch else {}
        stats, heads = restore_from_hub(man, session, workers=a.workers, **kw)
        verified, where = True, a.restore_where
    cors = None
    if not a.no_cors:
        g0 = man["groups"][0]
        base = base_of(man["stem"])
        cors = {role: cors_record(base + man["files"][g0][role]["rel"],
                                  heads[g0][role], origin=a.origin,
                                  measure=measure)
                for role in ("sum", "count")}
    rep = json.load(open(a.verify, encoding="utf-8")) if a.verify else None
    index = build_index(man, stats, f7, a.f7_index, cors=cors,
                        restore_verified=verified, restore_where=where,
                        verify=rep, fixture_dir=a.fixture_dir)
    write_json(a.index, index)
    return index


if __name__ == "__main__":
    main()
    sys.exit(0)

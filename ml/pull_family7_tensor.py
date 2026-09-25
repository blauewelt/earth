#!/usr/bin/env python3
"""E-083 §3.1 · Pull a family-7 tensor from the Hub with parallel range reads.

The Hub serves one stream at ≈ 2 MB/s from the sandbox; a box pulling the
61 GB of `tensors/<stem>/` one file at a time is bounded by that single
stream. This script splits every file into fixed-size CHUNKS and fetches them
with N workers, each a `Range: bytes=a-b` GET that must answer **206 with the
exact Content-Range asked for** (a 200 means the host ignored the range and is
sending the whole file — refused, never written at an offset). Chunks are
written in place into `<dest>/<name>.part` and recorded in `<name>.part.done`
only AFTER their bytes are on disk (flush, then mark — ml/CLAUDE.md §5.21), so
an interrupted pull resumes at the first missing chunk. When every chunk is
present the file's sha256 is compared with `data/family7_index.json` (the
group `.npy` sidecars) or the tensor's own `manifest.json` (the `.npz`), and
only a match is renamed into place; a mismatch deletes the part file and
refuses.

Credentials: `HF_TOKEN` from the environment, sent as `Authorization: Bearer`
on every request (the Hub drops it on the redirect to its CDN). Never argv.

`--method hub` uses `huggingface_hub.hf_hub_download` instead (hf_xet /
hf_transfer when installed) — same sha256 check, same MB/s line — as the
comparison when the range reader is not the fastest thing on a given box.

    HF_TOKEN=… python3 ml/pull_family7_tensor.py --dest /data \\
        [--index data/family7_index.json] [--groups g025,g100,oc025,rg100] \\
        [--workers 16] [--chunk-mb 64] [--method range|hub]
"""
import argparse
import hashlib
import json
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)

REPO_ID = "chfrank/earth-tensors"
HUB = "https://huggingface.co"
F7_INDEX = os.path.join(ROOT, "data", "family7_index.json")


def resolve_base(stem):
    return f"{HUB}/datasets/{REPO_ID}/resolve/main/tensors/{stem}/"


def auth_headers():
    """The token, from the environment, on every request. None -> anonymous."""
    t = os.environ.get("HF_TOKEN", "").strip()
    return {"Authorization": f"Bearer {t}"} if t else {}


class PullError(RuntimeError):
    pass


def sha256_file(path, buf=1 << 24):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for blk in iter(lambda: fh.read(buf), b""):
            h.update(blk)
    return h.hexdigest()


def chunks_of(size, chunk):
    """[(index, start, end_inclusive)] covering [0, size)."""
    return [(i, a, min(a + chunk, size) - 1)
            for i, a in enumerate(range(0, size, chunk))]


def read_done(path):
    """The chunk indices already on disk, from `<part>.done`."""
    if not os.path.exists(path):
        return set()
    out = set()
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line.isdigit():
                out.add(int(line))
    return out


def remote_size(session, url, timeout=60):
    """The file's byte count, from a one-byte ranged GET's Content-Range."""
    r = session.get(url, headers={**auth_headers(), "Range": "bytes=0-0"},
                    stream=True, timeout=timeout, allow_redirects=True)
    try:
        if r.status_code != 206:
            raise PullError(f"{url}: a ranged probe answered HTTP "
                            f"{r.status_code}, not 206 — this host does not "
                            f"serve ranges and a parallel pull is impossible")
        cr = r.headers.get("Content-Range", "")
        if "/" not in cr:
            raise PullError(f"{url}: 206 without a Content-Range total")
        return int(cr.rsplit("/", 1)[1])
    finally:
        r.close()


def fetch_chunk(session, url, fd, a, b, tries=6, sleep=time.sleep,
                progress=None, timeout=120):
    """GET bytes a..b into `fd` at offset a. Retries 429/5xx/connection
    errors with backoff; anything else is a definite answer and raises."""
    want = b - a + 1
    last = None
    for attempt in range(tries):
        got = 0
        try:
            r = session.get(url, headers={**auth_headers(),
                                          "Range": f"bytes={a}-{b}"},
                            stream=True, timeout=timeout, allow_redirects=True)
            try:
                if r.status_code in (429, 500, 502, 503, 504):
                    raise ConnectionError(f"HTTP {r.status_code}")
                if r.status_code != 206:
                    raise PullError(
                        f"{url} bytes {a}-{b}: HTTP {r.status_code} (want "
                        f"206) — refusing to write a non-ranged body at an "
                        f"offset")
                cr = r.headers.get("Content-Range", "")
                if not cr.startswith(f"bytes {a}-{b}/"):
                    raise PullError(f"{url}: asked for bytes {a}-{b}, got "
                                    f"Content-Range {cr!r}")
                for piece in r.iter_content(1 << 20):
                    if not piece:
                        continue
                    if got + len(piece) > want:
                        raise PullError(f"{url} bytes {a}-{b}: body longer "
                                        f"than the range")
                    os.pwrite(fd, piece, a + got)
                    got += len(piece)
                    if progress:
                        progress(len(piece))
            finally:
                r.close()
            if got != want:
                raise ConnectionError(f"short body {got}/{want}")
            return want
        except PullError:
            raise
        except Exception as e:                                # noqa: BLE001
            last = e
            if progress and got:
                progress(-got)
            nap = min(60.0, 2.0 * (2 ** attempt))
            print(f"    chunk {a}-{b}: {type(e).__name__}: {str(e)[:120]} — "
                  f"retry {attempt + 1}/{tries} in {nap:.0f}s", flush=True)
            sleep(nap)
    raise PullError(f"{url} bytes {a}-{b}: gave up after {tries} tries "
                    f"({last})")


class Meter:
    """Thread-safe byte counter that prints MB/s every `every` seconds."""

    def __init__(self, label, total, every=15.0, clock=time.time):
        self.label, self.total, self.every, self.clock = label, total, every, clock
        self.done = 0
        self.t0 = self.last = clock()
        self.lock = threading.Lock()

    def __call__(self, n):
        with self.lock:
            self.done += n
            now = self.clock()
            if now - self.last >= self.every:
                self.last = now
                print(f"  {self.label}: {self.done / 1e9:.2f}/"
                      f"{self.total / 1e9:.2f} GB · {self.rate():.1f} MB/s",
                      flush=True)

    def rate(self):
        return self.done / 1e6 / max(self.clock() - self.t0, 1e-9)


def pull_file(session, url, dest, sha256, *, workers=16, chunk=64 << 20,
              size=None, sleep=time.sleep):
    """Parallel, resumable range pull of `url` to `dest`, sha256-checked.

    Returns (bytes_fetched_this_call, seconds). Already-complete `dest` with
    the right sha256 is left alone (0 bytes)."""
    if os.path.exists(dest):
        if sha256 and sha256_file(dest) == sha256:
            print(f"  {os.path.basename(dest)}: present, sha256 ✓", flush=True)
            return 0, 0.0
        raise PullError(f"{dest} exists and does not match sha256 "
                        f"{sha256} — move it aside; this script never "
                        f"overwrites a finished file")
    part, donef = dest + ".part", dest + ".part.done"
    if size is None:
        size = remote_size(session, url)
    spans = chunks_of(size, chunk)
    done = read_done(donef)
    meta = dest + ".part.json"
    want_meta = {"url": url, "size": size, "chunk": chunk, "sha256": sha256}
    if os.path.exists(meta):
        have = json.load(open(meta, encoding="utf-8"))
        if have != want_meta:
            print(f"  {os.path.basename(dest)}: the partial pull was for "
                  f"{have} — starting over", flush=True)
            for p in (part, donef, meta):
                if os.path.exists(p):
                    os.remove(p)
            done = set()
    os.makedirs(os.path.dirname(os.path.abspath(dest)) or ".", exist_ok=True)
    with open(meta, "w", encoding="utf-8") as fh:
        json.dump(want_meta, fh)
    todo = [s for s in spans if s[0] not in done]
    have_b = sum(b - a + 1 for i, a, b in spans if i in done)
    print(f"  {os.path.basename(dest)}: {size / 1e9:.2f} GB in {len(spans)} "
          f"chunk(s) of {chunk >> 20} MB · {len(spans) - len(todo)} already "
          f"on disk ({have_b / 1e9:.2f} GB) · {workers} worker(s)", flush=True)
    fd = os.open(part, os.O_RDWR | os.O_CREAT, 0o644)
    lock = threading.Lock()
    meter = Meter(os.path.basename(dest), size - have_b)
    t0 = time.time()
    try:
        os.ftruncate(fd, size)
        with open(donef, "a", encoding="utf-8") as dfh, \
                ThreadPoolExecutor(max_workers=max(1, workers)) as ex:
            futs = {ex.submit(fetch_chunk, session, url, fd, a, b,
                              sleep=sleep, progress=meter): i
                    for i, a, b in todo}
            for f in as_completed(futs):
                f.result()                     # raises the first failure
                with lock:
                    dfh.write(f"{futs[f]}\n")  # flush, THEN mark
                    dfh.flush()
    finally:
        os.close(fd)
    secs = time.time() - t0
    fetched = size - have_b
    print(f"  {os.path.basename(dest)}: fetched {fetched / 1e9:.2f} GB in "
          f"{secs:.0f} s = {fetched / 1e6 / max(secs, 1e-9):.1f} MB/s; "
          f"verifying sha256 …", flush=True)
    if len(read_done(donef)) != len(spans):
        raise PullError(f"{dest}: {len(spans) - len(read_done(donef))} "
                        f"chunk(s) missing after the pull")
    if sha256:
        got = sha256_file(part)
        if got != sha256:
            for p in (part, donef, meta):
                os.remove(p)
            raise PullError(f"{dest}: sha256 {got} != {sha256} — the partial "
                            f"file was DELETED; re-run to pull it again")
        print(f"  {os.path.basename(dest)}: sha256 ✓", flush=True)
    else:
        print(f"  ::warning:: {os.path.basename(dest)}: no sha256 to check "
              f"against", flush=True)
    os.replace(part, dest)
    for p in (donef, meta):
        os.remove(p)
    return fetched, secs


def pull_hub(filename, dest_dir, sha256):
    """hf_hub_download (hf_xet / hf_transfer when installed), sha256-checked."""
    os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")
    from huggingface_hub import hf_hub_download
    t0 = time.time()
    p = hf_hub_download(REPO_ID, filename, repo_type="dataset",
                        local_dir=dest_dir,
                        token=os.environ.get("HF_TOKEN") or None)
    secs = time.time() - t0
    n = os.path.getsize(p)
    got = sha256_file(p)
    if sha256 and got != sha256:
        os.remove(p)
        raise PullError(f"{p}: sha256 {got} != {sha256} — deleted")
    # Same layout as the range reader: <dest>/<name>, beside its siblings, so
    # `load_tensor(<dest>/<stem>.npz)` finds the sidecars either way.
    flat = os.path.join(dest_dir, os.path.basename(filename))
    if os.path.abspath(p) != os.path.abspath(flat):
        os.replace(p, flat)
        p = flat
    print(f"  {filename}: {n / 1e9:.2f} GB in {secs:.0f} s = "
          f"{n / 1e6 / max(secs, 1e-9):.1f} MB/s · sha256 "
          f"{'✓' if sha256 else 'unchecked'}", flush=True)
    return p, n, secs


def plan(idx, groups, manifest=None):
    """[(filename, sha256)] — the npz first (small), then the group sidecars
    in the index's order. The npz's sha256 comes from the tensor's own
    manifest.json (the family-7 index carries group sha256s only)."""
    stem = idx["stem"]
    by_name = {}
    if manifest:
        files = manifest.get("files", manifest)
        if isinstance(files, list):
            by_name = {f["name"]: f.get("sha256") for f in files}
        else:
            by_name = {k: (v.get("sha256") if isinstance(v, dict) else v)
                       for k, v in files.items()}
    out = [(f"{stem}.npz", by_name.get(f"{stem}.npz"))]
    for g in (groups or list(idx["groups"])):
        if g not in idx["groups"]:
            raise PullError(f"group {g!r} is not in the index "
                            f"({list(idx['groups'])})")
        blk = idx["groups"][g]
        m = by_name.get(blk["file"])
        if m and m != blk["sha256"]:
            raise PullError(f"{blk['file']}: the Hub manifest says sha256 {m}"
                            f" and data/family7_index.json says "
                            f"{blk['sha256']} — they describe different "
                            f"bytes; refusing to guess which is right")
        out.append((blk["file"], blk["sha256"]))
    return out


def main(argv=None, session=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--dest", required=True)
    ap.add_argument("--index", default=F7_INDEX)
    ap.add_argument("--groups", default="")
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--chunk-mb", type=int, default=64)
    ap.add_argument("--method", choices=("range", "hub"), default="range")
    ap.add_argument("--no-manifest", action="store_true",
                    help="do not fetch manifest.json (the npz is then "
                         "pulled without a sha256 check)")
    a = ap.parse_args(argv)
    idx = json.load(open(a.index, encoding="utf-8"))
    stem = idx["stem"]
    base = resolve_base(stem)
    groups = [g.strip() for g in a.groups.split(",") if g.strip()]
    if session is None:
        import requests
        session = requests.Session()
        adapter = requests.adapters.HTTPAdapter(pool_maxsize=max(8, a.workers))
        session.mount("https://", adapter)
    os.makedirs(a.dest, exist_ok=True)

    manifest = None
    if not a.no_manifest:
        r = session.get(base + "manifest.json", headers=auth_headers(),
                        timeout=60, allow_redirects=True)
        if r.status_code == 200:
            manifest = r.json()
        else:
            print(f"  ::warning:: manifest.json: HTTP {r.status_code} — the "
                  f"npz will be pulled without a sha256 check", flush=True)
    files = plan(idx, groups, manifest)
    print(f"pull {stem}: {len(files)} file(s) → {a.dest} "
          f"(method {a.method})", flush=True)
    total_b, total_s = 0, 0.0
    t0 = time.time()
    for name, sha in files:
        if a.method == "hub":
            _, n, s = pull_hub(f"tensors/{stem}/{name}", a.dest, sha)
            total_b += n
        else:
            n, s = pull_file(session, base + name, os.path.join(a.dest, name),
                             sha, workers=a.workers, chunk=a.chunk_mb << 20)
            total_b += n
        total_s += s
    wall = time.time() - t0
    print(f"pull {stem}: done · {total_b / 1e9:.2f} GB fetched in "
          f"{wall / 60:.1f} min = {total_b / 1e6 / max(wall, 1e-9):.1f} MB/s "
          f"overall", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())

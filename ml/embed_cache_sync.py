#!/usr/bin/env python3
"""Move the frozen-codec embedding cache between a box and a GitHub release.

The embedding is the most expensive derived artefact in the project: ~95
minutes of a 4090 for 43.5M encoder forwards over the quarter-degree tensor,
and every stage-2 run on the same frozen codec needs exactly the same one. It
was also the only expensive artefact we did NOT ship anywhere — checkpoints go
to `model-checkpoints-v1`, tensors to `data-cache-v1`, and Z lived on one
rented box's disk, where the hygiene step deleted it whenever space ran short.
Chris, 2026-08-10: *"getting more disk space to keep training data long term
(or backing it up somewhere) is the better choice."* Vast will not sell us the
disk (it accepts a resize and ignores it), so this is the "somewhere".

Two properties matter more than the transfer itself:

  1. **The asset is named by the CODEC'S WEIGHT HASH**, via the same function
     temporal.py uses. A cache keyed by run name poisoned runs #10/#11 — the
     shape check passed, the embeddings belonged to a different codec, and two
     stage-2 models trained on z their decoder did not speak. Publishing that
     mistake to a release would spread it to every box instead of one.
  2. **A pull VERIFIES before it is trusted**: the reassembled file must parse
     as a .npy, carry the expected dtype, and have the exact byte length the
     header implies. A truncated chunk that still "loads" is the failure this
     guards against — half a cache is not a slow cache, it is a wrong one.

Usage:
  python3 ml/embed_cache_sync.py pull --run actions          # before stage 2
  python3 ml/embed_cache_sync.py push --run actions          # after it builds

`push` needs GITHUB_TOKEN (the job token is enough: `contents: write`).
`pull` needs nothing — the repo is public.
"""
import argparse
import json
import os
import shutil
import subprocess
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from temporal import (CACHE_DTYPE, codec_weight_hash, data_fingerprint,
                      embed_cache_path)

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.environ.get("GITHUB_REPOSITORY", "blauewelt/earth")
TAG = "embed-cache-v1"
# GitHub caps a release asset at 2 GiB. The fp16 quarter-degree cache is
# 5.2 GiB, so it ships in chunks — the same shape as the data-cache seed,
# which has moved far larger tars this way since #47.
CHUNK = 1500 * 1024 * 1024


def sh(cmd, **kw):
    return subprocess.run(cmd, shell=True, text=True, capture_output=True, **kw)


def cache_name(run, data):
    """(local path, asset base name, label) for this run's codec AND tensor.

    Both hashes are in the name. A codec-only key was enough while every box
    built the same tensor; they do not — b40f5b0b against adcbe700, measured
    2026-08-11 — and with a codec-only key a box pulls embeddings of the wrong
    dataset while the shape check, the dtype check and the length check all
    pass. Same failure as #10/#11, with "codec" replaced by "data".
    """
    ck = torch.load(os.path.join(HERE, "runs", run, "pixelmae.pt"),
                    map_location="cpu", weights_only=False)
    whash = codec_weight_hash(ck)
    dhash = data_fingerprint(data)
    return (embed_cache_path(run, whash, dhash),
            f"Z_{whash}_{dhash}.npy", f"{whash}/{dhash}")


def npy_expected_bytes(path):
    """Byte length the .npy header implies, so truncation is detectable."""
    # The private _read_array_header was renamed between numpy versions, so go
    # through the public per-version readers. Getting this wrong would make
    # verify() raise on every call, and since the caller treats an exception as
    # "best effort, carry on", the guard would be silently absent — the exact
    # shape of failure this whole file exists to prevent.
    with open(path, "rb") as f:
        major, minor = np.lib.format.read_magic(f)
        reader = {(1, 0): np.lib.format.read_array_header_1_0,
                  (2, 0): np.lib.format.read_array_header_2_0}[(major, minor)]
        shape, _, dtype = reader(f)
        return f.tell() + int(np.prod(shape)) * dtype.itemsize, shape, dtype


def verify(path):
    """A cache is trusted only if the file is exactly as long as its own
    header says, and carries the dtype we write.

    Measured rather than assumed: on this numpy a SHORT file raises
    ("mmap length is greater than file size"), so pure truncation would not
    silently return garbage. Two things still justify the check. A file that is
    the wrong length in the other direction — a chunk uploaded twice, or
    reassembled out of order — maps cleanly and returns embeddings that are
    real numbers belonging to the wrong months, which is the failure mode with
    no symptom. And catching it here turns an exception thrown halfway through
    a sixteen-hour job into "discard, rebuild, carry on"."""
    want, shape, dtype = npy_expected_bytes(path)
    have = os.path.getsize(path)
    if have != want:
        return False, (f"{os.path.basename(path)} is {have:,} bytes, header "
                       f"says {want:,} — truncated or corrupt")
    if dtype != np.dtype(CACHE_DTYPE):
        return False, f"dtype {dtype}, expected {np.dtype(CACHE_DTYPE)}"
    return True, f"{shape} {dtype}, {have / (1 << 30):.2f} GiB"


def pull(run, a_data):
    path, asset, whash = cache_name(run, a_data)
    if os.path.exists(path):
        ok, why = verify(path)
        if ok:
            print(f"embed cache already local and valid: {why}")
            return 0
        print(f"local cache rejected ({why}) — removing and re-pulling")
        os.remove(path)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    base = f"https://github.com/{REPO}/releases/download/{TAG}"
    tmp = path + ".pull"
    if os.path.exists(tmp):
        os.remove(tmp)
    got = 0
    for i in range(64):                       # 64 x 1.5 GiB is far past need
        suffix = f"{chr(97 + i // 26)}{chr(97 + i % 26)}"
        url = f"{base}/{asset}.{suffix}"
        r = sh(f'curl -fsSL --max-time 1800 --retry 3 --retry-delay 5 '
               f'-o "{tmp}.part" "{url}"')
        if r.returncode != 0:
            if i == 0:
                print(f"no embed cache published for codec {whash} — "
                      f"it will be built and, if push runs, uploaded")
                return 1
            break
        with open(tmp, "ab") as out, open(f"{tmp}.part", "rb") as chunk:
            while True:
                b = chunk.read(1 << 24)
                if not b:
                    break
                out.write(b)
        os.remove(f"{tmp}.part")
        got += 1
        print(f"  pulled chunk {suffix} ({os.path.getsize(tmp) / (1<<30):.2f} GiB)")
    os.replace(tmp, path)
    ok, why = verify(path)
    if not ok:
        os.remove(path)
        print(f"::warning::published embed cache failed verification ({why}) — "
              f"discarded; it will be rebuilt")
        return 1
    print(f"embed cache restored from the release in {got} chunk(s): {why}")
    print("this is the ~95 minutes of GPU that stage 2 no longer has to spend")
    return 0


def push(run, a_data):
    path, asset, whash = cache_name(run, a_data)
    if not os.path.exists(path):
        # The RAM path writes no cache. That is a legitimate outcome, not a
        # failure, and saying so is the difference between "nothing to do" and
        # a silent no-op that looks like success.
        print(f"no embed cache on disk for codec {whash} — nothing to publish "
              f"(the run built Z in RAM because the disk could not hold it)")
        return 0
    ok, why = verify(path)
    if not ok:
        print(f"::warning::refusing to publish an invalid cache: {why}")
        return 1
    token = os.environ.get("GITHUB_TOKEN", "")
    if not token:
        print("::warning::no GITHUB_TOKEN — cannot publish the embed cache")
        return 1

    api = "https://api.github.com"
    hdr = f'-H "Authorization: token {token}" -H "Accept: application/vnd.github+json"'
    r = sh(f'curl -fsSL {hdr} "{api}/repos/{REPO}/releases/tags/{TAG}"')
    if r.returncode != 0:
        r = sh(f'curl -fsSL -X POST {hdr} "{api}/repos/{REPO}/releases" '
               f"-d '{json.dumps({'tag_name': TAG, 'name': 'Embedding caches', 'body': 'Frozen-codec embedding caches (Z), named by codec weight hash. Derived data: deleting these costs GPU time, never correctness.'})}'")
        if r.returncode != 0:
            print(f"::warning::could not create release {TAG}: {r.stderr[:200]}")
            return 1
    rel = json.loads(r.stdout)
    rid = rel["id"]
    assets = {a["name"]: a for a in rel.get("assets", [])}
    existing = {k: v["id"] for k, v in assets.items()}

    total = os.path.getsize(path)
    n = (total + CHUNK - 1) // CHUNK

    # ALREADY PUBLISHED AND COMPLETE? Then do nothing.
    #
    # The upload replaces each chunk by DELETING it and re-POSTing, so a
    # re-publish of an identical cache is not merely 5.2 GiB of wasted
    # bandwidth per run — it opens a window in which the release holds a
    # PARTIAL cache. Watched live on 2026-08-11: the asset count went 4 → 3
    # while #142 replaced a cache byte-identical to the one already there, and
    # every queued arm would have done it again. A puller in that window gets
    # a short file; `verify()` catches it and discards, which turns a wasted
    # upload into someone else's wasted 95 minutes.
    #
    # The check is on the exact chunk names and their total size, because the
    # asset name already carries the codec's weight hash — so matching names
    # plus matching bytes means the same cache for the same codec.
    want = [f"{asset}.{chr(97 + i // 26)}{chr(97 + i % 26)}" for i in range(n)]
    have_all = all(w in assets for w in want)
    have_bytes = sum(assets[w]["size"] for w in want) if have_all else -1
    if have_all and have_bytes == total:
        print(f"embed cache for codec {whash} is already published and complete "
              f"({n} chunk(s), {total / (1<<30):.2f} GiB) — nothing to do")
        return 0
    if have_all:
        print(f"republishing: {n} chunk(s) present but {have_bytes:,} bytes "
              f"against {total:,} on disk")

    # DO NOT START A WRITE THE DISK CANNOT HOLD. Chunking to a temporary file
    # needs CHUNK bytes free, and on 2026-08-10 this ran on a box with under
    # 1.5 GiB left: the write failed with ENOSPC, the exception left the part
    # file behind, and the disk went to 50/50. Every subsequent job on that
    # box failed in "Set up job" — before any step, so the hygiene step that
    # would have cleaned up could never run. One unchecked write cost three
    # queued runs and took the box out of the fleet.
    #
    # ml/CLAUDE.md §5.18: size a guard from the allocation it guards. The
    # allocation is CHUNK, and it is knowable here.
    free = shutil.disk_usage(os.path.dirname(path)).free
    need = CHUNK + (256 << 20)
    if free < need:
        print(f"::warning::refusing to publish: chunking needs "
              f"{need / (1<<30):.2f} GiB free and the disk has "
              f"{free / (1<<30):.2f} GiB. Publishing would fill it and take "
              f"the box out of the fleet, which is worse than not publishing.")
        return 1

    print(f"publishing {asset} as {n} chunk(s), {total / (1<<30):.2f} GiB total")
    with open(path, "rb") as f:
        for i in range(n):
            suffix = f"{chr(97 + i // 26)}{chr(97 + i % 26)}"
            name = f"{asset}.{suffix}"
            part = f"{path}.{suffix}.up"
            # try/finally, so a part file NEVER outlives the attempt that made
            # it. Previously an ENOSPC while writing raised straight past the
            # os.remove below and left up to 1.5 GiB of garbage on a disk that
            # had just proved it had no room.
            try:
                with open(part, "wb") as o:
                    left = min(CHUNK, total - i * CHUNK)
                    while left:
                        b = f.read(min(1 << 24, left))
                        if not b:
                            break
                        o.write(b)
                        left -= len(b)
                if name in existing:          # replace, never duplicate
                    sh(f'curl -fsSL -X DELETE {hdr} '
                       f'"{api}/repos/{REPO}/releases/assets/{existing[name]}"')
                # `-T`, NEVER `--data-binary @file`. THIS is why the embed
                # cache had never once published, through every other fix
                # tonight: --data-binary reads the entire body into memory
                # before sending, and a 1.5 GiB chunk made curl die with
                # "option --data-binary: out of memory" on the first chunk,
                # every time, since the day the feature was written.
                #
                # Measured rather than assumed, on a 300 MiB file:
                #   --data-binary @file   peak RSS 226 MiB
                #   -T file               peak RSS  10 MiB
                # -T streams from the file and sets Content-Length from its
                # size, which is what the release upload endpoint wants.
                up = sh(f'curl -fsSL -X POST {hdr} '
                        f'-H "Content-Type: application/octet-stream" '
                        f'-T "{part}" '
                        f'"https://uploads.github.com/repos/{REPO}/releases/{rid}/'
                        f'assets?name={name}"')
            finally:
                if os.path.exists(part):
                    os.remove(part)
            if up.returncode != 0:
                print(f"::warning::chunk {name} failed: {up.stderr[:200]}")
                return 1
            print(f"  uploaded {name}")
    print(f"embed cache for codec {whash} is now durable: any box can pull it "
          f"instead of spending 95 minutes rebuilding it")
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=("pull", "push"))
    ap.add_argument("--run", required=True)
    # The tensor is part of the cache's identity, so it is a required
    # input rather than a default nobody passes.
    ap.add_argument("--data", required=True,
                    help="the tensor .npz these embeddings belong to")
    a = ap.parse_args()
    # THE EXIT CODE MUST MEAN SOMETHING. This read
    #     sys.exit(0 if push(...) == 0 else 0)
    # which is zero either way — so a failed push looked like a successful one,
    # the caller's `&& touch /tmp/embed-cache-pushed` fired, and the cache was
    # never uploaded and never retried. Written, on 2026-08-10, into the very
    # file whose docstring is about steps that report success while doing
    # nothing. Best-effort is the CALLER's decision (`|| true`), never a lie
    # told by the callee.
    try:
        rc = (pull if a.mode == "pull" else push)(a.run, a.data)
    except Exception as e:                    # noqa: BLE001
        print(f"::warning::embed cache {a.mode} failed: {type(e).__name__}: {e}")
        rc = 1
    sys.exit(0 if rc == 0 else 1)


if __name__ == "__main__":
    main()

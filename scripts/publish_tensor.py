#!/usr/bin/env python3
"""Publish the box's built tensor to the data-cache release, keyed by hash.

The tensor is the LAST single-copy artefact in the pipeline. `data-cache-v1`
holds the sources (RG tars, wind tars, the base grid) but not the built
`family3_na025.npz` — each box builds its own, and 2026-08-11 measured two
boxes holding different bytes under that one filename (`b40f5b0b` vs
`adcbe700`). Every E-010 result is on `adcbe700`, which exists only on
gpu-box-35586926: destroy that box and the sweep becomes unreproducible,
because a REBUILT tensor is not guaranteed to hash the same.

Everything here is a pattern that failed once tonight and was fixed:
  · chunked to 1.5 GiB parts, uploaded with `curl -T` — NEVER
    `--data-binary "@file"`, which buffers the whole part and died OOM on
    every publish attempt for a day;
  · refuses to start a chunk the disk cannot hold (ENOSPC mid-write filled a
    box to 50/50 and took three runs down in `Set up job`);
  · part files removed in `finally`, so an exception cannot strand 1.5 GiB;
  · idempotent — if the exact chunk names are present and sum to the file's
    size, it does nothing, because the upload replaces by DELETE + POST and a
    re-publish opens a window where the release holds a partial artefact;
  · exit codes are honest; best-effort is the CALLER's `|| echo`.

Usage (on a box, GITHUB_TOKEN in env):
  python3 scripts/publish_tensor.py --data ml/cache/family3_na025.npz [--dry-run]
"""
import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys

REPO = os.environ.get("GITHUB_REPOSITORY", "blauewelt/earth")
TAG = "data-cache-v1"
CHUNK = 1500 * 1024 * 1024


def sh(cmd):
    return subprocess.run(cmd, shell=True, text=True, capture_output=True)


def fingerprint(path, chunk=1 << 24):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(chunk), b""):
            h.update(b)
    return h.hexdigest()[:10]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--dry-run", action="store_true",
                    help="print the plan (asset name, chunking, skip/publish "
                         "decision) and stop before any upload")
    a = ap.parse_args()

    total = os.path.getsize(a.data)
    dhash = fingerprint(a.data)
    stem = os.path.basename(a.data).replace(".npz", "")
    asset = f"{stem}_{dhash}.npz"
    n = (total + CHUNK - 1) // CHUNK
    names = [f"{asset}.{chr(97 + i // 26)}{chr(97 + i % 26)}" for i in range(n)]
    print(f"tensor {a.data}: {total:,} bytes, fingerprint {dhash}")
    print(f"asset {asset} in {n} chunk(s)")

    token = os.environ.get("GITHUB_TOKEN", "").strip()
    if not a.dry_run and not token:
        print("::warning::no GITHUB_TOKEN — cannot publish the tensor")
        return 1

    api = "https://api.github.com"
    hdr = (f'-H "Authorization: token {token}" '
           f'-H "Accept: application/vnd.github+json"')
    if a.dry_run:
        print("--dry-run: stopping before the release lookup")
        return 0
    r = sh(f'curl -fsSL {hdr} "{api}/repos/{REPO}/releases/tags/{TAG}"')
    if r.returncode != 0:
        print(f"::warning::cannot read release {TAG}: {r.stderr[:200]}")
        return 1
    rel = json.loads(r.stdout)
    rid = rel["id"]
    assets = {x["name"]: x for x in rel.get("assets", [])}

    have_all = all(nm in assets for nm in names)
    if have_all and sum(assets[nm]["size"] for nm in names) == total:
        print(f"already published and complete — nothing to do")
        return 0

    free = shutil.disk_usage(os.path.dirname(os.path.abspath(a.data))).free
    if free < CHUNK + (256 << 20):
        print(f"::warning::refusing: chunking needs {CHUNK/(1<<30):.1f} GiB "
              f"free and the disk has {free/(1<<30):.1f} GiB")
        return 1

    with open(a.data, "rb") as f:
        for i, nm in enumerate(names):
            part = f"{a.data}.{nm[-2:]}.up"
            try:
                with open(part, "wb") as o:
                    left = min(CHUNK, total - i * CHUNK)
                    while left:
                        b = f.read(min(1 << 24, left))
                        if not b:
                            break
                        o.write(b)
                        left -= len(b)
                if nm in assets:
                    sh(f'curl -fsSL -X DELETE {hdr} '
                       f'"{api}/repos/{REPO}/releases/assets/{assets[nm]["id"]}"')
                up = sh(f'curl -fsSL -X POST {hdr} '
                        f'-H "Content-Type: application/octet-stream" '
                        f'-T "{part}" '
                        f'"https://uploads.github.com/repos/{REPO}/releases/'
                        f'{rid}/assets?name={nm}"')
            finally:
                if os.path.exists(part):
                    os.remove(part)
            if up.returncode != 0:
                print(f"::warning::chunk {nm} failed: {up.stderr[:200]}")
                return 1
            print(f"  uploaded {nm}")

    # Assert the effect: re-list and check names + total.
    r2 = sh(f'curl -fsSL {hdr} "{api}/repos/{REPO}/releases/tags/{TAG}"')
    got = {x["name"]: x["size"] for x in json.loads(r2.stdout).get("assets", [])}
    tot2 = sum(got.get(nm, 0) for nm in names)
    if tot2 == total:
        print(f"tensor {dhash} is durable: {n} chunk(s), {total:,} bytes "
              f"confirmed on {TAG}")
        return 0
    print(f"::warning::uploaded but re-list shows {tot2:,} of {total:,} bytes")
    return 1


if __name__ == "__main__":
    sys.exit(main())

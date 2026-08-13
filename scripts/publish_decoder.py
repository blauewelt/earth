#!/usr/bin/env python3
"""Publish a retrained decoder (weights + audit JSON) to model-checkpoints-v1.

The dectrain window mode's last step. Same discipline as publish_tensor.py:
replace-don't-accumulate (DELETE then POST under a deterministic name),
`curl -T` streaming uploads, and an assert-the-EFFECT re-list at the end —
the upload only counts if the release afterwards holds both assets at the
local byte sizes. Exit codes are honest; best-effort belongs to the caller.

Usage (on a box, GITHUB_TOKEN in env):
  python3 scripts/publish_decoder.py --tag dec1536x3s0
publishes ml/runs/recon_decoder/dec1536x3s0.{json,pt}
as        dec1536x3s0__decoder.{json,pt}
"""
import argparse
import json
import os
import subprocess
import sys

REPO = os.environ.get("GITHUB_REPOSITORY", "blauewelt/earth")
TAG = "model-checkpoints-v1"
API = "https://api.github.com"


def sh(cmd):
    return subprocess.run(cmd, shell=True, text=True, capture_output=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True,
                    help="artefact stem, e.g. dec1536x3s0")
    ap.add_argument("--dir", default="ml/runs/recon_decoder")
    a = ap.parse_args()

    token = os.environ.get("GITHUB_TOKEN", "").strip()
    if not token:
        print("::error::no GITHUB_TOKEN — cannot publish")
        return 1
    hdr = (f'-H "Authorization: token {token}" '
           f'-H "Accept: application/vnd.github+json"')

    files = {}
    for ext in ("json", "pt"):
        p = os.path.join(a.dir, f"{a.tag}.{ext}")
        if not os.path.isfile(p) or os.path.getsize(p) == 0:
            print(f"::error::expected artefact {p} missing or empty")
            return 1
        files[f"{a.tag}__decoder.{ext}"] = p

    r = sh(f'curl -fsSL {hdr} "{API}/repos/{REPO}/releases/tags/{TAG}"')
    if r.returncode != 0:
        print(f"::error::cannot read release {TAG}: {r.stderr[:200]}")
        return 1
    rel = json.loads(r.stdout)
    rid = rel["id"]
    assets = {x["name"]: x for x in rel.get("assets", [])}

    for nm, p in files.items():
        if nm in assets:
            sh(f'curl -fsSL -X DELETE {hdr} '
               f'"{API}/repos/{REPO}/releases/assets/{assets[nm]["id"]}"')
        up = sh(f'curl -fsSL -X POST {hdr} '
                f'-H "Content-Type: application/octet-stream" -T "{p}" '
                f'"https://uploads.github.com/repos/{REPO}/releases/'
                f'{rid}/assets?name={nm}"')
        if up.returncode != 0:
            print(f"::error::upload of {nm} failed: {up.stderr[:200]}")
            return 1
        print(f"  uploaded {nm} ({os.path.getsize(p):,} bytes)")

    # Assert the effect: re-list, compare byte sizes.
    r2 = sh(f'curl -fsSL {hdr} "{API}/repos/{REPO}/releases/tags/{TAG}"')
    got = {x["name"]: x["size"]
           for x in json.loads(r2.stdout).get("assets", [])}
    ok = True
    for nm, p in files.items():
        want, have = os.path.getsize(p), got.get(nm, -1)
        print(f"  {nm}: local {want:,} / release {have:,}"
              f"{'' if want == have else '  MISMATCH'}")
        ok &= want == have
    if ok:
        print(f"decoder {a.tag} is durable on {TAG}")
        return 0
    print("::error::release re-list does not match the local artefacts")
    return 1


if __name__ == "__main__":
    sys.exit(main())

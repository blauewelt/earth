#!/usr/bin/env python3
"""Publish the E-069 cone samples to the Hugging Face Hub, and write the index.

`ml/export_cone_sample.py` writes one JSON per anchor — a few megabytes each,
which is far too much for git and exactly the size the Hub is for. This uploads
them to the dataset repo `chfrank/earth-tensors` under `cone_samples/` and
writes `data/cone_samples.json`, the small in-repo index the globe app reads
before it fetches anything.

THE TWO CHECKS THIS DOES AND WILL NOT SKIP, both of them `ml/CLAUDE.md` §0.2
("a step that reports success is not evidence it did anything"):

  1. **The restore.** Every file is downloaded BACK and sha256-compared against
     the bytes that were uploaded — `ml/hf_mirror.py`'s rule, for the same
     reason: an upload that returns 200 says nothing about retrievability.
  2. **CORS.** The browser reads these cross-origin, so a `resolve/main/...`
     URL that answers 200 to curl and has no `access-control-allow-origin`
     header is a file the page cannot use. The measured headers are printed
     and recorded in the index, because CLAUDE.md §3 admits `huggingface.co`
     on measured properties rather than assumed ones.

Credentials: `HF_TOKEN` in the environment, or `/home/claude/.hf_token`. Never
argv (the permission classifier blocks tokens on command lines, correctly).

Run:
    export HF_TOKEN=$(cat ~/.hf_token)
    python3 ml/upload_cone_samples.py --dir /home/claude/cone_out
"""
import argparse
import datetime as _dt
import hashlib
import json
import os
import sys
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
if HERE not in sys.path:
    sys.path.insert(0, HERE)

REPO_ID = "chfrank/earth-tensors"
REPO_TYPE = "dataset"
PREFIX = "cone_samples"
INDEX = os.path.join(ROOT, "data", "cone_samples.json")
TOKEN_FILE = "/home/claude/.hf_token"


def token():
    t = os.environ.get("HF_TOKEN")
    if t:
        return t.strip()
    if os.path.exists(TOKEN_FILE):
        return open(TOKEN_FILE, encoding="utf-8").read().strip()
    raise SystemExit("no HF_TOKEN in the environment and no "
                     f"{TOKEN_FILE} — see claude/huggingface-access.md")


def sha256(path, buf=1 << 22):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for blk in iter(lambda: fh.read(buf), b""):
            h.update(blk)
    return h.hexdigest()


def resolve_url(name):
    return (f"https://huggingface.co/datasets/{REPO_ID}/resolve/main/"
            f"{PREFIX}/{name}")


def measure_cors(url, origin="https://blauewelt.github.io"):
    """What a BROWSER on our origin would actually get back.

    A GET with an `Origin` header, following the redirect the Hub issues to its
    CDN, and reporting the CORS headers on whatever finally answers — because
    that response is the one the browser's fetch is subject to.
    """
    req = urllib.request.Request(url, headers={"Origin": origin})
    with urllib.request.urlopen(req, timeout=60) as r:
        h = {k.lower(): v for k, v in r.headers.items()}
        body = r.read()
        return dict(
            status=r.status, final_url=r.url, bytes=len(body),
            access_control_allow_origin=h.get("access-control-allow-origin"),
            access_control_expose_headers=h.get("access-control-expose-headers"),
            content_type=h.get("content-type"),
            sha256=hashlib.sha256(body).hexdigest(),
        )


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--dir", required=True,
                    help="directory of exported <anchor>.json files")
    ap.add_argument("--index", default=INDEX)
    ap.add_argument("--no-upload", action="store_true",
                    help="re-verify and rewrite the index without uploading")
    a = ap.parse_args(argv)

    from huggingface_hub import HfApi, hf_hub_download
    api = HfApi(token=token())
    who = api.whoami()
    print(f"hub user {who['name']} · isPro {who.get('isPro')}", flush=True)

    files = sorted(f for f in os.listdir(a.dir)
                   if f.endswith(".json") and not f.startswith("_"))
    if not files:
        raise SystemExit(f"{a.dir}: no <anchor>.json files")

    anchors, common = [], None
    for f in files:
        path = os.path.join(a.dir, f)
        s = json.load(open(path, encoding="utf-8"))
        m = s["meta"]
        src = sha256(path)
        size = os.path.getsize(path)
        name = f
        if not a.no_upload:
            print(f"  > {name} ({size / 1e6:.2f} MB) uploading …", flush=True)
            api.upload_file(path_or_fileobj=path,
                            path_in_repo=f"{PREFIX}/{name}",
                            repo_id=REPO_ID, repo_type=REPO_TYPE)
        # (1) the restore
        back = hf_hub_download(REPO_ID, f"{PREFIX}/{name}",
                               repo_type=REPO_TYPE, token=token())
        got = sha256(back)
        if got != src:
            raise SystemExit(
                f"{name}: uploaded sha256 {src} but downloaded {got} — the "
                f"mirror is not trustworthy")
        # (2) CORS, as a browser on our origin would see it
        cors = measure_cors(resolve_url(name))
        if cors["sha256"] != src:
            raise SystemExit(f"{name}: the resolve/ URL returned different "
                             f"bytes ({cors['sha256']} != {src})")
        print(f"    restored ✓ · HTTP {cors['status']} · "
              f"access-control-allow-origin: "
              f"{cors['access_control_allow_origin']!r}", flush=True)

        anchors.append(dict(
            id=m["anchor"]["id"], name=m["anchor"]["name"],
            lat=m["anchor"]["lat"], lon=m["anchor"]["lon"],
            row=m["anchor"]["row"], col=m["anchor"]["col"],
            file=f"{PREFIX}/{name}", url=resolve_url(name),
            bytes=size, sha256=src,
            n_inadmissible=m["n_inadmissible"],
            cors=cors["access_control_allow_origin"],
        ))
        if common is None:
            common = dict(
                tensor=m["tensor"], dates=m["dates"], bins=m["bins"],
                outer_lags=m["outer"]["lags"], outer=m["outer"],
                channels=m["channels"], scoreable_channels=m["scoreable_channels"],
                holdout_years=m["holdout_years"],
                terminal_train_last_year=m["terminal_train_last_year"],
                value_space={k: v for k, v in m["value_space"].items()
                             if k in ("raw", "anomaly", "anomaly_note")},
                exporter=m["exporter"], exporter_commit=m["exporter_commit"],
                produced_by=m["produced_by"], pentad_note=m["pentad_note"],
                L_in=m["L_in"], future_lags=m["future_lags"])
        else:
            if m["dates"] != common["dates"]:
                raise SystemExit(f"{name}: a different date list from the "
                                 f"first anchor — the index carries one")

    index = dict(
        _source="ml/upload_cone_samples.py — do not hand-edit",
        generated_utc=_dt.datetime.now(_dt.timezone.utc)
                        .strftime("%Y-%m-%dT%H:%M:%SZ"),
        repo=f"{REPO_TYPE}s/{REPO_ID}",
        base=f"https://huggingface.co/datasets/{REPO_ID}/resolve/main/{PREFIX}/",
        fixture="data/cone_samples/fixture.json",
        cors_measured=dict(
            origin="https://blauewelt.github.io",
            status=cors["status"],
            final_url_host=cors["final_url"].split("/")[2],
            access_control_allow_origin=cors["access_control_allow_origin"],
            access_control_expose_headers=cors["access_control_expose_headers"],
            measured_utc=_dt.datetime.now(_dt.timezone.utc)
                            .strftime("%Y-%m-%dT%H:%M:%SZ")),
        anchors=anchors,
        **common,
    )
    text = json.dumps(index, sort_keys=True, indent=1,
                      separators=(",", ": ")) + "\n"
    os.makedirs(os.path.dirname(os.path.abspath(a.index)), exist_ok=True)
    with open(a.index, "w", encoding="utf-8") as fh:
        fh.write(text)
    print(f"wrote {a.index} ({len(text)} bytes) · {len(anchors)} anchors · "
          f"{sum(x['bytes'] for x in anchors) / 1e6:.1f} MB on the Hub")
    return index


if __name__ == "__main__":
    main()

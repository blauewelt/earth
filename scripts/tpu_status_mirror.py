#!/usr/bin/env python3
"""Mirror TPU-run telemetry from gs://earth-tpu-staging into the ml-live-tpu
branch, so the status page can see TPU runs at all.

WHY THIS EXISTS. status.html's run list is built from ml-train.yml's Actions
runs plus the ml-live-<n>/ml-metrics branches — every one of them an artefact
of the Vast/Actions pipeline. A TPU run (E-051 was the first) has NO Actions
run, NO run number and NO git branch: its launcher ships metrics.jsonl and
checkpoints to the GCS bucket every ~10 minutes and nothing else. Chris,
2026-08-26: "I don't see E-051 anywhere on our status page." The bucket
cannot be read by the page directly — it is private and the tpu-runner SA is
deliberately not allowed to change bucket IAM (measured: 403 on
storage.buckets.getIamPolicy) — so the telemetry is MIRRORED into the git
surface the page already reads, on raw.githubusercontent.com, which is
CDN-served and not counted against the page's 60/h API budget.

WHAT IT WRITES, to refs/heads/ml-live-tpu as an ORPHAN, FORCE-updated ref
(the ml-live-* convention: ours alone, history is noise, the branch is a
mailbox not a ledger):

    index.json                      {"updated", "runs": [{node, exp, state,
                                     last, steps_total, metrics_updated, ...}]}
    <node>/metrics.jsonl            the run's own file, copied verbatim
    <node>/verify_report.json       when the run shipped one

`state` is derived from two measurements, never guessed: the TPU nodes API
says which nodes EXIST now (live) and the bucket says what each run last
shipped. exists + fresh metrics = TRAINING; exists + no metrics yet = SETUP;
gone + finals present (a *.pt beside the metrics) = FINISHED; gone without
finals = REAPED (the EXIT trap ships logs even on failure, so REAPED with no
log is itself a signal). A run whose directory exists is never dropped from
the index — the bucket is the durable record and the page is its reader.

The push is skipped when nothing changed: the previous index.json is fetched
from raw and compared field-for-field EXCLUDING the "updated" stamp, so a
quiet fleet produces zero commits rather than 96 stamps a day.

Runs in two places by design: GitHub Actions (cron, GCP_TPU_SA_KEY secret,
GITHUB_TOKEN push) and a session (SA key file + PAT). Stdlib only — the
Actions runner and the sandbox both have bare python3.

  GCP_SA_KEY=/path/sa.json GH_TOKEN=<token-or-path> python3 scripts/tpu_status_mirror.py
"""
import base64
import json
import os
import re
import sys
import urllib.error
import urllib.parse
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from tpu_box import mint_token, nodes_url  # noqa: E402

REPO = os.environ.get("REPO", "blauewelt/earth")
BUCKET = os.environ.get("BUCKET", "earth-tpu-staging")
PROJECT = os.environ.get("GCP_PROJECT", "earth-tpu-blauewelt")
ZONE = os.environ.get("GCP_ZONE", "us-west1-c")
# Nodes live in whichever zone had capacity (the spot ladder spans five
# zones), but this mirror used to ask ONLY $GCP_ZONE — so a node in any
# other zone read as "gone" and, with *.pt checkpoints in the bucket, was
# labeled FINISHED mid-run. Measured 2026-08-29: e059-window TRAINING at
# step 150k in us-west4-a while index.json said FINISHED. Ask every zone
# the launcher can use; a zone that errors is skipped with a warning
# rather than killing the mirror.
ZONES = [z.strip() for z in os.environ.get(
    "GCP_ZONES",
    ",".join(dict.fromkeys([ZONE, "us-west1-c", "us-west4-a", "us-west4-b",
                            "us-central1-a", "us-east5-a", "us-east5-b",
                            "us-east5-c"]))).split(",") if z.strip()]
BRANCH = "ml-live-tpu"
GS = "https://storage.googleapis.com/storage/v1/b/%s" % BUCKET
# metrics.jsonl is copied verbatim up to this cap; past it, the TAIL is kept
# (the page reads the last lines; a runaway file must not break the mirror).
METRICS_CAP = 2 * 1024 * 1024


def http(method, url, headers=None, body=None, ok404=False):
    req = urllib.request.Request(url, headers=headers or {}, method=method,
                                 data=body)
    try:
        with urllib.request.urlopen(req, timeout=60) as r:
            return r.read()
    except urllib.error.HTTPError as e:
        if e.code == 404 and ok404:
            return None
        raise SystemExit(f"{method} {url.split('?')[0]} -> HTTP {e.code}: "
                         f"{e.read().decode(errors='replace')[:300]}")


def gcs_list(tok, prefix):
    out, page = [], ""
    while True:
        url = (f"{GS}/o?prefix={prefix}&fields=items(name,size,updated),"
               f"nextPageToken&maxResults=1000{page}")
        d = json.loads(http("GET", url, {"Authorization": "Bearer " + tok}))
        out += d.get("items", [])
        if not d.get("nextPageToken"):
            return out
        page = "&pageToken=" + d["nextPageToken"]


def gcs_read(tok, name):
    url = f"{GS}/o/{urllib.parse.quote(name, safe='')}?alt=media"
    return http("GET", url, {"Authorization": "Bearer " + tok}, ok404=True)


def live_nodes(tok):
    out = {}
    for zone in ZONES:
        try:
            d = json.loads(http("GET", nodes_url(PROJECT, zone),
                                {"Authorization": "Bearer " + tok},
                                ok404=True) or b"{}")
        except SystemExit as e:
            print(f"warning: node list failed for {zone}: {e}",
                  file=sys.stderr)
            continue
        for n in d.get("nodes", []):
            out[n["name"].rsplit("/", 1)[-1]] = n
    return out


def gh_token():
    t = os.environ.get("GH_TOKEN") or os.environ.get("GITHUB_TOKEN")
    if not t:
        raise SystemExit("GH_TOKEN/GITHUB_TOKEN unset")
    if os.path.exists(t):
        t = open(t).read().strip()
    return t


def gh(method, path, tok, body=None, ok404=False):
    raw = http(method, "https://api.github.com" + path,
               {"Authorization": "Bearer " + tok,
                "Accept": "application/vnd.github+json",
                "User-Agent": "tpu-status-mirror"},
               json.dumps(body).encode() if body is not None else None,
               ok404=ok404)
    return json.loads(raw) if raw else None


def last_metric_line(blob):
    lines = [l for l in blob.decode(errors="replace").strip().split("\n") if l]
    for l in reversed(lines):
        try:
            return json.loads(l)
        except json.JSONDecodeError:
            continue
    return {}


def main():
    tok = mint_token(os.environ.get("GCP_SA_KEY", "/home/claude/.gcp_sa.json"))
    tok = tok if isinstance(tok, str) else tok[0]
    objs = gcs_list(tok, "runs/")
    nodes = live_nodes(tok)

    by_run = {}
    for o in objs:
        m = re.match(r"runs/([^/]+)/(.+)$", o["name"])
        if m:
            by_run.setdefault(m.group(1), {})[m.group(2)] = o

    runs, files = [], {}
    for node, fs in sorted(by_run.items()):
        met = fs.get("metrics.jsonl")
        finals = [k for k in fs if k.endswith(".pt") or k.endswith(".npz")]
        exists = node in nodes
        if exists:
            state = "TRAINING" if met else "SETUP"
            if str(nodes[node].get("health", "")).startswith("UNHEALTHY"):
                state += " (UNHEALTHY)"
        else:
            state = "FINISHED" if finals else "REAPED"
        rec = {"node": node,
               # e051-k144-full -> E-051; a node named outside the eNNN
               # convention keeps its own name as the label.
               "exp": (m.group(1).upper().replace("E0", "E-0", 1)
                       if (m := re.match(r"(e\d+)", node)) else node),
               "state": state,
               "metrics_updated": met["updated"] if met else None,
               "finals": sorted(finals)[:6]}
        if met:
            blob = gcs_read(tok, f"runs/{node}/metrics.jsonl") or b""
            if len(blob) > METRICS_CAP:
                blob = blob[-METRICS_CAP:].split(b"\n", 1)[-1]
            files[f"{node}/metrics.jsonl"] = blob
            rec["last"] = last_metric_line(blob)
        vr = fs.get("verify_report.json")
        if vr:
            files[f"{node}/verify_report.json"] = \
                gcs_read(tok, f"runs/{node}/verify_report.json") or b"{}"
            rec["verify_report"] = True
        runs.append(rec)

    index = {"runs": runs}

    # Skip the push when nothing but the clock moved.
    prev = http("GET", f"https://raw.githubusercontent.com/{REPO}/{BRANCH}/"
                       f"index.json", ok404=True)
    if prev is not None:
        try:
            p = json.loads(prev)
            p.pop("updated", None)
            if p == index:
                print("unchanged — no push")
                return
        except json.JSONDecodeError:
            pass
    import datetime
    index["updated"] = datetime.datetime.now(
        datetime.timezone.utc).isoformat(timespec="seconds")
    files["index.json"] = json.dumps(index, indent=1).encode()

    # --dump <dir>: write the files and stop. The session sandbox's egress
    # proxy rejects python POSTs to api.github.com (ml/CLAUDE.md §7 — node
    # passes, python does not), so a session runs the GCS half here and
    # pushes the dump with scripts/push_tree.mjs; GitHub Actions has no such
    # proxy and takes the direct python path below.
    if "--dump" in sys.argv:
        out = sys.argv[sys.argv.index("--dump") + 1]
        for path, blob in files.items():
            p = os.path.join(out, path)
            os.makedirs(os.path.dirname(p), exist_ok=True)
            with open(p, "wb") as fh:
                fh.write(blob)
        print(f"dumped {len(files)} file(s) to {out}")
        return

    g = gh_token()
    tree = []
    for path, blob in sorted(files.items()):
        sha = gh("POST", f"/repos/{REPO}/git/blobs", g,
                 {"content": base64.b64encode(blob).decode(),
                  "encoding": "base64"})["sha"]
        tree.append({"path": path, "mode": "100644", "type": "blob",
                     "sha": sha})
    tsha = gh("POST", f"/repos/{REPO}/git/trees", g, {"tree": tree})["sha"]
    csha = gh("POST", f"/repos/{REPO}/git/commits", g,
              {"message": f"tpu mirror {index['updated']} "
                          f"({len(runs)} run(s))",
               "tree": tsha, "parents": []})["sha"]
    # Orphan + force is the ml-live-* convention: the branch is a mailbox.
    if gh("PATCH", f"/repos/{REPO}/git/refs/heads/{BRANCH}", g,
          {"sha": csha, "force": True}, ok404=True) is None:
        gh("POST", f"/repos/{REPO}/git/refs", g,
           {"ref": f"refs/heads/{BRANCH}", "sha": csha})
    print(f"pushed {BRANCH} @ {csha[:9]}: {len(runs)} run(s), "
          f"{len(files)} file(s)")


if __name__ == "__main__":
    main()

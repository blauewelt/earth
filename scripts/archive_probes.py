#!/usr/bin/env python3
"""Put a run's PROBE RESULTS somewhere permanent, from the box itself.

This is a port of scripts/archive_probes.mjs, and the reason for the port is
the whole point of the file: **the Vast boxes have no `node`.** The .mjs
version was added on 2026-08-10 to stop probe results expiring with their
30-day artifact, was called from the workflow, and every single invocation
died with `node: command not found` — logged as a warning nobody read, on a
step written to be best-effort. It never archived anything.

That is the same failure as `gh release download` on these boxes (ml/CLAUDE.md
§7): a tool assumed present because it is present everywhere else. The boxes
have Python — the entire pipeline is Python — so this uses nothing but the
standard library.

Three storage tiers exist and the scientific output was in the wrong one:

  ml-metrics branch   permanent   training CURVES (loss, lr, stage-2 steps)
  GitHub releases     permanent   checkpoints, tensors, embeddings
  Actions artifacts   30 DAYS     probe_kfold, temporal, dip_check …

The last row is the row with the answers in it.

  python3 scripts/archive_probes.py --run-number 131 --dir ml/runs/actions
"""
import argparse
import base64
import json
import os
import sys
import urllib.error
import urllib.request

REPO = os.environ.get("GITHUB_REPOSITORY", "blauewelt/earth")
BRANCH = "ml-metrics"

# temporal.json carries the STAGE-2 answer — rapid_probe, rapid_probe_kfold,
# and the forecast MSEs against persistence. probe_kfold.json scores the
# CODEC and is identical for every run freezing the same codec, so for a
# stage-2 sweep it is the control rather than the result. Both belong here.
WANT = ["probe_kfold.json", "temporal.json", "rollout_eval.json",
        "probe_sequence.json", "project_amoc.json", "rollout_spatial.json",
        "dip_check.json", "probe_head.json", "probe_head_raw3x3.json",
        "probe_head_raw.json", "rollout.json", "provenance.json"]

# ...AND EVERY OTHER probe_head*.json. Since 2026-08-21 that family is
# generated rather than enumerated: --target adds a suffix per transport
# series and --wind-only adds the head's own unpooled BAR, so the three names
# above are only the RAPID defaults. A hand-kept list would have archived an
# unpooled head number and silently dropped the unpooled bar it must be read
# against, which is the comparison ml/CLAUDE.md §3 exists to protect. The
# explicit names stay so a missing one is still REPORTED as missing.
WANT_GLOB = ["probe_head*.json"]


def api(url, token, method="GET", payload=None):
    data = json.dumps(payload).encode() if payload is not None else None
    req = urllib.request.Request(url, data=data, method=method)
    # `token`, not `Bearer` — matches every other script in this repo and the
    # job token accepts it.
    req.add_header("Authorization", f"token {token}")
    req.add_header("Accept", "application/vnd.github+json")
    if data:
        req.add_header("Content-Type", "application/json")
    try:
        with urllib.request.urlopen(req, timeout=120) as r:
            return r.status, json.loads(r.read().decode() or "{}")
    except urllib.error.HTTPError as e:
        body = e.read().decode()[:300]
        try:
            return e.code, json.loads(body or "{}")
        except json.JSONDecodeError:
            return e.code, {"raw": body}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-number", required=True)
    ap.add_argument("--dir", required=True)
    a = ap.parse_args()

    token = os.environ.get("GITHUB_TOKEN", "").strip()
    if not token:
        print("::warning::no GITHUB_TOKEN — cannot archive the probe results")
        return 1

    # THE PROBES ARE NOT ALL IN ONE DIRECTORY. probe_kfold.py writes
    # ml/runs/probe_kfold.json, one level ABOVE the per-run directory, because
    # it sweeps several runs at once; everything else writes into
    # ml/runs/actions/. A single --dir silently dropped the codec control.
    dirs = [a.dir, os.path.join(a.dir, "..")]
    bundle = {"run_number": int(a.run_number), "files": {}}
    missing = []
    import glob as _glob
    extra = sorted({os.path.basename(p) for pat in WANT_GLOB for d in dirs
                    for p in _glob.glob(os.path.join(d, pat))} - set(WANT))
    if extra:
        print(f"  glob {WANT_GLOB} also matched: {', '.join(extra)}")
    for f in WANT + extra:
        path = next((p for p in (os.path.join(d, f) for d in dirs)
                     if os.path.exists(p)), None)
        if not path:
            missing.append(f)
            continue
        try:
            with open(path) as fh:
                bundle["files"][f] = json.load(fh)
        except (OSError, json.JSONDecodeError) as e:
            print(f"  skipping {f}: {e}")
    if not bundle["files"]:
        print(f"::warning::no probe results under {' or '.join(dirs)}")
        return 1
    print(f"bundling {len(bundle['files'])} file(s): "
          f"{', '.join(bundle['files'])}")
    if missing:
        print(f"  not present: {', '.join(missing)}")
    if "probe_kfold.json" in missing:
        print("::warning::no probe_kfold.json — this bundle has no CODEC "
              "control, so its head numbers cannot be checked against the "
              "codec being held fixed")

    path = f"probes-{a.run_number}.json"
    url = f"https://api.github.com/repos/{REPO}/contents/{path}"
    st, cur = api(f"{url}?ref={BRANCH}", token)
    sha = cur.get("sha") if st == 200 else None

    body = {"message": f"archive probe results for run #{a.run_number}",
            "content": base64.b64encode(
                json.dumps(bundle, indent=1).encode()).decode(),
            "branch": BRANCH}
    if sha:
        body["sha"] = sha
    st, res = api(url, token, method="PUT", payload=body)
    if st not in (200, 201):
        print(f"::warning::archive failed: {st} {str(res)[:200]}")
        return 1
    print(f"archived: https://github.com/{REPO}/blob/{BRANCH}/{path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

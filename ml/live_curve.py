#!/usr/bin/env python3
"""Plot the training curve of a run that is STILL RUNNING on Actions.

The ml-train workflow publishes ml/runs/actions/metrics.jsonl to the
orphan branch `ml-live-<run_number>` every 5 minutes while train.py is
alive (scripts/publish_live_metrics.sh). This tool fetches that file into
runs/live<N>/ and renders it through the same plot_run.py path as any
finished run — one code path for every curve, live or done.

Usage:  python3 ml/live_curve.py --run-number 9
        → ml/runs/live9/curve.png (and prints progress: last step, loss,
          latest probe point — enough to judge "promising or not")
"""
import argparse
import json
import os
import subprocess
import sys
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
UA = {"User-Agent": "earth-live-curve/1.0"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-number", type=int, required=True)
    ap.add_argument("--repo", default="blauewelt/earth")
    a = ap.parse_args()
    name = f"live{a.run_number}"
    url = (f"https://raw.githubusercontent.com/{a.repo}/"
           f"ml-live-{a.run_number}/metrics.jsonl")
    run_dir = os.path.join(HERE, "runs", name)
    os.makedirs(run_dir, exist_ok=True)
    try:
        req = urllib.request.Request(url, headers=UA)
        body = urllib.request.urlopen(req, timeout=60).read()
    except Exception as e:
        sys.exit(f"no live metrics for run #{a.run_number} ({e}) — the run "
                 f"may predate live publishing, not have reached its first "
                 f"5-minute push, or already be finished (use the artifact).")
    open(os.path.join(run_dir, "metrics.jsonl"), "wb").write(body)

    last_loss, last_probe, n_loss = None, None, 0
    for line in body.decode().splitlines():
        m = json.loads(line)
        if "loss_rec" in m:
            last_loss, n_loss = m, n_loss + 1
        elif "linear_r_deseas" in m:
            last_probe = m
    if last_loss:
        print(f"run #{a.run_number}: step {last_loss.get('step')} · "
              f"rec {last_loss.get('loss_rec'):.4f} · "
              f"nei {last_loss.get('loss_nei', float('nan')):.4f} "
              f"({n_loss} loss points)")
    if last_probe:
        print(f"  latest probe @ step {last_probe.get('step')}: "
              f"chan {last_probe.get('chan_vs_persistence_pct')}% · "
              f"r_lin {last_probe.get('linear_r_deseas')} · "
              f"r_tmp {last_probe.get('temporal_r_deseas')}")
    subprocess.run([sys.executable, os.path.join(HERE, "plot_run.py"),
                    "--run", name], check=True)
    print(os.path.join(run_dir, "curve.png"))


if __name__ == "__main__":
    main()

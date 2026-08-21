#!/usr/bin/env python3
"""Assemble the master comparison table from the runs' own JSON.

The master table in LEADERBOARD.md was transcribed by hand across a dozen
runs and five metric files each. That is precisely the sort of arithmetic a
person gets wrong once and then quotes forever, so it is generated here
instead — every number read from the file the measuring script wrote.

Columns are chosen to answer the two questions the programme is actually
asking, side by side and with the confounds visible:

  · what the CODEC alone can do — k-fold RAPID r from single-month
    embeddings (with its bootstrap CI), and reconstruction skill;
  · what the frozen codec plus DYNAMICS can do — stage-2 z-space and
    decoded channel-space gains over persistence, and the transport probe
    from the transformer's section state;
  · and the two things that make runs incomparable if left unstated —
    the CHANNEL COUNT and the STEP COUNT. A 40k-step run next to a 15k-step
    one is not a fair fight, and the table has to say so rather than let a
    reader assume otherwise.

Usage:
    python3 ml/make_table.py                      # every run in ml/runs
    python3 ml/make_table.py --runs global14b global25 patch14
    python3 ml/make_table.py --markdown >> ml/LEADERBOARD.md
"""
import argparse
import json
import os
import sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
RUNS = os.path.join(HERE, "runs")


def load(run, name):
    p = os.path.join(RUNS, run, name)
    if not os.path.exists(p):
        return None
    try:
        with open(p) as f:
            return json.load(f)
    except json.JSONDecodeError:
        return None


def kfold_of(run, target="rapid"):
    """probe_kfold writes ONE file for all runs it was asked about, and
    older runs also carry a per-run copy — check both, per-run first.

    THIS IS THE LEGACY POOLED NUMBER. It reads `Z.mean(1)` over the section
    (ml/probe_kfold.py) and every block it returns carries `probe:
    "pooled-ridge"` saying so. It stays in the table because it is the
    comparability bridge to a 107-run k-fold column and 183 archived
    bundles; it is not the verdict (ml/CLAUDE.md §3, 2026-08-21).
    """
    for path in (os.path.join(RUNS, run, "probe_kfold.json"),
                 os.path.join(RUNS, "probe_kfold.json")):
        if not os.path.exists(path):
            continue
        with open(path) as f:
            d = json.load(f)
        block = d.get(run, d).get(target) if isinstance(d, dict) else None
        if block:
            return block
    return None


def head_of(run, kind="", target="rapid"):
    """ml/probe_head.py's UNPOOLED number, or None.

    THIS FILE HAD NO probe_head CODE PATH AT ALL until 2026-08-21, which is
    why the master table it generates has never carried an unpooled column —
    the one read-out ml/CLAUDE.md §3 now calls the verdict was invisible to
    the surface that assembles the verdicts.

    `kind` selects the rung: "" the head over the codec embedding,
    "raw3x3" its matched receptive-field control, "wind" its matched
    UNPOOLED BAR. RAPID keeps the historical file names exactly; another
    target adds a suffix.
    """
    tsfx = "" if target == "rapid" else f"_{target}"
    return load(run, f"probe_head{'_' + kind if kind else ''}{tsfx}.json")


def row(run):
    ck_path = os.path.join(RUNS, run, "pixelmae.pt")
    if not os.path.exists(ck_path):
        return None
    ck = torch.load(ck_path, map_location="cpu", weights_only=False)
    a = ck.get("args", {})
    tp = load(run, "trainprobe.json") or {}
    tm = load(run, "temporal.json") or {}
    kf = kfold_of(run) or {}
    ev = load(run, "eval.json") or {}
    hd = head_of(run) or {}                    # THE VERDICT: unpooled
    hd3 = head_of(run, "raw3x3") or {}         # its receptive-field control
    hdw = head_of(run, "wind") or {}           # its matched UNPOOLED bar

    def g(d, *path, default=None):
        for k in path:
            if not isinstance(d, dict) or k not in d:
                return default
            d = d[k]
        return d

    ci = kf.get("ci95") or [None, None]
    hci = hd.get("ci95") or [None, None]
    return {
        "run": run,
        "C": len(ck.get("chan", [])),
        "d_z": ck.get("d_z"),
        "patch": a.get("patch", 1),
        "steps": a.get("steps"),
        # ---- THE VERDICT: unpooled, with its own unpooled bars -----------
        # A blank here is UNAVAILABLE, never back-filled from the pooled
        # column beside it. A pooled number wearing an unpooled heading is
        # the one thing worse than a gap, because a gap can be noticed.
        "head_r": hd.get("r_kfold_deseas"),
        "head_ci": (f"[{hci[0]:+.2f},{hci[1]:+.2f}]"
                    if hci[0] is not None else "—"),
        "head_raw3x3": hd3.get("r_kfold_deseas"),
        "head_wind": hdw.get("r_kfold_deseas"),
        # ---- LEGACY, retained as the comparability bridge ----------------
        "legacy_pooled_kfold_r": kf.get("r_kfold_deseas"),
        "ci": f"[{ci[0]:+.2f},{ci[1]:+.2f}]" if ci[0] is not None else "—",
        "rmse": kf.get("rmse_sv"),
        "legacy_pooled_wind": g(kf, "wind_only_baseline", "r"),
        "lin_r": tp.get("linear_r_deseas"),
        "chan_pct": tp.get("chan_vs_persistence_pct"),
        "s2_z": g(tm, "z_t+1", "beats_persistence"),
        "s2_chan": g(tm, "chan_t+1", "pct_vs_persistence",
                     default=g(tm, "chan_t+1", "beats_persistence")),
        "s2_r": g(tm, "rapid_probe", "r_deseasonalised"),
        "codec_r": g(ev, "rapid_probe", "pearson_heldout_years"),
    }


def fmt(v, n=3):
    if v is None:
        return "—"
    if isinstance(v, bool):
        return "yes" if v else "no"
    if isinstance(v, float):
        return f"{v:+.{n}f}" if abs(v) < 10 else f"{v:.{n}f}"
    return str(v)


def _verdict_note(n_rows, n_head, n_bar):
    """The one paragraph that says which column is the answer.

    Printed under EVERY rendering, including the plain-text one, because the
    table is copied out of a terminal at least as often as it is piped into
    LEADERBOARD.md, and a column heading alone has never stopped anyone
    quoting the number in it.
    """
    lines = [
        "",
        "UNPOOLED IS THE VERDICT (ml/CLAUDE.md §3, 2026-08-21). "
        "`UNPOOLED head r` is ml/probe_head.py — one learned query attending "
        "over the section's individual pixels. Geostrophic transport at "
        "26.5N is the east-minus-west contrast ACROSS the section, and the "
        "`legacy_pooled_*` columns read Z.mean(1) over it: measured on the "
        "run-62 cache (ml/project_amoc.py), z correlates r 0.99 at one cell, "
        "0.88 at five and 0.35 at eighty, so that mean averages ~2.5 "
        "effective independent pixels of 265.",
        "",
        "READ THE BARS THAT SIT BESIDE EACH NUMBER, NOT ACROSS THE TABLE. "
        "`raw3x3 bar` and `wind bar` are unpooled and belong to the head; "
        "`legacy_pooled wind` is pooled and belongs to the pooled ridge. "
        "Comparing an unpooled head against a pooled bar measures the "
        "read-out and reports it as the codec.",
        "",
        f"COVERAGE: {n_head}/{n_rows} rows carry an unpooled head, "
        f"{n_bar}/{n_rows} carry its unpooled wind bar. A dash is "
        f"UNAVAILABLE — these runs predate `head_probe: true` becoming the "
        f"workflow default, and a pooled number has NOT been copied into the "
        f"unpooled column to fill the gap.",
        "",
        "The pooled columns are kept, not deprecated: they are the only "
        "thing that makes 183 archived bundles readable next to a new run. "
        "They are never a verdict.",
    ]
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="*", default=None)
    ap.add_argument("--markdown", action="store_true")
    a = ap.parse_args()
    runs = a.runs or sorted(
        r for r in os.listdir(RUNS)
        if os.path.exists(os.path.join(RUNS, r, "pixelmae.pt")))
    rows = [r for r in (row(x) for x in runs) if r]
    if not rows:
        sys.exit("no runs with checkpoints found")

    # COLUMN ORDER IS THE CLAIM. The unpooled head and its two unpooled bars
    # come first and are the verdict; everything prefixed legacy_pooled_ is
    # the comparability bridge to the archive and is never read as one
    # (ml/CLAUDE.md §3, 2026-08-21). Both bars sit BESIDE the number they
    # bar, so nobody has to know which of two columns is the matched one.
    head = ["run", "C", "d_z", "patch", "steps",
            "UNPOOLED head r ↑", "CI95", "raw3x3 bar", "wind bar",
            "legacy_pooled r", "legacy CI95", "legacy RMSE Sv",
            "legacy_pooled wind", "linear r", "chan% t+1",
            "legacy_pooled stage2 r"]
    keys = ["run", "C", "d_z", "patch", "steps",
            "head_r", "head_ci", "head_raw3x3", "head_wind",
            "legacy_pooled_kfold_r", "ci", "rmse",
            "legacy_pooled_wind", "lin_r", "chan_pct", "s2_r"]
    PRE = ("ci", "head_ci")             # already formatted strings
    table = [[r[k] if k in PRE else fmt(r[k]) for k in keys] for r in rows]
    n_head = sum(r["head_r"] is not None for r in rows)
    n_bar = sum(r["head_wind"] is not None for r in rows)

    if a.markdown:
        print("| " + " | ".join(head) + " |")
        print("|" + "|".join("---" for _ in head) + "|")
        for t in table:
            print("| " + " | ".join(t) + " |")
        print("\n*Generated by `python3 ml/make_table.py --markdown`. "
              "`steps` and `C` are shown because they are what makes two rows "
              "incomparable: a 40k-step codec beside a 15k-step one is a "
              "difference in schedule, not in architecture.*")
        print(_verdict_note(len(rows), n_head, n_bar))
    else:
        w = [max(len(head[i]), *(len(t[i]) for t in table)) for i in range(len(head))]
        print("  ".join(h.ljust(w[i]) for i, h in enumerate(head)))
        for t in table:
            print("  ".join(c.ljust(w[i]) for i, c in enumerate(t)))
        print(_verdict_note(len(rows), n_head, n_bar))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Render a run's training curve: steps vs loss, and steps vs held-out
predictive metrics. One PNG per run — `runs/<run>/curve.png`.

Data sources, most→least preferred (panels appear only if data exists):
  · runs/<run>/metrics.jsonl   — loss + probe entries (train.py --eval-every)
  · runs/<run>.log or runs/<run>/train.log — "step N/M rec X nei Y" lines,
    reconstructs the LOSS curve for runs that predate metrics.jsonl
  · runs/<run>/trainprobe.json — one backfilled probe point (final step)
  · runs/<run>/eval.json       — final t+1 (converted to % vs persistence)

Layout per the dataviz method: three stacked panels sharing the step axis
(loss · t+1 %-vs-persistence · RAPID probe r) — separate panels, never a
dual axis. Categorical slots in fixed order; text in ink, not series color.

Usage: python3 ml/plot_run.py --run pilot4_anom [--publish]
       (--publish also copies to ml/curves/<run>.png — the committed home
        for curves of runs whose containers are ephemeral)
"""
import argparse
import json
import os
import re
import shutil

HERE = os.path.dirname(os.path.abspath(__file__))

# validated categorical palette (light mode), fixed slot order
C1, C2 = "#2a78d6", "#eb6834"          # slots 1-2, reused per panel (each
INK, INK2, GRID = "#1a1a19", "#6f6e66", "#e5e4de"   # panel is its own chart)


def load_points(run):
    run_dir = os.path.join(HERE, "runs", run)
    loss, probe = [], []
    mj = os.path.join(run_dir, "metrics.jsonl")
    if os.path.exists(mj):
        for line in open(mj):
            m = json.loads(line)
            if "loss_rec" in m:
                loss.append(m)
            if "chan_vs_persistence_pct" in m:
                probe.append(m)
    if not loss:                       # runs that predate the jsonl: parse log
        for cand in (os.path.join(HERE, "runs", f"{run}.log"),
                     os.path.join(run_dir, "train.log")):
            if os.path.exists(cand):
                for ln in open(cand):
                    m = re.search(r"step\s+(\d+)/\d+\s+rec ([\d.]+)\s+nei ([\d.]+)", ln)
                    if m:
                        loss.append({"step": int(m[1]), "loss_rec": float(m[2]),
                                     "loss_nei": float(m[3])})
                break
    if not probe:                      # backfilled single point at final step
        tp = os.path.join(run_dir, "trainprobe.json")
        if os.path.exists(tp):
            m = json.load(open(tp))
            m["step"] = max((p["step"] for p in loss), default=0) or None
            probe.append(m)
    return loss, probe


def render(run, publish=False):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    loss, probe = load_points(run)
    panels = (1 if loss else 0) + (2 if probe else 0)
    if not panels:
        raise SystemExit(f"no curve data for {run}")
    n = (1 if loss else 0) + (1 if probe else 0) * 2
    fig, axes = plt.subplots(n, 1, figsize=(7.2, 2.1 * n + 0.7), sharex=True,
                             constrained_layout=True)
    axes = [axes] if n == 1 else list(axes)
    fig.suptitle(f"{run} — training curve (blocked holdout, protocol v2)",
                 fontsize=11, color=INK, x=0.02, ha="left")
    for ax in axes:
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        for s in ("left", "bottom"):
            ax.spines[s].set_color(GRID)
        ax.grid(True, color=GRID, linewidth=0.6)
        ax.tick_params(colors=INK2, labelsize=8)

    i = 0
    if loss:
        ax = axes[i]; i += 1
        xs = [p["step"] for p in loss]

        def ema(vals, a=0.85):
            out, m = [], vals[0]
            for v in vals:
                m = a * m + (1 - a) * v
                out.append(m)
            return out

        for key, col, lab in (("loss_rec", C1, "reconstruction"),
                              ("loss_nei", C2, "neighbour")):
            ys = [p[key] for p in loss]
            if len(ys) > 30:                   # dense curve: raw faint + EMA
                ax.plot(xs, ys, color=col, lw=0.8, alpha=0.25)
                ax.plot(xs, ema(ys), color=col, lw=2, label=lab)
            else:                              # sparse (parsed old log): raw
                ax.plot(xs, ys, color=col, lw=2, label=lab)
        ax.set_ylabel("training loss", fontsize=9, color=INK2)
        ax.legend(frameon=False, fontsize=8, labelcolor=INK2)
    if probe:
        xs = [p["step"] for p in probe]
        marker = "o" if len(probe) == 1 else None
        ax = axes[i]; i += 1
        ax.axhline(0, color=INK2, lw=1, ls=":")
        ax.plot(xs, [p["chan_vs_persistence_pct"] for p in probe], color=C1,
                lw=2, marker=marker, ms=6, label="channel space")
        ax.plot(xs, [p["z_vs_persistence_pct"] for p in probe], color=C2,
                lw=2, marker=marker, ms=6, label="embedding space")
        ax.text(0.995, 0.06, "0 = persistence", transform=ax.transAxes,
                ha="right", fontsize=7.5, color=INK2)
        ax.set_ylabel("held-out t+1\n% better than\npersistence", fontsize=9, color=INK2)
        ax.legend(frameon=False, fontsize=8, labelcolor=INK2)

        ax = axes[i]; i += 1
        ax.axhspan(-0.17, 0.17, color=GRID, alpha=0.5, lw=0)
        ax.axhline(0, color=INK2, lw=1, ls=":")
        ax.plot(xs, [p["linear_r_deseas"] for p in probe], color=C1,
                lw=2, marker=marker, ms=6, label="linear probe")
        ax.plot(xs, [p["temporal_r_deseas"] for p in probe], color=C2,
                lw=2, marker=marker, ms=6, label="mini temporal")
        ax.text(0.995, 0.06, "band: ±1σ sampling noise (36 held-out months)",
                transform=ax.transAxes, ha="right", fontsize=7.5, color=INK2)
        ax.set_ylabel("RAPID probe\nr (deseason.)", fontsize=9, color=INK2)
        ax.set_ylim(min(-0.2, *[p["temporal_r_deseas"] for p in probe]) - 0.05, 0.75)
        ax.legend(frameon=False, fontsize=8, labelcolor=INK2)
    axes[-1].set_xlabel("training steps", fontsize=9, color=INK2)

    out = os.path.join(HERE, "runs", run, "curve.png")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    fig.savefig(out, dpi=150, facecolor="white")
    print(f"wrote {out}  (loss pts: {len(loss)} · probe pts: {len(probe)})")
    if publish:
        pub = os.path.join(HERE, "curves", f"{run}.png")
        os.makedirs(os.path.dirname(pub), exist_ok=True)
        shutil.copyfile(out, pub)
        print(f"published {pub}")
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--publish", action="store_true")
    a = ap.parse_args()
    render(a.run, publish=a.publish)

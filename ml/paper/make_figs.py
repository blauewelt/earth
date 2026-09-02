#!/usr/bin/env python3
"""Figures for the technical report (ml/paper/paper.tex).

Stage-2 series are read from data/report_data.json, which extract_data.py
builds from the public result bundles (probes-516 on the ml-metrics branch)
and the stage-2 training records. Codec probe numbers carry a `# src:`
comment naming the archived result they are transcribed from.

Run from anywhere:  python3 ml/paper/make_figs.py [--dark]
--dark writes the same figures for the dark page to figs_dark/.
"""
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DARK = "--dark" in sys.argv
FIGS = os.path.join(HERE, "figs_dark" if DARK else "figs")
os.makedirs(FIGS, exist_ok=True)
D = json.load(open(os.path.join(HERE, "data", "report_data.json")))

if DARK:
    BG = "#14140f"
    INK, INK2, GRID = "#e8e6df", "#a5a396", "#3a3a35"
    C = ["#5b9ee8", "#f08a5c", "#5fba82", "#c58be0", "#e0c05b", "#7fd3d9", "#e07f9b", "#a5a396"]
else:
    BG = "white"
    INK, INK2, GRID = "#1a1a19", "#6f6e66", "#e5e4de"
    C = ["#2a78d6", "#eb6834", "#3a9a5c", "#8e4bc7", "#b8900c", "#1e9aa6", "#c94d6f", "#6f6e66"]

plt.rcParams.update({
    "figure.facecolor": BG, "savefig.facecolor": BG, "axes.facecolor": BG,
    "figure.dpi": 150, "savefig.bbox": "tight",
    "axes.edgecolor": INK2, "axes.labelcolor": INK,
    "text.color": INK, "xtick.color": INK2, "ytick.color": INK2,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.7,
    "axes.axisbelow": True, "font.size": 9, "legend.frameon": False,
    "lines.linewidth": 1.6,
})


def strip(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


def save(fig, name):
    fig.savefig(os.path.join(FIGS, name))
    plt.close(fig)
    print("wrote", os.path.join(FIGS, name))


STEP_DAYS = 5

# ---- Fig 1: the codec's transport read-out, pooled, across generations -----
# src: LEADERBOARD.md k-fold table + probe_kfold.json wind-only baseline
runs = [("12-ch, $d_z$=8", 0.111, 0.01, 0.20, C[0]),
        ("12-ch, $d_z$=16", 0.151, 0.01, 0.28, C[0]),
        ("12-ch, $d_z$=32", 0.182, 0.05, 0.31, C[0]),
        ("12-ch, $d_z$=128", 0.166, 0.072, 0.295, C[0]),
        ("12-ch, $d_z$=64", 0.308, 0.13, 0.46, C[0]),
        ("25-ch, pixel", 0.536, 0.378, 0.683, C[2]),
        ("24-ch, 3×3 patch", 0.543, 0.428, 0.659, C[2]),
        ("14-ch + wind, NA window", 0.604, 0.474, 0.720, C[1]),
        ("14-ch + wind, global", 0.602, 0.461, 0.728, C[1]),
        ("39-ch, 0.25°, 0.92M", 0.620, 0.484, 0.741, C[3]),
        ("39-ch, 0.25°, 40.7M", 0.631, 0.513, 0.732, C[3])]
fig, ax = plt.subplots(figsize=(6.0, 3.8))
ys = np.arange(len(runs))
for i, (name, r, lo, hi, col) in enumerate(runs):
    ax.barh(i, r, color=col, alpha=0.85, height=0.62)
    ax.plot([lo, hi], [i, i], color=INK, lw=1.4)
ax.axvline(0.531, color=INK2, ls="--", lw=1.2)
ax.text(0.531, -0.45, "wind-only ridge, 1° (0.53) ", color=INK2, fontsize=7.5, va="top", ha="right")
ax.axvline(0.568, color=INK2, ls=":", lw=1.2)
ax.text(0.568, len(runs) - 0.3, " wind-only, 0.25° (0.57)", color=INK2, fontsize=7.5, va="top")
ax.set_yticks(ys)
ax.set_yticklabels([r[0] for r in runs], fontsize=8)
ax.set_xlabel("year-blocked k-fold $r$ vs monthly RAPID transport (pooled ridge, 95% CI)")
ax.set_xlim(0, 0.85)
ax.set_ylim(-1.3, len(runs) - 0.2)
strip(ax)
fig.tight_layout()
save(fig, "fig_kfold.pdf")

# ---- Fig 2: the probe ladder — same embeddings, three read-outs -------------
# src: probe_kfold.json (ridge), probe_kfold_mlp.json, probe_head.json
codecs = ["14-ch, pixel\n(global)", "25-ch, pixel\n(global)", "24-ch, 3×3 patch\n(global)"]
ridge = [0.602, 0.536, 0.543]
mlp = [0.582, 0.514, np.nan]
head = [0.635, 0.617, 0.690]
head_lo = [0.528, 0.486, 0.570]
head_hi = [0.736, 0.729, 0.781]
x = np.arange(3)
w = 0.26
fig, ax = plt.subplots(figsize=(6.0, 3.1))
ax.bar(x - w, ridge, w, color=C[0], label="ridge on the section mean")
ax.bar(x, mlp, w, color=C[1], label="MLP on the section mean")
ax.bar(x + w, head, w, color=C[2], label="attention head over the unpooled section")
for i in range(3):
    ax.plot([x[i] + w, x[i] + w], [head_lo[i], head_hi[i]], color=INK, lw=1.4)
ax.axhline(0.531, color=INK2, ls="--", lw=1.2)
ax.text(2.45, 0.531, " wind-only\n 0.53", color=INK2, fontsize=7.5, va="center")
ax.set_xticks(x)
ax.set_xticklabels(codecs, fontsize=8)
ax.set_ylabel("RAPID k-fold $r$")
ax.set_ylim(0.45, 0.82)
ax.legend(fontsize=7.5, loc="upper left")
strip(ax)
fig.tight_layout()
save(fig, "fig_ladder.pdf")

# ---- Fig 3: stage-2 held-out one-step ratio by width, and the RAPID probe ---
cv = D["curves"]
fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.2))
ax = axes[0]
for key, col, lab in (("e060a", C[2], "7.6M"), ("e060b", C[3], "40.4M"),
                      ("e059", C[0], "206.7M"), ("e060c", C[4], "400M (stopped at 2k)")):
    c = cv[key]
    vp = c["val_persistence"]
    st = np.array(c["step"])
    m = st <= 20000
    ax.plot(st[m] / 1000.0, np.array(c["val_zmse"])[m] / vp, color=col, label=lab, marker="o", ms=2.5, lw=1.2)
ax.set_xscale("log")
ax.set_xlabel("training step (thousands, log)")
ax.set_ylabel("held-out one-step z-MSE / persistence")
ax.set_ylim(0.55, 0.95)
ax.legend(fontsize=7.5, loc="upper right", title="parameters", title_fontsize=7.5)
strip(ax)

ax = axes[1]
for key, col, lab in (("e060a", C[2], "7.6M"), ("e060b", C[3], "40.4M"), ("e059", C[0], "206.7M")):
    pr = [p for p in cv[key]["probe"] if p["step"] <= 200000]
    ax.plot([p["step"] / 1000 for p in pr], [p["rapid_r_deseas"] for p in pr], color=col, label=lab, marker="o", ms=2.5, lw=1.2)
ax.set_xlabel("training step (thousands)")
ax.set_ylabel("RAPID probe $r$ (in-training, pooled)")
ax.set_ylim(0.45, 0.68)
ax.legend(fontsize=7.5, title="parameters", title_fontsize=7.5)
strip(ax)
fig.tight_layout()
save(fig, "fig_stage2.pdf")

# ---- Fig 4: the roll — the head against its references, and per channel ----
r0 = D["r0_clean_516"]["corridor"]
lead = np.array([r["h"] for r in r0]) * STEP_DAYS
pc = D["r0_clean_516"]["corridor_per_channel"]
fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.2))
ax = axes[0]
ax.plot(lead, [r["msss_pers"] for r in r0], color=C[2], label="vs raw persistence")
ax.plot(lead, [r["msss_clim"] for r in r0], color=C[0], label="vs climatology (zero anomaly)")
ax.plot(lead, [r["msss_damped"] for r in r0], color=C[3], label="vs damped persistence (AR(1))")
ax.axhline(0, color=INK2, lw=0.8)
ax.set_xlabel("lead (days)")
ax.set_ylabel("MSSS, corridor")
ax.set_ylim(-0.8, 0.45)
ax.legend(fontsize=7.5, loc="lower left")
strip(ax)

ax = axes[1]
order = ["sst", "ssh", "cur_speed", "log_mld", "tau_x", "tau_y", "tau_x_std", "tau_y_std"]
for i, ch in enumerate(order):
    rr = pc[ch]
    ax.plot([r["h"] * STEP_DAYS for r in rr], [r["acc"] for r in rr], color=C[i], label=ch,
            lw=1.8 if ch in ("sst", "ssh") else 1.0)
ax.axhline(0, color=INK2, lw=0.8)
ax.set_xlabel("lead (days)")
ax.set_ylabel("anomaly correlation, corridor")
ax.set_ylim(-0.15, 0.9)
ax.legend(fontsize=7, ncol=2, loc="upper right")
strip(ax)
fig.tight_layout()
save(fig, "fig_roll.pdf")

# ---- Fig 5: the head against a linear inverse model, and the calibration identity
lim = D["lim_527"]
acc = np.array([r["acc"] for r in r0])
amp = np.array([r["amp_ratio"] for r in r0])
msss = np.array([r["msss_clim"] for r in r0])
ident = 1 - (1 + amp**2 - 2 * amp * acc)
calib = acc**2
fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.1))
ax = axes[0]
ax.plot(lead, acc, color=C[0], label="206.7M head")
w520 = D.get("width_rolls", {}).get("520")
if w520:
    rr = w520["corridor"]
    ax.plot([r["h"] * STEP_DAYS for r in rr], [r["acc"] for r in rr], color=C[0], ls="--", lw=1.2, label="7.6M head")
w523 = D.get("width_rolls", {}).get("523")
if w523:
    rr = w523["corridor"]
    ax.plot([r["h"] * STEP_DAYS for r in rr], [r["acc"] for r in rr], color=C[0], ls=":", lw=1.2, label="40.4M head")
for key, col, lab in (("lim_k50", C[5], "LIM, 50 modes"), ("lim_k100", C[4], "LIM, 100 modes"), ("lim_k200", C[1], "LIM, 200 modes")):
    rr = lim[key]["corridor"]
    ax.plot([r["h"] * STEP_DAYS for r in rr], [r["acc"] for r in rr], color=col, lw=1.2, label=lab)
ax.axhline(0, color=INK2, lw=0.8)
ax.set_xlabel("lead (days)")
ax.set_ylabel("anomaly correlation, corridor")
ax.set_ylim(-0.1, 0.7)
ax.legend(fontsize=7.5, loc="upper right")
strip(ax)
ax = axes[1]
ax.plot(lead, msss, color=C[0], label="206.7M head, measured")
ax.plot(lead, ident, color=C[0], ls=":", lw=1.0, label=r"head, $1-(1+a^2-2a\,\mathrm{ACC})$")
ax.plot(lead, calib, color=C[2], label=r"head rescaled to $a=\mathrm{ACC}$")
rr = lim["lim_k200"]["corridor"]
ax.plot([r["h"] * STEP_DAYS for r in rr], [r["msss_clim"] for r in rr], color=C[1], label="LIM, 200 modes")
ax.axhline(0, color=INK2, lw=0.8)
ax.set_xlabel("lead (days)")
ax.set_ylabel("MSSS vs climatology, corridor")
ax.set_ylim(-0.8, 0.45)
ax.legend(fontsize=7.5, loc="lower left")
strip(ax)
fig.tight_layout()
save(fig, "fig_lim.pdf")

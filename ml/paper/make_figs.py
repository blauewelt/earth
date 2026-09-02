#!/usr/bin/env python3
"""Figures for the technical report (ml/paper/paper.tex), reset edition.

Every series is read from data/reset_data.json, which extract_data.py builds
from the public result bundles (probes-510/513/516 on the ml-metrics branch)
and the stage-2 training records. Nothing here is typed in by hand.

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
D = json.load(open(os.path.join(HERE, "data", "reset_data.json")))

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
cv = D["curves"]

# ---- Fig 1: the pool bug in the training curves, and the width ladder -----
fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.3))
ax = axes[0]
for key, col, lab in (("e051", C[1], "E-051 · endpoint-contaminated pool"),
                      ("e059", C[0], "E-059 · window pool (clean)")):
    c = cv[key]
    vp = c["val_persistence"]
    st = np.array(c["step"]) / 1000.0
    ax.plot(st, np.array(c["train_zmse"]) / vp, color=col, ls="--", lw=1.2)
    ax.plot(st, np.array(c["val_zmse"]) / vp, color=col, label=lab)
ax.plot([], [], color=INK2, ls="--", lw=1.2, label="train (dashed)")
ax.plot([], [], color=INK2, label="held-out (solid)")
ax.set_xlabel("training step (thousands)")
ax.set_ylabel("one-step z-MSE / persistence z-MSE")
ax.set_ylim(0, 1.0)
ax.set_title("206.66M head, identical except for the pool", fontsize=9)
ax.legend(fontsize=7.5, loc="upper right")
strip(ax)

ax = axes[1]
for key, col, lab in (("e060a", C[2], "7.6M"), ("e060b", C[3], "40.4M"),
                      ("e059", C[0], "206.7M"), ("e060c", C[4], "400M (reaped at 2k)")):
    c = cv[key]
    vp = c["val_persistence"]
    st = np.array(c["step"])
    m = st <= 20000
    ax.plot(st[m] / 1000.0, np.array(c["val_zmse"])[m] / vp, color=col, label=lab, marker="o", ms=2.5, lw=1.2)
ax.set_xscale("log")
ax.set_xlabel("training step (thousands, log)")
ax.set_ylabel("held-out one-step ratio")
ax.set_ylim(0.55, 0.95)
ax.set_title("clean pool: every width bottoms out inside ~2,000 steps", fontsize=9)
ax.legend(fontsize=7.5, loc="upper right", title="parameters", title_fontsize=7.5)
strip(ax)
fig.tight_layout()
save(fig, "fig_curves.pdf")

# ---- Fig 2: lead decay, clean vs contaminated; and the three baselines -----
r0 = D["r0_clean_516"]["corridor"]
tw = D["contaminated_twin_510"]["corridor"]
lead = np.array([r["h"] for r in r0]) * STEP_DAYS
fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.3))
ax = axes[0]
ax.plot(lead, [r["acc"] for r in tw], color=C[1], label="contaminated head (E-051, run #510)")
ax.plot(lead, [r["acc"] for r in r0], color=C[0], label="clean head (E-059, run #516)")
ax.axhline(0, color=INK2, lw=0.8)
ax.set_xlabel("lead (days)")
ax.set_ylabel("anomaly correlation, corridor")
ax.set_ylim(-0.1, 1.02)
ax.set_title("same battery, same starts, one variable: the training pool", fontsize=9)
ax.legend(fontsize=7.5, loc="center right")
strip(ax)

ax = axes[1]
ax.plot(lead, [r["msss_pers"] for r in r0], color=C[2], label="vs raw persistence")
ax.plot(lead, [r["msss_clim"] for r in r0], color=C[0], label="vs climatology (zero anomaly)")
ax.plot(lead, [r["msss_damped"] for r in r0], color=C[3], label="vs damped persistence (AR(1))")
ax.axhline(0, color=INK2, lw=0.8)
ax.set_xlabel("lead (days)")
ax.set_ylabel("MSSS of the clean head, corridor")
ax.set_ylim(-0.8, 0.45)
ax.set_title("the clean head against three nulls", fontsize=9)
ax.legend(fontsize=7.5, loc="lower left")
strip(ax)
fig.tight_layout()
save(fig, "fig_decay.pdf")

# ---- Fig 3: per channel ----------------------------------------------------
pc = D["r0_clean_516"]["corridor_per_channel"]
order = ["sst", "ssh", "cur_speed", "log_mld", "tau_x", "tau_y", "tau_x_std", "tau_y_std"]
fig, ax = plt.subplots(figsize=(6.4, 3.4))
for i, ch in enumerate(order):
    rr = pc[ch]
    ax.plot([r["h"] * STEP_DAYS for r in rr], [r["acc"] for r in rr], color=C[i], label=ch,
            lw=1.8 if ch in ("sst", "ssh") else 1.1)
ax.axhline(0, color=INK2, lw=0.8)
ax.set_xlabel("lead (days)")
ax.set_ylabel("anomaly correlation, corridor")
ax.set_ylim(-0.15, 0.9)
ax.set_title("clean head, per channel (the 32 Argo channels score nothing at any lead)", fontsize=9)
ax.legend(fontsize=7.5, ncol=2, loc="upper right")
strip(ax)
fig.tight_layout()
save(fig, "fig_channels.pdf")

# ---- Fig 4: the calibration identity -----------------------------------------
acc = np.array([r["acc"] for r in r0])
amp = np.array([r["amp_ratio"] for r in r0])
msss = np.array([r["msss_clim"] for r in r0])
ident = 1 - (1 + amp**2 - 2 * amp * acc)
calib = acc**2          # msss at a = ACC
fig, ax = plt.subplots(figsize=(6.4, 3.3))
ax.plot(lead, msss, color=C[0], label="measured MSSS vs climatology")
ax.plot(lead, ident, color=C[1], ls="--", label=r"identity $1-(1+a^2-2a\,\mathrm{ACC})$ from the head's own $a$, ACC")
ax.plot(lead, calib, color=C[2], label=r"same states rescaled to $a=\mathrm{ACC}$ (= ACC$^2$)")
ax.axhline(0, color=INK2, lw=0.8)
ax.set_xlabel("lead (days)")
ax.set_ylabel("MSSS, corridor")
ax.set_ylim(-0.8, 0.45)
ax.set_title("the negative score is amplitude, not sign: mean |identity − measured| = %.4f" % np.mean(np.abs(ident - msss)), fontsize=9)
ax.legend(fontsize=7.5, loc="lower left")
strip(ax)
fig.tight_layout()
save(fig, "fig_calibration.pdf")

# ---- Fig 5: ensemble dispersion inside and past the record (contaminated head)
disp = D["dispersion_513"]
fig, ax = plt.subplots(figsize=(6.4, 3.2))
for ph, col, lab in (("long", C[1], "36 months INSIDE the training record (2005–2007)"),
                     ("future", C[3], "36 months PAST its end (2025–2027)")):
    fv = np.sqrt(np.array(disp[ph]["field_var"]))
    ax.plot(np.arange(1, len(fv) + 1), fv, color=col, label=lab, marker="o", ms=2.5, lw=1.2)
ax.set_xlabel("months rolled")
ax.set_ylabel("member spread of the decoded field\n(sd, corridor mean)")
ax.set_title("8-member ensemble, contaminated head (run #513): confident where it memorised", fontsize=9)
ax.legend(fontsize=7.5, loc="upper left")
strip(ax)
fig.tight_layout()
save(fig, "fig_dispersion.pdf")

# ---- Fig 6: the RAPID probe along training, by width --------------------------
fig, ax = plt.subplots(figsize=(6.4, 3.0))
for key, col, lab in (("e060a", C[2], "7.6M"), ("e060b", C[3], "40.4M"), ("e059", C[0], "206.7M")):
    pr = [p for p in cv[key]["probe"] if p["step"] <= 200000]
    ax.plot([p["step"] / 1000 for p in pr], [p["rapid_r_deseas"] for p in pr], color=col, label=lab, marker="o", ms=2.5, lw=1.2)
ax.set_xlabel("training step (thousands)")
ax.set_ylabel("RAPID probe r (in-training, pooled)")
ax.set_ylim(0.45, 0.68)
ax.set_title("the transport probe drifts down with training in the large arm only", fontsize=9)
ax.legend(fontsize=7.5, title="parameters", title_fontsize=7.5)
strip(ax)
fig.tight_layout()
save(fig, "fig_probe.pdf")

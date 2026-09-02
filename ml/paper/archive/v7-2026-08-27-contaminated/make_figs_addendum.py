#!/usr/bin/env python3
"""Figures for the contamination addendum of the archived v7 report.

Reads ../../data/report_data.json (built by ../../extract_data.py from the
public result bundles probes-510/513/516 and the stage-2 training records)
and writes fig_addendum_*.pdf into figs/ (or figs_dark/ with --dark).

Run from anywhere:  python3 make_figs_addendum.py [--dark]
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
D = json.load(open(os.path.join(HERE, "..", "..", "data", "report_data.json")))

if DARK:
    BG = "#14140f"
    INK, INK2, GRID = "#e8e6df", "#a5a396", "#3a3a35"
    C1, C2, C3, C4 = "#5b9ee8", "#f08a5c", "#5fba82", "#c58be0"
else:
    BG = "white"
    INK, INK2, GRID = "#1a1a19", "#6f6e66", "#e5e4de"
    C1, C2, C3, C4 = "#2a78d6", "#eb6834", "#3a9a5c", "#8e4bc7"

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

# ---- A1: the same battery under the two pools ------------------------------
c510 = D["contaminated_twin_510"]
c516 = D["r0_clean_516"]
fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.2))
ax = axes[0]
for src, col, lab in ((c510, C2, "endpoint pool (E-051 head)"), (c516, C1, "window pool (E-059 head)")):
    rows = src["corridor"]
    ax.plot([r["h"] * STEP_DAYS for r in rows], [r["acc"] for r in rows], color=col, label=lab)
ax.axhline(0, color=INK2, lw=0.8)
ax.set_xlabel("lead (days)")
ax.set_ylabel("field anomaly correlation, corridor")
ax.set_ylim(-0.1, 1.05)
ax.legend(fontsize=7.5, loc="center right")
strip(ax)

ax = axes[1]
bands = ["h1-18_5-90d", "h19-36_95-180d", "h37-73_185-365d"]
labels = ["5–90 d", "95–180 d", "185–365 d"]
x = np.arange(3)
w = 0.36
ax.bar(x - w / 2, [c510["amoc_bands_unpooled"][b]["r"] for b in bands], w, color=C2, label="endpoint pool")
ax.bar(x + w / 2, [c516["amoc_bands_unpooled"][b]["r"] for b in bands], w, color=C1, label="window pool")
ax.axhline(0, color=INK2, lw=0.8)
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.set_ylabel("RAPID transport $r$ (unpooled read-out)")
ax.set_ylim(-0.35, 0.75)
ax.legend(fontsize=7.5, loc="upper left")
strip(ax)
fig.tight_layout()
save(fig, "fig_addendum_pools.pdf")

# ---- A2: trained vs held-out longitudes ---------------------------------------
sp = D["spatial_split_513"]
scopes = ["gate", "corridor", "window"]
fig, ax = plt.subplots(figsize=(6.0, 3.0))
x = np.arange(3)
w = 0.2
for i, (key, col, lab) in enumerate((("fgn_m8_head", C3, "145-point stencil, 8-member mean"),
                                     ("gate_head_e017", C4, "1-point gate head"))):
    hb = sp[key]
    ax.bar(x + (i - 0.5) * 2 * w - w / 2, [hb[s + "_trainlon"]["horizon_auc"] for s in scopes], w, color=col, alpha=0.95, label=lab + ", trained longitudes")
    ax.bar(x + (i - 0.5) * 2 * w + w / 2, [hb[s + "_holdlon"]["horizon_auc"] for s in scopes], w, color=col, alpha=0.4, label=lab + ", held-out longitudes")
ax.axhline(0, color=INK2, lw=0.8)
ax.set_xticks(x)
ax.set_xticklabels(scopes)
ax.set_ylabel("mean MSSS vs climatology")
ax.set_ylim(0, 1.3)
ax.legend(fontsize=6.5, loc="upper left", ncol=2)
strip(ax)
fig.tight_layout()
save(fig, "fig_addendum_split.pdf")

# ---- A3: ensemble dispersion inside and past the record -----------------------
dp = D["dispersion_513"]
fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.0))
for ax, key, ylab in ((axes[0], "sv_spread", "transport spread across 8 members (Sv)"),
                      (axes[1], "field_var", "field variance across members")):
    for ph, col, lab in (("long", C1, "roll 2005-01 → 2007-12 (inside the record)"),
                         ("future", C2, "roll 2025-01 → 2027-12 (past the record)")):
        ax.plot(np.arange(1, 37), dp[ph][key], color=col, label=lab, marker="o", ms=2.5, lw=1.2)
    ax.set_xlabel("months into the roll")
    ax.set_ylabel(ylab)
    ax.legend(fontsize=7.5, loc="upper left")
    strip(ax)
fig.tight_layout()
save(fig, "fig_addendum_dispersion.pdf")

# ---- A4: the two training records -----------------------------------------------
cv = D["curves"]
fig, ax = plt.subplots(figsize=(6.0, 3.0))
for key, col, lab in (("e051", C2, "endpoint pool"), ("e059", C1, "window pool")):
    c = cv[key]
    vp = c["val_persistence"]
    st = np.array(c["step"]) / 1000.0
    ax.plot(st, np.array(c["val_zmse"]) / vp, color=col, label=lab + ", held-out one-step")
    ax.plot(st, np.array(c["train_zmse"]) / vp, color=col, ls="--", lw=1.0, label=lab + ", training")
ax.set_xlabel("training step (thousands)")
ax.set_ylabel("one-step z-MSE / persistence")
ax.set_yscale("log")
ax.legend(fontsize=7.5, loc="lower left", ncol=2)
strip(ax)
fig.tight_layout()
save(fig, "fig_addendum_curves.pdf")

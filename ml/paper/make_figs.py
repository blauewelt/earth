#!/usr/bin/env python3
"""Comparison figures for the tech report (ml/paper/paper.tex).

Every figure compares RUNS AGAINST EACH OTHER on one metric — the
cross-run views the per-run curves in ml/curves/ cannot give. House
style follows ml/plot_run.py (same palette, separate panels over dual
axes, text in ink). Data provenance: numbers with a `# src:` comment are
transcribed from the named run's committed results (LEADERBOARD.md /
runs/<run>/*.json); series-bearing JSONs are read directly.

Run from repo root:  python3 ml/paper/make_figs.py
"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ML = os.path.dirname(HERE)

# --dark: same figures on a dark page (reading in a dark room), written to
# figs_dark/ so the light build never has to be regenerated to switch back.
import sys
DARK = "--dark" in sys.argv
FIGS = os.path.join(HERE, "figs_dark" if DARK else "figs")
os.makedirs(FIGS, exist_ok=True)

if DARK:
    BG = "#14140f"
    INK, INK2, GRID = "#e8e6df", "#a5a396", "#3a3a35"
    C1, C2, C3 = "#5b9ee8", "#f08a5c", "#5fba82"
else:
    BG = "white"
    INK, INK2, GRID = "#1a1a19", "#6f6e66", "#e5e4de"
    C1, C2, C3 = "#2a78d6", "#eb6834", "#3a9a5c"

plt.rcParams.update({
    "figure.facecolor": BG, "savefig.facecolor": BG, "axes.facecolor": BG,
    "figure.dpi": 150, "savefig.bbox": "tight",
    "axes.edgecolor": INK2, "axes.labelcolor": INK,
    "text.color": INK, "xtick.color": INK2, "ytick.color": INK2,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.7,
    "axes.axisbelow": True, "font.size": 9,
})


def strip(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


# ---- Fig 1: bottleneck width vs the two things it buys --------------------
# src: LEADERBOARD.md d_z sweep (12-ch NA tensor, matched 30k steps) + k-fold
dz = [8, 16, 32, 64, 128]
chan = [28.6, 30.3, 30.5, 30.6, 30.2]
kf = [0.111, 0.151, 0.182, 0.308, 0.166]
kf_lo = [0.01, 0.01, 0.05, 0.13, 0.072]
kf_hi = [0.20, 0.28, 0.31, 0.46, 0.295]
fig, (a1, a2) = plt.subplots(2, 1, figsize=(4.6, 4.6), sharex=True,
                             gridspec_kw={"hspace": 0.12})
a1.plot(dz, chan, "o-", color=C1, lw=2)
a1.set_ylabel("field skill chan% (t+1)")
a1.set_ylim(27.5, 31.5)
a2.errorbar(dz, kf, yerr=[np.array(kf) - kf_lo, np.array(kf_hi) - kf],
            fmt="o-", color=C2, lw=2, capsize=3)
a2.set_ylabel("AMOC probe, k-fold $r$")
a2.set_xlabel("bottleneck width $d_z$")
a2.set_xscale("log", base=2)
a2.set_xticks(dz)
a2.set_xticklabels(dz)
for a in (a1, a2):
    strip(a)
a1.set_title("Field prediction saturates at $d_z$=16-32; "
             "the transport read-out peaks at 64 and turns over",
             loc="left", fontsize=9)
fig.savefig(os.path.join(FIGS, "fig_dz.pdf"))
plt.close(fig)

# ---- Fig 2: the defensible probe ranking, with the wind-only line ---------
# src: LEADERBOARD.md k-fold table + probe_kfold.json wind-only baseline
runs = [("12-ch $d_z$=8", 0.111, 0.01, 0.20, C1),
        ("12-ch $d_z$=16", 0.151, 0.01, 0.28, C1),
        ("12-ch $d_z$=32", 0.182, 0.05, 0.31, C1),
        ("12-ch $d_z$=128", 0.166, 0.072, 0.295, C1),
        ("12-ch $d_z$=64", 0.308, 0.13, 0.46, C1),
        ("25-ch pixel (#17)", 0.536, 0.378, 0.683, C3),
        ("24-ch patch 3x3 (#18)", 0.543, 0.428, 0.659, C3),
        ("14-ch wind, NA", 0.604, 0.474, 0.720, C2),
        ("14-ch wind, global", 0.602, 0.461, 0.728, C2)]
fig, ax = plt.subplots(figsize=(5.6, 3.4))
ys = np.arange(len(runs))
for i, (name, r, lo, hi, col) in enumerate(runs):
    ax.barh(i, r, color=col, alpha=0.85, height=0.62)
    ax.plot([lo, hi], [i, i], color=INK, lw=1.4)
ax.axvline(0.531, color=INK2, ls="--", lw=1.4)
ax.text(0.531, len(runs) - 0.28, " wind-stress-only ridge (0.53)",
        color=INK2, fontsize=8, va="top")
ax.set_yticks(ys)
ax.set_yticklabels([r[0] for r in runs])
ax.set_xlabel("year-blocked k-fold $r$ vs monthly deseasonalized RAPID (95% CI)")
ax.set_xlim(0, 0.80)
strip(ax)
fig.savefig(os.path.join(FIGS, "fig_kfold.pdf"))
plt.close(fig)

# ---- Fig 2b: where the channel expansion moved the skill ------------------
# src: runs/*/probe_kfold.json (RAPID and MOVE, 2026-08-08)
labels = ["RAPID 26.5N\n(wind-dominated)", "MOVE 16N\n(deep, density)"]
old = [0.582, 0.111]          # global15sst, 15 channels
new = [0.536, 0.206]          # pixel25_40k, 25 channels
old_lp = [0.64, 0.379]
new_lp = [0.452, 0.623]
x = np.arange(2)
w = 0.34
fig, (b1, b2) = plt.subplots(1, 2, figsize=(7.0, 2.9))
b1.bar(x - w / 2, old, w, color=C1, label="15 channels")
b1.bar(x + w / 2, new, w, color=C2, label="25 channels")
b1.axhline(0, color=INK2, lw=1)
b1.set_xticks(x); b1.set_xticklabels(labels, fontsize=8)
b1.set_ylabel("k-fold $r$ (monthly)")
b1.legend(frameon=False, fontsize=8)
b1.set_title("monthly", loc="left", fontsize=9)
b2.bar(x - w / 2, old_lp, w, color=C1)
b2.bar(x + w / 2, new_lp, w, color=C2)
b2.axhline(0, color=INK2, lw=1)
b2.set_xticks(x); b2.set_xticklabels(labels, fontsize=8)
b2.set_ylabel("18-month low-passed $r$")
b2.set_title("low-frequency", loc="left", fontsize=9)
for a in (b1, b2):
    strip(a)
fig.suptitle("Sixteen Argo T/S levels move skill from the wind-driven "
             "target to the density-driven one", fontsize=9, x=0.02, ha="left")
fig.savefig(os.path.join(FIGS, "fig_tradeoff.pdf"))
plt.close(fig)


# ---- Fig 2c: the probe ladder — same embeddings, three read-outs ----------
# src: probe_kfold.json (ridge), probe_kfold_mlp.json, probe_head.json
codecs = ["global14\n(C=14, pixel)", "pixel25\n(C=25, pixel)", "patch24\n(C=24, 3x3 patch)"]
ridge = [0.602, 0.536, 0.543]
mlp = [0.582, 0.514, np.nan]
head = [0.635, 0.617, 0.690]
head_lo = [0.528, 0.486, 0.570]
head_hi = [0.736, 0.729, 0.781]
x = np.arange(3)
w = 0.26
fig, ax = plt.subplots(figsize=(6.2, 3.2))
ax.bar(x - w, ridge, w, color=C1, label="ridge (mean-pooled)")
ax.bar(x, mlp, w, color=C2, label="MLP (mean-pooled)")
ax.bar(x + w, head, w, color=C3, label="attention head (unpooled section)")
for i in range(3):
    ax.plot([x[i] + w, x[i] + w], [head_lo[i], head_hi[i]], color=INK, lw=1.4)
ax.axhline(0.531, color=INK2, ls="--", lw=1.4)
ax.text(2.42, 0.531, " wind-only\n 0.53", color=INK2, fontsize=7.5, va="center")
ax.set_xticks(x); ax.set_xticklabels(codecs, fontsize=8)
ax.set_ylabel("RAPID k-fold $r$ (deseasonalized)")
ax.set_ylim(0.45, 0.82)
ax.legend(frameon=False, fontsize=8, loc="upper left")
ax.set_title("Same frozen embeddings, three read-outs: nonlinearity adds "
             "nothing, section structure adds everything", loc="left", fontsize=9)
strip(ax)
fig.savefig(os.path.join(FIGS, "fig_ladder.pdf"))
plt.close(fig)

# ---- Fig 3: rollout comparison (MSSS and ACC vs horizon) ------------------
def load_rollout(run):
    j = json.load(open(os.path.join(ML, "runs", run, "rollout.json")))
    h = [c["h"] for c in j["chan_skill"]]
    return (h, [c["msss_clim"] for c in j["chan_skill"]],
            [c["acc"] for c in j["chan_skill"]],
            [c["msss_damped"] for c in j["chan_skill"]])

fig, (a1, a2) = plt.subplots(1, 2, figsize=(7.4, 3.0))
for run, label, col in (("global14", "global codec (global domain)", C1),
                        ("wind14", "NA codec (NA domain)", C2)):
    h, msss, acc, mdamp = load_rollout(run)
    a1.plot(h, msss, "o-", color=col, lw=2, ms=3.5, label=label)
    a2.plot(h, acc, "o-", color=col, lw=2, ms=3.5, label=label)
a1.axhline(0, color=INK2, lw=1, ls=":")
a1.set_xlabel("forecast horizon (months)")
a1.set_ylabel("MSSS vs climatology")
a1.legend(frameon=False, fontsize=8)
a2.axhline(0.5, color=INK2, lw=1, ls="--")
a2.text(11.9, 0.505, "ACC 0.5", color=INK2, fontsize=8, ha="right", va="bottom")
a2.set_xlabel("forecast horizon (months)")
a2.set_ylabel("anomaly correlation (ACC)")
for a in (a1, a2):
    strip(a)
    a.set_xticks(range(1, 13, 2))
fig.savefig(os.path.join(FIGS, "fig_rollout.pdf"))
plt.close(fig)

# ---- Fig 4: stage-2 capacity (chan% vs parameters) ------------------------
# src: SCALING.md steps-matched sweep, 12-ch NA embeddings, seeds shown
params = [0.11, 0.35, 1.81, 3.98]
seeds = [(30.7, 29.6), (31.4, 30.4), (35.5, 36.2), (37.6, 37.7)]
fig, ax = plt.subplots(figsize=(4.6, 3.0))
mid = [np.mean(s) for s in seeds]
ax.plot(params, mid, "-", color=C1, lw=2)
for p, s in zip(params, seeds):
    ax.plot([p, p], list(s), "o", color=C1, ms=4)
ax.plot(0.35, 31.9, "s", color=C2, ms=6)
ax.annotate("mid, 4000 steps\n(steps-matched control)", (0.35, 31.9),
            textcoords="offset points", xytext=(8, -2), fontsize=7.5,
            color=C2)
ax.set_xscale("log")
ax.set_xlabel("stage-2 parameters (millions)")
ax.set_ylabel("chan% (t+1 error reduction)")
ax.set_title("Capacity buys dynamics; still rising at the value-count anchor",
             loc="left", fontsize=9)
strip(ax)
fig.savefig(os.path.join(FIGS, "fig_stage2.pdf"))
plt.close(fig)

# ---- Fig 5: the 2009-10 collapse, out-of-fold ------------------------------
# src: dip_check.py runs (transcribed from the 2026-08-07/08 harvests)
months = ["2009-09", "2009-10", "2009-11", "2009-12", "2010-01",
          "2010-02", "2010-03", "2010-04", "2010-05", "2010-06"]
obs = [-6.03, -4.14, -5.69, -8.42, -7.84, -7.12, -5.28, -0.97, 1.41, 2.74]
p_wind = [0.24, -1.00, 0.52, -3.45, -4.64, -5.74, -3.83, -0.69, 0.68, 1.98]
p_12ch = [0.76, -0.88, 1.89, 0.63, -2.96, -1.27, -3.96, -0.18, -0.02, 3.32]
x = np.arange(len(months))
fig, ax = plt.subplots(figsize=(6.4, 3.0))
ax.fill_between(x, obs, 0, color=INK2, alpha=0.15)
ax.plot(x, obs, "o-", color=INK, lw=2, label="observed (RAPID, deseasonalized)")
ax.plot(x, p_12ch, "s--", color=C1, lw=1.8, ms=4,
        label="12-ch codec (no wind): 16% of dip")
ax.plot(x, p_wind, "^-", color=C2, lw=1.8, ms=4,
        label="14-ch codec (wind): 50% of dip")
ax.axhline(0, color=INK2, lw=1, ls=":")
ax.set_xticks(x)
ax.set_xticklabels([m[2:].replace("-", "/") for m in months], fontsize=8)
ax.set_ylabel("transport anomaly (Sv)")
ax.set_title("Winter 2009-10 collapse — out-of-fold predictions", loc="left",
             fontsize=9)
ax.legend(frameon=False, fontsize=8, loc="lower right")
strip(ax)
fig.savefig(os.path.join(FIGS, "fig_dip.pdf"))
plt.close(fig)

# ---- Fig 6: training curves of the codec generations ----------------------
def load_loss(run):
    steps, loss = [], []
    p = os.path.join(ML, "runs", run, "metrics.jsonl")
    for line in open(p):
        m = json.loads(line)
        if "loss_rec" in m:
            steps.append(m["step"]); loss.append(m["loss_rec"])
    k = 9
    sm = np.convolve(loss, np.ones(k) / k, mode="same") if len(loss) > k else loss
    return np.array(steps), sm

fig, ax = plt.subplots(figsize=(5.6, 3.0))
for run, label, col in (("wind14", "NA window, 14-ch (6.6 epochs)", C2),
                        ("global14", "global window, 14-ch (0.8 epochs)", C1),
                        ("pixel25_40k", "global, 25-ch (GPU, 40k steps)", C3)):
    try:
        s, l = load_loss(run)
        ax.plot(s, l, color=col, lw=1.8, label=label)
    except FileNotFoundError:
        pass
ax.set_xlabel("training steps")
ax.set_ylabel("masked reconstruction loss")
ax.legend(frameon=False, fontsize=8)
ax.set_title("Same architecture, more data and more channels",
             loc="left", fontsize=9)
strip(ax)
fig.savefig(os.path.join(FIGS, "fig_train.pdf"))
plt.close(fig)

# ---- Fig 7: the capacity ladder at fixed input (sunflower-55) -------------
# src: EXPERIMENTS.md E-027 2x2 (#282-#307) + eval #304 (probes-304) +
#      E-028 (#308-#310) + eval #333 (ml-metrics probes-333.json).
#      Params are MEASURED from each run's own archived head (params_M).
cap_params = [34.098, 87.971, 205.4]
cap_labels = ["576$\\times$8\n34.1M", "768$\\times$12\n88.0M",
              "1024$\\times$16\n205.4M"]
cap_ratio = [[0.17860, 0.17692, 0.17380],          # base55 s0-s2
             [0.14997, 0.14728, 0.14562],          # big55 s0-s2
             [0.13308, 0.12947]]                   # xl55 s1-s2 (two quoted)
cap_auc = [[0.5690, 0.5730],                       # base55 s0-s1
           [0.6210, 0.6210, 0.6220],               # big55 s0-s2
           [0.664, 0.663, 0.664]]                  # xl55 s0-s2 (#333)
fig, (a1, a2) = plt.subplots(1, 2, figsize=(7.4, 3.0))
for ax, data, ylab in ((a1, cap_ratio, "forecast ratio (lower = better)"),
                       (a2, cap_auc, "rolled corridor AUC (higher = better)")):
    mid = [np.mean(s) for s in data]
    ax.plot(cap_params, mid, "-", color=C1, lw=2)
    for p, s in zip(cap_params, data):
        ax.plot([p] * len(s), s, "o", color=C1, ms=4)
    ax.set_xscale("log")
    ax.set_xticks(cap_params)
    ax.set_xticklabels(cap_labels, fontsize=7.5)
    ax.set_xlabel("stage-2 parameters")
    ax.set_ylabel(ylab)
    ax.minorticks_off()
    strip(ax)
a2.axhline(0.589, color=INK2, ls=":", lw=1.2)
a2.text(90, 0.5905, "no-neighbour baseline 0.589", color=INK2, fontsize=7)
a2.axhline(0.6043, color=INK2, ls="--", lw=1.2)
a2.text(90, 0.598, "base-scale champion\n(ring-8@222) 0.604",
        color=INK2, fontsize=7, va="top")
a1.set_title("one-step forecast", loc="left", fontsize=9)
a2.set_title("12-month roll, AMOC corridor", loc="left", fontsize=9)
fig.suptitle("Capacity improves BOTH axes and does not saturate through 205M "
             "parameters (fixed sunflower-55 input, 60k steps)",
             fontsize=9, x=0.02, ha="left")
fig.savefig(os.path.join(FIGS, "fig_capacity.pdf"))
plt.close(fig)

# ---- Fig 8: the secondary axes — input width and step budget --------------
# src: EXPERIMENTS.md E-027 (width, 768x12 @60k), E-029d (#318/#319/#321/
#      #324/#325 sun89 60k), E-029d-ext (#334/#335 120k, #350/#351 200k),
#      E-028 xl55 (#308-#310).
fig, (a1, a2) = plt.subplots(1, 2, figsize=(7.4, 3.0))
wpts = [34, 55, 89]
wdata = [[0.15186, 0.14911, 0.14827],
         [0.14997, 0.14728, 0.14562],
         [0.14624, 0.14556, 0.14420]]
mid = [np.mean(s) for s in wdata]
a1.plot(wpts, mid, "-", color=C3, lw=2)
for p, s in zip(wpts, wdata):
    a1.plot([p] * len(s), s, "o", color=C3, ms=4)
a1.set_xticks(wpts)
a1.set_xlabel("sunflower input points (768$\\times$12, 60k steps)")
a1.set_ylabel("forecast ratio")
a1.set_title("width: still paying at 89 points", loc="left", fontsize=9)
steps = [60, 120, 200]
s1 = [0.14556, 0.13842, 0.13663]
s2 = [0.14420, 0.13689, 0.13550]
a2.plot(steps, s1, "o-", color=C2, lw=2, ms=4, label="seed 1")
a2.plot(steps, s2, "o-", color=C1, lw=2, ms=4, label="seed 2")
a2.axhspan(0.12947, 0.13308, color=INK2, alpha=0.18, lw=0)
a2.text(200, 0.1312, "xl55 (205M)\nat only 60k steps", color=INK2,
        fontsize=7, ha="right", va="center")
a2.set_xticks(steps)
a2.set_xticklabels(["60k", "120k", "200k"])
a2.set_xlabel("training steps (sunflower-89, 768$\\times$12)")
a2.set_ylabel("forecast ratio")
a2.legend(frameon=False, fontsize=8)
a2.set_title("steps: decelerating toward $\\approx$0.135", loc="left",
             fontsize=9)
for a in (a1, a2):
    strip(a)
fig.suptitle("The secondary axes: width and step budget both pay, and both "
             "pay less than parameters", fontsize=9, x=0.02, ha="left")
fig.savefig(os.path.join(FIGS, "fig_axes.pdf"))
plt.close(fig)

print("figures written to", FIGS)

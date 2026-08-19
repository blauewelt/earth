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
# The two reference lines sit 0.015 apart, so their labels must be pushed to
# OPPOSITE sides of their own lines — stacked between them they overlap, which
# is how this figure first shipped (caught in the page-17 render, 2026-08-16).
a2.axhline(0.589, color=INK2, ls=":", lw=1.2)
# Labels are kept SHORT and pinned to the right edge: the trend line sweeps
# diagonally through this whole y-range, so any long annotation is crossed by
# it somewhere. The lines' identities are spelled out in the caption.
a2.text(205, 0.5875, "no-neighbour 0.589", color=INK2, fontsize=7,
        va="top", ha="right")
a2.axhline(0.6043, color=INK2, ls="--", lw=1.2)
a2.text(205, 0.6028, "base champion 0.604", color=INK2, fontsize=7,
        va="top", ha="right")
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


# ---- Fig 9: the AMOC read-out of the two widest 205M arms, rolled ---------
# src: run #355 (xl89) and #356 (xl144) rollout_spatial.json, staged here as
#      roll_355.json / roll_356.json; truth from
#      `rollout_spatial.py --dump-truth` (the evaluator's own train-month
#      climatology rule, so the deseasonalisation matches the roll's).
# The interactive version with the per-arm table is
#      ml/paper/figs/amoc_roll.html -> blauewelt.github.io/earth/...
# Heads are DEDUPLICATED BY CHECKPOINT: both runs re-roll the frozen gate,
# and drawing it twice would misstate how many arms are on the page.
def _amoc_arms():
    seen, arms = set(), []
    for src, colour in (("roll_355.json", C1), ("roll_356.json", C3)):
        d = json.load(open(os.path.join(HERE, src)))
        for key, h in d["heads"].items():
            ident = h.get("meta", {}).get("file", key)
            if ident in seen:
                continue
            seen.add(ident)
            slots = int(h.get("meta", {}).get("stencil", 1) or 1)
            lab = {1: "gate (no neighbours)"}.get(slots, "%d inputs" % (slots - 1))
            arms.append({"lab": "%s, s%s" % (lab, h["meta"].get("seed", 0)),
                         "c": INK2 if slots == 1 else colour,
                         "gate": slots == 1,
                         "long": h["long"], "future": h["future"],
                         "r": h["long"].get("r_heldout")})
    return arms


def _sm(v, w=18):
    v = np.asarray(v, float)
    if len(v) < w:
        return v
    k = np.ones(w) / w
    out = np.convolve(v, k, mode="same")
    half = w // 2                      # convolve's edges average in zeros
    out[:half] = np.nan
    out[len(out) - half:] = np.nan
    return out


def _yr(yms):
    return np.array([int(s[:4]) + (int(s[5:7]) - 1) / 12.0 for s in yms])


try:
    arms = _amoc_arms()
    truth = json.load(open(os.path.join(HERE, "truth_rapid.json")))
    ty, tv = _yr(truth["ym"]), np.asarray(truth["sv_des"], float)
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(7.6, 3.1),
                                 gridspec_kw={"width_ratios": [1.25, 1]})

    for yr in (2009, 2017, 2023):                 # the never-trained years
        a1.axvspan(yr, yr + 1, color=INK2, alpha=0.13, lw=0)
        a1.text(yr + 0.5, 7.0, str(yr), color=INK2, fontsize=6.5, ha="center")
    a1.plot(ty, _sm(tv), color=INK, lw=2.4, zorder=5)
    a1.text(ty[-1], _sm(tv)[~np.isnan(_sm(tv))][-1], "  observed",
            color=INK, fontsize=7, va="center", fontweight="bold")
    for arm in arms:
        a1.plot(_yr(arm["long"]["roll_ym"]), _sm(arm["long"]["sv_des"]),
                color=arm["c"], lw=1.5 if arm["gate"] else 1.1,
                ls="--" if arm["gate"] else "-", alpha=0.95)
    a1.set_ylim(-8.5, 8.5)
    a1.set_ylabel("deseasonalised anomaly (Sv)")
    a1.set_xlabel("18-month running mean; shaded = held out")
    a1.set_title("hindcast, rolled from a 2004 context", loc="left", fontsize=9)

    lo, hi = np.percentile(tv, [5, 95])
    a2.axhspan(lo, hi, color=INK2, alpha=0.13, lw=0)
    a2.text(2043, hi - 0.4, "observed 90% range", color=INK2, fontsize=6.5,
            ha="right", va="top")
    a2.plot(ty, _sm(tv), color=INK, lw=2.4, zorder=5)
    a2.axvline(2025, color=INK2, lw=0.9, ls=":")
    for arm in arms:
        a2.plot(_yr(arm["future"]["roll_ym"]), _sm(arm["future"]["sv_des"]),
                color=arm["c"], lw=1.5 if arm["gate"] else 1.1,
                ls="--" if arm["gate"] else "-", alpha=0.95)
    a2.set_ylim(-8.5, 8.5)
    a2.set_xlabel("no forcing arrives after the dotted seam")
    a2.set_title("unforced roll, 20 years past the record", loc="left",
                 fontsize=9)

    for a in (a1, a2):
        strip(a)
    hand = [matplotlib.lines.Line2D([], [], color=a["c"], lw=1.4,
                                    ls="--" if a["gate"] else "-",
                                    label="%s  (r=%s)" % (a["lab"], a["r"]))
            for a in arms]
    fig.legend(handles=hand, frameon=False, fontsize=6.5, ncol=3,
               loc="lower center", bbox_to_anchor=(0.5, -0.13))
    fig.suptitle("The corridor's best arms, read out as transport: the gain "
                 "does not transfer (r = held-out years)",
                 fontsize=9, x=0.02, ha="left")
    fig.savefig(os.path.join(FIGS, "fig_amoc_roll.pdf"))
    plt.close(fig)
except FileNotFoundError as e:
    print("fig_amoc_roll skipped:", e)


# ---- Fig 10: where the rolled skill actually sits, per pixel --------------
# src: run #356 (xl144) rollout_spatial.json, staged as roll_356.json; the
#      audit block's `map_msss_clim_window` — 84,405 per-pixel MSSS values
#      against climatology at h = 6 months, pooled over all 39 channels and
#      every evaluation start-month.
#
# THE PIXELS CARRY NO COORDINATES. rollout_spatial.py does `ys, xs =
# np.where(ocean)` — row-major over the window grid, rows SOUTH-FIRST — and
# its export_mask() writes exactly that boolean into data/amoc_eval_mask.json
# as code>=1 (2 = corridor, 3 = RAPID section), with NO [::-1] flip. So the
# geography is recovered by taking the mask's cells with code >= 1 in the
# same row-major order. The first version of that mask writer DID flip, and
# put the Gulf Stream at the latitude of the Norwegian Sea while still
# looking like a plausible ocean (root CLAUDE.md §2). The asserts below are
# the price of not repeating that: counts against the mask's own recorded
# totals, and the RAPID section's reconstructed latitude against 26.5 N.
def _amoc_map(head_key, src_file):
    mk = os.path.join(os.path.dirname(ML), "data", "amoc_eval_mask.json")
    m = json.load(open(mk))
    nx, ny = m["nx"], m["ny"]
    packed = m["packed"]
    assert len(packed) == nx * ny, "packed grid is not nx*ny"
    code = np.array([0 if c == "." else int(c) for c in packed],
                    np.uint8).reshape(ny, nx)
    lats = m["south"] + (np.arange(ny) + 0.5) * m["dlat"]   # row 0 = south
    lons = m["west"] + (np.arange(nx) + 0.5) * m["dlon"]
    ys, xs = np.where(code >= 1)                 # == the eval's ys/xs order

    h = json.load(open(os.path.join(HERE, src_file)))["heads"][head_key]
    v = np.array([np.nan if q is None else q
                  for q in h["audit"]["map_msss_clim_window"]], float)

    cnt = m["counts"]
    assert len(v) == len(ys) == cnt["rolled"], (
        "%d map values, %d mask cells, %d recorded — the mask and the roll "
        "disagree about the window" % (len(v), len(ys), cnt["rolled"]))
    assert int((code >= 2).sum()) == cnt["corridor"], "corridor count moved"
    assert int((code == 3).sum()) == cnt["section"], "section count moved"
    sec = code[ys, xs] == 3
    sec_lat = np.unique(lats[ys[sec]])
    assert len(sec_lat) == 1 and abs(sec_lat[0] - 26.5) < 1e-6, (
        "the RAPID section reconstructs at %s N, not 26.5 — the grid is "
        "mirrored and the map would look fine and be wrong" % sec_lat)
    sec_lo = (lons[xs[sec]].min(), lons[xs[sec]].max())
    assert -80.5 < sec_lo[0] < -79.5 and -14.5 < sec_lo[1] < -12.5, \
        "section spans %s, not the Florida-to-Africa transect" % (sec_lo,)
    cor = code[ys, xs] >= 2
    com = (float(lats[ys[cor]].mean()), float(lons[xs[cor]].mean()))
    assert 10.0 < com[0] < 45.0 and com[1] < -40.0, (
        "corridor centre of mass at %.1f N %.1f E — not the subtropical "
        "western basin; refusing to draw it" % com)

    field = np.full((ny, nx), np.nan)
    field[ys, xs] = v                            # land / off-window stay NaN
    return {"f": field, "cor": (code >= 2), "lats": lats, "lons": lons,
            "sec_lat": float(sec_lat[0]), "sec_lo": sec_lo, "com": com,
            "meta": h["meta"], "hh": h["audit"]["map_h"], "m": m}


try:
    D = _amoc_map("s145rspiral:111-4444-0.71-0.5_s0", "roll_356.json")
    m, f = D["m"], D["f"]
    ext = [m["west"], m["east"], m["south"], m["north"]]

    # Diverging, hard-centred on 0: zero is "no better than climatology",
    # which is the line the whole figure is about. Symmetric ±1 (MSSS's own
    # ceiling), so the midpoint cannot drift to wherever the data happens to
    # sit. House palette: C2 (warm) below climatology, C1 (cool) above; the
    # neutral centre is a TINT, not the page, because land is the page and an
    # unmodelled cell must not read as a zero-skill cell.
    # On the dark page the ramp is lightness-INVERTED (both ends pale, a mid
    # grey at zero) so that a near-zero cell cannot be mistaken for the black
    # land it sits next to.
    stops = ([(0.0, "#6e2708"), (0.30, "#d9622b"), (0.5, "#ddd9cd"),
              (0.70, C1), (1.0, "#0b2f5e")] if not DARK else
             [(0.0, "#f2b48a"), (0.30, C2), (0.5, "#7d7b70"),
              (0.70, C1), (1.0, "#cfe3fb")])
    cmap = matplotlib.colors.LinearSegmentedColormap.from_list("msss", stops)
    cmap = cmap.copy()
    cmap.set_bad(BG)                             # land / outside the window

    fig, ax = plt.subplots(figsize=(6.6, 3.9))
    im = ax.imshow(f, origin="lower", extent=ext, cmap=cmap, vmin=-1, vmax=1,
                   interpolation="nearest", aspect=1.35)
    ax.contour(D["lons"], D["lats"], D["cor"].astype(float), levels=[0.5],
               colors=[INK2], linewidths=0.4, alpha=0.75)
    ax.plot([D["sec_lo"][0], D["sec_lo"][1]], [D["sec_lat"]] * 2,
            color=C3, lw=1.4, solid_capstyle="butt")
    hand = [matplotlib.lines.Line2D([], [], color=INK2, lw=0.8,
                                    label="corridor (29,627 px)"),
            matplotlib.lines.Line2D([], [], color=C3, lw=1.4,
                                    label="RAPID 26.5°N section")]
    ax.legend(handles=hand, frameon=False, fontsize=6.5, ncol=2,
              loc="upper left", bbox_to_anchor=(0.0, -0.10),
              handlelength=1.8, columnspacing=1.6)
    ax.set_xlim(m["west"], m["east"])
    ax.set_ylim(m["south"], m["north"])
    ax.set_xticks([-100, -80, -60, -40, -20, 0, 20])
    ax.set_yticks([0, 20, 40, 60])
    ax.set_xticklabels(["100°W", "80°W", "60°W", "40°W", "20°W", "0°", "20°E"],
                       fontsize=7.5)
    ax.set_yticklabels(["0°", "20°N", "40°N", "60°N"], fontsize=7.5)
    ax.grid(False)
    for s in ("top", "right", "bottom", "left"):
        ax.spines[s].set_visible(True)
        ax.spines[s].set_linewidth(0.6)
    ax.set_title("Rolled skill against climatology at 6 months, per pixel",
                 loc="left", fontsize=9)
    cb = fig.colorbar(im, ax=ax, fraction=0.030, pad=0.02,
                      ticks=[-1, -0.5, 0, 0.5, 1])
    cb.set_label("MSSS vs climatology  (0 = no better)", fontsize=7.5)
    cb.ax.tick_params(labelsize=7)
    cb.outline.set_edgecolor(INK2)
    cb.outline.set_linewidth(0.6)
    fig.savefig(os.path.join(FIGS, "fig_gulfstream.pdf"))
    plt.close(fig)
    print("fig_gulfstream: %s seed %s · corridor CoM %.1f N %.1f E"
          % (D["meta"]["file"], D["meta"]["seed"], D["com"][0], D["com"][1]))
except FileNotFoundError as e:
    print("fig_gulfstream skipped:", e)

print("figures written to", FIGS)

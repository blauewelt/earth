#!/usr/bin/env python3
"""E-080 deck figures. Dark-background matplotlib, 200 dpi PNG."""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon, Ellipse, Circle, FancyArrow, Rectangle
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D
from matplotlib.path import Path as MplPath
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.environ.get(
    "E080_FIG_OUT", os.path.join(os.path.dirname(HERE), "E080_figures"))
os.makedirs(OUT, exist_ok=True)

BG = "#0D1117"
PANEL = "#0D1117"
TEXT = "#E6EDF3"
MUTED = "#7D8590"
GRID = "#30363D"
BLUE = "#4493F8"
BLUE_D = "#1F6FEB"
ORANGE = "#E8734A"
GOLD = "#E3B341"

plt.rcParams.update({
    "figure.facecolor": BG,
    "savefig.facecolor": BG,
    "axes.facecolor": PANEL,
    "text.color": TEXT,
    "axes.labelcolor": TEXT,
    "xtick.color": MUTED,
    "ytick.color": MUTED,
    "axes.edgecolor": GRID,
    "font.family": "DejaVu Sans",
    "font.size": 8,
})


def style(ax):
    for s in ax.spines.values():
        s.set_color(GRID)
        s.set_linewidth(0.8)


# ---------------------------------------------------------------- figure 1
# The hourglass SHAPE, shared by figure 1 (slide 3) and the summary redraw
# (slide 21) so there is one definition of the geometry, not two.
def _hourglass_shape(ax, W0, S, drift=0.0, lw=1.0, lags=None, hw=None,
                     waist=None, alpha_past=0.20, alpha_fut=0.10, hatch="////"):
    """The hourglass polygons. `lags`/`hw` override the default linear cone so
    stage 2's capped reach can be drawn by the same code; the defaults are what
    figure 1 and the summary redraw use and must not change."""
    if lags is None:
        lags = np.arange(0, 6.01, 0.02)
    # past cone (below): widening downward
    cx = -drift * lags
    if hw is None:
        hw = W0 + S * lags
    past = np.concatenate([np.column_stack([cx - hw, -lags]),
                           np.column_stack([(cx + hw)[::-1], -lags[::-1]])])
    ax.add_patch(Polygon(past, closed=True, facecolor=BLUE, alpha=alpha_past,
                         edgecolor=BLUE, lw=lw, zorder=2))
    # future cone (above): mirrored through the anchor
    cxf = drift * lags
    fut = np.concatenate([np.column_stack([cxf - hw, lags]),
                          np.column_stack([(cxf + hw)[::-1], lags[::-1]])])
    ax.add_patch(Polygon(fut, closed=True, facecolor="none", edgecolor=ORANGE,
                         lw=lw, hatch=hatch, zorder=2))
    ax.add_patch(Polygon(fut, closed=True, facecolor=ORANGE, alpha=alpha_fut,
                         edgecolor="none", zorder=1))
    # waist
    wb = W0 if waist is None else waist
    ax.plot([-wb, wb], [0, 0], color=GOLD, lw=3.6, solid_capstyle="round", zorder=6)
    ax.plot([0], [0], marker="o", ms=5, mfc=GOLD, mec=BG, mew=1.0, zorder=7)
    # anchor column
    ax.axvline(0, color=TEXT, ls=(0, (4, 3)), lw=0.9, alpha=0.75, zorder=5)


def hourglass():
    fig = plt.figure(figsize=(7.94, 5.0))
    gs = fig.add_gridspec(1, 2, width_ratios=[2.05, 1.0], left=0.075, right=0.985,
                          top=0.90, bottom=0.10, wspace=0.22)
    axA = fig.add_subplot(gs[0, 0])
    axB = fig.add_subplot(gs[0, 1])

    W0, S = 260.0, 455.0          # waist half-width, growth per pentad (km)

    def cone(ax, drift=0.0, lw=1.0, labels=True):
        _hourglass_shape(ax, W0, S, drift=drift, lw=lw)

    # ---- panel A
    ax = axA
    cone(ax, drift=0.0, lw=1.2)
    ax.set_xlim(-3300, 3300)
    ax.set_ylim(-8.7, 8.7)
    ax.set_xticks([-3000, -1500, 0, 1500, 3000])
    ax.set_xticklabels(["−3000", "−1500", "0", "1500", "3000"])
    ax.set_yticks(range(-6, 7, 2))
    ax.set_xlabel("space along one direction (km)", color=MUTED, fontsize=8.5)
    ax.set_ylabel("time (pentads)", color=MUTED, fontsize=8.5)
    ax.grid(color=GRID, lw=0.45, alpha=0.55)
    ax.set_axisbelow(True)
    style(ax)
    ax.set_title("the hourglass: one anchor, two cones", color=TEXT,
                 fontsize=10.5, pad=8)

    ax.text(0, 7.45, "future cone — targets only, never read", color=ORANGE,
            fontsize=9.0, ha="center", va="center", fontweight="bold")
    ax.text(0, -7.55, "past cone — input, read", color=BLUE, fontsize=9.0,
            ha="center", va="center", fontweight="bold")
    ax.annotate("waist: several cells —\nz is made here", xy=(-290, 0.05),
                xytext=(-1300, 2.35), color=GOLD, fontsize=8.6, ha="right",
                va="center",
                arrowprops=dict(arrowstyle="->", color=GOLD, lw=0.9,
                                shrinkA=3, shrinkB=3))
    ax.annotate("anchor column (always inside)", xy=(0, -4.6),
                xytext=(380, -6.3), color=TEXT, fontsize=8.2, ha="left",
                va="center",
                arrowprops=dict(arrowstyle="->", color=TEXT, lw=0.8,
                                shrinkA=2, shrinkB=3))

    # ---- panel B (tilted)
    ax = axB
    cone(ax, drift=430.0, lw=1.0)
    ax.annotate("", xy=(-2580, -6.0), xytext=(0, 0),
                arrowprops=dict(arrowstyle="-|>", color=GOLD, lw=1.6,
                                shrinkA=3, shrinkB=2))
    ax.text(-3450, -7.2, "drift  d", color=GOLD, fontsize=9, ha="center",
            fontweight="bold")
    ax.set_xlim(-6500, 6500)
    ax.set_ylim(-8.7, 8.7)
    ax.set_xticks([-6000, 0, 6000])
    ax.set_xticklabels(["−6000", "0", "6000"])
    ax.set_yticks(range(-6, 7, 2))
    ax.set_xlabel("km", color=MUTED, fontsize=8.5)
    ax.grid(color=GRID, lw=0.45, alpha=0.55)
    ax.set_axisbelow(True)
    style(ax)
    ax.set_title("a cone the model may tilt:\ndrift d", color=TEXT,
                 fontsize=10.0, pad=6)

    fig.savefig(f"{OUT}/hourglass_spacetime.png", dpi=200)
    plt.close(fig)


# ---------------------------------------------------------------- figure 2
GOLDEN = np.deg2rad(137.508)


def sunflower_pts(n=24, phase=0.0):
    i = np.arange(n)
    r = np.sqrt((i + 0.5) / n)
    th = i * GOLDEN + phase
    return r * np.cos(th), r * np.sin(th)


def warped_sunflower():
    fig = plt.figure(figsize=(12.0, 2.85))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.0, 1.0, 1.75], left=0.025,
                          right=0.985, top=0.870, bottom=0.115, wspace=0.16)
    ux, uy = sunflower_pts(24)

    # (a) canonical
    ax = fig.add_subplot(gs[0, 0])
    ax.add_patch(Circle((0, 0), 1.0, facecolor=BLUE, alpha=0.13,
                        edgecolor=BLUE, lw=1.0))
    ax.scatter(ux, uy, s=17, c=BLUE, edgecolors=BG, linewidths=0.4, zorder=3)
    ax.plot([0], [0], marker="o", ms=6, mfc=GOLD, mec=BG, mew=0.8, zorder=4)
    ax.set_title("(a)  canonical sunflower, unit disc\n24 points, golden angle 137.508°",
                 color=TEXT, fontsize=8.6, pad=5)
    ax.set_xlim(-1.35, 1.35); ax.set_ylim(-1.3, 1.3)
    ax.set_aspect("equal"); ax.axis("off")

    # (b) rotated + stretched
    ax = fig.add_subplot(gs[0, 1])
    th = np.deg2rad(30.0)
    R = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
    P = R @ np.vstack([1.30 * ux, 0.65 * uy])
    ax.add_patch(Ellipse((0, 0), 2 * 1.30, 2 * 0.65, angle=30, facecolor=BLUE,
                         alpha=0.13, edgecolor=BLUE, lw=1.0))
    ax.scatter(P[0], P[1], s=17, c=BLUE, edgecolors=BG, linewidths=0.4, zorder=3)
    ax.plot([0], [0], marker="o", ms=6, mfc=GOLD, mec=BG, mew=0.8, zorder=4)
    # theta arc
    ax.plot([0, 1.30], [0, 0], color=MUTED, lw=0.8, ls=(0, (3, 2)))
    ax.plot([0, 1.30 * np.cos(th)], [0, 1.30 * np.sin(th)], color=GOLD, lw=1.0)
    ax.text(0.86, 0.17, "θ", color=GOLD, fontsize=10)
    ax.text(1.02, -0.50, "a", color=TEXT, fontsize=9, style="italic")
    ax.text(-0.52, 0.52, "b", color=TEXT, fontsize=9, style="italic")
    ax.set_title("(b)  rotated by θ = 30°, stretched to 2:1\nsemi-axes a, b",
                 color=TEXT, fontsize=8.6, pad=5)
    ax.set_xlim(-1.75, 1.75); ax.set_ylim(-1.3, 1.3)
    ax.set_aspect("equal"); ax.axis("off")
    ax.text(0.5, -0.085, "S = log Σ = m·I + k·[cos 2θ, sin 2θ; sin 2θ, −cos 2θ]",
            transform=ax.transAxes, color=GOLD, fontsize=7.8, ha="center",
            va="top", family="monospace")

    # (c) three lags
    ax = fig.add_subplot(gs[0, 2])
    d = np.array([-1.00, 0.62])             # per-lag drift, toward upper-left
    shades = {1: "#79BBD2", 3: BLUE, 6: "#1F6FEB"}
    for lag in (6, 3, 1):
        c = d * lag
        a, b = 0.62 + 0.30 * lag, 0.34 + 0.15 * lag
        ax.add_patch(Ellipse(c, 2 * a, 2 * b, angle=30, facecolor=shades[lag],
                             alpha=0.16, edgecolor=shades[lag], lw=1.1, zorder=2))
        vx, vy = sunflower_pts(24, phase=lag * GOLDEN)   # phase per lag
        P = R @ np.vstack([a * vx, b * vy])
        ax.scatter(P[0] + c[0], P[1] + c[1], s=11, c=shades[lag],
                   edgecolors=BG, linewidths=0.3, zorder=3)
        if lag == 1:
            ax.text(c[0] + a + 0.35, c[1] - 0.10, f"ℓ = {lag}", color=shades[lag],
                    fontsize=8.6, ha="left", va="center", fontweight="bold",
                    zorder=4)
        else:
            ax.text(c[0], c[1] - b - 0.42, f"ℓ = {lag}", color=shades[lag],
                    fontsize=8.6, ha="center", fontweight="bold", zorder=4)
    ax.annotate("", xy=(d[0] * 2.45, d[1] * 2.45), xytext=(0, 0),
                arrowprops=dict(arrowstyle="-|>", color=GOLD, lw=1.7,
                                shrinkA=4, shrinkB=2), zorder=5)
    ax.text(0.30, 2.25, "d — toward the source", color=GOLD, fontsize=8.8,
            ha="left", fontweight="bold", zorder=6)
    ax.plot([0], [0], marker="o", ms=7, mfc=GOLD, mec=BG, mew=0.9, zorder=6)
    ax.text(-0.40, -1.15, "anchor (0, 0) — a fixed dot\nat every lag",
            color=TEXT, fontsize=8.4, ha="right", va="center", zorder=6)
    ax.set_title("(c)  the same pattern at three lags: shifted by ℓ·d, grown with ℓ",
                 color=TEXT, fontsize=8.6, pad=5)
    ax.set_xlim(-9.8, 5.0); ax.set_ylim(-2.25, 5.55)
    ax.set_aspect("equal"); ax.axis("off")
    ax.text(0.5, -0.085,
            "phase rotated by the golden angle per lag — the slices interleave in space-time",
            transform=ax.transAxes, color=GOLD, fontsize=7.8, ha="center",
            va="top")

    fig.savefig(f"{OUT}/warped_sunflower.png", dpi=200)
    plt.close(fig)


# ------------------------------------------------- figure 2b: channel apertures
def _ap_geom():
    """Shared geometry for the two aperture figures."""
    ux, uy = sunflower_pts(24)
    LAG = 3
    th = np.deg2rad(30.0)
    R = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
    A, B = 0.62 + 0.30 * LAG, 0.34 + 0.15 * LAG          # 1.52, 0.79
    d = np.array([-1.00, -0.62])                          # drift: south-west
    c = d * LAG
    P = (R @ np.vstack([A * ux, B * uy])).T               # dots, centred on c
    XY = P + c
    cmap = LinearSegmentedColormap.from_list(
        "ap", ["#16202C", "#1F4F86", "#4493F8", "#CFE0F7"])

    def weights(scale):
        Sinv = np.linalg.inv(R @ np.diag([(A * scale) ** 2, (B * scale) ** 2]) @ R.T)
        return np.exp(-0.5 * np.einsum("ij,jk,ik->i", P, Sinv, P))

    return A, B, c, P, XY, cmap, weights


def channel_apertures_ab():
    """(a) one dot set, three apertures.  (b) the same dots, weighted."""
    A, B, c, P, XY, cmap, weights = _ap_geom()

    fig = plt.figure(figsize=(12.2, 4.05))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.21, 2.24],
                          left=0.012, right=0.992, top=0.885, bottom=0.015,
                          wspace=0.045)

    # ---------------- (a) one dot set, three apertures
    ax = fig.add_subplot(gs[0, 0])
    ax.add_patch(Ellipse(c, 2 * A * 1.60, 2 * B * 1.60, angle=30,
                         facecolor="#9ACBB0", alpha=0.08, edgecolor="none",
                         zorder=1))
    ax.scatter(XY[:, 0], XY[:, 1], s=22, c="#7D8590", edgecolors=BG,
               linewidths=0.4, zorder=3)
    for sc, col, ls, lw in ((1.60, "#9ACBB0", (0, (5, 2.5)), 2.4),
                            (1.00, BLUE, "solid", 3.0),
                            (0.62, GOLD, (0, (1.5, 1.8)), 2.6)):
        ax.add_patch(Ellipse(c, 2 * A * sc, 2 * B * sc, angle=30,
                             facecolor="none", edgecolor=col, lw=lw, ls=ls,
                             zorder=6))
    ax.annotate("", xy=(c[0] * 0.90, c[1] * 0.90), xytext=(0, 0),
                arrowprops=dict(arrowstyle="-|>", color=TEXT, lw=2.0,
                                shrinkA=5, shrinkB=3), zorder=3)
    ax.text(-1.22, 0.02, "ℓ·d", color=TEXT, fontsize=13.5, ha="right",
            va="center", fontweight="bold", zorder=7)
    ax.plot([0], [0], marker="o", ms=11, mfc=GOLD, mec=BG, mew=1.1, zorder=5)
    ax.text(0.34, 0.14, "anchor", color=GOLD, fontsize=12.0, ha="left", zorder=5)
    ax.set_title("(a)  one dot set at lag 3, three apertures",
                 color=TEXT, fontsize=14.0, pad=9)
    ax.set_xlim(-5.8, 2.2); ax.set_ylim(-6.0, 0.6)
    ax.set_aspect("equal"); ax.axis("off")
    leg = [
        Line2D([], [], color=BLUE, lw=3.0,
               label="currents, sea-surface height · s = 0"),
        Line2D([], [], color="#9ACBB0", lw=2.4, ls=(0, (5, 2.5)),
               label="sea-surface temperature at lags 0–1 · s > 0\n(stirred by the atmosphere)"),
        Line2D([], [], color=GOLD, lw=2.4, ls=(0, (1.5, 1.8)),
               label="a channel with s < 0"),
    ]
    lg = ax.legend(handles=leg, loc="lower right", fontsize=10.6, frameon=True,
                   facecolor="#161B22", edgecolor=GRID, labelcolor=TEXT,
                   borderpad=0.5, handlelength=1.9, labelspacing=0.55,
                   bbox_to_anchor=(1.02, -0.02))
    lg.set_zorder(9)

    # ---------------- (b) the same dots, coloured by the aperture weight
    ax = fig.add_subplot(gs[0, 1])
    SEP = 7.8
    for j, (scale, lab) in enumerate(((1.00, "currents: the inner dots count"),
                                      (1.60, "SST at lag 1: more dots count"))):
        w = weights(scale)
        off = j * SEP
        ax.add_patch(Ellipse((c[0] + off, c[1]), 2 * A * scale, 2 * B * scale,
                             angle=30, facecolor="none", edgecolor="#8B949E",
                             lw=1.3, ls=(0, (2.5, 2)), zorder=2))
        ax.scatter(XY[:, 0] + off, XY[:, 1], s=46 + 108 * w, c=w, cmap=cmap,
                   vmin=0, vmax=1, edgecolors=BG, linewidths=0.55, zorder=4)
        ax.plot([off], [0], marker="o", ms=10, mfc=GOLD, mec=BG, mew=1.0,
                zorder=5)
        if j == 0:
            ax.text(0.34, 0.14, "anchor", color=GOLD, fontsize=11.0, ha="left",
                    zorder=5)
        ax.text(c[0] + off, -4.30, lab, color=TEXT, fontsize=12.0, ha="center",
                va="center", zorder=5)
    ax.set_title("(b)  the same dots, coloured by w(c, i) = exp(−½ pᵀ Σ_c(ℓ)⁻¹ p)",
                 color=TEXT, fontsize=14.0, pad=9)
    ax.set_xlim(-6.0, 8.8); ax.set_ylim(-5.6, 1.0)
    ax.set_aspect("equal"); ax.axis("off")
    cax = ax.inset_axes([0.355, 0.015, 0.29, 0.058])
    cax.imshow(np.linspace(0, 1, 256).reshape(1, -1), aspect="auto", cmap=cmap)
    cax.set_xticks([]); cax.set_yticks([])
    for sp in cax.spines.values():
        sp.set_color(GRID); sp.set_linewidth(0.8)
    cax.text(-0.05, 0.5, "0  dim", transform=cax.transAxes, color=MUTED,
             fontsize=10.6, ha="right", va="center")
    cax.text(1.05, 0.5, "bright  1", transform=cax.transAxes, color=MUTED,
             fontsize=10.6, ha="left", va="center")

    fig.savefig(f"{OUT}/channel_apertures_ab.png", dpi=200)
    plt.close(fig)


def channel_apertures_c():
    """(c) token anatomy: one token per dot, weighted per channel."""
    _, _, _, _, _, cmap, _ = _ap_geom()

    fig = plt.figure(figsize=(12.2, 3.30))
    ax = fig.add_axes([0.0, 0.0, 1.0, 1.0])
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
    ax.set_facecolor(BG)

    CH = ["cur", "ssh", "sst", "mld"]
    # matches the worked example in the inset: currents/ssh 0.30, sst/mld 0.73
    WROW = [0.30, 0.30, 0.73, 0.73]
    CW, GAP, BGAP, X0 = 0.0585, 0.005, 0.022, 0.072

    def cell_x(i):
        return X0 + i * (CW + GAP) + (BGAP if i >= 4 else 0.0)

    def strip(y, h, tops, bots, fills, edge, tcol=None, botbold=False):
        for i in range(8):
            fl = fills[i]
            rgb = fl[:3] if isinstance(fl, tuple) else (0.1, 0.13, 0.17)
            lum = 0.299 * rgb[0] + 0.587 * rgb[1] + 0.114 * rgb[2]
            t1 = tcol or ("#0B1219" if lum > 0.45 else TEXT)
            t2 = (tcol and MUTED) if not botbold else t1
            if not botbold and not tcol:
                t2 = "#0B1219" if lum > 0.45 else MUTED
            ax.add_patch(Rectangle((cell_x(i), y), CW, h, facecolor=fl,
                                   edgecolor=edge, lw=1.3, zorder=3))
            ax.text(cell_x(i) + CW / 2, y + h * 0.66, tops[i], color=t1,
                    fontsize=11.0 if botbold else 11.5, ha="center",
                    va="center", zorder=4)
            ax.text(cell_x(i) + CW / 2, y + h * 0.26, bots[i], color=t2,
                    fontsize=11.0 if botbold else 10.5, ha="center",
                    va="center", zorder=4,
                    fontweight="bold" if botbold else "normal")

    strip(0.475, 0.245, ["value"] * 4 + ["obs"] * 4, CH * 2,
          ["#1B2733"] * 4 + ["#151D27"] * 4, BLUE, TEXT)
    strip(0.105, 0.245, [f"w {ch},i" for ch in CH * 2],
          [f"{w:.2f}" for w in WROW] * 2,
          [cmap(w) for w in WROW] * 2, "#8B949E", botbold=True)
    ax.text(0.040, 0.475, "×", color=GOLD, fontsize=26, ha="center",
            va="center", fontweight="bold")
    ax.text(X0, 0.855, "one per-location token at dot i", color=MUTED,
            fontsize=11.5, ha="left", va="center")
    ax.text(X0, 0.965, "(c)  token anatomy", color=TEXT, fontsize=14.0,
            ha="left", va="center")

    # arrow to the projection
    ax.annotate("", xy=(0.655, 0.475), xytext=(0.596, 0.475),
                arrowprops=dict(arrowstyle="-|>", color=GOLD, lw=2.6))
    ax.text(0.664, 0.475, "projection", color=GOLD, fontsize=12.0, ha="left",
            va="center", fontweight="bold")
    ax.text(0.606, 0.345, "the same dot,\nweighted per channel,\nbefore the projection",
            color=MUTED, fontsize=10.2, ha="left", va="top", linespacing=1.35)

    # worked example, right end
    BX0, BX1, BY0, BY1 = 0.752, 0.993, 0.050, 0.900
    ax.add_patch(Rectangle((BX0, BY0), BX1 - BX0, BY1 - BY0,
                           facecolor="#161B22", edgecolor=BLUE, lw=1.4,
                           zorder=3))
    ax.text((BX0 + BX1) / 2, 0.815, "dot at 400 km, lag 1", color=TEXT,
            fontsize=11.8, ha="center", va="center", fontweight="bold",
            zorder=4)
    for yy, wv, lab in ((0.625, 0.30, "currents  σ = 259 km → w ≈ 0.30"),
                        (0.455, 0.73, "SST         σ = 500 km → w ≈ 0.73"),
                        (0.225, 0.14,
                         "wind stress, lag 4:\ntime factor e$^{-20/10}$ ≈ 0.14")):
        ax.plot([BX0 + 0.018], [yy], marker="o", ms=9, mfc=cmap(wv),
                mec="#8B949E", mew=0.8, zorder=4)
        ax.text(BX0 + 0.033, yy, lab, color=TEXT, fontsize=10.5, ha="left",
                va="center", zorder=4, linespacing=1.45)

    fig.savefig(f"{OUT}/channel_apertures_c.png", dpi=200)
    plt.close(fig)


# ---------------------------------------------------------------- figure 3
US_EAST = [(-80.05, 25.20), (-80.10, 26.70), (-80.55, 27.50), (-80.50, 28.50),
           (-80.95, 29.50), (-81.30, 30.40), (-81.50, 31.20), (-80.90, 32.10),
           (-79.90, 32.80), (-78.90, 33.80), (-77.90, 34.20), (-76.50, 34.70),
           (-75.50, 35.20), (-75.90, 36.60), (-75.10, 38.50), (-74.00, 39.50),
           (-73.90, 40.60)]
GULF = [(-85.00, 29.80), (-84.30, 30.00), (-83.60, 29.90), (-83.00, 29.20),
        (-82.60, 28.60), (-82.70, 27.80), (-82.10, 26.90), (-81.80, 26.00),
        (-81.10, 25.10), (-80.05, 25.20)]
CUBA = [(-84.95, 21.90), (-84.00, 21.90), (-82.80, 21.60), (-81.80, 22.10),
        (-80.50, 22.10), (-79.30, 22.40), (-77.80, 22.00), (-77.00, 21.00),
        (-75.80, 20.70), (-74.30, 20.10), (-74.10, 20.60), (-75.60, 21.10),
        (-76.80, 21.60), (-78.50, 22.40), (-79.80, 22.90), (-81.20, 23.05),
        (-82.60, 23.15), (-83.90, 22.90)]
BAHAMAS = [(-78.30, 24.60, 0.55, 1.15), (-78.55, 26.60, 0.85, 0.28),
           (-77.20, 26.40, 0.75, 0.30), (-76.30, 25.25, 0.28, 0.75),
           (-75.35, 23.55, 0.28, 0.60), (-75.55, 24.45, 0.25, 0.42),
           (-74.50, 22.50, 0.30, 0.35)]

LAT0 = 30.0
KMLAT = 111.0


def deg(km_x, km_y, lat=LAT0):
    return km_x / (KMLAT * np.cos(np.deg2rad(lat))), km_y / KMLAT


def amoc_cones():
    fig = plt.figure(figsize=(6.05, 5.30))
    ax = fig.add_axes([0.075, 0.055, 0.905, 0.845])
    ax.set_facecolor("#0B1219")

    alat, alon = 27.0, -77.0

    def axes_of(lag):
        return 0.46 + 0.215 * lag, 0.30 + 0.125 * lag

    # --- surface past cones: filled, UNDER the land
    SURF = ((-80.0 + 77.0) / 6.0, (24.0 - 27.0) / 6.0)   # upstream: south-west
    for lag in (6, 3, 1):
        c = (alon + SURF[0] * lag, alat + SURF[1] * lag)
        a, b = axes_of(lag)
        ax.add_patch(Ellipse(c, 2 * a, 2 * b, facecolor=BLUE, alpha=0.26,
                             edgecolor=BLUE, lw=1.1, zorder=2))
        if lag != 1:
            ax.text(c[0] - a - 0.20, c[1], f"\u2212{lag}", color=BLUE,
                    fontsize=8.0, ha="right", va="center", fontweight="bold",
                    zorder=9)

    # --- land
    land = US_EAST + [(-73.90, 41.5), (-85.0, 41.5)] + GULF
    ax.add_patch(Polygon(land, closed=True, facecolor="#1C232B",
                         edgecolor="#4A5460", lw=0.9, zorder=5))
    ax.add_patch(Polygon(CUBA, closed=True, facecolor="#1C232B",
                         edgecolor="#4A5460", lw=0.9, zorder=5))
    for lon, lat, w, h in BAHAMAS:
        ax.add_patch(Ellipse((lon, lat), 2 * w, 2 * h, facecolor="#1C232B",
                             edgecolor="#4A5460", lw=0.7, zorder=5))
    ax.text(-83.2, 34.6, "UNITED\nSTATES", color="#6E7781", fontsize=7.6,
            ha="center", va="center", zorder=6)
    ax.text(-76.6, 21.05, "CUBA", color="#6E7781", fontsize=7.4, ha="center",
            zorder=6)
    ax.text(-84.7, 38.9, "schematic coastline \u2014 not a projection",
            color="#6E7781", fontsize=7.0, ha="left", va="center", zorder=6)

    # --- surface future cones: dashed outlines, OVER the land
    for lag in (6, 3, 1):
        c = (alon - SURF[0] * lag, alat - SURF[1] * lag)
        a, b = axes_of(lag)
        ax.add_patch(Ellipse(c, 2 * a, 2 * b, facecolor="none",
                             edgecolor="#9EC7FF", lw=1.1, ls=(0, (3.5, 2.5)),
                             alpha=0.95, zorder=6))
        if lag != 1:
            ax.text(c[0], c[1], f"+{lag}", color="#9EC7FF", fontsize=7.6,
                    ha="center", va="center", fontweight="bold", zorder=7)

    # --- deep Argo: NOT ellipses, the nearest profile tokens
    R_SEARCH = 420.0
    rsx, rsy = deg(R_SEARCH, R_SEARCH, alat)
    rng = np.random.RandomState(7)
    land_path = MplPath(land)
    cuba_path = MplPath(CUBA)
    pts = []
    while len(pts) < 46:
        t = rng.uniform(0, 2 * np.pi)
        r = np.sqrt(rng.uniform(0, 1))
        p = (alon + rsx * r * np.cos(t), alat + rsy * r * np.sin(t))
        if land_path.contains_point(p) or cuba_path.contains_point(p):
            continue
        if abs(p[0] - alon) < 0.30 and abs(p[1] - alat) < 0.30:
            continue
        pts.append(p)
    pts = np.array(pts)
    s0, s1 = np.array([-79.0, 26.0]), np.array([-74.5, 33.0])   # the slope
    u = (s1 - s0) / np.linalg.norm(s1 - s0)
    rel = pts - s0
    perp = np.abs(rel[:, 0] * u[1] - rel[:, 1] * u[0])
    attended = (perp < 0.95) & (pts[:, 1] > alat + 0.55)
    ax.add_patch(Ellipse((alon, alat), 2 * rsx, 2 * rsy, facecolor="none",
                         edgecolor=ORANGE, lw=0.9, ls=(0, (1.6, 2.2)),
                         alpha=0.55, zorder=6))
    ax.scatter(pts[~attended, 0], pts[~attended, 1], s=18, c=ORANGE,
               alpha=0.42, edgecolors="none", zorder=7)
    ax.scatter(pts[attended, 0], pts[attended, 1], s=66, c=ORANGE,
               edgecolors="#FFD9C9", linewidths=0.8, zorder=8)
    ax.annotate("nearest-profile\nsearch radius", xy=(-79.55, 24.35),
                xytext=(-84.7, 22.3), color=ORANGE, fontsize=7.0, ha="left",
                va="center", zorder=9, alpha=0.9,
                arrowprops=dict(arrowstyle="->", color=ORANGE, lw=0.8,
                                alpha=0.7, shrinkA=2, shrinkB=2))

    # --- v2 safety ring
    rx, ry = deg(700.0, 700.0, alat)
    ax.add_patch(Ellipse((alon, alat), 2 * rx, 2 * ry, facecolor="none",
                         edgecolor="#8B949E", lw=1.0, ls=(0, (2, 2)),
                         alpha=0.85, zorder=6))
    ax.annotate("v2 safety ring (fixed)", xy=(-82.05, 31.45),
                xytext=(-84.7, 36.9), color="#8B949E", fontsize=7.6,
                ha="left", va="center", zorder=9,
                arrowprops=dict(arrowstyle="->", color="#8B949E", lw=0.8,
                                shrinkA=2, shrinkB=2))

    # --- RAPID line
    ax.plot([-85, -60], [26.5, 26.5], color=GOLD, lw=1.2, ls=(0, (5, 3)),
            zorder=7)
    ax.text(-60.35, 26.75, "RAPID 26.5\u00b0N", color=GOLD, fontsize=8.0,
            ha="right", va="bottom", zorder=9)

    # --- arrows
    ax.annotate("", xy=(-79.3, 24.7), xytext=(alon - 0.30, alat - 0.30),
                arrowprops=dict(arrowstyle="-|>", color=BLUE, lw=2.0), zorder=8)
    ax.annotate("", xy=(-75.5, 31.6), xytext=(alon + 0.22, alat + 0.55),
                arrowprops=dict(arrowstyle="-|>", color=ORANGE, lw=2.0), zorder=9)
    ax.annotate("deep profiles the model should\nattend to: north, up the DWBC",
                xy=(-75.35, 31.95), xytext=(-73.1, 35.6), color=ORANGE,
                fontsize=7.8, ha="left", va="center", zorder=11,
                fontweight="bold",
                arrowprops=dict(arrowstyle="->", color=ORANGE, lw=0.9,
                                shrinkA=3, shrinkB=3))

    # --- anchor
    ax.plot([alon], [alat], marker="o", ms=9.0, mfc=GOLD, mec="#0B1219",
            mew=1.2, zorder=10)
    ax.text(-73.4, 27.05, "anchor  27\u00b0N, 77\u00b0W", color=GOLD, fontsize=8.2,
            ha="center", va="center", zorder=11, fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.28", facecolor="#0B1219",
                      edgecolor=GRID, linewidth=0.7, alpha=0.92))
    ax.annotate("", xy=(alon + 0.28, alat + 0.03), xytext=(-75.05, 27.05),
                arrowprops=dict(arrowstyle="->", color=GOLD, lw=0.8,
                                shrinkA=1, shrinkB=1), zorder=10)

    ax.set_xlim(-85, -60)
    ax.set_ylim(20, 40)
    ax.set_aspect(1.0 / np.cos(np.deg2rad(LAT0)))
    ax.set_xticks([-85, -80, -75, -70, -65, -60])
    ax.set_xticklabels(["85\u00b0W", "80\u00b0W", "75\u00b0W", "70\u00b0W",
                        "65\u00b0W", "60\u00b0W"], fontsize=7.4)
    ax.set_yticks([20, 25, 30, 35, 40])
    ax.set_yticklabels(["20\u00b0N", "25\u00b0N", "30\u00b0N", "35\u00b0N",
                        "40\u00b0N"], fontsize=7.4)
    ax.grid(color=GRID, lw=0.4, alpha=0.45)
    ax.set_axisbelow(True)
    style(ax)
    ax.set_title("pre-registered expectation, not a result", color=TEXT,
                 fontsize=10.5, pad=9)

    handles = [
        Line2D([], [], color=BLUE, lw=7, alpha=0.6,
               label="surface: learned d = upstream = south-west\n"
                     "past \u2212\u2113 filled \u00b7 future +\u2113 dashed \u00b7 lags 1, 3, 6"),
        Line2D([], [], color="none", marker="o", ms=7, mfc=ORANGE,
               mec="#FFD9C9", mew=0.8,
               label="deep Argo: no ellipse \u2014 attention over the nearest\n"
                     "profiles, expected to favour the north"),
    ]
    leg = ax.legend(handles=handles, loc="lower right", fontsize=7.2,
                    frameon=True, facecolor="#161B22", edgecolor=GRID,
                    labelcolor=TEXT, borderpad=0.55, handlelength=1.6,
                    borderaxespad=0.5, labelspacing=0.7)
    leg.set_zorder(12)

    fig.savefig(f"{OUT}/amoc_cones.png", dpi=200)
    plt.close(fig)


# ------------------------------------------- figure 6: the summary slide's cone
def summary_cone():
    """A compact re-arrangement of warped_sunflower() for slide 21.

    Same dots, same phase rotation, same drift-and-grow rule — only the
    arrangement and the label budget differ, because this one is rendered at
    3.65 in instead of 10.75 in and travels on its own. Aspect 4.0 / 2.52 =
    1.587, so it lands exactly 2.30 in tall at that width.
    """
    fig = plt.figure(figsize=(4.0, 2.52))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.0, 1.30],
                          left=0.015, right=0.985, top=0.885, bottom=0.045,
                          wspace=0.04, hspace=0.78)
    ux, uy = sunflower_pts(24)
    th = np.deg2rad(30.0)
    R = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])

    # (a) the canonical sunflower on the unit disc
    ax = fig.add_subplot(gs[0, 0])
    ax.add_patch(Circle((0, 0), 1.0, facecolor=BLUE, alpha=0.13,
                        edgecolor=BLUE, lw=1.0))
    ax.scatter(ux, uy, s=13, c=BLUE, edgecolors=BG, linewidths=0.35, zorder=3)
    ax.plot([0], [0], marker="o", ms=5, mfc=GOLD, mec=BG, mew=0.7, zorder=4)
    ax.set_title("(a)  24 points, golden angle", color=TEXT, fontsize=8.5, pad=4)
    ax.set_xlim(-1.35, 1.35); ax.set_ylim(-1.25, 1.25)
    ax.set_aspect("equal"); ax.axis("off")

    # (b) rotated by theta and stretched to 2:1
    ax = fig.add_subplot(gs[0, 1])
    P = R @ np.vstack([1.30 * ux, 0.65 * uy])
    ax.add_patch(Ellipse((0, 0), 2 * 1.30, 2 * 0.65, angle=30, facecolor=BLUE,
                         alpha=0.13, edgecolor=BLUE, lw=1.0))
    ax.scatter(P[0], P[1], s=13, c=BLUE, edgecolors=BG, linewidths=0.35, zorder=3)
    ax.plot([0], [0], marker="o", ms=5, mfc=GOLD, mec=BG, mew=0.7, zorder=4)
    ax.plot([0, 1.30], [0, 0], color=MUTED, lw=0.8, ls=(0, (3, 2)))
    ax.plot([0, 1.30 * np.cos(th)], [0, 1.30 * np.sin(th)], color=GOLD, lw=1.0)
    ax.text(0.80, 0.14, "θ", color=GOLD, fontsize=9.5)
    ax.set_title("(b)  rotated, stretched 2:1", color=TEXT, fontsize=8.5, pad=4)
    ax.set_xlim(-1.75, 1.75); ax.set_ylim(-1.25, 1.25)
    ax.set_aspect("equal"); ax.axis("off")

    # the matrix-logarithm caption, centred under the top row
    fig.text(0.5, 0.505, "S = log Σ = m·I + k·[cos 2θ, sin 2θ; sin 2θ, −cos 2θ]",
             color=GOLD, fontsize=7.4, ha="center", va="center",
             family="monospace")

    # (c) the same pattern at three lags, shifted by l*d and grown with l
    gsc = gs[1, :].subgridspec(1, 2, width_ratios=[2.45, 1.0], wspace=0.02)
    ax = fig.add_subplot(gsc[0, 0])
    d = np.array([-1.00, 0.28])              # per-lag drift, toward the source
    shades = {1: "#79BBD2", 3: BLUE, 6: "#1F6FEB"}
    for lag in (6, 3, 1):
        c = d * lag
        a, b = 0.55 + 0.26 * lag, 0.30 + 0.13 * lag
        ax.add_patch(Ellipse(c, 2 * a, 2 * b, angle=30, facecolor=shades[lag],
                             alpha=0.16, edgecolor=shades[lag], lw=1.0, zorder=2))
        vx, vy = sunflower_pts(24, phase=lag * GOLDEN)   # phase per lag
        P = R @ np.vstack([a * vx, b * vy])
        ax.scatter(P[0] + c[0], P[1] + c[1], s=6, c=shades[lag],
                   edgecolors=BG, linewidths=0.25, zorder=3)
        ax.text(c[0] + {1: 0.75, 3: -0.25, 6: -0.35}[lag], -1.30,
                f"ℓ = {lag}", color=shades[lag], fontsize=8.2, ha="center",
                va="top", fontweight="bold", zorder=4)
    ax.annotate("", xy=(d[0] * 2.7, d[1] * 2.7), xytext=(0, 0),
                arrowprops=dict(arrowstyle="-|>", color=GOLD, lw=1.6,
                                shrinkA=3, shrinkB=1), zorder=5)
    ax.text(d[0] * 2.7 - 0.35, d[1] * 2.7 + 0.45, "d", color=GOLD, fontsize=10,
            ha="center", va="center", fontweight="bold", zorder=6)
    ax.plot([0], [0], marker="o", ms=6, mfc=GOLD, mec=BG, mew=0.8, zorder=6)
    ax.text(0.30, -0.35, "anchor", color=TEXT, fontsize=8.2, ha="left",
            va="top", zorder=6)
    ax.set_title("(c)  the footprint at lags 1, 3, 6", color=TEXT, fontsize=8.5,
                 pad=3)
    ax.set_xlim(-8.3, 2.8); ax.set_ylim(-2.1, 3.5)
    ax.set_aspect("equal"); ax.axis("off")

    axt = fig.add_subplot(gsc[0, 1])
    axt.axis("off")
    axt.text(0.0, 1.00, "d — toward\nthe source", color=GOLD, fontsize=8.5,
             ha="left", va="top", fontweight="bold", transform=axt.transAxes,
             linespacing=1.3)
    axt.text(0.0, 0.55, "shifted by ℓ·d,\nwidened by ℓ²·Σ_v,\nphase-rotated\nper lag",
             color=TEXT, fontsize=7.8, ha="left", va="top",
             transform=axt.transAxes, linespacing=1.28)

    fig.savefig(f"{OUT}/summary_cone.png", dpi=200)
    plt.close(fig)


# --------------------------------------- figure 7: the summary slide's hourglass
def summary_hourglass():
    """Slide 21's hourglass: the same shape as figure 1, one panel, big labels.

    Figure 1 is drawn for a 7.25 in slot and its tick labels are unreadable at
    the summary's 3.65 in column, so this is a redraw with a label budget for
    that width — no ticks, axis WORDS only, three large labels. Same figsize
    and aspect as summary_cone (4.0 x 2.52 = 1.587), so the two pictures and
    the results panel share one baseline on the slide.
    """
    fig = plt.figure(figsize=(4.0, 2.52))
    ax = fig.add_axes([0.085, 0.115, 0.900, 0.845])
    _hourglass_shape(ax, 260.0, 455.0, drift=0.0, lw=1.2)

    ax.set_xlim(-5600, 3400)
    ax.set_ylim(-8.8, 8.8)
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_xlabel("space", color=MUTED, fontsize=9.5, labelpad=2)
    ax.set_ylabel("time:  past → present → future", color=MUTED, fontsize=9,
                  labelpad=3)
    style(ax)

    XC = -1100.0                      # the AXES centre, not the cone's
    ax.text(XC, 7.5, "future cone — targets only, never read", color=ORANGE,
            fontsize=10, ha="center", va="center", fontweight="bold", zorder=8)
    ax.text(XC, -7.5, "past cone — input, read", color=BLUE, fontsize=10,
            ha="center", va="center", fontweight="bold", zorder=8)
    ax.annotate("waist — several cells;\nthe embedding\nis made here",
                xy=(-290, 0.0), xytext=(-1400, 0.0), color=GOLD, fontsize=9.5,
                ha="right", va="center", fontweight="bold", linespacing=1.3,
                arrowprops=dict(arrowstyle="->", color=GOLD, lw=1.0,
                                shrinkA=4, shrinkB=2))

    fig.savefig(f"{OUT}/summary_hourglass.png", dpi=200)
    plt.close(fig)


# --------------------------------- figure 8: the same shape at two scales, in 3-D
# Stage 2's reach comes from ml/cone.py itself (pure numpy) rather than a second
# copy of the formula; the fallback exists only so this file still runs outside
# the repo, and it says so rather than pretending it imported.
try:
    import sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(HERE)))   # .../ml
    from cone import outer_reach_km, reach_km, ASPECT            # noqa: E402
except Exception as exc:                                         # pragma: no cover
    print(f"WARNING: ml/cone.py not importable ({exc}); using a local copy "
          "of reach_km / outer_reach_km / ASPECT for two_scales_3d()")
    ASPECT = 0.71

    def reach_km(family, lag, dt_days=5.0):
        return 0.3 * 86400.0 * dt_days * (1.0 + lag) / 1000.0

    def outer_reach_km(k, dt_days=5.0):
        return min(4444.0, max(111.0, 0.3 * 86400.0 * dt_days * (1.0 + k) / 1000.0))


_R30 = np.array([[np.cos(np.deg2rad(30.0)), -np.sin(np.deg2rad(30.0))],
                 [np.sin(np.deg2rad(30.0)),  np.cos(np.deg2rad(30.0))]])


def _footprint(k, reach, drift, sign, n_ring=72):
    """One lag's footprint: the ellipse outline and the 24 sunflower dots.

    `sign` is −1 for the past (below the present) and +1 for the future, which
    is the point mirror: centre −l*d against the past's +l*d, same ellipse.
    The dots are the deck's own sunflower, phase-rotated by the golden angle
    per lag exactly as warped_sunflower draws them.
    """
    a, b = reach, reach * ASPECT
    cx, cy = (-sign * k * drift[0], -sign * k * drift[1])
    t = np.linspace(0, 2 * np.pi, n_ring)
    ring = _R30 @ np.vstack([a * np.cos(t), b * np.sin(t)])
    ux, uy = sunflower_pts(24, phase=k * GOLDEN)
    dots = _R30 @ np.vstack([a * ux, b * uy])
    return (ring[0] + cx, ring[1] + cy), (dots[0] + cx, dots[1] + cy)


def _cone3d(ax, lags, reach_of, drift, colour, sign, dot_s, ribs=24,
            face_alpha=0.10, dots=True, lw=1.0):
    """Draw one half of the hourglass as stacked footprints plus rib lines.

    The surface is a FAN OF LINES rather than plot_surface: mplot3d sorts whole
    artists, so a translucent surface crossing the dots renders in the wrong
    order at some azimuths, while thin lines read correctly at every one.
    """
    rings = []
    for k in lags:
        r = reach_of(k)
        (rx, ry), (dx_, dy_) = _footprint(k, r, drift, sign)
        z = sign * k
        ax.plot(rx, ry, zs=z, zdir="z", color=colour, lw=lw, alpha=0.85,
                zorder=2)
        ax.add_collection3d(
            Poly3DCollection([list(zip(rx, ry, np.full_like(rx, float(z))))],
                             facecolor=colour, alpha=face_alpha,
                             edgecolor="none"))
        if dots:
            ax.scatter(dx_, dy_, z, s=dot_s, c=colour, depthshade=False,
                       edgecolors=BG, linewidths=0.25, zorder=4)
        rings.append((rx, ry, z))
    step = max(1, len(rings[0][0]) // ribs)
    for i in range(0, len(rings[0][0]) - 1, step):
        ax.plot([r[0][i] for r in rings], [r[1][i] for r in rings],
                [r[2] for r in rings], color=colour, lw=0.5, alpha=0.30,
                zorder=1)
    return rings


def _waist3d(ax, radius, drift_unused=None):
    t = np.linspace(0, 2 * np.pi, 72)
    ax.plot(radius * np.cos(t), radius * np.sin(t), zs=0, zdir="z", color=GOLD,
            lw=1.6, zorder=6)
    ax.add_collection3d(
        Poly3DCollection([list(zip(radius * np.cos(t), radius * np.sin(t),
                                   np.zeros_like(t)))],
                         facecolor=GOLD, alpha=0.40, edgecolor="none"))
    ax.scatter([0], [0], [0], s=14, c=GOLD, depthshade=False, zorder=7)


def _style3d(ax, xy_lim, xy_ticks, z_lim, z_ticks, z_ticklabels, xy_ticklabels):
    pane = (0.051, 0.067, 0.090, 1.0)                      # BG as RGBA
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.set_pane_color(pane)
        axis.line.set_color(MUTED)
        axis._axinfo["grid"]["color"] = (1, 1, 1, 0.06)
    ax.set_xlim(-xy_lim, xy_lim); ax.set_ylim(-xy_lim, xy_lim)
    ax.set_zlim(-z_lim, z_lim)
    ax.set_xticks(xy_ticks); ax.set_yticks(xy_ticks); ax.set_zticks(z_ticks)
    ax.set_xticklabels(xy_ticklabels, fontsize=8.5, color=MUTED)
    ax.set_yticklabels(xy_ticklabels, fontsize=8.5, color=MUTED)
    ax.set_zticklabels(z_ticklabels, fontsize=8.5, color=MUTED)
    # a negative pad makes the "−" of a tick label collide with the axis line
    ax.tick_params(axis="both", pad=1.5, length=0)
    # only ONE of the two horizontal axes is named: both are kilometres, and a
    # second "km" at the other bottom corner only crowds the tick numbers.
    ax.set_xlabel("km", color=MUTED, fontsize=9, labelpad=-6)
    ax.set_ylabel("")
    ax.set_zlabel("time (pentads)", color=MUTED, fontsize=9, labelpad=-2)
    ax.view_init(elev=22, azim=-55)
    ax.set_box_aspect((1, 1, 1.05))


def two_scales_3d():
    """Slide 22: the mirrored double cone in 3-D, at stage 1's and stage 2's
    scale. Same sunflower, same golden-angle phase and same three colours as
    slides 8 and 21; only the reach and the axis numbers differ."""
    fig = plt.figure(figsize=(11.0, 4.8))
    gs = fig.add_gridspec(1, 2, left=0.005, right=0.995, top=1.03, bottom=0.045,
                          wspace=0.02)

    # ------------------------------------------------------------- stage 1
    ax = fig.add_subplot(gs[0, 0], projection="3d")
    ax.set_facecolor(BG)
    L1 = list(range(1, 7))
    D1 = (-27.0, -27.0)                    # ~38 km per pentad, south-west
    _cone3d(ax, L1, lambda k: reach_km("B", k), D1, BLUE, -1, dot_s=9)
    _cone3d(ax, L1, lambda k: reach_km("B", k), D1, ORANGE, +1, dot_s=9)
    _waist3d(ax, 56.0)                     # ~13 cells at 0.25 degrees
    # a few past dots held out by a dropout pattern
    for k, idx in ((3, (1, 9, 17)), (4, (5, 21))):
        _, (dx_, dy_) = _footprint(k, reach_km("B", k), D1, -1)
        ax.scatter([dx_[i] for i in idx], [dy_[i] for i in idx], -k, s=26,
                   facecolors=BG, edgecolors=BLUE, linewidths=1.0,
                   depthshade=False, zorder=8)
    # the drift, once
    ax.plot([0, 6 * D1[0]], [0, 6 * D1[1]], [0, -6], color=GOLD, lw=1.6,
            zorder=9)
    _style3d(ax, 1450, [-1000, 1000], 7.6, [-6, 0, 6],
             ["−6", "0", "+6"], ["−1,000", "+1,000"])
    ax.set_title("stage 1 — the codec, raw values", color=TEXT, fontsize=11,
                 fontweight="bold", pad=-2, y=0.97)
    ax.text2D(0.50, 0.855, "future cone — targets", color=ORANGE, fontsize=9.5,
              fontweight="bold", ha="center", transform=ax.transAxes)
    ax.text2D(0.50, 0.075, "past cone — input", color=BLUE, fontsize=9.5,
              fontweight="bold", ha="center", transform=ax.transAxes,
              bbox=dict(facecolor=BG, edgecolor="none", pad=1.5))
    ax.text2D(0.035, 0.50, "waist — present\n(≈ 13 cells)", color=GOLD,
              fontsize=9.5, fontweight="bold", ha="left", va="center",
              linespacing=1.3, transform=ax.transAxes)
    ax.text2D(0.035, 0.255, "d — toward the source", color=GOLD, fontsize=9,
              ha="left", transform=ax.transAxes,
              bbox=dict(facecolor=BG, edgecolor="none", pad=1.5))
    ax.text2D(0.020, 0.800, "hollow dots =\nheld out (dropout)", color=BLUE,
              fontsize=9, ha="left", va="center", linespacing=1.3,
              transform=ax.transAxes)

    # ------------------------------------------------------------- stage 2
    ax = fig.add_subplot(gs[0, 1], projection="3d")
    ax.set_facecolor(BG)
    # lags 7...143 (the spec's stage-2 window), sampled DENSELY where the cone
    # still opens and sparsely above it: reach grows as 129.6*(1+k) km until it
    # meets the 4,444 km cap at lag 33, so a uniform "every dozen lags" spacing
    # spends nine of eleven rings on the straight part and the taper vanishes.
    L2 = [7, 12, 18, 24, 33, 72, 143]
    K_CAP = 33                                   # 129.6*(1+33) = 4,406 ~ cap
    # the lean is the same FRACTION of the reach as at stage 1 (~25% at the
    # mouth), not the same km per pentad — at 38 km/pentad a 143-pentad cone
    # would sit 5,400 km off its own anchor and read as a shear, not a cone.
    D2 = (-5.4, -5.4)
    _cone3d(ax, L2, outer_reach_km, D2, BLUE, -1, dot_s=3.5, face_alpha=0.085)
    _cone3d(ax, L2, outer_reach_km, D2, ORANGE, +1, dot_s=3.5, face_alpha=0.085)
    _waist3d(ax, 56.0)
    # where the opening stops: the cap ring, drawn once on each side
    for sgn, col in ((-1, BLUE), (+1, ORANGE)):
        (rx, ry), _ = _footprint(K_CAP, outer_reach_km(K_CAP), D2, sgn)
        ax.plot(rx, ry, zs=sgn * K_CAP, zdir="z", color=col, lw=1.8,
                alpha=1.0, zorder=5)
    # stage 1's whole double cone, to scale, at the centre
    for sgn in (-1, +1):
        for k in (3, 6):
            (rx, ry), _ = _footprint(k, reach_km("B", k), (0.0, 0.0), sgn)
            ax.plot(rx, ry, zs=sgn * k, zdir="z", color=TEXT, lw=0.8,
                    alpha=0.9, zorder=9)
            ax.add_collection3d(
                Poly3DCollection([list(zip(rx, ry,
                                           np.full_like(rx, float(sgn * k))))],
                                 facecolor=BG, alpha=0.85, edgecolor="none"))
    _style3d(ax, 6300, [-4444, 4444], 178, [-143, 0, 143],
             ["−143", "0", "+143"], ["−4,444", "+4,444"])
    ax.set_title("stage 2 — the forecaster, embeddings", color=TEXT,
                 fontsize=11, fontweight="bold", pad=-2, y=0.97)
    ax.text2D(0.50, 0.855, "future cone — targets", color=ORANGE, fontsize=9.5,
              fontweight="bold", ha="center", transform=ax.transAxes)
    ax.text2D(0.50, 0.075, "past cone — input", color=BLUE, fontsize=9.5,
              fontweight="bold", ha="center", transform=ax.transAxes,
              bbox=dict(facecolor=BG, edgecolor="none", pad=1.5))
    ax.annotate("stage 1, to scale —\nthe near field\nthe codec already read",
                xy=(0.485, 0.487), xytext=(0.005, 0.255), xycoords="axes fraction",
                textcoords="axes fraction", color=TEXT, fontsize=9, ha="left",
                va="center", linespacing=1.3,
                arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.8,
                                shrinkA=2, shrinkB=2))
    ax.text2D(0.010, 0.800, "reach stops growing at\nthe 4,444 km cap (lag 33)",
              color=MUTED, fontsize=8.5, ha="left", va="center",
              linespacing=1.3, transform=ax.transAxes)
    ax.text2D(0.010, 0.680, "each dot is\nan embedding", color=BLUE, fontsize=9,
              ha="left", va="center", linespacing=1.3, transform=ax.transAxes)

    fig.text(0.5, 0.016, "same shape  ·  ×24 in time  ·  ×5 in space",
             color=MUTED, fontsize=9.5, ha="center", va="center")

    fig.savefig(f"{OUT}/two_scales_3d.png", dpi=200)
    plt.close(fig)


hourglass()
warped_sunflower()
channel_apertures_ab()
channel_apertures_c()
amoc_cones()
summary_cone()
summary_hourglass()
two_scales_3d()
print("figures written to", OUT)

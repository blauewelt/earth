import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

BG, INK, MUT = "#fcfcfb", "#1a1a19", "#8a8a86"
C1, C2 = "#2a78d6", "#eb6834"

rows = [
 ("E-044b · #427/#432",  "stride 1 · K=24 · span 120 d · predict +5 d",   1, 24, 5,  "0.506 / 0.504", INK, "", False),
 ("E-045-A9 · #446 · input-quant", "stride 1 · K=24 · 120 d · +5 d · 8-level lattice", 1, 24, 5, "0.492", INK, "quant helps at 5 d…", False),
 ("E-045-A8 · #452",     "stride 2 · K=24 · span 240 d · +10 d",          2, 24, 10, "0.338", INK, "", False),
 ("E-045-A7 · #441", "stride 3 · K=24 · span 360 d · +15 d", 3, 24, 15, "0.162", INK, "…not at 15 d (A10 #458 +quant: 0.169)", False),
 ("E-045-A2a/b · #435/#439", "stride 6 · K=24 · span 720 d · +30 d — reaches the year-back frame", 6, 24, 30, "0.071 / 0.073", INK, "", False),
 ("E-045-A11 · #464",    "stride 6 · K=4 · span 120 d · +30 d — year-back frame cut off", 6, 4, 30, "0.527", C2, "jumped to the 5-day class", False),
 ("E-045.2 · #462", "stride 2 · K=72 · span 720 d · +10 d — year-back frame back in reach", 2, 72, 10, "0.080", INK, "", False),
 ("E-045.3 · #476/#484", "stride 3 · K=48 · span 720 d · +15 d",          3, 48, 15, "BLOCKED", MUT, "strided stage-2 falls to CPU pace, 2 of 2 copies — parked pending the strided-Z fix", False),
 ("E-045.1 · #478 · DECISIVE", "stride 1 · K=144 · span 720 d · +5 d — full two years at 5-day steps", 1, 144, 5, "0.082", C2, "pre-registered ~0.07–0.15 vs ~0.5 — lands in the context band", True),
]

H = 1.32
fig, ax = plt.subplots(figsize=(12.2, 2.4 + 0.86*len(rows)), dpi=130)
fig.patch.set_facecolor(BG); ax.set_facecolor(BG)
ax.set_xlim(-770, 620); ax.set_ylim(-H*len(rows)-1.5, 1.75); ax.axis("off")

ax.text(-770, 1.55, "THE FACTORIAL, closed at the decisive rung (23:33Z 08-25) — context span is the story at every step size",
        fontsize=15.5, color=INK, fontweight="bold", va="top")
ax.text(-770, 1.12, "blue tick = one z snapshot fed to the head · orange = predicted frame · ratio: lower is better · t−1 year outlined where the context reaches it",
        fontsize=11.3, color=MUT, va="top")

for i,(lab,sub,stride,K,step,res,rcol,note,new) in enumerate(rows):
    ytop = -i*H - 0.30            # label line
    ystrip = -i*H - 0.92          # tick strip
    ax.text(-770, ytop, lab, fontsize=12.3, color=(C2 if new else INK), fontweight="bold", va="center")
    ax.text(-140, ytop, sub, fontsize=10.2, color=MUT, va="center")
    span = stride*5*K
    xs = [-(span) + stride*5*j for j in range(K)]
    tickw = 3.0 if K<=24 else (2.0 if K<=72 else 1.4)
    for x in xs:
        ax.add_patch(Rectangle((x-tickw/2, ystrip-0.15), tickw, 0.30, color=C1, lw=0, alpha=0.95 if K<=72 else 0.75))
    ax.add_patch(Rectangle((step-2.5-2.2, ystrip-0.15), 4.4, 0.30, color=C2, lw=0))
    if span >= 365:
        ax.add_patch(Rectangle((-365-3.0, ystrip-0.23), 6.0, 0.46, fill=False, ec=C2, lw=1.5))
        if i == 4:
            ax.text(-365, ystrip-0.30, "t − 1 year", fontsize=9.5, color=C2, ha="center", va="top")
    ax.plot([-770, 460], [ystrip-0.44, ystrip-0.44], color="#e4e4e0", lw=0.9)
    ax.text(470, ystrip+0.09, res, fontsize=15 if rcol!=MUT else 11.5, color=rcol,
            fontweight="bold" if rcol!=MUT else "normal", va="center")
    if note:
        ax.text(470, ystrip-0.22, note, fontsize=9.6, color=MUT, va="center")

yf = -H*len(rows)-0.75
ax.text(-770, yf+0.30,
 "Same step, only the span moved:  5 d steps 0.506 → 0.082 (E-044b → #478, 6.2×) · 10 d 0.338 → 0.080 (A8 → #462) · 30 d 0.527 → 0.071 (A11 → A2a).",
 fontsize=11.2, color=INK, va="top")
ax.text(-770, yf-0.06,
 "Same span, only the step moved:  at 120 d, 0.506 ≈ 0.527 · at 720 d, 0.071 ≈ 0.080 ≈ 0.082 — a flat line against the K-fixed cliff. Mechanism on\n"
 "record: the seasonal analog — a 720 d window contains last year's same-calendar frame (outlined); a 120 d window cannot.",
 fontsize=11.2, color=MUT, va="top")
plt.tight_layout(pad=0.6)
import os
out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fig_factorial2.png")
plt.savefig(out, facecolor=BG, bbox_inches="tight")
print("saved")

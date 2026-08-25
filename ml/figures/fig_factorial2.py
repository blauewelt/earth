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
 ("E-045.2 · #462 · NEW TONIGHT", "stride 2 · K=72 · span 720 d · +10 d — year-back frame back in reach", 2, 72, 10, "0.080", C2, "10-day steps drop 0.338 → 0.080", True),
 ("E-045.3 · #467",      "stride 3 · K=48 · span 720 d · +15 d",          3, 48, 15, "queued", MUT, "3rd in the H100 line (first try OOMed a 24 GB card)", False),
 ("E-045.1 · #463 · THE DECISIVE RUNG", "stride 1 · K=144 · span 720 d · +5 d — full two years at 5-day steps", 1, 144, 5, "RUNNING — embed 70%", MUT, "head trains from ~08Z · pre-registered: ~0.07–0.15 vs ~0.5", False),
]

H = 1.32
fig, ax = plt.subplots(figsize=(12.2, 2.4 + 0.86*len(rows)), dpi=130)
fig.patch.set_facecolor(BG); ax.set_facecolor(BG)
ax.set_xlim(-770, 620); ax.set_ylim(-H*len(rows)-1.5, 1.75); ax.axis("off")

ax.text(-770, 1.55, "THE FACTORIAL, updated 06:20Z — E-045.2 confirms it: span moves skill, step size barely does",
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
 "Same step, only the span moved:  10 d steps 0.338 → 0.080 (A8 → E-045.2) · 30 d steps 0.527 → 0.071 (A11 → A2a) · 5 d steps 0.506 → #463 (in flight).",
 fontsize=11.2, color=INK, va="top")
ax.text(-770, yf-0.06,
 "Same span, only the step moved:  at 120 d, 0.506 ≈ 0.527 · at 720 d, 0.071 ≈ 0.080. Mechanism on record: the seasonal analog — a 720 d window\n"
 "contains last year's same-calendar frame (outlined); a 120 d window cannot. E-045.1 is on the H100 now; E-045.3 follows in its queue.",
 fontsize=11.2, color=MUT, va="top")
plt.tight_layout(pad=0.6)
plt.savefig("/home/claude/fig_factorial2.png", facecolor=BG, bbox_inches="tight")
print("saved")

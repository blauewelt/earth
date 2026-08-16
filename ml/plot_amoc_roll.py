#!/usr/bin/env python3
"""AMOC curves for the STENCIL heads: hindcast over the held-out years, and
the unforced roll past the record — one self-contained HTML page.

Chris, 2026-08-16: *"Given our new best model: Can you rerender the AMOC
prediction curves for the held out years and prediction into the future?
Both with 89 and 144 inputs."*

Why this script exists beside `plot_projection.py`. That one renders E-021's
ENSEMBLE fan and reads `project_amoc.json`, which is produced by
`project_amoc.py` — and `project_amoc.py` refuses a stencil head, correctly:
it rolls the section's 265 pixels only, which is exact when a head attends
over time per pixel with no cross-pixel coupling, and simply wrong once the
head reads 89 neighbours per pixel. The stencil heads' rolls therefore come
from `rollout_spatial.py`, which advances all 84,405 window pixels so the
neighbourhood inputs exist at every step, and whose `long` / `future` blocks
already carry exactly the two curves asked for. This script reads THAT file.

Consequences worth stating rather than hiding, both of which change how the
picture should be read against E-021's:

  * These are DETERMINISTIC single trajectories. `rollout_spatial` rolls one
    path per head, so there are no ensemble bands here. The honest reading of
    spread is the SPREAD BETWEEN ARMS AND SEEDS, which is why every head is
    drawn rather than averaged into one line.
  * E-021's negative result stands and is restated on the page: the apparent
    hindcast tracking is substantially replay of the training period. The
    held-out years are the part that is not, which is why they are shaded and
    scored separately — that split is the whole point of the first panel.

House style, colours, smoothing filter, crosshair and CSS are imported from
`plot_projection.py` rather than re-implemented: two copies of a plotting
rule is two chances to disagree, the same defect that put a cosine on the
status page's continuation charts.

Usage:
  python3 ml/plot_amoc_roll.py --json rollout_spatial.json [more.json ...] \\
      --truth ml/cache/truth_rapid.json --out amoc_roll.html
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from plot_projection import panel, smooth                       # noqa: E402

HOLD = ("2009", "2017", "2023")     # the years no codec or head ever trained on

# Arm labels: a head's tag prefix says what it was, and the reader should not
# have to decode "s90rspiral:111-4444-0.71-0.5_s0". Keyed by the stencil slot
# count, which is what the evaluator records and what actually differs.
ARMS = {
    1:   ("gate", "s3"),
    90:  ("89 in", "s1"),
    145: ("144 in", "s2"),
}
# Long labels are direct-labelled at the line's end and ran off the right
# edge in the first render ("no neighbours (gate) · se…"), so the legend
# names are short and the caption carries the full description.
ARM_LONG = {1: "no neighbours (the gate head)",
            90: "89 sunflower inputs (xl89)",
            145: "144 sunflower inputs (xl144)"}


def arm_of(head_key, meta):
    """(label, colour role, slots) for a head, from its own recorded stencil."""
    slots = int(meta.get("stencil", 1) or 1)
    lab, role = ARMS.get(slots, (f"{slots - 1} inputs", "s1"))
    seed = meta.get("seed", 0)
    return f"{lab} s{seed}", role, slots


def pearson(xs, ys):
    pr = [(x, y) for x, y in zip(xs, ys) if x is not None and y is not None]
    if len(pr) < 8:
        return float("nan"), len(pr)
    n = len(pr)
    mx = sum(x for x, _ in pr) / n
    my = sum(y for _, y in pr) / n
    sxy = sum((x - mx) * (y - my) for x, y in pr)
    sdx = sum((x - mx) ** 2 for x, _ in pr) ** 0.5
    sdy = sum((y - my) ** 2 for _, y in pr) ** 0.5
    return (sxy / (sdx * sdy) if sdx and sdy else float("nan")), n


def load_heads(paths):
    """Every head from every rollout_spatial.json given, tagged by arm.

    DEDUPLICATED BY CHECKPOINT. Every evaluation re-rolls the frozen gate head
    (that is the point of the gate), so handing this script two runs draws the
    gate twice — two identical lines, two identical legend entries, two
    identical table rows, and a reader who counts arms gets the wrong number.
    The identity of a head is the checkpoint file it was loaded from, not the
    run it happened to be evaluated in; first occurrence wins and the duplicate
    is recorded so the caller can see what was folded.
    """
    out, seen = [], {}
    for p in paths:
        d = json.load(open(p))
        for key, entry in d.get("heads", {}).items():
            meta = entry.get("meta", {})
            ident = meta.get("file") or key
            if ident in seen:
                seen[ident]["also_in"].append(os.path.basename(p))
                continue
            lab, role, slots = arm_of(key, meta)
            h = {"key": key, "label": lab, "role": role, "slots": slots,
                 "seed": meta.get("seed", 0),
                 "long": entry.get("long"), "future": entry.get("future"),
                 "src": os.path.basename(p), "ident": ident, "also_in": []}
            seen[ident] = h
            out.append(h)
    for h in out:
        if h["also_in"]:
            print("  folded duplicate %s (also in %s)"
                  % (h["ident"], ", ".join(h["also_in"])), file=sys.stderr)
    # gate first (it is the reference), then by slot count, then seed
    out.sort(key=lambda h: (h["slots"] != 1, h["slots"], h["seed"]))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", nargs="+", required=True,
                    help="one or more rollout_spatial.json files")
    ap.add_argument("--truth", required=True,
                    help="RAPID truth json: {ym: [...], sv_des: [...]}")
    ap.add_argument("--out", required=True)
    ap.add_argument("--smooth", type=int, default=18)
    ap.add_argument("--drop-gate", action="store_true",
                    help="omit the no-neighbour gate head from the panels")
    a = ap.parse_args()

    heads = load_heads(a.json)
    if a.drop_gate:
        heads = [h for h in heads if h["slots"] != 1]
    if not heads:
        sys.exit("no heads found in the given rollout_spatial.json files")

    t = json.load(open(a.truth))
    tmap = dict(zip(t["ym"], t["sv_des"]))

    # ---------------- panel 1: the hindcast over the held-out years ------
    hs = [h for h in heads if h.get("long")]
    if not hs:
        sys.exit("no `long` block in any input — the eval was run with "
                 "--long-months 0, or died before the hindcast")
    hym = hs[0]["long"]["roll_ym"]
    for h in hs:
        if h["long"]["roll_ym"] != hym:
            sys.exit(f"{h['key']} rolls a different month axis — refusing to "
                     f"draw two heads on one axis that do not share it")
    obs = [tmap.get(ym) for ym in hym]
    ho_i = [i for i, m in enumerate(hym) if m[:4] in HOLD]
    tr_i = [i for i, m in enumerate(hym) if m[:4] not in HOLD]

    series, rows = [], []
    for h in hs:
        sv = h["long"]["sv_des"]
        r_ho, n_ho = pearson([sv[i] for i in ho_i], [obs[i] for i in ho_i])
        r_tr, _ = pearson([sv[i] for i in tr_i], [obs[i] for i in tr_i])
        sm_m, sm_o = smooth(sv, a.smooth), smooth(obs, a.smooth)
        r_lp, _ = pearson(sm_m, sm_o)
        pr = [(x, y) for x, y in zip(sm_m, sm_o) if x is not None and y is not None]
        amp = (float("nan") if len(pr) < 24 else
               (sum((x - sum(p[0] for p in pr) / len(pr)) ** 2 for x, _ in pr)
                / sum((y - sum(p[1] for p in pr) / len(pr)) ** 2 for _, y in pr)) ** 0.5)
        series.append((h["label"], sv, h["role"], 0, "raw"))
        series.append((h["label"], smooth(sv, a.smooth), h["role"], 0, "smooth"))
        rows.append((h["label"], r_tr, r_ho, n_ho, r_lp, amp))
    series.append(("", obs, "s4", 0, "raw"))
    series.append((f"observed · {a.smooth}-mo", smooth(obs, a.smooth), "s4", 0, "smooth"))

    allv = [v for v in obs if v is not None]
    for h in hs:
        allv += h["long"]["sv_des"]
    ylo, yhi = min(allv) - 0.6, max(allv) + 0.6
    xl = [(i, ym[:4]) for i, ym in enumerate(hym)
          if ym.endswith("-01") and int(ym[:4]) % 2 == 0]
    spans = []
    for y in HOLD:
        idx = [i for i, m in enumerate(hym) if m[:4] == y]
        if idx:
            spans.append((idx[0], idx[-1] + 1, y))

    best_ho = max((r[2] for r in rows if r[2] == r[2]), default=float("nan"))
    p1 = panel(
        "Hindcast — rolled from a 2004 context across the held-out years",
        f"context ends {hs[0]['long']['context_end']} · one deterministic "
        f"trajectory per head, no data enters after the context · shaded: the "
        f"three years no model in this project ever trained on · "
        f"deseasonalised anomaly",
        series, [], xl, ylo, yhi, len(hym), spans=spans,
        notes=[
            "<strong>Read the shaded years, not the rest.</strong> Outside "
            "them the roll is replaying months its codec saw during "
            "self-supervised pretraining, which is why the correlation there "
            "is high and means little — E-021 established this with a control "
            "(the future roll correlates only +0.05 with the same period). "
            "The held-out columns are the honest test, and there the best arm "
            f"reads r = {best_ho:+.2f}.",
            f"<strong>The heavy lines are {a.smooth}-month running means</strong> "
            "— the filter the AMOC-reconstruction literature reports and the "
            "one behind <code>r_lowpass18</code> in our experiment log, so "
            "these curves mean what that number means. The pale lines behind "
            "them are the monthly series.",
            "There are no uncertainty bands here because these rolls are "
            "deterministic: one path per head. The spread between the drawn "
            "arms and seeds is the only honest spread on this page, and it is "
            "much narrower than the real AMOC's variability — a shrunk, "
            "over-smoothed replay, exactly as the amplitude column below says.",
        ], pid="p1")

    tbl = "".join(
        f"<tr><td>{lab}</td><td>{rt:+.2f}</td><td><strong>{rh:+.2f}</strong></td>"
        f"<td>{nh}</td><td>{rl:+.2f}</td><td>{am:.2f}×</td></tr>"
        for lab, rt, rh, nh, rl, am in rows)

    # ---------------- panel 2: the unforced roll past the record --------
    fs = [h for h in heads if h.get("future")]
    p2 = ""
    if fs:
        fym = fs[0]["future"]["roll_ym"]
        hist_n = 20 * 12
        hist_ym = [ym for ym in t["ym"]][-hist_n:]
        hist_sv = [tmap[ym] for ym in hist_ym]
        nh_ = len(hist_ym)
        s2, allv2 = [], list(hist_sv)
        for h in fs:
            sv = h["future"]["sv_des"]
            allv2 += sv
            s2.append((h["label"], [None] * nh_ + sv, h["role"], 0, "raw"))
            s2.append((h["label"], [None] * nh_ + smooth(sv, a.smooth),
                       h["role"], 0, "smooth"))
        s2.append(("", hist_sv + [None] * len(fym), "s4", 0, "raw"))
        s2.append((f"observed · {a.smooth}-mo",
                   smooth(hist_sv, a.smooth) + [None] * len(fym), "s4", 0, "smooth"))
        ylo2, yhi2 = min(allv2) - 0.6, max(allv2) + 0.6
        xl2 = [(i, ym[:4]) for i, ym in enumerate(hist_ym + fym)
               if ym.endswith("-01") and int(ym[:4]) % 5 == 0]
        obs_sorted = sorted(t["sv_des"])
        o5 = obs_sorted[int(0.05 * len(obs_sorted))]
        o95 = obs_sorted[int(0.95 * len(obs_sorted))]
        drift = {h["label"]: (sum(h["future"]["sv_des"][-120:]) / 120)
                 for h in fs}
        drift_s = " · ".join(f"{k.split(' · ')[0]} {v:+.2f}" for k, v in
                             list(drift.items())[:3])
        p2 = panel(
            "Unforced roll — 20 years past the end of the record",
            f"context ends {fs[0]['future']['context_end']} · no emissions "
            f"pathway, no wind or heat-flux forcing arrives after that month · "
            f"mean anomaly over the final decade: {drift_s} Sv",
            s2, [], xl2, ylo2, yhi2, nh_ + len(fym), seam=nh_,
            ref=(o5, o95, "observed 90% range"),
            notes=[
                "<strong>This is not a forecast, and its flatness is the "
                "reason.</strong> Nothing forces the model after the seam, so "
                "what is drawn is what the learned dynamics does when left "
                "alone: it relaxes toward climatology. The grey band is the "
                "range the observed record actually occupies — the projected "
                "curves sit well inside it and stop moving, which is the "
                "signature of a damped attractor, not of a quiet ocean.",
                "The useful content here is the comparison between arms, not "
                "the level of any single line: two input widths and two seeds "
                "each, rolled identically, show how much of the trajectory is "
                "the model and how much is the seed.",
            ], pid="p2")

    from plot_projection import main as _  # noqa: F401  (import-time check)
    import plot_projection as pp
    src = open(pp.__file__).read()
    css = src.split('    css = """', 1)[1].split('"""', 1)[0]
    js = src.split('    js = """', 1)[1].split('"""', 1)[0]
    # two extra roles this page needs: a third arm colour and the observed
    # series. Kept next to the imported palette so the pair stays validated.
    #
    # s4 is the OBSERVED record, and it is deliberately ink rather than a hue.
    # It shipped as #eb6834 — byte-identical to s2, the third arm colour — so
    # the page drew three orange lines and reality was one of them (Chris,
    # 2026-08-16: "can you render the reality in a different color"). Ink is
    # the right answer rather than a fourth hue: the arms are members of one
    # comparison and should share a hue family, while the measurement is not
    # an arm at all, and every other page in this project already draws the
    # observations heavier than the model lines. It also matches
    # make_figs.py's fig_amoc_roll, so the printed and interactive versions
    # of this figure cannot disagree about which line is the ocean.
    # The gate is DASHED, not merely grey. Once the observed series stopped
    # being orange, two neutral lines shared the page — the gate's smooth
    # curve and the observed monthly series behind it — and hue alone no
    # longer told them apart. A dash is carried by the line itself, so it
    # survives greyscale printing and the dark-mode palette swap.
    #
    # Note the stroke-width lives on `.ln.smooth.s4`, never on `.s4`: the
    # direct labels are <text class="dlab s4">, the base sheet neutralises
    # them with `.dlab{stroke:none}`, and a bare `.s4{stroke-width:…}` has
    # equal specificity and comes later — which drew every legend label with
    # a 2.6px outline and made "observed · 18-mo" look struck through.
    css += """
.s3{stroke:#6f6e66}.s4{stroke:#1a1a19}
.ln.smooth.s3{stroke-dasharray:6 4}
.ln.smooth.s4{stroke-width:3.2}
.dlab.s3{fill:#6f6e66}.dlab.s4{fill:#1a1a19;stroke:none}
@media (prefers-color-scheme:dark){
 .s3{stroke:#a5a396}.s4{stroke:#e8e6df}
 .dlab.s3{fill:#a5a396}.dlab.s4{fill:#e8e6df}
}
"""
    html = f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>AMOC — hindcast and 20-year roll, stencil heads</title>
<style>{css}</style></head>
<body>
<h1>AMOC at 26.5°N — the current best models, rolled</h1>
<p class="lede">Both panels come from <code>ml/rollout_spatial.py</code>, which
advances every one of the 84,405 window pixels a month at a time — the only
evaluator that can feed these heads their neighbourhood inputs, and therefore
the only one whose AMOC curve is defined for them at all. Each line is one
deterministic trajectory from one head; the arms differ in how many neighbouring
pixels the model reads (89 or 144, laid out as a golden-angle sunflower reaching
111–4444&nbsp;km) and in training seed. Values are deseasonalised anomalies in
sverdrups.</p>
{p1}
<table><tr><th>arm</th><th>r, trained years</th><th>r, HELD-OUT years</th>
<th>n</th><th>r at {a.smooth}-mo</th><th>amplitude</th></tr>{tbl}</table>
<p class="note">The third column is the one to read: the other correlations
include years the representation saw during pretraining. Amplitude is the
model's low-frequency standard deviation over the observations' — below 1.0
means the swings are rendered smaller than they were.</p>
{p2}
<script>{js}</script>
</body></html>"""
    with open(a.out, "w") as f:
        f.write(html)
    print(f"wrote {a.out} ({os.path.getsize(a.out):,} bytes) — "
          f"{len(hs)} hindcast heads, {len(fs)} future heads")


if __name__ == "__main__":
    main()

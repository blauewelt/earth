#!/usr/bin/env python3
"""Render E-021's two AMOC figures as ONE self-contained HTML page.

Chris: "plot the current for the next 20 years as well as predicted vs
measured for the past 20 years."

Two panels, in that order:

  1. THE FAN — the pooled ensemble rolled 240 months past the record.
     Median line, 50% and 90% bands, and RAPID's observed record drawn to
     the left of the seam so the projection is read against the thing it
     continues.
  2. HINDCAST — the same machinery started from a 2004-12 context, rolled
     over 2005-2024, against RAPID truth on the same axis. This is the
     calibration panel: a fan that does not cover reality here says the
     future fan's width is decorative.

Both are in DESEASONALISED Sv (the anomaly the probe is fit on) with the
train-month RAPID climatology available for re-seasonalising; the
deseasonalised space is what every number in EXPERIMENTS.md is quoted in,
so the plots do not silently change units on the reader.

No plotting library: SVG built here, inlined, with a vanilla crosshair.
That makes the file openable on a phone with no network and no build.

Colors are the validated data-viz defaults — blue #2a78d6 (model) and
orange #eb6834 (observed) in light, #3987e5 / #d95926 in dark; the pair
clears every gate in both modes (CVD ΔE 24.7 / 26.8, normal 33.6 / 31.8,
contrast ≥ 3:1). Two series, so the legend is always present and both are
direct-labelled.

Usage:
  python3 ml/plot_projection.py --json project_amoc.json --out fan.html
"""
import argparse
import json
import os

W, H = 860, 330          # plot box per panel
PAD = dict(l=58, r=132, t=34, b=42)


def _scales(xs_n, ylo, yhi):
    iw = W - PAD["l"] - PAD["r"]
    ih = H - PAD["t"] - PAD["b"]
    def sx(i):
        return PAD["l"] + (iw * i / max(xs_n - 1, 1))
    def sy(v):
        return PAD["t"] + ih * (1 - (v - ylo) / (yhi - ylo))
    return sx, sy


def _band(xs, lo, hi, sx, sy):
    up = " ".join(f"{sx(i):.1f},{sy(v):.1f}" for i, v in enumerate(lo))
    dn = " ".join(f"{sx(i):.1f},{sy(v):.1f}"
                  for i, v in reversed(list(enumerate(hi))))
    return f"{up} {dn}"


def _line(vals, sx, sy, x0=0):
    return " ".join(f"{sx(x0 + i):.1f},{sy(v):.1f}"
                    for i, v in enumerate(vals) if v is not None)


def smooth(vals, k=18, min_valid=12):
    """Centred k-month running mean — the SAME filter `lowpass_r` in
    probe_kfold.py applies (k=18 after Frajka-Williams 2015 /
    Sanchez-Franks 2021, the convention the AMOC-reconstruction literature
    reports). Using the house filter matters: the curve drawn here then
    means exactly what `r_lowpass18` in the experiment log means, instead
    of being a prettier line with its own private definition.

    Windows with fewer than `min_valid` finite months yield None, so gaps
    stay gaps rather than being interpolated across."""
    n = len(vals)
    half = k // 2
    out = [None] * n
    for i in range(n):
        lo, hi = max(0, i - half), min(n, i + half)
        w = [v for v in vals[lo:hi] if v is not None]
        if len(w) >= min_valid:
            out[i] = sum(w) / len(w)
    return out


def panel(title, subtitle, series, bands, xlabels, ylo, yhi, n, seam=None,
          notes=(), pid="p", spans=(), ref=None):
    """series: [(label, values, color_role, x0)] · bands: [(lo, hi, opacity)]
    spans: [(i0, i1, label)] shaded vertical regions (held-out years)
    ref:   (lo, hi, label) a horizontal reference band (observed spread)"""
    sx, sy = _scales(n, ylo, yhi)
    out = [f'<figure class="panel"><figcaption><h2>{title}</h2>'
           f'<p class="sub">{subtitle}</p></figcaption>',
           f'<svg viewBox="0 0 {W} {H}" role="img" '
           f'aria-label="{title}. {subtitle}" preserveAspectRatio="xMidYMid meet">']
    # observed-spread reference FIRST, so it sits behind everything: the
    # single most important comparison on the fan panel is "how wide is the
    # model's band against how much the real thing actually moves".
    if ref is not None:
        rlo, rhi, rlab = ref
        out.append(f'<rect class="ref" x="{PAD["l"]}" y="{sy(rhi):.1f}" '
                   f'width="{W-PAD["l"]-PAD["r"]}" height="{sy(rlo)-sy(rhi):.1f}"/>')
        out.append(f'<text class="reflab" x="{W-PAD["r"]-6}" '
                   f'y="{sy(rhi)+13:.1f}">{rlab}</text>')
    for i0, i1, slab in spans:
        out.append(f'<rect class="span" x="{sx(i0):.1f}" y="{PAD["t"]}" '
                   f'width="{max(sx(i1)-sx(i0),1):.1f}" '
                   f'height="{H-PAD["t"]-PAD["b"]}"/>')
        out.append(f'<text class="spanlab" x="{(sx(i0)+sx(i1))/2:.1f}" '
                   f'y="{PAD["t"]-5}">{slab}</text>')
    # y grid — recessive
    step = 1 if (yhi - ylo) <= 8 else 2
    v = int(ylo) - (int(ylo) % step)
    while v <= yhi:
        if v >= ylo:
            y = sy(v)
            out.append(f'<line class="grid" x1="{PAD["l"]}" x2="{W-PAD["r"]}" '
                       f'y1="{y:.1f}" y2="{y:.1f}"/>')
            out.append(f'<text class="tick ty" x="{PAD["l"]-8}" y="{y+4:.1f}">'
                       f'{v:+g}</text>')
        v += step
    # zero line, emphasised (it is the climatological mean)
    if ylo < 0 < yhi:
        out.append(f'<line class="zero" x1="{PAD["l"]}" x2="{W-PAD["r"]}" '
                   f'y1="{sy(0):.1f}" y2="{sy(0):.1f}"/>')
    for xi, lab in xlabels:
        out.append(f'<text class="tick tx" x="{sx(xi):.1f}" y="{H-PAD["b"]+18}">'
                   f'{lab}</text>')
    if seam is not None:
        out.append(f'<line class="seam" x1="{sx(seam):.1f}" x2="{sx(seam):.1f}" '
                   f'y1="{PAD["t"]}" y2="{H-PAD["b"]}"/>')
        out.append(f'<text class="seamlab" x="{sx(seam)+6:.1f}" '
                   f'y="{PAD["t"]+12}">record ends</text>')
    for lo, hi, op, x0 in bands:
        pts = " ".join(f"{sx(x0+i):.1f},{sy(v):.1f}" for i, v in enumerate(lo))
        pts += " " + " ".join(f"{sx(x0+i):.1f},{sy(v):.1f}"
                              for i, v in reversed(list(enumerate(hi))))
        out.append(f'<polygon class="band" points="{pts}" opacity="{op}"/>')
    used_y = []          # direct labels must not stack on each other
    for spec in series:
        lab, vals, role, x0 = spec[:4]
        cls = spec[4] if len(spec) > 4 else ""       # "" | "raw" | "smooth"
        # A smoothed series is the SAME ENTITY as its raw series, so it
        # keeps the entity's hue and differs by weight — a third hue here
        # would claim a third thing is being measured.
        out.append(f'<polyline class="ln {role} {cls}" '
                   f'points="{_line(vals, sx, sy, x0)}"/>')
        if cls == "raw":                              # labelled by its smooth twin
            continue
        last = [i for i, v in enumerate(vals) if v is not None]
        if last:
            i = last[-1]
            lx, ly = sx(x0 + i), sy(vals[i]) + 4
            # A series that ENDS MID-PLOT (the observed record against a
            # projection that continues past it) must be labelled to its
            # LEFT — labelling right put "observed" on top of the model
            # line in the middle of the fan panel.
            inside = lx < W - PAD["r"] - 40
            if inside:
                # Anchoring to the endpoint still crosses the curve, because
                # a right-anchored label runs back over the series it names.
                # Park it in the clear space at the top of the plot instead.
                ly = PAD["t"] + 16
            while any(abs(ly - u) < 15 for u in used_y):
                ly += 15
            used_y.append(ly)
            anchor = ' style="text-anchor:end"' if inside else ""
            out.append(f'<text class="dlab {role}"{anchor} '
                       f'x="{(lx - 7) if inside else (lx + 7):.1f}" '
                       f'y="{ly:.1f}">{lab}</text>')
    out.append(f'<rect id="{pid}-hit" x="{PAD["l"]}" y="{PAD["t"]}" '
               f'width="{W-PAD["l"]-PAD["r"]}" height="{H-PAD["t"]-PAD["b"]}" '
               f'fill="transparent"/>')
    out.append(f'<line id="{pid}-cross" class="cross" y1="{PAD["t"]}" '
               f'y2="{H-PAD["b"]}" x1="0" x2="0" style="display:none"/>')
    out.append(f'<text class="axlab" x="{PAD["l"]-44}" y="{PAD["t"]-14}">Sv</text>')
    out.append("</svg>")
    out.append(f'<div class="tip" id="{pid}-tip" hidden></div>')
    for nt in notes:
        out.append(f'<p class="note">{nt}</p>')
    out.append("</figure>")
    return "\n".join(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--smooth", type=int, default=18,
                    help="centred running-mean window in months; 18 is the "
                         "house convention (probe_kfold.lowpass_r) so the "
                         "curve means what r_lowpass18 means")
    ap.add_argument("--family", default="sde",
                    choices=("sde", "ic"),
                    help="sde = per-step stochastic forcing (the honest fan); "
                         "ic = perturbed initial condition only")
    a = ap.parse_args()
    d = json.load(open(a.json))
    fam = a.family
    pooled = d["pooled"]
    truth_ym = d["rapid_truth"]["ym"]
    truth_sv = d["rapid_truth"]["sv_des"]

    # ---------------- panel 1: the future fan --------------------------
    fut = pooled["future"][fam]
    fq, fym = fut["q"], fut["roll_ym"]
    # observed record drawn to the LEFT of the seam, on the same axis
    hist_years = 20
    hist = [(ym, sv) for ym, sv in zip(truth_ym, truth_sv)]
    hist = hist[-hist_years * 12:]
    nh = len(hist)
    n1 = nh + len(fym)
    obs_vals = [v for _, v in hist] + [None] * len(fym)
    mod_vals = [None] * nh + fq["p50"]
    lo90 = fq["p5"]
    hi90 = fq["p95"]
    lo50 = fq["p25"]
    hi50 = fq["p75"]
    allv = [v for v in truth_sv] + lo90 + hi90
    ylo, yhi = min(allv) - 0.6, max(allv) + 0.6
    xl1 = []
    for i, ym in enumerate([y for y, _ in hist] + fym):
        if ym.endswith("-01") and int(ym[:4]) % 5 == 0:
            xl1.append((i, ym[:4]))
    import statistics as _st
    obs_p5 = sorted(truth_sv)[int(0.05 * len(truth_sv))]
    obs_p95 = sorted(truth_sv)[int(0.95 * len(truth_sv))]
    band_w = (hi90[0] - lo90[0] + hi90[-1] - lo90[-1]) / 2
    obs_w = obs_p95 - obs_p5
    p1 = panel(
        "AMOC at 26.5°N — rolled 20 years past the record",
        f"pooled ensemble of {fut['n_members']} trajectories "
        f"(6 heads × 12 members, {fam} family) · deseasonalised anomaly, "
        f"context ends {d['months_record_end']} · model 90% band ≈ "
        f"{band_w:.1f} Sv against {obs_w:.1f} Sv of observed spread",
        [("", obs_vals, "s2", 0, "raw"),
         (f"observed · {a.smooth}-mo mean", smooth(obs_vals, a.smooth), "s2", 0, "smooth"),
         ("model median", mod_vals, "s1", 0)],
        [(lo90, hi90, 0.16, nh), (lo50, hi50, 0.28, nh)],
        xl1, ylo, yhi, n1, seam=nh,
        ref=(obs_p5, obs_p95, "observed 90% range"),
        notes=[
            f"The heavy orange line is an {a.smooth}-month running mean of "
            "the observations (the filter the AMOC literature reports); the "
            "pale line behind it is the monthly record it comes from. It is "
            "drawn here so the projection can be judged against the scale of "
            "real low-frequency swings rather than against monthly noise.",
            "<strong>Read the width, not the line.</strong> The blue band is "
            f"the ensemble's own 90% range — about {band_w:.1f} Sv — while the "
            f"observed record spans {obs_w:.1f} Sv over the same kind of "
            "interval (grey). The model is <em>badly under-dispersed</em>: it "
            "relaxes to a near-constant state and its members barely disagree, "
            "so this band is the model's internal spread, <em>not</em> a "
            "credible uncertainty range for the real AMOC.",
            "No emissions pathway enters the model, and no wind or heat-flux "
            "forcing arrives after the context ends — this is what the learned "
            "dynamics does <em>unforced</em>, which is precisely why it decays "
            "toward climatology instead of continuing to vary.",
        ], pid="p1")

    # ---------------- panel 2: hindcast vs measured --------------------
    hk = [k for k in pooled if k != "future"]
    p2 = ""
    if hk:
        key = hk[0]
        hc = pooled[key][fam]
        hq, hym = hc["q"], hc["roll_ym"]
        tmap = dict(zip(truth_ym, truth_sv))
        obs2 = [tmap.get(ym) for ym in hym]
        both = [v for v in obs2 if v is not None]
        allv2 = both + hq["p5"] + hq["p95"]
        ylo2, yhi2 = min(allv2) - 0.6, max(allv2) + 0.6
        xl2 = [(i, ym[:4]) for i, ym in enumerate(hym)
               if ym.endswith("-01") and int(ym[:4]) % 2 == 0]
        n_cov = sum(1 for o, lo, hi in zip(obs2, hq["p5"], hq["p95"])
                    if o is not None and lo <= o <= hi)
        n_obs = sum(1 for o in obs2 if o is not None)
        cov = 100.0 * n_cov / max(n_obs, 1)
        # THE split that decides what this panel means: the three years the
        # codec never trained on, against everything else.
        HOLD = {"2009", "2017", "2023"}
        med = hq["p50"]
        def _r(sel):
            xs = [(med[i], obs2[i]) for i in sel if obs2[i] is not None]
            if len(xs) < 8:
                return float("nan")
            mx = sum(a for a, _ in xs) / len(xs)
            my = sum(b for _, b in xs) / len(xs)
            sxy = sum((a - mx) * (b - my) for a, b in xs)
            sxx = sum((a - mx) ** 2 for a, _ in xs) ** 0.5
            syy = sum((b - my) ** 2 for _, b in xs) ** 0.5
            return sxy / (sxx * syy) if sxx and syy else float("nan")
        tr_i = [i for i, y in enumerate(hym) if y[:4] not in HOLD]
        ho_i = [i for i, y in enumerate(hym) if y[:4] in HOLD]
        r_tr, r_ho = _r(tr_i), _r(ho_i)
        # The smoothed band is where the AMOC's interesting variability
        # lives, so quote agreement AND amplitude there — a high r with a
        # third of the amplitude is a different animal from a good fit.
        sm_o, sm_m = smooth(obs2, a.smooth), smooth(med, a.smooth)
        pr = [(x, y) for x, y in zip(sm_m, sm_o)
              if x is not None and y is not None]
        if len(pr) > 24:
            mx = sum(x for x, _ in pr) / len(pr)
            my = sum(y for _, y in pr) / len(pr)
            sxy = sum((x - mx) * (y - my) for x, y in pr)
            sdx = (sum((x - mx) ** 2 for x, _ in pr) / len(pr)) ** 0.5
            sdy = (sum((y - my) ** 2 for _, y in pr) / len(pr)) ** 0.5
            r_lp = sxy / (len(pr) * sdx * sdy)
            amp_lp = sdx / sdy
        else:
            r_lp = amp_lp = float("nan")
        spans = []
        for y in sorted(HOLD):
            idx = [i for i, m in enumerate(hym) if m[:4] == y]
            if idx:
                spans.append((idx[0], idx[-1] + 1, y))
        p2 = panel(
            "The same roll started from 2004 — and why it is not a forecast",
            f"context ends {key} · {n_obs} months with RAPID truth · "
            f"r = {r_tr:+.2f} on months the model TRAINED on, "
            f"{r_ho:+.2f} on the three held-out years (shaded) · "
            f"at the {a.smooth}-month band r = {r_lp:+.2f} but amplitude "
            f"only {amp_lp:.2f}× · {cov:.0f}% of observations inside the 90% band",
            [("", obs2, "s2", 0, "raw"),
             ("", med, "s1", 0, "raw"),
             (f"observed · {a.smooth}-mo", smooth(obs2, a.smooth), "s2", 0, "smooth"),
             (f"model · {a.smooth}-mo", smooth(med, a.smooth), "s1", 0, "smooth")],
            [(hq["p5"], hq["p95"], 0.16, 0), (hq["p25"], hq["p75"], 0.28, 0)],
            xl2, ylo2, yhi2, len(hym), spans=spans,
            notes=[
                "<strong>The tracking you see is mostly memorisation.</strong> "
                "This roll receives no data after 2004-12, yet it follows the "
                f"record at r = {r_tr:+.2f} — impossible for a genuine 20-year "
                "forecast. The control settles it: the <em>future</em> roll "
                "(2025–2044) correlates only +0.05 with 2005–2024, so the "
                "signal is specific to the initial condition, and all six "
                "independently trained heads reproduce it to ±0.01. A model "
                "trained on these years can replay them from a matching "
                f"starting state. On the three years it never saw, r falls to "
                f"{r_ho:+.2f}.",
                "The second failure is visible without any statistics: the "
                "blue band almost never contains the orange line. A calibrated "
                "90% band would cover ~90% of observations; this one covers "
                f"{cov:.0f}%.",
                f"<strong>The heavy lines are {a.smooth}-month running means</strong> "
                "— the filter the AMOC-reconstruction literature reports, and "
                "the same one behind <code>r_lowpass18</code> in our experiment "
                "log, so the curve means what that number means. They are worth "
                "reading on their own: the smoothed model line is nearly flat "
                f"while the observations swing about {amp_lp and 1/amp_lp:.1f}× "
                "wider — the 2009–10 trough and the 2022–24 low barely register "
                "in it. Even where the model gets the SHAPE of the "
                f"low-frequency wiggle (r = {r_lp:+.2f}), it renders it at "
                f"{amp_lp:.2f}× the real size. Shape without amplitude is the "
                "signature of a shrunk, over-smoothed replay.",
            ], pid="p2")

    css = """
:root{color-scheme:light dark}
*{box-sizing:border-box}
body{margin:0;padding:20px 16px 40px;font:15px/1.55 -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,Helvetica,Arial,sans-serif;
 background:#fcfcfb;color:#0b0b0b}
h1{font-size:20px;margin:0 0 4px}
h2{font-size:16px;margin:0 0 2px;font-weight:600}
.lede{color:#52514e;margin:0 0 22px;max-width:62ch}
.panel{margin:0 0 30px;padding:0}
.sub{color:#52514e;font-size:13px;margin:0 0 6px;max-width:70ch}
svg{width:100%;height:auto;display:block;overflow:visible}
.grid{stroke:#e6e5e1;stroke-width:1}
.zero{stroke:#b9b8b3;stroke-width:1;stroke-dasharray:none}
.seam{stroke:#9a9994;stroke-width:1;stroke-dasharray:4 4}
.seamlab{fill:#52514e;font-size:11px}
.tick{fill:#52514e;font-size:11px}
.ty{text-anchor:end}.tx{text-anchor:middle}
.axlab{fill:#52514e;font-size:11px}
.band{fill:#2a78d6;stroke:none}
.ref{fill:#9a9994;opacity:.13}
.reflab{fill:#52514e;font-size:11px;text-anchor:end}
.span{fill:#9a9994;opacity:.14}
.spanlab{fill:#52514e;font-size:10px;text-anchor:middle}
.ln{fill:none;stroke-width:2;stroke-linejoin:round;stroke-linecap:round}
.ln.raw{stroke-width:1;opacity:.34}
.ln.smooth{stroke-width:2.6}
.s1{stroke:#2a78d6}.s2{stroke:#eb6834}
.dlab{font-size:12px;font-weight:600;stroke:none}
.dlab.s1{fill:#2a78d6}.dlab.s2{fill:#eb6834}
.cross{stroke:#9a9994;stroke-width:1}
.note{color:#52514e;font-size:12.5px;margin:8px 0 0;max-width:74ch}
.tip{position:fixed;pointer-events:none;background:#ffffff;border:1px solid #d8d7d2;
 border-radius:6px;padding:6px 9px;font-size:12px;box-shadow:0 2px 10px rgba(0,0,0,.12);z-index:9}
table{border-collapse:collapse;font-size:13px;margin-top:6px}
th,td{padding:4px 12px 4px 0;text-align:left}
th{color:#52514e;font-weight:600}
details{margin-top:10px}summary{cursor:pointer;color:#52514e;font-size:13px}
@media (prefers-color-scheme:dark){
 body{background:#1a1a19;color:#fff}
 .lede,.sub,.tick,.axlab,.note,.seamlab,summary,th{color:#c3c2b7}
 .grid{stroke:#2e2e2c}.zero{stroke:#4a4a46}.seam{stroke:#66655f}
 .ref,.span{fill:#c3c2b7;opacity:.10}
 .reflab,.spanlab{fill:#c3c2b7}
 .band{fill:#3987e5}
 .s1{stroke:#3987e5}.s2{stroke:#d95926}
 .dlab.s1{fill:#3987e5}.dlab.s2{fill:#d95926}
 .tip{background:#232322;border-color:#3a3a37;color:#fff}
}
"""
    js = """
document.querySelectorAll('figure.panel').forEach(fig=>{
  const svg=fig.querySelector('svg'), tip=fig.querySelector('.tip');
  const hit=fig.querySelector('[id$="-hit"]'), cr=fig.querySelector('[id$="-cross"]');
  if(!hit) return;
  const lines=[...fig.querySelectorAll('polyline.ln')];
  hit.addEventListener('pointermove',e=>{
    const r=svg.getBoundingClientRect();
    const vb=svg.viewBox.baseVal;
    const vx=(e.clientX-r.left)/r.width*vb.width;
    cr.setAttribute('x1',vx);cr.setAttribute('x2',vx);cr.style.display='';
    let rows=[];
    for(const ln of lines){
      const pts=ln.getAttribute('points').trim().split(/\\s+/).map(p=>p.split(',').map(Number));
      let best=null,bd=1e9;
      for(const p of pts){const dd=Math.abs(p[0]-vx); if(dd<bd){bd=dd;best=p;}}
      if(best&&bd<14){
        const lab=fig.querySelector('text.dlab.'+[...ln.classList].find(c=>c[0]==='s'));
        const y=best[1],vb2=svg.viewBox.baseVal;
        rows.push([lab?lab.textContent:'', y]);
      }
    }
    if(!rows.length){tip.hidden=true;return;}
    // invert y -> value using the two extreme grid ticks
    const ticks=[...fig.querySelectorAll('text.ty')].map(t=>[+t.getAttribute('y'),parseFloat(t.textContent)]);
    const inv=y=>{const a=ticks[0],b=ticks[ticks.length-1];
      return (a[1]+(y-a[0])*(b[1]-a[1])/(b[0]-a[0]));};
    tip.innerHTML=rows.map(r=>r[0]+': <strong>'+inv(r[1]).toFixed(2)+' Sv</strong>').join('<br>');
    tip.hidden=false;
    tip.style.left=Math.min(e.clientX+14,innerWidth-160)+'px';
    tip.style.top=(e.clientY+14)+'px';
  });
  hit.addEventListener('pointerleave',()=>{tip.hidden=true;cr.style.display='none';});
});
"""
    dec = pooled["future"][fam].get("decadal_mean", [])
    dec_rows = "".join(
        f"<tr><td>years {i*10+1}–{i*10+10}</td><td>{v:+.2f} Sv</td></tr>"
        for i, v in enumerate(dec))
    html = f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>AMOC — 20-year projection and hindcast</title><style>{css}</style></head>
<body>
<h1>AMOC at 26.5°N — a 20-year roll, and the check that says not to trust it</h1>
<p class="lede">Both panels come from one experiment (E-021): six independently
trained 32M-parameter temporal heads rolled forward on frozen codec embeddings,
each with twelve ensemble members whose spread is set by that head's <em>own
measured one-step error</em> — not by an assumed noise level. Values are
deseasonalised anomalies in sverdrups. <strong>The headline is negative and it
is in the second panel:</strong> the apparent 20-year tracking is largely
memorisation of the training period, and the ensemble is under-dispersed by
roughly an order of magnitude. The first panel is shown so the second can be
read against it — not as a forecast.</p>
{p1}
{p2}
<details><summary>Decadal means of the projected fan</summary>
<table><tr><th>window</th><th>ensemble mean anomaly</th></tr>{dec_rows}</table>
</details>
<script>{js}</script>
</body></html>"""
    with open(a.out, "w") as f:
        f.write(html)
    print(f"wrote {a.out} ({os.path.getsize(a.out):,} bytes)")


if __name__ == "__main__":
    main()

// E-080 deck builder — 22 slides (revision 3.1), house style of E-069.
// Also writes standalone one-slide exports of slides 21 and 22.
const pptxgen = require("pptxgenjs");
const fs = require("fs");

const path = require("path");
const HERE = __dirname;
const PLANS = process.env.E080_PLANS || path.dirname(HERE);
const BUILD = process.env.E080_BUILD || path.join(HERE, "build");
const C = JSON.parse(fs.readFileSync(path.join(BUILD, "content.json"), "utf8"));
const FIG = process.env.E080_FIG_OUT || path.join(PLANS, "E080_figures");
const OUT = path.join(PLANS, "E080_hourglass_cone_deck.pptx");
const OUT_SUMMARY = path.join(PLANS, "E080_hourglass_cone_summary.pptx");
const OUT_TWO_SCALES = path.join(PLANS, "E080_hourglass_two_scales.pptx");

const BG = "0D1117", CARD = "161B22", LINE = "30363D", TXT = "E6EDF3",
      MUT = "7D8590", FOOT = "4A5460", BLUE = "4493F8", GOLD = "E3B341",
      ORANGE = "E8734A", RED = "F85149";
const FS = "Calibri", FH = "Georgia";
const FOOTER = "E-080 · the cut-off mirrored double cone";

// URLs used in the deck body (the spec's own links)
const U = {
  temporal: "https://github.com/blauewelt/earth/blob/main/ml/temporal.py",
  cone: "https://github.com/blauewelt/earth/blob/main/ml/cone.py",
  e071: "https://blauewelt.github.io/earth/docs.html?f=ml/plans/E071_cone_v2.md",
  e069: "https://blauewelt.github.io/earth/docs.html?f=ml/plans/E069_cone_codec.md",
  e076: "https://blauewelt.github.io/earth/docs.html?f=ml/plans/E076_family8_nearest_observations.md",
  codec: "https://github.com/blauewelt/earth/blob/main/ml/cone_codec.py",
  traincone: "https://github.com/blauewelt/earth/blob/main/ml/train_cone.py",
  aniso: "https://github.com/blauewelt/earth/blob/main/ml/measure_flow_anisotropy.py",
  zhu: "https://arxiv.org/abs/2010.04159",
  adam: "https://arxiv.org/abs/1412.6980",
  exps: "https://blauewelt.github.io/earth/docs.html?f=ml/EXPERIMENTS.md#e-069",
  f7: "https://blauewelt.github.io/earth/docs.html?f=ml/plans/E070_family7_build.md",
};

const pres = new pptxgen();
pres.layout = "LAYOUT_WIDE";            // 13.333 x 7.5
pres.author = "Deck builder";
pres.title = "The cut-off mirrored double cone";

let SLIDE_N = 0;

// ---------------------------------------------------------------- rich text
// [label](url) -> hyperlink ; **bold** -> bold accent ; {{x}} -> italic gold
// ~{x} -> subscript ; ^{x} -> superscript ; backticks are dropped
function subsup(str, base, out) {
  const re = /~\{[^}]*\}|\^\{[^}]*\}/g;
  let last = 0, m;
  const push = (t, o) => { if (t) out.push({ text: t.replace(/`/g, ""), options: o }); };
  while ((m = re.exec(str)) !== null) {
    if (m.index > last) push(str.slice(last, m.index), Object.assign({}, base));
    const tok = m[0];
    push(tok.slice(2, -1),
         Object.assign({}, base, tok[0] === "~" ? { subscript: true } : { superscript: true }));
    last = re.lastIndex;
  }
  if (last < str.length) push(str.slice(last), Object.assign({}, base));
}

function rt(str, base) {
  base = base || {};
  const accent = base.accent || BLUE;
  const plain = Object.assign({}, base); delete plain.accent;
  const out = [];
  const re = /\[[^\]]+\]\([^)]+\)|\*\*[^*]+\*\*|\{\{[^}]+\}\}/g;
  let last = 0, m;
  while ((m = re.exec(str)) !== null) {
    if (m.index > last) subsup(str.slice(last, m.index), plain, out);
    const tok = m[0];
    if (tok[0] === "[") {
      const mm = /^\[([^\]]+)\]\(([^)]+)\)$/.exec(tok);
      subsup(mm[1], Object.assign({}, plain, {
        color: BLUE, underline: { style: "sng" }, hyperlink: { url: mm[2] },
      }), out);
    } else if (tok[0] === "*") {
      subsup(tok.slice(2, -2), Object.assign({}, plain, { bold: true, color: accent }), out);
    } else {
      subsup(tok.slice(2, -2), Object.assign({}, plain, { italic: true, color: GOLD }), out);
    }
    last = re.lastIndex;
  }
  if (last < str.length) subsup(str.slice(last), plain, out);
  if (out.length === 0) out.push({ text: str.replace(/`/g, ""), options: plain });
  return out;
}

// stacked paragraphs of different styles inside one text box
function stack(entries) {
  const out = [];
  entries.forEach((e, i) => {
    const rr = rt(e[0], e[1]);
    rr[0].options = Object.assign({}, rr[0].options, { paraSpaceAfter: e[2] === undefined ? 6 : e[2] });
    rr.forEach(r => out.push(r));
    if (i < entries.length - 1) {
      out[out.length - 1].options = Object.assign({}, out[out.length - 1].options, { breakLine: true });
    }
  });
  return out;
}

// NOTE: pptxgenjs starts a NEW PARAGRAPH at every run carrying `bullet`,
// so paragraph props go on the FIRST run of an item only.
function para(items, base, gap, bulleted) {
  const out = [];
  items.forEach((it, i) => {
    const rr = rt(it, base);
    rr[0].options = Object.assign({}, rr[0].options,
      { paraSpaceAfter: gap === undefined ? 7 : gap });
    if (bulleted) rr[0].options.bullet = { indent: 16 };
    rr.forEach(r => out.push(r));
    out[out.length - 1].options = Object.assign({}, out[out.length - 1].options,
      { breakLine: i < items.length - 1 });
  });
  return out;
}

function runs(items, base, gap) { return para(items, base, gap, true); }
function lines(items, base, gap) { return para(items, base, gap === undefined ? 6 : gap, false); }

// ---------------------------------------------------------------- primitives
function card(s, x, y, w, h, opts) {
  opts = opts || {};
  const line = { color: opts.line || LINE, width: opts.lw || 1 };
  if (opts.dash) line.dashType = opts.dash;
  s.addShape(pres.ShapeType.roundRect, {
    x, y, w, h, rectRadius: 0.06,
    fill: { color: opts.fill || CARD },
    line,
  });
}

// one line if it fits at >= 23pt, else two lines
function titleSize(t) {
  const one = Math.floor(1300 / t.length);
  if (one >= 30) return 30;
  if (one >= 23) return one;
  return Math.min(26, Math.floor(2600 / t.length));
}

// the notes pane gets plain text; a markdown link becomes "label — url"
function notesText(n) {
  return C.notes[String(n)].replace(/\[([^\]]+)\]\(([^)]+)\)/g, "$1 — $2");
}

function newSlide(kicker, title, sub) {
  SLIDE_N += 1;
  const s = pres.addSlide();
  s.background = { color: BG };
  s.addText(kicker.toUpperCase(), {
    x: 0.6, y: 0.30, w: 12.13, h: 0.26, fontSize: 11, bold: true, color: BLUE,
    fontFace: FS, charSpacing: 1.3, isTextBox: true, margin: 0, valign: "middle",
  });
  s.addText(title, {
    x: 0.6, y: 0.56, w: 12.13, h: 0.84, fontSize: titleSize(title), bold: true,
    color: TXT, fontFace: FH, isTextBox: true, margin: 0, valign: "middle",
  });
  if (sub) {
    s.addText(rt(sub, { fontSize: 13.5, color: MUT, fontFace: FS }), {
      x: 0.6, y: 1.36, w: 12.13, h: 0.3, isTextBox: true, margin: 0, valign: "middle",
    });
  }
  s.addText(FOOTER, {
    x: 0.6, y: 6.95, w: 9.0, h: 0.3, fontSize: 10, color: FOOT, fontFace: FS,
    isTextBox: true, margin: 0, valign: "middle",
  });
  s.addText(String(SLIDE_N), {
    x: 11.73, y: 6.95, w: 1.0, h: 0.3, fontSize: 10, color: FOOT, fontFace: FS,
    align: "right", isTextBox: true, margin: 0, valign: "middle",
  });
  s.addNotes(notesText(SLIDE_N));
  return s;
}

function bulletCard(s, x, y, w, h, header, items, fontSize, gap) {
  card(s, x, y, w, h);
  let ty = y + 0.22;
  if (header) {
    s.addText(rt(header, { fontSize: 12.5, bold: true, color: BLUE, fontFace: FS }), {
      x: x + 0.28, y: ty, w: w - 0.56, h: 0.28, isTextBox: true, margin: 0, valign: "middle",
    });
    ty += 0.38;
  }
  s.addText(runs(items, { fontSize: fontSize || 11.5, color: TXT, fontFace: FS }, gap), {
    x: x + 0.28, y: ty, w: w - 0.56, h: y + h - ty - 0.18, isTextBox: true,
    margin: 0, valign: header ? "top" : "middle",
  });
}

function labelCard(s, x, y, w, h, label, body, opts) {
  opts = opts || {};
  card(s, x, y, w, h, opts);
  s.addText(stack([
    [label, { fontSize: opts.labelSize || 12.5, bold: true, color: opts.labelColor || BLUE, fontFace: FS }, 8],
    [body, { fontSize: opts.size || 10.5, color: opts.bodyColor || TXT, fontFace: FS }, 0],
  ]), { x: x + 0.26, y: y + 0.16, w: w - 0.52, h: h - 0.32, isTextBox: true, margin: 0, valign: "middle" });
}

function statCard(s, x, y, w, h, big, label, note, colour) {
  card(s, x, y, w, h);
  const entries = [
    [big, { fontSize: 30, bold: true, color: colour || BLUE, fontFace: FH }, 6],
    [label, { fontSize: 12.5, bold: true, color: TXT, fontFace: FS }, note ? 6 : 0],
  ];
  if (note) entries.push([note, { fontSize: 10.5, color: MUT, fontFace: FS }, 0]);
  s.addText(stack(entries), {
    x: x + 0.26, y: y + 0.16, w: w - 0.52, h: h - 0.32, isTextBox: true,
    margin: 0, valign: "middle",
  });
}

function stepCard(s, x, y, w, h, num, head, body) {
  card(s, x, y, w, h);
  s.addShape(pres.ShapeType.ellipse, {
    x: x + 0.24, y: y + (h - 0.34) / 2, w: 0.34, h: 0.34,
    fill: { color: "1F3A5F" }, line: { color: BLUE, width: 1 },
  });
  s.addText(String(num), {
    x: x + 0.24, y: y + (h - 0.34) / 2, w: 0.34, h: 0.34, fontSize: 12, bold: true,
    color: BLUE, fontFace: FS, align: "center", valign: "middle", isTextBox: true, margin: 0,
  });
  const tx = x + 0.72;
  s.addText([
    ...rt(head + "  ", { fontSize: 11.5, bold: true, color: TXT, fontFace: FS }),
    ...rt(body, { fontSize: 10.5, color: MUT, fontFace: FS }),
  ], { x: tx, y: y + 0.12, w: x + w - tx - 0.26, h: h - 0.24, isTextBox: true, margin: 0, valign: "middle" });
}

// ================================================================ slide 1
{
  const s = newSlide("E-080 · design proposal · 15 September 2026",
                     "The cut-off mirrored double cone");
  s.addText("Stage 1 as an hourglass — and a cone the model shapes itself", {
    x: 0.6, y: 1.44, w: 8.0, h: 0.5, fontSize: 17, italic: true, color: MUT,
    fontFace: FH, isTextBox: true, margin: 0, valign: "middle",
  });
  card(s, 0.6, 2.35, 6.8, 1.15);
  s.addText([
    ...rt("E-080 · design proposal · 15 September 2026 · ", { fontSize: 13.5, color: TXT, fontFace: FS }),
    ...rt("nothing here is measured yet", { fontSize: 13.5, bold: true, color: GOLD, fontFace: FS }),
  ], { x: 0.88, y: 2.55, w: 6.24, h: 0.75, isTextBox: true, margin: 0, valign: "middle" });

  s.addText("The deck turns Chris's four points into a specification: a waist instead of a tip, a mirrored cone of targets, a random mix of four tasks, and a cone whose shape the model learns per region — one direction per flow, one scale per channel.", {
    x: 0.6, y: 3.75, w: 6.8, h: 1.2, fontSize: 12.5, color: MUT, fontFace: FS,
    isTextBox: true, margin: 0, valign: "top",
  });

  // hourglass motif
  s.addShape(pres.ShapeType.triangle, {
    x: 9.3, y: 1.95, w: 3.2, h: 1.95, rotate: 180,
    fill: { color: ORANGE, transparency: 82 }, line: { color: ORANGE, width: 1.25 },
  });
  s.addShape(pres.ShapeType.triangle, {
    x: 9.3, y: 4.25, w: 3.2, h: 1.95,
    fill: { color: BLUE, transparency: 80 }, line: { color: BLUE, width: 1.25 },
  });
  s.addShape(pres.ShapeType.roundRect, {
    x: 10.35, y: 4.0, w: 1.1, h: 0.16, rectRadius: 0.07,
    fill: { color: GOLD }, line: { color: GOLD, width: 1 },
  });
  s.addText("future cone · targets only", {
    x: 8.6, y: 1.58, w: 4.6, h: 0.3, fontSize: 11, bold: true, color: ORANGE,
    fontFace: FS, align: "center", isTextBox: true, margin: 0, valign: "middle",
  });
  s.addText("past cone · input", {
    x: 8.6, y: 6.28, w: 4.6, h: 0.3, fontSize: 11, bold: true, color: BLUE,
    fontFace: FS, align: "center", isTextBox: true, margin: 0, valign: "middle",
  });
  s.addText("waist", {
    x: 7.9, y: 3.93, w: 2.3, h: 0.3, fontSize: 11, bold: true, color: GOLD,
    fontFace: FS, align: "right", isTextBox: true, margin: 0, valign: "middle",
  });
}

// ================================================================ slide 2
{
  const s = newSlide("Where we come from", "One cone, five seeds, one clear reading");
  bulletCard(s, 0.6, 1.60, 5.95, 3.05, "E-069 · the cone-native codec", [
    "30 days of dots around a North Atlantic cell, one 3×3 patch at the present, targets at +1 and +2 pentads",
    "History did **not** put the cell's own velocity into the embedding — five seeds, two objectives, refuted",
    "History **did** buy persistent fields (hidden sea-surface height at ~0.36 of the bar vs 0.75–0.90) and a 7–8 % better one- and two-pentad forecast",
  ], 12.5);
  bulletCard(s, 6.78, 1.60, 5.95, 3.05, "E-071 · cone v2", [
    "Global, harmonic climatology, profile tokens — the geometry's known faults, fixed",
    "Reach from the fastest mechanism × 1.5 — a hemisphere by lag 3, so dots go log-radial",
  ], 12.5);
  card(s, 0.6, 4.95, 12.13, 1.3, { fill: "11202F", line: BLUE });
  s.addText([
    ...rt("The open question v2 left: ", { fontSize: 13, bold: true, color: BLUE, fontFace: FS }),
    ...rt("a cone wide enough to never exclude a driver samples the near field at one dot per thousands of kilometres.", { fontSize: 13, color: TXT, fontFace: FS }),
  ], { x: 0.9, y: 5.15, w: 11.53, h: 0.9, isTextBox: true, margin: 0, valign: "middle" });
}

// ================================================================ slide 3
{
  const s = newSlide("The hourglass in one picture", "Past cone → waist → future cone");
  s.addImage({ path: `${FIG}/hourglass_spacetime.png`, x: 0.6, y: 1.62, w: 7.25, h: 4.57 });
  bulletCard(s, 8.15, 1.62, 4.58, 4.57, null, [
    "**Past cone** (lags −6…−1): read as input, dots at the per-lag ellipse plus the anchor column",
    "**Waist** (lag 0): a disc of several cells — the cut-off tip — read as input and reconstructed; the embedding z summarises the waist",
    "**Future cone** (leads +1…+6): never read; sampled the same way and used as targets only",
    "**Point-mirrored:** the future cone at lead +ℓ is the past cone at lag −ℓ reflected through the anchor",
  ], 12);
}

// ================================================================ slide 4
{
  const s = newSlide("Point 1 — the waist: why the tip is cut off",
                     "A gradient does not exist within a single cell");
  bulletCard(s, 0.6, 1.60, 8.1, 4.6, null, [
    "The waist is a disc of radius r~{0} around the anchor, sampled as a small sunflower: ~13 cells at the default (one centre + 12), never fewer than the 3×3 patch's nine",
    "Read as input at lag 0 (values + observed flags) and reconstructed as the anchor family's target",
    "**What the waist buys:** thermal wind is a density gradient; geostrophy is a sea-surface-height slope — both need neighbours at the same instant, not in the past",
    "The waist has the same shape parameters as the cone at lag 0 (orientation, two semi-axes), so a boundary current can have an elongated waist along the flow",
    "The embedding z summarises the waist; one z per anchor, as today",
  ], 12.5);
  statCard(s, 8.9, 1.60, 3.83, 2.2, "≈ 13", "cells at the waist",
           "one centre + 12, never fewer than the nine of the 3×3 patch");
  statCard(s, 8.9, 4.00, 3.83, 2.2, "1 z", "per anchor, as today",
           "so stage 2 and every archived probe see the same kind of object", GOLD);
}

// ================================================================ slide 5
{
  const s = newSlide("Point 2 — the prediction cone", "Sampled like the past cone, read by nobody");
  labelCard(s, 0.60, 1.60, 3.89, 1.25, "+1 … +6 pentads", "30 days ahead, matching the 30 days of history", { labelSize: 15, size: 11 });
  labelCard(s, 4.72, 1.60, 3.89, 1.25, "targets only", "nothing in the future cone is ever read by the encoder", { labelSize: 15, size: 11, labelColor: ORANGE });
  labelCard(s, 8.84, 1.60, 3.89, 1.25, "frozen at evaluation", "the fixed v2 hourglass scores every arm and every seed", { labelSize: 15, size: 11, labelColor: GOLD });
  bulletCard(s, 0.6, 3.10, 12.13, 3.1, null, [
    "Leads +1…+6 pentads (30 days ahead), the mirror image of the past cone through the anchor: centre c(+ℓ) = −c(−ℓ), same axes and orientation at the same |ℓ|",
    "Dots are **targets only**: each is a decoder query (channel, Δx, Δy, lead, depth) → (μ, log σ^{2}), scored with the Gaussian negative log-likelihood already in cone_codec.py",
    "The anchor column at every lead is always among the targets — the fixed-frame forecast stage 2 needs",
    "Per-lead and per-family loss, each reported against its own predict-the-mean bar (the E-069b rule)",
    "Evaluation geometry is **frozen** (the fixed v2 hourglass), whatever the training cone learns — so every number stays comparable across arms and seeds",
  ], 12.5);
}

// ================================================================ slide 6
{
  const s = newSlide("Point 3 — the sampling regime", "One model, four questions, drawn at random per example");
  const hdr = (t) => ({ text: t, options: { bold: true, color: BLUE, fontSize: 11.5, fontFace: FS } });
  const cel = (t, o) => ({ text: t, options: Object.assign({ color: TXT, fontSize: 11, fontFace: FS }, o || {}) });
  const rows = [
    [hdr("task"), hdr("input"), hdr("targets"), hdr("share"), hdr("what it teaches")],
    [cel("T1 forecast", { bold: true }), cel("waist + past cone"), cel("future cone"), cel("0.35", { align: "center", color: GOLD, bold: true }), cel("the full problem")],
    [cel("T2 snapshot forecast", { bold: true }), cel("waist only (past cone fully hidden)"), cel("future cone"), cel("0.15", { align: "center", color: GOLD, bold: true }), cel("the built-in twin: what history buys, inside one model")],
    [cel("T3 nowcast from history", { bold: true }), cel("past cone under a dropout pattern; waist hidden"), cel("waist"), cel("0.25", { align: "center", color: GOLD, bold: true }), cel("the present from the past — persistence, advection, memory")],
    [cel("T4 fill-in", { bold: true }), cel("waist + past cone, both under a dropout pattern"), cel("the hidden waist cells and dots"), cel("0.25", { align: "center", color: GOLD, bold: true }), cel("E-069b's reconstruction family, kept")],
  ];
  s.addTable(rows, {
    x: 0.6, y: 1.60, w: 12.13, colW: [2.15, 3.35, 1.85, 0.75, 4.03],
    rowH: [0.34, 0.46, 0.46, 0.46, 0.46],
    border: { type: "solid", color: LINE, pt: 0.75 },
    fill: { color: CARD }, valign: "middle", margin: [4, 7, 4, 7],
  });
  s.addText([
    ...rt("Dropout patterns for T3 and T4, drawn independently ", { fontSize: 12.5, bold: true, color: TXT, fontFace: FS }),
    ...rt("(each fires with its own probability)", { fontSize: 12.5, color: MUT, fontFace: FS }),
  ], { x: 0.6, y: 4.30, w: 12.13, h: 0.3, isTextBox: true, margin: 0, valign: "middle" });
  const pats = [
    ["channel drop", "at the waist — lag 0 only, chan_drop_scope lag0"],
    ["recent-lag band", "hide every dot at lag ≤ ℓ~{0}, ℓ~{0} ∈ {1, 2, 3}"],
    ["bearing wedge", "hide a 90° sector"],
    ["random dots", "each dot hidden with p = 0.3"],
    ["whole waist", "T3 only"],
  ];
  pats.forEach((p, i) => {
    labelCard(s, 0.6 + i * 2.46, 4.72, 2.31, 1.5, p[0], p[1], { labelSize: 11.5, size: 10.5, bodyColor: MUT });
  });
}

// ================================================================ slide 7
{
  const s = newSlide("Point 3 — the rules that make the mix safe", "Three rules, all learned the hard way in E-069");
  labelCard(s, 0.60, 1.60, 3.89, 2.6, "1 · Knowable targets only",
    "A target channel must be visible somewhere in the input (any lag, any dot). Under the first E-069 plan 80 % of hidden-dot targets broke this and 64 % of the loss was predict-the-mean.",
    { size: 12 });
  labelCard(s, 4.72, 1.60, 3.89, 2.6, "2 · No copy targets",
    "The anchor family scores hidden channels only (anchor_hidden_only); a target visible in the input is a copy, not a question.",
    { size: 12 });
  labelCard(s, 8.84, 1.60, 3.89, 2.6, "3 · Every family against its own bar",
    "Per (task, family, lead) loss, each divided by its predict-the-mean score; a task is “learned” only where the ratio is below one at held-out anchors.",
    { size: 12 });
  bulletCard(s, 0.6, 4.45, 12.13, 1.75, null, [
    "The task is not told to the model by a token; the masks are visible (the mask_tok / miss_tok distinction stays), and the lead is in every query's coordinates — that is enough",
    "Loss weights: equal across tasks per example; the shares in the table set the mix, not the weights",
  ], 12.5);
}

// ================================================================ slide 8
{
  const s = newSlide("Point 4 — the cone as a warped sunflower",
                     "Eight numbers per region shape the whole hourglass — and none of them is in kilometres");
  s.addImage({ path: `${FIG}/warped_sunflower.png`, x: 1.29, y: 1.44, w: 10.75, h: 2.55 });
  card(s, 0.6, 4.07, 12.13, 0.80, { fill: "11202F", line: LINE });
  s.addText("For each region and channel group, the past cone at lag ℓ is a Gaussian footprint centred at anchor + ℓ·d with covariance Σ(ℓ), and the dots are the canonical sunflower mapped through it:", {
    x: 0.9, y: 4.15, w: 11.53, h: 0.24, fontSize: 10.5, color: MUT, fontFace: FS,
    isTextBox: true, margin: 0, valign: "middle",
  });
  s.addText([
    ...rt("Σ(ℓ) = Σ~{0} + ℓ^{2}·Σ~{v}", { fontSize: 15, color: TXT, fontFace: "Cambria" }),
    ...rt("          p(ℓ, i) = anchor + ℓ·", { fontSize: 15, color: TXT, fontFace: "Cambria" }),
    ...rt("d", { fontSize: 15, bold: true, color: GOLD, fontFace: "Cambria" }),
    ...rt(" + Σ(ℓ)^{½} · u~{i}", { fontSize: 15, color: TXT, fontFace: "Cambria" }),
  ], { x: 0.9, y: 4.41, w: 11.53, h: 0.40, isTextBox: true, margin: 0, valign: "middle" });

  const left = [
    "**u~{i}** : the canonical sunflower on the unit disc (Vogel spiral, golden angle — the E-026 pattern in [`ml/temporal.py::spiral_offsets`](" + U.temporal + ")), fixed, 24 points",
    "**Phase-rotated by the golden angle per lag**: bearing~{i,ℓ} = i·137.5° + ℓ·137.5°, so the six slices interleave in space-time instead of repeating the same bearings",
    "**Time itself stays dense**: one slice per pentad — the finest cadence we have, and the 30-day window is the codec's whole job",
    "**d** (2 numbers): the drift per pentad — where the footprint's centre moves for each pentad back in time; **the arrow toward the source**. Stored dimensionless: d = tanh(x) · v_design·Δt",
  ];
  const right = [
    "**Σ~{0}** (3): the waist ellipse, stored as its **matrix logarithm** S~{0} = log Σ~{0} — a symmetric 2×2 in log-km. Its trace is the size; its traceless part is (log aspect ratio) × (cos 2θ, sin 2θ); no 180° wrap, and a circle is S = m·I",
    "**Σ~{v}** (3): the velocity-spread ellipse, also as log Σ~{v} — how fast the footprint widens with lag (standard deviation ∝ ℓ, as the reach ∝ (1 + ℓ) in [`ml/cone.py::reach_km`](" + U.cone + "))",
    "**Why log form**: under Adam every parameter moves about one learning rate per step in its own units. A radius in km moves a fixed number of km (frozen at 850 km, twitchy at 30 km); a log-radius moves a fixed **fraction** — the same at any size",
    "**Future cone:** reflected — centre −ℓ·d, the same Σ(ℓ); the anchor column (0, 0) is always added. Per-channel scale and memory: next slide",
  ];
  s.addText(runs(left, { fontSize: 10, color: TXT, fontFace: FS }, 6),
            { x: 0.6, y: 4.95, w: 5.95, h: 1.85, isTextBox: true, margin: 0, valign: "top" });
  s.addText(runs(right, { fontSize: 10, color: TXT, fontFace: FS }, 6),
            { x: 6.78, y: 4.95, w: 5.95, h: 1.85, isTextBox: true, margin: 0, valign: "top" });
}

// ================================================================ slide 9
{
  const s = newSlide("Point 4 — one direction per flow, one scale per channel",
                     "The shape and the drift belong to the flow; only the size belongs to the channel");
  s.addImage({ path: `${FIG}/channel_apertures_ab.png`, x: 1.17, y: 1.44, w: 11.00, h: 3.65 });
  bulletCard(s, 0.6, 5.17, 12.13, 1.73, null, [
    "**1 · Shared** per region, lag and channel group: the shape Σ(ℓ) and the drift **d** of slide 8. Everything the water carries is carried the same way — the arrow toward the source and the elongation along the jet are properties of the **flow**, not of the tracer. {{One direction per flow.}}",
    "**2 · Per channel**: two log-parameters — a **scale** s~{c} (two for the L-shaped channels: lags 0–1 and 2–6) and a **memory** τ~{c}. In space, S~{c}(ℓ) = S(ℓ) + s~{c}·I, i.e. Σ~{c}(ℓ) = exp(s~{c}) · Σ(ℓ) — the same ellipse inflated or shrunk, **never turned**. In time, lag ℓ is weighted by exp(−ℓΔt/τ~{c}) — how far back this channel is worth reading. {{One scale and one memory per channel; the aperture is a space-time object.}}",
    "**3 · Initialised from the reach-and-memory table** in [`ml/cone.py::FAMILIES`](" + U.cone + "): wind stress 500 km with τ = 10 days (lags 0–1 count; lag 6 is at e^{−3}); currents and sea-surface height 129.6 km·(1 + ℓ), long memory; sea-surface temperature and mixed-layer depth max(that, 500 km) at lags ≤ 1, then the ocean's growth; land 400 km flat, long memory. At step 0 every channel reads the reach and the lags the current geometry gives it",
  ], 10, 5);
}

// ================================================================ slide 10
{
  const s = newSlide("Point 4 — read at zero extra tokens: the per-channel aperture",
                     "One token per dot carries every channel; each channel's entries are weighted by its own aperture before the projection");
  s.addImage({ path: `${FIG}/channel_apertures_c.png`, x: 0.60, y: 1.50, w: 12.13, h: 3.28 });
  bulletCard(s, 0.6, 4.88, 12.13, 2.02, null, [
    "A **per-location token** ([E-071 §6.4](" + U.e071 + ")) carries all channels of a group at one dot position p~{i}: a vector of values and a vector of observed flags, one token per dot — not one token per channel per dot",
    "Each channel gets its own weight at that dot, w~{c,i,ℓ} = exp(−½ · p~{i}ᵀ Σ~{c}(ℓ)⁻¹ p~{i}) · exp(−ℓΔt/τ~{c}) — **a space factor and a time factor** — and it multiplies that channel's value and observed-flag entries before the projection. A dot outside channel c's aperture in space {{or}} beyond its memory in time contributes ≈ 0 for c and still carries the others at full weight",
    "**Worked example**, a dot 400 km from the anchor at lag 1, apertures at their initial values: currents σ = 259 km → w ≈ 0.30; sea-surface temperature σ = 500 km → w ≈ 0.73 (both with a long memory, time factor ≈ 1). For wind stress at lag 4 the time factor alone is e^{−20/10} ≈ 0.14 — the “lags 0–1 only” rule, as a soft weight",
    "**The gradient reaches S, d, s~{c} and τ~{c} through w from every dot at once** — not through four neighbouring cells — which is what lets the geometry learn (next slide)",
    "The fixed safety ring (slide 13) is **not gated**: read at full weight, always. A safety net the model can switch off is not one",
  ], 11, 6);
}

// ================================================================ slide 11
{
  const s = newSlide("Point 4 — how the numbers learn: gate first, harden later",
                     "A soft aperture over fixed candidate dots in phase 1; the dot table re-baked from the learned ellipses in phase 2");
  labelCard(s, 0.60, 1.60, 5.95, 2.45, "Phase 1 — gate",
    "The candidate dots per region and group are **fixed**: cone v2's log-radial set out to the design reach ([E-071 §4.4](" + U.e071 + "), 36 per lag) plus the anchor column. The data loader's reads are regular, cacheable, and never depend on the model's weights. Each candidate token is weighted by the per-channel aperture of slide 10, and the ellipses, drift and offsets learn through those weights — from every candidate at once",
    { size: 11 });
  labelCard(s, 6.78, 1.60, 5.95, 2.45, "Phase 2 — harden",
    "Every N steps (2 k to start) the dense dot table per region is re-baked from the current ellipses — the warped sunflower of slide 8, 24 dots per lag — and the loader swaps tables at the epoch boundary. The fixed ring of 24 stays as the safety net; the apertures stay on (they **are** the per-channel scale). Reads remain regular within an epoch",
    { size: 11, labelColor: GOLD });
  bulletCard(s, 0.6, 4.20, 12.13, 2.45, null, [
    "**Why not read at continuous positions** (the deformable gather of revision 1): it couples the loader's read pattern to the live weights — irregular reads on the sampler's memory-mapped tensor and on the JAX/TPU port of the cone trainer — and its gradient sees only the four cells around each dot. The gate has neither problem, and the per-tensor pyramid of revision 1 is no longer needed",
    "**Learning rate**: the geometry parameters get their own multiplier (×10 the codec's to start — the deformable-attention practice, [Zhu et al. 2021](" + U.zhu + ")), recorded in the run's metrics",
    "Evaluation geometry stays **frozen** (slide 5) whatever phase training is in",
  ], 12);
}

// ================================================================ slide 12
{
  const s = newSlide("Point 4 — where the numbers live",
                     "A 1° map per channel group, interpolated to the anchor");
  bulletCard(s, 0.6, 1.60, 7.55, 4.6, null, [
    "A **1° grid** over the globe (180 × 360 = 64,800 cells); the anchor's parameters are the bilinear blend of the four surrounding map cells — smooth by construction, no seams",
    "**1° because** the currents that carry the overturning are ~100 km wide (the Florida Current sits in an 80 km strait); a 10° cell would average the Gulf Stream with its recirculation and learn a drift near zero",
    "**Hierarchical prior:** each map is a coarse 10° map plus a 1° residual, the residual under L2 shrinkage and a Laplacian smoothness penalty — a data-poor cell (ice edge, coast) inherits its neighbourhood's arrow instead of fitting noise. Each 1° cell still sees 16 anchors × 3,142 pentads ≈ 50 k anchor-pentads for eight numbers",
    "**No Argo ellipse.** The Argo channels are the k nearest profile tokens ([E-071 §3](" + U.e071 + "), [E-076](" + U.e076 + ")) — placed where floats were, each carrying its own offset, age and depth. There is no elliptic shape to learn; the search radius stays as specified there. Depth-dependent sourcing (at the RAPID line the upper limb flows north, the Deep Western Boundary Current south) is left to attention over those tokens, which see depth and offset together",
    "**Optional, phase 2:** a first-harmonic seasonal term on the drift, d(τ) = d~{0} + d~{1} cos τ + d~{2} sin τ — the Somali Current reverses with the monsoon",
  ], 11);
  card(s, 8.35, 1.60, 4.38, 1.95);
  s.addText("one map per channel group", {
    x: 8.63, y: 1.78, w: 3.82, h: 0.28, fontSize: 12.5, bold: true, color: BLUE,
    fontFace: FS, isTextBox: true, margin: 0, valign: "middle",
  });
  s.addText(lines([
    "**ocean surface** — currents, sea-surface height, sea-surface temperature, mixed layer",
    "**atmosphere** — drift only at lags 0–1",
    "**land** — no drift: v = 0, axes and angle only",
  ], { fontSize: 10.5, color: TXT, fontFace: FS }, 6),
    { x: 8.63, y: 2.14, w: 3.82, h: 1.25, isTextBox: true, margin: 0, valign: "top" });
  statCard(s, 8.35, 3.75, 4.38, 2.45, "≈ 1.6 M", "geometry parameters, nominal",
           "64,800 cells × 3 groups × 8 numbers, about half of it live (ocean maps over ocean, land maps over land), plus a scale s_c and a memory τ_c per channel (~100 numbers, global — slide 9) — cheap, and no compute", GOLD);
}

// ================================================================ slide 13
{
  const s = newSlide("Point 4 — constraints, the mirror, and the initial cone",
                     "Bounded by physics, tied to the fixed cone, started isotropic");
  const W4 = 2.867, X4 = [0.60, 3.687, 6.774, 9.861];
  labelCard(s, X4[0], 1.60, W4, 2.35, "Speed cap",
    "|**d**| ≤ v_design · Δt = 5.4 m/s × 5 d = 2,333 km per pentad (E-071's fastest mechanism × 1.5), via a tanh parameterisation — the learned cone lives inside the v2 envelope", { size: 10.5 });
  labelCard(s, X4[1], 1.60, W4, 2.35, "Log-space, with soft floors",
    "Σ~{0}, Σ~{v}, s~{c} and τ~{c} are stored as logarithms (slide 8), so Adam's step is a fixed fraction at any size; the waist's eigenvalues are floored at one cell (28 km) and Σ~{v}'s at the design speed, by a softplus offset in log space rather than a clip", { size: 10.5 });
  labelCard(s, X4[2], 1.60, W4, 2.35, "Prior",
    "Coarse 10° + 1° residual, L2 shrinkage and Laplacian smoothness on the residual (slide 12) — one weight, recorded in the run's metrics", { size: 10.5 });
  labelCard(s, X4[3], 1.60, W4, 2.35, "Mirror tie, by default",
    "The future cone shares the eight numbers (reflected); an untied arm is the ablation, not the design", { size: 10.5 });
  const X = [0.60, 4.72, 8.84], W = 3.89;
  labelCard(s, X[0], 4.20, W, 2.35, "Stop-gradient on the target side",
    "The loss reaches the geometry only through the **input** reads; the positions of targets are treated as constants", { size: 11.5 });
  labelCard(s, X[1], 4.20, W, 2.35, "The v2 safety net stays, ungated",
    "In addition to the learned ellipse, each lag keeps one fixed log-radial ring of 24 dots to the antipode, read at full weight — the cone can be directed without asserting that a driver could not have come from elsewhere", { size: 11.5 });
  labelCard(s, X[2], 4.20, W, 2.35, "Init",
    "**d** = 0; S~{0} = the log of a 3-cell circle with the measured aspect 0.71 ([`ml/measure_flow_anisotropy.py`](" + U.aniso + ")); Σ~{v} from the 0.3 m/s of [`ml/cone.py::FAMILIES`](" + U.cone + "); s~{c} and τ~{c} from the same table per channel — the fixed hourglass exactly, so step 0 of the learned arm is the control", { size: 11, labelColor: GOLD });
}

// ================================================================ slide 14
{
  const s = newSlide("Point 4 — the AMOC example, drawn",
                     "One anchor on the western boundary: a learned surface cone, and deep profiles read from the other side");
  s.addImage({ path: `${FIG}/amoc_cones.png`, x: 0.6, y: 1.60, w: 5.54, h: 4.85 });
  bulletCard(s, 6.37, 1.60, 6.36, 4.85, null, [
    "**Surface group:** the learned d should point south-west (up the Florida Current, toward the Loop Current); the future cone leans north-east along the Gulf Stream",
    "**Deep Argo:** no ellipse to learn. The profile tokens within the search radius carry their own offset and depth; the model should put its attention on the deep (> 1000 dbar) profiles **north** of the anchor — up the Deep Western Boundary Current — and on the shallow ones to the south",
    "Neither is programmed: these are the **pre-registered expectations** — a geometry check for the surface, an attention check for the deep",
    "If the learned surface drift correlates with −u_clim (the climatological surface current, reversed) over the world ocean, the cone learned “upstream” from the data alone",
  ], 12);
}

// ================================================================ slide 15
{
  const s = newSlide("The loophole and its fix", "A learnable cone can learn to ask easy questions");
  card(s, 0.6, 1.60, 12.13, 0.82, { fill: "2A1518", line: RED });
  s.addText([
    ...rt("The loophole:  ", { fontSize: 13, bold: true, color: RED, fontFace: FS }),
    ...rt("if target positions could move, the gradient would move them to smooth water — a lower loss that means nothing.", { fontSize: 13, color: TXT, fontFace: FS }),
  ], { x: 0.9, y: 1.74, w: 11.53, h: 0.55, isTextBox: true, margin: 0, valign: "middle" });
  const W4 = 2.867, X4 = [0.60, 3.687, 6.774, 9.861];
  labelCard(s, X4[0], 2.72, W4, 2.2, "Fix 1 · stop-gradient on target positions",
    "Slide 13. The geometry is trained only by “which inputs help”, never by “which targets are easy”.", { size: 11 });
  labelCard(s, X4[1], 2.72, W4, 2.2, "Fix 2 · the mirror tie",
    "The future cone cannot be tuned separately from the past cone.", { size: 11 });
  labelCard(s, X4[2], 2.72, W4, 2.2, "Fix 3 · frozen evaluation geometry",
    "Held-out numbers are always scored on the fixed hourglass, for every arm.", { size: 11 });
  labelCard(s, X4[3], 2.72, W4, 2.2, "Fix 4 · the aperture never touches the loss weights",
    "In phase 1 the targets are the fixed mirrored candidate set with fixed weights; a model that shrinks its aperture cannot thereby shrink the set of questions it is scored on.", { size: 11 });
  card(s, 0.6, 5.20, 12.13, 1.2, { fill: "11202F", line: GOLD });
  s.addText([
    ...rt("Residual second-order effect:  ", { fontSize: 12.5, bold: true, color: GOLD, fontFace: FS }),
    ...rt("as the input cone moves, the mirrored targets move with it. Monitored, not assumed away — report the target-variance under the learned cone vs the fixed cone at every eval; a drop > 10 % is a flag.", { fontSize: 12.5, color: TXT, fontFace: FS }),
  ], { x: 0.9, y: 5.36, w: 11.53, h: 0.88, isTextBox: true, margin: 0, valign: "middle" });
}

// ================================================================ slide 16
{
  const s = newSlide("What it costs", "Tokens, gathers, parameters — the same regime as v2");
  const hdr = (t, c) => ({ text: t, options: { bold: true, color: c || BLUE, fontSize: 11.5, fontFace: FS } });
  const cel = (t, o) => ({ text: t, options: Object.assign({ color: TXT, fontSize: 11, fontFace: FS }, o || {}) });
  const rows = [
    [hdr(""), hdr("E-069 (measured)", MUT), hdr("this design (budget)")],
    [cel("waist", { bold: true }), cel("42 patch tokens (14 ch × 3×3)"), cel("13 per-location tokens carrying all surface channels")],
    [cel("past cone", { bold: true }), cel("706 dot tokens"), cel("phase 1: 6 lags × (1 + 36 candidates) × 4 groups ≈ 888  ·  phase 2: 6 × (1 + 24 ellipse + 24 ring) × 4 ≈ 1,176")],
    [cel("Argo", { bold: true }), cel("192 (32 × 6, column only)"), cel("26 profile tokens (E-071 §3)")],
    [cel("targets", { bold: true }), cel("42 + 84"), cel("256 dot queries per example, drawn from the future cone and the hidden waist / dots (as n_dot_queries today)")],
    [cel("geometry parameters", { bold: true }), cel("0"), cel("≈ 1.6 M nominal, ~half live (slide 12); no compute")],
    [cel("extra per-tensor artefact", { bold: true }), cel("—"), cel("none (revision 1's pyramid is gone); the per-region dot tables re-baked at epoch boundaries are small")],
  ];
  s.addTable(rows, {
    x: 0.6, y: 1.60, w: 12.13, colW: [2.6, 3.0, 6.53],
    rowH: [0.34, 0.42, 0.58, 0.42, 0.58, 0.42, 0.58],
    border: { type: "solid", color: LINE, pt: 0.75 },
    fill: { color: CARD }, valign: "middle", margin: [4, 7, 4, 7],
  });
  bulletCard(s, 0.6, 5.25, 12.13, 1.4, null, [
    "Perceiver cost is linear in tokens: ~1.2× (phase 1) to ~1.6× (phase 2) E-069's encoder time, inside v2's own 2.1× estimate; the aperture weights are one elementwise multiply per token",
    "Training cost per seed: E-069 seeds ran under one 4090-hour on the North Atlantic; budget 2–3 4090-hours per seed on the global tensor",
  ], 11.5);
}

// ================================================================ slide 17
{
  const s = newSlide("The experiment — E-080, pre-registered",
                     "Three arms, three seeds, four read-outs, one verdict rule");
  const X = [0.60, 4.72, 8.84], W = 3.89;
  labelCard(s, X[0], 1.58, W, 1.95, "A0 · fixed",
    "The hourglass with the v2 geometry (isotropic ellipses, ring, mirror), no learning — the control, and the eval geometry for everyone", { size: 10.5 });
  labelCard(s, X[1], 1.58, W, 1.95, "A1 · learned",
    "The eight numbers per region plus the per-channel scales, gate-then-harden (slide 11), initialised at A0", { size: 10.5 });
  labelCard(s, X[2], 1.58, W, 1.95, "A2 · learned, primed",
    "As A1, but **d** initialised at −u_clim · Δt (the climatological surface current, reversed) for the ocean surface group — tests whether the gradient can find upstream on its own (A1) or only keep it (A2)", { size: 10.5, labelColor: GOLD });
  s.addText([
    ...rt("Held fixed: ", { fontSize: 11.5, bold: true, color: TXT, fontFace: FS }),
    ...rt("same tensor (family 7.2, global), same 7 M codec, 20 k steps, frozen protocol (train ≤ 2020; 2008–09, 2016–17, 2021–24 held out), 3 seeds each.", { fontSize: 11.5, color: MUT, fontFace: FS }),
  ], { x: 0.6, y: 3.62, w: 12.13, h: 0.3, isTextBox: true, margin: 0, valign: "middle" });
  card(s, 0.6, 3.98, 12.13, 1.42);
  s.addText(lines([
    "**R1** T1 future-cone loss per lead vs bar  ·  **R2** the T1 − T2 gap (what history buys)  ·  **R3** drift-vs-upstream correlation, world ocean, |u_clim| > 0.2 m/s",
    "**R4** the western-boundary depth check — surface drift south-west (geometry); attention mass on deep (> 1000 dbar) profile tokens north of the anchor, on shallow ones south (attention diagnostic)",
    "**R5** target-variance flag (slide 15)",
  ], { fontSize: 11, color: TXT, fontFace: FS }, 5),
    { x: 0.9, y: 4.12, w: 11.53, h: 1.14, isTextBox: true, margin: 0, valign: "top" });
  card(s, 0.6, 5.50, 12.13, 1.30, { fill: "11202F", line: GOLD });
  s.addText([
    ...rt("Verdict:  ", { fontSize: 11.5, bold: true, color: GOLD, fontFace: FS }),
    ...rt("A1 adopted if it beats A0 on R1 at every lead ≤ 3, paired at all three seeds, with R5 clean. A2 > A1 with A1 ≈ A0 means the gradient cannot find upstream — then geometry is set from climatology, not learned. Neither → keep A0 and move to a dynamic (per-example) cone.   ", { fontSize: 11.5, color: TXT, fontFace: FS }),
    ...rt("First ablation after the arms: ", { fontSize: 11.5, bold: true, color: GOLD, fontFace: FS }),
    ...rt("A1 without the far ring — does the ring pay for its tokens?", { fontSize: 11.5, color: TXT, fontFace: FS }),
  ], { x: 0.9, y: 5.62, w: 11.53, h: 1.06, isTextBox: true, margin: 0, valign: "middle" });
}

// ================================================================ slide 18
{
  const s = newSlide("Build order", "Five steps, each with its test",
                     "in ml/cone.py · ml/cone_sampler.py · ml/cone_codec.py · ml/train_cone.py");
  const steps = [
    ["Waist + future cone in the geometry (cone.py)", "waist_dots, future_dots = reflected inner_dots, the per-lag golden-angle phase in inner_dots; coverage_report extended with the mirror identity (future set = −past set) and re-asserted per lag under the phase — pure numpy, CPU test"],
    ["The task mix (cone_codec.py::default_plan → plan_v3)", "T1–T4 with shares and the five dropout patterns; test: every target's channel is visible somewhere in the input, for 10 k drawn examples, zero violations"],
    ["The aperture gate (cone_codec.py)", "per-channel space-time weights w_{c,i,ℓ} on the per-location candidate tokens, ring ungated; test: with every s_c → +∞ and τ_c → +∞ the forward pass equals the ungated model bit-for-bit, and the finite-difference gradient of the loss w.r.t. S₀, Σ_v, d, s_c, τ_c matches autograd"],
    ["The parameter maps and the harden step (cone_codec.py::ConeGeometry, cone_sampler.py)", "1° maps per group as coarse 10° + residual, matrix-log storage, the log-space floors, the prior, mirror, stop-gradient, init = A0; the sampler's dot table becomes a per-region lookup swapped at epoch boundaries; test: the table re-baked at init equals step 1's fixed dots to < 0.1 cell, and a swap mid-run changes no read of the current epoch"],
    ["Read-outs in the trainer (train_cone.py)", "per-task bars, R3 drift maps exported as data/cone_drift_<group>.json for the app's Cones tab, R4 attention-by-depth-and-bearing at the western-boundary anchors, R5 flag; then the three arms as recipes f7l2-hourglass-{fixed,learned,primed}"],
  ];
  steps.forEach((st, i) => stepCard(s, 0.6, 1.80 + i * 0.90, 12.13, 0.84, i + 1, st[0], st[1]));
  s.addText("Each step lands on main with its test before the next starts; step 4 is the only one that touches the data path, and only at epoch boundaries.", {
    x: 0.6, y: 6.38, w: 12.13, h: 0.3, fontSize: 11, italic: true, color: MUT,
    fontFace: FS, isTextBox: true, margin: 0, valign: "middle",
  });
}

// ================================================================ slide 19
{
  const s = newSlide("Open decisions for Chris", "Two settled, seven still open — all reversible");
  card(s, 0.6, 1.56, 12.13, 0.88, { fill: "11202F", line: GOLD });
  s.addText([
    ...rt("Settled 15 Sep (Chris):  ", { fontSize: 12.5, bold: true, color: GOLD, fontFace: FS }),
    ...rt("**1°** parameter maps with the coarse-plus-residual prior; **no Argo ellipses** — the nearest-profile tokens stay as E-071 §3 specifies them.", { fontSize: 12.5, color: TXT, fontFace: FS }),
  ], { x: 0.9, y: 1.70, w: 11.53, h: 0.6, isTextBox: true, margin: 0, valign: "middle" });
  const W4 = 2.867, X4 = [0.60, 3.687, 6.774, 9.861];
  labelCard(s, X4[0], 2.66, W4, 1.92, "Future horizon",
    "**+6 pentads** to mirror the past; or shorter (+1…+2 as E-069) with denser targets?", { size: 10.5 });
  labelCard(s, X4[1], 2.66, W4, 1.92, "Task shares",
    "**0.35 / 0.15 / 0.25 / 0.25** — any of them a lever you want set differently?", { size: 10.5 });
  labelCard(s, X4[2], 2.66, W4, 1.92, "Prior strength on the 1° residual",
    "One weight, to be set by held-out loss on the training years — or fixed by hand for the first arm?", { size: 10.5 });
  labelCard(s, X4[3], 2.66, W4, 1.92, "The v2 safety ring",
    "**Kept** at every lag (24 tokens per lag per group); dropping it halves the past-cone tokens but restores v2's “too narrow” risk", { size: 10.5 });
  const X = [0.60, 4.72, 8.84], W = 3.89;
  labelCard(s, X[0], 4.74, W, 1.92, "Seasonal drift term",
    "**Deferred** to phase 2; or in from the start (monsoon regions)?", { size: 11 });
  labelCard(s, X[1], 4.74, W, 1.92, "Harden interval and geometry LR",
    "**N = 2 k steps** and the geometry learning-rate multiplier **×10** — first guesses, both recorded per run", { size: 11 });
  labelCard(s, X[2], 4.74, W, 1.92, "Time inside the 30-day window",
    "**Dense**, one slice per pentad. A log-spaced inner time ramp (1, 2, 3, 5, 8, 13 pentads at the same six slices) is an **ablation**, not the design — it would move the codec's window into stage 2's", { size: 11 });
}

// ================================================================ slide 20
{
  const s = newSlide("E-080 · the cut-off mirrored double cone", "Sources");
  bulletCard(s, 0.6, 1.60, 7.55, 4.6, "the record this deck stands on", [
    "**E-069 · Two stencils, one cone** — [ml/plans/E069_cone_codec.md](" + U.e069 + "); results in [ml/EXPERIMENTS.md#e-069](" + U.exps + ") (five seeds, H1 refuted; what history buys)",
    "**E-069b masking-plan fix** — [`ml/cone_codec.py::default_plan`](" + U.codec + ") (chan_drop_scope, anchor_hidden_only), [`ml/train_cone.py`](" + U.traincone + ")",
    "**E-071 · Cone v2** — [ml/plans/E071_cone_v2.md](" + U.e071 + ") (§3 profile tokens, §4 design speeds and log-radial dots, §4.5 the split in time, §6.4 per-location tokens)",
    "**E-076 · Family 8 nearest observations** — [ml/plans/E076_family8_nearest_observations.md](" + U.e076 + ") (the (value, Δx, Δy, Δt, n_R) token)",
    "**The geometry as built** — [`ml/cone.py`](" + U.cone + ") (FAMILIES, reach_km, slots, inner_dots, outer_spiral, coverage_report)",
    "**E-026 · the sunflower stencil** — [`ml/temporal.py::spiral_offsets`](" + U.temporal + "), the pattern the warped sunflower reuses",
  ], 10.5);
  bulletCard(s, 8.35, 1.60, 4.38, 4.6, "and on", [
    "**The geometry learning-rate multiplier and soft sampling weights:** [Zhu et al. 2021, Deformable DETR](" + U.zhu + ")",
    "**The step-size argument for log-space parameters:** [Kingma & Ba 2015, Adam](" + U.adam + ") — the update is ≈ lr per step in the parameter's own units",
    "**The matrix logarithm of a symmetric 2×2 covariance:** S = m·I + k·[cos 2θ, sin 2θ; sin 2θ, −cos 2θ] with m = ln σ~{1} + ln σ~{2}, k = ln σ~{1} − ln σ~{2} — standard, no reference needed",
    "**Currents:** E-071 §4.1's table (Somali 3.6 m/s, Gulf Stream ~2.5, DWBC 0.02–0.1 m/s); RAPID 26.5°N array for the limb directions",
  ], 10.5);
}

// ================================================================ slide 21
// The body is a function because it is drawn TWICE: here, as the deck's last
// slide, and again on its own in the standalone one-slide export below.
const SUMMARY_HEADLINE =
  "An hourglass that reads the past through a cone it shapes itself, tested three ways against its own fixed twin";

// a compact label card: tighter padding than labelCard, top-aligned, so the
// column's picture and its text share the height without the body touching
// the border
function sumCard(s, x, y, w, h, label, body, size) {
  card(s, x, y, w, h);
  s.addText(stack([
    [label, { fontSize: 10.5, bold: true, color: GOLD, fontFace: FS }, 5],
    [body, { fontSize: size, color: TXT, fontFace: FS }, 0],
  ]), { x: x + 0.22, y: y + 0.10, w: w - 0.44, h: h - 0.20, isTextBox: true,
        margin: 0, valign: "top" });
}

// the small gold cap above each column's picture
function colLabel(s, x, w, text) {
  s.addText(text.toUpperCase(), {
    x, y: 1.50, w, h: 0.22, fontSize: 10, bold: true, color: GOLD,
    fontFace: FS, charSpacing: 1.1, isTextBox: true, margin: 0,
    valign: "middle",
  });
}

function summaryBody(s) {
  // Three columns, each a picture over its text: this slide also travels on
  // its own, so its reader has not seen slides 3 and 8.
  const C1 = 0.60, C2 = 4.45, C3 = 8.30, CW = 3.65, C3W = 4.43;
  const IY = 1.78, IH = 2.30;            // both pictures are 1.587 : 1
  const FSZ = 8.5;

  // ---- column 1 · the hourglass
  colLabel(s, C1, CW, "the hourglass");
  s.addImage({ path: `${FIG}/summary_hourglass.png`, x: C1, y: IY, w: CW, h: IH });
  sumCard(s, C1, 4.16, CW, 2.59, "The design · slides 3–7",
    "The present is a **waist** of ~13 cells, not a single tip; the past cone reads the last 30 days at every pentad; a **prediction cone** — the past cone mirrored through the anchor, +1…+6 pentads — is sampled the same way and used only as targets. Each example draws one of **four tasks**: forecast from waist + past (0.35), from the waist alone (0.15), nowcast the hidden present from the past under five dropout patterns (0.25), fill-in (0.25) — under E-069b's rules: knowable targets only, no copy targets, every loss family scored against its own predict-the-mean bar",
    9);

  // ---- column 2 · the cone the model shapes
  colLabel(s, C2, CW, "the cone the model shapes");
  s.addImage({ path: `${FIG}/summary_cone.png`, x: C2, y: IY, w: CW, h: IH });
  sumCard(s, C2, 4.16, CW, 2.59, "The cone the model shapes · slides 8–12",
    "A fixed 24-point sunflower, phase-rotated by the golden angle per lag, warped by **eight dimensionless numbers per 1° cell and channel group**: a drift **d** toward the source (capped at cone v2's design speed) and two ellipses stored as **matrix logarithms** — no kilometres, no angle to wrap — with Σ(ℓ) = Σ~{0} + ℓ^{2}Σ~{v}. **One direction per flow, one scale and one memory per channel**: s~{c} and τ~{c} inflate or shrink the shared ellipse and set how far back the channel reads, applied as a Gaussian **aperture** over the group's shared per-location tokens — zero extra tokens. Learned **gate-first** (a soft aperture over fixed dots, so the loader never depends on live weights), then **hardened** (dot tables re-baked at epoch boundaries)",
    FSZ);

  // ---- column 3 · the experiment, over the reserved panel
  colLabel(s, C3, C3W, "the experiment");
  card(s, C3, IY, C3W, IH, { fill: "11202F", line: GOLD, lw: 1.5, dash: "dash" });
  s.addText("RESULTS — figure to be inserted", {
    x: C3, y: IY + 0.60, w: C3W, h: 0.42, fontSize: 16, bold: true,
    color: MUT, fontFace: FS, align: "center", isTextBox: true, margin: 0,
    valign: "middle",
  });
  s.addText("R1 · future-cone loss per lead, A0 fixed / A1 learned / A2 primed, 3 seeds · R3 · learned drift vs upstream · R4 · the western-boundary depth check", {
    x: C3 + 0.40, y: IY + 1.12, w: C3W - 0.80, h: 0.80, fontSize: 10,
    color: MUT, fontFace: FS, align: "center", isTextBox: true, margin: 0,
    valign: "top",
  });
  s.addText("Reserved for the measured result — empty until a run has produced it.", {
    x: C3, y: 4.10, w: C3W, h: 0.30, fontSize: 9.5, italic: true, color: MUT,
    fontFace: FS, isTextBox: true, margin: 0, valign: "middle",
  });
  sumCard(s, C3, 4.42, C3W, 1.05, "The guards · slides 13, 15",
    "Coarse-10°-plus-1°-residual prior; log-space soft floors; mirror tie; stop-gradient on target positions; evaluation geometry frozen for every arm; the far ring kept and never gated; the aperture never touches the loss weights — so the cone cannot learn to ask easy questions",
    FSZ);
  // NOTE: at most ONE emphasised run may follow the hyperlink in this
  // paragraph — with two after it LibreOffice renders the link in the body
  // colour instead of accent blue (measured; the PPTX markup is identical to
  // every other link in the deck). Hence "3 seeds each" sits before it.
  sumCard(s, C3, 5.53, C3W, 1.22, "The experiment · verdict · slide 17",
    "**A0** fixed hourglass — the control and everyone's eval geometry · **A1** learned from A0's init · **A2** learned, surface drift primed at −u_clim·Δt. Same 7 M codec, 20 k steps, frozen protocol, **3 seeds each**, one tensor ([family 7.2](" + U.f7 + ")); read-outs R1–R5. **Verdict**: A1 beats A0 on R1 at every lead ≤ 3, paired at all seeds, R5 clean → adopt; A2 > A1 ≈ A0 → geometry from climatology; neither → keep A0",
    FSZ);
}

let SUMMARY_N, TWO_SCALES_N;
{
  const s = newSlide("The deck on one slide — with room for the result",
                     SUMMARY_HEADLINE);
  SUMMARY_N = SLIDE_N;               // captured: the standalone reuses these notes
  summaryBody(s);
}

// ================================================================ slide 22
const TWO_SCALES_HEADLINE =
  "Stage 1 reads raw values through a small hourglass; stage 2 reads stage 1's embeddings through the same shape, twenty times longer and five times wider";

// The figure is 11.0 × 4.8 in (aspect 2.2917); the width is set from the height
// the slide can spare (1.44 → 6.35), which leaves the italic line and the
// legend strip room above the footer, and it is centred on what is left.
function twoScalesBody(s) {
  const FW = 11.24, FH_ = FW / 2.2917, FX = (13.333 - FW) / 2, FY = 1.44;
  s.addImage({ path: `${FIG}/two_scales_3d.png`, x: FX, y: FY, w: FW, h: FH_ });

  s.addText("Stage 2's mirrored future side is the E-080 shape carried up a level — design intent; the stage-2 heads scored so far predict the next embedding and roll it forward.", {
    x: 0.6, y: 6.41, w: 12.13, h: 0.24, fontSize: 9, italic: true, color: MUT,
    fontFace: FS, isTextBox: true, margin: 0, valign: "middle",
  });

  // legend strip: a coloured square + one line of meaning, three times
  const chips = [
    [0.60, 3.85, BLUE, "**past cone** · input, never a forecast target, sometimes held out"],
    [4.55, 3.35, GOLD, "**waist** · present — input in T1/T2/T4, predicted in T3"],
    [8.05, 2.50, ORANGE, "**future cone** · targets only, never input"],
  ];
  chips.forEach(([x, w, col, text]) => {
    s.addShape(pres.ShapeType.rect, {
      x, y: 6.735, w: 0.13, h: 0.13,
      fill: { color: col }, line: { color: col, width: 0.5 },
    });
    s.addText(rt(text, { fontSize: 9, color: TXT, fontFace: FS, accent: col }), {
      x: x + 0.20, y: 6.66, w: w - 0.20, h: 0.28, isTextBox: true, margin: 0,
      valign: "middle",
    });
  });
  // sources, bottom right — no emphasised run follows either link (the
  // LibreOffice link-colour quirk documented in the README)
  s.addText(rt("[cone.py::outer_spiral](" + U.cone + ")  ·  [E-071 §4.5](" + U.e071 + ")",
               { fontSize: 8.5, color: MUT, fontFace: FS }), {
    x: 10.58, y: 6.66, w: 2.15, h: 0.28, isTextBox: true, margin: 0,
    align: "right", valign: "middle",
  });
}

{
  const s = newSlide("The same hourglass at two scales", TWO_SCALES_HEADLINE);
  TWO_SCALES_N = SLIDE_N;
  twoScalesBody(s);
}

// ================================================ standalone one-slide exports
// Same layout, same background, same notes — no slide number in the footer.
function standalone(title, kicker, headline, notesN, body) {
  const p = new pptxgen();
  p.layout = "LAYOUT_WIDE";
  p.author = "Deck builder";
  p.title = title;
  const s = p.addSlide();
  s.background = { color: BG };
  s.addText(kicker.toUpperCase(), {
    x: 0.6, y: 0.30, w: 12.13, h: 0.26, fontSize: 11, bold: true, color: BLUE,
    fontFace: FS, charSpacing: 1.3, isTextBox: true, margin: 0, valign: "middle",
  });
  s.addText(headline, {
    x: 0.6, y: 0.56, w: 12.13, h: 0.84, fontSize: titleSize(headline),
    bold: true, color: TXT, fontFace: FH, isTextBox: true, margin: 0,
    valign: "middle",
  });
  s.addText(FOOTER, {
    x: 0.6, y: 6.95, w: 12.13, h: 0.3, fontSize: 10, color: FOOT, fontFace: FS,
    isTextBox: true, margin: 0, valign: "middle",
  });
  s.addNotes(notesText(notesN));
  body(s);
  return p;
}

const pres2 = standalone(
  "The cut-off mirrored double cone — one-slide summary",
  "E-080 · the cut-off mirrored double cone · one-slide summary",
  SUMMARY_HEADLINE, SUMMARY_N, summaryBody);

const pres3 = standalone(
  "The cut-off mirrored double cone — the hourglass at two scales",
  "E-080 · the cut-off mirrored double cone · the hourglass at two scales",
  TWO_SCALES_HEADLINE, TWO_SCALES_N, twoScalesBody);

pres.writeFile({ fileName: OUT })
  .then(() => console.log("wrote", OUT, "slides:", SLIDE_N))
  .then(() => pres2.writeFile({ fileName: OUT_SUMMARY }))
  .then(() => console.log("wrote", OUT_SUMMARY, "slides: 1"))
  .then(() => pres3.writeFile({ fileName: OUT_TWO_SCALES }))
  .then(() => console.log("wrote", OUT_TWO_SCALES, "slides: 1"));

// Geospatial representation models — stencil-level overview
const pptxgen = require("pptxgenjs");
const pres = new pptxgen();
pres.layout = "LAYOUT_WIDE"; // 13.33 x 7.5
pres.author = "Earth 2 research";
pres.title = "Geospatial representation models — stencil-level overview";

// ---------- palette ----------
const NAVY = "0B1F3A";   // dark bg / headings
const INK = "1B2430";    // body text
const MUTED = "6B7A8F";  // captions
const TEAL = "1C7293";   // secondary
const SPACE = "2E86AB";  // spatial-stencil colour
const TIME = "E07A2F";   // temporal-stencil colour
const EMB = "6A4C93";    // embedding colour
const OUTPX = "F2C14E";  // output pixel highlight
const PALE = "EEF3F8";   // card tint
const WHITE = "FFFFFF";
const GRIDLINE = "B8C4D3";
const FONT_H = "Georgia";   // in both Office and Google Slides (Cambria is not)
const FONT_B = "Calibri";   // confirmed present in Google Slides

const W = 13.33, H = 7.5;
const MADE = [];                                  // every slide in creation order
{ const _add = pres.addSlide.bind(pres); pres.addSlide = function (...a) { const sl = _add(...a); MADE.push(sl); return sl; }; }

// ---------- helpers ----------
function title(slide, text, sub) {
  slide._deckTitle = text; slide._deckSub = sub || "";
  // Georgia is wider than Cambria and Google substitutes fonts on import, so size the title by
  // length rather than trusting it to fit: a wrapped title collides with the subtitle below.
  const n = text.length;
  const fs = n <= 44 ? 30 : n <= 56 ? 26 : n <= 68 ? 23 : 21;
  slide.addText(text, { x: 0.6, y: 0.34, w: 12.1, h: 0.72, fontFace: FONT_H, fontSize: fs, bold: true, color: NAVY, valign: "middle", isTextBox: true, margin: 0 });
  if (sub) slide.addText(sub, { x: 0.6, y: 1.06, w: 12.1, h: 0.4, fontFace: FONT_B, fontSize: (sub.length > 120 ? 12.5 : 14), color: MUTED, italic: true, isTextBox: true, margin: 0 });
}
let pageNo = 1;
function footer(slide) {
  pageNo++;
  slide.addText(`Representation-model survey · Aug 2026 · ${pageNo}`, { x: 0.6, y: 7.05, w: 8, h: 0.3, fontFace: FONT_B, fontSize: 9, color: MUTED, isTextBox: true, margin: 0 });
}
function bullets(slide, items, opts) {
  const arr = items.map((t, i) => {
    const o = { bullet: true, breakLine: i < items.length - 1, paraSpaceAfter: 6 };
    if (typeof t === "string") return { text: t, options: o };
    return { text: t.text, options: Object.assign(o, t.options || {}) };
  });
  slide.addText(arr, Object.assign({ fontFace: FONT_B, fontSize: 12, color: INK, valign: "top", isTextBox: true, margin: 2 }, opts));
}
function label(slide, text, x, y, w, h, extra) {
  slide.addText(text, Object.assign({ x, y, w, h, fontFace: FONT_B, fontSize: 10, color: INK, isTextBox: true, margin: 0, valign: "top" }, extra || {}));
}
// fact rows: bold key + value
function facts(slide, rows, x, y, w, fontSize = 11.5) {
  const runs = [];
  rows.forEach((r, i) => {
    runs.push({ text: r[0] + "  ", options: { bold: true, color: TEAL, fontSize } });
    runs.push({ text: r[1], options: { color: INK, fontSize, breakLine: i < rows.length - 1, paraSpaceAfter: 7 } });
  });
  slide.addText(runs, { x, y, w, h: 5.2, fontFace: FONT_B, valign: "top", isTextBox: true, margin: 2 });
}

// ---------- stencil drawing ----------
// spec = { space:{...}, time:{...}, out:{...} }
function drawStencil(slide, x0, y0, spec) {
  const gridSize = 2.6;
  // caption
  slide.addText("Stencil — what feeds one embedding", { x: x0, y: y0, w: 7.2, h: 0.3, fontFace: FONT_B, fontSize: 11, bold: true, color: NAVY, isTextBox: true, margin: 0 });
  // SPACE panel
  const gx = x0, gy = y0 + 0.4;
  slide.addText("SPACE", { x: gx, y: gy - 0.05, w: 1.2, h: 0.25, fontFace: FONT_B, fontSize: 9, bold: true, color: SPACE, isTextBox: true, margin: 0, charSpacing: 2 });
  const ggy = gy + 0.55;
  drawGrid(slide, gx, ggy, gridSize, spec.space);
  // space labels under grid
  slide.addText(spec.space.lines.map((t, i) => ({ text: t, options: { breakLine: i < spec.space.lines.length - 1 } })),
    { x: gx, y: ggy + gridSize + 0.08, w: gridSize + 0.4, h: 0.95, fontFace: FONT_B, fontSize: 9.5, color: INK, valign: "top", isTextBox: true, margin: 0 });

  // TIME panel
  const tx = x0 + 3.1, tw = 2.45;
  slide.addText("TIME", { x: tx, y: gy - 0.05, w: 1.2, h: 0.25, fontFace: FONT_B, fontSize: 9, bold: true, color: TIME, isTextBox: true, margin: 0, charSpacing: 2 });
  drawTimeline(slide, tx, ggy, tw, gridSize, spec.time);
  slide.addText(spec.time.lines.map((t, i) => ({ text: t, options: { breakLine: i < spec.time.lines.length - 1 } })),
    { x: tx, y: ggy + gridSize + 0.08, w: tw, h: 0.95, fontFace: FONT_B, fontSize: 9.5, color: INK, valign: "top", isTextBox: true, margin: 0 });

  // arrows into embedding
  const ex = x0 + 5.95, ey = ggy + 0.35, ew = 0.75, eh = gridSize - 0.7;
  slide.addShape(pres.shapes.RIGHT_ARROW, { x: tx + tw + 0.1, y: ggy + gridSize / 2 - 0.15, w: 0.3, h: 0.3, fill: { color: GRIDLINE }, line: { color: GRIDLINE, width: 0 } });
  slide.addShape(pres.shapes.RIGHT_ARROW, { x: gx + gridSize + 0.12, y: ggy + gridSize / 2 - 0.15, w: 0.3, h: 0.3, fill: { color: GRIDLINE }, line: { color: GRIDLINE, width: 0 } });
  // EMBEDDING vector: stacked cells
  slide.addText("OUTPUT", { x: ex - 0.1, y: gy - 0.05, w: 1.2, h: 0.25, fontFace: FONT_B, fontSize: 9, bold: true, color: EMB, isTextBox: true, margin: 0, charSpacing: 2 });
  const cells = 8, ch = eh / cells;
  for (let i = 0; i < cells; i++) {
    const shade = ["6A4C93", "7B5EA7", "8C70BA", "9D82CD", "8C70BA", "7B5EA7", "6A4C93", "5A3D80"][i];
    slide.addShape(pres.shapes.RECTANGLE, { x: ex, y: ey + i * ch, w: ew, h: ch, fill: { color: shade }, line: { color: WHITE, width: 0.75 } });
  }
  slide.addText(spec.out.dim, { x: ex - 0.42, y: ey + eh + 0.05, w: 1.6, h: 0.35, fontFace: FONT_B, fontSize: 12, bold: true, color: EMB, align: "center", isTextBox: true, margin: 0 });
  slide.addText(spec.out.lines.map((t, i) => ({ text: t, options: { breakLine: i < spec.out.lines.length - 1 } })),
    { x: ex - 0.42, y: ey + eh + 0.4, w: 1.6, h: 1.0, fontFace: FONT_B, fontSize: 8.5, color: INK, align: "center", valign: "top", isTextBox: true, margin: 0 });
}

// grid: {n, tile:[c0,c1] cells of tile within n grid (or null=whole), token:[r,c,size], outPx:[r,c], dashed:[sizes], layers:k, note}
function drawGrid(slide, x, y, size, g) {
  const n = g.n, cs = size / n;
  // modality layers behind
  const layers = g.layers || 1;
  for (let l = layers - 1; l >= 1; l--) {
    slide.addShape(pres.shapes.RECTANGLE, { x: x + 0.08 * l, y: y - 0.08 * l, w: size, h: size, fill: { color: PALE }, line: { color: GRIDLINE, width: 0.75 } });
  }
  // base
  slide.addShape(pres.shapes.RECTANGLE, { x, y, w: size, h: size, fill: { color: WHITE }, line: { color: GRIDLINE, width: 0.75 } });
  // grid lines
  const lw = n > 16 ? 0.4 : 0.6;
  for (let i = 1; i < n; i++) {
    slide.addShape(pres.shapes.LINE, { x: x + i * cs, y, w: 0, h: size, line: { color: GRIDLINE, width: lw } });
    slide.addShape(pres.shapes.LINE, { x, y: y + i * cs, w: size, h: 0, line: { color: GRIDLINE, width: lw } });
  }
  // context/tile outline (whole grid = tile)
  if (g.tileOutline) {
    slide.addShape(pres.shapes.RECTANGLE, { x, y, w: size, h: size, fill: { type: "none" }, line: { color: SPACE, width: 2.25 } });
  }
  // dashed multiscale squares centred
  if (g.dashed) {
    g.dashed.forEach(k => {
      const s = k * cs;
      slide.addShape(pres.shapes.RECTANGLE, { x: x + (size - s) / 2, y: y + (size - s) / 2, w: s, h: s, fill: { type: "none" }, line: { color: SPACE, width: 1.25, dashType: "dash" } });
    });
  }
  // token
  if (g.token) {
    const [r, c, k] = g.token;
    slide.addShape(pres.shapes.RECTANGLE, { x: x + c * cs, y: y + r * cs, w: k * cs, h: k * cs, fill: { color: SPACE, transparency: 25 }, line: { color: SPACE, width: 1.25 } });
  }
  // output pixel
  if (g.outPx) {
    const [r, c] = g.outPx;
    slide.addShape(pres.shapes.RECTANGLE, { x: x + c * cs, y: y + r * cs, w: cs, h: cs, fill: { color: OUTPX }, line: { color: "B58900", width: 1 } });
  }
  // in-grid annotations
  (g.notes || []).forEach(nt => {
    slide.addText(nt.text, { x: x + nt.x, y: y + nt.y, w: nt.w, h: nt.h || 0.25, fontFace: FONT_B, fontSize: 8.5, color: nt.color || SPACE, bold: true, isTextBox: true, margin: 0, align: nt.align || "left", fill: nt.fill ? { color: WHITE, transparency: 15 } : undefined });
  });
}

// timeline: {window:[a,b] fraction, rows:[{label, ticks:[fractions] , style:'dot'|'bar'}], axisLabel, windowLabel, dashedWindow}
function drawTimeline(slide, x, y, w, h, t) {
  const axisY = y + h * 0.62;
  // window shading
  if (t.window) {
    const [a, b] = t.window;
    slide.addShape(pres.shapes.RECTANGLE, { x: x + a * w, y: y + 0.15, w: Math.max((b - a) * w, 0.06), h: h * 0.62 - 0.15 + 0.12, fill: { color: TIME, transparency: 82 }, line: { color: TIME, width: 1, dashType: t.dashedWindow ? "dash" : "solid" } });
    const lw = Math.max((b - a) * w, 1.2) + 0.5;
    slide.addText(t.windowLabel || "", { x: x + (a + b) / 2 * w - lw / 2, y: y - 0.12, w: lw, h: 0.26, fontFace: FONT_B, fontSize: 7.5, bold: true, color: TIME, align: "center", valign: "bottom", isTextBox: true, margin: 0 });
  }
  // axis
  slide.addShape(pres.shapes.LINE, { x, y: axisY, w, h: 0, line: { color: INK, width: 1, endArrowType: "triangle" } });
  slide.addText(t.axisLabel || "time", { x, y: axisY + 0.06, w, h: 0.22, fontFace: FONT_B, fontSize: 8.5, color: MUTED, align: "right", isTextBox: true, margin: 0 });
  // axis ticks (e.g. months)
  (t.axisTicks || []).forEach(f => {
    slide.addShape(pres.shapes.LINE, { x: x + f * w, y: axisY - 0.04, w: 0, h: 0.08, line: { color: INK, width: 0.75 } });
  });
  // rows of observations
  const rows = t.rows || [];
  const rowH = rows.length ? (h * 0.62 - 0.6) / rows.length : 0;
  rows.filter(r => r.ticks && r.ticks.length).forEach((r, i) => {
    const ry = y + 0.5 + i * rowH + rowH / 2;
    const [wa, wb] = t.window || [0, 1];
    slide.addText(r.label, { x: x + wa * w + 0.05, y: ry - 0.25, w: (wb - wa) * w - 0.1, h: 0.2, fontFace: FONT_B, fontSize: 7.5, color: r.color || INK, fill: { color: WHITE, transparency: 20 }, isTextBox: true, margin: 1 });
    r.ticks.forEach(f => {
      if (r.style === "bar") {
        slide.addShape(pres.shapes.RECTANGLE, { x: x + f * w - 0.02, y: ry - 0.06, w: 0.04, h: 0.16, fill: { color: r.color || TIME }, line: { width: 0 } });
      } else {
        const d = r.size || 0.07;
        slide.addShape(pres.shapes.OVAL, { x: x + f * w - d / 2, y: ry - d / 2, w: d, h: d, fill: { color: r.color || TIME }, line: { width: 0 } });
      }
    });
  });
  // snapshot marker
  if (t.snapshot !== undefined) {
    const sx = x + t.snapshot * w;
    slide.addShape(pres.shapes.LINE, { x: sx, y: y + 0.3, w: 0, h: axisY - y - 0.3, line: { color: TIME, width: 2.25 } });
    slide.addShape(pres.shapes.OVAL, { x: sx - 0.07, y: axisY - 0.07, w: 0.14, h: 0.14, fill: { color: TIME }, line: { width: 0 } });
    slide.addText(t.snapshotLabel || "single acquisition", { x: sx - 1.0, y: y + 0.02, w: 2.0, h: 0.25, fontFace: FONT_B, fontSize: 8.5, bold: true, color: TIME, align: "center", isTextBox: true, margin: 0 });
  }
}

function seq(n, a = 0, b = 1) { return Array.from({ length: n }, (_, i) => a + (b - a) * (i + 0.5) / n); }
function jitter(n, a, b, seed) { let s = seed; const r = () => { s = (s * 9301 + 49297) % 233280; return s / 233280; }; return Array.from({ length: n }, () => a + (b - a) * r()).sort(); }

// model slide
function modelSlide(n, name, tagline, rows, stencil, notes, fs) {
  const s = pres.addSlide();
  s.background = { color: WHITE };
  title(s, name, tagline);
  // facts card
  s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x: 0.6, y: 1.55, w: 5.0, h: 5.3, fill: { color: PALE }, line: { width: 0 }, rectRadius: 0.08 });
  facts(s, rows, 0.75, 1.65, 4.75, fs || 11.5);
  drawStencil(s, 5.75, 1.55, stencil);
  footer(s);
  if (notes) s.addNotes(notes);
  return s;
}

// =====================================================================
// 1. Title
{
  const s = pres.addSlide();
  s.background = { color: NAVY };
  const TW = 8.5;    // text column stops well clear of the motif at x = 9.75
  s.addText("Geospatial representation models", { x: 0.8, y: 2.15, w: TW, h: 0.8, fontFace: FONT_H, fontSize: 32, bold: true, color: WHITE, valign: "middle", isTextBox: true, margin: 0 });
  s.addText("A stencil-level overview of AlphaEarth, TESSERA, OlmoEarth, TerraMind, Prithvi-EO 2.0 and IBM's Prithvi-based ocean model (Granite-Geospatial-Ocean)", { x: 0.8, y: 3.1, w: TW, h: 0.95, fontFace: FONT_B, fontSize: 17, color: "CADCFC", isTextBox: true, margin: 0 });
  s.addText("What area in space and what span of time goes into each embedding — and what each model actually hands you", { x: 0.8, y: 4.35, w: TW, h: 0.7, fontFace: FONT_B, fontSize: 13.5, italic: true, color: "9FB3C8", isTextBox: true, margin: 0 });
  s.addText("Earth 2 representation-model research · 31 Aug 2026", { x: 0.8, y: 6.6, w: 8, h: 0.4, fontFace: FONT_B, fontSize: 12, color: "9FB3C8", isTextBox: true, margin: 0 });
  // decorative stencil motif on the right
  const gx = 9.75, gy = 1.95, size = 2.6, n = 8, cs = size / n;
  s.addShape(pres.shapes.RECTANGLE, { x: gx + 0.12, y: gy - 0.12, w: size, h: size, fill: { color: "16305A" }, line: { color: "3A5A8C", width: 0.75 } });
  s.addShape(pres.shapes.RECTANGLE, { x: gx, y: gy, w: size, h: size, fill: { color: "10284C" }, line: { color: "3A5A8C", width: 0.75 } });
  for (let i = 1; i < n; i++) {
    s.addShape(pres.shapes.LINE, { x: gx + i * cs, y: gy, w: 0, h: size, line: { color: "3A5A8C", width: 0.5 } });
    s.addShape(pres.shapes.LINE, { x: gx, y: gy + i * cs, w: size, h: 0, line: { color: "3A5A8C", width: 0.5 } });
  }
  s.addShape(pres.shapes.RECTANGLE, { x: gx + 2 * cs, y: gy + 2 * cs, w: 4 * cs, h: 4 * cs, fill: { color: SPACE, transparency: 40 }, line: { color: SPACE, width: 1.5 } });
  s.addShape(pres.shapes.RECTANGLE, { x: gx + 3 * cs, y: gy + 3 * cs, w: cs, h: cs, fill: { color: OUTPX }, line: { width: 0 } });
  // time bar
  s.addShape(pres.shapes.LINE, { x: gx, y: gy + size + 0.6, w: size, h: 0, line: { color: "9FB3C8", width: 1, endArrowType: "triangle" } });
  s.addShape(pres.shapes.RECTANGLE, { x: gx + 0.5, y: gy + size + 0.35, w: 1.5, h: 0.5, fill: { color: TIME, transparency: 70 }, line: { color: TIME, width: 1 } });
  seq(9, 0.5 / size, 2.0 / size).forEach(f => s.addShape(pres.shapes.OVAL, { x: gx + f * size - 0.035, y: gy + size + 0.6 - 0.035, w: 0.07, h: 0.07, fill: { color: TIME }, line: { width: 0 } }));
  s.addNotes("Deck purpose: compare six geospatial representation models at the level that matters for our own codec design — what spatial footprint and temporal window feed one embedding, and what each project actually releases (weights, precomputed embeddings, or both).");
}

// =====================================================================
// 2. How to read the stencil
{
  const s = pres.addSlide();
  s.background = { color: WHITE };
  title(s, "How to read the stencil diagrams", "Every model slide uses the same three-panel convention");
  drawStencil(s, 0.9, 1.55, {
    space: { n: 14, tileOutline: true, token: [5, 5, 3], outPx: [6, 6], layers: 3, lines: ["Grid = input tile the encoder sees at once (blue outline).", "Blue square = one token / receptive field. Yellow = the output pixel.", "Stacked sheets = co-registered modalities / channels."] },
    time: { window: [0.15, 0.85], windowLabel: "temporal window", rows: [{ label: "observations used", ticks: seq(9, 0.15, 0.85) }], axisTicks: seq(12), lines: ["Shaded band = window summarised by one embedding.", "Dots = observations sampled inside it. A single tall bar = one snapshot acquisition."] },
    out: { dim: "D dims", lines: ["one vector per pixel, per token, or per tile"] }
  });
  // legend column
  const lx = 8.6;
  s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x: lx, y: 1.55, w: 4.1, h: 5.3, fill: { color: PALE }, line: { width: 0 }, rectRadius: 0.08 });
  s.addText("Four questions we ask of each model", { x: lx + 0.2, y: 1.7, w: 3.7, h: 0.4, fontFace: FONT_B, fontSize: 14, bold: true, color: NAVY, isTextBox: true, margin: 0 });
  bullets(s, [
    { text: "Per-pixel or patch?  Does the embedding of a pixel see its neighbours (patch/ViT) or only its own time series (pixel-wise)?", options: {} },
    "Snapshot or period summary?  One acquisition, a handful of dates, or a whole year folded into one vector?",
    "Product or model?  Do we get precomputed global embeddings, open weights we can run on our own grid, or both?",
    "Does it touch the ocean?  Land-only masking is the norm; only one of the six is trained on sea pixels.",
  ], { x: lx + 0.2, y: 2.15, w: 3.75, h: 4.5, fontSize: 12 });
  footer(s);
  s.addNotes("Legend slide. The colours are fixed across the deck: blue = space, amber = time, purple = the output embedding vector, yellow = the single output pixel/location the embedding is attached to.");
}

// =====================================================================
// 3. AlphaEarth
modelSlide(3, "AlphaEarth Foundations (Google DeepMind)", "An 'embedding field': every 10 m land pixel → 64-D unit vector per year, 2017–2024",
  [
    ["Who / when", "DeepMind + Google Earth Engine; arXiv 2507.22291 (Jul 2025, v2 Sep 2025)."],
    ["Inputs", "Sentinel-2, Landsat 8/9, Sentinel-1, PALSAR-2, GEDI lidar, ERA5-Land, GRACE, GLO-30 DEM, NLCD, plus text (Wikipedia, GBIF) — >3 B observations."],
    ["Architecture", "Space-Time-Precision (STP) encoder — 15 blocks, each with three operators at three resolutions (spatial attention @ L/16, time-axial attention @ L/8, 3×3 convs @ L/2; widths 1024 / 512 / 128) exchanging state through a learned Laplacian pyramid (next slide). Teacher/student video embedders + text alignment; ~480 M deployed (1 B trained). Losses: reconstruction, batch-uniformity on S⁶³, consistency, text-contrastive."],
    ["Stencil", "Output 10 m pixel. Input frame 1.28 km × 1.28 km (128 × 128 px; supplement S2.1/S8.2): inference runs on 960 m tiles buffered by 160 m, so a pixel's context is at most the 1.28 km frame, with spatial attention global inside it. Continuous 'valid period' [t₀, t₁) — arbitrary windows in principle; released product = calendar-year summaries."],
    ["Output", "64-D, unit-norm, int8 in the public dataset; ~1.4 T footprints / yr. Land + coastal/inland water; no open ocean."],
    ["Availability", "Dataset only (CC-BY 4.0) in Earth Engine / GCS. Weights and code not released."],
    ["Results", "~23.9 % average error reduction vs. prior featurizations over 15 evaluations (10.4 % at 10-shot, 4.2 % at 1-shot); evapotranspiration regression R² 0.58. Per-dataset scores are charts only."],
  ],
  {
    space: { n: 16, tileOutline: true, token: [7, 7, 2], outPx: [8, 8], layers: 3, notes: [{ text: "frame 128 px = 1.28 km", x: 0.05, y: 0.02, w: 1.6 }, { text: "space-attention cell 160 m", x: 1.35, y: 1.42, w: 1.3, h: 0.25 }],
      lines: ["Output pixel: 10 m × 10 m (drawn 8× too large).", "Context: the 1.28 km frame. Spatial attention is global inside it on an 8 × 8 grid of 160 m cells; the outer 80 m is trimmed at inference, so no pixel sees further than ~640 m.", "≈10 co-registered sources at 10 m … 300 km native resolution."] },
    time: { window: [0.1, 0.9], windowLabel: "valid period — arbitrary; product = 1 calendar year", axisTicks: seq(12, 0.1, 0.9),
      rows: [
        { label: "S-2 (~5 d), S-1 (6–12 d)", ticks: seq(28, 0.1, 0.9), size: 0.05 },
        { label: "Landsat (16 d), PALSAR-2 (14 d)", ticks: seq(14, 0.1, 0.9), size: 0.06 },
        { label: "ERA5 daily, GRACE monthly, static DEM/LC", ticks: seq(6, 0.1, 0.9), size: 0.07, color: "B85C1E" },
      ],
      lines: ["All observations in the window are summarised — a period embedding, not a snapshot.", "Model supports interpolation/extrapolation in time; only annual layers are public."] },
    out: { dim: "64-D", lines: ["per 10 m pixel per year", "int8, unit sphere"] }
  },
  "AlphaEarth is the clearest statement of the 'embedding field' idea we are transplanting to the ocean — same 64-D width. Key caveat for us: it is a dataset, not a model. You cannot request a custom time window or run it over ocean pixels. Frame size and block count come from the supplement (S2.1, S2.4, S8.2), which the main text does not repeat.",
  11
);


// 3b. AlphaEarth — inside the STP encoder
{
  const s = pres.addSlide();
  s.background = { color: WHITE };
  title(s, "AlphaEarth — inside the STP encoder", "L = side length of the square input frame, in pixels — L = 128 (1.28 km at 10 m) for training and inference (supplement S2.1, S8.2). Three operators at L/16, L/8 and L/2 exchange state through a learned Laplacian pyramid after every one of 15 blocks.");
  const laneY = [1.95, 3.25, 4.55], laneH = 1.0;
  const lanes = [
    ["SPACE operator", "grid of L/16 × L/16 cells", "ViT-like spatial self-attention over the whole (down-scaled) frame — global context, coarse grid", SPACE],
    ["TIME operator", "grid of L/8 × L/8 cells", "time-axial self-attention along each pixel's sequence; every element conditioned on its sinusoidal timecode", TIME],
    ["PRECISION operator", "grid of L/2 × L/2 cells", "3×3 convolutions — keeps fine spatial detail near output resolution", "1F7A4D"],
  ];
  const lx = 0.6, lw = 2.55;
  lanes.forEach(([n, r, d, col], i) => {
    s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x: lx, y: laneY[i], w: lw, h: laneH, fill: { color: PALE }, line: { width: 0 }, rectRadius: 0.06 });
    s.addText([{ text: n, options: { bold: true, color: col, fontSize: 11, breakLine: true } }, { text: r, options: { bold: true, color: INK, fontSize: 10, breakLine: true } }, { text: d, options: { color: INK, fontSize: 8.5 } }],
      { x: lx + 0.12, y: laneY[i] + 0.06, w: lw - 0.2, h: laneH - 0.1, fontFace: FONT_B, valign: "top", isTextBox: true, margin: 0 });
  });
  // input projector
  const ix = 3.45;
  s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x: ix, y: 1.95, w: 1.05, h: 3.6, fill: { color: "F4F6F9" }, line: { color: GRIDLINE, width: 0.75 }, rectRadius: 0.06 });
  s.addText([{ text: "Input projectors", options: { bold: true, breakLine: true } }, { text: "one per source; frames + timestamps → feature maps at 1/2 L. Σ Nᵢ frames in, Σ Nᵢ maps out.", options: { fontSize: 8.5 } }],
    { x: ix + 0.08, y: 2.0, w: 0.9, h: 3.5, fontFace: FONT_B, fontSize: 10, color: INK, valign: "top", isTextBox: true, margin: 0 });
  // blocks
  const bx0 = 4.8, bw = 1.15, gap = 0.7, nB = 3;
  const boxH = 0.62;
  const cols = [SPACE, TIME, "1F7A4D"];
  for (let b = 0; b < nB; b++) {
    const bx = bx0 + b * (bw + gap);
    const ghost = b === nB - 1;
    s.addText(ghost ? "block 15" : `block ${b + 1}`, { x: bx, y: 1.6, w: bw, h: 0.25, fontFace: FONT_B, fontSize: 9, bold: true, color: MUTED, align: "center", isTextBox: true, margin: 0 });
    lanes.forEach((_, i) => {
      s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x: bx, y: laneY[i] + (laneH - boxH) / 2, w: bw, h: boxH, fill: { color: cols[i], transparency: ghost ? 70 : 15 }, line: { color: cols[i], width: 1, dashType: ghost ? "dash" : "solid" }, rectRadius: 0.05 });
      s.addText(["space", "time", "precision"][i], { x: bx, y: laneY[i] + (laneH - boxH) / 2, w: bw, h: boxH, fontFace: FONT_B, fontSize: 9, bold: true, color: ghost ? cols[i] : WHITE, align: "center", valign: "middle", isTextBox: true, margin: 0 });
    });
    // pyramid exchange: 3x3 arrows to next block
    if (b < nB - 1) {
      const x1 = bx + bw, x2 = bx + bw + gap;
      lanes.forEach((_, i) => lanes.forEach((_, j) => {
        const y1 = laneY[i] + laneH / 2, y2 = laneY[j] + laneH / 2;
        const flipV = y2 < y1;
        s.addShape(pres.shapes.LINE, { x: x1 + 0.02, y: Math.min(y1, y2), w: gap - 0.04, h: Math.abs(y2 - y1), flipV, line: { color: "8A97A8", width: 0.75, endArrowType: "triangle" } });
      }));
      if (b === 0) s.addText("learned Laplacian-pyramid rescaling: every operator feeds every operator of the next block", { x: x1 - 0.5, y: 5.6, w: gap + 1.0, h: 0.5, fontFace: FONT_B, fontSize: 8, color: MUTED, align: "center", isTextBox: true, margin: 0 });
    }
  }
  // ellipsis between block 2 and N
  s.addText("· · ·", { x: bx0 + 2 * bw + gap + 0.1, y: 3.3, w: gap - 0.2, h: 0.4, fontFace: FONT_B, fontSize: 16, color: MUTED, align: "center", valign: "middle", isTextBox: true, margin: 0 });
  // input arrow
  s.addShape(pres.shapes.RIGHT_ARROW, { x: ix + 1.05 + 0.03, y: 3.6, w: bx0 - ix - 1.05 - 0.06, h: 0.3, fill: { color: GRIDLINE }, line: { width: 0 } });
  // output
  const ox = bx0 + nB * bw + (nB - 1) * gap + 0.25;
  s.addShape(pres.shapes.RIGHT_ARROW, { x: ox - 0.22, y: 3.6, w: 0.2, h: 0.3, fill: { color: GRIDLINE }, line: { width: 0 } });
  s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x: ox, y: 1.95, w: 1.75, h: 3.6, fill: { color: PALE }, line: { width: 0 }, rectRadius: 0.06 });
  s.addText([
    { text: "Output head", options: { bold: true, color: EMB, fontSize: 11, breakLine: true } },
    { text: "final learned spatial resampling up to the precision resolution, then to 10 m", options: { fontSize: 8.5, breakLine: true } },
    { text: " ", options: { fontSize: 5, breakLine: true } },
    { text: "64-D per pixel", options: { bold: true, color: EMB, fontSize: 12, breakLine: true } },
    { text: "unit-norm (S⁶³), quantized to 64 bytes (int8) in the released layers", options: { fontSize: 8.5, breakLine: true } },
    { text: " ", options: { fontSize: 5, breakLine: true } },
    { text: "Conditioning", options: { bold: true, color: TIME, fontSize: 10, breakLine: true } },
    { text: "a 'valid period' (t_s, t_e) is injected as timecodes, so the same inputs can be summarised over any date range — interpolating or extrapolating past the observations", options: { fontSize: 8.5 } },
  ], { x: ox + 0.1, y: 2.0, w: 1.55, h: 3.5, fontFace: FONT_B, fontSize: 10, color: INK, valign: "top", isTextBox: true, margin: 0 });
  // bottom strip: training setup + what is / isn't stated
  s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x: 0.6, y: 6.05, w: 12.1, h: 0.85, fill: { color: "F4F6F9" }, line: { width: 0 }, rectRadius: 0.06 });
  s.addText([
    { text: "Training trio  ", options: { bold: true, color: TEAL } },
    { text: "teacher video embedder with implicit decoders · student with identical architecture (consistency + batch-uniformity on the sphere) · text-alignment model with a frozen language model (CLIP-style). ~1 B and ~480 M variants trained; the 480 M one is deployed.   ", options: {} },
    { text: "In the supplement (S2.4, S8.2):  ", options: { bold: true, color: "1F7A4D" } },
    { text: "15 STP blocks; widths D_S = 1024 (space), D_T = 512 (time), D_P = 128 (precision); implicit decoders = 2-hidden-layer MLPs of width 512; vMF bottleneck κ = 8·10³; inference on 960 m tiles buffered to 1.28 km, outer 80 m trimmed. So the spatial receptive field is at most 1.28 km — the whole frame.   ", options: {} },
    { text: "Still not stated:  ", options: { bold: true, color: "A23B3B" } },
    { text: "the exact pyramid up/down-sampling kernels and how the three states are merged.", options: {} },
  ], { x: 0.75, y: 6.1, w: 11.8, h: 0.75, fontFace: FONT_B, fontSize: 9.5, color: INK, valign: "middle", isTextBox: true, margin: 0 });
  footer(s);
  s.addNotes("STP = Space, Time, Precision. The three operators run on the same signal at different resolutions and are re-synchronised after each block by learned Laplacian-pyramid up/down-sampling, so information moves between coarse global attention and fine convolutional detail every block. The time operator is where the 'valid period' timestamps enter (sinusoidal timecodes). Relevance for us: it is a genuinely different way of buying spatial context — attention at 1/16 of the frame plus convolutions near full resolution — compared with our fixed 3×3 patch, and it is the mechanism behind 'continuous time' summaries.");
}


// 3c–3f. STP walk-through (4 build slides — click through as an animation)
// L = 128 px @ 10 m (1.28 km frame) is the paper's frame (suppl. S2.1/S8.2): 1/2 L = 64 px (20 m); 1/8 L = 16 px (80 m); 1/16 L = 8 px (160 m). 15 blocks in the paper; 4 drawn.
function gridLite(slide, x, y, size, n, o) {
  o = o || {};
  const cs = size / n;
  const layers = o.layers || 1;
  for (let l = layers - 1; l >= 1; l--) {
    slide.addShape(pres.shapes.RECTANGLE, { x: x + 0.07 * l, y: y - 0.07 * l, w: size, h: size, fill: { color: o.layerFill || PALE }, line: { color: GRIDLINE, width: 0.6 } });
  }
  slide.addShape(pres.shapes.RECTANGLE, { x, y, w: size, h: size, fill: { color: o.fill || WHITE }, line: { color: o.border || GRIDLINE, width: o.borderW || 0.75 } });
  const lw = n > 32 ? 0.25 : n > 12 ? 0.4 : 0.6;
  if (!o.noLines) for (let i = 1; i < n; i++) {
    slide.addShape(pres.shapes.LINE, { x: x + i * cs, y, w: 0, h: size, line: { color: o.lineColor || GRIDLINE, width: lw } });
    slide.addShape(pres.shapes.LINE, { x, y: y + i * cs, w: size, h: 0, line: { color: o.lineColor || GRIDLINE, width: lw } });
  }
  (o.cells || []).forEach(c => {
    slide.addShape(pres.shapes.RECTANGLE, { x: x + c.c * cs, y: y + c.r * cs, w: (c.k || 1) * cs, h: (c.k || 1) * cs, fill: { color: c.color, transparency: c.t == null ? 20 : c.t }, line: { color: c.line || c.color, width: c.lw == null ? 1 : c.lw, dashType: c.dash || "solid" } });
  });
  if (o.title) slide.addText(o.title, { x: x - 0.3, y: y - 0.34 - 0.07 * (layers - 1), w: size + 0.6, h: 0.28, fontFace: FONT_B, fontSize: 9.5, bold: true, color: o.titleColor || NAVY, align: "center", isTextBox: true, margin: 0 });
  if (o.sub) slide.addText(o.sub, { x: x - 0.4, y: y + size + 0.05, w: size + 0.8, h: o.subH || 0.5, fontFace: FONT_B, fontSize: 8.5, color: INK, align: "center", valign: "top", isTextBox: true, margin: 0 });
  return cs;
}
function stpStrip(slide, active) {
  const stages = [
    ["Inputs", "many sources, native resolutions, dated frames"],
    ["Input projectors", "→ ½ L feature maps (64 px @ 20 m)"],
    ["STP block", "space ⅟₁₆ L · time ⅛ L · precision ½ L"],
    ["Pyramid exchange", "every operator → every operator, × N"],
    ["Output head", "→ L (10 m), 64-D unit vector"],
  ];
  const x0 = 0.6, gap = 0.28, w = (12.1 - 4 * gap) / 5, y = 1.5, h = 0.62;
  stages.forEach(([t, d], i) => {
    const on = i === active;
    const x = x0 + i * (w + gap);
    slide.addShape(pres.shapes.ROUNDED_RECTANGLE, { x, y, w, h, fill: { color: on ? NAVY : "F4F6F9" }, line: { color: on ? NAVY : GRIDLINE, width: 0.75 }, rectRadius: 0.06 });
    slide.addText([{ text: t, options: { bold: true, fontSize: 10.5, color: on ? WHITE : NAVY, breakLine: true } }, { text: d, options: { fontSize: 8, color: on ? "CADCFC" : MUTED } }],
      { x: x + 0.1, y, w: w - 0.2, h, fontFace: FONT_B, valign: "middle", isTextBox: true, margin: 0 });
    if (i < 4) slide.addShape(pres.shapes.RIGHT_ARROW, { x: x + w + 0.04, y: y + h / 2 - 0.1, w: gap - 0.08, h: 0.2, fill: { color: GRIDLINE }, line: { width: 0 } });
  });
  slide.addText("L = side length of the square input frame in pixels. L = 128 px @ 10 m (1.28 km) is the paper's own frame (supplement S2.1, S8.2); the real depth is N = 15 blocks, we draw 4. Frame counts per year are typical values, not from the paper.", { x: 0.6, y: 6.75, w: 12.1, h: 0.28, fontFace: FONT_B, fontSize: 9, italic: true, color: MUTED, isTextBox: true, margin: 0 });
}
const STP_SUB = "STP walk-through";

// ---- step 1: inputs → projectors
{
  const s = pres.addSlide(); s.background = { color: WHITE };
  title(s, "STP, step 1 of 4 — input projectors bring every source to ½ L", "One frame of the same 1.28 km footprint from each sensor, at its own native resolution, becomes a 64 × 64 feature map at 20 m");
  stpStrip(s, 1);
  const gy = 3.0, gs = 1.55;
  gridLite(s, 0.8, gy, gs, 16, { layers: 4, title: "Sentinel-2 · 10 m", titleColor: SPACE, sub: "128 × 128 px (drawn 16 × 16)\n~70 dated frames / yr", cells: [{ r: 7, c: 7, color: OUTPX, t: 0 }] });
  gridLite(s, 3.0, gy, gs, 8, { layers: 3, title: "Landsat · 30 m", titleColor: SPACE, sub: "≈ 43 × 43 px (drawn 8 × 8)\n~20 frames / yr", cells: [{ r: 3, c: 3, color: OUTPX, t: 0 }] });
  gridLite(s, 5.2, gy, gs, 1, { layers: 2, title: "ERA5-Land · 9 km", titleColor: SPACE, sub: "the whole frame is one cell\ndaily → monthly", cells: [{ r: 0, c: 0, color: OUTPX, t: 85 }] });
  s.addText("… + S-1, PALSAR-2, GEDI, GRACE, DEM, land-cover, text", { x: 0.7, y: gy + gs + 0.75, w: 6.2, h: 0.3, fontFace: FONT_B, fontSize: 9, italic: true, color: MUTED, isTextBox: true, margin: 0 });
  // arrow
  s.addShape(pres.shapes.RIGHT_ARROW, { x: 7.05, y: gy + gs / 2 - 0.25, w: 0.9, h: 0.5, fill: { color: SPACE, transparency: 30 }, line: { width: 0 } });
  s.addText("learned projector per source\n(down-scale to ½ L)", { x: 6.75, y: gy + gs / 2 + 0.3, w: 1.5, h: 0.5, fontFace: FONT_B, fontSize: 8, color: SPACE, align: "center", isTextBox: true, margin: 0 });
  // output stack of 1/2 L maps
  const ox = 8.6, os = 2.2;
  gridLite(s, ox, gy - 0.2, os, 64, { layers: 5, title: "½ L feature maps · 64 × 64 @ 20 m", titleColor: NAVY, cells: [{ r: 31, c: 31, k: 2, color: OUTPX, t: 0 }] });
  s.addText([
    { text: "Σ Nᵢ frames in → Σ Nᵢ maps out.  ", options: { bold: true, color: NAVY } },
    { text: "Every frame keeps its acquisition timestamp as a sinusoidal timecode, so the block that follows knows when each map was observed. The yellow 10 m pixel is now a 20 m feature cell — the first (and only) fixed loss of resolution before the output head restores it.", options: {} },
  ], { x: ox - 0.2, y: gy + os - 0.05, w: 4.1, h: 1.3, fontFace: FONT_B, fontSize: 9.5, color: INK, valign: "top", isTextBox: true, margin: 0 });
  footer(s);
  s.addNotes("Step 1 of the STP walk-through. Frame: L = 128 px at 10 m — the paper's own training and inference frame (supplement S2.1, S8.2). Each source has its own learned input projector; whatever the native resolution, the output is a 1/2 L map (here 64×64 at 20 m) per input frame, with the timestamp carried alongside. Frame counts per year are typical values, not from the paper.");
}

// ---- step 2: one block, three operators
{
  const s = pres.addSlide(); s.background = { color: WHITE };
  title(s, "STP, step 2 of 4 — three operators, three resolutions", "Same 1.28 km footprint, three grids: what one cell can 'see' in each operator");
  stpStrip(s, 2);
  const gy = 2.9, gs = 2.05, xs = [0.9, 5.05, 9.2];
  // space 8x8: global attention
  const csA = gridLite(s, xs[0], gy, gs, 8, { title: "SPACE operator · ⅟₁₆ L = 8 × 8 cells of 160 m", titleColor: SPACE, cells: [{ r: 3, c: 3, color: SPACE, t: 0 }, { r: 0, c: 6, color: SPACE, t: 55 }, { r: 6, c: 1, color: SPACE, t: 55 }, { r: 7, c: 7, color: SPACE, t: 55 }, { r: 1, c: 1, color: SPACE, t: 55 }, { r: 5, c: 6, color: SPACE, t: 55 }],
    sub: "ViT-like self-attention: the cell attends to every other cell of the frame. Receptive field = whole frame, at coarse grain.", subH: 0.7 });
  [[0, 6], [6, 1], [7, 7], [1, 1], [5, 6]].forEach(([r, c]) => {
    const x1 = xs[0] + 3.5 * csA, y1 = gy + 3.5 * csA, x2 = xs[0] + (c + 0.5) * csA, y2 = gy + (r + 0.5) * csA;
    s.addShape(pres.shapes.LINE, { x: Math.min(x1, x2), y: Math.min(y1, y2), w: Math.abs(x2 - x1), h: Math.abs(y2 - y1), flipH: (x2 < x1) !== (y2 < y1), line: { color: SPACE, width: 1, dashType: "dash" } });
  });
  // time 16x16 with stacked frames: axial attention
  const csB = gridLite(s, xs[1], gy, gs, 16, { layers: 4, layerFill: "FCEFE4", title: "TIME operator · ⅛ L = 16 × 16 cells of 80 m", titleColor: TIME, cells: [{ r: 7, c: 7, color: TIME, t: 0 }],
    sub: "Time-axial self-attention: the cell attends only to itself across all dated frames (stacked sheets), each tagged with its timecode.", subH: 0.7 });
  [1, 2, 3].forEach(l => s.addShape(pres.shapes.RECTANGLE, { x: xs[1] + 7 * csB + 0.07 * l, y: gy + 7 * csB - 0.07 * l, w: csB, h: csB, fill: { color: TIME, transparency: 50 }, line: { color: TIME, width: 0.75 } }));
  s.addShape(pres.shapes.LINE, { x: xs[1] + 7.5 * csB, y: gy + 7.5 * csB - 0.21, w: 0.21, h: 0.21, flipH: true, line: { color: TIME, width: 1.5, endArrowType: "triangle", beginArrowType: "triangle" } });
  // precision 64x64: 3x3 conv
  gridLite(s, xs[2], gy, gs, 64, { title: "PRECISION operator · ½ L = 64 × 64 cells of 20 m", titleColor: "1F7A4D", cells: [{ r: 30, c: 30, k: 3, color: "1F7A4D", t: 55 }, { r: 31, c: 31, color: "1F7A4D", t: 0 }],
    sub: "3 × 3 convolutions: the cell sees only its immediate neighbours (60 m), but at the finest grain the block carries.", subH: 0.7 });
  // magnifier for precision
  s.addShape(pres.shapes.OVAL, { x: xs[2] + gs + 0.15, y: gy + gs / 2 - 0.45, w: 0.9, h: 0.9, fill: { color: WHITE }, line: { color: "1F7A4D", width: 1 } });
  s.addShape(pres.shapes.LINE, { x: xs[2] + gs * 0.5 + 0.05, y: gy + gs * 0.49, w: gs * 0.5 + 0.1, h: 0, line: { color: "1F7A4D", width: 0.75, dashType: "dash" } });
  const mx = xs[2] + gs + 0.15 + 0.15, my = gy + gs / 2 - 0.45 + 0.15, ms = 0.6 / 3;
  for (let r = 0; r < 3; r++) for (let c = 0; c < 3; c++) s.addShape(pres.shapes.RECTANGLE, { x: mx + c * ms, y: my + r * ms, w: ms, h: ms, fill: { color: "1F7A4D", transparency: r === 1 && c === 1 ? 0 : 60 }, line: { color: WHITE, width: 0.5 } });
  s.addText([{ text: "Three views of one place, computed simultaneously. ", options: { bold: true, color: NAVY } }, { text: "Space knows the whole frame coarsely, time knows the whole year at one spot, precision knows the fine local texture — none of them alone is the stencil; the exchange in step 3 is.", options: {} }],
    { x: 0.6, y: 6.05, w: 12.1, h: 0.6, fontFace: FONT_B, fontSize: 10, color: INK, valign: "top", isTextBox: true, margin: 0 });
  footer(s);
  s.addNotes("Step 2. The three operators of one STP block. Ratios (1/16, 1/8, 1/2 of L) and L = 128 px are from the paper (main text + supplement S2.1), so the cell sizes 160 / 80 / 20 m are real, not illustrative. Attention in the space operator is over the whole down-scaled frame; time attention is axial (one location, all frames); precision is a 3×3 convolution at 1/2 L.");
}

// ---- step 3: pyramid exchange
{
  const s = pres.addSlide(); s.background = { color: WHITE };
  title(s, "STP, step 3 of 4 — the pyramid exchange between blocks", "End of block k: each of the three maps is resampled to all three resolutions and summed into block k+1's inputs — repeat for N blocks");
  stpStrip(s, 3);
  const gy = [2.85, 4.1, 5.35], gs = 1.05, xl = 1.6, xr = 9.6;
  const cols = [SPACE, TIME, "1F7A4D"], ns = [8, 16, 64], names = ["space ⅟₁₆ L", "time ⅛ L", "precision ½ L"];
  // left: block k outputs, one highlighted feature each
  const lhl = [[{ r: 3, c: 3, color: SPACE, t: 0 }], [{ r: 6, c: 9, color: TIME, t: 0 }], [{ r: 40, c: 20, k: 3, color: "1F7A4D", t: 0 }]];
  ns.forEach((n, i) => gridLite(s, xl, gy[i] - gs / 2 + 0.05, gs, n, { title: i === 0 ? "block k · outputs" : undefined, cells: lhl[i], border: cols[i], borderW: 1.25 }));
  ns.forEach((n, i) => s.addText(names[i], { x: 0.3, y: gy[i] - 0.15, w: 1.25, h: 0.3, fontFace: FONT_B, fontSize: 9, bold: true, color: cols[i], align: "right", isTextBox: true, margin: 0 }));
  // right: block k+1 inputs, each carrying all three features, resampled
  const rhl = [
    [{ r: 3, c: 3, color: SPACE, t: 0 }, { r: 3, c: 4, color: TIME, t: 35 }, { r: 5, c: 2, color: "1F7A4D", t: 35 }],                       // 8x8: time cell (6,9)/2 -> (3,4); precision (40,20)/8 -> (5,2)
    [{ r: 6, c: 6, k: 2, color: SPACE, t: 45 }, { r: 6, c: 9, color: TIME, t: 0 }, { r: 10, c: 5, color: "1F7A4D", t: 35 }],               // 16x16: space (3,3)*2 -> (6,6) 2x2; precision (40,20)/4 -> (10,5)
    [{ r: 24, c: 24, k: 8, color: SPACE, t: 65 }, { r: 24, c: 36, k: 4, color: TIME, t: 45 }, { r: 40, c: 20, k: 3, color: "1F7A4D", t: 0 }], // 64x64: space (3,3)*8 -> (24,24) 8x8; time (6,9)*4 -> (24,36) 4x4
  ];
  ns.forEach((n, i) => gridLite(s, xr, gy[i] - gs / 2 + 0.05, gs, n, { title: i === 0 ? "block k+1 · inputs" : undefined, cells: rhl[i], border: cols[i], borderW: 1.25 }));
  // 9 arrows through a resampling hub
  const hx = 6.2, hw = 1.5;
  s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x: hx, y: 2.6, w: hw, h: 3.3, fill: { color: "F4F6F9" }, line: { color: GRIDLINE, width: 0.75 }, rectRadius: 0.08 });
  s.addText([{ text: "learned Laplacian-pyramid rescaling", options: { bold: true, breakLine: true, color: NAVY } }, { text: "↑ up-sample coarse → fine\n↓ down-sample fine → coarse\n= keep (same level)\nthen sum per level", options: { fontSize: 8.5 } }],
    { x: hx + 0.08, y: 2.65, w: hw - 0.16, h: 3.2, fontFace: FONT_B, fontSize: 9.5, color: INK, align: "center", valign: "middle", isTextBox: true, margin: 0 });
  ns.forEach((_, i) => ns.forEach((_, j) => {
    const y1 = gy[i], y2 = gy[j];
    s.addShape(pres.shapes.LINE, { x: xl + gs + 0.05, y: Math.min(y1, y2), w: hx - (xl + gs) - 0.1, h: Math.abs(y2 - y1), flipV: y2 < y1, line: { color: cols[i], width: 1, transparency: 30 } });
    s.addShape(pres.shapes.LINE, { x: hx + hw + 0.05, y: Math.min(y1, y2), w: xr - (hx + hw) - 0.1, h: Math.abs(y2 - y1), flipV: y2 < y1, line: { color: cols[i], width: 1, transparency: 30, endArrowType: "triangle" } });
  }));
  // legend of the highlighted features' journey
  s.addText([
    { text: "Follow the colours. ", options: { bold: true, color: NAVY } },
    { text: "The blue 160 m space cell arrives in the fine map as an 8 × 8 block of 20 m cells; the green 3 × 3 precision patch (60 m) arrives in the coarse map as a fraction of one 160 m cell; the orange time cell lands as 2 × 2 / 4 × 4. After the sum, every operator in block k+1 starts from all three views. Repeat N times — the receptive field of the fine path grows to the whole frame while it keeps 20 m grain.", options: {} },
  ], { x: 0.6, y: 6.12, w: 12.1, h: 0.62, fontFace: FONT_B, fontSize: 9.5, color: INK, valign: "top", isTextBox: true, margin: 0 });
  s.addText("× 15 blocks (we draw 4)", { x: xr + gs + 0.1, y: 5.35 - 0.12, w: 1.6, h: 0.25, fontFace: FONT_B, fontSize: 9, bold: true, color: MUTED, align: "center", isTextBox: true, margin: 0 });
  footer(s);
  s.addNotes("Step 3. This is the sentence being animated: 'after each block, learned Laplacian-pyramid rescaling lets every operator feed every operator of the next block.' Nine paths (3 sources × 3 destinations); same-level paths are identity-like, cross-level paths are learned up/down-sampling. The sum per level is our reading of how the states are combined — the paper says 'pass its state to each of the operators in the subsequent block' without specifying the merge.");
}

// ---- step 4: output head
{
  const s = pres.addSlide(); s.background = { color: WHITE };
  title(s, "STP, step 4 of 4 — back to 10 m, 64-D per pixel", "After block N: the ½ L maps are resampled to the precision resolution and then to L; each output pixel is a unit vector on S⁶³");
  stpStrip(s, 4);
  const gy = 2.85, gsmall = 1.05;
  const cols = [SPACE, TIME, "1F7A4D"], ns = [8, 16, 64];
  ns.forEach((n, i) => gridLite(s, 0.8, 2.45 + i * 1.3, gsmall, n, { border: cols[i], borderW: 1.25, title: i === 0 ? "block N · outputs" : undefined, cells: [{ r: Math.floor(n / 2), c: Math.floor(n / 2), color: cols[i], t: 0 }] }));
  // merge arrows
  [0, 1, 2].forEach(i => s.addShape(pres.shapes.LINE, { x: 0.8 + gsmall + 0.05, y: Math.min(2.45 + i * 1.3 + gsmall / 2, gy + 1.3), w: 0.9, h: Math.abs(2.45 + i * 1.3 + gsmall / 2 - (gy + 1.3)), flipV: (2.45 + i * 1.3 + gsmall / 2) > (gy + 1.3), line: { color: cols[i], width: 1.25, endArrowType: "triangle" } }));
  s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x: 2.85, y: gy + 0.55, w: 1.45, h: 1.5, fill: { color: "F4F6F9" }, line: { color: GRIDLINE, width: 0.75 }, rectRadius: 0.08 });
  s.addText([{ text: "final learned spatial resampling", options: { bold: true, breakLine: true, color: NAVY } }, { text: "½ L → L\n64 px → 128 px\n20 m → 10 m", options: { fontSize: 9 } }], { x: 2.9, y: gy + 0.6, w: 1.35, h: 1.4, fontFace: FONT_B, fontSize: 9.5, color: INK, align: "center", valign: "middle", isTextBox: true, margin: 0 });
  s.addShape(pres.shapes.RIGHT_ARROW, { x: 4.4, y: gy + 1.1, w: 0.5, h: 0.4, fill: { color: GRIDLINE }, line: { width: 0 } });
  // output frame L: 128x128 drawn 32x32, receptive field tint
  const ox = 5.1, os = 2.6;
  gridLite(s, ox, gy - 0.1, os, 32, { title: "output frame · L = 128 × 128 px @ 10 m (drawn 32 × 32)", titleColor: NAVY, fill: "EAF2F8",
    cells: [{ r: 12, c: 12, k: 8, color: SPACE, t: 80, line: SPACE, dash: "dash" }, { r: 14, c: 14, k: 4, color: TIME, t: 65, line: TIME, dash: "dash" }, { r: 15, c: 15, k: 2, color: "1F7A4D", t: 45 }, { r: 15, c: 15, color: OUTPX, t: 0, line: "B58900" }],
    sub: "Tinted background = whole frame reaches the pixel through the space path; dashed squares = coarse cells it inherited; green = fine 3×3 texture; yellow = the 10 m output pixel.", subH: 0.75 });
  // vector
  const ex = 8.75, ey = gy + 0.15, ew = 0.6, eh = 2.1, cells = 8, ch = eh / cells;
  s.addShape(pres.shapes.RIGHT_ARROW, { x: ox + os + 0.15, y: gy + 1.1, w: 0.7, h: 0.4, fill: { color: GRIDLINE }, line: { width: 0 } });
  for (let i = 0; i < cells; i++) s.addShape(pres.shapes.RECTANGLE, { x: ex, y: ey + i * ch, w: ew, h: ch, fill: { color: ["6A4C93", "7B5EA7", "8C70BA", "9D82CD", "8C70BA", "7B5EA7", "6A4C93", "5A3D80"][i] }, line: { color: WHITE, width: 0.75 } });
  s.addText("64-D", { x: ex - 0.3, y: ey + eh + 0.05, w: ew + 0.6, h: 0.3, fontFace: FONT_B, fontSize: 12, bold: true, color: EMB, align: "center", isTextBox: true, margin: 0 });
  s.addText([
    { text: "Per 10 m pixel: ", options: { bold: true, color: EMB } }, { text: "a 64-D vector, normalised to the unit sphere S⁶³ (the batch-uniformity loss spreads embeddings over it), stored as 64 int8 bytes in the released layers.", options: { breakLine: true } },
    { text: " ", options: { fontSize: 5, breakLine: true } },
    { text: "Per valid period: ", options: { bold: true, color: TIME } }, { text: "the pair (t_s, t_e) entered the time operators as timecodes in every block, so this whole frame of vectors is a summary of that window — one calendar year in the public product.", options: { breakLine: true } },
    { text: " ", options: { fontSize: 5, breakLine: true } },
    { text: "So what is the stencil? ", options: { bold: true, color: NAVY } }, { text: "Spatially the whole 1.28 km frame, coarsely (8 × 8 cells of 160 m through the space path) plus a fine local neighbourhood (20 m cells through the precision path); temporally every dated frame in the valid period. The number to remember: an AlphaEarth pixel never sees further than ~640 m in any direction.", options: {} },
  ], { x: 9.7, y: gy - 0.1, w: 3.0, h: 3.9, fontFace: FONT_B, fontSize: 9.5, color: INK, valign: "top", isTextBox: true, margin: 0 });
  footer(s);
  s.addNotes("Step 4. The paper: 'STP itself terminates with a final learned spatial resampling to the resolution of the precision operator', and embeddings are then produced at 10 m. We draw the up-sampling to L explicitly. The receptive-field picture on the output frame is the consequence of steps 2–3: coarse-global through the space path, fine-local through precision, all frames in the window through time.");
}

// 4. TESSERA
modelSlide(4, "TESSERA (University of Cambridge)", "Pixel-wise 10 m annual embeddings from S-1 + S-2 time series — fully open (CC0), CVPR 2026",
  [
    ["Who / when", "Cambridge Centre for Earth Observation; arXiv 2506.20380 (Jun 2025), CVPR 2026. Code github.com/ucam-eo/tessera, access via geotessera."],
    ["Inputs", "Sentinel-2 L2A (10 bands, 60 m bands dropped, resampled to 10 m) + Sentinel-1 RTC (VV, VH). Two modality-specific branches, fused late."],
    ["Architecture", "Per-pixel temporal transformer (4 layers) + GRU pooling per modality; encoder ~46 M params (1.34 B projector used only in training). Barlow Twins self-supervision on ~800 M pixel-series from 3,012 MGRS tiles; ~6,200 GPU-h on MI300X."],
    ["Stencil", "Strictly one 10 m pixel — no spatial context at all. One calendar year; 40 valid dates sampled per modality with learnable day-of-year embeddings (irregular cadence tolerated)."],
    ["Output", "128-D per pixel-year, int8 (QAT). Years 2017–2025, global land (water masked); tile-sampled coverage, gaps on request."],
    ["Availability", "Weights (CC0), code (MIT), precomputed embeddings via GeoTessera CDN. Self-run: ~10 h per 110 km tile-year (128-core CPU + A30)."],
    ["Results", "Frozen + MLP head: PASTIS-R mIoU 50.7, TreeSatAI-TS F1 78.0, canopy height RMSE 12.2 m; holds up until <~20 valid S-2 obs/yr."],
  ],
  {
    space: { n: 9, outPx: [4, 4], layers: 2, notes: [{ text: "no neighbours used", x: 0.9, y: 0.95, w: 1.4, h: 0.25, color: "B58900", align: "center" }],
      lines: ["Output pixel: 10 m × 10 m.", "Receptive field = the same single pixel. Neighbouring pixels never enter.", "Two modality sheets: S-2 (10 bands) and S-1 (2 bands)."] },
    time: { window: [0.1, 0.9], windowLabel: "1 calendar year", axisTicks: seq(12, 0.1, 0.9),
      rows: [
        { label: "S-2: 40 sampled valid dates", ticks: jitter(40, 0.1, 0.9, 7), size: 0.05 },
        { label: "S-1: 40 sampled valid dates", ticks: jitter(40, 0.1, 0.9, 13), size: 0.05, color: "B85C1E" },
      ],
      lines: ["Fixed window; dates irregular, encoded by day-of-year.", "Sampling with replacement pads pixel-years with <40 clear observations."] },
    out: { dim: "128-D", lines: ["per 10 m pixel per year", "int8"] }
  },
  "TESSERA is the purest 'temporal embedding' design: zero spatial context by construction. That is the opposite extreme from our patch codec, and a useful control idea — how much of our skill comes from the 3×3 neighbourhood versus the per-pixel history?"
);

// 5. OlmoEarth
modelSlide(5, "OlmoEarth (Ai2)", "Open multimodal space-time ViT family (Nano→Large); embeddings computed on demand, not precomputed",
  [
    ["Who / when", "Allen Institute for AI (+ UW, ASU, UBC). v1 Nov 2025 (arXiv 2511.13655), v1.1 May 2026, v1.2 later 2026. Weights on Hugging Face; custom 'OlmoEarth Artifact License' (no military/extractive use)."],
    ["Inputs", "Sentinel-1, Sentinel-2, Landsat-8 monthly time series, all resampled to 10 m; static layers (OSM, WorldCover, CDL, SRTM, canopy height, WorldCereal) as inputs/targets."],
    ["Architecture", "ViT over space × time × band-group tokens (FlexiViT patches 1–8 px in v1; single-bandset token merging in v1.1, RoPE in v1.2). 'Latent MIM Lite': masked modelling with patch-discrimination + instance-contrastive losses. Sizes: Nano 1.4 M · Tiny 6.2 M · Base 89 M · Large 308 M (encoder)."],
    ["Stencil", "Tile 2.56 km × 2.56 km = 256 × 256 px @ 10 m; token = 1–8 px patch (≤ 80 m). Up to 12 monthly steps over one year, missing months masked."],
    ["Output", "Per-patch, per-timestep tokens (Base 768-D, Large 1024-D) + pooled vector. No standing global product; Studio exports embeddings for any AOI at 10–80 m, 1–12 monthly steps (int8 COGs)."],
    ["Results", "Best on 15/24 tasks (kNN/linear) and 19/29 (fine-tuned) vs. Prithvi v2, TerraMind, Galileo, DINOv3; matches/exceeds AlphaEarth after fine-tuning on 5 tasks."],
    ["Pretraining data", "285 k samples × 2.56 km × 1 yr, stratified over 120 OSM classes (~10 TB) — land/infrastructure-biased."],
  ],
  {
    space: { n: 16, tileOutline: true, token: [7, 7, 1], outPx: null, layers: 3, notes: [{ text: "tile 2.56 km", x: 0.05, y: 0.02, w: 1.2 }, { text: "token 8 px = 80 m", x: 1.55, y: 1.15, w: 1.2, h: 0.25 }],
      lines: ["Tile: 256 × 256 px @ 10 m = 2.56 km (drawn 16 × 16 for legibility).", "Token: flexible 1–8 px patch → 10–80 m per token, full attention across the tile.", "Sheets: S-1, S-2, Landsat + static maps."] },
    time: { window: [0.1, 0.9], windowLabel: "1 year window, ≤12 monthly steps", axisTicks: seq(12, 0.1, 0.9),
      rows: [
        { label: "monthly composites, each its own tokens", ticks: seq(12, 0.1, 0.9), style: "bar" },
        { label: "static map layers (single sheet)", ticks: [0.5], size: 0.09, color: "B85C1E" },
      ],
      lines: ["Each month is tokenised separately; attention runs across space, time and modality.", "Temporal masking (v1.2) trains the model to fill missing months."] },
    out: { dim: "768-D", lines: ["per patch per month (Base)", "+ pooled tile vector"] }
  },
  "OlmoEarth is the closest analogue to a general space-time ViT we could re-train: open weights, monthly tokens, masked latent objective. Embeddings are on demand via OlmoEarth Studio for a polygon — convenient but no global archive."
);

// 6. TerraMind
modelSlide(6, "TerraMind (IBM Research + ESA Φ-lab)", "Any-to-any generative multimodal EO model; 'Thinking in Modalities' — ICCV 2025",
  [
    ["Who / when", "IBM Research Europe, ESA Φ-lab, ETH Zürich, FZ Jülich, NASA IMPACT, Univ. of Iceland. arXiv 2504.11171 (Apr 2025); TerraMesh dataset 2504.11172. Apache 2.0, weights on Hugging Face (ibm-esa-geospatial), TerraTorch integration. Tiny/Small edge variants Jun–Jul 2025."],
    ["Inputs", "Pixel-level: S-2 L2A & L1C, S-1 GRD & RTC, DEM, RGB, NDVI. Token-level: LULC classes, captions, geolocation. All co-registered at 10 m in 224 px tiles."],
    ["Architecture", "Dual-scale encoder–decoder: token path (cross-modal correlation) + pixel path (fine detail). Per-modality FSQ-VAE tokenizers (16 k codebook). Objective: masked cross-modal token reconstruction over 500 B tokens from ~9 M TerraMesh samples. Base: 6 d, Large: 10 d on 32 A100s."],
    ["Stencil", "Tile 224 × 224 px @ 10 m = 2.24 km; token 16 × 16 px = 160 m → 14 × 14 = 196 tokens per modality. Single co-registered timestamp per sample — no multi-temporal fusion."],
    ["Output", "196 tokens per modality of 768-D (Base) / 1024-D (Large), mergeable to one vector; plus generated rasters of missing modalities (TiM). No precomputed embedding archive."],
    ["Results", "PANGAEA avg mIoU: Large 59.6, Base 58.4 (≥3 pp over CROMA, DOFA and the other GeoFMs in that table); only GeoFM beating task-specific U-Nets across the benchmark."],
    ["Ocean", "TerraMesh is land/ecoregion-stratified; no marine modalities. Open-ocean behaviour untested."],
  ],
  {
    space: { n: 14, tileOutline: true, token: [6, 6, 1], layers: 4, notes: [{ text: "224 px = 2.24 km", x: 0.05, y: 0.02, w: 1.3 }, { text: "token 16 px = 160 m", x: 1.45, y: 1.0, w: 1.3, h: 0.25 }],
      lines: ["Tile: 224 × 224 px @ 10 m = 2.24 km × 2.24 km.", "Token: 16 × 16 px = 160 m; 14 × 14 grid per modality.", "Up to ~9 modality sheets, each with its own tokenizer."] },
    time: { snapshot: 0.5, snapshotLabel: "one co-registered timestamp", axisTicks: seq(12), rows: [],
      lines: ["All modalities come from the same date and place.", "A sample is a snapshot: every modality is aligned to one acquisition.", "Temporal change is not part of the pretraining stencil."] },
    out: { dim: "768-D", lines: ["per 160 m token, per modality (Base; Large 1024-D)", "or merged per tile"] }
  },
  "TerraMind's distinguishing trick is generative: it can synthesise a missing modality (e.g. LULC from S-2) and feed it back in. Ocean translation: generate missing altimetry from SST+wind? But it is single-snapshot, so dynamics are absent."
);

// 7. Prithvi-EO-2.0
modelSlide(7, "Prithvi-EO 2.0 (IBM + NASA)", "Multi-temporal HLS masked autoencoder at 30 m; 300 M and 600 M, with temporal-location (TL) variants",
  [
    ["Who / when", "IBM Research + NASA IMPACT, trained at JSC; arXiv 2412.02732 (Dec 2024, v3 Mar 2026). Weights on Hugging Face (ibm-nasa-geospatial); fine-tuning via TerraTorch. Succeeds Prithvi-EO 1.0 (100 M)."],
    ["Inputs", "HLS v2 (Harmonized Landsat–Sentinel-2) at 30 m, 6 bands (B02–B07: blue, green, red, NIR-narrow, SWIR1, SWIR2). 4.2 M training samples, 2014–2023; sea-only tiles removed with Fmask, Greenland excluded."],
    ["Architecture", "ViT-L (300 M: 1024-D, 24 layers) / ViT-H (600 M: 1280-D, 32 layers) masked autoencoder, mask ratio 0.75, 3D sin-cos positions. TL variants add lat/lon + date encodings with metadata dropout."],
    ["Stencil", "Tile 224 × 224 px @ 30 m = 6.72 km. Token 16 px = 480 m (300 M) or 14 px = 420 m (600 M); tubelet depth 1, so each of 4 timestamps gets its own token grid. Frames 1–6 months apart (seasonal), not a fixed window; single-frame inference supported."],
    ["Output", "4 × 196 (or 4 × 256) tokens of 1024/1280-D. No precomputed embedding product; typical use is full fine-tuning of encoder + light decoder."],
    ["Results", "GEO-Bench: 600 M-TL best on average, ~8 % over v1.0 and ahead of six other GFMs incl. DOFA and Scale-MAE (per-set numbers are charts); tasks: floods, burn scars, crops, landslides, PASTIS, Sen4Map."],
    ["In orbit", "May 2026: deployed on Kanyini satellite and ISS IMAGIN-e for onboard flood/cloud detection (still the 2.0 architecture)."],
  ],
  {
    space: { n: 14, tileOutline: true, token: [6, 6, 1], layers: 1, notes: [{ text: "224 px = 6.72 km", x: 0.05, y: 0.02, w: 1.3 }, { text: "token 16 px = 480 m", x: 1.45, y: 1.0, w: 1.3, h: 0.25 }],
      lines: ["Tile: 224 × 224 px @ 30 m = 6.72 km × 6.72 km.", "Token: 16 × 16 px = 480 m (300 M) · 14 × 14 px = 420 m (600 M).", "One sheet: 6 HLS bands."] },
    time: { window: [0.08, 0.92], windowLabel: "4 dates, 1–6 months apart (span varies)", dashedWindow: true, axisTicks: seq(12, 0.08, 0.92),
      rows: [{ label: "4 acquisitions, each its own token grid", ticks: [0.14, 0.38, 0.55, 0.86], style: "bar" }],
      lines: ["Time is a stack of 4 snapshots with 3D position codes — not a period summary.", "TL variants inject acquisition date + lat/lon as extra encodings."] },
    out: { dim: "1024-D", lines: ["per 480 m token, per date", "300 M: 1024-D · 600 M: 1280-D"] }
  },
  "Prithvi-EO 2.0 is the land-surface backbone the IBM ocean model copies. Note the tubelet depth of 1: the four dates are separate token grids, so temporal fusion happens only inside attention — same design choice as our stage-1 codec treating each month independently."
);

// 8. Granite-Geospatial-Ocean
modelSlide(8, "Granite-Geospatial-Ocean (IBM · STFC · PML · Exeter)", "The Prithvi-EO recipe moved onto Sentinel-3 ocean colour + SST — github.com/ibm-granite/geospatial",
  [
    ["Who / when", "IBM Research Europe (UK), STFC Hartree Centre, Plymouth Marine Lab, Univ. of Exeter (UK HNCDI). Released 6 Oct 2025; arXiv 2509.21273 'A Sentinel-3 foundation model for ocean colour'. Apache 2.0; weights on Hugging Face, code + TerraTorch notebook at github.com/ibm-granite/geospatial."],
    ["Inputs", "Sentinel-3 OLCI Level-2 reflectance, 16 bands (Oa01–Oa12, Oa16–18, Oa21) + SLSTR sea-surface temperature = 17 channels at 300 m. ~512 k tiles (the paper also gives 470 k train + 50 k val), 2017–2021, balanced over 83 Longhurst provinces and months."],
    ["Architecture", "Prithvi-EO ViT-MAE with the tile shrunk for the coarser sensor: embed 512-D, 12 layers, 8 heads, decoder 256-D × 4; ~50 M params (larger gave no gain). RMSE reconstruction loss on masked patches."],
    ["Stencil", "Tile 42 × 42 px @ 300 m = 12.6 km × 12.6 km; token 2 × 2 px = 600 m → 21 × 21 = 441 tokens. num_frames = 1: a single acquisition, no temporal stacking (fine-tuning pairs each in-situ sample with cloud-free imagery from a 6-day window centred on it)."],
    ["Output", "441 × 512-D tokens per tile; downstream pixel-wise regression heads (TerraTorch) for chlorophyll-a and primary production."],
    ["Results", "Chl-a (188 in-situ points): RMSE 0.14 ± 0.05 vs 0.16 ± 0.10 log₁₀ mg m⁻³ from scratch; primary production (103 points): 0.39 ± 0.04 vs 0.42 ± 0.07 log₁₀ mgC m⁻² d⁻¹. Gains grow as labels shrink (useful at 25 % of labels)."],
    ["Caveats", "Surface-only, cloud-limited optical; single snapshot; authors say not yet competitive with operational algorithms; under-predicts bloom-level chl-a."],
  ],
  {
    space: { n: 21, tileOutline: true, token: [10, 10, 1], layers: 2, notes: [{ text: "42 px = 12.6 km", x: 0.05, y: 0.02, w: 1.3 }, { text: "token 2 px = 600 m", x: 1.45, y: 1.55, w: 1.3, h: 0.25 }],
      lines: ["Tile: 42 × 42 px @ 300 m = 12.6 km × 12.6 km (21 × 21 tokens drawn to scale).", "Token: 2 × 2 px = 600 m.", "Two sheets: 16 OLCI bands + 1 SST band."] },
    time: { snapshot: 0.5, snapshotLabel: "one Sentinel-3 pass", axisTicks: seq(12), rows: [],
      lines: ["num_frames = 1 — no temporal stacking.", "Single-timestep stencil: the 4-frame Prithvi option is switched off.", "Fine-tuning pairs each in-situ measurement with imagery from a 6-day window centred on it."] },
    out: { dim: "512-D", lines: ["per 600 m token", "per single acquisition"] }
  },
  "This is the one model of the six trained on sea pixels — but it is an ocean-colour (biology) model, not an ocean-state model: optical surface only, one snapshot, 12.6 km tiles. Nothing here sees the interior, altimetry or transport. Repo: github.com/ibm-granite/geospatial → granite-geospatial-ocean (config confirms 42 px, 17 channels, embed 512, depth 12, patch [1,2,2]).",
  10.5
);

// =====================================================================
// 9. Stencils side by side (space × time map)
{
  const s = pres.addSlide();
  s.background = { color: WHITE };
  title(s, "Stencils side by side", "Spatial footprint that feeds one embedding (log scale) vs. how much time it summarises");
  const px = 1.3, py = 1.7, pw = 10.4, ph = 4.2;
  // plot frame
  s.addShape(pres.shapes.RECTANGLE, { x: px, y: py, w: pw, h: ph, fill: { color: "FAFBFD" }, line: { color: GRIDLINE, width: 0.75 } });
  // x axis: log10 metres from 1 (10^0) to 6 (10^6)
  const xmin = 0.7, xmax = 6.0;
  const X = m => px + (Math.log10(m) - xmin) / (xmax - xmin) * pw;
  [[10, "10 m"], [100, "100 m"], [1000, "1 km"], [10000, "10 km"], [100000, "100 km"], [1000000, "1000 km"]].forEach(([m, lab]) => {
    s.addShape(pres.shapes.LINE, { x: X(m), y: py, w: 0, h: ph, line: { color: GRIDLINE, width: 0.5, dashType: "dash" } });
    s.addText(lab, { x: X(m) - 0.5, y: py + ph + 0.05, w: 1.0, h: 0.25, fontFace: FONT_B, fontSize: 9.5, color: MUTED, align: "center", isTextBox: true, margin: 0 });
  });
  s.addText("Spatial extent of the input that shapes one embedding (side length)", { x: px, y: py + ph + 0.32, w: pw, h: 0.3, fontFace: FONT_B, fontSize: 10.5, bold: true, color: SPACE, align: "center", isTextBox: true, margin: 0 });
  // y rows (categorical)
  const rows = ["single snapshot", "a few dates (4)", "12 monthly steps", "1 year folded (12–40 dates)", "1 year, all sources (arbitrary window)"];
  const Y = i => py + ph - (i + 0.5) * ph / rows.length;
  rows.forEach((r, i) => {
    s.addShape(pres.shapes.LINE, { x: px, y: Y(i), w: pw, h: 0, line: { color: GRIDLINE, width: 0.5, dashType: "dash" } });
    s.addText(r, { x: 0.15, y: Y(i) - 0.3, w: px - 0.25, h: 0.6, fontFace: FONT_B, fontSize: 9.5, color: TIME, align: "right", valign: "middle", isTextBox: true, margin: 0 });
  });
  s.addText("Temporal window per embedding", { x: 0.15, y: py - 0.35, w: 3, h: 0.3, fontFace: FONT_B, fontSize: 10.5, bold: true, color: TIME, isTextBox: true, margin: 0 });
  // points: [metres, rowIndex, label, colour, sublabel]
  const pts = [
    [10, 3, "TESSERA", SPACE, "10 m px · 128-D"],
    [1280, 4, "AlphaEarth", SPACE, "1.28 km frame · 10 m px · 64-D"],
    [2240, 0, "TerraMind", SPACE, "2.24 km tile · 160 m token · 768-D"],
    [2560, 2, "OlmoEarth", SPACE, "2.56 km tile · ≤80 m token · 768-D"],
    [6720, 1, "Prithvi-EO 2.0", SPACE, "6.72 km tile · 480 m token · 1024-D"],
    [12600, 0, "Granite-Geo-Ocean", "1C7293", "12.6 km tile · 600 m token · 512-D"],
    [84000, 0, "Earth 2 codec (ours)", EMB, "3×3 nbhd @ ¼° ≈ 84 km · monthly/pentad mean · 64-D"],
    [28000, 3, "proposed pixel-year arm", EMB, "1 px @ ¼° ≈ 28 km · 12 months → one z (last slide)"],
  ];
  pts.forEach(([m, row, lab, col, sub], i) => {
    const cx = X(m), cy = Y(row);
    const d = 0.26;
    const hollow = lab.startsWith("proposed");
    s.addShape(pres.shapes.OVAL, { x: cx - d / 2, y: cy - d / 2, w: d, h: d, fill: hollow ? { color: WHITE } : { color: col }, line: { color: hollow ? col : WHITE, width: 1.5, dashType: hollow ? "dash" : "solid" } });
    let lx = cx + 0.2, ly = cy - 0.42, align = "left", lw = 3.2;
    if (lab === "AlphaEarth") { ly = cy - 0.42; }
    if (lab === "Granite-Geo-Ocean") { ly = cy + 0.1; lx = cx - 0.6; }
    if (lab === "TerraMind") { ly = cy + 0.05; lx = cx - 1.6; align = "right"; lw = 3.0; lx = cx - lw - 0.2; }
    if (lab === "Earth 2 codec (ours)") { ly = cy - 0.7; lx = cx - 1.2; lw = 3.4; }
    if (lab === "Prithvi-EO 2.0") { ly = cy - 0.55; }
    s.addText([{ text: lab, options: { bold: true, color: col, breakLine: true } }, { text: sub, options: { color: INK, fontSize: 8.5 } }],
      { x: lx, y: ly, w: lw, h: 0.5, fontFace: FONT_B, fontSize: 10.5, align, isTextBox: true, margin: 0, valign: "top" });
  });
  s.addText("Earth 2 shown for reference: the codec's channel token sees a 3 × 3 neighbourhood of ¼° (or 1°) cells at one monthly or 5-day mean — a snapshot stencil ~7× wider than the widest EO tile here (Granite, 12.6 km).", { x: 0.6, y: 6.7, w: 12.1, h: 0.35, fontFace: FONT_B, fontSize: 10, italic: true, color: MUTED, isTextBox: true, margin: 0 });
  footer(s);
  s.addNotes("Reading: the six EO models cluster at 10 m–13 km footprints. Two axes separate them — pixel vs patch, and snapshot vs period summary. AlphaEarth and TESSERA are period summaries over a year; TerraMind and Granite are single snapshots; Prithvi and OlmoEarth stack dated frames. Our codec is a snapshot of a time-mean field at a footprint two orders of magnitude larger.");
}

// =====================================================================
// 10. Summary table A — what goes in
{
  const s = pres.addSlide();
  s.background = { color: WHITE };
  title(s, "Summary I — what goes into each embedding", "Inputs, spatial stencil, temporal stencil, output vector");
  const hdr = (t) => ({ text: t, options: { bold: true, color: WHITE, fill: { color: NAVY }, fontSize: 10.5, valign: "middle" } });
  const c = (t, o) => ({ text: t, options: Object.assign({ fontSize: 10.5, color: INK, valign: "top" }, o || {}) });
  const nm = (t) => ({ text: t, options: { bold: true, color: NAVY, fontSize: 10, valign: "top" } });
  const rows = [
    [hdr("Model"), hdr("Inputs (sensors / modalities)"), hdr("Spatial stencil"), hdr("Temporal stencil"), hdr("Output per embedding")],
    [nm("AlphaEarth (DeepMind)"), c("S-2, Landsat 8/9, S-1, PALSAR-2, GEDI, ERA5-Land, GRACE, DEM, NLCD, text"), c("10 m output pixel inside a 1.28 km (128 px) frame; attention global within the frame"), c("Period summary; arbitrary valid window in the model, calendar year in the product"), c("64-D unit vector, int8; per pixel per year")],
    [nm("TESSERA (Cambridge)"), c("S-2 L2A (10 bands) + S-1 RTC (VV, VH)"), c("Single 10 m pixel — no spatial context"), c("1 calendar year; 40 sampled dates per modality, day-of-year encoded"), c("128-D int8; per pixel per year")],
    [nm("OlmoEarth (Ai2)"), c("S-1, S-2, Landsat-8 monthly + static maps (OSM, WorldCover, CDL, SRTM, canopy, WorldCereal)"), c("2.56 km tile (256 px @ 10 m); 1–8 px tokens (10–80 m)"), c("≤12 monthly steps over 1 year, each step tokenised; temporal masking"), c("768-D (Base) / 1024-D (Large) per patch per step + pooled vector")],
    [nm("TerraMind (IBM/ESA)"), c("S-2 L2A/L1C, S-1 GRD/RTC, DEM, RGB, NDVI, LULC, captions, geolocation"), c("2.24 km tile (224 px @ 10 m); 16 px = 160 m tokens, 14×14 grid"), c("Single co-registered timestamp"), c("196 × 768-D tokens per modality; merged 768-D; generated modalities")],
    [nm("Prithvi-EO 2.0 (IBM/NASA)"), c("HLS v2 at 30 m, 6 bands (blue, green, red, NIR, SWIR1, SWIR2)"), c("6.72 km tile (224 px @ 30 m); 480 m (300 M) or 420 m (600 M) tokens"), c("4 dated frames, 1–6 months apart, tubelet depth 1; single frame allowed"), c("1024-D (300 M) / 1280-D (600 M) per token per frame")],
    [nm("Granite-Geo-Ocean (IBM/STFC/PML)"), c("Sentinel-3 OLCI L2, 16 bands + SLSTR SST (17 ch) at 300 m"), c("12.6 km tile (42 px @ 300 m); 2 px = 600 m tokens, 21×21 grid"), c("Single acquisition (num_frames = 1)"), c("512-D per token; regression heads for chl-a, primary production")],
  ];
  s.addTable(rows, { x: 0.6, y: 1.55, w: 12.1, colW: [1.9, 3.0, 2.6, 2.5, 2.1], fontFace: FONT_B, border: { type: "solid", color: GRIDLINE, pt: 0.5 }, rowH: [0.4, 0.85, 0.7, 0.95, 0.8, 0.8, 0.8], margin: 0.06, autoPage: false });
  footer(s);
}

// 11. Summary table B — what each provides
{
  const s = pres.addSlide();
  s.background = { color: WHITE };
  title(s, "Summary II — what each project hands you", "Weights, precomputed embeddings, licence, ocean coverage, and the intended way to use it");
  const hdr = (t) => ({ text: t, options: { bold: true, color: WHITE, fill: { color: NAVY }, fontSize: 10.5, valign: "middle" } });
  const c = (t, o) => ({ text: t, options: Object.assign({ fontSize: 10.5, color: INK, valign: "top" }, o || {}) });
  const nm = (t) => ({ text: t, options: { bold: true, color: NAVY, fontSize: 10, valign: "top" } });
  const yes = (t) => c(t, { color: "1F7A4D", bold: true });
  const no = (t) => c(t, { color: "A23B3B", bold: true });
  const rows = [
    [hdr("Model"), hdr("Open weights / code"), hdr("Precomputed embeddings"), hdr("Licence"), hdr("Ocean?"), hdr("Intended use / evaluation style")],
    [nm("AlphaEarth"), no("No — model and code unreleased"), yes("Yes — global 10 m annual 2017–2024 in Earth Engine + GCS"), c("CC-BY 4.0 (data)"), no("Coastal/inland water only"), c("Frozen embeddings + linear/shallow heads on sparse labels; 'featurization' replacement")],
    [nm("TESSERA"), yes("Yes — weights CC0, code MIT, full pipeline"), yes("Yes — 10 m annual 2017–2025 via GeoTessera (tile-sampled)"), c("CC0 / MIT"), no("Land only (water masked)"), c("Frozen per-pixel embeddings + MLP/conv heads; label-efficient mapping")],
    [nm("OlmoEarth"), yes("Yes — 4 sizes on Hugging Face"), c("On demand — Studio exports for any AOI; no global archive"), c("OlmoEarth Artifact License (use restrictions)"), no("Not in pretraining sample"), c("Both: kNN/linear probes and full fine-tuning; platform for NGOs/agencies")],
    [nm("TerraMind"), yes("Yes — tiny/small/base/large + tokenizers"), no("No"), c("Apache 2.0"), no("Land-stratified TerraMesh"), c("Fine-tuning via TerraTorch; any-to-any generation; edge/onboard variants")],
    [nm("Prithvi-EO 2.0"), yes("Yes — 300 M / 600 M, ±TL"), no("No"), c("Apache 2.0 (HF) / MIT (GitHub) — verify"), no("Sea-only tiles removed"), c("Full fine-tuning of encoder + decoder (TerraTorch); GEO-Bench")],
    [nm("Granite-Geo-Ocean"), yes("Yes — HF weights + GitHub notebook"), no("No"), c("Apache 2.0"), yes("Yes — open ocean, surface optical + SST"), c("Fine-tune regression heads on ~100–200 in-situ points; ocean-colour biogeochemistry")],
  ];
  s.addTable(rows, { x: 0.6, y: 1.55, w: 12.1, colW: [1.6, 2.3, 2.6, 1.7, 1.6, 2.3], fontFace: FONT_B, border: { type: "solid", color: GRIDLINE, pt: 0.5 }, rowH: [0.4, 0.85, 0.8, 0.85, 0.7, 0.7, 0.8], margin: 0.06, autoPage: false });
  footer(s);
}


// 11b. Who embeds the ocean surface?
{
  const s = pres.addSlide();
  s.background = { color: WHITE };
  title(s, "Who embeds the ocean — and how deep?", "Representation models whose pretraining contains sea pixels: the published ones stop at the surface; Earth 2 is the one with the interior (checked Aug 2026)");
  const hdr = (t) => ({ text: t, options: { bold: true, color: WHITE, fill: { color: NAVY }, fontSize: 10, valign: "middle" } });
  const c = (t, o) => ({ text: t, options: Object.assign({ fontSize: 8.5, color: INK, valign: "top" }, o || {}) });
  const nm = (t) => ({ text: t, options: { bold: true, color: NAVY, fontSize: 9.5, valign: "top" } });
  const yes = (t) => c(t, { color: "1F7A4D", bold: true });
  const part = (t) => c(t, { color: "B58900", bold: true });
  const no = (t) => c(t, { color: "A23B3B", bold: true });
  const rows = [
    [hdr("Model (org, year)"), hdr("Ocean data in pretraining"), hdr("Open ocean? depth?"), hdr("Stencil"), hdr("Embedding / availability"), hdr("What it is for")],
    [nm("Earth 2 codec (ours, 2026)"), c("Gridded ocean state at ¼° / 1°: T and S at depth from Argo (plus pre-Argo gaps marked 'never measured'), SSH, currents, wind and surface forcing; 14–25 channels, monthly or pentad means"), yes("Yes — open ocean, surface to ~2000 dbar interior"), c("3 × 3 patch per channel token; one pixel-month snapshot; 24-month history in stage 2 with a 222–4444 km stencil"), c("64-D per pixel-month (128-D variant training); weights + embeddings ours to release"), c("Ocean-state field prediction; frozen-embedding probes to AMOC transport (RAPID, MOVE)")],
    [nm("Granite-Geospatial-Ocean (IBM · STFC · PML · Exeter, 2025)"), c("Sentinel-3 OLCI 16 bands + SLSTR SST, 300 m; 512 k tiles 2017–2021, balanced over 83 Longhurst provinces"), yes("Yes — surface optical only"), c("42 px = 12.6 km tile; 600 m tokens; single acquisition"), c("512-D per token; Apache 2.0 weights on HF + TerraTorch notebook"), c("Ocean-colour biogeochemistry: chl-a, primary production (surface optical)")],
    [nm("OceanSAR-1 / -2 (Galeio, Apr 2025 / Jan 2026)"), c("Sentinel-1 Wave-Mode SAR vignettes over open sea, ~12 M images 2015–2024, VV, downsampled to 50 m"), yes("Yes — surface roughness only"), c("256 × 256 px @ 50 m = 12.8 km; single acquisition; DINO / DINOv2 with dynamic dataset curation"), c("ResNet50 2048-D · ViT-S/B; v2 ViT 384-D; v1 Apache 2.0 on HF"), c("Sea-state: TenGeoP geophysical classes, wave height (RMSE 0.63–0.72 m), wind speed (RMSE ~1.4 m s⁻¹), icebergs")],
    [nm("Copernicus-FM / Copernicus-Pretrain (TUM, 2025)"), c("S-1, S-2, S-3 OLCI (300 m), S-5P (1 km), DEM on a 0.25° ERA5 grid; 18.7 M images, ~312 k grid cells"), part("Partial — 'whole land surface and near-land ocean'"), c("per-modality patches inside 0.25° cells; dynamic hypernetwork accepts any spectral/non-spectral input + metadata"), c("ViT encoders; code, data and weights released (github zhu-xlab); Copernicus-Bench, 15 tasks"), c("Unified multi-sensor EO backbone; coastal seas incidental, not a target")],
    [nm("Weather/climate FMs (Prithvi WxC, Aurora, ClimaX, AtmoRep)"), c("Reanalysis grids (MERRA-2, ERA5) whose surface fields exist over ocean cells; Aurora has an ocean-wave fine-tune (HRES-WAM)"), part("Ocean cells present, but as reanalysis background — SST/ocean state not the learning target"), c("0.25°–0.5° global grids; 1–2 lead-time steps"), c("Open weights (varies); no ocean embedding product"), c("Atmospheric forecasting; wave forecasting (Aurora)")],
    [nm("Location encoders (SatCLIP, GeoCLIP)"), c("Coordinates ↔ imagery/photos; SatCLIP samples S-2 globally, GeoCLIP uses geotagged photos"), no("Effectively no — sea coordinates unverified or absent"), c("point location → vector"), c("Open"), c("Geographic priors, not ocean state")],
  ];
  s.addTable(rows, { x: 0.6, y: 1.55, w: 12.1, colW: [2.0, 2.7, 1.55, 2.2, 1.95, 1.7], fontFace: FONT_B, border: { type: "solid", color: GRIDLINE, pt: 0.5 }, rowH: [0.35, 0.8, 0.75, 0.8, 0.8, 0.7, 0.5], margin: 0.05, autoPage: false });
  s.addText([
    { text: "Verdict.  ", options: { bold: true, color: TEAL } },
    { text: "Two purpose-built ocean encoders exist besides ours — both surface, both single-snapshot, one optical (colour + SST) and one radar (sea state). Everything else masks the sea out or carries it as reanalysis background; forecast-only systems (GLONET, Xihe, WenHai, AI-GOMS, Samudra) learn ocean dynamics but publish forecasts, not reusable embeddings. Earth 2 is, to our knowledge, the only embedding model whose inputs include the ocean interior through time.", options: {} },
  ], { x: 0.6, y: 6.3, w: 12.1, h: 0.7, fontFace: FONT_B, fontSize: 10, color: INK, valign: "top", isTextBox: true, margin: 0 });
  footer(s);
  s.addNotes("Ranked by how squarely the pretraining data is ocean. Granite and OceanSAR are the only two that sample open sea deliberately; Copernicus-Pretrain includes 'near-land ocean' because its 0.25° cells hug the coast. OceanSAR-2 (Jan 2026) did not state weight availability at the time of checking. Weather FMs carry ocean grid cells only because reanalysis fields are global.");
}


// 11b2. Four spheres
{
  const s = pres.addSlide(); s.background = { color: WHITE };
  title(s, "Four interacting spheres — who embeds which", "Atmospheric weather · ocean weather (incl. the interior) · ocean biosphere · land biosphere. Nobody represents two of them jointly today.");
  const hdr = (t) => ({ text: t, options: { bold: true, color: WHITE, fill: { color: NAVY }, fontSize: 10, valign: "middle" } });
  const c = (t, o) => ({ text: t, options: Object.assign({ fontSize: 8.5, color: INK, valign: "top" }, o || {}) });
  const nm = (t, col) => ({ text: t, options: { bold: true, color: col || NAVY, fontSize: 9.5, valign: "top" } });
  const rows = [
    [hdr("Sphere"), hdr("Who embeds it today"), hdr("Native cadence · memory"), hdr("Observing geometry"), hdr("Couples to the others through")],
    [nm("a) Atmospheric weather", TIME), c("Forecast FMs: Prithvi WxC, Aurora, ClimaX, GraphCast/AIFS/Pangu (states, not reusable embeddings). AlphaEarth ingests ERA5-Land only as an input."), c("hours–days · ~10 d predictability; slow modes (ENSO, NAO) months"), c("dense gridded reanalysis; radiances and stations upstream"), c("wind stress, heat & freshwater flux → ocean; precipitation, radiation → both biospheres")],
    [nm("b) Ocean weather", SPACE), c("Surface only: OceanSAR (sea state), Granite (SST as one band), AIFS-ocean 2026 (SST/SSS/SSH/currents, forecast). Interior: Earth 2 — the only embedding with T, S at depth."), c("days (surface) to decades (interior) · mixed layer months, interior years; top 2.5 m hold the heat of the whole atmospheric column"), c("gridded SST/SSH from satellites; sparse Argo profiles below; nothing under ice"), c("SST → atmosphere; mixed-layer depth, nutrients, light → ocean biosphere; carbon uptake → land via CO₂ (slow)")],
    [nm("c) Ocean biosphere", "1F7A4D"), c("Granite-Geospatial-Ocean (colour → chl-a, primary production); carbon-sink mappers SOM-FFN, CSIR-ML6; CANYON-B for BGC-Argo — all task models, one FM"), c("days–seasons · blooms weeks, seasonal cycle, decadal sink trends"), c("cloud-gapped optical swaths (Sentinel-3, MODIS); BGC-Argo sparse; pCO₂ ship tracks"), c("phytoplankton ← MLD, SST, wind (physics); CO₂ flux → atmosphere; export → interior carbon")],
    [nm("d) Land biosphere / surface", EMB), c("The mainstream: AlphaEarth, TESSERA, OlmoEarth, TerraMind, Prithvi-EO 2.0 — all fold a year, none touch the sea"), c("seasonal–annual · phenology, multi-year land-use memory"), c("dense 10–30 m tiles, frequent revisits; LiDAR, SAR"), c("runoff, dust, nutrients → coastal ocean; evapotranspiration, albedo → atmosphere")],
  ];
  s.addTable(rows, { x: 0.6, y: 1.5, w: 12.1, colW: [1.7, 3.3, 2.3, 2.2, 2.6], fontFace: FONT_B, border: { type: "solid", color: GRIDLINE, pt: 0.5 }, rowH: [0.35, 0.7, 0.85, 0.75, 0.6], margin: 0.05, autoPage: false });
  // sequencing strip
  const sy = 5.05, sh = 1.05, seq4 = [
    ["1  Ocean interior", SPACE, "the anchor: longest memory, least covered by anyone else — what we already have"],
    ["2  + Ocean colour", "1F7A4D", "same grid, Sentinel-3 recipe exists (Granite); closes physics ↔ biology via MLD, light, SST"],
    ["3  + Atmosphere as forcing", TIME, "already in our channels as wind/flux; at monthly cadence it is a wide, shallow forcing (cone slide), not a state to roll"],
    ["4  + Land biosphere", EMB, "last: the mainstream does it well, and the coupling to the ocean is slow (rivers, dust, CO₂)"],
  ];
  seq4.forEach(([h, col, d], i) => {
    const x = 0.6 + i * 3.05;
    s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x, y: sy, w: 2.9, h: sh, fill: { color: PALE }, line: { color: col, width: 1.25 }, rectRadius: 0.08 });
    s.addText([{ text: h, options: { bold: true, color: col, fontSize: 10.5, breakLine: true } }, { text: d, options: { fontSize: 8.5, color: INK } }], { x: x + 0.12, y: sy + 0.05, w: 2.66, h: sh - 0.1, fontFace: FONT_B, valign: "top", isTextBox: true, margin: 0 });
    if (i < 3) s.addShape(pres.shapes.RIGHT_ARROW, { x: x + 2.9 + 0.02, y: sy + sh / 2 - 0.1, w: 0.11, h: 0.2, fill: { color: GRIDLINE }, line: { width: 0 } });
  });
  s.addText([
    { text: "Three design consequences.  ", options: { bold: true, color: TEAL } },
    { text: "(i) Memories differ by orders of magnitude, so each sphere needs its own (v, τ) cone and the coupling variables (SST, fluxes, chlorophyll–light–MLD) are the L-shaped unions. (ii) Biospheres run on the calendar, physics on the clock: a joint model needs period tokens (pixel-year, slide on the TESSERA arm) for c) and d) and dated-state tokens for a) and b). (iii) The observing systems differ in geometry — grids, sparse profiles, cloud-gapped swaths, 10 m tiles — and our masked-vs-never-measured token distinction is the primitive that lets one encoder take all of them.", options: {} },
  ], { x: 0.6, y: 6.2, w: 12.1, h: 0.8, fontFace: FONT_B, fontSize: 8.5, color: INK, valign: "top", isTextBox: true, margin: 0 });
  footer(s);
  s.addNotes("Yannick's framing (31 Aug): four interacting spheres — atmospheric weather, ocean weather with its much larger heat capacity, ocean biosphere (carbon sinks), land biosphere — and the ambition to build first an embedding model, then a system, that combines all relevant data. The sequencing grows outward from the ocean interior because it is the sphere with the longest memory and the least existing coverage; ocean colour is the cheapest next step. The heat-capacity figure: the upper ~2.5 m of ocean has the same heat capacity as the full atmospheric column.");
}

// 11c–11g. Head-to-head numbers (tables transcribed from the papers)
// helper: table from numeric rows; bold the best per column (higher or lower is better), grey our-six vs context rows
function scoreTable(slide, x, y, w, colW, header, rows, opts) {
  opts = opts || {};
  const lowerBetter = opts.lowerBetter || [];   // column indices (1-based within data cols) where lower is better
  const nCols = header.length - 1;
  const best = [];
  for (let c = 0; c < nCols; c++) {
    let vals = rows.map(r => typeof r.v[c] === "number" ? r.v[c] : null).filter(v => v !== null);
    if (!vals.length) { best.push(null); continue; }
    best.push(lowerBetter.includes(c + 1) ? Math.min(...vals) : Math.max(...vals));
  }
  const hdr = header.map((h, i) => ({ text: h, options: { bold: true, color: WHITE, fill: { color: NAVY }, fontSize: opts.fs || 9, valign: "middle", align: i ? "center" : "left" } }));
  const body = rows.map(r => {
    const name = { text: r.n, options: { bold: true, color: r.ours ? NAVY : MUTED, fontSize: opts.fs || 9, valign: "middle", fill: r.ours ? { color: "EEF3F8" } : undefined } };
    const cells = r.v.map((v, c) => {
      const isBest = typeof v === "number" && v === best[c];
      const txt = v === null || v === undefined ? "–" : (typeof v === "number" ? (opts.fmt ? opts.fmt(v, c) : String(v)) : v);
      return { text: txt, options: { fontSize: opts.fs || 9, color: isBest ? "1F7A4D" : (r.ours ? INK : MUTED), bold: isBest, align: "center", valign: "middle", fill: r.ours ? { color: "EEF3F8" } : undefined } };
    });
    return [name, ...cells];
  });
  slide.addTable([hdr, ...body], { x, y, w, colW, fontFace: FONT_B, border: { type: "solid", color: GRIDLINE, pt: 0.5 }, rowH: opts.rowH || 0.26, margin: 0.04, autoPage: false });
}
function tableTitle(slide, text, x, y, w) {
  slide.addText(text, { x, y, w, h: 0.3, fontFace: FONT_B, fontSize: 11, bold: true, color: NAVY, isTextBox: true, margin: 0 });
}

// ---- H1: frozen embeddings, OlmoEarth Table 2
{
  const s = pres.addSlide(); s.background = { color: WHITE };
  title(s, "Head-to-head I — frozen embeddings, one protocol", "OlmoEarth paper, Table 2: kNN (k = 20, cosine) for single-image classification, linear probe for segmentation and multi-temporal tasks. Five of our six appear.");
  const header = ["Model (frozen encoder)", "BigEarthNet µF1", "So2Sat acc", "EuroSAT acc", "CropHarvest-Togo S1+S2 acc", "m-cashew mIoU", "SA-crop mIoU", "PASTIS (S2) mIoU", "MADOS mIoU", "Sen1Floods11 mIoU"];
  const rows = [
    { n: "OlmoEarth Base (89 M)", ours: true, v: [62.4, 67.7, 94.7, 82.0, 32.3, 28.9, 50.6, 67.2, 79.2] },
    { n: "OlmoEarth Large (308 M)", ours: true, v: [62.0, 68.2, 96.3, 79.7, 30.9, 28.5, 51.8, 66.4, 79.8] },
    { n: "TerraMind Base", ours: true, v: [63.9, 46.7, 85.6, 77.5, 46.0, 30.4, 40.9, 66.0, 78.7] },
    { n: "TerraMind Large", ours: true, v: [63.9, 47.4, 90.0, 75.2, 50.4, 31.2, 41.3, 67.5, 78.4] },
    { n: "Prithvi-EO 2.0 300M", ours: true, v: [51.6, 34.7, 82.2, null, 46.8, 24.7, 37.2, 50.9, null] },
    { n: "Prithvi-EO 2.0 600M", ours: true, v: [51.0, 34.0, 81.2, null, 45.8, 26.6, 37.5, 52.2, null] },
    { n: "TESSERA (precomputed)", ours: true, v: [null, null, null, 81.0, null, null, null, null, null] },
    { n: "Galileo Base", v: [58.3, 55.7, 92.8, 77.5, 28.9, 25.3, 39.6, 68.4, 79.4] },
    { n: "DINOv3-Sat 7B", v: [61.6, 50.1, 91.3, null, 54.1, 31.7, 26.3, 59.7, null] },
    { n: "Panopticon Base", v: [64.9, 60.5, 95.2, 76.5, 32.7, 27.3, 30.2, 66.1, 78.0] },
    { n: "Copernicus-FM Base", v: [64.6, 50.3, 84.7, 70.6, 32.2, 28.4, 32.1, 63.9, 77.6] },
    { n: "CROMA Large", v: [59.2, 48.2, 85.8, 81.0, 27.0, 30.4, 42.7, 66.4, 78.8] },
  ];
  scoreTable(s, 0.6, 1.6, 12.1, [2.25, 1.15, 1.0, 1.0, 1.4, 1.1, 1.0, 1.1, 1.0, 1.1], header, rows, { rowH: 0.27, fmt: v => v.toFixed(1) });
  s.addText([
    { text: "Reading the table.  ", options: { bold: true, color: TEAL } },
    { text: "Green = best in column (all rows). Shaded rows are the models in this deck; grey rows are context. On frozen features OlmoEarth leads the classification columns (So2Sat, EuroSAT, Sen1Floods11, PASTIS), TerraMind leads BigEarthNet and the cashew/SA-crop segmentation, and Prithvi-EO 2.0 trails on every classification column but is competitive on m-cashew. TESSERA appears once (CropHarvest-Togo, 81.0 — within 1 pp of OlmoEarth Base). AlphaEarth is not in this table (its embeddings are annual and 10 m, so the authors compared it separately — next slides). Caveat: authored by the OlmoEarth team.", options: {} },
  ], { x: 0.6, y: 5.35, w: 12.1, h: 1.3, fontFace: FONT_B, fontSize: 9.5, color: INK, valign: "top", isTextBox: true, margin: 0 });
  footer(s);
  s.addNotes("Source: OlmoEarth, arXiv 2511.13655, Table 2 (kNN/linear-probe on research benchmarks). Numbers transcribed from the paper's table; column subset chosen so that as many of our models as possible appear. TESSERA only appears on the CropHarvest tasks (Togo S1+S2 81.0, PRC S1+S2 72.2). OlmoEarth Nano reaches 83.7 on Togo — the best in that column overall — but is omitted here to keep the rows readable.");
}

// ---- H2: fine-tuned (Olmo Table 3) + PANGAEA (TerraMind Table 6)
{
  const s = pres.addSlide(); s.background = { color: WHITE };
  title(s, "Head-to-head II — fine-tuning and PANGAEA", "Top: OlmoEarth Table 3 (full fine-tuning, encoder unfrozen after 20 % of epochs). Bottom: TerraMind Table 6 (PANGAEA, frozen encoder + UPerNet, mIoU).");
  tableTitle(s, "Fine-tuned — accuracy / mIoU (OlmoEarth paper, Table 3)", 0.6, 1.45, 12);
  const h1 = ["Model (fine-tuned)", "BigEarthNet", "So2Sat", "EuroSAT", "m-cashew", "SA-crop", "PASTIS", "MADOS", "Sen1Floods11"];
  const r1 = [
    { n: "OlmoEarth Base", ours: true, v: [72.0, 68.6, 98.7, 79.8, 39.6, 64.3, 77.8, 79.8] },
    { n: "OlmoEarth Large", ours: true, v: [72.4, 68.1, 98.5, 80.6, 40.8, 66.3, 81.8, 79.8] },
    { n: "TerraMind Base", ours: true, v: [72.6, 66.1, 97.6, 80.9, 39.2, 59.9, 73.2, 79.5] },
    { n: "TerraMind Large", ours: true, v: [74.0, 65.4, 97.8, 81.3, 41.1, 60.9, 71.5, 79.5] },
    { n: "Prithvi-EO 2.0 600M", ours: true, v: [70.6, 64.7, 96.8, 81.1, 38.8, 58.6, 69.3, null] },
    { n: "Galileo Base", v: [69.2, 64.7, 97.8, 78.8, 35.7, 61.2, 71.9, 79.7] },
    { n: "Copernicus-FM", v: [71.3, 66.8, 98.5, 78.7, 33.6, 54.6, 66.0, 78.6] },
    { n: "OlmoEarth Base, random init (no pretraining)", v: [61.0, 48.9, 80.3, 43.0, 27.5, 43.9, 45.6, 77.0] },
  ];
  scoreTable(s, 0.6, 1.75, 12.1, [3.0, 1.15, 1.1, 1.1, 1.15, 1.1, 1.1, 1.1, 1.3], h1, r1, { rowH: 0.23, fs: 8, fmt: v => v.toFixed(1) });
  tableTitle(s, "PANGAEA — mIoU, frozen encoder + UPerNet (TerraMind paper, Table 6; Prithvi appears only as v1.0)", 0.6, 3.95, 12);
  const h2 = ["Model", "HLS Burn Scars", "MADOS", "PASTIS-R", "Sen1Floods11", "FiveBillionPx", "DynamicEarthNet", "CropType S.Sudan", "SpaceNet 7", "AI4SmallFarms", "Average", "Avg rank"];
  const r2 = [
    { n: "TerraMind Large", ours: true, v: [82.93, 75.57, 43.13, 90.78, 63.38, 37.89, 55.04, 59.98, 27.47, 59.57, 3.44] },
    { n: "TerraMind Base", ours: true, v: [82.42, 69.52, 40.51, 90.62, 59.72, 37.87, 55.80, 60.61, 28.12, 58.35, 3.94] },
    { n: "Prithvi-EO 1.0 100M", ours: true, v: [83.62, 49.98, 33.93, 90.37, 46.81, 27.86, 43.07, 56.54, 26.86, 51.00, 11.00] },
    { n: "CROMA", v: [82.42, 67.55, 32.32, 90.89, 51.83, 38.29, 49.38, 59.28, 25.65, 55.29, 6.61] },
    { n: "DOFA", v: [80.63, 59.58, 30.02, 89.37, 43.18, 39.29, 51.33, 61.84, 27.07, 53.59, 8.22] },
    { n: "U-Net trained from scratch", v: [84.51, 54.79, 31.60, 91.42, 60.47, 39.46, 47.57, 62.09, 46.34, 57.58, 4.89] },
  ];
  scoreTable(s, 0.6, 4.25, 12.1, [2.1, 1.0, 0.85, 0.9, 1.0, 1.0, 1.1, 1.05, 0.9, 1.0, 0.8, 0.4], h2, r2, { rowH: 0.23, fs: 8, lowerBetter: [11], fmt: (v, c) => v.toFixed(2) });
  s.addText([
    { text: "Reading.  ", options: { bold: true, color: TEAL } },
    { text: "Once fine-tuned, the gap closes: TerraMind Large and OlmoEarth Large split the columns (TerraMind: BigEarthNet, cashew, SA-crop; OlmoEarth: So2Sat, EuroSAT, PASTIS, MADOS) and Prithvi-EO 2.0 600M sits 1–3 pp behind on the classification columns but 8–13 pp behind on PASTIS and MADOS. The random-init row is the size of the pretraining effect itself (+10 to +37 pp). On PANGAEA, TerraMind is the only foundation model that beats a U-Net trained from scratch on average (59.6 vs 57.6); Prithvi-EO 1.0 loses to it. Prithvi-EO 2.0's own GEO-Bench table (600M-TL best on 6 of 12 sets vs DOFA / Scale-MAE / DeCUR / Satlas) contains none of the other models here.", options: {} },
  ], { x: 0.6, y: 6.32, w: 12.1, h: 0.65, fontFace: FONT_B, fontSize: 8.5, color: INK, valign: "top", isTextBox: true, margin: 0 });
  footer(s);
  s.addNotes("Sources: OlmoEarth arXiv 2511.13655 Table 3; TerraMind arXiv 2504.11171 Table 6 (PANGAEA; xView2 and BioMassters excluded by the PANGAEA authors' advice). Prithvi-EO 2.0 GEO-Bench numbers (arXiv 2412.02732 Tables A1–A4) are not shown because none of the other models in this deck appear in them.");
}

// ---- H3: the embedding products — TESSERA vs AlphaEarth, AlphaEarth vs OlmoEarth
{
  const s = pres.addSlide(); s.background = { color: WHITE };
  title(s, "Head-to-head III — the two embedding products", "TESSERA paper, Table 1 (frozen embeddings + light head, 100 % labels unless noted) · OlmoEarth paper, Table 7 · an independent Local-Climate-Zone study (Bern)");
  tableTitle(s, "TESSERA vs AlphaEarth vs frozen RSFMs (TESSERA paper) — F1 / mIoU ↑, RMSE ↓", 0.6, 1.5, 8);
  const h1 = ["Model", "TreeSatAI-TS F1", "Austrian crop F1 (30 % labels)", "PASTIS-R mIoU", "Austrian crop seg. mIoU", "BioMassters RMSE (t/ha)", "Borneo canopy RMSE (m)"];
  const r1 = [
    { n: "TESSERA (128-D, frozen)", ours: true, v: [77.96, 82.09, 50.68, 53.12, 27.43, 12.21] },
    { n: "AlphaEarth (64-D, frozen)", ours: true, v: [76.90, 56.36, 51.08, 25.70, 29.59, 16.11] },
    { n: "Prithvi-EO (frozen)", ours: true, v: [65.86, null, 34.09, 16.25, 41.12, 19.71] },
    { n: "Presto (frozen)", v: [67.81, 57.89, null, 34.04, null, 17.88] },
    { n: "Galileo (frozen)", v: [69.44, null, 27.92, 22.80, 38.52, 20.48] },
    { n: "SkySense (frozen)", v: [69.35, null, 32.87, 26.82, 32.52, 16.56] },
    { n: "U-Net baseline (frozen features)", v: [71.39, null, 30.16, 27.00, 36.41, 17.79] },
  ];
  scoreTable(s, 0.6, 1.82, 8.0, [1.95, 0.95, 1.15, 0.9, 1.05, 1.0, 1.0], h1, r1, { rowH: 0.27, fs: 8.5, lowerBetter: [5, 6], fmt: v => v.toFixed(2) });
  // AEF vs OlmoEarth
  tableTitle(s, "AlphaEarth vs OlmoEarth Base (OlmoEarth paper, Table 7)", 8.9, 1.5, 4);
  const h2 = ["Model · protocol", "Nandi acc", "AWF acc", "Ecosystem acc", "LFMC L1 ↓", "Solar farm mIoU"];
  const r2 = [
    { n: "AlphaEarth · kNN", ours: true, v: [55.6, 81, 60.6, null, null] },
    { n: "AlphaEarth · frozen + decoder", ours: true, v: [66.0, 75.9, 61.2, 23.1, 77.5] },
    { n: "AlphaEarth · full fine-tune", ours: true, v: ["n/a", "n/a", "n/a", "n/a", "n/a"] },
    { n: "OlmoEarth · kNN", ours: true, v: [66.2, 82, 59.3, null, null] },
    { n: "OlmoEarth · frozen + decoder", ours: true, v: [62.9, 84.0, 61.1, 19.9, 84.8] },
    { n: "OlmoEarth · full fine-tune", ours: true, v: [82.2, 86.0, 62.4, 17.9, 86.7] },
  ];
  scoreTable(s, 8.9, 1.82, 3.8, [1.35, 0.5, 0.45, 0.55, 0.45, 0.5], h2, r2, { rowH: 0.3, fs: 7.5, lowerBetter: [4], fmt: v => v.toFixed(1) });
  // LCZ independent
  tableTitle(s, "Independent: LCZ mapping, Bern (arXiv 2606.20034)", 8.9, 4.4, 4);
  const h3 = ["Input to Attention U-Net", "Accuracy", "IoU"];
  const r3 = [
    { n: "Sentinel-1 + 2 composites", v: [0.87, 0.77] },
    { n: "AlphaEarth 64-D", ours: true, v: [0.89, 0.81] },
    { n: "TESSERA 128-D", ours: true, v: [0.90, 0.82] },
  ];
  scoreTable(s, 8.9, 4.72, 3.8, [2.2, 0.8, 0.8], h3, r3, { rowH: 0.27, fs: 8.5, fmt: v => v.toFixed(2) });
  s.addText([
    { text: "Reading.  ", options: { bold: true, color: TEAL } },
    { text: "TESSERA beats AlphaEarth on 5 of these 6 tasks in TESSERA's own table (AlphaEarth edges PASTIS-R by 0.4 mIoU), with the largest margins on pixel-wise crop tasks where per-pixel time series matter most; both precomputed products beat every frozen ViT-style RSFM including Prithvi by wide margins. Against OlmoEarth, AlphaEarth wins only where OlmoEarth is also frozen (Nandi frozen+decoder); once OlmoEarth is fine-tuned it wins all five — and AlphaEarth cannot be fine-tuned. The independent Bern study puts the two products within 1 pp of each other and 2–5 pp above raw composites. AlphaEarth's own paper reports a 23.9 % average error reduction vs the next-best method over 15 evaluations (per-dataset values are charts only), losing on just one (LCMAP land-use change).", options: {} },
  ], { x: 0.6, y: 4.15, w: 8.0, h: 2.6, fontFace: FONT_B, fontSize: 9, color: INK, valign: "top", isTextBox: true, margin: 0 });
  footer(s);
  s.addNotes("Sources: TESSERA arXiv 2506.20380 Table 1 (frozen-embedding rows; RSFM rows shown at 'orig' = frozen encoder — the fine-tuned RSFM values are 1–5 pp higher and still below both products); OlmoEarth arXiv 2511.13655 Table 7; 'AlphaEarth and TESSERA for Local Climate Zone Mapping', arXiv 2606.20034, Experiment II. AlphaEarth arXiv 2507.22291: 15 evaluations, ~23.9 % error reduction at max-shot, 10.4 % at 10-shot, 4.2 % at 1-shot.");
}

// ---- H4: ocean models against their own baselines
{
  const s = pres.addSlide(); s.background = { color: WHITE };
  title(s, "Head-to-head IV — the ocean encoders", "Granite-Geospatial-Ocean, Table 1 (5-fold CV RMSE, log₁₀ units) · OceanSAR-1, Tables 2–3 (Sentinel-1 wave-mode tasks)");
  tableTitle(s, "Granite-Geospatial-Ocean — fine-tuned FM vs from scratch vs random forest (RMSE ↓)", 0.6, 1.5, 6.2);
  const h1 = ["Model", "Chl-a RMSE log₁₀ mg m⁻³", "Primary production RMSE log₁₀ mgC m⁻² d⁻¹"];
  const r1 = [
    { n: "OLCI + SST · foundation model, fine-tuned", ours: true, v: [0.14, 0.39] },
    { n: "OLCI only · foundation model, fine-tuned", ours: true, v: [0.16, 0.39] },
    { n: "OLCI + SST · same net from scratch", v: [0.16, 0.42] },
    { n: "OLCI only · same net from scratch", v: [0.16, 0.43] },
    { n: "Random forest on pixel values", v: [0.16, 0.40] },
  ];
  scoreTable(s, 0.6, 1.82, 5.9, [2.7, 1.5, 1.7], h1, r1, { rowH: 0.3, fs: 8.5, lowerBetter: [1, 2], fmt: v => v.toFixed(2) });
  s.addText("± spreads: chl-a 0.03–0.10, PP 0.04–0.07 across folds — the FM gain (0.02 / 0.03–0.04) is inside the fold spread; the authors' stronger claim is label-efficiency (gain grows as labels shrink to 25 %) and spatial pattern (SSIM vs the operational L2 product: FM 0.88, scratch 0.82, decision tree 0.68). No comparison to OC4Me-type algorithms is given.", { x: 0.6, y: 3.7, w: 5.9, h: 1.2, fontFace: FONT_B, fontSize: 9, color: INK, valign: "top", isTextBox: true, margin: 0 });
  tableTitle(s, "OceanSAR-1 — kNN on frozen features vs other SAR FMs (acc ↑, RMSE ↓)", 6.8, 1.5, 6.0);
  const h2 = ["Model", "TenGeoP acc %", "Wave height RMSE (m)", "Wind speed RMSE (m s⁻¹)"];
  const r2 = [
    { n: "OceanSAR-1 ViT-B/8", ours: true, v: [83.6, 0.63, 1.37] },
    { n: "OceanSAR-1 ViT-S/8", ours: true, v: [82.1, 0.64, 1.38] },
    { n: "OceanSAR-1 ResNet50", ours: true, v: [75.5, 0.75, 1.62] },
    { n: "SoftCon ViT-B/14", v: [74.8, 1.01, 2.08] },
    { n: "CROMA ViT-B/8", v: [65.4, 0.78, 1.95] },
    { n: "DOFA ViT-L/16", v: [63.4, 0.95, 2.43] },
    { n: "MoCo ResNet50", v: [60.9, 0.83, 1.98] },
  ];
  scoreTable(s, 6.8, 1.82, 5.9, [2.1, 1.1, 1.3, 1.4], h2, r2, { rowH: 0.27, fs: 8.5, lowerBetter: [2, 3], fmt: (v, c) => c === 0 ? v.toFixed(1) : v.toFixed(2) });
  s.addText("With linear-probe / LoRA fine-tuning OceanSAR-1 ViT-B/8 reaches 88.3 % / 0.54 m / 1.29 m s⁻¹ (MoCo-FT 86.5 / 0.77 / 1.80). Ocean-only pretraining beats land-trained SAR FMs by 9–23 pp on sea-state classification and cuts wind-speed error by 30–45 %.", { x: 6.8, y: 4.05, w: 5.9, h: 0.9, fontFace: FONT_B, fontSize: 9, color: INK, valign: "top", isTextBox: true, margin: 0 });
  s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x: 0.6, y: 5.25, w: 12.1, h: 1.2, fill: { color: PALE }, line: { width: 0 }, rectRadius: 0.08 });
  s.addText([
    { text: "Why no cross-model ocean table.  ", options: { bold: true, color: TEAL } },
    { text: "The two ocean encoders measure different physics (optical colour vs radar roughness) against different in-situ truths, and neither has been run on a land benchmark or on each other's tasks. The only common denominator with the land models is the protocol — frozen features + small head, and 'FM vs same net from scratch' as the pretraining control — which is also the protocol of our attribution matrix. Our numbers (RAPID r 0.617 pixel / 0.672 patch vs raw 0.613 / 0.659) are the same kind of statement as Granite's 0.14 vs 0.16: a pretraining margin measured against a matched from-scratch control, and in both cases a small one.", options: {} },
  ], { x: 0.8, y: 5.32, w: 11.7, h: 1.3, fontFace: FONT_B, fontSize: 9.5, color: INK, valign: "top", isTextBox: true, margin: 0 });
  footer(s);
  s.addNotes("Sources: arXiv 2509.21273 Table 1 (RMSE only; no R²/bias; label-fraction sweep is Figure 8 without printed values); arXiv 2504.06962 Tables 1–3. Earth 2 numbers from paper.tex attribution matrix.");
}

// ---- H5: digest — who wins where
{
  const s = pres.addSlide(); s.background = { color: WHITE };
  title(s, "Who wins where — a digest of the tables", "Rankings restricted to the models in this deck, per protocol; every source table was written by one of the contestants");
  const hdr = (t) => ({ text: t, options: { bold: true, color: WHITE, fill: { color: NAVY }, fontSize: 10, valign: "middle" } });
  const c = (t, o) => ({ text: t, options: Object.assign({ fontSize: 9.5, color: INK, valign: "top" }, o || {}) });
  const nm = (t) => ({ text: t, options: { bold: true, color: NAVY, fontSize: 10, valign: "top" } });
  const rows = [
    [hdr("Protocol · metric family"), hdr("Ranking among our models"), hdr("Margin / evidence"), hdr("Source & caveat")],
    [nm("Frozen features · classification accuracy / µF1"), c("OlmoEarth > TerraMind > Prithvi-EO 2.0"), c("So2Sat 68 vs 47 vs 35; EuroSAT 96 vs 90 vs 82; BigEarthNet TerraMind 63.9 edges OlmoEarth 62.4"), c("OlmoEarth Table 2 (own paper)")],
    [nm("Frozen features · segmentation mIoU"), c("split: TerraMind (cashew 50.4, SA-crop 31.2) · OlmoEarth (PASTIS 51.8, MADOS 67.2, floods 79.8) · Prithvi last"), c("Prithvi-EO 2.0 competitive only on m-cashew (46.8)"), c("OlmoEarth Table 2")],
    [nm("Fine-tuned · accuracy / mIoU"), c("TerraMind-L ≈ OlmoEarth-L > Prithvi-EO 2.0 600M"), c("columns split 3/4 between the two leaders; Prithvi 1–3 pp behind on classification, 8–13 pp on PASTIS/MADOS; pretraining itself worth +10 to +37 pp (random-init row)"), c("OlmoEarth Table 3")],
    [nm("PANGAEA · mIoU, frozen + UPerNet"), c("TerraMind-L 59.6 > TerraMind-B 58.4 > U-Net scratch 57.6 > … > Prithvi-EO 1.0 51.0"), c("only GFM above the from-scratch U-Net; Prithvi 2.0 not evaluated"), c("TerraMind Table 6")],
    [nm("GEO-Bench-2 · capability ranks (of 14)"), c("Prithvi 600M-TL core rank 5 · TerraMind-L 6 · Prithvi 300M-TL 9 · TerraMind-B 11; TerraMind-L rank 1 on multi-temporal and multi-spectral"), c("ranks only, no raw scores"), c("GEO-Bench-2, arXiv 2511.15658 (independent); no AlphaEarth / TESSERA / OlmoEarth / Granite")],
    [nm("Precomputed products · frozen + head"), c("TESSERA > AlphaEarth on 5/6 TESSERA tasks; ≈ tie on Bern LCZ (0.82 vs 0.81 IoU); both ≫ frozen Prithvi"), c("Austrian crop F1 82 vs 56; canopy RMSE 12.2 vs 16.1 m; PASTIS AlphaEarth +0.4"), c("TESSERA Table 1 (own) · LCZ study (independent)")],
    [nm("Product vs fine-tunable model"), c("fine-tuned OlmoEarth > AlphaEarth on 5/5; frozen-vs-frozen mixed (AlphaEarth wins Nandi 66.0 vs 62.9)"), c("Nandi 82.2 vs 66.0; solar farm 86.7 vs 77.5"), c("OlmoEarth Table 7 (own)")],
    [nm("Ocean · RMSE vs in-situ, FM vs scratch"), c("Granite: 0.14 vs 0.16 (chl-a), 0.39 vs 0.42 (PP) · OceanSAR-1: 83.6 % vs 74.8 % best land-trained SAR FM"), c("Granite gain inside fold spread; OceanSAR gain large"), c("own papers; no shared task with land models")],
  ];
  s.addTable(rows, { x: 0.6, y: 1.5, w: 12.1, colW: [2.3, 4.0, 3.3, 2.5], fontFace: FONT_B, border: { type: "solid", color: GRIDLINE, pt: 0.5 }, rowH: [0.35, 0.45, 0.5, 0.5, 0.45, 0.55, 0.5, 0.45, 0.45], margin: 0.05, autoPage: false });
  s.addText([
    { text: "Two structural caveats.  ", options: { bold: true, color: "A23B3B" } },
    { text: "No benchmark contains all six models — the widest single table (OlmoEarth's) has five and omits AlphaEarth; the only independent multi-model source (GEO-Bench-2) has two. And no public benchmark has an ocean-state, transport or interior task, so none of these rankings transfer to our problem; the transferable lesson is the protocol — frozen probe vs fine-tune, and a from-scratch control at matched receptive field, which is what our attribution matrix already does.", options: {} },
  ], { x: 0.6, y: 6.3, w: 12.1, h: 0.7, fontFace: FONT_B, fontSize: 9, color: INK, valign: "top", isTextBox: true, margin: 0 });
  footer(s);
  s.addNotes("Digest of slides H1–H4. Rankings are read off the transcribed tables and restricted to the models of this deck; 'own' marks a table authored by one of the compared teams.");
}


// =====================================================================
// 12. Takeaways for Earth 2
{
  const s = pres.addSlide();
  s.background = { color: WHITE };
  title(s, "What this means for our ocean representation work", "Six models, two design axes, one gap");
  const cards = [
    ["Axis 1 — pixel vs. patch", "TESSERA proves a per-pixel encoder with zero spatial context is competitive on land; every other model buys context with a ViT patch grid. We have already priced this axis: the attribution matrix reads single-pixel raw 0.613 vs codec 0.617, and 3×3 raw 0.659 vs codec 0.672 (two seeds) — the neighbourhood is worth ≈ +0.05, pretraining ≈ +0.01.", SPACE],
    ["Axis 2 — snapshot vs. period", "TerraMind and Granite are snapshots; Prithvi and OlmoEarth stack dated frames; AlphaEarth and TESSERA fold a full year into one vector. Our stage-1 codec is a snapshot of a monthly (or 5-day) mean and the month-block codec folds only ~6 pentads. The year-folded design is the untested axis — hence the proposed arm on the last slide.", TIME],
    ["Product vs. model", "AlphaEarth is a dataset you cannot re-run; TESSERA and OlmoEarth ship both weights and embeddings. For the ocean nothing precomputed exists — whatever we release is the first embedding-field product for the ocean interior.", EMB],
    ["The gap", "Only two published encoders are trained on sea pixels — Granite-Geospatial-Ocean (colour + SST) and OceanSAR (sea-state radar) — and both are surface, single-instant models. To our knowledge no published model embeds the gridded ocean state (T, S, SSH, currents) through time — the territory our codec occupies.", "1F7A4D"],
  ];
  cards.forEach(([h, body, col], i) => {
    const cx = 0.6 + (i % 2) * 6.2, cy = 1.6 + Math.floor(i / 2) * 2.65;
    s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x: cx, y: cy, w: 5.9, h: 2.45, fill: { color: PALE }, line: { width: 0 }, rectRadius: 0.08 });
    s.addShape(pres.shapes.OVAL, { x: cx + 0.25, y: cy + 0.25, w: 0.42, h: 0.42, fill: { color: col }, line: { width: 0 } });
    s.addText(String(i + 1), { x: cx + 0.25, y: cy + 0.25, w: 0.42, h: 0.42, fontFace: FONT_B, fontSize: 14, bold: true, color: WHITE, align: "center", valign: "middle", isTextBox: true, margin: 0 });
    s.addText(h, { x: cx + 0.8, y: cy + 0.25, w: 4.9, h: 0.42, fontFace: FONT_B, fontSize: 15, bold: true, color: NAVY, valign: "middle", isTextBox: true, margin: 0 });
    s.addText(body, { x: cx + 0.3, y: cy + 0.8, w: 5.3, h: 1.55, fontFace: FONT_B, fontSize: 13, color: INK, valign: "top", isTextBox: true, margin: 0 });
  });
  footer(s);
  s.addNotes("Discussion slide. Correction from the first draft: the paper's attribution matrix already contains the single-pixel (p=1) codec and raw controls, so the spatial 1×1 ablation is done. What TESSERA adds that we have not tried is folding a whole year of one pixel into one vector — the proposed arm.");
}


// 12b. Proposed arm — TESSERA-style pixel-year codec
modelSlide(0, "Proposed arm: a TESSERA-style pixel-year codec", "Fold the year, not the neighbourhood — the one TESSERA ingredient the programme has not yet tested",
  [
    ["Why not a 1×1 spatial ablation", "Already done. The attribution matrix has p = 1 and 3×3 at both raw and codec: 0.613 / 0.617 vs 0.659 / 0.672. The spatial axis is priced; the temporal axis is not."],
    ["What TESSERA does that we don't", "One embedding per pixel-YEAR: a 4-layer temporal transformer + GRU pooling over ~40 dated observations of a single pixel, day-of-year embeddings, Barlow Twins objective (no reconstruction), int8 QAT. Our stage-1 z is one pixel-MONTH; stage 2 then runs over the monthly z's."],
    ["Arm A — pixel-year codec", "Encoder over the 12 (or 24) pixel-months of one ¼° pixel: channel tokens carry month-of-year; p = 1 to isolate the temporal fold; emit one z ∈ ℝ⁶⁴ per pixel-year (or rolling 12-month window ending at t). Decoder queries (channel, Δmonth) up to ±12. Token count ≈ 12 × (C+2) ≈ 310 at 24 channels — same order as the 282-token month-block codec, so it fits the existing recipe."],
    ["Arm B — objective swap", "Same encoder, Barlow Twins on two random date-subsamples of the same pixel-year (TESSERA's augmentation) instead of masked reconstruction. Tests whether a non-reconstructive objective keeps the deep-density detail the decoder had to be re-taught to read."],
    ["Controls (matched)", "Mean of the 12 monthly z's (pooling control) · stage-2 over monthly z's (current path) · raw 12-month stack into the same attention head (end-to-end control, as in the attribution matrix). Two codec seeds minimum — head numbers are never quoted from one."],
    ["Read-outs", "Probe ladder → RAPID r and RMSE (Sv); round-trip variance lost per channel; rolled corridor AUC at h = 12 with a year-folded z as the state. Success = the folded z beats the pooled control by more than the two-seed spread."],
  ],
  {
    space: { n: 9, outPx: [4, 4], layers: 2, notes: [{ text: "p = 1 — no neighbours", x: 0.75, y: 0.95, w: 1.6, h: 0.25, color: "B58900", align: "center" }],
      lines: ["Output pixel: one ¼° cell (≈ 28 km).", "Receptive field = the same cell; the 3×3 patch is deliberately switched off so only the time axis changes.", "Sheets: the 24–25 ocean channels (T, S, SSH, wind, …)."] },
    time: { window: [0.1, 0.9], windowLabel: "12 (or 24) months → one z", axisTicks: seq(12, 0.1, 0.9),
      rows: [
        { label: "monthly means, month-of-year encoded", ticks: seq(12, 0.1, 0.9), style: "bar" },
        { label: "Arm B: two random date-subsamples", ticks: [0.14, 0.27, 0.47, 0.6, 0.8], size: 0.07, color: "B85C1E" },
      ],
      lines: ["A period embedding of the ocean state, TESSERA-style, instead of a snapshot of one monthly mean.", "Moves our point on the space × time map from 'single snapshot' to '1 year folded'."] },
    out: { dim: "64-D", lines: ["per ¼° pixel per year", "(or rolling 12-month window)"] }
  },
  "The honest framing: the 1×1 spatial control I suggested in the first draft already exists in the paper (attribution matrix). What TESSERA genuinely adds is the temporal fold — a year of one pixel into one vector — plus a non-reconstructive objective and int8 quantisation-aware training (the paper's own 8-level input-alphabet finding points the same way). Arm A is cheap because the token budget matches the month-block codec; Arm B reuses the encoder. Both are ablations of the stage-1 codec and leave stage 2 untouched.",
  10.5
);



// small timeline that defines the lag symbol ℓ:  t−ℓ (input) … t (now) … t+Δt (target)
function drawLagTimeline(slide, x, y, w, opts) {
  opts = opts || {};
  const fs = opts.fs || 8.5;
  const xIn = x + 0.45, xNow = x + w * 0.6, xTg = x + w - 0.3, yAx = y + 0.42;
  slide.addShape(pres.shapes.LINE, { x, y: yAx, w, h: 0, line: { color: INK, width: 1, endArrowType: "triangle" } });
  [[xIn, "t − ℓ", "input observed", TIME], [xNow, "t", "now", NAVY], [xTg, "t + Δt", "target", EMB]].forEach(([tx, lab, sub, col]) => {
    slide.addShape(pres.shapes.OVAL, { x: tx - 0.06, y: yAx - 0.06, w: 0.12, h: 0.12, fill: { color: col }, line: { width: 0 } });
    slide.addText([{ text: lab, options: { bold: true, color: col, breakLine: true } }, { text: sub, options: { color: MUTED, fontSize: fs - 1 } }], { x: tx - 0.7, y: yAx + 0.08, w: 1.4, h: 0.4, fontFace: FONT_B, fontSize: fs, align: "center", valign: "top", isTextBox: true, margin: 0 });
  });
  // braces above: ℓ (input→now), Δt (now→target), and Δt+ℓ spanning both
  const brace = (x1, x2, yy, lab, col) => {
    slide.addShape(pres.shapes.LINE, { x: x1, y: yy, w: x2 - x1, h: 0, line: { color: col, width: 1 } });
    slide.addShape(pres.shapes.LINE, { x: x1, y: yy, w: 0, h: 0.07, line: { color: col, width: 1 } });
    slide.addShape(pres.shapes.LINE, { x: x2, y: yy, w: 0, h: 0.07, line: { color: col, width: 1 } });
    slide.addText(lab, { x: (x1 + x2) / 2 - 1.0, y: yy - 0.2, w: 2.0, h: 0.18, fontFace: FONT_B, fontSize: fs, bold: true, color: col, align: "center", isTextBox: true, margin: 0 });
  };
  brace(xIn, xNow, yAx - 0.12, "ℓ  (lag)", TIME);
  brace(xNow, xTg, yAx - 0.12, "Δt  (step)", EMB);
  brace(xIn, xTg, yAx - 0.36, "Δt + ℓ  = time the signal has to travel", NAVY);
}

// 12c–12d. Velocity-specific dependency cones (proposal from Yannick, 31 Aug)
{
  const s = pres.addSlide(); s.background = { color: WHITE };
  title(s, "A velocity-specific dependency cone per channel", "What can influence one pixel Δt ahead is bounded by how fast each process propagates (v) and how long it remembers (τ): Δx ≤ v·Δt, Δt ≤ τ");
  // log-log plot: x = how far back / ahead in time (days), y = distance (km)
  const px = 0.9, py = 1.75, pw = 7.0, ph = 4.3;
  const tmin = 1, tmax = 3650, xmin = 10, xmax = 20000;
  const TX = d => px + (Math.log10(d) - Math.log10(tmin)) / (Math.log10(tmax) - Math.log10(tmin)) * pw;
  const XY = km => py + ph - (Math.log10(km) - Math.log10(xmin)) / (Math.log10(xmax) - Math.log10(xmin)) * ph;
  s.addShape(pres.shapes.RECTANGLE, { x: px, y: py, w: pw, h: ph, fill: { color: "FAFBFD" }, line: { color: GRIDLINE, width: 0.75 } });
  [[1, "1 day"], [10, "10 d"], [30, "1 month"], [365, "1 year"], [3650, "10 yr"]].forEach(([d, l]) => {
    s.addShape(pres.shapes.LINE, { x: TX(d), y: py, w: 0, h: ph, line: { color: GRIDLINE, width: 0.5, dashType: "dash" } });
    s.addText(l, { x: TX(d) - 0.5, y: py + ph + 0.04, w: 1.0, h: 0.22, fontFace: FONT_B, fontSize: 9, color: MUTED, align: "center", isTextBox: true, margin: 0 });
  });
  [[10, "10 km"], [100, "100 km"], [1000, "1,000 km"], [10000, "10,000 km"]].forEach(([k, l]) => {
    s.addShape(pres.shapes.LINE, { x: px, y: XY(k), w: pw, h: 0, line: { color: GRIDLINE, width: 0.5, dashType: "dash" } });
    s.addText(l, { x: 0.05, y: XY(k) - 0.11, w: px - 0.1, h: 0.22, fontFace: FONT_B, fontSize: 9, color: MUTED, align: "right", isTextBox: true, margin: 0 });
  });
  s.addText("time separation Δt (log)", { x: px, y: py + ph + 0.28, w: pw, h: 0.22, fontFace: FONT_B, fontSize: 9.5, bold: true, color: TIME, align: "center", isTextBox: true, margin: 0 });
  s.addText("distance Δx (log)", { x: 0.05, y: py - 0.32, w: 2.5, h: 0.22, fontFace: FONT_B, fontSize: 9.5, bold: true, color: SPACE, isTextBox: true, margin: 0 });
  // region for a process: v in km/day, tau in days
  function cone(v, tau, color, transp, dash) {
    const t0 = Math.max(tmin, xmin / v);          // where the diagonal enters the plot from below
    const xTop = Math.min(v * tau, xmax);
    const tTop = Math.min(tau, xmax / v);         // where the diagonal hits the cap
    const pts = [];
    const P = (t, k) => ({ x: TX(t) - px, y: XY(k) - py });
    if (t0 > tmin) pts.push(P(tmin, xmin)); else pts.push(P(tmin, v * tmin));
    pts.push(P(t0, Math.max(xmin, v * t0)));
    pts.push(P(tTop, xTop));
    if (tTop < tau) pts.push(P(tau, xmax));
    pts.push(P(tau, xmin));
    if (t0 > tmin) { /* already started at (tmin,xmin) */ } else pts.push(P(tmin, xmin));
    pts.push({ close: true });
    s.addShape(pres.shapes.CUSTOM_GEOMETRY, { x: px, y: py, w: pw, h: ph, points: pts, fill: { color, transparency: transp }, line: { color, width: 1.25, dashType: dash || "solid" } });
  }
  // processes (illustrative, order-of-magnitude)
  cone(864, 10, TIME, 70);          // synoptic atmosphere: 10 m/s, ~10 d memory
  cone(8.6, 240, SPACE, 65);        // surface currents / mixed layer: 0.1 m/s, ~8 months
  cone(2.6, 3650, "1C7293", 78);    // interior / Rossby: 0.03 m/s, ~10 yr
  // labels
  s.addText([{ text: "ATMOSPHERE (wind, SLP)", options: { bold: true, breakLine: true } }, { text: "v ≈ 10 m/s · τ ≈ 10 d → wide, shallow", options: {} }], { x: TX(1.15), y: XY(19500), w: 2.6, h: 0.42, fontFace: FONT_B, fontSize: 8.5, color: "B85C1E", isTextBox: true, margin: 2 });
  s.addText([{ text: "SURFACE OCEAN (SST, mixed layer)", options: { bold: true, breakLine: true } }, { text: "v ≈ 0.1 m/s · τ ≈ 8 mo → narrow, months deep", options: {} }], { x: TX(12), y: XY(600) - 0.02, w: 2.6, h: 0.45, fontFace: FONT_B, fontSize: 8.5, color: SPACE, fill: { color: WHITE, transparency: 12 }, isTextBox: true, margin: 2 });
  s.addText([{ text: "INTERIOR (T, S below mixed layer)", options: { bold: true, breakLine: true } }, { text: "v ≈ 0.03 m/s · τ ≈ years → narrow, very deep", options: {} }], { x: TX(190), y: XY(60) - 0.02, w: 2.15, h: 0.45, fontFace: FONT_B, fontSize: 8.5, color: "1C7293", fill: { color: WHITE, transparency: 12 }, isTextBox: true, margin: 2 });
  // fast waveguides line (Kelvin / boundary, 2.5 m/s) dashed
  { const v = 216; s.addShape(pres.shapes.LINE, { x: TX(1), y: XY(Math.min(v * 92, xmax)), w: TX(92) - TX(1), h: XY(v * 1) - XY(Math.min(v * 92, xmax)), flipV: true, line: { color: MUTED, width: 1, dashType: "sysDash" } });
    s.addText("equatorial / coastal Kelvin waves ≈ 2.5 m/s (anisotropic, along waveguides)", { x: TX(1.1), y: XY(130), w: 3.75, h: 0.22, fontFace: FONT_B, fontSize: 8, italic: true, color: MUTED, fill: { color: WHITE, transparency: 12 }, isTextBox: true, margin: 2 }); }
  // our stencils
  const mark = (t, k, col, lab, dx, dy, w) => {
    s.addShape(pres.shapes.OVAL, { x: TX(t) - 0.1, y: XY(k) - 0.1, w: 0.2, h: 0.2, fill: { color: col }, line: { color: WHITE, width: 1.25 } });
    s.addText(lab, { x: TX(t) + dx, y: XY(k) + dy, w: w || 2.6, h: 0.4, fontFace: FONT_B, fontSize: 8, bold: true, color: col, fill: { color: WHITE, transparency: 12 }, isTextBox: true, margin: 2 });
  };
  mark(30, 4444, EMB, "sunflower-89 @ 4444 km, monthly step\n= 1.7 m/s per step", 0.15, -0.1);
  mark(30, 222, EMB, "ring-8 @ 222 km, monthly step = 0.085 m/s", 0.15, 0.02, 2.8);
  mark(5, 4444, "9D82CD", "pentad codec, same reach\n= 10 m/s per step", -0.55, 0.12, 1.5);
  s.addShape(pres.shapes.LINE, { x: TX(730), y: py, w: 0, h: XY(200) - py, line: { color: EMB, width: 1.25, dashType: "dash" } });
  s.addText("history the stage-2\nmodel sees (24 mo)", { x: TX(730) - 1.35, y: py + 0.05, w: 1.3, h: 0.4, align: "right", fontFace: FONT_B, fontSize: 8, bold: true, color: EMB, isTextBox: true, margin: 0 });
  // right panel: the model
  const rx = 8.3, rw = 4.45;
  s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x: rx, y: 1.6, w: rw, h: 5.15, fill: { color: PALE }, line: { width: 0 }, rectRadius: 0.08 });
  s.addText([
    { text: "The model", options: { bold: true, fontSize: 13, color: NAVY, breakLine: true } },
    { text: "For a process p with signal speed vₚ and memory τₚ, the state at (x, t+Δt) can depend on the past only inside", options: { breakLine: true } },
    { text: "Cₚ(Δt) = { (Δx, ℓ) : Δx ≤ vₚ (Δt + ℓ),  Δt + ℓ ≤ τₚ }", options: { bold: true, color: NAVY, breakLine: true } },
    { text: "Δt is the forecast step (we predict t+Δt from data up to now, t); ℓ is the lag of an input — it was observed at t−ℓ, so it sits Δt+ℓ before the target (ℓ = 0 is the newest input). Δx ≤ v·(Δt+ℓ) is the domain of dependence (the CFL / light-cone argument); Δt+ℓ ≤ τ says that beyond the process's own predictability its history is noise, not signal.", options: { breakLine: true } },
    { text: " ", options: { fontSize: 5, breakLine: true } },
    { text: "The union rule", options: { bold: true, fontSize: 13, color: NAVY, breakLine: true } },
    { text: "A channel c is forced by several processes, so its cone is the union: C_c = ∪ₚ∈P(c) Cₚ. Temperature is forced by the atmosphere (fluxes) and by currents (advection), so C_SST = C_atm ∪ C_ocean — wide-and-shallow plus narrow-and-deep, which is not a cone but an L-shape in (Δx, Δt).", options: { breakLine: true } },
    { text: " ", options: { fontSize: 5, breakLine: true } },
    { text: "Where our stencils sit", options: { bold: true, fontSize: 13, color: NAVY, breakLine: true } },
    { text: "A stencil is a cylinder — the same reach at every lag. ring-8 @ 222 km per month is an ocean-advection speed (0.085 m/s); sunflower-89 @ 4444 km is 1.7 m/s, the speed of the fast ocean waveguides and inside the atmosphere's cone at Δt = 1 month. The rolled forecast rebuilds the cone recursively (dependency-cone note, §1), so reach-per-step × horizon is what the target finally sees.", options: {} },
  ], { x: rx + 0.2, y: 1.7, w: rw - 0.4, h: 5.0, fontFace: FONT_B, fontSize: 9.5, color: INK, valign: "top", isTextBox: true, margin: 0 });
  // notation strip
  s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x: 0.6, y: 6.62, w: 7.6, h: 0.4, fill: { color: "F4F6F9" }, line: { width: 0 }, rectRadius: 0.06 });
  s.addText([{ text: "Notation  ", options: { bold: true, color: NAVY } }, { text: "t = now (last observed step) · Δt = forecast step · ℓ = lag of an input (observed at t−ℓ; ℓ = 0 newest) · both in months here.", options: {} }], { x: 0.7, y: 6.62, w: 7.4, h: 0.4, fontFace: FONT_B, fontSize: 8.5, color: INK, valign: "middle", isTextBox: true, margin: 0 });
  footer(s);
  s.addNotes("Yannick's proposal (31 Aug): the propagation velocity of each channel sets a dependency cone beyond which inputs cannot matter; temperature inherits the union of the wind and current cones. Formalised here with two parameters per process: signal speed v and memory τ. Velocities are order-of-magnitude illustrations: synoptic systems ~10 m/s with ~10-day predictability; surface currents 0.1–0.3 m/s with mixed-layer memory of months (re-emergence up to a year); interior/gyre and midlatitude Rossby speeds 0.02–0.05 m/s with multi-year memory; equatorial Kelvin ~2.8 m/s and coastal-trapped waves 1–3 m/s along waveguides. The plot is log–log so a constant speed is a straight diagonal.");
}

{
  const s = pres.addSlide(); s.background = { color: WHITE };
  title(s, "What the cone predicts — and how to test it on our stencil", "Per channel group: dominant processes, the useful reach per monthly step, the useful history, and the falsifiable consequence");
  const hdr = (t) => ({ text: t, options: { bold: true, color: WHITE, fill: { color: NAVY }, fontSize: 9.5, valign: "middle" } });
  const c = (t, o) => ({ text: t, options: Object.assign({ fontSize: 8.5, color: INK, valign: "top" }, o || {}) });
  const nm = (t) => ({ text: t, options: { bold: true, color: NAVY, fontSize: 9, valign: "top" } });
  const rows = [
    [hdr("Channel group"), hdr("Processes (v, τ)"), hdr("Useful reach per monthly step"), hdr("Useful history"), hdr("Design consequence")],
    [nm("10 m wind, SLP, fluxes"), c("synoptic advection 10 m/s, τ ≈ 5–10 d; planetary modes (NAO, ENSO teleconnections) τ ≈ months, forced by SST"), c("causal reach in one month = the globe, but τ < Δt: monthly wind is not predictable from its own history — only via slow ocean modes"), c("1 step (its own); the ocean's history through SST"), c("treat as wide, shallow forcing: large stencil, no depth. At pentad Δt ≈ τ the wind becomes partly self-predictable — a pentad-only gain to look for")],
    [nm("SST, mixed-layer T"), c("surface currents 0.1–0.3 m/s (260–800 km/mo); air–sea forcing (atmospheric cone); mixed-layer memory τ ≈ 3–12 mo"), c("advective 300–800 km, plus the atmospheric cone via forcing ≈ synoptic correlation length 1,000–3,000 km"), c("several months (re-emergence up to a year)"), c("union: needs BOTH the wide wind stencil and months of history — the one group that justifies the full cylinder")],
    [nm("Interior T, S (below ML)"), c("gyre advection 0.02–0.1 m/s (50–260 km/mo); midlatitude Rossby 0.02–0.05 m/s; τ ≈ years–decades"), c("50–300 km — a 3 × 3 patch at ¼° already covers it; 4444 km is 15–90× too wide for one step"), c("24 months and more; skill should keep rising with history"), c("narrow, deep: small reach, long history. Reach can be cut with no loss — the concrete answer to the open question in the cone note (§5)")],
    [nm("SSH"), c("baroclinic Rossby 0.03 m/s (mid-lat) → ~0.3 m/s (10°); Kelvin / coastal waves 1–3 m/s along waveguides; barotropic adjustment in hours (averaged out at monthly Δt)"), c("interior: hundreds of km; along coasts and the equator: thousands of km in one step"), c("months"), c("anisotropic: an along-boundary / along-equator arm on the stencil, narrow elsewhere — a shape the sunflower approximates by brute force")],
    [nm("Transport target (RAPID 26°N)"), c("thermal wind = east–west density difference across the basin; set by interior Rossby adjustment (yrs) and by fast boundary waves (weeks)"), c("the whole section, both boundaries"), c("years for the interior arm; the current step for the boundary arm"), c("why the pooled ridge failed and the unpooled attention head worked (paper §ladder): the target's cone is the section, not its mean")],
  ];
  s.addTable(rows, { x: 0.6, y: 1.5, w: 12.1, colW: [1.6, 3.0, 2.6, 1.7, 3.2], fontFace: FONT_B, border: { type: "solid", color: GRIDLINE, pt: 0.5 }, rowH: [0.32, 0.75, 0.7, 0.7, 0.7, 0.62], margin: 0.05, autoPage: false });
  // predictions & caveats
  s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x: 0.6, y: 5.45, w: 6.0, h: 1.5, fill: { color: PALE }, line: { width: 0 }, rectRadius: 0.08 });
  s.addText([
    { text: "Four falsifiable predictions (each an ablation we can run)", options: { bold: true, color: NAVY, fontSize: 10, breakLine: true } },
    { text: "1  Group-specific reach: cutting reach to ~500 km for interior T/S slots costs no corridor AUC; cutting it for wind/SST slots does.", options: { breakLine: true } },
    { text: "2  Group-specific history: wind slots beyond lag 1 add nothing; interior slots gain from 24 → 48 months.", options: { breakLine: true } },
    { text: "3  A cone-shaped stencil (reach ∝ Δt + lag) matches the 4444 km cylinder at a fraction of the slots.", options: { breakLine: true } },
    { text: "4  The pentad stage-2 shows one-step wind skill the monthly one cannot, because Δt ≈ τ_atm.", options: {} },
  ], { x: 0.75, y: 5.5, w: 5.7, h: 1.4, fontFace: FONT_B, fontSize: 8.5, color: INK, valign: "top", isTextBox: true, margin: 0 });
  s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x: 6.8, y: 5.45, w: 5.9, h: 1.5, fill: { color: "FBF1E9" }, line: { width: 0 }, rectRadius: 0.08 });
  s.addText([
    { text: "Where the simple picture needs care", options: { bold: true, color: "B85C1E", fontSize: 10, breakLine: true } },
    { text: "v is the fastest relevant signal speed, not the mean flow — waves (Rossby, Kelvin, barotropic) outrun advection. τ is predictability, not causality: the cone is truncated by memory, so wind's 'large area' is bounded by v·τ (~4,000–8,000 km), not by v·horizon. Cones are tilted, not symmetric: upstream along the flow, westward for Rossby waves, along boundaries for coastal waves. Time-averaging to monthly means turns anything with τ < Δt into spatially-correlated forcing — the atmosphere is a stochastic boundary condition at our cadence, not a dynamical state. Eddy diffusion adds only √(K·t) ≈ 50 km/month at K ≈ 10³ m² s⁻¹ — negligible at ¼°.", options: {} },
  ], { x: 6.95, y: 5.5, w: 5.6, h: 1.4, fontFace: FONT_B, fontSize: 8.5, color: INK, valign: "top", isTextBox: true, margin: 0 });
  footer(s);
  s.addNotes("The table applies the cone model channel group by channel group. Its sharpest consequence is that a single stencil shape for all channels is wrong by construction: wind wants width without depth, the interior wants depth without width, SST wants both. The four predictions are phrased so that the existing evaluator (12-month rolled corridor AUC, frozen gate head) can decide them; prediction 1 is the direct answer to the open 'can reach come down?' question in the dependency-cone note, and it says: for the interior channels yes, for the forced surface channels no. Predictions are hypotheses, not results — nothing here has been run.");
}



// 12d2–12d5. Worked example: the surface current at one pixel (mirrors the Dependency-Cone Explorer artifact)
const DRV = [
  // name, colour, v (m/s), tau (days), Lcorr (km), viaOcean (fast driver integrated by mixed layer), mechanism, how it enters
  { k: "wind", n: "Wind stress", col: "C9922A", v: 10, tau: 10, L: 1500, mech: "Ekman transport & surface drift — responds within ~1 day, locally", enter: "direct forcing; τ < Δt ⇒ lag 0 only, not itself predictable" },
  { k: "air", n: "Air temperature / heat flux", col: "D9512C", v: 0.15, tau: 180, L: 1500, mech: "net heat flux → mixed-layer T → density gradient → geostrophic current", enter: "fast driver integrated by a slow medium: inherits τ_ML ≈ 6 mo and, for old lags, ocean advection speed" },
  { k: "sst", n: "Ocean surface T (mixed layer)", col: "3B6FD4", v: 0.15, tau: 240, L: 0, mech: "horizontal density gradient → thermal wind", enter: "via the gradient ⇒ neighbours needed (≥ 3 × 3); advected at 0.15 m/s" },
  { k: "int", n: "Interior T, S (below ML)", col: "1C7C7C", v: 0.03, tau: 1825, L: 0, mech: "thermal wind integrated from depth; Rossby adjustment", enter: "via the gradient; 0.03 m/s ⇒ 78 km per month — the 3 × 3 patch covers lag 0" },
  { k: "ssh", n: "SSH", col: "6A4C93", v: 0.03, tau: 365, L: 0, mech: "pressure gradient → barotropic geostrophic current", enter: "via the gradient; Rossby 0.03 m/s westward + Kelvin arm 2.5 m/s along coasts (anisotropic)" },
  { k: "own", n: "Own history (u, v at P)", col: "6E8B5E", v: 0.15, tau: 90, L: 0, mech: "momentum persistence, mesoscale eddies (westward drift ~0.03 m/s; eddies live 4+ months but pass through a fixed pixel in weeks–months)", enter: "direct; eddy memory ≈ 3 months" },
];
const MONTH_D = 30, DT_M = 1, NLAG = 24, RCAP = 10000, SUN_R = 4444, SUN_N = 89;
const kmPerMonth = v => v * 86400 * MONTH_D / 1000;
function reachKm(d, lag) { const adv = kmPerMonth(d.v) * (DT_M + lag); return Math.min(d.k === "air" ? d.L + adv : Math.max(adv, d.L), RCAP); }
function lagUseful(d, lag) { return lag === 0 || (DT_M + lag) * MONTH_D <= d.tau; }
function slots(r) { return Math.min(SUN_N, Math.max(9, Math.round(SUN_N * (r / SUN_R) ** 2))); }
function budget(d) { let b = 0; for (let l = 0; l < NLAG; l++) if (lagUseful(d, l)) b += slots(reachKm(d, l)); return b; }
const fmtKm = r => r >= 10000 ? "10,000 (cap)" : Math.round(r).toLocaleString("en-US");

// ---- W1: the target and its drivers
{
  const s = pres.addSlide(); s.background = { color: WHITE };
  title(s, "Worked example — the surface current at one pixel", "Target: (u, v) at a ¼° pixel P, one month ahead. Which drivers, how each one moves the current, and what (v, τ) it carries. Live version: the Dependency-Cone Explorer artifact.");
  const hdr = (t) => ({ text: t, options: { bold: true, color: WHITE, fill: { color: NAVY }, fontSize: 9.5, valign: "middle" } });
  const c = (t, o) => ({ text: t, options: Object.assign({ fontSize: 8.5, color: INK, valign: "top" }, o || {}) });
  const rows = [[hdr("Driver"), hdr("Mechanism → current"), hdr("How it enters the stencil"), hdr("v"), hdr("τ"), hdr("L_corr"), hdr("Useful lags (Δt = 1 mo)"), hdr("Reach: lag 0 → last useful lag")]];
  DRV.forEach(d => {
    const useful = []; for (let l = 0; l < NLAG; l++) if (lagUseful(d, l)) useful.push(l);
    const last = useful[useful.length - 1];
    rows.push([
      { text: d.n, options: { bold: true, color: d.col, fontSize: 9, valign: "top" } }, c(d.mech), c(d.enter),
      c(d.k === "air" ? "10 m/s (atm.)\n0.15 m/s (ocean)" : d.v + " m/s", { align: "center" }),
      c(d.tau >= 365 ? (d.tau / 365).toFixed(0) + " yr" : d.tau >= 30 ? Math.round(d.tau / 30) + " mo" : d.tau + " d", { align: "center" }),
      c(d.L ? d.L.toLocaleString("en-US") + " km" : "—", { align: "center" }),
      c(useful.length === 1 ? "lag 0 only" : `0 – ${last}  (${useful.length} of 24)`, { align: "center" }),
      c(last === 0 ? `${fmtKm(reachKm(d, 0))} km` : `${fmtKm(reachKm(d, 0))} → ${fmtKm(reachKm(d, last))} km`, { align: "center" }),
    ]);
  });
  s.addTable(rows, { x: 0.6, y: 1.5, w: 12.1, colW: [1.6, 2.6, 2.75, 0.85, 0.6, 0.7, 1.2, 1.8], fontFace: FONT_B, border: { type: "solid", color: GRIDLINE, pt: 0.5 }, rowH: [0.35, 0.62, 0.72, 0.6, 0.6, 0.62, 0.55], margin: 0.05, autoPage: false });
  s.addText([
    { text: "Two rules do all the work.  ", options: { bold: true, color: TEAL } },
    { text: "reach(ℓ) = min( max( v·(Δt+ℓ), L_corr ), 10,000 km ) — a signal observed ℓ before now has Δt+ℓ to travel; a field is never less coherent than its own correlation length. A lag is useful iff Δt+ℓ ≤ τ (lag 0 is always kept). One exception: a fast driver acting through a slow medium (air temperature → heat flux → mixed layer) adds instead of maxing — reach = L_corr + v_ocean·(Δt+ℓ) — because the flux pattern is laid down over L_corr and then carried away. Wind is the extreme case: at monthly cadence its memory is shorter than the step, so it enters as one wide sheet of forcing at lag 0 and nothing older carries information about next month's wind. Air temperature is the subtle case: the atmosphere forgets in 10 days, but the mixed layer remembers the heat it delivered for months and carries it along at 0.15 m/s — so the channel inherits the ocean's τ and v.", options: {} },
  ], { x: 0.6, y: 5.75, w: 8.3, h: 1.2, fontFace: FONT_B, fontSize: 9.5, color: INK, valign: "top", isTextBox: true, margin: 0 });
  s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x: 9.1, y: 5.7, w: 3.6, h: 1.3, fill: { color: PALE }, line: { width: 0 }, rectRadius: 0.06 });
  s.addText("Notation: the lag ℓ", { x: 9.2, y: 5.73, w: 3.4, h: 0.22, fontFace: FONT_B, fontSize: 9, bold: true, color: NAVY, isTextBox: true, margin: 0 });
  drawLagTimeline(s, 9.2, 6.12, 3.4, { fs: 8 });
  footer(s);
  s.addNotes("Worked example answering: 'if I chose wind, air and ocean temperature among other driving forces for the current, how would it work?'. Parameters are order-of-magnitude illustrations and are editable in the Dependency-Cone Explorer artifact, which uses exactly these formulas. Month = 30 days.");
}

// ---- W2: wedge chart
{
  const s = pres.addSlide(); s.background = { color: WHITE };
  title(s, "Step 1 — each driver becomes a wedge in (lag, distance)", "x: how long before now the input was observed · y: how far from P it may sit and still matter (log). Bold outline = the union the stencil should follow.");
  const px = 0.9, py = 1.7, pw = 7.6, ph = 4.5;
  const xmin = 0, xmax = 24, ymin = 10, ymax = 20000;
  const X = l => px + (l - xmin) / (xmax - xmin) * pw;
  const Y = km => py + ph - (Math.log10(km) - Math.log10(ymin)) / (Math.log10(ymax) - Math.log10(ymin)) * ph;
  s.addShape(pres.shapes.RECTANGLE, { x: px, y: py, w: pw, h: ph, fill: { color: "FAFBFD" }, line: { color: GRIDLINE, width: 0.75 } });
  [0, 3, 6, 9, 12, 15, 18, 21, 24].forEach(l => { s.addShape(pres.shapes.LINE, { x: X(l), y: py, w: 0, h: ph, line: { color: GRIDLINE, width: 0.5, dashType: "dash" } }); s.addText(String(l), { x: X(l) - 0.3, y: py + ph + 0.04, w: 0.6, h: 0.22, fontFace: FONT_B, fontSize: 9, color: MUTED, align: "center", isTextBox: true, margin: 0 }); });
  [[10, "10 km"], [100, "100 km"], [1000, "1,000 km"], [10000, "10,000 km"]].forEach(([k, l]) => { s.addShape(pres.shapes.LINE, { x: px, y: Y(k), w: pw, h: 0, line: { color: GRIDLINE, width: 0.5, dashType: "dash" } }); s.addText(l, { x: 0.05, y: Y(k) - 0.11, w: px - 0.1, h: 0.22, fontFace: FONT_B, fontSize: 9, color: MUTED, align: "right", isTextBox: true, margin: 0 }); });
  s.addText("lag ℓ (months before now)", { x: px, y: py + ph + 0.28, w: pw, h: 0.22, fontFace: FONT_B, fontSize: 9.5, bold: true, color: TIME, align: "center", isTextBox: true, margin: 0 });
  s.addText("distance from P (log)", { x: 0.05, y: py - 0.32, w: 2.5, h: 0.22, fontFace: FONT_B, fontSize: 9.5, bold: true, color: SPACE, isTextBox: true, margin: 0 });
  // cylinder + patch
  s.addShape(pres.shapes.RECTANGLE, { x: px, y: Y(SUN_R), w: pw, h: Y(ymin) - Y(SUN_R), fill: { type: "none" }, line: { color: MUTED, width: 1, dashType: "dash" } });
  s.addText("cylinder: 89 slots to 4,444 km at all 24 lags", { x: px + pw - 3.2, y: Y(SUN_R) - 0.24, w: 3.15, h: 0.22, fontFace: FONT_B, fontSize: 8, color: MUTED, align: "right", isTextBox: true, margin: 0 });
  s.addShape(pres.shapes.LINE, { x: px, y: Y(84), w: pw, h: 0, line: { color: MUTED, width: 0.75, dashType: "sysDot" } });
  s.addText("3 × 3 patch, 84 km", { x: px + pw - 1.6, y: Y(84) - 0.22, w: 1.55, h: 0.2, fontFace: FONT_B, fontSize: 8, color: MUTED, align: "right", isTextBox: true, margin: 0 });
  // wedges (draw slow/deep first so shallow ones sit on top)
  const order = ["int", "ssh", "sst", "air", "own", "wind"];
  order.forEach(k => {
    const d = DRV.find(x => x.k === k);
    let last = 0; for (let l = 0; l < NLAG; l++) if (lagUseful(d, l)) last = l;
    const pts = [{ x: X(0) - px, y: Y(ymin) - py }];
    for (let l = 0; l <= last; l++) { pts.push({ x: X(l) - px, y: Y(reachKm(d, l)) - py }); pts.push({ x: X(l + 1) - px, y: Y(reachKm(d, l)) - py }); }
    pts.push({ x: X(last + 1) - px, y: Y(ymin) - py }); pts.push({ close: true });
    s.addShape(pres.shapes.CUSTOM_GEOMETRY, { x: px, y: py, w: pw, h: ph, points: pts, fill: { color: d.col, transparency: 72 }, line: { color: d.col, width: 1.25 } });
  });
  // union outline
  const upts = [];
  for (let l = 0; l < NLAG; l++) { let r = 0; DRV.forEach(d => { if (lagUseful(d, l)) r = Math.max(r, reachKm(d, l)); }); upts.push({ x: X(l) - px, y: Y(r) - py }); upts.push({ x: X(l + 1) - px, y: Y(r) - py }); }
  s.addShape(pres.shapes.CUSTOM_GEOMETRY, { x: px, y: py, w: pw, h: ph, points: upts, fill: { type: "none" }, line: { color: NAVY, width: 2.25 } });
  // legend
  let lx = 8.8, ly = 1.6;
  s.addText("Drivers (Δt = 1 month)", { x: lx, y: ly, w: 4, h: 0.3, fontFace: FONT_B, fontSize: 11, bold: true, color: NAVY, isTextBox: true, margin: 0 });
  DRV.forEach((d, i) => {
    const y = ly + 0.4 + i * 0.62;
    s.addShape(pres.shapes.RECTANGLE, { x: lx, y: y + 0.03, w: 0.22, h: 0.22, fill: { color: d.col, transparency: 40 }, line: { color: d.col, width: 1 } });
    let last = 0; for (let l = 0; l < NLAG; l++) if (lagUseful(d, l)) last = l;
    s.addText([{ text: d.n, options: { bold: true, color: d.col, breakLine: true } }, { text: last === 0 ? `lag 0 only · reach capped at 10,000 km` : `lags 0–${last} · reach ${fmtKm(reachKm(d, 0))} → ${fmtKm(reachKm(d, last))} km`, options: { color: INK, fontSize: 8.5 } }],
      { x: lx + 0.32, y, w: 3.6, h: 0.55, fontFace: FONT_B, fontSize: 9.5, valign: "top", isTextBox: true, margin: 0 });
  });
  s.addText([{ text: "Reading. ", options: { bold: true, color: TEAL } }, { text: "The union is an L: a tall thin spike at lag 0 (the atmosphere, wide but instant), a shoulder out to ~6–8 months (mixed-layer T and the integrated heat flux, 2–4,000 km), then a long low tail to 24 months (interior T/S at ≤ 1,900 km). The cylinder over-covers the tail by 2–50× and cannot cover the lag-0 spike at all.", options: {} }],
    { x: lx, y: ly + 0.4 + 6 * 0.62 + 0.1, w: 3.9, h: 1.3, fontFace: FONT_B, fontSize: 8.5, color: INK, valign: "top", isTextBox: true, margin: 0 });
  footer(s);
  s.addNotes("Wedge chart computed from the same formulas as the explorer (reach = min(max(v·(Δt+ℓ), L_corr), 10,000 km); lag useful iff Δt+ℓ ≤ τ). Each lag is drawn as a column [ℓ, ℓ+1] months. The union outline is what a cone-shaped stencil should follow.");
}

// ---- W3: slot map
{
  const s = pres.addSlide(); s.background = { color: WHITE };
  title(s, "Step 2 — the union becomes a per-channel, lag-dependent stencil", "Slots needed at each lag = 89·(reach / 4,444 km)², floored at the 9-slot 3 × 3 patch and capped at 89. Same 89-slot sunflower, different footprint per channel and lag.");
  const px = 3.4, py = 1.7, cw = 8.3 / NLAG, rh = 0.48;
  s.addText("lag ℓ (months) →", { x: 0.6, y: py - 0.3, w: 2.7, h: 0.22, fontFace: FONT_B, fontSize: 9, bold: true, color: TIME, align: "right", isTextBox: true, margin: 0 });
  for (let l = 0; l < NLAG; l += 3) s.addText(String(l), { x: px + l * cw - 0.15, y: py - 0.3, w: 0.4, h: 0.22, fontFace: FONT_B, fontSize: 8, color: MUTED, align: "center", isTextBox: true, margin: 0 });
  let total = 0;
  DRV.forEach((d, i) => {
    const y = py + i * (rh + 0.12);
    const b = budget(d); total += b;
    s.addText([{ text: d.n, options: { bold: true, color: d.col, breakLine: true } }, { text: `${b.toLocaleString("en-US")} slot-lags · ${Math.round(100 * (1 - b / (SUN_N * NLAG)))} % saved`, options: { color: INK, fontSize: 8 } }], { x: 0.6, y, w: 2.7, h: rh, fontFace: FONT_B, fontSize: 9, valign: "middle", isTextBox: true, margin: 0 });
    for (let l = 0; l < NLAG; l++) {
      const on = lagUseful(d, l);
      const n = on ? slots(reachKm(d, l)) : 0;
      const h = on ? Math.max(0.06, rh * n / SUN_N) : 0.03;
      s.addShape(pres.shapes.RECTANGLE, { x: px + l * cw + 0.02, y: y + rh - h, w: cw - 0.04, h, fill: { color: on ? d.col : GRIDLINE, transparency: on ? 15 : 40 }, line: { width: 0 } });
      if (on && (l === 0 || l % 6 === 5 || n !== slots(reachKm(d, Math.max(0, l - 1))) && l < 8)) s.addText(String(n), { x: px + l * cw - 0.1, y: y + rh - h - 0.2, w: cw + 0.2, h: 0.18, fontFace: FONT_B, fontSize: 7, color: d.col, align: "center", isTextBox: true, margin: 0 });
    }
    s.addShape(pres.shapes.LINE, { x: px, y: y + rh, w: NLAG * cw, h: 0, line: { color: GRIDLINE, width: 0.5 } });
  });
  const cyl = SUN_N * NLAG * DRV.length;
  const yT = py + DRV.length * (rh + 0.12) + 0.05;
  s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x: 0.6, y: yT, w: 12.1, h: 0.85, fill: { color: PALE }, line: { width: 0 }, rectRadius: 0.06 });
  s.addText([
    { text: `${total.toLocaleString("en-US")}`, options: { bold: true, fontSize: 20, color: NAVY } }, { text: "  cone-shaped slot-lags for the six channels   vs   ", options: { fontSize: 11, color: INK } },
    { text: `${cyl.toLocaleString("en-US")}`, options: { bold: true, fontSize: 20, color: MUTED } }, { text: `  for the cylinder (89 × 24 × 6)   →   `, options: { fontSize: 11, color: INK } },
    { text: `${(100 * (1 - total / cyl)).toFixed(1)} % fewer`, options: { bold: true, fontSize: 20, color: "1F7A4D" } },
  ], { x: 0.8, y: yT + 0.05, w: 11.7, h: 0.5, fontFace: FONT_B, valign: "middle", isTextBox: true, margin: 0 });
  s.addText("Bar height = slots needed at that lag (max 89). Grey stubs = lags outside the driver's memory: dropped. Numbers label the first lag, every sixth, and the first few changes.", { x: 0.8, y: yT + 0.55, w: 11.7, h: 0.3, fontFace: FONT_B, fontSize: 8.5, italic: true, color: MUTED, isTextBox: true, margin: 0 });
  s.addText([
    { text: "What the shapes say.  ", options: { bold: true, color: TEAL } },
    { text: "Wind is one full-width sheet at lag 0 and nothing else. Air temperature is a wide sheet for six lags (the mixed layer's memory). Surface T is a true cone: 9 slots at lag 0 growing to 44 at lag 7. Interior T/S is a thin 24-lag column at the 9-slot floor for a year and a half before it widens. SSH is a thin 12-lag column (plus the coastal arm the isotropic count ignores). Own history is three short lags. The cylinder spends 2,136 on each of them; the budget frees ~93 % of the slots for the ones that matter — or lets us afford 48-month history for the interior channels at no extra cost.", options: {} },
  ], { x: 0.6, y: yT + 0.95, w: 12.1, h: 0.8, fontFace: FONT_B, fontSize: 8.5, color: INK, valign: "top", isTextBox: true, margin: 0 });
  footer(s);
  s.addNotes("Slot budget per driver computed from the same formulas as the explorer: Σ over useful lags of clamp(round(89·(reach/4444)²), 9, 89). Totals: wind 89, air T 232, surface T 157, interior 238, SSH 108, own history 27 → 851 vs 12,816 (93.4 % saved). These numbers depend on the illustrative (v, τ, L_corr) choices; the explorer lets you move them.");
}

// ---- W4: mechanisms + what is learned vs fixed + ablations
{
  const s = pres.addSlide(); s.background = { color: WHITE };
  title(s, "Step 3 — the physics behind each wedge", "The stencil fixes the support; attention learns the weights. Four ablations on this target decide whether the support is right.");
  const cards = [
    ["Ekman — wind → drift", "C9922A", "Surface stress τ drives a transport ≈ τ/(ρ f) at 90° to the wind within ~1 inertial period (≈ 1 day at 30°). Local, fast, shallow (tens of m). At monthly cadence this is same-step forcing: the wind of month t+1 is unknown, the wind of month t is already in the ocean state."],
    ["Thermal wind — T, S gradients → shear", "3B6FD4", "∂u/∂z = (g/(f ρ₀)) ∂ρ/∂y: horizontal density gradients set the vertical shear of the geostrophic current, integrated from depth. A gradient does not exist inside one pixel — this is why the 3 × 3 patch is the floor for T/S channels and why heat flux matters through its spatial pattern, not its local value."],
    ["Pressure gradient — SSH → surface geostrophy", "6A4C93", "u_g = −(g/f) ∂η/∂y. SSH anomalies arrive as slow westward Rossby waves in the interior (0.03 m/s) and as fast Kelvin/coastal waves along boundaries (2.5 m/s): a thin column with one long arm — the one wedge the isotropic count under-states."],
    ["Advection & eddies — own history", "6E8B5E", "Momentum persists and mesoscale eddies drift westward at ~0.03 m/s. Individual eddies live 4 months and more (Chelton et al. 2011), but at a fixed pixel the velocity decorrelates in weeks to a few months because they pass through — hence τ ≈ 3 months for own history. Upstream cells at 0.15 m/s × (Δt+ℓ) carry what will arrive. Short cone, tilted upstream."],
  ];
  cards.forEach(([h, col, body], i) => {
    const x = 0.6 + i * 3.05;
    s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x, y: 1.55, w: 2.9, h: 2.55, fill: { color: PALE }, line: { color: col, width: 1.25 }, rectRadius: 0.08 });
    s.addText([{ text: h, options: { bold: true, color: col, fontSize: 10.5, breakLine: true } }, { text: body, options: { fontSize: 10, color: INK } }], { x: x + 0.14, y: 1.62, w: 2.62, h: 2.42, fontFace: FONT_B, valign: "top", isTextBox: true, margin: 0 });
  });
  // fixed vs learned
  s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x: 0.6, y: 4.3, w: 5.95, h: 2.55, fill: { color: "F4F6F9" }, line: { width: 0 }, rectRadius: 0.08 });
  s.addText([
    { text: "What the stencil fixes vs what attention learns", options: { bold: true, color: NAVY, fontSize: 11, breakLine: true } },
    { text: "The cone is the support — which (cell, lag, channel) tokens may enter at all. It encodes only two physical facts per driver: how fast its signal travels and how long it is predictable. Everything else — which upstream cell matters this month, how strongly the SSH gradient projects onto u, whether the heat-flux pattern is the one that tilts the density field — is left to the attention weights inside the support. A support that is too small cannot be learned around; one that is too large costs slots and lets noise in. The four ablations to the right test whether the support is right, run with the existing 12-month rolled corridor-AUC evaluator and the frozen gate head.", options: { fontSize: 10.5, color: INK } },
  ], { x: 0.75, y: 4.38, w: 5.65, h: 2.4, fontFace: FONT_B, valign: "top", isTextBox: true, margin: 0 });
  s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x: 6.75, y: 4.3, w: 5.95, h: 2.55, fill: { color: PALE }, line: { width: 0 }, rectRadius: 0.08 });
  s.addText([
    { text: "The four ablations, specialised to this target", options: { bold: true, color: NAVY, fontSize: 11, breakLine: true } },
    { text: "1  Interior T/S slots at 500 km reach instead of 4,444 km: predicted no loss (their wedge never exceeds 1,900 km).", options: { fontSize: 10, breakLine: true } },
    { text: "2  Wind and heat-flux slots restricted to lags 0–1: predicted no loss (τ_atm < Δt); the same cut on surface T: predicted loss (its wedge runs to lag 7).", options: { fontSize: 10, breakLine: true } },
    { text: "3  Cone-shaped stencil (851 slot-lags) vs cylinder (12,816): predicted equal corridor AUC at 7 % of the slots — and headroom for 48-month interior history.", options: { fontSize: 10, breakLine: true } },
    { text: "4  Pentad codec (Δt = 5 d ≈ τ_atm): predicted one-step wind skill that the monthly model cannot show, because wind's own wedge now has lags > 0.", options: { fontSize: 10, breakLine: true } },
    { text: "Two codec seeds minimum; head numbers are never quoted from one. Explore the parameters live in the Dependency-Cone Explorer artifact (same formulas, editable v, τ, L_corr, Δt, anisotropy).", options: { fontSize: 9.5, italic: true, color: MUTED } },
  ], { x: 6.9, y: 4.38, w: 5.65, h: 2.4, fontFace: FONT_B, color: INK, valign: "top", isTextBox: true, margin: 0 });
  footer(s);
  s.addNotes("Closes the worked example. Ekman: transport = τ/(ρ f), spin-up ~ one inertial period. Thermal wind and geostrophy are the two gradient relations that make neighbours mandatory. The 'fixed vs learned' split is the design principle: the stencil is a hard prior on support, not on weights. Predictions are hypotheses; nothing here has been run.");
}

// 12e–12f. Related work for the cone approach and the multi-sphere approach
{
  const s = pres.addSlide(); s.background = { color: WHITE };
  title(s, "Related work I — the dependency cone vs. the literature", "Who has used 'speed × time' to size what a model sees, and who has not (survey Aug 2026)");
  const hdr = (t) => ({ text: t, options: { bold: true, color: WHITE, fill: { color: NAVY }, fontSize: 9.5, valign: "middle" } });
  const c = (t, o) => ({ text: t, options: Object.assign({ fontSize: 7.5, color: INK, valign: "top" }, o || {}) });
  const nm = (t) => ({ text: t, options: { bold: true, color: NAVY, fontSize: 8, valign: "top" } });
  const yes = (t) => c(t, { color: "1F7A4D" });
  const no = (t) => c(t, { color: "A23B3B" });
  const rows = [
    [hdr("Work"), hdr("What it does"), hdr("Shares with the cone idea"), hdr("Does not")],
    [nm("Courant, Friedrichs & Lewy 1928 — domain of dependence"), c("Numerical PDE stability: the numerical stencil must contain the physical domain of dependence |Δx| ≤ v Δt."), yes("The origin of the argument; our 'Δx ≤ v·Δt' is exactly this"), no("No notion of memory τ; about stability, not information")],
    [nm("OceanNet — Chattopadhyay et al. 2024, Sci. Rep. (arXiv 2310.00813)"), c("Regional FNO for Gulf Stream SSH, 4–5-day mean lead, 80 M params; justifies dropping lateral boundary conditions: 'Rossby waves at 40°N ≈ 1 cm/s … eddies travel only 30–120 km from the open boundary' over the horizon."), yes("Explicit wave-speed × time argument in an ocean ML model"), no("Used to shrink a boundary, not to shape a per-channel stencil; single process")],
    [nm("PARADIS 'Learning to Advect' — Pereira et al. 2026 (arXiv 2601.21151)"), c("Neural semi-Lagrangian layer: gathers features at the departure point x_d = x − Δt·u(x); stability becomes a Lipschitz condition on u instead of CFL; a 20-parameter NSL layer moves a tracer a 3-layer U-Net's receptive field cannot."), yes("Input support set by velocity × Δt, learned per point — the anisotropic, tilted cone made operational"), no("One transport field for all variables; no τ truncation; atmosphere only")],
    [nm("ClimODE — Verma et al., ICLR 2024"), c("Neural ODE with the continuity/advection equation; learns a velocity field, but reads inputs with fixed 3×3 convs plus a global-attention branch."), yes("Advection as inductive bias"), no("Local/global mix fixed, not sized by v·Δt per variable")],
    [nm("Limited-area ML weather: Neural-LAM (Oskarsson 2023/24), stretched grid (Nipen 2024), Diffusion-LAM 2025"), c("Boundary forcing zone of a fixed 10 grid nodes (~100 km) at every step, or a smooth global→regional mesh."), yes("A boundary halo is a cone cut by the domain edge — our 'window edge' problem"), no("Halo width is static; not derived from wind speed × lead time")],
    [nm("Global ML weather: Keisler 2022 (9 GNN hops / 6 h), GraphCast (multi-mesh, 16 layers), Pangu (1/3/6/24 h chain), FuXi (0–5/5–10/10–15 d cascade)"), c("Per-step reach is set by hop count or window size; longer horizons handled by bigger steps or horizon-specific models."), yes("Implicit speed bound per step; growing the cone with lead time by changing Δt"), no("Same reach for every channel; no receptive-field-vs-lead-time ablation published")],
    [nm("Per-variable tokenisation: ClimaX 2023, Aurora 2024/25, Prithvi WxC 2024, Stormer"), c("Each variable gets its own embedding / patch tokens, aggregated by attention."), yes("Variable-specific processing as a design pattern"), no("Variable-specific embedding, not variable-specific spatial or temporal extent")],
    [nm("Ocean emulators: Samudra 2024/25 (1°, 19 levels, 5-day step, ConvNeXt, atmosphere prescribed), ACE2-SOM (mirror: ML atmosphere + slab ocean), Xihe, GLONET"), c("One medium is treated as prescribed forcing at a cadence set by the other's dynamics."), yes("'Atmosphere as forcing' at slow cadence — our monthly-Δt inversion in practice"), no("Not stated as a τ-vs-Δt rule; no channel-specific stencils")],
    [nm("Earth 2 dependency-cone note (16 Aug) + this deck"), c("Cone recursion proved bit-equal to a full roll with NaN poisoning; reach 222 vs 4444 km per month measured as 29.5 % vs 100 % of the world ocean after 12 months; now: per-process (v, τ) cones, union rule, channel-group stencils."), yes("—"), c("Open: the τ-truncation + union → per-channel stencil principle was not found stated anywhere in our search; it is a hypothesis to test (four ablations), not a result")],
  ];
  s.addTable(rows, { x: 0.6, y: 1.5, w: 12.1, colW: [2.6, 4.3, 2.7, 2.5], fontFace: FONT_B, border: { type: "solid", color: GRIDLINE, pt: 0.5 }, rowH: [0.3, 0.38, 0.55, 0.6, 0.38, 0.45, 0.5, 0.38, 0.5, 0.6], margin: 0.04, autoPage: false });
  s.addText("Reading. The 'speed × time' half of the idea has two clear precedents — OceanNet's Rossby-speed boundary argument and PARADIS's semi-Lagrangian gather — and the 'one medium as forcing' half is standard practice in ocean emulators. What we did not find is the combination: truncating each process's cone by its own memory, taking the union per channel, and letting that dictate a channel-specific stencil shape. Absence in a one-day search is not absence in the literature; treat it as 'not yet found', and cite OceanNet and PARADIS as the nearest neighbours.", { x: 0.6, y: 6.5, w: 12.1, h: 0.5, fontFace: FONT_B, fontSize: 8, italic: true, color: INK, valign: "top", isTextBox: true, margin: 0 });
  footer(s);
  s.addNotes("Verified directly: OceanNet quote (arXiv 2310.00813 discussion), PARADIS departure-point equation and CFL→Lipschitz statement (arXiv 2601.21151v2), Neural-LAM 10-node boundary, Keisler 9 hops, Samudra configuration. GraphCast/Pangu/FuXi facts from their papers. 'Unverified' items from the agent search were dropped.");
}

{
  const s = pres.addSlide(); s.background = { color: WHITE };
  title(s, "Related work II — the four-sphere plan vs. the literature", "Coupled emulators, observation-native models, 'Earth-system' foundation models, biosphere ML, coupled data assimilation (survey Aug 2026)");
  const hdr = (t) => ({ text: t, options: { bold: true, color: WHITE, fill: { color: NAVY }, fontSize: 9.5, valign: "middle" } });
  const c = (t, o) => ({ text: t, options: Object.assign({ fontSize: 7.5, color: INK, valign: "top" }, o || {}) });
  const nm = (t) => ({ text: t, options: { bold: true, color: NAVY, fontSize: 8, valign: "top" } });
  const yes = (t) => c(t, { color: "1F7A4D" });
  const no = (t) => c(t, { color: "A23B3B" });
  const rows = [
    [hdr("Family · nearest works"), hdr("What they do"), hdr("Shares with our plan"), hdr("Does not")],
    [nm("Coupled emulators — Samudra → SamudrACE (Dheeshjith 2025; Duncan 2026), ACE2-SOM (2025), ACE2-NEMO (2026)"), c("Samudra: 3-D global ocean emulator (T, S, currents at depth, 1°, 5-day step); SamudrACE couples it to Ai2's ML atmosphere ACE2. ACE2-SOM/NEMO couple an ML atmosphere to a slab / dynamical ocean."), yes("SamudrACE is the closest structural analogue: an ocean-interior model first, atmosphere joined later"), no("Emulators of a simulator, not embeddings of observations; no ocean colour, no land biosphere")],
    [nm("AIFS surface ocean — Hahner, Zampieri et al., ECMWF 2026 (arXiv 2604.25559)"), c("One encoder–processor–decoder forecasts atmosphere + SST, SSS, SSH, surface currents, waves, sea ice 'in a component-agnostic way'; slowly evolving ocean/ice fields get larger loss-scaling factors."), yes("Joint multi-sphere state space; explicit handling of the timescale mismatch"), no("Surface ocean only; forecast model, no reusable embedding; no biosphere")],
    [nm("Aurora — Bodnar et al., Nature 2025 ('A foundation model for the Earth system'); Aurora 1.5"), c("One pretrained 3-D Swin backbone, fine-tuned to air quality, ocean waves, cyclones."), yes("One representation, many sphere-specific heads"), no("Atmosphere-centric; no interior ocean, no two-way coupling, no biosphere. Note the title collision")],
    [nm("Observation-native — Aardvark Weather (Allen et al., Nature 2025), GraphDOP (ECMWF 2024/25), 4DVarNet (Fablet et al. 2023/24)"), c("Aardvark/GraphDOP learn forecasts or a latent state directly from raw observations, skipping reanalysis; 4DVarNet learns a variational solver that fuses sparse altimetry + SST into gap-free surface fields."), yes("The recipe for taking Argo profiles, swaths and tracks natively — what 'combine all relevant data' needs"), no("Atmosphere (Aardvark, GraphDOP) or 2-D surface ocean (4DVarNet); no Argo-depth encoder found")],
    [nm("'Earth-system' FMs — ESFM (Ozdemir et al. 2026, arXiv 2605.00850), Copernicus-FM (2025), Prithvi family (EO, WxC, Surya), AlphaEarth (2025)"), c("ESFM: Aurora-style backbone with per-variable tokens over ERA5, CMIP6, sparse MODIS and station data with missingness handling. Copernicus-FM unifies Sentinel sensors incl. S-5P. Prithvi: separate sphere-specific FMs. AlphaEarth: embedding field with ERA5-Land as an input."), yes("Heterogeneous sources in one embedding; missingness as a first-class token (ESFM) — same primitive as our masked / never-measured tokens"), no("All atmosphere- or land-centric; none has the ocean interior or the ocean biosphere as a target")],
    [nm("Ocean biosphere / carbon ML — SOM-FFN (Landschützer), CSIR-ML6 (Gregor 2019), pCO₂-Residual (Bennington 2022), CANYON-B (Sauzède 2017), Granite-Geospatial-Ocean (2025)"), c("Data-driven surface-pCO₂ and carbon-sink maps from SST, SSS, chl, MLD; NN inference of nutrients/carbonate from T, S, O₂ for BGC-Argo; one ocean-colour FM."), yes("Exactly the physics → biology couplings step 2 of our sequence would learn end-to-end"), no("Task models with hand-chosen predictors; no shared representation with the physical state")],
    [nm("Coupled data assimilation — Penny & Hamill 2017 (BAMS); Penny et al. 2019 (JAMES); ECMWF CERA"), c("The canonical statement that atmosphere (days) and ocean (months–years) memories differ by orders of magnitude; strongly vs weakly coupled DA; production systems still weakly coupled."), yes("The multi-timescale problem is known and hard; our cone-per-sphere is a representation-side answer to it"), no("Assimilation, not learned representation")],
    [nm("Platforms & twins — NVIDIA Earth-2, Destination Earth (ECMWF/CMCC/ESA), EU Digital Twin of the Ocean (EDITO)"), c("Generative km-scale weather tooling (NVIDIA); physics-based coupled km-scale twins (DestinE); ocean data/model platform (EDITO)."), yes("Same four-sphere integration goal at production scale"), no("Not learned embeddings; NVIDIA 'Earth-2' is a name collision we must disambiguate")],
  ];
  s.addTable(rows, { x: 0.6, y: 1.5, w: 12.1, colW: [2.7, 4.1, 2.7, 2.6], fontFace: FONT_B, border: { type: "solid", color: GRIDLINE, pt: 0.5 }, rowH: [0.3, 0.58, 0.55, 0.42, 0.58, 0.68, 0.58, 0.48, 0.48], margin: 0.04, autoPage: false });
  s.addText("Reading. The nearest lineage on the physics side is Samudra → SamudrACE (ocean interior first, atmosphere second — but emulating a simulator); on the representation side it is ESFM and Copernicus-FM (heterogeneous sources, missingness tokens — but no ocean). Nobody starts from the ocean interior as an embedding, and nobody joins the ocean biosphere to a physical-state representation. Two names to disambiguate in any write-up: NVIDIA's 'Earth-2' platform and Aurora's 'A foundation model for the Earth system'.", { x: 0.6, y: 6.5, w: 12.1, h: 0.5, fontFace: FONT_B, fontSize: 8, italic: true, color: INK, valign: "top", isTextBox: true, margin: 0 });
  footer(s);
  s.addNotes("Verified directly: AIFS surface-ocean paper (authors, joint model quote, loss-scaling quote). Others from abstracts and the agents' search; ACE2-NEMO author list and FuXi ocean naming were not verifiable and are cited by arXiv id / lineage only.");
}

// 13. Sources
{
  const s = pres.addSlide();
  s.background = { color: WHITE };
  title(s, "Sources", "Papers, model cards and repositories consulted (accessed 31 Aug 2026)");
  const src = [
    ["AlphaEarth", "arxiv.org/abs/2507.22291 · deepmind.google/blog/alphaearth-foundations-helps-map-our-planet-in-unprecedented-detail · developers.google.com/earth-engine/datasets/catalog/GOOGLE_SATELLITE_EMBEDDING_V1_ANNUAL"],
    ["TESSERA", "arxiv.org/abs/2506.20380 · github.com/ucam-eo/tessera · github.com/ucam-eo/geotessera · geotessera.org · CVPR 2026 open-access page"],
    ["OlmoEarth", "arxiv.org/abs/2511.13655 · allenai.org/blog/olmoearth-models · github.com/allenai/olmoearth_pretrain · huggingface.co/blog/allenai/olmoearth-v1-1 · allenai.org/papers/olmoearth-v1-2"],
    ["TerraMind", "arxiv.org/abs/2504.11171 · arxiv.org/abs/2504.11172 (TerraMesh) · github.com/IBM/terramind · huggingface.co/ibm-esa-geospatial · philab.esa.int (tiny/small release)"],
    ["Prithvi-EO 2.0", "arxiv.org/abs/2412.02732 · github.com/NASA-IMPACT/Prithvi-EO-2.0 · huggingface.co/ibm-nasa-geospatial (config.json for 300M/600M) · research.ibm.com/blog/prithvi2-geospatial · science.nasa.gov (in-orbit deployment)"],
    ["Granite-Geospatial-Ocean", "arxiv.org/abs/2509.21273 · github.com/ibm-granite/geospatial (granite-geospatial-ocean: config + prithvi_vit_S3.py) · huggingface.co/ibm-granite/granite-geospatial-ocean · research.ibm.com/blog/oceans-AI-model · hartree.stfc.ac.uk case study (Jul 2026) · pml.ac.uk news"],
    ["Ocean-including models", "OceanSAR-1: huggingface.co/galeio-research/OceanSAR-1 · arxiv.org/abs/2504.06962 · OceanSAR-2: arxiv.org/abs/2601.07392 · Copernicus-FM: arxiv.org/abs/2503.11849 · huggingface.co/datasets/wangyi111/Copernicus-Pretrain · github.com/zhu-xlab/Copernicus-FM · Aurora: microsoft.github.io/aurora · wekeo.copernicus.eu (ocean FM workspace)"],
    ["Benchmarks & head-to-head tables", "OlmoEarth Tables 2, 3, 7 (arxiv.org/abs/2511.13655) · TerraMind Table 6 / PANGAEA (arxiv.org/abs/2504.11171; arxiv.org/abs/2412.04204) · Prithvi-EO 2.0 Tables A1–A4 / GEO-Bench (arxiv.org/abs/2412.02732; arxiv.org/abs/2306.03831) · TESSERA Table 1 (arxiv.org/abs/2506.20380) · GEO-Bench-2 ranks (arxiv.org/abs/2511.15658) · AlphaEarth & TESSERA for LCZ mapping (arxiv.org/abs/2606.20034) · Granite-Ocean Table 1 (arxiv.org/abs/2509.21273) · OceanSAR-1 Tables 1–3 (arxiv.org/abs/2504.06962)"],
    ["Related work", "OceanNet arxiv.org/abs/2310.00813 · PARADIS arxiv.org/abs/2601.21151 · ClimODE arxiv.org/abs/2404.10024 · Neural-LAM arxiv.org/abs/2309.17370 · stretched grid arxiv.org/abs/2409.02891 · Keisler arxiv.org/abs/2202.07575 · GraphCast arxiv.org/abs/2212.12794 · Pangu arxiv.org/abs/2211.02556 · FuXi arxiv.org/abs/2306.12873 · Samudra arxiv.org/abs/2412.03795 · SamudrACE doi 10.1029/2025GL119340 · ACE2-SOM arxiv.org/abs/2412.04418 · ACE2-NEMO arxiv.org/abs/2603.28704 · AIFS surface ocean arxiv.org/abs/2604.25559 · Aurora doi 10.1038/s41586-025-09005-y · Aardvark doi 10.1038/s41586-025-08897-0 · GraphDOP arxiv.org/abs/2412.15687 · 4DVarNet doi 10.1029/2023MS003609 · ESFM arxiv.org/abs/2605.00850 · Penny & Hamill 2017 BAMS · CSIR-ML6 doi 10.5194/gmd-12-5113-2019 · CANYON-B doi 10.3389/fmars.2017.00128"],
    ["Also checked", "Prithvi WxC (arxiv.org/abs/2409.13598) — atmosphere-only, no ocean downstream tasks found; NASA marine-debris work — not Prithvi-based. Earth 2 numbers quoted from paper/paper.tex (attribution matrix, probe ladder)."],
  ];
  const runs = [];
  src.forEach((r, i) => {
    runs.push({ text: r[0] + "  ", options: { bold: true, color: TEAL, fontSize: 11 } });
    runs.push({ text: r[1], options: { color: INK, fontSize: 10, breakLine: i < src.length - 1, paraSpaceAfter: 8 } });
  });
  s.addText(runs, { x: 0.6, y: 1.55, w: 12.1, h: 5.3, fontFace: FONT_B, valign: "top", isTextBox: true, margin: 2 });
  footer(s);
}

const NOTES = require("./notes.js");
if (NOTES.length !== MADE.length) throw new Error(`notes.js has ${NOTES.length} entries but the deck has ${MADE.length} slides`);
MADE.forEach((sl, i) => {
  sl._slideObjects = sl._slideObjects.filter(o => o._type !== "notes");   // drop the short technical notes
  sl.addNotes(NOTES[i]);
});
require("fs").writeFileSync("/home/claude/deck/titles.json", JSON.stringify(MADE.map(sl => ({ title: sl._deckTitle || "Geospatial representation models", sub: sl._deckSub || "" })), null, 1));
pres.writeFile({ fileName: "/home/claude/deck/geospatial-representation-models.pptx" }).then(f => console.log("wrote", f, MADE.length, "slides"));

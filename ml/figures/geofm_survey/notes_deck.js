// Builds a landscape "speaker notes" slide for every slide of the main deck, so that a PDF can
// interleave slide N with notes N. Same fonts and palette as the main deck.
const pptxgen = require("pptxgenjs");
const NOTES = require("./notes.js");
const SOURCES = require("./sources.js");
const TITLES = require("./titles.json");
const pres = new pptxgen();
pres.layout = "LAYOUT_WIDE";

const NAVY = "0B1F3A", INK = "1B2430", MUTED = "6B7A8F", TEAL = "1C7293", PALE = "EEF3F8", WHITE = "FFFFFF";
const FONT_H = "Georgia", FONT_B = "Calibri";

// split a note into two columns of roughly equal length, never breaking inside a paragraph
function twoColumns(note) {
  const paras = note.split(/\n\s*\n/).map(p => p.trim()).filter(Boolean);
  if (paras.length < 2) return [paras, []];
  const total = paras.reduce((a, p) => a + p.length, 0);
  let best = 1, bestDiff = Infinity, acc = 0;
  for (let k = 1; k < paras.length; k++) {          // split after paragraph k-1; pick the most balanced split
    acc += paras[k - 1].length;
    const diff = Math.abs(acc - (total - acc));
    if (diff < bestDiff) { bestDiff = diff; best = k; }
  }
  return [paras.slice(0, best), paras.slice(best)];
}
function runs(paras, fs) {
  return paras.map((p, i) => ({ text: p, options: { fontSize: fs, color: INK, breakLine: i < paras.length - 1, paraSpaceAfter: 8 } }));
}

NOTES.forEach((note, i) => {
  const s = pres.addSlide();
  s.background = { color: WHITE };
  const n = i + 1;
  s.addText(`SPEAKER NOTES  ·  SLIDE ${n}`, { x: 0.6, y: 0.4, w: 8, h: 0.3, fontFace: FONT_B, fontSize: 10, bold: true, color: TEAL, charSpacing: 2, isTextBox: true, margin: 0 });
  s.addText(TITLES[i].title, { x: 0.6, y: 0.7, w: 12.1, h: 0.6, fontFace: FONT_H, fontSize: 22, bold: true, color: NAVY, valign: "middle", isTextBox: true, margin: 0 });
  // body: two columns; font size adapts to length so the longest notes still fit
  // sources strip at the bottom: clickable labels (hyperlinks survive the PDF export); its height follows the number of links
  const src = SOURCES[i];
  const srcChars = src.reduce((a, x) => a + x.t.length + 3, 0);
  const srcLines = Math.max(1, Math.ceil(srcChars / 175));
  const stripH = 0.22 + 0.155 * srcLines;
  const bodyH = 5.45 - stripH - 0.08;
  const len = note.length;
  const shrink = bodyH < 4.9 ? 0.5 : 0;
  const fs = (len > 2600 ? 12 : len > 2100 ? 12.5 : len > 1500 ? 13.5 : 14.5) - shrink;
  const [L, R] = twoColumns(note);
  s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x: 0.6, y: 1.45, w: 12.1, h: bodyH, fill: { color: PALE }, line: { width: 0 }, rectRadius: 0.08 });
  s.addText(runs(L, fs), { x: 0.85, y: 1.6, w: 5.7, h: bodyH - 0.3, fontFace: FONT_B, valign: "top", isTextBox: true, margin: 0 });
  if (R.length) s.addText(runs(R, fs), { x: 6.8, y: 1.6, w: 5.7, h: bodyH - 0.3, fontFace: FONT_B, valign: "top", isTextBox: true, margin: 0 });
  const sy = 1.45 + bodyH + 0.08;
  s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x: 0.6, y: sy, w: 12.1, h: stripH, fill: { color: WHITE }, line: { color: "B8C4D3", width: 0.75 }, rectRadius: 0.08 });
  const srcRuns = [{ text: "Sources  ", options: { bold: true, color: NAVY, fontSize: 8.5 } }];
  src.forEach((x, j) => {
    srcRuns.push({ text: x.t, options: { color: TEAL, fontSize: 8.5, hyperlink: { url: x.u } } });
    if (j < src.length - 1) srcRuns.push({ text: "  ·  ", options: { color: MUTED, fontSize: 8.5 } });
  });
  s.addText(srcRuns, { x: 0.75, y: sy + 0.04, w: 11.8, h: stripH - 0.08, fontFace: FONT_B, valign: "top", isTextBox: true, margin: 0 });
  s.addText(`Representation-model survey · Aug–Sep 2026 · notes for slide ${n}`, { x: 0.6, y: 7.05, w: 8, h: 0.3, fontFace: FONT_B, fontSize: 9, color: MUTED, isTextBox: true, margin: 0 });
});

pres.writeFile({ fileName: "/home/claude/deck/notes-pages.pptx" }).then(() => console.log("wrote notes-pages.pptx", NOTES.length));

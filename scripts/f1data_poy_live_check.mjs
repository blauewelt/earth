#!/usr/bin/env node
// scripts/f1data_poy_live_check.mjs — E-093's pentad-of-year climatology through
// src/f1data.js against the REAL Hub, compared with the reader's own native
// path over the same bins.
//
//     node scripts/f1data_poy_live_check.mjs [group] [--index URL-or-path] [--box w,s,e,n]
//
// For each case: step "poy" for one month's slots over a span of years (some
// with a year left out), against the native bins of each slot — every bin
// whose start falls in that slot, read one by one with step "native" (its
// own year, month and start day) and averaged here with the finite counts.
// Counts must be equal; means within 1e-5 relative to the field's scale (the
// float32 rounding of the running totals is ≤ ~4e-6 relative).
import { createRequire } from "node:module";
import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
const require = createRequire(import.meta.url);
const ROOT = path.dirname(path.dirname(fileURLToPath(import.meta.url)));
const F1 = require(path.join(ROOT, "src/f1data.js"));
const args = process.argv.slice(2);
const group = args.find((a) => !a.startsWith("--") && !/^-?\d/.test(a)) || "g100";
const opt = (k, d) => (args.includes(k) ? args[args.indexOf(k) + 1] : d);
const ixArg = opt("--index", "data/gridded_poy_index.json");
const [bw, bs, be, bn] = opt("--box", "-40,30,-30,40").split(",").map(Number);
const sf = async (url, init) => {
  if (String(url).startsWith("file://")) {
    const p = fileURLToPath(url);
    if (!fs.existsSync(p)) return new Response("not found", { status: 404 });
    return new Response(fs.readFileSync(p), { status: 200 });
  }
  return fetch(url, init);
};
F1.configure({ registries: F1.DEFAULT_REGISTRIES, siteBase: "file://" + ROOT + "/", fetch: sf, gridPoy: ixArg });
const reg = await F1.loadRegistry();
const d = reg.stores.find((x) => x.name === group && x.family === "10");
if (!d) throw new Error("no family-10 store " + group);
if (!d.poyClimatology) throw new Error(group + ": no pentad-of-year climatology offered (" + JSON.stringify(reg.errors) + ")");
const [Y0, Y1] = d.poyClimatology.years;
console.log(group, "poy years", Y0, Y1, "channels", d.channels.map((c) => c.name).join(","));
const chans = [d.channels[0].name, d.channels[d.channels.length - 1].name];
const bbox = { w: bw, e: be, s: bs, n: bn };
const EPOCH = Date.UTC(1982, 0, 1);
const slotOf = (ms) => {
  const t = new Date(ms), y = t.getUTCFullYear();
  const doy = Math.round((Date.UTC(y, t.getUTCMonth(), t.getUTCDate()) - Date.UTC(y, 0, 1)) / 86400000) + 1;
  return [y, Math.min(72, Math.floor((doy - 1) / 5))];
};
const binFirst = d._raw ? d._raw.bin_first : 0;
const nBins = d._raw ? d._raw.n_bins : 0;
const cases = [
  { yearStart: Math.max(Y0, Y1 - 19), yearEnd: Y1, months: [12], excludeYears: [] },          // incl. slot 72
  { yearStart: Y0, yearEnd: Y0 + 10, months: [2], excludeYears: [Y0 + 4] },                   // a year left out, February incl. slot 11
  { yearStart: Y0 + 2, yearEnd: Y0 + 2, months: [7], excludeYears: [] },                      // one year
];
let worst = 0, bad = 0;
for (const c of cases) {
  const base = { family: d.family, store: d.name, channels: chans, bbox, res: "native", ...c };
  const t0 = Date.now();
  const A = await F1.run({ ...base, step: "poy" });
  console.log(`  poy ${c.yearStart}-${c.yearEnd} m${c.months} ex[${c.excludeYears}]: T=${A.time.length} ${A.stats.requests} req ${(A.stats.bytes / 1e6).toFixed(2)} MB ${Date.now() - t0} ms`);
  const HW = A.lat.length * A.lon.length, C = chans.length;
  for (let i = 0; i < A.time.length; i++) {
    const slot = Math.round((A.time[i] * 1000 - Date.UTC(2000, 0, 1)) / 432000000);
    // the slot's bins in the span, natively
    const sum = new Float64Array(C * HW), cnt = new Int32Array(C * HW);
    let nb = 0, req = 0, bytes = 0;
    const t1 = Date.now();
    for (let b = binFirst; b < binFirst + nBins; b++) {
      const ms = EPOCH + b * 432000000;
      const [y, s] = slotOf(ms);
      if (s !== slot || y < c.yearStart || y > c.yearEnd || c.excludeYears.includes(y)) continue;
      const t = new Date(ms);
      const B = await F1.run({ ...base, excludeYears: [], yearStart: y, yearEnd: y, months: [t.getUTCMonth() + 1],
        days: [t.getUTCDate(), t.getUTCDate()], step: "native" });
      if (B.time.length !== 1) throw new Error(`bin ${b}: the native read returned ${B.time.length} frames`);
      req += B.stats.requests; bytes += B.stats.bytes; nb++;
      for (let j = 0; j < C * HW; j++) { const v = B.data[j]; if (v === v) { sum[j] += v; cnt[j]++; } }
    }
    let md = 0, cd = 0, scale = 0, n = 0;
    for (let k = 0; k < C; k++) for (let j = 0; j < HW; j++) {
      const q = k * HW + j, oi = (i * C + k) * HW + j;
      const a = A.data[oi], ca = A.count[oi], cb = cnt[q];
      if (ca !== cb) { cd++; continue; }
      if (cb) { const ref = sum[q] / cb; md = Math.max(md, Math.abs(a - ref)); scale = Math.max(scale, Math.abs(ref)); n++; }
      else if (a === a) cd++;
    }
    const rel = scale ? md / scale : 0;
    worst = Math.max(worst, rel);
    if (cd || rel > 1e-5) bad++;
    console.log(`    slot ${slot}: ${nb} native bins, ${n} cells, max |Δ| ${md.toExponential(2)} (rel ${rel.toExponential(2)}), count mismatches ${cd}; native ${req} req ${(bytes / 1e6).toFixed(2)} MB ${Date.now() - t1} ms`);
  }
}
console.log(bad ? `FAILED: ${bad} slot(s) differ` : `OK — worst relative difference ${worst.toExponential(2)}`);
process.exit(bad ? 1 : 0);

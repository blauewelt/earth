#!/usr/bin/env node
// scripts/f1data_doy_live_check.mjs — E-091's day-of-year climatology through
// src/f1data.js against the REAL Hub, compared with the reader's own native
// path (every native map of that calendar day).
//
//     node scripts/f1data_doy_live_check.mjs [store] [--index URL-or-path]
//
// For each case: step "doy" for one month's days over a span of years, against
// step "all" on the native path with days d..d for each day — the same mean
// computed from every native frame. Counts must be equal; means within 1e-4
// relative to the field's scale.
import { createRequire } from "node:module";
import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
const require = createRequire(import.meta.url);
const ROOT = path.dirname(path.dirname(fileURLToPath(import.meta.url)));
const F1 = require(path.join(ROOT, "src/f1data.js"));
const args = process.argv.slice(2);
const store = args.find((a) => !a.startsWith("--")) || "oisst025d";
const ixArg = args.includes("--index") ? args[args.indexOf("--index") + 1] : "data/gridded_doy_index.json";
// site-relative files are read from the working tree
const sf = async (url, init) => {
  if (String(url).startsWith("file://")) {
    const p = fileURLToPath(url);
    if (!fs.existsSync(p)) return new Response("not found", { status: 404 });
    return new Response(fs.readFileSync(p), { status: 200 });
  }
  return fetch(url, init);
};
F1.configure({ registries: F1.DEFAULT_REGISTRIES, siteBase: "file://" + ROOT + "/", fetch: sf, gridDoy: ixArg });
const reg = await F1.loadRegistry();
const d = reg.stores.find((x) => x.name === store);
if (!d) throw new Error("no store " + store);
if (!d.doyClimatology) throw new Error(store + ": no day-of-year climatology offered (" + JSON.stringify(reg.errors.filter((e) => e.store === store)) + ")");
console.log(store, "doy years", d.doyClimatology.years, "channels", d.channels.map((c) => c.name).join(","));
const ch = [d.channels[d.channels.length > 2 ? Math.floor(d.channels.length / 2) : 0].name];
const bbox = { w: -40, e: -30, s: 30, n: 40 };
const [Y0, Y1] = d.doyClimatology.years;
const cases = [
  { yearStart: Y0, yearEnd: Y0 + 2, months: [2], days: [27, 31], excludeYears: [] },          // from the first year; 29 Feb
  { yearStart: Y0 + 8, yearEnd: Y0 + 12, months: [7], days: [14, 15], excludeYears: [Y0 + 10] },
  { yearStart: Y1 - 3, yearEnd: Y1, months: [1], days: [1, 2], excludeYears: [] },
];
let worst = 0, bad = 0;
for (const c of cases) {
  const base = { family: d.family, store: d.name, channels: ch, bbox, res: "native", ...c };
  const t0 = Date.now();
  const A = await F1.run({ ...base, step: "doy" });
  console.log(`  doy ${c.yearStart}-${c.yearEnd} m${c.months} d${c.days} ex[${c.excludeYears}]: T=${A.time.length} ${A.stats.requests} req ${(A.stats.bytes / 1e6).toFixed(2)} MB ${Date.now() - t0} ms`);
  const HW = A.lat.length * A.lon.length;
  for (let i = 0; i < A.time.length; i++) {
    const dt = new Date(A.time[i] * 1000), day = dt.getUTCDate();
    const t1 = Date.now();
    let B;
    try { B = await F1.run({ ...base, days: [day, day], step: "all", path: "native" }); }
    catch (e) { console.log("    native failed:", e.message); bad++; continue; }
    let md = 0, cd = 0, scale = 0, n = 0;
    for (let j = 0; j < HW; j++) {
      const a = A.data[i * HW + j], b = B.data.length ? B.data[j] : NaN;
      const ca = A.count[i * HW + j], cb = B.count && B.count.length ? B.count[j] : 0;
      if (ca !== cb) cd++;
      if (ca && cb) { md = Math.max(md, Math.abs(a - b)); scale = Math.max(scale, Math.abs(b)); n++; }
      else if ((a === a) !== (b === b)) cd++;
    }
    const rel = scale ? md / scale : 0;
    worst = Math.max(worst, rel);
    if (cd || rel > 1e-4) bad++;
    console.log(`    day ${day}: ${n} cells, max |Δ| ${md.toExponential(2)} (rel ${rel.toExponential(2)}), count mismatches ${cd}; native ${B.stats.requests} req ${(B.stats.bytes / 1e6).toFixed(2)} MB ${Date.now() - t1} ms`);
  }
}
console.log(bad ? `FAILED: ${bad} day(s) differ` : `OK — worst relative difference ${worst.toExponential(2)}`);
process.exit(bad ? 1 : 0);

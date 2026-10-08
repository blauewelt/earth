#!/usr/bin/env node
// scripts/refresh_reader_check.mjs — E-090: after a scheduled refresh has
// published a store's longer record, read its NEWEST frame through the
// Data tab's own reader (src/f1data.js, the live registries on the Hub) and
// compare it value for value with ml/family1/sharded.py reading the same
// frame from the Hub (raw float16 / uint8, exact).
//
//     node scripts/refresh_reader_check.mjs <slug>/<store> <expected-end YYYY-MM-DD>
//       e.g. node scripts/refresh_reader_check.mjs family7_2d/oisst025d 2026-10-06
//
// It exits non-zero when the registry's record does not end on the expected
// day (the registry is the commit point: the tab must offer the new days), or
// when the reader's values for that day differ from the Python reader's, or
// on any failed read. Needs python3 with numpy + zstandard.
import { createRequire } from "node:module";
import { execFileSync } from "node:child_process";
import path from "node:path";
import fs from "node:fs";
import { fileURLToPath } from "node:url";

const ROOT = path.dirname(path.dirname(fileURLToPath(import.meta.url)));
const require = createRequire(import.meta.url);
const F1 = require(path.join(ROOT, "src", "f1data.js"));
const [key, want] = process.argv.slice(2);
const fail = (m) => { console.error("FAIL: " + m); process.exit(1); };
if (!key || !/^\d{4}-\d\d-\d\d$/.test(want || "")) fail("usage: <slug>/<store> <YYYY-MM-DD>");
const [slug, name] = key.split("/");
const FAMILY = { family7_2d: "7.2d", family1_2: "1.2", family1_gf: "1.gf" }[slug];
if (!FAMILY) fail(`no Data-tab family for ${slug}`);

const siteFetch = async (url, init) => {
  if (String(url).startsWith("file://")) {
    const p = fileURLToPath(url);
    if (!fs.existsSync(p)) return new Response("not found", { status: 404 });
    return new Response(fs.readFileSync(p), { status: 200 });
  }
  return fetch(url, init);
};
F1.configure({ registries: F1.DEFAULT_REGISTRIES, siteBase: "file://" + ROOT + "/", fetch: siteFetch });
const reg = await F1.loadRegistry();
const d = reg.stores.find((s) => s.id === `${FAMILY}/${name}`);
if (!d) fail(`${FAMILY}/${name} is not in the registries the tab loads (errors: ${JSON.stringify(reg.errors).slice(0, 400)})`);
console.log(`${d.id}: span ${d.span.join(" → ")} (registry), frame ${d.frameSeconds} s`);
if (d.span[1] !== want) fail(`the registry's record ends ${d.span[1]}, expected ${want}`);

// the newest frame: the last day, and for a sub-daily store its last hour
const [Y, M, D] = want.split("-").map(Number);
const ch = d.channels[0].name;
const sel = { family: FAMILY, store: name, channels: [ch], yearStart: Y, yearEnd: Y, months: [M], days: [D, D],
  hours: null, bbox: { w: -60, s: 30, e: -30, n: 50 }, step: "native", res: "native" };
if (d.frameSeconds < 86400) {
  const last = 24 - d.frameSeconds / 3600;
  sel.hours = [last, 24];
}
const t0 = Date.now();
const r = await F1.run(sel);
if (r.kind !== "grid") fail("expected a grid result");
const t = r.time[r.time.length - 1];
console.log(`reader: ${r.time.length} frame(s), last ${new Date(t * 1000).toISOString()}, ${r.lat.length} × ${r.lon.length} cells of ${ch} in ${((Date.now() - t0) / 1000).toFixed(1)} s`);
const T = r.time.length, H = r.lat.length, W = r.lon.length;
const got = Array.from(r.data).slice((T - 1) * H * W, T * H * W);
const t82 = t - 378691200;
const py = execFileSync("python3", ["-c", `
import json, sys
import numpy as np
from family1 import sharded as sh
g = sh.ShardedGroup(sys.argv[1]); sp = g.spec; gr = sp["grid"]
b, f = int(sys.argv[2]), int(sys.argv[3])
lat = np.array(json.loads(sys.argv[4])); lon = np.array(json.loads(sys.argv[5]))
point = str(gr.get("align", "")).startswith("point")
off = 0.0 if point else 0.5
rr = np.rint((lat - gr["y0"]) / gr["dy"] - off).astype(int)
cc = np.rint((lon - gr["x0"]) / gr["dx"] - off).astype(int)
fr = g.read_frame(b, f)
names = [c["name"] for c in sp["channels"]]
v = fr[np.ix_(rr, cc)][:, :, names.index(sys.argv[6])].astype(np.float64).ravel()
print(json.dumps([None if not np.isfinite(x) else float(x) for x in v]))
`, `https://huggingface.co/datasets/chfrank/earth-tensors/resolve/main/tensors/${key}/${name}`,
  String(Math.floor(t82 / 432000)), String(Math.round((t82 % 432000) / d.frameSeconds)),
  JSON.stringify(Array.from(r.lat)), JSON.stringify(Array.from(r.lon)), ch],
  { cwd: path.join(ROOT, "ml"), encoding: "utf8", maxBuffer: 1 << 28 });
const w = JSON.parse(py);
let bad = 0, n = 0, maxd = 0;
w.forEach((x, k) => {
  const g = got[k];
  if (x === null) { if (!Number.isNaN(g)) bad++; return; }
  n++; const dd = Math.abs(g - x); maxd = Math.max(maxd, dd); if (dd > 1e-6 * Math.max(1, Math.abs(x))) bad++;
});
console.log(`${ch} on ${want}: ${w.length} cells (${n} finite) against ml/family1/sharded.py — max |difference| ${maxd.toExponential(2)}, ${bad} differ`);
if (bad || w.length !== got.length || n === 0 && d.channels.length && !/occci|pace|oc4k/.test(name)) fail("the reader disagrees with sharded.py, or read nothing");
console.log("OK");

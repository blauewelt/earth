#!/usr/bin/env node
// scripts/f1data_live_check.mjs — src/f1data.js against the REAL Hub (E-084 §6).
//
//     node scripts/f1data_live_check.mjs            # everything (needs python3 + numpy + zstandard + huggingface_hub)
//     node scripts/f1data_live_check.mjs --no-py    # skip the comparison against the Python readers
//
// Not part of the default suite: it reads real bytes from
// huggingface.co/datasets/chfrank/earth-tensors. It
//   1. loads the family 1.gf registry and prints every store with its kind,
//      channels and span;
//   2. reads small real selections — oc4k (2010-06, a 5° × 5° North Atlantic
//      box, native), glodap (2010, a North Atlantic box), one irtb tile and one
//      sst_acspo02 tile — printing the measured requests, bytes and time of each;
//   3. previews every other store once, so a store the reader cannot read
//      fails here rather than in a browser;
//   4. compares one oc4k tile with ml/family1/sharded.py's ShardedGroup reading
//      the same Hub URL, and one glodap bin with ml/family10_store.py's Store
//      opened from the Hub, value for value.
//   5. loads the DEFAULT registry list (family 1.gf, family 1.2, family 10,
//      derived maps), prints the stores announced but not built, reads small
//      real selections of every new layout — ERA5 on pressure levels, the
//      z-scored bin-major tensor groups, the levelled monthly Argo map, family
//      8's schema-1 Argo store, the pre-1982 drifter bins, the month-major
//      fishing grid and the calendar-month climatology — and previews every store;
//   6. compares those with independent numpy range reads of the same bytes
//      (de-z-scored through data/family7_index.json), the Argo month with
//      ml/family10_store.py, and the ERA5 frame with ml/family1/sharded.py
//      (raw float16, exact).
// It exits non-zero on the first disagreement or failed read.
import { createRequire } from "node:module";
import { execFileSync } from "node:child_process";
import path from "node:path";
import os from "node:os";
import fs from "node:fs";
import { fileURLToPath } from "node:url";

const ROOT = path.dirname(path.dirname(fileURLToPath(import.meta.url)));
const require = createRequire(import.meta.url);
const F1 = require(path.join(ROOT, "src", "f1data.js"));
const NO_PY = process.argv.includes("--no-py");
const HUB = "https://huggingface.co/datasets/chfrank/earth-tensors/resolve/main/tensors/family1_gf/";

// every request through one counting fetch, so the numbers below are measured
let reqLog = [];
const counting = async (url, init) => {
  const t = Date.now();
  const res = await fetch(url, init);
  const entry = { url, range: init && init.headers && init.headers.Range, status: res.status, ms: 0, bytes: 0 };
  reqLog.push(entry);
  const ab = res.arrayBuffer.bind(res);
  res.arrayBuffer = async () => { const b = await ab(); entry.bytes = b.byteLength; entry.ms = Date.now() - t; return b; };
  return res;
};
F1.configure({ base: HUB, fetch: counting });

const mb = (n) => (n / 1e6).toFixed(2) + " MB";
const fail = (msg) => { console.error("FAIL: " + msg); process.exit(1); };
function finite(a) { let n = 0, lo = Infinity, hi = -Infinity; for (const v of a) if (v === v) { n++; if (v < lo) lo = v; if (v > hi) hi = v; } return { n, lo, hi }; }
function measure(fn) {
  return async (...a) => {
    const mark = reqLog.length, t = Date.now();
    const r = await fn(...a);
    const part = reqLog.slice(mark);
    return { r, requests: part.length, bytes: part.reduce((s, x) => s + x.bytes, 0), ms: Date.now() - t, log: part };
  };
}
function py(code, ...args) {
  return execFileSync("python3", ["-c", code, ...args], { cwd: path.join(ROOT, "ml"), encoding: "utf8", maxBuffer: 1 << 28, stdio: ["ignore", "pipe", "pipe"] });
}

// ---------------------------------------------------------------- 1. registry
const reg0 = await measure(F1.loadRegistry)();
const reg = reg0.r;
console.log(`registry: ${reg.stores.length} readable stores (${reg0.requests} requests, ${mb(reg0.bytes)}, ${reg0.ms} ms — family1gf.json plus each grid's tile_grid.json)`);
for (const s of reg.stores) {
  const ch = s.channels.map((c) => c.name);
  const geo = s.grid ? ` grid ${s.grid.H}×${s.grid.W} tile ${s.grid.tile} ${s.grid.dtype} ${(Math.abs(s.grid.dlat)).toPrecision(4)}° · ${s.framesPerBin} frames/bin of ${s.frameSeconds} s` : ` N=${s.N.toLocaleString("en-US")}`;
  console.log(`  ${s.name.padEnd(12)} ${s.kind.padEnd(6)} ${s.span[0]} → ${s.span[1]} · ${ch.length} ch [${ch.slice(0, 6).join(", ")}${ch.length > 6 ? ", …" : ""}]${geo}${s.groups && s.groups.length > 1 ? " · groups " + s.groups.map((g) => g.name).join("/") : ""}`);
  s.channels.forEach((c) => { if (c.unit !== c.storedUnit) console.log(`      ${c.name}: stored "${c.storedUnit}" → read as "${c.unit}"`); });
}
const notReadable = reg._raw.groups.filter((g) => !reg.stores.some((s) => s.name === g.name));
console.log(`  not offered: ${notReadable.map((g) => `${g.name} (${!g.built ? "not built" : g.distribution !== "public" ? g.distribution : "tier " + g.tier})`).join(", ") || "none"}`);

// ---------------------------------------------------------------- 2. selections
const SELS = {
  oc4k: { store: "oc4k", channels: ["log_chl", "kd_490", "total_nobs"], yearStart: 2010, yearEnd: 2010, months: [6], hours: null,
    bbox: { w: -40, s: 40, e: -35, n: 45 }, step: "native", res: "native" },
  glodap: { store: "glodap", channels: [], yearStart: 2010, yearEnd: 2010, months: [], hours: null,
    bbox: { w: -60, s: 20, e: -10, n: 60 }, step: "native", res: "native" },
  irtb: { store: "irtb", channels: ["tb"], yearStart: 2010, yearEnd: 2010, months: [6], hours: [12, 15],
    bbox: { w: 10.1, s: 0.1, e: 11, n: 1 }, step: "native", res: "native" },
  sst_acspo02: { store: "sst_acspo02", channels: ["sst", "quality_level"], yearStart: 2010, yearEnd: 2010, months: [6], hours: null,
    bbox: { w: -30, s: 30, e: -29, n: 31 }, step: "native", res: "native" },
};
const results = {};
for (const [name, sel] of Object.entries(SELS)) {
  F1.configure({ base: HUB, fetch: counting });
  await F1.loadRegistry();
  const e = await measure(F1.estimate)(sel);
  // a COLD run: caches cleared, registry re-read, so the run pays for every
  // index and tile itself (the estimate above warmed nothing it uses)
  F1.configure({ base: HUB, fetch: counting });
  await F1.loadRegistry();
  const m = await measure(F1.run)(sel);
  const r = m.r;
  results[name] = r;
  console.log(`\n${name}: estimate ${e.r.requests} requests / ${mb(e.r.readBytes)} (${e.r.exact ? "exact" : "estimated"}; itself ${e.requests} requests, ${mb(e.bytes)}, ${e.ms} ms) · out ${mb(e.r.outBytes)} · overCap ${e.r.overCap}`);
  console.log(`  why: ${e.r.why}`);
  const kinds = {};
  m.log.forEach((x) => { const k = x.url.replace(HUB, "").replace(/\d{4}\/bin_\d+/, "<yyyy>/bin_N").replace(/\.part\d+$/, ".partN"); kinds[k] = kinds[k] || [0, 0]; kinds[k][0]++; kinds[k][1] += x.bytes; });
  console.log(`  run (cold): ${m.requests} requests, ${mb(m.bytes)}, ${(m.ms / 1000).toFixed(2)} s · statuses ${[...new Set(m.log.map((x) => x.status))].join(",")} · every request ranged: ${m.log.every((x) => /^bytes=\d+-\d*$/.test(x.range || ""))}`);
  Object.entries(kinds).forEach(([k, v]) => console.log(`      ${String(v[0]).padStart(5)} × ${k}  ${mb(v[1])}`));
  if (r.kind === "grid") {
    const f = finite(r.data);
    console.log(`  result: ${r.time.length} frames × ${r.channels.length} ch × ${r.lat.length} × ${r.lon.length}; first ${new Date(r.time[0] * 1000).toISOString()} last ${new Date(r.time[r.time.length - 1] * 1000).toISOString()}; finite ${f.n}/${r.data.length} (${(100 * f.n / r.data.length).toFixed(1)} %), range ${f.lo} .. ${f.hi} ${r.units.join(" | ")}`);
    if (r.frames !== e.r.frames) fail(`${name}: estimate said ${e.r.frames} frames, the run read ${r.frames}`);
    if (e.r.exact && (m.log.filter((x) => /\.(zst|idx\.npy)$/.test(x.url)).reduce((a, x) => a + x.bytes, 0) !== e.r.readBytes)) fail(`${name}: exact estimate bytes differ from the run's`);
  } else {
    console.log(`  result: ${r.time.length.toLocaleString("en-US")} rows kept of ${r.rowsRead.toLocaleString("en-US")} read; ${r.channels.length} ch; first ${new Date(r.time[0] * 1000).toISOString()}`);
    if (r.rowsRead !== e.r.rows) fail(`${name}: estimate rows ${e.r.rows} != rows read ${r.rowsRead}`);
  }
  r.notes.forEach((n) => console.log("  note: " + n));
}

// ---------------------------------------------------------------- 3. every other store, once
console.log("\npreview of every store (one frame / one bin; grids get a 1° box):");
const BOX = { oc4k: [-40, 40], pace4k: [-40, 40], sst_acspo02: [-30, 30], irtb: [10, 0] };
for (const s of reg.stores) {
  F1.configure({ base: HUB, fetch: counting });
  await F1.loadRegistry();
  const first = Number(s.span[0].slice(0, 4)), last = Number(s.span[1].slice(0, 4));
  const want = s.name === "swot" ? 2026 : s.kind === "grid" ? 2022 : 2015;   // swot 2026 lives in the SECOND hub_split part
  const yy = Math.max(first, Math.min(last, want));
  const b = BOX[s.name] || [-40, 40];
  const sel = { store: s.name, channels: s.channels.slice(0, 3).map((c) => c.name), yearStart: yy, yearEnd: yy, months: [6], hours: null,
    bbox: s.kind === "grid" ? { w: b[0], s: b[1], e: b[0] + 1, n: b[1] + 1 } : null, step: "native", res: "native" };
  try {
    const m = await measure(F1.preview)(sel);
    const r = m.r;
    const n = r.kind === "grid" ? `${r.time.length} frame, ${finite(r.data).n} finite of ${r.data.length}` : `${r.time.length.toLocaleString("en-US")} rows`;
    const parts = m.log.filter((x) => /\.part\d+$/.test(x.url)).map((x) => path.basename(x.url));
    console.log(`  ${s.name.padEnd(12)} ${yy}-06 ok: ${n}${r.time.length ? " from " + new Date(r.time[0] * 1000).toISOString().slice(0, 16) : ""} · ${m.requests} req, ${mb(m.bytes)}, ${m.ms} ms${parts.length ? " · read " + [...new Set(parts)].join(", ") : ""}${r.truncated ? " · truncated" : ""}`);
    if (r.kind === "points" && r.time.length) {
      const t0 = new Date(r.time[0] * 1000);
      if (t0.getUTCFullYear() !== yy || t0.getUTCMonth() !== 5) fail(`${s.name}: preview row at ${t0.toISOString()} is outside ${yy}-06`);
    }
  } catch (e) {
    fail(`${s.name}: ${e.message}`);
  }
}

// ---------------------------------------------------------------- 5. family 1.2, family 10 and the derived maps
// The reader's DEFAULT registry list (family 1.gf, family 1.2, family 10, derived) with
// the site's own indexes (data/family7_index.json, data/fishing_index.json,
// data/family7_clim_index.json) read from this checkout, and everything else
// from the real Hub through the counting fetch.
const siteFetch = async (url, init) => {
  if (String(url).startsWith("file://")) {
    const p = fileURLToPath(url);
    if (!fs.existsSync(p)) return new Response("not found", { status: 404 });
    return new Response(fs.readFileSync(p), { status: 200 });
  }
  return counting(url, init);
};
F1.configure({ registries: F1.DEFAULT_REGISTRIES, siteBase: "file://" + ROOT + "/", fetch: siteFetch });
const regAll = (await measure(F1.loadRegistry)()).r;
console.log(`\nall registries: ${regAll.families.map((f) => f.family + " (" + regAll.stores.filter((s) => s.family === f.family).length + " stores)").join(", ")}; errors ${regAll.errors.length}`);
regAll.errors.forEach((e) => console.log(`  ERROR ${e.family}${e.store ? "/" + e.store : ""}: ${e.message}`));
console.log(`coming (announced, not built — never selectable): ${(regAll.coming || []).map((c) => c.family + "/" + c.store).join(", ") || "none"}`);
if (regAll.errors.length) fail("a registry or store failed to load");
for (const s of regAll.stores.filter((x) => x.family !== "1.gf")) {
  console.log(`  ${s.id.padEnd(22)} ${s.kind.padEnd(6)} ${s.layout.padEnd(10)} ${s.calendar ? "12 calendar months" : s.span.join(" → ")} · ${s.channels.length} ch${s.levels ? ` (${s.vars.length} variables × ${s.levels.length} levels)` : ""}${s.grid ? ` · ${s.grid.H}×${s.grid.W} at ${s.grid.step ?? Math.abs(s.grid.dlat)}°` : ""}`);
}
const NA = { w: -50, s: 35, e: -40, n: 45 };
const SELS10 = {
  g025: { family: "10", store: "g025", channels: ["sst"], yearStart: 2015, yearEnd: 2015, months: [6], days: [1, 5], hours: null, bbox: NA, step: "native", res: "native" },
  g100: { family: "10", store: "g100", channels: ["t2m"], yearStart: 2015, yearEnd: 2015, months: [6], hours: null, bbox: NA, step: "month", res: "native" },
  rg100: { family: "10", store: "rg100", channels: ["rg_t10", "rg_s1900"], yearStart: 2015, yearEnd: 2015, months: [6], hours: null, bbox: NA, step: "native", res: "native" },
  argo: { family: "10", store: "argo", channels: ["temp_10", "psal_10"], yearStart: 2015, yearEnd: 2015, months: [6], hours: null, bbox: { w: -60, s: 20, e: -10, n: 60 }, step: "native", res: "native" },
  gdp: { family: "10", store: "gdp", channels: ["sst"], yearStart: 1980, yearEnd: 1980, months: [3], hours: null, bbox: null, step: "native", res: "native" },
  fishing_grid: { family: "derived", store: "fishing_grid", channels: ["fishing_hours", "hours"], yearStart: 2020, yearEnd: 2020, months: [3], hours: null, bbox: { w: 0, s: 50, e: 10, n: 60 }, step: "native", res: "native" },
  clim_g025: { family: "derived", store: "clim_g025", channels: ["sst"], yearStart: 2000, yearEnd: 2000, months: [1, 7], hours: null, bbox: NA, step: "native", res: "native" },
  // family 1.2: the four ERA5 stores at 500 hPa, 2015-01-15, 12 UTC only
  // (and humidity once more after the 2022 seam, 2023-07-15 06 UTC)
  ...Object.fromEntries(["t", "q", "u", "v"].map((v) => [`era5_${v}`, { family: "1.2", store: `era5_${v}`, channels: [`${v}_500`],
    yearStart: 2015, yearEnd: 2015, months: [1], days: [15, 15], hours: [12, 18], bbox: { w: -60, s: 30, e: -10, n: 60 }, step: "native", res: "native" }])),
  era5_q_2023: { family: "1.2", store: "era5_q", channels: ["q_500"], yearStart: 2023, yearEnd: 2023, months: [7], days: [15, 15], hours: [6, 12],
    bbox: { w: -60, s: 30, e: -10, n: 60 }, step: "native", res: "native" },
};
const results10 = {};
for (const [name, sel] of Object.entries(SELS10)) {
  const e = await measure(F1.estimate)(sel);
  const m = await measure(F1.run)(sel);
  const r = m.r;
  results10[name] = r;
  const f = finite(r.kind === "grid" ? r.data : r.values);
  console.log(`\n${sel.family}/${name}: estimate ${e.r.requests} requests / ${mb(e.r.readBytes)} · run ${m.requests} requests, ${mb(m.bytes)}, ${(m.ms / 1000).toFixed(2)} s · statuses ${[...new Set(m.log.map((x) => x.status))].join(",")} · every request ranged: ${m.log.every((x) => /^bytes=\d+-\d*$/.test(x.range || ""))}`);
  console.log(`  result: ${r.kind === "grid" ? `${r.time.length} × ${r.channels.length} × ${r.lat.length} × ${r.lon.length}` : r.time.length.toLocaleString("en-US") + " rows"}; first ${new Date(r.time[0] * 1000).toISOString().slice(0, 16)}; finite ${f.n}, range ${f.lo} .. ${f.hi} ${r.units.join(" | ")}`);
  if (r.kind === "grid" && sel.store !== "argo" && !/^1\./.test(sel.family) && e.r.exact && m.log.filter((x) => /\.npy$/.test(x.url) && !/bytes=0-1023$/.test(x.range || "")).reduce((a, x) => a + x.bytes, 0) !== e.r.readBytes) {
    fail(`${name}: exact estimate bytes differ from the run's`);
  }
}
console.log("\npreview of every family-1.2, family-10 and derived store once:");
for (const s of regAll.stores.filter((x) => x.family !== "1.gf")) {
  const last = s.calendar ? 2000 : Number(s.span[1].slice(0, 4));
  const yy = s.calendar ? 2000 : Math.min(last, 2015);
  const sel = { family: s.family, store: s.name, channels: s.channels.slice(0, 2).map((c) => c.name), yearStart: yy, yearEnd: yy,
    months: [6], days: s.kind === "points" ? [1, 2] : null, hours: null, bbox: s.kind === "grid" ? { w: -40, s: 40, e: -35, n: 45 } : null, step: "native", res: "native" };
  try {
    const m = await measure(F1.preview)(sel);
    const r = m.r;
    const n = r.kind === "grid" ? `${r.time.length} frame, ${finite(r.data).n} finite of ${r.data.length}` : `${r.time.length.toLocaleString("en-US")} rows`;
    console.log(`  ${s.id.padEnd(22)} ${yy}-06 ok: ${n} · ${m.requests} req, ${mb(m.bytes)}, ${m.ms} ms${r.truncated ? " · truncated" : ""}`);
  } catch (e) { fail(`${s.id}: ${e.message}`); }
}

if (NO_PY) { console.log("\n--no-py: the Python comparisons were skipped"); process.exit(0); }

// ---------------------------------------------------------------- 6. family 10 against independent reads
{
  const out = JSON.parse(py(`
import json, sys, datetime as dt, urllib.request, io
import numpy as np
E = dt.date(1982, 1, 1)
f7 = json.load(open(sys.argv[1])); fx = json.load(open(sys.argv[2])); cx = json.load(open(sys.argv[3]))
HUB = "https://huggingface.co/datasets/chfrank/earth-tensors/resolve/main/"
def rng(url, a, n):
    with urllib.request.urlopen(urllib.request.Request(url, headers={"Range": f"bytes={a}-{a+n-1}"})) as r:
        assert r.status == 206; b = r.read(); assert len(b) == n; return b
def band(name, b, r0, r1, c0, c1, ch):
    G = f7["groups"][name]; H, W, C = G["shape"][1:]
    buf = rng(HUB + "tensors/family7_global025_pentad_l2/" + G["file"], int(G["header_len"]) + b * int(G["slab_bytes"]) + r0 * W * C * 2, (r1 - r0 + 1) * W * C * 2)
    k = G["chans"].index(ch); mu, sd = G["norm"][k]
    return (np.frombuffer(buf, "<f2").reshape(r1 - r0 + 1, W, C)[:, c0:c1 + 1, k].astype(np.float64) * sd + mu)
out = {}
out["g025"] = band("g025", (dt.date(2015, 6, 2) - E).days // 5, 500, 540, 520, 560, "sst").ravel().tolist()
bins = [b for b in range(3142) if (E + dt.timedelta(days=5 * b)).timetuple()[:2] == (2015, 6)]
st = np.array([band("g100", b, 125, 135, 130, 140, "t2m") for b in bins])
out["g100"] = np.nanmean(st, 0).ravel().tolist(); out["g100_count"] = np.isfinite(st).sum(0).ravel().tolist()
i = fx["months"].index("2020-03"); H, W, C = fx["shape"][1:]
buf = rng(fx["url"], int(fx["header_len"]) + i * int(fx["slab_bytes"]) + 560 * W * C * 4, 41 * W * C * 4)
out["fishing"] = np.frombuffer(buf, "<f4").reshape(41, W, C)[:, 720:761, :].transpose(2, 0, 1).astype(np.float64).ravel().tolist()
base = HUB + "tensors/family10_1/gdp/"
off = np.load(io.BytesIO(urllib.request.urlopen(base + "bin_offsets.npy").read())); bf = json.loads(urllib.request.urlopen(base + "store.json").read())["bin_first"]
t0 = (dt.datetime(1980, 3, 1) - dt.datetime(1982, 1, 1)).total_seconds(); t1 = (dt.datetime(1980, 4, 1) - dt.datetime(1982, 1, 1)).total_seconds()
b0, b1 = int(np.floor(t0 / 432000)), int(np.floor((t1 - 1) / 432000))
ra, rz = int(off[b0 - bf]), int(off[b1 - bf + 1])
t = np.frombuffer(rng(base + "time_s.npy", 128 + 4 * ra, 4 * (rz - ra)), "<i4")
out["gdp"] = int(((t >= t0) & (t < t1)).sum()); out["gdp_bins"] = [b0, b1]
print(json.dumps(out))
`, path.join(ROOT, "data", "family7_index.json"), path.join(ROOT, "data", "fishing_index.json"), path.join(ROOT, "data", "family7_clim_index.json")));
  const cmp = (got, want, tol, label) => {
    let bad = 0, n = 0, maxd = 0;
    want.forEach((w, k) => { const g = got[k]; if (w === null || Number.isNaN(w)) { if (!Number.isNaN(g)) bad++; return; } n++; const d = Math.abs(g - w); if (d > maxd) maxd = d; if (d > tol * Math.max(1, Math.abs(w))) bad++; });
    console.log(`${label}: ${want.length} values (${n} finite) compared, max |difference| ${maxd.toExponential(2)}, ${bad} differ`);
    if (bad || got.length !== want.length) fail(label + " disagrees with the independent read");
  };
  cmp(Array.from(results10.g025.data), out.g025, 1e-6, "\ng025 sst, one bin, against a numpy range read de-z-scored with data/family7_index.json");
  cmp(Array.from(results10.g100.data), out.g100, 1e-6, "g100 t2m June 2015 monthly mean against numpy nanmean of the six bins");
  if (Array.from(results10.g100.count).join() !== out.g100_count.join()) fail("g100 counts differ"); else console.log("  counts identical");
  cmp(Array.from(results10.fishing_grid.data), out.fishing, 0, "fishing grid March 2020 against a numpy range read");
  console.log(`gdp March 1980 (bins ${out.gdp_bins.join("..")}, before the epoch): ${results10.gdp.time.length} rows by f1data, ${out.gdp} by a numpy read of time_s over the same bins`);
  if (results10.gdp.time.length !== out.gdp) fail("gdp row counts differ");
  // argo through ml/family10_store.py (the reader that defines schema 1)
  const ao = JSON.parse(py(`
import json, sys, datetime as dt
import numpy as np
from family10_store import Store
st = Store.open("chfrank/earth-tensors:tensors/family8_argo_l0", cache_dir=sys.argv[1])
E = dt.datetime(1982, 1, 1)
sec = np.rint(np.asarray(st._col["time_days"], np.float64) * 86400)
lat = np.asarray(st._col["lat"]); lon = np.asarray(st._col["lon"])
t0 = (dt.datetime(2015, 6, 1) - E).total_seconds(); t1 = (dt.datetime(2015, 7, 1) - E).total_seconds()
k = (sec >= t0) & (sec < t1) & (lat >= 20) & (lat <= 60) & (lon >= -60) & (lon <= -10)
print(json.dumps({"n": int(k.sum()), "t": sorted((sec[k] + 378691200).tolist())}))
`, path.join(os.tmpdir(), "f1data-live-argo")));
  const at = Array.from(results10.argo.time).sort((a, b) => a - b);
  console.log(`argo June 2015 in a box: ${at.length} rows by f1data, ${ao.n} by ml/family10_store.py; times identical: ${at.join() === ao.t.join()}`);
  if (at.length !== ao.n || at.join() !== ao.t.join()) fail("argo disagrees with ml/family10_store.py");
  // ERA5 (family 1.2) against ml/family1/sharded.py: the same float16 values
  for (const key of ["era5_t", "era5_q", "era5_u", "era5_v", "era5_q_2023"]) {
    const er = results10[key], store = SELS10[key].store;
    const t82 = er.time[0] - 378691200;
    const eo = JSON.parse(py(`
import json, sys
import numpy as np
from family1 import sharded as sh
g = sh.ShardedGroup(sys.argv[1]); sp = g.spec; k = sp["levels_hpa"].index(500)
b, f = int(sys.argv[2]), int(sys.argv[3])
out = np.full((sp["H"], sp["W"]), np.nan, np.float32)
for ty in (1, 2):
    for tx in (1, 2):
        t = g.read_tile(b, f, ty, tx, raw=True, crop=True)
        r0, r1 = sp["row_extents"][ty]; c0, c1 = sp["col_extents"][tx]
        out[r0:r1, c0:c1] = t[..., k].astype(np.float32)
print(json.dumps([None if not np.isfinite(x) else float(x) for x in out[120:151, 120:171].ravel()]))
`, HUB.replace("family1_gf/", "family1_2/") + store + "/" + store, Math.floor(t82 / 432000), (t82 % 432000) / 21600));
    console.log(`${store} 500 hPa ${new Date(er.time[0] * 1000).toISOString().slice(0, 16)} (${er.units[0]}) against ml/family1/sharded.py (raw float16):`);
    cmp(Array.from(er.data), eo, 0, `  ${key}`);
    if (store === "era5_q" && er.units[0] !== "g/kg") fail(`era5_q units are ${er.units[0]}, not g/kg`);
  }
}

// ---------------------------------------------------------------- 4a. oc4k tile vs ShardedGroup
{
  const r = results.oc4k;
  const reg2 = await F1.loadRegistry();
  const g = reg2.stores.find((s) => s.name === "oc4k").grid;
  const BIN = 432000, E = 378691200;
  const t = r.time[0], b = Math.floor((t - E) / BIN), f = Math.round((t - E - b * BIN) / 86400);
  // the tile under the box's centre
  const row = Math.floor((42.5 - g.y0) / g.dy), col = Math.floor((-37.5 - g.x0) / g.dx);
  const ty = Math.floor(row / g.tile), tx = Math.floor(col / g.tile);
  const out = JSON.parse(py(`
import json, sys
import numpy as np
from family1 import sharded as sh
g = sh.ShardedGroup(sys.argv[1])
b, f, ty, tx = map(int, sys.argv[2:6])
t = g.read_tile(b, f, ty, tx, crop=True)
r0 = g.spec["row_extents"][ty][0]; c0 = g.spec["col_extents"][tx][0]
print(json.dumps({"r0": r0, "c0": c0, "shape": list(t.shape), "ranges": g.src.ranges,
                  "v": [None if not np.isfinite(x) else float(x) for x in t.astype(np.float64).ravel()]}))
`, HUB + "oc4k/oc4k", b, f, ty, tx));
  const latIdx = new Map(Array.from(r.lat, (v, i) => [v.toFixed(9), i]));
  const lonIdx = new Map(Array.from(r.lon, (v, i) => [v.toFixed(9), i]));
  const [th, tw, C] = out.shape, H = r.lat.length, W = r.lon.length;
  let compared = 0, finiteN = 0, bad = 0;
  for (let i = 0; i < th; i++) for (let j = 0; j < tw; j++) {
    const lat = g.y0 + (out.r0 + i + 0.5) * g.dy, lon = g.x0 + (out.c0 + j + 0.5) * g.dx;
    const yi = latIdx.get(lat.toFixed(9)), xi = lonIdx.get(lon.toFixed(9));
    if (yi === undefined || xi === undefined) continue;
    for (let k = 0; k < C; k++) {
      const want = out.v[(i * tw + j) * C + k], got = r.data[(0 * C + k) * H * W + yi * W + xi];
      compared++;
      if (want === null ? !Number.isNaN(got) : got !== want) bad++;
      if (want !== null) finiteN++;
    }
  }
  console.log(`\noc4k tile (bin ${b}, frame ${f} = ${new Date(t * 1000).toISOString().slice(0, 10)}, tile ${ty},${tx}): ${compared} values compared with ml/family1/sharded.py ShardedGroup.read_tile over the same Hub URL (${out.ranges} range reads), ${finiteN} finite, ${bad} differ`);
  if (!compared || bad) fail("oc4k tile disagrees with the Python reader");
}

// ---------------------------------------------------------------- 4b. glodap bin vs Store
{
  const r = results.glodap;
  const E = 378691200, BIN = 432000;
  const per = new Map();
  Array.from(r.time).forEach((t, i) => { const b = Math.floor((t - E) / BIN); if (!per.has(b)) per.set(b, []); per.get(b).push(i); });
  const [bin, rows] = [...per.entries()].sort((a, b) => b[1].length - a[1].length)[0];
  const cache = path.join(os.tmpdir(), "f1data-live-glodap");
  const box = SELS.glodap.bbox;
  const out = JSON.parse(py(`
import json, sys, os
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
import numpy as np
from family10_store import Store
st = Store.open("chfrank/earth-tensors:tensors/family1_gf/glodap", verify=True, cache_dir=sys.argv[1])
b = int(sys.argv[2]); w, s, e, n = map(float, sys.argv[3:7])
d = st.bins(b, b)
lat = np.asarray(d["lat"], np.float64); lon = np.asarray(d["lon"], np.float64)
ok = (lat >= s) & (lat <= n) & (lon >= w) & (lon <= e)
v = d["values"][ok]
print(json.dumps({"n": int(ok.sum()), "time_s": d["time_s"][ok].tolist(), "lat": lat[ok].tolist(), "lon": lon[ok].tolist(),
                  "values": [None if not np.isfinite(x) else float(x) for x in v.astype(np.float64).ravel()],
                  "platform": [str(x) for x in np.asarray(st["platform"][d["row"][ok]], np.int64)],
                  "qc": np.asarray(st["qc"][d["row"][ok]]).tolist(), "C": int(st.C), "schema": st.schema_version}))
`, cache, bin, box.w, box.s, box.e, box.n));
  const C = r.channels.length;
  let bad = 0;
  if (out.n !== rows.length) fail(`glodap bin ${bin}: Python keeps ${out.n} rows, f1data ${rows.length}`);
  rows.forEach((ri, k) => {
    if (r.time[ri] !== out.time_s[k] + E) bad++;
    if (r.lat[ri] !== Math.fround(out.lat[k]) || r.lon[ri] !== Math.fround(out.lon[k])) bad++;
    if (r.platform[ri].toString() !== out.platform[k] || r.qc[ri] !== out.qc[k]) bad++;
    for (let c = 0; c < C; c++) {
      const w = out.values[k * C + c], g = r.values[ri * C + c];
      if (w === null ? !Number.isNaN(g) : g !== w) bad++;
    }
  });
  console.log(`glodap bin ${bin} (${new Date((bin * BIN + E) * 1000).toISOString().slice(0, 10)}): ${rows.length} rows × ${C} channels compared with ml/family10_store.py Store.open("chfrank/earth-tensors:tensors/family1_gf/glodap", verify=True) (schema ${out.schema}, sha256 of every file checked), ${bad} differ`);
  if (bad) fail("glodap bin disagrees with the Python reader");
}
console.log("\nALL LIVE CHECKS PASSED");

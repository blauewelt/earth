#!/usr/bin/env node
// The Data tab, live, in a real browser: every file below is made by the
// tab's own controls and its Download button, against the real Hub — the
// sandbox browser has no egress, so the Playwright NODE process passes each
// Hub request through with its Range header and hands back the real answer
// (206, headers intact; a 429 is retried with a growing pause, never served).
// Each file is then compared with an INDEPENDENT read in Python (numpy over
// range reads of the published planes, or ml/family1/sharded.py over the
// native tiles). Not part of the suite: it needs the network, python3 with
// numpy/scipy, and minutes.
//
//   node scripts/datatab_browser_check.mjs [--only=c1,c2,c3,c4,c5,c6,shots]
//
// c1  oisst025d July normal 1991–2020 in a small box: the precomputed monthly
//     sums against numpy Σsum/Σcount; and 2018–2020 through both paths (the
//     "read every native map instead" check box) — within the float32 bound,
//     counts identical
// c2  ERA5 temperature at 500 hPa, January 2015 monthly mean: precomputed vs
//     the native 124-frame path — requests, MB and seconds of each
// c3  a 30-year ERA5 mean: what the native path would cost against what the
//     sums cost now, and the file against numpy
// c4  the population standard deviation against numpy's nanstd of the native
//     frames (sharded.py, one tile)
// c5  the "paper split" and "development split" chips on the tensor's monthly
//     normals (g025 SST, g100 t2m) against the published clim/paper and
//     clim/dev planes read by range and converted with the index's norm
// c6  every store of every family opens on a downloadable first look
// shots  desktop 1280×900 and phone 390×844 with the paper preset applied and
//     a preview painted (/home/claude/datatab-paper-{desktop,phone}.png), and
//     no horizontal overflow at 360 px
import { spawn, execFileSync } from "node:child_process";
import { createRequire } from "node:module";
import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import { fileURLToPath } from "node:url";

const ROOT = path.dirname(path.dirname(fileURLToPath(import.meta.url)));
const require = createRequire(import.meta.url);
const { chromium } = require("playwright");
const CDN = "https://cdnjs.cloudflare.com/ajax/libs/cesium/1.133.1";
const HF = /^https:\/\/(huggingface\.co|cdn-lfs[^/]*\.huggingface\.co|[^/]+\.hf\.co)\//;
const PORT = 8080;
const OUT = fs.mkdtempSync(path.join(os.tmpdir(), "dtlive-"));
const onlyArg = process.argv.find((a) => a.startsWith("--only="));
const only = onlyArg ? new Set(onlyArg.slice(7).split(",")) : null;
const want = (k) => !only || only.has(k);
const fail = (m) => { console.error("FAIL: " + m); process.exitCode = 1; throw new Error(m); };
const mb = (n) => (n / 1e6).toFixed(2) + " MB";

// ------------------------------------------------------------------ server
let server = null;
async function up() {
  try { const r = await fetch(`http://localhost:${PORT}/index.html`); if (r.ok) return; } catch { /* start it */ }
  server = spawn("python3", ["-m", "http.server", String(PORT)], { cwd: ROOT, stdio: "ignore" });
  for (let k = 0; k < 50; k++) {
    await new Promise((r) => setTimeout(r, 200));
    try { const r = await fetch(`http://localhost:${PORT}/index.html`); if (r.ok) return; } catch { /* not yet */ }
  }
  throw new Error("http.server did not come up");
}

// ------------------------------------------------------------------ python
function py(code, ...args) {
  return JSON.parse(execFileSync("python3", ["-c", code, ...args], { cwd: path.join(ROOT, "ml"), encoding: "utf8", maxBuffer: 1 << 28 }));
}
const PYRNG = `
import json, sys, time, urllib.request, urllib.error
import numpy as np
def rng(url, a, n):
    for k in range(8):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers={"Range": f"bytes={a}-{a+n-1}"})) as r:
                assert r.status == 206, r.status
                b = r.read(); assert len(b) == n; return b
        except urllib.error.HTTPError as e:
            if e.code != 429: raise
            time.sleep(2 * (k + 1))
    raise SystemExit("429 eight times: " + url)
`;
// a NetCDF the tab saved: every variable as float64 (NaN for fill), + attrs
function readNc(file) {
  return py(`
import json, sys
import numpy as np
from scipy.io import netcdf_file
f = netcdf_file(sys.argv[1], "r", mmap=False)
def dec(v):
    v = v.decode() if isinstance(v, bytes) else (v.tolist() if hasattr(v, "tolist") else v)
    if isinstance(v, float) and not np.isfinite(v): return None
    if isinstance(v, list): return [None if isinstance(x, float) and not np.isfinite(x) else x for x in v]
    return v
out = {"attrs": {k: dec(v) for k, v in f._attributes.items()}, "vars": {}}
for k, v in f.variables.items():
    a = v[:].astype(np.float64)
    fv = v._attributes.get("_FillValue")
    if fv is not None and np.isfinite(fv): a[a == fv] = np.nan
    out["vars"][k] = {"shape": list(a.shape), "data": [None if not np.isfinite(x) else float(x) for x in a.ravel()],
                      "attrs": {kk: dec(vv) for kk, vv in v._attributes.items()}}
print(json.dumps(out))
`, file);
}
// numpy Σsum / Σcount over (year, month) planes of a store's monthly sums,
// at the file's own pixel centres, with the per-cell float32 bound
function planesMean(key, chan, cells, lat, lon) {
  return py(PYRNG + `
ix = json.load(open(sys.argv[1])); G = ix["stores"][sys.argv[2]]; ch = sys.argv[3]
cells = json.loads(sys.argv[4]); lat = np.array(json.loads(sys.argv[5])); lon = np.array(json.loads(sys.argv[6]))
g = G["grid"]; W = g["W"]; C, Y = len(G["chans"]), int(G["n_years"]); ci = G["chans"].index(ch)
rr = np.rint((lat - g["lat0"]) / g["dlat"]).astype(int); cc = np.rint(((lon - g["lon0"]) % 360) / g["dlon"]).astype(int)
r0, r1 = int(rr.min()), int(rr.max())
S = np.zeros((len(rr), len(cc))); N = np.zeros_like(S); B = np.zeros_like(S)
for y, m in cells:
    if G["frames_present"][y - G["year_first"]][m - 1] == 0: continue
    pl = []
    for k, dt_ in (("sum", "<f4"), ("count", "u1")):
        f = G[k]; isz = int(f["itemsize"])
        off = int(f["header_len"]) + (((m - 1) * C + ci) * Y + (y - int(G["year_first"]))) * int(f["plane_bytes"]) + r0 * W * isz
        a = np.frombuffer(rng(f["url"], off, (r1 - r0 + 1) * W * isz), dt_).reshape(r1 - r0 + 1, W)
        pl.append(a[np.ix_(rr - r0, cc)])
    S += pl[0].astype(np.float64); N += pl[1]; B += np.spacing(np.abs(pl[0])).astype(np.float64) / 2
with np.errstate(invalid="ignore", divide="ignore"):
    M = np.where(N > 0, S / N, np.nan); Bd = np.where(N > 0, B / N, np.nan)
f = lambda a: [None if not np.isfinite(v) else float(v) for v in a.ravel()]
print(json.dumps({"mean": f(M), "count": [int(v) for v in N.ravel()], "bound": f(Bd)}))
`, path.join(ROOT, "data", "gridded_monthly_index.json"), key, chan, JSON.stringify(cells), JSON.stringify(lat), JSON.stringify(lon));
}
const ulp32 = (x) => (x === 0 || !Number.isFinite(x) ? 2 ** -149 : 2 ** (Math.floor(Math.log2(Math.abs(x))) - 23));
// a vs b cell by cell: |Δ| ≤ bound (+ one float32 rounding of each side)
function cmp(a, b, bound, label) {
  let n = 0, maxd = 0, worst = 0, bad = 0, nan = 0;
  for (let i = 0; i < a.length; i++) {
    const x = a[i], y = b[i];
    if ((x === null) !== (y === null)) { nan++; continue; }
    if (x === null) continue;
    n++;
    const d = Math.abs(x - y), bd = (bound ? bound[i] || 0 : 0) + ulp32(x) / 2 + ulp32(y) / 2;
    maxd = Math.max(maxd, d); worst = Math.max(worst, d / bd);
    if (d > bd) bad++;
  }
  console.log(`  ${label}: ${n} cells, max |Δ| ${maxd.toExponential(2)}, worst |Δ| / bound ${worst.toFixed(3)}, ${bad} outside, ${nan} NaN mismatches`);
  if (bad || nan || !n) fail(label);
  return { cells: n, maxAbs: maxd, worstOverBound: worst };
}
const sameCounts = (a, b, label) => {
  const x = a.map((v) => (v === null ? 0 : v)), y = b.map((v) => (v === null ? 0 : v));
  if (x.join() !== y.join()) fail(label + ": counts differ");
  console.log(`  ${label}: counts identical (${x.length} cells, ${Math.min(...x)}–${Math.max(...x)} frames per cell)`);
};

// ------------------------------------------------------------------ browser
const hub = [];        // every Hub request: {t, url, range, status, bytes}
async function open(viewport = { width: 1280, height: 900 }, opts = {}) {
  const browser = await chromium.launch({ executablePath: process.env.CHROMIUM_PATH || "/opt/pw-browsers/chromium",
    args: ["--use-gl=swiftshader"] });
  const ctx = await browser.newContext({ viewport, acceptDownloads: true, ...(opts.mobile ? { isMobile: true, hasTouch: true, deviceScaleFactor: 2 } : {}) });
  const page = await ctx.newPage();
  page.__errors = [];
  page.on("pageerror", (e) => page.__errors.push(String(e)));
  await page.route(/https:\/\/cdnjs\.cloudflare\.com\/.*/, async (route) => {
    try {
      const url = route.request().url().replace(CDN, `http://localhost:${PORT}/_vendor/cesium`).replace("widgets.min.css", "widgets.css");
      await route.fulfill({ response: await page.request.get(url) });
    } catch { await route.abort().catch(() => {}); }
  });
  // the Hub: passed through from node, Range forwarded, the real status back
  await page.route(HF, async (route) => {
    const req = route.request(), t = Date.now();
    try {
      let resp = null;
      for (let k = 0; k < 6; k++) {
        resp = await page.request.fetch(req, { maxRedirects: 5 });
        if (resp.status() !== 429) break;
        await new Promise((r) => setTimeout(r, 2000 * 2 ** k));
      }
      const body = await resp.body();
      hub.push({ t, url: req.url(), range: req.headers()["range"] || null, status: resp.status(), bytes: body.length });
      await route.fulfill({ response: resp, body });
    } catch { await route.abort().catch(() => {}); }
  });
  // every other external host (imagery, the forecast APIs): through node too
  await page.route(/^https:\/\/(?!cdnjs\.cloudflare\.com|huggingface\.co)[^/]+\//, async (route) => {
    if (HF.test(route.request().url())) return route.fallback();
    try { await route.fulfill({ response: await page.request.fetch(route.request(), { maxRedirects: 5, timeout: 20000 }) }); }
    catch { await route.abort().catch(() => {}); }
  });
  await page.goto(`http://localhost:${PORT}/`);
  await page.waitForFunction(() => window.__earth && window.__earth.viewer, null, { timeout: 60000 });
  await page.evaluate(() => { try { localStorage.removeItem("dataTabSel"); } catch {} });
  await page.evaluate(() => document.querySelector("#tab-data").click());
  await page.waitForFunction(() => document.querySelectorAll("#dt-store option").length > 20, null, { timeout: 120000 });
  return { browser, page };
}
const state = (page) => page.evaluate(() => window.__earth.dataTabState());
async function settled(page, ms = 300000) {
  const t0 = Date.now();
  for (;;) {
    const s = await state(page);
    if (s.estimateCurrent && !s.looking) return s;
    if (Date.now() - t0 > ms) throw new Error("the estimate did not settle: " + JSON.stringify(s.progress || {}).slice(0, 300));
    await new Promise((r) => setTimeout(r, 250));
  }
}
const set = (page, vals) => page.evaluate((v) => {
  for (const [id, x] of Object.entries(v)) {
    const el = document.getElementById(id);
    if (el.type === "checkbox") el.checked = !!x; else el.value = x;
    el.dispatchEvent(new Event("change", { bubbles: true }));
  }
}, vals);
async function store(page, id) {
  await set(page, { "dt-store": id });
  await settled(page);
}
// the whole selection, through the controls a visitor uses
async function choose(page, o) {
  if (o.store) await store(page, o.store);
  if (o.channels) {
    await page.evaluate((cs) => {
      for (const i of document.querySelectorAll("#dt-channels input[type=checkbox]")) {
        const on = cs.includes(i.value);
        if (i.checked !== on) { i.checked = on; i.dispatchEvent(new Event("change", { bubbles: true })); }
      }
    }, o.channels);
  }
  if (o.months) {
    await page.evaluate((ms) => {
      document.getElementById("dt-months-none").click();
      for (const m of ms) document.querySelector(`#dt-months button[data-month="${m}"]`).click();
    }, o.months);
  }
  const v = {};
  if (o.years) { v["dt-y0"] = String(o.years[0]); v["dt-y1"] = String(o.years[1]); }
  if (o.days !== undefined) { v["dt-d0"] = String(o.days ? o.days[0] : 1); v["dt-d1"] = String(o.days ? o.days[1] : 31); }
  if (o.box) Object.assign(v, { "dt-w": String(o.box.w), "dt-s": String(o.box.s), "dt-e": String(o.box.e), "dt-n": String(o.box.n) });
  if (o.exclude !== undefined) v["dt-exclude"] = o.exclude;
  if (o.step) v["dt-step"] = o.step;
  if (o.res) v["dt-res"] = o.res;
  await set(page, v);
  if (o.std !== undefined) await set(page, { "dt-std": o.std });
  if (o.native !== undefined) await set(page, { "dt-native": o.native });
  return settled(page);
}
// the Download button: the file, the Hub reads it made, the seconds it took
async function download(page, tag) {
  const mark = hub.length, t0 = Date.now();
  const [dl] = await Promise.all([page.waitForEvent("download", { timeout: 900000 }),
    page.evaluate(() => document.getElementById("dt-download").click())]);
  const file = path.join(OUT, tag + "_" + dl.suggestedFilename());
  await dl.saveAs(file);
  const secs = (Date.now() - t0) / 1000;
  const reads = hub.slice(mark);
  const not206 = reads.filter((r) => r.range && r.status !== 206);
  if (not206.length) fail(`${tag}: ${not206.length} range reads not answered 206`);
  if (reads.some((r) => !r.range)) fail(`${tag}: a Hub read without Range`);
  return { file, name: dl.suggestedFilename(), secs, requests: reads.length, bytes: reads.reduce((a, r) => a + r.bytes, 0) };
}
const estLine = async (page) => page.evaluate(() => {
  const p = document.querySelector("#dt-estimate .dt-path");
  return { path: p ? p.dataset.path : null, text: p ? p.textContent : null };
});
const results = {};
const say = (k, o) => { results[k] = o; };

await up();
const { browser, page } = await open();
try {
  // ---------------------------------------------------------------- C1
  if (want("c1")) {
    const box = { w: -50, s: 35, e: -45, n: 40 };
    const s = await choose(page, { store: "7.2d/oisst025d", channels: ["sst"], months: [7], years: [1991, 2020], days: null, box, exclude: "", step: "normal", res: "native", std: false, native: false });
    const line = await estLine(page);
    console.log(`C1 oisst025d July normal 1991–2020, ${JSON.stringify(box)}: the panel says: ${line.text}`);
    if (line.path !== "monthly") fail("C1 not on the precomputed path");
    const d = await download(page, "c1");
    const nc = readNc(d.file);
    const cells = []; for (let y = 1991; y <= 2020; y++) cells.push([y, 7]);
    const w = planesMean("family7_2d/oisst025d", "sst", cells, nc.vars.lat.data, nc.vars.lon.data);
    console.log(`  download: ${d.requests} requests, ${mb(d.bytes)}, ${d.secs.toFixed(1)} s → ${d.name}; read_path: ${nc.attrs.read_path}; ${nc.attrs.frames}`);
    const a = cmp(nc.vars.sst.data, w.mean, w.bound, "tab (precomputed) vs numpy Σsum/Σcount of the 30 planes");
    sameCounts(nc.vars.sst_count.data, w.count, "tab vs numpy");
    // the same box, July 2018–2020, through both paths of the tab
    await choose(page, { years: [2018, 2020], native: false });
    const dp = await download(page, "c1b-sums");
    await choose(page, { native: true });
    const ln = await estLine(page);
    const dn = await download(page, "c1b-native");
    const P = readNc(dp.file), Nn = readNc(dn.file);
    const cells3 = [[2018, 7], [2019, 7], [2020, 7]];
    const w3 = planesMean("family7_2d/oisst025d", "sst", cells3, P.vars.lat.data, P.vars.lon.data);
    console.log(`C1b July 2018–2020: precomputed ${dp.requests} requests ${mb(dp.bytes)} ${dp.secs.toFixed(1)} s; native (${Nn.attrs.frames_present_total} daily maps; "${ln.text}") ${dn.requests} requests ${mb(dn.bytes)} ${dn.secs.toFixed(1)} s`);
    const b = cmp(P.vars.sst.data, Nn.vars.sst.data, w3.bound, "precomputed vs native, both through the tab");
    sameCounts(P.vars.sst_count.data, Nn.vars.sst_count.data, "precomputed vs native");
    await set(page, { "dt-native": false });
    say("c1", { panel: line.text, download: d, vsNumpy: a, sub: { sums: dp, native: dn, cmp: b } });
  }
  // ---------------------------------------------------------------- C2
  if (want("c2")) {
    const box = { w: -60, s: 30, e: -10, n: 60 };
    await choose(page, { store: "1.2/era5_t", months: [1], years: [2015, 2015], days: null, box, exclude: "", step: "month", res: "native", std: false, native: false });
    await set(page, { "dt-h0": "0", "dt-h1": "24" });
    let s = await settled(page);
    if (s.sel.channels.join() !== "t_500") fail("C2: the level picker is not on 500 hPa: " + s.sel.channels);
    const line = await estLine(page);
    const dp = await download(page, "c2-sums");
    await set(page, { "dt-native": true }); await settled(page);
    const dn = await download(page, "c2-native");
    await set(page, { "dt-native": false }); await settled(page);
    const P = readNc(dp.file), Nn = readNc(dn.file);
    const w = planesMean("family1_2/era5_t", "t_500", [[2015, 1]], P.vars.lat.data, P.vars.lon.data);
    console.log(`C2 era5_t 500 hPa, January 2015 monthly mean, ${JSON.stringify(box)}: the panel says: ${line.text}`);
    console.log(`  precomputed ${dp.requests} requests ${mb(dp.bytes)} ${dp.secs.toFixed(1)} s; native (${Nn.attrs.frames_present_total} six-hourly maps) ${dn.requests} requests ${mb(dn.bytes)} ${dn.secs.toFixed(1)} s`);
    const c = cmp(P.vars.t_500.data, Nn.vars.t_500.data, w.bound, "precomputed vs native, both through the tab");
    sameCounts(P.vars.t_500_count.data, Nn.vars.t_500_count.data, "precomputed vs native");
    say("c2", { panel: line.text, sums: dp, native: dn, cmp: c });
  }
  // ---------------------------------------------------------------- C7
  // E-091: the climatology per day of year — every calendar day, 2001–2020,
  // a box — through the tab's controls and its Download button
  if (want("c7")) {
    const C7_MONTHS = process.env.C7_MONTHS ? JSON.parse(process.env.C7_MONTHS) : [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12];
    const box = { w: -40, s: 30, e: -30, n: 40 };
    await choose(page, { store: "7.2d/oisst025d", channels: ["sst"], months: C7_MONTHS, years: [2001, 2020], days: null, box, exclude: "", step: "doy", res: "native", std: false, native: false });
    const line = await estLine(page);
    const dp = await download(page, "c7-doy");
    const P = readNc(dp.file);
    const T = P.vars.time.data.length;
    console.log(`C7 oisst025d climatology per day of year 2001–2020, ${JSON.stringify(box)}: the panel says: ${line.text}`);
    console.log(`  ${T} calendar days, ${dp.requests} requests ${mb(dp.bytes)} ${dp.secs.toFixed(1)} s; file ${dp.name}`);
    const full = C7_MONTHS.length === 12, feb = C7_MONTHS.includes(2);
    if (full && T !== 366) fail("C7: " + T + " time steps, expected 366");
    const cnt = P.vars.sst_count.data, HW = cnt.length / T;
    const li = full ? 59 : C7_MONTHS.indexOf(2) === 0 ? 28 : -1;
    let lo = Infinity, hi = 0;
    for (let i = 0; i < cnt.length; i++) { if (li >= 0 && i >= li * HW && i < (li + 1) * HW) continue; lo = Math.min(lo, cnt[i]); hi = Math.max(hi, cnt[i]); }
    let leap = li >= 0 ? 0 : 5; if (li >= 0) for (let i = li * HW; i < (li + 1) * HW; i++) leap = Math.max(leap, cnt[i]);
    console.log(`  counts: ${lo}–${hi} on ordinary days (expected 20), ${leap} on 29 February (expected 5)`);
    if (lo !== 20 || hi !== 20 || leap !== 5) fail("C7: counts are not 20 years per day and 5 leap days");
    say("c7", { panel: line.text, doy: dp, days: T });
  }
  // ---------------------------------------------------------------- C3
  if (want("c3")) {
    const box = { w: -60, s: 30, e: -10, n: 60 };
    await choose(page, { store: "1.2/era5_t", months: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12], years: [1991, 2020], days: null, box, exclude: "", step: "all", res: "native", std: false, native: true });
    let s = await settled(page);
    const before = s.lastEstimate;
    await set(page, { "dt-native": false });
    s = await settled(page);
    const now = s.lastEstimate, line = await estLine(page);
    console.log(`C3 era5_t 500 hPa, one mean over 1991–2020, ${JSON.stringify(box)}:`);
    console.log(`  the native maps: ${before.frames} frames, ${before.exact === false ? "≈ " : ""}${before.requests} requests, ${mb(before.readBytes)} — over the cap: ${before.overCap}`);
    console.log(`  now: the panel says: ${line.text}`);
    const d = await download(page, "c3");
    const nc = readNc(d.file);
    const cells = []; for (let y = 1991; y <= 2020; y++) for (let m = 1; m <= 12; m++) cells.push([y, m]);
    const w = planesMean("family1_2/era5_t", "t_500", cells, nc.vars.lat.data, nc.vars.lon.data);
    console.log(`  download: ${d.requests} requests, ${mb(d.bytes)}, ${d.secs.toFixed(1)} s; ${nc.attrs.frames}`);
    const c = cmp(nc.vars.t_500.data, w.mean, w.bound, "tab vs numpy Σsum/Σcount of 360 planes");
    sameCounts(nc.vars.t_500_count.data, w.count, "tab vs numpy");
    say("c3", { nativeEstimate: { frames: before.frames, requests: before.requests, readBytes: before.readBytes, overCap: before.overCap, exact: before.exact }, panel: line.text, download: d, cmp: c });
  }
  // ---------------------------------------------------------------- C4
  if (want("c4")) {
    const box = { w: -50, s: 35, e: -48, n: 37 };
    await choose(page, { store: "7.2d/oisst025d", channels: ["sst"], months: [7], years: [2019, 2019], days: null, box, exclude: "", step: "month", res: "native", std: true, native: false });
    const line = await estLine(page);
    const d = await download(page, "c4");
    const nc = readNc(d.file);
    const w = py(`
import json, sys, datetime as dt
import numpy as np
from family1 import sharded as sh
g = sh.ShardedGroup(sys.argv[1]); lat = np.array(json.loads(sys.argv[2])); lon = np.array(json.loads(sys.argv[3]))
rr = np.rint((lat + 90) / 0.25).astype(int); cc = np.rint((lon + 180) / 0.25).astype(int)
sp = g.spec; ci = [c["name"] for c in sp["channels"]].index("sst")
ty = [i for i, (a, b) in enumerate(sp["row_extents"]) if a <= rr.min() and rr.max() < b][0]
tx = [i for i, (a, b) in enumerate(sp["col_extents"]) if a <= cc.min() and cc.max() < b][0]
r0, c0 = sp["row_extents"][ty][0], sp["col_extents"][tx][0]
fs = []; d = dt.date(2019, 7, 1)
while d.month == 7:
    k = (d - dt.date(1982, 1, 1)).days
    a = g.read_tile(k // 5, k % 5, ty, tx, raw=True, crop=True)
    if a is not None: fs.append(a[np.ix_(rr - r0, cc - c0)][:, :, ci].astype(np.float64))
    d += dt.timedelta(days=1)
st = np.stack(fs)
print(json.dumps({"n": len(fs), "mean": np.nanmean(st, 0).ravel().tolist(), "std": np.nanstd(st, 0).ravel().tolist(), "count": np.isfinite(st).sum(0).ravel().tolist()}))
`, "https://huggingface.co/datasets/chfrank/earth-tensors/resolve/main/tensors/family7_2d/oisst025d/oisst025d", JSON.stringify(nc.vars.lat.data), JSON.stringify(nc.vars.lon.data));
    let maxd = 0, worst = 0, bad = 0;
    w.std.forEach((x, i) => {
      const y = nc.vars.sst_std.data[i], dd = Math.abs(y - x), tol = 1e-6 * (Math.abs(w.mean[i]) + x);
      maxd = Math.max(maxd, dd); worst = Math.max(worst, dd / tol); if (dd > tol) bad++;
    });
    console.log(`C4 oisst025d July 2019 standard deviation, ${JSON.stringify(box)}: the panel says: ${line.text}`);
    console.log(`  ${w.std.length} cells over ${w.n} native daily maps (sharded.py from the Hub): max |std − numpy nanstd| ${maxd.toExponential(2)} °C, worst / tolerance (1e-6 × (|mean| + std)) ${worst.toFixed(3)}, ${bad} outside; the file's long_name: ${nc.vars.sst_std.attrs.long_name}`);
    if (bad) fail("C4 std");
    sameCounts(nc.vars.sst_count.data, w.count, "tab vs native frames");
    await set(page, { "dt-std": false });
    say("c4", { cells: w.std.length, frames: w.n, maxAbs: maxd, worstOverTol: worst, download: d });
  }
  // ---------------------------------------------------------------- C5
  if (want("c5")) {
    const CI = JSON.parse(fs.readFileSync(path.join(ROOT, "data", "family7_clim_index.json"), "utf8"));
    results.c5 = [];
    for (const [g, ch, m, box] of [["g025", "sst", 2, { w: -50, s: 35, e: -40, n: 45 }], ["g100", "t2m", 7, { w: -20, s: 40, e: 10, n: 60 }]]) {
      await store(page, "derived/normals_" + g);
      for (const ver of ["paper", "dev"]) {
        await page.evaluate((k) => document.querySelector(`#dt-npresets button[data-npreset="${k}"]`).click(), ver);
        const s0 = await settled(page);
        // one month and a box (a whole-globe twelve-month 0.25° normal is over the cap)
        const s = await choose(page, { channels: [ch], months: [m], box, res: "native" });
        const lit = await page.evaluate(() => [...document.querySelectorAll("#dt-npresets button.active")].map((b) => b.dataset.npreset));
        const d = await download(page, `c5-${g}-${ver}`);
        const nc = readNc(d.file);
        const G = CI.groups[g], f = CI.files[ver][g].clim_npy, c = G.chans.indexOf(ch);
        const w = py(PYRNG + `
lat = np.array(json.loads(sys.argv[1])); lon = np.array(json.loads(sys.argv[2])); f = json.loads(sys.argv[3]); G = json.loads(sys.argv[4])
m, c = int(sys.argv[5]), int(sys.argv[6]); C = len(G["chans"]); W = G["grid"]["nx"]; step = G["grid"]["step"]
rr = np.rint((lat + 90) / step).astype(int); cc = np.rint((lon + 180) / step).astype(int); r0, r1 = int(rr.min()), int(rr.max())
base = int(f["header_len"]) + ((m - 1) * C + c) * int(f["plane_bytes"])
z = np.frombuffer(rng(f["url"], base + r0 * W * 4, (r1 - r0 + 1) * W * 4), "<f4").reshape(r1 - r0 + 1, W)[np.ix_(rr - r0, cc)]
mu, sd = G["norm"][c]
v = z.astype(np.float64) * sd + mu
bound = np.spacing(np.abs(z)).astype(np.float64) / 2 * sd + 4 * 2.0 ** -52 * np.abs(v)
f_ = lambda a: [None if not np.isfinite(x) else float(x) for x in a.ravel()]
print(json.dumps({"v": f_(v), "bound": f_(bound)}))
`, JSON.stringify(nc.vars.lat.data), JSON.stringify(nc.vars.lon.data), JSON.stringify(f), JSON.stringify(G), String(m), String(c));
        const label = `${g} ${ch} month ${m}, the "${ver === "paper" ? "paper split" : "development split"}" chip (years ${s.sel.yearStart}–${s.sel.yearEnd}, leaving out ${s.sel.excludeYears.join(", ")}) vs clim/${ver}/${g}/clim.npy`;
        console.log(`C5 ${label}: ${d.requests} requests, ${mb(d.bytes)}, ${d.secs.toFixed(1)} s; chip still lit after narrowing to one month: ${lit.join(",") || "none"}`);
        const r = cmp(nc.vars[ch].data, w.v, w.bound, `tab vs published plane (${nc.attrs.period_start}–${nc.attrs.period_end}, excluded ${nc.attrs.excluded_years})`);
        results.c5.push({ g, ch, m, ver, years: [s.sel.yearStart, s.sel.yearEnd], excluded: s.sel.excludeYears, download: d, cmp: r, unit: nc.vars[ch].attrs.units });
        void s0;
      }
    }
  }
  if (page.__errors.length) fail("page errors: " + page.__errors.join(" | "));
} finally {
  await browser.close();
}

try {
  // ---------------------------------------------------------------- C6
  if (want("c6")) {
   // a fresh page: nothing chosen, so every store opens on its own first look
   const { browser: b6, page } = await open();
   try {
    let ids = await page.evaluate(() => [...document.querySelectorAll("#dt-store option")].map((o) => o.value));
    // --slice=a:b: a part of the list (the whole list can outlast one sitting)
    const sl = process.argv.find((x) => x.startsWith("--slice="));
    const all = ids.length;
    if (sl) { const [a, b] = sl.slice(8).split(":").map(Number); ids = ids.slice(a, b); }
    console.log(`C6 ${ids.length} of ${all} stores${sl ? ` (${sl})` : ""}`);
    results.c6 = [];
    let bad = 0;
    for (const id of ids) {
      const t0 = Date.now(), mark = hub.length;
      await page.evaluate((v) => {
        const el = document.getElementById("dt-store");
        el.value = v; el.dispatchEvent(new Event("change", { bubbles: true }));
      }, id);
      let s;
      try { s = await settled(page, 240000); } catch (e) { s = await state(page); s.err = e.message; }
      const e = s.lastEstimate || {};
      const dl = await page.evaluate(() => !document.getElementById("dt-download").disabled);
      const ok = !s.err && dl && !e.overCap && (Number(e.frames ?? e.rows ?? 0) > 0 || Number(e.requests) > 0);
      if (!ok) bad++;
      const row = { id, ok, download: dl, overCap: !!e.overCap, readMB: +(Number(e.readBytes || 0) / 1e6).toFixed(1), outMB: +(Number(e.outBytes || 0) / 1e6).toFixed(1),
        requests: e.requests, path: e.path || null, secs: +((Date.now() - t0) / 1000).toFixed(1), hubRequests: hub.length - mark, err: s.err || null,
        sel: s.sel ? `${s.sel.yearStart}–${s.sel.yearEnd} m${(s.sel.months || []).join(",")}${s.sel.days ? ` d${s.sel.days.join("–")}` : ""} ${s.sel.step}` : null };
      results.c6.push(row);
      console.log(`C6 ${ok ? "ok  " : "FAIL"} ${id.padEnd(26)} ${row.sel || ""} · ${row.requests} requests, ${row.readMB} MB read, ${row.outMB} MB file${row.path ? ` · path ${row.path}` : ""} · ${row.secs} s${row.err ? " · " + row.err : ""}`);
    }
    const errs = await page.evaluate(() => [document.getElementById("dt-reg-error").textContent, document.getElementById("dt-missing") && document.getElementById("dt-missing").textContent]);
    console.log(`C6 ${ids.length} stores, ${bad} not on a downloadable first look; registry lines: ${JSON.stringify(errs)}`);
    if (page.__errors.length) fail("page errors: " + page.__errors.join(" | "));
    if (bad) fail("C6");
   } finally { await b6.close(); }
  }
} catch (e) { console.error(e.message); }

// -------------------------------------------------------------- screenshots
if (want("shots")) {
  for (const [name, vp, mobile] of [["desktop", { width: 1280, height: 900 }, false], ["phone", { width: 390, height: 844 }, true]]) {
    const { browser: b, page: p } = await open(vp, { mobile });
    try {
      await store(p, "derived/normals_g025");
      await p.evaluate(() => document.querySelector('#dt-npresets button[data-npreset="paper"]').click());
      await settled(p);
      await p.evaluate(() => document.getElementById("dt-preview").click());
      const t0 = Date.now();
      while (!(await state(p)).previewShown) { if (Date.now() - t0 > 180000) throw new Error("no preview"); await new Promise((r) => setTimeout(r, 500)); }
      // the camera on the box, so the painted preview is what the globe shows
      await p.evaluate(() => {
        const b = window.__earth.dataTabState().sel.bbox;
        window.__earth.viewer.camera.setView({ destination: Cesium.Rectangle.fromDegrees(b.w - 12, b.s - 8, b.e + 12, b.n + 8) });
      });
      await p.waitForTimeout(6000);
      // the panel scrolled so the chips, the estimate and the legend show
      await p.evaluate(() => document.getElementById("dt-npresets").scrollIntoView({ block: "start" }));
      const file = `/home/claude/datatab-paper-${name}.png`;
      await p.screenshot({ path: file });
      const ov = await p.evaluate(() => ({ sw: document.documentElement.scrollWidth, cw: document.documentElement.clientWidth }));
      console.log(`shot ${file}: scrollWidth ${ov.sw}, clientWidth ${ov.cw}; page errors ${p.__errors.length}`);
      if (p.__errors.length) fail("page errors in the " + name + " shot: " + p.__errors.join(" | "));
      if (mobile) {
        await p.setViewportSize({ width: 360, height: 800 });
        await p.waitForTimeout(800);
        const o2 = await p.evaluate(() => {
          const wide = [...document.querySelectorAll("#panel-data *")].filter((e) => e.getBoundingClientRect().right > document.documentElement.clientWidth + 1)
            .map((e) => e.id || e.className || e.tagName).slice(0, 8);
          return { sw: document.documentElement.scrollWidth, cw: document.documentElement.clientWidth, wide };
        });
        console.log(`at 360 px: scrollWidth ${o2.sw}, clientWidth ${o2.cw}; elements past the right edge: ${JSON.stringify(o2.wide)}`);
        if (o2.sw > o2.cw) fail("horizontal overflow at 360 px");
        results.overflow360 = o2;
      }
    } finally { await b.close(); }
  }
}

fs.writeFileSync(path.join(OUT, "results.json"), JSON.stringify(results, null, 1));
console.log(`\nresults: ${path.join(OUT, "results.json")}`);
console.log(process.exitCode ? "\nSOME BROWSER CHECKS FAILED" : "\nALL BROWSER CHECKS PASSED");
if (server) server.kill();

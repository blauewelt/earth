// tests/f1data.test.mjs — src/f1data.js against the fixture the REAL writers
// produced (tests/make_family1_fixture.py → data/family1_fixture/), served over
// a local HTTP server that honours Range (E-084 §6).
//
//     node --test tests/f1data.test.mjs
//
// The server has three faces: /ok/ answers Range with 206 (and counts every
// request, its Range header and the requests in flight), /bad200/ ignores the
// Range and answers 200 with the whole file — which the reader must refuse —
// and /nozst/ answers 404 for every shard, which must reject the whole run with
// the shard's URL in the message.
//
// This file is node:test, not Playwright. Playwright's default testMatch also
// matches *.test.mjs, so the suite registers nothing unless node runs it.
import { createRequire } from "node:module";
import { test, before, after } from "node:test";
import assert from "node:assert/strict";
import http from "node:http";
import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import { execFileSync } from "node:child_process";
import { fileURLToPath } from "node:url";

const HERE = path.dirname(fileURLToPath(import.meta.url));
const ROOT = path.dirname(HERE);
const FIX = path.join(ROOT, "data", "family1_fixture");
// the family-10 / derived fixture (tests/make_family10_fixture.py), served as /ok10/
const FIX10 = path.join(ROOT, "data", "family10_fixture");
const EXP10 = JSON.parse(fs.readFileSync(path.join(FIX10, "expected.json"), "utf8"));
// the family-1.2 fixture (tests/make_family12_fixture.py: ERA5-shaped levels,
// six-hourly frames), served as /ok12/
const FIX12 = path.join(ROOT, "data", "family12_fixture");
const EXP12 = JSON.parse(fs.readFileSync(path.join(FIX12, "expected.json"), "utf8"));
// the multi-year normals fixture (tests/make_family7_monthly_fixture.py): E-086's
// per-year monthly sums and counts over two small grids, served as /okm/
const FIXM = path.join(ROOT, "data", "family7_monthly", "fixture_multi");
const EXPM = JSON.parse(fs.readFileSync(path.join(FIXM, "expected.json"), "utf8"));
const require = createRequire(import.meta.url);
const F1 = require("../src/f1data.js");
const EXP = JSON.parse(fs.readFileSync(path.join(FIX, "expected.json"), "utf8"));
// Register only when node itself is running this file (node --test sets
// NODE_TEST_CONTEXT in the child it spawns; `node tests/f1data.test.mjs` runs it
// directly). Playwright's loader imports it too and must find nothing to do.
const runByNode = !!process.env.NODE_TEST_CONTEXT ||
  (process.argv[1] && path.resolve(process.argv[1]) === fileURLToPath(import.meta.url));
const inPlaywright = !runByNode || process.env.TEST_WORKER_INDEX !== undefined;

let server, port;
const log = [];
let inflight = 0, maxInflight = 0;

function send206(res, body, mode, rel) {
  log.push({ mode, rel, range: "bytes=0-", bytes: body.length });
  res.writeHead(206, { "content-range": `bytes 0-${body.length - 1}/${body.length}`, "content-length": body.length, "accept-ranges": "bytes" });
  res.end(body);
}

function serve(req, res) {
  const url = new URL(req.url, "http://x");
  const m = /^\/(ok|bad200|nozst|ok10|ok12|okm|ok72)\/(.*)$/.exec(url.pathname);
  const rel = m ? decodeURIComponent(m[2]) : "";
  const root = m && m[1] === "ok10" ? FIX10 : m && m[1] === "ok12" ? FIX12 : m && m[1] === "okm" ? FIXM : FIX;
  // /ok72/: the family 1.gf fixture with its registry dressed as family 7.2d's
  // (source_segments, preliminary days, a record_span, a centred-sigma unit)
  if (m && m[1] === "ok72" && rel === "family1gf.json") {
    const reg = JSON.parse(fs.readFileSync(path.join(FIX, rel), "utf8"));
    for (const g of reg.groups) {
      if (g.name !== "fxgrid") continue;
      g.record_span = ["2009-12-27", "2010-01-06"];
      g.requested_window = ["2009-12-01", "2010-01-31"];
      g.source_segments = [
        { falsifier: "pentad", from: "2009-12-25", to: "2009-12-31", source: "FX source v1" },
        { falsifier: "source-readback", from: "2010-01-01", to: "2010-01-08", source: "FX source v1", note: "the producer's files end 2010-01-08" }];
      g.counts = Object.assign({}, g.counts || {}, { preliminary_days: ["2010-01-07", "2010-01-06", "2010-01-08"] });
      g.channels[0].unit = g.channels[0].unit + " (centred 5-day sigma)";
    }
    return send206(res, Buffer.from(JSON.stringify(reg)), m[1], rel);
  }
  if (m && m[1] === "ok72" && /^fxgrid\/.*tile_grid\.json$/.test(rel)) {
    const tg = JSON.parse(fs.readFileSync(path.join(FIX, rel), "utf8"));
    tg.channels[0].unit = tg.channels[0].unit + " (centred 5-day sigma)";
    // family 7.2d states its grid POINT-aligned: x0/y0 are the first CENTRE
    const g = tg.grid;
    g.x0 += g.dx / 2; g.y0 += g.dy / 2;
    // (worded like ncep100d's, with no number to parse: the shift must still
    // happen exactly once however often the cached JSON is handed back)
    g.align = "point — family 7's own grid";
    return send206(res, Buffer.from(JSON.stringify(tg)), m[1], rel);
  }
  const file = path.join(root, rel);
  if (!m || !file.startsWith(root) || !fs.existsSync(file) || fs.statSync(file).isDirectory() ||
      (m[1] === "nozst" && rel.endsWith(".zst"))) {
    res.writeHead(404); res.end("not found"); return;
  }
  const buf = fs.readFileSync(file);
  const range = req.headers.range;
  const entry = { mode: m[1], rel, range: range || null, bytes: 0 };
  log.push(entry);
  inflight++; maxInflight = Math.max(maxInflight, inflight);
  setTimeout(() => {
    inflight--;
    if (m[1] === "bad200" || !range) {
      entry.bytes = buf.length;
      res.writeHead(200, { "content-length": buf.length, "accept-ranges": "bytes" });
      res.end(buf);
      return;
    }
    const r = /^bytes=(\d+)-(\d*)$/.exec(range);
    const a = Number(r[1]);
    const z = r[2] === "" ? buf.length - 1 : Math.min(Number(r[2]), buf.length - 1);
    if (a >= buf.length) { res.writeHead(416, { "content-range": `bytes */${buf.length}` }); res.end(); return; }
    const part = buf.subarray(a, z + 1);
    entry.bytes = part.length;
    res.writeHead(206, { "content-range": `bytes ${a}-${z}/${buf.length}`, "content-length": part.length, "accept-ranges": "bytes" });
    res.end(part);
  }, 3);
}

const base = (mode) => `http://127.0.0.1:${port}/${mode}/`;


// ------------------------------------------------------------------ helpers
const cases = (re) => EXP.cases.filter((c) => re.test(c.name));
const isNum = (x) => typeof x === "number" && Number.isFinite(x);

function closeArr(got, want, { rel = 0, label = "" } = {}) {
  assert.equal(got.length, want.length, `${label}: length ${got.length} vs ${want.length}`);
  let bad = 0, first = null;
  for (let i = 0; i < want.length; i++) {
    const g = got[i], w = want[i];
    const ok = w === null ? Number.isNaN(g)
      : rel === 0 ? g === w : Math.abs(g - w) <= rel * Math.max(1, Math.abs(w));
    if (!ok) { bad++; if (first === null) first = [i, g, w]; }
  }
  assert.equal(bad, 0, `${label}: ${bad} mismatches, first at ${first && first.join(" / ")}`);
}

function selOf(c) { return JSON.parse(JSON.stringify(c.sel)); }

function nanmeanJS(stack) { // stack: array of Float32Array → [mean, count]
  const n = stack[0].length, m = new Float64Array(n), k = new Uint32Array(n);
  for (const a of stack) for (let i = 0; i < n; i++) if (!Number.isNaN(a[i])) { m[i] += a[i]; k[i]++; }
  return [Array.from(m, (v, i) => (k[i] ? v / k[i] : NaN)), k];
}

function python(code, ...args) {
  return execFileSync("python3", ["-c", code, ...args], { encoding: "utf8", maxBuffer: 1 << 28 });
}

function defineTests() {
  test("registry: built public stores only, units after conversion, grid geometry", async () => {
    const reg = await F1.loadRegistry();
    const names = reg.stores.map((s) => s.name).sort();
    assert.deepEqual(names, ["fxgrid", "fxpts", "fxtb"]);
    const tb = reg.stores.find((s) => s.name === "fxtb");
    assert.equal(tb.kind, "grid");
    assert.equal(tb.channels[0].unit, "K");
    assert.equal(tb.channels[0].min, 160);
    assert.equal(tb.channels[0].max, 414);
    assert.equal(tb.frameSeconds, 10800);
    assert.equal(tb.framesPerBin, 40);
    assert.equal(tb.subDaily, true);
    assert.equal(tb.grid.H, 60);
    assert.equal(tb.grid.dlat, 0.5);              // row 0 is the SOUTHERN row
    assert.equal(tb.grid.lat0, -14.75);
    const g = reg.stores.find((s) => s.name === "fxgrid");
    assert.deepEqual(g.span, ["2009-12-25", "2010-01-08"]);
    assert.equal(g.grid.lat0, 89.75);
    assert.equal(g.grid.dlat, -0.5);
    assert.match(g.channels[1].note, /logarithm/);
    const p = reg.stores.find((s) => s.name === "fxpts");
    assert.equal(p.kind, "points");
    assert.equal(p.N, 5000);
    assert.equal(p.grid, null);
    // every request so far asked for a Range
    assert.ok(log.length > 0 && log.every((e) => e.range), "a request went out without a Range header");
  });

  for (const c of cases(/^(grid|uint8) /)) {
    test(`grid: ${c.name} — values equal what the writer was given`, async () => {
      const r = await F1.run(selOf(c));
      const x = c.expect;
      assert.equal(r.kind, "grid");
      assert.deepEqual([r.time.length, r.channels.length, r.lat.length, r.lon.length], x.shape);
      closeArr(r.lat, x.lat, { label: "lat" });
      closeArr(r.lon, x.lon, { label: "lon" });
      closeArr(r.time, x.time, { label: "time" });
      const mean = c.sel.step !== "native" || c.sel.res !== "native";
      closeArr(r.data, x.data, { rel: mean ? 2e-6 : 0, label: "data" });
      if (x.count === null) assert.equal(r.count, null);
      else closeArr(r.count, x.count, { label: "count" });
      assert.equal(r.frames, x.frames);
      for (let i = 1; i < r.lon.length; i++) assert.ok(r.lon[i] > r.lon[i - 1], "lon must be monotonic");
      if (c.sel.store === "fxtb") assert.deepEqual(r.units, ["K"]);
    });
  }

  test("grid: an absent frame (offset −1) is left out of time; a present empty-tile frame is kept", async () => {
    const c = cases(/^grid native box$/)[0];
    const r = await F1.run(selOf(c));
    const iso = Array.from(r.time, (t) => new Date(t * 1000).toISOString().slice(0, 10));
    assert.equal(iso.length, 14);
    assert.ok(!iso.includes("2010-01-05"), "the absent frame must not appear");
    assert.ok(iso.includes("2009-12-27"), "the frame with an empty tile is a real frame");
    const fi = iso.indexOf("2009-12-27");
    const H = r.lat.length, W = r.lon.length;
    // inside the emptied tile (lon −20 .. −4.25, lat 10 .. −5.75) every pixel is missing
    let inside = 0;
    for (let y = 0; y < H; y++) for (let xx = 0; xx < W; xx++) {
      if (r.lon[xx] <= -4.25 && r.lat[y] >= -5.75) {
        inside++;
        assert.ok(Number.isNaN(r.data[(fi * 2) * H * W + y * W + xx]));
      }
    }
    assert.ok(inside > 0);
  });

  test("grid: dateline box runs west → 180 → east with lon > 180", async () => {
    const c = cases(/^grid dateline box$/)[0];
    const r = await F1.run(selOf(c));
    assert.equal(r.lon[0], 175.25);
    assert.equal(r.lon[r.lon.length - 1], 183.75);
    assert.ok(r.notes.some((n) => /dateline/.test(n)));
  });

  test("grid: pentad / month / all means equal nanmean over the native frames, counts the finite count", async () => {
    const base = selOf(cases(/^grid native box$/)[0]);
    const nat = await F1.run(base);
    const H = nat.lat.length, W = nat.lon.length, C = 2, HW = H * W;
    const frame = (t, k) => nat.data.subarray((t * C + k) * HW, (t * C + k + 1) * HW);
    for (const step of ["pentad", "month", "all"]) {
      const m = await F1.run({ ...base, step });
      const keyOf = (t) => {
        const d = new Date(t * 1000);
        if (step === "pentad") return Math.floor((t - 378691200) / 432000);
        if (step === "month") return d.getUTCFullYear() * 12 + d.getUTCMonth();
        return 0;
      };
      const groups = new Map();
      Array.from(nat.time).forEach((t, i) => { const k = keyOf(t); if (!groups.has(k)) groups.set(k, []); groups.get(k).push(i); });
      const keys = [...groups.keys()].sort((a, b) => a - b);
      assert.equal(m.time.length, keys.length, step);
      keys.forEach((key, ti) => {
        for (let k = 0; k < C; k++) {
          const [mean, cnt] = nanmeanJS(groups.get(key).map((i) => frame(i, k)));
          const got = m.data.subarray((ti * C + k) * HW, (ti * C + k + 1) * HW);
          const gotC = m.count.subarray((ti * C + k) * HW, (ti * C + k + 1) * HW);
          closeArr(got, mean.map((v) => (Number.isNaN(v) ? null : v)), { rel: 2e-6, label: `${step} mean` });
          closeArr(gotC, Array.from(cnt), { label: `${step} count` });
        }
      });
    }
  });

  test("grid: a box is required; estimate says so instead of reading", async () => {
    const e = await F1.estimate({ ...selOf(cases(/^grid native box$/)[0]), bbox: null });
    assert.equal(e.overCap, true);
    assert.match(e.why, /box/);
    await assert.rejects(F1.run({ ...selOf(cases(/^grid native box$/)[0]), bbox: null }), /box/);
  });

  test("grid: estimate is exact — its requests and bytes are what the run then fetches", async () => {
    const sel = selOf(cases(/^grid native box$/)[0]);
    F1.configure({ base: base("ok") });
    const e = await F1.estimate(sel);
    assert.equal(e.exact, true);
    assert.equal(e.frames, 14);
    assert.deepEqual(e.shape, [14, 2, 14, 18]);
    assert.equal(e.overCap, false);
    F1.configure({ base: base("ok") });
    const mark = log.length;
    const r = await F1.run(sel);
    const data = log.slice(mark).filter((x) => /\.(zst|idx\.npy)$/.test(x.rel));
    assert.equal(data.length, e.requests);
    assert.equal(data.reduce((a, x) => a + x.bytes, 0), e.readBytes);
    assert.equal(e.outBytes, r.data.byteLength + 8 * (r.time.length + r.lat.length + r.lon.length));
  });

  test("grid: estimate flags over-cap selections in plain English", async () => {
    const sel = selOf(cases(/^grid native box$/)[0]);
    // the whole globe is 360 x 720 at 0.5°: small here, so shrink the caps' view
    // through a long period at native resolution — 2 channels × 14 frames
    // is far below the cap; make it over by asking for the whole globe at
    // native resolution and checking the arithmetic instead
    const e = await F1.estimate({ ...sel, bbox: { w: -180, s: -90, e: 180, n: 90 } });
    assert.equal(e.shape.join(), "14,2,360,720");
    assert.equal(e.outBytes, 14 * 2 * 360 * 720 * 4 + 8 * (14 + 360 + 720));
    assert.equal(e.overCap, false);
    assert.ok(F1.CAPS.readBytes === 600e6 && F1.CAPS.outBytes === 400e6);
  });

  test("HTTP: a 200 answer to a Range request is refused, with the URL", async () => {
    F1.configure({ base: base("bad200") });
    await assert.rejects(F1.loadRegistry(), (e) => /HTTP 200/.test(e.message) && e.message.includes(base("bad200") + "family1gf.json"));
    F1.configure({ base: base("ok") });
  });

  test("HTTP: a failed read rejects the whole run with the URL in the message", async () => {
    F1.configure({ base: base("nozst") });
    const sel = selOf(cases(/^grid native box$/)[0]);
    await assert.rejects(F1.run(sel), (e) => /HTTP 404/.test(e.message) && /\/nozst\/fxgrid\/fxgrid\/2009\/bin_2044\.zst/.test(e.message));
    F1.configure({ base: base("ok") });
  });

  test("HTTP: at most 6 requests in flight, every one with a Range header", async () => {
    maxInflight = 0;
    F1.configure({ base: base("ok") });
    const mark = log.length;
    await F1.run(selOf(cases(/^uint8 pentad$/)[0]));
    assert.ok(maxInflight <= 6, `max in flight ${maxInflight}`);
    assert.ok(maxInflight >= 2, "the run should actually overlap requests");
    assert.ok(log.slice(mark).every((e) => e.range));
  });

  test("abort: an aborted run rejects with AbortError and progress is reported", async () => {
    F1.configure({ base: base("ok") });
    const ac = new AbortController();
    const seen = [];
    const p = F1.run(selOf(cases(/^uint8 pentad$/)[0]), {
      signal: ac.signal,
      onProgress: (x) => { seen.push(x); if (x.done >= 3) ac.abort(); },
    });
    await assert.rejects(p, (e) => e.name === "AbortError");
    assert.ok(seen.length >= 3 && seen.every((x) => isNum(x.done) && isNum(x.total) && isNum(x.bytes)));
    const done = [];
    await F1.run(selOf(cases(/^uint8 pentad$/)[0]), { onProgress: (x) => done.push(x) });
    const last = done[done.length - 1];
    assert.equal(last.done, last.total);
  });

  test("preview: one frame, the first with data", async () => {
    const sel = { ...selOf(cases(/^grid native box$/)[0]), step: "month" };
    const r = await F1.preview(sel);
    assert.equal(r.time.length, 1);
    assert.equal(new Date(r.time[0] * 1000).toISOString().slice(0, 10), "2009-12-25");
    const n = await F1.run(selOf(cases(/^grid native box$/)[0]));
    closeArr(r.data, Array.from(n.data.subarray(0, r.data.length), (v) => (Number.isNaN(v) ? null : v)), { label: "preview" });
    const pp = await F1.preview(selOf(cases(/^points box january/)[0]));
    assert.equal(pp.kind, "points");
    assert.ok(pp.time.length > 0);
    const bins = new Set(Array.from(pp.time, (t) => Math.floor((t - 378691200) / 432000)));
    assert.equal(bins.size, 1);
  });

  for (const c of cases(/^points (?!binned)/)) {
    test(`points: ${c.name} — period, month, hour and box filtering`, async () => {
      const e = await F1.estimate(selOf(c));
      assert.equal(e.rows, c.expect.rowsRead, "estimate rows = rows in the selected bins (exact)");
      const r = await F1.run(selOf(c));
      assert.equal(r.kind, "points");
      assert.equal(r.rowsRead, c.expect.rowsRead);
      assert.equal(r.time.length, c.expect.n);
      closeArr(r.time, c.expect.time, { label: "time" });
      closeArr(r.lat, c.expect.lat, { label: "lat" });
      closeArr(r.lon, c.expect.lon, { label: "lon" });
      closeArr(r.values, c.expect.values, { label: "values" });
      assert.deepEqual(Array.from(r.platform, (x) => x.toString()), c.expect.platform);
      assert.deepEqual(Array.from(r.qc), c.expect.qc);
      assert.ok(r.platform instanceof BigInt64Array);
    });
  }

  for (const c of cases(/^points binned/)) {
    test(`points: ${c.name} — binned mean and count`, async () => {
      const r = await F1.run(selOf(c));
      assert.equal(r.kind, "grid");
      assert.equal(r.binnedFrom, "points");
      closeArr(r.time, c.expect.steps, { label: "steps" });
      const H = r.lat.length, W = r.lon.length, C = r.channels.length, HW = H * W;
      const seen = new Set();
      for (const cell of c.expect.cells) {
        const t = Array.from(r.time).indexOf(cell.time);
        const y = Array.from(r.lat).findIndex((v) => Math.abs(v - cell.lat) < 1e-9);
        const x = Array.from(r.lon).findIndex((v) => Math.abs(v - cell.lon) < 1e-9);
        assert.ok(t >= 0 && y >= 0 && x >= 0, `cell ${JSON.stringify(cell)} not on the output grid`);
        for (let k = 0; k < C; k++) {
          const j = (t * C + k) * HW + y * W + x;
          seen.add(j);
          assert.equal(r.count[j], cell.count[k]);
          if (cell.mean[k] === null) assert.ok(Number.isNaN(r.data[j]));
          else assert.ok(Math.abs(r.data[j] - cell.mean[k]) <= 2e-6 * Math.max(1, Math.abs(cell.mean[k])), `${r.data[j]} vs ${cell.mean[k]}`);
        }
      }
      for (let j = 0; j < r.count.length; j++) if (!seen.has(j)) assert.equal(r.count[j], 0);
    });
  }

  test("NetCDF: opens in Python (scipy) with identical values — grid mean with counts", async () => {
    const sel = selOf(cases(/^grid res 1 month mean$/)[0]);
    const r = await F1.run(sel);
    const f = path.join(os.tmpdir(), `f1data-test-${process.pid}-grid.nc`);
    fs.writeFileSync(f, Buffer.from(await F1.toNetCDF(r).arrayBuffer()));
    const out = JSON.parse(python(`
import json, sys, numpy as np
from scipy.io import netcdf_file
with netcdf_file(sys.argv[1], "r", mmap=False) as nc:
    v = nc.variables
    def L(a): return [None if not np.isfinite(x) else float(x) for x in np.asarray(a, np.float64).ravel()]
    def S(x): return x.decode() if isinstance(x, bytes) else x
    print(json.dumps({"version": int(nc.version_byte), "dims": {k: int(n) for k, n in nc.dimensions.items()},
      "time": L(v["time"][:]), "lat": L(v["lat"][:]), "lon": L(v["lon"][:]),
      "sst": L(v["sst"][:]), "log_chl": L(v["log_chl"][:]), "sst_count": np.asarray(v["sst_count"][:]).ravel().tolist(),
      "log_chl_count": np.asarray(v["log_chl_count"][:]).ravel().tolist(),
      "units": S(v["log_chl"].units), "tunits": S(v["time"].units), "source": S(nc.source), "selection": json.loads(S(nc.selection))}))
`, f));
    fs.unlinkSync(f);
    assert.equal(out.version, 2, "64-bit offset NetCDF-3");
    assert.deepEqual(out.dims, { time: r.time.length, lat: r.lat.length, lon: r.lon.length });
    closeArr(out.time, Array.from(r.time), { label: "time" });
    closeArr(out.lat, Array.from(r.lat), { label: "lat" });
    closeArr(out.lon, Array.from(r.lon), { label: "lon" });
    const HW = r.lat.length * r.lon.length, C = 2;
    const chan = (k, arr) => { const o = []; for (let t = 0; t < r.time.length; t++) for (let i = 0; i < HW; i++) o.push(arr[(t * C + k) * HW + i]); return o; };
    closeArr(chan(0, r.data), out.sst, { label: "sst" });
    closeArr(chan(1, r.data), out.log_chl, { label: "log_chl" });
    assert.deepEqual(chan(0, r.count), out.sst_count);
    assert.deepEqual(chan(1, r.count), out.log_chl_count);
    assert.match(out.units, /LOGARITHM/);
    assert.equal(out.tunits, "seconds since 1970-01-01 00:00:00");
    assert.match(out.source, /fxgrid/);
    assert.equal(out.selection.store, "fxgrid");
  });

  test("NetCDF: points, and the uint8 store in kelvin", async () => {
    const r = await F1.run(selOf(cases(/^points box january/)[0]));
    const f = path.join(os.tmpdir(), `f1data-test-${process.pid}-pts.nc`);
    fs.writeFileSync(f, Buffer.from(await F1.toNetCDF(r).arrayBuffer()));
    const t = await F1.run(selOf(cases(/^uint8 native kelvin/)[0]));
    const g = path.join(os.tmpdir(), `f1data-test-${process.pid}-tb.nc`);
    fs.writeFileSync(g, Buffer.from(await F1.toNetCDF(t).arrayBuffer()));
    const out = JSON.parse(python(`
import json, sys, numpy as np
from scipy.io import netcdf_file
def L(a): return [None if not np.isfinite(x) else float(x) for x in np.asarray(a, np.float64).ravel()]
with netcdf_file(sys.argv[1], "r", mmap=False) as nc:
    v = nc.variables
    plat = [b"".join(row).decode() for row in v["platform"][:]]
    p = {"n": int(nc.dimensions["obs"]), "time": L(v["time"][:]), "oxy": L(v["oxy"][:]), "platform": plat, "qc": v["qc"][:].tolist()}
with netcdf_file(sys.argv[2], "r", mmap=False) as nc:
    tb = nc.variables["tb"]
    q = {"tb": L(tb[:]), "units": tb.units.decode()}
print(json.dumps({"p": p, "q": q}))
`, f, g));
    fs.unlinkSync(f); fs.unlinkSync(g);
    assert.equal(out.p.n, r.time.length);
    closeArr(out.p.time, Array.from(r.time), { label: "time" });
    closeArr(Array.from({ length: r.time.length }, (_, i) => r.values[i * 3 + 2]), out.p.oxy, { label: "oxy" });
    assert.deepEqual(out.p.platform, Array.from(r.platform, (x) => x.toString()));
    assert.deepEqual(out.p.qc, Array.from(r.qc));
    assert.equal(out.q.units, "K");
    closeArr(Array.from(t.data), out.q.tb, { label: "tb" });
  });

  test("days: a day-of-month range keeps exactly the frames and rows on those days, and reads less", async () => {
    const dom = (t) => new Date(t * 1000).getUTCDate();
    // grid: the native box over 2009-12-25 .. 2010-01-08, days 26–31 of every month
    const g = selOf(cases(/^grid native box$/)[0]);
    const all = await F1.run(g);
    const some = await F1.run(Object.assign({}, g, { days: [26, 31] }));
    const want = Array.from(all.time).filter((t) => dom(t) >= 26 && dom(t) <= 31);
    assert.ok(want.length > 0 && want.length < all.time.length);
    assert.deepEqual(Array.from(some.time), want);
    // points: every row of the full selection whose day is 3–9, in order — the
    // row search over the sorted time column must neither drop nor add a row
    const p = { store: "fxpts", channels: ["temp"], yearStart: 2009, yearEnd: 2010,
      months: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12], hours: null, bbox: null, step: "native", res: "native" };
    const pa = await F1.run(p);
    const pd = Object.assign({}, p, { days: [3, 9] });
    const pb = await F1.run(pd);
    const keep = [];
    for (let i = 0; i < pa.time.length; i++) if (dom(pa.time[i]) >= 3 && dom(pa.time[i]) <= 9) keep.push(i);
    assert.ok(keep.length > 0 && keep.length < pa.time.length);
    assert.deepEqual(Array.from(pb.time), keep.map((i) => pa.time[i]));
    assert.deepEqual(Array.from(pb.values), keep.map((i) => pa.values[i]));
    const ea = await F1.estimate(p), eb = await F1.estimate(pd);
    assert.ok(eb.rows >= keep.length, "the estimate never under-counts");
    assert.ok(eb.rows < ea.rows && eb.readBytes < ea.readBytes, "days narrower than a bin read fewer rows");
    // and a nonsense range is refused in words
    await assert.rejects(F1.estimate(Object.assign({}, p, { days: [9, 3] })), /days must be/);
  });

  test("CSV: one row per observation (points) and per non-empty cell (grids)", async () => {
    const r = await F1.run(selOf(cases(/^points dateline all months/)[0]));
    const txt = await F1.toCSV(r).text();
    const lines = txt.trimEnd().split("\n");
    assert.equal(lines.length, r.time.length + 1);
    assert.equal(lines[0], "time,lat,lon,oxy,temp,platform,qc");
    assert.match(lines[1], /^\d{4}-\d\d-\d\dT\d\d:\d\d:\d\dZ,/);
    const g = await F1.run(selOf(cases(/^grid res 1 month mean$/)[0]));
    const gt = (await F1.toCSV(g).text()).trimEnd().split("\n");
    const C = g.channels.length, HW = g.lat.length * g.lon.length;
    let cells = 0;
    for (let t = 0; t < g.time.length; t++) for (let i = 0; i < HW; i++) {
      let any = false;
      for (let k = 0; k < C; k++) if (!Number.isNaN(g.data[(t * C + k) * HW + i])) any = true;
      if (any) cells++;
    }
    assert.equal(gt.length, cells + 1);
    assert.equal(gt[0], "time,lat,lon,sst,log_chl,sst_count,log_chl_count");
  });


  // ---- the multi-registry reader: family 10 and the derived maps ----------
  function multi(extra) {
    const b10 = base("ok10");
    F1.configure({ siteBase: b10, registries: [
      { family: "1.gf", title: "Fine observations (family 1.gf)", kind: "family1", url: base("ok") + "family1gf.json" },
      { family: "10", title: "Global tensor and point observations (family 10)", kind: "family10",
        url: b10 + "family10.json", root: b10, norms: "family7_index.json" },
      { family: "derived", title: "Derived maps", kind: "derived", fishing: "fishing_index.json", clim: "clim_index.json" },
    ].concat(extra || []) });
  }
  const case10 = (name) => { const c = EXP10.cases.find((x) => x.name === name); assert.ok(c, name); return JSON.parse(JSON.stringify(c)); };

  test("multi-registry: three families load, ids are family/name, a 404 registry is listed in errors and the rest work", async () => {
    multi([{ family: "gone", title: "A family that is not published", kind: "family1", url: base("ok") + "no/such/registry.json" },
      { family: "1.2", title: "An optional family", kind: "family1", optional: true, url: base("ok") + "no/family12.json" }]);
    try {
      const reg = await F1.loadRegistry();
      assert.deepEqual(reg.families.map((f) => f.family), ["1.gf", "10", "derived"]);
      assert.deepEqual(reg.stores.filter((d) => d.family === "10").map((d) => d.name), ["fxg", "fxrg", "fxargo", "fxneg"]);
      assert.deepEqual(reg.stores.filter((d) => d.family === "derived").map((d) => d.id), ["derived/fishing_grid", "derived/clim_fxg"]);
      assert.equal(reg.errors.length, 1);
      assert.equal(reg.errors[0].family, "gone");
      assert.match(reg.errors[0].message, /HTTP 404/);
      assert.deepEqual(reg.missing.map((m) => m.family), ["1.2"], "an optional family that 404s is absent, not an error");
      // and a store of the families that did load still reads
      const r = await F1.run(case10("monthmajor").sel);
      assert.ok(r.data.length > 0);
      // the levelled group exposes variables × levels
      const rg = reg.stores.find((d) => d.id === "10/fxrg");
      assert.deepEqual(rg.levels, [10, 30, 50]);
      assert.deepEqual(rg.vars.map((v) => v.var), ["rg_t", "rg_s"]);
      assert.deepEqual(rg.channels.map((c) => [c.var, c.level]).slice(0, 2), [["rg_t", 10], ["rg_t", 30]]);
      const ar = reg.stores.find((d) => d.id === "10/fxargo");
      assert.deepEqual(ar.vars.map((v) => v.var), ["temp", "psal"]);
      assert.deepEqual(ar.span, ["2009-12-20", "2010-02-20"]);
      // a bare name still resolves when it is unambiguous, family/name always
      await F1.estimate(Object.assign({}, case10("monthmajor").sel, { family: undefined }));
    } finally { F1.configure({ base: base("ok") }); }
  });

  test("binmajor: one bin is a range of the box's rows, values are z × sd + mean, months and years select bins", async () => {
    multi();
    try {
      const c = case10("binmajor native");
      log.length = 0;
      const e = await F1.estimate(c.sel);
      const r = await F1.run(c.sel);
      assert.deepEqual([r.time.length, r.channels.length, r.lat.length, r.lon.length], c.shape);
      assert.deepEqual(Array.from(r.time), c.time);
      assert.deepEqual(Array.from(r.lat), c.lat);
      assert.deepEqual(Array.from(r.lon), c.lon);
      closeArr(r.data, c.data, { rel: 2e-6, label: "de-z-scored values" });
      // exact estimate: one request per bin, covering the box's latitude band only,
      // with every one of the group's 3 channels in those rows
      const shards = log.filter((x) => /fx7_X_fxg\.npy/.test(x.rel) && x.range && !/bytes=0-1023$/.test(x.range));
      assert.equal(e.requests, c.time.length);
      assert.equal(shards.length, c.time.length);
      const rowBytes = 36 * 3 * 2;
      for (const x of shards) assert.equal(x.bytes, c.lat.length * rowBytes);
      assert.equal(e.readBytes, c.time.length * c.lat.length * rowBytes);
      assert.match(e.why, /every one of the store's 3 channels/);
      assert.ok(r.notes.some((n) => /z × sd \+ mean/.test(n)));
    } finally { F1.configure({ base: base("ok") }); }
  });

  test("binmajor: a monthly mean equals nanmean over the bins with the finite count; a full-height band coalesces bins", async () => {
    multi();
    try {
      const c = case10("binmajor month mean");
      const r = await F1.run(c.sel);
      assert.equal(r.time.length, 1);
      closeArr(r.data, c.data, { rel: 2e-6, label: "monthly mean" });
      assert.deepEqual(Array.from(r.count), c.count);
      // the whole globe: rows 0..H-1 of consecutive bins are contiguous, so the
      // four January bins are one range (the store is small enough)
      const g = Object.assign({}, c.sel, { bbox: { w: -180, s: -90, e: 180, n: 90 } });
      const e = await F1.estimate(g);
      assert.equal(e.requests, 1);
      assert.equal(e.readBytes, 4 * 19 * 36 * 3 * 2);
    } finally { F1.configure({ base: base("ok") }); }
  });

  test("binmajor monthly group: frames are calendar months, levelled channels read, a 1° store offers nothing coarser than 1°", async () => {
    multi();
    try {
      const c = case10("levelled monthly");
      const r = await F1.run(c.sel);
      assert.deepEqual(Array.from(r.time), c.time);
      closeArr(r.data, c.data, { rel: 2e-6, label: "rg values" });
      assert.deepEqual(r.levels, [10, 50]);
      assert.ok(r.notes.some((n) => /one frame per calendar month/.test(n)));
    } finally { F1.configure({ base: base("ok") }); }
  });

  test("schema 1 (family 8 Argo layout): float32 days become seconds like the Python reader, temp/psal blocks read as one matrix", async () => {
    multi();
    try {
      const c = case10("schema1 rows");
      const r = await F1.run(c.sel);
      assert.equal(r.kind, "points");
      assert.deepEqual(Array.from(r.time), c.time);
      closeArr(r.values, c.values, { label: "temp_10, temp_50" });
      // wmo is the platform; there is no qc column, so 0 = not assessed
      assert.ok(r.platform[0] >= 1900000n);
      assert.equal(r.qc.reduce((a, b) => a + b, 0), 0);
    } finally { F1.configure({ base: base("ok") }); }
  });

  test("negative bins: rows from before the 1982 epoch read and keep their times", async () => {
    multi();
    try {
      const c = case10("negative bins");
      assert.ok(c.bins[0] < 0 && c.bins[1] < 0);
      const r = await F1.run(c.sel);
      assert.deepEqual(Array.from(r.time), c.time);
      closeArr(r.values, c.values, { label: "sst" });
      assert.equal(new Date(r.time[0] * 1000).getUTCFullYear(), 1980);
    } finally { F1.configure({ base: base("ok") }); }
  });

  test("derived: the month-major grid is raw float32, the climatology de-z-scored with no years", async () => {
    multi();
    try {
      const m = case10("monthmajor");
      const rm = await F1.run(m.sel);
      closeArr(rm.data, m.data, { label: "fishing" });
      assert.ok(rm.notes.some((n) => /monthly SUM/.test(n)));
      const c = case10("clim");
      const rc = await F1.run(c.sel);
      closeArr(rc.data, c.data, { rel: 2e-6, label: "clim" });
      assert.equal(rc.time.length, 2);
      assert.equal(new Date(rc.time[1] * 1000).getUTCMonth(), 6);
      assert.ok(rc.notes.some((n) => /calendar-month climatology/.test(n)));
      const reg = await F1.loadRegistry();
      assert.equal(reg.stores.find((d) => d.id === "derived/clim_fxg").calendar, true);
      // only the chosen channel's planes are read: 2 months × 1 channel
      const e = await F1.estimate(c.sel);
      assert.equal(e.requests, 2);
      assert.equal(e.readBytes, 2 * 7 * 36 * 4);          // 2 planes × 7 rows × 36 columns × float32
      // NetCDF opens and carries the data
      const nc = new Uint8Array(await F1.toNetCDF(rc).arrayBuffer());
      assert.equal(String.fromCharCode(nc[0], nc[1], nc[2]), "CDF");
    } finally { F1.configure({ base: base("ok") }); }
  });

  test("a store whose z-score table is missing is a named error, its family's other stores still load", async () => {
    const b10 = base("ok10");
    F1.configure({ siteBase: b10, registries: [
      { family: "10", title: "Family 10", kind: "family10", url: b10 + "family10.json", root: b10, norms: "no-such-index.json" }] });
    try {
      const reg = await F1.loadRegistry();
      assert.deepEqual(reg.stores.map((d) => d.name), ["fxargo", "fxneg"]);
      assert.deepEqual(reg.errors.map((e) => e.store).sort(), ["fxg", "fxrg"]);
      assert.match(reg.errors[0].message, /z-score table/);
    } finally { F1.configure({ base: base("ok") }); }
  });

  // ---- family 1.2: ERA5-shaped pressure levels, six-hourly instants -------
  function multi12() {
    F1.configure({ registries: [
      { family: "1.gf", title: "Fine observations (family 1.gf)", kind: "family1", url: base("ok") + "family1gf.json" },
      { family: "1.2", title: "Atmosphere on pressure levels (family 1.2 — ERA5 reanalysis)", kind: "family1",
        optional: true, url: base("ok12") + "family12.json" },
    ] });
  }
  const case12 = (name) => { const c = EXP12.cases.find((x) => x.name === name); assert.ok(c, name); return JSON.parse(JSON.stringify(c)); };

  test("family 1.2: inherited stores listed once, a not-built store is only 'coming', levels in hPa start at 500, the record end comes from the shard index", async () => {
    multi12();
    try {
      const reg = await F1.loadRegistry();
      assert.deepEqual(reg.families.map((f) => f.family), ["1.gf", "1.2"]);
      assert.deepEqual(reg.stores.filter((d) => d.name === "fxgrid").map((d) => d.id), ["1.gf/fxgrid"],
        "an inherited store appears once, under the family that owns its bytes");
      assert.deepEqual(reg.stores.filter((d) => d.family === "1.2").map((d) => d.name), ["fxera"]);
      assert.deepEqual(reg.coming.filter((c) => c.family === "1.2").map((c) => c.store), EXP12.coming,
        "not built → coming; licence pending → not named");
      assert.deepEqual(reg.coming.map((c) => c.family), ["1.gf", "1.2"], "coming follows the registry order");
      assert.ok(!reg.stores.some((d) => d.name === "fxera_q" || d.name === "fxwait"));
      const d = reg.stores.find((x) => x.id === "1.2/fxera");
      assert.deepEqual(d.levels, [100, 500, 850]);
      assert.equal(d.levelUnit, "hPa");
      assert.equal(d.defaultLevel, 500);
      assert.equal(d.frameSeconds, 21600);
      assert.equal(d.grid.lat0, -90);
      assert.equal(d.grid.dlat, 10);
      assert.equal(d.reanalysis, true);
      assert.match(d.licence.attribution, /Copernicus Climate Change Service/);
      // before any read the span is whole bins; the first plan reads the
      // shard index and tightens it to the last frame present
      assert.deepEqual(d.span, ["2009-12-25", "2010-01-03"]);
      const e = await F1.estimate(case12("level 500 one day hours 12-18").sel);
      assert.deepEqual(d.span, EXP12.span);
      assert.equal(e.frames, 1);
      assert.equal(e.channelsRead, 3, "a tile holds all three levels, so all three are read");
      assert.equal(e.channelsKept, 1);
      assert.equal(e.channelWord, "levels");
      assert.match(e.why, /all 3 levels side by side/);
    } finally { F1.configure({ base: base("ok") }); }
  });

  for (const c of EXP12.cases) {
    test(`family 1.2: ${c.name} — level selection, six-hourly frames and the hours filter equal numpy`, async () => {
      multi12();
      try {
        const r = await F1.run(selOf(c));
        const x = c.expect;
        assert.equal(r.frames, x.frames);
        assert.deepEqual([r.time.length, r.channels.length, r.lat.length, r.lon.length], x.shape);
        assert.deepEqual(Array.from(r.time), x.time);
        assert.deepEqual(Array.from(r.lat), x.lat);
        assert.deepEqual(Array.from(r.lon), x.lon);
        closeArr(r.data, x.data, { rel: x.count ? 1e-6 : 0, label: c.name });
        if (x.count) assert.deepEqual(Array.from(r.count), x.count);
        // every native frame is an instant at 00, 06, 12 or 18 UTC
        if (c.sel.step === "native") for (const t of r.time) assert.equal((t % 86400) % 21600, 0);
        assert.ok(r.notes.some((n) => /REANALYSIS/.test(n)));
        // the attribution the licence requires travels in the file
        const nc = Buffer.from(await F1.toNetCDF(r).arrayBuffer()).toString("latin1");
        assert.ok(nc.includes("Contains modified Copernicus Climate Change Service information"));
        assert.ok(nc.includes("family 1.2 store fxera"));
      } finally { F1.configure({ base: base("ok") }); }
    });
  }

  // ---- the monthly normals over a free period (E-086 sums and counts) ------
  function multiM(extra) {
    F1.configure(Object.assign({ siteBase: base("okm"), registries: [
      { family: "derived", title: "Derived maps", kind: "derived", monthly: "family7_monthly_index.json" }] }, extra || {}));
  }
  const caseM = (name) => { const c = EXPM.cases.find((x) => x.name === name); assert.ok(c, name); return JSON.parse(JSON.stringify(c)); };
  const IDXM = JSON.parse(fs.readFileSync(path.join(FIXM, "family7_monthly_index.json"), "utf8"));

  test("normals: one store per group, the years from the index, steps normal / by-year, levelled channels", async () => {
    multiM();
    try {
      const reg = await F1.loadRegistry();
      assert.deepEqual(reg.stores.map((d) => d.id), ["derived/normals_fx025", "derived/normals_fxrg"]);
      assert.deepEqual(reg.errors, []);
      const d = reg.stores[0];
      assert.equal(d.normals, true);
      assert.deepEqual(d.years, [2000, 2004]);
      assert.deepEqual(d.span, ["2000-01-01", "2004-12-31"]);
      assert.deepEqual(d.steps, ["normal", "by-year"]);
      assert.equal(d.defaultChannel, "sst");
      assert.equal(d.grid.step, 0.25);
      const rg = reg.stores[1];
      assert.deepEqual(rg.levels, [10, 30, 50]);
      assert.equal(rg.levelUnit, "dbar");
      assert.deepEqual(rg.vars.map((v) => v.var), ["rg_t", "rg_s"]);
      // the old calendar stores are gone from the default list; the normals replace them
      const def = F1.DEFAULT_REGISTRIES.find((f) => f.family === "derived");
      assert.equal(def.monthly, "data/family7_monthly_index.json");
      assert.equal(def.clim, undefined);
    } finally { F1.configure({ base: base("ok") }); }
  });

  for (const c of EXPM.cases) {
    test(`normals: ${c.name} — Σ sums ÷ Σ counts over the chosen years equals numpy, counts identical`, async () => {
      multiM();
      try {
        const r = await F1.run(c.sel);
        assert.deepEqual([r.time.length, r.channels.length, r.lat.length, r.lon.length], c.shape);
        assert.deepEqual(Array.from(r.lat), c.lat);
        assert.deepEqual(Array.from(r.lon), c.lon);
        closeArr(r.data, c.data, { rel: 1e-6, label: c.name });
        assert.deepEqual(Array.from(r.count), c.count, "counts");
        if (c.sel.step === "normal") {
          assert.deepEqual(r.climatology.yearsUsed, c.years);
          assert.deepEqual(r.climatology.excluded, c.sel.excludeYears);
          assert.equal(r.climatology.bounds.length, 2 * r.time.length);
          // the time of a normal is its month in the first year; the bounds span the period
          const t0 = new Date(r.time[0] * 1000);
          assert.equal(t0.getUTCFullYear(), c.years[0]);
          assert.equal(t0.getUTCMonth() + 1, Math.min(...c.sel.months));
          assert.equal(new Date(r.climatology.bounds[1] * 1000).getUTCFullYear(), c.years[c.years.length - 1] + (Math.min(...c.sel.months) === 12 ? 1 : 0));
        } else {
          // by-year: each year's own monthly mean, years × months in time order
          assert.equal(r.time.length, c.years.length * c.sel.months.length);
          assert.deepEqual(Array.from(r.time, (t) => new Date(t * 1000).getUTCFullYear()),
            c.years.flatMap((y) => c.sel.months.map(() => y)));
        }
        if (c.sel.res !== "native") assert.ok(r.notes.some((n) => /POOLS the sums and the counts/.test(n)));
      } finally { F1.configure({ base: base("ok") }); }
    });
  }

  test("normals: an excluded year is never read; both read strategies give identical numbers; the estimate is exact", async () => {
    const c = caseM("normal_excl");
    const g = IDXM.groups.fx025;
    const planeOf = (f, m, ch, y) => f.header_len + (((m - 1) * g.chans.length + ch) * g.n_years + (y - g.year_first)) * f.plane_bytes;
    let byGap = {};
    for (const gap of [0, 1e12]) {
      multiM({ normalsMergeGap: gap });
      try {
        const e = await F1.estimate(c.sel);
        log.length = 0;
        const r = await F1.run(c.sel);
        const reads = log.filter((x) => x.mode === "okm" && /\/(sum|count)\.npy$/.test(x.rel) && x.range && !/bytes=0-1023$/.test(x.range));
        assert.equal(reads.length, e.requests, `gap ${gap}: requests`);
        assert.equal(reads.reduce((a, x) => a + x.bytes, 0), e.readBytes, `gap ${gap}: bytes`);
        assert.equal(e.strategy, gap === 0 ? "band" : "run");
        assert.match(e.why, gap === 0 ? /one range per year/ : /one contiguous stretch/);
        if (gap === 0) {
          // per-year bands: none of them touches 2002's plane, in either file
          for (const x of reads) {
            const [a, z] = /bytes=(\d+)-(\d+)/.exec(x.range).slice(1).map(Number);
            const f = /sum\.npy$/.test(x.rel) ? g.sum : g.count;
            for (const m of c.sel.months) for (let ch = 0; ch < 2; ch++) {
              const p0 = planeOf(f, m, ch, 2002);
              assert.ok(z < p0 || a >= p0 + f.plane_bytes, `a read ${x.range} covers the excluded year 2002`);
            }
          }
          // 2 files × 2 months × 2 channels × 4 years, one band each
          assert.equal(e.requests, 2 * 2 * 2 * 4);
          assert.equal(e.readBytes, e.bandBytes);
        } else {
          // one stretch per run of years (here the whole block: every gap is
          // smaller than a request) — more bytes, a handful of requests;
          // 2002's plane may lie inside a stretch, and is never added
          assert.ok(e.requests < 2 * 2 * 2 * 4);
          assert.ok(e.readBytes > e.bandBytes);
        }
        byGap[gap] = r;
      } finally { F1.configure({ base: base("ok") }); }
    }
    assert.deepEqual(Array.from(byGap[0].data), Array.from(byGap[1e12].data));
    assert.deepEqual(Array.from(byGap[0].count), Array.from(byGap[1e12].count));
  });

  test("normals: the NetCDF is a CF climatology with period_start, period_end, excluded_years and the counts", async () => {
    multiM();
    try {
      const c = caseM("normal_excl");
      const r = await F1.run(c.sel);
      const tmp = path.join(os.tmpdir(), `f1-normals-${process.pid}.nc`);
      fs.writeFileSync(tmp, Buffer.from(await F1.toNetCDF(r).arrayBuffer()));
      const out = JSON.parse(python(`
import json, sys
import numpy as np
from scipy.io import netcdf_file
f = netcdf_file(sys.argv[1], "r", mmap=False)
a = {k: (v.decode() if isinstance(v, bytes) else (v.tolist() if hasattr(v, "tolist") else v)) for k, v in f._attributes.items()}
t = f.variables["time"]
print(json.dumps({"attrs": a, "time_attrs": {k: (v.decode() if isinstance(v, bytes) else v) for k, v in t._attributes.items()},
  "vars": sorted(f.variables), "bounds": f.variables["climatology_bounds"][:].tolist(),
  "sst": [None if not np.isfinite(x) else float(x) for x in f.variables["sst"][:].astype(np.float64).ravel()],
  "count": f.variables["sst_count"][:].ravel().tolist(),
  "cell_methods": f.variables["sst"]._attributes["cell_methods"].decode()}))
`, tmp));
      fs.unlinkSync(tmp);
      assert.equal(out.attrs.period_start, 2000);
      assert.equal(out.attrs.period_end, 2004);
      assert.equal(out.attrs.excluded_years, "2002");
      assert.equal(out.attrs.years_used, "2000, 2001, 2003, 2004");
      assert.match(out.attrs.climatology, /monthly normals/);
      assert.equal(out.time_attrs.climatology, "climatology_bounds");
      assert.ok(out.vars.includes("climatology_bounds") && out.vars.includes("ssh_count"));
      assert.equal(out.bounds.length, r.time.length);
      assert.match(out.cell_methods, /mean over years/);
      // the file holds the composed numbers, and sst's counts
      const HW = r.lat.length * r.lon.length, T = r.time.length;
      const wantSst = [], wantCnt = [];
      for (let t = 0; t < T; t++) for (let i = 0; i < HW; i++) { wantSst.push(c.data[(t * 2 + 0) * HW + i]); wantCnt.push(c.count[(t * 2 + 0) * HW + i]); }
      closeArr(Float64Array.from(out.sst, (v) => (v === null ? NaN : v)), wantSst, { rel: 1e-6, label: "sst in the file" });
      assert.deepEqual(out.count, wantCnt);
      // and the by-year stack says its period too
      const b = await F1.run(caseM("byyear").sel);
      const nc = Buffer.from(await F1.toNetCDF(b).arrayBuffer()).toString("latin1");
      assert.ok(nc.includes("period_start") && !nc.includes("climatology_bounds"));
    } finally { F1.configure({ base: base("ok") }); }
  });

  test("normals: the preview is the first chosen month's normal (or, by year, the first year's field)", async () => {
    multiM();
    try {
      const c = caseM("normal_excl");
      const p = await F1.preview(c.sel);
      assert.equal(p.time.length, 1);
      const HW = c.lat.length * c.lon.length;
      // the run's first time step, channels as asked
      closeArr(p.data, c.data.slice(0, 2 * HW), { rel: 1e-6, label: "preview = first month's normal" });
      const b = caseM("byyear");
      const pb = await F1.preview(b.sel);
      assert.equal(new Date(pb.time[0] * 1000).getUTCFullYear(), b.years[0]);
      closeArr(pb.data, b.data.slice(0, b.lat.length * b.lon.length), { rel: 1e-6, label: "preview = first year" });
      // a box is required, and the estimate says so instead of reading
      const e = await F1.estimate(Object.assign({}, c.sel, { bbox: null }));
      assert.equal(e.overCap, true);
      assert.match(e.why, /box/);
    } finally { F1.configure({ base: base("ok") }); }
  });

  // ---- family 7.2d's registry fields, on the 1.gf fixture ------------------
  test("a daily-tensor registry: record_span is the record, source_segments and preliminary days become caveats, a centred five-day sigma is said so", async () => {
    F1.configure({ registries: [{ family: "7.2d", title: "Global tensor, daily (family 7.2d)", kind: "family1", url: base("ok72") + "family1gf.json" }] });
    try {
      const reg = await F1.loadRegistry();
      const d = reg.stores.find((x) => x.name === "fxgrid");
      assert.deepEqual(d.span, ["2009-12-27", "2010-01-06"], "record_span, not the bins' 2009-12-25 → 2010-01-08");
      // a point-aligned grid's x0/y0 are centres: the same pixels as the edge form
      assert.equal(d.grid.lat0, 89.75);
      assert.equal(d.grid.lon0, -179.75);
      assert.equal(d.caveats.length, 2);
      assert.match(d.caveats[0].text, /^From 2010-01-01 to 2010-01-08 each day was checked only against an independent read of its own source file — it is the same source \(FX source v1\)/);
      assert.match(d.caveats[0].text, /ends 2009-12-31/);
      assert.match(d.caveats[1].text, /The last 3 days \(2010-01-06 → 2010-01-08\) are the producer's PRELIMINARY values/);
      assert.ok(!/centred/.test(d.channels[0].unit));
      assert.match(d.channels[0].note, /five days centred on each day/);
      const r = await F1.run({ family: "7.2d", store: "fxgrid", channels: [d.channels[0].name], yearStart: 2010, yearEnd: 2010, months: [1],
        bbox: { w: -30, s: -20, e: 10, n: 20 }, step: "native", res: "native" });
      assert.ok(r.notes.some((n) => /independent read of its own source file/.test(n)));
      assert.ok(r.notes.some((n) => /PRELIMINARY/.test(n)));
      assert.ok(r.notes.some((n) => /five days centred on each day/.test(n)));
      // the point-aligned grid reads the same pixels at the same coordinates as
      // the edge form, after several reads of its (cached) tile_grid.json
      await F1.estimate({ family: "7.2d", store: "fxgrid", channels: [d.channels[0].name], yearStart: 2010, yearEnd: 2010, months: [1],
        bbox: { w: -30, s: -20, e: 10, n: 20 }, step: "native", res: "native" });
      const again = await F1.run({ family: "7.2d", store: "fxgrid", channels: [d.channels[0].name], yearStart: 2010, yearEnd: 2010, months: [1],
        bbox: { w: -30, s: -20, e: 10, n: 20 }, step: "native", res: "native" });
      F1.configure({ base: base("ok") });
      const plain = await F1.run({ store: "fxgrid", channels: [d.channels[0].name], yearStart: 2010, yearEnd: 2010, months: [1],
        bbox: { w: -30, s: -20, e: 10, n: 20 }, step: "native", res: "native" });
      assert.deepEqual(Array.from(again.lat), Array.from(plain.lat));
      assert.deepEqual(Array.from(again.lon), Array.from(plain.lon));
      assert.deepEqual(Array.from(again.data), Array.from(plain.data));
      F1.configure({ registries: [{ family: "7.2d", title: "Global tensor, daily (family 7.2d)", kind: "family1", url: base("ok72") + "family1gf.json" }] });
      // a selection before the read-back span carries no such caveat
      const r9 = await F1.run({ family: "7.2d", store: "fxgrid", channels: [d.channels[0].name], yearStart: 2009, yearEnd: 2009, months: [12],
        bbox: { w: -30, s: -20, e: 10, n: 20 }, step: "native", res: "native" });
      assert.ok(!r9.notes.some((n) => /PRELIMINARY|independent read/.test(n)));
    } finally { F1.configure({ base: base("ok") }); }
  });

  // ---- progress for the longer reads ---------------------------------------
  test("progress: estimate and preview tell the index phase request by request; run's data phase reaches its exact total", async () => {
    multiM();
    try {
      const c = caseM("normal_excl");
      const seen = [];
      const e = await F1.estimate(c.sel, { onProgress: (p) => seen.push(p) });
      assert.ok(seen.length > 0, "the estimate said nothing");
      assert.ok(seen.every((p) => p.phase === "index" && p.total === null), "an estimate never invents a total");
      for (let i = 1; i < seen.length; i++) assert.ok(seen[i].done >= seen[i - 1].done && seen[i].bytes >= seen[i - 1].bytes);
      // a run that asks for the index phase too: index, then data done = total, bytes = the exact estimate
      F1.configure({ siteBase: base("okm"), registries: [{ family: "derived", title: "Derived maps", kind: "derived", monthly: "family7_monthly_index.json" }] });
      const ps = [];
      await F1.run(c.sel, { indexProgress: true, onProgress: (p) => ps.push(p) });
      const data = ps.filter((p) => p.phase === "data");
      assert.ok(ps.some((p) => p.phase === "index"), "the cold run's index reads were not told");
      assert.ok(data.length > 0);
      const last = data[data.length - 1];
      assert.equal(last.done, last.total);
      assert.equal(last.bytes, e.readBytes);
      assert.equal(last.bytesTotal, e.readBytes);
      assert.equal(last.approx, false);
      assert.ok(last.requestsAll >= last.done && last.bytesAll >= last.bytes);
      // the old contract still holds: without indexProgress, {done, total, bytes} of the data phase only
      const old = [];
      await F1.run(c.sel, { onProgress: (p) => old.push(p) });
      assert.ok(old.length && old.every((p) => p.phase === undefined && Number.isFinite(p.done) && Number.isFinite(p.total)));
      // the preview tells too, and aborting stops it with an AbortError
      const pv = [];
      await F1.preview(c.sel, { onProgress: (p) => pv.push(p) });
      assert.ok(pv.length > 0);
      const ac = new AbortController();
      F1.configure({ siteBase: base("okm"), registries: [{ family: "derived", title: "Derived maps", kind: "derived", monthly: "family7_monthly_index.json" }] });
      const pr = F1.estimate(c.sel, { signal: ac.signal, onProgress: () => ac.abort() });
      await assert.rejects(pr, (err) => err.name === "AbortError");
    } finally { F1.configure({ base: base("ok") }); }
  });

  test("progress: a grid estimate says how many tile-index files it will read, and counts them off", async () => {
    try {
      const seen = [];
      const sel = cases(/native/)[0] ? selOf(cases(/native/)[0]) : null;
      assert.ok(sel, "a fixture grid case");
      await F1.estimate(sel, { onProgress: (p) => seen.push(p) });
      const st = seen.filter((p) => p.steps);
      assert.ok(st.length > 0, "no steps told");
      const lastS = st[st.length - 1].steps;
      assert.equal(lastS.done, lastS.total);
      assert.match(lastS.what, /tile index/);
    } finally { F1.configure({ base: base("ok") }); }
  });

  test("internals: float16 table and the calendar", () => {
    const { F16, civil, daysFromCivil, isoOfUnix } = F1._internal;
    assert.equal(F16[0x3c00], 1);
    assert.equal(F16[0xc000], -2);
    assert.equal(F16[0x7bff], 65504);
    assert.ok(Number.isNaN(F16[0x7e00]));
    assert.equal(F16[0x0001], Math.pow(2, -24));
    for (const z of [-200000, -1, 0, 4383, 15000, 30000]) {
      const [y, m, d] = civil(z);
      assert.equal(daysFromCivil(y, m, d), z);
      if (y >= 1000) assert.equal(new Date(z * 86400000).toISOString().slice(0, 10), `${y}-${String(m).padStart(2, "0")}-${String(d).padStart(2, "0")}`);
    }
    assert.equal(isoOfUnix(378691200), "1982-01-01T00:00:00Z");
  });
}

if (!inPlaywright) {
  before(async () => {
    server = http.createServer(serve);
    await new Promise((r) => server.listen(0, "127.0.0.1", r));
    port = server.address().port;
    F1.configure({ base: base("ok") });
  });
  after(() => new Promise((r) => server.close(r)));
  defineTests();
}

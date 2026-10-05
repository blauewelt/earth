/*
 * src/f1data.js — the family 1.gf reader behind the Data tab (E-084 §4).
 *
 * PLAIN ENGLISH. Family 1.gf is the set of public observation stores on the
 * Hugging Face dataset `chfrank/earth-tensors` under `tensors/family1_gf/`,
 * each kept at its own native resolution. This file reads them straight from
 * the Hub in the browser (or from node ≥ 18 for tests), by HTTP Range reads
 * only, and turns a selection — store, channels, period, months, hours, box,
 * time step, resolution — into arrays, a NetCDF file or a CSV file. It has no
 * DOM code and no dependency beyond the vendored zstd decoder (lib/fzstd.js).
 *
 * TWO LAYOUTS, both defined by Python and read here byte for byte:
 *
 *   GRID ("tier G, sharded", ml/family1/sharded.py). Per group:
 *     tile_grid.json        H, W, C, tile size T, dtype, frames per bin F,
 *                           frame_seconds, the affine coordinate rule
 *                           (lon = x0 + (col+.5)·dx, lat = y0 + (row+.5)·dy),
 *                           the .idx.npy header length.
 *     shard_index.npy       one row per stored five-day bin (structured npy):
 *                           bin, frame_mask (bit f = frame f present), nbytes …
 *     <yyyy>/bin_<NNNN>.zst      the bin's tiles, each its own zstd frame.
 *     <yyyy>/bin_<NNNN>.idx.npy  int64 [F, nty, ntx, 2] of (offset, length):
 *                           length > 0 a stored tile; length 0 a present
 *                           frame's tile with no valid pixel (all missing);
 *                           offset −1 the FRAME is absent — it is left out of
 *                           `time`, never zero-filled.
 *     A tile is T×T×C, C order [row, col, channel], little-endian; float16
 *     missing = NaN, uint8 missing = 255. Edge tiles are padded.
 *
 *   POINTS ("tier P", ml/family10_store.py). N rows sorted by (bin, time_s):
 *     time_s.npy int32 (schema 2) or int64 (schema 3) seconds since
 *     1982-01-01T00:00:00Z · lat/lon float32 · values float16 [N, C] ·
 *     platform int64 · qc uint8 · bin_offsets.npy int64 CSR over bins
 *     bin_first .. bin_last (bin = floor(time_s / 432000)). A column over the
 *     Hub's file limit is stored as parts (store.json `hub_split`) and read
 *     across them.
 *
 * RULES (plan §4, Chris 2026-09-14 "no download failed → skip"):
 *   - every read sends Range and REFUSES a 200 — a host that ignores Range is
 *     sending the whole file; any failed read rejects the whole run with the
 *     URL in the message; at most 6 requests in flight; AbortSignal honoured.
 *   - physical conversions documented by the store are applied and the unit
 *     says so (irtb's uint8 "K − 160" → kelvin; pace4k's "nm above 400" →
 *     nm); oc4k/pace4k chlorophyll stays log10, as its unit says.
 *   - means are NaN-aware and come with a count; they accumulate streaming,
 *     so memory follows the OUTPUT, not the frames read.
 *
 * SELECTION SEMANTICS (what the tests in tests/f1data.test.mjs pin):
 *   - period: whole calendar years yearStart..yearEnd (UTC); months: 1..12,
 *     empty = all. A grid frame belongs to the month and year of its own
 *     START time; a point row to those of its own time_s.
 *   - hours [h0, h1): half-open UTC hours, wrapping when h0 > h1 (21..3 is
 *     night), h0 == h1 selects nothing, [0, 24] everything. Applied to grid
 *     frames only when frame_seconds < 86400, to point rows always.
 *   - box: a pixel is in when its CENTRE is inside [w, e] × [s, n], edges
 *     included. w > e crosses the dateline. Output latitude ascends (south
 *     first); output longitude is MONOTONIC: a dateline box runs w .. 180 ..
 *     e + 360 (CF allows longitudes past 180; subtract 360 if you need −180..180).
 *   - res 0.25 | 1: cells aligned to whole multiples of res (cell k spans
 *     [k·res, (k+1)·res)); a grid gets the cells its box's pixels fall in, a
 *     binned point store every cell of the box (a point exactly on the box's
 *     east or north edge joins the last cell).
 *   - step: native = every frame apart (grids) / every row (points);
 *     pentad = the five-day bin (time = bin start); month = calendar month
 *     (time = the 1st, 00:00 UTC); all = one mean (time = the first frame's
 *     start for grids, the first contributing bin's start for points).
 *     Points are BINNED (a grid Result with mean + count) when res is not
 *     'native' or step is not 'native'; binning defaults to 0.25° and to the
 *     five-day bin for whichever of the two was left native.
 *   - count is a Uint16Array, or a Uint32Array when the largest possible
 *     count would not fit 16 bits (irtb's "all" over decades, say).
 *   - absent frames (index offset −1) are not in `time`; a present frame
 *     whose box tiles are empty (length 0) is in `time` and all NaN.
 *   - point previews start at the period's first second (an interpolation
 *     search on time_s) and stop at PREVIEW_MAX_ROWS rows or after
 *     PREVIEW_SCAN_BYTES without a match; Result.truncated says so.
 *
 * THE CONTRACT (plan §4 — names and shapes fixed; optional additions marked +):
 *   F1Data.configure({base, fetch, concurrency})  base = URL of the family1_gf
 *        folder (the one holding family1gf.json); default the Hub's
 *        resolve/main URL. Clears every cache.
 *   F1Data.loadRegistry() → {stores:[{name,title,gist,kind,channels:[{name,
 *        unit,min,max,+note,+storedUnit}],span,frameSeconds,framesPerBin,
 *        grid,folderUrl,+code,+cadence,+subDaily,+N,+binFirst,+binLast,
 *        +groups}]}
 *   F1Data.estimate(sel, +{signal, exactDays}) → {requests, readBytes, outBytes,
 *        frames|rows, overCap, why, +exact, +shape}
 *   F1Data.run(sel, {onProgress, signal}) → Result
 *   F1Data.preview(sel, {signal}) → Result for one frame / one bin
 *   F1Data.toNetCDF(result) → Blob     F1Data.toCSV(result) → Blob
 *   sel = {store, channels, yearStart, yearEnd, months, hours:[h0,h1]|null, +days:[d0,d1]|null,
 *          bbox:{w,s,e,n}|null, step, res, +group}
 *   Result(grid) = {kind:'grid', store, channels, units, lat, lon, time, data,
 *          count, sel, +notes, +frames, +group, +title, +source, +stats
 *          {requests, bytes, ms}, +binnedFrom:'points', +rowsRead,
 *          +truncated}
 *   Result(points) = {kind:'points', store, channels, units, time, lat, lon,
 *          values, platform (BigInt64Array, exact), qc (Uint8Array), sel,
 *          +notes, +rowsRead, +title, +source, +stats, +truncated}
 *   F1Data.CAPS, F1Data.DEFAULT_BASE (+); F1Data._internal is for tests only.
 */
(function (root, factory) {
  if (typeof module === "object" && module.exports) {
    module.exports = factory(function () { return require("../lib/fzstd.js"); });
  } else {
    root.F1Data = factory(function () { return root.fzstd; });
  }
})(typeof self !== "undefined" ? self : this, function (getZstd) {
  "use strict";

  // ------------------------------------------------------------ constants --
  var DEFAULT_BASE = "https://huggingface.co/datasets/chfrank/earth-tensors/resolve/main/tensors/family1_gf/";
  var EPOCH_UNIX = 378691200;            // 1982-01-01T00:00:00Z
  var EPOCH_DAYS = 4383;                 // days 1970-01-01 → 1982-01-01
  var BIN_S = 432000;                    // one five-day bin
  var MB = 1e6;
  var CAP_READ = 600 * MB;               // plan §5
  var CAP_OUT = 400 * MB;
  var MAX_CONCURRENCY = 6;
  var GAP_IDX = 65536;                   // merge index ranges this close (one request per bin, in practice)
  var GAP_TILE = 16384;                  // merge tile ranges this close
  var MAX_RANGE = 8 * MB;                // never ask for more in one request
  var EXACT_IDX_BINS = 24;               // estimate reads every index up to here
  var SAMPLE_IDX_BINS = 8;               // … and a sample beyond it
  var CHUNK_BYTES = 8 * MB;              // point stores: bytes per row chunk
  var IDX_CACHE_BYTES = 32 * MB;
  var PREVIEW_SCAN_BYTES = 32 * MB;      // a point preview gives up looking after this
  var DAY_TRIM_WINDOWS = 24;             // days narrower than a bin: rows searched for at most this many windows
  var PREVIEW_MAX_ROWS = 50000;          // … and stops once it holds this many rows
  var PREVIEW_CHUNK_ROWS = 65536;
  var PREVIEW_GRID_TRIES = 8;            // frames a grid preview tries for one with data
  var U8_MISSING = 255;

  if (new Uint8Array(new Uint16Array([1]).buffer)[0] !== 1) {
    throw new Error("f1data.js assumes a little-endian host");
  }

  var cfg = { base: DEFAULT_BASE, fetch: null, concurrency: MAX_CONCURRENCY };
  var cache = new Map();                 // url/key → Promise
  var idxCache = new Map();              // url|off|len → Uint8Array (LRU)
  var idxCacheBytes = 0;

  function fetchFn() {
    var f = cfg.fetch || (typeof fetch === "function" ? fetch : null);
    if (!f) throw new Error("no fetch() available");
    return f;
  }

  function configure(o) {
    o = o || {};
    if (o.base != null) {
      var b = String(o.base);
      cfg.base = b.endsWith("/") ? b : b + "/";
    }
    if (o.fetch !== undefined) cfg.fetch = o.fetch;
    if (o.concurrency != null) {
      cfg.concurrency = Math.max(1, Math.min(MAX_CONCURRENCY, o.concurrency | 0));
    }
    cache.clear();
    idxCache.clear();
    idxCacheBytes = 0;
    return { base: cfg.base, concurrency: cfg.concurrency };
  }

  // --------------------------------------------------------- small helpers --
  function abortError(signal) {
    if (signal && signal.reason instanceof Error) return signal.reason;
    var e = new Error("aborted");
    e.name = "AbortError";
    return e;
  }

  function checkAbort(signal) {
    if (signal && signal.aborted) throw abortError(signal);
  }

  function fmtMB(n) {
    if (n < 0.1 * MB) return (n / 1e3).toFixed(n < 1e4 ? 1 : 0) + " kB";
    return (n / MB).toFixed(n < 10 * MB ? 1 : 0) + " MB";
  }

  // days since 1970-01-01 → [y, m, d]   (H. Hinnant, proleptic Gregorian)
  function civil(z) {
    z += 719468;
    var era = Math.floor(z / 146097);
    var doe = z - era * 146097;
    var yoe = Math.floor((doe - Math.floor(doe / 1460) + Math.floor(doe / 36524)
      - Math.floor(doe / 146096)) / 365);
    var y = yoe + era * 400;
    var doy = doe - (365 * yoe + Math.floor(yoe / 4) - Math.floor(yoe / 100));
    var mp = Math.floor((5 * doy + 2) / 153);
    var d = doy - Math.floor((153 * mp + 2) / 5) + 1;
    var m = mp < 10 ? mp + 3 : mp - 9;
    return [m <= 2 ? y + 1 : y, m, d];
  }

  function daysFromCivil(y, m, d) {
    y -= m <= 2 ? 1 : 0;
    var era = Math.floor(y / 400);
    var yoe = y - era * 400;
    var doy = Math.floor((153 * (m + (m > 2 ? -3 : 9)) + 2) / 5) + d - 1;
    var doe = yoe * 365 + Math.floor(yoe / 4) - Math.floor(yoe / 100) + doy;
    return era * 146097 + doe - 719468;
  }

  // seconds since 1982 → {y, m, h}
  function ymh82(t) {
    var days = Math.floor(t / 86400);
    var c = civil(days + EPOCH_DAYS);
    return { y: c[0], m: c[1], d: c[2], h: Math.floor((t - days * 86400) / 3600) };
  }

  function sec82OfCivil(y, m, d) {
    return (daysFromCivil(y, m, d) - EPOCH_DAYS) * 86400;
  }

  function isoOfUnix(u) {
    var days = Math.floor(u / 86400);
    var s = u - days * 86400;
    var c = civil(days);
    var hh = Math.floor(s / 3600), mm = Math.floor((s % 3600) / 60), ss = Math.floor(s % 60);
    var yy = c[0] < 0 ? "-" + String(-c[0]).padStart(6, "0")
      : String(c[0]).padStart(4, "0");
    return yy + "-" + p2(c[1]) + "-" + p2(c[2]) + "T" + p2(hh) + ":" + p2(mm) + ":" + p2(ss) + "Z";
  }
  function p2(n) { return n < 10 ? "0" + n : "" + n; }

  function isoDate82(t) { return isoOfUnix(EPOCH_UNIX + t).slice(0, 10); }

  function hourOk(h, hours) {
    if (!hours) return true;
    var h0 = hours[0], h1 = hours[1];
    if (h0 === h1) return false;
    return h0 < h1 ? (h >= h0 && h < h1) : (h >= h0 || h < h1);
  }

  // float16 → float32 lookup, built once (Float16Array is not assumed)
  var F16 = (function () {
    var t = new Float32Array(65536);
    for (var h = 0; h < 65536; h++) {
      var s = h >> 15, e = (h >> 10) & 31, f = h & 1023, v;
      if (e === 0) v = f * Math.pow(2, -24);
      else if (e === 31) v = f ? NaN : Infinity;
      else v = (1 + f / 1024) * Math.pow(2, e - 15);
      t[h] = s ? -v : v;
    }
    return t;
  })();

  // ---------------------------------------------------------- concurrency --
  var active = 0;
  var waiters = [];
  function acquire(signal) {
    if (active < cfg.concurrency) { active++; return Promise.resolve(); }
    return new Promise(function (res, rej) {
      var w = { res: res, rej: rej };
      waiters.push(w);
      if (signal) {
        signal.addEventListener("abort", function () {
          var i = waiters.indexOf(w);
          if (i >= 0) { waiters.splice(i, 1); rej(abortError(signal)); }
        }, { once: true });
      }
    });
  }
  function release() {
    var w = waiters.shift();
    if (w) w.res(); else active--;
  }

  // ------------------------------------------------------------ the reads --
  // ONE function issues every HTTP request in this file. It asks for a Range,
  // demands 206, checks Content-Range and the length, retries a dropped
  // connection / 429 / 5xx twice, and otherwise rejects with the URL.
  function refuse(msg, url) {
    var e = new Error(msg + " — " + url);
    e.url = url;
    e.noRetry = true;
    return e;
  }

  async function rangeRead(url, off, len, ctx, opts) {
    ctx = ctx || {};
    opts = opts || {};
    var signal = ctx.signal;
    if (len === 0) return new Uint8Array(0);
    var whole = len == null;
    var hdr = whole ? "bytes=" + off + "-" : "bytes=" + off + "-" + (off + len - 1);
    var lastErr = null;
    for (var attempt = 0; attempt < 3; attempt++) {
      checkAbort(signal);
      await acquire(signal);
      var res = null;
      try {
        checkAbort(signal);
        res = await fetchFn()(url, { headers: { Range: hdr }, signal: signal });
        if (res.status === 206) {
          var buf = new Uint8Array(await res.arrayBuffer());
          var cr = res.headers.get("content-range") || "";
          var m = /bytes\s+(\d+)-(\d+)\/(\d+|\*)/.exec(cr);
          if (m && Number(m[1]) !== off) {
            throw refuse("asked for " + hdr + ", the host answered " + cr, url);
          }
          if (whole) {
            if (m && m[3] !== "*" && Number(m[2]) + 1 !== Number(m[3])) {
              throw refuse("asked for the whole file, got " + cr, url);
            }
          } else if (buf.length !== len && !(opts.allowShort && buf.length > 0 && buf.length < len)) {
            throw refuse("asked for " + len + " bytes (" + hdr + "), got " + buf.length, url);
          }
          if (ctx.stats) { ctx.stats.requests++; ctx.stats.bytes += buf.length; }
          if (ctx.onRead) ctx.onRead(buf.length);
          return buf;
        }
        try { if (res.body && res.body.cancel) res.body.cancel(); } catch (e) { /* ignore */ }
        if (res.status === 200) {
          throw refuse("HTTP 200 where 206 was asked for (" + hdr + "): the host ignored the Range " +
            "and is sending the whole file; refused", url);
        }
        if (res.status === 416 && opts.allowShort) {
          throw refuse("HTTP 416 for " + hdr, url);
        }
        var e2 = new Error("HTTP " + res.status + " for " + url + " (" + hdr + ")");
        e2.url = url;
        if (!(res.status === 429 || res.status >= 500)) e2.noRetry = true;
        throw e2;
      } catch (e) {
        if (signal && signal.aborted) throw abortError(signal);
        if (e && e.name === "AbortError") throw e;
        if (e && e.noRetry) throw e;
        lastErr = e;
      } finally {
        release();
      }
      await new Promise(function (r) { setTimeout(r, 400 * (attempt + 1)); });
    }
    var msg = lastErr && lastErr.message ? lastErr.message : String(lastErr);
    var fin = new Error((msg.indexOf(url) >= 0 ? msg : msg + " — " + url) + " (after 3 attempts)");
    fin.url = url;
    throw fin;
  }

  function cached(key, make) {
    var p = cache.get(key);
    if (!p) {
      p = make();
      cache.set(key, p);
      p.catch(function () { if (cache.get(key) === p) cache.delete(key); });
    }
    return p;
  }

  function readJSON(url, ctx) {
    return cached("json:" + url, async function () {
      var b = await rangeRead(url, 0, null, { signal: ctx && ctx.signal, stats: ctx && ctx.stats });
      try {
        return JSON.parse(new TextDecoder("utf-8").decode(b));
      } catch (e) {
        throw new Error("not JSON: " + url + " (" + e.message + ")");
      }
    });
  }

  // ------------------------------------------------------------ npy files --
  var DT_SIZE = { b1: 1, i1: 1, u1: 1, i2: 2, u2: 2, f2: 2, i4: 4, u4: 4, f4: 4, i8: 8, u8: 8, f8: 8 };

  function dtypeSize(d) {
    var m = /^[<>|=]?([a-zA-Z])(\d+)$/.exec(d);
    if (!m) throw new Error("unsupported npy dtype " + d);
    if (m[1] === "U") return 4 * Number(m[2]);
    if (m[1] === "S" || m[1] === "a" || m[1] === "V") return Number(m[2]);
    var s = DT_SIZE[m[1] + m[2]];
    if (!s) throw new Error("unsupported npy dtype " + d);
    return s;
  }

  function parseNpyHeader(u8, url) {
    if (u8.length < 10 || u8[0] !== 0x93 || String.fromCharCode(u8[1], u8[2], u8[3], u8[4], u8[5]) !== "NUMPY") {
      throw new Error("not an .npy file: " + url);
    }
    var major = u8[6], hl, start;
    if (major === 1) { hl = u8[8] | (u8[9] << 8); start = 10; }
    else { hl = (u8[8] | (u8[9] << 8) | (u8[10] << 16)) + u8[11] * 16777216; start = 12; }
    var dataOffset = start + hl;
    if (u8.length < dataOffset) return { need: dataOffset };
    var txt = "";
    for (var i = start; i < dataOffset; i++) txt += String.fromCharCode(u8[i]);
    var fo = /'fortran_order':\s*(True|False)/.exec(txt);
    var sh = /'shape':\s*\(([^)]*)\)/.exec(txt);
    var dl = /'descr':\s*(\[.*\])\s*,\s*'fortran_order'/.exec(txt);
    var ds = /'descr':\s*'([^']*)'/.exec(txt);
    if (!fo || !sh || !(dl || ds)) throw new Error("unreadable .npy header " + JSON.stringify(txt) + ": " + url);
    if (fo[1] === "True") throw new Error("Fortran-ordered .npy is not supported: " + url);
    var shape = sh[1].split(",").map(function (s) { return s.trim(); })
      .filter(function (s) { return s.length; }).map(Number);
    var out = { dataOffset: dataOffset, shape: shape };
    if (dl) {
      // a structured dtype: [('bin', '<i4'), ('valid_pixels', '<i8', (3,)), …]
      var fields = [], off = 0, re = /\(\s*'([^']+)'\s*,\s*'([^']+)'\s*(?:,\s*\(([^)]*)\))?\s*\)/g, mm;
      while ((mm = re.exec(dl[1]))) {
        var sub = (mm[3] || "").split(",").map(function (s) { return s.trim(); })
          .filter(function (s) { return s.length; }).map(Number);
        var n = sub.reduce(function (a, b) { return a * b; }, 1);
        if (/^>/.test(mm[2]) && dtypeSize(mm[2]) > 1) throw new Error("big-endian npy field " + mm[1] + ": " + url);
        var sz = dtypeSize(mm[2]);
        fields.push({ name: mm[1], descr: mm[2], offset: off, size: sz, count: n });
        off += sz * n;
      }
      out.fields = fields;
      out.itemsize = off;
      out.descr = "struct";
    } else {
      if (/^>/.test(ds[1]) && dtypeSize(ds[1]) > 1) throw new Error("big-endian npy " + ds[1] + ": " + url);
      out.descr = ds[1].replace(/^[<|=]/, "");
      out.itemsize = dtypeSize(ds[1]);
    }
    return out;
  }

  async function npyHeaderAt(url, ctx) {
    return cached("npyhdr:" + url, async function () {
      var b = await rangeRead(url, 0, 1024, ctx, { allowShort: true });
      var h = parseNpyHeader(b, url);
      if (h.need) {
        b = await rangeRead(url, 0, h.need, ctx);
        h = parseNpyHeader(b, url);
      }
      return h;
    });
  }

  async function readWholeNpy(url, ctx) {
    return cached("npy:" + url, async function () {
      var b = await rangeRead(url, 0, null, ctx);
      var h = parseNpyHeader(b, url);
      if (h.need) throw new Error("truncated .npy: " + url);
      var n = h.shape.reduce(function (a, c) { return a * c; }, 1);
      if (b.length !== h.dataOffset + n * h.itemsize) {
        throw new Error(".npy is " + b.length + " bytes, its header implies " + (h.dataOffset + n * h.itemsize) + ": " + url);
      }
      return { hdr: h, body: b.subarray(h.dataOffset), n: n };
    });
  }

  function i64At(u8, o) {   // little-endian int64 → Number (exact below 2^53)
    var lo = (u8[o] | (u8[o + 1] << 8) | (u8[o + 2] << 16)) + u8[o + 3] * 16777216;
    var hi = (u8[o + 4] | (u8[o + 5] << 8) | (u8[o + 6] << 16)) + (u8[o + 7] << 24);
    return hi * 4294967296 + lo;
  }
  function i32At(u8, o) { return u8[o] | (u8[o + 1] << 8) | (u8[o + 2] << 16) | (u8[o + 3] << 24); }
  function u32At(u8, o) { return i32At(u8, o) >>> 0; }

  // copy into an aligned buffer when a view would be misaligned
  function aligned(u8, bytesPer) {
    if (u8.byteOffset % bytesPer === 0) return u8;
    return u8.slice();
  }

  // ------------------------------------------------------------ registry --
  var GISTS = {
    oc4k: ["Ocean colour (ESA OC-CCI), 4 km daily",
      "Satellite ocean colour: chlorophyll as its base-10 logarithm, water clarity (Kd 490) and how many observations went into each pixel, every day at 4 km."],
    pace4k: ["Ocean colour and plankton (NASA PACE), 4 km daily",
      "NASA's PACE instrument: chlorophyll (base-10 logarithm), particulate and phytoplankton carbon, the apparent colour of the water, and three plankton cell counts, every day at 4 km."],
    sst_acspo02: ["Sea-surface temperature (NOAA ACSPO), 2 km daily",
      "Sea-surface temperature as satellites measured it at 0.02°, clear sky only — the gaps are clouds, not missing data — with the product's quality grade."],
    irtb: ["Cloud-top brightness temperature (geostationary IR), 4 km 3-hourly",
      "Infrared brightness temperature of whatever the geostationary satellites see — cloud tops or the surface — over 30° S to 30° N every three hours, in kelvin."],
    icoads: ["Ship and buoy reports (ICOADS)",
      "Marine weather reports from ships and buoys since 1662: sea and air temperature, pressure, wind, dew point, waves and cloud, one row per report."],
    wod: ["Ocean casts (World Ocean Database)",
      "Ship casts of temperature, salinity, oxygen and nutrients at 16 pressures, one row per cast, back to 1772."],
    glodap: ["Bottle samples: ocean carbon (GLODAP)",
      "Bottle samples of carbon, alkalinity, pH, oxygen and nutrients from research cruises, one row per bottle, with depth as the first channel."],
    bgcargo: ["Biogeochemical floats (BGC-Argo)",
      "Robotic floats measuring oxygen, nitrate, pH, chlorophyll, backscatter and light at 16 pressures, one row per profile."],
    oceansites: ["Open-ocean moorings (OceanSITES)",
      "Moored buoys in the open ocean: surface weather and temperature/salinity at ten depths, one row per site and hour."],
    xco2: ["Column CO₂ soundings (OCO-2, OCO-3, GOSAT)",
      "Satellite soundings of the average CO₂ in the column of air beneath, one row per good sounding, each a few km² across."],
    swh: ["Significant wave height along altimeter tracks (ESA CCI)",
      "Wave height measured by radar altimeters along their ground tracks, one row per second (about 7 km apart), 1991–2023."],
    swot: ["Sea-level anomaly on SWOT's 2 km swaths",
      "The SWOT satellite's wide-swath sea-surface height anomaly at 2 km, one row per valid pixel — a very large store, so keep the period short."]
  };

  // documented physical conversions, keyed on the STORED unit's own words
  function conversionOf(unit) {
    var u = String(unit || "");
    if (/^K\s*-\s*160\b/.test(u) || /add 160 for kelvin/i.test(u)) {
      return { unit: "K", offset: 160, note: "stored as kelvin − 160 in one byte; 160 added back, so the values are kelvin" };
    }
    if (/nm above 400/i.test(u) || /add 400 to get nanometres/i.test(u)) {
      return { unit: "nm", offset: 400, note: "stored as nanometres − 400; 400 added back, so the values are nanometres" };
    }
    if (/log10/i.test(u) && /LOGARITHM/i.test(u)) {
      return { unit: u, offset: 0, note: "kept as stored: the base-10 logarithm, not the concentration (10^value gives mg m-3)" };
    }
    return { unit: u, offset: 0, note: null };
  }

  function binStartIso(b) { return isoDate82(b * BIN_S); }
  function binEndIso(b) { return isoDate82(b * BIN_S + BIN_S - 1); }

  function folderUrlOf(rel) {
    var b = cfg.base;
    if (/\/resolve\/main\//.test(b)) return b.replace("/resolve/main/", "/tree/main/") + rel;
    return b + rel;
  }

  async function registryRaw(ctx) {
    return readJSON(cfg.base + "family1gf.json", ctx);
  }

  function relPath(reg, path) {
    var root = String(reg.hf_root || "tensors/family1_gf").replace(/\/+$/, "");
    var p = String(path || "");
    if (p.indexOf(root + "/") === 0) return p.slice(root.length + 1);
    throw new Error("store path " + p + " is not under the registry's hf_root " + root);
  }

  async function storeDesc(name, ctx) {
    var reg = await loadRegistry(ctx);
    var s = reg._byName[name];
    if (!s) throw new Error("no readable store named " + JSON.stringify(name) + " in the registry");
    return s;
  }

  function loadRegistry(ctx) {
    return cached("registry", async function () {
      var reg = await registryRaw(ctx);
      var groups = Array.isArray(reg.groups) ? reg.groups : Object.values(reg.groups || {});
      var stores = [], byName = {};
      var tgJobs = [];
      groups.forEach(function (g) {
        if (!g || !g.built || g.distribution !== "public") return;
        if (g.tier !== "G" && g.tier !== "P") return;
        var rel = relPath(reg, g.path);
        var gist = GISTS[g.name];
        var chans = (g.channels || []).map(function (c) {
          var cv = conversionOf(c.unit);
          var o = { name: c.name, unit: cv.unit, min: c.min, max: c.max, storedUnit: c.unit, note: cv.note };
          if (cv.offset) { o.min = c.min + cv.offset; o.max = c.max + cv.offset; }
          return o;
        });
        var d = {
          name: g.name, code: g.name,
          title: gist ? gist[0] : (g.title || g.name),
          gist: gist ? gist[1] : (g.title || g.name),
          kind: g.tier === "G" ? "grid" : "points",
          channels: chans, span: null,
          frameSeconds: g.tier === "G" ? g.frame_seconds : null,
          framesPerBin: g.tier === "G" ? g.frames_per_bin : null,
          grid: null, folderUrl: folderUrlOf(rel), cadence: g.cadence || null,
          subDaily: g.tier === "P" ? true : (g.frame_seconds < 86400),
          N: g.N == null ? null : g.N, binFirst: null, binLast: null,
          groups: null
        };
        hide(d, "_rel", rel);
        hide(d, "_raw", g);
        if (g.tier === "G") {
          var sg = g.store_groups || {};
          var names = Object.keys(sg).sort();
          if (!names.length) return;            // built but nothing to read
          d.groups = names.map(function (gn) { var o = { name: gn, grid: null }; hide(o, "_sg", sg[gn]); return o; });
          d.binFirst = Math.min.apply(null, names.map(function (gn) { return sg[gn].bin_first; }));
          d.binLast = Math.max.apply(null, names.map(function (gn) { return sg[gn].bin_last; }));
          d.groups.forEach(function (gg) {
            tgJobs.push(tileGrid(d, gg.name, ctx).then(function (tg) { gg.grid = gridOf(tg, gg.name); }));
          });
        } else {
          d.binFirst = g.bin_first;
          d.binLast = g.bin_last;
        }
        if (d.binFirst != null && d.binLast != null) d.span = [binStartIso(d.binFirst), binEndIso(d.binLast)];
        else if (g.record_span) d.span = g.record_span.slice();
        stores.push(d);
        byName[d.name] = d;
      });
      await Promise.all(tgJobs);
      stores.forEach(function (d) { if (d.groups) d.grid = d.groups[0].grid; });
      var out = { stores: stores, generated: reg.generated_utc || null, base: cfg.base };
      Object.defineProperty(out, "_byName", { value: byName, enumerable: false });
      Object.defineProperty(out, "_raw", { value: reg, enumerable: false });
      return out;
    });
  }

  function hide(o, k, v) { Object.defineProperty(o, k, { value: v, enumerable: false, writable: true }); }

  function gridOf(tg, group) {
    var g = tg.grid;
    return {
      group: group, H: tg.H, W: tg.W, C: tg.C, tile: tg.tile,
      nTilesX: tg.n_tiles_x, nTilesY: tg.n_tiles_y, dtype: tg.dtype,
      lat0: g.y0 + 0.5 * g.dy, lon0: g.x0 + 0.5 * g.dx, dlat: g.dy, dlon: g.dx,
      x0: g.x0, y0: g.y0, dx: g.dx, dy: g.dy, crs: g.crs || null,
      extent: g.extent || null
    };
  }

  function groupUrl(d, group) {
    var gg = (d.groups || []).find(function (x) { return x.name === group; });
    var prefix = gg && gg._sg && gg._sg.prefix ? gg._sg.prefix : group;
    return cfg.base + d._rel + "/" + prefix + "/";
  }

  function tileGrid(d, group, ctx) {
    var gg = (d.groups || []).find(function (x) { return x.name === group; });
    var rel = gg && gg._sg && gg._sg.tile_grid ? gg._sg.tile_grid : group + "/tile_grid.json";
    var url = cfg.base + d._rel + "/" + rel;
    return readJSON(url, ctx).then(function (tg) {
      if (tg.format !== "family1-sharded/1") {
        throw new Error("tile_grid.json format " + JSON.stringify(tg.format) + " is not family1-sharded/1: " + url);
      }
      if (!(tg.dtype === "float16" || tg.dtype === "uint8")) throw new Error("tile dtype " + tg.dtype + " unsupported: " + url);
      return tg;
    });
  }

  function shardIndex(d, group, ctx) {
    var gg = (d.groups || []).find(function (x) { return x.name === group; });
    var rel = gg && gg._sg && gg._sg.shard_index ? gg._sg.shard_index : group + "/shard_index.npy";
    var url = cfg.base + d._rel + "/" + rel;
    return cached("shardindex:" + url, async function () {
      var r = await readWholeNpy(url, ctx);
      var h = r.hdr;
      if (h.descr !== "struct") throw new Error("shard_index.npy is not a structured array: " + url);
      var F = {};
      h.fields.forEach(function (f) { F[f.name] = f; });
      ["bin", "frame_mask", "nbytes", "frames_present"].forEach(function (k) {
        if (!F[k]) throw new Error("shard_index.npy has no field " + k + ": " + url);
      });
      var n = r.n, body = r.body, isz = h.itemsize;
      var rows = new Map();
      for (var i = 0; i < n; i++) {
        var o = i * isz;
        var b = i32At(body, o + F.bin.offset);
        var mo = o + F.frame_mask.offset;
        rows.set(b, {
          bin: b,
          maskLo: u32At(body, mo), maskHi: u32At(body, mo + 4),
          nbytes: i64At(body, o + F.nbytes.offset),
          framesPresent: F.frames_present.size === 2
            ? (body[o + F.frames_present.offset] | (body[o + F.frames_present.offset + 1] << 8))
            : i32At(body, o + F.frames_present.offset)
        });
      }
      return { url: url, rows: rows };
    });
  }

  function frameBit(row, f) {
    return f < 32 ? ((row.maskLo >>> f) & 1) : ((row.maskHi >>> (f - 32)) & 1);
  }

  // ------------------------------------------------------------ selection --
  function normSel(sel, d) {
    if (!sel || typeof sel !== "object") throw new Error("a selection object is required");
    var s = Object.assign({}, sel);
    var names = d.channels.map(function (c) { return c.name; });
    s.channels = (sel.channels && sel.channels.length) ? sel.channels.slice() : names.slice();
    s.channels.forEach(function (c) {
      if (names.indexOf(c) < 0) throw new Error("store " + d.name + " has no channel " + JSON.stringify(c));
    });
    s.yearStart = Number(sel.yearStart);
    s.yearEnd = Number(sel.yearEnd);
    if (!Number.isInteger(s.yearStart) || !Number.isInteger(s.yearEnd)) throw new Error("yearStart and yearEnd must be whole years");
    if (s.yearEnd < s.yearStart) throw new Error("yearEnd is before yearStart");
    s.months = (sel.months && sel.months.length) ? sel.months.map(Number) : [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12];
    s.months.forEach(function (m) { if (!(m >= 1 && m <= 12 && Number.isInteger(m))) throw new Error("month " + m + " is not 1..12"); });
    s.hours = sel.hours ? [Number(sel.hours[0]), Number(sel.hours[1])] : null;
    // + days: an inclusive day-of-month range [d0, d1] inside every chosen
    // month (null = the whole month). It narrows the READ, unlike the box and
    // the hours of a point store: the dense swath stores (SWOT reads ~3 GB a
    // month) are only downloadable a few days at a time.
    s.days = sel.days ? [Number(sel.days[0]), Number(sel.days[1])] : null;
    if (s.days && !(Number.isInteger(s.days[0]) && Number.isInteger(s.days[1]) &&
        s.days[0] >= 1 && s.days[1] <= 31 && s.days[0] <= s.days[1])) {
      throw new Error("days must be two whole days of the month 1..31, the first not after the second");
    }
    if (s.days && s.days[0] === 1 && s.days[1] === 31) s.days = null;
    if (s.hours && !(s.hours[0] >= 0 && s.hours[0] <= 24 && s.hours[1] >= 0 && s.hours[1] <= 24)) {
      throw new Error("hours must be two numbers 0..24");
    }
    s.step = sel.step || "native";
    if (["native", "pentad", "month", "all"].indexOf(s.step) < 0) throw new Error("step must be native, pentad, month or all");
    s.res = sel.res == null ? "native" : sel.res;
    if (s.res === "0.25") s.res = 0.25;
    if (s.res === "1") s.res = 1;
    if (!(s.res === "native" || s.res === 0.25 || s.res === 1)) throw new Error("res must be 'native', 0.25 or 1");
    s.bbox = sel.bbox ? normBox(sel.bbox) : null;
    return s;
  }

  function wrap180(x) {
    var y = ((x + 180) % 360 + 360) % 360 - 180;
    return y;
  }

  function normBox(b) {
    var w = Number(b.w), s = Number(b.s), e = Number(b.e), n = Number(b.n);
    if (![w, s, e, n].every(isFinite)) throw new Error("bbox needs four numbers w, s, e, n");
    if (s > n) throw new Error("bbox south " + s + " is north of its north " + n);
    s = Math.max(-90, s); n = Math.min(90, n);
    if (e - w >= 360) { w = -180; e = 180; }
    else {
      var w2 = wrap180(w), e2 = wrap180(e);
      if (e2 === -180 && e !== -180) e2 = 180;     // keep an east edge ON the seam
      w = w2; e = e2;
    }
    return { w: w, s: s, e: e, n: n };
  }

  function chanIndex(d, names) {
    var all = d.channels.map(function (c) { return c.name; });
    return names.map(function (c) { return all.indexOf(c); });
  }

  // =============================================================== GRIDS ===
  function pickGroup(d, sel) {
    if (sel.group) {
      if (!d.groups.some(function (g) { return g.name === sel.group; })) {
        throw new Error("store " + d.name + " has no group " + sel.group);
      }
      return sel.group;
    }
    if (d.groups.length > 1) {
      throw new Error("store " + d.name + " has several groups (" + d.groups.map(function (g) { return g.name; }).join(", ") +
        "): name one as sel.group");
    }
    return d.groups[0].name;
  }

  // the pixel geometry of a box on a grid: which rows/cols, in output order
  function boxGeometry(tg, bbox, res) {
    var g = tg.grid, H = tg.H, W = tg.W, T = tg.tile;
    var rows = [];
    for (var r = 0; r < H; r++) {
      var lat = g.y0 + (r + 0.5) * g.dy;
      if (lat >= bbox.s && lat <= bbox.n) rows.push([lat, r]);
    }
    rows.sort(function (a, b) { return a[0] - b[0]; });
    var cols = [];
    var dl = bbox.w > bbox.e;
    for (var c = 0; c < W; c++) {
      var lon = wrap180(g.x0 + (c + 0.5) * g.dx);
      if (!dl) { if (lon >= bbox.w && lon <= bbox.e) cols.push([lon, c]); }
      else if (lon >= bbox.w) cols.push([lon, c]);
      else if (lon <= bbox.e) cols.push([lon + 360, c]);
    }
    cols.sort(function (a, b) { return a[0] - b[0]; });
    var outLat, outLon, rowOut = new Map(), colOut = new Map(), maxPerCell = 1;
    if (res === "native") {
      outLat = Float64Array.from(rows, function (x) { return x[0]; });
      outLon = Float64Array.from(cols, function (x) { return x[0]; });
      rows.forEach(function (x, i) { rowOut.set(x[1], i); });
      cols.forEach(function (x, i) { colOut.set(x[1], i); });
    } else {
      var rk = rows.map(function (x) { return Math.floor(x[0] / res); });
      var ck = cols.map(function (x) { return Math.floor(x[0] / res); });
      var ur = Array.from(new Set(rk)).sort(function (a, b) { return a - b; });
      var uc = Array.from(new Set(ck)).sort(function (a, b) { return a - b; });
      var ri = new Map(ur.map(function (k, i) { return [k, i]; }));
      var ci = new Map(uc.map(function (k, i) { return [k, i]; }));
      outLat = Float64Array.from(ur, function (k) { return (k + 0.5) * res; });
      outLon = Float64Array.from(uc, function (k) { return (k + 0.5) * res; });
      rows.forEach(function (x, i) { rowOut.set(x[1], ri.get(rk[i])); });
      cols.forEach(function (x, i) { colOut.set(x[1], ci.get(ck[i])); });
      var rc = new Map(), cc = new Map();
      rk.forEach(function (k) { rc.set(k, (rc.get(k) || 0) + 1); });
      ck.forEach(function (k) { cc.set(k, (cc.get(k) || 0) + 1); });
      maxPerCell = Math.max.apply(null, [1].concat(Array.from(rc.values()))) *
        Math.max.apply(null, [1].concat(Array.from(cc.values())));
    }
    // per tile row / tile column: (local index, output index) pairs
    var byTy = new Map(), byTx = new Map();
    rows.forEach(function (x) {
      var ty = Math.floor(x[1] / T);
      if (!byTy.has(ty)) byTy.set(ty, []);
      byTy.get(ty).push(x[1] - ty * T, rowOut.get(x[1]));
    });
    cols.forEach(function (x) {
      var tx = Math.floor(x[1] / T);
      if (!byTx.has(tx)) byTx.set(tx, []);
      byTx.get(tx).push(x[1] - tx * T, colOut.get(x[1]));
    });
    var tys = Array.from(byTy.keys()).sort(function (a, b) { return a - b; });
    var txs = Array.from(byTx.keys()).sort(function (a, b) { return a - b; });
    var txRuns = [];
    txs.forEach(function (tx) {
      var last = txRuns[txRuns.length - 1];
      if (last && last[1] === tx - 1) last[1] = tx; else txRuns.push([tx, tx]);
    });
    return {
      outLat: outLat, outLon: outLon, Ho: outLat.length, Wo: outLon.length,
      byTy: byTy, byTx: byTx, tys: tys, txs: txs, txRuns: txRuns,
      nRows: rows.length, nCols: cols.length, maxPerCell: maxPerCell
    };
  }

  function yearFolder(b) { return String(civil(Math.floor(b * BIN_S / 86400) + EPOCH_DAYS)[0]).padStart(4, "0"); }
  function binName(b) { return "bin_" + (b < 0 ? "-" + String(-b).padStart(4, "0") : String(b).padStart(4, "0")); }

  async function gridPlan(sel, ctx, opts) {
    opts = opts || {};
    var d = await storeDesc(sel.store, ctx);
    if (d.kind !== "grid") throw new Error("store " + d.name + " is not a grid");
    var s = normSel(sel, d);
    if (!s.bbox) {
      var e = new Error("a box (bbox) is required for the gridded store " + d.name);
      e.needBox = true;
      throw e;
    }
    var group = pickGroup(d, s);
    var tg = await tileGrid(d, group, ctx);
    var si = await shardIndex(d, group, ctx);
    var geo = boxGeometry(tg, s.bbox, s.res);
    var F = tg.frames_per_bin, fs = tg.frame_seconds;
    if (F * fs !== BIN_S) throw new Error(d.name + ": " + F + " frames × " + fs + " s is not one five-day bin");
    var allC = tg.channels.map(function (c) { return c.name; });
    var chIdx = s.channels.map(function (c) {
      var i = allC.indexOf(c);
      if (i < 0) throw new Error("tile_grid.json of " + d.name + "/" + group + " has no channel " + c);
      return i;
    });
    var conv = s.channels.map(function (c) { return conversionOf(tg.channels[allC.indexOf(c)].unit); });
    var monthOk = {};
    s.months.forEach(function (m) { monthOk[m] = true; });
    var t0 = sec82OfCivil(s.yearStart, 1, 1), t1 = sec82OfCivil(s.yearEnd + 1, 1, 1) - 1;
    var b0 = Math.floor(t0 / BIN_S), b1 = Math.floor(t1 / BIN_S);
    var bins = [], frames = [];
    var bk = Array.from(si.rows.keys()).filter(function (b) { return b >= b0 && b <= b1; }).sort(function (a, b) { return a - b; });
    bk.forEach(function (b) {
      var row = si.rows.get(b), fl = [];
      for (var f = 0; f < F; f++) {
        if (!frameBit(row, f)) continue;
        var t = b * BIN_S + f * fs;
        var q = ymh82(t);
        if (q.y < s.yearStart || q.y > s.yearEnd || !monthOk[q.m]) continue;
        if (s.days && (q.d < s.days[0] || q.d > s.days[1])) continue;
        if (fs < 86400 && !hourOk(q.h, s.hours)) continue;
        var fr = { b: b, f: f, t: t, y: q.y, m: q.m };
        fl.push(fr);
        frames.push(fr);
      }
      if (fl.length) bins.push({ b: b, frames: fl, row: row });
    });
    if (opts.firstBinOnly && bins.length) bins = bins.slice(0, 1);
    // steps
    var stepKey = function (fr) {
      if (s.step === "native") return fr.b * 64 + fr.f;
      if (s.step === "pentad") return fr.b;
      if (s.step === "month") return fr.y * 12 + fr.m - 1;
      return 0;
    };
    var keys = [], keyIdx = new Map(), perStep = new Map();
    frames.forEach(function (fr) {
      var k = stepKey(fr);
      if (!keyIdx.has(k)) { keyIdx.set(k, keys.length); keys.push(k); }
      perStep.set(k, (perStep.get(k) || 0) + 1);
      fr.si = keyIdx.get(k);
    });
    var times = new Float64Array(keys.length);
    keys.forEach(function (k, i) {
      if (s.step === "native") times[i] = EPOCH_UNIX + Math.floor(k / 64) * BIN_S + (k % 64) * fs;
      else if (s.step === "pentad") times[i] = EPOCH_UNIX + k * BIN_S;
      else if (s.step === "month") times[i] = (daysFromCivil(Math.floor(k / 12), k % 12 + 1, 1)) * 86400;
      else times[i] = EPOCH_UNIX + frames[0].t;
    });
    var mean = !(s.step === "native" && s.res === "native");
    var maxFrames = Math.max.apply(null, [1].concat(Array.from(perStep.values())));
    var maxCount = (s.step === "native" ? 1 : maxFrames) * geo.maxPerCell;
    var Cs = s.channels.length;
    var cells = keys.length * Cs * geo.Ho * geo.Wo;
    var countBytes = mean ? (maxCount > 65535 ? 4 : 2) : 0;
    var outBytes = cells * (4 + countBytes) + 8 * (keys.length + geo.Ho + geo.Wo);
    // which index entries each bin needs, as coalesced byte ranges
    var nty = tg.n_tiles_y, ntx = tg.n_tiles_x, ih = tg.index_header_bytes;
    var gurl = groupUrl(d, group);
    bins.forEach(function (bn) {
      var spans = [];
      bn.frames.forEach(function (fr) {
        geo.tys.forEach(function (ty) {
          geo.txRuns.forEach(function (run) {
            var a = (fr.f * nty + ty) * ntx + run[0], z = (fr.f * nty + ty) * ntx + run[1];
            spans.push([ih + 16 * a, ih + 16 * (z + 1)]);
          });
        });
      });
      bn.idxUrl = gurl + yearFolder(bn.b) + "/" + binName(bn.b) + ".idx.npy";
      bn.zstUrl = gurl + yearFolder(bn.b) + "/" + binName(bn.b) + ".zst";
      bn.idxRanges = coalesce(spans, GAP_IDX, MAX_RANGE);
    });
    return {
      d: d, s: s, group: group, tg: tg, geo: geo, bins: bins, frames: frames,
      times: times, T: keys.length, Cs: Cs, chIdx: chIdx, conv: conv, mean: mean,
      countBytes: countBytes, outBytes: outBytes, maxCount: maxCount
    };
  }

  function coalesce(spans, gap, maxLen) {
    spans.sort(function (a, b) { return a[0] - b[0]; });
    var out = [];
    spans.forEach(function (sp) {
      var last = out[out.length - 1];
      if (last && sp[0] - last[1] <= gap && Math.max(last[1], sp[1]) - last[0] <= maxLen) {
        if (sp[1] > last[1]) last[1] = sp[1];
        last.items.push(sp);
      } else {
        var n = [sp[0], sp[1]];
        n.items = [sp];
        out.push(n);
      }
    });
    return out;
  }

  function idxCacheGet(k) {
    var v = idxCache.get(k);
    if (v) { idxCache.delete(k); idxCache.set(k, v); }
    return v;
  }
  function idxCachePut(k, v) {
    idxCache.set(k, v);
    idxCacheBytes += v.length;
    while (idxCacheBytes > IDX_CACHE_BYTES && idxCache.size > 1) {
      var first = idxCache.keys().next().value;
      idxCacheBytes -= idxCache.get(first).length;
      idxCache.delete(first);
    }
  }

  // the (offset, length) entries one bin needs: Map flat → [off, len]
  async function readBinIndex(plan, bn, ctx) {
    var tg = plan.tg, nty = tg.n_tiles_y, ntx = tg.n_tiles_x, ih = tg.index_header_bytes;
    var ent = new Map();
    for (var i = 0; i < bn.idxRanges.length; i++) {
      var r = bn.idxRanges[i];
      var key = bn.idxUrl + "|" + r[0] + "|" + (r[1] - r[0]);
      var buf = idxCacheGet(key);
      if (!buf) {
        buf = await rangeRead(bn.idxUrl, r[0], r[1] - r[0], ctx);
        idxCachePut(key, buf);
      } else if (ctx.onCached) ctx.onCached();
      for (var o = 0; o + 16 <= buf.length; o += 16) {
        var flat = (r[0] + o - ih) / 16;
        ent.set(flat, [i64At(buf, o), i64At(buf, o + 8)]);
      }
    }
    // cross-check the index against the shard index's frame mask: a frame the
    // mask says is present must not read -1, and an absent one is never asked
    var tiles = [];
    bn.frames.forEach(function (fr) {
      plan.geo.tys.forEach(function (ty) {
        plan.geo.txs.forEach(function (tx) {
          var flat = (fr.f * nty + ty) * ntx + tx;
          var e = ent.get(flat);
          if (!e) throw new Error("index entry (frame " + fr.f + ", tile " + ty + "," + tx + ") not read: " + bn.idxUrl);
          if (e[0] === -1) {
            throw new Error("frame " + fr.f + " of bin " + bn.b + " reads as absent (offset −1) in " + bn.idxUrl +
              " but shard_index.npy's frame mask says it is present");
          }
          if (e[0] < 0 || e[1] < 0) throw new Error("bad index entry " + e + " in " + bn.idxUrl);
          if (e[1] > 0) tiles.push({ fr: fr, ty: ty, tx: tx, off: e[0], len: e[1] });
        });
      });
    });
    return tiles;
  }

  function tileRequests(bn, tiles) {
    var spans = tiles.map(function (t) { var s = [t.off, t.off + t.len]; s.tile = t; return s; });
    return coalesce(spans, GAP_TILE, MAX_RANGE).map(function (r) {
      return { url: bn.zstUrl, off: r[0], len: r[1] - r[0], tiles: r.items.map(function (x) { return x.tile; }) };
    });
  }

  async function gridEstimate(plan, ctx) {
    var idxReq = 0, idxBytes = 0;
    plan.bins.forEach(function (bn) {
      idxReq += bn.idxRanges.length;
      bn.idxRanges.forEach(function (r) { idxBytes += r[1] - r[0]; });
    });
    var exact = plan.bins.length <= EXACT_IDX_BINS;
    var tileReq = 0, tileBytes = 0, how;
    if (!plan.bins.length) {
      how = "nothing to read";
    } else if (exact) {
      var all = await pool(plan.bins, function (bn) { return readBinIndex(plan, bn, ctx); }, ctx);
      plan.bins.forEach(function (bn, i) {
        tileRequests(bn, all[i]).forEach(function (q) { tileReq++; tileBytes += q.len; });
      });
      how = "exact, from the tile index of every five-day file it reads";
    } else {
      // sample bins evenly, measure the box's share of each sampled shard, and
      // scale the shard index's byte counts by it
      var nS = SAMPLE_IDX_BINS, ratioB = 0, ratioN = 0, reqPerFrame = 0, framesS = 0;
      var sample = [];
      for (var k0 = 0; k0 < nS; k0++) sample.push(plan.bins[Math.floor((k0 + 0.5) * plan.bins.length / nS)]);
      var sampled = await pool(sample, function (bn) { return readBinIndex(plan, bn, ctx); }, ctx);
      for (var k = 0; k < nS; k++) {
        var bn = sample[k];
        var qs = tileRequests(bn, sampled[k]);
        var by = qs.reduce(function (a, q) { return a + q.len; }, 0);
        var share = bn.row.nbytes * bn.frames.length / Math.max(1, bn.row.framesPresent);
        if (share > 0) { ratioB += by; ratioN += share; }
        reqPerFrame += qs.length;
        framesS += bn.frames.length;
      }
      var ratio = ratioN > 0 ? ratioB / ratioN : 0;
      plan.bins.forEach(function (bn) {
        tileBytes += bn.row.nbytes * bn.frames.length / Math.max(1, bn.row.framesPresent) * ratio;
      });
      tileReq = Math.round(reqPerFrame / Math.max(1, framesS) * plan.frames.length);
      tileBytes = Math.round(tileBytes);
      how = "estimated: the tile index of " + nS + " sampled five-day files, scaled by each file's size";
    }
    return { requests: idxReq + tileReq, readBytes: idxBytes + tileBytes, exact: exact, how: how };
  }

  async function gridRun(plan, ctx, onProgress) {
    var tg = plan.tg, T = tg.tile, C = tg.C, geo = plan.geo, Cs = plan.Cs;
    var HW = geo.Ho * geo.Wo, nOut = plan.T * Cs * HW;
    var data = new Float32Array(nOut);
    var count = null;
    if (plan.mean) count = plan.countBytes === 4 ? new Uint32Array(nOut) : new Uint16Array(nOut);
    else data.fill(NaN);
    var u8 = tg.dtype === "uint8", isz = u8 ? 1 : 2, want = T * T * C * isz;
    var offs = plan.conv.map(function (c) { return c.offset || 0; });
    var zstd = getZstd();
    if (!zstd || !zstd.decompress) throw new Error("the zstd decoder (lib/fzstd.js, global fzstd) is not loaded");
    var progress = { done: 0, total: 0, bytes: 0 };
    var tell = function () { if (onProgress) { try { onProgress({ done: progress.done, total: progress.total, bytes: progress.bytes }); } catch (e) { /* the caller's */ } } };
    progress.total = plan.bins.reduce(function (a, bn) { return a + bn.idxRanges.length; }, 0);
    var rctx = Object.assign({}, ctx, {
      onRead: function (n) { progress.done++; progress.bytes += n; tell(); },
      onCached: function () { progress.done++; tell(); }
    });
    tell();
    // phase 1: the tile index of every bin
    var reqs = [];
    var perBin = await pool(plan.bins, function (bn) { return readBinIndex(plan, bn, rctx); }, ctx);
    plan.bins.forEach(function (bn, i) { reqs = reqs.concat(tileRequests(bn, perBin[i])); });
    var readBytes = reqs.reduce(function (a, q) { return a + q.len; }, 0);
    plan.bins.forEach(function (bn) { bn.idxRanges.forEach(function (r) { readBytes += r[1] - r[0]; }); });
    if (readBytes > CAP_READ) {
      throw new Error("over the cap: this selection reads " + fmtMB(readBytes) + " of index and tiles (limit " + fmtMB(CAP_READ) + ")");
    }
    progress.total += reqs.length;
    tell();
    // phase 2: the tiles, decoded straight into the output
    function put(tile, raw) {
      var fr = tile.fr, rowsT = geo.byTy.get(tile.ty), colsT = geo.byTx.get(tile.tx);
      for (var k = 0; k < Cs; k++) {
        var ch = plan.chIdx[k], add = offs[k];
        var base = (fr.si * Cs + k) * HW;
        for (var a = 0; a < rowsT.length; a += 2) {
          var lr = rowsT[a], ob = base + rowsT[a + 1] * geo.Wo, rb = lr * T;
          for (var c = 0; c < colsT.length; c += 2) {
            var p = (rb + colsT[c]) * C + ch, v;
            if (u8) { var q = raw[p]; v = q === U8_MISSING ? NaN : q + add; }
            else { v = F16[raw[2 * p] | (raw[2 * p + 1] << 8)] + add; }
            var oi = ob + colsT[c + 1];
            if (!plan.mean) data[oi] = v;
            else if (v === v) { data[oi] += v; count[oi]++; }
          }
        }
      }
    }
    await pool(reqs, async function (q) {
      var buf = await rangeRead(q.url, q.off, q.len, rctx);
      q.tiles.forEach(function (tile) {
        var blob = buf.subarray(tile.off - q.off, tile.off - q.off + tile.len), raw;
        try { raw = zstd.decompress(blob); }
        catch (e) { throw new Error("zstd: " + e.message + " — tile at bytes " + tile.off + "+" + tile.len + " of " + q.url); }
        if (raw.length !== want) {
          throw new Error("a tile decompressed to " + raw.length + " bytes, the declared tile is " + want + " — " + q.url + " @" + tile.off);
        }
        put(tile, raw);
      });
    }, ctx);
    if (plan.mean) {
      for (var i = 0; i < nOut; i++) data[i] = count[i] ? data[i] / count[i] : NaN;
    }
    return { data: data, count: count };
  }

  function gridNotes(plan) {
    var n = [];
    plan.conv.forEach(function (c, k) { if (c.note) n.push(plan.s.channels[k] + ": " + c.note); });
    if (plan.s.bbox && plan.s.bbox.w > plan.s.bbox.e) n.push("the box crosses the dateline: longitudes run past 180° (subtract 360 for −180..180)");
    if (plan.mean) n.push("each value is the mean of the finite observations in its cell and time step; the count arrays say how many");
    return n;
  }

  // ============================================================== POINTS ===
  function pointStoreJson(d, ctx) {
    return readJSON(cfg.base + d._rel + "/store.json", ctx);
  }

  async function pointColumns(d, ctx) {
    return cached("pcols:" + d.name + ":" + cfg.base, async function () {
      var meta = await pointStoreJson(d, ctx);
      var split = meta.hub_split || {};
      var base = cfg.base + d._rel + "/";
      var names = ["time_s", "lat", "lon", "values", "platform", "qc"];
      var cols = {};
      await Promise.all(names.map(async function (nm) {
        var file = nm + ".npy", parts;
        if (split[file]) {
          var cum = 0;
          parts = split[file].parts.map(function (p) {
            var o = { url: base + p.name, start: cum, bytes: p.bytes };
            cum += p.bytes;
            return o;
          });
          if (cum !== split[file].bytes) throw new Error("hub_split parts of " + file + " sum to " + cum + ", store.json says " + split[file].bytes);
        } else {
          parts = [{ url: base + file, start: 0, bytes: Infinity }];
        }
        var h = await npyHeaderAt(parts[0].url, ctx);
        if (h.descr === "struct") throw new Error(file + " is a structured array: " + parts[0].url);
        var rowElems = h.shape.slice(1).reduce(function (a, b) { return a * b; }, 1);
        cols[nm] = { name: nm, parts: parts, hdr: h.dataOffset, descr: h.descr, itemsize: h.itemsize, shape: h.shape, rowBytes: h.itemsize * rowElems, width: rowElems };
      }));
      var N = cols.lat.shape[0];
      names.forEach(function (nm) {
        if (cols[nm].shape[0] !== N) throw new Error(nm + ".npy has " + cols[nm].shape[0] + " rows, lat.npy " + N + " (" + d.name + ")");
      });
      var want = { time_s: ["i4", "i8"], lat: ["f4"], lon: ["f4"], values: ["f2"], platform: ["i8"], qc: ["u1"] };
      names.forEach(function (nm) {
        if (want[nm].indexOf(cols[nm].descr) < 0) throw new Error(nm + ".npy is " + cols[nm].descr + ", expected " + want[nm].join(" or ") + " (" + d.name + ")");
      });
      var sch = Number(meta.schema_version || 2);
      if ((sch === 3) !== (cols.time_s.descr === "i8")) {
        throw new Error("store.json says schema " + sch + " but time_s.npy is " + cols.time_s.descr + " (" + d.name + ")");
      }
      var offs = await readWholeNpy(base + "bin_offsets.npy", ctx);
      if (offs.hdr.descr !== "i8") throw new Error("bin_offsets.npy is " + offs.hdr.descr + " (" + d.name + ")");
      var off = new Float64Array(offs.n);
      for (var i = 0; i < offs.n; i++) off[i] = i64At(offs.body, 8 * i);
      if (off[0] !== 0 || off[offs.n - 1] !== N) throw new Error("bin_offsets.npy runs " + off[0] + ".." + off[offs.n - 1] + " over " + N + " rows (" + d.name + ")");
      return { meta: meta, cols: cols, N: N, C: cols.values.width, off: off, binFirst: Number(meta.bin_first), nBins: offs.n - 1 };
    });
  }

  // read rows [r0, r1) of one column, across hub_split parts if need be
  async function readColumnRows(col, r0, r1, ctx) {
    if (r1 <= r0) return new Uint8Array(0);
    var a = col.hdr + r0 * col.rowBytes, z = col.hdr + r1 * col.rowBytes;
    var pieces = [];
    for (var i = 0; i < col.parts.length; i++) {
      var p = col.parts[i], pa = p.start, pz = p.start + p.bytes;
      var lo = Math.max(a, pa), hi = Math.min(z, pz);
      if (hi > lo) pieces.push({ p: p, lo: lo, hi: hi });
    }
    var got = 0;
    pieces.forEach(function (x) { got += x.hi - x.lo; });
    if (got !== z - a) throw new Error("rows " + r0 + ".." + r1 + " of " + col.name + ".npy fall outside its parts");
    if (pieces.length === 1) {
      var x = pieces[0];
      return rangeRead(x.p.url, x.lo - x.p.start, x.hi - x.lo, ctx);
    }
    var bufs = [];
    for (var k = 0; k < pieces.length; k++) {
      var y = pieces[k];
      bufs.push(await rangeRead(y.p.url, y.lo - y.p.start, y.hi - y.lo, ctx));
    }
    var out = new Uint8Array(z - a), o = 0;
    bufs.forEach(function (b) { out.set(b, o); o += b.length; });
    return out;
  }

  // the first row in [r0, r1) whose time_s >= target. Rows are sorted by
  // (bin, time_s) and bin = floor(time_s / 432000), so time_s is sorted
  // through the whole store; an interpolation search on small windows finds
  // the row in a few reads.
  async function firstRowAtOrAfter(col, r0, r1, target, ctx) {
    var W = 4096, i64 = col.descr === "i8", sz = i64 ? 8 : 4;
    var tAt = function (u8, i) { return i64 ? i64At(u8, i * sz) : i32At(u8, i * sz); };
    var lo = r0, hi = r1;
    var ends = await Promise.all([readColumnRows(col, lo, lo + 1, ctx), readColumnRows(col, hi - 1, hi, ctx)]);
    var tlo = tAt(ends[0], 0), thi = tAt(ends[1], 0);
    if (tlo >= target) return lo;
    if (thi < target) return hi;
    for (var it = 0; it < 40; it++) {
      if (hi - lo <= W) {
        var all = await readColumnRows(col, lo, hi, ctx);
        for (var i = 0; i < hi - lo; i++) if (tAt(all, i) >= target) return lo + i;
        return hi;
      }
      var g = lo + Math.floor((target - tlo) / Math.max(1, thi - tlo) * (hi - lo)) - (W >> 1);
      if (it % 3 === 2) g = lo + ((hi - lo) >> 1) - (W >> 1);       // guard against a skewed column
      g = Math.max(lo, Math.min(hi - W, g));
      var win = await readColumnRows(col, g, g + W, ctx);
      var a = tAt(win, 0), z = tAt(win, W - 1);
      if (a >= target) { hi = g + 1; thi = a; if (g === lo) return lo; }
      else if (z < target) { lo = g + W; tlo = z; }
      else {
        for (var j = 0; j < W; j++) if (tAt(win, j) >= target) return g + j;
      }
    }
    throw new Error("firstRowAtOrAfter did not converge on " + col.name + ".npy");
  }

  // the period as time windows [a, z] (seconds since 1982, inclusive), one
  // per chosen month of every year, narrowed to the chosen days
  function periodWindows(s) {
    var wins = [];
    for (var y = s.yearStart; y <= s.yearEnd; y++) {
      s.months.forEach(function (m) {
        var mEnd = m === 12 ? sec82OfCivil(y + 1, 1, 1) : sec82OfCivil(y, m + 1, 1);
        var a = sec82OfCivil(y, m, s.days ? s.days[0] : 1);
        // daysFromCivil is linear in the day, so day d1 + 1 of a month is the
        // next month's first day when d1 is the last; clamp to the month end
        var z = Math.min(mEnd, s.days ? sec82OfCivil(y, m, s.days[1] + 1) : mEnd) - 1;
        if (z >= a) wins.push([a, z]);             // (days 30–31 of February: none)
      });
    }
    return wins;
  }

  function pointBinRuns(pc, s) {
    var want = new Set();
    periodWindows(s).forEach(function (w) {
      for (var b = Math.floor(w[0] / BIN_S); b <= Math.floor(w[1] / BIN_S); b++) want.add(b);
    });
    var bf = pc.binFirst, bl = pc.binFirst + pc.nBins - 1;
    var bins = Array.from(want).filter(function (b) { return b >= bf && b <= bl; }).sort(function (a, b) { return a - b; });
    var runs = [];
    bins.forEach(function (b) {
      var last = runs[runs.length - 1];
      if (last && last.b1 === b - 1) last.b1 = b; else runs.push({ b0: b, b1: b });
    });
    var rows = 0;
    runs.forEach(function (r) {
      r.r0 = pc.off[r.b0 - bf];
      r.r1 = pc.off[r.b1 - bf + 1];
      rows += r.r1 - r.r0;
    });
    return { runs: runs.filter(function (r) { return r.r1 > r.r0; }), rows: rows, bins: bins.length };
  }

  async function trimRunsToWindows(pc, wins, br, ctx) {
    var bf = pc.binFirst, bl = pc.binFirst + pc.nBins - 1, col = pc.cols.time_s;
    var cut = await pool(wins, async function (w) {
      var b0 = Math.max(bf, Math.floor(w[0] / BIN_S)), b1 = Math.min(bl, Math.floor(w[1] / BIN_S));
      if (b1 < b0) return null;
      var r0 = pc.off[b0 - bf], r1 = pc.off[b1 - bf + 1];
      if (r1 <= r0) return null;
      var a = await firstRowAtOrAfter(col, r0, r1, w[0], ctx);
      var z = a >= r1 ? a : await firstRowAtOrAfter(col, a, r1, w[1] + 1, ctx);
      return z > a ? { b0: b0, b1: b1, r0: a, r1: z } : null;
    }, ctx);
    var runs = [];
    cut.filter(Boolean).sort(function (x, y) { return x.r0 - y.r0; }).forEach(function (r) {
      var last = runs[runs.length - 1];
      if (last && r.r0 <= last.r1) { last.r1 = Math.max(last.r1, r.r1); last.b1 = Math.max(last.b1, r.b1); }
      else runs.push(r);
    });
    var rows = runs.reduce(function (t, r) { return t + (r.r1 - r.r0); }, 0);
    return { runs: runs, rows: rows, bins: br.bins, trimmed: true };
  }

  async function pointPlan(sel, ctx, popts) {
    var d = await storeDesc(sel.store, ctx);
    if (d.kind !== "points") throw new Error("store " + d.name + " is not a point store");
    var s = normSel(sel, d);
    var pc = await pointColumns(d, ctx);
    if (pc.C !== d.channels.length) throw new Error("values.npy is " + pc.C + " wide, the registry lists " + d.channels.length + " channels (" + d.name + ")");
    var br = pointBinRuns(pc, s);
    // Days narrower than a five-day bin: the bins alone would read up to five
    // times the rows wanted (one SWOT day is ~4 million rows; its bin ~19
    // million), so cut each window to its exact rows by searching the sorted
    // time column — a few 16 kB reads per window, bounded so a long period
    // falls back to whole bins rather than to thousands of searches. The row
    // filter in pointRun still applies the days, so this only reads less.
    if (s.days && !(popts && popts.exactDays === false)) {
      var wins = periodWindows(s);
      if (wins.length <= DAY_TRIM_WINDOWS && br.rows > 0) br = await trimRunsToWindows(pc, wins, br, ctx);
    }
    var binned = s.res !== "native" || s.step !== "native";
    var res = s.res === "native" ? 0.25 : s.res;
    var step = s.step === "native" ? "pentad" : s.step;
    var rowBytesCore = pc.cols.time_s.rowBytes + 8;
    var rowBytesRest = pc.cols.values.rowBytes + 9;
    var chunkRows = Math.max(4096, Math.min(1 << 20, Math.floor(CHUNK_BYTES / (rowBytesCore + rowBytesRest))));
    var chunks = [];
    br.runs.forEach(function (r) {
      for (var a = r.r0; a < r.r1; a += chunkRows) chunks.push([a, Math.min(r.r1, a + chunkRows)]);
    });
    var conv = s.channels.map(function (c) { return conversionOf(d.channels.find(function (x) { return x.name === c; }).storedUnit); });
    var plan = {
      d: d, s: s, pc: pc, br: br, chunks: chunks, binned: binned, res: res, step: step,
      chIdx: chanIndex(d, s.channels), Cs: s.channels.length, conv: conv,
      rowBytesCore: rowBytesCore, rowBytesRest: rowBytesRest
    };
    if (binned) {
      // the cells covering the box: [floor(s/res), floor(n/res)], less the
      // last one when the edge falls exactly on a cell boundary (a point ON
      // that edge is clamped into the last cell)
      var bx = s.bbox || { w: -180, s: -90, e: 180, n: 90 };
      var eU = bx.w > bx.e ? bx.e + 360 : bx.e;
      var edge = function (lo, hi) {
        var a = Math.floor(lo / res), z = Math.floor(hi / res);
        if (z > a && z * res === hi) z--;
        return [a, z];
      };
      var rr = edge(bx.s, bx.n), cr = edge(bx.w, eU);
      plan.rk0 = Math.max(-Math.round(90 / res), rr[0]);
      plan.rk1 = Math.min(Math.round(90 / res) - 1, rr[1]);
      plan.ck0 = cr[0];
      plan.ck1 = cr[1];
      plan.Ho = plan.rk1 - plan.rk0 + 1;
      plan.Wo = plan.ck1 - plan.ck0 + 1;
      plan.eU = eU;
      var nSteps;
      if (step === "pentad") nSteps = br.bins;
      else if (step === "month") nSteps = (s.yearEnd - s.yearStart + 1) * s.months.length;
      else nSteps = 1;
      plan.maxSteps = nSteps;
    }
    return plan;
  }

  function pointEstimateOf(plan) {
    var rows = plan.br.rows;
    var readMax = rows * (plan.rowBytesCore + plan.rowBytesRest);
    var reqMax = plan.chunks.length * 6;
    var outMax;
    if (plan.binned) outMax = plan.maxSteps * plan.Cs * plan.Ho * plan.Wo * 6 + 8 * (plan.maxSteps + plan.Ho + plan.Wo);
    else outMax = rows * (8 + 4 + 4 + 4 * plan.Cs + 8 + 1);
    return { rows: rows, readBytes: readMax, requests: reqMax, outBytes: outMax };
  }

  async function pointRun(plan, ctx, onProgress, opts) {
    opts = opts || {};
    var s = plan.s, pc = plan.pc, cols = pc.cols, Cs = plan.Cs, C = pc.C;
    var monthOk = [];
    s.months.forEach(function (m) { monthOk[m] = true; });
    var bx = s.bbox, dl = bx && bx.w > bx.e;
    var i64time = cols.time_s.descr === "i8";
    var offs = plan.conv.map(function (c) { return c.offset || 0; });
    var progress = { done: 0, total: plan.chunks.length * 3, bytes: 0 };
    var tell = function () { if (onProgress) { try { onProgress({ done: progress.done, total: progress.total, bytes: progress.bytes }); } catch (e) { /* the caller's */ } } };
    var rctx = Object.assign({}, ctx, { onRead: function (n) { progress.done++; progress.bytes += n; tell(); } });
    tell();
    var slabs = new Map();       // binned: step key → {sum, cnt}
    var res = plan.res, minT = Infinity;
    var chunkOut = new Array(plan.chunks.length);
    var stopAt = null;           // preview: stop after the first bin with a kept row

    async function doChunk(ci) {
      var ch = plan.chunks[ci], r0 = ch[0], r1 = ch[1], n = r1 - r0;
      if (stopAt != null && r0 >= stopAt.r1) return;
      var parts = await Promise.all([
        readColumnRows(cols.time_s, r0, r1, rctx),
        readColumnRows(cols.lat, r0, r1, rctx),
        readColumnRows(cols.lon, r0, r1, rctx)]);
      var tb = parts[0];
      var latB = aligned(parts[1], 4);
      var lat = new Float32Array(latB.buffer, latB.byteOffset, n);
      var lonB = aligned(parts[2], 4);
      var lon = new Float32Array(lonB.buffer, lonB.byteOffset, n);
      var keep = new Uint8Array(n), first = -1, last = -1, t82 = new Float64Array(n);
      var curDay = NaN, cy = 0, cm = 0, cd = 0;
      for (var i = 0; i < n; i++) {
        if (stopAt != null && r0 + i >= stopAt.r1) break;
        var t = i64time ? i64At(tb, 8 * i) : i32At(tb, 4 * i);
        t82[i] = t;
        var day = Math.floor(t / 86400);
        if (day !== curDay) { var c = civil(day + EPOCH_DAYS); cy = c[0]; cm = c[1]; cd = c[2]; curDay = day; }
        if (cy < s.yearStart || cy > s.yearEnd || !monthOk[cm]) continue;
        if (s.days && (cd < s.days[0] || cd > s.days[1])) continue;
        if (s.hours && !hourOk(Math.floor((t - day * 86400) / 3600), s.hours)) continue;
        if (bx) {
          var la = lat[i], lo = lon[i];
          if (!(la >= bx.s && la <= bx.n)) continue;
          if (!dl ? !(lo >= bx.w && lo <= bx.e) : !(lo >= bx.w || lo <= bx.e)) continue;
        }
        keep[i] = 1;
        if (first < 0) first = i;
        last = i;
      }
      if (opts.preview && first >= 0) {
        // keep only the first bin that has a kept row
        var bFirst = Math.floor(t82[first] / BIN_S);
        for (var j = first; j < n; j++) if (Math.floor(t82[j] / BIN_S) !== bFirst) { keep.fill(0, j); break; }
        last = first;
        for (var j2 = n - 1; j2 >= first; j2--) if (keep[j2]) { last = j2; break; }
        var bEndRow = pc.off[bFirst - pc.binFirst + 1];
        if (stopAt == null || bEndRow < stopAt.r1) stopAt = { r1: bEndRow };
      }
      progress.total += first >= 0 ? 3 : 0;
      if (first < 0) { chunkOut[ci] = null; return; }
      var a = r0 + first, z = r0 + last + 1, m = z - a;
      var more = await Promise.all([
        readColumnRows(cols.values, a, z, rctx),
        plan.binned ? null : readColumnRows(cols.platform, a, z, rctx),
        plan.binned ? null : readColumnRows(cols.qc, a, z, rctx)]);
      if (plan.binned) progress.done += 2;
      var vb = more[0];
      var nk = 0;
      for (var q = first; q <= last; q++) nk += keep[q];
      if (plan.binned) {
        for (var q2 = first; q2 <= last; q2++) {
          if (!keep[q2]) continue;
          var tq = t82[q2];
          if (tq < minT) minT = tq;
          var key;
          if (plan.step === "pentad") key = Math.floor(tq / BIN_S);
          else if (plan.step === "month") { var cc = civil(Math.floor(tq / 86400) + EPOCH_DAYS); key = cc[0] * 12 + cc[1] - 1; }
          else key = 0;
          var sl = slabs.get(key);
          if (!sl) {
            sl = { sum: new Float32Array(Cs * plan.Ho * plan.Wo), cnt: new Uint32Array(Cs * plan.Ho * plan.Wo) };
            slabs.set(key, sl);
          }
          var la2 = lat[q2], lo2 = lon[q2];
          if (dl && lo2 < bx.w) lo2 += 360;
          var rk = Math.min(plan.rk1, Math.max(plan.rk0, Math.floor(la2 / res))) - plan.rk0;
          var ck = Math.min(plan.ck1, Math.max(plan.ck0, Math.floor(lo2 / res))) - plan.ck0;
          var rowBase = (q2 - first) * C * 2;
          for (var k = 0; k < Cs; k++) {
            var p = rowBase + 2 * plan.chIdx[k];
            var v = F16[vb[p] | (vb[p + 1] << 8)];
            if (v === v) {
              var oi = (k * plan.Ho + rk) * plan.Wo + ck;
              sl.sum[oi] += v + offs[k];
              sl.cnt[oi]++;
            }
          }
        }
        chunkOut[ci] = null;
        return;
      }
      var pb = aligned(more[1], 8), qb = more[2];
      var pl = new BigInt64Array(pb.buffer, pb.byteOffset, m);
      var out = { time: new Float64Array(nk), lat: new Float32Array(nk), lon: new Float32Array(nk), values: new Float32Array(nk * Cs), platform: new BigInt64Array(nk), qc: new Uint8Array(nk) };
      var o = 0;
      for (var q3 = first; q3 <= last; q3++) {
        if (!keep[q3]) continue;
        var rr = q3 - first;
        out.time[o] = EPOCH_UNIX + t82[q3];
        out.lat[o] = lat[q3];
        out.lon[o] = lon[q3];
        for (var k2 = 0; k2 < Cs; k2++) {
          var p2_ = rr * C * 2 + 2 * plan.chIdx[k2];
          out.values[o * Cs + k2] = F16[vb[p2_] | (vb[p2_ + 1] << 8)] + offs[k2];
        }
        out.platform[o] = pl[rr];
        out.qc[o] = qb[rr];
        o++;
      }
      chunkOut[ci] = out;
    }

    var truncated = null;
    if (opts.preview && plan.br.runs.length) {
      // start the preview at the period's first second, not at the start of
      // the bin that holds it (one swot day is ~8 M rows)
      var run0 = plan.br.runs[0], startT = Infinity;
      for (var yy = s.yearStart; yy <= s.yearEnd && startT === Infinity; yy++) {
        for (var mm = 1; mm <= 12; mm++) if (monthOk[mm]) { startT = sec82OfCivil(yy, mm, s.days ? s.days[0] : 1); break; }
      }
      var rs = await firstRowAtOrAfter(cols.time_s, run0.r0, run0.r1, startT, rctx);
      var pcr = Math.min(PREVIEW_CHUNK_ROWS, plan.chunks.length ? plan.chunks[0][1] - plan.chunks[0][0] : PREVIEW_CHUNK_ROWS);
      plan.chunks = [];
      plan.br.runs.forEach(function (r, k) {
        for (var a = k === 0 ? rs : r.r0; a < r.r1; a += pcr) plan.chunks.push([a, Math.min(r.r1, a + pcr)]);
      });
      chunkOut = new Array(plan.chunks.length);
    }
    var idxs = plan.chunks.map(function (_c, i) { return i; });
    if (opts.preview) {
      // one bin — the first holding a kept row — but never an unbounded read:
      // give up after PREVIEW_SCAN_BYTES of times and positions without a hit,
      // and stop once PREVIEW_MAX_ROWS rows are held (one swot bin is ~40 M)
      var scanned = 0;
      for (var ii = 0; ii < idxs.length; ii++) {
        await doChunk(ii);
        scanned += (plan.chunks[ii][1] - plan.chunks[ii][0]) * plan.rowBytesCore;
        if (stopAt != null) {
          if (ii + 1 >= idxs.length || plan.chunks[ii + 1][0] >= stopAt.r1) break;
          var held = 0;
          chunkOut.forEach(function (x) { if (x) held += x.time.length; });
          if (held >= PREVIEW_MAX_ROWS || (plan.binned && scanned > PREVIEW_SCAN_BYTES)) {
            truncated = "the preview holds the first " + held.toLocaleString("en-US") + " rows of its five-day bin, not all of it";
            break;
          }
        } else if (scanned > PREVIEW_SCAN_BYTES) {
          truncated = "no row matched in the first " + fmtMB(scanned) + " of the period's times and positions; the preview stopped looking";
          break;
        }
      }
    } else {
      await pool(idxs, doChunk, ctx);
    }
    tell();
    if (plan.binned) {
      var keys = Array.from(slabs.keys()).sort(function (a, b) { return a - b; });
      var HW = plan.Ho * plan.Wo, Tn = keys.length, nOut = Tn * Cs * HW;
      var maxC = 0;
      slabs.forEach(function (sl) { for (var i = 0; i < sl.cnt.length; i++) if (sl.cnt[i] > maxC) maxC = sl.cnt[i]; });
      var data = new Float32Array(nOut);
      var count = maxC > 65535 ? new Uint32Array(nOut) : new Uint16Array(nOut);
      var times = new Float64Array(Tn);
      keys.forEach(function (key, ti) {
        var sl = slabs.get(key);
        if (plan.step === "pentad") times[ti] = EPOCH_UNIX + key * BIN_S;
        else if (plan.step === "month") times[ti] = daysFromCivil(Math.floor(key / 12), key % 12 + 1, 1) * 86400;
        else times[ti] = EPOCH_UNIX + Math.floor(minT / BIN_S) * BIN_S;
        var base = ti * Cs * HW;
        for (var i = 0; i < Cs * HW; i++) {
          var cn = sl.cnt[i];
          data[base + i] = cn ? sl.sum[i] / cn : NaN;
          count[base + i] = cn;
        }
        slabs.delete(key);
      });
      var latA = new Float64Array(plan.Ho), lonA = new Float64Array(plan.Wo);
      for (var r = 0; r < plan.Ho; r++) latA[r] = (plan.rk0 + r + 0.5) * res;
      for (var c2 = 0; c2 < plan.Wo; c2++) lonA[c2] = (plan.ck0 + c2 + 0.5) * res;
      return { grid: true, data: data, count: count, time: times, lat: latA, lon: lonA, truncated: truncated };
    }
    var tot = 0;
    chunkOut.forEach(function (x) { if (x) tot += x.time.length; });
    var R = { time: new Float64Array(tot), lat: new Float32Array(tot), lon: new Float32Array(tot), values: new Float32Array(tot * Cs), platform: new BigInt64Array(tot), qc: new Uint8Array(tot) };
    var at = 0;
    chunkOut.forEach(function (x) {
      if (!x) return;
      R.time.set(x.time, at); R.lat.set(x.lat, at); R.lon.set(x.lon, at);
      R.values.set(x.values, at * Cs); R.platform.set(x.platform, at); R.qc.set(x.qc, at);
      at += x.time.length;
    });
    R.truncated = truncated;
    return R;
  }

  // ================================================================ pool ===
  // run fn over items, at most cfg.concurrency at a time; the first failure
  // stops the rest (and aborts in-flight reads through ctx.ac)
  async function pool(items, fn, ctx) {
    var out = new Array(items.length), i = 0, err = null;
    async function loop() {
      while (!err && i < items.length) {
        var k = i++;
        try { out[k] = await fn(items[k], k); }
        catch (e) {
          if (!err) { err = e; if (ctx.ac) ctx.ac.abort(); }
        }
      }
    }
    var n = Math.max(1, Math.min(cfg.concurrency, items.length));
    var ws = [];
    for (var w = 0; w < n; w++) ws.push(loop());
    await Promise.all(ws);
    if (err) {
      if (ctx.userSignal && ctx.userSignal.aborted) throw abortError(ctx.userSignal);
      throw err;
    }
    return out;
  }

  function makeCtx(signal) {
    var ac = new AbortController();
    if (signal) {
      if (signal.aborted) ac.abort();
      else signal.addEventListener("abort", function () { ac.abort(); }, { once: true });
    }
    return { signal: ac.signal, ac: ac, userSignal: signal || null, stats: { requests: 0, bytes: 0 } };
  }

  function wrapAbort(ctx, e) {
    if (ctx.userSignal && ctx.userSignal.aborted) return abortError(ctx.userSignal);
    return e;
  }

  // ============================================================= public ===
  async function estimate(sel, opts) {
    var ctx = makeCtx(opts && opts.signal);
    var d = await storeDesc(sel && sel.store, ctx);
    var cap = function (o) {
      var over = [];
      if (o.readBytes > CAP_READ) over.push("would read " + fmtMB(o.readBytes) + " (the limit is " + fmtMB(CAP_READ) + ")");
      if (o.outBytes > CAP_OUT) over.push("would build " + fmtMB(o.outBytes) + " of arrays (the limit is " + fmtMB(CAP_OUT) + ")");
      o.overCap = over.length > 0;
      if (o.overCap) {
        o.why = "Too large: this selection " + over.join(" and ") + ". " + o.shrink +
          " Or download the store's own files: " + d.folderUrl;
      }
      delete o.shrink;
      return o;
    };
    if (d.kind === "grid") {
      var plan;
      try { plan = await gridPlan(sel, ctx); }
      catch (e) {
        if (e.needBox) {
          return { requests: 0, readBytes: 0, outBytes: 0, frames: 0, overCap: true, exact: true, shape: null,
            why: "Draw or type a box first: a gridded store is read tile by tile, so a box is required." };
        }
        throw e;
      }
      var g = await gridEstimate(plan, ctx);
      var o = {
        requests: g.requests, readBytes: g.readBytes, outBytes: plan.outBytes, frames: plan.frames.length,
        exact: g.exact, shape: [plan.T, plan.Cs, plan.geo.Ho, plan.geo.Wo],
        why: plan.frames.length + " frame" + (plan.frames.length === 1 ? "" : "s") + " from " + plan.bins.length +
          " five-day file" + (plan.bins.length === 1 ? "" : "s") + ": " + g.requests + " requests, " + fmtMB(g.readBytes) +
          " to read (" + g.how + "); the result is " + plan.T + " × " + plan.Cs + " × " + plan.geo.Ho + " × " + plan.geo.Wo +
          " (" + fmtMB(plan.outBytes) + ")." + (plan.mean ? " Coarser steps or cells shrink the file, not the read." : ""),
        shrink: "Shorten the period, pick fewer months or fewer days, or shrink the box."
      };
      return cap(o);
    }
    // + opts.exactDays === false: count days by whole five-day bins (an
    // upper bound, no row search) — for a caller sizing many candidates
    var pp = await pointPlan(sel, ctx, opts);
    var pe = pointEstimateOf(pp);
    var po = {
      requests: pe.requests, readBytes: pe.readBytes, outBytes: pe.outBytes, rows: pe.rows, exact: false,
      shape: pp.binned ? [pp.maxSteps, pp.Cs, pp.Ho, pp.Wo] : [pe.rows, pp.Cs],
      why: pe.rows.toLocaleString("en-US") + " rows in the period's " + pp.br.bins + " five-day bins (exact, from the store's offsets): at most " +
        pe.requests + " requests and " + fmtMB(pe.readBytes) + " to read — the box and the hours are applied after reading each row's " +
        "time and position, so fewer rows may be kept." + (pp.binned ? " Binned to " + pp.res + "° cells per " +
        (pp.step === "pentad" ? "five-day bin" : pp.step) + ": at most " + fmtMB(pe.outBytes) + " of arrays." : ""),
      shrink: "A point store reads the whole period's times and positions whatever the box, so shorten the period, or pick fewer months or fewer days."
    };
    return cap(po);
  }

  async function run(sel, opts) {
    opts = opts || {};
    var ctx = makeCtx(opts.signal);
    var t0 = Date.now();
    try {
      var d = await storeDesc(sel && sel.store, ctx);
      if (d.kind === "grid") {
        var plan = await gridPlan(sel, ctx);
        if (plan.outBytes > CAP_OUT) throw new Error("over the cap: the result would be " + fmtMB(plan.outBytes) + " of arrays (limit " + fmtMB(CAP_OUT) + ")");
        var g = await gridRun(plan, ctx, opts.onProgress);
        return gridResult(plan, g, ctx, t0);
      }
      var pp = await pointPlan(sel, ctx);
      var pe = pointEstimateOf(pp);
      if (pe.readBytes > CAP_READ) throw new Error("over the cap: the selection reads up to " + fmtMB(pe.readBytes) + " (limit " + fmtMB(CAP_READ) + ")");
      if (pe.outBytes > CAP_OUT) throw new Error("over the cap: the result could be " + fmtMB(pe.outBytes) + " of arrays (limit " + fmtMB(CAP_OUT) + ")");
      var r = await pointRun(pp, ctx, opts.onProgress);
      return pointResult(pp, r, ctx, t0);
    } catch (e) {
      throw wrapAbort(ctx, e);
    }
  }

  async function preview(sel, opts) {
    opts = opts || {};
    var ctx = makeCtx(opts.signal);
    var t0 = Date.now();
    try {
      var d = await storeDesc(sel && sel.store, ctx);
      if (d.kind === "grid") {
        // the first selected native frame whose box holds a stored tile
        var s1 = Object.assign({}, sel, { step: "native" });
        var plan = await gridPlan(s1, ctx);
        // the first frame with a stored tile in the box AND a finite value
        // inside the box (a stored tile can be all cloud where the box is);
        // at most PREVIEW_GRID_TRIES frames are decoded
        var tries = 0, fallback = null;
        for (var i = 0; i < plan.bins.length && tries < PREVIEW_GRID_TRIES; i++) {
          var bn = plan.bins[i];
          var tiles = await readBinIndex(plan, bn, ctx);
          for (var k = 0; k < bn.frames.length && tries < PREVIEW_GRID_TRIES; k++) {
            var fr = bn.frames[k];
            if (!tiles.some(function (t) { return t.fr === fr; })) continue;
            tries++;
            var one = sliceGridPlan(plan, fr);
            var g = await gridRun(one, ctx, null);
            var res1 = gridResult(one, g, ctx, t0);
            for (var q = 0; q < g.data.length; q++) if (g.data[q] === g.data[q]) return res1;
            if (!fallback) fallback = res1;
          }
        }
        if (fallback) { fallback.notes.push("no frame among the first " + tries + " with data in the box had a finite value there"); return fallback; }
        if (plan.frames.length) {
          var one2 = sliceGridPlan(plan, plan.frames[0]);
          return gridResult(one2, await gridRun(one2, ctx, null), ctx, t0);
        }
        return gridResult(plan, { data: new Float32Array(0), count: null }, ctx, t0, true);
      }
      var pp = await pointPlan(sel, ctx);
      var r = await pointRun(pp, ctx, null, { preview: true });
      return pointResult(pp, r, ctx, t0);
    } catch (e) {
      throw wrapAbort(ctx, e);
    }
  }

  function sliceGridPlan(plan, fr) {
    var bn = plan.bins.find(function (b) { return b.b === fr.b; });
    var one = Object.assign({}, plan);
    var f2 = Object.assign({}, fr, { si: 0 });
    var tg = plan.tg, nty = tg.n_tiles_y, ntx = tg.n_tiles_x, ih = tg.index_header_bytes;
    var spans = [];
    plan.geo.tys.forEach(function (ty) {
      plan.geo.txRuns.forEach(function (run) {
        var a = (fr.f * nty + ty) * ntx + run[0], z = (fr.f * nty + ty) * ntx + run[1];
        spans.push([ih + 16 * a, ih + 16 * (z + 1)]);
      });
    });
    one.bins = [Object.assign({}, bn, { frames: [f2], idxRanges: coalesce(spans, GAP_IDX, MAX_RANGE) })];
    one.frames = [f2];
    one.T = 1;
    one.times = new Float64Array([EPOCH_UNIX + fr.t]);
    one.mean = plan.s.res !== "native";
    one.countBytes = one.mean ? (plan.geo.maxPerCell > 65535 ? 4 : 2) : 0;
    return one;
  }

  function gridResult(plan, g, ctx, t0, empty) {
    var units = plan.conv.map(function (c) { return c.unit; });
    return {
      kind: "grid", store: plan.d.name, channels: plan.s.channels.slice(), units: units,
      lat: plan.geo.outLat, lon: plan.geo.outLon, time: empty ? new Float64Array(0) : plan.times,
      data: g.data, count: g.count, sel: plan.s,
      notes: gridNotes(plan), frames: empty ? 0 : plan.frames.length, group: plan.group,
      source: groupUrl(plan.d, plan.group), title: plan.d.title,
      stats: { requests: ctx.stats.requests, bytes: ctx.stats.bytes, ms: Date.now() - t0 }
    };
  }

  function pointResult(plan, r, ctx, t0) {
    var units = plan.conv.map(function (c) { return c.unit; });
    var notes = [];
    plan.conv.forEach(function (c, k) { if (c.note) notes.push(plan.s.channels[k] + ": " + c.note); });
    if (r.truncated) notes.push(r.truncated);
    var common = {
      store: plan.d.name, channels: plan.s.channels.slice(), units: units, sel: plan.s,
      rowsRead: plan.br.rows, source: cfg.base + plan.d._rel + "/", title: plan.d.title,
      stats: { requests: ctx.stats.requests, bytes: ctx.stats.bytes, ms: Date.now() - t0 },
      truncated: !!r.truncated
    };
    if (r.grid) {
      notes.push("binned from point observations to " + plan.res + "° cells per " + (plan.step === "pentad" ? "five-day bin" : plan.step) +
        "; each value is the mean of the observations in its cell and step, and the count arrays say how many");
      return Object.assign({ kind: "grid", lat: r.lat, lon: r.lon, time: r.time, data: r.data, count: r.count, notes: notes, binnedFrom: "points" }, common);
    }
    return Object.assign({ kind: "points", time: r.time, lat: r.lat, lon: r.lon, values: r.values, platform: r.platform, qc: r.qc, notes: notes }, common);
  }

  // ============================================================== NetCDF ===
  // NetCDF-3 classic, 64-bit offset format (CDF-2), big-endian, no record
  // dimension. https://docs.unidata.ucar.edu/netcdf-c/current/file_format_specifications.html
  var NC = { BYTE: 1, CHAR: 2, SHORT: 3, INT: 4, FLOAT: 5, DOUBLE: 6 };
  var NC_SIZE = { 1: 1, 2: 1, 3: 2, 4: 4, 5: 4, 6: 8 };

  function ncName(s) {
    var n = String(s).replace(/[^A-Za-z0-9_.@+\-]/g, "_");
    if (!/^[A-Za-z_]/.test(n)) n = "v_" + n;
    return n;
  }

  function utf8(s) { return new TextEncoder().encode(String(s)); }

  function HeaderWriter() { this.parts = []; this.len = 0; }
  HeaderWriter.prototype.bytes = function (u8) { this.parts.push(u8); this.len += u8.length; };
  HeaderWriter.prototype.i32 = function (v) { var b = new Uint8Array(4); new DataView(b.buffer).setInt32(0, v, false); this.bytes(b); };
  HeaderWriter.prototype.i64 = function (v) { var b = new Uint8Array(8); var dv = new DataView(b.buffer); dv.setUint32(0, Math.floor(v / 4294967296), false); dv.setUint32(4, v % 4294967296, false); this.bytes(b); return b; };
  HeaderWriter.prototype.pad = function () { var p = (4 - this.len % 4) % 4; if (p) this.bytes(new Uint8Array(p)); };
  HeaderWriter.prototype.name = function (s) { var u = utf8(s); this.i32(u.length); this.bytes(u); this.pad(); };
  HeaderWriter.prototype.attrs = function (list) {
    if (!list.length) { this.i32(0); this.i32(0); return; }
    this.i32(12); this.i32(list.length);
    var self = this;
    list.forEach(function (a) {
      self.name(a[0]);
      var v = a[1];
      if (typeof v === "string") {
        var u = utf8(v); self.i32(NC.CHAR); self.i32(u.length); self.bytes(u); self.pad();
      } else {
        var t = a[2] || NC.DOUBLE, arr = Array.isArray(v) ? v : [v];
        self.i32(t); self.i32(arr.length);
        var b = new Uint8Array(arr.length * NC_SIZE[t]), dv = new DataView(b.buffer);
        arr.forEach(function (x, i) {
          if (t === NC.DOUBLE) dv.setFloat64(8 * i, x, false);
          else if (t === NC.FLOAT) dv.setFloat32(4 * i, x, false);
          else if (t === NC.INT) dv.setInt32(4 * i, x, false);
          else if (t === NC.SHORT) dv.setInt16(2 * i, x, false);
        });
        self.bytes(b); self.pad();
      }
    });
  };

  // vars: [{name, dims:[dimIdx], type, attrs:[[k,v,t]], n, fill(DataView, byteOffset)}]
  function writeNetCDF(dims, gatts, vars) {
    dims.forEach(function (d) { if (!(d[1] > 0)) throw new Error("NetCDF: dimension " + d[0] + " is empty — there is nothing to write"); });
    var h = new HeaderWriter();
    h.bytes(new Uint8Array([0x43, 0x44, 0x46, 0x02]));
    h.i32(0);
    h.i32(10); h.i32(dims.length);
    dims.forEach(function (d) { h.name(d[0]); h.i32(d[1]); });
    h.attrs(gatts);
    h.i32(11); h.i32(vars.length);
    var begins = [];
    vars.forEach(function (v) {
      h.name(v.name);
      h.i32(v.dims.length);
      v.dims.forEach(function (di) { h.i32(di); });
      h.attrs(v.attrs);
      h.i32(v.type);
      var raw = v.n * NC_SIZE[v.type];
      v.vsize = raw + (4 - raw % 4) % 4;
      if (v.vsize > 4294967292) throw new Error("NetCDF-3: variable " + v.name + " is over 4 GiB");
      h.i32(v.vsize);
      begins.push(h.i64(0));
    });
    var pos = h.len;
    var blobs = h.parts.slice();
    vars.forEach(function (v, i) {
      var dv = new DataView(begins[i].buffer);
      dv.setUint32(0, Math.floor(pos / 4294967296), false);
      dv.setUint32(4, pos % 4294967296, false);
      pos += v.vsize;
    });
    vars.forEach(function (v) {
      // write in pieces of at most 16 MB so a large variable never needs a
      // second full-size copy
      var bytesPer = NC_SIZE[v.type], per = Math.max(1, Math.floor(16 * MB / bytesPer)), done = 0;
      while (done < v.n) {
        var k = Math.min(per, v.n - done);
        var buf = new ArrayBuffer(k * bytesPer), dvw = new DataView(buf);
        v.fill(dvw, done, k);
        blobs.push(new Uint8Array(buf));
        done += k;
      }
      var padn = v.vsize - v.n * bytesPer;
      if (padn) blobs.push(new Uint8Array(padn));
    });
    return new Blob(blobs, { type: "application/x-netcdf" });
  }

  function f32Filler(arr, mapIndex) {
    return function (dv, start, k) { for (var i = 0; i < k; i++) dv.setFloat32(4 * i, arr[mapIndex(start + i)], false); };
  }

  function globalAttrs(result, extra) {
    var selJson = JSON.stringify(result.sel);
    var a = [
      ["Conventions", "CF-1.8"],
      ["title", "family 1.gf store " + result.store + (result.title ? " — " + result.title : "")],
      ["source", "Hugging Face dataset chfrank/earth-tensors, tensors/family1_gf/" + result.store + " (family 1.gf), read by blauewelt.org's Data tab (src/f1data.js)"],
      ["source_url", result.source || cfg.base],
      ["selection", selJson],
      ["history", new Date().toISOString() + " written by src/f1data.js from HTTP range reads of the store"]
    ];
    if (result.notes && result.notes.length) a.push(["comment", result.notes.join("; ")]);
    if (result.frames != null) a.push(["frames_read", result.frames, NC.INT]);
    return a.concat(extra || []);
  }

  function toNetCDF(result) {
    if (!result || !result.kind) throw new Error("toNetCDF needs a Result from run() or preview()");
    if (result.kind === "grid") {
      var T = result.time.length, H = result.lat.length, W = result.lon.length, C = result.channels.length, HW = H * W;
      var dims = [["time", T], ["lat", H], ["lon", W]];
      var vars = [
        { name: "time", dims: [0], type: NC.DOUBLE, n: T, attrs: [["standard_name", "time"], ["long_name", "start of each time step"], ["units", "seconds since 1970-01-01 00:00:00"], ["calendar", "proleptic_gregorian"], ["axis", "T"]],
          fill: function (dv, s, k) { for (var i = 0; i < k; i++) dv.setFloat64(8 * i, result.time[s + i], false); } },
        { name: "lat", dims: [1], type: NC.DOUBLE, n: H, attrs: [["standard_name", "latitude"], ["long_name", "latitude of the cell centre"], ["units", "degrees_north"], ["axis", "Y"]],
          fill: function (dv, s, k) { for (var i = 0; i < k; i++) dv.setFloat64(8 * i, result.lat[s + i], false); } },
        { name: "lon", dims: [2], type: NC.DOUBLE, n: W, attrs: [["standard_name", "longitude"], ["long_name", "longitude of the cell centre (monotonic; may exceed 180 across the dateline)"], ["units", "degrees_east"], ["axis", "X"]],
          fill: function (dv, s, k) { for (var i = 0; i < k; i++) dv.setFloat64(8 * i, result.lon[s + i], false); } }
      ];
      var used = { time: 1, lat: 1, lon: 1 };
      var names = result.channels.map(function (c) {
        var n = ncName(c);
        while (used[n]) n += "_";
        used[n] = 1;
        return n;
      });
      result.channels.forEach(function (c, k) {
        var map = function (j) { var t = Math.floor(j / HW); return (t * C + k) * HW + (j - t * HW); };
        var at = [["long_name", c], ["units", result.units[k] || ""], ["_FillValue", NaN, NC.FLOAT]];
        if (result.count) at.push(["cell_methods", "time: mean (of finite observations) area: mean"]);
        vars.push({ name: names[k], dims: [0, 1, 2], type: NC.FLOAT, n: T * HW, attrs: at, fill: f32Filler(result.data, map) });
      });
      if (result.count) {
        result.channels.forEach(function (c, k) {
          var nm = names[k] + "_count";
          while (used[nm]) nm += "_";
          used[nm] = 1;
          var map = function (j) { var t = Math.floor(j / HW); return (t * C + k) * HW + (j - t * HW); };
          vars.push({ name: nm, dims: [0, 1, 2], type: NC.INT, n: T * HW,
            attrs: [["long_name", "number of finite observations averaged into " + c], ["units", "1"]],
            fill: function (dv, s, kk) { for (var i = 0; i < kk; i++) dv.setInt32(4 * i, result.count[map(s + i)], false); } });
        });
      }
      return writeNetCDF(dims, globalAttrs(result), vars);
    }
    if (result.kind === "points") {
      var N = result.time.length, Cp = result.channels.length, SL = 20;
      var pdims = [["obs", N], ["platform_strlen", SL]];
      var pvars = [
        { name: "time", dims: [0], type: NC.DOUBLE, n: N, attrs: [["standard_name", "time"], ["units", "seconds since 1970-01-01 00:00:00"], ["calendar", "proleptic_gregorian"]],
          fill: function (dv, s, k) { for (var i = 0; i < k; i++) dv.setFloat64(8 * i, result.time[s + i], false); } },
        { name: "lat", dims: [0], type: NC.FLOAT, n: N, attrs: [["standard_name", "latitude"], ["units", "degrees_north"]], fill: f32Filler(result.lat, function (j) { return j; }) },
        { name: "lon", dims: [0], type: NC.FLOAT, n: N, attrs: [["standard_name", "longitude"], ["units", "degrees_east"]], fill: f32Filler(result.lon, function (j) { return j; }) }
      ];
      var pused = { time: 1, lat: 1, lon: 1, qc: 1, platform: 1 };
      result.channels.forEach(function (c, k) {
        var n = ncName(c);
        while (pused[n]) n += "_";
        pused[n] = 1;
        pvars.push({ name: n, dims: [0], type: NC.FLOAT, n: N, attrs: [["long_name", c], ["units", result.units[k] || ""], ["_FillValue", NaN, NC.FLOAT]],
          fill: f32Filler(result.values, function (j) { return j * Cp + k; }) });
      });
      pvars.push({ name: "qc", dims: [0], type: NC.SHORT, n: N, attrs: [["long_name", "quality flag: 0 not assessed, 1 good, 2 probably good, 3+ the source's worse grades"]],
        fill: function (dv, s, k) { for (var i = 0; i < k; i++) dv.setInt16(2 * i, result.qc[s + i], false); } });
      pvars.push({ name: "platform", dims: [0, 1], type: NC.CHAR, n: N * SL, attrs: [["long_name", "platform id as a decimal integer (int64 in the store)"]],
        fill: function (dv, s, k) {
          for (var i = 0; i < k; i++) {
            var j = s + i, row = Math.floor(j / SL), col = j - row * SL;
            var str = result.platform[row].toString();
            dv.setUint8(i, col < str.length ? str.charCodeAt(col) : 0);
          }
        } });
      return writeNetCDF(pdims, globalAttrs(result), pvars);
    }
    throw new Error("unknown result kind " + result.kind);
  }

  // ================================================================= CSV ===
  function f32str(v) {
    if (v !== v) return "";
    for (var p = 1; p <= 9; p++) {
      var s = v.toPrecision(p);
      if (Math.fround(Number(s)) === v) return String(Number(s));
    }
    return String(v);
  }
  function f64str(v) {
    if (v !== v) return "";
    return String(Number(v.toPrecision(12)));
  }

  function toCSV(result) {
    if (!result || !result.kind) throw new Error("toCSV needs a Result from run() or preview()");
    var parts = [], buf = [], size = 0;
    function line(s) {
      buf.push(s);
      size += s.length + 1;
      if (size > 1 << 20) { parts.push(buf.join("\n") + "\n"); buf = []; size = 0; }
    }
    var esc = function (s) { return /[",\n]/.test(s) ? '"' + s.replace(/"/g, '""') + '"' : s; };
    if (result.kind === "points") {
      var C = result.channels.length;
      line(["time", "lat", "lon"].concat(result.channels.map(esc), ["platform", "qc"]).join(","));
      for (var i = 0; i < result.time.length; i++) {
        var row = [isoOfUnix(result.time[i]), f32str(result.lat[i]), f32str(result.lon[i])];
        for (var k = 0; k < C; k++) row.push(f32str(result.values[i * C + k]));
        row.push(result.platform[i].toString(), String(result.qc[i]));
        line(row.join(","));
      }
    } else {
      // one row per (time, lat, lon) cell in which at least one channel is
      // finite; an all-missing cell writes no row
      var Cg = result.channels.length, H = result.lat.length, W = result.lon.length, HW = H * W;
      var head = ["time", "lat", "lon"].concat(result.channels.map(esc));
      if (result.count) head = head.concat(result.channels.map(function (c) { return esc(c + "_count"); }));
      line(head.join(","));
      for (var t = 0; t < result.time.length; t++) {
        var ts = isoOfUnix(result.time[t]);
        for (var y = 0; y < H; y++) {
          for (var x = 0; x < W; x++) {
            var any = false, vals = [], cnts = [];
            for (var c = 0; c < Cg; c++) {
              var j = (t * Cg + c) * HW + y * W + x, v = result.data[j];
              if (v === v) any = true;
              vals.push(f32str(v));
              if (result.count) cnts.push(String(result.count[j]));
            }
            if (any) line([ts, f64str(result.lat[y]), f64str(result.lon[x])].concat(vals, cnts).join(","));
          }
        }
      }
    }
    if (buf.length) parts.push(buf.join("\n") + "\n");
    return new Blob(parts, { type: "text/csv" });
  }

  return {
    configure: configure,
    loadRegistry: function () { return loadRegistry(makeCtx(null)); },
    estimate: estimate,
    run: run,
    preview: preview,
    toNetCDF: toNetCDF,
    toCSV: toCSV,
    CAPS: { readBytes: CAP_READ, outBytes: CAP_OUT, concurrency: MAX_CONCURRENCY },
    DEFAULT_BASE: DEFAULT_BASE,
    // for tests only — not part of the contract
    _internal: { parseNpyHeader: parseNpyHeader, civil: civil, daysFromCivil: daysFromCivil, F16: F16, normBox: normBox, conversionOf: conversionOf, isoOfUnix: isoOfUnix }
  };
});

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
  var HUB_ROOT = "https://huggingface.co/datasets/chfrank/earth-tensors/resolve/main/";
  // ml/build_family1_registry.py names family 7.2d's registry family72d.json
  var F72D_REGISTRY_URL = HUB_ROOT + "tensors/family7_2d/family72d.json";
  var DEFAULT_BASE = HUB_ROOT + "tensors/family1_gf/";
  // The families the Data tab offers, in the order its store list shows them.
  // `url` is a registry JSON on the Hub; `norms` / `fishing` / `clim` are the
  // site's own indexes (relative to the page, or to cfg.siteBase in node).
  // A registry that fails to load is reported in `registry.errors` and the
  // others still load; one marked `optional` that answers 404 is simply absent.
  var DEFAULT_REGISTRIES = [
    { family: "1.gf", title: "Fine observations (family 1.gf)", kind: "family1",
      url: DEFAULT_BASE + "family1gf.json" },
    // family 1.2 = family 1.gf (inherited BY REFERENCE: its stores are the
    // same bytes, so they are listed once, under 1.gf) + ERA5 on pressure
    // levels. A store the registry marks not built is never selectable; it is
    // named in `registry.coming` and appears the day the registry flips it.
    { family: "1.2", title: "Atmosphere on pressure levels (family 1.2 — ERA5 reanalysis)", kind: "family1",
      optional: true, url: HUB_ROOT + "tensors/family1_2/family12.json" },
    { family: "10", title: "Global tensor and point observations (family 10)", kind: "family10",
      url: HUB_ROOT + "tensors/family10_2/family10.json", norms: "data/family7_index.json" },
    // family 7.2d: the global tensor's channels as DAILY frames, in the
    // family-1 registry schema (sharded tier-G groups, physical units).
    // Optional: while the file answers 404 the group is silently absent.
    { family: "7.2d", title: "Global tensor, daily (family 7.2d)", kind: "family1",
      optional: true, url: F72D_REGISTRY_URL },
    // the derived maps: the fishing-effort grid, and the monthly normals of
    // the global tensor composed over any period from its per-year monthly
    // sums and counts (E-086) — which replaced the four fixed "all years"
    // climatology stores (`clim:` still loads for a caller that asks)
    { family: "derived", title: "Derived maps", kind: "derived",
      fishing: "data/fishing_index.json", monthly: "data/family7_monthly_index.json" }
  ];
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

  var cfg = { base: DEFAULT_BASE, fetch: null, concurrency: MAX_CONCURRENCY,
    registries: DEFAULT_REGISTRIES.slice(), siteBase: null };
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
      // the phase-1 form: one family 1.gf registry at `base` and nothing else
      var b = String(o.base);
      cfg.base = b.endsWith("/") ? b : b + "/";
      cfg.registries = [{ family: "1.gf", title: "Fine observations (family 1.gf)", kind: "family1",
        url: cfg.base + "family1gf.json" }];
    }
    if (o.registries != null) {
      if (!Array.isArray(o.registries)) throw new Error("registries must be a list");
      cfg.registries = o.registries.map(function (r) { return Object.assign({}, r); });
    }
    if (o.siteBase !== undefined) cfg.siteBase = o.siteBase;
    if (o.fetch !== undefined) cfg.fetch = o.fetch;
    if (o.concurrency != null) {
      cfg.concurrency = Math.max(1, Math.min(MAX_CONCURRENCY, o.concurrency | 0));
    }
    // the normals' merge gap (bytes): for measuring the two read strategies
    if (o.normalsMergeGap !== undefined) cfg.normalsMergeGap = o.normalsMergeGap == null ? null : Number(o.normalsMergeGap);
    cache.clear();
    idxCache.clear();
    idxCacheBytes = 0;
    return { base: cfg.base, concurrency: cfg.concurrency, registries: cfg.registries.slice() };
  }

  // a site-relative URL ("data/x.json") against the page, or cfg.siteBase
  function siteUrl(u) {
    if (/^[a-z]+:\/\//i.test(u)) return u;
    var b = cfg.siteBase;
    if (!b && typeof document !== "undefined" && document.baseURI) b = document.baseURI;
    if (!b && typeof location !== "undefined" && location.href) b = location.href;
    if (!b) throw new Error("no base to resolve " + JSON.stringify(u) + " against — configure({siteBase})");
    return new URL(u, b).href;
  }

  function dirOf(u) { return String(u).replace(/[^/]*$/, ""); }

  // the site's own small JSON (an index we publish beside the page): a plain
  // GET — Range is the rule for the store's data, not for our own metadata,
  // and a static test server answers it 200
  function readSiteJSON(url, ctx) {
    return cached("sitejson:" + url, async function () {
      checkAbort(ctx && ctx.signal);
      var res = await fetchFn()(url, { signal: ctx && ctx.signal });
      if (res.status !== 200) {
        var e = new Error("HTTP " + res.status + " for " + url);
        e.status = res.status; e.url = url;
        throw e;
      }
      try { return JSON.parse(await res.text()); }
      catch (e2) { throw new Error("not JSON: " + url + " (" + e2.message + ")"); }
    });
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
      "The SWOT satellite's wide-swath sea-surface height anomaly at 2 km, one row per valid pixel — a very large store, so keep the period short."],
    // family 10 — [title, one-sentence gist, cadence label (optional)]
    "10/g025": ["Ocean surface state, 0.25° five-day (the global tensor)",
      "The forecaster's own global input at 0.25°: surface current speed and direction, mixed-layer depth, sea-surface height, observed sea-surface temperature and sea ice, as five-day means since 1982."],
    "10/g100": ["Air–sea fluxes, weather and land, 1° five-day (the global tensor)",
      "The global tensor's 1° group: wind stress, 2 m air and skin temperature, 10 m wind, surface pressure, precipitation, snow, soil moisture and temperature, and the turbulent heat fluxes, as five-day means since 1982."],
    "10/oc025": ["Ocean colour, 0.25° five-day (the global tensor)",
      "Satellite chlorophyll (as its base-10 logarithm) and the clear-sky fraction behind each value, averaged onto 0.25° five-day cells, from September 1997."],
    "10/rg100": ["Ocean temperature and salinity at 16 depths, 1° monthly (Argo)",
      "The Roemmich–Gilson monthly map built from Argo floats: temperature and salinity at 16 pressures from 10 to 1,900 dbar, on a 1° grid from 2004.", "monthly"],
    "10/argo": ["Argo float profiles (temperature and salinity at 16 depths)",
      "Every quality-controlled Argo float profile since 2004, interpolated to 16 pressures from 10 to 1,900 dbar, one row per profile."],
    "10/gdp": ["Surface drifters (Global Drifter Program)",
      "Drifting buoys' surface current (eastward and northward), sea-surface temperature and whether the drogue was attached, every six hours since 1979."],
    "10/gtmba": ["Tropical moored buoys (TAO, PIRATA, RAMA)",
      "The tropical mooring arrays' daily sea and air temperature, salinity, wind, currents and temperature at depth, one row per site and day since 1977."],
    "10/socat": ["Ship measurements of ocean CO₂ (SOCAT)",
      "Measurements of the ocean surface's CO₂ fugacity from ships, with the sea temperature, salinity and air pressure taken alongside, since 1957."],
    "10/slatrack": ["Sea-level anomaly along altimeter tracks",
      "Sea-level anomaly (filtered and unfiltered) and the mean dynamic topography along every altimeter's ground track since 1993, one row per second — a very large store, so keep the period short."],
    "10/fishing": ["Fishing effort per vessel and day (AIS, Global Fishing Watch)",
      "Apparent fishing hours and hours broadcasting, one row per vessel, day and 0.1° cell, from AIS since 2012 — Powered by Global Fishing Watch, CC BY-NC 4.0; absence of effort is not absence of fishing."],
    // family 1.2 — ERA5, a model-filled reanalysis, not an observation
    "1.2/era5_t": ["Air temperature on 13 pressure levels (ERA5), 1° six-hourly",
      "ECMWF's ERA5 reanalysis — a weather model filled in with observations, ~31 km natively and averaged here onto a 1° grid, not a measurement — giving air temperature in kelvin at 13 pressure levels from 50 hPa (~20 km up) to 1000 hPa (the surface), as instants at 00, 06, 12 and 18 UTC since 1982.",
      "six-hourly instants"],
    "1.2/era5_q": ["Specific humidity on 13 pressure levels (ERA5), 1° six-hourly",
      "ECMWF's ERA5 reanalysis — model-filled, ~31 km natively and averaged onto 1°, not a measurement — giving specific humidity in grams of water vapour per kilogram of air (1000 × ERA5's kg/kg) at 13 pressure levels, every six hours since 1982.",
      "six-hourly instants"],
    "1.2/era5_u": ["Eastward wind on 13 pressure levels (ERA5), 1° six-hourly",
      "ECMWF's ERA5 reanalysis — model-filled, ~31 km natively and averaged onto 1°, not a measurement — giving the eastward wind in m/s (positive toward the east) at 13 pressure levels, every six hours since 1982.",
      "six-hourly instants"],
    "1.2/era5_v": ["Northward wind on 13 pressure levels (ERA5), 1° six-hourly",
      "ECMWF's ERA5 reanalysis — model-filled, ~31 km natively and averaged onto 1°, not a measurement — giving the northward wind in m/s (positive toward the north) at 13 pressure levels, every six hours since 1982.",
      "six-hourly instants"],
    // the monthly normals over a period you choose (E-086)
    "derived/normals_g025": ["Monthly normals of the global tensor — sea-surface temperature, currents, sea-surface height, mixed layer and sea ice (0.25°)",
      "The forecaster's own 0.25° ocean channels averaged per calendar month over the years you choose — a climatology for any period (say 1991–2020), or each year's monthly mean side by side — composed from the tensor's per-year monthly sums and counts.",
      "monthly means over the years you choose"],
    "derived/normals_g100": ["Monthly normals of the global tensor — air–sea fluxes, weather and land (1°)",
      "The global tensor's 1° atmosphere-and-land channels (wind stress, 2 m air and skin temperature, wind, pressure, rain, snow, soil, heat fluxes) averaged per calendar month over the years you choose.",
      "monthly means over the years you choose"],
    "derived/normals_oc025": ["Monthly normals of the global tensor — ocean colour (0.25°)",
      "Satellite chlorophyll (as its base-10 logarithm) and its clear-sky coverage, averaged per calendar month over the years you choose, from September 1997.",
      "monthly means over the years you choose"],
    "derived/normals_rg100": ["Monthly normals of the global tensor — ocean temperature and salinity at 16 depths (1°)",
      "The Argo gridded temperature and salinity at 16 pressures from 10 to 1,900 dbar, averaged per calendar month over the years you choose, from 2004.",
      "monthly means over the years you choose"],
    // derived maps
    "derived/fishing_grid": ["Fishing effort map, 0.25° monthly (AIS)",
      "Global Fishing Watch's apparent fishing hours and AIS broadcasting hours summed per 0.25° cell and month since 2012 — CC BY-NC 4.0; zero means no vessel broadcast there, not that nobody fished.", "monthly sums"],
    "derived/clim_g025": ["What the forecaster calls normal: ocean surface, 0.25°",
      "The model climatology — the calendar-month average over all years 1982–2024 that the forecaster is trained and scored against — for the 0.25° ocean surface channels; twelve maps, no years."],
    "derived/clim_g100": ["What the forecaster calls normal: fluxes, weather and land, 1°",
      "The model climatology (all years 1982–2024, one map per calendar month) for the 1° flux, weather and land channels; twelve maps, no years."],
    "derived/clim_oc025": ["What the forecaster calls normal: ocean colour, 0.25°",
      "The model climatology (all years, one map per calendar month) of the 0.25° chlorophyll channels; twelve maps, no years."],
    "derived/clim_rg100": ["What the forecaster calls normal: ocean temperature and salinity at depth, 1°",
      "The model climatology (all years, one map per calendar month) of the Argo temperature and salinity at 16 depths; twelve maps, no years."]
  };

  var VAR_LABELS = { "1.2/era5_t": "air temperature", "1.2/era5_q": "specific humidity",
    "1.2/era5_u": "eastward wind", "1.2/era5_v": "northward wind" };

  // documented physical conversions, keyed on the STORED unit's own words
  function conversionOf(unit) {
    var u = String(unit || "");
    if (/^g\/kg\s*\(1000\s*x/i.test(u)) {
      return { unit: "g/kg", offset: 0, note: "grams of water vapour per kilogram of air, as stored: 1000 × ERA5's kg/kg" };
    }
    var mw = /^m\/s\s*\((eastward|northward)\)\s*$/i.exec(u);
    if (mw) return { unit: "m/s", offset: 0, note: null, direction: mw[1].toLowerCase() };
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

  function treeUrl(u) {
    return /\/resolve\/main\//.test(u) ? u.replace("/resolve/main/", "/tree/main/") : u;
  }

  function relPath(reg, path) {
    var root = String(reg.hf_root || "tensors/family1_gf").replace(/\/+$/, "");
    var p = String(path || "");
    if (p.indexOf(root + "/") === 0) return p.slice(root.length + 1);
    throw new Error("store path " + p + " is not under the registry's hf_root " + root);
  }

  // a store by `family/name`, by sel {family, store}, or by a bare name when
  // exactly one family has it
  async function storeDesc(sel, ctx) {
    var reg = await loadRegistry(ctx);
    var fam = sel && typeof sel === "object" ? sel.family : null;
    var name = sel && typeof sel === "object" ? sel.store : sel;
    var key = fam != null && fam !== "" ? fam + "/" + name : String(name);
    var s = reg._byName[key];
    if (s === AMBIGUOUS) {
      throw new Error("the store name " + JSON.stringify(name) + " exists in several families (" +
        reg.stores.filter(function (x) { return x.name === name; }).map(function (x) { return x.family; }).join(", ") +
        "): name one as sel.family");
    }
    if (!s) throw new Error("no readable store " + JSON.stringify(key) + " in the registry");
    return s;
  }
  var AMBIGUOUS = { ambiguous: true };

  // Channels whose names encode a LEVEL ("rg_t10", "temp_1900", "DOXY_10"):
  // a variable is levelled when at least three of its channels differ only
  // by a trailing number. The tab then offers a variable × level picker.
  function annotateLevels(d) {
    var byVar = {};
    d.channels.forEach(function (c) {
      var m = /^(.+)_(\d+(?:\.\d+)?)$/.exec(c.name) || /^(.*[A-Za-z])(\d+(?:\.\d+)?)$/.exec(c.name);
      if (m) (byVar[m[1]] = byVar[m[1]] || []).push([c, Number(m[2])]);
    });
    var levels = new Set(), vars = [];
    Object.keys(byVar).forEach(function (v) {
      if (byVar[v].length < 3) return;
      byVar[v].forEach(function (x) { x[0].var = v; x[0].level = x[1]; levels.add(x[1]); });
    });
    if (!levels.size) return;
    var seen = {};
    d.channels.forEach(function (c) {
      var v = c.var || c.name;
      if (!c.var) { c.var = c.name; c.level = null; }
      if (seen[v]) { seen[v].levels.push(c.level); return; }
      var lab = c.label ? String(c.label).replace(/\s+at\s+[\d.]+\s*[a-zA-Z]*\s*$/, "") : null;
      seen[v] = { var: v, label: lab, unit: c.unit, levels: c.level == null ? [] : [c.level] };
      vars.push(seen[v]);
    });
    d.levels = Array.from(levels).sort(function (a, b) { return a - b; });
    d.vars = vars;
    var lu = (d.channels.find(function (c) { return c.label && /\bdbar\b/.test(c.label); }) || {}).label;
    d.levelUnit = d.levelUnitHint || "dbar";
  }

  function baseDesc(fam, name, extra) {
    var gist = GISTS[fam.family + "/" + name] || GISTS[name];
    var d = Object.assign({
      name: name, code: name, family: fam.family, familyTitle: fam.title, id: fam.family + "/" + name,
      title: gist ? gist[0] : name, gist: gist ? gist[1] : name,
      kind: "grid", channels: [], span: null, frameSeconds: null, framesPerBin: null,
      grid: null, folderUrl: null, cadence: null, cadenceLabel: null, subDaily: false,
      N: null, binFirst: null, binLast: null, groups: null, layout: null,
      levels: null, vars: null, calendar: false
    }, extra || {});
    if (gist && gist[2]) d.cadenceLabel = gist[2];
    return d;
  }

  // ---- family 1.gf (and any registry in its shape: a `groups` list of tier
  // G sharded grids and tier P point stores under `hf_root`)
  async function loadFamily1(fam, ctx, out) {
    var url = /^[a-z]+:\/\//i.test(fam.url) ? fam.url : siteUrl(fam.url);
    var reg = await readJSON(url, ctx);
    var base = fam.base || dirOf(url);
    var groups = Array.isArray(reg.groups) ? reg.groups : Object.values(reg.groups || {});
    // a registry that inherits another family's stores BY REFERENCE (family
    // 1.2 ⊃ 1.gf) lists them once, under the family that owns the bytes
    var inherited = new Set(reg.inherits_block && Array.isArray(reg.inherits_block.groups) ? reg.inherits_block.groups : []);
    var jobs = [];
    groups.forEach(function (g) {
      if (!g || inherited.has(g.name)) return;
      if (g.distribution !== "public" || (g.tier !== "G" && g.tier !== "P")) return;
      if (!g.built) {
        // published in the registry, not yet on the Hub: named, never offered.
        // A store whose licence still waits on the producer is not "coming" —
        // nothing says it will be published — so it is not named either.
        if (g.licence && (g.licence.pending || g.licence.redistribution_confirmed === false)) return;
        var cg = GISTS[fam.family + "/" + g.name] || GISTS[g.name];
        out.coming.push({ family: fam.family, familyTitle: fam.title, store: g.name,
          title: cg ? cg[0] : String(g.title || g.name).replace(/\s+—\s+family.*$/, "") });
        return;
      }
      var rel = relPath(reg, g.path);
      var chans = (g.channels || []).map(function (c) {
        var cv = conversionOf(c.unit);
        var o = { name: c.name, unit: cv.unit, min: c.min, max: c.max, storedUnit: c.unit, note: cv.note, label: c.label || null };
        if (cv.offset) { o.min = c.min + cv.offset; o.max = c.max + cv.offset; }
        return o;
      });
      var d = baseDesc(fam, g.name, {
        kind: g.tier === "G" ? "grid" : "points",
        layout: g.tier === "G" ? "sharded" : "points",
        channels: chans,
        frameSeconds: g.tier === "G" ? g.frame_seconds : null,
        framesPerBin: g.tier === "G" ? g.frames_per_bin : null,
        folderUrl: treeUrl(base + rel), cadence: g.cadence || null,
        subDaily: g.tier === "P" ? true : (g.frame_seconds < 86400),
        N: g.N == null ? null : g.N
      });
      if (!GISTS[fam.family + "/" + g.name] && !GISTS[g.name] && g.title) { d.title = g.title; d.gist = g.title; }
      // the licence the registry states, carried into the panel and the file
      if (g.licence && (g.licence.name || g.licence.attribution)) {
        d.licence = { name: g.licence.name || null, attribution: g.licence.attribution || null };
      }
      if (/reanalysis/i.test(String(g.notes || ""))) d.reanalysis = true;
      // levels the registry names in hPa (ERA5): the picker's unit, and one
      // mid-troposphere level to start on rather than the stratosphere
      if (Array.isArray(g.levels_hpa) && g.levels_hpa.length) {
        d.levelUnitHint = "hPa";
        d.defaultLevel = g.levels_hpa.indexOf(500) >= 0 ? 500 : g.levels_hpa[Math.floor(g.levels_hpa.length / 2)];
        var vl = VAR_LABELS[fam.family + "/" + g.name];
        if (vl) d.channels.forEach(function (c) {
          var m = /_(\d+)$/.exec(c.name);
          if (!c.label && m) c.label = vl + " at " + m[1] + " hPa";
        });
      }
      hide(d, "_base", base + rel + "/");
      hide(d, "_raw", g);
      if (g.tier === "G") {
        var sg = g.store_groups || {};
        var names = Object.keys(sg).sort();
        if (!names.length) return;            // built but nothing to read
        d.groups = names.map(function (gn) { var o = { name: gn, grid: null }; hide(o, "_sg", sg[gn]); return o; });
        d.binFirst = Math.min.apply(null, names.map(function (gn) { return sg[gn].bin_first; }));
        d.binLast = Math.max.apply(null, names.map(function (gn) { return sg[gn].bin_last; }));
        jobs.push(Promise.all(d.groups.map(function (gg) {
          return tileGrid(d, gg.name, ctx).then(function (tg) { gg.grid = gridOf(tg, gg.name); });
        })).then(function () { d.grid = d.groups[0].grid; return d; }));
      } else {
        d.binFirst = g.bin_first;
        d.binLast = g.bin_last;
        jobs.push(Promise.resolve(d));
      }
      if (d.binFirst != null && d.binLast != null) d.span = [binStartIso(d.binFirst), binEndIso(d.binLast)];
      else if (g.record_span) d.span = g.record_span.slice();
    });
    await settleStores(fam, jobs, out);
    return reg;
  }

  // one failing store is a named error line, never the whole family's
  async function settleStores(fam, jobs, out) {
    var res = await Promise.all(jobs.map(function (p) {
      return p.then(function (d) { return { d: d }; }, function (e) { return { e: e }; });
    }));
    res.forEach(function (r) {
      if (r.d) { annotateLevels(r.d); out.stores.push(r.d); }
      else out.errors.push({ family: fam.family, title: fam.title, store: r.e && r.e.store || null,
        message: r.e && r.e.message ? r.e.message : String(r.e) });
    });
  }

  function hubRootOf(fam, url) {
    if (fam.root) return /^[a-z]+:\/\//i.test(fam.root) ? fam.root : siteUrl(fam.root);
    var m = /^(.*\/resolve\/main\/)/.exec(url);
    return m ? m[1] : dirOf(url);
  }

  // the five-day bins as dates: t82 seconds → {y, m, d}
  function frameOfBin(b, i) {
    var t = b * BIN_S, q = ymh82(t);
    return { i: i, t: t, y: q.y, m: q.m, d: q.d };
  }

  // ---- family 10: the four family-7.2 tensor groups (one bin-major .npy
  // each, z-scored) and the tier-P point stores (family 10.1 / 10.2, and
  // family 8's schema-1 Argo store)
  async function loadFamily10(fam, ctx, out) {
    var url = /^[a-z]+:\/\//i.test(fam.url) ? fam.url : siteUrl(fam.url);
    var reg = await readJSON(url, ctx);
    var root = hubRootOf(fam, url);
    var norms = null, normErr = null;
    if (fam.norms) {
      try { norms = await readSiteJSON(siteUrl(fam.norms), ctx); }
      catch (e) { normErr = e; }
    }
    var groups = Array.isArray(reg.groups) ? reg.groups : Object.values(reg.groups || {});
    var jobs = [];
    groups.forEach(function (g) {
      if (!g || (g.tier !== "G" && g.tier !== "P")) return;
      var prefix = String(g.prefix || g.path || "").replace(/\/+$/, "");
      var chans = (g.channels || []).map(function (c) {
        var cv = conversionOf(c.unit);
        return { name: c.name, unit: cv.unit, min: c.min == null ? null : c.min, max: c.max == null ? null : c.max,
          storedUnit: c.unit, note: cv.note, label: c.label || null };
      });
      if (g.tier === "P") {
        var d = baseDesc(fam, g.name, {
          kind: "points", layout: "points", channels: chans, folderUrl: treeUrl(root + prefix),
          cadence: g.cadence || null, subDaily: true, N: g.N == null ? null : g.N,
          binFirst: g.bin_first, binLast: g.bin_last != null ? g.bin_last : g.bin_first + (g.n_bins || 1) - 1,
          schema: g.schema_version || null
        });
        // the record the store states, else the bins it indexes (family 8's
        // index runs from the epoch whatever the floats did)
        if (Array.isArray(g.date_range) && g.date_range.length === 2) d.span = g.date_range.slice();
        else if (d.binFirst != null && d.binLast != null) d.span = [binStartIso(d.binFirst), binEndIso(d.binLast)];
        hide(d, "_base", root + prefix + "/");
        hide(d, "_raw", g);
        jobs.push(Promise.resolve(d));
        return;
      }
      // tier G: one bin-major file per group
      jobs.push((async function () {
        var f = (g.files || [])[0];
        if (!f || !f.name) { var e0 = new Error("group " + g.name + " lists no file"); e0.store = g.name; throw e0; }
        if (!norms) {
          var e1 = new Error("the z-score table for " + g.name + " (" + fam.norms + ") could not be read: " +
            (normErr && normErr.message ? normErr.message : "not configured"));
          e1.store = g.name;
          throw e1;
        }
        var ng = (norms.groups || {})[g.name];
        if (!ng || !Array.isArray(ng.norm) || ng.norm.length !== g.C) {
          var e2 = new Error("no (mean, sd) for every channel of " + g.name + " in " + fam.norms); e2.store = g.name; throw e2;
        }
        var gr = g.grid || ng.grid, step = Number(gr.step);
        var dd = baseDesc(fam, g.name, {
          kind: "grid", layout: "binmajor", channels: chans, folderUrl: treeUrl(root + prefix),
          cadence: g.cadence || null, subDaily: false,
          frameSeconds: g.cadence === "monthly" ? null : BIN_S
        });
        if (!dd.cadenceLabel) dd.cadenceLabel = g.cadence === "monthly" ? "monthly" : "five-day means";
        dd.channels.forEach(function (c, k) {
          if (ng.units && ng.units[c.name] && !c.unit) c.unit = ng.units[c.name];
          if (ng.labels && ng.labels[c.name] && !c.label) c.label = ng.labels[c.name];
          var mu = Number(ng.norm[k][0]), sd = Number(ng.norm[k][1]);
          c.norm = [mu, sd];
          if (c.min == null) { c.min = mu - 2.5 * sd; c.max = mu + 2.5 * sd; c.rangeFromNorm = true; }
        });
        var frames = [];
        if (g.live_bins_only || ng.live_only === true || ng.live_only === "True") {
          var bi = ng.bin_index || [];
          if (bi.length !== g.n_bins) { var e3 = new Error(g.name + ": bin_index has " + bi.length + " entries, the group " + g.n_bins + " frames"); e3.store = g.name; throw e3; }
          bi.forEach(function (b, i) {
            // a monthly frame is filed under the pentad holding its month's
            // 15th: the frame IS that calendar month
            var q = ymh82(b * BIN_S + 2.5 * 86400);
            frames.push({ i: i, t: sec82OfCivil(q.y, q.m, 1), y: q.y, m: q.m, d: null, monthly: true });
          });
        } else {
          for (var i = 0; i < g.n_bins; i++) frames.push(frameOfBin(g.bin_first + i, i));
        }
        hide(dd, "_dense", {
          url: root + prefix + "/" + f.name, order: "HWC", H: gr.ny, W: gr.nx, C: g.C,
          dtype: String(g.dtype || ng.dtype).replace(/^[<|=]/, ""), headerLen: g.header_len, slabBytes: g.slab_bytes,
          shape: g.shape, lat0: gr.lat0, lon0: gr.lon0, step: step, southFirst: gr.south_first !== false,
          frames: frames, zscored: true, normFrom: fam.norms
        });
        setDenseMeta(dd);
        hide(dd, "_raw", g);
        return dd;
      })().catch(function (e) { if (!e.store) e.store = g.name; throw e; }));
    });
    await settleStores(fam, jobs, out);
    return reg;
  }

  // the derived grids: the fishing-effort month-major grid and the model
  // climatology, both addressed from the site's own indexes
  async function loadDerived(fam, ctx, out) {
    var jobs = [];
    if (fam.fishing) jobs.push((async function () {
      var ixUrl = siteUrl(fam.fishing);
      var ix = await readSiteJSON(ixUrl, ctx);
      var gr = ix.grid, months = ix.months || [];
      var furl = new URL(ix.url, ixUrl).href;          // absolute on the site; relative in a fixture
      var d = baseDesc(fam, "fishing_grid", {
        kind: "grid", layout: "monthmajor", folderUrl: treeUrl(dirOf(furl)),
        channels: (ix.chans || []).map(function (c) {
          return { name: c, unit: (ix.units || {})[c] || "", label: (ix.labels || {})[c] || null, min: 0, max: null };
        })
      });
      if (!d.cadenceLabel) d.cadenceLabel = "monthly sums";
      var frames = months.map(function (ym, i) {
        var y = Number(ym.slice(0, 4)), m = Number(ym.slice(5, 7));
        return { i: i, t: sec82OfCivil(y, m, 1), y: y, m: m, d: null, monthly: true };
      });
      hide(d, "_dense", {
        url: furl, order: "HWC", H: gr.ny, W: gr.nx, C: ix.chans.length,
        dtype: String(ix.dtype).replace(/^[<|=]/, ""), headerLen: ix.header_len, slabBytes: ix.slab_bytes,
        shape: ix.shape, lat0: gr.lat0, lon0: gr.lon0, step: Number(gr.step), southFirst: gr.south_first !== false,
        frames: frames, zscored: false, sums: true
      });
      setDenseMeta(d);
      return d;
    })().catch(function (e) { if (!e.store) e.store = "fishing_grid"; throw e; }));
    if (fam.monthly) {
      // E-086: per group, sum.npy (float32 z-units) and count.npy (uint8),
      // both [month, channel, year, lat, lon] — the normal over ANY set of
      // years is Σ sums / Σ counts, composed here
      var mx = null, mxErr = null, mxUrl = null;
      try { mxUrl = siteUrl(fam.monthly); mx = await readSiteJSON(mxUrl, ctx); } catch (e) { mxErr = e; }
      if (!mx) {
        out.errors.push({ family: fam.family, title: fam.title, store: "normals",
          message: "the monthly-normals index (" + fam.monthly + ") could not be read: " + (mxErr && mxErr.message) });
      } else {
        Object.keys(mx.groups || {}).forEach(function (gname) {
          jobs.push((async function () {
            var G = mx.groups[gname], gr = G.grid;
            var file = function (f) {
              if (!f || !f.url) { var e0 = new Error(gname + ": no " + (f ? f.url : "file") + " in the index"); throw e0; }
              return { url: new URL(f.url, mxUrl).href, hdr: Number(f.header_len), isz: Number(f.itemsize),
                plane: Number(f.plane_bytes), shape: f.shape, dtype: String(f.dtype).replace(/^[<|=]/, "") };
            };
            var sum = file(G.sum), cnt = file(G.count);
            var C = G.chans.length, Y = Number(G.n_years), H = gr.ny, W = gr.nx;
            var want = [12, C, Y, H, W];
            [sum, cnt].forEach(function (f) {
              if (!Array.isArray(f.shape) || f.shape.join() !== want.join()) throw new Error(gname + ": shape [" + f.shape + "], expected [" + want + "] — " + f.url);
              if (f.plane !== H * W * f.isz) throw new Error(gname + ": plane_bytes " + f.plane + " ≠ H·W·itemsize — " + f.url);
            });
            if (sum.dtype !== "f4" || cnt.dtype !== "u1") throw new Error(gname + ": sum must be float32 and count uint8 (" + sum.dtype + ", " + cnt.dtype + ")");
            if (!Array.isArray(G.norm) || G.norm.length !== C) throw new Error(gname + ": no (mean, sd) for every channel");
            var d = baseDesc(fam, "normals_" + gname, {
              kind: "grid", layout: "monthly", normals: true, folderUrl: treeUrl(dirOf(sum.url)),
              channels: G.chans.map(function (c, k) {
                var mu = Number(G.norm[k][0]), sd = Number(G.norm[k][1]);
                return { name: c, unit: (G.units || {})[c] || "", label: (G.labels || {})[c] || null,
                  norm: [mu, sd], min: mu - 2.5 * sd, max: mu + 2.5 * sd, rangeFromNorm: true };
              })
            });
            if (!d.cadenceLabel) d.cadenceLabel = "monthly means over the years you choose";
            d.years = [Number(G.year_first), Number(G.year_last)];
            d.steps = ["normal", "by-year"];
            d.defaultChannel = G.chans.indexOf("sst") >= 0 ? "sst" : G.chans[0];
            d.maxBinsPerMonth = Number(G.max_count) || 7;
            hide(d, "_monthly", { sum: sum, count: cnt, C: C, Y: Y, H: H, W: W, year0: d.years[0],
              lat0: gr.lat0, lon0: gr.lon0, step: Number(gr.step), southFirst: gr.south_first !== false,
              rowKind: G.row_kind || "pentad", index: mxUrl, combine: mx.combine || null, monthRule: mx.month_rule || null });
            var step = Number(gr.step), dlat = gr.south_first !== false ? step : -step;
            d.grid = { H: H, W: W, lat0: gr.lat0, lon0: gr.lon0, dlat: dlat, dlon: step, step: step };
            d.span = [d.years[0] + "-01-01", d.years[1] + "-12-31"];
            return d;
          })().catch(function (e) { if (!e.store) e.store = "normals_" + gname; throw e; }));
        });
      }
    }
    if (fam.clim) {
      var cx = null, cxErr = null, cxUrl = null;
      try { cxUrl = siteUrl(fam.clim); cx = await readSiteJSON(cxUrl, ctx); } catch (e) { cxErr = e; }
      if (!cx) {
        out.errors.push({ family: fam.family, title: fam.title, store: "clim",
          message: "the model climatology index (" + fam.clim + ") could not be read: " + (cxErr && cxErr.message) });
      } else {
        var ver = fam.climVersion || cx.default_version || "all";
        var files = (cx.files || {})[ver] || {};
        Object.keys(files).sort().forEach(function (gname) {
          jobs.push((async function () {
            var fl = files[gname].clim_npy, cg = (cx.groups || {})[gname];
            if (!fl || !cg) { var e0 = new Error("no clim.npy for " + gname); e0.store = "clim_" + gname; throw e0; }
            var gr = cg.grid, curl = new URL(fl.url, cxUrl).href;
            var d = baseDesc(fam, "clim_" + gname, {
              kind: "grid", layout: "clim", calendar: true, folderUrl: treeUrl(dirOf(curl)),
              channels: cg.chans.map(function (c, k) {
                var mu = Number(cg.norm[k][0]), sd = Number(cg.norm[k][1]);
                return { name: c, unit: (cg.units || {})[c] || "", label: (cg.labels || {})[c] || null,
                  norm: [mu, sd], min: mu - 2.5 * sd, max: mu + 2.5 * sd, rangeFromNorm: true };
              })
            });
            if (!d.cadenceLabel) d.cadenceLabel = "one map per calendar month (a climatology — no years)";
            var frames = [];
            for (var m = 1; m <= 12; m++) frames.push({ i: m - 1, t: sec82OfCivil(2000, m, 1), y: null, m: m, d: null, monthly: true });
            hide(d, "_dense", {
              url: curl, order: "MCHW", H: gr.ny, W: gr.nx, C: cg.chans.length,
              dtype: String(fl.dtype).replace(/^[<|=]/, ""), headerLen: fl.header_len, planeBytes: fl.plane_bytes,
              lat0: gr.lat0, lon0: gr.lon0, step: Number(gr.step), southFirst: gr.south_first !== false,
              frames: frames, zscored: true, normFrom: fam.clim, version: ver
            });
            setDenseMeta(d);
            return d;
          })().catch(function (e) { if (!e.store) e.store = "clim_" + gname; throw e; }));
        });
      }
    }
    await settleStores(fam, jobs, out);
  }

  // what the tab reads about a dense grid: its geometry and record
  function setDenseMeta(d) {
    var G = d._dense;
    var lat0 = G.southFirst ? G.lat0 : G.lat0, dlat = G.southFirst ? G.step : -G.step;
    d.grid = { H: G.H, W: G.W, lat0: lat0, lon0: G.lon0, dlat: dlat, dlon: G.step, step: G.step };
    if (G.frames.length && !d.calendar) {
      var f0 = G.frames[0], f1 = G.frames[G.frames.length - 1];
      var endIso = f1.monthly
        ? isoOfUnix(EPOCH_UNIX + (f1.m === 12 ? sec82OfCivil(f1.y + 1, 1, 1) : sec82OfCivil(f1.y, f1.m + 1, 1)) - 86400).slice(0, 10)
        : binEndIso(Math.floor(f1.t / BIN_S));
      d.span = [isoDate82(f0.t), endIso];
    }
  }

  function loadRegistry(ctx) {
    return cached("registry", async function () {
      var out = { stores: [], errors: [], missing: [], families: [], coming: [] };
      var raws = {};
      await Promise.all(cfg.registries.map(async function (fam) {
        out.families.push({ family: fam.family, title: fam.title });
        try {
          if (fam.kind === "family1") raws[fam.family] = await loadFamily1(fam, ctx, out);
          else if (fam.kind === "family10") raws[fam.family] = await loadFamily10(fam, ctx, out);
          else if (fam.kind === "derived") await loadDerived(fam, ctx, out);
          else throw new Error("unknown registry kind " + JSON.stringify(fam.kind));
        } catch (e) {
          var msg = e && e.message ? e.message : String(e);
          if (fam.optional && /HTTP 404/.test(msg)) out.missing.push({ family: fam.family, title: fam.title, url: fam.url });
          else out.errors.push({ family: fam.family, title: fam.title, store: null, message: msg });
        }
      }));
      // registry order, then each family's own order
      var order = cfg.registries.map(function (f) { return f.family; });
      out.families.sort(function (a, b) { return order.indexOf(a.family) - order.indexOf(b.family); });
      var seq = new Map(out.stores.map(function (d, i) { return [d, i]; }));
      out.stores.sort(function (a, b) {
        return (order.indexOf(a.family) - order.indexOf(b.family)) || (seq.get(a) - seq.get(b));
      });
      out.families = out.families.filter(function (f) { return out.stores.some(function (d) { return d.family === f.family; }); });
      out.coming.sort(function (a, b) { return order.indexOf(a.family) - order.indexOf(b.family); });
      var byName = {};
      out.stores.forEach(function (d) {
        byName[d.id] = d;
        byName[d.name] = byName[d.name] && byName[d.name] !== d ? AMBIGUOUS : d;
      });
      out.generated = null;
      out.base = cfg.base;
      Object.defineProperty(out, "_byName", { value: byName, enumerable: false });
      // the raw registry JSONs, by family (`_raw` = family 1.gf's, as before)
      Object.defineProperty(out, "_raws", { value: raws, enumerable: false });
      Object.defineProperty(out, "_raw", { value: raws["1.gf"] || raws[Object.keys(raws)[0]] || null, enumerable: false });
      if (!out.stores.length && out.errors.length) {
        throw new Error("no store could be loaded: " + out.errors.map(function (e) { return e.family + (e.store ? "/" + e.store : "") + ": " + e.message; }).join("; "));
      }
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
    return d._base + prefix + "/";
  }

  function tileGrid(d, group, ctx) {
    var gg = (d.groups || []).find(function (x) { return x.name === group; });
    var rel = gg && gg._sg && gg._sg.tile_grid ? gg._sg.tile_grid : group + "/tile_grid.json";
    var url = d._base + rel;
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
    var url = d._base + rel;
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

  // The record a sharded store REALLY holds, from its shard index: the first
  // and last frame present. At load time only the bins are known, and a last
  // bin can be partly filled (ERA5's ends 2026-06-30 inside a bin that runs to
  // 07-03), so the span is tightened the first time the index is read.
  function exactSpan(d, si, F, fs) {
    if (d.spanExact) return;
    var bins = Array.from(si.rows.keys()).sort(function (a, b) { return a - b; });
    var first = null, last = null;
    for (var i = 0; i < bins.length && first === null; i++) {
      for (var f = 0; f < F; f++) if (frameBit(si.rows.get(bins[i]), f)) { first = bins[i] * BIN_S + f * fs; break; }
    }
    for (var j = bins.length - 1; j >= 0 && last === null; j--) {
      for (var f2 = F - 1; f2 >= 0; f2--) if (frameBit(si.rows.get(bins[j]), f2)) { last = bins[j] * BIN_S + f2 * fs; break; }
    }
    if (first === null || last === null) return;
    d.span = [isoDate82(first), isoDate82(last + fs - 1)];
    d.spanExact = true;
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
    if (d.normals) {
      // a normals store composes over YEARS: one mean per calendar month over
      // the period (the climatology) or one per year and month (the stack)
      if (s.step === "native" || s.step === "month") s.step = "by-year";
      if (s.step === "all") s.step = "normal";
      if (["normal", "by-year"].indexOf(s.step) < 0) throw new Error("step must be normal or by-year for the normals store " + d.name);
    } else if (["native", "pentad", "month", "all"].indexOf(s.step) < 0) throw new Error("step must be native, pentad, month or all");
    // + excludeYears: whole calendar years left out of the composition
    // (the paper's split is 1982–2020 excluding 2009 and 2017)
    s.excludeYears = (sel.excludeYears || []).map(Number);
    s.excludeYears.forEach(function (y) { if (!Number.isInteger(y)) throw new Error("excludeYears must be whole years: " + y); });
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
    var d = await storeDesc(sel, ctx);
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
    exactSpan(d, si, F, fs);
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
    if (plan.d.reanalysis) n.push("a REANALYSIS — a weather model's analysis constrained by observations (ERA5 at ~31 km, averaged here onto a 1° grid), not an observation; each frame is an instant (00, 06, 12 or 18 UTC), not a six-hour mean");
    plan.conv.forEach(function (c, k) { if (c.note) n.push(plan.s.channels[k] + ": " + c.note); });
    if (plan.s.bbox && plan.s.bbox.w > plan.s.bbox.e) n.push("the box crosses the dateline: longitudes run past 180° (subtract 360 for −180..180)");
    if (plan.mean) n.push("each value is the mean of the finite observations in its cell and time step; the count arrays say how many");
    return n;
  }

  // ======================================================== DENSE GRIDS ===
  // Grids stored as ONE uncompressed .npy whose frames are contiguous slabs:
  //   binmajor   [T, H, W, C] float16, z-scored — family 7.2's tensor groups
  //              (family 10's g025, g100, oc025, rg100); one five-day bin (or,
  //              for rg100, one month) per slab;
  //   monthmajor [M, H, W, C] float32, raw — the fishing-effort grid;
  //   clim       [12, C, H, W] float32, z-scored — the model climatology, one
  //              calendar month per slab, one channel per plane.
  // A box is a band of ROWS: within a frame the rows are contiguous, so one
  // frame is one range per band (split at MAX_RANGE), and frames whose band
  // is the whole height run on into each other and are coalesced. Columns
  // are cut after reading, and in the H-W-C layouts every channel of those
  // rows is read whatever the channels asked for — the estimate counts it.
  async function denseSpec(d, ctx) {
    var G = d._dense;
    return cached("densespec:" + G.url, async function () {
      var h = await npyHeaderAt(G.url, ctx);
      var want = G.order === "MCHW" ? [G.frames.length, G.C, G.H, G.W] : [G.frames.length, G.H, G.W, G.C];
      if (h.descr !== G.dtype) throw new Error(d.name + ": the .npy is " + h.descr + ", the index says " + G.dtype + " — " + G.url);
      if (h.shape.length !== 4 || h.shape.some(function (v, k) { return v !== want[k]; })) {
        throw new Error(d.name + ": the .npy shape is [" + h.shape.join(", ") + "], expected [" + want.join(", ") + "] — " + G.url);
      }
      if (G.headerLen != null && Number(G.headerLen) !== h.dataOffset) {
        throw new Error(d.name + ": the .npy header is " + h.dataOffset + " bytes, the index says " + G.headerLen + " — " + G.url);
      }
      var isz = h.itemsize, rowBytes = G.order === "MCHW" ? G.W * isz : G.W * G.C * isz;
      var slab = G.order === "MCHW" ? G.C * G.H * G.W * isz : G.H * rowBytes;
      if (G.slabBytes != null && G.order !== "MCHW" && Number(G.slabBytes) !== slab) {
        throw new Error(d.name + ": a frame is " + slab + " bytes by the header, " + G.slabBytes + " by the index — " + G.url);
      }
      if (G.planeBytes != null && G.order === "MCHW" && Number(G.planeBytes) !== G.H * G.W * isz) {
        throw new Error(d.name + ": a plane is " + (G.H * G.W * isz) + " bytes by the header, " + G.planeBytes + " by the index — " + G.url);
      }
      return { hdr: h.dataOffset, isz: isz, rowBytes: rowBytes, slab: slab, plane: G.H * G.W * isz };
    });
  }

  function denseTg(G) {
    // boxGeometry's pseudo tile grid: one "tile" covering the whole array
    var dy = G.southFirst ? G.step : -G.step;
    return { H: G.H, W: G.W, tile: Math.max(G.H, G.W),
      grid: { x0: G.lon0 - G.step / 2, y0: G.lat0 - dy / 2, dx: G.step, dy: dy } };
  }

  async function densePlan(d, sel, ctx, opts) {
    opts = opts || {};
    var G = d._dense;
    var s = normSel(sel, d);
    if (!s.bbox) {
      var e = new Error("a box (bbox) is required for the gridded store " + d.name);
      e.needBox = true;
      throw e;
    }
    var sp = await denseSpec(d, ctx);
    var tg = denseTg(G);
    var geo = boxGeometry(tg, s.bbox, s.res);
    var rowsT = geo.byTy.get(0) || [], rowOut = new Int32Array(G.H).fill(-1), rmin = Infinity, rmax = -1;
    for (var a = 0; a < rowsT.length; a += 2) {
      rowOut[rowsT[a]] = rowsT[a + 1];
      if (rowsT[a] < rmin) rmin = rowsT[a];
      if (rowsT[a] > rmax) rmax = rowsT[a];
    }
    var colsT = geo.byTx.get(0) || [];
    var allC = d.channels.map(function (c) { return c.name; });
    var chIdx = s.channels.map(function (c) { return allC.indexOf(c); });
    var monthOk = {};
    s.months.forEach(function (m) { monthOk[m] = true; });
    var frames = G.frames.filter(function (fr) {
      if (!monthOk[fr.m]) return false;
      if (d.calendar) return true;
      if (fr.y < s.yearStart || fr.y > s.yearEnd) return false;
      if (s.days && !fr.monthly && (fr.d < s.days[0] || fr.d > s.days[1])) return false;
      return true;
    });
    if (opts.onlyFrame) frames = [opts.onlyFrame];
    var step = s.step === "pentad" && frames.length && frames[0].monthly ? "native" : s.step;
    var stepKey = function (fr) {
      if (step === "native" || step === "pentad") return fr.i;
      if (step === "month") return d.calendar ? fr.m : fr.y * 12 + fr.m - 1;
      return 0;
    };
    var keys = [], keyIdx = new Map(), perStep = new Map();
    frames = frames.map(function (fr) {
      var k = stepKey(fr);
      if (!keyIdx.has(k)) { keyIdx.set(k, keys.length); keys.push(fr); }
      perStep.set(k, (perStep.get(k) || 0) + 1);
      return Object.assign({}, fr, { si: keyIdx.get(k) });
    });
    var times = new Float64Array(keys.length);
    keys.forEach(function (fr, i) {
      if (step === "month" && !d.calendar) times[i] = daysFromCivil(fr.y, fr.m, 1) * 86400;
      else times[i] = EPOCH_UNIX + fr.t;
    });
    var mean = !(step === "native" && s.res === "native");
    var maxFrames = Math.max.apply(null, [1].concat(Array.from(perStep.values())));
    var maxCount = (step === "native" ? 1 : maxFrames) * geo.maxPerCell;
    var Cs = s.channels.length;
    var countBytes = mean ? (maxCount > 65535 ? 4 : 2) : 0;
    var outBytes = keys.length * Cs * geo.Ho * geo.Wo * (4 + countBytes) + 8 * (keys.length + geo.Ho + geo.Wo);
    // the byte segments: one per (frame[, channel]) over rows rmin..rmax
    var segs = [];
    if (rmax >= 0 && geo.nCols > 0) {
      frames.forEach(function (fr) {
        var parts = G.order === "MCHW"
          ? chIdx.map(function (c) { return { c: c, base: sp.hdr + (fr.i * G.C + c) * sp.plane }; })
          : [{ c: null, base: sp.hdr + fr.i * sp.slab }];
        parts.forEach(function (pt) {
          // split the band into pieces of at most MAX_RANGE
          var per = Math.max(1, Math.floor(MAX_RANGE / sp.rowBytes));
          for (var r0 = rmin; r0 <= rmax; r0 += per) {
            var r1 = Math.min(rmax, r0 + per - 1);
            segs.push({ fr: fr, c: pt.c, r0: r0, r1: r1, a: pt.base + r0 * sp.rowBytes, z: pt.base + (r1 + 1) * sp.rowBytes });
          }
        });
      });
    }
    // coalesce byte-contiguous segments into requests of at most MAX_RANGE
    var reqs = [];
    segs.forEach(function (sg) {
      var last = reqs[reqs.length - 1];
      if (last && last.z === sg.a && sg.z - last.a <= MAX_RANGE) { last.z = sg.z; last.segs.push(sg); }
      else reqs.push({ a: sg.a, z: sg.z, segs: [sg] });
    });
    var readBytes = reqs.reduce(function (t, q) { return t + (q.z - q.a); }, 0);
    var conv = s.channels.map(function () { return { unit: null, offset: 0, note: null }; });
    s.channels.forEach(function (c, k) { conv[k].unit = d.channels[chIdx[k]].unit; });
    return {
      dense: true, d: d, s: s, G: G, sp: sp, geo: geo, rowOut: rowOut, colsT: colsT, rmin: rmin, rmax: rmax,
      frames: frames, times: times, T: keys.length, Cs: Cs, chIdx: chIdx, conv: conv, mean: mean, step: step,
      countBytes: countBytes, outBytes: outBytes, maxCount: maxCount, reqs: reqs, readBytes: readBytes,
      group: d.name, source: G.url
    };
  }

  async function denseRun(plan, ctx, onProgress) {
    var G = plan.G, sp = plan.sp, geo = plan.geo, Cs = plan.Cs, C = G.C;
    var HW = geo.Ho * geo.Wo, nOut = plan.T * Cs * HW;
    var data = new Float32Array(nOut);
    var count = null;
    if (plan.mean) count = plan.countBytes === 4 ? new Uint32Array(nOut) : new Uint16Array(nOut);
    else data.fill(NaN);
    var f16 = sp.isz === 2, colsT = plan.colsT, rowOut = plan.rowOut, W = G.W;
    var norm = plan.chIdx.map(function (ci) { var c = plan.d.channels[ci]; return G.zscored ? c.norm : [0, 1]; });
    var progress = { done: 0, total: plan.reqs.length, bytes: 0 };
    var tell = function () { if (onProgress) { try { onProgress({ done: progress.done, total: progress.total, bytes: progress.bytes }); } catch (e) { /* the caller's */ } } };
    var rctx = Object.assign({}, ctx, { onRead: function (n) { progress.done++; progress.bytes += n; tell(); } });
    tell();
    var f32 = null;
    await pool(plan.reqs, async function (q) {
      var buf = await rangeRead(G.url, q.a, q.z - q.a, rctx);
      var dv = f16 ? null : new DataView(buf.buffer, buf.byteOffset, buf.byteLength);
      q.segs.forEach(function (sg) {
        var off0 = sg.a - q.a, fr = sg.fr;
        for (var r = sg.r0; r <= sg.r1; r++) {
          var orow = rowOut[r];
          if (orow < 0) continue;
          var rb = off0 + (r - sg.r0) * sp.rowBytes;
          for (var k = 0; k < Cs; k++) {
            var ch = plan.chIdx[k];
            if (sg.c != null && sg.c !== ch) continue;
            var mu = norm[k][0], sd = norm[k][1];
            var base = (fr.si * Cs + k) * HW + orow * geo.Wo;
            for (var cc = 0; cc < colsT.length; cc += 2) {
              var col = colsT[cc];
              var e = sg.c != null ? col : col * C + ch;
              var v;
              if (f16) { var p = rb + 2 * e; v = F16[buf[p] | (buf[p + 1] << 8)]; }
              else v = dv.getFloat32(rb + 4 * e, true);
              v = v * sd + mu;
              var oi = base + colsT[cc + 1];
              if (!plan.mean) data[oi] = v;
              else if (v === v) { data[oi] += v; count[oi]++; }
            }
          }
        }
      });
    }, ctx);
    if (plan.mean) {
      for (var i = 0; i < nOut; i++) data[i] = count[i] ? data[i] / count[i] : NaN;
    }
    return { data: data, count: count };
  }

  function denseNotes(plan) {
    var n = [], d = plan.d, G = plan.G;
    if (G.zscored) n.push("stored z-scored; returned in physical units as value = z × sd + mean, with each channel's (mean, sd) from " + (G.normFrom || "the store's index"));
    d.channels.forEach(function (c) { if (plan.s.channels.indexOf(c.name) >= 0 && /log/i.test(c.name + " " + (c.unit || ""))) n.push(c.name + ": kept as the logarithm the unit names, not converted back"); });
    if (G.order !== "MCHW" && G.C > plan.Cs) n.push("the store keeps its " + G.C + " channels side by side, so reading " + plan.Cs + " of them read all " + G.C + " for the box's rows");
    if (d.calendar) n.push("a calendar-month climatology: each step is a calendar month averaged over the record (" + (G.version || "all") + " version); the year in the time axis is nominal (2000) and means nothing");
    if (G.frames.length && G.frames[0].monthly && !d.calendar) n.push("one frame per calendar month; the days filter does not apply");
    if (G.sums) n.push("each value is a monthly SUM of vessel-hours in its 0.25° cell; a coarser cell or a longer time step AVERAGES those sums (mean per 0.25° cell and month), it does not add them up; zero is a measurement (no vessel broadcast there), and absence of effort is not absence of fishing");
    if (d.levels) n.push("levelled channels are written one variable per channel and level; the name carries the level (" + d.levelUnit + ")");
    if (plan.s.bbox && plan.s.bbox.w > plan.s.bbox.e) n.push("the box crosses the dateline: longitudes run past 180° (subtract 360 for −180..180)");
    if (plan.mean) n.push("each value is the mean of the finite values in its cell and time step; the count arrays say how many");
    return n;
  }

  function denseEstimate(plan) {
    var nf = plan.frames.length;
    var what = plan.d.calendar ? "calendar month" : (plan.G.frames[0] && plan.G.frames[0].monthly ? "monthly frame" : "five-day frame");
    var wh = nf + " " + what + (nf === 1 ? "" : "s") + ": " + plan.reqs.length + " requests, " + fmtMB(plan.readBytes) +
      " to read (exact: the rows of the box";
    if (plan.G.order !== "MCHW" && plan.G.C > plan.Cs) wh += ", every one of the store's " + plan.G.C + " channels — they are stored side by side";
    wh += "); the result is " + plan.T + " × " + plan.Cs + " × " + plan.geo.Ho + " × " + plan.geo.Wo + " (" + fmtMB(plan.outBytes) + ")." +
      (plan.mean ? " Coarser steps or cells shrink the file, not the read." : "");
    return { requests: plan.reqs.length, readBytes: plan.readBytes, outBytes: plan.outBytes, frames: nf, exact: true,
      shape: [plan.T, plan.Cs, plan.geo.Ho, plan.geo.Wo], why: wh,
      channelsRead: plan.G.order !== "MCHW" ? plan.G.C : plan.Cs, channelsKept: plan.Cs, channelWord: plan.d.levels ? "levels" : "channels",
      shrink: plan.d.calendar ? "Pick fewer months or channels, or shrink the box." : "Shorten the period, pick fewer months, or shrink the box." };
  }

  function denseResult(plan, g, ctx, t0, empty) {
    return {
      kind: "grid", store: plan.d.name, family: plan.d.family, channels: plan.s.channels.slice(),
      units: plan.conv.map(function (c) { return c.unit; }),
      lat: plan.geo.outLat, lon: plan.geo.outLon, time: empty ? new Float64Array(0) : plan.times,
      data: g.data, count: g.count, sel: plan.s, notes: denseNotes(plan), frames: empty ? 0 : plan.frames.length,
      group: plan.d.name, source: plan.G.url, title: plan.d.title,
      levels: plan.d.levels ? plan.s.channels.map(function (c) { var x = plan.d.channels.find(function (y) { return y.name === c; }); return x && x.level != null ? x.level : null; }) : null,
      stats: { requests: ctx.stats.requests, bytes: ctx.stats.bytes, ms: Date.now() - t0 }
    };
  }

  // ============================================================ NORMALS ===
  // E-086's per-year monthly sums and counts: sum.npy (float32, z-units) and
  // count.npy (uint8), [month, channel, year, lat, lon]. A plane is one
  // (month, channel, year); the planes of a run of years for one (month,
  // channel) are contiguous, and a latitude band is contiguous within a
  // plane. The read is one byte span per (file, month, channel, year) over
  // the box's rows, coalesced when the gap between two spans is at most
  // NORMALS_MERGE_GAP: on a 0.25° grid the gap between two years' bands is
  // most of a 4 MB plane and every year is its own range; on a 1° grid the
  // whole 260 kB plane is cheaper than a request and a run of years becomes
  // one range (measured: a range request costs ~0.4–0.7 s of latency at
  // ~10 MB/s per stream, six streams in flight — ≈ 0.5 MB of transfer).
  var NORMALS_MERGE_GAP = 512 * 1024;

  function normalsYears(d, s) {
    var ex = {};
    (s.excludeYears || []).forEach(function (y) { ex[y] = true; });
    var out = [];
    for (var y = Math.max(s.yearStart, d.years[0]); y <= Math.min(s.yearEnd, d.years[1]); y++) if (!ex[y]) out.push(y);
    return out;
  }

  async function normalsSpec(d, ctx) {
    var M = d._monthly;
    return cached("normalsspec:" + M.sum.url, async function () {
      var hs = await npyHeaderAt(M.sum.url, ctx), hc = await npyHeaderAt(M.count.url, ctx);
      [[hs, M.sum, "f4"], [hc, M.count, "u1"]].forEach(function (x) {
        var h = x[0], f = x[1];
        if (h.descr !== x[2]) throw new Error(d.name + ": the .npy is " + h.descr + ", expected " + x[2] + " — " + f.url);
        if (h.shape.join() !== f.shape.join()) throw new Error(d.name + ": the .npy shape is [" + h.shape + "], the index says [" + f.shape + "] — " + f.url);
        if (h.dataOffset !== f.hdr) throw new Error(d.name + ": the .npy header is " + h.dataOffset + " bytes, the index says " + f.hdr + " — " + f.url);
      });
      return true;
    });
  }

  async function normalsPlan(d, sel, ctx, opts) {
    opts = opts || {};
    var M = d._monthly;
    var s = normSel(sel, d);
    if (!s.bbox) { var e = new Error("a box (bbox) is required for the gridded store " + d.name); e.needBox = true; throw e; }
    if (s.res !== "native" && !(s.res > M.step)) throw new Error(d.name + " is " + M.step + "°: no coarser resolution than " + s.res + "° is offered");
    await normalsSpec(d, ctx);
    var tg = denseTg({ H: M.H, W: M.W, lat0: M.lat0, lon0: M.lon0, step: M.step, southFirst: M.southFirst });
    var geo = boxGeometry(tg, s.bbox, s.res);
    var rowsT = geo.byTy.get(0) || [], rowOut = new Int32Array(M.H).fill(-1), rmin = Infinity, rmax = -1;
    for (var a = 0; a < rowsT.length; a += 2) {
      rowOut[rowsT[a]] = rowsT[a + 1];
      if (rowsT[a] < rmin) rmin = rowsT[a];
      if (rowsT[a] > rmax) rmax = rowsT[a];
    }
    var colsT = geo.byTx.get(0) || [];
    var allC = d.channels.map(function (c) { return c.name; });
    var chIdx = s.channels.map(function (c) { return allC.indexOf(c); });
    var years = normalsYears(d, s);
    var months = s.months.slice().sort(function (x, y) { return x - y; });
    if (opts.onlyMonth) months = [opts.onlyMonth];
    // steps: normal → one per month; by-year → one per (year, month), in time order
    var keys = [], keyOf = {};
    if (s.step === "normal") months.forEach(function (m, i) { keyOf["*:" + m] = i; keys.push({ m: m }); });
    else years.forEach(function (y) { months.forEach(function (m) { keyOf[y + ":" + m] = keys.length; keys.push({ y: y, m: m }); }); });
    var stepOf = function (y, m) { return s.step === "normal" ? keyOf["*:" + m] : keyOf[y + ":" + m]; };
    var times = new Float64Array(keys.length), bounds = null;
    if (s.step === "normal" && years.length) {
      bounds = new Float64Array(2 * keys.length);
      keys.forEach(function (k, i) {
        var y0 = years[0], y1 = years[years.length - 1];
        times[i] = daysFromCivil(y0, k.m, 1) * 86400;
        bounds[2 * i] = times[i];
        bounds[2 * i + 1] = (k.m === 12 ? daysFromCivil(y1 + 1, 1, 1) : daysFromCivil(y1, k.m + 1, 1)) * 86400;
      });
    } else keys.forEach(function (k, i) { times[i] = daysFromCivil(k.y, k.m, 1) * 86400; });
    // the byte spans: (file, month, channel, year) over rows rmin..rmax
    var reqs = [];
    if (rmax >= 0 && geo.nCols > 0 && years.length) {
      [["sum", M.sum], ["count", M.count]].forEach(function (ff) {
        var f = ff[1], rowB = M.W * f.isz, spans = [];
        months.forEach(function (m) {
          chIdx.forEach(function (c, k) {
            years.forEach(function (y) {
              var base = f.hdr + (((m - 1) * M.C + c) * M.Y + (y - M.year0)) * f.plane;
              var sp = [base + rmin * rowB, base + (rmax + 1) * rowB];
              sp.seg = { file: ff[0], k: k, si: stepOf(y, m), a: sp[0] };
              spans.push(sp);
            });
          });
        });
        coalesce(spans, cfg.normalsMergeGap != null ? cfg.normalsMergeGap : NORMALS_MERGE_GAP, MAX_RANGE).forEach(function (r) {
          reqs.push({ file: ff[0], url: f.url, isz: f.isz, a: r[0], z: r[1], segs: r.items.map(function (x) { return x.seg; }) });
        });
      });
    }
    var readBytes = reqs.reduce(function (t, q) { return t + (q.z - q.a); }, 0);
    var Cs = s.channels.length;
    var perStep = s.step === "normal" ? years.length : 1;
    var maxCount = d.maxBinsPerMonth * perStep * geo.maxPerCell;
    var countBytes = maxCount > 65535 ? 4 : 2;
    var outBytes = keys.length * Cs * geo.Ho * geo.Wo * (4 + countBytes) + 8 * (keys.length + geo.Ho + geo.Wo) + (bounds ? bounds.length * 8 : 0);
    var conv = chIdx.map(function (ci) { return { unit: d.channels[ci].unit, offset: 0, note: null }; });
    return {
      normals: true, d: d, s: s, M: M, geo: geo, rowOut: rowOut, colsT: colsT, rmin: rmin, rmax: rmax,
      years: years, months: months, keys: keys, times: times, bounds: bounds, T: keys.length, Cs: Cs, chIdx: chIdx,
      conv: conv, countBytes: countBytes, outBytes: outBytes, maxCount: maxCount, reqs: reqs, readBytes: readBytes,
      planes: years.length * months.length * Cs, mean: true, group: d.name, source: M.sum.url
    };
  }

  async function normalsRun(plan, ctx, onProgress) {
    var M = plan.M, geo = plan.geo, Cs = plan.Cs, HW = geo.Ho * geo.Wo, nOut = plan.T * Cs * HW;
    var S = new Float64Array(nOut), N = new Uint32Array(nOut);
    var colsT = plan.colsT, rowOut = plan.rowOut, W = M.W, rmin = plan.rmin, rmax = plan.rmax;
    var progress = { done: 0, total: plan.reqs.length, bytes: 0 };
    var tell = function () { if (onProgress) { try { onProgress({ done: progress.done, total: progress.total, bytes: progress.bytes }); } catch (e) { /* the caller's */ } } };
    var rctx = Object.assign({}, ctx, { onRead: function (n) { progress.done++; progress.bytes += n; tell(); } });
    tell();
    await pool(plan.reqs, async function (q) {
      var buf = await rangeRead(q.url, q.a, q.z - q.a, rctx);
      var dv = q.isz === 4 ? new DataView(buf.buffer, buf.byteOffset, buf.byteLength) : null;
      q.segs.forEach(function (sg) {
        var off0 = sg.a - q.a;
        for (var r = rmin; r <= rmax; r++) {
          var orow = rowOut[r];
          if (orow < 0) continue;
          var rb = off0 + (r - rmin) * W * q.isz;
          var base = (sg.si * Cs + sg.k) * HW + orow * geo.Wo;
          for (var cc = 0; cc < colsT.length; cc += 2) {
            var oi = base + colsT[cc + 1];
            if (dv) { var v = dv.getFloat32(rb + 4 * colsT[cc], true); if (v === v) S[oi] += v; }
            else N[oi] += buf[rb + colsT[cc]];
          }
        }
      });
    }, ctx);
    var data = new Float32Array(nOut);
    var count = plan.countBytes === 4 ? new Uint32Array(nOut) : new Uint16Array(nOut);
    var norm = plan.chIdx.map(function (ci) { return plan.d.channels[ci].norm; });
    for (var i = 0; i < nOut; i++) {
      var k = Math.floor(i / HW) % Cs;
      count[i] = N[i];
      data[i] = N[i] ? (S[i] / N[i]) * norm[k][1] + norm[k][0] : NaN;
    }
    return { data: data, count: count };
  }

  function normalsNotes(plan) {
    var n = [], s = plan.s, y = plan.years;
    n.push("composed from the global tensor's per-year monthly sums and counts (E-086): mean = Σ sums ÷ Σ counts over the chosen years, then value = z × sd + mean in the channel's unit; NaN where no year had a value");
    if (s.step === "normal") n.push("a CLIMATOLOGY: each time step is one calendar month averaged over " + (y.length ? y[0] + "–" + y[y.length - 1] : "no year") + " (" + y.length + " year" + (y.length === 1 ? "" : "s") + ")" + (s.excludeYears.length ? ", excluding " + s.excludeYears.join(", ") : "") + "; the time value is that month in the first year, and climatology_bounds gives the span");
    else n.push("one monthly mean per year and calendar month (the stack of the chosen months across the chosen years)");
    n.push("count = the number of " + (plan.M.rowKind === "monthly" ? "monthly rows (one per year)" : "five-day bins") + " that contributed; a bin belongs to the calendar month its five-day window OPENS in");
    if (s.res !== "native") n.push("each " + s.res + "° cell POOLS the sums and the counts of every native cell and year in it (Σ sums ÷ Σ counts) — not a mean of means");
    if (plan.d.levels) n.push("levelled channels are written one variable per channel and level; the name carries the level (" + plan.d.levelUnit + ")");
    if (s.bbox && s.bbox.w > s.bbox.e) n.push("the box crosses the dateline: longitudes run past 180° (subtract 360 for −180..180)");
    return n;
  }

  function normalsEstimate(plan) {
    var y = plan.years, s = plan.s;
    var wh = plan.planes + " monthly plane" + (plan.planes === 1 ? "" : "s") + " (" + y.length + " year" + (y.length === 1 ? "" : "s") + " × " +
      plan.months.length + " month" + (plan.months.length === 1 ? "" : "s") + " × " + plan.Cs + " channel" + (plan.Cs === 1 ? "" : "s") + "), sums and counts: " +
      plan.reqs.length + " requests, " + fmtMB(plan.readBytes) + " to read (exact: the box's rows of each year's plane, merged where the gap is smaller than a request costs); the result is " +
      plan.T + " × " + plan.Cs + " × " + plan.geo.Ho + " × " + plan.geo.Wo + " (" + fmtMB(plan.outBytes) + ").";
    return { requests: plan.reqs.length, readBytes: plan.readBytes, outBytes: plan.outBytes, frames: plan.planes, exact: true,
      shape: [plan.T, plan.Cs, plan.geo.Ho, plan.geo.Wo], why: wh, years: y.length, yearsUsed: y.slice(),
      channelsRead: plan.Cs, channelsKept: plan.Cs, channelWord: plan.d.levels ? "levels" : "channels",
      shrink: "Shorten the period, pick fewer months or channels, shrink the box (a narrower band of LATITUDES is what saves bytes), or take 1° cells." };
  }

  function normalsResult(plan, g, ctx, t0, empty) {
    var y = plan.years;
    return {
      kind: "grid", store: plan.d.name, family: plan.d.family, channels: plan.s.channels.slice(),
      units: plan.conv.map(function (c) { return c.unit; }),
      lat: plan.geo.outLat, lon: plan.geo.outLon, time: empty ? new Float64Array(0) : plan.times,
      data: g.data, count: g.count, sel: plan.s, notes: normalsNotes(plan), frames: empty ? 0 : plan.planes,
      group: plan.d.name, source: plan.M.sum.url, title: plan.d.title,
      countMeaning: plan.M.rowKind === "monthly" ? "number of monthly rows (years) averaged" : "number of five-day bins averaged",
      climatology: plan.s.step === "normal" && !empty ? { bounds: plan.bounds, period: y.length ? [y[0], y[y.length - 1]] : null,
        excluded: plan.s.excludeYears.slice(), yearsUsed: y.slice() } : null,
      levels: plan.d.levels ? plan.s.channels.map(function (c) { var x = plan.d.channels.find(function (z) { return z.name === c; }); return x && x.level != null ? x.level : null; }) : null,
      stats: { requests: ctx.stats.requests, bytes: ctx.stats.bytes, ms: Date.now() - t0 }
    };
  }

  // ============================================================== POINTS ===
  function pointStoreJson(d, ctx) {
    return readJSON(d._base + "store.json", ctx);
  }

  // The columns of a tier-P store. Two layouts are read:
  //   family 10 / 1.gf — time_s.npy (int32 s, schema 2; int64, schema 3),
  //     values.npy [N, C] float16, platform.npy int64, qc.npy uint8;
  //   family 8's Argo store (schema 1) — time_days.npy float32 DAYS, the
  //     values as separate [N, 16] float16 blocks temp.npy and psal.npy (read
  //     as one 32-column matrix, temp levels then psal levels, the way
  //     ml/family10_store.py reads it), wmo.npy int32 as the platform, no qc.
  async function pointColumns(d, ctx) {
    return cached("pcols:" + d.id + ":" + d._base, async function () {
      var meta = await pointStoreJson(d, ctx);
      var split = meta.hub_split || {};
      var base = d._base;
      var sch = Number(meta.schema_version || (d.schema || 2));
      var files = meta.sha256 || meta.files || null;
      var has = function (nm) {
        if (!files) return null;
        if (Array.isArray(files)) return files.some(function (f) { return f && f.name === nm + ".npy"; });
        return Object.prototype.hasOwnProperty.call(files, nm + ".npy");
      };
      var v1 = sch === 1 || has("time_days") === true;
      var blocks = v1 && has("values") !== true ? ["temp", "psal"] : null;
      var names = [v1 ? "time_days" : "time_s", "lat", "lon"].concat(blocks || ["values"]);
      var optional = v1 ? ["wmo"] : ["platform", "qc"];
      var cols = {};
      async function openCol(nm, opt) {
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
        var h;
        try { h = await npyHeaderAt(parts[0].url, ctx); }
        catch (e) { if (opt && /HTTP 404/.test(e.message || "")) return; throw e; }
        if (h.descr === "struct") throw new Error(file + " is a structured array: " + parts[0].url);
        var rowElems = h.shape.slice(1).reduce(function (a, b) { return a * b; }, 1);
        cols[nm] = { name: nm, parts: parts, hdr: h.dataOffset, descr: h.descr, itemsize: h.itemsize, shape: h.shape, rowBytes: h.itemsize * rowElems, width: rowElems };
      }
      await Promise.all(names.map(function (nm) { return openCol(nm, false); })
        .concat(optional.map(function (nm) { return openCol(nm, true); })));
      var N = cols.lat.shape[0];
      Object.keys(cols).forEach(function (nm) {
        if (cols[nm].shape[0] !== N) throw new Error(nm + ".npy has " + cols[nm].shape[0] + " rows, lat.npy " + N + " (" + d.name + ")");
      });
      var want = { time_s: ["i4", "i8"], time_days: ["f4"], lat: ["f4"], lon: ["f4"], values: ["f2"], temp: ["f2"], psal: ["f2"],
        platform: ["i8"], wmo: ["i4", "i8"], qc: ["u1"] };
      Object.keys(cols).forEach(function (nm) {
        if (want[nm].indexOf(cols[nm].descr) < 0) throw new Error(nm + ".npy is " + cols[nm].descr + ", expected " + want[nm].join(" or ") + " (" + d.name + ")");
      });
      if (!v1 && (sch === 3) !== (cols.time_s.descr === "i8")) {
        throw new Error("store.json says schema " + sch + " but time_s.npy is " + cols.time_s.descr + " (" + d.name + ")");
      }
      var time = cols[v1 ? "time_days" : "time_s"];
      time.kind = v1 ? "days32" : (time.descr === "i8" ? "s64" : "s32");
      var vblocks = blocks ? blocks.map(function (b) { return cols[b]; }) : [cols.values];
      var values = { blocks: vblocks, width: vblocks.reduce(function (a, b) { return a + b.width; }, 0),
        rowBytes: vblocks.reduce(function (a, b) { return a + b.rowBytes; }, 0) };
      var offs = await readWholeNpy(base + "bin_offsets.npy", ctx);
      if (offs.hdr.descr !== "i8") throw new Error("bin_offsets.npy is " + offs.hdr.descr + " (" + d.name + ")");
      var off = new Float64Array(offs.n);
      for (var i = 0; i < offs.n; i++) off[i] = i64At(offs.body, 8 * i);
      if (off[0] !== 0 || off[offs.n - 1] !== N) throw new Error("bin_offsets.npy runs " + off[0] + ".." + off[offs.n - 1] + " over " + N + " rows (" + d.name + ")");
      var bf = meta.bin_first != null ? Number(meta.bin_first) : (d.binFirst != null ? Number(d.binFirst) : 0);
      return { meta: meta, cols: { time: time, lat: cols.lat, lon: cols.lon, values: values,
        platform: cols.platform || cols.wmo || null, qc: cols.qc || null },
        schema: v1 ? 1 : sch, N: N, C: values.width, off: off, binFirst: bf, nBins: offs.n - 1 };
    });
  }

  // a time column's row i as SECONDS since 1982 (schema 1: float32 days,
  // rounded to the second — the precision the column has, nothing more)
  var _f32 = new Float32Array(1), _f32u8 = new Uint8Array(_f32.buffer);
  function timeAt(col, u8, i) {
    if (col.kind === "s64") return i64At(u8, 8 * i);
    if (col.kind === "s32") return i32At(u8, 4 * i);
    var o = 4 * i;
    _f32u8[0] = u8[o]; _f32u8[1] = u8[o + 1]; _f32u8[2] = u8[o + 2]; _f32u8[3] = u8[o + 3];
    return roundHalfEven(_f32[0] * 86400);
  }

  // numpy's rint — ml/family10_store.py::seconds_of_days rounds the schema-1
  // days this way, and a float32 day often lands exactly on half a second
  // (measured: 52 of 900 Argo profiles in one month), where Math.round would
  // put the profile one second later than the reference reader does
  function roundHalfEven(x) {
    var f = Math.floor(x), d = x - f;
    if (d > 0.5) return f + 1;
    if (d < 0.5) return f;
    return f % 2 === 0 ? f : f + 1;
  }

  // rows [r0, r1) of the value matrix as float16 bytes [n, C], whatever the
  // number of blocks it is stored in
  async function readValueRows(values, r0, r1, ctx) {
    if (values.blocks.length === 1) return readColumnRows(values.blocks[0], r0, r1, ctx);
    var parts = await Promise.all(values.blocks.map(function (b) { return readColumnRows(b, r0, r1, ctx); }));
    var n = r1 - r0, out = new Uint8Array(n * values.rowBytes);
    for (var r = 0; r < n; r++) {
      var o = r * values.rowBytes;
      for (var k = 0; k < parts.length; k++) {
        var rb = values.blocks[k].rowBytes;
        out.set(parts[k].subarray(r * rb, (r + 1) * rb), o);
        o += rb;
      }
    }
    return out;
  }

  async function readPlatformRows(col, r0, r1, ctx) {
    var n = r1 - r0;
    if (!col) return new BigInt64Array(n);
    var b = await readColumnRows(col, r0, r1, ctx);
    if (col.descr === "i8") { var pb = aligned(b, 8); return new BigInt64Array(pb.buffer, pb.byteOffset, n); }
    var out = new BigInt64Array(n);
    for (var i = 0; i < n; i++) out[i] = BigInt(i32At(b, 4 * i));
    return out;
  }

  async function readQcRows(col, r0, r1, ctx) {
    if (!col) return new Uint8Array(r1 - r0);
    return readColumnRows(col, r0, r1, ctx);
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
    var W = 4096;
    var tAt = function (u8, i) { return timeAt(col, u8, i); };
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
    var bf = pc.binFirst, bl = pc.binFirst + pc.nBins - 1, col = pc.cols.time;
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
    var d = await storeDesc(sel, ctx);
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
    var rowBytesCore = pc.cols.time.rowBytes + 8;
    var rowBytesRest = pc.cols.values.rowBytes + (pc.cols.platform ? pc.cols.platform.rowBytes : 0) + (pc.cols.qc ? 1 : 0);
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
    var tcol = cols.time;
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
        readColumnRows(tcol, r0, r1, rctx),
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
        var t = timeAt(tcol, tb, i);
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
        readValueRows(cols.values, a, z, rctx),
        plan.binned ? null : readPlatformRows(cols.platform, a, z, rctx),
        plan.binned ? null : readQcRows(cols.qc, a, z, rctx)]);
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
      var pl = more[1], qb = more[2];
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
      var rs = await firstRowAtOrAfter(tcol, run0.r0, run0.r1, startT, rctx);
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
    var d = await storeDesc(sel, ctx);
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
    if (d._monthly) {
      var np;
      try { np = await normalsPlan(d, sel, ctx); }
      catch (e) {
        if (e.needBox) {
          return { requests: 0, readBytes: 0, outBytes: 0, frames: 0, overCap: true, exact: true, shape: null,
            why: "Draw or type a box first: the normals are read row band by row band, so a box is required." };
        }
        throw e;
      }
      return cap(normalsEstimate(np));
    }
    if (d._dense) {
      var dp;
      try { dp = await densePlan(d, sel, ctx); }
      catch (e) {
        if (e.needBox) {
          return { requests: 0, readBytes: 0, outBytes: 0, frames: 0, overCap: true, exact: true, shape: null,
            why: "Draw or type a box first: a gridded store is read row band by row band, so a box is required." };
        }
        throw e;
      }
      return cap(denseEstimate(dp));
    }
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
        // a tile holds every channel side by side ([row, col, channel]), so
        // the read is all C of them whichever were ticked
        channelsRead: plan.tg.C, channelsKept: plan.Cs, channelWord: plan.d.levels ? "levels" : "channels",
        why: plan.frames.length + " frame" + (plan.frames.length === 1 ? "" : "s") + " from " + plan.bins.length +
          " five-day file" + (plan.bins.length === 1 ? "" : "s") + ": " + g.requests + " requests, " + fmtMB(g.readBytes) +
          " to read (" + g.how + (plan.tg.C > plan.Cs ? "; every tile holds all " + plan.tg.C + " " +
          (plan.d.levels ? "levels" : "channels") + " side by side, so all of them are read" : "") + "); the result is " + plan.T + " × " + plan.Cs + " × " + plan.geo.Ho + " × " + plan.geo.Wo +
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
      var d = await storeDesc(sel, ctx);
      if (d._monthly) {
        var np = await normalsPlan(d, sel, ctx);
        if (np.readBytes > CAP_READ) throw new Error("over the cap: this selection reads " + fmtMB(np.readBytes) + " (limit " + fmtMB(CAP_READ) + ")");
        if (np.outBytes > CAP_OUT) throw new Error("over the cap: the result would be " + fmtMB(np.outBytes) + " of arrays (limit " + fmtMB(CAP_OUT) + ")");
        return normalsResult(np, await normalsRun(np, ctx, opts.onProgress), ctx, t0);
      }
      if (d._dense) {
        var dp = await densePlan(d, sel, ctx);
        if (dp.readBytes > CAP_READ) throw new Error("over the cap: this selection reads " + fmtMB(dp.readBytes) + " (limit " + fmtMB(CAP_READ) + ")");
        if (dp.outBytes > CAP_OUT) throw new Error("over the cap: the result would be " + fmtMB(dp.outBytes) + " of arrays (limit " + fmtMB(CAP_OUT) + ")");
        return denseResult(dp, await denseRun(dp, ctx, opts.onProgress), ctx, t0);
      }
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
      var d = await storeDesc(sel, ctx);
      if (d._monthly) {
        // the first selected calendar month: its normal over the period (or,
        // for the stack, every year of it — then the first year's field)
        var p0 = await normalsPlan(d, sel, ctx);
        var p1 = await normalsPlan(d, Object.assign({}, sel), ctx, { onlyMonth: p0.months[0] });
        if (p1.s.step === "by-year" && p1.years.length) {
          p1 = await normalsPlan(d, Object.assign({}, sel, { yearStart: p1.years[0], yearEnd: p1.years[0] }), ctx, { onlyMonth: p0.months[0] });
        }
        if (!p1.reqs.length) return normalsResult(p1, { data: new Float32Array(0), count: null }, ctx, t0, true);
        return normalsResult(p1, await normalsRun(p1, ctx, null), ctx, t0);
      }
      if (d._dense) {
        // the first selected frame with a finite value in the box, at most
        // PREVIEW_GRID_TRIES frames read
        var s0 = Object.assign({}, sel, { step: "native" });
        var all = await densePlan(d, s0, ctx);
        var fb = null;
        for (var fi = 0; fi < all.frames.length && fi < PREVIEW_GRID_TRIES; fi++) {
          var one = await densePlan(d, s0, ctx, { onlyFrame: Object.assign({}, all.frames[fi], { si: undefined }) });
          var gg = await denseRun(one, ctx, null);
          var rr = denseResult(one, gg, ctx, t0);
          for (var qq = 0; qq < gg.data.length; qq++) if (gg.data[qq] === gg.data[qq]) return rr;
          if (!fb) fb = rr;
        }
        if (fb) { fb.notes.push("none of the first " + Math.min(all.frames.length, PREVIEW_GRID_TRIES) + " frames had a finite value in the box"); return fb; }
        return denseResult(all, { data: new Float32Array(0), count: null }, ctx, t0, true);
      }
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
      kind: "grid", store: plan.d.name, family: plan.d.family, channels: plan.s.channels.slice(), units: units,
      lat: plan.geo.outLat, lon: plan.geo.outLon, time: empty ? new Float64Array(0) : plan.times,
      data: g.data, count: g.count, sel: plan.s,
      notes: gridNotes(plan), frames: empty ? 0 : plan.frames.length, group: plan.group,
      source: groupUrl(plan.d, plan.group), title: plan.d.title, licence: plan.d.licence || null,
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
      rowsRead: plan.br.rows, source: plan.d._base, title: plan.d.title, family: plan.d.family, licence: plan.d.licence || null,
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
    var fam = result.family || "1.gf";
    var famWord = fam === "derived" ? "derived map" : "family " + fam + " store";
    var a = [
      ["Conventions", "CF-1.8"],
      ["title", famWord + " " + result.store + (result.title ? " — " + result.title : "")],
      ["source", "Hugging Face dataset chfrank/earth-tensors (" + famWord + " " + result.store + "), read by blauewelt.org's Data tab (src/f1data.js)"],
      ["source_url", result.source || cfg.base],
      ["selection", selJson],
      ["history", new Date().toISOString() + " written by src/f1data.js from HTTP range reads of the store"]
    ];
    if (result.licence && result.licence.name) a.push(["license", result.licence.name]);
    if (result.licence && result.licence.attribution) a.push(["attribution", result.licence.attribution]);
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
      var clim = result.climatology && result.climatology.bounds && result.climatology.bounds.length === 2 * T ? result.climatology : null;
      if (clim) {
        // CF §7.4: a climatological time axis — each value is a calendar
        // month, averaged over the years its bounds span
        dims.push(["nv", 2]);
        vars[0].attrs.push(["climatology", "climatology_bounds"]);
        vars[0].attrs[1] = ["long_name", "calendar month of the climatology (its value is that month in the first year of the period)"];
        vars.push({ name: "climatology_bounds", dims: [0, 3], type: NC.DOUBLE, n: 2 * T,
          attrs: [["long_name", "first and last instant of the years each climatological month averages"], ["units", "seconds since 1970-01-01 00:00:00"]],
          fill: function (dv, s, k) { for (var i = 0; i < k; i++) dv.setFloat64(8 * i, clim.bounds[s + i], false); } });
        used.climatology_bounds = 1; used.nv = 1;
      }
      var names = result.channels.map(function (c) {
        var n = ncName(c);
        while (used[n]) n += "_";
        used[n] = 1;
        return n;
      });
      result.channels.forEach(function (c, k) {
        var map = function (j) { var t = Math.floor(j / HW); return (t * C + k) * HW + (j - t * HW); };
        var at = [["long_name", c], ["units", result.units[k] || ""], ["_FillValue", NaN, NC.FLOAT]];
        if (clim) at.push(["cell_methods", "time: mean within years time: mean over years"]);
        else if (result.count) at.push(["cell_methods", "time: mean (of finite observations) area: mean"]);
        vars.push({ name: names[k], dims: [0, 1, 2], type: NC.FLOAT, n: T * HW, attrs: at, fill: f32Filler(result.data, map) });
      });
      if (result.count) {
        result.channels.forEach(function (c, k) {
          var nm = names[k] + "_count";
          while (used[nm]) nm += "_";
          used[nm] = 1;
          var map = function (j) { var t = Math.floor(j / HW); return (t * C + k) * HW + (j - t * HW); };
          vars.push({ name: nm, dims: [0, 1, 2], type: NC.INT, n: T * HW,
            attrs: [["long_name", (result.countMeaning || "number of finite observations averaged") + " into " + c], ["units", "1"]],
            fill: function (dv, s, kk) { for (var i = 0; i < kk; i++) dv.setInt32(4 * i, result.count[map(s + i)], false); } });
        });
      }
      var extra = [];
      if (result.climatology && result.climatology.period) {
        extra.push(["climatology_period_start", result.climatology.period[0], NC.INT],
          ["climatology_period_end", result.climatology.period[1], NC.INT],
          ["climatology_years_used", result.climatology.yearsUsed.length, NC.INT],
          ["climatology_years_excluded", result.climatology.excluded.length ? result.climatology.excluded.join(", ") : "none"]);
      }
      return writeNetCDF(dims, globalAttrs(result, extra), vars);
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
    DEFAULT_REGISTRIES: DEFAULT_REGISTRIES,
    // for tests only — not part of the contract
    _internal: { parseNpyHeader: parseNpyHeader, civil: civil, daysFromCivil: daysFromCivil, F16: F16, normBox: normBox, conversionOf: conversionOf, isoOfUnix: isoOfUnix }
  };
});

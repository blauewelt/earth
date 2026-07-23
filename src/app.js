/* earth — open climate data on a globe
 * CesiumJS + NASA GIBS (zero API keys). MIT licensed.
 */
"use strict";

/* ---------------------------------------------------------------- GIBS setup */

const GIBS_URL =
  "https://gibs.earthdata.nasa.gov/wmts/epsg4326/best/" +
  "{layer}/default/{time}/{tms}/{TileMatrix}/{TileRow}/{TileCol}.{ext}";

/* GIBS EPSG:4326 uses a non-standard tile pyramid (level 0 = 2x1 512px tiles
 * spanning 288° each, level 1 = 3x2, ...). Cesium's default GeographicTilingScheme
 * assumes a power-of-two pyramid, so we implement the GIBS scheme explicitly.
 * Degrees-per-pixel at level L is 0.5625 / 2^L for every GIBS 4326 matrix set;
 * the sets differ only in how many levels they have (250m: 9, 500m: 8, 1km: 7, 2km: 6).
 *
 * IMPORTANT: edge tiles are *partial* — GIBS pads them with empty pixels but the
 * image still represents the full nominal span. So tile rectangles must NOT be
 * clamped to the globe: declare the full span and let Cesium sample only the
 * valid part (otherwise the padding gets stretched across the Pacific).
 */
class GIBSGeographicTilingScheme {
  constructor(options = {}) {
    this._tileSize = 512;
    this._baseRes = 0.5625; // degrees per pixel at level 0
    this._ellipsoid = options.ellipsoid || Cesium.Ellipsoid.WGS84;
    this._projection = new Cesium.GeographicProjection(this._ellipsoid);
    this._rectangle = Cesium.Rectangle.fromDegrees(-180, -90, 180, 90);
  }
  get ellipsoid() { return this._ellipsoid; }
  get rectangle() { return this._rectangle; }
  get projection() { return this._projection; }
  _res(level) { return this._baseRes / 2 ** level; }
  getNumberOfXTilesAtLevel(level) {
    return Math.ceil(360 / (this._res(level) * this._tileSize));
  }
  getNumberOfYTilesAtLevel(level) {
    return Math.ceil(180 / (this._res(level) * this._tileSize));
  }
  rectangleToNativeRectangle(rectangle, result) {
    const west = Cesium.Math.toDegrees(rectangle.west);
    const south = Cesium.Math.toDegrees(rectangle.south);
    const east = Cesium.Math.toDegrees(rectangle.east);
    const north = Cesium.Math.toDegrees(rectangle.north);
    if (!result) return new Cesium.Rectangle(west, south, east, north);
    result.west = west; result.south = south; result.east = east; result.north = north;
    return result;
  }
  tileXYToNativeRectangle(x, y, level, result) {
    const span = this._res(level) * this._tileSize;
    const west = -180 + x * span;
    const north = 90 - y * span;
    const east = west + span;   // full nominal span — do NOT clamp (partial tiles are padded)
    const south = north - span;
    if (!result) return new Cesium.Rectangle(west, south, east, north);
    result.west = west; result.south = south; result.east = east; result.north = north;
    return result;
  }
  tileXYToRectangle(x, y, level, result) {
    const r = this.tileXYToNativeRectangle(x, y, level, result);
    r.west = Cesium.Math.toRadians(r.west);
    r.south = Cesium.Math.toRadians(r.south);
    r.east = Cesium.Math.toRadians(r.east);
    r.north = Cesium.Math.toRadians(r.north);
    return r;
  }
  positionToTileXY(position, level, result) {
    if (!Cesium.Rectangle.contains(this._rectangle, position)) return undefined;
    const span = this._res(level) * this._tileSize;
    const lon = Cesium.Math.toDegrees(position.longitude);
    const lat = Cesium.Math.toDegrees(position.latitude);
    let x = Math.floor((lon + 180) / span);
    let y = Math.floor((90 - lat) / span);
    x = Cesium.Math.clamp(x, 0, this.getNumberOfXTilesAtLevel(level) - 1);
    y = Cesium.Math.clamp(y, 0, this.getNumberOfYTilesAtLevel(level) - 1);
    if (!result) return new Cesium.Cartesian2(x, y);
    result.x = x; result.y = y;
    return result;
  }
}

// Verified against GIBS GetCapabilities (July 2026).
const GIBS_LAYERS = [
  {
    id: "viirs-truecolor",
    doc: "https://www.earthdata.nasa.gov/data/instruments/viirs",
    layer: "VIIRS_SNPP_CorrectedReflectance_TrueColor",
    title: "True color (VIIRS, daily)",
    ext: "jpg", tms: "250m", maxLevel: 8,
    start: "2015-11-24", timed: true, on: false,
    meta: "Daily global mosaic, ~3 h latency",
  },
  {
    id: "sst",
    colormap: "https://gibs.earthdata.nasa.gov/colormaps/v1.3/GHRSST_Sea_Surface_Temperature.xml",
    legend: "https://gibs.earthdata.nasa.gov/legends/GHRSST_Sea_Surface_Temperature_H.svg",
    doc: "https://podaac.jpl.nasa.gov/dataset/MUR-JPL-L4-GLOB-v4.1",
    layer: "GHRSST_L4_MUR_Sea_Surface_Temperature",
    title: "Sea surface temperature (MUR 1 km)",
    ext: "png", tms: "1km", maxLevel: 6,
    start: "2002-06-01", timed: true, on: true,
    meta: "GHRSST L4 analysis — watch the North Atlantic cold blob",
  },
  {
    id: "sst-anom",
    colormap: "https://gibs.earthdata.nasa.gov/colormaps/v1.3/GHRSST_Sea_Surface_Temperature_Anomalies.xml",
    legend: "https://gibs.earthdata.nasa.gov/legends/GHRSST_Sea_Surface_Temperature_Anomalies_H.svg",
    doc: "https://podaac.jpl.nasa.gov/dataset/MUR25-JPL-L4-GLOB-v04.2",
    layer: "GHRSST_L4_MUR25_Sea_Surface_Temperature_Anomalies",
    title: "SST anomalies (MUR 25 km)",
    ext: "png", tms: "2km", maxLevel: 5,
    start: "2002-09-01", timed: true, on: false,
    meta: "Anomaly vs climatology — AMOC fingerprint region",
  },
  {
    id: "precip",
    colormap: "https://gibs.earthdata.nasa.gov/colormaps/v1.3/GPM_Precipitation_Rate.xml",
    legend: "https://gibs.earthdata.nasa.gov/legends/GPM_Precipitation_Rate_H.svg",
    doc: "https://gpm.nasa.gov/data/imerg",
    layer: "IMERG_Precipitation_Rate",
    title: "Precipitation rate (IMERG)",
    ext: "png", tms: "2km", maxLevel: 5,
    start: "2000-06-01", timed: true, on: false,
    meta: "GPM merged precipitation",
  },
  {
    id: "seaice",
    colormap: "https://gibs.earthdata.nasa.gov/colormaps/v1.3/AMSR_Sea_Ice_Concentration.xml",
    legend: "https://gibs.earthdata.nasa.gov/legends/AMSR_Sea_Ice_Concentration_H.svg",
    doc: "https://nsidc.org/data/au_si12",
    layer: "AMSRU2_Sea_Ice_Concentration_12km",
    title: "Sea ice concentration (AMSR2)",
    ext: "png", tms: "2km", maxLevel: 5,
    start: "2012-07-02", timed: true, on: false,
    meta: "Passive-microwave, both poles (lags mission availability)",
  },
  {
    id: "snow",
    colormap: "https://gibs.earthdata.nasa.gov/colormaps/v1.3/MODIS_NDSI_Snow_Cover.xml",
    legend: "https://gibs.earthdata.nasa.gov/legends/MODIS_NDSI_Snow_Cover_H.svg",
    doc: "https://nsidc.org/data/mod10a1",
    layer: "MODIS_Terra_NDSI_Snow_Cover",
    title: "Snow cover (MODIS NDSI)",
    ext: "png", tms: "500m", maxLevel: 7,
    start: "2000-02-24", timed: true, on: false,
    meta: "Daily NDSI snow cover",
  },
  {
    id: "aod",
    colormap: "https://gibs.earthdata.nasa.gov/colormaps/v1.3/MODIS_Combined_Value_Added_AOD.xml",
    legend: "https://gibs.earthdata.nasa.gov/legends/MODIS_Combined_Value_Added_AOD_H.svg",
    doc: "https://atmosphere-imager.gsfc.nasa.gov/products/aerosol",
    layer: "MODIS_Combined_Value_Added_AOD",
    title: "Aerosol optical depth (MODIS)",
    ext: "png", tms: "2km", maxLevel: 5,
    start: "2017-04-19", timed: true, on: false,
    meta: "Smoke, dust and haze",
  },
  {
    id: "nightlights",
    doc: "https://blackmarble.gsfc.nasa.gov/",
    layer: "VIIRS_Black_Marble",
    title: "Night lights (Black Marble)",
    ext: "png", tms: "500m", maxLevel: 7,
    start: "2016-01-01", timed: false, fixedTime: "2016-01-01", on: false,
    meta: "VIIRS annual composite",
  },
];

function gibsProvider(cfg, dateStr) {
  const time = cfg.timed ? dateStr : (cfg.fixedTime || "default");
  const url = GIBS_URL
    .replace("{layer}", cfg.layer)
    .replace("{time}", time)
    .replace("{tms}", cfg.tms)
    .replace("{ext}", cfg.ext);
  return new Cesium.WebMapTileServiceImageryProvider({
    url,
    layer: cfg.layer,
    style: "default",
    format: cfg.ext === "jpg" ? "image/jpeg" : "image/png",
    tileMatrixSetID: cfg.tms,
    maximumLevel: cfg.maxLevel,
    tileWidth: 512,
    tileHeight: 512,
    tilingScheme: new GIBSGeographicTilingScheme(),
    credit: new Cesium.Credit("NASA GIBS / Worldview"),
  });
}

/* ------------------------------------------------------------------- viewer */

const baseProvider = new Cesium.WebMapTileServiceImageryProvider({
  url: GIBS_URL
    .replace("{layer}", "BlueMarble_ShadedRelief_Bathymetry")
    .replace("{time}", "default")
    .replace("{tms}", "500m")
    .replace("{ext}", "jpeg"),
  layer: "BlueMarble_ShadedRelief_Bathymetry",
  style: "default",
  format: "image/jpeg",
  tileMatrixSetID: "500m",
  maximumLevel: 7,
  tileWidth: 512,
  tileHeight: 512,
  tilingScheme: new GIBSGeographicTilingScheme(),
  credit: new Cesium.Credit("NASA Blue Marble (GIBS)"),
});

const viewer = new Cesium.Viewer("cesiumContainer", {
  baseLayer: new Cesium.ImageryLayer(baseProvider),
  baseLayerPicker: false,
  geocoder: false,
  timeline: false,
  animation: false,
  sceneModePicker: true,
  navigationHelpButton: false,
  homeButton: false,
  fullscreenButton: true,
  infoBox: true,
  selectionIndicator: true,
});
viewer.scene.globe.enableLighting = false;
viewer.scene.skyAtmosphere.show = true;

/* Zoom gestures: mouse wheel, touch pinch, AND trackpad pinch.
 * Browsers report a MacBook-style trackpad pinch as a wheel event with
 * ctrlKey set, which Cesium ignores unless registered explicitly. */
const sscc = viewer.scene.screenSpaceCameraController;
sscc.minimumZoomDistance = 40000; // keep zoom sane
sscc.zoomEventTypes = [
  Cesium.CameraEventType.WHEEL,
  Cesium.CameraEventType.PINCH,
  { eventType: Cesium.CameraEventType.WHEEL, modifier: Cesium.KeyboardEventModifier.CTRL },
];
// keep the browser from page-zooming on trackpad pinch over the globe
viewer.scene.canvas.addEventListener(
  "wheel",
  (e) => { if (e.ctrlKey) e.preventDefault(); },
  { passive: false }
);

const HOME = { lon: -30, lat: 28, height: 1.5e7 };
viewer.camera.setView({
  destination: Cesium.Cartesian3.fromDegrees(HOME.lon, HOME.lat, HOME.height),
});

/* ---------------------------------------------------------------- zoom controls */

function cameraHeight() {
  return viewer.camera.positionCartographic.height;
}
document.getElementById("zoom-in").addEventListener("click", () => {
  viewer.camera.zoomIn(cameraHeight() * 0.45);
});
document.getElementById("zoom-out").addEventListener("click", () => {
  viewer.camera.zoomOut(cameraHeight() * 0.8);
});
document.getElementById("zoom-home").addEventListener("click", () => {
  viewer.camera.flyTo({
    destination: Cesium.Cartesian3.fromDegrees(HOME.lon, HOME.lat, HOME.height),
    duration: 1.2,
  });
});

/* ------------------------------------------------------------ layer control */

const state = { date: defaultDate(), compareYears: 0, compareMode: "split", layers: {} };

function defaultDate() {
  const d = new Date(Date.now() - 2 * 864e5); // two days ago: safely available on GIBS
  return d.toISOString().slice(0, 10);
}

function compareDate() {
  if (!state.compareYears) return null;
  const [y, m, d] = state.date.split("-").map(Number);
  const day = m === 2 && d === 29 ? 28 : d; // leap-day safety
  return `${y - state.compareYears}-${String(m).padStart(2, "0")}-${String(day).padStart(2, "0")}`;
}


/* ------------------------------------------------- computed-delta (SST) mode */

const DELTA_LAYER_ID = "sst";
const DELTA_RANGE = 4; // °C, legend/scale limit
const DELTA_COOL = [37, 99, 235];   // cooler than N years ago
const DELTA_WARM = [230, 59, 46];   // warmer than N years ago

function parseColormap(xml) {
  // GIBS colormap v1.3: <ColorMapEntry rgb="r,g,b" transparent="false" ... value="[lo,hi)"/>
  const lut = new Map();
  const re = /<ColorMapEntry\s+rgb="(\d+),(\d+),(\d+)"\s+transparent="false"[^>]*?\svalue="\[([^,]+),([^)\]]+)[\)\]]"/g;
  let m;
  while ((m = re.exec(xml))) {
    const key = (+m[1] << 16) | (+m[2] << 8) | +m[3];
    const lo = parseFloat(m[4]);
    const hi = parseFloat(m[5]);
    const v = Number.isFinite(lo) && Number.isFinite(hi) ? (lo + hi) / 2
      : Number.isFinite(lo) ? lo : hi;
    if (Number.isFinite(v)) lut.set(key, v);
  }
  return lut;
}

function deltaColor(d) {
  // diverging: blue = cooler, red = warmer; opacity scales with |delta|
  const t = Cesium.Math.clamp(d / DELTA_RANGE, -1, 1);
  const a = Math.round(Math.min(1, Math.abs(t) + 0.06) * 235);
  if (Math.abs(d) < 0.05) return [0, 0, 0, 0];
  const c = t > 0 ? DELTA_WARM : DELTA_COOL;
  return [c[0], c[1], c[2], a];
}

/* Sample dates used to approximate a period mean from daily GIBS tiles:
 * month → 5 days spread across the month; year → the 15th of each month. */
function periodSampleDates(dateStr, period) {
  const [y, m] = dateStr.split("-").map(Number);
  const pad = (n) => String(n).padStart(2, "0");
  if (period === "day") return [dateStr];
  if (period === "month") return [3, 9, 15, 21, 27].map((d) => `${y}-${pad(m)}-${pad(d)}`);
  return Array.from({ length: 12 }, (_, i) => `${y}-${pad(i + 1)}-15`); // year
}

let sstLUTPromise = null;
function getSstLUT() {
  if (!sstLUTPromise) {
    sstLUTPromise = fetch("https://gibs.earthdata.nasa.gov/colormaps/v1.3/GHRSST_Sea_Surface_Temperature.xml")
      .then((r) => r.text())
      .then(parseColormap);
  }
  return sstLUTPromise;
}

/* Minimal custom ImageryProvider: fetches the SST tile for two dates, inverts
 * the GIBS colormap to °C, and renders the per-pixel difference. */
class SSTDeltaProvider {
  constructor(cfg, dateNow, datePast, period = "day") {
    this._cfg = cfg;
    this._period = period;
    this._datesNow = periodSampleDates(dateNow, period);
    this._datesPast = periodSampleDates(datePast, period);
    this.tilingScheme = new GIBSGeographicTilingScheme();
    this.rectangle = this.tilingScheme.rectangle;
    this.tileWidth = 512;
    this.tileHeight = 512;
    // averaged modes fetch many tiles per tile — cap the zoom to keep it snappy
    this.maximumLevel = period === "day" ? cfg.maxLevel : period === "month" ? 4 : 3;
    this.minimumLevel = 0;
    this.errorEvent = new Cesium.Event();
    this.credit = new Cesium.Credit(
      period === "day"
        ? "Δ computed client-side from NASA GIBS MUR SST"
        : `Δ of sampled ${period}ly means, computed client-side from NASA GIBS MUR SST`
    );
    this.tileDiscardPolicy = undefined;
    this.hasAlphaChannel = true;
    this.proxy = undefined;
    this.ready = true;
  }
  get period() { return this._period; }
  getTileCredits() { return undefined; }
  pickFeatures() { return undefined; }
  _url(date, x, y, level) {
    return GIBS_URL
      .replace("{layer}", this._cfg.layer)
      .replace("{time}", date)
      .replace("{tms}", this._cfg.tms)
      .replace("{ext}", this._cfg.ext)
      .replace("{TileMatrix}", level)
      .replace("{TileRow}", y)
      .replace("{TileCol}", x);
  }
  async _fetchTile(url) {
    try {
      const r = await fetch(url);
      if (!r.ok) return null;
      return await createImageBitmap(await r.blob());
    } catch {
      return null;
    }
  }
  /* Mean value per pixel across the period's sample dates (NaN where no data). */
  async _meanField(dates, x, y, level, lut, ctx) {
    const N = 512 * 512;
    const sum = new Float32Array(N);
    const cnt = new Uint8Array(N);
    const imgs = await Promise.all(dates.map((d) => this._fetchTile(this._url(d, x, y, level))));
    for (const img of imgs) {
      if (!img) continue;
      ctx.clearRect(0, 0, 512, 512);
      ctx.drawImage(img, 0, 0);
      const d = ctx.getImageData(0, 0, 512, 512).data;
      for (let p = 0, i = 0; p < N; p++, i += 4) {
        if (d[i + 3] === 0) continue;
        const v = lut.get((d[i] << 16) | (d[i + 1] << 8) | d[i + 2]);
        if (v === undefined) continue;
        sum[p] += v;
        cnt[p]++;
      }
    }
    return { sum, cnt };
  }
  async requestImage(x, y, level) {
    const lut = await getSstLUT();
    const canvas = document.createElement("canvas");
    canvas.width = 512;
    canvas.height = 512;
    const ctx = canvas.getContext("2d", { willReadFrequently: true });
    const [now, past] = await Promise.all([
      this._meanField(this._datesNow, x, y, level, lut, ctx),
      this._meanField(this._datesPast, x, y, level, lut, ctx),
    ]);
    const out = ctx.createImageData(512, 512);
    const o = out.data;
    for (let p = 0, i = 0; p < 512 * 512; p++, i += 4) {
      if (now.cnt[p] === 0 || past.cnt[p] === 0) continue;
      const d = now.sum[p] / now.cnt[p] - past.sum[p] / past.cnt[p];
      const [r, g, b, a] = deltaColor(d);
      o[i] = r; o[i + 1] = g; o[i + 2] = b; o[i + 3] = a;
    }
    ctx.clearRect(0, 0, 512, 512);
    ctx.putImageData(out, 0, 0);
    return canvas;
  }
}

function addLayer(cfg) {
  const entry = { cfg, layer: null, cmpLayer: null, isDelta: false, alpha: state.layers[cfg.id]?.alpha ?? 1.0 };
  const cmp = compareDate();
  if (cmp && cfg.timed && state.compareMode.startsWith("delta") && cfg.id === DELTA_LAYER_ID) {
    // computed per-pixel difference: value(now) - value(cmp); optionally of period means
    const period = state.compareMode === "delta" ? "day"
      : state.compareMode === "delta-month" ? "month" : "year";
    entry.layer = viewer.imageryLayers.addImageryProvider(
      new SSTDeltaProvider(cfg, state.date, cmp, period)
    );
    entry.layer.alpha = entry.alpha;
    entry.isDelta = true;
  } else {
    entry.layer = viewer.imageryLayers.addImageryProvider(gibsProvider(cfg, state.date));
    entry.layer.alpha = entry.alpha;
    if (cmp && cfg.timed && state.compareMode === "split") {
      entry.layer.splitDirection = Cesium.SplitDirection.RIGHT;      // right = current
      entry.cmpLayer = viewer.imageryLayers.addImageryProvider(gibsProvider(cfg, cmp));
      entry.cmpLayer.alpha = entry.alpha;
      entry.cmpLayer.splitDirection = Cesium.SplitDirection.LEFT;    // left = past
    }
  }
  state.layers[cfg.id] = entry;
  updateLegends();
}

function removeLayer(id) {
  const entry = state.layers[id];
  if (!entry) return;
  if (entry.layer) viewer.imageryLayers.remove(entry.layer, true);
  if (entry.cmpLayer) viewer.imageryLayers.remove(entry.cmpLayer, true);
  entry.layer = null;
  entry.cmpLayer = null;
  entry.isDelta = false;
  updateLegends();
}

function refreshTimedLayers() {
  for (const [id, entry] of Object.entries(state.layers)) {
    if (entry.layer && entry.cfg.timed) {
      removeLayer(id);
      addLayer(entry.cfg);
    }
  }
  updateSplitUI();
}

function anyTimedActive() {
  return Object.values(state.layers).some((e) => e.layer && e.cfg.timed);
}

/* ------------------------------------------------------- comparison (split) */

const splitHandle = document.getElementById("split-handle");
const splitLabels = document.getElementById("split-labels");

function updateSplitUI() {
  document.getElementById("compare-mode-row").classList.toggle("hidden", state.compareYears === 0);
  const active = state.compareYears > 0 && anyTimedActive() && state.compareMode === "split";
  splitHandle.classList.toggle("hidden", !active);
  splitLabels.classList.toggle("hidden", !active);
  if (active) {
    document.getElementById("split-label-left").textContent = compareDate();
    document.getElementById("split-label-right").textContent = state.date;
    positionSplit(viewer.scene.splitPosition || 0.5);
  }
}

function positionSplit(frac) {
  viewer.scene.splitPosition = frac;
  splitHandle.style.left = `${frac * 100}%`;
}

(function initSplitDrag() {
  let dragging = false;
  const container = document.getElementById("cesiumContainer");
  splitHandle.addEventListener("pointerdown", (e) => {
    dragging = true;
    splitHandle.setPointerCapture(e.pointerId);
  });
  splitHandle.addEventListener("pointermove", (e) => {
    if (!dragging) return;
    const rect = container.getBoundingClientRect();
    const frac = Cesium.Math.clamp((e.clientX - rect.left) / rect.width, 0.05, 0.95);
    positionSplit(frac);
  });
  splitHandle.addEventListener("pointerup", () => { dragging = false; });
  positionSplit(0.5);
})();

document.getElementById("compare-select").addEventListener("change", (e) => {
  state.compareYears = Number(e.target.value);
  refreshTimedLayers();
});

document.getElementById("compare-mode").addEventListener("change", (e) => {
  state.compareMode = e.target.value;
  refreshTimedLayers();
});

/* --------------------------------------------------------------- legends */

function updateLegends() {
  const panel = document.getElementById("legend-panel");
  if (!panel) return;
  panel.innerHTML = "";
  let any = false;
  for (const e of Object.values(state.layers)) {
    if (!e.layer) continue;
    if (e.isDelta) {
      panel.appendChild(deltaLegendEl());
      any = true;
    } else if (e.cfg.colormap || e.cfg.legend) {
      panel.appendChild(layerLegendEl(e.cfg));
      any = true;
    }
  }
  panel.classList.toggle("hidden", !any);
}

/* Interactive legends: rendered from the layer's GIBS colormap so hovering
 * reveals the exact value (with units) of the color under the cursor. */

const colormapCache = new Map();
function getColormapEntries(url) {
  if (!colormapCache.has(url)) {
    colormapCache.set(
      url,
      fetch(url).then((r) => r.text()).then(parseColormapEntries).catch(() => null)
    );
  }
  return colormapCache.get(url);
}

function parseColormapEntries(xml) {
  const units = (xml.match(/units="([^"]+)"/) || [])[1] || "";
  const entries = [];
  const re = /<ColorMapEntry\s+rgb="(\d+),(\d+),(\d+)"\s+transparent="false"[^>]*?\svalue="[\[\(]([^,]+),([^)\]]+)[\)\]]"/g;
  let m;
  while ((m = re.exec(xml))) {
    let lo = parseFloat(m[4]);
    let hi = parseFloat(m[5]);
    if (!Number.isFinite(lo) && !Number.isFinite(hi)) continue;
    if (!Number.isFinite(lo)) lo = hi;
    if (!Number.isFinite(hi)) hi = lo;
    entries.push({ rgb: [+m[1], +m[2], +m[3]], lo, hi });
  }
  entries.sort((a, b) => a.lo - b.lo);
  return { units, entries };
}

function fmtVal(v) {
  const a = Math.abs(v);
  return a >= 100 ? v.toFixed(0) : a >= 10 ? v.toFixed(1) : v.toFixed(2);
}

function layerLegendEl(cfg) {
  const div = document.createElement("div");
  div.className = "legend-item";
  div.innerHTML = `<div class="legend-title">${cfg.title}</div>`;
  const fallback = () => {
    if (cfg.legend) {
      div.insertAdjacentHTML(
        "beforeend",
        `<img src="${cfg.legend}" alt="${cfg.title} legend"/>`
      );
    }
  };
  if (cfg.colormap) {
    getColormapEntries(cfg.colormap).then((cm) => {
      if (cm && cm.entries.length >= 2) buildLegendBar(div, cm);
      else fallback();
    });
  } else {
    fallback();
  }
  return div;
}

function buildLegendBar(container, cm) {
  const wrap = document.createElement("div");
  wrap.className = "legend-bar-wrap";
  const canvas = document.createElement("canvas");
  const W = 268, H = 14;
  const dpr = window.devicePixelRatio || 1;
  canvas.width = W * dpr;
  canvas.height = H * dpr;
  canvas.style.height = H + "px";
  canvas.className = "legend-bar";
  const ctx = canvas.getContext("2d");
  ctx.scale(dpr, dpr);
  const n = cm.entries.length;
  for (let i = 0; i < n; i++) {
    const [r, g, b] = cm.entries[i].rgb;
    ctx.fillStyle = `rgb(${r},${g},${b})`;
    ctx.fillRect((i / n) * W, 0, W / n + 1, H);
  }
  const tip = document.createElement("div");
  tip.className = "legend-tip hidden";
  const range = document.createElement("div");
  range.className = "legend-range";
  range.innerHTML = `<span>${fmtVal(cm.entries[0].lo)}</span><span>${cm.units}</span><span>${fmtVal(cm.entries[n - 1].hi)}</span>`;
  canvas.addEventListener("mousemove", (e) => {
    const rect = canvas.getBoundingClientRect();
    const frac = Cesium.Math.clamp((e.clientX - rect.left) / rect.width, 0, 0.9999);
    const entry = cm.entries[Math.floor(frac * n)];
    const wide = entry.hi - entry.lo > 1;
    tip.textContent = (wide
      ? `${fmtVal(entry.lo)} – ${fmtVal(entry.hi)} ${cm.units}`
      : `${fmtVal((entry.lo + entry.hi) / 2)} ${cm.units}`).trim();
    tip.style.left = `${Math.min(Math.max(frac * rect.width - 28, 0), rect.width - 80)}px`;
    tip.classList.remove("hidden");
  });
  canvas.addEventListener("mouseleave", () => tip.classList.add("hidden"));
  wrap.appendChild(tip);
  wrap.appendChild(canvas);
  container.appendChild(wrap);
  container.appendChild(range);
}

function deltaLegendEl() {
  const div = document.createElement("div");
  div.className = "legend-item";
  const cmp = compareDate();
  const label = state.compareMode === "delta-month"
    ? `Δ SST: ${state.date.slice(0, 7)} mean minus ${cmp.slice(0, 7)} mean (sampled, °C)`
    : state.compareMode === "delta-year"
      ? `Δ SST: ${state.date.slice(0, 4)} mean minus ${cmp.slice(0, 4)} mean (sampled, °C)`
      : `Δ SST: ${state.date} minus ${cmp} (°C)`;
  div.innerHTML = `<div class="legend-title">${label}</div>`;
  const wrap = document.createElement("div");
  wrap.className = "legend-bar-wrap";
  const bar = document.createElement("div");
  bar.className = "delta-bar";
  const tip = document.createElement("div");
  tip.className = "legend-tip hidden";
  bar.addEventListener("mousemove", (e) => {
    const rect = bar.getBoundingClientRect();
    const frac = Cesium.Math.clamp((e.clientX - rect.left) / rect.width, 0, 1);
    const v = -DELTA_RANGE + frac * 2 * DELTA_RANGE;
    tip.textContent = `Δ ${v >= 0 ? "+" : ""}${v.toFixed(1)} °C ${Math.abs(v) < 0.2 ? "(little change)" : v > 0 ? "warmer than then" : "cooler than then"}`;
    tip.style.left = `${Math.min(Math.max(frac * rect.width - 40, 0), rect.width - 130)}px`;
    tip.classList.remove("hidden");
  });
  bar.addEventListener("mouseleave", () => tip.classList.add("hidden"));
  wrap.appendChild(tip);
  wrap.appendChild(bar);
  div.appendChild(wrap);
  const range = document.createElement("div");
  range.className = "legend-range";
  range.innerHTML = `<span>−${DELTA_RANGE}</span><span>°C</span><span>+${DELTA_RANGE}</span>`;
  div.appendChild(range);
  return div;
}

/* ----------------------------------------------------------- GIBS layer panel */

function buildLayerPanel() {
  const list = document.getElementById("layer-list");
  for (const cfg of GIBS_LAYERS) {
    const div = document.createElement("div");
    div.className = "layer-item";
    const title = cfg.doc
      ? `<a class="title-link" href="${cfg.doc}" target="_blank" rel="noopener" title="Open dataset documentation">${cfg.title}</a>`
      : `<span>${cfg.title}</span>`;
    div.innerHTML = `
      <div class="layer-head">
        <input type="checkbox" data-id="${cfg.id}" ${cfg.on ? "checked" : ""} title="Show / hide layer"/>
        ${title}
      </div>
      <div class="meta">${cfg.meta}${cfg.timed ? ` · from ${cfg.start}` : ""}</div>
      <input type="range" min="0" max="100" value="100" data-alpha="${cfg.id}"
             ${cfg.on ? "" : "style='display:none'"} title="opacity"/>`;
    list.appendChild(div);
    if (cfg.on) addLayer(cfg);
  }
  updateSplitUI();

  list.addEventListener("change", (e) => {
    const id = e.target.getAttribute("data-id");
    if (!id) return;
    const cfg = GIBS_LAYERS.find((l) => l.id === id);
    const slider = list.querySelector(`input[data-alpha="${id}"]`);
    if (e.target.checked) {
      addLayer(cfg);
      slider.style.display = "";
    } else {
      removeLayer(id);
      slider.style.display = "none";
    }
    updateSplitUI();
  });

  list.addEventListener("input", (e) => {
    const id = e.target.getAttribute("data-alpha");
    if (!id) return;
    const entry = state.layers[id];
    if (entry) {
      entry.alpha = e.target.value / 100;
      if (entry.layer) entry.layer.alpha = entry.alpha;
      if (entry.cmpLayer) entry.cmpLayer.alpha = entry.alpha;
    }
  });

  const dateInput = document.getElementById("layer-date");
  dateInput.value = state.date;
  dateInput.max = defaultDate();
  dateInput.addEventListener("change", () => {
    if (!dateInput.value) return;
    state.date = dateInput.value;
    refreshTimedLayers();
  });
}

/* ----------------------------------------------------- point data layers */

const pickCard = document.getElementById("pick-card");
const pointLayers = {}; // kind -> {collection, meta}

async function loadPointLayer(kind) {
  if (pointLayers[kind]) {
    pointLayers[kind].collection.show = true;
    return;
  }
  const cfgs = {
    climatetrace: {
      file: "data/climatetrace.json",
      build(json, col) {
        for (const [lon, lat, mt, name, country, sector] of json.assets) {
          col.add({
            position: Cesium.Cartesian3.fromDegrees(lon, lat),
            pixelSize: Math.max(4, Math.min(15, 4 + 11 * Math.sqrt(mt / 270))),
            color: Cesium.Color.fromCssColorString("#d95926").withAlpha(0.85),
            outlineColor: Cesium.Color.BLACK.withAlpha(0.6),
            outlineWidth: 1,
            id: {
              kind: "climatetrace",
              html: `<strong>${esc(name)}</strong><br/>${esc(country)} · ${esc(sector)}<br/>` +
                `<b>${mt.toFixed(1)} Mt CO₂e/yr</b> (${json.year})<br/>` +
                `<a href="https://climatetrace.org" target="_blank" rel="noopener">Climate TRACE ↗</a>`,
            },
          });
        }
        return `${json.assets.length} facilities · ${json.year} · snapshot ${json.snapshot}`;
      },
    },
    argo: {
      file: "data/argo.json",
      build(json, col) {
        for (const [lon, lat, id, date] of json.floats) {
          col.add({
            position: Cesium.Cartesian3.fromDegrees(lon, lat),
            pixelSize: 4,
            color: Cesium.Color.fromCssColorString("#3987e5").withAlpha(0.9),
            outlineColor: Cesium.Color.BLACK.withAlpha(0.5),
            outlineWidth: 1,
            id: {
              kind: "argo",
              html: `<strong>Argo float ${esc(id)}</strong><br/>Last profile: ${esc(date)}<br/>` +
                `<a href="https://fleetmonitoring.euro-argo.eu/float/${esc(id)}" target="_blank" rel="noopener">Float dashboard ↗</a>`,
            },
          });
        }
        return `${json.floats.length} active floats · snapshot ${json.snapshot}`;
      },
    },
  };
  const cfg = cfgs[kind];
  const json = await (await fetch(cfg.file)).json();
  const col = viewer.scene.primitives.add(new Cesium.PointPrimitiveCollection());
  const meta = cfg.build(json, col);
  pointLayers[kind] = { collection: col, meta };
  const metaEl = document.getElementById(`meta-${kind}`);
  if (metaEl) metaEl.textContent = meta;
}

function esc(s) {
  return String(s).replace(/[&<>"']/g, (c) => ({
    "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;",
  }[c]));
}

for (const kind of ["climatetrace", "argo"]) {
  document.getElementById(`toggle-${kind}`).addEventListener("change", (e) => {
    if (e.target.checked) loadPointLayer(kind);
    else if (pointLayers[kind]) pointLayers[kind].collection.show = false;
  });
}

// Click-picking for point primitives → info card
new Cesium.ScreenSpaceEventHandler(viewer.scene.canvas).setInputAction((click) => {
  const picked = viewer.scene.pick(click.position);
  if (picked?.id?.kind) {
    pickCard.innerHTML = picked.id.html;
    pickCard.classList.remove("hidden");
  } else if (!picked) {
    pickCard.classList.add("hidden");
  }
}, Cesium.ScreenSpaceEventType.LEFT_CLICK);

/* ----------------------------------------------------------------- stations */

let stationsDs = null;

async function loadStations() {
  stationsDs = await Cesium.GeoJsonDataSource.load("data/stations.geojson");
  for (const entity of stationsDs.entities.values) {
    const p = entity.properties;
    const type = p.type?.getValue() || "station";
    const isAmoc = type.includes("AMOC");
    entity.billboard = undefined;
    entity.point = new Cesium.PointGraphics({
      pixelSize: 9,
      color: isAmoc
        ? Cesium.Color.fromCssColorString("#f0883e")
        : Cesium.Color.fromCssColorString("#3fb950"),
      outlineColor: Cesium.Color.BLACK,
      outlineWidth: 1.5,
    });
    entity.label = new Cesium.LabelGraphics({
      text: p.name?.getValue() || "",
      font: "11px sans-serif",
      fillColor: Cesium.Color.WHITE,
      outlineColor: Cesium.Color.BLACK,
      outlineWidth: 2,
      style: Cesium.LabelStyle.FILL_AND_OUTLINE,
      pixelOffset: new Cesium.Cartesian2(0, -14),
      distanceDisplayCondition: new Cesium.DistanceDisplayCondition(0, 2.2e7),
    });
    entity.description = `
      <h3>${p.name?.getValue() || ""}</h3>
      <p><em>${type}</em></p>
      <p>${p.description?.getValue() || ""}</p>
      <p><a href="${p.url?.getValue() || "#"}" target="_blank" rel="noopener">Data access →</a></p>`;
  }
  viewer.dataSources.add(stationsDs);

  document.getElementById("toggle-stations").addEventListener("change", (e) => {
    stationsDs.show = e.target.checked;
  });
}

document.getElementById("fly-atlantic").addEventListener("click", () => {
  viewer.camera.flyTo({
    destination: Cesium.Cartesian3.fromDegrees(-40, 40, 1.15e7),
    duration: 2.2,
  });
});

/* ------------------------------------------------------------- AMOC dashboard */

let rapidData = null;

async function loadAmoc() {
  if (rapidData) return;
  rapidData = await (await fetch("data/rapid_moc.json")).json();
  const { t, moc } = rapidData;

  const vals = moc.filter((v) => v != null);
  const latest = moc[moc.length - 1];
  const early = mean(sliceByYears(t, moc, 2004, 2009));
  const recent = mean(sliceByYears(t, moc, 2019, 2025));

  setStat("amoc-latest", latest, `Sv · ${t[t.length - 1]}`);
  setStat("amoc-early", early, "Sv · 2004–08 mean");
  setStat("amoc-recent", recent, "Sv · last 5 yr mean");
  const delta = recent - early;
  const deltaEl = document.getElementById("amoc-delta");
  deltaEl.textContent = `${delta >= 0 ? "+" : ""}${delta.toFixed(1)} Sv since the array's first five years`;

  drawAmocChart(t, moc, Math.min(...vals), Math.max(...vals));
}

function sliceByYears(t, v, y0, y1) {
  return v.filter((x, i) => x != null && +t[i].slice(0, 4) >= y0 && +t[i].slice(0, 4) < y1);
}
function mean(a) { return a.reduce((s, x) => s + x, 0) / a.length; }
function setStat(id, val, sub) {
  document.querySelector(`#${id} .stat-value`).textContent = val.toFixed(1);
  document.querySelector(`#${id} .stat-sub`).textContent = sub;
}

function drawAmocChart(t, moc, vmin, vmax) {
  const canvas = document.getElementById("amoc-chart");
  const wrap = canvas.parentElement;
  const cssW = wrap.clientWidth, cssH = 170;
  const dpr = window.devicePixelRatio || 1;
  canvas.width = cssW * dpr; canvas.height = cssH * dpr;
  canvas.style.width = cssW + "px"; canvas.style.height = cssH + "px";
  const ctx = canvas.getContext("2d");
  ctx.scale(dpr, dpr);

  const M = { l: 26, r: 6, t: 8, b: 18 };
  const W = cssW - M.l - M.r, H = cssH - M.t - M.b;
  const y0 = Math.floor(vmin / 5) * 5, y1 = Math.ceil(vmax / 5) * 5;
  const X = (i) => M.l + (i / (t.length - 1)) * W;
  const Y = (v) => M.t + (1 - (v - y0) / (y1 - y0)) * H;

  ctx.clearRect(0, 0, cssW, cssH);
  ctx.font = "10px system-ui, sans-serif";

  // gridlines + y labels (muted ink, hairline grid)
  for (let v = y0; v <= y1; v += 5) {
    ctx.strokeStyle = "#2c2c2a";
    ctx.lineWidth = 1;
    ctx.beginPath(); ctx.moveTo(M.l, Y(v)); ctx.lineTo(cssW - M.r, Y(v)); ctx.stroke();
    ctx.fillStyle = "#898781";
    ctx.textAlign = "right"; ctx.textBaseline = "middle";
    ctx.fillText(String(v), M.l - 5, Y(v));
  }
  // x labels: every 5 years
  ctx.textAlign = "center"; ctx.textBaseline = "top";
  for (let i = 0; i < t.length; i++) {
    const yr = t[i].slice(0, 4);
    if (+yr % 5 === 0 && (i === 0 || t[i - 1].slice(0, 4) !== yr)) {
      ctx.fillStyle = "#898781";
      ctx.fillText(yr, X(i), M.t + H + 5);
    }
  }
  // series line (validated dark-mode blue, 2px)
  ctx.strokeStyle = "#3987e5";
  ctx.lineWidth = 2;
  ctx.lineJoin = "round";
  ctx.beginPath();
  let started = false;
  for (let i = 0; i < moc.length; i++) {
    if (moc[i] == null) { started = false; continue; }
    if (!started) { ctx.moveTo(X(i), Y(moc[i])); started = true; }
    else ctx.lineTo(X(i), Y(moc[i]));
  }
  ctx.stroke();

  // hover: crosshair + tooltip
  const tip = document.getElementById("amoc-tooltip");
  canvas.onmousemove = (e) => {
    const rect = canvas.getBoundingClientRect();
    const px = e.clientX - rect.left;
    const i = Cesium.Math.clamp(Math.round(((px - M.l) / W) * (t.length - 1)), 0, t.length - 1);
    if (moc[i] == null) { tip.classList.add("hidden"); return; }
    // redraw base then crosshair
    drawAmocChartStatic();
    ctx.strokeStyle = "#52514e";
    ctx.lineWidth = 1;
    ctx.beginPath(); ctx.moveTo(X(i), M.t); ctx.lineTo(X(i), M.t + H); ctx.stroke();
    ctx.fillStyle = "#3987e5";
    ctx.beginPath(); ctx.arc(X(i), Y(moc[i]), 3.5, 0, Math.PI * 2); ctx.fill();
    tip.textContent = `${t[i]} · ${moc[i].toFixed(1)} Sv`;
    tip.style.left = `${Math.min(Math.max(px - 40, 4), cssW - 110)}px`;
    tip.classList.remove("hidden");
  };
  canvas.onmouseleave = () => { tip.classList.add("hidden"); drawAmocChartStatic(); };

  function drawAmocChartStatic() {
    ctx.clearRect(0, 0, cssW, cssH);
    for (let v = y0; v <= y1; v += 5) {
      ctx.strokeStyle = "#2c2c2a"; ctx.lineWidth = 1;
      ctx.beginPath(); ctx.moveTo(M.l, Y(v)); ctx.lineTo(cssW - M.r, Y(v)); ctx.stroke();
      ctx.fillStyle = "#898781"; ctx.textAlign = "right"; ctx.textBaseline = "middle";
      ctx.fillText(String(v), M.l - 5, Y(v));
    }
    ctx.textAlign = "center"; ctx.textBaseline = "top";
    for (let i = 0; i < t.length; i++) {
      const yr = t[i].slice(0, 4);
      if (+yr % 5 === 0 && (i === 0 || t[i - 1].slice(0, 4) !== yr)) {
        ctx.fillStyle = "#898781";
        ctx.fillText(yr, X(i), M.t + H + 5);
      }
    }
    ctx.strokeStyle = "#3987e5"; ctx.lineWidth = 2; ctx.lineJoin = "round";
    ctx.beginPath();
    let s2 = false;
    for (let i = 0; i < moc.length; i++) {
      if (moc[i] == null) { s2 = false; continue; }
      if (!s2) { ctx.moveTo(X(i), Y(moc[i])); s2 = true; }
      else ctx.lineTo(X(i), Y(moc[i]));
    }
    ctx.stroke();
  }
}

/* ------------------------------------------------------------------ catalog */

let CATALOG = [];

async function loadCatalog() {
  const res = await fetch("data/catalog.json");
  const json = await res.json();
  CATALOG = json.records || [];

  const domSel = document.getElementById("catalog-domain");
  [...new Set(CATALOG.map((r) => r.domain))].sort().forEach((d) => {
    const o = document.createElement("option");
    o.value = d;
    o.textContent = d;
    domSel.appendChild(o);
  });

  for (const el of ["catalog-search", "catalog-domain", "filter-amoc", "filter-globe"]) {
    document.getElementById(el).addEventListener("input", renderCatalog);
  }
  renderCatalog();
}

function renderCatalog() {
  const q = document.getElementById("catalog-search").value.toLowerCase();
  const dom = document.getElementById("catalog-domain").value;
  const amocOnly = document.getElementById("filter-amoc").checked;
  const globeOnly = document.getElementById("filter-globe").checked;

  const hits = CATALOG.filter((r) => {
    if (dom && r.domain !== dom) return false;
    if (amocOnly && !r.amoc) return false;
    if (globeOnly && !r.globe) return false;
    if (!q) return true;
    return (r.name + " " + r.provider + " " + r.variables + " " + r.notes + " " + (r.subdomain || ""))
      .toLowerCase()
      .includes(q);
  });

  document.getElementById("catalog-count").textContent =
    `${hits.length} of ${CATALOG.length} datasets`;

  const list = document.getElementById("catalog-list");
  list.innerHTML = hits
    .slice(0, 150)
    .map(
      (r) => `
    <div class="cat-item">
      <div class="cat-name"><a href="${r.url}" target="_blank" rel="noopener">${r.name}</a>
        ${r.amoc ? '<span class="badge amoc">AMOC</span>' : ""}
        ${r.globe ? '<span class="badge globe">globe</span>' : ""}
      </div>
      <div class="cat-provider">${r.provider} · ${r.subdomain || r.domain} · ${r.temporal}</div>
      <div class="cat-note">${r.notes || ""}</div>
    </div>`
    )
    .join("");
}

/* --------------------------------------------------------------------- tabs */

const tabs = { layers: "panel-layers", amoc: "panel-amoc", catalog: "panel-catalog", about: "panel-about" };
for (const t of Object.keys(tabs)) {
  document.getElementById(`tab-${t}`).addEventListener("click", () => {
    for (const [k, panel] of Object.entries(tabs)) {
      document.getElementById(panel).classList.toggle("hidden", k !== t);
      document.getElementById(`tab-${k}`).classList.toggle("active", k === t);
    }
    if (t === "amoc") loadAmoc();
  });
}

/* --------------------------------------------------------------------- init */

buildLayerPanel();
updateLegends();
loadStations();
loadCatalog();

/* Test hook: stable handle for the Playwright suite (tests/) — not a public API. */
window.__earth = {
  viewer,
  parseColormap,
  parseColormapEntries,
  periodSampleDates,
  deltaColor,
  SSTDeltaProvider,
  state,
  pointLayers,
  GIBS_LAYERS,
  GIBSGeographicTilingScheme,
  compareDate,
  get stations() { return stationsDs; },
  get rapid() { return rapidData; },
  get catalog() { return CATALOG; },
};

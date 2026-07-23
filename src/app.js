/* earth — open climate data on a globe
 * CesiumJS + NASA GIBS (zero API keys). MIT licensed.
 */
"use strict";

/* ---------------------------------------------------------------- GIBS setup */

const GIBS_URL =
  "https://gibs.earthdata.nasa.gov/wmts/epsg4326/best/" +
  "{layer}/default/{time}/{tms}/{TileMatrix}/{TileRow}/{TileCol}.{ext}";

/* GIBS EPSG:4326 uses a non-standard tile pyramid (level 0 = 2x1 partial 512px
 * tiles spanning 288°, level 1 = 3x2, ...). Cesium's default GeographicTilingScheme
 * assumes a power-of-two pyramid, so we implement the GIBS scheme explicitly.
 * Degrees-per-pixel at level L is 0.5625 / 2^L for every GIBS 4326 matrix set;
 * the sets differ only in how many levels they have (250m: 9, 500m: 8, 1km: 7, 2km: 6).
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
    const east = Math.min(west + span, 180);
    const south = Math.max(north - span, -90);
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
    const rect = this._rectangle;
    if (!Cesium.Rectangle.contains(rect, position)) return undefined;
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
    layer: "VIIRS_SNPP_CorrectedReflectance_TrueColor",
    title: "True color (VIIRS, daily)",
    ext: "jpg", tms: "250m", maxLevel: 8,
    start: "2015-11-24", timed: true, on: false,
    meta: "Daily global mosaic, ~3 h latency",
  },
  {
    id: "sst",
    layer: "GHRSST_L4_MUR_Sea_Surface_Temperature",
    title: "Sea surface temperature (MUR 1 km)",
    ext: "png", tms: "1km", maxLevel: 6,
    start: "2002-06-01", timed: true, on: true,
    meta: "GHRSST L4 analysis — watch the North Atlantic cold blob",
  },
  {
    id: "sst-anom",
    layer: "GHRSST_L4_MUR25_Sea_Surface_Temperature_Anomalies",
    title: "SST anomalies (MUR 25 km)",
    ext: "png", tms: "2km", maxLevel: 5,
    start: "2002-09-01", timed: true, on: false,
    meta: "Anomaly vs climatology — AMOC fingerprint region",
  },
  {
    id: "precip",
    layer: "IMERG_Precipitation_Rate",
    title: "Precipitation rate (IMERG)",
    ext: "png", tms: "2km", maxLevel: 5,
    start: "2000-06-01", timed: true, on: false,
    meta: "GPM merged precipitation",
  },
  {
    id: "seaice",
    layer: "AMSRU2_Sea_Ice_Concentration_12km",
    title: "Sea ice concentration (AMSR2)",
    ext: "png", tms: "2km", maxLevel: 5,
    start: "2012-07-02", timed: true, on: false,
    meta: "Passive-microwave, both poles (ends when mission data lags)",
  },
  {
    id: "snow",
    layer: "MODIS_Terra_NDSI_Snow_Cover",
    title: "Snow cover (MODIS NDSI)",
    ext: "png", tms: "500m", maxLevel: 7,
    start: "2000-02-24", timed: true, on: false,
    meta: "Daily NDSI snow cover",
  },
  {
    id: "aod",
    layer: "MODIS_Combined_Value_Added_AOD",
    title: "Aerosol optical depth (MODIS)",
    ext: "png", tms: "2km", maxLevel: 5,
    start: "2017-04-19", timed: true, on: false,
    meta: "Smoke, dust and haze",
  },
  {
    id: "nightlights",
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
  homeButton: true,
  fullscreenButton: true,
  infoBox: true,
  selectionIndicator: true,
});
viewer.scene.globe.enableLighting = false;
viewer.scene.skyAtmosphere.show = true;

/* ------------------------------------------------------------ layer control */

const state = { date: defaultDate(), layers: {} }; // id -> {cfg, imageryLayer, alpha}

function defaultDate() {
  const d = new Date(Date.now() - 2 * 864e5); // two days ago: safely available on GIBS
  return d.toISOString().slice(0, 10);
}

function addLayer(cfg) {
  const provider = gibsProvider(cfg, state.date);
  const layer = viewer.imageryLayers.addImageryProvider(provider);
  layer.alpha = state.layers[cfg.id]?.alpha ?? 1.0;
  state.layers[cfg.id] = { cfg, layer, alpha: layer.alpha };
}

function removeLayer(id) {
  const entry = state.layers[id];
  if (entry?.layer) viewer.imageryLayers.remove(entry.layer, true);
  if (entry) entry.layer = null;
}

function refreshTimedLayers() {
  for (const [id, entry] of Object.entries(state.layers)) {
    if (entry.layer && entry.cfg.timed) {
      removeLayer(id);
      addLayer(entry.cfg);
    }
  }
}

function buildLayerPanel() {
  const list = document.getElementById("layer-list");
  for (const cfg of GIBS_LAYERS) {
    const div = document.createElement("div");
    div.className = "layer-item";
    div.innerHTML = `
      <label><input type="checkbox" data-id="${cfg.id}" ${cfg.on ? "checked" : ""}/> ${cfg.title}</label>
      <div class="meta">${cfg.meta}${cfg.timed ? ` · from ${cfg.start}` : ""}</div>
      <input type="range" min="0" max="100" value="100" data-alpha="${cfg.id}"
             ${cfg.on ? "" : "style='display:none'"} title="opacity"/>`;
    list.appendChild(div);
    if (cfg.on) addLayer(cfg);
  }

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
  });

  list.addEventListener("input", (e) => {
    const id = e.target.getAttribute("data-alpha");
    if (!id) return;
    const entry = state.layers[id];
    if (entry?.layer) {
      entry.alpha = e.target.value / 100;
      entry.layer.alpha = entry.alpha;
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
    destination: Cesium.Cartesian3.fromDegrees(-40, 40, 1.35e7),
    duration: 2.2,
  });
});

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

const tabs = { layers: "panel-layers", catalog: "panel-catalog", about: "panel-about" };
for (const t of Object.keys(tabs)) {
  document.getElementById(`tab-${t}`).addEventListener("click", () => {
    for (const [k, panel] of Object.entries(tabs)) {
      document.getElementById(panel).classList.toggle("hidden", k !== t);
      document.getElementById(`tab-${k}`).classList.toggle("active", k === t);
    }
  });
}

/* --------------------------------------------------------------------- init */

buildLayerPanel();
loadStations();
loadCatalog();

viewer.camera.setView({
  destination: Cesium.Cartesian3.fromDegrees(-30, 30, 2.2e7),
});

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
    deltaRange: 4,
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
    deltaRange: 3,
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
    // The daily product is a daily-MEAN rate (not an instant), so averaging N
    // of them gives the mean rate over the window — the same operation GPCP
    // performs at monthly scale. transparentZero: in IMERG tiles transparent
    // means "below 0.1 mm/hr", i.e. essentially no rain — NOT "unobserved" as
    // in clear-sky products — so the window mean must count those pixels as 0.
    // Excluding them would compute "mean rate on rainy days" (biased high
    // everywhere it rained once). ratioRange: log-distributed field, so
    // computed comparison renders log(mean_now/mean_past) — a ×-fold ratio,
    // saturating at ×8 — rather than an absolute difference, which would be
    // dominated by the log palette's value-proportional quantization.
    aggregable: true, transparentZero: true, ratioRange: 8,
    title: "Precipitation rate (GPM IMERG V07)",
    ext: "png", tms: "2km", maxLevel: 5,
    start: "2000-06-01", timed: true, on: false,
    meta: "GPM IMERG V07 daily merged precipitation (mm/hr)",
  },
  {
    id: "precip-30min",
    colormap: "https://gibs.earthdata.nasa.gov/colormaps/v1.3/GPM_Precipitation_Rate.xml",
    legend: "https://gibs.earthdata.nasa.gov/legends/GPM_Precipitation_Rate_H.svg",
    doc: "https://gpm.nasa.gov/data/imerg",
    layer: "IMERG_Precipitation_Rate_30min",
    // Neither aggregable nor differenceable: each frame is one half-hour
    // snapshot, and a multi-day window sampling ~12 arbitrary instants is not
    // an average of anything physical. Its role is intra-day: step through a
    // single day's storms with the ±30m time stepper (state.timeMin). For
    // multi-day rain, the daily layer above aggregates soundly.
    title: "Precipitation rate (IMERG V07, 30-min)",
    ext: "png", tms: "2km", maxLevel: 5,
    start: "2000-06-01", timed: true, subDaily: true, on: false,
    meta: "GPM IMERG V07 half-hourly rate — step through the day with ±30m",
  },
  {
    id: "seaice",
    deltaRange: 50,
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
    deltaRange: 50,  // NDSI %, snow-line advance/retreat between dates
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
    aggregable: true,  // mean AOD over a window is standard; day-vs-day differencing is noise
    ratioRange: 4,     // log-ish field → computed comparison is a ×-fold ratio of window means
    title: "Aerosol optical depth (MODIS)",
    ext: "png", tms: "2km", maxLevel: 5,
    start: "2017-04-19", timed: true, on: false,
    meta: "Smoke, dust and haze",
  },
  {
    id: "lst",
    deltaRange: 10,  // K, land skin-temperature change
    colormap: "https://gibs.earthdata.nasa.gov/colormaps/v1.3/MODIS_Land_Surface_Temp.xml",
    legend: "https://gibs.earthdata.nasa.gov/legends/MODIS_Land_Surface_Temp_H.svg",
    doc: "https://lpdaac.usgs.gov/products/mod11a1v061/",
    layer: "MODIS_Terra_Land_Surface_Temp_Day",
    title: "Land surface temperature (MODIS)",
    ext: "png", tms: "1km", maxLevel: 6,
    start: "2022-10-23", timed: true, on: false,
    meta: "Daytime land skin temperature (K) — the actual temperature of the ground",
  },
  {
    id: "soilmoisture",
    colormap: "https://gibs.earthdata.nasa.gov/colormaps/v1.3/AMSR_Soil_Moisture.xml",
    legend: "https://gibs.earthdata.nasa.gov/legends/AMSR_Soil_Moisture_H.svg",
    doc: "https://nsidc.org/data/au_land",
    layer: "AMSRU2_Soil_Moisture_NPD_Day",
    // Swathy like AOD/LST → averaging fills gaps; day-vs-day differencing
    // would mostly compare swath coverage, not soil. GIBS stops serving
    // tiles at 2025-09 (endTime clamp) — the hover card says so.
    aggregable: true,
    endTime: "2025-09-01",
    title: "Soil moisture (AMSR2)",
    ext: "png", tms: "2km", maxLevel: 5,
    start: "2012-07-24", timed: true, on: false,
    meta: "Top-centimetre soil water from passive microwave · tiles end 2025-09",
  },
  {
    id: "ndvi",
    colormap: "https://gibs.earthdata.nasa.gov/colormaps/v1.3/MODIS_L3_NDVI.xml",
    legend: "https://gibs.earthdata.nasa.gov/legends/MODIS_L3_NDVI_H.svg",
    doc: "https://lpdaac.usgs.gov/products/mod13a3v061/",
    layer: "MODIS_Terra_L3_NDVI_Monthly",
    deltaRange: 0.3,  // greening/browning between months or years is THE standard NDVI use
    title: "Vegetation index (MODIS NDVI, monthly)",
    ext: "png", tms: "1km", maxLevel: 6,
    start: "2000-03-01", timed: true, monthly: true, on: false,
    meta: "Monthly vegetation greenness (0–1)",
  },
  {
    id: "grace",
    colormap: "https://gibs.earthdata.nasa.gov/colormaps/v1.3/GRACE_Tellus_Liquid_Water_Equivalent_Thickness_Mascon_CRI.xml",
    legend: "https://gibs.earthdata.nasa.gov/legends/GRACE_Tellus_Liquid_Water_Equivalent_Thickness_Mascon_CRI_H.svg",
    doc: "https://grace.jpl.nasa.gov/data/get-data/monthly-mass-grids-land/",
    layer: "GRACE_Tellus_Liquid_Water_Equivalent_Thickness_Mascon_CRI",
    // Already an anomaly field (cm of water vs the 2004-09 baseline);
    // differencing two months = storage CHANGE, the quantity groundwater
    // studies actually use. GIBS tiles end 2022-07; 2017-18 has the
    // GRACE→GRACE-FO mission gap (blank months).
    deltaRange: 15,
    endTime: "2022-07-01",
    title: "Water storage anomaly (GRACE)",
    ext: "png", tms: "2km", maxLevel: 5,
    start: "2002-08-01", timed: true, monthly: true, on: false,
    meta: "Total water mass vs 2004–09 baseline, ~300 km blur · tiles end 2022-07",
  },
  {
    id: "chlor",
    colormap: "https://gibs.earthdata.nasa.gov/colormaps/v1.3/MODIS_Chlorophyll.xml",
    legend: "https://gibs.earthdata.nasa.gov/legends/MODIS_Chlorophyll_H.svg",
    doc: "https://oceancolor.gsfc.nasa.gov/",
    layer: "OCI_PACE_Chlorophyll_a",
    aggregable: true,  // time-averaging fills swath/cloud gaps
    ratioRange: 4,     // log-normal-ish field → compare as ×-fold ratio of window means, not absolute Δ
    title: "Chlorophyll-a (NASA Ocean Color, PACE)",
    ext: "png", tms: "1km", maxLevel: 6,
    start: "2024-02-25", timed: true, on: false,
    meta: "PACE/OCI ocean-colour chlorophyll — phytoplankton, log mg/m³",
  },
  {
    id: "salinity",
    colormap: "https://gibs.earthdata.nasa.gov/colormaps/v1.3/SMAP_Sea_Surface_Salinity.xml",
    legend: "https://gibs.earthdata.nasa.gov/legends/SMAP_Sea_Surface_Salinity_H.svg",
    doc: "https://www.catds.fr/",
    layer: "SMAP_L3_Sea_Surface_Salinity_CAP_Monthly",
    deltaRange: 1.5,  // PSU, freshening/salinification between dates
    title: "Sea surface salinity (SMAP, monthly)",
    ext: "png", tms: "2km", maxLevel: 5,
    start: "2015-04-01", timed: true, monthly: true, on: false,
    meta: "SMAP L-band salinity (PSU) — same quantity as SMOS/CATDS · monthly composite; 2024 has a mission data gap",
  },
  {
    id: "ssh-anom",
    colormap: "https://gibs.earthdata.nasa.gov/colormaps/v1.3/MEaSUREs_Sea_Surface_Height_Anomalies.xml",
    legend: "https://gibs.earthdata.nasa.gov/legends/MEaSUREs_Sea_Surface_Height_Anomalies_H.svg",
    doc: "https://podaac.jpl.nasa.gov/dataset/SEA_SURFACE_HEIGHT_ALT_GRIDS_L4_2SATS_5DAY_6THDEG_V_JPL2205",
    layer: "JPL_MEaSUREs_L4_Sea_Surface_Height_Anomalies",
    // The ocean's pressure gauge: differencing two epochs = local sea-level
    // change. 5-day cadence with two epoch anchors (the product was
    // re-anchored in 2017) — snap5d floors any date to a valid epoch.
    // GIBS tiles end 2019-01; the altimetry record itself continues
    // (see the Sea level tab for the global mean).
    deltaRange: 0.15,
    endTime: "2019-01-17",
    snap5d: ["1992-09-30", "2017-10-29"],
    title: "Sea surface height anomaly (altimetry)",
    ext: "png", tms: "2km", maxLevel: 5,
    start: "1992-09-30", timed: true, on: false,
    meta: "Sea level vs the mean sea surface, 5-day · tiles end 2019-01",
  },
  {
    id: "ceres",
    colormap: "https://gibs.earthdata.nasa.gov/colormaps/v1.3/CERES_EBAF_TOA_Net_Flux_All_Sky_Monthly.xml",
    legend: "https://gibs.earthdata.nasa.gov/legends/CERES_EBAF_TOA_Net_Flux_All_Sky_Monthly_H.svg",
    doc: "https://ceres.larc.nasa.gov/data/",
    layer: "CERES_EBAF_TOA_Net_Flux_All_Sky_Monthly",
    // Net absorbed energy at the top of the atmosphere — the pixel's energy
    // budget, the forcing behind everything else. Continuous W/m² field:
    // year-over-year differencing is sound. GIBS tiles end 2018-10.
    deltaRange: 50,
    endTime: "2018-10-01",
    title: "Energy balance (CERES net flux, monthly)",
    ext: "png", tms: "2km", maxLevel: 5,
    start: "2000-03-01", timed: true, monthly: true, on: false,
    meta: "Net radiation in minus out at top of atmosphere · tiles end 2018-10",
  },
  {
    id: "gpcp",
    grid: true, gridFile: "data/gpcp.json",
    ramp: "precip", vmin: 0, vmax: 3000, units: "mm/yr", maxLevel: 6,
    doc: "https://psl.noaa.gov/data/gridded/data.gpcp.html",
    title: "Precipitation climatology (GPCP v2.3)",
    meta: "Global mean-annual precipitation, 2.5° (NOAA GPCP)",
    on: false,
  },
  {
    id: "oisst",
    grid: true, gridFile: "data/oisst.json",
    ramp: "sst", vmin: -2, vmax: 32, units: "°C", maxLevel: 6,
    doc: "https://psl.noaa.gov/data/gridded/data.noaa.oisst.v2.highres.html",
    title: "SST climatology (OISST v2.1)",
    meta: "NOAA OI SST 1991–2020 mean, 0.25° → 1°",
    on: false,
  },
  {
    id: "eobs",
    grid: true, gridFile: "data/eobs.json", bounds: [-40.375, 25.375, 75.375, 75.375],
    ramp: "precip", vmin: 0, vmax: 2500, units: "mm/yr", maxLevel: 7,
    doc: "https://surfobs.climate.copernicus.eu/dataaccess/access_eobs.php",
    title: "Precipitation climatology (E-OBS v31, Europe)",
    meta: "European 0.25° gridded observations — regional (land only)",
    on: false,
  },
  {
    id: "meteoswiss",
    grid: true, gridFile: "data/meteoswiss.json", bounds: [5.761, 45.689, 10.692, 47.882],
    ramp: "precip", vmin: 0, vmax: 2500, units: "mm/yr", maxLevel: 9,
    doc: "https://opendatadocs.meteoswiss.ch/",
    title: "Precipitation normal (MeteoSwiss, Switzerland)",
    meta: "Swiss 1991–2020 precipitation normal, ~2 km — regional",
    on: false,
  },
  {
    id: "currents",
    grid: true, gridFile: "data/currents.json", snapshotGrid: true,
    ramp: "precip", vmin: 0, vmax: 1.5, units: "m/s", maxLevel: 6,
    doc: "https://data.marine.copernicus.eu/product/GLOBAL_MULTIYEAR_PHY_001_030/description",
    title: "Surface current speed (GLORYS)",
    meta: "Monthly-mean ocean current speed — the Gulf Stream drawn by physics",
    on: false,
  },
  {
    id: "mld",
    grid: true, gridFile: "data/mld.json", snapshotGrid: true,
    ramp: "precip", vmin: 0, vmax: 500, units: "m", maxLevel: 6,
    doc: "https://data.marine.copernicus.eu/product/GLOBAL_MULTIYEAR_PHY_001_030/description",
    title: "Mixed-layer depth (GLORYS)",
    meta: "How deep the surface ocean is stirred — deep winter mixing is where AMOC water forms",
    on: false,
  },
  {
    id: "argo-t300",
    grid: true, gridFile: "data/argo_t300.json", snapshotGrid: true,
    ramp: "anom", vmin: -2, vmax: 2, units: "°C", maxLevel: 6,
    doc: "https://sio-argo.ucsd.edu/RG_Climatology.html",
    title: "Subsurface temp anomaly (300 m, Argo)",
    meta: "Latest month vs 2004–18 same-month mean — subsurface marine heatwaves",
    on: false,
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

// GIBS TIME value for a layer: monthly products must be requested at the first
// of the month (a mid-month date returns a blank tile), sub-daily/daily use the
// raw date, and untimed layers use their fixed snapshot. The current month's
// composite is still accumulating and not yet published (GIBS 404s → an
// invisible layer), so a date in the current month falls back to the previous
// complete month.
function gibsTime(cfg, dateStr) {
  if (!cfg.timed) return cfg.fixedTime || "default";
  // Some archives stop being served as tiles before today (GRACE 2022-07,
  // CERES 2018-10, SSH anomalies 2019-01, AMSR2 soil moisture 2025-09): any
  // later date clamps to the last served one, so the layer shows its final
  // state instead of silently blanking. The hover card states the end date.
  if (cfg.endTime && dateStr > cfg.endTime) dateStr = cfg.endTime;
  if (cfg.monthly) {
    let d = dateStr.slice(0, 8) + "01";
    const currentMonth = defaultDate().slice(0, 8) + "01";
    if (!cfg.endTime && d >= currentMonth) {
      const [y, m] = d.split("-").map(Number);
      d = m === 1 ? `${y - 1}-12-01` : `${y}-${String(m - 1).padStart(2, "0")}-01`;
    }
    return d;
  }
  // 5-day products serve only exact epoch dates; floor to the nearest one.
  // Two anchors because the MEaSUREs product was re-anchored mid-record.
  if (cfg.snap5d) {
    const epoch = cfg.snap5d.reduce((best, e) => (dateStr >= e ? e : best), cfg.snap5d[0]);
    const steps = Math.floor((Date.parse(dateStr) - Date.parse(epoch)) / (5 * 864e5));
    return new Date(Date.parse(epoch) + Math.max(0, steps) * 5 * 864e5)
      .toISOString().slice(0, 10);
  }
  // Sub-daily layers get a full timestamp. GIBS serves distinct tiles per
  // half-hour (verified: TIME=...T13:00:00Z and ...T13:30:00Z differ); a bare
  // date resolves to 00:00. state.timeMin is the UI's time-of-day in minutes.
  if (cfg.subDaily && state.timeMin) {
    const h = String(Math.floor(state.timeMin / 60)).padStart(2, "0");
    const m = String(state.timeMin % 60).padStart(2, "0");
    return `${dateStr}T${h}:${m}:00Z`;
  }
  return dateStr;
}

function gibsProvider(cfg, dateStr) {
  const time = gibsTime(cfg, dateStr);
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

// The base Blue-Marble layer. A manual "Grayscale globe" toggle desaturates it so
// coloured overlays (e.g. a blue/red difference) stand out instead of blue-on-blue.
const baseImageryLayer = viewer.imageryLayers.get(0);
function updateBaseAppearance() {
  const gray = document.getElementById("toggle-grayscale")?.checked;
  baseImageryLayer.saturation = gray ? 0.0 : 1.0;
  baseImageryLayer.brightness = gray ? 0.6 : 1.0;
}

/* Zoom gestures: mouse wheel, touch pinch, AND trackpad pinch.
 * Browsers report a MacBook-style trackpad pinch as a wheel event with
 * ctrlKey set, which Cesium ignores unless registered explicitly. */
const sscc = viewer.scene.screenSpaceCameraController;
sscc.minimumZoomDistance = 20000;  // allow getting closer (20 km)
// Keep native touch-pinch zoom; drive wheel/trackpad zoom ourselves (below) so
// one gesture covers far more distance than Cesium's default.
sscc.zoomEventTypes = [Cesium.CameraEventType.PINCH];

// Strong, distance-proportional wheel zoom. deltaY is normalised across devices;
// trackpad pinch (ctrlKey) gets extra gain since its deltas are tiny. The amount
// is a fraction of the current camera height, capped so it can't shoot through
// the globe — so it's fast far out and still controllable up close.
// Sign convention matches standard map/trackpad behaviour: scrolling up, or
// spreading two fingers apart on a trackpad (negative deltaY), zooms IN;
// scrolling down, or pinching fingers together (positive deltaY), zooms OUT.
function wheelZoom(e) {
  e.preventDefault();
  let dy = e.deltaY;
  if (e.deltaMode === 1) dy *= 16;            // lines → ~px
  else if (e.deltaMode === 2) dy *= 400;      // pages → ~px
  const gain = e.ctrlKey ? 0.025 : 0.008;     // trackpad pinch vs mouse wheel
  const frac = Cesium.Math.clamp(dy * gain, -0.85, 0.85);
  const amount = cameraHeight() * frac;
  if (amount > 0) viewer.camera.zoomOut(amount);
  else if (amount < 0) viewer.camera.zoomIn(-amount);
}
viewer.scene.canvas.addEventListener("wheel", wheelZoom, { passive: false });
window.__wheelZoom = wheelZoom; // exposed for tests

const HOME = { lon: -30, lat: 28, height: 1.5e7 };
viewer.camera.setView({
  destination: Cesium.Cartesian3.fromDegrees(HOME.lon, HOME.lat, HOME.height),
});

/* ---------------------------------------------------------------- zoom controls */

function cameraHeight() {
  return viewer.camera.positionCartographic.height;
}
document.getElementById("zoom-in").addEventListener("click", () => {
  viewer.camera.zoomIn(cameraHeight() * 0.6);
});
document.getElementById("zoom-out").addEventListener("click", () => {
  viewer.camera.zoomOut(cameraHeight() * 1.5);
});
document.getElementById("zoom-home").addEventListener("click", () => {
  viewer.camera.flyTo({
    destination: Cesium.Cartesian3.fromDegrees(HOME.lon, HOME.lat, HOME.height),
    duration: 1.2,
  });
});

/* ------------------------------------------------- SST ensemble (mean/spread) */

/* Independent, key-free GHRSST L4 analyses that share the GIBS SST colormap,
 * so each tile can be inverted to °C with one LUT and combined per pixel.
 * MUR (JPL) + GAMSSA (Australian BoM) cover recent dates; OISST (NOAA) adds a
 * third member for the pre-2020 era. MUR25 is excluded — it is MUR regridded,
 * not an independent estimate. The provider uses whichever members return a
 * tile for the chosen date and needs at least two to render. */
const SST_ENSEMBLE_MEMBERS = [
  { name: "MUR (JPL)", layer: "GHRSST_L4_MUR_Sea_Surface_Temperature", tms: "1km" },
  { name: "OISST (NOAA)", layer: "GHRSST_L4_AVHRR-OI_Sea_Surface_Temperature", tms: "2km" },
  { name: "GAMSSA (BoM)", layer: "GHRSST_L4_GAMSSA_GDS2_Sea_Surface_Temperature", tms: "2km" },
];
const SPREAD_MAX = 2.0; // °C, top of the spread colour scale

// Forward colour lookup (value → rgb) built from the GHRSST colormap.
let sstForwardPromise = null;
function getSstForward() {
  if (!sstForwardPromise) {
    sstForwardPromise = getColormapEntries(
      "https://gibs.earthdata.nasa.gov/colormaps/v1.3/GHRSST_Sea_Surface_Temperature.xml"
    ).then((cm) => {
      const e = cm.entries;
      return (v) => {
        if (v <= e[0].lo) return e[0].rgb;
        if (v >= e[e.length - 1].hi) return e[e.length - 1].rgb;
        let lo = 0, hi = e.length - 1;
        while (lo < hi) {
          const mid = (lo + hi) >> 1;
          if (v < e[mid].lo) hi = mid - 1;
          else if (v >= e[mid].hi) lo = mid + 1;
          else return e[mid].rgb;
        }
        return e[lo].rgb;
      };
    });
  }
  return sstForwardPromise;
}

// Sequential ramp for spread (°C): transparent → cyan → yellow → magenta.
function spreadColor(s) {
  if (!(s > 0.02)) return [0, 0, 0, 0];
  const t = Cesium.Math.clamp(s / SPREAD_MAX, 0, 1);
  const stops = [
    [0.0, [8, 48, 107]], [0.35, [33, 145, 140]],
    [0.7, [253, 231, 37]], [1.0, [240, 59, 46]],
  ];
  let a = stops[0], b = stops[stops.length - 1];
  for (let i = 0; i < stops.length - 1; i++) {
    if (t >= stops[i][0] && t <= stops[i + 1][0]) { a = stops[i]; b = stops[i + 1]; break; }
  }
  const f = (t - a[0]) / (b[0] - a[0] || 1);
  const c = a[1].map((av, i) => Math.round(av + (b[1][i] - av) * f));
  return [c[0], c[1], c[2], Math.round((0.35 + 0.6 * t) * 255)];
}

class SSTEnsembleProvider {
  constructor(members, date, mode) {
    this._members = members;
    this._date = date;
    this._mode = mode; // "mean" | "spread"
    this.tilingScheme = new GIBSGeographicTilingScheme();
    this.rectangle = this.tilingScheme.rectangle;
    this.tileWidth = 512;
    this.tileHeight = 512;
    this.maximumLevel = 5; // limited by the 2 km members
    this.minimumLevel = 0;
    this.errorEvent = new Cesium.Event();
    this.credit = new Cesium.Credit("SST ensemble computed client-side from NASA GIBS (GHRSST L4)");
    this.hasAlphaChannel = true;
    this.ready = true;
  }
  get mode() { return this._mode; }
  getTileCredits() { return undefined; }
  pickFeatures() { return undefined; }
  _url(m, x, y, level) {
    return GIBS_URL
      .replace("{layer}", m.layer).replace("{time}", this._date)
      .replace("{tms}", m.tms).replace("{ext}", "png")
      .replace("{TileMatrix}", level).replace("{TileRow}", y).replace("{TileCol}", x);
  }
  async _tile(m, x, y, level, lut, ctx) {
    try {
      const r = await fetch(this._url(m, x, y, level));
      if (!r.ok) return null;
      const img = await createImageBitmap(await r.blob());
      ctx.clearRect(0, 0, 512, 512);
      ctx.drawImage(img, 0, 0);
      return ctx.getImageData(0, 0, 512, 512).data;
    } catch { return null; }
  }
  async requestImage(x, y, level) {
    const [lut, forward] = await Promise.all([getSstLUT(), getSstForward()]);
    const canvas = document.createElement("canvas");
    canvas.width = 512; canvas.height = 512;
    const ctx = canvas.getContext("2d", { willReadFrequently: true });
    const fields = [];
    for (const m of this._members) {
      const d = await this._tile(m, x, y, level, lut, ctx);
      if (d) fields.push(d);
    }
    const out = ctx.createImageData(512, 512);
    const o = out.data;
    if (fields.length >= 2) {
      const N = 512 * 512;
      for (let p = 0, i = 0; p < N; p++, i += 4) {
        let sum = 0, sumSq = 0, cnt = 0;
        for (const d of fields) {
          if (d[i + 3] === 0) continue;
          const v = lut.get((d[i] << 16) | (d[i + 1] << 8) | d[i + 2]);
          if (v === undefined) continue;
          sum += v; sumSq += v * v; cnt++;
        }
        if (cnt < 2) continue;
        const mean = sum / cnt;
        let rgba;
        if (this._mode === "spread") {
          rgba = spreadColor(Math.sqrt(Math.max(0, sumSq / cnt - mean * mean)));
        } else {
          const c = forward(mean);
          rgba = [c[0], c[1], c[2], 235];
        }
        o[i] = rgba[0]; o[i + 1] = rgba[1]; o[i + 2] = rgba[2]; o[i + 3] = rgba[3];
      }
    }
    ctx.clearRect(0, 0, 512, 512);
    ctx.putImageData(out, 0, 0);
    return canvas;
  }
}

let sstEnsembleLayer = null;
async function updateEnsembleLayer() {
  if (sstEnsembleLayer) {
    viewer.imageryLayers.remove(sstEnsembleLayer, true);
    sstEnsembleLayer = null;
  }
  const on = document.getElementById("toggle-sst-ensemble").checked;
  if (!on) { updateLegends(); return; }
  const mode = document.getElementById("ensemble-mode").value;
  sstEnsembleLayer = viewer.imageryLayers.addImageryProvider(
    new SSTEnsembleProvider(SST_ENSEMBLE_MEMBERS, state.date, mode)
  );
  sstEnsembleLayer.__ensembleMode = mode;
  updateLegends();
}

/* ------------------------------------------------------------ layer control */

const state = {
  date: defaultDate(),
  compareYears: 0,       // comparison offset (0 = not comparing)
  compareMode: "split",  // "split" | "delta" — display mode, orthogonal to the window
  windowDays: 1,         // rolling aggregation window ending at `date` (1 = single day)
  timeMin: 0,            // time of day (UTC minutes) for sub-daily layers, stepped ±30m
  layers: {},
};

function defaultDate() {
  const d = new Date(Date.now() - 2 * 864e5); // two days ago: safely available on GIBS
  return d.toISOString().slice(0, 10);
}

function addDays(dateStr, n) {
  const d = new Date(dateStr + "T00:00:00Z");
  d.setUTCDate(d.getUTCDate() + n);
  return d.toISOString().slice(0, 10);
}

function compareDate() {
  if (!state.compareYears) return null;
  const [y, m, d] = state.date.split("-").map(Number);
  const day = m === 2 && d === 29 ? 28 : d; // leap-day safety
  return `${y - state.compareYears}-${String(m).padStart(2, "0")}-${String(day).padStart(2, "0")}`;
}


/* ------------------------------------------------- computed-delta (SST) mode */

const DELTA_RANGE = 4;               // default ± scale (°C) for the SST legend helpers
const DELTA_COOL = [37, 99, 235];    // negative Δ (less / cooler than N years ago)
const DELTA_WARM = [230, 59, 46];    // positive Δ (more / warmer than N years ago)

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

function deltaColor(d, range = DELTA_RANGE) {
  // diverging: blue = decrease, red = increase; opacity scales with |delta|
  const t = Cesium.Math.clamp(d / range, -1, 1);
  if (Math.abs(d) < range * 0.0125) return [0, 0, 0, 0]; // small dead-zone
  const a = Math.round(Math.min(1, Math.abs(t) + 0.06) * 235);
  const c = t > 0 ? DELTA_WARM : DELTA_COOL;
  return [c[0], c[1], c[2], a];
}

/* Rolling window: sample up to 12 evenly-spaced days over the `windowDays`
 * ending at `endDate` (always the same interval length, independent of the
 * calendar month). windowDays === 1 → the single day. Averaging these samples
 * approximates the mean field over the window from daily GIBS tiles. */
function windowSampleDates(endDate, windowDays) {
  const w = Math.max(1, Math.round(windowDays));
  if (w <= 1) return [endDate];
  const samples = Math.min(12, w);
  const step = (w - 1) / (samples - 1);
  const out = [];
  for (let i = 0; i < samples; i++) out.push(addDays(endDate, -Math.round(i * step)));
  return [...new Set(out)];
}

// Zoom cap: single day → full detail; any averaged window fetches ~12 tiles per
// rendered tile, so cap the level to stay responsive.
function windowMaxLevel(cfg, windowDays) {
  return windowDays <= 1 ? cfg.maxLevel : 4;
}

function windowLabel(windowDays) {
  return windowDays <= 1 ? "single day" : `past ${Math.round(windowDays)} days`;
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

/* Shared helpers for the client-side aggregate/delta providers below. */
function sstFetchUrl(cfg, date, x, y, level) {
  return GIBS_URL
    .replace("{layer}", cfg.layer).replace("{time}", gibsTime(cfg, date))
    .replace("{tms}", cfg.tms).replace("{ext}", cfg.ext)
    .replace("{TileMatrix}", level).replace("{TileRow}", y).replace("{TileCol}", x);
}

/* Forward colour lookup (value → rgb) for ANY layer colormap, cached per URL.
 * The inverse of getValueLut: mean values are painted back through the layer's
 * own palette, so an aggregated layer looks like the original. */
const forwardCache = new Map();
function getForward(url) {
  if (!forwardCache.has(url)) {
    forwardCache.set(url, getColormapEntries(url).then((cm) => {
      if (!cm || cm.entries.length < 2) return null;
      const e = cm.entries;
      return (v) => {
        if (v <= e[0].lo) return e[0].rgb;
        if (v >= e[e.length - 1].hi) return e[e.length - 1].rgb;
        let lo = 0, hi = e.length - 1;
        while (lo < hi) {
          const mid = (lo + hi) >> 1;
          if (v < e[mid].lo) hi = mid - 1;
          else if (v >= e[mid].hi) lo = mid + 1;
          else return e[mid].rgb;
        }
        return e[Math.max(0, lo)].rgb;
      };
    }).catch(() => null));
  }
  return forwardCache.get(url);
}
async function sstFetchBitmap(url) {
  try {
    const r = await fetch(url);
    if (!r.ok) return null;
    return await createImageBitmap(await r.blob());
  } catch {
    return null;
  }
}
/* Mean °C per pixel across a set of sample dates (colormap-inverted). */
async function sstMeanField(cfg, dates, x, y, level, lut, ctx) {
  const N = 512 * 512;
  const sum = new Float32Array(N);
  const cnt = new Uint8Array(N);
  const imgs = await Promise.all(dates.map((d) => sstFetchBitmap(sstFetchUrl(cfg, d, x, y, level))));
  // What a transparent pixel MEANS differs by layer, and the mean must honour
  // it. Default (clear-sky products): transparent = unobserved → exclude the
  // sample, divide by the count of days that had data. transparentZero
  // (precipitation): transparent = below the palette floor ≈ no rain → count
  // it as 0, else the mean is "rate on rainy days only", biased high wherever
  // it rained once. (Only pixels from tiles that loaded are counted either
  // way, so a missing tile never masquerades as a dry day.)
  const zero = !!cfg.transparentZero;
  for (const img of imgs) {
    if (!img) continue;
    ctx.clearRect(0, 0, 512, 512);
    ctx.drawImage(img, 0, 0);
    const d = ctx.getImageData(0, 0, 512, 512).data;
    for (let p = 0, i = 0; p < N; p++, i += 4) {
      if (d[i + 3] === 0) {
        if (zero) cnt[p]++;                       // sum += 0
        continue;
      }
      const v = lut.get((d[i] << 16) | (d[i + 1] << 8) | d[i + 2]);
      if (v === undefined) continue;
      sum[p] += v;
      cnt[p]++;
    }
  }
  return { sum, cnt };
}

/* Colorized per-pixel mean of ANY continuous colormapped layer over the rolling
 * window (used for single-layer display and for each side of a windowed
 * side-by-side comparison). Averaging is per pixel: samples where the pixel is
 * missing (transparent — clouds, night, no retrieval) are simply excluded, and
 * the mean divides by the count of samples that HAD data at that pixel. So a
 * pixel observed on 3 of 12 sampled days shows the mean of those 3; only a
 * pixel observed on none stays empty. For clear-sky products like MODIS land
 * surface temperature this is what fills the daily cloud gaps. */
class AggregateProvider {
  constructor(cfg, endDate, windowDays) {
    this._cfg = cfg;
    // Snap sample dates to what the layer can actually serve (monthly layers →
    // first-of-month) and dedupe, so a 60-day window over a monthly product
    // averages 2-3 distinct months instead of re-counting the same composite.
    this._dates = [...new Set(windowSampleDates(endDate, windowDays).map((d) => gibsTime(cfg, d)))];
    this._window = windowDays;
    this.tilingScheme = new GIBSGeographicTilingScheme();
    this.rectangle = this.tilingScheme.rectangle;
    this.tileWidth = 512;
    this.tileHeight = 512;
    this.maximumLevel = windowMaxLevel(cfg, windowDays);
    this.minimumLevel = 0;
    this.errorEvent = new Cesium.Event();
    this.credit = new Cesium.Credit(
      `${cfg.title} mean over ${windowLabel(windowDays)}, from NASA GIBS`);
    this.hasAlphaChannel = true;
    this.ready = true;
  }
  get window() { return this._window; }
  get layerId() { return this._cfg.id; }
  getTileCredits() { return undefined; }
  pickFeatures() { return undefined; }
  async requestImage(x, y, level) {
    const [vlut, forward, cm] = await Promise.all([
      getValueLut(this._cfg.colormap), getForward(this._cfg.colormap),
      getColormapEntries(this._cfg.colormap)]);
    const canvas = document.createElement("canvas");
    canvas.width = 512; canvas.height = 512;
    const ctx = canvas.getContext("2d", { willReadFrequently: true });
    if (!vlut || !forward) return canvas;
    const f = await sstMeanField(this._cfg, this._dates, x, y, level, vlut.lut, ctx);
    // transparentZero layers keep the source convention on the way OUT too: a
    // mean below the palette floor (e.g. drizzle-of-drizzles < 0.1 mm/hr)
    // renders transparent, exactly as GIBS renders such values — without this,
    // forward() clamps tiny means up to the palette's first colour and the
    // whole ocean tints "light rain".
    const floor = this._cfg.transparentZero && cm?.entries?.length
      ? cm.entries[0].lo : -Infinity;
    const out = ctx.createImageData(512, 512);
    const o = out.data;
    for (let p = 0, i = 0; p < 512 * 512; p++, i += 4) {
      if (f.cnt[p] === 0) continue;
      const v = f.sum[p] / f.cnt[p];
      if (v < floor) continue;
      const c = forward(v);
      o[i] = c[0]; o[i + 1] = c[1]; o[i + 2] = c[2]; o[i + 3] = 235;
    }
    ctx.clearRect(0, 0, 512, 512);
    ctx.putImageData(out, 0, 0);
    return canvas;
  }
}
const SSTAggregateProvider = AggregateProvider;   // back-compat alias

/* Per-pixel difference of two rolling-window means for ANY continuous
 * colormapped layer (SST, SST anomalies, sea ice, …): value(now) − value(past),
 * with the layer's own colormap inverted to physical units and a ±deltaRange
 * diverging scale. */
class DeltaProvider {
  constructor(cfg, dateNow, datePast, windowDays = 1) {
    this._cfg = cfg;
    this._range = cfg.deltaRange || DELTA_RANGE;
    this._window = windowDays;
    this._datesNow = windowSampleDates(dateNow, windowDays);
    this._datesPast = windowSampleDates(datePast, windowDays);
    this.tilingScheme = new GIBSGeographicTilingScheme();
    this.rectangle = this.tilingScheme.rectangle;
    this.tileWidth = 512;
    this.tileHeight = 512;
    this.maximumLevel = windowMaxLevel(cfg, windowDays);
    this.minimumLevel = 0;
    this.errorEvent = new Cesium.Event();
    this.credit = new Cesium.Credit(
      `Δ ${cfg.title} (${windowLabel(windowDays)}) computed client-side from NASA GIBS`
    );
    this.hasAlphaChannel = true;
    this.ready = true;
  }
  get window() { return this._window; }
  get layerId() { return this._cfg.id; }
  getTileCredits() { return undefined; }
  pickFeatures() { return undefined; }
  async requestImage(x, y, level) {
    const vlut = await getValueLut(this._cfg.colormap);
    const canvas = document.createElement("canvas");
    canvas.width = 512;
    canvas.height = 512;
    const ctx = canvas.getContext("2d", { willReadFrequently: true });
    if (!vlut) return canvas;
    const [now, past] = await Promise.all([
      sstMeanField(this._cfg, this._datesNow, x, y, level, vlut.lut, ctx),
      sstMeanField(this._cfg, this._datesPast, x, y, level, vlut.lut, ctx),
    ]);
    const out = ctx.createImageData(512, 512);
    const o = out.data;
    for (let p = 0, i = 0; p < 512 * 512; p++, i += 4) {
      if (now.cnt[p] === 0 || past.cnt[p] === 0) continue;
      const d = now.sum[p] / now.cnt[p] - past.sum[p] / past.cnt[p];
      const [r, g, b, a] = deltaColor(d, this._range);
      o[i] = r; o[i + 1] = g; o[i + 2] = b; o[i + 3] = a;
    }
    ctx.clearRect(0, 0, 512, 512);
    ctx.putImageData(out, 0, 0);
    return canvas;
  }
}

/* Per-pixel ×-fold RATIO of two rolling-window means, for log-distributed
 * fields (precipitation, chlorophyll, aerosol) where an absolute difference is
 * the wrong object twice over: (a) the only access to values is the palette,
 * and a log palette's bins grow in proportion to the value, so the difference
 * of two large near-equal values is mostly quantization error; (b) for
 * log-normal-ish fields the meaningful "change" is multiplicative anyway.
 * log(mean_now / mean_past) is quantization-robust (bin error is a few % OF
 * the value, hence a small constant in log space) and reads naturally as
 * "×2 wetter", "half the plankton". Rendered through the same diverging
 * blue-less / red-more scale as DeltaProvider, saturating at ×ratioRange. */
class RatioProvider {
  constructor(cfg, dateNow, datePast, windowDays = 1) {
    this._cfg = cfg;
    this._range = cfg.ratioRange || 4;
    this._window = windowDays;
    this._datesNow = windowSampleDates(dateNow, windowDays);
    this._datesPast = windowSampleDates(datePast, windowDays);
    this.tilingScheme = new GIBSGeographicTilingScheme();
    this.rectangle = this.tilingScheme.rectangle;
    this.tileWidth = 512;
    this.tileHeight = 512;
    this.maximumLevel = windowMaxLevel(cfg, windowDays);
    this.minimumLevel = 0;
    this.errorEvent = new Cesium.Event();
    this.credit = new Cesium.Credit(
      `${cfg.title} ratio (${windowLabel(windowDays)} means) computed client-side from NASA GIBS`
    );
    this.hasAlphaChannel = true;
    this.ready = true;
  }
  get window() { return this._window; }
  get layerId() { return this._cfg.id; }
  getTileCredits() { return undefined; }
  pickFeatures() { return undefined; }
  async requestImage(x, y, level) {
    const [vlut, cm] = await Promise.all([
      getValueLut(this._cfg.colormap), getColormapEntries(this._cfg.colormap)]);
    const canvas = document.createElement("canvas");
    canvas.width = 512;
    canvas.height = 512;
    const ctx = canvas.getContext("2d", { willReadFrequently: true });
    if (!vlut) return canvas;
    // transparentZero fields have genuine zeros (dry pixels), so the ratio
    // needs a floor: eps = half the palette floor keeps rain-appearing /
    // rain-vanishing finite (dry→drizzle reads as a strong but bounded fold)
    // and makes dry÷dry exactly 1 → the dead zone → transparent.
    const eps = this._cfg.transparentZero && cm?.entries?.length
      ? cm.entries[0].lo / 2 : 0;
    const logRange = Math.log(this._range);
    const [now, past] = await Promise.all([
      sstMeanField(this._cfg, this._datesNow, x, y, level, vlut.lut, ctx),
      sstMeanField(this._cfg, this._datesPast, x, y, level, vlut.lut, ctx),
    ]);
    const out = ctx.createImageData(512, 512);
    const o = out.data;
    for (let p = 0, i = 0; p < 512 * 512; p++, i += 4) {
      if (now.cnt[p] === 0 || past.cnt[p] === 0) continue;
      const d = Math.log((now.sum[p] / now.cnt[p] + eps) / (past.sum[p] / past.cnt[p] + eps));
      const [r, g, b, a] = deltaColor(d, logRange);
      o[i] = r; o[i + 1] = g; o[i + 2] = b; o[i + 3] = a;
    }
    ctx.clearRect(0, 0, 512, 512);
    ctx.putImageData(out, 0, 0);
    return canvas;
  }
}

/* ----------------------------------------------------------- grid overlays */
/* GPCP, E-OBS, OISST and MeteoSwiss have no global tile service, so they ship
 * as a static regular lon/lat grid (data/<id>.json) that GridProvider paints on
 * the fly: for each tile pixel it looks up the nearest grid cell and maps the
 * value through a colour ramp. Regional grids (E-OBS, MeteoSwiss) declare a
 * bounded rectangle; everything outside a grid's coverage stays transparent. */

const RAMPS = {
  // low → high: white → teal → blue → indigo → violet (wetter = deeper)
  precip: [[0, 247, 252, 253], [0.15, 204, 236, 230], [0.35, 123, 204, 196],
           [0.55, 67, 162, 202], [0.75, 37, 78, 155], [1, 84, 39, 143]],
  // diverging anomaly: blue → near-invisible dark slate → red, matching the
  // app's blue=cooler/less, red=warmer/more convention; the dark middle lets
  // near-zero cells fade into the globe instead of painting it white
  anom: [[0, 37, 99, 235], [0.42, 84, 110, 160], [0.5, 45, 51, 63],
         [0.58, 160, 95, 84], [1, 230, 59, 46]],
  // cold → warm thermal ramp for SST
  sst: [[0, 49, 54, 149], [0.25, 116, 173, 209], [0.5, 255, 255, 191],
        [0.75, 244, 109, 67], [1, 165, 0, 38]],
};

function rampColor(name, t) {
  const stops = RAMPS[name] || RAMPS.precip;
  t = Math.max(0, Math.min(1, t));
  for (let i = 1; i < stops.length; i++) {
    if (t <= stops[i][0]) {
      const a = stops[i - 1], b = stops[i];
      const f = (t - a[0]) / (b[0] - a[0] || 1);
      return [Math.round(a[1] + f * (b[1] - a[1])),
              Math.round(a[2] + f * (b[2] - a[2])),
              Math.round(a[3] + f * (b[3] - a[3]))];
    }
  }
  const l = stops[stops.length - 1];
  return [l[1], l[2], l[3]];
}

function sampleGrid(g, lonDeg, latDeg) {
  if (lonDeg < g.west || lonDeg >= g.east || latDeg < g.south || latDeg >= g.north) return null;
  const ix = Math.floor((lonDeg - g.west) / g.dlon);
  const iy = Math.floor((latDeg - g.south) / g.dlat);
  if (ix < 0 || ix >= g.nx || iy < 0 || iy >= g.ny) return null;
  const v = g.values[iy * g.nx + ix];
  return v == null ? null : v;
}

const gridCache = new Map();
function loadGrid(cfg) {
  if (!gridCache.has(cfg.id)) {
    gridCache.set(cfg.id, fetch(cfg.gridFile).then((r) => r.json()).catch(() => null));
  }
  return gridCache.get(cfg.id);
}

class GridProvider {
  constructor(cfg) {
    this._cfg = cfg;
    this.tilingScheme = new Cesium.GeographicTilingScheme();
    this.rectangle = cfg.bounds
      ? Cesium.Rectangle.fromDegrees(cfg.bounds[0], cfg.bounds[1], cfg.bounds[2], cfg.bounds[3])
      : this.tilingScheme.rectangle;
    this.tileWidth = 256;
    this.tileHeight = 256;
    this.maximumLevel = cfg.maxLevel || 6;
    this.minimumLevel = 0;
    this.errorEvent = new Cesium.Event();
    this.credit = new Cesium.Credit(cfg.source || cfg.title);
    this.hasAlphaChannel = true;
    this.ready = true;
  }
  get layerId() { return this._cfg.id; }
  getTileCredits() { return undefined; }
  pickFeatures() { return undefined; }
  async requestImage(x, y, level) {
    const g = await loadGrid(this._cfg);
    const W = this.tileWidth, H = this.tileHeight;
    const canvas = document.createElement("canvas");
    canvas.width = W; canvas.height = H;
    const ctx = canvas.getContext("2d", { willReadFrequently: true });
    if (!g) return canvas;
    const rect = this.tilingScheme.tileXYToRectangle(x, y, level);
    const west = Cesium.Math.toDegrees(rect.west), east = Cesium.Math.toDegrees(rect.east);
    const north = Cesium.Math.toDegrees(rect.north), south = Cesium.Math.toDegrees(rect.south);
    const { vmin, vmax, ramp } = this._cfg;
    const out = ctx.createImageData(W, H);
    const o = out.data;
    for (let j = 0; j < H; j++) {
      const lat = north - ((j + 0.5) / H) * (north - south);
      for (let i = 0; i < W; i++) {
        const lon = west + ((i + 0.5) / W) * (east - west);
        const v = sampleGrid(g, lon, lat);
        if (v == null) continue;
        const c = rampColor(ramp, (v - vmin) / (vmax - vmin));
        const k = (j * W + i) * 4;
        o[k] = c[0]; o[k + 1] = c[1]; o[k + 2] = c[2]; o[k + 3] = 225;
      }
    }
    ctx.putImageData(out, 0, 0);
    return canvas;
  }
}

function addLayer(cfg) {
  const entry = { cfg, layer: null, cmpLayer: null, isDelta: false, isRatio: false,
    isAggregate: false, alpha: state.layers[cfg.id]?.alpha ?? 1.0 };
  const cmp = compareDate();
  const comparing = cmp && cfg.timed;
  const deltaable = cfg.deltaRange != null;              // continuous field with an invertible colormap
  const win = state.windowDays;
  // Rolling-window mean render applies to every layer whose values may be
  // meaningfully averaged: differenceable fields (deltaRange) plus fields
  // flagged aggregable-only (chlorophyll, aerosol — averaging fills gaps, but
  // day-vs-day differencing would be unsound). Essential for clear-sky
  // products where any single day is mostly gaps.
  const canWindow = (deltaable || cfg.aggregable) && !!cfg.colormap;
  const windowed = win > 1 && canWindow && cfg.timed;

  // While an aggregation window is active, everything on screen must actually
  // BE an average. A timed raster that cannot be time-averaged (photographic
  // true colour, instantaneous precipitation) would silently show one day's
  // data under a "past N days" label — so it displays NOTHING instead, and
  // the panel hint explains why. (Untimed composites and the climatology
  // grids stay: they already are long-period averages.)
  if (win > 1 && cfg.timed && !canWindow) {
    entry.suppressed = true;
    state.layers[cfg.id] = entry;
    updateLegends();
    return;
  }

  const add = (provider) => viewer.imageryLayers.addImageryProvider(provider);

  if (cfg.grid) {
    // Static climatology grid painted client-side from data/<id>.json
    entry.layer = add(new GridProvider(cfg));
    entry.layer.alpha = entry.alpha;
    state.layers[cfg.id] = entry;
    updateLegends();
    return;
  }

  if (comparing && state.compareMode === "delta" && deltaable) {
    // Computed per-pixel difference of window means (single-day if win === 1)
    entry.layer = add(new DeltaProvider(cfg, state.date, cmp, win));
    entry.layer.alpha = entry.alpha;
    entry.isDelta = true;
  } else if (comparing && state.compareMode === "delta" && cfg.ratioRange) {
    // Log-distributed field: computed comparison is a ×-fold ratio of means
    entry.layer = add(new RatioProvider(cfg, state.date, cmp, win));
    entry.layer.alpha = entry.alpha;
    entry.isRatio = true;
  } else if (comparing && state.compareMode === "split") {
    // Side-by-side: right = current, left = past. Windowed means for SST, raw tiles otherwise.
    entry.layer = add(windowed ? new SSTAggregateProvider(cfg, state.date, win) : gibsProvider(cfg, state.date));
    entry.layer.alpha = entry.alpha;
    entry.layer.splitDirection = Cesium.SplitDirection.RIGHT;
    entry.cmpLayer = add(windowed ? new SSTAggregateProvider(cfg, cmp, win) : gibsProvider(cfg, cmp));
    entry.cmpLayer.alpha = entry.alpha;
    entry.cmpLayer.splitDirection = Cesium.SplitDirection.LEFT;
    entry.isAggregate = windowed;
  } else {
    // Not comparing: single layer — windowed mean for SST, raw tile otherwise
    entry.layer = add(windowed ? new SSTAggregateProvider(cfg, state.date, win) : gibsProvider(cfg, state.date));
    entry.layer.alpha = entry.alpha;
    entry.isAggregate = windowed;
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
  entry.isRatio = false;
  entry.isAggregate = false;
  entry.suppressed = false;
  updateLegends();
}

function refreshTimedLayers() {
  // Suppressed entries (hidden because the active window can't average them)
  // must also refresh, so they reappear when the window returns to 1 day.
  for (const [id, entry] of Object.entries(state.layers)) {
    if ((entry.layer || entry.suppressed) && entry.cfg.timed) {
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
    const win = state.windowDays > 1 ? ` (${windowLabel(state.windowDays)})` : "";
    document.getElementById("split-label-left").textContent = compareDate() + win;
    document.getElementById("split-label-right").textContent = state.date + win;
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
  updateDeltaHint();
  refreshTimedLayers();
});

// Aggregation window slider (1..730 days) — orthogonal to the display mode.
const windowSlider = document.getElementById("window-days");
const windowValue = document.getElementById("window-value");
function syncWindowLabel() {
  windowValue.textContent = windowLabel(Number(windowSlider.value));
}
windowSlider.addEventListener("input", syncWindowLabel);
windowSlider.addEventListener("change", () => {
  state.windowDays = Number(windowSlider.value);
  syncWindowLabel();
  markWindowPreset();
  refreshTimedLayers();
  if (sstEnsembleLayer) updateEnsembleLayer();
});

// One-click window presets (1d/7d/30d/365d): drive the slider and fire its own
// change event so every consumer (label, presets, layers, ensemble) follows
// the exact same path as dragging it by hand.
document.getElementById("window-presets").addEventListener("click", (e) => {
  const win = e.target.getAttribute?.("data-win");
  if (!win) return;
  windowSlider.value = win;
  windowSlider.dispatchEvent(new Event("change", { bubbles: true }));
});
// highlight the preset matching the current window (if any)
function markWindowPreset() {
  for (const b of document.querySelectorAll("#window-presets button")) {
    b.classList.toggle("active", Number(b.dataset.win) === Number(windowSlider.value));
  }
}
syncWindowLabel();
markWindowPreset();

document.getElementById("toggle-grayscale").addEventListener("change", updateBaseAppearance);

// Note shown in computed-difference mode when a layer that can't be differenced
// is active — either a non-continuous raster (precip/aerosol) or a point/snapshot
// layer (glaciers, emissions, floats, biodiversity) that has no per-pixel time series.
function pointLayerActive() {
  return (glacierCollection && glacierCollection.show) ||
    (pointLayers.climatetrace && pointLayers.climatetrace.collection.show) ||
    (pointLayers.argo && pointLayers.argo.collection.show) ||
    !!gbifLayer;
}
function glaciersActive() {
  return glacierCollection && glacierCollection.show;
}
function updateDeltaHint() {
  const hint = document.getElementById("delta-hint");
  if (!hint) return;
  const msgs = [];

  // Aggregation warning — independent of comparison mode. A checked layer
  // that can't be time-averaged is hidden while a window is active; say so.
  const suppressedEntries = Object.values(state.layers).filter((e) => e.suppressed);
  const suppressed = suppressedEntries.map((e) => e.cfg.title);
  if (suppressed.length) {
    msgs.push(`⚠ <strong>${suppressed.join(", ")}</strong>: hidden while ` +
      `“${windowLabel(state.windowDays)}” aggregation is on — ` +
      (suppressed.length > 1 ? "these layers show" : "this layer shows") +
      " an instant, not an average, so a single day's picture under an averaged " +
      "label would mislead. Set Aggregate back to <em>single day</em> to show " +
      (suppressed.length > 1 ? "them" : "it") + " again." +
      // The 30-min case has a real alternative — say so instead of dead-ending.
      (suppressedEntries.some((e) => e.cfg.subDaily)
        ? " For rain over several days, the <em>daily</em> precipitation layer does average."
        : ""));
  }

  // Archive-end trap: for layers whose GIBS tiles stop before today (CERES
  // 2018-10, GRACE 2022-07, SSH 2019-01, soil moisture 2025-09), BOTH sides
  // of a comparison can clamp to the same last-served month — the difference
  // is then zero by construction and renders as nothing. Say so, and say how
  // to fix it, instead of leaving a silently empty comparison.
  if (state.compareYears) {
    const cmp = compareDate();
    const stuck = Object.values(state.layers).filter((e) =>
      (e.layer || e.suppressed) && e.cfg.timed && e.cfg.endTime &&
      gibsTime(e.cfg, state.date) === gibsTime(e.cfg, cmp));
    for (const e of stuck) {
      const end = e.cfg.endTime.slice(0, 7);
      msgs.push(`⚠ <strong>${e.cfg.title}</strong>: its tile archive ends ${end}, and both ` +
        `“${state.date}” and “${cmp}” fall after that — so both sides clamp to the same ` +
        `last month and the comparison is empty by construction. Set the date to ` +
        `<em>${end}</em> or earlier to compare within the archive (e.g. ${end} vs ` +
        `${Number(end.slice(0, 4)) - state.compareYears}-${end.slice(5)}).`);
    }
  }

  if (state.compareYears) {
    // Point/snapshot layers can't be compared over time (they have one state) —
    // relevant in BOTH side-by-side and computed-difference modes.
    if (pointLayerActive()) {
      msgs.push(glaciersActive()
        ? "⚠ The glacier layer is a single inventory (Randolph Glacier Inventory, ~year 2000), " +
          "so it can't be split or differenced by date — both sides would be identical. " +
          "Glacier <em>change</em> needs a time series; see the Temp/Sea-level tabs for the ice-loss signal."
        : "⚠ Point &amp; snapshot layers (emissions, floats, biodiversity) show a single state, " +
          "so they don't split or difference by date.");
    } else if (state.compareMode === "delta") {
      // Log-distributed fields render a ×-fold RATIO instead of a difference —
      // explain the different reading, and that single-day ratios are weather.
      if (Object.values(state.layers).some((e) => e.isRatio)) {
        msgs.push("ℹ Log-distributed fields (precipitation, chlorophyll, aerosol) compare " +
          "as a <strong>×-fold ratio</strong> of window means — red = more than then, " +
          "blue = less — because an absolute difference would mostly be palette " +
          "quantization error. A single-day ratio is weather (it rained <em>here</em> " +
          "today, <em>there</em> that day); widen <em>Aggregate</em> (e.g. 30+ days) " +
          "for a climate signal.");
      }
      // Rasters with NEITHER posture (true colour: a photograph, no colormap
      // to invert) really are shown as-is.
      const rasterNoDelta = Object.values(state.layers)
        .some((e) => e.layer && e.cfg.timed && !e.isDelta && !e.isRatio &&
                     e.cfg.deltaRange == null && !e.cfg.ratioRange);
      if (rasterNoDelta) {
        msgs.push("⚠ Computed change works on continuous rasters (temperatures, ice, snow, " +
          "salinity, sea-surface height, vegetation, water storage, energy balance) and, " +
          "as a ratio, on precipitation, chlorophyll &amp; aerosol. Photographic and " +
          "snapshot layers (true colour, 30-min precipitation) have nothing to invert, " +
          "so they are shown as-is.");
      }
    }
  }

  hint.innerHTML = msgs.join("<br/>");
  hint.classList.toggle("hidden", msgs.length === 0);
}

/* --------------------------------------------------------------- legends */

function updateLegends() {
  const panel = document.getElementById("legend-panel");
  if (!panel) return;
  panel.innerHTML = "";
  let any = false;
  if (sstEnsembleLayer) {
    panel.appendChild(ensembleLegendEl(sstEnsembleLayer.__ensembleMode));
    any = true;
  }
  for (const e of Object.values(state.layers)) {
    if (!e.layer) continue;
    if (e.isDelta) {
      panel.appendChild(deltaLegendEl(e.cfg));
      any = true;
    } else if (e.isRatio) {
      panel.appendChild(ratioLegendEl(e.cfg));
      any = true;
    } else if (e.cfg.grid) {
      panel.appendChild(gridLegendEl(e.cfg));
      any = true;
    } else if (e.cfg.colormap || e.cfg.legend) {
      panel.appendChild(layerLegendEl(e.cfg, e.isAggregate ? `${e.cfg.title} · ${windowLabel(state.windowDays)} mean` : null));
      any = true;
    }
  }
  panel.classList.toggle("hidden", !any);
  updateDeltaHint();
  updateBaseAppearance();
  updateActiveChips();
  updateTimeRow();
}

/* The ±30m time stepper only appears while a sub-daily layer is on (also while
 * one is merely suppressed by a window, so the control doesn't vanish and
 * reappear as the Aggregate slider moves). */
function updateTimeRow() {
  const row = document.getElementById("time-steps");
  if (!row) return;
  const active = Object.values(state.layers)
    .some((e) => e.cfg.subDaily && (e.layer || e.suppressed));
  row.classList.toggle("hidden", !active);
  const h = String(Math.floor(state.timeMin / 60)).padStart(2, "0");
  const m = String(state.timeMin % 60).padStart(2, "0");
  document.getElementById("time-value").textContent = `${h}:${m} UTC`;
}

/* ------------------------------------------- active-layer chips (on globe)
 * A chip per layer that is currently on, top-left of the globe. Solves two
 * things: turning a layer off without hunting for it in a long list, and
 * turning one off while a different sidebar tab is open (where the layer list
 * isn't rendered at all).
 *
 * Every layer — raster, grid, point, GBIF — is owned by exactly one checkbox,
 * so the chips drive those checkboxes instead of duplicating teardown logic:
 * unchecking and dispatching "change" runs the identical path as clicking the
 * box by hand, which keeps opacity sliders, toasts and legends in sync for
 * free. Adding a raster to GIBS_LAYERS wires up its chip automatically; a new
 * hand-written layer must be added to STATIC_LAYER_CHIPS. */
const STATIC_LAYER_CHIPS = [
  ["toggle-pixel", "Everything we know"],
  ["toggle-sst-ensemble", "SST ensemble"],
  ["toggle-climatetrace", "Facility emissions"],
  ["toggle-argo", "Argo floats"],
  ["toggle-stations", "Monitoring stations"],
  ["toggle-glaciers", "Glaciers"],
  ["toggle-gbif", "Biodiversity"],
];

function activeLayerChips() {
  const out = [];
  for (const cfg of GIBS_LAYERS) {
    const box = document.querySelector(`#layer-list input[data-id="${cfg.id}"]`);
    if (box && box.checked) {
      out.push({ box, title: cfg.title, warn: !!state.layers[cfg.id]?.suppressed });
    }
  }
  for (const [id, title] of STATIC_LAYER_CHIPS) {
    const box = document.getElementById(id);
    if (box && box.checked) out.push({ box, title, warn: false });
  }
  return out;
}

// Jumps the sidebar to the layer's row and outlines it briefly — the inverse
// of the chip's main job, for when you want the opacity slider or the notes.
function revealLayer(box) {
  document.getElementById("tab-layers").click();
  const item = box.closest(".layer-item");
  if (!item) return;
  item.scrollIntoView({ block: "center", behavior: "smooth" });
  item.classList.add("flash");
  setTimeout(() => item.classList.remove("flash"), 1400);
}

function turnOffLayer(box) {
  box.checked = false;
  box.dispatchEvent(new Event("change", { bubbles: true }));
}

function updateActiveChips() {
  const host = document.getElementById("active-layers");
  if (!host) return;
  const chips = activeLayerChips();
  host.innerHTML = "";
  host.classList.toggle("hidden", chips.length === 0);

  for (const c of chips) {
    const el = document.createElement("div");
    el.className = "chip" + (c.warn ? " chip-warn" : "");
    const label = document.createElement("button");
    label.className = "chip-label";
    label.textContent = (c.warn ? "⚠ " : "") + c.title;
    label.title = c.warn
      ? "Not shown: this layer can't be averaged over a window. Click to find it in the sidebar."
      : "Find this layer in the sidebar";
    label.addEventListener("click", () => revealLayer(c.box));
    const x = document.createElement("button");
    x.className = "chip-x";
    x.textContent = "×";
    x.title = `Turn off ${c.title}`;
    x.setAttribute("aria-label", `Turn off ${c.title}`);
    x.addEventListener("click", () => turnOffLayer(c.box));
    el.append(label, x);
    host.appendChild(el);
  }

  if (chips.length > 1) {
    const el = document.createElement("div");
    el.className = "chip chip-clear";
    const label = document.createElement("button");
    label.className = "chip-label";
    label.textContent = `Clear all ${chips.length}`;
    label.title = "Turn off every active layer";
    label.addEventListener("click", () => chips.forEach((c) => turnOffLayer(c.box)));
    el.appendChild(label);
    host.appendChild(el);
  }

  // Let the comparison date labels sit below the chips instead of under them.
  const container = document.getElementById("cesiumContainer");
  if (container) {
    container.style.setProperty(
      "--chips-h", chips.length ? `${host.offsetHeight + 8}px` : "0px");
  }
}

// Not every toggle routes through updateLegends(), and some controls switch a
// layer on indirectly (picking a species or a glacier mode ticks its box
// without firing that box's own event). Rather than enumerate those paths,
// refresh on any change event in the document — they're user-paced and rare,
// and the rebuild is a handful of DOM nodes. Dispatched events bubble here too.
document.addEventListener("change", updateActiveChips);

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
  // Handles both range entries value="[lo,hi)" and single-value entries
  // value="N" / value="[N]" (e.g. sea ice %, NDSI snow cover).
  const re = /<ColorMapEntry\s+rgb="(\d+),(\d+),(\d+)"\s+transparent="false"[^>]*?\svalue="([^"]+)"/g;
  let m;
  while ((m = re.exec(xml))) {
    const rgb = [+m[1], +m[2], +m[3]];
    let lo, hi;
    const rng = m[4].match(/^[\[(]\s*([^,]+),\s*([^)\]]+)[)\]]$/);
    if (rng) { lo = parseFloat(rng[1]); hi = parseFloat(rng[2]); }
    else {
      const single = m[4].match(/^[\[(]?\s*(-?[\d.eE+]+)\s*[)\]]?$/);
      if (!single) continue;
      lo = hi = parseFloat(single[1]);
    }
    if (!Number.isFinite(lo) && !Number.isFinite(hi)) continue;
    if (!Number.isFinite(lo)) lo = hi;
    if (!Number.isFinite(hi)) hi = lo;
    entries.push({ rgb, lo, hi });
  }
  entries.sort((a, b) => a.lo - b.lo);
  return { units, entries };
}

/* rgb → representative value map (+ units), from any GIBS colormap. Generalises
 * the SST LUT so the delta tool works for any continuous colormapped layer. */
const valueLutCache = new Map();
function getValueLut(url) {
  if (!valueLutCache.has(url)) {
    valueLutCache.set(url, getColormapEntries(url).then((cm) => {
      if (!cm) return null;
      const lut = new Map();
      for (const e of cm.entries) lut.set((e.rgb[0] << 16) | (e.rgb[1] << 8) | e.rgb[2], (e.lo + e.hi) / 2);
      return { units: cm.units, lut };
    }).catch(() => null));
  }
  return valueLutCache.get(url);
}

function fmtVal(v) {
  const a = Math.abs(v);
  return a >= 100 ? v.toFixed(0) : a >= 10 ? v.toFixed(1) : v.toFixed(2);
}

function layerLegendEl(cfg, titleOverride) {
  const div = document.createElement("div");
  div.className = "legend-item";
  div.innerHTML = `<div class="legend-title">${titleOverride || cfg.title}</div>`;
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

/* Legend for a client-rendered grid overlay: a ramp bar with min/mid/max and a
 * hover read-out, mirroring the GIBS colormap legends. */
function gridLegendEl(cfg) {
  const div = document.createElement("div");
  div.className = "legend-item";
  div.innerHTML = `<div class="legend-title">${cfg.title}</div>`;
  const wrap = document.createElement("div");
  wrap.className = "legend-bar-wrap";
  const canvas = document.createElement("canvas");
  const W = 268, H = 14, dpr = window.devicePixelRatio || 1;
  canvas.width = W * dpr;
  canvas.height = H * dpr;
  canvas.style.height = H + "px";
  canvas.className = "legend-bar";
  const ctx = canvas.getContext("2d");
  ctx.scale(dpr, dpr);
  const N = 120;
  for (let i = 0; i < N; i++) {
    const c = rampColor(cfg.ramp, i / (N - 1));
    ctx.fillStyle = `rgb(${c[0]},${c[1]},${c[2]})`;
    ctx.fillRect((i / N) * W, 0, W / N + 1, H);
  }
  const tip = document.createElement("div");
  tip.className = "legend-tip hidden";
  const range = document.createElement("div");
  range.className = "legend-range";
  range.innerHTML = `<span>${cfg.vmin}</span><span>${cfg.units}</span><span>${cfg.vmax}</span>`;
  canvas.addEventListener("mousemove", (e) => {
    const rect = canvas.getBoundingClientRect();
    const frac = Cesium.Math.clamp((e.clientX - rect.left) / rect.width, 0, 1);
    const v = cfg.vmin + frac * (cfg.vmax - cfg.vmin);
    tip.textContent = `${fmtVal(v)} ${cfg.units}`.trim();
    tip.style.left = `${Math.min(Math.max(frac * rect.width - 28, 0), rect.width - 80)}px`;
    tip.classList.remove("hidden");
  });
  canvas.addEventListener("mouseleave", () => tip.classList.add("hidden"));
  wrap.appendChild(tip);
  wrap.appendChild(canvas);
  div.appendChild(wrap);
  div.appendChild(range);
  return div;
}

function ensembleLegendEl(mode) {
  const div = document.createElement("div");
  div.className = "legend-item";
  if (mode === "spread") {
    div.innerHTML = `<div class="legend-title">SST ensemble spread — inter-analysis σ (°C)</div>`;
    const wrap = document.createElement("div");
    wrap.className = "legend-bar-wrap";
    const bar = document.createElement("div");
    bar.className = "spread-bar";
    const tip = document.createElement("div");
    tip.className = "legend-tip hidden";
    bar.addEventListener("mousemove", (e) => {
      const rect = bar.getBoundingClientRect();
      const frac = Cesium.Math.clamp((e.clientX - rect.left) / rect.width, 0, 1);
      tip.textContent = `${(frac * SPREAD_MAX).toFixed(2)} °C disagreement`;
      tip.style.left = `${Math.min(Math.max(frac * rect.width - 40, 0), rect.width - 110)}px`;
      tip.classList.remove("hidden");
    });
    bar.addEventListener("mouseleave", () => tip.classList.add("hidden"));
    wrap.appendChild(tip); wrap.appendChild(bar);
    div.appendChild(wrap);
    const range = document.createElement("div");
    range.className = "legend-range";
    range.innerHTML = `<span>0</span><span>°C</span><span>${SPREAD_MAX.toFixed(1)}+</span>`;
    div.appendChild(range);
    div.insertAdjacentHTML("beforeend", `<div class="legend-note">bright = analyses disagree (fronts, eddies, under-observed ocean)</div>`);
  } else {
    div.innerHTML = `<div class="legend-title">SST ensemble mean (°C)</div>`;
    getColormapEntries("https://gibs.earthdata.nasa.gov/colormaps/v1.3/GHRSST_Sea_Surface_Temperature.xml")
      .then((cm) => { if (cm) buildLegendBar(div, cm); });
    div.insertAdjacentHTML("beforeend", `<div class="legend-note">mean of independent GHRSST L4 analyses (MUR, OISST, GAMSSA) available for the date</div>`);
  }
  return div;
}

function deltaLegendEl(cfg) {
  const range = cfg.deltaRange || DELTA_RANGE;
  const div = document.createElement("div");
  div.className = "legend-item";
  const cmp = compareDate();
  const win = state.windowDays > 1 ? `, ${windowLabel(state.windowDays)} mean` : "";
  div.innerHTML = `<div class="legend-title">Δ ${cfg.title}: ${state.date} minus ${cmp}${win}</div>`;
  const wrap = document.createElement("div");
  wrap.className = "legend-bar-wrap";
  const bar = document.createElement("div");
  bar.className = "delta-bar";
  const tip = document.createElement("div");
  tip.className = "legend-tip hidden";
  // resolve units from the layer's colormap
  let units = "";
  getValueLut(cfg.colormap).then((v) => { if (v) units = v.units; });
  const more = "increase", less = "decrease";
  bar.addEventListener("mousemove", (e) => {
    const rect = bar.getBoundingClientRect();
    const frac = Cesium.Math.clamp((e.clientX - rect.left) / rect.width, 0, 1);
    const v = -range + frac * 2 * range;
    const dir = Math.abs(v) < range * 0.05 ? "(little change)" : v > 0 ? more : less;
    tip.textContent = `Δ ${v >= 0 ? "+" : ""}${fmtVal(v)} ${units} ${dir}`.replace("  ", " ");
    tip.style.left = `${Math.min(Math.max(frac * rect.width - 40, 0), rect.width - 130)}px`;
    tip.classList.remove("hidden");
  });
  bar.addEventListener("mouseleave", () => tip.classList.add("hidden"));
  wrap.appendChild(tip);
  wrap.appendChild(bar);
  div.appendChild(wrap);
  const rangeEl = document.createElement("div");
  rangeEl.className = "legend-range";
  getValueLut(cfg.colormap).then((v) => {
    const u = v ? v.units : "";
    rangeEl.innerHTML = `<span>−${range}</span><span>${u}</span><span>+${range}</span>`;
  });
  div.appendChild(rangeEl);
  div.insertAdjacentHTML("beforeend", `<div class="legend-note">blue = decrease · red = increase vs then · globe shown grey so the change stands out</div>`);
  return div;
}

/* Legend for a RatioProvider layer: the same diverging bar, but the axis is
 * multiplicative — ×N less | same | ×N more — matching the log rendering. */
function ratioLegendEl(cfg) {
  const range = cfg.ratioRange || 4;
  const div = document.createElement("div");
  div.className = "legend-item";
  const cmp = compareDate();
  const win = windowLabel(state.windowDays);
  div.innerHTML = `<div class="legend-title">${cfg.title}: ${state.date} ÷ ${cmp} (ratio of ${win} means)</div>`;
  const wrap = document.createElement("div");
  wrap.className = "legend-bar-wrap";
  const bar = document.createElement("div");
  bar.className = "delta-bar";
  const tip = document.createElement("div");
  tip.className = "legend-tip hidden";
  bar.addEventListener("mousemove", (e) => {
    const rect = bar.getBoundingClientRect();
    const frac = Cesium.Math.clamp((e.clientX - rect.left) / rect.width, 0, 1);
    const fold = Math.pow(range, 2 * frac - 1);      // log axis: range^-1 … range
    tip.textContent = Math.abs(Math.log(fold)) < Math.log(range) * 0.05
      ? "≈ same as then"
      : fold > 1 ? `×${fmtVal(fold)} more than then` : `×${fmtVal(1 / fold)} less than then`;
    tip.style.left = `${Math.min(Math.max(frac * rect.width - 40, 0), rect.width - 130)}px`;
    tip.classList.remove("hidden");
  });
  bar.addEventListener("mouseleave", () => tip.classList.add("hidden"));
  wrap.appendChild(tip);
  wrap.appendChild(bar);
  div.appendChild(wrap);
  const rangeEl = document.createElement("div");
  rangeEl.className = "legend-range";
  rangeEl.innerHTML = `<span>×${range} less</span><span>same</span><span>×${range} more</span>`;
  div.appendChild(rangeEl);
  div.insertAdjacentHTML("beforeend",
    `<div class="legend-note">×-fold ratio of window means (log scale) — the sound comparison ` +
    `for a log-distributed field. Widen <em>Aggregate</em> for a stabler signal.</div>`);
  return div;
}

/* ------------------------------------------------------------------- toasts */

/* A prominent, animated notification. Used when a user enables a layer that has
 * no date-specific data, so the date selector's lack of effect is never a
 * mystery. Auto-dismisses; identical messages are de-duped while on screen. */
const activeToastKeys = new Set();
function showToast(html, { key = html, timeout = 8000 } = {}) {
  const host = document.getElementById("toast-host");
  if (!host || activeToastKeys.has(key)) return;
  activeToastKeys.add(key);
  const el = document.createElement("div");
  el.className = "toast";
  el.setAttribute("role", "alert");
  el.innerHTML = `<span class="toast-ico">📅</span><div class="toast-body">${html}</div>` +
    `<button class="toast-close" title="Dismiss" aria-label="Dismiss">×</button>`;
  const dismiss = () => {
    if (!el.isConnected) return;
    el.classList.add("toast-out");
    el.addEventListener("animationend", () => el.remove(), { once: true });
    activeToastKeys.delete(key);
  };
  el.querySelector(".toast-close").addEventListener("click", dismiss);
  host.appendChild(el);
  setTimeout(dismiss, timeout);
  return el;
}

/* If a layer has no per-date data, return the toast HTML explaining why the
 * date selector won't affect it; otherwise null. Covers GIBS-panel layers
 * (grids = climatologies, night lights = fixed composite) and the data/point
 * layers (each a snapshot or all-time record). */
function datelessToast(id) {
  const cfg = GIBS_LAYERS.find((l) => l.id === id);
  if (cfg) {
    if (cfg.timed) return null;                              // genuinely date-driven
    if (cfg.grid) {
      // most grids are multi-decade climatologies; a snapshot grid (a single
      // recent month, like the Argo 300 m anomaly) says what it actually is
      return cfg.snapshotGrid
        ? `<strong>${cfg.title}</strong> is a single recent-month snapshot, so the ` +
          `<strong>date selector doesn't change it</strong> (refreshed with the data pipeline).`
        : `<strong>${cfg.title}</strong> is a long-term climatology (a multi-decade ` +
          `average), so it has no per-date data — the <strong>date selector doesn't change it</strong>.`;
    }
    return `<strong>${cfg.title}</strong> is a fixed composite, so the ` +
      `<strong>date selector doesn't change it</strong>.`;
  }
  const DATA_MSG = {
    gbif: "<strong>Biodiversity occurrences (GBIF)</strong> are all-time, so the " +
      "<strong>date selector doesn't change this layer</strong>. Rare taxa (e.g. humans) may " +
      "be too sparse to see when zoomed out — that's expected, not a date problem.",
    climatetrace: "<strong>Climate TRACE emissions</strong> is an annual inventory, so the " +
      "<strong>date selector doesn't change it</strong>.",
    argo: "<strong>Argo floats</strong> shows the fleet's latest positions (last ~10 days), so the " +
      "<strong>date selector doesn't change it</strong>.",
    stations: "<strong>Monitoring stations</strong> are fixed sites, so the " +
      "<strong>date selector doesn't change this layer</strong>.",
    glaciers: "<strong>Glaciers (RGI v7)</strong> is a single inventory (~year 2000), so the " +
      "<strong>date selector doesn't change it</strong>. Its melt-rate colouring is a 2000–2020 average.",
    "sst-ensemble": null,   // the ensemble IS date-driven
  };
  return DATA_MSG[id] || null;
}

function maybeDatelessToast(id) {
  const html = datelessToast(id);
  if (html) showToast(html, { key: id });
}

/* ----------------------------------------------------------- GIBS layer panel */

/* Recording period, time interval and spatial granularity for every layer,
 * shown as a hover card on the layer entry. "Recorded" is the span of the
 * underlying measurement record (≠ the date currently displayed). */
const LAYER_FACTS = {
  "viirs-truecolor": { rec: "2015-11 → present", int: "daily (mosaic of ~14 orbits)", sp: "250 m/pixel",
    sum: "What Earth actually looked like on the chosen day, stitched from the VIIRS " +
         "imager's ~14 daily orbits. Clouds, dust storms, wildfire smoke, algal blooms " +
         "and snow appear exactly as photographed — the visual ground truth under all " +
         "the other layers." },
  "sst": { rec: "2002-06 → present", int: "daily (gap-free L4 analysis)", sp: "1 km grid",
    sum: "The temperature of the ocean surface, every day, with no gaps: MUR blends " +
         "infrared and microwave satellites plus buoys into a 1 km analysis. The " +
         "workhorse layer for eddies, marine heatwaves, and the North Atlantic 'cold " +
         "blob' south of Greenland — a suspected AMOC fingerprint." },
  "sst-anom": { rec: "2002-09 → present", int: "daily", sp: "25 km grid",
    sum: "How unusual today's ocean temperature is: the same MUR analysis minus its " +
         "own climatology, so persistent warm/cold departures stand out regardless of " +
         "season. Marine heatwaves and the cold blob read directly in °C above or " +
         "below normal." },
  "precip": { rec: "2000-06 → present", int: "daily (mean of the day's 30-min scans)", sp: "~10 km (0.1°)",
    sum: "Where it rained today: IMERG merges the GPM core satellite with a " +
         "constellation of microwave sensors into a global map of the day's mean " +
         "rain rate. The Aggregate slider averages it over longer windows (dry " +
         "pixels count as zero, so it's a true mean, not 'rate when raining') — " +
         "for 'how much rain is normal here', see the GPCP/E-OBS/MeteoSwiss " +
         "climatologies." },
  "precip-30min": { rec: "2000-06 → present", int: "every 30 minutes — step with the ±30m buttons", sp: "~10 km (0.1°)",
    sum: "The same IMERG merged precipitation at its native half-hourly cadence — " +
         "sharp enough to watch individual storm systems and tropical cyclones " +
         "develop over the course of a single day, using the ±30m time stepper " +
         "under the date." },
  "soilmoisture": { rec: "this map: 2012-07 → 2025-09 (last date GIBS serves; the instrument continues)", int: "daily (swath — gaps are orbit coverage, not missing data)", sp: "~25 km",
    sum: "How wet the top centimetre of soil is, sensed by passive microwave " +
         "(AMSR2). The land half of drought: it dries in days, unlike the deep " +
         "storage GRACE sees. Daily maps are striped by orbit swaths — average a " +
         "window with the Aggregate slider for full coverage." },
  "ndvi": { rec: "2000-02 → present", int: "monthly composite (of 16-day maxima)", sp: "1 km",
    sum: "How green and dense vegetation is (NDVI, 0–1): the pulse of the " +
         "biosphere. Season swings it; climate shifts it — comparing the same " +
         "month across years (Compare → computed change) reveals greening, " +
         "browning, deforestation and drought stress directly." },
  "grace": { rec: "this map: 2002-08 → 2022-07 (last month GIBS serves; GRACE-FO continues) · 2017–18 has a between-missions gap", int: "monthly", sp: "~300 km (3° mascons)",
    sum: "Where Earth gained or lost water mass — all of it: groundwater, soil, " +
         "snow, ice — measured by how the mass below tugs at a pair of " +
         "satellites. The only direct observation of deep water storage; " +
         "aquifer depletion in India or California and ice-sheet loss appear " +
         "in the same map, in centimetres of equivalent water." },
  "ssh-anom": { rec: "this map: 1992-10 → 2019-01 (last epoch GIBS serves; altimetry continues — see the Sea level tab)", int: "5-day", sp: "~17 km grid",
    sum: "How high the sea surface stands versus its long-term mean — the " +
         "ocean's pressure gauge. Warm, expanded water and slow ocean eddies " +
         "read directly; the Gulf Stream's meanders are the string of bumps and " +
         "dips along the US east coast. Differencing two dates shows local " +
         "sea-level change." },
  "ceres": { rec: "this map: 2000-03 → 2018-10 (last month GIBS serves; the EBAF record continues)", int: "monthly", sp: "1° grid",
    sum: "Earth's energy budget per pixel: sunlight absorbed minus heat " +
         "radiated back to space (CERES). Positive means this place is banking " +
         "energy. The global imbalance of ~+1 W/m² IS global warming, measured " +
         "at the top of the atmosphere." },
  "currents": { rec: "one recent month (see legend) — not one date; GLORYS reanalysis covers 1993 → near-present", int: "monthly-mean snapshot, refreshed by the data pipeline", sp: "1/12° model, shipped at 1°",
    sum: "How fast the surface ocean moves: the monthly-mean current speed from " +
         "the GLORYS ocean reanalysis (which assimilates Argo, altimetry and " +
         "SST). The Gulf Stream, Kuroshio and Antarctic Circumpolar Current " +
         "leap out — the plumbing that carries the heat the AMOC story is " +
         "about. Click any ocean point for speed AND direction." },
  "mld": { rec: "one recent month (see legend) — not one date; deepest in late winter", int: "monthly-mean snapshot, refreshed by the data pipeline", sp: "1/12° model, shipped at 1°",
    sum: "How deep wind and cooling stir the surface ocean. Watch the subpolar " +
         "North Atlantic: in late winter the mixed layer there can reach " +
         "hundreds of metres to a kilometre — that deep convection is where " +
         "AMOC deep water is actually made, so its shrinking is an early " +
         "warning signal. In summer it shoals to tens of metres everywhere." },
  "argo-t300": { rec: "one recent month (see legend) vs the 2004–2018 mean for that same calendar month — not one date", int: "monthly snapshot, refreshed by the data pipeline", sp: "1° (Argo objective mapping)",
    sum: "Where the ocean is unusually warm or cool 300 m DOWN — measured by the " +
         "Argo float fleet, invisible to every satellite surface map. Subsurface " +
         "marine heatwaves matter because that heat is stored, not radiated away; " +
         "comparing against the same calendar month removes the seasonal cycle, so " +
         "red really means anomalous." },
  "seaice": { rec: "2012-07 → present, with gaps", int: "daily", sp: "12 km grid",
    sum: "The fraction of ocean covered by sea ice at both poles, sensed by passive " +
         "microwave (AMSR2), which sees through clouds and polar night. The " +
         "September Arctic minimum and its long-term decline are the field's " +
         "headline climate signal." },
  "snow": { rec: "2000-02 → present", int: "daily", sp: "500 m grid",
    sum: "Daily snow-covered area from MODIS's normalised-difference snow index. " +
         "Snow cover sets Earth's reflectivity (albedo) and spring meltwater supply; " +
         "its retreat is both a symptom and an amplifier of warming." },
  "aod": { rec: "this map: 2017-04 → present · MODIS has measured since 2000, but older dates aren't served as map tiles", int: "daily", sp: "10 km grid",
    sum: "How much smoke, dust and haze is in the air column: aerosol optical depth " +
         "from MODIS. Wildfire plumes, Saharan dust outbreaks and pollution episodes " +
         "show as bright bands; aerosols are also the largest source of uncertainty " +
         "in climate forcing." },
  "lst": { rec: "this map: 2022-10 → present · MODIS has measured since 2000, but older dates aren't served as map tiles", int: "daily (one daytime satellite pass)", sp: "1 km grid",
    sum: "The temperature of the ground itself (not the air above it), measured by " +
         "MODIS thermal infrared. Coverage on any single day is patchy by nature: " +
         "only cloud-free pixels seen on that day's pass can be measured — the gaps " +
         "are clouds, not missing data. Deserts exceed 60 °C; cities show as heat " +
         "islands." },
  "chlor": { rec: "this map: 2024-02 → present (PACE mission) · earlier ocean-colour missions reach back to 1997", int: "daily", sp: "~1.2 km",
    sum: "Phytoplankton concentration inferred from the colour of the ocean, from " +
         "NASA's newest ocean-colour mission (PACE). Phytoplankton are the base of " +
         "the marine food web and fix about as much carbon as all land plants; " +
         "blooms trace nutrient-rich currents and upwelling." },
  "salinity": { rec: "2015-04 → present (2024 data gap)", int: "monthly composite", sp: "~60 km",
    sum: "How salty the ocean surface is, sensed by SMAP's L-band radiometer. " +
         "Salinity traces the water cycle (river plumes, evaporation, rainfall) and " +
         "sets seawater density — a key control on the deep overturning circulation " +
         "watched in the AMOC tab. Same quantity as ESA's SMOS mission." },
  "gpcp": { rec: "measurements 1979 → present · the map shows the average over the whole record (not one date)", int: "source: monthly · shown: mean annual total", sp: "2.5° (~275 km)",
    sum: "The long-term average of global rainfall: gauge and satellite records " +
         "blended since 1979, shown here as mean annual precipitation. The tropical " +
         "rain band, monsoon regions and desert belts emerge cleanly — this is " +
         "'where it rains on average', complementing IMERG's 'where it rains now'." },
  "oisst": { rec: "average of the years 1991–2020 (a fixed 30-year baseline, not one date)", int: "source: monthly · shown: annual mean", sp: "0.25° source → 1° shown",
    sum: "The 30-year (1991–2020) average state of sea surface temperature from " +
         "NOAA's OISST record — the baseline against which today's anomalies are " +
         "judged. Compare with the live MUR layer to see how the current ocean " +
         "departs from its long-term normal." },
  "eobs": { rec: "measurements 1950 → 2024 (v31) · the map shows the average over the whole record (not one date)", int: "source: daily gauges · shown: mean annual total", sp: "0.25° (~28 km), Europe land",
    sum: "Europe's rainfall climate from thousands of ground rain gauges gridded " +
         "since 1950 (Copernicus E-OBS). At 0.25° the orographic detail appears — " +
         "wet Atlantic coasts and Alpine flanks, dry Iberian and Pannonian " +
         "interiors — that global products blur away. Land only, Europe only." },
  "meteoswiss": { rec: "average of the years 1991–2020 (the official 'normal period', not one date)", int: "one 30-year average", sp: "~2 km, Switzerland",
    sum: "The official Swiss precipitation normal at ~2 km, from MeteoSwiss's " +
         "open-data gridded climatology. The sharpest view in the app of how " +
         "mountains make rain: valley floors receive under 600 mm/yr while nearby " +
         "Alpine crests exceed 3,000 mm/yr." },
  "nightlights": { rec: "a composite of the whole year 2016 (fixed — ignores the date selector)", int: "one composite of a full year", sp: "500 m grid",
    sum: "Human presence seen from orbit at night: a cloud-free annual composite of " +
         "VIIRS low-light imagery (Black Marble). Cities, highways, gas flares and " +
         "fishing fleets shine; it doubles as a proxy map of energy use and " +
         "economic activity." },
};

function layerTipHtml(id) {
  const f = LAYER_FACTS[id];
  if (!f) return "";
  return `<div class="layer-tip">
      ${f.sum ? `<p class="tip-sum">${f.sum}</p>` : ""}
      <div><span>Recorded</span>${f.rec}</div>
      <div><span>Interval</span>${f.int}</div>
      <div><span>Spatial</span>${f.sp}</div>
    </div>`;
}

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
      <div class="meta">${cfg.meta}</div>
      <input type="range" min="0" max="100" value="100" data-alpha="${cfg.id}"
             ${cfg.on ? "" : "style='display:none'"} title="opacity"/>
      ${layerTipHtml(cfg.id)}`;
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
      maybeDatelessToast(id);
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
    if (sstEnsembleLayer) updateEnsembleLayer();
  });

  // Quick date stepping: real calendar arithmetic (−1m from Mar 31 → Feb 28,
  // −1y from Feb 29 → Feb 28), clamped to [layer availability, most recent].
  document.getElementById("date-steps").addEventListener("click", (e) => {
    const step = e.target.getAttribute?.("data-step");
    if (!step) return;
    let next;
    if (step === "today") {
      next = defaultDate();
    } else {
      const d = new Date(state.date + "T00:00:00Z");
      const n = step.startsWith("-") ? -1 : 1;
      const unit = step.slice(-1);
      if (unit === "d") d.setUTCDate(d.getUTCDate() + n);
      else if (unit === "m") {
        const day = d.getUTCDate();
        d.setUTCDate(1); d.setUTCMonth(d.getUTCMonth() + n);
        const last = new Date(Date.UTC(d.getUTCFullYear(), d.getUTCMonth() + 1, 0)).getUTCDate();
        d.setUTCDate(Math.min(day, last));
      } else {
        const day = d.getUTCDate(), mon = d.getUTCMonth();
        d.setUTCDate(1); d.setUTCFullYear(d.getUTCFullYear() + n); d.setUTCMonth(mon);
        const last = new Date(Date.UTC(d.getUTCFullYear(), mon + 1, 0)).getUTCDate();
        d.setUTCDate(Math.min(day, last));
      }
      next = d.toISOString().slice(0, 10);
    }
    if (next > defaultDate()) next = defaultDate();
    if (next < "2000-01-01") next = "2000-01-01";
    if (next === state.date) return;
    state.date = next;
    dateInput.value = next;
    refreshTimedLayers();
    if (sstEnsembleLayer) updateEnsembleLayer();
  });

  // ±30m time-of-day stepping for sub-daily layers. Crossing midnight rolls
  // the date (clamped to the layer-availability range like the day stepper).
  document.getElementById("time-steps").addEventListener("click", (e) => {
    const step = Number(e.target.getAttribute?.("data-tstep"));
    if (!step) return;
    let t = state.timeMin + step;
    let date = state.date;
    if (t < 0) { t = 1440 + t; date = addDays(date, -1); }
    else if (t >= 1440) { t = t - 1440; date = addDays(date, 1); }
    if (date > defaultDate() || date < "2000-01-01") return; // nothing to step to
    state.timeMin = t;
    if (date !== state.date) {
      state.date = date;
      dateInput.value = date;
      refreshTimedLayers();          // date change affects every timed layer
    } else {
      // Same date, new half-hour: only sub-daily layers see a different TIME,
      // so don't churn (refetch) the daily/monthly layers on every step.
      for (const [id, entry] of Object.entries(state.layers)) {
        if (entry.cfg.subDaily && (entry.layer || entry.suppressed)) {
          removeLayer(id);
          addLayer(entry.cfg);
        }
      }
    }
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

document.getElementById("toggle-sst-ensemble").addEventListener("change", updateEnsembleLayer);
document.getElementById("ensemble-mode").addEventListener("change", () => {
  document.getElementById("toggle-sst-ensemble").checked = true;
  updateEnsembleLayer();
});

for (const kind of ["climatetrace", "argo"]) {
  document.getElementById(`toggle-${kind}`).addEventListener("change", (e) => {
    if (e.target.checked) { loadPointLayer(kind).then(updateDeltaHint); maybeDatelessToast(kind); }
    else if (pointLayers[kind]) pointLayers[kind].collection.show = false;
    updateDeltaHint();
  });
}

/* Randolph Glacier Inventory v7 — ~274k glaciers as centroid points sized by area.
 * Two colourings: by extent (area), or by 2000-2020 thinning rate (Hugonnet 2021),
 * so you can see which glaciers are actually melting. Display-only for performance. */
let glacierCollection = null, glacierData = null;
const GLACIER_COLD = Cesium.Color.fromCssColorString("#8fd3ff");
const GLACIER_BIG = Cesium.Color.fromCssColorString("#ffffff");
const GLACIER_NODATA = Cesium.Color.fromCssColorString("#6b7280");

function glacierColor(mode, dhdt, area) {
  if (mode !== "change") return (area > 50 ? GLACIER_BIG : GLACIER_COLD).withAlpha(0.8);
  if (dhdt == null) return GLACIER_NODATA.withAlpha(0.35);
  // negative dhdt = thinning/melting → warm (red); positive = growing → cool (blue)
  const t = Cesium.Math.clamp(dhdt / 1.5, -1, 1);       // ±1.5 m/yr scale
  const a = 0.55 + 0.4 * Math.min(1, Math.abs(t) + 0.1);
  if (dhdt < 0) {                                        // melting: yellow → red by intensity
    const f = Math.min(1, -dhdt / 1.5);
    return new Cesium.Color(0.95, 0.75 - 0.6 * f, 0.15, a);
  }
  return new Cesium.Color(0.22, 0.55, 0.95, a);          // growing/stable: blue
}

function colorGlaciers() {
  if (!glacierCollection || !glacierData) return;
  const mode = document.getElementById("glacier-mode").value;
  for (let i = 0; i < glacierCollection.length; i++) {
    glacierCollection.get(i).color = glacierColor(mode, glacierData.dhdt[i], glacierData.area[i]);
  }
  const legend = document.getElementById("glacier-legend");
  if (legend) legend.classList.toggle("hidden", mode !== "change");
}

async function loadGlaciers() {
  if (glacierCollection) { glacierCollection.show = true; return; }
  glacierData = await (await fetch("data/glaciers.json")).json();
  const j = glacierData;
  const col = viewer.scene.primitives.add(new Cesium.PointPrimitiveCollection());
  const mode = document.getElementById("glacier-mode").value;
  for (let i = 0; i < j.lon.length; i++) {
    col.add({
      position: Cesium.Cartesian3.fromDegrees(j.lon[i], j.lat[i]),
      pixelSize: Math.max(1.5, Math.min(12, 1.5 + Math.sqrt(j.area[i]) * 0.9)),
      color: glacierColor(mode, j.dhdt[i], j.area[i]),
    });
  }
  glacierCollection = col;
  colorGlaciers();
  const meta = document.getElementById("meta-glaciers");
  if (meta) meta.textContent = `${j.count.toLocaleString()} glaciers · ${j.total_area_km2.toLocaleString()} km² · RGI v7 · ${j.dhdt_matched.toLocaleString()} with 2000–2020 melt rate`;
}
document.getElementById("toggle-glaciers").addEventListener("change", (e) => {
  if (e.target.checked) { loadGlaciers().then(updateDeltaHint); maybeDatelessToast("glaciers"); }
  else if (glacierCollection) glacierCollection.show = false;
  updateDeltaHint();
});
document.getElementById("glacier-mode").addEventListener("change", () => {
  document.getElementById("toggle-glaciers").checked = true;
  loadGlaciers().then(colorGlaciers);
});

/* ------------------------------------------------- biodiversity (GBIF) layer */

/* GBIF occurrence-density tiles are key-free PNGs on a standard power-of-two
 * geographic pyramid (2×1 at z0), so Cesium's built-in GeographicTilingScheme
 * fits directly. taxonKey filters to a single species; omit for all life. */
let gbifLayer = null;
let gbifSpecies = null;

let gbifData = null;
let gbifIndex = {};
async function initSpeciesUI() {
  const sel = document.getElementById("species-select");
  if (!sel) return;
  gbifData = await (await fetch("data/species.json")).json();
  gbifSpecies = gbifData.species;
  const fmt = (n) => Number(n).toLocaleString();

  // Broad taxonomic categories (kingdoms, major groups, humans), each an
  // <optgroup>, so the whole 3.9 B partitions into pickable slices.
  for (const cat of gbifData.categories || []) {
    const og = document.createElement("optgroup");
    og.label = cat.label;
    for (const it of cat.items) {
      const o = document.createElement("option");
      o.value = it.key;
      o.innerHTML = `${it.name} (${fmt(it.records)})`;
      o.dataset.note = it.name;
      og.appendChild(o);
    }
    sel.appendChild(og);
  }
  // Climate-indicator species
  const sog = document.createElement("optgroup");
  sog.label = "Climate-indicator species";
  for (const s of gbifSpecies) {
    const o = document.createElement("option");
    o.value = s.key;
    o.innerHTML = `${s.common} (${fmt(s.records)})`;
    o.dataset.note = s.note;
    sog.appendChild(o);
  }
  sel.appendChild(sog);

  // Flat lookup key → {name, records} across every category + species, so the
  // note can report the count and warn when a taxon is too sparse to see.
  gbifIndex = {};
  for (const cat of gbifData.categories || [])
    for (const it of cat.items) gbifIndex[it.key] = it;
  for (const s of gbifSpecies) gbifIndex[s.key] = { name: s.common, records: s.records, note: s.note };

  // Composition note explaining what "all recorded life" contains
  const noteEl = document.getElementById("species-note");
  const defaultNote = gbifData.note || noteEl.textContent;
  noteEl.textContent = defaultNote;

  document.getElementById("toggle-gbif").addEventListener("change", (e) => {
    updateGbifLayer();
    if (e.target.checked) maybeDatelessToast("gbif");
  });
  sel.addEventListener("change", () => {
    document.getElementById("toggle-gbif").checked = true;
    updateGbifLayer();
    noteEl.innerHTML = gbifNoteFor(sel.value, defaultNote);
  });
}

// Human-readable note for a GBIF selection, including a sparsity warning (so a
// rare taxon that paints only a few faint dots doesn't look broken) and the
// reminder that this layer is all-time and ignores the date selector.
const GBIF_SPARSE = 150000;   // fewer records worldwide → likely invisible when zoomed out
function gbifNoteFor(value, defaultNote) {
  const dateNote = "<em>Occurrences are all-time — the date selector doesn't change this layer.</em>";
  if (value === "") return `${defaultNote}<br/>${dateNote}`;
  const it = gbifIndex[value];
  if (!it) return `${defaultNote}<br/>${dateNote}`;
  const n = Number(it.records);
  let msg = it.note ? it.note : `${it.name.replace(/&amp;/g, "&")} — ${n.toLocaleString()} records.`;
  if (String(value) === "2436436") {
    msg = `Humans are recorded like any other species, but GBIF restricts human ` +
      `occurrences for privacy — only ${n.toLocaleString()} records worldwide.`;
  }
  if (n < GBIF_SPARSE) {
    msg += ` ⚠ Only ${n.toLocaleString()} records — dots are very sparse and easy to ` +
      `miss when zoomed out. Zoom in (or nothing may appear at global scale). This is ` +
      `expected, not a date problem.`;
  }
  return `${msg}<br/>${dateNote}`;
}

function updateGbifLayer() {
  if (gbifLayer) { viewer.imageryLayers.remove(gbifLayer, true); gbifLayer = null; }
  if (!document.getElementById("toggle-gbif").checked) { updateDeltaHint(); return; }
  const sel = document.getElementById("species-select");
  const noteEl = document.getElementById("species-note");
  if (noteEl && gbifData) noteEl.innerHTML = gbifNoteFor(sel.value, gbifData.note);
  const taxon = sel.value;
  const taxonParam = taxon ? `&taxonKey=${taxon}` : "";
  // point styles keep the background transparent so occurrences overlay the globe;
  // a warm palette for a single species, cool for all-life density
  const style = taxon ? "fire.point" : "purpleYellow.point";
  const url = `https://api.gbif.org/v2/map/occurrence/density/{z}/{x}/{y}@1x.png` +
    `?srs=EPSG:4326&style=${style}${taxonParam}`;
  gbifLayer = viewer.imageryLayers.addImageryProvider(
    new Cesium.UrlTemplateImageryProvider({
      url,
      tilingScheme: new Cesium.GeographicTilingScheme({ numberOfLevelZeroTilesX: 2, numberOfLevelZeroTilesY: 1 }),
      tileWidth: 512, tileHeight: 512, maximumLevel: 14,
      credit: new Cesium.Credit("Biodiversity: GBIF.org"),
    })
  );
  gbifLayer.alpha = 1.0;
  updateDeltaHint();
}

// Click-picking: point primitives → info card; bare globe → pixel inspector
new Cesium.ScreenSpaceEventHandler(viewer.scene.canvas).setInputAction((click) => {
  const picked = viewer.scene.pick(click.position);
  if (picked?.id?.kind) {
    pickCard.innerHTML = picked.id.html;
    pickCard.classList.remove("hidden");
    return;
  }
  if (!picked) pickCard.classList.add("hidden");
  // Nothing pickable under the cursor: if the inspector is engaged and the
  // click actually hit the globe (not sky), compose that point's state card.
  if (!picked && pixelInspectorEngaged()) {
    const cart = viewer.camera.pickEllipsoid(click.position, viewer.scene.globe.ellipsoid);
    if (cart) showPixelState(Cesium.Cartographic.fromCartesian(cart));
  }
}, Cesium.ScreenSpaceEventType.LEFT_CLICK);

/* ------------------------------------------------- hover value probe (raster) */

/* On hover, read the actual value of the top colormapped layer at the cursor by
 * inverting that layer's GIBS colormap on its source tile — so you get the value
 * in physical units, not just where it sits on the legend. */

const invLutCache = new Map();      // colormap URL → Promise<{units, lut: Map}>
function getInvLut(url) {
  if (!invLutCache.has(url)) {
    invLutCache.set(url, getColormapEntries(url).then((cm) => {
      if (!cm) return null;
      const lut = new Map();
      for (const e of cm.entries) lut.set((e.rgb[0] << 16) | (e.rgb[1] << 8) | e.rgb[2], e);
      return { units: cm.units, lut };
    }).catch(() => null));
  }
  return invLutCache.get(url);
}

const probeTileCache = new Map();   // "layer|date|z|x|y" → Promise<ImageBitmap|null>
function fetchProbeTile(cfg, date, z, x, y) {
  const key = `${cfg.layer}|${date}|${z}|${x}|${y}`;
  const time = gibsTime(cfg, date);
  const url = GIBS_URL
    .replace("{layer}", cfg.layer).replace("{time}", time)
    .replace("{tms}", cfg.tms).replace("{ext}", cfg.ext)
    .replace("{TileMatrix}", z).replace("{TileRow}", y).replace("{TileCol}", x);
  if (!probeTileCache.has(key)) {
    probeTileCache.set(key, sstFetchBitmap(url));
    if (probeTileCache.size > 48) probeTileCache.delete(probeTileCache.keys().next().value);
  }
  // A transient fetch failure must not be cached as "no tile" forever — with
  // the pixel card probing 14 layers at once, one dropped connection would
  // permanently blank that layer's row. On access, a null result retries once.
  return probeTileCache.get(key).then((img) => {
    if (img) return img;
    const again = sstFetchBitmap(url);
    probeTileCache.set(key, again);
    return again;
  });
}

const probeCanvas = document.createElement("canvas");
probeCanvas.width = probeCanvas.height = 512;
const probeCtx = probeCanvas.getContext("2d", { willReadFrequently: true });

// topmost active layer that has an invertible colormap
function topColormapLayer() {
  let best = null, bestIdx = -1;
  for (const e of Object.values(state.layers)) {
    if (e.layer && (e.cfg.colormap || e.cfg.grid)) {
      const idx = viewer.imageryLayers.indexOf(e.layer);
      if (idx > bestIdx) { bestIdx = idx; best = e; }
    }
  }
  return best;
}

// value of one pixel from a single source tile, colormap-inverted (or null)
async function probePixel(cfg, date, z, x, y, px, py, valueLut) {
  const img = await fetchProbeTile(cfg, date, z, x, y);
  if (!img) return null;
  probeCtx.clearRect(0, 0, 512, 512);
  probeCtx.drawImage(img, 0, 0);
  const d = probeCtx.getImageData(px, py, 1, 1).data;
  if (d[3] === 0) return null;
  const v = valueLut.get((d[0] << 16) | (d[1] << 8) | d[2]);
  return v === undefined ? null : v;
}
// Mean pixel value across a set of sample dates (rolling-window mean), with
// the same transparent-pixel semantics as the rendered mean (sstMeanField):
// for transparentZero layers a transparent pixel in a LOADED tile is a real 0
// and counts; only a missing tile is a missing sample. Without this the probe
// of an aggregated precip layer would read "mean rate on rainy days" — higher
// than the map it is probing.
async function probePixelMean(cfg, dates, z, x, y, px, py, valueLut) {
  if (!cfg.transparentZero) {
    const vals = await Promise.all(dates.map((dt) => probePixel(cfg, dt, z, x, y, px, py, valueLut)));
    const ok = vals.filter((v) => v != null);
    return ok.length ? ok.reduce((s, v) => s + v, 0) / ok.length : null;
  }
  const imgs = await Promise.all(dates.map((dt) => fetchProbeTile(cfg, dt, z, x, y)));
  let sum = 0, cnt = 0;
  for (const img of imgs) {
    if (!img) continue;
    probeCtx.clearRect(0, 0, 512, 512);
    probeCtx.drawImage(img, 0, 0);
    const d = probeCtx.getImageData(px, py, 1, 1).data;
    if (d[3] === 0) { cnt++; continue; }               // dry: counts as 0
    const v = valueLut.get((d[0] << 16) | (d[1] << 8) | d[2]);
    if (v === undefined) continue;
    sum += v; cnt++;
  }
  return cnt ? sum / cnt : null;
}

async function probeValueAt(carto) {
  const entry = topColormapLayer();
  if (!entry) return null;
  const cfg = entry.cfg;
  const lon = Cesium.Math.toDegrees(carto.longitude);
  const lat = Cesium.Math.toDegrees(carto.latitude);
  if (cfg.grid) {
    // Grid overlays: read the exact cell value straight from the loaded grid.
    const g = await loadGrid(cfg);
    const base = { title: cfg.title, units: cfg.units, lon, lat };
    if (!g) return { ...base, noData: true };
    const v = sampleGrid(g, lon, lat);
    return v == null ? { ...base, noData: true } : { ...base, value: v };
  }
  const computed = entry.isDelta || entry.isRatio || entry.isAggregate;
  const win = computed ? state.windowDays : 1;
  // match the rendered resolution (delta/ratio/aggregate cap the level)
  const z = computed ? windowMaxLevel(cfg, win) : cfg.maxLevel;
  const span = (0.5625 / 2 ** z) * 512;               // degrees per tile at level z
  const x = Math.floor((lon + 180) / span);
  const y = Math.floor((90 - lat) / span);
  const tileWest = -180 + x * span, tileNorth = 90 - y * span;
  const px = Math.min(511, Math.max(0, Math.floor((lon - tileWest) / span * 512)));
  const py = Math.min(511, Math.max(0, Math.floor((tileNorth - lat) / span * 512)));
  const vlut = await getValueLut(cfg.colormap);
  if (!vlut) return null;
  const base = { title: cfg.title, units: vlut.units, lon, lat };

  if (entry.isDelta) {
    // Δ = window-mean(now) − window-mean(past), matching the rendered delta
    const cmp = compareDate();
    const [now, past] = await Promise.all([
      probePixelMean(cfg, windowSampleDates(state.date, win), z, x, y, px, py, vlut.lut),
      probePixelMean(cfg, windowSampleDates(cmp, win), z, x, y, px, py, vlut.lut),
    ]);
    if (now == null || past == null) return { ...base, delta: true, noData: true };
    return { ...base, delta: true, value: now - past };
  }
  if (entry.isRatio) {
    // fold = mean(now)/mean(past), with the same eps floor as the rendering
    const cmp = compareDate();
    const [now, past, cm] = await Promise.all([
      probePixelMean(cfg, windowSampleDates(state.date, win), z, x, y, px, py, vlut.lut),
      probePixelMean(cfg, windowSampleDates(cmp, win), z, x, y, px, py, vlut.lut),
      getColormapEntries(cfg.colormap),
    ]);
    if (now == null || past == null) return { ...base, ratio: true, noData: true };
    const eps = cfg.transparentZero && cm?.entries?.length ? cm.entries[0].lo / 2 : 0;
    return { ...base, ratio: true, value: (now + eps) / (past + eps) };
  }
  if (entry.isAggregate) {
    const v = await probePixelMean(cfg, windowSampleDates(state.date, win), z, x, y, px, py, vlut.lut);
    if (v == null) return { ...base, aggregated: true, noData: true };
    return { ...base, aggregated: true, value: v };
  }
  const v = await probePixel(cfg, state.date, z, x, y, px, py, vlut.lut);
  if (v == null) return { ...base, noData: true };
  return { ...base, value: v };
}

const probeEl = document.getElementById("value-probe");

function renderProbe(res, sx, sy) {
  if (!res) { probeEl.classList.add("hidden"); return; }
  const coord = `${Math.abs(res.lat).toFixed(2)}°${res.lat >= 0 ? "N" : "S"}, ` +
                `${Math.abs(res.lon).toFixed(2)}°${res.lon >= 0 ? "E" : "W"}`;
  let head;
  if (res.noData) {
    head = `<span class="vp-val vp-nd">no data</span>`;
  } else if (res.delta) {
    const v = `${res.value >= 0 ? "+" : "−"}${fmtVal(Math.abs(res.value))}`;
    head = `<span class="vp-val">Δ ${v}</span> <span class="vp-unit">${res.units}</span>`;
  } else if (res.ratio) {
    // multiplicative reading: ×1.8 more / ×2.3 less / ≈ same
    const fold = res.value;
    const near = Math.abs(Math.log(fold)) < 0.05;
    head = near
      ? `<span class="vp-val">≈ same</span>`
      : `<span class="vp-val">×${fmtVal(fold >= 1 ? fold : 1 / fold)}</span> ` +
        `<span class="vp-unit">${fold >= 1 ? "more" : "less"}</span>`;
  } else {
    head = `<span class="vp-val">${fmtVal(res.value)}</span> <span class="vp-unit">${res.units}</span>`;
  }
  const suffix = res.delta ? ` · Δ vs ${compareDate()}${state.windowDays > 1 ? ", " + windowLabel(state.windowDays) + " mean" : ""}`
    : res.ratio ? ` · vs ${compareDate()}, ratio of ${windowLabel(state.windowDays)} means`
    : res.aggregated ? ` · ${windowLabel(state.windowDays)} mean` : "";
  probeEl.innerHTML = `${head}<div class="vp-meta">${res.title}${suffix}<br/>${coord}</div>`;
  probeEl.style.left = `${Math.min(sx + 14, viewer.scene.canvas.clientWidth - 150)}px`;
  probeEl.style.top = `${Math.max(sy - 10, 4)}px`;
  probeEl.classList.remove("hidden");
}

/* The probe only fires after the cursor *rests* (dwell), so rotating/panning the
 * globe never triggers per-frame tile reads. Any movement hides it and restarts
 * the dwell timer; it computes once the mouse has been still for PROBE_DWELL ms. */
const PROBE_DWELL = 650;
let probeDwellTimer = null;
async function runProbe(x, y) {
  const cart = viewer.camera.pickEllipsoid({ x, y }, viewer.scene.globe.ellipsoid);
  if (!cart) { probeEl.classList.add("hidden"); return; }
  try { renderProbe(await probeValueAt(Cesium.Cartographic.fromCartesian(cart)), x, y); }
  catch { probeEl.classList.add("hidden"); }
}
new Cesium.ScreenSpaceEventHandler(viewer.scene.canvas).setInputAction((m) => {
  probeEl.classList.add("hidden");           // hide immediately while moving
  if (probeDwellTimer) clearTimeout(probeDwellTimer);
  if (!topColormapLayer()) return;
  const x = m.endPosition.x, y = m.endPosition.y;
  probeDwellTimer = setTimeout(() => runProbe(x, y), PROBE_DWELL);
}, Cesium.ScreenSpaceEventType.MOUSE_MOVE);
// Clicking reads the top layer's value immediately (no dwell wait) — unless
// the click is opening the pixel-state card, which includes the same value;
// two overlapping read-outs of one click would be noise.
new Cesium.ScreenSpaceEventHandler(viewer.scene.canvas).setInputAction((c) => {
  if (probeDwellTimer) clearTimeout(probeDwellTimer);
  if (pixelInspectorEngaged()) return;
  if (topColormapLayer() && !viewer.scene.pick(c.position)?.id?.kind) {
    runProbe(c.position.x, c.position.y);
  }
}, Cesium.ScreenSpaceEventType.LEFT_CLICK);
window.__runProbe = runProbe; // for tests

/* --------------------------------------------------------- pixel inspector */
/* Click anywhere on the globe → one card composing everything the app can
 * know about that point: every GIBS raster probed at the current date, the
 * climatology grids (with anomalies where a now-field has a matching normal),
 * live weather + a 7-day forecast, and what's nearby in the point datasets.
 * This is docs/PIXEL_STATE.md made clickable — state, memory, future, context.
 *
 * Open-Meteo is the ONE deliberate exception to the "browser talks only to
 * GIBS + GBIF" rule (see CLAUDE.md §3): key-free, CORS-open, and hit only by
 * an explicit click for a single point — never tile streaming. */

const PIXEL_RASTERS = ["sst", "sst-anom", "ssh-anom", "precip", "seaice", "snow", "aod", "lst",
  "soilmoisture", "ndvi", "grace", "ceres", "chlor", "salinity"];
const PIXEL_GRIDS = ["oisst", "gpcp", "eobs", "meteoswiss"];

function haversineKm(lon1, lat1, lon2, lat2) {
  const r = Math.PI / 180, R = 6371;
  const a = Math.sin((lat2 - lat1) * r / 2) ** 2 +
    Math.cos(lat1 * r) * Math.cos(lat2 * r) * Math.sin((lon2 - lon1) * r / 2) ** 2;
  return 2 * R * Math.asin(Math.sqrt(a));
}

// Tile + pixel coordinates of a lon/lat at GIBS level z (512px tiles).
function tileCoordsAt(lon, lat, z) {
  const span = (0.5625 / 2 ** z) * 512;
  const x = Math.floor((lon + 180) / span);
  const y = Math.floor((90 - lat) / span);
  const px = Math.min(511, Math.max(0, Math.floor((lon - (-180 + x * span)) / span * 512)));
  const py = Math.min(511, Math.max(0, Math.floor(((90 - y * span) - lat) / span * 512)));
  return { x, y, px, py };
}

// One raster channel: value at the pixel on the app's current date (z capped
// at 4 — inspector reads don't need street-level tiles, and 9 layers fetch).
async function pixelRasterValue(cfg, lon, lat) {
  const vlut = await getValueLut(cfg.colormap);
  if (!vlut) return null;
  const z = Math.min(cfg.maxLevel, 4);
  const t = tileCoordsAt(lon, lat, z);
  const v = await probePixel(cfg, state.date, z, t.x, t.y, t.px, t.py, vlut.lut);
  return v == null ? null : { v, units: vlut.units };
}

const pixelJsonCache = new Map();   // small point datasets, fetched once
function pixelJson(file) {
  if (!pixelJsonCache.has(file)) {
    pixelJsonCache.set(file, fetch(file).then((r) => r.json()).catch(() => null));
  }
  return pixelJsonCache.get(file);
}

/* All Open-Meteo family endpoints share the exception documented in
 * CLAUDE.md §3: key-free, CORS-open, single-point, click-triggered, and a
 * failed call just omits its card section. */
function omGet(url) {
  return fetch(url).then((r) => (r.ok ? r.json() : null)).catch(() => null);
}
function omLL(lon, lat) {
  return `latitude=${lat.toFixed(3)}&longitude=${lon.toFixed(3)}`;
}
function fetchOpenMeteo(lon, lat) {
  return omGet("https://api.open-meteo.com/v1/forecast?" + omLL(lon, lat) +
    "&current=temperature_2m,relative_humidity_2m,precipitation,pressure_msl," +
    "wind_speed_10m,wind_direction_10m,soil_moisture_0_to_1cm,soil_temperature_0cm," +
    "shortwave_radiation" +
    "&daily=temperature_2m_max,temperature_2m_min,precipitation_sum,wind_speed_10m_max" +
    "&timezone=UTC&forecast_days=7");
}
function fetchAirQuality(lon, lat) {
  return omGet("https://air-quality-api.open-meteo.com/v1/air-quality?" + omLL(lon, lat) +
    "&current=pm2_5,pm10,ozone,nitrogen_dioxide,european_aqi");
}
function fetchRiver(lon, lat) {   // GloFAS discharge for the 0.05° river cell
  return omGet("https://flood-api.open-meteo.com/v1/flood?" + omLL(lon, lat) +
    "&daily=river_discharge&forecast_days=1");
}
function fetchMarine(lon, lat) {  // ocean points only; land → null and the row is omitted
  return omGet("https://marine-api.open-meteo.com/v1/marine?" + omLL(lon, lat) +
    "&current=wave_height,wave_period");
}
/* The decadal future axis: CMIP6-HighResMIP downscaled, daily, per model.
 * Two 5-year windows (2045-49 vs 1991-95), three models → the pixel's own
 * projected warming and precipitation change, with cross-model range. */
const OM_CLIMATE_MODELS = ["EC_Earth3P_HR", "MPI_ESM1_2_XR", "MRI_AGCM3_2_S"];
function fetchClimateWindow(lon, lat, a, b) {
  return omGet("https://climate-api.open-meteo.com/v1/climate?" + omLL(lon, lat) +
    `&start_date=${a}&end_date=${b}&models=${OM_CLIMATE_MODELS.join(",")}` +
    "&daily=temperature_2m_mean,precipitation_sum");
}
function climateWindowStats(js) {
  if (!js?.daily) return null;
  const out = {};
  for (const m of OM_CLIMATE_MODELS) {
    const t = js.daily[`temperature_2m_mean_${m}`]?.filter((v) => v != null);
    const p = js.daily[`precipitation_sum_${m}`]?.filter((v) => v != null);
    if (!t?.length || !p?.length) continue;
    out[m] = {
      t: t.reduce((s, v) => s + v, 0) / t.length,
      p: (p.reduce((s, v) => s + v, 0) / p.length) * 365.25,   // mm/yr
    };
  }
  return Object.keys(out).length ? out : null;
}

/* When does a globe click open the full card vs read one layer's value?
 * "Everything we know" checked → always the card (explicit intent). Otherwise
 * a click reads the top active layer's value — the specific question you're
 * visibly asking — and only falls back to the card when NO colormapped layer
 * is active, because then there is no layer value to read instead. */
function pixelInspectorEngaged() {
  return !!document.getElementById("toggle-pixel")?.checked || !topColormapLayer();
}

/* Sample the baked Argo column at a lon/lat: the clicked 2° cell, or the
 * nearest of its 8 neighbours (coastal cells are often half-land). Returns
 * {levels, tNow, tNorm, sNow, sNorm} in real units, or null over land. */
function oceanColumnAt(oc, lon, lat) {
  const ix0 = Math.floor((lon - oc.west) / oc.dlon);
  const iy0 = Math.floor((lat - oc.south) / oc.dlat);
  const tryCell = (ix, iy) => {
    if (ix < 0 || ix >= oc.nx || iy < 0 || iy >= oc.ny) return null;
    // A neighbour cell only counts if it is genuinely NEAR the click (a
    // coastal cell whose own centre is half-land). Without this, an inland
    // click quietly serves the nearest sea 300+ km away — Bern got a
    // Mediterranean profile, 38 PSU and all.
    const latc = oc.south + (iy + 0.5) * oc.dlat;
    const lonc = oc.west + (ix + 0.5) * oc.dlon;
    if (haversineKm(lon, lat, lonc, latc) > 220) return null;
    const i = iy * oc.nx + ix;
    if (oc.t_now[0][i] == null) return null;
    const grab = (f) => oc.levels.map((_, k) => {
      const v = oc[f][k][i];
      return v == null ? null : v / 100;
    });
    return { levels: oc.levels, tNow: grab("t_now"), tNorm: grab("t_norm"),
             sNow: grab("s_now"), sNorm: grab("s_norm") };
  };
  for (const [dx, dy] of [[0, 0], [1, 0], [-1, 0], [0, 1], [0, -1], [1, 1], [-1, -1], [1, -1], [-1, 1]]) {
    const c = tryCell(ix0 + dx, iy0 + dy);
    if (c) return c;
  }
  return null;
}

/* Inline SVG: temperature vs depth, now (warm tone) against the same-month
 * normal (cool tone). Depth increases downward, like the water does. */
function columnProfileSvg(col) {
  const W = 250, H = 130, L = 30, B = 14;
  const pts = col.levels.map((p, k) => ({ p, tN: col.tNow[k], tM: col.tNorm[k] }))
    .filter((d) => d.tN != null && d.tM != null);
  if (pts.length < 4) return "";
  const tAll = pts.flatMap((d) => [d.tN, d.tM]);
  const tLo = Math.floor(Math.min(...tAll) - 0.5), tHi = Math.ceil(Math.max(...tAll) + 0.5);
  const pMax = pts[pts.length - 1].p;
  const x = (t) => L + (t - tLo) / (tHi - tLo) * (W - L - 6);
  const y = (p) => 4 + Math.sqrt(p / pMax) * (H - B - 8);   // sqrt: expand the upper ocean
  const line = (f) => pts.map((d) => `${x(f(d)).toFixed(1)},${y(d.p).toFixed(1)}`).join(" ");
  const depthTicks = [0, 200, 700, 1975].filter((p) => p <= pMax);
  return `<svg class="px-profile" viewBox="0 0 ${W} ${H}" width="100%">` +
    depthTicks.map((p) =>
      `<line x1="${L}" y1="${y(p)}" x2="${W - 6}" y2="${y(p)}" stroke="#30363d" stroke-width="0.6"/>` +
      `<text x="2" y="${y(p) + 3}" fill="#8b949e" font-size="8">${p >= 1000 ? "2 km" : p + " m"}</text>`).join("") +
    `<text x="${L}" y="${H - 2}" fill="#8b949e" font-size="8">${tLo}°C</text>` +
    `<text x="${W - 6}" y="${H - 2}" fill="#8b949e" font-size="8" text-anchor="end">${tHi}°C</text>` +
    `<polyline points="${line((d) => d.tM)}" fill="none" stroke="#58a6ff" stroke-width="1.4"/>` +
    `<polyline points="${line((d) => d.tN)}" fill="none" stroke="#f0883e" stroke-width="1.6"/>` +
    `</svg>`;
}

const pixelCardEl = document.getElementById("pixel-card");
function pixelRow(label, value, cls = "") {
  return `<div class="px-row ${cls}"><span class="px-label">${label}</span><span class="px-val">${value}</span></div>`;
}

async function showPixelState(carto) {
  const lon = Cesium.Math.toDegrees(carto.longitude);
  const lat = Cesium.Math.toDegrees(carto.latitude);
  const coord = `${Math.abs(lat).toFixed(2)}°${lat >= 0 ? "N" : "S"} ` +
                `${Math.abs(lon).toFixed(2)}°${lon >= 0 ? "E" : "W"}`;
  pixelCardEl.innerHTML =
    `<div class="px-head"><strong>Pixel state</strong> · ${coord}` +
    `<button class="px-close" aria-label="Close">×</button></div>` +
    `<div class="px-body"><div class="px-loading">Reading this point…</div></div>`;
  pixelCardEl.classList.remove("hidden");
  pixelCardEl.querySelector(".px-close").addEventListener("click", () =>
    pixelCardEl.classList.add("hidden"));

  // Everything in parallel; the card renders once, complete.
  const rasterCfgs = PIXEL_RASTERS.map((id) => GIBS_LAYERS.find((l) => l.id === id));
  const gridCfgs = PIXEL_GRIDS.map((id) => GIBS_LAYERS.find((l) => l.id === id));
  const [rasters, grids, meteo, air, river, marine, climNow, climFut, oceanCol, oceanSurf, stations, trace, argo] = await Promise.all([
    Promise.all(rasterCfgs.map((cfg) => pixelRasterValue(cfg, lon, lat).catch(() => null))),
    Promise.all(gridCfgs.map((cfg) => loadGrid(cfg).then((g) => g && sampleGrid(g, lon, lat)).catch(() => null))),
    fetchOpenMeteo(lon, lat),
    fetchAirQuality(lon, lat),
    fetchRiver(lon, lat),
    fetchMarine(lon, lat),
    fetchClimateWindow(lon, lat, "1991-01-01", "1995-12-31"),
    fetchClimateWindow(lon, lat, "2045-01-01", "2049-12-31"),
    pixelJson("data/ocean_column.json"),
    pixelJson("data/ocean_surface.json"),
    pixelJson("data/stations.geojson"),
    pixelJson("data/climatetrace.json"),
    pixelJson("data/argo.json"),
  ]);
  if (pixelCardEl.classList.contains("hidden")) return;   // closed while loading

  const sec = [];

  /* -- live weather now + the near future (the prediction axis) ------------ */
  if (meteo?.current) {
    const c = meteo.current;
    let rows = pixelRow("Air temperature", `${fmtVal(c.temperature_2m)} °C`) +
      pixelRow("Wind", `${fmtVal(c.wind_speed_10m)} km/h from ${Math.round(c.wind_direction_10m)}°`) +
      pixelRow("Humidity · pressure", `${Math.round(c.relative_humidity_2m)} % · ${Math.round(c.pressure_msl)} hPa`) +
      (c.precipitation > 0 ? pixelRow("Precipitation now", `${fmtVal(c.precipitation)} mm/h`) : "");
    if (c.soil_moisture_0_to_1cm != null) {
      rows += pixelRow("Soil (top cm)", `${fmtVal(c.soil_moisture_0_to_1cm)} m³/m³ · ${fmtVal(c.soil_temperature_0cm)} °C`);
    }
    if (c.shortwave_radiation != null && c.shortwave_radiation > 0) {
      rows += pixelRow("Solar radiation", `${Math.round(c.shortwave_radiation)} W/m²`);
    }
    const mc = marine?.current;
    if (mc?.wave_height != null) {
      rows += pixelRow("Waves", `${fmtVal(mc.wave_height)} m` +
        (mc.wave_period != null ? ` every ${fmtVal(mc.wave_period)} s` : ""));
    }
    const rd = river?.daily?.river_discharge?.[0];
    if (rd != null && rd > 0) {
      rows += pixelRow("River discharge", `${fmtVal(rd)} m³/s in this GloFAS cell`);
    }
    if (Number.isFinite(meteo.elevation) && Math.abs(meteo.elevation) > 1) {
      rows = pixelRow("Elevation", `${Math.round(meteo.elevation)} m`) + rows;
    }
    const d = meteo.daily;
    if (d?.time?.length) {
      const days = d.time.map((t, i) =>
        `<div class="px-day"><span>${t.slice(5)}</span>` +
        `<span>${Math.round(d.temperature_2m_min[i])}–${Math.round(d.temperature_2m_max[i])}°</span>` +
        `<span>${d.precipitation_sum[i] > 0.4 ? fmtVal(d.precipitation_sum[i]) + " mm" : "·"}</span></div>`).join("");
      rows += `<div class="px-forecast">${days}</div>`;
    }
    sec.push(`<div class="px-sec"><div class="px-sec-title">Now &amp; next 7 days <span class="px-src">live · Open-Meteo</span></div>${rows}</div>`);
  }

  /* -- air quality (CAMS via Open-Meteo) ----------------------------------- */
  if (air?.current && air.current.pm2_5 != null) {
    const a = air.current;
    const rows =
      pixelRow("PM2.5 · PM10", `${fmtVal(a.pm2_5)} · ${fmtVal(a.pm10)} µg/m³`) +
      pixelRow("Ozone · NO₂", `${fmtVal(a.ozone)} · ${fmtVal(a.nitrogen_dioxide)} µg/m³`) +
      (a.european_aqi != null ? pixelRow("Air-quality index", `${Math.round(a.european_aqi)} (EU scale, lower is better)`) : "");
    sec.push(`<div class="px-sec"><div class="px-sec-title">Air quality <span class="px-src">live · CAMS</span></div>${rows}</div>`);
  }

  /* -- satellite fields at the app's current date -------------------------- */
  // No derived SST-vs-normal line: the baked OISST normal is the ANNUAL mean,
  // so the difference would mostly be the seasonal cycle (~±4 °C at
  // midlatitudes), not a climate signal. The seasonally-correct departure is
  // the "SST anomalies" row (MUR25 vs its own monthly climatology), and the
  // annual mean itself prints as its own line under Climate normals.
  const rrows = rasters.map((r, i) => {
    if (!r) return "";
    const cfg = rasterCfgs[i];
    return pixelRow(cfg.title.replace(/\s*\(.*\)$/, ""), `${fmtVal(r.v)} ${r.units}`);
  }).join("");
  if (rrows) {
    sec.push(`<div class="px-sec"><div class="px-sec-title">Satellite fields <span class="px-src">${state.date} · NASA GIBS</span></div>${rrows}</div>`);
  }

  /* -- long-term normals (the memory channels) ----------------------------- */
  const gtitles = ["SST 1991–2020 annual mean", "Precip normal (GPCP)", "Precip normal (E-OBS)", "Precip normal (MeteoSwiss)"];
  const gunits = ["°C", "mm/yr", "mm/yr", "mm/yr"];
  const grows = grids.map((v, i) =>
    v == null ? "" : pixelRow(gtitles[i], `${fmtVal(v)} ${gunits[i]}`)).join("");
  if (grows) {
    sec.push(`<div class="px-sec"><div class="px-sec-title">Climate normals</div>${grows}</div>`);
  }

  /* -- ocean circulation at the point (GLORYS monthly mean) ---------------- */
  if (oceanSurf) {
    const ix = Math.floor((lon - oceanSurf.west) / oceanSurf.dlon);
    const iy = Math.floor((lat - oceanSurf.south) / oceanSurf.dlat);
    const i = iy * oceanSurf.nx + ix;
    const u = oceanSurf.u?.[i], v = oceanSurf.v?.[i];
    if (u != null && v != null) {
      const spd = Math.hypot(u, v) / 100;                       // cm/s → m/s
      const brg = (Math.atan2(u, v) * 180 / Math.PI + 360) % 360; // "toward"
      const rose = "N NE E SE S SW W NW".split(" ")[Math.round(brg / 45) % 8];
      let rows = pixelRow("Surface current",
        `${fmtVal(spd)} m/s toward ${rose} (${Math.round(brg)}°)`);
      if (oceanSurf.mld?.[i] != null) rows += pixelRow("Mixed-layer depth", `${oceanSurf.mld[i]} m`);
      if (oceanSurf.zos?.[i] != null) {
        const z = oceanSurf.zos[i];
        rows += pixelRow("Sea surface height", `${z >= 0 ? "+" : "−"}${Math.abs(z)} cm`);
      }
      sec.push(`<div class="px-sec"><div class="px-sec-title">Ocean circulation <span class="px-src">GLORYS · ${oceanSurf.month}</span></div>${rows}</div>`);
    }
  }

  /* -- the ocean beneath: Argo T/S column, now vs the same-month normal ---- */
  const col = oceanCol ? oceanColumnAt(oceanCol, lon, lat) : null;
  if (col) {
    const dT = col.levels.map((_, k) =>
      col.tNow[k] != null && col.tNorm[k] != null ? col.tNow[k] - col.tNorm[k] : null);
    // thickness-weighted mean warming of the upper 700 m — where the heat goes
    let wsum = 0, w = 0;
    for (let k = 0; k < col.levels.length && col.levels[k] <= 700; k++) {
      if (dT[k] == null) continue;
      const thick = (col.levels[k + 1] ?? 700) - (k ? col.levels[k - 1] : 0);
      wsum += dT[k] * thick; w += thick;
    }
    const heat = w ? wsum / w : null;
    // warm-layer depth: where T first drops 0.5 °C below the 10 dbar value
    let wl = null;
    const t10 = col.tNow[1];
    if (t10 != null) {
      for (let k = 2; k < col.levels.length; k++) {
        if (col.tNow[k] != null && col.tNow[k] < t10 - 0.5) { wl = col.levels[k]; break; }
      }
    }
    let rows = columnProfileSvg(col) +
      `<div class="px-day"><span style="color:#f0883e">━ ${oceanCol.month}</span>` +
      `<span style="color:#58a6ff">━ normal</span><span>(same month, 2004–18)</span></div>`;
    if (dT[0] != null) {
      rows += pixelRow("Surface vs normal",
        `<span class="${dT[0] >= 0 ? "px-warm" : "px-cool"}">${dT[0] >= 0 ? "+" : "−"}${fmtVal(Math.abs(dT[0]))} °C</span>`);
    }
    if (heat != null) {
      rows += pixelRow("Upper 700 m vs normal",
        `<span class="${heat >= 0 ? "px-warm" : "px-cool"}">${heat >= 0 ? "+" : "−"}${fmtVal(Math.abs(heat))} °C</span> stored heat`);
    }
    if (wl != null) rows += pixelRow("Warm-layer depth", `~${wl} m`);
    if (col.sNow[0] != null && col.sNorm[0] != null) {
      const ds = col.sNow[0] - col.sNorm[0];
      rows += pixelRow("Surface salinity", `${fmtVal(col.sNow[0])} PSU ` +
        `<span class="px-src">(${ds >= 0 ? "+" : "−"}${Math.abs(ds).toFixed(2)} vs normal${ds < -0.05 ? " — fresher" : ds > 0.05 ? " — saltier" : ""})</span>`);
    }
    sec.push(`<div class="px-sec"><div class="px-sec-title">Ocean column 0–2000 m <span class="px-src">Argo floats · ${oceanCol.month}</span></div>${rows}</div>`);
  }

  /* -- the decadal future axis: this pixel's own 2050 trajectory ----------- */
  const base = climateWindowStats(climNow);
  const fut = climateWindowStats(climFut);
  if (base && fut) {
    const models = OM_CLIMATE_MODELS.filter((m) => base[m] && fut[m]);
    if (models.length >= 2) {
      const dt = models.map((m) => fut[m].t - base[m].t);
      const dp = models.map((m) => (fut[m].p - base[m].p) / Math.max(base[m].p, 1) * 100);
      const mean = (a) => a.reduce((s, v) => s + v, 0) / a.length;
      const rng = (a) => [Math.min(...a), Math.max(...a)];
      const [tlo, thi] = rng(dt), [plo, phi] = rng(dp);
      const mt = mean(dt), mp = mean(dp);
      const rows =
        pixelRow("Temperature", `<span class="${mt >= 0 ? "px-warm" : "px-cool"}">${mt >= 0 ? "+" : "−"}${fmtVal(Math.abs(mt))} °C</span>` +
          ` <span class="px-src">(models ${tlo >= 0 ? "+" : "−"}${fmtVal(Math.abs(tlo))}…${thi >= 0 ? "+" : "−"}${fmtVal(Math.abs(thi))})</span>`) +
        pixelRow("Precipitation", `${mp >= 0 ? "+" : "−"}${Math.round(Math.abs(mp))} %` +
          ` <span class="px-src">(${plo >= 0 ? "+" : "−"}${Math.round(Math.abs(plo))}…${phi >= 0 ? "+" : "−"}${Math.round(Math.abs(phi))} %)</span>`);
      sec.push(`<div class="px-sec"><div class="px-sec-title">2045–49 vs 1991–95 <span class="px-src">CMIP6-HighResMIP · ${models.length} models</span></div>${rows}</div>`);
    }
  }

  /* -- context: what observes / affects this point ------------------------- */
  const nrows = [];
  if (stations?.features) {
    let best = null, bd = Infinity;
    for (const f of stations.features) {
      const [slon, slat] = f.geometry.coordinates;
      const d = haversineKm(lon, lat, slon, slat);
      if (d < bd) { bd = d; best = f; }
    }
    if (best) nrows.push(pixelRow("Nearest monitoring site", `${best.properties.name} · ${Math.round(bd)} km`));
  }
  if (argo?.floats) {
    let n300 = 0, bd = Infinity;
    for (const [flon, flat] of argo.floats) {
      const d = haversineKm(lon, lat, flon, flat);
      if (d < bd) bd = d;
      if (d < 300) n300++;
    }
    if (bd < 3000) nrows.push(pixelRow("Argo floats", `${n300} within 300 km · nearest ${Math.round(bd)} km`));
  }
  if (trace?.assets) {
    let best = null, bd = Infinity, n100 = 0;
    for (const a of trace.assets) {
      const d = haversineKm(lon, lat, a[0], a[1]);
      if (d < bd) { bd = d; best = a; }
      if (d < 100) n100++;
    }
    if (bd < 500 && best) {
      nrows.push(pixelRow("Top-1000 emitters", `${n100} within 100 km · nearest: ${best[3]} (${fmtVal(best[2])} Mt/yr, ${Math.round(bd)} km)`));
    }
  }
  // glaciers only if the (7 MB) inventory is already loaded — no click-cost
  if (glacierData) {
    let n = 0, dh = 0, ndh = 0;
    for (let i = 0; i < glacierData.count; i++) {
      if (haversineKm(lon, lat, glacierData.lon[i], glacierData.lat[i]) < 100) {
        n++;
        const v = glacierData.dhdt[i];
        if (v != null) { dh += v; ndh++; }
      }
    }
    if (n) nrows.push(pixelRow("Glaciers within 100 km", `${n}${ndh ? ` · mean ${fmtVal(dh / ndh)} m/yr thickness change` : ""}`));
  }
  if (nrows.length) {
    sec.push(`<div class="px-sec"><div class="px-sec-title">Context nearby</div>${nrows.join("")}</div>`);
  }

  sec.push(`<div class="px-note">Channels &amp; roles: <a href="docs/PIXEL_STATE.md" target="_blank" rel="noopener">PIXEL_STATE.md</a> · weather &amp; forecast by <a href="https://open-meteo.com/" target="_blank" rel="noopener">Open-Meteo</a></div>`);
  pixelCardEl.querySelector(".px-body").innerHTML = sec.join("");
}

document.addEventListener("keydown", (e) => {
  if (e.key === "Escape") pixelCardEl.classList.add("hidden");
});

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
    if (e.target.checked) maybeDatelessToast("stations");
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

/* ---------------------------------------------------- sea-level budget dashboard */

let seaLevelData = null;
const SL_COMPONENTS = [
  { key: "steric", label: "Thermal expansion (steric)", color: "#3987e5" },
  { key: "glaciers", label: "Glaciers", color: "#d95926" },
  { key: "greenland", label: "Greenland Ice Sheet", color: "#199e70" },
  { key: "antarctica", label: "Antarctic Ice Sheet", color: "#c98500" },
  { key: "tws", label: "Land water storage", color: "#d55181" },
];

async function loadSeaLevel() {
  if (seaLevelData) return;
  seaLevelData = await (await fetch("data/sealevel.json")).json();
  const { years, components, altimetry } = seaLevelData;
  const obs = components.observed;

  // headline stats
  const total = obs[obs.length - 1] - obs[0];
  setStat("sl-total", total, "mm rise, 1900–2018");
  // satellite-era rate: linear fit of altimetry (mm vs decimal year)
  const rate = linTrend(altimetry.t, altimetry.v);
  document.querySelector("#sl-rate .stat-value").textContent = rate.toFixed(1);
  document.querySelector("#sl-rate .stat-sub").textContent = "mm/yr (satellite era)";
  // largest contributor over the record
  let big = SL_COMPONENTS[0], bigv = -1e9;
  for (const c of SL_COMPONENTS) {
    const v = components[c.key][components[c.key].length - 1] - components[c.key][0];
    if (v > bigv) { bigv = v; big = c; }
  }
  document.querySelector("#sl-driver .stat-value").textContent = `${Math.round(bigv)}`;
  document.querySelector("#sl-driver .stat-sub").textContent = `mm from ${big.label.split(" (")[0].toLowerCase()}`;

  const leg = document.getElementById("sl-legend");
  leg.innerHTML = `<span style="color:#fff"><b>━ Observed GMSL</b></span>` +
    `<span style="color:#898781">┄ Summed budget</span>` +
    SL_COMPONENTS.map((c) => `<span style="color:${c.color}">━ ${c.label}</span>`).join("") +
    `<span style="color:#4493f8">┈ Satellite altimetry</span>`;
  drawSeaLevelChart();
  window.addEventListener("resize", () => { if (!document.getElementById("panel-sealevel").classList.contains("hidden")) drawSeaLevelChart(); });
}

function linTrend(t, v) {
  const n = t.length;
  const mt = t.reduce((s, x) => s + x, 0) / n;
  const mv = v.reduce((s, x) => s + x, 0) / n;
  let num = 0, den = 0;
  for (let i = 0; i < n; i++) { num += (t[i] - mt) * (v[i] - mv); den += (t[i] - mt) ** 2; }
  return num / den;
}

function drawSeaLevelChart() {
  const { years, components, altimetry } = seaLevelData;
  const canvas = document.getElementById("sl-chart");
  const wrap = canvas.parentElement;
  const cssW = wrap.clientWidth, cssH = 210;
  const dpr = window.devicePixelRatio || 1;
  canvas.width = cssW * dpr; canvas.height = cssH * dpr;
  canvas.style.width = cssW + "px"; canvas.style.height = cssH + "px";
  const ctx = canvas.getContext("2d");
  ctx.scale(dpr, dpr);

  const M = { l: 30, r: 8, t: 8, b: 18 };
  const W = cssW - M.l - M.r, H = cssH - M.t - M.b;
  const compVals = SL_COMPONENTS.flatMap((c) => components[c.key]);
  const allV = [...components.observed, ...components.sum, ...compVals, ...altimetry.v];
  const y0 = Math.floor(Math.min(...allV) / 25) * 25;
  const y1 = Math.ceil(Math.max(...allV) / 25) * 25;
  const X = (yr) => M.l + ((yr - years[0]) / (years[years.length - 1] - years[0])) * W;
  const Y = (v) => M.t + (1 - (v - y0) / (y1 - y0)) * H;
  const line = (xs, ys, i0 = 0) => {
    ctx.beginPath();
    let started = false;
    for (let i = i0; i < xs.length; i++) {
      if (ys[i] == null) { started = false; continue; }
      if (!started) { ctx.moveTo(X(xs[i]), Y(ys[i])); started = true; }
      else ctx.lineTo(X(xs[i]), Y(ys[i]));
    }
    ctx.stroke();
  };

  const draw = (hoverYear) => {
    ctx.clearRect(0, 0, cssW, cssH);
    ctx.font = "10px system-ui, sans-serif";
    for (let v = y0; v <= y1; v += 50) {
      ctx.strokeStyle = "#2c2c2a"; ctx.lineWidth = 1;
      ctx.beginPath(); ctx.moveTo(M.l, Y(v)); ctx.lineTo(cssW - M.r, Y(v)); ctx.stroke();
      ctx.fillStyle = "#898781"; ctx.textAlign = "right"; ctx.textBaseline = "middle";
      ctx.fillText(String(v), M.l - 4, Y(v));
    }
    // zero reference line
    if (y0 < 0 && y1 > 0) {
      ctx.strokeStyle = "#4a4a47"; ctx.lineWidth = 1;
      ctx.beginPath(); ctx.moveTo(M.l, Y(0)); ctx.lineTo(cssW - M.r, Y(0)); ctx.stroke();
    }
    ctx.textAlign = "center"; ctx.textBaseline = "top";
    for (let yr = 1900; yr <= years[years.length - 1]; yr += 20) {
      ctx.fillStyle = "#898781"; ctx.fillText(String(yr), X(yr), M.t + H + 5);
    }
    // component lines
    ctx.lineWidth = 1.5; ctx.lineJoin = "round";
    for (const c of SL_COMPONENTS) { ctx.strokeStyle = c.color; line(years, components[c.key]); }
    // summed budget (grey dashed) — should track observed = closure
    ctx.strokeStyle = "#898781"; ctx.lineWidth = 1.5; ctx.setLineDash([5, 3]);
    line(years, components.sum); ctx.setLineDash([]);
    // observed GMSL (white, thick)
    ctx.strokeStyle = "#ffffff"; ctx.lineWidth = 2.5; line(years, components.observed);
    // satellite altimetry (accent dashed, modern era)
    ctx.strokeStyle = "#4493f8"; ctx.lineWidth = 1.5; ctx.setLineDash([2, 3]);
    line(altimetry.t, altimetry.v); ctx.setLineDash([]);
    // hover crosshair
    if (hoverYear != null) {
      const i = hoverYear - years[0];
      if (i >= 0 && i < years.length) {
        ctx.strokeStyle = "#52514e"; ctx.lineWidth = 1;
        ctx.beginPath(); ctx.moveTo(X(years[i]), M.t); ctx.lineTo(X(years[i]), M.t + H); ctx.stroke();
        ctx.fillStyle = "#ffffff";
        ctx.beginPath(); ctx.arc(X(years[i]), Y(components.observed[i]), 3, 0, 7); ctx.fill();
      }
    }
  };
  draw(null);

  const tip = document.getElementById("sl-tooltip");
  canvas.onmousemove = (e) => {
    const rect = canvas.getBoundingClientRect();
    const yr = Math.round(years[0] + ((e.clientX - rect.left - M.l) / W) * (years[years.length - 1] - years[0]));
    const i = yr - years[0];
    if (i < 0 || i >= years.length) { tip.classList.add("hidden"); return; }
    draw(yr);
    const parts = SL_COMPONENTS.map((c) => `<span style="color:${c.color}">■</span> ${c.label.split(" (")[0]}: ${(components[c.key][i] ?? 0).toFixed(0)} mm`).join("<br/>");
    tip.innerHTML = `<b>${yr}</b> · observed ${components.observed[i].toFixed(0)} mm<br/>${parts}`;
    tip.style.left = `${Math.min(Math.max(e.clientX - rect.left - 70, 4), cssW - 150)}px`;
    tip.classList.remove("hidden");
  };
  canvas.onmouseleave = () => { tip.classList.add("hidden"); draw(null); };
}

/* --------------------------------------------------- global temperature (GISTEMP) */

let gistempData = null;
async function loadTemp() {
  if (gistempData) return;
  gistempData = await (await fetch("data/gistemp.json")).json();
  const { years, land_ocean, land_only } = gistempData;
  const iLast = land_ocean.length - 1;
  document.querySelector("#temp-lo .stat-value").textContent = `+${land_ocean[iLast].toFixed(2)}`;
  const lastLand = [...land_only].reverse().find((v) => v != null);
  document.querySelector("#temp-land .stat-value").textContent = `+${lastLand.toFixed(2)}`;
  document.querySelector("#temp-since .stat-value").textContent =
    `+${(land_ocean[iLast] - land_ocean[0]).toFixed(2)}`;
  document.getElementById("temp-legend").innerHTML =
    `<span style="color:#d95926">━ Land only</span><span style="color:#3987e5">━ Land + ocean</span>`;
  drawTempChart();
  window.addEventListener("resize", () => {
    if (!document.getElementById("panel-temp").classList.contains("hidden")) drawTempChart();
  });
}

function drawTempChart() {
  const { years, land_ocean, land_only } = gistempData;
  const canvas = document.getElementById("temp-chart");
  const wrap = canvas.parentElement;
  const cssW = wrap.clientWidth, cssH = 200;
  const dpr = window.devicePixelRatio || 1;
  canvas.width = cssW * dpr; canvas.height = cssH * dpr;
  canvas.style.width = cssW + "px"; canvas.style.height = cssH + "px";
  const ctx = canvas.getContext("2d"); ctx.scale(dpr, dpr);
  const M = { l: 30, r: 8, t: 8, b: 18 };
  const W = cssW - M.l - M.r, H = cssH - M.t - M.b;
  const all = [...land_ocean, ...land_only].filter((v) => v != null);
  const y0 = Math.floor(Math.min(...all) * 2) / 2, y1 = Math.ceil(Math.max(...all) * 2) / 2;
  const X = (yr) => M.l + ((yr - years[0]) / (years[years.length - 1] - years[0])) * W;
  const Y = (v) => M.t + (1 - (v - y0) / (y1 - y0)) * H;
  const line = (arr) => {
    ctx.beginPath(); let started = false;
    for (let i = 0; i < years.length; i++) {
      if (arr[i] == null) { started = false; continue; }
      if (!started) { ctx.moveTo(X(years[i]), Y(arr[i])); started = true; }
      else ctx.lineTo(X(years[i]), Y(arr[i]));
    }
    ctx.stroke();
  };
  const draw = (hoverYear) => {
    ctx.clearRect(0, 0, cssW, cssH);
    ctx.font = "10px system-ui, sans-serif";
    for (let v = y0; v <= y1; v += 0.5) {
      ctx.strokeStyle = v === 0 ? "#4a4a47" : "#2c2c2a"; ctx.lineWidth = 1;
      ctx.beginPath(); ctx.moveTo(M.l, Y(v)); ctx.lineTo(cssW - M.r, Y(v)); ctx.stroke();
      ctx.fillStyle = "#898781"; ctx.textAlign = "right"; ctx.textBaseline = "middle";
      ctx.fillText(v.toFixed(1), M.l - 4, Y(v));
    }
    ctx.textAlign = "center"; ctx.textBaseline = "top";
    for (let yr = 1900; yr <= years[years.length - 1]; yr += 20) {
      ctx.fillStyle = "#898781"; ctx.fillText(String(yr), X(yr), M.t + H + 5);
    }
    ctx.lineWidth = 1.8; ctx.lineJoin = "round";
    ctx.strokeStyle = "#d95926"; line(land_only);   // land warms faster
    ctx.strokeStyle = "#3987e5"; line(land_ocean);
    if (hoverYear != null) {
      const i = hoverYear - years[0];
      if (i >= 0 && i < years.length) {
        ctx.strokeStyle = "#52514e"; ctx.lineWidth = 1;
        ctx.beginPath(); ctx.moveTo(X(years[i]), M.t); ctx.lineTo(X(years[i]), M.t + H); ctx.stroke();
      }
    }
  };
  draw(null);
  const tip = document.getElementById("temp-tooltip");
  canvas.onmousemove = (e) => {
    const rect = canvas.getBoundingClientRect();
    const yr = Math.round(years[0] + ((e.clientX - rect.left - M.l) / W) * (years[years.length - 1] - years[0]));
    const i = yr - years[0];
    if (i < 0 || i >= years.length) { tip.classList.add("hidden"); return; }
    draw(yr);
    const lo = land_ocean[i], la = land_only[i];
    tip.innerHTML = `<b>${yr}</b><br/><span style="color:#d95926">land</span> ${la != null ? "+" + la.toFixed(2) : "–"} °C<br/>` +
      `<span style="color:#3987e5">land+ocean</span> ${lo != null ? "+" + lo.toFixed(2) : "–"} °C`;
    tip.style.left = `${Math.min(Math.max(e.clientX - rect.left - 55, 4), cssW - 120)}px`;
    tip.classList.remove("hidden");
  };
  canvas.onmouseleave = () => { tip.classList.add("hidden"); draw(null); };
}

/* --------------------------------------------------------------------- tabs */

const tabs = { layers: "panel-layers", temp: "panel-temp", amoc: "panel-amoc", sealevel: "panel-sealevel",
  catalog: "panel-catalog", about: "panel-about" };
for (const t of Object.keys(tabs)) {
  document.getElementById(`tab-${t}`).addEventListener("click", () => {
    for (const [k, panel] of Object.entries(tabs)) {
      document.getElementById(panel).classList.toggle("hidden", k !== t);
      document.getElementById(`tab-${k}`).classList.toggle("active", k === t);
    }
    if (t === "amoc") loadAmoc();
    if (t === "sealevel") loadSeaLevel();
    if (t === "temp") loadTemp();
  });
}

/* --------------------------------------------------------------------- init */

buildLayerPanel();
updateLegends();
initSpeciesUI();
loadStations();
loadCatalog();

/* Test hook: stable handle for the Playwright suite (tests/) — not a public API. */
window.__earth = {
  viewer,
  get baseImageryLayer() { return baseImageryLayer; },
  parseColormap,
  parseColormapEntries,
  windowSampleDates,
  addDays,
  windowLabel,
  SSTAggregateProvider,
  AggregateProvider,
  DeltaProvider,
  RatioProvider,
  getValueLut,
  SSTEnsembleProvider,
  spreadColor,
  get ensembleLayer() { return sstEnsembleLayer; },
  deltaColor,
  state,
  pointLayers,
  GIBS_LAYERS,
  GIBSGeographicTilingScheme,
  compareDate,
  get stations() { return stationsDs; },
  get rapid() { return rapidData; },
  get sealevel() { return seaLevelData; },
  loadSeaLevel,
  loadTemp,
  get gistemp() { return gistempData; },
  linTrend,
  probeValueAt,
  showPixelState,
  pixelInspectorEngaged,
  oceanColumnAt,
  loadGlaciers,
  get glacierCollection() { return glacierCollection; },
  get glacierData() { return glacierData; },
  colorGlaciers,
  updateGbifLayer,
  get gbifLayer() { return gbifLayer; },
  get gbifSpecies() { return gbifSpecies; },
  get gbifData() { return gbifData; },
  get catalog() { return CATALOG; },
  GridProvider,
  activeLayerChips,
  updateActiveChips,
  STATIC_LAYER_CHIPS,
  showToast,
  datelessToast,
  loadGrid,
  sampleGrid,
  rampColor,
  gibsTime,
};

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
    // AMSR2 is aging: GIBS stops serving sea-ice tiles at 2025-09 (same
    // instrument family as the soil-moisture layer). endTime clamps later
    // dates to the last served one; clampToast explains on enable.
    endTime: "2025-09-01",
    title: "Sea ice concentration (AMSR2)",
    ext: "png", tms: "2km", maxLevel: 5,
    start: "2012-07-02", timed: true, on: false,
    meta: "Passive-microwave, both poles · tiles end 2025-09",
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
    meta: "Daytime land skin temperature — probe reads °C (GIBS legend image is in K)",
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
    id: "dist-alert",
    // CLASSIFICATION raster: the pixel carries a class (first detection /
    // provisional / confirmed, under or over 50% cover loss, and "finished"),
    // not a number — hence `classmap` rather than `colormap`. Averaging or
    // subtracting class codes is meaningless, so this layer takes neither
    // `aggregable` nor `deltaRange` (posture matrix, CLAUDE.md §2.5). The
    // change signal is already IN the product: it only paints where
    // vegetation was lost since the year began.
    classmap: "https://gibs.earthdata.nasa.gov/colormaps/v1.3/OPERA_Vegetation_Disturbance_Status.xml",
    legend: "https://gibs.earthdata.nasa.gov/legends/OPERA_Vegetation_Disturbance_Status_H.svg",
    doc: "https://www.jpl.nasa.gov/go/opera/products/dist-product-suite/",
    layer: "OPERA_L3_DIST-ALERT-HLS_Color_Index",
    classNote: "&lt;50% / &ge;50% is how much of the vegetation cover went · " +
      "confirmed = a second clear image agreed · finished = the loss stopped progressing",
    title: "Vegetation disturbance alerts (OPERA DIST-ALERT)",
    ext: "png", tms: "31.25m", maxLevel: 11,
    start: "2023-01-01", timed: true, on: false,
    meta: "30 m near-real-time vegetation loss — zoom in to see individual clearings",
  },
  {
    id: "dist-ann",
    classmap: "https://gibs.earthdata.nasa.gov/colormaps/v1.3/OPERA_Vegetation_Disturbance_Annual.xml",
    legend: "https://gibs.earthdata.nasa.gov/legends/OPERA_Vegetation_Disturbance_Annual_H.svg",
    doc: "https://www.jpl.nasa.gov/go/opera/products/dist-product-suite/",
    layer: "OPERA_L3_DIST-ANN-HLS_Color_Index",
    // GIBS serves one tile date per YEAR (2023/2024/2025-01-01): the annual
    // summary of confirmed disturbance. `annual` snaps the date to Jan 1 of
    // its year the way `monthly` snaps to the 1st of its month.
    annual: true,
    endTime: "2025-01-01",
    classNote: "the year's settled tally — everything provisional has been resolved",
    title: "Vegetation loss, annual summary (OPERA DIST-ANN)",
    ext: "png", tms: "31.25m", maxLevel: 11,
    start: "2023-01-01", timed: true, on: false,
    meta: "Confirmed vegetation loss for a whole year — the date's YEAR picks it (2023–2025)",
  },
  /* ------------------------------------------------------------- the fine tier
   * Everything below is 30 m or finer, served by GIBS in the same tile scheme
   * as the rest (so CLAUDE.md §3's "GIBS and GBIF only" still holds). The
   * daily ones are SWATH products: on any one date they cover the strips the
   * satellite flew, not the world, so a full-globe view of them is mostly
   * blank and every date step would fetch a fresh set of mostly-blank tiles.
   * `fine: <km>` is the gate: above that camera height the layer is kept but
   * HIDDEN (Cesium creates no tile skeletons for a hidden layer, so nothing is
   * requested at all — see fineGate()), and its row says to zoom in. The
   * static ones (elevation, built-up) have meaningful overviews and stay
   * ungated. Posture (§2.5): photographs and classifications take neither
   * `aggregable` nor `deltaRange`; elevation is untimed, so nothing to average. */
  {
    id: "hls-s30",
    doc: "https://lpdaac.usgs.gov/products/hlss30v002/",
    layer: "HLS_S30_Nadir_BRDF_Adjusted_Reflectance",
    title: "Sentinel-2 true colour (HLS, 30 m)",
    ext: "png", tms: "31.25m", maxLevel: 11, fine: 500,
    start: "2015-11-28", timed: true, on: false,
    meta: "Sentinel-2 surface reflectance, 30 m — swaths for the chosen day · loads below 500 km",
  },
  {
    id: "hls-l30",
    doc: "https://lpdaac.usgs.gov/products/hlsl30v002/",
    layer: "HLS_L30_Nadir_BRDF_Adjusted_Reflectance",
    title: "Landsat 8/9 true colour (HLS, 30 m)",
    ext: "png", tms: "31.25m", maxLevel: 11, fine: 500,
    start: "2013-03-22", timed: true, on: false,
    meta: "Landsat surface reflectance, 30 m — swaths for the chosen day · loads below 500 km",
  },
  {
    id: "sar-s1",
    doc: "https://www.jpl.nasa.gov/go/opera/products/rtc-product/",
    layer: "OPERA_L2_Radiometric_Terrain_Corrected_SAR_Sentinel-1",
    title: "Sentinel-1 radar backscatter (OPERA RTC, 30 m)",
    ext: "png", tms: "31.25m", maxLevel: 11, fine: 500,
    start: "2025-01-10", timed: true, on: false,
    meta: "C-band radar, sees through cloud and night — swaths for the chosen day · loads below 500 km",
  },
  {
    id: "nisar",
    doc: "https://nisar.jpl.nasa.gov/data/data-products/",
    layer: "NISAR_L2_Geocoded_Polarimetric_Covariance",
    title: "NISAR L-band radar backscatter (15 m)",
    ext: "png", tms: "15.625m", maxLevel: 12, fine: 300,
    start: "2025-10-29", timed: true, on: false,
    meta: "The finest layer here: NASA–ISRO L-band radar, provisional — swaths for the chosen day · loads below 300 km",
  },
  {
    id: "water-hls",
    // CLASSIFICATION raster (open water / partial water / snow-ice / cloud):
    // class codes neither average nor subtract, so no posture flags.
    classmap: "https://gibs.earthdata.nasa.gov/colormaps/v1.3/OPERA_Dynamic_Surface_Water_Extent.xml",
    legend: "https://gibs.earthdata.nasa.gov/legends/OPERA_Dynamic_Surface_Water_Extent_H.svg",
    doc: "https://www.jpl.nasa.gov/go/opera/products/dswx-product-suite/",
    layer: "OPERA_L3_Dynamic_Surface_Water_Extent-HLS",
    classNote: "partial = a 30 m pixel that is only part water (a bank, a marsh, a narrow channel) · cloud pixels are unobserved, not dry",
    title: "Surface water extent (OPERA DSWx from HLS, 30 m)",
    ext: "png", tms: "31.25m", maxLevel: 11, fine: 500,
    start: "2016-01-07", timed: true, on: false,
    meta: "Where there is open water on the chosen day, from optical imagery · 2018-08 → 2023-01 gap · loads below 500 km",
  },
  {
    id: "water-s1",
    classmap: "https://gibs.earthdata.nasa.gov/colormaps/v1.3/OPERA_Dynamic_Surface_Water_Extent_S1.xml",
    legend: "https://gibs.earthdata.nasa.gov/legends/OPERA_Dynamic_Surface_Water_Extent_S1_H.svg",
    doc: "https://www.jpl.nasa.gov/go/opera/products/dswx-product-suite/",
    layer: "OPERA_L3_Dynamic_Surface_Water_Extent-Sentinel-1",
    classNote: "inundated vegetation = flooded forest or crops the radar sees under the canopy · HAND-masked = too high above the nearest river to flood, not looked at",
    title: "Surface water extent (OPERA DSWx from Sentinel-1, 30 m)",
    ext: "png", tms: "31.25m", maxLevel: 11, fine: 500,
    start: "2023-12-15", timed: true, on: false,
    meta: "Open water and flooded vegetation from radar — works under cloud, so it is the flood layer · loads below 500 km",
  },
  {
    id: "elevation",
    // Continuous field in metres, but UNTIMED: terrain has no date to average
    // or difference over, so no posture flags. probeNative: read at 30 m, not
    // the usual z4 — a 4 km mean of an alpine pixel is not that pixel's height.
    colormap: "https://gibs.earthdata.nasa.gov/colormaps/v1.3/ASTER_GDEM_Color_Index.xml",
    legend: "https://gibs.earthdata.nasa.gov/legends/ASTER_GDEM_Color_Index_H.svg",
    doc: "https://lpdaac.usgs.gov/products/astgtmv003/",
    layer: "ASTER_GDEM_Color_Index",
    probeNative: true,
    datelessNote: "<strong>Elevation (ASTER GDEM, 30 m)</strong> is a fixed terrain model " +
      "(stereo imagery 2000–2013, v3), so the <strong>date selector doesn't change it</strong>.",
    title: "Elevation (ASTER GDEM, 30 m)",
    ext: "png", tms: "31.25m", maxLevel: 11,
    timed: false, on: false,
    meta: "Height above sea level from stereo imagery, 30 m, 83°N–83°S — hover reads metres",
  },
  {
    id: "builtup",
    classmap: "https://gibs.earthdata.nasa.gov/colormaps/v1.3/Landsat_Human_Built-up_And_Settlement_Extent.xml",
    legend: "https://gibs.earthdata.nasa.gov/legends/Landsat_Human_Built-up_And_Settlement_Extent_H.svg",
    doc: "https://www.earthdata.nasa.gov/data/catalog/sedac-ciesin-sedac-ulandsat-hbase-v1-1.0",
    layer: "Landsat_Human_Built-up_And_Settlement_Extent",
    classNote: "HBASE = buildings, paved surfaces and other human structures at 30 m, mapped once from 2010 Landsat",
    datelessNote: "<strong>Built-up extent (HBASE, 30 m)</strong> is one map from " +
      "<strong>2010</strong> Landsat imagery, so the <strong>date selector doesn't change it</strong>.",
    title: "Built-up extent (HBASE 2010, 30 m)",
    ext: "png", tms: "31.25m", maxLevel: 11,
    timed: false, on: false,
    meta: "Every building, road and paved surface the 2010 Landsat record could see, 30 m",
  },
  {
    id: "impervious",
    // Mixed palette: three classes (no data / not built / cloud) plus ten
    // percentage bins. Every bin carries a class-style legend entry, so the
    // classification path reads it whole — the probe answers "31 – 40" (a
    // bin label), which is the product's own precision.
    classmap: "https://gibs.earthdata.nasa.gov/colormaps/v1.3/Landsat_Global_Man-made_Impervious_Surface.xml",
    legend: "https://gibs.earthdata.nasa.gov/legends/Landsat_Global_Man-made_Impervious_Surface_H.svg",
    doc: "https://www.earthdata.nasa.gov/data/catalog/sedac-ciesin-sedac-ulandsat-gmis-v1-1.0",
    layer: "Landsat_Global_Man-made_Impervious_Surface",
    classNote: "percent of each 30 m pixel that is sealed (roofs, asphalt, concrete) — rain runs off it instead of soaking in",
    datelessNote: "<strong>Impervious surface (GMIS, 30 m)</strong> is one map from " +
      "<strong>2010</strong> Landsat imagery, so the <strong>date selector doesn't change it</strong>.",
    title: "Impervious surface % (GMIS 2010, 30 m)",
    ext: "png", tms: "31.25m", maxLevel: 11,
    timed: false, on: false,
    meta: "How sealed the ground is, 30 m, 2010 — the urban-heat and flash-flood map",
  },
  {
    id: "weld",
    doc: "https://lpdaac.usgs.gov/products/glweldv003/",
    layer: "Landsat_WELD_CorrectedReflectance_TrueColor_Global_Annual",
    // Annual composites anchored on DECEMBER 1 (1998-12-01 = Dec 1998 → Nov
    // 1999). `annual` snaps to Jan 1 and the measured domain then floors that
    // to the newest Dec 1 at or before it — i.e. asking for 1999 lands on
    // 1998-12-01, which IS the composite that covers 1999. Three separate
    // spans only (1984–86, 1989–91, 1999–2001); the domain snapping fills the
    // holes with the nearest earlier composite and the toast says so.
    annual: true, annualAnchor: "12-01",
    endTime: "2000-12-01",
    title: "Landsat true colour, historic (WELD annual, 30 m)",
    ext: "jpg", tms: "31.25m", maxLevel: 11,
    start: "1983-12-01", timed: true, on: false,
    meta: "Cloud-free yearly mosaics of 1980s–90s Landsat, 30 m — the date's YEAR picks it (1984–86, 1989–91, 1999–2001)",
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
    // The blank areas are the PRODUCT's own mask, not a bug: an L-band
    // radiometer can't retrieve salinity near coasts (land in the sidelobes),
    // under sea ice, or in heavy radio-frequency interference — which blanks
    // much of the North Sea, Baltic approaches and Mediterranean shores. And
    // the scale's catch-all bottom bin [0,30) is why brackish seas used to
    // probe as a flat "15": see the caps note on getValueLut.
    meta: "SMAP L-band salinity (PSU) — same quantity as SMOS/CATDS · monthly composite; 2024 has a mission data gap · masked near coasts, sea ice and radio interference; scale caps at <30 / ≥40",
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
    // Measured from the layer's own GIBS time domain, not read off the product
    // page: the last served epoch is 2019-01-22. It was typed here as
    // 2019-01-17 — one 5-day step early, and a date that quietly hid the final
    // frame of the archive. loadGibsDomain() re-measures this at runtime, so if
    // GIBS ever extends or trims the record the app follows without an edit.
    endTime: "2019-01-22",
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
    grid: true, gridFile: "data/currents.json", monthlyGrid: true,
    ramp: "speed", vmin: 0, vmax: 1.5, units: "m/s", maxLevel: 6,
    doc: "https://data.marine.copernicus.eu/product/GLOBAL_MULTIYEAR_PHY_001_030/description",
    title: "Surface current speed (GLORYS)",
    meta: "Monthly-mean ocean current speed — the date's month picks the map",
    on: false,
  },
  {
    id: "mld",
    grid: true, gridFile: "data/mld.json", monthlyGrid: true,
    ramp: "precip", vmin: 0, vmax: 500, units: "m", maxLevel: 6,
    doc: "https://data.marine.copernicus.eu/product/GLOBAL_MULTIYEAR_PHY_001_030/description",
    title: "Mixed-layer depth (GLORYS)",
    meta: "How deep the surface ocean is stirred — deep winter mixing is where AMOC water forms",
    on: false,
  },
  {
    id: "amoc-eval",
    // OUR OWN artefact, and the only layer on the globe that describes a model
    // run rather than the world. It is written BY the evaluator
    // (`python3 ml/rollout_spatial.py --export-mask-only`, which calls the same
    // corridor_pixels() the scoring calls) and never drawn here: a corridor
    // hand-traced in the frontend would be a second definition of the
    // experiment, and the second definition is the one that goes stale.
    // CATEGORICAL — a cell carries a ROLE, not a quantity — so classGrid and
    // the file's own palette, and therefore neither aggregable nor
    // deltaRange. Untimed: the geometry comes from the tensor's window and the
    // corridor recipe, not from a date.
    // NO catalog record, deliberately: §2.6 catalogues open DATASETS, and this
    // is a description of our own experiment — the same reason the city labels
    // have none. Its `doc` points at the experiment's plan instead.
    grid: true, classGrid: true, gridFile: "data/amoc_eval_mask.json",
    classNote: "the roles are NESTED — the section lies inside the corridor, which lies " +
      "inside the rolled window &mdash; and each cell shows its most specific one",
    datelessNote: "<strong>AMOC forecast: pixels rolled forward</strong> is the fixed " +
      "geometry of an experiment — the tensor window and the corridor recipe — so the " +
      "<strong>date selector doesn't change it</strong>. What changes with time is the " +
      "forecast itself, which lives in the <strong>AMOC tab</strong>.",
    maxLevel: 7,
    doc: "https://github.com/blauewelt/earth/blob/main/ml/plans/E022_spatial_coupling.md",
    title: "AMOC forecast: pixels rolled forward",
    meta: "Where the model actually computes — and which of those pixels its AMOC score is read from",
    on: false,
  },
  {
    id: "tides",
    grid: true, gridFile: "data/tides.json",
    // Not a climatology and not a snapshot: a harmonic ANALYSIS (fixed
    // constants fit to 1992–2019 altimetry). Neither aggregable nor
    // delta-able — the range is already the cycle's summary, and there is no
    // time axis to average or difference. The moving physics lives in the
    // Tides tab, which reconstructs h(t) from the same constituents.
    ramp: "speed", vmin: 0, vmax: 8, units: "m", maxLevel: 6,
    doc: "https://www.seanoe.org/data/00683/79489/",
    title: "Tidal range (EOT20)",
    meta: "How far the sea rises and falls when the main constituents align — the moving tide is animated in the Tides tab",
    on: false,
  },
  {
    id: "oisst-monthly",
    grid: true, gridFile: "data/oisst_monthly.json", monthlyGrid: true,
    // Exists because MUR's GIBS palette saturates at 32 degC (tiles arrive
    // painted). This is SST from the NUMBERS, scale to 36: the Persian Gulf
    // separates from the merely-warm tropics, and the probe reads exact
    // values with no caps. Monthly-grid posture like GLORYS: neither
    // aggregable nor delta-able through the raster machinery.
    ramp: "sst", vmin: -2, vmax: 36, units: "°C", maxLevel: 6,
    doc: "https://psl.noaa.gov/data/gridded/data.noaa.oisst.v2.highres.html",
    title: "Sea surface temperature (OISST monthly, to 36°)",
    meta: "Monthly-mean SST from the numbers, 1981-09 → now — the scale reaches 36 °C, so the hottest seas actually show",
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
    // The globe draws surface currents and MLD from ONE reanalysis. These two
    // draw what three of them cannot agree on. GREP estimates the same
    // 1993-2024 ocean three times — CGLORS (CMCC), GLORYS2V4 (Mercator),
    // ORAS5 (ECMWF) — from the same satellites and the same Argo floats, with
    // different models and different assimilation. The spread is therefore not
    // noise: it is the part of the ocean the observations do not pin down, and
    // no forecast of a cell can be more certain than its own inputs are.
    // Native 0.25 deg on the family-3/4 NA grid (the amoc-eval geometry), so
    // nothing here is regridded.
    id: "grep-spread-cur",
    grid: true, gridFile: "data/grep_spread_cur.json", snapshotGrid: true,
    ramp: "speed", vmin: 0, vmax: 0.5, units: "m/s", maxLevel: 7,
    doc: "https://data.marine.copernicus.eu/product/GLOBAL_MULTIYEAR_PHY_ENS_001_031/description",
    title: "Reanalysis disagreement: current speed",
    meta: "How far apart three reanalyses of the SAME years are — median 0.077 m/s, over 1 m/s in the Gulf Stream",
    on: false,
  },
  {
    id: "grep-spread-mld",
    grid: true, gridFile: "data/grep_spread_mld.json", snapshotGrid: true,
    ramp: "speed", vmin: 0, vmax: 200, units: "m", maxLevel: 7,
    doc: "https://data.marine.copernicus.eu/product/GLOBAL_MULTIYEAR_PHY_ENS_001_031/description",
    title: "Reanalysis disagreement: mixed-layer depth",
    meta: "Median 14 m, but 716 m at the subpolar convection sites — where the AMOC is made is where we know least",
    on: false,
  },
  {
    id: "drivers",
    // CATEGORICAL grid: the cell carries a driver CLASS, not a number, so it
    // paints from the file's own `classes` palette (WRI's, so the globe matches
    // every published figure) rather than a ramp — and it takes no aggregation
    // or difference posture, for the same reason the OPERA rasters don't.
    // Untimed on purpose: one 2001–2025 attribution, not a per-date field.
    grid: true, classGrid: true, gridFile: "data/drivers.json",
    classNote: "the driver that dominates each cell's loss over 2001&ndash;2025 &mdash; " +
      "blank means no mapped forest loss, not no forest",
    maxLevel: 7,
    doc: "https://datasets.wri.org/datasets/dominant-drivers-of-tree-cover-loss-at-1km",
    title: "Drivers of forest loss (WRI/DeepMind)",
    meta: "WHY forest was lost, 2001–2025 — the companion to the 30 m alerts, which only see THAT it was",
    on: false,
  },
  {
    id: "gfs-temp",
    grid: true, gridFile: "data/gfs_temp.json", monthlyGrid: true, forecastGrid: true,
    ramp: "t2m", vmin: -30, vmax: 40, units: "°C", maxLevel: 6,
    doc: "https://www.emc.ncep.noaa.gov/emc/pages/numerical_forecast_systems/gfs.php",
    title: "Temperature forecast (GFS, 2 m)",
    meta: "The next 10 days — with this on, the date selector runs into the future",
    on: false,
  },
  {
    id: "gfs-precip",
    grid: true, gridFile: "data/gfs_precip.json", monthlyGrid: true, forecastGrid: true,
    ramp: "rain", vmin: 0, vmax: 50, units: "mm/day", maxLevel: 6,
    doc: "https://www.emc.ncep.noaa.gov/emc/pages/numerical_forecast_systems/gfs.php",
    title: "Precipitation forecast (GFS, daily)",
    meta: "Forecast 24-h rain totals per day; dry (<0.5 mm) is transparent",
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

/* ------------------------------------------------ what GIBS actually serves */
/* Every timed layer used to be asked for "two days ago" (or the first of the
 * current month), on the assumption that an archive runs continuously up to
 * about now. Neither half of that assumption holds.
 *
 *  - Products lag by their own amounts. On 2026-08-03 the MODIS monthly NDVI
 *    composite's newest tile date was 2026-06-01 — 62 days behind the request.
 *  - Every archive has interior HOLES, not just a trailing edge: NDVI is
 *    missing 2025-04, SMAP salinity all of 2024, VIIRS 11–15 July 2026, and
 *    GRACE is irregular throughout.
 *
 * Ask for a date that isn't served and GIBS 404s, Cesium draws nothing, the
 * hover probe says "no data", and NOTHING on screen explains why. That is
 * exactly what a user saw: the vegetation layer on, the legend showing, the
 * globe grey, no green anywhere over Africa.
 *
 * So stop guessing. GIBS publishes each layer's exact time domain at
 *   /wmts/epsg4326/best/1.0.0/{layer}/default/{tms}/all/all.xml
 * as a comma-separated list of ISO-8601 start/end/period intervals. We fetch it
 * the first time a layer is switched on (small; cached for the session), and
 * snap every requested date DOWN to the newest instant actually served. Because
 * the answer is MEASURED it also replaces the hand-typed endTime constants,
 * which had already drifted: sea-surface-height's said 2019-01-17, the archive
 * says 2019-01-22.
 */

const GIBS_DOMAIN_URL =
  "https://gibs.earthdata.nasa.gov/wmts/epsg4326/best/1.0.0/" +
  "{layer}/default/{tms}/all/all.xml";

function gibsDomainUrl(cfg) {
  return GIBS_DOMAIN_URL.replace("{layer}", cfg.layer).replace("{tms}", cfg.tms);
}

// Domain instants come in two granularities — bare dates ("2026-06-01") for
// daily/monthly products, full timestamps for the sub-daily ones. Parse both,
// and treat a bare date as UTC midnight (Date.parse would too, but only by
// spec accident; being explicit stops a local-time reading from creeping in).
function domainMs(v) {
  return Date.parse(String(v).length === 10 ? `${v}T00:00:00Z` : v);
}

// Print an instant back in the SAME granularity the archive used, so the TIME
// value we send matches the one GIBS advertises character for character.
function domainFormat(sample, ms) {
  const iso = new Date(ms).toISOString();
  return String(sample).length === 10 ? iso.slice(0, 10) : iso.replace(/\.\d{3}Z$/, "Z");
}

/* ISO-8601 durations as GIBS uses them: P1D, P5D, P1M, P1Y, PT30M — plus, on
 * GRACE, a different irregular period for nearly every interval (P28D, P17D,
 * P13D, P33D…). Note M means months before the T and minutes after it; getting
 * that backwards would silently mis-step a whole archive. */
function parsePeriod(p) {
  const m = /^P(?:(\d+)Y)?(?:(\d+)M)?(?:(\d+)D)?(?:T(?:(\d+)H)?(?:(\d+)M)?(?:(\d+)S)?)?$/
    .exec(String(p || "").trim());
  if (!m) return { months: 0, ms: 864e5 };
  const n = (i) => (m[i] == null ? 0 : Number(m[i]));
  const months = n(1) * 12 + n(2);
  const ms = ((n(3) * 24 + n(4)) * 60 + n(5)) * 60000 + n(6) * 1000;
  if (!months && !ms) return { months: 0, ms: 864e5 };   // "P0D" and friends
  return { months, ms };
}

function parseGibsDomain(xml) {
  const m = /<Domain>([^<]*)<\/Domain>/.exec(String(xml || ""));
  if (!m) return null;
  const out = [];
  for (const chunk of m[1].split(",")) {
    const parts = chunk.trim().split("/");
    const s = parts[0];
    if (!s || !Number.isFinite(domainMs(s))) continue;
    let e = parts.length > 1 && parts[1] ? parts[1] : s;
    // Some upstream intervals are malformed — GRACE serves
    // "2020-01-20/2020-01-10/P1M", whose end precedes its start. Collapse those
    // (and genuine single instants) to one served instant rather than throwing
    // away a whole layer's domain over one bad row.
    if (!Number.isFinite(domainMs(e)) || domainMs(e) < domainMs(s)) e = s;
    out.push({ s, e, ...parsePeriod(parts[2]) });
  }
  if (!out.length) return null;
  out.sort((a, b) => domainMs(a.s) - domainMs(b.s));
  return out;
}

// Calendar-correct month arithmetic in UTC, clamping the day so a start on the
// 31st can't spill into the following month.
function addMonthsUTC(ms, n) {
  const d = new Date(ms);
  const day = d.getUTCDate();
  const out = new Date(Date.UTC(d.getUTCFullYear(), d.getUTCMonth() + n, 1,
    d.getUTCHours(), d.getUTCMinutes(), d.getUTCSeconds()));
  const inMonth = new Date(Date.UTC(out.getUTCFullYear(), out.getUTCMonth() + 1, 0)).getUTCDate();
  out.setUTCDate(Math.min(day, inMonth));
  return out.getTime();
}

// Newest instant this one interval serves at or before reqMs, or null if the
// interval starts after it.
function domainFloor(iv, reqMs) {
  const S = domainMs(iv.s), E = domainMs(iv.e);
  if (reqMs < S) return null;
  if (reqMs >= E) return iv.e;
  if (iv.months) {
    const s = new Date(S), r = new Date(reqMs);
    let steps = Math.floor(
      ((r.getUTCFullYear() - s.getUTCFullYear()) * 12 +
        (r.getUTCMonth() - s.getUTCMonth())) / iv.months);
    // The month difference above ignores the day, so a start of 2015-04-12 and
    // a request of 2015-05-01 reads as a whole step when it isn't one. Walk
    // back until the answer is genuinely at or before the request.
    let out = addMonthsUTC(S, steps * iv.months);
    while (steps > 0 && out > reqMs) out = addMonthsUTC(S, --steps * iv.months);
    return domainFormat(iv.s, Math.min(out, E));
  }
  const steps = Math.floor((reqMs - S) / iv.ms);
  return domainFormat(iv.s, Math.min(S + steps * iv.ms, E));
}

/* The newest instant the whole domain serves at or before `want`. Intervals can
 * overlap and are not guaranteed disjoint, so take the max over all of them
 * rather than trusting the first hit. */
function snapToDomain(intervals, want) {
  if (!intervals?.length) return null;
  const reqMs = domainMs(want);
  if (!Number.isFinite(reqMs)) return null;
  let best = null, bestMs = -Infinity;
  for (const iv of intervals) {
    const c = domainFloor(iv, reqMs);
    if (c == null) continue;
    const ms = domainMs(c);
    if (ms > bestMs) { bestMs = ms; best = c; }
  }
  // Before the archive begins there is nothing at or below the request. Show
  // the earliest instant served instead of a blank globe.
  return best ?? intervals[0].s;
}

const gibsDomains = new Map();          // layer id → intervals, or null = asked and got nothing
const gibsDomainPending = new Map();    // layer id → in-flight promise

function loadGibsDomain(cfg) {
  if (!cfg?.timed || !cfg.layer) return Promise.resolve(null);
  if (gibsDomains.has(cfg.id)) return Promise.resolve(gibsDomains.get(cfg.id));
  if (gibsDomainPending.has(cfg.id)) return gibsDomainPending.get(cfg.id);
  const p = fetch(gibsDomainUrl(cfg))
    .then((r) => (r.ok ? r.text() : Promise.reject(new Error(`HTTP ${r.status}`))))
    .then(parseGibsDomain)
    .catch(() => null)
    .then((v) => {
      // Cache failures too. If GIBS won't tell us, re-asking on every toggle
      // just spends the user's connection to hear the same silence; the layer
      // falls back to the typed endTime and behaves exactly as it did before.
      gibsDomains.set(cfg.id, v);
      gibsDomainPending.delete(cfg.id);
      if (v) {
        cfg.lastServed = v[v.length - 1].e;
        // A typed endTime means "this archive is closed". Keep that meaning,
        // but take the DATE from the measurement — the hand-typed ones drift.
        if (cfg.endTime) cfg.endTime = cfg.lastServed.slice(0, 10);
      }
      return v;
    });
  gibsDomainPending.set(cfg.id, p);
  return p;
}

/* Called when a timed layer is added. The globe must not wait on a metadata
 * fetch to paint, so the first tiles may go out at the un-snapped date and 404;
 * when the domain lands we rebuild the layer at a date that exists. That costs
 * one wasted round of tiles on the very first enable of a lagging layer, and
 * buys a globe that never sits blank with no explanation. */
function ensureGibsDomain(cfg) {
  if (!cfg?.timed || !cfg.layer || gibsDomains.has(cfg.id)) return;
  const before = gibsTime(cfg, state.date);
  loadGibsDomain(cfg).then((dom) => {
    if (!dom) return;
    const e = state.layers[cfg.id];
    if (!e || !(e.layer || e.suppressed)) return;      // switched off while we asked
    if (gibsTime(cfg, state.date) !== before) {
      removeLayer(cfg.id);
      addLayer(cfg);
    }
    maybeArchiveToast(cfg, { replace: true });
    maybeAnnualToast(cfg, { replace: true });
  });
}

// GIBS TIME value for a layer: monthly products must be requested at the first
// of the month (a mid-month date returns a blank tile), sub-daily/daily use the
// raw date, and untimed layers use their fixed snapshot. The current month's
// composite is still accumulating and not yet published (GIBS 404s → an
// invisible layer), so a date in the current month falls back to the previous
// complete month.
//
// This is the shape of the request BEFORE the archive gets a say. It is what we
// would ask for if every product were continuous and current; gibsTime() below
// then snaps it onto a date GIBS actually serves. Keep them separate: the toasts
// need to name both — what you asked for, and what you are looking at.
//
// `clampEnd: false` skips the archive-end clamp, which is how the toasts get at
// the date the user genuinely asked for. With it left on (the default, and what
// every tile URL uses) "asked" would already have been quietly moved, and a
// clamped layer would look like it had got exactly what it wanted.
function gibsTimeStatic(cfg, dateStr, { clampEnd = true } = {}) {
  if (!cfg.timed) return cfg.fixedTime || "default";
  // Forecast layers let the date run into the future; observation archives
  // can't follow. Clamp so GIBS never gets asked for tomorrow's tiles.
  if (dateStr > defaultDate()) dateStr = defaultDate();
  // Some archives stop being served as tiles before today (GRACE 2022-07,
  // CERES 2018-10, SSH anomalies 2019-01, AMSR2 soil moisture 2025-09): any
  // later date clamps to the last served one, so the layer shows its final
  // state instead of silently blanking. The hover card states the end date.
  // Annual products (OPERA DIST-ANN, Landsat WELD) are served at one date per
  // year: snap to the date's year, floored at the first year served and — the
  // archive-end clamp, applied to YEARS rather than dates, so that a
  // December-anchored product's last composite is still reachable — capped at
  // the last.
  if (cfg.annual) {
    let y = Number(dateStr.slice(0, 4));
    if (cfg.start && y < annualYearOf(cfg, cfg.start)) y = annualYearOf(cfg, cfg.start);
    if (clampEnd && cfg.endTime && y > annualYearOf(cfg, cfg.endTime)) y = annualYearOf(cfg, cfg.endTime);
    // A December-anchored product (Landsat WELD: 1998-12-01 = Dec 1998 → Nov
    // 1999) is asked for at the anchor of the PREVIOUS calendar year, which is
    // the composite that covers the year the user named.
    return cfg.annualAnchor ? `${y - 1}-${cfg.annualAnchor}` : `${y}-01-01`;
  }
  if (clampEnd && cfg.endTime && dateStr > cfg.endTime) dateStr = cfg.endTime;
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

/* The single source of truth for the TIME value in every tile URL and every
 * provenance stamp. Stays SYNCHRONOUS on purpose: it is called from inside
 * Cesium's tile requests and from the hover probe, and it must never return a
 * promise. Until the layer's domain has arrived (one small fetch, once per
 * session) this behaves exactly as it always did; afterwards every call site —
 * tiles, the pixel card, the hover stamp, the comparison hints — is corrected
 * at once, and they can never disagree about what a click just read. */
/* The calendar year an annual product's tile date stands for. Jan-1 products
 * name their own year; a December-anchored one (`annualAnchor: "12-01"`) names
 * the year that FOLLOWS its anchor. Every read-out — the provenance stamp, the
 * annual toast, the availability span — goes through this so they agree. */
function annualYearOf(cfg, t) {
  const y = Number(String(t).slice(0, 4));
  return cfg.annualAnchor ? y + 1 : y;
}

function gibsTime(cfg, dateStr) {
  const want = gibsTimeStatic(cfg, dateStr);
  if (!cfg.timed) return want;
  const dom = gibsDomains.get(cfg.id);
  if (!dom) return want;
  return snapToDomain(dom, want) ?? want;
}

/* ------------------------------------------------------ when a value is from */
/* Every number this app prints was observed at some time, and the honest
 * GRANULARITY of that time differs per dataset: MUR SST is a day, IMERG's
 * 30-min layer a half hour, NDVI a month, DIST-ANN a whole year, GLORYS a
 * monthly mean, and the 1991-2020 normals a fixed period that has no single
 * "when" at all. Both read-outs — the pixel card and the hover probe — build
 * their stamps here, so they can never disagree about what a click just read.
 *
 * This exists to close a real lie. The card used to head its entire satellite
 * section with state.date, but gibsTime() clamps and snaps PER LAYER: under a
 * "2026-08-03" heading, the GRACE row was really 2022-07 (tiles end there),
 * CERES 2018-10, sea ice 2025-09 — four to eight years stale, labelled fresh.
 * A stamp per row is the only arrangement in which that cannot happen.
 *
 * `kind` names the granularity, never the dataset:
 *   instant · halfhour · day · month · year · period (a fixed span: 1991-2020)
 */
function whenAt(kind, t) { return t ? { kind, t } : null; }

// The time a GIBS layer's tiles were actually REQUESTED for. gibsTime is the
// single source of truth, so every clamp (GRACE, CERES, SSH, soil moisture,
// sea ice) and every snap (monthly, annual, 5-day, sub-daily) lands in the
// stamp for free — and stays right if the layer's config later changes.
function whenOfGibs(cfg, dateStr = state.date) {
  const t = gibsTime(cfg, dateStr);
  if (!t || t === "default") return null;      // untimed and undated: say nothing
  if (cfg.annual) return whenAt("year", String(annualYearOf(cfg, t)));
  if (cfg.monthly) return whenAt("month", t.slice(0, 7));
  if (t.includes("T")) return whenAt("halfhour", t.slice(0, 16));
  return whenAt("day", t);
}

/* Baked grids state their own time IN THE FILE: `period` for a climatology
 * (1991-2020 normals, the 2001-2025 driver map), `month` for a single-month
 * snapshot (the Argo 300 m anomaly), the resolved key for a month- or
 * day-keyed archive (GLORYS, the GFS forecast). A grid with none of those gets
 * NO stamp on purpose: `snapshot` is the date we fetched the numbers, not a
 * date the world was in that state, and printing it would be the same class of
 * lie this whole helper exists to remove. */
function whenOfGrid(cfg, g) {
  if (!g) return null;
  if (g.period) return whenAt("period", g.period);
  if (cfg.monthlyGrid) return whenAt(g.keyLen === 10 ? "day" : "month", resolveGridMonth(g));
  if (g.month) return whenAt("month", g.month);
  return null;
}

function whenOfLayer(cfg, g) {
  return cfg?.grid ? whenOfGrid(cfg, g) : cfg ? whenOfGibs(cfg) : null;
}

/* Age units: hours, days, months, years. Pick the coarsest that reads at least
 * 2 — so 47 h stays "47 hours old" and 61 days becomes "2 months old" — but
 * never finer than the stamp's own granularity, because a value dated to a
 * month cannot honestly be "36 days old" and a daily composite is not "5 hours
 * old". Forecast frames run the other way and read "in 3 days". */
const AGE_UNITS = [
  { one: "hour", many: "hours", ms: 36e5, zero: "just now" },
  { one: "day", many: "days", ms: 864e5, zero: "today" },
  { one: "month", many: "months", ms: 30.44 * 864e5, zero: "this month" },
  { one: "year", many: "years", ms: 365.25 * 864e5, zero: "this year" },
];
const AGE_FLOOR = { instant: 0, halfhour: 0, day: 1, month: 2, year: 3 };

// The instant a stamp NAMES: the start of the span it covers, so a monthly
// mean ages from the 1st and an annual summary from January.
function whenMs(w) {
  if (!w) return NaN;
  const t = w.t;
  if (w.kind === "year") return Date.parse(`${t}-01-01T00:00:00Z`);
  if (w.kind === "month") return Date.parse(`${t}-01T00:00:00Z`);
  if (w.kind === "day") return Date.parse(`${t}T00:00:00Z`);
  return Date.parse(/[Zz]$|[+-]\d\d:?\d\d$/.test(t) ? t : `${t}Z`);
}

function whenAge(w, now = Date.now()) {
  // A fixed period has no age. "The 1991-2020 normal is 6 years old" is not a
  // fact about the normal; it stays what it is until a new normal is published.
  if (!w || w.kind === "period") return null;
  const ms = now - whenMs(w);
  if (!Number.isFinite(ms)) return null;
  let i = AGE_FLOOR[w.kind] ?? 0;
  while (i < AGE_UNITS.length - 1 && Math.abs(ms) / AGE_UNITS[i + 1].ms >= 2) i++;
  const u = AGE_UNITS[i];
  // A stamp names a period's START, so count the whole unit boundaries between
  // it and now: FLOOR into the past, CEIL into the future. Rounding instead
  // calls a 2026-08-01 reading "3 days old" on the evening of 08-03 and calls
  // last year's 2025 map "2 years old"; flooring a FORECAST frame is worse
  // still — tomorrow's frame, five hours out, would read "today".
  const n = ms < 0
    ? Math.ceil(-ms / u.ms)
    : Math.floor(ms / u.ms);
  if (!n) return u.zero;
  const unit = n === 1 ? u.one : u.many;
  return ms < 0 ? `in ${n} ${unit}` : `${n} ${unit} old`;
}

function whenText(w) {
  if (!w) return "";
  if (w.kind === "period") return w.t.replace("-", "–");     // 1991-2020 → en dash
  if (w.kind === "instant" || w.kind === "halfhour") {
    return `${w.t.replace("T", " ").replace(/[Zz]$/, "")} UTC`;
  }
  return w.t;
}

// The dim right-hand stamp every read-out row carries. No `when` → no stamp:
// an elevation or a station's distance is not an observation with a date, and
// inventing one would be worse than leaving the column empty.
/* The one string both read-outs print. The card wraps it in its own element;
 * the probe drops it into a line of dim meta text — but neither builds it, so a
 * click and a hover on the same pixel can never disagree about the date. */
function whenLabel(w) {
  if (!w) return "";
  const age = whenAge(w);
  return `${whenText(w)}${age ? ` · ${age}` : ""}`;
}

function whenStamp(w) {
  return w ? `<span class="px-when">${whenLabel(w)}</span>` : "";
}

/* The ONE place a GIBS tile URL is built — the template Cesium wants, with
 * {TileMatrix}/{TileRow}/{TileCol} left in for it to fill.
 *
 * A companion `gibsTileUrl(cfg, date, level, row, col)` used to live here,
 * filling those three in so playback could fetch a specific tile itself. It
 * has gone with the prefetch that needed it: GIBS answers `no-store`, nothing
 * we fetch by hand can be cached, and the only prefetch that works is a Cesium
 * layer at alpha 0 building its own URLs through this template. See the
 * preload ring. */
function gibsUrlTemplate(cfg, dateStr) {
  return GIBS_URL
    .replace("{layer}", cfg.layer)
    .replace("{time}", gibsTime(cfg, dateStr))
    .replace("{tms}", cfg.tms)
    .replace("{ext}", cfg.ext);
}

function gibsProvider(cfg, dateStr) {
  const url = gibsUrlTemplate(cfg, dateStr);
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

// The base Blue-Marble layer. Default mode is AUTO: the base desaturates
// whenever a colormapped data layer (raster, grid, ensemble, delta/ratio) is
// on — that's exactly when the map's own blues and greens fight the data
// colours (blue-on-blue SST, green-on-green NDVI) — and returns to full
// colour when the globe is bare or showing photographs (true colour, night
// lights) or plain point markers. "Always colour"/"Always grayscale"
// override; the choice persists.
const baseImageryLayer = viewer.imageryLayers.get(0);
function colormappedLayerActive() {
  if (sstEnsembleLayer) return true;
  return Object.values(state.layers).some((e) =>
    e.layer && (e.cfg.colormap || e.cfg.classmap || e.cfg.grid));
}
function updateBaseAppearance() {
  const mode = document.getElementById("base-mode")?.value || "auto";
  const gray = mode === "gray" || (mode === "auto" && colormappedLayerActive());
  baseImageryLayer.saturation = gray ? 0.0 : 1.0;
  baseImageryLayer.brightness = gray ? 0.6 : 1.0;
}

/* ------------------------------------------------------- place-name overlay
 *
 * A globe of pure data is beautiful and unnavigable: an SST anomaly off a
 * coastline you can't name tells you nothing about WHERE the ocean is warm.
 * Two things fix that, and they come from different places: city NAMES are
 * baked into data/cities.json (see below for why they can't come from GIBS),
 * and political/coastal LINEWORK is a GIBS raster overlay. Neither adds a
 * browser-facing host, so both stay inside the network rule in CLAUDE.md §3.
 *
 * These are NOT layers in the layer list. That list is a catalogue of
 * measurements: every entry has a legend, a date, a hover card and a record in
 * data/catalog.json. A basemap annotation has none of those and would have to
 * fake them all. It belongs with "Base globe" — the controls for how the map
 * LOOKS — and so it lives here, next to the desaturation logic, for the same
 * reason.
 */
/* Borders and coastlines come from GIBS as a raster overlay. The NAMES do not:
 * GIBS's Reference_Labels layer returns blank PNGs over this whole tile matrix
 * (Worldview draws those from a vector source, which Cesium would need an MVT
 * decoder to read), so city names are baked into data/cities.json instead —
 * the app's normal posture anyway, and it buys per-label control the raster
 * could never give. */
function bordersProvider() {
  return new Cesium.WebMapTileServiceImageryProvider({
    url: GIBS_URL.replace("{layer}", "Reference_Features_15m").replace("{time}", "default")
      .replace("{tms}", "15.625m").replace("{ext}", "png"),
    layer: "Reference_Features_15m", style: "default", format: "image/png",
    tileMatrixSetID: "15.625m", maximumLevel: 12,
    tileWidth: 512, tileHeight: 512,
    tilingScheme: new GIBSGeographicTilingScheme(),
    credit: new Cesium.Credit("NASA GIBS / Worldview"),
  });
}

/* Which places are on screen is Natural Earth's judgement, not mine: every
 * place carries `z`, the web-map zoom at which cartographers decided it earns a
 * label, and that becomes the far end of a per-label DistanceDisplayCondition.
 * Cesium then culls on the GPU with no per-frame JS, and the density stays
 * honest at every altitude — eleven world cities from orbit, every town in the
 * valley up close. FAR0 is set so Natural Earth's most important tier (z 1.7)
 * survives the full-globe view. */
const PLACE_FAR0 = 8.1e7;
/* Place names are SCENERY, not data. A click that lands on the word "Paris"
 * must still reach the globe underneath, or the pixel inspector would go quiet
 * in exactly the places the map is most legible — and a label glyph is a far
 * bigger pick target than it looks. Every city primitive carries this sentinel
 * id, and `seeThrough` makes the pick handlers look past it. */
const CITY_PICK = Object.freeze({ scenery: true });
const seeThrough = (p) => (p && p.id === CITY_PICK ? undefined : p);
let cityLabels = null, cityPoints = null, cityLoad = null;
let cityData = null, cityBuilt = 0, cityBuild = null;

/* One place. A dot as well as a name, because a word floating over nothing is
 * ambiguous by a few kilometres at these altitudes — kept small and neutral so
 * it never reads as one of the app's own data markers.
 *
 * Takes its collections as arguments because there are three sets of these now
 * and they must be visually identical: the Natural Earth labels, the deeper
 * GeoNames ones that fill in below them, and the single highlighted result of
 * a search. A place is a place; only the ladder rung and the colour differ. */
function addPlace(labels, points, c, far, tint) {
  const pos = Cesium.Cartesian3.fromDegrees(c.o, c.a);
  const ddc = far === Infinity
    ? undefined : new Cesium.DistanceDisplayCondition(0, far);
  points.add({
    id: CITY_PICK,
    position: pos, pixelSize: tint ? 7 : (c.cap ? 4 : 3),
    color: tint || Cesium.Color.WHITE.withAlpha(0.85),
    outlineColor: Cesium.Color.BLACK.withAlpha(tint ? 0.9 : 0.6),
    outlineWidth: tint ? 2 : 1,
    distanceDisplayCondition: ddc,
  });
  labels.add({
    id: CITY_PICK,
    position: pos, text: c.n,
    font: `${tint || c.cap ? "600 " : ""}${tint ? 13 : 11}px system-ui, sans-serif`,
    fillColor: tint || Cesium.Color.WHITE,
    outlineColor: Cesium.Color.BLACK, outlineWidth: 3,
    style: Cesium.LabelStyle.FILL_AND_OUTLINE,
    pixelOffset: new Cesium.Cartesian2(0, tint ? -14 : -11),
    // The outline is what makes a name readable over both a black ocean and a
    // yellow SST field without a background box eating the data.
    distanceDisplayCondition: ddc,
  });
}

function addCity(c) {
  addPlace(cityLabels, cityPoints, c, PLACE_FAR0 / Math.pow(2, c.z));
}

/* The rung the camera is currently standing on: invert the display condition
 * (a place with rung z shows while distance ≤ FAR0 / 2^z) to get the deepest
 * rung that could possibly be on screen. */
function cityRungAt(height) {
  return Math.log2(PLACE_FAR0 / Math.max(height, 1));
}

/* Materialise places down to a rung — and no further.
 *
 * Creating all 7,342 at once costs a 1.5-SECOND frame: Cesium rasterises every
 * glyph of every label into a texture atlas the first time it draws, so the
 * price is paid whether or not a single one of them is on screen. That stall
 * lands squarely on first paint, and it buys nothing — from orbit you can read
 * about sixty names.
 *
 * So build a PREFIX. data/cities.json is sorted by rung, which makes "every
 * place that could be visible right now" a contiguous slice from the front, and
 * makes this a `while` over a cursor rather than a spatial index. Chunked
 * across animation frames so even a deep tier can't produce one long frame.
 * Resolves when the prefix is complete, so tests (and anything else that needs
 * to know) can wait for it. */
function buildCitiesTo(zMax) {
  if (!cityData) return Promise.resolve(0);
  const wanted = () => cityBuilt < cityData.places.length && cityData.places[cityBuilt].z <= zMax;
  if (!wanted()) return cityBuild || Promise.resolve(cityBuilt);
  // A build already in flight is for a shallower rung; chain onto it rather
  // than interleaving two cursors over the same array.
  cityBuild = (cityBuild || Promise.resolve()).then(() => new Promise((done) => {
    const step = () => {
      const stop = cityBuilt + 300;
      while (cityBuilt < stop && wanted()) addCity(cityData.places[cityBuilt++]);
      if (wanted()) requestAnimationFrame(step);
      else { cityBuild = null; done(cityBuilt); }
    };
    step();   // first chunk synchronously: the top rung appears on this frame
  }));
  return cityBuild;
}

function ensureCities() {
  if (cityLoad) return cityLoad;
  cityLoad = fetch("data/cities.json").then((r) => r.json()).then(async (d) => {
    cityData = d;
    cityLabels = viewer.scene.primitives.add(new Cesium.LabelCollection());
    cityPoints = viewer.scene.primitives.add(new Cesium.PointPrimitiveCollection());
    applyPlacesMode();
    await buildCitiesTo(cityRungAt(viewer.camera.positionCartographic.height));
    return d;
  }).catch(() => null);
  return cityLoad;
}

/* Descending the ladder. `camera.changed` needs `percentageChanged` set to fire
 * during a move rather than only at its end, and half a screen is about the
 * coarsest step that still fills in ahead of the eye. The +1 look-ahead builds
 * one rung deeper than strictly visible, so a fast zoom finds the names already
 * there instead of watching them pop in. */
viewer.camera.percentageChanged = 0.5;
const descend = () => {
  const h = viewer.camera.positionCartographic.height;
  if (cityData) buildCitiesTo(cityRungAt(h) + 1);
  if (islData) buildIslandsTo(islandExtentAt(h));
  refreshGazetteerLabels();
};
viewer.camera.changed.addEventListener(descend);
viewer.camera.moveEnd.addEventListener(descend);
viewer.camera.changed.addEventListener(updateFineGates);
viewer.camera.moveEnd.addEventListener(updateFineGates);

let bordersLayer = null;
function placesMode() {
  return document.getElementById("places-mode")?.value || "labels";
}
function applyPlacesMode() {
  const mode = placesMode();
  if (cityLabels) { cityLabels.show = mode !== "off"; cityPoints.show = mode !== "off"; }
  if (gazLabels) { gazLabels.show = mode !== "off"; gazPoints.show = mode !== "off"; }
  if (islLabels) islLabels.show = mode !== "off";
}
function updatePlaces() {
  const mode = placesMode();
  if (bordersLayer) { viewer.imageryLayers.remove(bordersLayer, true); bordersLayer = null; }
  if (mode === "full") {
    bordersLayer = viewer.imageryLayers.addImageryProvider(bordersProvider());
    /* Near-full opacity, which sounds wrong for an annotation and isn't: GIBS
     * draws these as pale one-pixel hairlines, already about as quiet as a line
     * can be. Fading them further doesn't make them tactful, it makes them
     * invisible over a busy SST field — which is the only place you'd want a
     * coastline. The restraint is in the artwork, not in the alpha. */
    bordersLayer.alpha = 0.9;
  }
  if (mode !== "off") { ensureCities(); ensureIslands(); }
  applyPlacesMode();
  refreshGazetteerLabels();
  // A searched place keeps its marker whatever this control says — you asked
  // for it — but the altitude band its name occupies depends on whether there
  // is an ordinary label underneath to defer to.
  if (foundPlace) markFoundPlace(foundPlace);
}
/* Every data layer is appended to the TOP of the imagery stack, which would
 * bury the coastlines the moment anything is switched on. Rather than
 * remembering to re-raise at each of the ~six call sites that add imagery (data
 * layers, comparison pairs, the SST ensemble, GBIF), hook the collection's own
 * event — the one place that cannot be forgotten. raiseToTop fires layerMoved,
 * not layerAdded, so this doesn't re-enter. (The city names need none of this:
 * they are scene primitives, which always draw over imagery.) */
viewer.imageryLayers.layerAdded.addEventListener((layer) => {
  if (!bordersLayer || layer === bordersLayer) return;
  viewer.imageryLayers.raiseToTop(bordersLayer);
});

/* ------------------------------------------------------- the gazetteer
 *
 * Natural Earth is a cartographic SELECTION, not a gazetteer. It carries 7,342
 * places worldwide and twenty-four in all of Portugal, which is exactly the
 * right list to keep a map legible and exactly the wrong list to answer a
 * question with. A user looking at the sea off Peniche got a globe that could
 * neither name the town nor be asked about it — the report that produced all
 * of this.
 *
 * So there is a second file with the other job. data/gazetteer.json is GeoNames
 * cities5000 minus everything Natural Earth already has: 54,204 places, each
 * carrying a rung that CONTINUES Natural Earth's ladder rather than starting a
 * new one (the arithmetic is in refresh_data.py gazetteer — one rung down
 * quarters the visible area, so it can carry ~3.3× the places at the same
 * on-screen density, and that factor is measured from Natural Earth's own
 * counts). Peniche lands at rung 10.19, i.e. from about 70 km up.
 *
 * It is lazy, and the trigger is the ladder itself: it loads when you either
 * open the search box or descend past the rung where Natural Earth stops
 * having anything left to say. 1.15 MB gzipped is nothing to someone who wants
 * it and everything to someone who never leaves orbit. */
const GAZ_CAP = 900;   // most labels the deep tier may hold at once — see below
let gazData = null, gazLoad = null, gazFold = null, cityFold = null;
let gazLabels = null, gazPoints = null, gazRect = null;
let gazBuilt = new Set(), gazBuild = null, gazGen = 0;

/* Diacritic-insensitive, the same fold the baker uses. Someone typing "Zurich"
 * on an English keyboard means Zürich, and someone typing "Peniche" should not
 * have to guess whether the file spells it with an accent. */
// The class is written as escapes, not as literal combining marks: a bare
// U+0300 in source is a mark with nothing to sit on, and any tool that
// re-normalises the file would silently eat the range.
const foldName = (s) =>
  s.normalize("NFD").replace(/[\u0300-\u036f]/g, "").toLowerCase();

function ensureGazetteer() {
  if (gazLoad) return gazLoad;
  gazLoad = fetch("data/gazetteer.json").then((r) => r.json()).then((d) => {
    gazData = d;
    // Fold once, not once per keystroke: 54 k normalize() calls is ~40 ms, which
    // is fine on arrival and is not fine between two letters of a place name.
    gazFold = d.places.map((p) => foldName(p.n));
    gazLabels = viewer.scene.primitives.add(new Cesium.LabelCollection());
    gazPoints = viewer.scene.primitives.add(new Cesium.PointPrimitiveCollection());
    applyPlacesMode();
    refreshGazetteerLabels();
    return d;
  }).catch(() => null);
  return gazLoad;
}

/* The rung at which Natural Earth runs out — read from the file rather than
 * written down, so re-baking either file can't leave a gap or an overlap. */
function gazFromRung() {
  return gazData?.zFrom ?? (cityData ? cityData.places[cityData.places.length - 1].z : 9);
}

/* Clearing must also CANCEL: the build is chunked across animation frames, so
 * without bumping the generation an in-flight walk keeps adding labels into the
 * collection we just emptied — names from the town you left, hanging over the
 * globe you flew back to. */
function clearGazetteerLabels() {
  if (!gazLabels) return;
  gazGen++;
  gazBuild = null;
  gazLabels.removeAll(); gazPoints.removeAll();
  gazBuilt = new Set(); gazRect = null;
}

/* Deep-tier labels are SPATIAL, where the Natural Earth tier is a prefix.
 *
 * The prefix walk works up there because the whole file is 7,342 places: by the
 * time every rung is materialised you have paid for all of them and that is
 * survivable. Down here the file is 54,204, and rung 10.8 means "all of them" —
 * so the deep tier is bounded by the VIEW instead. Places stream in as you pan
 * (adding is cheap; removing and re-adding the ones you can still see is not),
 * and only when the accumulated set passes GAZ_CAP does it reset to whatever is
 * in front of you now. That makes the worst case a bounded rebuild rather than
 * an unbounded creep, without making every pan pay a teardown.
 *
 * The array is sorted by rung, i.e. by population, so "the first GAZ_CAP places
 * in this rectangle" is also "the most significant ones" — the cap degrades by
 * dropping villages, not by dropping whatever happened to be scanned last. */
function refreshGazetteerLabels() {
  if (!gazLabels) return;
  const rung = cityRungAt(viewer.camera.positionCartographic.height);
  /* "Nothing should be here" is checked BEFORE the in-flight-build guard, and
   * deliberately: leaving orbit-bound is exactly when a build is most likely to
   * still be running, and that is the one case where returning early would
   * leave the wrong labels on screen rather than merely late. */
  if (placesMode() === "off" || rung < gazFromRung()) { clearGazetteerLabels(); return; }
  if (gazBuild) return;

  const r = viewer.camera.computeViewRectangle(viewer.scene.globe.ellipsoid);
  if (!r || Number.isNaN(r.west) || r.west > r.east) return;   // limb, or antimeridian
  const D = Cesium.Math.DEGREES_PER_RADIAN;
  const view = { w: r.west * D, e: r.east * D, s: r.south * D, n: r.north * D };
  const inside = gazRect && view.w >= gazRect.w && view.e <= gazRect.e
    && view.s >= gazRect.s && view.n <= gazRect.n;
  if (inside && gazBuilt.size <= GAZ_CAP) return;

  if (gazBuilt.size > GAZ_CAP) clearGazetteerLabels();
  // Build for twice the visible box, so a slow pan finds the names already
  // there instead of watching them arrive at the edge of the screen.
  const mx = (view.e - view.w) / 2, my = (view.n - view.s) / 2;
  gazRect = { w: view.w - mx, e: view.e + mx, s: view.s - my, n: view.n + my };

  const box = gazRect, zMax = rung + 1, places = gazData.places;
  let i = 0;
  const gen = ++gazGen;
  gazBuild = new Promise((done) => {
    const step = () => {
      if (gen !== gazGen) { done(0); return; }   // superseded — drop this walk
      const stop = Date.now() + 8;   // one animation frame's worth of budget
      while (i < places.length && gazBuilt.size < GAZ_CAP) {
        const p = places[i];
        if (p.z > zMax) { i = places.length; break; }   // sorted: nothing deeper qualifies
        if (!gazBuilt.has(i) && p.o >= box.w && p.o <= box.e && p.a >= box.s && p.a <= box.n) {
          gazBuilt.add(i);
          addPlace(gazLabels, gazPoints, p, PLACE_FAR0 / Math.pow(2, p.z));
        }
        i++;
        if ((i & 1023) === 0 && Date.now() > stop) break;
      }
      if (i < places.length && gazBuilt.size < GAZ_CAP) requestAnimationFrame(step);
      else { if (gen === gazGen) gazBuild = null; done(gazBuilt.size); }
    };
    step();
  });
  return gazBuild;
}

/* --------------------------------------------------------------- the islands
 *
 * The third tier, and the first one that is not a settlement. A user zoomed to
 * the German Bight and reported Sylt missing; it was missing from both the
 * tiers above for the same structural reason. Natural Earth's populated places
 * and GeoNames cities5000 are gazetteers of PEOPLE. Westerland (9,000 people)
 * is in the second one; the 43 km island it stands on is in neither, because
 * nothing we shipped knew that islands exist. Coastlines are drawn by the
 * borders overlay, so the island was *visible* the whole time and simply had
 * no name — the worst of both, a shape you can see and can't identify.
 *
 * data/islands.json is 4,950 named coastline rings (bake: refresh_data.py
 * islands), each carrying its EXTENT in km rather than a ladder rung — and
 * that difference is the whole design. A town has no size, so its rung has to
 * be assigned by a proxy for importance. An island IS a size, so the rule for
 * when to draw its name can be geometric and exact:
 *
 *     an island earns its name when it is at least as wide on screen
 *     as the name is.
 *
 * Which is a relation between three things the baker cannot know — the canvas
 * width, the camera's field of view, and how wide "Sylt" actually renders in
 * this font — and all three are known HERE. So the client solves
 *
 *     extent_m ≥ (text_px / canvas_px) · 2 · h · tan(fov/2)
 *
 * for h and uses that as the label's far distance. It adapts to the window
 * size and to the length of the name (a wide island with a long name has to be
 * closer than a wide island with a short one, which is exactly right), and it
 * cannot flood the screen: filling the view with island names would require
 * the islands to be wider than the view.
 *
 * It is also, reassuringly, the same ladder the cartographers use. Bucketing
 * all 9,632 Natural Earth rings by the min_zoom of their feature gives median
 * extents falling 170 km → 2.3 km from rung 1 to rung 7: a factor of 2.06 per
 * rung, i.e. extent ∝ 2^-z to within 3%. Halving the camera height doubles the
 * number of islands wide enough to name — which is what the formula says, and
 * is the check that this is a real ladder rather than a preference. */
const ISL_FONT = "italic 11px system-ui, sans-serif";
// The shortest island name still worth building for. Used only to decide how
// much of the (extent-sorted) file to materialise — the per-label condition
// below is what actually decides what is drawn, name by name.
const ISL_MIN_PX = 14;
let islData = null, islLoad = null, islLabels = null;
let islBuilt = 0, islBuild = null, islWide = 0, islFold = null;
let islCtx = null;
const islTextW = new Map();

/* How wide this name renders, measured in the very font it will be drawn in —
 * not estimated from a character count, which would be wrong by a factor of
 * two between "Iō-jima" and "Wrangel Island". */
function islTextWidth(name) {
  let w = islTextW.get(name);
  if (w === undefined) {
    if (!islCtx) {
      islCtx = document.createElement("canvas").getContext("2d");
      if (islCtx) islCtx.font = ISL_FONT;
    }
    w = (islCtx && islCtx.measureText(name).width) || name.length * 6;
    islTextW.set(name, w);
  }
  return w;
}

/* Metres of ground per screen pixel is the only thing the criterion needs from
 * the camera, and it is one number: the view is 2·h·tan(fov/2) wide. Cesium's
 * `fov` is the HORIZONTAL angle whenever the canvas is landscape and the
 * vertical one when it is not, so a portrait window has to convert. */
function islMetresPerPixel(height) {
  const canvas = viewer.scene.canvas;
  const w = canvas.clientWidth || canvas.width || 1200;
  const h = canvas.clientHeight || canvas.height || w;
  const fr = viewer.camera.frustum;
  const fov = fr.fov || Cesium.Math.toRadians(60);
  const aspect = fr.aspectRatio || w / Math.max(h, 1);
  const fovx = aspect > 1 ? fov : 2 * Math.atan(Math.tan(fov / 2) * aspect);
  return { mpp: 2 * height * Math.tan(fovx / 2) / w, px: w };
}

/* The far distance for one island: the height at which its name stops fitting
 * inside it. */
function islandFar(isl) {
  const { mpp } = islMetresPerPixel(1);      // per unit height — linear in h
  return (isl.e * 1000) / (islTextWidth(isl.n) * mpp);
}

function addIsland(isl) {
  // No dot. A town is a point and deserves one; an island is the ground you
  // are already looking at, and a marker in the middle of it would claim a
  // precision — and a kind of thing — that isn't there. Italic is the
  // cartographic convention for a physical feature, and it does the same work
  // the dot does for towns: it says which sort of name this is.
  islLabels.add({
    id: CITY_PICK,
    position: Cesium.Cartesian3.fromDegrees(isl.o, isl.a),
    text: isl.n,
    font: ISL_FONT,
    fillColor: Cesium.Color.WHITE.withAlpha(0.92),
    outlineColor: Cesium.Color.BLACK, outlineWidth: 3,
    style: Cesium.LabelStyle.FILL_AND_OUTLINE,
    distanceDisplayCondition:
      new Cesium.DistanceDisplayCondition(0, islandFar(isl)),
  });
}

/* Same prefix walk as the Natural Earth towns, keyed on extent instead of
 * rung: the file is sorted biggest-first, so "every island that could possibly
 * be labelled at this altitude" is a slice from the front. The cutoff uses the
 * shortest name worth drawing, so the prefix is generous and the per-label
 * condition does the real work. */
function buildIslandsTo(minExtentKm) {
  if (!islData || !islLabels) return islBuild || Promise.resolve(islBuilt);
  const list = islData.islands;
  const wanted = () => islBuilt < list.length && list[islBuilt].e >= minExtentKm;
  if (!wanted()) return islBuild || Promise.resolve(islBuilt);
  islBuild = (islBuild || Promise.resolve()).then(() => new Promise((done) => {
    const step = () => {
      const stop = islBuilt + 300;
      while (islBuilt < stop && wanted()) addIsland(list[islBuilt++]);
      if (wanted()) requestAnimationFrame(step);
      else { islBuild = null; done(islBuilt); }
    };
    step();
  }));
  return islBuild;
}

function islandExtentAt(height) {
  // ÷2 is one rung of look-ahead, the same courtesy the town tier extends.
  return islMetresPerPixel(height).mpp * ISL_MIN_PX / 2000;
}

/* Every far distance is a function of the canvas width, so a window resize or
 * a sidebar drag invalidates all of them at once. The collection is built in
 * file order, so label i is island i and the retune is a walk. */
function retuneIslands() {
  if (!islLabels || !islData) return;
  const w = islMetresPerPixel(1).px;
  if (w === islWide) return;
  islWide = w;
  for (let i = 0; i < islLabels.length; i++) {
    islLabels.get(i).distanceDisplayCondition =
      new Cesium.DistanceDisplayCondition(0, islandFar(islData.islands[i]));
  }
  viewer.scene.requestRender();
}

function ensureIslands() {
  if (islLoad) return islLoad;
  islLoad = fetch("data/islands.json").then((r) => r.json()).then(async (d) => {
    islData = d;
    islLabels = viewer.scene.primitives.add(new Cesium.LabelCollection());
    islWide = islMetresPerPixel(1).px;
    applyPlacesMode();
    await buildIslandsTo(islandExtentAt(viewer.camera.positionCartographic.height));
    return d;
  }).catch(() => null);
  return islLoad;
}

if (typeof ResizeObserver === "function") {
  new ResizeObserver(retuneIslands).observe(viewer.scene.canvas);
}

/* ------------------------------------------------------------- place search
 *
 * Three lists, one box. Natural Earth answers first and answers in English
 * exonyms ("Lisbon", "Cologne"); GeoNames answers in the local name and knows
 * the other 54,204 places. Searching both is not redundancy — it is the only
 * way "Lisbon" and "Peniche" both work.
 *
 * A linear scan over 61 k pre-folded strings is ~3 ms, so there is no index and
 * no debounce beyond one frame: the ranking, not the lookup, is the hard part. */
function placeCountry(p) {
  // cities.json stores the country's name; gazetteer.json and islands.json
  // store ISO-3166 alpha-2 and ship the lookup, because "Portugal" × 500 is
  // the single most compressible thing in a 3.9 MB file.
  return (gazData?.countries?.[p.c]) || (islData?.countries?.[p.c]) || p.c || "";
}

/* One ladder, two ways of earning a place on it. A town carries its rung in
 * the file; an island derives one from the altitude at which its name fits
 * inside it (see islandFar). Expressing the island's geometry AS a rung is
 * what lets search ranking, the fly-to altitude and the found-marker's
 * complement stay one piece of code across all three tiers. */
function placeRung(p) {
  return p.z !== undefined ? p.z : cityRungAt(islandFar(p));
}

function searchPlaces(q, limit = 8) {
  const f = foldName(String(q || "").trim());
  if (f.length < 2) return [];
  if (cityData && !cityFold) cityFold = cityData.places.map((p) => foldName(p.n));
  if (islData && !islFold) islFold = islData.islands.map((p) => foldName(p.n));
  const hits = [];
  const scan = (places, folds) => {
    if (!places || !folds) return;
    for (let i = 0; i < places.length; i++) {
      const at = folds[i].indexOf(f);
      if (at < 0) continue;
      /* Rank by WHERE the match sits, then by the place's own rung. Both matter:
       * "york" must not put York behind New York on position alone, and
       * "san" must not lead with a village. Exact name beats prefix beats
       * start-of-a-later-word beats buried-in-the-middle. */
      const rank = at === 0 ? (folds[i].length === f.length ? 0 : 1)
        : (/[\s\-'’]/.test(folds[i][at - 1]) ? 2 : 3);
      hits.push({ place: places[i], rank });
    }
  };
  scan(cityData?.places, cityFold);
  scan(gazData?.places, gazFold);
  scan(islData?.islands, islFold);
  hits.sort((a, b) => a.rank - b.rank || placeRung(a.place) - placeRung(b.place)
    || (b.place.p || 0) - (a.place.p || 0));
  return hits.slice(0, limit).map((h) => h.place);
}

/* Where to stand to look at a place. Derived from the place's own rung, not
 * picked: fly to one rung closer than the altitude at which its label first
 * appears, and the label is guaranteed to be on screen when you arrive —
 * a hamlet gets a hamlet's altitude, a capital gets a continent's. */
function placeViewHeight(p) {
  return Math.min(4e6, Math.max(26e3, PLACE_FAR0 / Math.pow(2, placeRung(p) + 1)));
}

let foundLabels = null, foundPoints = null, foundPlace = null;
/* The searched place gets its own marker, in the accent colour and with no
 * distance condition at all. Without it, finding Peniche from orbit would fly
 * you to a stretch of coast and then leave you to guess which dot you asked
 * for — and at rungs above its own, the place you searched for is precisely the
 * one the declutter ladder has decided not to draw. */
function markFoundPlace(p) {
  if (!foundLabels) {
    foundLabels = viewer.scene.primitives.add(new Cesium.LabelCollection());
    foundPoints = viewer.scene.primitives.add(new Cesium.PointPrimitiveCollection());
  }
  foundLabels.removeAll(); foundPoints.removeAll();
  foundPlace = p || null;
  if (p) {
    addPlace(foundLabels, foundPoints, p, Infinity, Cesium.Color.fromCssColorString("#79c0ff"));
    /* The NAME, though, is the exact COMPLEMENT of the place's own rung: shown
     * only farther away than the altitude at which the ordinary label appears.
     * Otherwise arriving somewhere draws the name twice, a pixel apart, which
     * reads as a rendering fault. The dot stays visible at every altitude —
     * it is what tells you which of the fifteen names on screen you asked for.
     * (With place names off there is no other label to defer to, so the
     * complement collapses to "always".) */
    const own = placesMode() === "off" ? 0 : PLACE_FAR0 / Math.pow(2, placeRung(p));
    foundLabels.get(0).distanceDisplayCondition =
      new Cesium.DistanceDisplayCondition(own, Number.MAX_VALUE);
  }
  viewer.scene.requestRender();
}

function flyToPlace(p) {
  markFoundPlace(p);
  tideNotePlace(p);          // the tide dashboard follows the search
  viewer.camera.flyTo({
    destination: Cesium.Cartesian3.fromDegrees(p.o, p.a, placeViewHeight(p)),
    duration: 2,
    complete: () => { descend(); },
  });
}

(function wireSearch() {
  const box = document.getElementById("place-search");
  if (!box) return;
  const input = document.getElementById("ps-input");
  const list = document.getElementById("ps-results");
  const clear = document.getElementById("ps-clear");
  let results = [], cursor = -1;

  const close = () => { list.classList.add("hidden"); list.innerHTML = ""; cursor = -1; };
  const render = () => {
    list.innerHTML = "";
    if (!results.length) {
      // A "no match" that says WHICH list was searched, because the honest
      // answer to "why isn't my village here?" is "it is under 5,000 people".
      const li = document.createElement("li");
      li.className = "ps-empty";
      li.textContent = gazData
        ? "No place of that name over 5,000 people."
        : "Loading the gazetteer…";
      list.append(li);
    }
    results.forEach((p, i) => {
      const li = document.createElement("li");
      li.className = "ps-hit" + (i === cursor ? " ps-cursor" : "");
      li.setAttribute("role", "option");
      const country = placeCountry(p);
      li.innerHTML = `<span class="ps-name"></span><span class="ps-where"></span>`;
      li.querySelector(".ps-name").textContent = p.n;
      // An island has no population to report; it has a size, which is the
      // fact that distinguishes the two Melville Islands in the list.
      const extra = p.e !== undefined
        ? ` · island, ${p.e < 10 ? p.e.toFixed(1) : Math.round(p.e)} km across`
        : (p.p > 0 ? ` · ${p.p.toLocaleString()}` : "");
      li.querySelector(".ps-where").textContent = country + extra;
      li.addEventListener("click", () => choose(i));
      list.append(li);
    });
    list.classList.remove("hidden");
  };
  const choose = (i) => {
    const p = results[i];
    if (!p) return;
    input.value = p.n;
    close();
    input.blur();
    flyToPlace(p);
  };
  const run = () => {
    results = searchPlaces(input.value);
    if (input.value.trim().length >= 2) render(); else close();
  };

  // The gazetteer arrives on the first keystroke (and on focus, so it is
  // usually already there by the time two letters are in); the Natural Earth
  // list is already in memory whenever place names are on.
  const load = () => { ensureCities(); ensureIslands(); ensureGazetteer().then(() => { if (!list.classList.contains("hidden")) run(); }); };
  input.addEventListener("focus", load, { once: true });
  input.addEventListener("input", () => { load(); run(); });
  input.addEventListener("keydown", (e) => {
    if (e.key === "ArrowDown" || e.key === "ArrowUp") {
      e.preventDefault();
      if (!results.length) return;
      cursor = (cursor + (e.key === "ArrowDown" ? 1 : results.length - 1) + results.length + 1) % (results.length + 1) - 1;
      if (cursor < 0) cursor = e.key === "ArrowDown" ? 0 : results.length - 1;
      render();
    } else if (e.key === "Enter") {
      e.preventDefault();
      choose(cursor >= 0 ? cursor : 0);
    } else if (e.key === "Escape") {
      close(); input.blur();
    }
  });
  clear.addEventListener("click", () => {
    input.value = ""; close(); markFoundPlace(null); input.focus();
  });
  document.addEventListener("click", (e) => { if (!box.contains(e.target)) close(); });
})();

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
  compareYears: 0,       // comparison offset in years (0 = no offset)
  compareFixed: null,    // pinned comparison date; overrides the offset when set
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

/* ONE calendar stepper, shared by the Date row and the Compare-date row.
 * They are meant to behave identically — Chris, 2026-08-17: "let's make the
 * Date and Compare Date selections analogous" — and two copies of month/year
 * arithmetic is exactly how they would stop being identical. Real calendar
 * rules: -1m from Mar 31 lands on Feb 28, -1y from Feb 29 lands on Feb 28. */
function stepCalendar(dateStr, step) {
  const d = new Date(dateStr + "T00:00:00Z");
  const n = step.startsWith("-") ? -1 : 1;
  const unit = step.slice(-1);
  if (unit === "d") {
    d.setUTCDate(d.getUTCDate() + n);
  } else if (unit === "m") {
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
  return d.toISOString().slice(0, 10);
}

/* The comparison has TWO ways to name its date, and the difference is the
 * whole reason both exist:
 *   OFFSET  (state.compareYears > 0) — "10 years ago", RELATIVE, so it tracks
 *           the main date as you scrub and both sides stay in the same season,
 *           which is what makes a satellite comparison mean anything.
 *   PINNED  (state.compareFixed)     — an absolute date, for "everything vs
 *           July 2003" regardless of where the main date goes.
 * Pinned wins when set; picking an offset from the select clears it. */
function compareDateFor(dateStr) {
  if (state.compareFixed) return state.compareFixed;
  if (!state.compareYears) return null;
  const [y, m, d] = dateStr.split("-").map(Number);
  const day = m === 2 && d === 29 ? 28 : d; // leap-day safety
  return `${y - state.compareYears}-${String(m).padStart(2, "0")}-${String(day).padStart(2, "0")}`;
}

/* The comparison for an ARBITRARY date, not only for the one on screen. The
 * two callers are the live path (`compareDate()`, below, which is this with
 * `state.date` in it) and playback's preload ring, which builds the layers for
 * a frame the globe has not reached yet — and a frame's comparison is a
 * function of THAT frame's date, not of the one currently displayed. The
 * difference is exactly the OFFSET/PINNED distinction above: a pin is the same
 * date for every frame, an offset moves with the frame. Preloading an offset
 * comparison against the wrong past would produce a frame that renders
 * perfectly and states a difference nobody asked for — 2016 vs 2006 labelled
 * as 2017 vs 2007 — which is the class of error this app writes tests against
 * rather than the class it hopes to notice by eye. */
function compareDate() { return compareDateFor(state.date); }

const comparing = () => compareDate() !== null;

/* The comparison date is a DATE, so it obeys the same bounds the main one
 * does: never before GIBS's floor, never past what the UI currently offers. */
function clampUiDate(d) {
  const hi = uiMaxDate();
  return d > hi ? hi : d < "2000-01-01" ? "2000-01-01" : d;
}

/* Push the comparison state into its controls. Called whenever either date
 * moves, because in offset mode the compare date is a FUNCTION of the main
 * one and a stale read-out would be a lie about what is on screen. */
function syncCompareUi(opts) {
  const on = comparing();
  document.getElementById("compare-date-row").classList.toggle("hidden", !on);
  document.getElementById("compare-steps").classList.toggle("hidden", !on);
  const input = document.getElementById("compare-date");
  input.max = uiMaxDate();
  input.min = "2000-01-01";
  // `keepInput` is set by exactly one caller: the field's OWN change handler,
  // mid-edit. Writing back there resets the caret to the first segment (see
  // the comment on the change handler). It is deliberately not a focus check:
  // focus can still be in this field while a DIFFERENT control changes the
  // comparison, and that update must land or the read-out lies about what is
  // on the globe.
  // Belt and braces: `keepInput` covers the field's own handler, and the focus
  // check covers anything ELSE that fires while the user is mid-date
  // (syncDateMax, a layer rebuild). Either way the rule is the same one the
  // Date field gets for free — never write into the field being typed in.
  if (on && !opts?.keepInput && document.activeElement !== input) {
    input.value = compareDate();
  }
  const sel = document.getElementById("compare-select");
  const want = state.compareFixed ? "custom" : String(state.compareYears);
  if (sel.value !== want) sel.value = want;
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
/* ------------------------------------------- one GIBS budget, not two
 *
 * Every tile Cesium draws goes through `Cesium.RequestScheduler`, which caps
 * how many requests may be open to one server at a time
 * (`maximumRequestsPerServer`, 18 by default; the app has never overridden it
 * and, measured, has never needed to). The aggregate / delta / ratio providers
 * and the pixel probe do NOT draw tiles — they READ them, with a bare
 * `fetch()` — so the scheduler never sees them and that cap never applied.
 *
 * Measured 2026-08-21 in MIRROR mode, one aggregable layer, a 1280x720
 * viewport rendering four tiles: moving the Aggregate slider to 365 days
 * issued 48 tile fetches to gibs.earthdata.nasa.gov and **all 48 were in
 * flight simultaneously** (peak concurrency == total). The window's own
 * `windowSampleDates` cap of 12 samples bounds the COUNT correctly; nothing
 * bounded the CONCURRENCY. A full-screen desktop view (~11 rendered tiles)
 * with three aggregable layers on is ~400 simultaneous connections to one
 * public, taxpayer-funded NASA host from a single tab — which is what a
 * denial-of-service looks like from the far end, whatever it was meant as.
 *
 * The fix deliberately invents no number: the raw-read path is admitted
 * through the SAME per-server budget the scheduled path already respects,
 * read from the scheduler itself rather than typed in a second time. If a
 * future Cesium changes its default, both paths change together.
 *
 * `gibsSlowed` is the one thing that lowers it, and only on the server's own
 * say-so — see `gibsPushback`. */
function gibsRawLimit() {
  if (gibsSlowed) return 1;                 // the least that still progresses
  const n = Cesium.RequestScheduler.maximumRequestsPerServer;
  return typeof n === "number" && n > 0 ? n : 6;
}
let gibsRawActive = 0;
const gibsRawQueue = [];
function gibsRawAcquire() {
  if (gibsRawActive < gibsRawLimit()) { gibsRawActive++; return Promise.resolve(); }
  return new Promise((r) => gibsRawQueue.push(r));
}
function gibsRawRelease() {
  gibsRawActive = Math.max(0, gibsRawActive - 1);
  while (gibsRawQueue.length && gibsRawActive < gibsRawLimit()) {
    gibsRawActive++;
    gibsRawQueue.shift()();
  }
}

/* GIBS ASKING US TO SLOW DOWN IS AN ANSWER, AND IT USED TO BE INVISIBLE.
 * `sstFetchBitmap` returned null for every non-OK status, so a 429 or a 503
 * was indistinguishable from "this tile has no data" — the layer simply went
 * quiet, the reader blamed the archive, and the app carried on asking at the
 * same rate. That is precisely the behaviour that turns a rate limit into a
 * block, and a block on a public service cannot be bought back.
 *
 * So the two statuses that MEAN "you are asking too fast" say so once, in the
 * app's own toast, and collapse the GIBS budget to a single concurrent
 * request for the rest of the session. Nothing here guesses a back-off
 * interval: an invented delay would be a hand-picked threshold, and the
 * session-long floor of one is both the minimum that still makes progress and
 * the answer that cannot be wrong. A reload is the reset, and the toast says
 * so. */
let gibsSlowed = false;
function gibsPushback(status) {
  if (gibsSlowed) return;
  gibsSlowed = true;
  showToast(
    `<strong>NASA GIBS asked us to slow down</strong> (HTTP ${status}). ` +
    "The map now makes one tile request at a time, so it will fill in more " +
    "slowly. Reload the page to go back to full speed.",
    { key: "gibs-pushback", timeout: 12000 });
}

async function sstFetchBitmap(url) {
  await gibsRawAcquire();
  try {
    const r = await fetch(url);
    if (r.status === 429 || r.status === 503) { gibsPushback(r.status); return null; }
    if (!r.ok) return null;
    return await createImageBitmap(await r.blob());
  } catch {
    return null;
  } finally {
    gibsRawRelease();
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
  // current speed on a dark globe: still water is deep navy (reads as ocean),
  // fast water glows cyan → white-hot — the Gulf Stream draws itself. The old
  // precip ramp started at white, which painted the whole slow ocean
  // near-white and washed the globe out.
  speed: [[0, 12, 24, 48], [0.18, 20, 62, 118], [0.42, 28, 132, 186],
          [0.68, 90, 220, 210], [1, 250, 255, 220]],
  // diverging anomaly: blue → near-invisible dark slate → red, matching the
  // app's blue=cooler/less, red=warmer/more convention; the dark middle lets
  // near-zero cells fade into the globe instead of painting it white
  anom: [[0, 37, 99, 235], [0.42, 84, 110, 160], [0.5, 45, 51, 63],
         [0.58, 160, 95, 84], [1, 230, 59, 46]],
  // cold → warm thermal ramp for SST
  sst: [[0, 49, 54, 149], [0.25, 116, 173, 209], [0.5, 255, 255, 191],
        [0.75, 244, 109, 67], [1, 165, 0, 38]],
  // 2 m air temperature (-30..40 °C): saturated through the mid-range where
  // most of the inhabited world lives — the sst ramp's pale middle washed
  // the whole forecast globe out
  t2m: [[0, 40, 45, 150], [0.28, 62, 140, 214], [0.43, 92, 200, 190],
        [0.57, 245, 215, 90], [0.72, 240, 130, 48], [0.86, 205, 55, 35],
        [1, 130, 10, 25]],
  // forecast rain: dry is already transparent, so light rain starts as a soft
  // teal and heavy rain deepens to violet — the white-starting precip ramp
  // painted every drizzle cell white and blanketed the globe (same lesson as
  // the currents "speed" ramp)
  rain: [[0, 130, 200, 205], [0.25, 72, 150, 214], [0.55, 42, 84, 190],
         [0.8, 90, 45, 170], [1, 150, 30, 140]],
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

/* Month-keyed grids (GLORYS): the file carries months:{YYYY-MM: [...]} and the
 * date selector picks which month renders/samples. Out-of-range dates clamp to
 * the nearest baked month; in-range dates floor to the newest month <= date. */
function gridMonths(g) {
  if (!g) return null;
  if (g.monthsAvailable) return g.monthsAvailable;   // index file: full archive list
  return g.months ? Object.keys(g.months).sort() : null;
}
/* `dateStr` defaults to the date on screen, which is what every read-out and
 * every provider wants. Playback needs the same question asked about a date
 * that is NOT on screen yet — "what would this grid serve on 2015-03-04?" — to
 * decide whether that candidate frame differs from its predecessor at all, so
 * the date is a parameter rather than a global read. */
function resolveGridMonth(g, dateStr = state.date) {
  const ms = gridMonths(g);
  if (!ms) return null;
  // keyLen 7 = month-keyed (GLORYS); keyLen 10 = day-keyed (GFS forecast)
  const want = dateStr.slice(0, g.keyLen || 7);
  let best = ms[0];                       // dates before the range clamp up
  for (const m of ms) {
    if (m <= want) best = m;              // floor; dates after the range clamp down
    else break;
  }
  return best;
}
function gridValues(g) {
  return g.months ? g.months[resolveGridMonth(g)] : g.values;
}

/* The GLORYS index files inline only the latest year; older months live in
 * per-year files (data/currents_y/1993.json, ...) fetched on first use and
 * merged into g.months. Callers that sample must go through loadGridMonth. */
function ensureGridMonth(cfg, g, month) {
  if (!g || !month || g.months[month] || !g.yearDir) return Promise.resolve(g);
  g.__yearLoads = g.__yearLoads || {};
  const yr = month.slice(0, 4);
  if (!g.__yearLoads[yr]) {
    g.__yearLoads[yr] = fetch(`${g.yearDir}/${yr}.json`)
      .then((r) => r.json())
      .then((y) => { Object.assign(g.months, y.months); })
      .catch(() => { delete g.__yearLoads[yr]; });   // retry on next access
  }
  return Promise.resolve(g.__yearLoads[yr]).then(() => g);
}
async function loadGridMonth(cfg) {
  const g = await loadGrid(cfg);
  if (g && cfg.monthlyGrid) await ensureGridMonth(cfg, g, resolveGridMonth(g));
  return g;
}

function sampleGrid(g, lonDeg, latDeg) {
  if (lonDeg < g.west || lonDeg >= g.east || latDeg < g.south || latDeg >= g.north) return null;
  const ix = Math.floor((lonDeg - g.west) / g.dlon);
  const iy = Math.floor((latDeg - g.south) / g.dlat);
  if (ix < 0 || ix >= g.nx || iy < 0 || iy >= g.ny) return null;
  const vals = gridValues(g);
  if (!vals) return null;                 // month not loaded yet (year file in flight)
  const v = vals[iy * g.nx + ix];
  return v == null ? null : v;
}

const gridCache = new Map();
/* The resolved grids, readable WITHOUT awaiting. `loadGrid` is a promise cache
 * and every renderer is happy to await it; frame enumeration is not — it runs
 * synchronously from the panel and from tests, and a grid that has not arrived
 * yet must cost it nothing. So a resolved grid is also parked here, and the
 * enumerator simply omits any grid it cannot see (a signature missing a part is
 * COARSER — it can merge two frames that differ — never wrong about a date). */
const gridsLoaded = new Map();
function loadGrid(cfg) {
  if (!gridCache.has(cfg.id)) {
    gridCache.set(cfg.id, fetch(cfg.gridFile).then((r) => r.json())
      .then(unpackGrid)
      .then((g) => { if (g) gridsLoaded.set(cfg.id, g); return g; })
      .catch(() => null));
  }
  return gridCache.get(cfg.id);
}

/* Categorical grids ship `packed` — one character per cell, "." for empty —
 * instead of a `values` array. Same numbers, a quarter of the bytes, because a
 * JSON array of 800k single digits and nulls is mostly punctuation; the driver
 * grid goes 3.5 MB → 0.8 MB, which matters because the pixel inspector reads it
 * on a click. Expanded once here so every sampler downstream is unchanged. */
function unpackGrid(g) {
  if (g && g.packed && !g.values) {
    const p = g.packed, out = new Array(p.length);
    for (let i = 0; i < p.length; i++) {
      const c = p.charCodeAt(i);
      out[i] = c === 46 ? null : c - 48;         // "." → null, "0".."9" → digit
    }
    g.values = out;
  }
  return g;
}

/* Categorical grids carry their classes in the baked file itself (code, label,
 * rgb) rather than in the layer config: the palette is the DATA PRODUCER's, and
 * shipping it alongside the values keeps the two from drifting apart when the
 * dataset is re-baked with a class added or renamed. */
function gridClassPalette(g) {
  if (!g?.classes) return null;
  if (!g.__pal) g.__pal = new Map(g.classes.map((c) => [c.code, c.rgb]));
  return g.__pal;
}
function gridClassLabel(g, v) {
  return g?.classes?.find((c) => c.code === v)?.label ?? null;
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
    const g = await loadGridMonth(this._cfg);
    const W = this.tileWidth, H = this.tileHeight;
    const canvas = document.createElement("canvas");
    canvas.width = W; canvas.height = H;
    const ctx = canvas.getContext("2d", { willReadFrequently: true });
    if (!g) return canvas;
    const rect = this.tilingScheme.tileXYToRectangle(x, y, level);
    const west = Cesium.Math.toDegrees(rect.west), east = Cesium.Math.toDegrees(rect.east);
    const north = Cesium.Math.toDegrees(rect.north), south = Cesium.Math.toDegrees(rect.south);
    const { vmin, vmax, ramp } = this._cfg;
    // Categorical grid (drivers of forest loss): the cell value is a class
    // code, so it indexes a palette instead of running through a ramp. A ramp
    // would invent an ordering — "logging" is not between "wildfire" and
    // "settlements", it is simply a different thing.
    const pal = this._cfg.classGrid ? gridClassPalette(g) : null;
    const out = ctx.createImageData(W, H);
    const o = out.data;
    for (let j = 0; j < H; j++) {
      const lat = north - ((j + 0.5) / H) * (north - south);
      for (let i = 0; i < W; i++) {
        const lon = west + ((i + 0.5) / W) * (east - west);
        const v = sampleGrid(g, lon, lat);
        if (v == null) continue;
        const c = pal ? pal.get(v) : rampColor(ramp, (v - vmin) / (vmax - vmin));
        if (!c) continue;
        const k = (j * W + i) * 4;
        o[k] = c[0]; o[k + 1] = c[1]; o[k + 2] = c[2]; o[k + 3] = 225;
      }
    }
    ctx.putImageData(out, 0, 0);
    return canvas;
  }
}

/* WHAT A FRAME LOOKS LIKE, as a pure function of (layer config, date).
 *
 * This is the construction half of `addLayer` and nothing else: it decides
 * which providers a layer needs for a date — delta, ratio, split pair,
 * windowed mean, raw tile, painted grid — and returns them. It writes no
 * state, adds nothing to the globe, touches no DOM, fires no toast. All of
 * that stays in `addLayer` below.
 *
 * The split exists because playback now builds layers in TWO places: the live
 * path (`addLayer`, for the date on screen) and the preload ring (for frames
 * i+1…i+depth, at alpha 0). If those two ever disagreed about what frame N
 * looks like, playback would promote one picture and the paused globe would
 * then rebuild a different one — a bug that shows up as the animation and the
 * still image quietly contradicting each other, which is the hardest kind to
 * see and the easiest kind to disbelieve. One definition, consumed twice, is
 * the only version of this that cannot drift.
 *
 * The DATE IS A PARAMETER, and that is load-bearing rather than tidy: every
 * date-dependent decision inside — the comparison (`compareDateFor`), the
 * window's sample dates, the tile time — must be computed for the frame being
 * built, not for the frame currently displayed. See `compareDateFor`. */
function providersFor(cfg, dateStr) {
  const cmp = compareDateFor(dateStr);
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
  const out = { suppressed: false, providers: [], isDelta: false, isRatio: false, isAggregate: false };

  // While an aggregation window is active, everything on screen must actually
  // BE an average. A timed raster that cannot be time-averaged (photographic
  // true colour, instantaneous precipitation) would silently show one day's
  // data under a "past N days" label — so it displays NOTHING instead, and
  // the panel hint explains why. (Untimed composites and the climatology
  // grids stay: they already are long-period averages.)
  if (win > 1 && cfg.timed && !canWindow) {
    out.suppressed = true;
    return out;
  }

  if (cfg.grid) {
    // Climatology or month-keyed grid painted client-side from data/<id>.json.
    // The date does not enter the provider at all — a keyed grid resolves its
    // month at paint time, which is why grids need no preload ring (§ the ring
    // below) and why playing a grid layer costs no tile traffic.
    out.providers.push({ provider: new GridProvider(cfg) });
    return out;
  }

  if (comparing && state.compareMode === "delta" && deltaable) {
    // Computed per-pixel difference of window means (single-day if win === 1)
    out.providers.push({ provider: new DeltaProvider(cfg, dateStr, cmp, win) });
    out.isDelta = true;
  } else if (comparing && state.compareMode === "delta" && cfg.ratioRange) {
    // Log-distributed field: computed comparison is a ×-fold ratio of means
    out.providers.push({ provider: new RatioProvider(cfg, dateStr, cmp, win) });
    out.isRatio = true;
  } else if (comparing && state.compareMode === "split") {
    // Side-by-side: right = current, left = past. Windowed means for SST, raw tiles otherwise.
    out.providers.push({
      provider: windowed ? new SSTAggregateProvider(cfg, dateStr, win) : gibsProvider(cfg, dateStr),
      splitDirection: Cesium.SplitDirection.RIGHT,
    });
    out.providers.push({
      provider: windowed ? new SSTAggregateProvider(cfg, cmp, win) : gibsProvider(cfg, cmp),
      splitDirection: Cesium.SplitDirection.LEFT,
    });
    out.isAggregate = windowed;
  } else {
    // Not comparing: single layer — windowed mean for SST, raw tile otherwise
    out.providers.push({
      provider: windowed ? new SSTAggregateProvider(cfg, dateStr, win) : gibsProvider(cfg, dateStr),
    });
    out.isAggregate = windowed;
  }
  return out;
}

/* `providersFor` decided WHAT; everything here is the side effects — putting
 * the providers on the globe, recording the entry in `state.layers`, the
 * legend, the GIBS domain probe, the monthly-grid bookkeeping. Splitting the
 * two changed no behaviour: the branches below are the same branches, in the
 * same order, with the same flags. */
function addLayer(cfg) {
  const entry = { cfg, layer: null, cmpLayer: null, isDelta: false, isRatio: false,
    isAggregate: false, alpha: state.layers[cfg.id]?.alpha ?? 1.0 };
  const built = providersFor(cfg, state.date);

  if (built.suppressed) {
    entry.suppressed = true;
    state.layers[cfg.id] = entry;
    updateLegends();
    ensureGibsDomain(cfg);
    return;
  }

  const add = (p) => {
    const layer = viewer.imageryLayers.addImageryProvider(p.provider);
    layer.alpha = entry.alpha;
    if (p.splitDirection !== undefined) layer.splitDirection = p.splitDirection;
    return layer;
  };
  entry.layer = add(built.providers[0]);
  if (built.providers[1]) entry.cmpLayer = add(built.providers[1]);
  entry.isDelta = built.isDelta;
  entry.isRatio = built.isRatio;
  entry.isAggregate = built.isAggregate;
  applyFineGate(entry);          // a 30 m layer above its gate is kept, hidden, unrequested

  if (cfg.grid) {
    state.layers[cfg.id] = entry;
    if (cfg.monthlyGrid) {
      // remember which month rendered, so a date change knows when to repaint
      loadGrid(cfg).then((g) => {
        if (!g) return;
        entry.gridMonth = resolveGridMonth(g);
        if (cfg.forecastGrid && g.latest) {
          forecastMaxDate = forecastMaxDate && forecastMaxDate > g.latest
            ? forecastMaxDate : g.latest;
          syncDateMax();                   // open the date selector to the future
        }
      });
    }
    updateLegends();
    return;
  }

  state.layers[cfg.id] = entry;
  updateLegends();
  // Ask GIBS what this layer actually serves. No-op after the first enable
  // (cached for the session), and it never blocks the paint above.
  ensureGibsDomain(cfg);
  // PREFETCH the SST normals with the anomaly layer. The hover probe renders
  // synchronously, so a value that needs a round-trip cannot appear in it at
  // all — loading here is what lets a saturated pixel read its real departure
  // the moment it is hovered, instead of "≥ 3".
  if (cfg.id === "sst-anom") ensureSstNormals(state.date);
}

/* ------------------------------------------------------- the fine tier's gate
 * A layer with `fine: <km>` is a 30 m (or 15 m) product whose tiles only make
 * sense close up: the daily ones are satellite SWATHS, so a full-globe view is
 * mostly blank, and every date step would re-fetch a fresh set of blank tiles
 * for the whole visible globe. Above that camera height the layer stays in
 * `state.layers` — its chip, legend, opacity row and hover card all persist —
 * but its ImageryLayer is HIDDEN. That is the whole mechanism: Cesium creates
 * tile skeletons only for shown layers (`layer.show &&
 * _createTileImagerySkeletons`, the same fact the retirement queue rests on),
 * so a hidden layer requests nothing at all. Crossing the gate flips `show`
 * and the tiles for the area in view — a few dozen at 500 km — arrive then.
 * The playback ring honours the gate too (`playbackPreloadAdd`), so a fine
 * layer in a playing set costs nothing until the camera is low enough to see
 * it, and is then fetched for the current frame like any other. */
function fineGated(cfg) {
  return !!cfg?.fine && cameraHeight() > cfg.fine * 1000;
}
function applyFineGate(entry) {
  if (!entry?.cfg?.fine) return;
  const show = !fineGated(entry.cfg);
  if (entry.layer) entry.layer.show = show;
  if (entry.cmpLayer) entry.cmpLayer.show = show;
}
function fmtKm(m) {
  const km = m / 1000;
  return km >= 100 ? String(Math.round(km)) : km >= 10 ? km.toFixed(1) : km.toFixed(2);
}
/* The row's one-line status under a fine layer: what it is doing right now
 * and, when hidden, what would change that. Updated on every camera move so
 * the number a user reads is the height they are actually at. */
function updateFineGates() {
  for (const entry of Object.values(state.layers)) {
    const cfg = entry.cfg;
    if (!cfg.fine) continue;
    applyFineGate(entry);
    const hint = document.querySelector(`[data-finehint="${cfg.id}"]`);
    if (!hint) continue;
    const on = !!(entry.layer || entry.suppressed);
    hint.hidden = !on;
    if (!on) continue;
    const gated = fineGated(cfg);
    hint.classList.toggle("fine-gated", gated);
    hint.textContent = gated
      ? `⤵ zoom in — hidden above ${cfg.fine} km (you're at ${fmtKm(cameraHeight())} km)`
      : `showing ${cfg.tms === "15.625m" ? "15" : "30"} m tiles for the area in view`;
  }
}
/* Said once, on enable, when the layer will not appear until the camera comes
 * down — otherwise a checked box and an unchanged globe read as a broken layer. */
function maybeFineToast(cfg) {
  if (!cfg?.fine || !fineGated(cfg)) return;
  showToast(`<strong>${cfg.title}</strong> is a fine-resolution layer, so its tiles are ` +
    `fetched only for the area in view: <strong>zoom in below ${cfg.fine} km</strong> to load ` +
    `it (you're at ${fmtKm(cameraHeight())} km). ` +
    (cfg.timed ? `It shows the satellite's <strong>swaths for the chosen day</strong> — blank ` +
      `means no pass that day, not no data.` : ``), { key: `fine-${cfg.id}` });
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

/* ------------------------------------------------------- the retirement queue
 * `removeLayer` destroys the old ImageryLayer BEFORE the replacement has any
 * tiles, so for the length of one network round trip the globe shows bare base
 * map. On the date stepper you see it as a blink; at 2 fps of playback it is a
 * strobe, and it makes correct data look broken.
 *
 * Cesium settles the fix for us. In GlobeSurfaceTileProvider the tile skeletons
 * are created behind `layer.show && layer._createTileImagerySkeletons(...)`, so
 * a layer that is still SHOWN keeps painting what it already has and costs
 * nothing new, while the replacement — appended above it — requests its own
 * tiles and covers it the moment they arrive. Holding the old frame is
 * therefore a matter of postponing the destroy, not of rewriting `addLayer`:
 * delta/split/aggregate/window/grid all keep working, untouched.
 *
 * The queue is BOUNDED at three. On a slow network a scrub could otherwise
 * stack fifty live imagery layers, each still compositing, each holding its
 * tile textures — which trades a blink for a memory leak and a fps cliff. When
 * a fourth arrives the oldest is destroyed on the spot: at that depth it is
 * certainly covered, and if it is not, one blink is the honest price. */
const RETIRE_MAX = 3;
const retiring = [];
let lastRetireMs = 0;

function destroyRetired(layer) {
  try { viewer.imageryLayers.remove(layer, true); } catch { /* already gone */ }
}

function retireLayer(id) {
  const entry = state.layers[id];
  if (!entry) return;
  /* One held generation PER LAYER. Holding two old generations of the same
   * layer buys nothing — the newer one already covers the older, so the older
   * can only cost tile memory and compositing time — and a fast scrub is
   * exactly where that cost lands. */
  for (let k = retiring.length - 1; k >= 0; k--) {
    if (retiring[k].id === id) destroyRetired(retiring.splice(k, 1)[0].layer);
  }
  if (entry.layer) retiring.push({ id, layer: entry.layer });
  if (entry.cmpLayer) retiring.push({ id, layer: entry.cmpLayer });
  lastRetireMs = Date.now();
  // Over the bound, destroy from the FRONT — the oldest held frame is the one
  // most likely to be hidden under everything added since.
  while (retiring.length > RETIRE_MAX) destroyRetired(retiring.shift().layer);
  entry.layer = null;
  entry.cmpLayer = null;
  entry.isDelta = false;
  entry.isRatio = false;
  entry.isAggregate = false;
  entry.suppressed = false;
  updateLegends();
}

function sweepRetired() {
  while (retiring.length) destroyRetired(retiring.shift().layer);
}

/* "The new frame is on screen" is not a thing Cesium tells us directly; the
 * closest true statement is "the globe's tile queue is empty". Two subtleties
 * decide whether reading it means anything:
 *
 *  - A GRACE PERIOD is mandatory. Immediately after `addLayer` the queue is
 *    still empty because the new layer has not been asked for anything yet, so
 *    an unguarded read says "settled" about the frame we just replaced and we
 *    would destroy the old layer before the new one requested a single tile —
 *    exactly the blink this whole mechanism exists to remove.
 *  - It must POLL as well as listen. `tileLoadProgressEvent` fires on CHANGE,
 *    so a frame that needs no new tiles at all (a grid-only layer set, a
 *    suppressed layer, a repeat of a cached date) never fires it, and a
 *    listener alone would wait for an event that is never coming.
 *
 * Resolves "settled" or "ceiling", and cleans up on both paths — a leaked tile
 * listener per frame would be a slow poisoning of the render loop. */
const PLAY_SETTLE_GRACE_MS = 180;
const PLAY_SWEEP_MAX_MS = 10000;

function waitTilesSettled(maxMs) {
  return new Promise((resolve) => {
    const globe = viewer.scene.globe;
    let off = null, poll = null, graceT = null, done = false;
    const finish = (why) => {
      if (done) return;
      done = true;
      if (off) off();
      if (poll) clearInterval(poll);
      clearTimeout(graceT);
      clearTimeout(capT);
      resolve(why);
    };
    const capT = setTimeout(() => finish("ceiling"), maxMs);
    graceT = setTimeout(() => {
      if (done) return;
      off = globe.tileLoadProgressEvent.addEventListener((n) => { if (n === 0) finish("settled"); });
      poll = setInterval(() => { if (globe.tilesLoaded) finish("settled"); }, 120);
      if (globe.tilesLoaded) finish("settled");
    }, PLAY_SETTLE_GRACE_MS);
  });
}

// One sweep in flight at a time. The cap matters: a layer set that requests no
// tiles (everything suppressed by the aggregation window) would otherwise hold
// its retired predecessors on screen forever, which reads as the window having
// done nothing.
let sweepPending = false;
function scheduleSweep() {
  if (sweepPending) return;
  sweepPending = true;
  waitTilesSettled(PLAY_SWEEP_MAX_MS).then(() => { sweepPending = false; sweepRetired(); });
}

/* The fast path: the globe reporting an empty queue is the earliest honest
 * moment to drop the held frame. Guarded by the same grace as above, because
 * the queue is briefly empty between retiring the old layer and the new one
 * enqueueing its first request. */
viewer.scene.globe.tileLoadProgressEvent.addEventListener((remaining) => {
  if (remaining === 0 && retiring.length && Date.now() - lastRetireMs > PLAY_SETTLE_GRACE_MS) {
    sweepRetired();
  }
});

/* `hold: true` builds the new layer ON TOP of the old one and leaves the old
 * one painting until the replacement has covered it (see the retirement queue
 * above). Every date-driven call site passes it; the plain form is what the
 * single-date path uses when it wants the picture to be exactly what state
 * says it is — notably when playback stops. */
function refreshTimedLayers({ hold = false, keepPreload = false } = {}) {
  /* A rebuild of the timed layers means the CONFIGURATION changed — a layer
   * toggled, the comparison re-pointed, the aggregation window moved, the
   * date selector used by hand. Playback's preload ring holds layers built
   * for the configuration as it was, so every one of those is an
   * invalidation, and clearing here rather than at each call site is what
   * stops the ring drifting from the app the next time somebody adds a
   * control that calls this. A stale ring is strictly worse than no ring: it
   * would promote, instantly and invisibly, a frame built for a comparison
   * or a window the user has since changed.
   *
   * `keepPreload` has exactly one caller — playback's own per-frame fallback,
   * for a frame the ring did not happen to hold. That is not a configuration
   * change, and clearing there would destroy the ring on every frame the
   * player fails to preload, which is the one moment it is most needed. */
  if (!keepPreload) playbackPreloadClear();
  // The date moved, so the normals may be for the wrong month now. Reload in
  // the background if the anomaly layer is on; the stamp check makes this a
  // no-op while the month is unchanged.
  if (state.layers["sst-anom"]?.layer) ensureSstNormals(state.date);
  // Suppressed entries (hidden because the active window can't average them)
  // must also refresh, so they reappear when the window returns to 1 day.
  for (const [id, entry] of Object.entries(state.layers)) {
    if ((entry.layer || entry.suppressed) && entry.cfg.timed) {
      const cfg = entry.cfg;
      if (hold) retireLayer(id); else removeLayer(id);
      addLayer(cfg);
      // A date change can land in a hole as easily as an enable can — browsing
      // NDVI back through 2025 walks straight into a month GIBS never
      // published. Say so here too, or the globe just goes quiet mid-scrub.
      maybeArchiveToast(cfg, { replace: true });
      maybeAnnualToast(cfg, { replace: true });
    }
  }
  if (hold) scheduleSweep();
  updateSplitUI();
}

/* ------------------------------------------------- the date scrub coalescer
 *
 * A date change costs one whole visible tile set PER TIMED LAYER, and nothing
 * used to stand between the user's finger and that cost. Measured 2026-08-21
 * in MIRROR mode, a 1280x720 viewport rendering four tiles per layer:
 *
 *   one -1d click, 1 layer .................................. 4 tile requests
 *   one -1d click, 5 layers ................................ 20 tile requests
 *   the date field moved 60 days at 30 Hz (a HELD arrow key,
 *     which is what a browser's key-repeat does), 1 layer ... 240 requests
 *   the Play tab's scrub slider dragged, 40 input events .... 160 requests
 *
 * — i.e. exactly 4 per event, with no debounce, and (measured on the same
 * runs) **zero** of the superseded requests cancelled: every tile for a date
 * the user had already left completed and was thrown away. The count is a
 * pure function of how fast a finger can move, which is the definition of the
 * hazard: hold the arrow key for ten seconds with five layers on and a single
 * tab asks a public NASA service for ~6,000 tiles it will never show.
 *
 * The fix is not a timer and not a picked interval. The app already has a
 * definition of "this date is now on screen" — `waitTilesSettled`, the same
 * one the playback loop advances its playhead on — so the rule is simply
 * ONE DATE GENERATION IN FLIGHT AT A TIME: apply the first change
 * immediately, remember only the LATEST date requested while it paints, and
 * apply that one when the globe reports its queue empty. The request rate
 * stops being a function of the user's finger and becomes a function of the
 * network's own throughput, which is the only rate that was ever affordable.
 *
 * Two properties this deliberately keeps:
 *  - `state.date`, the date input, the comparison read-out and everything
 *    else that costs NOTHING still move at the user's rate. Only the part
 *    that issues requests is coalesced, so the UI never feels gated.
 *  - An ISOLATED change still applies synchronously, on the same tick as the
 *    click. That is what the retirement-queue test reads, and more
 *    importantly it is what a single date step should do.
 *
 * It also makes request CANCELLATION unnecessary rather than adding it: the
 * superseded requests measured above are not cancelled here, they are never
 * issued. CLAUDE.md 4.1 — prefer removing a failure mode over guarding it. */
let scrubBusy = false;
let scrubPending = null;
function scrubApply(fn) {
  if (scrubBusy) { scrubPending = fn; return; }   // newest wins; the rest never run
  scrubBusy = true;
  fn();
  waitTilesSettled(PLAY_FRAME_CEILING_MS).then(() => {
    scrubBusy = false;
    const next = scrubPending;
    scrubPending = null;
    if (next) scrubApply(next);
  });
}

/* Everything a date move has to rebuild. The closure reads `state.date` at
 * APPLY time rather than capturing it, so a coalesced burst lands on the date
 * the user actually stopped at, not on the one that happened to arrive first. */
function applyDateMove() {
  scrubApply(() => {
    refreshTimedLayers({ hold: true });
    refreshYearlyLayers();
    refreshMonthlyGrids();
    if (sstEnsembleLayer) updateEnsembleLayer();
  });
}

/* Forecast layers extend the date selector past today: while one is active,
 * the max selectable date is the last baked forecast day instead of the last
 * observed one. Everything else (GIBS requests, "today" button) stays pinned
 * to real time — gibsTime clamps future dates for observation layers. */
let forecastMaxDate = null;                // last forecast day, set on grid load
function uiMaxDate() {
  const anyForecast = Object.values(state.layers)
    .some((e) => e.layer && e.cfg.forecastGrid);
  return anyForecast && forecastMaxDate ? forecastMaxDate : defaultDate();
}
function syncDateMax() {
  const input = document.getElementById("layer-date");
  if (!input) return;
  const max = uiMaxDate();
  input.max = max;
  if (state.date > max) {                  // forecast switched off while in the future
    state.date = max;
    input.value = max;
    refreshTimedLayers();
    refreshYearlyLayers();
    refreshMonthlyGrids();
  }
  // The comparison lives on the same axis: when the axis shortens, a pinned
  // date past the new end has to come back with it, or the comparison would
  // silently request tiles the UI no longer admits exist.
  if (state.compareFixed && state.compareFixed > max) state.compareFixed = max;
  syncCompareUi();
}

/* Month-keyed grids don't listen to refreshTimedLayers (they're not `timed`),
 * so when the date's MONTH moves to a different baked month, rebuild them —
 * Cesium caches rendered tiles, so a repaint needs a fresh provider. */
async function refreshMonthlyGrids() {
  for (const [id, entry] of Object.entries(state.layers)) {
    if (!entry.layer || !entry.cfg.monthlyGrid) continue;
    const g = await loadGrid(entry.cfg);
    const m = g && resolveGridMonth(g);
    if (m && entry.gridMonth !== m) {
      removeLayer(id);
      addLayer(entry.cfg);
      maybeMonthlyGridToast(entry.cfg);   // say which month is now showing
    }
  }
}

function anyTimedActive() {
  return Object.values(state.layers).some((e) => e.layer && e.cfg.timed);
}

/* ------------------------------------------------------- comparison (split) */

const splitHandle = document.getElementById("split-handle");
const splitLabels = document.getElementById("split-labels");

function updateSplitUI() {
  document.getElementById("compare-mode-row").classList.toggle("hidden", !comparing());
  const active = comparing() && anyTimedActive() && state.compareMode === "split";
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
  const v = e.target.value;
  if (v === "custom") {
    // Seed the pin from whatever is on screen, so choosing "a specific date"
    // never blanks the comparison — it hands you the date you were already
    // looking at, to edit. A default of "today" would be a zero difference.
    state.compareFixed = state.compareFixed || compareDate() ||
      clampUiDate(stepCalendar(state.date, "-1y"));
    state.compareYears = 0;
  } else {
    state.compareYears = Number(v);
    state.compareFixed = null;   // an offset TRACKS; a pin does not
  }
  syncCompareUi();
  refreshTimedLayers({ hold: true });
});

/* Typing a comparison date PINS it — the offset is abandoned, because the two
 * cannot both be true and the one you just typed is the one you meant. */
/* Typing a comparison date PINS it — the offset is abandoned, because the two
 * cannot both be true and the one you just typed is the one you meant.
 *
 * Shaped exactly like the Date field above, and that shape IS the fix. An
 * `<input type="date">` fires `change` as each SEGMENT completes, so typing
 * "2010" reports the real dates 0002, 0020, 0201 on the way. The first
 * version answered each of those by clamping to the floor and writing the
 * clamp back into the field — which reset the caret to the first segment on
 * every keystroke, so the year could never get past its first digit (Chris,
 * 2026-08-18: "I cannot type 2010 into it ... editing only the first number
 * of the year"). Date never had the bug because Date never writes into its
 * own field. Neither does this now: `keepInput` suppresses exactly that one
 * write, and an out-of-range partial is simply ignored rather than corrected.
 * The steppers still clamp and still write back — also like Date's. */
document.getElementById("compare-date").addEventListener("change", (e) => {
  if (!e.target.value) return;
  state.compareFixed = e.target.value;
  state.compareYears = 0;
  // The select is a different element, so writing it is safe. The INPUT is
  // deliberately not touched — not even through syncCompareUi — which is the
  // whole of the fix and exactly what the Date field does.
  document.getElementById("compare-select").value = "custom";
  refreshTimedLayers({ hold: true });
});

/* The comparison's own steppers — the same calendar arithmetic as the Date
 * row, through the same function. Stepping pins, for the same reason typing
 * does: "10 years ago, minus a month" is not an offset any more. */
document.getElementById("compare-steps").addEventListener("click", (e) => {
  const step = e.target.getAttribute?.("data-cstep");
  if (!step) return;
  const from = compareDate();
  if (!from) return;
  const next = clampUiDate(stepCalendar(from, step));
  if (next === from) return;
  state.compareFixed = next;
  state.compareYears = 0;
  syncCompareUi();
  refreshTimedLayers({ hold: true });
});

document.getElementById("compare-mode").addEventListener("change", (e) => {
  state.compareMode = e.target.value;
  updateDeltaHint();
  refreshTimedLayers({ hold: true });
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
  refreshTimedLayers({ hold: true });
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

// Base-globe mode: persisted; "auto" is the default and needs no storage
const baseModeSel = document.getElementById("base-mode");
try {
  const savedBase = localStorage.getItem("baseMode");
  if (savedBase) baseModeSel.value = savedBase;
} catch { /* private mode */ }
baseModeSel.addEventListener("change", () => {
  try { localStorage.setItem("baseMode", baseModeSel.value); } catch { /* ok */ }
  updateBaseAppearance();
});

// Place names: persisted the same way. Default is ON — without them the globe
// is a pretty abstraction, and "where is that warm water" has no answer.
const placesSel = document.getElementById("places-mode");
try {
  const saved = localStorage.getItem("placesMode");
  if (saved) placesSel.value = saved;
} catch { /* private mode */ }
placesSel.addEventListener("change", () => {
  try { localStorage.setItem("placesMode", placesSel.value); } catch { /* ok */ }
  updatePlaces();
});
updatePlaces();

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
  if (comparing()) {
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
        `${Number(end.slice(0, 4)) - (Number(state.date.slice(0, 4)) -
           Number(cmp.slice(0, 4)))}-${end.slice(5)}).`);
    }
  }

  if (comparing()) {
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
  if (typeof tideLive !== "undefined" && tideLive.on) {
    panel.appendChild(tideLegendEl());
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
    } else if (e.cfg.classmap || e.cfg.colormap || e.cfg.legend) {
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
  ["toggle-tidelive", "Tide (live)"],
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
  host.classList.toggle("hidden", chips.length === 0 && !playback.playing);

  /* While playback runs, the date goes ON THE GLOBE. The picture is 55% of a
   * phone's screen and the Play panel is a scroll and a tab away, so a viewer
   * watching the animation would otherwise have no way to tell WHICH date is on
   * screen without looking away from it. Not a layer, so no × and no reveal —
   * it is a read-out that happens to live in the chip row. */
  if (playback.playing && playback.frames.length) {
    const el = document.createElement("div");
    el.className = "chip chip-play";
    const label = document.createElement("span");
    label.className = "chip-label";
    label.textContent = `▶ ${playback.frames[playback.i]}`;
    label.title = `Playback: frame ${playback.i + 1} of ${playback.frames.length}`;
    el.appendChild(label);
    host.appendChild(el);
  }

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
      "--chips-h", host.childElementCount ? `${host.offsetHeight + 8}px` : "0px");
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
    // An unbounded edge ("[40.00,+INF)") is collapsed to its finite bound so
    // every consumer can do arithmetic — but REMEMBERED, because a bin with no
    // far edge has no honest midpoint (see the caps in getValueLut).
    const openLo = !Number.isFinite(lo), openHi = !Number.isFinite(hi);
    if (openLo) lo = hi;
    if (openHi) hi = lo;
    entries.push({ rgb, lo, hi, openLo, openHi });
  }
  entries.sort((a, b) => a.lo - b.lo);
  return { units, entries };
}

/* ------------------------------------------------- classification colormaps
 *
 * Most GIBS colormaps are continuous: `value="[lo,hi)"` per colour, and a
 * pixel inverts to a NUMBER. A handful are classifications instead — OPERA's
 * vegetation-disturbance products carry `sourceValue="N"` plus a
 * `<Legend type="classification">` whose entries name each class ("Confirmed
 * >= 50%"). Those need their own parser (the continuous one matches nothing
 * here), their own legend (labelled swatches, not a gradient bar), and their
 * own probe read-out (a label, not a formatted float). Layers declare them as
 * `classmap:` instead of `colormap:`. */
const classCache = new Map();
function getClassEntries(url) {
  if (!classCache.has(url)) {
    classCache.set(url, fetch(url).then((r) => r.text()).then(parseClassEntries).catch(() => null));
  }
  return classCache.get(url);
}

function parseClassEntries(xml) {
  const classes = [];
  const re = /<LegendEntry\s+rgb="(\d+),(\d+),(\d+)"\s+tooltip="([^"]*)"\s+id="([^"]+)"/g;
  let m;
  while ((m = re.exec(xml))) {
    const label = m[4].replace(/&lt;/g, "<").replace(/&gt;/g, ">").replace(/&amp;/g, "&");
    // "No Data" is the transparent fill, not a class the user can click on.
    if (/^no data$/i.test(label)) continue;
    classes.push({ rgb: [+m[1], +m[2], +m[3]], label, code: m[5] });
  }
  return classes.length ? { classes } : null;
}

/* rgb → class label, for the probe and the pixel inspector. Keyed on the exact
 * packed colour: classification tiles are nearest-neighbour resampled, so the
 * colours arrive unblended. */
const classLutCache = new Map();
function getClassLut(url) {
  if (!classLutCache.has(url)) {
    classLutCache.set(url, getClassEntries(url).then((cm) => {
      if (!cm) return null;
      const lut = new Map();
      for (const c of cm.classes) lut.set((c.rgb[0] << 16) | (c.rgb[1] << 8) | c.rgb[2], c.label);
      return lut;
    }).catch(() => null));
  }
  return classLutCache.get(url);
}

async function probeClassPixel(cfg, date, z, x, y, px, py, lut) {
  const img = await fetchProbeTile(cfg, date, z, x, y);
  if (!img) return null;
  probeCtx.clearRect(0, 0, 512, 512);
  probeCtx.drawImage(img, 0, 0);
  const d = probeCtx.getImageData(px, py, 1, 1).data;
  if (d[3] === 0) return null;
  return lut.get((d[0] << 16) | (d[1] << 8) | d[2]) ?? null;
}

/* rgb → representative value map (+ units), from any GIBS colormap. Generalises
 * the SST LUT so the delta tool works for any continuous colormapped layer.
 *
 * `caps` marks the CATCH-ALL bins, keyed by the numeric value the lut hands
 * out for them. Many GIBS palettes pad their ends with one huge bucket — SMAP
 * salinity's scale is 30–40 PSU in 0.04-wide steps, but its first entry is
 * [0,30) and its last [40,+INF). The midpoint of a catch-all is an invented
 * number: printing "15.0 PSU" for a Baltic pixel (true value ~7) puts a wrong
 * measurement on screen next to right ones, which is how it was actually
 * noticed. A bin is a cap if its edge is unbounded or if it is an end bin an
 * order of magnitude wider than the palette's typical step; the probe then
 * says "< 30" / "≥ 40" instead of a midpoint. */
const valueLutCache = new Map();
function getValueLut(url) {
  if (!valueLutCache.has(url)) {
    valueLutCache.set(url, getColormapEntries(url).then((cm) => {
      if (!cm) return null;
      const lut = new Map();
      for (const e of cm.entries) lut.set((e.rgb[0] << 16) | (e.rgb[1] << 8) | e.rgb[2], (e.lo + e.hi) / 2);
      const caps = new Map();
      if (cm.entries.length > 2) {
        const widths = cm.entries.map((e) => e.hi - e.lo).filter((w) => w > 0).sort((a, b) => a - b);
        const med = widths[Math.floor(widths.length / 2)] || 0;
        const wide = (e) => med > 0 && e.hi - e.lo > 10 * med;
        const first = cm.entries[0], last = cm.entries[cm.entries.length - 1];
        if (first.openLo || wide(first)) caps.set((first.lo + first.hi) / 2, { sign: "<", bound: first.hi });
        if (last.openHi || wide(last)) caps.set((last.lo + last.hi) / 2, { sign: "≥", bound: last.lo });
      }
      return { units: cm.units, lut, caps };
    }).catch(() => null));
  }
  return valueLutCache.get(url);
}

function fmtVal(v) {
  // A missing number prints as a dash, it does not throw. Every read-out in
  // the app funnels through here, and the pixel card builds its entire body in
  // one pass — so one null field in one upstream response used to take down
  // the whole card, which then sat on "Reading this point…" forever. Formatting
  // is the wrong layer to enforce presence: a caller that cares whether a value
  // exists checks before it asks for a string.
  if (v == null || !Number.isFinite(Number(v))) return "–";
  v = Number(v);
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
  if (cfg.classmap) {
    getClassEntries(cfg.classmap).then((cm) => {
      if (cm) buildClassLegend(div, cm, cfg);
      else fallback();
    });
  } else if (cfg.colormap) {
    getColormapEntries(cfg.colormap).then((cm) => {
      if (cm && cm.entries.length >= 2) buildLegendBar(div, cm);
      else fallback();
    });
  } else {
    fallback();
  }
  return div;
}

/* Classification legend: one labelled swatch per class. A gradient bar would
 * imply an ordering between "provisional" and "finished" that doesn't exist. */
function buildClassLegend(container, cm, cfg) {
  const list = document.createElement("div");
  list.className = "legend-classes";
  for (const c of cm.classes) {
    const row = document.createElement("div");
    row.className = "legend-class";
    row.innerHTML =
      `<span class="legend-swatch" style="background:rgb(${c.rgb.join(",")})"></span>` +
      `<span class="legend-class-label">${c.label}</span>`;
    list.appendChild(row);
  }
  container.appendChild(list);
  if (cfg?.classNote) {
    container.insertAdjacentHTML("beforeend", `<div class="legend-note">${cfg.classNote}</div>`);
  }
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
  // Cap-aware end labels (same convention as the probe, CLAUDE.md Part 2):
  // a palette whose end bin is unbounded or >10x the median bin width
  // SATURATES there — 33-degree water renders as 32, and printing a bare
  // "32.0" presents the palette's ceiling as the ocean's ("hotter seas
  // these days", reported 2026-08-06). Say ">= 32" like we mean it.
  const widths = cm.entries.map((e) => e.hi - e.lo).filter((w) => isFinite(w)).sort((a, b) => a - b);
  const medW = widths[Math.floor(widths.length / 2)] || 1;
  const first = cm.entries[0], last = cm.entries[n - 1];
  const loLbl = (!isFinite(first.lo) || first.hi - first.lo > 10 * medW)
    ? `\u2264 ${fmtVal(first.hi)}` : fmtVal(first.lo);
  const hiLbl = (!isFinite(last.hi) || last.hi - last.lo > 10 * medW)
    ? `\u2265 ${fmtVal(last.lo)}` : fmtVal(last.hi);
  range.innerHTML = `<span>${loLbl}</span><span>${cm.units}</span><span>${hiLbl}</span>`;
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
  if (cfg.classGrid) {
    // Same swatch legend as the classification rasters, fed from the grid file
    // instead of a GIBS colormap — one shape for "the value is a category".
    loadGrid(cfg).then((g) => {
      if (!g?.classes) return;
      buildClassLegend(div, { classes: g.classes }, cfg);
    });
    return div;
  }
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
/* `replace: true` supersedes a toast already on screen under the same key
 * instead of being suppressed by it. The de-dupe key stops the SAME message
 * appearing twice; it must not stop a CORRECTION. (Used by maybeArchiveToast,
 * which speaks immediately from the typed archive end and then again, with a
 * different story, once GIBS has told it what is really served.) */
function showToast(html, { key = html, timeout = 8000, replace = false } = {}) {
  const host = document.getElementById("toast-host");
  if (!host) return;
  if (activeToastKeys.has(key)) {
    if (!replace) return;
    const olds = [...host.querySelectorAll(".toast")]
      .filter((t) => t.dataset.toastKey === key);
    // Replacing a message with the same message would restart the entry
    // animation on every date-selector keystroke. Only a CHANGED story earns a
    // new toast; an unchanged one is left alone, still counting down.
    if (olds.some((t) => t.querySelector(".toast-body")?.innerHTML === html)) return;
    for (const old of olds) old.remove();
    activeToastKeys.delete(key);
  }
  activeToastKeys.add(key);
  const el = document.createElement("div");
  el.className = "toast";
  el.dataset.toastKey = key;
  el.setAttribute("role", "alert");
  el.innerHTML = `<span class="toast-ico">📅</span><div class="toast-body">${html}</div>` +
    `<button class="toast-close" title="Dismiss" aria-label="Dismiss">×</button>`;
  const dismiss = () => {
    // Release the de-dupe key FIRST, unconditionally. It used to be released
    // only on the path that reaches the end of this function, so a toast whose
    // element left the DOM by any other route (a re-render of the host, a
    // second dismiss racing the first) stranded its key in the set forever —
    // and a stranded key silently suppresses that message for the rest of the
    // session. The key exists to stop duplicates on screen, not to remember.
    activeToastKeys.delete(key);
    if (!el.isConnected) return;
    el.classList.add("toast-out");
    el.addEventListener("animationend", () => el.remove(), { once: true });
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
    if (cfg.monthlyGrid) return null;      // month-aware — its own toast (maybeMonthlyGridToast)
    if (cfg.grid) {
      if (cfg.classGrid) {
        // Each categorical grid is dateless for its OWN reason and must say
        // which: the drivers map attributes a 25-year record, the AMOC-eval
        // mask is an experiment's fixed geometry. The sentence below was
        // written for the first one and would have been simply false for the
        // second, so a layer may carry its own `datelessNote`.
        if (cfg.datelessNote) return cfg.datelessNote;
        // Not a climatology and not a snapshot: one attribution computed over
        // the whole 2001–2025 record, so no single date owns it.
        return `<strong>${cfg.title}</strong> attributes the <strong>whole ` +
          `2001–2025 record</strong> in one map — the dominant driver per cell — so the ` +
          `<strong>date selector doesn't change it</strong>. For what is being lost ` +
          `right now, use the 30 m disturbance alerts.`;
      }
      if (cfg.id === "tides") {
        // neither a climatology nor a snapshot: a fixed harmonic analysis
        return `<strong>${cfg.title}</strong> is a fixed harmonic analysis (constants ` +
          `fit to 1992–2019 altimetry), so the <strong>date selector doesn't change ` +
          `it</strong>. The moving tide is animated in the <strong>Tides tab</strong>.`;
      }
      // most grids are multi-decade climatologies; a snapshot grid (a single
      // recent month, like the Argo 300 m anomaly) says what it actually is
      return cfg.snapshotGrid
        ? `<strong>${cfg.title}</strong> is a single recent-month snapshot, so the ` +
          `<strong>date selector doesn't change it</strong> (refreshed with the data pipeline).`
        : `<strong>${cfg.title}</strong> is a long-term climatology (a multi-decade ` +
          `average), so it has no per-date data — the <strong>date selector doesn't change it</strong>.`;
    }
    if (cfg.datelessNote) return cfg.datelessNote;   // says WHICH kind of fixed thing it is
    return `<strong>${cfg.title}</strong> is a fixed composite, so the ` +
      `<strong>date selector doesn't change it</strong>.`;
  }
  const DATA_MSG = {
    gbif: "<strong>Biodiversity occurrences (GBIF)</strong> are all-time, so the " +
      "<strong>date selector doesn't change this layer</strong>. Rare taxa (e.g. humans) may " +
      "be too sparse to see when zoomed out — that's expected, not a date problem.",
    climatetrace: null,   // yearly, not dateless — its own toast (climateTraceToast)
    argo: "<strong>Argo floats</strong> shows the fleet's latest positions (last ~10 days), so the " +
      "<strong>date selector doesn't change it</strong>.",
    stations: "<strong>Monitoring stations</strong> are fixed sites, so the " +
      "<strong>date selector doesn't change this layer</strong>.",
    glaciers: "<strong>Glaciers (RGI v7)</strong> is a single inventory (~year 2000), so the " +
      "<strong>date selector doesn't change it</strong>. Its melt-rate colouring is a 2000–2020 average.",
    tidelive: "<strong>Tide height (live)</strong> runs on its <strong>own clock</strong> " +
      "(real time by default \u2014 speed and pause in the Tides tab), so the date selector " +
      "doesn't change it. \u263e and \u2600 mark where moon and sun are overhead.",
    "sst-ensemble": null,   // the ensemble IS date-driven
  };
  return DATA_MSG[id] || null;
}

function maybeDatelessToast(id) {
  const html = datelessToast(id);
  if (html) showToast(html, { key: id });
}

/* Month-keyed grids (GLORYS) explain their month semantics on enable, with the
 * actual month rendered — mirroring the year-aware Climate TRACE toast. */
function maybeMonthlyGridToast(cfg) {
  if (!cfg?.monthlyGrid) return;
  loadGrid(cfg).then((g) => {
    const ms = gridMonths(g);
    if (!ms) return;
    const showM = resolveGridMonth(g);
    const lo = ms[0], hi = ms[ms.length - 1];
    if (cfg.forecastGrid) {
      const initS = g.init ? ` from the GFS run <strong>${g.init}</strong>` : "";
      showToast(`<strong>${cfg.title}</strong> is a <strong>forecast</strong>${initS}: ` +
        `the date selector now runs into the <strong>future</strong> — step forward day by day ` +
        `up to <strong>${hi}</strong>. Showing <strong>${showM}</strong>. Observation layers ` +
        `can't follow; they hold at their latest real date.`, { key: cfg.id });
      return;
    }
    const want = state.date.slice(0, 7);
    const note = ms.length === 1
      ? ` — only this month is baked so far (the data pipeline's <code>glorys</code> step backfills more)`
      : want !== showM
        ? ` (nearest baked month to your ${want}; available ${lo} → ${hi})`
        : ` — set the date's <em>month</em> anywhere in ${lo} → ${hi} to browse`;
    showToast(`<strong>${cfg.title}</strong> is a <strong>monthly-mean</strong> reanalysis: ` +
      `the day doesn't matter, but the <strong>month does</strong>. Showing ` +
      `<strong>${showM}</strong>${note}.`, { key: cfg.id });
  });
}

/* Annual products are not dateless and not monthly: the year drives them and
 * nothing else does — the same trap Climate TRACE has, one rung coarser. Say
 * so on enable, or the day/month buttons look broken. Fires straight away from
 * what is known, and again with `replace` once the measured domain lands, so
 * the span of years it quotes ends up being the archive's rather than a guess. */
function maybeAnnualToast(cfg, { replace = false } = {}) {
  if (!cfg?.annual) return;
  const shown = String(annualYearOf(cfg, gibsTime(cfg, state.date)));
  const lo = String(annualYearOf(cfg, cfg.start));
  const hi = String(annualYearOf(cfg, cfg.lastServed || cfg.endTime || defaultDate()));
  const want = state.date.slice(0, 4);
  const note = want !== shown
    ? ` (nearest available to your ${want}; the product covers ${lo}–${hi})`
    : ` — set the date's <em>year</em> anywhere in ${lo}–${hi} to switch years`;
  showToast(`<strong>${cfg.title}</strong> is an <strong>annual</strong> summary: ` +
    `the day and month don't matter, but the <strong>year does</strong>. ` +
    `Showing <strong>${shown}</strong>${note}.`, { key: `annual-${cfg.id}`, replace });
}

/* The date you asked for and the date GIBS can give you come apart for three
 * quite different reasons, and the user needs them told apart:
 *
 *   1. CLOSED — the archive stopped and nothing newer will ever exist (CERES
 *      2018-10, GRACE 2022-07, sea-surface height 2019-01, sea ice 2025-09).
 *   2. LAGGING — the archive is live but behind. NDVI's newest monthly
 *      composite was 62 days old on 2026-08-03; tomorrow it catches up a step.
 *   3. A HOLE — the archive skips the requested date specifically. NDVI has no
 *      2025-04 at all; SMAP salinity has no 2024; VIIRS lost 11–15 July 2026.
 *
 * From the outside all three looked identical: a blank globe, a legend, and a
 * probe saying "no data". This says which one happened, and always names the
 * date actually on screen.
 *
 * It speaks TWICE by design: immediately on enable from what is already known
 * (the typed archive end), so a clamped layer is never silently old while a
 * metadata fetch is in flight — and again, with `replace`, once the measured
 * domain lands and the story may have changed from "the archive ended" to "this
 * date has a hole in it". If GIBS never answers, the first message stands. */
function maybeArchiveToast(cfg, { replace = false } = {}) {
  if (!cfg?.timed || cfg.annual) return;      // annual has its own, clearer toast
  {
    const e = state.layers[cfg.id];
    if (!e || !(e.layer || e.suppressed)) return;   // switched off while we asked
    const shown = gibsTime(cfg, state.date);
    // What you asked for, WITHOUT the archive-end clamp — otherwise a layer
    // whose tiles stopped in 2018 looks like it got exactly the date it wanted.
    const asked = gibsTimeStatic(cfg, state.date, { clampEnd: false });
    if (domainMs(shown) === domainMs(asked)) return;    // you got what you asked for

    // Print at the product's own granularity: a month for a monthly composite,
    // a timestamp only where the half-hour actually distinguishes two frames.
    const fmt = (v) => (cfg.monthly ? String(v).slice(0, 7)
      : String(v).length > 10 ? String(v).replace("T", " ").replace(/:\d\dZ$/, " UTC")
        : String(v).slice(0, 10));
    // Read the newest served instant off the DOMAIN, not off cfg — the domain
    // is the measurement, cfg.lastServed only a copy of it, and a copy is one
    // more thing that can go stale.
    const dom = gibsDomains.get(cfg.id);
    const newest = (dom && dom[dom.length - 1].e) || cfg.lastServed || cfg.endTime;
    const atEdge = newest && domainMs(shown) >= domainMs(newest);
    const behind = cfg.monthly
      ? (() => {
        const n = (Number(asked.slice(0, 4)) - Number(shown.slice(0, 4))) * 12 +
          (Number(asked.slice(5, 7)) - Number(shown.slice(5, 7)));
        return `${n} month${n === 1 ? "" : "s"}`;
      })()
      : (() => {
        const n = Math.round((domainMs(asked) - domainMs(shown)) / 864e5);
        return n >= 1 ? `${n} day${n === 1 ? "" : "s"}` : "under a day";
      })();

    let html;
    if (cfg.endTime && atEdge) {
      html = `<strong>${cfg.title}</strong>: its tile archive ends ` +
        `<strong>${fmt(newest)}</strong> and nothing newer will be published, so ` +
        `you're seeing <strong>${fmt(shown)}</strong> — the last served date — ` +
        `not ${fmt(asked)}. Set the date on or before ${fmt(newest)} to browse ` +
        `the archive.`;
    } else if (atEdge) {
      html = `<strong>${cfg.title}</strong>: NASA hasn't published ` +
        `<strong>${fmt(asked)}</strong> yet — this product currently runs about ` +
        `${behind} behind. You're seeing <strong>${fmt(shown)}</strong>, the ` +
        `newest one served.`;
    } else {
      html = `<strong>${cfg.title}</strong>: GIBS has no tiles for ` +
        `<strong>${fmt(asked)}</strong> — a gap in the archive, not an error. ` +
        `You're seeing <strong>${fmt(shown)}</strong>, the newest date before it ` +
        `(${behind} earlier).`;
    }
    showToast(html, { key: `clamp-${cfg.id}`, replace });
  }
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
  "dist-alert": { rec: "this map: 2023-01 → present (2024 has a few gap dates near the start of the year — blank means no tile, not no disturbance)", int: "updated every few days as Landsat/Sentinel-2 pass over; each map is the running status of the CURRENT year", sp: "30 m — one of the fine-tier layers",
    sum: "Where vegetation has been lost, at the scale of a single clearing. " +
         "OPERA compares every new Landsat/Sentinel-2 image against that " +
         "pixel's own recent history, and flags the drop: first detection, " +
         "provisional, then confirmed once a second clear image agrees — and " +
         "separately whether under or over half the cover went. Deforestation, " +
         "fire scars, logging roads and storm damage all appear; the product " +
         "sees the loss, not the cause. Zoom right in: at global view a 30 m " +
         "alert is far smaller than a screen pixel." },
  "dist-ann": { rec: "2023, 2024 and 2025 (one map per year, the annual summary)", int: "yearly — the date's YEAR picks the map; day and month are ignored", sp: "30 m",
    sum: "The year's confirmed vegetation loss in one map, the settled version " +
         "of the alert layer: everything that was still provisional has been " +
         "resolved. Use this to compare whole years — how much of the Amazon " +
         "arc, the Congo basin or Southeast Asia changed in 2023 versus 2024 — " +
         "and the alert layer to watch the current year as it happens." },
  "hls-s30": { rec: "this map: 2015-11 → present (Sentinel-2A launched 2015-06; tiles run ~1 week behind)", int: "daily — the swaths flown that day; each place is revisited every 2–5 days, and clouds hide most passes", sp: "30 m (Sentinel-2's native 10 m resampled by HLS to Landsat's grid)",
    sum: "What Sentinel-2 saw on the chosen day, at the scale of a field or a " +
         "city block. HLS (Harmonized Landsat–Sentinel) corrects the raw image " +
         "for the atmosphere and view angle so that it can be compared, pixel " +
         "for pixel, with Landsat's. It is a SWATH product: on any one date only " +
         "the strips the satellite flew are painted, and a cloudy pass is a " +
         "cloudy image — blank means no pass, not no data. Loads only below " +
         "500 km; step the date to find a clear day." },
  "hls-l30": { rec: "this map: 2013-03 → present (Landsat 8 launched 2013-02, Landsat 9 2021-09)", int: "daily — the swaths flown that day; each place every 8 days with the two satellites", sp: "30 m (Landsat's native resolution)",
    sum: "The same Harmonized product built from Landsat 8 and 9 instead of " +
         "Sentinel-2: fewer passes, but the sensor lineage that reaches back to " +
         "1984 (the historic WELD layer below is its ancestor). Swaths for the " +
         "chosen day, blank where nothing flew; loads only below 500 km." },
  "sar-s1": { rec: "this map: 2025-01 → present (Sentinel-1 has flown since 2014, but only OPERA's terrain-corrected version is served as tiles)", int: "daily — the swaths flown that day; each place every 6–12 days", sp: "30 m",
    sum: "Radar instead of light: Sentinel-1 sends a C-band microwave pulse and " +
         "maps how much comes back. Water is smooth and returns nothing (dark), " +
         "cities and rough ground return a lot (bright), vegetation sits between " +
         "— and it works through cloud and at night, which optical imagery " +
         "cannot. This is the sensor behind flood maps, sea-ice edges, ship " +
         "detection and deformation monitoring. False colour: the two " +
         "polarisations are painted as separate channels. Swaths for the chosen " +
         "day; loads only below 500 km." },
  nisar: { rec: "this map: 2025-10 → present (launched 2025-07; PROVISIONAL products while calibration continues)", int: "daily — the swaths flown that day; each place every 12 days", sp: "15 m — the finest layer in this app",
    sum: "The NASA–ISRO radar: L-band, a wavelength four times longer than " +
         "Sentinel-1's, so it penetrates canopy and sees the ground under " +
         "forests, soil moisture, and the slow deformation of ice sheets and " +
         "fault lines. Launched July 2025 and now in public release; treat the " +
         "colours as provisional while the calibration settles. Swaths for the " +
         "chosen day; loads only below 300 km." },
  "water-hls": { rec: "this map: 2016-01 → 2018-08, then 2023-01 → present (the gap is the product's own reprocessing history)", int: "daily — from each clear Landsat/Sentinel-2 pass", sp: "30 m",
    sum: "Where water stands on the surface on the chosen day, pixel by pixel: " +
         "open water, partial water (a bank, a marsh, a channel narrower than " +
         "30 m), snow and ice, and cloud — which is unobserved, not dry. Step " +
         "through a flood and watch a river's footprint spread and recede; " +
         "compare a reservoir's outline in a wet year and a dry one. Optical, " +
         "so blind under cloud — that is what the Sentinel-1 version is for." },
  "water-s1": { rec: "this map: 2023-12 → present", int: "daily — from each Sentinel-1 pass, every 6–12 days per place", sp: "30 m",
    sum: "The flood layer: surface water from radar, which sees through the " +
         "very clouds that bring the flood. Open water, and inundated vegetation " +
         "— flooded forest or crops the radar picks out beneath the canopy. Two " +
         "masks are honest about what the method cannot judge: ground too high " +
         "above the nearest river to flood (HAND), and mountain slopes hidden in " +
         "radar shadow." },
  elevation: { rec: "fixed — one model from stereo images taken 2000–2013 (ASTER GDEM v3), ignores the date selector", int: "single model, no time axis", sp: "30 m, 83°N–83°S",
    sum: "Height above sea level, from a million stereo image pairs taken by " +
         "the ASTER instrument on Terra. The palette runs blue-green lowlands " +
         "through browns to white peaks; hover to read the height in metres at " +
         "30 m resolution. Terrain is the boundary condition under half the " +
         "other layers — where rain falls, where rivers go, where cities sit, " +
         "how far a rising sea reaches. Sea level itself is transparent, so the " +
         "ocean shows the base map." },
  builtup: { rec: "fixed — one map from 2010 Landsat imagery (HBASE v1), ignores the date selector", int: "single map", sp: "30 m",
    sum: "Every human structure the 2010 Landsat record could see: buildings, " +
         "roads, paved and built ground, mapped at 30 m over the whole planet " +
         "by NASA's SEDAC. It is a footprint, not a population count — a " +
         "sprawling suburb and a dense city block both read as built-up. Zoom " +
         "in on any city to see its shape; compare with night lights for " +
         "where that footprint is actually lit." },
  impervious: { rec: "fixed — one map from 2010 Landsat imagery (GMIS v1), ignores the date selector", int: "single map", sp: "30 m, in 10-percent bins",
    sum: "How sealed the ground is: the share of each 30 m pixel covered by " +
         "roofs, asphalt and concrete. Sealed ground shed rain instead of " +
         "absorbing it (flash floods) and stores heat instead of evaporating " +
         "water (urban heat islands), so this is the map that turns a built-up " +
         "footprint into a physical property. The probe reads the bin, e.g. " +
         "31–40 %." },
  weld: { rec: "three spans only: 1984–86, 1989–91 and 1999–2001 (one cloud-free mosaic per year); type a date — the steppers stop at 2000", int: "yearly — the date's YEAR picks the mosaic; other years show the nearest earlier one", sp: "30 m",
    sum: "The 1980s and 1990s at 30 m: the Landsat 4–7 record composited into " +
         "one cloud-free mosaic per year by the WELD project. Put it beside the " +
         "Sentinel-2 layer on today's date and forty years of change sit under " +
         "the cursor — a reservoir filled, a city grown, a forest gone. Only " +
         "three spans were produced, so the annual toast names the year " +
         "actually showing." },
  drivers: { rec: "2001–2025, attributed in one map (v1.3)", int: "not dated — re-baked when WRI publishes a new version", sp: "1 km source, binned here to 0.25° by dominant class",
    sum: "Why the forest went. WRI and Google DeepMind trained a classifier on " +
         "tens of thousands of hand-labelled sites to name the dominant cause " +
         "of tree-cover loss at every kilometre on Earth: permanent " +
         "agriculture, hard commodities (mining and energy), shifting " +
         "cultivation, logging, wildfire, settlements and infrastructure, or " +
         "other natural disturbance. The pattern is the point — the Amazon arc " +
         "and West Africa are agriculture, boreal Canada and Siberia are fire, " +
         "the US southeast and Scandinavia are logging, and the difference " +
         "between them is the difference between forest that is gone and " +
         "forest that will grow back. This is the companion to the OPERA alert " +
         "layers, which see the loss at 30 m but say nothing about its cause." },
  "amoc-eval": { rec: "fixed — the geometry of the 1982-01 → 2024-12 tensor the model is trained on, re-baked when the window or the corridor recipe changes", int: "not dated — one fixed geometry, not a measurement", sp: "0.25° (~27 km here) — 84,405 ocean pixels, of which 29,627 are scored",
    sum: "The only layer here that shows a MODEL rather than the world. The " +
         "forecaster works pixel by pixel: each ocean cell is compressed to a " +
         "64-number embedding, and the model predicts every cell's next month " +
         "from its own recent history and — since E-022 — its neighbours', " +
         "then feeds the prediction back in and steps again. Blue is every " +
         "pixel that gets advanced that way, all 84,405 of them, because a " +
         "model that reads its neighbours cannot roll a small region without " +
         "the region around it. Orange is the AMOC corridor the skill score is " +
         "actually read from — the fastest quarter of the ocean by mean " +
         "current speed, dilated two cells, which the data itself picks out as " +
         "the Loop Current, the Gulf Stream, the North Atlantic Current and " +
         "the return flow. Red is the RAPID array's 26.5°N section, where the " +
         "transport this whole project tries to predict is measured." },
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
  "currents": { rec: "recent months, baked by the data pipeline — the date's MONTH picks the map; GLORYS reanalysis covers 1993 → near-present", int: "monthly means (day of month doesn't matter)", sp: "1/12° model, shipped at 1°",
    sum: "How fast the surface ocean moves: the monthly-mean current speed from " +
         "the GLORYS ocean reanalysis (which assimilates Argo, altimetry and " +
         "SST). The Gulf Stream, Kuroshio and Antarctic Circumpolar Current " +
         "leap out — the plumbing that carries the heat the AMOC story is " +
         "about. Click any ocean point for speed AND direction." },
  "mld": { rec: "recent months, baked by the data pipeline — the date's MONTH picks the map; deepest in late winter", int: "monthly means (day of month doesn't matter)", sp: "1/12° model, shipped at 1°",
    sum: "How deep wind and cooling stir the surface ocean. Watch the subpolar " +
         "North Atlantic: in late winter the mixed layer there can reach " +
         "hundreds of metres to a kilometre — that deep convection is where " +
         "AMOC deep water is actually made, so its shrinking is an early " +
         "warning signal. In summer it shoals to tens of metres everywhere." },
  "gfs-temp": { rec: "the NEXT 10 days from the latest complete GFS run (see the toast for the init time) — the only layer that looks forward", int: "one frame per day, same hour each day; refreshed by the data pipeline", sp: "0.25° model, shipped at 1°",
    sum: "Where the atmosphere is heading: 2 m air temperature from NOAA's GFS " +
         "model, one frame per forecast day. Switch it on and the date selector " +
         "unlocks the future — step forward to watch heat domes build and cold " +
         "fronts sweep through. A physics baseline for the AI forecasts " +
         "(WeatherNext & co.) that may join it later." },
  "gfs-precip": { rec: "the NEXT ~9 full days from the latest complete GFS run — forecast, not observation", int: "24-h totals per forecast day (sum of the model's 6-h buckets); dry days transparent", sp: "0.25° model, shipped at 1°",
    sum: "Where it is FORECAST to rain: daily precipitation totals from NOAA's " +
         "GFS. Step the date forward to watch atmospheric rivers, monsoon " +
         "bursts and cyclones arrive days ahead — then flip to the IMERG " +
         "layer on past dates to see how the forecast did." },
  "grep-spread-cur": { rec: "1993–2024 — one fixed field, the time mean of the member spread, not a dated measurement", int: "mean over 384 monthly fields of (max − min) across the three members", sp: "0.25° native (the family-3/4 NA grid), 84,405 ocean cells",
    sum: "How much three reanalyses DISAGREE about the same ocean. GREP " +
         "estimates 1993–2024 three times — CGLORS, GLORYS2V4, ORAS5 — from " +
         "the same satellites and the same Argo floats, with different models " +
         "and different assimilation. Typical disagreement is 0.077 m/s, but " +
         "in the Gulf Stream and its rings it passes 1 m/s, comparable to the " +
         "current itself. This is the part of the ocean the observations do " +
         "not pin down, and nothing downstream can know a cell better than " +
         "its own inputs do." },
  "grep-spread-mld": { rec: "1993–2024 — one fixed field, the time mean of the member spread, not a dated measurement", int: "mean over 384 monthly fields of (max − min) across the three members", sp: "0.25° native (the family-3/4 NA grid), 84,405 ocean cells",
    sum: "The same three reanalyses on mixed-layer depth, where they agree " +
         "far less. The median cell differs by 14 m, but the Labrador and " +
         "Irminger Seas reach 716 m — the deep-convection sites that set the " +
         "dense water the overturning carries south. Where the AMOC is made " +
         "is precisely where the observing system constrains us least." },
  "argo-t300": { rec: "one recent month (see legend) vs the 2004–2018 mean for that same calendar month — not one date", int: "monthly snapshot, refreshed by the data pipeline", sp: "1° (Argo objective mapping)",
    sum: "Where the ocean is unusually warm or cool 300 m DOWN — measured by the " +
         "Argo float fleet, invisible to every satellite surface map. Subsurface " +
         "marine heatwaves matter because that heat is stored, not radiated away; " +
         "comparing against the same calendar month removes the seasonal cycle, so " +
         "red really means anomalous." },
  "seaice": { rec: "this map: 2012-07 → 2025-09 (last date GIBS serves; the AMSR2 instrument is aging)", int: "daily", sp: "12 km grid",
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
  "tides": { rec: "harmonic constants fit to 1992–2019 satellite altimetry (a fixed analysis, not one date)", int: "one analysis · the Tides tab animates the actual cycle from the same constants", sp: "0.125° source → 1° shown",
    sum: "How far the sea surface rises and falls: twice the summed amplitude of the " +
         "four main tidal constituents (M2, S2, K1, O1) — the range when they peak " +
         "together, roughly a large spring tide. Centimetres in the open ocean, " +
         "metres on wide shelves (Fundy, the Severn, Ungava), and near zero at the " +
         "amphidromic hubs the tide rotates around. From DGFI-TUM's EOT20 model, " +
         "fit to 27 years of satellite altimetry." },
  "oisst-monthly": { rec: "measurements 1981-09 → present · the date's month picks the map", int: "monthly means", sp: "0.25° source → 1° shown",
    sum: "Sea surface temperature as numbers, not paint: NOAA's OISST monthly " +
         "means on a scale that reaches 36 °C. The MUR satellite layer's palette " +
         "saturates at 32° — on it, the 35° Persian Gulf and 32° open tropics look " +
         "identical. Here they separate, and clicking reads the exact value. The " +
         "full record back to 1981 makes Compare-across-decades honest too." },
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
      ${cfg.fine ? `<div class="fine-hint" data-finehint="${cfg.id}" hidden></div>` : ""}
      <div class="alpha-row" data-alpharow="${cfg.id}" ${cfg.on ? "" : "style='display:none'"}>
        <span class="alpha-label">opacity</span>
        <input type="range" min="0" max="100" value="100" data-alpha="${cfg.id}"
               title="Layer opacity — lower it to see the layers underneath"/>
        <button class="alpha-half" data-alphahalf="${cfg.id}"
                title="Toggle 50% — overlay this layer half-transparent on the one below (e.g. SST over ocean currents)">½</button>
        <span class="alpha-val" data-alphaval="${cfg.id}">100%</span>
      </div>
      ${layerTipHtml(cfg.id)}`;
    list.appendChild(div);
    if (cfg.on) addLayer(cfg);
  }
  updateSplitUI();

  list.addEventListener("change", (e) => {
    const id = e.target.getAttribute("data-id");
    if (!id) return;
    const cfg = GIBS_LAYERS.find((l) => l.id === id);
    const row = list.querySelector(`[data-alpharow="${id}"]`);
    if (e.target.checked) {
      addLayer(cfg);
      row.style.display = "";
      maybeDatelessToast(id);
      maybeMonthlyGridToast(cfg);
      maybeArchiveToast(cfg);
      maybeAnnualToast(cfg);
      maybeFineToast(cfg);
      updateFineGates();
    } else {
      removeLayer(id);
      row.style.display = "none";
      updateFineGates();
      if (cfg?.forecastGrid) syncDateMax();   // may pull the date back to today
    }
    updateSplitUI();
  });

  const setAlpha = (id, pct) => {
    const entry = state.layers[id];
    if (entry) {
      entry.alpha = pct / 100;
      if (entry.layer) entry.layer.alpha = entry.alpha;
      if (entry.cmpLayer) entry.cmpLayer.alpha = entry.alpha;
    }
    const val = list.querySelector(`[data-alphaval="${id}"]`);
    if (val) val.textContent = `${pct}%`;
  };

  list.addEventListener("input", (e) => {
    const id = e.target.getAttribute("data-alpha");
    if (!id) return;
    setAlpha(id, Number(e.target.value));
  });

  // ½ toggles 50% ↔ 100%: the quick way to overlay two fields (e.g. SST at
  // half opacity over ocean currents to eyeball their correlation)
  list.addEventListener("click", (e) => {
    const id = e.target.getAttribute?.("data-alphahalf");
    if (!id) return;
    const slider = list.querySelector(`input[data-alpha="${id}"]`);
    const pct = Number(slider.value) === 50 ? 100 : 50;
    slider.value = pct;
    setAlpha(id, pct);
  });

  const dateInput = document.getElementById("layer-date");
  dateInput.value = state.date;
  dateInput.max = uiMaxDate();
  syncCompareUi();
  dateInput.addEventListener("change", () => {
    if (!dateInput.value) return;
    state.date = dateInput.value;
    syncCompareUi();          // an OFFSET comparison moved with it
    applyDateMove();          // coalesced onto the paint clock, never per keystroke
  });

  // Quick date stepping: real calendar arithmetic (−1m from Mar 31 → Feb 28,
  // −1y from Feb 29 → Feb 28), clamped to [layer availability, most recent].
  document.getElementById("date-steps").addEventListener("click", (e) => {
    const step = e.target.getAttribute?.("data-step");
    if (!step) return;
    const next = clampUiDate(step === "today" ? defaultDate()
                                              : stepCalendar(state.date, step));
    if (next === state.date) return;
    state.date = next;
    dateInput.value = next;
    syncCompareUi();          // an OFFSET comparison moved with it
    applyDateMove();          // a held stepper coalesces; one click does not
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
    if (date > uiMaxDate() || date < "2000-01-01") return; // nothing to step to
    state.timeMin = t;
    if (date !== state.date) {
      state.date = date;
      dateInput.value = date;
      scrubApply(() => {
        refreshTimedLayers({ hold: true }); // date change affects every timed layer
        refreshMonthlyGrids();       // crossing midnight can cross a month
      });
    } else {
      // Same date, new half-hour: only sub-daily layers see a different TIME,
      // so don't churn (refetch) the daily/monthly layers on every step.
      // Coalesced on the same paint clock as every other date move — this is
      // the FASTEST stepper in the app, so it is the one a repeated tap can
      // run furthest ahead of the network.
      scrubApply(() => {
        for (const [id, entry] of Object.entries(state.layers)) {
          if (entry.cfg.subDaily && (entry.layer || entry.suppressed)) {
            // Held, like every other date move: a half-hour step is the
            // fastest stepper in the app and blinked the hardest.
            retireLayer(id);
            addLayer(entry.cfg);
            scheduleSweep();
          }
        }
      });
    }
  });
}

/* -------------------------------------------------- tagline scene shortcuts
 * The header tagline's words are one-click scenes: each swaps the active
 * layers for a curated set that shows its theme (the chips make the swap
 * visible and reversible). Ids prefixed "toggle-" are static checkboxes;
 * bare ids are GIBS_LAYERS entries in #layer-list. "forecasts" is different:
 * nothing to draw — it arms the pixel inspector and says what to do next. */
/* ONE layer per scene, deliberately: stacked layers mostly hide each other,
 * and a scene named "sea ice" that also drops a one-off glacier inventory on
 * the globe over-promises. The link text matches exactly what appears. */
const SCENES = {
  satellites: ["viirs-truecolor"],   // daily, follows the date
  // The one sanctioned TWO-layer scene: SST covers ocean, LST covers land —
  // spatially disjoint, so they compose one seamless temperature field
  // rather than hiding each other (the exception that proves the
  // one-visual-field rule).
  temperature: ["sst", "lst"],
  seaice: ["seaice"],                // daily; tiles end 2025-09 (clamped + toast)
  currents: ["currents"],            // monthly GLORYS snapshot
  tides: ["toggle-tidelive"],        // the live harmonic tide, own clock
  floats: ["toggle-argo"],           // the live Argo fleet
  vegetation: ["ndvi"],              // monthly, follows the date
  "forest loss": ["dist-alert"],     // 30 m alerts — needs the flyTo below to be visible at all
  // The other half of the same question. No flyTo: at 0.25° the driver map is
  // a global pattern, and the default view is exactly where you want to read it.
  "why forests fall": ["drivers"],
  emissions: ["toggle-climatetrace"],// yearly, the date's year picks it
};
// A scene whose data lives somewhere specific flies the camera there — sea
// ice is invisible from the default equatorial view.
const SCENE_VIEWS = {
  seaice: { lon: -35, lat: 78, height: 1.15e7 },   // the Arctic fills the view
  // A 30 m alert is invisible from orbit — without a flyTo this scene looks
  // broken. Land on the Amazon "arc of deforestation" (Rondônia/Mato Grosso),
  // the densest, most legible cluster of clearings on the planet.
  "forest loss": { lon: -60, lat: -9, height: 2.2e6 },
};
function sceneBox(id) {
  return id.startsWith("toggle-")
    ? document.getElementById(id)
    : document.querySelector(`#layer-list input[data-id="${id}"]`);
}
function enableScene(key) {
  if (key === "inspect") {
    const box = document.getElementById("toggle-pixel");
    if (box && !box.checked) {
      box.checked = true;
      box.dispatchEvent(new Event("change", { bubbles: true }));
    }
    showToast(`<strong>Everything we know</strong> is armed — click anywhere on the globe ` +
      `for that point's full state, including its <strong>2045–49 climate outlook</strong>.`);
    return;
  }
  const ids = SCENES[key];
  if (!ids) return;
  // swap, don't pile up: a scene is a curated view, and the chips both show
  // what changed and undo it in one click
  for (const c of activeLayerChips()) turnOffLayer(c.box);
  for (const id of ids) {
    const box = sceneBox(id);
    if (box && !box.checked) {
      box.checked = true;
      box.dispatchEvent(new Event("change", { bubbles: true }));
    }
  }
  const v = SCENE_VIEWS[key];
  if (v) {
    viewer.camera.flyTo({
      destination: Cesium.Cartesian3.fromDegrees(v.lon, v.lat, v.height),
      duration: 2.0,
    });
  }
  // The tides scene is the one scene that also switches tabs: its clock,
  // speed control, spring/neap read-out and tapped-point curve live in the
  // Tides tab, and arriving at the water without its controls strands you
  // (the reverse direction — opening the tab enables the layer — already
  // holds, so the two entrances now meet in the same place).
  if (key === "tides") document.getElementById("tab-tides").click();
}
document.querySelectorAll(".tag-link").forEach((b) =>
  b.addEventListener("click", () => enableScene(b.dataset.scene)));

/* First-visit intro guide: open by default; once dismissed, stays dismissed
 * (localStorage) so returning users get their controls back on top. */
(() => {
  const intro = document.getElementById("intro-guide");
  if (!intro) return;
  try {
    if (localStorage.getItem("introClosed") === "1") intro.open = false;
  } catch { /* private mode — stays open, harmless */ }
  intro.addEventListener("toggle", () => {
    try { localStorage.setItem("introClosed", intro.open ? "0" : "1"); } catch { /* ok */ }
  });
})();

/* ------------------------------------------------- resizable sidebar */
/* The panel/globe boundary is draggable (#sidebar-resize). Width lives in the
 * --sidebar-w CSS variable so one drag moves the sidebar, the globe container
 * and the resize strip together; Cesium picks the container change up in its
 * render loop. Charts size themselves from clientWidth, so redraw while
 * dragging. Persisted in localStorage; double-click resets. */
(() => {
  const handle = document.getElementById("sidebar-resize");
  if (!handle) return;
  const DEFAULT_W = 380, MIN_W = 300;
  // Max is structural, not aesthetic: leave just enough globe to click on.
  const maxW = () => Math.max(MIN_W, window.innerWidth - 240);
  const setW = (px) => {
    const w = Math.round(Cesium.Math.clamp(px, MIN_W, maxW()));
    document.documentElement.style.setProperty("--sidebar-w", `${w}px`);
    return w;
  };
  const redraw = () => {
    if (!document.getElementById("panel-temp").classList.contains("hidden")) drawTempChart();
    if (eeiData && !document.getElementById("panel-energy").classList.contains("hidden")) {
      drawEeiChart();
      drawEeiRateChart();
    }
  };
  try {
    const saved = Number(localStorage.getItem("sidebarW"));
    if (saved) setW(saved);
  } catch { /* private mode etc. — default width is fine */ }

  let raf = null;
  handle.addEventListener("pointerdown", (e) => {
    e.preventDefault();
    handle.setPointerCapture(e.pointerId);
    handle.classList.add("dragging");
    const move = (ev) => {
      setW(ev.clientX);
      if (!raf) raf = requestAnimationFrame(() => { raf = null; redraw(); });
    };
    const up = (ev) => {
      handle.removeEventListener("pointermove", move);
      handle.removeEventListener("pointerup", up);
      handle.classList.remove("dragging");
      const w = setW(ev.clientX);
      try { localStorage.setItem("sidebarW", String(w)); } catch { /* ok */ }
      redraw();
      updateSplitUI();          // the swipe divider is positioned in globe px
    };
    handle.addEventListener("pointermove", move);
    handle.addEventListener("pointerup", up);
  });
  handle.addEventListener("dblclick", () => {
    setW(DEFAULT_W);
    try { localStorage.removeItem("sidebarW"); } catch { /* ok */ }
    redraw();
    updateSplitUI();
  });
})();

/* ----------------------------------------------------- point data layers */

const pickCard = document.getElementById("pick-card");
const pointLayers = {}; // kind -> {collection, meta}

// Climate TRACE is annual (2021–2025): the layer shows whichever year the
// date selector points at, clamped to the available range. Month and day are
// ignored — it is a yearly inventory, not a daily field.
let climateTraceLoadedYear = null;
function climateTraceYear(json) {
  const yrs = json.years;
  const want = Number(state.date.slice(0, 4));
  return Cesium.Math.clamp(want, yrs[0], yrs[yrs.length - 1]);
}

// Drop the cached climatetrace collection if it no longer matches the year the
// date selector points at, so the next load() rebuilds it. Covers both a live
// year change and re-enabling after the year moved while it was hidden.
function ensureClimateTraceYear() {
  const json = pointLayers.climatetrace?.__json;
  if (json && climateTraceYear(json) !== climateTraceLoadedYear) {
    viewer.scene.primitives.remove(pointLayers.climatetrace.collection);
    delete pointLayers.climatetrace;
  }
}
async function refreshYearlyLayers() {
  const box = document.getElementById("toggle-climatetrace");
  if (!box || !box.checked) return;
  ensureClimateTraceYear();
  await loadPointLayer("climatetrace");   // no-op if the year was unchanged
}

async function loadPointLayer(kind) {
  if (pointLayers[kind]) {
    pointLayers[kind].collection.show = true;
    return;
  }
  const cfgs = {
    climatetrace: {
      file: "data/climatetrace.json",
      build(json, col) {
        const yr = climateTraceYear(json);
        const assets = json.assets_by_year[String(yr)] || [];
        for (const [lon, lat, mt, name, country, sector] of assets) {
          col.add({
            position: Cesium.Cartesian3.fromDegrees(lon, lat),
            pixelSize: Math.max(4, Math.min(15, 4 + 11 * Math.sqrt(mt / 270))),
            color: Cesium.Color.fromCssColorString("#d95926").withAlpha(0.85),
            outlineColor: Cesium.Color.BLACK.withAlpha(0.6),
            outlineWidth: 1,
            id: {
              kind: "climatetrace",
              html: `<strong>${esc(name)}</strong><br/>${esc(country)} · ${esc(sector)}<br/>` +
                `<b>${mt.toFixed(1)} Mt CO₂e/yr</b> (${yr})<br/>` +
                `<a href="https://climatetrace.org" target="_blank" rel="noopener">Climate TRACE ↗</a>`,
            },
          });
        }
        return `top ${assets.length} emitters · ${yr} inventory · snapshot ${json.snapshot}`;
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
  pointLayers[kind] = { collection: col, meta, __json: json };
  if (kind === "climatetrace") climateTraceLoadedYear = climateTraceYear(json);
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

// Emissions is year-aware: it explains the yearly semantics on enable, with
// the actual displayed year, instead of a generic "date doesn't apply" toast.
function climateTraceToast() {
  const json = pointLayers.climatetrace?.__json;
  if (!json) return;
  const yr = climateTraceYear(json);
  const lo = json.years[0], hi = json.years[json.years.length - 1];
  const want = Number(state.date.slice(0, 4));
  const note = want !== yr
    ? ` (nearest available to your ${want}; data covers ${lo}–${hi})`
    : ` — set the date's <em>year</em> anywhere in ${lo}–${hi} to switch inventories`;
  showToast(`<strong>Climate TRACE emissions</strong> is a <strong>yearly</strong> inventory: ` +
    `the day and month don't matter, but the <strong>year does</strong>. Showing <strong>${yr}</strong>${note}.`,
    { key: "climatetrace" });
}
document.getElementById("toggle-climatetrace").addEventListener("change", (e) => {
  if (e.target.checked) {
    ensureClimateTraceYear();   // re-enabling after the year moved → rebuild
    loadPointLayer("climatetrace").then(() => { updateDeltaHint(); climateTraceToast(); });
  } else if (pointLayers.climatetrace) {
    pointLayers.climatetrace.collection.show = false;
  }
  updateDeltaHint();
});
document.getElementById("toggle-argo").addEventListener("change", (e) => {
  if (e.target.checked) { loadPointLayer("argo").then(updateDeltaHint); maybeDatelessToast("argo"); }
  else if (pointLayers.argo) pointLayers.argo.collection.show = false;
  updateDeltaHint();
});

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
  const picked = seeThrough(viewer.scene.pick(click.position));
  if (picked?.id?.kind) {
    pickCard.innerHTML = picked.id.html;
    pickCard.classList.remove("hidden");
    return;
  }
  if (!picked) pickCard.classList.add("hidden");
  // Nothing pickable under the cursor: if the inspector is engaged and the
  // click actually hit the globe (not sky), compose that point's state card.
  let tideTook = false;
  if (!picked && tideLive.on) {
    const cart = viewer.camera.pickEllipsoid(click.position, viewer.scene.globe.ellipsoid);
    if (cart) tideTook = tideSelectPoint(Cesium.Cartographic.fromCartesian(cart));
    // fall through: the inspector still composes its card on the same tap
  }
  if (!picked && pixelInspectorEngaged()) {
    const cart = viewer.camera.pickEllipsoid(click.position, viewer.scene.globe.ellipsoid);
    if (cart) showPixelState(Cesium.Cartographic.fromCartesian(cart));
  } else if (!picked && !topColormapLayer() && !tideLive.on && !tideTook) {
    // Nothing armed and nothing colormapped: the tap deliberately does
    // NOTHING — but silence looks broken, so say why, once per sitting.
    const cart = viewer.camera.pickEllipsoid(click.position, viewer.scene.globe.ellipsoid);
    if (cart) {
      showToast(
        `Nothing is switched on to read here. Enable a data layer to tap values ` +
        `off the map, or check <strong>Everything we know (pixel state)</strong> ` +
        `for the full report on any point.`,
        { key: "tap-nothing-armed" });
    }
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
    if (e.layer && (e.cfg.colormap || e.cfg.classmap || e.cfg.grid)) {
      const idx = viewer.imageryLayers.indexOf(e.layer);
      if (idx > bestIdx) { bestIdx = idx; best = e; }
    }
  }
  return best;
}

/* The geographic footprint of one source-tile pixel — the answer to "WHICH
 * pixel did that number come from". Every probe result carries its cell so
 * the globe can outline it: a read-out floating next to a tap says what the
 * value is, but only the drawn cell says where the instrument's sample sits,
 * and on a phone (fat finger, offset tooltip) the two are easily 50 km apart. */
function probeCellBounds(z, x, y, px, py) {
  const span = (0.5625 / 2 ** z) * 512;      // degrees per tile at level z
  const cs = span / 512;                     // degrees per source pixel
  const west = -180 + x * span + px * cs;
  const north = 90 - y * span - py * cs;
  return { west, south: north - cs, east: west + cs, north };
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

/* MODIS LST's colormap is calibrated in kelvin; a probe reading "303 K"
 * answers a question nobody asked. Absolute values convert to Celsius for
 * display; deltas are scale-free in K==degC and ratios are unitless. */
function kelvinToC(res) {
  if (res && !res.noData && !res.delta && !res.ratio && res.units === "K") {
    return {
      ...res, value: res.value - 273.15, units: "°C",
      cap: res.cap && { sign: res.cap.sign, bound: res.cap.bound - 273.15 },
    };
  }
  return res;
}

function colormapLayersTopDown() {
  return Object.values(state.layers)
    // a fine layer above its gate is hidden and must not answer for what is
    // not on screen
    .filter((e) => e.layer && e.layer.show !== false &&
      (e.cfg.colormap || e.cfg.classmap || e.cfg.grid))
    .map((e) => [viewer.imageryLayers.indexOf(e.layer), e])
    .sort((a, b) => b[0] - a[0])
    .map(([, e]) => e);
}

/* The tide as the value probe sees it, so it behaves like every other layer:
 * the same floating read-out, the same outlined source cell, the same date
 * stamp. The tide is a primitive drawn OVER the imagery layers, so it answers
 * first where it has water — and returns null on land, falling through to
 * whatever is underneath, exactly the rule probeValueAt already uses between
 * imagery layers. Exact cell only (rings 0): the probe reports the pixel
 * under the cursor and must not quietly borrow a neighbour's value. */
function tideProbeValue(carto) {
  if (!tideLive.on || !tideData || !tideFields) return null;
  const lon = Cesium.Math.toDegrees(carto.longitude);
  const lat = Cesium.Math.toDegrees(carto.latitude);
  const hit = tideNearestWater(lon, lat, 0);
  if (!hit) return null;
  const cm = tideHeightAt(hit.i, tideSim.t);
  if (!Number.isFinite(cm)) return null;
  const nx = tideExtrema(hit.i, tideSim.t, 30)[0];
  const clat = tideData.south + hit.iy, clon = tideData.west + hit.ix;
  const c = nx ? tideClock(nx.ms, tideTzFor(clon + 0.5, clat + 0.5), tideSim.t) : null;
  return {
    title: "Tide height (live)",
    units: "m", value: cm / 100, lon, lat,
    when: whenAt("instant", new Date(tideSim.t).toISOString().slice(0, 16)),
    cell: { west: clon, south: clat, east: clon + 1, north: clat + 1 },
    extra: nx ? `next ${nx.high ? "high" : "low"} water ${c.hhmm} ${c.abbr}${c.day} ` +
                `· in ${tideCountdown(nx.ms - tideSim.t)}` : "",
  };
}

async function probeValueAt(carto) {
  /* Try every colormapped/grid layer top-down: where the top layer is
   * transparent at this point (LST over ocean, SST over land, a dry forecast
   * cell), the probe falls through to the next layer instead of reading
   * "no data" off a pixel the user can plainly see is coloured by the layer
   * beneath. The temperature scene (SST+LST) depends on this. */
  const tide = tideProbeValue(carto);              // drawn on top → asked first
  if (tide) return tide;
  const entries = colormapLayersTopDown();
  if (!entries.length) return null;
  let first = null;
  for (const entry of entries) {
    const res = await probeEntryValue(entry, carto);
    if (!first) first = res;
    if (res && !res.noData) return kelvinToC(res);
  }
  return kelvinToC(first);
}

async function probeEntryValue(entry, carto) {
  const cfg = entry.cfg;
  const lon = Cesium.Math.toDegrees(carto.longitude);
  const lat = Cesium.Math.toDegrees(carto.latitude);
  if (cfg.grid) {
    // Grid overlays: read the exact cell value straight from the loaded grid.
    const g = await loadGridMonth(cfg);
    // `when` on every result, so the tooltip can say what the card says.
    const base = { title: cfg.title, units: cfg.units, lon, lat, when: whenOfGrid(cfg, g) };
    if (!g) return { ...base, noData: true };
    // the source cell of THIS grid — a 1° climatology cell, drawn as such
    const ix = Math.floor((lon - g.west) / g.dlon), iy = Math.floor((lat - g.south) / g.dlat);
    if (ix >= 0 && ix < g.nx && iy >= 0 && iy < g.ny) {
      base.cell = { west: g.west + ix * g.dlon, south: g.south + iy * g.dlat,
                    east: g.west + (ix + 1) * g.dlon, north: g.south + (iy + 1) * g.dlat };
    }
    const v = sampleGrid(g, lon, lat);
    if (v == null) return { ...base, noData: true };
    // Categorical grid: answer with the class NAME. Same rule as the
    // classification rasters — a code rendered as "4" tells the reader nothing.
    if (cfg.classGrid) {
      const label = gridClassLabel(g, v);
      return label == null ? { ...base, noData: true } : { ...base, label };
    }
    return { ...base, value: v };
  }
  if (cfg.classmap) {
    // Classification raster: read the class NAME under the cursor. No
    // delta/aggregate variants exist — class codes don't average or subtract.
    const lut = await getClassLut(cfg.classmap);
    if (!lut) return null;
    const z = cfg.maxLevel;
    const t = tileCoordsAt(lon, lat, z);
    const base = { title: cfg.title, lon, lat, when: whenOfGibs(cfg),
                   cell: probeCellBounds(z, t.x, t.y, t.px, t.py) };
    const label = await probeClassPixel(cfg, state.date, z, t.x, t.y, t.px, t.py, lut);
    return label == null ? { ...base, noData: true } : { ...base, label };
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
  // For a window mean or a delta the stamp is the window's END: the sample list
  // walks BACKWARD from state.date, and the "past N days mean" suffix already
  // carries the aggregation. `whenPast` is the OTHER end of a comparison, and it
  // is resolved through gibsTime like everything else — which matters, because
  // for a layer whose tiles stop at an endTime both ends clamp to the same date
  // and the difference is identically zero. Printing the requested date there
  // would present that zero as a real "no change".
  const base = { title: cfg.title, units: vlut.units, lon, lat, when: whenOfGibs(cfg),
                 cell: probeCellBounds(z, x, y, px, py) };

  if (entry.isDelta) {
    // Δ = window-mean(now) − window-mean(past), matching the rendered delta
    const cmp = compareDate();
    base.whenPast = whenOfGibs(cfg, cmp);
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
    base.whenPast = whenOfGibs(cfg, cmp);
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
  // A catch-all bin answers with its bound ("< 30"), not an invented midpoint.
  const cap = vlut.caps?.get(v);
  // Only the anomaly layer has a correctable cap: its ±3 bound is a palette
  // edge, not a limit of the instrument. SMAP's "< 30 PSU" is a genuine
  // retrieval floor and must keep reading as a bound.
  const trueAnom = cap && cfg.id === "sst-anom" ? sstAnomalyAt(lon, lat) : null;
  // `upgradable` marks the one cap the tooltip can improve on AFTER it has been
  // drawn: the daily 0.25° read needs a round-trip and this function must not
  // grow one, because the whole probe is on the dwell path. renderProbe fires
  // it and rewrites the value in place — see upgradeProbeAnomaly.
  return { ...base, value: v, cap, trueAnom, upgradable: !!cap && cfg.id === "sst-anom" };
}

const probeEl = document.getElementById("value-probe");

/* The probed cell, drawn on the globe. The tooltip floats NEXT TO the tap
 * (offset so a finger doesn't cover it), which on a phone left no way to tell
 * which pixel had been read — a tap near a salinity mask edge could land on
 * data or on blank and the read-out looked identical. Three entities, reused:
 * a translucent fill + outline on the exact source-cell footprint (visible
 * once you're zoomed near the data's own resolution), and a small ring at the
 * tap point that is visible at every zoom. */
let probeMarkEnts = null;
function probeMarkEntities() {
  if (probeMarkEnts) return probeMarkEnts;
  const accent = Cesium.Color.fromCssColorString("#4493f8");
  const fill = viewer.entities.add({
    show: false,
    rectangle: {
      coordinates: Cesium.Rectangle.fromDegrees(0, 0, 1, 1),
      material: accent.withAlpha(0.16),
      height: 0,     // plain primitive on the ellipsoid, not a ground primitive
    },
  });
  const edge = viewer.entities.add({
    show: false,
    polyline: {
      positions: [],
      width: 2,
      material: accent.withAlpha(0.9),
      // RHUMB: a lat/lon cell's edges follow parallels/meridians, not geodesics
      arcType: Cesium.ArcType.RHUMB,
    },
  });
  const dot = viewer.entities.add({
    show: false,
    point: {
      pixelSize: 5,
      color: Cesium.Color.TRANSPARENT,
      outlineColor: accent,
      outlineWidth: 2,
      disableDepthTestDistance: Number.POSITIVE_INFINITY,
    },
  });
  probeMarkEnts = { fill, edge, dot };
  return probeMarkEnts;
}

function showProbeMark(res) {
  const m = probeMarkEntities();
  m.dot.position = Cesium.Cartesian3.fromDegrees(res.lon, res.lat);
  m.dot.show = true;
  const c = res.cell;
  if (c) {
    m.fill.rectangle.coordinates = Cesium.Rectangle.fromDegrees(c.west, c.south, c.east, c.north);
    m.edge.polyline.positions = Cesium.Cartesian3.fromDegreesArray([
      c.west, c.south, c.east, c.south, c.east, c.north, c.west, c.north, c.west, c.south,
    ]);
    m.fill.show = true;
    m.edge.show = true;
  } else {
    m.fill.show = false;
    m.edge.show = false;
  }
}

/* keepMarks: while the pixel card is open its marks belong to the CARD's
 * point — pointer movement may hide the floating read-out, but must not
 * un-mark the pixel the open card is describing. */
function hideProbe(keepMarks = false) {
  probeEl.classList.add("hidden");
  if (probeMarkEnts && !keepMarks) {
    probeMarkEnts.fill.show = false;
    probeMarkEnts.edge.show = false;
    probeMarkEnts.dot.show = false;
  }
}

/* If the marked pixel sits under a read-out panel (the pixel card covers most
 * of a phone's globe view), rotate the globe so the mark is actually visible:
 * scan a coarse grid of canvas points for the spot farthest from every panel
 * and edge that still lies ON the globe, then fly the camera by the lon/lat
 * offset that puts the mark there. Same height, same heading/pitch — a
 * rotation, not a zoom. Returns true if it moved the camera. */
function ensureMarkVisible() {
  const m = probeMarkEnts;
  if (!m || !m.dot.show) return false;
  const scene = viewer.scene;
  const st = Cesium.SceneTransforms;
  const toWin = (st.worldToWindowCoordinates || st.wgs84ToWindowCoordinates).bind(st);
  const pos = m.dot.position.getValue(viewer.clock.currentTime);
  const crect = scene.canvas.getBoundingClientRect();
  // Inflated by a finger-width margin: a mark hugging the panel's edge is
  // as unreadable under a thumb as one strictly beneath it.
  const PAD = 16;
  const rects = [probeEl, pixelCardEl]
    .filter((el) => el && !el.classList.contains("hidden"))
    .map((el) => el.getBoundingClientRect())
    .filter((r) => r.width > 0 && r.height > 0)
    .map((r) => ({ left: r.left - PAD, right: r.right + PAD,
                   top: r.top - PAD, bottom: r.bottom + PAD }));
  const coveredBy = (x, y) => rects.some((r) =>
    x + crect.left >= r.left && x + crect.left <= r.right &&
    y + crect.top >= r.top && y + crect.top <= r.bottom);
  // worldToWindowCoordinates happily projects points on the FAR side of the
  // globe; the occluder says whether the mark is on the face the user sees.
  const occ = new Cesium.EllipsoidalOccluder(Cesium.Ellipsoid.WGS84, viewer.camera.position);
  const w = occ.isPointVisible(pos) ? toWin(scene, pos) : null;
  if (w && w.x >= 0 && w.y >= 0 && w.x <= crect.width && w.y <= crect.height &&
      !coveredBy(w.x, w.y)) return false;                    // already in view

  let best = null, bestScore = -Infinity;
  for (let i = 1; i < 8; i++) {
    for (let j = 1; j < 8; j++) {
      const x = (crect.width * i) / 8, y = (crect.height * j) / 8;
      if (coveredBy(x, y)) continue;
      const cart = viewer.camera.pickEllipsoid(new Cesium.Cartesian2(x, y), scene.globe.ellipsoid);
      if (!cart) continue;                                   // off the limb
      let score = Math.min(x, crect.width - x, y, crect.height - y);
      for (const r of rects) {
        const dx = Math.max(r.left - (x + crect.left), 0, (x + crect.left) - r.right);
        const dy = Math.max(r.top - (y + crect.top), 0, (y + crect.top) - r.bottom);
        score = Math.min(score, Math.hypot(dx, dy));
      }
      if (score > bestScore) { bestScore = score; best = { x, y, cart }; }
    }
  }
  if (!best) return false;
  // Move the camera by the same lon/lat offset that separates the mark from
  // the geography currently under the chosen spot — with height, heading and
  // pitch unchanged, the whole view translates and the mark lands there.
  const target = Cesium.Cartographic.fromCartesian(pos);
  const at = Cesium.Cartographic.fromCartesian(best.cart);
  const cam = viewer.camera.positionCartographic;
  const lon = Cesium.Math.negativePiToPi(cam.longitude + (target.longitude - at.longitude));
  const lat = Cesium.Math.clamp(cam.latitude + (target.latitude - at.latitude),
    -Cesium.Math.PI_OVER_TWO * 0.99, Cesium.Math.PI_OVER_TWO * 0.99);
  viewer.camera.flyTo({
    destination: Cesium.Cartesian3.fromRadians(lon, lat, cam.height),
    orientation: { heading: viewer.camera.heading, pitch: viewer.camera.pitch, roll: 0 },
    duration: 0.6,
    complete: () => {
      // the floating read-out follows its mark to the new screen position
      if (probeEl.classList.contains("hidden")) return;
      const w2 = toWin(scene, m.dot.position.getValue(viewer.clock.currentTime));
      if (w2) placeProbe(w2.x, w2.y);
    },
  });
  return true;
}

/* Anchor the read-out clear of the tap. The old rule was "+14px right of the
 * cursor, clamp at the screen edge" — built for a mouse. On a phone the tap
 * IS the point of interest, a finger is ~40 CSS px wide, and the edge-clamp
 * slid the box straight back OVER the tap: the user watched their own mark
 * disappear under the panel describing it. Sides are tried in order (right,
 * left, above, below) with finger-sized clearance; the box never sits on the
 * tapped point. Must run with the element VISIBLE (it is measured). */
function placeProbe(sx, sy) {
  const cw = viewer.scene.canvas.clientWidth, ch = viewer.scene.canvas.clientHeight;
  const w = probeEl.offsetWidth, h = probeEl.offsetHeight;
  const GAP = 32;
  let left, top;
  if (sx + GAP + w <= cw - 4) {          // room to the right
    left = sx + GAP;
    top = Math.max(4, Math.min(sy - 10, ch - h - 4));
  } else if (sx - GAP - w >= 4) {        // room to the left
    left = sx - GAP - w;
    top = Math.max(4, Math.min(sy - 10, ch - h - 4));
  } else {                                // narrow screen: go above, else below
    left = Math.max(4, Math.min(sx - w / 2, cw - w - 4));
    top = sy - GAP - h >= 4 ? sy - GAP - h : Math.min(sy + GAP, ch - h - 4);
  }
  probeEl.style.left = `${left}px`;
  probeEl.style.top = `${top}px`;
}

/* Which point the tooltip is currently about — the same guard as pixelCardSeq,
 * one rung down. The tooltip renders SYNCHRONOUSLY (that is the whole design of
 * the dwell probe), so the daily 0.25° read can only ever arrive after the box
 * is already on screen, and by then the cursor may be somewhere else entirely.
 * A hover that moved on must never receive another point's value: every render
 * takes a ticket, and the upgrade only writes if its ticket is still current. */
let probeSeq = 0;

/* The in-place upgrade: replace the capped read-out with the true departure for
 * the exact selected day, once the Hub answers. Deliberately NOT awaited by
 * renderProbe — the instant behaviour (monthly `trueAnom` when the normals are
 * resident, else the honest palette bound) is what the user sees at once, and
 * this only ever improves it. On failure it does nothing at all, which leaves
 * exactly the pre-E-040 tooltip. */
async function upgradeProbeAnomaly(res, turn) {
  const daily = await sstDailyAnomaly(res.lon, res.lat, state.date).catch(() => null);
  // Second choice, not a placeholder: if the Hub could not answer, the resident
  // monthly figure is still better than a bare bound — but it is a DIFFERENT
  // month at a coarser cell, so it arrives carrying that label rather than
  // impersonating an answer for the selected day.
  const val = daily || res.trueAnom;
  if (!val) return;
  if (turn !== probeSeq) return;                        // a newer point owns the box
  if (probeEl.classList.contains("hidden")) return;
  const head = probeEl.querySelector(".vp-head");
  if (!head) return;
  const qualifier = daily
    ? `(palette ${res.cap.sign} ${fmtVal(res.cap.bound)})`
    : `(1° monthly, ${val.month})`;
  head.innerHTML =
    `<span class="vp-val">${val.v >= 0 ? "+" : "−"}${fmtVal(Math.abs(val.v))}</span> ` +
    `<span class="vp-unit">°C</span> ` +
    `<span class="vp-unit">${qualifier}</span>`;
}

function renderProbe(res, sx, sy) {
  if (!res) { hideProbe(); return; }
  const myTurn = ++probeSeq;
  const coord = `${Math.abs(res.lat).toFixed(2)}°${res.lat >= 0 ? "N" : "S"}, ` +
                `${Math.abs(res.lon).toFixed(2)}°${res.lon >= 0 ? "E" : "W"}`;
  let head;
  if (res.noData) {
    head = `<span class="vp-val vp-nd">no data</span>`;
  } else if (res.label) {
    // classification layer: the answer is a category, not a number
    head = `<span class="vp-val vp-class">${res.label}</span>`;
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
  } else if (res.cap) {
    /* Catch-all colormap bin: the tile only says "off this scale".
     *
     * WHEN A BETTER READ IS COMING, SHOW THE BOUND AND WAIT. It is tempting to
     * fill the gap with the resident MONTHLY correction, and that is what this
     * did until 2026-08-18, when Chris clicked the Mediterranean and watched
     * the tooltip say 2.x and then 4.y. Both numbers were right and they
     * measured different things: the monthly path had fallen back to 2026-07
     * (the archive's newest month ≤ the selected date) on a 1° cell, while the
     * daily read answered for 2026-08-16 at 0.25°. Measured that day in the
     * Balearic: monthly +2.25 (25.8 vs 23.55, JULY) against daily +4.57
     * (29.55 vs 24.98, the selected day) — and the monthly figure even
     * contradicted the palette it was correcting, sitting below the ≥3 bound.
     *
     * A number that is replaced by a DIFFERENT number reads as the app
     * correcting a mistake. A bound replaced by a number reads as what it is:
     * a refinement. So the sequence is bound → best available value, and the
     * monthly figure becomes the upgrade's own fallback (labelled with its
     * month) rather than a placeholder pretending to be the answer. */
    const real = res.upgradable ? null : res.trueAnom;
    head = real
      ? `<span class="vp-val">${real.v >= 0 ? "+" : "−"}${fmtVal(Math.abs(real.v))}</span> ` +
        `<span class="vp-unit">°C</span> ` +
        `<span class="vp-unit">(palette ${res.cap.sign} ${fmtVal(res.cap.bound)})</span>`
      : `<span class="vp-val">${res.cap.sign} ${fmtVal(res.cap.bound)}</span> ` +
        `<span class="vp-unit">${res.units}</span>`;
  } else {
    head = `<span class="vp-val">${fmtVal(res.value)}</span> <span class="vp-unit">${res.units}</span>`;
  }
  // The comparison end says the date the tiles were ACTUALLY read at, not the
  // one the selector asked for — see probeEntryValue. Falls back to the request
  // only when the layer resolves to no date at all (untimed rasters).
  const past = whenLabel(res.whenPast) || compareDate();
  const suffix = res.delta ? ` · Δ vs ${past}${state.windowDays > 1 ? ", " + windowLabel(state.windowDays) + " mean" : ""}`
    : res.ratio ? ` · vs ${past}, ratio of ${windowLabel(state.windowDays)} means`
    : res.aggregated ? ` · ${windowLabel(state.windowDays)} mean` : "";
  // The date line the user asked for: every hovered or tapped value now says
  // when it was observed, at that dataset's own honest granularity.
  const stamp = whenLabel(res.when);
  // The value is wrapped so the daily upgrade below has ONE node to rewrite —
  // the read-out is otherwise three sibling spans and a partial replacement
  // would leave the old unit or the old bound standing next to a new number.
  probeEl.innerHTML = `<span class="vp-head">${head}</span>` +
    `<div class="vp-meta">${res.title}${suffix}` +
    `${res.extra ? `<br/>${res.extra}` : ""}` +
    `${stamp ? `<br/>${stamp}` : ""}<br/>${coord}</div>`;
  probeEl.classList.remove("hidden");
  placeProbe(sx, sy);
  // Rendered first, corrected after: the capped SST anomaly is the one read
  // where the tile genuinely cannot carry the answer, so it is worth a
  // round-trip the tooltip does not wait for.
  if (res.upgradable && res.cap) upgradeProbeAnomaly(res, myTurn);
  // even a "no data" read marks its cell — seeing WHERE the empty cell sits is
  // what tells a salinity mask edge apart from a broken layer. With the pixel
  // card open the marks belong to the card's point; a hover read-out has the
  // cursor itself as its pointer and must not steal them.
  if (!pixelInspectorEngaged()) showProbeMark(res);
}

/* The probe only fires after the cursor *rests* (dwell), so rotating/panning the
 * globe never triggers per-frame tile reads. Any movement hides it and restarts
 * the dwell timer; it computes once the mouse has been still for PROBE_DWELL ms. */
const PROBE_DWELL = 650;
let probeDwellTimer = null;
async function runProbe(x, y, ensure = false) {
  const cart = viewer.camera.pickEllipsoid({ x, y }, viewer.scene.globe.ellipsoid);
  if (!cart) { hideProbe(); return; }
  try {
    renderProbe(await probeValueAt(Cesium.Cartographic.fromCartesian(cart)), x, y);
    // Only a deliberate TAP earns a camera move (a phone tap can sit right
    // under the read-out); rotating the globe under a hovering cursor would
    // change what the cursor points at.
    if (ensure) ensureMarkVisible();
  }
  catch { hideProbe(); }
}
new Cesium.ScreenSpaceEventHandler(viewer.scene.canvas).setInputAction((m) => {
  hideProbe(pixelInspectorEngaged());        // hide immediately while moving
  if (probeDwellTimer) clearTimeout(probeDwellTimer);
  if (!topColormapLayer() && !tideLive.on) return;
  const x = m.endPosition.x, y = m.endPosition.y;
  probeDwellTimer = setTimeout(() => runProbe(x, y), PROBE_DWELL);
}, Cesium.ScreenSpaceEventType.MOUSE_MOVE);
// Clicking reads the top layer's value immediately (no dwell wait) — unless
// the click is opening the pixel-state card, which includes the same value;
// two overlapping read-outs of one click would be noise.
new Cesium.ScreenSpaceEventHandler(viewer.scene.canvas).setInputAction((c) => {
  if (probeDwellTimer) clearTimeout(probeDwellTimer);
  if (pixelInspectorEngaged()) return;
  if ((topColormapLayer() || tideLive.on) &&
      !seeThrough(viewer.scene.pick(c.position))?.id?.kind) {
    runProbe(c.position.x, c.position.y, true);
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

/* How long before the card appears at all. Two seconds, because the card
 * exists to answer "what is happening HERE" and a tap that shows nothing for
 * longer than that reads as a tap that didn't register (Chris, 2026-08-16:
 * "15s is still a very long time"). It is NOT a cutoff — nothing is dropped
 * for missing it. Sources that land later redraw the card in place, so the
 * cost of being slow is appearing a moment after the frame, not being left
 * out of it. */
const PIXEL_DEADLINE_MS = 2000;
/* Arrivals cluster — the whole Open-Meteo family answers within a few hundred
 * ms of itself — and every redraw rebuilds the card's DOM. Coalescing over
 * this window turns a burst into one rebuild. Long enough to catch a cluster,
 * short enough that no row is visibly late to its own section. */
const PIXEL_REDRAW_MS = 250;
const PIXEL_RASTERS = ["sst", "sst-anom", "ssh-anom", "precip", "seaice", "snow", "aod", "lst",
  "soilmoisture", "ndvi", "grace", "ceres", "chlor", "salinity", "dist-alert", "elevation"];
const PIXEL_GRIDS = ["oisst", "gpcp", "eobs", "meteoswiss", "tides"];
// The two CMIP6 windows, declared once: the rows are stamped with the same
// spans that were requested, so the label can never drift from the query.
const CLIM_BASE_WIN = ["1991-01-01", "1995-12-31"];
const CLIM_FUT_WIN = ["2045-01-01", "2049-12-31"];
const winPeriod = (w) => whenAt("period", `${w[0].slice(0, 4)}-${w[1].slice(0, 4)}`);

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
  if (cfg.classmap) {
    // Classification channel: read at native resolution — a 30 m disturbance
    // alert averaged down to level 4 would vanish, and the inspector's whole
    // job is "what is true AT this point".
    const lut = await getClassLut(cfg.classmap);
    if (!lut) return null;
    const t = tileCoordsAt(lon, lat, cfg.maxLevel);
    const label = await probeClassPixel(cfg, state.date, cfg.maxLevel, t.x, t.y, t.px, t.py, lut);
    return label == null ? null : { label };
  }
  const vlut = await getValueLut(cfg.colormap);
  if (!vlut) return null;
  // probeNative (elevation): a 4 km mean of alpine terrain is not the height
  // of the point that was tapped; read the 30 m tile.
  const z = cfg.probeNative ? cfg.maxLevel : Math.min(cfg.maxLevel, 4);
  const t = tileCoordsAt(lon, lat, z);
  const v = await probePixel(cfg, state.date, z, t.x, t.y, t.px, t.py, vlut.lut);
  if (v == null) return null;
  const cap = vlut.caps?.get(v);
  // same kelvin→Celsius courtesy as the probe (MODIS LST)
  return vlut.units === "K"
    ? { v: v - 273.15, units: "°C", cap: cap && { sign: cap.sign, bound: cap.bound - 273.15 } }
    : { v, units: vlut.units, cap };
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
/* One retry, then give up. The card fires six of these at once and each one
 * owns a whole section; a single dropped response used to delete "Now & next
 * 7 days" outright, which reads as a broken app rather than as a hiccup. Two
 * attempts is the honest bound: retrying harder would turn a rate-limit
 * (Open-Meteo counts per IP across its whole family) into a self-inflicted
 * one. 429/5xx and network errors retry; a 400 is our own bad URL and never
 * will succeed, so it doesn't. */
/* A REQUEST THAT NEVER SETTLES IS THE WORST FAILURE MODE HERE, so every
 * attempt carries its own deadline. `fetch()` has no timeout — measured
 * against the live site 2026-08-16, one climate-api call sat open for a full
 * minute and, because the card awaits Promise.all before it renders anything,
 * the whole inspector showed "Reading this point…" for 64 s. A phone user
 * reads that as "the app is broken", and correctly so. The retry above only
 * made it worse: with no deadline there is nothing to retry FROM.
 * The timeout is DELIBERATELY GENEROUS, and that is not a contradiction: it
 * is a stop on hanging, not a service-level target. Timed through the proxy,
 * three consecutive climate-api calls took 1.0 s, 23 s, and never — the same
 * query, minutes apart. Cutting the 23 s one off would throw away a good
 * answer to save time the card no longer spends waiting anyway, because
 * PIXEL_DEADLINE_MS already renders without it and the redraw collects it
 * when it lands. So: short deadline to DRAW, long timeout to GIVE UP. */
const OM_TIMEOUT_MS = 45000;
function omGet(url, tries = 2) {
  // AbortSignal.timeout is not in older Safari; the controller form is.
  const ctl = new AbortController();
  const timer = setTimeout(() => ctl.abort(), OM_TIMEOUT_MS);
  return fetch(url, { signal: ctl.signal }).then((r) => {
    if (r.ok) return r.json();
    if (tries > 1 && (r.status === 429 || r.status >= 500)) throw new Error(r.status);
    return null;
  }).catch(() => {
    if (tries <= 1) return null;
    return new Promise((res) => setTimeout(res, 700)).then(() => omGet(url, tries - 1));
  }).finally(() => clearTimeout(timer));
}
function omLL(lon, lat) {
  return `latitude=${lat.toFixed(3)}&longitude=${lon.toFixed(3)}`;
}
/* ONE forecast call carries both the weather rows and the heat-load rows.
 * It was briefly two — the heat block needs LOCAL calendar days (a "tropical
 * night" is defined on the local night) while the weather rows were pinned to
 * timezone=UTC — but Open-Meteo rate-limits per IP across its whole API
 * family, and the card already fires six requests at once; a seventh made the
 * burst drop sections at random. So: ask once, in local time, and convert the
 * instant stamps back to UTC with the offset the response carries (omUTC).
 * The daily strip is better on local days anyway — "the 11th" now means the
 * 11th where the pixel is. */
function fetchOpenMeteo(lon, lat) {
  return omGet("https://api.open-meteo.com/v1/forecast?" + omLL(lon, lat) +
    "&current=temperature_2m,relative_humidity_2m,precipitation,pressure_msl," +
    "wind_speed_10m,wind_direction_10m,soil_moisture_0_to_1cm,soil_temperature_0cm," +
    "shortwave_radiation,apparent_temperature" +
    "&daily=temperature_2m_max,temperature_2m_min,precipitation_sum,wind_speed_10m_max," +
    "apparent_temperature_max" +
    "&timezone=auto&forecast_days=7");
}

/* A local timestamp from a timezone=auto response, restated in UTC — so every
 * stamp on the card keeps meaning exactly what it says. */
function omUTC(t, offsetSec) {
  if (!t) return t;
  const ms = Date.parse(`${t}${/[Zz]|[+-]\d\d:?\d\d$/.test(t) ? "" : "Z"}`);
  if (!Number.isFinite(ms)) return t;
  return new Date(ms - (offsetSec || 0) * 1000).toISOString().slice(0, 16);
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
  // The checkbox IS the intent — nothing else. There used to be a fallback
  // (no colormapped layer active → a bare-globe click opened the card anyway,
  // "since there was nothing else to read"), and it read as a bug from a
  // phone: every box unchecked, yet a tap poured out the full card. An
  // unchecked control that behaves checked is worse than a tap that does
  // nothing — the click handler explains the nothing with a toast instead.
  return !!document.getElementById("toggle-pixel")?.checked;
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
/* Which point the card is currently about. The card draws twice when a source
 * is slow, so a tap on a NEW point can land between the two — and the old
 * point's straggler would then redraw its data under the new point's heading.
 * Every draw checks it still owns the card. */
let pixelCardSeq = 0;
/* Third argument is the row's own observation time (a whenAt() object), because
 * a section-wide heading cannot tell the truth here: gibsTime() clamps and snaps
 * PER LAYER, so rows sitting side by side under one heading are routinely years
 * apart. A row with no honest observation time — elevation, a station's name,
 * a glacier count — passes nothing and prints no stamp, which is the only other
 * truthful option. */
function pixelRow(label, value, when = null) {
  return `<div class="px-row"><span class="px-label">${label}</span>` +
    `<span class="px-val">${value}</span>${whenStamp(when)}</div>`;
}

/* THE TRUE SST ANOMALY, when the picture cannot carry it.
 *
 * GIBS serves colours, not numbers, and the MUR25 anomaly palette runs
 * -3..+3 degC with catch-all end bins (`[3.0,+INF)`). So at the height of a
 * strong El Nino the eastern Pacific saturates and every pixel inverts to
 * "at least 3" — the reading is capped exactly when the magnitude is the
 * whole story (Chris, 2026-08-18: "I understand if the legend caps at 3C. But
 * that doesn't mean we cannot know the values"). No client work recovers it
 * from the tile: the information is not in the picture.
 *
 * So this subtracts instead of inverting a colour: OISST monthly mean minus
 * that same calendar month's 1991-2020 normal, both from the archive we
 * already ship. Same dataset on both sides ON PURPOSE — mixing MUR's 1 km
 * daily field with a 1-degree monthly normal reads ~2 degC cold off Peru,
 * because MUR resolves the coastal upwelling tongue that a 1-degree cell
 * averages away, and Peru is precisely where this gets used.
 *
 * PREFETCHED, not fetched on demand: the hover probe renders synchronously,
 * so a value that needs a round-trip cannot appear in it at all. Enabling the
 * anomaly layer (or moving the date while it is on) loads the two files in the
 * background; `sstAnomalyAt` is then a pure lookup that either has the answer
 * or honestly does not. */
const sstNorm = { stamp: null, geom: null, clim: null, frame: null, loading: null };

function sstNormalsStamp(dateStr, oiIdx) {
  const want = (dateStr || state.date).slice(0, 7);
  const stamps = oiIdx?.monthsAvailable || [];
  return stamps.filter((s) => s <= want).pop() || stamps[0] || null;
}

function ensureSstNormals(dateStr) {
  // Serialise rather than skip: scrubbing the date while a month is in flight
  // used to return the IN-FLIGHT promise and never load the month actually
  // asked for, so the probe kept answering with the previous month's normals.
  const run = async () => {
    const [idx, oiIdx] = await Promise.all([
      pixelJson("data/oisst_clim.json"), pixelJson("data/oisst_monthly.json"),
    ]);
    if (!idx || !oiIdx) return null;
    const use = sstNormalsStamp(dateStr, oiIdx);
    if (!use) return null;
    if (sstNorm.stamp === use) return sstNorm;
    const [clim, yearFile] = await Promise.all([
      pixelJson(`${idx.monthDir}/${use.slice(5, 7)}.json`),
      pixelJson(`${oiIdx.yearDir}/${use.slice(0, 4)}.json`),
    ]);
    const frame = yearFile?.months?.[use];
    if (!clim?.values || !frame) return null;
    sstNorm.stamp = use;
    sstNorm.period = idx.period;
    sstNorm.geom = { west: idx.west, south: idx.south, east: idx.east, north: idx.north,
                     dlon: idx.dlon, dlat: idx.dlat, nx: idx.nx, ny: idx.ny };
    sstNorm.clim = clim.values;
    sstNorm.frame = frame;
    return sstNorm;
  };
  sstNorm.loading = sstNorm.loading ? sstNorm.loading.then(run, run) : run();
  return sstNorm.loading;
}

/* Synchronous by design — see the prefetch note above. Returns null when the
 * normals are not resident yet, which the callers render as the honest capped
 * bound rather than as a wrong number. */
function sstAnomalyAt(lon, lat) {
  if (!sstNorm.clim || !sstNorm.frame) return null;
  const now = sampleGrid({ ...sstNorm.geom, values: sstNorm.frame }, lon, lat);
  const norm = sampleGrid({ ...sstNorm.geom, values: sstNorm.clim }, lon, lat);
  if (now == null || norm == null) return null;
  return { v: now - norm, month: sstNorm.stamp, period: sstNorm.period,
           sst: now, norm };
}

/* THE SAME QUESTION, ANSWERED FOR THE ACTUAL DAY — read live from Hugging Face.
 *
 * The monthly path above is honest and blunt: it corrects a 25 km DAILY raster
 * with a 1° MONTHLY mean. E-040 (ml/plans/E040_daily_sst.md) removes that
 * mismatch on the measurement side. `sst/quarter/YEAR.i16` on the Hub stores
 * OISST at 0.25° PIXEL-MAJOR — value(px, day) at byte (px*days + day)*2 — so a
 * point query is ONE contiguous range read, and the transfer is bounded by the
 * question rather than by the archive: 730 bytes out of a 757 MB file for a
 * whole pixel-year. (Day-major would have forced a full frame per click and
 * killed the idea. This layout is the entire reason it is cheap.)
 *
 * Measured, not estimated (2026-08-18): CORS open, a browser range read on a
 * foreign origin returns HTTP 206 with exactly the bytes asked for, ~630 ms for
 * a pixel-year through the proxy, and the value round-trips bit-identically
 * against the source NetCDF (24.22 °C vs 24.22 °C).
 *
 * huggingface.co is the SECOND deliberate exception to CLAUDE.md §3, on the
 * same four conditions the Open-Meteo family was granted on: no key, CORS
 * verified rather than assumed, click-triggered single-point range reads and
 * never tile streaming, and it degrades — every failure here returns null and
 * the monthly path above serves the row instead, so a Hub outage costs
 * precision, not the feature. */
/* The "true SST anomaly" job's answer for an UNCAPPED pixel — there is nothing
 * to correct, and that is different from having failed to answer. */
const SST_ANOM_NONE = Object.freeze({ none: true });

const SST_DAILY_BASE =
  "https://huggingface.co/datasets/chfrank/earth-sst-daily/resolve/main/sst/quarter/";
const SST_DAILY_TIMEOUT_MS = 10000;   // a stalled read must not outlive the click
const SST_DAILY_CACHE_MAX = 50;       // ~50 × 730 B — a session of clicking costs nothing
/* Keyed `${year}/${px}`, insertion-ordered as an LRU. ONLY SUCCESSES ARE
 * CACHED: a failure here is a network event, not a fact about the archive, and
 * remembering it would make one dropped packet permanent for the session. The
 * next click retries, which is exactly the cost of one 730-byte request. */
const sstDailyCache = new Map();

/* The whole pixel-year in one ranged fetch. Returns an Int16Array of `days`
 * raw counts (scale 0.01 °C, `nodata` = -32768), or null on anything at all
 * going wrong. */
async function sstDailySeries(lon, lat, year) {
  // index.json goes through pixelJson, so it is fetched once per session and
  // shares the card's cache. Note that pixelJson remembers a FAILURE too —
  // deliberate here: if the index is unreachable the feature is simply off for
  // this session and every caller falls back, rather than each click paying a
  // round-trip to rediscover that.
  const idx = await pixelJson(SST_DAILY_BASE + "index.json");
  // A year absent from `years` is not on the Hub — asking for it would read
  // some other year's bytes at an offset computed from the wrong day count.
  const days = idx?.years?.[String(year)];
  if (!days || !idx.nx || !idx.ny) return null;
  const ix = Math.floor((lon - idx.west) / idx.dlon);
  const iy = Math.floor((lat - idx.south) / idx.dlat);
  if (ix < 0 || ix >= idx.nx || iy < 0 || iy >= idx.ny) return null;
  const px = iy * idx.nx + ix;
  const key = `${year}/${px}`;
  if (sstDailyCache.has(key)) {
    const hit = sstDailyCache.get(key);
    sstDailyCache.delete(key); sstDailyCache.set(key, hit);   // touch: most-recent last
    return hit;
  }
  const start = px * days * 2, bytes = days * 2;
  // AbortSignal.timeout is not in older Safari; the controller form is (same
  // reasoning as omGet). This deadline is SHORT where Open-Meteo's is long:
  // there is a good answer waiting on the other path, so waiting 45 s for a
  // better one would be spending the user's click on precision.
  const ctl = new AbortController();
  const timer = setTimeout(() => ctl.abort(), SST_DAILY_TIMEOUT_MS);
  try {
    const r = await fetch(`${SST_DAILY_BASE}${year}.i16`, {
      headers: { Range: `bytes=${start}-${start + bytes - 1}` },
      signal: ctl.signal,
    });
    // 206 is the contract. A 200 means the range was ignored and the body is
    // the whole 757 MB file — refuse it rather than read it.
    if (r.status !== 206) return null;
    const buf = await r.arrayBuffer();
    if (buf.byteLength !== bytes) return null;
    // Little-endian is explicit, not inherited from the host: Int16Array over
    // the buffer would read whatever the CPU does, and the file is LE by spec.
    const dv = new DataView(buf);
    const out = new Int16Array(days);
    for (let d = 0; d < days; d++) out[d] = dv.getInt16(d * 2, true);
    sstDailyCache.set(key, out);
    // Evict oldest first; the Map's insertion order is the LRU order.
    while (sstDailyCache.size > SST_DAILY_CACHE_MAX) {
      sstDailyCache.delete(sstDailyCache.keys().next().value);
    }
    return out;
  } catch {
    return null;
  } finally {
    clearTimeout(timer);
  }
}

/* The selected day's departure: OISST daily 0.25° minus the SAME CALENDAR
 * MONTH's 1991-2020 normal. Label-critical, and the card's note says so: this
 * is a DAILY reading against a MONTHLY normal, so the within-month part of the
 * seasonal cycle is not removed by it. That is still much closer to the
 * question than the monthly-vs-monthly answer, because the resolution mismatch
 * — a 1° cell averaging away the coastal upwelling tongue off Peru, measured
 * at 2.1 °C — dominates the within-month drift.
 *
 * Reads the climatology through pixelJson, deliberately NOT through the
 * sstNorm machinery: that object is a single-slot cache serving the monthly
 * path, and re-pointing it at another month here would silently answer the
 * hover probe with the wrong normals. */
async function sstDailyAnomaly(lon, lat, dateStr) {
  const date = (dateStr || state.date || "").slice(0, 10);
  const m = /^(\d{4})-(\d{2})-(\d{2})$/.exec(date);
  if (!m) return null;
  const year = Number(m[1]);
  const [idx, series, climIdx, clim] = await Promise.all([
    pixelJson(SST_DAILY_BASE + "index.json"),
    sstDailySeries(lon, lat, year),
    pixelJson("data/oisst_clim.json"),
    pixelJson(`data/oisst_clim/${m[2]}.json`),
  ]);
  if (!idx || !series || !climIdx || !clim?.values) return null;
  // Day index from Jan 1 in UTC — the file's own axis. Date arithmetic in
  // local time would put 1 January on 31 December west of Greenwich.
  const doy = Math.round(
    (Date.UTC(year, Number(m[2]) - 1, Number(m[3])) - Date.UTC(year, 0, 1)) / 86400000);
  if (!(doy >= 0 && doy < series.length)) return null;
  const raw = series[doy];
  if (raw === (idx.nodata ?? -32768)) return null;      // land, ice mask, or a gap
  const sst = raw * (idx.scale ?? 0.01);
  const norm = sampleGrid({
    west: climIdx.west, south: climIdx.south, east: climIdx.east, north: climIdx.north,
    dlon: climIdx.dlon, dlat: climIdx.dlat, nx: climIdx.nx, ny: climIdx.ny,
    values: clim.values,
  }, lon, lat);
  if (norm == null) return null;
  return { v: sst - norm, sst, norm, date, period: climIdx.period, daily: true };
}

async function showPixelState(carto) {
  const myTurn = ++pixelCardSeq;
  const lon = Cesium.Math.toDegrees(carto.longitude);
  const lat = Cesium.Math.toDegrees(carto.latitude);
  const coord = `${Math.abs(lat).toFixed(2)}°${lat >= 0 ? "N" : "S"} ` +
                `${Math.abs(lon).toFixed(2)}°${lon >= 0 ? "E" : "W"}`;
  pixelCardEl.innerHTML =
    `<div class="px-head"><strong>Pixel state</strong> · ${coord}` +
    `<button class="px-close" aria-label="Close">×</button></div>` +
    `<div class="px-body"><div class="px-loading">Reading this point…</div></div>`;
  pixelCardEl.classList.remove("hidden");
  pixelCardEl.querySelector(".px-close").addEventListener("click", () => {
    pixelCardEl.classList.add("hidden");
    hideProbe();
  });

  // Mark the tapped pixel on the globe — the card itself covers most of a
  // phone's globe view, so without this (and the rotate-into-view that
  // follows) the card could describe a point the user cannot see.
  {
    const top = topColormapLayer();
    let cell = null;
    if (top && !top.cfg.grid && (top.cfg.colormap || top.cfg.classmap)) {
      // the cell the CARD reads: classification rasters at native resolution,
      // continuous ones capped at level 4 (see pixelRasterValue)
      const z = (top.cfg.classmap || top.cfg.probeNative) ? top.cfg.maxLevel : Math.min(top.cfg.maxLevel, 4);
      const t = tileCoordsAt(lon, lat, z);
      cell = probeCellBounds(z, t.x, t.y, t.px, t.py);
    }
    showProbeMark({ lon, lat, cell });
    if (top?.cfg.grid) {
      // grid cell bounds need the loaded grid; upgrade the mark when it lands
      loadGridMonth(top.cfg).then((g) => {
        if (!g || pixelCardEl.classList.contains("hidden")) return;
        const ix = Math.floor((lon - g.west) / g.dlon), iy = Math.floor((lat - g.south) / g.dlat);
        if (ix < 0 || ix >= g.nx || iy < 0 || iy >= g.ny) return;
        showProbeMark({ lon, lat, cell: {
          west: g.west + ix * g.dlon, south: g.south + iy * g.dlat,
          east: g.west + (ix + 1) * g.dlon, north: g.south + (iy + 1) * g.dlat,
        } });
      });
    }
    ensureMarkVisible();
  }

  // Everything in parallel; the card renders once, complete.
  const rasterCfgs = PIXEL_RASTERS.map((id) => GIBS_LAYERS.find((l) => l.id === id));
  const gridCfgs = PIXEL_GRIDS.map((id) => GIBS_LAYERS.find((l) => l.id === id));
  /* THE CARD MUST RENDER, AND IT FILLS IN AS THE DATA ARRIVES. It composes
   * about twenty sources and used to show nothing until every one of them
   * answered, so the slowest single source decided whether the inspector
   * worked at all — and a stalled connection has no slowest, it simply never
   * returns. Reported 2026-08-16 ("load all data is no longer working"):
   * measured on the live site, one climate-api call hung and the card sat on
   * "Reading this point…" for 64 s.
   *
   * So the card appears at PIXEL_DEADLINE_MS with whatever is in hand, and
   * every source that lands after that redraws it. Not a fixed second pass —
   * Chris, 2026-08-16: "I hope that all the different data that come in can be
   * updated on the fly into the dialog box that renders after 2s." Two seconds
   * is short enough that most sources are still out at first draw, so a single
   * catch-up pass would leave the card visibly stale for as long as the
   * slowest one took. Redraws are coalesced over PIXEL_REDRAW_MS so a burst of
   * arrivals costs one rebuild, not fourteen.
   *
   * The promises are created ONCE and shared by every draw: re-requesting per
   * pass would multiply our Open-Meteo burst, which is rate-limited per IP
   * across the whole family. */
  /* [name, promise, empty] — `empty` is what this slot holds before its source
   * answers. It matters because the two collection sources are READ AS ARRAYS
   * by the draw (`rasters.forEach`, `grids.map`), and at a two-second first
   * paint they have usually not arrived: seeding them with null made the very
   * first draw throw, which the guard then turned into "something went wrong"
   * — an error message for the ordinary case of data still being in flight.
   * A scalar source's empty is null, which every section already tests for. */
  const rasterJob = Promise.all(
    rasterCfgs.map((cfg) => pixelRasterValue(cfg, lon, lat).catch(() => null)));
  const jobs = [
    ["satellite fields", rasterJob, []],
    // Costs NOTHING in the ordinary case: it waits for the anomaly probe and
    // returns null unless that probe came back CAPPED, so the two extra files
    // are fetched only when the picture genuinely could not carry the number.
    // The daily 0.25° read off the Hub is TRIED FIRST and the monthly archive
    // is the fallback, not the other way round: the daily value answers the
    // question that was asked (this point, this day) while the monthly one
    // answers a coarser neighbour of it. Any failure — no index, year not
    // uploaded, timeout, nodata — returns null and the monthly path serves the
    // row, so the Hub being down costs precision rather than the whole row.
    ["true SST anomaly", rasterJob.then(async (vals) => {
      const r = vals[PIXEL_RASTERS.indexOf("sst-anom")];
      // "Nothing to correct" is an ANSWER, not a failure. Returning null here
      // used to make deadNames() report "true SST anomaly didn't answer, so
      // that section is missing — tap the point again to retry" on every
      // ordinary uncapped pixel, promising a retry that could never produce
      // anything. The sentinel is truthy, so the late-source machinery counts
      // this job as answered; the card renders it as absence.
      if (!r || !r.cap) return SST_ANOM_NONE;
      const daily = await sstDailyAnomaly(lon, lat, state.date).catch(() => null);
      if (daily) return daily;
      await ensureSstNormals(state.date);
      return sstAnomalyAt(lon, lat);   // null now genuinely means "didn't answer"
    }).catch(() => null)],
    // {g, v}, not just v: the grid carries its own observation period/month, and
    // the row prints that — so the value and its date come from the same object.
    ["climate normals",
      Promise.all(gridCfgs.map((cfg) => loadGridMonth(cfg)
        .then((g) => (g ? { g, v: sampleGrid(g, lon, lat) } : null)).catch(() => null))), []],
    ["weather", fetchOpenMeteo(lon, lat)],
    ["air quality", fetchAirQuality(lon, lat)],
    ["river discharge", fetchRiver(lon, lat)],
    ["waves", fetchMarine(lon, lat)],
    ["climate outlook", fetchClimateWindow(lon, lat, ...CLIM_BASE_WIN)],
    ["climate outlook", fetchClimateWindow(lon, lat, ...CLIM_FUT_WIN)],
    ["ocean column", pixelJson("data/ocean_column.json")],
    ["ocean surface", pixelJson("data/ocean_surface.json")],
    ["stations", pixelJson("data/stations.geojson")],
    ["emitters", pixelJson("data/climatetrace.json")],
    ["Argo floats", pixelJson("data/argo.json")],
    // In the batch, not after it: the card must never serialise a round-trip.
    // Packed, so ~0.8 MB and cached thereafter.
    ["forest-loss drivers", loadGrid(GIBS_LAYERS.find((l) => l.id === "drivers")).catch(() => null)],
  ];
  const values = jobs.map(([, , empty]) => (empty === undefined ? null : empty));
  const done = new Array(jobs.length).fill(false);
  const pendingNames = () => [...new Set(jobs.filter((_, i) => !done[i]).map(([n]) => n))];

  /* A throw inside the draw must not leave "Reading this point…" on screen
   * forever — that is the same broken-looking outcome the deadline exists to
   * prevent, arrived at from the other side. So: say what happened, then
   * rethrow OUT of band. The rethrow matters as much as the message; swallowed
   * here, a rendering bug would show up as a slightly emptier card and the
   * suite's "loads without page errors" check would never see it. */
  const draw = (missing, final) => {
    if (myTurn !== pixelCardSeq) return;          // a newer point owns the card
    if (pixelCardEl.classList.contains("hidden")) return;
    try {
      drawPixelCard(values, missing, final);
    } catch (e) {
      const body = pixelCardEl.querySelector(".px-body");
      if (body) {
        body.innerHTML = `<div class="px-note px-late">Something went wrong ` +
          `composing this point. The details are in the browser console.</div>`;
      }
      setTimeout(() => { throw e; });
    }
  };

  let shown = false, queued = 0;
  const redraw = () => {
    // Coalesce: sources arrive in bursts (the whole Open-Meteo family answers
    // within a few hundred ms of each other) and each rebuild throws away the
    // card's DOM. One timer per burst, not one per arrival.
    if (!shown || queued) return;
    queued = setTimeout(() => {
      queued = 0;
      const pend = pendingNames();
      // Nothing pending means this is the last word, and "still waiting" would
      // become a lie. A source that never answered is named instead — silence
      // would leave a missing section indistinguishable from one that was
      // never built, and only this draw can tell those apart.
      draw(pend.length ? pend : deadNames(), !pend.length);
    }, PIXEL_REDRAW_MS);
  };
  /* Which sources never answered — for the LAST draw only, when "still
   * waiting" would be a lie. Restricted to the ones that were still out at
   * first paint, because plenty of sources return null perfectly correctly:
   * there are no waves inland and no river in mid-ocean, and naming those as
   * failures would cry wolf on every second click. A name counts as dead only
   * if every job carrying it came back empty (the CMIP6 outlook is two). */
  let lateAtFirst = [];
  const deadNames = () => [...new Set(lateAtFirst)]
    .filter((n) => jobs.every(([m, , empty], i) =>
      m !== n || values[i] == null || (Array.isArray(empty) && !values[i].some((v) => v != null))));

  const settled = jobs.map(([, p, empty], i) => Promise.resolve(p).catch(() => null).then((v) => {
    values[i] = v == null && empty !== undefined ? empty : v;
    done[i] = true;
    redraw();
    return v;
  }));
  const all = Promise.all(settled);

  // First paint: as soon as everything is in, or at the deadline — whichever
  // comes first. Below the deadline the common case is ONE draw, complete.
  await Promise.race([all, new Promise((r) => setTimeout(r, PIXEL_DEADLINE_MS))]);
  shown = true;
  const pend0 = pendingNames();
  lateAtFirst = pend0;
  draw(pend0.length ? pend0 : deadNames(), !pend0.length);
  if (!pend0.length) return;
  await all;
  // Flush the last coalesced redraw so this promise resolves with the card in
  // its final state — tests and callers may reasonably assume that.
  await new Promise((r) => setTimeout(r, PIXEL_REDRAW_MS + 20));

  function drawPixelCard(values, missing, final) {
    const [rasters, trueAnomRaw, grids, meteo, air, river, marine, climNow, climFut, oceanCol, oceanSurf, stations, trace, argo, driversGrid] = values;
    const trueAnom = trueAnomRaw && !trueAnomRaw.none ? trueAnomRaw : null;
    if (pixelCardEl.classList.contains("hidden")) return;   // closed while loading

    const sec = [];

    /* -- live weather now + the near future (the prediction axis) ------------ */
    if (meteo?.current) {
      const c = meteo.current;
      // Open-Meteo stamps its own current block; "live" in the heading was never
      // quite true — the model step behind it can be up to an hour old.
      const OFF = meteo.utc_offset_seconds || 0;
      const wNow = whenAt("instant", omUTC(c.time, OFF));
      let rows = pixelRow("Air temperature", `${fmtVal(c.temperature_2m)} °C`, wNow) +
        pixelRow("Wind", `${fmtVal(c.wind_speed_10m)} km/h from ${Math.round(c.wind_direction_10m)}°`, wNow) +
        pixelRow("Humidity · pressure", `${Math.round(c.relative_humidity_2m)} % · ${Math.round(c.pressure_msl)} hPa`, wNow) +
        (c.precipitation > 0 ? pixelRow("Precipitation now", `${fmtVal(c.precipitation)} mm/h`, wNow) : "");
      if (c.soil_moisture_0_to_1cm != null) {
        rows += pixelRow("Soil (top cm)", `${fmtVal(c.soil_moisture_0_to_1cm)} m³/m³ · ${fmtVal(c.soil_temperature_0cm)} °C`, wNow);
      }
      if (c.shortwave_radiation != null && c.shortwave_radiation > 0) {
        rows += pixelRow("Solar radiation", `${Math.round(c.shortwave_radiation)} W/m²`, wNow);
      }
      const mc = marine?.current;
      if (mc?.wave_height != null) {
        // separate API call, separate clock — its own stamp, not the weather one
        rows += pixelRow("Waves", `${fmtVal(mc.wave_height)} m` +
          (mc.wave_period != null ? ` every ${fmtVal(mc.wave_period)} s` : ""),
          whenAt("instant", omUTC(mc.time, marine.utc_offset_seconds || 0)));
      }
      const rd = river?.daily?.river_discharge?.[0];
      if (rd != null && rd > 0) {
        // GloFAS is a DAILY product: stamping it with the current hour would
        // claim a precision the number does not have.
        rows += pixelRow("River discharge", `${fmtVal(rd)} m³/s in this GloFAS cell`,
          whenAt("day", river.daily.time?.[0]));
      }
      if (Number.isFinite(meteo.elevation) && Math.abs(meteo.elevation) > 1) {
        // no stamp: terrain height is not an observation with a time
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
      // The forecast strip already prints a date per column, so it needs no stamp.
      sec.push(`<div class="px-sec"><div class="px-sec-title">Now &amp; next 7 days <span class="px-src">Open-Meteo</span></div>${rows}</div>`);
    }

    /* -- heat load on a body ------------------------------------------------- */
    // Air temperature is not what harms people: humidity, wind and sun decide
    // how much heat a body can shed, and the NIGHT decides whether it recovers.
    // Two numbers carry that — the day's felt peak, and whether the night stays
    // above 20 °C ("tropical night", the standard health threshold).
    if (meteo?.daily?.apparent_temperature_max) {
      const d = meteo.daily;
      const c = meteo.current;
      const OFF2 = meteo.utc_offset_seconds || 0;
      const fmtC = (v) => `${fmtVal(v)} °C`;
      let rows = "";
      if (c?.apparent_temperature != null) {
        const gap = c.apparent_temperature - c.temperature_2m;
        rows += pixelRow("Feels like now", `${fmtC(c.apparent_temperature)} · ` +
          `${gap >= 0 ? "+" : "−"}${fmtVal(Math.abs(gap))} vs air`,
          whenAt("instant", omUTC(c.time, OFF2)));
      }
      rows += pixelRow("Felt peak today", fmtC(d.apparent_temperature_max[0]),
                       whenAt("day", d.time[0]));
      // Tomorrow's daily minimum IS tonight's low: minima fall near dawn.
      const lowTonight = d.temperature_2m_min[1];
      if (lowTonight != null) {
        rows += pixelRow("Tonight's low", `${fmtC(lowTonight)}` +
          (lowTonight >= 20 ? ` · <strong>tropical night</strong>` : ""),
          whenAt("day", d.time[1]));
      }
      const tropical = d.temperature_2m_min.filter((v) => v != null && v >= 20).length;
      const peak = Math.max(...d.apparent_temperature_max.filter((v) => v != null));
      rows += pixelRow("Next 7 days", `felt peak ${fmtC(peak)} · ` +
        `${tropical} tropical night${tropical === 1 ? "" : "s"}`);
      // Name the index. A city climate-analysis map (PET) and UTCI model the
      // body's radiation balance explicitly and run warmer in sun, so their
      // 35 °C / 41 °C class limits do NOT transfer to this number.
      rows += `<div class="px-note">"Feels like" is Open-Meteo's apparent temperature —
        air temperature corrected for humidity, wind and sun. City heat maps use
        <em>PET</em> and heat warnings often use <em>UTCI</em>; both model a body's
        radiation balance explicitly and read warmer in direct sun, so their
        35/41 °C thresholds are not comparable with this figure. A night at or
        above 20 °C is the standard "tropical night", when the body gets no
        recovery.</div>`;
      sec.push(`<div class="px-sec"><div class="px-sec-title">Heat load ` +
        `<span class="px-src">Open-Meteo</span></div>${rows}</div>`);
    }

    /* -- air quality (CAMS via Open-Meteo) ----------------------------------- */
    if (air?.current && air.current.pm2_5 != null) {
      const a = air.current;
      const wAir = whenAt("instant", a.time);
      const rows =
        pixelRow("PM2.5 · PM10", `${fmtVal(a.pm2_5)} · ${fmtVal(a.pm10)} µg/m³`, wAir) +
        pixelRow("Ozone · NO₂", `${fmtVal(a.ozone)} · ${fmtVal(a.nitrogen_dioxide)} µg/m³`, wAir) +
        (a.european_aqi != null ? pixelRow("Air-quality index", `${Math.round(a.european_aqi)} (EU scale, lower is better)`, wAir) : "");
      sec.push(`<div class="px-sec"><div class="px-sec-title">Air quality <span class="px-src">CAMS</span></div>${rows}</div>`);
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
      const label = cfg.title.replace(/\s*\(.*\)$/, "");
      const w = whenOfGibs(cfg);
      if (r.label) return pixelRow(label, r.label, w);
      if (!r.cap) return pixelRow(label, `${fmtVal(r.v)} ${r.units}`, w);
      // A capped bin is the palette's edge, not a measurement. Where we can
      // compute the real figure, print it right here — the bound alone hides
      // the difference between "just over 3" and "nearly 5", which during an
      // El Nino is the entire question.
      const capped = pixelRow(label, `${r.cap.sign} ${fmtVal(r.cap.bound)} ${r.units}`, w);
      if (cfg.id !== "sst-anom" || !trueAnom) return capped;
      // The stamp follows the MEASUREMENT, not the row: the daily read is for
      // the exact selected day, the monthly fallback for a whole month, and
      // printing one granularity for both would misdate whichever lost.
      return capped + pixelRow("&nbsp;&nbsp;&#8627; actual departure",
        `<strong>${trueAnom.v >= 0 ? "+" : "−"}${fmtVal(Math.abs(trueAnom.v))} °C</strong>` +
        ` · ${fmtVal(trueAnom.sst)} vs ${fmtVal(trueAnom.norm)} normal`,
        trueAnom.daily ? whenAt("day", trueAnom.date) : whenAt("month", trueAnom.month));
    }).join("");
    if (rrows) {
      // The heading no longer claims a date. It used to say state.date for the
      // whole block, which was wrong for most of it: GRACE ends 2022-07, CERES
      // 2018-10, sea ice 2025-09, and the monthly layers snap to a first-of-month
      // — all of them were printed under today's date. Each row now says its own.
      // Two provenances, because they are two different measurements. The
      // daily line must say "daily reading, monthly normal" out loud: the
      // within-month part of the seasonal cycle is NOT removed by it, and a
      // reader who assumes a like-for-like anomaly would over-read a spring or
      // autumn value by the month's own drift.
      const anomNote = trueAnom
        ? `<div class="px-note">The anomaly palette stops at ±3 °C — its end bins are ` +
          `catch-alls, so the tile cannot express more and the probe can only say ` +
          `“≥ 3”. The actual departure above is computed instead of read off a ` +
          `colour: ` + (trueAnom.daily
            ? `NOAA OISST daily 0.25° for that exact day, read live from Hugging Face ` +
              `(730 bytes — one range read of the pixel's year), minus the same ` +
              `calendar month's ${trueAnom.period} normal. A daily reading against a ` +
              `monthly normal, so the within-month part of the seasonal cycle stays in.`
            : `NOAA OISST v2.1 monthly mean minus the same calendar month's ` +
              `${trueAnom.period} normal. Coarser than the raster it corrects — 1° and ` +
              `monthly, against 25 km and daily — and it is the newest month at or ` +
              `before the selected date, which in the weeks after a month ends is the ` +
              `PREVIOUS one. The stamp beside the value says which; that is why it can ` +
              `differ from the daily figure by more than rounding.`) +
          `</div>`
        : "";
      sec.push(`<div class="px-sec"><div class="px-sec-title">Satellite fields <span class="px-src">NASA GIBS</span></div>${rrows}${anomNote}</div>`);
    }

    /* -- why forest was lost here (categorical, so its own row) --------------- */
    {
      const g = driversGrid;
      const v = g && sampleGrid(g, lon, lat);
      const label = v == null ? null : gridClassLabel(g, v);
      if (label) {
        sec.push(`<div class="px-sec"><div class="px-sec-title">Forest loss ` +
          `<span class="px-src">WRI/DeepMind</span></div>` +
          pixelRow("Dominant driver", label, whenOfGrid(GIBS_LAYERS.find((l) => l.id === "drivers"), g)) +
          `</div>`);
      }
    }

    /* -- long-term normals (the memory channels) ----------------------------- */
    // No years in the labels any more — the stamp says them, read from the file
    // rather than typed here, so a re-bake over a longer record cannot leave a
    // stale span behind in the UI. (GPCP averages its whole record, not 1991–2020.)
    // Keyed by layer id, NOT a parallel array: PIXEL_GRIDS grew a fifth entry
    // (tides) while these stayed four long, and the card printed a literal
    // "undefined 1.20 undefined" for months. A map cannot drift out of step —
    // an id with no entry falls back to the layer's own title and units.
    const GRID_ROW = {
      oisst: ["SST annual mean", "°C"],
      gpcp: ["Precip normal (GPCP)", "mm/yr"],
      eobs: ["Precip normal (E-OBS)", "mm/yr"],
      meteoswiss: ["Precip normal (MeteoSwiss)", "mm/yr"],
      tides: ["Tidal range (EOT20)", "m"],
    };
    // Each normal states the years it averages, read from the baked `period` —
    // these are the one kind of row with no age at all, because a fixed span is
    // not "N years old", it simply is what it is.
    const grows = grids.map((r, i) => {
      if (r?.v == null) return "";
      const cfg = gridCfgs[i];
      const [title, unit] = GRID_ROW[cfg?.id] ||
        [String(cfg?.title || cfg?.id || "").replace(/\s*\(.*\)$/, ""), cfg?.units || ""];
      return pixelRow(title, `${fmtVal(r.v)} ${unit}`.trim(), whenOfGrid(cfg, r.g));
    }).join("");
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
        // GLORYS lands months behind real time; the stamp's age is the point.
        const wSurf = whenAt("month", oceanSurf.month);
        let rows = pixelRow("Surface current",
          `${fmtVal(spd)} m/s toward ${rose} (${Math.round(brg)}°)`, wSurf);
        if (oceanSurf.mld?.[i] != null) rows += pixelRow("Mixed-layer depth", `${oceanSurf.mld[i]} m`, wSurf);
        if (oceanSurf.zos?.[i] != null) {
          const z = oceanSurf.zos[i];
          rows += pixelRow("Sea surface height", `${z >= 0 ? "+" : "−"}${Math.abs(z)} cm`, wSurf);
        }
        sec.push(`<div class="px-sec"><div class="px-sec-title">Ocean circulation <span class="px-src">GLORYS</span></div>${rows}</div>`);
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
      // Each of these is an anomaly: the OBSERVED month minus a fixed 2004–18
      // baseline. The stamp is the observed month — the moving half, and the only
      // half that can go stale. The baseline is named in the legend row above.
      const wCol = whenAt("month", oceanCol.month);
      if (dT[0] != null) {
        rows += pixelRow("Surface vs normal",
          `<span class="${dT[0] >= 0 ? "px-warm" : "px-cool"}">${dT[0] >= 0 ? "+" : "−"}${fmtVal(Math.abs(dT[0]))} °C</span>`, wCol);
      }
      if (heat != null) {
        rows += pixelRow("Upper 700 m vs normal",
          `<span class="${heat >= 0 ? "px-warm" : "px-cool"}">${heat >= 0 ? "+" : "−"}${fmtVal(Math.abs(heat))} °C</span> stored heat`, wCol);
      }
      if (wl != null) rows += pixelRow("Warm-layer depth", `~${wl} m`, wCol);
      if (col.sNow[0] != null && col.sNorm[0] != null) {
        const ds = col.sNow[0] - col.sNorm[0];
        rows += pixelRow("Surface salinity", `${fmtVal(col.sNow[0])} PSU ` +
          `<span class="px-src">(${ds >= 0 ? "+" : "−"}${Math.abs(ds).toFixed(2)} vs normal${ds < -0.05 ? " — fresher" : ds > 0.05 ? " — saltier" : ""})</span>`, wCol);
      }
      sec.push(`<div class="px-sec"><div class="px-sec-title">Ocean column 0–2000 m <span class="px-src">Argo floats</span></div>${rows}</div>`);
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
        // Both windows are fixed spans, so these rows carry a period and NO age:
        // "2045–2049" is not "19 years away", it is simply the window projected.
        const wFut = winPeriod(CLIM_FUT_WIN);
        const rows =
          pixelRow("Temperature", `<span class="${mt >= 0 ? "px-warm" : "px-cool"}">${mt >= 0 ? "+" : "−"}${fmtVal(Math.abs(mt))} °C</span>` +
            ` <span class="px-src">(models ${tlo >= 0 ? "+" : "−"}${fmtVal(Math.abs(tlo))}…${thi >= 0 ? "+" : "−"}${fmtVal(Math.abs(thi))})</span>`, wFut) +
          pixelRow("Precipitation", `${mp >= 0 ? "+" : "−"}${Math.round(Math.abs(mp))} %` +
            ` <span class="px-src">(${plo >= 0 ? "+" : "−"}${Math.round(Math.abs(plo))}…${phi >= 0 ? "+" : "−"}${Math.round(Math.abs(phi))} %)</span>`, wFut);
        sec.push(`<div class="px-sec"><div class="px-sec-title">Projected change ` +
          `<span class="px-src">vs ${whenText(winPeriod(CLIM_BASE_WIN))} · CMIP6-HighResMIP · ${models.length} models</span>` +
          `</div>${rows}</div>`);
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
      // A station's position and name are not an observation — no stamp.
      if (best) nrows.push(pixelRow("Nearest monitoring site", `${best.properties.name} · ${Math.round(bd)} km`));
    }
    if (argo?.floats) {
      let n300 = 0, bd = Infinity, nearest = null;
      for (const f of argo.floats) {
        const d = haversineKm(lon, lat, f[0], f[1]);
        if (d < bd) { bd = d; nearest = f; }
        if (d < 300) n300++;
      }
      // Stamped with the NEAREST float's own last report, which is the freshness
      // that matters when you ask "is anything watching this water right now?".
      if (bd < 3000) nrows.push(pixelRow("Argo floats",
        `${n300} within 300 km · nearest ${Math.round(bd)} km`,
        whenAt("day", nearest?.[3])));
    }
    {
      // Climate TRACE ships one asset list PER YEAR and the app picks the year
      // from the date selector. This row read `trace.assets`, a key the baked file
      // has never had, so it silently never rendered; it reads the year now, and
      // that same year is what the row is stamped with.
      const tyear = trace?.years?.length ? climateTraceYear(trace) : null;
      const assets = tyear == null ? null : trace.assets_by_year?.[String(tyear)];
      if (assets?.length) {
        let best = null, bd = Infinity, n100 = 0;
        for (const a of assets) {
          const d = haversineKm(lon, lat, a[0], a[1]);
          if (d < bd) { bd = d; best = a; }
          if (d < 100) n100++;
        }
        if (bd < 500 && best) {
          nrows.push(pixelRow("Top-1000 emitters",
            `${n100} within 100 km · nearest: ${best[3]} (${fmtVal(best[2])} Mt/yr, ${Math.round(bd)} km)`,
            whenAt("year", String(tyear))));
        }
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
      // Two rows, because they have two different times: the RGI inventory is
      // compiled from imagery spanning decades and has no single honest date, so
      // the count carries no stamp — while the thinning RATE is measured over one
      // definite window, which travels in the file as `dhdt_period`.
      if (n) {
        nrows.push(pixelRow("Glaciers within 100 km", `${n}`));
        if (ndh) {
          nrows.push(pixelRow("Mean thickness change", `${fmtVal(dh / ndh)} m/yr`,
            whenAt("period", glacierData.dhdt_period)));
        }
      }
    }
    if (nrows.length) {
      sec.push(`<div class="px-sec"><div class="px-sec-title">Context nearby</div>${nrows.join("")}</div>`);
    }

    // Name what didn't arrive. A section that is absent because its source timed
    // out looks exactly like a section that was never built, and the difference
    // matters: one of them comes back if you tap again.
    if (missing.length) {
      const names = [...new Set(missing)];
      const one = names.length === 1;
      // At two seconds most of the twenty sources are still out, so the full
      // list would be a wall of text under a nearly empty card. Name a few and
      // count the rest: the point is that more is coming, not an inventory.
      const shownNames = names.length > 4 ? names.slice(0, 3) : names;
      const rest = names.length - shownNames.length;
      const list = shownNames.join(", ") + (rest ? ` and ${rest} more` : "");
      sec.push(`<div class="px-note ${final ? "px-late" : "px-loading"}">` + (final
        ? `${list} didn't answer, so ` +
          `${one ? "that section is" : "those sections are"} missing here. ` +
          `Tap the point again to retry.`
        : `Still loading ${list}…`) + `</div>`);
    }
    sec.push(`<div class="px-note">Channels &amp; roles: <a href="docs/PIXEL_STATE.md" target="_blank" rel="noopener">PIXEL_STATE.md</a> · weather &amp; forecast by <a href="https://open-meteo.com/" target="_blank" rel="noopener">Open-Meteo</a></div>`);
    // The body is rebuilt wholesale on every arrival, which resets the scroll
    // position — and the card is taller than a phone screen, so a user reading
    // the satellite rows would be yanked back to the top each time a source
    // landed. Hold the offset across the swap.
    const body = pixelCardEl.querySelector(".px-body");
    const top = body.scrollTop;
    body.innerHTML = sec.join("");
    if (top) body.scrollTop = top;
  }
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

/* Centered moving average over ~`days` of samples, null-tolerant: a value is
 * emitted when at least half the window is present, so array service gaps
 * thin the line rather than inventing bridges across them. The window is
 * derived from the data's own cadence (resolution_days), not a hard-coded
 * sample count — if the bake ever changes cadence, the smoothing follows. */
function movingMean(t, v, resolutionDays, days = 365) {
  const half = Math.max(1, Math.round(days / resolutionDays / 2));
  const out = new Array(v.length).fill(null);
  for (let i = 0; i < v.length; i++) {
    let s = 0, n = 0;
    for (let j = Math.max(0, i - half); j <= Math.min(v.length - 1, i + half); j++) {
      if (v[j] != null) { s += v[j]; n++; }
    }
    if (n >= half) out[i] = s / n;
  }
  return out;
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

  ctx.font = "10px system-ui, sans-serif";

  // The trend is what the tab exists to show, and the 10-day series buries it
  // in eddy noise (±5 Sv swings around an ~1 Sv/decade signal). So: raw data
  // faint and thin — still there, still hoverable, honest about the noise —
  // with a 12-month centred mean drawn bold on top. Annual smoothing is the
  // standard RAPID presentation because it also removes the seasonal cycle.
  const smooth = movingMean(t, moc, rapidData.resolution_days || 10, 365);

  function line(v, style, width, alpha = 1) {
    ctx.strokeStyle = style; ctx.lineWidth = width;
    ctx.globalAlpha = alpha; ctx.lineJoin = "round";
    ctx.beginPath();
    let started = false;
    for (let i = 0; i < v.length; i++) {
      if (v[i] == null) { started = false; continue; }
      if (!started) { ctx.moveTo(X(i), Y(v[i])); started = true; }
      else ctx.lineTo(X(i), Y(v[i]));
    }
    ctx.stroke();
    ctx.globalAlpha = 1;
  }

  function paint() {
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
    line(moc, "#3987e5", 1, 0.32);        // the 10-day data, quiet
    line(smooth, "#3987e5", 2.5);          // the 12-month mean, the story
    // tiny in-chart legend so the two weights read at a glance
    ctx.textAlign = "left"; ctx.textBaseline = "top";
    ctx.fillStyle = "#898781";
    ctx.fillText("— 12-month mean · thin: 10-day data", M.l + 4, M.t + 1);
  }
  paint();

  // hover: crosshair + tooltip (raw value plus the smoothed one beside it)
  const tip = document.getElementById("amoc-tooltip");
  canvas.onmousemove = (e) => {
    const rect = canvas.getBoundingClientRect();
    const px = e.clientX - rect.left;
    const i = Cesium.Math.clamp(Math.round(((px - M.l) / W) * (t.length - 1)), 0, t.length - 1);
    if (moc[i] == null) { tip.classList.add("hidden"); return; }
    paint();
    ctx.strokeStyle = "#52514e";
    ctx.lineWidth = 1;
    ctx.beginPath(); ctx.moveTo(X(i), M.t); ctx.lineTo(X(i), M.t + H); ctx.stroke();
    ctx.fillStyle = "#3987e5";
    ctx.beginPath(); ctx.arc(X(i), Y(moc[i]), 3.5, 0, Math.PI * 2); ctx.fill();
    tip.textContent = `${t[i]} · ${moc[i].toFixed(1)} Sv` +
      (smooth[i] != null ? ` · 12-mo mean ${smooth[i].toFixed(1)}` : "");
    tip.style.left = `${Math.min(Math.max(px - 40, 4), cssW - 110)}px`;
    tip.classList.remove("hidden");
  };
  canvas.onmouseleave = () => { tip.classList.add("hidden"); paint(); };
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
    if (document.getElementById("panel-temp").classList.contains("hidden")) return;
    drawTempChart();
  });
}

/* ---- Earth's energy imbalance (NOAA OHC → W/m²) ---- */
let eeiData = null;
async function loadEei() {
  if (eeiData) return;
  // ?v: cache-buster — GitHub Pages caches JSON for 10 min, and this file's
  // schema has grown (oni/volcanoes); a stale copy silently drops annotations
  eeiData = await (await fetch("data/eei.json?v=3")).json();
  // 0-700 m is strictly contained in 0-2000 m, so their difference is the
  // heat arriving in the 700-2000 m slab — the deep-penetration signal.
  // (Below 2000 m there is no yearly series: the abyss is surveyed by ship
  // once a decade; Purkey & Johnson put it near +0.06 W/m².)
  eeiData.ohcDeep = eeiData.y2000.map((yr, i) =>
    eeiData.ohc2000[i] - eeiData.ohc700[yr - eeiData.y700[0]]);
  document.querySelector("#eei-rate .stat-value").textContent = `+${eeiData.rate10.toFixed(2)}`;
  document.querySelector("#eei-total .stat-value").textContent = `+${eeiData.eei10.toFixed(2)}`;
  // the intro quotes the same numbers the tiles show, with their real window —
  // never a hardcoded "about 1 W/m²" detached from the data's date
  const w0 = eeiData.y2000[eeiData.y2000.length - 10], w1 = eeiData.y2000[eeiData.y2000.length - 1];
  document.getElementById("eei-intro-window").textContent = `${w0}–${w1}`;
  document.getElementById("eei-intro-total").textContent = `+${eeiData.eei10.toFixed(2)}`;
  document.getElementById("eei-intro-rate").textContent = `+${eeiData.rate10.toFixed(2)}`;
  // the famous Hiroshima equivalence, computed from the live number rather
  // than folklore: EEI [W/m²] × Earth surface [m²] ÷ 15 kt TNT [J].
  // Pure energy arithmetic (Hansen 2012 used ~4/s at the then-lower EEI).
  const bombsEl = document.getElementById("eei-bombs");
  if (bombsEl) {
    bombsEl.textContent = (eeiData.eei10 * 5.101e14 / 6.276e13).toFixed(1);
  }
  document.querySelector("#eei-zj .stat-value").textContent = `+${Math.round(eeiData.zj_gained)}`;
  document.querySelector("#eei-zj .stat-sub").textContent = `ZJ gained 0–2000 m since ${eeiData.zj_since}`;
  document.getElementById("eei-legend").innerHTML =
    `<span style="color:#3987e5">━ 0–700 m (since 1955)</span>` +
    `<span style="color:#d95926">━ 0–2000 m (since 2005)</span>` +
    `<span style="color:#a371f7">━ 700–2000 m slab (= difference)</span>`;
  document.getElementById("eei-rate-legend").innerHTML =
    `<span style="color:#3987e5">━ from 0–700 m OHC</span>` +
    `<span style="color:#d95926">━ from 0–2000 m OHC</span>` +
    `<span style="color:#a371f7">━ 700–2000 m slab</span>` +
    `<span style="color:#c9c4b4">┄ human push (total ERF)</span>` +
    `<span style="color:#69a765">┄ natural push (solar+volcanic)</span>` +
    `<span style="color:#e3b341">▮ El Niño (moderate+)</span>` +
    `<span style="color:#2fbfb4">▮ La Niña (moderate+)</span>` +
    `<span style="color:#8b949e">▲ eruption (Agung '63 · El Chichón '82 · Pinatubo '91 · Hunga Tonga '22)</span>`;
  drawEeiChart();
  drawEeiRateChart();
  window.addEventListener("resize", () => {
    if (document.getElementById("panel-energy").classList.contains("hidden")) return;
    drawEeiChart();
    drawEeiRateChart();
  });
  // Slider is the source of truth (1–20 yr); the preset buttons just snap it,
  // mirroring the main page's Aggregate slider + presets.
  const slider = document.getElementById("eei-smooth-slider");
  const applySmooth = () => {
    eeiSmooth = Number(slider.value);
    document.getElementById("eei-smooth-value").textContent = `${eeiSmooth} yr`;
    for (const b of document.querySelectorAll("#eei-smooth button")) {
      b.classList.toggle("active", Number(b.dataset.n) === eeiSmooth);
    }
    drawEeiChart();
    drawEeiRateChart();
  };
  slider.addEventListener("input", applySmooth);
  document.getElementById("eei-smooth").addEventListener("click", (e) => {
    const n = Number(e.target.getAttribute?.("data-n"));
    if (!n) return;
    slider.value = String(n);
    applySmooth();
  });
}

/* Smoothing for both Energy charts, chosen by the 1y/3y/5y/10y presets.
 * Ledger: centred moving average over N years (1 = raw). Rate: the heating
 * rate computed over the same window — a centred OLS slope for N ≥ 3, the
 * plain year-over-year gain for N = 1 — in W/m² of the whole Earth
 * (1e22 J/yr = 0.6213 W/m²). Computed client-side from the raw OHC series so
 * the control is instant. */
let eeiSmooth = 1;
const EEI_W_PER = 0.6213;
function movAvg(vals, n) {
  if (n <= 1) return vals;
  const h = Math.floor(n / 2);
  return vals.map((_, i) => {
    const a = Math.max(0, i - h), b = Math.min(vals.length, i + (n - h));
    const win = vals.slice(a, b).filter((v) => v != null);
    return win.length ? win.reduce((s, v) => s + v, 0) / win.length : null;
  });
}
function rateSeries(years, vals, n) {
  if (n <= 1) {
    return vals.map((v, i) =>
      i && v != null && vals[i - 1] != null ? (v - vals[i - 1]) * EEI_W_PER : null);
  }
  const h = Math.floor(n / 2);
  return vals.map((_, i) => {
    const a = Math.max(0, i - h), b = Math.min(vals.length, i + (n - h));
    const xs = years.slice(a, b), ys = vals.slice(a, b);
    if (xs.length < 3) return null;
    const mx = xs.reduce((s, v) => s + v, 0) / xs.length;
    const my = ys.reduce((s, v) => s + v, 0) / ys.length;
    let num = 0, den = 0;
    for (let k = 0; k < xs.length; k++) {
      num += (xs[k] - mx) * (ys[k] - my);
      den += (xs[k] - mx) ** 2;
    }
    return den ? (num / den) * EEI_W_PER : null;
  });
}

/* ENSO + volcano annotations shared by both Energy charts. Year bands are
 * tinted by DJF ONI (deeper colour = stronger event); eruptions get a small
 * triangle at the top of the plot (names live in the legend and tooltip). */
function ensoOf(yr) {
  const v = eeiData?.oni?.[String(yr)];
  return v == null ? null : v;
}
function drawEnsoBands(ctx, X, M, H, yr0, yr1) {
  for (let yr = yr0; yr <= yr1; yr++) {
    const v = ensoOf(yr);
    // Shade only MODERATE-or-stronger events (|ONI| >= 1.0). At the official
    // "weak event" threshold of 0.5, 70% of winters qualify — DJF is ENSO's
    // peak season and neutral is actually the minority state — which painted
    // the chart as if every year were an event. Weak years still show up in
    // the hover tooltip; the bands are reserved for the ~30% that match the
    // public sense of "an El Niño year".
    if (v == null || Math.abs(v) < 1.0) continue;
    const w = X(yr + 0.5) - X(yr - 0.5);
    // Full-height tint, strong enough to survive a dark theme. Hues chosen
    // to collide with NEITHER data line: El Nino is amber-gold (warm, but
    // nothing like the orange-red 0-2000 m line), La Nina is teal (cool, but
    // nothing like the blue 0-700 m line).
    const a = Math.min(0.30, 0.04 + 0.09 * Math.abs(v));
    ctx.fillStyle = v > 0 ? `rgba(224,177,61,${a})` : `rgba(45,190,180,${a})`;
    ctx.fillRect(X(yr - 0.5), M.t, w, H);
    // ...plus an unmissable solid event strip along the bottom of the plot
    ctx.fillStyle = v > 0 ? "#e3b341" : "#2fbfb4";
    ctx.fillRect(X(yr - 0.5), M.t + H - 3, w, 3);
  }
}
function drawVolcanoes(ctx, X, M, H, yr0, yr1, cssW) {
  let i = 0;
  for (const v of eeiData?.volcanoes || []) {
    if (v.y < yr0 || v.y > yr1) continue;
    const x = X(v.y);
    // dashed marker line through the plot, triangle + NAME at the top
    ctx.strokeStyle = "rgba(139,148,158,0.7)"; ctx.lineWidth = 1;
    ctx.setLineDash([3, 3]);
    ctx.beginPath(); ctx.moveTo(x, M.t + 10); ctx.lineTo(x, M.t + H); ctx.stroke();
    ctx.setLineDash([]);
    ctx.fillStyle = "#c9d1d9";
    ctx.beginPath();
    ctx.moveTo(x, M.t + 1); ctx.lineTo(x - 4, M.t + 8); ctx.lineTo(x + 4, M.t + 8);
    ctx.closePath(); ctx.fill();
    ctx.font = "9px system-ui, sans-serif";
    ctx.textBaseline = "top"; ctx.textAlign = "center";
    const tw = ctx.measureText(v.n).width;
    const tx = Math.max(M.l + tw / 2, Math.min(cssW - 6 - tw / 2, x));
    ctx.fillText(v.n, tx, M.t + 10 + (i % 2) * 10);   // staggered rows: no overlap
    ctx.font = "10px system-ui, sans-serif";
    i++;
  }
}
function annotationLines(yr) {
  const out = [];
  const v = ensoOf(yr);
  if (v != null && Math.abs(v) >= 0.5) {
    const str = Math.abs(v) >= 1.5 ? "strong" : Math.abs(v) >= 1.0 ? "moderate" : "weak";
    out.push(`${str} ${v > 0 ? "El Niño" : "La Niña"} (ONI ${v > 0 ? "+" : ""}${v.toFixed(1)})`);
  }
  const volc = (eeiData?.volcanoes || []).find((x) => x.y === yr);
  if (volc) out.push(`▲ ${volc.n} eruption`);
  return out;
}

/* The imbalance itself over time: the rolling 5-yr heating rate in W/m² of
 * the whole Earth — i.e. the SLOPE of the OHC chart, drawn explicitly so the
 * "does it vary?" question answers itself. */
function drawEeiRateChart() {
  const d = eeiData;
  const r700 = rateSeries(d.y700, d.ohc700, eeiSmooth);
  const r2000 = rateSeries(d.y2000, d.ohc2000, eeiSmooth);
  const rDeep = rateSeries(d.y2000, d.ohcDeep, eeiSmooth);
  const canvas = document.getElementById("eei-rate-chart");
  const wrap = canvas.parentElement;
  // ERF context curves (the "push"): anthropogenic total and natural
  // (solar+volcanic), AR6/IGCC annual series baked into eei.json. The chart
  // grows taller when they're present — the push tops out near 3 W/m².
  const hasErf = Array.isArray(d.erf_years) && d.erf_years.length > 0;
  const cssW = wrap.clientWidth, cssH = hasErf ? 210 : 160;
  const dpr = window.devicePixelRatio || 1;
  canvas.width = cssW * dpr; canvas.height = cssH * dpr;
  canvas.style.width = cssW + "px"; canvas.style.height = cssH + "px";
  const ctx = canvas.getContext("2d"); ctx.scale(dpr, dpr);
  const M = { l: 32, r: 8, t: 14, b: 18 };
  const W = cssW - M.l - M.r, H = cssH - M.t - M.b;
  const yr0 = d.y700[0], yr1 = d.y700[d.y700.length - 1];
  const all = [...r700, ...r2000, ...rDeep,
               ...(hasErf ? d.erf_anthro : []), ...(hasErf ? d.erf_natural : [])]
    .filter((v) => v != null);
  const v0 = Math.min(-0.25, Math.floor(Math.min(...all) * 4) / 4);
  const v1 = Math.ceil(Math.max(...all) * 4) / 4;
  const X = (yr) => M.l + ((yr - yr0) / (yr1 - yr0)) * W;
  const Y = (v) => M.t + (1 - (v - v0) / (v1 - v0)) * H;
  const line = (years, vals) => {
    ctx.beginPath(); let started = false;
    for (let i = 0; i < years.length; i++) {
      if (vals[i] == null) { started = false; continue; }
      if (!started) { ctx.moveTo(X(years[i]), Y(vals[i])); started = true; }
      else ctx.lineTo(X(years[i]), Y(vals[i]));
    }
    ctx.stroke();
  };
  const draw = (hoverYr) => {
    ctx.clearRect(0, 0, cssW, cssH);
    ctx.font = "10px system-ui, sans-serif";
    drawEnsoBands(ctx, X, M, H, yr0, yr1);
    for (let v = v0; v <= v1 + 1e-9; v += 0.25) {
      ctx.strokeStyle = Math.abs(v) < 1e-9 ? "#4a4a47" : "#2c2c2a"; ctx.lineWidth = 1;
      ctx.beginPath(); ctx.moveTo(M.l, Y(v)); ctx.lineTo(cssW - M.r, Y(v)); ctx.stroke();
      ctx.fillStyle = "#898781"; ctx.textAlign = "right"; ctx.textBaseline = "middle";
      ctx.fillText(v.toFixed(2), M.l - 4, Y(v));
    }
    ctx.fillStyle = "#898781"; ctx.textAlign = "left"; ctx.textBaseline = "top";
    ctx.fillText("W/m² (whole Earth)", M.l + 2, 2);
    ctx.textAlign = "center";
    for (let yr = 1960; yr <= yr1; yr += 20) {
      ctx.fillText(String(yr), X(yr), M.t + H + 5);
    }
    drawVolcanoes(ctx, X, M, H, yr0, yr1, cssW);
    if (hasErf) {
      // the push, drawn beneath the measured lines: human forcing climbs to
      // ~3 W/m²; natural (solar+volcanic) hugs zero except eruption dips —
      // which land exactly on the ▲ markers. The gap between the push and
      // the measured EEI below it is the planet's radiative answer to the
      // warming already realized — NOT a natural cooling term.
      ctx.lineWidth = 1.3; ctx.lineJoin = "round";
      ctx.setLineDash([5, 4]);                 // both push curves dashed: they
      ctx.strokeStyle = "#c9c4b4"; line(d.erf_years, d.erf_anthro);   // are context,
      ctx.strokeStyle = "#69a765"; line(d.erf_years, d.erf_natural);  // not measurements
      ctx.setLineDash([]);
    }
    ctx.lineWidth = 1.8; ctx.lineJoin = "round";
    ctx.strokeStyle = "#3987e5"; line(d.y700, r700);
    ctx.strokeStyle = "#d95926"; line(d.y2000, r2000);
    ctx.strokeStyle = "#a371f7"; line(d.y2000, rDeep);
    if (hoverYr != null) {
      ctx.strokeStyle = "#52514e"; ctx.lineWidth = 1;
      ctx.beginPath(); ctx.moveTo(X(hoverYr), M.t); ctx.lineTo(X(hoverYr), M.t + H); ctx.stroke();
    }
  };
  draw(null);
  const tip = document.getElementById("eei-rate-tooltip");
  canvas.onmousemove = (e) => {
    const rect = canvas.getBoundingClientRect();
    const yr = Math.round(yr0 + ((e.clientX - rect.left - M.l) / W) * (yr1 - yr0));
    const i7 = d.y700.indexOf(yr), i2 = d.y2000.indexOf(yr);
    if (i7 < 0 || r700[i7] == null) { tip.classList.add("hidden"); return; }
    draw(yr);
    const bits = [`<strong>${yr}</strong>`, `0–700 m: ${r700[i7].toFixed(2)} W/m²`];
    if (i2 >= 0 && r2000[i2] != null) bits.push(`0–2000 m: ${r2000[i2].toFixed(2)} W/m²`);
    if (i2 >= 0 && rDeep[i2] != null) bits.push(`700–2000 m slab: ${rDeep[i2].toFixed(2)} W/m²`);
    if (hasErf) {
      const ie = d.erf_years.indexOf(yr);
      if (ie >= 0) {
        bits.push(`human push (ERF): ${d.erf_anthro[ie].toFixed(2)} W/m²`);
        bits.push(`natural push: ${d.erf_natural[ie].toFixed(2)} W/m²`);
      }
    }
    bits.push(...annotationLines(yr));
    tip.innerHTML = bits.join("<br/>");
    tip.style.left = `${Math.min(e.clientX - rect.left + 12, cssW - 150)}px`;
    tip.style.top = "8px";
    tip.classList.remove("hidden");
  };
  canvas.onmouseleave = () => { tip.classList.add("hidden"); draw(null); };
}

function drawEeiChart() {
  const d = eeiData;
  // The ledger always draws RAW yearly values: a cumulative total is already
  // an integral — smoothing applies to its derivative (the rate chart below).
  const s700 = d.ohc700, s2000 = d.ohc2000, sDeep = d.ohcDeep;
  const r700 = rateSeries(d.y700, d.ohc700, Math.max(eeiSmooth, 1));
  const r2000 = rateSeries(d.y2000, d.ohc2000, Math.max(eeiSmooth, 1));
  const rDeep = rateSeries(d.y2000, d.ohcDeep, Math.max(eeiSmooth, 1));
  const canvas = document.getElementById("eei-chart");
  const wrap = canvas.parentElement;
  const cssW = wrap.clientWidth, cssH = 190;
  const dpr = window.devicePixelRatio || 1;
  canvas.width = cssW * dpr; canvas.height = cssH * dpr;
  canvas.style.width = cssW + "px"; canvas.style.height = cssH + "px";
  const ctx = canvas.getContext("2d"); ctx.scale(dpr, dpr);
  const M = { l: 32, r: 8, t: 14, b: 18 };
  const W = cssW - M.l - M.r, H = cssH - M.t - M.b;
  const yr0 = d.y700[0], yr1 = d.y700[d.y700.length - 1];
  const all = [...s700, ...s2000, ...sDeep].filter((v) => v != null);
  const v0 = Math.floor(Math.min(...all) / 5) * 5, v1 = Math.ceil(Math.max(...all) / 5) * 5;
  const X = (yr) => M.l + ((yr - yr0) / (yr1 - yr0)) * W;
  const Y = (v) => M.t + (1 - (v - v0) / (v1 - v0)) * H;
  const line = (years, vals) => {
    ctx.beginPath();
    years.forEach((yr, i) => (i ? ctx.lineTo(X(yr), Y(vals[i])) : ctx.moveTo(X(yr), Y(vals[i]))));
    ctx.stroke();
  };
  const draw = (hoverYr) => {
    ctx.clearRect(0, 0, cssW, cssH);
    ctx.font = "10px system-ui, sans-serif";
    drawEnsoBands(ctx, X, M, H, yr0, yr1);
    for (let v = v0; v <= v1; v += 10) {
      ctx.strokeStyle = v === 0 ? "#4a4a47" : "#2c2c2a"; ctx.lineWidth = 1;
      ctx.beginPath(); ctx.moveTo(M.l, Y(v)); ctx.lineTo(cssW - M.r, Y(v)); ctx.stroke();
      ctx.fillStyle = "#898781"; ctx.textAlign = "right"; ctx.textBaseline = "middle";
      ctx.fillText(String(v), M.l - 4, Y(v));
    }
    ctx.textAlign = "left"; ctx.textBaseline = "top";
    ctx.fillStyle = "#898781";
    ctx.fillText("accumulated heat, ×10²² J vs baseline", M.l + 2, 2);
    ctx.textAlign = "center";
    for (let yr = 1960; yr <= yr1; yr += 20) {
      ctx.fillStyle = "#898781"; ctx.fillText(String(yr), X(yr), M.t + H + 5);
    }
    drawVolcanoes(ctx, X, M, H, yr0, yr1, cssW);
    ctx.lineWidth = 1.8; ctx.lineJoin = "round";
    ctx.strokeStyle = "#3987e5"; line(d.y700, s700);
    ctx.strokeStyle = "#d95926"; line(d.y2000, s2000);
    ctx.strokeStyle = "#a371f7"; line(d.y2000, sDeep);
    if (hoverYr != null) {
      ctx.strokeStyle = "#52514e"; ctx.lineWidth = 1;
      ctx.beginPath(); ctx.moveTo(X(hoverYr), M.t); ctx.lineTo(X(hoverYr), M.t + H); ctx.stroke();
    }
  };
  draw(null);
  const tip = document.getElementById("eei-tooltip");
  canvas.onmousemove = (e) => {
    const rect = canvas.getBoundingClientRect();
    const yr = Math.round(yr0 + ((e.clientX - rect.left - M.l) / W) * (yr1 - yr0));
    const i7 = d.y700.indexOf(yr), i2 = d.y2000.indexOf(yr);
    if (i7 < 0) { tip.classList.add("hidden"); return; }
    draw(yr);
    const bits = [`<strong>${yr}</strong>`, `0–700 m: ${s700[i7].toFixed(1)}×10²² J`];
    if (i2 >= 0 && s2000[i2] != null) bits.push(`0–2000 m: ${s2000[i2].toFixed(1)}×10²² J`);
    if (i2 >= 0 && sDeep[i2] != null) bits.push(`700–2000 m slab: ${sDeep[i2].toFixed(1)}×10²² J`);
    const rate = i2 >= 0 && r2000[i2] != null ? r2000[i2] : r700[i7];
    if (rate != null) bits.push(`heating ≈ ${rate >= 0 ? "+" : ""}${rate.toFixed(2)} W/m²`);
    bits.push(...annotationLines(yr));
    tip.innerHTML = bits.join("<br/>");
    tip.style.left = `${Math.min(e.clientX - rect.left + 12, cssW - 150)}px`;
    tip.style.top = "8px";
    tip.classList.remove("hidden");
  };
  canvas.onmouseleave = () => { tip.classList.add("hidden"); draw(null); };
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

/* -------------------------------------------------------------------- tides
 * The tide renders ON THE GLOBE: a full-globe rectangle primitive whose
 * texture is a 360x180 canvas repainted from EOT20's harmonic constants —
 * h(cell, t) = sum_c A*cos(speed*t + V0 - G). Per frame that is 5 cos/sin
 * calls plus a fused multiply-add over precomputed P = A*cosG, Q = A*sinG
 * fields (~0.6 MFLOP), double-buffered so Cesium re-uploads the texture
 * only when a new frame was actually painted (assigning the OTHER canvas
 * to the material uniform is what signals the change). The primitive
 * carries the CITY_PICK sentinel, so every click handler sees through it —
 * the pixel inspector and the value probe keep working over the ocean.
 * The Tides tab is the control room (clock, speed, spring/neap, a tapped
 * point's 3-day curve); the picture itself lives with every other dataset:
 * on the globe. Clock starts at real now — in phase with today's ocean
 * (sans nodal f/u, +-10-15% on lunar amplitudes, stated in the tab). */

let tideData = null;          // baked constituents (lazy-fetched)
let tideFields = null;        // per-constituent {P, Q} + mask + cell areas
let tideSim = { t: 0, playing: true, speed: 3600, raf: 0, wall: 0,
                markCell: null, markPlace: null, markLon: 0, markLat: 0 };
const tideLive = { on: false, prim: null, mat: null, labels: null,
                   front: null, back: null, img: null, lastPaint: 0, opacity: 0.85 };
/* Colour scale, MEASURED not chosen (CLAUDE.md: no hand-picked thresholds).
 * Sampling every ocean cell every 3 h over a fortnight, |tide height| has
 * median 16 cm, 90th pct 55 cm, 95th 75 cm, 99th 136 cm; the largest value
 * anywhere at any time is 6.2 m. At the old ±2.5 m, 99.85% of the ocean sat
 * in the middle of the ramp and the open ocean — where the amphidromic
 * rotation actually lives — rendered nearly flat (±0.5 m spanned just 20% of
 * the colours). At ±1 m it spans 46%, and the 2.3% beyond still separates,
 * because the ramp is tanh-compressed rather than clipped: shelf seas keep
 * their structure, they only saturate. Reported 2026-08-08 as "how can the
 * global ocean swing by 0.3 m and Peniche by 3 m?" — the answer was true, but
 * the scale made the ocean look motionless. */
const TD_RANGE_CM = 100;

function tideAstro(ms) {
  // Low-precision ephemeris (~1 deg): sub-lunar/sub-solar points + phase.
  const d = (ms - Date.UTC(2000, 0, 1, 12)) / 86400000;   // days since J2000
  const rad = Math.PI / 180;
  const wrap = (x) => ((x % 360) + 360) % 360;
  const Lp = wrap(218.316 + 13.176396 * d);
  const Mp = wrap(134.963 + 13.064993 * d);
  const F = wrap(93.272 + 13.229350 * d);
  const lam = Lp + 6.289 * Math.sin(Mp * rad);
  const bet = 5.128 * Math.sin(F * rad);
  const eps = 23.439 - 0.0000004 * d;
  const sinDec = Math.sin(bet * rad) * Math.cos(eps * rad) +
    Math.cos(bet * rad) * Math.sin(eps * rad) * Math.sin(lam * rad);
  const decM = Math.asin(sinDec) / rad;
  const raM = Math.atan2(
    Math.sin(lam * rad) * Math.cos(eps * rad) - Math.tan(bet * rad) * Math.sin(eps * rad),
    Math.cos(lam * rad)) / rad;
  const g = wrap(357.529 + 0.98560028 * d);
  const q = wrap(280.459 + 0.98564736 * d);
  const lamS = q + 1.915 * Math.sin(g * rad) + 0.020 * Math.sin(2 * g * rad);
  const decS = Math.asin(Math.sin(eps * rad) * Math.sin(lamS * rad)) / rad;
  const raS = Math.atan2(Math.cos(eps * rad) * Math.sin(lamS * rad),
    Math.cos(lamS * rad)) / rad;
  const gmst = wrap(280.46062 + 360.98564737 * d);
  const lonOf = (ra) => wrap(ra - gmst + 540) - 180;
  const elong = wrap(lam - lamS);
  return { moon: { lon: lonOf(raM), lat: decM }, sun: { lon: lonOf(raS), lat: decS },
           elong, spring: Math.abs(Math.cos(elong * rad)) };
}

function tidePrepare(data) {
  const n = data.nx * data.ny;
  const rad = Math.PI / 180;
  const fields = { consts: [], mask: new Uint8Array(n), n, epochMs: Date.parse(data.epoch) };
  // Ocean fraction of each 1° cell (from the 0.125° source; baked 2026-08-07).
  // Used as per-cell alpha so coastal cells feather instead of stamping a
  // fully opaque 1° square over the land they mostly are (the British-Isles
  // "red over Birmingham" artifact). Older bakes lack it → alpha 1 everywhere.
  fields.frac = new Float32Array(n).fill(1);
  if (data.frac) for (let i = 0; i < n; i++) fields.frac[i] = data.frac[i] || 0;
  for (const c of data.constituents) {
    const P = new Float32Array(n), Q = new Float32Array(n);
    for (let i = 0; i < n; i++) {
      const a = c.amp[i], g = c.phase[i];
      if (a == null || g == null) continue;
      P[i] = a * Math.cos(g * rad);
      Q[i] = a * Math.sin(g * rad);
      fields.mask[i] = 1;
    }
    fields.consts.push({ speed: c.speed, V0: c.V0, P, Q });
  }
  fields.area = new Float32Array(n);
  for (let iy = 0; iy < data.ny; iy++) {
    const lat = data.south + (iy + 0.5) * (data.north - data.south) / data.ny;
    const a = 111.32 * Math.cos(lat * rad) * 110.57;    // km^2 per cell
    for (let ix = 0; ix < data.nx; ix++) fields.area[iy * data.nx + ix] = a;
  }
  return fields;
}

/* The largest range this point can ever produce: every constituent's crest
 * landing together (2 × the sum of the amplitudes). Printed beside the 3-day
 * range so a NEAP window — when S2 and N2 oppose M2 and the swing roughly
 * halves — reads as the moon's doing rather than as a broken layer. */
function tideSpringRange(i) {
  if (!tideFields || !tideFields.mask[i]) return 0;
  let s = 0;
  for (const c of tideFields.consts) s += Math.hypot(c.P[i], c.Q[i]);
  return 2 * s / 100;
}

function tideHeightAt(i, ms) {
  if (!tideFields || !tideFields.mask[i]) return NaN;
  const th = (ms - tideFields.epochMs) / 3600000;
  const rad = Math.PI / 180;
  let h = 0;
  for (const c of tideFields.consts) {
    const phi = (c.speed * th + c.V0) * rad;
    h += Math.cos(phi) * c.P[i] + Math.sin(phi) * c.Q[i];
  }
  return h;
}

/* d(height)/d(time) in cm per hour, ANALYTIC — the same sum differentiated,
 * not a finite difference. High and low water are exactly where this crosses
 * zero, so the turning points below are found by root-finding on an exact
 * function rather than by hunting for the largest sample. */
function tideRateAt(i, ms) {
  if (!tideFields || !tideFields.mask[i]) return NaN;
  const th = (ms - tideFields.epochMs) / 3600000;
  const rad = Math.PI / 180;
  let d = 0;
  for (const c of tideFields.consts) {
    const phi = (c.speed * th + c.V0) * rad;
    d += (Math.cos(phi) * c.Q[i] - Math.sin(phi) * c.P[i]) * c.speed * rad;
  }
  return d;
}

/* The turning points ahead of `fromMs`: high water where the rate goes + → −,
 * low water where it goes − → +. Scanned at 15-minute steps — the fastest
 * constituent here is S2 (30°/h, a 12-hour period), so no extremum can hide
 * between samples — then bisected to about a second. Returns
 * [{ms, cm, high}] in time order. */
function tideExtrema(i, fromMs, hours = 72, limit = 40) {
  const out = [];
  if (!tideFields || !tideFields.mask[i]) return out;
  const STEP = 15 * 60000;
  const end = fromMs + hours * 3600000;
  let t0 = fromMs, r0 = tideRateAt(i, t0);
  for (let t = fromMs + STEP; t <= end && out.length < limit; t += STEP) {
    const r1 = tideRateAt(i, t);
    if ((r0 > 0) !== (r1 > 0)) {
      let a = t0, b = t, ra = r0;
      for (let k = 0; k < 24; k++) {
        const m = (a + b) / 2, rm = tideRateAt(i, m);
        if ((ra > 0) !== (rm > 0)) b = m; else { a = m; ra = rm; }
      }
      const ms = (a + b) / 2;
      out.push({ ms, cm: tideHeightAt(i, ms), high: r0 > 0 });
    }
    t0 = t; r0 = r1;
  }
  return out;
}

/* ---- clock: the tide is told in the LOCAL time of the point ---------------
 *
 * "Next high water 16:42" is only useful in the time the person standing on
 * that beach reads off their phone. Three sources, best first:
 *
 *   1. the POINT's zone, from Open-Meteo's `timezone=auto` (§3's Open-Meteo
 *      exception: key-free, single-point, click-triggered). It answers with
 *      the IANA zone AND `utc_offset_seconds` already resolved for the date,
 *      so summer time is handled by the same database the coastline uses —
 *      Europe/Lisbon is UTC+1 today, WEST, and this says so.
 *   2. the BROWSER's zone, shown instantly while (1) is in flight, so the
 *      panel never waits on a network call to display a time.
 *   3. UTC, if a point somehow has neither.
 *
 * Longitude/15 was the tempting offline answer and is rejected: it is wrong
 * by an hour or more across most of Europe, all of China, and every place
 * that keeps summer time — a quiet lie in exactly the digits that matter. */
const tideTzCache = new Map();          // "lat,lon" (1° cell) → {offsetSec, abbr, zone}

function tideBrowserTz(ms) {
  const off = -new Date(ms).getTimezoneOffset() * 60;
  let abbr = "";
  try {
    abbr = new Intl.DateTimeFormat(undefined, { timeZoneName: "short" })
      .formatToParts(new Date(ms)).find((p) => p.type === "timeZoneName")?.value || "";
  } catch { abbr = ""; }
  let zone = "";
  try { zone = Intl.DateTimeFormat().resolvedOptions().timeZone || ""; } catch { zone = ""; }
  return { offsetSec: off, abbr: abbr || "local", zone, browser: true };
}

function tideTzFor(lon, lat) {
  const key = `${Math.floor(lat)},${Math.floor(lon)}`;
  return tideTzCache.get(key) || null;
}

function tideEnsureTz(lon, lat, onReady) {
  const key = `${Math.floor(lat)},${Math.floor(lon)}`;
  if (tideTzCache.has(key)) return;
  tideTzCache.set(key, null);                       // in flight: ask once per cell
  omGet(`https://api.open-meteo.com/v1/forecast?latitude=${lat.toFixed(3)}` +
        `&longitude=${lon.toFixed(3)}&timezone=auto&forecast_days=1&current=temperature_2m`)
    .then((j) => {
      if (!j || j.utc_offset_seconds == null) { tideTzCache.delete(key); return; }
      tideTzCache.set(key, {
        offsetSec: j.utc_offset_seconds,
        abbr: j.timezone_abbreviation || "",
        zone: j.timezone || "",
      });
      onReady?.();
    })
    .catch(() => tideTzCache.delete(key));
}

/* Wall-clock time at the point, plus a day marker when the answer falls on a
 * different local date than the sim clock — "00:57" alone would read as
 * fourteen hours ago instead of two hours away. */
function tideClock(ms, tz, refMs) {
  const z = tz || tideBrowserTz(ms);
  const shift = (t) => new Date(t + z.offsetSec * 1000);
  const d = shift(ms);
  const hhmm = `${String(d.getUTCHours()).padStart(2, "0")}:` +
               `${String(d.getUTCMinutes()).padStart(2, "0")}`;
  let day = "";
  if (refMs != null) {
    const dd = Math.round((Date.UTC(d.getUTCFullYear(), d.getUTCMonth(), d.getUTCDate()) -
      (() => { const r = shift(refMs);
               return Date.UTC(r.getUTCFullYear(), r.getUTCMonth(), r.getUTCDate()); })())
      / 86400000);
    if (dd === 1) day = " tomorrow";
    else if (dd > 1) day = ` +${dd} d`;
  }
  return { hhmm, abbr: z.abbr, day, zone: z.zone, browser: !!z.browser };
}

function tideUTC(ms) {
  const d = new Date(ms);
  return `${String(d.getUTCHours()).padStart(2, "0")}:` +
         `${String(d.getUTCMinutes()).padStart(2, "0")}`;
}

/* "in 2 h 46 min" — relative to the SIM clock, so it counts down as the
 * simulation runs. Timezone-free by construction, which is why it stays even
 * when the local zone is still loading. */
function tideCountdown(ms) {
  const m = Math.max(0, Math.round(ms / 60000));
  return m >= 60 ? `${Math.floor(m / 60)} h ${String(m % 60).padStart(2, "0")} min`
                 : `${m} min`;
}

let tideDataPromise = null;
function loadTideData() {
  if (!tideDataPromise) {
    tideDataPromise = fetch("data/tide_constituents.json")
      .then((r) => r.json())
      .then((j) => {
        tideData = j;
        tideFields = tidePrepare(j);
        tideSim.t = Date.now();
        return j;
      });
  }
  return tideDataPromise;
}

/* Paint one frame into the BACK canvas (rows flipped: image row 0 = north),
 * swap buffers, hand the fresh canvas to the material. Returns the volume
 * of water currently standing above mean sea level, in km^3. */
function tidePaint() {
  const { nx, ny } = tideData;
  const f = tideFields;
  if (!tideLive.front) {
    for (const side of ["front", "back"]) {
      const c = document.createElement("canvas");
      c.width = nx; c.height = ny;
      tideLive[side] = c;
    }
    tideLive.img = new ImageData(nx, ny);
  }
  const px = tideLive.img.data;
  const th = (tideSim.t - Date.parse(tideData.epoch)) / 3600000;
  const rad = Math.PI / 180;
  const cosns = [], sinns = [];
  for (const c of f.consts) {
    const phi = (c.speed * th + c.V0) * rad;
    cosns.push(Math.cos(phi)); sinns.push(Math.sin(phi));
  }
  let volUp = 0;
  for (let i = 0; i < f.n; i++) {
    const iy = (i / nx) | 0, ix = i - iy * nx;
    const o = ((ny - 1 - iy) * nx + ix) * 4;
    if (!f.mask[i]) { px[o + 3] = 0; continue; }
    let h = 0;
    for (let k = 0; k < f.consts.length; k++) {
      h += cosns[k] * f.consts[k].P[i] + sinns[k] * f.consts[k].Q[i];
    }
    if (h > 0) volUp += h * f.area[i] * f.frac[i];
    // Smooth compression instead of a hard clamp: tanh keeps gradient
    // structure alive above ±2.5 m (the whole southern North Sea used to
    // saturate into one flat red slab at high water). ±2.5 m sits at
    // 12%/88% of the ramp; the legend's ticks are placed to match.
    const t = 0.5 + 0.5 * Math.tanh(h / TD_RANGE_CM);
    const [r, g, b] = rampColor("anom", t);
    px[o] = r; px[o + 1] = g; px[o + 2] = b;
    px[o + 3] = Math.round(255 * f.frac[i]);
  }
  const c = tideLive.back;
  c.getContext("2d").putImageData(tideLive.img, 0, 0);
  tideLive.back = tideLive.front;
  tideLive.front = c;
  if (tideLive.mat) tideLive.mat.uniforms.image = c;   // new object => re-upload
  return volUp / 1e5;                                  // cm*km^2 -> km^3
}

function tideEnsurePrimitive() {
  if (tideLive.prim) { tideLive.prim.show = true; tideLive.labels.show = true; return; }
  tidePaint();
  tideLive.mat = new Cesium.Material({
    fabric: {
      uniforms: { image: tideLive.front, opacity: tideLive.opacity },
      components: {
        diffuse: "texture(image, materialInput.st).rgb",
        alpha: "texture(image, materialInput.st).a * opacity",
      },
    },
  });
  tideLive.prim = viewer.scene.primitives.add(new Cesium.Primitive({
    geometryInstances: new Cesium.GeometryInstance({
      geometry: new Cesium.RectangleGeometry({
        rectangle: Cesium.Rectangle.fromDegrees(-180, -90, 180, 90),
        vertexFormat: Cesium.EllipsoidSurfaceAppearance.VERTEX_FORMAT,
      }),
      id: CITY_PICK,                       // clicks see through, like labels
    }),
    appearance: new Cesium.EllipsoidSurfaceAppearance({
      material: tideLive.mat, translucent: true, aboveGround: false,
      flat: true,   // DATA colours: never sun-shaded (the base globe is)
    }),
    asynchronous: false,
  }));
  tideLive.labels = viewer.scene.primitives.add(new Cesium.LabelCollection());
  for (const [glyph, color] of [["\u263e", Cesium.Color.WHITE],
                                ["\u2600", Cesium.Color.fromCssColorString("#eda100")]]) {
    tideLive.labels.add({
      id: CITY_PICK, text: glyph, font: "24px sans-serif",
      fillColor: color, outlineColor: Cesium.Color.BLACK, outlineWidth: 3,
      style: Cesium.LabelStyle.FILL_AND_OUTLINE,
      position: Cesium.Cartesian3.fromDegrees(0, 0, 50000),
      disableDepthTestDistance: Number.POSITIVE_INFINITY,
    });
  }
}

function tideMarkers() {
  if (!tideLive.labels || !tideLive.labels.show) return;
  const ast = tideAstro(tideSim.t);
  tideLive.labels.get(0).position =
    Cesium.Cartesian3.fromDegrees(ast.moon.lon, ast.moon.lat, 50000);
  tideLive.labels.get(1).position =
    Cesium.Cartesian3.fromDegrees(ast.sun.lon, ast.sun.lat, 50000);
  return ast;
}

function tideLegendEl() {
  const div = document.createElement("div");
  div.className = "legend-item";
  div.innerHTML = `<div class="legend-title">Tide height (live) — vs mean sea level</div>`;
  const wrap = document.createElement("div");
  wrap.className = "legend-bar-wrap";
  const canvas = document.createElement("canvas");
  const W = 268, H = 14, dpr = window.devicePixelRatio || 1;
  canvas.width = W * dpr; canvas.height = H * dpr;
  canvas.style.height = H + "px";
  canvas.className = "legend-bar";
  const ctx = canvas.getContext("2d");
  ctx.scale(dpr, dpr);
  for (let x = 0; x < W; x++) {
    const [r, g, b] = rampColor("anom", x / (W - 1));
    ctx.fillStyle = `rgb(${r},${g},${b})`;
    ctx.fillRect(x, 0, 1, H);
  }
  // The colour scale is tanh-compressed (see tidePaint): ±2.5 m sit at
  // 12% / 88% of the ramp. Mark them so the bar reads honestly.
  ctx.fillStyle = "rgba(255,255,255,.85)";
  for (const tt of [0.119, 0.881]) ctx.fillRect(Math.round(tt * W), 0, 1, H);
  const read = document.createElement("div");
  read.className = "legend-read hidden";
  canvas.addEventListener("pointermove", (e) => {
    const t = Math.max(0.004, Math.min(0.996, e.offsetX / canvas.clientWidth));
    const u = 2 * t - 1;                       // invert the tanh compression
    const cm = Math.round(TD_RANGE_CM * 0.5 * Math.log((1 + u) / (1 - u)));
    read.textContent = `${cm > 0 ? "+" : ""}${cm} cm ${cm >= 0 ? "above" : "below"} mean sea level`;
    read.classList.remove("hidden");
  });
  canvas.addEventListener("pointerleave", () => read.classList.add("hidden"));
  wrap.appendChild(canvas);
  // The numbers sit UNDER their own tick marks, not at the bar's ends: the
  // ramp is tanh-compressed, so its ends are not ±2.5 m — they run on to the
  // deepest and highest water anywhere. (This row used to carry the class
  // `legend-labels`, which no CSS ever defined, so the three labels rendered
  // as one run-on string: "−2.5 m at tick0 · mean+2.5 m at tick".)
  const labels = document.createElement("div");
  labels.className = "legend-ticks";
  labels.innerHTML =
    `<span style="left:11.9%">−${TD_RANGE_CM / 100} m</span>` +
    `<span style="left:50%">0 · mean sea level</span>` +
    `<span style="left:88.1%">+${TD_RANGE_CM / 100} m</span>`;
  div.appendChild(wrap); div.appendChild(labels); div.appendChild(read);
  return div;
}

function tideTabVisible() {
  return !document.getElementById("panel-tides").classList.contains("hidden");
}

function tideStats(volUp, ast) {
  if (!tideTabVisible()) return;
  const dt = new Date(tideSim.t);
  document.querySelector("#td-clock .stat-value").textContent =
    dt.toISOString().slice(0, 16).replace("T", " ");
  if (volUp != null) {
    document.querySelector("#td-volume .stat-value").textContent =
      Math.round(volUp).toLocaleString("en-US");
  }
  ast = ast || tideAstro(tideSim.t);
  const phase = ast.elong < 22 || ast.elong > 338 ? "new" :
    Math.abs(ast.elong - 180) < 22 ? "full" :
    Math.abs(ast.elong - 90) < 22 || Math.abs(ast.elong - 270) < 22 ? "quarter" :
    ast.elong < 180 ? "waxing" : "waning";
  document.querySelector("#td-phase .stat-value").textContent =
    `${phase} \u00b7 ${ast.spring > 0.7 ? "\u2192 spring" : ast.spring < 0.3 ? "\u2192 neap" : "between"}`;
}

/* One loop serves both consumers (globe layer, tab stats); it stops itself
 * when neither needs it and is (re)started by either entry point. */
function tideLoop() {
  cancelAnimationFrame(tideSim.raf);
  tideSim.wall = performance.now();
  const step = (now) => {
    if (!tideLive.on && !tideTabVisible()) return;     // stop: nobody watching
    if (tideSim.playing) tideSim.t += (now - tideSim.wall) * tideSim.speed;
    tideSim.wall = now;
    if (now - tideLive.lastPaint > 100) {              // <=10 fps texture swap
      tideLive.lastPaint = now;
      const volUp = tideLive.on ? tidePaint() : null;
      if (tideLive.on) {
        viewer.clock.currentTime = Cesium.JulianDate.fromDate(new Date(tideSim.t));
      }
      const ast = tideMarkers();
      tideStats(volUp, ast);
      if (tideSim.markCell != null && tideTabVisible()) { tideCurve(); tideRenderHiLo(); }
      viewer.scene.requestRender();
    }
    tideSim.raf = requestAnimationFrame(step);
  };
  tideSim.raf = requestAnimationFrame(step);
}

async function ensureTideLive(on) {
  tideLive.on = on;
  // The sun lights the planet while the tide runs: Cesium's own solar
  // terminator, driven by the SIM clock (below), so daylight, the \u2600
  // marker and the S2 bulge all agree. The tide material itself is flat
  // (unlit) - data colours stay honest on the night side.
  viewer.scene.globe.enableLighting = on;
  if (on) {
    await loadTideData();
    tideEnsurePrimitive();
    tideLoop();
  } else if (tideLive.prim) {
    tideLive.prim.show = false;
    tideLive.labels.show = false;
    viewer.clock.currentTime = Cesium.JulianDate.now();
    viewer.scene.requestRender();
  }
  updateLegends();
}

/* The headline answer: when is the next high water, and the next low. Both
 * are read off tideExtrema, so the time is the true zero of the rate, not the
 * nearest plotted sample. Re-rendered only when the minute or the cell
 * changes — the animation loop calls this ten times a second. */
let tideHiLoKey = "";
function tideRenderHiLo(force = false) {
  const el = document.getElementById("td-next");
  const i = tideSim.markCell;
  if (!el || i == null) return;
  const tz = tideTzFor(tideSim.markLon, tideSim.markLat);
  const key = `${i}|${Math.floor(tideSim.t / 60000)}|${tz ? tz.offsetSec : "b"}`;
  if (key === tideHiLoKey && !force) return;
  tideHiLoKey = key;
  const ex = tideExtrema(i, tideSim.t, 30);
  const row = (e, name, cls) => {
    if (!e) return "";
    const m = (e.cm / 100).toFixed(2).replace("-", "−");
    const c = tideClock(e.ms, tz, tideSim.t);
    return `<div class="td-hl ${cls}"><span class="td-hl-k">next ${name}</span>` +
      `<span class="td-hl-t">${c.hhmm}<span class="td-hl-z"> ${c.abbr}${c.day}</span></span>` +
      `<span class="td-hl-v">${e.cm >= 0 ? "+" : ""}${m} m</span>` +
      `<span class="td-hl-in">in ${tideCountdown(e.ms - tideSim.t)}</span></div>`;
  };
  el.innerHTML = row(ex.find((e) => e.high), "high water", "td-high") +
                 row(ex.find((e) => !e.high), "low water", "td-low");
  const foot = document.getElementById("td-tz");
  if (foot) {
    const c = tideClock(tideSim.t, tz, null);
    foot.innerHTML = tz
      ? `Times are local at this point — <strong>${tz.zone || c.abbr}</strong> ` +
        `(${c.abbr}, UTC${tz.offsetSec >= 0 ? "+" : "−"}${Math.abs(tz.offsetSec / 3600)}), ` +
        `summer time included. Sim clock now ${tideUTC(tideSim.t)} UTC.`
      : `Times are in <strong>your device's</strong> timezone (${c.abbr}) while this ` +
        `point's own zone loads. Sim clock now ${tideUTC(tideSim.t)} UTC.`;
  }
}

/* The tab shows either a point's tide or an invitation to pick one — never a
 * blank space where a chart should be (which is how it read before: the
 * curve lives in a hidden block and the only prompt was a grey line of hint
 * text below the fold). */
function tideSyncPointUi() {
  const has = tideSim.markCell != null;
  const pt = document.getElementById("td-point");
  const empty = document.getElementById("td-empty");
  if (pt) pt.classList.toggle("hidden", !has);
  if (empty) empty.classList.toggle("hidden", has);
}

/* The 3-day curve, with the recent PAST included so "now" lands ON the line
 * rather than at its left edge. Reported 2026-08-08: starting the chart at
 * the present moment makes the single most useful fact invisible — whether
 * the water is rising or falling right now — because the trend you read a
 * curve by only exists once you can see where it came from. Six hours is
 * about half a semidiurnal cycle: enough to always contain the last turning
 * point, short enough that the future still owns most of the width. */
const TD_PAST_MS = 6 * 3600000;
const TD_FUTURE_MS = 72 * 3600000;

function tideCurve() {
  const c = document.getElementById("td-curve");
  const ctx = c.getContext("2d");
  const i = tideSim.markCell;
  const W = c.width, H = c.height;
  ctx.clearRect(0, 0, W, H);
  const NOW = tideSim.t;
  const T0 = NOW - TD_PAST_MS, SPAN = TD_PAST_MS + TD_FUTURE_MS;
  const X = (ms) => (ms - T0) / SPAN * W;

  const N = 312;                                    // ~15-minute resolution
  let lo = Infinity, hi = -Infinity;
  const pts = [];
  for (let s = 0; s <= N; s++) {
    const h = tideHeightAt(i, T0 + s / N * SPAN);
    pts.push(h); lo = Math.min(lo, h); hi = Math.max(hi, h);
  }
  const pad = Math.max(10, (hi - lo) * 0.12);
  lo -= pad; hi += pad;
  const y = (h) => H - (h - lo) / (hi - lo) * H;
  const xNow = X(NOW);

  // the past, behind everything: a dim panel, so "already happened" reads
  // without needing a legend
  ctx.fillStyle = "rgba(255,255,255,0.035)";
  ctx.fillRect(0, 0, xNow, H);

  // mean sea level
  ctx.strokeStyle = "#30363d"; ctx.lineWidth = 1;
  ctx.beginPath(); ctx.moveTo(0, y(0)); ctx.lineTo(W, y(0)); ctx.stroke();

  // LOCAL midnights. The label promised "day boundaries" while the code drew
  // marks every 24 h from now — a different thing entirely. With the point's
  // own zone known, these are real calendar days at the beach in question.
  const z = tideTzFor(tideSim.markLon, tideSim.markLat) || tideBrowserTz(NOW);
  const off = z.offsetSec * 1000;
  const DAY = ["Sun", "Mon", "Tue", "Wed", "Thu", "Fri", "Sat"];
  ctx.font = "9.5px sans-serif"; ctx.textAlign = "left";
  for (let m = Math.ceil((T0 + off) / 86400000) * 86400000 - off;
       m < T0 + SPAN; m += 86400000) {
    const x = X(m);
    ctx.strokeStyle = "#30363d"; ctx.setLineDash([3, 4]);
    ctx.beginPath(); ctx.moveTo(x, 0); ctx.lineTo(x, H); ctx.stroke();
    ctx.setLineDash([]);
    const d = new Date(m + off);
    ctx.fillStyle = "#6e7681";
    ctx.fillText(`${DAY[d.getUTCDay()]} ${d.getUTCDate()}`, x + 3, H - 3);
  }

  // one line, two states: the past thin and dim, the future at full weight
  const seg = (from, to, width, alpha) => {
    ctx.globalAlpha = alpha; ctx.strokeStyle = "#3987e5";
    ctx.lineWidth = width; ctx.lineJoin = "round";
    ctx.beginPath();
    let started = false;
    for (let s = 0; s <= N; s++) {
      const ms = T0 + s / N * SPAN;
      if (ms < from || ms > to) { started = false; continue; }
      const x = X(ms), yy = y(pts[s]);
      if (!started) { ctx.moveTo(x, yy); started = true; } else ctx.lineTo(x, yy);
    }
    ctx.stroke(); ctx.globalAlpha = 1;
  };
  seg(T0, NOW, 1.5, 0.45);
  seg(NOW - SPAN / N, T0 + SPAN, 2.2, 1);       // one step of overlap: no seam

  // every turning point dotted; the next of each kind named. Past ones stay
  // (dimmed) — "the last high water was at 04:12" is half of what a tide
  // table is for.
  let namedHigh = false, namedLow = false;
  for (const e of tideExtrema(i, T0, SPAN / 3600000)) {
    const x = X(e.ms), yy = y(e.cm), future = e.ms >= NOW;
    const first = future && (e.high ? !namedHigh : !namedLow);
    if (future) { if (e.high) namedHigh = true; else namedLow = true; }
    ctx.globalAlpha = future ? 1 : 0.45;
    ctx.fillStyle = e.high ? "#e34948" : "#3b82f6";
    ctx.beginPath(); ctx.arc(x, yy, first ? 4 : 2.5, 0, Math.PI * 2); ctx.fill();
    ctx.globalAlpha = 1;
    if (!first) continue;
    ctx.font = "10px sans-serif";
    const right = x > W - 46;
    ctx.textAlign = right ? "right" : "left";
    ctx.fillText(tideClock(e.ms, z, NOW).hhmm, x + (right ? -6 : 6),
                 yy + (e.high ? -6 : 13));
  }

  // NOW: an accent rule, a ring where it meets the water, and the word
  const hNow = tideHeightAt(i, NOW);
  ctx.strokeStyle = "#4493f8"; ctx.lineWidth = 1.5;
  ctx.beginPath(); ctx.moveTo(xNow, 0); ctx.lineTo(xNow, H); ctx.stroke();
  ctx.fillStyle = "#0a0f16"; ctx.lineWidth = 2.5;
  ctx.beginPath(); ctx.arc(xNow, y(hNow), 4.5, 0, Math.PI * 2);
  ctx.fill(); ctx.stroke();
  ctx.fillStyle = "#4493f8"; ctx.font = "600 10px sans-serif"; ctx.textAlign = "left";
  ctx.fillText("now", xNow + 6, 11);

  // the window's extremes, kept clear of the "now" label
  ctx.fillStyle = "#c3c2b7"; ctx.font = "11px sans-serif"; ctx.textAlign = "right";
  ctx.fillText(`${(Math.max(...pts) / 100).toFixed(1)} m`, W - 4, 12);
  ctx.fillText(`${(Math.min(...pts) / 100).toFixed(1)} m`, W - 4, H - 16);
  ctx.textAlign = "left";
}

/* The model grid is 1\u00b0, so a coastal town is often IN a land cell \u2014 tapping
 * the beach at Peniche would answer "no tide here" with the Atlantic in
 * frame. Search outward for the nearest cell the tide model actually solves;
 * `rings` bounds how far that search may reach (1 cell \u2248 110 km, i.e. still
 * "the coast here"; a tap in the middle of a continent still finds nothing). */
function tideNearestWater(lon, lat, rings = 1) {
  const ix0 = Math.floor(lon - tideData.west), iy0 = Math.floor(lat - tideData.south);
  for (let r = 0; r <= rings; r++) {
    let best = null, bestD = Infinity;
    for (let dy = -r; dy <= r; dy++) {
      for (let dx = -r; dx <= r; dx++) {
        if (Math.max(Math.abs(dx), Math.abs(dy)) !== r) continue;    // ring edge only
        const ix = ix0 + dx, iy = iy0 + dy;
        if (ix < 0 || ix >= tideData.nx || iy < 0 || iy >= tideData.ny) continue;
        const i = iy * tideData.nx + ix;
        if (!tideFields.mask[i]) continue;
        const d = dx * dx + dy * dy;
        if (d < bestD) { bestD = d; best = { i, ix, iy }; }
      }
    }
    if (best) return best;
  }
  return null;
}

/* Select the tide point for the tab's readout and 3-day curve \u2014 from a globe
 * tap, from a place search, or from the camera when the tab opens.
 * `place` names it when we know the name, because a panel that says
 * "Peniche" is answering the question the user actually asked. */
function tideSelectPoint(carto, rings = 1, place = null) {
  if (!tideData) return false;
  const lon = Cesium.Math.toDegrees(carto.longitude);
  const lat = Cesium.Math.toDegrees(carto.latitude);
  const hit = tideNearestWater(lon, lat, rings);
  if (!hit) return false;
  tideSim.markCell = hit.i;
  tideSim.markPlace = place;
  tideSyncPointUi();
  // The coordinates named are the CELL's centre, not the tap: when the tap
  // was on a land cell the answer comes from the water cell next door, and
  // the label must say which point it is actually reporting.
  const clat = tideData.south + hit.iy + 0.5, clon = tideData.west + hit.ix + 0.5;
  tideSim.markLat = clat; tideSim.markLon = clon;
  tideEnsureTz(clon, clat, () => tideRenderHiLo(true));
  const cm = Math.round(tideHeightAt(hit.i, tideSim.t));
  const cmTxt = `${cm > 0 ? "+" : ""}${cm} cm ${cm >= 0 ? "above" : "below"} mean`;
  const coord = `${Math.abs(clat).toFixed(1)}\u00b0${clat >= 0 ? "N" : "S"} ` +
                `${Math.abs(clon).toFixed(1)}\u00b0${clon >= 0 ? "E" : "W"}`;
  // The 3-day swing at THIS point \u2014 the answer to "the legend says \u00b12.5 m but
  // this curve says \u00b10.3 m": the legend is one fixed scale for the whole
  // globe, the curve auto-scales to the point.
  const ex3 = tideExtrema(hit.i, tideSim.t, 72);
  const swing = ex3.length
    ? (Math.max(...ex3.map((e) => e.cm)) - Math.min(...ex3.map((e) => e.cm))) / 100 : 0;
  const spring = tideSpringRange(hit.i);
  const el = document.getElementById("td-point-title");
  el.innerHTML =
    (place ? `<strong class="td-place">${place}</strong> \u00b7 ` : "") +
    `${coord} \u2014 tide now <strong>${cmTxt}</strong>` +
    (swing ? ` \u00b7 range <strong>${swing.toFixed(1)} m</strong> over these 3 days` +
             (spring ? `, up to <strong>${spring.toFixed(1)} m</strong> at spring tide` : "") : "");
  tideRenderHiLo(true);
  if (tideTabVisible()) tideCurve();
  else {
    const nx = ex3[0];
    const c = nx ? tideClock(nx.ms, tideTzFor(clon, clat), tideSim.t) : null;
    showToast(`Tide ${place ? `at <strong>${place}</strong>` : "here"} now: ` +
      `<strong>${cmTxt}</strong>` +
      (nx ? ` \u00b7 next ${nx.high ? "high" : "low"} water <strong>${c.hhmm} ${c.abbr}${c.day}</strong>` +
            ` (in ${tideCountdown(nx.ms - tideSim.t)})` : "") +
      ` \u2014 the full 3-day curve is in the <strong>Tides tab</strong>.`,
      { key: "tide-point", replace: true });
  }
  return true;
}

/* A place search IS the user saying "I mean here". Remember it, and if the
 * tide dashboard is already open, move it there \u2014 no globe tap required. */
let tidePlace = null;
function tideNotePlace(p) {
  tidePlace = { name: p.n, lon: p.o, lat: p.a };
  if (tideData && tideTabVisible()) {
    tideSelectPoint(Cesium.Cartographic.fromDegrees(p.o, p.a), 2, p.n);
  }
}

let tideUiWired = false;
async function loadTides() {
  await loadTideData();
  if (!tideUiWired) {
    tideUiWired = true;
    document.getElementById("td-scale").innerHTML =
      `colour: <span style="color:#3b82f6">\u2212${TD_RANGE_CM / 100} m below</span> \u2026 ` +
      `<span style="color:#e34948">+${TD_RANGE_CM / 100} m above</span> mean sea level ` +
      `(smoothly compressed beyond, so shelf seas keep their structure; ` +
      `coastal cells fade with their land share)`;
    document.getElementById("td-play").addEventListener("click", (e) => {
      tideSim.playing = !tideSim.playing;
      e.target.textContent = tideSim.playing ? "\u23f8" : "\u25b6";
    });
    document.getElementById("td-now").addEventListener("click", () => { tideSim.t = Date.now(); });
    document.getElementById("td-speed").addEventListener("change", (e) => {
      tideSim.speed = +e.target.value;
    });
  }
  // Opening the control room switches the picture on: the tab without the
  // layer is a dashboard about nothing.
  const box = document.getElementById("toggle-tidelive");
  if (!box.checked) {
    box.checked = true;
    box.dispatchEvent(new Event("change", { bubbles: true }));
  }
  // …and if nothing is selected yet, answer for whatever the camera is
  // looking at. Searching "Peniche" and opening this tab should show
  // Peniche's tide, not an empty panel with a tap-somewhere hint: the user
  // has already said where they mean. Two rings (~220 km) so a coastal view
  // finds its water; if the camera is over a continent or off the limb, the
  // empty state stands.
  if (tideSim.markCell == null) {
    if (tidePlace) {
      tideSelectPoint(Cesium.Cartographic.fromDegrees(tidePlace.lon, tidePlace.lat),
                      2, tidePlace.name);
    } else {
      const canvas = viewer.scene.canvas;
      const cart = viewer.camera.pickEllipsoid(
        new Cesium.Cartesian2(canvas.clientWidth / 2, canvas.clientHeight / 2),
        viewer.scene.globe.ellipsoid);
      if (cart) tideSelectPoint(Cesium.Cartographic.fromCartesian(cart), 2);
    }
  }
  tideSyncPointUi();
  tideLoop();
}

document.getElementById("toggle-tidelive").addEventListener("change", (e) => {
  ensureTideLive(e.target.checked);
  if (e.target.checked) maybeDatelessToast("tidelive");
});
document.getElementById("tidelive-alpha").addEventListener("input", (e) => {
  tideLive.opacity = e.target.value / 100;
  if (tideLive.mat) tideLive.mat.uniforms.opacity = tideLive.opacity;
  document.getElementById("tidelive-alpha-val").textContent = `${e.target.value}%`;
  viewer.scene.requestRender();
});

/* ---------------------------------------------------- installed-app updates
 * The app is installable and deliberately has NO service worker: a fresh
 * page load always gets the newest build (hash-stamped URLs). The missing
 * piece on a phone is the TRIGGER — a standalone instance has no reload
 * button and Android keeps it alive for days. So: compare the served
 * index.html's app.js stamp against our own whenever the app returns to
 * the foreground (and every 15 min while visible), and offer a one-tap
 * reload when they differ. A toast with a 10-minute timeout, keyed and
 * replace-safe, so it neither nags nor vanishes before it is seen. */
const OUR_BUILD = (document.querySelector('script[src*="src/app.js?v="]')
  ?.getAttribute("src") || "").split("v=")[1] || "";
let lastBuildCheck = 0;
async function checkForNewBuild() {
  if (!OUR_BUILD) return null;
  lastBuildCheck = Date.now();
  try {
    const html = await (await fetch(`index.html?fresh=${Date.now()}`,
                                    { cache: "no-store" })).text();
    const served = (html.match(/src\/app\.js\?v=([0-9a-f]{8})/) || [])[1];
    if (served && served !== OUR_BUILD) {
      showToast(
        `A <strong>newer build of earth</strong> is live. ` +
        `<button id="reload-now" class="td-btn" style="margin-left:6px">reload</button>`,
        { key: "new-build", replace: true, timeout: 600000 });
      document.getElementById("reload-now")
        ?.addEventListener("click", () => location.reload());
      return served;
    }
    return served || null;
  } catch { return null; }        // offline — silent, retry next trigger
}
document.addEventListener("visibilitychange", () => {
  if (document.visibilityState === "visible" && Date.now() - lastBuildCheck > 30000) {
    checkForNewBuild();
  }
});
setInterval(() => {
  if (document.visibilityState === "visible") checkForNewBuild();
}, 900000);
setTimeout(checkForNewBuild, 10000);

/* ==================================================================== playback
 * E-041. The one design decision that makes the rest of this small: PLAYBACK IS
 * A CLOCK THAT DRIVES state.date. It is not a rendering mode. Everything on
 * this globe already keys off that one string — GIBS tile times, month- and
 * day-keyed grids, the comparison, the rolling window, the legends, the probe —
 * so a player that sets the date and lets the ordinary machinery repaint
 * inherits all of it, correctly, and cannot drift from the single-date path the
 * way a bespoke animation renderer would. Press play with split compare on and
 * you get a wipe of moving present against pinned past; press play in delta
 * mode and you animate the anomaly; press play on a grid layer and it runs with
 * no tile traffic at all. None of that is code below. */

const PLAY_MAX_FRAMES = 500;
// Coarsest-last, because that is the direction the cap walks.
const PLAY_STEPS = ["1d", "5d", "1mo", "1y"];
const PLAY_STEP_LABEL = { "1d": "1 day", "5d": "5 days", "1mo": "1 month", "1y": "1 year" };
// A late frame beats a stuck player: past this, advance whatever the network is
// doing. Without it one unanswered tile request stops the animation dead and
// the app looks broken rather than slow.
const PLAY_FRAME_CEILING_MS = 8000;
/* How many frames ahead the preload ring holds (§ the ring, below). Two is the
 * default and it is not a free number: every ring layer holds the WHOLE
 * VISIBLE TILE SET, decoded and uploaded to the GPU at 512×512×4 bytes a tile
 * — about a megabyte each, eleven tiles in the default globe view, doubled
 * again by a split comparison. Two is the smallest depth that hides a frame's
 * load time completely (i+1 is warm before it is shown, and i+2 has already
 * started warming while i+1 is displayed); three and beyond buy progressively
 * less and cost linearly more texture memory. */
const PLAY_PRELOAD_DEPTH = 2;

/* Above this many tiles in view, the ring drops to depth 1. The visible tile
 * count multiplies every megabyte above it, and it is not bounded by anything
 * else: zoom in and `_tilesToRender` grows without the player noticing. At 32
 * tiles a depth-2 ring for a split comparison is already 32 × 2 × 2 ≈ 128 MB
 * of speculative texture, held on top of the live frame and up to three
 * retired ones — more GPU memory than it is reasonable to take from a phone
 * for frames the user may never reach. 32 is roughly three times the eleven
 * tiles a default view requests, i.e. the point at which the ring stops being
 * a rounding error against what the app is already holding. */
const PLAY_PRELOAD_TILE_LIMIT = 32;

const sleepMs = (ms) => new Promise((r) => setTimeout(r, ms));

/* The layers a frame actually shows something for. Two kinds qualify and the
 * distinction matters everywhere below: `timed` GIBS rasters (the date is a
 * tile URL) and month/day-keyed grids (the date is a key into a baked file).
 * Everything else — climatologies, night lights, the point layers — is
 * date-independent, so it neither sets the cadence nor changes between frames. */
function playbackLayers() {
  return Object.values(state.layers).filter(
    (e) => (e.layer || e.suppressed) && (e.cfg.timed || e.cfg.monthlyGrid));
}

// Stable identity of the CURRENT layer set, so a toggle mid-play is detectable
// without diffing objects (plan §4: the frame list is only valid for the layers
// it was enumerated from).
function playbackLayerKey() {
  return playbackLayers().map((e) => e.cfg.id).sort().join(",");
}

/* A layer's own cadence. Asking for finer than this produces byte-identical
 * requests; asking for coarser throws away frames the product actually has. */
function playbackStepOf(cfg) {
  if (cfg.annual) return "1y";
  if (cfg.monthly) return "1mo";
  if (cfg.snap5d) return "5d";
  if (cfg.monthlyGrid) {
    // keyLen 10 = day-keyed (the GFS forecast grids); anything else is monthly.
    const g = gridsLoaded.get(cfg.id);
    if (cfg.forecastGrid || g?.keyLen === 10) return "1d";
    return "1mo";
  }
  return "1d";                          // sub-daily and daily rasters alike
}

/* "auto" = the FINEST cadence among the layers on. Stepping a calendar day at a
 * time over a monthly product produces thirty identical frames and thirty
 * identical tile requests; stepping a month at a time over daily SST throws
 * away twenty-nine days of the thing you came to watch. The finest cadence is
 * the only choice that does neither, and the dedupe below removes what it
 * over-produces for the coarser layers in the same stack. */
function playbackAutoStep(layers) {
  let best = null;
  for (const e of layers) {
    const s = playbackStepOf(e.cfg);
    if (best === null || PLAY_STEPS.indexOf(s) < PLAY_STEPS.indexOf(best)) best = s;
  }
  return best || "1d";
}

// The walk, through the app's own calendar arithmetic — so playback lands on
// Feb 28 stepping a month from Jan 31 for exactly the same reason the date
// stepper does, rather than for a second reason written here.
function playAdvance(dateStr, step) {
  if (step === "1mo") return stepCalendar(dateStr, "+1m");
  if (step === "1y") return stepCalendar(dateStr, "+1y");
  let d = dateStr;
  const n = step === "5d" ? 5 : 1;
  for (let k = 0; k < n; k++) d = stepCalendar(d, "+1d");
  return d;
}

/* THE SIGNATURE: what the active layers would actually REQUEST for a candidate
 * date. This is the whole of the frame list's intelligence, and it is worth
 * more than it looks. It collapses the thirty identical days of a monthly
 * product to one frame; it collapses a scrub through a closed archive's dead
 * zone (GRACE ends 2022-07, CERES 2018-10, SSH 2019-01) from four hundred
 * identical frames to ONE; and it is the reason this feature cannot hammer GIBS
 * for tiles it already has.
 *
 * A grid whose file has not arrived yet contributes NOTHING rather than
 * blocking. That is deliberate and it is safe in one direction only: a missing
 * part can merge two frames that would have differed (you see fewer frames than
 * the data holds), and it can never invent a frame or move one to a wrong date.
 * The list is re-enumerated whenever the layer set changes, which is also when
 * a newly loaded grid becomes visible to it. */
function playbackSignature(dateStr, layers) {
  const parts = [];
  for (const e of layers) {
    const cfg = e.cfg;
    if (cfg.grid) {
      const g = gridsLoaded.get(cfg.id);
      if (g) parts.push(`${cfg.id}=${resolveGridMonth(g, dateStr)}`);
    } else {
      parts.push(`${cfg.id}=${gibsTime(cfg, dateStr)}`);
    }
  }
  return parts.join("|");
}

function playbackEnumerate(start, end, step, layers) {
  const out = [];
  let prev = null;
  let d = start;
  for (let guard = 0; guard < 200000; guard++) {
    const sig = playbackSignature(d, layers);
    if (sig !== prev) { out.push(d); prev = sig; }
    if (d >= end) break;
    const next = playAdvance(d, step);
    if (next <= d) break;                       // paranoia: never loop forever
    // The END of the range is always offered as a candidate, even when the walk
    // would step over it. The range the user asked for is never truncated; the
    // dedupe may still drop this candidate, and when it does that is the honest
    // answer — the last step of the walk already shows what the end date shows.
    d = next > end ? end : next;
  }
  return out;
}

/* frames for [start, end] at `step` ("auto" resolves against the layers on).
 * Returns the resolved step and a NOTE explaining anything the caller did not
 * ask for, because the two things a frame list can silently do to you are
 * change your step and shorten your range. It does the first and says so; it
 * never does the second. */
function playbackFrames(startDate, endDate, step = "auto") {
  const layers = playbackLayers();
  let start = startDate, end = endDate;
  if (start > end) { const t = start; start = end; end = t; }
  const asked = step === "auto" ? playbackAutoStep(layers) : step;
  let resolved = PLAY_STEPS.includes(asked) ? asked : "1d";
  let frames = playbackEnumerate(start, end, resolved, layers);
  const firstStep = resolved, firstCount = frames.length;

  /* The cap COARSENS, never truncates. A cap that quietly dropped the tail of
   * the range would be the same class of lie the rest of this app keeps writing
   * tests against: you would ask for 1993–2026 and watch 1993–1994 without ever
   * being told. Coarsening still shows the whole span, at a step the panel
   * names. */
  let i = PLAY_STEPS.indexOf(resolved);
  while (frames.length > PLAY_MAX_FRAMES && i < PLAY_STEPS.length - 1) {
    resolved = PLAY_STEPS[++i];
    frames = playbackEnumerate(start, end, resolved, layers);
  }

  let note = "";
  if (resolved !== firstStep) {
    note = `${start.slice(0, 4)}–${end.slice(0, 4)} at ${PLAY_STEP_LABEL[firstStep]} ` +
      `would be ${firstCount.toLocaleString("en-US")} frames; ` +
      `showing ${frames.length.toLocaleString("en-US")} at ${PLAY_STEP_LABEL[resolved]}`;
  }
  return { frames, step: resolved, note };
}

/* ------------------------------------------------------------ the transport */

const playback = {
  frames: [], i: 0, playing: false, fps: 2, loop: false,
  start: null, end: null, step: "auto",
  // "fps", "network" or "device": which limit actually decided the last frame's
  // duration. Requested fps is a SPEED LIMIT, not a promise, and a player that
  // claimed 8 fps while showing 1.4 would be lying about the thing the user is
  // staring at. The preload ring is what makes "fps" the common answer — and
  // what made "device" worth distinguishing from "network".
  bound: null,
  actualFps: null,
  // The effective lookahead of the preload ring for the frame last shown —
  // PLAY_PRELOAD_DEPTH, or 1 on a small device or a zoomed-in view, or 0 for a
  // frame set that needs no ring. Reported in the status line.
  preloadDepth: null,
  note: "",
  layerKey: "",
};

let playToken = 0;                      // bumped to abandon an in-flight loop

function playbackNearest(dateStr) {
  if (!playback.frames.length) return 0;
  let best = 0, bestD = Infinity;
  for (let k = 0; k < playback.frames.length; k++) {
    const d = Math.abs(Date.parse(playback.frames[k]) - Date.parse(dateStr));
    if (d < bestD) { bestD = d; best = k; }
  }
  return best;
}

// Re-enumerate from the panel (or from whatever the last programmatic range
// was), keeping the playhead on the same DATE rather than the same index — an
// index means nothing across two different frame lists.
function playbackRebuild() {
  // Re-enumerating changes which dates the frames ARE, so anything the ring
  // holds is keyed to a list that no longer exists.
  playbackPreloadClear();
  const at = playback.frames[playback.i] || state.date;
  const startEl = document.getElementById("pb-start");
  const endEl = document.getElementById("pb-end");
  const stepEl = document.getElementById("pb-step");
  playback.start = (startEl?.value) || playback.start || clampPlayDate(stepCalendar(state.date, "-1y"));
  playback.end = (endEl?.value) || playback.end || clampPlayDate(state.date);
  playback.step = (stepEl?.value) || playback.step || "auto";
  const r = playbackFrames(playback.start, playback.end, playback.step);
  playback.frames = r.frames;
  playback.resolvedStep = r.step;
  playback.note = r.note;
  playback.layerKey = playbackLayerKey();
  playback.i = Math.min(playbackNearest(at), Math.max(0, playback.frames.length - 1));
}

/* Show frame i. Still the ordinary date path — nothing about the PICTURE is
 * special, only the clock that set the date — with one shortcut: if the
 * preload ring already holds this frame's layers, warmed at alpha 0 while an
 * earlier frame was on screen, they are promoted instead of rebuilt. A promote
 * is a property assignment and costs no requests at all (see the ring, below);
 * the fallback is the same held refresh as before, unchanged, so a frame the
 * ring missed behaves exactly as every frame did yesterday. */
async function playbackShowFrame(i) {
  if (!playback.frames.length) return;
  playback.i = Math.max(0, Math.min(i, playback.frames.length - 1));
  state.date = playback.frames[playback.i];
  const input = document.getElementById("layer-date");
  if (input) input.value = state.date;     // the single-date UI must not lie
  syncCompareUi();                         // an OFFSET comparison moves with it
  // `keepPreload` on the fallback: this is not a configuration change, and
  // clearing the ring for a frame it merely failed to hold would throw away
  // the lookahead precisely when the player is behind.
  const promoted = playbackPromote(state.date);
  if (!promoted) refreshTimedLayers({ hold: true, keepPreload: true });
  await refreshMonthlyGrids();
  await refreshYearlyLayers();
  if (sstEnsembleLayer) updateEnsembleLayer();
  playbackRender();
  // Whether the ring served this frame. The run loop reads it to decide
  // whether there is anything left to wait for; every other caller ignores it.
  return promoted;
}

/* ------------------------------------------------------------------ read-out */

/* The date on the read-out is the frame's RESOLVED PER-LAYER time, not the
 * calendar date we asked for. gibsTime() clamps and snaps per layer, so a
 * playback over 2023–2026 with GRACE on is really showing 2022-07 in every
 * frame — and a player that printed the calendar date would be captioning a
 * four-year-old picture with today's date, on every frame, silently. Same
 * helper as the pixel card and the hover probe (whenOfLayer → whenLabel), so
 * the three can never disagree. */
function playbackStamps() {
  const out = [];
  for (const e of playbackLayers()) {
    const g = e.cfg.grid ? gridsLoaded.get(e.cfg.id) : null;
    const w = whenOfLayer(e.cfg, g);
    if (w) out.push(`<span class="pb-when">${esc(e.cfg.title)} · ${whenLabel(w)}</span>`);
  }
  return out.join("");
}

function playbackRender() {
  const readout = document.getElementById("pb-readout");
  const status = document.getElementById("pb-status");
  const scrub = document.getElementById("pb-scrub");
  const playBtn = document.getElementById("pb-play");
  const empty = document.getElementById("pb-empty");
  const nLayers = playbackLayers().length;

  if (empty) empty.classList.toggle("hidden", nLayers > 0);
  if (playBtn) {
    playBtn.disabled = nLayers === 0 || playback.frames.length < 2;
    playBtn.textContent = playback.playing ? "⏸" : "▶";
    playBtn.classList.toggle("playing", playback.playing);
  }
  if (scrub) {
    scrub.min = "0";
    scrub.max = String(Math.max(0, playback.frames.length - 1));
    scrub.value = String(playback.i);
    scrub.disabled = playback.frames.length < 2;
  }
  if (readout) {
    if (!playback.frames.length) {
      readout.innerHTML = `<div class="pb-frame">no frames</div>`;
    } else {
      readout.innerHTML =
        `<div class="pb-frame">frame ${playback.i + 1} / ${playback.frames.length} ` +
        `· ${playback.frames[playback.i]}</div>` +
        playbackStamps();
    }
  }
  if (status) {
    const bits = [];
    if (playback.frames.length) {
      const auto = playback.step === "auto" ? " (auto — the finest cadence of the layers on)" : "";
      bits.push(`step ${PLAY_STEP_LABEL[playback.resolvedStep] || playback.resolvedStep}${auto} ` +
        `· ${playback.frames.length} frame${playback.frames.length === 1 ? "" : "s"}`);
    }
    /* Say how deep the lookahead actually is. The ring is the difference
     * between a player bound by the network and one bound by the requested
     * fps, and its depth is reduced silently on a small device or a zoomed-in
     * view — so it is reported rather than left as magic, and a frame set that
     * needs no ring at all says that instead of showing a zero. */
    if (playback.frames.length) {
      const depth = playback.playing && playback.preloadDepth != null
        ? playback.preloadDepth : playbackPreloadDepth();
      bits.push(depth
        ? `${depth} frame${depth === 1 ? "" : "s"} preloaded`
        : "no preload needed (grids draw from a baked file)");
    }
    if (playback.note) bits.push(playback.note);
    if (playback.playing && playback.actualFps) {
      const why = { fps: "at the requested speed", network: "network-bound",
        device: "the browser cannot repaint any faster" }[playback.bound] || "";
      bits.push(`running at ${playback.actualFps.toFixed(1)} fps — ${why}`);
    }
    status.textContent = bits.join(" · ");
  }
  // The globe carries the date too: on a phone the picture is 55% of the
  // screen and the panel is a scroll away, so a chip beside the layer chips is
  // where the date is actually readable while watching.
  updateActiveChips();
}

/* ----------------------------------------------------------- the preload ring
 *
 * MEASURED 2026-08-18, and the reason this is not the obvious implementation:
 *
 * FACT 1 — GIBS FORBIDS HTTP CACHING. Every tile response carries
 *   `cache-control: max-age=0, no-store, no-cache, must-revalidate`
 * (checked on MODIS_Terra_CorrectedReflectance_TrueColor and
 * GHRSST_L4_MUR_Sea_Surface_Temperature, on 2015, 2026 and default dates).
 * `no-store` means the browser MUST NOT retain the response. So the shipped
 * first version of this — a CORS-free `fetch` of every visible tile of frame
 * i+1, up to sixty a frame, issued for no reason but to fill the HTTP cache —
 * could not warm anything, ever. (The string literal is deliberately not
 * written here: `tests/app.spec.js` asserts the source contains no quoted
 * no-cors, which is the cheapest possible guard against this coming back.) It
 * was a doubling of the request load on a public NASA service in exchange for
 * nothing at all, and it has been deleted. DO NOT RE-ADD IT: warming the HTTP
 * cache is the obvious idea, it is wrong for this service specifically, and it
 * fails silently in the one direction that looks like success (the tiles do
 * arrive — they arrive from the network, exactly as they would have anyway).
 *
 * FACT 2 — A CESIUM LAYER AT alpha 0 IS A WORKING PREFETCH. Measured in the
 * browser against the real app:
 *   layer added with show = false          →  0 tile requests
 *   layer added with show = true, alpha 0  →  11 requests (the whole visible set)
 *   promoting it (alpha 0 → 1)             →  0 NEW requests, globe.tilesLoaded true
 * The mechanism is the one the retirement queue already leans on: tile
 * skeletons are created behind `layer.show && _createTileImagerySkeletons(...)`,
 * so `show` gates loading and `alpha` does not. Cesium's own texture cache is
 * therefore the only cache available to us here — and it is BETTER than the
 * HTTP cache would have been, because the tiles it holds are already decoded
 * and uploaded to the GPU. Promoting a frame is a property assignment, not a
 * network round trip.
 *
 * So: while frame i is displayed, frames i+1…i+depth exist on the globe as
 * ordinary imagery layers at alpha 0, built by `providersFor` — the SAME
 * function the live path uses, which is the only reason the ring cannot show
 * something the paused globe would not.
 *
 * ORDERING, because it reads wrong: ring layers are appended, so they sit
 * ABOVE the current frame, and after a promote the frames still queued sit
 * above the visible one. They are at alpha 0, so they composite to nothing and
 * this is invisible — but anyone reading `viewer.imageryLayers` will see the
 * future stacked on top of the present and should know it is deliberate. */

// frame date → { key, built }, where `built` is one record per LAYER ID rather
// than a bare list of ImageryLayers: promoting means assigning back into
// `state.layers[id].layer` / `.cmpLayer`, so the id and the split pairing have
// to survive the wait.
const playPreload = new Map();

/* Everything the ring's layers were built FROM, except the date. Stored on each
 * entry and re-checked at promote time. The explicit invalidation
 * (`playbackPreloadClear`, called from `refreshTimedLayers`) is the real
 * defence; this is belt and braces behind it, because the cost of the two
 * disagreeing is a promoted frame that renders a comparison or a window the
 * user has already left — and the cost of the check is a string compare. */
function playbackFrameKey() {
  return [playbackLayerKey(), state.compareMode, state.compareYears,
    state.compareFixed || "", state.windowDays].join("|");
}

/* The EFFECTIVE depth, recomputed each frame because two of its three inputs
 * can change mid-playback. It is reported in the status line rather than kept
 * private: a device that quietly dropped to one frame of lookahead should say
 * so, or the difference between a smooth player and a stuttering one looks
 * like magic. */
function playbackPreloadDepth() {
  const layers = playbackLayers();
  if (!layers.length) return 0;
  /* Grid-only frame sets need no ring at all. A keyed grid paints from a baked
   * file rather than from tiles — the file is warmed a frame ahead by
   * `ensureGridMonth` below, and once it is in memory a GridProvider draws
   * synchronously — so a second copy of the layer at alpha 0 would buy nothing
   * and cost a full set of canvas tiles. This is also why playing currents or
   * the GFS forecast was already smooth before any of this existed. */
  if (layers.every((e) => e.cfg.grid)) return 0;
  let depth = PLAY_PRELOAD_DEPTH;
  // navigator.deviceMemory is Chromium-only and coarse (rounded to a power of
  // two, capped at 8), which is all we need it for: under 4 GB is a phone, and
  // a phone is where holding two speculative frames of GPU texture is felt.
  const mem = navigator.deviceMemory;
  if (typeof mem === "number" && mem > 0 && mem < 4) depth = 1;
  const tiles = viewer.scene.globe._surface?._tilesToRender?.length || 0;
  if (tiles > PLAY_PRELOAD_TILE_LIMIT) depth = 1;
  return depth;
}

// alpha 0, NOT show:false. That one line is the entire mechanism — see FACT 2.
function playbackPreloadAdd(p, cfg) {
  const layer = viewer.imageryLayers.addImageryProvider(p.provider);
  layer.alpha = 0;
  // …except a fine layer above its gate, which must not warm at all: hidden,
  // it holds no tiles and requests none, and is shown on promote if the
  // camera has come down by then (applyFineGate in playbackPromote).
  if (cfg?.fine) layer.show = !fineGated(cfg);
  if (p.splitDirection !== undefined) layer.splitDirection = p.splitDirection;
  return layer;
}

function playbackPreloadDestroy(rec) {
  try { if (rec.layer) viewer.imageryLayers.remove(rec.layer, true); } catch { /* already gone */ }
  try { if (rec.cmpLayer) viewer.imageryLayers.remove(rec.cmpLayer, true); } catch { /* already gone */ }
}

function playbackPreloadDrop(dateStr) {
  const held = playPreload.get(dateStr);
  if (!held) return;
  playPreload.delete(dateStr);
  for (const rec of held.built) playbackPreloadDestroy(rec);
}

/* Empty the ring. Called on stop, on halt, and — through `refreshTimedLayers`
 * — on every configuration change: the layer set, the comparison, the
 * aggregation window, a re-enumerated step or range. A stale ring is strictly
 * worse than no ring, because it promotes instantly and therefore hides the
 * fact that it is stale: the picture would simply be wrong, with no round trip
 * to notice. */
function playbackPreloadClear() {
  for (const d of [...playPreload.keys()]) playbackPreloadDrop(d);
}

/* Build frames i+1 … i+depth that the ring does not already hold. */
function playbackEnsurePreload(i) {
  const depth = playbackPreloadDepth();
  playback.preloadDepth = depth;

  const want = [];
  for (let k = i + 1; k <= i + depth && k < playback.frames.length; k++) {
    const d = playback.frames[k];
    if (!want.includes(d)) want.push(d);
  }

  /* The keyed grids are warmed for every wanted frame whatever the depth, and
   * this is the half of the old prefetch that always worked: it is a real
   * fetch of a real file WE host, one per year rather than one per tile, and
   * `loadGridMonth` will find it resolved instead of in flight. */
  for (const d of want) {
    for (const e of playbackLayers()) {
      if (!e.cfg.grid) continue;
      const g = gridsLoaded.get(e.cfg.id);
      if (!g) continue;
      try { ensureGridMonth(e.cfg, g, resolveGridMonth(g, d)); } catch { /* best effort */ }
    }
  }

  // Anything held that the playhead no longer wants is dead weight — a scrub, a
  // loop wrap or a jump strands entries that will never be promoted, and each
  // of them is a live imagery layer holding a full set of tiles.
  for (const d of [...playPreload.keys()]) {
    if (!want.includes(d)) playbackPreloadDrop(d);
  }
  if (!depth) return;

  const key = playbackFrameKey();
  for (const d of want) {
    if (playPreload.has(d)) continue;
    const built = [];
    let ok = true;
    for (const e of playbackLayers()) {
      const cfg = e.cfg;
      if (cfg.grid) continue;                 // warmed above; holds no tiles
      let r;
      try { r = providersFor(cfg, d); } catch { ok = false; break; }
      if (r.suppressed || !r.providers.length) continue;
      const rec = { id: cfg.id, layer: null, cmpLayer: null,
        isDelta: r.isDelta, isRatio: r.isRatio, isAggregate: r.isAggregate };
      rec.layer = playbackPreloadAdd(r.providers[0], cfg);
      if (r.providers[1]) rec.cmpLayer = playbackPreloadAdd(r.providers[1], cfg);
      built.push(rec);
    }
    /* A PARTIAL frame is not stored. If any layer failed to build, promoting
     * this entry would advance some layers to the new date and leave the rest
     * on the old one — a composite of two dates under one caption, which is
     * the exact failure mode §4 of the plan forbids. Falling back to the
     * ordinary refresh costs a round trip and is honest. */
    if (!ok || !built.length) {
      for (const rec of built) playbackPreloadDestroy(rec);
      continue;
    }
    playPreload.set(d, { key, built });
  }
}

/* Promote the ring's copy of `dateStr` into the live layers. Returns false if
 * the ring cannot serve this frame, in which case the caller takes the
 * ordinary `refreshTimedLayers({hold: true})` path unchanged.
 *
 * The cost of the promote is the assignment: the tiles are already decoded and
 * on the GPU (FACT 2), so `globe.tilesLoaded` is typically still true one tick
 * later and the player's wait collapses to the fps limit — which is the point
 * of the whole exercise, and why `playback.bound` should now normally read
 * "fps" rather than "network". */
function playbackPromote(dateStr) {
  const held = playPreload.get(dateStr);
  if (!held) return false;
  playPreload.delete(dateStr);
  if (held.key !== playbackFrameKey()) {
    // Every entry in the ring was built under the same key, so one stale entry
    // means the whole ring is stale — drop all of it rather than discovering
    // the same thing again two frames from now.
    for (const rec of held.built) playbackPreloadDestroy(rec);
    playbackPreloadClear();
    return false;
  }

  // The date moved, so the normals may be for the wrong month now — the same
  // background reload `refreshTimedLayers` does, and for the same reason: the
  // hover probe renders synchronously and cannot await one.
  if (state.layers["sst-anom"]?.layer) ensureSstNormals(state.date);

  let promoted = false;
  for (const rec of held.built) {
    const entry = state.layers[rec.id];
    if (!entry) { playbackPreloadDestroy(rec); continue; }
    // Through the EXISTING retirement queue: the outgoing generation keeps
    // painting until the globe reports its tiles settled, so the promote is
    // as blink-free as the ordinary held refresh — and bounded by the same
    // three-deep cap rather than by a second mechanism written here.
    retireLayer(rec.id);
    entry.layer = rec.layer;
    entry.layer.alpha = entry.alpha;
    if (rec.cmpLayer) {
      entry.cmpLayer = rec.cmpLayer;
      entry.cmpLayer.alpha = entry.alpha;
    }
    entry.isDelta = rec.isDelta;
    entry.isRatio = rec.isRatio;
    entry.isAggregate = rec.isAggregate;
    entry.suppressed = false;
    applyFineGate(entry);
    promoted = true;
  }
  if (!promoted) return false;

  /* The same two toasts the refresh path fires per frame, over the same set of
   * entries it fires them over (which includes SUPPRESSED ones — a layer
   * hidden by the window still has a date, and still has an archive that may
   * have ended). A frame that arrives instantly must not be a frame that says
   * less than a slow one: "never advance past a hole in silence" is a property
   * of the playback, not of the code path it happened to take. */
  for (const e of Object.values(state.layers)) {
    if (!(e.layer || e.suppressed) || !e.cfg.timed) continue;
    maybeArchiveToast(e.cfg, { replace: true });
    maybeAnnualToast(e.cfg, { replace: true });
  }
  updateLegends();
  scheduleSweep();
  updateSplitUI();
  return true;
}

/* ------------------------------------------------------------------ the loop
 * The playhead advances when the frame is ON SCREEN, not when a timer says so:
 *
 *     show frame i  →  wait for max(1/fps, tile queue empty)  →  i++
 *
 * setTimeout and promises, deliberately not requestAnimationFrame: these frames
 * are seconds apart, not 16 ms, and a rAF loop would spin sixty times for every
 * one of them. */
async function playbackRun(token) {
  while (playback.playing && token === playToken) {
    const t0 = performance.now();
    const promoted = await playbackShowFrame(playback.i);
    if (!playback.playing || token !== playToken) return;
    // Warm the NEXT frames while this one is on screen.
    playbackEnsurePreload(playback.i);

    /* A PROMOTED frame is already painted — its tiles were fetched, decoded and
     * uploaded to the GPU while an earlier frame was on screen — so there is
     * nothing left to wait for and the fps limit is the only one still binding.
     * That is the whole point of the ring, and the status line saying "fps"
     * rather than "network-bound" is how the user sees it.
     *
     * Skipping the wait here is not an optimisation, it is a correction: the
     * globe's tile queue is GLOBAL, so by this line it holds the speculative
     * requests `playbackEnsurePreload` just issued for frames i+1…i+depth.
     * Waiting on it would gate the VISIBLE playhead on frames nobody is looking
     * at yet — which is exactly the latency the buffer exists to absorb, paid
     * anyway. When the ring falls behind, frames stop being promoted, this
     * branch stops being taken, and the player goes back to advancing at the
     * network's pace on its own. */
    const why = promoted ? "settled" : await waitTilesSettled(PLAY_FRAME_CEILING_MS);
    if (!playback.playing || token !== playToken) return;
    const minMs = 1000 / (playback.fps || 1);
    const elapsed = performance.now() - t0;
    if (why === "settled" && elapsed < minMs) await sleepMs(minMs - elapsed);
    if (!playback.playing || token !== playToken) return;
    const total = performance.now() - t0;
    playback.actualFps = 1000 / Math.max(1, total);

    /* WHICH LIMIT ACTUALLY BOUND THIS FRAME — three answers, not two, because
     * the ring made the third one visible. Before it, a frame that took longer
     * than the fps budget was always waiting for tiles; now the tiles are
     * usually already there and the remaining cost is the app's own repaint
     * (the grid/yearly refreshes, and a render loop slow enough that even a
     * setTimeout fires late). Calling that "network-bound" would name the wrong
     * culprit, and calling it "at the requested speed" next to a measured
     * 0.5 fps would be the plain contradiction this panel exists to avoid. The
     * tolerance is deliberately loose: a frame within a third of its budget is
     * running at the requested speed for any purpose a viewer has. */
    playback.bound = (why !== "settled" || elapsed >= minMs) ? "network"
      : total > minMs * 1.35 ? "device"
        : "fps";

    if (playback.i + 1 < playback.frames.length) {
      playback.i++;
    } else if (playback.loop) {
      playback.i = 0;
    } else {
      playbackStop();
      return;
    }
  }
}

function playbackPlay() {
  if (playback.playing) return;
  if (!playback.frames.length) playbackRebuild();
  if (playback.frames.length < 2 || !playbackLayers().length) return;
  // Starting from the last frame with loop off would show one frame and stop.
  if (playback.i >= playback.frames.length - 1) playback.i = 0;
  playback.playing = true;
  playback.layerKey = playbackLayerKey();
  playbackRender();
  playbackRun(++playToken);
}

/* Stop leaves NOTHING of playback behind (plan §4). The retirement queue is
 * swept, so the globe holds no frame we have already left, and state.date stays
 * on the frame we stopped on — that is the picture on screen, and the date
 * input, the probe and the pixel card all read it.
 *
 * It deliberately does NOT rebuild the layers. There is nothing to hand back:
 * every frame WAS the ordinary date path (`refreshTimedLayers({hold: true})`),
 * so `state.layers` already holds exactly what the single-date path would have
 * built for this date. An unheld refresh here would destroy and refetch an
 * identical set of layers — which is to say it would end every playback with
 * the very blink the retirement queue exists to remove. */
function playbackStop() {
  playback.playing = false;
  playToken++;
  playback.bound = null;
  playback.actualFps = null;
  playback.preloadDepth = null;
  // Nothing of playback left behind includes the ring: those layers are frames
  // the user is no longer walking towards, each holding a full set of tiles,
  // and the next play may well be a different range entirely.
  playbackPreloadClear();
  sweepRetired();
  playbackRender();
}

/* A frame list is only valid for the layers it was enumerated from: switch NDVI
 * off mid-play and every remaining frame is a duplicate; switch daily SST on and
 * the monthly list is now showing one frame per month of a daily product. So a
 * layer change stops the player and re-enumerates, keeping the playhead's DATE
 * (its index means nothing in the new list). It does not auto-resume: the layer
 * set changed because the user changed it, and they are looking at the globe. */
function playbackInvalidate() {
  const at = playback.frames[playback.i] || state.date;
  playbackStop();
  playbackRebuild();
  playback.i = playbackNearest(at);
  playbackRender();
}

document.addEventListener("change", () => {
  const key = playbackLayerKey();
  if (key === playback.layerKey) return;
  if (playback.playing) { playbackInvalidate(); return; }
  // Not playing, but the panel is a read-out of a frame list that is a function
  // of the layers on — including the "switch a dated layer on" hint. An open
  // panel showing yesterday's layer set would be describing a playback that no
  // longer exists.
  if (playbackUiWired && !document.getElementById("panel-play")?.classList.contains("hidden")) {
    playbackRebuild();
    playbackRender();
  }
});

/* Streaming NASA tiles into a background tab is both rude and pointless
 * (plan §5). Resume only if we were the ones who paused it.
 *
 * This HALTS rather than stops: `playbackStop`'s parting act is an unheld
 * refresh, which would request a fresh set of tiles for a tab nobody is looking
 * at — the precise thing this handler exists to avoid. The picture is left
 * exactly as it was, which is also what the user comes back to. */
let playbackHiddenPause = false;
function playbackHalt() {
  playback.playing = false;
  playToken++;
  playback.bound = null;
  playback.actualFps = null;
  playback.preloadDepth = null;
  // The ring goes too, for the same reason the halt exists: frames warmed for
  // a tab nobody is looking at are texture memory spent on nothing.
  playbackPreloadClear();
  sweepRetired();
  playbackRender();
}
document.addEventListener("visibilitychange", () => {
  if (document.hidden) {
    if (playback.playing) { playbackHiddenPause = true; playbackHalt(); }
  } else if (playbackHiddenPause) {
    playbackHiddenPause = false;
    playbackPlay();
  }
});

/* ------------------------------------------------------------------- the tab */

// The whole record the layers on can actually show — the "all" preset. GIBS
// layers know their own `start`; a keyed grid's first baked month is its start
// (GLORYS reaches back to 1993, well before the date steppers' 2000 floor).
function playbackRecordStart() {
  let earliest = null;
  for (const e of playbackLayers()) {
    let s = null;
    if (e.cfg.grid) {
      const ms = gridMonths(gridsLoaded.get(e.cfg.id));
      if (ms && ms.length) s = ms[0].length === 7 ? `${ms[0]}-01` : ms[0];
    } else {
      s = e.cfg.start;
    }
    if (s && (!earliest || s < earliest)) earliest = s;
  }
  return earliest || "2000-01-01";
}

/* Same upper bound as every other date in the app (`uiMaxDate`, which a
 * forecast grid pushes past today), but a LOWER bound that follows the data
 * rather than the date steppers' 2000-01-01 GIBS floor: GLORYS currents are
 * baked back to 1993, and a playback panel that refused to start before 2000
 * would be hiding seven years of the archive the layer actually has. */
function clampPlayDate(d) {
  const hi = uiMaxDate();
  const rec = playbackRecordStart();
  const lo = rec < "2000-01-01" ? rec : "2000-01-01";
  return d > hi ? hi : d < lo ? lo : d;
}

let playbackUiWired = false;
function loadPlayback() {
  const startEl = document.getElementById("pb-start");
  const endEl = document.getElementById("pb-end");
  if (!startEl || !endEl) return;             // panel not in this build
  if (!playbackUiWired) {
    playbackUiWired = true;

    const rebuild = () => { playbackRebuild(); playbackRender(); };

    for (const el of [startEl, endEl]) {
      el.addEventListener("change", () => {
        if (!el.value) return;
        // Same bounds as the main date. Unlike the compare field this one may
        // safely write back: it is not the field whose caret a mid-typing
        // clamp would reset, because we only correct a COMPLETE out-of-range
        // date, and only downward to the max the UI admits exists.
        const c = clampPlayDate(el.value);
        if (c !== el.value) el.value = c;
        rebuild();
      });
    }

    document.getElementById("pb-presets")?.addEventListener("click", (e) => {
      const r = e.target.getAttribute?.("data-range");
      if (!r) return;
      const end = clampUiDate(state.date);
      let start;
      if (r === "all") {
        start = playbackRecordStart();
      } else {
        // stepCalendar walks one unit at a time, which is how "5 years back"
        // stays the app's own leap-day arithmetic instead of a second copy of
        // it written here.
        const years = r === "5y" ? 5 : 1;
        start = end;
        for (let k = 0; k < years; k++) start = stepCalendar(start, "-1y");
        start = clampPlayDate(start);
      }
      startEl.value = start;
      endEl.value = end;
      rebuild();
    });

    document.getElementById("pb-step")?.addEventListener("change", (e) => {
      playback.step = e.target.value;
      rebuild();
    });
    document.getElementById("pb-speed")?.addEventListener("change", (e) => {
      playback.fps = Number(e.target.value) || 2;
      playbackRender();
    });
    document.getElementById("pb-loop")?.addEventListener("change", (e) => {
      playback.loop = !!e.target.checked;
    });
    document.getElementById("pb-first")?.addEventListener("click", () => {
      playbackShowFrame(0);
    });
    document.getElementById("pb-last")?.addEventListener("click", () => {
      playbackShowFrame(playback.frames.length - 1);
    });
    document.getElementById("pb-play")?.addEventListener("click", () => {
      if (playback.playing) playbackStop(); else playbackPlay();
    });
    // Scrubbing while paused jumps to the frame. While PLAYING the slider is
    // an output, not an input — the loop owns the playhead, and letting both
    // write it would make the picture argue with the control.
    document.getElementById("pb-scrub")?.addEventListener("input", (e) => {
      if (playback.playing) return;
      /* Coalesced like every other date scrub — a slider drag fires one
       * `input` per pointer move (~60 a second), and each one used to cost a
       * whole visible tile set per layer. Reading `.value` at APPLY time
       * rather than capturing it means the frame that lands is the one the
       * thumb is on now, not the one it was on when the burst started. */
      scrubApply(() => playbackShowFrame(Number(e.target.value)));
    });
  }

  // Defaults on first open: end where the globe already is, start twelve months
  // back, both inside the bounds the date selector admits (a forecast grid can
  // push the max past today; `uiMaxDate` is the single answer to that).
  const max = uiMaxDate();
  const rec = playbackRecordStart();
  const min = rec < "2000-01-01" ? rec : "2000-01-01";
  for (const el of [startEl, endEl]) { el.max = max; el.min = min; }
  if (!endEl.value) endEl.value = clampPlayDate(state.date);
  if (!startEl.value) startEl.value = clampPlayDate(stepCalendar(endEl.value, "-1y"));
  const speed = document.getElementById("pb-speed");
  if (speed && speed.value) playback.fps = Number(speed.value) || playback.fps;
  const loop = document.getElementById("pb-loop");
  if (loop) playback.loop = !!loop.checked;
  const step = document.getElementById("pb-step");
  if (step && step.value) playback.step = step.value;

  playbackRebuild();
  playbackRender();
}

/* --------------------------------------------------------------------- tabs */

const tabs = { layers: "panel-layers", temp: "panel-temp", energy: "panel-energy",
  amoc: "panel-amoc", sealevel: "panel-sealevel", tides: "panel-tides",
  play: "panel-play", catalog: "panel-catalog", about: "panel-about" };
for (const t of Object.keys(tabs)) {
  document.getElementById(`tab-${t}`)?.addEventListener("click", () => {
    for (const [k, panel] of Object.entries(tabs)) {
      document.getElementById(panel)?.classList.toggle("hidden", k !== t);
      document.getElementById(`tab-${k}`)?.classList.toggle("active", k === t);
    }
    if (t === "amoc") loadAmoc();
    if (t === "sealevel") loadSeaLevel();
    if (t === "temp") loadTemp();
    if (t === "energy") loadEei();
    if (t === "tides") loadTides();
    if (t === "play") loadPlayback();
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
  // exported so a test can derive its own waits from the app's deadline
  // instead of hard-coding a second copy of the number
  PIXEL_DEADLINE_MS,
  get baseImageryLayer() { return baseImageryLayer; },
  parseColormap,
  parseColormapEntries,
  parseClassEntries,
  getClassEntries,
  getClassLut,
  windowSampleDates,
  addDays,
  windowLabel,
  SSTAggregateProvider,
  AggregateProvider,
  DeltaProvider,
  RatioProvider,
  getValueLut,
  probeCellBounds,
  get probeMark() { return probeMarkEnts; },
  ensureMarkVisible,
  SSTEnsembleProvider,
  spreadColor,
  get ensembleLayer() { return sstEnsembleLayer; },
  deltaColor,
  state,
  pointLayers,
  GIBS_LAYERS,
  GIBSGeographicTilingScheme,
  compareDate,
  sstAnomalyAt,
  ensureSstNormals,
  sstDailySeries,
  sstDailyAnomaly,
  get stations() { return stationsDs; },
  get rapid() { return rapidData; },
  get sealevel() { return seaLevelData; },
  loadSeaLevel,
  loadTides,
  get tides() { return tideData; },
  get tideSim() { return tideSim; },
  tideHeightAt,
  tideAstro,
  tideLive,
  tideSelectPoint,
  tideRateAt,
  tideExtrema,
  tideSpringRange,
  ensureTideLive,
  checkForNewBuild,
  movingMean,
  loadTemp,
  get gistemp() { return gistempData; },
  linTrend,
  probeValueAt,
  probeEntryValue,
  renderProbe,
  // provenance: what time a value was actually observed at (gibsTime is
  // already exported below — it is the input to all of these)
  whenOfGibs,
  whenOfGrid,
  whenOfLayer,
  whenAge,
  whenText,
  whenLabel,
  kelvinToC,
  colormapLayersTopDown,
  showPixelState,
  pixelInspectorEngaged,
  oceanColumnAt,
  SCENES,
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
  gridMonths,
  resolveGridMonth,
  refreshMonthlyGrids,
  ensureGridMonth,
  loadGridMonth,
  rampColor,
  gibsTime,
  gibsTimeStatic,
  gibsDomainUrl,
  parseGibsDomain,
  parsePeriod,
  snapToDomain,
  loadGibsDomain,
  gibsDomains,
  // the fine tier: the gate and the December-anchored annual arithmetic
  fineGated, updateFineGates, applyFineGate, annualYearOf, cameraHeight,
  // place names: the collections themselves, plus the pick-through helper, so a
  // test can prove a click on "Paris" still reaches the globe
  ensureCities,
  buildCitiesTo,
  cityRungAt,
  seeThrough,
  CITY_PICK,
  get cityLabels() { return cityLabels; },
  get cityPoints() { return cityPoints; },
  get bordersLayer() { return bordersLayer; },
  // the deep tier and the search over both tiers
  ensureGazetteer,
  refreshGazetteerLabels,
  searchPlaces,
  flyToPlace,
  markFoundPlace,
  placeViewHeight,
  placeCountry,
  placeRung,
  foldName,
  ensureIslands,
  buildIslandsTo,
  islandFar,
  islandExtentAt,
  get islData() { return islData; },
  get islLabels() { return islLabels; },
  get gazData() { return gazData; },
  get gazLabels() { return gazLabels; },
  get foundPlace() { return foundPlace; },
  get foundLabels() { return foundLabels; },
  get foundPoints() { return foundPoints; },
  // playback (E-041): the frame enumerator, the transport's state, the
  // retirement queue that stops the globe blinking between frames, and the
  // preload ring — `providersFor` is exported so a test can prove the ring and
  // the live path build a frame the same way, and `playPreload` so it can
  // count what is being held at alpha 0
  gibsProvider,
  providersFor,
  compareDateFor,
  get playPreload() { return playPreload; },
  playbackEnsurePreload,
  playbackPreloadDepth,
  playbackPreloadClear,
  playbackPromote,
  PLAY_PRELOAD_DEPTH,
  playbackFrames,
  playbackSignature,
  playbackAutoStep,
  playbackLayers,
  playback,
  playbackRebuild,
  playbackShowFrame,
  playbackPlay,
  playbackStop,
  loadPlayback,
  retireLayer,
  sweepRetired,
  get retiring() { return retiring; },
  waitTilesSettled,
  gibsUrlTemplate,
  refreshTimedLayers,
  applyDateMove,
  gibsRawLimit,
  get gibsRawActive() { return gibsRawActive; },
  get gibsSlowed() { return gibsSlowed; },
  removeLayer,
  addLayer,
  PLAY_MAX_FRAMES,
};

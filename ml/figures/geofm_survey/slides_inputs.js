// Slides 34–51: what a generic Earth embedding should see — input ladder, cones per input,
// the cone-native codec ("dots in, embedding out"), the zero-sum question, the four boundaries the
// sampler meets, the speeds the cone does not have, land/ice/air and the shared-channel set,
// phases, data continuity, El Niño 2026.
// Numbers: dataset specs verified 2 Sep 2026 (see generic-earth-embedding-inputs-proposal.md, Appendix A);
// speeds and memories are the same order-of-magnitude values the cone slides use.
module.exports = function (ctx) {
  const { pres, title, footer, NAVY, INK, MUTED, TEAL, SPACE, TIME, EMB, OUTPX, PALE, WHITE, GRIDLINE, FONT_H, FONT_B } = ctx;
  const LAND = "4E8A3E", OCEAN = SPACE, ATMOS = TIME, BIO = "1F9E89", STAT = "7A8797", DERIV = "A23B3B", OBS = "1F7A4D";
  const hdr = (t, o) => ({ text: t, options: Object.assign({ bold: true, color: WHITE, fill: { color: NAVY }, fontSize: 9, valign: "middle" }, o || {}) });
  const c = (t, o) => ({ text: t, options: Object.assign({ fontSize: 7.5, color: INK, valign: "top" }, o || {}) });
  const nm = (t, col) => ({ text: t, options: { bold: true, color: col || NAVY, fontSize: 8, valign: "top" } });
  const box = (s, x, y, w, h, col, fill) => s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x, y, w, h, fill: { color: fill || PALE }, line: { color: col || GRIDLINE, width: col ? 1.25 : 0.75 }, rectRadius: 0.08 });
  const txt = (s, runs, x, y, w, h, o) => s.addText(runs, Object.assign({ x, y, w, h, fontFace: FONT_B, fontSize: 9.5, color: INK, valign: "top", isTextBox: true, margin: 0 }, o || {}));
  const arrow = (s, x1, y1, x2, y2, col, wd) => s.addShape(pres.shapes.LINE, { x: Math.min(x1, x2), y: Math.min(y1, y2), w: Math.abs(x2 - x1), h: Math.abs(y2 - y1), flipH: x2 < x1, flipV: y2 < y1, line: { color: col || MUTED, width: wd || 1, endArrowType: "triangle" } });
  const reading = (s, t, y) => s.addText(t, { x: 0.6, y: y || 6.45, w: 12.1, h: 0.55, fontFace: FONT_B, fontSize: 8.5, italic: true, color: INK, valign: "top", isTextBox: true, margin: 0 });

  // ---------------------------------------------------------------- 35. the question + the ladder at a glance
  {
    const s = pres.addSlide(); s.background = { color: WHITE };
    title(s, "What should a generic Earth embedding see?", "The rule: every signal that is relevant to the four spheres at a point and cannot be derived from the other inputs. Thirteen rungs, finest first.");
    // ladder: 13 rungs, bar length ~ log10(resolution); finest at the bottom
    const rungs = [
      [0, "Anchor coordinates", "lat · lon · time · depth", 0, STAT, "the frame — not data"],
      [1, "10 m optical + radar", "Sentinel-2 · Sentinel-1", 10, LAND, "5–6 days"],
      [2, "30 m optical + thermal", "Landsat 8/9", 30, LAND, "8 days"],
      [3, "300 m–1 km daily radiometers", "Sentinel-3 OLCI/SLSTR · VIIRS", 500, OCEAN, "daily · first ocean rung"],
      [4, "Geostationary imagers", "GOES · MTG · Himawari", 2000, ATMOS, "every 10 min"],
      [5, "km-scale sparse sensors", "altimetry · SWOT · radar · lidar · TROPOMI", 5000, OCEAN, "tracks & swaths"],
      [6, "5–25 km gap-filled fields", "OSTIA · DUACS · IMERG · ASCAT · AMSR2 · SMAP", 10000, OCEAN, "daily–hourly · derived"],
      [7, "Reanalysis atmosphere", "ERA5 (31 km hourly) · IFS/GFS 0.25°", 31000, ATMOS, "hourly · derived"],
      [8, "Ocean interior", "Argo · BGC-Argo · drifters · moorings · GLORYS", 8000, OCEAN, "0–2000 m · first depth rung"],
      [9, "Ocean biosphere & carbon", "ocean colour L3 · PACE · SOCAT · BGC-Argo", 4000, BIO, "daily · point obs"],
      [10, "Land biosphere & water", "LAI · SIF · LST · soil · snow · fire · GRACE", 1000, BIO, "300 m–300 km"],
      [11, "In-situ atmosphere (points)", "stations · radiosondes · aircraft · ships · buoys", 0, ATMOS, "the ground truth"],
      [12, "Static context", "DEM · bathymetry · land cover · soil · canopy · population", 30, STAT, "v = 0, τ = ∞"],
    ];
    const y0 = 1.62, rowH = 0.385, x0 = 0.6, lx = 2.55, bx = 3.25, bwMax = 3.9;
    const len = r => r === 0 ? 0.12 : 0.35 + (Math.log10(r) - 1) * (bwMax - 0.35) / (Math.log10(31000) - 1);
    rungs.slice().reverse().forEach((r, i) => {
      const y = y0 + i * rowH;
      const [n, name, src, res, col, note] = r;
      s.addText([{ text: `${n}  `, options: { bold: true, color: col, fontSize: 9 } }, { text: name, options: { bold: true, color: INK, fontSize: 8.5 } }], { x: x0, y, w: lx, h: rowH, fontFace: FONT_B, valign: "middle", isTextBox: true, margin: 0 });
      const L = len(res);
      s.addShape(pres.shapes.RECTANGLE, { x: bx, y: y + 0.07, w: L, h: rowH - 0.14, fill: { color: col, transparency: 25 }, line: { width: 0 } });
      if (L >= 1.7) s.addText(src, { x: bx + 0.08, y, w: L - 0.12, h: rowH, fontFace: FONT_B, fontSize: 7.5, bold: true, color: WHITE, valign: "middle", isTextBox: true, margin: 0 });
      else s.addText(src, { x: bx + L + 0.08, y, w: 4.2 - L, h: rowH, fontFace: FONT_B, fontSize: 7.5, color: INK, valign: "middle", isTextBox: true, margin: 0 });
      s.addText(note, { x: 7.55, y, w: 1.6, h: rowH, fontFace: FONT_B, fontSize: 7, italic: true, color: MUTED, valign: "middle", isTextBox: true, margin: 0 });
    });
    s.addShape(pres.shapes.LINE, { x: bx, y: y0 + 13 * rowH + 0.02, w: bwMax + 0.3, h: 0, line: { color: GRIDLINE, width: 0.75, endArrowType: "triangle" } });
    s.addText("bar = native pixel size (log scale, 10 m → 31 km)", { x: bx, y: y0 + 13 * rowH + 0.05, w: 4.5, h: 0.25, fontFace: FONT_B, fontSize: 7.5, italic: true, color: MUTED, isTextBox: true, margin: 0 });
    // legend
    [[LAND, "land"], [OCEAN, "ocean"], [ATMOS, "atmosphere"], [BIO, "biosphere"], [STAT, "static / frame"]].forEach(([col, t], i) => {
      s.addShape(pres.shapes.RECTANGLE, { x: 7.55 + i * 1.02, y: y0 + 13 * rowH + 0.1, w: 0.14, h: 0.14, fill: { color: col, transparency: 25 }, line: { width: 0 } });
      s.addText(t, { x: 7.72 + i * 1.02, y: y0 + 13 * rowH + 0.04, w: 0.9, h: 0.25, fontFace: FONT_B, fontSize: 7.5, color: MUTED, isTextBox: true, margin: 0 });
    });
    // rule box
    box(s, 9.35, 1.55, 3.35, 5.15, NAVY, "F4F6F9");
    txt(s, [
      { text: "Three tests for a derived product", options: { bold: true, color: NAVY, fontSize: 11, breakLine: true } },
      { text: "Observations (a reflectance, a radar echo, a float's thermometer) are irreducible: always in. A derived product (a gap-filled map, a reanalysis, an ML reconstruction) is admitted only if:", options: { fontSize: 9, breakLine: true, paraSpaceAfter: 5 } },
      { text: "1  it carries information we do not ingest raw — ERA5 holds radiosondes, aircraft, radiances; GLORYS holds a full ocean model around the floats;", options: { fontSize: 9, breakLine: true, paraSpaceAfter: 4 } },
      { text: "2  it is flagged derived and dropped from the input more often, so the model must also work without it;", options: { fontSize: 9, breakLine: true, paraSpaceAfter: 4 } },
      { text: "3  its effective time is shifted by its assimilation window, so it is not a peek into the future (DUACS: ±6 weeks; GLORYS: 7 days; ERA5: 12 h).", options: { fontSize: 9, breakLine: true, paraSpaceAfter: 6 } },
      { text: "Dropped as derivable (targets only): NDVI and band ratios, solar geometry, slope/aspect, geostrophic currents from sea-level slope, per-scene class maps, Stokes drift.", options: { fontSize: 8.5, italic: true, color: MUTED } },
    ], 9.5, 1.65, 3.05, 5.0);
    footer(s);
  }

  // ---------------------------------------------------------------- 36–38. the ladder in detail (three tables)
  const ladderSlide = (n, sub, rows, rowH) => {
    const s = pres.addSlide(); s.background = { color: WHITE };
    title(s, `Input ladder ${n}`, sub);
    const head = [hdr("Rung"), hdr("Input (native resolution · cadence)"), hdr("What it contains, in plain English"), hdr("O / D", { align: "center" }), hdr("Adds — what nothing below it has")];
    const body = rows.map(r => [
      { text: String(r[0]), options: { bold: true, color: r[5], fontSize: 11, valign: "top", align: "center" } },
      { text: r[1], options: { bold: true, color: NAVY, fontSize: 8.5, valign: "top" } },
      c(r[2], { fontSize: 8.5 }),
      { text: r[3], options: { bold: true, color: r[3].startsWith("O") ? OBS : DERIV, fontSize: 9, valign: "top", align: "center" } },
      c(r[4], { fontSize: 8.5 }),
    ]);
    s.addTable([head, ...body], { x: 0.6, y: 1.5, w: 12.1, colW: [0.5, 3.1, 4.6, 0.55, 3.35], fontFace: FONT_B, border: { type: "solid", color: GRIDLINE, pt: 0.5 }, rowH: [0.3, ...rowH], margin: 0.05, autoPage: false });
    footer(s);
  };
  ladderSlide("I — fine and fast (rungs 0–4)", "Land at 10–30 m, the first ocean-covering rung at 300 m–1 km, and the atmosphere's 10-minute movie", [
    [0, "Anchor coordinates — latitude, longitude, time of year, time of day, depth", "Where and when we are looking. Not data, but the model needs it: 15 °C means something different in January at 60° N than in July at the equator.", "—", "The frame of reference; every other input is relative to it.", STAT],
    [1, "Sentinel-2 L2A (13 bands, 10/20/60 m, 5-day revisit with two satellites; three in orbit to end-2026) · Sentinel-1 GRD (VV/VH radar, 10 m pixels, 6-day repeat; S1C + S1D)", "Sentinel-2: a colour-plus-infrared photograph of the sunlit surface, corrected for the atmosphere — crops, forests, soil, water, snow, buildings. Sentinel-1: a radar image, day or night, through cloud, of how rough and how wet the surface is — floods, sea-ice edges, ships, soil moisture.", "O", "Field-scale texture and the fine state of the land biosphere; radar sees through cloud and at night. Land and coast only.", LAND],
    [2, "Landsat 8/9 Collection 2 (surface reflectance + surface temperature, 30 m, 8-day combined; archive to 1982)", "The same kind of photograph plus a thermal image — how warm the ground is — at field scale, with a forty-year archive.", "O", "Thermal at field scale; the long record for phenology and change.", LAND],
    [3, "Sentinel-3 OLCI (21 bands, 300 m, < 2 days) + SLSTR (500 m / 1 km, sea- and land-surface temperature, daily) · VIIRS NOAA-20/21 (375/750 m daily; MODIS retiring late 2026/27)", "Daily whole-planet pictures in visible, infrared and thermal. Over the ocean the colour of the water says how much phytoplankton is in it; the thermal channels give the skin temperature of sea and land; hot spots are fires.", "O", "First rung that covers the ocean surface, and the first daily one: ocean colour (the biosphere), fires, snow, daily land temperature.", OCEAN],
    [4, "GOES-19/18 ABI · Meteosat-12 FCI · Himawari-9 AHI (16 channels, 0.5–2 km, every 10 minutes; full disks to about ± 60–70° latitude)", "A continuous movie of clouds, water vapour and surface radiance from fixed points in the sky; clouds visibly form, move and dissolve between frames.", "O", "Atmospheric motion at minute scale — the only direct view of the wind field acting on clouds, and of the daily cycle. The fast end of the cone.", ATMOS],
  ], [0.62, 1.3, 0.72, 1.25, 1.0]);
  ladderSlide("II — fields and ocean memory (rungs 5–8)", "Sparse kilometre-scale sensors, the gap-filled daily maps, the reanalysis atmosphere, and the first rung below the surface", [
    [5, "SWOT (2 km swath sea level, 21-day) · nadir altimetry Sentinel-6/Jason-3/Sentinel-3 (~7 km along-track, ~10-day) · Sentinel-1 ocean winds (1 km on swaths) · ground weather radar (MRMS 1 km/2 min; OPERA 1 km/5 min) · GLM lightning (8 km) · GEDI & ICESat-2 lidar (25 m / 11 m footprints) · TROPOMI, OCO-2/3 (trace-gas columns)", "Sea-surface height to a few centimetres along the satellite's track — the ocean's pressure map, whose slopes drive currents; rain seen from the ground every few minutes; where lightning strikes; the height of trees and of sea ice; the gases in the air column.", "O", "Sea level (hence currents), storm-scale rain, 3-D structure of vegetation and ice, greenhouse-gas columns. All sparse in space or time — the case where a set of dots beats a grid.", OCEAN],
    [6, "OSTIA SST (5 km daily; MUR 1 km) · DUACS sea level L4 (0.125° daily, with geostrophic u, v) · IMERG rain (0.1°, 30 min) · ASCAT winds (12.5/25 km, ≈ twice daily) · AMSR2 sea-ice concentration (10 / 6.25 km daily) · ice drift (62.5 km, 2-day) · SMAP soil moisture (36 km daily; 9 km 3-hourly model) · SSS L4 (weekly) · Copernicus waves (1/12°, 3-hourly, model)", "The cleaned-up daily maps: sea temperature with the clouds filled in, sea level everywhere and not only under the track, rain everywhere, wind over the whole ocean, how much of the sea is ice and how fast it drifts, how wet the top centimetres of soil are.", "D*", "Completeness and a usable initial state for the slow media; they merge sources we do not ingest raw (AVHRR, microwave SST, every altimeter). Flag, mask heavily, time-shift by their windows. (*ASCAT, SMAP L3 and AMSR2 are swath observations on a grid: O.)", OCEAN],
    [7, "ERA5 (31 km native, 0.25° grid, hourly, 1940–, ~5-day latency): 2 m temperature, 10 m wind, sea-level pressure, precipitation, surface radiation, cloud cover, boundary-layer height, dewpoint, 37 pressure levels · real time: ECMWF IFS/AIFS open data or GFS (0.25°, 6-hourly)", "The best physically consistent estimate of the whole atmosphere at every hour — temperature, wind, pressure, humidity, radiation, clouds, at the surface and aloft — made by running a weather model and continuously nudging it towards millions of observations.", "D", "The forcing of everything else: heat and momentum into ocean and land; the vertical structure of the air. ERA5's SST and sea ice are its inputs, not outputs — never use them as ocean observations.", ATMOS],
    [8, "Argo core (T, S profiles 0–2000 m every 10 days; 4,372 floats, ~150 k profiles/yr) · Deep Argo (to 4–6 km; 219) · BGC-Argo (O₂, NO₃, pH, chlorophyll, backscatter, light; 989) · moorings (TAO, OceanSITES) · drifters (1,317; velocity at 15 m + SST) · GLORYS12 reanalysis (1/12°, 50 levels, daily, 1993–) · Copernicus 1/12° analysis/forecast (real time)", "What the ocean does below the surface: how warm and salty it is at each depth (heat content, density, the mixed layer), how fast and where it flows, where heat and fresh water are stored. Floats are sparse points; the reanalysis is a model's gridded estimate around them.", "O + D", "First rung below the surface: heat capacity, density gradients (thermal wind), the AMOC — the slow memory of the system. Only dots do this justice; gridding Argo throws away its depth resolution.", OCEAN],
  ], [1.42, 1.42, 1.1, 1.35]);
  ladderSlide("III — biospheres, truth, stage (rungs 9–13)", "Ocean carbon, land carbon and water, the point observations the reanalyses are fitted to, and the static boundary conditions", [
    [9, "Ocean colour L3 (OC-CCI / GlobColour, 4 km daily, cloud gaps kept) · PACE OCI (hyperspectral 1.2 km, 2-day, 2024–) · SOCAT surface CO₂ (44 M point obs) · BGC-Argo (rung 8) · Copernicus surface carbon (0.25° monthly, ML) · SeaFlux (1° monthly)", "How much plant life is in the surface ocean and of what kind; how much CO₂ the water holds relative to the air — whether the ocean is absorbing or releasing carbon; oxygen and nutrients at depth.", "O + D", "The ocean carbon sink and its drivers.", BIO],
    [10, "LAI / FAPAR (300 m 10-day) · SIF fluorescence (TROPOMI ~7 km daily) · LST (1 km daily) · evapotranspiration (500 m 8-day; GLEAM 0.1° daily — models) · soil moisture (ASCAT 12.5 km, ~2 h) · snow (500 m daily) · fire (VIIRS 375 m, < 3 h) · water storage (GRACE-FO, ~300 km monthly) · rivers (GRDC gauges; GloFAS 0.05° daily model; Hydroweb heights) · flux towers (half-hourly CO₂, water, heat)", "How much leaf area there is and how actively it photosynthesises (fluorescence is the plants' own faint glow), how much water evaporates, how much snow and soil water is stored, where it burns, how much water sits in rivers, lakes and aquifers.", "O + D", "The land carbon and water cycles — the fourth sphere. Almost all of it is column-only: it does not move sideways.", BIO],
    [11, "Surface stations (ISD > 14,000 active, hourly; GHCN-Daily > 100,000) · radiosondes (~800 in real time, 00/12 UTC) · aircraft (AMDAR) · ships & buoys (ICOADS, NDBC) · tide gauges (GLOSS)", "Thermometers, barometers and anemometers on the ground, balloons through the atmosphere, instruments on ships and buoys, sea-level gauges on coasts — the measurements rung 7 is fitted to.", "O", "Irreducible ground truth. Redundant while rung 7 is present; essential when it is masked — the model must rebuild the atmosphere from points.", ATMOS],
    [12, "Copernicus DEM (30 m) · GEBCO bathymetry (~450 m) · land cover (WorldCover 10 m; CCI 300 m annual) · SoilGrids (250 m) · canopy height (10 m / 1 m) · glacier outlines · population (100 m) · night lights (500 m) · roads", "The stage: how high, how deep, what kind of surface and soil, how tall the trees, where the people are.", "O / D", "Boundary conditions: depth sets what currents can do, terrain sets rain and rivers, soil sets what water does. One dot each — v = 0, τ = ∞.", STAT],
    [13, "Optional: aerosol optical depth (MAIAC 1 km daily) · MERRA-2 aerosols (0.5°, hourly, 3-week latency) · trace-gas columns (rung 5)", "Dust, smoke and pollution in the air, and the greenhouse-gas columns above each point.", "O + D", "Radiative-forcing detail and the carbon-cycle link between land, ocean and air.", ATMOS],
  ], [1.05, 1.5, 0.95, 0.95, 0.62]);

  // ---------------------------------------------------------------- 39. observed vs derived, and the leakage trap
  {
    const s = pres.addSlide(); s.background = { color: WHITE };
    title(s, "Observed, derived, or derivable — and the leakage trap", "Why the source of each dot is a token the model sees, and why delayed-time products must be time-shifted");
    const cols = [
      ["Always in — observations", OBS, "A number a sensor produced. Reflectances, radar echoes, radar ranges to the sea surface, float thermometers, gauges, towers, balloons. Irreducible: nothing else contains them.", ["Sentinel-1/2/3, Landsat, VIIRS, geostationary imagers", "altimeter tracks, SWOT swaths, lidar footprints", "Argo / BGC-Argo profiles, drifters, moorings", "ocean-colour L3 (gaps stay gaps), SOCAT", "stations, radiosondes, ships, buoys, tide gauges", "swath products on a grid (ASCAT, SMAP L3, AMSR2)"]],
      ["Admitted derived — flagged", DERIV, "A number a program produced from observations plus a model or a statistical prior. Admitted when it carries information we do not ingest raw. Every dot carries a source token; derived sources are dropped from the input at ≥ 50 % of anchors, so the model must also work from observations alone.", ["ERA5 / IFS / GFS (radiosondes, aircraft, radiances inside)", "GLORYS12, Copernicus 1/12° analysis (full ocean model around the floats)", "OSTIA, MUR, DUACS L4, IMERG, SSS L4 (multi-sensor merges)", "SMAP L4, GloFAS, GLEAM, MOD16 (land models)", "Copernicus carbon, SeaFlux (ML reconstructions)"]],
      ["Dropped — derivable from the ladder", MUTED, "A function of inputs already present. Not an input; an optional target (a masked-channel loss on it costs nothing and gives free supervision).", ["NDVI / EVI and other band ratios (rungs 1–3)", "solar zenith, day length, Coriolis f (position + time)", "slope, aspect, distance to coast (rung 12)", "geostrophic u, v and OSCAR (SSH, wind, SST)", "Dynamic World and per-scene classifiers (Sentinel-2)", "Stokes drift (wave spectrum)"]],
    ];
    cols.forEach(([h, col, body, items], i) => {
      const x = 0.6 + i * 4.1;
      box(s, x, 1.55, 3.9, 3.45, col, i === 2 ? "F4F6F9" : PALE);
      txt(s, [
        { text: h, options: { bold: true, color: col, fontSize: 12.5, breakLine: true, paraSpaceAfter: 5 } },
        { text: body, options: { fontSize: 10, breakLine: true, paraSpaceAfter: 7 } },
        ...items.map((t, j) => ({ text: t, options: { bullet: true, fontSize: 9.5, breakLine: j < items.length - 1, paraSpaceAfter: 3 } })),
      ], x + 0.15, 1.65, 3.6, 3.3);
    });
    // leakage band
    box(s, 0.6, 5.15, 12.1, 1.75, TIME, "FFF6EE");
    txt(s, [
      { text: "The leakage trap: delayed-time products look backwards and forwards.  ", options: { bold: true, color: TIME, fontSize: 12 } },
      { text: "A gap-filled map or a reanalysis for day t is built from observations on both sides of t. If the codec is trained to predict the present or the future from the past, a derived dot at lag ℓ secretly contains information from lag ℓ − (half its window) — sometimes from after the anchor. Windows to respect:", options: { fontSize: 10.5, breakLine: true, paraSpaceAfter: 5 } },
      { text: "DUACS sea level L4 — delayed-time grids use a centred ±6-week window of tracks (the NRT stream only the past 7 weeks)  ·  GLORYS12 — 7-day assimilation cycle  ·  ERA5 — 12-hour assimilation window  ·  OSTIA / MUR — a window of recent days  ·  ocean-colour L4 — space–time interpolation across cloud gaps (use L3)", options: { fontSize: 10, breakLine: true, paraSpaceAfter: 5 } },
      { text: "Rule: shift each derived source's effective time by its window, never use a delayed-time product at lag 0, and train the forecasting part of the objective on the near-real-time streams, which only see the past (ERA5T, OSTIA NRT, DUACS NRT, the Copernicus analysis). Like ERA5T vs ERA5, this yields two embeddings per anchor: an NRT one (honest for forecasting) and a final one.", options: { fontSize: 10, italic: true } },
    ], 0.75, 5.22, 11.8, 1.65);
    footer(s);
  }

  // ---------------------------------------------------------------- 40. cone families
  {
    const s = pres.addSlide(); s.background = { color: WHITE };
    title(s, "Five cone families — one shape per rung", "Apply Δx ≤ v·(Δt+ℓ), Δt+ℓ ≤ τ to every input: the shape of the context each channel needs differs by orders of magnitude");
    const cc = (t) => c(t, { fontSize: 8 });
    const rows = [
      [hdr(""), hdr("Family"), hdr("Carrier — what moves the information"), hdr("v (order of magnitude)"), hdr("τ — memory at a fixed point"), hdr("Reach after 1 day · 1 week · 1 month"), hdr("Ladder rungs")],
      ["A", nm("Fast, wide, short", ATMOS), cc("wind, synoptic weather"), cc("10 m/s ≈ 860 km/day"), cc("3–10 days (rain: hours to 1 day)"), cc("860 km · 6,000 km · capped at 10,000 km"), cc("4 · 5 (radar, lightning) · 6 (IMERG, ASCAT) · 7 (wind, pressure, cloud)")],
      ["B", nm("Slow, narrow, long", OCEAN), cc("ocean currents and eddies; Rossby waves for sea level; Kelvin / coastal waves along boundaries"), cc("currents 0.1–0.2 m/s ≈ 13 km/day · Rossby 0.03 m/s ≈ 2.6 km/day westward · coastal 2.5 m/s"), cc("eddies, sea level 2–6 months · interior T, S 1–10 yr (thermocline), decades (deep)"), cc("13 · 90 · 400 km (currents) · 3 · 18 · 80 km (Rossby)"), cc("5 (altimetry, SWOT) · 6 (DUACS, SSS) · 8 (Argo, GLORYS, drifters)")],
      ["C", nm("L-shaped union", TEAL), cc("fast atmospheric forcing on a slow medium: the forcing arm is A, the memory arm is B"), cc("A at short lags; B at long lags"), cc("SST, SSS 3–6 months (mixed-layer inertia) · sea ice seasonal + multi-year thickness · chlorophyll 1–4 weeks"), cc("forcing arm 860 km at lag 0; memory arm 100 km + 13 km/day"), cc("3 (SST) · 6 (OSTIA, MUR, ice) · 9 (chlorophyll)")],
      ["D", nm("Column-only (land)", LAND), cc("nothing horizontal — soil, plants, snow stay put; forced from above by A; rivers carry water downstream (~1 m/s along the network); sea ice drifts 5–10 km/day"), cc("0 horizontally (river network ≈ 90 km/day along channels)"), cc("surface soil moisture 1–3 weeks, root zone months · vegetation weeks to seasons · snow seasonal · water storage months to years · fire days"), cc("L_corr ≈ 1–10 km at every lag, plus the atmosphere disc at lags ≤ τ_atm"), cc("1 · 2 · 10 · part of 12")],
      ["E", nm("Static", STAT), cc("none"), cc("0"), cc("∞"), cc("one dot"), cc("12")],
    ].map((r, i) => i === 0 ? r : [{ text: r[0], options: { bold: true, color: r[1].options.color, fontSize: 14, align: "center", valign: "middle" } }, ...r.slice(1)]);
    const RH = [0.3, 0.8, 1.0, 0.95, 1.15, 0.5];
    s.addTable(rows, { x: 0.6, y: 1.5, w: 10.3, colW: [0.4, 1.25, 2.15, 1.75, 1.95, 1.35, 1.45], fontFace: FONT_B, border: { type: "solid", color: GRIDLINE, pt: 0.5 }, rowH: RH, margin: 0.04, autoPage: false });
    // glyphs: shape of each family drawn in (Δx horizontal, lag vertical, present at the bottom), one per row
    const gx = 11.05, gw = 1.65;
    const glyph = (i, col, draw) => {
      const rowTop = 1.5 + RH.slice(0, i + 1).reduce((a, b) => a + b, 0), rh = RH[i + 1];
      const gh = Math.min(0.86, rh - 0.08), y = rowTop + (rh - gh) / 2;
      s.addShape(pres.shapes.RECTANGLE, { x: gx, y, w: gw, h: gh, fill: { color: "FBFCFD" }, line: { color: GRIDLINE, width: 0.5 } });
      draw(gx + gw / 2, y + gh - 0.06, gh - 0.12, col);
    };
    // A: flat wide disc at the bottom (short history)
    glyph(0, ATMOS, (cx, yb, h, col) => { s.addShape(pres.shapes.RECTANGLE, { x: cx - 0.75, y: yb - 0.16, w: 1.5, h: 0.16, fill: { color: col, transparency: 30 }, line: { width: 0 } }); });
    // B: thin tall column, widening with lag and tilted upstream
    glyph(1, OCEAN, (cx, yb, h, col) => { s.addShape(pres.shapes.CUSTOM_GEOMETRY, { x: cx - 0.45, y: yb - h, w: 0.6, h, fill: { color: col, transparency: 30 }, line: { width: 0 }, points: [{ x: 0.4, y: h }, { x: 0.52, y: h }, { x: 0.6, y: 0 }, { x: 0, y: 0 }, { x: 0.4, y: h, close: true }] }); });
    // C: L — wide at the bottom, narrow going up
    glyph(2, TEAL, (cx, yb, h, col) => { s.addShape(pres.shapes.RECTANGLE, { x: cx - 0.75, y: yb - 0.14, w: 1.5, h: 0.14, fill: { color: col, transparency: 30 }, line: { width: 0 } }); s.addShape(pres.shapes.RECTANGLE, { x: cx - 0.1, y: yb - h, w: 0.24, h, fill: { color: col, transparency: 30 }, line: { width: 0 } }); });
    // D: column with a hat
    glyph(3, LAND, (cx, yb, h, col) => { s.addShape(pres.shapes.RECTANGLE, { x: cx - 0.05, y: yb - h, w: 0.1, h, fill: { color: col, transparency: 30 }, line: { width: 0 } }); s.addShape(pres.shapes.RECTANGLE, { x: cx - 0.75, y: yb - 0.12, w: 1.5, h: 0.12, fill: { color: ATMOS, transparency: 30 }, line: { width: 0 } }); });
    // E: a dot
    glyph(4, STAT, (cx, yb, h, col) => { s.addShape(pres.shapes.OVAL, { x: cx - 0.05, y: yb - 0.1, w: 0.1, h: 0.1, fill: { color: col }, line: { width: 0 } }); });
    reading(s, "Glyphs: Δx sideways, lag upwards, present at the bottom. Depth is a sixth axis, not a family: the mixed layer (top 10–100 m, deeper in winter) is stirred within hours and belongs to the surface cone; below it memory grows with depth — months at 100–300 m, years at 500–1,500 m, decades below 2,000 m — while vertical speeds are ~1 m/day. Depth dots are log-spaced (0, 10, 30, 60, 100, 200, 400, 700, 1000, 1500, 2000 m), dense near the surface, each band with its own τ; the atmosphere mirrors this at 1000, 850, 500, 250 hPa. Consequence: fine rungs want a small dense patch and a long local history; coarse physical rungs a large sparse far field and a short (A) or thin-and-long (B) history; the interior depth and years. No fixed grid × window serves all three — a per-channel dot sampler does.", 6.3);
    footer(s);
  }

  // ---------------------------------------------------------------- 41. the cone-native codec
  {
    const s = pres.addSlide(); s.background = { color: WHITE };
    title(s, "Proposal — a cone-native codec: dots in, embedding out", "The codec reads (Δx, Δy, depth, lag, channel, value) samples from each channel's cone and predicts the dots it does not see — including future ones");
    // left: cone with dots (space × lag, depth as a small stack)
    const px = 0.6, py = 1.6, pw = 4.0, ph = 3.65;
    s.addShape(pres.shapes.RECTANGLE, { x: px, y: py, w: pw, h: ph, fill: { color: "FBFCFD" }, line: { color: GRIDLINE, width: 0.5 } });
    const ax = px + 0.45, ay = py + ph - 0.45, aw = pw - 0.7, ah = ph - 0.75;
    arrow(s, ax, ay, ax + aw, ay, INK, 1); arrow(s, ax, ay, ax, ay - ah, INK, 1);
    s.addText("distance from the anchor Δx (log)", { x: ax, y: ay + 0.03, w: aw, h: 0.25, fontFace: FONT_B, fontSize: 7.5, color: MUTED, align: "center", isTextBox: true, margin: 0 });
    s.addText("lag ℓ (log) — into the past", { x: ax - 0.2 - 0.95, y: ay - ah / 2 - 0.12, w: 1.9, h: 0.25, fontFace: FONT_B, fontSize: 7.5, color: MUTED, align: "center", rotate: 270, isTextBox: true, margin: 0 });
    // cone wedges: A (wide, low), B (thin, tall), C union — as translucent polygons
    const poly = (pts, col) => s.addShape(pres.shapes.CUSTOM_GEOMETRY, { x: ax, y: ay - ah, w: aw, h: ah, fill: { color: col, transparency: 70 }, line: { color: col, width: 0.75 }, points: pts.map(p => ({ x: p[0] * aw, y: (1 - p[1]) * ah })).concat([{ x: pts[0][0] * aw, y: (1 - pts[0][1]) * ah, close: true }]) });
    poly([[0, 0], [0.95, 0], [0.95, 0.22], [0, 0.22]], ATMOS);            // wind: wide, short
    poly([[0, 0], [0.12, 0], [0.42, 0.95], [0, 0.95]], OCEAN);           // currents: thin, tall
    poly([[0, 0], [0.7, 0], [0.7, 0.12], [0.25, 0.12], [0.3, 0.6], [0, 0.6]], TEAL); // SST: L
    // dots
    const seed = [0.13, 0.71, 0.29, 0.86, 0.5, 0.07, 0.62, 0.94, 0.38, 0.19, 0.77, 0.55, 0.03, 0.46, 0.9, 0.24, 0.66, 0.33, 0.81, 0.11, 0.58, 0.97, 0.42, 0.15, 0.7];
    let k = 0; const rnd = () => seed[(k++) % seed.length];
    const dot = (fx, fy, col, big) => s.addShape(pres.shapes.OVAL, { x: ax + fx * aw - (big ? 0.07 : 0.045), y: ay - fy * ah - (big ? 0.07 : 0.045), w: big ? 0.14 : 0.09, h: big ? 0.14 : 0.09, fill: { color: col }, line: { color: WHITE, width: 0.5 } });
    for (let i = 0; i < 9; i++) dot(0.05 + rnd() * 0.88, rnd() * 0.2, ATMOS);
    for (let i = 0; i < 8; i++) { const fy = rnd() * 0.93; dot(rnd() * (0.1 + 0.32 * fy), fy, OCEAN); }
    for (let i = 0; i < 6; i++) { const fy = rnd() * 0.58; dot(rnd() * (0.65 - 0.4 * fy), fy, TEAL); }
    dot(0, 0, OUTPX, true);
    s.addText("anchor (x, y, z, t)", { x: ax + 0.1, y: ay - 0.32, w: 1.4, h: 0.22, fontFace: FONT_B, fontSize: 7.5, bold: true, color: INK, isTextBox: true, margin: 0 });
    // depth stack
    [0, 1, 2, 3].forEach(i => s.addShape(pres.shapes.OVAL, { x: ax + 0.02 + i * 0.05, y: ay + 0.02 + i * 0.05, w: 0.08, h: 0.08, fill: { color: OCEAN, transparency: 20 + i * 15 }, line: { width: 0 } }));
    s.addText("depth dots\n(profiles as they are)", { x: ax + 0.3, y: ay + 0.08, w: 1.5, h: 0.32, fontFace: FONT_B, fontSize: 6.5, italic: true, color: MUTED, isTextBox: true, margin: 0 });
    s.addText([{ text: "wind (A) · ", options: { color: ATMOS, bold: true } }, { text: "currents, SSH (B) · ", options: { color: OCEAN, bold: true } }, { text: "SST (C)", options: { color: TEAL, bold: true } }, { text: " — each channel samples its own cone", options: { color: MUTED } }], { x: px, y: py + ph + 0.04, w: pw, h: 0.25, fontFace: FONT_B, fontSize: 7.5, isTextBox: true, margin: 0 });
    // middle: pipeline boxes
    const bx = 5.0, bw = 3.65;
    const stage = (y, h, head, body, col) => { box(s, bx, y, bw, h, col, PALE); txt(s, [{ text: head, options: { bold: true, color: col, fontSize: 10, breakLine: true } }, { text: body, options: { fontSize: 8.5 } }], bx + 0.12, y + 0.07, bw - 0.24, h - 0.12); };
    stage(1.6, 1.0, "1  Dot sampler (fixed by physics)", "Per channel group: stratified log-polar × log-lag draws inside the cone, slot budget from the family. Point, profile and track channels contribute their actual observations. Fine imagery arrives as tokens from the local codec. Missing dots are simply absent.", INK);
    stage(2.7, 0.85, "2  Dot → token", "value embedding + Fourier features of (Δx, Δy in km, z in m, ℓ in days — all log) + channel embedding + source flag (observed swath / point / L4 / reanalysis / model / local token).", INK);
    stage(3.65, 1.0, "3  Encoder (Perceiver-style)", "K = 128–256 latents cross-attend to N ≈ 2,000–4,000 dots, then 4–8 self-attention blocks over the latents. Cost O(N·K), a few GFLOPs per anchor — below one ViT-Base image forward (≈ 17 GFLOPs).", EMB);
    stage(4.75, 0.6, "4  Pool → e_dyn (256-D, int8)", "the dynamic embedding: the state at the anchor, with its tendency and its context.", EMB);
    arrow(s, px + pw, py + ph / 2, bx, 2.1, MUTED, 1.25);
    [2.6, 3.55, 4.65].forEach(y => arrow(s, bx + bw / 2, y, bx + bw / 2, y + 0.1, MUTED, 1.25));
    // right: objective
    box(s, 8.9, 1.6, 3.8, 3.65, TIME, "FFF6EE");
    txt(s, [
      { text: "5  Objective: masked dots, including the future", options: { bold: true, color: TIME, fontSize: 10, breakLine: true, paraSpaceAfter: 4 } },
      { text: "A held-out dot is a query token (coordinates + channel, no value); a light decoder cross-attends from queries to the latents and predicts a distribution (Gaussian or bins; NLL / CRPS). Masking schemes mixed per batch:", options: { fontSize: 8.5, breakLine: true, paraSpaceAfter: 4 } },
      { text: "channel drop — remove a whole channel group (derived ones more often)", options: { bullet: true, fontSize: 8.5, breakLine: true } },
      { text: "lag-band drop — hide the recent past, predict it from the older past = forecasting inside pretraining", options: { bullet: true, fontSize: 8.5, breakLine: true } },
      { text: "future dots — queries at negative lags; the embedding becomes state + tendency", options: { bullet: true, fontSize: 8.5, breakLine: true } },
      { text: "sector drop — hide upstream; depth-band drop — predict interior from surface and back", options: { bullet: true, fontSize: 8.5, breakLine: true } },
      { text: "anchor reconstruction — always predict the anchor's own channels", options: { bullet: true, fontSize: 8.5, breakLine: true, paraSpaceAfter: 4 } },
      { text: "Never-measured vs masked survives on the target side: no loss where nothing was ever observed. Slow fields up-weighted (AIFS-ocean practice).", options: { fontSize: 8, italic: true, color: MUTED } },
    ], 9.05, 1.68, 3.55, 3.5);
    reading(s, "Why a set encoder and not a 4-D transformer over (x, y, z, t) grids: the input is ragged — a profile here, a swath there, a 10 m tile in the corner — and the cone is not a box. A set of dots handles both without padding, and retires the 'never-measured vs masked' distinction on the input side. Two products per anchor: an NRT embedding (observations + near-real-time streams) and a final one, exactly as ERA5T and ERA5; because the cone only looks backwards, an embedding is final once its sources have arrived and never needs recomputing.", 5.75);
    footer(s);
  }

  // ---------------------------------------------------------------- 42. where velocity comes from
  {
    const s = pres.addSlide(); s.background = { color: WHITE };
    title(s, "Where velocity comes from — two dots of the same field at two lags", "A photograph does not show speed; two photographs a known time apart do. Attention over dots with relative coordinates can implement feature tracking.");
    // Hovmöller diagram: x horizontal, time vertical (past at top → present at bottom); an advected feature is a tilted band
    const px = 0.6, py = 1.6, pw = 5.2, ph = 5.1;
    s.addShape(pres.shapes.RECTANGLE, { x: px, y: py, w: pw, h: ph, fill: { color: "FBFCFD" }, line: { color: GRIDLINE, width: 0.5 } });
    const ax = px + 0.55, ay = py + ph - 0.5, aw = pw - 0.85, ah = ph - 0.85;
    arrow(s, ax, ay, ax + aw, ay, INK, 1); arrow(s, ax, ay, ax, ay - ah, INK, 1);
    s.addText("position x (upstream ← → downstream)", { x: ax, y: ay + 0.05, w: aw, h: 0.25, fontFace: FONT_B, fontSize: 7.5, color: MUTED, align: "center", isTextBox: true, margin: 0 });
    s.addText("lag ℓ (past ↑, present at the bottom)", { x: ax - 0.2 - 1.1, y: ay - ah / 2 - 0.12, w: 2.2, h: 0.25, fontFace: FONT_B, fontSize: 7.5, color: MUTED, align: "center", rotate: 270, isTextBox: true, margin: 0 });
    // tilted band: the feature f(x - v t)
    const band = (x0f, wf, col, tr) => s.addShape(pres.shapes.CUSTOM_GEOMETRY, { x: ax, y: ay - ah, w: aw, h: ah, fill: { color: col, transparency: tr }, line: { width: 0 }, points: [{ x: x0f * aw, y: 0 }, { x: (x0f + wf) * aw, y: 0 }, { x: (x0f + wf + 0.5) * aw, y: ah }, { x: (x0f + 0.5) * aw, y: ah }, { x: x0f * aw, y: 0, close: true }] });
    band(0.08, 0.14, TEAL, 55); band(0.30, 0.10, TEAL, 70);
    // dots: anchor at present (bottom), upstream dot at lag ℓ (top-left)
    const dotAt = (fx, fy, col, big) => s.addShape(pres.shapes.OVAL, { x: ax + fx * aw - (big ? 0.08 : 0.06), y: ay - fy * ah - (big ? 0.08 : 0.06), w: big ? 0.16 : 0.12, h: big ? 0.16 : 0.12, fill: { color: col }, line: { color: WHITE, width: 0.75 } });
    const fyL = 0.7, fxU = 0.15 + 0.07, fxA = fxU + 0.5 * fyL;
    dotAt(fxA, 0.0, OUTPX, true); dotAt(fxU, fyL, OCEAN, true);
    // decoy dots at the same lag
    [0.05, 0.42, 0.62, 0.8].forEach(f => dotAt(f, fyL, GRIDLINE, false));
    [0.3, 0.55, 0.85].forEach(f => dotAt(f, 0.35, GRIDLINE, false));
    s.addShape(pres.shapes.LINE, { x: ax + fxU * aw, y: ay - fyL * ah, w: (fxA - fxU) * aw, h: fyL * ah, line: { color: OCEAN, width: 1.25, dashType: "dash" } });
    // Δx and ℓ brackets
    s.addShape(pres.shapes.LINE, { x: ax + fxU * aw, y: ay + 0.32, w: (fxA - fxU) * aw, h: 0, line: { color: INK, width: 0.75, beginArrowType: "triangle", endArrowType: "triangle" } });
    s.addText("Δx = v·ℓ", { x: ax + fxU * aw, y: ay + 0.33, w: (fxA - fxU) * aw, h: 0.22, fontFace: FONT_B, fontSize: 8, bold: true, color: INK, align: "center", isTextBox: true, margin: 0 });
    s.addShape(pres.shapes.LINE, { x: ax + fxA * aw + 0.55, y: ay - fyL * ah, w: 0, h: fyL * ah, line: { color: INK, width: 0.75, beginArrowType: "triangle", endArrowType: "triangle" } });
    s.addText("ℓ", { x: ax + fxA * aw + 0.6, y: ay - fyL * ah / 2 - 0.12, w: 0.4, h: 0.24, fontFace: FONT_B, fontSize: 9, bold: true, color: INK, isTextBox: true, margin: 0 });
    s.addText("anchor, now", { x: ax + fxA * aw - 0.1, y: ay - 0.34, w: 1.2, h: 0.22, fontFace: FONT_B, fontSize: 7.5, bold: true, color: INK, isTextBox: true, margin: 0 });
    s.addText("same water, ℓ ago", { x: ax + fxU * aw - 1.5, y: ay - fyL * ah - 0.38, w: 1.4, h: 0.22, fontFace: FONT_B, fontSize: 7.5, bold: true, color: OCEAN, align: "right", isTextBox: true, margin: 0 });
    s.addText("a feature carried at speed v traces a tilted band; grey dots are candidates the attention has to reject", { x: px + 0.1, y: py + 0.05, w: pw - 0.2, h: 0.35, fontFace: FONT_B, fontSize: 7.5, italic: true, color: MUTED, isTextBox: true, margin: 0 });
    // right: argument + three measures
    txt(s, [
      { text: "The argument", options: { bold: true, color: NAVY, fontSize: 12.5, breakLine: true, paraSpaceAfter: 4 } },
      { text: "If a field is carried along, f(x, t + ℓ) ≈ f(x − v·ℓ, t): the best predictor of the anchor's value is the dot at displacement −v·ℓ at lag ℓ. The cross-correlation between the anchor's neighbourhood now and the field ℓ ago peaks at Δx = v·ℓ. That is the maximum-cross-correlation method oceanographers use to track features in SST images (Emery et al. 1986). A query attending over dots with relative (Δx, ℓ) encodings can implement exactly this search — and the winning ratio Δx / ℓ is a velocity readout. Tendency (two lags at one place) and convergence (dots around the anchor) follow the same way. A per-pixel snapshot codec cannot do any of this; it has one lag.", options: { fontSize: 11, breakLine: true, paraSpaceAfter: 10 } },
      { text: "Making sure the capacity is used, not merely available", options: { bold: true, color: NAVY, fontSize: 12.5, breakLine: true, paraSpaceAfter: 4 } },
      { text: "Flow as maskable channels — drifter velocities, sea-ice drift, ASCAT / ERA5 wind, river discharge, and (flagged) OSCAR / GLORYS u, v are inputs, so the masked objective asks the embedding to produce velocity whenever the velocity channel is hidden.", options: { bullet: true, fontSize: 11, breakLine: true, paraSpaceAfter: 4 } },
      { text: "Multiple lags of the same field by construction — the B and C budgets always contain lags > 0, and the cone guarantees the upstream dots are in the set.", options: { bullet: true, fontSize: 11, breakLine: true, paraSpaceAfter: 4 } },
      { text: "Optional advected-sampling prior — draw a fraction of the dots along x − v̂·ℓ using a first-guess velocity (geostrophic u, v; ERA5 wind): PARADIS's semi-Lagrangian gather used as a sampler, not a layer.", options: { bullet: true, fontSize: 11, breakLine: true, paraSpaceAfter: 10 } },
      { text: "The test: a velocity probe", options: { bold: true, color: TIME, fontSize: 12.5, breakLine: true, paraSpaceAfter: 4 } },
      { text: "A ridge head from the frozen embedding to drifter velocity or GLORYS u, v at the anchor, with derived channels masked at test time. Prediction: the snapshot embedding scores near climatology (R² ≈ 0); the cone codec does not. This is the cheapest decisive experiment on the list.", options: { fontSize: 11 } },
    ], 6.1, 1.6, 6.6, 5.3);
    footer(s);
  }

  // ---------------------------------------------------------------- 43. zero-sum? pros and cons
  {
    const s = pres.addSlide(); s.background = { color: WHITE };
    title(s, "Is it zero-sum? Thin codec + thick stage 2 vs cone codec + thin stage 2", "A: embed one pixel-month, let a stage-2 stencil connect embeddings (today's Earth 2 and every GeoFM in the survey). B: embed the cone, keep stage 2 thin.");
    const g = (t) => c(t, { color: OBS, fontSize: 9 }); const b = (t) => c(t, { color: DERIV, fontSize: 9 }); const n = (t) => c(t, { fontSize: 9 });
    const rows = [
      [hdr(""), hdr("A — thin codec, thick stage 2"), hdr("B — thick (cone) codec, thin stage 2")],
      [nm("Cost per embedding"), g("Minimal: one pixel-month (27 tokens for 3 × 3 × 3 months)"), b("30–100 × more input per anchor (≈ 3,000 dots) — still ≈ one ViT-B image forward")],
      [nm("Reusability, caching"), g("Embedding = this place, this time; computed once, never changes"), n("Embedding = this place, this time and its past context; final once its sources have arrived (NRT and final versions)")],
      [nm("Interpretability"), g("Clear provenance: this pixel"), b("Provenance is a set of dots; attention maps recover it, with more work")],
      [nm("Dynamics: velocity, tendency, transport"), b("Not representable from a snapshot; stage 2 must rebuild it from compressed codes"), g("Representable in the codec (previous slide); supervised by flow channels")],
      [nm("Information bottleneck"), b("Codec compresses before it can know what stage 2 needs; sub-pixel phase and small gradients that carry velocity may be dropped as noise — what the code drops, stage 2 cannot recover"), g("Compression happens after the dynamics are visible; the objective (predict future and upstream dots) tells the codec what to keep")],
      [nm("Sparse channels: Argo, tracks, stations"), b("A lone profile with no context yields a near-empty code; gridding loses depth resolution"), g("Points enter natively as dots and are contextualised at encode time")],
      [nm("Cone geometry"), b("Re-learned per task and per channel by stage 2 (24 × 89 × channels tokens, each time)"), g("Fixed once by physics, shared by all tasks; 7 % of the cylinder's slots")],
      [nm("Sample efficiency of probes"), b("Attention head must learn the geometry from labels (AMOC: ~20 years of monthly labels)"), g("Geometry already inside; ridge / linear probes should suffice more often")],
      [nm("Static and mapping tasks"), g("Ideal: snapshot semantics (land cover, LCZ, canopy)"), n("Neutral to slightly worse — context can blur a static class; expose the local code alongside")],
      [nm("Forecasting in pretraining"), b("Not available: no time axis in the codec"), g("Native: future dots as targets")],
      [nm("Training complexity, shortcut risk"), g("Simple per-pixel pipeline; nothing to copy"), b("Heterogeneous dot sampler, bucketing, leakage control; derived channels can be copied — needs source flags and drop rates")],
      [nm("Comparability with published GeoFMs"), g("Direct"), b("Indirect: a different unit of embedding")],
    ];
    s.addTable(rows, { x: 0.6, y: 1.5, w: 12.1, colW: [2.3, 4.9, 4.9], fontFace: FONT_B, border: { type: "solid", color: GRIDLINE, pt: 0.5 }, rowH: [0.28, 0.38, 0.45, 0.36, 0.38, 0.6, 0.42, 0.42, 0.42, 0.42, 0.32, 0.48, 0.32], margin: 0.04, autoPage: false });
    reading(s, "Green = advantage, red = cost. A wins on cost, caching, provenance, simplicity; B on everything that involves time. Next slide: why this is not a symmetric trade.", 6.8);
    footer(s);
  }

  // ---------------------------------------------------------------- 44. not zero-sum: the asymmetry and the hybrid
  {
    const s = pres.addSlide(); s.background = { color: WHITE };
    title(s, "Not zero-sum — the asymmetry, and the split that follows from it", "The trade is about where the bottleneck sits relative to the dynamics, not about how much work there is in total");
    box(s, 0.6, 1.55, 5.9, 2.35, NAVY, "F4F6F9");
    txt(s, [
      { text: "Two facts break the symmetry", options: { bold: true, color: NAVY, fontSize: 11.5, breakLine: true, paraSpaceAfter: 4 } },
      { text: "1  B contains A. ", options: { bold: true, fontSize: 10.5 } }, { text: "A cone codec that ignores every dot but the anchor's own is a per-pixel codec. Whatever A can represent, B can; the reverse fails because of the bottleneck: information the per-pixel code discards before the dynamics are visible is gone for stage 2 (data-processing inequality). In capability the trade is one-sided; the price is compute and the cleanliness of the embedding's meaning.", options: { fontSize: 10.5, breakLine: true, paraSpaceAfter: 5 } },
      { text: "2  Compute is paid in different places. ", options: { bold: true, fontSize: 10.5 } }, { text: "A pays little per embedding but re-pays the context assembly in every stage-2 model, for every task, every time. B pays once per anchor and amortises across tasks — provided the cone is generic, which is exactly what a physics-set support promises: it depends on the channel, not on the task.", options: { fontSize: 10.5 } },
    ], 0.75, 1.63, 5.6, 2.25);
    box(s, 0.6, 4.05, 5.9, 2.85, TEAL, PALE);
    txt(s, [
      { text: "The principled split", options: { bold: true, color: TEAL, fontSize: 11.5, breakLine: true, paraSpaceAfter: 4 } },
      { text: "Into the embedding: ", options: { bold: true, fontSize: 10.5 } }, { text: "everything whose relevance is decided by physics — the cone (v, τ per channel), depth coupling, the local past. Identical for every task.", options: { fontSize: 10.5, breakLine: true, paraSpaceAfter: 4 } },
      { text: "Into stage 2: ", options: { bold: true, fontSize: 10.5 } }, { text: "everything whose relevance is decided by the task — which section to integrate over (AMOC), which basin to sum (carbon), which teleconnection matters (ENSO's Pacific state, 10,000 km and six months away, outside any single cone), which label.", options: { fontSize: 10.5, breakLine: true, paraSpaceAfter: 4 } },
      { text: "Where A stays right: ", options: { bold: true, fontSize: 10.5 } }, { text: "purely static targets (land cover, soil, canopy height, LCZ), pixel-level anomaly detection, and very-high-volume or on-board inference (Prithvi-EO's in-orbit deployment). Keep the local code for these.", options: { fontSize: 10.5, breakLine: true, paraSpaceAfter: 4 } },
      { text: "What existing evidence says: nothing yet. The attribution matrix's 3 × 3 codec-vs-raw contrast (0.672 vs 0.659) is a single-seed probe number inside the probe noise band, so it is consistent with parity; and every rolled-skill number from before 2 Sep 2026 was withdrawn with the paper reset (heads trained under the endpoint pool had seen the held-out years). The v8 paper's null ladder — a Linear Inverse Model in pixel space against one in embedding space — is the first measurement of the bottleneck row, and Phase 0 is built to run after it.", options: { fontSize: 9.5, italic: true, color: MUTED } },
    ], 0.75, 4.13, 5.6, 2.75);
    // right: hierarchy diagram
    const hx = 6.85, hw = 5.85;
    s.addText("The hybrid: split by scale, not by fiat", { x: hx, y: 1.55, w: hw, h: 0.3, fontFace: FONT_B, fontSize: 11.5, bold: true, color: NAVY, isTextBox: true, margin: 0 });
    const tier = (y, h, col, head, body, tag) => {
      box(s, hx, y, hw, h, col, WHITE);
      txt(s, [{ text: head, options: { bold: true, color: col, fontSize: 10, breakLine: true } }, { text: body, options: { fontSize: 8.5 } }], hx + 0.15, y + 0.07, hw - 1.75, h - 0.12);
      s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x: hx + hw - 1.5, y: y + 0.12, w: 1.35, h: h - 0.24, fill: { color: col, transparency: 85 }, line: { width: 0 }, rectRadius: 0.06 });
      s.addText(tag, { x: hx + hw - 1.5, y: y + 0.12, w: 1.35, h: h - 0.24, fontFace: FONT_B, fontSize: 8, bold: true, color: col, align: "center", valign: "middle", isTextBox: true, margin: 0.03 });
    };
    tier(1.95, 0.95, STAT, "Raw observations and admitted derived products", "the thirteen rungs; point observations stay points; every dot carries a source flag", "ladder\nrungs 0–13");
    tier(3.05, 1.05, LAND, "Local codec — per tile-date, cached, snapshot semantics", "the existing per-pixel / 3 × 3 MAE codec generalised to a tile (64 × 64 px at 10 m, STP-style pyramid or FlexiViT tokens); masked-pixel objective; a few 64-D tokens per tile-date reused by every anchor whose cone contains the tile", "e_loc 64-D\nmapping tasks");
    tier(4.25, 1.15, EMB, "Cone codec — per anchor, dynamic semantics", "reads local tokens + coarse dots + native point observations inside each channel's cone; masked-dot objective with future dots; the momentum features live here", "e_dyn 256-D\nprediction tasks");
    tier(5.55, 1.3, TIME, "Stage 2 — thin", "teleconnections beyond the cone (attention over distant anchors' e_dyn), aggregation along sections and over basins, task heads; the probe ladder stays (ridge → MLP → attention) with the expectation that more tasks stop at ridge", "AMOC, carbon,\nwarming, biosphere");
    [2.9, 4.1, 5.4].forEach(y => arrow(s, hx + 2.0, y, hx + 2.0, y + 0.15, MUTED, 1.5));
    footer(s);
  }

  // ---------------------------------------------------------------- 44b. two stencils, one cone (E-069 built and run; re-cut for v2)
  {
    const s = pres.addSlide(); s.background = { color: WHITE };
    title(s, "Two stencils, one cone — built, run, and re-cut for v2",
      "E-069 ran five seeds (3 + 2 objectives) on the 0.3 m/s geometry; H1 is refuted, history buys persistence, not velocity. Under cone v2 the split between the stencils moves from distance to time — E-071 §4.5.");
    const RED = "A23B3B";
    // ---------- left: the (lag, distance) map, on log axes, with BOTH geometries ----------
    const px = 0.6, py = 1.6, pw = 5.6, ph = 3.95;
    s.addShape(pres.shapes.RECTANGLE, { x: px, y: py, w: pw, h: ph, fill: { color: "FBFCFD" }, line: { color: GRIDLINE, width: 0.5 } });
    const ax = px + 0.84, ay = py + ph - 0.46, aw = pw - 1.06, ah = ph - 1.24;
    const LAGMIN = 1, LAGMAX = 144, RMIN = 28, RMAX = 20015;
    const fx = k => Math.log(Math.max(k, LAGMIN) / LAGMIN) / Math.log(LAGMAX / LAGMIN);
    const fy = r => Math.log(Math.max(r, RMIN) / RMIN) / Math.log(RMAX / RMIN);
    const KM069 = 0.3 * 86400 * 5 / 1000, KM2 = 5.4 * 86400 * 5 / 1000;   // km per (1 + lag): 129.6 and 2,332.8
    const in069 = k => KM069 * (1 + k);
    const out069 = k => Math.min(4444, Math.max(111, KM069 * (1 + k)));
    const in2 = k => Math.min(20015, KM2 * (1 + k));
    const out2 = k => Math.min(20015, Math.max(111, KM2 * (1 + k)));
    const poly = (pts, col, tr, dash) => s.addShape(pres.shapes.CUSTOM_GEOMETRY, { x: ax, y: ay - ah, w: aw, h: ah, fill: { color: col, transparency: tr }, line: { color: col, width: dash ? 1 : 0.75, dashType: dash || "solid" }, points: pts.map(p => ({ x: p[0] * aw, y: (1 - p[1]) * ah })).concat([{ x: pts[0][0] * aw, y: (1 - pts[0][1]) * ah, close: true }]) });
    const stepPoly = f => { const a = [[fx(1), 0]]; for (let k = 1; k <= 6; k++) { a.push([fx(k), fy(f(k))]); a.push([fx(k + 1) - 0.002, fy(f(k))]); } a.push([fx(7) - 0.002, 0]); return a; };
    const outPoly = f => { const a = []; for (let k = 7; k <= 143; k += 4) a.push([fx(k), fy(f(k))]); a.push([fx(143), fy(f(143))]); a.push([fx(143), fy(111)]); a.push([fx(7), fy(111)]); return a; };
    // cone v2 first (the larger, lighter pair), then E-069 as built on top of it
    poly(outPoly(out2), OBS, 88, "dash");
    poly(stepPoly(in2), OBS, 78, "dash");
    poly(outPoly(out069), TIME, 55);
    poly(stepPoly(in069), EMB, 55);
    // the two horizontal marks: the old outer cap, and the antipode
    [[4444, "4,444 km", RED, "E-069's outer cap", 0.45, 0.02], [20015, "20,015 km", RED, "antipode", 0.74, -0.17]].forEach(([r, lab, col, tag, fxt, dy]) => {
      s.addShape(pres.shapes.LINE, { x: ax, y: ay - fy(r) * ah, w: aw, h: 0, line: { color: col, width: 0.9, dashType: "sysDot" } });
      txt(s, [{ text: lab, options: { bold: true, color: col, fontSize: 6.5 } }], px + 0.02, ay - fy(r) * ah - 0.09, 0.80, 0.18, { align: "right" });
      txt(s, [{ text: tag, options: { italic: true, bold: true, color: col, fontSize: 6 } }], ax + fxt * aw, ay - fy(r) * ah + dy, 1.10, 0.15);
    });
    // anchor column (both stages, both geometries) — a bar along the bottom
    s.addShape(pres.shapes.RECTANGLE, { x: ax, y: ay - 0.05, w: aw, h: 0.05, fill: { color: OUTPX }, line: { width: 0 } });
    // axes + ticks
    arrow(s, ax, ay, ax + aw, ay, INK, 1); arrow(s, ax, ay, ax, ay - ah, INK, 1);
    [[1, "1"], [6, "6"], [7, "7"], [36, "36"], [144, "144"]].forEach(([k, t]) => s.addText(t, { x: ax + fx(k) * aw - 0.2, y: ay + 0.02, w: 0.4, h: 0.2, fontFace: FONT_B, fontSize: 7, color: MUTED, align: "center", isTextBox: true, margin: 0 }));
    s.addText("lag k (pentads, log) — 1 pentad = 5 days; 144 = 2 years. Lag 0 is the 3 × 3 patch.", { x: ax, y: ay + 0.21, w: aw, h: 0.2, fontFace: FONT_B, fontSize: 6.5, color: MUTED, align: "center", isTextBox: true, margin: 0 });
    [[100, "100 km"], [907, "907 km"]].forEach(([r, t]) => txt(s, [{ text: t, options: { fontSize: 6.5, color: MUTED } }], px + 0.02, ay - fy(r) * ah - 0.09, 0.80, 0.18, { align: "right" }));
    // the key, at the top of the panel
    txt(s, [
      { text: "E-069 as built  ", options: { bold: true, color: EMB, fontSize: 6.8 } },
      { text: "inner (codec): raw channels, lags 1–6, 130 → 907 km · ", options: { color: INK, fontSize: 6.8 } },
      { text: "outer (stage 2): ", options: { bold: true, color: TIME, fontSize: 6.8 } },
      { text: "embeddings, lags 7–143, 111 → 4,444 km", options: { color: INK, fontSize: 6.8, breakLine: true } },
      { text: "cone v2 (E-071 §4.5)  ", options: { bold: true, color: OBS, fontSize: 6.8 } },
      { text: "inner: 2,333 km already at lag 0, 16,330 km at lag 6 · outer: 18,664 km at lag 7, at the antipode from lag 8 — a global band", options: { color: INK, fontSize: 6.8, breakLine: true } },
      { text: "anchor column  ", options: { bold: true, color: "9A7B1E", fontSize: 6.8 } },
      { text: "the only overlap, in both geometries", options: { color: INK, fontSize: 6.8 } },
    ], px + 0.10, py + 0.05, pw - 0.20, 0.62);
    // ---------- the legend prose, under the panel ----------
    txt(s, [
      { text: "E-069:  ", options: { bold: true, color: EMB, fontSize: 7.5 } },
      { text: "the codec owns the near field, stage 2 the far field.    ", options: { fontSize: 7.5, color: INK } },
      { text: "v2:  ", options: { bold: true, color: OBS, fontSize: 7.5 } },
      { text: "the codec owns the last 30 days planet-wide; stage 2 owns 35 days – 2 years planet-wide. The overlap is still exactly the anchor column, and the empty annulus for k ≤ 6 still holds by construction.", options: { fontSize: 7.5, color: INK } },
    ], px, py + ph + 0.08, pw, 0.62);
    // ---------- right: what five seeds said ----------
    box(s, 6.45, 1.6, 6.25, 4.65, NAVY, "F4F6F9");
    txt(s, [
      { text: "What five seeds said (EXPERIMENTS.md § E-069, 3–4 Sep)", options: { bold: true, color: NAVY, fontSize: 11, breakLine: true, paraSpaceAfter: 6 } },
      { text: "1 · H1 refuted.  ", options: { bold: true, color: RED, fontSize: 8.6 } },
      { text: "The velocity probe — a ridge from the 32-number embedding to the anchor's own current, with the current channels hidden from the encoder — reads AT OR BELOW the raw-patch bar for the cone on both components at every seed, while the snapshot twin sits at the bar. History buys no velocity beyond geostrophy-from-the-sea-surface-height-patch at 7.05M parameters and 20k steps, under either objective. E-069 masking: cone cur_u 0.60 / 0.10 / 0.62 against a twin at ≈ 0.70; cur_v ≈ 0 at every seed. E-069b lag0-drop objective: cone 0.63 / 0.66 and 0.58 / 0.59 against bars 0.64 / 0.71 and 0.67 / 0.70. A CPU synthetic-advection test finds no displacement primitive at the scales tried — a reading, not a closure: the field's certified ceiling was 0.16.", options: { fontSize: 8.6, color: INK, breakLine: true, paraSpaceAfter: 4 } },
      { text: "2 · What history DOES buy.  ", options: { bold: true, color: OBS, fontSize: 8.6 } },
      { text: "A hidden sea-surface-height channel is reconstructed at ≈ 0.36 of its bar against the twin's 0.75–0.90; hidden sea surface temperature 0.53–0.66 against 0.99–1.27; forecasts at +1 and +2 pentads 7–8 % better. Persistence and one-step forecast, not velocity.", options: { fontSize: 8.6, color: INK, breakLine: true, paraSpaceAfter: 4 } },
      { text: "3 · Consequence for v2.  ", options: { bold: true, color: TEAL, fontSize: 8.6 } },
      { text: "Widening the reach 18× does not touch the mechanism H1 tested. v2's case rests on rolled skill (H2) from far-field context, and on the persistence signal being worth more when it is global, harmonic-anomaly and profile-aware. The velocity lever is capacity, the objective, or an explicit displacement primitive — not geometry.", options: { fontSize: 8.6, color: INK, breakLine: true, paraSpaceAfter: 4 } },
      { text: "4 · Two open decisions, deferred.  ", options: { bold: true, color: MUTED, fontSize: 8.6 } },
      { text: "The outer ring's dot count per lag, on the same per-octave rule as the inner cone; and K = 144 pentads (two years) against a log-spaced eight-year outer window for the deep western boundary current's advective path.", options: { fontSize: 8.6, color: INK } },
    ], 6.60, 1.70, 5.95, 4.45);
    reading(s, "Reading. The original split gave the codec the near field and stage 2 the far field. With the reach sized from the fastest mechanism the codec sees the planet within a week, and the honest description of the two stencils becomes thirty days of raw channels against two years of embeddings — both global. The identities the coverage test asserts (union = the whole cone, overlap = the anchor column) survive unchanged; what changed is what they mean.", 6.42);
    footer(s);
  }

  // ---------------------------------------------------------------- 44c. what feeds the cone codec today (derivation graph + ranked additions)
  {
    const s = pres.addSlide(); s.background = { color: WHITE };
    title(s, "What feeds the cone codec today — four products, 42 channels, one axis",
      "family 4 recipe r3 — the tensor the codec eats: North Atlantic 100° W–20° E, 0–70° N · 0.25° · 5-day bins · 1982-01-01 → 2024-12-31 · 84,405 ocean cells · built once and sha-pinned (run #535)");
    // ---------- left 60 %: the derivation graph, four rows ----------
    const CX = [0.6, 2.07, 3.54, 5.01, 6.48], CW = 1.35;   // five columns: 4 products + the labels
    const rowLabel = (t, y, note) => txt(s, [
      { text: t, options: { bold: true, color: NAVY, fontSize: 8, charSpacing: 1 } },
      { text: "   " + note, options: { color: MUTED, fontSize: 7.5, italic: true } },
    ], 0.6, y, 7.25, 0.13);
    const cell = (i, y, h, col, fill, runs) => { box(s, CX[i], y, CW, h, col, fill); txt(s, runs, CX[i] + 0.07, y + 0.05, CW - 0.14, h - 0.1); };
    const down = (i, y1, y2) => arrow(s, CX[i] + CW / 2, y1, CX[i] + CW / 2, y2, GRIDLINE, 1);

    // row 0 — observations
    rowLabel("0 · OBSERVATIONS", 1.5, "what an instrument actually measured");
    [["satellite altimetry · satellite SST (AVHRR) · Argo floats · ships, buoys, drifters"],
     ["satellite SST (AVHRR) · ships, buoys, drifters"],
     ["Argo floats only"],
     ["radiosondes, aircraft, ships, satellite radiances"],
     ["moorings at 26.5° N · the Florida Straits cable"]].forEach((o, i) =>
      cell(i, 1.72, 0.56, GRIDLINE, "F7F9FB", [{ text: o[0], options: { fontSize: 7, color: MUTED } }]));
    // row 1 — the products we download
    rowLabel("1 · PRODUCTS WE DOWNLOAD", 2.30, "four inputs + two labels; green = observation-led, red = model-derived");
    CX.forEach((_, i) => down(i, 2.41, 2.53));
    const prod = [
      [DERIV, "GLORYS12", "derived", "Copernicus Marine ocean reanalysis — a model nudged towards observations. 1/12°, daily, 1993→. Assimilates altimetry, SST and Argo on a 7-day cycle."],
      [OBS, "NOAA OISST v2.1", "analysed", "0.25°, daily, 1981-09→. Optimum interpolation of AVHRR radiances and in-situ measurements — an analysis of observations, no ocean model."],
      [OBS, "Roemmich–Gilson Argo", "analysed", "Scripps climatology, 1°, monthly, 2004→. Argo floats mapped onto a grid; nothing else enters it."],
      [DERIV, "NCEP/NCAR R1 momentum flux", "derived", "Atmospheric reanalysis 1, ~1.9°, daily, 1948→. The surface wind stress is model-diagnosed. Archive FROZEN since Mar 2026."],
      [TIME, "RAPID 26.5° N + Florida Current", "labels, never input", "Overturning transport 12-hourly 2004→; cable transport daily 1982→. Stored beside the tensor as truth; the trainer never reads them."],
    ];
    prod.forEach(([col, name, kind, body], i) => {
      box(s, CX[i], 2.54, CW, 1.06, col, i === 4 ? "FFF6EE" : PALE);
      txt(s, [
        { text: name, options: { bold: true, color: col, fontSize: 7.5, breakLine: true } },
        { text: kind.toUpperCase(), options: { bold: true, color: col, fontSize: 7, charSpacing: 0.5, breakLine: true, paraSpaceAfter: 2 } },
        { text: body, options: { fontSize: 7, color: INK } },
      ], CX[i] + 0.07, 2.58, CW - 0.14, 1.0);
    });
    // row 2 — the 42 channels
    rowLabel("2 · CHANNELS IN THE TENSOR", 3.61, "42 of them, plus the two labels; live-bin caveats in italics");
    CX.forEach((_, i) => down(i, 3.74, 3.87));
    const chan = [
      [DERIV, "cur_speed · cur_u · cur_v · log_mld · ssh", "5 channels, 1993+ only. cur_speed = hypot(cur_u, cur_v) by construction, so it is not independent of the two components."],
      [OBS, "sst", "1 channel, live on the whole axis (1982→). The only channel with no hole in it."],
      [OBS, "rg_t × 16 · rg_s × 16 (10–1900 dbar)", "32 channels — 80 % of the bytes. 2004+, and ONE live 5-day bin in six; `missing` tokens in the other five."],
      [DERIV, "tau_x · tau_y · tau_x_std · tau_y_std", "4 channels, whole axis. The two σ channels are computed from the daily fields, never from the pentad mean."],
      [TIME, "truth_rapid · truth_fc", "Two label series stored in the same file. Read by the probes, never by the encoder."],
    ];
    chan.forEach(([col, head, body], i) => {
      box(s, CX[i], 3.88, CW, 0.94, col, i === 4 ? "FFF6EE" : "FBFCFD");
      txt(s, [
        { text: head, options: { bold: true, color: col, fontSize: 7, breakLine: true, paraSpaceAfter: 2 } },
        { text: body, options: { fontSize: 7, color: INK, italic: true } },
      ], CX[i] + 0.07, 3.92, CW - 0.14, 0.88);
    });
    // row 3 — derived inside training
    rowLabel("3 · DERIVED INSIDE TRAINING", 4.83, "nothing here is downloaded — it is computed from the tensor itself");
    [0, 1, 2, 3].forEach(i => down(i, 4.96, 5.09));
    const DW = 1.72, DX = [0.6, 2.44, 4.28, 6.12];
    [[TEAL, "Anomaly, then z-score", "Each channel minus a per-calendar-month average built from TRAINING YEARS ONLY, then divided by its own spread. No outside climatology is used."],
     [TEAL, "Mask · holdout · certificate", "The ocean mask is where channel 0 is finite (84,405 cells). Calendar years 2009, 2017 and 2023 are held out, certified 0 violations in 4,096 anchors."],
     [TEAL, "Probe target · snapshot twin", "The velocity probe's target is the anchor's own cur_u, cur_v, hidden from the encoder. A second identical codec with no lags (L_in = 0) is the control."],
     [MUTED, "Pinned for stage 2, unused yet", "The run-415 control embedding on recipe r2, the linear-inverse-model null (8 scoreable channels), and the AMOC evaluation mask."]]
      .forEach(([col, head, body], i) => {
        box(s, DX[i], 5.10, DW, 0.82, col, col === MUTED ? "F4F6F9" : "F2F8FA");
        txt(s, [
          { text: head, options: { bold: true, color: col, fontSize: 7.5, breakLine: true, paraSpaceAfter: 2 } },
          { text: body, options: { fontSize: 7, color: INK } },
        ], DX[i] + 0.07, 5.14, DW - 0.14, 0.76);
      });
    // two annotations under the graph
    txt(s, [
      { text: "▲ ", options: { color: DERIV, fontSize: 8, bold: true } },
      { text: "The app's 1991–2020 sea-surface-temperature normal is deliberately NOT the anomaly baseline: it averages over the held-out years, so it would leak the test period into training.", options: { fontSize: 7.5, color: INK, breakLine: true, paraSpaceAfter: 2 } },
      { text: "▲ ", options: { color: DERIV, fontSize: 8, bold: true } },
      { text: "Derived products peek forward by half their assimilation window — the observations a reanalysis may look at around each day (GLORYS: 7 days). 5 of the 42 channels; they need a time shift.", options: { fontSize: 7.5, color: INK } },
    ], 0.6, 5.99, 7.25, 0.5);

    // ---------- right 40 %: what is missing ----------
    box(s, 7.95, 1.5, 4.75, 4.92, NAVY, "F4F6F9");
    const add = [
      ["Buoyancy forcing — ERA5", "net short- and long-wave radiation, sensible and latent heat, evaporation − precipitation, wind-stress curl. The only forcing entirely absent today, and it also replaces the frozen NCEP momentum flux."],
      ["Ocean interior at 0.25°, in 3-D — GREP", "temperature, salinity and both velocity components at 8 depths, 1993→ (+20 GB). Velocity at depth, and six times as many live subsurface bins. The data ladder's #1."],
      ["Observed sea level — DUACS altimetry", "sea-level anomaly, absolute dynamic topography and geostrophic velocities, 0.125°, daily, 1993→: the observation GLORYS is fitted to, as its own independent channel."],
      ["Statics", "bathymetry and its gradient, the Coriolis parameter f and its gradient β, distance to coast and to the 1000 m isobath, mean dynamic topography. ~3 MB; the ladder's #2."],
      ["Sea-ice concentration — OSTIA / OSI SAF", "the northern boundary condition of the window. Today the model sees ice-covered water as ordinary water."],
      ["Drifters — Global Drifter Program", "15 m velocity and SST as native dots: the observed velocity the H1 probe (does the embedding know the current?) should really be scored on."],
      ["Biosphere", "ocean-colour chlorophyll (OC-CCI 4 km 1997→, PACE 2024→), BGC-Argo oxygen / nitrate / chlorophyll profiles as dots, SOCAT surface CO₂. The ladder rejected chlorophyll as an INPUT for the overturning head — “no mechanistic path at these timescales” — so the biosphere enters first as what the embedding must PREDICT, and as a held-out sphere."],
      ["Sea-surface salinity — SMOS / SMAP", "2010→, short, but it is the freshwater lid. And RAPID's depth-resolved moc_vertical and boundary T/S files are already downloaded and unused."],
    ];
    txt(s, [
      { text: "Missing from “the ocean and its biosphere” — additions, ranked", options: { bold: true, color: NAVY, fontSize: 11, breakLine: true, paraSpaceAfter: 5 } },
      ...add.flatMap(([h, body], i) => [
        { text: `${i + 1}  ${h}  `, options: { bold: true, color: TEAL, fontSize: 8.5 } },
        { text: body, options: { fontSize: 8.5, color: INK, breakLine: true, paraSpaceAfter: 4 } },
      ]),
      { text: "Ranked by information per byte and by what the model cannot derive from what it already holds.", options: { fontSize: 8, italic: true, color: MUTED } },
    ], 8.1, 1.58, 4.45, 4.78);
    reading(s, "Reading. Today's codec sees the ocean's momentum forcing (wind stress) and its density structure in one column, and nothing else: no heat or freshwater forcing, no observed velocity, no boundaries — no ice, no bathymetry — and nothing living. Two thirds of the channel count is a single monthly Argo product that is live in one 5-day bin out of six. The additions on the right are ordered by information per byte and by what the model cannot derive from what it already has; the first three are also the three the E-069 plan (the cone codec being trained now) would use unchanged.", 6.6);
    footer(s);
  }

  // ---------------------------------------------------------------- 44d. boundaries — every edge the sampler meets
  {
    const s = pres.addSlide(); s.background = { color: WHITE };
    title(s, "Boundaries — four edges, four rules, and two of them now change",
      "Findings of 3 Sep; decisions of 4 Sep (Chris): the dataset is global, and the anomaly baseline is a day-of-year harmonic fit — E-071.");
    const RED = "A23B3B";
    const PX = [0.6, 6.75], PW = 5.95;
    const R1Y = 1.46, R1H = 2.44, R2Y = 3.96, R2H = 2.46;
    const head = (x, y, t, col) => txt(s, [{ text: t, options: { bold: true, color: col, fontSize: 9.5 } }], x + 0.13, y + 0.06, PW - 0.26, 0.24);
    const rule = (t) => ({ text: t + "  ", options: { bold: true, color: TEAL, fontSize: 7, charSpacing: 0.3 } });
    const body = (t, o) => ({ text: t, options: Object.assign({ fontSize: 7, color: INK }, o || {}) });
    const dec = (t) => ({ text: t + "  ", options: { bold: true, color: OBS, fontSize: 7, charSpacing: 0.4 } });
    const keep = (x, y) => txt(s, [{ text: "UNCHANGED — these rules were right", options: { bold: true, color: MUTED, fontSize: 6.5, charSpacing: 0.4 } }], x + 0.13, y, PW - 0.26, 0.16);

    // ---------- A · the window edge ----------
    box(s, PX[0], R1Y, PW, R1H, SPACE, "F7FAFC");
    head(PX[0], R1Y, "A · SPACE — the window edge (100° W – 20° E, 0–70° N)", SPACE);
    txt(s, [
      rule("FINDING, 3 SEP"),
      body("a dot outside the rectangle is INVALID, never wrapped: the tensor is a basin, so a wrap “would put the Iberian shelf one cell west of Florida” (ml/cone_sampler.py:19–23). The index is clipped, the value discarded, and valid = False becomes the ATTENTION MASK (ConeSampler.sample :293; the lag-0 3×3 patch likewise, _patch :263). Anchors are drawn uniformly over ocean cells with NO edge margin (ml/train_cone.py:358), so an edge anchor sees a truncated, one-sided cone.", { breakLine: true, paraSpaceAfter: 3 }),
      { text: "The asymmetry: ", options: { bold: true, color: RED, fontSize: 7 } },
      body("nothing tells the model that the missing dots are missing because of OUR rectangle rather than because of the ocean."),
    ], PX[0] + 0.13, R1Y + 0.32, PW - 0.26, 0.80);
    // sketch: the window rectangle with a cone truncated at the eastern edge
    {
      const gx = PX[0] + 0.14, gy = 2.62, gw = 2.20, gh = 0.44;
      s.addShape(pres.shapes.RECTANGLE, { x: gx, y: gy, w: gw, h: gh, fill: { color: "FFFFFF" }, line: { color: SPACE, width: 1 } });
      const ax = gx + gw - 0.14, ay = gy + gh / 2;
      [[0.13, 8], [0.26, 10]].forEach(([r, n]) => {
        for (let i = 0; i < n; i++) {
          const th = 2 * Math.PI * i / n;
          const cx = ax + r * Math.cos(th), cy = ay + r * 0.62 * Math.sin(th);
          const out = cx > gx + gw - 0.02;
          s.addShape(pres.shapes.OVAL, { x: cx - 0.028, y: cy - 0.028, w: 0.056, h: 0.056,
            fill: out ? { type: "none" } : { color: SPACE }, line: { color: out ? RED : SPACE, width: out ? 0.9 : 0 } });
        }
      });
      s.addShape(pres.shapes.OVAL, { x: ax - 0.035, y: ay - 0.035, w: 0.07, h: 0.07, fill: { color: OUTPX }, line: { color: NAVY, width: 0.75 } });
      txt(s, [{ text: "the tensor window — the dots do not wrap", options: { fontSize: 6.5, italic: true, color: MUTED } }], gx, gy + gh + 0.02, gw + 0.3, 0.16);
      txt(s, [
        { text: "Filled ", options: { bold: true, color: SPACE, fontSize: 7 } },
        { text: "= read. ", options: { fontSize: 7, color: INK } },
        { text: "Hollow ", options: { bold: true, color: RED, fontSize: 7 } },
        { text: "= off the rectangle: clipped, discarded, masked out of attention.", options: { fontSize: 7, color: INK, breakLine: true, paraSpaceAfter: 2 } },
        { text: "COST  ", options: { bold: true, color: TEAL, fontSize: 7 } },
        { text: "over all 84,405 ocean anchors with the real 706-dot stencil: 22.0 % of anchors have at least one off-grid dot; 2.68 % of all dots are off-grid.", options: { fontSize: 7, color: INK } },
      ], gx + gw + 0.16, gy - 0.02, PW - gw - 0.46, 0.66);
      txt(s, [
        dec("DECISION → GLOBAL (E-071 §1)"),
        body("family 7 — the first tensor covering the whole globe rather than the North Atlantic window — runs to the POLES: 721 × 1,440 cells, −90…90° N; longitude WRAPS (xx = (x + dx) mod W), and every dot is placed on the sphere as a destination point, so a pole crossing falls out of the formula and nothing is ever clipped. The 2.68 % off-grid dots become real dots and the 22 % truncated anchors see a full cone. ASPECT = 0.71 was measured on the North Atlantic and is re-measured per latitude band."),
      ], PX[0] + 0.13, 3.28, PW - 0.26, 0.58);
    }

    // ---------- B · land ----------
    box(s, PX[1], R1Y, PW, R1H, "4E8A3E", "F7FBF6");
    head(PX[1], R1Y, "B · SPACE — land", "4E8A3E");
    keep(PX[1], R1Y + 0.30);
    txt(s, [
      rule("RULE"),
      body("a dot on land is a REAL token whose value was never observed: obs = False → the codec’s ", { }),
      { text: "miss", options: { bold: true, color: "4E8A3E", fontSize: 7 } },
      body(" token — the same miss_tok PixelMAE (the pixel-level masked auto-encoder this codec inherits) uses — NOT an invalid/masked token.", { breakLine: true, paraSpaceAfter: 3 }),
      body("The distinction is the one the whole architecture rests on: “the data never observed this” is information; “this dot does not exist” is not. One is a token the model can learn from, the other is a hole in the attention matrix.", { breakLine: true, paraSpaceAfter: 3 }),
      rule("COST"),
      body("8.17 % of all dots land on land — on-grid, unobserved. Combined with the edge, ", {}),
      { text: "10.85 % of the 706 dots at a typical anchor carry no value.", options: { bold: true, color: RED, fontSize: 7, breakLine: true, paraSpaceAfter: 3 } },
      body("Land dots are still spent from the budget: they occupy their slot and are attended to, so a coastal anchor pays for its geography twice.", { italic: true, color: MUTED }),
    ], PX[1] + 0.13, R1Y + 0.50, PW - 0.26, 1.10);
    {
      const bx0 = PX[1] + 0.14, bw = PW - 0.28, by = 3.10;
      let cx = bx0;
      [[89.15, TEAL], [8.17, "4E8A3E"], [2.68, RED]].forEach(([p, col]) => {
        const w = bw * p / 100;
        s.addShape(pres.shapes.RECTANGLE, { x: cx, y: by, w, h: 0.18, fill: { color: col, transparency: 15 }, line: { color: WHITE, width: 0.6 } });
        cx += w;
      });
      txt(s, [{ text: "the 706 dots at one anchor", options: { fontSize: 6.5, italic: true, color: MUTED } }], bx0, by + 0.19, bw, 0.16);
      txt(s, [
        { text: "■ ", options: { color: TEAL, fontSize: 8 } }, { text: "89.15 % carry a value    ", options: { fontSize: 7, color: INK } },
        { text: "■ ", options: { color: "4E8A3E", fontSize: 8 } }, { text: "8.17 % land → miss token    ", options: { fontSize: 7, color: INK } },
        { text: "■ ", options: { color: RED, fontSize: 8 } }, { text: "2.68 % off-grid → invalid", options: { fontSize: 7, color: INK } },
      ], bx0, by + 0.36, bw, 0.20);
    }

    // ---------- C · the archive ends, and the holdout blocks ----------
    box(s, PX[0], R2Y, PW, R2H, TIME, "FFF9F3");
    head(PX[0], R2Y, "C · TIME — the archive ends, and the holdout blocks", TIME);
    keep(PX[0], R2Y + 0.30);
    txt(s, [
      rule("RULE 1 — the ends of the axis"),
      body("an anchor whose cone runs off either end of the tensor — it needs bins t−6 … t+2 — is INADMISSIBLE, not silently short (ConeSampler.admissible, :314). Nothing is padded and nothing is trained on half a cone.", { breakLine: true, paraSpaceAfter: 3 }),
      rule("RULE 2 — the holdout"),
      body("--holdout-scope window (the only scope implemented, ml/train_cone.py:116) says every bin the cone touches — six pentads back and both future targets — must be a training bin. A held-out year therefore casts a ", {}),
      { text: "shadow of eight pentads", options: { bold: true, color: TIME, fontSize: 7 } },
      body(" around itself — six back and two forward, the exact span of the cone — and no training anchor can peek into it by any path. A brute-force certify() (:337), written deliberately as a separate loop rather than as a reuse of the admissibility test, counts violations; run #537 (the E-069 cone-codec build) measured 0 violations in 4,096 anchors.", { breakLine: true, paraSpaceAfter: 3 }),
      rule("COST OF THE SHADOW"),
      body("development split (calendar years 2009, 2017, 2023 held out): 2,923 training bins → 2,891 admissible anchor bins, 32 bins lost (1.1 %). Frozen protocol (2008–09, 2016–17, 2021–24 — train ≤ 2020, test the terminal years): 2,557 → 2,533, 24 bins lost (0.9 %). Cheap, and it is what makes the holdout real.", { breakLine: true, paraSpaceAfter: 3 }),
      { text: "Held-out EVALUATION anchors are different on purpose: ", options: { bold: true, color: RED, fontSize: 7 } },
      body("their dots MAY come from held-out bins — otherwise the held-out set would be a second training pool, not a measurement."),
    ], PX[0] + 0.13, R2Y + 0.50, PW - 0.26, 1.90);

    // ---------- D · the calendar month ----------
    box(s, PX[1], R2Y, PW, R2H, RED, "FDF6F6");
    head(PX[1], R2Y, "D · TIME — the calendar month: the edge we introduced ourselves", RED);
    txt(s, [
      { text: "(i) The anomaly baseline.  ", options: { bold: true, color: TEAL, fontSize: 7 } },
      body("Every channel is a departure from a per-calendar-month climatology built on TRAINING YEARS ONLY (ml/trainprobe.py::anomaly_transform :140), and a pentad’s month label is the month of its FIRST day (ml/build_family4.py:960). So ", {}),
      { text: "411 of the 3,142 bins (13.1 %) straddle a month boundary", options: { bold: true, color: RED, fontSize: 7 } },
      body(", and 106 of those have only ONE of their five days inside the month whose climatology is subtracted — a sawtooth at every boundary, of order the month-to-month climatology difference, which in the seasonal thermocline is not negligible.", { breakLine: true, paraSpaceAfter: 3 }),
      { text: "(ii) Argo’s monthly cadence.  ", options: { bold: true, color: TEAL, fontSize: 7 } },
      body("Roemmich–Gilson is written into the single pentad holding the 15th of each month and is missing in the other five; forward-filling was REJECTED (E034_pentad_tensor.md:88–100). ", {}),
      { text: "252 live pentads, 2004–2024 — one in six.", options: { bold: true, color: RED, fontSize: 7 } },
      body(" A 35-day inner window holds ONE live Argo bin, occasionally two, and which one depends on where the anchor sits inside the month.", { breakLine: true, paraSpaceAfter: 3 }),
      body("The seasonal context token itself is the bin’s opening day-of-year (cone_sampler.pentad_doy), so it crosses month and year boundaries smoothly — that one is fine.", { italic: true, color: MUTED }),
    ], PX[1] + 0.13, R2Y + 0.32, PW - 0.26, 1.26);
    txt(s, [
      dec("DECISION → DAY-OF-YEAR HARMONIC CLIMATOLOGY (E-071 §2)"),
      body("clim(τ) = a₀ + Σₖ₌₁..₃ [aₖ cos kτ + bₖ sin kτ], with τ taken from the pentad’s MID-day over a 365.2422-day year, fitted per cell and per channel on training bins only: 7 coefficients instead of 12 box means, one pass with 35 accumulators instead of 24. K per channel is chosen by held-out variance explained (the mixed-layer depth’s spring step may need K = 4–6). No bin is ever charged for a month it barely touches, and Argo’s 252 live pentads are fitted together instead of ~21 per box. Verification: the last-pentad-of-month minus first-of-next residual, a sawtooth today, should be zero."),
    ], PX[1] + 0.13, 5.56, PW - 0.26, 0.82);

    reading(s, "Reading. Three of the four edges were handled by making the missing thing explicitly missing — an invalid token is dropped from attention, a miss token says the world was not observed, an inadmissible anchor is not trained on — and those three rules stay. The other two edges do not get handled better; they disappear. The window edge goes with the global tensor (E-071 §1); the month edge goes with the harmonic baseline (E-071 §2). Both are decided and specified, and both are UNIMPLEMENTED — nothing here is measured yet.", 6.46);
    footer(s);
  }

  // ---------------------------------------------------------------- 44e. the speeds the cone did not have — and the speeds it will
  {
    const s = pres.addSlide(); s.background = { color: WHITE };
    title(s, "The speeds the cone did not have — and the speeds it will",
      "Decision (Chris, 4 Sep): size every family from the MAXIMUM measured speed of its fastest mechanism, × 1.5 — not a typical speed, not an estimate (E-071 §4).");
    const RED = "A23B3B";
    const V_TODAY = 0.3, V2 = 5.4;
    // ---------- left: the log-scale speed ruler ----------
    const X0 = 3.15, PLW = 3.20, DEC = PLW / 3;      // 0.01 m/s -> 10 m/s
    const xv = v => X0 + (Math.log10(v) + 2) * DEC;
    const RY = 1.70, RH = 0.29, NR = 9;
    txt(s, [{ text: "MEASURED MAXIMA — what moves an overturning anomaly between latitudes. Log axis, 0.01 → 10 m/s", options: { bold: true, color: NAVY, fontSize: 8, charSpacing: 0.4 } }], 0.6, 1.32, 3.15, 0.34);
    // decade gridlines
    [0.01, 0.1, 1, 10].forEach(v => s.addShape(pres.shapes.LINE, { x: xv(v), y: RY - 0.02, w: 0, h: NR * RH + 0.06, line: { color: GRIDLINE, width: 0.5, dashType: "sysDot" } }));
    // the two dashed design speeds: what the cone has today, and what cone v2 gets
    s.addShape(pres.shapes.LINE, { x: xv(V_TODAY), y: RY - 0.16, w: 0, h: NR * RH + 0.20, line: { color: TIME, width: 1.1, dashType: "dash" } });
    s.addShape(pres.shapes.LINE, { x: xv(V2), y: RY - 0.16, w: 0, h: NR * RH + 0.20, line: { color: OBS, width: 1.6, dashType: "dash" } });
    txt(s, [{ text: "cone TODAY · 0.3 m/s", options: { bold: true, color: TIME, fontSize: 7 } }], xv(V_TODAY) - 0.52, RY - 0.34, 1.04, 0.17, { align: "center" });
    txt(s, [{ text: "CONE v2 · 5.4 m/s", options: { bold: true, color: OBS, fontSize: 7.5 } }], xv(V2) - 0.60, RY - 0.34, 1.20, 0.17, { align: "center" });
    txt(s, [{ text: "m/s", options: { bold: true, color: MUTED, fontSize: 7 } }], 6.45, RY - 0.18, 0.8, 0.17);
    const rows = [
      ["Deep Western Boundary Current", "standard oceanography — the slow return limb", 0.02, 0.1, TEAL, "0.02–0.1"],
      ["Long baroclinic Rossby wave", "Chelton et al. 2011 — westward, the interior", 0.03, null, SPACE, "0.03"],
      ["Eddies & boundary currents", "family B TODAY — ml/cone.py:97, v_ms = 0.3", 0.1, 0.3, TIME, "0.1–0.3"],
      ["Kuroshio off Taiwan", "our GLORYS bake, monthly 1° · data/currents_y", 1.62, null, SPACE, "1.62"],
      ["Agulhas Current core", "Wikipedia, Agulhas Current", 2.45, null, SPACE, "2.45"],
      ["Gulf Stream core", "NOAA Ocean Service — about 9 km/h", 2.5, null, SPACE, "~2.5"],
      ["Somali Current, our own bake", "July 2020, 2.5° N 47.5° E · data/currents_y", 2.79, null, EMB, "2.79"],
      ["First-baroclinic Kelvin wave", "Wikipedia; Chelton 1998 · Pacific in ~2 months", 2.8, null, EMB, "≈2.8"],
      ["Somali Current, peak", "Wikipedia — 7 knots, south-west monsoon", 3.6, null, RED, "3.6"],
    ];
    rows.forEach(([nme, gloss, lo, hi, col, tag], i) => {
      const y = RY + i * RH;
      txt(s, [
        { text: nme, options: { bold: true, color: col, fontSize: 7.5, breakLine: true } },
        { text: gloss, options: { fontSize: 7, color: MUTED, italic: true } },
      ], 0.6, y + 0.01, 2.45, RH - 0.03);
      if (hi) {
        s.addShape(pres.shapes.RECTANGLE, { x: xv(lo), y: y + RH / 2 - 0.05, w: xv(hi) - xv(lo), h: 0.10, fill: { color: col, transparency: 25 }, line: { width: 0 } });
      }
      const mx = xv(hi || lo);
      s.addShape(pres.shapes.OVAL, { x: mx - 0.055, y: y + RH / 2 - 0.055, w: 0.11, h: 0.11, fill: { color: col }, line: { color: WHITE, width: 0.6 } });
      txt(s, [{ text: tag, options: { bold: true, color: col, fontSize: 7 } }], 6.45, y + RH / 2 - 0.085, 0.8, 0.19);
    });
    // axis
    s.addShape(pres.shapes.LINE, { x: X0 - 0.05, y: RY + NR * RH + 0.06, w: PLW + 0.1, h: 0, line: { color: GRIDLINE, width: 0.75 } });
    [0.01, 0.1, 1, 10].forEach(v => txt(s, [{ text: String(v), options: { fontSize: 7, color: MUTED } }], xv(v) - 0.25, RY + NR * RH + 0.08, 0.5, 0.17, { align: "center" }));
    // the two mechanisms that are off the right-hand end of this ruler
    txt(s, [
      { text: "OFF THIS RULER, TO THE RIGHT.  ", options: { bold: true, color: NAVY, fontSize: 7, charSpacing: 0.4 } },
      { text: "Barotropic (surface) gravity wave ≈ 200 m/s and the jet stream ≈ 100 m/s (family A, wind stress). Both cross the planet inside one 5-day bin, so neither is a cone at all: E-071 §4.3 gives those families a RING of 24 dots at log-spaced radii out to the antipode — bearings, not density.", options: { fontSize: 7, color: INK } },
    ], 0.6, 4.60, 6.70, 0.36);

    // ---------- left bottom: reach vs lag, three speeds, y to the antipode ----------
    const CY0 = 5.30, CY1 = 6.08, CX0 = 1.20, CX1 = 4.00, KMAX = 20015;
    txt(s, [{ text: "HOW FAR A SIGNAL GETS IN SIX PENTADS — reach = v · Δt · (1 + ℓ), capped at the antipode", options: { bold: true, color: NAVY, fontSize: 8, charSpacing: 0.4 } }], 0.6, 5.02, 4.70, 0.20);
    const lx = k => CX0 + k * (CX1 - CX0) / 6, ly = km => CY1 - km / KMAX * (CY1 - CY0);
    s.addShape(pres.shapes.LINE, { x: CX0, y: CY1, w: CX1 - CX0 + 0.1, h: 0, line: { color: GRIDLINE, width: 0.75 } });
    s.addShape(pres.shapes.LINE, { x: CX0, y: CY0 - 0.05, w: 0, h: CY1 - CY0 + 0.05, line: { color: GRIDLINE, width: 0.75 } });
    [[0, "0", 0], [4444, "4,444 — E-069 cap", 2], [10000, "10,000", 0], [15000, "15,000", 0], [20015, "20,015 — antipode", 1]].forEach(([km, t, kind]) => {
      const col = kind ? (kind === 1 ? RED : TIME) : GRIDLINE;
      s.addShape(pres.shapes.LINE, { x: CX0, y: ly(km), w: CX1 - CX0 + 0.1, h: 0, line: { color: col, width: kind ? 0.9 : 0.4, dashType: "sysDot" } });
      txt(s, [{ text: t, options: { fontSize: kind ? 6 : 6.5, color: kind ? col : MUTED, bold: !!kind } }], 0.30, ly(km) - 0.08, 0.86, 0.16, { align: "right" });
    });
    for (let k = 0; k <= 6; k++) txt(s, [{ text: String(k), options: { fontSize: 7, color: MUTED } }], lx(k) - 0.14, CY1 + 0.02, 0.28, 0.16, { align: "center" });
    txt(s, [{ text: "lag ℓ, in pentads (one pentad = one 5-day bin)", options: { fontSize: 6.5, italic: true, color: MUTED } }], CX0 + 0.06, CY1 + 0.18, 2.4, 0.16);
    [[V2, OBS, "5.4 m/s — CONE v2", "2,333 → 16,330 km"], [2.8, EMB, "2.8 m/s — Kelvin", "8,467 km at lag 6"], [V_TODAY, TIME, "0.3 m/s — E-069 today", "907 km at lag 6"]].forEach(([v, col, l1, l2]) => {
      const step = v * 86400 * 5 / 1000;   // km per (1 + lag) unit
      for (let k = 0; k < 6; k++) {
        const a = Math.min(step * (1 + k), KMAX), b = Math.min(step * (2 + k), KMAX);
        s.addShape(pres.shapes.LINE, { x: lx(k), y: ly(b), w: lx(k + 1) - lx(k), h: Math.abs(ly(b) - ly(a)), flipV: true, line: { color: col, width: v === V2 ? 2 : 1.4 } });
      }
      const end = Math.min(step * 7, KMAX);
      s.addShape(pres.shapes.OVAL, { x: lx(6) - 0.05, y: ly(end) - 0.05, w: 0.1, h: 0.1, fill: { color: col }, line: { width: 0 } });
      txt(s, [{ text: l1, options: { bold: true, color: col, fontSize: 7, breakLine: true } },
              { text: l2, options: { color: col, fontSize: 6.5, italic: true } }], CX1 + 0.12, ly(end) - 0.12, 1.32, 0.26);
    });

    // ---------- the asymmetry note, beside the reach chart ----------
    txt(s, [
      { text: "Why the buffer is 1.5, and why it is spent on the WIDE side. ", options: { bold: true, color: RED, fontSize: 7 } },
      { text: "The tensor is a 5-day mean on a 0.25° grid and under-reads peaks; the maxima above are literature and reanalysis values, not the true extreme; and a warming ocean may run faster. A cone that is too WIDE only costs tokens. A cone that is too NARROW asserts that the driver cannot have arrived, and the model has no way to discover that the assertion was wrong.", options: { fontSize: 7, color: INK } },
    ], 5.48, 5.02, 1.82, 1.36);

    // ---------- right, box 1: the Argo channels ----------
    box(s, 7.45, 1.40, 5.25, 2.16, DERIV, "FDF6F6");
    txt(s, [{ text: "1 · Argo had no spatial reach at all — now it has 26 profile tokens", options: { bold: true, color: DERIV, fontSize: 9.5 } }], 7.58, 1.46, 4.99, 0.22);
    txt(s, [
      { text: "ml/cone.py::channel_family (:122) assigns rg_t* / rg_s* — the 32 Roemmich–Gilson temperature and salinity channels — to family B (0.3 m/s), but channel_dots (:261) OVERRIDES that and returns the ANCHOR COLUMN ONLY. Their effective reach is ", options: { fontSize: 7, color: INK } },
      { text: "0 km at every lag", options: { bold: true, color: RED, fontSize: 7 } },
      { text: ", so no propagation of a subsurface anomaly toward the anchor is visible at any speed — and those channels are still ", options: { fontSize: 7, color: INK } },
      { text: "192 of the 706 dots (27.2 %)", options: { bold: true, color: RED, fontSize: 7 } },
      { text: ", a quarter of the budget on one vertical column. The stated reason was token economy: Argo is live one pentad in six.", options: { fontSize: 7, color: INK } },
    ], 7.58, 1.70, 4.99, 0.72);
    txt(s, [
      { text: "DECISION → E-071 §3  ", options: { bold: true, color: OBS, fontSize: 7, charSpacing: 0.4 } },
      { text: "the two most recent LIVE pentads (lags 0…11, known from the calendar, not from the data), 13 locations each — the anchor plus a 12-dot sunflower at the v2 reach — and each location is ONE profile token carrying 16 T + 16 S values with the pressure ladder as an inner axis. That is 26 tokens instead of 192, every one of them observed. Later, with native floats: the k nearest live profiles inside the cone, no sunflower at all.", options: { fontSize: 7, color: INK } },
    ], 7.58, 2.46, 4.99, 0.60);
    {
      const bx0 = 7.58, bw = 4.99, by = 3.10;
      let cx = bx0;
      [[45.3, SPACE], [27.2, DERIV], [22.9, TIME], [4.5, MUTED]].forEach(([p, col]) => {
        const w = bw * p / 100;
        s.addShape(pres.shapes.RECTANGLE, { x: cx, y: by, w, h: 0.15, fill: { color: col, transparency: 15 }, line: { color: WHITE, width: 0.6 } });
        cx += w;
      });
      txt(s, [
        { text: "■ ", options: { color: SPACE, fontSize: 8 } }, { text: "currents + SSH 320 (45.3 %)   ", options: { fontSize: 7, color: INK } },
        { text: "■ ", options: { color: DERIV, fontSize: 8 } }, { text: "Argo depth 192 (27.2 %)   ", options: { fontSize: 7, color: INK } },
        { text: "■ ", options: { color: TIME, fontSize: 8 } }, { text: "SST + MLD 162 (22.9 %)   ", options: { fontSize: 7, color: INK } },
        { text: "■ ", options: { color: MUTED, fontSize: 8 } }, { text: "wind 32 (4.5 %)", options: { fontSize: 7, color: INK } },
      ], bx0, by + 0.17, bw, 0.20);
    }

    // ---------- right, box 2: the decision table and its two consequences ----------
    box(s, 7.45, 3.64, 5.25, 2.74, OBS, "F5FBF7");
    txt(s, [{ text: "2 · The design speeds — one per family, E-071 §4.2", options: { bold: true, color: OBS, fontSize: 9.5 } }], 7.58, 3.70, 4.99, 0.22);
    {
      const th = t => ({ text: t, options: { bold: true, color: WHITE, fill: { color: NAVY }, fontSize: 6.5, valign: "middle" } });
      const td = (t, o) => ({ text: t, options: Object.assign({ fontSize: 6.5, color: INK, valign: "top" }, o || {}) });
      const trows = [
        [th("family"), th("fastest mechanism that carries it"), th("max"), th("× 1.5 = v_design"), th("reach / pentad")],
        [td("B — ocean", { bold: true, color: SPACE }), td("Somali Current (advection) ≥ first-baroclinic Kelvin (adjustment)"), td("3.6 m/s"), td("5.4 m/s", { bold: true, color: OBS }), td("2,333 km")],
        [td("C — surface", { bold: true, color: TIME }), td("atmosphere at lags 0–1, then the ocean"), td("— / 3.6"), td("global / 5.4 m/s", { bold: true, color: OBS }), td("— / 2,333 km")],
        [td("A — wind", { bold: true, color: MUTED }), td("the jet stream (storms translate at 10–20 m/s)"), td("~100 m/s"), td("global", { bold: true, color: OBS }), td("within one pentad")],
      ];
      s.addTable(trows, { x: 7.58, y: 3.94, w: 4.99, colW: [0.72, 1.86, 0.55, 0.86, 1.00], fontFace: FONT_B, border: { type: "solid", color: GRIDLINE, pt: 0.5 }, rowH: [0.20, 0.30, 0.24, 0.24], margin: 0.03, autoPage: false });
    }
    txt(s, [
      { text: "(i) Dot placement becomes LOG-RADIAL, 36 dots per lag for the ocean families.  ", options: { bold: true, color: NAVY, fontSize: 7 } },
      { text: "Uniform-area placement at 24 dots gives 4.8 dots per octave of distance over 907 km but only 2.5 over 16,330 km — the near field, where the eddies live, would be starved by a wider cone at the same budget.", options: { fontSize: 7, color: INK, breakLine: true, paraSpaceAfter: 3 } },
      { text: "(ii) Token budget 748 → 1,600 per anchor  ", options: { bold: true, color: NAVY, fontSize: 7 } },
      { text: "(B 888 · C 444 · A 200 · Argo 26 · lag-0 patch 42): about 2.1× the encoder cost, and linear in a Perceiver (64 latents cross-attend to the tokens). The fallback that brings it back to ~500 is one token per (lag, location) carrying all channels as a value vector — kept OUT of v2 so that v2 measures the geometry and nothing else.", options: { fontSize: 7, color: INK, breakLine: true, paraSpaceAfter: 3 } },
      { text: "E-069 keeps its geometry — its five seeds are done and its H1 verdict stands (slide 44). Cone v2 lands on family 7 — the first tensor covering the whole globe rather than the North Atlantic window.", options: { fontSize: 7, italic: true, color: MUTED } },
    ], 7.58, 5.06, 4.99, 1.26);

    reading(s, "Reading. The cone was a one-speed model of a multi-speed ocean. Cone v2 sizes each family from the fastest thing that actually carries it — the Somali Current’s 3.6 m/s for the ocean, the jet stream for the wind — with the 1.5 buffer spent on the wide side, because a cone that is too narrow excludes true causes silently while one that is too wide only costs tokens. Nothing here is measured. And after E-069’s H1 (slide 44), the case for v2 is rolled skill and persistence context, not velocity: a codec that could not read a displacement at 130 km will not read one at 2,333 km.", 6.46);
    footer(s);
  }

  // ---------------------------------------------------------------- 44f. land, ice and air — the shared-channel principle (E-071 §6)
  {
    const s = pres.addSlide(); s.background = { color: WHITE };
    title(s, "Land, ice and air — nothing is dark by design",
      "Chris, 4 Sep: \u201Cwhy not add some data on land, too? \u2026 sparse for now \u2026 land surface temperature, satellite imagery, radar reflection, air temperature across heights\u201D and \u201Csome of our channels should be available for both the oceans and the land\u201D. E-071 \u00A76: a channel is a quantity and an instrument, never a sphere.");
    const RED = "A23B3B";
    const OTINT = "D6E7F5", LTINT = "E1EFD5";      // the two surfaces, tinted so a shared row reads as one row across both

    // ---------- LEFT (~60 %): the shared-channel table ----------
    const TX = 0.6, TW = 7.25;
    txt(s, [
      { text: "THE SHARED SET \u2014 one quantity, one instrument, two surfaces.  ", options: { bold: true, color: NAVY, fontSize: 8, charSpacing: 0.4 } },
      { text: "A per-cell SPHERE CODE (ocean \u00B7 land \u00B7 ice sheet \u00B7 inland water) says which retrieval filled the cell, so one channel means sea-surface temperature at one anchor and land-surface temperature at the next.", options: { fontSize: 7, color: INK } },
    ], TX, 1.60, TW, 0.32);
    {
      const th = t => ({ text: t, options: { bold: true, color: WHITE, fill: { color: NAVY }, fontSize: 7, valign: "middle" } });
      const q = t => ({ text: t, options: { bold: true, color: NAVY, fontSize: 7, valign: "top" } });
      const o = t => ({ text: t, options: { fontSize: 7, color: INK, valign: "top", fill: { color: OTINT } } });
      const l = t => ({ text: t, options: { fontSize: 7, color: INK, valign: "top", fill: { color: LTINT } } });
      const src = t => ({ text: t, options: { fontSize: 7, color: MUTED, valign: "top" } });
      const trows = [
        [th("quantity"), th("over ocean"), th("over land / ice"), th("source, at 0.25\u00B0 pentads")],
        [q("surface (skin) temperature"), o("SST"), l("land- / ice-surface temperature"),
         src("ERA5 skin T, gap-free 1940\u2192; then the observed OSTIA SST + MODIS MOD11 LST, 1 km daily clear-sky 2000\u2192")],
        [q("near-surface air"), o("2 m T, 2 m dew point, 10 m wind"), l("the same"),
         src("ERA5 \u2014 this row retires the FROZEN NCEP R1 wind source")],
        [q("air aloft \u2014 \u201Cacross heights\u201D"), o("T, wind, humidity at 850 / 500 / 250 hPa"), l("the same"),
         src("ERA5 pressure levels; IGRA radiosondes (~800 stations, 00/12 UTC) as native dots later")],
        [q("pressure, precipitation, radiation, fluxes"), o("pressure, precip, net SW + LW, latent + sensible heat"), l("the same"),
         src("ERA5; IMERG 0.1\u00B0 as the observed precipitation channel, 2000\u2192")],
        [q("frozen fraction"), o("sea-ice concentration"), l("snow-cover fraction; ice sheet \u2261 1"),
         src("OSI SAF / NSIDC sea ice 25 km 1978\u2192 \u00B7 MOD10A1 snow 500 m daily 2000\u2192 \u2014 ONE channel")],
        [q("albedo"), o("\u2248 constant, plus the ice"), l("vegetation, snow, soil, ice"),
         src("MCD43A3 500 m daily 2000\u2192 \u2014 one channel; the ocean value is the constant plus the ice")],
        [q("radar cross-section \u03C3\u2070"), o("wind and waves roughen it"), l("soil moisture and vegetation set it"),
         src("ASCAT 25 km twice daily 2007\u2192 \u2014 the SAME instrument over both. The \u201Cradar reflection\u201D channel, raw")],
        [q("optical reflectance"), o("ocean colour"), l("vegetation, snow, soil"),
         src("MODIS / VIIRS 500 m \u2014 red, near-IR, one short-wave IR, raw, 2000\u2192. Chlorophyll and NDVI become TARGETS")],
        [q("L-band brightness temperature"), o("sea-surface salinity"), l("soil moisture"),
         src("SMAP 36 km 2015\u2192 \u2014 one instrument; the raw T_B is the channel, both retrievals are targets")],
        [q("mass / water storage"), o("ocean bottom pressure"), l("terrestrial water storage; ice-sheet mass"),
         src("GRACE / GRACE-FO mascons, monthly 2002\u2192 (2017\u201318 gap) \u2014 already one global field")],
        [q("surface height anomaly"), o("sea-level anomaly (DUACS 0.125\u00B0, 1993\u2192)"), l("ice-sheet elevation change; lake and river height"),
         src("The same radar altimeters: DUACS over sea, CryoSat-2 (2010\u2192) / ICESat-2 (2018\u2192) over ice, Hydroweb over lakes")],
        [q("surface velocity"), o("currents (GLORYS u, v)"), l("ice velocity (ITS_LIVE); land \u2261 0"),
         src("One (u, v) pair. On land it is exactly zero \u2014 which is information: nothing advects")],
        [q("the column below"), o("T, S profiles (Argo, 0\u20132,000 m)"), l("soil layers (ERA5-Land, SMAP L4); firn on the ice sheets"),
         src("ONE profile-token type (E-071 \u00A73) with a sphere-specific depth ladder")],
      ];
      s.addTable(trows, { x: TX, y: 1.94, w: TW, colW: [1.30, 1.40, 1.60, 2.95], fontFace: FONT_B,
        border: { type: "solid", color: GRIDLINE, pt: 0.5 },
        rowH: [0.20, 0.30, 0.30, 0.30, 0.30, 0.30, 0.30, 0.30, 0.30, 0.30, 0.30, 0.30, 0.30, 0.30],
        margin: 0.03, autoPage: false });
    }
    txt(s, [
      { text: "Sphere-specific channels stay sphere-specific, and are ", options: { fontSize: 7, color: INK } },
      { text: "miss", options: { bold: true, color: LAND, fontSize: 7 } },
      { text: " elsewhere with a clear conscience: mixed-layer depth and sea-surface height (ocean); ESA CCI soil moisture 0.25\u00B0 daily 1978\u2192 \u2014 it covers the whole axis \u2014 snow water equivalent, leaf area, GLEAM evapotranspiration (land); ice velocity and elevation change (ice).", options: { fontSize: 7, color: INK } },
    ], TX, 6.04, TW, 0.36);

    // ---------- RIGHT (~40 %): three boxes ----------
    const BX = 8.05, BW = 4.68, IX = BX + 0.13, IW = BW - 0.26;
    // 1 · Antarctica
    box(s, BX, 1.56, BW, 1.52, SPACE, "F7FAFC");
    txt(s, [{ text: "1 \u00B7 Antarctica was never unobserved \u2014 the GRID was", options: { bold: true, color: SPACE, fontSize: 9 } }], IX, 1.62, IW, 0.20);
    txt(s, [
      { text: "E-070 chose \u221280\u202680\u00B0 because that is GLORYS12\u2019s extent \u2014 an ocean product stops where the ocean stops \u2014 and \u00A71 of the plan wrongly let it read as ", options: { fontSize: 7, color: INK } },
      { text: "no data", options: { bold: true, color: RED, fontSize: 7 } },
      { text: ". Cone v2\u2019s grid is ", options: { fontSize: 7, color: INK } },
      { text: "\u221290 \u2026 90\u00B0, 721 \u00D7 1,440", options: { bold: true, color: OBS, fontSize: 7 } },
      { text: ". The ice-sheet rows are filled by ERA5 (skin and 2 m temperature, winds, pressure, snowfall, radiation \u2014 the surface mass balance\u2019s inputs), MODIS LST and albedo in the clear-sky months, GRACE mass, CryoSat-2 (2010\u2192) and ICESat-2 (2018\u2192) elevation change, ITS_LIVE velocity, and around the continent the sea-ice concentration and drift.", options: { fontSize: 7, color: INK, breakLine: true, paraSpaceAfter: 3 } },
      { text: "Why it matters for the overturning: ", options: { bold: true, color: SPACE, fontSize: 7 } },
      { text: "Antarctic meltwater and sea-ice formation set Antarctic Bottom Water, the lower limb of the global overturning \u2014 and the ice sheet\u2019s freshwater flux is its largest uncertainty.", options: { fontSize: 7, color: INK } },
    ], IX, 1.84, IW, 1.18);
    // 2 · cone families for the new spheres
    box(s, BX, 3.13, BW, 1.36, LAND, "F7FBF6");
    txt(s, [{ text: "2 \u00B7 Cone families for the new spheres (E-071 \u00A76.3)", options: { bold: true, color: LAND, fontSize: 9 } }], IX, 3.19, IW, 0.20);
    txt(s, [
      { text: "atmosphere A\u2032 ", options: { bold: true, color: ATMOS, fontSize: 7 } },
      { text: "\u2014 global at once, lags 0\u20132, the 24-dot log-spaced ring to the antipode.", options: { fontSize: 7, color: INK, breakLine: true, paraSpaceAfter: 2 } },
      { text: "land surface state L ", options: { bold: true, color: LAND, fontSize: 7 } },
      { text: "\u2014 v = 0: the land does not advect, so the reach is the correlation length alone (~300\u2013500 km) and the memory is months. A column plus a 12-dot disc at every inner lag and into the outer window. The SAME channel over an OCEAN anchor follows family C at 5.4 m/s \u2014 the one place a family depends on the anchor\u2019s sphere.", options: { fontSize: 7, color: INK, breakLine: true, paraSpaceAfter: 2 } },
      { text: "ice sheet I ", options: { bold: true, color: SPACE, fontSize: 7 } },
      { text: "\u2014 column only, monthly cadence, the firn profile token.", options: { fontSize: 7, color: INK, breakLine: true, paraSpaceAfter: 2 } },
      { text: "The sphere code (ocean \u00B7 land \u00B7 ice \u00B7 inland water, from the rung-12 static mask) is a coordinate the codec reads, exactly as depth is.", options: { fontSize: 7, italic: true, color: MUTED } },
    ], IX, 3.41, IW, 1.02);
    // 3 · cost and the token change
    box(s, BX, 4.54, BW, 1.86, OBS, "F5FBF7");
    txt(s, [{ text: "3 \u00B7 Cost, and the token change this forces (\u00A76.4\u20136.5)", options: { bold: true, color: OBS, fontSize: 9 } }], IX, 4.60, IW, 0.20);
    txt(s, [
      { text: "One global channel at 0.25\u00B0 \u00D7 5 d \u00D7 1982\u20132024 \u2248 6.5 GB float16; the shared set of ~20 channels \u2248 130 GB, plus ~5 land-only \u2248 33 GB, on top of the 62 GB ocean tensor. Two sources \u2014 ERA5 and MODIS \u2014 supply three quarters; ERA5 needs the free CDS account.", options: { fontSize: 7, color: INK, breakLine: true, paraSpaceAfter: 2 } },
      { text: "With ~65 channels, per-channel tokens would be ~14,000 per anchor. So v2 makes a TOKEN = one (lag, location) carrying the full value vector of EVERY channel there, with a per-channel observed mask inside and the sphere code as a coordinate \u2014 the \u00A73 profile token generalised. ", options: { fontSize: 7, color: INK } },
      { text: "\u2248 300 tokens per anchor whatever the channel count, below E-069\u2019s 748.", options: { bold: true, color: OBS, fontSize: 7, breakLine: true, paraSpaceAfter: 2 } },
      { text: "PHASE L0 \u2014 \u201Csparse for now\u201D: ", options: { bold: true, color: NAVY, fontSize: 7 } },
      { text: "six channels, two sources, all gap-free, all 1982\u2192, all global to the poles, ~40 GB \u2014 ERA5 skin T, 2 m T, 10 m u/v (retiring the frozen NCEP R1), surface pressure, precipitation; plus the frozen fraction and ESA CCI soil moisture. MODIS / GRACE / SMAP / ASCAT (2000\u2192) follow as L1, once the harmonic climatology has shown it handles a channel that starts eighteen years into the axis.", options: { fontSize: 7, color: INK } },
    ], IX, 4.82, IW, 1.52);

    reading(s, "Reading. Today a land cell is told \u201Cthe world was not observed here\u201D 42 times per pentad, which is false. Cone v2 defines channels by what is measured and by which instrument, lets a sphere code say what surface lies under the cell, and finds that most of the input ladder already crosses the coastline \u2014 the same skin temperature, the same radar, the same reflectance, the same altimeter, the same column token. What stays dark is only what no instrument sees.", 6.48);
    footer(s);
  }

  // ---------------------------------------------------------------- 45. phases and the decisive experiment
  {
    const s = pres.addSlide(); s.background = { color: WHITE };
    title(s, "Phase 0 first — the decisive experiment fits the data we already have", "Then one sphere per phase, in the agreed order: ocean surface + atmosphere as forcing, interior, ocean biosphere, land, fast atmosphere");
    box(s, 0.6, 1.55, 12.1, 1.9, TIME, "FFF6EE");
    txt(s, [
      { text: "Phase 0 — the A/B on today's tokens (weeks).  ", options: { bold: true, color: TIME, fontSize: 12 } },
      { text: "Same tokens the stage-2 pool reads today: Earth 2 ocean pixel-months, 3 × 3 codec, sunflower-89 × 24 months. Train a cone codec over those tokens with channel drop + lag-band drop + anchor reconstruction. Compare at equal total context under the corrected window-scope pool (paper v8, 2 Sep 2026), after its null ladder exists, with n ≥ 3 seeds per arm:", options: { fontSize: 10.5, breakLine: true, paraSpaceAfter: 5 } },
      { text: "A  current codec + stage-2 attention head   ·   B  cone codec + ridge probe   ·   B′  cone codec + attention head", options: { fontSize: 10.5, bold: true, breakLine: true, paraSpaceAfter: 5 } },
      { text: "Verdict metric: rolled skill — MSSS vs climatology and vs damped persistence per lead, trained and held-out longitudes separately, block-bootstrap intervals, the LIM null beside every number. Diagnostics (not verdicts): a velocity probe (GLORYS u, v at the anchor; drifters where available; derived channels masked at test) and the RAPID transport probe.", options: { fontSize: 10.5, breakLine: true, paraSpaceAfter: 5 } },
      { text: "Predictions: B ≥ A on rolled skill with a linear head, and B beats the embedding-space LIM where A does not; velocity probe B ≫ A; static probes equal. If B needs B′ to match A, the 'work in the embedding' argument loses on sample efficiency and the hybrid keeps a thick stage 2. Effects inside the tier's replicate band are written as consistencies, never levels.", options: { fontSize: 10.5, italic: true } },
    ], 0.75, 1.62, 11.8, 1.8);
    const phases = [
      ["1", "Ocean surface + atmosphere as forcing", "rungs 6 + 7 as gridded dots (OSTIA, DUACS, ASCAT, IMERG, AMSR2 ice; ERA5 / IFS open data); L-shaped and A-family cones; pentad anchors at 25 km", "SST and SSH forecast skill at 1–4 weeks vs persistence and vs the Copernicus 1/12° forecast; sea-ice edge", OCEAN],
      ["2", "Interior", "Argo core / Deep / BGC and moorings as native dots; GLORYS flagged derived, dropped 50 %; depth-band drop objective", "AMOC again; mixed-layer depth; heat content 0–700 m", OCEAN],
      ["3", "Ocean biosphere", "ocean-colour L3 (gaps stay gaps), PACE, SOCAT, BGC-Argo variables", "carbon-flux reconstruction vs SeaFlux / Copernicus carbon (held out, not input)", BIO],
      ["4", "Land", "local codec on Sentinel-2/-1 and Landsat tiles (rungs 1–2); rung-10 fields as dots; column cones with the atmosphere hat", "LAI / SIF and soil-moisture forecasts, fire risk; standard GeoFM benchmarks on e_loc for comparability", LAND],
      ["5", "Fast atmosphere", "geostationary imagers and ground radar at 10-min cadence through a second local codec (spatio-temporal tubelets); family-A disc at hourly Δt", "nowcasting skill; wind-driven SST and ice response", ATMOS],
    ];
    const head = [hdr("Phase"), hdr("Sphere"), hdr("What enters"), hdr("New probes")];
    const body = phases.map(p => [{ text: p[0], options: { bold: true, color: p[4], fontSize: 12, align: "center", valign: "middle" } }, nm(p[1], p[4]), c(p[2], { fontSize: 8.5 }), c(p[3], { fontSize: 8.5 })]);
    s.addTable([head, ...body], { x: 0.6, y: 3.6, w: 8.2, colW: [0.6, 1.7, 3.3, 2.6], fontFace: FONT_B, border: { type: "solid", color: GRIDLINE, pt: 0.5 }, rowH: [0.28, 0.55, 0.45, 0.42, 0.55, 0.55], margin: 0.04, autoPage: false });
    box(s, 9.0, 3.6, 3.7, 3.25, NAVY, "F4F6F9");
    txt(s, [
      { text: "Ablations that run in every phase", options: { bold: true, color: NAVY, fontSize: 10.5, breakLine: true, paraSpaceAfter: 4 } },
      ...["cone-in vs cone-on-top at equal context", "drop lag bands", "drop the far field", "drop depth", "derived channels masked at test", "cone × 2 vs cone × ½ on v and τ"].map((t, i) => ({ text: `${i + 1}  ${t}`, options: { fontSize: 10, breakLine: true, paraSpaceAfter: 4 } })),
      { text: "Engineering first: a pre-materialised cone index per channel family (which tiles, dates and profiles fall in which anchors' cones). The sampler, not the network, is the bottleneck — the network is under an hour per year of 25 km pentad ocean embeddings at 100 TFLOP/s, about a day for 1 km monthly land.", options: { fontSize: 8.5, italic: true, color: MUTED } },
    ], 9.15, 3.68, 3.45, 3.15);
    footer(s);
  }

  // ---------------------------------------------------------------- 46. data continuity 2026–27
  {
    const s = pres.addSlide(); s.background = { color: WHITE };
    title(s, "Data continuity 2026–27 — build the archive sensor-agnostic", "What the verification pass found switching off or changing (agency notices, 2 Sep 2026). Channel-drop training turns a missing sensor into a masked channel, not a failure.");
    const c9 = (t) => c(t, { fontSize: 9 });
    const rows = [
      [hdr("Source"), hdr("What changes"), hdr("Consequence for the ladder")],
      [nm("Sentinel-1", LAND), c9("S1B failed Dec 2021; S1C launched 5 Dec 2024, S1D 4 Nov 2025 (data open 17 Apr 2026); S1A terminated 29 Jun 2026"), c9("Constellation is S1C + S1D at 6-day repeat, timeline shifted one day; radar record has a 2022–24 single-satellite gap")],
      [nm("Sentinel-2", LAND), c9("Three satellites (S2A/B/C) only until end-2026; S2C is prime"), c9("Revisit drops back to 5 days in 2027 — do not tune the local codec to the 2025–26 cadence")],
      [nm("MODIS → VIIRS", OCEAN), c9("Terra/Aqua shut down late 2026 / early 2027; Suomi-NPP data delivery ends 1 Nov 2026"), c9("Rung 3 continues on NOAA-21 (primary) and NOAA-20; the 2000–2026 MODIS archive is training data with a hard end")],
      [nm("Passive microwave", OCEAN), c9("DMSP SSMIS retiring ~Sep 2026; NSIDC NRT SSMIS sea ice retired 18 Jun 2026; NOAA AMSR2 NRT ends Sep 2026 for AMSR3 (GOSAT-GW, launched 28 Jun 2025)"), c9("NRT sea-ice concentration and drift now depend on one 2012 satellite (AMSR2) until AMSR3 products are operational")],
      [nm("Altimetry", OCEAN), c9("Sentinel-6B launched 17 Nov 2025, reference hand-over date not yet published; DUACS L4 now 0.125°; SWOT L4 only experimental"), c9("Sea-level rung stable; treat the 0.25° legacy grid and the new 0.125° grid as one channel with a resolution flag")],
      [nm("Climate records that stop", BIO), c9("OC-CCI v6 ends 31 Dec 2024; ESA CCI soil moisture and snow are annual retrospective releases; OSCAR final ends Aug 2022; FLUXNET2015 frozen (Feb 2020)"), c9("Use the NRT twins for 2025+ (GlobColour L3, ASCAT / SMAP, OSCAR NRT, ICOS / AmeriFlux); expect version seams"),],
      [nm("Licences to watch", STAT), c9("EN4 non-commercial; GRDC research-only, no redistribution; OPERA radar for EUMETNET members; GLEAM commercial use by approval; ECMWF 9 km IFS licensed (0.25° open data free)"), c9("Keep licensed sources out of any embedding meant for release; they can still supervise probes")],
    ];
    s.addTable(rows, { x: 0.6, y: 1.5, w: 12.1, colW: [1.9, 5.3, 4.9], fontFace: FONT_B, border: { type: "solid", color: GRIDLINE, pt: 0.5 }, rowH: [0.28, 0.55, 0.45, 0.55, 0.65, 0.55, 0.65, 0.65], margin: 0.04, autoPage: false });
    reading(s, "Also verified: ERA5's SST and sea ice are prescribed inputs (HadISST2, OSI SAF, OSTIA), so they must never be scored as ocean observations; ERA5-Land has no data assimilation at all — a land model replayed on ERA5 forcing; Copernicus waves are MFWAM, not WAVEWATCH; Copernicus surface carbon is 0.25°, not 1°; the OPERA composite is now 1 km / 5 min (CIRRUS), not the retired 2 km / 15 min ODYSSEY. Float counts are the live OceanOPS figures of 2 Sep 2026 and move daily.", 6.15);
    footer(s);
  }

  // ---------------------------------------------------------------- 46b. El Niño 2026 — catalogue audit and additions
  {
    const s = pres.addSlide(); s.background = { color: WHITE };
    title(s, "El Niño 2026 — what the catalog predicts, and what to add",
      "Our own OISST bake says Niño-3.4 (the central-Pacific sea-temperature anomaly that defines the event) went −0.6 °C in January to +2.1 °C in July 2026; NOAA's Climate Prediction Center, 13 Aug: El Niño Advisory, > 90 % chance of a very strong event, 69 % chance of exceeding +2.5 °C in Oct–Dec — stronger than any event since 1950");
    const COLD = SPACE, WARM = "C0443A";
    // ---------------- left: the bar chart ----------------
    const PX0 = 1.05, PX1 = 4.05, PY0 = 1.78, PY1 = 4.28;   // plot box
    const VMIN = -1.0, VMAX = 2.6, SC = (PY1 - PY0) / (VMAX - VMIN);
    const yOf = v => PY1 - (v - VMIN) * SC;
    s.addShape(pres.shapes.RECTANGLE, { x: PX0, y: PY0, w: PX1 - PX0, h: PY1 - PY0, fill: { color: "FBFCFD" }, line: { color: GRIDLINE, width: 0.5 } });
    [-1, 0, 1, 2].forEach(v => {
      s.addShape(pres.shapes.LINE, { x: PX0, y: yOf(v), w: PX1 - PX0, h: 0, line: { color: v === 0 ? MUTED : GRIDLINE, width: v === 0 ? 1 : 0.5 } });
      s.addText((v > 0 ? "+" : "") + v.toFixed(0), { x: 0.6, y: yOf(v) - 0.09, w: 0.4, h: 0.18, fontFace: FONT_B, fontSize: 7, color: MUTED, align: "right", isTextBox: true, margin: 0 });
    });
    // December peaks of the three big events, from the same bake
    [[2.46, "Dec 2015  +2.46"], [2.18, "Dec 1997  +2.18"], [2.01, "Dec 2023  +2.01"]].forEach(([v, t]) => {
      s.addShape(pres.shapes.LINE, { x: PX0, y: yOf(v), w: PX1 - PX0, h: 0, line: { color: MUTED, width: 0.75, dashType: "dash" } });
      s.addText(t, { x: PX0 + 0.05, y: yOf(v) - 0.135, w: 1.5, h: 0.14, fontFace: FONT_B, fontSize: 7, color: MUTED, isTextBox: true, margin: 0 });
    });
    const MONTHS = [["Jan", -0.57], ["Feb", -0.19], ["Mar", 0.02], ["Apr", 0.47], ["May", 0.94], ["Jun", 1.60], ["Jul", 2.09]];
    const slot = (PX1 - PX0) / MONTHS.length, bw = 0.30;
    MONTHS.forEach(([m, v], i) => {
      const bx = PX0 + i * slot + (slot - bw) / 2;
      const y0 = yOf(Math.max(v, 0)), h = Math.abs(yOf(v) - yOf(0));
      s.addShape(pres.shapes.RECTANGLE, { x: bx, y: y0, w: bw, h: Math.max(h, 0.012), fill: { color: v < 0 ? COLD : WARM }, line: { width: 0 } });
      s.addText((v > 0 ? "+" : "") + v.toFixed(2), { x: bx - 0.15, y: v < 0 ? yOf(v) + 0.02 : y0 - 0.17, w: bw + 0.3, h: 0.16, fontFace: FONT_B, fontSize: 7, bold: true, color: v < 0 ? COLD : WARM, align: "center", isTextBox: true, margin: 0 });
      s.addText(m, { x: bx - 0.1, y: PY1 + 0.03, w: bw + 0.2, h: 0.18, fontFace: FONT_B, fontSize: 7.5, color: INK, align: "center", isTextBox: true, margin: 0 });
    });
    txt(s, [
      { text: "Niño-3.4 monthly anomaly, 2026", options: { bold: true, color: NAVY, fontSize: 9, breakLine: true } },
      { text: "5° S–5° N, 170° W–120° W, 500 one-degree cells", options: { fontSize: 7.5, italic: true, color: MUTED } },
    ], 0.6, 1.5, 3.5, 0.3);
    txt(s, [
      { text: "Computed here from data/oisst_y (NOAA's daily satellite-and-buoy sea-temperature analysis, block-meaned to 1° monthly) against data/oisst_clim, the app's 1991–2020 normal. The Climate Prediction Center's July value is +1.4 °C — a different sea-temperature reconstruction (ERSSTv5), a different averaging and possibly the relative index CPC now favours; the 0.7 °C gap is unexplained here, and is itself the reason to carry the official index beside our own bake.", options: { fontSize: 8, color: INK } },
    ], 0.6, 4.56, 3.5, 1.8);
    // ---------------- middle: what the catalogue already has ----------------
    const MX = 4.25, MW = 3.7;
    box(s, MX, 1.5, MW, 4.86, TEAL, "F2F8FA");
    const pair = (a, b) => [
      { text: a + "  ", options: { bold: true, color: TEAL, fontSize: 8 } },
      { text: b, options: { fontSize: 8, color: INK, breakLine: true, paraSpaceAfter: 3 } },
    ];
    txt(s, [
      { text: "Already in the catalog", options: { bold: true, color: NAVY, fontSize: 10.5, breakLine: true } },
      { text: "274 records, compiled for the Atlantic overturning circulation — its `amoc` flag mis-sorts for El Niño. what → signal · limit", options: { fontSize: 7.5, italic: true, color: MUTED, breakLine: true, paraSpaceAfter: 4 } },
      ...pair("OISST v2.1", "→ the label itself, all four Niño boxes. Baked global 1° monthly 1981-09 → 2026-07."),
      ...pair("GLORYS12 surface currents + mixed-layer depth", "→ equatorial advection. Baked 1993 → 2026-05 · the depth levels are not baked."),
      ...pair("GPCP 1979→ / IMERG 2000→", "→ the rainfall shift towards the dateline that is El Niño's atmospheric expression · only a climatology is baked from GPCP; IMERG is live."),
      ...pair("MERRA-2 10 m wind", "→ trade winds · monthly tiles only, so a westerly wind burst (a 5–15 day reversal) is averaged away."),
      ...pair("NASA sea-surface height", "→ Kelvin and Rossby waves · the app's tiles END 2019-01."),
      ...pair("OSCAR surface currents", "→ zonal advection · tiles 2014-10 → 2024-07 only."),
      ...pair("CERES", "→ convection · tiles end 2018-10, and it is monthly NET radiation, not the daily outgoing-longwave field the index needs."),
      ...pair("TAO/TRITON + PIRATA moorings (`gtmba`)", "→ the single most El-Niño-specific record in the catalog · catalogued only for its Atlantic side, and flagged not-renderable."),
      ...pair("Catalogued but unused", "ORAS5 1958→ · SODA3 · ERSSTv5 (the substrate of the official index) · HadISST · EN4 1900→ · ERA5 · CFSR · the SEAS5 and NMME seasonal hindcasts 1981/82→."),
      ...pair("The machine-learning tensors are North-Atlantic-only", "family 7 — the first tensor covering the whole globe at the same 0.25°, 5-day grid (experiment E-070) — is mid-fetch, and its own plan names the equatorial waveguide as a target regime."),
    ], MX + 0.14, 1.56, MW - 0.28, 4.74);
    // ---------------- right: ranked additions ----------------
    const RX = 8.1, RW = 4.6;
    box(s, RX, 1.5, RW, 4.86, NAVY, "F4F6F9");
    const adds = [
      ["Warm Water Volume and 20 °C-isotherm depth", "NOAA PMEL, monthly 1980→. How much warm water is stacked up in the equatorial Pacific — the ~6-month-lead predictor; recharge–discharge theory IS this number."],
      ["GODAS or ORAS5", "NOAA's operational ocean analysis (1/3° × 1°, 40 levels, 5-day and monthly, 1980→, keyless OPeNDAP), or ECMWF's ORAS5 (1958→, 5 members): the subsurface temperature and thermocline the tensor has nowhere in the Pacific."],
      ["NOAA interpolated outgoing longwave radiation", "2.5°, daily, 1974→. Cold cloud tops mark deep convection — the Walker-circulation and Madden–Julian-Oscillation diagnostic CERES does not replace."],
      ["Index series, as labels and as nulls", "the Oceanic Niño Index (the official definition, on ERSSTv5, 1950→), weekly Niño-1+2/3/3.4/4, the Southern Oscillation Index and its equatorial twin, the Multivariate ENSO Index, the RMM Madden–Julian index (the wind-burst trigger), the Pacific Decadal Oscillation and Pacific Meridional Mode (spring precursors)."],
      ["Daily winds", "ERA5 10 m u and v (0.25°, hourly, 1940→) or CCMP 6-hourly 1987→ — westerly wind bursts and wind-stress curl at the cadence at which they actually happen."],
      ["TAO/TRITON moorings as native dots on the Pacific side", "temperature and salinity with depth, currents, winds along 137° E–95° W, 1980→. The array is already catalogued, as `gtmba`."],
      ["DUACS sea level directly", "daily 0.125°, 1993→ — equatorial Kelvin and Rossby waves, and a proxy for warm-water volume."],
      ["Benchmarks", "the IRI/CPC forecast plume (2002→) and the NMME and SEAS5 hindcasts already in the catalog and unused — what any El Niño head must beat, and the spring-barrier skill curve to beat it on."],
      ["In the app", "a `nino.json` bake beside eei.json and gistemp.json (~40 lines next to `oisst_monthly()` in scripts/refresh_data.py), and the four Niño boxes drawn on the globe."],
      ["The cone codec on family 7, with an El Niño head", "sea-surface height, mixed-layer depth, the temperature/salinity column and wind stress are already the classic feature set. What is missing is the rainfall/longwave channel and the label."],
    ];
    txt(s, [
      { text: "Add, ranked", options: { bold: true, color: NAVY, fontSize: 10.5, breakLine: true, paraSpaceAfter: 4 } },
      ...adds.flatMap(([h, b], i) => [
        { text: `${i + 1}  ${h}  `, options: { bold: true, color: TEAL, fontSize: 8.4 } },
        { text: b, options: { fontSize: 8.4, color: INK, breakLine: true, paraSpaceAfter: 3 } },
      ]),
    ], RX + 0.14, 1.56, RW - 0.28, 4.74);
    reading(s, "Reading. Onset is behind us — the event began in March 2026 — so “predict this year's El Niño” now means three different things, and each needs different data. Hindcast the ONSET from the November 2025 state (a −0.6 °C, mildly La-Niña-ish surface with a recharged subsurface): that is a test of whether the subsurface memory was readable. Forecast the PEAK amplitude and month (October–December) and the 2027 exit. And forecast the TELECONNECTIONS — the rainfall, drought and cyclone shifts that are what anyone outside the Pacific actually feels. All three need the subsurface first, the daily winds second, and an honest null beside them: persistence, a linear inverse model, and the operational forecast plume.", 6.48);
    footer(s);
  }
};
